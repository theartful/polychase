#include "gftt.h"

#include <opencv2/imgproc.hpp>

#include "utils.h"

struct greaterThanPtr {
    bool operator()(const float* a, const float* b) const {
        // Ensure a fully deterministic result of the sort
        return (*a > *b) ? true : (*a < *b) ? false : (a > b);
    }
};

void GoodFeaturesToTrack(cv::InputArray _image, cv::InputArray _mask,
                         cv::OutputArray _corners,
                         cv::OutputArray _corners_quality,
                         const GFTTOptions& options) {
    CHECK(options.quality_level > 0 && options.min_distance >= 0 &&
          options.max_corners >= 0);
    CHECK(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)));

    const cv::Mat image = _image.getMat();
    if (image.empty()) {
        _corners.release();
        _corners_quality.release();
        return;
    }

    cv::Mat eig;

    if (options.use_harris)
        cornerHarris(image, eig, options.block_size, options.gradient_size,
                     options.harris_k);
    else
        cornerMinEigenVal(image, eig, options.block_size,
                          options.gradient_size);

    // Apply grid-based thresholding for better feature distribution
    const int grid_rows = std::max(1, options.grid_rows);
    const int grid_cols = std::max(1, options.grid_cols);

    const int block_height = (image.rows + grid_rows - 1) / grid_rows;
    const int block_width = (image.cols + grid_cols - 1) / grid_cols;

    const cv::Mat mask = _mask.getMat();

    for (int grid_y = 0; grid_y < grid_rows; grid_y++) {
        for (int grid_x = 0; grid_x < grid_cols; grid_x++) {
            const int y_start = grid_y * block_height;
            const int x_start = grid_x * block_width;
            const int y_end = std::min(y_start + block_height, image.rows);
            const int x_end = std::min(x_start + block_width, image.cols);

            const cv::Rect block_rect(x_start, y_start, x_end - x_start,
                                      y_end - y_start);

            const cv::Mat eig_block = eig(block_rect);
            const cv::Mat mask_block =
                mask.empty() ? cv::Mat() : mask(block_rect);

            double maxVal = 0;
            cv::minMaxLoc(eig_block, nullptr, &maxVal, nullptr, nullptr,
                          mask_block);
            cv::threshold(eig_block, eig_block, maxVal * options.quality_level,
                          0, cv::THRESH_TOZERO);
        }
    }

    cv::Mat tmp;
    cv::dilate(eig, tmp, cv::Mat());

    const cv::Size imgsize = image.size();
    std::vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    for (int y = 1; y < imgsize.height - 1; y++) {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for (int x = 1; x < imgsize.width - 1; x++) {
            const float val = eig_data[x];
            if (val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]))
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<cv::Point2f> corners;
    std::vector<float> cornersQuality;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0) {
        _corners.release();
        _corners_quality.release();
        return;
    }

    std::sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

    if (options.min_distance >= 1) {
        // Partition the image into larger grids
        const int w = image.cols;
        const int h = image.rows;

        const int cell_size = cvRound(options.min_distance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width * grid_height);

        const double min_distance_sq =
            options.min_distance * options.min_distance;

        for (i = 0; i < total; i++) {
            const int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            const int y = (int)(ofs / eig.step);
            const int x = (int)((ofs - y * eig.step) / sizeof(float));

            const int x_cell = x / cell_size;
            const int y_cell = y / cell_size;

            const int x1 = std::max(x_cell - 1, 0);
            const int y1 = std::max(y_cell - 1, 0);
            const int x2 = std::min(x_cell + 1, grid_width - 1);
            const int y2 = std::min(y_cell + 1, grid_height - 1);

            bool good = true;

            for (int yy = y1; yy <= y2; yy++) {
                for (int xx = x1; xx <= x2; xx++) {
                    std::vector<cv::Point2f>& m = grid[yy * grid_width + xx];

                    if (m.size()) {
                        for (j = 0; j < m.size(); j++) {
                            const float dx = x - m[j].x;
                            const float dy = y - m[j].y;

                            if (dx * dx + dy * dy < min_distance_sq) {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

        break_out:

            if (good) {
                grid[y_cell * grid_width + x_cell].push_back(
                    cv::Point2f((float)x, (float)y));

                if (_corners_quality.needed()) {
                    cornersQuality.push_back(*tmpCorners[i]);
                }

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if (options.max_corners > 0 &&
                    (int)ncorners == options.max_corners)
                    break;
            }
        }
    } else {
        for (i = 0; i < total; i++) {
            if (_corners_quality.needed()) {
                cornersQuality.push_back(*tmpCorners[i]);
            }

            const int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            const int y = (int)(ofs / eig.step);
            const int x = (int)((ofs - y * eig.step) / sizeof(float));

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;

            if (options.max_corners > 0 && (int)ncorners == options.max_corners)
                break;
        }
    }

    cv::Mat(corners).convertTo(_corners,
                               _corners.fixedType() ? _corners.type() : CV_32F);

    if (_corners_quality.needed()) {
        cv::Mat(cornersQuality)
            .convertTo(_corners_quality, _corners_quality.fixedType()
                                             ? _corners_quality.type()
                                             : CV_32F);
    }
}
