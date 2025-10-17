#include "gftt.h"

#include <opencv2/imgproc.hpp>

#include "utils.h"

struct greaterThanPtr {
    bool operator()(const float* a, const float* b) const
    // Ensure a fully deterministic result of the sort
    {
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

    cv::Mat image = _image.getMat(), eig, tmp;
    if (image.empty()) {
        _corners.release();
        _corners_quality.release();
        return;
    }

    if (options.use_harris)
        cornerHarris(image, eig, options.block_size, options.gradient_size,
                     options.harris_k);
    else
        cornerMinEigenVal(image, eig, options.block_size,
                          options.gradient_size);

    double maxVal = 0;
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);
    cv::threshold(eig, eig, maxVal * options.quality_level, 0,
                  cv::THRESH_TOZERO);
    cv::dilate(eig, tmp, cv::Mat());

    cv::Size imgsize = image.size();
    std::vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    cv::Mat mask = _mask.getMat();
    for (int y = 1; y < imgsize.height - 1; y++) {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for (int x = 1; x < imgsize.width - 1; x++) {
            float val = eig_data[x];
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

    double min_distance = options.min_distance;

    if (min_distance >= 1) {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(min_distance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width * grid_height);

        min_distance *= min_distance;

        for (i = 0; i < total; i++) {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for (int yy = y1; yy <= y2; yy++) {
                for (int xx = x1; xx <= x2; xx++) {
                    std::vector<cv::Point2f>& m = grid[yy * grid_width + xx];

                    if (m.size()) {
                        for (j = 0; j < m.size(); j++) {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if (dx * dx + dy * dy < min_distance) {
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

                cornersQuality.push_back(*tmpCorners[i]);

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if (options.max_corners > 0 &&
                    (int)ncorners == options.max_corners)
                    break;
            }
        }
    } else {
        for (i = 0; i < total; i++) {
            cornersQuality.push_back(*tmpCorners[i]);

            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

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
