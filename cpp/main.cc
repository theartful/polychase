#include <Eigen/Core>
#include <iostream>

#include "database.h"

int main() {
    //
    auto database = Database("/home/theartful/test.db");

    KeypointsMatrix keypoints1 = KeypointsMatrix::Random(3, 2);
    database.WriteKeypoints(1, keypoints1);

    KeypointsMatrix keypoints2 = KeypointsMatrix::Random(3, 2);
    database.WriteKeypoints(2, keypoints2);

    KeypointsMatrix keypoints3 = KeypointsMatrix::Random(3, 2);
    database.WriteKeypoints(3, keypoints2);

    KeypointsIndicesMatrix src_kps_indices = KeypointsIndicesMatrix::Random(3, 1);
    KeypointsMatrix tgt_kps = KeypointsMatrix::Random(3, 2);
    FlowErrorsMatrix flow_errors = FlowErrorsMatrix::Random(3, 1);
    database.WriteImagePairFlow(1, 2, src_kps_indices, tgt_kps, flow_errors);

    {
        KeypointsIndicesMatrix src_kps_indices = KeypointsIndicesMatrix::Random(3, 1);
        KeypointsMatrix tgt_kps = KeypointsMatrix::Random(3, 2);
        FlowErrorsMatrix flow_errors = FlowErrorsMatrix::Random(3, 1);
        database.WriteImagePairFlow(1, 3, src_kps_indices, tgt_kps, flow_errors);
    }

    ImagePairFlow flow = database.ReadImagePairFlow(1, 2);

    std::cout << "W keypoints1: " << keypoints1 << '\n';
    std::cout << "R keypoints1: " << database.ReadKeypoints(1) << '\n';

    std::cout << "W keypoints2: " << keypoints2 << '\n';
    std::cout << "R keypoints2: " << database.ReadKeypoints(2) << '\n';

    std::cout << "W keypoints3: " << keypoints3 << '\n';
    std::cout << "R keypoints3: " << database.ReadKeypoints(3) << '\n';

    std::cout << "W src_kps_indices: " << src_kps_indices << '\n';
    std::cout << "R src_kps_indices: " << flow.src_kps_indices << '\n';

    std::cout << "W tgt_kps: " << tgt_kps << '\n';
    std::cout << "R tgt_kps: " << flow.tgt_kps << '\n';

    std::cout << "W flow_errors: " << flow_errors << '\n';
    std::cout << "R flow_errors: " << flow.flow_errors << '\n';

    {
        std::vector<uint32_t> image_id_from_1 = database.FindOpticalFlowsFromImage(1);
        std::cout << "len(image_id_from_1) = " << image_id_from_1.size() << '\n';
        for (auto id : image_id_from_1) {
            std::cout << id << '\n';
        }
    }

    {
        std::vector<uint32_t> image_id_to_1 = database.FindOpticalFlowsToImage(1);
        std::cout << "len(image_id_to_1) = " << image_id_to_1.size() << '\n';
        for (auto id : image_id_to_1) {
            std::cout << id << '\n';
        }
    }

    {
        std::vector<uint32_t> image_id_from_2 = database.FindOpticalFlowsFromImage(2);
        std::cout << "len(image_id_from_2) = " << image_id_from_2.size() << '\n';
        for (auto id : image_id_from_2) {
            std::cout << id << '\n';
        }
    }

    {
        std::vector<uint32_t> image_id_to_2 = database.FindOpticalFlowsToImage(2);
        std::cout << "len(image_id_to_2) = " << image_id_to_2.size() << '\n';
        for (auto id : image_id_to_2) {
            std::cout << id << '\n';
        }
    }

    {
        std::vector<uint32_t> image_id_to_3 = database.FindOpticalFlowsToImage(3);
        std::cout << "len(image_id_to_3) = " << image_id_to_3.size() << '\n';
        for (auto id : image_id_to_3) {
            std::cout << id << '\n';
        }
    }
}
