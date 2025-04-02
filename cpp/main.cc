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

    KeypointsIndicesMatrix src_kps_indices = KeypointsIndicesMatrix::Random(3, 1);
    KeypointsMatrix tgt_kps = KeypointsMatrix::Random(3, 2);
    FlowErrorsMatrix flow_errors = FlowErrorsMatrix::Random(3, 1);
    database.WriteImagePairFlow(1, 2, src_kps_indices, tgt_kps, flow_errors);

    ImagePairFlow flow = database.ReadImagePairFlow(1, 2);

    std::cout << "W keypoints1: " << keypoints1 << '\n';
    std::cout << "R keypoints1: " << database.ReadKeypoints(1) << '\n';

    std::cout << "W keypoints2: " << keypoints2 << '\n';
    std::cout << "R keypoints2: " << database.ReadKeypoints(2) << '\n';

    std::cout << "W src_kps_indices: " << src_kps_indices << '\n';
    std::cout << "R src_kps_indices: " << flow.src_kps_indices << '\n';

    std::cout << "W tgt_kps: " << tgt_kps << '\n';
    std::cout << "R tgt_kps: " << flow.tgt_kps << '\n';

    std::cout << "W flow_errors: " << flow_errors << '\n';
    std::cout << "R flow_errors: " << flow.flow_errors << '\n';
}
