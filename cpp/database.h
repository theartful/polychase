#pragma once

#include <sqlite3.h>

#include <Eigen/Core>
#include <string>

#include "eigen_typedefs.h"

using KeypointsMatrix = RowMajorArrayX2f;
using KeypointsIndicesMatrix = ArrayXu;
using FlowErrorsMatrix = Eigen::ArrayXf;

// Mostly following colmap's Database class style of implementation

struct ImagePairFlow {
    uint32_t image_id_from;
    uint32_t image_id_to;
    KeypointsIndicesMatrix src_kps_indices;
    KeypointsMatrix tgt_kps;
    FlowErrorsMatrix flow_errors;
};

class Database {
   public:
    explicit Database(const std::string& path);
    void Open(const std::string& path);
    void Close();

    KeypointsMatrix ReadKeypoints(uint32_t image_id) const;

    void WriteKeypoints(uint32_t image_id, const Eigen::Ref<KeypointsMatrix>& keypoints);

    void WriteImagePairFlow(uint32_t image_id_from, uint32_t image_id_to,
                            const Eigen::Ref<const KeypointsIndicesMatrix>& src_kps_indices,
                            const Eigen::Ref<const KeypointsMatrix>& tgt_kps,
                            const Eigen::Ref<const FlowErrorsMatrix>& flow_errors);

    void WriteImagePairFlow(const ImagePairFlow& image_pair_flow);

    ImagePairFlow ReadImagePairFlow(uint32_t image_id_from, uint32_t image_id_to);

   private:
    void CreateTables() const;
    void CreateKeypointsTable() const;
    void CreateOpticalFlowTable() const;
    void PrepareSQLStatements();
    void FinalizeSQLStatements();

    sqlite3* database_ = nullptr;
    sqlite3_stmt* sql_stmt_read_keypoints_ = nullptr;
    sqlite3_stmt* sql_stmt_write_keypoints_ = nullptr;
    sqlite3_stmt* sql_stmt_read_image_pair_flows_ = nullptr;
    sqlite3_stmt* sql_stmt_write_image_pair_flows_ = nullptr;
};
