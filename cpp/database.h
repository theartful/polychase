#pragma once

#include <sqlite3.h>

#include <Eigen/Core>
#include <limits>
#include <string>

#include "eigen_typedefs.h"

using KeypointsMatrix = RowMajorArrayX2f;
using KeypointsIndicesMatrix = ArrayXu;
using FlowErrorsMatrix = Eigen::ArrayXf;
using KeypointsQualitiessMatrix = Eigen::ArrayXf;

constexpr int32_t INVALID_ID = std::numeric_limits<int32_t>::max();

// Mostly following colmap's Database class style of implementation

struct ImagePairFlow {
    int32_t image_id_from;
    int32_t image_id_to;
    KeypointsIndicesMatrix src_kps_indices;
    KeypointsMatrix tgt_kps;
    FlowErrorsMatrix flow_errors;
};

class Database {
   public:
    explicit Database(const std::string& path);
    void Open(const std::string& path);
    void Close();

    KeypointsMatrix ReadKeypoints(int32_t image_id) const;

    void WriteKeypoints(int32_t image_id, const Eigen::Ref<const KeypointsMatrix>& keypoints);

    ImagePairFlow ReadImagePairFlow(int32_t image_id_from, int32_t image_id_to) const;

    void WriteImagePairFlow(int32_t image_id_from, int32_t image_id_to,
                            const Eigen::Ref<const KeypointsIndicesMatrix>& src_kps_indices,
                            const Eigen::Ref<const KeypointsMatrix>& tgt_kps,
                            const Eigen::Ref<const FlowErrorsMatrix>& flow_errors);

    void WriteImagePairFlow(const ImagePairFlow& image_pair_flow);

    std::vector<int32_t> FindOpticalFlowsFromImage(int32_t image_id_from) const;

    void FindOpticalFlowsFromImage(int32_t image_id_from, std::vector<int32_t>& result) const;

    std::vector<int32_t> FindOpticalFlowsToImage(int32_t image_id_to) const;

    void FindOpticalFlowsToImage(int32_t image_id_to, std::vector<int32_t>& result) const;

    bool KeypointsExist(int32_t image_id) const;

    bool ImagePairFlowExists(int32_t image_id_from, int32_t image_id_to) const;

    int32_t GetMinImageIdWithKeypoints() const;

    int32_t GetMaxImageIdWithKeypoints() const;

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
    sqlite3_stmt* sql_stmt_find_flows_from_image_ = nullptr;
    sqlite3_stmt* sql_stmt_find_flows_to_image_ = nullptr;
    sqlite3_stmt* sql_stmt_keypoints_exist_ = nullptr;
    sqlite3_stmt* sql_stmt_pair_flow_exist_ = nullptr;
    sqlite3_stmt* sql_stmt_min_image_id = nullptr;
    sqlite3_stmt* sql_stmt_max_image_id = nullptr;
};
