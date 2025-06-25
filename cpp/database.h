// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <sqlite3.h>

#include <Eigen/Core>
#include <limits>
#include <string>
#include <vector>

static constexpr int32_t kInvalidId = std::numeric_limits<int32_t>::max();

using Keypoints = std::vector<Eigen::Vector2f>;
using KeypointsIndices = std::vector<uint32_t>;
using FlowErrors = std::vector<float>;
using KeypointsQualitiess = std::vector<float>;

// Mostly following colmap's Database class style of implementation

struct ImagePairFlow {
    int32_t image_id_from;
    int32_t image_id_to;
    KeypointsIndices src_kps_indices;
    Keypoints tgt_kps;
    FlowErrors flow_errors;

    void Clear() {
        src_kps_indices.clear();
        tgt_kps.clear();
        flow_errors.clear();
    }
};

class Database {
   public:
    explicit Database(const std::string& path);
    Database(Database&& other);
    Database(const Database& other) = delete;

    void Open(const std::string& path);
    void Close();

    Keypoints ReadKeypoints(int32_t image_id) const;

    void ReadKeypoints(int32_t image_id, Keypoints& keypoints) const;

    void WriteKeypoints(int32_t image_id, const Keypoints& keypoints);

    ImagePairFlow ReadImagePairFlow(int32_t image_id_from,
                                    int32_t image_id_to) const;

    void ReadImagePairFlow(int32_t image_id_from, int32_t image_id_to,
                           ImagePairFlow& image_pair_flow) const;

    void WriteImagePairFlow(int32_t image_id_from, int32_t image_id_to,
                            const KeypointsIndices& src_kps_indices,
                            const Keypoints& tgt_kps,
                            const FlowErrors& flow_errors);

    void WriteImagePairFlow(const ImagePairFlow& image_pair_flow);

    std::vector<int32_t> FindOpticalFlowsFromImage(int32_t image_id_from) const;

    void FindOpticalFlowsFromImage(int32_t image_id_from,
                                   std::vector<int32_t>& result) const;

    std::vector<int32_t> FindOpticalFlowsToImage(int32_t image_id_to) const;

    void FindOpticalFlowsToImage(int32_t image_id_to,
                                 std::vector<int32_t>& result) const;

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
