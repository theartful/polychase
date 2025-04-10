#include "database.h"

#include <fmt/format.h>

#include "utils.h"

#define SQLITE3_EXEC(database, sql, callback)                                                               \
    {                                                                                                       \
        char* err_msg = nullptr;                                                                            \
        const int result_code = sqlite3_exec(database, sql, callback, nullptr, &err_msg);                   \
        if (result_code != SQLITE_OK) {                                                                     \
            sqlite3_free(err_msg);                                                                          \
            throw std::runtime_error(fmt::format("SQLite error [{}:{}]: {}", __FILE__, __LINE__, err_msg)); \
        }                                                                                                   \
    }

inline int SQLite3CallHelper(int result_code, const char* filename, int line) {
    switch (result_code) {
        case SQLITE_OK:
        case SQLITE_ROW:
        case SQLITE_DONE:
            return result_code;
        default:
            throw std::runtime_error(
                fmt::format("SQLite error [{}:{}]: {}", filename, line, sqlite3_errstr(result_code)));
    }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

Database::Database(const std::string& path) { Open(path); }

void Database::Open(const std::string& path) {
    Close();

    // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
    // mutex (so that we don't serialize the connection's operations).
    // Modifications to the database will still be serialized, but multiple
    // connections can read concurrently.
    SQLITE3_CALL(sqlite3_open_v2(path.c_str(), &database_,
                                 SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nullptr));

    // Don't wait for the operating system to write the changes to disk
    SQLITE3_EXEC(database_, "PRAGMA synchronous=OFF", nullptr);

    // Use faster journaling mode
    SQLITE3_EXEC(database_, "PRAGMA journal_mode=WAL", nullptr);

    // Store temporary tables and indices in memory
    SQLITE3_EXEC(database_, "PRAGMA temp_store=MEMORY", nullptr);

    // Disabled by default
    SQLITE3_EXEC(database_, "PRAGMA foreign_keys=ON", nullptr);

    // Enable auto vacuum to reduce DB file size
    SQLITE3_EXEC(database_, "PRAGMA auto_vacuum=1", nullptr);

    CreateTables();
    PrepareSQLStatements();
}

void Database::Close() {
    if (database_ != nullptr) {
        FinalizeSQLStatements();
        sqlite3_close_v2(database_);
        database_ = nullptr;
    }
}

void Database::CreateTables() const {
    CreateKeypointsTable();
    CreateOpticalFlowTable();
}

void Database::CreateKeypointsTable() const {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS keypoints(
            image_id   INTEGER  PRIMARY KEY  NOT NULL,
            rows       INTEGER               NOT NULL,
            keypoints  BLOB                  NOT NULL
        );
    )";

    SQLITE3_EXEC(database_, sql, nullptr);
}

void Database::CreateOpticalFlowTable() const {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS optical_flow(
            image_id_from           INTEGER  NOT NULL,
            image_id_to             INTEGER  NOT NULL,
            rows                    INTEGER  NOT NULL,
            src_keypoints_indices   BLOB     NOT NULL,
            tgt_keypoints           BLOB     NOT NULL,
            flow_errors             BLOB     NOT NULL,
            PRIMARY KEY(image_id_from, image_id_to),
            FOREIGN KEY(image_id_from) REFERENCES keypoints(image_id) ON DELETE CASCADE
        );
    )";

    SQLITE3_EXEC(database_, sql, nullptr);
}

template <typename MatrixType>
MatrixType ReadMatrixBlob(sqlite3_stmt* sql_stmt, uint32_t num_rows, uint32_t num_cols, uint32_t col) {
    CHECK(MatrixType::ColsAtCompileTime == Eigen::Dynamic ||
          MatrixType::ColsAtCompileTime == static_cast<Eigen::Index>(num_cols));
    CHECK(MatrixType::RowsAtCompileTime == Eigen::Dynamic ||
          MatrixType::RowsAtCompileTime == static_cast<Eigen::Index>(num_rows));

    MatrixType matrix{static_cast<Eigen::Index>(num_rows), static_cast<Eigen::Index>(num_cols)};

    const size_t num_bytes = static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
    CHECK_EQ(matrix.size() * sizeof(typename MatrixType::Scalar), num_bytes);

    memcpy(reinterpret_cast<char*>(matrix.data()), sqlite3_column_blob(sql_stmt, col), num_bytes);

    return matrix;
}

template <typename MatrixType>
void WriteMatrixBlob(sqlite3_stmt* sql_stmt, const MatrixType& matrix, uint32_t col) {
    const size_t num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);
    SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, col, reinterpret_cast<const char*>(matrix.data()),
                                   static_cast<int>(num_bytes), SQLITE_STATIC));
}

KeypointsMatrix Database::ReadKeypoints(uint32_t image_id) const {
    sqlite3_stmt* sql_stmt = sql_stmt_read_keypoints_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, image_id));
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc != SQLITE_ROW) {
        SQLITE3_CALL(sqlite3_reset(sql_stmt));
        return {};
    }
    const size_t rows = static_cast<size_t>(sqlite3_column_int(sql_stmt, 0));
    KeypointsMatrix keypoints = ReadMatrixBlob<KeypointsMatrix>(sql_stmt, rows, KeypointsMatrix::ColsAtCompileTime, 1);

    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return keypoints;
}

void Database::WriteKeypoints(uint32_t image_id, const Eigen::Ref<const KeypointsMatrix>& keypoints) {
    sqlite3_stmt* sql_stmt = sql_stmt_write_keypoints_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, static_cast<int>(image_id)));
    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 2, keypoints.rows()));
    WriteMatrixBlob(sql_stmt, keypoints, 3);

    SQLITE3_CALL(sqlite3_step(sql_stmt));
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
}

void Database::WriteImagePairFlow(uint32_t image_id_from, uint32_t image_id_to,
                                  const Eigen::Ref<const KeypointsIndicesMatrix>& src_kps_indices,
                                  const Eigen::Ref<const KeypointsMatrix>& tgt_kps,
                                  const Eigen::Ref<const FlowErrorsMatrix>& flow_errors) {
    sqlite3_stmt* sql_stmt = sql_stmt_write_image_pair_flows_;

    const Eigen::Index rows = src_kps_indices.rows();

    CHECK_EQ(tgt_kps.rows(), rows);
    CHECK_EQ(flow_errors.rows(), rows);

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, static_cast<int>(image_id_from)));
    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 2, static_cast<int>(image_id_to)));
    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 3, static_cast<int>(rows)));
    WriteMatrixBlob(sql_stmt, src_kps_indices, 4);
    WriteMatrixBlob(sql_stmt, tgt_kps, 5);
    WriteMatrixBlob(sql_stmt, flow_errors, 6);

    SQLITE3_CALL(sqlite3_step(sql_stmt));
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
}

void Database::WriteImagePairFlow(const ImagePairFlow& image_pair_flow) {
    WriteImagePairFlow(image_pair_flow.image_id_from, image_pair_flow.image_id_to, image_pair_flow.src_kps_indices,
                       image_pair_flow.tgt_kps, image_pair_flow.flow_errors);
}

ImagePairFlow Database::ReadImagePairFlow(uint32_t image_id_from, uint32_t image_id_to) const {
    sqlite3_stmt* sql_stmt = sql_stmt_read_image_pair_flows_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, image_id_from));
    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 2, image_id_to));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc != SQLITE_ROW) {
        SQLITE3_CALL(sqlite3_reset(sql_stmt));
        return {};
    }
    const size_t rows = static_cast<size_t>(sqlite3_column_int(sql_stmt, 0));
    KeypointsIndicesMatrix src_kps_indices =
        ReadMatrixBlob<KeypointsIndicesMatrix>(sql_stmt, rows, KeypointsIndicesMatrix::ColsAtCompileTime, 1);
    KeypointsMatrix tgt_kps = ReadMatrixBlob<KeypointsMatrix>(sql_stmt, rows, KeypointsMatrix::ColsAtCompileTime, 2);
    FlowErrorsMatrix flow_errors =
        ReadMatrixBlob<FlowErrorsMatrix>(sql_stmt, rows, FlowErrorsMatrix::ColsAtCompileTime, 3);

    SQLITE3_CALL(sqlite3_reset(sql_stmt));

    return ImagePairFlow{
        .image_id_from = image_id_from,
        .image_id_to = image_id_to,
        .src_kps_indices = std::move(src_kps_indices),
        .tgt_kps = std::move(tgt_kps),
        .flow_errors = std::move(flow_errors),
    };
}

std::vector<uint32_t> Database::FindOpticalFlowsFromImage(uint32_t image_id_from) const {
    sqlite3_stmt* sql_stmt = sql_stmt_find_flows_from_image_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, image_id_from));

    std::vector<uint32_t> result;
    while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
        const uint32_t image_id_to = sqlite3_column_int(sql_stmt, 0);
        result.push_back(image_id_to);
    }

    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return result;
}

std::vector<uint32_t> Database::FindOpticalFlowsToImage(uint32_t image_id_to) const {
    sqlite3_stmt* sql_stmt = sql_stmt_find_flows_to_image_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, image_id_to));

    std::vector<uint32_t> result;
    while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
        const uint32_t image_id_from = sqlite3_column_int(sql_stmt, 0);
        result.push_back(image_id_from);
    }

    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return result;
}

bool Database::KeypointsExist(uint32_t image_id) const {
    sqlite3_stmt* sql_stmt = sql_stmt_keypoints_exist_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, image_id));
    const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return exists;
}

bool Database::ImagePairFlowExists(uint32_t image_id_from, uint32_t image_id_to) const {
    sqlite3_stmt* sql_stmt = sql_stmt_pair_flow_exist_;

    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 1, image_id_from));
    SQLITE3_CALL(sqlite3_bind_int(sql_stmt, 2, image_id_to));
    const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return exists;
}

uint32_t Database::GetMinImageIdWithKeypoints() const {
    sqlite3_stmt* sql_stmt = sql_stmt_min_image_id;

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc != SQLITE_ROW) {
        SQLITE3_CALL(sqlite3_reset(sql_stmt));
        return INVALID_ID;
    }

    const uint32_t image_id = static_cast<uint32_t>(sqlite3_column_int(sql_stmt, 0));
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return image_id;
}

uint32_t Database::GetMaxImageIdWithKeypoints() const {
    sqlite3_stmt* sql_stmt = sql_stmt_max_image_id;

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc != SQLITE_ROW) {
        SQLITE3_CALL(sqlite3_reset(sql_stmt));
        return INVALID_ID;
    }

    const uint32_t image_id = static_cast<uint32_t>(sqlite3_column_int(sql_stmt, 0));
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
    return image_id;
}

void Database::PrepareSQLStatements() {
    const char* sql = nullptr;

    sql = "SELECT rows, keypoints FROM keypoints WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_read_keypoints_, 0));

    sql = "INSERT INTO keypoints(image_id, rows, keypoints) VALUES(?, ?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_write_keypoints_, 0));

    sql =
        "SELECT rows, src_keypoints_indices, tgt_keypoints, flow_errors FROM optical_flow WHERE image_id_from = ? AND "
        "image_id_to = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_read_image_pair_flows_, 0));

    sql =
        "INSERT INTO optical_flow(image_id_from, image_id_to, rows, src_keypoints_indices, tgt_keypoints, "
        "flow_errors) VALUES(?, ?, ?, ?, ?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_write_image_pair_flows_, 0));

    sql = "SELECT image_id_to FROM optical_flow WHERE image_id_from = ?";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_find_flows_from_image_, 0));

    sql = "SELECT image_id_from FROM optical_flow WHERE image_id_to = ?";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_find_flows_to_image_, 0));

    sql = "SELECT 1 FROM keypoints WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_keypoints_exist_, 0));

    sql = "SELECT 1 FROM optical_flow WHERE image_id_from = ? AND image_id_to = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_pair_flow_exist_, 0));

    sql = "SELECT MIN(image_id) FROM keypoints;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_min_image_id, 0));

    sql = "SELECT MAX(image_id) FROM keypoints;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql, -1, &sql_stmt_max_image_id, 0));
}

void Database::FinalizeSQLStatements() {
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_read_keypoints_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_write_keypoints_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_read_image_pair_flows_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_write_image_pair_flows_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_find_flows_from_image_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_find_flows_to_image_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_keypoints_exist_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_pair_flow_exist_));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_min_image_id));
    SQLITE3_CALL(sqlite3_finalize(sql_stmt_max_image_id));
}
