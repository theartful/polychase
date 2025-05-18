#include "refiner.h"

void RefineTrajectory(const Database& database, CameraTrajectory& traj) {
    //
}

void RefineTrajectory(const std::string& database_path, CameraTrajectory& traj) {
    Database database{database_path};
    RefineTrajectory(database, traj);
}
