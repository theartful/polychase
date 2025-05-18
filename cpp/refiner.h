#pragma once

#include <string>

#include "camera_trajectory.h"
#include "database.h"

void RefineTrajectory(const Database& database, CameraTrajectory& traj);
void RefineTrajectory(const std::string& database_path, CameraTrajectory& traj);
