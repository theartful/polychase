#pragma once

// Doesn't feel right to include pnp.h, while pnp.cc includes iterative_solver.h
#include "eigen_typedefs.h"
#include "pnp.h"

bool SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       PnPResult& result);

bool SolvePnPRansac(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                    PnPResult& result);
