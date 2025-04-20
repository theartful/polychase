#pragma once

#include <Eigen/Core>

#include "eigen_typedefs.h"
#include "pnp/solvers.h"

void PnPOpenGLPreprocessing(RefRowMajorMatrixX3f object_points, PnPResult& result);
void PnPOpenGLPostprocessing(RefRowMajorMatrixX3f object_points, PnPResult& result);

class OpenGLPnPAdapter {
   public:
    OpenGLPnPAdapter(const ConstRefRowMajorMatrixX3f& object_points, PnPResult& result)
        : object_points_(const_cast<float*>(object_points.data()), object_points.rows(), object_points.cols()),
          result_(result),
          is_opengl(result.camera.intrinsics.convention == CameraConvention::OpenGL) {
        if (is_opengl) {
            PnPOpenGLPreprocessing(object_points_, result_);
        }
    }

    ~OpenGLPnPAdapter() { PnPOpenGLPostprocessing(object_points_, result_); }

   private:
    Eigen::Map<RowMajorMatrixX3f> object_points_;
    PnPResult& result_;
    bool is_opengl;
};
