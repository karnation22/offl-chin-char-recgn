#include "caffe2/operators/pool_op.h"
#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

namespace caffe2 {

template <typename T>
class GLAveragePoolOp final : public ConvPoolOpBase<GLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(GLContext);
  GLAveragePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<GLContext>(operator_def, ws) {
  }
  ~GLAveragePoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;
private:
  arm_compute::GCPoolingLayer pooling_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_;
};

template<typename T>
class GLMaxPoolOp final : public ConvPoolOpBase<GLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(GLContext);
  GLMaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<GLContext>(operator_def, ws) {
  }
  ~GLMaxPoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;
private:
  arm_compute::GCPoolingLayer pooling_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_;
};

template <>
bool GLAveragePoolOp<DataType>::RunOnDeviceWithOrderNCHW() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = GLContext::getGLTensor<DataType>(Xblob);
  } else {
    X_ = GLContext::getGLTensor<DataType>(Xblob, X_.release());
  }

  int N = X_->dim32(0);
  int channels = X_->dim32(1);
  int height = X_->dim32(2);
  int width = X_->dim32(3);

  vector<int64_t> output_dims = {N, channels, 1, 1};
  if (!global_pooling_) {
    output_dims[2] = (height + pad_t() + pad_b() - kernel_h()) / stride_h() + 1;
    output_dims[3] = (width + pad_l() + pad_r() - kernel_w()) / stride_w() + 1;
  }

  GLTensor<DataType> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<DataType>>();
  if (first_run_) {
    first_run_ = false;
    CAFFE_ENFORCE_EQ(kernel_.size(), 2, "ARM OpenGL only supports 2D pooling");
    CAFFE_ENFORCE_EQ(kernel_h(), kernel_w(),
                     "ARM OpenGL only supports equal kernel size");
    Y->Resize(output_dims);
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    pooling_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation =Y->Resize(output_dims);
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
    if (need_allocation) {
      Y->allocate();
    }
    pooling_layer_.run();
  }
  return true;
}

template <> bool GLMaxPoolOp<DataType>::RunOnDeviceWithOrderNCHW() {

  auto *Xblob = OperatorBase::Inputs()[0];
  X_ = GLContext::getGLTensor<DataType>(Xblob, X_.release());

  int N = X_->dim32(0);
  int channels = X_->dim32(1);
  int height = X_->dim32(2);
  int width = X_->dim32(3);

  vector<int64_t> output_dims = {N, channels, 1, 1};
  if (!global_pooling_) {
    output_dims[2] = (height + pad_t() + pad_b() - kernel_h()) / stride_h() + 1;
    output_dims[3] = (width + pad_l() + pad_r() - kernel_w()) / stride_w() + 1;
  }
  GLTensor<DataType> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<DataType>>();

  if (first_run_) {
    first_run_ = false;
    CAFFE_ENFORCE_EQ(kernel_.size(), 2, "ARM OpenGL only supports 2D pooling");
    CAFFE_ENFORCE_EQ(kernel_h(), kernel_w(),
                     "ARM OpenGL only supports equal kernel size");
    Y->Resize(output_dims);
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    pooling_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = Y->Resize(output_dims);
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
    if (need_allocation) {
      Y->allocate();
    }
    pooling_layer_.run();
  }

  return true;
}

template <>
bool GLAveragePoolOp<DataType>::RunOnDeviceWithOrderNHWC() {
  return false;
}

template <>
bool GLMaxPoolOp<DataType>::RunOnDeviceWithOrderNHWC() {
  return false;
}

REGISTER_GL_OPERATOR(AveragePool, GLAveragePoolOp<DataType>);
REGISTER_GL_OPERATOR(MaxPool, GLMaxPoolOp<DataType>);

} // namespace caffe2
