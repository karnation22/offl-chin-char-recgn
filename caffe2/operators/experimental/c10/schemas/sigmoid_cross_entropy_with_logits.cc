#include "caffe2/operators/experimental/c10/schemas/sigmoid_cross_entropy_with_logits.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::SigmoidCrossEntropyWithLogits);

namespace {
struct LogDTrickParameter final {
  using type = bool;
  static constexpr const char* name() {
    return "log_D_trick";
  }
  static constexpr bool default_value() {
    return false;
  }
};
struct UnjoinedLRLossParameter final {
  using type = bool;
  static constexpr const char* name() {
    return "unjoined_lr_loss";
  }
  static constexpr bool default_value() {
    return false;
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::SigmoidCrossEntropyWithLogits,
    void,
    C10SigmoidCrossEntropyWithLogits_DontUseThisOpYet,
    ParameterHelper<LogDTrickParameter>,
    ParameterHelper<UnjoinedLRLossParameter>)
}
