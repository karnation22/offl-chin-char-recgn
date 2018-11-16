#include "caffe2/operators/experimental/c10/schemas/mul.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::Mul);

namespace {

struct LegacyBroadcastParameter final {
  using type = bool;
  static constexpr const char* name() {
    return "legacy_broadcast";
  }
  static constexpr bool default_value() {
    return true;
  }
};
struct AxisParameter final {
  using type = int;
  static constexpr const char* name() {
    return "axis";
  }
  static constexpr int default_value() {
    return -1;
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::Mul,
    void,
    C10Mul_DontUseThisOpYet,
    ParameterHelper<LegacyBroadcastParameter>,
    ParameterHelper<AxisParameter>)
}
