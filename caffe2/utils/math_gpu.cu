// Implements the math functions for GPU.

#include "caffe2/utils/math.h"

#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/conversions.h"

#include "caffe2/utils/fixed_divisor.h"
// TODO: Move this to fixed_divisor.h
#ifdef __HIPCC__
#define FIXED_DIVISOR int32_t
#define FIXED_DIVISOR_DIV(d, n) (n / d)
#define FIXED_DIVISOR_MOD(d, n) (n % d)
#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)
#else // __HIPCC__
#define FIXED_DIVISOR FixedDivisor<int32_t>
#define FIXED_DIVISOR_DIV(d, n) (d.Div(n))
#define FIXED_DIVISOR_MOD(d, n) (d.Mod(n))
#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) (d.DivMod(n, q, r))
#endif // __HIPCC__

#include "caffe2/utils/math_utils.h"

#if THRUST_VERSION >= 100800
#define THRUST_SUPPORTS_PER_THREAD
#endif // THRUST_VERSION >= 100800

namespace caffe2 {
namespace math {

namespace {

#define DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Func, expr)          \
  template <typename T>                                                 \
  struct Func##Functor {                                                \
    inline __host__ __device__ T                                        \
    operator()(const T& lhs, const T& rhs) const {                      \
      return lhs expr rhs;                                              \
    }                                                                   \
  };                                                                    \
  template <>                                                           \
  struct Func##Functor<at::Half> {                                      \
    inline __host__ __device__ at::Half operator()(                     \
        const at::Half& lhs,                                            \
        const at::Half& rhs) const {                                    \
      return convert::To<float, at::Half>(convert::To<at::Half, float>( \
          lhs) expr convert::To<at::Half, float>(rhs));                 \
    }                                                                   \
  };
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Add, +)
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Sub, -)
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Mul, *)
DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR(Div, /)
#undef DELEGATE_SIMPLE_HOST_DEVICE_BINARY_FUNCTOR

template <typename T>
__global__ void SinCosCUDAKernel(const int N, const T* X, T* S, T* C) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    c10::cuda::compat::sincos(__ldg(X + i), S + i, C + i);
#else
    c10::cuda::compat::sincos(X[i], S + i, C + i);
#endif
  }
}

template <typename TIn, typename TOut, class BinaryOperator>
__global__ void SimpleBinaryOpCUDAKernel(
    const int N,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    C[i] = op(A[i], B[i]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator, bool broadcast_1st>
__global__ void RowwiseBinaryOpCUDAKenel(
    const int size,
    const FIXED_DIVISOR cols,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  CUDA_1D_KERNEL_LOOP(C_index, size) {
    const int j = FIXED_DIVISOR_MOD(cols, C_index);
    const int A_index = broadcast_1st ? j : C_index;
    const int B_index = broadcast_1st ? C_index : j;
    C[C_index] = op(A[A_index], B[B_index]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator, bool broadcast_1st>
__global__ void ColwiseBinaryOpCUDAKenel(
    const int size,
    const FIXED_DIVISOR cols,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  CUDA_1D_KERNEL_LOOP(C_index, size) {
    const int i = FIXED_DIVISOR_DIV(cols, C_index);
    const int A_index = broadcast_1st ? i : C_index;
    const int B_index = broadcast_1st ? C_index : i;
    C[C_index] = op(A[A_index], B[B_index]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator, int D>
__global__ void BroadcastBinaryOpCUDAKernel(
    const int size,
    const SimpleArray<int, D> A_strides,
    const SimpleArray<int, D> B_strides,
    const SimpleArray<FIXED_DIVISOR, D> C_dims,
    const BinaryOperator op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  CUDA_1D_KERNEL_LOOP(C_index, size) {
    int A_index = 0;
    int B_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      FIXED_DIVISOR_DIV_MOD(C_dims.data[i], C_index_val, &C_index_val, &d);
      A_index += d * A_strides.data[i];
      B_index += d * B_strides.data[i];
    }
    C[C_index] = op(A[A_index], B[B_index]);
  }
}

template <typename TIn, typename TOut, class BinaryOperator>
CAFFE2_CUDA_EXPORT void BinaryOpWith2DBroadcasting(
    const int rows,
    const int cols,
    const bool rowwise_broadcast,
    const bool broadcast_1st,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C,
    CUDAContext* context) {
  if (rows == 0 || cols == 0) {
    return;
  }
  const int size = rows * cols;
  const FIXED_DIVISOR cols_div(cols);
  if (rowwise_broadcast) {
    if (broadcast_1st) {
      RowwiseBinaryOpCUDAKenel<TIn, TOut, BinaryOperator, true>
          <<<CAFFE_GET_BLOCKS(size),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context->cuda_stream()>>>(size, cols_div, op, A, B, C);
    } else {
      RowwiseBinaryOpCUDAKenel<TIn, TOut, BinaryOperator, false>
          <<<CAFFE_GET_BLOCKS(size),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context->cuda_stream()>>>(size, cols_div, op, A, B, C);
    }
  } else {
    if (broadcast_1st) {
      ColwiseBinaryOpCUDAKenel<TIn, TOut, BinaryOperator, true>
          <<<CAFFE_GET_BLOCKS(size),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context->cuda_stream()>>>(size, cols_div, op, A, B, C);
    } else {
      ColwiseBinaryOpCUDAKenel<TIn, TOut, BinaryOperator, false>
          <<<CAFFE_GET_BLOCKS(size),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context->cuda_stream()>>>(size, cols_div, op, A, B, C);
    }
  }
}

template <typename TIn, typename TOut, class BinaryOperator, int D>
CAFFE2_CUDA_EXPORT void BroadcastBinaryOpImpl(
    const int* A_dims,
    const int* B_dims,
    const int* C_dims,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C,
    CUDAContext* context) {
  SimpleArray<int, D> A_strides_array;
  SimpleArray<int, D> B_strides_array;
  SimpleArray<FIXED_DIVISOR, D> C_dims_array;
  int A_stride = 1;
  int B_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    if (C_dims[i] == 0) {
      return;
    }
    A_strides_array.data[i] = A_dims[i] == 1 ? 0 : A_stride;
    B_strides_array.data[i] = B_dims[i] == 1 ? 0 : B_stride;
    A_stride *= A_dims[i];
    B_stride *= B_dims[i];
    C_dims_array.data[i] = FIXED_DIVISOR(C_dims[i]);
  }
  const int size =
      std::accumulate(C_dims, C_dims + D, 1, std::multiplies<int>());
  BroadcastBinaryOpCUDAKernel<TIn, TOut, BinaryOperator, D>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          size, A_strides_array, B_strides_array, C_dims_array, op, A, B, C);
}

template <typename TIn, typename TOut, class BinaryOperator>
CAFFE2_CUDA_EXPORT void BroadcastBinaryOp(
    const int A_ndim,
    const int* A_dims,
    const int B_ndim,
    const int* B_dims,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C,
    CUDAContext* context) {
  const int ndim = std::max(A_ndim, B_ndim);
  std::vector<int> A_dims_array(ndim);
  std::vector<int> B_dims_array(ndim);
  std::vector<int> C_dims_array(ndim);
  utils::ComputeBroadcastBinaryOpDims(
      A_ndim,
      A_dims,
      B_ndim,
      B_dims,
      A_dims_array.data(),
      B_dims_array.data(),
      C_dims_array.data());
  if (A_dims_array == B_dims_array) {
    const int size = std::accumulate(
        C_dims_array.cbegin(), C_dims_array.cend(), 1, std::multiplies<int>());
    SimpleBinaryOpCUDAKernel<TIn, TOut, BinaryOperator>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(size, op, A, B, C);
    return;
  }
  int rows;
  int cols;
  bool broadcast_1st;
  if (utils::IsRowwiseBroadcastBinaryOp(
          ndim,
          A_dims_array.data(),
          B_dims_array.data(),
          &rows,
          &cols,
          &broadcast_1st)) {
    BinaryOpWith2DBroadcasting<TIn, TOut, BinaryOperator>(
        rows, cols, true, broadcast_1st, op, A, B, C, context);
    return;
  }
  if (utils::IsColwiseBroadcastBinaryOp(
          ndim,
          A_dims_array.data(),
          B_dims_array.data(),
          &rows,
          &cols,
          &broadcast_1st)) {
    BinaryOpWith2DBroadcasting<TIn, TOut, BinaryOperator>(
        rows, cols, false, broadcast_1st, op, A, B, C, context);
    return;
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_3(
      ndim,
      BroadcastBinaryOpImpl,
      TIn,
      TOut,
      BinaryOperator,
      A_dims_array.data(),
      B_dims_array.data(),
      C_dims_array.data(),
      op,
      A,
      B,
      C,
      context);
}

} // namespace

#define DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(T, Func, op)            \
  __global__ void Func##CUDAKernel(const int N, const T* X, T* Y) { \
    CUDA_1D_KERNEL_LOOP(i, N) {                                     \
      Y[i] = op(X[i]);                                              \
    }                                                               \
  }                                                                 \
  template <>                                                       \
  CAFFE2_CUDA_EXPORT void Func<T, CUDAContext>(                     \
      const int N, const T* x, T* y, CUDAContext* context) {        \
    Func##CUDAKernel<<<                                             \
        CAFFE_GET_BLOCKS(N),                                        \
        CAFFE_CUDA_NUM_THREADS,                                     \
        0,                                                          \
        context->cuda_stream()>>>(N, x, y);                         \
  }

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Exp, expf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Log, logf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cos, cosf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Acos, acosf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sin, sinf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Asin, asinf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tan, tanf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Atan, atanf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sinh, sinhf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cosh, coshf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tanh, tanhf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Abs, fabsf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqr, utils::Square<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqrt, sqrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Rsqrt, rsqrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cbrt, cbrtf)

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cube, utils::Cube<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Cube, utils::Cube<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Cube,
    utils::Cube<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Cube,
    utils::Cube<std::int64_t>)

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(bool, Not, utils::Not)

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Neg, utils::Negate<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Neg, utils::Negate<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Neg,
    utils::Negate<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Neg,
    utils::Negate<std::int64_t>)

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sign, utils::Sign<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Sign, utils::Sign<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Sign,
    utils::Sign<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Sign,
    utils::Sign<std::int64_t>)

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Inv, utils::Inv<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Inv, utils::Inv<double>)

#undef DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION

#define CAFFE2_SPECIALIZED_CUDA_SINCOS(T)                            \
  template <>                                                        \
  CAFFE2_CUDA_EXPORT void SinCos<T, CUDAContext>(                    \
      const int N, const T* x, T* ys, T* yc, CUDAContext* context) { \
    SinCosCUDAKernel<<<                                              \
        CAFFE_GET_BLOCKS(N),                                         \
        CAFFE_CUDA_NUM_THREADS,                                      \
        0,                                                           \
        context->cuda_stream()>>>(N, x, ys, yc);                     \
  }
CAFFE2_SPECIALIZED_CUDA_SINCOS(float)
CAFFE2_SPECIALIZED_CUDA_SINCOS(double)
#undef CAFFE2_SPECIALIZED_CUDA_SINCOS

#define DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(TIn, TOut, Func, Op) \
  template <>                                                     \
  CAFFE2_CUDA_EXPORT void Func<TIn, CUDAContext>(                 \
      const int N,                                                \
      const TIn* A,                                               \
      const TIn* B,                                               \
      TOut* C,                                                    \
      CUDAContext* context) {                                     \
    SimpleBinaryOpCUDAKernel<TIn, TOut, Op<TIn>>                  \
        <<<CAFFE_GET_BLOCKS(N),                                   \
           CAFFE_CUDA_NUM_THREADS,                                \
           0,                                                     \
           context->cuda_stream()>>>(N, Op<TIn>(), A, B, C);      \
  }

#define DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(EQ, thrust::equal_to)
DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(NE, thrust::not_equal_to)
DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(LT, thrust::less)
DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(LE, thrust::less_equal)
DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(GT, thrust::greater)
DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION(GE, thrust::greater_equal)

#undef DEFINE_SIMPLE_CUDA_COMPARE_FUNCTION

#define DEFINE_SIMPLE_CUDA_BINARY_FUNCTION(Func, Op)                         \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op) \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, float, Func, Op)               \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, double, Func, Op)             \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(at::Half, at::Half, Func, Op)

DEFINE_SIMPLE_CUDA_BINARY_FUNCTION(Add, AddFunctor)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTION(Sub, SubFunctor)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTION(Mul, MulFunctor)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTION(Div, DivFunctor)

#undef DEFINE_SIMPLE_CUDA_BINARY_FUNCTION

DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, bool, And, thrust::logical_and)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, bool, Or, thrust::logical_or)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, bool, Xor, thrust::bit_xor)

#define DEFINE_SIMPLE_CUDA_BITWISE_BINARY_FUNCTION(Func, Op)                 \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, bool, Func, Op)                 \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_SIMPLE_CUDA_BITWISE_BINARY_FUNCTION(BitwiseAnd, thrust::bit_and)
DEFINE_SIMPLE_CUDA_BITWISE_BINARY_FUNCTION(BitwiseOr, thrust::bit_or)
DEFINE_SIMPLE_CUDA_BITWISE_BINARY_FUNCTION(BitwiseXor, thrust::bit_xor)

#undef DEFINE_SIMPLE_CUDA_BITWISE_BINARY_FUNCTION

DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    float,
    float,
    ElemwiseMax,
    thrust::maximum);

#undef DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION

#define DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(TIn, TOut, Func, Op)   \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void Rowwise##Func<TIn, CUDAContext, true>(          \
      const int rows,                                                     \
      const int cols,                                                     \
      const TIn* A,                                                       \
      const TIn* B,                                                       \
      TOut* C,                                                            \
      CUDAContext* context) {                                             \
    if (rows == 0 || cols == 0) {                                         \
      return;                                                             \
    }                                                                     \
    const int size = rows * cols;                                         \
    const FIXED_DIVISOR cols_div(cols);                                   \
    RowwiseBinaryOpCUDAKenel<TIn, TOut, Op<TIn>, true>                    \
        <<<CAFFE_GET_BLOCKS(size),                                        \
           CAFFE_CUDA_NUM_THREADS,                                        \
           0,                                                             \
           context->cuda_stream()>>>(size, cols_div, Op<TIn>(), A, B, C); \
  }                                                                       \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void Rowwise##Func<TIn, CUDAContext, false>(         \
      const int rows,                                                     \
      const int cols,                                                     \
      const TIn* A,                                                       \
      const TIn* B,                                                       \
      TOut* C,                                                            \
      CUDAContext* context) {                                             \
    if (rows == 0 || cols == 0) {                                         \
      return;                                                             \
    }                                                                     \
    const int size = rows * cols;                                         \
    const FIXED_DIVISOR cols_div(cols);                                   \
    RowwiseBinaryOpCUDAKenel<TIn, TOut, Op<TIn>, false>                   \
        <<<CAFFE_GET_BLOCKS(size),                                        \
           CAFFE_CUDA_NUM_THREADS,                                        \
           0,                                                             \
           context->cuda_stream()>>>(size, cols_div, Op<TIn>(), A, B, C); \
  }                                                                       \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void Colwise##Func<TIn, CUDAContext, true>(          \
      const int rows,                                                     \
      const int cols,                                                     \
      const TIn* A,                                                       \
      const TIn* B,                                                       \
      TOut* C,                                                            \
      CUDAContext* context) {                                             \
    if (rows == 0 || cols == 0) {                                         \
      return;                                                             \
    }                                                                     \
    const int size = rows * cols;                                         \
    const FIXED_DIVISOR cols_div(cols);                                   \
    ColwiseBinaryOpCUDAKenel<TIn, TOut, Op<TIn>, true>                    \
        <<<CAFFE_GET_BLOCKS(size),                                        \
           CAFFE_CUDA_NUM_THREADS,                                        \
           0,                                                             \
           context->cuda_stream()>>>(size, cols_div, Op<TIn>(), A, B, C); \
  }                                                                       \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void Colwise##Func<TIn, CUDAContext, false>(         \
      const int rows,                                                     \
      const int cols,                                                     \
      const TIn* A,                                                       \
      const TIn* B,                                                       \
      TOut* C,                                                            \
      CUDAContext* context) {                                             \
    if (rows == 0 || cols == 0) {                                         \
      return;                                                             \
    }                                                                     \
    const int size = rows * cols;                                         \
    const FIXED_DIVISOR cols_div(cols);                                   \
    ColwiseBinaryOpCUDAKenel<TIn, TOut, Op<TIn>, false>                   \
        <<<CAFFE_GET_BLOCKS(size),                                        \
           CAFFE_CUDA_NUM_THREADS,                                        \
           0,                                                             \
           context->cuda_stream()>>>(size, cols_div, Op<TIn>(), A, B, C); \
  }

#define DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(EQ, thrust::equal_to)
DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(NE, thrust::not_equal_to)
DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(LT, thrust::less)
DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(LE, thrust::less_equal)
DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(GT, thrust::greater)
DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION(GE, thrust::greater_equal)

#undef DEFINE_2D_BROADCAST_CUDA_COMPARE_FUNCTION

#define DEFINE_2D_BROADCAST_CUDA_BINARY_FUNCTION(Func, Op)             \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(                          \
      std::int32_t, std::int32_t, Func, Op)                            \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(                          \
      std::int64_t, std::int64_t, Func, Op)                            \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(float, float, Func, Op)   \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(double, double, Func, Op) \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(at::Half, at::Half, Func, Op)

DEFINE_2D_BROADCAST_CUDA_BINARY_FUNCTION(Add, AddFunctor)
DEFINE_2D_BROADCAST_CUDA_BINARY_FUNCTION(Sub, SubFunctor)
DEFINE_2D_BROADCAST_CUDA_BINARY_FUNCTION(Mul, MulFunctor)
DEFINE_2D_BROADCAST_CUDA_BINARY_FUNCTION(Div, DivFunctor)

#undef DEFINE_2D_BROADCAST_CUDA_BINARY_FUNCTION

DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, And, thrust::logical_and)
DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Or, thrust::logical_or)
DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Xor, thrust::bit_xor)

#define DEFINE_2D_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(Func, Op) \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(                      \
      std::int32_t, std::int32_t, Func, Op)                        \
  DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION(                      \
      std::int64_t, std::int64_t, Func, Op)

DEFINE_2D_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(BitwiseAnd, thrust::bit_and)
DEFINE_2D_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(BitwiseOr, thrust::bit_or)
DEFINE_2D_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(BitwiseXor, thrust::bit_xor)

#undef DEFINE_2D_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION

#undef DELEGATE_2D_BROADCAST_CUDA_BINARY_FUNCTION

#define DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(TIn, TOut, Func, Op)  \
  template <>                                                         \
  CAFFE2_CUDA_EXPORT void Func<TIn, CUDAContext>(                     \
      const int A_ndim,                                               \
      const int* A_dims,                                              \
      const int B_ndim,                                               \
      const int* B_dims,                                              \
      const TIn* A,                                                   \
      const TIn* B,                                                   \
      TOut* C,                                                        \
      CUDAContext* context) {                                         \
    BroadcastBinaryOp<TIn, TOut, Op<TIn>>(                            \
        A_ndim, A_dims, B_ndim, B_dims, Op<TIn>(), A, B, C, context); \
  }

#define DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(EQ, thrust::equal_to)
DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(NE, thrust::not_equal_to)
DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(LT, thrust::less)
DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(LE, thrust::less_equal)
DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(GT, thrust::greater)
DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION(GE, thrust::greater_equal)

#undef DEFINE_BROADCAST_CUDA_COMPARE_FUNCTION

#define DEFINE_BROADCAST_CUDA_BINARY_FUNCTION(Func, Op)             \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(                          \
      std::int32_t, std::int32_t, Func, Op)                         \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(                          \
      std::int64_t, std::int64_t, Func, Op)                         \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(float, float, Func, Op)   \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(double, double, Func, Op) \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(at::Half, at::Half, Func, Op)

DEFINE_BROADCAST_CUDA_BINARY_FUNCTION(Add, AddFunctor)
DEFINE_BROADCAST_CUDA_BINARY_FUNCTION(Sub, SubFunctor)
DEFINE_BROADCAST_CUDA_BINARY_FUNCTION(Mul, MulFunctor)
DEFINE_BROADCAST_CUDA_BINARY_FUNCTION(Div, DivFunctor)

#undef DEFINE_BROADCAST_CUDA_BINARY_FUNCTION

DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, And, thrust::logical_and)
DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Or, thrust::logical_or)
DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Xor, thrust::bit_xor)

#define DEFINE_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(Func, Op) \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(bool, bool, Func, Op) \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(                      \
      std::int32_t, std::int32_t, Func, Op)                     \
  DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(BitwiseAnd, thrust::bit_and)
DEFINE_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(BitwiseOr, thrust::bit_or)
DEFINE_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION(BitwiseXor, thrust::bit_xor)

#undef DEFINE_BROADCAST_CUDA_BITWISE_BINARY_FUNCTION

#undef DELEGATE_BROADCAST_CUDA_BINARY_FUNCTION

#define DELEGATE_REDUCTION_FUNCTION(T, Funcname, func)                   \
  template <>                                                            \
  CAFFE2_CUDA_EXPORT void Funcname<T, CUDAContext>(                      \
      const int N,                                                       \
      const T* src,                                                      \
      T* dst,                                                            \
      Tensor* scratch_ptr,                                               \
      CUDAContext* context) {                                            \
    size_t memRequired = 0;                                              \
    cub::DeviceReduce::func(                                             \
        nullptr, memRequired, src, dst, N, context->cuda_stream());      \
    auto buffer_size =                                                   \
        static_cast<int64_t>((memRequired + sizeof(T) - 1) / sizeof(T)); \
    scratch_ptr->Resize(std::vector<int64_t>{buffer_size});              \
    cub::DeviceReduce::func(                                             \
        static_cast<void*>(scratch_ptr->mutable_data<T>()),              \
        memRequired,                                                     \
        src,                                                             \
        dst,                                                             \
        N,                                                               \
        context->cuda_stream());                                         \
  }

DELEGATE_REDUCTION_FUNCTION(float, ReduceMin, Min)
DELEGATE_REDUCTION_FUNCTION(float, ReduceMax, Max)
DELEGATE_REDUCTION_FUNCTION(int32_t, ReduceMax, Max)
DELEGATE_REDUCTION_FUNCTION(int64_t, ReduceMax, Max)

#undef DELEGATE_REDUCTION_FUNCTION

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <>
CAFFE2_CUDA_EXPORT void Gemm<float, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasSgemm(
      context->cublas_handle(),
      cu_trans_B,
      cu_trans_A,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      A,
      lda,
      &beta,
      C,
      N));
}

template <>
CAFFE2_CUDA_EXPORT void Gemm<at::Half, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const at::Half* A,
    const at::Half* B,
    const float beta,
    at::Half* C,
    CUDAContext* context,
    TensorProto::DataType math_type) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (math_type == TensorProto_DataType_FLOAT) {
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasSgemmEx(
        context->cublas_handle(),
        cu_trans_B,
        cu_trans_A,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_16F,
        ldb,
        A,
        CUDA_R_16F,
        lda,
        &beta,
        C,
        CUDA_R_16F,
        N));
  } else if (math_type == TensorProto_DataType_FLOAT16) {
    // convert alpha, beta from float -> __half
    const __half alpha_fp16 = at::Half(alpha);
    const __half beta_fp16 = at::Half(beta);
    // call cublasHgemm
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasHgemm(
        context->cublas_handle(),
        cu_trans_B,
        cu_trans_A,
        N,
        M,
        K,
        &alpha_fp16,
        (const __half*)B,
        ldb,
        (const __half*)A,
        lda,
        &beta_fp16,
        (__half*)C,
        N));
  } else {
    // fail
    CAFFE_THROW("Unsupported math type");
  }
#endif
}

template <>
CAFFE2_CUDA_EXPORT void BiasCHW<float, CUDAContext>(
    const float* bias,
    const float* bias_multiplier,
    const int bias_channels,
    const int image_size,
    float* image,
    CUDAContext* context) {
  Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      bias_channels,
      image_size,
      1,
      1,
      bias,
      bias_multiplier,
      1,
      image,
      context);
}

template <>
CAFFE2_CUDA_EXPORT void GemmBatched<float, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float** A,
    const float** B,
    const float beta,
    float** C,
    CUDAContext* context,
    TensorProto::DataType math_type) {
#if __CUDACC_VER_MAJOR__ < 8 || defined(__HIPCC__)
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    Gemm<float, CUDAContext>(
        trans_A,
        trans_B,
        M,
        N,
        K,
        alpha,
        A[i],
        B[i],
        beta,
        C[i],
        context,
        math_type);
  }
#else
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  thrust::device_vector<const float*> A_device(A, A + batch_size);
  thrust::device_vector<const float*> B_device(B, B + batch_size);
  thrust::device_vector<float*> C_device(C, C + batch_size);
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasSgemmBatched(
      context->cublas_handle(),
      cu_trans_B,
      cu_trans_A,
      N,
      M,
      K,
      &alpha,
      B_device.data().get(),
      ldb,
      A_device.data().get(),
      lda,
      &beta,
      C_device.data().get(),
      ldc,
      batch_size));
#endif
}

template <>
CAFFE2_CUDA_EXPORT void GemmStridedBatched<float, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int A_stride,
    const float* B,
    const int B_stride,
    const float beta,
    float* C,
    const int C_stride,
    CUDAContext* context,
    TensorProto::DataType math_type) {
#if __CUDACC_VER_MAJOR__ < 8 && !defined(__HIPCC__)
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    Gemm<float, CUDAContext>(
        trans_A, trans_B, M, N, K, alpha, A, B, beta, C, context, math_type);
    A += A_stride;
    B += B_stride;
    C += C_stride;
  }
#else
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasSgemmStridedBatched(
      context->cublas_handle(),
      cu_trans_B,
      cu_trans_A,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      B_stride,
      A,
      lda,
      A_stride,
      &beta,
      C,
      ldc,
      C_stride,
      batch_size));
#endif
}

template <>
CAFFE2_CUDA_EXPORT void GemmBatched<at::Half, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const at::Half** A,
    const at::Half** B,
    const float beta,
    at::Half** C,
    CUDAContext* context,
    TensorProto::DataType math_type) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
#if __CUDACC_VER_MAJOR__ < 9
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    Gemm<at::Half, CUDAContext>(
        trans_A,
        trans_B,
        M,
        N,
        K,
        alpha,
        A[i],
        B[i],
        beta,
        C[i],
        context,
        math_type);
  }
#else
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (math_type == TensorProto_DataType_FLOAT) {
#if CUDA_VERSION < 9010
    // loop over matrices in the batch
    for (int i = 0; i < batch_size; ++i) {
      Gemm<at::Half, CUDAContext>(
          trans_A,
          trans_B,
          M,
          N,
          K,
          alpha,
          A[i],
          B[i],
          beta,
          C[i],
          context,
          math_type);
    }
#else
    thrust::device_vector<const void*> A_device(A, A + batch_size);
    thrust::device_vector<const void*> B_device(B, B + batch_size);
    thrust::device_vector<void*> C_device(C, C + batch_size);
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasGemmBatchedEx(
        context->cublas_handle(),
        cu_trans_B,
        cu_trans_A,
        N,
        M,
        K,
        &alpha,
        B_device.data().get(),
        CUDA_R_16F,
        ldb,
        A_device.data().get(),
        CUDA_R_16F,
        lda,
        &beta,
        C_device.data().get(),
        CUDA_R_16F,
        ldc,
        batch_size,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
  } else if (math_type == TensorProto_DataType_FLOAT16) {
    // Convert alpha, beta from float -> __half
    const __half alpha_fp16 = at::Half(alpha);
    const __half beta_fp16 = at::Half(beta);
    std::vector<const __half*> A_array(batch_size);
    std::vector<const __half*> B_array(batch_size);
    std::vector<__half*> C_array(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      A_array[i] = reinterpret_cast<const __half*>(A[i]);
      B_array[i] = reinterpret_cast<const __half*>(B[i]);
      C_array[i] = reinterpret_cast<__half*>(C[i]);
    }
    thrust::device_vector<const __half*> A_device(
        A_array.cbegin(), A_array.cend());
    thrust::device_vector<const __half*> B_device(
        B_array.cbegin(), B_array.cend());
    thrust::device_vector<__half*> C_device(C_array.cbegin(), C_array.cend());
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasHgemmBatched(
        context->cublas_handle(),
        cu_trans_B,
        cu_trans_A,
        N,
        M,
        K,
        &alpha_fp16,
        B_device.data().get(),
        ldb,
        A_device.data().get(),
        lda,
        &beta_fp16,
        C_device.data().get(),
        ldc,
        batch_size));
  } else {
    CAFFE_THROW("Unsupported math type");
  }
#endif
#endif
}

template <>
CAFFE2_CUDA_EXPORT void GemmStridedBatched<at::Half, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const at::Half* A,
    const int A_stride,
    const at::Half* B,
    const int B_stride,
    const float beta,
    at::Half* C,
    const int C_stride,
    CUDAContext* context,
    TensorProto::DataType math_type) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
#if __CUDACC_VER_MAJOR__ < 8
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    Gemm<at::Half, CUDAContext>(
        trans_A, trans_B, M, N, K, alpha, A, B, beta, C, context, math_type);
    A += A_stride;
    B += B_stride;
    C += C_stride;
  }
#else
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (math_type == TensorProto_DataType_FLOAT) {
#if CUDA_VERSION < 9010
    // loop over matrices in the batch
    for (int i = 0; i < batch_size; ++i) {
      Gemm<at::Half, CUDAContext>(
          trans_A, trans_B, M, N, K, alpha, A, B, beta, C, context, math_type);
      A += A_stride;
      B += B_stride;
      C += C_stride;
    }
#else
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(
        context->cublas_handle(),
        cu_trans_B,
        cu_trans_A,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_16F,
        ldb,
        B_stride,
        A,
        CUDA_R_16F,
        lda,
        A_stride,
        &beta,
        C,
        CUDA_R_16F,
        ldc,
        C_stride,
        batch_size,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
  } else if (math_type == TensorProto_DataType_FLOAT16) {
    // Convert alpha, beta from float -> __half
    const __half alpha_fp16 = at::Half(alpha);
    const __half beta_fp16 = at::Half(beta);
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasHgemmStridedBatched(
        context->cublas_handle(),
        cu_trans_B,
        cu_trans_A,
        N,
        M,
        K,
        &alpha_fp16,
        (const __half*)B,
        ldb,
        B_stride,
        (const __half*)A,
        lda,
        A_stride,
        &beta_fp16,
        (__half*)C,
        ldc,
        C_stride,
        batch_size));
  } else {
    CAFFE_THROW("Unsupported math type");
  }
#endif
#endif
}

#if CUDA_VERSION >= 9000

// No change, but required. Defer to default CUDA engine
template <>
CAFFE2_CUDA_EXPORT void Gemm<float, CUDAContext, TensorCoreEngine>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  return Gemm<float, CUDAContext>(
      trans_A, trans_B, M, N, K, alpha, A, B, beta, C, context, math_type);
}

template <>
CAFFE2_CUDA_EXPORT void Gemm<at::Half, CUDAContext, TensorCoreEngine>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const at::Half* A,
    const at::Half* B,
    const float beta,
    at::Half* C,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // enable TensorCore for this call on this handle
  if (TensorCoreAvailable()) {
    CUBLAS_ENFORCE(
        cublasSetMathMode(context->cublas_handle(), CUBLAS_TENSOR_OP_MATH));
  }

  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasGemmEx(
      context->cublas_handle(),
      cu_trans_B,
      cu_trans_A,
      N,
      M,
      K,
      &alpha,
      B,
      CUDA_R_16F,
      ldb,
      A,
      CUDA_R_16F,
      lda,
      &beta,
      C,
      CUDA_R_16F,
      N,
      CUDA_R_32F,
      CUBLAS_GEMM_DFALT_TENSOR_OP));

  // Now disable TensorCore math for subsequent calls to this handle
  if (TensorCoreAvailable()) {
    CUBLAS_ENFORCE(
        cublasSetMathMode(context->cublas_handle(), CUBLAS_DEFAULT_MATH));
  }
}

template <>
CAFFE2_CUDA_EXPORT void
GemmStridedBatched<float, CUDAContext, TensorCoreEngine>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int A_stride,
    const float* B,
    const int B_stride,
    const float beta,
    float* C,
    const int C_stride,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  return GemmStridedBatched<float, CUDAContext, DefaultEngine>(
      trans_A,
      trans_B,
      batch_size,
      M,
      N,
      K,
      alpha,
      A,
      A_stride,
      B,
      B_stride,
      beta,
      C,
      C_stride,
      context,
      math_type);
}

template <>
CAFFE2_CUDA_EXPORT void
GemmStridedBatched<at::Half, CUDAContext, TensorCoreEngine>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const at::Half* A,
    const int A_stride,
    const at::Half* B,
    const int B_stride,
    const float beta,
    at::Half* C,
    const int C_stride,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  return GemmStridedBatched<at::Half, CUDAContext, DefaultEngine>(
      trans_A,
      trans_B,
      batch_size,
      M,
      N,
      K,
      alpha,
      A,
      A_stride,
      B,
      B_stride,
      beta,
      C,
      C_stride,
      context,
      math_type);
}

#endif // CUDA_VERSION >= 9000

template <>
CAFFE2_CUDA_EXPORT void GemmEx<float, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int lda,
    const float* B,
    const int ldb,
    const float beta,
    float* C,
    const int ldc,
    CUDAContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cu_trans_B =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasSgemm(
      context->cublas_handle(),
      cu_trans_B,
      cu_trans_A,
      N,
      M,
      K,
      &alpha,
      B,
      ldb,
      A,
      lda,
      &beta,
      C,
      ldc));
}

template <>
CAFFE2_CUDA_EXPORT void Gemv<float, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CUDAContext* context,
    TensorProto::DataType math_type) {
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasSgemv(
      context->cublas_handle(),
      cu_trans_A,
      N,
      M,
      &alpha,
      A,
      N,
      x,
      1,
      &beta,
      y,
      1));
}

// Batched Add variants
namespace {

template <typename T>
__global__ void AddStripedBatchKernel(
    const int N,
    const T* first,
    T* Y,
    const int stripe,
    const int batch) {
  for (int j = 0; j < batch; j++) {
    const T* x = first + j * stripe;
    CUDA_1D_KERNEL_LOOP(i, N) {
      float tmpY = convert::To<T, float>(Y[i]);
      tmpY += convert::To<T, float>(x[i]);
      Y[i] = convert::To<float, T>(tmpY);
    }
  }
}
} // namespace

#define CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH(T)              \
  template <>                                                     \
  CAFFE2_CUDA_EXPORT void AddStripedBatch<T, CUDAContext>(        \
      const int N,                                                \
      const T* first,                                             \
      T* Y,                                                       \
      const int stripe,                                           \
      const int batch,                                            \
      CUDAContext* context) {                                     \
    AddStripedBatchKernel<T>                                      \
        <<<CAFFE_GET_BLOCKS(N),                                   \
           CAFFE_CUDA_NUM_THREADS,                                \
           0,                                                     \
           context->cuda_stream()>>>(N, first, Y, stripe, batch); \
  }

CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH(float);
CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH(at::Half);
#undef CAFFE2_SPECIALIZED_CUDA_ADD_STRIPED_BATCH

template <>
CAFFE2_CUDA_EXPORT void Gemv<at::Half, CUDAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const at::Half* A,
    const at::Half* x,
    const float beta,
    at::Half* y,
    CUDAContext* context,
    TensorProto::DataType math_type) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
  const cublasOperation_t cu_trans_A =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

  // sort out what we need to call cublasSgemmEx / cublasHgemm
  const int m = (cu_trans_A == CUBLAS_OP_N) ? N : M;
  const int k = (cu_trans_A == CUBLAS_OP_N) ? M : N;
  const int lda = (cu_trans_A == CUBLAS_OP_N) ? m : k;
  const int ldc = m;

  if (math_type == TensorProto_DataType_FLOAT) {
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasSgemmEx(
        context->cublas_handle(),
        cu_trans_A,
        CUBLAS_OP_N,
        m,
        1,
        k,
        &alpha,
        A,
        CUDA_R_16F,
        lda,
        x,
        CUDA_R_16F,
        k,
        &beta,
        y,
        CUDA_R_16F,
        ldc));
  } else if (math_type == TensorProto_DataType_FLOAT16) {
    const __half alpha_fp16 = at::Half(alpha);
    const __half beta_fp16 = at::Half(beta);
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasHgemm(
        context->cublas_handle(),
        cu_trans_A,
        CUBLAS_OP_N,
        m,
        1,
        k,
        &alpha_fp16,
        (const __half*)A,
        lda,
        (const __half*)x,
        k,
        &beta_fp16,
        (__half*)y,
        ldc));
  } else {
    // fail
    CAFFE_THROW("Unsupported math type");
  }
#endif
}

namespace {

template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = alpha;
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_SET(T)                              \
  template <>                                                       \
  CAFFE2_CUDA_API void Set<T, CUDAContext>(                         \
      const size_t N, const T alpha, T* Y, CUDAContext* context) {  \
    if (N == 0) {                                                   \
      return;                                                       \
    }                                                               \
    if (alpha == T(0)) {                                            \
      cudaMemsetAsync(Y, 0, sizeof(T) * N, context->cuda_stream()); \
    } else {                                                        \
      SetKernel<T>                                                  \
          <<<CAFFE_GET_BLOCKS(N),                                   \
             CAFFE_CUDA_NUM_THREADS,                                \
             0,                                                     \
             context->cuda_stream()>>>(N, alpha, Y);                \
    }                                                               \
  }
CAFFE2_SPECIALIZED_CUDA_SET(float);
CAFFE2_SPECIALIZED_CUDA_SET(double);
CAFFE2_SPECIALIZED_CUDA_SET(bool);
CAFFE2_SPECIALIZED_CUDA_SET(int8_t);
CAFFE2_SPECIALIZED_CUDA_SET(int16_t);
CAFFE2_SPECIALIZED_CUDA_SET(int);
CAFFE2_SPECIALIZED_CUDA_SET(int64_t);
CAFFE2_SPECIALIZED_CUDA_SET(char);
CAFFE2_SPECIALIZED_CUDA_SET(uint8_t);
CAFFE2_SPECIALIZED_CUDA_SET(uint16_t);
#undef CAFFE2_SPECIALIZED_CUDA_SET

template <>
CAFFE2_CUDA_EXPORT void Set<at::Half, CUDAContext>(
    const size_t N,
    const at::Half alpha,
    at::Half* Y,
    CUDAContext* context) {
  if (N > 0) {
    SetKernel<at::Half>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(N, alpha, Y);
  }
}

namespace {
template <typename T>
__global__ void
UniformShift(const size_t N, const float min, const float max, T* x) {
  float scale = max - min;
  CUDA_1D_KERNEL_LOOP(i, N) {
    x[i] = convert::To<float, T>(convert::To<T, float>(x[i]) * scale + min);
  }
}

__global__ void
UniformIntFit(const size_t N, const int min, const int max, unsigned int* x) {
  int* x_int = reinterpret_cast<int*>(x);
  int range = (max - min + 1);
  CUDA_1D_KERNEL_LOOP(i, N) {
    x_int[i] = min + static_cast<int>(x[i] % range);
  }
}
} // namespace

template <>
CAFFE2_CUDA_EXPORT void RandUniform<float, CUDAContext>(
    const size_t n,
    const float min,
    const float max,
    float* r,
    CUDAContext* context) {
  CURAND_ENFORCE(curandGenerateUniform(context->curand_generator(), r, n));
  UniformShift<float>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, min, max, r);
}

template <>
CAFFE2_CUDA_EXPORT void RandUniform<double, CUDAContext>(
    const size_t n,
    const double min,
    const double max,
    double* r,
    CUDAContext* context) {
  CURAND_ENFORCE(
      curandGenerateUniformDouble(context->curand_generator(), r, n));
  UniformShift<double>
      <<<CAFFE_GET_BLOCKS(n),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(n, min, max, r);
}

template <>
CAFFE2_CUDA_EXPORT void RandUniform<int, CUDAContext>(
    const size_t n,
    const int min,
    const int max,
    int* r,
    CUDAContext* context) {
  CURAND_ENFORCE(curandGenerate(
      context->curand_generator(), reinterpret_cast<unsigned int*>(r), n));
  UniformIntFit<<<
      CAFFE_GET_BLOCKS(n),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      n, min, max, reinterpret_cast<unsigned int*>(r));
}

template <typename T>
size_t HandleOddLengthRandGaussian(
    const size_t n,
    const T mean,
    const T std,
    T* r,
    CUDAContext* context) {
  if (n % 2 == 1) {
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(mean, std);
    const T random_value = distribution(generator);
    Set<T, CUDAContext>(1, random_value, r + (n - 1), context);
    return n - 1;
  }
  return n;
}

template <>
CAFFE2_CUDA_EXPORT void RandGaussian<float, CUDAContext>(
    const size_t n,
    const float mean,
    const float std,
    float* r,
    CUDAContext* context) {
  // If n is odd, we add a random Gaussian value at the end manually
  // and generate n-1 random values using curandGenerateNormal.
  // curandGenerateNormal requires n to be even.
  const size_t even_n =
      HandleOddLengthRandGaussian<float>(n, mean, std, r, context);
  CURAND_ENFORCE(
      curandGenerateNormal(context->curand_generator(), r, even_n, mean, std));
}

template <>
CAFFE2_CUDA_EXPORT void RandGaussian<double, CUDAContext>(
    const size_t n,
    const double mean,
    const double std,
    double* r,
    CUDAContext* context) {
  const size_t even_n =
      HandleOddLengthRandGaussian<double>(n, mean, std, r, context);
  CURAND_ENFORCE(curandGenerateNormalDouble(
      context->curand_generator(), r, even_n, mean, std));
}

template <>
CAFFE2_CUDA_EXPORT void Dot<float, CUDAContext>(
    const int n,
    const float* a,
    const float* b,
    float* y,
    CUDAContext* context) {
  CUBLAS_ENFORCE(cublasSetPointerMode(
      context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_ENFORCE(cublasSdot(context->cublas_handle(), n, a, 1, b, 1, y));
}

template <>
CAFFE2_CUDA_EXPORT void Dot<at::Half, CUDAContext>(
    const int n,
    const at::Half* a,
    const at::Half* b,
    at::Half* y,
    CUDAContext* context) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
  // execute with 32-bit math
  CUBLAS_ENFORCE(cublasSetPointerMode(
      context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_ENFORCE(cublasDotEx(
      context->cublas_handle(),
      n,
      a,
      CUDA_R_16F,
      1,
      b,
      CUDA_R_16F,
      1,
      y,
      CUDA_R_16F,
      CUDA_R_32F));
#endif
}

// A previous version of caffe2 used Thrust but it turns out that thrust
// reduction has an implicit scratch space allocation and deallocation, which
// may interfere with NCCL and create a deadlock. Hence we are using a custom
// reduction here.
#define SUM_KERNEL_NTHREADS 128
template <typename T>
__global__ void SumKernel(const int N, const T* X, T* Y, bool square) {
  const int idx = threadIdx.x;
  __shared__ float reduction_buffer[SUM_KERNEL_NTHREADS];

  reduction_buffer[idx] = 0;

  // A multilevel reduction.
  // N -> 128
  if (!square) {
    for (int i = idx; i < N; i += SUM_KERNEL_NTHREADS) {
      reduction_buffer[idx] += convert::To<T, float>(X[i]);
    }
  } else {
    for (int i = idx; i < N; i += SUM_KERNEL_NTHREADS) {
      float Xi = convert::To<T, float>(X[i]);
      reduction_buffer[idx] += Xi * Xi;
    }
  }
  __syncthreads();
  // 128 -> 32
  if (idx < 32) {
    reduction_buffer[idx] += reduction_buffer[idx + 32] +
        reduction_buffer[idx + 64] + reduction_buffer[idx + 96];
  }
  __syncthreads();
  // 32 -> 1
  if (idx == 0) {
    float tmp = 0;
    for (int i = 0; i < 32; ++i) {
      tmp += reduction_buffer[i];
    }
    *Y = convert::To<float, T>(tmp);
  }
}

// According to the benchmarks script
// caffe2/caffe2/experiments/python/device_reduce_sum_bench.py,
// device reduce is slower for N <= 10000.
#define DEVICE_REDUCE_SIZE_THRESHOLD 10000

namespace {

template <typename T>
__global__ void SumConvertKernel(float* sum, T* dest) {
  *dest = convert::To<float, T>(*sum);
}

template <typename T, typename IterT>
CAFFE2_CUDA_EXPORT void SumGenericIter(
    const int N,
    IterT it,
    T*& dest,
    CUDAContext* context,
    Tensor* scratch_ptr) {
  size_t memRequired = 0;
  cub::DeviceReduce::Sum(
      nullptr, memRequired, it, dest, N, context->cuda_stream());
  auto buffer_size =
      static_cast<int64_t>((memRequired + sizeof(T) - 1) / sizeof(T));
  if (!dest) {
    // allocate one more T at the end of scratch for dest
    scratch_ptr->Resize(std::vector<int64_t>{buffer_size + 1});
    dest = scratch_ptr->template mutable_data<T>() + buffer_size;
  } else {
    scratch_ptr->Resize(std::vector<int64_t>{buffer_size});
  }
  cub::DeviceReduce::Sum(
      static_cast<void*>(scratch_ptr->template mutable_data<T>()),
      memRequired,
      it,
      dest,
      N,
      context->cuda_stream());
}
} // namespace

template <>
CAFFE2_CUDA_EXPORT void Sum<float, CUDAContext>(
    const int N,
    const float* x,
    float* y,
    CUDAContext* context,
    Tensor* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SumGenericIter<float>(N, x, y, context, scratch_ptr);
  } else {
    SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(
        N, x, y, false);
  }
}

template <>
CAFFE2_CUDA_EXPORT void Sum<int32_t, CUDAContext>(
    const int N,
    const int32_t* x,
    int32_t* y,
    CUDAContext* context,
    Tensor* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SumGenericIter<int32_t>(N, x, y, context, scratch_ptr);
  } else {
    SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(
        N, x, y, false);
  }
}

namespace {
template <typename T>
struct FloatTransform {
  inline __host__ __device__ float operator()(const T v) const {
    return convert::To<T, float>(v);
  }
};
} // namespace

#define CAFFE2_MATH_SUM_FUNC(T)                                           \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void Sum<T, CUDAContext>(                            \
      const int N,                                                        \
      const T* x,                                                         \
      T* y,                                                               \
      CUDAContext* context,                                               \
      Tensor* scratch_ptr) {                                              \
    if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {                \
      FloatTransform<T> transform;                                        \
      cub::TransformInputIterator<float, FloatTransform<T>, const T*> it( \
          x, transform);                                                  \
      float* sum = nullptr;                                               \
      SumGenericIter<float>(N, it, sum, context, scratch_ptr);            \
      SumConvertKernel<<<1, 1, 0, context->cuda_stream()>>>(sum, y);      \
    } else {                                                              \
      SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(   \
          N, x, y, false);                                                \
    }                                                                     \
  }

CAFFE2_MATH_SUM_FUNC(at::Half)
#undef CAFFE2_MATH_SUM_FUNC

namespace {
template <typename T>
struct SqrTransform {
  inline __host__ __device__ T operator()(const T v) const {
    return v * v;
  }
};
} //  namespace

template <>
CAFFE2_CUDA_EXPORT void SumSqr<float, CUDAContext>(
    const int N,
    const float* x,
    float* y,
    CUDAContext* context,
    Tensor* scratch_ptr) {
  if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {
    SqrTransform<float> transform;
    cub::TransformInputIterator<float, SqrTransform<float>, const float*> it(
        x, transform);
    SumGenericIter<float>(N, it, y, context, scratch_ptr);
  } else {
    SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(
        N, x, y, true);
  }
}

#define CAFFE2_MATH_SUMSQR_FUNC(T)                                      \
  template <>                                                           \
  CAFFE2_CUDA_EXPORT void SumSqr<T, CUDAContext>(                       \
      const int N,                                                      \
      const T* x,                                                       \
      T* y,                                                             \
      CUDAContext* context,                                             \
      Tensor* scratch_ptr) {                                            \
    if (scratch_ptr && N > DEVICE_REDUCE_SIZE_THRESHOLD) {              \
      FloatTransform<T> float_transform;                                \
      cub::TransformInputIterator<float, FloatTransform<T>, const T*>   \
          float_it(x, float_transform);                                 \
      SqrTransform<float> sqr_transform;                                \
      cub::TransformInputIterator<                                      \
          float,                                                        \
          SqrTransform<float>,                                          \
          decltype(float_it)>                                           \
          it(float_it, sqr_transform);                                  \
      float* sum = nullptr;                                             \
      SumGenericIter<float>(N, it, sum, context, scratch_ptr);          \
      SumConvertKernel<<<1, 1, 0, context->cuda_stream()>>>(sum, y);    \
    } else {                                                            \
      SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>( \
          N, x, y, true);                                               \
    }                                                                   \
  }

CAFFE2_MATH_SUMSQR_FUNC(at::Half)
#undef CAFFE2_MATH_SUMSQR_FUNC
#undef DEVICE_REDUCE_SIZE_THRESHOLD

namespace {
template <typename T>
__global__ void
SelectKernel(const int N, const int D, const T* x, const int* idx, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i * D + idx[i]];
  }
}
} // namespace

template <>
CAFFE2_CUDA_EXPORT void Select<float, CUDAContext>(
    const int N,
    const int D,
    const float* x,
    const int* idx,
    float* y,
    CUDAContext* context) {
  SelectKernel<float>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, D, x, idx, y);
}

template <>
CAFFE2_CUDA_EXPORT void Select<at::Half, CUDAContext>(
    const int N,
    const int D,
    const at::Half* x,
    const int* idx,
    at::Half* y,
    CUDAContext* context) {
  SelectKernel<at::Half>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, D, x, idx, y);
}

namespace {

template <typename TAlpha, typename TData>
__global__ void
ScaleCUDAKernel(const int n, const TAlpha alpha, const TData* x, TData* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    y[i] = __ldg(x + i) * static_cast<TData>(alpha);
#else
    y[i] = x[i] * static_cast<TData>(alpha);
#endif
  }
}

template <typename TAlpha, typename TData>
__global__ void
ScaleCUDAKernel(const int n, const TAlpha* alpha, const TData* x, TData* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    y[i] = __ldg(x + i) * static_cast<TData>(__ldg(alpha));
#else
    y[i] = x[i] * static_cast<TData>(*alpha);
#endif
  }
}

template <typename T>
__global__ void PowKernel(const int n, const T* x, const T exponent, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = powf(x[i], exponent);
  }
}

} // namespace

template <>
CAFFE2_CUDA_EXPORT void Powx<float, CUDAContext>(
    const int N,
    const float* a,
    const float b,
    float* y,
    CUDAContext* context) {
  PowKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, a, b, y);
}

#define DELEGATE_CUBLAS_SCALE_FUNCTION(TAlpha, TData, CuBLASFunc)            \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>(                 \
      const int N,                                                           \
      const TAlpha alpha,                                                    \
      const TData* x,                                                        \
      TData* y,                                                              \
      CUDAContext* context) {                                                \
    if (N == 0) {                                                            \
      return;                                                                \
    }                                                                        \
    if (x != y) {                                                            \
      cudaMemcpyAsync(                                                       \
          y,                                                                 \
          x,                                                                 \
          sizeof(TData) * N,                                                 \
          cudaMemcpyDeviceToDevice,                                          \
          context->cuda_stream());                                           \
    }                                                                        \
    if (alpha != TAlpha(1)) {                                                \
      CUBLAS_ENFORCE(cublasSetPointerMode(                                   \
          context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));              \
      CUBLAS_ENFORCE(CuBLASFunc(context->cublas_handle(), N, &alpha, y, 1)); \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>(                 \
      const int N,                                                           \
      const TAlpha* alpha,                                                   \
      const TData* x,                                                        \
      TData* y,                                                              \
      CUDAContext* context) {                                                \
    if (N == 0) {                                                            \
      return;                                                                \
    }                                                                        \
    if (x != y) {                                                            \
      cudaMemcpyAsync(                                                       \
          y,                                                                 \
          x,                                                                 \
          sizeof(TData) * N,                                                 \
          cudaMemcpyDeviceToDevice,                                          \
          context->cuda_stream());                                           \
    }                                                                        \
    CUBLAS_ENFORCE(cublasSetPointerMode(                                     \
        context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));              \
    CUBLAS_ENFORCE(CuBLASFunc(context->cublas_handle(), N, alpha, y, 1));    \
  }
DELEGATE_CUBLAS_SCALE_FUNCTION(float, float, cublasSscal)
DELEGATE_CUBLAS_SCALE_FUNCTION(double, double, cublasDscal)
#undef DELEGATE_CUBLAS_SCALE_FUNCTION

#define CAFFE2_SPECIALIZED_CUDA_SCALE(TAlpha, TData)         \
  template <>                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>( \
      const int N,                                           \
      const TAlpha alpha,                                    \
      const TData* x,                                        \
      TData* y,                                              \
      CUDAContext* context) {                                \
    if (N == 0) {                                            \
      return;                                                \
    }                                                        \
    if (alpha == TAlpha(1)) {                                \
      if (x != y) {                                          \
        cudaMemcpyAsync(                                     \
            y,                                               \
            x,                                               \
            sizeof(TData) * N,                               \
            cudaMemcpyDeviceToDevice,                        \
            context->cuda_stream());                         \
      }                                                      \
      return;                                                \
    }                                                        \
    ScaleCUDAKernel<TAlpha, TData>                           \
        <<<CAFFE_GET_BLOCKS(N),                              \
           CAFFE_CUDA_NUM_THREADS,                           \
           0,                                                \
           context->cuda_stream()>>>(N, alpha, x, y);        \
  }                                                          \
  template <>                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>( \
      const int N,                                           \
      const TAlpha* alpha,                                   \
      const TData* x,                                        \
      TData* y,                                              \
      CUDAContext* context) {                                \
    if (N == 0) {                                            \
      return;                                                \
    }                                                        \
    ScaleCUDAKernel<TAlpha, TData>                           \
        <<<CAFFE_GET_BLOCKS(N),                              \
           CAFFE_CUDA_NUM_THREADS,                           \
           0,                                                \
           context->cuda_stream()>>>(N, alpha, x, y);        \
  }
CAFFE2_SPECIALIZED_CUDA_SCALE(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_CUDA_SCALE(std::int64_t, std::int64_t)

#ifndef __HIPCC__
template <>
CAFFE2_CUDA_EXPORT void Scale<at::Half, at::Half, CUDAContext>(
    const int N,
    const at::Half alpha,
    const at::Half* x,
    at::Half* y,
    CUDAContext* context) {
  if (N == 0) {
    return;
  }
  if (x != y) {
    cudaMemcpyAsync(
        y,
        x,
        sizeof(at::Half) * N,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
  }
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasScalEx(
      context->cublas_handle(),
      N,
      &alpha,
      CUDA_R_16F,
      y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
}

template <>
CAFFE2_CUDA_EXPORT void Scale<at::Half, at::Half, CUDAContext>(
    const int N,
    const at::Half* alpha,
    const at::Half* x,
    at::Half* y,
    CUDAContext* context) {
  if (N == 0) {
    return;
  }
  if (x != y) {
    cudaMemcpyAsync(
        y,
        x,
        sizeof(at::Half) * N,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
  }
  CUBLAS_ENFORCE(cublasSetPointerMode(
      context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_ENFORCE(cublasScalEx(
      context->cublas_handle(),
      N,
      alpha,
      CUDA_R_16F,
      y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
}

template <>
CAFFE2_CUDA_EXPORT void Scale<float, at::Half, CUDAContext>(
    const int N,
    const float alpha,
    const at::Half* x,
    at::Half* y,
    CUDAContext* context) {
  if (N == 0) {
    return;
  }
  if (x != y) {
    cudaMemcpyAsync(
        y,
        x,
        sizeof(at::Half) * N,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
  }
  if (alpha != 1.0f) {
    CUBLAS_ENFORCE(cublasSetPointerMode(
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ENFORCE(cublasScalEx(
        context->cublas_handle(),
        N,
        &alpha,
        CUDA_R_32F,
        y,
        CUDA_R_16F,
        1,
        CUDA_R_32F));
  }
}

template <>
CAFFE2_CUDA_EXPORT void Scale<float, at::Half, CUDAContext>(
    const int N,
    const float* alpha,
    const at::Half* x,
    at::Half* y,
    CUDAContext* context) {
  if (N == 0) {
    return;
  }
  if (x != y) {
    cudaMemcpyAsync(
        y,
        x,
        sizeof(at::Half) * N,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
  }
  CUBLAS_ENFORCE(cublasSetPointerMode(
      context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_ENFORCE(cublasScalEx(
      context->cublas_handle(),
      N,
      alpha,
      CUDA_R_32F,
      y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
}

#else // __HIPCC__

namespace {
template <>
__global__ void ScaleCUDAKernel<at::Half, at::Half>(
    const int n,
    const at::Half alpha,
    const at::Half* x,
    at::Half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) *
        convert::To<at::Half, float>(alpha));
  }
}

template <>
__global__ void ScaleCUDAKernel<at::Half, at::Half>(
    const int n,
    const at::Half* alpha,
    const at::Half* x,
    at::Half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) *
        convert::To<at::Half, float>(*alpha));
  }
}

template <>
__global__ void ScaleCUDAKernel<float, at::Half>(
    const int n,
    const float alpha,
    const at::Half* x,
    at::Half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) * alpha);
  }
}

template <>
__global__ void ScaleCUDAKernel<float, at::Half>(
    const int n,
    const float* alpha,
    const at::Half* x,
    at::Half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) * (*alpha));
  }
}
} // namespace

CAFFE2_SPECIALIZED_HIP_SCALE(at::Half, at::Half)
CAFFE2_SPECIALIZED_HIP_SCALE(float, at::Half)
#endif // __HIPCC__

#undef CAFFE2_SPECIALIZED_CUDA_SCALE

template <>
CAFFE2_CUDA_EXPORT void Axpy<float, CUDAContext>(
    const int N,
    const float alpha,
    const float* X,
    float* Y,
    CUDAContext* context) {
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasSaxpy(context->cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
CAFFE2_CUDA_EXPORT void Axpy<double, CUDAContext>(
    const int N,
    const float alpha,
    const double* X,
    double* Y,
    CUDAContext* context) {
  double alpha_d{alpha};
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(
      cublasDaxpy(context->cublas_handle(), N, &alpha_d, X, 1, Y, 1));
}

template <>
CAFFE2_CUDA_EXPORT void Axpy<at::Half, CUDAContext>(
    const int N,
    const float alpha,
    const at::Half* X,
    at::Half* Y,
    CUDAContext* context) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
  CUBLAS_ENFORCE(
      cublasSetPointerMode(context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));
  CUBLAS_ENFORCE(cublasAxpyEx(
      context->cublas_handle(),
      N,
      &alpha,
      CUDA_R_32F,
      X,
      CUDA_R_16F,
      1,
      Y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
#endif
}

template <>
CAFFE2_CUDA_EXPORT void Axpy<float, CUDAContext>(
    const int N,
    const float* alpha,
    const float* X,
    float* Y,
    CUDAContext* context) {
  CUBLAS_ENFORCE(cublasSetPointerMode(
      context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_ENFORCE(cublasSaxpy(context->cublas_handle(), N, alpha, X, 1, Y, 1));
}

template <>
CAFFE2_CUDA_EXPORT void Axpy<at::Half, CUDAContext>(
    const int N,
    const float* alpha,
    const at::Half* X,
    at::Half* Y,
    CUDAContext* context) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP currently does not support FP16 yet.");
#else
  CUBLAS_ENFORCE(cublasSetPointerMode(
      context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_ENFORCE(cublasAxpyEx(
      context->cublas_handle(),
      N,
      alpha,
      CUDA_R_32F,
      X,
      CUDA_R_16F,
      1,
      Y,
      CUDA_R_16F,
      1,
      CUDA_R_32F));
#endif
}

namespace {

template <typename TCoeff, typename TData>
__global__ void AxpbyCUDAKernel(
    const int N,
    const TCoeff a,
    const TData* x,
    const TCoeff b,
    TData* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) * a + y[i] * b;
#else
    y[i] = x[i] * a + y[i] * b;
#endif
  }
}

template <>
__global__ void AxpbyCUDAKernel<float, at::Half>(
    const int N,
    const float a,
    const at::Half* x,
    const float b,
    at::Half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) * a +
        convert::To<at::Half, float>(y[i]) * b);
  }
}

template <typename TCoeff, typename TData>
__global__ void AxpbyCUDAKernel(
    const int N,
    const TCoeff* a,
    const TData* x,
    const TCoeff* b,
    TData* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) * __ldg(a) + y[i] * __ldg(b);
#else
    y[i] = x[i] * *a + y[i] * *b;
#endif
  }
}

template <>
__global__ void AxpbyCUDAKernel<float, at::Half>(
    const int N,
    const float* a,
    const at::Half* x,
    const float* b,
    at::Half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) * __ldg(a) +
        convert::To<at::Half, float>(y[i]) * __ldg(b));
#else
    y[i] = convert::To<float, at::Half>(
        convert::To<at::Half, float>(x[i]) * *a +
        convert::To<at::Half, float>(y[i]) * *b);
#endif
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_AXPBY(TCoeff, TData)         \
  template <>                                                \
  CAFFE2_CUDA_EXPORT void Axpby<TCoeff, TData, CUDAContext>( \
      const int n,                                           \
      const TCoeff a,                                        \
      const TData* x,                                        \
      const TCoeff b,                                        \
      TData* y,                                              \
      CUDAContext* context) {                                \
    AxpbyCUDAKernel<TCoeff, TData>                           \
        <<<CAFFE_GET_BLOCKS(n),                              \
           CAFFE_CUDA_NUM_THREADS,                           \
           0,                                                \
           context->cuda_stream()>>>(n, a, x, b, y);         \
  }                                                          \
  template <>                                                \
  CAFFE2_CUDA_EXPORT void Axpby<TCoeff, TData, CUDAContext>( \
      const int n,                                           \
      const TCoeff* a,                                       \
      const TData* x,                                        \
      const TCoeff* b,                                       \
      TData* y,                                              \
      CUDAContext* context) {                                \
    AxpbyCUDAKernel<TCoeff, TData>                           \
        <<<CAFFE_GET_BLOCKS(n),                              \
           CAFFE_CUDA_NUM_THREADS,                           \
           0,                                                \
           context->cuda_stream()>>>(n, a, x, b, y);         \
  }
CAFFE2_SPECIALIZED_CUDA_AXPBY(float, float)
CAFFE2_SPECIALIZED_CUDA_AXPBY(float, at::Half)
#undef CAFFE2_SPECIALIZED_CUDA_AXPBY

namespace {

template <typename T>
__global__ void Im2ColNCHWCUDAKernel(
    const int n,
    const int input_h,
    const int input_w,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w,
    const T* img_data,
    T* col_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int w_out = index % output_w;
    const int h_index = index / output_w;
    const int h_out = h_index % output_h;
    const int channel_in = h_index / output_h;
    const int channel_out = channel_in * kernel_h * kernel_w;
    const int h_in = h_out * stride_h - pad_t;
    const int w_in = w_out * stride_w - pad_l;
    const int output_size = output_h * output_w;
    T* col_data_ptr =
        col_data + (channel_out * output_h + h_out) * output_w + w_out;
    const T* img_data_ptr =
        img_data + (channel_in * input_h + h_in) * input_w + w_in;
    int dh = 0;
    for (int i = 0; i < kernel_h; ++i) {
      int dw = 0;
      for (int j = 0; j < kernel_w; ++j) {
        const int h = h_in + dh;
        const int w = w_in + dw;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
        *col_data_ptr = utils::IsAGeZeroAndALtB(h, input_h) &&
                utils::IsAGeZeroAndALtB(w, input_w)
            ? __ldg(img_data_ptr + dh * input_w + dw)
            : 0;
#else
        *col_data_ptr = utils::IsAGeZeroAndALtB(h, input_h) &&
                utils::IsAGeZeroAndALtB(w, input_w)
            ? img_data_ptr[dh * input_w + dw]
            : 0;
#endif
        col_data_ptr += output_size;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}

template <typename T>
__global__ void Im2ColNHWCCUDAKernel(
    const int n,
    const int input_h,
    const int input_w,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_w,
    const int channels,
    const T* img_data,
    T* col_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int channel_in = index % channels;
    const int w_out = index / channels % output_w;
    const int h_out = index / channels / output_w;
    const int h_in = h_out * stride_h - pad_t;
    const int w_in = w_out * stride_w - pad_l;
    T* col_data_ptr = col_data +
        (h_out * output_w + w_out) * channels * kernel_h * kernel_w +
        channel_in;
    int dh = 0;
    for (int i = 0; i < kernel_h; ++i) {
      int dw = 0;
      for (int j = 0; j < kernel_w; ++j) {
        const int h = h_in + dh;
        const int w = w_in + dw;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
        *col_data_ptr = utils::IsAGeZeroAndALtB(h, input_h) &&
                utils::IsAGeZeroAndALtB(w, input_w)
            ? __ldg(img_data + (h * input_w + w) * channels + channel_in)
            : 0;
#else
        *col_data_ptr = utils::IsAGeZeroAndALtB(h, input_h) &&
                utils::IsAGeZeroAndALtB(w, input_w)
            ? img_data[(h * input_w + w) * channels + channel_in]
            : 0;
#endif
        col_data_ptr += channels;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}

template <typename T>
__global__ void Col2ImNCHWCUDAKernel(
    const int n,
    const int input_h,
    const int input_w,
    const int patch_h,
    const int patch_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w,
    const T* col_data,
    T* img_data) {
  const int dpatch_h = dilation_h * (patch_h - 1) + 1;
  const int dpatch_w = dilation_w * (patch_w - 1) + 1;

  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    const int w = index % input_w + pad_l;
    const int h = index / input_w % input_h + pad_t;
    const int c = index / (input_h * input_w);

    // compute the start and end of the output
    const int w_col_start = (w < dpatch_w) ? 0 : (w - dpatch_w) / stride_w + 1;
    const int w_col_end = min(w / stride_w + 1, output_w);
    const int h_col_start = (h < dpatch_h) ? 0 : (h - dpatch_h) / stride_h + 1;
    const int h_col_end = min(h / stride_h + 1, output_h);

    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = (h - h_col * stride_h);
        int w_k = (w - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          const int col_data_index =
              (((c * patch_h + h_k) * patch_w + w_k) * output_h + h_col) *
                  output_w +
              w_col;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
          val += __ldg(col_data + col_data_index);
#else
          val += col_data[col_data_index];
#endif
        }
      }
    }
    img_data[index] = val;
  }
}

template <typename T>
__global__ void Col2ImNHWCCUDAKernel(
    const int n,
    const int input_w,
    const int channels,
    const int patch_h,
    const int patch_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w,
    const T* col_data,
    T* img_data) {
  const int dpatch_h = dilation_h * (patch_h - 1) + 1;
  const int dpatch_w = dilation_w * (patch_w - 1) + 1;

  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    const int c = index % channels;
    const int w = index / channels % input_w + pad_l;
    const int h = index / channels / input_w + pad_t;
    // compute the start and end of the output
    const int w_col_start = (w < dpatch_w) ? 0 : (w - dpatch_w) / stride_w + 1;
    const int w_col_end = min(w / stride_w + 1, output_w);
    const int h_col_start = (h < dpatch_h) ? 0 : (h - dpatch_h) / stride_h + 1;
    const int h_col_end = min(h / stride_h + 1, output_h);
    const int channels_col = patch_h * patch_w * channels;

    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = h - h_col * stride_h;
        int w_k = w - w_col * stride_w;
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          const int c_col = (h_k * patch_w + w_k) * channels + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
          val += __ldg(
              col_data + (h_col * output_w + w_col) * channels_col + c_col);
#else
          val += col_data[(h_col * output_w + w_col) * channels_col + c_col];
#endif
        }
      }
    }
    img_data[index] = val;
  }
}

template <typename T, int N, bool kCol2Im>
__global__ void Im2ColNdNCHWCUDAKernel(
    const int outer_size,
    const int inner_size,
    const int kernel_size,
    SimpleArray<int, N + 1> img_shape,
    SimpleArray<int, N + 1> col_shape,
    SimpleArray<int, N> kernel_shape,
    SimpleArray<int, N> stride,
    SimpleArray<int, N> dilation,
    SimpleArray<int, N> pad,
    const T* X_data,
    T* Y_data) {
  int d_offset[N];
  int d_iter[N];
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    int offset_i = i;
#pragma unroll
    for (int d_i = N - 1; d_i >= 0; --d_i) {
      d_offset[d_i] = offset_i % kernel_shape.data[d_i];
      offset_i /= kernel_shape.data[d_i];
    }
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int offset_j = j;
#pragma unroll
      for (int d_i = N - 1; d_i >= 0; --d_i) {
        d_iter[d_i] = offset_j % col_shape.data[d_i + 1];
        offset_j /= col_shape.data[d_i + 1];
      }
      const int col_index = i * inner_size + j;
      int img_index = i / kernel_size;
      bool is_padding = false;
#pragma unroll
      for (int d_i = 0; d_i < N; ++d_i) {
        const int d_img = d_iter[d_i] * stride.data[d_i] - pad.data[d_i] +
            d_offset[d_i] * dilation.data[d_i];
        is_padding |= !utils::IsAGeZeroAndALtB(d_img, img_shape.data[d_i + 1]);
        img_index = img_index * img_shape.data[d_i + 1] + d_img;
      }
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : __ldg(X_data + img_index);
      } else if (!is_padding) {
        atomicAdd(Y_data + img_index, __ldg(X_data + col_index));
      }
#else
      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : X_data[img_index];
      } else if (!is_padding) {
        atomicAdd(Y_data + img_index, X_data[col_index]);
      }
#endif
    }
  }
}

template <typename T, int N>
CAFFE2_CUDA_EXPORT void Im2ColNdNCHWCUDAImpl(
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    CUDAContext* context) {
  const int outer_size = col_shape[0];
  const int inner_size = col_size / outer_size;
  const int kernel_size = std::accumulate(
      kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
  SimpleArray<int, N + 1> img_shape_array;
  SimpleArray<int, N + 1> col_shape_array;
  SimpleArray<int, N> kernel_shape_array;
  SimpleArray<int, N> stride_array;
  SimpleArray<int, N> dilation_array;
  SimpleArray<int, N> pad_array;
  std::memcpy(img_shape_array.data, img_shape, (N + 1) * sizeof(int));
  std::memcpy(col_shape_array.data, col_shape, (N + 1) * sizeof(int));
  std::memcpy(kernel_shape_array.data, kernel_shape, N * sizeof(int));
  std::memcpy(stride_array.data, stride, N * sizeof(int));
  std::memcpy(dilation_array.data, dilation, N * sizeof(int));
  std::memcpy(pad_array.data, pad, N * sizeof(int));
  Im2ColNdNCHWCUDAKernel<T, N, false>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size,
          inner_size,
          kernel_size,
          img_shape_array,
          col_shape_array,
          kernel_shape_array,
          stride_array,
          dilation_array,
          pad_array,
          img_data,
          col_data);
}

template <typename T, int N>
CAFFE2_CUDA_EXPORT void Col2ImNdNCHWCUDAImpl(
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    CUDAContext* context) {
  const int outer_size = col_shape[0];
  const int inner_size = col_size / outer_size;
  const int kernel_size = std::accumulate(
      kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
  SimpleArray<int, N + 1> img_shape_array;
  SimpleArray<int, N + 1> col_shape_array;
  SimpleArray<int, N> kernel_shape_array;
  SimpleArray<int, N> stride_array;
  SimpleArray<int, N> dilation_array;
  SimpleArray<int, N> pad_array;
  std::memcpy(img_shape_array.data, img_shape, (N + 1) * sizeof(int));
  std::memcpy(col_shape_array.data, col_shape, (N + 1) * sizeof(int));
  std::memcpy(kernel_shape_array.data, kernel_shape, N * sizeof(int));
  std::memcpy(stride_array.data, stride, N * sizeof(int));
  std::memcpy(dilation_array.data, dilation, N * sizeof(int));
  std::memcpy(pad_array.data, pad, N * sizeof(int));
  Set<T, CUDAContext>(img_size, 0, img_data, context);
  Im2ColNdNCHWCUDAKernel<T, N, true>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size,
          inner_size,
          kernel_size,
          img_shape_array,
          col_shape_array,
          kernel_shape_array,
          stride_array,
          dilation_array,
          pad_array,
          col_data,
          img_data);
}

} // namespace

template <>
CAFFE2_CUDA_EXPORT void Im2Col<float, CUDAContext, StorageOrder::NCHW>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* img_data,
    float* col_data,
    CUDAContext* context,
    const int /* groups */) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = channels * output_h * output_w;
  Im2ColNCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          num_kernels,
          height,
          width,
          kernel_h,
          kernel_w,
          dilation_h,
          dilation_w,
          pad_t,
          pad_l,
          stride_h,
          stride_w,
          output_h,
          output_w,
          img_data,
          col_data);
}

template <>
CAFFE2_CUDA_EXPORT void Im2Col<float, CUDAContext, StorageOrder::NHWC>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* img_data,
    float* col_data,
    CUDAContext* context,
    const int groups) {
  CAFFE_ENFORCE_EQ(groups, 1, "groups must be 1 for GPU NHWC Im2Col");

  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = output_h * output_w * channels;
  Im2ColNHWCCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          num_kernels,
          height,
          width,
          kernel_h,
          kernel_w,
          dilation_h,
          dilation_w,
          pad_t,
          pad_l,
          stride_h,
          stride_w,
          output_w,
          channels,
          img_data,
          col_data);
}

template <>
CAFFE2_CUDA_EXPORT void Col2Im<float, CUDAContext, StorageOrder::NCHW>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* col_data,
    float* img_data,
    CUDAContext* context,
    const int /* groups */) {
  // In NCHW, the number of groups doesn't affect Col2Im.
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = channels * height * width;
  Col2ImNCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          num_kernels,
          height,
          width,
          kernel_h,
          kernel_w,
          dilation_h,
          dilation_w,
          pad_t,
          pad_l,
          stride_h,
          stride_w,
          output_h,
          output_w,
          col_data,
          img_data);
}

template <>
CAFFE2_CUDA_EXPORT void Col2Im<float, CUDAContext, StorageOrder::NHWC>(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* col_data,
    float* img_data,
    CUDAContext* context,
    const int groups) {
  CAFFE_ENFORCE_EQ(groups, 1, "groups must be 1 for GPU NHWC Col2Im");

  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int num_kernels = height * width * channels;
  Col2ImNHWCCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          num_kernels,
          width,
          channels,
          kernel_h,
          kernel_w,
          dilation_h,
          dilation_w,
          pad_t,
          pad_l,
          stride_h,
          stride_w,
          output_h,
          output_w,
          col_data,
          img_data);
}

template <>
CAFFE2_CUDA_EXPORT void Im2ColNd<float, CUDAContext, StorageOrder::NCHW>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    CUDAContext* context,
    const int /* groups */) {
  // In NCHW, the number of groups doesn't affect Im2Col.
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      N,
      Im2ColNdNCHWCUDAImpl,
      float,
      img_size,
      col_size,
      img_shape,
      col_shape,
      kernel_shape,
      stride,
      dilation,
      pad,
      img_data,
      col_data,
      context);
}

template <>
CAFFE2_CUDA_EXPORT void Im2ColNd<float, CUDAContext, StorageOrder::NHWC>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    CUDAContext* context,
    const int groups) {
  CAFFE_NOT_IMPLEMENTED;
}

template <>
CAFFE2_CUDA_EXPORT void Col2ImNd<float, CUDAContext, StorageOrder::NCHW>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    CUDAContext* context,
    int /* groups */) {
  // In NCHW, the number of groups doesn't affect Col2Im.
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      N,
      Col2ImNdNCHWCUDAImpl,
      float,
      img_size,
      col_size,
      img_shape,
      col_shape,
      kernel_shape,
      stride,
      dilation,
      pad,
      col_data,
      img_data,
      context);
}

template <>
CAFFE2_CUDA_EXPORT void Col2ImNd<float, CUDAContext, StorageOrder::NHWC>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    CUDAContext* context,
    int groups) {
  CAFFE_NOT_IMPLEMENTED;
}

template <>
CAFFE2_CUDA_EXPORT void CopyMatrix<CUDAContext>(
    const size_t itemsize,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb,
    CUDAContext* context,
    TypeMeta::Copy copy) {
  CAFFE_ENFORCE(!copy, "Copy constructor is not supported in CUDA context");
  cudaMemcpy2DAsync(
      B,
      ldb * itemsize,
      A,
      lda * itemsize,
      N * itemsize,
      M,
      cudaMemcpyDeviceToDevice,
      context->cuda_stream());
}

#define CAFFE2_SPECIALIZED_CUDA_COPY_MATRIX(T) \
  template <>                                  \
  void CopyMatrix<T, CUDAContext>(             \
      const int M,                             \
      const int N,                             \
      const T* A,                              \
      const int lda,                           \
      T* B,                                    \
      const int ldb,                           \
      CUDAContext* context) {                  \
    if (M == 0 || N == 0) {                    \
      return;                                  \
    }                                          \
    cudaMemcpy2DAsync(                         \
        B,                                     \
        sizeof(T) * ldb,                       \
        A,                                     \
        sizeof(T) * lda,                       \
        sizeof(T) * N,                         \
        M,                                     \
        cudaMemcpyDeviceToDevice,              \
        context->cuda_stream());               \
  }
CAFFE2_SPECIALIZED_CUDA_COPY_MATRIX(float)
CAFFE2_SPECIALIZED_CUDA_COPY_MATRIX(double)
CAFFE2_SPECIALIZED_CUDA_COPY_MATRIX(int)
CAFFE2_SPECIALIZED_CUDA_COPY_MATRIX(int64_t)
#undef CAFFE2_SPECIALIZED_CUDA_COPY_MATRIX

template <>
CAFFE2_CUDA_EXPORT void CopyVector<float, CUDAContext>(
    const int N,
    const float* src,
    float* dst,
    CUDAContext* context) {
  if (src != dst && N > 0) {
    cudaMemcpyAsync(
        dst,
        src,
        sizeof(float) * N,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
  }
}

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T, class Reducer>
__global__ void RowwiseReduceKernel(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      val = reducer(X[i * cols + j], val);
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val * alpha;
    }
    __syncthreads();
  }
}

template <typename T, class Reducer>
__global__ void ColwiseReduceKernel(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < cols; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
      val = reducer(X[j * cols + i], val);
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val * alpha;
    }
    __syncthreads();
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_ROWWISE_MAX(T)                            \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void RowwiseMax<T, CUDAContext>(                     \
      const int N, const int D, const T* x, T* y, CUDAContext* context) { \
    RowwiseReduceKernel<<<                                                \
        std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),                            \
        CAFFE_CUDA_NUM_THREADS,                                           \
        0,                                                                \
        context->cuda_stream()>>>(                                        \
        N, D, cub::Max(), std::numeric_limits<T>::lowest(), T(1), x, y);  \
  }
CAFFE2_SPECIALIZED_CUDA_ROWWISE_MAX(float)
#undef CAFFE2_SPECIALIZED_CUDA_ROWWISE_MAX

#define CAFFE2_SPECIALIZED_CUDA_COLWISE_MAX(T)                            \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void ColwiseMax<T, CUDAContext>(                     \
      const int N, const int D, const T* x, T* y, CUDAContext* context) { \
    ColwiseReduceKernel<<<                                                \
        std::min(D, CAFFE_MAXIMUM_NUM_BLOCKS),                            \
        CAFFE_CUDA_NUM_THREADS,                                           \
        0,                                                                \
        context->cuda_stream()>>>(                                        \
        N, D, cub::Max(), std::numeric_limits<T>::lowest(), T(1), x, y);  \
  }
CAFFE2_SPECIALIZED_CUDA_COLWISE_MAX(float)
#undef CAFFE2_SPECIALIZED_CUDA_COLWISE_MAX

namespace {
__global__ void
maximum_kernel(const int N, const float alpha, const float* x, float* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = fmaxf(x[i], alpha);
  }
}
} // namespace

template <>
CAFFE2_CUDA_EXPORT void Maximum(
    const int N,
    const float alpha,
    const float* x,
    float* y,
    CUDAContext* context) {
  maximum_kernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, alpha, x, y);
}

namespace {

template <typename T, class Reducer, int D>
__global__ void ReduceTensorCUDAKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<FIXED_DIVISOR, D> Y_dims,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T val = init;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int X_index = 0;
      int Y_index = i * inner_size + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], Y_index, &Y_index, &r);
        X_index += r * X_strides.data[d];
      }
#if __CUDA_ARCH__ >= 350
      val = reducer(val, __ldg(X + X_index));
#else
      val = reducer(val, X[X_index]);
#endif
    }
    val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
    if (threadIdx.x == 0) {
      Y[i] = val * alpha;
    }
    __syncthreads();
  }
}

template <typename T, class Reducer, int D>
CAFFE2_CUDA_EXPORT void ReduceTensorCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const Reducer& reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FIXED_DIVISOR, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FIXED_DIVISOR(dims[axes[i]]);
  }
  ReduceTensorCUDAKernel<T, Reducer, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size,
          inner_size,
          X_strides,
          Y_dims,
          reducer,
          init,
          alpha,
          X,
          Y);
}

template <typename T, class Reducer>
CAFFE2_CUDA_EXPORT void ReduceTensorCUDA(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const Reducer& reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y,
    CUDAContext* context) {
  CAFFE_ENFORCE_LE(num_axes, num_dims);
  std::vector<int> Y_dims_vector(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    Y_dims_vector[axes[i]] = 1;
  }
  const int* X_dims = dims;
  const int* Y_dims = Y_dims_vector.data();
  const int X_size =
      std::accumulate(X_dims, X_dims + num_dims, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + num_dims, 1, std::multiplies<int>());
  if (X_size == 0) {
    Set<T, CUDAContext>(Y_size, alpha * init, Y, context);
    return;
  }
  if (alpha == T(0)) {
    Set<T, CUDAContext>(Y_size, T(0), Y, context);
    return;
  }
  if (std::equal(X_dims, X_dims + num_dims, Y_dims)) {
    Scale<T, T, CUDAContext>(X_size, alpha, X, Y, context);
    return;
  }
  int rows;
  int cols;
  if (utils::IsRowwiseReduce(num_dims, X_dims, Y_dims, &rows, &cols)) {
    RowwiseReduceKernel<T>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(rows, cols, reducer, init, alpha, X, Y);
    return;
  }
  if (utils::IsColwiseReduce(num_dims, X_dims, Y_dims, &rows, &cols)) {
    ColwiseReduceKernel<T>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(rows, cols, reducer, init, alpha, X, Y);
    return;
  }
  std::vector<int> transpose_axes(num_dims);
  utils::ComputeTransposeAxesForReduceOp(
      num_dims, num_axes, axes, transpose_axes.data());
  const int outer_size = Y_size;
  const int inner_size = X_size / Y_size;
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
      num_dims,
      ReduceTensorCUDAImpl,
      T,
      Reducer,
      outer_size,
      inner_size,
      dims,
      transpose_axes.data(),
      reducer,
      init,
      alpha,
      X,
      Y,
      context);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(T)        \
  template <>                                        \
  CAFFE2_CUDA_EXPORT void ReduceMin<T, CUDAContext>( \
      const int num_dims,                            \
      const int* dims,                               \
      const int num_axes,                            \
      const int* axes,                               \
      const T alpha,                                 \
      const T* X,                                    \
      T* Y,                                          \
      CUDAContext* context) {                        \
    ReduceTensorCUDA(                                \
        num_dims,                                    \
        dims,                                        \
        num_axes,                                    \
        axes,                                        \
        cub::Min(),                                  \
        std::numeric_limits<T>::max(),               \
        alpha,                                       \
        X,                                           \
        Y,                                           \
        context);                                    \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(float)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN(double)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MIN

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(T)        \
  template <>                                        \
  CAFFE2_CUDA_EXPORT void ReduceMax<T, CUDAContext>( \
      const int num_dims,                            \
      const int* dims,                               \
      const int num_axes,                            \
      const int* axes,                               \
      const T alpha,                                 \
      const T* X,                                    \
      T* Y,                                          \
      CUDAContext* context) {                        \
    ReduceTensorCUDA(                                \
        num_dims,                                    \
        dims,                                        \
        num_axes,                                    \
        axes,                                        \
        cub::Max(),                                  \
        std::numeric_limits<T>::lowest(),            \
        alpha,                                       \
        X,                                           \
        Y,                                           \
        context);                                    \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(float)
CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX(double)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MAX

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(T)        \
  template <>                                        \
  CAFFE2_CUDA_EXPORT void ReduceSum<T, CUDAContext>( \
      const int num_dims,                            \
      const int* dims,                               \
      const int num_axes,                            \
      const int* axes,                               \
      const T alpha,                                 \
      const T* X,                                    \
      T* Y,                                          \
      CUDAContext* context) {                        \
    ReduceTensorCUDA(                                \
        num_dims,                                    \
        dims,                                        \
        num_axes,                                    \
        axes,                                        \
        cub::Sum(),                                  \
        T(0),                                        \
        alpha,                                       \
        X,                                           \
        Y,                                           \
        context);                                    \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(float)
CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM(double)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_SUM

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN(T)        \
  template <>                                         \
  CAFFE2_CUDA_EXPORT void ReduceMean<T, CUDAContext>( \
      const int num_dims,                             \
      const int* dims,                                \
      const int num_axes,                             \
      const int* axes,                                \
      const T alpha,                                  \
      const T* X,                                     \
      T* Y,                                           \
      CUDAContext* context) {                         \
    int scale = 1;                                    \
    for (int i = 0; i < num_axes; ++i) {              \
      scale *= dims[axes[i]];                         \
    }                                                 \
    ReduceTensorCUDA(                                 \
        num_dims,                                     \
        dims,                                         \
        num_axes,                                     \
        axes,                                         \
        cub::Sum(),                                   \
        T(0),                                         \
        alpha / static_cast<T>(scale),                \
        X,                                            \
        Y,                                            \
        context);                                     \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN(float)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN

namespace {

template <typename T, int D>
__global__ void BroadcastCUDAKernel(
    const int Y_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<FIXED_DIVISOR, D> Y_dims,
    const T alpha,
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(Y_index, Y_size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[i], Y_index_val, &Y_index_val, &d);
      X_index += d * X_strides.data[i];
    }
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    Y[Y_index] = __ldg(X + X_index) * alpha;
#else
    Y[Y_index] = X[X_index] * alpha;
#endif
  }
}

template <typename T, int D>
CAFFE2_CUDA_EXPORT void BroadcastCUDAImpl(
    const int X_ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides_array;
  SimpleArray<FIXED_DIVISOR, D> Y_dims_array;
  const int d = D - X_ndim;
  std::fill(X_strides_array.data, X_strides_array.data + d, 0);
  int cur_stride = 1;
  for (int i = D - 1; i >= d; --i) {
    CAFFE_ENFORCE(X_dims[i - d] == 1 || X_dims[i - d] == Y_dims[i]);
    X_strides_array.data[i] = X_dims[i - d] == 1 ? 0 : cur_stride;
    cur_stride *= X_dims[i - d];
  }
  for (int i = 0; i < D; ++i) {
    if (Y_dims[i] == 0) {
      return;
    }
    Y_dims_array.data[i] = FIXED_DIVISOR(Y_dims[i]);
  }
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + D, 1, std::multiplies<int>());
  BroadcastCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(Y_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          Y_size, X_strides_array, Y_dims_array, alpha, X, Y);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_BROADCAST(T)         \
  template <>                                        \
  CAFFE2_CUDA_EXPORT void Broadcast<T, CUDAContext>( \
      const int X_ndim,                              \
      const int* X_dims,                             \
      const int Y_ndim,                              \
      const int* Y_dims,                             \
      const T alpha,                                 \
      const T* X,                                    \
      T* Y,                                          \
      CUDAContext* context) {                        \
    CAFFE_ENFORCE_LE(X_ndim, Y_ndim);                \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(          \
        Y_ndim,                                      \
        BroadcastCUDAImpl,                           \
        T,                                           \
        X_ndim,                                      \
        X_dims,                                      \
        Y_dims,                                      \
        alpha,                                       \
        X,                                           \
        Y,                                           \
        context);                                    \
  }
CAFFE2_SPECIALIZED_CUDA_BROADCAST(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_BROADCAST(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_BROADCAST(float)
CAFFE2_SPECIALIZED_CUDA_BROADCAST(double)
#undef CAFFE2_SPECIALIZED_CUDA_BROADCAST

namespace {

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(cols);
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      const int X_index = i * cols + j;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const T mu = m_val * scale;
      mean[i] = mu;
      variance[i] = v_val * scale - mu * mu;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void ColwiseMomentsCUDAKernel(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(rows);
  for (int i = blockIdx.x; i < cols; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
      const int X_index = j * cols + i;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const T mu = m_val * scale;
      mean[i] = mu;
      variance[i] = v_val * scale - mu * mu;
    }
    __syncthreads();
  }
}

template <typename T, int D>
__global__ void MomentsCUDAKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<FIXED_DIVISOR, D> Y_dims,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(inner_size);
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int X_index = 0;
      int Y_index = i * inner_size + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], Y_index, &Y_index, &r);
        X_index += r * X_strides.data[d];
      }
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const T mu = m_val * scale;
      mean[i] = mu;
      variance[i] = v_val * scale - mu * mu;
    }
    __syncthreads();
  }
}

template <typename T, int D>
CAFFE2_CUDA_EXPORT void MomentsCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FIXED_DIVISOR, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FIXED_DIVISOR(dims[axes[i]]);
  }
  MomentsCUDAKernel<T, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          outer_size, inner_size, X_strides, Y_dims, X, mean, variance);
}

template <typename T>
CAFFE2_CUDA_EXPORT void MomentsCUDA(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    CUDAContext* context) {
  CAFFE_ENFORCE_LE(num_axes, num_dims);
  std::vector<int> Y_dims_vector(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    Y_dims_vector[axes[i]] = 1;
  }
  const int* X_dims = dims;
  const int* Y_dims = Y_dims_vector.data();
  const int X_size =
      std::accumulate(X_dims, X_dims + num_dims, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + num_dims, 1, std::multiplies<int>());
  if (X_size == 0) {
    Set<T, CUDAContext>(Y_size, T(0), mean, context);
    Set<T, CUDAContext>(Y_size, T(0), variance, context);
    return;
  }
  if (std::equal(X_dims, X_dims + num_dims, Y_dims)) {
    cudaMemcpyAsync(
        mean,
        X,
        sizeof(T) * X_size,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
    Set<T, CUDAContext>(Y_size, T(0), variance, context);
    return;
  }
  int rows;
  int cols;
  if (utils::IsRowwiseReduce(num_dims, X_dims, Y_dims, &rows, &cols)) {
    RowwiseMomentsCUDAKernel<T>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(rows, cols, X, mean, variance);
    return;
  }
  if (utils::IsColwiseReduce(num_dims, X_dims, Y_dims, &rows, &cols)) {
    ColwiseMomentsCUDAKernel<T>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(rows, cols, X, mean, variance);
    return;
  }
  std::vector<int> transpose_axes(num_dims);
  utils::ComputeTransposeAxesForReduceOp(
      num_dims, num_axes, axes, transpose_axes.data());
  const int pivot = num_dims - num_axes;
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= dims[transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < num_dims; ++i) {
    inner_size *= dims[transpose_axes[i]];
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      num_dims,
      MomentsCUDAImpl,
      T,
      outer_size,
      inner_size,
      dims,
      transpose_axes.data(),
      X,
      mean,
      variance,
      context);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_MOMENTS(T)                           \
  template <>                                                        \
  CAFFE2_CUDA_EXPORT void Moments<T, CUDAContext>(                   \
      const int num_dims,                                            \
      const int* dims,                                               \
      const int num_axes,                                            \
      const int* axes,                                               \
      const T* X,                                                    \
      T* mean,                                                       \
      T* variance,                                                   \
      CUDAContext* context) {                                        \
    MomentsCUDA<T>(                                                  \
        num_dims, dims, num_axes, axes, X, mean, variance, context); \
  }
CAFFE2_SPECIALIZED_CUDA_MOMENTS(float)
#undef CAFFE2_SPECIALIZED_CUDA_MOMENTS

namespace {

template <typename T>
__global__ void
InvStdCUDAKernel(const int N, const T epsilon, const T* var, T* inv_std);

#define DELEGATE_INV_STD_KERNEL_FUNCTION(T, Func)               \
  template <>                                                   \
  __global__ void InvStdCUDAKernel<T>(                          \
      const int N, const T epsilon, const T* var, T* inv_std) { \
    CUDA_1D_KERNEL_LOOP(i, N) {                                 \
      inv_std[i] = Func(var[i] + epsilon);                      \
    }                                                           \
  }
DELEGATE_INV_STD_KERNEL_FUNCTION(float, rsqrtf)
#undef DELEGATE_INV_STD_KERNEL_FUNCTION

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_INV_STD(T)                      \
  template <>                                                   \
  CAFFE2_CUDA_EXPORT void InvStd<T, CUDAContext>(               \
      const int N,                                              \
      const T epsilon,                                          \
      const T* var,                                             \
      T* inv_std,                                               \
      CUDAContext* context) {                                   \
    InvStdCUDAKernel<T>                                         \
        <<<CAFFE_GET_BLOCKS(N),                                 \
           CAFFE_CUDA_NUM_THREADS,                              \
           0,                                                   \
           context->cuda_stream()>>>(N, epsilon, var, inv_std); \
  }
CAFFE2_SPECIALIZED_CUDA_INV_STD(float)
#undef CAFFE2_SPECIALIZED_CUDA_INV_STD

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

// Splits the original matrix into submatrices with size 32 * 32.
// Each block transposes one submatrix by loading it into shared memory.
// Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename T>
__global__ void BatchTranspose2DCUDAKernel(
    const int N,
    const int H,
    const int W,
    const T* X,
    T* Y) {
  __shared__ T tile[kTileDim][kTileDim + 1];
  const int h = (H + kTileDim - 1) / kTileDim;
  const int w = (W + kTileDim - 1) / kTileDim;
  const int outer_size = N * h * w;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    const int n = i / (h * w);
    const int k = i % (h * w);
    const int r = k / w;
    const int c = k % w;
    const int offset = n * H * W;
    int x = c * kTileDim + threadIdx.x;
    int y = r * kTileDim + threadIdx.y;
    if (x < W) {
      for (int j = 0; j < kTileDim && y + j < H; j += kBlockRows) {
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
        tile[threadIdx.y + j][threadIdx.x] =
            __ldg(X + offset + (y + j) * W + x);
#else
        tile[threadIdx.y + j][threadIdx.x] = X[offset + (y + j) * W + x];
#endif
      }
    }
    __syncthreads();
    x = r * kTileDim + threadIdx.x;
    y = c * kTileDim + threadIdx.y;
    if (x < H) {
      for (int j = 0; j < kTileDim && y + j < W; j += kBlockRows) {
        Y[offset + (y + j) * H + x] = tile[threadIdx.x][threadIdx.y + j];
      }
    }
    __syncthreads();
  }
}

template <typename T, int D>
__global__ void TransposeCUDAKernel(
    const int size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<FIXED_DIVISOR, D> Y_dims,
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(Y_index, size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[i], Y_index_val, &Y_index_val, &d);
      X_index += d * X_strides.data[i];
    }
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    Y[Y_index] = __ldg(X + X_index);
#else
    Y[Y_index] = X[X_index];
#endif
  }
}

template <typename T, int D>
CAFFE2_CUDA_EXPORT void TransposeCUDAImpl(
    const int* dims,
    const int* axes,
    const T* X,
    T* Y,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FIXED_DIVISOR, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  int size = 1;
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FIXED_DIVISOR(dims[axes[i]]);
    size *= dims[i];
  }
  TransposeCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, X_strides, Y_dims, X, Y);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(T)                                 \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Transpose<T, CUDAContext>(                         \
      const int ndim,                                                        \
      const int* dims,                                                       \
      const int* axes,                                                       \
      const T* X,                                                            \
      T* Y,                                                                  \
      CUDAContext* context) {                                                \
    if (utils::IsIdentityPermutation(ndim, axes)) {                          \
      const int size =                                                       \
          std::accumulate(dims, dims + ndim, 1, std::multiplies<int>());     \
      context->template CopySameDevice<T>(size, X, Y);                       \
      return;                                                                \
    }                                                                        \
    if (utils::IsBatchTranspose2D(ndim, axes)) {                             \
      const int N =                                                          \
          std::accumulate(dims, dims + ndim - 2, 1, std::multiplies<int>()); \
      const int H = dims[ndim - 2];                                          \
      const int W = dims[ndim - 1];                                          \
      const int h = (H + kTileDim - 1) / kTileDim;                           \
      const int w = (W + kTileDim - 1) / kTileDim;                           \
      const int outer_size = N * h * w;                                      \
      const dim3 dim_block(kTileDim, kBlockRows, 1);                         \
      BatchTranspose2DCUDAKernel<T>                                          \
          <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),                 \
             dim_block,                                                      \
             0,                                                              \
             context->cuda_stream()>>>(N, H, W, X, Y);                       \
      return;                                                                \
    }                                                                        \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(                                  \
        ndim, TransposeCUDAImpl, T, dims, axes, X, Y, context);              \
  }
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(float)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(double)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(int)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(int64_t)
#undef CAFFE2_SPECIALIZED_CUDA_TRANSPOSE

namespace {

template <typename T, StorageOrder kOrder>
__global__ void AffineChannelCUDAKernel(
    const int size,
    const int C,
    const int HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int c = kOrder == StorageOrder::NCHW ? i / HxW % C : i % C;
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    Y[i] = __ldg(scale + c) * __ldg(X + i) + __ldg(bias + c);
#else
    Y[i] = scale[c] * X[i] + bias[c];
#endif
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(T, kOrder)              \
  template <>                                                          \
  CAFFE2_CUDA_EXPORT void AffineChannel<T, CUDAContext, kOrder>(       \
      const int N,                                                     \
      const int C,                                                     \
      const int HxW,                                                   \
      const T* X,                                                      \
      const T* scale,                                                  \
      const T* bias,                                                   \
      T* Y,                                                            \
      CUDAContext* context) {                                          \
    const int size = N * C * HxW;                                      \
    AffineChannelCUDAKernel<T, kOrder>                                 \
        <<<CAFFE_GET_BLOCKS(size),                                     \
           CAFFE_CUDA_NUM_THREADS,                                     \
           0,                                                          \
           context->cuda_stream()>>>(size, C, HxW, X, scale, bias, Y); \
  }
CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL

#define CAFFE2_SPECIALIZED_CUDA_NCHW2NHWC(T)               \
  template <>                                              \
  CAFFE2_CUDA_EXPORT void NCHW2NHWC<T, CUDAContext>(       \
      const int N,                                         \
      const int C,                                         \
      const int HxW,                                       \
      const T* X,                                          \
      T* Y,                                                \
      CUDAContext* context) {                              \
    const int h = (C + kTileDim - 1) / kTileDim;           \
    const int w = (HxW + kTileDim - 1) / kTileDim;         \
    const int outer_size = N * h * w;                      \
    const dim3 dim_block(kTileDim, kBlockRows, 1);         \
    BatchTranspose2DCUDAKernel<T>                          \
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS), \
           dim_block,                                      \
           0,                                              \
           context->cuda_stream()>>>(N, C, HxW, X, Y);     \
  }
CAFFE2_SPECIALIZED_CUDA_NCHW2NHWC(float)
#undef CAFFE2_SPECIALIZED_CUDA_NCHW2NHWC

#define CAFFE2_SPECIALIZED_CUDA_NHWC2NCHW(T)               \
  template <>                                              \
  CAFFE2_CUDA_EXPORT void NHWC2NCHW<T, CUDAContext>(       \
      const int N,                                         \
      const int C,                                         \
      const int HxW,                                       \
      const T* X,                                          \
      T* Y,                                                \
      CUDAContext* context) {                              \
    const int h = (HxW + kTileDim - 1) / kTileDim;         \
    const int w = (C + kTileDim - 1) / kTileDim;           \
    const int outer_size = N * h * w;                      \
    const dim3 dim_block(kTileDim, kBlockRows, 1);         \
    BatchTranspose2DCUDAKernel<T>                          \
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS), \
           dim_block,                                      \
           0,                                              \
           context->cuda_stream()>>>(N, HxW, C, X, Y);     \
  }
CAFFE2_SPECIALIZED_CUDA_NHWC2NCHW(float)
#undef CAFFE2_SPECIALIZED_CUDA_NHWC2NCHW

} // namespace math
} // namespace caffe2
