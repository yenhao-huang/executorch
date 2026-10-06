/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TempMemoryAllocator.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <limits>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TempMemoryAllocator;
using torch::executor::testing::TensorFactory;

std::tuple<Tensor&, Tensor&> op_sort_values(
    const Tensor& input,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  TempMemoryAllocator allocator = TempMemoryAllocator();
  executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext context(
      nullptr, &allocator);
  return torch::executor::aten::sort_outf(
      context, input, dim, descending, values, indices);
}

class OpSortValuesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }

  // All references below come from torch.sort(..., stable=True).
  template <ScalarType DTYPE>
  void run_dim0_test() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> tf_long;

    Tensor input = tf.make({3, 2}, {1, 4, 3, 2, 5, 0});
    Tensor values = tf.zeros({3, 2});
    Tensor indices = tf_long.zeros({3, 2});

    op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);

    EXPECT_TENSOR_EQ(values, tf.make({3, 2}, {1, 0, 3, 2, 5, 4}));
    EXPECT_TENSOR_EQ(indices, tf_long.make({3, 2}, {0, 2, 1, 1, 2, 0}));
  }
};

TEST_F(OpSortValuesTest, SortAlongDim0) {
  // Bool is covered separately: these values all collapse to true.
#define TEST_ENTRY(ctype, dtype) run_dim0_test<ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY)
#undef TEST_ENTRY
}

TEST_F(OpSortValuesTest, SortBool) {
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.make({3, 2}, {true, true, false, true, true, false});
  Tensor values = tf.zeros({3, 2});
  Tensor indices = tf_long.zeros({3, 2});

  op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);
  EXPECT_TENSOR_EQ(
      values, tf.make({3, 2}, {false, false, true, true, true, true}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({3, 2}, {1, 2, 0, 0, 2, 1}));

  op_sort_values(input, /*dim=*/1, /*descending=*/true, values, indices);
  EXPECT_TENSOR_EQ(
      values, tf.make({3, 2}, {true, true, true, false, true, false}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({3, 2}, {0, 1, 1, 0, 0, 1}));
}

TEST_F(OpSortValuesTest, SortAlongDim1Descending) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.make({3, 2}, {1, 4, 3, 2, 5, 0});
  Tensor values = tf.zeros({3, 2});
  Tensor indices = tf_long.zeros({3, 2});

  op_sort_values(input, /*dim=*/1, /*descending=*/true, values, indices);

  EXPECT_TENSOR_EQ(values, tf.make({3, 2}, {4, 1, 3, 2, 5, 0}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({3, 2}, {1, 0, 0, 1, 0, 1}));
}

TEST_F(OpSortValuesTest, NegativeDimCountsFromTheEnd) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.make({3, 2}, {1, 4, 3, 2, 5, 0});
  Tensor values = tf.zeros({3, 2});
  Tensor indices = tf_long.zeros({3, 2});

  op_sort_values(input, /*dim=*/-1, /*descending=*/false, values, indices);

  EXPECT_TENSOR_EQ(values, tf.make({3, 2}, {1, 4, 2, 3, 0, 5}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({3, 2}, {0, 1, 1, 0, 1, 0}));
}

TEST_F(OpSortValuesTest, EqualValuesKeepTheirOrder) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.make({5}, {2, 1, 2, 1, 3});
  Tensor values = tf.zeros({5});
  Tensor indices = tf_long.zeros({5});

  op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);
  EXPECT_TENSOR_EQ(values, tf.make({5}, {1, 1, 2, 2, 3}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({5}, {1, 3, 0, 2, 4}));

  op_sort_values(input, /*dim=*/0, /*descending=*/true, values, indices);
  EXPECT_TENSOR_EQ(values, tf.make({5}, {3, 2, 2, 1, 1}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({5}, {4, 0, 2, 1, 3}));
}

TEST_F(OpSortValuesTest, NanSortsAboveEveryNumber) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor input = tf.make({4}, {1, nan, 0, -2});
  Tensor values = tf.zeros({4});
  Tensor indices = tf_long.zeros({4});

  op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);
  EXPECT_TENSOR_EQ(indices, tf_long.make({4}, {3, 2, 0, 1}));

  op_sort_values(input, /*dim=*/0, /*descending=*/true, values, indices);
  EXPECT_TENSOR_EQ(indices, tf_long.make({4}, {1, 0, 2, 3}));
}

TEST_F(OpSortValuesTest, SortsAcrossOuterAndInnerDims) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input =
      tf.make({3, 2, 2}, {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3});
  Tensor values = tf.zeros({3, 2, 2});
  Tensor indices = tf_long.zeros({3, 2, 2});

  op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);

  EXPECT_TENSOR_EQ(
      values, tf.make({3, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_TENSOR_EQ(
      indices, tf_long.make({3, 2, 2}, {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0}));
}

TEST_F(OpSortValuesTest, ZeroDimTensor) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.make({}, {7});
  Tensor values = tf.zeros({});
  Tensor indices = tf_long.zeros({});

  op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);

  EXPECT_TENSOR_EQ(values, tf.make({}, {7}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({}, {0}));
}

TEST_F(OpSortValuesTest, EmptyTensor) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.make({0, 2}, {});
  Tensor values = tf.zeros({0, 2});
  Tensor indices = tf_long.zeros({0, 2});

  op_sort_values(input, /*dim=*/0, /*descending=*/false, values, indices);

  EXPECT_TENSOR_EQ(values, tf.make({0, 2}, {}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({0, 2}, {}));
}
