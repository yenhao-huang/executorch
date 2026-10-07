/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include <cstdlib>
#include <cstdint>

namespace torch {
namespace executor {
namespace testing {

/// A MemoryAllocator for kernels that call `KernelRuntimeContext::allocate_temp`
/// in a test, where no planned temp arena exists.
class TempMemoryAllocator final
    : public executorch::ET_RUNTIME_NAMESPACE::MemoryAllocator {
 private:
  // We allocate a little more than requested and use that memory as a node in
  // a linked list, pushing the allocated buffers onto a list that's iterated
  // and freed when the KernelRuntimeContext is destroyed.
  struct AllocationNode {
    void* data;
    AllocationNode* next;
  };

  AllocationNode* head_ = nullptr;

 public:
  TempMemoryAllocator() : MemoryAllocator(0, nullptr) {}

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    // Allocate enough memory for the node, the data and the alignment bump.
    size_t alloc_size = sizeof(AllocationNode) + size + alignment;
    void* node_memory = malloc(alloc_size);

    // If allocation failed, log message and return nullptr.
    if (node_memory == nullptr) {
      ET_LOG(Error, "Failed to allocate %zu bytes", alloc_size);
      return nullptr;
    }

    // Compute data pointer.
    uint8_t* data_ptr =
        reinterpret_cast<uint8_t*>(node_memory) + sizeof(AllocationNode);

    // Align the data pointer.
    void* aligned_data_ptr = alignPointer(data_ptr, alignment);

    // Assert that the alignment didn't overflow the allocated memory.
    ET_DCHECK_MSG(
        reinterpret_cast<uintptr_t>(aligned_data_ptr) + size <=
            reinterpret_cast<uintptr_t>(node_memory) + alloc_size,
        "aligned_data_ptr %p + size %zu > node_memory %p + alloc_size %zu",
        aligned_data_ptr,
        size,
        node_memory,
        alloc_size);

    // Construct the node.
    AllocationNode* new_node = reinterpret_cast<AllocationNode*>(node_memory);
    new_node->data = aligned_data_ptr;
    new_node->next = head_;
    head_ = new_node;

    // Return the aligned data pointer.
    return head_->data;
  }

  void reset() override {
    AllocationNode* current = head_;
    while (current != nullptr) {
      AllocationNode* next = current->next;
      free(current);
      current = next;
    }
    head_ = nullptr;
  }

  ~TempMemoryAllocator() override {
    reset();
  }
};


} // namespace testing
} // namespace executor
} // namespace torch
