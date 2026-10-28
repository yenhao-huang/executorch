# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Union

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpGather, OpGatherND, OpReshape, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Index(NodeVisitor):
    # schema = aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
    target = ["aten.index.Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> Union[PyQnnManager.PyQnnOpWrapper, List[PyQnnManager.PyQnnOpWrapper]]:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # e.g. x[:, index]:
        # > node.args[1] = [None, indices]
        # > axis = 1
        axis = len(node.args[1]) - 1
        indices_node = node.args[1][axis]
        indices_tensor = self.get_tensor(indices_node, node).to(torch.int32)
        assert indices_tensor.size(0) != 0, "Not support empty indices list"

        indices_tensor_wrapper = self.define_tensor(
            indices_node,
            node,
            indices_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        _, input_quant_config = self.get_quant_encoding_conf(input_node, node)
        use_gather_nd = (
            axis == 0
            and input_tensor.dim() == 2
            and indices_tensor.dim() == 1
            and self.get_data_type(input_tensor, input_quant_config)
            == PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16
        )
        if use_gather_nd:
            # HTP Gather can miscompile broadcast consumers of 16-bit tables.
            # One-component GatherNd indices preserve axis-zero indexing.
            nd_indices = indices_tensor.unsqueeze(-1)
            nd_indices_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_nd_indices",
                tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                quant_encoding=PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                quant_configs={},
                dims=nd_indices.size(),
                tensor=nd_indices,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            reshape_op = PyQnnManager.PyQnnOpWrapper(
                node.name + "_reshape_indices",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpReshape.op_name,
            )
            reshape_op.AddInputTensors([indices_tensor_wrapper])
            reshape_op.AddOutputTensors([nd_indices_wrapper])
            indices_tensor_wrapper = nd_indices_wrapper

        gather_input_tensors = [input_tensor_wrapper, indices_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        gather_output_tensors = [output_tensor_wrapper]

        gather_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGatherND.op_name if use_gather_nd else OpGather.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)

        if use_gather_nd:
            gather_op.AddScalarParam(
                OpGatherND.param_batch_dims,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {QCOM_DATA: np.uint32(0)},
            )
            return [reshape_op, gather_op]

        # If support tuple of tensor, need to refine it based on len
        gather_op.AddScalarParam(
            OpGather.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(axis)},
        )

        return gather_op
