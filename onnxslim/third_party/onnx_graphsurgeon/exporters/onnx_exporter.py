#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import IR_VERSION, ModelProto, defs

from onnxslim.third_party.onnx_graphsurgeon.exporters.base_exporter import BaseExporter
from onnxslim.third_party.onnx_graphsurgeon.ir.function import Function
from onnxslim.third_party.onnx_graphsurgeon.ir.graph import Graph
from onnxslim.third_party.onnx_graphsurgeon.ir.node import Node
from onnxslim.third_party.onnx_graphsurgeon.ir.tensor import (
    Constant,
    LazyValues,
    SparseValues,
    Tensor,
    Variable,
)
from onnxslim.third_party.onnx_graphsurgeon.logger import G_LOGGER
from onnxslim.third_party.onnx_graphsurgeon.util import misc


def dtype_to_onnx(dtype: np.dtype | onnx.TensorProto.DataType) -> int:
    """Converts a numpy dtype or ONNX data type to its integer representation."""
    if isinstance(dtype, int):
        return dtype
    return onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))


def check_duplicate_node_names(nodes: Sequence[Node], level=G_LOGGER.WARNING):
    """Check if node names are unique and log any duplicates based on the specified severity level."""
    # Note:
    # Empty string or None attribute values are not considered duplicates.
    name_map = {}
    for node in nodes:
        if not node.name:
            continue
        if node.name in name_map:
            msg = f"Found distinct Nodes that share the same name:\n[id: {id(name_map[node.name])}]:\n {name_map[node.name]}---\n[id: {id(node)}]:\n {node}\n"
            G_LOGGER.log(msg, level)
        else:
            name_map[node.name] = node


def update_import_domains(graph):
    """Update the import_domains field of a graph to include its ONNX opset and other used non-ONNX domains."""
    # as well as other non-ONNX domains which are used by this graph's nodes.
    # Returns the updated value of the import_domains field.

    # Add domain of the standard ONNX opset.
    if graph.import_domains is None:
        graph.import_domains = [onnx.helper.make_opsetid("", graph.opset)]

    # Crawl over all nodes in this graph and its subgraphs, and add the nodes' domains.
    all_used_domains = {node.domain for node in graph.nodes}
    for subgraph in graph.subgraphs(recursive=True):
        all_used_domains |= {n.domain for n in subgraph.nodes}
    all_used_domains.discard(None)

    # Update self.import_domains with any missing domains.
    current_domains = {opsetid.domain for opsetid in graph.import_domains}
    DEFAULT_CUSTOM_OPSET_VERSION = 1
    for used_domain in all_used_domains:
        if used_domain not in current_domains:
            graph.import_domains.append(onnx.helper.make_opsetid(used_domain, DEFAULT_CUSTOM_OPSET_VERSION))
            current_domains.add(used_domain)
    return graph.import_domains


class NumpyArrayConverter:
    def __init__(self, container, scalar_converter):
        self.func = np.vectorize(scalar_converter, otypes=[container])

    def __call__(self, arr):
        return self.func(arr)


_NUMPY_ARRAY_CONVERTERS = {
    onnx.TensorProto.BFLOAT16: NumpyArrayConverter(np.uint16, onnx.helper.float32_to_bfloat16),
    # FP8 in TensorRT supports negative zeros, no infinities
    # See https://onnx.ai/onnx/technical/float8.html#papers
    onnx.TensorProto.FLOAT8E4M3FN: NumpyArrayConverter(
        np.uint8, lambda x: onnx.helper.float32_to_float8e4m3(x, fn=True, uz=False)
    ),
}


def constant_to_onnx_tensor(tensor: Constant) -> onnx.TensorProto:
    source_dtype = dtype_to_onnx(tensor.dtype)
    target_dtype = dtype_to_onnx(tensor.export_dtype)

    if source_dtype != target_dtype:
        source_dtype_str = onnx.helper.tensor_dtype_to_string(source_dtype)
        target_dtype_str = onnx.helper.tensor_dtype_to_string(target_dtype)
        assert source_dtype == onnx.TensorProto.FLOAT, (
            f"Cannot convert onnx dtype {source_dtype_str} to {target_dtype_str}. "
            "Source dtype must be float32 to convert to numpy unsupported dtypes."
        )
        assert target_dtype in _NUMPY_ARRAY_CONVERTERS.keys(), (
            f"Cannot convert onnx dtype {source_dtype_str} to {target_dtype_str}. "
            f"Only float32 to {_NUMPY_ARRAY_CONVERTERS.keys()} is supported."
        )
        arr = _NUMPY_ARRAY_CONVERTERS[target_dtype](tensor.values)
        tensor_raw_bytes = arr.tobytes()
    else:
        tensor_raw_bytes = tensor.values.tobytes()

    return onnx.helper.make_tensor(
        name=tensor.name,
        data_type=target_dtype,
        dims=tensor.shape,
        vals=tensor_raw_bytes,
        raw=True,
    )


class OnnxExporter(BaseExporter):
    @staticmethod
    def export_tensor_proto(tensor: Constant) -> onnx.TensorProto:
        # Do *not* load LazyValues into an intermediate numpy array - instead, use
        # the original onnx.TensorProto directly.
        if isinstance(tensor._values, LazyValues):
            onnx_tensor = tensor._values.tensor
            onnx_tensor.name = tensor.name
        else:
            onnx_tensor = constant_to_onnx_tensor(tensor)

            if tensor.data_location is not None:
                onnx_tensor.data_location = tensor.data_location
        return onnx_tensor

    @staticmethod
    def export_sparse_tensor_proto(tensor: Constant) -> onnx.SparseTensorProto:
        """Exports a given Constant tensor as an ONNX SparseTensorProto."""
        return tensor._values.tensor

    @staticmethod
    def export_value_info_proto(tensor: Tensor, do_type_check: bool) -> onnx.ValueInfoProto:
        """Creates an ONNX ValueInfoProto from a Tensor, optionally checking for dtype information."""
        if do_type_check and tensor.dtype is None:
            G_LOGGER.critical(
                f"Graph input and output tensors must include dtype information. Please set the dtype attribute for: {tensor}"
            )

        if tensor.dtype is None:
            onnx_tensor = onnx.helper.make_empty_tensor_value_info(tensor.name)
        elif isinstance(tensor, Constant) or tensor.type == "tensor_type":
            onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape)
        elif tensor.type == "sequence_type":
            onnx_tensor = onnx.helper.make_tensor_sequence_value_info(
                tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape
            )
        elif tensor.type == "sparse_tensor_type":
            onnx_tensor = onnx.helper.make_sparse_tensor_value_info(
                tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape
            )
        return onnx_tensor

    @staticmethod
    def export_attributes(attrs: dict, subgraph_tensor_map=None) -> list[onnx.AttributeProto]:
        """Convert function attributes to ONNX AttributeProtos for model export."""
        onnx_attrs: list[onnx.AttributeProto] = []
        for key, val in attrs.items():
            if isinstance(val, Tensor):
                val = OnnxExporter.export_tensor_proto(val)
            elif isinstance(val, Graph):
                # Subgraphs don't need to have types specified for their tensors.
                graph = onnx.GraphProto()
                OnnxExporter.export_graph(graph, val, subgraph_tensor_map=subgraph_tensor_map, do_type_check=False)
                val = graph
            elif isinstance(val, Node.AttributeRef):
                onnx_attr = onnx.AttributeProto()
                onnx_attr.name = key
                onnx_attr.type = misc.convert_to_onnx_attr_type(val.type)

                # Netron has a bug which makes it crash if a Tensor attribute has no tensor data.
                # So provide some meaningless tensor data for Netron to read.
                if val.type == Tensor:
                    tensor_proto = OnnxExporter.export_tensor_proto(Constant("", np.array([0], dtype=np.float32)))
                    onnx_attr.t.CopyFrom(tensor_proto)

                onnx_attr.ref_attr_name = val.name
                onnx_attrs.append(onnx_attr)
                continue
            elif isinstance(val, type):
                # May be a numpy type
                try:
                    val = dtype_to_onnx(val)
                except TypeError:
                    pass
            onnx_attrs.append(onnx.helper.make_attribute(key, val))
        return onnx_attrs

    @staticmethod
    def export_node(node: Node, subgraph_tensor_map=None) -> onnx.NodeProto:
        # Cannot pass in attrs directly as make_node will change the order
        """Static method to convert an internal node to an ONNX node representation."""
        onnx_node = onnx.helper.make_node(
            node.op,
            inputs=[t.name for t in node.inputs],
            outputs=[t.name for t in node.outputs],
            name=node.name,
            domain=node.domain,
        )
        onnx_node.attribute.extend(OnnxExporter.export_attributes(node.attrs, subgraph_tensor_map))
        return onnx_node

    @staticmethod
    def export_function(func: Function) -> onnx.FunctionProto:
        """
        Export an onnx-graphsurgeon Function to an ONNX FunctionProto.

        Args:
            func (Function): The function to export.
        """
        # Unlike onnx Graphs, onnx Functions don't have an 'initializer' field.
        # So we need to replace all Constant tensors with onnx Constant nodes which produce them.
        # We need to be careful to (a) preserve topological ordering and (b) not make the new nodes visible to the user.
        func_nodes = func.nodes.copy()
        new_const_nodes = [
            Node("Constant", attrs={"value": tensor}, outputs=[tensor.copy()])
            for tensor in func.tensors().values()
            if isinstance(tensor, Constant)
        ]
        # Const nodes have no inputs, so this maintains a topological ordering.
        func_nodes = new_const_nodes + func_nodes

        check_duplicate_node_names(func_nodes, level=G_LOGGER.WARNING)
        nodes = [OnnxExporter.export_node(node) for node in func_nodes]

        # Update the import_domains field to include all domains used by this function.
        opset_imports = update_import_domains(func)

        onnx_inputs = [inp.name for inp in func.inputs]
        onnx_outputs = [out.name for out in func.outputs]

        attributes = []
        attribute_protos = {}
        for attr_name, default_val in func.attrs.items():
            if default_val is None:
                attributes.append(attr_name)
            else:
                attribute_protos[attr_name] = default_val
        attribute_protos = OnnxExporter.export_attributes(attribute_protos)

        return onnx.helper.make_function(
            func.domain or "",
            func.name,
            onnx_inputs,
            onnx_outputs,
            nodes,
            opset_imports,
            attributes=attributes,
            attribute_protos=attribute_protos,
            doc_string=func.doc_string,
        )

    @staticmethod
    def export_graph(
        graph_proto: onnx.GraphProto,
        graph: Graph,
        tensor_map: OrderedDict[str, Tensor] | None = None,
        subgraph_tensor_map: OrderedDict[str, Tensor] | None = None,
        do_type_check=True,
    ) -> None:
        """
        Export an onnx-graphsurgeon Graph to an ONNX GraphProto.

        Args:
            graph (Graph): The graph to export.

            do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
                                  Defaults to True.
        """
        check_duplicate_node_names(graph.nodes, level=G_LOGGER.WARNING)
        nodes = [OnnxExporter.export_node(node, subgraph_tensor_map) for node in graph.nodes]
        inputs = [OnnxExporter.export_value_info_proto(inp, do_type_check) for inp in graph.inputs]
        outputs = [OnnxExporter.export_value_info_proto(out, do_type_check) for out in graph.outputs]
        if tensor_map is None:
            tensor_map = graph.tensors()
            tensor_map = misc.unique_dicts(tensor_map, subgraph_tensor_map)
        else:
            tensor_map = misc.combine_dicts(tensor_map, subgraph_tensor_map)
        initializer = [
            OnnxExporter.export_tensor_proto(tensor)
            for tensor in tensor_map.values()
            if isinstance(tensor, Constant) and not isinstance(tensor._values, SparseValues)
        ]

        sparse_initializer = [
            OnnxExporter.export_sparse_tensor_proto(tensor)
            for tensor in tensor_map.values()
            if isinstance(tensor, Constant) and isinstance(tensor._values, SparseValues)
        ]

        # Remove inputs and outputs to export ValueInfoProtos
        for tensor in graph.inputs + graph.outputs:
            if tensor.name in tensor_map:
                del tensor_map[tensor.name]

        # Omit tensors from value_info if we don't know their shape/dtype
        def has_value_info(tensor):
            """Check if a tensor is a Variable with either a defined dtype or shape."""
            return isinstance(tensor, Variable) and (tensor.dtype is not None or tensor.shape is not None)

        value_info = [
            OnnxExporter.export_value_info_proto(tensor, do_type_check)
            for tensor in tensor_map.values()
            if has_value_info(tensor)
        ]

        if initializer is None:
            initializer = []
        if sparse_initializer is None:
            sparse_initializer = []
        if value_info is None:
            value_info = []

        graph_proto.node.extend(nodes)
        graph_proto.name = graph.name
        graph_proto.input.extend(inputs)
        graph_proto.output.extend(outputs)
        for init in initializer:
            graph_proto.initializer.add().CopyFrom(init)
        for sparse in sparse_initializer:
            graph_proto.sparse_initializer.add().CopyFrom(sparse)
        graph_proto.value_info.extend(value_info)
        if graph.doc_string:
            graph.doc_string = graph.doc_string

        return graph_proto


def export_onnx(graph: Graph, do_type_check=True, **kwargs) -> onnx.ModelProto:
    """
    Exports an onnx-graphsurgeon Graph to an ONNX model.

    Args:
        graph (Graph): The graph to export

        do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
                              Defaults to True.
        kwargs: Additional arguments to onnx.helper.make_model

    Returns:
        onnx.ModelProto: A corresponding ONNX model.
    """
    sub_graphs = graph.subgraphs(recursive=True)

    graph_constants_list = [
        {name: tensor for name, tensor in sub_graph.tensors().items() if isinstance(tensor, Constant)}
        for sub_graph in sub_graphs
    ]

    if not graph_constants_list:
        intersection = None
    else:
        intersection = (
            {
                key: graph_constants_list[0][key]
                for key in graph_constants_list[0]
                if all(key in d and graph_constants_list[0][key] == d[key] for d in graph_constants_list[1:])
            }
            if graph_constants_list
            else None
        )

    model = ModelProto()  # create in advance to avoid unnecessary copy
    OnnxExporter.export_graph(
        model.graph, graph, tensor_map=graph.tensors(), subgraph_tensor_map=intersection, do_type_check=do_type_check
    )
    onnx_functions = [OnnxExporter.export_function(func) for func in graph.functions]
    kwargs["functions"] = onnx_functions

    if "opset_imports" not in kwargs:
        kwargs["opset_imports"] = update_import_domains(graph)

    if "ir_version" not in kwargs and graph.ir_version is not None:
        kwargs["ir_version"] = graph.ir_version
    else:
        model.ir_version = IR_VERSION

    opset_imports = None
    opset_imports = kwargs.pop("opset_imports", None)  # type: ignore
    if opset_imports is not None:
        model.opset_import.extend(opset_imports)
    else:
        # Default import
        imp = model.opset_import.add()
        imp.version = defs.onnx_opset_version()

    functions = None
    functions = kwargs.pop("functions", None)  # type: ignore
    if functions is not None:
        model.functions.extend(functions)

    for k, v in kwargs.items():
        # TODO: Does this work with repeated fields?
        setattr(model, k, v)

    if graph.metadata_props is not None:
        model.metadata_props.extend(graph.metadata_props)
    model.producer_name = graph.producer_name
    model.producer_version = graph.producer_version
    return model
