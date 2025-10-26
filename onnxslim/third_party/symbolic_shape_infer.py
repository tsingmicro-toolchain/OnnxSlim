# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
import logging

import numpy as np
import onnx
import sympy
from onnx import helper, numpy_helper, shape_inference
from packaging import version

from onnxslim.third_party._sympy.functions import FloorDiv
from onnxslim.third_party._sympy.printers import PythonPrinter as _PythonPrinter
from onnxslim.third_party._sympy.solve import try_solve

assert version.parse(onnx.__version__) >= version.parse("1.8.0")

logger = logging.getLogger(__name__)


class PythonPrinter(_PythonPrinter):
    def doprint(self, expr: sympy.Expr, *, simplify: bool = True, p: bool = True) -> str:
        # TODO: why are people passing strings to the printer here :think:
        # if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
        #     expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)


pexpr = PythonPrinter().doprint


def get_attribute(node, attr_name, default_value=None):
    """Retrieve the value of an attribute from an ONNX node, returning a default if the attribute is not found."""
    found = [attr for attr in node.attribute if attr.name == attr_name]
    return helper.get_attribute_value(found[0]) if found else default_value


def get_dim_from_proto(dim):
    """Retrieve the dimension value from the ONNX protobuf object if it is a string."""
    return getattr(dim, dim.WhichOneof("value")) if type(dim.WhichOneof("value")) is str else None


def is_sequence(type_proto):
    """Check if the given ONNX proto type is a sequence."""
    cls_type = type_proto.WhichOneof("value")
    assert cls_type in {"tensor_type", "sequence_type"}
    return cls_type == "sequence_type"


def get_shape_from_type_proto(type_proto):
    """Extract the shape of a tensor from an ONNX type proto if available, otherwise return None."""
    assert not is_sequence(type_proto)
    if type_proto.tensor_type.HasField("shape"):
        return [get_dim_from_proto(d) for d in type_proto.tensor_type.shape.dim]
    else:
        return None  # note no shape is different from shape without dim (scalar)


def get_elem_type_from_type_proto(type_proto):
    """Return the element type from a given TypeProto object, either from sequence type or tensor type."""
    if is_sequence(type_proto):
        return type_proto.sequence_type.elem_type.tensor_type.elem_type
    else:
        return type_proto.tensor_type.elem_type


def get_shape_from_value_info(vi):
    """Return the shape from the given ValueInfoProto object, either from sequence type or tensor type."""
    cls_type = vi.type.WhichOneof("value")
    if cls_type is None:
        return None
    if not is_sequence(vi.type):
        return get_shape_from_type_proto(vi.type)
    if vi.type.sequence_type.elem_type.WhichOneof("value") == "tensor_type":
        return get_shape_from_type_proto(vi.type.sequence_type.elem_type)
    else:
        return None


def make_named_value_info(name):
    """Create and return an ONNX ValueInfoProto object with the specified name."""
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def get_shape_from_sympy_shape(sympy_shape):
    """Convert a sympy shape to a list with int, str, or None elements."""
    return [None if i is None else (int(i) if is_literal(i) else str(i)) for i in sympy_shape]


def is_literal(dim):
    """Check if a dimension is a literal number (int, np.int64, np.int32, sympy.Integer) or has an 'is_number'
    attribute.
    """
    return type(dim) in {int, np.int64, np.int32, sympy.Integer} or (hasattr(dim, "is_number") and dim.is_number)


def handle_negative_axis(axis, rank):
    """Convert a potentially negative axis to a positive axis based on the given rank."""
    assert axis < rank and axis >= -rank
    return axis if axis >= 0 else rank + axis


def get_opset(mp, domain=None):
    """Retrieve the opset version for a given model namespace, defaulting to common ONNX domains if no specific domain
    is provided.
    """
    domain = domain or ["", "onnx", "ai.onnx"]
    if type(domain) != list:  # noqa: E721
        domain = [domain]
    for opset in mp.opset_import:
        if opset.domain in domain:
            return opset.version

    return None


def as_scalar(x):
    """Convert input to scalar if input is a list with a single item or a NumPy ndarray."""
    if type(x) == list:  # noqa: E721
        assert len(x) == 1
        return x[0]
    elif type(x) == np.ndarray:
        return x.item()
    else:
        return x


def as_list(x, keep_none):
    """Convert input to list, optionally preserving None values."""
    if type(x) == list:  # noqa: E721
        return x
    elif type(x) == np.ndarray:
        return list(x)
    elif keep_none and x is None:
        return None
    else:
        return [x]


def sympy_reduce_product(x):
    """Reduce a list or element to a product using Sympy's Integer."""
    if type(x) == list:  # noqa: E721
        value = sympy.Integer(1)
        for v in x:
            value = value * v
    else:
        value = x
    return value


class SymbolicShapeInference:
    def __init__(self, int_max, auto_merge, guess_output_rank, verbose, prefix=""):
        """Initializes the SymbolicShapeInference class with configuration parameters for symbolic shape inference."""
        self.dispatcher_ = {
            "Add": self._infer_symbolic_compute_ops,
            "ArrayFeatureExtractor": self._infer_ArrayFeatureExtractor,
            "AveragePool": self._infer_Pool,
            "BatchNormalization": self._infer_BatchNormalization,
            "Cast": self._infer_Cast,
            "CategoryMapper": self._infer_CategoryMapper,
            "Compress": self._infer_Compress,
            "Concat": self._infer_Concat,
            "ConcatFromSequence": self._infer_ConcatFromSequence,
            "Constant": self._infer_Constant,
            "ConstantOfShape": self._infer_ConstantOfShape,
            "Conv": self._infer_Conv,
            "CumSum": self._pass_on_shape_and_type,
            "Div": self._infer_symbolic_compute_ops,
            "Einsum": self._infer_Einsum,
            "Expand": self._infer_Expand,
            "Equal": self._infer_symbolic_compute_ops,
            "Floor": self._infer_symbolic_compute_ops,
            "Gather": self._infer_Gather,
            "GatherElements": self._infer_GatherElements,
            "GatherND": self._infer_GatherND,
            "Identity": self._pass_on_shape_and_type,
            "AllReduce": self._pass_on_shape_and_type,
            "If": self._infer_If,
            "Loop": self._infer_Loop,
            "MatMul": self._infer_MatMul,
            "MatMulInteger16": self._infer_MatMulInteger,
            "MaxPool": self._infer_Pool,
            "Max": self._infer_symbolic_compute_ops,
            "MemcpyFromHost": self._pass_on_shape_and_type,
            "MemcpyToHost": self._pass_on_shape_and_type,
            "Min": self._infer_symbolic_compute_ops,
            "MoE": self._pass_on_shape_and_type,
            "Mul": self._infer_symbolic_compute_ops,
            "NonMaxSuppression": self._infer_NonMaxSuppression,
            "NonZero": self._infer_NonZero,
            "OneHot": self._infer_OneHot,
            "Pad": self._infer_Pad,
            "Range": self._infer_Range,
            "Reciprocal": self._pass_on_shape_and_type,
            "ReduceSum": self._infer_ReduceSum,
            "ReduceProd": self._infer_ReduceProd,
            "Reshape": self._infer_Reshape,
            "Resize": self._infer_Resize,
            "Round": self._pass_on_shape_and_type,
            "Scan": self._infer_Scan,
            "ScatterElements": self._infer_ScatterElements,
            "SequenceAt": self._infer_SequenceAt,
            "SequenceInsert": self._infer_SequenceInsert,
            "Shape": self._infer_Shape,
            "Size": self._infer_Size,
            "Slice": self._infer_Slice,
            "SoftmaxCrossEntropyLoss": self._infer_SoftmaxCrossEntropyLoss,
            "SoftmaxCrossEntropyLossInternal": self._infer_SoftmaxCrossEntropyLoss,
            "NegativeLogLikelihoodLossInternal": self._infer_SoftmaxCrossEntropyLoss,
            "Split": self._infer_Split,
            "SplitToSequence": self._infer_SplitToSequence,
            "Squeeze": self._infer_Squeeze,
            "Sub": self._infer_symbolic_compute_ops,
            "Tile": self._infer_Tile,
            "TopK": self._infer_TopK,
            "Transpose": self._infer_Transpose,
            "Unsqueeze": self._infer_Unsqueeze,
            "Where": self._infer_symbolic_compute_ops,
            "ZipMap": self._infer_ZipMap,
            "Neg": self._infer_symbolic_compute_ops,
            # contrib ops:
            "Attention": self._infer_Attention,
            "BiasAdd": self._infer_BiasAdd,
            "BiasGelu": self._infer_BiasGelu,
            "BiasSplitGelu": self._infer_BiasSplitGelu,
            "DecoderMaskedMultiHeadAttention": self._infer_DecoderMaskedMultiHeadAttention,
            "DequantizeLinear": self._infer_DequantizeLinear,
            "EmbedLayerNormalization": self._infer_EmbedLayerNormalization,
            "FastGelu": self._infer_FastGelu,
            "GatedRelativePositionBias": self._infer_GatedRelativePositionBias,
            "Gelu": self._infer_Gelu,
            "GemmFastGelu": self._infer_GemmFastGelu,
            "GemmFloat8": self._infer_GemmFloat8,
            "GroupNorm": self._infer_GroupNorm,
            "SkipGroupNorm": self._infer_SkipGroupNorm,
            "LayerNormalization": self._infer_LayerNormalization,
            "LongformerAttention": self._infer_LongformerAttention,
            "MultiHeadAttention": self._infer_MultiHeadAttention,
            "NhwcConv": self._infer_NhwcConv,
            "PackedAttention": self._infer_PackedAttention,
            "PackedMultiHeadAttention": self._infer_PackedMultiHeadAttention,
            "MultiScaleDeformableAttnTRT": self._infer_MultiScaleDeformableAttnTRT,
            "PythonOp": self._infer_PythonOp,
            "QuantizeLinear": self._infer_QuantizeLinear,
            "QuickGelu": self._infer_FastGelu,
            "RelativePositionBias": self._infer_RelativePositionBias,
            "RemovePadding": self._infer_RemovePadding,
            "RestorePadding": self._infer_RestorePadding,
            "RotaryEmbedding": self._infer_RotaryEmbedding,
            "SimplifiedLayerNormalization": self._infer_LayerNormalization,
            "SkipLayerNormalization": self._infer_SkipLayerNormalization,
            "SkipSimplifiedLayerNormalization": self._infer_SkipLayerNormalization,
        }
        self.aten_op_dispatcher_ = {
            "embedding": self._infer_Gather,
            "bitwise_or": self._infer_aten_bitwise_or,
            "diagonal": self._infer_aten_diagonal,
            "max_pool2d_with_indices": self._infer_aten_pool2d,
            "max": self._infer_aten_minmax,
            "min": self._infer_aten_minmax,
            "multinomial": self._infer_aten_multinomial,
            "unfold": self._infer_aten_unfold,
            "argmax": self._infer_aten_argmax,
            "avg_pool2d": self._infer_aten_pool2d,
            "_adaptive_avg_pool2d": self._infer_aten_pool2d,
            "numpy_T": self._infer_Transpose,
            "native_group_norm": self._infer_aten_group_norm,
            "upsample_nearest1d": self._infer_aten_upsample,
            "upsample_nearest2d": self._infer_aten_upsample,
            "upsample_nearest3d": self._infer_aten_upsample,
            "upsample_bicubic2d": self._infer_aten_upsample,
        }
        self.run_ = True
        self.suggested_merge_ = {}
        self.symbolic_dims_ = {}
        self.input_symbols_ = {}
        self.auto_merge_ = auto_merge
        self.guess_output_rank_ = guess_output_rank
        self.verbose_ = verbose
        self.int_max_ = int_max
        self.subgraph_id_ = 0
        self.prefix_ = prefix

    def _add_suggested_merge(self, symbols, apply=False):
        """Add suggested merges for input symbols, prioritizing literals, input symbolic dims, or existing symbolic
        dims.
        """
        assert all((type(s) == str and s in self.symbolic_dims_) or is_literal(s) for s in symbols)
        symbols = set(symbols)
        for k, v in self.suggested_merge_.items():
            if k in symbols:
                symbols.remove(k)
                symbols.add(v)
        map_to = None
        # if there is literal, map to it first
        for s in symbols:
            if is_literal(s):
                map_to = s
                break
        # when no literals, map to input symbolic dims, then existing symbolic dims
        if map_to is None:
            for s in symbols:
                if s in self.input_symbols_:
                    map_to = s
                    break
        if map_to is None:
            for s in symbols:
                if type(self.symbolic_dims_[s]) == sympy.Symbol:
                    map_to = s
                    break
        # when nothing to map to, use the shorter one
        if map_to is None:
            if self.verbose_ > 0:
                logger.warning(f"Potential unsafe merge between symbolic expressions: ({','.join(symbols)})")
            symbols_list = list(symbols)
            lens = [len(s) for s in symbols_list]
            map_to = symbols_list[lens.index(min(lens))]
            symbols.remove(map_to)

        for s in symbols:
            if s == map_to:
                continue
            if is_literal(map_to) and is_literal(s):
                assert int(map_to) == int(s)
            self.suggested_merge_[s] = int(map_to) if is_literal(map_to) else map_to
            for k, v in self.suggested_merge_.items():
                if v == s:
                    self.suggested_merge_[k] = map_to
        if apply and self.auto_merge_:
            self._apply_suggested_merge()

    def _apply_suggested_merge(self, graph_input_only=False):
        """Applies suggested merges to graph dimensions based on predefined rules in the `suggested_merge_`
        dictionary.
        """
        if not self.suggested_merge_:
            return
        for i in list(self.out_mp_.graph.input) + ([] if graph_input_only else list(self.out_mp_.graph.value_info)):
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param in self.suggested_merge_:
                    v = self.suggested_merge_[d.dim_param]
                    if is_literal(v):
                        d.dim_value = int(v)
                    else:
                        d.dim_param = v

    def _preprocess(self, in_mp):
        self.out_mp_ = in_mp
        self.graph_inputs_ = {i.name: i for i in list(self.out_mp_.graph.input)}
        self.initializers_ = {i.name: i for i in self.out_mp_.graph.initializer}
        self.known_vi_ = {i.name: i for i in list(self.out_mp_.graph.input)}
        self.known_vi_.update(
            {
                i.name: helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))
                for i in self.out_mp_.graph.initializer
            }
        )

    def _merge_symbols(self, dims):
        """Merge dimension symbols, handling automatic merging and validation of symbolic dimensions."""
        if any(type(d) != str for d in dims):  # noqa: E721
            if not self.auto_merge_:
                return None
            unique_dims = list(set(dims))
            is_int = [is_literal(d) for d in unique_dims]
            assert sum(is_int) <= 1  # if there are more than 1 unique ints, something is wrong
            if sum(is_int) == 1:
                int_dim = is_int.index(1)
                if self.verbose_ > 0:
                    logger.debug(
                        f"dim {unique_dims[:int_dim] + unique_dims[int_dim + 1 :]} has been merged with value {unique_dims[int_dim]}"
                    )
                self._check_merged_dims(unique_dims, allow_broadcast=False)
                return unique_dims[int_dim]
            else:
                if self.verbose_ > 0:
                    logger.debug(f"dim {unique_dims[1:]} has been merged with dim {unique_dims[0]}")
                return dims[0]
        if all(d == dims[0] for d in dims):
            return dims[0]
        merged = [self.suggested_merge_[d] if d in self.suggested_merge_ else d for d in dims]
        if all(d == merged[0] for d in merged):
            assert merged[0] in self.symbolic_dims_
            return merged[0]
        else:
            return None

    # broadcast from right to left, and merge symbolic dims if needed
    def _broadcast_shapes(self, shape1, shape2):
        """Broadcast two shapes from right to left, merging symbolic dimensions if necessary."""
        new_shape = []
        rank1 = len(shape1)
        rank2 = len(shape2)
        new_rank = max(rank1, rank2)
        for i in range(new_rank):
            dim1 = shape1[rank1 - 1 - i] if i < rank1 else 1
            dim2 = shape2[rank2 - 1 - i] if i < rank2 else 1
            if dim1 in [1, dim2]:
                new_dim = dim2
            elif dim2 == 1:
                new_dim = dim1
            else:
                new_dim = self._merge_symbols([dim1, dim2])
                if not new_dim:
                    # warning about unsupported broadcast when not auto merge
                    # note that auto merge has the risk of incorrectly merge symbols while one of them being 1
                    # for example, 'a' = 1, 'b' = 5 at runtime is valid broadcasting, but with auto merge 'a' == 'b'
                    if self.auto_merge_:
                        self._add_suggested_merge([dim1, dim2], apply=True)
                    else:
                        logger.warning(f"unsupported broadcast between {dim1!s} {dim2!s}")
            new_shape = [new_dim, *new_shape]
        return new_shape

    def _get_shape(self, node, idx):
        """Retrieve the shape of a tensor from a node's inputs based on known value info or initializers."""
        name = node.input[idx]
        if name in self.known_vi_:
            vi = self.known_vi_[name]
            return get_shape_from_value_info(vi)
        else:
            assert name in self.initializers_
            return list(self.initializers_[name].dims)

    def _try_get_shape(self, node, idx):
        """Attempts to retrieve the shape of the input node at the specified index if available."""
        if idx > len(node.input) - 1:
            return None
        name = node.input[idx]
        if name in self.known_vi_:
            vi = self.known_vi_[name]
            return get_shape_from_value_info(vi)
        if name in self.initializers_:
            return list(self.initializers_[name].dims)
        return None

    def _get_shape_rank(self, node, idx):
        """Return the rank (number of dimensions) of the shape of the input tensor at the specified index for a given
        node.
        """
        return len(self._get_shape(node, idx))

    def _get_sympy_shape(self, node, idx):
        """Return the symbolic shape dimensions using SymPy for the given input tensor at the specified index for a
        node.
        """
        sympy_shape = []
        for d in self._get_shape(node, idx):
            if type(d) == str:  # noqa: E721
                sympy_shape.append(
                    self.symbolic_dims_[d]
                    if d in self.symbolic_dims_
                    else sympy.Symbol(d, integer=True, nonnegative=True)
                )
            else:
                assert None is not d
                sympy_shape.append(d)
        return sympy_shape

    def _get_value(self, node, idx):
        """Retrieve the value associated with a node's input index from sympy_data_ or initializers_."""
        name = node.input[idx]
        assert name in self.sympy_data_ or name in self.initializers_
        return self.sympy_data_[name] if name in self.sympy_data_ else numpy_helper.to_array(self.initializers_[name])

    def _try_get_value(self, node, idx):
        """Try to retrieve the value associated with a node's input index from sympy_data_ or initializers_."""
        if idx >= len(node.input):
            return None
        name = node.input[idx]
        if name in self.sympy_data_ or name in self.initializers_:
            return self._get_value(node, idx)
        return None

    def _update_computed_dims(self, new_sympy_shape):
        """Update dimensions in new_sympy_shape based on suggested merges and computational expressions."""
        for i, new_dim in enumerate(new_sympy_shape):
            if not is_literal(new_dim) and type(new_dim) != str:  # noqa: E721
                str_dim = pexpr(new_dim)
                if str_dim in self.suggested_merge_:
                    if not is_literal(self.suggested_merge_[str_dim]):
                        new_sympy_shape[i] = self.symbolic_dims_[self.suggested_merge_[str_dim]]
                elif str_dim not in self.symbolic_dims_:
                    self.symbolic_dims_[str_dim] = new_dim

    def _onnx_infer_single_node(self, node):
        """Performs ONNX shape inference for a single node, skipping inference for specified operation types."""
        skip_infer = node.op_type in {
            "If",
            "Loop",
            "Scan",
            "SplitToSequence",
            "ZipMap",  # contrib ops
            "Attention",
            "BiasGelu",
            "EmbedLayerNormalization",
            "FastGelu",
            "Gelu",
            "GemmFastGelu",
            "LayerNormalization",
            "LongformerAttention",
            "DequantizeLinear",
            "QuantizeLinear",
            "RelativePositionBias",
            "RemovePadding",
            "RestorePadding",
            "SimplifiedLayerNormalization",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "PackedAttention",
            "PythonOp",
            "MultiHeadAttention",
            "GroupNorm",
            "SkipGroupNorm",
            "BiasSplitGelu",
            "BiasAdd",
            "NhwcConv",
            "QuickGelu",
            "RotaryEmbedding",
        }

        if not skip_infer:
            # Only pass initializers that satisfy the following condition:
            # (1) Operator need value of some input for shape inference.
            #     For example, Unsqueeze in opset 13 uses the axes input to calculate shape of output.
            # (2) opset version >= 9. In older version, initializer is required in graph input by onnx spec.
            # (3) The initializer is not in graph input. The means the node input is "constant" in inference.
            initializers = []
            if (get_opset(self.out_mp_) >= 9) and (
                node.op_type == "Unsqueeze" or node.op_type == "ReduceMax" or node.op_type == "ReduceMean"
            ):
                initializers = [
                    self.initializers_[name]
                    for name in node.input
                    if (name in self.initializers_ and name not in self.graph_inputs_)
                ]

            if (
                node.op_type
                in {
                    "Add",
                    "Sub",
                    "Mul",
                    "Div",
                    "MatMul",
                    "MatMulInteger",
                    "MatMulInteger16",
                    "Where",
                    "Sum",
                }
                and node.output[0] in self.known_vi_
            ):
                vi = self.known_vi_[node.output[0]]
                out_rank = len(get_shape_from_type_proto(vi.type))
                in_shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                for d in range(out_rank - (2 if node.op_type in {"MatMul", "MatMulInteger", "MatMulInteger16"} else 0)):
                    in_dims = [s[len(s) - out_rank + d] for s in in_shapes if len(s) + d >= out_rank]
                    if len(in_dims) > 1:
                        self._check_merged_dims(in_dims, allow_broadcast=True)

            # run single node inference with self.known_vi_ shapes
            tmp_graph = helper.make_graph(
                [node],
                "tmp",
                [self.known_vi_[i] for i in node.input if i],
                [make_named_value_info(i) for i in node.output],
                initializers,
            )

            kwargs = {}
            kwargs["opset_imports"] = self.out_mp_.opset_import
            kwargs["ir_version"] = self.out_mp_.ir_version

            model = helper.make_model(tmp_graph, **kwargs)
            model = shape_inference.infer_shapes(model)

        for i_o in range(len(node.output)):
            o = node.output[i_o]
            if o:  # skip optional output
                vi = self.out_mp_.graph.value_info.add()
                if not skip_infer:
                    vi.CopyFrom(model.graph.output[i_o])
                else:
                    vi.name = o
                self.known_vi_[o] = vi

    def _onnx_infer_subgraph(self, node, subgraph, use_node_input=True, inc_subgraph_id=True):
        """Infer shapes and types within a subgraph for a given ONNX node using temporary graphs and known value
        information.
        """
        if self.verbose_ > 2:
            logger.debug(f"Inferencing subgraph of node {node.name} with output({node.output[0]}...): {node.op_type}")
        # node inputs are not passed directly to the subgraph
        # it's up to the node dispatcher to prepare subgraph input
        # for example, with Scan/Loop, subgraph input shape would be trimmed from node input shape
        # besides, inputs in subgraph could shadow implicit inputs
        subgraph_inputs = {i.name for i in list(subgraph.initializer) + list(subgraph.input)}
        subgraph_implicit_input = {name for name in self.known_vi_ if name not in subgraph_inputs}
        tmp_graph = helper.make_graph(
            list(subgraph.node),
            "tmp",
            list(subgraph.input) + [self.known_vi_[i] for i in subgraph_implicit_input],
            [make_named_value_info(i.name) for i in subgraph.output],
        )
        tmp_graph.initializer.extend([i for i in self.out_mp_.graph.initializer if i.name in subgraph_implicit_input])
        tmp_graph.initializer.extend(subgraph.initializer)
        kwargs = {}
        kwargs["opset_imports"] = self.out_mp_.opset_import
        kwargs["ir_version"] = self.out_mp_.ir_version

        model = helper.make_model(tmp_graph, **kwargs)

        symbolic_shape_inference = SymbolicShapeInference(
            self.int_max_,
            self.auto_merge_,
            self.guess_output_rank_,
            self.verbose_,
            prefix=f"{self.prefix_}_{self.subgraph_id_!s}",
        )
        if inc_subgraph_id:
            self.subgraph_id_ += 1

        symbolic_shape_inference._preprocess(model)
        symbolic_shape_inference.suggested_merge_ = self.suggested_merge_.copy()
        while symbolic_shape_inference.run_:
            symbolic_shape_inference._infer_impl(self.sympy_data_.copy())
        symbolic_shape_inference._update_output_from_vi()
        if use_node_input:
            # if subgraph uses node input, it needs to update to merged dims
            subgraph.ClearField("input")
            subgraph.input.extend(symbolic_shape_inference.out_mp_.graph.input[: len(node.input)])
        subgraph.ClearField("output")
        subgraph.output.extend(symbolic_shape_inference.out_mp_.graph.output)
        subgraph.ClearField("value_info")
        subgraph.value_info.extend(symbolic_shape_inference.out_mp_.graph.value_info)
        subgraph.ClearField("node")
        subgraph.node.extend(symbolic_shape_inference.out_mp_.graph.node)
        # for new symbolic dims from subgraph output, add to main graph symbolic dims
        subgraph_shapes = [get_shape_from_value_info(o) for o in symbolic_shape_inference.out_mp_.graph.output]
        subgraph_new_symbolic_dims = {
            d
            for s in subgraph_shapes
            if s
            for d in s
            if type(d) == str and d not in self.symbolic_dims_  # noqa: E721
        }
        new_dims = {}
        for d in subgraph_new_symbolic_dims:
            assert d in symbolic_shape_inference.symbolic_dims_
            new_dims[d] = symbolic_shape_inference.symbolic_dims_[d]
        self.symbolic_dims_.update(new_dims)
        return symbolic_shape_inference

    def _get_int_or_float_values(self, node, broadcast=False, allow_float_values=False):
        """Extracts integer or float values from a node, with options for broadcasting and allowing float values."""

        def int_or_float(value, allow_float_values):
            """Converts a value to an integer unless precision loss occurs and allow_float_values is True."""
            return value if allow_float_values and value % 1 != 0 else int(value)

        values = [self._try_get_value(node, i) for i in range(len(node.input))]
        if all(v is not None for v in values):
            # some shape compute is in floating point, cast to int for sympy
            for i, v in enumerate(values):
                if type(v) != np.ndarray:
                    continue
                if len(v.shape) > 1:
                    new_v = None  # ignore value for rank > 1
                elif len(v.shape) == 0:
                    new_v = int_or_float(v.item(), allow_float_values)
                else:
                    assert len(v.shape) == 1
                    new_v = [int_or_float(vv, allow_float_values) for vv in v]
                values[i] = new_v
        values_len = [len(v) if isinstance(v, list) else 0 for v in values]
        max_len = max(values_len)
        if max_len >= 1 and broadcast:
            # broadcast
            for i, v in enumerate(values):
                if v is None:
                    continue  # don't broadcast if value is unknown
                if isinstance(v, list):
                    if len(v) < max_len:
                        values[i] = v * max_len
                    else:
                        assert len(v) == max_len
                else:
                    values[i] = [v] * max_len
        return values

    def _compute_on_sympy_data(self, node, op_func):
        """Calculate the result using Sympy data and a specified operation function."""
        assert len(node.output) == 1

        # Before mul & div operations
        # cast inputs into integer might lose decimal part and reduce precision
        # keep them as float, finish the operation, then cast the result into integer
        if node.op_type in {"Mul", "Div"}:
            values = self._get_int_or_float_values(node, broadcast=True, allow_float_values=True)
        else:
            values = self._get_int_or_float_values(node, broadcast=True)
        if all(v is not None for v in values):
            is_list = [isinstance(v, list) for v in values]
            as_list = any(is_list)
            if as_list:
                self.sympy_data_[node.output[0]] = [op_func(vs) for vs in zip(*values)]
            else:
                self.sympy_data_[node.output[0]] = op_func(values)

    def _pass_on_sympy_data(self, node):
        """Pass Sympy data through a node, validating input length or node operation type 'Reshape', 'Unsqueeze',
        'Squeeze'.
        """
        assert len(node.input) == 1 or node.op_type in {
            "Reshape",
            "Unsqueeze",
            "Squeeze",
        }
        self._compute_on_sympy_data(node, lambda x: x[0])

    def _pass_on_shape_and_type(self, node):
        """Propagates the shape and type information from input to output for a given node."""
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                get_elem_type_from_type_proto(self.known_vi_[node.input[0]].type),
                self._get_shape(node, 0),
            )
        )

    def _new_symbolic_dim(self, prefix, dim):
        """Create and return a new symbolic dimension, handling literal values and caching for repeated uses."""
        new_dim = f"{prefix}_d{dim}"
        if new_dim in self.suggested_merge_:
            v = self.suggested_merge_[new_dim]
            new_symbolic_dim = sympy.Integer(int(v)) if is_literal(v) else v
        else:
            new_symbolic_dim = sympy.Symbol(new_dim, integer=True, nonnegative=True)
            self.symbolic_dims_[new_dim] = new_symbolic_dim
        return new_symbolic_dim

    def _new_symbolic_dim_from_output(self, node, out_idx=0, dim=0):
        """Generates a new symbolic dimension for a given node's output using the node's operation type, prefix, and
        output index.
        """
        return self._new_symbolic_dim(
            f"{node.op_type}{self.prefix_}_{list(self.out_mp_.graph.node).index(node)}_o{out_idx}_",
            dim,
        )

    def _new_symbolic_shape(self, rank, node, out_idx=0):
        """Generate a new symbolic shape for a node output based on its rank and index."""
        return [self._new_symbolic_dim_from_output(node, out_idx, i) for i in range(rank)]

    def _compute_conv_pool_shape(self, node, channels_last=False):
        """Calculate the output shape of a convolutional or pooling layer node, optionally considering channels_last
        format.
        """
        sympy_shape = self._get_sympy_shape(node, 0)
        if len(node.input) > 1:
            W_shape = self._get_sympy_shape(node, 1)
            rank = len(W_shape) - 2  # number of spatial axes
            kernel_shape = W_shape[-rank - 1 : -1] if channels_last else W_shape[-rank:]
            sympy_shape[3 if channels_last else 1] = W_shape[0]
        else:
            W_shape = None
            kernel_shape = get_attribute(node, "kernel_shape")
            rank = len(kernel_shape)

        assert len(sympy_shape) == rank + 2

        # only need to symbolic shape inference if input has symbolic dims in spatial axes
        spatial_shape = sympy_shape[-rank - 1 : -1] if channels_last else sympy_shape[-rank:]
        is_symbolic_dims = [not is_literal(i) for i in spatial_shape]

        if not any(is_symbolic_dims):
            shape = get_shape_from_value_info(self.known_vi_[node.output[0]])
            if len(shape) > 0:
                assert len(sympy_shape) == len(shape)
                if channels_last:
                    sympy_shape[-rank - 1 : -1] = [sympy.Integer(d) for d in shape[-rank - 1 : -1]]
                else:
                    sympy_shape[-rank:] = [sympy.Integer(d) for d in shape[-rank:]]
                return sympy_shape

        dilations = get_attribute(node, "dilations", [1] * rank)
        strides = get_attribute(node, "strides", [1] * rank)
        effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]
        pads = get_attribute(node, "pads")
        if pads is None:
            pads = [0] * (2 * rank)
            auto_pad = get_attribute(node, "auto_pad", b"NOTSET").decode("utf-8")
            if auto_pad not in {"VALID", "NOTSET"}:
                try:
                    residual = [sympy.Mod(d, s) for d, s in zip(sympy_shape[-rank:], strides)]
                    total_pads = [
                        max(0, (k - s) if r == 0 else (k - r))
                        for k, s, r in zip(effective_kernel_shape, strides, residual)
                    ]
                except TypeError:  # sympy may throw TypeError: cannot determine truth value of Relational
                    total_pads = [
                        max(0, (k - s)) for k, s in zip(effective_kernel_shape, strides)
                    ]  # assuming no residual if sympy throws error
            elif auto_pad == "VALID":
                total_pads = []
            else:
                total_pads = [0] * rank
        else:
            assert len(pads) == 2 * rank
            total_pads = [p1 + p2 for p1, p2 in zip(pads[:rank], pads[rank:])]

        ceil_mode = get_attribute(node, "ceil_mode", 0)
        for i in range(rank):
            effective_input_size = sympy_shape[-rank + i + (-1 if channels_last else 0)]
            if len(total_pads) > 0:
                effective_input_size = effective_input_size + total_pads[i]
            if ceil_mode:
                strided_kernel_positions = sympy.ceiling(
                    (effective_input_size - effective_kernel_shape[i]) / strides[i]
                )
            else:
                strided_kernel_positions = FloorDiv((effective_input_size - effective_kernel_shape[i]), strides[i])
            sympy_shape[-rank + i + (-1 if channels_last else 0)] = strided_kernel_positions + 1
        return sympy_shape

    def _check_merged_dims(self, dims, allow_broadcast=True):
        """Checks merged dimensions for consistency, optionally allowing broadcasting."""
        if allow_broadcast:
            dims = [d for d in dims if not (is_literal(d) and int(d) <= 1)]
        if any(d != dims[0] for d in dims):
            self._add_suggested_merge(dims, apply=True)

    def _compute_matmul_shape(self, node, output_dtype=None):
        """Compute the output shape for a matrix multiplication operation based on input shapes and optionally infer the
        output data type.
        """
        lhs_shape = self._get_shape(node, 0)
        rhs_shape = self._get_shape(node, 1)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        lhs_reduce_dim = 0
        rhs_reduce_dim = 0
        assert lhs_rank > 0 and rhs_rank > 0
        if lhs_rank == 1 and rhs_rank == 1:
            new_shape = []
        elif lhs_rank == 1:
            rhs_reduce_dim = -2
            new_shape = [*rhs_shape[:rhs_reduce_dim], rhs_shape[-1]]
        elif rhs_rank == 1:
            lhs_reduce_dim = -1
            new_shape = lhs_shape[:lhs_reduce_dim]
        else:
            lhs_reduce_dim = -1
            rhs_reduce_dim = -2
            new_shape = [
                *self._broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2]),
                lhs_shape[-2],
                rhs_shape[-1],
            ]
        # merge reduce dim
        self._check_merged_dims(
            [lhs_shape[lhs_reduce_dim], rhs_shape[rhs_reduce_dim]],
            allow_broadcast=False,
        )
        if output_dtype is None:
            # infer output_dtype from input type when not specified
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _fuse_tensor_type(self, node, out_idx, dst_type, src_type):
        """Update dst_tensor_type to be compatible with src_tensor_type when dimension mismatches."""
        dst_tensor_type = (
            dst_type.sequence_type.elem_type.tensor_type if is_sequence(dst_type) else dst_type.tensor_type
        )
        src_tensor_type = (
            src_type.sequence_type.elem_type.tensor_type if is_sequence(src_type) else src_type.tensor_type
        )
        if dst_tensor_type.elem_type != src_tensor_type.elem_type:
            node_id = node.name or node.op_type
            raise ValueError(
                f"For node {node_id}, dst_tensor_type.elem_type != src_tensor_type.elem_type: "
                f"{onnx.onnx_pb.TensorProto.DataType.Name(dst_tensor_type.elem_type)} vs "
                f"{onnx.onnx_pb.TensorProto.DataType.Name(src_tensor_type.elem_type)}"
            )
        if dst_tensor_type.HasField("shape"):
            for di, ds in enumerate(zip(dst_tensor_type.shape.dim, src_tensor_type.shape.dim)):
                if ds[0] != ds[1]:
                    # create a new symbolic dimension for node/out_idx/mismatch dim id in dst_tensor_type for tensor_type
                    # for sequence_type, clear the dimension
                    new_dim = onnx.TensorShapeProto.Dimension()
                    if not is_sequence(dst_type):
                        new_dim.dim_param = str(self._new_symbolic_dim_from_output(node, out_idx, di))
                    dst_tensor_type.shape.dim[di].CopyFrom(new_dim)
        else:
            dst_tensor_type.CopyFrom(src_tensor_type)

    def _infer_ArrayFeatureExtractor(self, node):
        """Infer and update the shape and type information for the ArrayFeatureExtractor node using input data and
        indices shapes.
        """
        data_shape = self._get_shape(node, 0)
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape[:-1] + indices_shape,
            )
        )

    def _infer_symbolic_compute_ops(self, node):
        """Handles symbolic computation operations for given node based on predefined functions."""
        funcs = {
            "Add": lambda l: l[0] + l[1],  # noqa: E741
            "Div": lambda l: (
                int(l[0] // l[1]) if isinstance(l[0] // l[1], float) else l[0] // l[1]
            ),  # integer div in sympy
            "Equal": lambda l: l[0] == l[1],  # noqa: E741
            "Floor": lambda l: sympy.floor(l[0]),  # noqa: E741
            "Max": lambda l: (
                l[1]
                if is_literal(l[0]) and int(l[0]) < -self.int_max_
                else (l[0] if is_literal(l[1]) and int(l[1]) < -self.int_max_ else sympy.Max(l[0], l[1]))
            ),
            "Min": lambda l: (
                l[1]
                if is_literal(l[0]) and int(l[0]) > self.int_max_
                else (l[0] if is_literal(l[1]) and int(l[1]) > self.int_max_ else sympy.Min(l[0], l[1]))
            ),
            "Mul": lambda l: (int(l[0] * l[1]) if isinstance(l[0] * l[1], float) else l[0] * l[1]),  # noqa: E741
            "Sub": lambda l: l[0] - l[1],  # noqa: E741
            "Where": lambda l: l[1] if l[0] else l[2],  # noqa: E741
            "Neg": lambda l: -l[0],  # noqa: E741
        }
        assert node.op_type in funcs
        self._compute_on_sympy_data(node, funcs[node.op_type])

    def _infer_Cast(self, node):
        """Pass node's data to SymPy representation without alteration."""
        self._pass_on_sympy_data(node)

    def _infer_CategoryMapper(self, node):
        """Infer and set output tensor type for ONNX CategoryMapper nodes based on input tensor type."""
        input_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        if input_type == onnx.TensorProto.STRING:
            output_type = onnx.TensorProto.INT64
        else:
            output_type = onnx.TensorProto.STRING
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_type, self._get_shape(node, 0)))

    def _infer_Compress(self, node):
        """Infer the output shape and type for the Compress operation based on input shape and axis attribute."""
        input_shape = self._get_shape(node, 0)
        # create a new symbolic dimension for Compress output
        compress_len = str(self._new_symbolic_dim_from_output(node))
        axis = get_attribute(node, "axis")
        if axis is None:
            # when axis is not specified, input is flattened before compress so output is 1D
            output_shape = [compress_len]
        else:
            output_shape = input_shape
            output_shape[handle_negative_axis(axis, len(input_shape))] = compress_len
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

    def _infer_Concat(self, node):
        """Infer the output shape and type for the Concat operation based on input node values."""
        if any(i in self.sympy_data_ or i in self.initializers_ for i in node.input):
            values = self._get_int_or_float_values(node)
            if all(v is not None for v in values):
                assert get_attribute(node, "axis") == 0
                self.sympy_data_[node.output[0]] = []
                for i in range(len(node.input)):
                    value = values[i]
                    if isinstance(value, list):
                        self.sympy_data_[node.output[0]].extend(value)
                    else:
                        self.sympy_data_[node.output[0]].append(value)

        sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis"), len(sympy_shape))
        for i_idx in range(1, len(node.input)):
            input_shape = self._get_sympy_shape(node, i_idx)
            if input_shape:
                sympy_shape[axis] = sympy_shape[axis] + input_shape[axis]
        self._update_computed_dims(sympy_shape)
        # merge symbolic dims for non-concat axes
        for d in range(len(sympy_shape)):
            if d == axis:
                continue
            dims = [self._get_shape(node, i_idx)[d] for i_idx in range(len(node.input)) if self._get_shape(node, i_idx)]
            if all(d == dims[0] for d in dims):
                continue
            merged = self._merge_symbols(dims)
            if type(merged) == str:  # noqa: E721
                sympy_shape[d] = self.symbolic_dims_[merged] if merged else None
            else:
                sympy_shape[d] = merged
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_ConcatFromSequence(self, node):
        """Infers the output shape and type info for ConcatFromSequence operation in a computational graph node."""
        seq_shape = self._get_shape(node, 0)
        new_axis = 1 if get_attribute(node, "new_axis") else 0
        axis = handle_negative_axis(get_attribute(node, "axis"), len(seq_shape) + new_axis)
        concat_dim = str(self._new_symbolic_dim_from_output(node, 0, axis))
        new_shape = seq_shape
        if new_axis:
            new_shape = [*seq_shape[:axis], concat_dim, *seq_shape[axis:]]
        else:
            new_shape[axis] = concat_dim
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.sequence_type.elem_type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_Constant(self, node):
        """Infer the constant value for a given node and store it in sympy_data_."""
        t = get_attribute(node, "value")
        # Lower constant nodes to initializers
        t.name = node.output[0]
        self.initializers_[node.output[0]] = t
        self.sympy_data_[node.output[0]] = numpy_helper.to_array(t)

    def _infer_ConstantOfShape(self, node):
        """Infer the constant tensor of a given shape from a node and update sympy_data_."""
        sympy_shape = self._get_int_or_float_values(node)[0]
        vi = self.known_vi_[node.output[0]]
        if sympy_shape is not None:
            if type(sympy_shape) != list:  # noqa: E721
                sympy_shape = [sympy_shape]
            self._update_computed_dims(sympy_shape)
            # update sympy data if output type is int, and shape is known
            if vi.type.tensor_type.elem_type == onnx.TensorProto.INT64 and all(is_literal(x) for x in sympy_shape):
                self.sympy_data_[node.output[0]] = np.ones(
                    [int(x) for x in sympy_shape], dtype=np.int64
                ) * numpy_helper.to_array(get_attribute(node, "value", 0))
        else:
            # create new dynamic shape
            # note input0 is a 1D vector of shape, the new symbolic shape has the rank of the shape vector length
            sympy_shape = self._new_symbolic_shape(self._get_shape(node, 0)[0], node)

        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_Conv(self, node):
        """Infers the shape of the output tensor for a convolution operation node and updates the known value info."""
        sympy_shape = self._compute_conv_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_NhwcConv(self, node):
        """Infer the shape of the output tensor for a convolution operation with NHWC format."""
        sympy_shape = self._compute_conv_pool_shape(node, channels_last=True)
        self._update_computed_dims(sympy_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_DequantizeLinear(self, node):
        """Infer output type and shape for the DequantizeLinear node based on input 1's scale data type."""
        output_dtype = self.known_vi_[node.input[1]].type.tensor_type.elem_type

        # Get the output shape from the first input.
        output_shape = self._get_shape(node, 0)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_QuantizeLinear(self, node):
        """Infer the output data type and shape for the QuantizeLinear ONNX node, defaulting to uint8 if not
        specified.
        """
        # Otherwise, default to uint8
        output_dtype = onnx.TensorProto.UINT8
        if len(node.input) > 2 and node.input[2]:
            output_dtype = self.known_vi_[node.input[2]].type.tensor_type.elem_type

        # Get the output shape from the first input.
        output_shape = self._get_shape(node, 0)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_Einsum(self, node):
        """Infer the output shape and type for the Einsum operation as per ONNX standards: https://github.com/onnx/onnx/blob/v1.18.0/onnx/defs/math/defs.cc#L2472."""
        equation = get_attribute(node, "equation")
        equation = equation.replace(b" ", b"")
        mid_index = equation.find(b"->")
        left_equation = equation[:mid_index] if mid_index != -1 else equation

        num_operands = 0
        num_ellipsis = 0
        num_ellipsis_indices = 0
        num_labels = 0
        ellipsis_flag = True
        dims_value = []
        ellipsis_dims_value = []

        label_maps = {}
        repeated_labels = set()

        terms = left_equation.split(b",")
        for term in terms:
            ellipsis_index = term.find(b"...")
            shape = self._get_shape(node, num_operands)
            rank = len(shape)
            ellipsis_dims = 0
            term_size = 0
            num_illegal_char = 0

            for i in range(len(term)):
                if term[i] != 46:
                    term_size = term_size + 1

            index = 0
            while index < len(term):
                if index == ellipsis_index:
                    ellipsis_dims = rank - term_size
                    if ellipsis_flag:
                        ellipsis_flag = False
                        for i in range(ellipsis_dims):
                            ellipsis_dims_value.append(shape[index + i - num_illegal_char])
                    else:
                        for i in range(ellipsis_dims):
                            shape_dim = shape[index + i - num_illegal_char]
                            current_dim = ellipsis_dims_value[i]
                            ellipsis_dims_value[i] = max(current_dim, shape_dim)

                    num_illegal_char += 3
                    index += 3  # Skip all three characters in '...'
                    continue

                elif term[index] == 46:  # ASCII for '.'
                    num_illegal_char += 1
                    index += 1
                    continue

                char = term[index]
                if char not in label_maps:
                    label_maps[char] = num_labels
                    dims_value.append(shape[index + ellipsis_dims - num_illegal_char])
                    num_labels += 1
                else:
                    repeated_labels.add(char)

                index += 1

            if ellipsis_index != -1:
                # If there is an ellipsis, the number of dimensions it represents
                # must be total dim - letter dimensions
                if num_ellipsis == 0:
                    if rank < term_size:
                        raise ValueError("Ellipsis represents incompatible dimensions.")
                    num_ellipsis_indices = rank - term_size
                else:
                    if num_ellipsis_indices != rank - term_size:
                        raise ValueError("Ellipsis represents incompatible dimensions.")
                num_ellipsis += 1
            else:
                if rank != term_size:
                    raise ValueError("Rank of input ", num_operands, " does not match the equation indices.")
            num_operands += 1

        new_sympy_shape = []
        from collections import OrderedDict

        OrderedDict()
        if mid_index != -1:
            right_equation = equation[mid_index + 2 :]
            right_ellipsis_index = right_equation.find(b"...")
            if right_ellipsis_index != -1:
                for i in range(num_ellipsis_indices):
                    new_sympy_shape.append(ellipsis_dims_value[i])
            for c in right_equation:
                if c != 46:  # c != b'.'
                    new_sympy_shape.append(dims_value[label_maps[c]])
        else:
            for i in range(num_ellipsis_indices):
                new_sympy_shape.append(ellipsis_dims_value[i])
            for label, idx in label_maps.items():
                if label not in repeated_labels:
                    new_sympy_shape.append(dims_value[idx])

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_sympy_shape))

    def _infer_Expand(self, node):
        """Infers and updates the output shape for the Expand operation based on broadcasted input shapes."""
        expand_to_shape = as_list(self._try_get_value(node, 1), keep_none=True)
        if expand_to_shape is not None:
            # new_shape's dim can come from shape value
            self._update_computed_dims(expand_to_shape)
            shape = self._get_shape(node, 0)
            new_shape = self._broadcast_shapes(shape, get_shape_from_sympy_shape(expand_to_shape))
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    new_shape,
                )
            )

    def _infer_Gather(self, node):
        """Infer the output shape of the Gather operation based on the input data and indices shapes."""
        data_shape = self._get_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", 0), len(data_shape))
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape[:axis] + indices_shape + data_shape[axis + 1 :],
            )
        )
        # for 1D input, do some sympy compute
        if node.input[0] in self.sympy_data_ and len(data_shape) == 1 and get_attribute(node, "axis", 0) == 0:
            idx = self._try_get_value(node, 1)
            if idx is not None:
                data = self.sympy_data_[node.input[0]]
                if type(data) == list:  # noqa: E721
                    if type(idx) == np.ndarray and len(idx.shape) == 1:
                        self.sympy_data_[node.output[0]] = [data[int(i)] for i in idx]
                    else:
                        self.sympy_data_[node.output[0]] = data[int(idx)]
                else:
                    assert idx in {0, -1}
                    self.sympy_data_[node.output[0]] = data

    def _infer_GatherElements(self, node):
        """Infers the output shape and type for the GatherElements node based on input tensors and updates the node's
        value information.
        """
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                indices_shape,
            )
        )

    def _infer_GatherND(self, node):
        """Infers the output shape and type for the GatherND operation based on input data and indices shapes."""
        data_shape = self._get_shape(node, 0)
        data_rank = len(data_shape)
        indices_shape = self._get_shape(node, 1)
        len(indices_shape)
        last_index_dimension = indices_shape[-1]
        batch_dims = get_attribute(node, "batch_dims", 0)
        assert (
            is_literal(last_index_dimension)
            and is_literal(batch_dims)
            and (batch_dims + last_index_dimension) <= data_rank
        )
        new_shape = indices_shape[:-1] + data_shape[batch_dims + last_index_dimension :]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_If(self, node):
        """Infer the output shape for an If node, handling constant conditions to ensure shape consistency between
        branches.
        """
        subgraphs = [
            get_attribute(node, "then_branch"),
            get_attribute(node, "else_branch"),
        ]

        for i_sub, subgraph in enumerate(subgraphs):
            subgraph_infer = self._onnx_infer_subgraph(node, subgraph, use_node_input=False)
            for i_out in range(len(node.output)):
                vi = self.known_vi_[node.output[i_out]]
                if i_sub == 0:
                    vi.CopyFrom(subgraph.output[i_out])
                    vi.name = node.output[i_out]
                else:
                    self._fuse_tensor_type(node, i_out, vi.type, subgraph.output[i_out].type)
                # fixme
                if (
                    cond is not None
                    and i_sub == (0 if as_scalar(cond) > 0 else 1)
                    and subgraph.output[i_out].name in subgraph_infer.sympy_data_
                ):
                    self.sympy_data_[vi.name] = subgraph_infer.sympy_data_[subgraph.output[i_out].name]

    def _infer_Loop(self, node):
        """Infer the shape and type of variables produced by the 'Loop' operation in an ONNX graph."""
        subgraph = get_attribute(node, "body")
        assert len(subgraph.input) == len(node.input)
        num_loop_carried = len(node.input) - 2  # minus the length and initial loop condition
        # when sequence_type is used as loop carried input
        # needs to run subgraph infer twice if the tensor shape in sequence contains None
        for i, si in enumerate(subgraph.input):
            si_name = si.name
            si.CopyFrom(self.known_vi_[node.input[i]])
            si.name = si_name

        self._onnx_infer_subgraph(node, subgraph)

        # check subgraph input/output for shape changes in loop carried variables
        # for tensor_type, create new symbolic dim when changing, i.e., output = Concat(input, a)
        # for sequence_type, propagate from output to input
        need_second_infer = False
        for i_out in range(1, num_loop_carried + 1):
            so = subgraph.output[i_out]
            so_shape = get_shape_from_value_info(so)
            if is_sequence(so.type):
                if so_shape and None in so_shape:
                    # copy shape from output to input
                    # note that loop input is [loop_len, cond, input_0, input_1, ...]
                    # while loop output is [cond, output_0, output_1, ...]
                    subgraph.input[i_out + 1].type.sequence_type.elem_type.CopyFrom(so.type.sequence_type.elem_type)
                    need_second_infer = True
            else:
                si = subgraph.input[i_out + 1]
                si_shape = get_shape_from_value_info(si)
                for di, dims in enumerate(zip(si_shape, so_shape)):
                    if dims[0] != dims[1]:
                        new_dim = onnx.TensorShapeProto.Dimension()
                        new_dim.dim_param = str(self._new_symbolic_dim_from_output(node, i_out, di))
                        si.type.tensor_type.shape.dim[di].CopyFrom(new_dim)
                        so.type.tensor_type.shape.dim[di].CopyFrom(new_dim)
                        need_second_infer = True

        if need_second_infer:
            if self.verbose_ > 2:
                logger.debug(
                    f"Rerun Loop: {node.name}({node.output[0]}...), because of sequence in loop carried variables"
                )
            self._onnx_infer_subgraph(node, subgraph, inc_subgraph_id=False)

        # create a new symbolic dimension for iteration dependent dimension
        loop_iter_dim = str(self._new_symbolic_dim_from_output(node))
        for i in range(len(node.output)):
            vi = self.known_vi_[node.output[i]]
            vi.CopyFrom(subgraph.output[i + 1])  # first subgraph output is condition, not in node output
            if i >= num_loop_carried:
                assert not is_sequence(vi.type)  # TODO: handle loop accumulation in sequence_type
                subgraph_vi_dim = subgraph.output[i + 1].type.tensor_type.shape.dim
                vi.type.tensor_type.shape.ClearField("dim")
                vi_dim = vi.type.tensor_type.shape.dim
                vi_dim.add().dim_param = loop_iter_dim
                vi_dim.extend(list(subgraph_vi_dim))
            vi.name = node.output[i]

    def _infer_MatMul(self, node):
        """Infer the output shape of a matrix multiplication node."""
        self._compute_matmul_shape(node)

    def _infer_MatMulInteger(self, node):
        """Infer the output shape of an integer matrix multiplication node."""
        self._compute_matmul_shape(node, onnx.TensorProto.INT32)

    def _infer_NonMaxSuppression(self, node):
        """Infer the output shape of a NonMaxSuppression node and update the value info."""
        selected = str(self._new_symbolic_dim_from_output(node))
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [selected, 3]))

    def _infer_NonZero(self, node):
        """Infer the output shape of a NonZero node and update the value info."""
        input_rank = self._get_shape_rank(node, 0)
        # create a new symbolic dimension for NonZero output
        nz_len = str(self._new_symbolic_dim_from_output(node, 0, 1))
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], vi.type.tensor_type.elem_type, [input_rank, nz_len]))

    def _infer_OneHot(self, node):
        """Infer the shape and type of the output tensor for the OneHot node operation."""
        sympy_shape = self._get_sympy_shape(node, 0)
        depth = self._try_get_value(node, 1)
        axis = get_attribute(node, "axis", -1)
        axis = handle_negative_axis(axis, len(sympy_shape) + 1)
        new_shape = get_shape_from_sympy_shape(
            [
                *sympy_shape[:axis],
                depth if is_literal(depth) else self._new_symbolic_dim_from_output(node),
                *sympy_shape[axis:],
            ]
        )
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[2]].type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_Pad(self, node):
        """Infers the output shape and type for the Pad operation based on ONNX node attributes and opset version."""
        if get_opset(self.out_mp_) <= 10:
            pads = get_attribute(node, "pads")
        else:
            pads = self._try_get_value(node, 1)

        sympy_shape = self._get_sympy_shape(node, 0)
        rank = len(sympy_shape)

        if pads is not None:
            assert len(pads) == 2 * rank
            new_sympy_shape = [
                d + pad_up + pad_down for d, pad_up, pad_down in zip(sympy_shape, pads[:rank], pads[rank:])
            ]
            self._update_computed_dims(new_sympy_shape)
        else:
            # dynamic pads, create new symbolic dimensions
            new_sympy_shape = self._new_symbolic_shape(rank, node)
        output_tp = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[0], output_tp, get_shape_from_sympy_shape(new_sympy_shape))
        )

    def _infer_Pool(self, node):
        """Infer and update dimensions for pooling layers based on the input node."""
        sympy_shape = self._compute_conv_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        for o in node.output:
            if not o:
                continue
            vi = self.known_vi_[o]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    o,
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(sympy_shape),
                )
            )

    def _infer_aten_bitwise_or(self, node):
        """Infers the output shape for Aten bitwise OR operation based on input node shapes."""
        shape0 = self._get_shape(node, 0)
        shape1 = self._get_shape(node, 1)
        new_shape = self._broadcast_shapes(shape0, shape1)
        t0 = self.known_vi_[node.input[0]]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], t0.type.tensor_type.elem_type, new_shape))

    def _infer_aten_diagonal(self, node):
        """Infers the shape of the diagonal of a tensor given a node, offset, and dimensions."""
        sympy_shape = self._get_sympy_shape(node, 0)
        rank = len(sympy_shape)
        offset = self._try_get_value(node, 1)
        dim1 = self._try_get_value(node, 2)
        dim2 = self._try_get_value(node, 3)

        assert offset is not None and dim1 is not None and dim2 is not None
        dim1 = handle_negative_axis(dim1, rank)
        dim2 = handle_negative_axis(dim2, rank)

        new_shape = [val for dim, val in enumerate(sympy_shape) if dim not in {dim1, dim2}]
        shape1 = sympy_shape[dim1]
        shape2 = sympy_shape[dim2]
        if offset >= 0:
            diag_shape = sympy.Max(0, sympy.Min(shape1, shape2 - offset))
        else:
            diag_shape = sympy.Max(0, sympy.Min(shape1 + offset, shape2))
        new_shape.append(diag_shape)

        if node.output[0]:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_shape),
                )
            )

    def _infer_aten_multinomial(self, node):
        """Infers the output shape and type for the PyTorch multinomial operation in an ONNX graph node."""
        sympy_shape = self._get_sympy_shape(node, 0)
        rank = len(sympy_shape)
        assert rank in {1, 2}
        num_samples = self._try_get_value(node, 1)
        di = rank - 1
        last_dim = num_samples or str(self._new_symbolic_dim_from_output(node, 0, di))
        output_shape = [*sympy_shape[:-1], last_dim]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                onnx.TensorProto.INT64,
                get_shape_from_sympy_shape(output_shape),
            )
        )

    def _infer_aten_pool2d(self, node):
        """Infer the output shape of a 2D pooling operation in an ONNX graph node."""
        sympy_shape = self._get_sympy_shape(node, 0)
        assert len(sympy_shape) == 4
        sympy_shape[-2:] = [self._new_symbolic_dim_from_output(node, 0, i) for i in {2, 3}]
        self._update_computed_dims(sympy_shape)
        for i, o in enumerate(node.output):
            if not o:
                continue
            vi = self.known_vi_[o]
            elem_type = onnx.TensorProto.INT64 if i == 1 else self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(o, elem_type, get_shape_from_sympy_shape(sympy_shape)))

    def _infer_aten_minmax(self, node):
        """Infer the output shape and type for the ATen MinMax operation in an ONNX node."""
        vi = self.known_vi_[node.output[0]]
        if len(node.input) == 1:
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    [],
                )
            )
        else:
            assert len(node.input) == 3
            keepdim = self._try_get_value(node, 2)
            assert keepdim is not None  # can only handle known keepdim case.
            dim = self._try_get_value(node, 1)
            if dim is None:
                rank = self._get_shape_rank(node, 0)
                output_shape = self._new_symbolic_shape(rank if keepdim else rank - 1, node)
            else:
                shape = self._get_sympy_shape(node, 0)
                dim = handle_negative_axis(dim, len(shape))
                output_shape = shape[:dim]
                if keepdim:
                    output_shape += [1]
                output_shape += shape[dim + 1 :]

            output_shape = get_shape_from_sympy_shape(output_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    output_shape,
                )
            )
            vi1 = self.known_vi_[node.output[1]]
            vi1.CopyFrom(helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT64, output_shape))

    def _infer_aten_unfold(self, node):
        """Infer the tensor shape for the 'aten::unfold' operation based on input shape and parameters dimension, size, and step."""
        sympy_shape = self._get_sympy_shape(node, 0)
        dimension = self._try_get_value(node, 1)
        size = self._try_get_value(node, 2)
        step = self._try_get_value(node, 3)
        if dimension is not None and size is not None and step is not None:
            assert dimension < len(sympy_shape)
            sympy_shape[dimension] = (sympy_shape[dimension] - size) // step + 1
            sympy_shape.append(size)
        else:
            rank = len(sympy_shape)
            sympy_shape = self._new_symbolic_shape(rank + 1, node)
        self._update_computed_dims(sympy_shape)
        if node.output[0]:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(sympy_shape),
                )
            )

    def _infer_aten_argmax(self, node):
        """Infers the output shape for the ONNX ATen argmax operation."""
        new_shape = None
        if not node.input[1]:
            # The argmax of the flattened input is returned.
            new_shape = []
        else:
            dim = self._try_get_value(node, 1)
            keepdim = self._try_get_value(node, 2)
            if keepdim is not None:
                sympy_shape = self._get_sympy_shape(node, 0)
                if dim is not None:
                    dim = handle_negative_axis(dim, len(sympy_shape))
                    if keepdim:
                        sympy_shape[dim] = 1
                    else:
                        del sympy_shape[dim]
                else:
                    rank = len(sympy_shape)
                    sympy_shape = self._new_symbolic_shape(rank if keepdim else rank - 1, node)
                self._update_computed_dims(sympy_shape)
                new_shape = get_shape_from_sympy_shape(sympy_shape)
        if node.output[0] and new_shape is not None:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, new_shape))

    def _infer_aten_group_norm(self, node):
        """Infers the output shapes and types for the ATen GroupNorm operation based on the provided node
        information.
        """
        self._propagate_shape_and_type(node)
        input_shape = self._get_shape(node, 0)
        N = input_shape[0] if input_shape is not None and len(input_shape) != 0 else None
        group = self._try_get_value(node, 6)
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        for i in {1, 2}:
            if node.output[i]:
                vi = self.known_vi_[node.output[i]]
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[i],
                        output_dtype,
                        [
                            (N if N is not None else str(self._new_symbolic_dim_from_output(node, i, 0))),
                            (
                                as_scalar(group)
                                if group is not None
                                else str(self._new_symbolic_dim_from_output(node, i, 1))
                            ),
                        ],
                    )
                )

    def _infer_aten_upsample(self, node):
        """Infers the output shape for an aten::upsample operation based on the input shape and specified upsampling parameters."""
        new_shape = None
        input_shape = self._get_shape(node, 0)
        if input_shape is not None:
            new_shape = input_shape[:2]
            output_size = self._try_get_value(node, 1)
            if output_size is not None:
                new_shape += [dim_size.item() if type(dim_size) == np.int64 else dim_size for dim_size in output_size]
            else:
                rank = len(input_shape)
                new_shape += [str(self._new_symbolic_dim_from_output(node, 0, i)) for i in range(2, rank)]
        if node.output[0] and new_shape is not None:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_BatchNormalization(self, node):
        """Propagate the shape and type information for the BatchNormalization node."""
        self._propagate_shape_and_type(node)

        # this works for opsets < 14 and 14 since we check i < len(node.output) in the loop
        for i in {1, 2, 3, 4}:
            if i < len(node.output) and node.output[i]:
                # all of these parameters have the same shape as the 1st input
                self._propagate_shape_and_type(node, input_index=1, output_index=i)

    def _infer_Range(self, node):
        """Infers the shape and type for Range nodes based on the provided start, limit, and delta values."""
        vi = self.known_vi_[node.output[0]]
        input_data = self._get_int_or_float_values(node)
        if all(i is not None for i in input_data):
            start = as_scalar(input_data[0])
            limit = as_scalar(input_data[1])
            delta = as_scalar(input_data[2])
            new_sympy_shape = [sympy.Max(sympy.ceiling((limit - start) / delta), 0)]
        else:
            new_sympy_shape = [self._new_symbolic_dim_from_output(node)]
        self._update_computed_dims(new_sympy_shape)
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

    def _infer_ReduceSum(self, node):
        """Infer output shape for ReduceSum operation based on input shape, axes, and keep_dims attribute."""
        keep_dims = get_attribute(node, "keepdims", 1)
        if get_opset(self.out_mp_) >= 13 and len(node.input) > 1:
            # ReduceSum changes axes to input[1] in opset 13
            axes = self._try_get_value(node, 1)
            vi = self.known_vi_[node.output[0]]
            if axes is None:
                assert keep_dims  # can only handle keep_dims==True when axes is unknown, by generating new ranks
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(self._new_symbolic_shape(self._get_shape_rank(node, 0), node)),
                    )
                )
            else:
                shape = self._get_shape(node, 0)
                output_shape = []
                axes = [handle_negative_axis(a, len(shape)) for a in axes]
                for i, d in enumerate(shape):
                    if i in axes:
                        if keep_dims:
                            output_shape.append(1)
                    else:
                        output_shape.append(d)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        output_shape,
                    )
                )

    def _infer_ReduceProd(self, node):
        """Infer the ReduceProd operation on a node, considering axes and keep dimensions attributes."""
        axes = get_attribute(node, "axes")
        keep_dims = get_attribute(node, "keepdims", 1)
        if keep_dims == 0 and axes == [0]:
            data = self._get_int_or_float_values(node)[0]
            if data is not None:
                self.sympy_data_[node.output[0]] = sympy_reduce_product(data)

    def _infer_RelativePositionBias(self, node):
        """Infers the relative position bias for a given ONNX node."""
        seq_len = self._try_get_value(node, 1)
        real_seq_len = self._try_get_value(node, 2)
        if seq_len is None or real_seq_len is None:
            return
        num_heads = self._get_sympy_shape(node, 0)[1]

        new_shape = [1, num_heads, str(seq_len), str(real_seq_len)]

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_Reshape(self, node):
        """Infer the output shape for the Reshape operation based on the provided input shape and reshape parameters."""
        shape_value = self._try_get_value(node, 1)
        vi = self.known_vi_[node.output[0]]
        if shape_value is None:
            shape_shape = self._get_shape(node, 1)
            assert len(shape_shape) == 1
            shape_rank = shape_shape[0]
            assert is_literal(shape_rank)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(self._new_symbolic_shape(shape_rank, node)),
                )
            )
        else:
            input_sympy_shape = self._get_sympy_shape(node, 0)
            total = 1
            for d in input_sympy_shape:
                total = total * d
            new_sympy_shape = []
            deferred_dim_idx = -1
            non_deferred_size = 1
            for i, d in enumerate(shape_value):
                if type(d) == sympy.Symbol or d != 0:
                    new_sympy_shape.append(d)
                else:
                    new_sympy_shape.append(input_sympy_shape[i])
                    non_deferred_size = non_deferred_size * input_sympy_shape[i]
                if d == -1:
                    deferred_dim_idx = i
                elif d != 0:
                    non_deferred_size = non_deferred_size * d

            assert new_sympy_shape.count(-1) < 2
            if -1 in new_sympy_shape:
                new_dim = total // non_deferred_size
                new_sympy_shape[deferred_dim_idx] = new_dim

            self._update_computed_dims(new_sympy_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )

        self._pass_on_sympy_data(node)

    def _infer_Resize(self, node):
        """Infers and updates the shape of the output tensor for a Resize node based on scales or sizes."""
        vi = self.known_vi_[node.output[0]]
        input_sympy_shape = self._get_sympy_shape(node, 0)
        if get_opset(self.out_mp_) <= 10:
            scales = self._try_get_value(node, 1)
            if scales is not None:
                new_sympy_shape = [sympy.simplify(sympy.floor(d * s)) for d, s in zip(input_sympy_shape, scales)]
                self._update_computed_dims(new_sympy_shape)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(new_sympy_shape),
                    )
                )
        else:
            roi = self._try_get_value(node, 1)
            scales = self._try_get_value(node, 2)
            sizes = self._try_get_value(node, 3)
            if sizes is not None:
                new_sympy_shape = [sympy.simplify(sympy.floor(s)) for s in sizes]
                self._update_computed_dims(new_sympy_shape)
            elif scales is not None:
                rank = len(scales)
                if get_attribute(node, "coordinate_transformation_mode") == "tf_crop_and_resize":
                    assert len(roi) == 2 * rank
                    roi_start = list(roi)[:rank]
                    roi_end = list(roi)[rank:]
                else:
                    roi_start = [0] * rank
                    roi_end = [1] * rank
                if isinstance(scales, np.ndarray):
                    scales = scales.tolist()
                else:
                    scales = list(scales)
                new_sympy_shape = [
                    (sympy.floor(d * (end - start) * scale))
                    for d, start, end, scale in zip(input_sympy_shape, roi_start, roi_end, scales)
                ]
                self._update_computed_dims(new_sympy_shape)
            else:
                new_sympy_shape = self._new_symbolic_shape(self._get_shape_rank(node, 0), node)

            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )

    def _infer_Scan(self, node):
        """Infer shape and type information for the ONNX 'Scan' operator node."""
        subgraph = get_attribute(node, "body")
        num_scan_inputs = get_attribute(node, "num_scan_inputs")
        scan_input_axes = get_attribute(node, "scan_input_axes", [0] * num_scan_inputs)
        num_scan_states = len(node.input) - num_scan_inputs
        scan_input_axes = [
            handle_negative_axis(ax, self._get_shape_rank(node, i + num_scan_states))
            for i, ax in enumerate(scan_input_axes)
        ]
        # We may have cases where the subgraph has optional inputs that appear in both subgraph's input and initializer,
        # but not in the node's input. In such cases, the input model might be invalid, but let's skip those optional inputs.
        assert len(subgraph.input) >= len(node.input)
        subgraph_inputs = subgraph.input[: len(node.input)]
        for i, si in enumerate(subgraph_inputs):
            subgraph_name = si.name
            si.CopyFrom(self.known_vi_[node.input[i]])
            if i >= num_scan_states:
                scan_input_dim = si.type.tensor_type.shape.dim
                scan_input_dim.remove(scan_input_dim[scan_input_axes[i - num_scan_states]])
            si.name = subgraph_name
        self._onnx_infer_subgraph(node, subgraph)
        num_scan_outputs = len(node.output) - num_scan_states
        scan_output_axes = get_attribute(node, "scan_output_axes", [0] * num_scan_outputs)
        scan_input_dim = get_shape_from_type_proto(self.known_vi_[node.input[-1]].type)[scan_input_axes[-1]]
        for i, o in enumerate(node.output):
            vi = self.known_vi_[o]
            if i >= num_scan_states:
                shape = get_shape_from_type_proto(subgraph.output[i].type)
                new_dim = handle_negative_axis(scan_output_axes[i - num_scan_states], len(shape) + 1)
                shape = [*shape[:new_dim], scan_input_dim, *shape[new_dim:]]
                vi.CopyFrom(helper.make_tensor_value_info(o, subgraph.output[i].type.tensor_type.elem_type, shape))
            else:
                vi.CopyFrom(subgraph.output[i])
            vi.name = o

    def _infer_ScatterElements(self, node):
        """Infer the output shape and type for ScatterElements node and update known value infos."""
        data_shape = self._get_shape(node, 0)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape,
            )
        )

    def _infer_SequenceAt(self, node):
        """Infers the shape and type for the output of the 'SequenceAt' ONNX operation, handling symbolic dimensions if
        necessary.
        """
        seq_shape = self._get_shape(node, 0)
        if seq_shape is not None:
            vi = self.known_vi_[node.output[0]]
            for di, d in enumerate(seq_shape):
                if d is not None:
                    continue
                new_dim = onnx.TensorShapeProto.Dimension()
                new_dim.dim_param = str(self._new_symbolic_dim_from_output(node, 0, di))
                vi.type.tensor_type.shape.dim[di].CopyFrom(new_dim)

    def _infer_SequenceInsert(self, node):
        """Workaround ONNX's shape inference bug by inferring sequence insertion shapes and types for the provided
        node.
        """
        vi_seq = self.known_vi_[node.input[0]]
        vi_tensor = self.known_vi_[node.input[1]]
        vi_out_seq = self.known_vi_[node.output[0]]
        vi_out_seq.CopyFrom(vi_seq)
        vi_out_seq.name = node.output[0]
        self._fuse_tensor_type(node, 0, vi_out_seq.type, vi_tensor.type)

    def _infer_Shape(self, node):
        """Infers and sets the symbolic shape for the output node in the computation graph."""
        start = get_attribute(node, "start", 0)
        end = get_attribute(node, "end", None)

        full_sympy_shape = self._get_sympy_shape(node, 0)
        num_dims = len(full_sympy_shape)

        if start < 0:
            start = num_dims + start
        if end is None:
            end = num_dims
        elif end < 0:
            end = num_dims + end

        assert 0 <= start <= end <= num_dims, (
            f"reshape start/end invalid: start={start}, end={end}, total_dims={num_dims}"
        )

        target_sympy_shape = full_sympy_shape[start:end]
        self.sympy_data_[node.output[0]] = target_sympy_shape

    def _infer_Size(self, node):
        """Infers and sets the size of the output node by computing the product of its shape in the computation
        graph.
        """
        sympy_shape = self._get_sympy_shape(node, 0)
        self.sympy_data_[node.output[0]] = sympy_reduce_product(sympy_shape)
        self.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [])
        )

    def _infer_Slice(self, node):
        """Infer the shape and value information for the Slice node using SymPy and ONNX helper methods."""

        # even when the relation holds for both `a` and `b`.
        #
        # When given `expr` of form `min(a, b) + ...`, this function returns `[a + ..., b + ...]`,
        # so that we can prove inequalities for both expressions separately.
        #
        # If the number of `min(...)` subexpressions is not exactly one, this function just returns `[expr]`.
        def flatten_min(expr):
            """Returns a list with expressions split by min() for inequality proof or original expr if no single min()
            found.
            """
            assert isinstance(expr, sympy.Add), f"Expected a sum of two arguments, got {expr}"
            min_positions = [idx for idx in range(len(expr.args)) if isinstance(expr.args[idx], sympy.Min)]
            if len(min_positions) == 1:
                min_pos = min_positions[0]

                def replace_min_with_arg(arg_idx):
                    """Replace the sympy.Min() function at a specified position in a sympy.Add() expression with one of
                    its arguments.
                    """
                    replaced = list(expr.args)
                    assert isinstance(replaced[min_pos], sympy.Min), (
                        f"Expected a sympy.Min() at position {min_pos}, got {replaced[min_pos]}"
                    )
                    assert len(replaced[min_pos].args) == 2, (
                        f"Expected a sympy.Min() with exactly 2 arguments, got {replaced[min_pos]}"
                    )
                    replaced[min_pos] = replaced[min_pos].args[arg_idx]
                    return sympy.Add(*replaced)

                return [
                    replace_min_with_arg(0),
                    replace_min_with_arg(1),
                ]
            return [expr]

        def less_equal(x, y):
            """Returns True if x is less than or equal to y, otherwise False."""
            try:
                return x <= y
            except TypeError:
                pass
            try:
                return y >= x
            except TypeError:
                pass
            try:
                return -x >= -y
            except TypeError:
                pass
            try:
                return -y <= -x
            except TypeError:
                pass
            try:
                return y - x >= 0
            except TypeError:
                # the last attempt; this may raise TypeError
                return all(d >= 0 for d in flatten_min(y - x))

        def handle_negative_index(index, bound):
            """Normalizes a negative index to be in [0, bound)."""
            try:
                if not less_equal(0, index):
                    if is_literal(index) and index <= -self.int_max_:
                        # this case is handled separately
                        return index
                    return bound + index
            except TypeError:
                logger.warning(f"Cannot determine if {index} < 0")
            return index

        if get_opset(self.out_mp_) <= 9:
            axes = get_attribute(node, "axes")
            starts = get_attribute(node, "starts")
            ends = get_attribute(node, "ends")
            if not axes:
                axes = list(range(len(starts)))
            steps = [1] * len(axes)
        else:
            starts = as_list(self._try_get_value(node, 1), keep_none=True)
            ends = as_list(self._try_get_value(node, 2), keep_none=True)
            axes = self._try_get_value(node, 3)
            steps = self._try_get_value(node, 4)
            if axes is None and (starts is not None or ends is not None):
                axes = list(range(len(starts if starts is not None else ends)))
            if steps is None and (starts is not None or ends is not None):
                steps = [1] * len(starts if starts is not None else ends)
            axes = as_list(axes, keep_none=True)
            steps = as_list(steps, keep_none=True)

        new_sympy_shape = self._get_sympy_shape(node, 0)
        if starts is None or ends is None:
            if axes is None:
                for i in range(len(new_sympy_shape)):
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node, 0, i)
            else:
                new_sympy_shape = get_shape_from_sympy_shape(new_sympy_shape)
                for i in axes:
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node, 0, i)
        else:
            for i, s, e, t in zip(axes, starts, ends, steps):
                if is_literal(e):
                    e = handle_negative_index(e, new_sympy_shape[i])
                if is_literal(e):
                    if e >= self.int_max_:
                        e = new_sympy_shape[i]
                    elif e <= -self.int_max_:
                        e = 0 if s > 0 else -1
                    elif is_literal(new_sympy_shape[i]):
                        if e < 0:
                            e = max(0, e + new_sympy_shape[i])
                        e = min(e, new_sympy_shape[i])
                    else:
                        if e > 0:
                            e = (
                                sympy.Min(e, new_sympy_shape[i]) if e > 1 else e
                            )  # special case for slicing first to make computation easier
                else:
                    if is_literal(new_sympy_shape[i]):
                        if new_sympy_shape[i] < 0:
                            e = sympy.Min(e, new_sympy_shape[i])
                    else:
                        try:
                            if not less_equal(e, new_sympy_shape[i]):
                                e = new_sympy_shape[i]
                        except Exception:
                            if len(e.free_symbols) == 1:
                                if try_solve((e - new_sympy_shape[i]) >= 0, next(iter(e.free_symbols))) is None:
                                    logger.warning(
                                        f"Unable to determine if {e} <= {new_sympy_shape[i]}, treat as equal"
                                    )
                                    e = new_sympy_shape[i]
                            else:
                                logger.warning(f"Unable to determine if {e} <= {new_sympy_shape[i]}, treat as equal")
                                e = new_sympy_shape[i]

                s = handle_negative_index(s, new_sympy_shape[i])
                if is_literal(new_sympy_shape[i]) and is_literal(s):
                    s = max(0, min(s, new_sympy_shape[i]))

                new_sympy_shape[i] = sympy.simplify((e - s + t + (-1 if t > 0 else 1)) // t)

            self._update_computed_dims(new_sympy_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

        # handle sympy_data if needed, for slice in shape computation
        if (
            node.input[0] in self.sympy_data_
            and [0] == axes
            and starts is not None
            and len(starts) == 1
            and ends is not None
            and len(ends) == 1
            and steps is not None
            and len(steps) == 1
        ):
            input_sympy_data = self.sympy_data_[node.input[0]]
            if type(input_sympy_data) == list or (  # noqa: E721
                type(input_sympy_data) == np.array and len(input_sympy_data.shape) == 1
            ):
                self.sympy_data_[node.output[0]] = input_sympy_data[starts[0] : ends[0] : steps[0]]

    def _infer_SoftmaxCrossEntropyLoss(self, node):
        """Infer the softmax cross-entropy loss for a given node in the computation graph."""
        vi = self.known_vi_[node.output[0]]
        elem_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        # If output type is explicit specified in attribute, we use it as output tensor type.
        specified_output_type = get_attribute(node, "output_type", None)
        if specified_output_type is not None:
            elem_type = specified_output_type

        vi.type.tensor_type.elem_type = elem_type
        vi.type.tensor_type.shape.CopyFrom(onnx.TensorShapeProto())

        if len(node.output) > 1:
            data_shape = self._get_shape(node, 0)
            vi = self.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, elem_type, data_shape))

    def _infer_Split_Common(self, node, make_value_info_func):
        """Infers the output shape for the Split operator given an ONNX node and a function to create tensor value
        info.
        """
        input_sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", 0), len(input_sympy_shape))
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'split' are provided as attribute or via 2nd input
        if op_set < 13:
            split = get_attribute(node, "split")
            assert self._try_get_value(node, 1) is None
        else:
            split = self._try_get_value(node, 1)
            assert get_attribute(node, "split") is None

        if split is None:
            num_outputs = len(node.output)
            split = [input_sympy_shape[axis] / sympy.Integer(num_outputs)] * num_outputs
            self._update_computed_dims(split)
        else:
            split = [sympy.Integer(s) for s in split]

        for i_o in range(len(split)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(
                make_value_info_func(
                    node.output[i_o],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape([*input_sympy_shape[:axis], split[i_o], *input_sympy_shape[axis + 1 :]]),
                )
            )
            self.known_vi_[vi.name] = vi

    def _infer_Split(self, node):
        """Infers the output shapes and types for the Split operation node."""
        self._infer_Split_Common(node, helper.make_tensor_value_info)

    def _infer_SplitToSequence(self, node):
        """Infers the output shapes and types for the SplitToSequence operation node."""
        self._infer_Split_Common(node, helper.make_sequence_value_info)

    def _infer_Squeeze(self, node):
        """Infers the output shapes and types for the Squeeze operation node."""
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        if axes is None:
            # No axes have been provided (neither via attribute nor via input).
            # In this case the 'Shape' op should remove all axis with dimension 1.
            # For symbolic dimensions we guess they are !=1.
            output_shape = [s for s in input_shape if s != 1]
            if self.verbose_ > 0:
                symbolic_dimensions = [s for s in input_shape if type(s) != int]  # noqa: E721
                if symbolic_dimensions:
                    logger.debug(
                        f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                        f"Assuming the following dimensions are never equal to 1: {symbolic_dimensions}"
                    )
        else:
            axes = [handle_negative_axis(a, len(input_shape)) for a in axes]
            output_shape = []
            for i in range(len(input_shape)):
                if i not in axes:
                    output_shape.append(input_shape[i])
                else:
                    assert input_shape[i] == 1 or type(input_shape[i]) != int  # noqa: E721
                    if self.verbose_ > 0 and type(input_shape[i]) != int:  # noqa: E721
                        logger.debug(
                            f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                            f"Assuming the dimension '{input_shape[i]}' at index {i} of the input to be equal to 1."
                        )

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )
        self._pass_on_sympy_data(node)

    def _infer_Tile(self, node):
        """Infers the output shape for the Tile operation in a computation graph based on input shape and repeat
        values.
        """
        repeats_value = self._try_get_value(node, 1)
        new_sympy_shape = []
        if repeats_value is not None:
            input_sympy_shape = self._get_sympy_shape(node, 0)
            for i, d in enumerate(input_sympy_shape):
                new_dim = d * repeats_value[i]
                new_sympy_shape.append(new_dim)
            self._update_computed_dims(new_sympy_shape)
        else:
            new_sympy_shape = self._new_symbolic_shape(self._get_shape_rank(node, 0), node)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

    def _infer_TopK(self, node):
        """Infers the output shape for the TopK operation in an ONNX graph node based on input shape and specified
        axis.
        """
        rank = self._get_shape_rank(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", -1), rank)
        new_shape = self._get_shape(node, 0)

        if get_opset(self.out_mp_) <= 9:
            k = get_attribute(node, "k")
        else:
            k = self._get_int_or_float_values(node)[1]

        k = self._new_symbolic_dim_from_output(node) if k is None else as_scalar(k)
        if type(k) in {int, str}:
            new_shape[axis] = k
        else:
            new_sympy_shape = self._get_sympy_shape(node, 0)
            new_sympy_shape[axis] = k
            self._update_computed_dims(
                new_sympy_shape
            )  # note that TopK dim could be computed in sympy_data, so need to update computed_dims when it enters shape
            new_shape = get_shape_from_sympy_shape(new_sympy_shape)

        for i_o in range(len(node.output)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o], vi.type.tensor_type.elem_type, new_shape))

    def _infer_Transpose(self, node):
        """Infer and update the shape information for a Transpose node based on its input shape and permutation
        attributes.
        """
        if node.input[0] in self.sympy_data_:
            data_shape = self._get_shape(node, 0)
            perm = get_attribute(node, "perm", reversed(list(range(len(data_shape)))))
            input_data = self.sympy_data_[node.input[0]]
            self.sympy_data_[node.output[0]] = (
                np.transpose(np.array(input_data).reshape(*data_shape), axes=tuple(perm)).flatten().tolist()
            )

    def _infer_Unsqueeze(self, node):
        """Infers the output shape for the Unsqueeze operation based on the input shape and operator set."""
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        output_rank = len(input_shape) + len(axes)
        axes = [handle_negative_axis(a, output_rank) for a in axes]

        input_axis = 0
        output_shape = []
        for i in range(output_rank):
            if i in axes:
                output_shape.append(1)
            else:
                output_shape.append(input_shape[input_axis])
                input_axis += 1

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

        self._pass_on_sympy_data(node)

    def _infer_ZipMap(self, node):
        """Infer the type of keys for a ZipMap node based on its class labels attribute."""
        map_key_type = None
        if get_attribute(node, "classlabels_int64s") is not None:
            map_key_type = onnx.TensorProto.INT64
        elif get_attribute(node, "classlabels_strings") is not None:
            map_key_type = onnx.TensorProto.STRING

        assert map_key_type is not None
        new_vi = onnx.ValueInfoProto()
        new_vi.name = node.output[0]
        new_vi.type.sequence_type.elem_type.map_type.value_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        new_vi.type.sequence_type.elem_type.map_type.key_type = map_key_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(new_vi)

    def _infer_Attention(self, node):
        """Infer shape and data type for ONNX Attention node outputs given input shapes and attributes."""
        shape = self._get_shape(node, 0)
        shape_weights = self._get_shape(node, 1)
        shape_bias = self._try_get_shape(node, 2)
        if shape_bias is not None:
            assert len(shape_bias) == 1
        tripled_hidden_size = shape_bias[0] if shape_bias is not None else shape_weights[1]
        if shape and len(shape) == 3:
            qkv_hidden_sizes_attr = get_attribute(node, "qkv_hidden_sizes")
            if qkv_hidden_sizes_attr is not None:
                assert len(qkv_hidden_sizes_attr) == 3
                shape[2] = int(qkv_hidden_sizes_attr[2])
            elif isinstance(tripled_hidden_size, int):
                shape[2] = int(tripled_hidden_size / 3)
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))

            if len(node.output) > 1:
                # input shape: (batch_size, sequence_length, hidden_size)
                # past shape: (2, batch_size, num_heads, past_sequence_length, head_size)
                # mask shape: (batch_size, total_sequence_length) or (batch_size, sequence_length, total_sequence_length) or (batch_size, 1, max_seq_len, max_seq_len)
                # present shape: (2, batch_size, num_heads, total_sequence_length, head_size), where total_sequence_length=sequence_length+past_sequence_length
                input_shape = self._get_shape(node, 0)
                past_shape = self._get_shape(node, 4) if len(node.input) > 4 and node.input[4] else []
                mask_shape = self._get_shape(node, 3) if len(node.input) > 3 and node.input[3] else []

                if past_shape and len(past_shape) == 5:
                    if mask_shape and len(mask_shape) in {2, 3}:
                        past_shape[3] = mask_shape[-1]
                    elif input_shape and len(input_shape) == 3:
                        if isinstance(input_shape[1], int) and isinstance(past_shape[3], int):
                            past_shape[3] = input_shape[1] + past_shape[3]
                        else:
                            past_shape[3] = f"{past_shape[3]}+{input_shape[1]}"
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
                else:
                    num_heads = get_attribute(node, "num_heads")
                    head_size = input_shape[2] // num_heads
                    present_shape = [
                        2,
                        input_shape[0],
                        num_heads,
                        input_shape[1],
                        head_size,
                    ]
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))

    def _infer_GatedRelativePositionBias(self, node):
        """Infer the shape for gated relative position bias given the node attributes."""
        #   query_layer: (token_count, num_heads x head_size)
        #   token_offset: (batch_size, seq_len)
        # Otherwise:
        #   query_layer: (batch_size, seq_len, num_heads x head_size)
        #   token_offset: None
        # Output shape: (batch_size, num_heads, seq_len, seq_len)
        num_heads = get_attribute(node, "num_heads")

        token_offset_shape = self._try_get_shape(node, 6)
        if token_offset_shape is not None:
            output_shape = [
                token_offset_shape[0],
                num_heads,
                token_offset_shape[1],
                token_offset_shape[1],
            ]
        else:
            query_layer_shape = self._get_shape(node, 0)
            assert query_layer_shape is not None and len(query_layer_shape) == 3
            output_shape = [
                query_layer_shape[0],
                num_heads,
                query_layer_shape[1],
                query_layer_shape[1],
            ]

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_PackedAttention(self, node):
        """Infer shape and data type for PackedAttention nodes in a given computational graph."""
        shape = self._get_shape(node, 0)
        shape_weights = self._get_shape(node, 1)
        shape_bias = self._try_get_shape(node, 2)
        if shape_bias is not None:
            assert len(shape_bias) == 1
        tripled_hidden_size = shape_bias[0] if shape_bias is not None else shape_weights[1]
        if shape and len(shape) == 2:
            qkv_hidden_sizes_attr = get_attribute(node, "qkv_hidden_sizes")
            if qkv_hidden_sizes_attr is not None:
                assert len(qkv_hidden_sizes_attr) == 3
                shape[1] = int(qkv_hidden_sizes_attr[2])
            elif isinstance(tripled_hidden_size, int):
                shape[1] = int(tripled_hidden_size / 3)
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))

    def _infer_PackedMultiHeadAttention(self, node):
        """Infer the output shape for PackedMultiHeadAttention node in the computational graph."""
        shape_value = self._try_get_shape(node, 2)
        if shape_value is not None and len(shape_value) == 2:
            output_shape = shape_value
        else:
            shape_query = self._get_shape(node, 0)
            assert shape_query is not None and len(shape_query) == 4
            output_shape = [shape_query[0], shape_query[1] * shape_query[3]]

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_MultiScaleDeformableAttnTRT(self, node):
        shape_value = self._try_get_shape(node, 0)
        sampling_locations = self._try_get_shape(node, 3)
        output_shape = shape_value
        output_shape[1] = sampling_locations[1]
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_RemovePadding(self, node):
        """Infers the shape and data type for the output tensor after removing padding."""
        shape = self._get_shape(node, 0)
        if shape and len(shape) == 3:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, ["token_count", shape[2]]))

            vi_token_offset = self.known_vi_[node.output[1]]
            vi_token_offset.CopyFrom(
                helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT32, [shape[0], shape[1]])
            )

            vi_cumulated_seq_len = self.known_vi_[node.output[2]]
            vi_cumulated_seq_len.CopyFrom(
                helper.make_tensor_value_info(node.output[2], onnx.TensorProto.INT32, ["batch_size + 1"])
            )

            vi_max_seq_len = self.known_vi_[node.output[3]]
            vi_max_seq_len.CopyFrom(helper.make_tensor_value_info(node.output[3], onnx.TensorProto.INT32, [1]))

    def _infer_RestorePadding(self, node):
        """Infers the output shape and type for the RestorePadding operation."""
        shape_input = self._get_shape(node, 0)
        shape_token_offset = self._get_shape(node, 1)
        if shape_input and len(shape_input) == 2 and shape_token_offset and len(shape_token_offset) == 2:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]

            output_shape = [
                shape_token_offset[0],
                shape_token_offset[1],
                shape_input[1],
            ]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_BiasGelu(self, node):
        """Propagate shape and type information for BiasGelu node during inference."""
        self._propagate_shape_and_type(node)

    def _infer_MultiHeadAttention(self, node):
        """Propagate shape and type information for MultiHeadAttention node during inference."""
        # Q, K and V without packing:
        #   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
        #   Input 1 (key) has shape (batch_size, kv_sequence_length, hidden_size) or (batch_size, num_heads, kv_sequence_length, head_size)
        #   Input 2 (value) has shape (batch_size, kv_sequence_length, v_hidden_size) or (batch_size, num_heads, kv_sequence_length, head_size)
        # Packed KV:
        #   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
        #   Input 1 (batch_size, kv_sequence_length, num_heads, 2, head_size)
        #   Input 2  nullptr
        # Packed QKV:
        #   Input 0 (batch_size, sequence_length, num_heads, 3, head_size)
        #   Input 1  nullptr
        #   Input 2  nullptr

        query_shape = self._get_shape(node, 0)
        total_sequence_length = None
        output_dtype = None
        if query_shape is not None:
            if len(query_shape) == 3:
                key_shape = self._try_get_shape(node, 1)
                # By default, hidden size is same for Q/K/V. Only need check v_hidden_size when value is provided.
                output_shape = query_shape
                if key_shape is not None and len(key_shape) == 3:
                    value_shape = self._try_get_shape(node, 2)
                    if value_shape is not None and len(value_shape) == 3:
                        output_shape[2] = value_shape[2]
                    total_sequence_length = key_shape[1]

                output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                vi = self.known_vi_[node.output[0]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            elif len(query_shape) == 5:
                if isinstance(query_shape[2], int) and isinstance(query_shape[4], int):
                    output_shape = [
                        query_shape[0],
                        query_shape[1],
                        query_shape[2] * query_shape[4],
                    ]
                else:
                    output_shape = [
                        query_shape[0],
                        query_shape[1],
                        f"{query_shape[2]}*{query_shape[4]}",
                    ]

                total_sequence_length = query_shape[1]

                output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                vi = self.known_vi_[node.output[0]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            if len(node.output) > 1:
                batch_size = query_shape[0]
                num_heads = get_attribute(node, "num_heads")

                head_size = None
                if len(query_shape) == 3:
                    head_size = (
                        int(query_shape[2] / num_heads)
                        if isinstance(query_shape[2], int)
                        else f"{query_shape[2]}/{num_heads}"
                    )
                else:
                    head_size = query_shape[4]

                past_shape = self._try_get_shape(node, 6)

                if past_shape is not None:
                    if isinstance(past_shape[2], int) and isinstance(total_sequence_length, int):
                        total_sequence_length = past_shape[2] + total_sequence_length
                    else:
                        total_sequence_length = f"{past_shape[2]}+{total_sequence_length}"

                present_shape = [
                    batch_size,
                    num_heads,
                    total_sequence_length,
                    head_size,
                ]

                assert output_dtype is not None
                if len(node.output) > 2 and node.output[1] and node.output[2]:
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))
                    vi = self.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))

    def _infer_DecoderMaskedMultiHeadAttention(self, node):
        """Infers the output shape of the DecoderMaskedMultiHeadAttention node based on input shapes and attributes in
        the computational graph.
        """
        # Q, K and V without packing:
        #   Input 0 (query) has shape (batch_size, 1, hidden_size)
        #   Input 5 (past_key) if exists has shape (batch_size, num_heads, max_sequence_length, head_size)

        query_shape = self._get_shape(node, 0)
        if query_shape is not None:
            output_shape = query_shape
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            assert output_dtype is not None
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            if len(node.output) > 2 and node.output[1] and node.output[2]:
                past_shape = self._try_get_shape(node, 5)
                if past_shape is not None:
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
                    vi = self.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))

    def _infer_FastGelu(self, node):
        """Infers the output shapes and types for the FastGelu node using shape propagation."""
        self._propagate_shape_and_type(node)

    def _infer_Gelu(self, node):
        """Infers the output shapes and types for the Gelu node using shape propagation."""
        self._propagate_shape_and_type(node)

    def _infer_QuickGelu(self, node):
        """Infers the output shapes and types for the QuickGelu node using shape propagation."""
        self._propagate_shape_and_type(node)

    def _infer_GemmFastGelu(self, node):
        """Infers the output shapes and types for the GemmFastGelu node using matrix multiplication shape
        computation.
        """
        self._compute_matmul_shape(node)

    def _infer_GemmFloat8(self, node):
        """Infers the output shapes and types for the GemmFloat8 node using matrix multiplication shape computation."""
        self._compute_matmul_shape(node)

    def _infer_LayerNormalization(self, node):
        """Infers the output shapes and types for the LayerNormalization node, including handling mean and variance
        outputs.
        """
        self._propagate_shape_and_type(node)
        if len(node.output) > 1:
            axis = get_attribute(node, "axis")
            if axis is None:
                axis = -1
            x_shape = self._get_shape(node, 0)
            if x_shape is not None:
                rank = len(x_shape)
                axis = handle_negative_axis(axis, rank)
                mean_shape = x_shape[:axis] + [1 for _ in range(rank - axis)]
                mean_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                if mean_dtype in {
                    onnx.TensorProto.FLOAT16,
                    onnx.TensorProto.BFLOAT16,
                }:
                    mean_dtype = onnx.TensorProto.FLOAT
                vi = self.known_vi_[node.output[1]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[1], mean_dtype, mean_shape))
                if len(node.output) > 2:
                    vi = self.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(node.output[2], mean_dtype, mean_shape))

    def _infer_LongformerAttention(self, node):
        """Infer and propagate shape and type information for a LongformerAttention node."""
        self._propagate_shape_and_type(node)

    def _infer_EmbedLayerNormalization(self, node):
        """Infer and propagate shape and type information for an EmbedLayerNormalization node."""
        input_ids_shape = self._get_shape(node, 0)
        word_embedding_shape = self._get_shape(node, 2)
        assert len(input_ids_shape) == 2 and len(word_embedding_shape) == 2
        output_shape = [*input_ids_shape, word_embedding_shape[1]]

        word_embedding_dtype = self.known_vi_[node.input[2]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], word_embedding_dtype, output_shape))

        if len(node.output) > 1 and node.output[1]:
            mask_index_shape = [input_ids_shape[0]]
            vi = self.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT32, mask_index_shape))

        if len(node.output) > 2:
            # Optional output of add before layer normalization is done
            # shape is same as the output
            vi = self.known_vi_[node.output[2]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[2], word_embedding_dtype, output_shape))

    def _infer_SkipLayerNormalization(self, node):
        """Infer the output shape and type for a node with SkipLayerNormalization in an ONNX model."""
        self._propagate_shape_and_type(node)

        # If the SkipLayerNormalization node contains the optional
        # output for inference, infer the shape and type for it too
        if len(node.output) > 3:
            self._propagate_shape_and_type(node, 0, 3)

    def _infer_GroupNorm(self, node):
        """Infer the shape and type for Group Normalization in an ONNX model."""
        self._propagate_shape_and_type(node)

    def _infer_SkipGroupNorm(self, node):
        """Infer the shape and type for Skip Group Normalization in an ONNX model."""
        self._propagate_shape_and_type(node, 0, 0)
        if len(node.output) > 1:
            self._propagate_shape_and_type(node, 0, 1)

    def _infer_BiasSplitGelu(self, node):
        """Infer the shape and type for Bias Split Gelu in an ONNX model."""
        input_shape = self._get_shape(node, 0)
        bias_shape = self._get_shape(node, 1)
        if input_shape and bias_shape and isinstance(bias_shape[0], int):
            output_shape = input_shape
            output_shape[2] = int(bias_shape[0] / 2)
            vi = self.known_vi_[node.output[0]]
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, output_shape))

    def _infer_BiasAdd(self, node):
        """Infer the output shape and type for a BiasAdd node by propagating input shape and type information."""
        self._propagate_shape_and_type(node)

    def _infer_RotaryEmbedding(self, node):
        """Infer the output shape and type for a RotaryEmbedding node by appropriately propagating input shape and type
        information.
        """
        if len(node.output) == 1:
            self._propagate_shape_and_type(node)
        elif len(node.output) == 2:
            # Extraneous constant nodes outputted by RotaryEmbedding function made with `export_modules_as_functions`
            self._propagate_shape_and_type(node, input_index=1, output_index=0)
            self._propagate_shape_and_type(node, input_index=0, output_index=1)  # true output
        elif len(node.output) == 3:
            # Extraneous constant nodes outputted by RotaryEmbedding function made with `export_modules_as_functions`
            self._propagate_shape_and_type(node, input_index=1, output_index=0)
            self._propagate_shape_and_type(node, input_index=1, output_index=1)
            self._propagate_shape_and_type(node, input_index=0, output_index=2)  # true output

    def _infer_PythonOp(self, node):
        """Infer and propagate the shape and type information for a PythonOp node in the computation graph."""
        output_tensor_types = get_attribute(node, "output_tensor_types")
        assert output_tensor_types, f"PythonOp '{node.name}' has no output_tensor_types attribute."
        output_tensor_ranks = get_attribute(node, "output_tensor_ranks")
        assert output_tensor_ranks, f"PythonOp '{node.name}' has no output_tensor_ranks attribute."

        from onnxruntime.capi._pybind_state import get_shape_inference_function

        func_name = get_attribute(node, "func_name").decode()
        shape_inferer = get_shape_inference_function(func_name)

        # Set the context output separately.
        # The first output is torch.autograd.Function''s context.
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, []))

        if shape_inferer is not None:
            input_shapes = []
            input_dtypes = []
            for input_index in range(len(node.input)):
                shape = self._get_shape(node, input_index)
                input_shapes.append(shape)
                input_dtype = self.known_vi_[node.input[input_index]].type.tensor_type.elem_type
                input_dtypes.append(input_dtype)
            output_shapes, output_dtypes = shape_inferer(node, input_shapes, input_dtypes)
            assert len(output_shapes) == len(output_dtypes) == (len(node.output) - 1), (
                f"PythonOp '{func_name}' returned {len(output_shapes)} shapes and {len(output_dtypes)} dtypes, "
                f"but expected {len(node.output) - 1} outputs."
            )
            for i in range(len(node.output) - 1):
                output_index = i + 1
                vi = self.known_vi_[node.output[output_index]]
                vi.CopyFrom(
                    helper.make_tensor_value_info(node.output[output_index], output_dtypes[i], output_shapes[i])
                )
        else:
            # General shape inference for PythonOp.
            # Outputs after torch.autograd.Function's context are tensors.
            # We assume their ranks are fixed for different model inputs.
            for i in range(len(node.output) - 1):
                # Process the i-th tensor outputs.
                vi = self.known_vi_[node.output[i + 1]]
                sympy_shape = self._new_symbolic_shape(output_tensor_ranks[i], node)
                shape = get_shape_from_sympy_shape(sympy_shape)
                value_info = helper.make_tensor_value_info(node.output[i + 1], output_tensor_types[i], shape)
                vi.CopyFrom(value_info)

    def _propagate_shape_and_type(self, node, input_index=0, output_index=0):
        """Propagates the shape and type information from input to output tensors in a given node."""
        shape = self._get_shape(node, input_index)
        output_dtype = self.known_vi_[node.input[input_index]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[output_index]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[output_index], output_dtype, shape))

    def _is_none_dim(self, dim_value):
        """Check if dimension value is a string representing an unknown dimension that is not in symbolic_dims_."""
        if type(dim_value) != str:  # noqa: E721
            return False
        return dim_value not in self.symbolic_dims_ if "unk__" in dim_value else False

    def _is_shape_contains_none_dim(self, out_shape):
        """Check if any dimension in the given shape contains the 'None' dimension and return it if found."""
        for out in out_shape:
            if self._is_none_dim(out):
                return out
        return None

    def _infer_impl(self, start_sympy_data=None):
        """Infer implementation details and update symbolic data and input symbols."""
        self.sympy_data_ = start_sympy_data or {}
        self._apply_suggested_merge(graph_input_only=True)
        self.input_symbols_ = set()
        for i in self.out_mp_.graph.input:
            input_shape = get_shape_from_value_info(i)
            if input_shape is None:
                continue

            if is_sequence(i.type):
                input_dims = i.type.sequence_type.elem_type.tensor_type.shape.dim
            else:
                input_dims = i.type.tensor_type.shape.dim

            for i_dim, dim in enumerate(input_shape):
                if dim is None:
                    # some models use None for symbolic dim in input, replace it with a string
                    input_dims[i_dim].dim_param = str(self._new_symbolic_dim(i.name, i_dim))

            self.input_symbols_.update([d for d in input_shape if type(d) == str])  # noqa: E721

        for s in self.input_symbols_:
            if s in self.suggested_merge_:
                s_merge = self.suggested_merge_[s]
                assert s_merge in self.symbolic_dims_
                self.symbolic_dims_[s] = self.symbolic_dims_[s_merge]
            else:
                # Since inputs are not produced by other ops, we can assume positivity
                self.symbolic_dims_[s] = sympy.Symbol(s, integer=True, positive=True)
        # compute prerequisite for node for topological sort
        # node with subgraphs may have dependency on implicit inputs, which will affect topological sort
        prereq_for_node = {}  # map from node to all its inputs, including implicit ones in subgraph

        def get_prereq(node):
            """Compute and return the prerequisite inputs for a given node, including implicit inputs from subgraphs."""
            names = {i for i in node.input if i}
            subgraphs = []
            if node.op_type == "If":
                subgraphs = [
                    get_attribute(node, "then_branch"),
                    get_attribute(node, "else_branch"),
                ]
            elif node.op_type in {"Loop", "Scan"}:
                subgraphs = [get_attribute(node, "body")]
            for g in subgraphs:
                g_outputs_and_initializers = {i.name for i in g.initializer}
                g_prereq = set()
                for n in g.node:
                    g_outputs_and_initializers.update(n.output)
                for n in g.node:
                    g_prereq.update([i for i in get_prereq(n) if i not in g_outputs_and_initializers])
                names.update(g_prereq)
                # remove subgraph inputs from g_prereq since those are local-only
                for i in g.input:
                    if i.name in names:
                        names.remove(i.name)
            return names

        for n in self.out_mp_.graph.node:
            prereq_for_node[n.output[0]] = get_prereq(n)

        # topological sort nodes, note there might be dead nodes so we check if all graph outputs are reached to terminate
        sorted_nodes = []
        sorted_known_vi = {i.name for i in list(self.out_mp_.graph.input) + list(self.out_mp_.graph.initializer)}
        if any(o.name in sorted_known_vi for o in self.out_mp_.graph.output):
            # Loop/Scan will have some graph output in graph inputs, so don't do topological sort
            sorted_nodes = self.out_mp_.graph.node
        else:
            while any(o.name not in sorted_known_vi for o in self.out_mp_.graph.output):
                old_sorted_nodes_len = len(sorted_nodes)
                for node in self.out_mp_.graph.node:
                    if node.output[0] not in sorted_known_vi and all(
                        i in sorted_known_vi for i in prereq_for_node[node.output[0]] if i
                    ):
                        sorted_known_vi.update(node.output)
                        sorted_nodes.append(node)
                if old_sorted_nodes_len == len(sorted_nodes) and not all(
                    o.name in sorted_known_vi for o in self.out_mp_.graph.output
                ):
                    raise Exception("Invalid model with cyclic graph")

        for node in sorted_nodes:
            assert all([i in self.known_vi_ for i in node.input if i])
            self._onnx_infer_single_node(node)
            known_aten_op = False
            if node.op_type in self.dispatcher_:
                self.dispatcher_[node.op_type](node)
            elif node.op_type == "ConvTranspose":
                # onnx shape inference ops like ConvTranspose may have empty shape for symbolic input
                # before adding symbolic compute for them
                # mark the output type as UNDEFINED to allow guessing of rank
                vi = self.known_vi_[node.output[0]]
                if len(vi.type.tensor_type.shape.dim) == 0:
                    vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
            elif node.op_type == "ATen" and node.domain == "org.pytorch.aten":
                for attr in node.attribute:
                    # TODO: Is overload_name needed?
                    if attr.name == "operator":
                        aten_op_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                        if aten_op_name in self.aten_op_dispatcher_:
                            known_aten_op = True
                            self.aten_op_dispatcher_[aten_op_name](node)
                        break

            if self.verbose_ > 2:
                logger.debug(node.op_type + ": " + node.name)
                for i, name in enumerate(node.input):
                    logger.debug(
                        "  Input {}: {} {}".format(i, name, "initializer" if name in self.initializers_ else "")
                    )

            # onnx automatically merge dims with value, i.e. Mul(['aaa', 'bbb'], [1000, 1]) -> [1000, 'bbb']
            # symbolic shape inference needs to apply merge of 'aaa' -> 1000 in this case
            if node.op_type in {
                "Add",
                "Sub",
                "Mul",
                "Div",
                "MatMul",
                "MatMulInteger",
                "MatMulInteger16",
                "Where",
                "Sum",
            }:
                vi = self.known_vi_[node.output[0]]
                out_rank = len(get_shape_from_type_proto(vi.type))
                in_shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                for d in range(out_rank - (2 if node.op_type in {"MatMul", "MatMulInteger", "MatMulInteger16"} else 0)):
                    in_dims = [s[len(s) - out_rank + d] for s in in_shapes if len(s) + d >= out_rank]
                    if len(in_dims) > 1:
                        self._check_merged_dims(in_dims, allow_broadcast=True)

            for i_o in range(len(node.output)):
                # Special cases:
                # 1) We do not care about the training related outputs of SkipLayerNormalization
                # 2) We do not care about the extraneous constant outputs in RotaryEmbedding because
                # the RotaryEmbedding op created during export can be replaced by the RotaryEmbedding
                # contrib op
                if node.op_type in {
                    "SkipLayerNormalization",
                    "SkipSimplifiedLayerNormalization",
                } and i_o in {1, 2}:
                    continue
                if node.op_type == "RotaryEmbedding" and len(node.output) > 1:
                    # Skip symbolic shape inference for RotaryEmbedding functions that have extraneous outputs
                    # generated by `export_modules_as_functions`
                    continue

                vi = self.known_vi_[node.output[i_o]]
                out_type = vi.type
                out_type_kind = out_type.WhichOneof("value")

                # do not process shape for non-tensors
                if out_type_kind not in {"tensor_type", "sparse_tensor_type", None}:
                    if self.verbose_ > 2:
                        if out_type_kind == "sequence_type":
                            seq_cls_type = out_type.sequence_type.elem_type.WhichOneof("value")
                            if seq_cls_type == "tensor_type":
                                logger.debug(
                                    "  {}: sequence of {} {}".format(
                                        node.output[i_o],
                                        str(get_shape_from_value_info(vi)),
                                        onnx.TensorProto.DataType.Name(
                                            vi.type.sequence_type.elem_type.tensor_type.elem_type
                                        ),
                                    )
                                )
                            else:
                                logger.debug(f"  {node.output[i_o]}: sequence of {seq_cls_type}")
                        else:
                            logger.debug(f"  {node.output[i_o]}: {out_type_kind}")
                    continue

                out_shape = get_shape_from_value_info(vi)
                out_type_undefined = out_type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED
                if self.verbose_ > 2:
                    logger.debug(
                        f"  {node.output[i_o]}: {out_shape!s} {onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)}"
                    )
                    if node.output[i_o] in self.sympy_data_:
                        logger.debug("  Sympy Data: " + str(self.sympy_data_[node.output[i_o]]))

                # onnx >= 1.11.0, use unk__#index instead of None when the shape dim is uncertain
                if (
                    out_shape is not None and (None in out_shape or self._is_shape_contains_none_dim(out_shape))
                ) or out_type_undefined:
                    if self.auto_merge_:
                        if node.op_type in {
                            "Add",
                            "Sub",
                            "Mul",
                            "Div",
                            "MatMul",
                            "MatMulInteger",
                            "MatMulInteger16",
                            "Concat",
                            "Where",
                            "Sum",
                            "Equal",
                            "Less",
                            "Greater",
                            "LessOrEqual",
                            "GreaterOrEqual",
                            "Min",
                            "Max",
                        }:
                            shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                            if node.op_type in {
                                "MatMul",
                                "MatMulInteger",
                                "MatMulInteger16",
                            } and (None in out_shape or self._is_shape_contains_none_dim(out_shape)):
                                if None in out_shape:
                                    idx = out_shape.index(None)
                                else:
                                    idx = out_shape.index(self._is_shape_contains_none_dim(out_shape))
                                dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                # only support auto merge for MatMul for dim < rank-2 when rank > 2
                                assert len(shapes[0]) > 2 and dim_idx[0] < len(shapes[0]) - 2
                                assert len(shapes[1]) > 2 and dim_idx[1] < len(shapes[1]) - 2
                        elif node.op_type == "Expand":
                            # auto merge for cases like Expand([min(batch, 1), min(seq, 512)], [batch, seq])
                            shapes = [
                                self._get_shape(node, 0),
                                self._get_value(node, 1),
                            ]
                        else:
                            shapes = []

                        if shapes:
                            for idx in range(len(out_shape)):
                                if out_shape[idx] is not None and not self._is_none_dim(out_shape[idx]):
                                    continue
                                # note that the broadcasting rule aligns from right to left
                                # if a tensor has a lower rank (dim_idx[idx] < 0), it would automatically broadcast and need no merge
                                dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                if dim_idx:
                                    self._add_suggested_merge(
                                        [
                                            s[i] if is_literal(s[i]) else str(s[i])
                                            for s, i in zip(shapes, dim_idx)
                                            if i >= 0
                                        ]
                                    )
                            self.run_ = True
                        else:
                            self.run_ = False
                    else:
                        self.run_ = False

                    # create new dynamic dims for ops not handled by symbolic shape inference
                    if not self.run_ and node.op_type not in self.dispatcher_ and not known_aten_op:
                        is_unknown_op = out_type_undefined and (out_shape is None or len(out_shape) == 0)
                        if is_unknown_op:
                            # unknown op to ONNX, maybe from higher opset or other domain
                            # only guess the output rank from input 0 when using guess_output_rank option
                            out_rank = self._get_shape_rank(node, 0) if self.guess_output_rank_ else -1
                        else:
                            # valid ONNX op, but not handled by symbolic shape inference, just assign dynamic shape
                            out_rank = len(out_shape)

                        if out_rank >= 0:
                            new_shape = self._new_symbolic_shape(out_rank, node, i_o)
                            if out_type_undefined:
                                # guess output data type from input vi if not defined
                                out_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                            else:
                                # otherwise, use original data type
                                out_dtype = vi.type.tensor_type.elem_type
                            vi.CopyFrom(
                                helper.make_tensor_value_info(
                                    vi.name,
                                    out_dtype,
                                    get_shape_from_sympy_shape(new_shape),
                                )
                            )

                            if self.verbose_ > 0:
                                if is_unknown_op:
                                    logger.debug(
                                        f"Possible unknown op: {node.op_type} node: {node.name}, guessing {vi.name} shape"
                                    )
                                if self.verbose_ > 2:
                                    logger.debug(f"  {node.output[i_o]}: {new_shape!s} {vi.type.tensor_type.elem_type}")
                            self.run_ = True
                            continue  # continue the inference after guess, no need to stop as no merge is needed

                    if self.verbose_ > 0 or not self.auto_merge_ or out_type_undefined:
                        logger.debug("Stopping at incomplete shape inference at " + node.op_type + ": " + node.name)
                        logger.debug("node inputs:")
                        for i in node.input:
                            if i in self.known_vi_:
                                logger.debug(self.known_vi_[i])
                            else:
                                logger.debug(f"not in known_vi_ for {i}")
                        logger.debug("node outputs:")
                        for o in node.output:
                            if o in self.known_vi_:
                                logger.debug(self.known_vi_[o])
                            else:
                                logger.debug(f"not in known_vi_ for {o}")
                        if self.auto_merge_ and not out_type_undefined:
                            logger.debug("Merging: " + str(self.suggested_merge_))
                    return False

        self.run_ = False
        return True

    def _update_output_from_vi(self):
        """Update output attributes using known value information dictionary."""
        for output in self.out_mp_.graph.output:
            if output.name in self.known_vi_:
                output.CopyFrom(self.known_vi_[output.name])

    @staticmethod
    def infer_shapes(in_mp, int_max=2**31 - 1, auto_merge=False, guess_output_rank=False, verbose=0):
        """Perform symbolic shape inference on an ONNX model using the specified options to handle model shapes
        efficiently.
        """
        onnx_opset = get_opset(in_mp)
        if (not onnx_opset) or onnx_opset < 7:
            logger.warning("Only support models of onnx opset 7 and above.")
            return None
        symbolic_shape_inference = SymbolicShapeInference(int_max, auto_merge, guess_output_rank, verbose)
        all_shapes_inferred = False
        symbolic_shape_inference._preprocess(in_mp)
        while symbolic_shape_inference.run_:
            all_shapes_inferred = symbolic_shape_inference._infer_impl()
        symbolic_shape_inference._update_output_from_vi()
        if not all_shapes_inferred:
            raise Exception("Incomplete symbolic shape inference")
        return symbolic_shape_inference.out_mp_


def parse_arguments():
    """Parses command-line arguments for ONNX model transformation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The input model file")
    parser.add_argument("--output", help="The output model file")
    parser.add_argument(
        "--auto_merge",
        help="Automatically merge symbolic dims when confliction happens",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--int_max",
        help="maximum value for integer to be treated as boundless for ops like slice",
        type=int,
        default=2**31 - 1,
    )
    parser.add_argument(
        "--guess_output_rank",
        help="guess output rank to be the same as input 0 for unknown ops",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_as_external_data",
        help="Saving an ONNX model to external data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--all_tensors_to_one_file",
        help="Saving all the external data to one file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--external_data_location",
        help="The file location to save the external file",
        default="./",
    )
    parser.add_argument(
        "--external_data_size_threshold",
        help="The size threshold for external data",
        type=int,
        default=1024,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"input model: {args.input}")
    if args.output:
        logger.info(f"output model {args.output}")
    logger.info("Doing symbolic shape inference...")
    out_mp = SymbolicShapeInference.infer_shapes(
        onnx.load(args.input),
        args.int_max,
        args.auto_merge,
        args.guess_output_rank,
        args.verbose,
    )
    if args.output and out_mp:
        if args.save_as_external_data:
            onnx.save_model(
                out_mp,
                args.output,
                save_as_external_data=True,
                all_tensors_to_one_file=args.all_tensors_to_one_file,
                location=args.external_data_location,
                size_threshold=args.external_data_size_threshold,
                convert_attribute=False,
            )
        else:
            onnx.save(out_mp, args.output)
        logger.info("Done!")
