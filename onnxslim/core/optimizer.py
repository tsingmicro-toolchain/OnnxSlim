import contextlib

from collections import Counter, OrderedDict

import numpy as np
from loguru import logger

import onnxslim.onnx_graphsurgeon as gs
from onnxslim.onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx
from onnxslim.onnx_graphsurgeon.ir.tensor import Constant


DEFAULT_FUSION_PATTERNS = OrderedDict()


def register_fusion_pattern(layer_type):
    def insert(fn):
        if layer_type in DEFAULT_FUSION_PATTERNS.keys():
            raise
        DEFAULT_FUSION_PATTERNS[layer_type] = fn
        return fn

    return insert


def get_default_fusion_patterns():
    return DEFAULT_FUSION_PATTERNS


def get_node_users(node):
    users = []
    for output in node.outputs:  # output is a Variable
        for user in output.outputs:  # user is a Node
            users.append(user)
    return users


def get_constant_variable(node):
    for input in list(node.inputs):
        if isinstance(input, Constant):
            return input


def delete_node(node):
    input_variable = node.inputs[0]
    node_variable = node.outputs[0]
    next_nodes = get_node_users(node)
    if next_nodes:
        for next_node in next_nodes:
            index = next_node.inputs.index(node_variable)
            next_node.inputs.pop(index)
            next_node.inputs.insert(index, input_variable)
    else:
        input_node = node.i()
        input_node.outputs.remove(node.inputs[0])
        input_node.outputs.append(node.outputs[0])
        node.outputs.clear()


def graph_constant_fold_inplace(graph):
    for node in graph.nodes:
        if node.op == "Identity" or node.op == "Dropout":
            delete_node(node)

        elif node.op == "Pad":
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant):
                pad_value = node.inputs[1].values.tolist()
                pad_value = (
                    [pad_value] if not isinstance(pad_value, list) else pad_value
                )
                if all([value == 0 for value in pad_value]):
                    delete_node(node)
        elif node.op == "Cast":
            inp_dtype = [dtype_to_onnx(input.dtype) for input in node.inputs][0]
            if inp_dtype == node.attrs["to"]:
                delete_node(node)


@register_fusion_pattern("Conv")
def find_conv_nodes(node):
    # fmt: off
    '''
             x
             |
            Pad
             |
            Conv
    '''
    # fmt: on
    match = {}
    if node.op == "Conv":
        if node.i(0).op == "Pad":
            pad_node = node.i(0)
            pad_value = pad_node.inputs[1].values.tolist()
            input_variable = node.i(0).inputs[0]
            input_variable.outputs.remove(pad_node)

            pad_variable = node.i(0).outputs[0]  # pad output variable
            index = node.inputs.index(pad_variable)
            node.inputs.pop(index)
            node.inputs.insert(index, input_variable)

            inputs = list(node.inputs)
            outputs = list(node.outputs)
            attrs = node.attrs

            node.inputs.clear()
            node.outputs.clear()
            pad_node.inputs.clear()
            pad_node.outputs.clear()

            conv_pads = attrs["pads"]
            len_conv_pads = int(len(conv_pads) / 2)

            len_pads = int(len(pad_value) / 2)
            pads = (
                pad_value[len_pads - len_conv_pads : len_pads]
                + pad_value[len_pads + len_conv_pads :]
            )
            attrs["pads"] = pads

            match.update(
                {
                    node.name: {
                        "inputs": inputs,
                        "outputs": outputs,
                        "name": node.name,
                        "attrs": node.attrs,
                        "domain": None,
                    }
                }
            )

    return match


@register_fusion_pattern("ConvTranspose")
def find_conv_transpose_nodes(node):
    # fmt: off
    '''
             x
             |
        ConvTranspose
             |
      BatchNormalization
    '''
    # fmt: on
    match = {}
    if node.op == "BatchNormalization":
        if node.i(0).op == "ConvTranspose":
            conv_transpose_node = node.i(0)
            conv_transpose_weight = conv_transpose_node.inputs[1].values
            bn_node = node
            bn_scale = bn_node.inputs[1].values
            bn_bias = bn_node.inputs[2].values
            bn_running_mean = bn_node.inputs[3].values
            bn_running_var = bn_node.inputs[4].values
            bn_eps = bn_node.attrs["epsilon"]

            if len(conv_transpose_node.inputs) == 2:
                conv_transpose_bias = np.zeros_like(bn_running_mean)
            else:
                conv_transpose_bias = conv_transpose_node.inputs[2].values

            bn_var_rsqrt = 1.0 / np.sqrt(bn_running_var + bn_eps)
            shape = [1] * len(conv_transpose_weight.shape)
            shape[1] = -1
            conv_w = conv_transpose_weight * (bn_scale * bn_var_rsqrt).reshape(shape)
            conv_b = (
                conv_transpose_bias - bn_running_mean
            ) * bn_var_rsqrt * bn_scale + bn_bias

            input_variable = bn_node.inputs[0]
            input_variable.outputs.remove(bn_node)

            inputs = []
            inputs.append(list(conv_transpose_node.inputs)[0])
            weight_name = list(conv_transpose_node.inputs)[1].name
            bias_name = ".".join(weight_name.split(".")[:-1] + ["bias"])
            inputs.append(gs.Constant(weight_name, values=conv_w))
            inputs.append(gs.Constant(bias_name, values=conv_b))
            outputs = list(bn_node.outputs)

            bn_node.inputs.clear()
            bn_node.outputs.clear()

            match.update(
                {
                    conv_transpose_node.name: {
                        "inputs": inputs,
                        "outputs": outputs,
                        "name": conv_transpose_node.name,
                        "attrs": conv_transpose_node.attrs,
                        "domain": None,
                    }
                }
            )

    return match


@register_fusion_pattern("Slice")
def find_slice_nodes(node):
    # fmt: off
    '''
             x
             |
           Slice
             |
           Slice
    '''
    # fmt: on
    match = {}
    if node.op == "Slice":
        if node.i(0).op == "Slice":
            first_slice_node = node.i(0)
            first_slice_node_inputs = list(first_slice_node.inputs)
            if all(
                [isinstance(input, Constant) for input in first_slice_node_inputs[1:]]
            ):
                first_slice_node_users = get_node_users(first_slice_node)
                if all(
                    [
                        user.op == "Slice"
                        and all(
                            [
                                isinstance(input, Constant)
                                for input in list(user.inputs)[1:]
                            ]
                        )
                        for user in first_slice_node_users
                    ]
                ):
                    first_slice_node_starts = first_slice_node_inputs[1].values.tolist()
                    first_slice_node_ends = first_slice_node_inputs[2].values.tolist()
                    first_slice_node_axes = first_slice_node_inputs[3].values.tolist()
                    first_slice_node_steps = first_slice_node_inputs[4].values.tolist()

                    for user_node in first_slice_node_users:
                        second_slice_node = user_node
                        second_slice_node_inputs = list(second_slice_node.inputs)
                        second_slice_node_starts = second_slice_node_inputs[
                            1
                        ].values.tolist()
                        second_slice_node_ends = second_slice_node_inputs[
                            2
                        ].values.tolist()
                        second_slice_node_axes = second_slice_node_inputs[
                            3
                        ].values.tolist()
                        second_slice_node_steps = second_slice_node_inputs[
                            4
                        ].values.tolist()

                        new_starts = first_slice_node_starts + second_slice_node_starts
                        new_ends = first_slice_node_ends + second_slice_node_ends
                        new_axes = first_slice_node_axes + second_slice_node_axes
                        new_steps = first_slice_node_steps + second_slice_node_steps

                        inputs = []
                        inputs.append(list(first_slice_node.inputs)[0])
                        inputs.append(
                            gs.Constant(
                                second_slice_node_inputs[1].name,
                                values=np.array(new_starts, dtype=np.int64),
                            )
                        )
                        inputs.append(
                            gs.Constant(
                                second_slice_node_inputs[2].name,
                                values=np.array(new_ends, dtype=np.int64),
                            )
                        )
                        inputs.append(
                            gs.Constant(
                                second_slice_node_inputs[3].name,
                                values=np.array(new_axes, dtype=np.int64),
                            )
                        )
                        inputs.append(
                            gs.Constant(
                                second_slice_node_inputs[4].name,
                                values=np.array(new_steps, dtype=np.int64),
                            )
                        )
                        outputs = list(second_slice_node.outputs)

                        second_slice_node.inputs.clear()
                        second_slice_node.outputs.clear()

                        match.update(
                            {
                                second_slice_node.name: {
                                    "inputs": inputs,
                                    "outputs": outputs,
                                    "name": second_slice_node.name,
                                    "attrs": second_slice_node.attrs,
                                    "domain": None,
                                }
                            }
                        )

    return match


@register_fusion_pattern("Reshape")
def find_slice_nodes(node):
    # fmt: off
    '''
             x
             |
           Reshape
             |
           Reshape
    '''
    # fmt: on
    match = {}
    if node.op == "Reshape":
        if node.i(0).op == "Reshape":
            first_reshape_node = node.i(0)
            first_reshape_node_inputs = list(first_reshape_node.inputs)
            first_reshape_node_users = get_node_users(first_reshape_node)
            if len(first_reshape_node_users) == 1:
                second_reshape_node = node
                inputs = []
                inputs.append(first_reshape_node_inputs[0])
                inputs.append(second_reshape_node.inputs[1])

                outputs = list(second_reshape_node.outputs)

                first_reshape_node.outputs.clear()
                second_reshape_node.inputs.clear()
                second_reshape_node.outputs.clear()

                match.update(
                    {
                        second_reshape_node.name: {
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": second_reshape_node.name,
                            "attrs": second_reshape_node.attrs,
                            "domain": None,
                        }
                    }
                )

    return match


@register_fusion_pattern("Gemm")
def find_matmul_add_nodes(node):
    # fmt: off
    '''
             x
             |
           MatMul
             |
            Add
    '''
    # fmt: on
    match = {}
    if node.op == "Add":
        if (isinstance(node.inputs[1], Constant) and node.i(0).op == "MatMul") or (
            isinstance(node.inputs[0], Constant) and node.i(1).op == "MatMul"
        ):
            matmul_node = (
                node.i(0) if isinstance(node.inputs[1], Constant) else node.i(1)
            )
            matmul_bias_variable = get_constant_variable(matmul_node)
            input_variable = (
                matmul_node.inputs[0]
                if isinstance(matmul_node.inputs[1], Constant)
                else matmul_node.inputs[1]
            )
            users = get_node_users(matmul_node)
            if len(users) == 1 and matmul_bias_variable:
                if (
                    input_variable.shape
                    and len(input_variable.shape) > 2
                    and all([isinstance(value, int) for value in input_variable.shape])
                ):
                    pre_reshape_const = gs.Constant(
                        matmul_node.name + "_pre_reshape_in",
                        values=np.array(
                            [-1, matmul_bias_variable.values.shape[0]], dtype=np.int64
                        ),
                    )
                    inputs = []
                    inputs.append(input_variable)
                    inputs.append(pre_reshape_const)

                    reshape_out_variable = gs.Variable(
                        matmul_node.name + "_pre__reshape_out",
                        dtype=input_variable.dtype,
                    )
                    outputs = [reshape_out_variable]

                    match.update(
                        {
                            matmul_node.name
                            + "_pre__reshape": {
                                "op": "Reshape",
                                "inputs": inputs,
                                "outputs": outputs,
                                "name": matmul_node.name + "_pre__reshape",
                                "domain": None,
                            }
                        }
                    )

                    add_node = node
                    add_bias_variable = get_constant_variable(add_node)

                    output_variable = add_node.inputs[0]
                    output_variable.outputs.remove(add_node)

                    matmul_bias_transpose_constant = gs.Constant(
                        matmul_bias_variable.name, values=matmul_bias_variable.values.T
                    )

                    inputs = []
                    inputs.append(reshape_out_variable)
                    inputs.append(matmul_bias_transpose_constant)
                    inputs.append(add_bias_variable)

                    gemm_out_variable = gs.Variable(
                        matmul_node.name + "_gemm_out", dtype=output_variable.dtype
                    )
                    outputs = [gemm_out_variable]

                    match.update(
                        {
                            matmul_node.name: {
                                "inputs": inputs,
                                "outputs": outputs,
                                "name": matmul_node.name,
                                "attrs": {
                                    "alpha": 1.0,
                                    "beta": 1.0,
                                    "transA": 0,
                                    "transB": 1,
                                },
                                "domain": None,
                            }
                        }
                    )

                    values = input_variable.shape[:-1] + [
                        matmul_bias_variable.values.shape[-1]
                    ]
                    post_reshape_const = gs.Constant(
                        matmul_node.name + "_post_reshape_in",
                        values=np.array(values, dtype=np.int64),
                    )

                    inputs = []
                    inputs.append(gemm_out_variable)
                    inputs.append(post_reshape_const)
                    outputs = list(add_node.outputs)

                    matmul_node.outputs.clear()
                    add_node.inputs.clear()
                    add_node.outputs.clear()

                    match.update(
                        {
                            matmul_node.name
                            + "_post_reshape": {
                                "op": "Reshape",
                                "inputs": inputs,
                                "outputs": outputs,
                                "name": matmul_node.name + "_post_reshape",
                                "domain": None,
                            }
                        }
                    )

    return match


@register_fusion_pattern("Gelu")
def find_gelu_nodes(node):
    # fmt: off
    '''
             x
         /      \
         |     Div
         |      |
         |     Erf
         |      |
         |     Add
         \      /
            Mul
             |
            Mul
    '''
    # fmt: on
    match = {}
    if node.op == "Mul":
        if (
            node.i(0).op == "Mul"
            and node.i(0).i(1).op == "Add"
            and node.i(0).i(1).i(0).op == "Erf"
            and node.i(0).i(1).i(0).i(0).op == "Div"
        ):
            input_variable = node.i(0).i(1).i(0).i(0).inputs[0]
            mul_node = node.i(0)
            div_node = node.i(0).i(1).i(0).i(0)

            input_variable.outputs.remove(mul_node)
            input_variable.outputs.remove(div_node)

            output_variable = node.outputs[0]
            output_variable.inputs.clear()
            match.update(
                {
                    node.name: {
                        "inputs": [input_variable],
                        "outputs": [output_variable],
                        "domain": None,
                    }
                }
            )

    return match


@gs.Graph.register()
def replace_custom_layer(
    self, op, inputs, outputs, name, attrs=None, domain="ai.onnx.contrib"
):
    return self.layer(
        op=op,
        inputs=inputs,
        outputs=outputs,
        name=name,
        attrs=attrs,
        domain=domain,
    )


def find_matches(graph, fusion_patterns):
    match_map = {}
    counter = Counter()
    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for layer_type, func in fusion_patterns.items():
                with contextlib.suppress(IndexError):
                    matches = func(node)
                    if matches:
                        logger.debug(f"matched pattern {layer_type}")
                        for _, match in matches.items():
                            if "op" not in match:
                                match.update({"op": layer_type})
                            if "name" not in match:
                                match.update(
                                    {
                                        "name": "{}_{}".format(
                                            layer_type.lower(), counter[layer_type]
                                        )
                                    }
                                )
                            counter.update([layer_type])
                        match_map.update(matches)

    return match_map


def optimize_model(model):
    graph = gs.import_onnx(model)
    graph.fold_constants().cleanup()
    fusion_patterns = get_default_fusion_patterns()
    fusion_pairs = find_matches(graph, fusion_patterns)
    for _, match in fusion_pairs.items():
        graph.replace_custom_layer(**match)
    graph_constant_fold_inplace(graph)
    graph.cleanup(
        remove_unused_node_outputs=True, remove_unused_graph_inputs=True
    ).toposort()
    model = gs.export_onnx(graph)

    return model
