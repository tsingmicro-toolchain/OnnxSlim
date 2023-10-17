from loguru import logger
import contextlib
import numpy as np
import onnxslim.onnx_graphsurgeon as gs
from collections import OrderedDict, Counter

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


def graph_constant_fold_inplace(graph):
    for node in graph.nodes:
        if node.op == "Identity" or node.op == "Dropout":
            input_node = node.i()
            input_node.outputs.remove(node.inputs[0])
            input_node.outputs.append(node.outputs[0])
            node.outputs.clear()


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
        if (node.i(0).op == "Pad"):
            pad_node = node.i(0)
            pad_value = pad_node.inputs[1].values.tolist()
            input_variable = node.i(0).inputs[0]
            input_variable.outputs.remove(pad_node)
            
            pad_variable= node.i(0).outputs[0] # pad output variable
            index = node.inputs.index(pad_variable)
            node.inputs.pop(index)
            node.inputs.insert(index, input_variable)

            inputs = list(node.inputs)
            outputs = list(node.outputs)
            attrs = node.attrs

            node.inputs.clear()
            node.outputs.clear()
            
            conv_pads = attrs['pads']
            len_conv_pads = int(len(conv_pads) / 2)
            
            len_pads = int(len(pad_value) / 2)
            pads = pad_value[len_pads - len_conv_pads: len_pads] + pad_value[len_pads + len_conv_pads: ]
            attrs['pads'] = pads

            match.update(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": node.name,
                    "attrs": node.attrs,
                    "domain": None
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
        if (node.i(0).op == "ConvTranspose"):
            conv_transpose_node = node.i(0)
            conv_transpose_weight = conv_transpose_node.inputs[1].values
            bn_node = node
            bn_scale = bn_node.inputs[1].values
            bn_bias = bn_node.inputs[2].values
            bn_running_mean = bn_node.inputs[3].values
            bn_running_var = bn_node.inputs[4].values
            bn_eps = bn_node.attrs['epsilon']

            if len(conv_transpose_node.inputs) == 2:
                conv_transpose_bias = np.zeros_like(bn_running_mean)
            else:
                conv_transpose_bias = conv_transpose_node.inputs[2].values

            bn_var_rsqrt = 1.0 / np.sqrt(bn_running_var + bn_eps)
            shape = [1] * len(conv_transpose_weight.shape)
            shape[1] = -1
            conv_w = conv_transpose_weight * (bn_scale * bn_var_rsqrt).reshape(shape)
            conv_b = (conv_transpose_bias - bn_running_mean) * bn_var_rsqrt * bn_scale + bn_bias

            input_variable = bn_node.inputs[0]
            input_variable.outputs.remove(bn_node)

            inputs = []
            inputs.append(list(conv_transpose_node.inputs)[0])
            weight_name = list(conv_transpose_node.inputs)[1].name
            bias_name = '.'.join(weight_name.split('.')[:-1] + ['bias'])
            inputs.append(gs.Constant(weight_name, values=conv_w))
            inputs.append(gs.Constant(bias_name, values=conv_b))
            outputs = list(bn_node.outputs)

            bn_node.inputs.clear()
            bn_node.outputs.clear()

            match.update(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": conv_transpose_node.name,
                    "attrs": conv_transpose_node.attrs,
                    "domain": None
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
                    match = func(node)
                    if match:
                        logger.debug(f"matched pattern {layer_type}")
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
                        if node.name not in match_map:
                            match_map[node.name] = match

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
