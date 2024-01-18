from typing import Dict, List, Tuple, Union

import numpy as np

import onnx

from ..utils.font import GREEN, WHITE
from ..utils.tabulate import SEPARATING_LINE, tabulate


def format_bytes(size: Union[int, Tuple[int, ...]]) -> str:
    if isinstance(size, int):
        size = (size,)

    units = ["B", "KB", "MB", "GB"]
    formatted_sizes = []

    for size_in_bytes in size:
        unit_index = 0
        while size_in_bytes >= 1024 and unit_index < len(units) - 1:
            size_in_bytes /= 1024
            unit_index += 1

        formatted_size = "{:.2f} {}".format(size_in_bytes, units[unit_index])
        formatted_sizes.append(formatted_size)

    if len(formatted_sizes) == 1:
        return formatted_sizes[0]
    else:
        return f"{formatted_sizes[0]} ({formatted_sizes[1]})"


def onnx_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    import onnx.mapping as mapping

    return np.dtype(mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype])


def gen_onnxruntime_input_data(model: onnx.ModelProto) -> Dict[str, np.array]:
    input_info = []
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(None)
        dtype = onnx_dtype_to_numpy(input_tensor.type.tensor_type.elem_type)

        input_info.append([name, shape, dtype])

    input_data_dict = {}
    for name, shapes, dtype in input_info:
        shapes = [
            shape if (shape != -1 and not isinstance(shape, str)) else 1
            for shape in shapes
        ]
        shapes = shapes if shapes else [1]
        if dtype in [np.int32, np.int64]:
            random_data = np.random.randint(10, size=shapes).astype(dtype)
        else:
            random_data = np.random.rand(*shapes).astype(dtype)
        input_data_dict[name] = random_data

    return input_data_dict


def onnxruntime_inference(
    model: onnx.ModelProto, input_data: dict
) -> Dict[str, np.array]:
    import onnxruntime as rt

    sess = rt.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    onnx_output = sess.run(None, input_data)

    output_names = [output.name for output in sess.get_outputs()]
    onnx_output = dict(zip(output_names, onnx_output))

    return onnx_output


def print_model_info_as_table(model_name, opset, model_info_list: List[Dict]):
    assert (
        len(model_info_list) > 0
    ), "model_info_list must contain more than one model info"

    final_op_info = []
    if len(model_info_list) == 1:
        final_op_info.append(["Model Name", model_name])
        final_op_info.append([SEPARATING_LINE])
        final_op_info.append(["Op Set ", opset])
    else:
        final_op_info.append(
            ["Model Name", model_name, "Op Set: " + opset]
            + [""] * (len(model_info_list) - 2)
        )
    final_op_info.append([SEPARATING_LINE])

    final_op_info.append(
        ["Model Info", "Original Model"]
        + ["Slimmed Model"] * (len(model_info_list) - 1)
    )
    final_op_info.append([SEPARATING_LINE] * (len(model_info_list) + 1))

    all_inputs = list(model_info_list[0]["op_input_info"].keys())

    for inputs in all_inputs:
        input_info_list = [
            "IN: " + inputs,
        ]
        for model_info in model_info_list:
            inputs_shape = model_info["op_input_info"].get(inputs, "")
            input_info_list.append(inputs_shape)
        final_op_info.append(input_info_list)

    all_outputs = set(
        op_type
        for model_info in model_info_list
        for op_type in model_info.get("op_output_info", {})
    )

    for outputs in all_outputs:
        output_info_list = [
            "OUT: " + outputs,
        ]
        for model_info in model_info_list:
            outputs_shape = model_info["op_output_info"].get(outputs, "")
            output_info_list.append(outputs_shape)
        final_op_info.append(output_info_list)

    final_op_info.append([SEPARATING_LINE] * (len(model_info_list) + 1))
    final_op_info.append(
        ["Op Type", "Original Model"] + ["Slimmed Model"] * (len(model_info_list) - 1)
    )
    final_op_info.append([SEPARATING_LINE] * (len(model_info_list) + 1))

    all_ops = set(
        op_type
        for model_info in model_info_list
        for op_type in model_info.get("op_type_counts", {})
    )
    sorted_ops = list(all_ops)
    sorted_ops.sort()
    for op in sorted_ops:
        op_info_list = [op]
        float_number = model_info_list[0]["op_type_counts"].get(op, 0)
        op_info_list.append(float_number)
        for model_info in model_info_list[1:]:
            slimmed_number = model_info["op_type_counts"].get(op, 0)
            if float_number > slimmed_number:
                slimmed_number = GREEN + str(slimmed_number) + WHITE
            op_info_list.append(slimmed_number)

        final_op_info.append(op_info_list)
    final_op_info.append([SEPARATING_LINE] * (len(model_info_list) + 1))
    final_op_info.append(
        ["Model Size"]
        + [format_bytes(model_info["model_size"]) for model_info in model_info_list]
    )
    lines = tabulate(
        final_op_info,
        headers=[],
        tablefmt="pretty",
        maxcolwidths=[None] + [40] * len(model_info_list),
    ).split("\n")

    output = "\n".join([line if line != "| \x01 |" else lines[0] for line in lines])

    print(output)


def get_opset(model: onnx.ModelProto) -> int:
    try:
        for importer in model.opset_import:
            if importer.domain == "" or importer.domain == "ai.onnx":
                return importer.version

        return None
    except:
        return None
