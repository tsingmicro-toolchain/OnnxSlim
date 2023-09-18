import onnx
import numpy as np


def format_bytes(size_in_bytes):
    # Define the units and their corresponding suffixes
    units = ['B', 'KB', 'MB', 'GB']
    
    # Determine the appropriate unit and conversion factor
    unit_index = 0
    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024
        unit_index += 1
    
    # Format the result with two decimal places
    formatted_size = "{:.2f} {}".format(size_in_bytes, units[unit_index])
    return formatted_size


def onnx_dtype_to_numpy(onnx_dtype):
    import onnx.mapping as mapping
    return np.dtype(mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype])


def gen_onnxruntime_input_data(model):
    input_info = [(input_tensor.name, 
                [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim],
                onnx_dtype_to_numpy(input_tensor.type.tensor_type.elem_type)) 
                for input_tensor in model.graph.input]
    
    input_data_dict = {}
    for name, shapes, dtype in input_info:
        shapes = [shape if shape != -1 else 1 for shape in shapes]
        random_data = np.random.rand(*shapes).astype(dtype)
        input_data_dict[name] = random_data
        
    return input_data_dict
    

def onnxruntime_inference(model, input_data):
    import onnxruntime as rt
    sess = rt.InferenceSession(model.SerializeToString())
    onnx_output = sess.run(None, input_data)

    output_names = [output.name for output in sess.get_outputs()]
    onnx_output = dict(zip(output_names, onnx_output))

    return onnx_output
