import sys
import onnx
from onnx import checker
import numpy as np
from tabulate import tabulate, SEPARATING_LINE
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Constant

from loguru import logger
from .utils.font import GREEN, WHITE
from .utils.utils import format_bytes, gen_onnxruntime_input_data, onnxruntime_inference

class OnnxSlim():
    def __init__(self, model, log_level=1):
        self.init_logging(log_level)        
        self.model = onnx.load(model)
        self.float_info = self.summarize(self.model)


    def init_logging(self, log_level):
        logger.remove()
        if log_level == 0: # DEBUG
            logger.add(sys.stderr, level=G_LOGGER.DEBUG)
        elif log_level == 1: # INFO
            logger.add(sys.stderr, level=G_LOGGER.INFO)
        elif log_level == 2: # WARNING
            logger.add(sys.stderr, level=G_LOGGER.WARNING)
        elif log_level == 3: # ERROR
            logger.add(sys.stderr, level=G_LOGGER.ERROR)
        else:
            raise Exception("level must be 0, 1, 2 or 3")
        
        G_LOGGER.severity = G_LOGGER.ERROR


    def input_shape_modification(self, input_shapes):
        if not input_shapes:
            return

        graph = gs.import_onnx(self.model)
        input_names = [input.name for input in graph.inputs]
        tensors = graph.tensors()

        for input_shape in input_shapes:
            key, values = input_shape.split(':')
            values_list = [int(value) for value in values.split(',')]
            if key not in input_names:
                raise Exception(f"Input name {key} not found in model, available keys: {' '.join(input_names)}")
            tensors[key].shape = values_list

        for _, tensor in tensors.items():
            if tensor.name not in input_names:
                if isinstance(tensor, Constant):
                    continue
                tensor.shape = None

        self.model = gs.export_onnx(graph)


    def output_modification(self, outputs):
        graph = gs.import_onnx(self.model)
        graph.outputs.clear()
        tensors = graph.tensors()
        for output in outputs:
            values = output.split(':')
            if len(values) == 1:
                key = values[0]
                if key not in tensors.keys():
                    raise Exception(f"Output name {key} not found in model, available keys: {' '.join(tensors.keys())}")
                dtype = tensors[key].dtype
                if dtype == None:
                    dtype = np.float32
                    logger.warning(f"Output layer {key} has no dtype, set to default {dtype}")
            else:
                key, dtype = values
                if dtype == 'fp16':
                    dtype = np.float16
                elif dtype == 'fp32':
                    dtype = np.float32
                elif dtype == 'int32':
                    dtype = np.int32
                else:
                    raise Exception(f"Output layer {key} assigned unsupported dtype {dtype}")

            graph.outputs.append(tensors[key].to_variable(dtype=dtype, shape=tensors[key].shape))

        graph.cleanup(remove_unused_node_outputs=True, remove_unused_graph_inputs=True).toposort()
        self.model = gs.export_onnx(graph)


    def check_point(self):
        self.input_data_dict = gen_onnxruntime_input_data(self.model)
        self.raw_onnx_output = onnxruntime_inference(self.model, self.input_data_dict)        


    def shape_infer(self, data_prop=True):
        self.model = onnx.shape_inference.infer_shapes(self.model, strict_mode=False, data_prop=data_prop)


    def slim(self):
        graph = gs.import_onnx(self.model).toposort()
        graph.fold_constants().cleanup().toposort()
        self.model = gs.export_onnx(graph)


    def convert_data_format(self, dtype):
        if dtype == 'fp16':
            from onnxconverter_common import float16
            self.model = float16.convert_float_to_float16(self.model)
        elif dtype == 'fp32':
            graph = gs.import_onnx(self.model).toposort()
            for tensor in graph.tensors().values():
                if isinstance(tensor, gs.Variable) and tensor.dtype == np.float16:
                    tensor.dtype = np.float32
                elif isinstance(tensor, gs.Constant) and tensor.dtype == np.float16:
                    tensor.values = tensor.values.astype(np.float32)

        self.model = gs.export_onnx(graph)


    def summary(self):
        self.slimmed_info = self.summarize(self.model)
        final_op_info = []
        all_ops = set(list(self.float_info['op_type_counts'].keys()) + list(self.slimmed_info['op_type_counts'].keys()))
        sorted_ops = sorted(all_ops, key=lambda x: x[0])
        for op in sorted_ops:
            float_number = self.float_info['op_type_counts'].get(op, 0)
            slimmed_number = self.slimmed_info['op_type_counts'].get(op, 0)
            if float_number > slimmed_number:
                slimmed_number = GREEN+str(slimmed_number)+WHITE
            final_op_info.append([op, float_number, slimmed_number])
        final_op_info.append([SEPARATING_LINE, SEPARATING_LINE, SEPARATING_LINE])
        final_op_info.append(["model_size", format_bytes(self.float_info["model_size"]),
                                            format_bytes(self.slimmed_info["model_size"])])
        lines = tabulate(final_op_info, headers=["OP_TYPE", "Original Model", "Slimmed Model"],
                         tablefmt="pretty").split('\n')
        output = "\n".join([line if line !='| \x01 |' else lines[0] for line in lines])

        logger.info(WHITE+"\n" + output)


    def summarize(self, model):
        model_info = {}        
        
        model_size = model.ByteSize()
        model_info["model_size"] = model_size
        
        op_type_counts = {}

        for node in model.graph.node:
            op_type = node.op_type
            if op_type in op_type_counts:
                op_type_counts[op_type] += 1
            else:
                op_type_counts[op_type] = 1

        model_info["op_type_counts"] = op_type_counts

        return model_info             


    def save(self, model_path, no_model_check=False):
        if not no_model_check:
            self.slimmed_onnx_output = onnxruntime_inference(self.model, self.input_data_dict)
            if set(self.raw_onnx_output.keys()) != set(self.slimmed_onnx_output.keys()):
                logger.warning("Model output mismatch after slimming.")
                logger.warning("Raw model output keys: {}".format(self.raw_onnx_output.keys()))
                logger.warning("Slimmed model output keys: {}".format(self.slimmed_onnx_output.keys()))
                logger.warning("Please check the model carefully.")
                logger.warning("If you are sure that the model is correct, please use --no_model_check to skip model check.")
                return
            else:
                for key in self.raw_onnx_output.keys():
                    if not np.allclose(self.raw_onnx_output[key], self.slimmed_onnx_output[key], rtol=1e-03, atol=1e-04, equal_nan=True):
                        logger.warning("Model output mismatch after slimming.")
                        logger.warning("Please check the model carefully.")
                        logger.warning("If you are sure that the model is correct, please use --no_model_check to skip model check.")
                        return

        if not no_model_check:
            try:
                checker.check_model(self.model)
            except ValueError:
                logger.warning("Model too large and cannot be checked.")
            
        try:
            onnx.save(self.model, model_path)
        except ValueError:
            import os
            onnx.save(
                self.model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=os.path.basename(model_path) + '.data',
            )
            logger.warning("Model too large and saved as external data automatically.")
