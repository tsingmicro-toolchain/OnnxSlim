import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Union

import numpy as np
import onnx
from onnx import checker

import onnxslim.onnx_graphsurgeon as gs
from onnxslim.onnx_graphsurgeon.ir.tensor import Constant
from onnxslim.onnx_graphsurgeon.logger.logger import G_LOGGER

logging.basicConfig(level=logging.ERROR)

from loguru import logger

from ..utils.utils import (
    dump_model_info_to_disk,
    gen_onnxruntime_input_data,
    onnxruntime_inference,
    print_model_info_as_table,
)

from .optimizer import delete_node, optimize_model

DEBUG = bool(os.getenv("ONNXSLIM_DEBUG"))


class OnnxSlim:
    def __init__(self, model: Union[str, onnx.ModelProto], log_level: int = 1):
        self.init_logging(log_level)
        if isinstance(model, str):
            self.model = onnx.load(model)
            self.model_name = Path(model).name
        else:
            self.model = model
            self.model_name = "ONNX_Model"

        self.freeze()
        self.raw_size = 0
        self.float_info = self.summarize_model(self.model)

    def init_logging(self, log_level: int):
        logger.remove()
        if log_level == 0 or DEBUG:  # DEBUG
            logger.add(sys.stderr, level=G_LOGGER.DEBUG)
        elif log_level == 1:  # INFO
            logger.add(sys.stderr, level=G_LOGGER.INFO)
        elif log_level == 2:  # WARNING
            logger.add(sys.stderr, level=G_LOGGER.WARNING)
        elif log_level == 3:  # ERROR
            logger.add(sys.stderr, level=G_LOGGER.ERROR)
        else:
            raise Exception("level must be 0, 1, 2 or 3")

        G_LOGGER.severity = G_LOGGER.ERROR
        import onnxruntime as ort

        ort.set_default_logger_severity(3)

    def clear_value_info(self) -> onnx.ModelProto:
        graph = gs.import_onnx(self.model)
        input_names = [input.name for input in graph.inputs]
        tensors = graph.tensors()
        for _, tensor in tensors.items():
            if tensor.name not in input_names:
                if isinstance(tensor, Constant):
                    continue
                tensor.shape = None

        self.model = gs.export_onnx(graph)

    def input_shape_modification(self, input_shapes: str) -> onnx.ModelProto:
        if not input_shapes:
            return

        graph = gs.import_onnx(self.model)
        input_names = [input.name for input in graph.inputs]
        tensors = graph.tensors()

        for input_shape in input_shapes:
            key, values = input_shape.rsplit(":", 1)
            values_list = [int(value) for value in values.split(",")]
            if key not in input_names:
                raise Exception(
                    f"Input name {key} not found in model, available keys: {' '.join(input_names)}"
                )
            tensors[key].shape = values_list

        for _, tensor in tensors.items():
            if tensor.name not in input_names:
                if isinstance(tensor, Constant):
                    continue
                tensor.shape = None

        self.model = gs.export_onnx(graph)

    def output_modification(self, outputs: str) -> onnx.ModelProto:
        graph = gs.import_onnx(self.model)
        graph.outputs.clear()
        tensors = graph.tensors()
        for output in outputs:
            values = output.rsplit(":", 1)
            if len(values) == 1:
                key = values[0]
                if key not in tensors.keys():
                    raise Exception(
                        f"Output name {key} not found in model, available keys: {' '.join(tensors.keys())}"
                    )
                dtype = tensors[key].dtype
                if dtype == None:
                    dtype = np.float32
                    logger.warning(
                        f"Output layer {key} has no dtype, set to default {dtype}"
                    )
            else:
                key, dtype = values
                if dtype == "fp16":
                    dtype = np.float16
                elif dtype == "fp32":
                    dtype = np.float32
                elif dtype == "int32":
                    dtype = np.int32
                elif dtype == "bool":
                    dtype = bool
                else:
                    raise Exception(
                        f"Output layer {key} assigned unsupported dtype {dtype}"
                    )

            graph.outputs.append(
                tensors[key].to_variable(dtype=dtype, shape=tensors[key].shape)
            )

        graph.cleanup(remove_unused_graph_inputs=True).toposort()
        self.model = gs.export_onnx(graph)

    def freeze(self):
        inputs = self.model.graph.input
        name_to_input = {}
        for input in inputs:
            if input.name in name_to_input.keys():
                logger.warning(f"Duplicate input name: {input.name}")
            name_to_input[input.name] = input

        for initializer in self.model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])
                name_to_input.pop(initializer.name)

    def check_onnx(self):
        self.input_data_dict = gen_onnxruntime_input_data(self.model)
        self.raw_onnx_output = onnxruntime_inference(self.model, self.input_data_dict)

    def check_point(self):
        self.graph_check_point = gs.import_onnx(self.model)

    def shape_infer(self):
        import onnxruntime.tools.symbolic_shape_infer as onnxrt_symbolic_shape_inference

        try:
            self.model = (
                onnxrt_symbolic_shape_inference.SymbolicShapeInference.infer_shapes(
                    self.model, auto_merge=True
                )
            )
        except:
            if self.model.ByteSize() >= checker.MAXIMUM_PROTOBUF:
                tmp_dir = tempfile.TemporaryDirectory()
                tmp_path = os.path.join(tmp_dir.name, "tmp.onnx")
                tmp_infer_path = os.path.join(tmp_dir.name, "tmp_infer.onnx")
                self.save(tmp_path)
                onnx.shape_inference.infer_shapes_path(tmp_path, tmp_infer_path)
                self.model = onnx.load(tmp_infer_path)
            else:
                self.model = onnx.shape_inference.infer_shapes(self.model)
        if DEBUG:
            onnx.save(self.model, "debug_shape_infer.onnx")

    def slim(self, skip_fusion_patterns: str = None):
        graph = gs.import_onnx(self.model).toposort()
        graph.fold_constants().cleanup().toposort()
        self.model = gs.export_onnx(graph)
        self.model = optimize_model(self.model, skip_fusion_patterns)
        if DEBUG:
            onnx.save(self.model, "debug_slim.onnx")

    def convert_data_format(self, dtype: str) -> onnx.ModelProto:
        if dtype == "fp16":
            from onnxconverter_common import float16

            self.model = float16.convert_float_to_float16(self.model)
        elif dtype == "fp32":
            graph = gs.import_onnx(self.model).toposort()

            for node in graph.nodes:
                if node.op == "Cast":
                    inp_dtype = [input.dtype for input in node.inputs][0]
                    if inp_dtype == np.float16 or inp_dtype == np.float32:
                        delete_node(node)

            for tensor in graph.tensors().values():
                if isinstance(tensor, gs.Variable) and tensor.dtype == np.float16:
                    tensor.dtype = np.float32
                elif isinstance(tensor, gs.Constant) and tensor.dtype == np.float16:
                    tensor.values = tensor.values.astype(np.float32)

            graph.cleanup(remove_unused_graph_inputs=True).toposort()
            self.model = gs.export_onnx(graph)

    def summary(self, inspect: bool = False, dump_to_disk: bool = False):
        if inspect:
            print_model_info_as_table(self.model_name, [self.float_info])
            if dump_to_disk:
                dump_model_info_to_disk(self.model_name, self.float_info)

        else:
            self.slimmed_info = self.summarize_model(self.model)
            print_model_info_as_table(
                self.model_name,
                [self.float_info, self.slimmed_info],
            )

    def summarize_model(self, model: onnx.ModelProto) -> Dict:
        model_info = {}

        model_size = model.ByteSize()
        model_info["model_size"] = model_size
        if self.raw_size:
            model_info["model_size"] = [model_size, self.raw_size]

        graph = gs.import_onnx(model)
        op_info = {}
        op_type_counts = {}

        for node in graph.nodes:
            op_type = node.op
            if op_type in op_type_counts:
                op_type_counts[op_type] += 1
            else:
                op_type_counts[op_type] = 1

            op_info[node.name] = [
                node.op,
                [[output.dtype, output.shape] for output in node.outputs],
            ]

        model_info["op_set"] = str(graph.opset)
        model_info["op_info"] = op_info
        model_info["op_type_counts"] = op_type_counts
        model_info["op_input_info"] = {
            input.name: str(input.dtype) + ": " + str(input.shape)
            for input in graph.inputs
        }
        model_info["op_output_info"] = {
            output.name: str(output.dtype) + ": " + str(output.shape)
            for output in graph.outputs
        }

        return model_info

    def save(self, model_path: str, model_check: bool = False):
        if model_check:
            try:
                checker.check_model(self.model)
            except ValueError:
                logger.warning("Model too large and cannot be checked.")

        if model_check:
            self.slimmed_onnx_output = onnxruntime_inference(
                self.model, self.input_data_dict
            )
            if set(self.raw_onnx_output.keys()) != set(self.slimmed_onnx_output.keys()):
                logger.warning("Model output mismatch after slimming.")
                logger.warning(
                    "Raw model output keys: {}".format(self.raw_onnx_output.keys())
                )
                logger.warning(
                    "Slimmed model output keys: {}".format(
                        self.slimmed_onnx_output.keys()
                    )
                )
                logger.warning("Please check the model carefully.")
                return
            else:
                for key in self.raw_onnx_output.keys():
                    if not np.allclose(
                        self.raw_onnx_output[key],
                        self.slimmed_onnx_output[key],
                        rtol=1e-03,
                        atol=1e-04,
                        equal_nan=True,
                    ):
                        logger.warning("Model output mismatch after slimming.")
                        logger.warning("Please check the model carefully.")
                        return

        if model_path:
            if (
                self.model.ByteSize() <= checker.MAXIMUM_PROTOBUF
            ):  # model larger than 2GB can be saved, but compiler like trtexec won't parse it
                self.raw_size = 0
                onnx.save(self.model, model_path)
            else:
                import os

                self.raw_size = self.model.ByteSize()
                location = os.path.basename(model_path) + ".data"
                if os.path.exists(location):
                    os.remove(location)
                onnx.save(
                    self.model,
                    model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=location,
                )
                logger.debug(
                    "Model too large and saved as external data automatically."
                )

    def is_converged(self, iter: int) -> bool:
        logger.debug(f"optimization iter: {iter}")
        graph = gs.import_onnx(self.model)
        if graph == self.graph_check_point:
            logger.debug(f"converged at iter: {iter}")
            return True
        else:
            self.graph_check_point = graph
            return False
