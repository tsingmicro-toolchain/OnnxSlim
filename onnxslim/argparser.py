import argparse
import dataclasses
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union, get_args, get_origin, TypedDict, Dict, Literal

from .core.optimization import OptimizationSettings
from .core.pattern.registry import DEFAULT_FUSION_PATTERNS
from .version import __version__


class OnnxSlimKwargs(TypedDict, total=False):
    model_check: bool
    input_shapes: Dict[str, List[int]]
    inputs: List[str]
    outputs: List[str]
    no_shape_infer: bool
    skip_optimizations: List[str]
    dtype: Literal["float16", "float32", "uint8", "int8"]
    skip_fusion_patterns: List[str]
    size_threshold: int
    inspect: bool
    dump_to_disk: bool
    save_as_external_data: bool
    model_check_inputs: Optional[List[str]]
    verbose: bool

def _get_inner_type(arg_type):
    if get_origin(arg_type) is Union:
        return next((t for t in get_args(arg_type) if t is not type(None)), str)
    return arg_type


@dataclass
class ModelArguments:
    """
    Args:
        model (Union[str, onnx.ModelProto]): The ONNX model to be slimmed. It can be either a file path or an `onnx.ModelProto` object.

        output_model (str, optional): File path to save the slimmed model. If None, the model will not be saved.
    """

    input_model: str = field(metadata={"help": "input onnx model"})
    output_model: Optional[str] = field(default=None, metadata={"help": "output onnx model"})


@dataclass
class OptimizationArguments:
    """
    Args:
        no_shape_infer (bool, optional): Flag indicating whether to perform shape inference. Default is False.

        no_constant_folding (bool, optional): Flag indicating whether to perform constant folding. Default is False.

        skip_fusion_patterns (str, optional): String representing fusion patterns to skip. Default is None.
    """

    no_shape_infer: bool = field(default=False, metadata={"help": "whether to disable shape_infer, default false."})
    skip_optimizations: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "whether to skip some optimizations",
            "choices": list(OptimizationSettings.keys()),
        },
    )
    skip_fusion_patterns: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "whether to skip the fusion of some patterns",
            "choices": list(DEFAULT_FUSION_PATTERNS.keys()),
        },
    )
    size_threshold: int = field(
        default=None,
        metadata={
            "help": "size threshold in bytes, size larger than this value will not be folded, default None, which means fold all constants",
        },
    )


@dataclass
class ModificationArguments:
    """
    Args:
        input_shapes (str, optional): String representing the input shapes. Default is None.

        outputs (str, optional): String representing the outputs. Default is None.

        dtype (str, optional): Data type. Default is None.

        save_as_external_data (bool, optional): Flag indicating whether to split onnx as model and weight. Default is False.
    """

    input_shapes: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "input shape of the model, INPUT_NAME:SHAPE, e.g. x:1,3,224,224 or x1:1,3,224,224 x2:1,3,224,224"
        },
    )
    inputs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "input of the model, INPUT_NAME:DTYPE, e.g. y:fp32 or y1:fp32 y2:fp32. If dtype is not specified, the dtype of the input will be the same as the original model if it has dtype, otherwise it will be fp32, available dtype: fp16, fp32, int32"
        },
    )
    outputs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "output of the model, OUTPUT_NAME:DTYPE, e.g. y:fp32 or y1:fp32 y2:fp32. If dtype is not specified, the dtype of the output will be the same as the original model if it has dtype, otherwise it will be fp32, available dtype: fp16, fp32, int32"
        },
    )
    dtype: Optional[str] = field(
        default=None, metadata={"help": "convert data format to fp16 or fp32.", "choices": ["fp16", "fp32"]}
    )
    save_as_external_data: bool = field(
        default=False, metadata={"help": "split onnx as model and weight, default False."}
    )


@dataclass
class CheckerArguments:
    """
    Args:
        model_check (bool, optional): Flag indicating whether to perform model checking. Default is False.

        model_check_inputs (str, optional): The shape or tensor used for model check. Default is None.

        inspect (bool, optional): Flag indicating whether to inspect the model. Default is False.

        dump_to_disk (bool, optional): Flag indicating whether to dump the model detail to disk. Default is False.

        verbose (bool, optional): Flag indicating whether to print verbose logs. Default is False.
    """

    model_check: bool = field(default=False, metadata={"help": "enable model check"})
    model_check_inputs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Works only when model_check is enabled, Input shape of the model or numpy data path, INPUT_NAME:SHAPE or INPUT_NAME:DATAPATH, e.g. x:1,3,224,224 or x1:1,3,224,224 x2:data.npy. Useful when input shapes are dynamic."
        },
    )
    inspect: bool = field(default=False, metadata={"help": "inspect model, default False."})
    dump_to_disk: bool = field(default=False, metadata={"help": "dump model info to disk, default False."})
    verbose: bool = field(default=False, metadata={"help": "verbose mode, default False."})


class OnnxSlimArgumentParser(ArgumentParser):
    def __init__(self, *argument_dataclasses: Type, **kwargs):
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        self.argument_dataclasses = argument_dataclasses
        self.parser = argparse.ArgumentParser(
            description="OnnxSlim: A Toolkit to Help Optimizer Onnx Model",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_arguments()

    def _add_arguments(self):
        for dataclass_type in self.argument_dataclasses:
            if dataclass_type is ModelArguments:
                continue
            for field_name, field_def in dataclass_type.__dataclass_fields__.items():
                arg_type = _get_inner_type(field_def.type)
                default_value = field_def.default if field_def.default is not field_def.default_factory else None
                help_text = field_def.metadata.get("help", "")
                nargs = "+" if get_origin(arg_type) == list else None
                choices = field_def.metadata.get("choices", None)
                if choices and default_value is not None and default_value not in choices:
                    raise ValueError(
                        f"Invalid default value '{default_value}' for argument '{field_name}'. Must be one of {choices}."
                    )
                arg_type = get_args(arg_type)[0] if get_args(arg_type) else arg_type
                if arg_type == bool:
                    self.parser.add_argument(
                        f"--{field_name.replace('_', '-')}",
                        action="store_true",
                        default=default_value,
                        help=help_text,
                    )
                else:
                    self.parser.add_argument(
                        f"--{field_name.replace('_', '-')}",
                        type=arg_type,
                        default=default_value,
                        nargs=nargs,
                        choices=choices,
                        help=help_text,
                    )

        # Add positional arguments separately for ModelArguments
        self.parser.add_argument("input_model", help="input onnx model")
        self.parser.add_argument("output_model", nargs="?", default=None, help="output onnx model")
        self.parser.add_argument("-v", "--version", action="version", version=__version__)

    def parse_args_into_dataclasses(self):
        # Pre-parse arguments to check for `--inspect`
        pre_parsed_args, _ = self.parser.parse_known_args()
        if pre_parsed_args.inspect:
            for action in self.parser._actions:
                if action.dest == "input_model":
                    action.nargs = "+"
                    break

        args = self.parser.parse_args()
        args_dict = vars(args)

        outputs = []
        for dtype in self.argument_dataclasses:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args_dict.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        return (*outputs,)
