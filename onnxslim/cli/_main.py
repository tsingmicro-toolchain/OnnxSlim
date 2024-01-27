from typing import Union

import onnx


def slim(
    model: Union[str, onnx.ModelProto],
    output_model: str = None,
    model_check: bool = False,
    input_shapes: str = None,
    outputs: str = None,
    no_shape_infer: bool = False,
    no_constant_folding: bool = False,
    dtype: str = None,
    skip_fusion_patterns: str = None,
    inspect: bool = False,
    dump_to_disk: bool = False,
):
    import os

    from onnxslim.core.slim import OnnxSlim

    MAX_ITER = (
        10
        if not os.getenv("ONNXSLIM_MAX_ITER")
        else int(os.getenv("ONNXSLIM_MAX_ITER"))
    )

    slimmer = OnnxSlim(model)
    if inspect:
        slimmer.summary(inspect, dump_to_disk)
        return None

    if input_shapes:
        slimmer.input_shape_modification(input_shapes)

    if outputs:
        slimmer.output_modification(outputs)

    if model_check:
        slimmer.check_point()

    slimmer.shape_infer()

    if not no_constant_folding:
        while MAX_ITER > 0:
            slimmer.slim(skip_fusion_patterns)
            slimmer.shape_infer()
            if slimmer.is_converged(MAX_ITER):
                break
            MAX_ITER -= 1

    if dtype:
        slimmer.convert_data_format(dtype)

    slimmer.save(output_model, model_check)

    if not output_model:
        return slimmer.model
    else:
        slimmer.summary()


def main():
    import argparse

    from loguru import logger

    import onnxslim

    parser = argparse.ArgumentParser(
        description="OnnxSlim: A Toolkit to Help Optimizer Onnx Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_model", help="input onnx model")
    parser.add_argument(
        "output_model", nargs="?", default=None, help="output onnx model"
    )

    parser.add_argument("--model_check", action="store_true", help="enable model check")
    parser.add_argument(
        "-v", "--version", action="version", version=onnxslim.__version__
    )

    # Input Shape Modification
    parser.add_argument(
        "--input_shapes",
        nargs="+",
        type=str,
        help="input shape of the model, INPUT_NAME:SHAPE, e.g. x:1,3,224,224 or x1:1,3,224,224 x2:1,3,224,224",
    )
    # Output Modification
    parser.add_argument(
        "--outputs",
        nargs="+",
        type=str,
        help="output of the model, OUTPUT_NAME:DTYPE, e.g. y:fp32 or y1:fp32 y2:fp32. \
                                                                             If dtype is not specified, the dtype of the output will be the same as the original model \
                                                                             if it has dtype, otherwise it will be fp32, available dtype: fp16, fp32, int32",
    )
    # Shape Inference
    parser.add_argument(
        "--no_shape_infer",
        action="store_true",
        help="whether to disable shape_infer, default false.",
    )

    # Constant Folding
    parser.add_argument(
        "--no_constant_folding",
        action="store_true",
        help="whether to disable constant_folding, default false.",
    )

    # Data Format Conversion
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        help="convert data format to fp16 or fp32.",
    )

    parser.add_argument(
        "--skip_fusion_patterns",
        nargs="+",
        choices=list(onnxslim.DEFAULT_FUSION_PATTERNS.keys()),
        help="whether to skip the fusion of some patterns",
    )

    # Inspect Model
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="inspect model, default False.",
    )

    # Dump Model Info to Disk
    parser.add_argument(
        "--dump_to_disk",
        action="store_true",
        help="dump model info to disk, default False.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        logger.error(f"unrecognized options: {unknown}")
        return 1

    if args.inspect and args.output_model:
        parser.error("--inspect and output_model are mutually exclusive")

    if not args.inspect and args.dump_to_disk:
        parser.error("dump_to_disk can only be used with --inspect")

    slim(
        args.input_model,
        args.output_model,
        args.model_check,
        args.input_shapes,
        args.outputs,
        args.no_shape_infer,
        args.no_constant_folding,
        args.dtype,
        args.skip_fusion_patterns,
        args.inspect,
        args.dump_to_disk,
    )

    return 0
