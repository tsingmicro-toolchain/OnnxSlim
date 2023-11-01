def slim(
    model,
    model_check=None,
    output_model=None,
    optimization=None,
    input_shapes=None,
    outputs=None,
    shape_infer=None,
    constant_folding=None,
    dtype=None,
):
    import os

    from onnxslim.core.slim import OnnxSlim

    MAX_ITER = (
        10
        if not os.getenv("ONNXSLIM_MAX_ITER")
        else int(os.getenv("ONNXSLIM_MAX_ITER"))
    )

    slimmer = OnnxSlim(model)
    if optimization and input_shapes:
        slimmer.input_shape_modification(input_shapes)

    if optimization and outputs:
        slimmer.output_modification(outputs)

    if model_check:
        slimmer.check_point()

    if optimization == None:
        slimmer.shape_infer()
    elif shape_infer == "enable":
        data_prop = False
        if data_prop == "enable":
            data_prop = True
        slimmer.shape_infer()

    if optimization == None or constant_folding == "enable":
        while MAX_ITER > 0:
            slimmer.slim()
            slimmer.shape_infer()
            if slimmer.is_converged(MAX_ITER):
                break
            MAX_ITER -= 1

    if optimization and dtype:
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
    parser.add_argument("output_model", help="output onnx model")
    parser.add_argument("--model_check", action="store_true", help="enable model check")
    parser.add_argument(
        "-v", "--version", action="version", version=onnxslim.__version__
    )
    subparsers = parser.add_subparsers(title="Optimization", dest="optimization")

    optimization_parser = subparsers.add_parser(
        "optimization", help="perform optimization"
    )
    # Input Shape Modification
    optimization_parser.add_argument(
        "--input_shapes",
        nargs="+",
        type=str,
        help="input shape of the model, INPUT_NAME:SHAPE, e.g. x:1,3,224,224 or x1:1,3,224,224 x2:1,3,224,224",
    )
    # Output Modification
    optimization_parser.add_argument(
        "--outputs",
        nargs="+",
        type=str,
        help="output of the model, OUTPUT_NAME:DTYPE, e.g. y:fp32 or y1:fp32 y2:fp32. \
                                                                             If dtype is not specified, the dtype of the output will be the same as the original model \
                                                                             if it has dtype, otherwise it will be fp32, available dtype: fp16, fp32, int32",
    )
    # Shape Inference
    optimization_parser.add_argument(
        "--shape_infer",
        choices=["enable", "disable"],
        default="enable",
        help="whether to enable shape_infer, default enable.",
    )

    # Constant Folding
    optimization_parser.add_argument(
        "--constant_folding",
        choices=["enable", "disable"],
        default="enable",
        help="whether to enable shape_infer, default enable.",
    )

    # Data Format Conversion
    optimization_parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        help="whether to enable shape_infer, default enable.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        logger.error(f"unrecognized options: {unknown}")
        return 1

    inputs_shapes = None if not args.optimization else args.input_shapes
    outputs = None if not args.optimization else args.outputs
    shape_infer = None if not args.optimization else args.shape_infer
    constant_folding = None if not args.optimization else args.constant_folding
    dtype = None if not args.optimization else args.dtype

    slim(
        args.input_model,
        args.model_check,
        args.output_model,
        args.optimization,
        inputs_shapes,
        outputs,
        shape_infer,
        constant_folding,
        dtype,
    )

    return 0
