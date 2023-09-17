def main():
    import sys    
    import argparse
    from loguru import logger
    import onnxslim
    from onnxslim.slim import OnnxSlim

    common_parser = argparse.ArgumentParser(add_help=False)

    parser = argparse.ArgumentParser(
        description="OnnxSlim: A Toolkit to Help Optimizer Onnx Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_model", help="Input onnx model")
    parser.add_argument("output_model", help="Output onnx model")
    parser.add_argument("--no_model_check", action="store_true", help="Disable model check")
    parser.add_argument(
        "-v", "--version", action="version", version=onnxslim.__version__
    )
    subparsers = parser.add_subparsers(title="Optimization", dest="optimization")
    
    optimization_parser = subparsers.add_parser("optimization", help="Perform Optimization")
    # Input Shape Modification
    optimization_parser.add_argument("--input_shapes", nargs="+", type=str, help="Input shape of the model, INPUT_NAME:SHAPE, e.g. x:1,3,224,224 or x1:1,3,224,224 x2:1,3,224,224")
    #Output Modification
    optimization_parser.add_argument("--outputs", nargs="+", type=str, help="Output of the model, OUTPUT_NAME:DTYPE, e.g. y:fp32 or y1:fp32 y2:fp32. \
                                                                             If dtype is not specified, the dtype of the output will be the same as the original model \
                                                                             if it has dtype, otherwise it will be fp32, available dtype: fp16, fp32, int32")
    # Shape Inference
    optimization_parser.add_argument("--shape_infer", choices=['enable', 'disable'], default="enable", help="Whether to enable shape_infer, default enable.")
    optimization_parser.add_argument("--data_prop", choices=['enable', 'disable'], default="enable", help="Whether to do data_prop, default enable.")
    
    # Constant Folding
    optimization_parser.add_argument("--constant_folding", choices=['enable', 'disable'], default="enable", help="Whether to enable shape_infer, default enable.")

    # Data Format Conversion
    optimization_parser.add_argument("--dtype", choices=['fp16', 'fp32'], help="Whether to enable shape_infer, default enable.")
    args, unknown = parser.parse_known_args()       

    if unknown:                                                                                                                                                                                                                                                                                                                                                                                          
        logger.error(f"Unrecognized Options: {unknown}")
        return 1

    slimmer = OnnxSlim(args.input_model)
    if args.optimization and args.input_shapes:
        slimmer.input_shape_modification(args.input_shapes)

    if args.optimization and args.outputs:
        slimmer.output_modification(args.outputs)

    if not args.no_model_check:
        slimmer.check_point()

    if args.optimization == None:
        slimmer.shape_infer()
    elif args.shape_infer == 'enable':
        data_prop = False
        if args.data_prop == 'enable':
            data_prop = True
        slimmer.shape_infer(data_prop)

    if args.optimization == None or args.constant_folding == 'enable':
        slimmer.slim()

    if args.optimization and args.dtype:
        slimmer.convert_data_format(args.dtype)
    slimmer.summary()
    slimmer.save(args.output_model, args.no_model_check)

    return 0