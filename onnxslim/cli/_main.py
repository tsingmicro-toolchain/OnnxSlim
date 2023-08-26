def main():
    import argparse
    import sys

    from loguru import logger
    from onnxslim.slim import OnnxSlim

    parser = argparse.ArgumentParser(
        description="OnnxSlim: A Toolkit to Help Optimizer Onnx Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_model", help="Output ONNX model")    
    parser.add_argument(
        "-v", "--version", action="version", version="0.0.1"
    )

    args, unknown = parser.parse_known_args()       
                 
    if unknown:                                                                                                                                                                                                                                                                                                                                                                                          
        logger.error(f"Unrecognized Options: {unknown}")
        return 1

    slimmer = OnnxSlim(args.input_model)
    slimmer.shape_infer()
    slimmer.slim()
    # slimmer.convert_data_format()
    slimmer.summary()
    slimmer.save(args.output_model)

    return 0