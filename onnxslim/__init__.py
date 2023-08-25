__version__ = "0.0.1"

import os
import warnings


if os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "onnxslim"
):
    message = (
        "You are importing onnxslim within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "onnxslim project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))