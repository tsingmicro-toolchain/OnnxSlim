# OnnxSlim

OnnxSlim can help you slim your onnx model, with less operators, but same accuracy, better inference speed.

- 🚀 OnnxSlim is merged to [mnn-llm](https://github.com/wangzhaode/mnn-llm), performance increased by 5%
- 🚀 Rank 1st in the [AICAS 2024 LLM inference optimiztion challenge](https://tianchi.aliyun.com/competition/entrance/532170/customize440) held by Arm and T-head


# Installation
## Using Prebuilt
```bash
pip install onnxslim
```
## Build From Source
```
pip install .
```


# How to use
```
onnxslim your_onnx_model slimmed_onnx_model
```

<div align=left><img src="images/onnxslim.gif"></div>

For more usage, see onnxslim -h or refer to our [examples](./examples)

# References
> * [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
> * [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)
> * [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
> * [tabulate](https://github.com/astanin/python-tabulate)
> * [onnxruntime](https://github.com/microsoft/onnxruntime)

# Contact
Discord: https://discord.gg/nRw2Fd3VUS  
QQ Group: 873569894
