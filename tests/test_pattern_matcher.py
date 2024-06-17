import os
import pytest

import torch
import torch.nn as nn

from onnxslim import slim


class TestPatternMatcher:
    def test_gelu(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.relu0 = nn.ReLU()
                self.gelu = nn.GELU()
                self.relu1 = nn.ReLU()

            def forward(self, x):
                x = self.relu0(x)
                x = self.gelu(x)
                x = self.relu1(x)
                return x

        input = torch.randn(2)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        slim(filename, filename)

    def test_pad_conv(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.pad_0 = nn.ConstantPad2d(3, 0)
                self.conv_0 = nn.Conv2d(1, 1, 3)

                self.pad_1 = nn.ConstantPad2d(3, 2)
                self.conv_1 = nn.Conv2d(1, 1, 3, bias=False)

            def forward(self, x):
                x0 = self.pad_0(x)
                x0 = self.conv_0(x0)

                x1 = self.pad_1(x)
                x1 = self.conv_1(x1)

                return x0 + x1

        input = torch.randn(1, 1, 24, 24)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        slim(filename, filename)

    def test_conv_bn(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)
                self.bn = nn.BatchNorm2d(1)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        input = torch.randn(1, 1, 24, 24)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename, do_constant_folding=False)

        slim(filename, filename)

    def test_consecutive_slice(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)
                self.bn = nn.BatchNorm2d(1)

            def forward(self, x):
                return x[1:2, :2]

        input = torch.randn(3, 4)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        slim(filename, filename)

    def test_consecutive_reshape(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return x.view(2, 6).view(12, 1)

        input = torch.randn(3, 4)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        slim(filename, filename)

    def test_matmul_add(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.data = torch.randn(4, 3)

            def forward(self, x):
                x = torch.matmul(x, self.data)
                x += 1
                return x

        input = torch.randn(3, 4)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        slim(filename, filename)

    def test_reduce(self, request):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = torch.sum(x, dim=[-1], keepdim=False)
                x = x.unsqueeze(-1)
                return x

        input = torch.randn(3, 4)
        m = Model()
        directory = "tmp/" + request.node.name
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename, opset_version=11)

        slim(filename, filename)


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_pattern_matcher.py",
        ]
    )
