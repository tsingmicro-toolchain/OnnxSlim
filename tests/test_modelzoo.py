import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest

from onnxslim import slim
from onnxslim.utils import print_model_info_as_table, summarize_model

MODELZOO_PATH = "/data/modelzoo"


class TestModelZoo:
    def test_silero_vad(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            batch_size = 2
            input = np.zeros((batch_size, 256), dtype=np.float32)
            sr = np.array(16000)
            state = np.zeros((2, batch_size, 128), dtype=np.float32)

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"input": input, "sr": sr, "state": state})

            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert summary.op_type_counts["Slice"] == 4

    def test_decoder_with_past_model(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            batch_size = 2
            input_ids = np.ones((batch_size, 256), dtype=np.int64)
            encoder_hidden_states = np.zeros((batch_size, 128, 16), dtype=np.float32)

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states})

    def test_tiny_en_decoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"), model_check=True)

    def test_transformer_encoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        summary = summarize_model(slim(filename), tag=request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Mul"] == 57
        assert summary.op_type_counts["Div"] == 53

    def test_uiex(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        summary = summarize_model(slim(filename), tag=request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Range"] == 0
        assert summary.op_type_counts["Floor"] == 0
        assert summary.op_type_counts["Concat"] == 54

    def test_qwen_vl_vision_encoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        summary = summarize_model(slim(filename), tag=request.node.name)
        print_model_info_as_table(summary)
        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            import numpy as np
            import onnxruntime as ort

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            outputs = ort_sess.run(
                None,
                {"pixel_values": np.random.rand(256, 1176).astype(np.float32), "grid_thw": np.array([[1, 16, 16]])},
            )
            print(f"{outputs[0].shape=}")  # (64, 16)

    def test_layer_normalization_2d_axis0_expanded_ver18(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"), model_check=True)
            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert summary.op_type_counts["Reshape"] == 1

    def test_padconv(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(
                filename,
                os.path.join(tempdir, f"{name}_slim.onnx"),
                model_check=True,
                input_shapes=["/encoder/encoders0/encoders0.0/self_attn/Transpose_2_output_0:1,516,32"],
            )

    def test_wav2vec2_conformer(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            batch_size = 2
            input = np.zeros((batch_size, 256), dtype=np.float32)

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"input_values": input})

    def test_yolo11n_pose(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            input = np.zeros((1, 3, 256, 256), dtype=np.float32)

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"images": input})

    def test_linguistic(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"), no_shape_infer=False, verbose=False)
            import torch

            tokens = torch.LongTensor([[1] * 5]).numpy()
            word_div = torch.LongTensor([[2, 2, 1]]).numpy()
            word_dur = torch.LongTensor([[8, 3, 4]]).numpy()
            languages = torch.LongTensor([[0] * 5]).numpy()

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"tokens": tokens, "word_div": word_div, "word_dur": word_dur, "languages": languages})

    def test_linear_mul_fusion(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"), model_check=True)
            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert summary.op_type_counts["MatMul"] == 0
            assert summary.op_type_counts["Mul"] == 0
            assert summary.op_type_counts["Add"] == 0

            filename = f"{MODELZOO_PATH}/{name}/{name}2.onnx"
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"), model_check=True)
            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert summary.op_type_counts["MatMul"] == 0
            assert summary.op_type_counts["Mul"] == 1
            assert summary.op_type_counts["Add"] == 1

    def test_Qwen3_0_6B_Q4(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert summary.op_type_counts["Slice"] == 4

    def test_gpt2(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert summary.op_type_counts["Cast"] == 3

    def test_custom(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            summary = summarize_model(os.path.join(tempdir, f"{name}_slim.onnx"), tag=request.node.name)
            assert len(summary.op_info["/avgpool/GlobalAveragePool"].outputs) == 1
            assert summary.op_info["/avgpool/GlobalAveragePool"].outputs[0].shape == (1, 2048, 1, 1)


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:warnings",
                "-sv",
                "tests/test_modelzoo.py",
            ]
        )
    )
