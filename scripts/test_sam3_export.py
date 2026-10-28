"""Export real SAM3 operators with tiny RANDOM weights; this is not model accuracy validation.

Uses the same six wrappers as export_sam3.py without gated checkpoint access.
"""
import argparse
import json
from pathlib import Path
import torch
from export_sam3 import wrappers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("cache/sam3-tiny/onnx"))
    a = p.parse_args()
    from transformers import Sam3Model, Sam3Config, Sam3TrackerModel, Sam3TrackerConfig, CLIPTextConfig
    from transformers.models.sam3.configuration_sam3 import (Sam3VisionConfig, Sam3ViTConfig, Sam3GeometryEncoderConfig,
        Sam3DETREncoderConfig, Sam3DETRDecoderConfig, Sam3MaskDecoderConfig)
    from transformers.models.sam3_tracker.configuration_sam3_tracker import Sam3TrackerPromptEncoderConfig, Sam3TrackerMaskDecoderConfig
    import numpy as np
    import onnxruntime as ort
    import onnx
    torch.manual_seed(123)
    torch.set_num_threads(2)
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    v = Sam3VisionConfig(backbone_config=Sam3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=28, patch_size=14, window_size=2, global_attn_indexes=[0], pretrain_image_size=28),
        fpn_hidden_size=32, backbone_feature_sizes=[[8, 8], [4, 4], [2, 2]])
    c = Sam3Config(vision_config=v, text_config=CLIPTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        projection_dim=16, num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8, bos_token_id=1, eos_token_id=2, pad_token_id=0),
        geometry_encoder_config=Sam3GeometryEncoderConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_attention_heads=4),
        detr_encoder_config=Sam3DETREncoderConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_attention_heads=4),
        detr_decoder_config=Sam3DETRDecoderConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_attention_heads=4, num_queries=2),
        mask_decoder_config=Sam3MaskDecoderConfig(hidden_size=32, num_attention_heads=4))
    c._attn_implementation = "eager"
    model = Sam3Model(c).eval()
    W = wrappers(torch)
    x = torch.randn(1, 3, 28, 28)
    ids = torch.tensor([[1, 7, 2, 0, 0, 0, 0, 0]])
    attention = (ids != 0).long()
    with torch.inference_mode():
        features = W[0](model)(x)
        text = W[1](model)(ids, attention)
        original = model.get_text_features(input_ids=ids, attention_mask=attention).pooler_output
        torch.testing.assert_close(text, original, atol=1e-6, rtol=1e-5)
    graphs = []
    def save(name, module, inputs, ins, outs, dynamic=None):
        path = root / (name + ".onnx")
        module.eval()
        with torch.inference_mode():
            expected = module(*inputs)
            expected = expected if isinstance(expected, tuple) else (expected,)
            torch.onnx.export(module, inputs, str(path), input_names=ins, output_names=outs, opset_version=17,
                              dynamo=False, dynamic_axes=dynamic, do_constant_folding=True)
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        feed = {key: value.numpy() for key, value in zip(ins, inputs) if key in {i.name for i in session.get_inputs()}}
        actual = session.run(None, feed)
        errors = {}
        for key, ref, result in zip(outs, expected, actual):
            np.testing.assert_allclose(result, ref.numpy(), atol=2e-4, rtol=2e-3)
            errors[key] = float(np.max(np.abs(result - ref.numpy())))
        graphs.append({"name": name, "onnx": str(path), "inputs": ins, "outputs": outs, "onnx_pytorch_max_error": errors})
        print("PASS ONNX", name, errors, flush=True)
        return session
    save("detector_vision", W[0](model), (x,), ["pixel_values"], ["f0", "f1", "f2", "p0", "p1", "p2"])
    save("text_encoder", W[1](model), (ids, attention), ["input_ids", "attention_mask"], ["text_features"])
    save("text_decoder", W[2](model), (*features, text, attention),
         ["f0", "f1", "f2", "p0", "p1", "p2", "text_features", "attention_mask"], ["boxes", "scores", "masks"])
    tc = Sam3TrackerConfig(vision_config=v,
        prompt_encoder_config=Sam3TrackerPromptEncoderConfig(hidden_size=32, image_size=28, patch_size=14, mask_input_channels=8),
        mask_decoder_config=Sam3TrackerMaskDecoderConfig(hidden_size=32, mlp_dim=64, num_hidden_layers=1, num_attention_heads=4, iou_head_hidden_dim=32))
    tc._attn_implementation = "eager"
    tracker = Sam3TrackerModel(tc).eval()
    with torch.inference_mode():
        tf = W[3](tracker)(x)
    box = torch.tensor([[[2., 2., 24., 24.]]])
    points = torch.tensor([[[[10., 10.], [1., 1.]]]])
    labels = torch.tensor([[[1, 0]]])
    save("tracker_vision", W[3](tracker), (x,), ["pixel_values"], ["f0", "f1", "f2"])
    save("box_decoder", W[4](tracker), (*tf, box), ["f0", "f1", "f2", "input_boxes"], ["masks", "iou_scores"])
    session = save("interactive_decoder", W[5](tracker), (*tf, box, points, labels),
         ["f0", "f1", "f2", "input_boxes", "input_points", "input_labels"], ["masks", "iou_scores"],
         {"input_points": {2: "points"}, "input_labels": {2: "points"}})
    # Validate dynamic point lengths, not only the traced length of two.
    for n in [1, 4, 8]:
        pp = points[:, :, :1].repeat(1, 1, n, 1)
        ll = torch.ones(1, 1, n, dtype=torch.int64)
        with torch.inference_mode():
            expected = W[5](tracker)(*tf, box, pp, ll)
        feed = dict(zip(["f0", "f1", "f2", "input_boxes", "input_points", "input_labels"], [i.numpy() for i in (*tf, box, pp, ll)]))
        for actual, ref in zip(session.run(None, feed), expected):
            np.testing.assert_allclose(actual, ref.numpy(), atol=2e-4, rtol=2e-3)
    # A tiny deterministic vocabulary validates the C++ BPE protocol separately.
    vocab = {"b": 3, "u": 4, "s</w>": 5, "bu": 6, "bus</w>": 7}
    (root / "vocab.tsv").write_text("1\t2\t0\n" + "".join(f"{i}\t{s.encode().hex()}\n" for s, i in vocab.items()))
    (root / "merges.tsv").write_text("0\t62\t75\n1\t6275\t733c2f773e\n")
    manifest = {"checkpoint": "TINY_RANDOM_TEST_ONLY_NOT_PRETRAINED", "max_points": 8, "text_length": 8,
                "graphs": graphs, "status": "SYNTHETIC_OPERATOR_EXPORT_PASS"}
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
