"""Export official WeVisOne-VL weights into vision, cached language and FP32 pixel ONNX graphs.

Use a separate Python environment with the official dependencies (Transformers
5.12.1 / PyTorch 2.11), onnx and onnxscript. No weights are bundled with cvinfer.
The fixed vision grid is deliberate; text, KV length and object count are dynamic.
"""
import argparse
import gc
import json
from pathlib import Path

from wevisone_checkpoint import TASKS, export_tokenizer, sha256, source_path


def wrappers(torch):
    F = torch.nn.functional

    def rotate(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    class Vision(torch.nn.Module):
        def __init__(self, visual, gh, gw, merge):
            super().__init__()
            from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
            self.visual, self.gh, self.gw, self.merge = visual, gh, gw, merge
            grid = torch.tensor([[1, gh, gw]], device=visual.pos_embed.weight.device)
            indices, weights = get_vision_bilinear_indices_and_weights(
                grid, num_grid_per_side=visual.num_grid_per_side, spatial_merge_size=merge)
            pos = (visual.pos_embed(indices) * weights[:, :, None]).sum(0)
            position_ids = get_vision_position_ids(grid, merge)
            rotary = visual.rotary_pos_emb(position_ids).reshape(gh * gw, -1)
            rotary = torch.cat((rotary, rotary), dim=-1)
            self.register_buffer("absolute", pos.detach())
            self.register_buffer("cos", rotary.cos().detach())
            self.register_buffer("sin", rotary.sin().detach())

        def feature_map(self, x):
            h, w, m = self.gh, self.gw, self.merge
            return x.reshape(h // m, w // m, m, m, -1).permute(0, 2, 1, 3, 4).reshape(h, w, -1).permute(2, 0, 1)[None].float()

        def forward(self, pixel_values):
            v = self.visual
            x = v.patch_embed(pixel_values.to(v.patch_embed.proj.weight.dtype))
            x = x + self.absolute.to(x.dtype)
            deep, maps = [], []
            for i, block in enumerate(v.blocks):
                a = block.attn
                y = block.norm1(x)
                qkv = a.qkv(y).reshape(y.shape[0], 3, a.num_heads, -1).permute(1, 2, 0, 3)
                q, k, value = qkv.unbind(0)
                cos, sin = self.cos[None], self.sin[None]
                # Official vision RoPE computes in FP32, then casts back.
                q = (q.float() * cos + rotate(q.float()) * sin).to(value.dtype)
                k = (k.float() * cos + rotate(k.float()) * sin).to(value.dtype)
                weights = ((q @ k.transpose(-1, -2)) * a.scaling).float().softmax(-1).to(value.dtype)
                y = (weights @ value).transpose(0, 1).reshape(x.shape[0], -1)
                x = x + a.proj(y)
                x = x + block.mlp(block.norm2(x))
                if i in v.deepstack_visual_indexes:
                    maps.append(self.feature_map(x))
                    deep.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(i)](x).float())
            maps.append(self.feature_map(x))
            return (v.merger(x).float(), *deep, *maps)

    class Language(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.language = model.model.language_model
            self.lm_head = model.lm_head
            self.config = model.config.text_config
            rope = self.language.rotary_emb
            self.register_buffer("inv_freq", rope.inv_freq.detach().float())
            section = rope.mrope_section
            axes = torch.zeros(self.inv_freq.numel(), dtype=torch.long, device=self.inv_freq.device)
            axes[1:section[1] * 3:3] = 1
            axes[2:section[2] * 3:3] = 2
            self.register_buffer("rope_axes", axes)

        def forward(self, input_ids, position_ids, image_features, deepstack, visual_mask, past):
            lm = self.language
            dtype = lm.embed_tokens.weight.dtype
            x = lm.embed_tokens(input_ids)
            mask = visual_mask.to(dtype)
            x = x * (1 - mask) + image_features.to(dtype)
            # Each frequency selects T, H or W according to interleaved MRoPE.
            p = position_ids.permute(1, 2, 0).float()
            freqs = p.index_select(-1, self.rope_axes) * self.inv_freq
            rotary = torch.cat((freqs, freqs), dim=-1)
            cos, sin = rotary.cos().to(dtype)[:, None], rotary.sin().to(dtype)[:, None]
            length, previous = input_ids.shape[1], past.shape[3]
            qpos = torch.arange(length, device=x.device) + previous
            kpos = torch.arange(previous + length, device=x.device)
            # Cache slot 0 is a permanent masked sentinel. No zero-volume TRT inputs.
            allowed = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > 0)
            bias = torch.where(allowed, 0., -10000.).to(dtype)[None, None]
            present = []
            for i, layer in enumerate(lm.layers):
                a = layer.self_attn
                y = layer.input_layernorm(x)
                q = a.q_norm(a.q_proj(y).reshape(1, length, -1, a.head_dim)).transpose(1, 2)
                k = a.k_norm(a.k_proj(y).reshape(1, length, -1, a.head_dim)).transpose(1, 2)
                value = a.v_proj(y).reshape(1, length, -1, a.head_dim).transpose(1, 2)
                q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
                k = torch.cat((past[i, 0][None].to(dtype), k), dim=2)
                value = torch.cat((past[i, 1][None].to(dtype), value), dim=2)
                present.append(torch.stack((k[0], value[0]), dim=0))
                k_attention = k.repeat_interleave(a.num_key_value_groups, dim=1)
                v_attention = value.repeat_interleave(a.num_key_value_groups, dim=1)
                weights = ((q @ k_attention.transpose(-1, -2)) * a.scaling + bias).float().softmax(-1).to(dtype)
                y = (weights @ v_attention).transpose(1, 2).reshape(1, length, -1)
                x = x + a.o_proj(y)
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                if i < 3:
                    x = x + deepstack[i].to(dtype)
            # The official LastHiddenCapture hooks the final layer BEFORE lm.norm.
            return self.lm_head(lm.norm(x[:, -1])).float(), x[:, -1].float(), torch.stack(present)

    class Pixel(torch.nn.Module):
        def __init__(self, decoder, gh, gw):
            super().__init__()
            self.decoder, self.grid = decoder.float(), (gh, gw)

        def cascade(self, image, pixel_feature, queries, coarse):
            # Exact official FullCanvasAlphaCascadeHead with a symbolic query
            # count instead of int(queries.shape[0]), so one engine handles N objects.
            h = self.decoder.highres_refine
            n = queries.shape[0]
            s2 = tuple(coarse.shape[-2:])
            s1 = tuple(image.shape[-2:])
            s4 = (s2[0] // 2, s2[1] // 2)
            semantic2 = h.semantic(pixel_feature)
            semantic4 = h._resize(semantic2, s4)
            rgb4 = h.rgb_s4(h._resize(image, s4))
            coarse4 = h._resize(coarse[:, None], s4)[:, 0]
            f4 = h.fuse_s4(torch.cat((semantic4.expand(n, -1, -1, -1), rgb4.expand(n, -1, -1, -1),
                                     h.coarse_s4(coarse4[:, None].to(semantic4.dtype))), dim=1))
            f4 = h.blocks_s4(h._condition(f4, queries, h.query_s4))
            logits4 = coarse4.float() + h.output_s4(f4)[:, 0].float()
            previous2 = h._resize(logits4[:, None], s2)
            rgb2 = h.rgb_s2(h._resize(image, s2))
            f2 = h.fuse_s2(torch.cat((h.up_s4_to_s2(f4), semantic2.expand(n, -1, -1, -1),
                                     rgb2.expand(n, -1, -1, -1), h.coarse_s2(coarse[:, None].to(semantic2.dtype)),
                                     h.previous_s2(previous2.sigmoid().to(semantic2.dtype))), dim=1))
            f2 = h.blocks_s2(h._condition(f2, queries, h.query_s2))
            logits2 = coarse.float() + h.output_s2(f2)[:, 0].float()
            semantic1 = h._resize(semantic2, s1)
            rgb1 = h.rgb_s1(image)
            previous1 = h._resize(logits2[:, None], s1)
            f1 = h.fuse_s1(torch.cat((h.up_s2_to_s1(f2), semantic1.expand(n, -1, -1, -1),
                                     rgb1.expand(n, -1, -1, -1), h.previous_s1(previous1.sigmoid().to(semantic1.dtype))), dim=1))
            f1 = h.blocks_s1(h._condition(f1, queries, h.query_s1))
            return previous1[:, 0].float() + h.output_s1(f1)[:, 0].float()

        def forward(self, image, visual_0, visual_1, visual_2, visual_3, t_seg, boxes):
            d = self.decoder
            encoded, memory, shapes, starts = d.visual_encoder([visual_0, visual_1, visual_2, visual_3])
            content, position = d.query_builder(t_seg, boxes)
            queries = d.transformer(content[None], position[None], boxes, memory, shapes, starts)[0]
            pixels = d.pixel_fusion(image, encoded)
            coarse = d.mask_predictor(queries, pixels)
            return self.cascade(image, pixels, queries, coarse)

    return Vision, Language, Pixel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="cache/wevisone/2B/checkpoint")
    p.add_argument("--source", type=Path, help="Official repo, e.g. cache/wevisone/source/<revision>")
    p.add_argument("--output", type=Path, default=Path("cache/wevisone/2B/onnx"))
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--max-prefill", type=int, default=2048)
    p.add_argument("--max-objects", type=int, default=16)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--device", default="cpu", help="CPU export avoids loading the full 2B/4B model on the GPU")
    a = p.parse_args()
    if (not 0 < a.width <= 4096 or not 0 < a.height <= 4096
            or not 0 < a.max_prefill < a.max_context <= 131072 or not 0 < a.max_objects <= 1024):
        p.error("Invalid image/context/object capacities")
    source_path(a.source)
    import torch
    import transformers
    import onnx
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from wevisone_vl.config import Qwen3VLSegConfig
    from wevisone_vl.decoder import build_mask_decoder_from_qwen
    from huggingface_hub import hf_hub_download
    if transformers.__version__ != "5.12.1":
        raise RuntimeError("Use transformers==5.12.1: exported Qwen internals are version-specific")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "bundle.json").unlink(missing_ok=True)
    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(a.checkpoint, dtype=dtype, attn_implementation="eager").to(a.device).eval()
    processor = AutoProcessor.from_pretrained(a.checkpoint)
    config, vc, tc = model.config, model.config.vision_config, model.config.text_config
    if config.model_type != "qwen3_vl" or len(vc.deepstack_visual_indexes) != 3:
        raise ValueError("Expected released dense Qwen3-VL with three DeepStack levels")
    if tc.rope_parameters.get("rope_type") != "default" or not tc.rope_parameters.get("mrope_interleaved", True):
        raise ValueError("Unsupported Qwen MRoPE configuration")
    unit = vc.patch_size * vc.spatial_merge_size
    ip = processor.image_processor
    if (not ip.do_normalize or not ip.do_rescale or abs(ip.rescale_factor - 1 / 255) > 1e-10
            or int(ip.resample) != 3 or ip.patch_size != vc.patch_size or ip.merge_size != vc.spatial_merge_size
            or ip.temporal_patch_size != vc.temporal_patch_size):
        raise ValueError("Image processor differs from the native bicubic RGB/patch implementation")
    if a.width % unit or a.height % unit:
        p.error(f"Image dimensions must be multiples of {unit}")
    gh, gw = a.height // vc.patch_size, a.width // vc.patch_size
    if gh * gw // vc.spatial_merge_size ** 2 + 64 >= a.max_prefill:
        p.error("Vision grid leaves insufficient text space in --max-prefill")
    checkpoint = Path(a.checkpoint)
    sidecar = checkpoint / "mask_decoder.pt" if checkpoint.is_dir() else Path(hf_hub_download(a.checkpoint, "mask_decoder.pt"))
    data = torch.load(sidecar, map_location="cpu", weights_only=True)
    cfg = Qwen3VLSegConfig.from_dict(data.get("cfg"))
    if tuple(cfg.vision_feature_layers) != tuple(vc.deepstack_visual_indexes) + (vc.depth - 1,):
        raise ValueError("Pixel feature layers do not match the visual encoder")
    decoder = build_mask_decoder_from_qwen(model, cfg)
    decoder.load_state_dict(data["state_dict"], strict=True)
    decoder = decoder.to(device=a.device, dtype=torch.float32).eval()
    tokenizer = processor.tokenizer
    tokenizer_file = export_tokenizer(tokenizer, a.output)
    processor.save_pretrained(a.output / "processor")
    # Validate the exact chat template consumed by native C++; do not silently
    # accept future template changes in new upstream checkpoints.
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "probe"}]}]
    expected_chat = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>probe<|im_end|>\n<|im_start|>assistant\n"
    if processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) != expected_chat:
        raise ValueError("Checkpoint chat template differs from the native implementation")
    Vision, Language, Pixel = wrappers(torch)
    device = torch.device(a.device)
    z = lambda *shape: torch.zeros(shape, device=device, dtype=torch.float32)
    graphs = {}

    def export(name, module, inputs, input_names, output_names, dynamic=None):
        path = a.output / (name + ".onnx")
        kwargs = {"dynamic_axes": dynamic or {}, "dynamo": False}
        if name == "language":
            # Use the modern exporter for >2GB LLMs: external_data is an
            # explicit supported contract here, rather than a legacy fallback.
            sequence = torch.export.Dim("sequence", min=1, max=a.max_prefill)
            previous = torch.export.Dim("past_sequence", min=1, max=a.max_context)
            kwargs = {"dynamo": True, "dynamic_shapes": {
                "input_ids": {1: sequence}, "position_ids": {2: sequence},
                "image_features": {1: sequence}, "deepstack": {2: sequence},
                "visual_mask": {1: sequence}, "past": {3: previous}}}
        with torch.no_grad():
            torch.onnx.export(module.eval(), inputs, str(path), input_names=input_names, output_names=output_names,
                              opset_version=18, external_data=True, **kwargs)
        graph = onnx.load(path, load_external_data=False)
        # A 2B/4B language graph exceeds the protobuf 2GB limit. Hash and ship
        # every external weight file together with the ONNX graph.
        external = set()
        for tensor in graph.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                info = {entry.key: entry.value for entry in tensor.external_data}
                external.add(info["location"])
        files = {path.name: sha256(path)}
        for relative in sorted(external):
            asset = (a.output / relative).resolve()
            if not asset.is_relative_to(a.output.resolve()):
                raise ValueError("External ONNX weights escape export directory")
            files[relative] = sha256(asset)
        graphs[name] = {"onnx": path.name, "files": files}
        print("Exported", path, flush=True)
        del graph
        gc.collect()

    torch.backends.mha.set_fastpath_enabled(False)
    export("vision", Vision(model.model.visual, gh, gw, vc.spatial_merge_size),
           (z(gh * gw, 3 * vc.temporal_patch_size * vc.patch_size ** 2),), ["pixel_values"],
           ["image_features"] + [f"deepstack_{i}" for i in range(3)] + [f"visual_{i}" for i in range(4)])
    seq, previous = 8, 2
    cache = torch.zeros((tc.num_hidden_layers, 2, tc.num_key_value_heads, previous, tc.head_dim), device=device, dtype=dtype)
    ids = torch.zeros((1, seq), device=device, dtype=torch.long)
    pos = torch.arange(seq, device=device).reshape(1, 1, seq).expand(3, 1, seq)
    dynamic = {"input_ids": {1: "sequence"}, "position_ids": {2: "sequence"},
               "image_features": {1: "sequence"}, "deepstack": {2: "sequence"},
               "visual_mask": {1: "sequence"}, "past": {3: "past_sequence"}, "present": {3: "total_sequence"}}
    export("language", Language(model), (ids, pos, z(1, seq, tc.hidden_size), z(3, 1, seq, tc.hidden_size),
                                        z(1, seq, 1), cache),
           ["input_ids", "position_ids", "image_features", "deepstack", "visual_mask", "past"],
           ["logits", "last_hidden", "present"], dynamic)
    del cache, model
    gc.collect()
    # Trace with two queries and keep N symbolic throughout the cascade.
    n = min(2, a.max_objects)
    pixel_inputs = (z(1, 3, cfg.cnn_image_size, cfg.cnn_image_size),
                    *(z(1, vc.hidden_size, gh, gw) for _ in range(4)),
                    z(n, tc.hidden_size), torch.tensor([[.1, .1, .9, .9]], device=device).expand(n, 4).contiguous())
    export("pixel", Pixel(decoder, gh, gw), pixel_inputs,
           ["image", "visual_0", "visual_1", "visual_2", "visual_3", "t_seg", "boxes"], ["mask_logits"],
           {"t_seg": {0: "objects"}, "boxes": {0: "objects"}, "mask_logits": {0: "objects"}})
    manifest = {"format": "cvinfer-wevisone-v1", "checkpoint": a.checkpoint,
                "checkpoint_revision": getattr(config, "_commit_hash", None), "tasks": list(TASKS),
                "transformers": transformers.__version__, "torch": torch.__version__, "graphs": graphs,
                "tokenizer_sha256": sha256(tokenizer_file), "mask_decoder_sha256": sha256(sidecar),
                "vision": {"width": a.width, "height": a.height, "patch_size": vc.patch_size,
                           "temporal_patch_size": vc.temporal_patch_size, "spatial_merge_size": vc.spatial_merge_size,
                           "hidden_size": vc.hidden_size, "mean": processor.image_processor.image_mean,
                           "std": processor.image_processor.image_std},
                "language": {"hidden_size": tc.hidden_size, "layers": tc.num_hidden_layers,
                             "kv_heads": tc.num_key_value_heads, "head_dim": tc.head_dim, "vocab_size": tc.vocab_size,
                             "max_context": a.max_context, "max_prefill": a.max_prefill, "cache_dtype": a.dtype,
                             "image_token_id": config.image_token_id, "obj_token_id": tokenizer.convert_tokens_to_ids("<OBJ>")},
                "pixel": {"image_size": cfg.cnn_image_size, "max_objects": a.max_objects, "precision": "fp32"},
                "status": "EXPORTED_REAL_VALIDATION_REQUIRED"}
    (a.output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
