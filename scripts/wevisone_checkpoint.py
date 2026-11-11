"""WeVisOne checkpoint/tokenizer utilities shared by export and reference inference."""
import hashlib
import json
from pathlib import Path
import sys

TASKS = ("detection", "referring", "point", "count_box", "count_point", "ocr", "ocr_grounding",
         "segmentation", "matting", "visual_prompt", "freeform")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(source):
    if source:
        source = Path(source).resolve()
        if not (source / "wevisone_vl/engine.py").is_file():
            raise ValueError("--source must contain the official wevisone_vl package")
        sys.path.insert(0, str(source))


def export_tokenizer(tokenizer, output):
    """Export exact byte vocabulary, regex split and added tokens for native PCRE2+BPE."""
    raw = json.loads(tokenizer.backend_tokenizer.to_str())
    normalizer = raw.get("normalizer")
    if normalizer not in (None, {"type": "NFC"}) or raw["model"]["type"] != "BPE":
        raise ValueError("Only released Qwen byte BPE with optional NFC normalization is supported")
    model = raw["model"]
    if model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise ValueError("Unsupported BPE options")
    if model.get("byte_fallback") or model.get("ignore_merges"):
        raise ValueError("Unsupported BPE byte fallback/ignore_merges")
    pre = raw["pre_tokenizer"]
    if pre.get("type") != "Sequence" or len(pre["pretokenizers"]) != 2:
        raise ValueError("Expected Qwen Split + ByteLevel pretokenizer")
    split, bytelevel = pre["pretokenizers"]
    if (split.get("type") != "Split" or split.get("behavior") != "Isolated" or split.get("invert")
            or bytelevel.get("type") != "ByteLevel" or bytelevel.get("add_prefix_space") or bytelevel.get("use_regex")):
        raise ValueError("Unsupported Qwen pretokenizer options")
    pattern = split["pattern"]["Regex"]
    # GPT-2 reversible byte-to-Unicode alphabet, not UTF-8 decoding of vocab text.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = dict(zip(visible, visible))
    extra = 256
    for byte in range(256):
        if byte not in alphabet:
            alphabet[byte] = extra
            extra += 1
    reverse = {chr(code): byte for byte, code in alphabet.items()}

    def byte_hex(piece):
        return bytes(reverse[c] for c in piece).hex()

    added = raw.get("added_tokens", [])
    special = {}
    for item in added:
        if item.get("lstrip") or item.get("rstrip") or item.get("single_word"):
            raise ValueError("Added token whitespace/word constraints are unsupported")
        special[item["content"]] = item["id"]
    vocabulary = {byte_hex(word): index for word, index in model["vocab"].items() if word not in special}
    merges = []
    for pair in model["merges"]:
        left, right = pair.split(" ") if isinstance(pair, str) else pair
        merges.append([byte_hex(left), byte_hex(right)])
    result = {"format": "cvinfer-qwen-byte-bpe-v1", "pattern": pattern, "normalizer": normalizer,
              "vocab": vocabulary, "merges": merges, "special": special}
    path = Path(output) / "tokenizer_native.json"
    path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path
