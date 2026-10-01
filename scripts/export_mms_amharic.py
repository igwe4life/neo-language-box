"""Export facebook/mms-tts-amh (a HuggingFace `transformers` VITS checkpoint)
to ONNX, in the app's existing "mms" VitsSpeaker signature.

Why this one needs its own script rather than reusing the willwade
pre-converted community voices (see MMS_VOICES in scripts/fetch_models.py):
that repository has no Amharic folder, so it is converted here, once, at
build time, in the separate .venv-export environment (torch + transformers
never ship in the app itself - only onnxruntime does).

Why Amharic ALSO needs a runtime change the other MMS voices don't: this
checkpoint's vocabulary is 28 plain Latin letters plus space/apostrophe, not
Ge'ez script - facebook/mms-tts-amh was trained on romanised text. Amharic
translations come out of NLLB in native Ge'ez script, so sermon/tts.py
romanises with the `uroman` package before handing text to this voice
specifically (VitsSpeaker checks for a `romanize.flag` file next to the
model, written by this script).

noise_scale / length_scale / noise_scale_w are NOT real inputs of HF's
VitsModel (it reads them off `model.config` as plain floats, not forward()
arguments), so they are fixed here at export time rather than made dynamic -
the app's generic length_scale slider has no effect on this one voice. They
are still declared as unused ONNX graph inputs purely so this model matches
the same 5-input call signature VitsSpeaker already uses for every other
"mms"-style voice, and needs no runtime code change to support.

Usage (from the export venv):
    .venv-export/Scripts/python.exe scripts/export_mms_amharic.py \
        --out models/tts/am
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="facebook/mms-tts-amh")
    ap.add_argument("--out", required=True, help="output directory, e.g. models/tts/am")
    args = ap.parse_args()

    import torch
    from transformers import VitsModel, VitsTokenizer

    print(f"loading {args.repo}...")
    model = VitsModel.from_pretrained(args.repo)
    tokenizer = VitsTokenizer.from_pretrained(args.repo)
    model.eval()
    model.config.noise_scale = 0.667
    model.config.noise_scale_duration_predictor = 0.8
    model.config.length_scale = 1.0

    class Wrapper(torch.nn.Module):
        """Matches VitsSpeaker's "mms" signature: x/x_length/noise_scale/
        length_scale/noise_scale_w -> y. The three scale inputs are accepted
        for signature compatibility but not used - see module docstring."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, x_length, noise_scale, length_scale, noise_scale_w):
            # Built FROM x_length (not torch.ones_like(x)) so the tracer
            # keeps it as a real graph input instead of silently dropping it
            # as unused - onnxruntime then rejects the extra feed at
            # inference. Numerically identical to all-ones for the
            # single-utterance, no-padding case this app always calls with.
            positions = torch.arange(x.shape[1]).unsqueeze(0)
            attention_mask = (positions < x_length.unsqueeze(1)).to(x.dtype)
            waveform = self.inner(input_ids=x, attention_mask=attention_mask).waveform
            # Same reasoning for the three scale inputs: baked as fixed
            # config values above, but wired in as a zero-effect addition so
            # they too survive tracing as declared (if inert) graph inputs.
            zero = (noise_scale.sum() + length_scale.sum() + noise_scale_w.sum()) * 0.0
            return waveform + zero

    wrapper = Wrapper(model)

    vocab = tokenizer.get_vocab()
    example_ids = tokenizer("selam ewedeshalehu", return_tensors="pt")["input_ids"]
    x_length = torch.tensor([example_ids.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([0.667], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w = torch.tensor([0.8], dtype=torch.float32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    print("tracing and exporting to ONNX (this takes a minute)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example_ids, x_length, noise_scale, length_scale, noise_scale_w),
            str(onnx_path),
            dynamo=False,
            input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch", 1: "sequence"},
                "y": {0: "batch", 1: "time"},
            },
            opset_version=15,
        )
    print(f"wrote {onnx_path}  ({onnx_path.stat().st_size / 1e6:.0f} MB)")

    # tokens.txt in the same "char id" format VitsSpeaker._read_tokens
    # already parses for every other mms-style voice.
    tokens_path = out_dir / "tokens.txt"
    with tokens_path.open("w", encoding="utf-8") as f:
        for char, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            if char == tokenizer.unk_token:
                continue
            f.write(f"{char} {idx}\n")
    print(f"wrote {tokens_path}")

    # Tells VitsSpeaker to uroman-romanise text before encoding - this
    # checkpoint's vocabulary is Latin, not Ge'ez. See sermon/tts.py.
    (out_dir / "romanize.flag").write_text(
        "facebook/mms-tts-amh was trained on uroman-romanised text.\n", encoding="utf-8")
    print(f"wrote {out_dir / 'romanize.flag'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
