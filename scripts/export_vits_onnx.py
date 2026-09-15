"""Export a Coqui VITS checkpoint to ONNX so the app can run it without torch.

Why this exists: the app ships onnxruntime and deliberately does NOT ship
torch - adding it would roughly triple the installer. Some voices we want are
published only as Coqui `.pth` checkpoints, so they get converted ONCE here,
at build time, in the separate .venv-export environment.

The Igbo case that motivated it: Meta's MMS-TTS has no Igbo, and the community
MMS-style Igbo models all share a 43-character vocabulary that is missing
i-dot-below and u-dot-below - two of the commonest vowels in the language.
They render "ụmụnna" as "mnna". The OpenBible VITS checkpoint has the full
orthography, so it is worth the conversion step.

Usage (from the export venv):
    .venv-export/Scripts/python.exe scripts/export_vits_onnx.py \
        --config models/mms/ibo-openbible/config.json \
        --checkpoint models/mms/ibo-openbible/model_last.pth \
        --out models/mms/ibo-openbible/model.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _export_legacy(model, out: Path) -> None:
    """Coqui's own export_onnx() goes through torch.export (dynamo), which
    fails on VITS: its duration predictor produces data-dependent shapes and
    the tracer raises a data-dependent error before it emits a graph.

    The legacy TorchScript tracing exporter handles that fine, so replicate
    Coqui's wrapper here and call torch.onnx.export with dynamo=False.
    """
    import torch

    model.disc = None
    model.eval()

    def onnx_inference(text, text_lengths, scales):
        model.noise_scale = scales[0]
        model.length_scale = scales[1]
        model.noise_scale_dp = scales[2]
        return model.inference(
            text,
            aux_input={"x_lengths": text_lengths, "d_vectors": None,
                       "speaker_ids": None, "language_ids": None,
                       "durations": None},
        )["model_outputs"]

    model.forward = onnx_inference

    seq = torch.randint(low=0, high=2, size=(1, 100), dtype=torch.long)
    lengths = torch.LongTensor([seq.size(1)])
    scales = torch.FloatTensor([model.inference_noise_scale,
                                model.length_scale,
                                model.inference_noise_scale_dp])
    torch.onnx.export(
        model,
        (seq, lengths, scales),
        str(out),
        dynamo=False,
        input_names=["input", "input_lengths", "scales"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time1", 2: "time2"},
        },
        opset_version=15,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits

    cfg = VitsConfig()
    cfg.load_json(args.config)
    model = Vits.init_from_config(cfg)
    model.load_checkpoint(cfg, args.checkpoint, eval=True)
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _export_legacy(model, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    # The vocabulary travels with the model: the app tokenises by character and
    # must use exactly the inventory the checkpoint was trained on.
    chars = cfg.characters
    meta = {
        "characters": chars["characters"],
        "punctuations": chars["punctuations"],
        "pad": chars.get("pad"),
        "bos": chars.get("bos"),
        "eos": chars.get("eos"),
        "blank": chars.get("blank"),
        "sample_rate": cfg.audio["sample_rate"],
        "add_blank": bool(getattr(cfg, "add_blank", True)),
    }
    side = out.with_name("tokens.json")
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
