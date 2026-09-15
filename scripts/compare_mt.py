"""Side-by-side MT comparison: OPUS-MT bilingual vs NLLB-200 (600M / 1.3B).

Built to answer one question that benchmarks cannot: which of these is good
enough to put in front of an Igbo- or Yoruba-speaking congregation. BLEU on a
religious domain will not settle it and neither will my reading, so this
produces a page a native speaker can judge.

Two deliberate choices:

* Engine names are HIDDEN in the review output. Seeing "NLLB-1.3B" above a
  sentence primes you to prefer it. The key is written to a separate file.
  Labels are assigned ONCE PER LANGUAGE and stay fixed down the sheet. An
  earlier version reshuffled them per sentence, which kept the blinding but
  made the obvious answer - "A reads best" - impossible to interpret, since
  A was a different engine in almost every sentence.
* The NLLB plumbing is validated on FRENCH first. NLLB needs its language
  tokens assembled by hand, and a subtly wrong prompt still produces
  confident-looking output - which would be undetectable in a language the
  author cannot read, and would silently corrupt the whole comparison.

Usage:
    python scripts/compare_mt.py                  # review sheet + key
    python scripts/compare_mt.py --langs ig,yo
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# NLLB uses its own language codes; ours are two-letter.
NLLB_CODE = {
    "ig": "ibo_Latn", "yo": "yor_Latn", "ha": "hau_Latn",
    "zu": "zul_Latn", "sw": "swh_Latn",
    "fr": "fra_Latn", "es": "spa_Latn",     # validation only
}

# Real sermon material: the register the system will actually meet, including
# the scripture references and set phrases that trip low-resource models.
SENTENCES = [
    "Brothers and sisters, please open your Bibles with me this morning.",
    "We are looking at the book of Romans, chapter eight.",
    "Nothing in all creation will be able to separate us from the love of God.",
    "The Lord is my shepherd; I shall not want.",
    "Let us pray together before we continue.",
    "Faith comes by hearing, and hearing by the word of God.",
    "He was crucified, he was buried, and on the third day he rose again.",
    "I want you to turn to the person next to you and say, God is faithful.",
    "Give thanks to the Lord, for he is good; his mercy endures forever.",
    "The kingdom of heaven is like a mustard seed that a man planted in his field.",
    "Repent and be baptised, every one of you, in the name of Jesus Christ.",
    "May the grace of our Lord Jesus Christ be with you all.",
]


# ----------------------------------------------------------------- engines

class OpusEngine:
    name = "opus-mt"

    def __init__(self, lang):
        import ctranslate2, sentencepiece as spm
        d = MODELS / "mt" / f"opus-mt-en-{lang}"
        if not d.exists():
            raise FileNotFoundError(d)
        self.t = ctranslate2.Translator(str(d), device="cpu", compute_type="int8")
        self.sp = spm.SentencePieceProcessor(str(d / "source.spm"))
        self.tp = spm.SentencePieceProcessor(str(d / "target.spm"))

    def translate(self, text, lang):
        tok = self.sp.encode(text, out_type=str)
        res = self.t.translate_batch([tok], beam_size=2, max_decoding_length=320)
        return self.tp.decode(res[0].hypotheses[0])


class NllbEngine:
    """NLLB via CTranslate2.

    The source is the sentence tokens followed by </s> and the SOURCE language
    token; the target is prefixed with the TARGET language token. Getting this
    wrong yields fluent-looking nonsense rather than an error, which is why
    validate_french() exists.
    """

    def __init__(self, model_dir, spm_path, label):
        import ctranslate2, sentencepiece as spm
        self.name = label
        self.t = ctranslate2.Translator(str(model_dir), device="cpu",
                                        compute_type="int8")
        self.sp = spm.SentencePieceProcessor(str(spm_path))

    def translate(self, text, lang):
        tgt = NLLB_CODE[lang]
        tok = self.sp.encode(text, out_type=str) + ["</s>", "eng_Latn"]
        res = self.t.translate_batch([tok], target_prefix=[[tgt]],
                                     beam_size=2, max_decoding_length=320)
        out = res[0].hypotheses[0]
        if out and out[0] == tgt:
            out = out[1:]
        return self.sp.decode(out)


def build_engines(langs):
    """Return {label: engine}. A missing model is reported, not fatal."""
    engines = {}
    spm_path = MODELS / "nllb-600m" / "sentencepiece.bpe.model"
    for label, directory in (("nllb-600m", MODELS / "nllb-600m"),
                             ("nllb-1.3b", MODELS / "nllb-1.3b")):
        if not (directory / "model.bin").exists():
            print(f"  [skip] {label}: not downloaded")
            continue
        if not spm_path.exists():
            print(f"  [skip] {label}: tokenizer missing ({spm_path})")
            continue
        t0 = time.perf_counter()
        engines[label] = NllbEngine(directory, spm_path, label)
        print(f"  loaded {label} in {time.perf_counter() - t0:.1f}s")
    return engines


# -------------------------------------------------------------- validation

def validate_french(engines) -> bool:
    """Prove the NLLB prompt format is right, in a language we can check.

    If this fails, every African-language result below is worthless, so the
    comparison refuses to continue.
    """
    probe = "The Lord is my shepherd; I shall not want."
    expect = ("seigneur", "berger")          # must appear in any sane output
    ok = True
    print("\nValidating NLLB plumbing on French (a language we can read):")
    for label, eng in engines.items():
        try:
            out = eng.translate(probe, "fr")
        except Exception as exc:
            print(f"  {label}: FAILED {type(exc).__name__}: {exc}")
            ok = False
            continue
        low = out.lower()
        good = any(w in low for w in expect)
        print(f"  {label}: {out!r}  -> {'OK' if good else 'SUSPECT'}")
        if not good:
            ok = False
    return ok


# ------------------------------------------------------------------ report

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ig,yo",
                    help="comma-separated targets (default ig,yo)")
    ap.add_argument("--out", default=str(ROOT / "mt-comparison"))
    args = ap.parse_args()
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    print("Loading NLLB engines...")
    engines = build_engines(langs)
    if not engines:
        print("No NLLB engine available; nothing to compare against.")
        return 2
    if not validate_french(engines):
        print("\n!! NLLB output failed the French sanity check.")
        print("   The prompt format is wrong; African results would be")
        print("   meaningless. Fix that before trusting anything below.")
        return 3

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260915)
    key_rows, timings = [], {}

    for lang in langs:
        per_engine = dict(engines)
        try:
            per_engine["opus-mt"] = OpusEngine(lang)
        except Exception as exc:
            print(f"  [skip] opus-mt en-{lang}: {type(exc).__name__}")

        # One shuffle for the whole language: A is the same engine in every
        # sentence, so "A reads best" is a usable answer.
        labels = list(per_engine)
        rng.shuffle(labels)
        tags = {label: chr(ord("A") + i) for i, label in enumerate(labels)}
        for label, tag in tags.items():
            key_rows.append((lang, tag, label))

        lines = [f"# English -> {lang.upper()}", "",
                 "Three engines, hidden behind fixed labels: **A is the same",
                 "engine all the way down the page**, and so are B and C.",
                 "",
                 "Read through, then tell me which label reads best overall.",
                 "If one wins on some sentences and loses on others, say which",
                 "- that is useful too. The names are in comparison-key.md;",
                 "read it afterwards, not before.",
                 ""]
        for i, sentence in enumerate(SENTENCES, 1):
            lines += [f"### {i}. {sentence}", ""]
            for label in labels:
                t0 = time.perf_counter()
                try:
                    out = per_engine[label].translate(sentence, lang)
                except Exception as exc:
                    out = f"[FAILED: {type(exc).__name__}]"
                timings.setdefault(label, []).append(time.perf_counter() - t0)
                lines.append(f"- **{tags[label]}.** {out}")
            lines.append("")
        (outdir / f"comparison-{lang}.md").write_text(
            "\n".join(lines), encoding="utf-8")
        print(f"  wrote {outdir / f'comparison-{lang}.md'}")

    key = ["# Which engine was which", "",
           "Labels are fixed per language: one row each.", "",
           "| language | label | engine |",
           "|---|---|---|"]
    key += [f"| {l} | {t} | {e} |" for l, t, e in key_rows]
    key += ["", "## Speed (seconds per sentence, CPU)", "",
            "| engine | mean | max |", "|---|---|---|"]
    for label, ts in timings.items():
        key.append(f"| {label} | {sum(ts)/len(ts):.2f} | {max(ts):.2f} |")
    (outdir / "comparison-key.md").write_text("\n".join(key), encoding="utf-8")
    print(f"  wrote {outdir / 'comparison-key.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
