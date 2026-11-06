"""Check that everything the CV Parser needs is installed and downloaded, then run a short end-to-end test.

    python check.py              # check only; nothing is downloaded
    python check.py --download   # also download any missing model (the open LLM is about 8 GB)
    python check.py --skip-llm   # skip the open model (for machines without a GPU or Apple silicon)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" · {detail}" if detail else ""), flush=True)


def size_of(path: str | Path) -> str:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return f"{total / 1e9:.2f} GB" if total > 1e9 else f"{total / 1e6:.0f} MB"


def model_files(repo: str, download: bool, patterns=None) -> str | None:
    """Local folder holding the model, downloading it first if asked. None if it's missing."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
    except Exception:
        if not download:
            return None
    print(f"         downloading {repo} …", flush=True)
    return snapshot_download(repo, allow_patterns=patterns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download missing models")
    ap.add_argument("--skip-llm", action="store_true", help="don't check the open LLM")
    args = ap.parse_args()

    print(f"CV Parser check · Python {sys.version.split()[0]} · {sys.executable}\n", flush=True)

    # 1. Python and packages
    if sys.version_info < (3, 10):
        report(BAD, "Python 3.10 or newer", sys.version.split()[0])
    else:
        report(OK, "Python version", sys.version.split()[0])
    packages = {"gradio": "gradio", "torch": "torch", "transformers": "transformers",
                "sentence-transformers": "sentence_transformers", "pypdfium2": "pypdfium2",
                "python-docx": "docx", "markdown-it-py": "markdown_it", "openai": "openai", "anthropic": "anthropic"}
    missing = []
    for pip_name, module in packages.items():
        try:
            mod = importlib.import_module(module)
            report(OK, f"package {pip_name}", getattr(mod, "__version__", ""))
        except Exception as exc:
            missing.append(pip_name)
            report(BAD, f"package {pip_name}", f"{type(exc).__name__}: {exc}")
    if missing:
        print("\nInstall the missing packages with:  pip install -r requirements.txt\n")
        return summary()

    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    report(OK if device != "cpu" else WARN, "compute device", device +
           ("" if device != "cpu" else " (the open model will be very slow; use your API key or search-only mode)"))

    # 2. Parsing and keyword search on the sample CVs
    from src.index import EMBED_MODEL, CVIndex, Embedder, set_embedder
    from src.parse import load_cv

    samples = sorted((ROOT / "samples").glob("*.txt"))
    try:
        cvs = [load_cv(f) for f in samples]
        good = all(cv.name and cv.roles and cv.years > 0 for cv in cvs)
        report(OK if good and len(cvs) == 5 else BAD, "parse the sample CVs",
               ", ".join(f"{cv.name} {cv.years}y" for cv in cvs))
    except Exception as exc:
        report(BAD, "parse the sample CVs", f"{type(exc).__name__}: {exc}")
        return summary()

    # 3. Embedding model
    path = None
    try:
        path = model_files(EMBED_MODEL, args.download)
    except Exception as exc:
        report(BAD, f"embedding model {EMBED_MODEL}", f"download failed: {exc}")
    if path is None:
        report(BAD, f"embedding model {EMBED_MODEL}", "not downloaded yet; run again with --download")
    else:
        report(OK, f"embedding model {EMBED_MODEL}", f"downloaded ({size_of(path)})")
        try:
            t = time.time()
            set_embedder(Embedder())
            report(OK, "embedding model loads", f"{time.time() - t:.1f} s")
        except Exception as exc:
            report(BAD, "embedding model loads", f"{type(exc).__name__}: {exc}")

    index = CVIndex()
    for cv in cvs:
        index.add(cv)
    expected = {
        "Who has deployed models on edge devices?": "Aisha Rahman",
        "Who has built a RAG system with a vector database?": "Kenji Watanabe",
        "Who can build real-time streaming data pipelines?": "Daniel Okafor",
        "Who speaks Spanish?": "Sofia Martínez",
    }
    right = 0
    for q, who in expected.items():
        top = index.cvs[index.rank_candidates(q)[0][0]["cand"]].name
        right += top == who
        if top != who:
            report(WARN, f"search: {q}", f"top result {top}, expected {who}")
    mode = "hybrid (embeddings + keywords)" if index.semantic else "keywords only"
    report(OK if right == len(expected) else WARN, f"search ranks the right candidate ({mode})", f"{right}/{len(expected)}")
    match = index.match_job((ROOT / "app.py").read_text(encoding="utf-8").split('JOB = """')[1].split('"""')[0])
    top_match = index.cvs[match["order"][0]].name
    report(OK if top_match == "Aisha Rahman" else WARN, "job match", f"top candidate {top_match}")

    # 4. Open LLM
    if args.skip_llm:
        report(WARN, "open model", "skipped (--skip-llm)")
        return summary()
    from src import llm

    patterns = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "tokenizer*", "merges.txt", "vocab*"]
    try:
        path = model_files(llm.OPEN_MODEL, args.download, patterns)
    except Exception as exc:
        report(BAD, f"open model {llm.OPEN_MODEL}", f"download failed: {exc}")
        return summary()
    if path is None:
        report(BAD, f"open model {llm.OPEN_MODEL}", "not downloaded yet; run again with --download (about 8 GB)")
        return summary()
    report(OK, f"open model {llm.OPEN_MODEL}", f"downloaded ({size_of(path)})")
    t = time.time()
    llm.load_open_model()
    if llm.open_model_status().startswith("not loaded"):
        report(BAD, "open model loads", llm.open_model_status())
        return summary()
    report(OK, "open model loads", f"{llm.open_model_status()}, {time.time() - t:.0f} s")

    question = "Who has deployed models on edge devices?"
    prompt, hits = llm.build_context(index, question)
    t = time.time()
    first, text = None, ""
    for piece in llm.stream_open(llm.SYSTEM, f"{prompt}\n\nQuestion: {question}"):
        if first is None:
            first = time.time() - t
        text += piece
    total = time.time() - t
    words = len(text.split())
    report(OK if text.strip() else BAD, "open model answers",
           f"first words after {first or 0:.1f} s, {words} words in {total:.0f} s")
    import re

    cited = re.findall(r"\[(\d{1,2})", text)
    report(OK if "Aisha" in text else WARN, "answer names the right candidate", "Aisha Rahman" if "Aisha" in text else "Aisha not mentioned")
    report(OK if cited else WARN, "answer cites its sources", f"{len(cited)} citations" if cited else "no [n] citations found")
    print("\n--- answer ---\n" + text.strip() + "\n--------------")
    return summary()


def summary() -> int:
    fails = sum(1 for s, *_ in results if s == BAD)
    warns = sum(1 for s, *_ in results if s == WARN)
    print(f"\n{len(results)} checks · {fails} failed · {warns} warnings")
    if fails:
        print("Fix the FAIL lines above (often: run with --download, or pip install -r requirements.txt).")
    else:
        print("Everything needed is in place. Start the app with ./run.sh")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
