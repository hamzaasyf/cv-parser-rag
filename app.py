"""Gradio demo: parse CVs, ask questions across them with cited answers, and match them to a job."""

from __future__ import annotations

import os

try:  # on Hugging Face ZeroGPU, spaces must be imported before torch
    import spaces
except ImportError:
    spaces = None

import copy
import json
import re
import tempfile
from pathlib import Path

import gradio as gr
from markdown_it import MarkdownIt

from src import llm
from src.index import CVIndex, load_embedder, set_embedder
from src.parse import CATEGORIES, load_cv
from ui import (CSS, HEAD, THEME, answer_html, cv_html, fields_html, grid_html, header_html, rank_html,
                sources_html, status_html, why_html)

ON_ZERO_GPU = spaces is not None and bool(os.environ.get("SPACES_ZERO_GPU"))
SAMPLES = Path(__file__).parent / "samples"
MAX_UPLOADS = int(os.environ.get("MAX_UPLOADS", "10"))
MD = MarkdownIt("commonmark", {"html": False}).enable("table")

OPEN = f"Open model ({llm.OPEN_MODEL.split('/')[-1]}, free)"
KEY = "My API key (OpenAI or Anthropic)"
SEARCH = "Search only, no LLM"

EXAMPLES = [
    "Who has deployed models on edge devices?",
    "Who has built a RAG system with a vector database, and how well did it work?",
    "Compare the two strongest candidates for an MLOps role.",
    "Who has led or mentored a team?",
    "Who speaks Spanish?",
    "Who has 5+ years of experience and works with Kubernetes?",
]
JOB = """Machine Learning Engineer (Computer Vision)
- 5+ years building and shipping ML models in production
- Strong PyTorch and computer vision experience
- Deploying models on edge devices such as NVIDIA Jetson with TensorRT
- MLOps: experiment tracking, automated retraining and model monitoring
- Cloud experience, ideally AWS
- Has led or mentored a small engineering team"""

# ---------- startup: embeddings, the sample CVs and (on Hugging Face) the open model ----------
set_embedder(load_embedder())
BASE = CVIndex()
for f in sorted(SAMPLES.glob("*.txt")):
    BASE.add(load_cv(f))

if ON_ZERO_GPU or os.environ.get("LOAD_OPEN_MODEL") == "1":
    llm.load_open_model()  # ZeroGPU needs the model loaded and moved to the GPU here, at startup


def _open_stream(system: str, user: str):
    yield from llm.stream_open(system, user)


if spaces is not None:  # runs on a real GPU for each answer on ZeroGPU; no effect elsewhere
    _open_stream = spaces.GPU(duration=90)(_open_stream)


def default_mode() -> str:
    if ON_ZERO_GPU:
        return OPEN
    try:
        import torch

        return OPEN if (torch.cuda.is_available() or torch.backends.mps.is_available()) else SEARCH
    except Exception:
        return SEARCH


# ---------- helpers ----------
def _index(state) -> CVIndex:
    return state if state is not None else BASE


def _names(index: CVIndex) -> list[str]:
    """One label per CV; a repeated name gets (2), (3)... so every CV can be selected."""
    out, seen = [], {}
    for cv in index.cvs:
        label = f"{cv.name} · {cv.years} yrs"
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _pick(index: CVIndex, label) -> int:
    names = _names(index)
    return names.index(label) if label in names else 0


def _highlights(index: CVIndex, hits, cand: int) -> dict:
    out = {}
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        if c.cand != cand:
            continue
        key = (c.section, c.role, c.text)
        strength = max(0.0, min(1.0, (h.score - 0.5) / 0.4)) if index.semantic else max(0.0, min(1.0, h.score))
        prev = out.get(key, (0, []))
        out[key] = (max(prev[0], strength), prev[1] + [n])
    return out


def _render(text: str, n_sources: int) -> str:
    return llm.linkify_citations(MD.render(text), n_sources)


def _stream_llm(mode, key, model_name, system, user):
    """Yield the answer text as it grows."""
    if mode == KEY:
        provider = llm.provider_for(key)
        if not provider:
            raise gr.Error("Paste an OpenAI key (starts with sk-) or an Anthropic key (starts with sk-ant-).")
        gen = llm.stream_api(provider, key, model_name, system, user)
    else:
        gen = _open_stream(system, user)
    text = ""
    for piece in gen:
        text += piece
        yield text


def _meta(mode, model_name, key) -> str:
    if mode == KEY:
        provider = llm.provider_for(key) or "api"
        return f"Answer · {(model_name or '').strip() or llm.DEFAULT_MODELS.get(provider, '')} · from the excerpts below"
    if mode == OPEN:
        return f"Answer · {llm.OPEN_MODEL.split('/')[-1]} · from the excerpts below"
    return "Answer · built from the top search results, no LLM"


# ---------- tab 1: parse ----------
def show_cv(label, state):
    index = _index(state)
    i = _pick(index, label)
    cv = index.cvs[i]
    path = Path(tempfile.mkdtemp()) / (cv.name.lower().replace(" ", "_") + ".json")
    path.write_text(json.dumps(cv.as_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return cv_html(cv), fields_html(cv, CATEGORIES), str(path)


def add_cvs(files, state):
    if not files:
        raise gr.Error("Choose one or more CVs first (PDF, DOCX or TXT).")
    index = copy.deepcopy(_index(state))
    added, failed = [], []
    for f in files[:MAX_UPLOADS]:
        try:
            cv = load_cv(f)
            if not cv.raw.strip():
                raise ValueError("no text found (scanned PDF?)")
            index.add(cv)
            added.append(cv.name)
        except Exception as exc:
            failed.append(f"{Path(f).name}: {exc}")
    if failed:
        gr.Warning("Could not read " + "; ".join(failed))
    if len(files) > MAX_UPLOADS:
        gr.Warning(f"Only the first {MAX_UPLOADS} files were added.")
    names = _names(index)
    pick = names[-1] if added else names[0]
    return (index, gr.update(choices=names, value=pick), gr.update(choices=names, value=names[0]),
            status_html(len(index.cvs), len(index.chunks), index.semantic, llm.open_model_status()),
            header_html(f"{len(index.cvs)} CVs", stamped=bool(added)), None)


def reset(state):
    names = _names(BASE)
    return (None, gr.update(choices=names, value=names[0]), gr.update(choices=names, value=names[0]),
            status_html(len(BASE.cvs), len(BASE.chunks), BASE.semantic, llm.open_model_status()), header_html())


# ---------- tab 2: ask ----------
def ask(question, mode, key, model_name, state):
    question = (question or "").strip()
    if not question:
        raise gr.Error("Type a question first.")
    if mode == KEY and not llm.provider_for(key):
        raise gr.Error("Paste an OpenAI key (starts with sk-) or an Anthropic key (starts with sk-ant-), "
                       "or choose another answer mode.")
    index = _index(state)
    prompt, hits = llm.build_context(index, question)
    ranked, _ = index.rank_candidates(question)
    order = [r["cand"] for r in ranked]
    scores = [0.0] * len(index.cvs)
    for r in ranked:
        scores[r["cand"]] = r["score"]
    top = order[0]
    names = _names(index)
    src = sources_html(index, hits)
    rank = rank_html(index, order, scores, pct=False, compact=True)
    doc = cv_html(index.cvs[top], _highlights(index, hits, top))
    pick = gr.update(choices=names, value=names[top])
    meta = _meta(mode, model_name, key)

    if mode == SEARCH:
        yield answer_html(_render(llm.extractive_answer(index, question), 0), meta), src, rank, doc, pick, hits
        return
    waiting = "Reading the excerpts…"
    if mode == OPEN and not ON_ZERO_GPU and llm.open_model_status() == "not loaded yet":
        waiting = ("Loading the open model. The first time, this downloads it (about 8 GB for Qwen3 4B), "
                   "so it can take a few minutes. Later answers start straight away.")
    yield answer_html(f'<p class="empty">{waiting}</p>', meta), src, rank, doc, pick, hits
    user = f"{prompt}\n\nQuestion: {question}"
    text = ""
    try:
        for text in _stream_llm(mode, key, model_name, llm.SYSTEM, user):
            yield answer_html(_render(text, len(hits)), meta), src, rank, doc, pick, hits
    except Exception as exc:  # includes ZeroGPU quota errors: fall back to a search-only answer
        warn = _friendly(exc)
        body = _render(text, len(hits)) if text else _render(llm.extractive_answer(index, question), 0)
        yield answer_html(body, meta if text else _meta(SEARCH, "", ""), warn), src, rank, doc, pick, hits
        return
    # show the CV that the answer cites most
    cited = [int(n) for group in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d{1,2}", group)
             if 0 < int(n) <= len(hits)]
    if cited:
        counts = {}
        for n in cited:
            c = hits[n - 1].chunk.cand
            counts[c] = counts.get(c, 0) + 1
        best = max(counts, key=counts.get)
        if best != top:
            yield (answer_html(_render(text, len(hits)), meta), src, rank,
                   cv_html(index.cvs[best], _highlights(index, hits, best)), gr.update(choices=names, value=names[best]), hits)


def show_cited(label, hits, state):
    index = _index(state)
    i = _pick(index, label)
    return cv_html(index.cvs[i], _highlights(index, hits or [], i))


def _friendly(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "quota" in low or "zerogpu" in low or ("gpu" in low and "exceed" in low):
        return "The free GPU quota for this visitor is used up for now, so this answer comes from search only. Try again later, or use your own API key."
    if "auth" in low or "api key" in low or "401" in low:
        return "The API key was rejected. Check it and try again."
    if "model" in low and ("not found" in low or "404" in low or "does not exist" in low):
        return "That model name wasn't recognised by the provider. Check the model field."
    return f"The LLM call failed ({type(exc).__name__}: {msg[:160]}), so this answer comes from search only."


# ---------- tab 3: match a job ----------
def match(job, state):
    index = _index(state)
    if not (job or "").strip():
        raise gr.Error("Paste a job description first.")
    m = index.match_job(job)
    if not m["requirements"]:
        raise gr.Error("Write each requirement on its own line, ideally as a bullet starting with -.")
    names = _names(index)
    top = m["order"][0]
    return (rank_html(index, m["order"], m["totals"]), grid_html(index, m), why_html(index, m, top),
            gr.update(choices=[names[c] for c in m["order"]], value=names[top]), m)


def show_why(label, m, state):
    if not m:
        return ""
    index = _index(state)
    return why_html(index, m, _pick(index, label))


def notes(job, m, mode, key, model_name, state):
    index = _index(state)
    if not m:
        raise gr.Error("Rank the candidates first.")
    if mode == SEARCH:
        raise gr.Error("Shortlist notes need an LLM. Choose the open model or your API key in the Ask tab.")
    if mode == KEY and not llm.provider_for(key):
        raise gr.Error("Paste an OpenAI or Anthropic key in the Ask tab first.")
    prompt, hits = llm.build_match_prompt(index, job, m)
    meta = _meta(mode, model_name, key).replace("from the excerpts below", "shortlist notes")
    yield answer_html('<p class="empty">Writing notes…</p>', meta), sources_html(index, hits)
    text = ""
    try:
        for text in _stream_llm(mode, key, model_name, llm.SYSTEM, f"{prompt}\n\nTask: {llm.MATCH_TASK}"):
            yield answer_html(_render(text, len(hits)), meta), sources_html(index, hits)
    except Exception as exc:
        yield answer_html(_render(text, len(hits)), meta, _friendly(exc)), sources_html(index, hits)


# ---------- layout ----------
with gr.Blocks(title="CV Parser with RAG") as demo:
    state = gr.State(None)
    last_hits = gr.State([])
    last_match = gr.State(None)
    header = gr.HTML(header_html())
    status = gr.HTML(status_html(len(BASE.cvs), len(BASE.chunks), BASE.semantic, llm.open_model_status()))

    with gr.Tabs(elem_classes="maintabs"):
        with gr.Tab("01 · Parse"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=240, elem_classes="paper"):
                    cand = gr.Radio(_names(BASE), value=_names(BASE)[0], label="Candidates", elem_classes="cands")
                    files = gr.File(label="Add CVs (PDF, DOCX or TXT)", file_count="multiple",
                                    file_types=[".pdf", ".docx", ".txt"], elem_classes="drop paperfile")
                    add_btn = gr.Button("Parse and add", variant="primary", elem_classes="go")
                    reset_btn = gr.Button("Back to the 5 samples", variant="secondary")
                    gr.HTML('<p class="note">The five sample candidates are fictional. CVs you add stay in your browser '
                            'session only and are not stored.</p>')
                with gr.Column(scale=5, min_width=320):
                    doc = gr.HTML()
                with gr.Column(scale=4, min_width=280):
                    fields = gr.HTML()
                    json_file = gr.File(label="Download parsed CV (JSON)", elem_classes="paperfile")

        with gr.Tab("02 · Ask across CVs"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=260, elem_classes="paper"):
                    question = gr.Textbox(label="Question", lines=2, value=EXAMPLES[0],
                                          placeholder="Ask about all candidates, e.g. who has led a team?")
                    with gr.Row(equal_height=True):
                        ask_btn = gr.Button("Ask", variant="primary", elem_classes="go", scale=3)
                        stop_btn = gr.Button("Stop", variant="secondary", scale=1)
                    gr.Examples(EXAMPLES, inputs=question, label="Try", elem_id="ask-examples")
                    mode = gr.Radio([OPEN, KEY, SEARCH], value=default_mode(), label="Answer with", elem_classes="cands")
                    with gr.Accordion("Your API key (optional)", open=False, elem_classes="more"):
                        key = gr.Textbox(label="API key", type="password", placeholder="sk-... or sk-ant-...",
                                         elem_classes="key")
                        model_name = gr.Textbox(label="Model", placeholder="claude-haiku-4-5 or gpt-5-mini (default)")
                        gr.HTML('<p class="note">The key is sent to OpenAI or Anthropic for this request only. '
                                'It is never stored or logged.</p>')
                with gr.Column(scale=5, min_width=340):
                    answer = gr.HTML(answer_html())
                    sources = gr.HTML(sources_html(BASE, []))
                with gr.Column(scale=4, min_width=300):
                    ranking = gr.HTML()
                    cited_pick = gr.Dropdown(_names(BASE), value=_names(BASE)[0], label="CV with cited lines highlighted")
                    cited_doc = gr.HTML()

        with gr.Tab("03 · Match a job"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300, elem_classes="paper"):
                    job = gr.Textbox(JOB, label="Job description", lines=12,
                                     info="One requirement per line. 'N+ years' lines are checked against parsed experience; "
                                          "the rest are matched by meaning against every CV line.")
                    match_btn = gr.Button("Rank candidates", variant="primary", elem_classes="go")
                    notes_btn = gr.Button("Write shortlist notes (LLM)", variant="secondary")
                with gr.Column(scale=8):
                    m_rank = gr.HTML()
                    m_grid = gr.HTML()
                    why_pick = gr.Dropdown([], label="Evidence for", interactive=True)
                    m_why = gr.HTML()
                    m_notes = gr.HTML()
                    m_sources = gr.HTML()

    gr.HTML(
        '<footer class="foot">Built by <a href="https://www.linkedin.com/in/hamzaasyf/">Muhammad Hamza Asif</a>'
        ' · <a href="https://github.com/hamzaasyf/cv-parser-rag">Source on GitHub</a>'
        f' · bge-small-en-v1.5 embeddings · {llm.OPEN_MODEL.split("/")[-1]} · sample CVs are fictional</footer>'
    )

    demo.load(show_cv, [cand, state], [doc, fields, json_file])
    cand.change(show_cv, [cand, state], [doc, fields, json_file])
    add_btn.click(add_cvs, [files, state], [state, cand, cited_pick, status, header, files])
    reset_btn.click(reset, [state], [state, cand, cited_pick, status, header])

    ask_args = dict(fn=ask, inputs=[question, mode, key, model_name, state],
                    outputs=[answer, sources, ranking, cited_doc, cited_pick, last_hits])
    # a new question while an answer is still being written runs as soon as that one ends
    # (Gradio's default would silently ignore the click); Stop ends the current answer now
    ask_click = ask_btn.click(**ask_args, trigger_mode="always_last")
    ask_enter = question.submit(**ask_args, trigger_mode="always_last")
    cited_pick.input(show_cited, [cited_pick, last_hits, state], cited_doc)

    match_btn.click(match, [job, state], [m_rank, m_grid, m_why, why_pick, last_match])
    why_pick.input(show_why, [why_pick, last_match, state], m_why)
    notes_run = notes_btn.click(notes, [job, last_match, mode, key, model_name, state], [m_notes, m_sources],
                                trigger_mode="always_last")
    stop_btn.click(None, None, None, cancels=[ask_click, ask_enter, notes_run])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(theme=THEME, css=CSS, head=HEAD, show_error=True)
