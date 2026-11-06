"""Look and feel: a light recruiter workspace (white cards, Inter, teal) and the HTML components."""

from __future__ import annotations

import datetime as _dt
import html

import gradio as gr

TEAL = "#0F766E"
TEAL_DARK = "#0B5D57"
TEAL_SOFT = "#E6F4F1"
BG = "#F5F7F6"
CARD = "#FFFFFF"
INK = "#1A1D29"
MUTED = "#6B7280"
FAINT = "#8A8FA0"
LINE = "#E3E7E5"
MARK = "#FDF0C7"
RED = "#B42318"
# kept for the grid shading in grid_html
PAPER = CARD

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

_THEME_VARS = dict(
    body_background_fill=BG,
    body_text_color=INK,
    body_text_color_subdued=MUTED,
    background_fill_primary=CARD,
    background_fill_secondary=BG,
    block_background_fill=CARD,
    block_border_color=LINE,
    block_border_width="1px",
    block_radius="14px",
    block_label_background_fill=CARD,
    block_label_text_color=MUTED,
    block_label_border_color=LINE,
    block_title_text_color=FAINT,
    block_shadow="0 1px 2px rgba(20,30,40,.04)",
    border_color_primary=LINE,
    border_color_accent=TEAL,
    input_background_fill=CARD,
    input_border_color="#D6DCD9",
    input_border_color_focus=TEAL,
    input_placeholder_color=FAINT,
    input_radius="10px",
    color_accent=TEAL,
    color_accent_soft=TEAL_SOFT,
    slider_color=TEAL,
    checkbox_background_color=CARD,
    checkbox_background_color_selected=TEAL,
    checkbox_border_color="#CBD2CF",
    checkbox_border_color_selected=TEAL,
    checkbox_label_background_fill=CARD,
    checkbox_label_background_fill_selected=TEAL_SOFT,
    checkbox_label_border_color=LINE,
    checkbox_label_border_color_selected=TEAL,
    checkbox_label_text_color_selected=TEAL_DARK,
    button_primary_background_fill=TEAL,
    button_primary_background_fill_hover=TEAL_DARK,
    button_primary_text_color="#FFFFFF",
    button_primary_border_color=TEAL,
    button_secondary_background_fill=CARD,
    button_secondary_background_fill_hover=TEAL_SOFT,
    button_secondary_text_color=TEAL_DARK,
    button_secondary_border_color="#CFE5E0",
    button_large_radius="10px",
    button_small_radius="8px",
    panel_background_fill=CARD,
    panel_border_color=LINE,
    link_text_color=TEAL,
    link_text_color_hover=TEAL_DARK,
)

_base = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")],
    font_mono=[gr.themes.Font("ui-monospace"), gr.themes.Font("Menlo"), gr.themes.Font("monospace")],
    radius_size=gr.themes.sizes.radius_lg,
)
# One look in light and dark mode, so visitors with a dark browser or Hugging Face theme see the same page
_dark = {f"{k}_dark": v for k, v in _THEME_VARS.items() if hasattr(_base, f"{k}_dark")}
THEME = _base.set(**{k: v for k, v in _THEME_VARS.items() if hasattr(_base, k)},
                  **{k: v for k, v in _dark.items()})

CSS = f"""
:root, .dark {{ color-scheme: light; }}
.gradio-container {{ max-width: 1320px !important; padding: 0 28px 36px !important; }}
footer:not(.foot) {{ display: none !important; }}
body, .gradio-container {{ background:{BG} !important; }}
.gradio-container, .gradio-container * {{ font-family:'Inter', system-ui, sans-serif; }}

/* ---- top bar ---- */
.topbar {{ display:flex; align-items:center; gap:16px; padding:18px 0 14px; border-bottom:1px solid {LINE}; margin:0 0 4px; }}
.topbar .logo {{ width:36px; height:36px; border-radius:10px; background:{TEAL}; color:#fff !important; display:grid; place-items:center;
  font-weight:700; font-size:13px; letter-spacing:.02em; }}
.topbar .nm {{ font-weight:700; font-size:18px; color:{INK} !important; line-height:1.2; }}
.topbar .sb {{ font-size:12.5px; color:{MUTED} !important; }}
.topbar .right {{ margin-left:auto; text-align:right; font-size:12px; color:{FAINT} !important; }}
.topbar .right b {{ color:{TEAL} !important; font-weight:600; }}
.lede {{ color:{MUTED} !important; font-size:14px; max-width:860px; margin:12px 0 4px; line-height:1.6; }}
.status-line {{ color:{FAINT} !important; font-size:12px; }}
.status-line * {{ color:{FAINT} !important; }}

/* ---- main tabs as pills ---- */
.maintabs > .tab-wrapper, .maintabs .tab-container {{ border:none !important; }}
.maintabs button[role="tab"] {{ color:#4B5563 !important; font-weight:500; font-size:14px; border:none !important;
  background:transparent !important; border-radius:9px !important; padding:8px 14px !important; }}
.maintabs button[role="tab"][aria-selected="true"] {{ background:{TEAL_SOFT} !important; color:{TEAL_DARK} !important; font-weight:600; }}
.maintabs button[role="tab"][aria-selected="true"]::after {{ display:none !important; }}
.maintabs .tabitem {{ background:transparent !important; border:none !important; padding:14px 0 0 !important; }}

/* ---- control card ---- */
.paper {{ background:{CARD} !important; border:1px solid {LINE} !important; border-radius:14px !important; padding:16px !important;
  gap:12px !important; box-shadow:0 1px 2px rgba(20,30,40,.04); }}
.paper .block, .paper .form {{ border:none !important; background:transparent !important; box-shadow:none !important; }}
.paper label > span, .paper .block-title, .paper [data-testid="block-info"] {{ text-transform:uppercase; letter-spacing:.06em;
  font-size:11px !important; font-weight:600; color:{FAINT} !important; }}
.paper .info-text {{ color:{MUTED} !important; font-size:12px !important; text-transform:none !important; letter-spacing:0 !important; font-weight:400 !important; }}
.paper textarea, .paper input[type="text"], .paper input[type="password"] {{ border:1px solid #D6DCD9 !important;
  border-radius:10px !important; background:{CARD} !important; padding:9px 11px !important; }}
.paper textarea:focus, .paper input:focus {{ border-color:{TEAL} !important; box-shadow:0 0 0 3px {TEAL_SOFT} !important; }}
.go button, button.go {{ font-weight:600; font-size:14.5px !important; padding:11px !important; }}
.note {{ font-size:11.5px; color:{FAINT}; margin:0; line-height:1.5; }}
.more {{ border:1px solid {LINE} !important; border-radius:12px !important; }}
.drop {{ border:1px dashed #C9D3CF !important; border-radius:12px !important; }}
.paperfile {{ border-radius:12px !important; }}

/* candidate list and answer-mode picker: stacked options */
.cands .wrap {{ display:grid !important; gap:6px !important; }}
.cands .wrap label {{ margin:0 !important; border:1px solid {LINE} !important; background:{CARD} !important; padding:9px 11px !important;
  box-shadow:none !important; border-radius:10px !important; }}
.cands .wrap label input {{ display:none; }}
.cands .wrap label.selected, .cands .wrap label:has(input:checked) {{ background:{TEAL_SOFT} !important; border-color:{TEAL} !important; }}
.cands .wrap label span {{ text-transform:none !important; letter-spacing:0 !important; font-size:13.5px !important;
  color:{INK} !important; margin:0 !important; font-weight:500; }}
.cands .wrap label.selected span, .cands .wrap label:has(input:checked) span {{ color:{TEAL_DARK} !important; font-weight:600; }}

/* example questions as soft pills */
#ask-examples button, #ask-examples .gallery-item {{ background:#F1F3F2 !important; border:1px solid transparent !important; border-radius:99px !important;
  font-size:12.5px !important; color:#4A5160 !important; padding:5px 11px !important; text-align:left !important; }}
#ask-examples button:hover {{ border-color:{TEAL} !important; color:{TEAL_DARK} !important; }}
.key input {{ font-family:ui-monospace, Menlo, monospace !important; }}

/* ---- cards used by the HTML components ---- */
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; box-shadow:0 1px 2px rgba(20,30,40,.04); }}
.card * {{ color:{INK}; }}
.lbl {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; font-weight:600; color:{FAINT} !important; }}
.empty {{ padding:18px; color:{FAINT} !important; font-size:13.5px; }}

/* the CV */
.cvdoc {{ padding:24px 26px; font-size:14px; line-height:1.55; max-height:720px; overflow:auto; }}
.cvdoc .nm {{ font-size:22px; font-weight:700; line-height:1.2; }}
.cvdoc .tt {{ font-size:13.5px; font-weight:500; margin-top:2px; color:{TEAL} !important; }}
.cvdoc .ct {{ font-size:12.5px; color:{FAINT} !important; margin-top:4px; padding-bottom:12px; border-bottom:1px solid {LINE}; }}
.cvdoc h4 {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; margin:16px 0 6px; color:{FAINT} !important; font-weight:600; }}
.cvdoc p {{ margin:0 0 4px; }} .cvdoc ul {{ margin:2px 0 8px; padding-left:18px; }} .cvdoc li {{ margin:3px 0; }}
.cvdoc .role {{ display:flex; justify-content:space-between; gap:10px; margin-top:10px; font-weight:600; }}
.cvdoc .role span {{ color:{FAINT} !important; font-weight:400; font-size:12.5px; white-space:nowrap; }}
.cvdoc .hit {{ background:rgba(250,214,110,var(--a)); border-radius:4px; padding:1px 2px; box-decoration-break:clone; -webkit-box-decoration-break:clone; }}
.cvdoc .hit sup {{ display:inline-block; font-size:10px; line-height:1; padding:2px 4px; margin-left:3px; border-radius:4px;
  background:{TEAL}; color:#fff !important; font-weight:600; vertical-align:super; }}

/* parsed fields */
.fields {{ padding:18px; display:grid; gap:16px; }}
.kv {{ display:grid; grid-template-columns:auto minmax(0,1fr); gap:4px 14px; font-size:13.5px; }}
.kv span {{ color:{FAINT} !important; }} .kv b {{ font-weight:500; overflow-wrap:anywhere; }}
.big {{ font-size:32px; font-weight:700; line-height:1.1; color:{TEAL} !important; }}
.fields small {{ color:{FAINT} !important; font-size:11.5px; }}
.tl {{ display:grid; gap:8px; }} .tlr span {{ font-size:12px; color:{MUTED} !important; }}
.tlb {{ position:relative; height:8px; background:#EDF1EF; border-radius:9px; }}
.tlb i {{ position:absolute; top:0; bottom:0; background:#9CCFC6; border-radius:9px; }}
.tlr.now .tlb i {{ background:{TEAL}; }}
.chipset {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:4px; }}
.chipset span {{ font-size:12px; padding:2px 9px; border:1px solid {LINE}; border-radius:99px; color:#4A5160 !important; }}
.chipset span.ev {{ background:{TEAL_SOFT}; border-color:transparent; color:{TEAL_DARK} !important; font-weight:500; }}
.fields .cat {{ font-size:11.5px; color:{FAINT} !important; margin-top:8px; }}

/* answers */
.answer {{ padding:16px 20px; min-height:90px; font-size:15.5px; line-height:1.65; border-left:4px solid {TEAL}; }}
.answer .meta {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; font-weight:600; color:{FAINT} !important; margin-bottom:6px; }}
.answer .warn {{ font-size:12.5px; color:{RED} !important; margin-top:8px; }}
.answer p {{ margin:0 0 8px; }} .answer ul, .answer ol {{ margin:0 0 8px; padding-left:22px; }}
.answer strong {{ font-weight:600; }}
sup.cite {{ display:inline-block; font-size:10.5px; line-height:1; padding:2px 5px; margin:0 1px; border-radius:5px;
  background:{TEAL_SOFT}; color:{TEAL_DARK} !important; font-weight:600; vertical-align:super; }}
.sources {{ padding:14px 16px; max-height:460px; overflow:auto; display:grid; gap:10px; }}
.src {{ display:grid; grid-template-columns:26px minmax(0,1fr); gap:10px; font-size:13.5px; line-height:1.5; }}
.src i {{ font-style:normal; font-size:11px; font-weight:600; height:22px; display:grid; place-items:center; border-radius:6px;
  background:{TEAL_SOFT}; color:{TEAL_DARK} !important; }}
.src .w {{ color:{FAINT} !important; }} .src .w b {{ color:{INK} !important; font-weight:600; }}

/* ranking */
.rank {{ padding:14px 16px; display:grid; gap:10px; }}
.rank .r {{ display:grid; grid-template-columns:34px minmax(0,1fr) 110px 46px; gap:12px; align-items:center; font-size:14px; }}
.rank .av {{ width:34px; height:34px; border-radius:50%; background:{TEAL_SOFT}; color:{TEAL_DARK} !important; display:grid; place-items:center;
  font-size:11.5px; font-weight:600; }}
.rank .lbl + .r .av {{ background:{TEAL}; color:#fff !important; }}
.rank .nm b {{ font-weight:600; display:block; }} .rank .nm small {{ display:block; font-size:12px; color:{FAINT} !important; }}
.rank .bar {{ height:7px; background:#EDF1EF; border-radius:9px; overflow:hidden; }} .rank .bar i {{ display:block; height:100%; background:{TEAL}; border-radius:9px; }}
.rank .s {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; color:{TEAL} !important; font-size:13px; }}
.rank.compact .r {{ grid-template-columns:34px minmax(0,1fr) 64px 38px; gap:10px; }}

/* requirement grid and evidence */
.grid {{ overflow-x:auto; }}
.grid table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:13.5px; }}
.grid th {{ text-align:center; font-weight:600; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:{FAINT} !important;
  padding:12px 8px; border-bottom:1px solid {LINE}; background:{CARD} !important; }}
.grid th:first-child {{ text-align:left; padding-left:16px; }}
.grid td {{ border-bottom:1px solid #EEF1F0; padding:6px; background:{CARD} !important; }}
.grid tr:last-child td {{ border-bottom:none; }}
.grid table {{ border:none !important; }}
.grid table, .grid th, .grid td {{ border-left:none !important; border-right:none !important; border-top:none !important; }}
.grid th {{ border-bottom:1px solid {LINE} !important; }} .grid td {{ border-bottom:1px solid #EEF1F0 !important; }}
.grid tr:last-child td {{ border-bottom:none !important; }}
.grid td.req {{ min-width:230px; padding-left:16px; color:{INK} !important; }}
.grid td.v {{ width:78px; text-align:center; }}
.grid td.v span {{ display:block; border-radius:8px; padding:7px 0; font-variant-numeric:tabular-nums; font-weight:600; cursor:help; }}
.why {{ padding:16px 18px; display:grid; gap:12px; font-size:13.5px; line-height:1.5; }}
.why .item b.pc {{ display:inline-block; min-width:40px; text-align:center; border-radius:6px; padding:1px 6px; margin-right:8px;
  background:{TEAL_SOFT}; color:{TEAL_DARK} !important; font-weight:600; font-size:12px; }}
.why q {{ color:#374151 !important; }} .why small {{ color:{FAINT} !important; }}

.foot {{ color:{FAINT}; font-size:12px; margin-top:26px; text-align:center; }}
.foot a {{ color:{TEAL}; text-decoration:none; }}
.foot a:hover {{ text-decoration:underline; }}

@media (max-width: 760px) {{
  .gradio-container {{ padding:0 16px 24px !important; }}
  .topbar .right {{ display:none; }}
  .rank .r {{ grid-template-columns:34px minmax(0,1fr) 60px 40px; }}
  .cvdoc {{ padding:18px; }}
}}
"""


def esc(s) -> str:
    return html.escape(str(s or ""))


def header_html(file_info: str = "5 sample CVs", stamped: bool = False) -> str:
    return f"""
<div class="topbar">
  <span class="logo">CV</span>
  <div><div class="nm">CV Parser</div><div class="sb">Parse · ask · shortlist</div></div>
  <div class="right">Loaded: <b>{esc(file_info)}</b></div>
</div>
<p class="lede">Turn CVs into structured data, ask questions across all of them, and rank candidates against a job description.
Answers are written by an LLM from retrieved CV excerpts only, and every claim cites the excerpt it came from.</p>
"""


def _ym(t) -> int:
    return t[0] * 12 + t[1]


def cv_html(cv, highlights: dict | None = None) -> str:
    """The CV as a document. highlights maps (section, role index, text) to (strength 0-1, source numbers)."""
    hl = highlights or {}

    def mark(text, section, role=None):
        key = (section, role, text)
        if key not in hl:
            return esc(text)
        a, nums = hl[key]
        sup = "".join(f"<sup>{n}</sup>" for n in nums)
        return f'<span class="hit" style="--a:{0.18 + 0.32 * a:.2f}">{esc(text)}{sup}</span>'

    contact = " · ".join(x for x in [cv.location, cv.email, cv.phone, *cv.links] if x)
    s = [f'<div class="card cvdoc"><div class="nm">{esc(cv.name)}</div><div class="tt">{esc(cv.title)}</div><div class="ct">{esc(contact)}</div>']
    if cv.summary:
        s.append(f"<h4>Summary</h4><p>{mark(cv.summary, 'Summary')}</p>")
    if cv.roles:
        s.append("<h4>Experience</h4>")
        for ri, r in enumerate(cv.roles):
            head = f"{r.title} at {r.company}" if r.company else r.title
            head_key = head + (f" ({r.dates})" if r.dates else "")
            shown = esc(r.title) + (f" — {esc(r.company)}" if r.company else "")
            if ("Experience", ri, head_key) in hl:
                shown = mark(head_key, "Experience", ri).replace(esc(head_key), shown)
            s.append(f'<div class="role"><div>{shown}</div><span>{esc(r.dates)}</span></div><ul>')
            s += [f"<li>{mark(b, 'Experience', ri)}</li>" for b in r.bullets]
            s.append("</ul>")
    for sec, items in (("Education", cv.education), ("Certifications", cv.certifications), ("Projects", cv.projects)):
        if items:
            s.append(f"<h4>{sec}</h4>" + "".join(f"<p>{mark(e, sec)}</p>" for e in items))
    if cv.skills_listed:
        s.append(f"<h4>Skills</h4><p>{mark(', '.join(cv.skills_listed), 'Skills')}</p>")
    if cv.languages:
        joined = "Speaks " + ", ".join(cv.languages)
        inner = mark(joined, "Languages").replace("Speaks ", "", 1)
        s.append(f"<h4>Languages</h4><p>{inner}</p>")
    s.append("</div>")
    return "".join(s)


def fields_html(cv, categories) -> str:
    now = _dt.date.today()
    dated = [r for r in cv.roles if r.start]
    s = ['<div class="card fields">']
    s.append('<div class="kv">' + "".join(
        f"<span>{k}</span><b>{esc(v) or '–'}</b>" for k, v in
        [("Name", cv.name), ("Title", cv.title), ("Location", cv.location), ("Email", cv.email), ("Phone", cv.phone)]
    ) + "</div>")
    s.append(f'<div><div class="lbl">Total experience</div><div class="big">{cv.years} yrs</div>'
             f'<small>from role dates, overlapping jobs merged</small></div>')
    if dated:
        first = min(_ym(r.start) for r in dated)
        span = max(12, now.year * 12 + now.month - first + 1)
        rows = []
        for r in dated:
            left = (_ym(r.start) - first) / span * 100
            width = r.months / span * 100
            rows.append(f'<div class="tlr{" now" if r.present else ""}"><div class="tlb"><i style="left:{left:.1f}%;width:{width:.1f}%"></i></div>'
                        f'<span>{esc(r.title)} · {r.months / 12:.1f} y</span></div>')
        s.append('<div class="tl">' + "".join(rows) + f'<small>{first // 12} → now</small></div>')
    s.append(f'<div><div class="lbl">Skills · {len(cv.skills)} found</div>')
    for cat in categories:
        xs = [k for k in cv.skills if k["category"] == cat]
        if xs:
            chips = "".join(f'<span class="{"ev" if k["used_in_role"] else ""}" title="{"Used in a role" if k["used_in_role"] else "Listed only"}">{esc(k["name"])}</span>' for k in xs)
            s.append(f'<div class="cat">{cat}</div><div class="chipset">{chips}</div>')
    s.append('<small>Tinted = used in a role · outline = listed only</small></div>')
    for label, items in (("Education", cv.education), ("Certifications", cv.certifications), ("Languages", cv.languages)):
        if items:
            s.append(f'<div><div class="lbl">{label}</div>' + "".join(f"<div>{esc(e)}</div>" for e in items) + "</div>")
    s.append("</div>")
    return "".join(s)


def answer_html(body: str = "", meta: str = "", warn: str = "") -> str:
    if not body and not meta:
        return '<div class="card answer"><div class="empty">Ask a question, or pick one of the examples.</div></div>'
    w = f'<div class="warn">{esc(warn)}</div>' if warn else ""
    return f'<div class="card answer"><div class="meta">{esc(meta)}</div>{body}{w}</div>'


def sources_html(index, hits) -> str:
    if not hits:
        return '<div class="card sources"><div class="empty">The CV excerpts behind the answer appear here.</div></div>'
    rows = []
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        where = c.context if c.section == "Experience" and c.context else c.section
        rows.append(f'<div class="src"><i>{n}</i><div><span class="w"><b>{esc(index.cvs[c.cand].name)}</b> · {esc(where)}</span>'
                    f'<br>{esc(c.text)}</div></div>')
    return '<div class="card sources"><div class="lbl">Sources</div>' + "".join(rows) + "</div>"


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p[:1].isalpha()]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def rank_html(index, order, scores, pct: bool = True, compact: bool = False) -> str:
    if not order:
        return ""
    top = max(scores) or 1
    rows = []
    for c in order:
        cv = index.cvs[c]
        v = scores[c] if pct else scores[c] / top
        label = f"{round(scores[c] * 100)}%" if pct else f"{scores[c]:.2f}"
        rows.append(f'<div class="r"><span class="av">{esc(_initials(cv.name))}</span>'
                    f'<span class="nm"><b>{esc(cv.name)}</b><small>{esc(cv.title)} · {cv.years} yrs</small></span>'
                    f'<span class="bar"><i style="width:{max(0, min(1, v)) * 100:.0f}%"></i></span><span class="s">{label}</span></div>')
    return (f'<div class="card rank{" compact" if compact else ""}"><div class="lbl">Ranking</div>' + "".join(rows) + "</div>")


def grid_html(index, match) -> str:
    order = match["order"]
    head = "<tr><th>Requirement</th>" + "".join(f"<th>{esc(index.cvs[c].name.split()[0])}</th>" for c in order) + "</tr>"
    rows = []
    for row in match["requirements"]:
        cells = []
        for c in order:
            cell = row["cells"][c]
            v = cell["value"]
            tip = cell["why"] if cell["kind"] == "years" else f"“{cell['why']}” ({cell.get('context') or cell['section']})"
            shade = f"rgba(15,118,110,{0.06 + 0.84 * v:.2f})"
            fg = "#FFFFFF" if v > 0.55 else INK
            cells.append(f'<td class="v" title="{esc(tip)}"><span style="background:{shade};color:{fg} !important">{round(v * 100)}</span></td>')
        rows.append(f'<tr><td class="req">{esc(row["requirement"])}</td>{"".join(cells)}</tr>')
    return f'<div class="card grid"><table>{head}{"".join(rows)}</table></div>'


def why_html(index, match, cand: int | None = None) -> str:
    if not match["order"]:
        return ""
    cand = match["order"][0] if cand is None else cand
    cv = index.cvs[cand]
    items = []
    for row in match["requirements"]:
        cell = row["cells"][cand]
        pc = f'<b class="pc">{round(cell["value"] * 100)}</b>'
        if cell["kind"] == "years":
            items.append(f'<div class="item">{pc}{esc(row["requirement"])}<br><small>{esc(cell["why"])}</small></div>')
        else:
            where = cell.get("context") or cell["section"]
            items.append(f'<div class="item">{pc}{esc(row["requirement"])}<br><q>{esc(cell["why"])}</q> <small>· {esc(where)}</small></div>')
    return f'<div class="card why"><div class="lbl">Evidence for {esc(cv.name)}</div>{"".join(items)}</div>'


def status_html(n_cvs: int, n_chunks: int, semantic: bool, model_note: str) -> str:
    search = "bge-small embeddings + BM25 hybrid search" if semantic else "BM25 keyword search (embedding model not loaded)"
    return f'<div class="status-line">{n_cvs} CVs parsed · {n_chunks} chunks indexed · {search} · LLM: {esc(model_note)}</div>'
