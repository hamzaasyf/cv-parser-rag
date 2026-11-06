"""Turn CV text into structured fields: contact, roles with dates, experience, skills, education."""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------- skill taxonomy: canonical name, category, aliases ----------
_SKILLS = [
    ("Python", "Languages"), ("JavaScript", "Languages"), ("TypeScript", "Languages"), ("SQL", "Languages"),
    ("Java", "Languages"), ("Go", "Languages", ["Golang"]), ("C++", "Languages"), ("Rust", "Languages"),
    ("Bash", "Languages", ["shell scripting"]),
    ("PyTorch", "ML & AI"), ("TensorFlow", "ML & AI"), ("scikit-learn", "ML & AI", ["sklearn"]),
    ("XGBoost", "ML & AI"), ("Hugging Face", "ML & AI", ["transformers library"]), ("OpenCV", "ML & AI"),
    ("YOLO", "ML & AI", ["YOLOv5", "YOLOv8", "YOLO11"]), ("TensorRT", "ML & AI"), ("ONNX", "ML & AI"),
    ("CUDA", "ML & AI"), ("NVIDIA Jetson", "ML & AI", ["Jetson"]), ("spaCy", "ML & AI"),
    ("Computer vision", "ML & AI"), ("NLP", "ML & AI", ["natural language processing"]),
    ("Speech recognition", "ML & AI", ["ASR", "Whisper"]),
    ("LLMs", "LLM & RAG", ["large language models", "LLM"]), ("RAG", "LLM & RAG", ["retrieval-augmented generation"]),
    ("LangChain", "LLM & RAG"), ("LlamaIndex", "LLM & RAG"), ("FAISS", "LLM & RAG"), ("LoRA", "LLM & RAG"),
    ("Vector databases", "LLM & RAG", ["vector database", "vector index", "Pinecone", "Qdrant", "Weaviate", "pgvector"]),
    ("Prompt engineering", "LLM & RAG"),
    ("MLflow", "MLOps & Cloud"), ("Docker", "MLOps & Cloud"), ("Kubernetes", "MLOps & Cloud", ["k8s"]),
    ("Terraform", "MLOps & Cloud"), ("AWS", "MLOps & Cloud", ["EMR", "SageMaker"]),
    ("GCP", "MLOps & Cloud", ["Google Cloud"]), ("Azure", "MLOps & Cloud"), ("GitHub Actions", "MLOps & Cloud"),
    ("Linux", "MLOps & Cloud"), ("Prometheus", "MLOps & Cloud"), ("Grafana", "MLOps & Cloud"),
    ("Spark", "Data", ["Apache Spark"]), ("Airflow", "Data", ["Apache Airflow"]), ("Kafka", "Data"), ("dbt", "Data"),
    ("Snowflake", "Data"), ("PostgreSQL", "Data", ["Postgres"]), ("Pandas", "Data"), ("Metabase", "Data"),
    ("React", "Web"), ("Next.js", "Web"), ("Node.js", "Web"), ("GraphQL", "Web"),
    ("REST APIs", "Web", ["REST API", "REST"]), ("FastAPI", "Web"), ("Tailwind CSS", "Web", ["Tailwind"]),
    ("Accessibility", "Web", ["WCAG"]),
]
CATEGORIES = ["Languages", "ML & AI", "LLM & RAG", "MLOps & Cloud", "Data", "Web"]


def _skill_re(name, aliases, case_sensitive):
    alts = "|".join(re.escape(a) for a in [name, *aliases])
    return re.compile(rf"(^|[^A-Za-z0-9+#.])({alts})(?![A-Za-z0-9+#])", 0 if case_sensitive else re.I)


SKILLS = [
    (s[0], s[1], _skill_re(s[0], s[2] if len(s) > 2 else [], s[0] in ("dbt", "spaCy", "Go", "REST")))
    for s in _SKILLS
]


def skills_in(text: str) -> list[str]:
    return [name for name, _, rx in SKILLS if rx.search(" " + text)]


# ---------- dates and sections ----------
_MON = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_MONTH_ANY = rf"(?:{_MONTH}|0?[1-9]|1[0-2])"  # "Mar", "March" or a month number as in 03/2021
_DATE = re.compile(
    rf"(?:({_MONTH_ANY})[\s/.]+)?((?:19|20)\d\d)\s*(?:[–—-]|to)\s*"
    rf"(?:(?:({_MONTH_ANY})[\s/.]+)?((?:19|20)\d\d)|(Present|Current|Now|Today|Ongoing))",
    re.I,
)
_NOT_A_NAME = re.compile(r"^(curriculum vitae|cv|r[ée]sum[ée]|resume|profile)$", re.I)
_HEADINGS = {
    "summary": r"summary|profile|professional summary|about(?: me)?|objective",
    "experience": r"experience|work experience|employment(?: history)?|professional experience|work history",
    "education": r"education|academic background|qualifications",
    "skills": r"skills|technical skills|core skills|key skills|tech stack|technologies",
    "languages": r"languages?",
    "certifications": r"certifications?|licen[cs]es?(?: & certifications)?|certificates",
    "projects": r"projects?|selected projects",
}
_HEAD_RE = {k: re.compile(rf"^(?:{v})\s*:?$", re.I) for k, v in _HEADINGS.items()}
_BULLET = re.compile(r"^[-•*·▪●◦]\s*")


@dataclass
class Role:
    title: str
    company: str = ""
    dates: str = ""
    start: tuple | None = None   # (year, month)
    end: tuple | None = None
    present: bool = False
    bullets: list = field(default_factory=list)

    @property
    def months(self) -> int:
        if not (self.start and self.end):
            return 0
        return max(0, (self.end[0] * 12 + self.end[1]) - (self.start[0] * 12 + self.start[1]) + 1)


@dataclass
class CV:
    raw: str
    file: str = ""
    name: str = ""
    title: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list = field(default_factory=list)
    summary: str = ""
    roles: list = field(default_factory=list)
    education: list = field(default_factory=list)
    skills_listed: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    certifications: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    years: float = 0.0
    skills: list = field(default_factory=list)  # [{name, category, used_in_role}]

    def as_json(self) -> dict:
        def ym(t, present=False):
            return "present" if present else (f"{t[0]}-{t[1]:02d}" if t else None)

        return {
            "name": self.name, "title": self.title, "location": self.location, "email": self.email,
            "phone": self.phone, "links": self.links, "summary": self.summary,
            "total_experience_years": self.years,
            "roles": [{"title": r.title, "company": r.company, "start": ym(r.start), "end": ym(r.end, r.present),
                       "years": round(r.months / 12, 1), "highlights": r.bullets} for r in self.roles],
            "skills": self.skills, "education": self.education, "certifications": self.certifications,
            "languages": self.languages, "projects": self.projects,
        }


def _ym(month: str | None, year: str, end: bool = False) -> tuple:
    if not month:
        return int(year), (12 if end else 1)
    m = int(month) if month.isdigit() else _MON[month.lower()[:3]]
    return int(year), m


def parse_cv(text: str, file: str = "", today: _dt.date | None = None) -> CV:
    today = today or _dt.date.today()
    text = text.replace("\r", "")
    lines = [l.strip() for l in text.split("\n")]
    cv = CV(raw=text, file=file)
    non_empty = [l for l in lines if l]
    # skip a "Curriculum Vitae" / "Resume" heading above the name
    names = [l for l in non_empty[:3] if not _NOT_A_NAME.match(l)]
    cv.name = names[0] if names else (non_empty[0] if non_empty else "Unknown")

    if m := re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        cv.email = m.group(0).rstrip(".")
    for m in re.finditer(r"\+?\d[\d\s().-]{7,}\d", text):
        candidate = m.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        # a phone has 9+ digits; "2018 - 2021" or "2019 2020" is a date range, not a phone
        if len(digits) >= 9 and not re.fullmatch(r"(?:19|20)\d\d[\s.-]+(?:19|20)\d\d", candidate):
            cv.phone = candidate
            break
    cv.links = re.findall(r"\b(?:linkedin\.com|github\.com)/[\w/-]+|\b[\w-]+\.(?:dev|io|me)\b", text)

    # the contact line holds the email or phone; location is its first plain part, the title sits above it
    ci = next((i for i, l in enumerate(lines[:12]) if "@" in l or re.search(r"\+\d", l)), -1)
    name_i = lines.index(cv.name) if cv.name in lines else 0
    if ci >= 0:
        parts = re.split(r"\s*[·|•]\s*", lines[ci])
        cv.location = next((p for p in parts if not re.search(r"@|\+?\d{3}|\.com|\.dev|/", p)), "")
        above = [l for l in lines[name_i + 1:ci] if l]
        cv.title = above[0] if above else ""

    section, role = "header", None
    for line in lines:
        if not line:
            continue
        head = next((k for k, rx in _HEAD_RE.items() if rx.match(line)), None)
        if head:
            section = head
            continue
        if section == "summary":
            cv.summary += (" " if cv.summary else "") + line
        elif section == "experience":
            date = _DATE.search(line)
            if _BULLET.match(line):
                if role:
                    role.bullets.append(_BULLET.sub("", line))
            elif date and role and not role.start:
                _set_dates(role, date, today)
            elif role and not date and _looks_like_bullet(line):
                # PDFs often lose the bullet glyph; a sentence under a role is a highlight, not a new role
                role.bullets.append(line)
            else:
                # "Title — Company", "Title at Company" or "Title, Company"; dates may share the line
                head_text = re.sub(r"\(\s*\)", "", _DATE.sub("", line)).strip(" |–—-,") if date else line
                bits = re.split(r"\s+[—–-]\s+|\s+at\s+|,\s+|\s+\|\s+", head_text, maxsplit=1)
                role = Role(title=bits[0].strip(), company=bits[1].strip() if len(bits) > 1 else "")
                if date:
                    _set_dates(role, date, today)
                cv.roles.append(role)
        elif section == "education":
            cv.education.append(line)
        elif section == "skills":
            cv.skills_listed += [s for s in re.split(r"\s*[,;·|•]\s*", line.split(":", 1)[-1]) if s]
        elif section == "languages":
            cv.languages += [s for s in re.split(r"\s*[,;·]\s*", line) if s]
        elif section == "certifications":
            cv.certifications.append(_BULLET.sub("", line))
        elif section == "projects":
            cv.projects.append(_BULLET.sub("", line))

    cv.years = _total_years(cv.roles)
    role_text = [r.title + " " + " ".join(r.bullets) for r in cv.roles]
    listed = " , ".join(cv.skills_listed)
    for name, cat, rx in SKILLS:
        if not rx.search(" " + text):
            continue
        cv.skills.append({
            "name": name, "category": cat,
            "used_in_role": any(rx.search(" " + t) for t in role_text),
            "listed": bool(rx.search(" " + listed)),
        })
    return cv


def _looks_like_bullet(line: str) -> bool:
    """A full sentence (ends with a full stop, or 7+ words) rather than a 'Title — Company' heading."""
    if re.search(r"\s[—–]\s|\s\|\s", line):
        return False
    return line.rstrip().endswith(".") or len(line.split()) >= 7


def _set_dates(role: Role, m: re.Match, today: _dt.date) -> None:
    role.dates = m.group(0)
    role.start = _ym(m.group(1), m.group(2))
    if m.group(5):
        role.end, role.present = (today.year, today.month), True
    else:
        role.end = _ym(m.group(3), m.group(4), end=True)


def _total_years(roles) -> float:
    """Total experience with overlapping jobs merged, so parallel roles are not counted twice."""
    spans = sorted((r.start[0] * 12 + r.start[1], r.end[0] * 12 + r.end[1]) for r in roles if r.start and r.end)
    months, cur_s, cur_e = 0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e + 1:
            if cur_e is not None:
                months += cur_e - cur_s + 1
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        months += cur_e - cur_s + 1
    return round(months / 12, 1)


# ---------- file loading ----------
def read_text(path: str | Path) -> str:
    """Plain text from a PDF, DOCX or text file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(path)
    if suffix == ".docx":
        import docx  # python-docx

        doc = docx.Document(str(path))
        out = []
        for p in doc.paragraphs:
            t = p.text.strip()
            style = (p.style.name or "").lower() if p.style is not None else ""
            out.append(("- " + t) if t and "list" in style and not _BULLET.match(t) else t)
        for table in doc.tables:  # some CV templates put sections in tables
            for row in table.rows:
                out.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
        return "\n".join(out)
    return path.read_text(encoding="utf-8", errors="replace")


def _pdf_text(path: Path) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        pages = []
        for page in doc:
            tp = page.get_textpage()
            pages.append(tp.get_text_range())
            tp.close()
        text = "\n".join(pages)
    finally:
        doc.close()
    # PDF bullets come out as odd glyphs; normalise them so the parser sees list items
    text = re.sub(r"^[•▪●‣⁃]\s*", "- ", text, flags=re.M)
    return re.sub(r"[ \t]+\n", "\n", text)


def load_cv(path: str | Path) -> CV:
    return parse_cv(read_text(path), file=Path(path).name)
