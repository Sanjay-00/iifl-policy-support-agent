import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import POLICY_DIR


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    file: str
    title: str
    category: str
    version: str
    effective_date: str
    section: str
    text: str

    @property
    def source(self) -> str:
        return f"{self.file} > {self.section}"

    @property
    def index_text(self) -> str:
        # Title and heading carry most of the topic signal for short sections.
        return f"{self.title}. {self.section}. {self.text}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_front_matter(raw: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not match:
        raise ValueError("policy file is missing a front-matter block")
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, match.group(2)


def split_sections(body: str) -> list[tuple[str, str]]:
    # Anything before the first "## " heading (e.g. the synthetic-document notice) is not policy content.
    parts = re.split(r"^## (.+)$", body, flags=re.MULTILINE)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def is_in_force(meta: dict, today: date) -> bool:
    return meta.get("status") == "active" and date.fromisoformat(meta["effective_date"]) <= today


def load_chunks(policy_dir: Path = POLICY_DIR, today: date | None = None) -> list[Chunk]:
    today = today or date.today()
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
        if not is_in_force(meta, today):
            continue
        for section, text in split_sections(body):
            chunks.append(Chunk(
                chunk_id=f"{meta['document_id']}#{slugify(section)}",
                document_id=meta["document_id"],
                file=path.name,
                title=meta["title"],
                category=meta["category"],
                version=meta["version"],
                effective_date=meta["effective_date"],
                section=section,
                text=text,
            ))
    if not chunks:
        raise RuntimeError(f"no active policy content found in {policy_dir}")
    return chunks
