import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

POLICY_DIR = ROOT / "data" / "policies"
AUDIT_LOG = ROOT / "logs" / "audit.jsonl"
EMBEDDING_CACHE = ROOT / ".cache" / "embeddings.json"

GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "gemini-embedding-2"

TOP_K = 3
MAX_QUESTION_CHARS = 1000
