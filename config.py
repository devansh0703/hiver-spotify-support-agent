"""Central configuration for the Hiver Spotify support agent."""
import os
import re
import subprocess

from dotenv import load_dotenv

load_dotenv()


def _bashrc_exports(path: str | None = None) -> dict[str, str]:
    """Pull export VAR=val lines from ~/.bashrc (non-secret shell env bridging)."""
    path = path or os.path.expanduser("~/.bashrc")
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = re.match(r"^\s*export\s+([A-Z][A-Z0-9_]*)=(.+?)\s*$", line)
                if m:
                    val = m.group(2).strip().strip("'\"").strip()
                    out[m.group(1)] = val
    except OSError:
        pass
    return out


_bashrc = _bashrc_exports()
_NVIDIA_FROM_BASHRC = _NVIDIA_FROM_BASHRC = _bashrc.get("NVIDIA_API_KEY", "")

# ---- Brand ----
BRAND_HANDLE = os.getenv("BRAND_HANDLE", "SpotifyCares")
BRAND_NAME = os.getenv("BRAND_NAME", "Spotify")
RAW_CSV = os.getenv(
    "RAW_CSV", "/home/devansh/hiver/data/twcs/twcs.csv"
)
CASES_DIR = os.getenv("CASES_DIR", "/home/devansh/hiver/data/cases")
CASES_PATH = os.path.join(CASES_DIR, "spotify_cases.parquet")
MIN_REPLY_TOKENS = int(os.getenv("MIN_REPLY_TOKENS", "3"))
MAX_CASES = int(os.getenv("MAX_CASES", "16000"))

# ---- Banking77 (secondary dataset, intent few-shot only) ----
BANKING77_TRAIN = os.getenv("BANKING77_TRAIN", "/home/devansh/hiver/data/banking77/train.csv")
BANKING77_TEST = os.getenv("BANKING77_TEST", "/home/devansh/hiver/data/banking77/test.csv")

# ---- Embedding (local Ollama) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2:latest"
)
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
OLLAMA_BATCH = int(os.getenv("OLLAMA_BATCH", "32"))

# ---- Weaviate Cloud ----
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_CLOUD_MODEL = os.getenv(
    "WEAVIATE_CLOUD_MODEL", "Snowflake/snowflake-arctic-embed-l-v2.0"
)
COLLECTION = os.getenv("COLLECTION", "SpotifyCase")
MMR_CANDIDATES = int(os.getenv("MMR_CANDIDATES", "40"))
MMR_RESULTS = int(os.getenv("MMR_RESULTS", "12"))
MMR_BALANCE = float(os.getenv("MMR_BALANCE", "0.65"))
RERANK_TOP = int(os.getenv("RERANK_TOP", "5"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# ---- Jina rerank (direct API, fallback / explicit) ----
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_RERANK_URL = os.getenv(
    "JINA_RERANK_URL", "https://api.jina.ai/v1/rerank"
)
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v3")

# ---- NVIDIA LLM ----
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or _NVIDIA_FROM_BASHRC
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
AGENT_MODEL = os.getenv(
    "AGENT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
)
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b"
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# global minimum seconds between NVIDIA NIM requests (avoids RPM stampedes)
MIN_REQUEST_SPACING = float(os.getenv("MIN_REQUEST_SPACING", "1.5"))
EVAL_JUDGE_WORKERS = int(os.getenv("EVAL_JUDGE_WORKERS", "4"))

# ---- Eval ----
GOLDEN_PATH = os.getenv("GOLDEN_PATH", "/home/devansh/hiver/eval/golden_set.jsonl")
EVAL_OUT = os.getenv("EVAL_OUT", "/home/devansh/hiver/eval/results")
HUMAN_JUDGED_PATH = os.getenv("HUMAN_JUDGED_PATH", "/home/devansh/hiver/eval/human_judged.jsonl")

# ---- Intents (derived from data; see REPORT.md) ----
INTENTS = [
    "technical_issue",
    "account_admin",
    "billing_subscription",
    "account_access",
    "content_library",
    "device_integration",
    "feature_request_feedback",
    "closing_acknowledgement",
    "other_offtopic",
]

INTENT_DESCRIPTIONS = {
    "technical_issue": (
        "A bug, crash, playback failure, offline/connectivity problem, download failure, "
        "freeze, battery drain, or other malfunction in the Spotify app or web player."
    ),
    "account_admin": (
        "Administrative account action: merge accounts, change email/username/country, "
        "delete account, artist page ownership, privacy settings, connect/premium-linked account changes."
    ),
    "billing_subscription": (
        "Anything about money: subscription plans, being charged, refunds, discounts, promos, "
        "student/family/military plans, renewal, cancellation of a paid plan, gift cards."
    ),
    "account_access": (
        "Cannot get into the account: login/sign-in errors, forgotten password, account locked "
        "or hacked, logged-out unexpectedly, sign-in loop, device limit login issues."
    ),
    "content_library": (
        "Content on Spotify: missing/unavailable songs, albums or artists, greyed-out tracks, "
        "region/content availability, playlists, library, local files, radio, search results."
    ),
    "device_integration": (
        "Using Spotify on a device or via integration: car, TV, speakers/sonos, smart watch, "
        "browser/desktop client differences, Chromecast, connecting or controlling other devices."
    ),
    "feature_request_feedback": (
        "Asking for a new feature, suggesting an improvement, giving opinion/feedback/praise, "
        "or voting for a feature. No immediate product malfunction."
    ),
    "closing_acknowledgement": (
        "A gratitude/closing turn in a conversation: thanks, 'done', confirmation that a fix "
        "worked, farewell/goodbye. Contains no new issue."
    ),
    "other_offtopic": (
        "Spam, off-topic chatter, memes, abusive/troll content, or anything that does not fit "
        "the other intents."
    ),
}

# ---- Escalation rules ----
# These signals (in the raw/lowercased message) force escalation regardless of intent.
ESCALATION_KEYWORDS = [
    "lawsuit",
    "attorney",
    "lawyer",
    "legal",
    "police",
    "sue",
    "sued",
    "deceased",
    "refund my money",
    "class action",
    "chargeback",
    "billing dispute",
    "identity",
    "ssn",
    "social security",
]