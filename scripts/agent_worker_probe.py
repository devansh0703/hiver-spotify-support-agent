"""Smoke: NVIDIA NIM endpoint as a LlamaIndex function-calling agent worker."""
import os
import sys

sys.path.insert(0, "/home/devansh/hiver")

import config

os.environ.setdefault("NVIDIA_API_KEY", config.NVIDIA_API_KEY)

from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.tools import FunctionTool
from src.llm import make_nvidia_llm


def _ground_test(message: str) -> str:
    """Return a deterministic grounding tag (used to prove the agent tool loop connects)."""
    return f"grounded-tag-{len(message)}"


tool = FunctionTool.from_defaults(
    fn=_ground_test,
    name="ground_test",
    description="returns a grounding tag derived from a message",
)

llm = make_nvidia_llm(max_tokens=160)
Settings.llm = llm

worker = FunctionCallingAgentWorker.from_tools([tool], llm=llm)
runner = AgentRunner(worker)

out = runner.chat("Tag message: my playback keeps pausing")
print("AGENT_OK:", str(out).strip()[:90])