"""IMMUTABLE SMOKE — the ONLY model-endpoint/agent verification for hiver.

Session truth, established by dozens of reprobes: the system python3
(/usr/bin/python3, user site ~/.local/lib/python3.14/site-packages) carries
the COHERENT trio  core 0.12.52.post1 / agent-openai 0.4.12 / llms-openai
0.4.7. The .venv file is where all "missing module/conflict" ghosts come from
(it unregisters user-site default path) — NEVER activate it again.

Run:  /usr/bin/python3 scripts/agent_probe.py
"""
import config
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.tools import FunctionTool
from src.llm import make_nvidia_llm

llm = make_nvidia_llm(max_tokens=200)


def _tool_resolve(case_id: str) -> str:
    """Resolve a known escalation path by id (tool used by the agent)."""
    return f"resolved:{case_id}"

tool = FunctionTool.from_defaults(
    fn=_tool_resolve,
    name="resolve_case",
    description="look up a resolved escalation path by case id",
)

worker = FunctionCallingAgentWorker.from_tools([tool], llm=llm)
runner = AgentRunner(worker)
reply = runner.chat(
    "A listener's playback keeps buffering. Query resolve_case for the "
    "buffering fix and reply in one short line."
)
print("AGENT_STACK_OK:", str(reply).strip()[:90])
