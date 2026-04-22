"""
Classroom demo for MCP, RAG, and agent skills.

This script simulates three layers:

- RAG: retrieve relevant document chunks from a tiny course-note collection
- MCP: expose tools through a simple registry with names and descriptions
- skills: select a reusable workflow based on the goal

It uses only the Python standard library.

Example commands:

    python3 mcp_rag_skills_demo.py --demo rag
    python3 mcp_rag_skills_demo.py --demo mcp
    python3 mcp_rag_skills_demo.py --demo skills
    python3 mcp_rag_skills_demo.py --goal "Explain tensor parallelism from the notes."
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass


DOCUMENTS = {
    "ddp_note": (
        "Data parallelism replicates the model on each rank and splits the "
        "data across processes. Gradients are synchronized with all-reduce."
    ),
    "tensor_note": (
        "Tensor parallelism splits the math inside a layer across GPUs. It is "
        "usually kept within a high-bandwidth group because communication "
        "happens during layer execution."
    ),
    "pipeline_note": (
        "Pipeline parallelism splits model depth across stages and moves "
        "activations between neighboring stages."
    ),
    "rag_note": (
        "RAG retrieves relevant document chunks and places them into the model "
        "context so the answer is grounded in external knowledge."
    ),
    "mcp_note": (
        "MCP provides a standard interface for exposing tools and context "
        "providers to models and agents."
    ),
    "skill_note": (
        "An agent skill is a reusable workflow that combines instructions, "
        "tool choices, and expected output structure for a class of tasks."
    ),
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def bigrams(tokens: list[str]) -> set[str]:
    return {" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)}


def score_overlap(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(text)
    q_counts = {tok: q_tokens.count(tok) for tok in set(q_tokens)}
    d_counts = {tok: d_tokens.count(tok) for tok in set(d_tokens)}

    dot = sum(q_counts.get(tok, 0) * d_counts.get(tok, 0) for tok in set(q_counts) | set(d_counts))
    q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
    d_norm = math.sqrt(sum(v * v for v in d_counts.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    token_score = dot / (q_norm * d_norm)

    q_bigrams = bigrams(q_tokens)
    d_bigrams = bigrams(d_tokens)
    bigram_overlap = len(q_bigrams & d_bigrams)
    phrase_bonus = 0.25 * bigram_overlap

    return token_score + phrase_bonus


def rag_retrieve(query: str, top_k: int = 2) -> list[tuple[str, str, float]]:
    scored = []
    for doc_id, text in DOCUMENTS.items():
        scored.append((doc_id, text, score_overlap(query, text)))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


@dataclass
class MCPTool:
    name: str
    description: str


class MCPRegistry:
    def __init__(self):
        self.tools = {
            "retrieve_notes": MCPTool(
                name="retrieve_notes",
                description="Search the course-note collection and return relevant chunks.",
            ),
            "list_tools": MCPTool(
                name="list_tools",
                description="List available tools and their descriptions.",
            ),
        }

    def list_tools(self) -> list[MCPTool]:
        return list(self.tools.values())

    def call(self, name: str, **kwargs):
        if name == "retrieve_notes":
            return rag_retrieve(kwargs["query"], kwargs.get("top_k", 2))
        if name == "list_tools":
            return self.list_tools()
        raise ValueError(f"Unknown MCP tool: {name}")


def choose_skill(goal: str) -> str:
    goal = goal.lower()
    if any(word in goal for word in ["explain", "notes", "lecture", "parallelism"]):
        return "course_note_qa"
    if "mcp" in goal or "tool" in goal:
        return "tool_discovery"
    return "general_explainer"


def format_answer(goal: str, retrieved: list[tuple[str, str, float]]) -> str:
    if not retrieved:
        return "No relevant course-note chunks were retrieved."

    best_doc, best_text, _ = retrieved[0]
    if "tensor parallelism" in goal.lower():
        return (
            "Using retrieved notes: "
            f"{best_doc} says that {best_text}"
        )
    return f"Using retrieved notes: {best_text}"


def run_demo(goal: str):
    registry = MCPRegistry()

    print(f"GOAL {goal}")
    skill = choose_skill(goal)
    print(f"SKILL selected = {skill}")

    print("MCP available tools:")
    for tool in registry.call("list_tools"):
        print(f"  - {tool.name}: {tool.description}")

    if skill == "tool_discovery":
        print()
        print("FINAL ANSWER")
        print("MCP exposes tools in a standard way so the agent can discover and call them.")
        return

    print("RAG retrieval:")
    retrieved = registry.call("retrieve_notes", query=goal, top_k=2)
    for doc_id, text, score in retrieved:
        print(f"  - {doc_id} score={score:.3f}")
        print(f"    {text}")

    answer = format_answer(goal, retrieved)
    print()
    print("FINAL ANSWER")
    print(answer)


DEMO_GOALS = {
    "rag": "Explain tensor parallelism from the lecture notes.",
    "mcp": "What tools are available through MCP?",
    "skills": "Explain tensor parallelism from the lecture notes.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=sorted(DEMO_GOALS))
    parser.add_argument("--goal", type=str)
    args = parser.parse_args()

    if not args.demo and not args.goal:
        parser.error("provide --demo or --goal")

    goal = args.goal if args.goal else DEMO_GOALS[args.demo]
    run_demo(goal)


if __name__ == "__main__":
    main()
