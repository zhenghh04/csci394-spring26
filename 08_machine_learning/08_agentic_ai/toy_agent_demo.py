"""
Toy agent demo for classroom use.

This script intentionally avoids external APIs. It demonstrates the structure
of an agent with:

- a goal
- a planner/controller
- simple tools
- memory
- a multi-step execution trace

Example commands:

    python3 toy_agent_demo.py --demo explain
    python3 toy_agent_demo.py --demo efficiency
    python3 toy_agent_demo.py --goal "Explain tensor parallelism and store a short note."
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field


KNOWLEDGE_BASE = {
    "agentic ai": (
        "Agentic AI combines a model with planning, memory, tools, and an "
        "iterative control loop."
    ),
    "data parallelism": (
        "Data parallelism replicates the model and splits the training data "
        "across ranks. Gradients are synchronized after backward passes."
    ),
    "tensor parallelism": (
        "Tensor parallelism splits the math inside a layer across multiple "
        "accelerators. It is usually kept within a high-bandwidth group "
        "because communication happens during layer execution."
    ),
    "pipeline parallelism": (
        "Pipeline parallelism splits model depth across stages and sends "
        "activations between neighboring stages."
    ),
    "parallel efficiency": (
        "Parallel efficiency is speedup divided by the number of processors "
        "or GPUs. Efficiency = speedup / p."
    ),
}


SAFE_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}


def safe_eval(expression: str) -> float:
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINOPS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return SAFE_BINOPS[type(node.op)](left, right)
        raise ValueError(f"Unsupported expression: {expression}")

    tree = ast.parse(expression, mode="eval")
    return float(eval_node(tree))


def search_kb(query: str) -> str:
    query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
    best_key = None
    best_score = -1
    for key in KNOWLEDGE_BASE:
        key_words = set(re.findall(r"[a-zA-Z]+", key))
        score = len(query_words & key_words)
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is None or best_score == 0:
        return "No relevant note found."
    return f"{best_key}: {KNOWLEDGE_BASE[best_key]}"


def extract_numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def extract_speedup_and_processors(text: str) -> tuple[float | None, float | None]:
    speedup_match = re.search(r"speedup\s+(\d+(?:\.\d+)?)", text.lower())
    gpu_match = re.search(r"(\d+(?:\.\d+)?)\s*gpus?", text.lower())

    speedup = float(speedup_match.group(1)) if speedup_match else None
    processors = float(gpu_match.group(1)) if gpu_match else None

    if speedup is None or processors is None:
        numbers = extract_numbers(text)
        if len(numbers) >= 2:
            processors = numbers[0]
            speedup = numbers[1]

    return speedup, processors


@dataclass
class AgentState:
    goal: str
    observations: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    search_result: str | None = None
    numeric_result: float | None = None
    finished: bool = False


class ToyAgent:
    def __init__(self, trace: bool = True):
        self.trace = trace

    def log(self, message: str):
        if self.trace:
            print(message)

    def tool_search(self, state: AgentState, query: str):
        result = search_kb(query)
        state.search_result = result
        state.observations.append(result)
        self.log(f"TOOL search_kb({query!r})")
        self.log(f"OBS  {result}")

    def tool_calculator(self, state: AgentState, expression: str):
        value = safe_eval(expression)
        state.numeric_result = value
        result = f"{expression} = {value:.4f}"
        state.observations.append(result)
        self.log(f"TOOL calculator({expression!r})")
        self.log(f"OBS  {result}")

    def tool_remember(self, state: AgentState, note: str):
        state.memory.append(note)
        self.log(f"TOOL remember({note!r})")
        self.log(f"OBS  memory now has {len(state.memory)} item(s)")

    def decide_next_action(self, state: AgentState):
        goal = state.goal.lower()

        if "efficiency" in goal and state.search_result is None:
            return ("search", "parallel efficiency")

        if "efficiency" in goal and state.numeric_result is None:
            speedup, processors = extract_speedup_and_processors(goal)
            if speedup is not None and processors is not None:
                return ("calculate", f"{speedup} / {processors}")

        if any(word in goal for word in ["explain", "what is", "why", "compare"]) and state.search_result is None:
            query = "tensor parallelism" if "tensor" in goal else goal
            return ("search", query)

        if ("store" in goal or "save" in goal or "note" in goal) and not state.memory:
            note_parts = []
            if state.search_result:
                note_parts.append(state.search_result)
            if state.numeric_result is not None:
                note_parts.append(f"computed value = {state.numeric_result:.4f}")
            return ("remember", " | ".join(note_parts) if note_parts else "task completed")

        return ("finish", "")

    def finish(self, state: AgentState) -> str:
        pieces = []
        if state.search_result:
            pieces.append(state.search_result)
        if state.numeric_result is not None:
            pieces.append(
                f"Parallel efficiency = {state.numeric_result:.4f} "
                f"({100.0 * state.numeric_result:.2f}%)."
            )
        if state.memory:
            pieces.append(f"Saved note: {state.memory[-1]}")
        if not pieces:
            pieces.append("No useful result was produced.")
        state.finished = True
        return " ".join(pieces)

    def run(self, goal: str, max_steps: int = 6) -> str:
        state = AgentState(goal=goal)
        self.log(f"GOAL {goal}")

        for step in range(1, max_steps + 1):
            action, payload = self.decide_next_action(state)
            self.log(f"STEP {step} action={action}")

            if action == "search":
                self.tool_search(state, payload)
            elif action == "calculate":
                self.tool_calculator(state, payload)
            elif action == "remember":
                self.tool_remember(state, payload)
            elif action == "finish":
                answer = self.finish(state)
                self.log("DONE")
                return answer
            else:
                raise RuntimeError(f"Unknown action: {action}")

        return self.finish(state)


DEMO_GOALS = {
    "explain": "Explain tensor parallelism and why it is usually kept within a node.",
    "efficiency": "If 8 GPUs give speedup 6.2, what is the parallel efficiency? Save a short note.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=sorted(DEMO_GOALS.keys()))
    parser.add_argument("--goal", type=str)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    if not args.demo and not args.goal:
        parser.error("provide --demo or --goal")

    goal = args.goal if args.goal else DEMO_GOALS[args.demo]
    agent = ToyAgent(trace=not args.no_trace)
    answer = agent.run(goal)
    print()
    print("FINAL ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
