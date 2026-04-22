# 08 Agentic AI

This lesson introduces agentic AI from a systems point of view.

The key idea is that an agent is not just a single model response. An agent is
a control loop that:

- observes the current state
- plans what to do next
- uses tools
- stores and retrieves memory
- decides when to stop

In this course, the point is not to build a production-grade assistant. The
point is to help students see how modern AI systems combine language models
with software-engineering and HPC ideas such as orchestration, state, repeated
execution, and external tools.

Learning goals:

- distinguish a plain LLM response from an agentic workflow
- understand the roles of planning, memory, and tool use
- trace a simple observe-plan-act loop
- connect agent execution to systems costs such as latency, logging, and
  repeated model calls
- recognize common failure modes and why guardrails matter

Files in this directory:

- `AGENTIC_AI.md`
  Longer teaching note with definitions, architecture, and discussion prompts.
- `MCP_RAG_AGENT_SKILLS.md`
  Focused note that separates MCP, RAG, and agent skills and shows how they
  fit together in one system.
- `toy_agent_demo.py`
  Small runnable example of a tool-using agent with a transparent execution
  trace. It uses only the Python standard library.
- `mcp_rag_skills_demo.py`
  Small runnable example showing RAG retrieval, an MCP-style tool registry,
  and skill selection.

Suggested teaching path:

1. start by contrasting a one-shot chatbot answer with an iterative agent loop
2. introduce the architecture diagram and the roles of memory and tools
3. run `toy_agent_demo.py` live so students can see state change over steps
4. discuss where the real complexity appears in production systems
5. connect the topic back to HPC through repeated model calls, orchestration,
   and evaluation cost

Quick demo commands:

```bash
python3 toy_agent_demo.py --demo explain
python3 toy_agent_demo.py --demo efficiency
python3 toy_agent_demo.py --goal "Explain tensor parallelism and store a short note."
python3 mcp_rag_skills_demo.py --demo rag
python3 mcp_rag_skills_demo.py --demo mcp
```

Teaching notes:

- keep the first example deterministic and inspectable
- emphasize that the "agent" is the loop plus tools, not just the model
- point out that memory can be as simple as a list or as complex as a vector DB
- tie the discussion back to resource use: more steps means more compute and
  more latency
- use `MCP_RAG_AGENT_SKILLS.md` when students need a clean separation between
  knowledge retrieval, tool interfaces, and reusable workflows

Key takeaway:

Agentic AI extends an LLM into a software system that can plan, use tools, and
act over multiple steps.
