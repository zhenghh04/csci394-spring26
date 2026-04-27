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
- `TUTORIALS.md`
  Step-by-step student activities for running the demos, tracing the control
  loop, and extending an MCP-style tool server.
- `toy_agent_demo.py`
  Small runnable example of a tool-using agent with a transparent execution
  trace. It uses only the Python standard library.
- `mcp_rag_skills_demo.py`
  Small runnable example showing RAG retrieval, an MCP-style tool registry,
  and skill selection.
- `mcp_stdio_server.py`
  Minimal MCP-style JSON-RPC server over stdin/stdout. It is intentionally a
  teaching subset, not a complete MCP implementation.
- `mcp_stdio_client.py`
  Client that launches the local stdio server, lists tools, and calls them.

Suggested teaching path for one class:

1. Contrast a one-shot chatbot answer with an iterative agent loop.
2. Introduce the architecture diagram and the roles of memory and tools.
3. Run `toy_agent_demo.py` live so students can see state change over steps.
4. Run `mcp_rag_skills_demo.py` to separate retrieval, tool interfaces, and
   reusable workflows.
5. Run `mcp_stdio_client.py` to show the shape of MCP-style tool discovery and
   tool calls over a simple stdio transport.
6. Discuss where the real complexity appears in production systems: trust,
   permissions, latency, logging, retries, and evaluation.
7. Connect the topic back to HPC through repeated model calls, orchestration,
   external services, and the cost of evaluating multi-step workflows.

Quick demo commands:

```bash
cd CSCI394/csci394-spring26/08_machine_learning/08_agentic_ai

python3 toy_agent_demo.py --demo explain
python3 toy_agent_demo.py --demo efficiency
python3 toy_agent_demo.py --goal "Explain tensor parallelism and store a short note."

python3 mcp_rag_skills_demo.py --demo rag
python3 mcp_rag_skills_demo.py --demo mcp
python3 mcp_rag_skills_demo.py --demo skills

python3 mcp_stdio_client.py
python3 mcp_stdio_client.py --query "How is RAG different from MCP?"
python3 mcp_stdio_client.py --speedup 6.2 --workers 8
```

Teaching notes:

- keep the first example deterministic and inspectable
- emphasize that the "agent" is the loop plus tools, not just the model
- point out that memory can be as simple as a list or as complex as a vector DB
- tie the discussion back to resource use: more steps means more compute and
  more latency
- use `MCP_RAG_AGENT_SKILLS.md` when students need a clean separation between
  knowledge retrieval, tool interfaces, and reusable workflows
- use `TUTORIALS.md` for a lab session or homework-style walkthrough

Key takeaway:

Agentic AI extends an LLM into a software system that can plan, use tools, and
act over multiple steps.
