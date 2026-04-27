# MCP, RAG, and Agent Skills

This note explains three ideas that are often mentioned together in modern
agent systems:

- **MCP**
- **RAG**
- **agent skills**

They are related, but they are not the same thing.

Short version:

- **RAG** helps an agent get better information
- **MCP** helps an agent access tools and context in a standard way
- **skills** help an agent reuse a higher-level workflow

---

## 1. Why Students Mix These Up

All three concepts appear in tool-using LLM systems, so it is easy to blur
them together.

But they live at different layers:

| Concept | Main question it answers |
| --- | --- |
| RAG | How does the system get relevant knowledge? |
| MCP | How does the model or agent talk to external tools or context providers? |
| Agent skill | How does the system package a reusable way of solving a class of tasks? |

That is the most useful first distinction.

---

## 2. RAG: Retrieval-Augmented Generation

RAG stands for **retrieval-augmented generation**.

Basic idea:

1. user asks a question
2. system retrieves relevant documents or chunks
3. those retrieved materials are added to the model context
4. the model answers using both the prompt and the retrieved material

The goal is not to change the model weights. The goal is to improve the prompt
context at inference time.

### 2.1 Why Use RAG?

RAG is useful when:

- the needed information lives in a document collection
- the facts change more often than model retraining is practical
- the answer should be grounded in specific sources
- the base model does not reliably remember the needed detail

Examples:

- course notes
- internal documentation
- scientific papers
- product manuals
- code repositories

### 2.2 RAG Pipeline

A simple RAG pipeline looks like this:

1. split documents into chunks
2. build embeddings for each chunk
3. store them in a vector index
4. embed the user query
5. retrieve top-k relevant chunks
6. insert those chunks into the model prompt
7. generate the answer

### 2.3 What RAG Is Not

RAG is not:

- fine-tuning
- tool execution by itself
- a planning system by itself

RAG is a **knowledge access pattern**.

---

## 3. MCP: Model Context Protocol

MCP stands for **Model Context Protocol**.

The important idea for teaching is not the exact wire format. The important
idea is that MCP gives a model or agent a standardized way to interact with
external capabilities.

At a high level, MCP separates two roles:

- **client**
  the model application or agent runtime that wants context and tools
- **server**
  a process or service that exposes capabilities to the client

The common capabilities are:

- **tools**
  callable actions such as search, file lookup, database queries, or
  calculations
- **resources**
  context objects the client can read, such as documents or files
- **prompts**
  reusable prompt templates or workflows

The official MCP protocol uses JSON-RPC messages. Standard transports include
stdio and Streamable HTTP. In the stdio version, the client launches the server
as a subprocess and exchanges newline-delimited JSON messages over standard
input and standard output.

Examples of capabilities exposed through an MCP-style interface:

- file readers
- code execution
- search tools
- databases
- document stores
- application connectors

### 3.1 Why MCP Matters

Without a common interface, every tool integration becomes custom glue code.

With a common interface, tools can be exposed in a more uniform way:

- what tools exist
- what each tool does
- what inputs it expects
- what outputs it returns

That makes agent systems easier to extend and reason about.

### 3.2 A Good Mental Model

In classroom language:

- **RAG** is about retrieving knowledge
- **MCP** is about exposing tools and context providers in a standard way

So a retrieval system might itself be accessed through MCP.

That means:

- RAG and MCP are complementary
- one does not replace the other

### 3.3 Why This Is a Systems Topic

From an HPC or systems perspective, MCP raises questions like:

- how many external calls does the agent make?
- what is the latency of each call?
- what data is transferred?
- how are permissions and safety enforced?
- how is tool usage logged and debugged?

So even though the acronym sounds application-level, the engineering issues are
very much systems issues.

### 3.4 Course Demos

This directory uses two levels of MCP teaching examples:

1. `mcp_rag_skills_demo.py`
   A conceptual demo. It shows a registry of tools and how an agent might
   choose between retrieval, tool discovery, and a reusable skill.
2. `mcp_stdio_server.py` and `mcp_stdio_client.py`
   A protocol-shaped demo. It shows JSON-RPC requests over stdin/stdout with
   `tools/list` and `tools/call`.

The second demo is still intentionally small. It does not implement the full
MCP lifecycle, authentication, capability negotiation, or all content types.
Its purpose is to make the client/server boundary visible in a few lines of
standard-library Python.

---

## 4. Agent Skills

A skill is a reusable task pattern or workflow.

Examples:

- summarize a PDF
- debug a failing test
- answer questions over a document set
- prepare a meeting note
- query a benchmark report and compare runs

A skill usually includes some combination of:

- instructions
- routing logic
- tool preferences
- expected workflow steps
- output format expectations

### 4.1 Why Skills Exist

Skills reduce repeated reasoning overhead.

Instead of deciding everything from scratch on every task, the agent can reuse
a prepared pattern.

Benefits:

- more consistent behavior
- less prompt engineering repeated every time
- easier teaching and evaluation
- clearer boundaries for what the agent should do

### 4.2 Skills Are Higher-Level Than Tools

A tool is usually one action:

- search a document store
- read a file
- run a command

A skill is a structured way of combining actions to solve a class of problems.

So:

- **tool** = primitive operation
- **skill** = reusable strategy built on top of tools

---

## 5. Putting Them Together

These three concepts often appear in the same system.

Example:

1. user asks a question about course notes
2. the agent chooses a "document QA" skill
3. the skill uses a retrieval tool to fetch relevant notes
4. that retrieval service is exposed through MCP
5. the model answers using the retrieved context

Interpretation:

- the **skill** decided the workflow
- **MCP** exposed the available capability
- **RAG** provided the relevant knowledge

That layered view is the key teaching point.

---

## 6. One Concrete Example

Suppose the user asks:

> Explain tensor parallelism using our lecture materials.

A possible system behavior is:

1. choose the "course-note explainer" skill
2. call a retrieval tool to search lecture notes
3. retrieve the chunk that describes tensor parallelism
4. insert that chunk into the context
5. generate a grounded answer

If the retrieval tool is exposed via MCP, then the skill does not need to know
custom code for that connector. It just calls the tool through the shared
interface.

---

## 7. Comparison Table

| Feature | RAG | MCP | Agent skill |
| --- | --- | --- | --- |
| Main role | retrieve useful information | standardize access to tools/context | package a reusable workflow |
| Typical output | document chunks or context | structured tool results | completed task pattern |
| Changes model weights? | no | no | no |
| Helps with grounding? | yes | indirectly | sometimes |
| Helps with tool access? | sometimes | yes | yes, at workflow level |
| Helps with task reuse? | a little | a little | yes |

---

## 8. Failure Modes

### 8.1 RAG Failure Modes

- bad chunking
- wrong retrieval ranking
- stale documents
- too much irrelevant context

### 8.2 MCP Failure Modes

- wrong tool schema
- bad input validation
- permission mistakes
- slow or unreliable connectors

### 8.3 Skill Failure Modes

- wrong skill chosen for the task
- brittle instructions
- unnecessary steps
- hidden assumptions about available tools

These failures happen at different layers, which is another reason to keep the
concepts separate.

---

## 9. Suggested Teaching Path

1. Start with RAG because students already understand search and documents.
2. Introduce MCP as the interface layer for tools and context providers.
3. Introduce skills as reusable workflows that sit above tools.
4. Show how one agent request may use all three.
5. Run the stdio demo so students see tool discovery and tool calls as
   protocol messages.
6. Emphasize that none of these replaces the underlying model.

---

## 10. Board-Friendly Summary

- RAG retrieves relevant knowledge at inference time.
- MCP standardizes how tools and context sources are exposed.
- Skills package reusable multi-step workflows.
- A real agent may use all three in one task.
- These ideas live at different layers of the same system.

---

## 11. Discussion Prompts

- If an answer is wrong, which layer failed: retrieval, tool interface, skill,
  or model reasoning?
- Why is RAG often preferred over fine-tuning for fast-changing documents?
- Why does a shared protocol like MCP help large agent systems?
- When should a workflow be turned into a reusable skill instead of planned
  from scratch each time?

---

## 12. Key Takeaway

RAG, MCP, and agent skills are complementary.

- **RAG** gives the system better knowledge
- **MCP** gives the system a cleaner way to access external capabilities
- **skills** give the system reusable strategies

When students keep those layers separate, agent architectures become much
easier to understand.

References:

- Model Context Protocol specification: https://modelcontextprotocol.io/specification
- MCP transports: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
