# Agentic AI

This note introduces agentic AI as the next systems layer above a large
language model.

An ordinary LLM interaction is usually:

1. user sends a prompt
2. model generates one response
3. interaction stops

An agentic system is different:

1. receive a goal
2. inspect the current state
3. decide on the next action
4. call a tool or produce a sub-result
5. update memory
6. repeat until done

That shift from one-shot generation to iterative control is the main idea.

![Agentic AI architecture](../assets/agentic_ai_architecture.svg)

---

## 1. What Makes a System "Agentic"?

An agent is not just "a smarter chatbot."

The usual ingredients are:

- **model**
  produces candidate actions, plans, or text
- **controller**
  decides when to call the model, when to invoke tools, and when to stop
- **tools**
  calculators, code execution, search, APIs, databases, files, or robotics
- **memory**
  short-term task state and sometimes long-term stored knowledge
- **environment**
  the external world the agent observes and acts on
- **evaluation**
  checks whether the task is complete or whether the result is acceptable

The important teaching point is that the LLM is one component inside a larger
software loop.

---

## 2. Core Control Loop

The simplest agent loop looks like this:

1. observe the current state
2. reason about the next step
3. choose an action
4. execute the action
5. record the result
6. continue or stop

This is often described as:

- observe
- plan
- act
- reflect

or:

- thought
- tool call
- observation
- next thought

That is why agent logs are useful for teaching. Students can see that the work
happens over several iterations rather than in one response.

---

## 3. Planning, Memory, and Tool Use

### 3.1 Planning

Planning means breaking a goal into smaller steps.

Examples:

- read a file, then summarize it
- search for information, then compare sources
- compute a value, then write the result to a report

Without planning, the system tends to produce plausible text but may not make
real progress on the task.

### 3.2 Memory

Memory appears in two main forms:

- **short-term memory**
  the current goal, tool outputs, and intermediate steps
- **long-term memory**
  saved notes, retrieved documents, prior interactions, or a vector index

Short-term memory is necessary even in tiny classroom demos. The agent must
remember what it has already tried.

### 3.3 Tool Use

Tool use is what turns a text generator into an action-taking system.

Examples of tools:

- calculator
- Python interpreter
- web search
- file system access
- database lookup
- code execution

This is one of the clearest engineering boundaries:

- an LLM alone predicts tokens
- an agent uses software tools to change or inspect the world

---

## 4. Why This Topic Fits an HPC Course

Agentic AI might sound like a product topic, but it is also a systems topic.

Reasons:

- an agent may invoke the model many times for a single task
- tool calls introduce latency and state management
- memory systems require storage and retrieval
- evaluation can require repeated simulation or code execution
- production agents need logging, monitoring, and fault handling

This means agentic systems raise familiar HPC questions:

- where is the bottleneck?
- what is the cost of repeated execution?
- which parts are embarrassingly parallel?
- which parts are sequential and limit speedup?

The model may run on GPUs, but the surrounding orchestration is a full
distributed-systems problem.

---

## 5. Common Failure Modes

Students should see that agents are powerful but brittle.

Typical failure modes:

- **bad plans**
  the system decomposes the problem incorrectly
- **tool misuse**
  wrong arguments, wrong tool, or wrong sequence of calls
- **memory drift**
  the agent forgets or misinterprets previous results
- **looping**
  repeated actions without real progress
- **hallucinated success**
  claims completion without verifying the outcome
- **unsafe actions**
  takes actions that should have required validation or sandboxing

This is why production agents need:

- permissions
- sandboxes
- validation
- retry logic
- human approval points

---

## 6. A Classroom-Friendly Mental Model

One useful analogy is:

- a plain LLM is like asking a smart student one question
- an agent is like giving that student a goal, a calculator, notes, a shell,
  and permission to take several steps before reporting back

That does not guarantee correctness. It only changes what the system is able
to attempt.

---

## 7. Toy Example Structure

The `toy_agent_demo.py` example in this directory shows a tiny version of the
architecture.

It includes:

- a controller loop
- a small knowledge base
- a calculator tool
- memory storage
- a printed execution trace

This is intentionally simple. The goal is to make the control flow visible.

Suggested live demo:

```bash
python3 toy_agent_demo.py --demo explain
python3 toy_agent_demo.py --demo efficiency
```

What to point out during the demo:

- the goal is not answered immediately
- the agent chooses a tool
- the tool output becomes an observation
- the observation changes the next action
- the final answer is assembled from intermediate state

For a more focused discussion of retrieval and tool-interface layers, see
`MCP_RAG_AGENT_SKILLS.md` and `mcp_rag_skills_demo.py`.

---

## 8. Discussion Prompts

- What part of the system is the "agent" and what part is just a tool?
- Why is repeated model invocation more expensive than one-shot prompting?
- Why does tool use create both capability and risk?
- What would you log if you needed to debug an agent?
- Which parts of an agent workflow are easy to parallelize and which are not?

---

## 9. Key Takeaway

Agentic AI is a control loop wrapped around a model.

The core ingredients are:

- planning
- memory
- tool use
- iteration
- stopping criteria

From an HPC perspective, agentic AI is interesting because the intelligence is
not just in the model weights. It is in the repeated interaction between the
model, the tools, the memory system, and the execution environment.
