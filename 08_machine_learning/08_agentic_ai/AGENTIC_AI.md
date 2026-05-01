# Agentic AI

**CSCI 394 -- Spring 2026**

This note introduces agentic AI as a systems layer around a large language
model. The goal is not to treat an agent as a magical chatbot. The goal is to
understand the software architecture that lets a model plan, use tools, inspect
results, update state, and continue working over multiple steps.

An ordinary language-model interaction is usually:

1. the user sends a prompt
2. the model generates one response
3. the interaction stops

An agentic system is different:

1. receive a goal
2. inspect the current state
3. decide the next action
4. call a tool, retrieve context, or produce a partial result
5. record the observation
6. decide whether to continue or stop

That shift from one-shot generation to iterative control is the main idea.

![Agentic AI architecture](../assets/agentic_ai_architecture.svg)

---

## 1. Learning Objectives

After this lecture, students should be able to:

- distinguish a one-shot model response from an agentic workflow
- identify the model, controller, tools, memory, environment, and evaluator in
  an agent system
- trace the observe-plan-act loop through a small example
- explain why agent systems are systems-engineering problems, not just model
  prompts
- connect agentic workflows to scientific computing tasks such as job
  submission, debugging, data movement, and experiment orchestration
- identify common failure modes and the guardrails needed to control them

---

## 2. What Makes a System Agentic?

An agent is not just "a smarter chatbot." In this course, an agentic system is a
software loop that uses a model to choose actions and then executes those actions
through tools or external systems.

The usual ingredients are:

- **model**
  produces candidate actions, plans, explanations, or text
- **controller**
  decides when to call the model, when to invoke tools, how to handle errors,
  and when to stop
- **tools**
  calculators, code execution, search, APIs, databases, file systems, job
  schedulers, or scientific software
- **memory**
  short-term task state and sometimes long-term stored knowledge
- **environment**
  the external system the agent observes and acts on
- **evaluation**
  checks whether the task is complete, correct, safe, or worth continuing

The important teaching point is that the language model is one component inside
a larger software system. Much of the engineering difficulty is in the loop
around the model.

---

## 3. One-Shot Prompting vs. Agentic Execution

| Feature | One-shot model call | Agentic system |
| ------- | ------------------- | -------------- |
| User input | prompt | goal or task |
| Model role | answer directly | choose and revise actions |
| External tools | usually none | central to the workflow |
| State | mostly prompt context | explicit memory and observations |
| Failure mode | wrong answer | wrong plan, wrong tool call, looping, unsafe action |
| Systems concern | inference latency | orchestration, retries, logging, permissions, cost |

One-shot prompting is still useful. Many tasks do not need an agent. The agentic
approach becomes useful when the task requires several steps, outside
information, tool execution, state tracking, or verification.

---

## 4. Core Control Loop

The simplest agent loop looks like this:

1. **observe**
   read the goal, current state, tool outputs, logs, files, or retrieved context
2. **plan**
   decide what step should happen next
3. **act**
   call a tool, query a database, submit a job, write code, or produce a result
4. **record**
   store the observation and update task state
5. **evaluate**
   decide whether the result is complete, incomplete, or failed
6. **continue or stop**
   either take another step or return the final answer

This is often summarized as:

```text
observe -> plan -> act -> observe -> ... -> final answer
```

or:

```text
thought -> tool call -> observation -> next thought
```

Agent logs are useful for teaching because they make the intermediate work
visible. Students can see that the system did not simply produce an answer. It
made a sequence of decisions.

---

## 5. Planning, Memory, and Tool Use

### 5.1 Planning

Planning means breaking a goal into smaller steps.

Examples:

- read a file, then summarize it
- search for documentation, then compare sources
- generate a job script, then validate scheduler directives
- run a simulation, then inspect output logs
- compute a metric, then write the result to a report

Without planning, a model may produce plausible text without making real
progress on the task.

Planning does not need to be elaborate. A useful plan can be a short ordered
list of tool calls and checks. For scientific computing, the plan should also
include validation points, such as checking a queue status, confirming output
files exist, or verifying that the right module environment is loaded.

### 5.2 Memory

Memory appears in two main forms:

- **short-term memory**
  the current goal, tool outputs, intermediate results, retries, and error logs
- **long-term memory**
  saved notes, retrieved documents, prior interactions, source code, or a
  searchable index

Short-term memory is necessary even in tiny classroom demos. The agent must
remember what it has already tried.

Long-term memory is useful when the task depends on information that does not
fit in the prompt or changes over time. Facility documentation, scheduler
policies, software-module instructions, and project notes are examples.

### 5.3 Tool Use

Tool use is what turns a text generator into an action-taking system.

Examples of tools:

- calculator
- Python interpreter
- file-system read and write operations
- web or document search
- database lookup
- job scheduler interface
- Globus transfer interface
- code execution and tests

This is one of the clearest engineering boundaries:

- a model alone predicts tokens
- an agent uses software tools to inspect or change the world

Tool use creates capability, but it also creates risk. A tool call can submit a
bad job, delete a file, move data to the wrong location, or spend money. For
that reason, serious agent systems need permissions, sandboxing, audit logs,
and human approval points.

---

## 6. Why This Topic Fits a Scientific Computing Course

Agentic AI might sound like a product topic, but it is also a systems topic.

Reasons:

- an agent may invoke the model many times for a single task
- tool calls introduce latency and state management
- memory systems require storage and retrieval
- evaluation can require repeated simulation, code execution, or log analysis
- production agents need logging, monitoring, fault handling, and rollback
- some steps are parallel, while other steps are sequential and limit speedup

This means agentic systems raise familiar HPC questions:

- where is the bottleneck?
- what is the cost of repeated execution?
- which parts are embarrassingly parallel?
- which parts are sequential and limit speedup?
- how do we schedule work across local tools, remote APIs, and compute systems?
- how do we make the workflow reproducible?

The model may run on GPUs, but the surrounding orchestration is a full
distributed-systems problem.

---

## 7. Agents as Coordinators for HPC Workflows

An agent can act as a coordinator between a researcher and a computing facility.
The researcher gives a high-level goal, while the agent turns that goal into a
sequence of operational steps.

Example researcher requests:

- "Run a density-functional-theory calculation for this structure."
- "Fine-tune this language model on the new data set."
- "Move the output from the scratch file system to a project endpoint."
- "Find why my job failed and suggest the next submission."

A useful HPC agent needs several layers:

| Layer | Responsibility |
| ----- | -------------- |
| User request | high-level scientific or computing goal |
| Agent controller | planning, tool selection, error handling |
| Knowledge source | facility documentation, software instructions, project notes |
| Job tools | submit jobs, query status, cancel jobs |
| Data tools | list files, transfer data, verify outputs |
| Compute resources | machines such as Aurora, Polaris, Frontier, or Perlmutter |

The language model helps interpret the goal and choose steps. The tools do the
actual work. The knowledge source keeps the agent grounded in current system
rules.

---

## 8. HPC Workflow Examples

### 8.1 Submit and Monitor a Job

Goal:

> Run a Python workload on a supercomputer and report whether it completed.

Possible agent workflow:

1. retrieve machine-specific documentation for job submission
2. choose the right scheduler directives
3. generate or adapt the submission script
4. submit the job through a controlled job tool
5. poll job status
6. inspect output and error logs
7. summarize the result for the researcher

The hard part is not only generating a script. The hard part is choosing the
right script for the right machine and then verifying that it actually ran.

### 8.2 Batch Experiments

Goal:

> Sweep four learning rates across three model sizes and compare validation
> loss.

Possible agent workflow:

1. create a small experiment matrix
2. generate one configuration per run
3. submit runs with resource limits
4. monitor completion and failures
5. collect metrics
6. produce a comparison table

This is a natural place for agents because the work is repetitive, structured,
and full of small operational details. It also needs guardrails because an
uncontrolled sweep can waste a large allocation.

### 8.3 Interactive Debugging

Goal:

> My distributed training job failed. Figure out why.

Possible agent workflow:

1. read scheduler output
2. inspect training logs
3. identify whether the failure is from the scheduler, module environment,
   data path, memory, networking, or the Python program
4. propose a minimal fix
5. ask for approval before resubmitting

This is useful because failures in HPC workflows often cross several layers:
queue policy, resource request, shell environment, distributed launcher,
Python package versions, file-system paths, and application code.

### 8.4 Data Movement

Goal:

> Move simulation outputs from the facility file system to a project storage
> endpoint and verify the transfer.

Possible agent workflow:

1. list the output directory
2. identify files to transfer
3. start a transfer through a data-movement tool
4. monitor transfer status
5. verify destination files
6. record provenance

For scientific workflows, data movement is part of the computation. An agent
that ignores file paths, endpoints, permissions, and provenance is not useful.

---

## 9. Autonomous Scientific Discovery

Agentic AI also appears in autonomous scientific discovery. In that setting, an
agent is not only answering questions. It is helping choose experiments,
simulation parameters, analysis steps, and follow-up hypotheses.

A high-level discovery loop looks like this:

1. define a scientific objective
2. propose candidate experiments or simulations
3. run selected experiments or simulations
4. analyze results
5. update the hypothesis or search strategy
6. choose the next experiment

Examples:

- materials discovery: propose structures, run simulations, analyze stability
- molecular design: generate candidates, evaluate properties, refine search
- climate or fluid simulation: vary parameters, compare diagnostics
- spectroscopy: simulate spectra, compare to observations, update parameters

The key systems point is that autonomous discovery is a closed loop around
expensive tools. Each step may consume compute time, allocation hours, lab
resources, or human attention. The agent must therefore balance exploration,
cost, uncertainty, and safety.

Autonomy should be bounded. A scientific agent may be allowed to propose and
rank experiments, but expensive or risky actions should require validation or
approval.

---

## 10. Where Retrieval, Tool Protocols, and Skills Fit

This lecture separates the general agent loop from three related ideas:

- **retrieval**
  gives the system access to relevant external knowledge at inference time
- **tool protocol**
  gives the system a standard way to discover and call tools or context sources
- **reusable workflow skill**
  packages a multi-step procedure so the agent can repeat it reliably

These are different layers.

| Layer | Question it answers |
| ----- | ------------------- |
| Retrieval | How does the system get relevant knowledge? |
| Tool protocol | How does the system talk to external capabilities? |
| Workflow skill | How does the system reuse a known procedure? |
| Agent controller | How does the system decide what to do next? |

For example, a job-submission assistant might retrieve current scheduler
documentation, call a job-submission tool through a standard interface, and use
a reusable "submit and monitor" workflow skill.

The companion note `MCP_RAG_AGENT_SKILLS.md` gives a more focused treatment of
these layers.

---

## 11. Failure Modes

Students should see that agents are powerful but brittle.

Typical failure modes:

- **bad plans**
  the system decomposes the problem incorrectly
- **tool misuse**
  the wrong tool, wrong arguments, or wrong sequence of calls
- **retrieval failure**
  missing, stale, or irrelevant context
- **memory drift**
  the agent forgets or misinterprets previous results
- **looping**
  repeated actions without real progress
- **hallucinated success**
  claims completion without checking the result
- **unsafe actions**
  takes actions that should have required validation, approval, or sandboxing
- **silent partial failure**
  one step fails, but later steps proceed as if it succeeded

This is why production agents need:

- permissions
- sandboxing
- validation
- retry logic
- structured logs
- human approval points
- explicit stopping criteria
- tests or checks for important outputs

---

## 12. Safety and Control

A useful way to design an agent is to classify actions by risk.

| Action type | Example | Control |
| ----------- | ------- | ------- |
| Read-only | search docs, list jobs, inspect logs | usually safe |
| Local write | create a script, edit a config | log and diff |
| Remote action | submit a job, start a transfer | require confirmation or policy |
| Destructive action | delete files, cancel jobs | require explicit approval |
| Expensive action | large sweep, many nodes, long run | enforce budgets |

For scientific computing, the agent should not be judged only by whether it can
act. It should also be judged by whether it acts within a controlled policy.

Good control mechanisms include:

- typed tool schemas
- allowlists for tools and paths
- budget limits
- dry-run modes
- job-size caps
- provenance records
- human-in-the-loop approval for expensive operations

---

## 13. Cost Model

Agentic AI turns inference into a workflow scheduling problem.

Costs can include:

- model calls
- retrieval calls
- tool execution
- queue wait time
- compute-node allocation
- data transfer time
- repeated retries
- human review time

Some steps can run in parallel. For example, an agent may retrieve several
documents or inspect multiple log files independently. Other steps are
sequential. A job cannot be analyzed until it has started, and a failed
submission may need to be fixed before another run is meaningful.

For HPC workflows, the dominant cost may not be the language-model call. It may
be the supercomputer job, the queue delay, or the cost of making the wrong
operational decision.

---

## 14. Classroom-Friendly Mental Model

One useful analogy is:

- a plain language model is like asking a smart student one question
- an agent is like giving that student a goal, notes, a calculator, a shell,
  and permission to take several steps before reporting back

That does not guarantee correctness. It only changes what the system is able to
attempt.

The agent still needs:

- good instructions
- reliable tools
- relevant context
- checks on intermediate results
- clear stopping criteria
- boundaries on what it is allowed to do

---

## 15. Toy Example Structure

The `toy_agent_demo.py` example in this directory shows a small version of the
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

For step-by-step lab instructions, see `TUTORIALS.md`.

---

## 16. Discussion Prompts

- What part of the system is the "agent" and what part is just a tool?
- Why is repeated model invocation more expensive than one-shot prompting?
- Why does tool use create both capability and risk?
- What would you log if you needed to debug an agent?
- Which parts of an agent workflow are easy to parallelize and which are not?
- What HPC actions should always require human approval?
- How would you prevent an experiment-sweep agent from wasting an allocation?
- What does it mean for an autonomous scientific-discovery system to be
  reproducible?

---

## 17. Key Takeaways

- Agentic AI is a control loop wrapped around a model.
- The core ingredients are planning, memory, tool use, iteration, and stopping
  criteria.
- The model proposes actions, but the surrounding software system controls
  execution.
- In scientific computing, agents are useful because workflows involve many
  operational steps across documentation, software, schedulers, data movement,
  logs, and analysis.
- The main risks are wrong plans, wrong tool calls, stale knowledge, looping,
  unverified success, and unsafe actions.
- From an HPC perspective, agentic AI is interesting because intelligence is not
  only in the model weights. It is in the repeated interaction between the
  model, tools, memory, policies, and execution environment.
