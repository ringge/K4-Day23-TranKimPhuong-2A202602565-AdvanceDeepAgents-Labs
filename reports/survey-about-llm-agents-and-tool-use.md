# LLM Agents and Tool Use: Architectures, Evaluation, and Safety

## TL;DR
- Tool-using agents range from retrieval-mediated assistance to autonomous systems that decide when and how to invoke tools; prominent designs interleave reasoning and action or separate planning from execution. [1][2]
- Tool-use quality is not just final-answer accuracy: performance depends on tool retrieval, selection, call formatting, parameter correctness, and multi-step planning. [3]
- Benchmarks improve reproducibility by controlling unstable APIs, but simulated tools and automatic judges introduce their own validity concerns. [4][5]
- Tool responses and external content create security risks; current evidence points to persistent gaps in prompt-injection robustness and safe invocation. [6][7][8]
- Newer oversight work frames long-horizon monitoring around evidence of what agents did and consequential decisions, although the available summary does not establish comparative effectiveness. [9]

## Background
An LLM agent uses model outputs to select actions in an environment, often invoking APIs or other tools and incorporating returned observations into subsequent decisions. This differs from passive retrieval: the agent can autonomously choose and trigger tools, while tool-aware workflows require descriptions of available tools and a mechanism for translating model signals into invocations. [1] This matters because errors can arise at several points in the action loop, and actions can have consequences beyond an incorrect answer. [3][6]

## Interaction architectures: interleaving versus separation
A common pattern interleaves reasoning with action: the model reasons, calls a tool, observes the result, then continues. ReAct is a prominent example; its HF paper record characterizes the approach as integrating reasoning and action generation for decision-making. [1][2] A broader survey discusses tool invocation mechanisms including detecting generated call signals, completing a reasoning-and-acting cycle at each step, and confidence-based invocation. [1] 

Other designs explicitly plan before acting, decomposing tasks, selecting tools, and ordering calls; planning may rely on internal model reasoning or external reasoning tools. [1] Search-based planning can also explore action sequences with tree-search methods. These are different tradeoffs in control structure rather than evidence of a universally best architecture: the reviewed material does not provide a standardized quantitative comparison. [1] Tool use also differs from RAG-like retrieval, where information is retrieved for a query rather than chosen as an autonomous action. [1]

## Tool learning, retrieval, and execution are distinct capabilities
Tool competence is multi-stage. An agent may need to decide whether invocation is needed, retrieve or identify an appropriate tool, and then produce a valid call with correct arguments. [1][3] API-Bank evaluates planning, API retrieval, and API calling as distinct capabilities, and its reported error analysis includes API-name mismatch, parameter problems, absent or hallucinated calls, and malformed formats. [3]

Approaches target different bottlenecks. Tool-learning frameworks encompass tool discovery, invocation timing, and use; one survey describes Toolformer as learning API calls through augmented training examples and retaining calls when their outcomes improve predictive loss. [1] ToolLLM is described as searching a hierarchical API catalog, but the survey notes drawbacks in stability and prioritization among branches. [1] These methods address different pieces of the pipeline, so success at selecting a tool does not imply reliable argument generation or task completion. [1][3]

## Evaluation: task success, trajectories, and measurement stability
Benchmarks that score only the final response may hide failures in intermediate actions. API-Bank makes planning, API retrieval, and calling explicit and reports distinct call and parameter errors. [3] More generally, evaluation surveys identify invocation, selection, and retrieval as separate dimensions; robustness can be probed with paraphrases, misleading context, typos, and induced tool failures. [1]

Benchmark stability is itself a methodological issue. StableToolBench addresses failures caused by changing or unavailable online APIs through cached and simulated APIs, and proposes solvable pass and win rates using GPT-4 evaluation. These design choices make stability of API access and evaluation part of benchmark methodology; the notes do not establish how faithfully simulated APIs reproduce real-world tool behavior. [4] AgentRewardBench likewise focuses on judging web-agent trajectories; its HF summary warns that rule-based assessment may underreport success, underscoring that evaluation method can change measured performance. [5] These sources motivate reporting both task outcomes and intermediate behavior, alongside the limitations of the environment and evaluator. [4][5]

## Safety and control in the action loop
Tool use expands the threat surface because untrusted pages, documents, or tool responses can carry instructions that influence subsequent actions. A prompt-injection study organizes defenses by intervention stage and reports a tradeoff among trustworthiness, utility, and latency across its tested defenses; it also describes cases where checking whether an action is allowed does not ensure the reasoning leading to it remained intact. [6] Agent-SafetyBench evaluates tool-capable agents across interactive safety cases and reports that none of the tested agents exceeded 60% overall safety, a benchmark-specific result rather than a universal deployment estimate. [7]

Practical controls discussed in the sources include step-level guardrails and feedback for unsafe invocations, plus trust boundaries, permission checks, sandboxing, and auditing high-risk actions. [8] The HF summary for ToolSafe reports improved safety and task performance under adversarial conditions, but the retrieved notes do not provide enough setup detail to assess those results; operational recommendations in an Internet-Draft are proposals rather than validated guarantees. [8] The evidence therefore supports layered controls as an area of active work, not a claim that any single mitigation solves agent safety. [6][8]

## Trends and open problems
The recent evidence shifts attention from clean task completion toward robustness, oversight, and the integrity of long action sequences. The prompt-injection study reports that its tested defenses do not simultaneously deliver high trustworthiness, utility, and low latency; a broader survey also characterizes long-horizon and stateful risks as underrepresented in current evaluation. [6] HF-daily coverage includes an oversight benchmark focused on aligning requirements with observed behavior and identifying consequential autonomous decisions, but the available record is truncated and does not establish its results or effectiveness. [9]

Open problems include evaluating tool-use pipelines across retrieval, arguments, execution, and recovery; ensuring that API simulations and LLM judges do not distort results; and testing whether defenses preserve utility while preventing malicious tool outputs from redirecting actions. [3][4][5][6] The evidence is heterogeneous: it includes survey excerpts and brief HF summaries as well as benchmark reports, and some reported results are specific to selected models, tasks, and environments. Conclusions about real-world incidence or universal rankings are therefore not supported by these notes. [3][6][7][8][9]

## References
[1] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/html/2406.05804 (n.d.)
[2] ReAct: Synergizing Reasoning and Acting in Language Models. hf-search. https://huggingface.co/papers/2210.03629 (2022-10-06)
[3] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. web. https://arxiv.org/abs/2304.08244 (2023-04-17)
[4] StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models. hf-search. https://arxiv.org/pdf/2403.07714v5.pdf (2024-03-12)
[5] AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. hf-search. https://huggingface.co/papers/2504.08942 (2025-04-11)
[6] The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis. web. https://arxiv.org/html/2602.10453v1 (2026-02-11)
[7] Agent-SafetyBench: Evaluating the Safety of LLM Agents. web. https://arxiv.org/abs/2412.14470 (n.d.)
[8] ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback. hf-search. https://huggingface.co/papers/2601.10156 (2026-01-15)
[9] What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents. hf-daily. https://huggingface.co/papers/2610.06406 (2026-10-05)
