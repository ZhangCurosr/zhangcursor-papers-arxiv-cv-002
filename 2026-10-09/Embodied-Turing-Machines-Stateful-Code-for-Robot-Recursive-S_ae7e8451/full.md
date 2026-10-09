# Embodied Turing Machines Stateful Code for Robot Recursive Self-Improvement

Kairui Hu<sup>1,2,\*</sup> Siyuan Hu<sup>1,\*</sup> Fangzhou Hong<sup>1,2</sup> Zhaoxi Chen<sup>1,2</sup> Ziwei Liu<sup>1,2</sup>

<sup>1</sup>Nanyang Technological University <sup>2</sup>Ropedia

![](images/9234726a3e7829d7651d79f99d8a25bc768b6c5170b260248397a485a7112ddd.jpg)  
Figure 1. Code-Only-as-Policy (COAP). We view the embodied world as an Embodied Turing Machine whose tape is the state of the robot and its environment (top middle). If this state is represented accurately and the code is robust enough, no agent is needed in the control loop (top right). COAP is a shared library: different tasks use the same library, and within a task the code runs across episodes. The policy is therefore code only and reaches a 70.24% success rate on RoboDojo.

## Abstract

## Ropedia

Most robot policies keep a model in the control loop: a VLA maps observations to actions, and an Agent Harness, such as Agent-as-Policy or Harness VLA queries a VLM for decision making at run time. We propose a different view: the embodied world is an Embodied Turing Machine, whose tape is the robot and environment state and rules are the policy. If this state can be represented accurately, the decision making can be written entirely in code. We therefore propose Code-Only-as-Policy (COAP): code measures and tracks the robot, environment, and task state from camera images and proprioception, and makes every decision from it. The same code applies across episodes, and different tasks share one library without a VLM or VLA in the loop.

Compared with VLAs and Agent Harnesses, we analyze three advantages of COAP: (i) Explicit State: the state can be stored in code; (ii) Execution: code makes decision making controllable, recovers from failures flexibly, and runs fast and cheaply online; (iii) Extensibility: new tasks reuse, inherit, or extend the shared library, so capabilities can accumulate over tasks. These advantages make COAP a suitable medium for recursive self-improvement (RSI): coding agents develop the library in a closed loop, and each change is explicit and controllable. On RoboDojo’s 42 bimanual tasks, the resulting library reaches a success rate of 70.24% without a model at test time. The upper bound of COAP lies in how accurately the state is represented for decision making and how robust the code logic is. We thus propose COAP as a new paradigm for embodied tasks; since it applies across episodes, it can also serve as an efficient data engine for VLAs and Agent Harnesses.

## 1. Introduction

“It is possible to invent a single machine which can be used to compute any computable sequence.”

Alan Turing [60]

A Turing machine comprises a tape that stores symbols and a set of fixed rules that read from and write to the tape [60]. With the rules fixed and only the tape varying, the same rules can process arbitrary inputs. The same abstraction applies to the physical world, where the world state is analogous to the tape. A robot reads this tape through perception and writes to it through action, and its policy plays the role of the rules. We therefore model a robot and its environment as an Embodied Turing Machine, whose tape holds two kinds of state: the environment state (e.g., object poses and their spatial relations) and the robot state (e.g., arm pose and reachable poses).

Both kinds of state are accurately observable. The robot state is available directly through proprioception, and object poses can be recovered from RGB images using measured object geometry. A policy that acts on this state gains two advantages. First, more accurate state enables more precise action. Second, when objects are rearranged (like different episodes in a task), only the tape changes, and the same rules (policy) still apply.

A policy can map state to action in three ways, which differ in how explicit this mapping is. (i) Vision-language-action models (VLAs). A VLA takes observations, such as camera images, as input and learns the mapping to action, which remains implicit in its weights [4, 19]. (ii) Agent harnesses. The vision-language model (VLM) either controls the robot directly, as in Agent-as-Policy, or directs a VLA, as in Harness VLA [9, 55]. The harness makes part of the process explicit, such as textual memory, but the VLM and VLA remain black boxes, so we may still neither fully inspect nor control how an action is derived. (iii) Code-Only-as-Policy (ours). In contrast, code is a white box that implements the Embodied Turing Machine directly: the state, such as object poses, is stored in code variables, where it remains accurate, persistent, and explicit; the rules are written as code logic that computes each action from this state. With both the state and the rules explicit, no model is required at runtime: the code is frozen and serves as the sole policy during execution. Code thus becomes a universal interface to the physical world: a single code program takes over the roles of the VLAs, the prompts and skills of Agent-as-Policy, and the memory of Harness VLA.

![](images/842ef236ac59e4bd1f3f3c4a2c812abbe0269a488bed00795015d112a034fdae.jpg)  
Figure 2. Comparison of code-based robot policies. In prior work, a model writes code-as-policy online, turn by turn, as the robot runs (left), or one program takes its state from a learned model or the simulator oracle (middle). Our COAP reuses code across episodes and measures the state by code (right).

Existing code-based policies either (i) ask a model to write new code-as-policy online in each episode of a task [21, 59, 62], or (ii) take the state from a model or the simulator’s oracle [11, 39] (Figure 2). COAP instead measures the state with its own code and builds a shared code library for perception and execution. Different tasks run the same library code, and only a small part of the logic is task-specific: 83% of the code that a new task runs comes from the library. Nor is the code written for a particular episode: one program runs on every episode of a task. Because the task is fixed, code that is robust enough needs no VLM or VLA in the loop; if it still needs one, the code is not yet robust enough to run reliably across episodes.

Code-Only-as-Policy has three advantages (§4.3). (i) Explicit State: the state is stored in code variables, and every decision is made from them (§4.3.1). (ii) Execution: the code gives fine-grained control over the state and actions, recovers from failures flexibly by backtracking and re-attempting, and runs fast and cheaply online (§4.3.2). (iii) Extensibility: a new task reuses, inherits, or extends the shared library, so the code evolves quickly and capabilities accumulate (§4.3.3). Hence, COAP is a suitable medium for stable recursive self-improvement (RSI) (§4.4). Since the code is not specific to any episode, one policy runs on different episodes and can generate successful trajectories efficiently for VLAs and Agent Harnesses (§5.5).

We evaluate COAP on RoboDojo’s 42 bimanual tasks [7]. A coding agent (Opus 5.5 Max) develop the code library in a closed RSI loop that we design: under fixed rules, such as no hack for single episodes and no oracle state at test time, the coding agent summarizes, writes, and debugs the code without human in the loop. The development does not involve test set of a benchmark. The resulting COAP policy reaches 70.24% success rate, 38.9% above the state-ofthe-art method. The upper bound of COAP can be pushed further if the state representation gets more accurate and code logic grows more robust.

The weakness of COAP mainly lies in its generalization capabilities since it needs a small amount of task-specific code logic. However, Agent-as-Policy and Harness VLA also involve task-level memories, and some even rely on episode-level logic. In this work, our aim is to explore the upper bound of Code-Only-as-Policy for the community. Moreover, COAP also generates successful trajectories stably and efficiently, which can benefit the development of VLAs and Agent Harnesses (§5.5). Our contributions are:

(i) Code-Only-as-Policy. We view the embodied world as a Turing machine in which the robot and environment state can be represented explicitly. The same code applies to different episodes, and different tasks use a shared library developed offline by coding agent.

(ii) Advantages of Code-Only-as-Policy. We analyze three advantages of code over VLAs and Agent Harnesses: Explicit State, Execution, and Extensibility. We explore code-only as a new paradigm for embodied task: when the code is robust enough and applies across episodes, no VLM or VLA is required during runtime.

(iii) Code as the medium of RSI. Code iterates efficiently, and each change is controllable. Code-Only-as-Policy can also serve as an efficient data engine to produce beneficial trajectories as new training data for VLAs, Agent-as-Policy, etc.

## 2. Related Work

Existing robot policies built with foundation models differ in where they keep the state and how they choose each action from it (Table 1): a VLA encodes the state implicitly in its weights, and an Agent Harness (Agent-as-Policy or Harness VLA) stores it as text in a model’s context.

Vision-language-action models. RT-2, OpenVLA, �<sub>0</sub>, and $\pi _ { 0 . 5 }$ learn language-conditioned visuomotor policies end to end [4, 5, 19, 36], and recent foundation models scale data and training further [18, 48]. A VLA cannot be inspected or edited directly, so each improvement requires new data and a new training run, and LIBERO-PRO shows that benchmark success can reflect memorized training configurations [58]. In comparison, we keep the state in code variables, so we can backtrack from a failure to the variable that caused it and fix the code logic that derived it. Each accepted fix is merged into the code without breaking existing logic, so improvements accumulate, while training a VLA on a new task can degrade its performance on earlier tasks [61].

VLM (System-2) for robot control. ReAct interleaves reasoning with actions [53], and Toolformer teaches a model to call tools [38]. In robotics, SayCan grounds language-model plans in learned affordances [1], and Inner Monologue feeds environment feedback back to the planner [16]. Recent harnesses let a VLM agent call analytic or learned primitives [9, 55], and GPT-6 Astra [34] has been evaluated directly as an embodied policy [43, 56]. In these systems, the model is queried repeatedly during execution to output actions, so there is randomness at each call, even if the prompt is the same. Moreover, every episode is of high cost due to extensive model calls [35]. In our approach, models are used only offline: coding agents write the code, and the robot runs it without a model during online execution. The code computes each action from the state through explicit rules, so the robot’s behaviour is reproducible, and each decision is cheap because model calls are not required.

Code-as-Policy. Code-based policies differ in when the program is written and in the source of the state used (Figure 2). Code-as-Policies and ProgPrompt prompt a language model to write code that calls perception and control APIs [21, 41], and VoxPoser writes code that composes spatial value maps [17]. Later works add execution feedback, repeated generation, and skill discovery [12, 26, 30]. PyRUA-Lean has a model write code cell by cell as each episode runs [59], and Skill2Real has a model compose a new program from frozen skill memories at each decision stage [62]. In these systems, a model writes the program at runtime, and the state is provided by learned perception models: object detectors in Code as Policies, and SAM in PyRUA-Lean and Skill2Real. In RHO and EmbodiedSWE, coding agents write one program offline, but its state still comes from outside the code: RHO likewise obtains it from SAM and Molmo [6, 11], and the solvers of EmbodiedSWE read the simulator’s oracle values [39]. These perception models are black boxes, and their uncertainty are not controlled by the code. In contrast, our code tracks the state itself, so the state is programmable: when a measurement is wrong, coding agents fix the measurement code, and the state becomes more accurate over time.

Table 1. Four ways to build a robot policy with foundation models, compared on the properties of the three advantages in Section 4.3. ✓ yes, △ in part, ✗ no. In the rest of the paper, we refer to Agent-as-Policy and Harness VLA together as an Agent Harness.
<table><tr><td></td><td>Agent-as-Policy</td><td></td><td>Harness VLA</td><td></td><td>VLA/WAM</td><td>COAP (ours)</td></tr><tr><td colspan="7">Explicit State</td></tr><tr><td>Inspectable</td><td></td><td>△ reasoning text</td><td></td><td>△ memory text</td><td>X black box</td><td>state and rules</td></tr><tr><td>Persistent</td><td></td><td>△ model context</td><td></td><td>△ text memory</td><td>X recent frames</td><td>code variables</td></tr><tr><td>Accurate</td><td></td><td>△ pixels, some tools</td><td></td><td>△ pixels, some tools</td><td>X image pixels</td><td>measured values</td></tr><tr><td colspan="7">Execution</td></tr><tr><td>Controllable</td><td></td><td>△ steered by prompt Δ steered by prompt  weights decide</td><td></td><td></td><td></td><td>code decides</td></tr><tr><td>Recoverable</td><td></td><td> retries via LLM</td><td></td><td>Δ retries via memory X no state to return</td><td></td><td>√backtrack, re-attempt</td></tr><tr><td>Efficient</td><td></td><td>X LLM per decision</td><td></td><td>X LLM + VLA</td><td>△ network per step</td><td>√ CPU, no model</td></tr><tr><td colspan="7">Extensibility</td></tr><tr><td>Reusable</td><td></td><td>△ base model only</td><td>X memory per task</td><td></td><td> fine-tune per task</td><td>√ shared library</td></tr><tr><td>Fast to evolve</td><td></td><td>△ prompt edits</td><td>△ slow, linear gains</td><td></td><td>X data, retraining</td><td>reuse, inherit library</td></tr><tr><td>Cumulative</td><td></td><td>△ prompts affect all</td><td>Δ task memory only</td><td></td><td>X tasks overwrite</td><td>performance accumulates</td></tr></table>

Code as an interface for agents. Outside robotics, code has become the medium through which agents act and improve. CodeAct uses executable code as the action space of an agent [46], and exposing tools as a code API lets a model compose them with fewer tokens [2, 44]. Voyager grows a library of code skills through interaction [45], and software agents solve long-horizon repository tasks through tests and diffs [33, 50]. In these settings, the environment provides the state explicitly, as files, test results, or game API values. A robot, in contrast, must measure its state from sensors, and our code performs this measurement itself. In robotics, RoboRSI runs RSI as a loop of manager, planner, engineer, and reviewer agents with a no-regression gate, but a VLM still makes the decisions during execution [32]. A change that passes the gate only steers this black box, whose uncertainty and errors cannot be fully controlled, so the tested behaviour may not be the behaviour at runtime. Our policy is code with explicit state, so the tested behaviour is the behaviour at runtime, and coding agents can improve it the way software engineers improve a program: every change is tested offline, and only changes that improve results without regressions on other tasks are merged.

Benchmarks. LIBERO measures knowledge transfer in lifelong robot learning [24], RoboTwin 2.0 generates domain-randomized bimanual tasks [8], and RoboCasa365 scales household simulation [31]. RoboDojo spans memory, precision, open instructions, generalization, and long-horizon manipulation in one protocol [7]. Its memory and precision tasks directly test how well a policy retains and measures state, and the shared protocol lets us compare VLAs, world models, agents, and code on the same 42 tasks.

## 3. Embodied Turing Machine: Code-Only-as-Policy

Code-Only-as-Policy implements the Embodied Turing Machine (§1) as a shared code library, in which code variables hold the state and the source code constitutes the rules. Beyond storing the state, the perception code measures the environment from camera images, and the code

![](images/f644479c020e8fb02b54e75b9206918e2c82f6dfaff12f1c65a00564b0e71314.jpg)  
Figure 3. Code-Only-as-Policy. Development (left): coding agents improve the code offline in a loop. They run the current code $\theta _ { k }$ on development layouts, read the state, and run alternative diffs on branches in parallel. Shared library (middle): task programs call the library from the shared code. Execution (right): at runtime the robot runs one fixed task program across episode, and an agent and a VLA are not required.

updates these measurements as the robot acts. The code is frozen at runtime and improved offline by coding agents (Figure 3). The development does not involve test set of a benchmark.

## 3.1. Reading and writing the state

At step �, RoboDojo provides an observation $o _ { t }$ consisting of an instruction, three RGB images (head and two wrists), joint angles, end-effector poses, and the gripper command echo [7]. The code maintains an explicit state $x _ { t }$ with three parts, each of which has the same form in every task. (i) The environment state records the pose, size, and category of each object and the spatial relations between objects, such as one object resting on another; each record also stores the error, source, and time of its measurement. (ii) The robot state records the joint angles, end-effector pose, and gripper opening of each arm, together with the poses each arm can reach. These two parts represent the tape of the Embodied Turing Machine. (iii) The task state records the current stage, the arm assigned to each object, retry counts, and the information the task must retain, such as the color under each cup. It is not part of the tape and plays the role of the internal state of a Turing machine.

Each part is updated from a different source. The robot state is obtained from proprioception at every step. The environment state is measured by perception layer library. The task code updates the task state during execution.

The source code � constitutes the rules. It defines the transition

$$
( x _ { t + 1 } , a _ { t } ) = F _ { \theta } ( x _ { t } , o _ { t } ) ,\tag{1}
$$

```python
code_4.py
131 def run(ctx):
133 snap = ctx.snap(('head',))
134 cubes = blocks.measure_cubes(ctx, snap, edge=CUBE_EDGE, colours=COLORS, n=3)
135 cups = containers.measure_cups(ctx, snap, inverted=True, n=3)
141 reg = ctx.register (list(cubes) + list(cups))
142 slots = blocks.slots_by_x(reg[:3], reg[3:])
143 order = [('cover', i) for i in range(3)] + [('uncover', blocks.slot_of(slots,
c)) for c in COLORS]
144 memory = {s.colour: s.index for s in slots}
146 for k, (op, i) in enumerate(order):
147 stage = SPEC.stages[k]
153 cup, _src, dst = containers.move_for(slots, i, op)
154 outs, placed, verified = _move_cup(ctx, cup, dst, stage, HOME_EST_STEPS)
```

Code Example 1. The core of a task program. As in Figure 3, perception calls are blue, the manipulation step is teal, and writes to the state are outlined.

where $a _ { t }$ is the next trajectory segment. The code � is frozen at runtime, and only �<sub>�</sub> changes. A new episode layout therefore changes only the state, so one program runs in every episode layout, including the held-out test layouts (§4.1).

Code Example 1 shows the core of a task program that covers three cubes with cups and then uncovers them in a fixed color order; it uses all three parts of the state. reg holds the environment state: the measured records of three cubes and three cups, which the code pairs into cube-cup slots ordered from left to right. memory and stage hold the task state: the color under each cup and the active subgoal. The move step, shown in teal color, uses the robot state to select an arm and compute each motion. After the third move, the colors are no longer visible, so the code retrieves them from memory to determine the target of each uncover move.

## 3.2. Measuring the environment state

Perception is also ordinary code, built on NumPy and OpenCV, so the state is programmable: every measurement can be inspected and revised like any other rule. In each frame, the robot’s mesh is projected into the image using the measured joint angles, and the pixels it covers are masked out. Pixels that differ from the current table color form candidate regions, and each connected region whose lower edge rests on the table becomes an object candidate. For a boundary pixel with camera centre � and ray direction $d ,$ intersecting the ray with the table plane � = �<sub>0</sub> yields its position on the table:

$$
p = c + \lambda d , \qquad \lambda = ( z _ { 0 } - c _ { z } ) / d _ { z } .\tag{2}
$$

A perception for each kind of object then places the known 3D shape of the candidate near �, searches over its position and yaw, and keeps the fit with the largest overlap with the observed region. The camera models, robot meshes, object shapes, and table height used in these steps are published with the benchmark and do not change between episodes, so they are constants in the code, part of the rules �. What changes between episodes, such as which objects are present and where they are, is state, which the code measures from the images in every episode.

```python
grasp.py
122 LIFT_UP = (0.0, 0.0, 1.0) # default: the established vertical lift
312 def execute_grasp(c, ctx, attempt=1, ..., lift_dir=LIFT_UP,
313 lift_speed=None, lift_accel=None, band_at_close=None):
```

```python
code_5.py
46 class Rules(CO.Rules): # inherit an existing task's flow
57 def first(self, ctx, items, info): # override one decision
58 sort_language.label(items, self.cats, info.get('frame'))
73 def run(ctx):
82 return CO.run(ctx, Rules(cats))
```  
Code Example 2. A default-off parameter of a shared function (top) and a task that inherits another task and overrides one decision (bottom); the new code is highlighted.

Each matched measurement yields an object record with category, position, yaw, size, and confidence. Records persist in an episode-level WorldState: an object occluded by the arm keeps its last measured pose. Tasks that require millimetre accuracy re-measure the object with a wrist camera before the final approach.

## 3.3. Sharing one library across tasks that read the same state

Two design choices make the library reusable across tasks. (i) Every task operates on state of the same form (§3.1), so a module written for one task can be reused by other tasks. (ii) The code is layered, with dependencies pointing downward (Figure 3, middle): task programs call object-specific code, which calls manipulation code such as grasp, pick, place, push, and pour, which in turn calls perception and motion code. Task code specifies what to do: the order of subgoals, task-specific decisions, and recovery conditions. The lower layers determine how. For example, a single transfer call expands into grasp-candidate generation, inverse-kinematics and collision filtering, a pre-grasp re-measurement, approach, closure, a short lift with a hold check, carry, placement, release, and a final verification, each of which returns an outcome with measured evidence. As a result, 83% of the lines are shared across tasks (§4.3.3).

A new task extends the library in one of three ways (Code Example 2). It can compose existing modules, such as transfer. It can inherit an existing task and override a single decision: a task that sorts objects into the baskets named in its instruction reuses the entire flow of an existing sorting task and replaces only the rule that assigns objects to baskets. Or it can add a default-off parameter to a shared function: lift\_dir allows a grasped object to be lifted diagonally, which one task requires near a basket wall, while the default preserves the existing vertical lift.

Default-off parameters provide a compatibility guarantee. Let $B _ { i } ( \theta , o )$ be the execution trace of task � under code � and observations �. For every task � that does not pass the new argument, an extension � must satisfy

$$
B _ { i } ( \theta + \delta , o ) = B _ { i } ( \theta , o ) \qquad \forall o \in { \cal O } _ { \mathrm { t e s t } } ,\tag{3}
$$

where $O _ { \mathrm { t e s t } }$ denotes the fixtures exercised by the unit tests. Existing tasks therefore retain their

behaviour by construction, and only the adopting task changes.

## 3.4. Improving the code by backtracking failures to the state

Because the code records its state at every step of an episode, coding agents can improve the code offline from these records. Improvement proceeds in rounds and is carried out by Claude Opus 5.5 coding agents, which wrote all code in the library and in every task, ran the tests and evaluations, and debugged their own code autonomously. Let $\theta _ { k }$ be the accepted code after round �. Each agent works in its own git worktree, inspects the recorded state of failed development runs to locate the variable or rule at fault, and proposes a diff Δ. The resulting branches form a search frontier of runnable code:

$$
C _ { k } = \{ \theta _ { k } + \Delta _ { k , 1 } , \ldots , \theta _ { k } + \Delta _ { k , m } \} .\tag{4}
$$

Because each branch is complete, runnable code, branches can be evaluated concurrently under identical conditions. Before simulation, each branch passes two CPU checks: the task code is executed against a numeric scene to confirm that every stage reaches its target state, and the robot and object meshes are checked for inverse-kinematics solutions and collisions at every waypoint. Branches that pass are evaluated on the validation episodes (§4.1) in parallel.

Debugging and selection use separate data. Agents debug on a development set they define themselves, where they watch videos and compare the measured state against simulator ground truth; selection compares success counts on the validation split. Let $J _ { q } ( \theta ; V _ { q } )$ be the success count of task � on these layouts and $\epsilon _ { q }$ a margin above rerun noise. A branch is accepted if

$$
J _ { q } ( \theta _ { k } + \Delta ; V _ { q } ) \ \geq \ J _ { q } ( \theta _ { k } ; V _ { q } ) + \epsilon _ { q } ,\tag{5}
$$

in which case it becomes $\theta _ { k + 1 }$ and the base of all subsequent branches. Otherwise, the current code remains deployed and the diff is not merged. The compatibility condition (Equation 3) protects the tasks that a diff does not target, and the acceptance rule (Equation 5) ensures that the targeted task improves, so each accepted step is non-decreasing and improvements accumulate.

## 4. Experiments

This section answers three questions: (1) How well does Code-Only-as-Policy (COAP) perform on a hard benchmark (§4.1–§4.2)? (2) What does COAP offer that a VLA or an Agent Harness lacks (§4.3)? (3) Why does COAP suit RSI (§4.4)?

## 4.1. Setup

Benchmark and protocol. RoboDojo [7] contains 42 bimanual tasks that span five dimensions: Memory, Precision, Open instruction following, Generalization, and Long-Horizon manipulation. Each task is evaluated with three seeds of 50 layouts each. In the Generalization tasks, half of the layouts are randomized, with new object instances, a new room and table appearance, and up to 20 distractors. All reported numbers are produced by the official evaluation client, judges, and summary script.

Inputs. The policy receives only three RGB camera streams, proprioception (joint angles, endeffector poses, and the echo of the gripper command), and the language instruction. COAP uses no depth, no segmentation, and no model service at test time, and runs efficiently and cheaply.

Table 2. Results on RoboDojo. SR is the success rate (%), the share of episodes in which the whole task succeeds; every column except Score reports SR. Score is the benchmark’s progress score.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Memory (6)</td><td rowspan="2">Precision (8)</td><td rowspan="2">Open (8)</td><td rowspan="2">Generalization (12)</td><td rowspan="2">Long-Horizon (8)</td><td colspan="2">Overall</td></tr><tr><td>SR</td><td>Score</td></tr><tr><td>Agent Harness</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PhysicalRSI [15] (agent + VLA)</td><td>46.6</td><td>32.5</td><td>24.9</td><td>15.6</td><td>37.3</td><td>31.38</td><td>36.27</td></tr><tr><td>GPT-6 Astra [56]</td><td>38.7</td><td>4.0</td><td>31.0</td><td>30.5</td><td>8.2</td><td>22.48</td><td>28.97</td></tr><tr><td>World Action Model (WAM)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Awomo-0.5 [3]</td><td>41.2</td><td>33.5</td><td>22.9</td><td>23.8</td><td>26.8</td><td>29.64</td><td>35.34</td></tr><tr><td>VPP2-Preview [37]</td><td>50.3</td><td>25.6</td><td>5.4</td><td>24.6</td><td>22.2</td><td>25.62</td><td>31.40</td></tr><tr><td>Liber-0 Preview [23]</td><td>37.3</td><td>33.2</td><td>5.3</td><td>18.6</td><td>33.2</td><td>25.52</td><td>30.74</td></tr><tr><td>Liber-0 Lite [22]</td><td>35.4</td><td>31.1</td><td>2.6</td><td>19.1</td><td>32.9</td><td>24.23</td><td>29.24</td></tr><tr><td>InternW0-∆ [29]</td><td>34.0</td><td>23.2</td><td>10.2</td><td>22.8</td><td>29.3</td><td>23.91</td><td>30.77</td></tr><tr><td>ME-Brain-1.0 [14]</td><td>22.8</td><td>15.9</td><td>5.3</td><td>15.3</td><td>20.6</td><td>15.99</td><td>21.67</td></tr><tr><td>OpenWAM-α [47]</td><td>9.1</td><td>9.2</td><td>1.1</td><td>14.8</td><td>25.3</td><td>11.92</td><td>17.18</td></tr><tr><td>VLA Model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Simate-beta [40]</td><td>33.0</td><td>26.9</td><td>8.5</td><td>27.9</td><td>43.4</td><td>27.96</td><td>33.95</td></tr><tr><td>Rex-M1 Preview [13]</td><td>48.2</td><td>23.8</td><td>6.1</td><td>26.1</td><td>33.0</td><td>27.44</td><td>33.79</td></tr><tr><td>DM0.5 [10]</td><td>47.4</td><td>16.8</td><td>2.1</td><td>10.9</td><td>19.5</td><td>19.34</td><td>24.90</td></tr><tr><td>GalaxeaVLA (G0.5) [25]</td><td>7.3</td><td>20.4</td><td>1.6</td><td>12.8</td><td>32.2</td><td>14.88</td><td>20.23</td></tr><tr><td>Xiaomi-Robotics-1 [49]</td><td>6.6</td><td>18.8</td><td>3.6</td><td>17.0</td><td>23.7</td><td>13.93</td><td>20.07</td></tr><tr><td>Meituan-Robotics-0 [28]</td><td>8.9</td><td>7.8</td><td>4.2</td><td>8.2</td><td>18.6</td><td>9.53</td><td>14.95</td></tr><tr><td>SimpleMemVLA [54]</td><td>33.2</td><td>2.9</td><td>0.8</td><td>3.9</td><td>5.5</td><td>9.27</td><td>12.58</td></tr><tr><td>Hy-Embodied-0.5-VLA [57]</td><td>12.1</td><td>8.0</td><td>0.6</td><td>8.4</td><td>14.9</td><td>8.80</td><td>13.07</td></tr><tr><td>KinRT [52]</td><td>3.6</td><td>9.9</td><td>3.8</td><td>8.6</td><td>18.1</td><td>8.80</td><td>13.02</td></tr><tr><td>Spatial Forcing [20]</td><td>4.1</td><td>10.6</td><td>1.6</td><td>9.3</td><td>14.6</td><td>8.04</td><td>12.38</td></tr><tr><td>VLAct [51]</td><td>0.6</td><td>15.2</td><td>2.2</td><td>6.3</td><td>13.7</td><td>7.58</td><td>10.65</td></tr><tr><td>StarVLA-PI_v3 [42]</td><td>4.0</td><td>12.5</td><td>2.0</td><td>8.1</td><td>11.0</td><td>7.51</td><td>10.81</td></tr><tr><td>InternVLA-A1.5 [27]</td><td>3.6</td><td>10.2</td><td>1.4</td><td>6.8</td><td>13.8</td><td>7.14</td><td>11.15</td></tr><tr><td>π0.5 [36]</td><td>4.7</td><td>5.5</td><td>1.7</td><td>8.2</td><td>14.7</td><td>6.93</td><td>11.44</td></tr><tr><td>Code-Only-as-Policy (COAP)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COAP (ours)</td><td>89.9</td><td>75.1</td><td>64.1</td><td>65.7</td><td>56.4</td><td>70.24</td><td>75.45</td></tr><tr><td>vs. best</td><td>+39.6</td><td>+41.6</td><td>+33.1</td><td>+35.2</td><td>+13.0</td><td>+38.86</td><td>+39.18</td></tr></table>

Baselines. We compare with the official RoboDojo leaderboard [7]. Table 2 lists its 23 entries at or above $\pi _ { 0 . 5 } ,$ grouped into three categories. Agent Harness methods include PhysicalRSI [15] and GPT-6 Astra [56]. World Action Models (WAMs) include Awomo-0.5 [3], VPP2-Preview [37], Liber-0 Preview [23], Liber-0 Lite [22], InternW0-Δ [29], ME-Brain-1.0 [14], and OpenWAM-� [47]. VLA models include Simate-beta [40], Rex-M1 Preview [13], DM0.5 [10], GalaxeaVLA (G0.5) [25], Xiaomi-Robotics-1 [49], Meituan-Robotics-0 [28], SimpleMemVLA [54], Hy-Embodied-0.5-VLA [57], KinRT [52], Spatial Forcing [20], VLAct [51] and StarVLA-PI\_v3 [42], InternVLA-A1.5 [27], and � [36].

## 4.2. Main results

Table 2 compares Code-Only-as-Policy (COAP) with the leaderboard. COAP achieves a success rate of 70.24%, 38.9 percentage points above the current state-of-the-art method (31.38%), and more than 40 points above the strongest World Action Model (29.64%), and the strongest VLA (27.96%). COAP also has the highest success rate in each of the five dimensions. The margins are largest on Memory (89.9%, 39.6 points above the best baseline) and Precision (75.1%, 41.6 points), the two dimensions in which the next action depends on a stored value or a precise measurement. COAP reaches 65.7% on Generalization (77.3% on standard and 54.1% on randomized layouts) and 64.1% on Open. Its margin is smallest on Long-Horizon tasks.

![](images/4c93de85aac9d6f4028aeb76a1b4657879e22e9ed4dd376e593ea51d82fd74f5.jpg)

## 4.3. Advantages of Code-Only-as-Policy

We compare Code-Only-as-Policy (COAP) with a VLA and an Agent Harness on three advantages, summarized in Figure 4: Explicit State (§4.3.1), Execution (§4.3.2), and Extensibility (§4.3.3). For each baseline, we use the strongest leaderboard entry of its category (Table 2). Where the benchmark provides no measurement, we use numbers reported in prior work [4, 19, 36, 61].

![](images/95e58ea81e92534592e98705726b9a814817d86ca7741b04bd05309ba5b9c511.jpg)  
Figure 4. Three advantages of COAP, each with three properties.

## 4.3.1. Explicit State

Finding 1. Code-Only-as-Policy stores the robot, environment, and task state as explicit variables and computes actions from them. In comparison, an Agent Harness keeps the state in text memory and its model context, and a VLA keeps it implicitly in its weights.

A1 Inspectable. Code-Only-as-Policy is a white box. What the program perceives about the robot and the environment becomes explicit state in code: variables, data structures, and the parameters of functions. As shown in Figure 5 (left), the state has five categories: robot, object, environment, relation, and task. The coding agent can therefore write the decisions in advance as code over these values, so no agent is needed during online execution. In comparison, in a VLA, both the state and the reasoning are implicit in the weights. An Agent Harness keeps the task memory, but its decisions still come from model inference: an agent or a VLA still does the reasoning, so some states stay hidden in this model-in-the-loop system, which cannot be fully inspected (even the same memory can lead to different actions at different calls).

![](images/01517e9dcf809067daffa1fcc1c4018d8acc9abfcbc93472f68a526eaa2af423.jpg)

![](images/8406c748497ce1908c612ba0acf09e96137d36a0ee75acb03339d3aa74402b41.jpg)

![](images/e39aee665d0d2a67b9d1917c2bd925ce8e68c4802810c861178a2c39b8735c91.jpg)

![](images/2e9b1327e2977f53b0ba67d8104201eb5b3dcff6277c5a27fdf05a7dca6c1235.jpg)

![](images/08ca985995d430a12e0695712f166ac1badcb23e4831d989c1e80dba47257cc3.jpg)  
Figure 5. Explicit State. COAP keeps five categories of state as variables (left), remembers states and values (middle), and succeeds more often with more accurate measurements (right).

A2 Persistent. COAP stores each measured value in a variable of the program, and the value persists throughout the execution. For example, when a static object is covered, its position and pose are still tracked in code, as shown in Figure 5 (middle). Decisions that depend on memory, such as following a specific order, are also written explicitly in code. In comparison, a VLA sees only its recent frames, and an Agent Harness keeps what it has seen only in its model context or text memory. Memory-related tasks can therefore be solved easily by code, since the state is tracked and stored persistently in the program: on the Memory tasks of RoboDojo, COAP reaches 90%, against 47% for the Agent Harness and 33% for the VLA.

A3 Accurate. COAP acts on measured quantities, because the measurements can be estimated offline without accessing the simulator’s oracle values. The cameras, the robot’s 3D meshes, and the sizes and appearance of the objects are estimated before deployment, so at test time the program only locates each object, its position, size, and pose, and sets the grasp width and the clearance from them. During development, the oracle values help improve the perception functions, so the success rate on measured state can approach that on oracle values: as shown in Figure 5 (right), the same program succeeds in 63% of episodes on measured state and in 83% on oracle state, and each measurement that is fixed narrows this gap. In comparison, a VLA and an Agent Harness act mainly on image pixels and never make measurements explicit. COAP therefore has a higher upper bound on tasks that need precision: on the Precision tasks of RoboDojo, it reaches 75%, against 33% for the Agent Harness and 27% for the VLA.

![](images/824afa97ca10d600c6b342fc6702c621cea85cdb25a56fe76c0f295168e097b8.jpg)

![](images/ab2b243b4d49b753b26c47fff7c79e8f590c660cc063ed712a566c4a4b380ef5.jpg)  
Figure 6. Execution. Backtracking keeps failures from compounding over long tasks (left, middle), and code runs on a CPU with a small fraction of a model’s compute (right; COAP’s compute as a share of each model’s, from reported model sizes).

## 4.3.2. Execution

Finding 2. Code-Only-as-Policy controls the state and the decision making. Because the state is tracked, failures can be recovered by backtracking and re-attempting through the robust code design. Online execution is efficient and cheap since it is model-free.

B1 Controllable. COAP has no model in the online execution, so code controls the state, its variables, and the decision making. For example, when a bowl held by its rim slipped during fast carries, one line of code limited the carry speed of rim grasps, and the bowl no longer dropped. In comparison, a VLA, or a model in the loop of an Agent Harness, is a black box: some of its uncertainty and errors cannot be fully controlled and accumulate over the execution. More training trajectories for a VLA, or more text memory for an Agent Harness, still leave its decisions uncertain.

B2 Recoverable. Because the state is tracked and controlled, the program knows where the task stands after each step, so it can backtrack and re-attempt. As shown in Figure 6 (middle, bottom), when an initial grasp fails, the program backtracks, measures the object again, and re-attempts with a new grasp method, such as a new grasp angle. In episodes where something went wrong, the program noticed the problem itself in 76% and re-attempted with a new method in 34% (Figure 6, middle). Recovery matters more as tasks grow longer, since without it one failed step ends the task (Figure 6, left). In comparison, a VLA has no tracked state to return to.

B3 Efficient. COAP uses a model only offline, when the coding agent writes the code, and the code is shared by every episode of a task. A control step takes a median of 0.3 ms on a CPU, and an episode uses 0.8–2.4% of the compute of a VLA and less than 0.1% of that of an Agent Harness (Figure 6, right). In comparison, an Agent Harness, especially GPT-as-policy, is much slower and more expensive, and it reasons anew in every episode instead of sharing its decisions across the episodes of a task.

## 4.3.3. Extensibility

Finding 3. Code-Only-as-Policy builds every new task on a shared library by reusing, inheriting, or extending its code. Each new task is thereforefast to develop, and capabilities accumulate without harming the performance on old tasks.

C1 Reusable. A new task reuses, inherits, or extends the shared library, so its own code holds only task-specific logic, such as a check or a picking order. As shown in Figure 7 (left), 83% of the code that new tasks run comes from the library. The library can be shared because its functions are abstracted step by step at a sensible level (Figure 8). For example, grasping is not written for each object but by grasp method, such as from the top, by the rim, or by the handle, so a new object can use an existing method. In comparison, a VLA needs fine-tuning for new tasks. The memory of an Agent Harness is task-specific and cannot be shared, and its execution is specific to each episode, so it cannot be shared either.

C2 Fast to evolve. Since the library is shared, a new task can reuse, inherit, or extend the existing library, so COAP evolves easily to new tasks. Figure 7 (middle) shows development on new tasks from scratch, on tasks that had no program when the library was frozen. COAP reached an average success of 47% within 15 hours, without data or training. Its curve rises steeply, because the library becomes more robust as it grows, and every task benefits from it. The baseline lines are schematic. A VLA needs new demonstrations and retraining; OpenVLA, for example, reports 5–15 hours of fine-tuning per task on 8 A100 GPUs [19]. On the same tasks, the strongest VLA reaches 4.7% after training on demonstrations. An Agent Harness improves roughly linearly and much more slowly: most of its memory is task-specific, and only a small part, such as general insights, carries over between tasks.

![](images/351c9bfd017c26577c6903adeb5e451613a751872a094d63b4ffc6bbc3d551e9.jpg)

![](images/81adfc7c1f21b185baa6a409ff4e77978427ea98e3290cca4bae6a3bdd46d749.jpg)

![](images/ca6dc1eaaaa060f039f30274d31c13a5e4432a72dc63b916873c43ba7df24ae6.jpg)

Figure 7. Extensibility. New tasks run mostly library code (left), are developed from scratch within hours without data or training (middle; baseline lines are schematic), and add to what the policy can do (right, schematic).  
![](images/ea0053fa4482462fe999780976110af607b22d7d3c9ea0e7b06fd8875b301822.jpg)  
Figure 8. Code structure. (a) The layers of the library. (b) Functions are abstracted at a sensible level: grasping is written by grasp method, not by object. (c) A simplified task program; each call is colored by its layer.

C3 Cumulative. The library is designed to be refactorable: a new task adds functions that inherit from existing ones, or options that are off by default, instead of changing the logic of earlier tasks. Earlier tasks therefore keep running unchanged code, and capabilities can be added without making old tasks worse. As the code becomes more robust, each new task becomes simpler to add, so the capabilities of COAP grow exponentially (Figure 7, right; schematic). In comparison, an Agent Harness builds task-specific memory by running each task and reuses little of it, so its capabilities grow linearly. A VLA must be retrained for new tasks, and the effect on old tasks cannot be controlled: after sequential fine-tuning of � on five real-world tasks, its average score drops from 86.9 to 31.4 [61].

![](images/993b98e4b1eeb850465201b466c514cc7c4b350da9898b06bd7398aff5a5d971.jpg)  
Figure 9. Why Code-Only-as-Policy is a suitable medium for RSI. Each row is an attribute important for RSI, which comes from an advantage in Section 4.3 (check: yes; half circle: partly; cross: no).

![](images/13400cd8760f955293583ce492a4c0fb516069efb1a813bc097eedd43326018c.jpg)

![](images/d9375ab59300c1a437e415f95df8ac67c3b3653c19fd941a05f1571f2dd99864.jpg)  
accepted: beats its parent by ≥ 4 of 30 episodes

![](images/0b8014985146acc345192ff88ae775d94d9921bbea92f5fbc2c553d89ed02a31.jpg)  
branches per round, k  
branches per round, k

![](images/7e743c5995d80fc3a0e922c9b52130bbdfb63441572e38e74e0f6db3d5c5a0e2.jpg)  
Figure 10. Parallel branching over the variants of one decision. (a) Development as a BFS: eight grasps of a mug run at once, and the best code logic is chosen (decisions 2 and 3 are schematic). (b) Of 36 branches, each measured against its parent, few help. (c) With more branches, an accepted change is more likely and larger. (d) With more branches, an accepted change comes sooner in wall-clock time, for more GPU time.

## 4.4. Why Code-Only-as-Policy Suits RSI

Finding 4. Code-Only-as-Policy suits RSI because of its three advantages: Explicit State, Execution, and Extensibility. Because execution is controllable, development is also efficient. The variants of a decision can be developed in parallel, like a Breadth-First Search (BFS).

What are important attributes for RSI? Figure 9 lists four attributes that we think are important for RSI, and COAP is a better interface for all four. RSI repeats one loop: find a failure, change the policy, and keep the change if it helps.

Efficient: fast evolving through parallel branching. Development can be organized as a BFS over decisions. Each node is a decision, such as how to grasp a mug, and its branches are different ways to make it, such as grasping the mug by its rim, its handle, or its body. Because every variant is a complete program that needs no training, the branches are developed in parallel on the same layouts, and the best code logic is chosen before the search moves on to the next decision (Figure 10). In comparison, each branch of a VLA would be a training run, and an Agent Harness calls a large model in every rollout.

As shown in Figure 10, a single change rarely helps, but with more branches, a better change is found more often, its gain is larger, and it comes sooner in wall-clock time, at the cost of more GPU time. Our development records show the same trend: decisions tried with two or more variants found an accepted change in four of five cases.

## 5. Discussion

This section discusses what the state of Code-Only-as-Policy (COAP) should contain (§5.1), where COAP still fails (§5.2), what sets its upper bound (§5.3), its limitations (§5.4), and how it can serve as a data engine (§5.5).

## 5.1. What Should the State Contain?

The state is the tape of the Turing machine (§3), and its five categories are shown in Figure 5 (left). The robot state describes the robot itself: its joint angles, hand poses, gripper openings, and what each hand holds. The environment state describes the position, size, and class of each object, the table, and the objects’ spatial relations, such as which object stands on which. The task state records the task progress, such as the objects picked so far and the current subgoal. All values are in physical units and are stored in the program.

## 5.2. Error Analysis of COAP

Finding 5. COAP stillfails in some episodes. The current coding ability has limits, and the offline RSI loop needs further optimization so that coding agents write better code. The failures fall into two groups: limits of state measurement and limits of the code logic.

We assigned failed episodes of the runs in Table 2 to the class of the smallest change that would have prevented them. Figure 11 shows the share of each class and one example of each.

Perception error. The input is RGB only, and recognition is not always accurate, so a wrong value can enter the state. In Figure 11a, a dartboard in a randomized scene is recognized as the headset. Such errors may be more severe in real scenes, but better estimation and tools can help.

Contact error. In principle, the more clearly the state is represented, the higher the upper bound of the code. For contact, especially with deformable objects, the current representation of the state is coarse and unclear, so the program cannot always tell how an object is held. Such contact is inherently hard to represent, but it is still a limit of the state representation. In Figure 11b, the jaws close on a shirt’s hem, and the soft cloth slips out.

Incomplete logic. The code logic is not robust enough, for example because a retry is missing. In Figure 11c, the push function stops after one push, although its own check finds a cube still turned. Within the limited development time, the code was not made fully robust.

Poor code design. A function works in the cases it was written for but does not generalize, for example because of a hard-coded value, such as moving down by one centimetre. In Figure 11d, a fixed size threshold filters out a small watch. The fix is to derive the value from the state.

## 5.3. The Upper Bound of COAP

Finding 6. The upper bound of COAP is set by two factors: how accurately and comprehensively the program represents the state, and how robust the code logic is, for example whether it is aware of failures and retries.

State representation. Perception and contact errors, 62% of failures, are limits of state measurement (Figure 11, left). Perception errors arise because the input is RGB only, so some estimates are inaccurate. Contact errors arise because the physical state is hard to represent: for deformable or otherwise complex objects, part of the contact is not represented well. The more accurately the state is represented, the higher the upper bound of COAP.

Robust code logic. Incomplete logic and poor code design, 37% of failures, are limits of the logic (Figure 11, right). More robust code removes them: functions that notice their own failures, retry with another method, and derive values from the state.

## 5.4. Limitations

Simulation only. All results are currently in simulation. COAP’s perception relies on the public configuration and assets of the simulated benchmark, such as the camera parameters, the robot meshes, and the object sizes, which is a major limitation. On a real robot, these would have to be measured, and contact-related parameters, such as grip widths and carry speeds, tuned again.

Offline development. Online execution is fast, but offline development takes time: each iteration of the code needs evaluation and revision. Unlike a model such as GPT, which can be used directly as a policy, COAP has a development cost for every new task.

![](images/8e22dde9fce9513600ad308f33829cd69252303f58965166001fa8d8bfdc750c.jpg)  
store\_laptop\_and\_headphones Randomized scene: the dartboard wa recognized as the headset. Fix: better estimation and tools.

![](images/7a3dada361348970b847f73bfb4204c1fb59b756386e4410b069295646a063c8.jpg)  
The jaws closed on the shirt's hem, but the soft cloth slipped out. Fix: grasp the hem again.

![](images/a51ecf703efaaf7d492312278c4c248787929da539ca9e4035e2d6dfece9ba35.jpg)  
The push stopped after one try, though its check found a cube still turned. Fix: push until the check passes.

![](images/4b9991e3c20afbcf7347ba9ef870e07750c25b780a9465f8c2cf380e34b52062.jpg)  
classify\_objects\_by\_language A size threshold fixed for the task filtered out the small watch. Fix: derive it from the scene.

Figure 11. Four classes of error. Each column gives the share of failures in one class, two frames of one failed episode from the official video, and the fix.

Generalization. Code may have limited generalization capabilities. In principle, the upper bound of code is high, and it rises as the state is represented more clearly (§5.3), but COAP currently succeeds only in 70.24% of episodes. Part of the gap comes from incomplete logic and poor code design (§5.2), such as a hard-coded value that works only in the cases it was written for. In addition, the programs were developed on validation layouts and never on the test layouts, so lower success on layouts the code has not seen is to be expected, especially on the randomized layouts of the Generalization tasks, with new objects, rooms, and distractors (54.1%, against 77.3% on their standard layouts). Still, the code does not depend on the scene: each program is task-level code, the library is shared across tasks, and on standard layouts, success on test layouts (74.9%) stays close to that on validation layouts (76.7%). Although code has such limits, in scenes where the state is clear, it runs stably and can produce data reliably (§5.5).

## 5.5. COAP as an efficient Data Engine for VLAs, Agent-as-Policy and Harness VLA

Finding 8. COAP produces successful trajectories quickly and reproducibly, so it can serve as a data enginefor VLAs, Agent-as-Policy, and Harness VLA.

Once a task program works, it runs on a CPU without a model and can be executed on many layouts in parallel, so successful trajectories can be obtained very quickly. Each trajectory comes with the program’s explicit state and the decision behind every action, because the policy is a white box. These trajectories can train a VLA, and the recorded states and decisions can supervise the model inside an Agent Harness. COAP can therefore benefit the development of VLAs and Agent Harnesses.

A new direction. Code-Only-as-Policy differs in many ways from VLAs, World Action Models, and Agent Harnesses (Agent-as-Policy and Harness VLA, which keep a model in the control loop): its state is explicit, no model runs during execution, and the policy is improved by coding agents rather than by training on data. Since its approach differs in so many ways, we view it as a new direction. If the embodied world is a Turing machine, code can be a universal interface to the physical world (§1). Our analysis suggests that this direction has advantages (§4.3–§4.4), and we hope that our exploration can benefit VLAs, Agent-as-Policy, Harness VLA, and RSI.

## 6. Conclusion

We propose that the embodied world, a robot together with its environment, is in essence an Embodied Turing Machine: the world state is its tape, and the robot policy is its rules. On this basis, we introduce Code-Only-as-Policy (COAP), whose approach differs from those of existing directions, such as VLAs, World Action Models, and Agent-as-Policy: the state is explicit, the rules are code, and coding agents improve the code. On RoboDojo, COAP succeeds in 70.24% of episodes, 38.9% above the state-of-the-art method, and leads all five dimensions. Compared with a VLA and an Agent Harness, COAP has three advantages (§4.3): Explicit State, Execution, and Extensibility. Because the policy is a white box, it suits RSI, and its development is efficient (§4.4). Code still has an upper bound, set by the state representation and the code logic (§5.2– §5.3), but as coding agents become stronger, this upper bound will rise. We also believe that COAP can serve as an efficient data engine for VLAs, Agent-as-Policy, and Harness VLA (§5.5).

## References

[1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, et al. Do as I can, not as I say: Grounding language in robotic affordances. In Conference on Robot Learning (CoRL), 2022.

[2] Anthropic. Code execution with MCP: Building more efficient agents. Anthropic Engineering Blog, https: //www.anthropic.com/engineering/code-execution-with-mcp, 2025.

[3] Awomo-WAM Team. Awomo-0.5: World action model pre-trained at scale. Code repository, https://github.c om/Awomo-WestlakeDI/Awomo-0.5, 2026.

[4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, et al. �<sub>0</sub>: A vision-language-action flow model for general robot control. In Robotics: Science and Systems (RSS), 2025.

[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023.

[6] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, et al. SAM 3: Segment anything with concepts. In International Conference on Learning Representations (ICLR), 2026.

[7] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, et al. RoboDojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies. arXiv preprint arXiv:2607.04434, 2026.

[8] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In International Conference on Machine Learning (ICML), 2026.

[9] Yanzhe Chen, Zechen Bai, Zhijun Cao, Wenzheng Zeng, Kevin Qinghong Lin, Yiqi Lin, Guoqiang Liang, Kevin Yuchen Ma, et al. Show-Harness: Just a VLM agent can play robots. arXiv preprint arXiv:2609.10522, 2026.

[10] Dexmal. DM0.5: Designed for the open world, where generalization emerges. Blog post, https://www.dexmal .com/blog/dm0.5/index\_en.html, 2026.

[11] Karim Elmaaroufi, Justin Svegliato, Sarunas Kalade, Graham Schelle, Sanjit A. Seshia, and Matei Zaharia. RHO: Your coding agent is secretly a roboticist. arXiv preprint arXiv:2606.16458, 2026.

[12] Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, et al. CaP-X: A framework for benchmarking and improving coding agents for robot manipulation. In International Conference on Machine Learning (ICML), 2026.

[13] Futian Lab and VisIncept. Rex-M1 Preview. RoboDojo policy adapter, https://github.com/XPolicyLab/ XPolicyLab/pull/131, 2026.

[14] Wei He, Hengtao Li, Chenfeng Wang, Zhongrui Yu, Xuhan Zhu, Maokui He, Zide Liu, Xiyue Zhang, et al. ME-Brain-1.0: Memory, cognition and action for evolving embodied intelligence. arXiv preprint arXiv:2609.24271, 2026.

[15] HKU MMLab and Kinetix AI. PhysicalRSI 1.0: Recursive self-harness for scaling embodied skills. Project page, https://mmlab.hk/research/PhysicalRSI, 2026.

[16] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, et al. Inner monologue: Embodied reasoning through planning with language models. In Conference on Robot Learning (CoRL), 2022.

[17] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. VoxPoser: Composable 3D value maps for robotic manipulation with language models. In Conference on Robot Learning (CoRL), 2023.

[18] Dongyoung Kim, Huiwon Jang, Myungkyu Koo, Suhyeok Jang, Taeyoung Kim, Beomjun Kim, Byungjun Yoon, Changsung Jang, et al. RLDX-1 technical report. arXiv preprint arXiv:2605.03269, 2026.

[19] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, et al. OpenVLA: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024.

[20] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial Forcing: Implicit spatial representation alignment for vision-language-action model. In International Conference on Learning Representations (ICLR), 2026.

[21] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In IEEE International Conference on Robotics and Automation (ICRA), 2023.

[22] LiberAI. Liber-0 Lite. RoboDojo policy adapter, https://github.com/XPolicyLab/XPolicyLab/tree/ma in/policy/liber\_0\_lite, 2026.

[23] LiberAI. Liber-0 Preview. RoboDojo leaderboard entry, https://robodojo-benchmark.com/LeaderBoard, 2026.

[24] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.

[25] Yicheng Liu, Zibin Dong, Baijun Ye, Tianyuan Yuan, Tao Jiang, Anqi Yang, Shicheng Cao, Haonan Liu, et al. G0.5: One autoregressive stream for robot reasoning and action. arXiv preprint arXiv:2608.11739, 2026.

[26] Runyu Lu, Yubo Wu, Ethan Kou, Letian Fu, Wenli Xiao, Ajay Mandlekar, Yinzhen Xu, Guanya Shi, et al. ASPIRE: Agentic /skills discovery for robotics. arXiv preprint arXiv:2607.00272, 2026.

[27] Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, et al. InternVLA-A1.5: Unifying understanding, latent foresight, and action for compositional generalization. arXiv preprint arXiv:2607.04988, 2026.

[28] Meituan Robotics. Meituan-Robotics-0. RoboDojo policy adapter, https://github.com/XPolicyLab/XPol icyLab/tree/main/policy/Meituan\_Robotics\_0, 2026.

[29] Xingyu Miao, Zizun Li, Baole Fang, Kaiwen Song, Tenghui Wang, Hanxue Zhang, Yating Wang, Xudong Li, et al. InternW0-Δ: A world action model bridging predictive dynamics and actions with 20K+ hours of open data. arXiv preprint arXiv:2609.31394, 2026.

[30] Dhia Naouali, Minghan Wu, Claudia Wong, Abhinav Puthran, and Omar G. Younis. VLCP: Vision language control policy closed-loop code replanning for robot manipulation. arXiv preprint arXiv:2608.16978, 2026.

[31] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. RoboCasa365: A large-scale simulation framework for training and benchmarking generalist robots. In International Conference on Learning Representations (ICLR), 2026.

[32] Noematrix Team. RoboRSI: Stable, efficient, and reusable robot self-evolution in complex real-world environments. Blog post, https://lab.noematrix.ai/blog/2-roborsi/, 2026.

[33] OpenAI. Codex CLI. https://github.com/openai/codex, 2025.

[34] OpenAI. GPT-6 Astra: A new generation of intelligence. https://openai.com/index/gpt-6-astra/, 2026.

[35] OpenAI. Pricing. OpenAI API documentation, https://developers.openai.com/api/docs/pricing, 2026.

[36] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, et al. �<sub>0.5</sub>: a vision-language-action model with open-world generalization. In Conference on Robot Learning (CoRL), 2025.

[37] RobotEra. VPP2-Preview. RoboDojo leaderboard entry, https://robodojo-benchmark.com/LeaderBoard, 2026.

[38] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[39] Zeyu Shen, Haoxiang You, Yilang Liu, Zhicheng Zheng, Lihan Zha, Kashu Yamazaki, Mingtong Zhang, Suning Huang, et al. EmbodiedSWE: Coding agents for long horizon dexterous robotics. arXiv preprint arXiv:2609.27308, 2026.

[40] Simate. Simate-beta. RoboDojo policy adapter, https://github.com/XPolicyLab/XPolicyLab/tree/ma in/policy/Simate\_beta, 2026.

[41] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, et al. ProgPrompt: Generating situated robot task plans using large language models. In IEEE International Conference on Robotics and Automation (ICRA), 2023.

[42] StarVLA Community. StarVLA: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[43] Jiayi Su, Yixin Zheng, Mi Yan, Li Yi, Zhizheng Zhang, and He Wang. GPT 6 Astra as an embodied policy. Technical report, https://anonymous-report-421.github.io/public-website/, 2026.

[44] Kenton Varda and Sunil Pai. Code Mode: the better way to use MCP. Cloudflare Blog, https://blog.cloud flare.com/code-mode/, 2025.

[45] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research (TMLR), 2024

[46] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better LLM agents. In International Conference on Machine Learning (ICML), 2024.

[47] Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, et al. OpenWAM: An open, modular exploration towards systematic world-action model pretraining. arXiv preprint arXiv:2609.07398, 2026.

[48] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, et al. A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692, 2026.

[49] Xiaomi Robotics Team, Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, et al. Xiaomi-Robotics-1: Scaling vision-language-action models with over 100K hours of real-world trajectories. arXiv preprint arXiv:2607.15330, 2026.

[50] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[51] Senqiao Yang, Chengyao Wang, Yuxin Chen, Zixuan Wang, Longxiang Tang, Haokun Gui, Jinhui Ye, Changsheng Lu, et al. Beyond data scaling: Representation-centric continued pre-training for vision-language-action models. arXiv preprint arXiv:2608.27550, 2026.

[52] Tianhang Yang, Yanze Zheng, Junjie Wang, Wei-Bin Kou, Ruotong Li, and Yujiu Yang. Route by kinematics, act by observation: Kinematics-Supervised expert routing in MoE-Augmented VLA. arXiv preprint arXiv:2607.26807, 2026.

[53] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

[54] Cheng Yin, Wang Xu, Junpeng Yang, Sikyuen Tam, Hanyu Liu, Yuan Yao, Xiangrui Zeng, Junbo Cui, et al. SimpleMemVLA: A simple but effective native-video memory for vision-language-action models. arXiv preprint arXiv:2609.05533, 2026.

[55] Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Yi Nie, Chunyang Zhu, Jiaxing Qiu, et al. Harness VLA: Steering frozen VLAs into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

[56] Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, et al. An unexpected robot policy: Early evaluations of GPT-6 Astra on RoboDojo and beyond. arXiv preprint arXiv:2609.24170, 2026.

[57] He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, et al. Hy-Embodied-0.5-VLA: From vision-language-action models to a real-world robot learning stack. arXiv preprint arXiv:2606.14409, 2026.

[58] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.

[59] Ruiyang Si, Jianxin Bi, Shunyu Yang, Rui Ni, Wenbo Huang, Qiang Wang, Shulong Jiang, Duomin Wang, et al. Fewer tokens, better action: GPT-6 Astra robot agents with 14% higher success rate but 65% fewer tokens. arXiv preprint arXiv:2610.01939, 2026.

[60] Alan M. Turing. On computable numbers, with an application to the Entscheidungsproblem. Proceedings ofthe London Mathematical Society, s2-42(1):230–265, 1936.

[61] Jiarun Zhu, Yijun Hong, Xiaoquan Sun, Zetian Xu, Qijun He, Haijier Chen, Zhiyong Wang, Mingqi Yuan, et al. Can vision-language-action models learn from real-world data continually without forgetting? arXiv preprint arXiv:2605.26820, 2026.

[62] Xincheng He, Siyu Ma, Chang Yu, Yunuo Chen, Yanjia Huang, Ying Nian Wu, Yin Yang, and Chenfanfu Jiang. Skill2Real: Agentic skill learning for zero-shot sim-to-real robot manipulation. arXiv preprint arXiv:2610.02788, 2026.

## Appendix Contents

A Advantages of COAP: Details 24   
A.1 Execution . . 24   
A.2 Extensibility . 24   
B Parallel Branching: Details 26   
C Error Analysis: Details 27   
D Per-Task Results 27   
E Case Studies 27   
F Execution Trajectories 29

## A. Advantages of COAP: Details

## A.1. Execution

Failure recovery. Table 3 breaks down the validation episodes in which something went wrong. Reattempted: a failed step was followed by a new method, such as a re-grasp or a second press. Noticed and reported: the program ended the episode itself and reported why, such as an object not found or no feasible path.

Table 3. What COAP did when something went wrong. Left: outcomes of these episodes. Right: failed steps of the reattempted episodes.
<table><tr><td>Outcome</td><td>Share</td><td>Failed step (reattempted)</td><td>Share</td><td>Saved</td></tr><tr><td>Reattempted, succeeded</td><td>12%</td><td>grasp missed or slipped</td><td>49%</td><td>17%</td></tr><tr><td>Reattempted, failed</td><td>22%</td><td>dropped while carrying</td><td>32%</td><td>48%</td></tr><tr><td>Noticed and reported</td><td>42%</td><td>stuck on release</td><td>18%</td><td>23%</td></tr><tr><td>no further method</td><td>37%</td><td>press did not latch</td><td>8%</td><td>100%</td></tr><tr><td>step budget</td><td>4%</td><td>pour incomplete</td><td>1%</td><td>100%</td></tr><tr><td>Hit the step limit</td><td>1%</td><td></td><td></td><td></td></tr><tr><td>Not reported</td><td>22%</td><td></td><td></td><td></td></tr></table>

Decision time and compute. The policy makes a median of 352 decisions per episode. The median call returns in 0.3 ms, because most calls hand over an action that is already planned; about 25 calls per episode, perception and planning at the start of a segment, hold 99.4% of the decision time. Figure 6 (right) converts COAP’s measured decision time into compute at $1 0 ^ { 1 1 } \mathrm { F L O P / s } ,$ , the peak of one CPU core, an upper bound. The model lines are our estimates from reported model sizes, not measurements: a 3B VLA that predicts 50-step action chunks (�<sub>0</sub> [4]; we count one run of about 5 TFLOP per chunk, which underestimates its compute, since �<sub>0</sub> re-plans after 16 or 25 actions), a 7B VLA run every step (OpenVLA [19], about 4 TFLOP per step), and, for the Agent Harness, one call to a model with $2 \times 1 0 ^ { 1 1 }$ active parameters every 100 steps over a context that starts at 5k tokens and grows by 2k per call. Figure 12 shows where the model compute goes instead.

![](images/9254299ea743a65926f2e523b226e656a42e4a7f7364d700f876e73bbf661b72.jpg)  
Figure 12. Compute moves offline. COAP uses models only offline; a VLA and an Agent Harness run them online.

Success by task length. Table 4 groups the tasks by the benchmark’s step limit. COAP leads in every group; its success falls on the longest tasks, as every policy’s does, and the strongest Agent Harness loses most there.

## A.2. Extensibility

Grasp methods. Figure 13 shows one example of each grasp method of Figure 8b, rendered from the benchmark’s meshes at poses from our grasp code, and Table 5 defines them. A method is decided by the approach and the contact: a grasp that descends from above is a top grasp even when it leans.

Table 4. Success (%) by the step limit of a task. GPT-as-policy: an Agent Harness without a VLA.
<table><tr><td>Step limit</td><td>Tasks</td><td>COAP</td><td>Agent Harness</td><td>VLA</td><td>GPT-as-policy</td></tr><tr><td>200-500</td><td>12</td><td>78.1</td><td>47.3</td><td>21.9</td><td>26.5</td></tr><tr><td>550-800</td><td>15</td><td>78.4</td><td>30.6</td><td>38.0</td><td>26.9</td></tr><tr><td>900-1900</td><td>15</td><td>52.0</td><td>13.2</td><td>22.1</td><td>14.8</td></tr></table>

![](images/4a3212cefa9a9f46ebd73a0e5209720a09778e9dca647bcebad4825950c30190.jpg)  
Figure 13. One example of each grasp method.

Table 5. Grasp methods. Tasks: number of task programs using it.
<table><tr><td>Method</td><td>Approach</td><td>Contact</td><td>Examples</td><td>Tasks</td></tr><tr><td>Top</td><td>from above</td><td>across the body</td><td>blocks, digits, a lying pen, eggs</td><td>28</td></tr><tr><td>Side</td><td>level, toward the axis</td><td>around the body or neck</td><td>a standing bottle, a cup</td><td>6</td></tr><tr><td>Handle</td><td>toward the handle</td><td>around the bar</td><td>a basket&#x27;s arch, a broom, a mallet</td><td></td></tr><tr><td>Rim</td><td>down onto the wall</td><td>one jaw inside, one outside</td><td>a bowl, a mug, a plate</td><td></td></tr><tr><td>Edge</td><td>along the thin side</td><td>on the two faces</td><td>a coin, a bread slice, a key</td><td>3341</td></tr><tr><td>Cloth pinch</td><td>down onto the hem</td><td>fabric layers</td><td>a long-sleeved top</td><td></td></tr></table>

New tasks. Figure 7 (middle) plots the mean best validation success so far over the four tasks that had no program when the library was frozen; store\_tools\_in\_toolbox, which never got a successful program, counts as zero. The two baseline lines are schematic. The VLA line is a slow ramp over 5–15 h, the fine-tuning time per task reported for OpenVLA [19], toward the measured average of the strongest VLA on these tasks (4.7%). The Agent Harness line is a straight line with a small slope: its task-specific memory lets it improve, but slowly, because little carries over from earlier tasks. Table 6 gives the timeline of each new task.

Table 6. Timeline of the new tasks. Hours from the start, and validation success (%) at each merge.
<table><tr><td></td><td>make_toast</td><td>hang_mugs</td><td></td><td>sweep_blocks fill_pen_holder</td></tr><tr><td>First evaluation</td><td>3.2h</td><td>3.2h</td><td>3.9h</td><td>0.2h</td></tr><tr><td>First success</td><td>4.7h</td><td>5.2h</td><td>6.4h</td><td>4.0h</td></tr><tr><td>First merge</td><td>5.8h</td><td>6.8h</td><td>7.9h</td><td>2.7h</td></tr><tr><td>Success at merges</td><td>50 → 77</td><td>7→ 50→ 57</td><td>0→60→ 50</td><td>0 → 20 → 27</td></tr></table>

Old tasks. Between the frozen library and the final code, the old tasks whose own code did not change kept their success: 78.4% before and 79.9% after, on the same layouts (Cochran–Mantel– Haenszel test, � = 0.16; no task changed significantly). With the tasks whose code was later cleaned up included, success went from 80.0% to 81.4%. The test detects a pooled shift of about 4 points, so it rules out a large regression, not every change.

Extension without breaking. Figure 14 shows how new tasks extend the library. A new task adds its own files and changes no line of an existing one (a). New behaviour reaches other tasks by a call to an existing library function, by inheritance, or by a parameter that is off by default, so every other caller runs the same code path as before (Eq. 3) and only the tasks that pass the new parameter change (b). sweep\_blocks, for example, reuses the sliding function added to the library for push\_T (c).

![](images/9534ca06c9bae96ca9785005a4dbb2986f20fbbef0cce98ae1ec5fba8ec4264a.jpg)

![](images/527406060e133dbe3b66d6a3692d8f4b158701d7398cb32aff4ab85a4bf3db37.jpg)  
Figure 14. Extension without breaking. (a) The code each new task runs, by layer. (b) One library function reaches many tasks. (c) sweep\_blocks reuses slide, written for push\_T.

Layers. Table 7 lists the layers of the library (Figure 8a). Every task shares manipulation, perception, and motion, about 14k lines; the families and task programs are larger, but each of their modules serves few tasks.

Table 7. Layers of the library. Used by: share of the 40 task programs that use the layer’s most used module.
<table><tr><td>Layer</td><td>Holds</td><td>Lines</td><td>Used by</td></tr><tr><td>Task programs</td><td>order, parameters, special cases</td><td>23.5k</td><td>2.5%</td></tr><tr><td>Object families</td><td>what an object is, how to hold it</td><td>28.5k</td><td>18%</td></tr><tr><td>Manipulation</td><td>pick, carry, place, push; grasping</td><td>5.4k</td><td>78%</td></tr><tr><td>Perception</td><td>look, measure, confirm</td><td>5.3k</td><td>100%</td></tr><tr><td>Motion</td><td>inverse kinematics, planning, collision</td><td>3.3k</td><td>100%</td></tr></table>

## B. Parallel Branching: Details

Branches and the estimate. A branch is a variant of a program that changes one decision and is evaluated on the same seeds and layouts as its parent. Figure 10b uses every branch whose validation success was measured against its immediate parent. Repeated runs of the same program differ by one to three episodes per evaluation, so the selection margin of four episodes is one to two noise widths. The estimate of Figure 10c draws � effects with replacement from the measured ones and keeps the best if it reaches the margin, otherwise the parent; the bands are the 5th and 95th percentiles over bootstrap resamples. The estimate treats branches of one decision as independent and pools effects across tasks, and each gain is measured on the layouts that selected it, so the true gain is somewhat lower.

All tasks at once. In the busiest window of development, evaluations of 12 tasks ran in parallel and finished in 17% of the time they would have taken one after another.

Merges and test. After the library was frozen, task updates were merged one at a time, each changing shared code only through default-off parameters, and overall success rose at every merge, from 66.5% to 70.24%. Changes are selected on validation, and the gains carry over to test: on the 40 tasks with a program, standard success is 76.7% on validation and 74.9% on test, and randomized success 55.2% and 52.8%.

## C. Error Analysis: Details

How a failure is classified. We read each failed episode from the start and took the first error that was never recovered. Its class is the smallest change that would have prevented it, so each failure has exactly one class (§5.2); Table 8 lists typical cases. The labels are judgement: a trace shows what the program measured, not what happened, so we checked the examples against the official videos. Incomplete logic is judged relative to the robot we have; no fix assumes a new sensor or actuator.

Table 8. Typical cases of each class of error.
<table><tr><td>Class</td><td>Typical cases</td></tr><tr><td>Perception error Contact error Incomplete logic Poor code design Unknown</td><td>unrecognized randomized look; wrong final check; split or merged object slip or drop in the grip; push stops short; tip at release no grasp that also reaches the goal; untried order; unchecked collision fixed size filter; fixed grip-width band; fixed match threshold no lead in the trace</td></tr></table>

## D. Per-Task Results

Table 9 lists the success rate (SR) and the progress score of COAP on each task.

## E. Case Studies

Figure 15 follows three official episodes on development layouts. In cover\_blocks, the program measures the three cubes once, stores their colours by position, and uncovers them in the requested order after identical cups have hidden them (§4.3.1, A2). In a randomized stack\_bowls episode, the hold check after a rim grasp finds that the bowl did not follow the lift; the program grasps it again at a new wrist angle and completes the stack (§4.3.2, B2). The same fold\_clothes program folds the shirt in a standard and in a randomized scene.

Table 9. Per-task results of COAP, by dimension. <sup>†</sup>Task without a program.
<table><tr><td colspan="4"></td></tr><tr><td>Task Memory</td><td>SR Score</td><td>Task</td><td>SR Score</td></tr><tr><td colspan="4">Generalization</td></tr><tr><td>cover_blocks</td><td>100.0 100.0</td><td>push_T stack_blocks</td><td>97.3 94.7</td><td>97.3</td></tr><tr><td>press_by_number</td><td>100.0</td><td>100.0</td><td></td><td>95.2</td></tr><tr><td>swap_T</td><td>100.0</td><td>100.0</td><td>stack_bowls</td><td>93.3 94.0</td></tr><tr><td>swap_blocks</td><td>100.0</td><td>100.0</td><td>fold_clothes</td><td>90.0 91.6</td></tr><tr><td>imitate_sorting_sequence</td><td>74.7</td><td>78.5</td><td>pour_liquid_into_cup</td><td>89.3 89.3</td></tr><tr><td>match_and_pick_from_conveyor</td><td>64.7</td><td>64.7</td><td>arrange_largest_number</td><td>86.7 90.0</td></tr><tr><td>Precision</td><td></td><td></td><td>sort_nesting_dolls_by_size</td><td>64.7 64.7</td></tr><tr><td>play_Xylophone</td><td>99.3</td><td>99.3</td><td>make_toast</td><td>56.7 60.3</td></tr><tr><td>build_tower</td><td>96.7</td><td>97.3</td><td>store_laptop_and_headphones</td><td>55.3 77.3</td></tr><tr><td>insert_tubes</td><td>94.7</td><td>96.8</td><td>hang_mugs</td><td>32.0 41.9</td></tr><tr><td>pour_balls_into_vase</td><td>94.0</td><td>94.0</td><td>sweep_blocks 28.0</td><td>28.0</td></tr><tr><td>fasten_screws</td><td>91.3</td><td>95.7</td><td>pack_objects_into_box 0.7</td><td>6.5</td></tr><tr><td>deposit_coin</td><td>85.3</td><td>86.0</td><td>Long-Horizon</td><td></td></tr><tr><td>insert_key</td><td>39.3</td><td>48.4</td><td>play_tic_tac_toe</td><td>99.3 99.4</td></tr><tr><td>plug_in_charger†</td><td>0.0</td><td></td><td>make_kong</td><td>98.0 98.0</td></tr><tr><td colspan="3">Open play_stacking_toy</td><td></td><td></td><td>82.0 92.6</td></tr><tr><td>pour_by_language</td><td>100.0</td><td>100.0</td><td>put_bottles_into_dustbin</td><td>73.3</td><td>82.8</td></tr><tr><td>align_blocks</td><td>93.3</td><td>93.3</td><td>classify_objects</td><td>50.0</td><td>63.9</td></tr><tr><td>solve_equation</td><td>92.7</td><td>92.7</td><td>fill_egg_holder</td><td>24.7</td><td>50.8</td></tr><tr><td>stack_blocks_by_language</td><td>91.3</td><td>91.6</td><td>fill_pen_holder</td><td>24.0</td><td>49.0</td></tr><tr><td>general_pickup</td><td>61.3</td><td>61.3</td><td>organize_table</td><td>0.0</td><td>51.5</td></tr><tr><td>pick_from_conveyor_by_image</td><td>48.0</td><td>48.0</td><td></td><td></td><td></td></tr><tr><td>classify_objects_by_language</td><td>26.0</td><td>43.9</td><td></td><td></td><td></td></tr><tr><td>store_tools_in_toolbox†</td><td>0.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/a4410c90b227f769eff41629bdbe68d7f2167d525f77b83e2444b9a04c7afd6d.jpg)

## Memory

Uncover by stored colour The cups look alike: the order red, green, blue uses the stored colours.

memory = {red: left, blue: middle, green: right}

![](images/7aba2782f4bf35d7c6096e4f45d7ca72a64fe47ae804d45301964f6399d029e7.jpg)

## Recovery

A failed hold, then a new angle The bowl did not follow the lift: the program grasps it again at a new yaw.

yaw 60° not held yaw 75° held

![](images/be3821642b158ccf5b2748006f1ec827edccb51f5b41ccb1be4b7310ccd4ca85.jpg)

![](images/1a323307645dc24c47c0f1fa7f0a4f6d7f46a9e96856b21a4eec603d67b3fd69.jpg)  
measure the cubes  
grasp at the rim, yaw 60°  
all three covered

![](images/2f504fef5e885e14faffbd16f8ad8cec92d8f92712d9f2b4fb365b70991123e6.jpg)

![](images/c79f4379e1654118fc65d3e7d126ab77a21b64fac8b4a6e47dcea2b8268bf23d.jpg)  
check: the bowl did not follow

![](images/456774901021ad95f2a1821a38e08a21fd9d9b78e222b8ef43ddadf9a03c3b79.jpg)

Standard and randomized scenes The same fold\_clothes program on a standard and a randomized scene.

![](images/9dd9145eb3b008de07433e7fd569b2475717c296e85fff500150ed54ad59b5d0.jpg)  
grasp again, yaw 75°: held

![](images/7bfcd9bb9d3dfaf9bace1f9ee961dfa3b5b79794ccf94e85da9cc6d1b35b594b.jpg)  
standard: start

![](images/ccf2f2b52bc6bf230ad96c8bfd3d65f22c475d8c5b52b2d17abee58943e28db4.jpg)  
green: the right cup

## One program

![](images/49a350e088577251b410a0abc602105cc2b523ef0a250b8b9f19d944444d3b9e.jpg)  
standard: folded

![](images/6f2d12b134e50deec18a76279e17203e3cbcad614411c85c211a7c67f65fca96.jpg)  
randomized: start

![](images/afc22b1270d3ff2e536023c667d54568f765257714f49cbf768662bf616a26a3.jpg)  
red, green, blue uncovered

![](images/11aa8e4adc74b92bf29ecfc18121c30f3bfebab4b1d921475900978f3e0c5225.jpg)  
Figure 15. Case studies: memory, recovery, and one program in two scenes.  
the bowls stacked

![](images/1189a7c94cc1b20ff67d2354cb6f406c2517a092ee440d7c0d3465d0b69fe030.jpg)  
randomized: folded

## F. Execution Trajectories

Figures 16–19 show one successful official episode per task and variant: the head camera at the start, at one quarter, half, and three quarters of the episode, and at the end, then the final frames of the two wrist cameras. Across tasks the same library code recurs: objects are grasped from above or at a rim, carried along planned paths, and placed after a close look, while the task program decides their order and targets.

start  
1/4  
1/2  
3/4  
end  
left wrist  
right wrist  
![](images/6eca19aae31ecbb702ad97b8d8171f453147ca4d1101e05ebddc25467de25349.jpg)  
Figure 16. Execution trajectories (1/4). Successful official episodes.

![](images/61ed0272e33cc2b251e9e6ff6a4eeba3f9f7f89d17ea7768c35bfcdac9deffe3.jpg)  
Figure 17. Execution trajectories (2/4). Successful official episodes.

![](images/fb237e319dfadf5bbdf47bf1057e0ca74e9ed277f36990c1ab0a093007ea6211.jpg)  
Figure 18. Execution trajectories (3/4). Successful official episodes.

![](images/06a04ed862016c47b46ee0c63d35c8917bf0d0248906eacd5b4bb585168ba48d.jpg)  
Figure 19. Execution trajectories (4/4). Successful official episodes.