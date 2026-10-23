# Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation

Chenyuan Chen, Ran Ding, Lechen Zhang, Zewen He and Yoshihiko Nakamura Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi, UAE {chenyuan.chen, yoshihiko.nakamura}@mbzuai.ac.ae

Abstract— Humanoid loco-manipulation in dynamic scenes is non-Markovian: cues vanish as objects change or are hidden, or leave the view as the robot turns and walks, yet whole-body vision-language-action (VLA) policies see only the present. We present WholeBodyMem, an object-guided perceptual memory that keeps the pre-history of object events seen before the robot acts on its goal. SAM3 tracks the task objects, and their events causally select key frames within an adaptive budget, which the action expert reads by cross-attention together with object tokens encoding appearance, geometry and time. We also contribute HuMeGo, a humanoid memory benchmark with six simulated and six real-world Unitree G1 tasks. Keeping the right frames rather than more frames, WholeBodyMem surpasses memoryless and uniform-memory baselines, raising its base policy’s success from 29.1% to 72.6% in simulation and from 36.7% to 88.3% on the real robot, while also improving the standard SIMPLE tasks (70.2% to 86.9%). Project page: https://memory-before-goal.github.io

Index Terms— Humanoid robot systems, imitation learning, mobile manipulation, deep learning methods.

## I. INTRODUCTION

Whole-body vision-language-action (VLA) models [1]– [3], paired with learned whole-body controllers [4], [5], have advanced humanoid loco-manipulation considerably. Yet most of them act on the current observation alone, ignoring temporal dependencies, and thus fail in dynamic scenes where objects change state or are hidden [6], [7], moved or removed. Everyday cues are brief and seldom repeated: a guest waves from one of several identical tables while a waiter carries a tray, and when the waiter turns back, the hand is already down. Fig. 1 shows this situation on a real G1: a cola stops on two seats and a sprite on one, the robot turns away to fetch the cola, and when it comes back the seats look the same again, so a policy that sees only the current view cannot tell where the cola stopped first. Loco-manipulation in dynamic scenes is thus non-Markovian: earlier events determine later decisions, which calls for memory. We refer to it as pre-history memory: the record of the object events seen before the robot acts on its goal. Such memory is especially hard for a humanoid, whose head camera moves as it turns and walks, so cues leave the view, and whose memory must be updated causally while whole-body control runs. Even dual-system designs [2], in which a VLM planner acts as System 2, can reason over the history only if they locate the key frames. Existing memory strategies, mostly developed for tabletop arms, retain long frame histories [8], [9], which are costly and dilute brief cues; compress them [10], [11], losing detail; sample frames at fixed intervals [12], missing brief cues; or select event frames tied to the robot’s own operations or learned from demonstrations [13], [14].

![](images/6aee5b70cdb156fb7bdb68a0b18af89352bb24da906ddaa87868f7c828887aa5.jpg)  
Fig. 1. Memory in a real whole-body task. Top: the cola stops first on the middle seat (1.1 s), then a sprite and the cola stop on the other two; the robot turns away to fetch the cola and, when it returns at 30.0 s, the three seats look the same.

As humans remember the moments when the things they care about change rather than every moment [15], we propose WholeBodyMem, an object-guided perceptual memory for whole-body VLAs. First, a frozen SAM3 [16], a promptable video segmentation model, tracks the task objects in the head-camera stream, and we detect when they appear, disappear, move or change shape, compensating for the camera’s own motion. Second, these object events causally select the key frames that fill an adaptive memory budget, stored as native backbone tokens along with mask-pooled object tokens. Finally, the action expert reads this memory via cross-attention and FiLM, so that whole-body actions depend on what happened, not only on what is visible.

Existing humanoid benchmarks [17]–[20] keep the goal in view and do not test this problem, whereas memory benchmarks [12], [21] are tabletop manipulation. We therefore introduce a memory-dependent humanoid loco-manipulation benchmark. Our contributions are as follows:

• We propose WholeBodyMem, an object-guided perceptual memory for whole-body VLAs, in which object events, rather than even sampling, select the key frames

![](images/a6977cdf85c8b19a51222aed25239f978c9827bf733b92768681377bfbad6b3c.jpg)  
Fig. 2. Memory-dependent tasks in simulation, grouped by memory type. Each block shows one task and its memory suite (as in RoboMME [12]; Sec. III-A): the instruction, nine subtasks and the robot’s head-camera view at each of them. Bold text marks the cue and the answer; purple borders mark the cue frames, and a blue arrow points at the cue object.

of an adaptive budget, read together with object tokens. • We introduce HuMeGo, a memory-dependent humanoid loco-manipulation benchmark: six simulated tasks covering object, spatial and temporal memory, where the scene ends the cue and decisions are scored apart from manipulation, and six real-world G1 tasks, where the robot also turns away from the cue.

• WholeBodyMem keeps the right frames rather than more frames, outperforms memoryless and uniformmemory baselines in decisions and success, helps even on SIMPLE, and carries over to a real G1.

## II. RELATED WORK

## A. Whole-Body Humanoid Loco-Manipulation VLAs

Whole-body humanoid control uses learned whole-body controllers such as HOVER [22], AMO [4] and SONIC [5], and demonstrations are collected by whole-body teleoperation [23]. On top of such controllers, vision-languageaction (VLA) policies map images and instructions to wholebody actions. Generalist VLAs such as $\pi _ { 0 . 5 }$ [24] generalize across scenes and are adapted to whole-body action spaces; GR00T [2] is an open foundation model for humanoids trained on real, synthetic and video data; $\Psi _ { 0 }$ [1] pre-trains on human egocentric video and then fine-tunes on a smaller amount of real humanoid data; and WholeBodyVLA [3] learns a unified latent action space for loco-manipulation. These policies differ in their data, action spaces and controllers, but all of them condition on the current observation and a short proprioceptive history, so an event that has left the camera’s view can no longer affect their actions. Memory has so far been added mostly to robots that do not walk, e.g. EgoMI [25] keeps past views for a wheeled semihumanoid with an actuated head and SOMA [26] builds a spatial memory from a movable head camera; to our knowledge, no prior work studies memory for whole-body locomanipulation, where the robot’s own walking and turning move cues out of view. Concurrent work, POT-VLA [27], keeps a 3D record of where each task object is now but not what happened before; WholeBodyMem keeps the past views in which objects changed.

## B. Memory and Datasets for Robot Policies

Memory mechanisms. Memory for robot policies has been studied mainly on tabletop arms with fixed or wrist cameras [10], [12], [28], [29], and on mobile and navigation agents [30]. Long frame histories [8], [9] grow in cost and dilute a short cue; compression into gist tokens [11] or memory banks and object slots [10], [31] keeps the cost fixed but can lose the detail a decision needs; frame selection keeps native detail at a fixed budget, by uniform sampling [12], retrieval [29] or events [13], [14]. A humanoid adds two constraints: its own turning and walking move a cue out of the head camera’s view and change the viewpoint, and the memory must be updated causally while whole-body control runs. WholeBodyMem therefore selects frames by object events, measured relative to the scene after cameramotion removal, and reads them with action-query crossattention and FiLM. Unlike SAM2Act+ [28], which adopts SAM2’s memory mechanism, we use SAM3 only as a tracker whose object events select the past views the action expert reads. Datasets and benchmarks. Memory benchmarks are mostly tabletop [12], [21], [28], [31], and humanoid benchmarks [17]–[20] evaluate whole-body skills with the goal in view. Our benchmark fills this gap with whole-body humanoid loco-manipulation tasks whose decisions depend on a cue that is no longer in view.

![](images/1bb775b8ee66395fc8a85681a29edcd69a8d26607d97a2323341944f024f7768.jpg)  
Fig. 3. Overview of WholeBodyMem. ×L: the L=6 DiT blocks of the action expert.

## III. MEMORY-DEPENDENT LOCO-MANIPULATION BENCHMARK

## A. Task Definition

We call a humanoid loco-manipulation task memorydependent when, in a dynamic scene, the observation available at the decision does not determine the correct action. Each episode has three phases: in the cue phase, a scene change, such as a lamp that lights or a mug that is hidden, singles out one answer y from a small set Y (two or three pads); in the retention phase, the robot performs whole-body skills that are the same for every answer; and in the decision phase, from a time $t _ { d } \mathrm { { \ o n . } }$ a whole-body action must follow y. A cue can become unobservable in two ways: it can end, as when a light goes off or an object is hidden (scene-induced), or it can leave the head camera’s view as the robot turns and walks (motion-induced). We require that (i) the change is no longer observable at $t _ { d } ,$ whether scene-induced or motioninduced; (ii) neither the instruction nor the robot’s motion before $t _ { d }$ depends on $y ;$ and (iii) $y$ is balanced over Y. A policy without memory then cannot infer y from the current view (chance: $1 / | \mathcal { V } | )$ ), so we score memory-decision accuracy separately from manipulation success (Sec. V-A).

The kind of change determines the memory a decision needs: what changed (object memory, O), where an object went (spatial, S), or when and how often it changed (temporal, T), as in the reference, permanence and counting tasks of RoboMME [12]. Our six simulated tasks (Fig. 2) cover all three. A lamp flashes one to three times in FlashPad (T); a mug disappears behind one of three screens while a decoy mug leaves in HideMug (S); a ball is concealed under one of three cups, which are then swapped, in ShellGame (S); one of two coloured lamps lights up in LitLamp (O); and one of three pads blinks for 0.2 s in BlinkPad and TouchPad (O). Afterwards, the robot puts the carried object on the indicated pad or, in TouchPad, touches that pad. Because the cues are short-lived or occluded [6], [7] while the robot watches, these suites isolate scene-induced loss; the real-world tasks additionally introduce motion-induced loss (Sec. V-E).

## B. Data Generation

A scripted whole-body expert generates the demonstrations in six steps.

1) Task specification. A script sets the single-table scene (the object to carry, the answer pads, the cue objects), the cue timeline and the skill sequence. A seed fixes the object pose (±3 cm, ±8 cm, ±0.15 rad), the robot’s spawn offset (±3 cm), the cue layout and the appearance; the answer is drawn independently.

2) Expert rollout. In MuJoCo (2 ms physics, 50 Hz control), the cue plays for a fixed, task-specific window. In FlashPad, HideMug, ShellGame and LitLamp it starts 0.5 s into the episode, while the robot still stands at the table before the grasp and lift; in BlinkPad it starts 0.5 s after the lift, and in TouchPad at a random time 1.9–3.7 s into the episode, before the robot turns. A precomputed Dex3 grasp, cuRobo arm planning and AMO [4] leg and waist balancing drive the robot through the skill sequence.

3) Selection. An episode is kept if the object stays held after the lift (in the five carrying tasks) and the cue is visible to the head camera during its window; kept episodes are balanced over the answers.

4) Rendering. Recorded states are replayed in Isaac Sim and rendered from the head camera (640×360, 50 Hz)

TABLE I  
RESULTS ON THE 14 SIMPLE TASKS.
<table><tr><td>Task</td><td>InternVLA-M1 [32]</td><td>DP [33]</td><td>ACT [34]</td><td>EgoVLA [35]</td><td>π0.5 [24]</td><td>GR0OT [2]</td><td> $\Psi _ { 0 }$  [1]</td><td>Uniform memory</td><td>WholeBodyMem (ours)</td></tr><tr><td>T1 XMovePick</td><td>0 /  0 / 0</td><td>5 /  /  3</td><td>7  /  /  3</td><td>3 / 3 / 1</td><td>8 /  6 / 3</td><td>6/  6 /  4</td><td>9 /  10 / 6</td><td>10 / 10 / 5</td><td>10 / 10 / 7</td></tr><tr><td>T2 BendPickMP</td><td>4 / 5 /  4</td><td>10 / 9 / 10</td><td>9 / 9 / 9</td><td>6 / 9 / 6</td><td>8 /  8 /  6</td><td>10 / 10 / 9</td><td>10 / 10 / 9</td><td>10 / 10 / 10</td><td>10 / 10 / 10</td></tr><tr><td>T3 Handover</td><td>0 / 0 /  1</td><td>1  / 1 / 1</td><td>1 / 2 / 2</td><td>2 / 3 /  4</td><td>4/2 /3</td><td>6 /6 /7</td><td>7 /6 / 8</td><td>8 /7/ 9</td><td>10 / 9 / 9</td></tr><tr><td>T4 TabletopGraspMP</td><td>6 / 7 / 9</td><td>4 /  4 / 1</td><td>10 / 10 / 8</td><td>6 /  5 /  4</td><td>10 / 10 / 9</td><td>10 / 10 / 8</td><td>10 / 10 / 7</td><td>9 /  10 / 8</td><td>10 / 10 / 9</td></tr><tr><td>T5 XMoveBendPick</td><td>5  /  4 / 4</td><td>0 / 0 /0</td><td>2 /2 / 3</td><td>3 /6 / 6</td><td>0/0/0</td><td>6 /2 /4</td><td>9 / 4 /4</td><td>8 / 6 / 10</td><td>9 / 9 /8</td></tr><tr><td>T6 LocomotionPickBetweenTables</td><td>0 /  0  /  0</td><td>0 / 1 / 0</td><td>1  / 0 / 0</td><td>1 / 3 / 1</td><td>2 / 0 / 1</td><td> / 0 / 0</td><td>4/4/3</td><td>4 / 5 / 4</td><td>8/6/7</td></tr><tr><td>T7 PickAndPlaceAndHugContainer</td><td>0 /  0 / 0</td><td>0 / 0 / 0</td><td>3 / 6 / 6</td><td>1 / 0 / 0</td><td>2 / 3 / 1</td><td> / 0 / 0</td><td>4/5 /3</td><td>6 /4/ 6</td><td>7 /  6 /5</td></tr><tr><td>T8 OpenFaucet</td><td>0 / 0 / 0</td><td>2 /3 / 6</td><td>0 / 0 / 1</td><td> / /  0</td><td>4 / 2 / 0</td><td>0 / / 0</td><td>4/2 / 4</td><td>3/5/3</td><td>8/  5 /7</td></tr><tr><td>T9 CloseDoor</td><td>0 /  0 /  0</td><td>10 /  10 / 10</td><td>10 /  10 / 10</td><td>10 / 8 / 10</td><td>10 / 10 / 9</td><td>10 / 10 / 10</td><td>10  /  10 /  10</td><td>10 / 10 / 10</td><td>10 / 10 / 10</td></tr><tr><td>T10 OpenOven</td><td>8 /  6 /  8</td><td>0 / /  0</td><td>1 / 0 / 1</td><td>5/8/2</td><td>4 / 3 / 6</td><td>0 / 0 / 1</td><td>5 / 6 / 6</td><td>7 / 8 / 9</td><td>9 / 8 / 10</td></tr><tr><td>T11 OpenTrashCan</td><td>0 /  0 / 0</td><td>4/7/7</td><td>4/4/7</td><td>3 /  5 / 4</td><td>7 / 6 /5</td><td>2 /3 / 4</td><td>4/7/7</td><td>7/5/7</td><td>8 / 7  /  8</td></tr><tr><td>T12 PushOfficeChair</td><td>2 / 3 / 1</td><td>10 / 6 / 8</td><td>9 / 10 / 10</td><td>10 / 9 / 10</td><td>10 / 10 / 6</td><td>5 / 5 / 4</td><td>10 / 10 / 9</td><td>8 / 9 / 10</td><td>9 / 10 / 10</td></tr><tr><td>T13 BendHandover</td><td>5 /  2 /  2</td><td>0 / 0 / 0</td><td>0 / 0 / 0</td><td>3 / 2 / 4</td><td>8 / 9 / 8</td><td> / 0 / 0</td><td>10 / 8 / 9</td><td>9 /9 / 9</td><td>10 / 10 / 10</td></tr><tr><td>T14 BendPick</td><td>3 / 0 / 0</td><td>0 / 0 / 0</td><td>0 / 0 /  0</td><td>0 /  /  2</td><td>0 / 1 / 2</td><td>0 / /  1</td><td>8 / 7 / 7</td><td>9 / 10 / 7</td><td>9 / 10 / 8</td></tr><tr><td rowspan="2">Σ (14 tasks)</td><td>89/420</td><td>136/420</td><td>176/420</td><td>169/420</td><td>206/420</td><td>159/420</td><td>295/420</td><td>323/420</td><td>365/420</td></tr><tr><td>(21.2%)</td><td>(32.4%)</td><td>(41.9%)</td><td>(40.2%)</td><td>(49.0%)</td><td>(37.9%)</td><td>(70.2%)</td><td>(76.9%)</td><td>(86.9%)</td></tr></table>

All columns are our evaluation of the 14 SIMPLE tasks with the same client. Cells: successes at randomization levels L0 / L1 / L2 (10 episodes each). Bold: best per task; underlined: second best. Ψ<sub>0</sub>: our fine-tuned checkpoint; Uniform memory: WholeBodyMem without object tokens and with evenly spaced frames. On these tasks, WholeBodyMem uses the fixed seven-frame memory (without adaptive memory).

TABLE II  
COMPARISON WITH ROBOT MEMORY AND HUMANOID BENCHMARKS.
<table><tr><td>Benchmark</td><td>Robot</td><td>Whole- body</td><td>Memory</td><td>Data</td><td>Tasks Steps</td><td></td><td>Mem. (%)</td></tr><tr><td>MemoryBench [28]</td><td>Franka</td><td>x</td><td>√</td><td>Sim</td><td>3</td><td></td><td></td></tr><tr><td>LIBERO-Mem [31]</td><td>Franka</td><td>x</td><td>√</td><td>Sim</td><td>10</td><td></td><td></td></tr><tr><td>RoboMME [12]</td><td>Franka</td><td>x</td><td>√</td><td>Sim+Real</td><td>16+4</td><td>481</td><td>1</td></tr><tr><td>RoboMemArena [21]</td><td>Dual-arm</td><td>x</td><td>√</td><td>Sim+Real</td><td>26+5</td><td>1,076</td><td>一</td></tr><tr><td>HumanoidMimicGen [20] G1</td><td></td><td>√</td><td>x</td><td>Sim</td><td>9</td><td>818</td><td>0</td></tr><tr><td>HumanoidArena [19]</td><td>G1</td><td>√</td><td>x</td><td>Sim</td><td>7</td><td>611</td><td>0</td></tr><tr><td>SIMPLE [18]</td><td>G1</td><td>√</td><td>x</td><td>Sim</td><td>14</td><td>322</td><td>0</td></tr><tr><td>HuMeGo (Ours)</td><td>G1</td><td>√</td><td>√</td><td>Sim+Real</td><td>6+6</td><td>1,155</td><td>40.7</td></tr></table>

Whole-body: humanoid loco-manipulation. Memory: the decision depends on a vanished cue. Tasks: sim + real. Steps: mean episode length. Mem.: share of steps after the decision; –: not reported.

with per-episode randomized background distractors, materials and lighting.

5) Export. Episodes are exported in the LeRobot v2.1 format (32-D state, 36-D whole-body action, 50 Hz video), each ending 1 s after the placement.

6) Training and test split. Each memory task has 100 training demonstrations and 90 test scenes from seeds distinct from training. Test seeds pass the same selection as the training episodes and stay in distribution: layout and expert rollout within the training range, table materials seen in training. They are balanced over the answers and the three randomization levels.

## C. Data Analysis

As shown in Table II, we contrast our benchmark with robot memory benchmarks and with humanoid imitation benchmarks collected on the same platform and control frequency (Unitree G1, 50 Hz). Two properties set it apart. Extended horizons: an average episode spans 1,155 control steps (23 s), compared with 818, 611 and 322 steps in HumanoidMimicGen, HumanoidArena and SIMPLE, respectively. Decisions that require memory: each task presents a short-lived cue that occupies only 0.8–14.7% of an episode and has to be remembered for 13.5–22.5 s after it disappears (task medians). Overall, 40.7% of the steps come after the decision point, where, following conditions (i) and (ii) in Sec. III-A, the vanished cue provides the sole evidence for the correct action. In contrast, no step of the humanoid benchmarks depends on a vanished cue, and the memory benchmarks use arms, not humanoids.

## IV. METHOD: WHOLEBODYMEM

## A. Preliminaries: Whole-Body VLA Policies

A whole-body VLA policy $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t } ~ \vert ~ I _ { t } , s _ { t } , \boldsymbol { \ell } )$ maps the head-camera image $I _ { t } ,$ proprioceptive state $s _ { t }$ and instruction ℓ to H future actions $\mathbf { a } _ { t } = ( a _ { t } , \dots , a _ { t + H - 1 } )$ [33], [36] at 50 Hz: upper-body joint targets $( q _ { \mathrm { a r m } } , \ q _ { \mathrm { h a n d } } )$ and a lowerbody command ({torso, $h _ { b } , v \} \mathrm { i }$ : torso orientation, base height and base velocity) that a lower-body controller turns into $q _ { \mathrm { l o w e r } }$ (Fig. 3). Recent policies such as $\pi _ { 0 . 5 }$ [24], GR00T [2] and $\Psi _ { 0 }$ [1] encode $I _ { t }$ and ℓ with a vision-language backbone and generate $\mathbf { a } _ { t }$ with a flow-matching action expert [37], whose action tokens embed $s _ { t }$ and the noise ε (Fig. 3). Conditioned only on $( I _ { t } , s _ { t } , \ell )$ , they cannot respond to an event that has left the view. WholeBodyMem adds memory tokens M<sub>t</sub> of the past views in which task objects changed and object tokens $\mathbf { O } _ { t } ,$ giving $\begin{array} { r l } { \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t } } & { { } | I _ { t } , s _ { t } , \ell , \mathbf { M } _ { t } , \mathbf { O } _ { t } ) } \end{array}$ (Secs. IV-B and IV-C); selecting views by object events keeps this memory small yet retains short-lived cues.

## B. Object-Guided Memory Selection

The memory is built in three stages (Fig. 3, left): SAM3 tracks the task objects, object events flag the frames where they change, and a causal selector fills an adaptive budget.

Memory budget. A memory frame keeps the K=77 native tokens (7×11 grid) of the frozen vision tower, and the budget is $B _ { \mathrm { m a x } } { = } 7$ frames, the current and six past ones: 539 memory tokens plus at most 14 object tokens (Sec. IV-C). Tokens are cached, so re-selecting a frame is free. Unlike a pooled summary, native tokens keep the small regions a decision may hinge on (a cue can cover under 1% of the image), but only six past frames fit, so which are kept matters.

TABLE III  
MEMORY TASK RESULTS.
<table><tr><td></td><td>Counting (T)</td><td colspan="2">Permanence (S)</td><td colspan="3">Reference (O)</td><td></td></tr><tr><td>Method</td><td>FlashPad</td><td>HideMug</td><td>ShellGame</td><td>LitLamp</td><td>BlinkPad</td><td>TouchPad</td><td> $\operatorname { A v g } .$ </td></tr><tr><td colspan="8">Success (L0 / L1 / L2)</td></tr><tr><td> $\Psi _ { 0 }$  (no memory) [1]</td><td> $7 / 7 / 5$ </td><td> $1 1 / 1 4 / 1 2$ </td><td> $7 / 1 1 / 7$ </td><td> $1 1 ~ / 6 ~ / ~ 4$ </td><td> $1 1 / 1 1 / 1 1$ </td><td> $3 / 1 1 / 8$ </td><td>8.3 / 10.0 / 7.8</td></tr><tr><td>Uniform memory</td><td> $2 2 / 2 3 / 2 2$ </td><td> $1 2 / 2 0 / 1 5$ </td><td> $1 0 / 9 / 9$ </td><td> $1 2 / 1 2 / 1 1$ </td><td> $\overline { { 1 0 \mathrm { ~ / ~ } 9 \mathrm { ~ / ~ } 1 3 } }$ </td><td> $6 / 1 3 / 1 2$ </td><td>12.0 / 14.3 / 13.7</td></tr><tr><td>WholeBodyMem (ours)</td><td>26 /  27  /  23</td><td>21 /  20 / 20</td><td> $\bf { 2 1 } / \bf { 2 0 } / \bf { 1 9 }$ </td><td> $\overline { { 1 8 \ : / 1 5 \ : / 1 2 } }$ </td><td>26 /  22 / 25 26 / 25 / 26</td><td></td><td>23.0 / 21.5 / 20.8</td></tr><tr><td colspan="8">Memory-decision accuracy (%)</td></tr><tr><td>Chance</td><td>33</td><td>33</td><td>33</td><td>50</td><td>33</td><td>33</td><td>36</td></tr><tr><td> $\Psi _ { 0 }$  (no memory) [1]</td><td>30</td><td>47</td><td>33</td><td>47</td><td>45</td><td>32</td><td>39</td></tr><tr><td>Uniform memory</td><td>82</td><td>57</td><td>35</td><td>51</td><td>37</td><td>38</td><td>50</td></tr><tr><td>WholeBodyMem (ours)</td><td>97</td><td>99</td><td>84</td><td>100</td><td>87</td><td>95</td><td>94</td></tr></table>

Successes at L0 / L1 / L2 (30 episodes per level); memory-decision accuracy as defined in Sec. V-A.

Algorithm 1 Causal memory selection at time t   
Require: event store $\varepsilon$ (frame 7→ scored events), tracks available   
at t, budget $B _ { \mathrm { m a x } } ,$ gap $g { = } 5 ;$ uniform $( I , k ) \colon k$ evenly spaced   
frames of I including both ends (the last frame if $k { = } 1 )$   
1: add the events of the tracks available at t to E   
2: S ← uniform $( [ 0 , t ] ,$ min(3, t+1) ▷ first, middle, current   
3: if no reliable track yet or no event frame $f < t$ then return S   
4: for event frames $\scriptstyle f < t$ by decreasing score, until $B _ { \mathrm { m a x } } { - 3 }$ are   
added do   
5: if an object of $f$ has no frame in S within $g$ then add $f$   
6: return S ▷ 3 to $B _ { \mathrm { m a x } }$ frames

Object tracking. The task decides what to remember: a frozen SAM3 [16] tracks the task objects over the episode so far $( o _ { 0 : t }$ in Fig. 3), prompted with one or two short phrases (e.g. “blue cup” in ShellGame, “cracker box” in Handover, T3 in Table I), and returns a mask, a box and a confidence per frame and object. Prompts are written once per task by Claude Opus 5.5 and fixed before evaluation; the event rules and thresholds below are shared by all tasks. In simulation, SAM3 tracks every 50-Hz control step, and frame counts below refer to tracked frames. A similarity transform fitted to background features of consecutive frames removes camera motion, so object motion is measured relative to the scene.

Object events. A past frame becomes a memory candidate only if a tracked object in it appears, disappears (we keep its last visible frame), reappears, moves by over 3% of the image diagonal after camera-motion removal, or changes shape (mask overlap with the previous one below 65%). Events are ranked by type, disappearance and reappearance first, then by confidence; detections below 0.35 are ignored. For example, a blinking pad appears and a hidden mug disappears.

Causal selection. At each time t (Algorithm 1), the first, middle and current frames are always kept, and up to four slots go to the top-scoring event frames, skipping one whose objects all have a selected frame within 5 frames. The remaining slots stay empty, so the memory holds 3 to 7 frames (5.4 on average); the variant without adaptive memory, used on SIMPLE, fills all seven. Only frames up to t are eligible, so the memory never depends on later frames; seven evenly spaced frames form our Uniform memory control.

## C. Memory Cross-Attention in the Action Expert

Memory enters the policy only through the action expert: its action tokens read $\mathbf { M } _ { t }$ and $\mathbf { O } _ { t }$ by cross-attention, and the read-out modulates them through FiLM [38], as in the memory modulator of RoboMME [12].

Memory tokens. As the selected frames are irregularly spaced, each native token $f _ { i } \in \mathbb { R } ^ { 2 0 4 8 }$ gets a sinusoidal code $\phi ( \tau _ { i } , p _ { i } )$ of its frame index $\tau _ { i }$ and grid cell $p _ { i } ,$ and both are projected to the 1536-D width of the action expert:

$$
m _ { i } = W _ { m } [ f _ { i } ; \mathrm { S i L U } ( W _ { p } \phi ( \tau _ { i } , p _ { i } ) ) ] .\tag{1}
$$

Empty slots are masked; $\mathbf M _ { t } = [ m _ { 1 } , \dots , m _ { 5 3 9 } ]$ is computed once per policy call and shared by all blocks and flowmatching steps.

Memory retrieval. In each of the six blocks $j ,$ after joint attention, the H=30 action tokens query the memory with their RMS-normalized form $\bar { \mathbf { A } } _ { j } \mathrm { : }$

$$
{ \bf c } _ { j } = \mathrm { A t t n } \left( \bar { \bf A } _ { j } W _ { Q } , \ { \bf M } _ { t } W _ { K } , \ { \bf M } _ { t } W _ { V } \right) W _ { O } ,\tag{2}
$$

(multi-query attention; rotary codes place the action steps after the packed memory tokens), so each future action step retrieves the frames it needs while the pretrained joint attention stays unchanged.

Modulation. Rather than entering the residual stream, the read-out scales and shifts the normalized action tokens ahead of the block’s adaptive layer norm and feed-forward network:

$$
[ \gamma _ { j } , \beta _ { j } ] = W _ { F } \mathbf { c } _ { j } , \quad \tilde { \mathbf { A } } _ { j } = \bar { \mathbf { A } } _ { j } \odot ( 1 + \gamma _ { j } ) + \boldsymbol { \beta } _ { j } ,\tag{3}
$$

With $W _ { F }$ initialized near zero, the policy initially matches the pretrained one.

Object tokens. The SAM3 masks also give an objectlevel view (Object Masking in Fig. 3). For up to two tracked objects per memory frame, the native tokens are averaged with each cell’s mask coverage as weight and combined with a sinusoidal code of frame index and mask centroid, the box, area and visibility, and a learned slot embedding, giving up to 14 object tokens $\mathbf { O } _ { t } .$ A second cross-attention adds their read-out before Eq. (3), $\mathbf { c } _ { j } \gets \mathbf { c } _ { j } + g \mathrm { A t t n } ( \bar { \mathbf { A } } _ { j } , \mathbf { O } _ { t } )$ with a learnable $g$ (initially 1) and a zero-initialized output projection, so the branch starts silent yet trainable.

![](images/fb61e89eeaf6a8edd184c90b0851ccafe9d93fc4612b71fe3fc332f7ffdb7f51.jpg)  
Fig. 4. Real-world memory tasks on the Unitree G1. Each panel shows the task instruction and four steps of a teleoperated demonstration; the line below gives the difficulty and the memory the decision needs. Blue arrows mark the cue and the answer (dashed: where the cue was, after it is gone); pink arrows show the robot’s motion in each step, straight for walking. S1–S3: the three seats of the cafe tasks, from front to back´

Decision head. To map the remembered cue to a pad, a learned query attends over the memory and object tokens and a null token, and an MLP outputs one logit per answer, trained with a cross-entropy loss on the demonstrated answer from the decision step on. The softmax of these logits, the dynamic weights w (Fig. 3), enters every block’s adaptive layer norm through a zero-initialized map on the time embedding, so training starts from the head-free policy.

Pad tokens. The weights w also point the action expert at a pad. Each pad visible in the current frame gives one token (its native tokens pooled over the SAM3 box, the box geometry and a pad index), forming $\mathbf { P } _ { t }$ . In every block, a further cross-attention with a zero-initialized output reads $w \odot \mathbf { P } _ { t }$ by adding log $w _ { k }$ to the score of pad k and adds its read-out to $\mathbf { c } _ { j } ,$ , so a confident w hands the chosen pad’s token to the action tokens; the branch is active only when the largest weight and the weight on visible pads reach 0.6.

## D. Training Details

Each policy is fine-tuned per task from the pretrained base policy (Sec. V-A) with $\mathcal { L } = \mathbb { E } \| v _ { \theta } ( \mathbf { a } _ { t } ^ { \sigma } , \sigma ) - ( \varepsilon - \mathbf { a } _ { t } ) \| ^ { 2 }$ + $\lambda { \mathcal { L } } _ { \mathrm { { C E } } } .$ : the flow-matching loss of SIMPLE [18] and $\Psi _ { 0 }$ [1], where the action expert $v _ { \theta }$ sees $\mathbf { a } _ { t } ^ { \sigma } ~ = ~ ( 1 - \sigma ) \mathbf { a } _ { t } + \sigma \varepsilon$ with $\sigma \sim \mathcal { U } ( 0 , 1 )$ , plus the decision head’s cross-entropy $( \lambda { = } 0 . 5 )$ . The Qwen3-VL-2B backbone [39] stays frozen; the memory encoder, the memory attention (6 query heads of 256 dimensions, one shared key/value head) and $W _ { F }$ (standard deviation 0.002 at initialization) are trained from scratch. We use AdamW (β=(0.95, 0.999)), a learning rate of $1 0 ^ { - 4 }$ with cosine decay and 1k warm-up steps, bf16 mixed precision, dropout 0.1 and a global batch of 128, for 40k steps on the 94–100 demonstrations of a SIMPLE task or the 100 of a memory task. On the real robot, whose action space differs, each task is trained separately on its 15 teleoperated demonstrations: the action expert’s transformer blocks start from the post-trained action expert, the memory modules are trained from scratch, and the 5k-step checkpoint is deployed.

## V. EXPERIMENTS

Our experiments answer four questions:

Q1 Does adding memory cost reactive skills on the 14 standard SIMPLE tasks, or even improve them?

Q2 On our six memory tasks, does WholeBodyMem act on vanished cues that memoryless policies miss?

Q3 What is the role of pre-history memory: does object guidance, not capacity, explain the gain, and what do key frames, object tokens, the decision head, adaptive memory and memory attention each contribute?

Q4 Does memory carry over to a real Unitree G1, where the robot also turns away from the cue?

## A. Setup

Simulation. All simulated experiments use SIMPLE [18] with the robot, 36-D whole-body action and rendering of Sec. III-B. AMO [4] drives the lower body in the memory tasks and in the two SIMPLE motion-planning tasks (T2, T4), and SIMPLE’s decoupled whole-body controller drives it in the other 12 SIMPLE tasks, whose demonstrations are teleoperated. Each policy runs as a server that the unmodified SIMPLE evaluation client, which also runs our memory tasks, queries once per action chunk.

Real robot. We use a Unitree G1 (29 DoF) with Dex3 hands and a head-mounted RealSense D435 (640×360, 30 Hz). The policy runs on one RTX 5090 with real-time chunking [40]: of the H=30 predicted actions, the first 24 are executed before the next chunk takes over; SONIC [5] executes its 64-D motion token and 14-D hand targets at 50 Hz, and SAM3 tracks one frame per executed chunk (every 24 frames, about 1.25 Hz).

Baselines. All baselines are trained by us on the same demonstrations. (i) Policies without memory: InternVLA-M1 [32], Diffusion Policy [33], ACT [34], EgoVLA [35], $\pi _ { 0 . 5 }$ [24], GR00T [2] and $\Psi _ { 0 }$ [1], the memoryless base policy. (ii) Uniform memory (RoboMME-style): as no published memory method exists for whole-body humanoid VLAs, we adapt the best-performing memory of RoboMME [12], evenly sampled frames read through a memory modulator (FrameSamp + Modul), to our tasks. It is the Uniform memory control of Sec. IV-B, i.e. WholeBodyMem with evenly spaced frames and no object tokens or decision head, at the same 539-token budget (512 in RoboMME), with native instead of pooled tokens, as a cue can cover under 1% of the image; in an earlier evaluation on T5, T6 and T8, RoboMME’s original 32 pooled frames of 16 tokens scored the same as this adaptation (both 53.3%). On the real robot, we compare with $\Psi _ { 0 }$ (Sec. V-E).

Evaluation. Every policy follows the standardized protocol of SIMPLE [18] at three randomization levels (L0: visual and distractor randomization; L1: plus lighting; L2: plus spatial perturbation). The 14 SIMPLE tasks, which need no cue memory, use 10 episodes per level; our six memory tasks use 30 episodes per level (balanced answers; Sec. III-B). An episode succeeds if its final condition holds (e.g. the object lies on the correct pad). On the memory tasks we also report memory-decision accuracy, the fraction of episodes in which the decision head picks the cued answer, otherwise the fraction of placements or touches on the cued pad.

TABLE IV  
ABLATIONS ON THE MEMORY TASKS.
<table><tr><td>Method</td><td>Successes L0 / L1 / L2</td><td>Success (%)</td></tr><tr><td>Uniform memory</td><td>12.0 /  14.3 / 13.7</td><td>44.4</td></tr><tr><td>WholeBodyMem w/o object tokens &amp; head</td><td>18.0 / 19.0 / 17.3</td><td>60.4</td></tr><tr><td>WholeBodyMem w/o decision head</td><td>20.7 / 18.7 / 19.8</td><td>65.7</td></tr><tr><td>WholeBodyMem with memory as context</td><td>8.3 / 7.0 / 8.0</td><td>25.9</td></tr><tr><td>WholeBodyMem w/o adaptive memory</td><td>21.7 / 21.3 / 20.3</td><td>70.4</td></tr><tr><td>WholeBodyMem (ours)</td><td>23.0 / 21.5 / 20.8</td><td>72.6</td></tr></table>

Averages over the six tasks: successes per level (of 30 episodes). Uniform memory as in Table III; the ablations remove adaptive memory, the object tokens together with the decision head, the decision head alone, or the memory attention, replaced by memory tokens in the VLM input (memory as context); the last three are applied to the variant without adaptive memory.

## B. Results on Standard Loco-Manipulation Tasks (Q1)

Memory does not cost reactive skills. On the 14 SIMPLE tasks (Table I), WholeBodyMem matches or exceeds the best baseline on every task and reaches 86.9% success, against 70.2% for $\Psi _ { 0 }$ and 76.9% for Uniform memory. These tasks need no cue memory, but the moving head camera loses targets as the robot walks and bends, and one frame hides task progress; past frames fill these gaps, and object tokens locate objects more finely. Accordingly, the gain over Uniform memory is largest where the robot walks to or between its targets: T6 (21 vs. 13) and T8 (20 vs. 11) give 17 of the 42 extra successes, while both solve T2 and T9.

## C. Results on Memory-Dependent Tasks (Q2)

WholeBodyMem acts on the cue; memoryless policies guess. On the six memory tasks (Table III), $\Psi _ { 0 }$ decides near chance and Uniform memory, the best-performing memory of RoboMME [12] adapted to our tasks, is clearly above chance only on FlashPad and HideMug, as evenly spaced frames miss brief cues, whereas WholeBodyMem identifies the cue in 93.7% of its decisions. Success rises to 72.6%, against 29.1% $( \Psi _ { 0 } )$ and 44.4% (Uniform memory), and stays stable across randomization levels.

## D. Object-Guided vs. Uniform Memory (Q3)

Object guidance, not capacity, explains the gain. At Uniform memory’s budget (seven frames, 539 tokens), choosing frames by object events alone (WholeBodyMem w/o object tokens, which lacks the decision head) lifts success from 44.4% to 60.4% (Table IV), and adaptive memory loses nothing with 22.9% fewer tokens: the role of pre-history memory is to keep the right frames, not more.

Every part contributes. Of the 28.2-point gain over Uniform memory, key frames give 16.0, object tokens 5.3, the decision head 4.7 and adaptive memory 2.2 (Table IV): key frames keep brief cues that evenly spaced frames miss, object tokens identify which object changed, and the decision head maps the cue to a target pad, so its answer labels explain only about a sixth of the gain. Memory fed to the frozen VLM as context instead of through the memory attention falls below even the memoryless $\Psi _ { 0 }$

TABLE V  
REAL-WORLD RESULTS ON THE UNITREE G1.
<table><tr><td>Task</td><td>Mem.</td><td> $\Psi _ { 0 }$  [1]</td><td>WholeBodyMem</td></tr><tr><td>Trash bag</td><td>S</td><td>4/10</td><td>9/10</td></tr><tr><td>Remember-chip</td><td>0</td><td>3/10</td><td>10/10</td></tr><tr><td>Café A</td><td>T</td><td>4/10</td><td>9/10</td></tr><tr><td>Café B</td><td>O+S</td><td>3/10</td><td>9/10</td></tr><tr><td>Café C</td><td>O+S+T</td><td>3/10</td><td>8/10</td></tr><tr><td>Chip-cart</td><td>P</td><td>5/10</td><td>8/10</td></tr><tr><td>Total</td><td></td><td>22/60</td><td>53/60</td></tr></table>

Successes over 10 trials per task and policy with memory types: object (O), spatial (S), temporal (T) memory or task progress (P).

## E. Real-World Experiments (Q4)

We deploy WholeBodyMem and $\Psi _ { 0 }$ on the G1, with 10 trials per task and policy and balanced answers (Table V).

Tasks. Fig. 4 shows six tasks, graded by the memory types (what, where, when) a decision combines. In Tasks 1 and 3–5, a person gives the cue at one of three marked areas (“frames”) or seats on table B; the robot then turns to table A for the object and back, with the same motion for every answer, so the cue is out of view when it acts. In Task 2, the person touches one of the cans, leaving no trace, and in Chip-cart (Task 6) the robot must remember its progress once the remaining cans leave the view.

Data. For each task, we record 15 demonstrations (90 in total) by whole-body teleoperation (PICO headset, controllers and ankle trackers driving SONIC [5] at 50 Hz; Sec. IV-D), with every answer demonstrated five times in the cue tasks; each frame pairs the head-camera image and robot state with the SONIC motion token and hand targets as the action label.

Memory carries over to the real robot. Over all six tasks (Table V), WholeBodyMem succeeds in 88.3% of the trials (53/60), against 36.7% (22/60) for $\Psi _ { 0 } ,$ , even when a decision combines what, where and when.

## VI. LIMITATIONS AND CONCLUSION

We studied the role of pre-history memory, the object events seen before the robot acts on its goal, in humanoid loco-manipulation, with WholeBodyMem, an object-guided perceptual memory for whole-body VLAs, and a humanoid memory benchmark. Object events select the few past views that matter, which the action expert reads within an adaptive budget: keeping the right frames matters more than keeping more, so the policy acts on vanished cues in simulation and on a real G1 without losing reactive skills. SAM3 misses a cue whose object is never detected, online tracking adds inference cost, rendering leaves a sim-to-real gap, and the simulated demonstrations are scripted. Remembering before acting on the goal is a first step toward humanoids that act on what happened, not only on what is visible.

## REFERENCES

[1] S. Wei et al., “Ψ : An open foundation model towards universal humanoid loco-manipulation,” in Robotics: Science and Systems (RSS), 2026.

[2] NVIDIA et al., “GR00T N1: An open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, 2025.

[3] H. Jiang et al., “WholeBodyVLA: Towards unified latent VLA for whole-body loco-manipulation control,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[4] J. Li et al., “AMO: Adaptive motion optimization for hyper-dexterous humanoid whole-body control,” in Robotics: Science and Systems (RSS), 2025.

[5] Z. Luo et al., “SONIC: Supersizing motion tracking for natural humanoid whole-body control,” Science Robotics, vol. 11, no. 117, p. eaed4592, 2026.

[6] B. Van Hoorick et al., “Tracking through containers and occluders in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13 802–13 812.

[7] Y. Xia et al., “TARGO and TARGO-Net: Benchmarking target-driven object grasping under occlusions,” International Journal of Computer Vision, 2026.

[8] M. Torne et al., “MEM: Multi-scale embodied memory for vision language action models,” arXiv preprint arXiv:2603.03596, 2026.

[9] H. Jang et al., “ContextVLA: Vision-language-action model with amortized multi-frame context,” arXiv preprint arXiv:2510.04246, 2025.

[10] H. Shi et al., “MemoryVLA: Perceptual-cognitive memory in visionlanguage-action models for robotic manipulation,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[11] Z. Wang et al., “NativeMEM: Native memory compression for longhorizon robotic manipulation,” in Proceedings of the Conference on Robot Learning (CoRL), 2026.

[12] Y. Dai et al., “RoboMME: Benchmarking and understanding memory for robotic generalist policies,” in Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.

[13] Y. Zeng et al., “KEMO: Event-driven keyframe memory for longhorizon robot manipulation with VLA policies,” arXiv preprint arXiv:2606.23589, 2026.

[14] G. Yang et al., “EventVLA: Event-driven visual evidence memory for long-horizon vision-language-action policies,” in Proceedings of the Conference on Robot Learning (CoRL), 2026.

[15] J. M. Zacks et al., “Event perception: A mind-brain perspective,” Psychological Bulletin, vol. 133, no. 2, p. 273, 2007.

[16] N. Carion et al., “SAM 3: Segment anything with concepts,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[17] C. Sferrazza et al., “HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation,” in Robotics: Science and Systems (RSS), 2024.

[18] S. Wei et al., “SIMPLE: Simulation-based policy learning and evaluation for humanoid loco-manipulation,” in Proceedings of the Conference on Robot Learning (CoRL), 2026.

[19] T. Wang et al., “HumanoidArena: Benchmarking egocentric hierarchical whole-body learning,” arXiv preprint arXiv:2606.17833, 2026.

[20] K. Lin et al., “HumanoidMimicGen: Data generation for loco-manipulation via whole-body planning,” arXiv preprint arXiv:2605.27724, 2026.

[21] H. Lei et al., “RoboMemArena: A comprehensive and challenging robotic memory benchmark,” arXiv preprint arXiv:2605.10921, 2026.

[22] T. He et al., “HOVER: Versatile neural whole-body controller for humanoid robots,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 9989–9996.

[23] Y. Ze et al., “TWIST: Teleoperated whole-body imitation system,” in Proceedings of the 9th Conference on Robot Learning (CoRL), ser. Proceedings of Machine Learning Research, vol. 305. PMLR, 2025, pp. 2143–2154.

[24] K. Black et al., “π<sub>0.5</sub>: a vision-language-action model with openworld generalization,” in Proceedings of the 9th Conference on Robot Learning (CoRL), ser. Proceedings of Machine Learning Research, vol. 305. PMLR, 2025, pp. 17–40.

[25] J. Yu et al., “EgoMI: Learning active vision and whole-body manipulation from egocentric human demonstrations,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2026.

[26] P. Li et al., “Spatial memory for out-of-vision manipulation in visionlanguage-action,” in Proceedings of the 43rd International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 306. PMLR, 2026.

[27] P. Ren et al., “Closing the loop in humanoid VLA: Persistent 3D object tokens for verifiable loco-manipulation,” arXiv preprint arXiv:2607.18016, 2026.

[28] H. Fang et al., “SAM2Act: Integrating visual foundation model with a memory architecture for robotic manipulation,” in Proceedings of the 42nd International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 15 925–15 942.

[29] A. Sridhar et al., “MemER: Scaling up memory for robot control via experience retrieval,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[30] A.-C. Cheng et al., “NaVILA: Legged robot vision-language-action model for navigation,” in Robotics: Science and Systems (RSS), 2025.

[31] N. Chung et al., “Rethinking progression of memory state in robotic manipulation: An object-centric perspective,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, 2026, pp. 3407– 3415.

[32] X. Chen et al., “InternVLA-M1: A spatially guided vision-languageaction framework for generalist robot policy,” arXiv preprint arXiv:2510.13778, 2025.

[33] C. Chi et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems (RSS), 2023.

[34] T. Z. Zhao et al., “Learning fine-grained bimanual manipulation with low-cost hardware,” in Robotics: Science and Systems (RSS), 2023.

[35] R. Yang et al., “EgoVLA: Learning vision-language-action models from egocentric human videos,” arXiv preprint arXiv:2507.12440, 2025.

[36] C. Chen et al., “3D-CovDiffusion: 3D-aware diffusion policy for coverage path planning,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.

[37] Y. Lipman et al., “Flow matching for generative modeling,” in The Eleventh International Conference on Learning Representations (ICLR), 2023.

[38] E. Perez et al., “FiLM: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, 2018.

[39] S. Bai et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[40] K. Black, M. Y. Galliker, and S. Levine, “Real-time execution of action chunking flow policies,” in Advances in Neural Information Processing Systems (NeurIPS), 2025.