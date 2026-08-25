# ParallelWorld: Test-Time Scaling for Embodied Reasoning

Min Chen<sup>1,∗</sup>, Shengjun Zhang<sup>1,∗,†</sup>, Yuxin Li<sup>1</sup>, Zhang Zhang<sup>1</sup>, Xin Fei<sup>2</sup>, Chong Xia<sup>1</sup>, Yueqi Duan<sup>1</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>National University of Singapore

## Abstract

Embodied Reasoning constitutes a fundamental capability of embodied intelligence, serving as the basis for autonomous perception, reasoning, and interaction within physical environments. Recent studies have shifted the paradigm of embodied reasoning from static perception toward dynamic exploration, where agents acquire task-relevant information through interactions with the environment. However, existing active reasoning approaches generally generate exploration trajectories incrementally without long-horizon planning. Even recently emerged test-time scaling frameworks often resort to myopic, single-step lookaheads, which struggle to resolve the delayed feedback inherent in complex, occluded spatial environments. To address this limitation, we propose ParallelWorld, a multi-horizon test-time scaling framework for embodied reasoning. Instead of greedy, single-step trials, ParallelWorld empowers agents to simulate and evaluate multi-step future trajectories in parallel before committing to an action. Specifically, we introduce a verifier-guided tree-search paradigm. Starting from the current state, ParallelWorld branches into multiple parallel trajectories and rolls them out continuously across a multi-step horizon. At each simulation step, a verifier agent evaluates the intermediate state transitions, dynamically pruning unpromising branches and prioritizing paths with the highest information gain. Once the multi-step prospective simulation is complete, the agent synthesizes the long-horizon outcomes to commit to the optimal action sequence. Finally, an answer agent performs reasoning over the selected trajectory to produce the final reasoning. Extensive experiments on ESI-Bench demonstrate that ParallelWorld consistently improves active perception and reasoning performance.

Project Page: https://chen-min-22.github.io/ParallelWorld-page/

## 1 Introduction

Understanding and reasoning about the physical world is a fundamental capability of embodied agents [5, 21, 35]. Unlike conventional systems that reason primarily from fixed observations, embodied agents must build a goaldirected understanding of the physical world through interactions with their environment. Recent advances in multimodal foundation models [8, 16] have substantially improved the reasoning capabilities of embodied agents, enabling them to address increasingly complex tasks in simulated and real-world environments. However, achieving reliable embodied understanding remains challenging, as agents often need to make decisions under incomplete observations, ambiguous visual evidence, and dynamically changing environments.

To address these challenges, recent studies have shifted embodied understanding from passive perception toward active exploration, where agents are allowed to interact with the environment and gather additional information before making decisions [14, 23, 39, 40, 42, 43]. Rather than relying solely on existing observations, these approaches formulate understanding as an interactive process involving perception, action, and reasoning. By actively selecting informative observations or executing task-related actions, embodied agents can progressively improve their understanding of the surrounding environment.

Despite these progresses, existing active understanding approaches typically generate exploration behaviors in a sequential manner, where each action is selected based only on the current observation and previously acquired information. Such a formulation lacks explicit reasoning over alternative future trajectories and prevents agents from evaluating the potential outcomes of diferent actions before execution. In contrast, humans often consider multiple possible consequences and mentally simulate alternative strategies before taking actions. This raises an important question: can embodied agents similarly leverage future world simulation to evaluate possible exploration paths and make more efective decisions?

In this work, we propose ParallelWorld, a multi-horizon test-time scaling framework for active embodied reasoning. Starting from a restorable simulator state, ParallelWorld expands every retained branch with executable camera and task-dependent physical actions, and renders their prospective visual outcomes. A verifier agent evaluates the resulting frontier and prunes uninformative branches using a predefined branchwidth schedule, which preserves multiple hypotheses during expansion and converges toward the most promising trajectory at later steps. The selected states are then replayed and expanded in the next iteration. Since physical execution cannot simultaneously follow alternative trajectories or roll back to previous states, an answer agent reasons only over the highest-ranked root-to-leaf route, producing the final prediction when its confidence becomes suficiently high or the exploration budget is exhausted. In this way, ParallelWorld enables embodied agents to move beyond reactive single-trajectory exploration and make decisions using question-relevant evidence from simulated future worlds.

Extensive experiments on embodied understanding benchmarks demonstrate that ParallelWorld significantly improves the capability of embodied agents in active reasoning. Our results validate the efectiveness of future world simulation as a mechanism for enhancing embodied understanding and provide a new perspective toward building more capable autonomous agents. Our main contributions can be summarized as follows:

• We propose ParallelWorld, an multi-horizon test-time scaling framework that enables agents to explore potential future trajectories through simulated world interactions before execution.

• We introduce a verifier-guided exploration mechanism with a verifier agent and an answer agent, where the former evaluates candidate trajectories and the latter performs reasoning over selected evidence for final decision-making.

• Experimental results on embodied understanding benchmarks demonstrate that our method improves active perception and reasoning performance, validating the efectiveness of future trajectory simulation for embodied agents.

## 2 Related Work

## 2.1 Spatial Reasoning

Spatial reasoning has evolved from geometric estimation and explicit scene representations to MLLM-based spatial question answering. Existing methods can be broadly divided into two categories. Geometry-enhanced approaches introduce depth supervision, 3D annotations, scene graphs, or reconstruction objectives to improve spatial grounding[3, 4, 10, 31]. Reasoning-enhanced approaches instead ground inference in explicit spatial coordinates, intermediate reasoning chains, or visual drawings[20, 32]. More recent studies extend spatial reasoning and evaluation from static observations to remembered spaces, real-world videos, and dynamic 4D scenes[15, 29, 34]. Despite these advances, most existing methods reason over fixed observations and therefore remain limited in their ability to actively acquire task-relevant spatial evidence.

![](images/d55be506cf1e7c45e48ed7b4cc6f87b66da251038fc6fc88e815e80b7612c70d.jpg)  
Figure 1 Overview of ParallelWorld. Given a question, an initial observation, and the corresponding simulator state, ParallelWorld enumerates executable camera and physical actions from each retained world and renders their prospective outcomes. A verifier ranks the candidate worlds and applies a predefined branch-width schedule, retaining multiple branches at odd steps and converging to one branch at even steps. The selected simulator states are replayed and expanded in the next iteration. At each checkpoint, the answer agent reasons over the verifier’s top-1 root-to-leaf route and predicts the answer and confidence, terminating the exploration when the prediction is suficiently confident or the exploration budget is exhausted.

## 2.2 Active Embodied Reasoning

Active perception formulates sensing as a decision process in which an agent selects interactions to reduce uncertainty about task-relevant variables[1, 2, 17]. Embodied question answering further instantiates this principle by requiring agents to interact with an environment before producing an answer[9, 21]. Recent embodied agents extend this paradigm through confidence-aware exploration, goal-conditioned perception, multimodal tool use, spatial-belief construction, and retrospective trajectory selection[23, 39, 40, 42, 43]. ESI-Bench evaluates active embodied reasoning across diverse camera-motion and physical-manipulation tasks[14]. However, existing methods generally acquire evidence along a single realized trajectory, selecting each action from the current observation history. ParallelWorld difers from these methods by constructing multiple prospective trajectories and using a verifier to retain informative branches.

## 2.3 Test-Time Scaling

Test-time scaling improves model performance by allocating additional inference computation to repeated sampling, verification, and structured search[7, 19, 28, 33, 37]. In agentic settings, language-model reasoning has been combined with environment feedback and action-level tree search[41], while compute-optimal allocation of test-time resources has also been studied more generally[26].

A complementary line of work uses explicit or learned models of the environment to evaluate prospective outcomes. Learned dynamics support imagination-based control and long-horizon planning[6, 11–13, 24], whereas classical approaches perform lookahead through POMDP inference, Monte Carlo tree search, or model-predictive control[18, 25, 30]. Recent multimodal systems further apply visual world models to spatial reasoning, exploration, and robotic action refinement[22, 27, 36, 38]. ParallelWorld is complementary to learned world-model approaches: the present implementation isolates verifier-guided search under an exact, restorable simulator transition, while replacing this transition with a learned world model remains an important direction for real-world deployment.

## 3 Methods

## 3.1 Overview

ParallelWorld is an active test-time scaling framework that improves embodied reasoning by imagining multiple possible exploration trajectories before committing to a particular strategy. Instead of generating a single action sequence autoregressively, our method constructs a tree of future worlds. At each exploration step, all retained worlds are expanded, and a verifier agent selects the most informative branches. An answer agent then reasons over the accumulated evidence from the selected branches and determines whether further exploration is necessary. The overall framework is illustrated in Figure 1.

Given a question $q ,$ an initial environment state $s _ { 0 } .$ and an initial observation $o _ { 0 }$ , the objective is to predict an answer y through active interaction. We define the executable action space as

$$
\mathcal { A } = \mathcal { A } _ { \mathrm { { c a m } } } \cup \mathcal { A } _ { \mathrm { { t a s k } } } ,
$$

where $A _ { \mathrm { { c a m } } }$ contains camera translations and rotations, while $\mathcal { A } _ { \mathrm { t a s k } }$ contains task-dependent physical interactions, such as picking, placing, pouring, and stacking.

At step t, ParallelWorld maintains a set of retained trajectories

$$
\mathcal { B } _ { t } = \left\{ \tau _ { t } ^ { ( 1 ) } , \dots , \tau _ { t } ^ { ( K _ { t } ) } \right\} , \qquad \tau _ { t } ^ { ( i ) } = \left( a _ { 1 } ^ { ( i ) } , \dots , a _ { t } ^ { ( i ) } \right) ,
$$

where $K _ { t }$ denotes the branch retention width. Each trajectory corresponds to an independently restorable simulated world.

## 3.2 Prospective World Expansion

A simulated branch must preserve both geometric and task-dependent information. We therefore represent its runtime state as

$$
\xi _ { t } = ( s _ { t } , z _ { t } , p _ { t } , \omega _ { t } ) ,
$$

where $s _ { t }$ is the simulator state, $z _ { t }$ is the task-specific interaction state, $p _ { t }$ is the camera pose, and $\omega _ { t }$ records the states of relevant objects. The root branch is initialized as

$$
B _ { 0 } = \left\{ \left( \tau _ { 0 } , \xi _ { 0 } \right) \right\} , \qquad \tau _ { 0 } = \varnothing .
$$

At exploration step $t ,$ each retained branch is expanded with every executable action. For a parent branch i and action $a \in { \mathcal { A } } .$ , the prospective state and its corresponding observation are computed as

$$
\begin{array} { r } { \xi _ { t } ^ { ( i , a ) } = \mathcal { T } \left( \xi _ { t - 1 } ^ { ( i ) } , a \right) , \qquad o _ { t } ^ { ( i , a ) } = \mathcal { O } \left( \xi _ { t } ^ { ( i , a ) } \right) , } \end{array}
$$

where $\tau$ denotes the simulator transition and O denotes visual rendering. The complete candidate frontier is

$$
\begin{array} { r } { \mathcal { C } _ { t } = \left\{ \left( \tau _ { t - 1 } ^ { ( i ) } \oplus a , \xi _ { t } ^ { ( i , a ) } , o _ { t } ^ { ( i , a ) } \right) ~ \bigg | ~ \tau _ { t - 1 } ^ { ( i ) } \in \mathcal { B } _ { t - 1 } , ~ a \in \mathcal { A } \right\} . } \end{array}
$$

Consequently, the number of simulated future worlds at step t is

$$
| { \mathcal { C } } _ { t } | = | { \boldsymbol { B } } _ { t - 1 } | \cdot | { \boldsymbol { A } } | .
$$

Before simulating a candidate, its parent state is restored so that every action is evaluated from the same parent world. This operation is particularly important for physical actions, since object poses, grasp constraints, and task-specific variables may be modified during interaction. For actions that produce informative intermediate states, we additionally retain their intermediate visual frames. Thus, the evidence associated with a candidate c is

$$
e ( c ) = \left\{ o ( c ) , o _ { 1 } ^ { \mathrm { a c t } } ( c ) , \dots , o _ { m _ { c } } ^ { \mathrm { a c t } } ( c ) \right\} ,
$$

where $o ( c )$ is the post-action observation and $m _ { c }$ is the number of available intermediate frames.

## 3.3 Verifier-Guided Exploration

Directly retaining the entire frontier would cause the number of trajectories to grow exponentially. Parallel-World controls this growth using a verifier agent $V _ { \phi } .$ which evaluates the potential usefulness of each candidate for answering q. Its input consists of the question, the reference observations, the previous exploration history, and the visual evidence from the current candidate frontier:

$$
v _ { c } = V _ { \phi } \left( q , \mathcal { T } _ { \mathrm { r e f } } , \mathcal { H } _ { t - 1 } , \tau _ { c } , e ( c ) \right) , \qquad c \in \mathcal { C } _ { t } .
$$

Here, $v _ { c }$ denotes the verifier’s implicit assessment of the evidence quality of candidate c. The verifier favors branches that reveal task-relevant objects or relations, resolve visual ambiguity, and provide complementary rather than duplicated observations.

The retained branch set is written as

$$
\mathcal { B } _ { t } = \mathrm { K } _ { c \in \mathcal { C } _ { t } } \left( v _ { c } , K _ { t } \right) , \qquad | \mathcal { B } _ { t } | = \mathrm { m i n } ( K _ { t } , | \mathcal { C } _ { t } | ) .
$$

In practice, $V _ { \phi }$ directly outputs an ordered set of branch identifiers, and the above equation represents this set-valued selection process.

The retention width is determined a predefined branch-width strategy:

$$
K _ { t } = g ( t , q ) ,
$$

where g follows an alternating expansion-and-convergence strategy and assigns diferent retention widths according to the task category. Odd-numbered steps retain multiple branches to preserve diverse future hypotheses, whereas even-numbered steps set $K _ { t } = 1$ to consolidate the search around the most informative trajectory. Larger widths are used during early expansion, followed by a smaller stable width at greater depths.

Candidate simulations are initially used only for comparison. After verification, the selected actions are replayed from their parent snapshots:

$$
\widehat { \xi } _ { t } ^ { ( i ) } = \mathcal { T } \left( \xi _ { t - 1 } ^ { \pi ( i ) } , a _ { t } ^ { ( i ) } \right) , \qquad \tau _ { t } ^ { ( i ) } \in \mathcal { B } _ { t } ,
$$

where $\pi ( i )$ denotes the parent of branch i. Only these selected states are stored for the next expansion step. This replay operation ensures that the retained images, object states, and simulator states remain mutually consistent.

## 3.4 Top-1 Route Answering

In practical deployment, a robot can execute only one action sequence at a time and cannot simultaneously perform alternative physical interactions or roll back to a previous world state. We therefore distinguish internal prospective search from committed execution: the verifier may retain multiple simulator branches for future exploration, but the answer agent receives evidence from only one root-to-leaf route. This top-1 route corresponds to the single trajectory that would ultimately be executed by the robot and avoids combining observations from mutually incompatible worlds.

Although the verifier retains $K _ { t }$ branches for internal exploration, only the highest-ranked route is provided to the answer agent. We treat the verifier output as an ordered branch set

$$
\begin{array} { r } { \boldsymbol { B _ { t } } = \left( b _ { t } ^ { [ 1 ] } , b _ { t } ^ { [ 2 ] } , \dots , b _ { t } ^ { [ K _ { t } ] } \right) , } \end{array}
$$

where $b _ { t } ^ { [ 1 ] }$ denotes the top-ranked branch at step t. The remaining branches are preserved in the simulator and may be expanded in subsequent steps, but they do not contribute evidence to the current answer prediction.

Let ${ \mathcal { P } } ( b )$ denote the parent of branch b. Starting from the top-ranked branch $b _ { t } ^ { [ 1 ] }$ , we recover its complete ancestor chain:

$$
b _ { t , j } ^ { * } = \mathcal { P } ^ { t - j } \left( b _ { t } ^ { [ 1 ] } \right) , \qquad j \in \{ 1 , \dotsc , t \} .
$$

The top-1 trajectory at checkpoint t is therefore

$$
\rho _ { t } ^ { * } = \left( b _ { t , 1 } ^ { * } , b _ { t , 2 } ^ { * } , \ldots , b _ { t , t } ^ { * } \right) , \qquad b _ { t , t } ^ { * } = b _ { t } ^ { [ 1 ] } .
$$

By construction, all elements of $\rho _ { t } ^ { * }$ belong to the same root-to-leaf path:

$$
\mathcal { P } \left( b _ { t , j } ^ { * } \right) = b _ { t , j - 1 } ^ { * } , \qquad j = 2 , \ldots , t .
$$

The answer evidence contains one post-action observation from every depth along this route:

$$
\mathcal { E } _ { t } ^ { \mathrm { t o p 1 } } = \left\{ o \left( b _ { t , j } ^ { * } \right) \right\} _ { j = 1 } ^ { t } .
$$

Reference observations $\mathcal { T } _ { \mathrm { r e f } }$ are appended when available. Intermediate action frames may be used by the verifier to evaluate candidate outcomes, whereas the answer agent receives only the post-action observations along $\rho _ { t } ^ { * }$ . Thus, evidence from other selected branches is excluded:

$$
o ( b ) \notin \mathcal { E } _ { t } ^ { \mathrm { t o p 1 } } , \qquad \forall b \in \mathcal { B } _ { j } \setminus \left\{ b _ { t , j } ^ { * } \right\} .
$$

At each checkpoint, the answer agent predicts

$$
\left( \hat { y } _ { t } , c _ { t } , r _ { t } \right) = A _ { \theta } \left( q , \mathcal { T } _ { \mathrm { r e f } } , \mathcal { E } _ { t } ^ { \mathrm { t o p 1 } } \right) ,
$$

where $\hat { y } _ { t }$ is the predicted answer, $c _ { t } \in [ 0 , 1 ]$ is the confidence, and $r _ { t }$ is the corresponding reasoning. Because the top-ranked leaf may originate from a branch that was not ranked first at an earlier step, $\rho _ { t } ^ { * }$ is reconstructed from the current leaf rather than by concatenating the independently top-ranked branch from every checkpoint. This guarantees that all observations given to the answer agent correspond to one physically consistent trajectory.

We define a deterministic validity indicator to distinguish admissible, conclusive predictions from malformed or unresolved responses. Let $\mathcal { V } ( q )$ denote the task-specific answer space for question $q ,$ and let $y _ { \perp }$ denote invalid or abstaining responses, such as “unknown,” “uncertain,” or outputs that cannot be parsed into the required answer format. We define

$$
\mathrm { V a l i d } ( \hat { y } _ { t } ; q ) = \mathbb { I } \left[ \hat { y } _ { t } \in \mathcal { V } ( q ) \setminus \mathcal { V } _ { \perp } \right] ,
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. This validity check is implemented as a deterministic parser rather than an additional model call.

Exploration terminates when the predicted answer is valid and its confidence exceeds a threshold $\gamma \colon$

$$
c _ { t } > \gamma \quad \land \quad \mathrm { V a l i d } ( \hat { y } _ { t } ; q ) = 1 .
$$

Given a maximum exploration horizon $L ,$ the stopping checkpoint is

$$
T = \operatorname* { m i n } \left( \{ t \in \{ 1 , \dots , L \} : c _ { t } > \gamma \wedge \mathrm { V a l i d } ( \hat { y } _ { t } ; q ) = 1 \} \cup \{ L \} \right) .
$$

If no earlier checkpoint satisfies the stopping condition, then $T = L$ by construction. At the stopping checkpoint, the answer agent produces

$$
\left( \hat { y } _ { T } , c _ { T } , r _ { T } \right) = A _ { \theta } \left( q , \mathcal { T } _ { \mathrm { r e f } } , \mathcal { E } _ { T } ^ { \mathrm { t o p 1 } } \right) ,
$$

and the final prediction is

$$
\hat { y } = \hat { y } _ { T } .
$$

Overall, ParallelWorld implements verifier-guided tree search over simulated environment states. Prospective expansion explores alternative futures, verifier-based pruning concentrates computation on informative trajectories, and top-1 route answering converts the selected visual evidence into the final answer.

Table 1 Quantitative results on ESI-Bench [14]. We report accuracy across Passive Single-View, Active Exploration, ParallelWorld. All active methods use the same answer-model backbone.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Subcategory</td><td colspan="3">Agent Evaluation</td></tr><tr><td>Passive</td><td>Active</td><td>ParallelWorld</td></tr><tr><td rowspan="3">Perceptual Grounding</td><td>View Hallucination</td><td>53.32%</td><td>66.82%</td><td>71.40%</td></tr><tr><td>Partial Occlusion</td><td>36.84%</td><td>57.89%</td><td>77.89%</td></tr><tr><td>Material Transparency</td><td>69.72%</td><td>71.10%</td><td>73.85%</td></tr><tr><td rowspan="2">Physical Structure</td><td>Rigid Containment</td><td>55.00%</td><td>60.00%</td><td>80.00%</td></tr><tr><td>Deformable Objects</td><td>37.76%</td><td>40.82%</td><td>42.86%</td></tr><tr><td rowspan="2">Physical Dynamics</td><td>Inclined Plane</td><td>59.02%</td><td>83.61%</td><td>85.25%</td></tr><tr><td>Stacking  Stability</td><td>1.12%</td><td>45.98%</td><td>53.95%</td></tr><tr><td rowspan="3">Specular Reflection</td><td>Reflection Authoring</td><td>56.57%</td><td>51.52%</td><td>60.61%</td></tr><tr><td>Spatial Relations</td><td>38.94%</td><td>36.28%</td><td>51.33%</td></tr><tr><td>Correspondence</td><td>42.05%</td><td>47.73%</td><td>48.86%</td></tr><tr><td rowspan="3">Spatial Relations</td><td>Linear Alignment</td><td>52.13%</td><td>68.09%</td><td>70.21%</td></tr><tr><td>Geometric Configuration</td><td>16.55%</td><td>17.96%</td><td>18.31%</td></tr><tr><td>Physical Contact</td><td>66.39%</td><td>68.07%</td><td>71.43%</td></tr><tr><td rowspan="2">Metric Comparison</td><td>Dimensional Size</td><td>39.52% 61.18%</td><td>44.91%</td><td>49.10%</td></tr><tr><td>Spatial Distance</td><td></td><td>58.55%</td><td>67.11%</td></tr><tr><td rowspan="4">Cognitive Mapping</td><td>Connectivity</td><td>66.67%</td><td>55.00%</td><td>60.00%</td></tr><tr><td>Traversable Passage</td><td>68.33%</td><td>65.00%</td><td>75.00%</td></tr><tr><td>Regional Boundary Long-Term Navigation</td><td>45.00%</td><td>57.50%</td><td>63.75%</td></tr><tr><td></td><td>16.67%</td><td>16.67%</td><td>18.33%</td></tr><tr><td rowspan="6">Enumerative Perception</td><td>Counting with Occlusion</td><td>23.33%</td><td>26.67%</td><td>30.00%</td></tr><tr><td>Spatial Segmentation</td><td>20.00%</td><td>20.00%</td><td>23.33%</td></tr><tr><td>Merged Observation</td><td>0.00%</td><td>1.67%</td><td>6.67%</td></tr><tr><td>Category Ambiguity</td><td>0.00%</td><td>8.33%</td><td>13.33%</td></tr><tr><td>Structural Enclosure</td><td>2.50%</td><td>5.00%</td><td>7.50%</td></tr><tr><td>Illumination Variability</td><td>0.00%</td><td>6.00%</td><td>10.00%</td></tr><tr><td rowspan="2">Temporal Understanding</td><td>Unobserved Change</td><td>38.62%</td><td>70.95%</td><td>89.86%</td></tr><tr><td>Agent Observation</td><td>34.59%</td><td>48.87%</td><td>62.41%</td></tr><tr><td>Action Sequencing</td><td>Action Order Inference</td><td>37.66%</td><td>53.25%</td><td>61.04%</td></tr></table>

## 4 Experiments

We evaluate ParallelWorld on ESI-Bench [14] to investigate the following questions:

1. Does prospective world simulation improve embodied reasoning over sequential exploration?

2. How does the number of future worlds retained afect reasoning performance and test-time computation?

We first describe the experimental setup in section 4.1 and report quantitative results across all ESI-Bench [14] task categories in section 4.2. We then present a qualitative example illustrating how the verifier identifies informative future worlds in section 4.3. Finally, we conduct ablation study on the verifier retention width of our framework in section 4.4

## 4.1 Experimental Setup

Benchmark. We conduct experiments on ESI-Bench [14], a comprehensive benchmark for embodied spatial intelligence. The benchmark contains 10 task categories and 29 subcategories. Due to an environment issue that prevents the execution of pouring actions, we exclude the Liquid Volume subcategory and evaluate the remaining 28 subcategories. These tasks require agents to actively acquire information through camera movement, physical interaction, or both before producing an answer.

Baselines. We selected two paradigms from ESI-Bench [14] as baselines, including Passive Single-View, which predicts the answer from a single observation at the initial pose, and Active Exploration, which sequentially selects camera or physical actions to gather task-relevant evidence before answering.

![](images/788c59c23fcd0545590d21a6f42620396accc6752e46a8b51c4704875b6fbb23.jpg)  
Figure 2 Qualitative analysis of verifier-guided exploration. We present a illustrative qualitative comparison on the exploration process of Active Exploration and ParallelWorld on an ESI-Bench [14] task. Gray views indicate unselected candidate branches, whereas colored views indicate branches retained by the verifier.

Implementation details. We use GPT-5.4 as the answer agent and GPT-5.5 as the verifier agent. By default, the maximum exploration depth is set to L = 15, and the confidence threshold for adaptive stopping is set to γ = 0.8. The verifier is constrained to return candidate route identifiers, while the answer agent outputs the task answer, confidence, and reasoning.

## 4.2 Main Results

Table 1 summarizes the quantitative results across the 28 evaluated ESI-Bench subcategories. ParallelWorld consistently outperforms the conventional Active Exploration baseline under the same answer-model backbone. The improvements are particularly pronounced for Temporal Understanding, where the accuracy on Unobserved Change increases from 70.95% to 89.86%, and for Metric Comparison, where Spatial Distance improves from 58.55% to 67.11%. Although Passive Single-View performs best on Connectivity due to the task setting, ParallelWorld still improves over Active Exploration from 55.00% to 60.00%. Overall, these results demonstrate the efectiveness of prospective simulation for active embodied reasoning.

## 4.3 Qualitative Analysis

As shown in Figure 2, the sequential active exploration baseline follows a single trajectory whose observations remain ambiguous and eventually predicts an incorrect answer. In contrast, ParallelWorld evaluates multiple candidate actions in simulated future worlds, as illustrated by the stacked views at each step. The verifier selects trajectories that progressively expose the relative positions of the lamp and projector. The selected evidence clearly indicates that the gap of the two objects, enabling the answer agent to produce the correct prediction. This example demonstrates how prospective simulation and verifier-guided selection improve evidence acquisition over single-trajectory exploration.

## 4.4 Ablation Studies

As illustrated in Table 2, we compare fixed K = 2, 3, 4 search with our predefine K schedule. In our schedule setting, the verifier retains a wider set of branches during expansion steps and progressively converges to one branche at later steps, with the schedule adjusted according to the task category. This design balances trajectory diversity ,search concentration and time consumption.

Table 2 Ablation study on verifier branch width. We report accuracy (%), average runtime per question (s), and the mean number of exploration steps under fixed branch width K and the predefined K schedule settings. All configurations use the same answer model, maximum exploration depth, and stopping threshold.
<table><tr><td rowspan="2">Subcategory</td><td colspan="3">K = 2</td><td colspan="3">K = 3</td><td colspan="3">K = 4</td><td colspan="3">K Schedule</td></tr><tr><td>Acc.</td><td>Time</td><td>Steps</td><td>Acc.</td><td>Time</td><td>Steps</td><td>Acc.</td><td>Time</td><td>Steps</td><td>Acc.</td><td>Time</td><td>Steps</td></tr><tr><td>Rigid Containment</td><td>55.00</td><td>4246.8</td><td>5.65</td><td>40.00</td><td>4091.8</td><td>5.15</td><td>57.89</td><td>3964.3</td><td>4.95</td><td>80.00</td><td>1542.6</td><td>4.20</td></tr><tr><td>Partial Occlusion</td><td>56.84</td><td>192.4</td><td>1.15</td><td>49.47</td><td>168.6</td><td>1.08</td><td>55.79</td><td>178.0</td><td>1.08</td><td>63.16</td><td>158.3</td><td>1.09</td></tr><tr><td>Regional Boundary</td><td>66.15</td><td>285.0</td><td>2.20</td><td>65.15</td><td>486.2</td><td>2.56</td><td>56.41</td><td>315.5</td><td>1.97</td><td>61.25</td><td>290.5</td><td>3.72</td></tr><tr><td>Counting with Occlusion</td><td>23.33</td><td>230.8</td><td>1.63</td><td>23.33</td><td>456.7</td><td>3.33</td><td>23.33</td><td>553.5</td><td>3.33</td><td>30.00</td><td>325.2</td><td>3.80</td></tr><tr><td>Average</td><td>55.52</td><td>590.8</td><td>1.99</td><td>50.72</td><td>668.7</td><td>2.27</td><td>51.87</td><td>613.5</td><td>2.04</td><td>59.56</td><td>350.6</td><td>2.66</td></tr></table>

The schedule strategy achieves the best average accuracy of 59.56%, compared with 55.52%, 50.72%, and 51.87% for $K = 2 , 3 , 4$ , respectively. It also has the lowest average runtime (350.6 s per question), despite using a slightly larger average number of exploration steps. The results also show that increasing the fixed branch width does not consistently improve accuracy, since retaining additional branches may introduce redundant or less informative candidates. Our schedule setting does not outperform every fixed setting on every task; for example, K = 2 and K = 3 perform better on Regional Boundary. Nevertheless, it provides the strongest overall accuracy-eficiency trade-of, achieving the best mean performance across the evaluated subcategories.

## 5 Conclusion

We presented ParallelWorld, an active test-time scaling framework for embodied reasoning. ParallelWorld constructs multiple future worlds through simulated interactions, uses a verifier agent to retain informative exploration trajectories, and employs an answer agent to reason over the selected evidence. Experiments on ESI-Bench show consistent improvements over sequential Active Exploration across the evaluated task categories, with particularly clear gains in temporal understanding and metric comparison. These results demonstrate that prospective world simulation provides an efective mechanism for improving active embodied reasoning.

Limitations and Future Work. ParallelWorld incurs additional test-time computation because each retained world must be expanded over the executable action space. This cost can become substantial for tasks with many physical actions or long exploration horizons. Moreover, the quality of exploration depends on both simulator fidelity and verifier reliability, while retaining only selected trajectories may discard complementary evidence from alternative branches. Future work could investigate learned branch-pruning and value estimation, uncertainty-aware verifier models, and hierarchical action proposal strategies to improve the eficiency and robustness of prospective exploration. Extending ParallelWorld to real-world robots and dynamic environments is another important direction.

## References

[1] John Aloimonos, Isaac Weiss, and Amit Bandyopadhyay. Active vision. International Journal of Computer Vision, 1(4):333–356, 1988.

[2] Ruzena Bajcsy. Active perception. Proceedings of the IEEE, 76(8):966–1005, 1988.

[3] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.

[4] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. SpatialRGPT: Grounded spatial reasoning in vision-language models. In Advances in Neural Information Processing Systems, volume 37, pages 135062–135093, 2024.

[5] Zhili Cheng, Ran Li, Jinyi Hu, Yuge Tu, Shiqi Dai, Shengding Hu, Yang Shi, Lei Shi, and Maosong Sun. Embodiedeval: Evaluate multimodal llms as embodied agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11420–11432, 2026.

[6] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Advances in Neural Information Processing Systems, volume 31, pages 4754–4765, 2018.

[7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.

[8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[9] Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–10, 2018.

[10] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Peihao Wang, Huaizhi Qu, Shijie Zhou, et al. VLM-3R: Vision-language models augmented with instruction-aligned 3D reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 31054–31065, 2026.

[11] David Ha and Jürgen Schmidhuber. World models, 2018.

[12] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020.

[13] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025.

[14] Yining Hong, Jiageng Liu, Han Yin, Manling Li, Leonidas Guibas, Fei-Fei Li, Jiajun Wu, and Yejin Choi. ESI-Bench: Towards embodied spatial intelligence that closes the perception-action loop, 2026.

[15] Yuzhi Huang, Kairun Wen, Rongxin Gao, Dongxuan Liu, Yibin Lou, Jie Wu, Jing Xu, Jian Zhang, Zheng Yang, Yunlong Lin, et al. Thinking in dynamics: How multimodal large language models perceive, track, and reason dynamics in the physical 4D world. arXiv preprint arXiv:2603.12746, 2026.

[16] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[17] Stefan Isler, Reza Sabzevari, Jefrey Delmerico, and Davide Scaramuzza. An information gain formulation for active volumetric 3d reconstruction. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 3477–3484, 2016.

[18] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1–2):99–134, 1998.

[19] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024.

[20] Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yaochen Hu, Lingfeng Zhang, Yingxue Zhang, Shuang Wu, Tongtong Cao, Guowei Huang, et al. SpatialCoT: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning. arXiv preprint arXiv:2501.10074, 2025.

[21] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaf, Sneha Silwal, Paul McVay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Sasha Sax, and Aravind Rajeswaran. OpenEQA: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16488–16498, 2024.

[22] Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Gokhan Tur, Dilek Hakkani-Tür, Yunzhu Li, and Heng Ji. Current agents fail to leverage world model as tool for foresight. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13686–13723, San Diego, California, United States, July 2026. Association for Computational Linguistics.

[23] Allen Z. Ren, Jaden Clark, Anushri Dixit, Masha Itkina, Anirudha Majumdar, and Dorsa Sadigh. Explore until confident: Eficient exploration for embodied question answering, 2024.

[24] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.

[25] David Silver and Joel Veness. Monte-carlo planning in large POMDPs. In Advances in Neural Information Processing Systems, volume 23, pages 2164–2172, 2010.

[26] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more efective than scaling parameters for reasoning. In International Conference on Learning Representations, 2025.

[27] Zhenguo Sun, Yu Sun, Hande Huang, and Alois Knoll. ω-EVA: Envision, Verify, and Act with Latent Interactive World Models, 2026.

[28] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain-of-thought reasoning in language models. In International Conference on Learning Representations, 2023.

[29] Kairun Wen, Runyu Chen, Hui Zheng, Yunlong Lin, Panwang Pan, Chenxin Li, Wenyan Cong, Jian Zhang, Junbin Lu, Chenguo Lin, et al. Dynamicverse: A physically-aware multimodal framework for 4d world modeling. In Advances in Neural Information Processing Systems, volume 38, pages 108604–108634, 2026.

[30] Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, and Evangelos A. Theodorou. Information theoretic MPC for model-based reinforcement learning. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pages 1714–1721, 2017.

[31] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-MLLM: Boosting MLLM capabilities in visualbased spatial intelligence. In Advances in Neural Information Processing Systems, volume 38, pages 13569–13597, 2025.

[32] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. In Advances in Neural Information Processing Systems, volume 38, pages 143297–143330, 2025.

[33] Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, Min-Yen Kan, Junxian He, and Qizhe Xie. Self-evaluation guided beam search for reasoning. In Advances in Neural Information Processing Systems, volume 36, pages 41618–41650, 2023.

[34] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10632–10643. IEEE, 2025.

[35] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025.

[36] Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, and Chuang Gan. MindJourney: Test-time scaling with world models for spatial reasoning. In Advances in Neural Information Processing Systems, volume 38, 2025.

[37] Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Thomas L. Grifiths, Yuan Cao, and Karthik R. Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36, pages 11809–11822, 2023.

[38] Shoubin Yu, Yue Zhang, Zun Wang, Jaehong Yoon, Huaxiu Yao, Mingyu Ding, and Mohit Bansal. When and how much to imagine: Adaptive test-time scaling with world models for visual spatial reasoning, 2026.

[39] Gengyuan Zhang, Mingcong Ding, Jingpei Wu, Ruotong Liao, and Volker Tresp. ReEXplore: Improving MLLMs for embodied exploration with contextualized retrospective experience replay, 2025.

[40] Pingyue Zhang, Zihan Huang, Yue Wang, Jieyu Zhang, Letian Xue, Zihan Wang, Qineng Wang, Keshigeyan Chandrasegaran, Ruohan Zhang, Yejin Choi, et al. Theory of space: Can foundation models construct spatial beliefs through active exploration? In The Fourteenth International Conference on Learning Representations, 2026.

[41] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 62138–62160. PMLR, 2024.

[42] Weijie Zhou, Xuantang Xiong, Yi Peng, Manli Tao, Chaoyang Zhao, Honghui Dong, Ming Tang, and Jinqiao Wang. PhysVLM-AVR: Active visual reasoning for multimodal large language models in physical environments, 2025.

[43] Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, and Chunhua Shen. Active-O3: Empowering multimodal large language models with active perception via GRPO, 2025.