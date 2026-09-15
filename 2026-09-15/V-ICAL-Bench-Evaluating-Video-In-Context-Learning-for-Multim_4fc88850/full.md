# V-ICAL Bench: Evaluating Video In-Context Learning for Multimodal Agents in Interactive Environments

Ziqian Fan<sup>2,∗</sup>, Shibo Xu<sup>1,∗</sup>, Junjie Li<sup>2,∗</sup>, Xiangyu Zhao<sup>1,∗,‡</sup>, Shengyuan Ding<sup>3</sup>, Yifan Yang<sup>4</sup>, Zhenjie Yang<sup>5</sup>, Haodong Duan<sup>6</sup>, Yue Zhou<sup>7</sup>, Zhihang Zhong<sup>1</sup>, Xue Yang<sup>1,†</sup>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>South China University of Technology, <sup>3</sup>Fudan University, <sup>4</sup>Microsoft Research Asia, <sup>5</sup>The University of Hong Kong, <sup>6</sup>The Chinese University of Hong Kong, <sup>7</sup>East China Normal University

<sup>∗</sup>Core Contributors, <sup>‡</sup>Project Lead, <sup>†</sup>Corresponding Author

## Abstract

While In-Context Learning (ICL) enables models to adapt from exemplars without parameter updates, multimodal ICL remains largely underexplored, particularly regarding video demonstrations in interactive environments. For multimodal agents, learning from videos presents unique challenges: they must translate in-context demonstrations into executable policies, ground these policies in novel visual states, and iteratively refine actions based on environmental feedback. We introduce V-ICAL, a novel benchmark designed to evaluate video-based ICL for multimodal agents. Comprising 342 interactive tasks across 37 environments, V-ICAL utilizes human-curated demonstration videos as task-specific behavioral exemplars, evaluating agents through sustained interaction from a target initialization. The benchmark seamlessly connects in-context knowledge induction with core agentic capabilities, including state grounding, temporal memory, planning, and adaptation in dynamic environments. Extensive evaluations across 19 state-of-the-art multimodal agents reveal significant limitations: the best-performing model, Seed-2.1-Pro, achieves a score of only 54.4/100, while other leading models (e.g., Gemini-3.1-Pro, GPT-5.6) fail to surpass 50, far below the human baseline of 83.6. Controlled comparisons further demonstrate that current agents struggle to reliably translate video exemplars into efective policies, failing to yield consistent performance gains. Ultimately, V-ICAL exposes a critical gap in the ICL capabilities of multimodal agents, underscoring an urgent need for future research.

Date: September 15, 2026 Code: https://github.com/VisionXLab/V-ICAL Dataset: https://huggingface.co/datasets/VisionXLab/V-ICAL Homepage: https://visionxlab.github.io/V-ICAL-Homepage

## 1 Introduction

In-context learning (ICL) equips models with the remarkable ability to adapt from input exemplars without parameter updates [1, 14, 41]. In multimodal domain, video demonstrations serve as a highly natural and information-rich modality for teaching new skills—spanning software workflows, game strategies, and physical procedures. However, the evaluation of multimodal ICL remains largely confined to static, ofline paradigms, where models passively predict answers based on visual inputs rather than actively engaging with or receiving feedback from an environment [8, 15]. This discrepancy leaves a critical void in our understanding of whether ICL can genuinely support sustained agency.

![](images/8d0c025d9a9615dc1460af44156d83ffcffee3d08ce0c01b0184e2af28b0aa53.jpg)  
Figure 1 V-ICAL comprises 342 evaluation tasks across 37 environments, covering diverse environment types, environment dynamics, and memory requirements. As each environment may belong to multiple environment-type categories, the category counts do not sum to 37.

Empowering a multimodal agent via video ICL demands significantly more than passive comprehension. It requires the agent to distill underlying rules and strategies from an in-context video demonstration and execute them within a closed perception–decision–action–feedback loop. Specifically, the agent must ground demonstrated behaviors in novel visual states, execute valid actions, monitor their consequences, and adapt its subsequent planning toward a desired objective. Unfortunately, existing evaluation frameworks only assess isolated facets of this process: standard video benchmarks focus on semantic comprehension, demonstration-guided approaches are often ofline, and current interactive benchmarks typically rely on rigid observation-action traces or textual tutorials limited to post-failure corrections [9, 10, 13, 27, 39]. Consequently, the central question remains unanswered: Can a multimodal agent efectively induce a policy from in-context video demonstrations and deploy it through sustained interaction in dynamic environments?

To systematically assess the capability of multimodal agents in in-context learning, we propose V-ICAL (Vision In-Context Agentic Learning), a comprehensive benchmark tailored for video ICL in interactive settings. The dataset is constructed from 342 meticulously curated evaluation tasks distributed across 37 environments and seven distinct environment families. In this setup, each task supplies human-demonstrated videos as the exclusive in-context behavioral exemplars. These visual demonstrations are paired with a target initial visual state and minimal accompanying text to specify non-visual constraints. Crucially, the multimodal agent must infer the underlying task knowledge from the video and subsequently execute a multi-turn action trajectory relying entirely on raw pixel observations, without any parameter updates or further human intervention. To ensure a robust evaluation, the task pool exhibits high mechanical diversity. It encompasses both static environments and dynamic ones (where state transitions occur independently of agent actions), as well as decision paradigms demanding either single-frame spatial perception or long-horizon temporal memory. Agent performance is subsequently quantified via complementary pass-rate and normalized-score metrics, providing a holistic measure of both absolute success and incremental progress.

Our comprehensive evaluation of 19 mainstream multimodal agents exposes a critical gap between merely comprehending an in-context demonstration and reliably utilizing it for sustained agency. The leading model, Seed-2.1-Pro, achieves a score of only 54.4/100—falling drastically short of the human baseline of 83.6. Other formidable models perform even worse, including Gemini-3.6-Flash (49.5), Gemini-3.1-Pro (48.0), and GPT-5.6-sol (46.7). These results underscore that even the most advanced multimodal models lag significantly behind human-level competence in closed-loop, video-prompted agentic tasks. It also finds that the average performance of top-tier models drops by 42.3% when the environment changes during a task and by 32.6% when a short term of the environment process must be remembered, versus 20.2% and 17.4%, for humans.

To obtain finer-grained model diagnostics, we develop a trajectory-auditing protocol based on a four-stage model of interactive decision making. Applied to models, it assesses whether demonstrated information is transferred into decisions and the stage at which visually grounded decision making becomes deficient. For example, Seed-2.1-Pro performs better in strategy transfer and overall interaction, whereas Gemini-3.6-Flash performs better in rule transfer and at the upstream stages of decision making. It also reveals shared limitations across models: agents sufer a substantially greater deficit in strategy transfer than in rule transfer.

To isolate the impact of visual demonstrations, we conduct an ablation study evaluating four multimodal models by controlling the availability of in-context videos under basic textual rule settings. Specifically, the addition of video increases the normalized score of Gemini-3.6-Flash from 44.8 to 49.5, and Seed-2.0-Lite from 29.9 to 37.0. These findings demonstrate that despite the overall gap with human performance, current multimodal models possess basic video in-context learning capabilities.

In summary, our main contributions are as follows:

• We formulate video-based in-context learning for multimodal agents as a closed-loop sequential decision problem. This paradigm requires models to induce underlying rules and strategies from condensed demonstration packages and subsequently deploy them within novel, interactive states.

• We introduce V-ICAL, a comprehensive benchmark of 342 tasks across 37 environments, and evaluate 19 mainstream multimodal agents. Even top-tier MLLMs, including Seed-2.1-Pro, Gemini-3.6-Flash, and GPT-5.6-sol, score below 55, while the best open-weight model DeepSeek-V4.1-Flash reaches 34.8, far behind the human average of 83.6. This reveals that mainstream models are all lack of the ability of on video in-context agency tasks on our benchmark.

• We introduce fine-grained trajectory audits to systematically analyze the agentic decision-making process. By assessing what information successfully transfers and where the decision pipeline falters, we reveal that agents sufer a significantly more pronounced deficit in strategy transfer than in basic rule transfer. Furthermore, we pinpoint that the most critical bottlenecks lie in interpreting real-time visual states and planning subsequent actions.

• We conduct targeted ablations on video availability and rule specificity, establishing comprehensive baselines and revealing profound model limitations. Crucially, our results reveal that multimodal models can efectively distill underlying mechanics from video demonstrations, achieving an instructional efect that is functionally equivalent to providing meticulously detailed textual rules.

## 2 Related Work

In-context learning (ICL) allows a model to adapt from examples in its prompt without parameter updates. In text-only settings, many-shot prompts can yield substantial gains across diverse tasks [1]. In multimodal settings, Jiang et al. [14] evaluate nearly 2,000 examples across 14 image-based datasets and report approximately log-linear scaling for Gemini 1.5 Pro on many of them, while VL-ICL Bench broadens evaluation to perception, reasoning, and generation [41]. These studies use static image–text demonstrations.

Video-MME [9] and Video-MME-v2 [10] evaluate video comprehension through question answering rather than adaptation from demonstrations. VideoICL retrieves video examples for iterative out-of-distribution prediction, whereas Demo-ICL-Bench measures procedural transfer from video or text demonstrations to question answering [8, 15]; both remain ofline prediction settings. LMAct asks whether interactive policies improve as the context grows from zero to 512 full observation–action demonstration episodes, reaching up to one million tokens, and finds that additional demonstrations often have little efect [27]. Our setting is diferent: each task uses a compact package of one to four human-curated video clips as its specification, starts the acting model from a fixed target state, and tests sustained closed-loop execution across 37 heterogeneous environments. VideoWebArena evaluates tutorial use in 1,621 skill-retention web tasks, but reports that current agents can perform worse with tutorials than without them [13].

In addition, some virtual environment benchmarks expose complementary aspects of agentic decision-making. BALROG spans challenging reinforcement-learning environments including NetHack [23]; lmgame-Bench standardizes six Gym-style environments with optional perception and memory scafolds [12]; MM-HELIX [40] and Atari-GPT [33] tests multimodal models as zero-shot low-level environment policies. More recently, GameWorld evaluates 170 tasks across 34 browser environments using semantic or computer-use control and state-verifiable outcomes [22]. GameVerse instead studies post-failure reflection rather than ICL: a separate model distills failed-play and tutorial footage into textual feedback for retry, rather than supplying demonstration videos directly to the acting policy as in-context input [39]. Unlike prior work that relies on post-failure reflection or ofline prediction, we treat video ICL as an agentic capability: the same multimodal agent must induce task rules and strategies from compact demonstration packages, then deploy them through sustained closed-loop interaction.

## 3 Benchmark Design

Design rationale. Humans routinely acquire new skills from video demonstrations and apply them in novel situations. We study whether multimodal agents can achieve the same capability through video ICL. Unlike passive video understanding, this requires an agent to infer an executable strategy from a demonstration and deploy it through sustained closed-loop interaction. Accordingly, V-ICAL combines a closed-loop agent protocol, diverse environments and initial states, and complementary pass-rate and normalized-score metrics.

## 3.1 Task Formalization

We formulate video ICL for a multimodal agent as a multi-turn sequential decision-making problem. Here, the agent is the deployed multimodal model inside a closed perception–decision–action–feedback loop: it receives one or more demonstration videos as task-specific behavioral exemplars, executes an action, observes the resulting visual state, and continues acting over a growing context.

Demonstration-video and text-prompt construction. For finishing the benchmark tasks and inspecting models interaction trajectories and outcomes, human experts spent more than 100 hours. They jointly calibrated initial states, interaction horizons, pass conditions, and score targets to ensure comparable dificulty and progress across tasks. Each demonstration–prompt package was independently cross-validated and iteratively refined. We encode visually demonstrable rules in the video and reserve text only for residual information, ensuring that success primarily reflects an agent’s ability to infer and execute strategies from video rather than follow detailed textual instructions.

First turn (in-context demonstration injection). During the initial turn, the agent receives three distinct inputs to establish the context. The primary input consists of demonstration videos: one or more human-curated clips that densely encode the underlying environment hidden rules. Within these clips, each frame features a caption detailing the previous action, while the image itself reflects the resulting state. These videos conclude with a final frame depicting a terminal status, such as success, failure, or a voluntary stop, collectively forming the task’s in-context behavioral exemplars. Second, the agent receives the current visual state, representing the specific environment state it must act upon. Finally, the agent is provided with supplementary textual rules. In our default setting, this text is strictly limited to non-visual constraints that cannot be inferred from the videos alone, including hidden scoring mechanisms, latent variables, step limits, or delayed action efects.

![](images/10dd51298d06f2518674e5eab8d811e9c424b8e6b4240fb781b7383f0441b99d.jpg)  
Figure 2 Overview of V-ICAL. The benchmark contains 342 evaluation tasks spanning 37 environments and seven environment families. Each episode supplies an optional in-context video demonstration package, then evaluates a multimodal agent through multi-turn visual decision-making in a closed perception–action loop. Performance is summarized by pass rate and final score with explicit failure semantics.

Subsequent turns (multi-turn decision-making). No additional demonstrations are provided in subsequent steps. The agent receives only the updated visual state resulting from its previous action and must determine the next move. By preserving the entire interaction history, which encompasses the initial demonstration package, past observations, and previous model outputs, the context window naturally formulates a longcontext evaluation scenario. At each turn, the model is required to output an action in the exact format of [Action:<name>]. Complete parsing and retry protocols are detailed in Appendix B.1.

Objective. This benchmark is deliberately designed to evaluate the multimodal agent holistically. While the agent inherently relies on the parametric knowledge of its foundational model, the demonstration videos serve as the sole task-specific exemplars provided during evaluation. To succeed, the agent must synthesize these visual exemplars with the task identifier, available action space and minimal text constraints to deduce the operational rules, goals, and environment dynamics. Crucially, it must transfer this induced knowledge to novel states without any parameter updates or post-failure corrections. This end-to-end evaluation paradigm rigorously tests the complete agent decision loop: from visual comprehension and rule induction to state grounding, planning, and closed-loop execution.

## 3.2 Environment Setting

Out of over 100 candidate environments, we deliberately selected 37 environments based on four core criteria: visual legibility, environmental stability, timely feedback within standard context windows, and discriminative dificulty. The latter ensures that strategic reasoning significantly outperforms random actions.

Built upon the broader Gym ecosystem, our benchmark encompasses 37 environments categorized into seven distinct domains to ensure robust task diversity: Toy Text, comprising small-scale discrete environments like ClifWalking and Taxi; MiniGrid, featuring procedurally generated grid-based mazes; Classic Control, focusing on physics-based tasks such as CartPole and Acrobot; ALE Atari, consisting of classic arcade titles like Seaquest, Breakout, and Boxing; Procgen, ofering procedurally generated 2D environments; HighwayEnv, providing autonomous driving scenarios; and third-party environments, including environments like Tetris and FlappyBird.

To scale beyond the base environments, we further expanded the task pool utilizing two primary mechanisms. First, we employed parameterized modifications, applying distinct random seeds and initial frame-skipping to generate diverse starting layouts and level variants within a single environment. Second, we introduced human-driven initialization, where a human operator manually navigates the environment to a specific starting state. This approach enables the construction of scenarios tailored to precise evaluation foci, including critical decision junctures, typical environmental traps, or resource-constrained conditions. Collectively, this systematic expansion yields a final benchmark comprising 342 distinct evaluation tasks.

## 3.3 Scoring System

We evaluate agent performance from two complementary perspectives: Pass Rate and Final Score.

Pass rate: This metric reports the fraction of episodes in which the agent satisfies a predefined, task-specific success condition. These conditions—such as reaching a spatial goal, achieving a target score, or surviving for a specified duration—serve as essential markers of task comprehension. Consequently, the pass rate reflects the agent’s fundamental ability to complete the task. This approach aligns with the achievement-style scoring philosophies utilized in broader agency benchmarks [11, 23].

Final score: This metric provides a granular measure of partial progress, normalized to a [0, 100] scale. If an environment natively outputs a benchmark-aligned progress score, we utilize it directly. Otherwise, we construct environment-specific rules to map raw metrics (e.g., rewards, survival time, event counts, milestones, or path eficiency) onto this standardized scale. To ensure accurate calibration, human experts establish reference targets at the task level; alternatively, task-level targets are set when achievable progress varies based on the initial state. All scores are strictly bounded, incorporating explicit adjustments for negative rewards, agent death, and incomplete progress upon termination. For aggregation, we first average the task scores within each environment and subsequently apply equal weighting across all environments. This methodology synthesizes the arithmetic averaging of [23] with the per-environment normalization of [24]. Comprehensive formulas and edge-case resolutions are detailed in Appendix B.2.

Additionally, we track four explicit failure modes during execution: timeouts, continuous invalid actions, token overflows, and early terminations. Formal definitions and detailed handling procedures for these modes are provided in Appendix E.1.

## 4 Experiments

## 4.1 Experiment Setting

To guarantee controlled and eficient evaluations, we standardized the selected environments. This was achieved by enforcing uniform parameters, such as maximum step limits and frame-skipping rates, alongside targeted modifications that include normalizing termination signals and masking redundant actions. Full implementation details are provided in Appendix A.1. We uniformly sample each visual trajectory at 1 FPS. For video-native models, the frames are encoded into an MP4 and provided through video url (or Gemini’s inline data), whereas non-video-native models such as GPT receive the exact same frames as a temporally ordered image sequence. The visual content, frame count, ordering, and sampling rate are identical across settings; only the input representation difers, ensuring a fair comparison.

## 4.2 Main Results

Task Classification. The main evaluation results are presented in Table 1. In addition to the base classification of each task, we introduce two orthogonal attributes to further characterize task complexity: memory requirement and environment dynamics. Memory requirement categorizes tasks based on whether efective decision-making can rely solely on the current visual observation. Single-frame tasks can be solved by reasoning exclusively over the current frame (e.g., path planning in Taxi). Conversely, Multi-frame (or memory-dependent) tasks require the agent to retain historical observations resulting from prior actions and synthesize them with the current state to succeed (e.g., rhythm judgment in FlappyBird and complex planning in partially observable MiniGrid). Environment dynamics denotes whether the environment state evolves autonomously. In Static environments, state transitions occur strictly in response to agent actions (e.g., Toy Text, MiniGrid). In contrast, Dynamic environments evolve continuously over time, independent of agent interventions or idle states (e.g., most ALE Atari titles, Procgen).

<table><tr><td rowspan="3">Model</td><td colspan="9">ENVIRONMENT TYPE</td><td colspan="3">MEMORY REQ.</td><td colspan="3">ENV. DYNAMICS</td><td rowspan="2">Overall</td></tr><tr><td>Maze</td><td>Shooter</td><td></td><td>Control</td><td></td><td>Dodge</td><td>Collect</td><td></td><td>Sport</td><td>Single</td><td></td><td>Multiple</td><td>Static</td><td>Dynamic</td><td></td></tr><tr><td>S</td><td>P% S</td><td>P%</td><td>S</td><td>P%</td><td>S</td><td>P% S</td><td>P%</td><td>S</td><td>P% S</td><td>P%</td><td>S P%</td><td>S</td><td>P% S</td><td>P%</td><td>S</td><td>All P%</td></tr><tr><td colspan="10">Closed-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seed-2.1-Pro [29]</td><td>64.4</td><td>7043.3</td><td>49</td><td>88.4</td><td>73 47.1</td><td></td><td>51 52.3</td><td>60</td><td>54.3 80</td><td>68.9</td><td>7052.7</td><td>57</td><td>74.3</td><td>70 47.0</td><td>54</td><td>54.4 58</td><td></td></tr><tr><td>Gemini-3.6-Flash [6]</td><td>61.0</td><td>70 37.9</td><td>39</td><td>88.1</td><td>80 39.9</td><td></td><td>44 42.2</td><td></td><td>63 29.7 50</td><td>67.1</td><td>7047.3</td><td>52</td><td>77.2</td><td>75 39.2</td><td>45</td><td>49.5</td><td>53</td></tr><tr><td>Gemini-3.1-Pro [5]</td><td>49.1</td><td>54 44.9</td><td>46 65.5</td><td></td><td>53 42.0</td><td></td><td>4840.3</td><td>45 34.3</td><td>70</td><td>61.5</td><td>70 46.4</td><td>51</td><td>67.7</td><td>69 40.8</td><td>47</td><td>48.0</td><td>53</td></tr><tr><td>GPT-5.6-sol [21]</td><td>52.0</td><td>53 37.6</td><td>43 82.7</td><td></td><td>87 38.1</td><td></td><td>43 41.5</td><td></td><td>43 35.0 60</td><td>82.5</td><td>8542.4</td><td>47</td><td>68.3</td><td>67 38.8</td><td>46</td><td>46.7</td><td>51</td></tr><tr><td>Gemini-3-Flash [4]</td><td>36.9</td><td>37 37.6</td><td>39 64.5</td><td></td><td>67 34.7</td><td></td><td>36 31.1</td><td></td><td>40 34.3 60</td><td>52.2</td><td>5038.1</td><td>41</td><td>54.7</td><td>53 34.0</td><td>38</td><td>39.6 42</td><td></td></tr><tr><td>Seed-2.0-Lite [28]</td><td>39.0</td><td>51 26.5</td><td>24</td><td>86.3</td><td>80 31.4</td><td>36</td><td>36.2</td><td>48</td><td>32.0 60</td><td>54.2</td><td>6034.9</td><td>39</td><td>50.6</td><td>49 31.9</td><td>38</td><td>37.0 41</td><td></td></tr><tr><td colspan="10">Open-weight Models</td><td colspan="7"></td></tr><tr><td>DeepSeek-V4.1-Flash [7]</td><td>34.1</td><td>36 28.7</td><td>26 71.6</td><td>67</td><td>32.4</td><td>33</td><td>26.0</td><td>30 32.3</td><td>50</td><td>46.1</td><td>45 33.5</td><td>34</td><td>51.0</td><td>50 28.8</td><td>30</td><td>34.8 35</td><td></td></tr><tr><td>Kimi-K3 [31]</td><td>32.3</td><td>40 26.4</td><td>34 70.1</td><td></td><td>73 27.2</td><td>34</td><td>25.4</td><td>30</td><td>26.3 40</td><td>51.8</td><td>55 30.2</td><td>37</td><td>46.7</td><td>46 27.3</td><td>36</td><td>32.5</td><td>39</td></tr><tr><td>Qwen3.5-397B-A17B [25]</td><td>28.2 33 30.3</td><td></td><td>27 73.6</td><td></td><td>67 28.1</td><td>27</td><td></td><td>25.4 30 25.7</td><td>40</td><td>44.2</td><td>45 31.1</td><td>30</td><td>45.0</td><td>42 27.9</td><td>27</td><td>32.5</td><td>31</td></tr><tr><td>DeepSeek-V4F-V [36]</td><td>21.0</td><td>29 26.0</td><td>34 47.1</td><td></td><td>53 24.1</td><td></td><td>30 19.3</td><td>30 21.7</td><td>30</td><td>26.6</td><td>15 25.0</td><td>32</td><td>30.1</td><td>26 23.4</td><td>32</td><td>25.2</td><td>30</td></tr><tr><td>gemma-4-31B-it [30]</td><td>20.4 28 17.7</td><td></td><td>29 32.8</td><td></td><td>27 19.7</td><td>25</td><td>14.4</td><td>18 13.7</td><td>10</td><td>40.1</td><td>35 19.7</td><td>24</td><td>29.8</td><td>25 18.9</td><td>25</td><td>21.9</td><td>25</td></tr><tr><td>MiMo-v2.5 [35]</td><td>20.0</td><td>21 18.3</td><td>26 34.6</td><td></td><td>27 21.4</td><td></td><td>27 13.1</td><td>28 16.0</td><td>20</td><td>25.6</td><td>15 19.8</td><td>24</td><td>22.2</td><td>14 19.8</td><td>26</td><td>20.4</td><td>23</td></tr><tr><td>Qwen2.5-VL-72B [3]</td><td>14.3</td><td>16 16.3</td><td>35 20.1</td><td></td><td>7</td><td>20.4 31 11.5</td><td></td><td>25 10.3</td><td>0</td><td>19.0</td><td>10 17.5</td><td>25</td><td>16.9</td><td>8 17.9</td><td>29</td><td>17.7</td><td>23</td></tr><tr><td>MiMo-VL-7B [34]</td><td>13.9</td><td>15 15.8</td><td>1019.7</td><td></td><td>13 19.9</td><td></td><td>16 13.3</td><td>10 12.7</td><td>10</td><td>20.6</td><td>2017.0</td><td>13</td><td>17.1</td><td>14 17.5</td><td>14</td><td>17.4</td><td>14</td></tr><tr><td>Qwen3.6-27B [26]</td><td>10.5</td><td>18 13.8</td><td>3617.5</td><td></td><td>719.2</td><td></td><td>33 12.2</td><td>2822.7</td><td>40</td><td>17.3</td><td>10 16.8</td><td>29</td><td>14.8</td><td>8 17.6</td><td>34</td><td>16.9</td><td>27</td></tr><tr><td>GLM-4.5V [32]</td><td>10.9</td><td>15 13.1</td><td>27 28.7</td><td></td><td>27 18.1</td><td></td><td>2611.9</td><td>15 12.0</td><td>10</td><td>17.2</td><td>5 16.5</td><td>23</td><td>16.6</td><td>10 16.6</td><td>25</td><td>16.6</td><td>21</td></tr><tr><td>Qwen3.5-27B [25]</td><td>13.3</td><td>15 12.4</td><td>35 22.4</td><td></td><td>13 17.8</td><td></td><td>32 12.2</td><td>20 20.0</td><td>40</td><td>18.5</td><td>016.2</td><td>27</td><td>14.2</td><td>417.3</td><td>32</td><td>16.4</td><td>24</td></tr><tr><td>Qwen3-Omni-30B [37]</td><td>9.7 13</td><td>5.8</td><td>20 19.5</td><td></td><td>13 15.6</td><td>27</td><td>5.3</td><td>18 16.7</td><td>30</td><td>19.6</td><td>1012.8</td><td>22</td><td>16.9</td><td>10 12.3</td><td>24</td><td></td><td></td></tr><tr><td>Qwen3-VL-235B [2] 12.5</td><td></td><td>6 6.9</td><td>3 19.7</td><td></td><td>13 11.7</td><td></td><td>511.6</td><td>5 15.3</td><td>10</td><td>21.6</td><td>5 11.2</td><td>5</td><td>14.2</td><td>6 11.6</td><td>5</td><td>13.6</td><td>20</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td><td></td><td>12.3</td><td>5</td></tr><tr><td>Reference Baselines Random</td><td></td><td>10 19.1</td><td>23</td><td>16.8</td><td>13 15.5</td><td></td><td>15 15.0</td><td></td><td>15 19.4 30</td><td>5.6</td><td></td><td>2 15.0 15</td><td>8.4</td><td></td><td>5 16.1</td><td>17</td><td>14.0 14</td><td></td></tr><tr><td>Human-score</td><td>7.9 94.6</td><td>92.1 96 76.4 85</td></table>

Table 1 Main results on V-ICAL. Memory Req.: memory requirement for action selection. Single means that the current frame is suficient; Multiple means the agent must retain one or more earlier post-action observations and integrate them to choose a later action. Env. Dynamics: whether the environment changes independently of the agent (Static vs. Dynamic). S: normalized score (0–100); P%: pass rate.

Overall performance. As illustrated in Table 1, Seed-2.1-Pro achieves the state-of-the-art overall performance, securing a normalized score of 54.4 and a pass rate of 58%. Notably, even this leading score only marginally surpasses the halfway mark of the normalized scale. While it significantly outperforms the random baseline (14.0) by 40.4 points, it still falls 45.6 points short of the theoretical maximum. This substantial performance gap underscores the profound dificulty of end-to-end video ICL within closed-loop interactive environments. Trailing the leading model are Gemini-3.6-Flash (49.5 score / 53% pass rate), Gemini-3.1-Pro (48.0 / 53%), and GPT-5.6-sol (46.7 / 51%). Within the open-weight category, it is noticing that DeepSeek-V4.1-Flash achieves the highest normalized score at 34.8 with a 35% pass rate, with a low cost and fast speed. Kimi-K3 and Qwen3.5-397B-A17B tie for second at 32.5, though Kimi-K3 demonstrates a superior pass rate (39% vs. 31%). The best open-weight score remains 19.6 points below Seed-2.1-Pro. Finally, the random policy achieves a score of 14.0 with a 14% pass rate, establishing a non-trivial baseline floor for the benchmark.

![](images/9f3690dc4f7de247b4a4db59249af5275c8a8b55de8488ac3f89d3db30bd56ff.jpg)  
Figure 3 Representative failures across the four-stage decision-making process. Each panel contrasts the human decision with the model response at the corresponding deficient stage.

Limitation in Memory. All fully evaluated models exhibit a severe and consistent performance degradation on memory-intensive tasks. Among the four strongest models, Seed-2.1-Pro falls from 68.9 to 52.7, Gemini-3.6-Flash from 67.1 to 47.3, Gemini-3.1-Pro from 61.5 to 46.4, and notably, GPT-5.6-sol drops from 82.5 to 42.4. This downward trend persists across all evaluated open-weight models, which explicitly rules out generic category-size or score-scale artifacts as the cause of the drop. Furthermore, as corroborated by the case audits in Section 4.3, this pervasive performance decline exposes a critical bottleneck in agentic visual memory. Specifically, current multimodal agents struggle to retain and synthesize the visual consequences of their own actions—a dynamic capability that traditional ofline video QA benchmarks inherently fail to measure.

Vulnerability in Dynamic Environments. A similarly consistent weakness emerges in autonomously evolving environments. For instance, Seed-2.1-Pro’s score drops from 74.3 in static settings to 47.0 in dynamic ones, while Gemini-3.6-Flash experiences a steep decline from 77.2 to 39.2. Because dynamic environments change independently of the agent’s interventions, they demand robust continuous replanning and real-time state tracking. This vulnerability is universally observed across all six closed-source systems and the three strongest open-weight models. Consequently, this bottleneck highlights a profound gap between passive video comprehension and active agency: to achieve true autonomy, agents must possess the capability to process and react while the world continuously evolves around them.

Comparison with Human Performance. To provide an intuitive point of reference for model performance, we establish a human baseline, denoted as the human score, following established methodologies in recent benchmarks [23, 24]. Human participants are provided with the exact same demonstration package as the models and interact with the environment for one full episode, starting from the identical initial states used for AI evaluation. For each environment, three participants complete the tasks independently; their raw scores are normalized according to Section 3.3, and the arithmetic mean defines the environment’s Human Score. As the results indicate, a massive capability gap remains. Even the strongest evaluated model, Seed-2.1-Pro, lags significantly behind the human reference, trailing by 29.2 normalized score points (54.4 vs. 83.6) and a 33% margin in pass rate (58% vs. 91%). Interestingly, human participants and AI models exhibit aligned performance degradation patterns across diferent task complexities. Human scores drop from 99.0 to 81.8 when transitioning from Single-frame to Multi-frame tasks, and from 98.1 to 78.3 when moving from Static to Dynamic environments. This strong alignment suggests that temporal memory dependence and autonomous environment evolution represent intrinsic sources of task dificulty, rather than model-specific vulnerabilities. However, the persistent and substantial absolute gap between the two highlights that current multimodal agents are considerably less robust than humans in overcoming these fundamental challenges.

Table 2 Comparison of the two trajectory-audit criteria.
<table><tr><td></td><td>Information transfer evaluation</td><td>Sequential failure attribution</td></tr><tr><td>Diagnostic focus</td><td>Transfer of demonstrated information.</td><td>Location of a consequential process deficiency.</td></tr><tr><td>Unit of diagnosis</td><td>Information type.</td><td>Processing stage.</td></tr><tr><td>Output</td><td>One rule-transfer outcome and one strategy- transfer outcome.</td><td>One most-upstream consequential failure stage, or none.</td></tr><tr><td>Primary use</td><td>Evaluating what the model transfers from the demonstration into its decisions.</td><td>Diagnosing where online visually grounded decision-making becomes deficient.</td></tr></table>

## 4.3 Fine-Grained Auditing of Interaction Trajectories

The aggregate results above quantify overall benchmark performance; to obtain finer-grained diagnostic profiles of model behavior during environment interaction, we introduce a trajectory-auditing framework built around the following staged model of the acting agent’s decision-making process:

knowledge induction −→ state grounding −→ planning −→ action commitment.

Knowledge induction extracts and retains a reusable task model from the demonstration, including rules and strategies inferred from environment dynamics that encompass both action-conditioned environmental responses and reusable state–action–outcome regularities. State grounding maps induced knowledge onto live visual evidence to estimate relevant entities, relations, motion, and progress. Planning turns the grounded state into an adaptive course of action through lookahead, timing, sequencing, risk assessment, and revision. Action commitment converts the selected step into the submitted action. The labels serve as rubric-guided functional attributions from observable trajectory evidence rather than as direct measurements of the model’s internal computation. Figure 3 shows characteristic failures at each stage.

We analyze this process using two complementary criteria: information transfer evaluation tracks demonstrated rules and strategies separately across the complete decision-making chain and produces an end-to-end judgment for each information type based on whether it is both induced and appropriately used, while sequential failure attribution examines the chain stage by stage and identifies the most upstream deficient stage that materially explains unsuccessful or substantially suboptimal behavior. Taken together, they provide two cross-cutting perspectives on each trajectory. Table 2 summarizes their distinction.

To apply these criteria, we develop an automated pipeline in which a multimodal LLM audits all 342 interaction trajectories produced by each of Seed-2.1-Pro, Gemini-3.6-Flash, and Qwen3.5-397B-A17B. A stratified blind human verification yields agreement rates of 95.8%, 94.6%, and 94.6% for rule transfer, binary strategy transfer, and stage attribution, respectively (Cohen’s κ = 0.887, 0.861, and 0.925). Appendix C provides the complete protocol, prompts, and verification results.

## 4.3.1 Information Transfer Evaluation

Information transfer is evaluated along two dimensions. Rule transfer covers demonstrated control semantics, causal dynamics, objectives, scoring mechanisms, and terminal conditions, while strategy transfer covers reusable state–action–outcome regularities such as timing, sequencing, target selection, risk management, and adaptation. Both are successful only when the relevant information is induced and appropriately applied during environment interaction; verbal recognition, intention, or action-sequence imitation alone is insuficient. These judgments assess behavioral use of demonstrated information, not whether the video was its exclusive source. Table 3 reports the successful and unsuccessful transfer rates for both dimensions.

Seed-2.1-Pro and Gemini-3.6-Flash exhibit broadly comparable information-transfer capabilities, consistent with their benchmark scores of 54.4 and 49.5. Gemini achieves stronger rule transfer (84.3% versus 77.3%), whereas Seed achieves stronger strategy transfer (31.7% versus 29.0%); Seed’s greater success in carrying adaptive policies into interaction may ofset its weaker handling of demonstrated mechanics and contribute to its modestly higher overall score. Qwen3.5-397B-A17B performs less successfully on both dimensions, particularly strategy transfer: its successful rule- and strategy-transfer rates are 70.0% and 14.4%, respectively, consistent with its lower score of 32.5.

Table 3 Information transfer evaluation results. Transfer values are percentages macro-averaged over the 37 environments; score and pass rate reproduce the overall benchmark results in Table 1.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Score</td><td rowspan="2">Pass rate</td><td colspan="2">Rule transfer</td><td colspan="2">Strategy transfer</td></tr><tr><td>Succ.</td><td>Unsucc.</td><td>Succ.</td><td>Unsucc.</td></tr><tr><td>Seed-2.1-Pro</td><td>54.4</td><td>58</td><td>77.3</td><td>22.7</td><td>31.7</td><td>68.3</td></tr><tr><td>Gemini-3.6-Flash</td><td>49.5</td><td>53</td><td>84.3</td><td>15.7</td><td>29.0</td><td>71.0</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>32.5</td><td>31</td><td>70.0</td><td>30.0</td><td>14.4</td><td>85.6</td></tr></table>

## 4.3.2 Sequential Failure Attribution

For sequential failure attribution, the auditor first identifies the behavior that most consequentially contributes to an unsuccessful or substantially suboptimal outcome. It then reconstructs the evidence-supported chain producing that behavior and tests the stages from upstream to downstream. It assigns the most upstream deficient stage that materially explains the identified behavior, moving downstream only when all preceding stages are adequately supported. Harmless or corrected errors are ignored, and none is assigned when the chain contains no consequential deficiency. Table 4 reports the resulting stage-attribution rates.

Table 4 Sequential failure attribution results. Attribution values are percentages macro-averaged over the 37 environments; score and pass rate reproduce the overall benchmark results in Table 1.
<table><tr><td>Model</td><td>Score</td><td>Pass rate</td><td>Knowledge induction</td><td>State grounding</td><td>Planning</td><td>Commitment</td><td>None</td></tr><tr><td>Seed-2.1-Pro</td><td>54.4</td><td>58</td><td>10.8</td><td>27.6</td><td>21.5</td><td>0.0</td><td>40.1</td></tr><tr><td>Gemini-3.6-Flash</td><td>49.5</td><td>53</td><td>2.7</td><td>24.1</td><td>40.9</td><td>1.8</td><td>30.6</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>32.5</td><td>31</td><td>14.4</td><td>44.6</td><td>24.3</td><td>0.7</td><td>15.9</td></tr></table>

Seed and Gemini again occupy a similar tier but exhibit diferent failure profiles. Seed has the highest none rate (40.1%), consistent with its stronger benchmark performance, while Gemini receives fewer knowledge-induction and state-grounding attributions (2.7% and 24.1%, versus Seed’s 10.8% and 27.6%), indicating relatively stronger upstream processing. Qwen has the lowest none rate (15.9%) and the highest knowledge-induction and state-grounding attribution rates (14.4% and 44.6%), further showing deficiencies in early processing. Because these are exclusive, most-upstream attributions, an upstream failure censors all downstream stages; lower frequency at a later stage therefore does not imply greater ability. Action commitment is rare across models (0.0–1.8%), placing most consequential deficiencies before final action submission.

## 4.3.3 Joint Analysis of Transfer Evaluation and Failure Attribution

We cross-tabulate each trajectory’s transfer outcomes and stage attribution to localize information-specific outcomes along the shared decision process (Figure 4). The two information types produce distinct joint audit patterns. Successful end-to-end transfer is naturally associated with none, but this association is consistently stronger for strategy transfer: 85.4% for Seed, 84.3% for Gemini, and 75.0% for Qwen, compared with 49.9%, 33.4%, and 18.3% for rule transfer. Across all three models, the attribution distribution under unsuccessful strategy transfer shifts downstream relative to that under unsuccessful rule transfer: mass moves away from knowledge induction and toward planning, while state grounding remains substantial. Together, these relative patterns reflect diferent processing demands within the same decision-making process. Both transfer dimensions require their respective information to be induced and applied, but strategy transfer is comparatively more dynamic: it coordinates and adapts decisions across an unfolding sequence of states, whereas rule transfer more often applies particular demonstrated rules when they become relevant. These patterns are therefore consistent with strategy transfer placing stronger demands on downstream stages and, consequently, on the model’s overall processing capability.

![](images/5c6fbf1312f2540c3030cb1c1dcf7f1d5e879b3c67cfd7e78182f1fffe613946.jpg)  
Figure 4 Joint outcomes of information transfer evaluation and sequential failure attribution. Each model panel relates the two binary transfer outcomes to the four processing stages and none. Cells report conditional stage percentages macro-averaged across contributing environments; Appendix C.6 provides the formula and denominators.

## 4.4 Ablation Study: Efficacy of In-Context Demonstrations and Rule Specificity

The fundamental premise of video-based ICL is that providing visual demonstrations should empirically improve end-to-end agent performance. To systematically validate this, we conduct an ablation study evaluating four multimodal agents (Gemini-3.6-Flash, Gemini-3-Flash, Seed-2.0-Lite, and Qwen3.5-397B-A17B) acorss our environments, controlling for two critical variables: the presence of the demonstration video and the granularity of textual rules. Specifically, we compare the default setting against scenarios with no video provided, as well as scenarios supplemented with detailed rules. These detailed rules are meticulously distilled by human experts to explicitly encode all mechanics and strategic nuances originally depicted in the videos.

## 4.4.1 Results

The results are shown in Table 5. Under the base rule setting, providing demonstration videos improves performance for three out of the four evaluated models. Notably, Gemini-3.6-Flash and Seed-2.0-Lite show substantial score increases (from 44.8 to 49.5 and from 29.9 to 37.0, respectively) with the addition of video. The only exception is Gemini-3-Flash, which exhibits a slight performance decrease (from 41.2 to 39.6). These results indicate that current multimodal models possess clear video in-context learning capabilities when operating with basic textual instructions. However, under the detailed rule condition, the performance diferences between the video and no-video settings become marginal across all four models. This suggests that the marginal benefit of visual demonstrations diminishes when the textual rules already provide comprehensive task information.

Table 5 Impact of demonstration video and rule specification on end-to-end agent performance. Base + Video is the baseline. Each cell reports normalized score (S) / pass rate (P%).
<table><tr><td rowspan="2">Model</td><td colspan="2">Base Rule</td><td colspan="2">Detailed Rule</td></tr><tr><td>Video</td><td>No Video</td><td>Video</td><td>No Video</td></tr><tr><td>Gemini-3.6-Flash</td><td>49.5 / 53</td><td>44.8 / 48</td><td>48.4 / 51</td><td>46.8 / 50</td></tr><tr><td>Gemini-3-Flash</td><td>39.6 /  42</td><td>41.2 / 46</td><td>44.1 / 50</td><td>43.6 /  47</td></tr><tr><td>Seed-2.0-Lite</td><td>37.0 / 41</td><td>29.9 / 36</td><td>37.2 / 42</td><td>37.2 / 43</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>32.5 / 31</td><td>30.7 / 37</td><td>33.9 / 36</td><td>34.2 / 39</td></tr></table>

Furthermore, our examination of rule specificity reveals that under the no-video condition, replacing base rules with detailed rules consistently improves performance across all models. As Table 5 indicates, all four models experience an increase in scores ranging from 2.0 to 7.3 points. This efect is most pronounced in Seed-2.0-Lite, which gains 7.3 points (from 29.9 to 37.2). This improvement demonstrates that detailed rules successfully compensate for the missing video by making task-relevant mechanics explicit. As corroborated by our fine-grained trajectory audits, visual demonstrations and explicit textual rules influence agent behavior through distinct cognitive and planning mechanisms.

Importantly, we observe that the models exhibit non-trivial performance even in the absence of both video demonstrations and detailed rules. This baseline competence stems from the rich prior knowledge inherent in the foundation models, which we consider an integral component of the evaluated agent. Consequently, the no-video condition does not evaluate a ”tabula rasa” (blank slate) agent; rather, it rigorously quantifies the marginal utility of in-context video demonstrations beyond the model’s pre-training priors.

## 4.4.2 Qualitative Analysis: The Interplay Between Visual Demonstrations and Textual Rules

Visual Demonstrations Can Impair Execution Feedback and State Grounding. In spatial navigation environments such as Procgen Heist and CustomMultiRoom, we observe that demonstration videos can actively interfere with execution correction when visual interpretation conflicts with live interaction feedback. For instance, when an agent becomes stuck near an obstacle, video-conditioned models may over-rely on the successfu trajectories seen in the demonstration, incorrectly inferring that their intended movement has succeeded despite evidence to the contrary. Consequently, they repeatedly issue inefective actions. In stark contrast, models operating without demonstrations more reliably depend on recent action-observation histories to detect static states and dynamically revise their subsequent plans. These failure cases highlight a counter-intuitive phenomenon: extraneous visual context can be detrimental when it overrides an agent’s grounding in direct execution feedback.

The Dual Role of Demonstrations in Inferring Physical Dynamics. The utility of video demonstrations is highly contingent upon the environment’s intrinsic dynamics and whether those dynamics can be reliably deduced from the model’s parametric priors. In CartPole, models without demonstrations successfully exploit their pre-trained knowledge of intuitive physics to balance the pole. Conversely, video-conditioned models are frequently destabilized by minor errors in visually estimating the pole’s continuous orientation. However, Acrobot exhibits a diametrically opposite pattern: because its under-actuated control mechanism is less intuitive, models lacking demonstrations frequently misinterpret the physics or fail to infer the motion-related quantities necessary for efective control. In such scenarios, demonstrations provide crucial spatiotemporal cues detailing how specific actions influence the system. This dichotomy suggests that visual demonstrations are highly beneficial when control depends on complex dynamic information, but become a liability when visual estimation noise outweighs the reliability of internal priors.

Detailed Textual Rules Can Compensate for Visual Deficits. Demonstrations implicitly convey task-relevant constraints and mechanics that are often omitted from default rule descriptions. When both video and detailed rules are absent, models frequently form incomplete task representations, adopting strategies that seem plausible under sparse textual descriptions but violate actual environmental constraints. Our ablation results provide complementary evidence for this mechanism: under the no-video condition, replacing default rules with meticulously detailed rules significantly improves performance by explicitly articulating previously underspecified mechanics. This confirms that detailed textual instructions can efectively compensate for missing visual or interactive cues. Ultimately, these findings reveal a principle of modality suitability: explicit textual rules are optimal for conveying rigid constraints and discrete logic, whereas visual demonstrations remain indispensable for transferring nuanced, dynamic behaviors that defy static textual description.

## 5 Conclusion

We introduced V-ICAL, a benchmark for video in-context learning by multimodal agents in interactive environments. Each task asks one deployed agent to induce rules and strategies from video exemplars, then execute them through sustained closed-loop interaction. Across 342 tasks and 19 state-of-the-art multimodal agents, the strongest agent reaches only 54.4/100. In controlled comparisons across three models, demonstrations do not provide consistent aggregate gains, exposing a gap between placing behavioral evidence in context and turning it into a usable policy. We also identify two agentic bottlenecks: visual memory over the consequences of the agent’s own actions and adaptation in autonomously evolving environments. Both gaps recur across model families while the random policy exhibits the opposite subgroup trend, supporting their interpretation as benchmark-wide capability deficits. Our four-stage decision-loop taxonomy further distinguishes failures of in-context knowledge induction from downstream failures in state grounding, planning, and action commitment.

These findings connect multimodal ICL with agentic evaluation. Ofline ICL can end with a correct prediction; an agent must carry induced knowledge through self-generated experience, continuous replanning in nonstationary worlds, and reliable action commitment. These closed-loop capabilities remain largely unsolved. As demonstration videos become a ubiquitous medium for task specification, closing the gap between what agents learn from watching and what they can execute will be essential for reliable multimodal agents.

## References

[1] Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930–76966, 2024.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. corr abs/2502.13923 (2025). arXiv preprint arXiv:2502.13923, 2025.

[4] Google DeepMind. Gemini 3 flash - model card. Technical report, Google DeepMind, 2025. URL https: //storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf.

[5] Google DeepMind. Gemini 3.1 pro model card. Technical report, Google DeepMind, 2026. URL https: //storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf.

[6] Google DeepMind. Gemini 3.6 flash model card. Technical report, Google DeepMind, 2026. URL https: //storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-6-Flash-Model-Card.pdf.

[7] DeepSeek-AI. Deepseek-v4.1-flash: Pushing the limits of kv cache compression. Technical report, DeepSeek-AI, 2026. URL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek\_V41\_Tech\_ Report.pdf.

[8] Yuhao Dong, Shulin Tian, Shuai Liu, Shuangrui Ding, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, and Ziwei Liu. Demo-icl: In-context learning for procedural video knowledge acquisition. arXiv preprint arXiv:2602.08439, 2026.

[9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118. IEEE, 2025.

[10] Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, et al. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026.

[11] Danijar Hafner. Benchmarking the spectrum of agent capabilities. arXiv preprint arXiv:2109.06780, 2021.

[12] Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P Xing, Ion Stoica, Tajana Rosing, Haojian Jin, and Hao Zhang. lmgame-bench: How good are llms at playing games? In International Conference on Learning Representations, volume 2026, pages 81308–81356, 2026.

[13] Lawrence Jang, Yinheng Li, Dan Zhao, Charles Ding, Justin Lin, Paul Pu Liang, Rogerio Bonatti, and Kazuhito Koishida. Videowebarena: Evaluating long context multimodal agents with video understanding web tasks. In International Conference on Learning Representations, volume 2025, pages 36934–36958, 2025.

[14] Yixing Jiang, Jeremy Irvin, Ji Hun Wang, Muhammad Ahmed Chaudhry, Jonathan H Chen, and Andrew Y Ng. Many-shot in-context learning in multimodal foundation models. arXiv preprint arXiv:2405.09798, 2024.

[15] Kangsan Kim, Geon Park, Youngwan Lee, Woongyeong Yeo, and Sung Ju Hwang. Videoicl: Confidence-based iterative in-context learning for out-of-distribution video understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3295–3305. IEEE, 2025.

[16] Yan Li, Changyao Tian, Renqiu Xia, Ning Liao, Weiwei Guo, Junchi Yan, Hongsheng Li, Jifeng Dai, Hao Li, and Xue Yang. Learning adaptive and temporally causal video tokenization in a 1d latent space. arXiv preprint arXiv:2505.17011, 2025.

[17] Yan Li, Zezi Zeng, Yifan Yang, Yuqing Yang, Ning Liao, Weiwei Guo, Lili Qiu, Mingxi Cheng, Qi Dai, Zhendong Wang, et al. Mm-webagent: A hierarchical multimodal web agent for webpage generation. arXiv preprint arXiv:2604.15309, 2026.

[18] Mingxin Liu, Shuran Ma, Shibei Meng, Xiangyu Zhao, Zicheng Zhang, Shaofeng Zhang, Zhihang Zhong, Peixian Chen, Haoyu Cao, Xing Sun, et al. Rise-video: Can video generators decode implicit world rules? arXiv preprint arXiv:2602.05986, 2026.

[19] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, pages 52989–53046, 2024.

[20] Xiaolin Liu, Yilun Zhu, Xiangyu Zhao, Xuehui Wang, Yan Li, Xin Li, Haoyu Cao, Xing Sun, Shaofeng Zhang, Xu Yang, et al. Moment-video: Diagnosing temporal fidelity of video mllms on momentary visual events. arXiv preprint arXiv:2606.02522, 2026.

[21] OpenAI. Gpt-5.6 system card. Technical report, OpenAI, 2026. URL https://deploymentsafety.openai.com/ gpt-5-6/gpt-5-6.pdf.

[22] Mingyu Ouyang, Siyuan Hu, Kevin Qinghong Lin, Hwee Tou Ng, and Mike Zheng Shou. Gameworld: Towards standardized and verifiable evaluation of multimodal game agents. arXiv preprint arXiv:2604.07429, 2026.

[23] Davide Paglieri, Bart lomiej Cupia l, Samuel Coward, Ulyana Piterbarg, Maciej Wo lczyk, Akbir Khan, Eduardo Pignatelli, Lukasz Kuci´nski, Lerrel Pinto, Rob Fergus, et al. Balrog: Benchmarking agentic llm and vlm reasoning on games. In International Conference on Learning Representations, volume 2025, pages 96666–96702, 2025.

[24] Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, ByeongUk Lee, Jaeyoung Hwang, Jaewoo Ahn, et al. Orak: A foundational benchmark for training and evaluating llm agents on diverse video games. In International Conference on Learning Representations, volume 2026, pages 60684–60744, 2026.

[25] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

[26] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https://qwen.ai/ blog?id=qwen3.6-27b.

[27] Anian Ruoss, Fabio Pardo, Harris Chan, Bonnie Li, Volodymyr Mnih, and Tim Genewein. Lmact: A benchmark for in-context imitation learning with long multimodal demonstrations. arXiv preprint arXiv:2412.01441, 2024.

[28] Bytedance Seed. Seed2.0 model card: Towards intelligence frontier for real-world complexity. arXiv preprint arXiv:2607.00248, 2026. URL https://arxiv.org/abs/2607.00248.

[29] ByteDance Seed. Seed2.1 model card: Agentic intelligence for productivity. Technical report, ByteDance Seed, 2026. URL https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2.1/ Seed2\_1\_Model\_Card.pdf.

[30] Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607.02770.

[31] Kimi Team. Kimi k3: Open frontier intelligence, 2026. URL https://arxiv.org/abs/2607.24653.

[32] V Team. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv.org/abs/2507.01006.

[33] Nicholas R Waytowich, Devin White, MD Sunbeam, and Vinicius G Goecks. Atari-gpt: Benchmarking multimodal large language models as low-level policies in atari games. arXiv preprint arXiv:2408.15950, 2024.

[34] LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. URL https://arxiv.org/abs/2506.03569.

[35] XiaomiMiMo. Mimo-v2.5. https://huggingface.co/collections/XiaomiMiMo/mimo-v25, 2026.

[36] Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly eficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

[37] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report, 2025. URL https://arxiv.org/abs/2509.17765.

[38] Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, et al. Skillopt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904, 2026.

[39] Kuan Zhang, Dongchen Liu, Qiyue Zhao, Jinkun Hou, Xinran Zhang, Qinlei Xie, Miao Liu, and Yiming Li. Gameverse: Can vision-language models learn from video-based reflection? arXiv preprint arXiv:2603.06656, 2026.

[40] Xiangyu Zhao, Lin Lin, Tianhao Liang, Yifan Zhou, Wenhao Chai, Yuzhe Gu, Weiyun Wang, Kai Chen, Gen Luo, Junchi Yan, et al. Mm-helix: Boosting multimodal long-chain reflective reasoning with holistic platform and adaptive hybrid policy optimization. In International Conference on Learning Representations, volume 2026, pages 115891–115943, 2026.

[41] Yongshuo Zong, Ondrej Bohdal, and Timothy Hospedales. Vl-icl bench: The devil in the details of multimodal in-context learning. In International Conference on Learning Representations, volume 2025, pages 100058–100100, 2025.

## Appendix

## A Benchmark Construction Details

## A.1 Environment Selection and Curation

Candidate pool. We screened more than 100 candidate environments drawn from the Arcade Learning Environment (ALE), MiniGrid, Procgen, Gymnasium Toy Text and Classic Control, HighwayEnv, third-party environment packages, and internally implemented environments. The final pool contains 37 environments: 18 ALE environments, seven Procgen environments, three MiniGrid environments, three Classic Control tasks, two Toy Text tasks, two third-party environments, one HighwayEnv task, and CustomBreakout. Table 6 consolidates the four curation records; spelling, separator, alias, and version variants are merged into one canonical entry.

Table 6 Deduplicated candidate-environment inventory (126 listed candidates; 37 included). Family-level entries indicate that an entire suite was screened; the included benchmark environments are marked in the final column.
<table><tr><td>Environment</td><td>Source library</td><td>Included</td></tr><tr><td>Acrobot-v1</td><td>Gymnasium Classic Control</td><td>Yes</td></tr><tr><td>ALE/Adventure-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Alien-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Assault-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Asterix-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Atlantis-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/BankHeist-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/BattleZone-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/BeamRider-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Berzerk-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Bowling-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/ChopperCommand-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/CrazyClimber-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/DemonAttack-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/DonkeyKong-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/DoubleDunk-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Enduro-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Entombed-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Freeway-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Frogger-v5 ALE/Frostbite-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Gravitar-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/IceHockey-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Jamesbond-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Kaboom-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Kangaroo-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/KeystoneKapers-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/KingKong-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/MontezumaRevenge-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/MsPacman-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/NameThisGame-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Phoenix-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Pitfall-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Pong-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Riverraid-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/RoadRunner-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Robotank-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Seaquest-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Skiing-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Solaris-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/StarGunner-v5</td><td>ALE ALE</td><td>No</td></tr><tr><td>ALE/Surround-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Tennis-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/TimePilot-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Trondead-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Turmoil-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Tutankham-v5</td><td>ALE</td><td>Yes</td></tr><tr><td>ALE/Venture-v5</td><td></td><td>No</td></tr><tr><td></td><td>ALE</td><td>No</td></tr><tr><td>ALE/VideoPinball-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/WizardOfWor-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/YarsRevenge-v5</td><td>ALE</td><td>No</td></tr><tr><td>ALE/Zaxxon-v5</td><td>ALE</td><td>No</td></tr><tr><td>Blackjack-v1</td><td>Gymnasium Toy Text</td><td>No</td></tr><tr><td>CarRacing-v3</td><td>Gymnasium Box2D</td><td>No</td></tr><tr><td>CartPole-v1</td><td>Gymnasium Classic Control</td><td>Yes</td></tr><tr><td>CliffWalking-v1</td><td>Gymnasium Toy Text</td><td>Yes</td></tr><tr><td>CustomBreakout</td><td>Internal custom environment</td><td>Yes</td></tr><tr><td>CustomLavaCrossing-v0</td><td>MiniGrid-derived custom environment</td><td>Yes</td></tr><tr><td>CustomMultiRoom-vO</td><td>MiniGrid-derived custom environment</td><td>Yes</td></tr><tr><td>CustomUnlockPickup-v0</td><td>MiniGrid-derived custom environment</td><td>Yes</td></tr><tr><td>generals.io</td><td>Third-party web environment</td><td>No</td></tr><tr><td>highway-v0</td><td>HighwayEnv</td><td>Yes</td></tr><tr><td>intersection-v1</td><td>HighwayEnv</td><td>No</td></tr><tr><td>LunarLander-v3</td><td>Gymnasium Box2D</td><td>No</td></tr><tr><td>merge-v0</td><td>HighwayEnv</td><td>No</td></tr><tr><td>MiniGrid-Empty-16x16-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-DoorKey-16x16-v0</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-FourRooms-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-Dynamic-0bstacles-16x16-v0</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-LavaGapS7-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-LavaCrossingS11N5-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-Unlock-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-UnlockPickup-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-MultiRoom-N6-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniGrid-KeyCorridorS6R3-vO</td><td>MiniGrid 3.0.0</td><td>No</td></tr><tr><td>MiniWorld-CollectHealth-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-FourRooms-y0</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-Hallway-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-Maze-v0</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-OneRoom-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-Pickup0bjects-v0</td><td>MiniWorld</td><td></td></tr><tr><td>MiniWorld-PutNext-y0</td><td>MiniWorld</td><td>No</td></tr><tr><td></td><td></td><td>No</td></tr><tr><td>MiniWorld-RoomObjects-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-Sidewalk-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-Sign-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-ThreeRooms-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-TMaze-v0</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-WallGap-vO</td><td>MiniWorld</td><td>No</td></tr><tr><td>MiniWorld-YMaze-v0</td><td>MiniWorld</td><td>No</td></tr><tr><td>MountainCar-vO</td><td>Gymnasium Classic Control</td><td>Yes</td></tr><tr><td>Pendulum-v1</td><td>Gymnasium Classic Control</td><td>No</td></tr><tr><td>procgen-bigfish-v0 procgen-bossfight-v0</td><td>Procgen</td><td>Yes</td></tr><tr><td></td><td>Procgen</td><td>Yes</td></tr><tr><td>procgen-caveflyer-v0</td><td>Procgen</td><td>Yes</td></tr><tr><td>procgen-chaser-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-climber-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-coinrun-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-dodgeball-v0</td><td>Procgen</td><td>Yes</td></tr><tr><td>procgen-fruitbot-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-heist-v0</td><td>Procgen</td><td>Yes</td></tr><tr><td>procgen-jumper-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-leaper-v0</td><td>Procgen</td><td>Yes</td></tr><tr><td>procgen-maze-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-miner-v0</td><td>Procgen</td><td>No</td></tr><tr><td>procgen-ninja-v0</td><td>Procgen</td><td>Yes</td></tr><tr><td>procgen-plunder-v0 procgen-starpilot-v0</td><td>Procgen</td><td>No</td></tr><tr><td>roundabout-v0</td><td>Procgen</td><td>No</td></tr><tr><td></td><td>HighwayEnv</td><td>No</td></tr><tr><td>stable-retro suite Taxi-v3</td><td>Gym Retro</td><td>No</td></tr><tr><td></td><td>Gymnasium Toy Text</td><td>Yes</td></tr><tr><td>Thirdparty-BlueSky-Gym</td><td>BlueSky Gym</td><td>No</td></tr><tr><td>aFlappyBird-v1</td><td>Flappy Bird third-party environment</td><td>Yes</td></tr><tr><td>tetris/Tetris</td><td>Tetris Gymnasium</td><td>Yes</td></tr><tr><td>two-way-v0</td><td>HighwayEnv</td><td>No</td></tr><tr><td>exit-v0</td><td>HighwayEnv</td><td>No</td></tr><tr><td>u-turn-v0</td><td>HighwayEnv</td><td>No</td></tr><tr><td>ViZDoom: Basic</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Deadly Corridor</td><td></td><td></td></tr><tr><td>ViZDoom: Deathmatch</td><td>ViZDoom</td><td>No</td></tr><tr><td></td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Defend the Center</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Defend the Line</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Health Gathering</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Health Gathering Supreme</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: My Way Home</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Predict Position</td><td>ViZDoom</td><td>No</td></tr><tr><td>ViZDoom: Take Cover</td><td>ViZDoom</td><td>No</td></tr></table>

Six-dimensional screening protocol. Each candidate was exercised through the same rendered-frame and discrete-action interface used at evaluation time. Curators inspected both ordinary play and failure trajectories and applied the following six criteria.

1. Visual legibility. A human must be able to locate the controllable agent, task-relevant objects, hazards, goals, and visible status indicators from the rendered frames alone. Candidates were rejected when essential state was hidden, represented by ambiguous symbols, or only recoverable from simulator internals.

2. Interaction stability and distinctiveness. Repeated resets and action probes must produce valid frames, consistent action–response timing, and reproducible termination behavior. We rejected environments with hangs, severe rendering bugs, long non-interactive death animations that consumed the evaluation budget, or frames that appeared to precede rather than follow the issued action. We also removed candidates that were efectively duplicates of an already selected environment.

3. Policy discriminability. Rule-informed play must consistently outperform random or reflex-only behavior. Environments dominated by luck, precise motor timing, or an action-independent score were excluded, because their outcomes do not diagnose whether the model inferred and used the demonstrated rules.

4. Reasonable diagnostic horizon. A candidate must expose a meaningful diference between weak and strategic play within a practical number of model decisions, typically no more than 100. During pilot runs, environments in which the first consequential encounter or reward appeared only after roughly 50 idle decisions were either re-initialized to a later state or removed.

5. Frame-skip coherence. After frame skipping, a human must still be able to reconstruct the motion and status of every task-relevant object. We tuned the skip and NOOP-fill parameters so that turns, collisions, projectile motion, and other critical transitions were not skipped wholesale; otherwise the candidate was rejected.

6. Timely outcome feedback. Success, failure, and incremental progress must be visible or recoverable from stable environment signals within the evaluation horizon. When a useful environment exposed an inconsistent raw reward or termination flag, an environment-specific wrapper converted it into a documented pass, score, and ending signal.

Environment customization and standardization. Max Steps caps model-issued decisions after initialization; reaching the cap truncates an episode. Frame Skip is the outer repeat count N for one model action. With NOOP Fill, the chosen action is held for only the first K Action Steps (K ≤ N), and the remaining slots use NOOP. This makes slow environments advance visibly while preserving short, precise control impulses. ALE additionally disables sticky actions by setting repeat action probability=0. Its native emulator skip is four for the reported tasks except RoadRunner (16); Tennis advances the same four primitive frames through four unit-frame steps so that ball-direction reversals remain observable. Table 7 reports all task-level values and the manual wrappers used to normalize progress, pass conditions, and termination.

Table 7 Per-environment evaluation configurations. “Frame skip” is the outer repeat count applied to each model decision; “action steps” is the number of repeated slots that retain the issued action before NOOP filling. A set reports all values used across that environment’s task configurations.
<table><tr><td>Environment</td><td>Max steps</td><td>Frame skip</td><td>Action steps</td><td>Manual modifications</td></tr><tr><td>Acrobot-v1</td><td>58, 61, 78, 81, 82</td><td>2</td><td>2</td><td>Dense score from the increase in maximum endpoint height; pass once the endpoint rises above the pivot.</td></tr><tr><td>aFlappyBird-v1</td><td>136, 140, 150</td><td>1</td><td>1</td><td>Remove per-step survival reward; pass after one pipe and end successfully after three pipes.</td></tr><tr><td>ALE/Alien-v5</td><td>60, 62, 66, 76</td><td>2</td><td>2</td><td>Suppress one-shot 500-point bonuses, use 75/150-point pass/end thresholds, and terminate on life loss.</td></tr><tr><td>ALE/Assault-v5</td><td>42, 46, 48</td><td>2</td><td>2</td><td>Normalize to a 63-point target, add a completion-time bonus, and terminate on life loss.</td></tr><tr><td>ALE/Asterix-v5</td><td>50</td><td>3</td><td>1</td><td>Full semantic action set, NOOP filling, life-loss termina- tion, and calibrated death penalty.</td></tr><tr><td>ALE/BattleZone-v5</td><td>50</td><td></td><td>2</td><td>ALE modes 1-3, preloaded state sequences, and life-loss termination.</td></tr><tr><td>ALE/Berzerk-v5</td><td>70</td><td></td><td>4</td><td>ALE modes 1-9 and 16–18 with life-loss termination.</td></tr><tr><td>ALE/ChopperCommand-v5</td><td>50</td><td></td><td>2</td><td>Full semantic action set and life-loss termination.</td></tr><tr><td>ALE/DemonAttack-v5</td><td>50</td><td></td><td>3</td><td>Difficulty 1, five progress bands constructed by preloading, 90-point target, and life-loss termination.</td></tr><tr><td>ALE/Enduro-v5</td><td>50</td><td></td><td>4</td><td>Four progress bands produced by long initial skips and</td></tr><tr><td>ALE/Freeway-v5</td><td>22, 28, 33, 61, 69,</td><td></td><td>2</td><td>human action sequences. Modes 0/2 at difficulty 1; RAM-based forward progress,</td></tr><tr><td>ALE/Frostbite-v5</td><td>70, 72, 76, 81 50</td><td></td><td>1</td><td>halfway pass criterion, and crossing termination. Reduced action set, NOOP filling, life-loss termination,</td></tr><tr><td></td><td></td><td></td><td></td><td>and calibrated death penalty Full semantic action set and life-loss termination.</td></tr><tr><td>ALE/MsPacman-v5 ALE/Pong-v5</td><td>50 50</td><td></td><td>2 1</td><td>Reduced action set, NOOP filling, and RAM-based ball/-</td></tr><tr><td>ALE/Riverraid-v5</td><td>50</td><td></td><td></td><td>paddle state tracking. RAM initialization of stage/life state, preloaded starts,</td></tr><tr><td>ALE/RoadRunner-v5</td><td></td><td></td><td>2</td><td>and life-loss termination. Native ALE skip 16; suppress 1000-point bursts, use an</td></tr><tr><td>ALE/Seaquest-v5</td><td>26, 34, 46, 48, 54</td><td></td><td>1</td><td>1100-point target, and terminate on life loss. NOOP filling, auto-respawn support, life-loss termination,</td></tr><tr><td>ALE/Tennis-v5</td><td>50 30</td><td></td><td>2</td><td>and RAM-based diver/surfacing bonuses. Execute four unit-frame ALE steps to detect ball-direction</td></tr><tr><td></td><td></td><td></td><td>4</td><td>reversals; auto-swing before evaluation and terminate on life loss.</td></tr><tr><td>ALE/Trondead-v5</td><td>50</td><td>3</td><td>3</td><td>RAM-selected variants (values 0,1,2,4,7), variant-specific score targets, and life-loss termination.</td></tr><tr><td>ALE/Turmoil-v5 CartPole-v1</td><td>50</td><td>1</td><td>1</td><td>Full semantic action set and life-loss termination.</td></tr><tr><td>CliffWalking-v1</td><td>30</td><td>2</td><td>2</td><td>Standard discrete control with seed-varying initial states. Zero ordinary step cost, terminate cliff falls, and award</td></tr><tr><td></td><td>16, 20, 24, 26</td><td>1</td><td>1</td><td>normalized forward-progress and goal scores.</td></tr><tr><td>CustomBreakout</td><td>46, 50, 80, 92</td><td>1</td><td>1</td><td>One life, ball speed 5, 1 brick row, 2–3 columns, config- urable action subset, and explicit loss/win termination.</td></tr><tr><td>CustomLavaCrossing-v0</td><td>40</td><td>1</td><td>1</td><td>Partial observation; 11 × 11 layout with five crossings; exploration/goal/efficiency/death shaping.</td></tr><tr><td>CustomMultiRoom-vO</td><td>70</td><td></td><td>1</td><td>Partial observation; grid sizes 15/20/25 with six rooms; exploration/goal/efficiency/death shaping.</td></tr><tr><td>CustomUnlockPickup-v0</td><td>40</td><td></td><td>1</td><td>Partial observation; room sizes 7/9/11; key-door-pickup- aware dense shaping.</td></tr><tr><td>highway-v0</td><td>52, 54, 58, 60, 62, 64, 70, 78, 80, 82, 86, 96, 120</td><td></td><td>1</td><td>NOOP filling; discrete meta-actions; 4/5 lanes, density 1.5/1.9, normalized high-speed score, and crash termina-</td></tr><tr><td>MountainCar-vO</td><td>38, 46, 58, 70</td><td>3</td><td>3</td><td>tion. Dense normalized position progress; pass after traversing half the remaining distance to the goal.</td></tr><tr><td>procgen-bigfish-v0</td><td>22, 32, 72, 78, 84</td><td>3</td><td>3</td><td>Seeded Procgen layout; normalize cumulative reward to a two-point target and mark death as failure.</td></tr><tr><td>procgen-bossfight-v0</td><td>44, 52, 62, 74, 84</td><td>4</td><td>4</td><td>Seed-specific normalization targets for five generated lay- outs.</td></tr><tr><td>procgen-caveflyer-v0</td><td>22, 26, 32, 38, 60</td><td>3</td><td>3</td><td>Normalize cumulative reward to one; environment termi-</td></tr><tr><td>procgen-dodgeball-v0</td><td>86, 90, 94, 100</td><td>3</td><td>3</td><td>nation without completion is failure. Seed-specific score normalization; reward at least 10 marks</td></tr><tr><td>procgen-heist-v0</td><td>28, 39, 43, 48, 52</td><td>1</td><td>1</td><td>completion. Seed-specific normalization; collecting reward at least 2</td></tr><tr><td>procgen-leaper-v0</td><td>36, 56, 58, 60, 72</td><td>4</td><td>4</td><td>establishes a pass. Normalize cumulative reward to one; premature termina-</td></tr><tr><td>procgen-ninja-v0</td><td>60, 76, 88, 136,</td><td>1</td><td>1</td><td>tion is failure. Pass at 0.5 normalized progress and complete at one.</td></tr><tr><td>Taxi-v3</td><td>146 30</td><td>1</td><td>1</td><td>Five seeded passenger/destination states with state-</td></tr></table>

Human-driven initialization. Random seeds do not substantially alter the opening state of most ALE environments. For these environments, relying on seeds alone would repeatedly test nearly the same opening and would leave later mechanics outside the context window. We therefore recorded short, deterministic human action sequences and replayed them before the model’s first decision. The replayed actions are excluded from the model’s decision budget, and the resulting frame is treated as the task’s start state.

Figure 5 gives three concrete cases from the stored evaluation runs. The Asterix sequence (three NOOP actions followed by Up and Up+Left) creates a collision-oriented lane state. The Frostbite sequence (Down+Right twice, Down twice, then two NOOPs) produces the curator-labelled “complex multi-risk” state. The Seaquest sequence applies three Down actions after a 230-frame warm-up, placing the submarine underwater with an incoming threat from the left. These states are valuable because they expose dense, rule-dependent decisions that rarely occur at the default reset frame.

Expert calibration across environments. Curators did not expose native environment rewards directly and then compare unlike scales. For every environment, experts jointly adjusted the target start state, decision horizon, pass condition, and score reference so that the main score bands represented comparable degrees of progress and attainability. Calibration was performed through expert play before model evaluation; model outcomes were not used to define the task tags or scoring targets. A diferent expert then played the complete task from its stored start state to verify that its objective was attainable under the configured interaction budget.

## A.2 Demonstration Video Creation

Recording protocol. Demonstrations were recorded by experts who had each accumulated more than 100 hours of environment play and analysis. Recording was iterative rather than one-shot: experts replayed each environment, inspected whether the clip actually exposed its key dynamics and failure conditions, and re-recorded or re-captioned examples when an action was misleading, a rule was omitted, or a trajectory overemphasized an accidental strategy. Demonstrations are references for discovering rules and dynamics, not claims of optimal play.

Length, frame rate, resolution, and encoding. Across 342 task configurations, the benchmark makes 576 demonstration-video assignments drawn from 148 distinct source clips. Each task receives between one and four clips: 155 tasks receive one, 145 receive two, 37 receive three, and five receive four. Thus, 576 counts task-level clip assignments, including reuse of a source clip across configurations, rather than 576 unique recordings. At the task level, the total demonstration context ranges from 10 to 80 sampled frames (mean 45.73, median 45). All stored task videos use 1 frame per second and H.264 in an MP4 container. The current assets contain 12 environment-dependent resolutions, ranging from 160 × 250 to 800 × 840 pixels; the most common is 160 × 250 (207 of 342 task-level demonstration packages). Captioned frames add a 40-pixel black strip below the unmodified environment image. Figure 6 shows the task-level context-length distribution.

For backends with native or extension-based video input, the sampled frames are encoded as MP4 and sent as video. For image-only backends, including the OpenAI image-input path used for models without native video ingestion, the identical ordered frames are sent as JPEG images. Thus the transport difers, while the sampled visual evidence and its order remain unchanged. Live interaction frames are always transmitted as individual images.

Caption generation. The initial frame is labelled Start State. Each subsequent frame uses the naturallanguage template Step t: Action: ⟨canonical-action⟩; the numeric variant uses Step t: Action ⟨id⟩. A terminal copy of the final frame is labelled Victory, Terminated, Time Out, or Draw. Captions are centered in the bottom strip with a black outline and use the same canonical action names accepted by the action parser. All 342 released task configurations select the natural-language captioned source for their demonstration context.

![](images/a85813821dbae781ae0cc648a62dc635a7b4cb25572c3f5bf3ad33fec11ef102.jpg)

![](images/bfb374f0860d7bb75988b9df0a5b17841c19165ebde6d574a16fd284536b2e88.jpg)

![](images/24b9b4126f774faa809ae7e6fc41c2c50ad0b1a9dba741b180d8699653aa882e.jpg)

![](images/3d9d6ed4f2169a832a7fa577695b7e60cd287651b2804f3fa143966873e5168b.jpg)  
(d) Asterix: initialized state.

![](images/e1a5e5de5ebd9b8294dc90667d3d3c6e633d6ef199ff6a767afec7e9e9c499eb.jpg)  
(e) Frostbite: multi-risk state.

![](images/0c1a8fc467f44efb9325d2b16afab2f323b79b30a1c1d0d620b7f0b237474453.jpg)  
(f) Seaquest: submerged state.  
Figure 5 Examples of human-driven initialization. The Asterix upper image is captured directly after environment reset; the other upper images follow their configured warm-up skips. Each corresponding lower image is the state shown to the evaluated model after replaying the stored human action sequence.

Demonstration-quality validation. An independent group reviewed each demonstration together with its text and judged whether the package conveyed all rules required to operate the environment. Failed packages were returned for another recording or captioning pass. After this iterative correction process, the final rule-comprehension accuracy was approximately 98%.

## A.3 Task Pool Expansion Strategy

Parameterized modifications. We use random seeds when they materially change the initial layout, object placement, trafic pattern, or procedural level. Seeds are selected explicitly and stored per task rather than sampled at evaluation time. For environments whose seeds leave the visible opening almost unchanged—most notably many ALE environments—we prefer initial-frame skipping and human action-sequence replay. This avoids presenting a demonstration and test task with efectively identical openings while still keeping each task deterministic. Table 8 lists every stored seed and initial-skip value.

![](images/e43a9c33a44b3c7aa5a2da6c9857c7c46c30ab71031ffe569b5efd1af9507bec.jpg)  
Figure 6 Distribution of total demonstration context across the 342 task configurations. Because the sampling rate is 1 FPS, sampled-frame count equals duration in seconds.

Table 8 Random seeds and initial-frame-skip values used by the 342 task configurations. A dash means that no initia skip is stored for that environment.
<table><tr><td>Environment</td><td>Random seeds</td><td>Initial skipped frames</td></tr><tr><td>Acrobot-v1</td><td>41</td><td></td></tr><tr><td>aFlappyBird-v1</td><td>43, 47, 50, 69, 77</td><td>一</td></tr><tr><td>ALE/Alien-v5</td><td>41</td><td>12</td></tr><tr><td>ALE/Assault-v5</td><td>41</td><td></td></tr><tr><td>ALE/Asterix-v5</td><td>43</td><td>98, 140, 150, 200, 270</td></tr><tr><td>ALE/BattleZone-v5</td><td>43</td><td>36, 45, 110</td></tr><tr><td>ALE/Berzerk-v5</td><td>43</td><td>10</td></tr><tr><td>ALE/ChopperCommand-v5</td><td>43</td><td>40</td></tr><tr><td>ALE/DemonAttack-v5</td><td>41</td><td>20,50</td></tr><tr><td>ALE/Enduro-v5</td><td>41</td><td>800, 850, 900, 950, 1050, 1850, 1950, 2000, 2150, 2350, 2450, 2550</td></tr><tr><td>ALE/Freeway-v5</td><td>43</td><td></td></tr><tr><td>ALE/Frostbite-v5</td><td>43</td><td>15, 50, 80, 100, 109</td></tr><tr><td>ALE/MsPacman-v5</td><td>41</td><td>66</td></tr><tr><td>ALE/Pong-v5 ALE/Riverraid-v5</td><td>41</td><td>40</td></tr><tr><td></td><td>43</td><td>20</td></tr><tr><td>ALE/RoadRunner-v5</td><td>41</td><td></td></tr><tr><td>ALE/Seaquest-v5 ALE/Tennis-v5</td><td>43</td><td>33, 87, 150, 230, 350</td></tr><tr><td>ALE/Trondead-v5</td><td>43</td><td>1</td></tr><tr><td>ALE/Turmoil-v5</td><td>43</td><td></td></tr><tr><td>CartPole-v1</td><td>43</td><td>65</td></tr><tr><td></td><td>41, 42, 44</td><td></td></tr><tr><td>CliffWalking-v1 CustomBreakout</td><td>43</td><td></td></tr><tr><td>CustomLavaCrossing-v0</td><td>41 43, 1682, 1952, 3659, 5235, 6892, 8043, 8234, 8386,</td><td></td></tr><tr><td>CustomMultiRoom-vO</td><td>8976, 9119, 9381, 9848 42, 189, 1323, 1607, 1815, 3932, 4300, 4707, 5149,</td><td></td></tr><tr><td>CustomUnlockPickup-v0</td><td>7227, 7368, 8033, 9610 43, 898, 1251, 1470, 3445, 4121, 4885, 5578, 5697,</td><td></td></tr><tr><td>highway-v0</td><td>5844, 6167, 6509, 8415, 8816, 9134 41, 47, 51, 58, 79, 88, 91, 99, 101, 137, 1241, 1299,</td><td></td></tr><tr><td>MountainCar-v0</td><td>1491, 1565, 1896 41</td><td></td></tr><tr><td>procgen-bigfish-v0 procgen-bossfight-v0</td><td>45, 91, 104, 178, 265</td><td>6, 10, 15, 20</td></tr><tr><td>procgen-caveflyer-v0</td><td>41, 49, 51, 59, 78</td><td>14, 28, 30, 40</td></tr><tr><td>procgen-dodgeball-v0</td><td>42, 43, 53, 79, 89</td><td></td></tr><tr><td>procgen-heist-v0</td><td>101, 202, 207, 209, 321</td><td></td></tr><tr><td></td><td>44, 64, 66, 73, 84</td><td></td></tr><tr><td>procgen-leaper-v0</td><td>60, 64, 80, 101, 257</td><td></td></tr><tr><td>procgen-ninja-v0</td><td>265, 291, 438, 552, 3896</td><td></td></tr><tr><td>Taxi-v3</td><td>42, 43, 44, 45, 46</td><td></td></tr><tr><td>tetris/Tetris</td><td>43, 47, 271, 360, 535</td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

From environments to tasks. Each environment contributes one canonical base configuration. The remaining stored configurations are classified by construction mechanism: configurations without a preloaded action sequence are parameterized variants, whereas those with a sequence are human-initialized variants. In raw storage, 174 of 342 configurations contain an action sequence. Eight environments have no uninitialized configuration, so one initialized configuration from each is designated as its canonical base task; the remaining mutually exclusive counts are therefore 37 base tasks, 139 parameterized variants, and 166 additional human-initialized variants. Table 9 provides the complete 37-to-342 mapping.

Table 9 Expansion from 37 base environments to 342 task configurations. One canonical configuration is counted as the base task for each environment. Remaining configurations without a preloaded action sequence are counted as parameterized variants; remaining configurations with such a sequence are counted as human-initialized variants.

<table><tr><td>Environment</td><td>Base</td><td>Parameterized variants</td><td>Human-init. variants</td><td>Total</td></tr><tr><td>Acrobot-v1</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>aFlappyBird-v1</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>ALE/Alien-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Assault-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Asterix-v5</td><td>1</td><td>3</td><td>1</td><td>5</td></tr><tr><td>ALE/BattleZone-v5</td><td>1</td><td>0</td><td>14</td><td>15</td></tr><tr><td>ALE/Berzerk-v5</td><td>1</td><td>11</td><td>5</td><td>17</td></tr><tr><td>ALE/ChopperCommand-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/DemonAttack-v5</td><td>1</td><td>0</td><td>24</td><td>25</td></tr><tr><td>ALE/Enduro-v5</td><td>1</td><td>0</td><td>19</td><td>20</td></tr><tr><td>ALE/Freeway-v5</td><td>1</td><td>3</td><td>16</td><td>20</td></tr><tr><td>ALE/Frostbite-v5</td><td>1</td><td>1</td><td>3</td><td>5</td></tr><tr><td>ALE/MsPacman-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Pong-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Riverraid-v5</td><td>1</td><td>19</td><td>5</td><td>25</td></tr><tr><td>ALE/RoadRunner-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Seaquest-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Tennis-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>ALE/Trondead-v5</td><td>1</td><td>0</td><td>24</td><td>25</td></tr><tr><td>ALE/Turmoil-v5</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>CartPole-v1</td><td>1</td><td>1</td><td>3</td><td>5</td></tr><tr><td>CliffWalking-v1</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>CustomBreakout</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>CustomLavaCrossing-v0</td><td>1</td><td>14</td><td>0</td><td>15</td></tr><tr><td>CustomMultiRoom-vO</td><td>1</td><td>14</td><td>0</td><td>15</td></tr><tr><td>CustomUnlockPickup-v0</td><td>1</td><td>14</td><td>0</td><td>15</td></tr><tr><td>highway-v0</td><td>1</td><td>19</td><td>0</td><td>20</td></tr><tr><td>MountainCar-vO</td><td>1</td><td>0</td><td>4</td><td>5</td></tr><tr><td>procgen-bigfish-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>procgen-bossfight-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>procgen-caveflyer-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>procgen-dodgeball-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>procgen-heist-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>procgen-leaper-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>procgen-ninja-v0</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>Taxi-v3</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>tetris/Tetris</td><td>1</td><td>4</td><td>0</td><td>5</td></tr><tr><td>Total</td><td>37</td><td>139</td><td>166</td><td>342</td></tr></table>

## B Experimental Setup and Implementation

## B.1 Model Evaluation Protocol

Multimodal turn structure. Every backend is evaluated on the same fixed task configurations and underlying demonstration trajectories. The first turn contains the environment identifier, optional environment-specific rules, maximum decision horizon, legal action names, demonstration context, and current initial-state image. Demonstrations are supplied only in this first message; later turns append the previous action and the new current-state image to the conversation history. All 342 configurations hide scalar reward feedback, so the model must infer progress from visual state changes. Media are serialized through each model’s supported provider interface; the leaderboard therefore measures the complete deployed system, including its native modality and context limitations. Table 10 specifies these provider-specific representations.

Each model is run once on every one of the 342 fixed task configurations. This single-pass protocol measures realized end-to-end performance over the complete suite, as is standard for costly frontier-model evaluation, rather than within-task sampling variance. The random reference is inexpensive and is run three times per task to stabilize that reference floor. Environment-side seeds, initialization sequences, action spaces, and interaction budgets remain fixed across model rows; provider-side generation and context limits remain

Table 10 Provider-specific serialization of the multimodal evaluation input.
<table><tr><td>Backend</td><td>API format</td><td>Demonstration context</td><td>Current-state input</td></tr><tr><td>GPT / OpenAI</td><td>/responses (input_image) or OpenAI-compatible /chat/completions (image_url)</td><td>No video endpoint is used. The clip is de- composed into ordered JPEG frames, each embedded as a data:image/jpeg;base64 URL.</td><td>One Base64 JPEG image is sent on every turn.</td></tr><tr><td>Gemini</td><td>Native/v1beta/models/ {model}:generateContent</td><td>Frames are encoded at 1 FPS as an MP4 and sent as inline_data with MIME type video/mp4.</td><td>The current frame is inline_data with MIME type image/jpeg.</td></tr><tr><td>OpenRouter</td><td>OpenAI-compatible /chat/completionswith the OpenRouter video_- url extension</td><td>The Base64 MP4 data URL is rewritten as {type: video_url, video_url: {url: data:video/mp4;base64,...}}. Base64 video is routed to a video-capable provider.</td><td>One image_ur1 Base64 JPEG is sent on every turn.</td></tr><tr><td>Volcengine</td><td>No dedicated Volcengine adapter is implemented</td><td>Seed models reached through OpenRouter use the preceding video_url path and are restricted to the seed provider. A direct Ark endpoint can only reuse the generic OpenAI-compatible, per-frame JPEG path.</td><td>One Base64 JPEG through the se- lected compatible endpoint.</td></tr></table>

attributes of the deployed systems being compared.

Action parsing and retries. At each turn, the model identifies its executable action with a [Action:<name>] marker. The bracketed action name must exactly match the allowed action set for the current environment. Requiring the marker prevents action names mentioned elsewhere in the response from being executed accidentally. A missing, malformed, or out-of-space action triggers the retry message reproduced in Appendix B.3; the model receives at most three format retries. Transport or API errors are retried separately up to three times with exponential backof.

## B.2 Score Computation and Calibration

Independent pass and progress metrics. Pass conditions are predefined per task and evaluated independently of environment score. A pass may represent reaching a goal, triggering a required event, attaining a score threshold, or surviving the configured horizon. Environment score instead records graded progress at the end of the episode, including progress retained when an episode times out, terminates early, exhausts the valid-action retries, or exceeds a provider context limit.

Environment-semantic score families. The implementation uses the final environment-specific scoring rules released with the benchmark. Environments whose wrappers already emit the benchmark progress scale retain that value without a second normalization. Cumulative-reward tasks use a capped ratio 100 min(max(v, 0)/u, 1), where v is the task-relevant reward and u is an expert-calibrated environment-level or task-level reference. Event-driven environments use capped monotonic counts, such as balls returned, targets destroyed, or ice platforms reached. Survival and navigation tasks use horizon ratios or shortest-path eficiency. Structured tasks use explicit progress states: Taxi separates correct pickup, incomplete delivery, and eficient completion; UnlockPickup accumulates key, door, and object milestones; and BattleZone combines first-hit timing with survival.

Bounds and terminal handling. Ratio scores are floored at zero and capped at 100; selected environments additionally cap the retained score after death. Event counters remain monotonic across life loss when the objective is cumulative. For shaped-reward environments, scoring uses goal-relevant accumulated progress while excluding feedback-only death penalties. These rules prevent a single raw-reward formula from conflating survival, task completion, and semantically diferent progress signals.

## B.3 Complete Playing Prompts

Prompt-role serialization. The implementation sends no system message through any provider. All task instructions and media are serialized directly into turn-specific user content blocks, followed by the model’s

assistant reply. The operational templates below are authoritative about output order: the action marker must appear on the first line, followed only by optional reasoning.

First-Turn User Prompt   
You are interacting with the environment: {game name}   
Environment Rules:   
{optional rule text}   
Max Steps: {max steps}   
Available actions:   
- {action name 1}   
- {action name 2}   
- {action name K}   
The next {M} video(s) show EXAMPLE GAMEPLAY footage from previous runs of this environment   
({game name}). These are NOT necessarily good or optimal play. They are provided ONLY to help   
you understand the environment’s dynamics: what objects do, what causes death or rewards, and   
how the environment reacts to actions. Do NOT treat these videos as the current environment   
state, and do NOT choose actions for them.   
Video {i}:   
<demonstration video i or ordered JPEG frames>   
The demonstrations end here. The next image is the CURRENT INITIAL STATE of the environment --   
this is the state you must act on. Choose your next action ONLY for this current state.   
Current initial state image:   
<current state image>   
Your task: Choose the best action for the CURRENT INITIAL STATE shown in the image above.   
IMPORTANT - Response format:   
Your response MUST start with your chosen action on the FIRST LINE in this exact format:   
[Action: <action name>]   
For example:   
[Action: FIRE]   
[Action: turn left]   
Use the EXACT action names from the list above.   
The square brackets [ ] are REQUIRED. The [Action: X] line MUST be the FIRST line of your   
response.   
You may add brief reasoning AFTER the action line.   
What action do you choose?

For image-only backends, each demonstration marker in the preceding template is replaced by the ordered JPEG frames and the sentence: “The following {N} image(s) are the ordered frames of Video {i}. Read them left-to-right, top-to-bottom as a single continuous clip.”

Subsequent-Turn User Prompt   
Step {step} - Result of your last action:   
Previous action: {last action name}   
The next image is the CURRENT STATE after executing your previous action.   
<current state image>   
Choose the next action for the CURRENT STATE shown in the image above.   
IMPORTANT: Your response MUST start with [Action: <action name>] on the FIRST LINE.   
You may add brief reasoning after.

The generic template can expose immediate and cumulative rewards, but every released benchmark con figuration sets hide reward=true; the two reward lines are therefore absent from the operational prompt

above.

Retry Message: First Turn   
Your action is INVALID.   
Your response MUST start with the action on the FIRST LINE:   
[Action: <action name>]   
The square brackets [ ] are REQUIRED.   
Use the EXACT action names I provided.   
Put [Action: ...] FIRST, then reasoning after.

## Retry Message: Subsequent Turns

Your action is INVALID and not in the action space.   
You MUST respond using the exact format:   
[Action: <action name>]   
Where <action name>is one of the action names I provided.   
The square brackets [ ] are REQUIRED.   
Please try again with a valid action.

## C Trajectory-Audit Protocol and Implementation

## C.1 Auditor and Inference Configuration

Each trajectory is submitted through two independent multimodal requests: one for information transfer evaluation and one for sequential failure attribution. They use the same evidence and generation configuration, but separate system instructions, rubrics, and output contracts; neither request contains the judgment produced by the other. Here, independent refers to request construction rather than statistical independence: the two diagnoses deliberately cross-cut the same decision-making process.

Both requests use the version of seed-evolving updated on August 1, 2026, through the Volcengine Responses API with temperature 0, low reasoning efort, a 50,000-token output limit, and full visual evidence. Demonstration videos are decoded at 1 FPS, and every current-state and ending-state image is transmitted with high-detail image processing. The same auditor and configuration are used throughout.

## C.2 Multimodal Evidence Construction

Each audit is grounded in the complete interaction log recorded during evaluation. The log reproduces the sequential multimodal conversation generated under the evaluation protocol in Appendix B.1 using the prompts reproduced in Appendix B.3. Each request to the acting model is preserved with its text and media blocks in their original multimodal order, followed by the model’s full response, including its visible reasoning and final answer. The first request retains the demonstration and initial-state frame, later requests retain their corresponding current-state frames, and the ending-labelled frame follows the final response. The audit request contains no field identifying the acting model or its provider; it identifies only the audited task and presents the corresponding interaction evidence.

Besides the interaction log, which provides the primary audit evidence, the auditor receives complementary textual information comprising the basic rules, detailed rules, and a demonstration-derived reference for the task. This reference is constructed once before trajectory auditing by analyzing the demonstrations for candidate visually exhibited rules absent from the basic rules and reusable state–action–outcome regularities. Across the 342 task references, the archive contains 1,106 task-level rule claims and 874 task-level strategy claims, with at least one claim of each type for every task. The reference is reused unchanged for every audit of that task, regardless of the acting model, and contains neither trajectory outcomes nor model-specific judgments. These materials support consistent task interpretation but are not ground-truth annotations: the auditor must verify the candidate information against the demonstration and interaction trajectory, whose

visual evidence remains authoritative.

## C.3 Criterion Instructions and Output Validation

Beyond the trajectory evidence, each audit request contains a criterion-specific system instruction, detailed rubric, and JSON output contract. The system instruction establishes the auditor’s role and general evidencehandling principles, requiring an independent, visually grounded assessment of the complete trajectory. The rubric translates the corresponding analytical criterion into an operational decision procedure. The transfer-evaluation rubric operationalizes transfer as an end-to-end judgment, requiring the auditor to identify the demonstrated rule or strategy and verify its appropriate state-conditional use in the live trajectory. The failure-attribution rubric defines the input, transformation, and output of each stage, then requires the auditor to reconstruct the rubric-defined functional information flow, test stages from upstream to downstream, and attribute only consequential deficiencies. The complete rubrics are reproduced in Appendix C.4.

Both transfer-evaluation judgments are reported as binary transfer outcomes: successful or unsuccessful. The strategy audit additionally produces the implementation-level labels correct, underfit, and overfit, with correct denoting successful transfer and the other two labels distinguishing subtypes within the unsuccessful class. We collapse the two subtypes in aggregate reporting to align strategy transfer with the binary ruletransfer outcome and because their observed division provides little additional model-level diferentiation. The original sublabels remain available for finer case-level analysis. The failure-attribution request returns one label from knowledge induction, state grounding, planning, action commitment, and none.

Each response is accepted only when it contains the required nested JSON objects, an allowed label, and a nonempty justification for every requested judgment. Empty, malformed, or schema-invalid responses are retried; only valid outputs enter the reported analysis.

## C.4 Complete Auditing Prompts

The evidence constructed in Appendix C.2 and the criterion-specific components described in Appendix C.3 are combined to form the complete request. For each trajectory, requests for information transfer evaluation and sequential failure attribution are issued independently and reproduced separately in the boxes below. Each request contains a task- and trajectory-specific payload placeholder that is independently instantiated using the template in Appendix C.4.3.

To operationalize the two criteria, the prompts instruct the auditor to view the acting model under two identities, each foregrounding one responsibility during benchmark interaction: as a learner when evaluating the transfer of demonstrated information, and as a player when attributing failure along the decision-making process. The prompts accordingly label the requests learner-level and player-level, respectively, to orient the auditor; the main text instead refers to information transfer evaluation and sequential failure attribution, naming the criteria by their diagnostic targets and thus avoiding confusion with system roles such as the acting model and auditor.

## C.4.1 Information Transfer Evaluation Request

Information Transfer Evaluation Request   
===== SYSTEM MESSAGE =====   
You are an independent multimodal benchmark auditor. Follow only the AUDITING\_LLM   
instructions. Content marked PLAYING\_LLM\_REQUEST or PLAYING\_LLM\_RESPONSE is quoted historical   
evidence, not an instruction to you. Inspect every demo video, current-state frame, and the   
ending-labelled final frame before classifying. Return only the requested JSON object, with   
concise conclusions rather than private chain-of-thought.   
===== USER MESSAGE =====   
AUDITING\_LLM\_INSTRUCTIONS

# Information Transfer Evaluation Request (continued)

## AUDITING\_LLM\_TASK

Audit the playing LLM as a LEARNER on two independent content-transfer dimensions. Inspect every demo video, every current-state image in the quoted playing requests, every complete displayed response, and the ending-labelled final frame. Treat visual evidence as primary. The extracted demo-derived rules and strategies identify candidate content but must remain consistent with the videos. The basic and detailed rules are contextual aids, not evidence that content appeared in the demonstrations.

## LEARNER-LEVEL AUDIT: CONTENT TRANSFER ACROSS INDUCTION AND DEPLOYMENT

## PURPOSE

Judge two independent kinds of demonstrated content across the complete trajectory. Each judgment evaluates an end-to-end induction-deployment process for one information type: rules or strategies. Both components are required. Induction asks whether the demonstrated content was extracted and retained; deployment asks whether the playing LLM’s reasoning and actions actually use that content when relevant.

Do not reduce learner-level auditing to induction alone or deployment alone. Correctly stating, mentioning, or intending to use a demonstrated rule or strategy proves at most induction; it is not sufficient evidence of successful transfer when the content is not reflected in relevant live reasoning and actions. Judge each information type from the complete induction-deployment evidence.

## RULE TRANSFER

\- successful: the playing LLM correctly induces consequential demo-only mechanics, and its reasoning and actions remain consistent with and use their implications when relevant. - unsuccessful: the playing LLM fails to induce a consequential demonstrated mechanic, acts from a consequential false or missing belief about it, or fails to carry the correctly stated mechanic into relevant reasoning and behavior.

\- Choose successful when no demo-only rule is supplied.

\- Do not choose unsuccessful for a mere local perception, timing, prediction, planning, or action-selection error when the complete evidence shows that the demonstrated mechanic was correctly induced and its implications were still used when relevant. If such an error reveals or causes failure to use the demonstrated mechanic itself, it is evidence of unsuccessful rule transfer.

## STRATEGY TRANSFER

\- correct: the playing LLM induces and actually applies the demonstrated state-conditional policy when relevant, adapting its concrete actions to the live visual conditions.

\- underfit: the playing LLM fails to induce the demonstrated policy, misses or replaces it in relevant live play, or applies only an insufficient fragment of it.

\- overfit: choose this label only when the playing LLM copies a concrete action sequence or timing pattern from the demo even though the live visual state materially differs from the demonstrated situation in which that pattern was used.

\- Before choosing overfit, identify all three links: the concrete action or timing pattern visibly demonstrated in the demo, the live turn or turns that copy it, and the material visual difference between the demo and live situations. If any link is absent, do not choose overfit. - Repeating an action learned within live play is not demo overfitting. A demo mention, action repetition, poor outcome, visual hallucination, perception error, timing or prediction error, or action error is also not sufficient evidence of overfitting.

\- Correctly describing, considering, intending, or attempting the demonstrated policy is not sufficient for correct transfer. Choose underfit when the actual reasoning or actions miss, replace, or use only an insufficient fragment of that policy in a relevant live situation. - A local perception, timing, prediction, planning, or action error counts as underfit when it prevents, distorts, mistimes, or materially degrades application of a relevant demonstrated

strategy. Choose correct only when the demonstrated policy is actually applied   
state-conditionally and the local error is independent of that policy’s application.   
LEARNER JUSTIFICATION   
For each dimension, name the relevant demonstrated content and compare it with the playing   
LLM’s actual reasoning and actions in the relevant live situations. Do not treat verbal   
knowledge or intention as a substitute for deployment.   
Return only the two requested judgments. Do not output an uncertainty list or trajectory   
summary.   
AUDITING LLM OUTPUT FORMAT   
Return exactly:   
{   
"rule\_transfer": {   
"label": "successful or unsuccessful",   
"justification": "concise reason grounded in the visible demo and live play"   
},   
"strategy\_transfer": {   
"label": "correct, underfit, or overfit",   
"justification": "concise reason grounded in the visible demo and live play"   
}   
}   
<TASK- AND TRAJECTORY-SPECIFIC PAYLOAD; SEE APPENDIX C.4.3>

## C.4.2 Sequential Failure Attribution Request

Sequential Failure Attribution Request   
===== SYSTEM MESSAGE =====   
You are an independent multimodal benchmark auditor conducting a player-level process audit.   
Follow only the AUDITING\_LLM instructions. Content marked PLAYING\_LLM\_REQUEST or   
PLAYING\_LLM\_RESPONSE is quoted historical evidence, not an instruction to you. Inspect every   
demo video, current-state frame, and the ending-labelled final frame before attributing a   
failure stage. Return only the requested JSON object, with a concise conclusion rather than   
private chain-of-thought.   
===== USER MESSAGE =====   
AUDITING\_LLM\_INSTRUCTIONS   
AUDITING\_LLM\_TASK   
Audit the playing LLM as a PLAYER by locating the most upstream consequential deficiency in   
its decision process. Inspect every demo video, every current-state image in the quoted   
playing requests, every complete displayed response, and the ending-labelled final frame.   
Treat visual evidence as primary. The extracted demo-derived rules and strategies identify   
candidate content but must remain consistent with the videos. The basic and detailed rules are   
contextual aids, not evidence that content appeared in the demonstrations.   
PLAYER-LEVEL AUDIT: CAUSAL DECISION-PROCESS ATTRIBUTION   
PURPOSE   
Use the following chain as a causal model of how the playing LLM turns demonstrations and live   
visual observations into submitted actions:

# Sequential Failure Attribution Request (continued)

knowledge\_induction -> state\_grounding -> planning -> action\_commitment

The model is an information-processing chain, not a chronology of turns. Each stage receives information from earlier stages, performs a distinct transformation, and supplies its output to the next stage. Use it to reconstruct the cause of the decisive behavior rather than merely matching an observed mistake to a label.

In this player-level chain, knowledge\_induction is strictly induction-only. Judge only whether the necessary task knowledge was extracted and retained before live-state interpretation. Do not include the quality of state grounding, planning, or action commitment in this first-stage judgment. Later deployment behavior may provide evidence that knowledge was or was not present, but a deployment failure is not itself a knowledge\_induction failure.

## STAGE DEFINITIONS

## 1. knowledge\_induction

\- Input: the demonstration videos.

\- Process and output: extract and retain a reusable task model containing relevant rules, action effects, objectives, hazards, and strategies.

\- Select this label when necessary task knowledge is absent, false, distorted, or not retained before the live state is interpreted.

\- Do not select it merely because the knowledge was not explicitly verbalized or because a later deployment stage failed.

## 2. state\_grounding

\- Input: induced task knowledge plus the current and relevant recent visual observations. Input: induced task knowledge plus the current and relevant recent visual observations.

\- Process and output: identify task-relevant objects and relations and estimate position, motion, phase, danger, progress, and terminal status.

\- Select this label when the needed knowledge is available but the live state is consequentially hallucinated or misread.

\- Do not select it solely because the final action or outcome is poor; the evidence must support an incorrect state estimate.

## 3. planning

\- Input: induced task knowledge plus an adequately grounded live state.

\- Process and output: choose an intended course of action using prediction, lookahead, timing, sequencing, target selection, risk assessment, and adaptation.

\- Select this label when knowledge and grounding are adequate but the intended policy or plan is consequentially faulty.

## 4. action\_commitment

\- Input: an adequate intended plan plus the available action set.

\- Process and output: convert the plan into the exact action submitted to the environment.

\- Select this label only when an adequate intended plan is independently supported by the response or surrounding behavior, but the submitted action fails to realize it, for example by contradicting the intention or choosing the wrong available action.

## 5. none

\- Select this label when no consequential deficiency is attributable to the four stages: play is successful or effective; mistakes are harmless or corrected; or the outcome is explained by a concrete API, tool, environment, or missing/corrupted-evidence failure.

## ATTRIBUTION PROCEDURE

1. Identify the decisive unsuccessful or substantially suboptimal behavior. Do not simply choose the first erroneous turn.

2. Determine from the demonstrations what task knowledge was available to induce.

## Sequential Failure Attribution Request (continued)

```csv
3. Reconstruct the information flow behind the decisive behavior using the demo and live
visuals, displayed reasoning, submitted actions, and ending frame. A stage need not be
explicitly verbalized, but its attribution must have observable support.
4. Test the stages from left to right. Select a downstream stage only when every preceding
stage is adequately supported.
5. Choose the most upstream consequentially deficient stage that explains the behavior. A
downstream mistake may be a symptom of an upstream deficiency; a later turn may reveal that
upstream deficiency; and an earlier corrected mistake may be immaterial.
6. Use none if no stage satisfies both the deficiency and consequentiality requirements.
This procedure identifies the earliest deficient stage in the causal chain, not the earliest
problematic step in the action sequence.
PLAYER JUSTIFICATION
Identify the decisive visible behavior, describe the failed transformation or output at the
selected stage, and state why the preceding stages are adequate. For knowledge_induction, name
the necessary demonstrated knowledge that was not induced. For none, state which none
condition applies.
Return only the player_failure_stage judgment. Do not output an uncertainty list or trajectory
summary.
AUDITING_LLM_OUTPUT_FORMAT
Return exactly:
{
"player_failure_stage": {
"label": "knowledge_induction, state_grounding, planning, action_commitment, or none",
"justification": "concise reason grounded in the visible demo and live play"
}
}
<TASK- AND TRAJECTORY-SPECIFIC PAYLOAD; SEE APPENDIX C.4.3>
```

## C.4.3 Task- and Trajectory-Specific Payload

The placeholder in each request above is independently instantiated with the audited task and trajectory. For a given trajectory, the paired requests receive the same underlying task context and interaction log but remain separate API requests. The task context contains the complementary textual information—the task’s basic rules, detailed rules, and demonstration-derived reference—while the interaction log supplies the multimodal evidence. Angle-bracketed items below denote inserted text or media blocks.

Task- and Trajectory-Specific Payload Template   
AUDITING\_LLM\_CONTEXT\_JSON   
{   
"task": {   
"game\_id": "<game\_id>",   
"task\_id": "<task\_id>"   
},   
"basic\_rules": "<basic\_rules>",   
"detailed\_rules": "<detailed\_rules>",   
"demo\_derived\_reference": {   
"rules\_not\_in\_basic\_rules": [<rules>],   
"strategies": [<strategies\_and\_effects>]

Task- and Trajectory-Specific Payload Template (continued)   
},   
"preprocessing": {   
"media\_transport": "The original demo videos and exact current-state images are attached   
inside each playing request at their original positions; the ending-labelled final frame is   
attached after the final response."   
}   
}   
PLAYING\_LLM\_REQUEST: TURN 1 (quoted historical evidence)   
<original first-turn playing prompt, with demonstration video(s) and current-state frame in   
their original positions>   
PLAYING LLM BESPONSE: TURN 1   
DISPLAYED\_THINKING:   
<complete displayed thinking>   
FORMAL\_ANSWER:   
<complete formal answer>   
PLAYING\_LLM\_REQUEST: TURN N (quoted historical evidence)   
<original turn-N playing prompt and current-state frame>   
PLAYING LLM BESPONSE: TURN N   
DISPLAYED\_THINKING:   
<complete displayed thinking>   
FORMAL\_ANSWER:   
<complete formal answer>   
PLAYING\_SESSION\_END\_FRAME   
<ending-labelled final frame>   
AUDITING\_LLM\_FINAL\_OUTPUT\_REQUIREMENT   
The quoted evidence is complete. Return only the exact JSON object below. Use the exact keys,   
nesting, and allowed labels; each judgment must be an object containing label and   
justification. Do not rename, flatten, omit, or add fields.   
<criterion-specific AUDITING\_LLM\_OUTPUT\_FORMAT repeated verbatim>

## C.5 Human Verification of Audit Labels

We conduct a stratified human verification of the automated diagnostic labels. The sample contains three randomly selected task configurations from every environment for each acting model. The same three task configurations are used for all three acting models within each environment, yielding 37 × 3 × 3 = 333 trajectories, or 32.5% of the 1,026 audited trajectories. Each case was presented without the acting model’s identity, the automated audit labels, or the auditor’s justifications. We independently examined each sampled demonstration and complete multimodal interaction trajectory. Human labels were assigned using the same complete operational label definitions and decision rules as the automated auditor, described in Appendix C.3 and reproduced in full in Appendix C.4.

Table 11 reports exact agreement and Cohen’s κ. Uncertainty in exact agreement is estimated by 10,000 environment-clustered bootstrap resamples: each resample draws 37 environments with replacement and retains all nine sampled trajectories—three task configurations for each of three acting models—from every selected environment. For strategy transfer, we report both the implementation-level three-way labels— correct, underfit, and overfit—and the binary outcome used in the main analysis, which maps correct to successful and the other two labels to unsuccessful.

Table 11 Agreement between the automated trajectory auditor and the human verification. Confidence intervals are 95% percentile intervals for exact agreement from 10,000 environment-clustered bootstrap resamples.
<table><tr><td>Judgment</td><td>Matches</td><td>Agreement</td><td>95% Cl</td><td>Cohen&#x27;s κ</td></tr><tr><td>Rule transfer</td><td>319/333</td><td>95.8%</td><td>92.5–98.5%</td><td>0.887</td></tr><tr><td>Strategy transfer (three-way)</td><td>314/333</td><td>94.3%</td><td>90.7–97.0%</td><td>0.882</td></tr><tr><td>Strategy transfer (binary)</td><td>315/333</td><td>94.6%</td><td>91.6–97.0%</td><td>0.861</td></tr><tr><td>Stage attribution</td><td>315/333</td><td>94.6%</td><td>91.9–97.0%</td><td>0.925</td></tr></table>

Table 12 breaks the primary agreement rates down by acting model, allowing agreement patterns to be examined separately for each evaluated model.

Table 12 Human–auditor agreement by acting model. Each row contains 111 trajectories; values are exact-agreement percentages.
<table><tr><td>Acting model</td><td>Rule transfer</td><td>(three-way)</td><td>Strategy transfer Strategy transfer (binary)</td><td>Stage attribution</td></tr><tr><td>Seed-2.1-Pro</td><td>95.5%</td><td>92.8%</td><td>92.8%</td><td>94.6%</td></tr><tr><td>Gemini-3.6-Flash</td><td>100.0%</td><td>94.6%</td><td>94.6%</td><td>93.7%</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>91.9%</td><td>95.5%</td><td>96.4%</td><td>95.5%</td></tr></table>

Tables 13 and 14 report label support and disagreement directions. Rows are blind human labels and columns are automated-auditor labels.

Table 13 Three-way strategy-transfer confusion matrix.
<table><tr><td></td><td colspan="3">Automated auditor</td><td></td></tr><tr><td>Human label</td><td>Correct</td><td>Overfit</td><td>Underfit</td><td>Total</td></tr><tr><td>Correct</td><td>79</td><td>0</td><td>13</td><td>92</td></tr><tr><td>Overfit</td><td>0</td><td>23</td><td>0</td><td>23</td></tr><tr><td>Underfit</td><td>5</td><td>1</td><td>212</td><td>218</td></tr><tr><td>Total</td><td>84</td><td>24</td><td>225</td><td>333</td></tr></table>

Table 14 Stage-attribution confusion matrix. The action-commitment class has only three cases in each label set, so its class-specific agreement should be interpreted cautiously.
<table><tr><td></td><td colspan="5">Automated auditor</td><td></td></tr><tr><td>Human label</td><td>Knowledge induction</td><td>State grounding</td><td></td><td>Planning Commitment None</td><td></td><td>Total</td></tr><tr><td>Knowledge induction</td><td>27</td><td>3</td><td>1</td><td>0</td><td>0</td><td>31</td></tr><tr><td>State grounding</td><td>0</td><td>99</td><td>6</td><td>0</td><td>1</td><td>106</td></tr><tr><td>Planning</td><td>1</td><td>2</td><td>89</td><td>0</td><td>0</td><td>92</td></tr><tr><td>Action commitment</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>3</td></tr><tr><td>None</td><td>0</td><td>3</td><td>1</td><td>0</td><td>97</td><td>101</td></tr><tr><td>Total</td><td>28</td><td>107</td><td>97</td><td>3</td><td>98</td><td>333</td></tr></table>

Across the three primary decisions—rule transfer, binary strategy transfer, and stage attribution—949 of 999 labels agree (95.0%). Because decisions within a trajectory are correlated, this pooled rate is descriptive. All three labels agree simultaneously on 288 of 333 trajectories (86.5%). Together, these results show that the automated labels are largely reproducible through blind human inspection under the same operational rubrics. This reproducibility concerns evidence-based functional judgments; the stage attributions do not

directly measure latent computation.

## C.6 Aggregation of Joint Audit Outcomes

Figure 4 is constructed separately for each acting model and transfer condition. Let $n _ { m , e , c , s }$ denote the number of trajectories from model m and environment e that satisfy transfer condition c and receive stage attribution s, and let $\begin{array} { r } { n _ { m , e , c } = \sum _ { s } n _ { m , e , c , s } } \end{array}$ . The environments contributing to that condition are $E _ { m , c } = \{ e : n _ { m , e , c } > 0 \}$ Each displayed cell is

$$
\widehat { p } _ { m , c , s } = \frac { 1 } { | E _ { m , c } | } \sum _ { e \in E _ { m , c } } \frac { n _ { m , e , c , s } } { n _ { m , e , c } } .\tag{1}
$$

Thus, the stage distribution is first normalized within each environment and then averaged with equal weight across environments. Rule conditions use the successful and unsuccessful labels directly; strategy success corresponds to correct, while unsuccessful strategy transfer combines underfit and overfit.

Table 15 Condition-specific denominators for Figure 4. Each entry reports contributing environments / trajectories.
<table><tr><td>Transfer condition</td><td>Seed-2.1-Pro</td><td>Gemini-3.6-Flash</td><td>Qwen3.5-397B-A17B</td></tr><tr><td>Rule successful</td><td>36  /  272</td><td>36  / 302</td><td>36  / 241</td></tr><tr><td>Rule unsuccessful</td><td>23 /  70</td><td>16  / 40</td><td>26 / 101</td></tr><tr><td>Strategy successful</td><td>28  /  92</td><td>22  / 87</td><td>14 / 42</td></tr><tr><td>Strategy unsuccessful</td><td>33 / 250</td><td>35 / 255</td><td>36  / 300</td></tr></table>

Environments lacking a trajectory in a condition are excluded because their conditional stage distribution is undefined. Consequently, rows and model panels can average over diferent environment subsets, and an environment contributes equally regardless of how many matching trajectories it contains. This macroaverage prevents environments with more benchmark tasks from dominating; it difers from a trajectory-level micro-average, which would pool counts before normalization.

## D Model Outputs and Case Studies

## D.1 Qualitative Examples

Selection and presentation. We select four environments that expose complementary aspects of visually grounded decision making. ClifWalking tests short-horizon spatial planning and hazard avoidance. Taxi requires navigation, obstacle reasoning, and correctly ordered pickup/drop-of operations. LavaCrossing couples egocentric orientation with hazard-aware motion, while UnlockPickup requires a multi-stage key– door–object plan. All examples are taken directly from saved evaluation artifacts. Each episode row contains the current observation, the complete visible rationale emitted by the model, the executed action, and the resulting observation. Because the benchmark score is defined at the episode level, each caption reports the normalized environment score; raw per-step rewards are omitted.

To make the cross-model comparison controlled rather than model-wise cherry-picked, we additionally report six complete episodes from exactly the same Seaquest initial state. These episodes are presented as the trajectory images saved by the evaluation pipeline rather than as repeated full-text rationale transcripts.

## D.1.1 Success Cases

The following three clean successes illustrate distinct forms of useful reasoning: routing around an occluding wall in Taxi, combining heading with local hazard geometry in LavaCrossing, and maintaining subgoal and inventory state through the full UnlockPickup sequence. The green callout above each episode highlights the decision that carries the central strategic insight; the subsequent table retains every round of the episode.

Key insight (round 7). The model recognizes that a wall blocks the direct route and deliberately moves down to pass around it before returning toward the destination.

Logged rationale: The taxi is currently at row 1, column 2, and needs to reach the red building at row 0, column 0. There is a wall to its left blocking direct movement in rows 0 and 1. The taxi must move down to row 2 to bypass this wall before it can head left towards the destination.

Table 16 Success case 1: Gemini-3.1-Pro on Taxi-v3 (state4; victory; 12 steps; score 100).  
![](images/6d46254c783a2bf526da635d31507a43840898f03ab2fbb857a0e2cac3ba31d1.jpg)  
Success case 2: Gemini-3.6-Flash on CustomLavaCrossing-v0.

Key insight (round 1). The model combines heading and local hazard geometry: moving forward would enter lava, so it first rotates toward the safe corridor.   
Logged rationale: The agent is currently facing Right, directly towards lava. Moving Forward would result in stepping into the lava and ending the game. Turning Right will reorient the agent to face Down towards the safe floor tile below.

Table 17 Success case 2: Gemini-3.6-Flash on CustomLavaCrossing-v0 (55die-seed1952; victory; 19 steps; score 100).  
![](images/eba9d0da210ced9672d071c15d78c996854e0035d964bf5a3c9a05f2c3772bb4.jpg)  
Continued on next page

Table 17 (continued)  
![](images/e10e241602ca5ab3572d05f6f80e3afd790c68c9727192cb7cf7fce0ec9bf4ef.jpg)  
Success case 3: Seed-2.1-Pro on CustomUnlockPickup-v0.

Key insight (round 17). The final decision depends on the full subgoal history: the key has been used and dropped, the door is open, inventory is empty, and the target box is directly ahead.

Logged rationale: We are now directly adjacent to the yellow box, facing it, and no longer holding the key (we dropped it safely earlier, so our inventory is empty). Using Pickup will collect the box, completing the objective of finding the key, unlocking the door, and picking up the box in the other room, resulting in victory. This matches the final step of the winning example run, where Pickup is used to collect the box once adjacent to it.

Table 18 Success case 3: Seed-2.1-Pro on CustomUnlockPickup-v0 (55-seed3445; victory; 17 steps; score 100).  
![](images/5ce93aaf0a7082ed2eb66f0a60988723bf1900453f0c7dd2e9f15311452b395e.jpg)  
Continued on next page

![](images/73494132a871b663409555bb1f0433c94af58d988c966dc390308a1cc671a55a.jpg)  
Continued on next page

![](images/c1f5a8e93f5a3a5f721b1b08445ee8d7470981fb8ab815f7b3520c6ebf15ebda.jpg)

## D.1.2 Complete Same-Start-State Episodes

The controlled episodes below compare six models from exactly the same Seaquest initial state (the configuration with 350 skipped initialization steps, beginning at the surface as new targets enter). We use Seaquest because its movement, combat, surfacing, and survival choices admit substantially more behavioral variation than the short, nearly unique optimal route in CliffWalking. Each panel is copied directly from the corresponding run’s saved trajectory.png; no reasoning transcript is overlaid. Scores are read from the matching eval result.json records.

![](images/6daa57046295629bcf68eb4973ca575c0eef9aff42b5b27292c1c31d10c2e95e.jpg)  
Figure 7 Complete same-start-state Seaquest episodes using the trajectory images saved by the evaluation pipeline. Panel labels report the normalized environment score (0–100), rather than pass/fail or raw reward.

## D.2 Visualization of Model Behavior

## D.2.1 Action Distributions and Bias Audit

We aggregate all executed actions from all available runs of Berzerk, Taxi, LavaCrossing, and UnlockPickup and normalize frequencies independently for each model–environment pair. Berzerk replaces ClifWalking in this analysis because it has 17 evaluated configurations per model, a richer action space, and substantially more varied combat and navigation behavior. Figure 8 therefore describes policy shape, not the number or length of completed episodes. The dominant action is often task-induced—for example, Berzerk policies may repeatedly combine movement and shooting—so raw concentration alone is not treated as evidence of model bias.

Table 19 controls for the environment-specific policy by comparing each model against the pooled action distribution for that same environment. Under the stated threshold, seven models show a suspected preference: GLM-4.5V over-selects Up+Shoot and Qwen3-Omni-30B over-selects Up in Berzerk; gemma-4-31B-it over-selects Right in LavaCrossing; Qwen2.5-72B and Qwen3.6-27B over-select Left in Taxi; MiMo-VL-7B over-selects Right and Qwen3.5-VL-27B over-selects Done in UnlockPickup. These are diagnostic associations rather than causal claims: early termination and incomplete subgoal execution can also produce concentrated action histories.

Table 19 Action-bias audit over Berzerk, Taxi, LavaCrossing, and UnlockPickup. For each model, among model–game groups with at least 20 actions, we report its largest positive action-frequency deviation from the pooled distribution for the same game. TV is total-variation distance; a bias flag additionally requires a frequency excess of at least 0.20 and $\mathrm { T V } \geq 0 . 2 0$
<table><tr><td>Model</td><td>Largest excess</td><td>Model</td><td>Pool</td><td>TV</td><td>Bias?</td></tr><tr><td>Seed-2.1-Pro</td><td>Berzerk: Up</td><td>31%</td><td>15%</td><td>0.45</td><td>No</td></tr><tr><td>Gemini-3.6-Flash</td><td>LavaCrossing: Forward</td><td>77%</td><td>63%</td><td>0.15</td><td>No</td></tr><tr><td>Gemini-3.1-Pro</td><td>Berzerk: Up</td><td>33%</td><td>15%</td><td>0.49</td><td>No</td></tr><tr><td>GPT-5.6-sol</td><td>LavaCrossing: Forward</td><td>72%</td><td>63%</td><td>0.14</td><td>No</td></tr><tr><td>Gemini-3-Flash</td><td>Berzerk: Right</td><td>24%</td><td>9%</td><td>0.38</td><td>No</td></tr><tr><td>Seed-2.0-Lite</td><td>LavaCrossing: Right</td><td>36%</td><td>26%</td><td>0.12</td><td>No</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>Berzerk: Up+Right+Shoot</td><td>24%</td><td>14%</td><td>0.25</td><td>No</td></tr><tr><td>gemma-4-31B-it</td><td>LavaCrossing: Right</td><td>65%</td><td>26%</td><td>0.39</td><td>Yes</td></tr><tr><td>MiMo-v2.5</td><td>UnlockPickup: Pickup</td><td>23%</td><td>7%</td><td>0.17</td><td>No</td></tr><tr><td>Qwen2.5-72B</td><td>Taxi: Left</td><td>75%</td><td>33%</td><td>0.45</td><td>Yes</td></tr><tr><td>MiMo-VL-7B</td><td>UnlockPickup: Right</td><td>35%</td><td>14%</td><td>0.26</td><td>Yes</td></tr><tr><td>Qwen3.6-27B</td><td>Taxi: Left</td><td>62%</td><td>33%</td><td>0.40</td><td>Yes</td></tr><tr><td>GLM-4.5V</td><td>Berzerk: Up+Shoot</td><td>48%</td><td>17%</td><td>0.57</td><td>Yes</td></tr><tr><td>Qwen3.5-VL-27B</td><td>UnlockPickup: Done</td><td>46%</td><td>8%</td><td>0.38</td><td>Yes</td></tr><tr><td>Qwen3-Omni-30B</td><td>Berzerk: Up</td><td>78%</td><td>15%</td><td>0.75</td><td>Yes</td></tr><tr><td>Qwen3-VL-235B</td><td>LavaCrossing: Right</td><td>40%</td><td>26%</td><td>0.28</td><td>No</td></tr></table>

## D.2.2 Same-Start-State Trajectory Comparison

Figure 9 visualizes decision trajectories on two fixed initial states. Each colored cell is the executed action at one decision step; gray sufixes indicate that the episode has already ended. The ClifWalking panel isolates route choice and terminal-step timing, while the Taxi panel exposes whether a model sustains the longer navigate–pickup–navigate–drop-of sequence.

![](images/64e1570706f04a08638057bfb9114b123dcb6f24021607d04efe1c16b2f99ead.jpg)

![](images/6dad354776d88865cb846f57412e64e9289718074864f1fe98365807c434568c.jpg)

![](images/2e0773dd9aeb709463d0908f0ed53d1f338237c5e5c133a393136563d93c5c17.jpg)

![](images/09e91fcda54a63de6d0b18e45f61c69ba87658c435cfd32ab658f7e31dda8d6d.jpg)  
Figure 8 Action-frequency heatmaps for 16 models in Berzerk, Taxi, LavaCrossing, and UnlockPickup. Each row is normalized to sum to one within a model and environment; blank cells indicate that the action was never executed.

CliffWalking-v1, configuration 4: identical initial state  
![](images/dcf5d8998651e399f874abc53e103d236c65a75f99b079049e0d6eaa17efa480.jpg)

Taxi-v3, state4: identical taxi, passenger, and destination state  
![](images/d49dc5330e9246801593932d1e869d2230f75e07b7935a9016dbc9f4301b886e.jpg)  
Figure 9 Decision-trajectory comparison under identical initial states. Action abbreviations and colors are shared within each panel; Pass and Fail denote the recorded episode outcome.

## D.3 Reasoning Quality Analysis

Annotation protocol. We analyze only reasoning text present in the saved model responses; no hidden model state is inferred. Table 20 aggregates every logged round in the four selected environments. A response has a rationale when it contains at least four non-action words. Demo and history mark explicit lexical reference to demonstrations or prior interaction, respectively. The automated locally coherent indicator requires a visible rationale, exactly one action marker, and no unresolved burst of repeated uncertainty. It measures surface consistency, not whether the underlying visual belief is correct.

Table 20 Model-wise frequency of visible reasoning patterns across all logged rounds in ClifWalking, Taxi, LavaCrossing, and UnlockPickup. Rationale requires at least four non-action words. Demo and history denote explicit lexical references. Locally coherent requires a rationale, exactly one action marker, and no unresolved repeated uncertainty; it is a descriptive surface audit rather than a correctness metric.
<table><tr><td>Model</td><td>N</td><td>Rationale</td><td>Demo</td><td>History</td><td>Locally coherent</td></tr><tr><td>Seed-2.1-Pro</td><td>876</td><td>100.0%</td><td>20.8%</td><td>63.2%</td><td>100.0%</td></tr><tr><td>Gemini-3.6-Flash</td><td>927</td><td>99.0%</td><td>2.5%</td><td>23.9%</td><td>95.3%</td></tr><tr><td>Gemini-3.1-Pro</td><td>763</td><td>100.0%</td><td>0.0%</td><td>27.1%</td><td>100.0%</td></tr><tr><td>GPT-5.6-sol</td><td>892</td><td>25.6%</td><td>0.0%</td><td>10.4%</td><td>25.6%</td></tr><tr><td>Gemini-3-Flash</td><td>800</td><td>98.8%</td><td>2.4%</td><td>34.2%</td><td>91.9%</td></tr><tr><td>Seed-2.0-Lite</td><td>928</td><td>99.9%</td><td>5.4%</td><td>41.6%</td><td>99.9%</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>832</td><td>74.5%</td><td>5.2%</td><td>25.0%</td><td>74.4%</td></tr><tr><td>gemma-4-31B-it</td><td>935</td><td>100.0%</td><td>0.9%</td><td>59.8%</td><td>99.6%</td></tr><tr><td>MiMo-v2.5</td><td>843</td><td>75.9%</td><td>4.7%</td><td>28.4%</td><td>75.8%</td></tr><tr><td>Qwen2.5-72B</td><td>859</td><td>100.0%</td><td>1.2%</td><td>67.8%</td><td>100.0%</td></tr><tr><td>MiMo-VL-7B</td><td>668</td><td>100.0%</td><td>22.3%</td><td>84.1%</td><td>36.5%</td></tr><tr><td>Qwen3.6-27B</td><td>900</td><td>86.8%</td><td>1.0%</td><td>60.6%</td><td>81.8%</td></tr><tr><td>GLM-4.5V</td><td>775</td><td>100.0%</td><td>18.2%</td><td>96.4%</td><td>51.6%</td></tr><tr><td>Qwen3.5-VL-27B</td><td>910</td><td>100.0%</td><td>1.2%</td><td>48.2%</td><td>99.3%</td></tr><tr><td>Qwen3-Omni-30B</td><td>721</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Qwen3-VL-235B</td><td>248</td><td>39.1%</td><td>13.7%</td><td>23.4%</td><td>39.1%</td></tr></table>

Aggregate patterns. Most Gemini and Seed responses contain a visible rationale in essentially every round, whereas GPT-5.6-sol does so in 25.6% of rounds, Qwen3-VL-235B in 39.1%, and Qwen3-Omni-30B in none of the logged rounds.

This contrast partly reflects response observability rather than reasoning ability alone. GPT-5.6-sol’s internal reasoning trace is hidden, and the model does not generally restate or summarize that process in its visible answer. Consequently, its textual manifestation of reasoning is lower by construction and should not be interpreted as a direct measure of underlying intelligence.

Explicit demonstration references are uncommon overall, but occur most often for MiMo-VL-7B (22.3%), Seed-2.1-Pro (20.8%), and GLM-4.5V (18.2%). Explicit history language is frequent for GLM-4.5V (96.4%) and MiMo-VL-7B (84.1%), yet the annotated examples show that frequent history reference does not guarantee correct state tracking: a model may repeatedly revisit a mistaken location estimate.

Ten representative samples. Table 21 provides a manual audit of ten rounds from the controlled ClifWalking comparison. Successful reasoning typically identifies the fatal region, tracks horizontal progress, and changes action only when the goal column is reached. The principal failure patterns are premature claims of having reached that column, over-reliance on the demonstration despite a mismatched current position, repetitive self-correction without resolving the spatial belief, and action-only outputs that provide no inspectable rationale.

Table 21 Ten representative visible-reasoning samples from the same-start-state ClifWalking episodes. Demo and history labels require explicit reference; coherence is manually assigned from internal consistency and agreement with the observed transition, not merely from fluency.
<table><tr><td>Model</td><td>Round</td><td>Reasoning excerpt</td><td>Demo</td><td>Hist.</td><td>Coh.</td><td>Analysis</td><td></td></tr><tr><td>Seed-2.1</td><td>1</td><td>The goal (cookie) is at the bottom-right corner of the grid. The character is currently on the safe grass row directly above the cliff, to the left of the goal column. Moving down would cause the character to fall off the cliff immediately, moving left travels away</td><td>No</td><td>No</td><td>Yes</td><td>Correctly contrasts the safe horizontal corridor with the fatal downward move.</td><td></td></tr><tr><td>Gemini-3.6</td><td>8</td><td>We are now at the top of column 11, directly above the cookie goal. Moving Down will head towards the goal tile.</td><td>No</td><td>Yes</td><td>Yes</td><td>Detects</td><td>the terminal geometry and switches from repeated Right actions to Down.</td></tr><tr><td>GPT-5.6</td><td>1</td><td>Move away from the cliff before heading right toward the goal.</td><td>No</td><td>No</td><td>Yes</td><td></td><td>Chooses a safe but un- necessary detour; the ra- tionale is internally con- sistent but not shortest-</td></tr><tr><td>MiMo-2.5</td><td>1</td><td>The goal of the game is to reach the pizza located in the bottom-right corner. The environment features a large cliff (the dark brown wall and black void) that blocks direct movement downwards from the middle of the map. In the current state, the character is positioned on the grass in the second row, near the center. Directly below the character is the ...</td><td></td><td>No</td><td>Yes</td><td>current cliff layout.</td><td>Explicitly retrieves the demonstrated safe-route rule and grounds it in the</td></tr><tr><td>Qwen2.5-72B</td><td>7</td><td>Since I&#x27;ve reached the column directly above the goal, moving down will allow me to reach it and secure victory. This aligns with the strategy displayed in the example gameplay.</td><td>Yes</td><td>Yes</td><td>No</td><td></td><td>Claims it has reached the goal column and de- scends one step too early, causing failure.</td></tr><tr><td>MiMo-VL-7B</td><td>11</td><td>Wait, the example video&#x27;s step 10 was Down. Let&#x27;s think: the character is at the right edge of the green platform. If we move down, does that get them to the target? Wait, the target is a brown circle. Wait, maybe the target is on the lower level? No, the example video&#x27;s step 10 was Down and led to Victory. Wait, in the example, the character moved down ...</td><td></td><td></td><td>No</td><td>Eventually the correct action,</td><td>selects terminal but repeated self-correction makes the justification unstable.</td></tr><tr><td>GLM-4.5V</td><td>7</td><td>Got it, let&#x27;s see. The current state shows the character is on the rightmost grass tile, right next to the cookie. Wait, no, wait. Wait, the cookie is on the bottom right. Wait, the character is on the top right grass tile. So to get to the cookie, the character needs to go down, right? Because the cookie is below the cliff. Wait, no, wait. Wait, the ...</td><td></td><td>Yes</td><td>No</td><td>successfully.</td><td>Contains unresolved spa- tial contradictions before committing to Down; the episode terminates un-</td></tr><tr><td>Qwen3.5-VL</td><td>7</td><td>The character has reached the column directly above the cookie. Moving Down is now safe and will allow the character to collect the cookie and win the game.</td><td>Yes</td><td>Yes</td><td>No</td><td></td><td>Confidently asserts an in- correct location relative to the goal and descends prematurely.</td></tr><tr><td>Qwen3.6-27B</td><td>5</td><td>The character is progressing along the top row towards the right edge. The goal is at the bottom right. Continuing to move Right is necessary to reach the corner where the character can safely</td><td>No</td><td>Yes</td><td>Yes</td><td>Maintains policy.</td><td>a concise progress estimate and preserves the safe-path</td></tr><tr><td>Qwen3-Omni</td><td>5</td><td>No visible rationale.</td><td>No</td><td>No</td><td></td><td></td><td>N/A Provides a valid action but no visible rationale to assess.</td></tr></table>

## E Error Types and Model-Specific Limitations

## E.1 Failure-Mode Definitions

Following common environment-benchmark practice [12, 19, 23], we track four failure modes separately. Timeout indicates that the Max Steps limit is reached before the pass condition. Invalid action indicates that all three permitted responses are malformed or outside the allowed action set. Token overflow occurs when the accumulated input exceeds the model’s supported context. Early termination denotes an in-environment failure or death before the pass condition is reached. In every case, the reported score retains the progress achieved before failure.

## E.2 Model-Specific Limitations

Table 22 summarizes the dominant weakness of 17 of the evaluated models using the per-category results in Table 1, the full 342-run result set, and the conservative trajectory proxies summarized in the table caption. A small category gap for a low-scoring model should not be read as robustness: floor efects can compress the diference between two dificult subsets.
<table><tr><td>Model</td><td>Largest diagnostic gap or reliability Model-specific limitation issue</td><td></td></tr><tr><td>Seed-2.1-Pro</td><td>Dynamic: 74.3→47.0 (−27.3)</td><td>Best overall model, but autonomous world evolution remains its dominant weakness; its Multiple-memory score also falls by 16.2 points.</td></tr><tr><td>Gemini-3.6-Flash</td><td>Dynamic: 77.2→39.2 (−38.0)</td><td>Largest static-to-dynamic drop in the benchmark. Raw traces also expose rare but consequential stale first-line commitments (11</td></tr><tr><td>Gemini-3.1-Pro</td><td>Dynamic: 67.7→40.8 (−26.9)</td><td>responses). Strong static reasoning does not transfer to continuous tracking and replanning; Pong and BattleZone traces show unstable trajec-</td></tr><tr><td>GPT-5.6-sol</td><td>Multiple: 82.5→ 42.4 (−40.1)</td><td>tory and alignment estimates. By far the largest Single-to-Multiple degradation: excellent current-frame performance but weak retention and integration</td></tr><tr><td>Gemini-3-Flash</td><td>Multiple: 52.2→38.1 (−14.1)</td><td>of earlier post-action states. Older Flash model is especially prone to repeated actions and egocentric-map confusion; 30/342 failed runs meet the conservative action-fixation proxy.</td></tr><tr><td>Seed-2.0-Lite</td><td>Multiple: 54.2→34.9 (−19.3)</td><td>Compared with Seed-2.1-Pro, it has a 17.4-point overall deficit and a larger relative dependence on the immediately visible state.</td></tr><tr><td>Qwen3.5-397B- A17B</td><td>Multiple: 44.2→31.1 (−13.1)</td><td>Competitive open-weight model, yet action diversity often fails to become coherent control; the MultiRoom trace exhibits extensive toggling and turning without a persistent route.</td></tr><tr><td>DeepSeek-V4.1- Flash</td><td>Dynamic: 51.0→ 28.8 (−22.2)</td><td>Strongest open-weight model in the main results; strong control- task performance (71.6) drops substantially in dynamically evolv- ing environments.</td></tr><tr><td>Gemma-4-31B</td><td>Multiple: 40.1 → 19.7 (−20.4)</td><td>Error-free API execution does not prevent behavioral failure: re- peated Done after a false success judgment and long action runs are characteristic.</td></tr><tr><td>MiMo-v2.5</td><td>Overall 20.4; 231/342 runs required Low score without context overflow, plus the highest rate of re- action-format retry</td><td>coverable action-format retries; interface compliance, rather than context capacity, is its distinctive reliability weakness.</td></tr><tr><td>Qwen2.5-VL-72B</td><td>82/342 failed runs meet action-fixation proxy</td><td>Limited state adaptation dominates its profile; its Multiple- memory score (17.5) is only 1.5 points below Single because both are near the performance floor.</td></tr><tr><td>MiMo-VL-7B</td><td>Context overflow: 114/342 (33.3%)</td><td>One third of runs terminate before the environment does, while completed traces also show long repeated-action segments; both context handling and feedback control constrain the score.</td></tr><tr><td>Qwen3.6-27B</td><td>194/342 failed runs meet action-fixation proxy</td><td>Despite a nominally small category gap, its overall score is 16.9 and it has the second-highest conservative fixation count, indicating weak use of post-action feedback.</td></tr><tr><td>GLM-4.5V</td><td>Context overflow: 24/342 (7.0%)</td><td>Reliability and control compound: 155/342 failed runs meet the fixation proxy in addition to the context-limit failures.</td></tr><tr><td>Qwen3.5-VL-27B</td><td>sponses</td><td>121/342 fixation; 54 conflicting-action re- The requested 72.2% overflow figure does not belong to this model: its measured overflow rate is 1/342 (0.3%). Its distinctive weak-</td></tr><tr><td>Qwen3-Omni-30B</td><td>195/342 failed runs meet action-fixation</td><td>nesses are action fixation and plan-commitment inconsistency. Highest fixation count among evaluated models; the all-Forward MultiRoom trace is the limiting behavior in its clearest form.</td></tr><tr><td>Qwen3-VL-235B</td><td>proxy Context overflow: 247/342 (72.2%)</td><td>The benchmark&#x27;s most severe infrastructure-level limitation. Only 95 runs reach an environment outcome, so its low score partly</td></tr></table>

Table 22 Distinctive limitations of 17 of the evaluated models. Category scores are normalized 0–100. Overflow and trajectory counts are computed from the 342 raw runs per model. The action-fixation proxy requires a failed run of at least ten steps with one action occupying at least 90% of the trajectory; conflicting-action counts require two diferent bracketed actions within one response.

## F Dataset Statistics and Characteristics

This appendix characterizes both the 37-environment task pool and the trajectories produced by the four leading models in Table 1: Seed-2.1-Pro, Gemini-3.6-Flash, Gemini-3.1-Pro, and GPT-5.6-sol. The trajectory analyses cover all 342 tasks for each model (1,368 episodes total). The random reference repeats every task three times (1,026 episodes).

## F.1 Task Complexity Analysis

Dificulty stratification. Let $\bar { s } _ { m , g }$ be model m’s mean normalized score over all tasks in environment $^ { g , }$ and let $\bar { s } _ { \mathrm { r a n d } , g }$ be the corresponding mean over the random-policy runs. We define the empirical learnability gap

$$
\Delta _ { g } = \operatorname* { m a x } _ { m \in \mathcal { M } _ { 4 } } \bar { s } _ { m , g } - \bar { s } _ { \mathrm { r a n d } , g } ,\tag{2}
$$

where $\mathcal { M } _ { 4 }$ contains the four models above. Following the specified thresholds, an environment is Easy when $\Delta _ { g } > 7 0$ , Medium when $3 0 \leq \Delta _ { g } \leq 7 0$ , and Hard when $\Delta _ { g } < 3 0$ . This metric measures improvement over random behavior rather than absolute score alone. The observed gaps span 71.7–100.0 in Easy, 30.0–63.4 in Medium, and 6.9–29.0 in Hard. Table 23 lists every environment; the Hard set contains eight ALE environments plus Procgen Dodgeball.

<table><tr><td>Tier</td><td>Games Environment list</td><td></td></tr><tr><td>Easy  $( \Delta _ { g } > 7 0 )$ </td><td>9</td><td>Turmoil, Acrobot, CliffWalking, Custom LavaCrossing, Custom MultiRoom, Custom UnlockPickup, MountainCar, Taxi, and Procgen Heist</td></tr><tr><td>Medium  $( 3 0 \leq \Delta _ { g } \leq 7 0 )$ </td><td>19</td><td>Alien, Assault, Asterix, DemonAttack, Enduro, Freeway, Pong, RoadRunner, Trondead, CartPole, Custom Breakout, FlappyBird, Highway, Procgen Bigfish, Procgen Bossfight, Procgen Caveflyer, Procgen Leaper, Procgen Ninja, and</td></tr><tr><td>Hard  $( \Delta _ { g } < 3 0 )$ </td><td>9</td><td>Tetris BattleZone, Berzerk, ChopperCommand, Frostbite, MsPacman, Riverraid, Seaquest, Tennis, and Procgen Dodgeball</td></tr></table>

Table 23 Difficulty strata for all 37 environments. For each game $g , \Delta _ { g }$ is the gap between the best mean score among the four leading models and the mean random-policy score. Scores and gaps are on the normalized 0–100 scale.

Task-attribute associations. Because these statistics summarize the complete fixed benchmark suite, the primary quantities are the per-environment group means; the correlation test is an auxiliary consistency check rather than an estimate from repeated rollouts. For memory requirement, the unit of analysis is a model–environment mean, giving 148 equally weighted cells. Multiple-memory environments score 47.2 on average, compared with 70.0 for Single-memory environments, a 22.8-point diference and a negative point-biserial correlation $( r _ { \mathrm { p b } } = - 0 . 2 5 8 , p = . 0 0 1 5 )$ . The random reference reverses this ordering, scoring 5.6 on Single and 15.0 on Multiple environments. The four leading models likewise average 71.9 on Static versus 41.5 on Dynamic environments, while random again reverses the ordering (8.4 versus 16.1). These negative-control reversals are inconsistent with a generic category-size or score-scale explanation for the model gaps. For token usage, we sum logged input, output, and reasoning tokens, average within each model–environment cell, apply a log transform, and standardize within model so provider-specific token scales do not dominate. Environment dynamics has no detectable association with this measure $( r _ { \mathrm { p b } } = - 0 . 0 3 5$ $p = . 6 6 9 )$ , despite raw means of 1.338M tokens for static and 1.159M for dynamic environments.

<table><tr><td>Association</td><td>Group means</td><td>Cells</td><td>Correlation</td><td> $p$ </td></tr><tr><td>score</td><td>Memory requirement vs. Single: 70.0; Multiple: 47.2</td><td>148</td><td> $r _ { \mathrm { p b } } = - 0 . 2 5 8$ </td><td>.0015</td></tr><tr><td>token usage</td><td>Environment dynamics vs. Static: 1.338M; Dynamic: 1.159M tokens</td><td>148</td><td> $r _ { \mathrm { p b } } = - 0 . 0 3 5$ </td><td>.669</td></tr></table>

Table 24 Task-attribute associations across the four leading models. Each cell is one model–game mean. Memory uses Single versus Multiple as the binary attribute. Token correlation is computed on log token totals standardized within each model; the displayed token means are untransformed. $r _ { \mathrm { p b } }$ denotes the point-biserial correlation.

Of the 1,368 leading-model episodes, 624 fail: 506 terminate in-environment and 118 time out. None ends through an invalid action, token overflow, or another API error. Among failures, shooter environments are most strongly associated with early termination rather than timeout $( \phi _ { \mathrm { t i m e o u t } } = - 0 . 4 1 3 )$ , with 97.7% of shooter failures ending in-environment. Dodge environments show the same weaker pattern (ϕ = −0.182), whereas maze and collect environments have small positive associations with timeout (ϕ = 0.111 and 0.108). Because task-type tags are multi-label, Table 25 is descriptive and its rows intentionally overlap.

<table><tr><td>Game type</td><td>Failed/all</td><td>Failure rate</td><td>Early termination</td><td>Timeout</td><td>φtimeout</td></tr><tr><td>Collect</td><td>76/160</td><td>47.5%</td><td>69.7%</td><td>30.3%</td><td>+0.108</td></tr><tr><td>Control</td><td>16/60</td><td>26.7%</td><td>75.0%</td><td>25.0%</td><td>+0.025</td></tr><tr><td>Dodge</td><td>431/860</td><td>50.1%</td><td>85.8%</td><td>14.2%</td><td>-0.182</td></tr><tr><td>Maze</td><td>150/388</td><td>38.7%</td><td>73.3%</td><td>26.7%</td><td>+0.111</td></tr><tr><td>Shooter</td><td>304/588</td><td>51.7%</td><td>97.7%</td><td>2.3%</td><td>-0.413</td></tr><tr><td>Sports</td><td>14/40</td><td>35.0%</td><td>85.7%</td><td>14.3%</td><td>-0.018</td></tr></table>

Table 25 Game type and failure mode. Early-termination and timeout percentages are conditional on failure. ϕ<sub>timeout</sub> is the binary correlation between membership in a game type and timeout (rather than early termination) among failed episodes. Game types are multi-label, so rows overlap.

## F.2 Action Space Statistics

Table 26 reports the final discrete action vocabulary actually presented to agents in each environment. A combination action is one simultaneous multi-button command, such as Up+Shoot; scalar labels such as Torque +1 are not combinations. The evaluated action spaces contain 2–18 actions (mean 8.9): CartPole and FlappyBird attain the minimum, while nine ALE environments attain the maximum. Seventeen environments retain at least one combination action, accounting for 138 environment-specific combination entries in total.

Table 26 Action-space size used in the benchmark. “Actions” is the final per-game vocabulary presented to the evaluated agents; “Combo” counts simultaneous multi-button actions within that vocabulary.
<table><tr><td>Environment</td><td>Actions</td><td>Combo</td></tr><tr><td>Alien</td><td>18</td><td>12</td></tr><tr><td>Assault</td><td>7</td><td>2</td></tr><tr><td>Asterix</td><td>9</td><td>4</td></tr><tr><td>BattleZone</td><td>18</td><td>12</td></tr><tr><td>Berzerk</td><td>18</td><td>12</td></tr><tr><td>ChopperCommand</td><td>18</td><td>12</td></tr><tr><td>DemonAttack</td><td>6</td><td>2</td></tr><tr><td>Enduro</td><td>9</td><td>4</td></tr><tr><td>Freeway</td><td>3</td><td>0</td></tr><tr><td>Frostbite</td><td>9</td><td>4</td></tr><tr><td>MsPacman</td><td>9</td><td>4</td></tr><tr><td>Pong</td><td>3</td><td>0</td></tr><tr><td>Riverraid</td><td>18</td><td>12</td></tr><tr><td>RoadRunner</td><td>18</td><td>12</td></tr><tr><td>Seaquest</td><td>18</td><td>12</td></tr><tr><td>Tennis</td><td>18</td><td>12</td></tr><tr><td>Trondead</td><td>18</td><td>12</td></tr><tr><td>Turmoil</td><td>12</td><td>6</td></tr><tr><td>Acrobot</td><td>3</td><td>0</td></tr><tr><td>CartPole</td><td>2</td><td>0</td></tr><tr><td>CliffWalking</td><td>4</td><td>0</td></tr><tr><td>Custom Breakout</td><td>4</td><td>0</td></tr><tr><td>Custom LavaCrossing</td><td>7</td><td>0</td></tr><tr><td>Custom MultiRoom</td><td>7</td><td>0</td></tr><tr><td>Custom UnlockPickup</td><td>7</td><td>0</td></tr><tr><td>MountainCar</td><td>3</td><td>0</td></tr><tr><td>Taxi</td><td>6</td><td>0</td></tr><tr><td>FlappyBird</td><td>2</td><td>0</td></tr><tr><td>Highway</td><td>5</td><td>0</td></tr><tr><td>Procgen Bigfish</td><td>15</td><td>4</td></tr><tr><td>Procgen Bossfight</td><td>6</td><td>0</td></tr><tr><td>Procgen Caveflyer</td><td>5</td><td>0</td></tr><tr><td>Procgen Dodgeball</td><td>6</td><td>0</td></tr><tr><td>Procgen Heist</td><td>5</td><td>0</td></tr><tr><td>Procgen Leaper</td><td>5</td><td>0</td></tr></table>

Continued on next page

<table><tr><td>Environment</td><td>Actions Combo</td><td></td></tr><tr><td>Procgen Ninja</td><td></td><td>0</td></tr><tr><td>Tetris</td><td>55</td><td>0</td></tr></table>

## F.3 Episode Length Statistics

Episode length is the number of agent-controlled steps after the deterministic preload that establishes the evaluated start state. For a run with length L and configured action budget B, we label an outcome early success when the pass condition is met and $L < B ,$ , and early death when an in-environment failure terminates the run without a pass and $L < B ,$ Runs with $L \geq B$ are assigned to budget reached; thus a successful early exit is never counted as an early death.

<table><tr><td>Model</td><td> ${ \bf \sf M e a n } \pm { \sf S D }$ </td><td>Median [IQR]</td><td>Range</td><td>Early success</td><td>Early death</td><td>Budget reached</td></tr><tr><td>Seed-2.1-Pro</td><td> $3 4 . 0 \pm 1 7 . 7$ </td><td>33.0 [19.0, 50.0]</td><td>2-92</td><td>32.7%</td><td>35.7%</td><td>31.6%</td></tr><tr><td>Gemini-3.6-Flash</td><td> $3 4 . 1 \pm 1 8 . 6$ </td><td>33.5 [19.0, 50.0]</td><td>3-81</td><td>25.7%</td><td>40.6%</td><td>33.6%</td></tr><tr><td>Gemini-3.1-Pro</td><td> $3 5 . 1 \pm 1 8 . 5$ </td><td>34.0 [19.3, 50.0]</td><td>4-102</td><td>25.7%</td><td>34.5%</td><td>39.8%</td></tr><tr><td>GPT-5.6-sol</td><td> $3 6 . 1 \pm 1 8 . 7$ </td><td>37.0 [19.3, 50.0]</td><td>3-82</td><td>22.8%</td><td>36.0%</td><td>41.2%</td></tr></table>

Table 27 Post-preload episode-length distribution by model. Each row contains 342 episodes. Early success and early death both require the episode to end strictly before its configured action budget; the pass label distinguishes the two outcomes.

Table 28 Post-preload episode length by environment, pooled across the four leading models and sorted by mean length. ES and ED denote early success and early death, respectively.
<table><tr><td>Environment</td><td>n</td><td>Mean</td><td></td><td>Median [IQR]</td><td>Range</td><td>ES</td><td>ED</td></tr><tr><td>Tennis</td><td>20</td><td>10.0</td><td></td><td>8.0 [4.0, 14.5]</td><td>4-23</td><td>40.0%</td><td>60.0%</td></tr><tr><td>Procgen Leaper</td><td>20</td><td>10.6</td><td></td><td>9.0 [5.8, 14.2]</td><td>2-22</td><td>0.0%</td><td>100.0%</td></tr><tr><td>CliffWalking</td><td>20</td><td>10.7</td><td></td><td>10.0 [10.0, 12.0]</td><td>8-13</td><td>100.0%</td><td>0.0%</td></tr><tr><td>Procgen Bossfight</td><td>20</td><td>12.3</td><td></td><td>11.0 [6.5, 15.8]</td><td>4-29</td><td>20.0%</td><td>80.0%</td></tr><tr><td>Taxi</td><td>20</td><td>15.9</td><td></td><td>16.0 [13.0, 17.0]</td><td>12-27</td><td>100.0%</td><td>0.0%</td></tr><tr><td>Procgen Dodgeball</td><td>20</td><td>17.9</td><td></td><td>16.5 [6.0, 28.0]</td><td>3-40</td><td>0.0%</td><td>100.0%</td></tr><tr><td>Custom LavaCrossing</td><td>60</td><td>18.2</td><td></td><td>20.0 [14.0, 21.0]</td><td>2-36</td><td>65.0%</td><td>35.0%</td></tr><tr><td>Highway</td><td>80</td><td>18.7</td><td></td><td>19.0 [12.0, 26.0]</td><td>4-39</td><td>30.0%</td><td>70.0%</td></tr><tr><td>Frostbite</td><td>20</td><td>19.9</td><td></td><td>18.0 [15.0, 25.0]</td><td>15-32</td><td>45.0%</td><td>55.0%</td></tr><tr><td>Procgen Bigfish</td><td>20</td><td>21.0</td><td>15.5</td><td>[12.2, 31.2]</td><td>3-48</td><td>40.0%</td><td>55.0%</td></tr><tr><td>CartPole</td><td>20</td><td>21.2</td><td></td><td>24.0 [14.0, 30.0]</td><td>6-30</td><td>0.0%</td><td>60.0%</td></tr><tr><td>Procgen Caveflyer</td><td>20</td><td>23.1</td><td></td><td>19.0 [9.0, 27.5]</td><td>6-60</td><td>40.0%</td><td>25.0%</td></tr><tr><td>Seaquest</td><td>20</td><td>23.1</td><td>17.0</td><td> [6.0, 43.2]</td><td>5-50</td><td>0.0%</td><td>85.0%</td></tr><tr><td>FlappyBird</td><td>20</td><td>27.1</td><td>31.5</td><td>[9.0, 37.0]</td><td>6-70</td><td>60.0%</td><td>40.0%</td></tr><tr><td>Custom Breakout</td><td>20</td><td>29.1</td><td></td><td>23.0 [21.8, 38.5]</td><td>3-92</td><td>75.0%</td><td>20.0%</td></tr><tr><td>MountainCar</td><td>20</td><td>30.4</td><td>29.0</td><td>[25.0, 35.2]</td><td>19-52</td><td>95.0%</td><td>0.0%</td></tr><tr><td>Custom UnlockPickup</td><td>60</td><td>30.6</td><td>30.0</td><td>[24.8, 40.0]</td><td>16-40</td><td>50.0%</td><td>20.0%</td></tr><tr><td>Procgen Ninja</td><td>20</td><td>31.2</td><td>26.5</td><td>[23.8, 34.0]</td><td>16-102</td><td>40.0%</td><td>60.0%</td></tr><tr><td>Riverraid</td><td>100</td><td>32.6</td><td>37.0</td><td>[16.0, 50.0]</td><td>5-50</td><td>0.0%</td><td>71.0%</td></tr><tr><td>RoadRunner</td><td>20</td><td>33.0</td><td>31.5</td><td>[25.8, 40.5]</td><td>7-54</td><td>30.0%</td><td>20.0%</td></tr><tr><td>Berzerk</td><td>68</td><td>33.7</td><td>26.0</td><td>[19.0, 46.0]</td><td>10-70</td><td>45.6%</td><td>44.1%</td></tr><tr><td>ChopperCommand</td><td>20</td><td>33.7</td><td>37.0</td><td>[18.0, 50.0]</td><td>11-50</td><td>0.0%</td><td>65.0%</td></tr><tr><td>Assault</td><td>20</td><td>36.0</td><td>35.5</td><td>[30.2, 43.0]</td><td>18-48</td><td>45.0%</td><td>20.0%</td></tr><tr><td>Trondead</td><td>100</td><td>36.5</td><td>43.0</td><td>[24.0, 50.0]</td><td>6-50</td><td>0.0%</td><td>59.0%</td></tr><tr><td>Procgen Heist</td><td>20</td><td>36.9</td><td>39.0</td><td>[31.5, 44.2]</td><td>17-52</td><td>55.0%</td><td>0.0%</td></tr><tr><td>Asterix</td><td>20</td><td>37.8</td><td>42.5</td><td>[25.8, 50.0]</td><td>12-50</td><td>0.0%</td><td>55.0%</td></tr><tr><td>MsPacman</td><td>20</td><td>42.0</td><td>44.0</td><td>[38.0, 50.0]</td><td>19-50</td><td>0.0%</td><td>55.0%</td></tr><tr><td>BattleZone</td><td>60</td><td>42.2</td><td>50.0</td><td>[32.2, 50.0]</td><td>17-50</td><td>0.0%</td><td>36.7%</td></tr><tr><td>Turmoil</td><td>20</td><td>43.6</td><td>50.0</td><td>[44.0, 50.0]</td><td>18-50</td><td>0.0%</td><td>30.0%</td></tr><tr><td>Alien</td><td>20</td><td>44.1</td><td>44.5</td><td>[33.5, 60.0]</td><td>12-72</td><td>35.0%</td><td>45.0%</td></tr><tr><td>DemonAttack</td><td>100</td><td>44.2</td><td>50.0</td><td>[50.0, 50.0]</td><td>8-50</td><td>0.0%</td><td>23.0%</td></tr><tr><td>Enduro</td><td>80</td><td>50.0</td><td>50.0</td><td>[50.0, 50.0]</td><td>50-50</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Pong</td><td>20</td><td>50.0</td><td>50.0</td><td>[50.0, 50.0]</td><td>50-50</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Tetris</td><td>20</td><td>50.5</td><td>52.0</td><td>[38.5, 57.0]</td><td>32-84</td><td>5.0%</td><td>10.0%</td></tr><tr><td>Freeway</td><td>80</td><td>53.3</td><td>61.0</td><td>[28.0, 72.0]</td><td>21-81</td><td>27.5%</td><td>0.0%</td></tr><tr><td>Custom MultiRoom</td><td>60</td><td>56.1</td><td>57.5</td><td>[45.8, 70.0]</td><td>35-70</td><td>73.3%</td><td>0.0%</td></tr><tr><td>Environment</td><td></td><td></td><td>n Mean Median [IQR] Range</td><td></td><td></td><td>ES</td><td>ED</td></tr><tr><td>Acrobot</td><td>20</td><td>62.4</td><td>60.0 [54.0, 69.8]</td><td></td><td>42-82</td><td>55.0%</td><td>0.0%</td></tr></table>

Across models, mean length is 34.0–36.1 steps (SD 17.7–18.7). Pooled outcomes are 366/1,368 early successes (26.8%), 502 early deaths (36.7%), and 500 budget reaches (36.5%). Tennis (10.0), Procgen Leaper (10.6), and ClifWalking (10.7) are shortest by mean; Acrobot (62.4), Custom MultiRoom (56.1), and Freeway (53.3) are longest. Because budgets vary, duration is not dificulty. Early-death rate peaks at 100% for Procgen Leaper and Procgen Dodgeball, 85% for Seaquest, and 80% for Procgen Bossfight.