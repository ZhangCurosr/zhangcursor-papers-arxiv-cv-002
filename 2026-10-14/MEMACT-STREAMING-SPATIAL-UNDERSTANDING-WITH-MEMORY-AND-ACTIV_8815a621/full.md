# MEMACT: STREAMING SPATIAL UNDERSTANDING WITH MEMORY AND ACTIVE EXPLORATION

Yuxi Wei<sup>1</sup> Wei Huang<sup>1</sup> Qirui Chen<sup>3</sup> Lu Hou<sup>2,\*</sup> Xiaojuan Qi<sup>1,\*</sup>

<sup>1</sup>The University of Hong Kong <sup>2</sup>Yinwang Intelligent Technology Co. Ltd <sup>3</sup>Shanghai Jiao Tong University Corresponding authors

Project page: https://vfishc.github.io/memact/

## ABSTRACT

Spatial understanding is essential for embodied agents navigating and interacting with the physical world. As observations arrive incrementally, agents must reason in a streaming manner. Yet streaming understanding must also be closedloop: task-relevant evidence may require both recalling past observations and actively acquiring new views. Existing streaming approaches largely emphasize processing incoming observations, leaving task-driven active exploration underexplored. We introduce MemAct, a framework for streaming spatial understanding with memory and action. A shared vision–language model incrementally writes structured historical memory, retrieves records and past frames on demand, and acquires additional evidence through movement and turning. Retrieved evidence and action observations inform subsequent decisions, enabling the agent to continue exploring or answer within a unified reasoning loop. We develop the MemAct suite, a data generation pipeline and collection across real and simulated environments, with approximately 1M training examples for spatial reasoning, memory writing, retrieval, and exploration. Its MemAct-bench comprises 20,000 main QA pairs evaluating current-observation understanding, historical evidence use, and active exploration, plus 2,000 separate Timing questions. MemAct combines supervised fine-tuning for memory and interaction with counterfactual action optimization using cumulative evidence gains, action costs, and terminal answer quality to improve exploration and stopping decisions. Experiments show that MemAct outperforms existing spatial understanding and streaming baselines and approaches frontier multimodal models on MemAct-bench while retaining streaming inference efficiency.

## 1 INTRODUCTION

Spatial understanding enables embodied agents to reason about objects, their locations, geometry, and relationships while navigating and interacting with an environment (Yang et al., 2024a;b; Huang et al., 2022; Jatavallabhula et al., 2023; Gu et al., 2023). In a live setting, observations arrive incrementally (Das et al., 2017; Gordon et al., 2017; Yokoyama et al., 2023): future views are unavailable, previously seen objects may leave the visual context, and some task-relevant details may never appear unless the agent changes viewpoint. A spatial question may therefore be answerable from the current view, require recalling an earlier observation, or require acquiring a new one. Streaming spatial understanding is consequently a closed-loop process in which the agent uses available evidence to decide whether to answer, retrieve history, or explore.

Recent streaming work advances spatial reasoning over incrementally arriving observations (Xu et al., 2025; Yu et al., 2026; Wang et al., 2025b). Yet these methods generally follow a predetermined camera stream: earlier evidence may leave the retained context, while needed views may never be acquired (Yang et al., 2024b; Hong et al., 2026). Active embodied QA systems choose new viewpoints, often using geometric maps or structured spatial memories (Jiang et al., 2025; Yang et al., 2024b; Ginting et al., 2025). Some also retain experience from before a question arrives. We study a complementary challenge: training a single streaming VLM to build retrievable memory as RGB observations arrive and, once queried, decide whether to retrieve earlier evidence, acquire a new view, or answer. Each result informs the next decision, without supplying depth maps or a reconstructed 3D map to the model at inference time. Figure 1 contrasts this memory-and-action loop with offline and passive streaming QA.

![](images/dee5cbbaf4b5d99efab079d77613b8192e865f5a2ada95de0c3f2d17d466008b.jpg)  
Figure 1: Offline, passive streaming, and closed-loop streaming spatial understanding. MemAct combines reasoning over incremental observations, historical memory retrieval, and active acquisition of missing evidence in a shared exploration-and-answering loop.

This setting poses three distinct challenges. First, a streaming VLM must preserve potentially useful evidence before knowing the question, even as earlier views leave its limited context. Second, once queried, it must determine whether the needed evidence is visible now, recoverable from history, or obtainable only through a new camera view. Learning this distinction requires causal supervision over what the agent has seen and what its actions reveal. Third, an informative view may require preparatory movements with little immediate gain, so training must credit their delayed value while teaching the model when to stop exploring and answer.

We introduce MemAct, a simple design that makes memory and exploration native decisions of a streaming VLM. The model processes incoming RGB observations incrementally and periodically generates compact, timestamped records linked to stored frames. When queried, it uses the same generation interface to answer, request past evidence, or select a discrete camera action. A a simple rule-based interface returns the requested records or frames, or the resulting new view, to the model for its next decision. This lets the VLM learn what to record, retrieve, and observe without a separate spatial planner, depth maps, or a reconstructed 3D scene map as model inputs. (Section 4)

For data construction, we introduce MemAct-data, a data suite that combines streaming spatial QA with supervision for memory writing, retrieval, and multi-step exploration. An automated pipeline constructs approximately one million training examples from real and simulated visual streams; constructed action supervision from changes in target visibility, enabling the model to recognize insufficient evidence and learn how to explore for more informative views. We also introduce MemAct-bench, with 20,000 questions testing reasoning from current, historical, and newly acquired observations, plus 2,000 separate questions testing when to answer. (Section 3)

For training, supervised fine-tuning establishes the shared memory, retrieval, exploration, and answering protocol. To credit preparatory actions with delayed benefits, bandit-based counterfactual optimization compares camera actions and immediate answering from the same causal state through multi-step continuations. Cumulative evidence gains, action costs, and terminal answer quality encourage useful exploration beyond immediate gains while learning when to stop; supervised QA and memory replay preserve existing capabilities (Section 4.4).

On MemAct-bench, MemAct-9B achieves 67.27% overall multiple-choice accuracy and 60.23 numerical mean relative accuracy, compared with 61.39% and 48.65 for StreamingVLM. Its aggregate scores are close to those of the evaluated GPT-6 Astra model, while it leads the separate responsetiming evaluation. Ablations show that explicit memory substantially improves questions requiring historical evidence, and that active exploration and counterfactual optimization improve questions requiring new views. MemAct also remains competitive on established spatial reasoning bench marks, including VSI-Bench, ReVSI, MMSI-Bench, and ViewSpatial. Together, these results support spatial understanding that can use the present, recall the past, and actively acquire missing evidence within a causal stream.

Our contributions are fourfold: 1)Task formulation. We extend streaming spatial understanding with task-driven active exploration and introduce MemAct, unifying historical evidence retrieval, new-view acquisition, and answering. 2) Data construction. We develop an efficient, automated pipeline for generating memory, retrieval, and exploration supervision, producing the MemAct suite with approximately 1M training examples from real and simulated environments. 3) Training strategy. We combine supervised interaction learning and cumulative-return counterfactual optimization in multi-stage training to improve exploration and stopping decisions. 4) Evaluation. We build MemAct-bench: 20,000 main QA pairs and 2,000 separate Timing questions evaluating memory, exploration, task performance, and streaming inference efficiency.

## 2 RELATED WORK

Spatial Understanding. Recent work advances spatial understanding through dedicated benchmarks (Azuma et al., 2021; Ma et al., 2022; Fu et al., 2024) and spatially grounded training data (Chen et al., 2024; Cheng et al., 2024; Song et al., 2024). VSI-Bench (Yang et al., 2024a) evaluates spatial relations, measurement estimation, and spatiotemporal reasoning from indoor videos. SAT (Ray et al., 2024) synthesizes static and dynamic spatial QA in simulation, while Cambrian-S (Yang et al., 2025a) develops VSI-590K and spatial training, extending to continual sensing through latent frame prediction and surprise-driven memory management. These approaches strengthen spatial reasoning over supplied visual evidence. MemAct contributes an automated pipeline that constructs supervision for historical memory, retrieval, and multi-step explo ration in streaming spatial QA.

Streaming Spatial Understanding. Streaming methods address causal processing and efficient retention of continuously arriving observations (Zhang et al., 2024; Yao et al., 2025; He et al., 2024). StreamingVLM (Xu et al., 2025) combines a compact KV cache with overlapping-chunk SFT to align training and streaming inference. Stream3D-VLM (Yu et al., 2026) integrates incremental geometry priors and geometry-adaptive voxel compression for online spatial understanding. StreamEQA (Wang et al., 2025b) evaluates perception, interaction, and planning across backward, real-time, and forward reasoning on recorded videos. StreamingBench (Lin et al., 2024) evaluates online understanding; VideoChat-Online (Huang et al., 2024) and LiveCC (Chen et al., 2025) develop memory-based processing and real-time commentary, respectively. MemAct additionally learns observation acquisition: the model retrieves historical evidence or executes camera actions, incorporates their outcomes, and decides whether to continue exploring or answer.

Active Perception. Active perception acquires task-relevant evidence through viewpoint selection and environmental interaction (Majumdar et al., 2024). EXPRESS-Bench (Jiang et al., 2025) evaluates the exploration and reasoning capability of VLMs. ESI-Bench (Hong et al., 2026) evaluate the perception–action loop through perception, locomotion, and manipulation, exposing failures in action selection and premature commitment despite insufficient evidence. MemoryEQA (Zhai et al., 2025) uses structured memory throughout exploration, while Mind Palace (Ginting et al., 2025) combines long-term recall with exploration and stopping. MemAct contributes a shared streaming VLM trained for memory writing, retrieval, and camera-action selection, with cumulative-return optimization of exploration and stopping using downstream evidence and answer quality.

## 3 CONSTRUCTING THE MEMACT SUITE

Our pipeline builds supervision for recording evidence, retrieving history, and acquiring new views through five stages: observation acquisition, metadata alignment, memory and QA generation, action rendering, and stream assembly (Figure 2). Each question is linked to its arrival time, supporting observations, and memory or action supervision. Aligned metadata supports answer computation and camera-transition selection but remains separate from model inputs.

![](images/1e25b53f1201a410bdbe116613be789c547d7dee246a61e4bdce9d6243679d22.jpg)  
Figure 2: MemAct suite. Aligned temporal, camera, object, visibility, and scene metadata supports memory and spatial QA supervision from real and simulated observations. Dedicated rendering and geometric/evidence checks yield discrete multi-step camera-action data; all supervision forms causal streams.

## 3.1 OBSERVATION CONSTRUCTION AND EVIDENCE ALIGNMENT

Simulated observations use 3D-FRONT (Fu et al., 2021) layouts and assets after removing conspicuously interpenetrating or implausibly scaled objects. Camera trajectories scan rooms in randomized visitation orders using mesh-based A\* planning (Hart et al., 1968) and Bezier smoothing. Multi-´ ple trajectories diversify observation histories, while Cycles (Blender Foundation, 2026) rendering varies floor/ceiling textures and HDRI lighting. Trajectory construction is detailed in Appendix A.

Real sequences use ScanNet (Dai et al., 2017), ScanNet++ (Yeshwanth et al., 2023), and ARKitScenes (Baruch et al., 2021) videos, camera information, and 3D annotations. Both sources share the memory and QA pipeline; simulation additionally renders action-conditioned views with scene geometry, appearance, and imaging settings fixed within each interaction.

Unified Metadata and Temporal Evidence Alignment. We standardize categories, units, and coordinates across five metadata groups: temporal frame/time mappings, camera intrinsics and poses, object identities and geometry, visibility annotations and image-space measurements, and scene geometry and occupancy. Timestamps align projected objects and source visibility annotations with observations; field definitions appear in Appendix A.2.

Visibility indicators, projected-box occupancy, framing completeness, and visible-pixel fractions remain distinct. Initial simulated streams use source visibility and clipped box projections; action views additionally provide object masks. Projected occupancy measures image extent, not unoccluded surface area (Appendix A.2). At question time t, inputs contain only the causal prefix $O _ { \leq t } = \{ o _ { i } : t _ { i } \leq t \}$ ; action observations enter after their commands.

## 3.2 MEMORY AND SPATIAL QA SUPERVISION

Incremental memory targets. At 1 FPS, observations form five-second write windows. Memory targets contain timestamp-linked frame references, per-frame category counts, and optional imageplane relations. Visibility selects observed instances; projected positions on a 3 × 3 grid support unambiguous directional relations. Records describe observed content, not a complete scene graph; empty frames and final partial windows are retained.

Per-frame counts enumerate visible instances of each category; historical distinct-instance answers deduplicate sightings across the causal prefix using offline instance identities, which are excluded from model-visible memory (Appendix A).

Questions and evidence requirements. We select a time and template, bind eligible referents, compute the answer, and attach evidence. Object/camera metadata supplies distances, dimensions, and relative directions; temporal/visibility metadata supplies appearances, order, counts, and changes. Room-area questions require floor polygons and coverage checks. Candidates must satisfy visibility, referent-uniqueness, and answer-separation criteria.

Current QA uses the current observation or a specified recent window. Memory QA requires earlier appearances, previously observed instances not all currently visible, or historical relations; direct answering is allowed when this evidence remains in context. LLM rewriting varies wording while preserving referents, temporal constraints, and annotation-derived answers. Template calculations and examples appear in Appendix A.

Retrieval supervision. Historical evidence determines query categories, optional time ranges, and image requests. The deterministic router applies inference-time retrieval rules to constructed memory, yielding records and optional historical frames. Requests, responses, and answers form supervised traces; private instance-level checks exclude traces without sufficient evidence within budget without altering router outputs (Appendix C).

## 3.3 MULTI-ROUND ACTION EVIDENCE CONSTRUCTION

Dedicated simulation rendering supplies action supervision. We bind identifiable targets and compute fixed answers: size is the longest physical bounding-box side, and distance is the Euclidean distance between 3D centers. Scene, targets, question, and answer remain unchanged, so exploration changes evidence rather than the task.

Discrete atomic actions. Five parameter-free commands translate forward, left, or right by 0.5 m in the current camera’s horizontal frame, or rotate left/right by 45<sup>◦</sup>. Translation preserves orientation; rotation preserves position. The executor fixes magnitudes; supervision uses the model’s commands. Collision and clearance checks precede rendering each admissible resulting state.

Evidence-based sequence selection. Joint target support is the minimum image-occupancy score across required objects, ensuring evidence for both targets in pairwise questions. Endpoints must satisfy occupancy and framing thresholds and exceed the best pre-question joint support by both relative and absolute margins. We distinguish acquiring historically missing adequate views from improving existing ones; measurement definitions and selection thresholds are in Appendix A.

We execute and render up to three admissible actions sequentially, retaining the earliest endpoint meeting these criteria. Intermediate actions may reposition the camera without immediate gain. Each demonstration stores the causal prefix, fixed question, command–observation pairs, and answer. Counterfactual optimization additionally uses rendered alternative outcomes; demonstration thresholds and RL rewards are separately defined in Appendices A and D.2.

Streaming Assembly and Quality Control. Across all task types, observations, memory writes, questions, commands, and tool results are assembled chronologically. Validation checks media, temporal/object references, answers, duplicates, command–observation alignment, budgets, and predecessor-state continuity. Missing evidence, failed checks, or excessive token length exclude a sample; construction metadata and private labels remain separate from public events (Appendix A).

The MemAct-data contains approximately 1M examples, all for training, while MemAct-bench contains 20,000 main QA pairs covering current understanding, historical evidence use, and active exploration, plus 2,000 separate Timing questions. Training-data composition and action-length distributions are in Appendix B, with construction thresholds in Appendix A.

## 4 STREAMING SPATIAL REASONING WITH MEMORY AND ACTION

## 4.1 PROBLEM FORMULATION AND OVERVIEW

MemAct uses a shared VLM for memory and evidence-guided interaction. Figure 3 links inference to supervised interaction learning and counterfactual action optimization. At time t, the agent maintains causal context $C _ { t }$ in an incremental KV cache and external memory M . Following compactcache streaming (Xu et al., 2025), the oldest visual tokens are preferentially evicted at the token limit. A deterministic harness retains structured memory and historical images independently, enabling retrieval beyond the visual context.

![](images/8dd46d15d414de392eed15f5e51478fdede2fce8bd0dcf57166dd39d1a41ce7c.jpg)  
Figure 3: MemAct training and inference. A shared VLM learns incremental memory writing, answering, historical retrieval, and discrete-action view acquisition through SFT. Counterfactual optimization improves exploration and stopping using cumulative evidence gains, action costs, and terminal answer quality.

For question $q _ { t } ,$ , history $h _ { k }$ contains the question, available observations, prior commands/results, and remaining budgets at round k. The model answers, retrieves, or acts; periodic memory writing is separate during observation. The harness executes commands without a learned router. Decision inputs exclude future observations, ground-truth geometry, private instance identities, and rewards.

## 4.2 INCREMENTAL MEMORY WRITING AND RETRIEVAL

To preserve history beyond visual-cache eviction, MemAct writes increments $\Delta M _ { t }$ every five seconds: per-frame categories, counts, frame references, and optional image-plane relations. The update $M _ { t } \ = \ \mathrm { U p d a t e } ( M _ { t - 5 } , \Delta M _ { t } )$ validates structure and temporal references without semantic correction; storage and retry rules appear in Appendix C.

Structured records locate historical evidence; retrieved frames recover visual detail absent from those records. A use memory request specifies categories, an optional time range, and image requirements, returning matching records, relations, and optional frames to shared context. Each call returns at most two images, with four distinct historical images per question; filtering and pagination follow Appendix C.

## 4.3 EVIDENCE SELECTION AND MULTI-ROUND ACTION

Memory recovers past evidence; camera actions acquire views missing from retained observations. MemAct answers directly when evidence suffices, including historical QA; otherwise, it retrieves or selects one of five camera commands with executor-fixed magnitudes (Section 3.3).

A retrieval or camera command $u _ { k }$ and result $z _ { k }$ yield $h _ { k + 1 } = { \mathrm { A p p e n d } } ( h _ { k } , u _ { k } , z _ { k } )$ , informing subsequent decisions. Successive camera actions build on preceding states to improve evidence, with at most three attempts and an independent four-call retrieval budget per question. Answering terminates interaction; action-budget exhaustion requests a final answer. Action views remain in question context without automatic memory writes; the original stream resumes after exploration. Figure 4 illustrates current-view QA, memory-based last-observation QA, and three-action view improvement; Appendix E adds QA and memory-write examples.

## 4.4 TRAINING STRATEGY

Stage I: Supervised Fine-Tuning. SFT learns writing, retrieval, answering, and single-/multi-step exploration from causal MemAct-data traces, including preparatory actions with delayed benefits, together with VSI-590K (Yang et al., 2025a). We minimize next-token cross-entropy over assistant tokens in memory writes, retrieval requests, camera commands, and answers, averaged within each sequence. Observations, prompts, and tool results supply context but are loss-masked; every decision uses only preceding events. Training settings and loss normalization appear in Appendix D.

![](images/00448ddf662ab4743bc16e017c0c59e13adf582ba9a049e0c92a589dfafdf6e5.jpg)  
Figure 4: Current/Memory/Action QA: available observations, historical evidence, and newly acquired views.

Stage II: Counterfactual Optimization with Cumulative Returns. After SFT, a contextual bandit objective uses multi-step candidate values to improve exploration and stopping. At history h, we evaluate the five camera commands and answer now from independent copies with identical causal state and budget. Each camera attempt costs one unit before continuation under frozen policy snapshot $\mu .$ Values capture later evidence, answer quality, and preparatory-action benefits.

Evidence and answer rewards. View quality $E ( x , q ) \in [ 0 , 1 ]$ combines normalized visible-pixel fractions, framing completeness that discounts cropping, and joint support limited by the weaker target in pairwise questions. The best quality over pre-question observations and preceding action outcomes is $\Phi ( h , q ) = \mathrm { m a x } _ { x \in \mathcal { O } ( h ) } \ : E ( x , q )$

Within an action rollout, let $h _ { j }$ denote the history after j camera attempts. Attempt $a _ { j }$ receives

$$
r _ { j } = \alpha [ \Phi ( h _ { j + 1 } , q ) - \Phi ( h _ { j } , q ) ] _ { + } - \lambda _ { a } c ( a _ { j } ) - \lambda _ { \mathrm { i n v } } { \bf 1 } [ \mathrm { i n v a l i d } ( a _ { j } ) ] .\tag{1}
$$

Revisiting previously acquired evidence earns no repeated gain. Invalid actions preserve the observation, receive no evidence gain, and consume an attempt. The terminal answer receives $R _ { \mathrm { Q A } } ( \hat { y } , y ^ { \star } )$ using exact option correctness or numerical MRA. Evidence functions, coefficients, and reward ablations are detailed in Appendix D.2.

Cumulative values and policy updates. For a rollout of T camera attempts, decision j has return

$$
G _ { j } = \sum _ { \ell = j } ^ { T - 1 } \gamma ^ { \ell - j } r _ { \ell } + \gamma ^ { T - j } R _ { \mathrm { Q A } } ( \hat { y } , y ^ { \star } ) .\tag{2}
$$

Answer-now terminates without camera cost. We estimate candidate values from B continuations as $\begin{array} { r } { \widehat { Q } ^ { \mu } ( h , a ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } G ^ { ( b ) } ( h , a ) } \end{array}$ and train on initial and visited nonterminal histories. These rollouts use available causal context without initiating additional memory retrieval. Branch observations enter the context after their corresponding actions; branch validation is specified in Appendix A.

Summed conditional log probabilities of candidate decision-value tokens, without length normalization, define a categorical policy $\pi _ { \theta } ( a \mid h )$ . We optimize

$$
\mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } _ { h } \left[ - \sum _ { a \in A ( h ) } \pi _ { \theta } ( a \mid h ) \widehat { Q } ^ { \mu } ( h , a ) + \beta D _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot \mid h ) \| \pi _ { \mathrm { S F T } } ( \cdot \mid h ) ) \right] .\tag{3}
$$

KL regularization anchors the policy to the frozen SFT reference (Ouyang et al., 2022). We refresh $\mu$ before each update and detach estimated values within it. Supervised QA and memory replay interleave with action optimization to preserve answering, tool use, and memory writing. Appendix D.2 specifies sampling, loss normalization, and rollout settings.

Table 1: MemAct-bench results. ACC and MRA assess multiple-choice and numerical questions, respectively; on-time rate (T), joint timing-and-answer accuracy (JA), and timing-gated MRA (JM) assess response timing. Latency (seconds) spans image/context readiness to complete answer output (exclusions: Section 5.1).
<table><tr><td rowspan="2">Model / baseline</td><td colspan="2">Current</td><td colspan="2">Memory</td><td colspan="2">Action</td><td colspan="2">Overall QA</td><td>Timing</td><td rowspan="2">Latency s/QA</td></tr><tr><td>ACC</td><td>MRA</td><td>ACC MRA</td><td></td><td>ACC</td><td>MRA</td><td>ACC MRA</td><td>T/JA/JM</td></tr><tr><td colspan="10">Closed-source models</td></tr><tr><td>GPT-6 Astra</td><td>68.16 53.04</td><td></td><td>64.43</td><td>68.68</td><td>59.57</td><td>49.82</td><td>66.74 61.15</td><td></td><td>81.3 / 54.2 / 49.6</td><td></td></tr><tr><td>Gemini-3.8-Flash</td><td></td><td>59.2646.29</td><td>56.41</td><td>59.55</td><td>52.20 43.42</td><td></td><td>58.17 53.14</td><td></td><td>67.1 / 38.9 / 36.0</td><td></td></tr><tr><td colspan="9">Reference baselines</td></tr><tr><td>Uniform options</td><td>23.22</td><td></td><td>22.71</td><td></td><td>25.00</td><td></td><td>23.11</td><td></td><td></td><td></td></tr><tr><td>Modal answer content</td><td>32.62</td><td>29.51</td><td>22.36</td><td>41.64</td><td>54.76</td><td>34.04</td><td>29.89</td><td>36.73</td><td></td><td></td></tr><tr><td>Modal option letter</td><td>26.10</td><td></td><td>25.68</td><td></td><td>27.68</td><td></td><td>26.00</td><td></td><td></td><td></td></tr><tr><td colspan="9">Open-source models</td></tr><tr><td>Qwen3.5-4B</td><td>36.31</td><td>22.96</td><td>46.63</td><td>43.98</td><td>31.50</td><td>21.44</td><td>39.50 34.23</td><td></td><td>30.9 / 12.2 / 10.5</td><td>1.558</td></tr><tr><td>Qwen3.5-9B</td><td>42.82</td><td>27.11</td><td>55.31</td><td>51.78</td><td>36.66</td><td>24.59</td><td>46.67</td><td>40.24</td><td>42.3 / 19.8 / 16.7</td><td>2.475</td></tr><tr><td>Qwen3.8-27B</td><td>44.47</td><td>28.51</td><td>57.24</td><td>53.72</td><td>39.65</td><td>27.23</td><td>48.44</td><td>42.10</td><td>46.4 / 22.4 / 19.5</td><td>5.423</td></tr><tr><td>InternVL3.5-8B</td><td>37.34</td><td>23.56</td><td>47.73</td><td>44.98</td><td>32.43</td><td>22.17</td><td>40.55</td><td>35.07</td><td>33.9 / 13.7 / 11.8</td><td>2.495</td></tr><tr><td>InternVL3.5-38B</td><td>39.74</td><td>25.03</td><td>51.66</td><td>48.78</td><td>34.88</td><td>24.08</td><td>43.45</td><td>37.86</td><td>38.8 / 16.8 / 14.6</td><td>5.446</td></tr><tr><td colspan="9">Spatial models</td><td></td></tr><tr><td>VLM-3R</td><td>43.52</td><td>27.71</td><td>56.63</td><td>52.69</td><td>36.78</td><td>24.96</td><td>47.55</td><td>40.98</td><td>44.0 / 21.0 / 17.5</td><td>2.622</td></tr><tr><td>Cambrian-S</td><td>42.54</td><td>26.56</td><td>54.20</td><td>51.38</td><td>35.71</td><td>24.30</td><td>46.11</td><td>39.80</td><td>43.4 / 20.1 / 16.8</td><td>2.446</td></tr><tr><td>Cambrian-S-FT</td><td>66.19</td><td>41.54</td><td>63.46</td><td>61.85</td><td>57.76</td><td>39.87</td><td>65.10</td><td>52.40</td><td>73.3 / 47.6 / 38.5</td><td>2.523</td></tr><tr><td colspan="9">Streaming models</td><td></td></tr><tr><td>StreamingVLM</td><td>63.24</td><td>38.06</td><td>58.85</td><td>58.54</td><td>46.03</td><td>33.66</td><td>61.39</td><td>48.65</td><td>63.6 /39.1/ 31.3</td><td>0.377</td></tr><tr><td>Stream3D-VLM</td><td>60.42</td><td>36.78</td><td>55.88</td><td>56.46</td><td>43.76</td><td>32.32</td><td>58.54</td><td>46.92</td><td>61.0 / 35.7 / 29.2</td><td>0.410</td></tr><tr><td>MemAct-4B</td><td>62.01</td><td>37.50</td><td>64.13</td><td>63.80</td><td>51.54</td><td>37.68</td><td>62.43</td><td>51.87</td><td>84.9 / 53.1 / 44.0</td><td>0.383</td></tr><tr><td>MemAct-9B</td><td>67.22</td><td>52.28</td><td>68.11</td><td>67.72</td><td>58.28</td><td>48.74</td><td>67.27</td><td>60.23</td><td>85.3 / 57.4 / 50.9</td><td>0.420</td></tr></table>

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

We compare closed-/open-source general-purpose VLMs, spatial and streaming models, and uniform-option, modal-answer-content, and modal-option-letter baselines. Models include GPT-6 Astra (OpenAI, 2026), Gemini-3.8-Flash (Google DeepMind, 2026), Qwen3.5 (Qwen Team, 2026a), Qwen3.8-27B (Qwen Team, 2026b), InternVL3.5 (Wang et al., 2025a), VLM-3R (Fan et al., 2025), Cambrian-S (Yang et al., 2025a), StreamingVLM (Xu et al., 2025), and Stream3D-VLM (Yu et al., 2026). We train MemAct on our dataset and VSI-590K (Yang et al., 2025a). At each decision, offline models receive 32 uniformly sampled frames from the available prefix; streaming models observe causally at 1 FPS. Multiple-choice and numerical questions use accuracy (ACC) and mean relative accuracy (MRA), respectively, following VSI-Bench (Yang et al., 2024a). Overall QA pools Current, Memory, and Action across domains by question count (13,134 multiple-choice; 6,866 numerical), scoring multi-turn questions per turn. Action is simulation-only, with all models using the same discrete interface and attempt budget and receiving action observations before answering.

Timing uses a separate 2,000-question set balanced across domains. Questions precede evidence; models observe at 1 FPS and wait or answer without memory/action tools. The first answer is final and on time within the inclusive five-second window from the first evidence-sufficient checkpoint. We report on-time rate (T), joint timing-and-answer accuracy (JA), and timing-gated MRA (JM), averaged over applicable questions (Appendix D.4).

Latency spans image/context readiness to complete answer output, excluding frame acquisition, observation maintenance, memory operations, action execution/rendering, and network/queueing delays. Overall latency averages the two domains, each with 10,000 QA turns (Appendix D.3).

## 5.2 MAIN RESULTS

All streaming models in Table 1 are also fine-tuned on our QA dataset; Cambrian-S is fine-tuned on our dataset’s offline conversion as a spatial baseline (Cambrian-S-FT).

MemAct-9B achieves the highest Overall ACC (67.27) and leads all three QA categories in both ACC and MRA among open-source, spatial, and streaming models. It exceeds StreamingVLM by 5.88/11.58 percentage points in Overall ACC/MRA; MemAct-4B also improves both metrics, particularly on Memory and Action QA. GPT-6 Astra leads Current and Action QA and Overall

Table 2: MemAct-9B Memory QA ablations: (1) content, (2) retention length, (3) write frequency.
<table><tr><td colspan="3">A. Memory components</td></tr><tr><td>Setting</td><td>ACC</td><td>MRA</td></tr><tr><td>No explicit memory</td><td>54.52</td><td>52.24</td></tr><tr><td>Full memory</td><td>68.11</td><td>67.72</td></tr><tr><td>GT memory</td><td>70.77</td><td>71.06</td></tr><tr><td>w/o Frames</td><td>64.92</td><td>63.72</td></tr><tr><td>w/o Count</td><td>68.10</td><td>66.62</td></tr><tr><td>w/o Relations</td><td>67.91</td><td>67.72</td></tr></table>

<table><tr><td colspan="2">B. Retention window</td></tr><tr><td>Window</td><td>ACC MRA</td></tr><tr><td>10 s 58.52</td><td>57.45</td></tr><tr><td>20 s 60.91</td><td>60.12</td></tr><tr><td>30 s 63.20</td><td>62.48</td></tr><tr><td>40 s 65.04</td><td>64.47</td></tr><tr><td>50 s 66.50</td><td>65.95</td></tr><tr><td>Full history 68.11</td><td>67.72</td></tr></table>

<table><tr><td colspan="2">C. Write interval</td></tr><tr><td>Interval</td><td>ACC MRA</td></tr><tr><td>1 s</td><td>67.47 66.98</td></tr><tr><td>2 s 67.82</td><td>67.34</td></tr><tr><td>3 s 68.13</td><td>67.72</td></tr><tr><td>5 s 68.11</td><td>67.72</td></tr><tr><td>10 s 66.60</td><td>65.83</td></tr></table>

Table 3: MemAct-9B Action QA ablations: (1) RL algorithm, (2) reward design, and (3) action-step budget.
<table><tr><td>Setting</td><td>ACC (%) (∆ pp)</td><td>MRA (%) (∆ pp)</td><td>Evidence gain (points)</td><td>Effective gain rate (%)</td><td>Mean steps</td></tr><tr><td colspan="6">A. Training strategy</td></tr><tr><td>No action</td><td>47.45 (+0.00)</td><td>35.27 (+0.00)</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>SFT only</td><td>52.92 (+5.47)</td><td>41.87 (+6.60)</td><td>14.26</td><td>43.41</td><td>2.19</td></tr><tr><td>SFT + Bandit (full)</td><td>58.28 (+10.83)</td><td>48.74 (+13.47)</td><td>29.13</td><td>71.14</td><td>2.02</td></tr><tr><td>SFT + GRPO</td><td>57.92 (+10.47)</td><td>48.20 (+12.93)</td><td>28.34</td><td>69.78</td><td>2.08</td></tr><tr><td colspan="6">B. Reward composition (cumulative)</td></tr><tr><td>Pixel visibility</td><td>51.39 (+3.94)</td><td>39.63 (+4.36)</td><td>10.40</td><td>35.13</td><td>2.31</td></tr><tr><td>+ Full framing</td><td>52.55 (+5.10)</td><td>43.11 (+7.84)</td><td>17.79</td><td>50.90</td><td>2.20</td></tr><tr><td>+ Two-target evidence</td><td>55.81 (+8.36)</td><td>45.03 (+9.76)</td><td>21.69</td><td>58.07</td><td>2.14</td></tr><tr><td>+ Historical gain (full)</td><td>58.28 (+10.83)</td><td>48.74 (+13.47)</td><td>29.13</td><td>71.14</td><td>2.02</td></tr><tr><td colspan="6">C. Action budget</td></tr><tr><td>No action (0 steps)</td><td>47.45 (+0.00)</td><td>35.27 (+0.00)</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>At most 1 step</td><td>54.61 (+7.16)</td><td>43.26 (+7.99)</td><td>17.85</td><td>50.85</td><td>0.94</td></tr><tr><td>At most 2 steps</td><td>56.93 (+9.48)</td><td>46.73 (+11.46)</td><td>25.45</td><td>64.84</td><td>1.57</td></tr><tr><td>At most 3 steps (full)</td><td>58.28 (+10.83)</td><td>48.74 (+13.47)</td><td>29.13</td><td>71.14</td><td>2.02</td></tr></table>

MRA, but MemAct-9B achieves comparable aggregate performance: +0.53 points in Overall ACC and −0.92 in MRA. Its highest Memory ACC (68.11) exceeds Astra by 3.68 points. MemAct thus combines competitive spatial QA with incremental observation processing and, for MemAct-9B, answer-generation latency of 0.420 seconds per QA turn. MemAct-9B also leads all three Timing metrics (85.3 T, 57.4 JA, and 50.9 JM), combining timely responses with accurate answers.

## 5.3 ABLATION STUDIES

Memory Ablations. Table 2 ablates memory content, retention, and write frequency. Full memory improves ACC/MRA over no explicit memory by 13.59/15.48 points; ground-truth memory adds another 2.66/3.34 points, indicating room for better memory construction. Historical RGB contributes most: removing it costs 3.19/4.00 ACC/MRA points. Counts mainly benefit numerical answers, while relations yield smaller ACC gains.

Extending retention from 10 seconds to full history consistently raises ACC/MRA to 68.11/67.72, supporting evidence use beyond recent observations. More frequent writes do not consistently help: three-second intervals maximize ACC, but our five-second interval matches MRA with only 0.02 points lower ACC and fewer writes; ten-second intervals reduce both metrics.

Action Ablations. Table 3 compares training, rewards, and budgets using common evidence-gain metrics and mean camera-action attempts, including invalid attempts. Cumulative-return optimization adds 5.36/6.87 ACC/MRA points over SFT-only training and reduces mean attempts from 2.19 to 2.02. GRPO (Shao et al., 2024) achieves comparable performance. Reward ablations share terminal answer rewards, action costs, and cumulative returns. Framing adds 1.16/3.48 ACC/MRA points over pixel visibility alone; joint target evidence adds 3.26/1.92 points, supporting complete views and balanced evidence for relational questions. Replacing current-view gains with gains over the best historical view adds 2.47/3.71 points and raises the effective gain rate from 58.07% to 71.14%, encouraging acquisition of new evidence. Raising the action budget from one to three attempts adds 3.67/5.48 ACC/MRA points, supporting multi-step exploration.

## 5.4 EVALUATION ON OTHER SPATIAL BENCHMARKS

We evaluate MemAct-9B on VSI-Bench (Yang et al., 2024a), ReVSI (Zhang et al., 2026), MMSI-Bench (Yang et al., 2025b), and ViewSpatial (Li et al., 2025), comparing against publicly reported results (Li et al., 2026; Zhang et al., 2026; Dai et al., 2026; Wang et al., 2026; TaichuAI, 2026; SpatialAxiom, 2026) in Table 4. Despite operating as a streaming model, MemAct-9B remains competitive on these spatial benchmarks. It leads the listed models on ViewSpatial (52.21), outperforms both specialized spatial baselines on ReVSI and MMSI-Bench, and obtains 65.90 on VSI-Bench, behind Cambrian-S-7B’s 67.50. These results suggest that our training supports spatial understanding beyond our benchmark, while retaining incremental observation processing.

## 6 CONCLUSION

MemAct combines structured memory writing, historical evidence retrieval, and active exploration in a streaming vision–language model for spatial understanding. The MemAct suite supports training and evaluation in real and simulated environments. SFT teaches memory and interaction behaviors; cumulative-return counterfactual optimization improves exploration and stopping decisions. Experiments show that historical evidence and newly acquired views improve spatial reasoning. Future work will extend discrete camera commands to more continuous, comprehensive action spaces and adapt MemAct to larger, more cluttered, and more dynamic scenes.

Table 4: MemAct versus published results on spatial understanding benchmarks.
<table><tr><td colspan="4">VSI- MMSI- View Model Bench ReVSI Bench Spatial</td></tr><tr><td colspan="4">Closed-source models</td></tr><tr><td>GPT-5.2</td><td>54.49 50.90</td><td>41.30</td><td>47.30</td></tr><tr><td>Gemini-3-Pro</td><td>55.10 60.90</td><td>45.20</td><td>50.40</td></tr><tr><td colspan="4">Open-source models</td></tr><tr><td>Qwen3.5-9B</td><td>57.90 51.60</td><td>36.50</td><td>48.20</td></tr><tr><td colspan="4">Spatial models</td></tr><tr><td>VLM-3R-7B 60.90</td><td>50.10</td><td>27.90</td><td>40.50</td></tr><tr><td>Cambrian-S-7B 67.50</td><td>49.10</td><td>27.10</td><td>41.30</td></tr><tr><td colspan="4">Our method</td></tr><tr><td>MemAct</td><td>65.90 55.30</td><td>37.80</td><td>52.21</td></tr></table>

## AI USE STATEMENT

AI tools assisted discussions of research ideas, code implementation, and manuscript preparation, including literature search and checking, language revision, figure editing, and experiment evaluation. Tools and models used included Codex, GPT-5.6- sol, GPT-6-Astra, and Gemini-3.8-Flash. Dataset construction also used LLM-assisted QA wording refinement while preserving grounded referents and geometry-derived answers. Separately, the benchmark evaluates language and vision–language models as experimental subjects. The authors directed these uses, reviewed and revised the assisted content, and take responsibility for the final paper and its scientific claims.

## REPRODUCIBILITY STATEMENT

Data construction and quality control are described in Appendix A, with training-data composition and sampling in Appendix B. Appendix C specifies memory storage and retrieval, and Appendix D provides training settings, action-optimization objectives, and evaluation protocols, including the separate Timing evaluation. Qualitative examples are provided in Appendix E. We will provide supplementary files to support these descriptions and facilitate reproduction.

## REFERENCES

Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. ScanQA: 3D Question Answering for Spatial Scene Understanding. arXiv preprint arXiv:2112.10482, 2021. doi: 10. 48550/arXiv.2112.10482. URL https://arxiv.org/abs/2112.10482.

Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data. arXiv preprint arXiv:2111.08897, 2021. doi: 10.48550/arXiv.2111.08897. URL https://arxiv.org/ abs/2111.08897.

Blender Foundation. Cycles. Official renderer documentation, 2026. URL https://www. cycles-renderer.org/. Accessed 2026-09-24.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. arXiv preprint arXiv:2401.12168, 2024. doi: 10.48550/arXiv.2401. 12168. URL https://arxiv.org/abs/2401.12168.

Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale. arXiv preprint arXiv:2504.16030, 2025. doi: 10.48550/arXiv.2504.16030. URL https://arxiv.org/abs/2504.16030.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models. arXiv preprint arXiv:2406.01584, 2024. doi: 10.48550/arXiv.2406.01584. URL https://arxiv. org/abs/2406.01584.

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. arXiv preprint arXiv:1702.04405, 2017. doi: 10.48550/arXiv.1702.04405. URL https://arxiv.org/ abs/1702.04405.

Yalun Dai, Hao Li, Shulin Tian, Runmao Yao, Yuhao Dong, Fangzhou Hong, Zhaoxi Chen, Fangfu Liu, Baoliang Tian, Dingwen Zhang, Tao Wang, Kim-Hui Yap, and Ziwei Liu. S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence. arXiv preprint arXiv:2606.20515v1, 2026. doi: 10.48550/arXiv.2606.20515. URL https://arxiv.org/abs/2606.20515v1.

Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied Question Answering. arXiv preprint arXiv:1711.11543, 2017. doi: 10.48550/arXiv.1711. 11543. URL https://arxiv.org/abs/1711.11543.

Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Shijie Zhou, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, and Rakesh Ranjan. VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction. arXivpreprint arXiv:2505.20279, 2025. doi: 10.48550/arXiv.2505.20279. URL https://arxiv.org/abs/2505.20279.

Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, and Hao Zhang. 3D-FRONT: 3D Furnished Rooms With layOuts and semaNTics. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10933–10942, 2021. URL https: //openaccess.thecvf.com/content/ICCV2021/html/Fu\_3D-FRONT\_3D\_ Furnished\_Rooms\_With\_layOuts\_and\_semaNTics\_ICCV\_2021\_paper.html.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal Large Language Models Can See but Not Perceive. arXiv preprint arXiv:2404.12390, 2024. doi: 10.48550/arXiv.2404.12390. URL https://arxiv.org/abs/2404.12390.

Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai Krishna, Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim, Mykel J. Kochenderfer, Ali-akbar Agha-mohammadi, and Shayegan Omidshafiei. Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering. arXiv preprint arXiv:2507.12846, 2025. doi: 10.48550/arXiv.2507.12846. URL https://arxiv.org/ abs/2507.12846.

Google DeepMind. Gemini 3.8 Flash Model Card. Model card, 2026. URL https:// deepmind.google/models/model-cards/gemini-3-8-flash/. Accessed 2026- 09-24.

Daniel Gordon, Aniruddha Kembhavi, Mohammad Rastegari, Joseph Redmon, Dieter Fox, and Ali Farhadi. IQA: Visual Question Answering in Interactive Environments. arXiv preprint arXiv:1712.03316, 2017. doi: 10.48550/arXiv.1712.03316. URL https://arxiv.org/ abs/1712.03316.

Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, Chuang Gan, Celso Miguel de Melo, Joshua B. Tenenbaum, Antonio Torralba, Florian Shkurti, and Liam Paull. ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. arXiv preprint arXiv:2309.16650, 2023. doi: 10.48550/arXiv.2309.16650. URL https://arxiv.org/ abs/2309.16650.

Peter E. Hart, Nils J. Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2):100– 107, 1968. doi: 10.1109/TSSC.1968.300136. URL https://doi.org/10.1109/TSSC. 1968.300136.

Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding. arXiv preprint arXiv:2404.05726, 2024. doi: 10.48550/arXiv.2404.05726. URL https://arxiv.org/abs/2404.05726.

Yining Hong, Jiageng Liu, Han Yin, Manling Li, Leonidas Guibas, Li Fei-Fei, Jiajun Wu, and Yejin Choi. ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop. arXiv preprint arXiv:2605.18746, 2026. doi: 10.48550/arXiv.2605.18746. URL https: //arxiv.org/abs/2605.18746.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685, 2021. doi: 10.48550/arXiv.2106.09685. URL https://arxiv.org/ abs/2106.09685.

Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual Language Maps for Robot Navigation. arXiv preprint arXiv:2210.05714, 2022. doi: 10.48550/arXiv.2210.05714. URL https://arxiv.org/abs/2210.05714.

Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online Video Understanding: OVBench and VideoChat-Online. arXiv preprint arXiv:2501.00584, 2024. doi: 10.48550/arXiv.2501.00584. URL https://arxiv. org/abs/2501.00584.

Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. ConceptFusion: Open-set Multimodal 3D Mapping. arXiv preprint arXiv:2302.07241, 2023. doi: 10.48550/arXiv.2302.07241. URL https://arxiv.org/abs/2302.07241.

Kaixuan Jiang, Yang Liu, Weixing Chen, Jingzhou Luo, Ziliang Chen, Ling Pan, Guanbin Li, and Liang Lin. Beyond the destination: A novel benchmark for exploration-aware embodied question answering. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 9091–9101. IEEE, 2025. doi: 10.1109/ICCV51701.2025. 00850. URL https://doi.org/10.1109/ICCV51701.2025.00850.

Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. ViewSpatial Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models. arXiv preprint arXiv:2505.21500, 2025. doi: 10.48550/arXiv.2505.21500. URL https://arxiv. org/abs/2505.21500.

Yunheng Li, Guohong Mu, Hao Li, Shengsheng Qian, Dingwen Zhang, Qibin Hou, and Ming-Ming Cheng. Annotations as Rollouts: Efficient and Scalable Reinforcement Learning for Video MLLMs. arXiv preprint arXiv:2608.20492v1, 2026. doi: 10.48550/arXiv.2608.20492. URL https://arxiv.org/abs/2608.20492v1.

Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding. arXiv preprint arXiv:2411.03628, 2024. doi: 10.48550/arXiv.2411.03628. URL https:// arxiv.org/abs/2411.03628.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101, 2017. doi: 10.48550/arXiv.1711.05101. URL https://arxiv.org/ abs/1711.05101.

Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. SQA3D: Situated Question Answering in 3D Scenes. arXiv preprint arXiv:2210.07474, 2022. doi: 10.48550/arXiv.2210.07474. URL https://arxiv.org/abs/2210.07474.

Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Alexander Sax, and Aravind Rajeswaran. OpenEQA: Embodied Question Answering in the Era of Foundation Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16488–16498, June 2024. URL https://openaccess.thecvf.com/content/ CVPR2024/html/Majumdar\_OpenEQA\_Embodied\_Question\_Answering\_in\_ the\_Era\_of\_Foundation\_Models\_CVPR\_2024\_paper.html.

OpenAI. GPT-6 Astra System Card. System card, 2026. URL https://deploymentsafety. openai.com/gpt-6-astra. Accessed 2026-09-24.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. doi: 10.48550/arXiv.2203.02155. URL https://arxiv.org/abs/2203.02155.

Qwen Team. Qwen3.5: Towards Native Multimodal Agents. Official release; citation recommended by the Qwen3.5 model cards, 2026a. URL https://qwen.ai/blog?id=qwen3.5. Accessed 2026-09-24.

Qwen Team. Qwen3.8-27B. Official model card, 2026b. URL https://huggingface.co/ Qwen/Qwen3.8-27B. Accessed 2026-09-24.

Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, and Kate Saenko. SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models. arXiv preprint arXiv:2412.07755, 2024. doi: 10.48550/arXiv.2412.07755. URL https://arxiv.org/ abs/2412.07755.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024. doi: 10.48550/arXiv.2402.03300. URL https://arxiv.org/abs/2402.03300.

Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics. arXiv preprint arXiv:2411.16537, 2024. doi: 10.48550/arXiv.2411.16537. URL https: //arxiv.org/abs/2411.16537.

SpatialAxiom. SpatialAxiom-9B: Official Model Card and Benchmark Results. Official model card, 2026. URL https://huggingface.co/SpatialAxiom/SpatialAxiom-9B. Accessed 2026-09-24.

TaichuAI. ZDTaichu5.0-9B: Official Model Card and Benchmark Results. Official model card, 2026. URL https://huggingface.co/TaichuAI/ZDTaichu5.0-9B. Accessed 2026-09-24.

Chongyu Wang, Ting Huang, Chunyu Sun, Xinyu Ning, Di Wang, and Hao Tang. Let Geometry GUIDE: Layer-wise Unrolling of Geometric Priors in Multimodal LLMs. arXiv preprint arXiv:2604.05695v1, 2026. doi: 10.48550/arXiv.2604.05695. URL https://arxiv.org/ abs/2604.05695v1.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen

Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv preprint arXiv:2508.18265, 2025a. doi: 10.48550/arXiv.2508.18265. URL https://arxiv.org/abs/2508.18265.

Yifei Wang, Zhenkai Li, Tianwen Qian, Huanran Zheng, Zheng Wang, Yuqian Fu, and Xiaoling Wang. StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios. arXiv preprint arXiv:2512.04451, 2025b. doi: 10.48550/arXiv.2512.04451. URL https://arxiv. org/abs/2512.04451.

Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Yao Lu, and Song Han. StreamingVLM: Real-Time Understanding for Infinite Video Streams. arXiv preprint arXiv:2510.09608, 2025. doi: 10.48550/arXiv.2510.09608. URL https://arxiv.org/abs/2510.09608.

Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024a. doi: 10.48550/arXiv.2412.14171. URL https://arxiv. org/abs/2412.14171.

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, Daohan Lu, Rob Fergus, Yann LeCun, Li Fei-Fei, and Saining Xie. Cambrian-S: Towards Spatial Supersensing in Video. arXiv preprint arXiv:2511.04670, 2025a. doi: 10.48550/arXiv.2511.04670. URL https://arxiv.org/abs/2511.04670.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence. arXiv preprint arXiv:2505.23764, 2025b. doi: 10.48550/arXiv.2505.23764. URL https://arxiv.org/abs/2505.23764.

Yuncong Yang, Han Yang, Jiachen Zhou, Peihao Chen, Hongxin Zhang, Yilun Du, and Chuang Gan. 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning. arXiv preprint arXiv:2411.17735, 2024b. doi: 10.48550/arXiv.2411.17735. URL https://arxiv.org/ abs/2411.17735.

Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, Lingpeng Kong, Qi Liu, Yuanxing Zhang, and Xu Sun. TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos. arXiv preprint arXiv:2504.17343, 2025. doi: 10.48550/arXiv.2504.17343. URL https://arxiv.org/ abs/2504.17343.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. arXiv preprint arXiv:2308.11417, 2023. doi: 10.48550/ arXiv.2308.11417. URL https://arxiv.org/abs/2308.11417.

Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, and Bernadette Bucher. VLFM: Vision Language Frontier Maps for Zero-Shot Semantic Navigation. arXiv preprint arXiv:2312.03275, 2023. doi: 10.48550/arXiv.2312.03275. URL https://arxiv.org/abs/2312.03275.

Hanxun Yu, Xuan Qu, Lei Ke, Boqiang Zhang, Yuxin Wang, Jianke Zhu, and Dong Yu. Stream3D-VLM: Online 3D Spatial Understanding with Incremental Geometry Priors. arXiv preprint arXiv:2606.06891, 2026. doi: 10.48550/arXiv.2606.06891. URL https://arxiv.org/ abs/2606.06891.

Mingliang Zhai, Zhi Gao, Yuwei Wu, and Yunde Jia. Memory-Centric Embodied Question Answering. arXiv preprint arXiv:2505.13948, 2025. doi: 10.48550/arXiv.2505.13948. URL https://arxiv.org/abs/2505.13948.

Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams. arXiv preprint arXiv:2406.08085, 2024. doi: 10.48550/arXiv.2406.08085. URL https://arxiv.org/ abs/2406.08085.

Yiming Zhang, Jiacheng Chen, Jiaqi Tan, Yongsen Mao, Wenhu Chen, and Angel X. Chang. ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning. arXiv preprint arXiv:2604.24300, 2026. doi: 10.48550/arXiv.2604.24300. URL https://arxiv.org/abs/2604.24300.

## APPENDIX GUIDE

The linked guide below summarizes the supplementary material.

• Dataset Construction Details (Appendix A): observation alignment, evidence measurements, memory and QA targets, action trajectories, and question templates.

• Training Data Composition and Sampling (Appendix B): episode counts, training-data sources and composition, and action-length statistics.

• Memory Router Details (Appendix C): memory writes, category/time queries, pagination, and image retrieval.

• Training and Evaluation Details (Appendix D): optimization, action rewards and rollouts, QA scoring, and timing evaluation.

• Qualitative Examples (Appendix E): current-observation, historical-evidence, activeview, and memory-write examples.

## A DATASET CONSTRUCTION DETAILS

This section details how observations are converted into supervision for memory writing, spatial question answering, and camera actions. We first establish a shared temporal representation, then describe target construction and filtering, and finally specify the action space and multi-step trajectories. The question catalog provides examples and construction rules for individual templates.

## A.1 OBSERVATION PREPARATION AND ALIGNMENT

Simulation episodes use existing 3D-FRONT (Fu et al., 2021) layouts and assets. We remove objects with conspicuous interpenetration or erroneous scale while preserving the remaining scene structure. Scanning trajectories connect geometric centers and selected centers of maximal inscribed circles in randomized order, with the camera oriented toward the room interior. A\* (Hart et al., 1968) plans paths on a mesh-derived occupancy map over the camera’s relevant height range, and Bezier curves´ smooth the resulting paths. Multiple trajectories provide different visitation orders and observation histories for each layout. Floor and ceiling textures and randomly selected HDRI environments vary appearance, and Cycles (Blender Foundation, 2026) renders the RGB observations and auxiliary visibility annotations.

Real observations originate from ScanNet (Dai et al., 2017), ScanNet++ (Yeshwanth et al., 2023), and ARKitScenes (Baruch et al., 2021) videos. Frames are aligned with camera poses and source 3D annotations through physical timestamps. Coordinate transformations and projection associate annotated objects with individual observations. Both branches provide time-ordered RGB frames, category labels, instance correspondences, geometry, and per-frame visibility. We sample observations at 1 Hz and express geometric distances in meters. Per-frame visibility follows the source annotations. Pixel-level measurements are used where rendered masks are available; projected bounding-box area is separately identified when used as an evidence proxy.

For a question at time t, historical evidence is selected from the observation prefix $O _ { \leq t }$ . An actiongenerated observation becomes available only after the corresponding command is executed. Offline geometry and instance identifiers determine supervision, whereas the model receives RGB observations, timestamps, retained context, and returned tool content. Each example records its question time, evidence-frame references, answer, and construction provenance.

## A.2 METADATA AND EVIDENCE MEASUREMENTS

Temporal metadata maps frame references to timestamps. Camera metadata contains intrinsics and poses; object metadata supplies persistent identities, categories, 3D centers, dimensions, and available meshes or boxes. Scene metadata supplies geometry, occupancy, and floor polygons. Visibility metadata links these records to individual observations. Distances use meters, image fractions are dimensionless, and each evidence measurement records its provenance.

Visibility and image occupancy. Let I be the image rectangle, $N = W H$ its pixel count, $b _ { o } ( x )$ the source visibility indicator, and $n _ { o } ( x )$ the object’s visible mask-pixel count. Pixel visibility is $p _ { o } ( x ) = n _ { o } ( x ) / N$ . For observations using projected-box evidence, let $R _ { o } ( x )$ be the image-plane bounding rectangle of the projected world-axis-aligned 3D box. The occupancy proxy is

$$
u _ { o } ( x ) = b _ { o } ( x ) \frac { \mathrm { a r e a } ( R _ { o } ( x ) \cap I ) } { W H } .\tag{4}
$$

The indicator gates the proxy, but box area does not measure the unoccluded object surface. Missing source visibility is distinct from an explicit negative observation.

Framing completeness. Transform the corners of the mesh-local bounding box to the world and camera frames, project them using camera intrinsics, and form their convex hull $H _ { o } ( x )$ . Framing completeness is

$$
f _ { o } ( x ) = \frac { \mathrm { a r e a } ( H _ { o } ( x ) \cap I ) } { \mathrm { a r e a } ( H _ { o } ( x ) ) } .\tag{5}
$$

A valid ratio requires positive projected area and a box entirely beyond the 0.01 m near plane. Near-plane intersections are retained only as diagnostics, not valid framing scores; fully behindcamera and degenerate projections are unavailable. Framing describes image-boundary clipping, not occlusion by other objects.

Occlusion support. An isolated-object render uses the same camera and target geometry while hiding other scene objects. If its target mask contains $n _ { o } ^ { \mathrm { i s o } } ( x ) > 0$ pixels, define

$$
v _ { o } ^ { \mathrm { o c c } } ( x ) = \mathrm { m i n } \{ 1 , n _ { o } ( x ) / n _ { o } ^ { \mathrm { i s o } } ( x ) \} .\tag{6}
$$

The ratio is unavailable when the isolated mask is missing or empty, or the unbounded ratio exceeds 1.01. Values up to 1.01 are capped at one to accommodate small rasterization differences. This measures visible support relative to the isolated in-frame silhouette; framing is checked separately. Required unavailable measurements fail the corresponding quality check rather than being replaced with zeros.

## A.3 INCREMENTAL MEMORY TARGETS

Memory targets are generated over consecutive five-second windows. Each target contains one observation record per sampled frame, including frames with no visible objects. A record specifies the frame reference and visible category counts. Optional relations describe image-plane left, right, above, or below relationships, with subjects and objects localized on a $3 \times 3$ image grid. The final incomplete window is also retained.

Let $V _ { f , c }$ denote the set of visible instances of category c in frame $f .$ The per-frame count is $| V _ { f , c } |$ whereas the historical distinct-instance count is

$$
N _ { c } ( t ) = \left| \bigcup _ { f : t _ { f } \leq t } V _ { f , c } \right| .\tag{7}
$$

The latter is computed using offline instance correspondences. For example, observing the same two chairs in three frames yields per-frame counts of two and a historical count of two. Memory targets preserve frame references so that detailed appearance and instance evidence can be recovered from historical images.

## A.4 SPATIAL QA CONSTRUCTION

Each QA example is constructed by selecting a causal observation prefix, binding its referents, computing an answer from aligned annotations, checking evidence and ambiguity conditions, and instantiating a question template. Multiple-choice questions use distinct semantic alternatives and retain the mapping between option letters and answers. Numeric questions specify their units. All QA text undergoes one LLM polishing pass to diversify phrasing while preserving referents, temporal constraints, units, option meanings, and annotation-derived answers.

Current QA. Current questions use the latest observation or an explicitly specified recent window, with a default duration of five seconds. Camera-to-object and object-to-object distances use 3D centers. Trajectory length sums consecutive camera translations, endpoint displacement measures the distance between the initial and final positions, and orientation change measures the net endpoint rotation. Referents must be uniquely identifiable under the question wording. Nearest-camera queries require a minimum 0.15 m separation between the two closest candidates; designated-object nearestneighbor queries require 0.40 m. Appearance and disappearance questions require a unique category change between the specified observations.

Memory QA. Memory questions require evidence preceding the current window. First and last appearances are computed over the full causal prefix, historical counts use instance-set unions, and historical relationships are evaluated at the referenced frame. Appearance-order questions require successive first-observation times to differ by at least two seconds. Historical-neighbor questions bind a uniquely observed target that is absent from the current view and select neighbors among ob jects co-visible at the specified historical time. These evidence conditions define the question family; the model can answer directly when its retained context already contains the required information.

Chained questions and geometric examples. For linked questions, the first turn establishes an object or category and subsequent turns retain that referent. These linked QA turns are distinct from multiple camera actions taken to answer a single question. Each turn is associated with its own arrival time and evidence prefix. Height questions compare vertical extent rather than elevation above the floor, with a 0.30 m margin between the two tallest candidates. Room-area questions use the room’s floor polygon, including floor beneath furniture, and require at least eight observations, 90% boundary-angular coverage, and a viewing-direction span of at least 150<sup>◦</sup>.

A construction example. Consider a nearest-object question with visible object-center distances of 1.0, 1.4, and 2.1 m. The first object supplies the answer because its 0.4 m lead exceeds the 0.15 m ambiguity margin. Replacing the second distance with 1.08 m rejects the example. For a first-appearance question, the answer is the earliest visible frame in the complete prefix, and the example is assigned to Memory QA when that evidence precedes the current window.

## A.5 SELECTING QUESTION TIMES AND OBJECT REFERENTS

Construction operates on aligned frame records rather than RGB alone. Each record links a timestamp and image to its camera pose, visible-instance set, and category labels. Object records sup ply persistent instance identifiers, 3D centers, and the geometric fields required by each template; image-plane relations additionally use projected positions, and room-area questions use floor polygons. These annotations select candidates and compute labels offline. Public questions identify objects through category names, explicit nearest-instance qualifiers, or image regions, without exposing instance identifiers or geometric answers.

Candidate times and evidence windows. We enumerate candidate times on the sampled observation sequence and apply template-specific temporal conditions before constructing a question. Single-view questions use the question frame; motion and visibility-change questions require the specified earlier endpoint. Nearest-neighbor, fixed-reference relation, and height-comparison templates enumerate interior frames. Last-seen, historical coappearance, historical direction, and appearance-order templates are proposed every five sampled frames starting at index ten. Memory evidence must belong to the available prefix and be covered by completed memory writes. Roomarea candidates are proposed when a continuous within-room observation run meets the coverage conditions. A frame that does not satisfy a template’s conditions produces no example for that template.

Object eligibility and combinations. For nearest-neighbor, fixed-reference relation, and heightcomparison templates, candidate objects are visible at the question frame and its two adjacent frames, and each selected category has exactly one visible instance at the question frame. The adjacent-frame test selects stable observations offline; the following frame is not included in the question’s model input or used to compute its geometric answer. Category-conflict checks remove semantically overlapping options. We then enumerate reference–target pairs for relations and nearest-neighbor questions, or four distinct-category objects for height comparison. Missing geometric fields remove the affected combination. Other templates retain their own bindings: a nearest-instance qualifier can resolve multiple objects of one category, whereas historical singleinstance questions require a unique previously observed instance.

Answer computation and acceptance. For a bound combination, we first compute the geometric or temporal quantity, then check the template’s uniqueness and evidence conditions. A designatedobject nearest-neighbor question requires a 0.40 m gap between its two closest eligible neighbors. Height comparison uses world-vertical bounding-box extent and requires a 0.30 m gap between the two tallest objects. Fixed-reference left/right questions use the 3 × 3 image grid: the correct target lies in the reference object’s row and on the requested side; distractors come from the opposite side or categories absent from the scene. The category alternatives are filtered before option letters are assigned. These conditions apply to the corresponding templates, rather than serving as universal thresholds for all QA.

Historical selection and memory verification. The last-seen template selects a category absent at the question frame whose latest occurrence is at least five frame indices earlier. That historical frame also supplies candidates for coappearance and relative-direction questions, with its timestamp stated in the question. Coappearance distractors are absent from the referenced frame. Direction questions require unique category instances and a same-row or same-column grid relation. Firstappearance-order questions select three categories whose first sightings occur at least two seconds apart, with the latest first sighting at least five frame indices before the question. We replay the corresponding teacher memory writes and retrieval requests, reconstruct the answer from returned records or relations, and require agreement with the annotation-derived label. Direction examples additionally require the referenced image to be retrievable within the image budget.

Room evidence and retained records. Room-area construction uses simulation episodes with a valid floor polygon and unambiguous camera-to-room assignment. Leaving the assigned room resets the observation run. Each polygon edge contributes eight boundary samples; coverage counts samples falling within the horizontal viewing angle of at least one observation in the run. We require eight observations, 90% angular boundary coverage, and a 150<sup>◦</sup> heading span. This is a geometric coverage proxy rather than a wall-pixel visibility test. Each retained QA records its template, question time, evidence-frame references, bound objects where applicable, answer, and source-scene provenance. Missing evidence images and failed template checks are excluded before serialization.

Worked current and memory examples. Consider an illustrative current nearest-neighbor example at t = 12 s. A cabinet, bed, chair, and lamp have unique category referents and pass the adjacent-frame visibility check. Their center distances from the cabinet are 1.0, 1.6, and 2.2 m. The bed is retained as the answer because the nearest-neighbor margin is 0.6 m; the question refers to the cabinet, and its evidence is the frame at 12 s. If the chair were 1.2 m away, the combination would fail the 0.40 m margin. For an illustrative memory example at t = 20 s, a lamp was last visible at 8 s and is absent afterward. Its last-seen question has answer 8 s, with the frame at 8 s as positive evidence and the intervening prefix establishing that no later sighting occurred. The teacher write covering that frame is replayed, and a lamp-category query must recover the same latest timestamp. A later lamp sighting would change the label rather than preserve the earlier answer.

## A.6 ACTION-CONDITIONED DATA

Action data are constructed through dedicated rendering of one- to three-step discrete camera-action sequences. Each example fixes the starting observation, question, target objects, and geometryderived answer. Every action is executed from the preceding camera state and produces a newly rendered observation under unchanged scene geometry, appearance, and imaging settings. Evidence and geometric checks select demonstration sequences; alternative action outcomes are rendered and recorded separately for counterfactual optimization.

Start times and target binding. At candidate question times in simulated observation streams, we select objects with available geometry, a resolvable initial-view referent, and an evidence-area score of at least0.0002. Referents use a unique category or a unique category within a $3 \times 3$ image region. Up to sixteen objects are retained in descending score order. Each object supplies a size candidate, and each unordered pair supplies center-distance and distance-range candidates. Size is the longest physical bounding-box side, using source physical dimensions when available or transformed mesh dimensions otherwise; pair distance is the Euclidean distance between fixed 3D centers. The same bound objects and answer are retained throughout the path.

Evidence measurements and endpoint selection. Action construction uses mask-derived visiblepixel fractions for both pre-question observations and newly rendered action views. Corresponding target masks are required to evaluate the historical baseline, ensuring that evidence gains compare the same measurement. Let $p _ { j } ( x ) , f _ { j } ( x )$ , and $v _ { j } ^ { \mathrm { o c c } } ( x )$ denote visible-pixel fraction, framing completeness, and occlusion support, respectively, as defined in Appendix A.2. For the required objects $\mathcal { I } _ { q } .$ , define

$$
P ( x , q ) = \operatorname* { m i n } _ { j \in \mathcal { I } _ { q } } p _ { j } ( x ) , \qquad B _ { t } ( q ) = \operatorname* { m a x } _ { x \in O _ { \leq t } } P ( x , q ) .\tag{8}
$$

The minimum requires joint visual support for pairwise questions, while the historical maximum prevents improvement from being measured only against a weak starting view. An endpoint x is retained only if

$$
p _ { j } ( x ) \ge 0 . 0 0 5 , \quad f _ { j } ( x ) \ge 0 . 8 0 , \quad v _ { j } ^ { \mathrm { o c c } } ( x ) \ge 0 . 7 0
$$

$$
\forall j \in { \mathcal { I } } _ { q } ,\tag{9}
$$

$$
P ( x , q ) \geq 1 . 5 B _ { t } ( q ) , \qquad P ( x , q ) - B _ { t } ( q ) \geq 0 . 0 0 2 .\tag{10}
$$

Required unavailable measurements fail the quality check. These conditions define a single selection profile for all rendered demonstrations.

A historical view is adequate when every required object satisfies the same visibility, framing, and occlusion thresholds in that view. We retain separate labels for examples lacking an adequate historical view and examples improving an already adequate view. The historical baseline includes all pre-question observations, regardless of whether they satisfy the adequacy thresholds.

Discrete action interface and multi-step interaction. The action interface consists of five commands: forward, leftward, and rightward translation by 0.5 m, and left and right rotation by $4 5 ^ { \circ }$ Translation follows the current camera’s horizontal coordinate frame and preserves orientation; rotation preserves position. The model selects a command, and the runtime assigns its fixed magnitude. Each returned observation is associated with the resulting camera state. Interactions allow at most three action attempts, each continuing from its predecessor state and producing a specifically rendered observation. Demonstrations terminate at the earliest endpoint satisfying the unified quality and gain criteria above. Intermediate views need not satisfy the endpoint thresholds or improve evidence monotonically, allowing preparatory actions.

Geometric checks and serialization. For each segment, a mesh-based check rejects a surface intersection along the straight camera-center connection. Positions sampled at intervals no greater than 0.025 m require at least 0.125 m surface clearance and support within 2 m below. The serialized example contains the causal prefix, unchanged question, ordered command–image pairs, and final answer; future trajectory images enter the model context only after their corresponding commands. Duplicate candidates sharing the source trajectory, bound target or pair, endpoint, and template are removed. Distance-range and proximity candidates share the same geometric quantity. Identical serialized sequences are deduplicated across these template labels, and the final release retains the distance-range label.

Rendered alternatives for counterfactual optimization. Selected action sequences provide SFT demonstrations. Counterfactual optimization additionally requires alternative outcomes for the five discrete commands at each evaluated state, rendered under the same scene and imaging settings. Demonstration endpoint thresholds do not filter these alternatives: valid actions with little or negative evidence improvement remain available for comparison, and physically invalid actions are recorded separately. Branch validation distinguishes missing rendered outcomes (coverage errors) from physically invalid actions, which preserve the observation and consume an attempt. RL rewards use the mask-derived measurements defined in Appendix D.2.

Worked three-step example. Consider a size question about a uniquely identified cabinet whose longest physical side is 1.2 m. Suppose its best historical visible-pixel fraction is 0.001, and three successive rendered views yield 0.002, 0.004, and 0.007. The first two fail the 0.005 visibility threshold. The third achieves a gain ratio of 7 and an absolute gain of 0.006; with framing completeness 0.85 and occlusion support 0.80, it satisfies all endpoint conditions. If all transitions pass geometric checks, we retain the causal prefix, fixed question, three discrete command–image pairs, and answer 1.2 m. A failed transition rejects this sequence; if an earlier view satisfies every condition, the demonstration ends there instead.

## A.7 ASSEMBLY AND QUALITY CONTROL

We assemble observations, memory targets, questions, and tool outcomes into causally ordered training sequences. The training release contains all retained QA examples; the main test and Timing sets are separate evaluation collections. Validation checks timestamps, frame references, referent consistency, answer computations, duplicate semantic questions, tool-image correspondence, and interaction budgets. Examples with missing required evidence, failed constraints, or excessive sequence length are excluded. Public model inputs and private annotation fields are stored separately.

## A.8 QUESTION CATALOG AND CONSTRUCTION EXAMPLES

Tables 5, 6, and 7 list the Current, Memory, and Action QA templates, respectively, with representative questions, semantic answers, and construction rules. Current templates use present or recent evidence; Memory templates require historical observations; Action templates specify questions supported by newly acquired views. For multiple-choice examples, we display the semantic answer rather than its option letter. Linked templates show the opening question and specify the subsequent turn in the construction rule.

Table 5: Current QA templates and examples.
<table><tr><td>Template</td><td>Example and construction</td></tr><tr><td>approached object category</td><td>Question: What is the category whose object had the largest decrease in camera-to-3D-center distance among objects visible both now and 5 seconds ago? Choose from the listed options. Answer: storage unit. Construction: Compare objects visible at both endpoints. Require a</td></tr><tr><td>became visible cat- egory</td><td>distance decrease of at least 0.20 m and a 0.10 m lead over the runner- up. Question: Select the option that identifies the category that became vis- ible between 5 seconds ago and now. Answer: plant. Construction: Take the difference between endpoint category sets and</td></tr><tr><td>camera bucket</td><td>require one newly visible category. motion Question: What is the range of the camera's straight-line endpoint dis- placement over the last 5 seconds? Choose from the listed options. Answer: almost stationary (at most 0.2 meters). Construction: Discretize straight-line camera endpoint displacement,</td></tr><tr><td>m</td><td>excluding ambiguous interval boundaries. camera path length Question: Estimate the total length of the camera's trajectory over the last 5 seconds. Answer in meters. Answer: 0.0. Construction: Sum Euclidean translations between consecutive camera</td></tr><tr><td>camera rotation deg</td><td>positions in the specified window. Question: Provide your numerical estimate of the net difference be- tween the camera's starting and ending orientations over the last 5 sec- onds. Answer in degrees. Answer: 120.0. Construction: Compute net rotation between endpoint camera orienta- tions, expressed in degrees.</td></tr><tr><td>camera to object distance bucket</td><td>Question: What is the camera-to-3D-center distance range for the clos- est visible sofa? Choose from the listed options. Answer: very near (at most 1.25 meters) Construction: Bind the nearest visible instance of the named category</td></tr><tr><td>camera to object distance m</td><td>and discretize its center distance to the camera. Question: Give a numerical value for the camera-to-3D-center distance to the closest visible sofa. Answer in meters. Answer: 1.06.</td></tr><tr><td>closer than before</td><td>Construction: Bind the nearest visible instance of the named category and compute its center distance to the camera. Question: Select the option that identifies the change in camera-to-3D- center distance to the table over the last 5 seconds, considering the object</td></tr><tr><td></td><td>visible at both times. Answer: much closer (decrease of at least 0.75 meters). Construction: Uniquely match the category referent across the two ob- servations and classify its camera-distance change.</td></tr><tr><td>closest category then distance</td><td>Question: Choose the answer corresponding to the category of the cur- rently visible object closest to the camera by 3D center distance. Answer: sofa. Construction: First identify the nearest visible category; then query the</td></tr><tr><td>closest category then relation</td><td>distance interval of the same object. Count both QA turns. Question: Determine the category of the currently visible object closest to the camera by 3D center distance and select the matching choice. Answer: sofa. Construction: First bind the nearest visible object; then identify its</td></tr><tr><td>closest neighbor category</td><td>nearest neighbor among currently co-visible objects. Question: Which option describes the listed category whose currently visible object has its 3D center closest to the cabinet? Answer: bed. Construction: Require visibility at the question frame and its adjacent</td></tr><tr><td>closest object cate- gory</td><td>frames for offline stability selection, and unique category referents; the nearest two candidates must differ by at least 0.40 m. Question: Choose the answer corresponding to the category of the cur- rently visible object closest to the camera by 3D center distance.</td></tr><tr><td>disappeared cate-</td><td>Answer: sofa. Construction: Rank currently visible objects by camera-to-center dis- tance; require a nearest-candidate margin of 0.15 m. Question: From the options below, indicate the category that disap- peared from view between 5 seconds ago and now.</td></tr><tr><td>fixed reference rela-</td><td>Answer: sofa. Construction: Take the reverse endpoint category-set difference and require one disappeared category. Question: From the options below, indicate the listed category whose currently visible object is to the left of the desk in the image at the time</td></tr><tr><td></td><td>of this question. Answer: double bed. Construction: Use projected image positions with unique referents; se- lect a same-row left/right relation and place distractors on the opposite</td></tr><tr><td>multi height</td><td>side. candidate Question: Which option describes the listed category whose currently visible object has the greatest vertical height (not elevation above the floor)?</td></tr><tr><td>object category then later visibility</td><td>Question: Determine the category of the currently visible object closest to the camera by 3D center distance and select the matching choice. Answer: media unit. Construction: Bind a category/object in the first turn and ask about its</td></tr><tr><td>object relative to camera</td><td>visibility only after the later observation arrives. Question: Which option describes the direction of the closest visible bed relative to the camera now? Answer: far to the right.</td></tr><tr><td>room floor area</td><td>Construction: Transform the bound object center into the current cam- era frame and assign the corresponding direction category. Question: Calculate the total floor-footprint area of the room just sur- veyed that the camera is currently inside, including floor beneath furni- ture from the observations. Answer in square meters. Answer: 8.43. Construction: Compute the valid room floor-polygon area. Require at least eight frames, 90 percent boundary-angular coverage, and a 150-</td></tr></table>

Table 6: Memory QA templates and examples.
<table><tr><td>Template</td><td>Example and construction</td></tr><tr><td>first appearance or- der</td><td>Question: Identify the first-observation order of the categories arm- chair, dining table, storage unit since this episode began by selecting an option. Answer: storage unit → dining table → armchair.</td></tr><tr><td>first seen category then timestamp</td><td>at least two seconds and preceding the current window. Question: Determine the category seen earliest among the listed choices so far in this episode and select the matching choice. Answer: lighting. Construction: Identify the earliest category among the options, then</td></tr><tr><td>first seen object cat- egory</td><td>query its original first-observation timestamp. Question: Among the choices, which one gives the category seen earli- est among the listed choices so far in this episode? Answer: lighting.</td></tr><tr><td>first seen timestamp S</td><td>require a unique earliest option before the current window. Question: Estimate the first-seen timestamp of the lighting category, measured from the start of this episode. Answer in seconds. Answer: 0.0. Construction: Return the earliest observed timestamp for the category</td></tr><tr><td>historical coappear- ance category</td><td>over the complete causal prefix. Question: Choose the answer corresponding to the listed category also visible alongside the lighting at timestamp 3 seconds. Answer: cabinet.</td></tr><tr><td>historical relative direction</td><td>rect category is co-visible with the anchor and distractors are absent. Question: From the options below, indicate the image-plane position of the storage unit relative to the table in the observation at timestamp 44 seconds. Answer: left.</td></tr><tr><td>last seen timestamp</td><td>Question: Report the last timestamp at which the lighting category was observed before this question as a number. Answer in seconds. Answer: 44.0. Construction: Take the last occurrence before the question; require cur-</td></tr><tr><td>seen count by cate- gory</td><td>rent absence and an occurrence preceding the pending write window. Question: Report the number of distinct table objects observed so far in this episode as a number. Answer with an integer count (objects). Answer: 4. Construction: Union visible instance IDs over the prefix; require his-</td></tr><tr><td>single seen object then closest cate- gory</td><td>torical write coverage and that the current view does not contain all counted instances. Question: Choose the answer corresponding to the category nearest to that lighting by 3D center distance, among the other objects visible at timestamp 0 seconds alongside the lighting seen earlier and absent now. Answer: table.</td></tr><tr><td>single seen ob- ject then closest distance</td><td>Construction: Bind the only previously observed instance of the target category, absent now; find its nearest co-visible historical neighbor. Question: What is the 3D center distance from that lighting to its near- est neighbor, among the other objects visible at timestamp 0 seconds alongside the lighting seen earlier and absent now? Answer in meters. Answer: 0.776. Construction: Use the same historical target binding and return its min- imum center distance to another object visible in the referenced frame.</td></tr></table>

Table 7: Action QA templates and examples.
<table><tr><td>Template</td><td>Example and construction</td></tr><tr><td>object size m</td><td>Question: Estimate the longest physical side length of the 3D bounding box of the drawer chest / corner cabinet visible in the initial view. An- swer in meters. Answer: 1.185. Construction: Use source physical dimensions or transformed mesh bounding-box dimensions to obtain the longest physical side in meters;</td></tr><tr><td>object to object dis- tance bucket</td><td>hold the bound object fixed across views. Question: Identify the 3D center-to-center distance range between the sofa in the middle left of the initial view and the table in the lower left of the initial view by selecting an option. Answer: near (1 to 1.75 meters).</td></tr><tr><td>object to object dis- tance m</td><td>Construction: Map the bound pair&#x27;s center distance to an unambiguous interval; require evidence for both objects. Question: Report the 3D center-to-center distance between the sofa in the lower left of the initial view and the chair visible in the initial view as a number. Answer in meters. Answer: 3.041.</td></tr></table>

## B TRAINING DATA COMPOSITION AND SAMPLING

We report source composition, supervision counts, and action-path statistics separately. A QA turn is one question–answer pair; a linked two-turn example contributes two QA turns. Memory writes count completed supervision windows, while camera-action counts refer to executed transitions. Unique questions, demonstration paths, and alternative transitions have separate denominators. Demonstration length counts executed camera actions.

Table 8: Episode composition. The simulation row covers Current and Memory QA; real-world rows count original recordings and unique retained segments.
<table><tr><td>Source</td><td>Layouts / recordings</td><td>Episodes / segments</td></tr><tr><td>3D-FRONT</td><td>4,823</td><td>14,534</td></tr><tr><td>ScanNet</td><td>1,412</td><td>1,511</td></tr><tr><td>ScanNet++</td><td>796</td><td>1,457</td></tr><tr><td>ARKitScenes</td><td>4,681</td><td>5,764</td></tr><tr><td>Real-world total</td><td>6,889</td><td>8,732</td></tr></table>

Table 9: Training QA composition by family. Counts denote answer turns.
<table><tr><td>Family</td><td>Training QA turns</td></tr><tr><td>Current</td><td>302,661</td></tr><tr><td>Memory</td><td>402,662</td></tr><tr><td>Action</td><td>294,677</td></tr><tr><td>Total</td><td>1,000,000</td></tr></table>

Sampling is organized hierarchically by QA family, semantic template, and source scene. Within each family, we balance templates over eligible examples and preserve scene diversity. Linked question groups are selected together, and memory-writing targets remain attached to their causal windows. Dataset composition and training exposure are reported separately: sampling an example multiple times does not increase the number of distinct QA examples.

The training release contains 1,000,000 QA answer turns: 302,661 Current QA, 402,662 Memory QA, and 294,677 Action QA. All of these turns are training data. MemAct-bench uses a separate 20,000-turn main QA test set and an additional 2,000-question Timing set. Training and evaluation sets are disjoint at the scene level. The 1,000,000 training QA turns are serialized into 342,868 streams covering 11,724 source-qualified scene identifiers. Retained sequences satisfy the 180,000- token budget; the longest contains 178,705 tokens. Table 9 breaks down training answer turns by QA family, while Table 10 separates their sources and shows that Action QA is simulation-only. Table 11 reports demonstration lengths; one-action paths are most common, with two- and threeaction paths providing multi-step supervision. Table 12 reports training-set fractions, rather than repeated exposure during optimization, and Table 13 gives the finer template-level counts, including linked answer turns.

## B.1 EPISODE COMPOSITION AND TEMPORAL SEGMENTATION

For 3D-FRONT, an episode is an observation sequence associated with a layout, camera trajectory, and appearance configuration. Current and Memory QA use 14,534 simulated episodes from 4,823 layouts. The Action subset contributes 12 additional layouts outside this set, giving 4,835 simulated layouts across the complete release. Action observations are not counted as independent episodes in Table 8.

Long real-world recordings are divided into shorter temporal segments. The final release uses 8,732 unique segments from 6,889 original recordings. A recording is counted when at least one segment is retained, without requiring its full duration to be used. All retained segments reported here belong to the training collection. These recording identities are distinct from episode IDs and serialized training streams. Together, 4,835 simulated layout IDs and 6,889 real recording IDs account for the 11,724 source-qualified scene identifiers reported above; recording identifiers do not establish that every recording depicts a different physical environment.

Table 10: Training QA composition by source.
<table><tr><td>Source</td><td>Current</td><td>Memory</td><td>Action</td></tr><tr><td>3D-FRONT</td><td>135,671</td><td>218,983</td><td>294,677</td></tr><tr><td>ScanNet</td><td>31,304</td><td>30,431</td><td>0</td></tr><tr><td>ScanNet++</td><td>19,160</td><td>29,201</td><td>0</td></tr><tr><td>ARKitScenes</td><td>116,526</td><td>124,047</td><td>0</td></tr></table>

Table 11: Training action demonstration lengths after selection.
<table><tr><td>Actions</td><td>Training QA turns</td></tr><tr><td>1</td><td>216,418</td></tr><tr><td>2</td><td>46,858</td></tr><tr><td>3</td><td>31,401</td></tr></table>

Table 13: Template counts in the training release. Linked templates count all retained answer turns.
<table><tr><td>Template Training QA turns</td></tr><tr><td>Current QA</td></tr><tr><td>approached object category 18,733</td></tr><tr><td>became visible category 18,733</td></tr><tr><td>camera motion bucket 18,733</td></tr><tr><td>camera path length m 18,733</td></tr><tr><td>camera rotation deg 18,733</td></tr><tr><td>camera to object distance bucket 18,733</td></tr><tr><td>camera to object distance m 18,733</td></tr><tr><td>closer than before 18,733</td></tr><tr><td>closest category then distance 18,732</td></tr><tr><td>closest category then relation 18,732</td></tr><tr><td>closest neighbor category 18,739</td></tr><tr><td>closest object category 18,732</td></tr><tr><td>disappeared category 18,732</td></tr><tr><td>fixed reference relation 1,618</td></tr><tr><td>multi candidate height 1,316</td></tr><tr><td>object category then later visibility 18,732</td></tr><tr><td>object relative to camera 18,732</td></tr><tr><td>room floor area 18,732</td></tr><tr><td>Memory QA</td></tr><tr><td>first appearance order 61,034</td></tr><tr><td>first seen category then timestamp 36,870</td></tr><tr><td>first seen object category 18,435</td></tr><tr><td>first seen timestamp s 18,435</td></tr><tr><td>historical coappearance category 60,843</td></tr><tr><td>historical relative direction 26,743</td></tr><tr><td>last seen timestamp 60,847</td></tr><tr><td>seen count by category 24,399</td></tr><tr><td>single seen object then closest category 47,528</td></tr><tr><td>single seen object then closest distance 47,528</td></tr><tr><td>Action QA object size m 128,469</td></tr><tr><td>object to object distance bucket 70,293</td></tr><tr><td>object to object distance m 95,915</td></tr></table>

Table 12: Training-set composition by QA family and source. Each fraction uses the 1,000,000 training QA turns as its denominator.
<table><tr><td>Group</td><td>QA turns</td><td>Fraction</td></tr><tr><td>Current</td><td>302,661</td><td>30.27%</td></tr><tr><td>Memory</td><td>402,662</td><td>40.27%</td></tr><tr><td>Action</td><td>294,677</td><td>29.47%</td></tr><tr><td>ARKitScenes</td><td>240,573</td><td>24.06%</td></tr><tr><td>ScanNet</td><td>61,735</td><td>6.17%</td></tr><tr><td>ScanNet++</td><td>48,361</td><td>4.84%</td></tr><tr><td>3D-FRONT</td><td>649,331</td><td>64.93%</td></tr></table>

## C MEMORY ROUTER DETAILS

The memory router maintains model-written records and retrieves historical evidence through deterministic rules. Its responsibilities are structural validation, persistent storage, category/time filtering, and budgeted image return. The shared model supplies the semantic content of memory writes and formulates retrieval requests.

## C.1 WRITING AND STORAGE

A memory write uses the fields type, observations, and relations. Observation entries contain a frame reference and a list of category–count pairs. The router requires contiguous halfopen windows $[ t _ { s } , t _ { e } )$ , exactly one record per supplied frame, normalized lowercase category strings, positive integer counts, and valid relation endpoints. Empty object lists represent frames without visible objects. A valid write is stored with its frame/time mapping and a payload receipt. Repeating an identical write returns already stored; a different payload for the same window is rejected as a conflicting retry. Semantic predictions are stored as generated by the model.

## C.2 QUERIES, PAGINATION, AND IMAGES

Each question allows at most four memory-retrieval calls. A retrieval request specifies categories, an optional time range, and whether images are required. Category matching returns the union of matching records, with an additional co-visibility indicator for records containing all requested categories; an empty category list selects all categories. Results are ordered by time and paginated at eight records per page. A continuation cursor is bound to the original query and its record snapshot.

When images are requested, the router selects one representative per contiguous matching run, preferring co-visible records and the timestamp nearest their midpoint. Additional candidates are ordered by the summed counts of requested categories. Each call returns at most two images, with at most four distinct historical images per question. Returned images remain in context and are not resent. New questions reset pagination and image budgets; textual frame references remain available after the image budget is exhausted. These limits concern retrieved history, not action observations or the entire context.

Relations contain a frame, a predicate, and subject/object endpoints specified by category and imagegrid region. Query time ranges are half-open. Category filtering selects frame records, and the response includes all relations attached to frames in the returned text page or image list. It does not independently filter relation endpoints by query category.

Illustrative protocol. The following payload illustrates a single-frame write at t = 1 s. A complete write contains one observation entry for every supplied frame in its window. The chair query returns this frame’s category counts and the chair–table relation, together with the image if budget permits.

{"type":"memory\_write",   
"observations":[{"frame":"f000001","objects":[   
{"category":"chair","count":1},

```jsonl
{"category":"table","count":1}]}],
"relations":[{"frame":"f000001",
"subject":{"category":"chair","region":"middle_left"},
"predicate":"left_of_in_image",
"object":{"category":"table","region":"middle_right"}}]}
{"type":"use_memory","categories":["chair"],
"time_range":[0,5],"with_frames":true}
```

## D TRAINING AND EVALUATION DETAILS

Training first learns the shared output protocol through supervised fine-tuning and then optimizes action decisions using cumulative continuation returns. Both stages use the same vision–language model and structured outputs. This section specifies optimization, causal rollout construction, and answer scoring.

## D.1 SHARED CONFIGURATION AND SUPERVISED FINE-TUNING

We use Qwen3.5-4B and Qwen3.5-9B (Qwen Team, 2026a) with LoRA (Hu et al., 2021) applied to the query and value projections, using rank eight, scaling factor sixteen, and zero dropout. Basemodel parameters, including the visual encoder, are frozen. We use our dataset together with VSI-590K (Yang et al., 2025a) for SFT. Both stages use AdamW (Loshchilov & Hutter, 2017) with a learning rate of $1 0 ^ { - 5 }$ and gradient clipping at 1.0. Training uses eight GPUs, one training item per GPU per microbatch, and eight gradient-accumulation steps, giving 64 items per update. SFT items are serialized sequences; action-optimization items are decision histories whose candidate rollouts are evaluated internally. Each serialized training sequence is limited to 180,000 tokens, including text, visual tokens, and supervised assistant outputs. Sequences exceeding this budget are discarded without truncation or splitting.

SFT sequences preserve the order of observations, memory triggers, questions, and tool results. Only assistant outputs contribute to the token loss: memory records, retrieval requests, camera commands, and answers. Prompt and tool-result tokens provide context. The loss is averaged over supervised assistant tokens within each sequence and then over the effective batch. Linked questions retain their original referents and causal context.

## D.2 CUMULATIVE-RETURN ACTION OPTIMIZATION

Action optimization evaluates camera and stopping decisions using evidence increments and terminal answer quality. Construction thresholds select examples; the following scores define training rewards and common evidence metrics.

View-level evidence. For the one or two required objects ${ \mathcal { O } } _ { q } ,$ use the visible-pixel fraction $p _ { o } ( x )$ and framing completeness $f _ { o } ( x )$ defined in Appendix A.2. Define

$$
v _ { o } ( x ) = \mathrm { m i n } \{ p _ { o } ( x ) / 0 . 1 0 , 1 \} ,
$$

$$
s _ { o } ( x ) = v _ { o } ( x ) f _ { o } ( x ) ,\tag{11}
$$

$$
E _ { \mathrm { p i x e l } } ( x , q ) = \frac { 1 } { | \mathcal { O } _ { q } | } \sum _ { o \in \mathcal { O } _ { q } } v _ { o } ( x ) ,\tag{12}
$$

$$
E _ { \mathrm { f r a m e } } ( x , q ) = \frac { 1 } { | \mathcal { O } _ { q } | } \sum _ { o \in \mathcal { O } _ { q } } s _ { o } ( x ) ,\tag{13}
$$

$$
E _ { \mathrm { j o i n t } } ( x , q ) = \operatorname* { m i n } _ { o \in { \mathcal O } _ { q } } s _ { o } ( x ) .\tag{14}
$$

Normalization saturates at ten percent image occupancy. Framing discounts cropped targets, and joint evidence is limited by the less-supported object. For single-object questions, framing and joint scores coincide. These reward inputs use mask-derived pixel fractions, not projected-box occupancy.

Historical gain and cumulative returns. Let k index camera-action decisions from zero and T denote the total number of attempted camera actions before answering; write $[ z ] _ { + } = \operatorname* { m a x } ( z , 0 )$

The full configuration maintains $\Phi ( h , q ) = \mathrm { m a x } _ { x \in \mathcal { O } ( h ) } E _ { \mathrm { j o i n t } } ( x , q )$ over pre-question observations and preceding action results, with increment $d _ { k } = \Phi ( h _ { k + 1 } , q ) - \Phi ( h _ { k } , q )$ . The first three ablation configurations instead use $d _ { k } = [ E ( x _ { k + 1 } , q ) - E ( x _ { k } , q ) ]$ <sub>+</sub> with their respective view score. Local increments can reward recovery of previously acquired evidence; historical gain removes this repeated reward.

$$
r _ { k } = 0 . 1 0 d _ { k } - 0 . 0 3 - 0 . 2 0 { \bf 1 } [ \mathrm { i n v a l i d } ( a _ { k } ) ] , \qquad G _ { k } = \sum _ { j = k } ^ { T - 1 } r _ { j } + R _ { \mathrm { Q A } } .\tag{15}
$$

We use unit action costs, discount $\gamma = 1$ , and at most three camera-action attempts. Invalid actions preserve the observation, receive no evidence gain, and consume an attempt. Answer-now terminates without camera cost; budget exhaustion requests a final answer. $R _ { \mathrm { Q A } }$ scores the model’s generated answer by exact option correctness or numerical MRA, as defined in Appendix D.3. In the full configuration, evidence increments telescope to $0 . 1 0 [ \Phi ( h _ { T } , q ) - \Phi ( h _ { 0 } , q ) ]$ ], while action costs favor shorter paths with equal final evidence and answer quality.

Counterfactual candidate values. On rendered branch states, candidates are the five fixedmagnitude commands and answer now. All candidates start from an independent copy of the same history and remaining total budget. We average $B = 4$ continuations per candidate:

$$
\widehat { Q } ^ { \mu } ( h , a ) = \frac { 1 } { 4 } \sum _ { b = 1 } ^ { 4 } G ^ { ( b ) } ( h , a ) .\tag{16}
$$

A frozen snapshot of the current policy is refreshed before each update. Subsequent camera decisions are sampled from its categorical policy at temperature one; terminal answers use greedy decoding. Training includes initial histories and visited nonterminal histories. No new memory retrieval is introduced inside these rollouts.

Candidate scores sum conditional log probabilities of decision-value tokens and are normalized across candidates without length normalization. The action loss is

$$
\mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } _ { h } \left[ - \sum _ { a } \pi _ { \theta } ( a \mid h ) \widehat { Q } ^ { \mu } ( h , a ) + 0 . 0 5 D _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot \mid h ) \| \pi _ { \mathrm { S F T } } ( \cdot \mid h ) ) \right] .\tag{17}
$$

Values are detached during updates and retain their original scale. The loss sums over all candidate actions using the four-rollout value estimates. The SFT reference remains frozen. Private scores and alternative outcomes never enter the decision prefix.

Supervised replay. Training items are sampled with probabilities $\rho _ { \mathrm { a c t i o n } } = 0 . 5 0 , \rho _ { \mathrm { Q A } } = 0 . 2 5 .$ and $\rho _ { \mathrm { m e m } } = 0 . 2 5$ . An action item contributes the counterfactual loss for a decision history; QA and memory items contribute assistant-token-averaged SFT losses for answers/tool calls and incremental writes, respectively. Averaging item losses gives the mixture objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m i x } } = \mathbb { E } _ { u \sim \rho } \mathbb { E } _ { z \sim \mathcal { D } _ { u } } [ \ell _ { u } ( z ) ] . } \end{array}\tag{18}
$$

The sampling probabilities set item frequencies, and sampled item losses are averaged with equal weights. Rollout branches are evaluated within each action item.

Ablations and common metrics. Reward variants progressively add framing, joint aggregation, and historical gain to pixel visibility, retaining the same cumulative-return definition. Memorycomponent and reward variants are retrained under their respective configurations. GT memory replaces model-written records with the annotation-derived memory targets described in Appendix A. Action-budget comparisons use the full policy with limits of one, two, or three attempts. All reward variants are evaluated with the same full-history joint score:

$$
\Delta _ { i } = \Phi ( h _ { T _ { i } } , q _ { i } ) - \Phi ( h _ { 0 _ { i } } , q _ { i } ) ,\tag{19}
$$

$$
{ \mathrm { E v i d e n c e G a i n } } = { \frac { 1 0 0 } { N } } \sum _ { i } \Delta _ { i } ,\tag{20}
$$

$$
\mathrm { E f f e c t i v e G a i n R a t e } = \frac { 1 0 0 } { N } \sum _ { i } \mathbf { 1 } [ \Delta _ { i } > 0 . 1 0 ] .\tag{21}
$$

The initial history includes all observations available before acting. Both metrics include all questions, including zero-action answers. Mean steps counts attempted camera actions, including invalid attempts.

## D.3 STREAMING INPUTS AND MAIN QA

All streaming models process observations chronologically at 1 FPS. Offline models receive up to 32 uniformly sampled frames from the available prefix, using all available frames when fewer are present. Future frames are withheld. On the Action subset, comparison models can also issue actions under the same action interface and attempt budget, receiving the corresponding observations before answering. Multiple-choice answers are scored by exact option agreement. For nonzero references $y \ne 0 ,$ , numeric answers use mean relative accuracy (Yang et al., 2024a) over thresholds $\mathcal { T } = \{ 0 . 0 5 , 0 . 1 0 , \hdots , 0 . 5 0 \}$

$$
\mathrm { M R A } ( \hat { y } , y ) = \frac { 1 } { | T | } \sum _ { \tau \in T } \mathbf { 1 } \bigg [ \frac { | \hat { y } - y | } { | y | } \leq \tau \bigg ] .\tag{22}
$$

For zero-valued references, we use exact zero agreement. We report results by QA family, and separate one-, two-, and three-action budgets when evaluating active reasoning. All compared models share the same question times, referents, and available observation prefixes, represented according to their streaming or offline input protocols.

Offline QA prompts. Offline QA prompts present chronologically ordered frames with timestamps, followed by the question and its labeled options or numerical unit. When submitting a final answer, models return exactly one JSON object, {"type":"answer","answer":value}. For multiple-choice questions, value must be a single uppercase option letter rather than the option text. For numerical questions, it must be a JSON number in the unit specified by the question; object counts must be integers. Linked questions include preceding questions and the model’s own predictions as context.

Main QA aggregation and latency. The main evaluation contains 10,000 QA turns from each domain. Overall ACC pools the 13,134 multiple-choice turns, and Overall MRA pools the 6,866 numerical turns, weighting each task and domain by its applicable question count. Action contributes simulated questions only; no zero score is imputed for the real domain. Linked questions are scored per answer turn. Latency starts when the required images and context are ready and ends when the full answer has been produced. It excludes frame acquisition, ongoing observation maintenance, memory-tool operations, action execution or rendering, and network or queueing delays. The overall mean uses all 20,000 main QA turns, equivalent to averaging the two domain means with equal weights.

## D.4 TIMING EVALUATION

Timing contains 1,000 simulated and 1,000 real-world questions presented before their evidence appears. Frames arrive chronologically at 1 FPS, retaining observed history and withholding future frames. At each checkpoint the model outputs wait or answer; the first answer is final. Memory and action tools are disabled.

The first answerable time $t _ { i } ^ { * }$ is derived from visibility and evidence references: the unique earliest option-category appearance, the queried category’s first appearance, the earliest prefix leaving one consistent appearance-order option, or the arrival of the referenced historical frame for coappearance and direction questions. Tied first appearances are excluded, and derived answers must match ground truth. The model does not receive $t _ { i } ^ { * }$ ; no example is answerable at the initial frame.

For first-answer time $t _ { i } ^ { a }$ , let $z _ { i } = \mathbf { 1 } [ t _ { i } ^ { * } \leq t _ { i } ^ { a } \leq t _ { i } ^ { * } + 5 \mathrm { s } ]$ , or zero if unanswered. The inclusive window uses video time, not inference latency. An answer at the deadline is allowed; waiting at that

checkpoint is unanswered. We report

$$
T = \frac { 1 0 0 } { | \mathcal { Q } | } \sum _ { i \in \mathcal { Q } } z _ { i } ,\tag{23}
$$

$$
J A = \frac { 1 0 0 } { \left| \mathcal { Q } _ { c } \right| } \sum _ { i \in \mathcal { Q } _ { c } } z _ { i } \mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ] ,\tag{24}
$$

$$
J M = \frac { 1 0 0 } { \left| \mathcal { Q } _ { n } \right| } \sum _ { i \in \mathcal { Q } _ { n } } z _ { i } m _ { i } .\tag{25}
$$

Here $m _ { i } \in [ 0 , 1 ]$ is numerical MR $\mathrm { A } ; | \mathcal { Q } | = 2 , 0 0 0 , | \mathcal { Q } _ { c } | = 1 , 7 8 7$ , and $\left| \mathcal { Q } _ { n } \right| = 2 1 3$ . Denominators include all applicable questions, with early, late, and unanswered cases contributing zero. Timing is separate from Overall QA.

## E QUALITATIVE EXAMPLES

We provide visual examples of the evidence used by each QA family and of the incremental memory records that support historical retrieval.

Current observations and recent changes. Figure 5 illustrates approaching objects, newly visible categories, and camera displacement using observations five seconds apart. Figure 6 distinguishes trajectory length, camera rotation, and camera-to-object distance, while Figure 7 combines metric distance, changes in distance, and a linked category–distance query. Figure 8 shows closestobject identification, a follow-up relation query, and disappearance from view. Figure 9 illustrates tracking a selected referent across turns, camera-relative direction, and height comparison; Figure 10 shows nearest-neighbor and fixed-reference spatial relations. Together, these examples distinguish geometric reasoning in the current view from short-window temporal comparisons.

Historical evidence. Figure 11 illustrates retrieving the first observed category and its timestamp, and Figure 12 contrasts first-appearance lookup with counting distinct previously observed instances. Figure 13 uses an earlier observation to recover the nearest category or distance for a referent no longer visible. Figure 14 shows historical coappearance and image-plane direction; Figure 15 uses timestamps to recover the last observation and first-appearance order. Figure 16 provides further distinct-instance counting and historical nearest-neighbor examples, illustrating why the current view alone may be insufficient.

Acquiring additional views. Figure 17 shows turns that expose a target for size or proximity questions, and Figure 18 illustrates categorical and numerical inter-object distance queries after a turn. Figure 19 shows a turn and a lateral movement that provide additional views for inter-object distance questions. Figure 20 illustrates multi-step translations that increase target-pixel evidence relative to the best historical view before size estimation. These examples visualize the evidence acquired by actions; metric answers are derived from the scene annotations.

Incremental memory writes. Figure 21 shows five-second write windows in which objects leave view or new categories and counts appear. Figure 22 illustrates changing categories and a framelinked spatial relation. Both figures retain individual frame entries within each write window: perframe visible counts are not cumulative distinct-instance counts.

CURRENT Questions grounded in current observations and short temporal context  
![](images/484f79810bbd516bc631c16b1d96b2385ea72cd4329a299535dc1e34677051ff.jpg)  
Figure 5: Current QA sample 1

CURRENT Questions grounded in current observations and short temporal contex  
![](images/ee02268470d005b4a1182e90b6f9a279cd557476bf137991cdeb7e1ca23e7978.jpg)  
Figure 6: Current QA sample 2

![](images/63a7beb2031bd509601a24384f3c42b8a105d66ce94e357236fcb3bdce54b0ca.jpg)  
Figure 7: Current QA sample 3

CURRENT Questions grounded in current observations and short temporal context  
![](images/5c27db3ba03ab9c83cbc4858ed93bb990ecd25f16c415739f75523edf682102b.jpg)  
Figure 8: Current QA sample 4

![](images/999432dc3237cf8f0c621a4ba45fee0a49d6a57a6e78fcdabd51e87f1496ef2b.jpg)  
Figure 9: Current QA sample 5

CURRENT Questions grounded in current observations and short temporal context  
![](images/6416a75b4f4e3c993687507edcc7ae57d2b44ee5507cd4f257f306b00b27ff17.jpg)  
Figure 10: Current QA sample 6

MEMORY Questions grounded in recorded historical observations

![](images/759f83424fb9c6bf998fd9435e326084303474d59ff088d72cba487073ae6fab.jpg)  
Figure 11: Memory QA sample 1

MEMORY Questions grounded in recorded historical observations  
![](images/90b30acc32a7dba2ac711f67496e7a95887f83d13d35da68da9c8dfe684192a4.jpg)  
Figure 12: Memory QA sample 2

MEMORY Questions grounded in recorded historical observations  
![](images/096129e84ca4d33deb378a221dc5f94ec212d0df84aba4ac376cb9be939b94ee.jpg)  
Figure 13: Memory QA sample 3

MEMORY Questions grounded in recorded historical observations  
![](images/f29b3375f38886766315244fbbcca9f31466b8551717eeb8b2b973b5a90a5298.jpg)  
Figure 14: Memory QA sample 4

![](images/ba62cedfd7152d978d935c3d39d5ae809f43ce2318f559b4a1eab02501bbe78c.jpg)  
Figure 15: Memory QA sample 5

MEMORY Questions grounded in recorded historical observations  
![](images/647119ebbde318475313a3af293bc3027bbb41de25adaabb00c8d6fff7d09a00.jpg)

![](images/7672fdf6eb96bda66859e45f78bd8846020634c8fb9ac5f3f30a4e38f65cef89.jpg)  
Figure 16: Memory QA sample 6

ACTION Questions answered after acquiring additional visual evidence  
![](images/c7465bf575328ae0a0c9eb44a9735689c0d4238a1a00c312131eb9f226de436a.jpg)  
Figure 17: Action QA sample 1

ACTION Questions answered after acquiring additional visual evidence  
![](images/afdaae1cb1763244aa1e8653c1144638c893d0779d62d779f93db7f5b612d86a.jpg)  
Figure 18: Action QA sample 2

ACTION Questions answered after acquiring additional visual evidence  
![](images/c8cf214a63021bb0f74d2f5ae762c5b9857c350f7efddda7d6f750390b3ba681.jpg)  
Figure 19: Action QA sample 3

ACTION Questions answered after acquiring additional visual evidence  
![](images/9635b8528bb2845c199a3f0f1b25cd952ff5e1a438fb3a2c7e773830e25f599a.jpg)  
Figure 20: Action QA sample 4

STREAMING MEMORY WRITE Observe five frames at 1 FPS; write one structured record per 5-second window  
![](images/cad128ba018c4afefb3f1e36707402733405fd6862440d6b5b30a54975ab7757.jpg)  
Figure 21: Memory write sample 1

![](images/8cc663661353674daafe2c21a74fe3321963abbcbc70587aa4152cd55699745a.jpg)  
Figure 22: Memory write sample 2