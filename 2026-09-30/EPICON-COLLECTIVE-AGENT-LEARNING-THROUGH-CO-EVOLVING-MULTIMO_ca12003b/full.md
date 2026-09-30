# EPICON: COLLECTIVE AGENT LEARNING THROUGH CO-EVOLVING MULTIMODAL MEMORY

Ziyun Zeng<sup>1,2∗</sup> Hang Hua<sup>1†</sup> Shaden Alshammari<sup>1,3</sup> Rogerio Feris<sup>1</sup> William T. Freeman<sup>3</sup> Jiebo Luo<sup>2</sup>

<sup>1</sup> MIT-IBM Computing Research Lab <sup>2</sup> University of Rochester

<sup>3</sup> Massachusetts Institute of Technology

{zzeng24,jluo}@cs.rochester.edu, hang.hua1@ibm.com, {shaden,billf}@mit.edu, rsferis@us.ibm.com

## ABSTRACT

Agents can learn from past executions, but enabling different agents to reuse and build on one another’s experience remains challenging. We introduce EpiCon, a shared multimodal memory framework for agent collective learning without updating host model parameters. EpiCon links question-level memory evolution to a persistent experience bank through two independently trained 2B models: a memory controller and a tree self-organizer. The controller jointly refines textual guidance and visual evidence across attempts and selectively includes visual memory. The self-organizer consolidates lessons hierarchically and retrieves experience and rules for new problems. We evaluate EpiCon on eleven benchmarks spanning four multimodal task domains, using two harnesses and multiple backbones. A frozen bank improves other systems even with a single solving attempt. A second harness raises the original system’s macro-average score by 2.6 points across eleven benchmarks. Across four host configurations, EpiCon improves macro-average scores by 1.7 to 4.9 points over No Memory and reduces memory-operation time by 67% to 74% relative to backbone-sized memory models. Resources available at https://zzzmyyzeng.github.io/EpiCon.

## 1 INTRODUCTION

Multi-agent systems (MAS) built on language models combine reasoning, tool use, and collaboration to solve complex tasks (Wu et al., 2023; Fourney et al., 2024; Hua et al., 2024b; Zeng et al., 2026a; Lin et al., 2026). Their executions produce experience about solution procedures, failure modes, and relevant evidence. External memory allows this experience to inform subsequent inference without updating host model parameters (Wang et al., 2024b; Ouyang et al., 2026). As tasks accumulate, a system can both draw on earlier experience and contribute new lessons. This motivates a shared memory that supports continued learning within the same MAS while keeping experience useful across different harnesses and backbones.

Recent work has advanced agent memory through linked notes and hierarchical experience structures (Xu et al., 2026; Zhang et al., 2026b), while shared memory systems enable experience reuse across models and frameworks (Tang et al., 2025; Chang et al., 2026). Multimodal memory preserves visual evidence alongside textual guidance (Zeng et al., 2026c). Lessons may depend on diagram regions or document details, and both guidance and supporting evidence may need revision across attempts. Accumulated lessons also require consolidation for later retrieval and reuse. We study how to connect feedback-driven multimodal refinement with a shared experience bank that different agent systems can reuse and update.

We introduce EpiCon (Episodic Consolidation), a shared multimodal memory framework that supports agent collective learning through experience accumulation and reuse. Its dedicated memory harness connects temporary question-level memory with a persistent experience bank. Within a question, a Memory Controller supports textual and visual memory co-evolution, jointly revising actionable guidance and the associated image regions in response to successive attempts and feedback. As the guidance evolves, the controller can revise the visual evidence and adaptively decide whether to include visual memory in the next attempt. Across questions, a Tree Self-Organizer organizes and consolidates lessons in a shared bank maintained independently of the MAS. Accumulated experience can guide later solving within the same MAS and remain useful even if the harness or backbone changes. During bank construction and expansion, different harnesses can both reuse existing experience and contribute new lessons, allowing the evolved bank to support subsequent solving by its contributors. We implement memory control and tree organization with two independently trained 2B models.

We evaluate EpiCon on eleven benchmarks across four multimodal task domains. Experiments with two MAS harnesses and multiple backbones show improved task performance and reuse of experience across configurations. An evolved bank that incorporates experience from another harness also improves the original harness’s subsequent solving. The trained 2B models reduce memoryoperation time relative to backbone-sized memory models, and performance gains extend to Codex with GPT-5.6-Luna. Our contributions are summarized as follows:

• Collective agent learning through shared multimodal memory. We introduce EpiCon, which lets systems share experience without updating host parameters. Frozen-bank reuse improves performance across harnesses and backbones with a single attempt. Contributions from another harness raise the original system’s macro-average score by 2.1–3.2 points.

• Co-evolution of textual guidance and visual evidence. A memory controller jointly refines guidance and supporting image regions across attempts and selectively includes visual memory. A tree self-organizer consolidates lessons hierarchically and retrieves experience and rules for new problems.

• Efficient memory management with compact models. Both modules are independently trained 2B models. On eleven benchmarks spanning four multimodal domains, the 2B variant improves macro-average scores by 1.7–4.9 points over No Memory across four host configurations and reduces memory-operation time by 67–74% relative to backbone-sized memory models.

## 2 RELATED WORK

## 2.1 MULTI-AGENT SYSTEMS

Large language model (LLM) agents are increasingly organized into multi-agent systems (MAS) to divide complex tasks across specialized roles and coordinate complementary capabilities. Existing systems support collaboration through role-based dialogue and structured workflows (Li et al., 2023; Wu et al., 2023; Chen et al., 2024; Hong et al., 2024; Qian et al., 2024), multi-agent debate (Du et al., 2023), and adaptive interaction structures such as optimized agent graphs, orchestration, and team generation (Zhuge et al., 2024; Fourney et al., 2024; Yuan et al., 2025). These approaches make communication and coordination central to MAS design. However, long-horizon and repeated interactions introduce a further requirement: agents must retain useful observations, decisions, failures, and collaboration patterns beyond the current exchange. Communication determines how information moves among agents, whereas memory determines what remains available over time. Recent work therefore begins to model multi-agent memory explicitly, including agent-specific and cross trial collaboration histories (Zhang et al., 2026b). Our work treats memory as a persistent substrate through which multimodal experiences from different agents can be organized and reused.

## 2.2 AGENT MEMORY AND MULTIMODAL EXPERIENCE

Agent memory has evolved from retaining information to managing reusable experience. Early systems store and retrieve interaction histories or episodic records to support persistent behavior (Park et al., 2023; Packer et al., 2023; Zhong et al., 2024). Later work adds extraction, consolidation, hi erarchical organization, associative retrieval, dynamic linking, and updating (Chhikara et al., 2025; Xu et al., 2026; Gutierrez et al. ´ , 2024). Other approaches store trajectories, workflows, or learned skills as reusable experience (Zheng et al., 2024; Wang et al., 2023; 2024b; Zeng et al., 2026b; Hua et al., 2025b). Most language-agent memories remain primarily textual. Multimodal agents preserve perceptual evidence alongside higher-level knowledge for long-horizon reasoning (Li et al., 2024; Long et al., 2026; Zeng et al., 2026c; Hua et al., 2024a; 2025a). Beyond reasoning, multimodal systems have also been developed for visual generation(Yu et al., 2024; 2025; 2026), together with benchmarks and agentic evaluators for visual generation (Hua et al., 2025c; Zeng et al., 2026d). In MAS, agents may contribute different parts of the same experience, making memory sharing important (Zhang et al., 2026b). EpiCon jointly refines textual guidance and visual evidence across attempts and shares the resulting experience across systems.

![](images/da08a8609e52bf39e1250eecf15071c3d103b9d10edaf2b727c7e6bda9c9bb69.jpg)  
Figure 1: Overview of EpiCon. The memory controller refines question-level memory, while the tree self-organizer maintains accumulated lessons for subsequent retrieval and reuse. The memory harness coordinates these operations with MAS execution. Bank maintenance occurs during construction; the historical bank remains frozen during evaluation.

## 2.3 REFLECTION AND SELF-EVOLVING AGENTS

Reflection and self-improvement study how agents can turn interaction outcomes into better future behavior. Self-refinement and critique methods revise outputs using model-generated or external feedback (Madaan et al., 2023; Shinn et al., 2023; Gou et al., 2024), while reflection- and searchbased agents use execution outcomes to guide later attempts (Shinn et al., 2023; Zhou et al., 2023). Experiential-learning methods go further by abstracting trajectories into reusable lessons, workflows, policies, or skills (Zheng et al., 2024; Wang et al., 2023; 2024b; Li et al., 2025; Zhao et al., 2024; Zhang et al., 2024; Hu et al., 2023; Zheng et al., 2025). Recent work on self-evolving agents extends this process by integrating successful and failed experience into persistent reasoning memory or broader capability updates (Ouyang et al., 2026; Yan et al., 2026; Yang et al., 2026; Cheng et al., 2026). These works form a progression from retry, to reflection, to experience consolidation. Our work connects multimodal memory evolution within a question with the reuse of accumulated experience across questions. We study these two roles separately, evaluating historical experience reuse with a single solving attempt so that its benefits do not depend on additional attempts.

## 3 DATA CURATION

We construct two supervision datasets for memory control and tree organization using Qwen3.8- 27B (Qwen Team, 2026b) as the initial teacher and Qwen3.8-Flash-Next (Qwen Team, 2026a) for selective refinement. The initial teacher proposes joint text–visual memory updates, lessons, and tree-operation demonstrations. Qwen3.8-Flash-Next revises unsuccessful updates, improves lesson quality, and corrects tree-operation targets where needed.

Source datasets. We collect experience from tasks in MathNet (Alshammari et al., 2026), MathV360K (Shi et al., 2024), ChartQA (Masry et al., 2022), InfoVQA (Mathew et al., 2022), DocVQA (Mathew et al., 2021), and ChartNet (Kondic et al., 2026). Our training tasks cover three domains: visual mathematics, document understanding, and reasoning over charts and infographics.

![](images/e6870557f0eb7cd4cfaa90f18e22e4e70e809fd4698e7e3ec0041412865e4708.jpg)  
Figure 2: Illustrative example of EpiCon. (a) Textual and visual memory co-evolve across attempts. (b) and (c) Related lessons are organized and consolidated into a reusable routing rule. (d) Retrieved guidance supports a single solving attempt on a new map. Trajectories and tree operations are reconstructed for illustration.

MAS executions on these tasks provide the records we use to construct memory updates, lessons, and tree-operation supervision.

Memory controller. We collect MAS trajectories using CAMEL (Li et al., 2023), Codex (OpenAI, 2026a), and DeepSeek-Harness (DeepSeek-AI, 2026). Given the question, images, previous memory, latest attempt, and correctness feedback, the teachers generate and refine joint text–visual memory updates. We replay the previous and proposed memories with the same MAS host, retaining Repair transformations that turn failed attempts into successful ones and Compress transformations that preserve success while shortening text without increasing crop area. Reference answers are used to assess attempts during this offline screening. Accepted joint updates provide approximately 25K supervision examples.

Tree self-organizer. Lessons derived from the collected controller memories seed small memory banks, with historical lessons separated from retrieval queries. The teachers generate and selectively refine demonstrations of placement, merging, splitting, consolidation, and retrieval. We retain approximately 6.5K operation demonstrations that pass schema, node-reference, partition, and provenance checks, including valid empty retrievals when no candidate applies. These checks establish structural validity; individual tree operations are not validated through downstream replay. Appendix A provides construction and filtering details.

## 4 METHODOLOGY

EpiCon combines a dedicated memory harness with two independently trained 2B models: a Mem ory Controller $C _ { \theta }$ and a Tree Self-Organizer $S _ { \phi }$ (Figure 1). The controller updates textual and visual memory within a question, while the organizer maintains a shared experience bank for reuse across questions and agent systems. The harness coordinates memory updates, retrieval, visual cropping, and storage. Both models are trained on the supervision described in Section 3; the host MAS retains its own orchestration and backbone, whose parameters remain unchanged.

## 4.1 SHARED MULTIMODAL MEMORY

EpiCon maintains a persistent experience bank B independently of the MAS that uses it. A system can retrieve guidance from this bank and contribute lessons from its own task executions. These lessons support subsequent tasks within the same MAS and can also be reused across different

Table 1: Main results across two MAS harnesses, two backbones, and eleven benchmarks. All methods make a single solving attempt per question, without question-level memory updates. Both EpiCon variants use MC + Tree. Parentheses identify the memory models: ours 2B uses two independently trained 2B models; named backbones operate zero-shot. Time and token usage separate MAS execution from memory operations.
<table><tr><td rowspan="2">MAS Harness</td><td rowspan="2">MAS Backbone</td><td rowspan="2">Memory System</td><td colspan="3">Document Understanding</td><td colspan="3">Visual-to-Code</td><td colspan="3">Vision-Grounded Math</td><td colspan="2">General VL Reasoning</td><td rowspan="2">Time (×)</td><td colspan="3">Tokens (×)</td></tr><tr><td>DocVQA2026</td><td></td><td>MP-DocVQA ParseBench</td><td>Vision2Code</td><td>Omni-12C</td><td>ChartMimic</td><td>MATH-Vision</td><td>WeMath</td><td>WorldBench</td><td>ReasonMap</td><td>BabyVision</td><td></td><td></td><td></td></tr><tr><td rowspan="10">Codex</td><td>No Memory</td><td>15.7</td><td></td><td>79.5 81.6</td><td>57.1 61.9</td><td>53.0</td><td>66.2</td><td>66.4 20.4</td><td>81.0</td><td>49.2</td><td>40.1</td><td>14.2</td><td>1.00</td><td>0.00</td><td>Memory 1.00</td><td>Memory 0.00</td></tr><tr><td>Mem0 (Qwen3.8-27B)</td><td>1.4</td><td></td><td></td><td>61.3</td><td>70.6</td><td>78.0</td><td>24.4</td><td>80.6</td><td>48.6</td><td>39.6</td><td>12.2</td><td>1.33</td><td>0.31</td><td>1.23</td><td>0.11</td></tr><tr><td>Cognee (Qwen3.8-27B)</td><td>14.3</td><td>73.7</td><td>60.1</td><td>61.5</td><td>69.9</td><td>78.1</td><td>44.2</td><td>78.2</td><td>51.2</td><td>35.4</td><td>11.5</td><td>1.16</td><td>1.66</td><td>1.11</td><td>0.75</td></tr><tr><td>Qwen3.8-27B A-Mem (Qwen3.8-27B)</td><td>12.9</td><td>80.2</td><td>66.3</td><td>60.0</td><td>72.3</td><td>78.6</td><td>25.4</td><td>81.8</td><td>45.4</td><td>40.6</td><td>13.5</td><td>1.23</td><td>0.24</td><td>1.27</td><td>0.81</td></tr><tr><td>Agent-KB (Qwen3.8-27B)</td><td>11.4</td><td>78.6</td><td>59.5</td><td>25.8</td><td>67.4</td><td>69.8</td><td>25.2</td><td>80.8</td><td>45.0</td><td>24.1</td><td>15.6</td><td>1.54</td><td>0.31</td><td>1.12</td><td>0.14</td></tr><tr><td>EpiCon (ours 2B)</td><td>18.6</td><td>66.1</td><td>67.3</td><td>59.9</td><td>71.4</td><td>75.2</td><td>45.2</td><td>84.2</td><td>52.4</td><td>41.5</td><td>15.3</td><td>1.16</td><td>0.30</td><td>1.04</td><td>0.67</td></tr><tr><td>EpiCon (Qwen3.8-27B)</td><td>22.9</td><td>83.5</td><td>68.5</td><td>61.7</td><td>73.0</td><td>79.3</td><td>48.6</td><td>85.4</td><td>53.8</td><td>43.4</td><td>16.3</td><td>1.15</td><td>0.92</td><td>1.03</td><td>0.64</td></tr><tr><td>No Memory</td><td></td><td></td><td>46.2 43.3</td><td>51.7</td><td>65.9</td><td>65.6</td><td>25.8</td><td>85.0</td><td>45.4</td><td>31.6</td><td>17.7</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td></td><td>Mem0 (Gemma4-31B)</td><td>5.7 7.1</td><td>54.0</td><td>51.2</td><td>65.1</td><td>62.2</td><td>29.2</td><td>86.2</td><td>44.4</td><td>27.8</td><td>12.2</td><td>1.40</td><td>0.53</td><td>1.24</td><td>0.12</td></tr><tr><td></td><td>Cognee (Gemma4-31B)</td><td>4.3</td><td>49.9</td><td>33.4 27.6</td><td>54.6</td><td>49.0</td><td>63.7 37.4</td><td>72.4</td><td></td><td>42.2 27.4</td><td>12.8</td><td>0.92</td><td>7.23</td><td>1.07</td><td>1.02</td></tr><tr><td></td><td>Gemma4-31B A-Mem (Gemma4-31B)</td><td>8.6</td><td>55.5</td><td>37.0</td><td>54.9</td><td>68.6</td><td>63.1</td><td>29.8</td><td>88.0</td><td>44.6</td><td>31.6</td><td>13.9</td><td>1.23 0.39</td><td>1.31</td><td>0.97</td></tr><tr><td></td><td>Agent-KB (Gemma4-31B)</td><td>5.7</td><td>51.4</td><td>27.4</td><td>52.8</td><td>62.1</td><td>45.9</td><td>26.2</td><td>87.2</td><td>46.8</td><td>32.5</td><td>16.3</td><td>1.41 0.31</td><td>1.04</td><td>0.13</td></tr><tr><td>EpiCon (ours 2B)</td><td></td><td>8.6</td><td>58.8</td><td>44.2</td><td>54.3</td><td>67.5</td><td>66.7 38.8</td><td>87.8</td><td></td><td>53.2 32.1</td><td></td><td>18.1 1.33</td><td>0.47</td><td>1.01</td><td>0.74</td></tr><tr><td>EpiCon (Gemma4-31B) No Memory</td><td></td><td>8.6</td><td>59.6</td><td>45.8 57.1</td><td></td><td>67.8 66.4</td><td>57.6</td><td>88.4</td><td>54.2</td><td>36.3</td><td>19.1</td><td>1.20</td><td>1.56</td><td>1.02</td><td>0.67</td></tr><tr><td></td><td></td><td>14.3</td><td>79.5</td><td>52.4 46.1</td><td>53.7</td><td>61.4</td><td>65.3</td><td>42.2</td><td>88.8</td><td>54.0 22.6</td><td></td><td>20.1 1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td rowspan="8">Qwen3.8-27B Deepseek</td><td>Mem0 (Qwen3.8-27B)</td><td>11.4</td><td>80.5</td><td></td><td>54.9</td><td>61.0</td><td>68.8</td><td>38.8</td><td>90.0</td><td>52.2</td><td>21.2</td><td>17.0</td><td>1.10</td><td>0.15</td><td>1.87</td><td>0.41</td></tr><tr><td>Cognee (Qwen3.8-27B)</td><td>15.7</td><td>76.6</td><td>50.9</td><td>54.0</td><td>60.6</td><td>68.3</td><td>53.6</td><td>91.2</td><td>50.2</td><td>19.3</td><td>18.4</td><td>0.95</td><td>0.85</td><td>1.33</td><td>2.70</td></tr><tr><td>A-Mem (Qwen3.8-27B)</td><td>17.1</td><td>80.6</td><td>55.4</td><td>53.7</td><td>63.4</td><td>65.5</td><td>43.6</td><td>89.6</td><td>52.2</td><td>22.2</td><td>17.4</td><td>1.09</td><td>0.12</td><td>2.01</td><td>2.98</td></tr><tr><td>Agent-KB (Qwen3.8-27B)</td><td>20.0</td><td>79.5</td><td>50.5</td><td>52.5</td><td>62.6</td><td>63.8</td><td>42.6</td><td>90.4</td><td>52.6</td><td>13.7</td><td>20.5</td><td>1.13</td><td>0.15</td><td>1.51</td><td>0.64</td></tr><tr><td>EpiCon (ours 2B)</td><td>18.6</td><td>81.8</td><td>53.6</td><td>56.9</td><td>63.5</td><td>65.2</td><td>56.6</td><td>92.2</td><td>57.4</td><td>24.5</td><td>17.4</td><td>1.08</td><td>0.15</td><td>1.06</td><td>1.92</td></tr><tr><td>EpiCon (Qwen3.8-27B)</td><td>20.0</td><td>82.9</td><td>57.0</td><td>61.2</td><td>62.8</td><td>65.4</td><td>58.2</td><td>91.8</td><td>58.2</td><td>27.4</td><td>23.3</td><td>1.07</td><td>0.57</td><td>1.10</td><td>2.88</td></tr><tr><td>No Memory</td><td>10.0</td><td>2.7</td><td>35.6</td><td>51.4</td><td>55.4</td><td>32.8</td><td>16.0</td><td>91.0</td><td>49.6</td><td>27.4</td><td>11.5</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td>51.6</td><td>51.6</td><td>32.6</td><td>11.4</td><td>88.6</td><td>49.2</td><td>26.4</td><td>12.5</td><td>0.93</td><td>0.20</td><td>1.80</td><td>0.38</td></tr><tr><td rowspan="5">Gemma4-31B A-Mem (Gemma4-31B)</td><td>Mem0 (Gemma4-31B) Cognee (Gemma4-31B)</td><td>7.1 5.7</td><td>9.1 0.5</td><td>35.7 37.7</td><td>49.7</td><td>44.8</td><td>29.9</td><td>12.6</td><td>89.4</td><td>49.8</td><td>30.7</td><td>12.8</td><td>0.82</td><td>2.23</td><td>1.22</td><td>3.06</td></tr></table>

harnesses and backbones. Collective learning therefore takes place through the accumulation and reuse of shared experience, without updating host model parameters.

For a question $x _ { i }$ with its images and public task information, the organizer retrieves experience $R _ { i }$ once from B. The MAS uses the retrieved experience to produce an answer $\hat { y } _ { i , t }$ together with an execution trace $\tau _ { i , t }$ . When question-level evolution is enabled, the controller maintains a separate memory $W _ { i , t } ,$ initially empty, and revises it using the question, images, existing memory, answer, and execution trace. Our evaluations of historical experience reuse disable this loop and use a single solving attempt.

During bank construction or expansion, accepted lessons and their revisions are incorporated into B, allowing experience to accumulate across tasks. Let $\mathcal { U } _ { k }$ denote the batch of accepted lessons and revisions processed at maintenance step k:

$$
B ^ { ( k + 1 ) } = S _ { \phi } ( B ^ { ( k ) } , \mathcal { U } _ { k } ) .\tag{1}
$$

The index k counts maintenance steps, not individual lessons or questions, and a step can process multiple lessons together. The update represents the organizer’s proposals after validation and execution by the memory harness. During evaluation, the shared bank is frozen and updates remain local to the current question.

## 4.2 QUESTION-LEVEL MULTIMODAL MEMORY EVOLUTION

Textual and visual memory co-evolution. Question-level memory is represented as $W _ { i , t } ~ =$ $( M _ { i , t } ^ { \mathrm { t e x t } } , M _ { i , t } ^ { \mathrm { v i s } } )$ ). Text records actionable guidance, applicability conditions, and cautions; visual memory points to supporting regions in source images. When another attempt is available, the controller updates memory using the previous state, current question and images, latest answer, and execution trace:

$$
W _ { i , t } = C _ { \theta } ^ { \mathrm { u p d a t e } } ( W _ { i , t - 1 } , x _ { i } , \hat { y } _ { i , t } , \tau _ { i , t } ) .\tag{2}
$$

The equation denotes the accepted update after harness validation. The controller jointly proposes revised text, a source image, and a region. It can retain, replace, or remove visual evidence as the guidance evolves. The harness validates the proposal and generates the requested crop; invalid proposals leave memory unchanged. The accepted text and visual evidence guide the next attempt.

Adaptive visual memory injection. The controller decides whether visual memory should be included in the next input through $g _ { i , t } \in \{ 0 , 1 \}$ . For retrieved experience, the decision is based on the current question and descriptions of the retrieved memory. For question-level memory, the joint update supplies the decision together with the revised text and region. Let $T _ { i , t }$ and $V _ { i , t }$ denote the text and available visual memory for attempt t, drawn from retrieved experience or the latest question-level state. The memory input is

![](images/db60f2f9ed4c8b8150468cc7d0c1b8a57914262a5cb62cb294916842c6bdf473.jpg)  
Figure 3: Benchmark profiles and performance–time trade-offs with Qwen3.8-27B. (a) Scores are averaged across the two harnesses and normalized by the best method mean per benchmark; the radial axis starts at 0.25. (b) and (c) show scores averaged equally over eleven benchmarks; time sums normalized MAS and memory costs. Dashed lines connect No Memory and the two EpiCon variants as visual guides.

$$
c _ { i , t } = \left\{ \begin{array} { l l } { ( T _ { i , t } , \emptyset ) , } & { g _ { i , t } = 0 , } \\ { ( T _ { i , t } , V _ { i , t } ) , } & { g _ { i , t } = 1 . } \end{array} \right.\tag{3}
$$

Here $V _ { i , t }$ is bounded by the memory-image budget. The gate selects input content without deleting stored visual memory. The current question’s original images remain available in both cases.

## 4.3 SHARED EXPERIENCE CONSOLIDATION AND REUSE

Organizing accumulated lessons. The shared bank uses a tree to organize lessons with their guidance, source references, and visual evidence. Leaves store question-specific lessons, while internal nodes group and summarize related experience. During maintenance, the Tree Self-Organizer can propose Place, Merge, Split + Lift, and Consolidation operations to assign categories, combine related groups, separate broad groups, and summarize shared procedures and conditions. The harness validates and executes these proposals and can enforce capacity-based partitions.

Abstracting reusable rules. Consolidation captures common guidance and conflicts across lessons. The harness determines abstraction levels and rule eligibility from support, failures, conflicts, and evidence strength, while the organizer generates summaries. Internal nodes can retain qualified summaries or become reusable rules when the evidence permits. This organization allows later tasks to access specific experience and guidance consolidated from multiple lessons.

Retrieving experience for subsequent tasks. Retrieval first recalls candidates and then selects relevant experience:

$$
R _ { i } = S _ { \phi } ^ { \mathrm { r e t r i e v e } } ( x _ { i } , \mathrm { R e c a l l } ( x _ { i } , B ) ) .\tag{4}
$$

Here Recall uses a frozen text encoder to retrieve candidates from node indices describing their topic, scope, and question anchor. The organizer examines the current question and candidate memories, including textual guidance and associated visual evidence. It selects applicable rules, a concrete experience, or nothing when no candidate applies. The selected content is available to the receiving MAS regardless of which harness produced it, with visual delivery controlled by $C _ { \theta }$ Retrieval uses existing bank content; further construction or expansion can add lessons and revise its organization for subsequent use. Figure 2 illustrates how question-level memory co-evolution connects to experience consolidation and reuse across transit maps.

Table 2: Cross-backbone and cross-harness experience transfer from a bank built by Codex with Qwen3.8-27B. The evaluated MAS performs one solve; EpiCon (ours 2B) reuses the frozen bank without question-level updates.
<table><tr><td rowspan="2">MAS Harness</td><td rowspan="2">MAS Backbone</td><td rowspan="2">Memory System</td><td colspan="3">Document Understanding</td><td colspan="3">Visual-to-Code</td><td colspan="2">Vision-Grounded Math</td><td colspan="3">General VL Reasoning</td></tr><tr><td>DocVQA2026</td><td>MP-DocVQA</td><td>ParseBench</td><td>Vision2Code</td><td>Omni-I2C</td><td>ChartMimic</td><td>MATH-Vision</td><td>WeMath</td><td>WorldBench</td><td>ReasonMap</td><td>BabyVision</td></tr><tr><td>Codex</td><td>Gemma4-31B</td><td>No Memory</td><td>5.7</td><td>46.2</td><td>43.3</td><td>51.7</td><td>65.9</td><td>65.6</td><td>25.8</td><td>85.0</td><td>45.4</td><td>31.6</td><td>17.7</td></tr><tr><td rowspan="2">DeepSeek</td><td rowspan="2"></td><td>EpiCon (ours 2B)</td><td>5.7</td><td>60.4</td><td>45.3</td><td>53.8</td><td>67.8</td><td>68.2</td><td>33.0</td><td>85.0</td><td>47.8</td><td>34.4</td><td>18.1</td></tr><tr><td>No Memory</td><td>14.3</td><td>79.5</td><td>52.4</td><td>53.7</td><td>61.4</td><td>65.3</td><td>42.2</td><td>88.8</td><td>54.0</td><td>22.6</td><td>20.1</td></tr><tr><td>Harness</td><td>Qwen3.8-27B</td><td>EpiCon (ours 2B)</td><td>18.6</td><td>79.1</td><td>65.7</td><td>59.2</td><td>63.3</td><td>68.7</td><td>56.0</td><td>90.6</td><td>52.2</td><td>26.4</td><td>20.5</td></tr></table>

Table 3: Cross-harness memory evolution with EpiCon (ours 2B). Each evaluated harness builds the original bank; the other harness uses and evolves it. The two stages each use 1,000 disjoint construction questions. Both bank conditions use a single solving attempt.
<table><tr><td rowspan="2">MAS Harness</td><td rowspan="2">MAS Backbone</td><td rowspan="2">Memory Bank</td><td colspan="3">Document Understanding</td><td colspan="3">Visual-to-Code</td><td colspan="2">Vision-Grounded Math</td><td colspan="3">General VL Reasoning</td></tr><tr><td>DocVQA2026</td><td>MP-DocVQA</td><td>ParseBench</td><td>Vision2Code</td><td>Omni-I2C</td><td>ChartMimic</td><td>MATH-Vision</td><td>WeMath</td><td>WorldBench</td><td>ReasonMap</td><td>BabyVision</td></tr><tr><td>Codex</td><td>Qwen3.8-27B</td><td>Original bank</td><td>18.6</td><td>66.1</td><td>63.4</td><td>59.9</td><td>71.4</td><td>75.2</td><td>45.2</td><td>84.2</td><td>52.4</td><td>38.3</td><td>14.3</td></tr><tr><td rowspan="2">DeepSeek</td><td rowspan="2"></td><td>Evolved bank</td><td>18.6</td><td>80.3</td><td>58.9</td><td>61.3</td><td>70.9</td><td>76.7</td><td>45.4</td><td>88.4</td><td>55.2</td><td>40.1</td><td>16.8</td></tr><tr><td>Original bank</td><td>18.6</td><td>81.8</td><td>52.3</td><td>56.9</td><td>63.5</td><td>65.2</td><td>56.6</td><td>92.2</td><td>57.4</td><td>23.5</td><td>19.3</td></tr><tr><td rowspan="2">Harness</td><td rowspan="2">Qwen3.8-27B</td><td>Evolved bank</td><td></td><td>82.7</td><td>65.9</td><td>58.7</td><td>69.2</td><td>72.1</td><td>53.4</td><td>91.2</td><td>57.6</td><td>29.6</td><td></td></tr><tr><td></td><td>20.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>22.3</td></tr></table>

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Baselines and model configurations. Our main experiments use two multi-agent system (MAS) harnesses, Codex (OpenAI, 2026a) and DeepSeek-Harness (DeepSeek-AI, 2026), each paired with Qwen3.8-27B (Qwen Team, 2026b) and Gemma4-31B (Team, 2026). We additionally evaluate Codex with GPT-5.6-Luna (OpenAI, 2026b). We compare EpiCon with No Memory, Mem0 (Chhikara et al., 2025), Cognee (Markovic et al., 2025), A-Mem (Xu et al., 2026), and Agent-KB (Tang et al., 2025). External memory systems use the corresponding MAS backbone zero-shot for memory operations. Backbone-sized EpiCon uses the same backbone for memory control and tree organization; the 2B variant uses two independently trained 2B models.

Evaluation suite. We evaluate 4,538 questions from 11 benchmarks spanning document understanding, visual-to-code generation, vision-grounded mathematics, and general visual-language reasoning: DocVQA2026 (Llabres et al.´ , 2027), MP-DocVQA (Tito et al., 2022), ParseBench (Zhang et al., 2026a), Vision2Code (Periasami et al., 2026), Omni-I2C (Zhou et al., 2026), Chart-Mimic (Yang et al., 2025), MATH-Vision (Wang et al., 2024a), WeMath (Qiao et al., 2025), World-Bench (Yin et al., 2026), ReasonMap (Feng et al., 2026), and BabyVision (Chen et al., 2026).

Evaluation protocol. Across all experiments, No Memory makes a single solving attempt per question. For the main comparison, each memory-based method builds its own bank using a shared set of 1,000 construction questions from the same benchmarks, disjoint from the evaluation questions under our project-defined splits. ReasonMap is split by map, with no map shared between construction and evaluation. All evaluations with historical memory retrieval use a single solving attempt per question, with question-level memory updates disabled. In Table 1, this single-attempt protocol applies to every method. Each attempt follows the harness’s standard internal workflow. Model parameters and historical banks remain frozen during evaluation. Reference answers are used only for offline scoring. We report scores, time, and token usage, with MAS and memory costs separated. Time and token usage are normalized to the corresponding No Memory MAS costs. All experiments are run using NVIDIA H100 GPUs; GPT-5.6-Luna is accessed through its API.

## 5.2 MAIN RESULTS

With a single solving attempt per question, backbone-sized EpiCon achieves the highest elevenbenchmark macro-average in all four host configurations, exceeding the strongest external memory baseline in each by 1.9 to 5.9 points (Table 1). Macro-averages give equal weight to each benchmark. For Codex with Qwen3.8-27B, this variant improves ParseBench from 57.1 to 68.5 and MATH-

Vision from 20.4 to 48.6 over No Memory. Averaged over the two Qwen3.8-27B host configurations, it also scores highest on ten of eleven benchmarks (Figure 3(a)).

Replacing backbone-sized memory models with our two trained 2B models reduces memoryoperation time by approximately 67% to 74% and total (MAS + memory) time by 25% to 35%, with macro-average scores lower by 0.9 to 3.6 points. The 2B variant still improves macro-average scores over No Memory by 1.7 to 4.9 points across the four host configurations. In Figure 3(b) and (c), both EpiCon configurations lie on the empirical Pareto front among the compared methods for each harness with Qwen3.8-27B.

Our two 2B memory models are trained on supervision from three domains: visual mathematics, document understanding, and reasoning over charts and infographics (Section 3). They also support gains on general visual-language reasoning: the 2B variant improves WorldBench in all four configurations (+3.2 to +7.8 points), while changes on ReasonMap (+0.5 to +1.9) and BabyVision (−2.7 to +4.8) are smaller and less consistent. These results suggest that the trained memory models can manage experience from additional task types without further parameter updates.

## 5.3 EXPERIENCE TRANSFER AND MEMORY EXPANSION

We evaluate experience reuse across harnesses and backbones, followed by continued accumulation in a shared bank. All comparisons use EpiCon with fixed Memory Controller (MC) and Tree Self-Organizer (Tree) models.

Cross-backbone and cross-harness transfer. A bank built by Codex with Qwen3.8-27B is reused by Codex with Gemma4-31B and DeepSeek-Harness with Qwen3.8-27B, changing only the backbone or the harness, respectively. Each evaluated MAS performs one solve with or without the frozen bank, disabling question-level updates to assess historical experience reuse. Table 2 com pares performance across all eleven benchmarks, with macro-average improvements of 3.2 and 4.2 points for backbone and harness transfer, respectively. For example, backbone transfer improves MATH-Vision from 25.8 to 33.0, while harness transfer improves ParseBench from 52.4 to 65.7. These gains show that a system can benefit from existing shared experience without first constructing its own bank, even when its harness or backbone differs from that of the original contributor.

Cross-harness memory evolution. We test whether experience contributed by another harness can benefit the harness that built the bank. One harness builds the original bank on 1,000 construction questions; the other uses and evolves it on a separate set of 1,000 questions. We evaluate both directions between Codex and DeepSeek-Harness, with Qwen3.8-27B as the backbone throughout. In each direction, the harness that built the original bank is evaluated with the original and evolved banks on the same 4,388 questions across eleven benchmarks, disjoint from both construction sets. Both conditions use one retrieval and a single solving attempt, without question-level memory updates. Table 3 compares the original and evolved banks. Macro-average scores rise from 53.5 to 55.7 for Codex and from 53.4 to 56.6 for DeepSeek-Harness, gains of 2.1 and 3.2 points. The evolved banks improve eight and nine of eleven benchmark scores, respectively. The comparison measures the overall effect of bank evolution and expansion; construction splits are provided in Appendix C.

The benefits of cross-harness evolution depend on the task and contributing system. On ParseBench, where Codex is stronger in Table 1, its score drops from 63.4 to 58.9 after DeepSeek-Harness evolves its bank, while DeepSeek-Harness improves from 52.3 to 65.9 after Codex evolves its bank. On MATH-Vision, Codex changes only slightly from 45.2 to 45.4, while DeepSeek-Harness declines from 56.6 to 53.4. These task-dependent patterns suggest that experience from different systems can provide complementary guidance, although individual benchmarks do not improve uniformly. Taken together, Tables 2 and 3 support collective learning through shared memory: a system can benefit from experience accumulated by another system, then contribute additional experience that improves subsequent solving by the original contributor.

## 5.4 MEMORY ABLATIONS

Question-level memory control. With historical retrieval disabled, Table 4 compares No Memory, MC (ours 2B), and MC (Qwen3.8-27B). No Memory makes a single solving attempt, while MC revises memory across attempts using the question, images, existing memory, answer, and execution trace. Refinement stops after at most five attempts or earlier when the system judges the answer ready to return. With the trained 2B controller, refinement improves all four scores under both harnesses, including Vision2Code from 53.7 to 57.8 with DeepSeek-Harness. The larger controller scores higher in seven of eight comparisons, while the 2B controller requires less MAS execution time and no more memory-operation time.

Table 4: Question-level memory control with no memory bank. No Memory makes a single solving attempt; MC settings allow up to five attempts. MC (Qwen3.8-27B) uses Qwen3.8-27B as a zero-shot memory controller and MC (ours 2B) uses our trained 2B controller. We report task performance on four benchmarks. Time and token usage are normalized separately within each MAS harness using the corresponding No Memory MAS cost as the 1× baseline; the same baseline normalizes memory costs.
<table><tr><td rowspan="2">MAS</td><td rowspan="2">MAS Backbone</td><td rowspan="2">Memory Setting</td><td colspan="4">Task Performance</td><td colspan="2">Time (×)</td><td colspan="2">Tokens (×)</td></tr><tr><td>Document ParseBench</td><td>Visual-to-Code Vision2Code</td><td>Vision Math MATH-Vision</td><td>General VL BabyVision</td><td>MAS</td><td>Memory</td><td>MAS</td><td>Memory</td></tr><tr><td rowspan="3">Codex</td><td rowspan="3">Qwen3.8-27B</td><td>No Memory</td><td>57.1</td><td>53.0</td><td>20.4</td><td>14.2</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>MC (ours 2B)</td><td>59.8</td><td>55.9</td><td>41.2</td><td>17.7</td><td>2.16</td><td>0.23</td><td>2.40</td><td>0.96</td></tr><tr><td>MC (Qwen3.8-27B)</td><td>63.1</td><td>56.6</td><td>49.6</td><td>13.9</td><td>2.57</td><td>0.23</td><td>3.07</td><td>0.95</td></tr><tr><td rowspan="3">DeepSeek Harness</td><td rowspan="3">Qwen3.8-27B</td><td>No Memory</td><td>52.4</td><td>53.7</td><td>42.2</td><td>20.1</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>MC (ours 2B)</td><td>55.1</td><td>57.8</td><td>51.4</td><td>23.3</td><td>2.10</td><td>0.13</td><td>2.44</td><td>3.23</td></tr><tr><td>MC (Qwen3.8-27B)</td><td>64.5</td><td>65.8</td><td>61.6</td><td>35.1</td><td>2.47</td><td>0.16</td><td>2.65</td><td>3.25</td></tr></table>

Table 5: Multimodal memory ablation with trained MC 2B and no historical bank. Memory-enabled settings share a budget of up to five attempts; the row with three dashes in the memory configuration columns denotes No Memory with a single solving attempt. Visual memory can be frozen or evolving, and is injected either on every attempt (Always) or adaptively (Adaptive). Original task images remain available in every setting.
<table><tr><td rowspan="2">MAS</td><td rowspan="2">MAS Backbone</td><td colspan="3">Memory Configuration</td><td colspan="4">Task Performance</td><td colspan="2">Time (×)</td><td colspan="2">Tokens (×)</td></tr><tr><td>Text Memory</td><td>Visual Memory</td><td>Visual Injection</td><td>Document ParseBench</td><td>Visual-to-Code Vision2Code</td><td>Vision Math MATH-Vision</td><td>General VL BabyVision</td><td>MAS</td><td>Memory</td><td>MAS</td><td>Memory</td></tr><tr><td rowspan="6">Codex</td><td rowspan="6">Qwen3.8-27B</td><td>一</td><td>一</td><td>一</td><td>57.1</td><td>53.0</td><td>20.4</td><td>14.2</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>√</td><td>=</td><td></td><td>58.7</td><td>55.7</td><td>27.2</td><td>16.0</td><td>1.43</td><td>0.28</td><td>1.23</td><td>0.61</td></tr><tr><td>√</td><td>Frozen</td><td>Always</td><td>55.2</td><td>55.0</td><td>40.6</td><td>16.0</td><td>2.82</td><td>0.43</td><td>3.00</td><td>0.94</td></tr><tr><td>√</td><td>Frozen</td><td>Adaptive</td><td>57.1</td><td>55.9</td><td>41.0</td><td>16.3</td><td>2.02</td><td>0.25</td><td>2.42</td><td>0.94</td></tr><tr><td>√</td><td>Evolving</td><td>Always</td><td>55.3</td><td>55.2</td><td>41.0</td><td>16.3</td><td>2.89</td><td>0.44</td><td>3.02</td><td>0.93</td></tr><tr><td>√</td><td>Evolving</td><td>Adaptive</td><td>59.8</td><td>55.9</td><td>41.2</td><td>17.7</td><td>2.16</td><td>0.23</td><td>2.40</td><td>0.96</td></tr><tr><td rowspan="6">DeepSeek Harness</td><td rowspan="6">Qwen3.8-27B</td><td>一</td><td>=</td><td>一</td><td>52.4</td><td>53.7</td><td>42.2</td><td>20.1</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>√</td><td></td><td></td><td>51.8</td><td>55.1</td><td>47.4</td><td>21.2</td><td>1.33</td><td>0.13</td><td>1.25</td><td>2.37</td></tr><tr><td>√</td><td>Frozen</td><td>Always</td><td>51.2</td><td>57.5</td><td>50.2</td><td>20.5</td><td>2.52</td><td>0.21</td><td>3.13</td><td>3.75</td></tr><tr><td>√</td><td>Frozen</td><td>Adaptive</td><td>52.8</td><td>57.3</td><td>50.8</td><td>21.2</td><td>1.95</td><td>0.13</td><td>2.41</td><td>3.58</td></tr><tr><td>√</td><td>Evolving</td><td>Always</td><td>52.1</td><td>57.0</td><td>51.8</td><td>21.2</td><td>2.53</td><td>0.21</td><td>3.15</td><td>3.72</td></tr><tr><td>√</td><td>Evolving</td><td>Adaptive</td><td>55.1</td><td>57.8</td><td>51.4</td><td>23.3</td><td>2.10</td><td>0.13</td><td>2.44</td><td>3.23</td></tr></table>

Visual memory and adaptive injection. Table 5 fixes MC (ours 2B), disables historical retrieval, and uses the same budget of up to five attempts across memory-enabled settings. No Memory makes a single solving attempt; original task images remain available throughout. With adaptive injection fixed, evolving visual memory improves seven scores over frozen memory and ties the remaining score, supporting the benefit of memory co-evolution beyond selective injection. With always-on injection, frozen visual memory improves Codex’s MATH-Vision score from 27.2 to 40.6 over textonly memory. Adaptive injection improves seven of eight scores with either frozen or evolving visual memory. For evolving memory, it also reduces MAS execution time by approximately 25% with Codex and 17% with DeepSeek-Harness, with lower MAS token usage under both harnesses.

Experience organization. Table 6 compares flat and tree memory using lessons derived from the same source questions with the trained MC 2B. The flat bank stores lessons without hierarchical organization or consolidation, while the tree bank organizes and consolidates them for retrieval. Both configurations use the same harness, backbone, attempt budget, and memory-context budget. Tree memory improves all scores, including ParseBench from 52.2 to 67.3 and BabyVision from 13.2 to 15.3. These gains support organizing and consolidating experience for subsequent reuse.

## 5.5 GENERALIZATION TO A STRONGER SOLVER

We evaluate Codex with GPT-5.6-Luna to test whether EpiCon remains useful with a stronger API based solver beyond the open-source backbones. The 2B memory components and historical bank remain fixed, without target-specific retraining. Both No Memory and EpiCon make a single solving attempt; EpiCon retrieves experience from the frozen bank. Table 7 shows improvements on all four benchmarks, including ParseBench from 63.8 to 66.5.

Table 6: Memory organization with EpiCon (ours 2B). Flat and tree memory use the same source questions to create lessons.
<table><tr><td></td><td></td><td>Memory ParseBench Vision2Code MATH-Vision BabyVision</td><td></td><td></td></tr><tr><td>Flat</td><td>52.2</td><td>57.5</td><td>43.6</td><td>13.2</td></tr><tr><td>Tree</td><td>67.3</td><td>59.9</td><td>45.2</td><td>15.3</td></tr></table>

Table 7: Memory effectiveness on Codex with GPT-5.6-Luna using EpiCon (ours 2B). Both settings make a single solving attempt.
<table><tr><td>Memory</td><td>ParseBench Vision2Code MATH-Vision BabyVision</td><td></td><td></td><td></td></tr><tr><td>No Memory</td><td>63.8</td><td>65.8</td><td>51.2</td><td>11.1</td></tr><tr><td>EpiCon</td><td>66.5</td><td>67.5</td><td>57.2</td><td>14.6</td></tr></table>

## 6 CONCLUSION

We presented EpiCon, which connects question-level textual and visual memory co-evolution with the accumulation and reuse of shared experience. Our experiments show that experience can remain useful beyond the system that generated it: another harness can benefit from the existing bank and contribute new experience that improves subsequent solving by the original contributor. These findings support agent collective learning through shared external multimodal memory, allowing systems to build on one another’s experience while host model parameters remain fixed.

## REFERENCES

Shaden Alshammari, Kevin Wen, Abrar Zainal, Mark Hamilton, Navid Safaei, Albarakati Albarakati, William Freeman, and Antonio Torralba. Mathnet: A global multimodal benchmark for mathematical reasoning and retrieval. In International Conference on Learning Representations, volume 2026, pp. 12477–12507, 2026.

Yurui Chang, Yiran Wu, Qingyun Wu, and Lu Lin. Memcollab: Cross-agent memory collaboration via contrastive trajectory distillation. arXiv e-prints, pp. arXiv–2603, 2026.

Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Yiping Bao, et al. Babyvision: Visual reasoning beyond language. arXiv preprint arXiv:2601.06521, 2026.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In International Conference on Learning Representations, volume 2024, pp. 20094–20136, 2024.

Zihao Cheng, Zeming Liu, Yingyu Shan, Xinyi Wang, Xiangrong Zhu, Yunpu Ma, Hongru Wang, Yuhang Guo, Wei Lin, and Yunhong Wang. Mem2evolve: Towards self-evolving agents via coevolutionary capability expansion and experience distillation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 20784– 20831, 2026.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

DeepSeek-AI. Deepseek harness: Everything is a plugin. https://github.com/ deepseek-ai/deepseek-harness, 2026.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

Sicheng Feng, Song Wang, Shuyi Ouyang, Lingdong Kong, Zikai Song, Jianke Zhu, Huan Wang, and Xinchao Wang. Reasonmap: Towards fine-grained visual reasoning from transit maps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 41077– 41088, 2026.

Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, et al. Magentic-one: A generalist multi-agent system for solving complex tasks. arXiv preprint arXiv:2411.04468, 2024.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Nan Duan, Weizhu Chen, et al. Critic: Large language models can self-correct with tool-interactive critiquing. In International Conference on Learning Representations, volume 2024, pp. 57734–57811, 2024.

Bernal J Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiolog-´ ically inspired long-term memory for large language models. Advances in neural information processing systems, 37:59532–59569, 2024.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Steven Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for a multiagent collaborative framework. In International Conference on Learning Representations, volume 2024, pp. 23247–23275, 2024.

Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo. Promptcap: Prompt-guided image captioning for vqa with gpt-3. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2951–2963. IEEE, 2023.

Hang Hua, Jing Shi, Kushal Kafle, Simon Jenni, Daoan Zhang, John Collomosse, Scott Cohen, and Jiebo Luo. Finematch: Aspect-based fine-grained image and text mismatch detection and correction. In European Conference on Computer Vision, pp. 474–491. Springer, 2024a.

Hang Hua, Yunlong Tang, Ziyun Zeng, Liangliang Cao, Zhengyuan Yang, Hangfeng He, Chenliang Xu, and Jiebo Luo. MMComposition: Revisiting the Compositionality of Pre-trained Vision-Language Models. arXiv preprint arXiv:2410.09733, 2024b. URL https://arxiv.org/ abs/2410.09733.

Hang Hua, Qing Liu, Lingzhi Zhang, Jing Shi, Soo Ye Kim, Zhifei Zhang, Yilin Wang, Jianming Zhang, Zhe Lin, and Jiebo Luo. Finecaption: Compositional image captioning focusing on wherever you want at any granularity. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24763–24773. IEEE, 2025a.

Hang Hua, Yunlong Tang, Chenliang Xu, and Jiebo Luo. V2xum-llm: Cross-modal video summarization with temporal prompt instruction tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 3599–3607, 2025b.

Hang Hua, Ziyun Zeng, Yizhi Song, Yunlong Tang, Liu He, Daniel Aliaga, Wei Xiong, and Jiebo Luo. Mmigbench: Towards comprehensive and explainable evaluation of multi-modal image generation models. arXiv preprint arXiv:2505.19415, 2025c.

Jovana Kondic, Pengyuan Li, Dhiraj Joshi, Isaac Sanchez, Ben Wiesel, Shafiq Abedin, Amit Alfassy, Eli Schwartz, Daniel Caraballo, Yagmur Gizem Cinar, et al. Chartnet: A million-scale, highquality multimodal dataset for robust chart understanding. arXiv preprint arXiv:2603.27064, 2026.

Dasong Li, Sizhuo Ma, Hang Hua, Wenjie Li, Jian Wang, Chris Wei Zhou, Fengbin Guan, Xin Li, Zihao Yu, Yiting Lu, et al. Vquala 2025 challenge on engagement prediction for short videos: Methods and results. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3422–3432, 2025.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large language model society. Advances in neural information processing systems, 36:51991–52008, 2023.

Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks. Advances in neural information processing systems, 37:49881–49913, 2024.

Weikai Lin, Sheng Zhao, Ian Ross, Carl Marshall, Sushant Kondguli, and Yuhao Zhu. Lowpowar: Power-constrained tone mapping for augmented reality. arXiv preprint arXiv:2607.19509, 2026.

Artemis Llabres, Marc Serra Ortega, Tom´ as Ockier, Samuel Ortega Cuadra, Amritpal Singh, Chris-\` tos Georgakilas, Andrey Barsky, Ernest Valveny, and Dimosthenis Karatzas. Icdar2026 competition on multimodal reasoning over documents in multiple domains. In Gernot A. Fink, Alicia Fornes, Koichi Kise, and Daniel Lopresti (eds.),´ Document Analysis and Recognition – ICDAR 2026, pp. 336–352, Cham, 2027. Springer Nature Switzerland. ISBN 978-3-032-36042-7.

Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. In International Conference on Learning Representations, volume 2026, pp. 146197–146246, 2026.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

Vasilije Markovic, Lazar Obradovic, Laszlo Hajdu, and Jovan Pavlovic. Optimizing the interface between knowledge graphs and llms for complex reasoning. arXiv preprint arXiv:2505.24478, 2025.

Ahmed Masry, Jia Qing Tan, Shafiq Joty, Enamul Hoque, et al. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pp. 2263–2279, 2022.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 2199– 2208. IEEE, 2021.

Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawa- \` har. Infographicvqa. In 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2582–2591. IEEE, 2022.

OpenAI. Codex CLI. https://learn.chatgpt.com/docs/codex/cli, 2026a. Official documentation. Accessed: 2026-09-20.

OpenAI. GPT-5.6 Luna. https://developers.openai.com/api/docs/models/ gpt-5.6-luna, 2026b. Official model documentation. Accessed: 2026-09-20.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. In International Conference on Learning Representations, volume 2026, pp. 94327–94354, 2026.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pp. 1–22, 2023.

Ajay Vikram Periasami, Junlin Wang, and Bhuwan Dhingra. Vision2code: A multi-domain benchmark for evaluating image-to-code generation. arXiv preprint arXiv:2605.11307, 2026.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. In Proceedings ofthe 62nd annual meeting ofthe associationfor computational linguistics (volume 1: Long papers), pp. 15174–15186, 2024.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd An nual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 20023–20070, 2025.

Qwen Team. Qwen3.8-Flash-Next: A new architecture, towards ultimate cost-efficiency, August 2026a. URL https://qwen.ai/blog?id=qwen3.8-flash-next.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026b. URL https: //qwen.ai/blog?id=qwen3.8.

Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 4663– 4680, 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607. 02770.

Ruben Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for\` multi-page docvqa. arXiv preprint arXiv:2212.05935, 2022.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024a.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024b.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multiagent conversation. arXiv preprint arXiv:2308.08155, 2023.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2026.

Zhiling Yan, Dingjie Song, Hanrong Zhang, Wei Liang, Yuxuan Zhang, Yutong Dai, Lifang He, Philip S Yu, Ran Xu, Xiang Li, et al. Openskill: Open-world self-evolution for llm agents. arXiv preprint arXiv:2606.06741, 2026.

Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. In International Conference on Learning Representations, volume 2025, pp. 26590–26646, 2025.

Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Jia Xu, et al. Towards self-evolving agents: Enabling autonomy through interactive experience refinement. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 30424–30451, 2026.

Yida Yin, Harish Krishnakumar, Chung Peng Lee, Boya Zeng, Wenhao Chai, Shengbang Tong, Wenhu Chen, Hu Xu, Xingyu Fu, Gabriel Sarch, et al. Worldbench: A challenging and visually diverse multimodal reasoning benchmark. arXiv preprint arXiv:2606.06538, 2026.

Yongsheng Yu, Ziyun Zeng, Hang Hua, Jianlong Fu, and Jiebo Luo. Promptfix: You prompt and we fix the photo. Advances in Neural Information Processing Systems, 37:40000–40031, 2024.

Yongsheng Yu, Ziyun Zeng, Haitian Zheng, and Jiebo Luo. Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17324–17334. IEEE, 2025.

Yongsheng Yu, Ziyun Zeng, Zhiyuan Xiao, Zhenghong Zhou, Hang Hua, Wei Xiong, and Jiebo Luo. Aurora: Unified video editing with a tool-using agent. arXiv preprint arXiv:2605.18748, 2026.

Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Dongsheng Li, and Deqing Yang. Evoagent: Towards automatic multi-agent generation via evolutionary algorithms. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6192–6217, 2025.

Ziyun Zeng, Junyu Chen, Noha Rashwan, Nisreen Al Jallad, Jin Xiao, and Jiebo Luo. Automated detection and quantitative assessment of dental plaque in intraoral images. ACM Transactions on Computingfor Healthcare, 7(2):1–12, 2026a.

Ziyun Zeng, Hang Hua, and Jiebo Luo. Mira: Multimodal iterative reasoning agent for image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9563–9573, 2026b.

Ziyun Zeng, Hang Hua, Bocheng Zou, Mu Cai, Rogerio Feris, and Jiebo Luo. Mementogui: Learning agentic multimodal memory control for long-horizon gui agents. arXiv preprint arXiv:2605.18652, 2026c.

Ziyun Zeng, Zixuan Wang, Yongsheng Yu, Hang Hua, and Jiebo Luo. Videoargus: Agentic rubricgrounded unified evaluation for video generation and editing. arXiv preprint arXiv:2608.05485, 2026d.

Boyang Zhang, Sebastian G Acosta, Preston Carlson, Sacha Bron, Pierre-Lo ´ ¨ıc Doulcet, Daniel B Ospina, and Simon Suo. Parsebench: A document parsing benchmark for ai agents. arXiv preprint arXiv:2604.08538, 2026a.

Guibin Zhang, Muxin Fu, Kun Wang, Frank Wan, Miao Yu, and Shuicheng Yan. G-memory: Tracing hierarchical memory for multi-agent systems. Advances in Neural Information Processing Systems, 38:12988–13018, 2026b.

Wenqi Zhang, Ke Tang, Hai Wu, Mengna Wang, Yongliang Shen, Guiyang Hou, Zeqi Tan, Peng Li, Yueting Zhuang, and Weiming Lu. Agent-pro: Learning to evolve via policy-level reflection and optimization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5348–5375, 2024.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024.

Boyuan Zheng, Michael Y Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, et al. Skillweaver: Web agents can self-improve by discovering and honing skills. arXiv preprint arXiv:2504.07079, 2025.

Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. Synapse: Trajectory-as-exemplar prompting with memory for computer control. In International Conference on Learning Representations, volume 2024, pp. 19036–19066, 2024.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 19724–19731, 2024.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406, 2023.

Jiawei Zhou, Chi Zhang, Xiang Feng, Qiming Zhang, Haibo Qiu, Lihuo He, Dengpan Ye, Xinbo Gao, and Jing Zhang. Omni-i2c: A holistic benchmark for high-fidelity image-to-code generation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 37761–37798, 2026.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jurgen¨ Schmidhuber. Language agents as optimizable graphs. arXiv preprint arXiv:2402.16823, 2024.

## A DATA CONSTRUCTION DETAILS

## A.1 SOURCE TASKS AND DATA COLLECTION

Training tasks are drawn from MathNet (Alshammari et al., 2026), MathV360K (Shi et al., 2024), ChartQA (Masry et al., 2022), InfoVQA (Mathew et al., 2022), DocVQA (Mathew et al., 2021), and ChartNet (Kondic et al., 2026). They cover three domains: visual mathematics, document understanding, and reasoning over charts and infographics. The source tasks provide the questions and images on which we collect multi-agent system (MAS) executions. Memory updates, lessons, and tree-operation targets are subsequently generated from these execution records.

We collect trajectories using CAMEL (Li et al., 2023), Codex (OpenAI, 2026a), and DeepSeek-Harness (DeepSeek-AI, 2026). Qwen3.8-27B (Qwen Team, 2026b) generates initial memory updates and operation demonstrations. Qwen3.8-Flash-Next (Qwen Team, 2026a) selectively revises unsuccessful updates, improves lesson quality, and corrects tree-operation targets. The accepted targets provide separate supervision for the Memory Controller and Tree Self-Organizer.

## A.2 REPLAY-VERIFIED JOINT MEMORY TRANSFORMATIONS

Joint text–visual targets. Given the question, images, previous memory, latest attempt, and cor rectness feedback, Qwen3.8-27B proposes a complete memory state

$$
m ^ { \prime } = ( \ell ^ { \prime } , g ^ { \prime } , p ^ { \prime } , b ^ { \prime } ) ,\tag{5}
$$

where $\ell ^ { \prime }$ is guidance text, $g ^ { \prime }$ specifies visual use, and $p ^ { \prime }$ and $b ^ { \prime }$ identify a source image and normalized bounding box. When $g ^ { \prime } = 0$ , the visual-reference fields are null. The target jointly specifies textual guidance and its associated visual evidence. The evaluator uses reference answers to assess attempts and produce correctness feedback. Qwen3.8-Flash-Next revises updates that fail to support successful solving and refines lessons where needed.

Repair and compression. Replay compares the previous and proposed memory states with the same MAS host. Repair transformations are retained when the new memory turns a failed attempt into a successful one. Compress transformations preserve successful solving while shortening text without increasing crop area:

$$
\mathrm { t o k e n s } ( \ell ^ { \prime } ) < \mathrm { t o k e n s } ( \ell ) , \qquad \mathrm { a r e a } ( b ^ { \prime } ) \leq \mathrm { a r e a } ( b ) .\tag{6}
$$

Here ℓ and b denote the previous text and visual region; crop area is normalized relative to the source image and is zero when no visual region is retained. The original task images remain available during replay. The final accepted memory, including any teacher revision, is the target selected through replay.

Each training example pairs the update inputs with the accepted memory JSON. The dataset con tains approximately 25K memory transformations. Multiple transformations may originate from one source question, so this count denotes supervision examples rather than independent tasks. Replay validates the observed memory transition on the source task, not a guarantee of improvement on every subsequent task.

## A.3 TREE ORGANIZATION AND RETRIEVAL DEMONSTRATIONS

We derive seed lessons from the collected controller memories and organize them into small memory banks. Historical lessons and retrieval queries are separated within each bank. Qwen3.8-27B produces demonstrations of placement, merging, splitting, consolidation, and retrieval; Qwen3.8- Flash-Next selectively revises the operation targets.

The final targets are checked for schema validity, node references, partition coverage, selection constraints, and provenance. Valid empty retrievals are retained when no candidate applies. These checks establish structural validity; tree-operation demonstrations are not individually verified through downstream MAS replay. The dataset contains approximately 6.5K operation demonstrations, which count supervised operations rather than seed lessons or independent questions.

Training and bank construction. The supervision datasets train the two memory models. The benchmark-derived construction questions described in Appendix C are subsequently used to build external experience banks with the models fixed. These construction questions add memory content without updating model parameters. Evaluation uses separate questions and frozen historical banks.

## B TRAINING THE TWO MEMORY POLICIES

We independently fine-tune two Qwen3.5-2B models on the datasets in Section 3. For a policy $P _ { \psi }$ and its operation dataset $\mathcal { D } _ { P }$ , supervised fine-tuning minimizes the target-sequence loss

$$
\mathcal { L } _ { P } ( \psi ) = \mathbb { E } _ { ( z , y ) \sim \mathcal { D } _ { P } } \left[ - \sum _ { j = 1 } ^ { | y | } \log p _ { \psi } ( y _ { j } \mid z , y _ { < j } ) \right] ,\tag{7}
$$

where z contains the operation prompt and its available inputs, and $y$ is the validated memory or operation JSON. Controller targets cover repair and compression of joint text–visual memory. Tree targets cover batch placement, merging, splitting, consolidation, and retrieval; only retrieval examples include images.

Both models use LoRA with rank 16 and scaling factor 32, targeting linear layers while freezing the visual encoder and aligner. Training uses three epochs, a learning rate of $1 0 ^ { - 4 }$ , maximum sequence length 8,192, per-device batch size 2, and gradient accumulation over 8 steps. Checkpoints are selected by validation loss: step 1,000 for the controller and step 258 for the tree memory selforganizer.

Training and deployment interfaces. Offline training-data construction uses correctness feedback to generate and screen memory targets. During evaluation with Codex and DeepSeek-Harness, MC revises memory from the question, images, existing memory, answer, and execution trace. The joint update supervises text, visual use, and the source image region together. Adaptive visual injection controls which stored visual memory enters the next solver input; omitting visual memory from an input does not delete it from the stored state.

## C EXPERIMENTAL PROTOCOL AND DATA SPLITS

Construction and evaluation sets. Table 8 lists the benchmark-specific question counts. Experiments that use historical memory share an initial set of 1,000 construction questions: 10 from DocVQA2026, 90 from MP-DocVQA, and 100 from each of the other nine benchmarks. Each memory setting builds its bank from the corresponding MAS execution records, or reuses the specified donor bank in transfer experiments. These questions construct memory content without updating model parameters. No Memory and the question-level ablations do not use a historical bank. The standard evaluation split contains 4,538 questions across eleven benchmarks; experiments reporting a subset of benchmarks use the corresponding evaluation subsets. Construction and evaluation are project-defined splits and are disjoint. For ReasonMap, questions are grouped by their source map before splitting: a map is assigned either to construction or to evaluation, never to both. This maplevel separation also applies when adding the second construction set. Construction counts refer to source questions, not the number of accepted memory episodes.

Cross-harness memory evolution. The first construction stage uses the same 1,000 questions as the other historical-memory experiments. A second, non-overlapping set of 1,000 questions is used by the other harness to use and evolve the original bank. Table 8 gives the distribution for both stages. The evaluation split contains 4,388 questions and excludes both construction sets. Codex and DeepSeek-Harness exchange the roles of initial construction and subsequent evolution, with Qwen3.8-27B and the two trained 2B memory models fixed. For each evaluated harness, the original and evolved banks are compared on identical questions with one historical retrieval and a single solving attempt, without question-level memory updates. Table 3 follows the same singleattempt protocol as Table 1; its evaluation split additionally excludes the second construction set.

Evaluation controls. Model parameters and historical banks remain frozen during evaluation. The main comparison and all evaluations with historical memory retrieval use a single solving attempt per question, with question-level memory updates disabled. Each attempt includes the harness’s standard internal agent and tool interactions. Where enabled in other experiments, question-level updates affect only the current question and are not written back to the bank. Reference answers and official scores are used only for offline evaluation and do not enter memory updates, stopping decisions, or answer selection. We report the last completed answer. Cross-backbone and cross-harness transfer compare one solve with and without the donor bank, with question-level updates disabled. No Memory makes a single solving attempt in every experiment, including the question-level controller comparison, multimodal ablation, and stronger-solver comparison. The question-level controller comparison and multimodal ablation disable historical retrieval and allow up to five solving attempts for memory-enabled settings, including the initial attempt. The system may stop earlier when it judges the current answer ready to return. Their comparisons against No Memory measure memory-guided refinement as a whole, including additional attempts. Memory-enabled settings in the multimodal ablation share the attempt budget. Original task images remain available in the multimodal ablation. Results may be reused across experiments only when the samples, harness, backbone, memory configuration, bank, and inference protocol match; a different evaluation subset requires recomputing scores from the corresponding predictions.

Table 8: Construction and evaluation questions by benchmark. Initial construction is shared by historical-memory experiments and the first stage of cross-harness memory evolution. Additional construction is used only for memory evolution; its evaluation set excludes both construction sets.
<table><tr><td rowspan="2">Benchmark</td><td colspan="2">Construction</td><td colspan="2">Evaluation</td></tr><tr><td>Initial</td><td>Additional</td><td>Standard</td><td>Evolution</td></tr><tr><td>DocVQA2026</td><td>10</td><td>0</td><td>70</td><td>70</td></tr><tr><td>MP-DocVQA</td><td>90</td><td>100</td><td>500</td><td>500</td></tr><tr><td>ParseBench</td><td>100</td><td>50</td><td>468</td><td>418</td></tr><tr><td>Vision2Code</td><td>100</td><td>200</td><td>500</td><td>500</td></tr><tr><td>Omni-I2C</td><td>100</td><td>150</td><td>500</td><td>500</td></tr><tr><td>ChartMimic</td><td>100</td><td>0</td><td>500</td><td>500</td></tr><tr><td>MATH-Vision</td><td>100</td><td>100</td><td>500</td><td>500</td></tr><tr><td>WeMath</td><td>100</td><td>100</td><td>500</td><td>500</td></tr><tr><td>WorldBench</td><td>100</td><td>210</td><td>500</td><td>500</td></tr><tr><td>ReasonMap</td><td>100</td><td>40</td><td>212</td><td>162</td></tr><tr><td>BabyVision</td><td>100</td><td>50</td><td>288</td><td>238</td></tr><tr><td>Total</td><td>1,000</td><td>1,000</td><td>4,538</td><td>4,388</td></tr></table>

Score and cost reporting. We use each benchmark’s evaluation metric and give equal weight to each benchmark when computing macro-average scores. Time and token usage are reported separately for MAS execution and memory operations. For each comparison, let $T _ { 0 }$ and $N _ { 0 }$ be the corresponding No Memory MAS time and token usage under the same harness, backbone, evaluation set, and baseline protocol. We normalize each cost component c as

$$
\widetilde { T } _ { c } = \frac { T _ { c } } { T _ { 0 } } , \qquad \widetilde { N } _ { c } = \frac { N _ { c } } { N _ { 0 } } , \qquad c \in \{ \mathrm { M A S } , \mathrm { M e m o r y } \} .\tag{8}
$$

Thus, No Memory MAS time and token usage are each 1×, and its memory-operation costs are zero. Memory costs use the same MAS denominators rather than a separate memory baseline.

Hardware. All experiments are run using NVIDIA H100 GPUs. GPT-5.6-Luna is accessed through its API.

## D ADDITIONAL PERFORMANCE AND COST COMPARISONS

Figures 4–5 extend Figure 3 using the same main-results measurements. With Gemma4-31B, backbone-sized EpiCon improves the eleven-benchmark mean over No Memory by 7.0 points on Codex and 2.7 points on DeepSeek-Harness; the trained 2B variant gains 4.2 and 1.7 points with lower total time than the backbone-sized variant. Figure 5 compares token usage on DeepSeek-Harness with the two backbones. Relative to backbone-sized EpiCon, the trained 2B variant uses 25.1% fewer total tokens with Qwen3.8-27B and 16.8% fewer with Gemma4-31B, while yielding lower mean benchmark scores. These results show the performance–token trade-off of the two trained 2B memory models across backbones.

![](images/945d8d2855435a319572e8debddc8210adcaadf52d9376ab94b20c1d836bd7f4.jpg)  
Figure 4: Benchmark profiles and performance–time trade-offs with Gemma4-31B, using the results in Table 1. (a) Scores are averaged across Codex and DeepSeek-Harness, then divided by the best method mean for each benchmark; the radial axis starts at 0.25, as in Figure 3. (b) and (c) show performance as the equally weighted mean across eleven benchmarks. Total time sums the normalized MAS and memory costs within each harness. Dashed lines connect No Memory, EpiCon (ours 2B), and EpiCon (Gemma4-31B) as visual guides, not Pareto fronts.

![](images/6491f30e1bfde36ca6528feea6381a5285d48712cc5ffbc11884730c6f4ff9a9.jpg)  
Figure 5: Performance–token trade-offs on DeepSeek-Harness with (a) Qwen3.8-27B and (b) Gemma4-31B. Performance is the equally weighted mean across eleven benchmarks in Table 1. Total token usage sums MAS and memory tokens, normalized by the corresponding No Memory MAS usage for each backbone. Memory methods use the backbone named below each panel, except EpiCon (ours 2B), which uses two independently trained 2B memory models. The panels share the token scale and use separate score ranges. Dashed lines connect No Memory and the two EpiCon variants as visual guides, not Pareto fronts.