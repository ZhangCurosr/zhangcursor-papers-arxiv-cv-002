# RECHART: REASONING-EFFICIENT CHART EDITING WITH LARGE REASONING MODELS

Yuanbang Liu, Chenxi Ruan, Yihan Hou, Qiong Luo & Wei Zeng ∗ HKUST(GZ)

## ABSTRACT

Chart editing requires inferring and modifying visualization code from a reference chart image based on an editing instruction, challenging fine-grained visual reasoning, instruction following, and executable code synthesis capabilities of MLLMs. Large reasoning models (LRMs) with extended Chain-of-Thought (CoT) reasoning are suitable for tackling such complex multimodal tasks. However, our preliminary study reveals an “inverted-U” relationship between reasoning length and chart-editing performance: Excessive reasoning often leads to “overthinking,” where models drift toward hallucinated visual details or get stuck in redundant reasoning loops. To address the gap, we introduce REChart, a two-stage training framework that provides process-level supervision over intermediate reasoning steps, improving both editing fidelity and reasoning efficiency. First, we synthesize 200k high-quality reasoning trajectories for supervised fine-tuning from a large image-instruction-code pool, using a role-specialized agentic Reason-Score-Refine workflow that iteratively refine the chart code toward higher quality. Second, we optimize the model via reinforcement learning with two complementary rewards: a fidelity reward evaluating code correctness, visual fidelity, and structural consistency, and an efficiency reward that assigns each rollout a random thinking budget, truncates the reasoning process, and credits the final reasoning segment according to its contribution to the output. On the ChartEdit and ChartMIMIC benchmarks, our model achieves state-of-the-art chart-editing performance among open-source models of comparable scale, while mitigating overthinking and reducing average reasoning token usage by 79.0% under a maximum thinking budget of 16,384 tokens compared with the base model.

## 1 INTRODUCTION

Refinement of visual content plays an important role in scientific research and business intelligence (Friendly, 2008). Chart editing, which involves iteratively adapting and refining visualization code by jointly interpreting source chart images and nuanced editing instructions, is a key aspect of this refinement process. Effective chart editing requires models to refine visual elements, adjust visual encodings, and adapt charts to new data, making it a challenging task for evaluating the reasoning and code generation capabilities of multimodal large language models (MLLMs) (Yang et al., 2025a; Zhao et al., 2025a).

The advent of large reasoning models (LRMs), characterized by extended Chain-of-Thought (CoT) processes, has shown immense potential in tackling complex reasoning tasks (Wei et al., 2022). While prior evidence suggests that scaling the length of reasoning tokens often correlates with improved performance (Guo et al., 2025; Team, 2025a; OpenAI, 2024), recent studies have identified a nonmonotonic relationship (Aggarwal et al., 2026; Chen et al., 2025a; Sui et al., 2025). Specifically, excessive reasoning may trigger overthinking, in which models become preoccupied with trivial details or amplify initial errors, leading to a noticeable decline in output quality (Feng et al., 2025; Wu et al., 2026). To address this, several works have explored strategies such as length-penalized training, budget forcing, and adaptive reasoning depth to encourage more efficient use of reasoning tokens (Xiang et al., 2025; Muennighoff et al., 2025; Li et al., 2025). Despite these efforts, it remains unclear whether extended CoT improves chart editing, where precise visual perception must be coordinated with complex code synthesis.

We first conduct a preliminary study investigating the scaling behavior of LRMs on the ChartEdit benchmark (Zhao et al., 2025a) under varying thinking budgets. Our findings reveal a distinct inverted-U relationship between reasoning token length and model performance: while initial CoT expansion enhances accuracy, excessive reasoning eventually triggers performance degradation as the model hallucinates visual details and falls into circular reasoning. Besides, we observe that stronger models reach their peak performance with shorter reasoning chains, while harder tasks demand longer optimal budgets. Together, these findings expose interrelated weaknesses of current LRMs in chart editing: reasoning efficiency and editing fidelity, as models routinely overshoot the optimal thinking budget that may lead to visual hallucinations and inconsistent code generation. While recent advances in multimodal RL have introduced dense rewards for terminal code structures (Chen et al., 2026a; 2025b), they largely leave the intermediate reasoning process unsupervised, which may lead to correct code via inconsistent thinking.

To enhance the model’s reasoning efficiency and editing fidelity, we propose a two-stage training framework. In the first stage, we construct a large-scale SFT dataset of 200k trajectories via a multi-agent collaborative generation framework. Our framework generates trajectories that instantiate a Reason-Score-Refine loop, progressively refining intermediate outputs toward high-quality code generation. In the second stage, we optimize the model via reinforcement learning with a hybrid reward that jointly guides the reasoning process and the final output. Specifically, the reward combines an efficiency reward that samples a thinking budget for each rollout and credits the final reasoning segment before truncation, and a fidelity reward evaluating the finally generated chart across three dimensions, including code correctness, visual fidelity, and structural consistency.

We evaluate our model on ChartEdit and ChartMimic benchmarks across multiple tasks, achieving state-of-the-art performance among models of comparable scale. Compared to the base model, our approach yields a 13.3-point average overall performance gain across ChartEdit, ChartMIMIC Customized Mimic, and ChartMIMIC Direct Mimic. Beyond accuracy, our method substantially improves reasoning efficiency. Under a maximum thinking budget of 16,384 tokens, average reasoning token consumption is reduced by 79.0% relative to the base model, with the largest reduction of 83.0% on ChartMIMIC Customized Mimic.

In summary, our contributions are as follows:

• We present a systematic empirical study revealing a distinct inverted-U relationship between reasoning token length and chart editing performance.

• We propose REChart, a two-stage training framework comprising SFT with a structured Reason-Score-Refine trajectory dataset and reinforcement learning with a hybrid reward guiding both the reasoning process and final output.

• Extensive evaluations demonstrate that REChart achieves state-of-the-art chart editing performance among open-source models of comparable scale, with a 13.3-point average overall improvement and 79.0% reduction in reasoning token cost over the base model.

## 2 RELATED WORK

## 2.1 MLLMS FOR CHART UNDERSTANDING

Chart understanding with MLLMs is a rapidly growing field with many directions (Huang et al., 2025; Davila et al., 2021). Some works focus on generating natural language outputs from charts, including chart question answering and summarization (Masry et al., 2022; 2023; Liu et al., 2024; Zeng et al., 2024; Rahman et al., 2023; Kantharaj et al., 2022; Zhang et al., 2024), and also table extraction from charts to facilitate reasoning (Liu et al., 2023a;b; Xu et al., 2025; Meng et al., 2024). Other works focus on chart-to-code generation, with dedicated benchmarks covering diverse chart types and evaluation dimensions (Yang et al., 2025a; Wu et al., 2025; Zhao et al., 2025b;a). To improve generation fidelity, recent methods leverage reinforcement learning with both textual and visual feedback. For example, Tan et al. (2025) proposed a chart similarity reward measuring both attribute and visual similarity, while Chen et al. (2026a) introduced a multi-granularity structured reward combining textual and visual evaluation. While these methods improve chart-to-code generation quality, they do not optimize the reasoning process itself, nor do they explore the applicability of their approaches to large reasoning models. It remains unclear how to harness stronger reasoning capabilities efficiently for chart editing, a task that inherently demands long and complex reasoning to interpret visual elements, understand user intent, and synthesize precise code modifications.

## 2.2 LARGE REASONING MODELS

Large reasoning models (LRMs) such as OpenAI o1 (OpenAI, 2024), DeepSeek-R1 (Guo et al., 2025), and Kimi k1.5 (Team, 2025b) extend the capabilities of language models by generating explicit chain-of-thought reasoning traces before producing a final answer. This paradigm has demonstrated substantial gains on tasks (Wei et al., 2022; Sprague et al., 2025; Chen et al., 2023) such as complex logical inference, mathematical reasoning, and structured code generation, where deliberate step-by step thinking proves critical to correctness. However, the extended reasoning process also introduces significant token overhead, as models tend to generate verbose and sometimes redundant thinking traces regardless of task (Sui et al., 2025; Balachandran et al., 2025). This has motivated a growing line of work on reasoning efficiency, exploring strategies such as length-penalized training (Xiang et al., 2025; Liu et al., 2026; Aggarwal & Welleck, 2025; Hou et al., 2026), budget forcing (Muennighoff et al., 2025; Han et al., 2025; Yang et al., 2025b), and adaptive reasoning depth (Li et al., 2025; Zhang et al., 2025; Xiang et al., 2025; WANG et al., 2026) to reduce unnecessary computation while preserving output quality. Chart editing is a particularly representative case, as it demands both fine-grained visual perception and structured code synthesis, making efficient reasoning critical for both quality and computational cost.

## 3 PROBLEM DEFINITION AND PRELIMINARY STUDY

## 3.1 PROBLEM DEFINITION

We focus on the chart-editing task, where a model generates updated chart code given a reference chart image and a natural-language editing instruction. A chart is formally represented as a tuple (I, C), where I denotes the rendered image and C denotes the executable script (e.g., using Matplotlib or Seaborn in Python) that produces I. Given a reference chart image $I _ { \mathrm { r e f } }$ and an editing instruction T, a model M outputs an output script:

$$
C _ { o u t } = \mathcal { M } ( I _ { \mathrm { r e f } } , T ) .\tag{1}
$$

The reference code $C _ { \mathrm { r e f } }$ remains hidden during inference, requiring the model to ground edits in visual perception rather than direct code manipulation. Our goal is to develop a model that can effectively grounds edits along three dimensions: code correctness and structural consistency of $C _ { o u t }$ , and visual fidelity of $I _ { o u t }$ generated from $C _ { o u t }$

Given the diversity of visual components and editing instructions, chart editing is inherently complex. To systematically address this complexity, we built a taxonomy of edit operations through a multi-stage process. First, we aggregated a diverse set of editing samples from prior chart-editing datasets (Zhao et al., 2025a; Yang et al., 2025c) and editing tasks documented in the visualization community (Vaithilingam et al., 2024; Xie et al., 2026). Next, two visualization experts independently labeled and clustered these samples based on the induced operation. The experts focused on the type of modification (e.g., adding a data series, changing a color palette, rotating axis labels, filtering data points) rather than the specific code implementation. Through iterative consolidation, this process yielded six non-overlapping edit categories formally defined in Table 1. Among these, Direct Reproduction serves as a no-edit baseline isolating raw image-to-code capability, and the remaining five categories encompass the diverse edit operations we distilled.

## 3.2 REASONING SCALING BEHAVIOR OF LRMS

We conduct an empirical study on the ChartEdit benchmark (Zhao et al., 2025a), evaluating the Qwen3-VL series (4B, 8B, 32B) (Bai et al., 2025a), GLM-4.1V-9B (Hong et al., 2025), and Gemin 3.1 Pro (Pichai et al., 2025) under varying thinking budgets.

Figure 1a shows a consistent inverted-U pattern between reasoning length and chart-editing performance. Allowing LRMs to reason up to an appropriate budget improves accuracy over the non-thinking setting, indicating that explicit reasoning helps models coordinate visual perception, instruction following, and code synthesis. However, extending the budget to the longest setting often degrades performance, especially for open-source models. This suggests that additional reasoning is not uniformly beneficial: after a model passes its effective reasoning budget, the chain of thought can drift away from the visual evidence and reduce final execution or visual fidelity.

Table 1: Taxonomy of chart-editing tasks across six functional categories.
<table><tr><td>Category</td><td>Key Editing Aspects</td></tr><tr><td>Direct Reproduction</td><td>No-edit baseline: reconstruction of the input chart from image to code.</td></tr><tr><td>Content Manipulation</td><td>Data filtering, aggregation, point/series addition or removal, and trend-line overlays.</td></tr><tr><td>Style &amp; Aesthetics</td><td>Modifications to backgrounds, grids, markers, colormaps, and visual effects.</td></tr><tr><td>Layout Edits</td><td>Subplot composition, legend and colorbar positioning, spacing, and figure sizing.</td></tr><tr><td>Chart Transformation Axes &amp; Annotations</td><td>Chart type swapping, orientation changes, and coordinate system transformations. Title and label content and typography, axis scaling, ticks, and data annotations.</td></tr></table>

![](images/1fd2b644d92f5d4350dc0a45e8a5bdd8088db8a6299ee079e2d4109e0db46d5b.jpg)  
(a) Model performance under non-thinking, optimal-budget, and longest-budget settings.

![](images/13b6a15e3c8c8a0cba126b4ccbfe18b3ed34cc899fe0dd200f24d6f96348f072.jpg)  
(b) Error patterns in Qwen3-VL-8B reasoning under increasing thinking budgets.  
Figure 1: Reasoning scaling on ChartEdit. (a) Performance follows an inverted-U trend with reasoning length. (b) Error patterns shift toward hallucination and redundancy as the budget grows.

To understand why this degradation occurs, we further analyze Qwen3-VL-8B on the ChartEdit human set and categorize the errors appearing in its reasoning traces under different thinking budgets. As shown in Figure 1b, longer budgets do not simply add useful intermediate steps. Instead, visual hallucination steadily increases as the model spends more tokens describing chart elements that are not present or over-specifying visual details that later misguide the generated code. Redundant reasoning also grows sharply with the budget, showing that the model often revisits the same observations without producing new evidence for the edit. By contrast, details-missing errors decrease as the budget increases, indicating that longer reasoning initially helps the model inspect more visual components. The resulting trade-off explains the inverted-U behavior: moderate reasoning reduces under-analysis, but excessive reasoning introduces hallucinated or repetitive intermediate claims that eventually harm chart-editing performance.

## 4 RECHART

## 4.1 REASONING TRAJECTORY CONSTRUCTION FOR CHART EDITING

To train a model capable of efficient and high-fidelity chart-to-code generation, we build the supervised training data by first curating an image-instruction-code corpus from real-world scientific figures and then synthesizing structured reasoning trajectories from it.

![](images/90283ef15dd9808c8855b3bc8f4da84b711da3fa2f9754ca9c984e35e86baa15.jpg)  
Figure 2: The data collection and annotation pipeline consists of three stages: (1) Chart Collection and Filtering; (2) Chart-Code Pair Generation; and (3) Chart Editing Augmentation.

## 4.1.1 INSTRUCTION CORPUS CONSTRUCTION

In this work, we construct a real-world dataset of 709k image-instruction-code triplets for fine-tuning. The corpus construction pipeline has three stages: 1) chart image collection, 2) code synthesis, and 3) editing augmentation (see Figure 2).

Chart Image Collection. To cover diverse real-world chart styles and topics, we collect 2024 arXiv papers from computer science, bioinformatics, and mathematics, retaining roughly 100k CC BY 4.0 articles. From their LAT X sources, we extract 854k embedded figures and use an MLLM classifier to filter out non-chart figures, yielding 622k chart images.

Code Synthesis. To balance efficiency and quality, we synthesize visualization code with a tiered set of generators (Qwen3-VL-32B, Qwen3-VL-235B-A22B, and Gemini 3 Flash). Rather than requiring pixel-level reproduction, we retain image-code pairs that pass a Qwen3-VL-32B audit on readability, completeness, visual integrity, and code-visual consistency; failed cases are forwarded to the next generator in the cascade. To verify audit reliability, we manually inspect a random sample of 1,000 pairs. This process yields 521k high-quality image-code pairs, with full prompts and filtering statistic provided in Appendix B.

Editing Augmentation. We then build the editing portion of the dataset by prompting Qwen3- VL-32B to generate instruction-code pairs over the five editing categories defined in Table 1. Since each of the 521k chart-code pairs is prompted three times, the accepted candidates across all pairs amount to 567k editing samples, while the 142k pairs for which no candidate passes the audit are repurposed as direct-reproduction samples. The final dataset comprises approximately 709k image-instruction-code triplets spanning the six functional dimensions of the taxonomy, covering both direct reproduction and chart editing tasks. Detailed pipeline statistics, model assignments, and prompts are provided in Appendix B.

## 4.1.2 REASONING TRAJECTORY SYNTHESIS

Building on the preliminary study in Section 3, we synthesize SFT trajectories that provide explicit step-level supervision to address the overthinking problem of LRMs in chart editing. Naively prompting a single model to produce longer CoT traces does not resolve these issues, since the resulting traces are still optimized only by the final answer and may contain redundant or misleading reasoning steps. To address these issues at the data-construction stage, we use a role-specialized multiagent workflow as illustrated in Figure 3. Inspired by MRT that estimate single reasoning episode’s contribution to the final outcome to guide the reasoning process (Qu et al., 2025), the workflow uses an external scorer to measure episode-level progress, prune stagnant reasoning segments, and redirect the trajectory through critique when the current path stops improving.

(1) Agent Roles. The system comprises five role-specialized agents, each operating under a dedicated prompt. The Planner Agent (P) receives the reference image, the editing instruction, and the ground truth code, and produces a concise overall plan that decomposes the task into a sequence of high-level steps. The Reasoner Agent (T ) operates only on the reference image, the instruction, and the overall plan, and at each iteration appends a single new reasoning episode to the running context, mimicking the step-wise self-talk of o1-style reasoning. The Generator Agent (G) consumes the accumulated context and emits a candidate Python script. The Critic Agent (C) is invoked when a candidate fails verification, and produces a first-person refine instruction (e.g., “Wait, I should re-examine the legend layout. . .”) that pinpoints a concrete weakness in the current reasoning, framed as the model’s own self-correction inside the trajectory. The Finalizer Agent (F) is invoked once a candidate clears the quality bar, and composes a brief concluding thought that closes off the chain of thought before the final code block. The detailed implementation and setups of the pipeline are detailed in Appendix C.1.

![](images/7bae17eed99f6981c27467bd26aeee0f0d7f40460a0c399d659552b4fea1e720.jpg)  
Figure 3: Multi-agent pipeline for generating the reasoning path.

(2) Episode-Level Reason-Score-Refine Loop. A trajectory begins with P producing the overall plan, after which the system enters an iterative loop in which each iteration constitutes one episode. At iteration t, T extends the context with a new reasoning episode, G produces candidate code from the updated context, and an external multi-level scorer $s _ { t } \in [ 0 , 1 0 0 ]$ evaluates the rendered chart against the reference image using the instruction, the code, and an intermediate SVG (see in Section 4.2.1). We treat the score change $\Delta _ { t } = s _ { t } - \operatorname* { m a x } _ { t ^ { \prime } < t }$ s<sub>t</sub>′ as a measurable proxy for the per-episode progress in the sense of Qu et al. (2025): an episode that advances the running best contributes to eventual success, while one that fails to do so accrues regret without informational return.

If $s _ { t }$ reaches the acceptance threshold τ = 85, the loop terminates and control passes to F. Otherwise, the system applies a stagnation check before continuing. When three consecutive episodes have failed to surpass the running best $( \Delta _ { t - 2 } , \Delta _ { t - 1 } , \Delta _ { t } \leq 0 )$ , we interpret this as a high-regret excursion: the corresponding three episodes are removed from the context, and the context is rolled back to the snapshot that produced the running best. This explicit pruning prevents the trajectory from drifting along an unproductive path and bounds the regret accumulated within any single trajectory, which is the property our downstream SFT loss inherits from the data. C then generates a refine instruction conditioned on the current state, which is appended to the context to bias the next episode toward the deficiency it identifies, and control returns to T. The loop is capped at 9 iterations, and trajectories that fail to converge within this budget are discarded.

(3) Trajectory Assembly. The accepted trajectory comprises the overall plan from P, an alternating sequence of reasoning episodes from T and refine instructions from C, the closing thought from F, and the final code from G. Following the standard o1-style format, the reasoning portion is wrapped in <think> . . . </think> tags and followed by the final output with executable code inside. By construction, the synthesized trajectories retain only reasoning episodes that collectively advance the fidelity score over successive ones, exposing the model to structured reasoning behavior where each retained sequence of steps contributes measurable progress toward the final output.

## 4.2 REINFORCEMENT FINETUNING

![](images/aae2345893c7664ba222c4d803ca645d3cf0a369954e662cfb776711e4117df2.jpg)  
Figure 4: Overview of our RL framework guided by an episode-level efficiency reward and a multidimensional fidelity reward.

While SFT equips the model with structured reasoning behavior over synthesized trajectories, it may overfit to the synthetic data distribution and exhibit performance degradation on out-of-distribution editing instructions. To address this, we further optimize the model via reinforcement learning with two complementary rewards: a fidelity reward evaluating the final generated chart across code correctness, visual fidelity, and structural consistency, and an efficiency reward that randomly budgets each rollout, truncates the reasoning process, and credits the final reasoning segment before the answer (see Figure 4).

## 4.2.1 FIDELITY REWARD

For each rollout, the code extracted from the model’s final output is executed in a sandboxed environment to obtain both the rendered SVG and a rasterized image. We assess fidelity across the following three dimensions.

Code Similarity. We apply CodeBLEU (Ren et al., 2020) to measure the overall code similarity between the generated and ground-truth solution.

Structural Similarity. Rather than comparing raw SVG markup, we construct a lightweight structural representation of the chart. Specifically, we parse the SVG into its DOM tree, prune renderingirrelevant nodes, and classify the path elements into geometric primitives based on their d attributes. Each node is then reduced to its essential attributes, including element type, text content, and color. The resulting simplified trees of the generated and ground-truth charts are compared to produce a structural similarity score, reflecting whether the chart’s compositional layout is correctly reproduced.

Visual Fidelity. We employ a multimodal embedding model Qwen3-VL-Embedding (Li et al., 2026) to evaluate instruction-following quality. Specifically, we measure the embedding similarity between the generated chart and the input context, which consists of the editing instruction paired with the reference chart image. This score captures whether the output faithfully reflects the required visual changes.

The fidelity reward $R _ { \mathrm { f i d e l i t y } }$ is computed as a weighted average of the three dimension scores:

$$
R _ { \mathrm { f i d } } = w _ { c o d e } \cdot s _ { \mathrm { c o d e } } + w _ { s t r u c t } \cdot s _ { \mathrm { s t r u c t } } + w _ { v i s } \cdot s _ { \mathrm { v i s } }\tag{2}
$$

## 4.2.2 EFFICIENCY REWARD.

Optimizing solely on the outcome reward (fidelity reward $R _ { f i d e l i t y } )$ suffers from the creditassignment problem (Qu et al., 2025), where a successful rollout reinforces all intermediate reasoning episodes equally, regardless of their actual contribution to the final outcome. Following Qu et al. (2025), we incorporate an episode-level efficiency reward that measures whether the final reasoning segment before the answer improves the fidelity reward under a sampled compute budget.

Concretely, for each rollout, we randomly sample a thinking budget B and allow the model to reason until it either emits the end-of-thinking token or reaches the budget. If the rollout reaches B before naturally stopping, we truncate the reasoning context and force the model to enter the answer phase. Let n denote the actual number of reasoning segments exposed under this procedure, which may be smaller than or equal to the sampled budget B. We denote by $e _ { n }$ the final reasoning segment immediately preceding the answer, by $h _ { < n }$ the truncated context before $e _ { n } ,$ , and by $h _ { \leq n } = \bar { h _ { < n } } \bar { \cup } \{ e _ { n } \}$ the context that includes it. We decode answers from both contexts, evaluate them under the fidelity reward to obtain $R _ { \mathrm { f i d } } ^ { ( < n ) }$ and $R _ { \mathrm { f i d } } ^ { ( \leq n ) }$ , and define the efficiency reward as:

$$
R _ { \mathrm { e f f } } ( e _ { n } ) = \operatorname* { m a x } ( R _ { \mathrm { f i d } } ^ { ( \leq n ) } - R _ { \mathrm { f i d } } ^ { ( < n ) } , 0 ) .\tag{3}
$$

The reward therefore credits only the last reasoning segment actually exposed before answer generation, encouraging the model to make the reasoning immediately before generation concise and useful rather than rewarding arbitrarily selected spans from a completed trajectory.

## 4.2.3 POLICY OPTIMIZATION

The composite reward used for policy optimization is then:

$$
R = w _ { \mathrm { f i d } } R _ { \mathrm { f i d } } + w _ { \mathrm { e f f } } R _ { \mathrm { e f f } } .\tag{4}
$$

We optimize the model with this reward via GRPO (Shao et al., 2024), which samples a group of candidate responses per input and normalizes rewards within the group to compute relative advantages for policy updates.

Table 2: Performance on ChartEdit w/o Code and ChartMIMIC (customized mimic) benchmarks.
<table><tr><td rowspan="2">Model</td><td rowspan="2"># Params.</td><td colspan="4">ChartEdit w/o Code</td><td colspan="4">ChartMIMIC Customized</td></tr><tr><td>Exec.</td><td>Code</td><td>Chart</td><td>Overall</td><td>Exec.</td><td>Low</td><td>High</td><td>Overall</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>GPT-4o (Hurst et al., 2024)</td><td></td><td>91.5</td><td>60.0</td><td>79.9</td><td>70.0</td><td>96.5</td><td>82.1</td><td>84.3</td><td>83.2</td></tr><tr><td>Claude-3.7-Sonnet (Anthropic, 2025)</td><td></td><td>96.3</td><td>73.4</td><td>87.5</td><td>80.5</td><td>80.6</td><td>67.4</td><td>78.1</td><td>72.3</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al., 2025)</td><td></td><td>95.3</td><td>82.9</td><td>89.2</td><td>86.1</td><td>91.7</td><td>81.3</td><td>83.5</td><td>82.4</td></tr><tr><td colspan="10">Open-Source Models</td></tr><tr><td>Qwen2.5-VL-72B (Bai et al., 2025b)</td><td>72B</td><td>89.8</td><td>57.7</td><td>71.0</td><td>64.4</td><td>85.3</td><td>67.6</td><td>69.1</td><td>68.4</td></tr><tr><td>Qwen2.5-VL-32B (Bai et al., 2025b)</td><td>32B</td><td>88.4</td><td>61.7</td><td>72.5</td><td>67.1</td><td>85.6</td><td>66.4</td><td>69.5</td><td>68.0</td></tr><tr><td>Qwen3-VL-32B-Thinking (Bai et al., 2025a)</td><td>32B</td><td>85.5</td><td>72.9</td><td>76.3</td><td>74.6</td><td>84.0</td><td>67.0</td><td>72.3</td><td>69.6</td></tr><tr><td>Qwen3-VL-32B-Instruct (Bai et al., 2025a)</td><td>32B</td><td>84.9</td><td>69.0</td><td>73.4</td><td>71.2</td><td>88.2</td><td>72.4</td><td>61.3</td><td>66.9</td></tr><tr><td>Qwen3-VL-8B-Thinking (Bai et al., 2025a)</td><td>8B</td><td>79.1</td><td>66.6</td><td>68.0</td><td>67.3</td><td>77.6</td><td>59.6</td><td>63.4</td><td>61.5</td></tr><tr><td>GLM4.1v-9B-Thinking (Hong et al., 2025)</td><td>9B</td><td>78.9</td><td>68.4</td><td>68.7</td><td>68.6</td><td>80.7</td><td>63.7</td><td>65.7</td><td>64.7</td></tr><tr><td>Qwen3-VL-8B-Instruct (Bai et al., 2025a)</td><td>8B</td><td>84.9</td><td>59.6</td><td>69.3</td><td>64.5</td><td>78.1</td><td>61.4</td><td>60.8</td><td>61.1</td></tr><tr><td>LLaVA-1.5-13B (Liu et al., 2023c)</td><td>13B</td><td>48.8</td><td>11.4</td><td>16.7</td><td>14.1</td><td>49.0</td><td>23.6</td><td>24.7</td><td>24.2</td></tr><tr><td>ChartLlama (Han et al., 2023)</td><td>13B</td><td>52.3</td><td>15.8</td><td>27.4</td><td>21.6</td><td></td><td></td><td></td><td></td></tr><tr><td>ChartCoder (Zhao et al., 2025b)</td><td>7B</td><td>89.8</td><td>47.3</td><td>74.5</td><td>60.9</td><td>85.2</td><td>58.3</td><td>62.0</td><td>60.2</td></tr><tr><td>MSRL (Chen et al., 2026a)</td><td>7B</td><td>97.2</td><td>57.6</td><td>84.8</td><td>71.2</td><td>67.1</td><td>39.6</td><td>46.5</td><td>43.0</td></tr><tr><td>RRVF (Chen et al., 2025b)</td><td>7B</td><td>96.1</td><td>44.1</td><td>74.2</td><td>59.2</td><td>85.9</td><td>57.6</td><td>59.3</td><td>58.5</td></tr><tr><td>Qwen3-VL-4B-Instruct (Bai et al., 2025a)</td><td>4B</td><td>79.5</td><td>55.6</td><td>63.2</td><td>59.4</td><td>66.0</td><td>47.6</td><td>49.1</td><td>48.4</td></tr><tr><td>Qwen3-VL-4B-Thinking (Bai et al., 2025a)</td><td>4B</td><td>79.3</td><td>64.3</td><td>67.0</td><td>65.6</td><td>72.9</td><td>54.1</td><td>57.6</td><td>55.8</td></tr><tr><td>TinyChart (Zhang et al., 2024)</td><td>3B</td><td>36.3</td><td>18.3</td><td>25.1</td><td>21.7</td><td></td><td></td><td></td><td></td></tr><tr><td>ChartEditor (Chen et al., 2026b)</td><td>3B</td><td>66.5</td><td>40.2</td><td>55.3</td><td>47.8</td><td>61.6</td><td>52.1</td><td>57.9</td><td>55.0</td></tr><tr><td>Ours</td><td>8B</td><td>96.7</td><td>73.3</td><td>86.1</td><td>79.7</td><td>88.6</td><td>67.7</td><td>72.5</td><td>70.1</td></tr></table>

## 5 EXPERIMENT

## 5.1 BASELINES AND BENCHMARKS

We compare REChart against three groups of baselines: proprietary MLLMs including GPT-4o (Hurst et al., 2024), Claude-3.7-Sonnet (Anthropic, 2025), and Gemini-2.5-Pro (Comanici et al., 2025) as upper-bound references; general-purpose open-source MLLMs of comparable or larger scale, including the LLaVA-1.5-13B (Liu et al., 2023c), the Qwen3-VL series (Bai et al., 2025a) and GLM-4.1V-9B-Thinking (Hong et al., 2025); and domain-specialized models including ChartCoder (Zhao et al., 2025b), ChartEditor (Chen et al., 2026b), TinyChart (Zhang et al., 2024), and RRVF (Chen et al., 2025b). We evaluate on two benchmarks covering both chart editing and direct chart-to-code generation: ChartEdit w/o code (Zhao et al., 2025a), ChartMIMIC (Yang et al., 2025a) (customized mimic and direct mimic). Further details on evaluation configurations are provided in Appendix E.

## 5.2 TRAINING SETUP

We fine-tune Qwen3-VL-8B-Thinking (Bai et al., 2025a) as our base model across two stages. In the SFT stage, we train on 200k synthesized trajectories for one epoch using LLaMA-Factory (Zheng et al., 2024), with a learning rate of 5e-5 and an effective batch size of 256 via gradient accumulation, completing in approximately 28 hours on 8 NVIDIA A800 80GB GPUs. In the RL stage, we further train on 82k samples for one epoch using Verl (Sheng et al., 2025), with a GRPO group size of 8. For the hybrid reward, the three fidelity dimensions are weighted equally $( w _ { \mathrm { c o d e } } = w _ { \mathrm { s t r u c t } } = w _ { \mathrm { v i s } } = 0 . 3 3 )$ and the fidelity and efficiency rewards contribute equally to the final reward $( w _ { \mathrm { f i d } } = w _ { \mathrm { e f f } } = 0 . 5 )$ . This stage requires approximately 65 hours on 8 NVIDIA H100 GPUs.

## 5.3 MAIN RESULTS

Chart Editing. Table 2 reports results on ChartEdit w/o Code and ChartMIMIC Customized Mimic set. On ChartEdit, REChart achieves an overall score of 79.71, surpassing all open-source baselines, including models with substantially more parameters, such as Qwen3-VL-32B-Thinking (74.6) and Qwen3-VL-32B-Instruct (71.2). The gains are consistent across all evaluation dimensions, with

Table 4: Ablation studies on training and reward design. We start from the base model, add SFT, then apply base GRPO, budgeted rollout training, and the efficiency reward.
<table><tr><td rowspan="2">Variant</td><td colspan="2">ChartEdit</td><td colspan="2">ChartMIMIC Customized</td><td colspan="2">ChartMIMIC Direct</td></tr><tr><td>Exec.</td><td>Overall</td><td>Exec.</td><td>Overall</td><td>Exec.</td><td>Overall</td></tr><tr><td>Base</td><td>79.15</td><td>67.30</td><td>77.61</td><td>61.50</td><td>75.00</td><td>58.40</td></tr><tr><td>SFT</td><td>87.76</td><td>73.40</td><td>80.78</td><td>63.90</td><td>84.06</td><td>67.80</td></tr><tr><td>+ Base GRPO</td><td>95.66</td><td>79.60</td><td>89.61</td><td>70.66</td><td>94.94</td><td>77.59</td></tr><tr><td>+ Budgeted GRPO</td><td>95.87</td><td>80.24</td><td>86.67</td><td>68.76</td><td>96.00</td><td>78.96</td></tr><tr><td>+ Efficiency Reward</td><td>96.65</td><td>79.71</td><td>88.61</td><td>70.09</td><td>94.72</td><td>77.15</td></tr></table>

REChart attaining a code score of 73.28 and a chart score of 86.14, demonstrating improvements in both code-level correctness and chart-level fidelity. On ChartMIMIC Customized Mimic, REChart achieves an overall score of 70.09, outperforming all open-source baselines, including Qwen3-VL-32B-Thinking (69.6), GLM4.1V-9B-Thinking (64.7), and Qwen3-VL-8B-Thinking (61.5). Compared to domain-specialized baselines, REChart consistently outperforms prior chart-specific models such as ChartCoder, ChartEditor, and RRVF across both benchmarks. While proprietary models such as Gemini-2.5-Pro retain an overall advantage, REChart substantially narrows this gap as an 8B open-source model.

Table 3: Results on ChartMimic Direct Mimic task.
<table><tr><td rowspan="8">tion. Table 3 reports results on the ChartMIMIC Direct Mimic set, which evaluates chart-to-code reproduction without any editing instruction. REChart achieves an overall score of 77.15, outperforming all open-source baselines of comparable or larger scale, including ChartCoder (73.3),</td><td>Overall</td></tr><tr><td>Model Exec. Low High</td></tr><tr><td>93.2 79.0 83.5 81.2</td></tr><tr><td>GPT-4o (Hurst et al., 2024) GeminiProVision (Team et al., 2023) 68.2 53.8 53.3 53.6</td></tr><tr><td>Claude-3-opus (Anthropic, 2024) 83.3 60.5 60.1 60.3</td></tr><tr><td>61.8 34.4 38.9 36.6</td></tr><tr><td>InternVL2-8B (Chen et al., 2024) 59.7 20.7 21.3 21.0</td></tr><tr><td>LLaVA-Mistral-7B (Liu et al., 2023c) Qwen3-VL-32B-Thinking (Bai et al., 2025a) 82.8 65.2 72.2 68.7</td></tr><tr><td>Qwen3-VL-8B-Thinking (Bai et al., 2025a) 75.0 56.0 60.9 58.4</td></tr><tr><td>Qwen3-VL-4B-Thinking (Bai et al., 2025a) 69.8 51.5 56.0 53.7 65.7</td></tr><tr><td>Qwen3-VL-32B-Thinking and GLM4.1V-9B- GLM4.1v-9B-Thinking (Hong et al., 2025) 79.5 63.7 67.7</td></tr><tr><td>ChartEditor (Chen et al., 2026b) 69.6 52.7 56.4 54.5 73.3</td></tr><tr><td>Thinking (65.7). Among ChartCoder (Zhao et al., 2025b) 91.4 72.574.0 domain-specialized</td></tr><tr><td>models, RRVF (Chen et al., 2025b) 97.8 60.9 67.9 64.4 REChart outperforms Chart-</td></tr><tr><td>Ours 94.7 72.8 81.5 77.2 Coder (73.3), RRVF (64.4),</td></tr></table>

overall score. These results suggest that the visual reasoning and code generation capabilities acquired through our two-stage training generalize effectively beyond the chart editing setting.

Summary. Across the two benchmarks, our 8B model establishes the best chart-editing performance among open-source models of comparable scale and remains competitive on direct generation, demonstrating that REChart handles both editing and direct generation effectively.

## 5.4 ABLATION STUDY

We conduct ablation studies to validate the effect of SFT, reinforcement learning, budgeted rollout training, and the efficiency reward, with results reported on ChartEdit and ChartMIMIC benchmarks.

Training and Reward Ablation. Table 4 shows that SFT provides a clear first-stage gain over the base model, improving the overall score by 6.10 points on ChartEdit, 2.40 points on ChartMIMIC Customized Mimic, and 9.40 points on ChartMIMIC Direct Mimic. Reinforcement learning then provides another large performance gain over the SFT checkpoint across all three settings. Adding base GRPO improves the overall score by 6.20 points on ChartEdit, 6.76 points on ChartMIMIC Customized Mimic, and 9.79 points on ChartMIMIC Direct Mimic. Budgeted GRPO further improves ChartEdit and Direct Mimic performance, reaching 80.24 and 78.96 overall scores, respectively.

Table 5: Average reasoning token usage under a maximum thinking budget of 16,384 tokens, starting from the base model and cumulatively adding SFT and RL variants.
<table><tr><td>Variant</td><td>ChartEdit</td><td>ChartMIMIC Direct</td><td>ChartMIMIC Customized</td></tr><tr><td>Base</td><td>6521.12</td><td>9575.09</td><td>8889.43</td></tr><tr><td>+ SFT</td><td>2790.66</td><td>2267.25</td><td>2251.27</td></tr><tr><td>+ Base GRPO</td><td>3184.46</td><td>2320.49</td><td>2285.46</td></tr><tr><td>+ Budgeted GRPO</td><td>3040.28</td><td>1997.75</td><td>1995.39</td></tr><tr><td>+ Efficiency Reward</td><td>2005.18</td><td>1727.69</td><td>1515.22</td></tr></table>

Adding the efficiency reward preserves most of the GRPO-level performance improvement, with overall scores of 79.71 on ChartEdit, 70.09 on Customized Mimic, and 77.15 on Direct Mimic, while enabling the largest reduction in reasoning cost as shown below.

Efficiency Reward Ablation. Table 5 reports average reasoning token usage under a maximum thinking budget of 16,384 tokens. Starting from the base model, SFT alone already reduces average reasoning length from 8,328.55 to 2,436.39 tokens, a 70.8% reduction, showing that supervised trajectory construction improves reasoning efficiency before RL is applied. The efficiency reward further reduces token consumption across all three settings, lowering the average to 1,749.36 tokens, a 79.0% reduction relative to the base model. Compared with base GRPO, adding the efficiency reward reduces reasoning tokens by 37.0% on ChartEdit, 25.5% on Direct Mimic, and 33.7% on Customized Mimic. These results show that $R _ { \mathrm { e f f } }$ improves the efficiency-accuracy trade-off: it retains the large performance gains brought by GRPO while suppressing unnecessary late-stage reasoning.

## 6 CONCLUSION

In this work, we aim to improve LRMs toward both higher editing performance and better reasoning efficiency on chart editing. Our experiments show that reasoning indeed helps chart editing, but excessive reasoning leads to severe overthinking that ultimately degrades performance without control. Motivated by this, we propose REChart, a two-stage framework that supervises both the reasoning trajectory and its terminal output. We synthesize 200k Reason-Score-Refine trajectories via a multiagent pipeline for SFT, and then optimize the model with reinforcement learning under a hybrid reward combining multi-dimensional fidelity reward over code, structure, and visual faithfulness with an episode-level efficiency reward. On chart editing benchmarks, REChart achieves state-of-the-art performance among open-source models at a comparable scale while reducing reasoning cost by over 60% compared to the base model.

Limitations. Due to computational constraints, we fine-tune only an 8B base model, and scaling our framework to larger LRMs may yield further gains in both fidelity and reasoning efficiency. In addition, our study focuses exclusively on chart editing; extending the framework to broader visual code generation tasks such as HTML and SVG illustrations is a promising direction we leave to future work.

## AI USE STATEMENT

LLMs serve as integral components in our data construction pipeline, including chart filtering, code synthesis, editing instruction generation, and multi-agent reasoning trajectory synthesis (Sections 4.1.1 and 4.1.2); the specific models assigned to each role (Qwen3-VL-32B, Qwen3-VL-235B-A22B, Gemini 3 Flash) are detailed in the corresponding sections and the Appendix. Beyond these methodological uses, LLMs were also used solely for grammar correction and language polishing of the manuscript, with no involvement in ideation, experimental design, or derivation of the scientific contributions. We have reviewed all AI-assisted work and take responsibility for the final content of this work, including text, claims, code, data, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

We acknowledge the broader ethical implications of generative AI. Our chart corpus is built exclusively from CC BY 4.0 licensed arXiv articles, ensuring compliance with content licensing requirements. The dataset contains no personally identifiable information (PII) and has been screened to exclude offensive content. For the human audit conducted during data filtering (Section 4.1.1), all annotators participated voluntarily with informed consent and were compensated fairly; the audit involved only assessing visual fidelity and code-image consistency of synthesized chart-code pairs, posing no foreseeable risks to participants. The release of REChart and its associated dataset poses no foreseeable harm to society and does not facilitate the generation of malicious content. All base models used in our pipeline are publicly available, and we will release our code, data, and model checkpoints upon publication.

## REPRODUCIBILITY STATEMENT

To facilitate reproducibility, we provide a representative subset of the dataset, source code, and scripts in the supplemental material, with the full 200k SFT trajectory dataset and model checkpoints to be released on Hugging Face upon publication. Detailed experimental configurations, including agent prompts, reward formulations, training hyperparameters, and evaluation protocols, are provided in the Appendix.

## REFERENCES

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. In Second Conference on Language Modeling, 2025. URL https: //openreview.net/forum?id=4jdIxXBNve.

Pranjal Aggarwal, Seungone Kim, Jack Lanchantin, Sean Welleck, Jason E Weston, Ilia Kulikov, and Swarnadeep Saha. Optimalthinkingbench: Evaluating over and underthinking in LLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=N5kWa3sRJt.

Anthropic. Claude 3.7 sonnet. https://www.anthropic.com/news/ claude-3-7-sonnet, 2025.

AI Anthropic. Introducing the next generation of claude. https://www. anthropic. com/news/claude-3- family, 2024.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Vidhisha Balachandran, Jingya Chen, Lingjiao Chen, Shivam Garg, Neel Joshi, Yash Lara, John Langford, Besmira Nushi, Vibhav Vineet, Yue Wu, et al. Inference-time scaling for complex tasks: Where we stand and what lies ahead. arXiv preprint arXiv:2504.00294, 2025.

Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, and Lin Ma. Breaking the SFT plateau: Multimodal structured reinforcement learning for chart-to-code generation. In The Fourteenth International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=14qYxv2jki.

Liangyu Chen, Yichen Xu, Jianzhe Ma, Yuqi Liu, Donglu Yang, Liang Zhang, Zihao Yue, Wenxuan Wang, and Qin Jin. ChartEditor: A reinforcement learning framework for robust chart editing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 20199–20207, 2026b.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum? id=YfZ4ZPt8zd.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of long reasoning models. In Forty-second International Conference on Machine Learning, 2025a. URL https://openreview.net/ forum?id=MSbU3L7V00.

Yang Chen, Yufan Shen, Wenxuan Huang, Sheng Zhou, Qunshu Lin, Xinyu Cai, Zhi Yu, Jiajun Bu, Botian Shi, and Yu Qiao. Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback. CoRR, abs/2507.20766, 2025b. doi: 10.48550/ARXIV.2507.20766. URL https://doi.org/10.48550/arXiv.2507.20766.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Kenny Davila, Srirangaraj Setlur, David Doermann, Bhargava Urala Kota, and Venu Govindaraju. Chart mining: A survey of methods for automated chart analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):3799–3819, 2021. doi: 10.1109/TPAMI.2020.2992028.

Yunzhen Feng, Julia Kempe, Cheng Zhang, Parag Jain, and Anthony Hartshorn. What characterizes effective reasoning? revisiting length, review, and structure of cot. In NeurIPS 2025 Workshop on Efficient Reasoning, 2025. URL https://openreview.net/forum?id=8VNId3aihj.

Michael Friendly. A Brief History of Data Visualization, pp. 15–56. Springer Berlin Heidelberg, Berlin, Heidelberg, 2008. ISBN 978-3-540-33037-0. doi: 10.1007/978-3-540-33037-0\_2. URL https://doi.org/10.1007/978-3-540-33037-0\_2.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nat., 645(8081):633–638, 2025. doi: 10.1038/S41586-025-09422-Z. URL https://doi.org/10.1038/s41586-025-09422-z.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-budget-aware LLM reasoning. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 24842–24855, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1274. URL https://aclanthology.org/2025.findings-acl.1274/.

Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A multimodal llm for chart understanding and generation. arXiv preprint arXiv:2311.16483, 2023.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv.org/abs/ 2507.01006.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of LLMs via reinforcement learning. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/ forum?id=V51gPu1uQD.

Kung-Hsiang Huang, Hou Pong Chan, May Fung, Haoyi Qiu, Mingyang Zhou, Shafiq Joty, Shih-Fu Chang, and Heng Ji. From pixels to insights: A survey on automatic chart understanding in the era of large foundation models. IEEE Transactions on Knowledge and Data Engineering, 37(5): 2550–2568, 2025. doi: 10.1109/TKDE.2024.3513320.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Shankar Kantharaj, Rixie Tiffany Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq Joty. Chart-to-text: A large-scale benchmark for chart summarization. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4005–4023, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 277. URL https://aclanthology.org/2022.acl-long.277/.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, and Zhifang Sui. Selfbudgeter: Adaptive token allocation for efficient LLM reasoning. CoRR, abs/2505.11274, 2025. doi: 10.48550/ARXIV.2505. 11274. URL https://doi.org/10.48550/arXiv.2505.11274.

Fangyu Liu, Julian Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, and Yasemin Altun. DePlot: One-shot visual language reasoning by plot-to-table translation. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings ofthe Associationfor Computational Linguistics: ACL 2023, pp. 10381–10399, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.660. URL https://aclanthology.org/2023.findings-acl.660/.

Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, and Julian Eisenschlos. MatCha: Enhancing visual language pretraining with math reasoning and chart derendering. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12756–12770, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.714. URL https://aclanthology.org/2023.acl-long.714/.

Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. MMC: Advancing multimodal chart understanding with large-scale instruction tuning. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1287–1310, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.70. URL https://aclanthology.org/2024.naacl-long.70/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023c. URL https: //openreview.net/forum?id=w0H2xGHlkw.

Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=hj9eKpqxQl.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Findings of the Association for Computational Linguistics: ACL 2022, pp. 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.177. URL https://aclanthology.org/ 2022.findings-acl.177/.

Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, and Shafiq Joty. UniChart: A universal vision-language pretrained model for chart comprehension and reasoning. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 14662–14684, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.906. URL https://aclanthology.org/2023.emnlp-main.906/.

Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. ChartAssistant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 7775–7803, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.463. URL https://aclanthology.org/2024.findings-acl.463/.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple testtime scaling. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20275–20321, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1025. URL https: //aclanthology.org/2025.emnlp-main.1025/.

OpenAI. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https://doi.org/10.48550/arXiv.2412.16720.

Sundar Pichai, Demis Hassabis, and Koray Kavukcuoglu. A new era of intelligence with gemini 3, November 2025. URL https://blog.google/products-and-platforms/ products/gemini/gemini-3/#gemini-3.

Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta reinforcement finetuning. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=TqODUDsU4u.

Raian Rahman, Rizvi Hasan, Abdullah Al Farhad, Md. Tahmid Rahman Laskar, Md. Hamjajul Ashmafee, and Abu Raihan Mostofa Kamal. Chartsumm: A comprehensive benchmark for automatic chart summarization of long and short summaries. In Amílcar Soares, Farhana H. Zulkernine, Renata Dividino, Reihaneh Rabbany, Qiang Ye, David Beach, and Karim Ali (eds.), 36th Canadian Conference on Artificial Intelligence, Canadian AI 2023, Montreal, Canada, June 5-9, 2023, Proceedings. Canadian Artificial Intelligence Association, 2023. doi: 10.21428/ 594757DB.0B1F96F6. URL https://doi.org/10.21428/594757db.0b1f96f6.

Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. Codebleu: a method for automatic evaluation of code synthesis. CoRR, abs/2009.10297, 2020. URL https://arxiv.org/abs/2009.10297.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, EuroSys ’25, pp. 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400711961. doi: 10.1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, and Wai Lam. A thorough examination of decoding methods in the era of LLMs. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8601–8629, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.489. URL https://aclanthology.org/2024.emnlp-main.489/.

Zayne Rea Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. To cot or not to cot? chain-ofthought helps mainly on math and symbolic reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= w6nlcS8Kkn.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, Hanjie Chen, and Xia Hu. Stop overthinking: A survey on efficient reasoning for large language models. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=HvoG8SxggZ.

Wentao Tan, Qiong Cao, Chao Xue, Yibing Zhan, Changxing Ding, and Xiaodong He. Chartmaster: Advancing chart-to-code generation with real-world charts and chart similarity reinforcement learning. CoRR, abs/2508.17608, 2025. doi: 10.48550/ARXIV.2508.17608. URL https: //doi.org/10.48550/arXiv.2508.17608.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Kimi Team. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534, 2025a. doi: 10.48550/ ARXIV.2507.20534. URL https://doi.org/10.48550/arXiv.2507.20534.

Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025b. doi: 10.48550/ARXIV.2501.12599. URL https://doi.org/10.48550/arXiv.2501. 12599.

Qwen Team. Qwen3 technical report. CoRR, abs/2505.09388, 2025c. doi: 10.48550/ARXIV.2505. 09388. URL https://doi.org/10.48550/arXiv.2505.09388.

Priyan Vaithilingam, Elena L. Glassman, Jeevana Priya Inala, and Chenglong Wang. Dynavis: Dynamically synthesized ui widgets for visualization editing. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, CHI ’24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400703300. doi: 10.1145/3613904.3642639. URL https://doi.org/10.1145/3613904.3642639.

Jiaqi WANG, Kevin Qinghong Lin, James Cheng, and Mike Zheng Shou. Think or not? selective reasoning via reinforcement learning for vision-language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview. net/forum?id=qI95wZZCWh.

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. How far can camels go? exploring the state of instruction tuning on open resources. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=w4zZNC4ZaV.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum? id=\_VjQlMeSB\_J.

Chengyue Wu, Zhixuan Liang, Yixiao Ge, Qiushan Guo, Zeyu Lu, Jiahao Wang, Ying Shan, and Ping Luo. Plot2Code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific plots. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Findings of the Association for Computational Linguistics: NAACL 2025, pp. 3006–3028, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979- 8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl.164. URL https://aclanthology. org/2025.findings-naacl.164/.

Yuyang Wu, Yifei Wang, Ziyu Ye, Tianqi Du, Stefanie Jegelka, and Yisen Wang. When more is less: Understanding chain-of-thought length in LLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= 6QDFsYxtI1.

Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang T. Truong, Chelsea Finn, and Nick Haber. Just enough thinking: Efficient reasoning with adaptive length penalties reinforcement learning. CoRR, abs/2506.05256, 2025. doi: 10.48550/ARXIV.2506.05256. URL https: //doi.org/10.48550/arXiv.2506.05256.

Liwenhan Xie, Yanna Lin, Can Liu, Huamin Qu, and Xinhuan Shu. Datawink: Reusing and adapting svg-based visualization examples with large multimodal models. IEEE Transactions on Visualization and Computer Graphics, 32(1):824–834, 2026. doi: 10.1109/TVCG.2025.3634635.

Zhengzhuo Xu, Bowen Qu, Yiyan Qi, SiNan Du, Chengjin Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of diversely aligned expert connector for chart understanding. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=o5TsWTUSeF.

Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran XU, Xinyu Zhu, Siheng Li, Yuxiang Zhang, Gongye Liu, Xiaomei Nie, Deng Cai, and Yujiu Yang. Chartmimic: Evaluating LMM’s cross-modal reasoning capability via chart-to-code generation. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum?id=sGpCzsfd1K.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. CoRR, abs/2504.15895, 2025b. doi: 10.48550/ ARXIV.2504.15895. URL https://doi.org/10.48550/arXiv.2504.15895.

Donglu Yang, Liang Zhang, Zihao Yue, Liangyu Chen, Yichen Xu, Wenxuan Wang, and Qin Jin. ChartM<sup>3</sup>: Benchmarking chart editing with multimodal instructions. In Proceedings ofthe 33rd ACM International Conference on Multimedia, MM ’25, pp. 5001–5009, New York, NY, USA, 2025c. Association for Computing Machinery. ISBN 9798400720352. doi: 10.1145/3746027. 3755714. URL https://doi.org/10.1145/3746027.3755714.

Xingchen Zeng, Haichuan Lin, Yilin Ye, and Wei Zeng. Advancing multimodal large language models in chart question answering with visualization-referenced instruction tuning. IEEE Transactions on Visualization and Computer Graphics, 31(1):525–535, 2024.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. AdaptThink: Reasoning models can learn when to think. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 3716–3730, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.184. URL https: //aclanthology.org/2025.emnlp-main.184/.

Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. TinyChart: Efficient chart understanding with program-of-thoughts learning and visual token merging. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 1882–1898, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.emnlp-main.112. URL https://aclanthology.org/2024.emnlp-main.112/.

Xuanle Zhao, Xuexin Liu, Yang Haoyue, Xianzhen Luo, Fanhu Zeng, Jianling Li, Qi Shi, and Chi Chen. ChartEdit: How far are MLLMs from automating chart analysis? evaluating MLLMs’ capability via chart editing. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 3616–3630, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.185. URL https://aclanthology.org/2025.findings-acl.185/.

Xuanle Zhao, Xianzhen Luo, Qi Shi, Chi Chen, Shuo Wang, Zhiyuan Liu, and Maosong Sun. ChartCoder: Advancing multimodal large language model for chart-to-code generation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7333–7348, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.363. URL https://aclanthology.org/2025.acl-long.363/.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Yixin Cao, Yang Feng, and Deyi Xiong (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 400–410, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-demos.38. URL https://aclanthology.org/2024.acl-demos.38/.

Prompt for Stop Reasoning   
Time is up.   
I need to provide the final answer based on the existing considerations.  
Figure 5: Prompt for terminating the reasoning phase within the given thinking budget and generating the final output.

## A EXPERIMENTAL DETAILS OF THE PRELIMINARY STUDY

## A.1 MODEL CONFIGURATION

We follow the standard configuration adopted in prior work Wang et al. (2023); Shi et al. (2024) when querying the evaluated models. For the open-source models in our suite (the Qwen3-VL series and GLM-4.1v-9B), we set the sampling temperature to 0.1 and serve each model through the latest release of vLLM on a single NVIDIA A800 (80GB) GPU. For the proprietary Gemini 3.1 Pro, we set the temperature to 0 and query it via its official API endpoint. All other decoding parameters are left at their respective defaults.

## A.2 THINKING BUDGET IMPLEMENTATION

We implement the thinking budget through a simple two-stage decoding scheme commonly used in prior work (Team, 2025c; Qu et al., 2025). In the first stage, the model is allowed to reason freely with the maximum generation length set to the target thinking budget B. Once generation reaches B tokens, decoding is interrupted, and we append (i) an end-of-thinking prompt that instructs the model to stop reasoning and produce the final code (Figure 5) and (ii) the special end-of-thinking token </think>. In the second stage, the model continues decoding from this augmented context with a fixed maximum generation length of 4,096 tokens, and its output is taken as the final answer (i.e., the generated code). For the no-thinking baseline, we skip the first stage entirely and directly prompt the model with the end-of-thinking marker, so that decoding begins in the answer phase.

## A.3 EVALUATION METHOD

We adopt the official evaluation method of the ChartEdit benchmark (Zhao et al., 2025a). Each generated script is executed in a sandboxed Python environment, and its rendered output together with the source code is then scored by GPT-4o (Hurst et al., 2024) under the official ChartEdit evaluation prompt, which assesses correctness along multiple dimensions covering both code fidelity and visual fidelity, following ChartEdit (Zhao et al., 2025a). Samples whose generated code fails to execute are assigned a score of zero on all dimensions. The final score reported for each model and thinking budget is the average across all evaluation samples.

## A.4 ADDITIONAL REASONING SCALING ANALYSIS

Beyond the main preliminary analysis in Section 3, we further examine two factors that modulate the effective reasoning budget. Figure 6a compares the optimal thinking budget across models and shows that the best budget is not fixed across model families or scales. Figure 6b analyzes Qwen3-VL-8B across task difficulties and shows that harder examples can benefit from longer reasoning, although performance still declines once the model exceeds a useful budget. These observations support our choice to train models toward budget-aware and contribution-aware reasoning rather than simply encouraging longer chains of thought.

## A.5 TRAINING-INDUCED REASONING ERROR SHIFT

SFT already reduces all three error types relative to the base model. RL further lowers hallucination and intent drift, while slightly increasing details-missing errors relative to SFT. This suggests that later-stage training mainly suppresses spurious reasoning rather than uniformly improving every error category.

![](images/6ceafe054244c24a3fd072155edf4e6a95cb548d3932eb9fc94b5865988b4421.jpg)  
(a) Optimal thinking budget across models of different scale and capability.

![](images/d987331802fc96920a15363b729d7dfd2ff389b321fbb4ff50ec0adecaef6180.jpg)  
(b) Qwen3-VL-8B performance under different thinking budgets and task difficulties.

Figure 6: Additional analysis of reasoning budget sensitivity. (a) The optimal thinking budget differs across model scales and capabilities, suggesting that stronger models can often compress useful reasoning into shorter chains. (b) For Qwen3-VL-8B, the preferred budget also varies with task difficulty: harder editing tasks benefit from longer reasoning before overthinking begins.  
Table 6: Reasoning error counts on the ChartEdit human set before and after training.
<table><tr><td>Variant</td><td>Visual Hallucination</td><td>Edit Intent Drift</td><td>Details Missing</td></tr><tr><td>Base</td><td>125</td><td>16</td><td>73</td></tr><tr><td>+ SFT</td><td>105</td><td>13</td><td>66</td></tr><tr><td>+ RL</td><td>84</td><td>9</td><td>69</td></tr></table>

## B DATASET CONSTRUCTION DETAILS

## (1) Chart Collection and Filtering.

To cover the diversity of chart types and usage scenarios in real-world charts, we begin with roughly 244k arXiv papers published in 2024, of which around 100k carry a CC BY 4.0 license and form our source corpus. From their LaTeX sources, we extract about 854k figures spanning PNG, JPG, JPEG, and PDF formats. PDF figures are rasterized to PNG, and the remaining formats are normalized to PNG so that every input shares a uniform modality. We then apply Qwen3-VL-Flash to classify each figure as chart or non-chart, leaving about 622k chart images for downstream processing.

## (2) Chart-Code Pair Generation.

To generate image-code pairs at scale, we adopt a three-pass cascaded strategy that progressively assigns stronger generators to harder cases, balancing cost and quality. In each pass, a generator is prompted with the chart image (full prompt in Figure 7) to produce visualization code. The rendered output is then evaluated against the reference chart by a Qwen3-VL-32B-Instruct judge along four dimensions (readability, completeness, visual integrity, and consistency), following the rubric in Figure 8. Charts whose rendered code passes the judge are retained, while those that fail are forwarded to the next pass with a stronger generator. We started with Qwen3-VL-32B-Instruct, then handed over the failed cases to Qwen3-VL-235B-A22B. The most challenging cases that are still unresolved are then forwarded to Gemini 3 Flash. This process yields around 521k high-quality chart-code pairs in total.

## (3) Chart Editing Augmentation.

To extend the dataset to chart editing, we augment each chart-code pair through an editing-instruction synthesis pipeline driven by Qwen3-VL-32B-Instruct. For each pair, the model is prompted (Figure 9) to select an appropriate editing operation from the five editing categories defined in Section 3.1, compose a natural-language editing instruction, and produce the corresponding edited code. We repeat this process three times per pair to encourage operation and instruction diversity. Each candidate edit is then evaluated against the criteria in Figure 10, which rates the rendered edit on four dimensions (alignment with the instruction, integrity of unmodified regions, linguistic quality, and visual legibility), and is accepted only if it meets the required standards.

This step yields about 567k accepted editing samples. The remaining roughly 142k chart-code pairs, for which no editing candidate is accepted, are repurposed as direct-reproduction samples, where the instruction asks the model to reconstruct the reference chart and the ground-truth code generates that reference. The final dataset comprises around 700k image-instruction-code triplets spanning the six functional dimensions of Table 1, covering both direct generation and editing tasks. Detailed per-stage statistics, including the per-model contribution to the cascade and the distribution across editing categories, are reported in Appendix C.

## C DATASET STATISTIC

Table 7: Distribution of chart types in the Chart Edit dataset (by percentage).
<table><tr><td>Chart Type</td><td>Ratio (%)</td><td>Chart Type</td><td>Ratio (%)</td></tr><tr><td>line</td><td>39.74</td><td>area</td><td>4.70</td></tr><tr><td>scatters</td><td>14.97</td><td>graph</td><td>3.52</td></tr><tr><td>bar</td><td>8.70</td><td>contour</td><td>3.19</td></tr><tr><td>heatmap</td><td>8.50</td><td>hist</td><td>3.15</td></tr><tr><td>errorbar</td><td>6.99</td><td>density</td><td>2.12</td></tr><tr><td>box</td><td>1.43</td><td>quiver</td><td>0.77</td></tr><tr><td>scatter</td><td>0.66</td><td>violin</td><td>0.43</td></tr><tr><td>radar</td><td>0.37</td><td>pie</td><td>0.28</td></tr><tr><td>surface</td><td>0.17</td><td>other</td><td>0.25</td></tr><tr><td>errorpoint</td><td>0.08</td><td></td><td></td></tr></table>

Table 8: Statistics of visualization code generation models and their respective contribution to the final dataset. The table shows the number of chart-code pairs generated by each model and their proportion in the total dataset (N=521,338).
<table><tr><td>Model</td><td>Number of Charts</td><td>Proportion</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>118,577</td><td>22.74%</td></tr><tr><td>Qwen3-VL-253B-A22B</td><td>187,907</td><td>36.04%</td></tr><tr><td>Gemini 3 Flash</td><td>214,854</td><td>41.21%</td></tr><tr><td>Total</td><td>521,338</td><td>100%</td></tr></table>

Table 9: Distribution of edit types in the Chart Edit dataset (by percentage).
<table><tr><td>Task Type</td><td>Ratio (%)</td><td>Task</td><td>Ratio (%)</td></tr><tr><td>Axes &amp; Annotations</td><td>27.11</td><td>Layout &amp; Spatial Organization</td><td>10.02</td></tr><tr><td>Content Manipulation &amp; Data Logic</td><td>20.68</td><td>Style &amp; Aesthetics</td><td>12.68</td></tr><tr><td>Chart Transformation</td><td>11.61</td><td>direct</td><td>17.91</td></tr></table>

## C.1 MULTI-AGENT SYSTEM CONFIGURATION

Table 11 lists the backbone model assigned to each agent in the trajectory synthesis pipeline. The Reasoner and Generator agents share the backbone we use for downstream SFT (Qwen3-VL-8B-Thinking), so that the synthesized reasoning episodes lie within the distribution the student model is best able to internalize during fine-tuning. The remaining three agents serve only as data-construction supervisors and are powered by the stronger Qwen3-VL-32B-Instruct, which provides more reliable planning, critique, and trajectory finalization.

Table 10: Pipeline statistics for image-instruction-code triplet construction. Each row reports the count of items remaining after the corresponding stage.
<table><tr><td>Stage</td><td>Count</td></tr><tr><td>arXiv papers (2024)</td><td>244,029</td></tr><tr><td>Papers under CC BY 4.0 license</td><td>100,180</td></tr><tr><td>Figures extracted from LATEX sources Chart images after classification</td><td>854,484</td></tr><tr><td>Chart-code pairs after cascade filtering</td><td>622,672</td></tr><tr><td>Editing samples accepted by audit</td><td>521,338</td></tr><tr><td>Direct-reproduction samples (no edit accepted)</td><td>567,049</td></tr><tr><td>Total image-instruction-code triplets</td><td>142,735 709,784</td></tr></table>

Prompt for Code Generation   
Act as an expert data scientist. Convert the provided chart image into high  
fidelity Python code (using Matplotlib or Seaborn) that perfectly replicates its vi  
sual style, data trends, and layout. You must strictly initialize the plot using   
plt.figure(figsize=({width}, {height})) to match the required dimensions.   
Ensure all labels, legends, and LaTeX scientific notations are preserved, and provide only the   
executable code block.  
Figure 7: Prompt for code reproduction based on the given image from the arXiv papers.

## D SFT TRAJECTORY SYNTHESIS ANALYSIS

## D.1 TRAJECTORY COMPONENT ABLATION

The planner and critic are complementary. Keeping both modules yields the strongest pass@k and the lowest token cost, while removing either module hurts trajectory quality. In particular, the planner is most important for reaching high pass@k, whereas the critic mainly helps the trajectory recover from unproductive reasoning.

## D.2 SYNTHESIS COST

The synthesis pipeline is compute-intensive but manageable with two coordinated machines. The planner, critic, and finalizer run on A800-class hardware, while the reasoner and generator occupy H100-class GPUs, and the scorer dominates the throughput budget.

## D.3 SFT FILTERING STATISTICS

These statistics show that the editing portion is harder to retain through the cascade, while directreproduction samples have a slightly higher pass rate. Across both splits, the synthesis loop averages about 5.4 rounds before producing an accepted trajectory.

## D.4 EFFECT OF SFT DATA SCALE

The full SFT set gives the strongest downstream results on both benchmarks. Performance improves steadily as more synthesized trajectories are added, indicating that the data scaling effect is still positive even after 150k samples, although the gains become smaller at the largest scale.

Prompt for Image-Code Pair Evaluation   
You are a data visualization expert and a QA engineer.   
Task:   
Analyze the provided chart image and its corresponding code. Identify any quality issues that   
make this data pair unsuitable for a high-quality machine learning dataset.   
Evaluation Criteria:   
Readability: Are labels, titles, or legends overlapping or cut off?   
Completeness: Are the axes, labels, and legends present and correct?   
Visual Integrity: Is the scaling appropriate? Is there any glitching in the rendering?   
Consistency: Does the visual representation in the image accurately reflect the logic in the   
code?   
Output Format:   
You should first analyze the entire chart, then respond in the following JSON format, evaluat  
ing based on the four dimensions mentioned above:   
{   
"Readability": { "Issues": [ ... ], "Score": int,   
"Status": bool },   
"Completeness": ...,   
"Visual": ...,   
"Consistency": ...   
}   
Among the evaluation of a specific dimension:   
- Issues: List specific issues, or an empty array [].   
- Score: 1-10, where 10 is production-ready.   
- Status: true represents Pass, while false represents Fail.  
Figure 8: Prompt for evaluating chart-code pairs during dataset construction. The prompt instructs the MLLM (Multimodal Large Language Model) assessor to check quality across four key dimensions: readability, completeness, visual integrity, and consistency. Each dimension is scored 1-10, with a binary pass/fail status. This quality filtering ensures the dataset contains only high-fidelity image-code pairs.

Table 11: Backbone model assignment for each agent in the trajectory synthesis pipeline.
<table><tr><td>Agent</td><td>Symbol</td><td>Backbone</td></tr><tr><td>Planner</td><td>P</td><td>Qwen3-VL-32B-Instruct</td></tr><tr><td>Reasoner</td><td>T</td><td>Qwen3-VL-8B-Thinking</td></tr><tr><td>Generator</td><td>g</td><td>Qwen3-VL-8B-Thinking</td></tr><tr><td>Critic</td><td>C</td><td>Qwen3-VL-32B-Instruct</td></tr><tr><td>Finalizer</td><td>F</td><td>Qwen3-VL-32B-Instruct</td></tr></table>

## E EXPERIMENTAL DETAILS

## E.1 EVALUATION CONFIGURATION

Following prior work (Wang et al., 2023; Shi et al., 2024), we adopt the following decoding configu rations for all evaluations. For open-source models, we set the temperature to 0.1 and top-p to 1. For proprietary models, we set the temperature to 0 and top-p to 1, with a maximum output length of 4096 tokens. For all open-source reasoning models, we set the thinking budget to 4096 tokens and the maximum output length for final code generation to 4096 tokens; the implementation of thinking budget control follows the protocol described in Appendix A.2. All open-source models are served via the vLLM inference library on NVIDIA A800 80GB GPUs.

Table 12: Ablation of the Reason-Score-Refine synthesis pipeline. We evaluate whether the planner and critic are both needed for effective trajectory construction.
<table><tr><td>Planner</td><td>Critic</td><td>Pass@3</td><td>Pass@6</td><td>Pass@9</td><td>Avg. Token</td></tr><tr><td rowspan="2">√</td><td>V</td><td>24.75%</td><td>35.17%</td><td>42.58%</td><td>2086.33</td></tr><tr><td>√</td><td>24.00%</td><td>35.75%</td><td>42.58%</td><td>2416.21</td></tr><tr><td rowspan="2">V</td><td></td><td>16.58%</td><td>22.00%</td><td>24.75%</td><td>2200.22</td></tr><tr><td></td><td>15.33%</td><td>19.75%</td><td>23.42%</td><td>2300.86</td></tr></table>

Table 13: Estimated throughput and compute footprint of the SFT synthesis pipeline.
<table><tr><td>Module</td><td>Hardware</td><td>Throughput</td></tr><tr><td>Planner</td><td>1×A800</td><td>3500/h</td></tr><tr><td>Reasoner</td><td>1×H100</td><td>4000/h</td></tr><tr><td>Generator</td><td>1×H100</td><td>4000/h</td></tr><tr><td>Scorer</td><td>64 Core GPU + 512GB Memory</td><td>9000/h</td></tr><tr><td>Critic</td><td>1×A800</td><td>18000/h</td></tr><tr><td>Finalizer</td><td>1×A800</td><td>9000/h</td></tr><tr><td>Total</td><td>Running on two machines (8×A800+64C CPU + 512GB Mem; 8×H100+64C CPU + 512GB Mem)</td><td>About 2 weeks (380 hours)</td></tr></table>

## E.2 BENCHMARK DESCRIPTIONS

ChartEdit (Zhao et al., 2025a). ChartEdit is a chart editing benchmark that requires models to modify an existing chart according to a natural language instruction, without access to the original source code. Each sample provides a reference chart image and an editing instruction, and the model is expected to generate executable visualization code that reflects the requested changes. Performance is assessed across three complementary dimensions: code-level correctness, which measures the syntactic and structural validity of the generated code; chart-level fidelity, which evaluates the visual alignment between the generated and target charts; and execution-level success, which checks whether the code runs without errors. An official overall score aggregates these dimensions into a single metric.

ChartMIMIC (Yang et al., 2025a). ChartMIMIC evaluates models on two distinct tasks. The Customized Mimic task requires instruction-conditioned chart editing, where the model must adapt a reference chart according to a given instruction; samples are divided into Low and High difficulty tiers based on the complexity of the required edits. The Direct Mimic task measures chart-to-code reproduction ability without any editing instruction, providing a complementary assessment of the model’s direct chart generation capability. Both tasks report execution success rate and an overall score computed from visual and structural similarity between the generated and reference charts.

Table 14: Statistics of the SFT data synthesis pipeline after filtering.
<table><tr><td>Task Type</td><td>Total Input</td><td>Filtered</td><td>Final Output</td><td>Pass Rate</td><td>Avg. Round Num</td></tr><tr><td>Edit Task</td><td>562,485</td><td>383,867</td><td>178,618</td><td>31.76%</td><td>5.39</td></tr><tr><td>Direct Task</td><td>75,275</td><td>45,957</td><td>29,318</td><td>38.95%</td><td>5.42</td></tr><tr><td>Total</td><td>637,760</td><td>429,824</td><td>207,936</td><td>32.60%</td><td>5.40</td></tr></table>

Table 15: Effect of SFT dataset size on downstream performance.
<table><tr><td>SFT Dataset Size</td><td colspan="2">ChartEdit w/o Code</td><td colspan="2">ChartMIMIC Direct</td></tr><tr><td></td><td>Exec. Rate</td><td>Overall</td><td>Exec. Rate</td><td>Overall</td></tr><tr><td>base</td><td>79.15</td><td>67.30</td><td>75.00</td><td>58.45</td></tr><tr><td>50k</td><td>83.56</td><td>68.76</td><td>79.06</td><td>61.75</td></tr><tr><td>100k</td><td>84.48</td><td>69.12</td><td>80.72</td><td>63.58</td></tr><tr><td>150k</td><td>86.98</td><td>71.01</td><td>82.11</td><td>65.05</td></tr><tr><td>full</td><td>87.76</td><td>73.43</td><td>84.06</td><td>67.78</td></tr></table>

![](images/b0ce7b6ab80f0af8c03d41be98d800b48e6a741ebb622aa9718e8c37bbe47bcd.jpg)  
Figure 9: Prompt for generating chart editing instructions, which defines the role, inputs, task, domain-specific categories, and structured output format. This prompt is used to guide a multimodal LLM in producing high-quality instruction-code pairs for fine-tuning.

```jsonl
System Prompt for Visual Editing Quality Audit
You are a Senior Data Visualization Auditor and Quality Control Expert for AI training
datasets. Your task is to evaluate the quality of a "Chart Edit" pair consisting of an instruction
and the resulting visual transformation.
Evaluation Criteria
Please rate the following dimensions on a scale of 1 to 25 (1 = Poor, 25 = Excellent):
1. Alignment (Faithfulness): Does the Modified Image accurately reflect the Edit Instruction?
- 25: Perfectly executed as requested.
- 1: The change is unrelated to the instruction or factually wrong.
2. Integrity (Minimal Disturbance): Were non-requested elements preserved?
- 25: Only the requested parts were changed; all other data and styles remain consistent.
- 1: Significant, unintended changes occurred (e.g., data points shifted, titles vanished).
3. Linguistic Quality: Is the instruction natural, clear, and human-like?
- 25: Sounds like a real user; professional and clear.
- 1: Robotic, repetitive, or contains logical contradictions.
4. Visual Legibility: Is the resulting chart professional and readable?
- 25: Clear, no overlapping text, appropriate contrast, and professional aesthetics.
- 1: Broken layout, overlapping elements, or illegible fonts.
Output Format (Strict JSON)
Analyze the visual differences between the two images first, then provide the scores and brief
comments in json format.
{
"alignment": { "comment": ${comment}, "score": ${subscore} },
"integrity": { "comment": ${comment}, "score": ${subscore} },
"linguistic": { "comment": ${comment}, "score": ${subscore} },
"legibility": { "comment": ${comment}, "score": ${subscore} },
}
```  
Figure 10: System prompt for the quality audit of a visual editing process. The prompt defines the auditor’s role, required inputs, evaluation criteria, with a detailed 1-25 scoring scale, and a strict JSON output format. The auditor must assess faithfulness, integrity, linguistic quality, and visual legibility of the edit.