# RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing

Zijian Kan<sup>1,2</sup>, Wei Wang<sup>2</sup>, Long Luo<sup>2</sup>, Bing Zhao<sup>3,∗</sup>, Xuan Ren<sup>2</sup>, Weixu Qiao<sup>2</sup>, Wenbo Li<sup>2</sup>, Hu Wei<sup>2</sup>, Lin Qu<sup>2</sup>

<sup>1</sup>Peking University <sup>2</sup>Alibaba Group <sup>3</sup>Alibaba DAMO Academy kanzijian26@stu.pku.edu.cn xiongdao@alibaba-inc.com

## Abstract

Reward models play an essential role in aligning visual generative models, yet most existing visual reward models use a single scalar score or rely on fixed criteria that cannot adapt to diferent instructions. This limits both interpretability and task sensitivity, especially for text-to-image generation and instructionbased image editing, where diferent inputs require diferent evaluation dimensions. We propose RubricRM, a pairwise generative reward modeling framework that first produces an input-specific rubric with evaluation dimensions, weights, and scoring criteria, and then applies the rubric to score candidate images. We train dedicated RubricRM models for text-to-image generation and image editing using a two-stage training pipeline: supervised fine-tuning teaches the model the rubric-based scoring paradigm, while GRPO further improves scoring through fine-grained dimension-level rewards. Experiments on multiple generation and editing benchmarks show that RubricRM outperforms existing specialized reward models and remains competitive with strong proprietary MLLM judges despite using smaller backbones. Our models, data, and code are available at https://github. com/zijiankan/RubricRM.

## 1 Introduction

Reward models are essential for the post-training alignment of visual generative models, as they convert human preferences into learnable reward signals (Ouyang et al., 2022; Black et al., 2024; Fan et al., 2023; Wallace et al., 2024).

Existing reward models fall into several categories. CLIP-based metrics (Hessel et al., 2021) ofer eficient image-text alignment signals yet exhibit limited sensitivity to fine-grained compositional semantics. Models such as ImageReward (Xu et al., 2023), HPS v2 (Wu et al., 2023b,a),

HPS v3 (Ma et al., 2025), PickScore (Kirstain et al., 2023), and EditReward (Wu et al., 2025) are trained on large-scale preference data but produce scalar scores that ofer little interpretability. Recent generative reward models employ multimodal large language models to generate evaluation rationales (Ku et al., 2024; Wang et al., 2026b). Yet these approaches typically adopt predefined and fixed evaluation dimensions, whereas diferent instructions naturally demand distinct assessment criteria. For example, realistic prompts prioritize physical plausibility, whereas creative prompts value novelty, highlighting the need for adaptive evaluation rubrics.

To address this limitation, we introduce automated rubric generation for evaluating text-toimage generation and image editing. Our central idea is to use a rubric as an intermediate evaluation protocol between the instruction and the preference decision. The rubric specifies what aspects should be evaluated, how much each aspect matters, and how each score level should be interpreted, allowing the evaluation criteria to adapt to the input while keeping the scoring process inspectable.

Based on this concept, we propose Rubric-based Reward Model (RubricRM), a generative reward modeling framework for text-to-image generation and image editing. Given an instruction and two candidate outputs, RubricRM constructs an inputspecific rubric covering task intent, atomic evaluation dimensions, dimension weights, and scoring criteria. It then scores both candidates along each dimension and aggregates the weighted dimension scores to derive the final preference.

RubricRM is trained via a two-stage pipeline that targets two capabilities: learning the rubricbased scoring pattern and refining scoring accuracy. In the supervised fine-tuning stage, the model imitates teacher-synthesized rubric trajectories to learn how to decompose an instruction into intent, dimensions, weights, and graded criteria. In the subsequent reinforcement learning stage, dimension-level rewards align the model’s predicted score diferences with reference score diferences on each rubric dimension and apply stronger penalties when the preference direction is reversed.

![](images/93f2f1b4c721ba4003bfc4beed6a3870614522257909c6852b8f4b91d3198db1.jpg)  
Figure 1: Inference and training pipeline of RubricRM. (A) RubricRM supports both text-to-image and imageediting preference selection. At inference time, the model first dynamically generates a rubric conditioned on the input, including scoring criteria and corresponding weights, and then scores each image along each dimension based on the rubric, all within a single inference pass. (B) Training is conducted in two stages. The model is first supervised fine-tuned on synthesized rubric data to learn the paradigm of rubric generation followed by scoring, and is then further optimized with GRPO using dimension-level rewards.

Learning dynamic rubrics requires broad data coverage, since a narrow training distribution can bias the learned rubrics and impair generalization. To this end, we collect multiple open-source preference datasets and analyze instruction types in text-to-image generation and image editing. We design prompt and editing-instruction classification criteria to support taxonomy annotation, balanced sampling, and the identification of underrepresented categories. For these categories, we synthesize supplementary prompts, generate candidate images, and collect expert preference annotations. We then synthesize rubric trajectories for each preference sample, yielding 31.8K text-toimage and 30.7K image-editing training samples.

We train RubricRM on Qwen3.5-4B and Qwen3.5-9B backbone models (Qwen Team, 2026). For text-to-image generation, RubricRM-Gen-9B attains accuracies of 72.00, 74.12, and 84.45 on MMRB2, GenAI-Bench, and GenAI-Bench-Verified, respectively. For image editing, RubricRM-Edit-9B reaches 75.40, 46.40, and 85.64 on MMRB2, EditReward-ERB, and EditScore-ERB, outperforming existing reward-model baselines across both tasks.

In summary, our key contributions are as follows:

• We introduce a dynamic rubric-based reward modeling paradigm for text-to-image generation and image editing and develop specialized models for both tasks.

• We construct rubric trajectories from preference data and train RubricRM with a two-stage procedure that combines rubrictrajectory SFT with dimension-level GRPO.

• We obtain strong results on multiple reward benchmarks, outperforming specialized reward-model baselines across generation and editing tasks.

## 2 Related Work

Text-to-image generation evaluation. Text-toimage evaluation has moved from global alignment metrics to preference-aligned reward modeling. CLIPScore (Hessel et al., 2021) measures imagetext similarity with CLIP (Radford et al., 2021), but it is unreliable for compositional semantics (Huang et al., 2023; Jiang et al., 2024). Preference-trained models such as ImageReward (Xu et al., 2023), HPS v2 (Wu et al., 2023b,a), HPS v3 (Ma et al.,

2025), and PickScore (Kirstain et al., 2023) improve human alignment, but their standard interface is still a scalar reward, which provides limited evidence for the decision. VisionReward (Xu et al., 2026) introduces multi-dimensional scoring, but the dimensions and weights are predefined rather than generated from the instruction.

Recent generative reward models use MLLMs to produce evaluative reasoning: UnifiedReward (Wang et al., 2025) supports multiple multimodal reward tasks, and UnifiedReward-Think (Wang et al., 2026b) adds explicit reasoning while relying on fixed scoring criteria. UnifiedReward-Flex (Wang et al., 2026a) further introduces personalized, context-adaptive assessment for vision generation, but it does not explicitly specify the detailed requirements of each scoring criterion or assign weights to the criteria.

Image editing evaluation. Instruction-guided image editing (Brooks et al., 2023) sharpens the need for adaptive criteria: an edit must satisfy the instruction without corrupting irrelevant content. EditReward (Wu et al., 2025) learns from over 200K editing preference pairs, but remains a discriminative scalar reward model. EditScore (Luo et al., 2025) builds high-fidelity generative reward models and uses them for online RL in image editing, but it does not define instructionspecific rubrics with explicit weights and scoring levels. RubricRM targets this layer between freeform rationale and scalar preference, where the model must decide which aspects of edit execution, source preservation, and visual quality should be emphasized for each instruction.

Rubric-based evaluation. Rubrics have recently emerged as a mechanism for making evaluation criteria explicit. AutoRubric (Rao and Callison-Burch, 2026) generates task-adapted rubrics, RubricHub (Li et al., 2026) constructs large-scale rubric resources, and OpenRubrics (Liu et al., 2025) studies scalable synthetic rubric generation for reward modeling and alignment. In multimodal settings, AutoRubric-R1V (Jia et al., 2025) uses rubric-based process rewards for multimodal reasoning RL, while Proxy-GRM (Qiu et al., 2026) learns transferable rubrics for visionlanguage response preference judgment through proxy-guided critique. AutoRubric-T2I (Kao et al., 2026) learns a compact, globally shared set of weighted natural-language rules from preference data; an external frozen VLM scores each image against these rules, whose scores are aggregated for text-to-image evaluation and downstream generation-model RL. Auto-Rubric as Reward (Tian et al., 2026) instead uses frozen VLMs to generate and verify prompt-conditioned binary criteria, then makes a holistic rubric-conditioned preference decision for text-to-image generation and image editing. Unlike these approaches, RubricRM trains a standalone, self-contained generative reward model that produces input-specific dimensions, normalized weights, graded scoring criteria, and explicit per-dimension scores for both candidates in a single end-to-end inference pass.

## 3 RubricRM

This section first defines the rubric-based visual reward modeling task, then describes the rubric trajectory data, and finally presents the two-stage training procedure.

## 3.1 Task Definition

We formally define the task of rubric-based visual reward modeling as follows. For text-to-image generation, the input consists of a text prompt q and two candidate images $I _ { A } , I _ { B }$ . For image editing, $q$ additionally includes a source image $I _ { \mathrm { s r c } }$ and an editing instruction. The goal is to predict a preference label $y \in \{ A , B \}$ indicating which image better satisfies the given instruction.

RubricRM produces a structured rubric together with dimension-level assessments for both candidates in a single forward pass. Formally, given q and the two candidate images, the model outputs a rubric

$$
\mathcal { R } = \{ ( d _ { i } , w _ { i } , \mathcal { L } _ { i } ) \} _ { i = 1 } ^ { N }\tag{1}
$$

and, for each image $I _ { k }$ with $k \in \{ A , B \}$ , a set of dimension scores $\{ s _ { i } ^ { k } \} _ { i = 1 } ^ { N }$ . Here $N$ is the number of atomic evaluation dimensions, $d _ { i }$ is the description of the i-th dimension, $w _ { i } \in [ 0 , 1 ]$ is its relative weight with $\begin{array} { r } { \sum _ { i = 1 } ^ { N } w _ { i } = 1 } \end{array}$ , and $\mathcal { L } _ { i }$ gives the specific scoring criteria for this dimension. Each $s _ { i } ^ { k } \in [ 0 , 4 ]$ is determined by referencing $d _ { i }$ and $\mathcal { L } _ { i }$ The rubric thus serves as an interpretable blueprint that specifies what to evaluate, how important each aspect $i s ,$ and what each score level means.

The overall score for image $I _ { k }$ is then computed as the weighted sum

$$
S ( I _ { k } ) = \sum _ { i = 1 } ^ { N } w _ { i } \cdot s _ { i } ^ { k }\tag{2}
$$

The final preference decision is made by comparing $S ( I _ { A } )$ and $S ( I _ { B } )$ , with the higher-scoring image being selected as the preferred one. Complete running cases for text-to-image generation and image editing are provided in Appendices D.1 and D.2.

## 3.2 Rubric Trajectory Data

Classification-Driven Balanced Sampling. RubricRM relies on dynamically generated scoring criteria, so the training data should cover diverse instruction types and visual scenarios. To improve coverage, we design a classificationdriven balanced sampling strategy.

Specifically, we analyze the subtasks in text-toimage generation and image editing, and develop a set of prompt classification criteria from multiple dimensions. We then employ DeepSeek-V3.2 for automatic category annotation. Since the same image may belong to diferent categories when viewed from diferent dimensions (e.g., a portrait photo falls under “portrait” from the object perspective, but may be classified as “close-up” from the photographic style perspective), the model is allowed to return multiple labels for each text prompt or editing instruction. This classification strategy serves three purposes: guiding balanced sampling from open-source data, identifying underrepresented categories for targeted supplementary synthesis, and facilitating the split of SFT and RL data.

During data splitting, most text-to-image samples carry multiple labels and cannot be assigned to a single category. We therefore represent each sample as a set of labels and adopt multi-label stratified sampling to balance coverage across secondlevel labels. Table 1 summarizes the training data. Appendix A.1 gives additional details on the RL split construction.

Text-to-Image Preference Data. The text-toimage data comprise open-source datasets and our additionally annotated data. Each sample consists of a generation prompt, two candidate images, and a preference label. The open-source data are sourced from HPD v3 (Ma et al., 2025), $\mathrm { O I P ^ { 1 } }$ and EvalMuse-40K (Han et al., 2026). We observed that some categories are underrepresented in the open-source data. Following the prompt synthesis procedure of Wang et al. (2026c), we use DeepSeek-V3.2 to synthesize supplementary prompts for these categories and manually screen the generated prompts for quality. We then render the retained prompts with ten image generation models, including GPT-Image, Gemini-3.1- Pro, Seedream-5.0, and Wan-2.6. Three human experts rate each image on a 10-point scale, and we retain only pairs whose score diference exceeds 5 points with unanimous preference agreement. The final text-to-image training set contains 31,835 samples.

Table 1: Training data statistics for rubric learning. P/S denotes primary/secondary labels.
<table><tr><td>Metric</td><td>Text-to-Image</td><td>Image Editing</td></tr><tr><td>#Pairs</td><td>31,835</td><td>30,695</td></tr><tr><td>SFT/RL</td><td>16,835/15,000</td><td>15,695/15,000</td></tr><tr><td>Taxonomy (P/S)</td><td>6/42</td><td>12/30</td></tr><tr><td>Avg. tags</td><td>3.3</td><td>1.4</td></tr><tr><td>Multi-tag (%)</td><td>94.9</td><td>31.8</td></tr><tr><td>Avg. dim.</td><td>3.9</td><td>3.2</td></tr><tr><td>A/B (%)</td><td>51.6/48.4</td><td>55.7/44.3</td></tr></table>

Image Editing Preference Data. The image editing data are built upon EditReward-Data (Wu et al., 2025). Each sample contains a source image, an editing instruction, two candidate edited results, and a preference label. Since EditReward-Data already provides relatively balanced instruction coverage, we do not synthesize additional editing prompts. The final image editing training set comprises 30,695 samples.

Distribution Analysis. Classification results show that in text-to-image generation, portraits and realistic photographic styles account for a relatively high proportion, whereas categories such as text rendering (6.45%), logical reasoning (8.98%), and world knowledge (15.83%) are less prevalent. Our balanced sampling strategy is designed to retain coverage of these long-tail categories. In image editing, object and attribute editing are the most common, while reasoningbased editing (8.70%), text editing (11.17%), and style editing (10.18%) appear less frequently. Figure 2 visualizes primary-category coverage for both tasks, and Tables 6–7 list the complete second-level taxonomy coverage.

Rubric Trajectory Synthesis. Training RubricRM requires jointly learning rubric generation and dimension-level scoring, yet preference labels alone are insuficient to provide these supervision signals. Since manually authoring full rubrics with scoring for large-scale data does not scale, we employ Gemini 3.1 Pro as the teacher model to synthesize training trajectories.

We use label-conditioned trajectory synthesis. To prevent synthesized trajectory quality from being bottlenecked by the teacher model’s standalone preference judgment, we provide the human preference label $y _ { j }$ to the teacher together with the prompt and candidate images. Thus, the teacher model is not used as an independent preference annotator; instead, it converts an existing human preference label into a structured rubric trajectory that supervises both criterion formulation and dimension-level scoring. The label anchors the trajectory to the human preference while still requiring the teacher model to specify evaluation dimensions, assign weights, define scoring criteria, and provide dimension-level evidence.

Concretely, given the text prompt $q _ { j }$ , the candidate images $I _ { A } ^ { j } , I _ { B } ^ { j } ,$ , and the preference label $y _ { j }$ the teacher model is prompted to generate four fields: task intent analysis, scoring dimensions with weight assignments, graded scoring descriptors for each dimension, and dimension-level assessments with scores for both candidate images.

Trajectory Quality Control. We apply a twostep quality control procedure after synthesis. First, we conduct structured parsing and discard trajectories with missing dimensions, out-of-range scores, weights that do not sum to 100%, or those from which a final preference cannot be parsed. Second, we verify that the weighted total scores of the synthesized trajectory yield a preference direction consistent with the label $y _ { j } ;$ if the teacher model’s final preference $\hat { y } _ { j }$ contradicts the label, the sample is removed.

To characterize the synthesized rubrics, $\mathsf { A p - }$ pendix A.1 further aggregates teacher-generated evaluation dimensions into semantic groups for text-to-image generation and image editing in Tables 8 and 9. Appendix B.2 examines rubric stability, diversity, and instruction-type adaptation, while Appendix B.3 reports a controlled analysis of sensitivity to the teacher family.

## 3.3 Two-Stage Training

We train RubricRM on the Qwen3.5-4B and Qwen3.5-9B backbone models for text-to-image generation and image editing, respectively. Both tasks follow the same training methodology: first, supervised fine-tuning enables the model to acquire the rubric-based scoring paradigm; then, reinforcement learning with fine-grained dimensionlevel rewards further refines the scoring accuracy.

## 3.3.1 Stage 1: Supervised Fine-Tuning

Let $\mathcal { D } _ { \mathrm { S F T } } = \{ ( x _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M _ { \mathrm { S F T } } }$ denote the filtered SFT dataset. Each input $\overset { \cdot } { x _ { j } } = ( q _ { j } , I _ { A } ^ { j } , I _ { B } ^ { j } )$ consists of a text prompt $q _ { j }$ and two candidate images, and $t _ { j }$ is the complete target trajectory synthesized by the teacher model, which sequentially contains task intent analysis, scoring dimensions with weights, graded scoring descriptors for each dimension, and dimension-level assessments with scores for both candidate images.

The model is optimized using the following loss function:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { j } \sum _ { t = 1 } ^ { T _ { j } } \log \pi _ { \theta } ( t _ { j , t } \mid x _ { j } , t _ { j , < t } )\tag{3}
$$

The core objective of this stage is for the model to learn the complete rubric-basedjudgment pipeline: decomposing an instruction into intent, selecting appropriate scoring dimensions and assigning weights, defining scoring criteria for each dimension, and producing dimension-level analysis leading to a final preference decision.

## 3.3.2 Stage 2: GRPO with Dimension-Level Rewards

Motivation. The SFT stage teaches the model a structured rubric-based evaluation pipeline, including generating evaluation dimensions, assigning weights, and producing dimension-level scores. However, trajectory imitation alone does not directly optimize final preference accuracy, and the model may still produce inaccurate dimension scores when it applies a rubric to fine-grained candidate comparison. We therefore introduce GRPO after SFT and use dimension-level rewards to optimize scoring behavior conditioned on a rubric.

Directly rewarding only the final preference provides a sparse signal and may assign the same reward to trajectories with substantially diferent scoring quality. We instead define rewards at a finer, dimension level, so that the model must not only select the correct preference but also align its score diferences with the reference trajectory on each dimension.

Fixed-Rubric Rollout. Dimension-level rewards require the model-produced scoring dimensions to correspond one-to-one with the reference trajectory. During RL, we therefore fix the synthesized rubric for each sample and roll out only the subsequent dimension-level analysis and scores. Concretely, we split each teacher trajectory into a rubric part $\mathcal { R } _ { j }$ and a scoring part $\mathcal { I } _ { j }$ . During training, $\mathcal { R } _ { j }$ is treated as a given evaluation protocol, so rollouts for the same sample score under consistent dimensions, weights, and criteria. The predicted dimension scores can then be compared directly against the reference scores, allowing GRPO to focus on improving preference scoring accuracy conditioned on a rubric. This fixed-rubric design is used only during training. At inference time, RubricRM still generates the rubric and scores the candidates end to end. Further discussion of this design is provided in Appendix A.2.

Dimension-Level Reward Design. When designing the reward, we consider two factors: first, whether the predicted score diferences align in direction with the reference on each dimension, i.e., which of A or B is better; second, the magnitude of the score diference, as a large margin and a narrow margin should be rewarded diferently even if both indicate the same preference. For each dimension $d _ { i }$ , let $( s _ { i } ^ { A , g t } , s _ { i } ^ { B , \dot { g } t } )$ be the reference scores and $( s _ { i } ^ { A , p r e d } , s _ { i } ^ { B , p r e d } )$ be the rollout scores. We first compute the two score gaps:

$$
\Delta _ { i } ^ { g t } = s _ { i } ^ { A , g t } - s _ { i } ^ { B , g t } , \qquad \Delta _ { i } ^ { p r e d } = s _ { i } ^ { A , p r e d } - s _ { i } ^ { B , p r e d } .\tag{4}
$$

The magnitude factor compares these gaps, normalized by the maximum possible gap diference:

$$
b _ { i } = 1 - \frac { \left| \Delta _ { i } ^ { p r e d } - \Delta _ { i } ^ { g t } \right| } { 2 ( s _ { \operatorname* { m a x } } - s _ { \operatorname* { m i n } } ) } .\tag{5}
$$

We further assign a direction factor $\phi _ { i }$ according to the signs of $\Delta _ { i } ^ { g t }$ and $\Delta _ { i } ^ { p r e d } \colon \phi _ { i } = 1 . 0$ when the signs match, $\phi _ { i } = 0 . 6$ when exactly one gap is zero, and $\phi _ { i } ~ = ~ 0 . 1$ when the two nonzero signs are opposite. This ordering reflects the relative severity of scoring errors: exact directional agreement is left unpenalized, tie mismatches are partially discounted, and complete reversals receive the largest penalty. The final rollout reward is the rubric-weighted aggregation:

$$
R = \sum _ { i = 1 } ^ { N } w _ { i } b _ { i } \phi _ { i } .\tag{6}
$$

This reward prioritizes correct dimension-level preference direction through $\phi _ { i } .$ , while $b _ { i }$ further diferentiates rollouts that have the same direction but diferent score-gap accuracy. Appendix A.4 gives concrete cases illustrating why the reward distinguishes reversed dimension directions, tie mismatches, and inaccurate score-diference magnitudes.

Saturated Group Filtering. Not all rollout groups provide reliable relative preference signals. For each conditioned input, GRPO samples G responses and converts their rewards into grouprelative advantages:

$$
\hat { A } _ { i } = \frac { R _ { i } - \mu _ { R } } { \sigma _ { R } + \epsilon } ,\tag{7}
$$

where $\mu _ { R }$ and $\sigma _ { R }$ are the reward mean and standard deviation within the rollout group. This normalization is useful when rollouts difer meaningfully, but it becomes unreliable when all responses in a group receive nearly identical continuous rewards. In such low-spread groups, a tiny fluctuation in one rollout’s reward can be amplified into a large relative advantage, causing the policy to update on noise rather than a meaningful scoring diference. We therefore filter low-standard-deviation groups by setting

$$
A _ { i } = \left\{ { \begin{array} { l l } { 0 , } & { \sigma _ { R } < \sigma _ { \mathrm { t h } } , } \\ { { \hat { A } } _ { i } , } & { \sigma _ { R } \geq \sigma _ { \mathrm { t h } } . } \end{array} } \right.\tag{8}
$$

We use $\sigma _ { \mathrm { t h } } ~ = ~ 0 . 0 5$ for rewards in [0, 1]. Because the decision is made from the reward spread of the current rollout group, the filtered groups change as the policy evolves; this is a dynamic training-time filter rather than an ofline data removal step. Appendix A.3 gives the final GRPO objective, and Figure 3 reports the corresponding training-stability curves.

## 4 Experiments

This section evaluates RubricRM on pairwise preference benchmarks for text-to-image generation and image editing. We first describe the benchmarks, baselines, and implementation details, then compare RubricRM with specialized reward models and MLLM judges, and finally analyze the contribution of the two-stage training pipeline.

## 4.1 Experimental Setup

Benchmarks For text-to-image generation, we adopt the text-to-image subset of MMRB2 (Hu et al., 2025), GenAI-Bench (Jiang et al., 2024), and our filtered GenAI-Bench-Verified. MMRB2 contains 1,000 pairwise comparison samples for assessing fine-grained preference judgment on textto-image outputs. GenAI-Bench comprises approximately 1,700 pairwise comparisons and has been widely used in prior reward modeling work. We remove tie and both-bad cases, retaining only samples with a clear preference. Additionally, we observed that some samples in GenAI-Bench suffer from image mismatches or incorrect preference labels. To address this, we construct an expertverified subset, GenAI-Bench-Verified, by inviting three expert annotators to re-label the candidate pairs and keeping only those samples where all three annotators agree and the consensus matches the original label. This filtering process yields a higher-confidence subset while maintaining compatibility with the original benchmark.

Table 2: Comparison on text-to-image generation benchmarks. Bold and underlined scores denote the best and second-best results among reward models, respectively.
<table><tr><td>Model</td><td>MMRB2</td><td>GenAI- Bench</td><td>GenAI- Bench- Verified</td></tr><tr><td colspan="4">Proprietary MLLMs</td></tr><tr><td>Claude Sonnet 4.6 GPT-5.4 Gemini 2.5 Pro</td><td>70.8 67.5 70.5</td><td>65.8 64.2 67.8 73.9</td><td>75.3 74.2 77.4 84.8</td></tr><tr><td colspan="4">Gemini 3.1 Pro 74.4 Open-source MLLMs</td></tr><tr><td>Qwen3-VL-8B Qwen3-VL-235B-A22B Qwen3.5-9B</td><td>61.2 66.6 66.3</td><td>63.3 61.5 63.3</td><td>72.5 69.7 70.7</td></tr><tr><td colspan="4">Qwen3.5-397B-A17B 72.7 66.2 77.0</td></tr><tr><td>Reward Models HPSv2</td><td>55.0</td><td>68.8</td><td>78.1</td></tr><tr><td>PickScore</td><td>57.6</td><td>70.0</td><td>79.2</td></tr><tr><td>HPSv3</td><td>60.2</td><td>70.9 69.2</td><td>81.0</td></tr><tr><td>UnifiedReward-9B UnifiedReward-Think-9B</td><td>57.9 65.5</td><td>72.8</td><td>72.8 81.7</td></tr><tr><td></td><td>69.2</td><td>73.4</td><td>84.2</td></tr><tr><td>UnifiedReward-Flex-8B</td><td></td><td></td><td></td></tr><tr><td>RubricRM-Gen-4B (Ours)</td><td>70.5</td><td>73.2</td><td>83.1</td></tr><tr><td>RubricRM-Gen-9B (Ours)</td><td>72.0</td><td>74.1</td><td>84.5</td></tr></table>

For image editing, we use the image editing subset of MMRB2 (Hu et al., 2025), EditReward-ERB (Wu et al., 2025), and EditScore-ERB (Luo et al., 2025). Since both Wu et al. (2025) and Luo et al. (2025) refer to their editing evaluation sets as EditReward-Bench, we use EditReward-ERB and EditScore-ERB to distinguish the two benchmark sources. MMRB2 also contains 1,000 test instances. EditReward-ERB evaluates K-way preference prediction with K = 2, 3, 4 and also reports an overall benchmark score, which we denote as Avg in our table. EditScore-ERB contains 3,072 human-annotated preference pairs over 13 editing subtasks grouped into Subject, Appearance, Scene, and Advanced. It reports Prompt Following (PF), Consistency (C), Overall Quality (O), and an aggregate Avg score.

Baselines. We compare with general MLLM baselines, including open-source models Qwen3- VL-8B and Qwen3-VL-235B-A22B (Bai et al., 2025), Qwen3.5-9B and Qwen3.5-397B-A17B (Qwen Team, 2026), as well as proprietary models Claude Sonnet 4.6 (Anthropic, 2026), GPT-5.4 (OpenAI, 2026), Gemini 2.5 Pro (Comanici et al., 2025), and Gemini 3.1 Pro (Google DeepMind, 2026). All MLLM judges are evaluated with chainof-thought pairwise comparison prompts. We also compare with specialized reward models. Text-toimage baselines include HPSv2 (Wu et al., 2023a), PickScore (Kirstain et al., 2023), HPSv3 (Ma et al., 2025), UnifiedReward-9B (Wang et al., 2025), UnifiedReward-Think-9B (Wang et al., 2026b), and UnifiedReward-Flex-8B (Wang et al., 2026a). For UnifiedReward and UnifiedReward-Think, we use their newly released Qwen3.5-9B checkpoints. Image editing baselines include EditReward-7B (Wu et al., 2025) and EditScore-7B (Luo et al., 2025).

Implementation Details. We train separate RubricRM models for generation and editing on Qwen3.5-4B and Qwen3.5-9B backbones. The SFT stage uses learning rate 5e−6 with cosine decay and runs for 2 epochs. The GRPO stage uses learning rate 5e−7, group size G = 8, and KL coeficient β = 0.05. SFT and RL use disjoint training splits, and the RL split contains approximately 15K samples per task. Training is performed on 8 NVIDIA H20 GPUs. The taskspecific inference prompt templates are shown in Appendices C.1 and C.2.

## 4.2 Experimental Results

Table 2 reports text-to-image results. RubricRM-Gen-9B achieves the highest scores among rewardmodel baselines on all three benchmarks, reaching 72.00 on MMRB2, 74.12 on GenAI-Bench, and 84.45 on GenAI-Bench-Verified. Compared with UnifiedReward-Flex-8B, the strongest prior reward-model baseline, RubricRM-Gen-9B improves by 2.80 points on MMRB2, 0.72 points on GenAI-Bench, and 0.25 points on GenAI-Bench-Verified. RubricRM-Gen-4B also performs competitively, obtaining 70.50, 73.20, and 83.11, despite using a smaller backbone than several general MLLM baselines.

Table 3: Comparison on image editing benchmarks. Avg denotes the benchmark-level average score. Bold and underlined scores denote the best and second-best results among reward models, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">MMRB2</td><td colspan="4">EditReward-ERB</td><td colspan="4">EditScore-ERB</td></tr><tr><td>K=2</td><td>K=3</td><td>K=4</td><td>Avg</td><td>PF</td><td>C</td><td>0</td><td>Avg</td></tr><tr><td colspan="10">Proprietary MLLMs</td></tr><tr><td>Claude Sonnet 4.6</td><td>71.7</td><td>79.0</td><td>46.0</td><td>10.8</td><td>44.1</td><td>87.6</td><td>67.4</td><td>81.4</td><td>79.3</td></tr><tr><td>GPT-5.4</td><td>68.5</td><td>78.2</td><td>40.7</td><td>12.2</td><td>42.5</td><td>83.7</td><td>62.8</td><td>76.3</td><td>74.6</td></tr><tr><td>Gemini 2.5 Pro</td><td>71.3</td><td>80.5</td><td>38.0</td><td>12.2</td><td>42.2</td><td>81.4</td><td>65.5</td><td>77.4</td><td>75.2</td></tr><tr><td>Gemini 3.1 Pro</td><td>74.9</td><td>78.2</td><td>45.3</td><td>14.9</td><td>45.0</td><td>87.7</td><td>71.6</td><td>84.1</td><td>81.6</td></tr><tr><td colspan="10">Open-source MLLMs</td></tr><tr><td>Qwen3-VL-8B</td><td>63.4</td><td>68.4</td><td>31.3</td><td>25.2</td><td>40.9</td><td>84.5</td><td>64.9</td><td>79.6</td><td>76.9</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>64.8</td><td>70.7</td><td>30.7</td><td>6.1</td><td>34.6</td><td>87.2</td><td>66.0</td><td>81.6</td><td>78.8</td></tr><tr><td>Qwen3.5-9B</td><td>64.4</td><td>75.2</td><td>33.3</td><td>7.4</td><td>37.4</td><td>79.5</td><td>60.6</td><td>74.6</td><td>72.0</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>73.7</td><td>76.7</td><td>43.3</td><td>14.9</td><td>43.9</td><td>88.1</td><td>69.8</td><td>84.2</td><td>81.2</td></tr><tr><td colspan="10">Reward Models</td></tr><tr><td>EditReward-7B</td><td>67.2</td><td>56.5</td><td>42.7</td><td>11.5</td><td>38.4</td><td>85.9</td><td>67.2</td><td>80.4</td><td>78.3</td></tr><tr><td>EditScore-7B</td><td>55.6</td><td>62.4</td><td>23.3</td><td>4.1</td><td>28.8</td><td>59.2</td><td>59.1</td><td>65.9</td><td>61.9</td></tr><tr><td>RubricRM-Edit-4B (Ours)</td><td>73.2</td><td>76.7</td><td>48.0</td><td>14.9</td><td>45.5</td><td>88.7</td><td>77.5</td><td>88.9</td><td>85.5</td></tr><tr><td>RubricRM-Edit-9B (Ours)</td><td>75.4</td><td>81.2</td><td>42.0</td><td>19.6</td><td>46.4</td><td>88.6</td><td>79.7</td><td>87.7</td><td>85.6</td></tr></table>

Table 3 reports image-editing results. RubricRM-Edit-9B achieves the highest rewardmodel scores on MMRB2 (75.40), EditReward-ERB Avg (46.40), and EditScore-ERB Avg (85.64). Among reward-model baselines, it also leads EditReward-ERB at K=2 and K=4, while RubricRM-Edit-4B obtains the highest K=3 score. The fine-grained EditScore-ERB metrics further support the motivation for dynamic rubrics. Image editing requires balancing instruction following, source-image consistency, and visual quality. RubricRM obtains high scores across all three aspects: RubricRM-Edit-4B achieves the highest Prompt Following (88.67) and Overall Quality (88.85), while RubricRM-Edit-9B achieves the highest Consistency (79.66) and Avg score. Appendix B.1 provides the full category-wise results. This indicates that rubric-based scoring helps the model compare edited images through task-relevant criteria rather than relying on a single undiferentiated quality score.

## 4.3 Ablation Study: Two-Stage Training

Tables 4 and 5 ablate the two stages of RubricRM by comparing the base Qwen3.5 backbones, the models after rubric-trajectory SFT, and the final models after dimension-level GRPO. The comparison is reported separately for text-to-image generation and image editing.

Table 4: Text-to-image ablation of the two-stage training pipeline. Green deltas denote changes relative to the previous row within the same backbone.
<table><tr><td>Configuration</td><td>MMRB2</td><td>GenAI-Bench</td><td>GenAI-Bench-Verified</td></tr><tr><td>Qwen3.5-4B</td><td>63.3</td><td>61.9</td><td>69.7</td></tr><tr><td>+ SFT</td><td>70.1 (↑6.8)</td><td>72.0 (↑10.1)</td><td>82.9 (↑13.1)</td></tr><tr><td>+ RL</td><td>70.5 (↑0.4)</td><td>73.2 (↑1.2)</td><td>83.1 (↑0.2)</td></tr><tr><td>Qwen3.5-9B</td><td>66.9</td><td>63.4</td><td>72.5</td></tr><tr><td>+ SFT</td><td>70.3 (↑3.4)</td><td>73.0 (↑9.6)</td><td>83.2 (↑10.7)</td></tr><tr><td>+RL</td><td>72.0 (↑1.7)</td><td>74.1 (↑1.2)</td><td>84.5 (↑1.2)</td></tr></table>

Table 5: Image-editing ablation of the two-stage training pipeline. Green deltas denote changes relative to the previous row within the same backbone.
<table><tr><td>Configuration</td><td>MMRB2</td><td>EditReward-ERB</td><td>EditScore-ERB</td></tr><tr><td>Qwen3.5-4B</td><td>64.7</td><td>37.1</td><td>72.2</td></tr><tr><td>+ SFT</td><td>71.5 (↑6.8)</td><td>43.6 (↑6.5)</td><td>83.3 (↑11.1)</td></tr><tr><td>+RL</td><td>73.2 (↑1.7)</td><td>45.5 (↑1.9)</td><td>85.5 (↑2.2)</td></tr><tr><td>Qwen3.5-9B</td><td>64.4</td><td>37.4</td><td>72.0</td></tr><tr><td>+ SFT</td><td>73.8 (↑9.4)</td><td>45.0 (↑7.6)</td><td>85.1 (↑13.1)</td></tr><tr><td>+RL</td><td>75.4 (↑1.6)</td><td>46.4 (↑1.4)</td><td>85.6 (↑0.5)</td></tr></table>

Rubric-trajectory SFT accounts for most of the improvement. On text-to-image benchmarks, SFT improves the 4B backbone by 6.8 to 13.1 points and the 9B backbone by 3.4 to 10.7 points. On imageediting benchmarks, the corresponding gains are 6.5 to 11.1 points for 4B and 7.6 to 13.1 points for 9B. These gains suggest that a substantial part of the task lies in learning the rubric-based evaluation protocol itself, including criterion selection, weight assignment, and structured dimension-level scoring.

Dimension-level GRPO yields smaller but consistent additional gains across all benchmarks. This pattern is consistent with the role of the two stages: SFT establishes the rubric-scoring format, whereas GRPO calibrates its application by aligning predicted score gaps with the reference trajectory. Additional ablation results are provided in Appendix B.1.

## 5 Conclusion

We introduce RubricRM, a generative reward model that represents visual evaluation as dynamic rubric generation followed by rubric-grounded scoring. This formulation makes the scoring protocol explicit: the model first determines what should be evaluated, then applies those criteria through weighted per-dimension scores. A twostage training procedure separates these capabilities, using SFT for rubric formulation and GRPO for scoring alignment. Across generation and editing benchmarks, RubricRM improves over prior specialized reward models while preserving an inspectable judgment trace.

## Limitations

RubricRM has several limitations. First, rubric trace synthesis relies on a proprietary teacher model, so teacher biases may be inherited by the student models. Second, the current framework evaluates static images and does not address video generation or temporal editing, where temporal consistency would require additional dimensions and scoring mechanisms. Future work will explore teacher-free rubric improvement, human-inthe-loop rubric correction, and extensions to video assessment.

## Ethical Considerations

RubricRM learns evaluative signals from human preference labels and teacher-synthesized rubric trajectories. These supervision sources may reflect subjective or systematic biases that can affect the learned rubrics and preference judgments. Although the explicit dimensions, weights, and scores improve inspectability, they should not be interpreted as objective standards or guaranteed causal explanations of a decision. This is especially important when the reward model is used to optimize downstream text-to-image generation models and image editing models, where biased reward signals may be amplified. We therefore recommend auditing RubricRM on the target content distribution and treating its rubric outputs as model judgments, not authoritative assessments. Since visual preference datasets may still contain people or sensitive content, users should adhere to the access conditions of the underlying datasets and consider content-specific audits before downstream deployment.

## References

Anthropic. 2026. Claude Sonnet 4.6 system card. Technical report, Anthropic.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. 2024. Training difusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pages 4965–4987.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. 2023. Dpok: Reinforcement learning for fine-tuning text-to-image difusion models. Advances in Neural Information Processing Systems, 36:79858–79885.

Google DeepMind. 2026. Gemini 3.1 Pro model card.

Shuhao Han, Haotian Fan, Jiachen Fu, Liang Li, Tao Li, Junhui Cui, Yunqiu Wang, Yang Tai, Jingwei Sun, Chun-Le Guo, and 1 others. 2026. Evalmuse-40k: A fine-grained benchmark with comprehensive human

annotations for text-to-image generation model alignment evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 4583–4591.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings ofthe 2021 conference on empirical methods in natural language processing, pages 7514–7528.

Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, and Marjan Ghazvininejad. 2025. Multimodal rewardbench 2: Evaluating omni reward models for interleaved text and image. arXiv preprint arXiv:2512.16899.

Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2023. T2i-compbench: A comprehensive benchmark for open-world compositional textto-image generation. Advances in Neural Information Processing Systems, 36:78723–78747.

Mengzhao Jia, Zhihan Zhang, Ignacio Cases, Zheyuan Liu, Meng Jiang, and Peng Qi. 2025. Autorubric-r1v: Rubric-based generative rewards for faithful multimodal reasoning. arXiv preprint arXiv:2510.14738.

Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. 2024. Genai arena: An open evaluation platform for generative models. Advances in Neural Information Processing Systems, 37:79889–79908.

Kuei-Chun Kao, Daixuan Huo, Yuanhao Ban, and Cho-Jui Hsieh. 2026. Autorubric-t2i: Robust rule-based reward model for text-to-image alignment. arXiv preprint arXiv:2605.17602.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. 2023. Pick-apic: An open dataset of user preferences for text-toimage generation. Advances in neural information processing systems, 36:36652–36663.

Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. 2024. Viescore: Towards explainable metrics for conditional image synthesis evaluation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12268–12290.

Sunzhu Li, Jiale Zhao, Miteto Wei, Huimin Ren, Yang Zhou, Jingwen Yang, Shunyu Liu, Kaike Zhang, and Wei Chen. 2026. Rubrichub: A comprehensive and highly discriminative rubric dataset via automated coarse-to-fine generation. arXiv preprint arXiv:2601.08430.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. 2025. Openrubrics: Towards scalable synthetic rubric generation for reward modeling and llm alignment. arXiv preprint arXiv:2510.07743.

Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, and 1 others. 2025. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909.

Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. 2025. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095.

OpenAI. 2026. GPT-5.4 Thinking system card.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Weijie Qiu, Dai Guan, Junxin Wang, Zhihang Li, Yongbo Gai, Mengyu Zhou, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Rationale matters: Learning transferable rubrics via proxy-guided critique for vlm reward models. arXiv preprint arXiv:2603.16600.

Qwen Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.

Delip Rao and Chris Callison-Burch. 2026. Autorubric: Unifying rubric-based llm evaluation. arXiv preprint arXiv:2603.00077.

Juanxi Tian, Fengyuan Liu, Jiaming Han, Yilei Jiang, Yongliang Wu, Yesheng Liu, Haodong Li, Furong Xu, and Wanhua Li. 2026. Auto-rubric as reward: From implicit preferences to explicit multimodal generative criteria. arXiv preprint arXiv:2605.08354.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. 2024. Difusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238.

Yibin Wang, Yuhang Zang, Feng Han, Jiazi Bu, Yujie Zhou, Cheng Jin, and Jiaqi Wang. 2026a. Unified personalized reward model for vision generation. arXiv preprint arXiv:2602.02380.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. 2025. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236.

Yibin Wang, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, Jiaqi Wang, and 1 others. 2026b. Unified multimodal chain-of-thought reward model through reinforcement fine-tuning. Advances in Neural Infor mation Processing Systems, 38:159130–159157.

Zengbin Wang, Xuecai Hu, Yong Wang, Feng Xiong, Man Zhang, and Xiangxiang Chu. 2026c. Everything in its place: Benchmarking spatial intelligence of text-to-image models. arXiv preprint arXiv:2601.20354.

Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. 2025. Editreward: A humanaligned reward model for instruction-guided image editing. arXiv preprint arXiv:2509.26346.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023a. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341.

Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023b. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, and 1 others. 2026. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 11269–11277.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935.

## A Method Details

This appendix provides additional methodological details that support the training pipeline: data coverage and taxonomy statistics, the advantage filtering rule used for GRPO, and representative dimension-level reward cases.

## A.1 Training Data and Taxonomy Statistics

RL split construction. The RL split is selected with stratified sampling over taxonomy annotations. We start from a sampled split and iteratively swap examples when a secondary label deviates from the target coverage. This keeps the maximum secondary-label coverage drift below 0.5 percentage points for both tasks, preserving lowfrequency categories while reducing overrepresentation of frequent categories.

Table 6: Second-level coverage of the text-to-image prompt taxonomy. Labels are multi-label; coverage is measured over all text-to-image training examples.
<table><tr><td>Primary</td><td>Secondary label</td><td>Cov. (%)</td></tr><tr><td>Common Objects</td><td>People</td><td>45.81</td></tr><tr><td>Style</td><td>Realistic Photography</td><td>38.83</td></tr><tr><td>Composition &amp;</td><td>Other</td><td>26.34</td></tr><tr><td>Perspective</td><td></td><td></td></tr><tr><td>Common Objects</td><td>Everyday Items</td><td>24.55</td></tr><tr><td>Common Objects</td><td>Architecture &amp; Interior</td><td>23.33</td></tr><tr><td>Common Objects</td><td>Natural Scenery</td><td>15.53</td></tr><tr><td>Composition &amp;</td><td>Close-up &amp; Macro</td><td>14.64</td></tr><tr><td>Perspective</td><td></td><td></td></tr><tr><td>Common Objects</td><td>Fictional Characters &amp; Creatures</td><td>11.93</td></tr><tr><td>Common Objects</td><td>Animals</td><td>11.89</td></tr><tr><td>Style</td><td>Illustration &amp; Hand-drawn</td><td>11.86</td></tr><tr><td>Common Objects</td><td>Plants &amp; Flowers</td><td>8.41</td></tr><tr><td>Style</td><td>3D Rendering</td><td>7.66</td></tr><tr><td>Logical Reasoning</td><td>Spatial Relations</td><td>7.41</td></tr><tr><td>World Knowledge</td><td>History &amp; Culture</td><td>7.24</td></tr><tr><td>Common Objects</td><td>Vehicles</td><td>7.03</td></tr><tr><td>Style</td><td>Traditional Art</td><td>5.55</td></tr><tr><td>Composition &amp;</td><td>Panoramic &amp; Wide-angle</td><td>5.46</td></tr><tr><td>Perspective</td><td>Abstract Art</td><td></td></tr><tr><td>Style</td><td></td><td>5.06</td></tr><tr><td>Text Rendering</td><td>English</td><td>4.29</td></tr><tr><td>Composition &amp;</td><td>Overhead &amp; Low-angle</td><td>4.24</td></tr><tr><td>Perspective</td><td></td><td></td></tr><tr><td>Style Style</td><td>Cyberpunk &amp; Steampunk</td><td>4.12 4.04</td></tr><tr><td></td><td>Anime</td><td></td></tr><tr><td>Common Objects</td><td>Food &amp; Drinks</td><td>3.97</td></tr><tr><td>World Knowledge</td><td>IP Characters &amp; Brands</td><td>3.96</td></tr><tr><td>Logical Reasoning</td><td>Counting</td><td>3.94</td></tr><tr><td>World Knowledge</td><td>Geography &amp; Landmarks</td><td>3.24</td></tr><tr><td>Style</td><td>Minimalist &amp; Flat</td><td>2.78</td></tr><tr><td>World Knowledge</td><td>Famous People</td><td>2.48</td></tr><tr><td>Composition &amp;</td><td>Symmetrical</td><td>2.39</td></tr><tr><td>Perspective</td><td></td><td></td></tr><tr><td>Logical Reasoning</td><td>Common Sense Pixel Art</td><td>1.62</td></tr><tr><td>Style</td><td>Logo</td><td>1.36</td></tr><tr><td>Text Rendering</td><td></td><td>1.26</td></tr><tr><td>World Knowledge</td><td>Professional Domains</td><td>1.11</td></tr><tr><td>Logical Reasoning</td><td>Causal &amp; Temporal</td><td>1.05</td></tr><tr><td>Text Rendering</td><td>Multilingual &amp; Symbols</td><td>0.90</td></tr><tr><td>Text Rendering</td><td>Handwriting &amp; Calligraphy</td><td>0.68</td></tr><tr><td>Logical Reasoning</td><td>Linguistic Logic</td><td>0.64</td></tr><tr><td>Text Rendering</td><td>Chinese</td><td>0.60</td></tr><tr><td>Logical Reasoning</td><td>Scientific Reasoning</td><td>0.59</td></tr><tr><td>Style</td><td>Pop Art</td><td>0.50</td></tr><tr><td>Style</td><td>Ink Wash Painting</td><td>0.30</td></tr></table>

Primary-category coverage. For text-to-image generation, the most frequent primary categories are Common Objects (92.4%), Style (69.0%), and Composition & Perspective (52.1%). Long-tail but important categories remain represented: World Knowledge covers 15.8%, Logical Reasoning covers 9.0%, and Text Rendering covers 6.5% of examples. For image editing, Object Editing covers 45.9%, Attribute Editing 25.7%, Background Editing 18.2%, Text Editing 11.2%, Style Editing 10.2%, Reasoning-based Editing 8.7%, and Inpainting & Repair 6.5%. Figure 2 visualizes these primary-category distributions for both tasks.

Secondary labels and co-occurrence. The most frequent text-to-image secondary labels are People (45.8%), Realistic Photography (38.8%), Other Composition (26.3%), Everyday Items (24.6%), and Architecture & Interior (23.3%). The

![](images/436ac30e22e45a0944f7ea57cc10e33e73634d9f737e5de4c4fba8975c7896f3.jpg)  
Text-to-image generation

![](images/6736418bce35dcb2a23a30d0fb991baa9251619dbcf858b5d8d6bdf31b23fe05.jpg)  
Image editing  
Figure 2: Primary-category coverage of the rubric-learning data for text-to-image generation and image editing.

Table 7: Second-level coverage of the image-editing instruction taxonomy. Labels are multi-label; coverage is measured over all image-editing training examples.

<table><tr><td>Primary</td><td>Secondary label</td><td>Cov. (%)</td></tr><tr><td>Object Editing</td><td>Add Object</td><td>21.01</td></tr><tr><td>Attribute Editing</td><td>Color Change</td><td>14.30</td></tr><tr><td>Object Editing</td><td>Remove Object</td><td>12.52</td></tr><tr><td>Background Editing</td><td>Background Replacement</td><td>11.82</td></tr><tr><td>Text Editing</td><td>Text Editing</td><td>11.17</td></tr><tr><td>Object Editing</td><td>Replace Object</td><td>10.77</td></tr><tr><td>Style Editing</td><td>Art Style Transfer</td><td>8.01</td></tr><tr><td>Attribute Editing</td><td>Material &amp; Texture</td><td>5.75</td></tr><tr><td>Attribute Editing</td><td>Appearance &amp; Shape</td><td>5.59</td></tr><tr><td>Reasoning-based Editing</td><td>Implicit Intent</td><td>4.68</td></tr><tr><td>Reasoning-based Editing</td><td>Complex Instruction</td><td>4.44</td></tr><tr><td>Background Editing</td><td>Lighting &amp; Environment</td><td>4.06</td></tr><tr><td>Inpainting &amp; Repair</td><td>Watermark &amp; Occlusion Removal</td><td>3.65</td></tr><tr><td>Background Editing</td><td>Season &amp; Weather Change</td><td>3.05</td></tr><tr><td>Inpainting &amp; Repair</td><td>Blemish Removal</td><td>2.85</td></tr><tr><td>Object Editing</td><td>Move &amp; Rearrange</td><td>2.63</td></tr><tr><td>Pose &amp; Action Editing</td><td>Body Pose</td><td>2.21</td></tr><tr><td>Attribute Editing</td><td>Size Adjustment</td><td>1.25</td></tr><tr><td>Style Editing</td><td>Era Style</td><td>1.08</td></tr><tr><td>Attribute Editing</td><td>Quantity Change</td><td>0.98</td></tr><tr><td>Style Editing</td><td>Material Style</td><td>0.96</td></tr><tr><td>Face Editing</td><td>Facial Attributes</td><td>0.95</td></tr><tr><td>Style Editing</td><td>Tone &amp; Filter</td><td>0.64</td></tr><tr><td>Pose &amp; Action Editing</td><td>Expression Change</td><td>0.56</td></tr><tr><td>Background Editing</td><td>Background Removal</td><td>0.51</td></tr><tr><td>Viewpoint Change</td><td>Viewpoint Change</td><td>0.30</td></tr><tr><td>Multi-image Composition</td><td>Multi-image Composition</td><td>0.27</td></tr><tr><td>Quality Enhancement</td><td>Denoise &amp; Deblur</td><td>0.14</td></tr><tr><td>Quality Enhancement</td><td>Super Resolution</td><td>0.12</td></tr><tr><td>Face Editing</td><td>Age Change</td><td>0.11</td></tr></table>

strongest primary-category co-occurrences are Common Objects+Style (63.6%), Common Objects+Composition & Perspective (49.5%), and Composition & Perspective+Style (37.3%). These co-occurrences show that many prompts simultaneously specify subject matter, visual style, and composition, motivating instruction-specific rubric dimensions.

For image editing, the most frequent secondary labels are Add Object (21.0%), Color Change (14.3%), Remove Object (12.5%), Background Replacement (11.8%), Text Editing (11.2%), Replace Object (10.8%), and Art Style Transfer (8.0%). The strongest co-occurrences are Attribute Editing+Object Editing (6.6%), Object Editing+Reasoning-based Editing (5.2%), Inpainting & Repair+Object Editing (4.5%), and Background Editing+Object Editing (3.4%). Tables 6 and 7 list the complete second-level taxonomy coverage used to diagnose long-tail categories and construct balanced training splits.

Rubric-dimension aggregation. To check whether synthesized rubrics cover task-relevant evaluation aspects, we parse teacher-generated dimension titles and aggregate them into semantic groups. Tables 8 and 9 show the resulting dimension shares and average teacher-assigned weights. The aggregated dimensions align with the two task settings: text-to-image rubrics emphasize subject fidelity, visual quality, scene composition, style, and prompt alignment, whereas imageediting rubrics additionally emphasize source preservation and edit integration.

## A.2 Fixed-Rubric Rollout during GRPO

Our dimension-level reward compares the predicted and reference score gaps for each evaluation dimension, which requires a one-to-one correspondence between the rollout and reference dimensions. If each rollout generated its own dimensions and weights, diferent rollouts would be evaluated under diferent criteria, making the dimension-level rewards no longer directly comparable and introducing substantial training noise. We therefore treat the teacher-generated rubric as a training-time conditioning context and roll out only the subsequent dimension-level assessments and scores. This design provides a denser signal than rewarding only the final binary preference. Importantly, the restriction applies only during GRPO: rubric generation is learned from complete trajectories during SFT, and at inference time no reference rubric is provided, and the model generates an input-specific rubric and scores the candidates end to end. Thus, GRPO should be understood as calibrating rubric-conditioned scoring rather than directly optimizing rubric generation.

Table 8: Semantic aggregation of parsed text-to-image rubric dimensions. Dim. share is computed over parsed teacher-generated rubric dimensions; avg. wt. is the average teacher-assigned dimension weight.
<table><tr><td>Semantic group</td><td>Rubric atom</td><td>Dim. share (%)</td><td>Avg. wt.</td><td>Representative original dimension titles</td></tr><tr><td>Semantic Alignment</td><td>Prompt alignment</td><td>7.99</td><td>32.7</td><td>Content Accuracy; Semantic Accuracy; Semantic Alignment</td></tr><tr><td>Semantic Alignment</td><td>Text rendering</td><td>1.93</td><td>32.2</td><td>Text Accuracy; Text Rendering; Text Rendering Accuracy</td></tr><tr><td>Semantic Alignment</td><td>Reasoning &amp; knowledge</td><td>1.78</td><td>29.7</td><td>Composition &amp; Spatial Logic; Logical Consistency; Compositional Logic</td></tr><tr><td>Subject &amp; Attributes</td><td>Subject fidelity</td><td>14.89</td><td>30.3</td><td>Subject Accuracy; Subject Representation; Character Accuracy</td></tr><tr><td>Subject &amp; Attributes</td><td>Anatomy / pose</td><td>9.58</td><td>27.9</td><td>Anatomical Integrity; Anatomical Correctness; Anatomical Coherence</td></tr><tr><td>Subject &amp; Attributes</td><td>Attribute fidelity</td><td>8.64</td><td>26.5</td><td>Color Palette; Attire Accuracy; Lighting &amp; Color Palette</td></tr><tr><td>Scene &amp; Composition</td><td>Scene / background</td><td>11.81</td><td>24.3</td><td>Environmental Context; Setting &amp; Atmosphere; Environment &amp; Lighting</td></tr><tr><td>Scene &amp; Composition</td><td>Composition / perspective</td><td>7.25</td><td>25.6</td><td>Spatial Composition; Composition &amp; Perspective; Composition &amp; Framing</td></tr><tr><td>Scene &amp; Composition</td><td>Lighting / atmosphere</td><td>6.01</td><td>24.2</td><td>Lighting &amp; Atmosphere; Atmosphere &amp; Lighting; Lighting &amp; Contrast</td></tr><tr><td>Style &amp; Quality</td><td>Visual quality</td><td>17.94</td><td>26.2</td><td>Image Quality &amp; Coherence; Structural Coherence; Image Quality</td></tr><tr><td>Style &amp; Quality Style &amp; Quality</td><td>Style adherence</td><td>9.25</td><td>26.8</td><td>Stylistic Execution; Artistic Style; Image Quality &amp; Aesthetics</td></tr><tr><td></td><td>Realism</td><td>2.94</td><td>24.2</td><td>Image Quality &amp; Realism; Visual Quality &amp; Realism; Realism &amp; Image Quality</td></tr></table>

Table 9: Semantic aggregation of parsed image-editing rubric dimensions. Dim. share is computed over parsed teacher-generated rubric dimensions; avg. wt. is the average teacher-assigned dimension weight.
<table><tr><td>Semantic group</td><td>Rubric atom</td><td>Dim. share (%)</td><td>Avg. wt.</td><td>Representative original dimension titles</td></tr><tr><td>Instruction &amp; Edit</td><td>Instruction alignment</td><td>8.88</td><td>39.6</td><td>Instruction Alignment; Instruction Compliance; Instruction Adherence</td></tr><tr><td>Instruction &amp; Edit</td><td>Object edit</td><td>7.72</td><td>36.4</td><td>Object Removal; Object Addition; Object Replacement</td></tr><tr><td>Instruction &amp; Edit</td><td>Text editing</td><td>5.47</td><td>35.9</td><td>Text Accuracy; Text Rendering Accuracy; Text Placement</td></tr><tr><td>Instruction &amp; Edit</td><td>Attribute edit</td><td>4.45</td><td>35.4</td><td>Color Modification; Material Transformation; Color Transformation</td></tr><tr><td>Preservation</td><td>Source preservation</td><td>20.90</td><td>31.4</td><td>Content Preservation; Image Preservation; Structural Preservation</td></tr><tr><td>Preservation</td><td>Subject preservation</td><td>5.78</td><td>33.5</td><td>Subject Preservation; Context Preservation; Foreground Preservation</td></tr><tr><td>Scene Integration</td><td>Background edit</td><td>10.85</td><td>30.9</td><td>Background Preservation; Inpainting Quality; Background Reconstruction</td></tr><tr><td>Scene Integration</td><td>Visual integration</td><td>9.59</td><td>27.7</td><td>Realism &amp; Integration; Visual Integration; Realism &amp; Blending</td></tr><tr><td>Scene Integration</td><td>Spatial placement</td><td>2.51</td><td>31.3</td><td>Spatial Placement; Spatial Positioning; Placement Accuracy</td></tr><tr><td>Quality &amp; Realism</td><td>Quality / artifacts</td><td>13.12</td><td>31.8</td><td>Image Quality; Visual Quality; Editing Accuracy</td></tr><tr><td>Quality &amp; Realism</td><td>Realism</td><td>6.43 3.97</td><td>26.2 35.5</td><td>Visual Realism; Image Quality &amp; Realism; Image Realism</td></tr><tr><td>Quality &amp; Realism Quality &amp; Realism</td><td>Style transfer</td><td>0.32</td><td>38.4</td><td>Stylistic Transformation; Stylistic Accuracy; Aesthetic Quality</td></tr><tr><td></td><td>Reasoning</td><td></td><td></td><td>Semantic Accuracy; Semantic Transformation; Semantic Preservation</td></tr></table>

## A.3 GRPO Objective and Saturated-Group Filtering

Section 3 defines the dimension-level reward, group-relative advantage normalization, and saturated-group filtering rule. Using the filtered advantages $A _ { i } ,$ , the final GRPO objective is

$$
\begin{array} { l l l } { \displaystyle \mathcal { T } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { \boldsymbol { \tilde { x } } , \{ o _ { i } \} } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \Big ( \operatorname* { m i n } \big ( \rho _ { i } A _ { i } , } \\ { \quad } \\ { \quad } & { \quad } \end{array}\tag{9}
$$

where $\rho _ { i } \ = \ \pi _ { \theta } ( o _ { i } \ \mid \ \tilde { x } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i } \ | \ \tilde { x } )$ , and $D _ { \mathrm { K L } } ^ { ( i ) }$ estimates the KL divergence between the response $o _ { i }$ and a reference policy. The empirical trainingstability curves for saturated-group filtering are reported in Figure 3.

## A.4 Dimension-Level Reward Case Studies

Dimension-level rewards supervise more than the final pairwise preference. The following trainingrollout cases illustrate the three behaviors encoded by our reward: reversed dimension directions receive the strongest discount, tie mismatches receive a partial discount, and score-diference magnitudes are still rewarded when their directions are correct. Table 10 reports the per-dimension terms and the final weighted reward for each case.

Case 1: Correct final preference with a reversed dimension. For the prompt “writer creates stories, Overhead shot,” both the reference and the rollout select candidate B as the final preference. However, Table 10 shows that the rollout reverses the reference direction on Thematic Depiction: the reference scores indicate $A > B$ , whereas the predicted scores indicate $A < B$ . A final-preferenceonly reward would not expose this incorrect scoring rationale.

Case 2: Tie mismatches receive a partial discount. For the prompt “Person lying down with legs up, next to a dog,” two rollouts both select the correct final preference A. Rollout 1 in Table 10 preserves the reference tie on Anatomical

<table><tr><td>Case</td><td>Rollout</td><td>Dimension</td><td> $w _ { i }$ </td><td>Reference A/B</td><td>Predicted A/B</td><td> $b _ { i }$ </td><td> $\phi _ { i }$ </td><td>ri</td><td>Total R</td></tr><tr><td rowspan="3">1 Reverse</td><td rowspan="3"></td><td>Camera Perspective</td><td>0.40</td><td>1/4</td><td>0/4</td><td>0.875</td><td>1.0</td><td>0.875</td><td rowspan="3">0.6313</td></tr><tr><td>Thematic Depiction</td><td>0.30</td><td>4/2</td><td>2/3</td><td>0.625</td><td>0.1</td><td>0.0625</td></tr><tr><td>Structural Integrity</td><td>0.30</td><td>0 / 1</td><td>0/2</td><td>0.875</td><td>1.0</td><td>0.875</td></tr><tr><td rowspan="5">2 Tie</td><td rowspan="3">1</td><td>Prompt Adherence</td><td>0.40</td><td>2/1</td><td>3/1</td><td>0.875</td><td>1.0</td><td>0.875</td><td rowspan="3">0.950</td></tr><tr><td>Anatomical Correctness</td><td>0.40</td><td>0/0</td><td>0/0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Subject Separation</td><td>0.20</td><td>2/0</td><td>2/0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="3">2</td><td>Prompt Adherence</td><td>0.40</td><td>2/1</td><td>3/1</td><td>0.875</td><td>1.0</td><td>0.875</td><td rowspan="3">0.760</td></tr><tr><td>Anatomical Correctness</td><td>0.40</td><td>0/0</td><td>1/0</td><td>0.875</td><td>0.6</td><td>0.525</td></tr><tr><td>Subject Separation</td><td>0.20</td><td>2/0</td><td>2/0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="3">3</td><td rowspan="3"></td><td>Entity Accuracy</td><td>0.40</td><td></td><td></td><td></td><td>1.0</td><td>1.0</td><td rowspan="3">0.8875</td></tr><tr><td>Spatial Obstruction</td><td>0.30</td><td>4/0 4/0</td><td>4/0</td><td>1.0</td><td>1.0</td><td>0.625</td></tr><tr><td>Relative Size</td><td>0.30</td><td>4/0</td><td>1/0 4/0</td><td>0.625 1.0</td><td>1.0</td><td>1.0</td></tr></table>

Table 10: Dimension-level reward cases illustrating a reversed dimension direction, a tie mismatch, and an underestimated score-diference margin. Here $r _ { i } = b _ { i } \phi _ { i }$ denotes the per-dimension reward.

Table 11: Detailed results on EditScore-ERB across four task categories and overall. PF = Prompt Following, C = Consistency, O = Overall Quality. Scores are percentages. Bold and underlined scores denote the best and secondbest results among reward models, respectively.
<table><tr><td>Model</td><td colspan="3">Subject</td><td colspan="3">Appearance</td><td colspan="3">Scene</td><td colspan="3">Advanced</td><td colspan="3">Overall</td></tr><tr><td></td><td>PF</td><td>C</td><td>0</td><td>PF</td><td>C</td><td>0</td><td>PF</td><td>C</td><td>0</td><td>PF</td><td>C</td><td>0</td><td>PF</td><td>C</td><td>0</td></tr><tr><td>Proprietary MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Sonnet 4.6</td><td>83.9</td><td>64.5</td><td>82.9</td><td>86.8</td><td>69.2</td><td>76.2</td><td>93.8</td><td>74.5</td><td>88.8</td><td>88.7</td><td>63.5</td><td>83.2</td><td>87.6</td><td>67.4</td><td>81.4</td></tr><tr><td>GPT-5.4</td><td>86.1</td><td>56.2</td><td>79.7</td><td>82.8</td><td>62.8</td><td>73.3</td><td>91.9</td><td>72.4</td><td>82.5</td><td>79.2</td><td>62.8</td><td>74.8</td><td>83.7</td><td>62.8</td><td>76.3</td></tr><tr><td>Gemini 2.5 Pro</td><td>81.6</td><td>61.3</td><td>77.5</td><td>75.9</td><td>69.8</td><td>75.9</td><td>82.6</td><td>72.4</td><td>80.1</td><td>85.7</td><td>62.3</td><td>79.7</td><td>81.4</td><td>65.5</td><td>77.4</td></tr><tr><td>Gemini 3.1 Pro</td><td>86.5</td><td>65.9</td><td>86.3</td><td>83.0</td><td>75.1</td><td>77.8</td><td>90.1</td><td>76.4</td><td>90.8</td><td>91.7</td><td>70.3</td><td>87.5</td><td>87.7</td><td>71.6</td><td>84.1</td></tr><tr><td>Open-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-8B</td><td>85.4</td><td>67.6</td><td>84.7</td><td>84.8</td><td>64.8</td><td>75.2</td><td>88.4</td><td>74.6</td><td>82.2</td><td>81.5</td><td>57.5</td><td>78.9</td><td>84.5</td><td>64.9</td><td>79.6</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>90.0</td><td>69.9</td><td>85.4</td><td>85.8</td><td>62.5</td><td>75.9</td><td>91.2</td><td>76.4</td><td>87.2</td><td>84.7</td><td>62.3</td><td>82.9</td><td>87.2</td><td>66.0</td><td>81.6</td></tr><tr><td>Qwen3.5-9B</td><td>79.8</td><td>55.7</td><td>74.1</td><td>79.3</td><td>61.3</td><td>71.4</td><td>87.5</td><td>72.4</td><td>84.0</td><td>77.8</td><td>58.3</td><td>75.8</td><td>79.5</td><td>60.6</td><td>74.6</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>92.2</td><td>71.9</td><td>86.5</td><td>84.7</td><td>69.9</td><td>78.9</td><td>90.1</td><td>77.5</td><td>91.2</td><td>86.8</td><td>64.0</td><td>85.8</td><td>88.1</td><td>69.8</td><td>84.2</td></tr><tr><td>Reward Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EditReward-7B</td><td>86.6</td><td>66.2</td><td>83.1</td><td>84.2</td><td>65.8</td><td>76.5</td><td>96.2</td><td>75.5</td><td>89.2</td><td>83.2</td><td>65.3</td><td>79.9</td><td>85.9</td><td>67.2</td><td>80.4</td></tr><tr><td>EditScore-7B</td><td>59.0</td><td>58.5</td><td>74.0</td><td>57.3</td><td>61.2</td><td>66.3</td><td>74.4</td><td>73.5</td><td>78.9</td><td>53.6</td><td>50.3</td><td>52.9</td><td>59.2</td><td>59.1</td><td>65.9</td></tr><tr><td>RubricRM-Edit-4B (Ours)</td><td>90.7</td><td>77.8</td><td>91.5</td><td>86.4</td><td>83.3</td><td>87.5</td><td>90.7</td><td>80.4</td><td>90.1</td><td>88.7 87.7</td><td>70.5 73.6</td><td>88.0 85.4</td><td>88.7</td><td>77.5</td><td>88.9</td></tr><tr><td>RubricRM-Edit-9B (Ours)</td><td>90.6</td><td>78.3</td><td>90.7</td><td>86.0</td><td>84.0</td><td>86.2</td><td>93.0</td><td>85.9</td><td>90.8</td><td></td><td></td><td></td><td>88.6</td><td>79.7</td><td>87.7</td></tr></table>

Correctness, whereas Rollout 2 introduces an unsupported $A > B$ direction. This tie mismatch applies $\lambda _ { \mathrm { t i e } } = 0 . 6$ and lowers the aggregate reward from 0.950 to 0.760.

Case 3: Correct direction with an inaccurate margin. For the prompt “a Speaker is partially obstructed by a Snail, and the Speaker is larger than the Snail,” the rollout selects the correct final preference A and matches the direction on every dimension. However, Table 10 shows that it underestimates the reference margin on Spatial Obstruction, predicting 1/0 instead of the reference 4/0.

## B Experiments

## B.1 Supplementary Results

Detailed editing results. Table 11 expands the image-editing results on EditScore-ERB by reporting Prompt Following, Consistency, and Overall

Quality for each task category. These detailed scores support the aggregate trend in Table 3: RubricRM remains competitive across subject, appearance, scene, and advanced editing categories rather than improving only a single aspect.

Label-only supervision ablation. Table 12 compares direct label supervision with rubric-based SFT. This ablation uses the same backbones and preference data, but replaces the teachersynthesized rubric trajectory with a direct pairwise preference target. It therefore isolates whether the improvement comes only from the training data and preference labels, or also from the rubricbased scoring paradigm.

The label-only models improve over the base models, showing that the preference labels are useful on their own. However, they remain consistently below rubric-based SFT on most benchmarks. For text-to-image generation, the gap is especially clear: label-only SFT is lower than rubricbased SFT by 3.3–5.1 points for the 4B backbone and 1.9–3.9 points for the 9B backbone. On image editing, label-only SFT also falls behind on MMRB2 and EditReward-ERB, where instruction following and edit-specific criteria are central. These results indicate that RubricRM’s gains are not merely due to additional exposure to the same training examples. They also come from the proposed paradigm: the rubric trajectory teaches the model to decompose an instruction into evaluation criteria before making the pairwise judgment.

![](images/17989c0e1b3eac4634d86a743c9b832fdfeb61fd95ba9b21296a5427757d2d88.jpg)

![](images/cd73d37a5f5bfa2c936684f38861f1f58d9d5828d1e5289c0f20ae530c921005.jpg)

![](images/5db0ca5a87363bbfdcfe366fd81de8223420a737e92ee1e42d068e722c75fb07.jpg)

![](images/d7b7ca94cfec5f4f82714df3d357755b14e393d0ed75eed03692e54a26f48006.jpg)

![](images/ccb1dbaadf1c771c7e8f53167f019380d4665f3dcb5c343787a12991f9a3a181.jpg)

![](images/c80bee88a92c5180fb80c58d970afa941e93f60b3a0425e82d2db548a9884a2a.jpg)  
Figure 3: Efect of saturated-group filtering on GRPO training stability. The columns show reward mean, entropy, and response length from left to right; the top row is without filtering and the bottom row is with filtering. Without filtering, late training shows reward-mean collapse, sharp response-length growth, and entropy drift followed by collapse. Filtering low-standard-deviation groups keeps all three signals stable.

Table 12: Label-only versus rubric-based SFT ablation. Scores are percentages; deltas in Label-only SFT rows are relative to Rubric SFT within each backbone.
<table><tr><td>Backbone</td><td>Supervision</td><td colspan="3">Text-to-Image</td><td colspan="3">Image Editing</td></tr><tr><td></td><td></td><td>MMRB2</td><td>GenAI- Bench</td><td>GenAI- Bench- Verified</td><td>MMRB2</td><td>EditReward- ERB</td><td>EditScore- ERB</td></tr><tr><td>Qwen3.5-4B</td><td>Rubric SFT Label-only SFT</td><td>70.1 65.1 (↓5.0)</td><td>72.0 68.7 (↓3.3)</td><td>82.9 77.8 (↓5.1)</td><td>71.5 69.2 (↓2.3)</td><td>43.6 39.4 (↓4.2)</td><td>83.3 85.4 (↑2.1)</td></tr><tr><td>Qwen3.5-9B</td><td>Rubric SFT Label-only SFT</td><td>70.3 66.4 (↓3.9)</td><td>73.0 71.1 (↓1.9)</td><td>83.2 80.9 (↓2.3)</td><td>73.8  $6 9 . 2 \ : ( \downarrow 4 . 6 )$ </td><td>45.0 40.8 (↓4.2)</td><td>85.1  $8 3 . 9 \ ( \downarrow 1 . 2 )$ </td></tr></table>

Saturated-group filtering analysis. Figure 3 provides an empirical view of the saturated-group filtering mechanism described in Section 3. In our GRPO stage, rewards are continuous dimensionlevel scores and advantages are normalized within each rollout group. When all rollouts for the same input receive nearly identical rewards, the group contains little reliable relative preference signal; nevertheless, within-group normalization can amplify small numerical fluctuations into large policygradient updates. This is reflected in the unfiltered curves: the reward mean remains high in early training but collapses sharply in the late stage, response length grows abruptly, and entropy first drifts upward before collapsing. These coupled changes indicate that low-discrimination rollout groups can destabilize the policy update rather than improve rubric application. After filtering groups whose reward standard deviation falls below the threshold, the reward mean fluctuates within a narrow range, response length remains bounded, and entropy evolves smoothly. The comparison supports the role of saturated-group filtering as a training-stability device for continuous dimensionlevel GRPO.

## B.2 Analysis of Generated Rubrics

We conduct a rubric-only analysis on MMRB2 to examine whether the generated rubrics remain stable under stochastic decoding while adapting to diferent instructions. For each instance, we sample five rubrics with temperature 0.7 and top-p = 0.9. For each rubric pair, we first match evaluation dimensions by semantic similarity and then measure dimension similarity and weight diference. Higher dimension similarity indicates more similar evaluation criteria, whereas lower weight difference indicates more consistent importance assigned to the matched criteria.

<table><tr><td>Task</td><td>Same inst.</td><td>Same type</td><td>Diff. type</td><td>Weight diff.</td></tr><tr><td>Gen</td><td>0.763</td><td>0.597</td><td>0.563</td><td>0.102</td></tr><tr><td>Edit</td><td>0.800</td><td>0.651</td><td>0.612</td><td>0.110</td></tr></table>

Table 13: Stability and adaptation of generated rubrics on MMRB2. “Same inst.” denotes the same instruction and “Dif. type” denotes diferent instruction types. The three similarity columns report dimension similarity.

Stability. As shown in Table 13, repeated rubrics for the same instruction have dimension similarities of 0.763 for generation and 0.800 for editing, while their weight diferences remain small at 0.102 and 0.110, respectively. Thus, stochastic decoding does not produce arbitrary evaluation criteria: both the selected dimensions and their relative importance remain largely consistent.

Diversity and adaptation. For each instruction, we compare its rubrics with those produced for other prompts of the same instruction type and for prompts of diferent types. Similarity follows the expected order for both tasks: same instruction > same type > diferent type. Rubrics therefore preserve shared evaluation patterns among related instructions while varying across instruction types. The higher same-type than diferent-type similarity for generation (0.597 vs. 0.563) and editing (0.651 vs. 0.612) further indicates instruction-type adaptation. Qualitative inspection likewise shows that the selected evaluation dimensions change according to task requirements.

## B.3 Sensitivity to Teacher Choice

The teacher model in our pipeline is not an independent preference annotator. Each image-pair preference label is inherited from the original humanannotated data, and the teacher only converts that fixed preference into a rubric trajectory. Teacher bias can therefore afect dimension selection and granularity, but it does not determine the underlying preference label.

To quantify the remaining sensitivity, we conduct a controlled cross-teacher analysis on a fixed MMRB2 subset containing 200 generation and 200 editing comparisons. For every comparison, we keep the input and human preference label fixed and regenerate rubrics with three teacher families: Gemini 3.1 Pro, Claude Sonnet 5, and GPT-5.6 Luna. For each teacher pair, we semantically match the rubric dimensions and report whether at least one dimension is shared, whether at least half of the smaller dimension set is covered, and the average overlap ratio.

<table><tr><td>Teacher pair</td><td>shared</td><td>At least one At least half Average covered</td><td>overlap</td></tr><tr><td>Claude vs. Gemini</td><td>93.0%</td><td>58.5%</td><td>0.540</td></tr><tr><td>Claude vs. GPT</td><td>98.0%</td><td>80.0%</td><td>0.608</td></tr><tr><td>Gemini vs. GPT</td><td>83.5%</td><td>47.5%</td><td>0.453</td></tr><tr><td>Average</td><td>91.5%</td><td>62.0%</td><td>0.534</td></tr></table>

Table 14: Cross-teacher rubric overlap on a fixed MMRB2 subset. Inputs and human preference labels are held constant across teacher families.

Across teacher pairs, 91.5% of comparisons share at least one semantic dimension, 62.0% cover at least half of the smaller rubric, and the average overlap is 0.534. Because the generated rubrics generally contain only three to four dimensions, these results indicate substantial agreement across teacher families. Complete agreement is neither observed nor expected: image preference evaluation is inherently subjective, and diferent valid criteria can support the same human preference. Overall, the teachers recover a shared evaluation core while retaining reasonable diferences in criterion selection and granularity.

## C Prompt Templates

This appendix provides the inference prompt templates used by RubricRM. Sections C.1 and C.2 specify the required output structure, including task analysis, dynamic dimension selection, dimension-level scoring, weighted aggregation, and final preference selection.

## C.1 Text-to-Image Preference Evaluation

Text-to-Image Generation   
# Role   
You are an expert evaluator for text−to−image (T2I) generation, specializing in multi−dimensional image   
assessment. You are skilled at identifying the most relevant evaluation dimensions, assigning appropriate weights   
, and delivering precise, logically grounded comparisons.   
# Workflow   
1. Intent Mining: Analyze the prompt to identify its core objective, including required content, key attributes,   
stylistic constraints, compositional requirements, and any other critical conditions, while also taking into   
account general image evaluation principles.   
2. Dimension Selection & Weighting: Select 3−5 evaluation dimensions that are most relevant to the task. All   
dimensions must be atomic: each dimension should assess exactly one distinct aspect and must not combine multiple   
criteria into a single dimension. Assign weights dynamically based on their importance. The total weight must   
sum to 100%.   
3. Dimension−based Scoring: Evaluate each image on every selected dimension using the following 0−4 rubric.   
Scores must be assigned relative to the specific dimension being evaluated.   
0: Failed: Does not satisfy the dimension; severe errors or breakdowns are present.   
1: Poor: Satisfies the dimension only weakly; major deviations, omissions, or artifacts are present.   
2: Fair: Partially satisfies the dimension; the intended quality is present, but notable issues remain.   
3: Good: Satisfies the dimension well; only minor flaws are present.   
4: Excellent: Fully satisfies the dimension; highly consistent and essentially free of noticeable flaws.   
# Output Format   
## [Thinking Process]   
Task Analysis: Systematically analyze the prompt.   
− Selected Dimensions & Weights: List the chosen dimensions and explain why each one is important, including the   
rationale for its assigned weight.   
## [Detailed Evaluation]   
[Dimension Name] ([Weight]%)   
− Image A: [brief analysis] → Score: X/4   
− Image B: [brief analysis] → Score: X/4   
## [Final Conclusion]   
− Weighted Total Score: For each image, show the weighted score calculation and provide the final result rounded   
to 2 decimal places.   
− Summary: Briefly summarize the key reasons behind the evaluation.   
− Preference: You ∗∗must∗∗ output exactly one answer −− \boxed{{A}} or \boxed{{B}}. A definitive choice is   
required even when scores are tied.   
# Prompt   
{prompt}   
# Images   
Image A: <image>   
Image B: <image>

## C.2 Image-Editing Preference Evaluation

```markdown
Image Editing
# Role
You are an expert evaluator for image editing, specializing in multi−dimensional assessment of edited images. You
are skilled at identifying the most relevant evaluation dimensions, assigning appropriate weights, and
delivering precise, logically grounded comparisons.
# Workflow
1. Intent Mining: Analyze the editing instruction and original image to identify its core objective, including
required edits, key attributes, stylistic constraints, content preservation requirements, and any other critical
conditions, while also taking into account general image evaluation principles.
2. Dimension Selection & Weighting: Select 3−5 evaluation dimensions that are most relevant to the task. All
dimensions must be atomic: each dimension should assess exactly one distinct aspect and must not combine multiple
criteria into a single dimension. Assign weights dynamically based on their importance. The total weight must
sum to 100%.
3. Dimension−based Scoring: Evaluate each image on every selected dimension using the following 0−4 rubric.
Scores must be assigned relative to the specific dimension being evaluated.
0: Failed: Does not satisfy the dimension; severe errors or breakdowns are present.
1: Poor: Satisfies the dimension only weakly; major deviations, omissions, or artifacts are present.
2: Fair: Partially satisfies the dimension; the intended quality is present, but notable issues remain.
3: Good: Satisfies the dimension well; only minor flaws are present.
4: Excellent: Fully satisfies the dimension; highly consistent and essentially free of noticeable flaws.
# Output Format
## [Thinking Process]
Task Analysis: Systematically analyze the editing instruction.
Selected Dimensions & Weights: List the chosen dimensions and explain why each one is important, including the
rationale for its assigned weight.
## [Detailed Evaluation]
[Dimension Name] ([Weight]%)
− Image A: [brief analysis] → Score: X/4
− Image B: [brief analysis] → Score: X/4
## [Final Conclusion]
Weighted Total Score: For each image, show the weighted score calculation and provide the final result rounded
to 2 decimal places.
Summary: Briefly summarize the key reasons behind the evaluation.
− Preference: You ∗∗must∗∗ output exactly one answer −− \boxed{{A}} or \boxed{{B}}. A definitive choice is
required even when scores are tied.
# Editing Instruction
{prompt}
# Images
Image A: <image>
Image B: <image>
```

## D Running Cases

## D.1 Text-to-Image Generation

## Input Prompt

A cranberry bog flooded for harvest with visible red berries, featuring geese flying in V−shaped flocks during autumn.

![](images/d0575b7d7546a1e63888b4428012899ac673f3bb784cda9cd533db299cfb7367.jpg)  
Image A

![](images/2c448007f498235cbd228df5b49b5872cff305d838c80f78138d64f42dca8382.jpg)  
Image B

## Model Output

```markdown
## [Thinking Process]
− ∗∗Task Analysis∗∗: The prompt requires a specific agricultural scene: a flooded cranberry bog during harvest.
Key elements include visible red berries, geese flying in a V−shape, and an autumn setting. The goal is to
capture the unique "harvest" look of cranberries (which float when flooded) while maintaining naturalistic
details and seasonal atmosphere.
− ∗∗Selected Dimensions & Weights∗∗:
1. ∗∗Prompt Adherence (Content)∗∗ (30%): Evaluates the presence of all required elements: flooded bog, red
berries, and geese in a V−shape.
2. ∗∗Visual Realism & Detail∗∗ (30%): Assesses the quality of the rendering, specifically the texture of the
berries and the anatomical correctness of the geese.
3. ∗∗Composition & Aesthetics∗∗ (20%): Evaluates the layout, lighting, and how well the "autumn" atmosphere
is conveyed.
4. ∗∗Logical Consistency∗∗ (20%): Checks for AI artifacts, such as impossible geometry or illogical placement
of objects (e.g., berries in the water vs. on the ground).
## [Detailed Evaluation]
− ∗∗Prompt Adherence (Content)∗∗ (30%)
− Image A: Includes a flooded bog, visible red berries, and geese in a V−shape. The autumn setting is clear. →
Score: 4/4
− Image B: Includes a flooded bog, red berries, and geese. However, the geese are not in a clear V−shape, and
the "flooded" aspect is rendered as strange horizontal stripes. → Score: 2/4
− ∗∗Visual Realism & Detail∗∗ (30%)
− Image A: The berries have a realistic texture and scale. The geese are silhouetted but clearly defined. The
lighting is natural. → Score: 4/4
− Image B: The berries look like a repetitive, low−resolution texture. The geese are poorly rendered, with
distorted wings and unnatural poses. → Score: 1/4
− ∗∗Composition & Aesthetics∗∗ (20%)
− Image A: Excellent use of perspective and lighting. The sunset/autumn colors in the trees provide a beautiful
contrast to the red berries. → Score: 4/4
− Image B: The composition is very flat and repetitive. The horizontal banding of the bog is visually jarring
and lacks depth. → Score: 1/4
− ∗∗Logical Consistency∗∗ (20%)
− Image A: The scene is logically consistent. The berries are floating on the surface as they would during a
harvest. → Score: 4/4
− Image B: The image contains significant logical errors. The horizontal stripes of berries and water are
illogical for a flooded bog. Some geese appear to be floating or walking on the water surface in a physically
impossible way. → Score: 0/4
## [Final Conclusion]
− ∗∗Weighted Total Score∗∗:
− Image A: (4 ∗ 0.30) + (4 ∗ 0.30) + (4 ∗ 0.20) + (4 ∗ 0.20) = ∗∗4.00∗∗
− Image B: (2 ∗ 0.30) + (1 ∗ 0.30) + (1 ∗ 0.20) + (0 ∗ 0.20) = ∗∗1.10∗∗
− ∗∗Summary∗∗: Image A is a closer realization of the prompt. It captures the specific agricultural look of a
cranberry harvest with realistic details, autumn lighting, and logical composition. Image B is weaker in realism
and logic, featuring repetitive textures, distorted birds, and an illogical "striped" interpretation of a flooded
bog.
− ∗∗Preference∗∗: \boxed{A}
```

## D.2 Image Editing

![](images/e3cfd78c87ffc09161bed7aa79469fe8f6b7c64704a1d65d66a89784f88bca39.jpg)

Model Output   
## [Thinking Process]   
− ∗∗Task Analysis∗∗: The user wants to transform a simple image of a toy airplane into a colorful educational   
poster for a kindergarten classroom. The instruction includes several strict requirements:   
1. ∗∗Style∗∗: Colorful, educational poster for a kindergarten classroom.   
2. ∗∗Centerpiece∗∗: The original toy airplane image.   
3. ∗∗Text & Placement∗∗:   
Top: "Let's Learn About Airplanes!"   
∗ Left side: "Airplanes travel high in the sky and can go really fast!"   
∗ Right side: "Always fasten your seatbelt when flying!"   
∗ Bottom: "Can you name the colors of this plane?" in bold letters.   
− ∗∗Selected Dimensions & Weights∗∗:   
1. ∗∗Text Accuracy & Placement (40%)∗∗: This is the most complex and specific part of the prompt. It evaluates   
whether all four text elements are present, spelled correctly, and placed in the exact requested locations.   
2. ∗∗Thematic Style & Aesthetics (30%)∗∗: Evaluates whether the overall design successfully captures the "   
colorful educational poster for a kindergarten classroom" vibe.   
3. ∗∗Content Preservation (30%)∗∗: Evaluates how well the original toy airplane is preserved as the   
centerpiece, without introducing unwanted artifacts or altering its core identity.   
## [Detailed Evaluation]   
− ∗∗Text Accuracy & Placement (40%)∗∗   
− Image A: All four text elements are present, spelled correctly, and placed exactly where requested (top, left   
, right, bottom). The bottom text is also bold, as requested. → Score: 4/4   
− Image B: All four text elements are present and spelled correctly. The text is placed in the correct   
locations. The bottom text is bold. → Score: 4/4   
− ∗∗Thematic Style & Aesthetics (30%)∗∗   
− Image A: The image features a bright, colorful, and playful design with clouds, a sunburst background, and   
kid−friendly graphics. It perfectly captures the requested kindergarten classroom poster aesthetic. → Score: 4/4   
− Image B: The image is colorful and features a sky background, but the overall aesthetic feels a bit more   
generic and less cohesive as a kindergarten poster compared to Image A. → Score: 3/4   
− ∗∗Content Preservation (30%)∗∗   
− Image A: The original toy airplane is perfectly preserved in the center. The image also includes some nice,   
non−distracting background elements (like a small cartoon plane and city skyline) that fit the theme. → Score:   
4/4   
− Image B: The original toy airplane is preserved, but the model added strange, incorrect text labels ("Black")   
pointing to the propeller and wings, which detracts from the original image's integrity. → Score: 2/4   
## [Final Conclusion]   
− ∗∗Weighted Total Score∗∗:   
− Image A: (4 ∗ 0.40) + (4 ∗ 0.30) + (4 ∗ 0.30) = 1.60 + 1.20 + 1.20 = 4.00   
− Image B: (4 ∗ 0.40) + (3 ∗ 0.30) + (2 ∗ 0.30) = 1.60 + 0.90 + 0.60 = 3.10   
− ∗∗Summary∗∗: Both images successfully followed the strict text and placement instructions. However, Image A is   
vastly superior in its overall aesthetic, creating a much more engaging and appropriate kindergarten poster.   
Furthermore, Image A perfectly preserves the original toy airplane, whereas Image B introduces unwanted and   
incorrect text labels pointing to the plane.   
− ∗∗Preference∗∗: \boxed{A}