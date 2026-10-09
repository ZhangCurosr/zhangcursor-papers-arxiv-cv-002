# GroundSight at GroundLM 2026 Shared Tasks: GoldenViewVQA

Kun Wang<sup>1,3</sup>, Yupeng Hu<sup>1</sup>, Ruping Cao<sup>1</sup>, Hao Liu<sup>1</sup>, Zhiran Li<sup>1</sup> Qianlong Xiang<sup>2</sup>, Harry Cheng<sup>3</sup>

<sup>1</sup>School of Software, Shandong University, Jinan, China <sup>2</sup>School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China <sup>3</sup>School of Computing, National University of Singapore, Singapore khylon.kun.wang@gmail.com huyupeng@sdu.edu.cn caoruping657@gmail.com liuh90210@gmail.com zhiranli325@gmail.com xiangqianlongcs@gmail.com xacheng1996@gmail.com

## Abstract

GoldenViewVQA requires models to jointly answer driving-scene questions and identify the camera view containing the supporting visual evidence, making precise evidence localization as important as answer correctness. We present CoVeR-VQA, a training-free multistage verification and correction framework for grounded multi-view VQA. Starting from GPT-5.6 zero-shot predictions, CoVeR-VQA progressively applies view-specific verification with Gemini-3.6-Flash, prior-guided joint verification with Claude-Opus-5, and cross-split group-level verification that exploits semantically filtered question groups from shared multi-view scenes and validation-derived prior knowledge. On the official GoldenViewVQA test set, the four-stage CoVeR-VQA pipeline achieves 84.75% Joint Accuracy, improving the GPT-5.6 zero-shot baseline by 13.56 percentage points, while reaching 94.92% Answer Accuracy and 86.44% View Accuracy. The final submitted run achieves 88.14% Joint Accuracy after two additional evaluator-informed post-hoc corrections. Our analysis shows that supporting-view localization remains the primary source of residual errors, highlighting the importance of explicit evidence verification for reliable multi-view multimodal reasoning.

## 1 Introduction

Multimodal large language models (MLLMs) have demonstrated strong capabilities in visual representation and understanding, visual question answering, cross-modal retrieval and multimodal evaluation [Bai et al., 2025, Liu et al., 2025a, Wang et al., 2025, Li et al., 2026a]. Visual question answering has also been increasingly explored in autonomousdriving scenarios [Qian et al., 2024]; however, producing a correct answer alone is insufficient for reliable grounded reasoning. A reliable system should also identify the visual evidence that supports its prediction. This requirement is particularly challenging in multi-view settings, where a scene is simultaneously observed from multiple cameras with different fields of view [Yeh et al., 2026]. Relevant evidence may appear in only one view or be ambiguous across several cameras, making precise evidence localization substantially harder than answer prediction itself [Hu et al., 2021a,b, 2023, Wang et al., 2024, Liu et al., 2025b, Wang et al., 2026a, Hu et al., 2026a].

GroundLM 2026 comprises two shared tasks, GoldenViewVQA [Wang et al., 2026b] and Lit-TraceQA [Liu et al., 2026a]; the overall task setup and findings are summarized in the shared-task findings paper [Wang et al., 2026c]. In this work, we focus on GoldenViewVQA, which explicitly evaluates view-level visual evidence identification in multi-view autonomous-driving scenarios. Each instance contains six synchronized driving-scene camera views and a multiple-choice question, and systems are required to predict both the correct answer and its supporting camera view. The primary metric, Joint Accuracy, considers a prediction correct only when both outputs are correct. Our zero-shot experiments reveal a clear gap between the two objectives: GPT-5.6 achieves 91.53% Answer Accuracy but only 74.58% View Accuracy, resulting in 71.19% Joint Accuracy. This suggests that improving grounded multi-view reasoning requires explicit mechanisms for verifying visual evidence rather than relying solely on stronger answer generation.

To address this issue, we propose CoVeR-VQA, a collaborative multi-stage verification and correction framework for grounded multi-view VQA. CoVeR-VQA starts from a zero-shot joint answer– view prediction produced by GPT-5.6 and subsequently applies three complementary verification stages for view auditing, joint correction, and crosssplit group-level reasoning. Gemini-3.6-Flash first performs a dedicated audit of the predicted supporting view. Claude-Opus-5 then conducts confidenceaware joint verification of both the answer and the view. In the final stage, semantically filtered question groups across shared multi-view scenes are considered collectively to provide group-level prior knowledge for further verification and correction. When validation instances are available in the same group, their annotated answers and supporting views are additionally exploited as verification priors. This staged design targets different sources of error while reducing unnecessary changes to already reliable predictions. The resulting pipeline improves Joint Accuracy from 71.19% to 84.75%, with Answer Accuracy and View Accuracy reaching 94.92% and 86.44%, respectively. The final submitted run achieves 88.14% Joint Accuracy on the official evaluator.

In summary, our contributions are:

• We propose CoVeR-VQA, a training-free multi-stage framework that jointly improves answer prediction and view-level visual grounding in multi-view driving VQA.

• We introduce complementary verification and correction mechanisms that target viewselection errors, coupled answer–view errors, and prediction errors that can be resolved using semantically filtered group-level priors derived from related questions and, when available, validation instances within the same scene.

• We analyze the contribution of individual stages and show that explicit evidence verification substantially improves joint grounded reasoning over strong zero-shot MLLM baselines.

## 2 Task and Data

## 2.1 Task Definition

GoldenViewVQA [Wang et al., 2026b] evaluates multi-view visual question answering with explicit evidence-source identification. For each instance, the model receives six synchronized camera observations from the NuScenes autonomous-driving dataset [Caesar et al., 2020], corresponding to the front, front-left, front-right, back, back-left, and back-right views. Each instance additionally contains a natural-language question and four candidate answers. The system is required to jointly predict the answer and the camera view that provides the supporting visual evidence.

Formally, an input instance is represented as

$$
x _ { i } = ( \mathbb { Z } _ { i } , q _ { i } , \mathbb { \mathcal { A } } _ { i } ) ,\tag{1}
$$

where $\mathcal { T } _ { i } = \{ I _ { i } ^ { 1 } , \ldots , I _ { i } ^ { 6 } \}$ denotes the six synchronized camera images, q<sub>i</sub> denotes the question, and $\mathcal { A } _ { i } = \{ A _ { i } , B _ { i } , C _ { i } , D _ { i } \}$ denotes the four answer candidates. Given the input, the system predicts

$$
\begin{array} { r } { ( \hat { v } _ { i } , \hat { a } _ { i } ) = f ( \mathbb { Z } _ { i } , q _ { i } , \mathcal { A } _ { i } ) , } \end{array}\tag{2}
$$

where $\hat { v } _ { i }$ is the predicted supporting view and $\hat { a } _ { i }$ is the predicted answer. The supporting-view prediction is selected from the six camera views or NONE\_OF\_THE\_ABOVE, while the answer prediction satisfies $\hat { a } _ { i } \in \{ A , B , C , D \}$

The benchmark covers three categories of driving-scene reasoning: causality, counterfactual reasoning, and intent prediction. Causality questions require identifying the visual factors that explain a driving decision or maneuver. Counterfactual questions ask how the situation would change under a hypothetical condition, while intentprediction questions require inferring the likely future behavior of surrounding road users. Solving these questions often requires localized evidence involving traffic signals, pedestrians, road geometry, parked or moving vehicles, and other scene-specific constraints.

The primary evaluation metric is Joint Accuracy, which considers an instance correct only when both the answer and the supporting view are predicted correctly:

$$
\mathrm { J o i n t A c c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf 1 } \left[ \hat { v } _ { i } = v _ { i } \wedge \hat { a } _ { i } = a _ { i } \right] .\tag{3}
$$

In addition to Joint Accuracy, the official evaluator reports Answer Accuracy, View Accuracy, and View Macro Accuracy. These complementary metrics allow us to distinguish answer-generation errors from evidence-localization errors, which is particularly important because a model may produce the correct answer while grounding it in an incorrect camera view.

## 2.2 Dataset

GoldenViewVQA is constructed from multi-view driving scenes in the NuScenes dataset [Caesar et al., 2020]. Each example contains six synchronized camera images together with a multiplechoice question, four candidate answers, and a supporting-view annotation. The shared-task release contains 114 instances in total, including 55 labeled validation examples and 59 unlabeled test examples, corresponding to 330 and 354 camera images, respectively. In the validation split, causality, counterfactual reasoning, and intent prediction account for 49.09%, 27.27%, and 23.64% of the instances, respectively; the corresponding proportions in the test split are 35.59%, 30.51%, and 33.90%. Across the complete shared-task release, causality accounts for 48 instances (42.11%), while counterfactual reasoning and intent prediction each contain 33 instances (28.95%).

![](images/e6568b3513f7b0fcbaf3a82cf82fa23af1dda8d0eb77190dff23bcc3c267c16f.jpg)  
Figure 1: Overview of the proposed CoVeR-VQA framework and its four-stage prediction and verification pipeline.

For validation examples, both the correct answer and the supporting camera view are provided. The official test set follows the same input format, but its answer and supporting-view annotations are withheld; system predictions are evaluated through the GroundLM online evaluator. We use the public validation set to understand the task structure, examine common grounding ambiguities, and design the prompting and verification strategies used in CoVeR-VQA. Although the validation and test splits are evaluated separately, some instances correspond to the same underlying multi-view driving scenes. This cross-split scene relationship enables our Stage 4 verification strategy.

## 3 CoVeR-VQA System

As illustrated in Figure 1, CoVeR-VQA adopts a progressive multi-model verification framework for grounded multi-view visual question answering. Starting from a zero-shot joint answer–view prediction produced by GPT-5.6 [OpenAI, 2026], the system sequentially applies view-specific verification with Gemini-3.6-Flash [Google DeepMind, 2026], confidence-aware joint verification with Claude-Opus-5 [Anthropic, 2026], and cross-split grouplevel verification with GPT-5.6. Each stage receives the prediction from the preceding stage and targets a distinct source of error, including supportingview mislocalization, coupled answer–view errors, and residual errors that can be identified through semantically related question groups and validationderived prior knowledge.

## 3.1 Stage 1: Zero-Shot Joint Prediction

The first stage uses GPT-5.6 to obtain an initial joint prediction. For each instance, all six synchronized camera views are presented together with their camera identifiers, the question, and four candidate answers. The model is instructed to jointly determine the most appropriate supporting view and the corresponding answer choice.

We retain all six views during this stage rather than selecting a camera in advance. This allows the model to directly compare evidence across different viewing directions and reduces the risk of discarding useful information through premature view selection.

The resulting prediction is represented as

$$
\begin{array} { r } { P _ { i } ^ { ( 1 ) } = \left( \hat { v } _ { i } ^ { ( 1 ) } , \hat { a } _ { i } ^ { ( 1 ) } \right) , } \end{array}\tag{4}
$$

where $\hat { v } _ { i } ^ { ( 1 ) }$ and $\hat { a } _ { i } ^ { ( 1 ) }$ denote the predicted supporting view and answer, respectively.

Although this zero-shot setting already provides strong answer prediction, our experiments show that errors in supporting-view identification remain considerably more frequent. This discrepancy motivates the explicit view verification introduced in the next stage.

## 3.2 Stage 2: View-Specific Verification

The second stage focuses exclusively on the supporting-view prediction. Gemini-3.6-Flash receives the original six camera views, the question and answer candidates, as well as the answer–view pair produced by Stage 1.

Rather than solving the complete task again, the reviewer is instructed to determine whether the current supporting view actually contains the visual evidence required to justify the predicted answer. It compares the candidate views and re-evaluates relevant objects, spatial relations, and scene constraints.

Importantly, the answer produced in Stage 1 is kept unchanged during this stage. Only the supporting-view prediction is allowed to be corrected:

$$
P _ { i } ^ { ( 2 ) } = \left( \hat { v } _ { i } ^ { ( 2 ) } , \hat { a } _ { i } ^ { ( 1 ) } \right) .\tag{5}
$$

This design is motivated by the observation that a model may already produce the correct answer while grounding it in an incorrect camera view. Restricting Stage 2 to view verification therefore reduces the risk of replacing an already-correct answer while correcting the grounding decision.

## 3.3 Stage 3: Prior-Guided Confidence-Aware Joint Verification

Although Stage 2 improves supporting-view localization, it cannot correct cases in which the answer itself is incorrect or both components of the prediction are unreliable. We therefore introduce a Claude-based joint verification stage that simultaneously re-examines the answer and supporting view.

After Stage 2, the official evaluation metrics provide additional information about the aggregate error structure of the current predictions. Specifically, among the 59 test instances, 44 predictions are jointly correct, 10 contain only a view error, 3 contain only an answer error, and 2 contain errors in both components. We treat this aggregate distribution as Stage-2-derived prior knowledge. Importantly, this prior describes only the error distribution of the entire test set and does not reveal which individual instances belong to each category.

For each instance, Claude receives the six synchronized camera views, the question and answer candidates, the current Stage 2 prediction $P _ { i } ^ { ( 2 ) }$ , and the Stage-2-derived prior.

Claude independently audits the current prediction and estimates its likelihood under four possible

states:

$$
\mathcal { Z } = \{ \mathrm { C } , \mathrm { V } , \mathrm { A } , \mathrm { B } \} ,\tag{6}
$$

where C indicates that both the view and answer are correct, V represents a view-only error, A represents an answer-only error, and B indicates that both predictions are incorrect. The reviewer also proposes corresponding correction candidates when it identifies potential errors.

The aggregate prior is used only to provide the reviewer with information about the overall reliability and error composition of the Stage 2 predictions. We do not enforce the 44/10/3/2 distribution as a hard assignment over individual instances. Instead, the final correction decision is determined by Claude’s instance-level confidence.

Let $p _ { i } ( \mathrm { C } )$ denote Claude’s estimated probability that the current answer–view pair is jointly correct. To avoid unnecessarily altering reliable predictions [Hu et al., 2026b, Liu et al., 2026b], we apply a conservative confidence threshold and revise only instances for which

$$
p _ { i } ( \mathrm { C } ) \leq 0 . 5 .\tag{7}
$$

The Stage 3 prediction is therefore defined as

$$
P _ { i } ^ { ( 3 ) } = \left\{ { \begin{array} { l l } { { \tilde { P } _ { i } ^ { ( 3 ) } , } } & { { p _ { i } ( \mathrm { C } ) \leq 0 . 5 , } } \\ { { P _ { i } ^ { ( 2 ) } , } } & { { p _ { i } ( \mathrm { C } ) > 0 . 5 , } } \end{array} } \right.\tag{8}
$$

where $\tilde { P } _ { i } ^ { ( 3 ) }$ denotes the correction proposed by Claude after jointly reconsidering the supporting view and answer.

This design combines collection-level prior knowledge with instance-level multimodal verification. The prior informs Claude that most Stage 2 predictions are already reliable and provides information about the relative prevalence of different error types, while the confidence threshold prevents this aggregate information from forcing a predetermined number of modifications. As a result, Stage 3 selectively targets predictions for which the reviewer finds stronger evidence against the current answer–view pair [Liu et al., 2026c, Yu et al., 2026].

## 3.4 Stage 4: Cross-Split Group-Level Prior-Guided Verification

The previous stages process each question largely independently, although multiple instances may correspond to the same multi-view scene and involve related entities or spatial relations. Stage 4 therefore introduces group-level verification by exploiting semantically related questions across both the validation and test splits.

We first merge the validation and test instances and construct candidate scene groups by exploiting the shared base sfall\_XXXX identifier together with synchronized camera observations. Since questions from the same scene may still refer to different objects or require different evidence, we further apply GPT-5.6 as a text-only semantic filter. The model receives only the question stems and retains questions in the same group when they either concern the same core entity and local spatial relation, or exhibit a sufficiently strong semantic, causal, or behavioral connection within the scene. Semantically unrelated questions are separated, and singleton groups are discarded. The resulting group set is

$$
\mathcal { G } = \{ G _ { 1 } , G _ { 2 } , \dots , G _ { M } \} , \qquad | G _ { m } | \geq 2 .\tag{9}
$$

For each retained group, GPT-5.6 is provided with the shared six-view scene, the questions, answer options, available annotations of group members, and the current predictions from Stage 3. It then re-evaluates the answer and supporting view of each test instance. We denote the textual group prior by $\Pi _ { \mathrm { t e x t } } ( G _ { m } )$ and the validationderived prior by $\Pi _ { \mathrm { v a l } } ( G _ { m } )$ . For test-only groups, $\Pi _ { \mathrm { v a l } } ( G _ { m } ) = \emptyset$ , whereas for cross-split groups it contains the ground-truth answers and supporting views of the related validation instances. The complete group-level prior is therefore

$$
\Pi ( G _ { m } ) = \Pi _ { \mathrm { t e x t } } ( G _ { m } ) \cup \Pi _ { \mathrm { v a l } } ( G _ { m } ) .\tag{10}
$$

Validation annotations provide a particularly strong reference when the validation and test questions concern the same core entity or local spatial configuration. In such cases, GPT-5.6 is encouraged to maintain supporting-view consistency with the validation instances. This consistency is not enforced as a hard constraint, however, since different questions from the same scene may require different decisive evidence. A different view may therefore be selected when supported by clear question-specific visual evidence.

For a test instance i belonging to group $G _ { m }$ Stage 4 jointly conditions on the original input $x _ { i }$ the previous prediction $P _ { i } ^ { ( 3 ) }$ , and the corresponding group-level prior $\Pi ( G _ { m } )$

$$
P _ { i } ^ { ( 4 ) } = f _ { \mathrm { g r o u p } } \left( x _ { i } , P _ { i } ^ { ( 3 ) } , \Pi ( G _ { m } ) \right) .\tag{11}
$$

By combining semantic group filtering with validation-derived cross-split priors, Stage 4 uses related questions as targeted prior knowledge while avoiding indiscriminate information transfer between questions that merely share the same scene. This design is particularly useful for correcting residual answer–view inconsistencies that are difficult to resolve from individual questions alone.

## 4 Experimental Results

## 4.1 Experimental Details

Our experiments involve four multimodal large language models: Qwen3-VL-8B [Bai et al., 2025], GPT-5.6 [OpenAI, 2026], Gemini-3.6- Flash [Google DeepMind, 2026], and Claude-Opus-5 [Anthropic, 2026]. Qwen3-VL-8B is used as an open-model zero-shot baseline, while GPT-5.6, Gemini-3.6-Flash, and Claude-Opus-5 are used at different stages of the CoVeR-VQA pipeline. We do not perform task-specific finetuning or use additional labeled training data; all improvements are obtained through inference-time prediction, verification, and correction. Detailed prompts are provided in the accompanying code submission.

Stage 3 uses only aggregate evaluator statistics from the complete Stage 2 submission as a collection-level prior and applies a fixed confidence-based verification procedure to all instances. Stage 4 further uses semantically filtered relationships between validation and test instances. Validation annotations are only introduced within cross-split groups identified by shared scenes and semantic relevance. The resulting fourstage CoVeR-VQA pipeline achieves 84.75% Joint Accuracy on the official GoldenViewVQA evaluator. For the final submission, we additionally examined two instances that Claude identified as potentially erroneous but that did not satisfy the correction threshold. Based on subsequent evaluator feedback, we applied two instance-level post-hoc corrections, increasing Joint Accuracy to 88.14%. Since these corrections do not constitute a generalizable inference component, we report the final submission separately from the four-stage pipeline. The two corrected instances are revisited in Section 4.4 as representative pre-correction failure cases.

<table><tr><td>Method</td><td>Joint</td><td>Ans.</td><td>View</td><td>V-Macro</td></tr><tr><td>Organizer Baseline</td><td>20.34</td><td>23.73</td><td>77.97</td><td>16.67</td></tr><tr><td>Qwen3-VL-8B Zero Shot</td><td>66.10</td><td>88.14</td><td>71.19</td><td>49.88</td></tr><tr><td>① GPT-5.6 Zero Shot</td><td>71.19</td><td>91.53</td><td>74.58</td><td>65.82</td></tr><tr><td>+ GPT-5.6 View Review</td><td>72.88</td><td>91.53</td><td>77.97</td><td>64.13</td></tr><tr><td>(2) Gemini View Review</td><td>74.58</td><td>91.53</td><td>79.66</td><td>54.11</td></tr><tr><td>3 Claude Joint Review</td><td>79.66</td><td>93.22</td><td>83.05</td><td>54.83</td></tr><tr><td>Cross-Split Group Verify 4</td><td>84.75</td><td>94.92</td><td>86.44</td><td>65.94</td></tr><tr><td>CoVeR-VQA Pipeline</td><td>84.75</td><td>94.92</td><td>86.44</td><td>65.94</td></tr><tr><td>Final Submission</td><td>88.14</td><td>96.61</td><td>89.83</td><td>74.64</td></tr></table>

Table 1: Results on the GoldenViewVQA test set (%). Shaded rows marked with ⃝1 –⃝4 denote the four stages of the CoVeR-VQA pipeline, while “+” denotes an additional ablation setting.  
![](images/c1e90c6a05c0da33214bee5179245fc22a5f6bc80104a2a4b924f569e4f8c24a.jpg)  
View Prediction  
Figure 2: Distribution of answer and supporting-view correctness for the final submission on the Golden-ViewVQA test set.

## 4.2 Overall Performance

Table 1 summarizes the performance of the evaluated systems on the GoldenViewVQA test set. The organizer baseline achieves only 20.34% Joint Accuracy despite a relatively high View Accuracy of 77.97%, indicating that accurate view prediction alone is insufficient without reliable answer reasoning. The zero-shot Qwen3-VL-8B baseline substantially improves Joint Accuracy to 66.10%, while GPT-5.6 further raises it to 71.19%, together with 91.53% Answer Accuracy and 74.58% View Accuracy.

The complete four-stage CoVeR-VQA pipeline reaches 84.75% Joint Accuracy, improving the GPT-5.6 zero-shot baseline by 13.56 percentage points. Answer Accuracy increases from 91.53% to 94.92%, while View Accuracy improves from 74.58% to 86.44%. These results show that the proposed verification stages improve not only answer correctness but, more importantly, the consistency between the predicted answer and its supporting visual evidence.

The final submitted system achieves 88.14% Joint Accuracy, 96.61% Answer Accuracy, and 89.83% View Accuracy. As illustrated in Figure 2,

![](images/d6c629b90dadc19972d7f02e35bad71a241e75f1e3f55274f61acce6bcae7e94.jpg)  
Figure 3: Example of a view-selection error caused by confusing object saliency with decisive spatial evidence.

52 of the 59 test instances are correct in both answer and view prediction. Among the remaining seven instances, five have the correct answer but an incorrect supporting view, while only one instance contains an answer-only error and one contains errors in both components. This error distribution further suggests that view localization remains the dominant source of residual error after answer prediction has largely been resolved.

## 4.3 Stage-wise Analysis and Ablations

The stage-wise results in Table 1 reveal complementary effects across the CoVeR-VQA pipeline. Compared with the GPT-5.6 self-review ablation, Gemini-3.6-Flash increases View Accuracy from 77.97% to 79.66% and Joint Accuracy from 72.88% to 74.58%, while Answer Accuracy remains unchanged at 91.53%. This comparison confirms the effectiveness of the Stage 2 review mechanism and further indicates that cross-model verification is more effective than self-review with the same model for supporting-view correction.

The largest improvement is obtained at Stage 3. Claude-based joint verification raises Joint Accuracy from 74.58% to 79.66%, with simultaneous gains in Answer Accuracy and View Accuracy from 91.53% to 93.22% and from 79.66% to 83.05%, respectively. This result supports the motivation for jointly reconsidering answer and view predictions after view-specific auditing, as some remaining errors cannot be resolved by modifying only the supporting view.

Stage 4 further improves Joint Accuracy to 84.75% and Answer Accuracy to 94.92%, while View Accuracy further increases to 86.44%. This suggests that group-level verification provides complementary information for correcting residual prediction errors, where related question contexts offer semantic priors and validation annotations provide stronger evidence-aware guidance when available. Notably, View Macro Accuracy does not increase monotonically across stages, indicating that improvements in overall view correctness are not uniformly distributed across all view categories.

![](images/d46c0db5722dccf831253a7b99c96837f2726e3b37c4112666fae66b4a8786b3.jpg)  
Figure 4: Example of unsupported semantic inference under ambiguous visual evidence.

Overall, the ablation results show that the gains of CoVeR-VQA arise from complementary verification mechanisms rather than repeated applications of the same reviewer. View-specific auditing primarily improves grounding, joint verification corrects coupled answer–view errors, and grouplevel prior knowledge provides additional gains for difficult residual cases.

## 4.4 Error Analysis

A major source of error is the confusion between object visibility and decisive spatial evidence. As illustrated in Figure 3, the red parked vehicle is more visually prominent in the front-right camera, which can easily attract the model toward that view. However, the task does not ask which camera provides the clearest or largest appearance of the target object; it asks which view contains the evidence needed to establish the relevant spatial relation. In this example, the front view better reveals the position of the parked vehicle with respect to the ego lane and the usable roadway, thereby supporting the conclusion that the left side is constrained. This failure suggests that supporting-view selection should be based on the complete actionrelevant scene configuration—including object position, lane geometry, ego orientation, and their spatial relations—rather than object saliency alone [Li et al., 2025, Wang et al., 2026d]. In other words, the golden view is the view that provides the decisive evidence for answering the question, not necessarily the view in which the key object appears most prominently.

A second recurring failure mode is unsupported semantic inference. Figure 4 shows an intentprediction example in which pedestrians are visible near the right-side corner, but the available views do not provide sufficient evidence to determine whether they intend to cross, wait for transportation, or continue along the sidewalk. Nevertheless, the model tends to select a visually plausible view and infer a likely behavior instead of recognizing the absence of decisive evidence. Such behavior reflects a form of multimodal hallucination, a known failure mode of large vision-language models [Li et al., 2023, Guan et al., 2024]: when visual evidence is ambiguous or incomplete, the MLLM may fill the missing information with a semantically plausible interpretation derived from common driving scenarios. This is particularly problematic for GoldenViewVQA, where NONE\_OF\_THE\_ABOVE is a valid supporting-view label and some answer choices may explicitly indicate that the available evidence is insufficient. Effective grounded reasoning therefore requires not only identifying supporting evidence when it exists, but also calibrating uncertainty and refraining from committing to an answer–view pair when the observations do not provide sufficient support.

## 5 Conclusion

We presented CoVeR-VQA, a training-free multistage framework for joint answer prediction and view-level evidence grounding in Golden-ViewVQA. By progressively combining zero-shot prediction, view-specific auditing, prior-guided joint verification, and cross-split group-level verification with semantic filtering and validationderived priors, CoVeR-VQA improves Joint Accuracy from 71.19% to 84.75%, with the final submitted run reaching 88.14%. Stage-wise analysis demonstrates that the verification components provide complementary gains in answer correctness and supporting-view localization. Error analysis further reveals that the remaining difficulty lies primarily in identifying decisive spatial evidence rather than salient objects, as well as in avoiding unsupported predictions when visual evidence is insufficient. These findings suggest that reliable grounded multimodal reasoning requires not only stronger semantic reasoning, but also explicit evidence verification and better uncertainty calibration.

## Limitations

Our approach has several limitations. First, the public validation set contains only 55 labeled instances, which limits systematic prompt optimization and error analysis and may make observations derived from the validation data sensitive to individual examples.

Second, CoVeR-VQA remains strongly dependent on the underlying reasoning, calibration, and behavioral reliability of MLLMs [Li et al., 2026b]. As discussed in our error analysis, current MLLMs may produce plausible interpretations even when the available visual evidence is insufficient, rather than explicitly selecting NONE\_OF\_THE\_ABOVE. Such hallucination behavior remains a major challenge for reliable evidence-grounded reasoning, while residual vulnerabilities under model-level interventions represent a broader reliability concern [Xiang et al., 2026, 2025].

Finally, Stage 4 relies on semantically related question groups within shared multi-view scenes and, for cross-split groups, on annotations from related validation instances. This strategy exploits structural relationships available in the released benchmark and is therefore specific to settings where such cross-instance relationships are accessible. These conditions are generally unavailable in standard independent-test or real-time autonomousdriving scenarios, limiting the general applicability of Stage 4 beyond the shared-task setting.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Hao Liu, Kun Wang, Yudong Han, Haocong Wang, Yupeng Hu, Chunxiao Wang, and Liqiang Nie. Curmim: Curriculum masked image modeling. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 1–5, 2025a.

Kun Wang, Yupeng Hu, Hao Liu, Lirong Jie, and Liqiang Nie. Redundancy mitigation: Towards accurate and efficient image-text retrieval. IEEE Transactions on Circuits and Systems for Video Technology, 2025.

Yanlin Li, Minghui Guo, Kaiwen Zhang, Shize Zhang, Yiran Zhao, Haodong Li, Congyue Zhou, Weijie Zheng, Yushen Yan, Shengqiong Wu, et al. Unim: A unified any-to-any interleaved multimodal benchmark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15902–15911, 2026a.

Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal vi

sual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4542–4550, 2024.

Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12000– 12008, 2026.

Yupeng Hu, Meng Liu, Xiaobin Su, Zan Gao, and Liqiang Nie. Video moment localization via deep cross-modal hashing. IEEE Transactions on Image Processing, 30:4667–4677, 2021a.

Yupeng Hu, Liqiang Nie, Meng Liu, Kun Wang, Yinglong Wang, and Xian-Sheng Hua. Coarse-to-fine semantic alignment for cross-modal moment localization. IEEE Transactions on Image Processing, 30: 5933–5943, 2021b.

Yupeng Hu, Kun Wang, Meng Liu, Haoyu Tang, and Liqiang Nie. Semantic collaborative learning for cross-modal moment localization. ACM Transactions on Information Systems, 42(2):1–26, 2023.

Kun Wang, Hao Liu, Lirong Jie, Zixu Li, Yupeng Hu, and Liqiang Nie. Explicit granularity and implicit scale correspondence learning for point-supervised video moment localization. In Proceedings of the ACM International Conference on Multimedia, pages 9214–9223, 2024.

Hao Liu, Yupeng Hu, Kun Wang, Yinwei Wei, and Liqiang Nie. Gaming for boundary: Elastic localization for frame-supervised video moment retrieval. In Proceedings ofthe International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 917–926, 2025b.

Kun Wang, Yupeng Hu, Hao Liu, Jiang Shao, and Liqiang Nie. Cross-modal representation shift refinement for point-supervised video moment retrieval. ACM Transactions on Information Systems, 44(3): 1–30, 2026a.

Yupeng Hu, Han Jiang, Hao Liu, Kun Wang, Haoyu Tang, and Liqiang Nie. Visual self-paced iterative learning for unsupervised temporal action localization. ACM Transactions on Multimedia Computing, Communications and Applications, 2026a.

Yimu Wang, Yee Man Choi, Barry Zhang, Mozhgan Nasr Azadani, Sean Sedwards, and Krzysztof Czarnecki. Where does the answer come from? benchmarking view-level visual evidence identification in multi-view mllms for autonomous driving, 2026b. URL https://arxiv.org/abs/ 2606.09644.

Xuye Liu, Yimu Wang, Peng Shi, Bo Xue, Xiangrui Ke, Songcheng Cai, Kath Choi, Di Wu, Freda Shi, and Krzysztof Czarnecki. Littraceqa: A benchmark

for multi-stage grounding and verification in scientific question answering, 2026a. URL https: //arxiv.org/abs/2608.07370.

Yimu Wang, Xuye Liu, Yee Man Choi, Bo Xue, et al. Findings of the first groundlm shared tasks: Evaluating grounded language models across visual and scientific evidence. In Proceedings of the 1st Workshop on Grounding Language Models: Learning Faithfully and Efficiently (GroundLM 2026), 2026c.

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11618–11628, 2020.

OpenAI. GPT-5.6 System Card, July 2026. URL https://deploymentsafety.openai. com/gpt-5-6. Accessed: 2026-08-16.

Google DeepMind. Gemini 3.6 Flash Model Card, July 2026. URL https://deepmind. google/models/model-cards/ gemini-3-6-flash/. Accessed: 2026-08-16.

Anthropic. Claude Opus 5 System Card, July 2026. URL https://www.anthropic.com/ system-cards. Accessed: 2026-08-16.

Yupeng Hu, Hao Liu, Kun Wang, Ruping Cao, Yinwei Wei, and Liqiang Nie. From a glance to a boundary: Uncertainty-aware distillation for glance-supervised video moment localization. IEEE Transactions on Multimedia, 2026b.

Hao Liu, Yupeng Hu, Kun Wang, Junchao Wang, Ruping Cao, Yutao Yao, and Zilu Cai. From interference to stability: Adversarial reliability correction for video moment retrieval with relevance feedback. In Proceedings of the 49th International ACM SI-GIR Conference on Research and Development in Information Retrieval, pages 1152–1162, 2026b.

Hao Liu, Ruping Cao, Kun Wang, Zhiran Li, Fan Liu, Yupeng Hu, and Liqiang Nie. Chartlens: A dual-branch framework for chart data correction and factual summary refinement. arXiv preprint arXiv:2606.10640, 2026c.

LeKai Yu, Hao Liu, Kun Wang, Zhiran Li, Ruping Cao, Fan Liu, and Yupeng Hu. Parsefixer: An agentic framework for document parsing via selective multimodal correction. arXiv preprint arXiv:2606.11977, 2026.

Ming Li, Yupeng Hu, Yinwei Wei, Hao Liu, Haocong Wang, and Weili Guan. Dcount: Decoupled spatial perception and attribute discrimination for referring expression counting. In Proceedings of the ACM International Conference on Multimedia, pages 5306– 5315, 2025.

Kun Wang, Yupeng Hu, Zhiran Li, Hao Liu, Qianlong Xiang, and Liqiang Nie. Visage @ ntire 2026 challenge on video saliency prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 2697–2702, 2026d.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.

Yanlin Li, Hao Liu, Huimin Liu, Kun Wang, Yinwei Wei, and Yupeng Hu. Mist: Multi-dimensional implicit bias evaluation of llms for theory of mind. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, volume 48, 2026b.

Qianlong Xiang, Miao Zhang, Haoyu Zhang, Kun Wang, Junhui Hou, and Liqiang Nie. Tina: Text-free inversion attack for unlearned text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 30076–30086, 2026.

Qianlong Xiang, Miao Zhang, Yuzhang Shang, Jianlong Wu, Yan Yan, and Liqiang Nie. Dkdm: Data-free knowledge distillation for diffusion models with any architecture. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2965, 2025.