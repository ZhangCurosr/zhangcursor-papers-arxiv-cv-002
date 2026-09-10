# VLX-VR: An Agentic-Aware Video Reasoning Model

Sheng Li Peng Liu Qianqian Zhang Tiancheng Zhao

Om AI Research

tianchez@zju-bj.com

Correspondence: Tiancheng Zhao

## Abstract

Real-world video understanding requires integrating visual, audio, textual, and temporal evidence distributed across a video. Yet many pipelines use a fixed video context and single-pass inference, limiting adaptive evidence acquisition when observations are incomplete, ambiguous, or conflicting. We present VLX-VR, an agentic-aware video reasoning model trained within a video reasoning framework defined by a Think–Memory–Observation loop. At each step, VLX-VR determines the needed evidence, invokes read memory or write memory, incorporates the returned Observation, and decides whether to continue or produce the task output. We train VLX-VR with multimodal data, including videos and agent trajectories, using reinforcement learning to learn evidence acquisition, memory use, and termination. On MINERVA, VLX-VR achieves state-of-the-art performance among the models included in our comparison, with 78.79% accuracy. Under the original three duration groups, its accuracies are 76.70%, 78.73%, and 80.92%, with a cross-duration accuracy variance of $2 . 9 7 \mathrm { p p } ^ { 2 }$ . On correctly answered samples, 96.20% ofVLX-VR’s reasoning traces are consistent with the MINERVA reference reasoning traces and the evidence described by them, while approximately 75.80% ofall evaluated samples satisfy both answer correctness and this evidence-grounded trace criterion. These results show strong performance and broadly stable behavior across durations, while counting, state changes, causal reasoning, and spatial perception remain challenging.

## 1. Introduction

In everyday scenarios, people understand videos by combining visual, audio, textual, and temporal evidence to analyze events and make judgments. Determining what happened, why it happened, or how a situation changed may require locating relevant moments, comparing states before and after an event, reading text in a scene, and relating spoken content to visual actions. Supporting such real-world reasoning requires more than recognizing objects or isolated frames: a model must identify, integrate, and reassess evidence distributed across the video before reaching a reliable conclusion.

In many standard video-analysis and VideoQA pipelines, video large language models such as Video-LLaVA [8], Qwen3-VL [16], and VideoLLaMA 3 [26] typically receive sampled frames or video clips as a fixed input and produce an output in a single inference pass. This fixed-input, single-pass pattern cannot fully satisfy the real-world eventanalysis requirements described above. In such scenarios, a model may need to revisit an earlier event, seek additional evidence after an initial observation, or compare information distributed across time and modalities before making a judgment. Because the evidence is determined before reasoning begins, the model cannot adaptively acquire new evidence after detecting missing, ambiguous, or conflicting information. As a result, insufficient observation may omit a decisive event, whereas indiscriminate observation increases context length and inference cost.

Existing video agents move closer to these real-world requirements through iterative information gathering, temporal localization, and memory queries [4, 18, 31]. However, they may still fall short when a general-purpose VLM is merely placed inside an external agent loop without being trained for its role in that loop. In this setting, the model has not necessarily learned to treat evidence acquisition, memory use, and termination as parts of its reasoning policy. It may request redundant evidence, fail to retain important intermediate states, or produce a result before the available evidence is sufficient. Therefore, reliably analyzing events and making judgments in real-world videos still requires stronger temporal localization, multimodal evidence selection, state maintenance, and consistency between reasoning and actual observations [11].

These limitations motivate the following question: Can a video reasoning model be trained not merely to participate in an agentic pipeline, but to learn evidence acquisition, memory use, and termination as integral parts of its

reasoning process?

To address this question, we introduce a video reasoning framework organized as a Think–Memory–Observation loop and present VLX-VR, an agentic-aware video reasoning model trained to operate within this framework. VLX-VR targets video-analysis tasks that require temporal localization, multimodal evidence selection, state tracking, and multi-step reasoning over both short and long videos. Instead of relying on a fixed video context, VLX-VR learns to acquire and reassess multimodal evidence through the framework-defined loop with direct access to multimodal memory. We train VLX-VR with multimodal data and agent trajectories using reinforcement learning so that evidence acquisition, memory use, and termination become parts of its learned reasoning policy. Section 3 describes the framework and training procedure in detail.

## Our contributions are:

1. We introduce a video reasoning framework that organizes task-conditioned evidence acquisition as a Think– Memory–Observation loop with direct multimodalmemory access.

2. We train VLX-VR as an agentic-aware video reasoning model with multimodal data and agent trajectories using reinforcement learning, enabling the model to learn evidence acquisition, memory read/write behavior, Observation-based reasoning, and termination within the framework-defined loop.

3. VLX-VR achieves state-of-the-art performance on MIN-ERVA, reaching 78.79% accuracy among the models included in our comparison. We further evaluate its crossduration stability and the evidence-grounded agreement between its reasoning traces and the MINERVA reference reasoning traces.

## 2. Related Work

## 2.1. General Video Understanding Models

Vision-language models (VLMs) extend language models with visual perception and cross-modal alignment, providing the foundation for multimodal video understanding [15]. Video-LLaVA learns a shared representation space for images and videos [8]. Qwen2.5-VL advances visual understanding, temporal localization, and long-video processing [1], while VideoLLaMA 3 develops a visioncentric training strategy for general image and video understanding [26]. These models support tasks such as video description, dialogue, event interpretation, temporal analysis, and VideoQA. Within this line of research, OmChat [30] focuses on native multimodal modeling with strong long-context and video-understanding capabilities. It combines dynamic visual encoding with progressive multimodal pretraining to process images, multiple images, and long videos in a unified model. VLX-Flow extends modellevel video understanding from offline clips to continuous video streams. It incrementally maintains visual context and semantic memory, allowing previously observed content to be reused without reprocessing the complete video history [12].

These studies strengthen the perceptual, alignment, temporal-modeling, and context-maintenance capabilities required for video understanding. VLX-VR builds on these foundations but focuses on a broader control problem: learning how to acquire task-relevant evidence, retain useful intermediate information, and determine whether the available evidence is sufficient to complete the current videoanalysis task.

## 2.2. From Perception to Visual Reasoning

Beyond perception and cross-modal alignment, recent research has increasingly focused on enabling models to perform explicit and multi-step reasoning [22, 27]. Chainof-thought prompting demonstrates that generating intermediate reasoning steps can improve performance on complex language tasks [20, 23, 28]. Reasoning models such as OpenAI o1 and DeepSeek-R1 further show that reinforcement learning can train models to refine intermediate reasoning strategies before producing a final response [3, 13].

This direction has subsequently been extended to visual reasoning. VLM-R1 applies rule-based reinforcement learning to general vision-language tasks and studies the stability and generalization of R1-style training for VLMs [14]. Video-R1 introduces temporal-aware reinforcement learning for video reasoning and combines image and video reasoning data during training [5]. Together, these studies mark a transition from recognizing visual content to deriving conclusions through structured reasoning over visual and temporal evidence.

VLX-VR follows this transition but extends the learning target beyond generating an intermediate reasoning trace. Its reasoning process also controls multimodal-memory reads and writes, incorporates the Observations returned by these operations, and determines whether additional evidence is required before producing the task output. Reasoning therefore serves not only to explain a result, but also to control how the model observes and analyzes the video.

## 2.3. Multimodal Agents for Visual Analysis

A complementary line of research combines multimoda models with memory, tools, and iterative interaction [10, 19]. ReAct establishes the reasoning–action–observation paradigm, in which intermediate results guide subsequent reasoning and actions [24]. For complex video understanding, OmAgent introduces a multimodal agent framework that stores and retrieves relevant video frames and uses a task divide-and-conquer loop to dynamically invoke tools and process complex video-analysis requests [29].

Related video-agent systems explore different mechanisms for adaptive evidence acquisition [9, 25]. VideoAgent by Wang et al. iteratively identifies and collects taskrelevant visual information from long-form videos using visual tools [18]. VideoAgent by Fan et al. constructs structured event and object memory and uses temporal localization and memory-query tools [4]. VideoAgent2 further employs uncertainty-aware reasoning to regulate information gathering [31]. These studies demonstrate that iterative observation, external memory, and tool use can reduce the limitations of analyzing a fixed and preselected video context.

Building on these agent-based approaches, VLX-VR is trained as an agentic-aware VLM rather than relying solely on external inference-time control. Within the framework-defined Think–Memory–Observation loop, VLX-VR conditions its decisions on the task instruction, current reasoning state, accumulated multimodal memory, and returned Observations. Memory directly provides the read memory and write memory operations; VLX-VR selects one of them, integrates the resulting evidence or update status, and decides whether to continue the loop or produce the task output. The central objective is to make interaction with multimodal memory part of the model’s learned reasoning policy.

## 3. The Proposed Method

## 3.1. Framework Overview

Given a video and a task instruction, the agentic-aware VLX-VR model operates within the framework-defined Think–Memory–Observation loop. VLX-VR first performs Think to determine what evidence is needed. It then enters Memory and directly invokes the provided read memory or write memory operation to retrieve relevant multimodal evidence or retain intermediate state. The result is returned as an Observation, which VLX-VR uses to update its reasoning state. VLX-VR then decides whether to gather additional evidence or produce a final result. As illustrated in Figure 1, evidence acquisition is conditioned on the task objective, accumulated memory, and returned Observations rather than fixed before reasoning begins.

## 3.2. Think–Memory–Observation loop

At reasoning step t, VLX-VR maintains a state s<sub>t</sub> that includes the task instruction, observed evidence, multimodal content stored in memory, current output hypotheses, and unresolved uncertainty. VLX-VR generates Think from s<sub>t</sub> and decides whether to invoke read memory or write memory during Memory, or terminate. This statedependent decision allows the evidence request to change as new Observations become available.

The reasoning loop consists of three stages:

• Think: VLX-VR interprets the task objective, proposes the next evidence need, and estimates whether the current evidence is sufficient.

• Memory: Memory directly provides read memory and write memory; VLX-VR invokes one operation to retrieve or retain multimodal evidence.

• Observation: The result of the selected Memory operation is returned to VLX-VR for the next decision.

When evidence is sufficient, VLX-VR generates the final output with supporting evidence. When evidence is insufficient or conflicting, the model returns to Think and starts another Memory step. The minimal operation space avoids dependence on a large collection of specialized tools and makes training and evaluation easier to standardize.

The multimodal memory serves as an external and inspectable reasoning state rather than only a cache of textual summaries. Video, audio, supporting evidence, Observations, and intermediate states can be read, written, recorded, and reused across reasoning steps. This design also makes it possible to analyze whether the final output is supported by the evidence actually observed during inference.

Evidence and reasoning traces play different but related roles in VLX-VR. Evidence refers to task-relevant multimodal content acquired from the video or returned by Memory, including visual regions, audio segments, textual cues, timestamps, and intermediate state information. A reasoning trace is the ordered record of how VLX-VR interprets the task, selects Memory operations, incorporates returned Observations, connects evidence across steps, and reaches the final output. Thus, evidence provides the basis for the reasoning trace, while the reasoning trace determines which evidence to acquire, how to organize and interpret it, and whether the available evidence is sufficient for the task. Agreement between a reasoning trace and its supporting evidence indicates that the stated reasoning is grounded in the observed information; it does not by itself establish that the trace faithfully reveals the model’s internal decision process.

The Memory stage provides two operations directly to VLX-VR. read memory retrieves video, audio, and other multimodal content relevant to the current task objective. write memory stores the current Observation, evidence, and state information in multimodal memory. Each operation returns an Observation containing either the requested evidence or the status of the memory update.

The inference process can be written as:

s\_0 = initialize(video, instruction)   
while not stop(s\_t):   
think\_t, call\_t = VLX-VR(s\_t)   
obs\_t = execute(call\_t)   
s\_{t+1} = update(s\_t, obs\_t)   
output = VLX-VR(s\_t)

Here, call t is either read memory or

![](images/33443f4d88cc2d91090353d2348cbece06790ab1e1fe3c30dce4c6abc07a1857.jpg)  
Figure 1. Overview of the framework-defined Think–Memory–Observation loop used to train and run VLX-VR, an agentic-aware video reasoning model. Memory directly provides read memory and write memory; VLX-VR invokes one of these operations, and the resulting Observation guides the next Think step or the final output.

write memory, selected directly by VLX-VR from the current reasoning state, and obs t is the resulting Observation. The stopping condition depends on evidence sufficiency, output confidence, loop budget, and failure state. Premature termination with insufficient evidence should receive a lower reward. Repeatedly reading the same content or extending the loop without a useful objective should be penalized through cost and redundancy terms.

## 3.3. Training

Reinforcement learning has been used to incentivize language-model reasoning in DeepSeek-R1 [3]. For visual tasks, VLM-R1 investigates rule-based reinforcement learning for visual grounding and open-vocabulary object detection, reporting improved generalization over supervised fine-tuning in the evaluated settings [14]. Video-R1 studies video reasoning through temporal-aware T-GRPO optimization and mixed image–video training [5]. Search-R1 provides a complementary text-domain example of learning interleaved reasoning and search-tool interactions with reinforcement learning [7]. These works motivate reinforcement learning for visual tasks and tool interaction, without implying that they implement our Think–Memory– Observation loop.

We train VLX-VR as an agentic-aware video reasoning model within this framework rather than only connecting a general-purpose model to an externally defined loop at inference time. Multimodal data, including videos and agent trajectories, are used for reinforcement learning. The training objective considers task-output correctness, evidence validity, memory read/write behavior, agreement between the model’s reasoning trace and the reference reasoning trace, termination quality, and interaction cost. Through this training, VLX-VR learns what evidence to acquire, what information to retain, how to use returned Observations, and when the available evidence is sufficient to complete the

task.

A possible reward decomposition is:

$$
\begin{array} { r } { R = R _ { \mathrm { t a s k } } + \lambda _ { e } R _ { \mathrm { e v i d e n c e } } + \lambda _ { m } R _ { \mathrm { m e m o r y } } } \\ { + \lambda _ { c } R _ { \mathrm { c o n s i s t e n c y } } - \lambda _ { \mathrm { c o s t } } R _ { \mathrm { c o s t } } . \qquad } \end{array}\tag{1}
$$

Here, $R _ { \mathrm { t a s k } }$ measures task-output quality (final-answer correctness for VideoQA), $R _ { \mathrm { e v i d e n c e } }$ measures whether the acquired evidence covers the task-relevant events, states, and cues identified by the reference reasoning trace, $R _ { \mathrm { { m e m o r y } } }$ measures whether VLX-VR selects and executes useful read and write operations, $R _ { \mathrm { c o n s i s t e n c y } }$ measures agreement between the model’s reasoning trace and the reference reasoning trace, and $R _ { \mathrm { c o s t } }$ penalizes unnecessary loops, tool calls, and token usage. The evidence and consistency rewards therefore measure complementary properties rather than the same behavior.

## 4. Dataset

We select MINERVA as the primary evaluation dataset because it combines broad temporal coverage with explicit reasoning supervision. It includes both short and long videos, and provides detailed, human-annotated reasoning traces in addition to final answer labels. This makes it suitable for evaluating not only whether VLX-VR selects the correct answer, but also whether its reasoning trace is grounded in the relevant evidence described by the annotation [11]. Each question contains a video, a question, five answer choices, and a human-annotated reasoning trace. In our analysis, the events, states, temporal locations, and cues described by this trace define the reference evidence against which evidence acquisition and reasoning consistency are interpreted. The questions require at least two reasoning skills and cover sports, tutorials, daily activities, travel, and other domains. Video-MME also evaluates video understanding across domains, durations, and modality conditions [6], while LongVideoBench targets long-context interleaved video-language understanding and referring reasoning [21]. Compared with these complementary benchmarks, MINERVA is particularly suitable for our analysis of answer accuracy and evidence-grounded reasoning.

The reference reasoning traces contain approximately 92 words on average. About 99.6% contain timestamps, with approximately four timestamps per trace [11]. These annotations support temporal localization, reference-evidence identification, and reasoning-consistency analysis. The human accuracy reported by MINERVA is approximately 92.5% [11]; we report it as human reference performance rather than as a theoretical upper bound or a result obtained under exactly the same system conditions.

We follow the original MINERVA task protocol [11]. The model receives a video, a question, and five candidate answers, and returns a final choice together with a VLX-VR reasoning trace for analysis. The main metric is multiplechoice accuracy. VLX-VR results are taken from the current evaluation report. Seed2.1 Pro, Seed2.1 Turbo, Gemini 3.5 Flash, and Gemini 3.1 Pro comparison numbers are linked to the official Seed2.1 page [2], while the remaining comparison numbers come from the public MINERVA evaluation [11].

## 5. Experimental Results

This evaluation uses the agentic-aware VLX-VR model obtained from the training procedure described above. VLX-VR operates within the framework-defined Think– Memory–Observation loop to acquire relevant multimodal evidence before answering questions from the MINERVA dataset. During Memory, VLX-VR directly invokes the provided read memory or write memory operation.

## 5.1. Overall Accuracy

The primary metric is multiple-choice accuracy:

$$
\mathrm { A c c u r a c y } = \frac { N _ { \mathrm { c o r r e c t } } } { N _ { \mathrm { q u e s t i o n s } } } .\tag{2}
$$

VLX-VR obtains 78.79% overall accuracy on MINERVA. The complete comparison with other models is reported in Table 1.

VLX-VR is higher than the public comparison results in the current aggregate table, but remains below the human reference level. This suggests strong performance on MINERVA’s multi-step, multi-skill video reasoning task, but does not establish universal superiority across video understanding tasks.

## 5.2. Performance across Video Durations

Table 2 evaluates long-video performance using MIN-ERVA’s three duration groups and public baseline results [11].

Table 1. Comparison on MINERVA. Model rows are ordered by accuracy; human and random reference levels are listed separately. Seed2.1 and Gemini 3.x results are from the Seed2.1 page [2]; other external results are from MINERVA [11].
<table><tr><td>Model</td><td>Accuracy (%)</td><td>∆ vs. VLX-VR (pp)</td><td>Source</td></tr><tr><td>VLX-VR</td><td>78.79</td><td></td><td>This work</td></tr><tr><td>Seed2.1 Pro</td><td>70.70</td><td>-8.09</td><td>[2]</td></tr><tr><td>Gemini 3.5 Flash</td><td>68.60</td><td>-10.19</td><td>[2]</td></tr><tr><td>Gemini 2.5 Pro Thinking</td><td>66.20</td><td>-12.59</td><td>[11]</td></tr><tr><td>Seed2.1 Turbo</td><td>65.90</td><td>-12.89</td><td>[2]</td></tr><tr><td>Gemini 3.1 Pro</td><td>63.50</td><td>-15.29</td><td>[2]</td></tr><tr><td>Gemini 2.5 Flash Thinking</td><td>57.30</td><td>-21.49</td><td>[11]</td></tr><tr><td>GPT-4.1</td><td>53.99</td><td>-24.80</td><td>[11]</td></tr><tr><td>GPT-40</td><td>45.54</td><td>-33.25</td><td>[11]</td></tr><tr><td>OpenAI ol</td><td>43.48</td><td>-35.31</td><td>[11]</td></tr><tr><td>VideoLLaMA3</td><td>35.91</td><td>-42.88</td><td>[11]</td></tr><tr><td>InternVideo2.5</td><td>35.18</td><td>-43.61</td><td>[11]</td></tr><tr><td>Qwen2.5-VL</td><td>35.05</td><td>-43.74</td><td>[11]</td></tr><tr><td>Claude 3.5 Sonnet v2</td><td>31.28</td><td>-47.51</td><td>[11]</td></tr><tr><td>Human</td><td>92.54</td><td>+13.75</td><td>[11]</td></tr><tr><td>Random</td><td>20.00</td><td>-58.79</td><td>[11]</td></tr></table>

Table 2. Accuracy across three video-duration groups.
<table><tr><td>Model</td><td>Below 5 min</td><td>5–15 min</td><td>Above 15 min</td></tr><tr><td>VLX-VR</td><td>76.70</td><td>78.73</td><td>80.92</td></tr><tr><td>Gemini 2.5 Pro Thinking</td><td>68.87</td><td>66.84</td><td>57.97</td></tr><tr><td>GPT-4.1</td><td>58.84</td><td>54.79</td><td>47.25</td></tr><tr><td>OpenAI o1</td><td>48.28</td><td>41.45</td><td>40.38</td></tr><tr><td>Claude 3.5 Sonnet v2</td><td>40.90</td><td>33.68</td><td>28.30</td></tr></table>

![](images/a668b897fdae06a10d88700a0ec72c3332a880e82c392d1d007ac51bf9b4b6a0.jpg)  
Figure 2. Accuracy comparison across video-duration groups. VLX-VR obtains 76.70%, 78.73%, and 80.92% in the three groups. The current result does not show a monotonic decrease with duration.

Figure 2 visualizes the three-group comparison. VLX-VR increases from 76.70% below 5 minutes to 78.73% at 5– 15 minutes and 80.92% above 15 minutes, whereas all four public baselines decline as video duration increases. We quantify cross-duration stability with the equally weighted population variance of accuracy across duration groups:

$$
\mathrm { C D A V } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( \mathrm { A c c } _ { i } - \overline { { \mathrm { A c c } } } ) ^ { 2 } .\tag{3}
$$

Here, $\operatorname { A c c } _ { i }$ is measured in percentage points, so CDAV is reported in squared percentage points $( \mathrm { p p } ^ { 2 } )$ . Table 3 applies this definition to the three groups in Table 2, with equal weighting across duration groups and statistics computed from the displayed rounded accuracies.

Table 3. Cross-duration performance statistics under MINERVA’s original duration grouping. Lower CDAV indicates less variation across duration groups, but does not by itself imply higher accuracy.
<table><tr><td>Model</td><td>Mean accuracy (%)</td><td>CDAV  $( \mathrm { p p } ^ { 2 } )$ </td><td>Std. (pp)</td><td>Range (pp)</td></tr><tr><td>VLX-VR</td><td>78.78</td><td>2.97</td><td>1.72</td><td>4.22</td></tr><tr><td>Gemini 2.5 Pro Thinking</td><td>64.56</td><td>22.40</td><td>4.73</td><td>10.90</td></tr><tr><td>GPT-4.1</td><td>53.63</td><td>23.06</td><td>4.80</td><td>11.59</td></tr><tr><td>OpenAI o1</td><td>43.37</td><td>12.24</td><td>3.50</td><td>7.90</td></tr><tr><td>Claude 3.5 Sonnet v2</td><td>34.29</td><td>26.65</td><td>5.16</td><td>12.60</td></tr></table>

Under MINERVA’s original duration grouping, VLX-VR has a mean accuracy of 78.78%, a CDAV of 2.97 $\mathrm { p p } ^ { 2 }$ a standard deviation of 1.72 pp, and a range of 4.22 pp. It has the lowest CDAV among the five models while also achieving the highest mean accuracy. The other four models have CDAV values of 22.40, 23.06, 12.24, and $2 6 . 6 5 ~ \mathrm { p p ^ { 2 } }$ respectively. Thus, under the original MINERVA grouping, VLX-VR maintains stable performance as duration increases; however, the aggregated Above 15 min group does not reveal how performance varies within longer videos.

## 5.3. Performance Analysis on Different Durations

The analysis under MINERVA’s original grouping shows the unusual pattern that VLX-VR performs better as video duration increases, reaching 80.92% in the aggregated Above 15 min group. To determine whether this improvement is uniform across longer videos, we divide that group into 15–30 min and Above 30 min intervals while retaining the original Below 5 min and 5–15 min groups. Table 4 reports the resulting refined analysis.

Table 4. Refined duration analysis with videos longer than 15 minutes split at 30 minutes.
<table><tr><td>Video duration</td><td>Accuracy (%)</td></tr><tr><td>0-5 min</td><td>76.70</td></tr><tr><td>5–15 min</td><td>78.73</td></tr><tr><td>15–30 min</td><td>83.40</td></tr><tr><td>Above 30 min</td><td>74.75</td></tr></table>

The highest accuracy occurs in the 15–30 min interval at 83.40%, whereas accuracy decreases to 74.75% above 30 minutes. The apparent improvement in the original Above 15 min result is therefore driven primarily by the strong 15– 30 min interval rather than by a uniform increase across all longer videos. We next compare the cross-duration statistics under the original and refined groupings in Table 5.

Table 5. Cross-duration performance statistics for VLX-VR under the original MINERVA and refined duration groupings.
<table><tr><td>Grouping</td><td>Groups</td><td>Mean accuracy (%)</td><td>CDAV  $( \mathrm { p p } ^ { 2 } )$ </td><td>Std. (pp)</td><td>Range (pp)</td></tr><tr><td>Original MINERVA</td><td>3</td><td>78.78</td><td>2.97</td><td>1.72</td><td>4.22</td></tr><tr><td>Refined</td><td>4</td><td>78.40</td><td>10.33</td><td>3.21</td><td>8.65</td></tr></table>

Both analyses assign equal weight to each duration group, and the statistics are computed from the rounded accuracies reported in Tables 2 and 4. Splitting the longvideo group changes the equal-weight mean only slightly, from 78.78% to 78.40%, but increases CDAV from 2.97 to $1 0 . 3 3 ~ \mathrm { p p ^ { 2 } }$ , standard deviation from 1.72 to 3.21 pp, and range from 4.22 to 8.65 pp. The higher variation under the refined grouping shows that the original MINERVA grouping smooths over the 15–30 min peak and the subsequent decline above 30 minutes. The strong 15–30 min result suggests that the learned evidence-acquisition behavior of the agentic-aware VLX-VR model broadens the duration range over which it can effectively analyze video. Although performance declines beyond 30 minutes, the refined-group accuracies remain within 74.75–83.40%, indicating broadly stable performance rather than a collapse on longer videos.

## 5.4. Performance across Reasoning Skills

Table 6 reports the accuracy distribution across reasoning skills.

Table 6. VLX-VR accuracy by reasoning skill.
<table><tr><td>Skill</td><td>Accuracy (%)</td></tr><tr><td>Counting</td><td>66.67</td></tr><tr><td>Object recognition</td><td>82.01</td></tr><tr><td>Event occurrence</td><td>82.81</td></tr><tr><td>Listening</td><td>75.19</td></tr><tr><td>Temporal reasoning</td><td>85.95</td></tr><tr><td>Reading</td><td>90.76</td></tr><tr><td>Numerical reasoning</td><td>85.71</td></tr><tr><td>Spatial perception</td><td>73.47</td></tr><tr><td>Cause and effect</td><td>72.73</td></tr><tr><td>Counterfactual reasoning</td><td>74.19</td></tr><tr><td>Situational awareness</td><td>90.32</td></tr><tr><td>Goal reasoning</td><td>76.92</td></tr><tr><td>State changes</td><td>69.23</td></tr></table>

VLX-VR is strongest on reading, situational awareness, temporal reasoning, and numerical reasoning. Event occurrence and object recognition are also above the overall accuracy. Counting, state changes, cause and effect, and spatial perception are below the overall level, suggesting the need for more reliable object tracking, state modeling, local evidence retrieval, and causal judgment. The skill results describe the capability distribution and do not by themselves prove that a particular module caused a given difference.

## 5.5. Analysis of Reference-trace Agreement

On correctly answered samples, 96.20% of the VLX-VR reasoning traces are judged consistent with the MINERVA reference reasoning traces and the evidence described by them. Across the full evaluation set, the proportion of samples estimated to have both a correct answer and a VLX-VR reasoning trace consistent with the reference reasoning trace is 78.79% × 96.20% ≈ 75.80%. This joint criterion is stricter than answer accuracy alone. Nevertheless, the estimated joint rate is approximately 5.10 percentage points higher than the published answer-only accuracy of Doubao Seed2.1 Pro (70.70%) [2].

This comparison is descriptive because the two results use different success criteria: VLX-VR is evaluated by the joint requirement of answer correctness and agreement between its reasoning trace, the MINERVA reference reasoning trace, and the evidence described by that trace, whereas Seed2.1 Pro reports answer accuracy only. The reported agreement is therefore an evaluation of evidence-grounded consistency with the annotation, not a direct measurement of the raw evidence itself or a guarantee that the trace faithfully reveals the model’s internal decision process [17].

## 6. Case Studies

To connect the aggregate results above with concrete model behavior, this section presents detailed positive and negative cases covering temporal reasoning, numerical reasoning, situational awareness, counting, and state changes. The videos, questions, answer options, reference answers, and reference reasoning traces are from MINERVA [11]; the VLX-VR answers, reasoning traces, and case analyses are from our evaluation. The cases distinguish the evidence observed by VLX-VR from the reasoning trace that organizes and interprets that evidence, making the reference answer, the VLX-VR answer, and the resulting strengths or limitations explicit.

## 6.1. Correct case 1: Temporal reasoning

Video: resource-169422, approximately 257 seconds.

Question: Who is the second person to speak in the video?

Options: A narrator; The brunette woman; It is a silent film; The blonde woman; The brunette woman’s fiancee.

Reference answer: The blonde woman.

Reference reasoning trace: I watch the video until I see the brunette woman speak at 00:41. When I see the blonde woman speak at 00:47, then, I know that she is the second woman to speak in the video.

VLX-VR answer: The blonde woman; correct.

VLX-VR reasoning trace: VLX-VR first reviewed the full video and confirmed that only two women appear. It then divided the 40–48 second interval into multiple observation windows, identifying the brunette woman as the first speaker around 41 seconds and the blonde woman as the responder around 45.6–48 seconds based on lip motion and shot reversals. It further used later alternating dialogue to cross-check the speaking order.

Analysis: The question requires event ordering rather than simple person identification. The VLX-VR reasoning trace contains a chain of first-speaking timestamp, speaker identity, and subsequent alternating dialogue, connecting local visual observations with temporal order. Within the framework-defined Think–Memory–Observation loop, VLX-VR uses the first Observation to locate the first utterance, the second to verify the second speaker, and later observations to test whether the speaking sequence remains coherent. Figure 3 summarizes the full-video scan, the decisive keyframes, and the supporting evidence described in the VLX-VR reasoning trace.

## 6.2. Correct case 2: Numerical reasoning

Video: resource-169908, approximately 5,686 seconds.

Question: When orange first takes control of the four territories of Australia, what is the average value per territory that green controls?

Options: 2.9; 2.8; 3.2; 2.7; 2.3.

Reference answer: 2.7.

Reference reasoning trace: I identified Australia on the map as the continent in the bottom right section of the frame. At 12:11, I saw the leftmost territory of mainland Australia attack the chain of islands west of New Zealand. I saw that the attack was successful, as the chain of islands was now colored orange and had orange “10” on it. At this point, this territory, New Zealand, and the left and right mainland territories of Australia, all glowed orange. I understood this to mean that the orange player now controlled the entire continent. I observed the rest of the map and looked for green territories. I counted 10 territories colored green, each with their own cloud-shaped green icon on them that had a number inside. I understood from my knowledge of Risk that this corresponds to the number of troops in that territory. From left to right, top to bottom, I recorded the number of troops present in each of the territories: 1, 1, 4, 4, 3, 3, 6, 3, 1, 1. I added these together to get the total value: 1+1+4+4+3+3+6+3+1+1=27. To find the average value per territory, I divided 27/10=2.7.

![](images/dd333dee2679f4afb6eaa32e48f2e6f36085ae83f114cab051892046283f7efb.jpg)  
Figure 3. Visualization of the temporal-reasoning case for resource-169422. The full-video overview supports coarse temporal localization, while the keyframes at 00:41 and 00:47 identify the first and second speakers. The lower panels align the MINERVA reference reasoning trace with the VLX-VR reasoning trace.

VLX-VR answer: 2.7; correct.

VLX-VR reasoning trace: VLX-VR first used a coarse full-video scan to locate the interval around 695–775 seconds. It then used fine-grained observation to confirm that western Australia changed from black to orange around 712–713 seconds. It read the green player’s panel, verified 10 green territories and a total value of 27, and computed 27 / 10 = 2.7.

Analysis: This case shows a long-video chain of event trigger, state reading, and numerical computation. Rather than guessing from a local frame, the model first localized the state change in which orange completed control of Australia and then read the global panel for green. During Memory, VLX-VR can use write memory to retain the time of the territory change and the “green 27 / 10” state, while read memory retrieves this evidence for the final arithmetic step. The retrieved evidence is returned as an Observation.

## 6.3. Correct case 3: Situational awareness

Video: resource-169911, approximately 338 seconds.

Question: Before the man kisses the other man, what does he do before this?

Options: He looks down; He walks away; He gestures around; He looks up; He looks around.

Reference answer: He looks around.

Reference reasoning trace: I watched the video to find the moment when the man kisses the other man. This occurs from 03:06–03:10. The most recent scene before this which features both men runs from 02:53–03:01. During this scene, the man looks down and around the two of them, up and down the stairwell where they are standing. Thus, he looks around before he kisses the other man.

VLX-VR answer: He looks around; correct.

VLX-VR reasoning trace: VLX-VR localized a continuous stairwell segment and recorded the actions from approximately 173–181 seconds. The man first looked at the other person, then turned his gaze downward, upward, and toward the upper-left area along the railing before turning back and approaching the other man. VLX-VR ruled out “walks away” and the isolated “looks down/looks up” options and selected “looks around.”

Analysis: The decisive evidence is not a single action but a sequence of gaze directions forming a situation. The model interpreted looking down and looking up as instances of the higher-level action “looking around,” while using the surrounding interaction to disambiguate the options. This suggests that situational-awareness questions can benefit from repeated Observations that aggregate local actions before Think performs semantic abstraction.

## 6.4. Incorrect case 1: Counting under occlusion and uncertainty

Video: resource-169417, approximately 221 seconds.

Question: How many women appear in the first row of the audience?

Options: 6; 4; 5; 3; 2.

Reference answer: 5.

Reference reasoning trace: I watched the video to identify the times that the camera focuses on the audience. This happens at 01:04, 01:08, 02:18, and 03:39. Each instance features the same shot of the audience. From there, I counted how many women appeared sitting in the first row, which is 5.

VLX-VR answer: 4; incorrect.

VLX-VR reasoning trace: VLX-VR reviewed multiple audience windows and concluded that the first row contained seven seats: four people whose gender it could confirm as female, two males, and one person whose torso and legs were visible but whose face was cropped out. It adopted a conservative policy, excluded the uncertain person, and returned 4.

Error cause: The model did not fail because it lacked all relevant evidence. Instead, it applied a “count only explicitly confirmed instances” policy under occlusion and cropping. This disagreed with the benchmark annotation, which counted five women, producing an undercount of one. The failure lies in the definition of the target set and the treatment of uncertain instances, rather than in arithmetic.

Model limitation: Large models remain sensitive to occlusion, cropping, row assignment, and identity uncertainty in multi-object counting. The model can identify local attributes, but it may fail to merge observations across shots into the same object set as the benchmark. When visual evidence is incomplete, both conservative exclusion and unsupported inference can cause errors. VLX-VR may request additional observations to reduce uncertainty, but the framework-defined loop in which the agentic-aware VLX-VR model operates cannot guarantee recovery of an attribute that is absent from the available visual evidence.

## 6.5. Incorrect case 2: State changes and temporalchain tracking

Video: resource-169421, approximately 527 seconds.

Question: What change of color occurs to the letters added to the word “ZING” by Chris Cree?

Options: They turn from red to yellow; They turn from yellow to red; They turn from green to red; They turn from yellow to green; They turn from red to green.

Reference answer: They turn from yellow to green.

Reference reasoning trace: I watched for the section of video dedicated to Chris Cree and his addition to the word “ZING”, which I found from 03:06–03:38. Since the letters “ANNUALI” are added to the “ZING,” “ANNUALI” are then the letters whose color I pay attention to. Initially, the letters “ANNUALI” are displayed in yellow, but this color changes to green at 03:05 after they are played on the board. Therefore, the letters added to the word “ZING” by Chris Cree change color by turning from yellow to green.

VLX-VR answer: They turn from yellow to red; incorrect.

VLX-VR reasoning trace: VLX-VR localized the Chris Cree segment around 03:06–03:38 and identified the added letters as “ANNUALI.” It observed that the letters turned green after being placed on the board around 185 seconds, but later observed them turn red around 215–218 seconds. It treated the later transition as the answer and described the change as “green to red.” The final option marker was also inconsistent with the textual conclusion: it corresponded to “yellow to red,” while the explanation described “green to red.”

Error cause: The question asks for the transition from the initial state to the state immediately after the letters are placed. The model continued tracking a later “unacceptable” state and allowed this salient event to overwrite the earlier transition required by the question. It detected multiple real changes but failed to select the correct pair of states under the question’s temporal boundary. The inconsistency between the explanation and the final option further indicates state drift between reasoning and answer decoding.

Model limitation: Large models can be attracted to later salient events in state-change questions and may fail to maintain a complete temporal chain involving object identity, initial state, target state, and event boundary. Even when the model recognizes the letters and multiple color states, it can answer incorrectly if it does not lock onto the first valid transition required by the question. During Memory, VLX-VR should use write memory to store object identity and time-ordered state records explicitly, while read memory should return structured “before–trigger– after” evidence as an Observation. Merely increasing the number of observations does not automatically solve eventboundary selection.

## 6.6. Case-study summary: separate observation from evidence decisions

The three correct cases show that VLX-VR can form coherent evidence chains for temporal order, numerical computation, and situational action understanding. The two incorrect cases show that VLX-VR may observe relevant frames yet fail in uncertain-object counting, selecting among multiple state transitions, or mapping its reasoning to the final option. Case analysis should therefore evaluate more than whether VLX-VR “saw” the relevant content. It should separately measure whether VLX-VR found the correct time interval, maintained the correct object identity, selected the state transition requested by the question, and kept the VLX-VR answer consistent with its reasoning trace. These are the behaviors that process rewards and structured agent trajectories should reinforce and validate in VLX-VR.

## 7. Conclusion

We presented VLX-VR, an agentic-aware video reasoning model trained to operate within a video reasoning framework organized as a Think–Memory–Observation loop. Within this framework-defined loop, VLX-VR identifies what evidence is needed, selects and executes the read memory or write memory operation provided by Memory, integrates the returned Observation, and decides whether to continue gathering evidence or produce the task output. Multimodal memory serves as an external and inspectable reasoning state for video, audio, supporting evidence, Observations, and intermediate states. Consequently, evidence acquisition and memory use become model behaviors that can be recorded, analyzed, and trained rather than remaining implicit components of an external inference pipeline.

On MINERVA, VLX-VR achieves state-of-the-art performance among the models included in our comparison, with 78.79% overall accuracy. Across MINERVA’s three duration groups, it obtains accuracies of 76.70%, 78.73%, and 80.92%, together with the lowest CDAV under the original MINERVA grouping among the five models analyzed. The finer-grained duration analysis shows that the apparent improvement for longer videos is driven primarily by the 83.40% result in the 15–30 min interval; accuracy decreases to 74.75% above 30 minutes rather than increasing monotonically with duration. On correctly answered samples, 96.20% of VLX-VR’s reasoning traces are consistent with the MINERVA reference reasoning traces and the evidence described by them, and approximately 75.80% of all evaluated samples satisfy both answer correctness and this evidence-grounded trace criterion.

These results support the effectiveness of VLX-VR across varied video durations and reasoning skills, while also defining clear limitations. Performance remains weaker for counting, state changes, cause and effect, and spatial perception, and evidence-grounded agreement with the reference reasoning trace indicates consistency with annotated reasoning and observed evidence rather than guaranteed faithfulness to the model’s internal decision process. Separating the causal contributions of the framework-defined loop, direct multimodal-memory access, and reinforcement-learning training will require matched ablations and remains an important direction for future evaluation.

## References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. 2

[2] ByteDance Seed. Seed2.1: Official model evaluation results. https://seed.bytedance.com/en/seed2\_ 1, 2026. 5, 7

[3] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 4

[4] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. VideoAgent: A memory-augmented multimodal agent for video understanding. arXiv preprint arXiv:2403.11481, 2024. 1, 3

[5] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-R1: Reinforcing video reasoning in MLLMs. arXiv preprint arXiv:2503.21776, 2025. 2, 4

[6] Chaoyou Fu, Yuhan Dai, Yongdong Luo, et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. arXiv preprint arXiv:2405.21075, 2024. 4

[7] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025. 4

[8] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 2

[9] Ye Liu, Kevin Qinghong Lin, Chang-Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for temporal-grounded video reasoning. In International Con ference on Learning Representations, pages 57481–57506, 2026. 3

[10] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai. Drvideo: Document retrieval based long video understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18936–18946. IEEE, 2025. 2

[11] Arsha Nagrani, Sachit Menon, Ahmet Iscen, Shyamal Buch, Ramin Mehran, Nilpa Jha, Anja Hauth, Yukun Zhu, Carl Vondrick, Mikhail Sirotenko, Cordelia Schmid, and Tobias Weyand. Minerva: Evaluating complex video reasoning. arXiv preprint arXiv:2505.00681, 2025. 1, 4, 5, 7

[12] Om AI Lab. VLX-Flow: Continuous video understanding for real-time multimodal interaction. https://omai-lab.github.io/2026\_06\_26\_vlx\_flow\_en. html, 2026. Official technical blog. 2

[13] OpenAI. OpenAI o1 System Card. https://openai. com/index/openai-o1-system-card/, 2024. 2

[14] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. VLM-R1: A stable and generalizable R1-style large vision

language model. arXiv preprint arXiv:2504.07615, 2025. 2, 4

[15] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18221–18232. IEEE, 2024. 2

[16] Qwen Team et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 6(7):13, 2025. 1

[17] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. arXiv preprint arXiv:2305.04388, 2023. 7

[18] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. VideoAgent: Long-form video understanding with large language model as agent. arXiv preprint arXiv:2403.10517, 2024. 1, 3

[19] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer, 2024. 2

[20] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, pages 24824–24837, 2022. 2

[21] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024. 5

[22] Yuan Xie, Tianshui Chen, Zheng Ge, and Lionel Ni. Videomtr: Reinforced multi-turn reasoning for long video understanding. arXiv preprint arXiv:2508.20478, 2025. 2

[23] Jie Yang, Feipeng Ma, Zitian Wang, Dacheng Yin, Kang Rong, Fengyun Rao, and Ruimao Zhang. Wethink: Toward general-purpose vision-language reasoning via reinforcement learning. arXiv preprint arXiv:2506.07905, 2025. 2

[24] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022. 2

[25] Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, and Sung Ju Hwang. Worldmm: Dynamic multimodal memory agent for long video reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25599–25609, 2026. 3

[26] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. VideoLLaMA 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 1, 2

[27] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He,

and Yansong Tang. Thinking with videos: Multimodal toolaugmented reinforcement learning for long video reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32903–32914, 2026. 2

[28] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via stepwise group relative policy optimization. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1859–1869. IEEE, 2025. 2

[29] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyu song Lee. OmAgent: A multi-modal agent framework for complex video understanding with task divide-and-conquer. In Proceedings of the 2024 Conference on Empirical Meth ods in Natural Language Processing, pages 10031–10045, 2024. 2

[30] Tiancheng Zhao, Qianqian Zhang, Kyusong Lee, Peng Liu, Lu Zhang, Chunxin Fang, Jiajia Liao, Kelei Jiang, Yibo Ma, and Ruochen Xu. OmChat: A recipe to train multimodal language models with strong long context and video under standing. arXiv preprint arXiv:2407.04923, 2024. 2

[31] Zhuo Zhi, Qiangqiang Wu, Minghe Shen, Wenbo Li, Yinchuan Li, Kun Shao, and Kaiwen Zhou. VideoAgent2: Enhancing the LLM-based agent system for long-form video understanding by uncertainty-aware CoT. arXiv preprint arXiv:2504.04471, 2025. 1, 3