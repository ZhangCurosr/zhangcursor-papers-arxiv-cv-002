# Hallucination-aware Faithful Reasoning in Remote Sensing

Yi Liu<sup>1,2</sup>, Jing Zhang<sup>1,2†</sup>, Xiang Feng<sup>1</sup>, Xiaoyu Tian<sup>3</sup>, Haonan Guo<sup>1,2</sup>, Di Wang<sup>1,2</sup>, Zengmao Wang<sup>1</sup>, Bo Du<sup>1,2†and</sup> <sup>a</sup> <sup>road.</sup> <sup>Within</sup> <sup>this</sup> <sup>grassy</sup> <sup>area,</sup> <sup>near</sup> <sup>the</sup> <sup>center-left</sup> of the frame, a small structure with a triangular roof and <sup>observe</sup> <sup>the</sup> <sup>positions</sup> <sup>of</sup> <sup>the</sup> <sup>vehicles.</sup> <sup>The</sup> <sup>cars</sup> <sup>are</sup> <sup>all</sup> parked within designated parking spaces. ...... Therefore, <sup>court</sup> <sup>with</sup> <sup>visible</sup> <sup>markings</sup> <sup>(lines</sup> <sup>and</sup> <sup>a</sup> <sup>center</sup> <sup>circle)</sup>situated between a long building and a grassy area. ...... <sup>courts.</sup> <sup>Within</sup> <sup>this</sup> <sup>lot,</sup> <sup>among</sup> <sup>several</sup> <sup>other</sup> <sup>vehicles,</sup> <sup>one</sup> <sub>car</sub> <sub>is</sub> <sub>distinctly</sub> <sub>brown</sub> <sub>in</sub> <sub>color.</sub> <sub>It</sub> <sub>is</sub> <sub>positioned</sub> <sub>in</sub> <sub>the</sub>

<sup>1</sup>School of Computer Science, Wuhan University, <sup>2</sup>Zhongguancun Academy,of cars pulling into the parking lot. on this visual evidence, the color is identified as beige <sup>and</sup> <sup>a</sup> <sup>white</sup> <sup>car</sup> <sup>on</sup>

<sup>3</sup>School of Computer Science, Chongqing University

jingzhang.cv@gmail.com; dubo@whu.edu.cn

![](images/9294c4fc5113cdd9ab792662bf2878f7ef5fa251e060a18af16fb3628ccb1911.jpg)  
Figure 1: Examples of typical process-level hallucinations in remote sensing visual question answering. (a) Nonexistent object: the model fabricates a swing-like structure with a triangular roof and a suspended seat in an area where no such object is visible. (b) Target omission: the model fails to recognize vehicles that are entering the parking lot. (c) Attribute misdescription: the model describes the surface color of the basketball court in a manner that is inconsistent with the image evidence. (d) Spatial relation misjudgment: the model incorrectly infers the relative position of the target vehicle.

## Abstract

Multimodal large language models (MLLMs) have demonstrated promising capabilities for remote sensing reasoning, yet they frequently generate visually unsupported reasoning that contradicts the visual evidence. Existing post-training methods optimize primarily for final-answer correctness, providing limited supervision for identifying and correcting hallucinations that arise during the reasoning process. To address this limitation, we propose FaithRS, a hallucination-aware self-reflection framework for faithful reasoning in remote sensing. We first construct SFHD-43K, a dataset of 43K structured, fine-grained hallucination annotations, and train a detector to identify hallucination types, underlying causes, and supporting visual evidence. FaithRS then incorporates this diagnostic feedback into a Safe Detector-Guided Corrective Rollout framework, where a safety-aware gate selectively determines whether and how to revise the response. Training is driven by a composite reward that jointly optimizes answer accuracy, reasoning faithfulness, hallucination reduction, format compliance, and revision safety. Experiments with Qwen3.5 backbones (0.8B, 4B, and 9B) show consistent improvements across four remote sensing benchmarks, with gains of up to 11.64 points. Code and data will be publicly available at https://github.com/MiliLab/FaithRS.

## 1 Introduction

MLLMs (Dai et al., 2023; Liu et al., 2023; Chen et al., 2024; Yang et al., 2025) have expanded the capability of remote sensing analysis beyond conventional recognition and classification toward open-ended image interpretation, visual question answering, object localization, and spatial reasoning (Hu et al., 2023; Kuckreja et al., 2023; Luo et al., 2024). However, large scale variation, ambiguous object boundaries, and dense spatial relations make remote sensing MLLMs particularly prone to visual hallucinations. As shown in Fig. 1(a), the model fabricates a swing-like structure in a region where no such object exists, undermining the reliability of its reasoning.

Among the many factors contributing to hallucinations (Bai et al., 2024; Huang et al., 2025), a critical factor lies in reinforcement learning with verifiable rewards (RLVR), a widely adopted post-training paradigm that typically assesses only the final output holistically. Even when visual evidence is incorporated into the reward, such coarse-grained feedback ofers no indication of the specific error span, the error type (e.g., object recognition, attribute perception, spatial reasoning, or logical deduction), or the appropriate correction strategy.

Addressing this limitation requires structured diagnosis that pinpoints where hallucinations occur, why they conflict with the visual evidence, and how they should be corrected, thereby converting coarse correctness signals into actionable guidance. A natural way to exploit such diagnosis is to let the model revise its own responses accordingly, in the spirit of selfreflection. However, correction itself is a double-edged sword: a response may arrive at the correct answer through unfaithful reasoning, in which case indiscriminately rewriting the entire response risks corrupting an answer that is already correct. Efective hallucination mitigation therefore demands not only fine-grained diagnosis, but also a correction mechanism that acts selectively on the flawed portion of a response.

![](images/9d35802a9675abee4d1a492d749321ac9cf1e74f7e15c9d1db82e399855c4126.jpg)  
Figure 2: Overview of the structured fine-grained hallucination diagnosis pipeline.

Motivated by these observations, we propose FaithRS, a hallucination-aware self-reflection framework for multimodal reasoning in remote sensing, which converts structured hallucination diagnosis into dense supervisory signals for RLVR training. Specifically, we first construct SFHD-43K, a structured fine-grained hallucination diagnosis dataset built from diverse model outputs and consolidated through multi-model judge consensus, on which we train a detector that identifies hallucination categories and fine-grained types together with their causes, visual evidence, and diagnostic rationale. FaithRS then integrates these diagnostics into a detector-guided correction pipeline, where a safety gate determines the correction strategy based on hallucination status and answer correctness: reliable responses are retained, responses with both flawed reasoning and incorrect answers are fully rewritten, and responses with correct answers but unfaithful reasoning are revised while the answer is preserved—thereby realizing the selective correction discussed above and avoiding the answer degradation caused by indiscriminate rewriting. The correction trajectory is optimized with a composite reward covering answer accuracy, visual faithfulness, hallucination reduction, output validity, and correction safety, elevating hallucination diagnosis from a post-hoc evaluation signal to an explicit training constraint.

Experiments at the 0.8B, 4B, and 9B parameter scales on the Qwen3.5 backbone demonstrate consistent average gains across the MME-RealWorld-RS, RSVQA-HR, CHOICE, and LHRS benchmarks, with a maximum average improvement of 11.64 points; FaithRS-9B attains the best overall average performance among all evaluated models. Ablation studies show that detector-guided reinforcement learning and self-reflection contribute complementary gains, and training dynamics reveal that the model progressively improves the hallucination correction rate while maintaining a low answer degradation rate.

The main contributions are summarized as follows:

• We construct SFHD-43K and train a structured hallucination detector that identifies hallucination types, causes, and supporting visual evidence.

• We propose FaithRS, which uses detector-guided corrective rollout and safety-aware gating to learn whether and how to revise hallucinated reasoning.

• Experiments across three model scales and four remote sensing benchmarks show consistent average gains of up to 11.64 points, with ablations confirming the efectiveness of the proposed components.

## 2 Methodology

## 2.1 Structured Fine-grained Hallucination Diagnosis

To provide hallucination diagnostic signals for subsequent trustworthy reasoning post-training, as shown in Fig. 2, we first build a Structured Fine-grained Hallucination Detection Model. Unlike evaluators that return only scalar scores, the output of $D _ { \phi }$ is constrained to output a parseable hallucination diagnosis structure. In this way, the detector can determine not only whether a model output contains hallucinations, but also the underlying source of each hallucination and its inconsistency with the visual evidence.

![](images/19797a3472b10c491ebb20b0e428de0e36bc3ed7e671046c7e9633fb4283a256.jpg)  
Figure 3: Overview of the hallucination-aware self-reflection training framework.

## Stage 1: SFHD-43K Dataset Construction.

To train a detector capable of producing structured diagnostic results, we construct SFHD-43K, a structured fine-grained hallucination diagnosis dataset. The construction process consists of four steps: image-question sampling, candidate reasoning generation, multi-model judge hallucination annotation, and consistency merging. We first sample 5k image-question pairs from GeoLLaVA-8K (Wang et al., 2025b). Each sample contains a remote sensing image, a natural-language question, and an answer, which provide the basic inputs for candidate reasoning generation and hallucination annotation. We then use multiple open-source and closed-source MLLMs to generate responses for the same input, forming a candidate reasoning set. Since diferent models exhibit diferent biases in visual perception, language reasoning, and spatial relation understanding, the resulting candidate set covers diverse output patterns. Compared with manually constructed error samples, this multi-model generation strategy more naturally captures typical failure modes of current MLLMs in remote sensing reasoning.

We then use three strong judge models to annotate hallucinations in the candidate outputs. Each judge receives the image, question, reference answer, model reasoning, and model answer, and produces a structured diagnosis $d = ( h , c , T , \rho , e , j )$ where h denotes whether a hallucination exists, c denotes the hallucination category, T denotes the set of fine-grained hallucination types, ρ describes the error causes, e specifies the relevant visual evidence, and j provides the diagnostic justification. We merge the hallucination indicator h using strict majority voting. If no majority decision can be reached, the sample is assigned an uncertain label and removed from the training set. For the category c and type set T, we retain only the annotations from judges whose hallucination decisions agree with the majority result, and then perform order-preserving deduplication. For the explanatory components ρ, e, and j, we prioritize the output of Qwen3.5-397B-A17B (Qwen Team, 2026) to improve the completeness and consistency of the error analysis and visual evidence description. After removing uncertain and invalid samples, we obtain SFHD-43K, which contains 38,339 samples for supervised fine-tuning (SFT) and 4,917 samples for RL.

## Stage 2: Training the Hallucination Detector.

We use Qwen3.5-2B (Qwen Team, 2026) as the backbone ofthe hallucination detector $D _ { \phi }$ and train it through SFT followed by RL. The SFT stage consists of two phases. In the first phase, the detector is trained with short-form diagnostic targets $d _ { \mathrm { s h o r t } } =$ $( h , c , \tau )$ . This phase enables the detector to learn hallucination classification, coarse-grained category prediction, and finegrained type recognition, while establishing a stable and parseable output format. In the second phase, the supervision is extended to the full diagnosis $d = ( h , c , T , \rho , e , j )$ . This allows the detector to provide structured explanations and evidencegrounded judgments for subsequent corrective reasoning.

After SFT, we further optimize the detector with RL. The detector is trained with rewards for structural validity, hallucination decision accuracy, fine-grained label consistency, and output conciseness. Specifically, the format reward verifies whether the prediction forms a valid structured output with all required components. The hallucination reward measures whether the predicted decision h agrees with the reference diagnosis. The type reward evaluates the consistency of the predicted category c and type set T with their reference annotations. A repetition penalty discourages templated and redundant generations.

## Diagnostic Feedback for Trustworthy Reasoning.

After two-stage SFT and RL optimization, $D _ { \phi }$ can convert freeform model reasoning outputs into structured hallucination diagnostic feedback. This feedback is used in subsequent FaithRS training to guide the revision decisions of the Safe Gate and to provide parseable process-level signals for reward design. In the reward design section, we further define a deterministic mapping from the diagnostic result d to the

visual faithfulness score $S ( d )$

## 2.2 Hallucination-aware Self-reflection Reasoning Model

After obtaining the structured hallucination detector, we further train the Hallucination-aware Self-reflection Reasoning Model (FaithRS), whose overall training framework is illustrated in Fig. 3. The goal is to enable the generation model to explicitly use detector feedback to correct reasoning content that is not grounded in visual evidence. Given an input $x =$ $( I , Q )$ , the model first generates an initial response. The hallucination detector $D _ { \phi }$ then diagnoses the reasoning process in a structured form. If the diagnosis indicates hallucinations in the reasoning process, the model uses this feedback to generate a corrected response. In this way, hallucination feedback is no longer used only as a post-hoc evaluation signal. Instead, it becomes an explicit training condition that guides the model to revise its reasoning process under visual evidence constraints.

## Data Construction for FaithRS.

The training corpus for FaithRS is composed of a supervised cold-start subset and a detector-guided RL subset. For the cold-start stage, we first construct an instruction-tuning dataset using a curated collection of remote sensing VQA data (Zhan et al., 2023) that covers both object-centric scenes and highresolution aerial imagery. Specifically, we randomly sample 30k image, question, answer triples, then prompt GPT-4o to generate concise evidence-grounded reasoning for each item, normalising every sample into a reasoning, answer pair. This stage familiarises the model with the desired “think then-answer” format and anchors the reasoning to observable visual cues, providing a stable template for the subsequent self-reflection RL phase.

For the RL stage, we construct a remote sensing visual question answering training pool from GeoLLaVA (Wang et al., 2025b). We first collect about 20k image, question, answer triples and use the SFT-trained model to generate candidate responses for each sample. We then apply the trained hallucination detector $D _ { \phi }$ to diagnose whether each response contains a hallucination. According to answer correctness and hallucination risk, the samples are divided into an easy training pool and a hard training pool. Easy samples are defined as those with correct initial answers and no detected hallucinations, whereas hard samples include those with incorrect initial answers or detected hallucination risks. This process yields 6,750 easy samples and 13,250 hard samples. We use the hard split for detector-guided self-reflection training, since these samples contain more complex errors and therefore require diagnostic feedback for efective correction.

## Hallucination-aware Self-reflection.

Given an input $x = ( I , Q )$ , the reasoning model first generates an initial response:

$$
y _ { 1 } = ( r _ { 1 } , a _ { 1 } ) \sim \pi _ { \theta } ( \cdot \mid x ) ,\tag{1}
$$

where $r _ { 1 }$ denotes the initial reasoning process and $a _ { 1 }$ denotes the initial answer. Given the reference answer $a ^ { * }$ , we define the correctness of the initial answer as

$$
c _ { 1 } = C ( a _ { 1 } , a ^ { * } ) .\tag{2}
$$

The hallucination detector $D _ { \phi }$ then performs structured diagnosis on the initial response:

$$
d _ { 1 } = D _ { \phi } ( I , Q , a ^ { * } , r _ { 1 } , a _ { 1 } ) .\tag{3}
$$

Based on the initial diagnosis $d _ { 1 }$ and answer correctness $c _ { 1 }$ FaithRS applies a Safe Gate to determine whether revision is required and which content should be revised:

$$
m = \mathcal { G } ( d _ { 1 } , c _ { 1 } ) = \left\{ \begin{array} { l l } { \mathsf { m } _ { \mathrm { s k i p } } , } & { H ( d _ { 1 } ) = 0 , } \\ { \mathsf { m } _ { \mathrm { a l l } } , } & { H ( d _ { 1 } ) = 1 , c _ { 1 } = 0 , } \\ { \mathsf { m } _ { \mathrm { r } 1 } , } & { H ( d _ { 1 } ) = 1 , c _ { 1 } = 1 , } \end{array} \right.\tag{4}
$$

where $H ( d _ { 1 } ) \in 0 $ , 1 denotes whether the detector identifies hallucinated content, and $c _ { 1 } \in 0$ , 1 indicates whether the initial answer is correct. Under ${ \mathsf { m } } _ { \mathrm { s k i p } } ,$ the initial response is retained without an additional generation step. Under $\mathsf { m } _ { \mathrm { a l l } } .$ , both the reasoning process and final answer are revised because the initial response contains hallucination and yields an incorrect answer. Under $\mathsf { m } _ { \mathrm { r } 1 }$ , only the reasoning process is revised while the correct answer is preserved. This routing mechanism avoids unnecessary revision when no hallucination is detected and prevents answer degradation when correcting an unfaithful reasoning chain.

When the routing function triggers revision, the structured diagnostic result $d _ { 1 }$ is converted into a textual self-reflection signal:

$$
\boldsymbol { u } _ { 1 } = \Psi ( d _ { 1 } , m ) ,\tag{5}
$$

where Ψ extracts the hallucination type, error cause, supporting visual evidence, and diagnostic rationale from the hallucination diagnosis, and organizes them into a revision prompt according to the routing mode. The model then generates a revised response conditioned on the input, the initial response, and the diagnostic feedback:

$$
y _ { 2 } = ( r _ { 2 } , a _ { 2 } ) \sim \pi _ { \theta } ( \cdot \mid x , y _ { 1 } , u _ { 1 } ) .\tag{6}
$$

We denote the final response as $y _ { f } = ( r _ { f } , a _ { f } )$ , where $r _ { f }$ and $a _ { f }$ are the final self-reflective reasoning process and the final answer, respectively. For skipped samples, $y _ { f } = y _ { 1 }$ ; for revised samples, $y _ { f } = y _ { 2 }$ . The final answer correctness is then computed, and the hallucination detector is invoked again to obtain the final diagnosis:

$$
c _ { 2 } = C ( a _ { f } , a ^ { * } ) , \qquad d _ { 2 } = D _ { \phi } ( I , Q , a ^ { * } , r _ { f } , a _ { f } ) .\tag{7}
$$

Each training trajectory therefore contains both the initial state $( y _ { 1 } , d _ { 1 } , c _ { 1 } )$ and the final state $( y _ { f } , d _ { 2 } , c _ { 2 } )$ . This paired structure allows the training objective to evaluate not only the quality of the final response, but also whether the model actually corrects hallucinated reasoning in the initial response.

The policy model is optimized with a trajectory-level composite reward. For each two-stage training trajectory $\tau = ( x , y _ { 1 } , d _ { 1 } , m , y _ { f } , d _ { 2 } )$ , we construct the reward from five aspects: final answer correctness $R _ { \mathrm { a c c } }$ , final hallucination faithfulness $R _ { \mathrm { h a l l u } } ,$ hallucination improvement before and after revision $R _ { \Delta }$ , output format $R _ { \mathrm { f m t } }$ , and revision safety $R _ { \mathrm { s a f e } }$

$$
\begin{array} { r } { R ( \tau ) = 0 . 4 0 R _ { \mathrm { a c c } } + 0 . 2 0 R _ { \mathrm { h a l l u } } + 0 . 2 0 R _ { \Delta } } \\ { + 0 . 1 0 R _ { \mathrm { f m t } } + 0 . 1 0 R _ { \mathrm { s a f e } } . \qquad } \end{array}\tag{8}
$$

The answer accuracy reward is given by the final answer correctness:

$$
R _ { \mathrm { a c c } } = c _ { 2 } = C ( a _ { f } , a ^ { \ast } ) .\tag{9}
$$

The hallucination-aware reward is mapped from the final diagnostic result $d _ { 2 } \mathbf { : }$

$$
R _ { \mathrm { h a l l u } } = S ( d _ { 2 } ) ,\tag{10}
$$

where $S ( \cdot ) \in [ 0 , 1 ]$ denotes the hallucination-aware faithfulness score (Appendix B.1) determined by the structured diagnostic result. A higher score indicates more faithful rea soning and a lower degree of hallucination. The hallucination improvement reward explicitly measures the change in faithfulness before and after revision. Let $s _ { 1 } = S ( d _ { 1 } ) , s _ { 2 } = S ( d _ { 2 } )$ and $\Delta s = s _ { 2 } - s _ { 1 }$ . We use a monotonic piecewise function to convert ∆s into an improvement reward:

$$
R _ { \Delta } = \Gamma ( \Delta s ) ,\tag{11}
$$

where Γ(·) assigns higher rewards to substantial improvements and penalizes faithfulness degradation (Appendix B.2). This encourages the model not only to produce a high-quality final response, but also to repair hallucinated reasoning in the initial response. The format reward $R _ { \mathrm { f m t } }$ constrains the model to produce a concise and parseable reasoning-answer structure. The safety reward $R _ { \mathrm { s a f e } }$ penalizes answer degradation caused by revision. In particular, under the $\mathsf { m } _ { \mathrm { r } 1 }$ mode, if the initial answer is reliable but the revised answer loses correctness, the safety reward is set to a low value. This prevents the model from damaging an originally correct answer while repairing the reasoning process.

For the same input x<sub>i</sub>, the policy model samples G two-stage trajectories $\{ \tau _ { i , g } \} _ { g = 1 } ^ { G }$ and computes the composite reward for each trajectory:

$$
R _ { i , g } = R ( \tau _ { i , g } ) .\tag{12}
$$

The rewards are then normalized within the candidate trajectory group of the same input to obtain the relative advantage:

$$
\hat { A } _ { i , g } = \frac { R _ { i , g } - \operatorname* { m e a n } _ { g ^ { \prime } } ( R _ { i , g ^ { \prime } } ) } { \mathrm { s t d } _ { g ^ { \prime } } ( R _ { i , g ^ { \prime } } ) + \varepsilon } .\tag{13}
$$

This within-group normalization makes the optimization focus on the relative quality of diferent candidate trajectories under the same image-question input, rather than on absolute reward-scale diferences across samples.

We optimize FaithRS with GSPO (Zheng et al., 2025). Unlike methods based on token-level (Shao et al., 2024) importance ratios, GSPO computes the likelihood ratio between the new and old policies at the sequence level. For the final response $y _ { f , i , g }$ of the g-th trajectory, we denote its context as $\operatorname { c t x } _ { i , g } .$ . If the sample does not trigger revision, then ct $\mathbf { x } _ { i , g } = x _ { i }$ and $y _ { f , i , g } = y _ { 1 , i , g } .$ If the sample triggers revision, then ct ${ \mathfrak { c } } _ { i , g } = ( x _ { i } , y _ { 1 , i , g } , u _ { 1 , i , g } )$ and $y _ { f , i , g } = y _ { 2 , i , g } .$ For brevity, define $p _ { \eta , i , g , t } = \pi _ { \eta } ( y _ { f , i , g , t } \mid \mathrm { c t x } _ { i , g } , y _ { f , i , g , < t } )$ for $\eta \in \{ \theta , \theta _ { \mathrm { o l d } } \}$ . The sequence-level importance ratio is defined as

$$
w _ { i , g } ^ { \mathrm { G S P O } } ( \theta ) = \exp \left( \frac { 1 } { | y _ { f , i , g } | } \sum _ { t = 1 } ^ { | y _ { f , i , g } | } \log \frac { p _ { \theta , i , g , t } } { p _ { \theta _ { \mathrm { o l d } } , i , g , t } } \right) .\tag{14}
$$

Based on this sequence-level ratio, the clipped GSPO objective is

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { G S P O } } ( \theta ) = \mathbb { E } _ { i , g } \big [ \operatorname* { m i n } \bigl ( w _ { i , g } ^ { \mathrm { G S P O } } ( \theta ) \hat { A } _ { i , g } , } \\ & { \qquad \mathrm { c l i p } \left( w _ { i , g } ^ { \mathrm { G S P O } } ( \theta ) , 1 - \epsilon _ { c } , 1 + \epsilon _ { c } \right) \hat { A } _ { i , g } \bigr ) \big ] . } \end{array}\tag{15}
$$

where $\epsilon _ { c }$ is the clipping coeficient, and $\theta _ { \mathrm { o l d } }$ denotes the old policy parameters used to sample the trajectories.

## 3 Experiments

## 3.1 Experimental Setup

## Implementation Details.

We train FaithRS using Qwen3.5 backbone models at three scales, namely 0.8B, 4B, and 9B (Qwen Team, 2026). Before reinforcement learning, we conduct supervised cold-start training with LoRA (Hu et al., 2022) applied to all linear layers. The learning rate is set to $2 \times 1 0 ^ { - 4 }$ , and the maximum sequence length is set to 4096, allowing the model to learn a stable reasoning-answer output format. During reinforcement learning, we optimize the model with GSPO (Zheng et al., 2025). For each input question, the model samples 8 rollouts. The per-device batch size is set to 4, the model is trained for 1 epoch, and the learning rate is set to $1 \times 1 0 ^ { - 5 }$ . The sampling temperature is set to 0.8. All training experiments are conducted on 4 A100 GPUs with 80GB memory.

## Evaluation Benchmarks.

We evaluate task performance on four remote sensing multimodal understanding benchmarks, including MME-RealWorld-RS (Zhang et al., 2024), RSVQA-HR (Lobry et al., 2020), CHOICE (An et al., 2025), and LHRS (Muhtar et al., 2024). These benchmarks cover low- and high-resolution remote sensing scene understanding and reasoning in real-world scenarios, providing a broad evaluation of remote sensing perception, question answering, and reasoning abilities.

<table><tr><td>Model</td><td>Publication</td><td>MME-RealWorld-RS</td><td>RSVQA-HR</td><td>CHOICE</td><td>LHRS</td><td>Avg</td></tr><tr><td>General Domain MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o(Hurst et al., 2024)</td><td>arXiv 2024</td><td>21.36</td><td>60.82</td><td>65.66</td><td>66.15</td><td>53.50</td></tr><tr><td>Gemini-2.5-Pro(Comanici et al., 2025)</td><td>arXiv 2025</td><td>38.06</td><td>67.49</td><td>72.48</td><td>66.41</td><td>61.11</td></tr><tr><td>Claude-Haiku-4.5</td><td></td><td>21.83</td><td>38.79</td><td>59.82</td><td>54.03</td><td>43.62</td></tr><tr><td>Qwen3.5-0.8B (Qwen Team, 2026)</td><td></td><td>38.36</td><td>53.48</td><td>51.90</td><td>45.40</td><td>47.29</td></tr><tr><td>Qwen3.5-4B (Qwen Team, 2026)</td><td></td><td>53.22</td><td>65.92</td><td>70.84</td><td>68.54</td><td>64.63</td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026)</td><td></td><td>62.09</td><td>67.97</td><td>75.15</td><td>70.54</td><td>68.94</td></tr><tr><td>LLaVA-v1.6-Vicuna-7B (Liu et al., 2023)</td><td>NeurIPS 2023</td><td>38.87</td><td></td><td>53.32</td><td>52.87</td><td></td></tr><tr><td>LLaMA3-LLaVA-8B (Chen and Xing, 2024)</td><td></td><td>47.57</td><td></td><td>63.47</td><td>55.50</td><td></td></tr><tr><td>InternVL3.5-8B (Wang et al., 2025c)</td><td>arXiv 2025</td><td>52.00</td><td>73.83</td><td>62.70</td><td>54.52</td><td>60.76</td></tr><tr><td>DeepSeek-VL-7B-Chat (Lu et al., 2024)</td><td>arXiv 2024</td><td>34.96</td><td>67.54</td><td>62.07</td><td>47.98</td><td>53.14</td></tr><tr><td>Remote sensing specialized MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GeoChat (Kuckreja et al., 2023)</td><td>CVPR 2024</td><td>29.24</td><td>66.70</td><td>52.89</td><td>44.09</td><td>48.23</td></tr><tr><td>VHM (Pang et al., 2025)</td><td>AAAI 2025</td><td>38.38</td><td>75.93</td><td>58.35</td><td>49.42</td><td>55.52</td></tr><tr><td>EarthDial (Soni et al., 2025)</td><td>CVPR 2025</td><td>33.88</td><td>70.91</td><td>59.75</td><td>49.30</td><td>53.46</td></tr><tr><td>GeoZero (Wang et al., 2025a)</td><td>arXiv 2025</td><td>51.44</td><td>79.03</td><td>74.81</td><td>64.45</td><td>67.43</td></tr><tr><td>RemoteReasoner (Yao et al., 2026)</td><td>AAAI 2026</td><td>30.29</td><td>66.77</td><td>69.02</td><td>57.18</td><td>55.82</td></tr><tr><td>RSThinker (Liu et al., 2026)</td><td>ICLR 2026</td><td>38.41</td><td>63.52</td><td>56.70</td><td>58.35</td><td>54.25</td></tr><tr><td>FaithRS-0.8B</td><td>Ours</td><td>47.00</td><td>78.23</td><td>57.17</td><td>53.30</td><td>58.93</td></tr><tr><td>FaithRS-4B</td><td>Ours</td><td>57.33</td><td>77.55</td><td>72.72</td><td>74.52</td><td>70.53</td></tr><tr><td>FaithRS-9B</td><td>Ours</td><td>57.51</td><td>81.85</td><td>76.86</td><td>72.71</td><td>72.23</td></tr></table>

Table 1: Quantitative comparison of general-purpose and remote sensing MLLMs on four remote sensing benchmarks. Avg denotes the arithmetic mean across MME-RealWorld-RS, RSVQA-HR, CHOICE, and LHRS. Dark, medium, and light green indicate the best, second-best, and third-best results among models with reported scores in each column, respectively.

## Baselines.

To comprehensively evaluate the efectiveness of FaithRS, we compare it with two groups of baselines, following the categorization in Table 1. The first group consists of generalpurpose MLLMs, including the closed-source models GPT-4o (Hurst et al., 2024), Gemini-2.5-Pro (Comanici et al., 2025), and Claude-Haiku-4.5, as well as the open-source models Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B (Qwen Team, 2026), LLaVA-v1.6-Vicuna-7B (Liu et al., 2023), LLaMA3- LLaVA-8B (Chen and Xing, 2024), InternVL3.5-8B (Wang et al., 2025c), and DeepSeek-VL-7B-Chat (Lu et al., 2024). The second group consists of remote sensing MLLMs, including GeoChat (Kuckreja et al., 2023), VHM (Pang et al., 2025), EarthDial (Soni et al., 2025), GeoZero (Wang et al., 2025a), RemoteReasoner (Yao et al., 2026), and RSThinker (Liu et al., 2026). We re-evaluate all available baseline models using their oficial checkpoints under identical prompts, decoding configurations, and evaluation scripts.

## 3.2 Main Results

Table 1 reports the quantitative results on four remote sensing multimodal understanding benchmarks. Overall, FaithRS consistently improves the average performance of all three Qwen3.5 backbones. FaithRS-0.8B achieves an average score of 58.93, improving upon Qwen3.5-0.8B by 11.64 points. FaithRS-4B increases the average score from 64.63 to 70.53, yielding a gain of 5.90 points. FaithRS-9B further reaches 72.23, outperforming its backbone by 3.29 points and achieving the best average result among all evaluated models. These improvements across 0.8B, 4B, and 9B backbones demonstrate that the proposed hallucination-aware self-reflection training framework is efective across diferent model capacities.

Compared with existing remote sensing specialized MLLMs, FaithRS exhibits a clear overall advantage. FaithRS-4B achieves an average score of 70.53, surpassing GeoZero by 3.10 points despite using a smaller backbone than many competing models. FaithRS-9B further improves the average score to 72.23 and establishes the strongest overall performance in the table. Even FaithRS-0.8B reaches 58.93, outperforming GeoChat, VHM, EarthDial, RemoteReasoner, and RSThinker in terms of average score. These results indicate that remote sensing instruction tuning or task-specific adaptation alone does not fully exploit the reasoning capacity of MLLMs. By incorporating structured fine-grained hallucination diagnosis and detector-guided self-reflection into post-training, FaithRS achieves stronger and more balanced performance across remote sensing perception, question answering, and reasoning tasks.

Although FaithRS-9B improves substantially on RSVQA-HR, CHOICE, and LHRS, its MME-RealWorld-RS score decreases from 62.09 to 57.51. This suggests that the benefits of hallucination-aware self-reflection are task dependent and may involve a trade-of between conservative evidence-grounded generation and performance on certain real-world evaluation formats. Nevertheless, the consistent improvements in average performance across all three model scales, together with the strong results on RSVQA-HR, CHOICE, and LHRS, confirm the overall efectiveness and scalability of FaithRS.

![](images/18526f9a9d9491bbfe6dc4e0b13169170021fb6c3002cb53f50e2d7a318462bb.jpg)  
Figure 4: Ablation study of FaithRS at the 0.8B and 4B scales.

## 3.3 Ablation Study

To examine the contribution of the main post-training components, we conduct ablation studies on FaithRS-0.8B and FaithRS-4B. We evaluate a variant without RL, denoted as w/o RL, and a variant without the self-reflection mechanism, denoted as w/o Self-reflection. Figure 4 compares these variants with their corresponding Qwen3.5 backbones and the FaithRS models on four remote sensing benchmarks and the average score.

At the 0.8B scale, both ablated variants improve over the original backbone, but neither matches the full model. FaithRS-0.8B achieves an average score of 58.93, compared with 56.39 for w/o RL and 55.41 for w/o Self-reflection, and consistently performs best across all four benchmarks. A similar trend appears at the 4B scale, where the full model reaches 70.53, while removing reinforcement learning or self-reflection reduces the average score to 67.69 and 62.99, respectively. The larger drops on MME-RealWorld-RS, RSVQA-HR, and LHRS indicate that detector-guided reinforcement learning and self-reflection are important for complex visual reasoning and corrective generation. Overall, supervised cold-start training provides stable reasoning and answer generation, while detector-guided reinforcement learning and self-reflection enable the model to use diagnostic feedback for targeted correction. Their combination yields the strongest and most balanced performance.

## 3.4 Training Dynamics Analysis

To examine how FaithRS acquires self-reflective revision ability, we analyze the training dynamics of FaithRS-0.8B during Two-Stage Corrective Rollout. Figure 5(a) compares the hallucination rates of the initial and final responses. The initial hallucination rate remains high because the RL training set consists of challenging samples that require correction. As training proceeds, the hallucination rate of the final responses decreases steadily, while the hallucination repair rate rises substantially. This trend indicates that the model progressively learns to use structured detector feedback to identify unreliable content and produce revisions that are better grounded in visual evidence. Figure 5(b) presents the answer change patterns before and after revision. Answer preservation and answer improvement account for most revisions, whereas answer degradation remains consistently rare. This behavior is consistent with the intended routing strategy: the model preserves reliable answers while correcting unsupported reasoning, and performs broader revisions only when both the reasoning and answer are unreliable. FaithRS therefore learns a selective, feedback-conditioned revision policy rather than an indiscriminate rewriting strategy. Figure 5(c) shows the relationship between the training reward and answer degradation. Although the reward increases throughout training, the answer degradation rate remains low and exhibits no corresponding upward trend. This result suggests that FaithRS does not improve its reward by sacrificing answer correctness. Instead, the Safe Gate and safety reward constrain the revision process and reduce the risk of damaging initially reliable answers.

Overall, these training dynamics show that FaithRS gradually develops the ability to reduce hallucinations without compromising answer reliability. Structured diagnosis identifies what should be corrected, Two-Stage Corrective Rollout turns this feedback into learnable revision behavior, and the safety mechanism controls the risk introduced by revision.

## 4 Conclusion

This work investigates visually unfaithful reasoning in remote sensing MLLMs and proposes FaithRS, a hallucination-aware self-reflection framework for faithful remote sensing reasoning. FaithRS first employs a structured fine-grained hallucination detector to identify error types, causes, and supporting visual evidence. It then incorporates the diagnostic feedback into Safe Detector-Guided Corrective Rollout, where a safetyaware gate determines whether and how the initial response should be revised. A composite reward further encourages answer accuracy, visual faithfulness, efective hallucination correction, valid output format, and revision safety. Unlike conventional single-stage reinforcement learning based only on terminal feedback, FaithRS explicitly learns a diagnosticfeedback-conditioned revision policy while concentrating optimization on the final response. Experiments across multiple remote sensing benchmarks show that FaithRS consistently improves diferent Qwen3.5 backbones, reduces factual and logical hallucinations, and maintains a low answer degradation rate during revision. These results demonstrate the potential of diagnosis-guided self-reflection to improve the reliability of remote sensing multimodal reasoning through structured diagnosis, conditional revision, and safety-aware optimization.

![](images/e7c1129ffe4b2ef6e38e1fe3f78fedffeb3c3c156958ee84d98a76c01c1c205d.jpg)

![](images/1bec0dc78be23f5b8666844cd8222335849bb571a9f2af241aaaddf7e4885bec.jpg)

![](images/5b182b8a673db416f9eda2d4de25dcffaf2f7c7f0d13156f8ab5b94103e53f55.jpg)  
Figure 5: Training dynamics of FaithRS-0.8B during detector-guided corrective rollout.

## References

Xiao An, Jiaxing Sun, Zihan Gui, and Wei He. Choice: Benchmarking the remote sensing capabilities of large vision-language models. In Advances in Neural Information Processing Systems, 2025.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

Lin Chen and Long Xing. Open-llava-next: An open-source implementation of llava-next series for facilitating the large multi-modal model community. 2024. doi: 10.5281/zenodo. 13935471.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, and Xiang Li. Rsgpt: A remote sensing vision language model and benchmark. arXiv preprint arXiv:2307.15266, 2023.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. arXiv preprint arXiv:2311.15826, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Jiaqi Liu, Lang Sun, Ronghao Fu, and Bo Yang. Towards faith ful reasoning in remote sensing: A perceptually-grounded geospatial chain-of-thought for vision-language models. In International Conference on Learning Representations, 2026.

Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. Rsvqa: Visual question answering for remote sensing data. arXiv preprint arXiv:2003.07333, 2020.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525, 2024.

Junwei Luo, Zhen Pang, Yongjun Zhang, Tingzhu Wang, Linlin Wang, Bo Dang, Jiangwei Lao, Jian Wang, Jingdong Chen, Yihua Tan, and Yansheng Li. Skysensegpt: A finegrained instruction tuning dataset and model for remote sensing vision-language understanding. arXiv preprint arXiv:2406.10100, 2024.

Dilxat Muhtar, Zhenshi Li, Feng Gu, Xueliang Zhang, and Pengfeng Xiao. Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model. arXiv preprint arXiv:2402.02544, 2024.

Chao Pang, Xingxing Weng, Jiang Wu, Jiayu Li, Yi Liu, Jiaxing Sun, Weijia Li, Shuai Wang, Litong Feng, Gui-Song Xia, et al. Vhm: versatile and honest vision language model for remote sensing image analysis. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence, pages 6381–6388, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sagar Soni, Akshay Dudhane, Hiyam Debary, Mustansar Fiaz, Muhammad Akhtar Munir, Muhammad Sohail Danish, Paolo Fraccaro, Campbell D Watson, Levente J Klein, Fahad Shahbaz Khan, et al. Earthdial: Turning multisensory earth observations to interactive dialogues. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14303–14313. IEEE, 2025.

Di Wang, Shunyu Liu, Wentao Jiang, Fengxiang Wang, Yi Liu, Xiaolei Qin, Zhiming Luo, Chaoyang Zhou, Haonan Guo, Jing Zhang, Bo Du, Dacheng Tao, and Liangpei Zhang. Geozero: Incentivizing reasoning from scratch on geospatial scenes. arXiv preprint arXiv:2511.22645, 2025a.

Fengxiang Wang, Mingshuo Chen, Yueying Li, Di Wang, Haotian Wang, Zonghao Guo, Zefan Wang, Boqi Shan, Long Lan, Yulin Wang, Hongzhen Wang, Wenjing Yang, Bo Du, and Jing Zhang. Geollava-8k: Scaling remotesensing multimodal large language models to 8k resolution. arXiv preprint arXiv:2505.21375, 2025b.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265, 2025c.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Liang Yao, Fan Liu, Hongbo Lu, Chuanyi Zhang, Rui Min, Shengxiang Xu, Shimin Di, and Pai Peng. Remotereasoner: Towards unifying geospatial reasoning workflow. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 11883–11891, 2026.

Yang Zhan, Zhitong Xiong, and Yuan Yuan. Rsvg: Exploring data and models for visual grounding on remote sensing data. IEEE Transactions on Geoscience and Remote Sensing, 61: 1–13, 2023. doi: 10.1109/TGRS.2023.3250471.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are dificult for humans? arXiv preprint arXiv:2408.13257, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

## Appendices

## A Additional Related Work

## A.1 Multimodal Reasoning Models for Remote Sensing

In recent years, MLLMs have been increasingly applied to remote sensing image understanding. Early studies typically adapt general-domain MLLMs to the remote sensing domain through supervised fine-tuning on remote sensing image–text pairs, visual question answering samples, and instructionfollowing data. These adaptations enable models to describe remote sensing scenes, recognize objects, and interact with users through natural-language instructions. Representative methods, including RSGPT, GeoChat, SkySenseGPT, RS-GPT4V, SkyEyeGPT, and LHRS-Bot, construct remote sensing instruction datasets and employ vision–language alignment strategies to improve performance in scene classification, object localization, visual question answering, and relation understanding (Hu et al., 2023; Kuckreja et al., 2023; Luo et al., 2024; Xu et al., 2024; Zhan et al., 2024; Muhtar et al., 2024). Other studies incorporate stronger visual encoders, region-level visual representations, or preference optimization to improve perception and response generation in complex remote sensing scenes (Soni et al., 2025; Pang et al., 2024; Wang et al., 2025). Collectively, these studies demonstrate the potential of MLLMs for open-ended remote sensing understanding.

Recent studies have begun to investigate reliability evaluation and hallucination detection for remote sensing MLLMs, aiming to determine whether model responses are consistent with the evidence contained in remote sensing images (Pang et al., 2024; Liu et al., 2026). However, most existing studies focus primarily on evaluation or detection. Although they can identify the presence of hallucinations, they rarely transform diagnostic information into learnable supervision for model training. For remote sensing multimodal reasoning, determining whether a response contains hallucinations is insuficient. A reliable model must also identify whether an error arises from object recognition, attribute description, spatial relation judgment, or logical inference, while preserving reliable content during revision. In contrast to existing methods, this work explicitly integrates fine-grained hallucination diagnosis into the training process. The Hallucination Detector produces structured diagnostic feedback, enabling the reasoning model to learn a more faithful self-correction policy under constraints imposed by remote sensing visual evidence.

## A.2 Multimodal Hallucinations and Visual Faithfulness

Multimodal hallucinations generally refer to inconsistencies between generated textual responses and the provided visual content (Li et al., 2023; Liu et al., 2026). Existing studies indicate that hallucinations in MLLMs arise from the combined efects of training data, model architecture, and optimization objectives. At the data level, large-scale image–text corpora are often collected from the web or filtered using heuristic rules, which can introduce noisy annotations, weakly aligned samples, and statistical biases. Instruction-tuning data frequently relies on language-model-generated content, which may further introduce descriptions unsupported by visual evidence. At the model level, the perceptual capability of the visual encoder, the representation capacity of the cross-modal alignment interface, and the knowledge priors encoded in the language model jointly afect how visual evidence is incorporated into generation. At the training level, MLLMs generally inherit the autoregressive next-token prediction objective, which optimizes the likelihood of text generation but imposes no explicit consistency constraints on visual facts such as objects, attributes, and spatial relations. These limitations are further amplified in remote sensing scenes because of small object scales, ambiguous land-cover boundaries, dense spatial layouts, and complex topological relations. Consequently, models are more likely to generate content that conflicts with visual evidence regarding object existence, attribute descriptions, and spatial relations (Kuckreja et al., 2023; Wang et al., 2025; Liu et al., 2026).

## B Complete Reward Definitions

## B.1 Hallucination-Aware Faithfulness Score S(d)

The structured detector diagnosis is $d = ( h , c , T , \rho , e , j )$ Here, $h \in \{ 0 , 1 \}$ indicates whether a hallucination is detected, c denotes the set of hallucination categories, $\tau$ denotes the set of fine-grained hallucination types, $\rho$ describes the error causes, e specifies the relevant visual evidence, and $j$ provides the diagnostic justification. Only $h , c ,$ and $\tau$ contribute to the scalar faithfulness score. The explanatory fields $\rho , e ,$ and $j$ are used to construct revision feedback and support qualitative auditing.

The final two-stage reward uses the following visual faithfulness score:

$$
S ( d ) = \left\{ \begin{array} { l l } { 0 . 5 0 , } & { d \mathrm { i s } \mathrm { m i s s i n g } } \\ { 1 . 0 0 , } & { \mathrm { o r c a n n o t } \mathrm { b e } \mathrm { p a r s e d } } \\ { 0 . 2 5 r _ { \mathrm { b i n } } ( h ) + 0 . 2 5 r _ { \mathrm { c a t } } ( c ) } & { h = 1 . } \\ { + 0 . 5 0 r _ { \mathrm { s u b } } ( \mathcal { T } ) } & { h = 0 , } \end{array} \right.\tag{16}
$$

Here, $r _ { \mathrm { b i n } } ( h ) = 0$ for a hallucinated response. The categorylevel score is defined as

$$
r _ { \mathrm { c a t } } ( c ) = \left\{ 0 , \begin{array} { l l } { 0 , } & { F a c t u a l H a l l u c i n a t i o n \in c , } \\ { 0 . 3 , } & { L o g i c a l H a l l u c i n a t i o n \in c . } \end{array} \right.\tag{17}
$$

To compute the subtype-level score, we first assign each

fine-grained hallucination type a severity value:

$$
\begin{array} { r l } & { \mu ( \mathrm { A 1 } ) = \mu ( \mathrm { A 2 } ) = \mu ( \mathrm { A 3 } ) = 1 . 0 0 , } \\ & { \mu ( \mathrm { B 1 } ) = \mu ( \mathrm { B 3 } ) = 0 . 4 0 , } \\ & { \mu ( \mathrm { B 2 } ) = 0 . 3 0 , \qquad \mu ( \mathrm { B 4 } ) = 0 . 2 0 . } \end{array}\tag{18}
$$

Unknown or missing subtype labels are assigned a default severity of 0.52. For a diagnosis containing multiple subtypes, we use the maximum subtype severity and apply an additional multiplicity penalty:

$$
p ( \mathcal { T } ) = \operatorname* { m i n } \left( 0 . 1 0 , 0 . 0 3 \operatorname* { m a x } \left( | \mathcal { T } | - 1 , 0 \right) \right) .\tag{19}
$$

The subtype-level faithfulness score is then

$$
r _ { \mathrm { s u b } } ( \mathcal { T } ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( 1 - \operatorname* { m a x } _ { t \in \mathcal { T } } \mu ( t ) - p ( \mathcal { T } ) \right) ,\tag{20}
$$

where the maximum severity is set to 0.52 when T is empty. Therefore, more severe or more numerous hallucination types result in a lower faithfulness score.

## B.2 Delta Reward $R _ { \Delta }$

Let $d _ { k } = \left( h _ { k } , c _ { k } , \mathcal { T } _ { k } , \rho _ { k } , e _ { k } , j _ { k } \right)$ , for $k \in \{ 1 , 2 \}$ , denote the detector diagnoses of the initial and final responses, respectively. We define the change in the hallucination score as

$$
\Delta s = S ( d _ { 2 } ) - S ( d _ { 1 } ) .\tag{21}
$$

The delta reward is computed using the following piecewise function:

$$
\Gamma ( \Delta s ) = \left\{ \begin{array} { l l } { 1 . 0 0 , } & { \Delta s \geq 0 . 4 , } \\ { 0 . 8 0 , } & { 0 . 2 \leq \Delta s < 0 . 4 , } \\ { 0 . 6 5 , } & { 1 0 ^ { - 6 } < \Delta s < 0 . 2 , } \\ { 0 . 5 0 , } & { | \Delta s | \leq 1 0 ^ { - 6 } , } \\ { 0 . 2 5 , } & { - 0 . 2 < \Delta s < - 1 0 ^ { - 6 } , } \\ { 0 . 0 0 , } & { \Delta s \leq - 0 . 2 . } \end{array} \right.\tag{22}
$$

Accordingly, the reward wrapper returns $R _ { \Delta } = \Gamma ( \Delta \boldsymbol { s } )$ when both cached hallucination scores are available. Otherwise, it assigns the neutral reward $R _ { \Delta } = 0 . 5$

## B.3 Safe Reward $R _ { \mathrm { s a f e } }$

Let $a _ { 1 }$ and $a _ { 2 }$ be the initial and revised answer correctness scores. The safe reward is

$$
R _ { \mathrm { s a f e } } = \left\{ \begin{array} { l l } { 1 . 0 0 , } & { a _ { 2 } > a _ { 1 } + 1 0 ^ { - 6 } , } \\ { 0 . 7 0 , } & { a _ { 2 } \geq a _ { 1 } - 1 0 ^ { - 6 } , } \\ { 0 . 0 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{23}
$$

In the mode $\mathsf { m } _ { \mathrm { r } 1 } , \mathrm { i f } a _ { 1 } \geq \tau$ and the revised answer degrades by more than the epsilon tolerance, the reward is forced to 0. This additional rule protects answer-correct samples from being rewritten into worse answers while the reasoning is revised.

## B.4 GSPO Objective

For a prompt group indexed by i and a sampled completion indexed by g, the code uses group normalization:

$$
\hat { A } _ { i , g } = \frac { R _ { i , g } - \operatorname* { m e a n } ( \{ R _ { i , j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { i , j } \} _ { j = 1 } ^ { G } ) + \epsilon } .\tag{24}
$$

The sequence-level importance ratio is computed from the average log-ratio across completion tokens:

$$
w _ { i , g } ^ { \mathrm { G S P O } } ( \theta ) = \left( \frac { \pi _ { \theta } ( y _ { i , g } \mid x _ { i } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , g } \mid x _ { i } ) } \right) ^ { \frac { 1 } { | y _ { i , g } | } } .\tag{25}
$$

The trainer then applies asymmetric clipping with lower and upper bounds $1 { - } \epsilon _ { \mathrm { l o w } }$ and $1 + \epsilon _ { \mathrm { h i g h } }$ , where the script sets $\epsilon _ { \mathrm { l o w } } =$ $3 \times 1 0 ^ { - 4 }$ and $\epsilon _ { \mathrm { h i g h } } = 4 \times 1 0 ^ { - 4 }$ . The optimization is finalresponse-only because the loss scale is set to last\_round.

## C Complete Prompt Templates

## C.1 FaithRS Initial Generation Prompt

The generator SFT and RL prompts share a common remote sensing VQA instruction:

Initial Generation Prompt   
You are a remote sensing visual question answering   
expert.   
Answer based solely on what you directly observe in   
the image.   
Use <think>...</think> for your reasoning, then write   
ONLY the final answer after </think>.   
Answer format rules:   
- Yes/No questions: answer ’yes’ or ’no’   
- Count questions: answer with a number only   
- Other questions: answer in 1-5 words

The RL data append /nothink to the user turn so that the generator remains in the raw-answer data format used by the training scripts.

## C.2 Revision Prompts

The two revision prompts used by the Safe Gate are:

Revision Prompt: Reasoning and Answer   
The previous answer contains hallucination and may be   
incorrect.   
Detector feedback:   
- Types: {types}   
- Problems: {reasons}   
- Evidence / missing evidence: {evidence}   
- Justification: {justification}   
You must:   
1. Remove unsupported claims.

2. Correct wrong object, attribute, count, or spatial   
claims.   
3. Revise both reasoning and answer if needed.   
4. Use only visible evidence from the image.   
Output strict JSON:   
{   
"reasoning": "...",   
"answer": "..."   
}

Revision Prompt: Reasoning Only   
The previous final answer matches the ground truth,   
but the reasoning contains unsupported claims.   
Detector feedback:   
- Types: {types}   
- Problems: {reasons}   
- Evidence / missing evidence: {evidence}   
- Justification: {justification}   
You must:   
1. Preserve the final answer unless the image clearly   
contradicts it.   
2. Remove unsupported reasoning.   
3. Replace hallucinated claims with visually grounded   
evidence.   
4. Output strict JSON only:   
{   
"reasoning": "...",   
"answer": "..."   
}

## D Additional Training Analysis

This section first examines which structured diagnostic fields and training stages are important for the hallucination detector, and then analyzes the optimization behavior of FaithRS using the accompanying training diagnostics. In the curve analysis, all horizontal axes report relative global steps, solid curves denote the reported means, and shaded regions denote the corresponding standard deviations. Because several rates are computed over diferent event subsets, we interpret their trajectories separately rather than assuming that all curves share the same denominator.

## D.1 Hallucination Detector Ablations

We evaluate the hallucination detector from two complementary perspectives. First, we compare detector performance across the short-form SFT, full-form SFT, and RL stages to examine how structured prediction quality evolves throughout training. Second, we compare the baseline RL configuration with configurations involving diferent combinations of the explanatory fields ρ, e, and j, which correspond to hallucination\_reasons, evidence\_basis, and justification, respectively. Higher values indicate better performance for all reported metrics.

<table><tr><td>Stage</td><td>Parse (%)</td><td>Acc.</td><td>F1</td><td>MCC</td></tr><tr><td>SFT-Short</td><td>22.30</td><td>0.5331</td><td>0.3497</td><td>0.2386</td></tr><tr><td>SFT-Full</td><td>56.90</td><td>0.6737</td><td>0.6126</td><td>0.4690</td></tr><tr><td>RL (Best)</td><td>98.40</td><td>0.8865</td><td>0.9006</td><td>0.7685</td></tr></table>

Table 2: Comparison across hallucination detector training stages. Bold values indicate the best result in each column.

Training-stage comparison. Table 2 compares detector performance after short-form SFT, full-form SFT, and RL optimization. Moving from short-form to full-form SFT increases the parse rate by 34.60 percentage points, accuracy by 0.1406, F1 by 0.2629, and MCC by 0.2304. Subsequent RL optimization further improves the parse rate by 41.50 percentage points, accuracy by 0.2128, F1 by 0.2880, and MCC by 0.2995. These results show a consistent improvement throughout the training pipeline. Full-form supervision provides richer structured targets and substantially improves both output validity and classification performance, while RL further strengthens format stability and diagnostic accuracy. Because these stages are applied sequentially, the results should be interpreted as a comparison of training progression rather than as a strictly controlled component ablation.

Diagnostic-field ablation during RL. Table 3 compares the baseline RL configuration with configurations involving diferent combinations of the explanatory fields. The Baseline achieves the strongest overall performance, including the best parse rate, accuracy, precision, F1, MCC, and Factual-F1. Specifically, it reaches a parse rate of 98.40%, an overall F1 of 0.9006, and an MCC of 0.7685. The configuration involving e achieves the highest recall of0.9649 and the highest Logical-F1 of 0.7385, but its precision and MCC decrease to 0.8050 and 0.6959, respectively. Similar recall-oriented shifts are observed in several other field configurations, indicating a trade-of between sensitivity and balanced classification performance.

The field-level results further show that parseability and classification quality do not necessarily improve together. Several configurations retain parse rates above 96% but exhibit lower F1 and MCC than the Baseline. The largest MCC reduction is 0.1168 for the e+j configuration. This indicates that producing structurally valid outputs is not suficient to guarantee balanced discrimination between hallucinated and non-hallucinated responses or between factual and logical hallucinations.

Overall, the training-stage comparison shows that full-form SFT and subsequent RL progressively improve structured prediction quality. However, the diagnostic-field configurations do not outperform the Baseline in terms of overall balanced performance. Although some configurations improve recall or Logical-F1, these gains are accompanied by reductions in precision, F1, or MCC. Since the tables report point estimates without confidence intervals, the observed diferences establish the empirical ordering of the evaluated settings but should not be interpreted as evidence of statistical significance.

<table><tr><td>Setting</td><td>Parse (%)</td><td>Acc.</td><td>Prec.</td><td>Recall</td><td>F1</td><td>MCC</td><td></td><td>Factual-F1 Logical-F1</td></tr><tr><td>Baseline</td><td>98.40</td><td>0.8865</td><td>0.9014</td><td>0.8998</td><td>0.9006</td><td>0.7685</td><td>0.9033</td><td>0.7221</td></tr><tr><td>w/ρ</td><td>91.50</td><td>0.8534</td><td>0.8784</td><td>0.8629</td><td>0.8706</td><td>0.7018</td><td>0.8672</td><td>0.7322</td></tr><tr><td>w/e</td><td>97.70</td><td>0.8464</td><td>0.8050</td><td>0.9649</td><td>0.8777</td><td>0.6959</td><td>0.8794</td><td>0.7385</td></tr><tr><td>w/ j</td><td>95.30</td><td>0.8484</td><td>0.8472</td><td>0.8963</td><td>0.8711</td><td>0.6891</td><td>0.8338</td><td>0.6391</td></tr><tr><td>w/ρ+e</td><td>96.60</td><td>0.8685</td><td>0.8911</td><td>0.8770</td><td>0.8840</td><td>0.7323</td><td>0.8623</td><td>0.7060</td></tr><tr><td>w/  $\rho { + } j$ </td><td>92.30</td><td>0.8404</td><td>0.8024</td><td>0.9561</td><td>0.8725</td><td>0.6818</td><td>0.8760</td><td>0.7348</td></tr><tr><td>w/e+j</td><td>96.80</td><td>0.8243</td><td>0.7830</td><td>0.9578</td><td>0.8617</td><td>0.6517</td><td>0.8616</td><td>0.7051</td></tr><tr><td>w/  $\rho { + } e { + } j$ </td><td>96.70</td><td>0.8544</td><td>0.8897</td><td>0.8506</td><td>0.8697</td><td>0.7059</td><td>0.8569</td><td>0.6648</td></tr></table>

Table 3: Ablation of structured diagnostic fields during detector RL. Bold values indicate the best result in each column.

![](images/df378d82000a4629a897670e0732215cdb09d7c66d6a3e7a31ed7235fc821c23.jpg)  
Figure 6: Evolution of the composite reward and its constituent terms. Solid curves show means and shaded regions show standard deviations.

## D.2 Detector-Guided RL Training Dynamics

The training diagnostics reveal a coherent two-phase optimization process with a common transition around relative step 600. As shown in Figure 6, the mean composite reward first decreases slightly from approximately 0.49 to 0.45 during the first 200 steps, then rises to about 0.82 near step 600 and stabilizes around 0.89–0.91 after step 900. The early decrease coincides with a temporary reduction in the answer-accuracy reward, whereas the subsequent improvement is shared across all reward components.

The hallucination reward contributes the largest relative change, increasing from roughly 0.14 at the beginning of training to above 0.90 near convergence, with its sharpest rise between approximately steps 550 and 650. Over the same period, the answer-accuracy reward recovers from an early minimum near 0.42 to approximately 0.90, and the delta reward increases from about 0.50 to 0.86. In contrast, the safe reward remains comparatively stable within an approximately 0.75– 0.87 range. Thus, the improvement in hallucination-sensitive rewards is not accompanied by a collapse of the answer preservation objective. The generally narrower uncertainty bands near convergence further indicate more stable late-stage behavior.

The two detector passes further localize the source of this improvement. In the left panel of Figure 7, the initial-diagnosis hallucination rate (D1 Hall) remains high throughout training and decreases only moderately from approximately 90% to around 80%, which is consistent with the use of the hard detector-guided RL subset. By contrast, the reported final hallucination rate falls from roughly 85%–90% to below 10%, while the fix rate rises from below 10% to above 90%. The largest separation between the initial and final diagnoses appears around steps 580–650, matching the transition in the reward curves. This pattern is consistent with the policy learning to use detector feedback on dificult first-pass responses rather than obtaining a higher reward merely by avoiding revision.

As correction becomes more reliable, the Safe Gate also adopts a more selective routing distribution. In the middle panel of Figure 7, the broad revise-all route decreases from approximately 80% to the mid-60% range, while the keep route grows from below 10% to around 20% and the preserve-answer route increases from roughly 8%–10% to approximately 15%. The policy therefore does not improve by sending an increasing fraction of samples through unrestricted rewriting. Instead, the higher reward coincides with more frequent use of routes that retain reliable content or restrict the scope of revision.

The accuracy transitions provide a final safety check on this routing behavior. The right panel of Figure 7 shows that the proportion of revisions that improve answer correctness rises from approximately 30% during early training to around 58%–62% after step 400. The preserved-correctness rate decreases from roughly 65% to approximately 35%–42% as more samples move into the improved category, whereas the degraded fraction remains small and declines from an early range of approximately 3%–5% to mostly 1%–3% during later training.

Taken together, the detector ablations and generator diagnostics support a coherent training account. Complete structured targets and RL provide the strongest observed combination of detector parseability and diagnostic quality, while detectorguided generator training aligns higher reward with more successful repair, more selective routing, and infrequent answer degradation. These results characterize the evaluated training process and do not replace held-out evaluation. In particular, they do not establish causal efects, formal statistical significance, or generalization beyond the evaluated settings.

![](images/15b9471853974aa7c6f70056f3c68830cee7604037f6c350c69f8d17c0dc1143.jpg)

![](images/78b232fb3a8816df9930dfb8b1e7a0e8dec3e60f79be9c7217be0a85dd2da1b9.jpg)

![](images/4139c802781521bf96e329091c434bc116a7736163a4fd135273b491eee47acf.jpg)  
Figure 7: Core detector-guided training dynamics. Left: initial hallucination, final hallucination, and fix rates. Middle: Safe-Gate routing rates. Right: improved, preserved, and degraded answer-correctness outcomes.

Moreover, the plotted rates may use diferent conditioning sets and should not be combined arithmetically; the supported evidence lies in their individual trajectories and aligned transition points.

## E Qualitative Examples on MME-RealWorld-RS

We present nine representative examples from MME-RealWorld-RS covering color recognition, object counting, and spatial localization. For each question type, the selected cases include stable successes shared by all three FaithRS scales, scale-dependent corrections, and representative reasoning failures. The consistently correct examples demonstrate that FaithRS preserves reliable visual question answering behavior across diferent model capacities, while the failure cases reveal limitations that are not fully reflected by aggregate accuracy.

Color recognition. Figure 8 presents a scale-dependent correction, a stable success, and a shared failure. In Case 1, FaithRS-0.8B classifies the L-shaped house as gray and explains that gray is a typical color for residential buildings. This explanation relies partly on a generic scene prior rather than on the observed target roof. FaithRS-4B and FaithRS-9B instead identify the roof as green, consistent with the reference answer. Case 2 provides a stable positive example: all three models correctly identify the ship at the bottom of the image as green and support the answer with concise and mutually consistent visual descriptions. This result shows that the target attribute can be reliably preserved across model scales when the object is suficiently distinguishable. By contrast, Case 3 remains challenging for all three models. Each model describes the sports field as blue and white, while the reference answer specifies blue, red, and yellow. The explanations further rationalize the incomplete prediction by referring to colors that are typical of sports fields or pool-like structures. This shared error indicates that increasing model capacity alone does not fully resolve fine-grained multicolor recognition when the target occupies only a small region of a wide remote sensing scene.

Counting and target absence. Figure 9 illustrates three distinct counting behaviors. In Case 4, FaithRS-0.8B explicitly states that no red cars are visible on the highway, but selects option A, which denotes one red car. The visual conclusion in the reasoning is therefore inconsistent with the final multiplechoice answer. FaithRS-4B and FaithRS-9B correctly map the absence judgment to option E. Case 5 reflects a diferent failure mode. FaithRS-0.8B and FaithRS-4B fail to detect the two red cars beside the J-shaped building, whereas FaithRS-9B correctly identifies both targets. This result suggests that small-object counting in dense urban scenes remains sensitive to model capacity. Case 6 provides a stable absence-detection example: all three models correctly determine that no white squares are present in the specified region and consistently select option E. The contrast between Cases 5 and 6 shows that selecting the no-target option is not inherently dificult. The main challenge is distinguishing genuine target absence from the presence of visually small or weakly contrasted objects.

Spatial localization and semantic faithfulness. Figure 10 includes a stable localization success, a scale-dependent spatial correction, and a case in which correct answers coexist with unnecessary semantic interpretation. In Case 7, all three models correctly locate the roundabout to the right of the image center. Their explanations consistently refer to the visible circular road structure, providing a stable example of evidencegrounded spatial localization across model scales. In Case 8, FaithRS-0.8B places the red truck in the lower-left region near the marina, while FaithRS-4B and FaithRS-9B correctly locate it above the gray houses along the right edge. The smaller model produces a fluent and spatially specific description, but the described region is not grounded in the actual target location. Case 9 provides a complementary example in which all three models select the correct upper-right location of the white building. However, some explanations introduce additional interpretations of the surrounding black open space. In particular, FaithRS-9B describes the region as a storage area for coal or ore, although this functional interpretation is not required by the question and cannot be verified directly from the image. This case shows that correct spatial answers do not necessarily guarantee that every explanatory statement is visually supported.

Overall, the selected cases reveal three complementary properties of FaithRS. First, Cases 2, 6, and 7 demonstrate stable visual recognition, absence detection, and spatial localization across all three model scales, indicating that the learned corrective behavior does not prevent the models from retaining reliable task competence. Second, Cases 1, 4, 5, and 8 show that increasing model capacity can improve subtle attribute recognition, answer selection, small-target detection, and spatial grounding. Third, Cases 3 and 9 demonstrate that larger models remain vulnerable to shared perceptual errors and unsupported semantic elaboration. These qualitative results therefore complement the quantitative evaluation by showing both the reliable capabilities preserved by FaithRS and the reasoning failures that remain unresolved.

![](images/9b359a279dba883de25fc4070891ad59b688f3350b9a2c875745869d17dbda60.jpg)  
Figure 8: Representative color-recognition examples from MME-RealWorld-RS. Case 1 shows a scale-dependent correction of a prior-driven attribute error. Case 2 is answered correctly by all three FaithRS scales, demonstrating stable color recognition. Case 3 remains a shared multicolor-recognition failure across all three models.

![](images/7b6d9597e1b23f0d023961a2e164fda0cadec09c22be4c5d16bd0673a164009c.jpg)  
Figure 9: Representative counting examples from MME-RealWorld-RS. Case 4 exposes an inconsistency between the absence judgment in th reasoning and the final option selected by FaithRS-0.8B. Case 5 shows a scale-dependent small-target omission, with only FaithRS-9B correctl identifying the two red cars. Case 6 is answered correctly by all three models and demonstrates stable target-absence recognition.

![](images/89b584fd94eefa586947e93b5951d8a5ab7da94a5c9498bb5b158b2b48ddcb44.jpg)  
Figure 10: Representative spatial-localization examples from MME-RealWorld-RS. Case 7 is answered correctly by all three FaithRS scales and provides a stable evidence-grounded localization example. Case 8 shows scale-dependent recovery from spatial misgrounding by FaithRS-0.8B. In Case 9, all three models select the correct location, although some explanations introduce semantic interpretations that are not directly verifiable from the image.

## Appendix References

Yushi Hu, Jianlong Yuan, Congcong Wen, Xiaoqiang Lu, and Xiang Li. RSGPT: A remote sensing vision language model and benchmark. arXiv preprint arXiv:2307.15266, 2023.

Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. GeoChat: Grounded large vision-language model for remote sensing. arXiv preprint arXiv:2311.15826, 2023.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023.

Yi Liu, Jing Zhang, Di Wang, Xiaoyu Tian, Haonan Guo, and Bo Du. Seeing clearly without training: Mitigating hallucinations in multimodal llms for remote sensing. arXiv preprint arXiv:2603.02754, 2026.

Junwei Luo, Zhen Pang, Yusheng Zhang, Ting Wang, Linlin Wang, Bo Dang, Jiangwei Lao, Jian Wang, Jingdong Chen, Yihong Tan, and Yu Li. SkySenseGPT: A finegrained instruction tuning dataset and model for remote sensing vision-language understanding. arXiv preprint arXiv:2406.10100, 2024.

Dilxat Muhtar, Zhenshi Li, Feng Gu, Xuelong Zhang, and Pengfeng Xiao. LHRS-Bot: Empowering remote sensing with VGI-enhanced large multimodal language model. arXiv preprint arXiv:2402.02544, 2024.

Chao Pang, Xinyi Weng, Jiannan Wu, Jiang Li, Yi Liu, Jia Sun, Weijia Li, Shuai Wang, Lin Feng, Gui-Song Xia, and Conghui He. VHM: Versatile and honest vision language model for remote sensing image analysis. arXiv preprint arXiv:2403.20213, 2024.

Sagar Soni, Akshay Dudhane, Hiyam Debary, Mustansar Fiaz, Muhammad Akhtar Munir, Muhammad Sohail Danish, Paolo Fraccaro, Campbell D. Watson, Levente J. Klein, and Salman Khan. Earthdial: Turning multi-sensory earth observations to interactive dialogues. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Fei Wang, Mingjie Chen, Yuxuan Li, Di Wang, Hao Wang, Zhe Guo, Ziyi Wang, Bowen Shan, Li Lan, Yi Wang, Haoran Wang, Wenkai Yang, Bo Du, and Jian Zhang. GeoLLaVA-8K: Scaling remote-sensing multimodal large language models to 8K resolution. arXiv preprint arXiv:2505.21375, 2025.

Linrui Xu, Ling Zhao, Wang Guo, Qiujun Li, Kewang Long, Kaiqi Zou, Yuhan Wang, and Haifeng Li. Rsgpt4v: A unified multimodal instruction-following dataset for remote sensing image understanding. arXiv preprint arXiv:2406.12479, 2024.

Yang Zhan, Zhitong Xiong, and Yuan Yuan. Skyeyegpt: Unifying remote sensing vision-language tasks via instruction tuning with large language model. arXiv preprint arXiv:2401.09712, 2024.