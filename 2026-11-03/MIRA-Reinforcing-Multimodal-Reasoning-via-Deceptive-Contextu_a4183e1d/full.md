# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

Zhihan Yin<sup>1,2</sup> Jianxin Liang<sup>1</sup> Yifeng Yao<sup>1</sup> Nonghai Zhang<sup>1,2</sup>   
Bingyue Peng<sup>2</sup> Chunguang Qie<sup>2</sup> Huishuai Zhang<sup>1,3,∗</sup> Dongyan Zhao<sup>1,3,4,∗</sup>   
<sup>1</sup>Peking University <sup>2</sup>ByteDance <sup>3</sup>Wangxuan Institute of Computer Technology   
<sup>4</sup>National Engineering Research Center of New Electronic Publishing Technologies <sup>∗</sup>Corresponding authors.

## Abstract

While Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced multimodal reasoning, existing frameworks suffer from a critical “perception–reasoning gap.” Due to an over-reliance on seemingly relevant visual tokens, even minor perceptual perturbations can propagate through the reasoning process, leading to compounding hallucinations. To address this issue, we propose MIRA, a training framework that improves robustness to erroneous visual contexts while enhancing logical reasoning ability. By injecting filtered, deceptive visual contexts during training, we construct challenging “hard traps” that stress-test the model’s robustness to misleading information. Furthermore, we introduce a group-wise reflection-triggering mechanism that enables the model to autonomously detect and rectify inconsistencies between visual cues and logical constraints, which is ultimately internalized as an intrinsic policy behavior. Importantly, MIRA does not rely on human annotations. Experimental results across multiple reasoning and perception benchmarks show that MIRA yields an average 7% improvement on indomain benchmarks, with consistent improvements on vision-centric benchmarks. These findings highlight that active error recovery is essential for robust multimodal reasoning. Comprehensive ablation studies and analyses further provide insights into why MIRA is effective. Code is available at Github.

## 1 Introduction

Recent advances in large language models (LLMs) have demonstrated that Reinforcement Learning with Verifiable Rewards (RLVR) can substantially enhance reasoning capabilities, enabling strong performance on complex tasks such as mathematics and programming [Shao et al., 2024, Guo et al., 2025]. As a natural extension of LLMs, Vision-Language Models (VLMs) are required to tackle more challenging multimodal reasoning scenarios [Liu et al., 2023, Bai et al., 2025a], where models must not only perform long-context reasoning in the language space but also effectively integrate and interpret information across modalities.

To address this challenge, a series of multimodal reasoning approaches have been proposed, achieving notable progress. Methods such as Vision-R1 [Huang et al., 2026] and VL-R1 [Zhang et al., 2025] successfully transfer the long-chain reasoning capabilities of LLMs to the multimodal domain through RLVR-based post-training, enabling VLMs to exhibit sophisticated reasoning behaviors [Meng et al., 2025, Wang et al., 2025]. However, while these methods strengthen logical reasoning, they largely overlook the model’s ability to reflect on and correct errors in the visual space [Yang et al., 2025a]. Prior studies show that approximately 70% of errors in multimodal reasoning originate from visual perception [Wang et al., 2026b, Rezaei et al., 2026]. Once a model misinterprets visual information, it reasons based on incorrect premises, causing errors to propagate and compound throughout the reasoning process, ultimately leading to consistently incorrect conclusions. This further highlights that accurate visual understanding is a fundamental prerequisite for reliable multimodal reasoning.

Existing efforts attempt to mitigate this issue by enhancing visual grounding [Yu et al., 2025a], reweighting visual tokens [Lu et al., 2026, Wang et al., 2026a] and etc. While these approaches improve visual understanding to some extent, they do not explicitly equip models with the ability to perform self-correction in the visual space, and thus fail to fundamentally prevent error propagation. Moreover, these methods often require substantial modifications to the RLVR training pipeline, raising concerns about their scalability and generalization to broader settings.

To address these limitations, we propose MIRA (MIsleading-context Robustness Augmentation), a multimodal reasoning framework centered on constructing deceptive contextual augmentations. Rather than directly improving the model’s visual perception accuracy, our goal is to equip the model with the ability to recover from initial misinterpretations, enabling it to revise incorrect perceptions and still arrive at correct conclusions. Specifically, we simulate potential perception errors by injecting perturbed visual descriptions into the context, and introduce a group-wise enforced reflection mechanism that encourages the model to correct such errors during reasoning. Notably, our method enhances visual self-correction capabilities without requiring additional supervised fine-tuning (SFT) cold-start data.

Technically, MIRA first samples question-relevant visual descriptions using the policy model, and then employs an expert model to generate adversarial perturbations. A filtering step is applied to remove perturbations that do not affect the final answer, ensuring the effectiveness of the training signal. These perturbed descriptions are then injected into the context as if they were the model’s own previous outputs, thereby simulating realistic error scenarios. To prevent the model from ignoring contextual inconsistencies and to explicitly encourage corrective behavior, we introduce reflection triggers within sample groups, allowing the model to internalize the benefits of visual self-correction as part of its policy. Importantly, our method applies specialized training only to augmented data and does not require additional human annotations. By retaining a proportion of original data during training, we preserve the model’s generalization ability and mitigate potential overfitting or policy degeneration.

Extensive experiments demonstrate the effectiveness of MIRA. Our method improves the base model by approximately 7 percentage points on average across multiple mathematical benchmarks. Moreover, without introducing additional general-purpose visual data, MIRA achieves consistent gains on several multimodal benchmarks and outperforms a range of existing approaches. We further conduct comprehensive ablation studies to analyze the contributions of each component.

## 2 Related Works

Vision-Language Models Vision-Language Models (VLMs) have made significant progress in aligning visual and textual representations through large-scale pretraining and modular architectures [Liu et al., 2023, Bai et al., 2025b,a]. Existing approaches generally follow two paradigms: (1) end-to-end training on paired multimodal data, jointly optimizing all parameters [Wang et al., 2024b], and (2) freezing the language backbone while introducing lightweight modules (e.g., adapters or projection layers) to bridge modalities [Liu et al., 2023, Zhu et al., 2024, Li et al., 2023, Alayrac et al., 2022]. More recent works explore unified multimodal architectures to improve generality and scalability [Deng et al., 2025, Xin et al., 2025, Li et al., 2025]. Despite these advances, most methods primarily focus on cross-modal alignment and fusion, while overlooking the modeling of error propagation in visual perception. As a result, VLMs remain sensitive to initial perceptual errors, which can cascade through subsequent reasoning.

Multimodal Reinforcement Learning Reinforcement learning has emerged as an effective paradigm for enhancing reasoning abilities in large models without fine-grained supervision [Shao et al., 2024, Guo et al., 2025]. RLHF aligns models with human preferences [Ouyang et al., 2022], while outcome-based methods such as GRPO demonstrate that verifiable rewards alone can drive strong long-chain reasoning [Shao et al., 2024, Yu et al., 2025b, Zheng et al., 2025]. These ideas have been extended to multimodal settings via reinforcement learning with verifiable rewards (RLVR), significantly improving reasoning capabilities [Huang et al., 2026, Shen et al., 2025, Zhang et al., 2025, Xia et al., 2025, Li et al., 2026, Leng et al., 2025, Fan et al., 2026]. However, existing methods exhibit a clear limitation. Approaches focused on reasoning (e.g., Vision-R1 [Huang et al., 2026])

often neglect perception robustness, leading to visual hallucinations, while perception-enhanced RL methods [Wang et al., 2026a, Ouyang et al., 2025, Pan et al., 2026, Xiao et al., 2026, Liu et al., 2025, Zhu et al., 2026] improve visual perception but is still unable to deal with existing perceptual errors propagate through the reasoning process. Consequently, a trade-off emerges between perception and reasoning. In contrast, our method jointly improves both, while enabling visual-space self-reflection and error correction without supervised fine-tuning.

## 3 Preliminaries

Let $\pi _ { \theta }$ denote a policy VLM parameterized by θ. Given a multimodal input consisting of an image I and a textual query ${ \dot { Q } } .$ , the model generates an output sequence $y = ( y _ { 1 } , \dotsc , y _ { N } )$ . The conditional probability of generating y is defined as:

$$
\pi _ { \boldsymbol { \theta } } ( y \mid I , Q ) = \prod _ { t = 1 } ^ { N } \pi _ { \boldsymbol { \theta } } ( y _ { t } \mid I , Q , y _ { < t } ) ,\tag{1}
$$

where $\pi _ { \theta } ( y _ { t } \mid I , Q , y _ { < t } )$ denotes the probability of generating the t-th token conditioned on the input and previous tokens. In reinforcement learning, the model receives a scalar reward $r ( y , I , Q )$ based on the generated output y. The objective is to maximize the expected reward:

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( I , Q ) \sim \mathcal { D } } \left[ \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot | I , Q ) } \left[ r ( y , I , Q ) \right] \right] .\tag{2}
$$

Group Relative Policy Optimization (GRPO) samples a group of rollouts $y _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid I , Q )$ from the current policy, and refines the objective by normalizing rewards within each group. Specifically, rewards are standardized at the group level to compute relative advantages, which are then used to guide policy updates. Finally, the GRPO objective is defined as:

$$
\begin{array} { r l } { \mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { ( I , Q ) \sim p _ { D } , \ \{ y _ { i } \} _ { i = 1 } ^ { n } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | I , Q ) } \Bigg [ } & { } \\ { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathrm { m i n } \left( \frac { \pi _ { \theta } \left( y _ { i } \mid I , Q \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { i } \mid I , Q \right) } \hat { A } _ { i } , } & { \mathrm { c l i p } \left( \frac { \pi _ { \theta } \left( y _ { i } \mid I , Q \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { i } \mid I , Q \right) } , 1 - \epsilon , 1 + \epsilon \right) \hat { A } _ { i } \right) \Bigg ] } \end{array}\tag{3}
$$

where ϵ is the clipping range. The Kullback-Leibler divergence term between $\pi _ { \theta }$ and $\pi _ { \mathrm { o l d } }$ that penalizes large deviations from a reference policy $\pi _ { \mathrm { r e f } }$ is omitted for simplicity in Equation 3.

## 4 Method

Recent studies suggest that, in complex multimodal reasoning tasks, errors often stem from imperfect visual understanding rather than purely logical failures [Wang et al., 2026b, Rezaei et al., 2026]. Such inaccuracies in visual perception can propagate and negatively affect downstream reasoning.

Instead of enhancing visual perception by improving the vision encoder, our approach focuses on strengthening the interaction between visual and linguistic representations. Specifically, while reinforcement learning improves the model’s reasoning ability, we introduce a visual-space reflection mechanism that enables the model to revisit and refine its visual understanding during inference. This allows the model to remain stable in disturbed visual context, leading to more robust reasoning.

## 4.1 Construction of Deceptive Visual Context

We construct deceptive visual-context data to encourage the model to re-examine its visual understanding during generation. Specifically, we prepend to the model input a description of the image content disguised as a model-generated output. This description is intended to mislead the model’s normal visual information extraction and reasoning process.

Given an input pair $( I , Q )$ sampled from the publicly available dataset $\begin{array} { r } { \mathcal { D } _ { 0 } , } \end{array}$ , we first prompt the policy model $\pi _ { \theta }$ to generate a visual description summarizing the visual information relevant to the question:

$$
C _ { \mathrm { p o s } } \sim \pi _ { \theta } ( \cdot \mid I , Q , q _ { \mathrm { d e s } } ) ,\tag{4}
$$

![](images/09cd87a53197bd70e5219926cc23e51a3d4be52d6a2b551a416ef21bd9d12ace.jpg)  
Figure 1: Overview of MIRA. MIRA consists of two parts. The top illustrates the construction of deceptive visual context using the policy model and an expert model. The bottom shows the training process; compared to GRPO, the key changes are modified inputs and the use of intra-group triggers.

where $q _ { \mathrm { d e s } }$ denotes the corresponding system prompt. We intentionally use the same policy model, rather than a stronger external model, to reduce distribution mismatch and ensure that the generated description is naturally utilized during subsequent reasoning. $C _ { \mathrm { p o s } }$ is not guaranteed to be fully correct; the correction reward supervises only the expert-injected errors in M.

Next, an expert model Θ modifies $C _ { \mathrm { p o s } }$ into a plausible but incorrect description $C _ { \mathrm { n e g } } .$ , conditioned on the image, question, and ground-truth:

$$
( C _ { \mathrm { n e g } } , M ) = \Theta ( q _ { \mathrm { m o d i f y } } , I , Q , C _ { \mathrm { p o s } } ) ,\tag{5}
$$

where M records the applied modifications as pairs of before-and-after descriptions. This $C _ { \mathrm { n e g } }$ is consistent with $C _ { \mathrm { p o s } }$ in both structure and overall content, but differs in specific visual description details, and such differences affect the final answer to the question. Figure 1 illustrates a example.

To ensure the injected errors effectively influence the model, we retain only samples where a text-only verifier answers correctly from $C _ { \mathrm { p o s } }$ and incorrectly from $C _ { \mathrm { n e g } } .$ , without access to the image. The resulting dataset is:

$$
\mathcal { D } _ { \mathrm { a u g } } = \{ ( I , Q , C _ { \mathrm { n e g } } , M ) | \hat { y } _ { \mathrm { p o s } } = y ^ { * } , \hat { y } _ { \mathrm { n e g } } \neq y ^ { * } \} ,\tag{6}
$$

where $\hat { y } _ { \mathrm { n e g } }$ and $\hat { y } _ { \mathrm { p o s } }$ denote the verifier predictions, compared with ground truth $y ^ { * }$ , conditioned on $C _ { \mathrm { n e g } }$ and $C _ { \mathrm { p o s } } ^ { \mathrm { } }$ , respectively. We retain all original samples in $\mathcal { D } _ { 0 }$ and append their augmented versions in $\mathcal { D } _ { \mathrm { a u g } }$ , without introducing new questions:

$$
\mathcal { D } _ { \mathrm { M I R A } } = \mathcal { D } _ { \mathrm { a u g } } \cup \mathcal { D } _ { 0 }\tag{7}
$$

## 4.2 Intra-group Refinement Trigger

When deceptive visual context is introduced, the policy model, which has not been explicitly trained on such input patterns, may exhibit suboptimal behaviors. It may either over-rely on the context for reasoning or ignore it entirely by implicitly reconstructing a plausible caption from visual features, which weakens the intended perturbation effect.

Inspired by VL-Rethinker [Wang et al., 2025], we introduce a refinement trigger to encourage the model to revisit and correct its visual understanding. Specifically, we adopt a hybrid rollout strategy: for each group of N samples, we randomly select K samples and append a refinement trigger g at the end of $C _ { \mathrm { n e g } }$ . The trigger is sampled from a predefined trigger pool, with prompts such as “Wait, I may have misinterpreted the image.” The remaining $N - \bar { K }$ samples are kept unchanged. As GRPO-style RL algorithms derive their learning signal from relative rewards within the same group of rollouts, covering both behavior patterns, being prompted to re-examine the context and continuing naturally, under the same erroneous context enables the model to better learn their relative advantage and produces more stable reward differences. In our setup, we set $\begin{array} { r } { K = \frac { N } { 2 } } \end{array}$ . In contrast, when K is close to 0 or N, the behaviors within a group become more homogeneous, weakening the relative differences available for credit assignment.

This intra-group random injection encourages the model to learn a self-refinement behavior without supervised chain-of-thought data. Since triggers are only applied to samples with corrupted visual captions, the model avoids unnecessary reflection when no perceptual error is present. Through grouprelative optimization, this mechanism encourages the model to internalize visual-space reasoning and correction capabilities.

## 4.3 Policy Optimization and Reward Design

In our training framework, the deceptive visual context is not directly used as a user prompt. Instead, it is inserted as an initial assistant prefix, which conditions the subsequent generation process and encourages the model to self-correct during reasoning. The full input sequence can be written as:

## user:[image],[question] assistant:<think>[deceptive context],[trigger]

where <think> is the predefined format token which indicates the beginning of the reasoning phase. During reinforcement learning, this assistant prefix is only used during rollout to influence the generation trajectory. It is not treated as an action in the policy optimization objective. Including the prefix in the action space would introduce off-policy bias and violate the standard policy gradient formulation, since the prefix is externally constructed rather than sampled from the policy distribution.

We adopt GRPO as the optimization algorithm. The reward function is defined as:

$$
R _ { i } = \alpha r _ { \mathrm { f o r m a t } } + \beta r _ { \mathrm { a c c } } + \gamma r _ { \mathrm { c o r } } \times r _ { \mathrm { f o r m a t } } \times r _ { \mathrm { a c c } } ,\tag{8}
$$

where $r _ { \mathrm { f o r m a t } }$ enforces structured output in the form <think> · · · </think> <answer> · · · </answer>, and $r _ { \mathrm { a c c } }$ is a binary reward indicating whether the final answer matches the ground truth. The correction reward $r _ { \mathrm { c o r } }$ evaluates whether the model correctly identifies and corrects visual misperceptions introduced in $C _ { \mathrm { n e g } }$ . It is defined as:

$$
r _ { \mathrm { c o r } } ( X ) = \left\{ \begin{array} { l l } { 0 , } & { X \in \mathcal { D } _ { 0 } , } \\ { \frac { 1 } { | M | } \sum _ { \substack { ( d _ { \mathrm { b e f o r e } } , d _ { \mathrm { a f t e r } } ) \in M } } \mathbb { I } \Big ( \Theta ( X , d _ { \mathrm { b e f o r e } } , d _ { \mathrm { a f t e r } } ) = 1 \Big ) , } & { X \in \mathcal { D } _ { \mathrm { a u g } } , } \end{array} \right.\tag{9}
$$

where Θ is a judge model that verifies whether the output explicitly rejects the modified erroneous description $d _ { \mathrm { a f t e r } }$ and recovers the original visual evidence $d _ { \mathrm { b e f o r e } }$ . In Equation 8, the correction reward is applied only when the answer is correct, preventing reward hacking on subjective or hard-to-verify responses.

## 5 Experiments

Dataset We train our model on two multimodal reasoning datasets, ViRL-39K [Wang et al., 2025] and MMK12 [Meng et al., 2025]. ViRL-39K emphasizes visual information and logic-centric reasoning tasks, while MMK12 focuses on mathematical and compositional reasoning. We sample approximately 18K valid examples from these datasets to form the original dataset $\mathcal { D } _ { 0 }$ , with the detailed sampling procedure provided in Appendix C. After applying the data construction pipeline (Sec. 4.1), we retain the original samples and append their augmented versions, with a ratio of $| \bar { \mathcal { D } } _ { \mathrm { a u g } } |$ $| { \mathcal { D } } _ { 0 } | \approx 1 : 2 .$ . We evaluate our method on both reasoning-intensive and vision-centric benchmarks. The former includes MathVista [Lu et al., 2024], MathVerse [Zhang et al., 2024], MathVision [Wang et al., 2024a], and WeMath [Qiao et al., 2025], while the latter includes HallusionBench [Guan et al., 2024] (denote as “HalluBench”), and FREAK [Yin et al., 2026]. We also evaluate general multimodal reasoning on MMMU-Pro [Yue et al., 2025]. For MMMU-Pro, we report the accuracy on the 10-option subset and vision subset.

Table 1: Evaluation results of different reasoning methods. All results are reproduced using our evaluation scripts. Detailed evaluation settings are provided in Appendix D. Results of Vision-R1 on MathVision and WeMath are omitted due to potential training data contamination. Best (green) and second-best (blue) results among open-source multimodal reasoning methods are highlighted. Averages are computed over available benchmarks, excluding missing entries.
<table><tr><td rowspan="2"></td><td colspan="4">Math Benchmarks</td><td colspan="3">General / Visual Benchmarks</td><td rowspan="2">Avg.</td></tr><tr><td>MathVista</td><td>MathVerse</td><td>MathVision</td><td>WeMath</td><td>MMMUpro</td><td>HalluBench</td><td>FREAK</td></tr><tr><td colspan="8">General-purpose Models</td></tr><tr><td>GPT-40</td><td>62.6</td><td>50.4</td><td>31.6</td><td>67.6</td><td>53.2</td><td>73.5</td><td>33.4</td><td>53.2</td></tr><tr><td>InternVL2.5-8B</td><td>59.6</td><td>34.8</td><td>19.2</td><td>56.3</td><td>33.2</td><td>67.5</td><td>29.7</td><td>42.9</td></tr><tr><td>InternVL3-8B</td><td>71.6</td><td>47.7</td><td>29.5</td><td>62.9</td><td>41.9</td><td>67.5</td><td>29.7</td><td>50.1</td></tr><tr><td>Qwen2.5-VL-7B</td><td>68.2</td><td>46.7</td><td>24.3</td><td>61.4</td><td>38.5</td><td>65.7</td><td>30.0</td><td>47.8</td></tr><tr><td>Qwen2.5-VL-32B</td><td>76.1</td><td>55.6</td><td>36.8</td><td>70.9</td><td>41.2</td><td>72.3</td><td>26.7</td><td>54.2</td></tr><tr><td colspan="9">Multimodal Reasoning Methods</td></tr><tr><td>R1-Onevision-7B</td><td>60.7</td><td>40.9</td><td>20.5</td><td>57.9</td><td>33.7</td><td>67.0</td><td>22.7</td><td>43.3</td></tr><tr><td>Vision-R1-7B</td><td>73.2</td><td>46.7</td><td></td><td></td><td>36.1</td><td>67.9</td><td>24.1</td><td>49.6</td></tr><tr><td>MM-Eureka-7B</td><td>71.2</td><td>50.6</td><td>28.7</td><td>65.5</td><td>39.1</td><td>68.6</td><td>27.8</td><td>50.2</td></tr><tr><td>PAPOG-7B</td><td>73.6</td><td>51.2</td><td>27.7</td><td>66.3</td><td>41.1</td><td>68.7</td><td>31.1</td><td>51.4</td></tr><tr><td>Perception-R1-7B</td><td>71.9</td><td>50.9</td><td>27.5</td><td>70.7</td><td>40.7</td><td>67.5</td><td>24.8</td><td>50.6</td></tr><tr><td>SophiaVL-R1-7B</td><td>69.7</td><td>46.9</td><td>26.5</td><td>62.6</td><td>39.8</td><td>68.7</td><td>29.0</td><td>49.0</td></tr><tr><td>VL-Rethinker-7B</td><td>73.8</td><td>53.9</td><td>27.3</td><td>65.1</td><td>40.5</td><td>67.4</td><td>28.8</td><td>51.0</td></tr><tr><td>Ours</td><td>75.2</td><td>53.8</td><td>29.0</td><td>70.7</td><td>41.0</td><td>69.8</td><td>30.6</td><td>52.9</td></tr><tr><td>∆ vs Qwen2.5-VL-7B</td><td>+7.0</td><td>+7.1</td><td>+4.7</td><td>+9.3</td><td>+2.5</td><td>+4.1</td><td>+0.4</td><td>+5.0</td></tr></table>

Baselines We compare against a comprehensive set of recent multimodal RL baselines, including Vision-R1 [Huang et al., 2026], VL-Rethinker [Wang et al., 2025], SophiaVL-R1 [Fan et al., 2026], MM-Eureka [Meng et al., 2025], R1-OneVision [Yang et al., 2025b], Perception-R1 [Xiao et al., 2026], and PAPO [Wang et al., 2026b]. All methods are built upon the same backbone model, Qwen2.5-VL-7B [Bai et al., 2025b], and trained using GRPO-style algorithm to ensure fairness.

Implementation Details For fair comparison, we adopt Qwen2.5-VL-7B as the backbone model. All models are trained for 2 epochs with a learning rate of $1 \times 1 0 ^ { - 6 }$ . The rollout batch size is set to 128, and the maximum generation length is 2048 tokens. For the reward function defined in Eq. 8, we set the coefficients as $\stackrel { - } { \alpha } = 0 . 4 , \beta = \bar { 1 . } 0 , \gamma = 0 . 2$ . The expert model used for data construction is GPT-5.4. The intra-group triggers are defined as Figure 6. For more stable computation of accuracy reward, we use Qwen3-VL-32B as the judge model. During evaluation, we set the decoding temperature to 0.0 for all experiments.

## 5.1 Main Results

As shown in Table 1, we present the main evaluation results of MIRA. Under the same parameter scale, our method achieves leading performance across multiple multimodal benchmarks, including MathVista (75.2), MathVision (29.0), WeMath (70.7), as well as non-mathematical reasoning benchmarks such as MMMU-Pro (41.0) and HallusionBench (69.8). On math tasks, MIRA achieves the best average performance compared to all baselines. For vision-centric benchmarks, whose task distributions are partially out-of-domain relative to the training data, our method still yields consistent performance improvements, demonstrating strong generalization ability. Overall, under the same model scale, MIRA exhibits clear advantages in both reasoning and visual understanding. These results highlight the importance of jointly improving reasoning capability and visual robustness in multimodal models. Detailed comparison between MIRA and vanilla GRPO is provided in Sec. 5.3.

## 5.2 Generalization across Backbones

Table 2: Results on additional backbones (%). Each MIRA/GRPO pair uses the same source data and training configuration.
<table><tr><td>Backbone</td><td>Method</td><td>MathVista</td><td>MathVerse</td><td>MathVision</td><td>WeMath</td></tr><tr><td rowspan="2">Qwen2-VL-7B</td><td>GRPO</td><td>62.0</td><td>35.6</td><td>18.9</td><td>48.1</td></tr><tr><td>MIRA</td><td>63.5</td><td>36.5</td><td>19.1</td><td>49.5</td></tr><tr><td rowspan="2">Qwen3-VL-4B</td><td>GRPO</td><td>77.1</td><td>59.6</td><td>45.4</td><td>75.6</td></tr><tr><td>MIRA</td><td>77.8</td><td>61.6</td><td>47.0</td><td>78.7</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>GRPO</td><td>78.0</td><td>66.1</td><td>51.1</td><td>79.1</td></tr><tr><td>MIRA</td><td>79.9</td><td>66.1</td><td>52.9</td><td>81.2</td></tr></table>

Table 3: Comparison across benchmarks with different expert models and data ratio settings. Data ratio sensitivity experiment is conducted with 14K dataset. Refer Appendix D.3 for details.
<table><tr><td>Configuration</td><td>MathVista</td><td>MathVerse</td><td>MathVision</td><td>WeMath</td><td>Avg.</td></tr><tr><td colspan="6">Expert Model Ablation (Expert Type)</td></tr><tr><td>GPT 5.4</td><td>75.2</td><td>53.8</td><td>29.0</td><td>70.7</td><td>57.18</td></tr><tr><td>GPT 5.4-mini</td><td>74.3</td><td>52.5</td><td>29.7</td><td>69.6</td><td>56.53</td></tr><tr><td>GPT 5.1-mini</td><td>74.2</td><td>53.1</td><td>29.0</td><td>69.0</td><td>56.33</td></tr><tr><td colspan="6">Data Ratio Sensitivity (Augmentation Ratio)</td></tr><tr><td>0 (GRPO*)</td><td>71.5</td><td>51.0</td><td>28.4</td><td>68.0</td><td>54.7</td></tr><tr><td>0.3</td><td>72.5</td><td>52.2</td><td>28.5</td><td>67.9</td><td>55.3</td></tr><tr><td>0.5</td><td>73.5</td><td>53.2</td><td>28.2</td><td>68.1</td><td>55.8</td></tr><tr><td>0.7</td><td>71.5</td><td>51.3</td><td>28.9</td><td>68.0</td><td>54.9</td></tr></table>

We additionally evaluate MIRA on Qwen2-VL-7B and Qwen3-VL-4B/8B. The first two use the original data and protocol; Qwen3-VL-8B uses difficulty-filtered ViRL-39K and MMK12, with the same data and training configuration for MIRA and GRPO. Table 2 shows gains in 11 of 12 comparisons and a tie on Qwen3-VL-8B/MathVerse, supporting transfer across Qwen-VL generations and model sizes.

## 5.3 Ablation and Sensitivity Studies

Data Sensitivity We first conduct a sensitivity analysis on the construction of augmented data. Specifically, we employ GPT-5.4, GPT-5.4-mini, and GPT-5.1-mini as expert models to rewrite the visual descriptions. As shown in the upper part of Table 3, using different expert models for description rewriting leads to varying degrees of improvement in MIRA’s reasoning performance. In particular, stronger expert models yield better overall results, which may be attributed to their superior perception of the original images and their ability to produce higher-quality modified descriptions. In addition, we investigate model performance under different proportions of augmented data. The lower part of Table 3 presents the evaluation results across varying data ratios; details of the dataset construction are provided in Appendix D.3. Among the selected ratios, the best performance is achieved when the proportion of augmented data is set to 50%. In contrast, training without augmentation (i.e., vanilla GRPO) brings only limited improvements. Notably, when the proportion of augmented data becomes too high, performance degrades. We attribute this to the reduced diversity of training signals: although the model becomes better at handling misleading contexts, its intrinsic reasoning ability is not sufficiently improved. It is worth noting that reflection triggers are only introduced in the augmented data.

Reward and Intra-Group Trigger. We further conduct a comprehensive ablation study to analyze the effects of both intra-group triggers and reward design on model performance. Table 4 reports results under different configurations. We consider the following settings: (i) MIRA, the full model with both correction reward and K triggers in N rollouts; (ii) w/o correction reward, where the correction reward is removed while keeping the original trigger design; (iii) + full trigger w/o correction reward, where the correction reward is removed and triggers are applied to all samples within each group; (iv) w/o correction reward & w/o trigger, where both the correction reward and triggers are removed; and (v) GRPO, which further removes the augmented context on top of (iv), reducing the method to standard GRPO training. The results show that removing the correction reward already leads to noticeable performance degradation, indicating its role in encouraging effective error correction. As triggers are further simplified or removed, performance drops significantly, eventually approaching the level of vanilla GRPO. Without triggers, the model tends to ignore the injected deceptive context and instead rely on shortcut patterns to directly predict correct answers. Since reinforcement learning optimizes only the final answer reward, these shortcut-based solutions still receive high rewards. Consequently, such rollouts dominate the training signal, encouraging the model to disregard contextual information. This weakens its reliance on visual context and ultimately degrades its step-by-step reasoning ability.

Table 4: Ablation study on reward design and trigger mechanisms.
<table><tr><td>Setting</td><td>MathVista</td><td>MathVerse</td><td>WeMath</td><td>MathVision</td><td>Avg.</td></tr><tr><td>MIRA (full)</td><td>75.2</td><td>53.8</td><td>70.7</td><td>29.0</td><td>57.2</td></tr><tr><td>w/o correction reward</td><td>73.8</td><td>53.2</td><td>70.7</td><td>29.0</td><td>56.7</td></tr><tr><td>+ full trigger</td><td>73.6</td><td>52.6</td><td>68.8</td><td>28.4</td><td>55.8</td></tr><tr><td>- remove trigger</td><td>70.9</td><td>51.8</td><td>68.8</td><td>29.0</td><td>55.1</td></tr><tr><td>- remove context (GRPO)</td><td>72.9</td><td>51.5</td><td>68.8</td><td>29.3</td><td>55.6</td></tr></table>

Table 5: (a) Quantitative comparison of different context injection strategy. Training dataset is collected using GPT-5.4.
<table><tr><td>Benchmark</td><td>Assistant (Ours)</td><td>User</td></tr><tr><td>MathVista</td><td>73.8</td><td>70.7</td></tr><tr><td>MathVerse</td><td>53.2</td><td>52.9</td></tr><tr><td>WeMath</td><td>70.7</td><td>69.4</td></tr><tr><td>MathVision</td><td>29.0</td><td>28.8</td></tr><tr><td>MMMU</td><td>40.6</td><td>39.7</td></tr><tr><td>Average</td><td>53.5</td><td>52.3</td></tr></table>

Figure 2: (b) Evaluation dynamics on sampled 224 items from MathVista.  
![](images/be70032c9d8c09698162fc1b29c12434ef89fc3dfda5184bf90c30d2fb57092e.jpg)

MIRA Injection Strategy. The motivation of MIRA is to improve the model’s reasoning robustness against misleading contextual information. To this end, we disguise incorrect context information as if it were generated by the model itself. To investigate its effect, in this ablation study we inject the perturbed context into the user prompt (represent as “User”) in the following format: "[image]. The content shown in the image above is: [deceptive contex]. The image description might contain errors. Please answer the question strictly based on the visual content: [question]". The correction reward is removed in this experiment. The comparison of training accuracy on the MathVista subset between our method (represent as “Assistant”) and the variant without this injection strategy (“User”) is shown in Figure 2 and Table 5. It can be observed that directly injecting deceptive context into the user prompt yields limited gains, with only marginal improvements compared to injecting it into the assistant prompt. Under this setting, the model fails to effectively enhance its perceptual and reasoning robustness from the erroneous context, and its performance on MathVista is nearly on par with vanilla GRPO training. Notably, the user-prompt injection method does not employ triggers.

## 6 Analysis

In this section, we provide a detailed analysis and case study of our method. We investigate why MIRA improves model performance compared to vanilla RL algorithms, and examine how the model’s outputs evolve after being trained with MIRA.

## 6.1 Method Analysis

We analyze why MIRA achieves better performance than vanilla GRPO. From a methodological perspective, MIRA is designed to enhance the model’s reasoning capability by improving its robustness to hallucinated or misleading contextual information. Empirically, we observe that injecting incorrect context into originally easy samples significantly degrades model performance at the early stage of training. This enables such simple samples to provide non-trivial gradients and actively participate in model updates during MIRA training. Furthermore, according to the design of Eq. 8, for correctly answered samples, the correction reward further differentiates the rewards across different rollouts, thereby alleviating the issue where all responses within a group are correct and the gradient becomes zero. In contrast, under vanilla GRPO, these easy samples are typically answered correctly across all sampled responses, resulting in zero advantage for the entire group and thus zero gradients, which leads to slow or even stalled optimization. From the perspective of data augmentation, introducing augmented data increases the diversity of task types and implicitly raises the difficulty of the training data, mitigating the ineffectiveness of easy samples. As shown in the middle panel of Figure 3, the number of sample groups with zero advantage variance is significantly reduced under MIRA compared to vanilla GRPO. This indicates that augmented data provides more informative training signals, validating the effectiveness of MIRA.

![](images/859351a85b1b8c89b952e4e822fc2eefe18becc861e2e0d051a8c70efc473293.jpg)

![](images/8793f90eaa62a399bc667c891467330d5f87304b691f7e41a1733d6e07c38a8f.jpg)

![](images/8db69b23897e2a64901915b951e974d1e5d79394ad17d970262196de163aa615.jpg)  
Figure 3: Comparison between vanilla GRPO and MIRA. Left: training accuracy comparison; Middle: ratio of groups that the variance is 0; Right: accuracy comparison between normal samples and augmented samples. the reported result of MIRA is trained using GPT-5.4 as the expert model.

When training with MIRA, the dataset can be divided into two subsets: normal data and augmented data. The left panel of Figure 3 shows the accuracy dynamics on normal data under both vanilla GRPO and MIRA. As training progresses, MIRA gradually outperforms GRPO on normal data. For the augmented data, the model progressively learns to handle misleading context, and its accuracy eventually becomes comparable to that on normal data, as illustrated in the right panel of Figure 3.

## 6.2 Case Study

Figure 4 presents a training example along with the model’s output during training. In this sample, the input incorrectly describes the relationship between two resistors in the image, and this erroneous description is inserted at the beginning of the assistant’s response. No intra-group trigger is applied in this case. We observe that the model initially relies on the incorrect visual information for reasoning, but explicitly corrects these errors in the later stages of inference and arrives at the correct answer based on updated visual evidence.

This case demonstrates that, even when the context contains corrupted or misleading visual information, the model does not directly rely on the erroneous textual descriptions associated with the visual input. Instead, it progressively re-evaluates and corrects its visual perception during reasoning, and ultimately leverages the correct visual evidence to produce its answer. This case suggests that, through MIRA training, the model exhibits a degree of visual robustness in long-form reasoning, enabling it to rectify incorrect visual cues even in later stages of the reasoning process, rather than merely relying on a superficial early-stage correction pattern.

## 6.3 Quantitative Analysis of Reflection

We report reflection rate (RR), answer accuracy among reflections (SRR), and false correction rate (FCR) using a vision-capable LLM judge (Table 6). For n inputs, let $z _ { i } , a _ { i } ,$ and f indicate reflection, answer correctness, and false correction. The rates are $\begin{array} { r } { \mathrm { R R } = \dot { \sum } _ { i } z _ { i } / n , \mathrm { S R R } = \sum _ { i } z _ { i } a _ { i } / \sum _ { i } z _ { i } } \end{array}$ , and $\begin{array} { r } { \mathrm { F C R } = \sum _ { i } f _ { i } / n } \end{array}$ . Recall measures reflection on augmented inputs. Panel (a) samples augmented and normal groups separately, with SRR pooling both groups’ reflections.

![](images/0158ab16a9815a132eeac250b6730442638f9ee7f8e144641e3e72e88776a192.jpg)  
Figure 4: Training case of MIRA. Notice that in this case no triggers are inserted. The model autonomously corrects both erroneous context and reasoning errors. Incorrect and correct parts are highlighted in red and green, respectively.

Table 6: Quantitative reflection results on controlled and real inputs (%).
<table><tr><td colspan="2">(a) Controlled inputs</td><td>Recall</td><td>SRR</td><td>Aug. acc.</td><td>Normal acc.</td></tr><tr><td colspan="2">Base MIRA</td><td>27.42 42.96</td><td>36.33</td><td>19.48</td><td>43.51</td></tr><tr><td colspan="2">(b) Benchmark</td><td>Accuracy</td><td>59.54 RR</td><td>50.66</td><td>68.69 FCR</td></tr><tr><td rowspan="2">MathVista</td><td>Method GRPO</td><td>72.9</td><td>4.1</td><td>SRR</td><td></td></tr><tr><td>MIRA</td><td>75.2</td><td>5.4</td><td>85.4 66.7</td><td>0.30 (3/1000) 0.20 (2/1000)</td></tr><tr><td rowspan="2">MathVerse</td><td>GRPO</td><td>51.5</td><td>8.9</td><td></td><td></td></tr><tr><td>MIRA</td><td>53.8</td><td>15.8</td><td>57.8 69.8</td><td>0.30 (12/3940) 1.07 (42/3940)</td></tr></table>

On augmented data, MIRA increases recall by 15.54 percentage points and SRR by 23.21 points, while improving both groups’ accuracy. On MathVerse, RR increases from 8.9% to 15.8% and SRR from 57.8% to 69.8%; FCR also rises from 0.30% to 1.07%. On MathVista, accuracy improves from 72.9% to 75.2% despite lower SRR. These results complement the qualitative case, while showing that reflection can still introduce incorrect revisions.

## 7 Conclusion

In this paper, we introduced MIRA, a novel training framework designed to bridge the "perceptionreasoning gap" in multimodal large language models. By systematically constructing deceptive visual contexts and employing an intra-group reflection-triggering mechanism, we enable models to autonomously detect and rectify perceptual inconsistencies without relying on human-annotated chain-of-thought data. Experimental results across diverse benchmarks demonstrate that MIRA significantly enhances both logical reasoning and visual robustness, outperforming several recent reinforcement learning baselines. Furthermore, our analysis confirms that this self-correction capability is internalized as an intrinsic policy behavior, allowing the model to re-attend to visual evidence even in the absence of explicit triggers. While the current validation of MIRA is focused on a 7B-parameter scale due to computational constraints, we aim to explore the scaling properties of this self-correction mechanism on larger foundation models in future work. Ultimately, these findings underscore the necessity of active error recovery for reliable multimodal reasoning and provide a scalable path toward more resilient vision-language systems.

## Acknowledgments and Disclosure of Funding

The work was supported in part by Beijing Natural Science Foundation (L2608055) and National Natural Science Foundation of China (Grant No. 62576015). The authors sincerely thank the anonymous reviewers for offering invaluable suggestions.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning, 2022. URL https://arxiv.org/abs/2204.14198.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025b. URL https://arxiv.org/abs/2502.13923.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining, 2025. URL https://arxiv.org/abs/2505.14683.

Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. SophiaVL-r1: Reinforcing MLLMs reasoning with thinking reward. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=0tzvmjMcXC.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14375–14385, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu,

Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645 (8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

Wenxuan Huang, Bohan Jia, Shaosheng Cao, Zheyu Ye, Fei zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=UZIjskfbfU.

Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang, Hang Zhang, Xin Li, Lidong Bing, et al. Mmr1: Enhancing multimodal reasoning with variance-aware sampling and open resources. arXiv preprint arXiv:2509.21268, 2025.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. URL https://arxiv. org/abs/2301.12597.

Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Feize Wu, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, Shaodong Wang, Xinhua Cheng, and Li Yuan. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback, 2025. URL https://arxiv.org/abs/2510.16888.

Zongxia Li, Wenhao Yu, Chengsong Huang, Zhenwen Liang, Rui Liu, Fuxiao Liu, Jingxi Chen, Dian Yu, Jordan Lee Boyd-Graber, Haitao Mi, and Dong Yu. Vision-SR1: Self-rewarding vision-language model via reasoning decomposition and multi-reward policy optimization. In The Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=C1M4ETatgM.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. URL https://arxiv.org/abs/2304.08485.

Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. Noisyrollout: Reinforcing visual reasoning with data augmentation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=9zD2i7YRot.

Jinda Lu, Junkang Wu, Jinghan Li, Kexin Huang, Guoyin Wang, Jiancan Wu, Xiang Wang, and Xiangnan He. Bridging perception and reasoning: Token reweighting for RLVR in multimodal LLMs, 2026. URL https://openreview.net/forum?id=Q4KIDjDDRJ.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025. URL https://arxiv.org/abs/2503.07365.

Kun Ouyang, Yuanxin Liu, Linli Yao, Yishuo Cai, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Conan: Progressive learning to reason like a detective over multi-scale visual evidence, 2025. URL https://arxiv.org/abs/2510.20470.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155.

Zhiyu Pan, Yizheng Wu, Jiashen Hua, Junyi Feng, Shaotian Yan, Bing Deng, Zhiguo Cao, and Jieping Ye. Through the lens of contrast: Self-improving visual reasoning in VLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=ZymCPON45y.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma GongQue, Shanglin Lei, YiFan Zhang, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Xiao Zong, Yida Xu, Peiqing Yang, Zhimin Bao, Muxi Diao, Chen Li, and Honggang Zhang. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20023–20070, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.983. URL https://aclanthology.org/2025.acl-long.983/.

Ahmad Rezaei, Mohsen Gholami, Saeed Ranjbar Alvar, Kevin Cannons, Mohammad Asiful Hossain, Zhou Weimin, Shunbo Zhou, Yong Zhang, and Mohammad Akbari. Cppo: Contrastive perception for vision language policy optimization, 2026. URL https://arxiv.org/abs/2601.00501.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathemat ical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model, 2025. URL https://arxiv.org/abs/ 2504.07615.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024a. URL https://arxiv. org/abs/2402.14804.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024b. URL https://arxiv.org/abs/2409.18869.

Zengbin Wang, Feng Xiong, Liang Lin, Xuecai Hu, Yong Wang, Yanlin Wang, Man Zhang, and Xiangxiang Chu. Visually-guided policy optimization for multimodal reasoning, 2026a. URL https://arxiv.org/abs/2604.09349.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru WANG, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, and Heng Ji. Perception-aware policy optimization for multimodal reasoning. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=izbBqTL8vb.

Jiaer Xia, Yuhang Zang, Peng Gao, Sharon Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning, 2025. URL https://arxiv.org/abs/2505. 14677.

Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, and Enhong Chen. Perceptionr1: Advancing multimodal reasoning capabilities of MLLMs via visual perception reward. In The Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=KttCXdjj4w.

Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, Jinbin Bai, Qian Yu, Dengyang Jiang, Yuandong Pu, Haoxing Chen, Le Zhuo, Junjun He, Gen Luo, Tianbin Li, Ming Hu, Jin Ye, Shenglong Ye, Bo Zhang, Chang Xu, Wenhai Wang, Hongsheng Li, Guangtao Zhai, Tianfan Xue, Bin Fu, Xiaohong Liu, Yu Qiao, and Yihao Liu. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding, 2025. URL https://arxiv.org/abs/2510.06308.

Shuo Yang, Yuwei Niu, Yuyang Liu, Yang Ye, Bin Lin, and Li Yuan. Look-back: Implicit visual re-focusing in mllm reasoning, 2025a. URL https://arxiv.org/abs/2507.03019.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2376–2385, 2025b.

Zhihan Yin, Jianxin Liang, Yueqian Wang, Yifeng Yao, Huishuai Zhang, and Dongyan Zhao. FREAK: A fine-grained hallucination evaluation benchmark for advanced MLLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=YeagC09j2K.

En Yu, Kangheng Lin, Liang Zhao, jisheng yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, and Wenbing Tao. Perception-r1: Pioneering perception policy with reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://openreview.net/forum?id=BeXcXrXetA.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https://openreview.net/forum? id=2a36EMSSTp.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMUpro: A more robust multi-discipline multimodal understanding benchmark. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.736. URL https://aclanthology.org/ 2025.acl-long.736/.

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization, 2025. URL https://arxiv.org/abs/2503.12937.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?, 2024. URL https://arxiv.org/abs/2403.14624.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= 1tZbq88f27.

Linghao Zhu, Yiran Guan, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Bin Qin, Jian Luan, Yuliang Liu, and Xiang Bai. Shuffle-r1: Efficient RL framework for multimodal large language models via data-centric dynamic shuffle. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=mYP33u1QBK.

## A Overview of the Appendix

This appendix is organized as follows: Sec. B discusses the benchmarks and baselines used for evaluation. Sec. C presents details of data construction. Sec. D discusses implementation details, including training and evaluation procedures, and presents additional experimental results. Sec. E clarifies conditioning and optimization with external prefixes. Sec. G presents the prompts used during training and evaluation.

## B Benchmarks and Baselines

## B.1 Benchmarks

MathVista MathVista [Lu et al., 2024] is a unified benchmark for multimodal mathematical reasoning. We evaluate on its testmini split, which contains 1,000 problems across five subtasks: Textbook Question Answering, Visual Question Answering, Geometry Problem Solving, Math Word Problems, and Figure Question Answering. Compared to other math benchmarks, MathVista places greater emphasis on visual information extraction, and the logical difficulty of its problems is lower.

MathVerse MathVerse [Zhang et al., 2024] is a benchmark designed to evaluate the reasoning abilities of multimodal large language models (MLLMs) under varying proportions of textual and visual information. The testmini split of MathVerse contains 3,940 problems, including multiplechoice and fill-in-the-blank questions. It is further divided into five subsets based on the balance between text and visual content: Text Dominant (TD), Text Lite (TL), Vision Intensive (VI), Vision Dominant (VD), and Vision Only (VO). MathVerse covers three subtasks: Plane Geometry (2,550 problems), Functions (795 problems), and Solid Geometry (595 problems).

MathVision MathVision [Wang et al., 2024a] consists of 3,040 high-quality mathematical problems with visual contexts sourced from real mathematical competitions. In terms of mathematical reasoning difficulty, it is more challenging than both MathVista and MathVerse. The dataset includes five difficulty levels and 16 subtasks. In our evaluation, we do not differentiate between difficulty levels or subtasks.

WeMath WeMath [Qiao et al., 2025] is a benchmark specifically designed to explore problemsolving principles beyond end-to-end performance. It covers 67 hierarchical knowledge concepts across five levels of knowledge granularity. All questions in WeMath are multiple-choice, and its testmini split contains 1,740 problems in total. The dataset emphasizes the understanding and application of existing mathematical knowledge as well as visual perception.

MMMU-Pro MMMU-Pro [Yue et al., 2025] is a carefully filtered version of the multimodal benchmark MMMU. It improves upon MMMU in three aspects: (1) removing questions that can be answered solely by text-based models; (2) increasing the number of multiple-choice options, making random guessing more difficult; and (3) introducing a vision-only setting, where questions are embedded within images for evaluation. In our experiments, we use the Vision split and the 10-Options split of MMMU-Pro, each containing 1,730 questions. We compute accuracy on both splits separately and report the final MMMU-Pro overall result as the average accuracy across the two settings.

HallusionBench HallusionBench [Guan et al., 2024] is a widely used benchmark for evaluating multimodal hallucinations. It contains 346 images and 1,129 questions. The images are carefully edited by human experts to violate real-world or visual commonsense, thereby intentionally inducing hallucinations in multimodal models through conflicts with their internal textual knowledge.

FREAK FREAK [Yin et al., 2026] is a fine-grained hallucination evaluation benchmark designed for next-generation multimodal large language models. It includes 1,799 diverse images and corresponding questions. Compared to HallusionBench, FREAK constructs counterfactual images using an automated pipeline. These images are of higher resolution, focus on more complex real-life scenarios, and involve localized modifications to object attributes. Existing studies show that state-of-the-art multimodal models perform poorly on FREAK.

Table 7: Summary of baselines, including venue, data scale, and data source.
<table><tr><td>Baseline</td><td>Venue</td><td>Data Scale</td></tr><tr><td>R1-Onevision [Yang et al., 2025b]</td><td>ICCV&#x27;2025</td><td>R1-Onevision-155K</td></tr><tr><td>Vision-R1 [Huang et al., 2026]</td><td>ICLR&#x27;2026</td><td>200K SFT + 10K RL</td></tr><tr><td>MM-Eureka [Meng et al., 2025]</td><td></td><td>MMK12 (15.6K)</td></tr><tr><td>PAPO [Wang et al., 2026b]</td><td>ICLR&#x27;2026</td><td>ViRL-39K</td></tr><tr><td>Perception-R1 [Xiao et al., 2026]</td><td>ICLR&#x27;2026</td><td>Geometry3K</td></tr><tr><td>SophiaVL-R1 [Fan et al., 2026]</td><td>ICLR&#x27;2026</td><td>SophiaVL-130K</td></tr><tr><td>VL-Rethinker [Wang et al., 2025]</td><td>NeurIPS’2025</td><td>ViRL-39K</td></tr><tr><td>MIRA (Ours)</td><td></td><td>ViRL+MMK12 (18K)</td></tr></table>

## B.2 Baselines

To ensure a fair comparison with the baselines, Table 7 summarizes the basic information of the baselines considered in this paper, including their venue and publication year, as well as the scale and sources of their training data. According to the table, the data scale used by MIRA is comparable to or slightly smaller than that of the baselines, and all the data are sourced from existing open-source works, which helps ensure the fairness of the comparison to a certain extent.

## C Data Construction

The construction of visual error data is a core component of our method. During training, we use a mixture of original samples and preprocessed visual error data, both derived from existing datasets. We first introduce the datasets used, followed by a detailed description of the data processing pipeline.

We use ViRL-39K [Wang et al., 2025] and MMK12 [Meng et al., 2025]. We initially sample 20K instances, with 63% from ViRL-39K and 37% from MMK12, and remove instances with invalid images to obtain an approximately 18K-example source pool. Figure 5 shows its composition.

We then preprocess the data by introducing a sampling probability p. An augmented counterpart is constructed for each original sample with probability p, and is otherwise left without an augmented counterpart. For samples selected for augmentation, we first use a policy model (Qwen2.5-VL-7B in this work) to generate an initial question-relevant visual context, which is not guaranteed to be fully correct. Specifically, given an image and a question, the model is prompted to first perform visual understanding by extracting all information relevant to the question (i.e., generating a question-related caption), and then produce the final answer. We retain only the visual understanding component, denoted as $C _ { \mathrm { p o s } }$ , and discard the answer.

Next, we employ an expert model to transform $C _ { \mathrm { p o s } }$ into an incorrect version $C _ { \mathrm { n e g } }$ using a carefully designed instruction prompt (Fig. 7). The prompt guides the model to introduce targeted errors into key visual information while performing self-verification to ensure the effectiveness of the modifications. The expert model is also required to provide structured explanations, including explicit annotations of the original and modified text. We extract these modifications using regular expressions to construct a modification set M, where each entry records detailed before-and-after changes.

Due to model limitations, not all differences between $C _ { \mathrm { p o s } }$ and $C _ { \mathrm { n e g } }$ are relevant to solving the task. For example, if the question concerns region A while the modification occurs in an irrelevant region B, the injected error does not affect the answer. To eliminate such ineffective perturbations, we further validate $C _ { \mathrm { n e g } }$ . Specifically, we construct two inputs: (a) question $Q$ with correct context $C _ { \mathrm { p o s } } .$ and (b) question $Q$ with incorrect context $C _ { \mathrm { n e g } }$ . A sample is retained only if the verification model answers correctly in (a) and incorrectly in (b).

Importantly, we do not use the policy model as the verifier. Since the policy model is relatively weak during training, it may fail even when provided with correct context, leading to the removal of valid samples. As the purpose of this step is to identify truly misleading $C _ { \mathrm { n e g } } ,$ using a stronger model significantly improves filtering quality and data efficiency.

We retain the entire original pool $\mathcal { D } _ { 0 }$ and append the filtered augmented counterparts $\mathcal { D } _ { \mathrm { a u g } }$ . Thus $\mathcal { D } _ { \mathrm { M I R A } } = \mathcal { D } _ { 0 } \cup \mathcal { D } _ { \mathrm { a u g } } .$ , with an augmented-to-original instance ratio of approximately $1 : { \bar { 2 } } \quad$ . Augmented instances reuse source image–question pairs and do not add new questions. The separate 14K ratio study below uses a reconstructed pool and should not be confused with this main mixture.

![](images/b795c9fef087bfce095d017a359fea4b3f214ce61ff8bf9736872115dc890b02.jpg)  
Figure 5: The statistics of training data.

Although $C _ { \mathrm { p o s } }$ is not intentionally corrupted, it may contain policy perception errors. Answer-based filtering selects task-relevant perturbations but does not exhaustively validate the description. The correction reward evaluates only the explicitly recorded local errors; unannotated background error are not directly scored by this term.

## D Implementation Details

## D.1 Training Details

Table 8: Training hyperparameters we used for main results.
<table><tr><td colspan="2">Training Configuration</td><td colspan="2">RL Configuration</td></tr><tr><td>Training epochs</td><td>2</td><td>Rollout number</td><td>8</td></tr><tr><td>Gradient accumulation steps</td><td>4</td><td>KL Coefficient</td><td>0.004</td></tr><tr><td>Effective batch size</td><td>128</td><td>Rollout Engine</td><td>vLLM</td></tr><tr><td>Sequence length cutoff</td><td>5,000</td><td>Temperature</td><td>1.0</td></tr><tr><td>Optimizer</td><td>AdamW</td><td colspan="2">Trigger Configuration</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Rollouts with trigger K</td><td>4</td></tr><tr><td>Max gradient norm</td><td>1.0</td><td>Rollouts per group N</td><td>8</td></tr><tr><td>Precision</td><td>bfloat16</td><td>Augmented:clean</td><td>≈1:2</td></tr></table>

We adopt veRL as the reinforcement learning framework. During GRPO training, rule-based verification is first performed via regular expression matching; if this step fails, a judge model is used for further evaluation. In our implementation, we employ Qwen3-VL-32B as the judge model. Notably, both the policy model and the judge model are co-deployed on the same GPUs, without requiring additional dedicated GPU devices. The detailed hyperparameter settings for reinforcement learning are provided in Table 8. Due to disk I/O bottlenecks during training, the entire training process takes approximately 16 hours.

## D.2 Evaluation Details

We evaluate model performance using a custom-built evaluation pipeline. Across all benchmarks, we employ a large language model (GPT-4o) as the judge model. Specifically, we first extract the final answer from model outputs using regular expressions; when extraction fails or further verification is needed, the judge model is used to determine whether the predicted answer matches the ground truth. Due to the inherent stochasticity of API-based outputs, this evaluation process introduces minor randomness. This evaluation variability is distinct from variation across independent training runs.

To ensure fairness in evaluation, the system prompts used for different benchmarks are not derived from those used during training. Instead, we adopt a unified system prompt (shown in Fig. 9) across all selected benchmarks. As this prompt is not tailored to any specific baseline, it provides a fair and consistent evaluation setting.

## D.3 More Experiment Results

In Table 3, the ablation study on expert models uses the data described in Appendix D. However, for the data ratio sensitivity analysis, we reconstruct the dataset based on the above data. Specifically, we first merge the augmented data generated by three expert models, namely GPT-5.4, GPT-5.4-mini, and GPT-5.1-mini. For each sample, we select its augmented version following the priority order GPT-5.4 → GPT-5.4-mini → GPT-5.1-mini. If a sample is a normal (non-augmented) instance in all three datasets, it is discarded. This process results in a fully augmented dataset of 14K samples (i.e., 100% augmented). We then perform random degradation by converting a portion of the augmented samples back to normal samples, thereby constructing datasets with augmented data ratios of 0.7, 0.5, 0.3, and 0. So, in this table, GRPO<sup>∗</sup> denotes training on the 14K dataset, which differs from the data used for vanilla GRPO in the other tables. The verified four-benchmark expert comparison is reported in Table 3.

## E Optimization and Implementation Clarification

This section clarifies external-context conditioning rather than claiming a new theoretical result. A corrupted description is generated offline and fixed during RL training. A trigger is also supplied externally, with a selection rule independent of policy parameters. Neither is a policy-sampled action.

Let $x = ( I , Q ) , c = C _ { \mathrm { n e g } } ,$ and $g _ { i }$ be the supplied trigger for rollout i (empty for untriggered rollouts). The state is $s _ { i } = ( x , c , g _ { i } )$ , and the sampled continuation follows

$$
\pi _ { \boldsymbol { \theta } } ( y _ { i } \mid s _ { i } ) = \prod _ { t = 1 } ^ { T _ { i } } \pi _ { \boldsymbol { \theta } } ( y _ { i , t } \mid s _ { i } , y _ { i , < t } ) .\tag{10}
$$

The policy and behavior-policy likelihoods must condition on the same supplied prefix. At a generated token, the corresponding ratio is

$$
\rho _ { i , t } = \frac { \pi _ { \theta } ( y _ { i , t } \mid s _ { i } , y _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid s _ { i } , y _ { i , < t } ) } .\tag{11}
$$

The loss mask excludes the image/question prompt, injected description, and externally supplied trigger; generated continuation tokens contribute to the clipped GRPO objective. Treating fixed prefix tokens as sampled actions would incorrectly attach a policy likelihood to externally provided text. Masking those tokens does not prevent the continuation loss from depending on the prefix through its contextual representations.

This observation concerns the likelihood accounting for the injected context. It does not establish that a clipped GRPO surrogate with group-normalized advantages is an unbiased estimator of an unclipped expected-reward gradient. It also does not eliminate the distribution gap between synthetic descriptions and naturally occurring perception errors. MIRA targets robustness to controlled misleading contexts; the unaugmented benchmark analysis assesses behavioral transfer separately.

## F Additional Behavioral Analyses

## F.1 Trigger Wording Sensitivity

We evaluate the trained Qwen2.5-VL-7B policy on 500 additional augmented examples with no trigger and with each of seven English triggers. The trigger pool also contains Chinese templates to match the training data’s language distribution. Table 9 reports the mean and standard deviation across English phrasings; these are not independent training runs. Final-answer accuracy varies less across phrasings than reflection frequency. This experiment tests prompted behavior at evaluation time, whereas standard benchmark evaluation uses no trigger.

Table 9: Trigger wording sensitivity on 500 augmented examples (%). Variability is across seven English phrasings.
<table><tr><td>Metric</td><td>No trigger</td><td>Seven triggers (mean ± std.)</td></tr><tr><td>Reflection rate</td><td>42.2</td><td> $9 0 . 3 6 \pm 3 . 2 2$ </td></tr><tr><td>SRR</td><td>39.8</td><td> ${ \bf 5 6 . 6 4 \pm 1 . 0 7 }$ </td></tr><tr><td>Accuracy</td><td>42.2</td><td> ${ \bf 5 4 . 7 3 \pm 1 . 1 8 }$ </td></tr></table>

## F.2 Reflection Metrics and Interpretation

Sampling and generation. The evaluator reads two groups from a parquet file: type 1 supplies an erroneous description as an unfinished assistant prefix, while type 0 ends with a user prompt and contains no injected prefix. It samples each group separately using a fixed random seed, stores a sample manifest, and evaluates every checkpoint on exactly the same inputs. The two groups are not assumed to be question-paired. The groups are deterministically shuffled together before generation. The supplied evaluator uses greedy decoding (temperature 0, top-k = 1); the controlled comparison calls generation without an added reflection trigger.

Vision-capable LLM judge. For each response, the judge receives the image, question, reference answer, and model continuation. For augmented inputs it also receives the injected description and each wrong-claim/correct-evidence pair. It labels explicit reflection, the number of corrected modifications, whether all modifications are corrected, and final-answer correctness independently, with a supporting quote and explanation. Generic phrases such as “let me check,” routine step-by-step reasoning, or a correct answer without an expressed revision do not qualify as reflection. For normal inputs, reflection requires reconsideration of an earlier belief, calculation, or visual interpretation in the response itself.

Aggregation. Reflection recall is the fraction of valid augmented responses with reflection. The answer-based SRR pools all reflected responses from both controlled groups. Augmented and normal accuracies are computed separately. A stricter successful-correction label additionally requires explicit correction of every injected modification and a correct final answer; it is not the SRR column. Judge failures are excluded from valid-response denominators. On normal inputs, the strict injectedmodification correction label is inapplicable and set to false. Reflection on normal inputs may be legitimate self-correction: counting it as a false positive for injected-error detection does not make it a false correction.

For the unaugmented benchmark analysis, FCR divides erroneous-revision counts by the full bench mark size. This is distinct from the controlled evaluator’s detection false positives. Counts are retained in Table 6 to make the normalization explicit. The conditional statistics do not identify a causal effect of reflection, since the models may reflect on different subsets.

## F.3 User-Side Injection Prompt

The user-side ablation uses: “[image]. The content shown in the image above is: [deceptive context]. The image description might contain errors. Please answer the question strictly based on the visual content: [question].” This setting omits reflection triggers and the correction reward.

## F.4 Failure Modes

![](images/44aa40326d9810dcc84e4d7947c069046028c6a570bb376591567d35dd21da37.jpg)  
Figure 6: The pre-defined triggers set of intra-group trigger.

![](images/de40b51fa3c4dc0297fd6be8fe90122ece2b8c5da77f16b9c6902ba7294d5934.jpg)  
Figure 7: The prompt we used for expert modification in data construction.

![](images/eed697d0ed4c34e63ea4addd7ba6587beaef046213138f6de90382cf2345f102.jpg)  
Figure 8: The system prompt we used for training.

![](images/e7ac578364818f46416d7f9f6e3ff3613baa4cb1a6018b0663ddd2e65b0224cd.jpg)  
Figure 9: The system prompt we used for evaluation.