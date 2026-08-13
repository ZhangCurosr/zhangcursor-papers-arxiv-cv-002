# SCOUT: Unlocking Enhanced Spatial Reasoning via Structured Chain-of-Thought and Multi-Objective Process Reward

Zile Zhou, Huining Yuan, Weichen Zhang, Xinlei Chen, and Xiao-ping Zhang

Shenzhen International Graduate School, Tsinghua University, Shenzhen, China

Abstract. Existing Vision-Language Models (VLMs) exhibits a critical bottleneck in robust spatial reasoning. Recent reinforcement learning (RL) methods aim to close this gap with verifiable outcomes, yet they sufer from poor credit assignment across intermediate reasoning steps. Concurrently, structured reasoning approaches overlook the critical depth perception necessary for comprehensive 3D understanding. To address these challenges, we propose SCOUT (Structured Chain-Of-Thought Utilizing Process-Supervised RL Training). Specifically, we design a structured Chain-of-Thought (CoT) framework that explicitly models 3D environmental perception to ensure robust spatial understanding and reasoning. Furthermore, we introduce a novel RL algorithm featuring multi-objective process rewards and a tailored advantage estimation method, facilitating fine-grained credit assignment across distinct segments of the reasoning trajectory. To support our framework, we develop SCOUT-24k, a structured spatial reasoning CoT dataset synthesized through a customized pipeline. Extensive evaluations demonstrate that SCOUT-3B improves upon baseline models by 16.85% and 6.3% on general spatial benchmarks and complex spatial reasoning tasks respectively. Notably, our larger SCOUT-7B even outperforms GPT-4o by a margin of 4.28%. Moreover, despite being trained exclusively on single image, SCOUT-7B exhibits robust out-of-domain generalization to multi-image and video scenarios. These empirical results render SCOUT as a critical step towards next generation of spatially-aware VLMs.

Keywords: Vision-Language Models · Spatial Reasoning · Reinforcement Learning

## 1 Introduction

In recent years, Vision-Language Models (VLMs) have achieved remarkable success in fundamental visual tasks [3,10,23]. Despite this progress, in many critical downstream applications that require advanced spatial reasoning, such as robot navigation [29, 51], robot manipulation [13, 21, 56], autonomous driving [43, 45], and virtual reality systems [6,50], current models still sufer from significant performance gaps. Enhancing the spatial reasoning capabilities of VLMs represents a critical frontier for real-world artificial intelligence applications [55].

![](images/8a048c05b0e4877097e5c07f422128ffa391df8f4421dd833a04a7f825173169.jpg)  
Fig. 1: We present SCOUT, a novel spatial reasoning framework for VLMs, equipped with a structured CoT optimized via process-supervised RL. (a) Qualitative example: the baseline produces an incorrect answer due to a lack of spatial awareness, whereas our model generates accurate prediction by explicitly analyzing object depth and scene information. (b) Quantitative evaluations on multiple spatial reasoning benchmarks demonstrate the superiority of SCOUT-3B over Qwen2.5-VL-3B and existing spatial-specialized methods.

To enhance the spatial reasoning capabilities of VLMs, early data-driven approaches based on Supervised Fine-Tuning (SFT) typically rely on post-training on massive synthetic datasets with auxiliary spatial representations [5, 7, 8, 37]. However, these works are highly data-intensive, requiring substantial efort for data curation. Moreover, training via SFT tends to induce strict rote memorization, which ultimately restricts the models’ generalization capabilities [9].

To develop robust and generalizable spatial reasoning capabilities, recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) are introduced to the spatial reasoning domain [26, 36, 39, 47, 53]. These works leverage Chain-of-Thought (CoT) prompting alongside verifiable outcome rewards to improve reasoning performance. However, relying exclusively on sparse outcome rewards hinders fine-grained advantage estimation, leading to inaccurate credit assignment for intermediate reasoning steps. Concurrently, another line of research proposes structured CoT templates, aiming to improve perception in human-like cognitive process [34] and enhance the interpretability of the reasoning trajectories [4, 20]. Nevertheless, these studies critically overlook depth-related information within their templates, fundamentally restricting the comprehension of 3D spaces. Furthermore, they also sufer from poor credit assignment, failing to reward specific reasoning behaviors in the structured information.

To address these critical gaps, we propose a structured reasoning CoT framework that explicitly models 3D spatial perception, ensuring robust environmental understanding. As illustrated in Fig. 1 (a), during inference, the model first extracts 3D information from the given image, followed by logical deduction grounded in the perceived spatial observations. This reasoning approach ofers three primary advantages: First, it significantly improves the interpretability of the reasoning process, facilitating precise information extraction. Second, explicit modeling of spatial attributes ensures that the acquired reasoning capabilities are fundamentally grounded in physical reality. More importantly, by partitioning the CoT into independently extractable perception and analysis modules, it unlocks the potential for fine-grained, targeted verification and accurate credit assignment for each reasoning step during RL training.

![](images/55f65670c85ea335a3e711fb0d6f68e4d271d4deb5ace6949160b004dac43622.jpg)  
Fig. 2: Method overview of SCOUT. Stage 1 initializes the training through a SFT cold-start using our curated CoT data. Stage 2 employs reinforcement learning with multi-objective rewards and fine-grained advantage estimation to efectively assign credit across diferent tokens and update the policy model.

Building on the structured CoT, we introduce a novel reinforcement learning algorithm that integrates multi-objective process rewards coupled with an advanced advantage estimation method. Specifically, we design dedicated reward functions for object’s bounding box(bbox) and depth perception, providing precise supervisory signals to optimize the model’s 3D spatial understanding capabilities. Furthermore, we propose a reasoning consistency reward that evaluates the alignment between the reasoning trajectory and the final answer, efectively guiding the model to generate coherent deductions. Crucially, the proposed advantage estimation algorithm integrates these process rewards to facilitate finegrained credit assignment across diferent segments of the CoT trace. By explicitly attributing the contribution of each reasoning module to the final outcome, our RL algorithm fundamentally fosters advanced spatial reasoning in VLMs.

To support this framework, we develop a scalable data synthesis pipeline and introduce SCOUT-24k, a comprehensive structured CoT dataset. This dataset encompasses diverse spatial reasoning tasks, spanning the entire spectrum from fundamental relationship comprehension and relative distance prediction to complex perspective transformations.

Leveraging the proposed methodology and the SCOUT-24k dataset, we present SCOUT-3B and SCOUT-7B, initialized from Qwen2.5-VL-3B and Qwen2.5-VL-7B [3], respectively. As illustrated in Fig. 1(b), SCOUT-3B achieves substantial gains over the baseline models, yielding improvements of 16.85% on general spatial benchmarks and 6.3% on complex spatial reasoning tasks. Consistent trends are observed upon scaling the model to 7B. Notably, SCOUT-7B further achieves an average accuracy improvement of 4.28% and 0.87% over GPT-4o [23] on two types of spatial reasoning benchmarks, respectively. More importantly, despite being trained exclusively on single-image, SCOUT extends its spatial reasoning proficiency to multi-image and video domains. This is evidenced by performance gains of 2.46% on ViewSpatial [25] dataset and 3.13% on the multiple-choice section of VSI-Bench [48]. Collectively, these results establish SCOUT as a crucial step toward the next generation of spatially grounded VLMs.

Our main contributions can be summarized as follows:

1. We propose a structured reasoning CoT framework that explicitly models the perception of 3D spatial environments, ensuring comprehensive spatial understanding and robust reasoning.

2. We introduce a novel RL algorithm integrating multi-objective process rewards with advanced advantage estimation, facilitating fine-grained credit assignment across diferent segments of the reasoning trajectory.

3. We develop a scalable data synthesis pipeline and construct the SCOUT-24k dataset, systematically spanning spatial reasoning tasks from fundamental relationship comprehension to complex perspective transformations.

4. Extensive experiments demonstrate that SCOUT significantly outperforms baselines and GPT-4o in both general spatial benchmarks and complex spatial reasoning tasks, while exhibiting robust generalization to multi-image and video domain, validating the eficacy of our approach.

## 2 Method

In this section, we present the algorithmic and data construction pipeline of SCOUT, as illustrated in Fig. 2 and Fig. 3. To equip VLMs with advanced spatial reasoning capabilities, our methodology integrates two essential components: a structured CoT template to facilitate robust 3D understanding and a novel RL algorithm that efectively incentivizes spatial reasoning within the structured CoT. For data curation, we create a scalable data synthesis pipeline, generating a comprehensive dataset of spatial reasoning tasks.

## 2.1 Structured Reasoning Process

We propose a novel structured CoT framework to enhance accurate physical perception of 3D environments in complex spatial reasoning tasks. The entire reasoning trajectory is encapsulated within a <think> block, which strictly guides the model through a progressive, modular cognitive pipeline before yielding the final answer. We initiates this reasoning process with the <caption> module, where the model generates a global semantic description of the visual input. Following this, the model transitions to the <scene> module, which quantitatively describe key objects with their bounding boxes and depths. Based on this finegrained 3D spatial perception, the model then enters the <analyze> phase to perform explicit logical reasoning and deduction, seamlessly executing numerical comparisons and relative reasoning based on the extracted depth values. After completing comprehensive visual extraction and rigorous logical formulation, the model exits the </think> wrapper and outputs the final conclusion strictly within the <answer> tag. In summary, our proposed reasoning template adopts

the following sequential format: <think> <caption>...</caption> <scene> ...   
</scene> <analyze> ... </analyze> </think> <answer> ... </answer>.

As illustrated in Fig. 2, we perform a preliminary SFT cold-start to guide the model for structured reasoning formats, establishing a stable foundation for efective policy optimization in the subsequent RL phase.

## 2.2 Multi-Objective Process Rewards

To facilitate efective RL training, we design multiple process rewards for the verification of intermediate reasoning steps within our proposed structured CoT. Concretely, this involves 5 distinct rewards for a multi-objective process supervision of the reasoning process:

Regularized Grounding Reward. To evaluate model’s perception of environment, we first align the predicted objects $\mathcal { O } ^ { \mathrm { p r e d } }$ with the ground-truth objects $\mathcal { O } ^ { \mathrm { g t } }$ via the Hungarian algorithm [24]. The matching cost $\mathcal { C } _ { i , j }$ between the i-th prediction and the j-th ground truth is:

$$
\mathcal { C } _ { i , j } = \lambda _ { \mathrm { s e m } } ( 1 - \sin ( l _ { i } , l _ { j } ) ) + \lambda _ { \mathrm { i o u } } ( 1 - \mathrm { E I o U } ( b _ { i } , b _ { j } ) ) + \lambda _ { \mathrm { d e p } } ( 1 - \delta ( d _ { i } , d _ { j } ) ) ,\tag{1}
$$

where $l , b ,$ and d denote the semantic label, bounding box, and depth, respectively. Here sim(·) is the semantic similarity, EIoU(·) is the Eficient IoU [54], and $\begin{array} { r } { \delta ( d _ { i } , d _ { j } ) = \exp \left( - 2 \frac { | d _ { i } - d _ { j } | } { d _ { j } } \right) } \end{array}$ measures depth consistency. We prioritize geometric localization by setting the coeficients to $\lambda _ { \mathrm { s e m } } = 2 . 0 , \lambda _ { \mathrm { i o u } } = 3 . 0 $ , and $\lambda _ { \mathrm { d e p } } = 0 . 5$

To mitigate reward hacking (e.g., bbox proliferation) seen in prior work [4], we introduce a cardinality penalty. The final regularized grounding reward is:

$$
r _ { \mathrm { g r o u n d i n g } } = \underbrace { \frac { 1 } { | \mathcal { M } | } \sum _ { ( i , j ) \in \mathcal { M } } \mathrm { E I o U } ( b _ { i } , b _ { j } ) } _ { \mathrm { Q u a l i t y ~ T e r m } } - \underbrace { \eta \cdot \operatorname* { m a x } ( 0 , | \mathcal { O } ^ { \mathrm { p r e d } } | - | \mathcal { O } ^ { \mathrm { g t } } | ) } _ { \mathrm { O v e r - g e n e r a t i o n ~ P e n a l t y } } ,\tag{2}
$$

where M is the set of matched pairs and the penalty coeficient η is set to 0.2.

Depth Reward. Based on the matched pairs M obtained above, we impose a continuous Depth reward to penalize deviations in the z-axis:

$$
r _ { \mathrm { d e p t h } } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( i , j ) \in \mathcal { M } } \delta ( d _ { i } , d _ { j } ) .\tag{3}
$$

This reward incentivizes the model’s spatial understanding to 3D environment.

Reasoning Consistency Reward. To prevent models from arriving at correct answers through flawed or disconnected reasoning, we employ a blind verification mechanism. We feed only the textual question and the generated reasoning chain into base model, omitting all visual inputs. The reward $r _ { \mathrm { c o n s i s t e n c y } }$ is 1 if the verifier derives the ground truth solely from this textual input, and 0 otherwise. This ensures the reasoning path logically entails the final prediction.

Accuracy Reward. This is the standard outcome-based reward. Given the multiple-choice format, we define a binary reward $r _ { \mathrm { a c c u r a c y } } ~ \in ~ \{ 0 , 1 \}$ , where $r _ { \mathrm { a c c u r a c y } } = 1$ if the prediction matches the ground-truth answer, and 0 otherwise.

Format Reward. Finally, to encourage format following, we assign a binary reward $r _ { \mathrm { f o r m a t } } = 1 \ \mathrm { i f }$ the model correctly encloses its reasoning process, spatial perception, and final output within the <think>, <scene>, and <answer> tags, respectively; otherwise, $r _ { \mathrm { f o r m a t } } = 0$

## 2.3 Advantage Estimation for Fine-grained Credit Assignment

Standard Group Relative Policy Optimization (GRPO) [40] assigns a single reward and advantage to the entire response sequence, sufering from coarse credit assignment across diferent reasoning steps. To address this, we propose a finegrained Advantage Estimation mechanism based on our multi-objective rewards. Leveraging the explicit tags of our structured CoT, we assign distinct credit to three functional segments: Perception (from <think> to </scene>), Analysis (from <analyze> to </think>), and Answer (from <answer> to </answer>.)

Advantage Aggregation and Mixing. For a group of generated outputs, we first apply standard z-score normalization to each raw reward $r _ { k }$ to yield $\tilde { r } _ { k }$ mitigating scale discrepancies between heterogeneous rewards:

$$
\tilde { r } _ { k } ^ { ( i ) } = \frac { r _ { k } ^ { ( i ) } - \mu _ { k } } { \sigma _ { k } } ,\tag{4}
$$

where $r _ { k } ^ { ( i ) }$ represents the k-th reward term for the i-th sample, and $\mu _ { k }$ and $\sigma _ { k }$ denote the group mean and standard deviation, respectively. To align with the distinct functions of each CoT stage, we aggregate these normalized rewards into stage-specific base advantages:

$$
\begin{array} { r l } & { \mathcal { A } _ { \mathrm { s c e n e } } = \tilde { r } _ { \mathrm { g r o u n d i n g } } + \tilde { r } _ { \mathrm { d e p t h } } , } \\ & { \mathcal { A } _ { \mathrm { a n a l y z e } } = \tilde { r } _ { \mathrm { c o n s i s t e n c y } } , } \\ & { \mathcal { A } _ { \mathrm { o u t c o m e } } = \tilde { r } _ { \mathrm { f o r m a t } } + \tilde { r } _ { \mathrm { a c c } } . } \end{array}\tag{5}
$$

A strictly decoupled reward might incentivize models to optimize intermediate steps while ignoring the correctness of final answer. Therefore, we blend local process supervision with the global outcome advantage:

$$
\begin{array} { r } { \hat { A } _ { \mathrm { s c e n e } } = \alpha _ { 1 } A _ { \mathrm { s c e n e } } + ( 1 - \alpha _ { 1 } ) A _ { \mathrm { o u t c o m e } } , } \\ { \hat { A } _ { \mathrm { a n a l y z e } } = \alpha _ { 2 } A _ { \mathrm { a n a l y z e } } + ( 1 - \alpha _ { 2 } ) A _ { \mathrm { o u t c o m e } } . } \end{array}\tag{6}
$$

Here, parameters $\alpha _ { 1 } , \alpha _ { 2 } \in [ 0 , 1 ]$ control the trade-of between local process alignment and global task success. We set $\alpha _ { 1 } = 0 . 3$ and $\alpha _ { 2 } = 0 . 3$ to ensure intermediate verification is strongly anchored by the final prediction accuracy.

![](images/f6c5c306f57bbce75d1c51aef0fef8b076697bda3809e1da1c1cc24e3c172f0b.jpg)  
Fig. 3: Overview of SCOUT-24k dataset construction pipeline from raw images to final spatial reasoning data generation with representative Relative Distance task.

Token-Level Credit Assignment and Optimization. Finally, we allocate these advantages to specific tokens based on their functional segments. Let $T _ { \mathrm { s c e n e } }$ and $T _ { \mathrm { t h i n k } }$ denote the indices of the $< /$ scene> and </think> tags, respectively. The token-wise advantage $A _ { t }$ is defined as:

$$
A _ { t } = \left\{ \begin{array} { l l } { \hat { A } _ { \mathrm { s c e n e } } } & { \mathrm { i f ~ } t \leq T _ { \mathrm { s c e n e } } , } \\ { \hat { A } _ { \mathrm { a n a l y z e } } } & { \mathrm { i f ~ } T _ { \mathrm { s c e n e } } < t \leq T _ { \mathrm { t h i n k } } , } \\ { \mathcal { A } _ { \mathrm { o u t c o m e } } } & { \mathrm { i f ~ } t > T _ { \mathrm { t h i n k } } . } \end{array} \right.\tag{7}
$$

This strategy ensures that the policy gradients for perception tokens are primarily driven by visual grounding and depth perception accuracy, while those for reasoning tokens are governed by logical consistency. Crucially, both remain strictly anchored to the final answer accuracy. Finally, we optimize the policy parameters θ using a clipped surrogate objective with KL regularization:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( \rho _ { t } ( \theta ) A _ { t } , \operatorname { c l i p } ( \rho _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) \right] - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) ,\tag{8}
$$

where $\begin{array} { r } { \rho _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( y _ { t } | x , y < t \right) } { \pi _ { \mathrm { o l d } } \left( y _ { t } | x , y < t \right) } } \end{array}$ denotes the probability ratio between the current policy and the old policy at time step t, and $\beta$ serves as an adaptive coeficient controlling the KL divergence to ensure that the policy does not deviate excessively from the reference model $\pi _ { \mathrm { r e f } }$

## 2.4 Data Construction

To facilitate SFT and RL training of the proposed structured reasoning framework, high-quality training data for perception and reasoning is essential. We introduce SCOUT-24k, which contains four types of structured spatial reasoning data. Figure 3 illustrates our construction pipeline.

Task Design and Categories. The dataset comprises four categories of spatially related QAs: (1) Spatial Relation Understanding, which requires the model to identify and reason about the relative spatial relationships among multiple objects; (2) Relative Distance Prediction, where the model is asked to determine the relative proximity between objects; (3) Perspective Transformation Reasoning, which evaluates whether the model can infer changes in relative spatial relationships when the viewpoint shifts; and (4) Object-Centric Spatial Reasoning, which assesses the model’s ability to reason about object-level spatial attributes, such as size and existence.

Construction Pipeline. An overview of our construction pipeline is illustrated in Fig. 3. Source images are collected from EmbSpatial [14] and STVQA [4] datasets, both of which provide detailed bounding box annotations and object labels.

In the first stage, we extract spatial and semantic information from each image. We leverage Qwen-VL-Max together with a state-of-the-art monocular depth estimator, Depth-Anything-3 [27], to generate scene captions and estimate the center coordinates $( u , v , z )$ for each object. The depth is sampled at the center of the corresponding bounding box. This process yields a JSON-formatted structured scene representation, which serves as the perception ground truth in the subsequent chain-of-thought reasoning.

In the second stage, we construct the textual reasoning process. Using the derived 3D spatial information, we synthesize reasoning steps and corresponding answers for relative distance prediction and perspective transformation reasoning tasks, based on templates adapted from CV-Bench [44] and Spatial-SSRL [31].

Finally, we refine the generated reasoning processes. We prompt Qwen-VL-Max [3] with the scene caption and synthesized reasoning steps, and retain only those samples whose predicted answers match the ground-truth labels. Subsequently, expert human annotators further review and refine the selected samples to ensure quality and consistency.

Notably, data from STVQA and EmbSpatial already contain answer annotations. For these samples, we execute only the first stage of the pipeline and reserve them for reinforcement learning. Additional details regarding dataset construction are provided in the supplementary material.

## 3 EXPERIMENTS

## 3.1 Experimental Settings

Implementation Details. We train our SCOUT models based on the Qwen2.5- VL series models [3] (3B and 7B). Our training pipeline consists of two sequential stages: (1) SFT Cold-Start: We apply Low-Rank Adaptation (LoRA) [18] to all linear modules with a rank of $r = 8$ . The model is fine-tuned for one epoch on the SFT subset of SCOUT-24k using a learning rate of $1 \times 1 0 ^ { - 4 }$ . (2) RL Training: We train the model using our proposed RL algorithm for 200 steps, with a global batch size of 128 and a learning rate of $1 \times 1 0 ^ { - 6 }$ . To stabilize policy updates, we enforce KL divergence penalty with a coeficient of $\beta = 0 . 0 1$ During exploration, we sample N = 8 trajectories per prompt at a temperature of $T = 1 . 0$ . Checkpoints are saved every 50 steps, and we report the performance of the best-performing checkpoint. Complete implementation details can be found in the supplementary material.

Table 1: Performance comparison on general spatial benchmarks. We include SCOUT in the comparison groups. The best results in each comparison group are highlighted in bold, and the second-best results are underlined.
<table><tr><td rowspan=1 colspan=2>Model</td><td rowspan=1 colspan=1>EmbSpatial</td><td rowspan=1 colspan=3>CV-Bench    BLINKOverall2D  3D  Avg (Depth)</td></tr><tr><td rowspan=1 colspan=6>Proprietary Models</td></tr><tr><td rowspan=1 colspan=2>GPT-40</td><td rowspan=1 colspan=1>70.27</td><td rowspan=1 colspan=1>75.44 83.08 79.26</td><td rowspan=1 colspan=1>76.61</td><td rowspan=1 colspan=1>75.38</td></tr><tr><td rowspan=1 colspan=2>Open So</td><td rowspan=1 colspan=4>urce and Specialized Spatial Models</td></tr><tr><td rowspan=2 colspan=2>Intern-VL3.5-4BIntern-VL3.5-8BSpaceLLaVASpatialBot</td><td rowspan=2 colspan=1>70.9674.8647.8551.67</td><td rowspan=1 colspan=1>70.30 79.75 75.03</td><td rowspan=1 colspan=1>62.90</td><td rowspan=1 colspan=1>69.63</td></tr><tr><td rowspan=1 colspan=1>77.0582.9179.9859.8962.8361.3666.1767.1666.67</td><td rowspan=1 colspan=1>63.7057.2559.67</td><td rowspan=1 colspan=1>72.8555.4959.34</td></tr><tr><td rowspan=1 colspan=6>Spatial Models Based on Qwen2.5-VL-3B</td></tr><tr><td rowspan=3 colspan=2>Qwen2.5-VL-3BSpaceOm-3BSpatialLadder-3BSpatialThinker-3B</td><td rowspan=3 colspan=1>59.8661.0758.1061.89</td><td rowspan=1 colspan=1>66.55 63.50 65.03</td><td rowspan=1 colspan=1>57.25</td><td rowspan=1 colspan=1>60.71</td></tr><tr><td rowspan=1 colspan=1>71.2771.5071.39</td><td rowspan=2 colspan=1>57.2566.1262.90</td><td rowspan=2 colspan=1>63.2464.6366.46</td></tr><tr><td rowspan=1 colspan=1>73.0176.1674.59</td></tr><tr><td rowspan=1 colspan=2>SCOUT-3B (Ours)</td><td rowspan=1 colspan=1>76.95</td><td rowspan=1 colspan=1>75.73 80.91 78.32</td><td rowspan=1 colspan=1>77.42</td><td rowspan=1 colspan=1>77.56</td></tr><tr><td rowspan=4 colspan=2>Qwen2.5-VL-7BSpatialReasoner-7BSpatialThinker-7B</td><td rowspan=1 colspan=4>Spatial Models Based on Qwen2.5-VL-7B</td></tr><tr><td rowspan=1 colspan=1>59.91</td><td rowspan=1 colspan=1>74.89 79.41 77.15</td><td rowspan=3 colspan=1>73.3877.4272.58</td><td rowspan=3 colspan=1>70.1562.2371.55</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>47.30</td><td rowspan=1 colspan=1>47.3576.5861.97</td></tr><tr><td rowspan=1 colspan=1>63.35</td><td rowspan=1 colspan=1>75.5981.8378.71</td></tr><tr><td rowspan=1 colspan=2>SCOUT-7B (Ours)</td><td rowspan=1 colspan=1>78.76</td><td rowspan=1 colspan=1>78.85 83.50 81.18</td><td rowspan=1 colspan=1>79.03</td><td rowspan=1 colspan=1>79.66</td></tr></table>

Evaluation Setup. We compare SCOUT against a comprehensive suite of baselines, categorized into four distinct groups: (i) Proprietary Models, such as GPT-4o [23]; (ii) Open-Source Models, such as the Intern-VL3.5 series [46]; (iii) Spatial Specialist Models, including SpatialBot [5] and SpaceLLaVA [1] (a public reimplementation of SpatialVLM [7]); and (iv) Spatial Models Based on the Qwen2.5-VL Series [3], encompassing variants built upon both the 3B and 7B architectures (e.g., SpaceOm [2], SpatialLadder [26], SpatialReasoner [36], and SpatialThinker [4]).

Evaluations are conducted on a comprehensive suite of six single-image benchmarks, ranging from general spatial understanding to complex spatial reasoning. Specifically, we utilize EmbSpatial [14], CV-Bench [44], and BLINK [15] for general spatial evaluation while employing RoboSpatial [42], SpatialBench [5], and 3DSRBench [35] for complex spatial reasoning tasks. These benchmarks collectively assess fine-grained spatial reasoning capabilities, including depth estimation, relative distance reasoning, and perspective-dependent understanding. Furthermore, to verify our model’s generalization to multi-image and video scenarios, we report the performance achieved on ViewSpatial [25] and VSI-Bench [48].

Table 2: Performance comparison on complex spatial reasoning benchmarks. SCOUT are included for comparison. The best results in each group are highlighted in bold, and the second-best results are underlined.
<table><tr><td rowspan=1 colspan=3>Model</td><td rowspan=1 colspan=2>|RoboSpatial|SpatialBench</td><td rowspan=1 colspan=2>|3DSRBench|Overall</td></tr><tr><td rowspan=1 colspan=7>Proprietary Models</td></tr><tr><td rowspan=1 colspan=3>GPT-40</td><td rowspan=1 colspan=1>70.17</td><td rowspan=1 colspan=1>67.81</td><td rowspan=1 colspan=1>44.78</td><td rowspan=1 colspan=1>60.92</td></tr><tr><td rowspan=1 colspan=7>Open source and specialized spatial models</td></tr><tr><td rowspan=3 colspan=3>Intern-VL3.5-4BIntern-VL3.5-8BSpaceLLaVASpatialBot</td><td rowspan=1 colspan=1>63.59</td><td rowspan=1 colspan=1>57.47</td><td rowspan=1 colspan=1>42.10</td><td rowspan=1 colspan=1>54.39</td></tr><tr><td rowspan=1 colspan=1>74.1247.36</td><td rowspan=1 colspan=1>64.9436.78</td><td rowspan=1 colspan=1>42.0642.02</td><td rowspan=1 colspan=1>60.3742.05</td></tr><tr><td rowspan=1 colspan=1>54.38</td><td rowspan=1 colspan=1>54.02</td><td rowspan=1 colspan=1>45.83</td><td rowspan=1 colspan=1>51.41</td></tr><tr><td rowspan=1 colspan=7>Spatial Models based on Qwen2.5-VL-3B</td></tr><tr><td rowspan=5 colspan=3>Qwen2.5-VL-3BSpaceOm-3BSpatialLadder-3BSpatialThinker-3B</td><td rowspan=1 colspan=1>59.64</td><td rowspan=1 colspan=1>55.74</td><td rowspan=1 colspan=1>40.65</td><td rowspan=1 colspan=1>52.01</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>69.29</td><td rowspan=2 colspan=1>57.47</td><td rowspan=2 colspan=1>41.07</td><td rowspan=2 colspan=1>55.94</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70.61</td><td rowspan=1 colspan=1>55.74</td><td rowspan=1 colspan=1>42.44</td><td rowspan=1 colspan=1>56.26</td></tr><tr><td rowspan=1 colspan=1>71.05</td><td rowspan=1 colspan=1>59.77</td><td rowspan=1 colspan=1>43.31</td><td rowspan=1 colspan=1>58.04</td></tr><tr><td rowspan=1 colspan=3>SCOUT-3B (Ours)</td><td rowspan=1 colspan=1>72.81</td><td rowspan=1 colspan=1>58.04</td><td rowspan=1 colspan=1>44.08</td><td rowspan=1 colspan=1>58.31</td></tr><tr><td rowspan=4 colspan=7>Spatial Models based on Qwen2.5-VL-7BQwen2.5-VL-7B            64.91           62.64           46.13      57.89SpatialReasoner-7B        61.84          50.00          53.79      55.21SpatialThinker-7B                            63.79</td></tr><tr><td rowspan=1 colspan=1>64.91</td><td rowspan=1 colspan=1>62.64</td><td rowspan=1 colspan=1>46.13</td><td rowspan=1 colspan=1>57.89</td></tr><tr><td rowspan=1 colspan=1>61.84</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>53.79</td><td rowspan=1 colspan=1>55.21</td></tr><tr><td rowspan=1 colspan=1>74.12</td><td rowspan=1 colspan=1>63.79</td><td rowspan=1 colspan=1>46.59</td><td rowspan=1 colspan=1>61.50</td></tr><tr><td rowspan=1 colspan=3>SCOUT-7B (Ours)</td><td rowspan=1 colspan=1>71.05</td><td rowspan=1 colspan=1>66.09</td><td rowspan=1 colspan=1>48.23</td><td rowspan=1 colspan=1>61.79</td></tr></table>

We adopt a strict zero-shot evaluation protocol to assess the models’ intrinsic capabilities. To ensure reproducibility, inference is conducted using greedy decoding (temperature T = 0.0). For a fair comparison, we apply model-specific CoT instructions for reasoning-oriented models where applicable. Accuracy serves as the primary evaluation metric across all tasks. Our evaluation pipeline extends the codebase of SpatialThinker [4], adapting it to support diverse input formats. Further evaluation details are available in the supplementary material.

## 3.2 Main Results

Table 1 comprehensively summarizes the evaluation results on the general spatial benchmarks. SCOUT achieves new state-of-the-art performance at both the 3B and 7B scales, outperforming all baselines in their respective categories by significant margins of 16.85% and 9.51%. Most remarkably, SCOUT-7B strictly outperforms the proprietary GPT-4o across all individual benchmarks. Impressively, even our smaller SCOUT-3B variant achieves a higher overall score (77.56)

Table 3: Performance comparison on ViewSpatial (Multi-Image) and VSI-Bench (Video). We compare the baseline Qwen2.5-VL with our SCOUT across diferent scales. NQ and MCQ for VSI-Bench denotes numerical question and multiple-choice question, respectively.
<table><tr><td rowspan="2">Model</td><td>ViewSpatial</td><td colspan="2">VSI-Bench</td></tr><tr><td>Overall</td><td>NQ</td><td>MCQ</td></tr><tr><td rowspan="2">Qwen2.5-VL-3B SCOUT-3B</td><td>36.20</td><td>25.03</td><td>31.65</td></tr><tr><td>38.91</td><td>19.26</td><td>32.97</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B SCOUT-7B</td><td>38.03</td><td>40.52</td><td>34.94</td></tr><tr><td>40.49</td><td>29.35</td><td>38.07</td></tr></table>

than GPT-4o (75.38), decisively demonstrating the efectiveness of our proposed method.

Table 2 further highlights SCOUT’s capabilities in complex spatial reasoning. At the 3B scale, SCOUT achieves 72.81% accuracy on RoboSpatial (+13.17% over Qwen2.5-VL-3B) and secures the highest overall score (58.31) in its group. This advantage amplifies at the 7B scale. Remarkably, SCOUT-7B achieves the highest overall score (61.79) in the entire evaluation, surpassing even the proprietary GPT-4o (60.92). These results confirm SCOUT’s profound capability to tackle highly complex spatial reasoning tasks rather than merely memorizing simple spatial patterns.

## 3.3 Out-of-domain Evaluation.

Table 3 validates the out-of-domain generalization on spatial reasoning tasks in multi-image and video scenarios. Despite being trained exclusively on single image, SCOUT transfers robustly to these domains. On ViewSpatial benchmark, SCOUT achieves overall accuracies of 38.91% (3B) and 40.49% (7B), successfully surpassing their respective baselines. Similar robust transferability is observed in the multiple-choice questions of VSI-Bench across both scales. Notably, we observe a performance drop in the numerical questions of VSI-Bench. This indicates a significant gap between the single-image and video domain in terms of temporal object tracking, requiring a more specialized set of spatial reasoning capabilities to perform absolute numerical estimation across dynamic scenes.

Nevertheless, the consistent gains across other metrics strongly demonstrate that the proposed training approach fundamentally enhances intrinsic structural spatial understanding, enabling generalization to diverse visual formats.

## 3.4 Ablation Study

We conduct ablation study at the 3B scale. Table 4 and Fig. 4 validate the necessity of each reward component within our purposed approach. Overall, the full method achieves the highest average accuracy of 67.94%, significantly outperforming standard GRPO baselines trained without process rewards (65.15%) or without fine-grained credit assignment (65.24%). This substantial gap indicates that standard policy optimization struggles to balance perception and reasoning without our proposed advantage estimating algorithm.

Table 4: Ablation study on reward configurations. We compare the full SCOUT-3B model against several baseline variants: Supervised Fine-Tuning (SFT), GRPO with vanilla Chain-of-Thought $\mathrm { ( G R P O ~ + }$ vanila CoT), GRPO without purposed multiobjective process rewards $\mathrm { ( w / o ~ P r o c e s s . ) }$ , GRPO without fine-grained credit assignment $\mathrm { ( w / o ~ C r e d i t . ) }$ , and our method omitting either the perception advantage $( \alpha _ { 1 } = 0 )$ or the reasoning advantage $( \alpha _ { 2 } = 0 )$ . Across six spatial benchmarks, the best results are bolded and second-best are underlined.
<table><tr><td>Method</td><td colspan="5">|EmbSpatial CV-Bench BLINK(Depth) RoboSpatial SpatialBench 3DSRBench| Avg.</td><td></td></tr><tr><td>SFT</td><td>60.52</td><td>69.66</td><td>58.06</td><td>62.71</td><td>45.40</td><td>34.74</td><td>|55.18</td></tr><tr><td>GRPO + vanila CoT</td><td>76.07</td><td>75.63</td><td>65.32</td><td>69.73</td><td>54.02</td><td>42.63</td><td>63.90</td></tr><tr><td>GRPO w/o Process.</td><td>74.01</td><td>76.02</td><td>70.96</td><td>68.85</td><td>58.62</td><td>42.44</td><td>65.15</td></tr><tr><td> $\mathrm { G R P O ~ w / o ~ C r e d i t } .$ </td><td>75.00</td><td>75.90</td><td>72.58</td><td>72.81</td><td>52.87</td><td>42.29</td><td>65.24</td></tr><tr><td> $\alpha _ { 1 } = 0$ </td><td>74.25</td><td>77.55</td><td>71.77</td><td>71.05</td><td>55.17</td><td>44.57</td><td>65.73</td></tr><tr><td>α2 = 0</td><td>75.76</td><td>78.11</td><td>71.77</td><td>67.54</td><td>55.74</td><td>42.25</td><td>65.20</td></tr><tr><td>SCOUT (Ours)</td><td>76.95</td><td>78.32</td><td>77.42</td><td>72.81</td><td>58.04</td><td>44.08</td><td>67.94</td></tr></table>

![](images/9ee01d517d81b8e48abbfa9d87a043afad70e7f6bd9e739a3b706b3c6f139ffe.jpg)  
Fig. 4: Training dynamics under diferent reward configurations. We track four specific reward components across training steps. Notably, ablating the perception advantage $( \alpha _ { 1 } = 0 )$ leads to a complete failure in optimizing Grounding and Depth reward. Conversely, excluding the reasoning advantage $( \alpha _ { 2 } = 0 )$ causes a severe collapse in the Consistency reward. SCOUT-3B efectively balances all components, achieving superior convergence and stability.

The importance of our fine-grained advantage formulation is clearly demonstrated by isolating specific components. As shown in Fig. 4, when the perception advantage is excluded during optimization $( \alpha _ { 1 } = 0 )$ , the grounding and depth reward fail to optimize entirely. This causes the overall average to drop to 65.73%, severely impacting fundamental perception metrics like BLINK(Depth).

Conversely, excluding the reasoning advantage $( \alpha _ { 2 } \ = \ 0 )$ leads to a catastrophic collapse in the consistency reward during training. This removal causes severe overall performance degradation (dropping to 65.20%), particularly impairing complex spatial tasks that rely on strict logical progression. For instance, accuracy on the RoboSpatial benchmark falls drastically from 72.81% to 67.54%.

Table 5: Sensitivity analysis of the factor α. We evaluate the performance of our model under diferent configurations of $\alpha _ { 1 } = \alpha _ { 2 } = \alpha \in \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ . Here, $\alpha = 0 . 3$ corresponds to our default SCOUT-3B model. Across six spatial benchmarks, the best results are bolded and second-best are underlined.
<table><tr><td>Method</td><td colspan="6">|EmbSpatial CV-Bench BLINK(Depth) RoboSpatial SpatialBench 3DSRBench| Avg.</td></tr><tr><td>α = 0.7</td><td>75.30</td><td>77.86</td><td>75.00</td><td>70.61</td><td>55.74</td><td>42.59</td><td>66.18</td></tr><tr><td>α = 0.5</td><td>75.90</td><td>77.79</td><td>75.80</td><td>71.92</td><td>56.32</td><td>43.01</td><td>66.79</td></tr><tr><td> $\alpha = 0 . 3 ~ \mathrm { ( O u r s ) }$ </td><td>76.95</td><td>78.32</td><td>77.42</td><td>72.81</td><td>58.04</td><td>44.08</td><td>67.94</td></tr></table>

We further conduct a sensitivity analysis for the factors by setting $\alpha _ { 1 } = \alpha _ { 2 }$ As shown in Tab. $5 , \ \alpha = 0 . 3$ achieves the best overall performance, while the average accuracy steadily degrades as α increases. This trend suggests that while process rewards are crucial for guiding step-by-step reasoning, overly emphasizing them may weaken the anchoring efect of the final correctness reward.

## 3.5 Case Study

Qualitative analysis reveals that SCOUT exhibits advanced analytical capabilities by developing structured spatial reasoning trajectories. Figure 5 demonstrates that precise perception with 2D bounding box and 3D depth acts as a reliable foundation for generating logical reasoning chains.

As shown in Case 1, when handling relative distance estimation tasks, SCOUT systematically compares exact extracted depth metrics to resolve spatial ambiguities, avoiding reliance on mere 2D visual cues. Meanwhile, in Case 2, by integrating the visual orientation of objects with their perceived 3D spatial layout, the model successfully infers relative positions from the viewpoint of the boy. Ultimately, SCOUT consistently delivers accurate conclusions accompanied by transparent reasoning trajectories, proving that explicit structured CoT with 3D spatial perception is crucial for supporting robust reasoning.

## 4 Related Work

## 4.1 Spatial Reasoning via Supervised Fine-Tuning

Despite the success of VLMs in general vision-language tasks [3, 10, 12, 28, 32], robust spatial reasoning remains a critical bottleneck. Existing SFT methods address this either via data-intensive post-training on synthetic datasets [5, 7, 8] or by integrating specialized geometric encoders [11, 17, 19, 30, 37]. However, these approaches incur high data/computational overhead or limit architectural versatility. Crucially, SFT-based models tend to memorize training templates rather than genuinely generalizing spatial principles.

![](images/f0abb68897356a008486c9dc86571447e9bfa4038739d6103ddbebd0e560c420.jpg)  
Fig. 5: Qualitative examples of the reasoning process in SCOUT. SCOUT decomposes complex spatial reasoning into three steps: scene comprehension, fine-grained physical perception, and logical analysis.

## 4.2 Spatial Reasoning via Verifiable Reinforcement Learning

To improve generalization, recent works adapt Reinforcement Learning with Verifiable Rewards (RLVR) to multimodal domains [16,22,33,38,41,49,52], including spatial reasoning tasks and structured CoT design [4, 20, 26, 34, 36, 39, 47]. Nonetheless, current approaches face two key limitations: (1) their structured CoTs overlook critical depth-related information for 3D perception, and (2) their reliance on sparse outcome rewards hinders accurate credit assignment for intermediate steps. To resolve these, we propose a depth-aware structured CoT to capture 3D cues, coupled with multi-objective process rewards and a fine-grained advantage estimation algorithm to ensure precise credit assignment.

## 5 Conclusion

In this work, we present SCOUT, a novel model designed to address the critical bottleneck of VLMs in 3D spatial reasoning capabilities. By coupling a depthaware structured CoT with a tailored process-supervised reinforcement learning algorithm, SCOUT efectively resolves the persistent challenges of sparse rewards and inaccurate credit assignment. Additionally, we introduce SCOUT-24k, a comprehensive dataset spanning from basic spatial relations to complex perspective transformations. Extensive evaluations demonstrate that the SCOUT-7B model achieves state-of-the-art performance from general spatial benchmarks to complex spatial reasoning tasks, while exhibiting robust out-of-domain generalization across multi-image and video scenarios. This research establishes a new approach for cultivating spatial reasoning in VLMs, paving a highly promising pathway toward the next generation of spatially aware AI systems.

## References

1. AI, R., Mayorquin, S.: Spacellava models. https://huggingface.co/remyxai/ SpaceLLaVA (2024), accessed: 2026-02-03 9

2. AI, R., Mayorquin, S.: Spaceom models. https://huggingface.co/remyxai/ SpaceOm (2025), accessed: 2026-02-03 9

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 1, 3, 8, 9, 13

4. Batra, H., Tu, H., Chen, H., Lin, Y., Xie, C., Clark, R.: Spatialthinker: Reinforcing 3d reasoning in multimodal llms via spatial rewards. arXiv preprint arXiv:2511.07403 (2025) 2, 5, 8, 9, 10, 14

5. Cai, W., Ponomarenko, I., Yuan, J., Li, X., Yang, W., Dong, H., Zhao, B.: Spatialbot: Precise spatial understanding with vision language models. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 9490–9498. IEEE (2025) 2, 9, 13

6. Chandrasegaran, K., Gupta, A., Hadzic, L.M., Kota, T., He, J., Eyzaguirre, C., Durante, Z., Li, M., Wu, J., Fei-Fei, L.: Hourvideo: 1-hour video-language understanding. Advances in Neural Information Processing Systems 37, 53168–53197 (2024) 1

7. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14455–14465 (2024) 2, 9, 13

8. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems 37, 135062–135093 (2024) 2, 13

9. Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q.V., Levine, S., Ma, Y.: Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161 (2025) 2

10. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025) 1, 13

11. Daxberger, E., Wenzel, N., Grifiths, D., Gang, H., Lazarow, J., Kohavi, G., Kang, K., Eichner, M., Yang, Y., Dehghan, A., et al.: Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7395–7408 (2025) 13

12. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighof, N., Lo, K., Soldaini, L., et al.: Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 91–104 (2025) 13

13. Driess, D., Ha, J.S., Toussaint, M., Tedrake, R.: Learning models as functionals of signed-distance fields for manipulation planning. In: Conference on robot learning. pp. 245–255. PMLR (2022) 1

14. Du, M., Wu, B., Li, Z., Huang, X.J., Wei, Z.: Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 346–355 (2024) 8, 9

15. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. In: European Conference on Computer Vision. pp. 148–166. Springer (2024) 9

16. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025) 14

17. Hong, Y., Zhen, H., Chen, P., Zheng, S., Du, Y., Chen, Z., Gan, C.: 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems 36, 20482–20494 (2023) 13

18. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022) 8

19. Huang, J., Yong, S., Ma, X., Linghu, X., Li, P., Wang, Y., Li, Q., Zhu, S.C., Jia, B., Huang, S.: An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871 (2023) 13

20. Huang, T., Zhang, Z., Tang, H.: 3d-r1: Enhancing reasoning in 3d vlms for unified scene understanding. arXiv preprint arXiv:2507.23478 (2025) 2, 14

21. Huang, W., Wang, C., Li, Y., Zhang, R., Fei-Fei, L.: Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652 (2024) 1

22. Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., Lin, S.: Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749 (2025) 14

23. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024) 1, 3, 9

24. Kuhn, H.W.: The hungarian method for the assignment problem. Naval research logistics quarterly 2(1-2), 83–97 (1955) 5

25. Li, D., Li, H., Wang, Z., Yan, Y., Zhang, H., Chen, S., Hou, G., Jiang, S., Zhang, W., Shen, Y., et al.: Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500 (2025) 4, 10

26. Li, H., Li, D., Wang, Z., Yan, Y., Wu, H., Zhang, W., Shen, Y., Lu, W., Xiao, J., Zhuang, Y.: Spatialladder: Progressive training for spatial reasoning in visionlanguage models. arXiv preprint arXiv:2510.08531 (2025) 2, 9, 14

27. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025) 8

28. Liu, F., Emerson, G., Collier, N.: Visual spatial reasoning. Transactions of the Association for Computational Linguistics 11, 635–651 (2023) 13

29. Liu, S., Zhang, H., Qi, Y., Wang, P., Zhang, Y., Wu, Q.: Aerialvln: Vision-andlanguage navigation for uavs. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15384–15394 (2023) 1

30. Liu, Y., Ma, M., Yu, X., Ding, P., Zhao, H., Sun, M., Huang, S., Wang, D.: Ssr: Enhancing depth perception in vision-language models via rationale-guided spatial reasoning. arXiv preprint arXiv:2505.12448 (2025) 13

31. Liu, Y., Zhang, B., Zang, Y., Cao, Y., Xing, L., Dong, X., Duan, H., Lin, D., Wang, J.: Spatial-ssrl: Enhancing spatial understanding via self-supervised reinforcement learning. arXiv preprint arXiv:2510.27606 (2025) 8

32. Liu, Z., Zhu, L., Shi, B., Zhang, Z., Lou, Y., Yang, S., Xi, H., Cao, S., Gu, Y., Li, D., et al.: Nvila: Eficient frontier visual language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4122– 4134 (2025) 13

33. Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785 (2025) 14

34. Ma, W., Sun, S., Yu, T., Wang, R., Chua, T.S., Bian, J.: Thinking with blueprints: Assisting vision-language models in spatial reasoning via structured object representation. arXiv preprint arXiv:2601.01984 (2026) 2, 14

35. Ma, W., Chen, H., Zhang, G., Chou, Y.C., Chen, J., de Melo, C., Yuille, A.: 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6924–6934 (2025) 9

36. Ma, W., Chou, Y.C., Liu, Q., Wang, X., Melo, C.M.d., Xie, J., Yuille, A.: Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. In: Advances in Neural Information Processing Systems. vol. 38 (2025) 2, 9, 14

37. Ma, W., Ye, L., de Melo, C.M., Yuille, A., Chen, J.: Spatialllm: A compound 3dinformed design towards spatially-intelligent large multimodal models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17249–17260 (2025) 2, 13

38. Meng, F., Du, L., Liu, Z., Zhou, Z., Lu, Q., Fu, D., Han, T., Shi, B., Wang, W., He, J., et al.: Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365 (2025) 14

39. Ouyang, K., Liu, Y., Wu, H., Liu, Y., Zhou, H., Zhou, J., Meng, F., Sun, X.: Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805 (2025) 2, 14

40. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024) 6

41. Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al.: Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 (2025) 14

42. Song, C.H., Blukis, V., Tremblay, J., Tyree, S., Su, Y., Birchfield, S.: Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15768–15780 (2025) 9

43. Tian, X., Gu, J., Li, B., Liu, Y., Wang, Y., Zhao, Z., Zhan, K., Jia, P., Lang, X., Zhao, H.: Drivevlm: The convergence of autonomous driving and large visionlanguage models. arXiv preprint arXiv:2402.12289 (2024) 1

44. Tong, P., Brown, E., Wu, P., Woo, S., IYER, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems 37, 87310–87356 (2024) 8, 9

45. Wang, S., Yu, Z., Jiang, X., Lan, S., Shi, M., Chang, N., Kautz, J., Li, Y., Alvarez, J.M.: Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. CoRR (2024) 1

46. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025) 9

47. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747 (2025) 2, 14

48. Yang, J., Yang, S., Gupta, A.W., Han, R., Fei-Fei, L., Xie, S.: Thinking in space: How multimodal large language models see, remember, and recall spaces. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10632– 10643 (2025) 4, 10

49. Yang, Y., He, X., Pan, H., Jiang, X., Deng, Y., Yang, X., Lu, H., Yin, D., Rao, F., Zhu, M., et al.: R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615 (2025) 14

50. Yang, Y., Sun, F.Y., Weihs, L., VanderBilt, E., Herrasti, A., Han, W., Wu, J., Haber, N., Krishna, R., Liu, L., et al.: Holodeck: Language guided generation of 3d embodied ai environments. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16227–16237 (2024) 1

51. Yokoyama, N., Ha, S., Batra, D., Wang, J., Bucher, B.: Vlfm: Vision-language frontier maps for zero-shot semantic navigation. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 42–48. IEEE (2024) 1

52. Yu, E., Lin, K., Zhao, L., Yin, J., Wei, Y., Peng, Y., Wei, H., Sun, J., Han, C., Ge, Z., et al.: Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954 (2025) 14

53. Zhan, X., Huang, W., Sun, H., Fu, X., Ma, C., Cao, S., Jia, B., Lin, S., Yin, Z., Bai, L., et al.: Actial: Activate spatial reasoning ability of multimodal large language models. arXiv preprint arXiv:2511.01618 (2025) 2

54. Zhang, Y.F., Ren, W., Zhang, Z., Jia, Z., Wang, L., Tan, T.: Focal and eficient iou loss for accurate bounding box regression. Neurocomputing 506, 146–157 (2022) 5

55. Zheng, X., Dongfang, Z., Jiang, L., Zheng, B., Guo, Y., Zhang, Z., Albanese, G., Yang, R., Ma, M., Zhang, Z., et al.: Multimodal spatial reasoning in the large model era: A survey and benchmarks. arXiv preprint arXiv:2510.25760 (2025) 1

56. Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al.: Rt-2: Vision-language-action models transfer web knowledge to robotic control. In: Conference on Robot Learning. pp. 2165–2183. PMLR (2023) 1

## A Additional Dataset Construction Details

Table 6: Question templates for spatial reasoning tasks.
<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Question Template</td></tr><tr><td rowspan=1 colspan=1>RelativeDistancePrediction</td><td rowspan=1 colspan=1>Which object is closer to the {reference object}, the {object a} orthe {object b}? (Options: (A) {object a}, (B) {object b}, (C) theyare equidistant)</td></tr><tr><td rowspan=1 colspan=1>PerspectiveTransformationReasoning</td><td rowspan=1 colspan=1>Assuming a camera is positioned at the {camera position} and fac-ing {facing direction}, according to this camera&#x27;s perspective, whereis the {target object} located ({choice a}, {choice b}, {choicec}, or {choice d})?</td></tr></table>

## A.1 Structured Chain-of-Thought Generation

Structured Chain-of-Thought (CoT) spatial reasoning question-answer pairs are synthesized utilizing the raw data sourced from EmbSpatial and STVQA. These generated question-answer pairs span from fundamental spatial relations to complex spatial reasoning challenges. Table 6 details the question templates utilized to generate the structured CoT data. The generation process involves the following tasks:

1. Spatial Relation Understanding: This task relies on the existing answer annotations from the EmbSpatial dataset, thereby eliminating the necessity to regenerate corresponding questions and answers. We start by generating appropriate captions and scene background information. Subsequently, the reasoning process is directly synthesized based on the 3D coordinates and refined.

2. Relative Distance Prediction: This task involves a target object alongside two candidate objects. We establish a distance threshold of 0.03 meters. Variations below this threshold are classified as equidistant scenarios, whereas diferences exceeding it definitively identify the strictly closer object as the ground truth. Based on their 3D spatial coordinates, we compute the Euclidean distance from the target to each candidate to determine the nearest one.

3. Perspective Transformation Reasoning: This task places a virtual camera at a reference object with a designated orientation. The questions are systematically categorized by 3D spatial ofsets: significant horizontal displacement with negligible depth diference yields lateral queries (e.g., left/right); predominant depth variance leads to longitudinal queries (e.g., front/back);

and substantial ofsets across both axes produce composite directional queries (e.g., front-left or back-right). We compute the relative coordinates based on the x-axis and z-axis, then apply a perspective transformation to construct the reasoning process from the camera’s point of view.

4. Object-Centric Spatial Reasoning: Similar to the first task, the data are extracted from the STVQA answer annotations. The initial stage of the processing pipeline is utilized to extract scene information and captions, efectively bypassing the question-answer generation step.

## A.2 Data Refinement

A comprehensive curation pipeline is deployed to maximize the reliability of the proposed dataset. During question answer generation, we formulate exactly one specific question type for each image in the EmbSpatial training set, which can actively preventing environmental overfitting. For the caption phase, we prompt Qwen-VL-MAX to construct detailed caption of scene. The development of reasoning trajectories is initially grounded in five predefined templates across eleven question types. To avoid rigid linguistic patterns and foster generalized reasoning capabilities, large language model is introduced to systematically rephrase and diversify these initial paths. To improve question answer quality, we employ Qwen-VL-MAX to generate answer based on our scene information, and strictly retain instances where the generated answers match the ground truth labels. Finally, human experts verify the remaining samples to confirm the logical consistency between the reasoning steps and the final conclusions. Following QA quality assessments, a subset of the data generated by Qwen-VL-MAX is selected and employed as supplementary data for cold-start SFT. The prompt we used are provided in the subsequent box.

## Caption Generation Prompt

System Prompt: Task Context:

\- Question: {prompt\_text}

Instructions:

Output the response in the following structure. Briefly introduce the scene in natural language in the <caption> </caption> tags.

## Reasoning Refinement Prompt

## [System Prompt]

You are an expert in Chain-of-Thought reasoning optimization. Your task is to rewrite the ‘Draft Analysis’ to make it natural, coherent, and logical, while STRICTLY PRESERVING all facts, numbers, equations, and object names.

## [User Prompt]

## Context:

Question: {question}

Ground Truth Answer: {answer}

## Draft Analysis (To be rewritten):

{draft\_analysis}

## Instruction:

Rewrite the Draft Analysis above. Combine the calculation steps and the logical deduction into a fluent paragraph. Do NOT change any coordinate values or calculation results. Make it sound like a step-bystep reasoning process.

## Answer Generation Prompt

## System Prompt: Task Context:

\- Question: {prompt\_text}

Metadata (Ground Truth):

\- Scene Objects: {scene\_info\_str}

Precision: BBox = Integer; Depth = 2 decimal places.

## Instructions:

Output the response in the following structure. Do NOT mention you were provided the ground truth or meta data.

Briefly introduce the scene in natural language in the <caption> </caption> tags and output the perception result in the specific JSON format: <scene>[{"label": "red apple", "bbox\_2d": [250, 400, 350, 550], "depth": 2.0}, ...]</scene>. Output JSON lists in a compact, single-line format (no newlines inside JSON).

Then, reason through the scene information to generate reasoning process in the <analyze> </analyze> tags. Output the final option in <answer> </answer> tags.

## A.3 The Statics of SCOUT-24k

Table 7: Detailed statistics of our proposed SCOUT-24k dataset.
<table><tr><td>Subset</td><td>Relation</td><td>Relative</td><td>Perspective Understanding Distance Transformation Centric</td><td>Object</td><td>Total</td></tr><tr><td>SFT</td><td>2,111</td><td>1,500</td><td>1,441</td><td>1,000</td><td>6,052</td></tr><tr><td>RL</td><td>6,917</td><td>3,403</td><td>6,352</td><td>1,942</td><td>18,614</td></tr></table>

The distribution of spatial reasoning tasks in our constructed SCOUT-24k is presented in Tab. 7.

## B Additional Implement Details

## B.1 Prompt Configuration for SCOUT

The specific user prompt designed for the SCOUT framework is detailed in the text box below. To maintain strict alignment between the training and evaluation phases, the specific prompt is applied uniformly across Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and final inference.

![](images/ff3c6d7ce90494466df5adf1cc864c6a50d5a399d48024e4ee915323ef04c67e.jpg)

## B.2 Details of SFT and Reinforcement Learning

All training experiments are conducted on a hardware cluster comprising 7 × NVIDIA L20 GPUs (48GB memory per device). The training of SCOUT is divided into two sequential stages: a SFT cold-start and a subsequent reinforcement learning phase.

Table 8: Hyperparameter used in SFT training.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>lora_rank</td><td>8</td></tr><tr><td>cutoff length</td><td>16384</td></tr><tr><td>per device train batch size 1</td><td></td></tr><tr><td>gradient accumulation steps 16</td><td></td></tr><tr><td>bf16</td><td>true</td></tr><tr><td>gradient_checkpointing</td><td>true</td></tr><tr><td>learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>lr_scheduler_type</td><td>cosine</td></tr><tr><td>warmup_ratio</td><td>0.1</td></tr><tr><td>num_train_epochs</td><td>1</td></tr></table>

Table 9: Hyperparameter used in RL training.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>num generations</td><td>8</td></tr><tr><td>temperature</td><td>1.0</td></tr><tr><td>learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>per_device_train_batch_size 1</td><td></td></tr><tr><td>bf16</td><td>true</td></tr><tr><td>global batch size</td><td>128</td></tr><tr><td>num_train_steps</td><td>200</td></tr><tr><td>β</td><td>0.01</td></tr></table>

SFT Cold-Start Initialization To endow the model with fundamental instruction following capabilities and the ability to produce structured outputs, we conduct a preliminary SFT phase. This cold-start procedure guarantees that the model can generate coherent, task-relevant responses, which serves as a critical prerequisite for the subsequent RL optimization. Implementation is carried out using the LLaMA-Factory framework, with the specific hyperparameters detailed in Tab. 8.

Reinforcement Learning Phase Building upon the initial SFT model, we apply our proposed RL methodology. Crucially, we perform a full-parameter update during this stage. This advanced training phase is executed via the EasyR1 framework, and all corresponding hyperparameters are specified in Tab. 9.

## C Additional Evaluation Details

## C.1 Benchmarks.

To comprehensively evaluate the spatial reasoning capabilities of the proposed model, we select six distinct benchmarks spanning from foundational spatial tasks to complex reasoning scenarios. For general spatial understanding, Embspatial, CV-Bench, and BLINK are employed. Specifically, Embspatial assesses basic spatial relationships (e.g., left/right, above/below, far/close); CV-Bench measures 2D spatial relations, object counting, depth ordering, and distance reasoning; and the Relative Depth subset of BLINK focuses on depth perception. To evaluate 3D capabilities, 3DSRBench is specifically adopted for spatial reasoning in real world. Finally, to probe spatial reasoning within embodied environments, we utilize SpatialBench and RoboSpatial. SpatialBench broadly covers spatial comprehension across object existence, positional relationships, physical interactions (e.g., reachability), and size comparisons. Furthermore, evaluations on the Configuration and Compatibility subsets of RoboSpatial simulate complex embodied tasks that demand a holistic understanding of object-to-object relationships.

Table 10: Detailed Accuracy Results on 3DSRBench (%)
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>3DSRBench TasksHeight Location Orientation Multi-Object</td><td rowspan=1 colspan=1>Avg.</td></tr><tr><td rowspan=1 colspan=3>Proprietary Models</td></tr><tr><td rowspan=1 colspan=1>GPT-40</td><td rowspan=1 colspan=1>54.9   59.8      21.8        39.5</td><td rowspan=1 colspan=1>44.78</td></tr><tr><td rowspan=1 colspan=1>Open Sou</td><td rowspan=1 colspan=1>rce and Specialized Spatial Models</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Intern-VL3.5-4BIntern-VL3.5-8BSpaceLLaVASpatialBot</td><td rowspan=1 colspan=1>37.43  51.89     38.1       36.5744.29  49.26    35.24       38.0641.71   52      23.43       43.3150.29  60.69    20.19       44.57</td><td rowspan=1 colspan=1>42.142.0642.0245.83</td></tr><tr><td rowspan=1 colspan=1>Spatia</td><td rowspan=1 colspan=1>l Models Based on Qwen2.5-VL-3B</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-3B</td><td rowspan=4 colspan=1>38   52.69     33.9       33.7136.29  54.29      32        35.243.71  52.46    25.33       42.1737.71  56.11     33.9        38.4</td><td rowspan=4 colspan=1>40.6541.0742.4443.31</td></tr><tr><td rowspan=1 colspan=1>SpaceOm-3B</td></tr><tr><td rowspan=1 colspan=1>SpatialLadder-3B</td></tr><tr><td rowspan=1 colspan=1>SpatialThinker-3B</td></tr><tr><td rowspan=1 colspan=1>SCOUT-3B (Ours)</td><td rowspan=1 colspan=1>42.86 55.31      32        40.57</td><td rowspan=1 colspan=1>44.08</td></tr><tr><td rowspan=1 colspan=1>Spatia</td><td rowspan=1 colspan=1>l Models Based on Qwen2.5-VL-7B</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-7B</td><td rowspan=2 colspan=1>46.57  59.54    36.57       38.2946   68.91     47.43       43.31</td><td rowspan=2 colspan=1>46.1353.03</td></tr><tr><td rowspan=1 colspan=1>SpatialReasoner-7B</td></tr><tr><td rowspan=1 colspan=1>SpatialThinker-7B</td><td rowspan=1 colspan=1>44.29  57.83     37.9       41.49</td><td rowspan=1 colspan=1>46.59</td></tr><tr><td rowspan=1 colspan=1>SCOUT-7B (Ours)</td><td rowspan=1 colspan=1>41.43  63.66    34.29       43.89</td><td rowspan=1 colspan=1>48.23</td></tr></table>

## C.2 Baselines

To evaluate the proposed SCOUT models, we compare them against a diverse set of baselines grouped into three categories: (1) Proprietary Models: We include GPT-4o, which represents the current state-of-the-art in commercial multimodal reasoning. It serves as an upper bound for spatial generalization under closed-source training regimes. (2) Open-Source General VLMs: We evaluate Qwen2.5-VL (3B and 7B), representing cutting-edge architectures that ofer strong general visual reasoning but lack specific spatial tuning. Additionally, we include InternVL-3.5 (4B and 8B), an advanced VLM featuring enhanced training strategies and superior data quality compared to its predecessors. (3) Specialized Spatial VLMs: We compare with several open-source models explicitly tailored for spatial reasoning. These include SpaceLLaVA (a public re-implementation of SpatialVLM) and SpatialBot, which integrates RGB-D inputs for robust spatial perception. Furthermore, to enable a direct comparison within the Qwen2.5-VL ecosystem, we evaluate four Qwen-based spatial models: SpatialThinker (fine-tuned for structured spatial reasoning), SpaceOm (incorporating deeper chain-of-thought traces and Robo2VLM data), SpatialLadder (employing a progressive framework to enhance spatial perception), and Spatial-Reasoner (optimized via reinforcement learning and explicit 3D representations).

Table 11: Detailed Accuracy Results on ViewSpatial (%)
<table><tr><td rowspan="2">Models</td><td colspan="3">Camera perspective</td><td colspan="4">Person perspective</td></tr><tr><td>Direction</td><td>Relative Object View Orientation</td><td>Total</td><td>Scene Simulation Object View Relative Relative Direction</td><td>Orientation Direction</td><td></td><td>Total</td></tr><tr><td>Qwen-2.5-VL-3B</td><td>44.22</td><td>31.93</td><td>39.80</td><td>26.70</td><td>41.16</td><td>31.00</td><td>32.82</td></tr><tr><td>SCOUT-3B</td><td>46.87</td><td>29.02</td><td>40.45</td><td>28.24</td><td>49.50</td><td>35.39</td><td>37.48</td></tr><tr><td>Qwen-2.5-VL-7B</td><td>47.66</td><td>30.32</td><td>41.42</td><td>27.87</td><td>41.57</td><td>35.99</td><td>34.83</td></tr><tr><td>SCOUT-7B</td><td>50.14</td><td>22.59</td><td>40.23</td><td>26.87</td><td>55.62</td><td>41.33</td><td>40.74</td></tr></table>

Table 12: Detailed Accuracy Results on VSI-BENCH (%)
<table><tr><td rowspan="3">Models</td><td colspan="5">NQ</td><td colspan="7">MCQ</td></tr><tr><td>obj count est</td><td>obj size room size obj abs</td><td>est</td><td>dist</td><td>AVG</td><td>rel dir (hard)</td><td>rel dir (medium)</td><td>rel dir (easy)</td><td>rel dist</td><td>appear order</td><td>route planning</td><td>AVG</td></tr><tr><td>Qwen-2.5-VL-3B 15.24</td><td></td><td>25.83</td><td>31.91</td><td>28.38</td><td>25.03</td><td>30.56</td><td>30.95</td><td>32.2631.97</td><td></td><td>32.20</td><td>31.44</td><td>31.65</td></tr><tr><td>SCOUT-3B</td><td>44.53</td><td>11.98</td><td>13.19</td><td>12.54</td><td>19.26</td><td>26.27</td><td>32.54</td><td></td><td>47.47 31.69</td><td>33.82</td><td>32.47</td><td>32.97</td></tr><tr><td>Qwen-2.5-VL-7B 47.20</td><td></td><td>48.30</td><td>45.45</td><td>25.41</td><td>40.52</td><td>23.86</td><td>26.98</td><td>47.47</td><td>41.55</td><td>35.76</td><td>30.93</td><td>34.94</td></tr><tr><td>SCOUT-7B</td><td>59.15</td><td>18.24</td><td>27.60</td><td>22.47</td><td>29.35</td><td>30.56</td><td>47.35</td><td>51.15 39.85</td><td></td><td>32.68</td><td>30.41</td><td>38.07</td></tr></table>

## C.3 Detailed Experiments Result

The detailed spatial reasoning results on 3DSRBench, ViewSpatial, and VSI-Bench are provided in Tabs. 10 to 12.

## D Limitations and Future Work

While the proposed approach demonstrates promising results, it remains subject to several limitations. First, due to computational resource constraints, our experiments are currently restricted to models at the 3B and 7B parameter scales. Second, the datasets rely on annotations of bounding boxes and object labels, lacking broader modalities such as multi-image contexts or video data. Finally, the proposed reinforcement learning method requires a strictly structured chain-of-thought format to extract perception results, which restricts flexibility and may be sub-optimal for other visual reasoning tasks.

To address these limitations, future work can focus on several key directions. First, scaling the framework to larger foundation models (e.g., 14B, 32B, and beyond) will facilitate the investigation of how increased model capacity afects spatial reasoning and RL optimization. Second, integrate multi-image and video modalities will extend our research from static 2D understanding to dynamic, spatiotemporal reasoning and incorporate locations and captions generated by VLMs will reduce the reliance on dense bounding box and label annotations. Finally, relaxing the strict dependency on a structured CoT format may enable more flexible, free-form reasoning pathways that generalize across a broader spectrum of complex visual reasoning tasks.