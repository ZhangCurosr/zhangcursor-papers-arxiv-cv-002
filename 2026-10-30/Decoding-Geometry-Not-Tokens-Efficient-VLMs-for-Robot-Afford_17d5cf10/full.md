# Decoding Geometry, Not Tokens: Efficient VLMs for Robot Affordance and Trajectory Prediction

Akila Ayanthi<sup>1,2</sup> Darshana Priyasad<sup>1</sup> Tharindu Fernando<sup>1</sup> Dimity Miller<sup>1</sup> Sridha Sridharan<sup>1</sup> Clinton Fookes<sup>1</sup> Peyman Moghadam<sup>2</sup>

<sup>1</sup>SAIVT, School of Electrical Engineering and Robotics, Queensland University of Technology, Australia <sup>2</sup>CSIRO Robotics, CSIRO, Australia

<sup>1</sup>{a.mudiyanselage, dp.don, t.warnakulasuriya, d24.miller, s.sridharan, c.fookes}@qut.edu.au, <sup>2</sup>peyman.moghadam@csiro.au

## Abstract

Predicting where to grasp and how to move is fundamentally a geometry problem. Vision-language models treat it as language generation instead. They tokenize spatial coordinates into text and decode them autoregressively. This mechanism ignores spatial structure and truncates continuous precision. Errors compound across coordinate tokens during decoding. In this paper, we show that compact off-the-shelf VLMs with 0.5–2B parameters already encode rich spatial information in their intermediate hidden states. Autoregressive decoding is the dominant bottleneck for trajectory prediction, while affordance prediction depends additionally on feature depth and decoder architecture. We expose the latent spatial information with a lightweight, geometry-compatible prediction head that attaches to a frozen backbone. It reads features from an intermediate hidden layer, conditions a learned query on the task instruction and refines it via cross-attention over the visual tokens to regress affordance boxes and trajectory waypoints directly as continuous values, without modifying the backbone. We evaluate on ShareRobot-Bench and AGD20K. Our 2B variant reduces trajectory error by over 60% and improves ShareRobot-Bench affordance mAP by 85% relative to the autoregressive baseline. It approaches RoboBrain on trajectory prediction (0.251 vs 0.191 DFD) at 3.8× higher throughput. These results demonstrate a favorable accuracy–efficiency trade-off for instructionconditioned 2D geometric prediction.

## 1. Introduction

Vision-language models (VLMs) are powerful tools for general-purpose reasoning, demonstrating strong performance across visual question answering [6, 20], image captioning [21, 38], and scene understanding [10, 31].

![](images/9e8a526bd9d5d8957dffd98c3e6420986ab2cb7da9efccc3f2ef056f2ea4b94a.jpg)  
Figure 1. Accuracy, efficiency, and model size. Each bubble represents a model; bubble size encodes the number of active parameters. Color denotes model family: LLaVA-OV, Qwen2-VL, Robo-Brain. Dashed arrows indicate improvement over the corresponding base VLM. (Top) Trajectory prediction DFD (↓) vs. inference speed (FPS on H100) on ShareRobot-Bench-Trajectory. (Bottom) Affordance mAP (↑) vs. inference speed (FPS on H100) on AGD20K. Our method outperforms both base VLMs, approaches larger-model accuracy, and retains favorable parameter and inference efficiency.

However, using VLMs efficiently for robotic manipulation remains challenging, particularly for instructionconditioned geometric prediction prior to physical execution. Manipulation requires spatially precise outputs describing where to interact with the scene (i.e., affordance prediction) and how to move there (i.e., trajectory prediction), whereas standard VLMs are designed primarily for tokenized language generation. Practical robotic systems also operate under constrained compute budgets [11, 44], while modern VLMs can contain billions of parameters [1, 2]. We therefore ask: can compact VLMs provide accurate geometric predictions for robotic affordance and trajectory prediction, while retaining efficient inference?

Existing approaches typically use VLMs for robotic manipulation by producing tokenized integers that are decoded autoregressively [16, 28, 40]. This effectively treats geometric prediction as a language generation problem. Although this allows direct utilization of pre-trained VLMs and their existing architectures, it introduces a structural mismatch between the prediction mechanism and the task objective: Token-level cross-entropy learning objectives do not penalize geometric distance, finite token vocabularies do not allow for continuous spatial outputs and introduce quantization error, and autoregressive generation compounds errors across coordinate tokens. While larger VLM capacity, particularly when paired with large-scale training on robotics data [9, 13, 16] can yield performance gains for robotics manipulation in spite of this mismatch, this is at the cost of substantially lower inference speeds and higher compute requirements.

In this paper, we investigate how compact, off-the-shelf VLMs can be repurposed for geometric prediction. We replace token-based coordinate generation with a lightweight geometry-compatible decoder that directly regresses affordance boxes and trajectory waypoints from a frozen VLM. Our systematic ablations show that direct continuous prediction is the primary source of improvement for trajectory prediction, while affordance prediction depends more strongly on the interaction between prediction interface, representation depth, and decoder architecture. Layer-wise analysis further reveals that useful spatial information is already present across the VLM representation hierarchy, although its accessibility varies with task and depth. Our contribution is therefore not the DETR architecture itself, but identifying the decoding-interface mismatch and showing how existing geometric representations can be exploited efficiently.

Overall, we make three empirically validated claims:

• A geometry-compatible prediction interface substantially improves affordance and trajectory prediction while preserving the efficiency of compact VLMs (Sec. 4.2).

• Direct continuous prediction is the primary source of improvement for trajectory prediction, while affordance prediction depends on the interaction between prediction interface, representation depth, and decoder architecture (Sec. 4.3).

• Compact VLMs already encode useful spatial information across their representation hierarchy, with its accessibility varying across tasks and model depth (Sec. 4.4).

## 2. Related Work

## 2.1. Vision-Language Models

Vision-language models (VLMs) pretrained on internetscale datasets have demonstrated remarkable capabilities across tasks, including visual question answering [6, 20] and object localization [8, 30]. Scaling to tens of billions of parameters, for example, Flamingo-80B [1], InternVL3- 78B [45], Qwen2.5-VL-72B [2], has driven strong zero and few-shot performance, but at high computational cost for deployment.

Recent efforts have focused on improving efficiency through architectural design and model compression, leading to compact models such as LLaVA-OneVision (0.5B, 7B) [19], Qwen2-VL (2B, 4B, 7B) [38], Molmo (1B, 7B) [7]. Despite their strong semantic capabilities, most VLMs are trained using autoregressive next-token prediction and are therefore optimized for language generation rather than geometric reasoning. Consequently, spatially demanding tasks often require full fine-tuning or parameter-efficient adaptation to learn task-specific geometric representations [5, 24]. In this work, we instead treat compact VLMs as frozen feature extractors and ask whether their internal multimodal representations already contain sufficient geometric information for affordance and trajectory prediction. We use lightweight, geometry-aware decoders on top of frozen features, avoiding any modifications to the backbone.

## 2.2. VLMs for Embodied Tasks

VLMs have been widely adopted in robotics for representation learning [15, 17, 25], high-level planning [35, 41], and reward or feedback generation [33, 42]. More recently, Vision-Language-Action (VLA) models have extended this paradigm by directly mapping visual and language observations to low-level actions [18, 27, 37]. Closely related embodied foundation models further integrate perception, reasoning, memory, and planning capabilities [16, 36, 40]. Examples include AffordanceLLM [28], which predicts affordance tokens for dense mask generation, and models such as LLARVA [26], RoboBrain [16], and RoboPoint [40], which rely on extensive backbone adaptation using large embodied datasets.

Although these approaches achieve impressive generalization, they inherit the computational cost of large VLM backbones and remain challenging to deploy on resourceconstrained platforms [43, 44]. Recent studies identify the VLM backbone itself as a major computational bottleneck [11, 39] and motivate efficient variants such as EdgeVLA [3, 32]. Unlike prior work that designs new VLA architectures or heavily fine-tunes backbones, we use compact off-the-shelf VLMs as frozen perception modules and learn lightweight geometry-aware decoders for affordance and trajectory prediction.

## 3. Methodology

Our goal is to bridge the performance gap between large embodied foundation models and compact, off-the-shelf VLMs, while substantially reducing the computational overhead associated with larger embodied foundation mod els. We pursue this via a simple but principled hypothesis: compact VLMs already encode sufficient geometric and semantic information for spatial reasoning; a key bottleneck lies in the decoding strategy rather than insufficient representational capacity. Existing approaches either repurpose autoregressive text generation for coordinate regression, an inherently misaligned objective, or rely on heavyweight embodied foundation models that are impractical on resourceconstrained hardware. We instead keep the VLM entirely frozen and attach a lightweight geometric prediction head that operates in its hidden feature space. The prediction head maps intermediate VLM representations to continuous spatial outputs using a DETR-style transformer decoder followed by task-specific feed-forward networks (Fig. 2). We describe the model architecture and training strategy in the following subsections.

## 3.1. Model Architecture

Our model comprises two modules: (i) a frozen VLM that serves as a shared visual-linguistic encoder, and (ii) a lightweight geometry prediction head consisting of a DETR-style transformer decoder and task-specific feedforward networks (FFNs) that regress continuous spatial outputs. All learned parameters reside exclusively in the prediction head, keeping the added footprint minimal.

## 3.1.1. Vision-Language Model (VLM) Backbone

We first instantiate an off-the-shelf VLM as a frozen backbone. We assume a canonical architecture: a visual encoder extracts patch features that are projected into the LLM embedding space via a learned adapter, yielding a sequence of visual token features $\mathbf { F } _ { I } \in \mathbb { R } ^ { N _ { v } \times d }$ , where $N _ { v }$ is the number of visual tokens and d is the LLM hidden dimension. The text tokenizer maps the natural-language task instruction to token embeddings $\mathbf { F } _ { T } \in \mathbb { R } ^ { N _ { t } \times d }$ . The two sequences are concatenated and processed by the LLM, which attends jointly over visual and textual context to produce a contextualized representation. Crucially, no weights of the VLM backbone are modified at any point; improvements therefore arise from the learned geometric prediction interface operating on frozen VLM representations rather than from backbone adaptation.

## 3.1.2. Geometry Prediction Head

To decode geometric information latent in the frozen VLM, we adopt a DETR-style architecture [4]. The frozen VLM backbone acts as the transformer encoder, and taskspecific transformer decoder-FFN pairs independently predict bounding box coordinates (for affordance localization) and trajectory waypoints. This design avoids any autoregressive generation of coordinate tokens and instead treats geometric prediction as a direct regression problem, better aligned with the continuous and spatially structured nature of manipulation outputs.

Transformer Decoder: Each decoder consists of M layers, each comprising multi-head self-attention, multi-head cross-attention over the frozen visual memory tokens V, and a feed-forward sub-layer, with Pre-LN normalization throughout. We use a single learned query per head; the key design choice is how this query is initialized.

In autoregressive LLMs, the hidden state at the final text token position at layer $l ^ { * }$ , denoted $\mathbf { h } _ { \mathrm { l a s t } } \in \mathbb { R } ^ { d }$ , attends to the full multimodal context and therefore encodes a holistic, instruction-grounded summary of the scene. We exploit this by conditioning the initial query on this representation:

$$
{ \bf q } _ { 0 } = { \bf e } _ { \mathrm { p o s } } + W _ { c } { \bf h } _ { \mathrm { l a s t } } ,\tag{1}
$$

where $\mathbf { e } _ { \mathrm { p o s } } \in \mathbb { R } ^ { d _ { d e c } }$ is a learned positional embedding and $W _ { c } \in \dot { \mathbb { R } ^ { d _ { d e c } \times d } }$ is a learned linear projection. This ensures that from the first cross-attention layer, the query is already biased toward image regions relevant to the specific instruction, rather than relying on the decoder to discover that bias from scratch. The decoder then refines ${ \bf q } _ { 0 }$ across M layers via cross-attention to V, yielding a final representation $\mathbf { q } _ { M } \in \mathbb { R } ^ { d _ { \mathrm { d e c } } }$ that is both semantically grounded in the instruction and spatially precise enough to decode into geometric coordinates.

Prediction Feed-Forward Networks (FFNs): After M decoder layers, the refined query $\mathbf { q } _ { M } \in \mathbb { R } ^ { d _ { d e c } }$ is mapped to continuous spatial outputs by a task-specific prediction FFN: a 3-layer MLP with GeLU activations and hidden dimension $d _ { d e c }$ . We maintain two independent decoder-FFN pairs, one for affordance prediction and one for trajectory prediction. Both receive the same frozen VLM features as encoder context and the same conditioning signal $\mathbf { h } _ { \mathrm { l a s t } } ,$ but maintain separate weights and are trained with separate losses. This allows each head to specialize its crossattention patterns and query refinement toward its respective geometric output, without interference between the two tasks.

Affordance head: The affordance FFN regresses to a normalized bounding box in corner format:

$$
\hat { b } = \mathrm { F F N } _ { \mathrm { a f f } } \left( \mathbf { q } _ { M } ^ { \mathrm { a f f } } \right) = \left( \hat { x } _ { 1 } , \hat { y } _ { 1 } , \hat { x } _ { 2 } , \hat { y } _ { 2 } \right) .\tag{2}
$$

![](images/a02aa1be5f4da1c5bd07e80720b3698cc7550dceb84ce9ecd5da753f4e081a08.jpg)  
Figure 2. Architecture overview. A frozen VLM processes an RGB image and task instruction; hidden states from an intermediate layer are passed to a lightweight geometric prediction head. A DETR-style transformer decoder, conditioned on the instruction via the final text token hidden state, refines a learned query against the visual memory tokens. Two independent FFNs decode the query into continuous outputs: affordance boxes (L1+GIoU loss) and trajectory waypoints (L1+startpoint+endpoint loss). No gradient flows into the VLM.

Trajectory head: The trajectory FFN unrolls T waypoints in a single forward pass:

$$
\hat { \tau } = \mathrm { F F N } _ { \mathrm { t r a j } } ( \mathbf { q } _ { M } ^ { \mathrm { t r a j } } ) = \{ ( \hat { x } _ { t } , \hat { y } _ { t } ) \} _ { t = 1 } ^ { T } .\tag{3}
$$

In both cases, a sigmoid activation in the output layer constrains all predicted coordinates to [0, 1], normalized with respect to the dimensions of the input image. At inference time, predictions are de-normalized back to pixel coordinates by multiplying by the original image width and height, respectively.

## 3.2. Training

All training signals flow exclusively through the geometry prediction head; the VLM backbone remains frozen throughout. We define separate losses for the two tasks, each designed to handle the precise, spatial prediction for robot manipulation.

## 3.2.1. Affordance Loss

Given a predicted bounding box $\hat { \textbf { b } } = \mathbf { \kappa } \left( \hat { x } _ { 1 } , \hat { y } _ { 1 } , \hat { x } _ { 2 } , \hat { y } _ { 2 } \right) \mathbf { \Omega } \in$ $[ 0 , 1 ] ^ { 4 }$ and ground-truth $\mathbf { b } = ( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ) \in [ 0 , 1 ] ^ { 4 }$ , we combine coordinate regression with a geometry-aware overlap term:

$$
\mathcal { L } _ { \mathrm { a f o r d } } = \lambda _ { \mathrm { L 1 } } \mathcal { L } _ { \mathrm { L 1 } } ( \hat { \mathbf { b } } , \mathbf { b } ) + \lambda _ { \mathrm { G I o U } } \mathcal { L } _ { \mathrm { G I o U } } ( \hat { \mathbf { b } } , \mathbf { b } ) .\tag{4}
$$

The coordinate term ${ \mathcal { L } } _ { \mathrm { L 1 } }$ is a Smooth L1 loss applied element-wise to the box corners, and the overlap term ${ \mathcal { L } } _ { \mathrm { G I o U } }$ is the Generalized IoU loss [29]. We use a tighter Smooth L1 threshold and standard GIoU; full definitions are provided in the supplementary material (Section A.1).

## 3.2.2. Trajectory Loss

Given predicted waypoints $\hat { \mathcal { T } } = \{ \hat { \mathbf { p } } _ { t } \} _ { t = 1 } ^ { T }$ and ground truth $\mathscr { T } = \{ \mathbf { p } _ { t } \} _ { t = 1 } ^ { T }$ , with pˆ<sub>t</sub>, p<sub>t</sub> $\in [ 0 , 1 ] ^ { 2 }$ , we define:

$$
{ \mathcal { L } } _ { \mathrm { t r a j } } = \lambda _ { \mathrm { L 1 } } { \mathcal { L } } _ { \mathrm { L 1 } } + \lambda _ { \mathrm { s t a r t } } { \mathcal { L } } _ { \mathrm { s t a r t } } + \lambda _ { \mathrm { e n d } } { \mathcal { L } } _ { \mathrm { e n d } } .\tag{5}
$$

The main term ${ \mathcal { L } } _ { \mathrm { L 1 } }$ is a Smooth L1 loss applied uniformly over all T waypoints, and $\mathcal { L } _ { \mathrm { s t a r t } }$ and $\mathcal { L } _ { \mathrm { e n d } }$ are additional Smooth L1 penalties on the first and last waypoints that upweight errors at the contact and termination points of the trajectory. We use the same Smooth L1 defined in the affordance loss; full formulas are given in the supplementary material (Section A.2).

## 4. Experimental Analysis

In this section we evaluate our method on both trajectory and affordance prediction and analyze the efficiencyaccuracy trade-off against larger VLM baselines. We first discuss the experimental setup, including the baselines, datasets, evaluation metrics and implementation details.

## 4.1. Experimental Setup

## 4.1.1. Baselines

We evaluate our approach on two families of VLMs, Qwen2-VL [38], using the 2B variety as our small VLM, and LLaVA-OneVision [19], using the 0.5B variety. We compare against four categories of baseline methods:

• Baseline small VLMs: We test both the compact Qwen2- VL-2B and LLaVA-OV-0.5B in their original decoding configurations: we generate coordinates autoregressively as text tokens with no task-specific training on ShareRobot [16], representing the zero-shot geometric capability of off-the-shelf VLMs.

• Baseline large VLMs: We additionally test the larger 7B counterparts of our small VLMs (Qwen2-VL-7B and LLaVA-OV-7B) in their original decoding configurations, as a reference point for performance gains that can be unlocked with larger model capacity, despite incurring additional computational cost.

• LoRA fine-tuned VLMs: We fine-tune Qwen2-VL-2B and LLaVA-OV-0.5B on ShareRobot [16] via next-token prediction on tokenized coordinates using LoRA [14].

• RoboBrain [16]: A VLM with large-scale training on embodied data. It is a generalist model that offers stateof-the-art performance across a range of robotic manipulation tasks, albeit at substantially greater computational cost.

## 4.1.2. Datasets

We train on the ShareRobot dataset [16], which aggregates data from 23 sources in the Open-X Embodiment collection [27]. For affordance prediction, the training set contains 6,022 images and 500 validation images with bounding-box annotations $\{ l _ { x } , l _ { y } , r _ { x } , r _ { y } \}$ denoting the affordance region for a given high-level task description. For trajectory prediction, it contains 6,370 images and 500 validation images with gripper trajectories of up to T = 4 waypoints $\{ ( x _ { t } , y _ { t } ) \} _ { t = 1 } ^ { T }$ in image space.

We evaluate the performance of our method and the baselines on two affordance benchmarks, AGD20K [23] and ShareRobot-Bench-Affordance [16]. We also evaluate on ShareRobot-Bench-Trajectory [16] for the trajectory prediction task. AGD20K provides out-of-distribution evaluation on diverse everyday objects, while ShareRobot-Bench assesses performance on manipulation-specific scenarios drawn from the same domain as the training data. Additional dataset and annotation details are provided in the Supplementary Section B.

## 4.1.3. Evaluation Metrics

Following prior work, for affordance prediction we report mean Average Precision (mAP↑) across multiple IoU thresholds (0.25, 0.50, 0.75, 0.90) [16, 36]. For trajectory prediction, we report three complementary metrics [16, 36]: Discrete Frechet Distance (DFD´ ↓) [12], which measures order-sensitive trajectory shape; Hausdorff Distance (HD↓) [34], which captures the worst-case spatial gap; and Root Mean Squared Error (RMSE↓), which measures average point-wise deviation. Inference efficiency is reported as frames per second (FPS) and per-sample floating-point operations (TFLOPs). The detailed efficiency measurement protocol is described in Supplementary Section C.

## 4.1.4. Implementation Details

Our DETR-style transformer decoder comprises $M \ = \ 3$ layers with hidden dimension $d _ { d e c } = 2 5 6$ and 8 attention heads. The prediction FFN is a 3-layer MLP with hidden dimension 256 and GeLU activations. We optimize with AdamW [22] at a learning rate of $1 \times 1 0 ^ { - 4 }$ with linear warmup followed by cosine decay, training for 20 epochs at batch size 16. Unless indicated otherwise, experiments are run on a single NVIDIA H100 GPU with bf16 mixed precision. Only the decoder and FFN heads are updated during training; the VLM backbone is frozen throughout. Implementation and training cost of LoRA fine-tuning can be found in Supplementary Section D.

4.2. Trajectory and Affordance Prediction Results Table 1 presents the trajectory and affordance prediction performance across all models and baselines on ShareRobot-Bench and AGD20K, along with their inference speeds. Across both model families, our approach reduces trajectory DFD by over 68% and improves affordance mAP by up to 8.7× over the corresponding vanilla small VLM baselines. This supports our first claim that pairing small VLMs with geometrically compatible prediction heads enables substantial performance gains. These performance gains are accompanied by higher measured throughput on the H100 (7.9-21 FPS), as compared to 4.1-15.2 FPS for the larger VLM baselines. We discuss the performance of our method for the two tasks in more detail below.

## 4.2.1. Trajectory Prediction

Our proposed geometry head yields the largest and most consistent gains on this task. Across both backbones, replacing autoregressive coordinate generation with direct regression achieves a relative reduction in DFD exceeding 60%, with proportional reductions in HD and RMSE, confirming improvements in both global shape fidelity and point-wise accuracy. In particular, our Qwen2-VL-2B and LLaVA-OV-0.5B variants achieve DFDs of 0.251 and 0.225, respectively, approaching RoboBrain at 0.191 DFD, while achieving up to 3.8× faster inference on the H100 with 5–10× fewer parameters.

Together, these results validate our first claim that pairing small VLMs with a geometrically compatible prediction head enables substantial performance gains while maintaining superior efficiency over larger VLM counterparts.

LoRA fine-tuning of the base VLMs fails to deliver consistent improvements and remains substantially below our method across all metrics. For both backbones, fine-tuning only marginally reduces DFD, from 0.798 to 0.752 ± 0.06 for Qwen and from 0.777 to 0.695±0.001 for LLaVA, while remaining substantially worse than the geometry-head variants. These results suggest that autoregressive text generation is a poorly aligned objective for continuous trajectory regression.

Table 1. Accuracy and inference efficiency comparison. Our Geometric Head variants on frozen Qwen2-VL-2B and LLaVA-OV-0.5B backbones (Emb. = ✓ indicates embodied training; Head ∈ {token, geo}). We report active backbone parameters (#P(B)), accuracy metrics (DFD, HD, RMSE, mAP), throughput on H100 and RTX 3090 (FPS↑), and compute cost (TFLOPs↓). <sup>†</sup>Fully trained on embodied data; al other large VLMs are evaluated zero-shot. SRB=ShareRobot-Bench. Geo-Head substantially outperforms zero-shot and LoRA baselines and approaches 7B/RoboBrain accuracy at a fraction of the parameters and compute.
<table><tr><td colspan="6"></td><td colspan="6">Trajectory</td><td colspan="6">Affordance</td></tr><tr><td rowspan="3">Category</td><td rowspan="3">Model</td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3">Emb. Head |#P(B)</td><td colspan="4">SRB</td><td rowspan="3">FPS↑ |H100 RTX</td><td rowspan="3"></td><td rowspan="3">TFLOPs↓ |#P(B)|</td><td rowspan="3">AGD20K</td><td rowspan="3"></td><td rowspan="3">SRB</td><td rowspan="3"></td><td rowspan="3">FPS↑</td><td rowspan="3">TFLOPs↓ |H100 RTX</td></tr><tr><td></td><td>DFD↓</td><td>RMSE↓</td><td>mAP↑</td></tr><tr><td>Large</td><td>RoboBrain† [16]</td><td>√</td><td></td><td>token</td><td>0.191</td><td>0.171</td><td>0.133</td><td>5.5</td><td>1.2</td><td>31.0</td><td>8.2</td><td>21.78</td><td>20.57</td><td></td><td>4.1</td><td>0.8</td><td>80.0</td></tr><tr><td>VLMs</td><td>Qwen2-VL-7B [38]</td><td>×</td><td>token</td><td>8.2 8.3</td><td>0.602</td><td>0.591</td><td>0.494</td><td>15.2</td><td>6.8</td><td>4.4</td><td>8.3</td><td>18.79</td><td>18.65</td><td>6.2</td><td>1.7</td><td>22.0</td></tr><tr><td></td><td>LLaVA-OV-7B [19]</td><td>×</td><td>token</td><td>8.0</td><td>0.641</td><td>0.591</td><td>0.494</td><td>8.7</td><td>1.6</td><td>30.3</td><td>8.0</td><td>11.87</td><td>7.45</td><td>4.1</td><td>0.8</td><td>80.0</td></tr><tr><td rowspan="7">Small</td><td>Qwen2-VL-2B [38]</td><td>×</td><td>token</td><td></td><td>2.2 0.798</td><td>0.777</td><td>0.597</td><td>16.6</td><td>11.1</td><td>1.63</td><td>2.2</td><td>7.30</td><td>6.27</td><td>6.9</td><td>2.6</td><td>8.9</td></tr><tr><td>w/LoRA</td><td>√</td><td>token</td><td>2.22</td><td>0.752 ± 0.061 0.739 ± 0.050 0.595 ± 0.015</td><td></td><td></td><td>16.4</td><td>9.4</td><td>1.65</td><td>2.22</td><td>2.25 ± 0.60</td><td>7.20 ± 0.21</td><td>8.0</td><td>2.6</td><td>9.2</td></tr><tr><td>w/ Geo-Head (Ours)</td><td>√</td><td>geo</td><td></td><td>1.6</td><td>0.251 ± 0.010 0.219 ± 0.008 0.177 ± 0.007</td><td></td><td>21.0</td><td>13.8</td><td>1.2</td><td>1.5</td><td></td><td>12.37 ± 1.16 11.58 ± 0.58</td><td>7.9</td><td>3.0</td><td>6.7</td></tr><tr><td></td><td>×</td><td>token</td><td></td><td>0.777</td><td>0.768</td><td>0.650</td><td>14.0</td><td>6.6</td><td>4.2</td><td>0.9</td><td>1.09</td><td>0.94</td><td>7.7</td><td>3.2</td><td>11.8</td></tr><tr><td>LLaVA-OV-0.5B [19]</td><td>√</td><td>token</td><td></td><td>0.9 0.91</td><td>0.695 ± 0.001 0.700 ± 0.003 0.581 ± 0.002</td><td></td><td>13.6</td><td>6.4</td><td>4.4</td><td>0.91</td><td>2.39 ± 0.11</td><td>2.51 ± 1.10</td><td>7.7</td><td>3.2</td><td>12.0</td></tr><tr><td>w/LoRA w/ Geo-Head (Ours)</td><td>V</td><td>geo</td><td></td><td>0.8</td><td>0.225 ± 0.039 0.225 ± 0.003 0.179 ± 0.003</td><td></td><td>15.3</td><td>7.2</td><td>4.2</td><td>0.7</td><td>9.53 ± 1.44</td><td>6.82 ± 0.24</td><td>7.9</td><td>3.4</td><td>10.8</td></tr></table>

## 4.2.2. Affordance Prediction

Our geometry head yields strong gains on affordance prediction as well. Equipping the small VLMs with our geometry head increases affordance mAP for Qwen2-VL-2B by 1.7× (7.30 to $1 2 . 3 7 \pm 1 . 1 6 )$ and yields larger gains for LLaVA-OneVision-0.5B (1.09 to $9 . 5 3 \pm 1 . 4 4 )$ , indicating that intermediate representations already encode rich spatial cues that token-based decoding fails to exploit. These gains come with equal or better runtime than the corresponding token heads and bring the small-geometry models close to, and in some cases competitive with, much larger 7B baselines, reinforcing our first claim that geometrically compatible prediction heads unlock strong affordance performance from compact VLMs while preserving their efficiency.

LoRA fine-tuning again fails to close the gap and in some cases actively degrades affordance quality relative to the zero-shot token baselines. Despite using the same backbones, LoRA-tuned models do not consistently improve over their frozen counterparts and remain substantially below our geometry-head variants, underscoring that simply pushing the language decoder harder is a poor fit for precise spatial localization.

## 4.2.3. Inference Efficiency

Complementing the accuracy gains above, our geometryhead design preserves and often improves the runtime advantages of compact VLMs over their larger counterparts.

Table 1 reports inference efficiency for trajectory and affordance prediction on both an H100 and an RTX 3090, with the latter representing a lower-compute reference platform. A key structural advantage of our approach is that inference is truncated at the selected intermediate layer rather than executing the full VLM, reducing the active computation and FLOPs without modifying the backbone architecture. We provide a more detailed analysis of the efficiency– performance trade-off across layer indices in Supplementary Sec. E. Importantly, the selected layer l<sup>∗</sup> is backboneand task-dependent; therefore, Qwen and LLaVA skip different fractions of the language-model computation. This leads to different FPS gains across models despite using the same geometric decoding strategy.

Across both tasks, geometry-head models generally achieve the highest throughput, with the 0.5B and 2B backbones attaining the fastest frame rates while operating at a fraction of the parameters and compute of their 7B counterparts. RoboBrain, while achieving the strongest overall performance, runs at substantially lower frame rates on an RTX 3090 (1.2/0.8 FPS), illustrating the substantially greater inference cost of the larger model. Together with the accuracy gains reported above, these results confirm that pairing small VLMs with a geometry-head delivers substantial performance improvements while maintaining superior efficiency over larger VLM counterparts.

## 4.3. Disentangling Prediction Interface, Representation Depth and Decoder Architecture

To disentangle the source of the performance gains, we factorize the design into three components: the prediction interface (autoregressive tokens vs. direct continuous regression), representation depth, and decoder architecture (MLP vs. DETR) (see Table 2).

For trajectory prediction, the prediction interface is the primary source of improvement. Decoding the final frozen VLM representation with a simple continuous MLP reduces DFD from 0.798 to 0.297±0.002 for Qwen and from 0.777 to 0.298±0.002 for LLaVA. This corresponds to 91.6% and 86.8% of the full improvement achieved by our l<sup>∗</sup>-DETR model, showing that most of the trajectory gain does not require either intermediate feature selection or a DETR decoder.

This interpretation is supported further by the LoRA controls. LoRA-AR fine-tunes the VLM with LoRA while retaining autoregressive token generation, with l<sup>∗</sup>-LoRA-AR applying LoRA only to layers after l . LoRA-Reg replaces autoregressive generation with a learned regression token decoded by an MLP into continuous coordinates. As shown in Table 2, LoRA-Reg substantially improves over LoRA-AR for both VLM architectures. Conversely, retaining autoregressive prediction in l<sup>∗</sup>-LoRA-AR remains substantially weaker than the continuous-regression variants. Together, these controls support the prediction interface as the dominant bottleneck for trajectory prediction.

Table 2. Disentangling the source of performance gains. Continuous regression from the final VLM representation already substantially outperforms autoregressive token decoding, while intermediate features and DETR provide additional gains. Pre-LLM controls test performance before language-model contextualization. AR denotes autoregressive decoding and $L T { \mathrel { + } } M L P$ denotes direct regression from a learned token using an MLP. Results are reported as mean ± standard deviation where applicable.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Representation</td><td rowspan="2">Backbone Adaptation</td><td rowspan="2">Decoder</td><td rowspan="2">Output</td><td colspan="2">Qwen2-VL-2B</td><td colspan="2">LLaVA-OV-0.5B</td></tr><tr><td>Traj. DFD↓</td><td>Aff. mAP↑</td><td>Traj. DFD↓</td><td>Aff. mAP↑</td></tr><tr><td>Base VLM</td><td> $\mathrm { F u l l \ L L M }$ </td><td>None</td><td>AR</td><td>Tokens</td><td>0.798</td><td> $7 . 3 0$ </td><td> $_ { 0 . 7 7 7 }$ </td><td>1.09</td></tr><tr><td>LoRA-AR</td><td> $\mathrm { F u l l \ L L M }$ </td><td>LoRA</td><td>AR</td><td>Tokens</td><td> $0 . 7 5 2 \pm 0 . 0 6 1$ </td><td> $2 . 2 5 \pm 0 . 6 0$ </td><td> $0 . 6 9 5 \pm 0 . 0 0 1$ </td><td> $2 . 3 9 \pm 0 . 1 1$ </td></tr><tr><td>l*-LoRA-AR</td><td>Full LLM</td><td>LoRA after  $l ^ { * }$ </td><td>AR</td><td>Tokens</td><td> $0 . 6 8 0 \pm 0 . 0 1 0$ </td><td> $2 . 3 9 \pm 0 . 2 0$ </td><td> $0 . 6 8 0 \pm 0 . 0 3 0$ </td><td> $2 . 0 4 \pm 0 . 2 1$ </td></tr><tr><td>LoRA-Reg</td><td>Final token</td><td>LoRA</td><td>LT+MLP</td><td>Continuous</td><td> $0 . 3 4 \pm 0 . 0 0 8$ </td><td> $3 . 2 4 \pm 0 . 0 6$ </td><td> $0 . 4 3 \pm 0 . 2$ </td><td> $5 . 7 8 \pm 1 . 0 4$ </td></tr><tr><td>l0-MLP</td><td> $l _ { 0 }$ </td><td>Frozen</td><td>MLP</td><td>Continuous</td><td> $0 . 2 9 1 \pm 0 . 0 0 2$ </td><td> $8 . 7 4 \pm 0 . 9 7$ </td><td> $0 . 3 1 1 \pm 0 . 0 0 2$ </td><td> $8 . 6 8 \pm 0 . 1 0$ </td></tr><tr><td> $l _ { 0 } { \mathrm { - D E T R } }$ </td><td> $l _ { 0 }$ </td><td>Frozen</td><td>DETR</td><td>Continuous</td><td> $0 . 2 6 5 \pm 0 . 0 0 2$ </td><td> $1 1 . 9 6 \pm 1 . 1 2$ </td><td> $0 . 2 9 3 \pm 0 . 0 0 1$ </td><td> $9 . 0 3 \pm 1 . 6 0$ </td></tr><tr><td>Final-MLP</td><td>Final</td><td>Frozen</td><td>MLP</td><td>Continuous</td><td> $0 . 2 9 7 \pm 0 . 0 0 2$ </td><td> $7 . 9 2 \pm 0 . 5 4$ </td><td> $0 . 2 9 8 \pm 0 . 0 0 2$ </td><td> $9 . 0 6 \pm 0 . 1 4$ </td></tr><tr><td>Final-DETR</td><td>Final</td><td>Frozen</td><td>DETR</td><td>Continuous</td><td> $0 . 2 8 9 \pm 0 . 0 3 7$ </td><td> $7 . 9 7 \pm 0 . 3 2$ </td><td> $0 . 2 2 5 \pm 0 . 0 3 9$ </td><td> $7 . 9 8 \pm 0 . 1 9$ </td></tr><tr><td>l*-MLP</td><td> $l ^ { * }$ </td><td>Frozen</td><td>MLP</td><td>Continuous</td><td> $0 . 2 6 7 \pm 0 . 0 0 1$ </td><td> $9 . 3 6 \pm 0 . 0 3$ </td><td> $0 . 2 9 8 \pm 0 . 0 0 2$ </td><td> $9 . 1 6 \pm 1 . 3 0$ </td></tr><tr><td>l*-DETR (Ours)</td><td> ${ { l } ^ { * } }$ </td><td>Frozen</td><td>DETR</td><td>Continuous</td><td> $\mathbf { 0 . 2 5 1 \pm 0 . 0 1 0 }$  </td><td> $\mathbf { 1 2 . 3 7 \pm 1 . 1 6 }$ </td><td> $\mathbf { 0 . 2 2 5 \pm 0 . 0 3 9 }$ </td><td> ${ \bf 9 . 5 3 \pm 1 . 4 4 }$ </td></tr></table>

Affordance prediction exhibits more backbone-specific behavior. For LLaVA, Final-MLP already recovers 94.4% of the full mAP improvement, with l<sup>∗</sup>-MLP recovering 95.6%, indicating that direct continuous prediction accounts for most of the gain. For Qwen, however, Final-MLP provides only a modest improvement, while moving the same MLP to $l ^ { * }$ recovers 40.6% of the full gain; replacing the MLP with DETR at the same representation depth provides a further substantial improvement. Thus, for Qwen affordance prediction, representation depth and decoder architecture contribute more strongly alongside the prediction interface. The LoRA controls reinforce this interpretation: replacing autoregressive prediction with continuous regression improves affordance mAP when compared to LoRA-AR, while l<sup>∗</sup>-LoRA-AR is comparable or worse, indicating that the prediction interface is important but does not alone explain the full affordance gains.

Overall, these results support our second claim: autoregressive decoding is the dominant bottleneck for trajectory prediction. For affordance prediction, it is a major bottleneck which can have stronger backbone-dependent interaction with representation depth and decoder architecture.

## 4.4. Accessibility of Spatial Information Across VLM Representations

Having established that direct continuous prediction can expose substantial spatial information from frozen VLM representations, we next examine how the accessibility of this information varies across representation depth. The pre-LLM (l\_0 ) controls in Table 2 demonstrated that strong geometric prediction is already possible from the projected multimodal features before any LLM transformer layer, indicating that the LLM does not create the spatial signal from scratch. We then attach the same geometry head to hidden states at different VLM depths to examine how prediction quality changes across layers.

![](images/8df76af828bb0759fedf03230c1966baf284c9afac4bedbf3f4f997a87399abc.jpg)  
Figure 3. Layer-wise representational analysis. Trajectory DFD on ShareRobot-Bench-Trajectory and affordance mAP on AGD20K across all language-model layers for Qwen2-VL-2B and LLaVA-OV-0.5B, compared against the base VLM and LoRA fine-tuned autoregressive baselines. Colored markers indicate the selected layer used in the final model.

Figure 3 presents a full layer sweep across both backbones and tasks, together with the base VLM and LoRA fine-tuned autoregressive baselines. For each backbone, we select l<sup>∗</sup> independently for each task: the layer with minimum trajectory error (l = 13 for Qwen and l = 20 for LLaVA) and highest affordance mAP (l = 12 for Qwen and l = 23 for LLaVA).

For trajectory prediction, every evaluated hidden layer yields substantially lower DFD than the corresponding autoregressive baselines for both backbones. Together with the strong pre-LLM result, this suggests that trajectoryrelevant spatial information is already broadly accessible throughout the VLM representation hierarchy, but is not effectively exposed by next-token coordinate prediction.

![](images/f4ce7e18633d2a21d78a2b6f8d0496e1ed1797a76ebf4e6d7e1189ea1f8f94db.jpg)  
Figure 4. Qualitative trajectory predictions. Red: predicted, green: ground truth. Top row: vanilla VLM baseline. Bottom row: with our geometry head.

Affordance prediction is more depth-dependent. The geometry head outperforms the autoregressive baselines across most layers, but the magnitude of the improvement varies substantially with depth. Combined with the strong pre-LLM performance, this suggests that affordancerelevant spatial information is already present early in the representation hierarchy, while subsequent layers can make it more or less accessible for geometric decoding.

Overall, compact VLMs contain substantial spatial information from the input to the LLM onward. Its accessibility is relatively stable across depth for trajectory prediction, but more strongly representation-dependent for affordance prediction. Intermediate LLM layers can refine or reorganize this information, but deeper representations are not necessarily better suited to precise geometric prediction.

We additionally ablate two design choices in the geometric head: decoder-query conditioning and explicit start/endpoint supervision. The preferred query representation is backbone- and task-dependent, while endpoint supervision consistently improves trajectory accuracy and reduces collapse toward a dataset-level mean. Full results are provided in Supplementary Sec. F.

## 4.5. Qualitative Results

Figures 4 and 5 complement the quantitative results. For trajectory prediction, vanilla VLMs produce fragmented, spatially inconsistent waypoints, whereas the geometry head yields coherent trajectories with well-anchored endpoints. For affordance prediction, vanilla VLMs often roughly localize the relevant interaction region, suggesting that useful spatial information is already present but not precisely exposed. The geometry head turns these representations into tighter, more coherent predictions across both tasks, visually supporting our core claims. Additional qualitative results are provided in Supplementary Sec. G.

![](images/c1f95a7565db494351d579337b4381a858380efea04d8e77b5e6cb1b377329c8.jpg)  
Figure 5. Qualitative affordance predictions. Red: predicted box, green: ground truth. Top row: vanilla VLM baseline. Bottom row: with our geometry head.

## 5. Conclusion

Compact VLMs of 0.5B–2B parameters already encode useful spatial information across their representation hierarchy, although its accessibility varies with task and model depth. We show that direct continuous prediction is the primary source of improvement for trajectory prediction, while affordance prediction depends more strongly on the interaction between prediction interface, representation depth, and decoder architecture. Building on this, we attach a lightweight geometry-compatible prediction head to a frozen, off-the-shelf VLM and directly regress affordance boxes and trajectory waypoints without modifying the backbone. This interface improves both tasks while preserving compact-VLM efficiency. Our 2B model approaches RoboBrain on trajectory prediction while running 3.8× faster (21 vs. 5.5 FPS) and requiring substantially less computation (1.2 vs. 31 TFLOPs). These results suggest a broader principle for efficient robot perception: before scaling the backbone, first ensure that the prediction interface is aligned with the geometric structure of the task.

Three limitations motivate future work. First, affordance and trajectory are predicted with separate heads; a unified decoder could better exploit their geometric interdependence. Second, early exit reduces active computation but still requires storing the backbone up to l<sup>∗</sup>; quantization or distillation into a truncated model could further reduce deployment cost. Third, our evaluation is limited to instruction-conditioned 2D geometric prediction prior to execution. Future work should extend the approach to 3D and closed-loop robotic settings and evaluate downstream manipulation success.

## Acknowledgment

The authors would like to acknowledge support from the CSIRO’s Data61 Embodied AI Cluster and partial support from the Australian Research Council (ARC) Discovery Project Grant DP250103634.

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 2

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 2

[3] Paweł Budzianowski, Wesley Maa, Matthew Freed, Jingxiang Mo, Winston Hsiao, Aaron Xie, Tomasz Młoduchowski, Viraj Tipnis, and Benjamin Bolte. Edgevla: Efficient visionlanguage-action models. arXiv preprint arXiv:2507.14049, 2025. 2

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 3

[5] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024. 2

[6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 1, 2

[7] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 91–104, 2025. 2

[8] Sivan Doveh, Nimrod Shabtay, Eli Schwartz, Hilde Kuehne, Raja Giryes, Rogerio Feris, Leonid Karlinsky, James Glass, Assaf Arbelle, Shimon Ullman, et al. Teaching vlms to localize specific objects from in-context examples. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9572–9582, 2025. 2

[9] Danny Driess, Jost Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, et al. Knowledge insulating vision-language-action models: Train fast, run fast, generalize better. Advances in Neural Information Processing Systems, 38:102867–102888, 2026. 2

[10] Mohsen Gholami, Ahmad Rezaei, Zhou Weimin, Sitong Mao, Shunbo Zhou, Yong Zhang, and Mohammad Akbari. Spatial reasoning with vision-language models in ego-centric

multi-view scenes. arXiv preprint arXiv:2509.06266, 2025. 1

[11] Utkarsh Grover, Ravi Ranjan, Mingyang Mao, Trung Tien Dong, Satvik Praveen, Zhenqi Wu, J Morris Chang, Tinoosh Mohsenin, Yi Sheng, Agoritsa Polyzou, et al. Embodied foundation models at the edge: A survey of deploy ment constraints and mitigation strategies. arXiv preprint arXiv:2603.16952, 2026. 2

[12] Jiayuan Gu, Sean Kirmani, Paul Wohlhart, Yao Lu, Montserrat Gonzalez Arenas, Kanishka Rao, Wenhao Yu, Chuyuan Fu, Keerthana Gopalakrishnan, Zhuo Xu, et al. Rt-trajectory: Robotic task generalization via hindsight trajectory sketches. arXiv preprint arXiv:2311.01977, 2023. 5

[13] Asher J Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, and Anirudha Majumdar. Actions as language: Fine-tuning vlms into vlas without catastrophic forgetting. arXiv preprint arXiv:2509.22195, 2025. 2

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. 2021. 5

[15] Siyuan Huang, Iaroslav Ponomarenko, Zhengkai Jiang, Xiaoqi Li, Xiaobin Hu, Peng Gao, Hongsheng Li, and Hao Dong. Manipvqa: Injecting robotic affordance and physically grounded information into multi-modal large language models. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7580–7587. IEEE, 2024. 2

[16] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1724–1734, 2025. 2, 5, 6, 1

[17] Siddharth Karamcheti, Suraj Nair, Annie S Chen, Thomas Kollar, Chelsea Finn, Dorsa Sadigh, and Percy Liang. Language-driven representation learning for robotics. arXiv preprint arXiv:2302.12766, 2023. 2

[18] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025. 2

[19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 4, 6

[20] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 1, 2

[21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PMLR, 2023. 1

[22] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5

[23] Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, and Dacheng Tao. Learning affordance grounding from exocentric images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2252–2261, 2022. 5

[24] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. Advances in neural information processing systems, 37:68803–68832, 2024. 2

[25] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022. 2

[26] Dantong Niu, Yuvan Sharma, Giscard Biamby, Jerome Quenum, Yutong Bai, Baifeng Shi, Trevor Darrell, and Roei Herzig. Llarva: Vision-action instruction tuning enhances robot learning. arXiv preprint arXiv:2406.11815, 2024. 2

[27] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 2, 5

[28] Shengyi Qian, Weifeng Chen, Min Bai, Xiong Zhou, Zhuowen Tu, and Li Erran Li. Affordancellm: Grounding affordance from vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7587–7597, 2024. 2

[29] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 658–666, 2019. 4, 1

[30] Timothy Schaumloffel, Martina G Vilas, and Gemma Roig.¨ Mechanisms of object localization in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 31356–31365, 2026. 2

[31] Surya Selvam, Ravi K Rajendran, Murugan Sankaradas, Anand Raghunathan, and Srimat T Chakradhar. Simcache: similarity caching for efficient vlm-based scene understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3327–3336, 2025. 1

[32] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025. 2

[33] Sumedh Sontakke, Jesse Zhang, Seb Arnold, Karl Pertsch,´ Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, and Laurent Itti. Roboclip: One demonstration is enough to learn robot policies. Advances in Neural Information Processing Systems, 36:55681–55693, 2023. 2

[34] Roniel S De Sousa, Azzedine Boukerche, and Antonio AF Loureiro. Vehicle trajectory similarity: Models, methods,

and applications. ACM Computing Surveys (CSUR), 53(5): 1–32, 2020. 5

[35] Qi Sun, Pengfei Hong, Tej Deep Pala, Vernon Toh, U-Xuan Tan, Deepanway Ghosal, and Soujanya Poria. Emma-x: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Compu tational Linguistics (Volume 1: Long Papers), pages 14199– 14214, 2025. 2

[36] Huajie Tan, Xiaoshuai Hao, Cheng Chi, Minglan Lin, Yaoxu Lyu, Mingyu Cao, Dong Liang, Zhuo Chen, Mengsi Lyu, Cheng Peng, et al. Roboos: A hierarchical embodied framework for cross-embodiment and multi-agent collaboration. arXiv preprint arXiv:2505.03673, 2025. 2, 5

[37] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 2

[38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 4, 6

[39] Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan Zhang, Ji Zhang, Zheng Wang, Lianli Gao, Jingkuan Song, Nicu Sebe, and Heng Tao Shen. A survey on efficient vision-languageaction models. arXiv preprint arXiv:2510.24795, 2025. 2

[40] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024. 2

[41] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 2

[42] Xiaohan Zhang, Yan Ding, Saeid Amiri, Hao Yang, Andy Kaminski, Chad Esselink, and Shiqi Zhang. Grounding classical task planners via vision-language models. arXiv preprint arXiv:2304.08587, 2023. 2

[43] Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, et al. A survey on visionlanguage-action models: An action tokenization perspective. arXiv preprint arXiv:2507.01925, 2025. 2

[44] Kaijun Zhou, Qiwei Chen, Da Peng, Zhiyang Li, Xijun Li, and Jinyu Gu. Characterizing vision-language-action mod els across xpus: Constraints and acceleration for on-robot deployment. arXiv preprint arXiv:2604.24447, 2026. 2

[45] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2

# Decoding Geometry, Not Tokens: Efficient VLMs for Robot Affordance and Trajectory Prediction

Supplementary Material

## A. Training

## A.1. Affordance Loss

Smooth L1 coordinate loss. Given predicted and groundtruth boxes b<sup>ˆ</sup>, $, \mathbf { b } \in [ 0 , 1 ] ^ { 4 }$ , the coordinate regression term is

$$
\mathcal { L } _ { \mathrm { L 1 } } ( \hat { \mathbf { b } } , \mathbf { b } ) = \frac { 1 } { 4 } \sum _ { i = 1 } ^ { 4 } \ell _ { \beta } \Big ( \hat { b } _ { i } - b _ { i } \Big ) ,\tag{6}
$$

where the Smooth L1 function $\ell _ { \beta }$ is defined as

$$
\ell _ { \beta } ( x ) = \left\{ { \begin{array} { l l } { \displaystyle { \frac { x ^ { 2 } } { 2 \beta } } } & { { \mathrm { i f ~ } } | x | < \beta , } \\ { | x | - { \frac { \beta } { 2 } } } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{7}
$$

We set $\beta \ : = \ : 0 . 0 5$ , making the loss substantially more sensitive to small coordinate errors than the standard $\beta = 1$ formulation; this is important for precise affordance localization in manipulation tasks.

GIoU overlap loss. The overlap term is the Generalized IoU loss [29]:

$$
\mathcal { L } _ { \mathrm { G I o U } } ( \hat { \mathbf { b } } , \mathbf { b } ) = 1 - \left[ \mathrm { I o U } ( \hat { \mathbf { b } } , \mathbf { b } ) - \frac { \vert \mathcal { C } \setminus ( \hat { \mathbf { b } } \cup \mathbf { b } ) \vert } { \vert \mathcal { C } \vert } \right] ,\tag{8}
$$

where C is the smallest axis-aligned enclosing box of b<sup>ˆ</sup> and b.

## A.2. Trajectory Loss

Given predicted waypoints $\hat { \mathcal { T } } = \{ \hat { \mathbf { p } } _ { t } \} _ { t = 1 } ^ { T }$ and ground truth $\mathcal { T } = \{ { \bf p } _ { t } \} _ { t = 1 } ^ { T }$ , with $\hat { \mathbf p } _ { t } , \mathbf p _ { t } \in [ 0 , 1 ] ^ { 2 }$ , the same Smooth L1 is uniformly applied over all T waypoints, with $\beta = 0 . 0 5$ making the loss more sensitive to small coordinate errors than the standard $\beta = 1$ formulation:

$$
\mathcal { L } _ { \mathrm { L 1 } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \ell _ { \beta } \big ( \hat { \mathbf { p } } _ { t } - \mathbf { p } _ { t } \big ) ,\tag{9}
$$

Because the first and last waypoints determine where contact is made and where the manipulation terminates, we add explicit endpoint penalties:

$$
{ \mathcal { L } } _ { \mathrm { s t a r t } } = \ell _ { \beta } \bigl ( { \hat { \mathbf { p } } } _ { 1 } - \mathbf { p } _ { 1 } \bigr ) , \qquad { \mathcal { L } } _ { \mathrm { e n d } } = \ell _ { \beta } \bigl ( { \hat { \mathbf { p } } } _ { T } - \mathbf { p } _ { T } \bigr ) ,\tag{10}
$$

which upweight the contribution of the start and end of the trajectory relative to intermediate waypoints.

## B. AGD20K Affordance Box Construction

AGD20K does not provide ground-truth bounding boxes for affordance prediction. We therefore construct box annotations from the evaluation-set affordance masks, similar to [16]. For each binary mask, Connected Component Analysis (CCA) identifies contiguous non-zero regions, from which bounding boxes are generated. Box extent is controlled by a threshold during mask-to-box conversion, with larger values producing tighter boxes. We use a threshold of 0.7, which produced stable annotations across the evaluation set. These boxes serve as ground-truth labels and are evaluated using the same bounding-box protocol as in the main paper.

## C. Inference Efficiency Protocol

We measure inference efficiency on NVIDIA H100 and RTX 3090 GPUs with batch size 1. Qwen2- VL uses dynamic-resolution preprocessing, while LLaVA-OneVision uses aspect-ratio-preserving resizing and highresolution image-grid processing; the same preprocessing is used across variants of each backbone. Model loading is excluded, and FPS is measured from preprocessed inputs to final predictions after 20 warm-up iterations, averaged over 1000 samples.

FLOPs are computed with PyTorch Profiler over 10 samples using the same input resolution and model configuration. For early-exit variants, FLOPs include computation through l and the geometric head, excluding later layers. All models follow the same profiling procedure.

## D. Implementation Details

The LoRA baselines use rank 16 with $\alpha = 8$ and are applied to all linear projection layers in the LLM. They contain 18.46M and 8.80M trainable parameters for Qwen2- VL-2B and LLaVA-OV-0.5B, respectively, compared with 4.08M for the geometric head. Training to convergence on a single NVIDIA H100 GPU required approximately 15 h for LoRA, compared with 4 h for the geometric head.

## E. Efficiency–Performance Trade-off Across Exit Depth

Figure 6 shows the trajectory accuracy–efficiency trade-off for Qwen2-VL-2B across different exit layers. Earlier exits increase throughput, while deeper exits reduce FPS without consistently improving accuracy. The best DFD is obtained at $l ^ { * } = 1 3$ , which retains substantially higher throughput than the deepest exits. These results show that intermediatelayer decoding is not only beneficial for representation quality, but also provides a favorable accuracy–efficiency tradeoff by avoiding unnecessary later-layer computation.

![](images/554deb3535d7638a95aaed52056bef2528d83d5faeb055f631509ba3a5c67e96.jpg)  
Figure 6. Efficiency–performance trade-off across layers on Qwen2-VL-2B for the trajectory prediction task.

Table 3. Ablation of query token selection for trajectory prediction on ShareRobot-Bench-Trajectory and affordance prediction on AGD20K.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Query</td><td colspan="3">Trajectory</td><td rowspan="2">Affordance</td></tr><tr><td>DFD↓</td><td>HD↓</td><td>RMSE↓</td></tr><tr><td rowspan="2">Qwen2-VL-2B</td><td>Last token</td><td> $0 . 2 5 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 8$ </td><td> $0 . 1 7 7 \pm 0 . 0 0 7$ </td><td>mAP↑  $1 2 . 3 7 \pm 1 . 1 6$ </td></tr><tr><td>Attn. pool</td><td> $0 . 2 1 5 \pm 0 . 0 0 2$ </td><td> $0 . 1 9 4 \pm 0 . 0 0 1$ </td><td> $0 . 1 5 5 \pm 0 . 0 0 2$ </td><td> $8 . 6 7 \pm 1 . 0 1$ </td></tr><tr><td rowspan="2">LLaVA-OV-0.5B</td><td>Last token</td><td> $0 . 2 2 5 \pm 0 . 0 3 9$ </td><td> $0 . 2 2 5 \pm 0 . 0 0 3$ </td><td> $0 . 1 7 9 \pm 0 . 0 0 3$ </td><td> $9 . 5 3 \pm 1 . 4 4$ </td></tr><tr><td>Attn. pool</td><td> $0 . 2 5 6 \pm 0 . 0 0 4$ </td><td> $0 . 2 4 \pm 0 . 0 1$ </td><td> $0 . 1 8 9 \pm 0 . 0 0 2$ </td><td> $7 . 2 7 \pm 1 . 2$ </td></tr></table>

Table 4. Ablation on start and end point supervision for trajectory prediction head (Qwen2-VL-2B, ShareRobot-Bench).
<table><tr><td>Model</td><td>DFD↓</td><td>HD↓</td><td>RMSE↓</td></tr><tr><td>w/ start &amp; end point loss</td><td> $0 . 2 5 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 8$ </td><td> $0 . 1 7 7 \pm 0 . 0 0 7$ </td></tr><tr><td>w/ only overall point loss</td><td> $0 . 3 2 2 \pm 0 . 0 2$ </td><td> $0 . 2 6 \pm 0 . 0 1$ </td><td> $0 . 2 6 \pm 0 . 0 3$ </td></tr></table>

## F. Additional Ablation Studies

## F.1. Query Conditioning Strategy

We compare two strategies for conditioning the DETR decoder query: the hidden state of the final text token at layer l<sup>∗</sup>, and attention pooling over task instruction tokens. As shown in Table 3, the impact differs across architectures: attention pooling improves trajectory prediction on Qwen2-VL-2B but degrades LLaVA-OV-0.5B, while the final text token is consistently stronger for affordance prediction across both backbones. To maintain a consistent decoder design across architectures and tasks, we use the final text token as the query-conditioning representation in all main-paper experiments.

## F.2. Trajectory Loss Components

Table 4 shows that explicitly supervising the start and end points of predicted trajectories yields consistent improvements across all metrics (DFD: $0 . 2 5 1 \pm 0 . 0 1 0 \ \mathrm { v s }$ $0 . 3 2 2 \pm 0 . 0 2$ , HD: $0 . 2 1 9 \pm 0 . 0 0 8$ vs. $0 . 2 6 \pm 0 . 0 1$ , RMSE: $0 . 1 7 7 \pm 0 . 0 0 7$ vs. $0 . 2 6 \pm 0 . 0 3 )$ . Without these endpoint terms, the geometry head tends to collapse to a datasetlevel mean trajectory, whereas adding start/end supervision restores spatially diverse, input-dependent predictions. This improvement shows that the decoding objective strongly influences precise geometric prediction even with a frozen backbone.

![](images/5d538e21b659f6bfeb164eb0c2b5f4eaf8020854006b0d4bb15d38eec7314f6f.jpg)  
(a)

![](images/01fc296886cf3463145068189cb2c330f21baa05c1c28ffc6f840af674bed5dc.jpg)

![](images/a0bf2ab5842c9241bba085555aea6aa91aa2c98ebd5869091f4e0ff642ac911d.jpg)  
(c)

(b)  
![](images/6b7fb98bb717d81ab66153afa99df79ec2f313ef3d386abffed08b750ddd9658.jpg)  
(d)  
Figure 7. Predicted waypoint heatmaps over the test set, visualizing how endpoint supervision shapes the spatial distribution of trajectory predictions. (a) Ground-truth distribution: waypoints are broadly spread, reflecting the spatial diversity of manipulation trajectories in the dataset. (b) Zero-shot Qwen2-VL-2B baseline: predictions collapse to the image boundaries, confirming that vanilla autoregressive generation fails to produce spatially meaningful trajectories. (c) Geometry head without endpoint supervision: predictions collapse to a single central region, as the uniform L1 objective is minimised by regressing toward the dataset mean. (d) Geometry head with endpoint supervision: the distribution recovers spatial diversity matching the ground truth, demonstrating that boundary supervision forces input-specific localization rather than dataset-level averaging.

The heatmaps in Figure 7 illustrate how endpoint supervision shapes spatial prediction behavior. Without it, the geometry head collapses to the image center — the uniform L1 objective is minimized by regressing toward the dataset mean. Adding explicit start and end point supervision breaks this collapse and recovers the spatially diverse distribution seen in the ground truth.

## G. Qualitative Results

Figures 8–11 show additional trajectory and affordance predictions for both Qwen2-VL-2B and LLaVA-OV-0.5B across a diverse set of manipulation scenarios.

![](images/2260c36b98c0ef27689f67d185c6e12527a5267a1ac178d3ce686f8b47759664.jpg)  
Figure 8. Additional visualizations of diverse trajectories predicted by Qwen2-VL-2B + Geo-Head. Green curve represents the ground-truth and the red curve represents the prediction. The visualizations in the first three rows demonstrate that our geometry head effectively generates end-effector manipulation trajectories based on the observation and task instruction. The fourth row presents some failure cases arising from object-localization errors or insufficient spatial understanding.

![](images/fa9930b8facf39452a28730119999cc846a2f89f2223a0dacbb52c5117cacab7.jpg)  
Figure 9. Additional visualizations of diverse trajectories predicted by LLaVA-OV-0.5B + Geo-Head. Green curve represents the ground-truth and the red curve represents the prediction. The visualizations in the first three rows demonstrate that our geometry head effectively generates end-effector manipulation trajectories based on the observation and task instruction. The fourth row presents some failure cases.

![](images/d40305e4aad869e56628ba186dff99514ceec7ecb5255b8a06fdd4163eefc2fe.jpg)  
Figure 10. Additional visualizations of diverse affordance regions predicted by Qwen2-VL-2B + Geo-Head. Green box represents the ground-truth and the red box represents the prediction. The visualizations in the first three rows demonstrate that our geometry head effectively generates affordance boxes based on the observation and task instruction. The fourth row presents some failure cases.

![](images/3a5589d51d94df93de30167cc5b57b2881ef07a5053b2a45b1c6e5c0fda03688.jpg)  
Drink from cup

![](images/e9d3d225f02c7c0b8618ebf714e0afd28fb50fa784cd536131accd1fa2fbe8a7.jpg)  
Catch rugby ball

![](images/f36348d5b0fadce7ed2c2df2cd0739da1ebb0d959d7b420d4a3df744b8b2156b.jpg)  
Lie on bench

![](images/0262605352c90fb0051e395bfddfc7eb64227ef71fb1d44d1ca251db680be0f3.jpg)

![](images/c07803f811b4279a03ad579d75105ba61888a683bfd2f7d3d67c8e80e2513323.jpg)  
Pick up skis

![](images/c7a24bbad1764d088cd371217c116fc15600f38e3db53e27e12157b94d5f0595.jpg)  
Beat drum

![](images/91dcaa79304d9d1441271aaea83e44dc4297dda7210ed7b564a440553e9e8ddb.jpg)

![](images/a809e2b55a3cbb07da478be4dd671532b34cf4bdddd18050a410631fd078e7b7.jpg)  
Hold knife

![](images/187be3341a65a5ec010a399caaa12fdfb747e104723f8f75806965db354466c7.jpg)  
Wash glass

![](images/31246c57737212e95b970602127c2825e439c9bcb5e31757e6a9c15979c904b1.jpg)  
Drag bag

![](images/aa0f47e57ceefc64b26a31036ac4d6d5bd13ecec79f9cca7a35ff327f0f8c745.jpg)  
Hold fork

![](images/67600e07256fb09c305eae51f09065500e9ccadbc5ca78fec7f2265aeb43b8fe.jpg)  
Hold scissors  
Figure 11. Additional visualizations of diverse affordance regions predicted by LLaVA-OV-0.5B + Geo-Head. Green box represents the ground-truth and the red box represents the prediction. The visualizations in the first three rows demonstrate that our geometry head effectively generates affordance boxes based on the observation and task instruction. The fourth row presents some failure cases.