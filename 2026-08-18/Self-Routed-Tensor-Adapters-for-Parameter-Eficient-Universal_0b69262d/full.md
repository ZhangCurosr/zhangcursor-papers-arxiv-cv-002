# Self-Routed Tensor Adapters for Parameter-Eficient Universal Visual Adaptation

Suraj Yadav

Indraprastha Institute of Information Technology Delhi suraj24098@iiitd.ac.in

Abstract. Universal visual representations require adaptation mechanisms that adapt across heterogeneous domains without fragmenting knowledge into domain-specific modules. Parameter-eficient fine-tuning adapts frozen visual foundation models eficiently, but standard low-rank adapters use a fixed subspace for all inputs, which can be restrictive when domains difer in style, background, and semantic context. MoE-based adapters improve specialization through multiple expert pathways, but often rely on external routers and large expert banks, adding parameters and separating routing from adaptation. We propose Self-Routed Tensor Adapters, a compact framework for multi-domain visual adaptation. SRTA projects each input into a low-rank space, computes routing weights from this representation using a learnable domain matrix, and uses these weights to blend slices of a shared Tucker core. This produces a sample-specific adaptation matrix without an external gating network, allowing shared visual factors to be reused while supporting domainaware specialization. To strengthen pathway learning, we introduce a progressive depth-weighted routing objective that supervises routing decisions across adapter layers. Across five heterogeneous multi-domain visual classification benchmarks, SRTA achieves competitive or slightly stronger average accuracy than MoE-style PEFT baselines while using substantially fewer trainable parameters. At rank 64, SRTA uses 2.77M parameters in the 4-domain setting compared with 9.52M for MoLoRA, and 3.00M in the 6-domain setting compared with 14.31M. Overall, SRTA ofers an efective accuracy-parameter trade-of for adapting visual foundation models toward universal multi-domain representations. GitHub

## 1 Introduction

Building universal visual representations [2, 16] requires models that remain effective across heterogeneous domains, tasks, and distribution shifts. In practice, however, adapting a large pre-trained vision backbone to multiple downstream domains remains challenging. Diferent domains may vary in visual style, background context, acquisition conditions, and object appearance. A useful adaptation mechanism should therefore preserve the reusable knowledge of the frozen foundation model while allowing compact, input-dependent specialization.

Parameter-eficient fine-tuning has emerged as a practical approach for adapting large pre-trained models while keeping the backbone frozen. LoRA [4] and related methods [5, 8, 15] learn compact low-rank updates, but they usually apply the same adaptation subspace to every input. In heterogeneous multi-domain settings, this fixed subspace can be too restrictive: some domains require shared factors, while others require domain-specific adaptation. This may lead to negative transfer and uneven optimization across domains.

Recent work has addressed this limitation by combining PEFT with Mixtureof-Experts (MoE) architectures [17]. MoE-based adapters, such as MoLoRA [22], MOELoRA [14], and MALoRA [21], improve specialization by assigning inputs to multiple low-rank expert pathways. However, these methods typically compute routing probabilities using a separate gating network [14, 17, 22]. This introduces additional routing-specific parameters and separates expert selection from the low-rank representation used for adaptation. As a result, routing decisions may become decoupled from the geometry of the learned adapter features, leading to redundant experts, unstable routing behavior, and ineficient sharing across related domains. Motivated by this limitation, we ask whether routing can instead be derived directly from the adapter representation itself.

In this paper, we study multi-domain visual adaptation from the perspective of universal representation learning. Instead of treating each domain as requiring an isolated expert, we ask whether heterogeneous domains can be organized within a shared, structured adaptation space. To this end, we propose Self-Routed Tensor Adapters (SRTA), a parameter-eficient adaptation framework that formulates the update space as a Tucker-decomposed tensor [7, 19]. SRTA replaces discrete externally routed experts with a shared core tensor whose domain-indexed slices are dynamically blended for each input.

The key idea behind SRTA is intrinsic domain-aware routing. Rather than introducing a separate gating network, SRTA derives routing probabilities directly from the interaction between the adapter’s low-rank input projection and a learnable domain-coordinate matrix. This allows the adapter to route itself using the same representation that is used for adaptation. From a representationlearning perspective, the Tucker core serves as a shared adaptation basis, while the domain-coordinate matrix provides domain-aware coordinates over it. As a result, SRTA can reuse common visual factors across domains while softly specializing to domain-specific variations.

To encourage the learned pathways to remain both reusable and discriminative, we introduce a structural training objective. Specifically, we apply progressive depth-weighted routing supervision [9] across adapter layers, providing direct learning signals to intermediate routing decisions and mitigating weak supervision in early layers. This objective helps reduce domain interference while preserving a compact shared adaptation space.

Our contributions are summarized as follows.

– Self-routed tensor adapters. We introduce SRTA, a parameter-eficient adapter that computes routing probabilities directly from the adapter’s lowrank representation, removing the need for a separate MoE gating network.

– Input-conditioned Tucker adaptation. We represent the adaptation space using a shared Tucker core tensor. For each input, SRTA dynamically blends core slices to produce a sample-specific low-rank adaptation matrix.

– Progressive routing supervision. We introduce a progressive depth weighted routing loss that improves domain-aware pathway learning across adapter layers.

Accuracy-parameter trade-of. Across five multi-domain visual classification benchmarks, SRTA achieves competitive or slightly better average accuracy than MoE-style PEFT baselines while requiring substantially fewer parameters than MoLoRA.

## 2 Related Work

## 2.1 Universal Visual Representations and Multi-Domain Adaptation

A central goal of modern foundation models is to learn representations that remain transferable across tasks, domains, and distribution shifts. In visual recognition, this is closely related to domain generalisation, where models are trained on multiple source domains and expected to perform well under heterogeneous or unseen domain conditions. Benchmarks such as PACS [11], VLCS [18], Ofice-Home [20], Digits-DG [24], and NICO++ [23] expose this challenge through variations in style, object appearance, background, acquisition conditions, and visual context. Although large pre-trained vision transformers provide strong general-purpose features [2], adapting them to multiple downstream domains remains non-trivial: full fine-tuning can overwrite useful pre-trained knowledge, while a single shared adaptation module may be insuficient for domains with conflicting visual statistics. This motivates adaptation mechanisms that preserve reusable representations while allowing controlled domain-specific specialization. Our work follows this direction by introducing a self-routed Tucker-decomposed adaptation space that blends shared and domain-aware factors.

## 2.2 Parameter-Eficient Fine-Tuning

Parameter-Eficient Fine-Tuning (PEFT) adapts large pre-trained models by updating only a small subset of parameters while keeping the backbone frozen. Prompt-based methods, such as Prefix-Tuning [12] and Prompt-Tuning [10], optimize continuous prompts, while Low-Rank Adaptation (LoRA) [4] learns trainable low-rank updates to frozen weight matrices and has become a widely used PEFT technique. Several extensions improve eficiency or adaptation quality: DoRA [15] decomposes updates into magnitude and direction components, while VeRA [8] and VB-LoRA [13] reduce trainable parameters through shared or reused adaptation components. However, most PEFT methods rely on static adaptation spaces, which can be limiting in multi-domain settings where a single low-rank update may sufer from negative transfer and domain interference. In contrast, our method introduces an input-dependent self-routing mechanism that dynamically blends adaptation pathways while remaining parameter-eficient.

## 2.3 Mixture-of-Experts for Multi-Domain Adaptation

Mixture-of-Experts (MoE) [3] architectures improve multi-domain adaptation by assigning inputs to specialized expert modules. In PEFT, MoE-based adapters combine low-rank eficiency with expert specialization: MoLoRA [22] extends LoRA with multiple low-rank experts and a routing mechanism, MOELoRA [14] introduces task-conditioned gating to combine LoRA experts for multi-task adaptation, and MALoRA [21] improves parameter eficiency by sharing parts of the low-rank structure across experts. Despite their efectiveness, these methods commonly rely on external gating networks to compute routing probabilities, creating a separation between the adapter representation space and the routing mechanism. This decoupling can add parameters, increase computation, and lead to redundant expert assignments. Our work removes the external router and derives routing probabilities directly from the low-rank adapter representation through its interaction with a learnable domain-coordinate matrix, aligning expert blending with the representation geometry used for adaptation.

## 2.4 Tensor Decompositions and Factorized Adapters

Tensor decompositions provide a principled way to represent high-dimensional parameter spaces using compact multilinear factors. Classical decompositions such as CP [1] and Tucker decomposition [7, 19] have been widely used for compression, factor analysis, and eficient parameterization. In neural adaptation, tensor and factorized methods have also been explored to reduce fine-tuning cost; for example, FacT [6] applies tensor-based factorization to lightweight adaptation in vision transformers. Most tensor-based adaptation methods use decomposition primarily as a compression mechanism. In contrast, our work uses a Tucker-decomposed parameter space as a dynamic routing structure, where the shared core tensor captures reusable adaptation factors and the domaincoordinate matrix defines domain-aware coordinates for blending them. This enables both shared and domain-specific structure to be represented within a single compact adaptation module.

## 3 Methodology

## 3.1 Problem Setup

We consider multi-domain parameter-eficient adaptation of a frozen pre-trained vision model. Let $W _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ <sup>×dout</sup> denote a frozen weight matrix in a transformer layer, and let $\boldsymbol { x } \in \mathbb { R } ^ { B \times T \times d _ { \mathrm { i n } } }$ be an input sequence with batch size B and sequence length T. During training, each sample is associated with a class label $y ^ { * }$ and a domain label $d ^ { * }$ . The class label supervises the main recognition objective, while the domain label is used only as auxiliary supervision for routing. At inference time, SRTA does not require the domain label; routing probabilities are computed directly from the input representation.

The goal is to learn a compact input-conditioned adaptation module that supports heterogeneous domains while preserving a shared representation space. In standard Low-Rank Adaptation (LoRA), the update is parameterized as

$$
\varDelta W = A B ,\tag{1}
$$

where $A \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times r }$ and $B \in \mathbb { R } ^ { r \times d _ { \mathrm { o u t } } }$ are trainable low-rank matrices. The adapted layer output is then

$$
y = x W _ { 0 } + s ( x A B ) ,\tag{2}
$$

where s is a scaling factor. While this static low-rank update is eficient, it uses the same adaptation subspace for all inputs. This can be restrictive in heterogeneous multi-domain settings, where diferent domains may require partially shared but also domain-specific adaptation pathways.

MoE-based LoRA methods address this limitation by introducing N low-rank experts:

$$
y = x W _ { 0 } + s \sum _ { t = 1 } ^ { N } \alpha _ { t } ( x A _ { t } B _ { t } ) ,\tag{3}
$$

where $\alpha _ { t }$ denotes the routing probability for expert t. These routing probabilities are usually produced by an external gating network. Although this enables input-dependent specialization, the router is separated from the adapter representation itself. This introduces additional routing-specific parameters and may decouple expert selection from the low-rank representation being adapted.

## 3.2 Self-Routed Tensor Adapters

To avoid external routing while still enabling domain-aware specialization, we propose Self-Routed Tensor Adapters (SRTA). The key idea is to formulate the adaptation space as a Tucker-decomposed tensor and derive routing probabilities intrinsically from the adapter’s own low-rank representation.

Instead of representing the update using a single low-rank matrix product, SRTA introduces four trainable components:

– an input projection matrix $V \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times r _ { 1 } }$ ，

– an output projection matrix $U \in \mathbb { R } ^ { r _ { 2 } \times d _ { \mathrm { o u t } } }$

– a shared Tucker core tensor $\mathcal { G } \in \mathbb { R } ^ { r _ { 3 } \times r _ { 1 } \times r _ { 2 } }$ ，

– a domain-coordinate matrix $C \in \mathbb { R } ^ { r _ { 1 } \times r _ { 3 } }$

Here, $r _ { 1 }$ and $r _ { 2 }$ are the low-rank input and output dimensions, while $r _ { 3 }$ denotes the number of domain-aware adaptation pathways. In our experiments, $r _ { 3 }$ is set to the number of domains in the benchmark. The columns of $C$ act as learnable pathway coordinates in the low-rank adapter space.

![](images/5835ffadd88c8eed6b9c750aeb7b639b0e654880e5307093a439d1090f0ebf9f.jpg)  
Fig. 1: Overview of the Self-Routed Tensor Adapter pipeline. The input projection V and domain-coordinate matrix C generate intrinsic routing probabilities, which are used to blend slices of the shared Tucker core tensor ${ \mathcal { G } } .$ . This removes the need for an external MoE gating network.

From a representation-learning perspective, the shared core tensor $\mathcal { G }$ stores reusable adaptation factors, while the domain-coordinate matrix $C$ provides coordinates for routing over this shared basis. This allows SRTA to softly combine shared and domain-specific factors instead of allocating each input to a fully isolated expert.

## 3.3 Intrinsic Self-Routing

Given an input sequence $x ,$ SRTA first projects it into the low-rank adapter space:

$$
z = x V , \qquad z \in \mathbb { R } ^ { B \times T \times r _ { 1 } } .\tag{4}
$$

Instead of relying only on the global classification token, we compute a routing signature by averaging the projected patch-token representations. Using zerobased indexing, we treat token 0 as the [CLS] token, we omit this token and aggregate over the remaining patch tokens:

$$
z _ { \mathrm { r o u t e } } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } z [ : , i , : ] , \qquad z _ { \mathrm { r o u t e } } \in \mathbb { R } ^ { B \times r _ { 1 } } .\tag{5}
$$

This routing signature encourages routing decisions to depend on distributed patch-level visual evidence rather than a single global token representation.

Routing logits are then computed by comparing the pooled low-rank representation with the learnable domain-coordinate matrix:

$$
q = \frac { z _ { \mathrm { r o u t e } } C } { \tau } , \qquad q \in \mathbb { R } ^ { B \times r _ { 3 } } ,\tag{6}
$$

where $\tau$ is a temperature hyperparameter that controls the sharpness of the routing distribution. The routing probabilities are obtained using a softmax function:

$$
\alpha = \mathrm { S o f t m a x } ( q ) , \qquad \alpha \in \mathbb { R } ^ { B \times r _ { 3 } } .\tag{7}
$$

Unlike MoE-based adapters, SRTA does not introduce a separate gating network. The routing probabilities are computed from the same low-rank representation that is used for adaptation. Therefore, routing and feature adaptation are coupled within the same representation space.

## 3.4 Dynamic Tucker Core Blending

The routing probabilities α are used to dynamically blend the domain-pathway slices of the shared Tucker core tensor. Let $\mathcal { G } _ { t } \in \mathbb { R } ^ { r _ { 1 } \times r _ { 2 } }$ denote the t-th slice of the core tensor. For each input sample b, SRTA constructs a sample-specific core matrix:

$$
\begin{array} { r } { \boldsymbol { \Sigma } _ { b } ( \boldsymbol { x } ) = \displaystyle \sum _ { t = 1 } ^ { r _ { 3 } } \alpha _ { b , t } \boldsymbol { \mathcal { G } } _ { t } , \qquad \boldsymbol { \Sigma } _ { b } ( \boldsymbol { x } ) \in \mathbb { R } ^ { r _ { 1 } \times r _ { 2 } } . } \end{array}\tag{8}
$$

The adapter output for sample b is then computed as

$$
y _ { \mathrm { a d a p t } , b } = ( z _ { b } \Sigma _ { b } ( x ) ) U ,\tag{9}
$$

where $\boldsymbol { z } _ { b } \in \mathbb { R } ^ { T \times r _ { 1 } }$ is the low-rank projected representation of sample b. The final adapted layer output is

$$
y _ { b } = x _ { b } W _ { 0 } + s \cdot y _ { \mathrm { a d a p t } , b } ,\tag{10}
$$

where s is the adapter scaling factor.

This formulation can be viewed as an input-conditioned Tucker factorization of the adapter update. Instead of selecting from independent expert matrices, SRTA learns a shared tensor basis and constructs a sample-specific adaptation matrix by softly blending core slices. This allows related domains to share adaptation factors while still permitting domain-specific specialization.

## 3.5 Progressive Depth-Weighted Routing Supervision

Routing decisions are computed independently at each adapter layer. However, if routing is supervised only through the final classification loss, the routing parameters in earlier layers may receive weak or delayed learning signals. To address this, we introduce progressive depth-weighted routing supervision.

Let M denote the total number of adapter layers. For adapter layer $\ell ,$ let $q ^ { ( \ell ) }$ be the routing logits and let $d ^ { * }$ be the ground-truth domain label. We define the layer-wise routing loss as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r o u t e } } ^ { ( \ell ) } = \mathcal { L } _ { \mathrm { C E } } \left( q ^ { ( \ell ) } , d ^ { * } \right) , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is the cross-entropy loss. Each layer is assigned a progressive depth weight:

$$
w _ { \ell } = \frac { \ell } { M } .\tag{12}
$$

This assigns smaller weights to shallow layers and larger weights to deeper layers, reflecting the intuition that deeper representations are more semantically abstract and more suitable for domain-discriminative routing. The weighted routing supervision provides direct learning signals to intermediate routing decisions while keeping inference unchanged.

## 3.6 Training Objective

The final training objective combines the main classification loss with the auxiliary depth-weighted routing loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C E } } \left( \hat { y } , y ^ { * } \right) + \lambda _ { \mathrm { r o u t e } } \frac { 1 } { M } \sum _ { \ell = 1 } ^ { M } w _ { \ell } \mathcal { L } _ { \mathrm { r o u t e } } ^ { ( \ell ) } .\tag{13}
$$

Here, $\hat { y }$ is the model prediction, $y ^ { * }$ is the ground-truth class label, $d ^ { * }$ is the domain label used inside the routing loss, and $\lambda _ { \mathrm { { r o u t e } } }$ controls the strength of the auxiliary routing supervision.

The domain label is used only during training for the progressive depthweighted routing loss. During inference, SRTA computes routing probabilities directly from the input representation and does not require domain labels. Therefore, the auxiliary routing supervision improves pathway learning without adding inference-time inputs or parameters.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We conduct experiments on five widely used multi-domain visual benchmarks: PACS [11], VLCS [18], Ofice-Home [20], Digits-DG [24], and NICO++ [23]. These datasets contain heterogeneous domains with variations in visual style, object appearance, background context and semantic overlap. For each dataset, we treat every domain as a distinct task or domain pathway, allowing us to evaluate whether SRTA can learn reusable adaptation factors while maintaining domainaware routing behavior. PACS, VLCS, Ofice-Home, and Digits-DG contain four domains, while NICO++ contains six domains in our experimental setup.

Data Split. For each benchmark, we use an 80/20 train-test split. From the training portion, we uniformly sample 2400 images, using 2000 images for training and 400 images for validation. This controlled setup allows us to evaluate adaptation under limited data while maintaining balanced domain supervision.

Backbone. All methods use a frozen vit-base-patch16-224-in21k [2] backbone. Only the adaptation modules and the classification head are trained. This setting isolates the efect of each PEFT method and evaluates its ability to adapt a shared pre-trained representation to heterogeneous visual domains.

Baselines. We compare SRTA against representative PEFT and MoE-PEFT baselines:

– LoRA [4]: standard low-rank adaptation.

– DoRA [15]: weight-decomposed low-rank adaptation.

– MOELoRA [14]: MOE-LoRA with task-conditioned expert routing.

– MALoRA [21]: MoE-LoRA with shared low-rank structure.

– MoLoRA [22]: MoE-based LoRA with multiple low-rank experts.

Implementation Details. Adapters are injected into the Query and Value projection matrices of all transformer layers unless otherwise specified. All models are trained for 30 epochs using the AdamW optimizer with a batch size of 64. The learning rate is set to $5 \times 1 0 ^ { - 4 }$ with a weight decay of 0.01 and a warmup ratio of 0.1.

For SRTA, the routing temperature is set to $\tau = 0 . 1$ , and the adapter scaling factor is set to $s = 4 . 0$ . The routing loss coeficient is set to $\lambda _ { \mathrm { r o u t e } } = 1$ . Unless otherwise stated, the Tucker ranks are set to $r _ { 1 } = 6 4$ and $r _ { 2 } = 6 4$ , while $r _ { 3 }$ is set to the number of domains in the dataset.

## 4.2 Main Results

Table 1 reports the classification accuracy across five heterogeneous multi-domain visual benchmarks at rank 64. SRTA achieves the highest average accuracy of 88.1%, slightly outperforming MoLoRA. While the absolute accuracy gain is modest, SRTA achieves this result without an external gating network, suggesting that its main advantage lies in the accuracy-parameter trade-of rather than a large standalone accuracy improvement.

The results show that static PEFT methods, such as LoRA and DoRA, perform competitively on PACS and Digits-DG, but are weaker on more heterogeneous benchmarks such as Ofice-Home and $_ \mathrm { N I C O + + }$ . This suggests that a single static low-rank adaptation subspace may be insuficient when domains exhibit diverse visual styles, backgrounds, and semantic contexts. MoE-based methods improve performance on these harder benchmarks by introducing multiple expert pathways, with MoLoRA and MOELoRA performing strongly on Ofice-Home and NICO++.

Table 1: Accuracy (%) across heterogeneous multi-domain visual benchmarks (Rank 64).
<table><tr><td>Method</td><td>PACS</td><td></td><td>VLCS Office-Home Digits-DG NICO++ Average</td><td></td><td></td><td></td></tr><tr><td>LoRA</td><td> $9 5 . 1 { \pm } 0 . 1 $ </td><td> $8 2 . 8 { \pm } 0 . 7$ </td><td> $7 7 . 3 { \pm } 0 . 3 $ </td><td> $\mathbf { 9 3 . 8 { \pm } 0 . 3 }$ </td><td> $7 7 . 6 { \pm } 0 . 3 $ </td><td>85.3</td></tr><tr><td>DoRA</td><td> $9 5 . 0 { \pm } 0 . 2 $ </td><td> $8 2 . 8 { \pm } 0 . 3 $ </td><td> $7 7 . 3 { \pm } 0 . 3 $ </td><td> $\mathbf { 9 3 . 8 { \scriptstyle \pm 0 . 4 } }$ </td><td> $7 7 . 4 { \pm } 0 . 3 $ </td><td>85.3</td></tr><tr><td>MALoRA</td><td> $9 4 . 3 { \pm } 0 . 5 $ </td><td>82.8±0.1</td><td> $8 5 . 0 { \pm } 0 . 0 $ </td><td> $9 0 . 2 { \pm } 0 . 4 $ </td><td> $8 2 . 3 { \pm } 0 . 3 $ </td><td>86.9</td></tr><tr><td>MOELoRA</td><td> $9 4 . 5 { \pm } 0 . 3 $ </td><td> $8 4 . 2 { \pm } 0 . 5 $ </td><td> $8 5 . 2 { \scriptstyle \pm 0 . 0 }$ </td><td> $8 8 . 9 { \pm } 1 . 3 $ </td><td> ${ \bf 8 3 . 3 2 1 . 0 }$ </td><td>87.2</td></tr><tr><td>MoLoRA</td><td> $9 5 . 1 { \pm } 0 . 3 8 4 . 4 { \pm } 0 . 9$ </td><td></td><td> ${ \bf 8 5 . 3 2 0 . 6 }$ </td><td> $9 1 . 7 { \pm } 0 . 5 $ </td><td> ${ \bf 8 3 . 3 \pm 0 . 3 }$ </td><td>88.0</td></tr><tr><td>SRTA</td><td> $\mathbf { 9 5 . 2 { \scriptstyle \pm 0 . 7 ~ 8 4 . 7 \scriptstyle \pm 0 . 3 } }$ </td><td></td><td> $8 4 . 8 { \pm } 0 . 7$ </td><td> $9 2 . 7 { \scriptstyle \pm 0 . 6 }$ </td><td> $8 3 . 0 { \pm } 0 . 4$ </td><td>88.1</td></tr></table>

SRTA remains competitive with these MoE-based methods while using intrinsic self-routing and a shared Tucker core instead of an external router or independent expert banks. It achieves the best results on PACS and VLCS, and strong performance on Digits-DG and $_ \mathrm { N I C O + + }$ . Although MoLoRA performs slightly better on Ofice-Home, SRTA achieves the best overall average accuracy while maintaining a more compact adaptation structure. These results indicate that input-conditioned Tucker-based routing can efectively organize heterogeneous domains within a shared adaptation space.

## 4.3 Parameter Eficiency

A key advantage of SRTA is that it avoids the parameter overhead introduced by externally routed MoE adapters. Table 2 compares the number of trainable parameters across diferent rank settings for both 4-domain and 6-domain configurations, excluding the classifier to reflect only the cost of the adaptation modules. At rank 64, SRTA requires 2.77M trainable parameters in the 4-domain setting and 3.00M parameters in the 6-domain setting, whereas MoLoRA requires 9.52M and 14.31M parameters, respectively. This corresponds to about $3 . 4 \times$ fewer parameters than MoLoRA in the 4-domain setting and about 4.8× fewer parameters in the 6-domain setting.

These results show that SRTA’s main advantage is not simply accuracy, but achieving MoE-level adaptation performance with a substantially smaller parameter budget. Compared with LoRA and DoRA, SRTA introduces only a modest parameter increase while enabling input-dependent domain-aware routing. Overall, these results show that strong multi-domain adaptation does not require large independent expert banks; instead, a compact Tucker-decomposed shared tensor space can provide an efective accuracy-parameter trade-of by reusing common adaptation factors while supporting domain-specific specialization.

## 4.4 Ablation on Loss Components

We evaluate the efect of the progressive depth-weighted routing supervision by varying the auxiliary routing loss coeficient $\lambda _ { \mathrm { { r o u t e } } } .$ . Table 3 compares SRTA without routing supervision $( \lambda _ { \mathrm { r o u t e } } = 0 . 0 )$ and with routing supervision $( \lambda _ { \mathrm { r o u t e } } = 1 . 0 )$

Table 2: Parameter comparison across ranks for 4-domain and 6-domain settings. Parameter counts exclude the classifier.
<table><tr><td rowspan="2">Method</td><td colspan="4">4 Domains</td><td colspan="4">6 Domains</td></tr><tr><td>r=8</td><td>r=16</td><td>r=32</td><td> $r { = } 6 4$ </td><td> $r { = } 8$ </td><td>r=16</td><td>r=32</td><td>r=64</td></tr><tr><td>LoRA</td><td>0.29M</td><td>0.59M</td><td>1.18M</td><td>2.36M</td><td>0.29M</td><td>0.59M</td><td>1.18M</td><td>2.36M</td></tr><tr><td>DoRA</td><td>0.31M</td><td>0.61M</td><td>1.20M</td><td>2.38M</td><td>0.31M</td><td>0.61M</td><td>1.20M</td><td>2.38M</td></tr><tr><td>MALoRA</td><td>0.53M</td><td>0.98M</td><td>1.90M</td><td>3.82M</td><td>0.75M</td><td>1.35M</td><td>2.59M</td><td>5.17M</td></tr><tr><td>MOELoRA</td><td>0.30M</td><td>0.59M</td><td>1.18M</td><td>2.36M</td><td>0.44M</td><td>0.89M</td><td>1.77M</td><td>3.54M</td></tr><tr><td>MoLoRA</td><td>1.26M</td><td>2.44M</td><td>4.80M</td><td>9.52M</td><td>1.93M</td><td>3.70M</td><td>7.23M</td><td>14.31M</td></tr><tr><td>SRTA</td><td>0.31M</td><td>0.62M</td><td>1.29M</td><td>2.77M</td><td>0.35M</td><td>0.68M</td><td>1.38M</td><td>3.00M</td></tr></table>

Table 3: Ablation of SRTA loss components at rank 64.
<table><tr><td> $\lambda _ { \mathrm { { r o u t e } } }$ </td><td></td><td></td><td>PACS VLCS Office-Home Digits-DG NICO++ Average</td><td></td><td></td><td></td></tr><tr><td>0.0</td><td></td><td> ${ \bf 9 5 . 2 \pm 0 . 3 8 2 . 3 \pm 0 . 9 }$ </td><td> $8 3 . 5 { \pm } 0 . 3 $ </td><td> $9 2 . 6 { \pm } 0 . 4$ </td><td> $8 1 . 7 { \scriptstyle \pm 0 . 6 }$ </td><td>87.1</td></tr><tr><td>1.0</td><td></td><td> $\mathbf { 9 5 . 2 { \scriptstyle \pm 0 . 7 ~ 8 4 . 7 \scriptstyle \pm 0 . 4 } }$ </td><td> $\mathbf { 8 4 . 8 \pm 0 . 8 }$ </td><td> $\mathbf { 9 2 . 7 { \scriptstyle \pm 0 . 7 } }$ </td><td> ${ \bf 8 3 . 0 { \pm 0 . 4 } }$ </td><td>88.1</td></tr></table>

at rank 64. Adding the routing loss improves the average accuracy from 87.1% to 88.1%, showing that direct supervision of intermediate routing decisions helps the adapter learn more efective domain-aware pathways. The improvement is especially clear on VLCS, Ofice-Home, and NICO++, while PACS remains unchanged, suggesting that routing supervision is most useful when domains are more heterogeneous or semantically overlapping. These results confirm that progressive depth-weighted routing supervision strengthens domain-aware pathway learning and improves the overall stability of SRTA without adding inferencetime parameters.

## 4.5 Ablation on Adapter Ranks

To analyze the trade-of between adaptation capacity and parameter eficiency, we evaluate all PEFT and MoE-PEFT methods under diferent rank configurations. For LoRA and DoRA, we vary the low-rank dimension r. For MALoRA, we vary the shared rank and expert rank. For MoELoRA, we vary the number of routed low-rank expert dimensions following its baseline configuration. For MoLoRA, we vary the expert rank. For SRTA, we vary the Tucker ranks $r _ { 1 }$ and $r _ { 2 } ,$ , while keeping $r _ { 3 }$ equal to the number of domains. Table 4 reports the accuracy across all benchmarks.

Across static PEFT methods, increasing the rank generally improves performance, but the gains saturate at higher ranks. LoRA and DoRA show steady improvements from rank 8 to rank 64, but they remain weaker on more heterogeneous datasets such as Ofice-Home and $\mathrm { N I C O + + }$ , where a single static low-rank subspace is less efective. MoE-based methods perform strongly by introducing multiple expert pathways, but their improvements are not always monotonic with rank. For example, MoLoRA achieves highly competitive results at rank 32, but its performance does not consistently improve at rank 64, suggesting that simply increasing expert capacity does not necessarily resolve multi-domain interference.

Table 4: Rank ablation across all PEFT and MoE-PEFT methods. Results are reported as mean ± standard deviation across multiple runs. For SRTA, r<sub>1</sub> and $r _ { 2 }$ denote Tucker ranks, while $r _ { 3 }$ is set to the number of domains.
<table><tr><td>Method</td><td>Rank-1 Rank-2 Extra Rank PACS</td><td></td><td>VLCS Office-Home Digits-DG NICO++</td><td></td><td></td><td></td></tr><tr><td>LoRA</td><td>8</td><td></td><td> $9 3 . 4 { \pm } 0 . 6 ~ 8 1 . 1 { \pm } 0 . 7 $ </td><td> $7 1 . 7 { \pm } 1 . 2 $ </td><td>91.6±0.7</td><td>67.9±1.4</td></tr><tr><td>LoRA</td><td>16</td><td></td><td> $9 4 . 1 { \pm } 0 . 5 ~ 8 1 . 5 { \pm } 0 . 6$ </td><td> $7 5 . 9 { \pm } 1 . 4 $ </td><td>92.2±0.9</td><td>71.5±0.8</td></tr><tr><td>LoRA</td><td>32</td><td></td><td> $9 4 . 7 { \scriptstyle \pm 0 . 4 ~ 8 2 . 4 \pm 0 . 3 }$ </td><td> $7 7 . 1 { \pm } 0 . 6 $ </td><td>92.9±0.4</td><td>75.2±0.9</td></tr><tr><td>LoRA</td><td>64</td><td></td><td> $9 5 . 1 { \scriptstyle \pm 0 . 1 ~ 8 2 . 8 \pm 0 . 7 }$ </td><td> $7 7 . 3 { \pm } 0 . 3 $ </td><td>93.8±0.3</td><td>77.6±0.3</td></tr><tr><td>DoRA</td><td>8</td><td></td><td> $9 3 . 5 { \pm } 0 . 7 ~ 8 0 . 8 { \pm } 1 . 1$ </td><td> $7 2 . 1 { \pm } 1 . 2 $ </td><td>91.4±0.7</td><td>67.1±1.2</td></tr><tr><td>DoRA</td><td>16</td><td></td><td> $9 3 . 9 { \pm } 0 . 3 ~ 8 1 . 9 { \pm } 0 . 8$ </td><td> $7 6 . 1 \pm 1 . 4$ </td><td>92.2±0.5</td><td>71.7±0.5</td></tr><tr><td>DoRA</td><td>32</td><td></td><td> $9 4 . 8 { \pm } 0 . 5 ~ 8 2 . 3 { \pm } 0 . 6$ </td><td> $7 7 . 3 { \pm } 0 . 3 $ </td><td>92.9±0.3</td><td>74.8±1.0</td></tr><tr><td>DoRA</td><td>64</td><td></td><td> $9 5 . 0 { \pm } 0 . 2 \ 8 2 . 8 { \pm } 0 . 3$ </td><td>77.3±0.3</td><td>93.8±0.4</td><td>77.4±0.3</td></tr><tr><td>MALoRA</td><td>8</td><td>4</td><td> $9 4 . 6 { \pm } 0 . 6 \ 8 3 . 8 { \pm } 0 . 6$ </td><td>84.6±0.3</td><td>88.7±0.3</td><td>82.4±0.7</td></tr><tr><td>MALoRA</td><td>16</td><td>8</td><td> $9 4 . 7 { \scriptstyle \pm 0 . 6 } 8 3 . 3 { \scriptstyle \pm 0 . 4 }$ </td><td>85.0±1.0</td><td>90.4±0.5</td><td>82.5±0.2</td></tr><tr><td>MALoRA</td><td>32</td><td>16</td><td> $9 4 . 5 { \pm } 0 . 5 ~ 8 4 . 4 { \pm } 0 . 8$ </td><td>84.8±0.8</td><td>89.2±0.2</td><td>82.4±0.2</td></tr><tr><td>MALoRA</td><td>64</td><td>32</td><td> $9 4 . 3 { \pm } 0 . 5 ~ 8 2 . 8 { \pm } 0 . 1$ </td><td>85.0±0.0</td><td>90.2±0.4</td><td>82.3±0.3</td></tr><tr><td>MOELoRA</td><td>8</td><td>4</td><td> $9 3 . 6 { \pm } 0 . 5 ~ 8 4 . 1 { \pm } 0 . 6$ </td><td>85.3±0.1</td><td>88.6±1.4</td><td>83.8±0.9</td></tr><tr><td>MOELoRA</td><td>16</td><td>8</td><td> $9 4 . 0 { \pm } 1 . 1 8 3 . 3 { \pm } 1 . 9$ </td><td>84.7±0.3</td><td>89.0±1.7</td><td>83.5±0.1</td></tr><tr><td>MOELoRA</td><td>32</td><td>16</td><td> $9 3 . 4 { \pm } 1 . 7 ~ 8 4 . 6 { \pm } 0 . 9$ </td><td>85.0±0.4</td><td>88.9±1.1</td><td>83.5±1.2</td></tr><tr><td>MOELoRA</td><td>64</td><td>32</td><td> $9 4 . 5 { \pm } 0 . 3 ~ 8 4 . 2 { \pm } 0 . 5$ </td><td>85.2±0.0</td><td>88.9±1.3</td><td>83.3±1.0</td></tr><tr><td>MoLoRA</td><td>8</td><td></td><td> $9 5 . 2 { \pm } 0 . 2 8 4 . 7 { \pm } 0 . 8$ </td><td>85.5±0.4</td><td>91.3±1.0</td><td>83.3±0.3</td></tr><tr><td>MoLoRA</td><td>16</td><td></td><td> $9 5 . 0 { \pm } 0 . 2 \ 8 4 . 1 { \pm } 0 . 6$ </td><td>85.4±0.3</td><td>92.2±0.6</td><td>83.8±0.5</td></tr><tr><td>MoLoRA</td><td>32</td><td></td><td> $9 5 . 3 { \scriptstyle \pm 0 . 3 } 8 5 . 4 { \scriptstyle \pm 0 . 7 }$ </td><td> $8 5 . 6 { \pm } 0 . 5 $ </td><td> $9 2 . 4 { \pm } 0 . 2 $ </td><td>83.9±0.5</td></tr><tr><td>MoLoRA</td><td>64</td><td></td><td> $9 5 . 1 { \scriptstyle \pm 0 . 3 } ~ 8 4 . 4 { \scriptstyle \pm 0 . 9 }$ </td><td> $8 5 . 3 { \pm } 0 . 6 $ </td><td>91.7±0.5</td><td>83.3±0.3</td></tr><tr><td>SRTA</td><td>8</td><td>8</td><td> $9 3 . 5 { \pm } 0 . 6 ~ 8 4 . 4 { \pm } 0 . 4$ </td><td> $8 3 . 8 { \pm } 0 . 4$ </td><td>88.7±0.8</td><td>82.8±0.3</td></tr><tr><td>SRTA</td><td>16</td><td>16</td><td> $9 4 . 6 { \pm } 0 . 4 ~ 8 4 . 8 { \pm } 0 . 3$ </td><td>84.2±0.3</td><td>90.0±0.3</td><td>82.7±0.6</td></tr><tr><td>SRTA</td><td>32</td><td>32</td><td> $9 4 . 6 { \pm } 0 . 5 ~ 8 4 . 3 { \pm } 0 . 8$ </td><td> $8 4 . 4 { \pm } 0 . 5 $ </td><td>91.7±0.5</td><td>82.9±0.8</td></tr><tr><td>SRTA</td><td>64</td><td>64</td><td> $9 5 . 2 { \scriptstyle \pm 0 . 7 ~ 8 4 . 7 { \scriptstyle \pm 0 . 3 } }$ </td><td> $8 4 . 8 { \pm } 0 . 7 $ </td><td> $9 2 . 7 { \pm } 0 . 6 $ </td><td>83.0±0.4</td></tr></table>

SRTA shows a more stable scaling trend as the Tucker ranks increase from 8 to 64. The rank-32 configuration already achieves competitive performance, while the rank-64 configuration gives the strongest overall result for SRTA. This indicates that SRTA can increase adaptation capacity smoothly through its shared Tucker-decomposed representation space, without relying on a large bank of independent experts or an external gating network. Overall, the rank ablation demonstrates that SRTA provides a favorable balance between capacity, stability, and parameter eficiency across heterogeneous visual domains.

## 4.6 Routing Behavior Analysis

To better understand intrinsic routing, we visualize the average routing probabilities assigned to each Tucker core slice for every domain in Fig. 2. The heatmaps show that SRTA learns domain-consistent routing patterns under auxiliary domain supervision. For visually distinct datasets such as PACS and Digits-DG, the routing distributions are sharp and nearly diagonal, indicating that each domain strongly activates a specific core slice. VLCS also shows largely domain-specific routing, with some sharing between related domains.

For more overlapping datasets such as Ofice-Home and NICO++, the routing distributions are softer, indicating greater sharing across domains. This supports the design of SRTA: visually separable domains can activate distinct pathways, while related domains can reuse shared Tucker core slices within a compact adaptation space. Since routing is supervised during training using domain labels, these visualizations should be interpreted as evidence of learned domainaware routing rather than unsupervised domain discovery.

![](images/c42ab1d7d0d8f1af9fa59b8196e24032916f33655c43758607dd32a525ab8598.jpg)  
Fig. 2: Intrinsic routing probabilities (α) across core tensor slices.

## 5 Conclusion

In this paper, we introduced Self-Routed Tensor Adapters (SRTA), a compact parameter-eficient adaptation framework for heterogeneous multi-domain visual representation learning. SRTA removes the need for an external MoE router by deriving routing probabilities directly from the adapter’s low-rank representation. These probabilities are used to blend slices of a shared Tucker core tensor, producing input-conditioned adaptation within a compact shared parameter space.

Experiments across five multi-domain visual classification benchmarks show that SRTA achieves competitive or slightly stronger average accuracy than MoEstyle PEFT baselines while using substantially fewer trainable parameters. At rank 64, SRTA uses about 3.4× fewer parameters than MoLoRA in the 4-domain setting and about 4.8× fewer parameters in the 6-domain setting. These results suggest that SRTA’s main strength is its accuracy-parameter trade-of: it provides MoE-level adaptation performance without relying on large independent expert banks or an external gating network.

The routing analysis further shows that SRTA learns interpretable domainaware pathways, with sharper routing for visually distinct domains and softer sharing for overlapping domains. Overall, SRTA demonstrates that universal visual adaptation can be supported through a shared tensor-factorized adaptation space rather than through separate expert modules, making self-routed tensor adapters a promising building block for scalable adaptation of visual foundation models.

## 6 Limitations

While SRTA shows strong results on multi-domain image classification benchmarks, its application to broader visual tasks such as detection, segmentation, video understanding, and vision-language learning remains an important direction for future work. Our current experiments also use known domain labels for routing supervision, whereas real-world settings may involve unknown or continuously changing domains. Future extensions can explore unsupervised or adaptive routing mechanisms that discover domain structure automatically.

## 7 LLM Usage

The authors used an LLM to assist with grammatical and stylistic editing only. All changes were reviewed by the authors, who take full responsibility for the final manuscript.

## 8 Acknowledgement

The author acknowledges the GPU compute support provided by the Infosys Center for Artificial Intelligence at IIIT-Delhi. The author also thanks Siddharth Yadav from IIIT-Delhi for his support in creating the plots for this paper and Parth Goyal from IIIT-Delhi for his valuable review and suggestions that helped improve the writing of the paper.

## References

1. Carroll, J.D., Chang, J.J.: Analysis of individual diferences in multidimensional scaling via an n-way generalization of “eckart-young” decomposition. Psychometrika 35(3), 283–319 (1970)

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

3. Fedus, W., Zoph, B., Shazeer, N.: Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research 23(120), 1–39 (2022)

4. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)

5. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European conference on computer vision. pp. 709–727. Springer (2022)

6. Jie, S., Deng, Z.H.: Fact: Factor-tuning for lightweight adaptation on vision transformer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 1060–1068 (2023)

7. Kolda, T.G., Bader, B.W.: Tensor decompositions and applications. SIAM Review 51(3), 455–500 (2009). https://doi.org/10.1137/07070111X, https://doi.org/ 10.1137/07070111X

8. Kopiczko, D., Blankevoort, T., Asano, Y.: Vera: Vector-based random matrix adaptation. In: International Conference on Learning Representations. vol. 2024, pp. 6815–6835 (2024)

9. Lee, C.Y., Xie, S., Gallagher, P., Zhang, Z., Tu, Z.: Deeply-supervised nets. In: Artificial intelligence and statistics. pp. 562–570. Pmlr (2015)

10. Lester, B., Al-Rfou, R., Constant, N.: The power of scale for parameter-eficient prompt tuning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 3045–3059 (2021)

11. Li, D., Yang, Y., Song, Y.Z., Hospedales, T.M.: Deeper, broader and artier domain generalization. In: Proceedings of the IEEE international conference on computer vision. pp. 5542–5550 (2017)

12. Li, X.L., Liang, P.: Prefix-tuning: Optimizing continuous prompts for generation. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). pp. 4582–4597 (2021)

13. Li, Y., Han, S., Ji, S.: Vb-lora: Extreme parameter eficient fine-tuning with vector banks. Advances in Neural Information Processing Systems 37, 16724–16751 (2024)

14. Liu, Q., Wu, X., Zhao, X., Zhu, Y., Xu, D., Tian, F., Zheng, Y.: When moe meets llms: Parameter eficient fine-tuning for multi-task medical applications. In: Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval. pp. 1104–1114 (2024)

15. Liu, S.Y., Wang, C.Y., Yin, H., Molchanov, P., Wang, Y.C.F., Cheng, K.T., Chen, M.H.: Dora: Weight-decomposed low-rank adaptation. In: Forty-first International Conference on Machine Learning (2024)

16. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

17. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017)

18. Torralba, A., Efros, A.A.: Unbiased look at dataset bias. In: CVPR 2011. pp. 1521–1528. IEEE (2011)

19. Tucker, L.R.: Some mathematical notes on three-mode factor analysis. Psychometrika 31(3), 279–311 (1966)

20. Venkateswara, H., Eusebio, J., Chakraborty, S., Panchanathan, S.: Deep hashing network for unsupervised domain adaptation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5018–5027 (2017)

21. Wang, X., Zhao, H., Wang, S., Wang, H., Liu, Z.: Malora: Mixture of asymmetric low-rank adaptation for enhanced multi-task learning. In: Findings of the Association for Computational Linguistics: NAACL 2025. pp. 5609–5626 (2025)

22. Zadouri, T., Üstün, A., Ahmadian, A., Ermis, B., Locatelli, A., Hooker, S.: Pushing mixture of experts to the limit: Extremely parameter eficient moe for instruction tuning. In: International Conference on Learning Representations. vol. 2024, pp. 25401–25420 (2024)

23. Zhang, X., He, Y., Xu, R., Yu, H., Shen, Z., Cui, P.: Nico++: Towards better benchmarking for domain generalization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16036–16047 (2023)

24. Zhou, K., Yang, Y., Hospedales, T., Xiang, T.: Deep domain-adversarial image generation for domain generalisation. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 13025–13032 (2020)