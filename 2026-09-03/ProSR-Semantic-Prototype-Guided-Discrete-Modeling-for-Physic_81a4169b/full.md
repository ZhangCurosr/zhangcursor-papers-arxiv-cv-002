# ProSR: Semantic-Prototype-Guided Discrete Modeling for Physically Consistent SAR Super-Resolution

Byoungwoo Kim and Munchurl Kim

Korea Advanced Institute of Science and Technology, Daejeon, Republic of Korea {quddn826, mkimee}@kaist.ac.kr

![](images/e7f12ad5e722fec24005e1f7b00e6f538d9344bdd629466b9dc4c44efb0a89f4.jpg)  
Fig. 1: Qualitative comparison on SAR ISR (×4) results on small objects. Our ProSR restores a structural integrity of tiny targets (red boxes), whereas existing difusion methods sufer from distortion or blurring. (Zoom in for better view.)

Abstract. High-resolution Synthetic Aperture Radar (SAR) imagery is critical for precision analysis such as automatic target recognition, yet its acquisition is costly. Although generative image super-resolution (ISR) models ofer a promising alternative, current smooth-approximationbased difusion frameworks often struggle to preserve the coherent scattering statistics, causing stochastic structural distortions that are less consistent with real SAR physics. To address this, we propose Semantic-Prototype-Guided Super-Resolution (ProSR), reformulating SAR ISR as a semantically-guided discrete token prediction task within a quantized latent space. By mapping signal features to discrete scattering primitives, ProSR preserves the impulsive nature of SAR without over-smoothing. Furthermore, we integrate a Self-Supervised Learning backbone into SAR ISR to extract label-free semantic priors, overcoming label scarcity. Guided by these priors, we introduce Semantic-Aligned Detail Encoding to decouple high-frequency signals into discrete scattering primitives. In parallel, the Semantic Prototype Map Generator explicitly constructs semantic prototype maps, allowing Prototype-Map-Guided Attention to route the information flows within identical categories and mitigate inter-class interference. To validate our approach, we present a large-scale 0.25 m resolution benchmark from the Umbra Open Dataset. Experimen tal results show ProSR achieves superior visual quality while preserving essential scattering characteristics required for practical SAR applications.

Keywords: High-Resolution SAR Image Super-Resolution · Discrete Generative Modeling · Self-Supervised Semantic Guidance

![](images/d6202aa3746f2695c625cfb9d3e2eeea0ebc2873db9c218d0d431b91671757c1.jpg)  
Fig. 2: Smooth-approximation-based vs. semantic-category-based distribution learning. (a, d) vanilla difusion models can exhibit stochastic structure distortion (e.g., intensity merging) by averaging neighboring signals. (b, e) ProSR (Ours) preserves distinct scattering peaks via predicting semantic-category-based representation.

## 1 Introduction

High-resolution (HR) Synthetic Aperture Radar (SAR) imagery is vital for high-precision analysis—such as automatic target recognition (ATR) [72] and infrastructure monitoring [33]—ofering rich scattering information derived from unique interactions between electromagnetic wave and terrain geometry [16]. However, acquiring such high-fidelity data is costly, due to sensor costs and orbital constraints. Consequently, image super-resolution (ISR) has emerged as a powerful framework to bridge the gap between accessible low-resolution (LR) data and the high-fidelity requirements of practical applications [3, 4, 61].

Generative models, led by smooth-approximation-based difusion processes, have demonstrated exceptional capabilities in synthesizing sharp, high-resolution natural images [26, 50, 53, 54, 67, 69]. However, when applied to the SAR domain, these models often struggle to preserve SAR physical integrity, sufering from what we term ‘stochastic structural distortions’. This degradation can be understood from mode interpolation [1]—a phenomenon where smooth transitions between disjoint data modes generate samples entirely outside the true data distribution. Fig. 2 compares a smooth-approximation-based versus a semantic-category-based distribution learning and their efects on generated SAR ISR results. The vanilla difusion models [50, 67, 69] frequently struggle to separate edges that are very close together as shown in Fig. 2-(a). The two nearby edges (gray color) are dificult to be modeled by the vanilla difusion models that can yield an overlapped and blurred edge (red color), creating one single fake intensity between the true scattering peaks after post quantization (green color). This structural degradation can be related to the learned score function being a smooth approximation of the data manifold [52, 55], which often makes it dificult to maintain the sharp discontinuities essential to SAR physics (Fig. 2-(d)). A detailed analysis of a toy example regarding unresolved conditional scatterer modes is provided in the suppl. C to ofer intuition for this behavior. To address this problem, we propose Semantic-Prototype-Guided Super-Resolution (ProSR). ProSR reformulates SAR ISR as a semantically-guided discrete token prediction task, implemented via Prototype-Map-Guided Masked Generative Modeling (PMG). By leveraging a discrete latent space to map features into a finite set of physically valid codebook entries [21], PMG treats reconstruction as an iterative selection of categorical tokens rather than smooth density approximation. Unlike vanilla difusion models, as illustrated in Fig. 2-(a), that tend to average out complex signal distributions, two nearby distinct edges can be faithfully generated, as shown in Fig. 2-(b), by quantized representation of our ProSR that enforces a strict reconstruction of SAR-specific scattering patterns [16]. By doing so, the generation process is inherently prevented from generating ambiguous and smeared scattering signals, which is shown in Fig. 2-(e) unlike the result shown in Fig. 2-(d).

Furthermore, while recent trends in SAR ISR [3,4,61] aim to improve structural fidelity by incorporating task-oriented objectives (e.g., detection losses), these approaches are limited by the scarcity of labeled SAR data. To overcome this label dependency, we introduce a self-supervised representation framework to the SAR ISR task, leveraging foundational models [5, 44] for label-free semantic guidance. This approach allows us (i) to establish Semantic-Aligned Detail Encoding (SADE), which maps discrete codes to specific scattering classes via a semanticaligned detail codebook, and (ii) to define a semantic-aware perceptual loss. While these components provide a robust semantic foundation, they cannot inherently prevent contextual leakage or semantic drift during the cross-attention process. To address this, we introduce Semantic Prototype Map Generator (SPMG) and Prototype-Map-Guided Attention (PMGA). Specifically, within PMG, the SPMG constructs spatial-semantic maps that guide PMGA to restrict attention flows to semantically consistent regions. By acting as a spatial-semantic guidance, PMGA eliminates inter-class signal leakage, ensuring that the reconstruction is both physically accurate and aligned with the actual structure. Finally, we provide a 0.25 m slant range resolution SAR ISR benchmark from the Umbra Open Dataset [57] as a public resource to mitigate data scarcity. In summary, the primary contributions of our work are as follows:

To the best of our knowledge, ProSR is the first discrete generative framework tailored for SAR ISR. It utilizes PMG to suppresses stochastic structural distortions by mapping signal features into a finite set of physically valid codebook entries, ensuring high-frequency fidelity.

– We establish a label-free semantic guidance strategy by integrating a selfsupervised representation model. This enables SADE, SPMG, and PMGA to achieve precise structural anchoring and suppress inter-class interference without requiring scarce manual annotations.

– We establish a large-scale 0.25 m resolution SAR ISR benchmark from the Umbra Open Dataset [57] enabling robust evaluation and future research.   
– Our ProSR establishes a new state-of-the-art in SAR ISR, demonstrating superior visual realism while better preserving the essential scattering characteristics required for practical SAR applications.

## 2 Related Works

## 2.1 Single Image Super-Resolution (SISR)

SISR has transitioned from traditional optimization to deep generative models [56]. While early pixel-wise architectures [19, 32, 38, 71] often produced over-smoothed results, GAN-based models [34, 37, 42, 60] significantly improved sharpness but frequently introduced unrealistic hallucinations due to adversarial instability. Consequently, recent research has shifted toward difusion models [9, 35, 39, 51, 59, 64, 65, 67, 69] for higher perceptual fidelity and training stability; however, these difusion frameworks inherently rely on a smooth approximation of the data distribution [1, 52, 55]. This induces mode interpolation [1], where the model generates samples that smoothly bridge disjoint data modes, resulting in structural hallucinations. Although such artifacts may appear plausible in natural images, they degrade the physical integrity required for high-precision SAR image analysis.

## 2.2 Discrete Latent Representations

Discrete representation learning, popularized by VQ-VAE [58] and VQ-GAN [21], maps high-dimensional data into a finite set of quantized codebook entries. Unlike continuous models, these formulations replace latent values with a categorical selection from a fixed vocabulary. By reformulating reconstruction as a discrete token prediction paradigm [2, 6, 7, 10, 47], researchers utilize quantization as a formative bottleneck. While lossy for natural images, this bottleneck uniquely favors SAR, precluding blurred intermediate values and forcing the model to select from distinct entries. This mechanism suppresses the averaging efects common in continuous spaces. Leveraging these advantages, we propose ProSR, which incorporates a scattering primitive guidance within the discrete bottleneck. This enables our ProSR to achieve realistic reconstruction while preserving physical scattering distributions, overcoming the blurring artifacts of continuous spaces.

## 2.3 SAR Image Super-Resolution (SAR ISR)

SAR ISR is uniquely challenging due to the presence of speckle noise and complex electromagnetic interactions with geometric structures of the ground. Recent works have moved from mathematical priors [29, 30] to generative paradigms [12, 13, 28, 68] to synthesize high-frequency signatures. These realistic reconstructions improve downstream performance (e.g., ATR), proving that realistic scattering patterns are vital for operational analysis. To further enhance fidelity, taskdriven SAR ISR incorporates auxiliary structural priors such as edge maps or Electro-Optical imagery [66, 73], or downstream losses such as a detection loss [3, 4, 61]. Despite their success, these methods are heavily bottlenecked by the scarcity of labeled SAR datasets. This label dependency significantly limits the generalizability and scalability of task-driven models. Unlike these label-dependent methods, our ProSR overcomes the scarcity of labeled data by leveraging a self-supervised representation model as a semantic anchor.

## 2.4 Self-supervised Representation Learning in SAR

To overcome label scarcity, Self-Supervised Learning (SSL) has become a prevailing approach for learning robust representations from unlabeled data [11, 23, 49].

![](images/5f40e1f21e8339e338b14d56a5fafbda12c373943361b906fd845008c9ac4fec.jpg)  
Fig. 3: Overall architecture of the proposed ProSR framework.

Although frameworks like MAE [22] and DINO [5] have shown that rich semantic features can be extracted from vast unlabeled imagery, their application in SAR has remained largely focused on high-level tasks such as classification or semantic segmentation [36, 44]. The potential of SSL-derived semantic priors to serve as a guidance mechanism for low-level reconstruction remains largely untapped. Our ProSR bridges this gap by integrating SSL-derived semantic priors into the reconstruction pipeline. Specifically, we construct a detail codebook aligned with the semantic features of LR inputs and employ a semantic prototype map to guide cross-attention. This enables input-consistent detail recovery while ensuring physical consistency without the need for manual labels.

## 3 Methodology

## 3.1 Overview of ProSR

Fig. 3 illustrates the overall architecture of our ProSR framework, which bridges continuous semantic priors with discrete high-frequency details using a spatialsemantic map to address SAR stochastic structural distortions. The framework consists of three stages: (i) Stage 1 - Semantic-Aligned Detail Encoding (SADE), (ii) Stage 2 - Semantic Prototype Map Generation (SPMG), and (iii) Stage 3 - Prototype-Guided Masked Generative Modeling (PMG). It should be noted that Stage 1 - SADE is first pretrained with HR SAR ground-truth (GT) images and their LR versions to learn separated representations of details (high-frequency) via the VQGAN Encoder (denoted as $\mathcal { E } _ { \mathrm { d } } )$ [21] and the Detail Vector Quantization module, and structural information (low-frequency) via a self-supervised model [44](denoted as ${ \mathcal E } _ { \mathrm { s } } )$ and an Adaptor (denoted as A) [8], where both representations are jointly optimized in conjunction with a shared VQGAN Decoder (denoted as D) [21]. Then, the pretrained $\mathcal { E } _ { s }$ is incorporated in Stage 2 for training and inference. In Stage 3, the pretrained $\mathcal { E } _ { \mathrm { d } } , \mathcal { E } _ { \mathrm { s } }$ and A are used for training while the pretrained ${ \mathcal { E } } _ { \mathrm { s } } , A$ and D are for inference.

## 3.2 Stage 1 - Semantic-Aligned Detail Encoding (SADE)

SADE isolates the high-frequency SAR details of HR (GT) $I _ { \mathrm { H R } } \in \mathbb { R } ^ { H \times W }$ into codebook representations where H and W are the height and width for $I _ { \mathrm { H R } }$ respectively, while maintaining strict alignment with LR input $I _ { \mathrm { L R } } \in \mathbb { R } ^ { H / 4 \times W / 4 }$ which is upscaled to the same size of $I _ { \mathrm { H R } }$ , as shown in Fig. 3-(a). To isolate such high-frequency SAR image details, we employ a dual-encoder architecture. We extract raw semantic features $\mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { O r i } } = \mathcal { E } _ { \mathrm { s } } ( I _ { \mathrm { L R } } ) \stackrel {  } { \in } \mathbb { R } ^ { 1 9 2 \times ( H / 8 \cdot W / 8 ) }$ for $I _ { \mathrm { L R } }$ . For this, we used the SSL backbone [44] as ${ \mathcal E } _ { \mathrm { s } }$ that was pretrained with our SAR training data. $\mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { O r i } }$ is then projected into a lower-dimensional latent space as $\mathbf { F } _ { \mathrm { s e m } } = \breve { \mathcal { A } } ( \mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { O r i } } ) \stackrel { \circ \mathrm { \scriptscriptstyle S m } } { \in } \mathbb { R } ^ { 3 2 \times ( H / \hat { \otimes } \cdot W \stackrel { \circ } { / } 8 ) }$ via a lightweight adapter A [8]. While $\mathbf { F } _ { \mathrm { s e m } }$ serves as a compact semantic anchor for detail decomposition, $\mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { O r i } }$ can be used as high-capacity keys and values for the cross-attention mechanism in Stage 3.

Simultaneously, $\mathcal { E } _ { \mathrm { d } }$ as a detail encoder maps $I _ { \mathrm { H R } }$ into a latent representation $z = \mathcal { E } _ { \mathrm { d } } ( I _ { \mathrm { H R } } ) \in \dot { \mathbb { R } } ^ { 3 2 \times ( H / 8 \cdot W / 8 ) }$ <sup>)</sup>. We define the latent detail feature ${ \boldsymbol { z } } _ { \mathrm { d e t } }$ as the residual between z and $\mathbf { F } _ { \mathrm { s e m } } ,$ which is denoted as $z _ { \mathrm { d e t } } = z - \mathbf { F } _ { \mathrm { s e m } }$ . This residual is then quantized into ${ \hat { z } } _ { \mathrm { d e t } }$ which then becomes a codebook entry. By doing so, we restrict the codebook to high-frequency details, efectively decoupling them from spatial semantic context (low-frequency components).

To ensure the isolation of high-frequency details of $I _ { \mathrm { H R } }$ , we employ a stochastic training strategy. With a probability of $p , I _ { \mathrm { L R } } = { \mathcal { D } } ( \mathbf { F } _ { \mathrm { s e m } } )$ is reconstructed by compelling the decoder D to distill the semantic structure from $\mathbf { F } _ { \mathrm { s e m } }$ independently. With the remaining $1 - p$ probability, D integrates $\hat { z } _ { \mathrm { d e t } }$ with $\mathbf { F } _ { \mathrm { s e m } }$ to synthesize $I _ { \mathrm { H R } }$ . The reconstructed outputs for the two branches are obtained as:

$$
\hat { I } = \left\{ \begin{array} { l l } { \hat { I } _ { \mathrm { L R } } = \mathcal { D } ( \mathbf { F } _ { \mathrm { s e m } } ) } & { \mathrm { w i t h ~ } p } \\ { \hat { I } _ { \mathrm { H R } } = \mathcal { D } ( \hat { z } _ { \mathrm { d e t } } + \mathbf { F } _ { \mathrm { s e m } } ) } & { \mathrm { w i t h ~ } 1 - p } \end{array} \right.\tag{1}
$$

where $p$ is empirically set to 0.25. This stochastic strategy prevents the detail codebook from encoding redundant structures, isolating pure details, by forcing $\mathbf { F } _ { \mathrm { s e m } }$ to maintain a self-suficient structural representation. Consequently, zˆ<sub>det</sub> focuses on semantic-aligned scattering signatures (high-frequency) of $I _ { \mathrm { H R } }$ while $\mathbf { F } _ { \mathrm { s e m } }$ acts as an independent structural anchor that dictates semantic structures (low-frequency) of $I _ { \mathrm { L R } }$ , making ${ \hat { z } } _ { \mathrm { d e t } }$ inherently predictable from $\mathbf { F } _ { \mathrm { s e m } }$ in Stage 3. Gumbel Vector Quantization. We employ Gumbel Vector Quantization (Gumbel-VQ) [21, 27] to map ${ \boldsymbol { z } } _ { \mathrm { d e t } }$ into a discrete codebook entry, ensuring stable and diferentiable training. To prevent codebook collapse and ensure its uniform utilization, we adopt the KL regularization $\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \pi _ { k } \log ( \pi _ { k } \cdot K ) \right] } \end{array}$ 2 where $\pi _ { k } = P ( q ( z _ { \mathrm { d e t } } ) = e _ { k } )$ denotes the probability of ${ \boldsymbol { z } } _ { \mathrm { d e t } }$ being assigned to the k-th entry (codebook assignment) $e _ { k }$ via the quantizer $q ( \cdot )$ . This objective maximizes codebook entropy, forcing tokens to represent a diverse and comprehensive library of SAR scattering primitives.

Training Objectives. Stage 1 is optimized using a stochastic objective function that adaptively selects loss terms from the reconstruction branches as:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \left\{ \begin{array} { l l } { { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { p e r } } { \mathcal { L } } _ { \mathrm { p e r } } } & { { \mathrm { w i t h ~ } } p } \\ { { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { p e r } } { \mathcal { L } } _ { \mathrm { p e r } } + \lambda _ { \mathrm { a d v } } { \mathcal { L } } _ { \mathrm { a d v } } + \lambda _ { \mathrm { K L } } { \mathcal { L } } _ { \mathrm { K L } } } & { { \mathrm { w i t h ~ } } 1 - p } \end{array} \right. }\tag{2}
$$

where $\mathcal { L } _ { \mathrm { r e c } }$ and ${ \mathcal { L } } _ { \mathrm { a d v } }$ [21] are an $\mathcal { L } _ { 1 }$ -reconstruction and adversarial losses, respectively. To capture domain-specific signatures without labeled SAR data, we define the perceptual loss $\mathcal { L } _ { \mathrm { p e r } }$ as $\mathcal { L } _ { \mathrm { S A F E } }$ using pre-trained a SAR SSL model ${ \mathcal E } _ { \mathrm { s } }$ [44], enabling physically-grounded supervision from unlabeled imagery. This approach ensures the preservation of essential scattering patterns that pixel-wise losses often overlook. Restricting $\mathcal { L } _ { \mathrm { a d v } }$ and ${ \mathcal { L } } _ { \mathrm { K L } }$ to the $1 - p$ branch forces the codebook toward realistic high-frequency signatures, while $\mathbf { F } _ { \mathrm { s e m } }$ preserves the fundamental semantic structural layout.

## 3.3 Stage 2 - Semantic Prototype Map Generation (SPMG)

The SPMG converts continuous semantic features into a discrete spatial-semantic map (Fig. 3-b), bridging the encoding-generation gap stemming from ignoring semantic information during cross-attention.

Step 1: Mapping Spatial Features to Semantic Prototype Tokens. We define a set of $K = 3$ learnable prototype tokens, $P = \{ p _ { \mathrm { T } } , p _ { \mathrm { S } } , p _ { \mathrm { C } } \} \in \mathbb { R } ^ { 3 \times C }$ representing the fundamental scattering primitives in the SAR domain: target (T), shadow (S), and clutter (C). The choice of a token ensures a corresponding semantic interpretability by aligning it with a physical scattering primitive. Empirically, we found that $K > 3$ often leads to redundant or inactive prototype tokens, whereas $K = 3$ maintains an efective representation throughout all our experiments. For the semantic feature $\mathbf { F } _ { \mathrm { S e m } } ^ { \mathrm { O r i } }$ , we denote $f _ { i } \in \mathbb { R } ^ { C }$ as a spatial feature vector at spatial position i of $\mathbf { F } _ { \mathrm { S e m } } ^ { \mathrm { O r i } }$ . For $f _ { i } ,$ we compute a sparse similarity assignment $S _ { i , k }$ for a prototype token $p _ { k } \in P$ using the Sparsemax [43] as:

$$
S _ { i , k } = \left[ \mathrm { S p a r s e m a x } ( - \| f _ { i } - { \pmb { p } } _ { k } \| _ { 2 } ^ { 2 } ) \right] _ { k }\tag{3}
$$

Step 2: Semantic Map generation via Dynamic Discretization. Instead of a naive argmax, which often blurs the physical distinction between target and clutter, we employ dynamic discretization to isolate high-intensity regions as structural anchors, ensuring a physical reliability (see Suppl. B for sensitivity analysis). At each spatial position i and for each class $k \in \{ \mathrm { T } , \mathrm { S } \}$ , we define a binary indicator mask $\mathcal { T } _ { i , k } \in \{ 0 , 1 \}$ as:

$$
\mathcal { T } _ { i , k } = \mathbb { 1 } ( S _ { i , k } \geq \tau _ { k } \mathrm { ~ a n d ~ } S _ { i , k } \geq S _ { i , m } ) , \quad k , m \in \{ \mathrm { T } , \mathrm { S } \} , \quad m \neq k\tag{4}
$$

where $\tau _ { k } = \gamma { \cdot } \operatorname* { m a x } _ { j } ( S _ { j , k } )$ is a relative threshold with $j$ spanning all spatial indices. We empirically set $\gamma = 0 . 7$ to prioritize high-confidence scattering centers while filtering stochastic artifacts (see Suppl. B for sensitivity analysis). To ensure semantic exclusivity, clutter regions (C) are defined as the regions that do not belong to target and shadow regions, satisfying $\begin{array} { r } { \sum _ { k \in \{ \mathrm { T } , \mathrm { S } \} } \mathcal { T } _ { i , k } = 0 } \end{array}$ . To isolate stable scattering signatures, we intersect the top c% global $S _ { i , \mathrm C }$ pixels with the clutter region where c is empirically set to 30. This sparse filtering, prioritizing the classes with relatively higher similarity scores, yields a non-overlapping Semantic Prototype Map $\left( M _ { \mathrm { s e m } } \right)$ that robustly guides Stage 3 to enable semantic-guided cross-attention while eliminating boundary ambiguity.

Prototype Optimization Objectives. To anchor the learnable prototype tokens to physical reality, we reconstruct feature map $\hat { \mathbf { F } } \in \mathbb { R } ^ { 1 9 2 \times ( H / 8 \cdot \hat { W } / 8 ) }$ using prototype tokens P. The reconstructed feature at position i is formulated as $\begin{array} { r } { \hat { \pmb { f } } _ { i } = \sum _ { k \in \{ \mathrm { T } , \mathrm { S } , \mathrm { C } \} } S _ { i , k } { \pmb { \cdot p } } _ { k } } \end{array}$ , where $S _ { i , k }$ is the sparse similarity score from Eq. (3) and $p _ { k } \in P$ is the corresponding learnable prototype token. We optimize $\hat { \mathbf { F } }$ against despeckled HR features $\mathbf { F } _ { \mathrm { H R } } ^ { \mathrm { { \check { d e s p e c } } } } = \mathcal { E } _ { \mathrm { s } } \dot { ( } I _ { \mathrm { H R } } ^ { \mathrm { d e s p e c } } )$ using a feature reconstruction loss $\mathcal { L } _ { \mathrm { r e c } } = \| \mathbf { F } _ { \mathrm { H R } } ^ { \mathrm { d e s p e c } } - \hat { \mathbf { F } } \| _ { 1 }$ , where $I _ { \mathrm { H R } } ^ { \mathrm { d e s p e c } }$ is obtained following [15]. To ensure distinctness between target, shadow, and clutter, we use a Repulsion Loss $\mathcal { L } _ { \mathrm { r e p } } =$ $\begin{array} { r } { \sum _ { m \neq n } \left| { \frac { { \pmb p } _ { m } \cdot { \pmb p } _ { n } } { \| { \pmb p } _ { m } \| \| { \pmb p } _ { n } \| } } \right| } \end{array}$ that enforces the orthogonality between prototype token pairs. The final objective is $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } }$ , ensuring that the learnable tokens $P$ are both physically representative and semantically discriminative.

## 3.4 Stage 3 - Prototype-Map-Guided Masked Generative Modeling

Prototype-Map-Guided Masked Generative Modeling (PMG) (Fig. 3-c) synthesizes HR tokens using a ProSR Transformer based on the MaskGIT framework [7]. By reformulating reconstruction as an iterative selection of categorical tokens rather than smooth density approximation, our ProSR suppresses mode interpolation [1] and ensures that generated signals remain within physically valid scattering primitives. Input embeddings for the ProSR transformers are formed by concatenating masked detail tokens with $\mathbf { F } _ { \mathrm { s e m } }$ from Stage 1, providing a strong semantic prior of $I _ { \mathrm { L R } }$ throughout the iterative process.

Prototype-Map-Guided Attention (PMGA). While self-attention captures long-range spatial dependencies, unconstrained cross-attention between latent tokens and semantic priors $( \mathbf { F } _ { \mathrm { { s e m } } } ^ { \mathrm { { O r i } } } )$ can lead to contextual leakage due to the absence of semantic guidance, sufering from semantic drift, where scattering signatures from disparate terrain types may interfere during the cross-attention process. To address this, we introduce PMGA as a spatial-semantic guidance. Using $M _ { \mathrm { s e m } }$ (defined in Sec. 3.3), PMGA restricts cross-attention weights $A _ { i j } =$ Softmax $( Q _ { i } K _ { j } ^ { T } / \sqrt { d } + M _ { i j } )$ such that HR tokens only aggregate information from semantically consistent regions. Here, $Q _ { i }$ is the latent query derived from self-attention at position $i ,$ and $K _ { j }$ is latent key with $\mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { O r i } }$ at position $j .$ . The Semantic Prototype Mask $M _ { i j }$ is defined as:

$$
M _ { i j } = \left\{ { \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } \ell ( \mathrm { p o s } _ { i } ) = \ell ( \mathrm { p o s } _ { j } ) \medskip } \\ { - \infty , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{5}
$$

where $t ( \mathrm { p o s } _ { i } ) \in \{ \mathrm { T } , \mathrm { S } , \mathrm { C } \}$ denotes a semantic label at position i obtained from $M _ { \mathrm { s e m } }$ . For unassigned regions, cross-attention is bypassed to block background interference. Enforcing intra-class attention suppresses inter-class feature leakage while reinforcing the distinct signatures of individual scattering centers, preserving the sharp intensity distributions and speckle statistics inherent to SAR signals.

Training Objectives. PMG is optimized by minimizing the cross-entropy (CE) loss between predicted and HR (GT) tokens, conditioned on the visible context and semantic guides:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t a g e 3 } } = \mathbb { E } _ { \hat { z } _ { \mathrm { d e t } } , M } \left[ \sum _ { i \in M } \mathcal { L } _ { \mathrm { C E } } \left( \hat { z } _ { \mathrm { d e t } } ^ { ( i ) } , p \theta ( \hat { z } _ { \mathrm { S R } } ^ { ( i ) } \mid \hat { z } _ { \mathrm { d e t } } ^ { \vec { M } } , \mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { o r i } } , \mathbf { F } _ { \mathrm { s e m } } , M _ { \mathrm { s e m } } ) \right) \right] } \end{array}\tag{6}
$$

where $\hat { z } _ { \mathrm { d e t } } = \{ \hat { z } _ { \mathrm { d e t } } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ is the collection of HR tokens, $\hat { z } _ { \mathrm { d e t } } ^ { ( i ) }$ denotes the token at index i, and $p _ { \theta }$ is the likelihood predicted by PMG Transformer θ. The indices are partitioned into a masked set M and a visible set M<sup>¯</sup> , where $\hat { z } _ { \mathrm { d e t } } ^ { \bar { M } }$ serves as the context for predicting the masked tokens. By minimizing $\mathcal { L } _ { \mathrm { s t a g e 3 } }$ in the discrete latent space, PMG learns to predict high-frequency scattering primitives strictly aligned by the spatial-semantic guidance map $M _ { \mathrm { s e m } }$ . This formulation efectively circumvents the mode interpolation problem, helping the reconstructed intensities remain within the support of physically plausible SAR characteristics.

Inference: Iterative Sampling with PMGA. At inference, our ProSR synthesizes the HR detail tokens via an iterative decoding process following the MaskGIT framework [7]. Starting from a fully masked grid, the number of tokens to be unmasked at each step t is determined by a decreasing masking schedule $\gamma ( t )$ [7]. At each iteration, the highest-confidence tokens are fixed while the remainder are re-masked for the next step. Throughout this process, PMGA remains active to ensure that the ProSR Transformer only attends to semantically consistent regions in $\mathbf { F } _ { \mathrm { s e m } } ^ { \mathrm { O r i } }$ . This iterative refinement, guided by $M _ { \mathrm { s e m } } .$ , produces high-fidelity SAR image details aligned with the physical scattering layout.

## 4 Experimental Results

## 4.1 Dataset Construction

To evaluate our ProSR under realistic conditions, we curated a high-fidelity SAR dataset from the Umbra Open Dataset [57]. We utilized X-band Single Look Complex (SLC) data to serve as the source for HR reference imagery.

HR-LR Pairs. The source SLC data possesses a high-precision native resolution (∼0.25 m azimuth, ∼0.25 m range). To establish a uniform benchmark, we standardized these to 0.25 m for SAR HR references and 1.0 m for SAR LR counterparts. Following the sub-aperture decomposition methodology [44], the SAR LR references were achieved via spectral domain cropping (sub-sampling the Doppler and range spectra) rather than naive image-space interpolation. This ensures physically grounded resolution degradation while preserving authentic speckle statistics and phase integrity in slant-range geometry. Subsequently, the SAR LR images were oversampled by zero-padding GT spectra in the frequency domain prior to the inverse transform.

Amplitude Extraction and Normalization. Following [17], we adopted a normalization strategy optimized for generative SAR tasks. To manage the high dynamic range, we scaled the amplitude (A) using the scene-level mean $( \mu )$ and standard deviation (σ): $A _ { \mathrm { n o r m } } = A / ( \mu + 3 \sigma )$ . The normalized values were clipped to [0, 1] to preserve predominant scattering while mitigating extreme outliers, then quantized into 8-bit integers for storage.

Data Filtering and Diversity. Physical consistency is paramount for model convergence. Since over 97% of our Umbra samples [57] are in VV polarization, we constructed a VV SAR dataset to prevent training biases from heterogeneous scattering signatures. We further restricted incidence angles from 10<sup>◦</sup> to $5 0 ^ { \circ }$ excluding extreme angles due to significant physical divergence and data scarcity. Preprocessing and Statistics. The SAR images were tiled into non-overlapping 1024 × 1024 patches. To ensure a balanced representation of terrestrial features, we applied a selective quality control filter that prioritized patches with significant scattering structures while retaining a controlled portion of low-signal regions. The curated dataset comprises 132, 452 VV-polarized patches from 502 unique SAR images, with heights and widths ranging from approximately 4.5K to 66K and 10K to 94K pixels, respectively. These images span a diverse global geographic distribution (See Fig. 6 in Suppl. A) to ensure broad environmental representation, with their detailed statistics summarized in Table 6 of Suppl. A. More details for the characteristics of SAR images are described in Suppl. A.

Data Availability. Our curated dataset is available at https://github.com/ KAIST-VICLab/ProSR.

## 4.2 Implementation Details

ProSR was implemented in PyTorch [46] and trained on dual NVIDIA A6000 GPUs using $2 5 6 \times 2 5 6$ random crops. Data augmentation included horizontal flips and random rotations $( 9 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } )$ . For a fair comparison, all baseline SR networks (Table 1) were trained from scratch on our data without metric-specific early stopping, following their oficial implementations with SAR-adapted inputs. For AE-based baselines, pretrained AEs remained frozen while only the difusion ISR networks were trained from scratch. For full-image evaluation, all methods use 256×256 sliding-window inference (stride 128) and weighted averaging.

Stage 1 - SADE. Stage 1 follows the ResShift [67] architecture $( f = 8 )$ with the encoder $( { \mathcal { E } } _ { \mathrm { d } } ) ^ { \prime }$ s final projection modified to 32 dimensions. We employed Gumbel-VQ [21, 27] with 1,024 codebook entries and 32 embedding dimensions, stabilized by $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 6 }$ . SADE was trained for 500K iterations using AdamW (batch size 12 on a single GPU, learning rate $1 \times 1 0 ^ { - 4 } )$ [41] with a composite loss: $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { S A F E } } \mathcal { L } _ { \mathrm { S A F E } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } }$ , where $\lambda _ { \mathrm { S A F E } }$ is set to 0.5 and, $\lambda _ { \mathrm { a d v } }$ is adaptively balanced [21]. Simultaneously, semantic features were extracted via a SAFE ViT-tiny model ${ \mathcal E } _ { \mathrm { s } }$ [44], pre-trained on our SAR dataset, and projected to 32 embedding dimensions via a lightweight adapter [8].

Stage 2 - SPMG. The embedding space of P was aligned with the ViT-Tiny feature space [20]. The SPMG was trained for 15k iterations (AdamW, LR $1 \times 1 0 ^ { - 4 }$ , batch size 64, $\lambda _ { r e p } = 2 )$ , with other hyperparameters following Sec. 3.3.

Stage 3 - PMG. PMG Transformer [20] comprises 20 layers with 8 attention heads and a 512-dimensional hidden space. We utilized a Positional Encoding Generator [14], facilitating stable sliding-window inference. Training was conducted for 600K iterations using AdamW $( \mathrm { L R } \ 1 \times 1 0 ^ { - 4 }$ , weight decay 0.045, $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 6 )$ with a total batch size 32. After a 10K-iteration linear warm-up, the learning rate was governed by a Cosine Annealing schedule [40], decaying from $1 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 1 \times 1 0 ^ { - 7 }$ . Additionally, a label smoothing factor of 0.1 was applied to the CE loss to ensure robust token synthesis.

Inference and Sampling. For token synthesis, we followed the MaskGIT [7] sampling process, employing a cosine schedule γ(t) over 8 steps with a temperature of 4.5 for logit sampling.

## 4.3 Performance Comparison

Evaluation Metrics. To comprehensively evaluate our ProSR with other methods, we used three categories of metrics calculated in the SAR amplitude domain:

Standard Fidelity: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure(SSIM) [62] evaluate pixel-level fidelity and local structural agreement, respectively. However, these metrics can favor spatially conservative or over-smoothed reconstructions and therefore do not fully characterize SAR-specific scattering fidelity. Furthermore, SSIM depends strongly on the local covariance $( \sigma _ { x y } )$ between spatially aligned structures; small scatterer displacements, or merging can reduce this covariance and lower SSIM, in some cases even below that of oversampled LR images. We therefore interpret PSNR and SSIM jointly with complementary SAR-oriented structural, perceptual, and distributional metrics.

Target and Structural Integrity: To evaluate physical and semantic consistency, we prioritize metrics focusing on fine-grained structural fidelity and scattering-sensitive signal integrity. This includes Information-content Weighted SSIM (IW-SSIM) [63] to assess fidelity in target-dense areas and Haar Perceptual Similarity Index (HaarPSI) [48] for local structural coherence and edge integrity. Additionally, we employ Target-to-Clutter Ratio (TCR)—calculated using prototype-derived target masks—to verify the radiometric fidelity. Rather than simply seeking higher TCR, we evaluate the absolute deviation from the GT (|∆TCR|) to ensure the reconstructed scattering intensity remains physically consistent with the original observation.

Statistical and Perceptual Realism: Fréchet Inception Distance (FID) [25], Density (Dens), and Coverage (Cov) [45] are used to evaluate statistical distributions, quantifying the alignment and overlap between the synthesized and GT data manifolds. Perceptual Realism is measured via Learned Perceptual Image Patch Similarity (LPIPS) [70], and Deep Image Structure and Texture Similarity (DISTS) [18] to evaluate structural and textural similarity.

Overall Quantitative Comparison. Table 1 compares ProSR with SOTA methods. Although the evaluated difusion baselines (UPSR [69], LDM [50], and ResShift [67]) obtain higher pixel-aligned PSNR/SSIM, these gains do not guarantee better SAR scattering realism. ESRGAN [60] achieves competitive LPIPS/DISTS but struggle to preserve physically consistent speckle patterns. In contrast, our ProSR dominates in Target and Structural Integrity (|∆TCR|, IW-SSIM, HaarPSI), capturing both structural and radiometric fidelity. Notably, ProSR excels in FID, Density, and Coverage, demonstrating its superior statistical realism. These results indicate that our ProSR better captures the underlying distribution of SAR imagery while preserving essential scattering characteristics.

![](images/7d31f72e2f10abfca64d9ce66e06d1d8eee2470e7a942e59f9a1fdaf2a23750e.jpg)  
Fig. 4: Qualitative comparison of SAR ISR (×4) results (Zoom in for better view)

Table 1: Quantitative comparison on our SAR ISR (×4) benchmark. The best and second-best results are highlighted in bold red and blue, respectively.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Standard PSNR ↑</td><td rowspan="2">Fidelity SSIM ↑</td><td colspan="3">Target &amp; Structural Integrity |∆TCR| ↓ (TCR) IW-SSIM ↑ HaarPSI ↑</td><td colspan="5">Statistical &amp; Perceptual Realism Cov ↑ LPIPS ↓ DISTS ↓</td></tr><tr><td></td><td></td><td></td><td></td><td>FID ↓ Dens ↑</td><td></td><td></td><td></td></tr><tr><td colspan="2">GT (Reference)|</td><td></td><td>0.0 (3.3168)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Oversampled</td><td>16.2056</td><td>0.1545</td><td>0.6646 (3.9814)</td><td>0.3478</td><td>0.4211</td><td>70.86</td><td>0.2887</td><td>0.4379</td><td>0.8062</td><td>0.4370</td></tr><tr><td>16.1841</td><td>0.1012</td><td>0.2223 (3.0945)</td><td>0.3480</td><td>0.4655</td><td>50.84</td><td>0.7249</td><td>0.6912</td><td>0.2977</td><td>0.1372</td></tr><tr><td rowspan="2">ESRGAN SwinIR-GAN SPSR</td><td>15.9363</td><td>0.0959</td><td>0.5142 (2.8026)</td><td>0.3306</td><td>0.4407</td><td>64.73</td><td>0.5156</td><td>0.4362</td><td>0.3071</td><td>0.2237</td></tr><tr><td>16.2756</td><td>0.1028</td><td>0.2831 (3.0337)</td><td>0.3451</td><td>0.4578</td><td>48.38</td><td>0.5886</td><td>0.5936</td><td>0.3184</td><td>0.1470</td></tr><tr><td rowspan="2">LDM-15</td><td>17.6427</td><td>0.1277</td><td>0.0892 (3.2276)</td><td>0.4065</td><td>0.4017</td><td>46.91</td><td>0.4946</td><td>0.6057</td><td>0.3888</td><td>0.2708</td></tr><tr><td>17.4784</td><td>0.1273</td><td>0.0657 (3.3825)</td><td>0.4066</td><td>0.4359</td><td>34.19</td><td>0.6156</td><td>0.7022</td><td>0.3575</td><td>0.2199</td></tr><tr><td rowspan="2">ResShift UPSR</td><td>17.8545</td><td>0.1323</td><td>0.2815 (3.0353)</td><td>0.4201</td><td>0.4262</td><td>45.73</td><td>0.6429</td><td>0.6665</td><td>0.3801</td><td>0.2210</td></tr><tr><td>16.9293</td><td>0.1088</td><td>0.0329 (3.2839)</td><td>0.4206</td><td>0.4766</td><td>23.70</td><td>0.9741</td><td>0.8592</td><td>0.3010</td><td>0.1532</td></tr></table>

Table 3: Ablation on quantization types.

Table 2: Stage 1 reconstruction comparison. LDM and ResShift use the same pretrained VQGAN autoencoder [21].
<table><tr><td>Models</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>LDM/ResShift AE [50, 67]</td><td>18.3558</td><td>0.4072</td><td>0.1681</td></tr><tr><td>Ours (1024)</td><td>18.1018</td><td>0.3799</td><td>0.2044</td></tr></table>

<table><tr><td>Quantization Types</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>HR-VQ</td><td>16.5177</td><td>0.1141</td><td>0.2765</td></tr><tr><td>Dual-VQ</td><td>17.6195</td><td>0.3296</td><td>0.2165</td></tr><tr><td>Detail-VQ (Ours)</td><td>18.1018</td><td>0.3799</td><td>0.2044</td></tr></table>

Table 4: Ablations on codebook size K.  
Table 5: Ablation study on the PMGA.
<table><tr><td>Codebook Size K</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>512</td><td>17.9659</td><td>0.3641</td><td>0.2124</td></tr><tr><td>1024 (Selected)</td><td>18.1018</td><td>0.3799</td><td>0.2044</td></tr><tr><td>2048</td><td>18.0894</td><td>0.3892</td><td>0.2020</td></tr></table>

<table><tr><td>Attention Types</td><td>IW-SSIM 个</td><td>FID ↓</td><td>Dens ↑</td><td>Cov ↑</td></tr><tr><td>Vanilla Cross-Attn.</td><td>0.4186</td><td>25.39</td><td>0.9404</td><td>0.8499</td></tr><tr><td>PMGA (Ours)</td><td>0.4206</td><td>23.70</td><td>0.9741</td><td>0.8592</td></tr></table>

Quantitative Comparison of Autoencoder Reconstruction in Stage 1. Table 2 compares the reconstruction performances of several autoencoders (AE) in Stage 1. Since the baseline models built upon LDM [50] and ResShift [67] benefit from massive pre-training of AE at a much larger scale, the Stage 1 metrics of our ProSR are lower. Nevertheless, leveraging disentangled representations, our ProSR efectively captures discrete scattering anchors through Stages 2 and 3, providing a stronger semantic prior for the SR task, as shown in Table 1.

Qualitative Comparison. Fig. 4 visually compares the SAR ISR results. Since GAN-based models [37, 42, 60] frequently introduce gritty artifacts and vanilla difusion baselines [50, 67, 69] yield oversmoothed textures, both paradigms are prone to structural hallucinations, resulting in a practical issue for reliable SARbased ground observation. In contrast, our ProSR preserves sharp point-scatterer intensities and structural clarity with significantly reduced hallucination. By leveraging discrete scattering anchors, our ProSR provides categorical guidance that prevents from blurred averages or ungrounded features, ensuring highfidelity restoration of physically consistent scattering patterns. More qualitative comparisons are provided in Figs. 11, 12 and 13 of Suppl. B.

![](images/607ef4a5c475e381043f6f846a967451abe2da0c8a6912a6f12f873993a94383.jpg)  
Fig. 5: Impact of $M _ { \mathbf { s e m } }$ (Red: Target, Green: Clutter, Blue: Shadow) on attention. The heatmaps represent pixel-wise averaged attention weights, normalized to the 90th percentile for improved visual clarity.

Downstream Utility Validation (ATR). To validate practical utility, we evaluate downstream ATR on MSTAR [31]. To fairly evaluate the recovery of HRdomain scattering characteristics, all methods use a fixed classifier [24] trained solely on original HR images without SR-specific adaptation. By accurately restoring structural primitives, our ProSR achieves 88.39% accuracy—outperforming the oversampled LR (23.87%) and the strongest baseline (84.18%, a 4.21%p gain). This confirms that preserving scattering physics is more critical for SAR imagery than merely optimizing pixel-level metrics. (See Suppl. B for detailed setups)

## 4.4 Ablation Studies

We conduct a series of ablation experiments to validate the efectiveness of the key components in our ProSR.

Efect of Detail-only Quantization. We compared three types of quantization (Table 3): standard HR-VQ, Dual-VQ (quantizing LR semantic features as well), and our Detail-VQ. (Note that across all comparative ablations, the HR codebook size is fixed to N = 1024, while Dual-VQ utilizes an extra N = 256 codebook for LR quantization.) Standard HR-VQ performs poorest across all metrics, producing over-vivid artifacts by forcedly discretizing stochastic speckles into fixed tokens. While Dual-VQ achieves competitive LPIPS, its pixel-level fidelity (PSNR/SSIM) drops significantly as quantizing the LR components introduces structural errors. In contrast, Detail-VQ optimizes the codebook exclusively for high-frequency details, efectively decoupling the representation from LR distortions. By doing so, Detail-VQ contributes to a cleaner latent space, preventing over-sharpening while preserving superior statistical and structural fidelity.

Impact of Codebook Size. We evaluated performance across N ∈ {512, 1024, 2048}, as summarized in Table 4. While N = 512 is insuficient for complex textures, N = 1024 and 2048 yield comparable LPIPS and SSIM. We selected N = 1024 because it achieves highest PSNR and, more importantly, reduces the complexity of training Stage 3 classification task compared to a larger-sized codebook. This selected size ensures stable training and eficient convergence while maintaining a superior representational power for SAR primitives.

Efectiveness of PMGA Semantic-Guidance. Table 5 shows semantic guidance of PMGA reduces FID by 1.69. Furthermore, Fig. 5 illustrates that PMGA discriminates individual scattering centers, preventing indiscriminate blurred merging in vanilla cross-attention models. PMGA-driven generation, guided by $M _ { \mathrm { s e m } }$ , suppresses structural hallucinations while preserving intrinsic highfrequency integrity, leading to higher Density and IW-SSIM. Ultimately, our mechanism improves both generative fidelity (Dens to 0.9741) and diversity (Cov to 0.8592), ensuring physically reliable SAR reconstruction.

Discrete vs. Continuous (DiT-style) Modeling. To validate the advantages of discrete token modeling over score-based generation, we replaced our Stage 3 transformer with a continuous DiT-style variant—by modifying only the in/output linear projections—under the identical framework. As detailed in Suppl. B, the DiT-style approach struggled with dense, impulse-like scatterers of SAR imagery, often resulting in stochastic structural distortions and degraded LPIPS/FID (0.3211/40.43 for DiT vs. 0.3010/23.70 for ProSR). These results suggest discrete modeling efectively mitigates the structural merging of scatterers.

## 5 Conclusion

In this paper, we proposed ProSR, a generative framework to overcome the limitations of smooth-approximation-based difusion models in SAR ISR, while addressing labeled SAR data scarcity. Unlike standard difusion models that are often prone to stochastic structural distortions—misaligning complex scattering distributions and yielding over-smoothed textures—our ProSR performs semantically-guided discrete token prediction within a discrete detail latent space defined by SADE. To enable semantically-guided prediction, the SPMG utilizes an SSL backbone to extract label-free semantic priors, constructing explicit $M _ { \mathrm { s e m } }$ that guide PMGA to route information flows strictly within consistent categories. By mitigating inter-class confusion, ProSR restores sharp, physically consistent scattering without requiring manually labeled data. Extensive experiments on our 0.25 m resolution benchmark validate that ProSR achieves superior physical realism and structural accuracy while suppressing structural hallucinations. Future work will extend this paradigm to downstream tasks such as object detection, and develop physics-informed no-reference metrics to quantify signal authenticity.

Acknowledgements. This work was supported by National Research Foundation of Korea (NRF) grant funded by the Korean Government [Ministry of Science and ICT (Information and Communications Technology)] (Project Number: RS-2024-00338513, Project Title: AI-based Computer Vision Study for Satellite Image Processing and Analysis).

## References

1. Aithal, S.K., Maini, P., Lipton, Z.C., Zico Kolter, J.: Understanding hallucinations in difusion models through mode interpolation. arXiv e-prints pp. arXiv–2406 (2024)

2. Austin, J., Johnson, D.D., Ho, J., Tarlow, D., Van Den Berg, R.: Structured denoising difusion models in discrete state-spaces. Advances in neural information processing systems 34, 17981–17993 (2021)

3. Awais, C.M., Reggiannini, M., Moroni, D., Karakus, O.: A classification-aware superresolution framework for ship targets in sar imagery. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2026)

4. Bhattacharjee, S., Shanmugam, P., Das, S.: A hybrid algorithm for construction of super-resolution sar imagery for ship detection applications. IETE Journal of Research pp. 1–22 (2025)

5. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

6. Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M.H., Murphy, K., Freeman, W.T., Rubinstein, M., et al.: Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704 (2023)

7. Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: Maskgit: Masked generative image transformer. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2022)

8. Chen, B., Bi, S., Tan, H., Zhang, H., Zhang, T., Li, Z., Xiong, Y., Zhang, J., Zhang, K.: Aligning visual foundation encoders to tokenizers for difusion models. arXiv preprint arXiv:2509.25162 (2025)

9. Chen, C., Abdolshah, M., Shevchenko, V., Li, H., Xu, C., Purkait, P.: Srsr: Enhancing semantic accuracy in real-world image super-resolution with spatially re-focused text-conditioning. arXiv preprint arXiv:2510.22534 (2025)

10. Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., Sutskever, I.: Generative pretraining from pixels. In: International conference on machine learning. pp. 1691– 1703. PMLR (2020)

11. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597–1607. PmLR (2020)

12. Chen, Z., Zhang, C., Wan, C., Zhang, S., Xiong, B.: Dadsr: Degradation-aware difusion super-resolution model for object-level sar image. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2025)

13. Chen, Z., Zhang, S., Xiong, B.: Super-resolving sar images with difusion models: A dual evaluation of metrics and applications. In: 2024 IEEE 17th International Conference on Signal Processing (ICSP). pp. 340–346. IEEE (2024)

14. Chu, X., Tian, Z., Zhang, B., Wang, X., Shen, C.: Conditional positional encodings for vision transformers. In: ICLR 2023 (2023), https://openreview.net/forum? id=3KWnuT-R1bh, accessed 2026-06-30

15. Dalsasso, E., Denis, L., Tupin, F.: As if by magic: Self-supervised training of deep despeckling networks with merlin. IEEE Transactions on Geoscience and Remote Sensing 60, 1–13 (2021)

16. Datcu, M., Huang, Z., Anghel, A., Zhao, J., Cacoveanu, R.: Explainable, physicsaware, trustworthy artificial intelligence: A paradigm shift for synthetic aperture radar. IEEE Geoscience and Remote Sensing Magazine 11(1), 8–25 (2023)

17. Debuysère, S., Trouvé, N., Letheule, N., Lévêque, O., Colin, E.: Quantitative comparison of fine-tuning techniques for pretrained latent difusion models in the generation of unseen sar images. arXiv preprint arXiv:2506.13307 (2025)

18. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44(5), 2567–2581 (2020)

19. Dong, C., Loy, C.C., He, K., Tang, X.: Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence 38(2), 295–307 (2015)

20. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

21. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)

22. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022)

23. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum contrast for unsupervised visual representation learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9729–9738 (2020)

24. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

25. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

26. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

27. Jang, E., Gu, S., Poole, B.: Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 (2016)

28. Jiang, N., Zhao, W., Wang, H., Luo, H., Chen, Z., Zhu, J.: Lightweight superresolution generative adversarial network for sar images. Remote Sensing 16(10), 1788 (2024)

29. Karakuş, O., Achim, A.: On solving sar imaging inverse problems using nonconvex regularization with a cauchy-based penalty. IEEE Transactions on Geoscience and Remote Sensing 59(7), 5828–5840 (2020)

30. Karimi, N., Taban, M.R.: A convex variational method for super resolution of sar image with speckle noise. Signal processing: Image communication 90, 116061 (2021)

31. Keydel, E.R., Lee, S.W., Moore, J.T.: Mstar extended operating conditions: A tutorial. Algorithms for synthetic aperture radar imagery III 2757, 228–242 (1996)

32. Kim, J., Lee, J.K., Lee, K.M.: Accurate image super-resolution using very deep convolutional networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1646–1654 (2016)

33. Kyriou, A., Mpelogianni, V., Nikolakopoulos, K., Groumpos, P.P.: Review of remote sensing approaches and soft computing for infrastructure monitoring. Geomatics 3(3), 367–392 (2023)

34. Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al.: Photo-realistic single image super-resolution using a generative adversarial network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4681–4690 (2017)

35. Li, H., Yang, Y., Chang, M., Chen, S., Feng, H., Xu, Z., Li, Q., Chen, Y.: Srdif: Single image super-resolution with difusion probabilistic models. Neurocomputing 479, 47–59 (2022)

36. Li, W., Yang, W., Hou, Y., Liu, L., Liu, Y., Li, X.: Saratr-x: Toward building a foundation model for sar target recognition. IEEE Transactions on Image Processing 34, 869–884 (2025)

37. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1833–1844 (2021)

38. Lim, B., Son, S., Kim, H., Nah, S., Mu Lee, K.: Enhanced deep residual networks for single image super-resolution. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops. pp. 136–144 (2017)

39. Liu, Z., Zhang, Z., Tang, H.: Semantic-guided difusion model for single-step image super-resolution. arXiv preprint arXiv:2505.07071 (2025)

40. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016)

41. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

42. Ma, C., Rao, Y., Cheng, Y., Chen, C., Lu, J., Zhou, J.: Structure-preserving super resolution with gradient guidance. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7769–7778 (2020)

43. Martins, A., Astudillo, R.: From softmax to sparsemax: A sparse model of attention and multi-label classification. In: International conference on machine learning. pp. 1614–1623. PMLR (2016)

44. Muzeau, M., Frontera-Pons, J., Ren, C., Ovarlez, J.P.: Safe: a sar feature extractor based on self-supervised learning and masked siamese vits. arXiv preprint arXiv:2407.00851 (2024)

45. Naeem, M.F., Oh, S.J., Uh, Y., Choi, Y., Yoo, J.: Reliable fidelity and diversity metrics for generative models. In: International conference on machine learning. pp. 7176–7185. PMLR (2020)

46. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high performance deep learning library. Advances in neural information processing systems 32 (2019)

47. Peng, J., Luo, X., Fu, J., Liu, D.: Confidence-based iterative generation for realworld image super-resolution. In: European Conference on Computer Vision. pp. 323–341. Springer (2024)

48. Reisenhofer, R., Bosse, S., Kutyniok, G., Wiegand, T.: A haar wavelet-based perceptual similarity index for image quality assessment. Signal Processing: Image Communication 61, 33–43 (2018)

49. Ren, X., Wei, W., Xia, L., Huang, C.: A comprehensive survey on self-supervised learning for recommendation. ACM Computing Surveys 58(1), 1–38 (2025)

50. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

51. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image superresolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence 45(4), 4713–4726 (2022)

52. Salimans, T., Ho, J.: Progressive distillation for fast sampling of difusion models. arXiv preprint arXiv:2202.00512 (2022)

53. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. pmlr (2015)

54. Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

55. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456 (2020)

56. Su, H., Li, Y., Xu, Y., Fu, X., Liu, S.: A review of deep-learning-based superresolution: From methods to applications. Pattern Recognition 157, 110935 (2025)

57. Umbra Space: Umbra open dataset: Very high-resolution SAR imagery. https: //umbra.space/open-data (2023), accessed: 2026-02-17

58. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017)

59. Wang, J., Yue, Z., Zhou, S., Chan, K.C., Loy, C.C.: Exploiting difusion prior for real-world image super-resolution. International Journal of Computer Vision 132(12), 5929–5949 (2024)

60. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., Loy, C.C.: Esrgan: Enhanced super-resolution generative adversarial networks. In: The European Conference on Computer Vision Workshops (ECCVW) (September 2018)

61. Wang, Y., Li, S., Dong, G., Liu, H.: Metric or task: A new perspective of sar super-resolution imaging. IEEE Transactions on Geoscience and Remote Sensing (2025)

62. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

63. Wang, Z., Li, Q.: Information content weighting for perceptual image quality assessment. IEEE Transactions on image processing 20(5), 1185–1198 (2010)

64. Wu, R., Yang, T., Sun, L., Zhang, Z., Li, S., Zhang, L.: Seesr: Towards semanticsaware real-world image super-resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 25456–25467 (2024)

65. Xiao, J., Zhang, J., Zou, D., Zhang, X., Ren, J., Wei, X.: Semantic segmentation prior for difusion-based real-world super-resolution. arXiv preprint arXiv:2412.02960 (2024)

66. Yanshan, L., Li, Z., Fan, X., Shifu, C.: Ogsrn: Optical-guided super-resolution network for sar image. Chinese Journal of Aeronautics 35(5), 204–219 (2022)

67. Yue, Z., Wang, J., Loy, C.C.: Resshift: Eficient difusion model for image superresolution by residual shifting. Advances in neural information processing systems 36, 13294–13307 (2023)

68. Zhang, C., Zhang, Z., Deng, Y., Zhang, Y., Chong, M., Tan, Y., Liu, P.: Blind superresolution for sar images with speckle noise based on deep learning probabilistic degradation model and sar priors. Remote Sensing 15(2), 330 (2023)

69. Zhang, L., You, W., Shi, K., Gu, S.: Uncertainty-guided perturbation for image super-resolution difusion model. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17980–17989 (2025)

70. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

71. Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., Fu, Y.: Image super-resolution using very deep residual channel attention networks. In: Proceedings of the European conference on computer vision (ECCV). pp. 286–301 (2018)

72. Zhou, J., Liu, Y., Liu, L., Li, W., Peng, B., Song, Y., Kuang, G., Li, X.: Fifty years of sar automatic target recognition: The road forward. arXiv preprint arXiv:2509.22159 (2025)

73. Zhu, Y., Huang, Y., Yang, M., Mao, D., Zhang, Y., Jiao, L., Zhang, Y., Yang, J.: Sar image super-resolution based on multi-scale edge texture-oriented gan approach. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2025)

# ProSR: Semantic-Prototype-Guided Discrete Modeling for Physically Consistent SAR Super-Resolution - Supplementary Material -

Byoungwoo Kim and Munchurl Kim

Korea Advanced Institute of Science and Technology, Daejeon, Republic of Korea {quddn826, mkimee}@kaist.ac.kr

Supplementary Overview
<table><tr><td rowspan=1 colspan=2>Section Descriptions for Analysis and Discussion</td></tr><tr><td rowspan=1 colspan=1>Sec. A</td><td rowspan=1 colspan=1>Dataset Analysis:SAR dataset characteristics and data distributions</td></tr><tr><td rowspan=1 colspan=1>Sec. B</td><td rowspan=1 colspan=1>Additional Experiments:ATR, qualitative results, DiT/prototype ablations, AE stochastic training</td></tr><tr><td rowspan=1 colspan=1>Sec. C</td><td rowspan=1 colspan=1>Additional Discussions:diffusion-model toy theoretical analysis, Factors affecting SSIM in generative SAR super-resolution</td></tr><tr><td rowspan=1 colspan=1>Sec. D</td><td rowspan=1 colspan=1>Model Complexity and Efficiency:Analysis of parameter counts, FLOPs, and inference runtime</td></tr><tr><td rowspan=1 colspan=1>Sec. E</td><td rowspan=1 colspan=1>Failure Cases and Limitations:Challenges with sub-resolution targets, SSL capacity constraints, validation set diversity</td></tr></table>

In this Supplementary Material, we provide further details and extensive experimental results to complement the main paper. Sec. A expands on the dataset analysis, detailing SAR dataset characteristics and data distributions. Sec. B presents additional experiments, including extra qualitative results, MSTAR ATR evaluation, discrete vs. continuous DiT-style modeling analysis, semantic proto type ablations, and AE stochastic training analysis. Sec. C provides additional discussions, including a toy theoretical analysis of difusion-based modeling and an analysis of SSIM behavior in SAR super-resolution. Sec. D ofers an analysis of model complexity and eficiency, evaluating parameter counts, FLOPs, and inference runtime. Finally, Sec. E discusses failure cases and limitations, including challenges associated with sub-resolution targets, SSL capacity constraints, and validation set diversity.

## A Dataset Analysis

Fig. 6 illustrates the global geographic distribution of the dataset, while Table 6 details the image and patch statistics across classes and incidence angles (10<sup>◦</sup>–50<sup>◦</sup>). The 502 SAR images [57] were manually categorized into five primary classes: Airport, Urban, Industrial, Port, and Natural. This classification process was based on the predominant environmental context and the accompanying site metadata for each acquired scene, merging minor sub-classes (e.g., residential areas, forests) to ensure statistical coherence. To facilitate an intuitive understanding of these semantic categories, Figs. 7 and 8 showcase comprehensive visual examples that exhibit the distinct characteristics of each class. To rigorously evaluate spatial generalization, we partitioned the dataset into 468 training images (124,749 patches) and 34 geographically disjoint validation images (7,703 patches), as detailed in Table 7. Notably, we avoided a standard random split to prevent image-level data leakage. Instead, the validation set was curated with non-overlapping, complex environments. These selected scenes provide a comprehensive and challenging evaluation testbed, encompassing diverse scattering phenomena. This geographical separation ensures that our evaluation reflects true physical generalization against complex SAR scattering, rather than simply memorizing local background statistics. Furthermore, during patch extraction, a selective quality control filter was applied. Since our dataset contains extensive port scenes, naive cropping yields excessive homogeneous, low-backscatter sea patches. We selectively discarded a substantial portion of these uniform regions to prevent generative mode collapse, ensuring a balanced ratio between information-dense target structures and homogeneous speckle areas during training.

![](images/4c3c781d2b0209caff3b68e45d12979334091a8dfbe5e1504389ac6ad0ae4aaa.jpg)  
Fig. 6: Global geographic distribution of the 502 unique SAR scenes in our curated dataset. The markers indicate the diverse locations covering various continents and environmental conditions.

Table 6: Detailed statistics by category and incidence Table 7: Patch distribution angles. I and P denote Images and Patches. for training and validation.
<table><tr><td></td><td colspan="2">10-19°</td><td colspan="2">20-29°</td><td colspan="2">30-39°</td><td colspan="2">40-50°</td><td colspan="2">Total</td></tr><tr><td>Category</td><td>I</td><td>P</td><td>I</td><td>P</td><td>I</td><td>P</td><td>I</td><td>P</td><td>I</td><td>P</td></tr><tr><td>Airport</td><td>2</td><td>242</td><td>51</td><td>14,669</td><td>13</td><td>3,933</td><td>7</td><td>1,914</td><td>73</td><td>20,758</td></tr><tr><td>Urban</td><td>3</td><td>470</td><td>20</td><td>5,800</td><td>26</td><td>6,557</td><td>21</td><td>5,832</td><td>70</td><td>18,659</td></tr><tr><td>Industrial</td><td>0</td><td>0</td><td>3</td><td>625</td><td>6</td><td>1,625</td><td>9</td><td>2,074</td><td>18</td><td>4,324</td></tr><tr><td>Port</td><td>6</td><td>864</td><td>55</td><td>15,544</td><td>89</td><td>24,490</td><td>35</td><td>10,223</td><td>185</td><td>51,121</td></tr><tr><td>Natural</td><td>19</td><td>2,896</td><td>94</td><td>25,391</td><td>25</td><td>5,387</td><td>18</td><td>3,916</td><td>156</td><td>37,590</td></tr><tr><td>Total</td><td></td><td>30 4,472</td><td>223</td><td>62,029</td><td>159</td><td>41,992</td><td>90</td><td>23,959</td><td>502</td><td>132,452</td></tr></table>

<table><tr><td></td><td colspan="3">Patches</td></tr><tr><td>Category</td><td>Train</td><td>Val</td><td>Total</td></tr><tr><td>Airport</td><td>20,486</td><td>272</td><td>20,758</td></tr><tr><td>Urban</td><td>17,725</td><td>934</td><td>18,659</td></tr><tr><td>Industrial</td><td>4,122</td><td>202</td><td>4,324</td></tr><tr><td>Port</td><td>45,701</td><td>5,420</td><td>51,121</td></tr><tr><td>Natural</td><td>36,715</td><td>875</td><td>37,590</td></tr><tr><td colspan="4">Total 124,749 7,703 132,452</td></tr></table>

## B Additional Experiments

## B.1 MSTAR ATR: A Probe for HR Scattering Fidelity

To evaluate whether our SR results preserve HR scattering characteristics, we conduct an ATR transfer experiment on MSTAR dataset [31]. Complex-valued SAR data was degraded to LR via spectral cropping as described in the main text, with a fixed 80%/20% train/test split. The AE of ProSR was fine-tuned for comparable baseline fidelity, and all ISR models pretrained on our data were fine-tuned for 100 epochs on the train split, and evaluated using the final checkpoint; the test split was used only for final evaluation.

Fixed HR-domain Evaluator. We trained a ResNet-50 classifier [24] solely on HR training images. Crucially, this frozen classifier evaluates all oversampled LR and SR outputs without SR-specific retraining or domain adaptation. This protocol evaluates SR reconstruction quality: high accuracy is achieved only if the SR method faithfully preserves target-discriminative HR domain scattering characteristics, rather than relying on classifier adaptation to LR or SR domains. Results and Interpretation. As shown in Table 8, ProSR achieves the highest accuracy (88.39%), outperforming oversampled LR (23.87%) and the strongest baseline, UPSR (84.18%) [69]. Although oversampling and UPSR achieve the highest SSIM and PSNR respectively, they struggle to recover fine target-dependent cues required by the HR-trained ATR model. Our ProSR’s superior ATR performance, aligning with its best FID (12.40) and LPIPS (0.1869), confirms that preserving statistical/structural scattering characteristics is more important for downstream SAR recognition than merely optimizing pixel-wise metrics.

Table 8: MSTAR ATR transfer results using the frozen HR-trained ResNet-50 evaluator.
<table><tr><td colspan="6">Input to ResNet-50 PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Accuracy (%) ↑</td></tr><tr><td>GT reference</td><td></td><td></td><td></td><td></td><td>98.23</td></tr><tr><td>LR (oversampled)</td><td>16.5894</td><td>0.3277</td><td>0.4432</td><td>388.78</td><td>23.87</td></tr><tr><td>ESRGAN</td><td>15.4460</td><td>0.1126</td><td>0.2048</td><td>14.17</td><td>82.11</td></tr><tr><td>SwinIR-GAN</td><td>15.9641</td><td>0.1229</td><td>0.2157</td><td>49.86</td><td>72.06</td></tr><tr><td>SPSR</td><td>15.2255</td><td>0.1114</td><td>0.2179</td><td>16.49</td><td>76.87</td></tr><tr><td>LDM</td><td>16.8514</td><td>0.1475</td><td>0.2669</td><td>86.41</td><td>54.92</td></tr><tr><td>ResShift</td><td>16.6290</td><td>0.1637</td><td>0.2291</td><td>49.18</td><td>81.82</td></tr><tr><td>UPSR</td><td>17.5166</td><td>0.2275</td><td>0.2161</td><td>39.93</td><td>84.18</td></tr><tr><td>ProSR (ours)</td><td>15.9330</td><td>0.1875</td><td>0.1869</td><td>12.40</td><td>88.39</td></tr></table>

## B.2 Additional Qualitative Comparison

For a more comprehensive qualitative evaluation, we provide extended comparisons across diverse scenarios. Fig. 11 highlights the model’s capability to reconstruct tiny scattering points. Figs. 12 and 13 showcase results across various complex environments, such as cities, residential areas and natural terrains.

![](images/1d0141ca5a5bc1bfa40e3d6fa24b9d90d586c8b55817c711838f92612f0459b2.jpg)  
(c) Full-scene overview of industrial images

Fig. 7: Full-scene overview of each category (airport, urban, industrial)

![](images/f13f7811a5df2f93b9c31156bf65206911fdfec41212a1a9645a7d57553764ff.jpg)  
(b) Full-scene overview of natural images

Fig. 8: Full-scene overview of each category (port, natural)

## B.3 Analysis of Discrete vs. Continuous DiT-style Modeling

![](images/4659c039c788ef2b2f074ca26d8db736aed5cfde03a20a35480189fe14c9268a.jpg)  
Fig. 9: Visual comparison in dense scattering regions. (c) The continuous DiT-style model stochastically merges neighboring responses due to conditional averaging. (d) Our discrete token formulation (ProSR) enforces a categorical constraint, mitigating interpolation and preserving sharp, distinct impulse-like signatures.

In the main text, we claim that continuous score-based difusion models may struggle to preserve the dense, impulse-like scatterers characteristic of SAR imagery due to their smooth approximation of the data distribution. To empirically validate this, we establish a strictly controlled comparison between our discrete token modeling (ProSR) and a Difusion Transformer (DiT)-style variant.

Experimental Setup. To isolate the efect of discrete vs. continuous latent modeling, we modified our Stage 3 Transformer to predict the continuous latent features before quantization, conceptually similar to LDM [50]. Specifically, we replaced the discrete token embedding and classification head with linear input/output projections. All other architectural components and training settings—including the transformer backbone, total iterations, identical semantic conditionings $( \mathbf { F } _ { s e m }$ and $M _ { s e m } )$ , and the standard noise-prediction (ϵ) objective—were kept strictly identical to ensure a fair comparison. For inference, we employed 15-step DDIM [54] sampling, aligning with LDM evaluation protocols.

## Analysis of Results. As summarized

in Table 9, the DiT-style variant exhibits degradation in perceptual and statistical realism compared to the discrete ProSR, showing higher FID (40.43 vs. 23.70) and LPIPS (0.3211 vs. 0.3010). This quantitative gap may

Table 9: Quantitative comparison between continuous and discrete latent modeling.
<table><tr><td>Modeling</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓ FID ↓</td><td></td></tr><tr><td>Continuous (DiT-style)</td><td>16.5210</td><td>0.1034</td><td>0.3211</td><td>40.43</td></tr><tr><td>Discrete (ProSR)</td><td>16.9293</td><td>0.1088</td><td>0.3010</td><td>23.70</td></tr></table>

be attributed to diferences in how continuous and discrete latent representations model SAR scattering structures. SAR imagery often contains numerous discrete, high-intensity point scatterers. Under the highly ill-posed LR-to-HR reconstruction setting, the continuous model tends to average over multiple plausible solutions during denoising. In dense scattering regions, this can merge neighboring scattering responses and reduce structural fidelity (see $\mathrm { F i g . \ 9 ( c ) } )$ In contrast, our discrete token formulation (Fig. 9(d)) introduces a categorical constraint through a finite dictionary of learned scattering primitives, reducing interpolated intermediate representations. As a result, ProSR better preserves distinct scattering structures and sharper impulse-like signatures.

## B.4 Efect of the Number of Prototype tokens K

Fig. 10 illustrates the efect of the number of prototype tokens (K) in the Semantic Prototype Map Generator (SPMG) on the resulting semantic maps. For $K = 2$ and $K = 4$ , all training configurations except for the token count were kept identical to those of our proposed setting $\left( K = 3 \right)$ . At $K = 2$ , the number of prototype tokens is insuficient, leading to under-segmentation where the target and clutter regions overlap and fail to separate properly. For $K = 3 ,$ , the model clearly separates target, clutter, and shadow with high confidence, aligning well with the physical structure of SAR imagery.

However, when $K \geq 4$ , redundant prototype tokens begin to appear. Instead of capturing new physical structures, these extra tokens overfit to non-physical background patterns (e.g., Fig. 10-f). As a result, they absorb probability mass from valid regions, reducing the confidence of target and shadow while completely degrading the confidence of the clutter slot (Fig. 10-d). Consequently, increasing K beyond 3 introduces redundant or unused slots that degrade the overall semantic coherence and suppress the assignment of valid regions.

![](images/cadda6a29fa0acfecf9809d99b1f09016ae8461738c3431468fa22a2c3cf9fb9.jpg)  
Fig. 10: Efect of varying the number of prototype tokens (K). (a) LR SAR image. (b) Hard assignment masks with a relative dynamic threshold $\gamma = 0 . 7 . ~ ( \mathrm { c } ) – ( \mathrm { f } )$ Raw probability maps (soft scores) representing specific semantic classes. Note that for $K = 2$ and $K = 4 .$ , only the dynamic thresholding was applied without the residual region extraction logic for clutter areas.

![](images/fcb24b91ff8becf58284ccf3087f693a664e3bb173c417e221b4e576d2e5c449.jpg)  
Fig. 11: Qualitative comparison of SAR ISR (×4) results (Zoom for details)

![](images/a6de6a6d8b1900dfb40e72e0c4e11beeed4a0628802f58b47d6b79d29d32435e.jpg)  
Fig. 12: Comparison: (a) LR, (b) ESRGAN [60], (c) Swin-IR [37], (d) SPSR [42], (e) LDM [50], (f) ResShift [67], (g) UPSR [69], (h) ProSR (Ours), (i) GT. (Zoom for details.)

![](images/d67cfda8b51862ae20ad504d80a803ceec8abe254f7c73c8d2fd01de00c8ba5d.jpg)  
Fig. 13: Comparison: (a) LR, (b) ESRGAN [60], (c) Swin-IR [37], (d) SPSR [42], (e) LDM [50], (f) ResShift [67], (g) UPSR [69], (h) ProSR (Ours), (i) GT. (Zoom for details.)

## B.5 Sensitivity Analysis of Relative Masking Threshold $( \gamma )$

To visually assess the impact of the threshold $\gamma ,$ Fig. 14 presents the generated semantic prototype maps $\left( M _ { \mathrm { s e m } } \right)$ alongside the LR image. The clutter masks are obtained by intersecting the remaining clutter region with the top $c = 3 0 \%$ of pixels ranked by the global clutter score $S _ { i , \mathrm C }$ , isolating stable scattering signatures. For brevity, we omit a separate visual ablation of $c , \mathrm { ~ a s ~ } c = 3 0 \%$ already ensures representative coverage of the predicted clutter distribution. As shown in ${ \mathrm { F i g } } .$ 14, the choice of $\gamma$ afects the structural precision of $M _ { \mathrm { s e m } }$ , which in turn determines the fidelity of the reconstructed scattering primitives.

![](images/972cb93251cca01803c240596ccace4e0e6526e26334eac5638567646de72708.jpg)  
Fig. 14: Visual comparison of semantic prototype maps with varying masking thresholds. While the naive argmax (b) fails to distinguish clutter characteristics within targets, $\gamma = 0 . 7$ (d) provides the most balanced representation of scattering geometry. In contrast, extreme thresholds lead to either the over-inclusive expansion of semantic regions at $\gamma = 0 . 5 ~ \mathrm { ( c ) }$ , or structural omission at $\gamma = 0 . 9 \ ( \mathrm { e } )$ .

Visual Observations and Alignment Analysis: We evaluate the alignment of $M _ { \mathrm { s e m } }$ with the dominant scattering centers under various configurations:

– Naive Argmax: This approach lacks a filtering mechanism, leading to bound ary ambiguity. By over-prioritizing localized scattering peaks, it often misclassifies clutter components within target regions, failing to represent their distinctive clutter characteristics.

– γ = 0.5 (Over-inclusive): At this lower threshold, the over-expansion of target and shadow regions leads to semantic contamination. This hinders the efective exchange of information corresponding to class-specific characteristics, as heterogeneous features are mixed within a single mask.

$- \ \gamma = 0 . 9$ (Under-inclusive): Conversely, an overly restrictive threshold omits valid structural segments. This results in an insuficient pool of pixels for intraclass information exchange in target/shadow regions, leading to fragmented and disconnected scattering representations that fail to capture the complete geometry of the scene.

$- \ \gamma = 0 . 7$ (selected threshold): Our chosen threshold of 0.7 provides the proper balance for informative class-specific interactions. It ensures semantic purity within each region while maintaining structural connectivity, allowing the model to leverage distinct scattering characteristics for high-fidelity reconstruction.

## B.6 Efectiveness of Stochastic Training Strategy in Autoencoder

Table 10 summarizes the impact of the stochastic training probability p on reconstruction performance. At $p = 0 .$ , the model yields the poorest performance. In this setting, the AE behaves similarly to standard HR-only quantization, inevitably causing the model to overrely on the information-rich HR features. As a result, it fails to disentangle the underlying LR structural features from the high-frequency

Table 10: Ablation study on the stochastic training probability p. Best and second-best are bolded and underlined, respectively.
<table><tr><td>Prob. (p)</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>p = 0</td><td>16.5090</td><td>0.1024</td><td>0.3033</td></tr><tr><td> $\bar { p } = 0 . 1$ </td><td>18.0618</td><td>0.3780</td><td>0.2036</td></tr><tr><td> $\begin{array} { l } { p = 0 . 1 } \\ { n = 0 . 2 5 } \end{array}$   $\mathbf { \bar { \rho } } _ { p } = \mathbf { 0 . 2 5 }$ </td><td>18.1018</td><td>0.3799</td><td>0.2044</td></tr><tr><td> $p = 0 . 5$ </td><td>18.1822</td><td>0.3757</td><td>0.2159</td></tr></table>

details. To overcome this, the stochastic inclusion of LR patches during training acts as a regularization mechanism. By exposing the AE to LR inputs, it prevents the model from simply memorizing clean HR textures and encourages it to extract consistent structural features from LR patches. Since SSIM [62] measures the preservation of structural information between images, achieving a high SSIM requires the model to faithfully capture both the overall structural integrity and the fine edge details within it. Therefore, the peak SSIM at $p = 0 . 2 5$ demonstrates that the model successfully preserves essential SAR structural scattering patterns, efectively reconstructing high-frequency HR details anchored on the LR structural foundation. Conversely, when this structural-detail equilibrium collapses, the model becomes biased: $p = 0 . 1$ overemphasizes high-frequency textures to enhance perceptual quality (yielding the best LPIPS [70]), while $p = 0 . 5$ over-relies on low-frequency features to minimize pixel-wise errors (yielding the best PSNR).

## C Additional Discussions

## C.1 Toy analysis of close-mode separation in difusion models

We consider an ambiguous LR observation y associated with two equally plausible HR scatterer-location hypotheses, $x _ { 0 } = - a$ and $x _ { 0 } = a \colon$

$$
p ( x _ { 0 } \mid y ) = { \frac { 1 } { 2 } } \delta ( x _ { 0 } + a ) + { \frac { 1 } { 2 } } \delta ( x _ { 0 } - a ) , \qquad a > 0 .\tag{1}
$$

After Gaussian perturbation, $x _ { t } = x _ { 0 } + \sigma _ { t } \epsilon$ , the conditional distribution becomes

$$
p _ { t } ( x _ { t } \mid y ) = \frac { 1 } { 2 } \mathcal { N } ( x _ { t } ; - a , \sigma _ { t } ^ { 2 } ) + \frac { 1 } { 2 } \mathcal { N } ( x _ { t } ; a , \sigma _ { t } ^ { 2 } ) .\tag{2}
$$

For an ℓ<sub>2</sub>-trained denoiser, the Bayes-optimal prediction is

$$
\mathbb { E } [ x _ { 0 } \mid x _ { t } , y ] = a \left[ \frac { \mathcal { N } ( x _ { t } ; a , \sigma _ { t } ^ { 2 } ) - \mathcal { N } ( x _ { t } ; - a , \sigma _ { t } ^ { 2 } ) } { \mathcal { N } ( x _ { t } ; a , \sigma _ { t } ^ { 2 } ) + \mathcal { N } ( x _ { t } ; - a , \sigma _ { t } ^ { 2 } ) } \right] = a \operatorname { t a n h } \left( \frac { a x _ { t } } { \sigma _ { t } ^ { 2 } } \right) .\tag{3}
$$

When $\sigma _ { t } \gg a .$ , this prediction approaches the midpoint, illustrating the averaging tendency under strong ambiguity.

More importantly, the local geometry of the difusion score $s _ { t } ( x ) = \nabla _ { x } \log p _ { t } ( x \mid y )$ satisfies $\mathsf { \bar { s } } _ { t } ^ { \prime } ( 0 ) = \mathsf { \bar { ( } } a ^ { 2 } - \sigma _ { t } ^ { 2 } ) / \sigma _ { t } ^ { 4 }$ . This formulation reveals a clear phase transition depending on the noise scale:

$$
\left\{ \begin{array} { l l } { \sigma _ { t } > a : } & { x = 0 \mathrm { ~ i s ~ t h e ~ p e a k ~ o f ~ a ~ m e r g e d ~ u n i m o d a l ~ d i s t r i b u t i o n } , } \\ { \sigma _ { t } < a : } & { x = 0 \mathrm { ~ b e c o m e s ~ a ~ v a l l e y ~ s e p a r a t i n g ~ t w o ~ m o d e s } . } \end{array} \right.\tag{4}
$$

Therefore, closely spaced peaks become distinguishable only at low noise levels. Although an ideally exact difusion process can eventually recover both peaks, smaller a postpones their separation to later stages of reverse sampling. In practice, score-approximation errors and finite sampling steps may under-resolve this late-emerging separation, leaving probability mass between the valid peaks and producing overlapped or smeared scattering responses.

ProSR mitigates this vulnerability by representing structural ambiguity through categorical codebook selection. Hard selection reduces direct interpolation between distinct structural codes and encourages reconstructions that remain consistent with learned scattering patterns.

## C.2 Factors Afecting SSIM in Generative SAR ISR

As briefly discussed in the main text, generative models often yield lower SSIM than oversampled LR images. This counterintuitive behavior stems from the strictly pixel-aligned nature of SSIM. An oversampled LR image does not recover missing high-frequency details, but it safely preserves the low-frequency backscatter arrangement, maintaining a baseline covariance with the HR reference. In contrast, generative models are penalized through two primary mechanisms:

Spatial Misalignment of High-Frequency Responses. SSIM relies heavily on local cross-covariance $( \sigma _ { x y } )$ . For sharply localized SAR scatterers, $\sigma _ { x y }$ drops rapidly under minor spatial shifts. Even if a generative model reconstructs a scatterer with near-perfect amplitude and shape, a displacement of just 1-2 pixels drastically penalizes the structural factor of SSIM.

Stochasticity of SAR Speckle. SAR speckle is highly stochastic. A generative model may synthesize a realistic speckle realization (n˜) that perfectly matches the marginal distribution of the true speckle realization (n) present in the HR ground truth. However, because they remain conditionally independent, their cross-covariance approaches zero $( \mathrm { C o v } ( n , \tilde { n } ) \approx 0 )$ . This heavily degrades SSIM even when the generated texture is visually authentic.

Regression-Generation Trade-of. This SSIM degradation reflects the fundamental regression-generation trade-of. Unlike conservative $\ell _ { 1 } / \ell _ { 2 }$ models that yield smoothed outputs to maximize metrics, generative models match the realistic HR distribution. While this trade-of exists in natural images, the high-frequency dominance of localized scatterers and stochastic speckle in SAR causes a far more drastic and counterintuitive drop in SSIM compared to oversampled images.

## D Model Complexity and Eficiency

Table 11 provides a comparative analysis of model complexity and computational eficiency between our ProSR and difusion-based baselines. Notably, our ProSR features the most compact generative core (85.83M) and requires fewer Floating Point Operations (FLOPs) than ResShift [67] via fewer decoding steps. However, it exhibits a higher inference runtime compared to the baselines. We acknowledge this latency as a trade-of to prioritize reconstruction accuracy. Specifically, the 20-layer global attention is essential for capturing long-range structural and semantic dependencies from sparse tokens during early MaskGIT [7] decoding, which efectively suppresses content-inconsistent hallucinations. In practical SAR applications, such as target recognition and infrastructural monitoring, ensuring physical fidelity is strictly prioritized over real-time processing speed. Therefore, we consider this computational cost a necessary and acceptable compromise to guarantee the reliability of the reconstructed scattering signatures. Future work will explore linear-complexity attention and inference step reduction to accelerate sampling while preserving the global receptive field.

Table 11: Comparison of model complexity, FLOPs, and inference runtime. Metrics are measured for generating a $2 5 6 \times 2 5 6$ image (batch = 1) on a single RTX 4090 GPU. Model parameters are denoted as Generative + Other. For our ProSR, ‘Other’ includes the AE (62.85M) [21], SAFE [44] (5.54M) and learnable prototype tokens (576).
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs (G)</td><td>Runtime (ms)</td></tr><tr><td>LDM-15 [50]</td><td>113.60 + 55.32 (AE)</td><td>1,842.22</td><td>90.1</td></tr><tr><td>ResShift-15 [67]</td><td>118.59 + 55.32 (AE)</td><td>2,451.56</td><td>213.4</td></tr><tr><td>UPSR-5 [69]</td><td>119.09 + 2.49 (Aux. SR model)</td><td>748.02</td><td>85.6</td></tr><tr><td>ProSR (Ours)</td><td>85.83 + 68.39 (AE + SAFE)</td><td>1,945.50</td><td>244.5</td></tr></table>

## E Failure Case and Limitations

## E.1 Failure Case

Fig. 15 illustrates representative failure cases of the proposed method when encountering dense sub-resolution targets. While our ProSR reconstructs prominent scatterers well, it struggles in regions where structural geometries become indistinguishable during LR image formation. This limitation arises from the sub-aperture downsampling process, which approximates the efective spatial bandwidth reduction that occurs in practical SAR imaging systems. In areas with highly dense sub-resolution targets or low-contrast boundaries, the expansion of the resolution cell causes the target’s complex signal to undergo coherent vector addition with the surrounding background clutter.

This phase mixing can cause destructive interference, making structural cues indistinguishable from speckle noise in the LR domain. As a result, the geometric information is irreversibly degraded rather than simply blurred. Unlike optical images where edges are preserved under blur, the coherent nature of SAR causes structural cues to be fundamentally altered by destructive interference, leaving no valid geometric evidence for the model to exploit. This lack of reliable signal leaves the semantic priors (e.g., SSL features) with insuficient signal to reconstruct the original topology. Under such severe physical degradation, recovering the exact scattering points remains a highly ill-posed challenge.

![](images/eeffc10d67cb0220b70e4ab8a139dd0f52c6b8da0983153225bec5c2873f1698.jpg)  
(a) LR Image

![](images/429c99103e78f41976d7a7aaea9ee7d221f16f920cde960ef606dc560d6d3fb3.jpg)  
(b) Zoomed LR

![](images/ce9bc56cea3266196a063d5413e22ad68f3b768db96fc5a3dc6763eddb3ffc2d.jpg)  
(c) ESRGAN [60]

![](images/5c748ec8c60a22af8ea74f733f7b5d43475f32cf67b75df6061f27d1806f2458.jpg)  
(d) Swin-IR [37]

![](images/9cff19f942a903c45678c88307118d27fcd931f5ab6f48cc3ca3b19744d7749a.jpg)  
(e) SPSR [42]

![](images/c26b2da8485d54bb39ead488b0adcdb55f5098c8ccdaa44364fc684f8dc3a2f2.jpg)  
(f) LDM [50]

![](images/f89cf170e7af78b5e8aff8ac25ccfee9d8731b7def0b69a82c3fa7a45d295dc2.jpg)  
(g) Resshift [67]

![](images/3429ccd856470e97357a43ec63e191a77b7d2acaba7b95439d3c38e53fc39784.jpg)  
(h) UPSR [69]

![](images/94b9f00d4cf52125055f5aafc1d9ca9ea9fcb6260104d827eaa9fd20ad214934.jpg)  
(i) ProSR (Ours)

![](images/13573874b8cfcd3b387f0312ff61db6272bc0a643b2567ef0389ed5aa833c248.jpg)  
Fig. 15: Failure cases on sub-resolution targets. The GT image shows a distinct, thin structural geometry ((j) red arrow). During the sub-aperture downsampling process, this sub-resolution target undergoes coherent phase mixing, becoming indistinguishable from surrounding speckle clutter in the LR domain (b). Consequently, without suficient underlying geometric evidence, our ProSR (as well as other baselines) fails to faithfully reconstruct the scattering points, resulting in fragmented outputs.  
(j) GT

## E.2 Limitations

Despite its high fidelity, ProSR is constrained by the representational capacity of the SSL features used for $M _ { \mathrm { s e m } }$ and ProSR’s cross-attention $K / V$ values. Therefore, reconstruction precision depends on the richness of the SSL latent space. While local $M _ { \mathrm { s e m } }$ inaccuracies—arising from the semantic ambiguity within SSL latent representations—may cause minor misalignments, ProSR outperforms other SR baselines in statistical fidelity even without Prototype-Map-Guided Attention (PMGA) (Table 5 in the main paper). This indicates that while PMGA further refines structural details and individual scattering primitives, our discrete generative framework is inherently more efective at modeling global scattering distributions and characteristics than difusion-based baselines. Future work will scale the SSL ViT-tiny backbone [20, 44] to enhance semantic guidance.

Finally, regarding dataset diversity, our validation set is concentrated on specific categories, predominantly port patches (Table 7). To maximize the overall training volume and prevent image-level data leakage, we avoided a standard random split. Instead, the validation set was specifically curated with nonoverlapping, highly complex environments. Port images provide a comprehensive evaluation environment, capturing both the most challenging SAR scattering phenomena—such as ships and metallic structures—and diverse surrounding topographies like urban and natural areas within their extensive spatial footprints. Evaluating on these rigorous and composite scenarios provides a robust testbed for assessing complex SAR characteristics and generalization across diverse terrains, while preserving the maximum data scale for training. We intend to incorporate broader sources to establish more balanced benchmarks in future research.