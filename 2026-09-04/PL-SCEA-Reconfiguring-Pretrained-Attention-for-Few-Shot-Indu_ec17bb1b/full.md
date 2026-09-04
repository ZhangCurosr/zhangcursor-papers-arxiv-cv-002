# PL-SCEA: Reconfiguring Pretrained Attention for Few-Shot Industrial Anomaly Detection

Xiaoyu Yang<sup>1</sup>, Qixing Wu<sup>1</sup>, Huixian Zhao<sup>1</sup>, Changlong Jin<sup>1∗</sup>

<sup>1</sup> School of Airspace Science and Engineering, Shandong University, Shandong, China {yangxiaoyu,wuqixing,zhaohuixian}@mail.sdu.edu.cn, cljin@sdu.edu.cn

## Abstract

Vision Foundation Models (VFMs) provide transferable patch representations for few-shot industrial anomaly detection, but their attention computation is typically inherited from pretraining objectives centered on semantic aggregation. This creates a potential mismatch: token relations that support semantic recognition may not adequately expose the localized texture and structural deviations required for anomaly localization. We therefore investigate the hypothesis that the attention computation of a frozen VFM can be reconfigured as a task-relevant component of anomaly detection. We instantiate this idea with Power-Law Self-Correlation Enhanced Attention (PL-SCEA), which retains the semantic context of pretrained query-key attention while constructing tokenadaptive self-correlations over contextualized value features. Positive-correlation filtering and power-law reweighting then emphasize relations that are salient relative to each token’s relational background, without introducing additional trainable attention projections. The resulting features are modeled by a lightweight variational autoencoder that provides a fixed-size reconstruction-based representation of category-specific normality. The two stages serve complementary roles: attention reconfiguration shapes how local relational deviations are represented, while reconstruction-based modeling converts deviations from learned normality into anomaly scores. Across MVTec AD and VisA, the complete framework achieves competitive image-level detection and consistently strong pixellevel localization across the evaluated few-shot settings. Ablations further show that PL-SCEA improves localization with either the VAE or a memory bank under the tested setting. These results support the view that task-aligned attention reconfiguration can improve the anomaly-localization capability of frozen pretrained representations.

## Introduction

Industrial anomaly detection typically trains on normal images to identify defective samples and localize anomalous regions. Feature-matching (Defard et al. 2021; Roth et al. 2022), knowledge-distillation (Deng and Li 2022; Salehi et al. 2021), and reconstruction (Zavrtanik, Kristan, and Skočaj 2021) methods perform well on MVTec AD (Bergmann et al. 2019) and VisA (Zou et al. 2022), but generally require category-specific normal data that cover texture, pose, and structural variation. Rebuilding such datasets for new product categories is costly. Few-shot industrial anomaly detection must therefore establish categoryspecific normality from only K normal reference images while remaining sensitive to local defects.

In few-shot settings, the limited normal references make the quality and organization of pretrained patch representations particularly important. RADIO (Ranzinger et al. 2024b) distills complementary capabilities from DINO (Caron et al. 2021), CLIP (Radford et al. 2021), and SAM (Kirillov et al. 2023), while RADIOv3 (Ranzinger et al. 2024a) further improves the spatial detail of dense representations. Nevertheless, these models inherit attention computations optimized primarily for general-purpose semantic representation. Their pretrained features therefore provide a strong starting point, but the way token information is aggregated may remain mismatched with the localized texture and structural deviations that characterize industrial anomalies.

Attention computation determines how each output patch aggregates evidence from the remaining tokens. In the example shown in Figure 2, the original QK attention preserves semantic context but produces relatively fragmented responses around boundaries and high-contrast regions. Direct QQ, KK, and VV correlations form more spatially continuous patterns, but some also merge locally distinct regions and reduce defect contrast. This observation suggests complementary roles for semantic conditioning and same-type token relations. It further raises a broader question: can the attention computation of a frozen VFM be reconfigured for anomaly localization without discarding the semantic structure learned during pretraining?

Motivated by this question, we argue that the attention computation of a frozen VFM should be treated as a taskrelevant design variable rather than an immutable part of the feature extractor. Importantly, modifying attention is not beneficial by default: directly replacing QK attention with QQ, KK, or VV correlations may discard useful semantic conditioning or over-smooth local structures. An anomalyoriented reconfiguration should instead retain pretrained QK context while selectively reshaping token relations according to their relative relational salience.

We instantiate this design principle with Power-Law Self-Correlation Enhanced Attention (PL-SCEA). PL-SCEA first contextualizes value features through the original QK attention and then computes pairwise self-correlations over the contextualized tokens. Token-wise standardization and power-law reweighting emphasize relations that are salient relative to each token’s relational background, and the resulting weights are used to reaggregate the original value tokens. This design preserves pretrained semantic conditioning while altering how local relational evidence is represented, without introducing additional trainable attention projections.

Attention reconfiguration alone does not define how category-specific normality should be represented from a few reference images. We therefore pair PL-SCEA with a lightweight variational autoencoder (Kingma and Welling 2013) that reconstructs relation-enhanced normal patch features. The two stages address diferent parts of the detection pipeline: PL-SCEA shapes how local relational deviations are exposed in the feature space, whereas the reconstruction model summarizes normality and converts deviations from that normality into patch-level anomaly scores. The resulting parametric normality model has a fixed storage footprint and avoids nearest-neighbor retrieval over an explicit feature memory bank.

We evaluate the complete framework on MVTec AD and VisA across 1–8-shot settings. It maintains competitive image-level detection performance and improves pixel-level F1-max on MVTec AD by 4.1–4.8 percentage points over AnomalyDINO-S. Controlled comparisons further show that replacing QK attention with PL-SCEA improves localization with either the VAE or a memory bank in the MVTec AD 1-shot setting. Meanwhile, the PL-SCEA–VAE combination obtains the strongest pixel-level results in the component comparison. These findings suggest that attention reconfiguration and reconstruction-based normality modeling provide complementary benefits rather than attributing the complete improvement to either component alone. Figure 1 summarizes the training and inference pipeline. Our contributions are summarized as follows:

• We formulate and empirically examine the hypothesis that the attention computation of a frozen vision foundation model can be reconfigured to improve few-shot industria anomaly localization, rather than treating pretrained token interactions as fixed.

• We instantiate this design principle with PL-SCEA, which retains pretrained QK semantic conditioning while constructing token-adaptive, power-law-reweighted selfcorrelations without introducing additional trainable attention projections.

• We combine relation-aware feature construction with reconstruction-based latent normality modeling. Experiments on MVTec AD and VisA show strong localization performance across 1–8-shot settings, while controlled comparisons with both a VAE and a memory bank indicate that the efect of PL-SCEA is not restricted to a single normality model.

## Related Work

Vision Foundation Models transfer broad visual priors to downstream tasks. For anomaly detection, CLIP (Radford et al. 2021) contributes vision-language alignment,

DINO (Caron et al. 2021) contributes self-supervised features, and SAM (Kirillov et al. 2023) contributes dense region awareness. RADIO (Ranzinger et al. 2024b) distills several teachers rather than relying on one pretraining objective. Its encoder combines the global semantics, local correspondence, and dense spatial cues of DINO, CLIP, and SAM. RADIOv3 (Ranzinger et al. 2024a) further sharpens high-resolution dense features. We use a frozen RADIOv3 encoder and modify only how its existing tokens interact.

Few-Shot Anomaly Detection Given a few normal reference images, few-shot industrial anomaly detection aims to learn category-specific normality and localize regions that deviate from it. We focus on low-level visual anomalies, such as scratches, cracks, contamination, and local structural damage, rather than semantic anomalies such as incorrect object categories. Because limited normal samples cannot cover variations in texture, pose, and structure, the central challenge is to construct a robust category-specific normality model from pretrained visual knowledge.

Cross-category methods transfer anomaly-detection knowledge to new categories through unsupervised metalearning (Wu et al. 2021), hierarchical generation (Sheynin, Benaim, and Wolf 2021), or registration-based alignment (Huang et al. 2022). However, they generally require auxiliary category data or target-category adaptation.

Feature-matching methods encode normal images into patch features and use the nearest-neighbor distance to normal features as the anomaly score. PatchCore (Roth et al. 2022) compresses the feature bank through coreset sampling, GraphCore (Xie et al. 2023) models graph relations among support features, and AnomalyDINO (Damm et al. 2025) shows that high-quality DINO features with nearestneighbor retrieval can achieve strong few-shot performance. FastRecon (Fang et al. 2023) reconstructs query features from support features and scores their reconstruction discrepancy. These methods typically remain dependent on referencefeature storage or retrieval.

Despite their diferent scoring mechanisms, these approaches primarily operate on patch representations that have already been formed by the pretrained backbone. They therefore address how normal features are stored, matched, or reconstructed, rather than whether the backbone’s token interaction rule is itself appropriate for anomaly localization.

Vision–language methods compare image or patch features with normal and anomalous textual prompts. Win-CLIP (Jeong et al. 2023), PromptAD (Li et al. 2024), and InCTR (Zhu and Pang 2024) introduce multi-scale windows, learnable prompts, and normal-image context, respectively, while AnoPLe (Lee et al. 2026) and KAG-prompt (Tao et al. 2025) further model interactions among visual prompts, textual prompts, and multi-level features. AnomalyGPT (Gu et al. 2024) extends this paradigm to instruction-guided anomaly analysis. Although these methods can use language semantics to alleviate normal-data scarcity, their localization performance still depends on the quality of dense visual features.

Attention Reshaping in Vision Transformers. Final-layer attention has also been modified to improve local features for dense prediction. SCLIP (Wang, Mei, and Yuille 2024)

![](images/e485ad23b4772a5aff34c3641423b217603cb10dac972135819a6e0296a497e5.jpg)  
Figure 1: Overview of the proposed framework. The first stage reconfigures the final-layer attention computation of a frozen vision foundation model. PL-SCEA preserves QK semantic conditioning, constructs self-correlations over contextualized value features, and uses the reweighted relations to produce relation-aware patch representations. The second stage models these representations with a lightweight VAE. During inference, cosine discrepancies between the PL-SCEA-enhanced features and their deterministic VAE reconstructions are converted into an anomaly map. The two stages respectively control how relational deviations are represented and how deviations from learned normality are scored.

builds attention from query-query and key-key similarities, NACLIP (Hajimiri, Ben Ayed, and Dolz 2025) imposes a spatial-neighborhood prior, and ClearCLIP (Lan et al. 2024) combines same-type attention with a simplified architecture to limit interference from global semantics. For anomaly detection, AnomalyCLIP (Zhou et al. 2024) substitutes valuevalue attention for the original QK relation to sharpen anomalous regions in CLIP features. These studies show that pretrained attention need not remain fixed when dense local representations are required. However, directly replacing QK attention with a same-type relation does not explain how pretrained semantic conditioning and anomaly-sensitive relational structure should be combined. We investigate this issue in few-shot industrial anomaly detection by retaining QK conditioning while reconfiguring the self-correlation structure of contextualized tokens.

## Attention Reconfiguration and Latent Normality Modeling

Our framework separates few-shot anomaly detection into two complementary stages. The first stage concerns representation formation: PL-SCEA reconfigures the final-layer token interactions of a frozen VFM to produce relation-aware patch features. The second stage concerns normality estimation: Latent Normality Modeling reconstructs these features and uses their directional reconstruction discrepancies as anomaly evidence. This decomposition allows us to examine whether attention reconfiguration contributes beyond a particular downstream normality model.

## Token Correlations for Anomaly Detection

The final self-attention block directly influences how output patch representations aggregate information from the remaining tokens. Because the pretrained model is optimized for general-purpose representation learning, its original attention computation may emphasize semantic or highcontrast structures that are useful for recognition but less aligned with fine-grained anomaly localization.

Figure 2 illustrates this mismatch in one example. The original QK attention retains semantic context but produces relatively fragmented responses, whereas QQ, KK, and VV correlations form more continuous relation maps. However, direct same-type correlations can also merge locally distinct patterns. The comparison indicates that changing the attention relation is not suficient by itself; the form of the reconfiguration matters.

Tokens from repetitive textures and regular structures often exhibit consistent relational patterns, while local defects may disturb these patterns. This motivates an attention design that retains QK semantic conditioning but recalibrates token relations according to their relative correlation background. PL-SCEA implements this design by computing self-correlations over semantically contextualized value features and selectively reweighting the resulting relations.

![](images/e62cfaab28e49cc3a50884a895688f045d441dcb6e5a43a513959955d27bdd2c.jpg)  
Figure 2: Comparison of anomaly responses produced by diferent attention branches. The ring-shaped responses in the non-PL-SCEA maps appear to be associated with the shadow above the object, which forms a strong intensity boundary.

## Power-Law Self-Correlation Enhanced Attention

To implement this task-aligned attention reconfiguration, PL-SCEA first obtains semantically conditioned value representations using the original QK attention, extracts the token self-correlation structure from these representations, and constructs new attention relations through token-adaptive standardization and power-law reweighting. Rather than indiscriminately increasing token similarity, PL-SCEA emphasizes above-average self-correlations on a per-token basis, thereby increasing the relative contrast between strongly and weakly related tokens.

Let $\mathbf { X } \in \mathbb { R } ^ { N \times C }$ be the spatial-token input to the final attention layer, where N is the number of spatial patch tokens and C is the feature dimension. Prefix tokens are excluded from the PL-SCEA correlation construction and downstream patch-level anomaly scoring. For a multi-head attention module with H heads, each head has dimension $d = C / H$ . Using the pretrained QKV projections, we obtain Q, K, $\mathbf { V } \in \mathbb { R } ^ { N \times d }$ . For clarity, we formulate the computation for a single head. The original attention is

$$
\mathbf { A } ^ { q k } = \operatorname { S o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } \right) .\tag{1}
$$

The contextualized value representation is then

$$
\mathbf { Y } = \mathbf { A } ^ { q k } \mathbf { V } .\tag{2}
$$

Unlike standard self-attention, we do not use Y directly as the final attention output. Instead, it serves as a semantically conditioned intermediate representation for estimating relational consistency among tokens.

Semantically conditioned self-correlation. To reduce the influence of feature magnitude and emphasize directional consistency among token representations, we first apply $L _ { 2 }$ normalization to each contextualized token along the feature dimension:

$$
\tilde { \mathbf { y } } _ { i } = \frac { \mathbf { y } _ { i } } { \| \mathbf { y } _ { i } \| _ { 2 } + \epsilon } ,\tag{3}
$$

where ϵ ensures numerical stability. Pairwise self-correlation is then

$$
S _ { i j } = \tilde { \bf { y } } _ { i } ^ { \top } \tilde { \bf { y } } _ { j } ,\tag{4}
$$

where $S _ { i j }$ is the cosine similarity between tokens i and j after the initial semantic aggregation.

Token-adaptive correlation standardization. The correlation distributions may difer substantially across query tokens and attention heads. For token i, PL-SCEA computes the row-wise mean and standard deviation:

$$
\mu _ { i } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } S _ { i j } , \qquad \sigma _ { i } = \sqrt { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( S _ { i j } - \mu _ { i } ) ^ { 2 } } .\tag{5}
$$

The standardized correlation is

$$
\hat { S } _ { i j } = \frac { S _ { i j } - \mu _ { i } } { \sigma _ { i } + \epsilon } .\tag{6}
$$

Power-law correlation enhancement. This operation converts absolute cosine similarity into relational salience relative to the correlation background of the current token. When $\hat { S } _ { i j } \ > \ 0$ , the relation between tokens i and $j$ is stronger than the average correlation of token i. Each token is therefore calibrated against its own relational distribution rather than a single global criterion. Weak or moderately strong token interactions may introduce relational noise and obscure subtle local anomalies. PL-SCEA consequently retains only above-average correlations and enhances their contrast with a power-law transformation:

$$
R _ { i j } = \left[ \operatorname* { m a x } ( 0 , \hat { S } _ { i j } ) \right] ^ { \gamma } , \qquad \gamma > 0 .\tag{7}
$$

The exponent $\gamma > 0$ controls the concentration of the retained positive standardized correlations. When $0 < \gamma < 1$ the power transformation compresses the relative diferences between stronger and weaker positive correlations. When $\gamma = 1$ , it preserves the magnitudes of the positive standardized correlations while removing non-positive entries. When $\gamma > 1$ , it increases their relative contrast, assigning greater normalized weight to the strongest relations. We use $\gamma = 2$ by default.

Value reaggregation. We normalize the enhanced correlations row-wise to obtain the final PL-SCEA weights:

$$
D _ { i } = \sum _ { j = 1 } ^ { N } R _ { i j } , \qquad A _ { i j } ^ { \mathrm { { P L - S C E A } } } = \left\{ \begin{array} { l l } { \displaystyle { \frac { R _ { i j } } { D _ { i } } } , } & { \displaystyle { D _ { i } > 0 } , } \\ { \displaystyle { \delta _ { i j } } , } & { \displaystyle { D _ { i } = 0 } , } \end{array} \right.\tag{8}
$$

where $\delta _ { i j }$ is the Kronecker delta. The fallback branch retains the token itself when a row contains no positively salient relation. Therefore,

$$
A _ { i j } ^ { \mathrm { P L - S C E A } } \geq 0 , \qquad \sum _ { j = 1 } ^ { N } A _ { i j } ^ { \mathrm { P L - S C E A } } = 1 .\tag{9}
$$

Finally, PL-SCEA uses the new attention weights to reaggregate the original value tokens:

$$
\mathbf { O } = \mathbf { A } ^ { \mathrm { P L - S C E A } } \mathbf { V } .\tag{10}
$$

The head-wise outputs are concatenated and passed through the frozen pretrained output projection. The surrounding residual connection, LayerNorm placement, and feedforward/MLP sublayer remain unchanged from the original RADIOv3 block. Only the resulting spatial-token features are passed to Latent Normality Modeling and patch-level anomaly scoring.

## Reconstruction-Based Latent Normality Modeling

After attention reconfiguration, Latent Normality Modeling (LNM) uses a lightweight VAE (Kingma and Welling 2013) to model the PL-SCEA-enhanced patch representations. Its role is distinct from that of PL-SCEA: the attention module determines how relational evidence is represented, whereas LNM learns a compact category-specific model of normality and converts reconstruction discrepancies into anomaly scores. Let $\mathbf { x } _ { i }$ denote the i-th enhanced patch feature. During training, $\mathbf { x } _ { i }$ is first $L _ { 2 } .$ -normalized to obtain the clean reconstruction target $\bar { \bf x } _ { i } .$ . Gaussian noise is then added, followed by another normalization, to obtain the encoder input $\widetilde { \mathbf { x } } _ { i } .$

The encoder $E _ { \phi }$ maps $\widetilde { \mathbf { x } } _ { i }$ to the mean $\pmb { \mu } _ { i }$ and standard deviation $\sigma _ { i }$ of the Gaussian posterior $q _ { \phi } ( \mathbf { z } _ { i } \mid  { \widetilde { \mathbf { x } } } _ { i } )$ . During training, the latent variable $\mathbf { z } _ { i }$ is sampled using reparameterization, and the decoder $D _ { \theta }$ reconstructs the normalized patch feature $\widehat { \mathbf { x } } _ { i }$ . The LNM objective combines cosine reconstruction loss with KL regularization:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { L N M } } = \underbrace { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ 1 - \mathrm { s i m } _ { \mathrm { c o s } } \left( \widehat { \mathbf { x } } _ { i } , \bar { \mathbf { x } } _ { i } \right) \right] } _ { \mathcal { L } _ { \mathrm { c o s } } } } \\ & { \quad \quad \quad + \beta \underbrace { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } D _ { \mathrm { K L } } \left[ q _ { \phi } \left( \mathbf { z } _ { i } \mid \widetilde { \mathbf { x } } _ { i } \right) \| \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \right] } _ { \mathcal { L } _ { \mathrm { K L } } } . } \end{array}\tag{11}
$$

The reconstruction term encourages the model to preserve the directional structure of normal features, while the KL term regularizes the latent distribution toward a standard Gaussian. We use a small $\beta$ to limit the risk that excessive regularization suppresses fine-grained variations among normal patches.

At inference, no stochastic sampling is performed. The encoder receives the normalized test feature $\bar { \bf x } _ { i } ,$ and its posterior mean $\pmb { \mu } _ { i }$ is passed to the decoder for deterministic reconstruction. The raw and standardized anomaly scores are

$$
a _ { i } = 1 - \mathrm { s i m } _ { \mathrm { c o s } } \left( \bar { \mathbf { x } } _ { i } , \mathrm { N o r m } _ { 2 } \left[ D _ { \theta } ( \pmb { \mu } _ { i } ) \right] \right) ,\tag{12}
$$

$$
\tilde { a } _ { i } = \frac { a _ { i } - m _ { a } } { s _ { a } + \epsilon } ,\tag{13}
$$

where $m _ { a }$ and $s _ { a }$ are the mean and standard deviation of the reconstruction scores computed from normal training patches. Under this reconstruction-based modeling assumption, normal features are expected to produce smaller directional discrepancies than anomalous features. We therefore use the standardized reconstruction discrepancy as the patch-level anomaly score.

Considering only the category-specific normality state, LNM compresses a variable number of normal patch features into the fixed-size parameter set $\Theta = \{ \phi , \theta \}$ , where $\phi$ and θ denote the encoder and decoder parameters, respectively. Its additional storage complexity is therefore ${ \cal O } ( | \Theta | )$ and does not grow with the number of reference samples. In contrast, the corresponding memory-bank state grows with the number of stored patch features. The frozen backbone, which is shared by both alternatives, is excluded from this comparison.

## Experiments

Datasets. MVTec AD (Bergmann et al. 2019) contains over 5,000 high-resolution images from 15 texture and object categories, with image-level anomaly labels and pixel-level masks. VisA (Zou et al. 2022) contains 9,621 normal and 1,200 anomalous images from products with complex structures and positional variation, covering scratches, dents, discoloration, cracks, and structural defects.

Evaluation Metrics. We report image-level AUROC, AUPR, and F1-max, and pixel-level AUROC, F1-max, and PRO. PRO measures region overlap across thresholds. All metrics are percentages. For PL-SCEA, the main results in Tables 1 and 2 are category macro-averages over three supportset seeds, reported as mean ± sample standard deviation. Baseline entries are taken from the cited sources; because support-set selection and implementation details may difer, comparisons with our local results are descriptive rather than strictly controlled.

Implementation Details. We follow AnomalyDINO (Damm et al. 2025) for preprocessing. A frozen RADIOv3-B serves as the encoder, with PL-SCEA $( \gamma ~ = ~ 2 )$ replacing selfattention in its final Transformer block. L<sub>2</sub>-normalized patch features are modeled by a lightweight VAE with a 64- dimensional latent space. We train for up to 600 epochs using AdamW with a learning rate of $8 \times 1 0 ^ { - 4 }$ , weight decay of $1 \times 1 0 ^ { - 5 }$ , and batch size of 512 on one 12-GB NVIDIA RTX 3080 Ti. At inference, patch scores are standardized using normal training reconstruction errors and Gaussiansmoothed; the image score is the mean of the top 1% patch scores.

## Quantitative Results

MVTec AD. Among the methods listed in Table 1, PL-SCEA obtains the highest or tied-highest reported mean on all image- and pixel-level metrics in the 1- and 2-shot settings. With 4 and 8 shots, its image-level AUROC is respectively 0.4 and 0.2 points below that of AnomalyDINO-S, while it retains the highest reported means on all pixel-level metrics and the highest or tied-highest image-level AUPR and F1-max. In particular, PL-SCEA improves pixel-level F1-max over AnomalyDINO-S by 4.1–4.8 points across the four shot settings. These results indicate that the complete framework provides consistent localization gains while maintaining competitive image-level detection.

VisA. Among the methods listed in Table 2, the complete framework obtains the highest reported means on all classification and segmentation metrics in the 2-, 4-, and 8-shot settings. With one shot, it obtains the highest classification F1-max and all three pixel-level metrics, while its imagelevel AUROC and AUPR are respectively 0.1 and 1.0 points below the highest reported means. The reported performance of PL-SCEA increases with the number of normal references across all six metrics.

<table><tr><td rowspan="2">Shots</td><td rowspan="2">Method</td><td colspan="3">Image-level</td><td colspan="3">Pixel-level</td></tr><tr><td>AUROC</td><td>AUPR</td><td> $F _ { \mathrm { 1 } } \mathrm { - m a x }$ </td><td>AUROC</td><td>F1-max</td><td>PRO</td></tr><tr><td rowspan="6">1-shot</td><td>PatchCore (Roth et al. 2022)</td><td>83.4±3.0</td><td> $9 2 . 2 { \pm } 1 . 5 $ </td><td> $9 0 . 5 { \pm } 1 . 5 $ </td><td> $9 2 . 0 { \pm } 1 . 0 $ </td><td> $5 0 . 4 { \pm } 2 . 1 $ </td><td>79.7±2.0</td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>92.0±0.3</td><td> $9 5 . 8 { \pm } 0 . 2 $ </td><td> $9 2 . 4 { \pm } 0 . 2 $ </td><td> $9 5 . 1 { \pm } 0 . 1 $ </td><td> $5 4 . 2 { \pm } 0 . 0 $ </td><td>90.6±0.2</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>93.1±2.0</td><td> $9 6 . 5 { \pm } 0 . 9 $ </td><td> $9 3 . 7 { \pm } 1 . 1 $ </td><td> $9 5 . 2 { \pm } 0 . 5 $ </td><td> $5 5 . 9 { \pm } 2 . 7 $ </td><td>87.1±1.2</td></tr><tr><td>AdaptCLIP (Gao et al. 2026)</td><td>94.5±0.5</td><td> $9 7 . 5 { \pm } 0 . 1 $ </td><td> $9 5 . 0 { \pm } 0 . 0 $ </td><td> $9 4 . 3 { \pm } 0 . 1 $ </td><td>54.0±0.7</td><td></td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>96.6±0.4</td><td> ${ \bf 9 8 . 2 } \pm { \bf 0 . 2 }$ </td><td> $9 5 . 8 { \pm } 0 . 5 $ </td><td> $9 6 . 8 { \pm } 0 . 1 $ </td><td>60.2±1.1</td><td>92.7±0.1</td></tr><tr><td>PL-SCÉA (Proposed)</td><td>96.9±0.9</td><td> ${ \bf 9 8 . 2 \pm 0 . 8 }$ </td><td> ${ \bf 9 5 . 9 2 0 . 7 }$ </td><td> ${ \bf 9 7 . 5 { \pm 0 . 1 } }$ </td><td>64.3±1.4</td><td>93.7±0.2</td></tr><tr><td rowspan="6">2-shot</td><td>PatchCore (Roth et al. 2022)</td><td>86.3±3.3</td><td> $9 3 . 8 { \pm } 1 . 7 $ </td><td> $9 2 . 0 { \pm } 1 . 5 $ </td><td> $9 3 . 3 { \pm } 0 . 6 $ </td><td>53.0±1.7</td><td>82.3±1.3</td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>92.4±0.3</td><td> $9 6 . 0 { \pm } 0 . 2 $ </td><td> $9 2 . 6 { \pm } 0 . 1 $ </td><td>95.5±0.0</td><td>55.9±0.5</td><td>91.3±0.1</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>94.4±1.3</td><td> $9 7 . 0 { \pm } 0 . 7 $ </td><td> $9 4 . 4 { \pm } 0 . 8 $ </td><td>96.0±0.3</td><td>58.4±1.7</td><td>88.4±0.9</td></tr><tr><td>AdaptCLIP (Gao et al. 2026)</td><td>95.7±0.6</td><td> $9 7 . 9 { \pm } 0 . 2 $ </td><td> $9 5 . 4 { \pm } 0 . 1 $ </td><td>94.5±0.0</td><td>55.0±0.3</td><td></td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>96.9±0.7</td><td> $9 8 . 2 { \pm } 0 . 5 $ </td><td> $9 6 . 1 { \pm } 0 . 3 $ </td><td>97.0±0.2</td><td>61.0±0.5</td><td> $9 3 . 1 { \pm } 0 . 2 $ </td></tr><tr><td>PL-SCÉA (Proposed)</td><td>97.0±1.0</td><td> ${ \bf 9 8 . 6 { \pm 0 . 4 } }$ </td><td> ${ \bf 9 6 . 7 \pm 0 . 0 }$ </td><td>97.9±0.0</td><td> ${ \bf 6 5 . 7 \pm 0 . 5 }$ </td><td>94.2±0.2</td></tr><tr><td rowspan="6">4-shot</td><td>PatchCore (Roth et al. 2022)</td><td> $8 8 . 8 { \pm } 2 . 6 $ </td><td> $9 4 . 5 { \pm } 1 . 5 $ </td><td> $9 2 . 6 { \pm } 1 . 6 $ </td><td> $9 4 . 3 { \pm } 0 . 5 $ </td><td>55.0±1.9</td><td>84.3±1.6</td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td> $9 2 . 8 { \pm } 0 . 2 $ </td><td> $9 6 . 3 { \pm } 0 . 1 $ </td><td> $9 2 . 8 { \pm } 0 . 1 $ </td><td> $9 5 . 9 { \pm } 0 . 0 $ </td><td>56.9±0.1</td><td>91.8±0.1</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>95.2±1.3</td><td> $9 7 . 3 { \pm } 0 . 6 $ </td><td> $9 4 . 7 { \pm } 0 . 8 $ </td><td> $9 6 . 2 { \pm } 0 . 3 $ </td><td>59.5±1.8</td><td>89.0±0.8</td></tr><tr><td>AdaptCLIP (Gao et al. 2026)</td><td> $9 6 . 6 { \pm } 0 . 3 $ </td><td> $9 8 . 4 { \pm } 0 . 2 $ </td><td> $9 6 . 0 { \pm } 0 . 0 \ $ </td><td> $9 4 . 8 { \pm } 0 . 1 $ </td><td> $5 6 . 8 { \pm } 0 . 7 $ </td><td></td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td> $\mathbf { 9 7 . 7 } { \pm } \mathbf { 0 . 2 }$ </td><td> $9 8 . 7 { \pm } 0 . 1 $ </td><td> $9 6 . 6 { \pm } 0 . 0 $ </td><td> $9 7 . 2 { \pm } 0 . 1 $ </td><td> $6 1 . 8 { \pm } 0 . 1 $ </td><td>93.4±0.1</td></tr><tr><td>PL-SCEA (Proposed)</td><td> $9 7 . 3 { \pm } 1 . 1 $ </td><td> ${ \bf 9 8 . 9 2 0 . 3 }$ </td><td> ${ \bf 9 6 . 9 2 0 . 2 }$ </td><td> ${ \bf 9 8 . 0 { \pm 0 . 2 } }$ </td><td> ${ \bf 6 6 . 3 { \pm 1 . 0 } }$ </td><td>94.6±0.4</td></tr><tr><td rowspan="4">8-shot</td><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>93.1±0.2</td><td> $9 6 . 4 { \pm } 0 . 2 $ </td><td> $9 3 . 1 { \pm } 0 . 2 $ </td><td> $9 6 . 2 { \pm } 0 . 1 $ </td><td>57.7±0.2</td><td>92.4±0.2</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>94.6±0.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>98.2±0.2</td><td> $9 8 . 7 { \pm } 0 . 1 $ </td><td> ${ \bf 9 7 . 4 } \pm { \bf 0 . 2 }$ </td><td> $9 7 . 4 { \pm } 0 . 1 $ </td><td> $6 2 . 3 { \pm } 0 . 1 $ </td><td> $9 3 . 8 { \pm } 0 . 1 $ </td></tr><tr><td>PL-SCÈA (Proposed)</td><td> $9 8 . 0 { \pm } 0 . 2 \ $ </td><td> ${ \bf 9 9 . 1 { \pm 0 . 1 } }$ </td><td> ${ \bf 9 7 . 4 } \pm { \bf 0 . 2 }$ </td><td> ${ \bf 9 8 . 2 \pm 0 . 1 }$ </td><td> ${ \bf 6 7 . 1 \pm 0 . 7 }$ </td><td>95.0±0.2</td></tr></table>

Table 1: Few-shot anomaly-detection results on MVTec AD. Higher is better for all metrics. Values are mean ± standard deviation; – indicates an unreported result. The highest reported mean within each shot is in bold.  
![](images/add057145a069ee90856e3a1da5ed5fd5e104649d6c5c36dcecac669d5d04239.jpg)  
Figure 3: Qualitative results on the MVTec AD and VisA datasets.

<table><tr><td rowspan="2">Shots</td><td rowspan="2">Method</td><td colspan="3">Classification</td><td colspan="3">Segmentation</td></tr><tr><td>AUROC</td><td>AUPR</td><td>F1-max</td><td>AUROC</td><td>F1-max</td><td>PRO</td></tr><tr><td rowspan="6">1-shot</td><td>PatchCore (Roth et al. 2022)</td><td>79.9±2.9</td><td>82.8±2.3</td><td>81.7±1.6</td><td>95.4±0.6</td><td>38.0±1.9</td><td>80.5±2.5</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>83.8±4.0</td><td>85.1±4.0</td><td>83.1±1.7</td><td>96.4±0.4</td><td>41.3±2.3</td><td>85.1±2.1</td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>87.4±1.2</td><td>89.0±1.0</td><td>84.3±0.5</td><td>97.8±0.1</td><td>45.1±0.9</td><td>92.5±0.5</td></tr><tr><td>AdaptCLIP (Gao et al. 2026)</td><td>90.5±1.2</td><td>92.3±0.9</td><td>86.5±1.0</td><td>96.8±0.0</td><td>44.6±0.4</td><td></td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>91.2±0.8</td><td>93.3±0.8</td><td>86.9±0.6</td><td>96.0±0.0</td><td>38.5±0.3</td><td>90.0±0.1</td></tr><tr><td>PL-SCEA (Proposed)</td><td>91.1±2.4</td><td>92.3±1.6</td><td>88.3±1.4</td><td>98.0±0.1</td><td>46.8±0.8</td><td>93.1±0.2</td></tr><tr><td rowspan="6">2-shot</td><td>PatchCore (Roth et al. 2022)</td><td>81.6±4.0</td><td>84.8±3.2</td><td>82.5±1.8</td><td>96.1±0.5</td><td>41.0±3.9</td><td>82.6±2.3</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>84.6±2.4</td><td>85.8±2.7</td><td>83.0±1.4</td><td>96.8±0.3</td><td>43.5±3.3</td><td>86.2±1.4</td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>89.7±1.3</td><td>90.7±0.8</td><td>86.3±1.2</td><td>98.0±0.1</td><td>47.6±0.5</td><td>93.4±0.6</td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>92.2±0.3</td><td>94.2±0.3</td><td>87.7±0.3</td><td>96.2±0.0</td><td>39.3±0.2</td><td>90.1±0.1</td></tr><tr><td>AdaptCLIP (Gao et al. 2026)</td><td>92.2±0.8</td><td>93.6±0.6</td><td>88.0±0.7</td><td>97.1±0.0</td><td>46.1±0.4</td><td></td></tr><tr><td>PL-SCEA (Proposed)</td><td>94.0±0.3</td><td>94.7±0.2</td><td>89.9±0.6</td><td>98.3±0.1</td><td>48.7±1.2</td><td>94.2±0.1</td></tr><tr><td rowspan="6">4-shot</td><td>PatchCore (Roth et al. 2022)</td><td>85.3±2.1</td><td>87.5±2.1</td><td>84.3±1.3</td><td>96.8±0.3</td><td>43.9±3.1</td><td>84.9±1.4</td></tr><tr><td>WinCLIP+ (Jeong et al. 2023)</td><td>87.3±1.8</td><td>88.8±1.8</td><td>84.2±1.6</td><td>97.2±0.2</td><td>47.0±3.0</td><td>87.6±0.9</td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>92.6±0.4</td><td>94.5±0.3</td><td>88.4±0.5</td><td>96.2±0.0</td><td>40.0±0.1</td><td>90.2±0.1</td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>92.6±0.9</td><td>92.9±0.7</td><td>88.8±0.9</td><td>98.2±0.0</td><td>49.4±0.3</td><td>94.1±0.1</td></tr><tr><td>AdaptCLIP (Gao et al. 2026)</td><td>93.1±0.2</td><td>94.3±0.2</td><td>88.5±0.2</td><td>97.3±0.0</td><td>47.2±0.5</td><td></td></tr><tr><td>PL-SCEA (Proposed)</td><td>94.9±0.3</td><td>95.5±0.2</td><td>91.1±0.5</td><td>98.6±0.0</td><td>50.6±1.0</td><td>95.1±0.1</td></tr><tr><td rowspan="4">8-shot</td><td>WinCLIP+ (Jeong et al. 2023)</td><td>85.0±0.0</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>APRIL-GAN (Chen, Han, and Zhang 2023)</td><td>93.0±0.2</td><td>94.9±0.3</td><td>88.8±0.2</td><td>96.3±0.0</td><td>40.2±0.1</td><td>90.2±0.0</td></tr><tr><td>AnomalyDINO-S (Damm et al. 2025)</td><td>93.8±0.3</td><td>94.3±0.4</td><td>90.0±0.1</td><td>98.4±0.0</td><td>51.1±0.4</td><td>94.8±0.2</td></tr><tr><td>PL-SCEA (Proposed)</td><td>96.5±0.2</td><td>97.0±0.2</td><td>92.9±0.2</td><td>98.9±0.0</td><td>52.1±0.3</td><td>95.6±0.0</td></tr></table>

Table 2: Few-shot anomaly-detection results on VisA. Higher is better for all metrics. Values are mean ± standard deviation; – indicates an unreported result. The highest reported mean within each shot is in bold.

## Qualitative Analysis

Figure 3 presents qualitative examples covering fine scratches, local damage, and larger structural defects on MVTec AD and VisA. In the displayed cases, the predicted responses overlap the annotated anomalous regions across diferent defect shapes and scales. The 4-shot maps also appear more concentrated than their 1-shot counterparts, with fewer responses in normal textures and backgrounds. These observations are consistent with the quantitative localization results, although the qualitative examples alone do not establish performance across all categories.

## Ablation Studies

Unless otherwise stated, all ablations use a fixed MVTec AD 1-shot support set and report single-run category macroaverages over 15 categories. These controlled comparisons are not numerically interchangeable with the three-seed means in Table 1. Efect of Attention Reconfiguration. Table 3 examines whether changing the attention computation is suficient by itself and whether the form of the reconfiguration matters. Among the tested formulations, PL-SCEA obtains the highest image-level AUROC, image-level AUPR, pixel-level AUROC, and pixel-level F1-max. Relative to the original QK attention, it improves pixel-level F1-max by 3.78 points. Direct QQ, KK, and VV replacements do not reproduce the same overall gains, and VV reduces pixel-level F1-max by 6.35 points relative to QK. These results show that attention modification is not universally beneficial; retaining QK conditioning while selectively reweighting con-

textualized self-correlations is important under this setting. QQ obtains the highest PRO of 93.77%, while PL-SCEA remains within 0.03 points at 93.74%.
<table><tr><td>Attn.</td><td>I-AUROC</td><td>I-AUPR</td><td>P-AUROC</td><td>P-F1</td><td>PRO</td></tr><tr><td>QK</td><td>96.54</td><td>98.06</td><td>97.12</td><td>61.54</td><td>93.55</td></tr><tr><td>QQ</td><td>96.65</td><td>98.31</td><td>97.14</td><td>63.81</td><td>93.77</td></tr><tr><td>KK</td><td>95.82</td><td>97.30</td><td>97.14</td><td>61.74</td><td>93.37</td></tr><tr><td>VV</td><td>92.44</td><td>95.34</td><td>95.45</td><td>55.19</td><td>91.53</td></tr><tr><td>PL-SCEA</td><td>97.26</td><td>98.65</td><td>97.56</td><td>65.32</td><td>93.74</td></tr></table>

Table 3: Ablation of attention variants on MVTec AD under the fixed single-seed 1-shot setting. Values are category macro-averages over 15 categories.

<table><tr><td>Layer I-AUROC P-AUROC P-F1</td><td colspan="3">PRO</td></tr><tr><td>2</td><td>96.0</td><td>97.1 62.5</td><td>93.4</td></tr><tr><td>4</td><td>95.0</td><td>97.0</td><td>61.5 91.9</td></tr><tr><td>6</td><td>95.6</td><td>97.1</td><td>62.3 92.9</td></tr><tr><td>8</td><td>95.8</td><td>96.8</td><td>60.7 92.4</td></tr><tr><td>10</td><td>97.0</td><td>97.3</td><td>63.3 93.5</td></tr><tr><td>12</td><td>97.3</td><td>97.6</td><td>65.3 93.7</td></tr></table>

Table 4: Efect of the PL-SCEA insertion layer on MVTec AD under the fixed single-seed 1-shot setting. Values are rounded to one decimal.

Efect of the Insertion Layer. Table 4 shows that applying

PL-SCEA to the final RADIOv3 layer obtains the highest reported values on all selected metrics, reaching 97.3% imagelevel AUROC, 97.6% pixel-level AUROC, 65.3% pixel-level F1-max, and 93.7% PRO. Compared with layer 2, the final layer improves image-level AUROC and pixel-level F1-max by 1.3 and 2.8 points, respectively. This empirical trend is consistent with the hypothesis that later, more contextualized features provide token relations better suited to the proposed reconfiguration. Based on these results, we apply PL-SCEA only to the final Transformer block.
<table><tr><td>Attn.</td><td>Model</td><td>I-AUROC P-AUROC P-F1 PRO</td></tr><tr><td>PL-SCEA VAE</td><td>97.3</td><td>97.6 65.3 93.7</td></tr><tr><td>QK VAE</td><td>96.5</td><td>97.1 61.5 93.5</td></tr><tr><td>PL-SCEA Memory bank</td><td>97.5</td><td>97.0 64.9 93.1</td></tr><tr><td>QK Memory bank</td><td>97.1</td><td>96.3 61.4 92.9</td></tr></table>

Table 5: Complementarity between PL-SCEA and the normality model on MVTec AD under the fixed single-seed 1-shot setting. QK denotes the original attention. Values are rounded to one decimal; the VAE rows correspond to Table 3.

Complementarity with Normality Modeling. Table 5 separates the efects of attention formulation and normality modeling. Relative to QK attention, PL-SCEA improves pixel-level F1-max by 3.8 points with the VAE and by 3.5 points with the memory bank. It also improves pixel-level AUROC and PRO under both normality models. The effect of attention reconfiguration is therefore observed with both tested downstream alternatives rather than being restricted to the VAE. Meanwhile, pairing PL-SCEA with the VAE obtains the highest pixel-level AUROC, F1-max, and PRO, whereas the memory-bank variant obtains a 0.2-point higher image-level AUROC. These results suggest complementary roles for relation-aware feature construction and reconstruction-based normality modeling, but do not establish a strictly synergistic interaction between them.

Inference-State Memory. Figure 4 compares the category-specific inference state required by the two normality models. The VAE maintains a fixed 14.0-MB footprint across 1–8 shots, whereas the memory bank grows linearly from 34.8 to 278.4 MB. At 8 shots, the VAE requires approximately 5.0% of the memory used by the memory bank, corresponding to a 95.0% reduction. This result highlights the storage benefit of reconstruction-based parametric normality modeling; it does not include the frozen backbone shared by both alternatives.

Efect of the Power Exponent. Figure 5 shows that the reported image-level AUROC and AUPR increase as γ changes from 0.5 to 2, reaching 97.25% and 98.66%, respectively. Both metrics are lower at larger tested values; at $\gamma = 6 ,$ they decrease to 97.12% and 98.43%. The results therefore favor a moderate exponent. This pattern suggests that moderate reweighting can emphasize informative relations, whereas stronger concentration may suppress complementary contextual information. We use $\gamma = 2$ in the remaining experiments.

![](images/59d15941ad658b6efa281506dd3970379fe7245b31ccfb6daa26e9a8b6729260.jpg)

Figure 4: Category-specific inference-state memory across diferent few-shot settings on MVTec AD. The shared frozen backbone is excluded.  
![](images/50dd203e138dd6b2c0ccf88a9e5e54a60d45ce14a9dea61090623f9e7ca42e10.jpg)  
Figure 5: Efect of the power exponent γ on image-level AUROC and AUPR on MVTec AD under the fixed singleseed 1-shot setting.

## Conclusion

This study indicates that the attention computation of a frozen vision foundation model is not merely an inherited implementation detail; when reconfigured in a task-aligned manner, it can improve few-shot industrial anomaly localization. PL-SCEA operationalizes this principle by retaining pretrained QK conditioning while standardizing and powerlaw reweighting the self-correlations of contextualized tokens. Reconstruction-based latent normality modeling complements this representation stage by compactly summarizing normal features and converting their reconstruction discrepancies into anomaly scores. Results on MVTec AD and VisA support the efectiveness of the complete framework, while the MVTec AD 1-shot ablations show that the localization gains of PL-SCEA persist with either a VAE or a memory bank. Within the tested datasets and backbone, these findings support attention reconfiguration as a useful design direction for adapting pretrained representations to localized anomaly detection.

## References

Bergmann, P.; Fauser, M.; Sattlegger, D.; and Steger, C. 2019. MVTec AD–A comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision andpattern recognition, 9592–9600.

Caron, M.; Touvron, H.; Misra, I.; Jégou, H.; Mairal, J.; Bojanowski, P.; and Joulin, A. 2021. Emerging Properties in Self-Supervised Vision Transformers. In Proceedings of the International Conference on Computer Vision.

Chen, X.; Han, Y.; and Zhang, J. 2023. APRIL-GAN: A zero-/few-shot anomaly classification and segmentation method for CVPR 2023 VAND workshop challenge tracks 1&2: 1st place on zero-shot AD and 4th place on few-shot AD. arXiv preprint arXiv:2305.17382.

Damm, S.; Laszkiewicz, M.; Lederer, J.; and Fischer, A. 2025. AnomalyDINO: Boosting patch-based few-shot anomaly detection with DINOv2. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 1319–1329.

Defard, T.; Setkov, A.; Loesch, A.; and Audigier, R. 2021. PaDiM: A patch distribution modeling framework for anomaly detection and localization. In International conference on pattern recognition, 475–489. Springer.

Deng, H.; and Li, X. 2022. Anomaly detection via reverse distillation from one-class embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9737–9746.

Fang, Z.; Wang, X.; Li, H.; Liu, J.; Hu, Q.; and Xiao, J. 2023. FastRecon: Few-shot industrial anomaly detection via fast feature reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 17481–17490.

Gao, B.-B.; Zhou, Y.; Yan, J.; Cai, Y.; Zhang, W.; Wang, M.; Liu, J.; Liu, Y.; Wang, L.; and Wang, C. 2026. AdaptCLIP: Adapting CLIP for universal visual anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 4095–4103.

Gu, Z.; Zhu, B.; Zhu, G.; Chen, Y.; Tang, M.; and Wang, J. 2024. AnomalyGPT: Detecting industrial anomalies using large vision-language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, 1932–1940.

Hajimiri, S.; Ben Ayed, I.; and Dolz, J. 2025. Pay attention to your neighbours: Training-free open-vocabulary semantic segmentation. In Proceedings of the Winter Conference on Applications ofComputer Vision, 5061–5071.

Huang, C.; Guan, H.; Jiang, A.; Zhang, Y.; Spratling, M.; and Wang, Y.-F. 2022. Registration based few-shot anomaly detection. In European conference on computer vision, 303– 319. Springer.

Jeong, J.; Zou, Y.; Kim, T.; Zhang, D.; Ravichandran, A.; and Dabeer, O. 2023. WinCLIP: Zero-/few-shot anomaly classification and segmentation. In Proceedings of the IEEE/CVF

conference on computer vision and pattern recognition, 19606–19616.

Kingma, D. P.; and Welling, M. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollár, P.; and Girshick, R. 2023. Segment Anything. arXiv:2304.02643.

Lan, M.; Chen, C.; Ke, Y.; Wang, X.; Feng, L.; and Zhang, W. 2024. ClearCLIP: Decomposing CLIP representations for dense vision-language inference. In European Conference on Computer Vision, 143–160. Springer.

Lee, Y.; Kim, S.; Moon, D.; Jang, S.; and Yoon, H. 2026. Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 35577–35586.

Li, X.; Zhang, Z.; Tan, X.; Chen, C.; Qu, Y.; Xie, Y.; and Ma, L. 2024. PromptAD: Learning prompts with only normal samples for few-shot anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16838–16848.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International conference on machine learning, 8748–8763. PmLR.

Ranzinger, M.; Barker, J.; Heinrich, G.; Molchanov, P.; Catanzaro, B.; and Tao, A. 2024a. PHI-S: Distribution Balancing for Label-Free Multi-Teacher Distillation. arXiv:2410.01680.

Ranzinger, M.; Heinrich, G.; Kautz, J.; and Molchanov, P. 2024b. AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12490–12500.

Roth, K.; Pemula, L.; Zepeda, J.; Schölkopf, B.; Brox, T.; and Gehler, P. 2022. Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14318–14328.

Salehi, M.; Sadjadi, N.; Baselizadeh, S.; Rohban, M. H.; and Rabiee, H. R. 2021. Multiresolution knowledge distillation for anomaly detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 14902– 14912.

Sheynin, S.; Benaim, S.; and Wolf, L. 2021. A hierarchical transformation-discriminating generative model for few shot anomaly detection. In Proceedings of the IEEE/CVF international conference on computer vision, 8495–8504.

Tao, F.; Xie, G.-S.; Zhao, F.; and Shu, X. 2025. Kernelaware graph prompt learning for few-shot anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 7347–7355.

Wang, F.; Mei, J.; and Yuille, A. 2024. SCLIP: Rethinking self-attention for dense vision-language inference. In European conference on computer vision, 315–332. Springer.

Wu, J.-C.; Chen, D.-J.; Fuh, C.-S.; and Liu, T.-L. 2021. Learning unsupervised metaformer for anomaly detection. In Proceedings of the IEEE/CVF international conference on computer vision, 4369–4378.

Xie, G.; Wang, J.; Liu, J.; Zheng, F.; and Jin, Y. 2023. Pushing the limits of few-shot anomaly detection in industry vision: GraphCore. arXiv preprint arXiv:2301.12082.

Zavrtanik, V.; Kristan, M.; and Skočaj, D. 2021. DRAEM— A discriminatively trained reconstruction embedding for surface anomaly detection. In Proceedings of the IEEE/CVF international conference on computer vision, 8330–8339.

Zhou, Q.; Pang, G.; Tian, Y.; He, S.; and Chen, J. 2024. AnomalyCLIP: Object-agnostic prompt learning for zeroshot anomaly detection. In International Conference on Learning Representations.

Zhu, J.; and Pang, G. 2024. Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 17826–17836.

Zou, Y.; Jeong, J.; Pemula, L.; Zhang, D.; and Dabeer, O. 2022. Spot-the-diference self-supervised pre-training for anomaly detection and segmentation. In European conference on computer vision, 392–408. Springer.