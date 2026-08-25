# MorphoCLIP: Text-Supervised Contrastive Learning for Perturbation Matching in Cell Painting Images

Sukhrobbek Ilyosbekov

Northeastern University

Portland, ME, USA

ilyosbekov.s@northeastern.edu

Shubham Gajjar   
Northeastern University   
Portland, ME, USA   
gajjar.shu@northeastern.edu   
Rongfei Jin   
Northeastern University   
Portland, ME, USA   
jin.ron@northeastern.edu

Abstract—Cell Painting microscopy captures how cells change after a chemical or genetic perturbation. Connecting these images to the perturbations that produced them could make large imaging screens easier to search and interpret, but the task remains difficult because biological effects are subtle and technical variation is substantial. We introduce MorphoCLIP, a contrastive model that links Cell Painting profiles with text descriptions of compounds, CRISPR knockouts and ORF overexpressions. The model keeps its vision and language backbones frozen and trains only a compact cross-channel module and projection layers, so it can be trained on a single consumer GPU. On held-out CPJUMP1 data, MorphoCLIP searches in both directions: from a cell image to its perturbation description and from a description to matching cell images. In both cases, a correct match appears among the top ten results much more often than expected by chance. Adding a replicate-alignment loss makes profiles from repeated experiments more consistent, although this improvement does not yet translate into reliable gene–compound matching. Gene-aware labels and plate correction also show no consistent retrieval benefit. These findings suggest that text supervision can help organize chemical and genetic Cell Painting data. Matching compounds with genetic perturbations, however, remains an open problem.

Index Terms—contrastive learning, cell painting, morphological profiling, CLIP, perturbation matching

## I. INTRODUCTION

Phenotypic drug discovery studies how cells respond to chemical and genetic perturbations. Unlike target-specific assays, it can reveal broad changes in cell state without assuming in advance which biological pathway matters. This matters in a drug-development process that remains slow, expensive and prone to failure [1], [2].

High-content microscopy provides this broader view at scale. The Cell Painting assay [3], [4] uses fluorescent stains to label major cellular structures, producing a five-channel image of each field of view. Changes in cell shape, organelles and spatial organization can reveal how a perturbation acts. The CPJUMP1 benchmark [5] asks whether a model can recover these relationships across a large collection of compound treatments, gene knockouts and gene overexpression experiments.

Existing approaches address different parts of this problem. CellProfiler [6] uses handcrafted measurements, while CLOOME [7] and MolPhenix [8] learn image representations from molecular fingerprints and are therefore limited to compounds. CellCLIP [9] uses text prompts, allowing it to represent both chemical and genetic perturbations, but relies on a large vision backbone and still struggles with crossclass retrieval. CWA-MSN [10] focuses on batch effects and improves gene–gene retrieval, but it has no text branch and cannot be searched with natural-language descriptions.

MorphoCLIP brings text supervision, chemical and genetic perturbations, and optional plate correction into one model. Following CLIP [11], it learns to place a Cell Painting well near the text describing its perturbation. The vision and language backbones remain frozen, and their outputs are cached before training. Only the cross-channel transformer and projection layers are optimized, which keeps the trainable part of the model relatively small.

Our main contributions are:

1) A text-supervised model built on cached DINOv3 features [12] that learns jointly from compounds, CRISPR knockouts and ORF overexpressions.

2) An evaluation of replicate alignment, gene-aware soft labels and condition-relative plate correction. Replicate alignment improves agreement between repeated experiments, while the other changes do not show a consistent benefit.

3) A retrieval protocol with analytic chance baselines, together with evaluation on the standard CPJUMP1 benchmark.

## II. PROBLEM STATEMENT

Given a five-channel Cell Painting image $\mathbf { x } \in \mathbb { R } ^ { 5 \times H \times W }$ and a text description t of the applied perturbation, we learn image and text encoders that map both inputs to the same normalized embedding space:

$$
f _ { \mathrm { i m g } } ( \mathbf { x } ) , \ f _ { \mathrm { t x t } } ( t ) \in \mathbb { R } ^ { d } , \quad \| f _ { \mathrm { i m g } } ( \mathbf { x } ) \| _ { 2 } = \| f _ { \mathrm { t x t } } ( t ) \| _ { 2 } = 1 .\tag{1}
$$

Matching image–text pairs should have high cosine similarity, while unrelated pairs should be farther apart. We set d = 512. Although the equation is written for one image, the model operates at the well level by combining all imaged sites from a well into one vector.

The trained encoders support three retrieval tasks:

• Text-to-image retrieval. Given a perturbation description, retrieve the wells that show the corresponding phenotype.

• Image-to-text retrieval. Given a well, return the perturbation description that best matches its morphology.

• Image-to-image retrieval. Compare well embeddings by cosine similarity, without text at inference time. This is the setting of the standard CPJUMP1 benchmark: do replicate wells of one perturbation match, and do perturbations that share a target match across modalities?

This task is harder than ordinary image–caption alignment for three reasons.

a) Batch effects.: The same perturbation can look different across plates and imaging days. Changes in illumination, staining or focus may be stronger than the biological effect of interest. In CPJUMP1, reproducibility also varies by perturbation type and cell line [5].

b) Imperfect labels.: Biological similarity is not simply positive or negative. Two compounds may share a primary target but differ in their secondary effects, while a gene knockout and a compound may influence the same pathway through different mechanisms. This motivates softer supervision than a standard loss in which every non-matching pair is treated as equally negative.

c) Scale and compute.: CPJUMP1 contains several terabytes of microscopy data. Training both foundation models end to end is impractical on a single consumer GPU, so we run the frozen backbones once, cache their outputs and train only the smaller modules on top.

## III. RELATED WORK

## A. Morphological Profiling

Cell Painting [3] established the five-stain assay now widely used for high-content morphological profiling [4]. The standard analysis pipeline, CellProfiler [6], measures more than a thousand handcrafted properties of each cell. These features remain a strong baseline on CPJUMP1 [5], although matching compounds to genetic perturbations is still difficult.

Handcrafted pipelines are limited to a predefined set of measurements. Self-supervised vision transformers have shown stronger transfer on drug-target and gene-family classification [13], which motivates our use of frozen foundation-model features.

## B. Contrastive Learning for Cell Painting

CLOOME [7] introduced CLIP-style training for Cell Painting by pairing a ResNet image encoder with molecular fingerprints. Its compound retrieval results showed that contrastive supervision works for Cell Painting, but the method does not cover genetic perturbations and treats the five fluorescence channels as RGB.

MolPhenix [8] uses a frozen Phenom-1 image encoder, averages replicate embeddings and introduces softer labels. It improves compound retrieval but depends on a proprietary backbone and likewise does not represent genetic perturbations.

CellCLIP [9] is the closest prior work. It combines templated text prompts with per-channel image encoding, crosschannel attention, attention-based cell pooling and a continuously weighted contrastive loss. Its text branch can describe chemical and genetic perturbations in a common form. However, it relies on a much larger $\mathrm { D I N O v } 2 \mathrm { - g }$ backbone, and its cross-class results show that text supervision alone does not solve technical variation between experiments.

TABLE I  
CELL PAINTING FLUORESCENCE CHANNELS USED IN MORPHOCLIP.
<table><tr><td>Ch</td><td>Stain</td><td>Target</td></tr><tr><td>1</td><td>MitoTracker / Alexa 647</td><td>Mitochondria</td></tr><tr><td>2</td><td>Phalloidin / Alexa 568</td><td>Actin</td></tr><tr><td>3</td><td>WGA / Alexa 488</td><td>Golgi / plasma membrane</td></tr><tr><td>4</td><td>Concanavalin A / Alexa 488</td><td>Endoplasmic reticulum</td></tr><tr><td>5</td><td>Hoechst 33342</td><td>DNA / nucleus</td></tr></table>

## C. Batch-Effect Correction

CWA-MSN [10] addresses batch effects within a masked Siamese network by aligning cells that received the same perturbation across wells and plates. This improves gene–gene retrieval without relying on explicit batch labels. The model has no text branch, but its alignment strategy is closely related to the replicate loss studied here.

## D. Foundation Models for Microscopy

DINOv2 [14] and DINOv3 [12] give self-supervised visual features that transfer to biological imaging despite naturalimage pretraining [13]. DINOv3 adds gram anchoring and a revised training recipe. Channel Vision Transformers [15] show that per-channel encoding followed by cross-channel attention works for multi-channel microscopy, which is the pattern our channel aggregator follows.

On the text side, BioClinical ModernBERT [16] is pretrained on biomedical and clinical literature. A single model can encode both small-molecule and gene descriptions, unlike encoders designed only for chemical strings. This makes it a natural fit for a dataset that mixes compounds with genetic perturbations.

## IV. METHOD

MorphoCLIP learns a common representation for microscopy images and perturbation descriptions. The image branch summarizes the five-channel images from a well, while the text branch encodes a prompt describing the treatment. Both produce normalized 512-dimensional vectors. Training brings matching vectors together and can optionally encourage agreement across replicate wells or correct for plate-specific drift. Figure 1 summarizes the model.

## A. Image Encoder

A Cell Painting site has five fluorescence channels (Table I). Each channel is resized to $2 2 4 \times 2 2 4$ , copied into three channels to match the backbone input and normalized with the backbone’s preprocessing statistics. A frozen DINOv3 ViT-L/16 [12] then produces one 1024-dimensional CLS token per channel,

$$
\left\{ \mathbf { z } _ { c } \right\} _ { c = 1 } ^ { 5 } , \quad \mathbf { z } _ { c } \in \mathbb { R } ^ { 1 0 2 4 } .\tag{2}
$$

![](images/060622da7c3e03cdf9c06b061f31d6a93f179922437077dbeae4bdb2ef77ecb8.jpg)  
Trained: CrossChannelFormer, two projection heads, and logit scale (about 14M parameters). Frozen: DINOv3 and ModernBERT, run once and cached.  
Fig. 1. Overview of MorphoCLIP. A frozen DINOv3 encoder processes each fluorescence channel separately, and the CrossChannelFormer combines the resulting tokens. Site representations are averaged into a well representation and projected into the shared embedding space. A frozen BioClinical ModernBERT encodes the perturbation prompt, followed by a trainable text projection. Optional plate correction is applied to the image embedding before the training losses.

The resulting channel tokens are cached to disk and reused across experiments, so the vision backbone does not run during model training.

a) CrossChannelFormer (CCF).: A one-layer, four-head transformer [17] combines the five channel tokens into one site representation, following the channel-aware designs of CellCLIP [9] and Channel ViT [15]. We first normalize each token and add a learned channel embedding $\mathbf { e } _ { c }$ so the transformer can distinguish the stains. A learned aggregation token q collects information across channels, and its output becomes the site vector:

$$
\mathbf { h } _ { \mathrm { s i t e } } = \big [ \operatorname { C C F } \big ( \mathbf { q } , \hat { \mathbf { z } } _ { 1 } + \mathbf { e } _ { 1 } , \dots , \hat { \mathbf { z } } _ { 5 } + \mathbf { e } _ { 5 } \big ) \big ] _ { 0 } , \quad \hat { \mathbf { z } } _ { c } = \mathbf { z } _ { c } / \| \mathbf { z } _ { c } \| _ { 2 } .\tag{3}
$$

The layer uses pre-normalization, a feed-forward width of 4096 and GELU activations. This allows the model to represent changes that appear across more than one stain. The CCF contains most of MorphoCLIP’s roughly 14 million trainable parameters.

b) Site-to-well pooling.: The training unit is a well. We average its valid site vectors under a mask, producing a fixedsize representation regardless of the number or order of imaged sites.

c) Image projection head.: A two-layer MLP with LayerNorm, GELU and dropout (p = 0.3) maps the well vector $\mathbf { h } _ { \mathrm { i m g } }$ to the shared space:

$$
\begin{array} { r } { f _ { \mathrm { i m g } } ( \mathbf { x } ) = \mathrm { L 2 N o r m } \big ( \mathbf { W } _ { 2 } \operatorname { D r o p } ( \mathrm { G E L U } ( \mathbf { L N } ( \mathbf { W } _ { 1 } \mathbf { h } _ { \mathrm { i m g } } ) ) ) \big ) , } \end{array}\tag{4}
$$

with $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { 5 1 2 \times 1 0 2 4 }$ and $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { 5 1 2 \times 5 1 2 }$

## B. Text Encoder

Each perturbation is described with a template built from CPJUMP1 metadata and external annotations:

• Compound: “Cell Painting morphological profile of {cell line} cells treated with the compound {compound name}. Chemical structure (SMILES): {smiles}. Known target gene: {target gene}. Target protein function: {gene function}. Perturbation modality: chemical compound.”

• CRISPR: “. . . with CRISPR-Cas9 knockout of gene {gene symbol}. Gene description: {gene description}. Gene function: {gene function}. Perturbation modality: CRISPR knockout.”

• ORF: “. . . overexpressing gene {gene symbol} via open reading frame construct. Gene description: {gene description}. Gene function: {gene function}. Perturbation modality: ORF overexpression.”

• Negative control: “. . . treated with DMSO vehicle control. No active perturbation applied.”

Missing fields become “unknown”. Compound annotations come from ChEMBL and the CPJUMP1 metadata. Gene descriptions and functions come from UniProt. Prompts are keyed by the perturbation identifier (broad\_sample), so replicate wells of one perturbation share one text vector.

We encode each prompt with a frozen BioClinical Modern-BERT [16] (150M parameters, 768-d CLS token). A trainable projection head with the same shape as the image head maps the CLS token to the shared space:

$$
\begin{array} { r } { f _ { \mathrm { t x t } } ( t ) = \mathrm { L 2 N o r m } \big ( \mathbf { W } _ { 4 } \operatorname { D r o p } ( \mathrm { G E L U } ( \mathrm { L N } ( \mathbf { W } _ { 3 } \mathbf { h } _ { \mathrm { t x t } } ) ) ) \big ) , } \end{array}\tag{5}
$$

with $\mathbf { W } _ { 3 } \in \mathbb { R } ^ { 5 1 2 \times 7 6 8 }$ and $\mathbf { W } _ { 4 } \in \mathbb { R } ^ { 5 1 2 \times 5 1 2 }$ . The 768-d BERT outputs are cached separately from the projected 512-d vectors, so a change to the projection head never re-runs BERT.

## C. Training Objective

a) Continuously weighted contrastive loss (CWCL).: Standard InfoNCE [18] treats every off-diagonal pair in a batch as a negative. This is inappropriate when two wells share a perturbation and may also be too strict when different perturbations share a target. Following CellCLIP [9], we use a soft-label contrastive loss. For a batch of N wells with logits $\mathbf { S } = \tau \mathbf { F } _ { \mathrm { i m g } } \mathbf { F } _ { \mathrm { t x t } } ^ { \top }$ , the affinity matrix W assigns a weight of 1 to the same perturbation, α to different perturbations with overlapping target genes and 0 otherwise. The loss is

$$
\mathcal { L } _ { \mathrm { C W C L } } = - \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \Bigl ( \tilde { w } _ { i j } \log p _ { i j } + \tilde { w } _ { i j } ^ { \top } \log p _ { i j } ^ { \top } \Bigr ) ,\tag{6}
$$

where $p _ { i j }$ is the softmax of row i of S, $\tilde { w } _ { i j } = w _ { i j } / \sum _ { k } w _ { i k }$ and the transposed terms use $\mathbf { S } ^ { \top }$ and $\mathbf { W } ^ { \top }$ normalized over their own rows. The gene weight α is a hyperparameter; the base model uses $\alpha \ = \ 0$ (binary labels) and the softlabel ablation uses $\alpha = 0 . 6$ . The scale τ is learned through its logarithm, which starts at $\ln ( 1 / 0 . 0 7 )$ and is clamped to [0, ln 100] after every step.

b) Replicate alignment loss.: The text objective brings each well close to its shared description but does not directly bring replicate wells close to one another. We therefore add an image–image contrastive term. Let $P _ { i }$ denote the other wells in the batch with the same perturbation as well i, and let $\mathbf { G } = \tau \mathbf { F } _ { \mathrm { i m g } } \mathbf { F } _ { \mathrm { i m g } } ^ { \top }$ with its diagonal masked. Then

$$
\mathcal { L } _ { \mathrm { r e p } } = - \frac { 1 } { | \{ i : P _ { i } \neq \emptyset \} | } \sum _ { i : P _ { i } \neq \emptyset } \frac { 1 } { | P _ { i } | } \sum _ { j \in P _ { i } } \log \frac { \exp G _ { i j } } { \sum _ { k \neq i } \exp G _ { i k } } .\tag{7}
$$

Wells with no in-batch replicate are dropped from the mean, and the term shares the learned temperature with the text loss. The training loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { C W C L } } + \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } } ;$ the base model uses $\lambda _ { \mathrm { { r e p } } } = 0$ and the ablation uses 0.3. Validation loss is the text term alone, so model selection is not influenced by the added term.

c) Perturbation-aware batches.: The replicate loss requires repeated wells to appear in the same batch. Our sampler therefore groups wells by perturbation, forms pairs, shuffles those pairs and packs them into batches. Where possible, each batch spans more than one plate. This changes batch composition but does not remove plate confounding from individual negative pairs.

d) Condition-relative plate offsets.: Plate effects are a major source of technical variation, but in CPJUMP1 each plate is also tied closely to an experimental condition. Simply subtracting the plate mean would remove useful information about cell line, perturbation type and timepoint. Instead, we measure each plate relative to other plates from the same condition. At the start of each epoch, a gradient-free pass computes the plate means $\bar { \mathbf { f } } _ { p }$ , and each well is shifted by its plate’s offset from the corresponding condition mean:

$$
\tilde { \mathbf { f } } _ { w } = \frac { \mathbf { f } _ { w } - \delta _ { p ( w ) } } { \lVert \mathbf { f } _ { w } - \delta _ { p ( w ) } \rVert _ { 2 } } , \quad \delta _ { p } = \bar { \mathbf { f } } _ { p } - \frac { 1 } { | C ( p ) | } \sum _ { q \in C ( p ) } \bar { \mathbf { f } } _ { q } ,\tag{8}
$$

where $p ( w )$ is the plate containing well w and $C ( \boldsymbol p )$ is the set of plates with the same experimental condition. Offsets within a condition sum to zero, so the correction removes

plate-to-plate drift without shifting the condition mean. They are treated as constants, applied only to image embeddings and stored in the checkpoint for consistent evaluation.

## D. Optimization

We train with AdamW [19] at learning rate $1 0 ^ { - 4 } , ( \beta _ { 1 } , \beta _ { 2 } ) =$ (0.9, 0.999) and $\varepsilon = 1 0 ^ { - 8 }$ . Weight decay 0.2 is applied to multi-dimensional weights but not to biases, normalization parameters or the learned temperature. The learning rate warms up linearly for 100 steps and then follows a cosine schedule to zero. We use FP16 mixed precision, gradient clipping at norm 1.0 and a batch of 256 wells. The base model uses a 100-epoch schedule; ablation runs use a 30-epoch schedule with early stopping after eight epochs without improvement. All backbones remain frozen during optimization.

## V. EXPERIMENTAL SETUP

## A. Dataset

We use the CPJUMP1 pilot [5], a public dataset from the JUMP Cell Painting Consortium. It contains 303 smallmolecule compounds, 160 CRISPR knockouts and 176 ORF overexpressions measured in U2OS and A549 cells. Most perturbation types were collected at two timepoints. Each well contains several imaged sites with five fluorescence channels; the three available brightfield channels are not used in this study.

## B. Data Pipeline

Because the raw images occupy several terabytes, we process one plate at a time. After DINOv3 produces the perchannel features, the raw local copy is removed and the cached features are reused in every training run. Text prompts are handled similarly: ModernBERT encodes each prompt once, and the resulting feature is stored for later training.

## C. Splits

We split the data by perturbation rather than by well. A fixed hash assigns each perturbation to the training, validation or test set in an 80/10/10 ratio, so replicate wells of one perturbation never appear in different splits. All three perturbation types are represented in each split, and control wells are excluded from training and retrieval evaluation. The validation set contains 2,220 wells from 98 perturbations, and the test set contains 1,860 wells from 86 perturbations.

## D. Retrieval Protocol

Replicate wells share a text description. A split therefore contains $P$ distinct text vectors and N well vectors. We evaluate four retrieval settings and compute an analytic chance baseline for each one.

• Well-level image-to-text. Each of the N wells ranks the P unique texts. One candidate is correct. Random Recall@k is $k / P$

• Well-level text-to-image. Each of the P texts ranks all N wells and is scored on the rank of its first correct replicate. With m replicate wells among N candidates, random Recall@k is $\mathbf { \bar { 1 } } - \binom { N - m } { k } / \binom { N } { k }$

• Perturbation-level, both directions. The well embeddings of each perturbation are averaged and re-normalized into one profile, and the P profiles are ranked against the P texts. This is the unit of the standard CPJUMP1 protocol. Random Recall@k is $k / P _ { - }$

We report Recall@{1, 5, 10} and the median rank of the first correct match. Checkpoints are selected using validation text loss and then evaluated on both validation and test data.

## E. Standard CPJUMP1 Benchmark

We also evaluate exported well embeddings with the standard CPJUMP1 benchmark of Chandrasekaran et al. [5]. We follow the reference pipeline: every well is encoded in fp32, including controls, and the negative-control mean is removed before scoring. The benchmark uses the 40 plates that meet its standard experimental criteria and covers a short and long timepoint for each perturbation type. It measures replicability between repeated wells, target matching between perturbations of the same type and gene–compound matching across perturbation types. Results are reported as mean average precision (mAP) and as the fraction of perturbations that pass the benchmark’s significance threshold.

The benchmark’s significance test uses random permutations, so fraction retrieved varies slightly between otherwise identical runs. Replicability mAP is deterministic. We therefore treat small differences in fraction retrieved as descriptive rather than conclusive.

## F. Baselines

• CellProfiler [6] provides the established handcraftedfeature baseline. We use its published CPJUMP1 targetmatching range because we did not rerun the full pipeline.

• CellCLIP [9] is the closest text-supervised baseline. We report its published results and also run the released checkpoint through our short-timepoint benchmark pipeline. This comparison is approximate because Cell-CLIP applies an additional KernelPCA correction and the benchmark significance test is stochastic.

• CWA-MSN [10] is included as a related batch-aware method. Its published benchmarks differ from ours, so a direct numerical comparison is not possible.

## G. Ablation Design

The base model uses binary CWCL labels, random batches, no replicate loss and no plate correction. The ablations use perturbation-aware batches, a shorter schedule and early stopping. Starting from this shared control, we add gene-aware soft labels $( \alpha = 0 . 6 )$ , replicate alignment $( \lambda _ { \mathrm { { r e p } } } = 0 . 3 )$ and condition-relative plate offsets one at a time, followed by a run that combines all three. All runs use seed 42, batch size 256 and the same data split. Since a single retrieved perturbation changes recall by about one percentage point, small differences should not be interpreted as reliable effects.

TABLE II  
BASE-MODEL RETRIEVAL ON VALIDATION AND TEST DATA. IMAGE-TO-TEXT RETRIEVAL RANKS ONE DESCRIPTION PER PERTURBATION. TEXT-TO-IMAGE RETRIEVAL RANKS EITHER INDIVIDUAL WELLS OR PERTURBATION-LEVEL MEAN PROFILES. RANDOM ROWS SHOW ANALYTIC CHANCE PERFORMANCE.
<table><tr><td>Split</td><td>Direction</td><td>R@1</td><td>R@5</td><td>R@10</td><td>Med. rank</td></tr><tr><td rowspan="6">Val</td><td>well image → text</td><td>5.9</td><td>23.3</td><td>39.4</td><td>14</td></tr><tr><td>well text → image</td><td>4.1</td><td>15.3</td><td>23.5</td><td>30</td></tr><tr><td>pert. image → text</td><td>4.1</td><td>20.4</td><td>38.8</td><td>13</td></tr><tr><td>pert. text → image</td><td>5.1</td><td>18.4</td><td>39.8</td><td>13</td></tr><tr><td>random, single positive</td><td>1.0</td><td>5.1</td><td>10.2</td><td>49.5</td></tr><tr><td>random, well text → image</td><td>1.0</td><td>5.0</td><td>9.7</td><td>99</td></tr><tr><td rowspan="6">Test</td><td>well image → text</td><td>5.5</td><td>22.0</td><td>41.4</td><td>13</td></tr><tr><td>well text → image</td><td>4.7</td><td>18.6</td><td>26.7</td><td>25</td></tr><tr><td>pert. image → text</td><td>4.7</td><td>17.4</td><td>37.2</td><td>14</td></tr><tr><td>pert. text → image</td><td>8.1</td><td>23.3</td><td>44.2</td><td>13</td></tr><tr><td>random, single positive</td><td>1.2</td><td>5.8</td><td>11.6</td><td>43.5</td></tr><tr><td>random, well text → image</td><td>1.2</td><td>5.7</td><td>11.0</td><td>86</td></tr></table>

## H. Implementation

All experiments run on one RTX 5080 with 16 GB of memory. We use AdamW with a learning rate of $1 0 ^ { - 4 }$ weight decay of 0.2, 100 warm-up steps followed by cosine decay, FP16 training and gradient clipping at 1.0. The Cross-ChannelFormer has one layer and four attention heads; both projection heads use a hidden width of 512 and dropout of 0.3. Code, configurations and split manifests are released with the paper.

## VI. RESULTS AND DISCUSSION

Retrieval is measured on the held-out splits described in Section V-D, with chance performance shown alongside the model results. The standard benchmark uses its eligible CPJUMP1 plates.

## A. Cross-Modal Retrieval

Table II reports the base model on validation and test data. At the perturbation level, the correct match appears in the top ten for roughly two out of five queries in either direction. This is more than three times the expected chance rate. Similar performance on the unseen test perturbations suggests that the model is learning a relationship between morphology and perturbation text rather than memorizing the training set.

Text-to-image retrieval is harder at the well level because each description must search a much larger pool of individual wells. Even so, the first matching replicate usually appears near the top of the ranking. Averaging replicate wells into perturbation profiles produces more balanced results in the two retrieval directions.

## B. Ablations

Table III lists the single-factor runs.

No single addition improves retrieval consistently across validation and test data. The ranking of the variants changes between splits, and a difference of several percentage points represents only a few perturbations. The combined model is competitive in both directions but is not uniformly best. The observed differences are therefore too small to establish a reliable retrieval gain.

TABLE III  
SINGLE-FACTOR ABLATIONS FROM A SHARED CONTROL. VALUES ARE PERTURBATION-LEVEL RECALL@10 IN PERCENT AND MEDIAN RANK. RANDOMRECALL@10 IS 10.2 ON VALIDATION AND 11.6 ON TEST.
<table><tr><td></td><td></td><td colspan="3">Validation</td><td colspan="3">Test</td></tr><tr><td>Run</td><td>Change from control</td><td>i→t</td><td>t→i</td><td>Med.</td><td>i→t</td><td>t→i</td><td>Med.</td></tr><tr><td>Base model</td><td>100-epoch schedule, random batches</td><td>38.8</td><td>39.8</td><td>13</td><td>37.2</td><td>44.2</td><td>14</td></tr><tr><td>Control</td><td>none</td><td>37.8</td><td>37.8</td><td>13</td><td>50.0</td><td>41.9</td><td>10</td></tr><tr><td>Soft labels</td><td> $\alpha = 0 . 6$ </td><td>37.8</td><td>42.9</td><td>14</td><td>44.2</td><td>47.7</td><td>12</td></tr><tr><td>Replicate loss</td><td> $\lambda _ { \mathrm { { r e p } } } = 0 . 3$ </td><td>44.9</td><td>41.8</td><td>13</td><td>47.7</td><td>46.5</td><td>12</td></tr><tr><td>Plate offsets</td><td>condition-relative correction</td><td>41.8</td><td>38.8</td><td>13</td><td>41.9</td><td>43.0</td><td>12</td></tr><tr><td>All three</td><td>soft + replicate + offsets</td><td>42.9</td><td>43.9</td><td>13</td><td>46.5</td><td>45.3</td><td>11</td></tr></table>

## C. Standard CPJUMP1 Benchmark

Table IV reports fraction retrieved and mAP on the standard benchmark for the base model, the replicate-loss run, the plateoffset run and the combined run.

a) Replicability.: The replicate loss improves mAP on all twelve tracks, increasing pooled mAP from 0.298 to 0.343. The combined model reaches 0.369, while plate offsets alone do not improve the pooled result. This is the clearest change in the study, but it is also closely tied to the training objective: the loss explicitly pulls replicate wells together, and the benchmark measures how well those wells retrieve one another.

b) Target matching.: No variant is consistently better on the larger compound tracks. The smaller CRISPR tracks are especially sensitive to the benchmark’s significance test. Overall, the target-matching results do not show that replicate alignment or plate correction improves relationships beyond perturbation identity.

c) Gene–compound matching.: Cross-modality retrieval is weak for every variant. Almost no gene–compound relationships pass the significance threshold, and pooled mAP shows no consistent ordering. The tested training changes therefore do not solve the central problem of matching compounds to genetic perturbations with the same target.

## D. Comparison with Baselines

a) Same-harness comparison.: Table V compares the released CellCLIP checkpoint with MorphoCLIP on the five short-timeline CPJUMP1 tracks available for both models. The shared-track mean gives a clearer summary than any single cell line or perturbation type. MorphoCLIP’s base model and CellCLIP are close on average, while the replicateloss and combined variants are higher on most tracks. The comparison remains approximate because CellCLIP uses its own encoder path and a control-fitted KernelPCA correction, whereas MorphoCLIP does not.

Because fraction retrieved depends on a stochastic significance test, the differences in Table V should not be read as a definitive ranking. The changes from the MorphoCLIP base to its own variants are more informative than the unpaired comparison with CellCLIP.

b) Published benchmark context.: Table VI groups published results only where the dataset, task and metric match. On CPJUMP1 compound target matching, the MorphoCLIP base spans a similar range to CellProfiler and has a four-track mean of 16.3%. On the separate RxRx3-core CORUM benchmark, CWA-MSN improves over CellCLIP, but MorphoCLIP has not been evaluated there.

## E. Cached Feature Analysis

We checked whether channel identity survives the frozen backbone using 500 sites sampled with seed 42 from plate BR00116991. The reproducible diagnostic output is stored with the figure sources. A variance decomposition attributes 31.1% of variance to between-channel differences, 37.1% to between-site differences and 31.8% to the residual (Figure 2). Same-channel tokens from different sites have mean cosine similarity 0.84, while cross-channel tokens from different sites average 0.75 (Figure 3). The two-dimensional PCA projection in Figure 4 provides a qualitative view of this channel structure.

## VII. CONCLUSION

MorphoCLIP provides a single text-supervised representation for compounds, CRISPR knockouts and ORF overexpressions in Cell Painting data. Because the vision and language backbones stay frozen, only a small set of components needs to be trained. On held-out perturbations, the model retrieves the correct image or description well above chance rates; a shared image–text space can therefore capture meaningful morphological signal.

The replicate-alignment loss consistently brings repeated wells closer together and improves replicability across all benchmark tracks. This is evidence that the loss achieves its immediate goal, but it should not be taken as proof of better biological understanding because replicability is also the quantity it directly optimizes. Gene-aware labels and conditionrelative plate correction do not provide a clear retrieval benefit in the current experiments.

TABLE IV  
STANDARD CPJUMP1 BENCHMARK. EACH ENTRY GIVES FRACTION RETRIEVED, WITH MAP IN PARENTHESES. SUMMARY FRACTIONS ARE UNWEIGHTED ACROSS TRACKS; SUMMARY MAP IS POOLED ACROSS SCORED PROFILES.
<table><tr><td>Track</td><td></td><td>Base</td><td>Replicate loss</td><td>Plate offsets</td><td>All three</td></tr><tr><td rowspan="2">Compound A549</td><td>short</td><td>0.353 (0.392)</td><td>0.435 (0.458)</td><td>0.327 (0.384)</td><td>0.497 (0.493)</td></tr><tr><td>long</td><td>0.513 (0.526)</td><td>0.650 (0.583)</td><td>0.503 (0.513)</td><td>0.703 (0.626)</td></tr><tr><td rowspan="2">Compound U2OS</td><td>short</td><td>0.297 (0.344)</td><td>0.379 (0.404)</td><td>0.310 (0.351)</td><td>0.422 (0.439)</td></tr><tr><td>long</td><td>0.369 (0.413)</td><td>0.471 (0.474)</td><td>0.395 (0.421)</td><td>0.503 (0.496)</td></tr><tr><td rowspan="2">CRISPR A549</td><td>short</td><td>0.173 (0.286)</td><td>0.271 (0.343)</td><td>0.092 (0.239)</td><td>0.340 (0.375)</td></tr><tr><td>long</td><td>0.183 (0.296)</td><td>0.376 (0.356)</td><td>0.134 (0.282)</td><td>0.366 (0.385)</td></tr><tr><td rowspan="2">CRISPR U2OS</td><td>short</td><td>0.026 (0.175)</td><td>0.049 (0.207)</td><td>0.052 (0.182)</td><td>0.078 (0.232)</td></tr><tr><td>long</td><td>0.170 (0.277)</td><td>0.219 (0.290)</td><td>0.160 (0.259)</td><td>0.258 (0.318)</td></tr><tr><td rowspan="2">ORF A549</td><td>short</td><td>0.006 (0.136)</td><td>0.012 (0.158)</td><td>0.012 (0.138)</td><td>0.019 (0.174)</td></tr><tr><td>long</td><td>0.000 (0.128)</td><td>0.006 (0.142)</td><td>0.000 (0.126)</td><td>0.006 (0.141)</td></tr><tr><td rowspan="2">ORF U2OS</td><td>short</td><td>0.043 (0.171)</td><td>0.099 (0.223)</td><td>0.062 (0.209)</td><td>0.099 (0.221)</td></tr><tr><td>long</td><td>0.006 (0.133)</td><td>0.012 (0.146)</td><td>0.012 (0.140)</td><td>0.031 (0.160)</td></tr><tr><td>Replicability summary (12 tracks)</td><td></td><td>0.178 (0.298)</td><td>0.248 (0.343)</td><td>0.172 (0.292)</td><td>0.277 (0.369)</td></tr><tr><td>Target-matching summary (8 tracks)</td><td></td><td>0.339 (0.197)</td><td>0.262 (0.160)</td><td>0.420 (0.194)</td><td>0.172 (0.148)</td></tr><tr><td>Gene-compound summary</td><td></td><td>0.005 (0.123)</td><td>0.000 (0.082)</td><td>0.010 (0.140)</td><td>0.029 (0.082)</td></tr></table>

TABLE V

SHORT-TIMELINE CPJUMP1 REPLICABILITY ON TRACKS SHARED BY CELLCLIP AND MORPHOCLIP. VALUES ARE FRACTION RETRIEVED; THE FINAL COLUMN IS THE UNWEIGHTED MEAN OF THE FIVE TRACKS. CELLCLIP USES THE RELEASED CHECKPOINT AND ITS OWN PREPROCESSING PATH.
<table><tr><td>Model</td><td>Compound A549</td><td>Compound U2OS</td><td>CRISPR A549</td><td>CRISPR U2OS</td><td>ORF U2OS</td><td>Mean</td></tr><tr><td>CellCLIP [9]</td><td>0.366</td><td>0.353</td><td>0.020</td><td>0.118</td><td>0.119</td><td>0.195</td></tr><tr><td>MorphoCLIP base</td><td>0.353</td><td>0.297</td><td>0.173</td><td>0.026</td><td>0.043</td><td>0.178</td></tr><tr><td>+ replicate loss</td><td>0.435</td><td>0.379</td><td>0.271</td><td>0.049</td><td>0.099</td><td>0.247</td></tr><tr><td>MorphoCLIP, all three</td><td>0.497</td><td>0.422</td><td>0.340</td><td>0.078</td><td>0.099</td><td>0.287</td></tr></table>

TABLE VI

PUBLISHED BENCHMARK CONTEXT. RESULTS ARE COMPARABLE WITHIN EACH DATASET–TASK GROUP, NOT ACROSS GROUPS. FR DENOTES FRACTION RETRIEVED.
<table><tr><td>Method</td><td>Dataset</td><td>Task</td><td>Metric</td><td>Result</td></tr><tr><td>CellProfiler [5]</td><td>CPJUMP1</td><td>Compound target matching</td><td>FR range</td><td>4.3-25.1%</td></tr><tr><td>MorphoCLIP base</td><td>CPJUMP1</td><td>Compound target matching</td><td>FR range</td><td>7.5–31.5%</td></tr><tr><td>CellCLIP [9]</td><td>RxRx3-core</td><td>CORUM gene-gene</td><td>Recall</td><td>0.354</td></tr><tr><td>CWA-MSN [10]</td><td>RxRx3-core</td><td>CORUM gene-gene</td><td>Recall</td><td>0.386</td></tr></table>

The main unresolved problem is gene–compound matching. None of the tested variants reliably connects genetic perturbations with compounds that act on the same targets. MorphoCLIP therefore succeeds as a retrieval model for perturbation identity and replicate structure, but not yet as a general model of shared biological mechanism.

## A. Limitations and Future Work

Small differences between variants remain uncertain, and the benchmark’s permutation test introduces additional variation in fraction retrieved. Comparisons with CellProfiler and CellCLIP should be read as context rather than a strict ranking because their preprocessing and evaluation procedures are not identical. More importantly, a stronger case for biological usefulness will require improvement on tasks that are not directly optimized by the replicate loss.

Feature Variance Decomposition  
![](images/3a9e55d05d1956d72e5f04a78870bacc1ca8803a4c0a7e7170afec3ba2f3e6fc.jpg)  
Fig. 2. Variance decomposition of the cached DINOv3 tokens on plate BR00116991: between-channel, between-site and residual components.

One likely source of the weak cross-modality result is the mismatch between the primary-target annotations used during training and the broader target lists used by the benchmark. Future work should align these definitions and test whether richer target supervision improves gene–compound retrieval.

PCA of CLS Features (colored by channel)  
![](images/c172df90baa3ac7877ea47ee96e71d42926f021c91ffc0d04e464f35cd2685a4.jpg)  
Fig. 3. Mean cosine similarity between fluorescence channels in the cached token space. The DNA channel is the most distinct.

![](images/22e08839c072da2432d4aa6984ca70ede1b48fd3f69d96aabe05644339a51570.jpg)  
Fig. 4. First two principal components of the cached 1024-d channel tokens, colored by fluorescence channel.

Prompt ablations are also needed to determine whether chemical structures and gene-function descriptions contribute useful information. Finally, evaluation on the larger and more diverse JUMP corpus is needed to test whether the learned representation transfers across sites, cell lines and experimental conditions.

Code, configurations, split manifests and the evaluation harness are available at https://github.com/suxrobGM/ morphoclip.

## REFERENCES

[1] O. J. Wouters, M. McKee, and J. Luyten, “Estimated research and development investment needed to bring a new medicine to market,

2009–2018,” JAMA, vol. 323, no. 9, pp. 844–853, 2020.

[2] D. Sun, W. Gao, H. Hu, and S. Zhou, “Why 90% of clinical drug development fails and how to improve it?” Acta Pharmaceutica Sinica B, vol. 12, no. 7, pp. 3049–3062, 2022.

[3] M.-A. Bray, S. Singh, H. Long, and A. E. Carpenter, “Cell painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes,” Nature Protocols, vol. 11, no. 9, pp. 1757–1774, 2016.

[4] S. Seal et al., “Cell painting: A decade of discovery and innovation in cellular imaging,” Nature Methods, 2024.

[5] S. N. Chandrasekaran, J. Ackerman, E. Alix, D. M. Ando, J. Arevalo, M. Berber, N. Beresford, L. Bornholdt, B. Bostock et al., “JUMP cell painting dataset: morphological impact of 136,000 chemical and genetic perturbations,” Nature Methods, vol. 21, pp. 1114–1121, 2024.

[6] A. E. Carpenter, T. R. Jones, M. R. Lamprecht, C. Clarke, I. H. Kang, O. Friman, D. A. Guertin, J. H. Chang, R. A. Lindquist, J. Moffat et al., “CellProfiler: image analysis software for identifying and quantifying cell phenotypes,” Genome Biology, vol. 7, no. 10, p. R100, 2006.

[7] A. Sanchez-Fern ´ andez, T. Langer, G. Ecker ´ et al., “CLOOME: Contrastive learning unlocks bioimaging databases for queries with chemical structures,” Nature Communications, 2023.

[8] P. Fradkin et al., “How molecules impact cells: Unlocking contrastive PhenoMolecular retrieval,” in Advances in Neural Information Processing Systems, 2024.

[9] M. Lu, E. Weinberger, C. Kim, and S.-I. Lee, “CellCLIP: Learning perturbation effects in cell painting via text-guided contrastive learning,” in Advances in Neural Information Processing Systems, 2025.

[10] P.-J. Huang, Y.-H. Liao, S. Kim, N. Park, J. Park, and D. Shin, “Efficient cell painting image representation learning via cross-well aligned masked siamese network,” arXiv preprint arXiv:2509.19896, 2025.

[11] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.

[12] O. Simeoni et al., “DINOv3: Self-supervised learning via gram anchoring,” arXiv preprint arXiv:2508.10104, 2025.

[13] N. Moshkov et al., “Self-supervision advances morphological profiling by unlocking powerful image representations,” Scientific Reports, 2025.

[14] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[15] Y. Bao, S. Sivanandan, and T. Karaletsos, “Channel vision transformers: An image is worth 1 × 16 × 16 words,” in International Conference on Learning Representations, 2024.

[16] T. Sounack et al., “BioClinical ModernBERT: A state-of-the-art longcontext encoder for biomedical and clinical NLP,” arXiv preprint arXiv:2506.10896, 2025.

[17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[18] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” in Advances in Neural Information Processing Systems, 2018.

[19] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.