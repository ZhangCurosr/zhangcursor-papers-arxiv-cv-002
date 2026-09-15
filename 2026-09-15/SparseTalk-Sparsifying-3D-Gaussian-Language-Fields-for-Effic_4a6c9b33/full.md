# SparseTalk - Sparsifying 3D Gaussian Language Fields for Efficient 3D Visual Question Answering

Davit Soselia Joseph JaJa Amitabh Varshney

University of Maryland, College Park, MD, USA

dsoselia@umd.edu josephj@umd.edu varshney@umd.edu

## Abstract

3D Gaussian languagefields provide an explicit, spatially grounded representation for 3D visual question answering (VQA), but their dense semantic features can require tens of thousands of embeddings per scene, resulting in substantial storage, memory, and inference costs. We investigate how much of this representation is actually necessary for downstream reasoning. Starting from a full embedding representation, we systematically sparsify its semantic embeddings, including the previously underexplored regime below a single image-equivalent block down to 8 visual tokens. We compare random, geometric, semantic, and joint spatialsemantic selection strategies and introduce an object-based sparsification method that distributes the token budget across detected object instances while retaining background context. Experiments on ScanQA and MV-ScanQA reveal substantial redundancy in dense Gaussian languagefields. Strong VQA performance is retained with only a few hundred semantic embeddings, corresponding to less than 1% of the original representation. Object-based selection performs well relative to others, with only modest observed changes down to 256 tokens. At this budget, SparseTalk retains 0.80% ofSplatTalk’s 32,076-token inference input and 0.332% of the mean 77,207-Gaussian dense field, increasing inference throughput while reducing decoded-feature memory 125- fold.

## 1. Introduction

As vision-language models are increasingly used in many tasks, including robotics, physical AI, augmented reality, autonomous systems, and industrial environments, it has become increasingly important for these models to understand, identify objects in, and reason about the 3D world. A direct approach is to provide a sequence of images to a vision-language model (VLM), effectively treating different observations of a scene as frames of a video. While this approach benefits from strong pretrained image and video

models, it requires the model to infer persistent 3D structure from a collection of perspective-dependent 2D observations. Existing VLMs continue to struggle with cross-view integration and spatial relationships between objects, particularly when the answer requires a world-centric rather than cameracentric understanding of the scene [3, 9, 35].

![](images/610a8c8fec20a9940abf0f837bce373682b9e27e12acb73826174aa039b3c896.jpg)  
Figure 1. By using object-based sparsification, we reduce the number of semantic embeddings from tens of thousands to a few hundred, leading to faster inference and less memory usage, while retaining most of the VQA performance across the ScanQA and MV-ScanQA datasets.

An alternative direction is to encode semantic information directly within a 3D representation. Methods such as LangSplat, ChatSplat, and SplatTalk [4, 27, 29] associate learned language or vision-language features with the primitives of a 3D Gaussian Splatting representation [15]. These methods often use an encoder or autoencoder to compress high-dimensional visual features, then train the Gaussian representation so that rendered feature maps align with features extracted from the original views. These methods produce a spatially grounded semantic representation that can support open-vocabulary querying, conversation, or 3D visual question answering.

However, this representation also introduces a substantial computational cost. A reconstructed scene may contain tens of thousands of Gaussians, with a separate semantic embedding associated with each primitive. These embeddings increase storage, memory use, transfer bandwidth, and the number of visual tokens that must be processed by the language model. Moreover, even simple objects may be represented by hundreds of nearby Gaussians. It is unlikely that every one of these embeddings provides unique semantic information. Although prior work has compressed the dimensionality of Gaussian features and compared several token-selection strategies, the amount of redundancy in these representations, particularly below one image-worth of tokens, remains underexplored.

Our contributions are threefold. First, we present a controlled study of question-independent, post-hoc subset sparsification for frozen 3D Gaussian language fields, comparing random, geometric, semantic, and joint spatial-semantic selection across budgets from 729 down to 8 retained visual tokens on ScanQA and MV-ScanQA. Second, we introduce an object-based selector that associates multi-view instance masks with 3D Gaussians and constructs a token ranking across foreground-object tracks while preserving a contextual background pool. Third, we characterize the resulting quality-efficiency trade-off using standard metrics and Scaled Visually Attributable Performance (SVAP), a blindadjusted measure of sparse-model performance relative to full-context performance. Object-based selection achieves the strongest observed aggregate performance among the evaluated selectors, while uniform random sampling provides a surprisingly strong baseline. At k = 256, SparseTalk retains only 0.80% of SplatTalk’s 32,076-token inference input, reduces decoded-feature memory from 229.92 MB to 1.84 MB, and increases measured throughput from 0.58 to 14.3 questions per second, with only modest observed changes in answer quality.

## 2. Related Work

3D visual question answering models. ScanQA [1] introduced free-form question answering over reconstructed indoor scenes and established a point-cloud-based benchmark for spatial scene understanding. SQA3D [23] extended this setting by conditioning questions on an agent’s situated pose and orientation. Subsequent 3D large multimodal models connect geometric scene representations to language models. 3D-LLM [10] injects learned 3D features into a Large Language Model (LLM) for a range of grounded language tasks, while LEO [12] trains an embodied generalist agent over object-centric 3D observations. LLaVA-3D [37] instead augments a pretrained multimodal model with 3Daware tokens derived from multi-view inputs. These approaches demonstrate the value of explicit 3D structures, but typically introduce a dedicated 3D encoder, object proposal pipeline, or task-specific alignment stage. SplatTalk [29] takes a complementary route: it learns language features inside a feed-forward Gaussian-splatting representation and decodes Gaussian features directly into the visual-token space of LLaVA-OneVision [18]. Our study starts from SplatTalk’s published representation and asks how many of these Gaussian tokens are actually needed at inference time.

Language-embedded neural fields and Gaussian splatting. Three-dimensional Gaussian Splatting (3DGS) [15] represents a scene with anisotropic Gaussian primitives and supports high-quality real-time rendering. Feature 3DGS [36] attaches distilled semantic features to Gaussian primitives, allowing a 3D field to inherit representations from 2D foundation models. LangSplat [27] compresses CLIP features into a language-aware Gaussian field for openvocabulary querying, and OpenGaussian [31] develops pointlevel open-vocabulary understanding over 3D Gaussians. These methods primarily evaluate localization, segmentation, or open-vocabulary recognition. SplatTalk [29] differs by reconstructing free-form LMM visual features and using individual Gaussian features as inputs to a language model for 3D VQA. Its published inference procedure ranks decoded Gaussian features by entropy and retains up to the LMM context limit. We revisit this design under controlled token budgets and compare uncertainty-based ranking with featureblind, geometric, semantic, and joint spatial-semantic alternatives.

Visual-token reduction. Visual-token redundancy has motivated pruning and merging methods for large multimodal models. FastV [5] removes low-attention visual tokens inside the language model after early transformer layers; VisionZip [33] identifies dominant and contextual tokens before LLM inference and merges redundant visual content. More recent question-aware methods [21, 25] select a compact relevance anchor and then recover complementary context, rather than treating relevance and diversity as a single ranking objective. Token reduction has also begun to incorporate 3D structure.

Fast3D [14] predicts global attention to prune object-centric tokens in 3D multimodal models, while Geo3DPruner [19] removes redundant spatial-video tokens using cross-view geometry and voxel-level coverage. SeGPruner [20] combines attention-based semantic saliency with a geometry-aware diversity stage for multi-view 3D question answering. Lai et al. [17] instead prune multi-view tokens online by projecting observed patches into a shared voxel space and suppressing spatially repeated evidence.

Recent analysis of conventional multimodal language models has shown that projected image tokens contain substantial semantic redundancy. Fan et al. [7] partition visual tokens into sink, dead, and alive categories and find that approximately 40% of tokens carry little or no image-specific semantic information. While this result concerns 2D patch tokens, we investigate whether substantially stronger redundancy is present in spatially grounded 3D Gaussian language fields.

![](images/99a050c043b9326ac6f1045712c8ef6d33690c7fea470a0c7fb604d0d8b65d1a.jpg)  
Figure 2. Overview of our object-based Gaussian sparsification pipeline. From up to 100 uniformly spaced finite-pose RGB views, Florence-2 and SAM2.1 produce per-view object masks. Gaussian alpha-compositing contributions project these proposals into 3D, where weighted-Jaccard and Hungarian association form detected object tracks; ambiguous and structural Gaussians enter a shared background pool. Per-pool random orders are combined using equal foreground-track allocation and a 30% background weight, to produce one nested scene ranking. The first k Gaussian features are decoded by the autoencoder into k 3584-dimensional LLaVA-OneVision visual tokens used for VQA.

GaussianVLM [8] trains a prompt-conditioned module that re-tokenizes SceneSplat features into 128 aggregated scene tokens, rather than retaining 128 original Gaussians. We instead study question-independent sparsification to isolate redundancy in the existing representation.

Our setting is distinct in two ways. First, the candidates are unstructured Gaussian primitives whose features already lie in the LMM visual-token space, rather than image patches, video tokens, or object proposals. Second, our work deliberately keeps selection independent of the question, isolating the information retained by the 3D scene representation itself.

## 3. Methodology

## 3.1. Post-Hoc and Training-Time Sparsification

Post-hoc sparsification measures the redundancy of the learned representation and can be applied directly to existing checkpoints. Let a reconstructed scene contain a set of Gaussian primitives $\mathcal { G } = \{ g _ { i } \} _ { i = { \ i } } ^ { N }$ , with a learned semantic feature $\mathbf { z } _ { i } \in \mathbb { R } ^ { d }$ associated with each primitive. Given a token budget k, a selector produces an ordered subset

$$
S _ { k } = S ( \mathcal { G } , k ) , \qquad S _ { k } \subseteq \{ 1 , \dots , N \} , \qquad | S _ { k } | = k .\tag{1}
$$

We define the retention ratio as $\rho = k / N$ . Unless stated otherwise, the ranking is computed once per scene and independently of the question and its answers.

In the post-hoc setting, we begin with the complete, trained semantic Gaussian field and keep all model and autoencoder weights frozen. We rank the Gaussian embeddings using object-based, random, geometric, semantic, and joint spatial-semantic criteria and retain the first k entries. For a budget k, we decode the selected features into $\mathbf { F } _ { k } \in \mathbb { R } ^ { k \times 3 5 8 4 }$ and pass exactly these k embeddings to SplatTalk’s direct-feature interface in the selector-defined ranking order, without padding or token duplication.

Prior work evaluates semantic Gaussian sampling at budgets of one or more image-equivalent token blocks, with one block containing $2 7 \times 2 7 = 7 2 9$ tokens [29]. However, our initial experiments show that strong 3D VQA performance is retained even at substantially smaller budgets. We therefore remove the 729-token block restriction and study the sub-block regime from $k = 8 \mathrm { t o } k = 7 2 9$ . This evaluation measures both the performance curve and the point at which reducing the representation begins to remove necessary scene information.

Finally, we transfer the best-performing selector to the training pipeline. In this training-time setting, only the selected Gaussian primitives receive and optimize semantic features. The remaining Gaussians continue to represent scene appearance and geometry, but do not carry language embeddings. This provides additional benefits of reducing semantic-feature training cost by avoiding the construction of a dense language field.

## 3.2. Embedding selection

Each selector constructs a deterministic ordered ranking $\pi _ { s } = ( \pi _ { s , 1 } , \ldots , \pi _ { s , 7 2 9 } )$ for scene s. A condition with budget k consumes the exact prefix $\{ \pi _ { s , 1 } , \ldots , \pi _ { s , k } \}$

We evaluate the following ranking strategies.

Object-based. Global sampling methods may repeatedly select Gaussians belonging to the same large object or structural surface while failing to represent smaller objects. This is particularly limiting at low token budgets, where a small number of redundant selections may remove an object from the retained representation entirely. We therefore introduce object-stratified Gaussian sampling, which distributes the available token budget across detected object tracks before sampling within each track.

![](images/c7d202390ba51dfbee3af36f28733ea4aea9470ec740d39af664e5b423e61ce0.jpg)  
Figure 3. Performance on ScanQA when sparsifying from 729 down to 8 Gaussian embeddings, with the object-based selection offering the best results, maintaining performance down to 256 tokens. SplatTalk-32k denotes the original 32,076-token inference budget; Blind uses no visual tokens.

Following [29], to reduce computational overhead, we uniformly select up to 100 finite-pose RGB observations from each scene as nearby views are often redundant in image space. We first detect and segment objects. We apply Florence-2-large using its generic <OD> task prompt and use each detected bounding box to prompt SAM 2.1 for an instance mask [28, 32]. No category list is supplied. We discard invalid, low-confidence, and very small masks and suppress strongly overlapping proposals. Detector labels are canonicalized by text post-processing. They are lowercased and normalized for punctuation and whitespace, then resolved to class identifiers from the published Open Images vocabulary [16]. Detections labeled as wall, floor, or ceiling are treated as structural background rather than foreground objects. Overlapping mask pixels are assigned deterministically using segmentation confidence, mask area, canonical label, and proposal order.

For each detected proposal, we compute the contribution of every 3D Gaussian to its mask using the Gaussian rasterizer. Let $M _ { v r } ( p ) \in \{ 0 , 1 \}$ indicate whether pixel p in view v belongs to proposal r, and let $c _ { v g } ( p )$ denote the alpha-compositing contribution of Gaussian g to pixel p in view v. The proposal-to-Gaussian association mass is

$$
m _ { v g r } = \sum _ { p } M _ { v r } ( p ) c _ { v g } ( p ) .\tag{2}
$$

Thus, a Gaussian receives high association mass when it contributes strongly to pixels covered by a proposal mask.

Because the detector produces independent proposals in each image, proposals must be associated across views. We represent each proposal by the smallest set of Gaussian indices whose cumulative contribution accounts for 95% of its compositing mass, together with their normalized association weights. Additional implementation details are provided in the supplementary material. We compare proposals using the generalized weighted-Jaccard overlap of these sparse Gaussian supports. Proposals are matched between views using deterministic one-to-one Hungarian matching, with thresholds of 0.15 for proposals having the same canonical label and 0.40 otherwise. Unmatched proposals initialize new object tracks.

![](images/542144952e22708b13205afbce1cbc83ce765725560ac44901c5627ff9eaaf55.jpg)  
SplatTalk-32k Blind Object-based (ours)

Figure 4. Sparsification from 729 to 8 embeddings on the MV-ScanQA dataset. We see modest performance drops in the 729 to 256 region across the metrics.

After aggregating all proposals belonging to track $^ { O , }$ we define its association mass with Gaussian g as

$$
C _ { g o } = \sum _ { ( v , r ) \in o } m _ { v g r } .\tag{3}
$$

The normalized association is

$$
a _ { g o } = \frac { C _ { g o } } { \sum _ { v } \sum _ { p } c _ { v g } ( p ) + \epsilon } .\tag{4}
$$

Gaussian g is assigned to the object track with the largest $a _ { g o }$ when that score is at least $\tau _ { \mathrm { a s s o c } } = 0 . 5$ . Ambiguous and unobserved Gaussians are assigned to the background pool.

A direct allocation proportional to object size would cause large objects and room surfaces to dominate the selected representation.

Instead, we combine uniform object coverage with sublinear size-based allocation. Let O denote the set of foreground tracks and let $s _ { o }$ denote the visible contribution of track o. Its normalized allocation weight is

$$
p _ { o } = ( 1 - \lambda ) \frac { 1 } { | \mathcal O | } + \lambda \frac { s _ { o } ^ { \gamma } } { \sum _ { j \in \mathcal O } s _ { j } ^ { \gamma } } .\tag{5}
$$

We assign a scheduling weight $q _ { \mathrm { b g } }$ to the background pool. The pool weights are therefore

$$
w _ { o } = ( 1 - q _ { \mathrm { b g } } ) p _ { o } , \qquad w _ { \mathrm { b g } } = q _ { \mathrm { b g } } .\tag{6}
$$

We use $\lambda = 0 . 2 5 , \gamma = 0 . 2 5$ , and $q _ { \mathrm { b g } } = 0 . 3$ . Thus, 30% of the scheduling weight is assigned to background, while the remaining 70% is distributed among foreground tracks using a mixture of 75% uniform track coverage and 25% sublinear size-dependent allocation. Within each object, we consider both random selection and farthest-point selection; we find random selection to be the better of the two and use it in most of the experiments.

Uniform random. Uniform random sampling is the simplest sparsification baseline, yet we see that it provides surprisingly robust performance. For each scene and global random seed, we derive a deterministic 64-bit seed by hashing the scene identifier and seed configuration with SHA-256. This seed initializes a pseudorandom permutation of all Gaussian indices, from which we retain the first k Gaussians for a token budget k. Sampling is performed without replacement and is independent of Gaussian position, appearance, opacity, semantic features, and the question being answered.

Opacity top-k. As a simple 3DGS-parameter baseline, we rank Gaussians by decreasing opacity $a _ { i } .$ . Opacity is available in the encoded Gaussian payload, so this method is independent from language embeddings. It tests whether primitives with the strongest learned visual contribution are also the most useful for VQA.

Table 1. Resource scaling with the number of retained Gaussian tokens k on the ScanQA validation scenes. Inference-token retention is measured relative to the 32,076-token inference baseline, while dense-field retention is computed as $7 1 k / 5$ ,481,731. The dense-field row uses the mean scene size $\bar { N } _ { s } = 7 7 { , } 2 0 7$ . Storage values are decimal MB and describe the semantic Gaussian payload.
<table><tr><td>k</td><td>Inference-token retention</td><td>Dense-field retention</td><td>Throughput (q/s)</td><td>Persistent selected payload (MB)</td><td>Decoded feature tensor (MB)</td></tr><tr><td> $\bar { N } _ { s } = 7 7 { , } 2 0 7$ </td><td></td><td>100.000%</td><td></td><td>83.08</td><td>553.42</td></tr><tr><td>32,076</td><td>100.00%</td><td>41.545%</td><td>0.58</td><td>34.50</td><td>229.92</td></tr><tr><td>729</td><td>2.27%</td><td>0.944%</td><td>10.2</td><td>0.79</td><td>5.23</td></tr><tr><td>512</td><td>1.60%</td><td>0.663%</td><td>12.3</td><td>0.56</td><td>3.67</td></tr><tr><td>256</td><td>0.80%</td><td>0.332%</td><td>14.3</td><td>0.27</td><td>1.84</td></tr><tr><td>128</td><td>0.40%</td><td>0.166%</td><td>16.7</td><td>0.14</td><td>0.92</td></tr></table>

Farthest-point sampling. FPS operates only on min-maxnormalized Gaussian centers. The first Gaussian is the one farthest from the scene centroid. Each subsequent Gaussian maximizes its minimum Euclidean distance to the selected set. FPS therefore promotes geometric coverage but does not use the learned language features.

Semantic k-center. We $\ell _ { 2 } { \mathrm { - n o r m a l i z e } }$ the 256-dimensional Gaussian features and choose one representative per occupied voxel: the member closest in cosine distance to that voxel’s semantic centroid. Starting from the representative nearest the global semantic centroid, greedy k-center repeatedly selects the candidate with the largest minimum cosine distance to the current set.

Decoded-feature entropy. This mimics the selection score in [29]. For a decoded feature $\mathbf { y } _ { i } \in \mathbb { R } ^ { 3 5 8 4 }$ , we compute

$$
\mathbf { q } _ { i } = \mathrm { s o f t m a x } ( \mathbf { y } _ { i } ) , \qquad H _ { i } = - \sum _ { c = 1 } ^ { 3 5 8 4 } q _ { i c } \log ( q _ { i c } + 1 0 ^ { - 8 } ) ,\tag{7}
$$

and rank rows by decreasing $H _ { i }$ . Unlike the two k-center methods, entropy scores each Gaussian independently and does not enforce spatial or semantic coverage.

Joint spatial-semantic k-center. This selector uses the same voxel representatives and initialization as semantic k-center. For normalized centers $\widetilde { \mathbf { p } } _ { i } \in [ 0 , 1 ] ^ { 3 }$ , we define

$$
d _ { \mathrm { s p } } ( i , j ) = \frac { \lVert \widetilde { \bf p } _ { i } - \widetilde { \bf p } _ { j } \rVert _ { 2 } } { \sqrt { 3 } } .\tag{8}
$$

At step t, each unselected candidate receives the score

$$
\begin{array} { r l } & { r _ { t } ( i ) = \alpha \underset { j \in S _ { s , t - 1 } } { \operatorname* { m i n } } d _ { \mathrm { s p } } ( i , j ) } \\ & { \quad \quad \quad + \left( 1 - \alpha \right) \underset { j \in S _ { s , t - 1 } } { \operatorname* { m i n } } d _ { \mathrm { s e m } } ( i , j ) , } \end{array}\tag{9}
$$

and the candidate with maximum $r _ { t } ( i )$ is selected. We report the tested setting $\alpha = 0 . 2 5$ , which places greater weight on semantic diversity while retaining an explicit spatial-coverage term.

## 3.3. Evaluation

We report EM@1 [1], EM@1-Refined [29], METEOR [2], ROUGE-L [22], and BLEU-1 [26], together with inference throughput and storage requirements.

Since we find that some of the questions in benchmarks can be answered in the blind or text-only mode, we present performance on questions not answered correctly by the blind model using Scaled Visually Attributable Performance (SVAP), defined as

$$
\operatorname { S V A P } ( k ) = 1 0 0 { \frac { \sum _ { i } ( 1 - b _ { i } ) s _ { k , i } } { \sum _ { i } ( 1 - b _ { i } ) f _ { i } } } .\tag{10}
$$

where $b _ { i } , f _ { i } ,$ and $s _ { k , i }$ indicate whether question i is answered correctly by the text-only, full-context, and k-token models, respectively.

## 4. Results

## 4.1. Experimental setup

We conduct our primary experiments on ScanQA [1] and MV-ScanQA [24], using their respective ScanNet validation splits. The ScanQA evaluation set contains 4,675 questions across 71 scenes, while the MV-ScanQA validation set contains 2,230 questions across 66 scenes. MV-ScanQA places greater emphasis on questions that require evidence to be integrated across multiple viewpoints, complementing the predominantly single-view-solvable questions in ScanQA.

We use a model fine-tuned on the ScanQA training set for both datasets to further test generalizability. Because the model is not fine-tuned on MV-ScanQA, its performance on this benchmark measures how well the learned representation transfers to more explicitly multi-view reasoning tasks.

k=32: window  
BLIND BASELINE  
ALL FAIL  
k=8: white  
Table 2. Dataset statistics for ScanQA and MV-ScanQA. Scenes denotes the number of Gaussians with semantic embeddings per scene in the vanilla setting; Objects denotes the number of groundtruth foreground objects per scene. Gaussian counts are reported in thousands (K).
<table><tr><td>Quantity</td><td>N</td><td>Mean</td><td>Std. Dev.</td><td>Min</td><td>Max</td></tr><tr><td>ScanQA Scenes</td><td>71</td><td>77.2K</td><td>6.87K</td><td>60.1K</td><td>89.5K</td></tr><tr><td>MV-ScanQA Scenes</td><td>66</td><td>77.2K</td><td>7.08K</td><td>60.1K</td><td>89.5K</td></tr><tr><td>ScanQA Objects</td><td>71</td><td>27.17</td><td>17.49</td><td>3</td><td>101</td></tr><tr><td>MV-ScanQÀ Objects</td><td>66</td><td>28.55</td><td>17.37</td><td>3</td><td>101</td></tr></table>

We also evaluate on [13], an object-centric benchmark designed to identify visual ignorance and inconsistencies between grounding and question answering.

Some of the questions in the benchmarks can potentially be answered without any visual information. We find that the blind-baseline EM@1-R scores are 24.61 and 28.77 for ScanQA and MV-ScanQA, respectively, compared to 38.52 and 44.52 achieved under full embeddings.

## 4.2. Post-Hoc Sparsification

Fig. 3 and Fig. 4 compare post-hoc sparsification strategies on ScanQA and MV-ScanQA. Across both datasets, objectbased selection obtains strong results at every evaluated token budget and across all six answer-quality metrics. This supports the hypothesis that explicitly distributing the token budget across object instances preserves more useful scene information than global sampling or feature-based ranking.

METEOR, ROUGE-L, BLEU-1, and EM@1 scores follow the same trend, with minimal performance drop when decreasing from 729 to 256 embeddings. The trend is even more pronounced on MV-ScanQA, where EM@1-R stays within 0.1 points from 729 to 256 tokens, with a slight drop from 44.24 to 43.08 at 64 embeddings, and a falloff after 32. Uniform random sparsification also shows surprisingly strong performance, achieving higher scores than entropy top-k and joint k-center on MV-ScanQA in the 16 to 256 region on METEOR, ROUGE-L and BLEU-1. Entropy top-k shows the largest drop across the datasets.

We found early in our experimental testing that 3DGSparameter-based opacity top-k and FPS significantly underperform random selection, achieving only 30.29 - 33.42 and 31.12 - 33.42 EM@1-R respectively in the 8 to 729 region on a 1,188-question random ScanQA subset, so we dropped them from subsequent experiments.

Entropy-based ranking consistently underperforms the other selectors, despite requiring access to all densely decoded features. Together, these findings indicate that decoded-feature uncertainty is not a reliable proxy for question-independent scene coverage, while explicitly preventing small objects from disappearing from the retained representation provides a more robust selection criterion.

MORE TOKENS RECOVER THE ANSWER  
![](images/c33a221a67f9054db7ee2002aa6b58209319098269cb42607943adf465a625a2.jpg)  
(a) What color is the object located next to the tall brown wardrobe?

![](images/c76ba54540c3adec3ae3ef465b4d6bdf12489727abacfa89e21d47a794bbc9d6.jpg)  
(b) What object is located on the right side of the room in the middle, facing the chair at the head of the table?

![](images/28a5efbcd9ed8f5021241712ab66720baa22a5d7b150ca7291aa27898c920c16.jpg)  
(c) How many legs does the nightstand have that has a lamp with a white lampshade on top of it?

![](images/d32e4a5f09ff44ff84f0da3ac69dd6018afa212f7bb36963f99347bd9f4a4a80.jpg)  
(d) What color is the bookshelf located on the right side when sitting in the tan leather chair?  
Blind: brown.  
k=256: brown

![](images/1602ad7d13ab583d90cd69fe12f7a7d90f239a832bb79528777f9c374c5482c7.jpg)  
(e) What is the object located in front of the armchair that is also up against the left side of the dark grey breaker box?

![](images/709629e497419275cb62e960c06bcc1789042312d05eb2f14649e185bd1ef4d3.jpg)  
Figure 5. Samples where inclusion of more tokens leads to improvements in the answer quality (a-c), where even blind inference guesses the correct answer (d), and where all sparsification levels fail (e-f). Ground truth or answers evaluated as correct are denoted in green, and incorrect in red.

The modest score changes between 256 and 729 tokens accompany substantial downstream savings, as shown in Tab. 1. At 256 only 0.80% of the 32,076-token baseline representation is retained with stable EM@1 and EM@1- R scores. This provides a 24.7x speedup compared to full context length and 40% speedup over 729 tokens, while persistent encoded-feature storage decreases from 34.50 MB to 0.27 MB and decoded-feature memory decreases from 229.92 MB to 1.84 MB.

Reducing the budget to 128 tokens further increases throughput to 16.7 questions per second, but the quality curves begin to decline more noticeably.

## 4.3. Question answering capacity

We examine how correlated the successfully answered questions are between the sparsification methods. We also examine the extent to which questions answered correctly at lower k remain correctly answered at higher k. We calculate the mean pairwise Jaccard similarity among the sparsification approaches, with the mean at each level reported in Tab. 3.

Table 3. Mean pairwise Jaccard similarity across different values of k across sparsification approaches.
<table><tr><td>k</td><td>8</td><td>32</td><td>128</td><td>256</td><td>512</td><td>729</td></tr><tr><td>%</td><td>78.9</td><td>76.4</td><td>76.4</td><td>76.8</td><td>77.0</td><td>77.2</td></tr></table>

When comparing different levels of sparsification, we find that <sup>|correct</sup> <sup>at</sup> <sup>lower</sup> <sup>k</sup> <sup>∩</sup> <sup>correct</sup> <sup>at</sup> <sup>higher</sup> <sup>k|</sup> for nearby k values tested |correct at lower k| is greater than 96% across methods, but when comparing 32 to 729 or 16 to 729 the average drops to 82.4%. The object-based sparsification performs better at 83.75%, some examples can be seen in Fig. 5.

Table 4. training-time performance on BEACON3D using the dense semantic field and [5% retained / 95% removed]
<table><tr><td>k</td><td>EM@1</td><td>EM@1-R</td></tr><tr><td>Full</td><td>25.1</td><td>41.2</td></tr><tr><td>5%</td><td>25.2</td><td>40.5</td></tr></table>

## 4.4. Fine-tuning

We use object-based sparsification during the fine-tuning process by enforcing the selected k value. We only test Beacon3D at 5% sparsification Tab. 4 as a pilot experiment due to computational limitations.

## 4.5. Ablation

We compare random and farthest-point sampling within each object pool. FPS is better at the most extreme eight-token budget by 0.53 EM@1, but random sampling is better by 0.14–0.42 points from 32 through 729 tokens. The differences are consistently below one EM@1 point. We therefore use random within-object sampling in the main experiments because it is simpler and performs slightly better across the practically relevant 32 to 729 budget range. Further ablation details are provided in the supplementary material.

Table 5. EM@1 difference between random and within-pool FPS ordering in the object-based selector on ScanQA. Positive values favor random ordering.
<table><tr><td>k</td><td>Random − FPS</td></tr><tr><td>8</td><td>-0.53</td></tr><tr><td>32</td><td>0.14</td></tr><tr><td>128</td><td>0.38</td></tr><tr><td>256</td><td>0.38</td></tr><tr><td>512</td><td>0.39</td></tr><tr><td>729</td><td>0.42</td></tr></table>

## 5. Conclusion

We present SparseTalk, a systematic study of sparsifying semantic embeddings in 3D Gaussian language fields for 3D visual question answering. Our experiments show that dense Gaussian language representations contain substantial redundancy: strong VQA performance can be retained with only a few hundred semantic embeddings per scene. Among the evaluated selection strategies, object-based sparsification achieves the highest observed aggregate scores under constrained token budgets, suggesting that maintaining coverage across object instances is more important than globally ranking Gaussians. In particular, performance remains largely stable around 256 retained tokens, despite using only 0.80% of the 32,076-token inference baseline, while substantially reducing semantic-feature storage and decoded-feature memory and increasing inference throughput. We also find that uniform random sampling offers a surprisingly strong sparsification, outperforming other methods at most low embedding counts. These results indicate that explicit 3D semantic representations need not be dense to provide useful spatial grounding for downstream language reasoning.

Our study also exposes several limitations and directions for future work. First, the primary sparsification experiments are conducted on ScanQA and MV-ScanQA. A broader evaluation across datasets, scene types, and 3D language-field architectures is needed to establish the generality of the observed redundancy. Second, our object-based selector relies on 2D detection and segmentation followed by cross-view association to construct 3D object pools. Its effectiveness may consequently depend on the quality of these intermediate detections, particularly for small, occluded, or poorly segmented objects, and it introduces additional scene-level preprocessing. Finally, the presence of substantial blindbaseline performance makes it important for future evaluations to further study representation efficiency and dataset bias.

## References

[1] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. ScanQA: 3d question answering for spatial scene understanding. In CVPR, 2022. 2, 6

[2] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, 2005. 6

[3] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024. 1

[4] Hanlin Chen, Fangyin Wei, and Gim Hee Lee. Chat-Splat: 3D conversational gaussian splatting. arXiv preprint arXiv:2412.00734, 2024. 1

[5] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. arXiv preprint arXiv:2403.06764, 2024. 2

[6] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richlyannotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5828–5839, 2017. 1

[7] Yingqi Fan, Junlong Tong, Anhao Zhao, and Xiaoyu Shen. What do visual tokens really encode? uncovering sparsity and redundancy in multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11987–11997, 2026. 2

[8] Anna-Maria Halacheva, Jan-Nico Zaech, Xi Wang, Danda Pani Paudel, and Luc Van Gool. Gaussianvlm: Scenecentric 3d vision-language models using language-aligned gaussian splats for embodied reasoning and beyond, 2025. 3

[9] Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen, Joshua B. Tenenbaum, and Chuang Gan. 3D concept learning and reasoning from multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9202–9212, 2023. 1

[10] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3d world into large language models. In NeurIPS, 2023. 2

[11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 2

[12] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. In ICML, 2024. 2

[13] Jiangyong Huang, Baoxiong Jia, Yan Wang, Ziyu Zhu, Xiongkun Linghu, Qing Li, Song-Chun Zhu, and Siyuan

Huang. Unveiling the mist over 3d vision-language understanding: Object-centric evaluation with chain-of-analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24570–24581, 2025. 7

[14] Wencan Huang, Daizong Liu, and Wei Hu. Fast3D: Accelerating 3d multi-modal large language models for efficient 3d scene understanding. In ACM MM, 2025. 2

[15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG (SIGGRAPH), 42(4), 2023. 1, 2

[16] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal ofComputer Vision, 128: 1956–1981, 2020. 4, 3

[17] Ruei-Chi Lai, Bolivar Solarte, Chin-Hsuan Wu, Yi-Hsuan Tsai, and Min Sun. Seeing once is enough? online geometryaware token pruning for 3d question answering. In ICLR Workshop on Efficient Spatial Reasoning, 2026. 2

[18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 1

[19] Han Li, Zehao Huang, Jiahui Fu, Naiyan Wang, and Si Liu. Geometry-guided 3d visual token pruning for video-language models. In CVPR, 2026. 2

[20] Wenli Li, Kai Zhao, Haoran Jiang, Enquan Yang, Yi Su, and Dan Zeng. SeGPruner: Semantic-geometric visual token pruner for 3d question answering. arXiv preprint arXiv:2603.29437, 2026. 2

[21] Xu Li, Yi Zheng, Mengyang Zhao, Yuxuan Liang, Zhe Liu, Rui Zhu, Xiaolei Chen, Wei Zhou, Baoquan Zhao, and Juncen Guo. CRISP: Pre-LLM yet text-driven visual token pruning for efficient LVLM inference. In ICME, 2026. 2

[22] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, 2004. 6

[23] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. SQA3D: Situated question answering in 3d scenes. In ICLR, 2023. 2

[24] Wentao Mo, Qingchao Chen, Yuxin Peng, Siyuan Huang, and Yang Liu. Advancing 3d scene understanding with mv-scanqa: Multi-view reasoning evaluation and tripalign pre-training dataset. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12973–12980, 2025. 6

[25] Kyuan Oh and Bumsoo Kim. AnchorPrune: Relevanceanchored contextual expansion for visual token pruning. In ECCV, 2026. 2

[26] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: A method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting ofthe Association for Computational Linguistics, pages 311–318, 2002. 6

[27] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3d language gaussian splatting. In CVPR, 2024. 1, 2

[28] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting¨ Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer.´ SAM 2: Segment anything in images and videos. In International Conference on Learning Representations, 2025. 4, 3

[29] Anh Thai, Songyou Peng, Kyle Genova, Leonidas Guibas, and Thomas Funkhouser. SplatTalk: 3d VQA with gaussian splatting. In ICCV, 2025. 1, 2, 3, 4, 6

[30] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. FreeSplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. In NeurIPS, 2024. 1

[31] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Jian Zhang. OpenGaussian: Towards point-level 3d gaussian-based open vocabulary understanding. In NeurIPS, 2024. 2

[32] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024. 4, 3

[33] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. VisionZip: Longer is better but not necessary in vision language models. In CVPR, 2025. 2

[34] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pages 11975–11986, 2023. 1

[35] Hantao Zhang, Jinru Sui, Ed Li, Dirk Bergemann, and Zhuoran Yang. MultiView-Bench: A diagnostic benchmark for world-centric multi-view integration in VLMs. arXiv preprint arXiv:2607.08970, 2026. 1

[36] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3DGS: Supercharging 3d gaussian splatting to enable distilled feature fields. In CVPR, 2024. 2

[37] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. LLaVA-3D: A simple yet effective pathway to empowering LMMs with 3d capabilities. In ICCV, 2025. 2

# SparseTalk - Sparsifying 3D Gaussian Language Fields for Efficient 3D Visual Question Answering

Supplementary Material

## 6. Repeatability of Sparsification and Decoding

We examine how the results are affected by the randomness in the sparsification and decoding processes.

For stochastic sparsification methods such as uniform random sampling, variation arises from the scene-level selection seed and therefore from which Gaussians are retained. Methods such as joint and semantic k-center produce deterministic rankings; however, their results may still vary across repeated executions of the end-to-end pipeline. As a narrower diagnostic, we hold the exact decoded visual embeddings and their ordering fixed and vary only the model random seed, measuring any resulting variation in answers and metrics under greedy decoding.

## 6.1. Model-Seed Sensitivity

Changing the seed may also affect the language model itself, making it unclear whether observed score variation originates from Gaussian selection or from answer generation. Although our inference protocol uses fixed weights and greedy decoding, nondeterministic GPU operations or seeddependent model execution could still, in principle, alter predictions when token probabilities are close. We therefore isolate model-seed sensitivity while holding the complete visual input fixed.

We select 100 ScanQA validation questions from scene0231 00 and use the same ordered prefix of 729 visual tokens produced by object-based sparsification. Only the model seed is changed over {0, 1, 2, 3, 4}.

Across 500 generations and all ten pairwise seed comparisons, the prediction strings agree exactly for every question. Thus, under our fixed-weight greedy-decoding protocol, nominal model seeds introduce no measurable answer or metric variance when the visual tokens are unchanged.

## 6.2. End-to-End Variance

We measure repeatability on ScanQA questions. For each selector and token budget, we execute the complete sparsification, decoding, and answer-generation pipeline three times and report the mean and sample standard deviation of EM@1-R Tab. 6 and SVAP Tab. 7. Stochastic selectors use independent scene-level selection seeds, whereas deterministic selectors reuse the same ranking. Their variance therefore measures execution-level repeatability. Together with the fixed-input model-seed experiment above, these measurements distinguish sensitivity to Gaussian selection from variability in answer generation.

We see that standard deviation is highest for uniform random sampling, which is expected as it is the most stochastic method. The variance is notable for object-based sparsification when using random assignment within the objects, at 0.08 for k = 729 and 0.35 for k = 8, however the trends between the methods remain stable even when accounting for this.

## 7. Models Used

We conduct most experiments on independently reproduced trainable components of SplatTalk [29]: the visual-token autoencoder, the language-augmented Gaussian reconstruction model, and the ScanQA+SQA3D LoRA adapter. The pretrained SigLIP vision encoder [34] and LLaVA-OneVision Qwen2-7B foundation model [18] were initialized from their public checkpoints and kept fixed except for the newly initialized LoRA parameters. After the release of the public models we compared the performance and found that the reproduced models achieved similar performance to the original models.

For each posed RGB observation, we extract LLaVA-OneVision visual tokens after the SigLIP encoder and multimodal projector. Each resulting $2 7 \times 2 7 \times 3 5 8 4$ feature map is flattened into 729 token embeddings. We train a scene-general autoencoder with encoder dimensions 3584→2048→1024→512→256 and decoder dimensions $2 5 6 \to 5 1 2 \to 1 0 2 4 \to 2 0 4 8 \to 2 0 4 8 \to 3 5 8 4$ . The encoder uses batch normalization and GeLU activations, and its 256- dimensional output is normalized to the unit hypersphere. Training minimized the sum of feature-space mean-squared error and cosine distance. We used a scene-disjoint training/validation split, a batch size of 256, AdamW with learning rate $1 0 ^ { - 4 }$ , and 100 epochs, retaining the checkpoint with the lowest validation reconstruction loss.

We train the SplatTalk reconstruction model on 500 Scan-Net scenes [6] from the ScanQA training split. Following the FreeSplat architecture [30], the Gaussian decoder predicts the standard position, covariance, opacity, and appearance parameters together with a 256-dimensional language feature for every Gaussian. We use a parallel differentiable rasterizer. We use Adam with an initial learning rate of $1 0 ^ { - 4 }$ , a short linear warm-up, cosine decay, batch size one scene, and gradient clipping.

The trained autoencoder decoder maps each Gaussian’s 256-dimensional feature back into the 3584-dimensional LLaVA token space. These decoded embeddings are packed as multimodal input tokens and paired with the combined

Table 6. ScanQA EM@1-Refined by sparsification method and token budget k. Entries are mean ± standard deviation; the best mean in each column is bold.
<table><tr><td>Sparsification method</td><td> $k = 7 2 9$ </td><td> $k = 5 1 2$ </td><td> $k = 2 5 6$ </td><td> $k = 1 2 8$ </td><td> $k = 3 2$ </td><td> $k = 8$ </td></tr><tr><td>Joint k-center (α = 0.25)</td><td> $3 6 . 3 3 \pm 0 . 0 0$ </td><td> $3 6 . 1 9 \pm 0 . 0 0$ </td><td> $3 6 . 0 9 \pm 0 . 0 0$ </td><td> $3 5 . 1 8 \pm 0 . 0 0$ </td><td> $3 4 . 1 0 \pm 0 . 0 0$ </td><td> $3 3 . 1 4 \pm 0 . 0 0$ </td></tr><tr><td>Object Based</td><td> ${ \bf 3 8 . 4 5 \pm 0 . 0 8 }$ </td><td> ${ \bf 3 7 . 9 3 \pm 0 . 5 5 }$ </td><td> ${ \bf 3 7 . 2 0 \pm 0 . 5 3 }$ </td><td> ${ \bf 3 6 . 5 1 \pm 0 . 1 3 }$ </td><td> ${ \bf 3 5 . 4 1 \pm 0 . 1 8 }$ </td><td> ${ \bf 3 3 . 8 8 \pm 0 . 3 5 }$ </td></tr><tr><td>Entropy top-k</td><td> $3 2 . 5 4 \pm 0 . 0 0$ </td><td> $3 2 . 4 4 \pm 0 . 0 0$ </td><td> $3 1 . 7 3 \pm 0 . 0 0$ </td><td> $3 1 . 6 3 \pm 0 . 0 0$ </td><td> $3 0 . 5 3 \pm 0 . 0 0$ </td><td> $2 9 . 7 1 \pm 0 . 0 0$ </td></tr><tr><td>Semantic k-center</td><td> $3 6 . 4 1 \pm 0 . 0 0$ </td><td> $3 6 . 2 5 \pm 0 . 1 0$ </td><td> $3 6 . 1 1 \pm 0 . 0 0$ </td><td> $3 5 . 3 4 \pm 0 . 0 0$ </td><td> $3 3 . 8 8 \pm 0 . 0 0$ </td><td> $3 2 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>Uniform random</td><td> $3 6 . 2 2 \pm 0 . 8 5$ </td><td> $3 6 . 0 3 \pm 0 . 9 8$ </td><td> $3 5 . 7 0 \pm 0 . 3 4$ </td><td> $3 5 . 2 0 \pm 0 . 1 8$ </td><td> $3 3 . 3 7 \pm 0 . 4 8$ </td><td> $3 2 . 0 1 \pm 0 . 5 1$ </td></tr></table>

![](images/1041b06ebdd07264aebe1ab052ac60a75371e5aedb2bf8781bd7c3cfc954a057.jpg)  
Figure 6. Visualization of the embedding locations at four k sparsification budgetsoverlayed on the 3d scene. Blue dots indicate the object allocations and the gray points indicate the backgorund/structural reserve.

ScanQA and SQA3D training annotations. We initialize a new rank-16 LoRA adapter [11] with scaling factor 64 and dropout 0.05. LoRA updates are applied to the language model’s query, key, value, and output projections and to its gate, up, and down MLP projections. The base language model and visual feature pipeline remain frozen.

We fine-tune for one epoch in bfloat16 precision on an NVIDIA H200 using AdamW, a peak learning rate of $1 0 ^ { - 5 }$

Table 7. ScanQA SVAP by sparsification method and embedding budget k. Entries are mean ± standard deviation; the best mean in each column is bold.
<table><tr><td>Sparsification method</td><td> $k = 7 2 9$ </td><td> $k = 5 1 2$ </td><td> $k = 2 5 6$ </td><td> $k = 1 2 8$ </td><td> $k = 3 2$ </td><td> $k = 8$ </td></tr><tr><td>Joint k-center  $( \alpha = 0 . 2 5 )$ </td><td> $9 7 . 9 6 \pm 0 . 0 0$ </td><td> $9 6 . 5 4 \pm 0 . 0 0$ </td><td> $9 4 . 3 4 \pm 0 . 0 0$ </td><td> $8 6 . 1 6 \pm 0 . 0 0$ </td><td> $7 4 . 5 3 \pm 0 . 0 0$ </td><td> $5 5 . 0 3 \pm 0 . 0 0$ </td></tr><tr><td>Object-Based</td><td> ${ \pm \ : 9 9 . 8 7 \pm 1 . 1 4 }$ </td><td> ${ \pm 9 9 . 4 3 \pm 2 . 8 8 }$ </td><td> ${ \pm } \thinspace 2 . 8 8 \pm 2 . 0 1$ </td><td> $9 7 . 7 5 \pm 2 . 6 2$ </td><td> ${ \bf 8 8 . 2 9 \pm 1 . 7 5 }$ </td><td> ${ \bf 7 3 . 9 0 \pm 4 . 6 2 }$ </td></tr><tr><td>Entropy top-k</td><td> $5 2 . 5 2 \pm 0 . 0 0$ </td><td> $5 1 . 5 7 \pm 0 . 0 0$ </td><td> $4 2 . 4 5 \pm 0 . 0 0$ </td><td> $3 7 . 8 9 \pm 0 . 0 0$ </td><td> $2 6 . 5 7 \pm 0 . 0 0$ </td><td> $1 6 . 8 2 \pm 0 . 0 0$ </td></tr><tr><td>Semantic k-center</td><td> $9 8 . 4 2 \pm 0 . 0 0$ </td><td> $9 6 . 8 4 \pm 0 . 3 3$ </td><td> $9 5 . 7 5 \pm 0 . 0 0$ </td><td> $8 9 . 4 7 \pm 0 . 0 0$ </td><td> $7 2 . 9 6 \pm 0 . 0 0$ </td><td> $5 4 . 2 5 \pm 0 . 0 0$ </td></tr><tr><td>Uniform random</td><td> $9 8 . 5 1 \pm 3 . 0 2$ </td><td> $9 6 . 1 3 \pm 5 . 0 8$ </td><td> $9 3 . 1 8 \pm 2 . 0 6$ </td><td> $8 7 . 1 1 \pm 0 . 6 6$ </td><td> $6 7 . 5 8 \pm 2 . 4 9$ </td><td> $4 7 . 0 8 \pm 5 . 9 7$ </td></tr></table>

3% linear warm-up, cosine decay, zero weight decay, gradient accumulation, and gradient checkpointing. Sequences use the Qwen conversation format and a maximum context length of 32,768 tokens. At evaluation time, we use the same Gaussian preprocessing, 44-block visual-token budget, prompt construction, and greedy decoding protocol for the reproduced and reference models.

## 8. Object-Based Sparsification Details

Here, we provide details of the detection, association, and ranking procedures for object-based sparsification.

We numerically sort the posed RGB observations, discard frames with non-finite camera poses, and uniformly select up to 100 views across the remaining sequence. If fewer than 100 valid observations are available, we use all of them. We apply Florence-2-large with its generic <OD> task prompt, deterministic three-beam decoding, and no category list or question input. Each detected bounding box is then supplied to SAM 2.1 Hiera Large as an instance-segmentation prompt [28, 32].

We reject invalid boxes, boxes covering less than 0.05% or more than 95% of the image, masks with a SAM confidence below 0.80, and masks containing fewer than 64 pixels at the 180 × 320 association resolution. Mask nonmaximum suppression is applied at IoU 0.80 for proposals with the same canonical label and 0.95 otherwise. We retain at most 128 proposals per view.

The box-area limits exclude both extremely small detections and nearly full-image regions, while the SAM confidence and mask-area thresholds suppress uncertain or poorly supported masks. The same-label NMS threshold removes duplicate detections, whereas the higher cross-label threshold suppresses proposals with different labels only when their masks are nearly identical.

Detector outputs are normalized using a fixed, sceneindependent mapping to canonical object names derived from the Open Images vocabulary [16]. Normalization consists of lowercasing, punctuation and whitespace normalization, and fixed singular and synonym mappings. Labels without a mapped synonym retain their normalized detector name. Canonical labels are used only for cross-view association and structural-background identification. Detections corresponding to walls, floors, or ceilings are treated as structural background.

Before Gaussian association, the retained masks are converted into a disjoint proposal map. Pixels covered by multiple proposals are assigned deterministically according to segmentation confidence, mask area, canonical label, and original proposal order. Pixels not assigned to a retained foreground proposal belong to the contextual/background region.

Let $\mathcal { P } _ { v }$ denote the pixel domain of view $v ,$ let $M _ { v r } ( p ) \in$ $\{ 0 , 1 \}$ indicate whether pixel p belongs to proposal r, and let $c _ { v g } ( p )$ denote the alpha-compositing contribution of Gaussian g to pixel $p .$ The contribution mass associating Gaussian $g$ with proposal r in view v is

$$
m _ { v g r } = \sum _ { p \in \mathcal { P } _ { v } } M _ { v r } ( p ) c _ { v g } ( p ) .\tag{11}
$$

We obtain these masses directly from the differentiable Gaussian rasterizer. For each view, one auxiliary feature channel is assigned to each disjoint proposal region, and all per-Gaussian auxiliary features are initialized to zero. Let $R _ { v j } ( p ; F )$ denote rendered auxiliary channel $j$ under per-Gaussian features $F .$ Since the scene geometry, opacity, and compositing weights are fixed, the rendered output is linear in $F .$ The gradient of the mask-weighted rendered channel with respect to Gaussian $\boldsymbol { g ^ { \prime } } \mathbf { s }$ corresponding auxiliary feature therefore equals $m _ { v g r } .$ This gradient-based computation avoids materializing a dense Gaussian-by-pixel contribution tensor. We require all contribution masses to be finite and nonnegative. Because the disjoint proposal and background channels partition the rendered image, their summed contribution also recovers the corresponding all-pixel compositing mass.

Storing a dense Gaussian contribution vector for every proposal would be unnecessarily expensive. We therefore represent proposal $( v , r )$ by the smallest stable set of Gaussian indices $\boldsymbol { S _ { v r } }$ whose cumulative contribution accounts for at least 95% of its total mass, subject to a maximum of 2,048 Gaussians. If reaching 95% requires more than 2,048 Gaussians, we retain the 2,048 highest-contributing Gaussians. Each support’s weights are normalized by the proposal’s total contribution, quantized to $1 0 ^ { - 6 }$ , and stored

in Gaussian-index order.

Let $u _ { g } ^ { A }$ and $u _ { g } ^ { B }$ denote the sparse normalized weights of two proposals or tracks, with missing entries assigned zero weight. Their generalized weighted-Jaccard similarity is

$$
J ( A , B ) = \frac { \sum _ { g } \operatorname* { m i n } ( u _ { g } ^ { A } , u _ { g } ^ { B } ) } { \sum _ { g } \operatorname* { m a x } ( u _ { g } ^ { A } , u _ { g } ^ { B } ) } .\tag{12}
$$

The selected frames are processed in their original order. Proposals from the current frame are matched one-to-one with existing tracks by maximizing weighted-Jaccard similarity using Hungarian matching. A match is permitted when $J \ge 0 . 1 5$ for equal canonical labels or $J \geq 0 . 4 0$ when the labels differ. The stricter cross-label threshold accommodates variation in detector vocabulary while requiring stronger geometric evidence when canonical labels disagree. Unmatched proposals initialize new tracks.

After adding a proposal, we update the track’s accumulated Gaussian contributions and recompute its sparse support. The track label is the canonical label receiving the largest accumulated visible contribution. Similarities are quantized before assignment, and stable proposal and track identifiers resolve exact ties. To bound computation, we retain at most 512 foreground tracks per scene, prioritized by accumulated visible contribution. Association mass outside this set is incorporated into the shared contextual/background pool.

In subsequent aggregation, omitted support entries are treated as zero, and retained normalized weights are rescaled by the proposal’s total contribution mass. For notational simplicity, we continue to denote these reconstructed sparse masses by $m _ { v g r } .$ . After aggregating all proposals assigned to track $^ { O , }$ we define its association mass with Gaussian g as $\begin{array} { r } { C _ { g o } \ = \ \sum _ { \substack { ( v , r ) \in o } } m _ { v g r } } \end{array}$ . We define the total visible contribution of Gaussian g across the selected views as $V _ { g } =$ $\begin{array} { r } { \sum _ { v } \sum _ { p \in \mathcal { P } _ { v } } c _ { v g } ( p ) } \end{array}$ . The normalized track association is

$$
a _ { g o } = \frac { C _ { g o } } { V _ { g } + \epsilon } .\tag{13}
$$

Gaussian g is assigned to arg max $a _ { g o }$ when the winning association confidence is at least $\tau _ { \mathrm { a s s o c } } = 0 . 5$ . All remaining Gaussians, including structural and weakly associated regions, are collected in the shared contextual/background pool. Retaining this pool preserves scene layout and visual evidence not represented by a confident foreground track.

Let O denote the set of confidently assigned foreground tracks, let $\pi ( g )$ denote the pool assignment of Gaussian $^ { g , }$ and define the visible size of track o as $\begin{array} { r } { s _ { o } = \sum _ { g : \pi ( g ) = o } V _ { g } , } \end{array}$ We combine uniform track coverage with sublinear visiblesize weighting:

$$
p _ { o } = ( 1 - \lambda ) \frac { 1 } { | \mathcal O | } + \lambda \frac { s _ { o } ^ { \gamma } } { \sum _ { j \in \mathcal O } s _ { j } ^ { \gamma } } .\tag{14}
$$

We reserve a fraction $q _ { \mathrm { b g } }$ of the scheduling weight for the contextual/background pool. Thus, each foreground track receives weight $w _ { o } = ( 1 - q _ { \mathrm { b g } } ) p _ { o }$ , while the background pool receives $w _ { \mathrm { b g } } = q _ { \mathrm { b g } }$ . For the configuration reported in the paper, we use $\lambda = 0 . 2 5 , \gamma = 0 . 2 5$ , and $q _ { \mathrm { b g } } = 0 . 3$ . Consequently, the foreground allocation combines 75% uniform track coverage with 25% fourth-root visible-size weighting, while the contextual/background pool receives 30% of the scheduling weight.

For the primary random variant, we construct a deterministic random permutation independently within every foreground and background pool. The random seed is derived from the scene identifier, experiment seed, and pool identifier, but not from the requested token budget. We combine these pool-specific orders using weighted deficit scheduling. At each step, every nonempty pool accumulates credit in proportion to its scheduling weight normalized over the currently active pools. We emit the next Gaussian from the pool with the largest credit and then subtract one from that pool’s credit. When a pool is exhausted, its allocation is redistributed among the remaining pools. Stable scene-dependent hashes resolve exact scheduling ties.

For the farthest-point variant, the within-pool sequence begins with the Gaussian having the greatest visible contribution and then repeatedly selects the Gaussian with the largest minimum Euclidean distance from the previously selected points. The same weighted interleaving procedure is used for both within-pool selectors.

The random variant produces one immutable 729- Gaussian ranking per scene and seed, while the farthest-point variant produces one deterministic ranking per scene. A token budget k consumes exactly the first k entries, making all evaluated budgets nested prefixes of the same ranking. Only these k encoded Gaussian features are decoded into 3,584-dimensional LLaVA features. They are provided to the multimodal model with shape $1 \times 3 5 8 4 \times 1 \times k ,$ , which the existing LLaVA input preparation flattens into exactly k visual tokens.

## 8.1. Allocation Parameter Ablation

We ablate the three parameters controlling object-based token allocation: the background scheduling weight $q _ { \mathrm { b g } } .$ , the foreground size-mixture coefficient $\lambda ,$ and the size exponent $\gamma .$ . Foreground track o receives normalized weight

$$
p _ { o } = ( 1 - \lambda ) \frac { 1 } { | \mathcal { O } | } + \lambda \frac { s _ { o } ^ { \gamma } } { \sum _ { j \in \mathcal { O } } s _ { j } ^ { \gamma } } ,\tag{15}
$$

with final scheduling weights

$$
w _ { o } = ( 1 - q _ { \mathrm { b g } } ) p _ { o } , \qquad w _ { \mathrm { b g } } = q _ { \mathrm { b g } } .\tag{16}
$$

Our default configuration is $q _ { \mathrm { b g } } ~ = ~ 0 . 3 , ~ \lambda ~ = ~ 0 . 2 5$ , and $\gamma = 0 . 2 5$

Table 8. Allocation parameter ablation on the fixed 1,188-question ScanQA development subset at embedding budget $k = 2 5 6$ . Each block varies one parameter while holding the other two at their default values. Bold indicates the default setting.
<table><tr><td>Parameter</td><td>Value</td><td>Avg. EM@1-R</td></tr><tr><td> $q _ { \mathrm { b g } }$ </td><td>0.0</td><td>35.10</td></tr><tr><td> $q _ { \mathrm { b g } }$ </td><td>0.1</td><td>37.10</td></tr><tr><td> $q _ { \mathrm { b g } }$ </td><td>0.3</td><td>37.21</td></tr><tr><td> $q _ { \mathrm { b g } }$ </td><td>0.5</td><td>36.80</td></tr><tr><td> $\lambda$ </td><td>0.0</td><td>37.02</td></tr><tr><td> $\lambda$ </td><td>0.25</td><td>37.21</td></tr><tr><td> $\lambda$ </td><td>0.5</td><td>37.09</td></tr><tr><td> $\gamma$ </td><td>0.25</td><td>37.21</td></tr><tr><td> $\gamma$ </td><td>0.5</td><td>37.12</td></tr></table>

We evaluate parameter sensitivity on a fixed development subset of 1,188 ScanQA questions and report EM@1-R on embedding budget of 256.

The clearest effect comes from the background allocation. Removing it entirely, $q _ { \mathrm { b g } } = 0 ;$ , reduces average EM@1-R by approximately 2.11 points relative to $q _ { \mathrm { b g } } = 0 . 3 ,$ , indicating that contextual and background Gaussians provide useful scene information beyond detected foreground objects. However, performance is relatively insensitive to the exact nonzero value: $q _ { \mathrm { b g } } = 0 . 1$ and 0.3 perform similarly, while $q _ { \mathrm { b g } } = 0 . 5$ gives only a modest decrease.

The foreground allocation parameters have substantially smaller effects. A positive λ provides a slight improvement over purely uniform object allocation, but performance between $\lambda \in \{ 0 . 2 5 , 0 . 5 \}$ is similar. Likewise, changing γ from 0.25 to 0.5 has negligible impact. Overall, the ablation suggests that preserving a nonzero background allocation is important, whereas the precise values of λ and $\gamma$ are not critical within the tested ranges.