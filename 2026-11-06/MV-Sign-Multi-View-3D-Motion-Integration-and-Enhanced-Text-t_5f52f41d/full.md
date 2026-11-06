# MV-Sign: Multi-View 3D Motion Integration and Enhanced Text-to-Motion Conditioning for Sign Language Production

Singyu Muk<sup>1</sup> Kaustubh Kundu<sup>2</sup> Kunyuan Xie<sup>2</sup> Weijie Wang<sup>3,∗</sup> Kalin Stefanov<sup>4,∗</sup> <sup>1</sup>Independent Researcher, Australia <sup>2</sup>Monash University, Australia <sup>3</sup>University of Trento, Italy <sup>4</sup>Japan Advanced Institute of Science and Technology, Japan ronggong000@gmail.com Kaustubh.Kundu@monash.edu kunyuan.xie@monash.edu weijie.wang@unitn.it kalins@jaist.ac.jp

![](images/51c464cacb6fcac1b702dadc2ed474d7e13dcd117779aa6ed054d5dd628ee41d.jpg)  
Figure 1. MV-Sign overview. On the left, our multi-view data curation pipeline reconstructs high-quality 3D signing motion from noisy videos by combining per-view hand fusion, spatial-temporal alignment, and cross-view refinement. On the right, our generative mode conditioned on pseudo-gloss and keyframe-aware retrieval prompts produces accurate continuous signing motion from text input

## Abstract

Sign language production (SLP) aims to generate sign language motion from text. However, current 3D SLP methods suffer from noisy supervision and weak text conditioning. Relying on pseudo-3D motion lifted from single-view frontal videos introduces severe self-occlusion and loses fine-grained hand details, degrading downstream production. To address this, we propose a framework that integrates multi-view 3D motion data and optimizes text conditioning. First, we introduce How2Sign-MV3D, a highfidelity dataset curated via cross-view fusion to resolve occlusions and filter non-signing segments. Second, we design a robust SLP model incorporating a sign-aware spatiotemporal VQ-VAE (ST-VQVAE) for joint body-hand tokenization, a pseudo-gloss adapter, and keyframe-aware retrieval to enhance text-to-motion alignment. Evaluated on the How2Sign benchmark, ourframework reduces body and hand DTW-PA-JPE by 32.8% and 20.9% respectively compared to the state-of-the-art. Finally, we release two more new high-quality 3D datasets, ASLCitizen-3D and MM-WLAuslan-MV3D, alongside SGNify-MV as a multi-view video benchmark to support broader community research.

## 1. Introduction

Sign languages are the main way of communication in deaf communities [18, 71]. Sign language production (SLP), also called text-to-sign generation, aims to produce a signing motion from spoken language text. Unlike sign language translation and understanding, which have recently advanced through large-scale pretraining and multimodal language models [23, 36, 58, 67], SLP must produce motion that is both semantically accurate and physically realistic. These requirements make the availability of highquality 3D signing motion data a central bottleneck.

Recent SLP methods supervise sign generation by fitting SMPL-X [41] to frontal-view videos [5, 12, 73, 79] using datasets such as How2Sign [12]. Although How2Sign contains side-view videos, they suffer from temporal misalignment and heavy occlusions. Because of this, the side view is ignored by most work. However, How2Sign front-view videos also have significant issues, including motion blur, low frame rates, editing errors, and self-occlusion. These issues lead to noisy and incorrect pseudo-3D motion data and are propagated downstream into production.

To address these issues, we propose MV-Sign, a unified framework coupling asymmetric cross-view data curation with a denoised 3D SLP model. Project resources and dataset release information are available at https: //github.com/ronggong000/MV-Sign. Our main contributions are:

• We propose an asymmetric cross-view motion curation pipeline that uses side-view evidence to repair occlusions, blur, and tracking failures in frontal videos.

• We design a 3D SLP model integrating a sign-aware spatio-temporal VQ-VAE (ST-VQVAE) for joint body– hand tokenization, a pseudo-gloss adapter, and keyframeaware retrieval.

• We construct and release three 3D sign language datasets (How2Sign-MV3D, ASLCitizen-3D, and MM-WLAuslan-MV3D) alongside SGNify-MV, a controlled multi-view diagnostic benchmark.

## 2. Related Work

## 2.1. 3D Signing Data and Motion Curation

Progress in sign language research has long been constrained by data availability [7]. Early benchmarks focus on isolated sign recognition or dictionary-style resources [3, 6, 13, 14, 29, 33, 64, 65, 74], while larger corpora such as BSL-1K, BBC-Oxford BSL, the Public DGS Corpus, PHOENIX-2014T, and How2Sign enable sentencelevel recognition and generation research [1, 2, 9, 12, 19, 25, 51]. Although recent multi-view datasets such as MM-WLAuslan [53] provide synchronized videos from multiple views, most sign language resources remain 2D. Therefore, 3D SLP commonly relies on pseudo-3D motion by fitting parametric models such as SMPL-X [41], MANO [46], and FLAME [34] to videos. Consequently, SLP pipelines such as NSA [5], SignAvatars [73], SOKE [79], and avatar reconstruction methods [32] recover sign motion from videos using general whole-body recovery methods [8, 16, 31, 35, 37, 38, 47, 56, 75]. In parallel, dedicated hand reconstruction methods, including ACR [72], HaMeR [42], and WiLoR [43], address fine-grained 3D hand estimation. However, signing requires a finer wrist, handshape, and finger fidelity than generic pose recovery. Unlike multi-view human reconstruction in controlled settings, sign language corpora often have uncalibrated, occluded, or temporally offset side views. We therefore use the side view(s) as asymmetric evidence to verify and repair front-view failures, framing our contribution as signing-specific 3D motion curation.

## 2.2. Sign Language Production

Early SLP methods use recurrent or sequence-to-sequence pose generation [21, 62], followed by Transformer-based methods such as Progressive Transformer [49, 63]. Recent research has expanded SLP towards pose-guided video generation, diffusion-based 3D signing, and tokenized motion generation [5, 55, 73]. Diffusion-based approaches generate sign pose sequences directly from text [5, 15, 20, 45], sometimes with anatomically informed backbones. The tokenizer-language-model approaches first discretize the signing motion with VQ-VAE-style tokenizers and then predict the motion codes autoregressively [11, 70, 73, 79]. SOKE [79] is especially relevant because it uses separate token streams for the body, left, and right hand for efficient decoding. Building on SOKE [79] and SALAD [22], we retain the three-stream decoder interface but defer part factorization to the tokenizer bottleneck to preserve body–hand coupling.

## 2.3. Gloss, Retrieval, and Evaluation

Gloss-like sequences can provide motion-aligned intermediate representations for SLP, but gold gloss annotations are expensive and often unavailable [40]. Therefore, recent methods explore gloss-free modeling, visual-language pretraining, multilingual translation, LLM-assisted translation, and pseudo-glosses derived from text [18, 67–69, 76, 77]. Following Zhang et al. [76], we use vocabulary-constrained pseudo-glosses as a conditioning interface and apply classifier attribution to filter non-lexical preparation and retraction frames from retrieved sign prompts [50, 78, 79]. However, sign clips include preparation, retraction, and idle frames, which can consume the limited prompt budget of an autoregressive motion decoder. We instead use attribution from an isolated sign recognizer to retain informative tokens. For evaluation, we follow prior motion-space protocols rather than back-translation, since text metrics derived from separate sign language translation systems can be unreliable for pose-based SLP, especially on challenging datasets such as How2Sign [28, 60].

## 3. Cross-View Motion Curation

This section outlines the proposed cross-view 3D data curation pipeline (Figure 2) used for the creation of How2Sign-MV3D, ASLCitizen-3D, and MM-WLAuslan-MV3D, reconstructed from How2Sign [12], ASLCitizen [10], and MM-WLAuslan [53] datasets, respectively. Detailed mathematical formulations are deferred to Appendix A.

## 3.1. Reliability-Guided Per-View Fusion

AiOS [56] is used to process multiple frames and reconstruct a stable global reference for the body and wrists. In parallel, WiLoR [43] is used to reconstruct fine handshapes from individual frames. Since WiLoR is susceptible to failure under motion blur and occlusion, we verify WiLoR against the AiOS wrist references to identify incorrect predictions. Specifically, a WiLoR hand frame is marked as unreliable if it meets any of three criteria: (i) the hand detection is missing, (ii) the SO(3) wrist orientation severely disagrees with the AiOS reference, or (iii) the wrist exhibits severe angular jitter unsupported by the stable AiOS motion. Once these unreliable frames are identified, the corrupted wrist and finger intervals are repaired in rotational space using boundary propagation and interpolation across frames. The final fused per-view SMPL-X [41] sequence is then obtained by replacing the original AiOS hand parameters with the repaired WiLoR wrist and finger blocks, preserving the stable AiOS global translation, body pose, and shape. Detailed mathematical formulations are provided in Appendix A.1, while view-specific detection statistics and examples are presented in Appendix E.

![](images/bb926207c748f288fa3374342761aa504daaa04cd3f8a5d3f8399f61cde10314.jpg)  
Figure 2. Cross-view motion curation pipeline. We first build per-view SMPL-X [41] sequences using AiOS [56] as a stable body backbone and WiLoR [43] as detailed hand evidence. The frontal and side streams are then aligned, synchronized, cleaned of non-signing frames, and used for cross-view body and finger repair. The side view acts as verifier and repair evidence; the final training target for SL is the refined frontal stream.

## 3.2. Alignment, Synchronization, Non-Sign Frames

To compare the streams from different cameras, we first extract the 3D joint coordinates $J _ { t } ^ { v }$ from the fused per-view SMPL-X sequences obtained in the previous step. However, the streams cannot be compared directly because the side-view pose estimates are globally shifted, and the videos may have a temporal offset. To resolve this, we first apply a robust spatial alignment followed by a temporal synchronization search that minimizes the Mean Per Joint Position Error (MPJPE):

$$
\delta ^ { \star } = \arg \operatorname* { m i n } _ { \delta \in \mathcal { D } } \frac { 1 } { T _ { \delta } | B | } \sum _ { t \in \mathcal { T } _ { \delta } } \sum _ { j \in B } \left\| J _ { t , j } ^ { \mathrm { f } } - \tilde { J } _ { t - \delta , j } ^ { \mathrm { s } } \right\| _ { 2 } .\tag{1}
$$

Here, $\delta ^ { \star }$ is the optimal temporal offset selected from a search range $\mathcal { D } \left( i . e . , [ - 8 , 8 ] \right.$ frames), $\mathcal { T } _ { \delta }$ denotes the set of overlapping frame indices, and $T _ { \delta } ~ = ~ | T _ { \delta } |$ . A positive δ aligns source frame $t - \delta$ with target frame t. The notation $J _ { t , j } ^ { \mathrm { f } }$ denotes the 3D position of joint j in the frontal view. The subset B restricts the calculation to the 22 stable body joints, deliberately excluding the hand joints because they are the noisiest parts before refinement. The term $\tilde { J } _ { t , j } ^ { \mathrm { s } }$ represents the side-view joints after a three-stage spatial alignment to the frontal view (which calculates relative frame-wise rotations, extracts a stable average rotation via singular value decomposition (SVD) to eliminate jitter, and aligns the global centroids and pelvis).

Even after synchronization, continuous signing clips can include preparation, ending motion, pauses, and recording artifacts. Pause and neutral position cues have long been used for continuous sign segmentation [30], and recent approaches also treat temporal localization as a central problem [25, 44]. This is consistent with high-quality signing resources that annotate or curate the signing interval rather than treating all recorded frames as lexical motion. We therefore mask out these non-signing frames using a combination of geometric and motion cues. Specifically, a frame is classified as active signing only when three conditions are simultaneously met: (i) sufficient mouth activity, (ii) significant finger motion, and (iii) sufficient wrist activity gated by scale-normalized wrist–elbow geometry. Long inactive runs are removed jointly from all views, while short gaps are bridged via interpolation to avoid discontinuities. The conjunction of these cues provides a high-precision mask: wrist position alone confuses held rest poses with signs, while finger or mouth motion alone can be triggered by incidental activity. To ensure optimal tracking accuracy, this activity detection is performed solely on the frontal view, but the resulting temporal mask is applied to truncate and edit all views consistently. Further details on this step are given in Appendix A.2.

## 3.3. Cross-View Body and Finger Repair

With the streams synchronized and non-signing frames removed, we cross-verify the sequences to repair abrupt failures. For body joints, we assume that a true fast signing motion should be consistently visible in all cameras. To quantify this, we compute the frame-to-frame geodesic angular velocity for each joint in SO(3). A spatial jitter is flagged for correction only if its angular velocity exhibits a severe spike in the target frontal view but remains strictly below a low-jitter threshold in the opposite view. This mutual-exclusion condition ensures we only target artifactual glitches, preventing the suppression of legitimate, consistent fast movements. The corrupted frames are then inpainted using spherical linear interpolation and smoothed.

The most sign-critical residual error is missing finger extensions. Because single-frame estimators struggle to reconstruct fast-moving individual fingers under severe motion blur, we design a targeted repair mechanism based on isolated extension statistics. First, we extract the X-axis flexion component in each joint’s geometric finger frame and map it to a normalized straightness score $S _ { k } \in [ 0 , 1 ]$ for each finger k (Appendix A.3.1). We then evaluate a reliable source observation (denoted as src, representing either the opposite view or a temporally adjacent frame) to detect anomalously isolated extensions. Specifically, a finger k is flagged as confidently extended if its straightness exceeds 1.15 times the mean of the remaining fingers, while the variance of the remaining fingers is strictly below a threshold $\tau _ { \mathrm { v a r } }$ . This geometric filter effectively isolates intended pointing or gesturing motions from a globally bent hand. Among the candidates satisfying the above isolation conditions, we select the most prominent one $k ^ { \star } =$ arg max<sub>k</sub> $E _ { k } ^ { \mathrm { s r c } }$ , where $E _ { k } ^ { \mathrm { s r c } } = S _ { k } ^ { \mathrm { s r c } } - \mu ( S _ { \backslash k } ^ { \mathrm { s r c } } )$ is the excess straightness. We then verify if this specific extended finger has falsely collapsed in the target destination observation (dst) due to motion blur, and apply a bounded directional repair:

$$
\begin{array} { r l } & { c _ { k ^ { \star } } = \mathbf { 1 } [ S _ { k ^ { \star } } ^ { \mathrm { d s t } } < 0 . 6 5 S _ { k ^ { \star } } ^ { \mathrm { s r c } } ] , } \\ & { \phi _ { x , j } ^ { \mathrm { d s t } }  \mathrm { s i g n } ( \phi _ { x , j } ^ { \mathrm { d s t } } ) \operatorname* { m a x } ( \vert \phi _ { x , j } ^ { \mathrm { d s t } } \vert - c _ { k ^ { \star } } \lambda w _ { j } , 0 ) . } \end{array}\tag{2}
$$

Here, $c _ { k ^ { \star } } ~ = ~ 1$ triggers the repair if the target straightness falls below 65% of the reliable source observation. The variable $\phi _ { x , j }$ is the adjusted-frame flexion angle of joint j in the target finger, with $w = ( 0 . 7 , 0 . 7 , 0 . 5 )$ . The scale $\lambda = 1 . 2 E _ { k ^ { \star } } ^ { \mathrm { s r c } } / \sum _ { i = 1 } ^ { 2 } a _ { i } w _ { i } / u _ { k ^ { \star } , i }$ converts normalized straightness to angular units, using $a = ( 0 . 7 , 0 . 3 )$ and the joint bounds u specified in Appendix A.3.2. This two-pass process, first anchoring on the opposite view, then propagating to adjacent frames ensures that we only restore reliably observed extensions lost to blur, without forcing spurious open-hand artifacts. Further details on this step are given in Appendix A.3.

## 4. Conditional Sign Language Production

This section outlines the proposed tokenizer-languagemodel SLP method. The method retains the efficient threestream interface of SOKE [79] and introduces information connectivity across tokenizer streams and enhanced retrieval. Detailed mathematical formulations are deferred to Appendix B and Appendix C.

![](images/cac08f8a9d8844c0fd3674bf67165ce1de28c5984acc584d88cfc4dd90a48a6d.jpg)  
Figure 3. Sign-aware ST-VQVAE. Unlike independent body/hand tokenizers, our encoder processes the 43-joint upperbody and hands skeleton jointly, pools it to three semantic slots only at the bottleneck, and quantizes the slots with body, left, and right hand codebooks. This preserves body-hands coupling while retaining the three-token-stream interface used by autoregressive SLP models.

## 4.1. Sign-Aware ST-VQVAE

SOKE discretizes different parts $( i . e . ,$ body, left, and right hand) with a separate VQ-VAE. This matches the decoder interface, but removes cross-part interaction inside the tokenizer. We adapt the skeleton-aware latent hierarchy VAE of SALAD [22] to sign motion: instead of modeling a continuous full-body motion skeleton, our encoder pools a 43-joint upper-body and hands skeleton into three terminal slots aligned with the decoder interface (Figure 3). Sign meaning depends on coordinated body, wrist, and finger articulation: the same handshape can be interpreted differently depending on its trajectory, orientation, and the opposite hand. We therefore encode the skeleton jointly and factorize only at the bottleneck; each token stream is predicted separately but is formed from a full signing configuration.

As illustrated in Figure 3, given an SMPL-X motion sequence, our encoder progressively pools the spatial skeleton from 43 joints down to 3 terminal slots, while temporally downsampling the sequence to one-quarter of its original frame rate. Spatial resolutions correspond to progressive anatomical groupings. This results in a final continuous latent sequence where each time step contains three 512- dimensional semantic slots representing the body, left hand, and right hand. These continuous slot representations are then quantized into discrete tokens using part-specific codebooks. Because hands require more capacity to capture rich lexical handshape variations, we allocate larger codebook sizes for the left and right hands (192 each) compared to the body codebook (96). The decoder mirrors the encoder and reconstructs the SMPL-X motion. We train the tokenizer with rotation reconstruction, mesh-level geometric supervision, and vector-quantization regularization:

$$
\mathcal { L } _ { \mathrm { t o k } } = \lambda _ { \mathrm { r o t } } \mathcal { L } _ { \mathrm { r o t } } + \lambda _ { \mathrm { m e s h } } \mathcal { L } _ { \mathrm { m e s h } } + \lambda _ { \mathrm { v q } } \mathcal { L } _ { \mathrm { v q } } .\tag{3}
$$

Rotation supervision preserves local articulation, while mesh-level supervision penalizes visible geometric errors after SMPL-X decoding. This combination matters for sign motion because a small rotation error can still create a visibly wrong fingertip or wrist trajectory.

## 4.2. Pseudo-Gloss Conditioning

Raw text is a noisy source interface for SLP because sign languages differ in lexical choice, word order, and alignment granularity. It is also a weak interface for retrieval: an isolated sign dictionary is indexed by sign-level labels (glosses) rather than text tokens. Therefore, we convert text sentences x into a pseudo-gloss sequence $g = \Phi _ { \mathrm { L L M } } ( x )$ offline using GPT-5.2 with few-shot prompting. The prompt follows the appendix template of Zhang et al. [76], with an additional ASLCitizen [10] vocabulary constraint. We use pseudo-gloss as a practical intermediate representation; the goal is to provide the generator and retrieval module with a source sequence closer to sign-level motion. To maintain SOKE [79] generation structure, we introduce a lightweight gloss adapter, consisting of a word-level gloss tokenizer, 512-d learned gloss and positional embeddings, a 4-layer Transformer encoder (8 heads, 2048 FFN, 0.1 dropout), and a linear projection to the 1024-d mBART hidden space. The downstream decoder remains unchanged. Pseudo-gloss evaluation settings, reference-agreement results, and vocabulary coverage are provided in Appendix C.1.1.

## 4.3. Keyframe-Aware Retrieval

In SOKE-style retrieval, isolated sign clips are processed into discrete tokens via a VQ-VAE to serve as auxiliary prompts. However, these clips often include non-lexical preparation and retraction motions; these frames can be treated as retrieval prompt noise. We identify lexically informative regions in isolated sign clips using a TCNbased [4] classifier trained on ASLCitizen [10] (which provides approximately 20 video instances per sign). Because extracting attribution maps from the same data used for training yields overfitted and biased saliency, we adopt a 3-fold cross-validation protocol: we train identical TCNs on two data splits and extract inferences on the held-out third. We compute these out-of-fold attributions using Integrated Gradients (IG) [57], empirically choosing it over alternatives such as Grad-CAM [52]. (See Appendix C.5 for IG vs. Grad-CAM comparisons and soft-weighting ablations.) For each isolated sign clip, the frozen ST-VQVAE encodes the motion into discrete tokens, the TCN classifier processes the raw 43-joint skeleton and produces a jointtime attribution map. To align this continuous saliency with the discrete generator, we first average the IG values across all 43 joints and normalize the scores within each clip from 0 to 1 (Figure 4). Since the main SLP generator operates at a 4× temporally downsampled rate, we apply a corresponding 4× temporal average pooling to the 1D saliency array to match the downsampled token timeline. Finally, a constrained retrieval budget of up to 10 tokens is selected from the highest-weighted regions of the encoded sequence and concatenated with the source condition. This processing strategy keeps the retrieval module compatible with the three-stream token interface, preventing the concatenation of overly long input sequences from degrading the generator’s training, while ensuring the prompt budget is strictly concentrated on informative lexical motion.

![](images/d9d5b96d8009dc4fab033af1126ecab5da2229b5810d3408b6d0f847e0ed0458.jpg)  
Figure 4. Keyframe distribution. In the ASLCitizen [10] dataset, the average and maximum IG [57] saliency of joints over time follow a rough Gaussian trend overall yet vary widely across signs, motivating per-clip rather than global-envelope selection.

## 4.4. Autoregressive Token Prediction

The final source representation concatenates the glossadapted sequence and retrieval prompt embeddings:

$$
H _ { \mathrm { s r c } } = [ \hat { H } _ { g } ; H _ { \mathrm { r e t } } ] ,\tag{4}
$$

with $H _ { \mathrm { s r c } } = \hat { H } _ { g }$ when no retrieval entry is available. The generator predicts the three token streams autoregressively:

$$
\mathcal { L } _ { \mathrm { g e n } } = - \sum _ { t = 1 } ^ { T ^ { \prime } } \sum _ { p \in \{ b , l , r \} } \log p _ { \theta } \left( z _ { t } ^ { ( p ) } \mid H _ { \mathrm { s r c } } , Z _ { < t } \right) .\tag{5}
$$

Training uses teacher-forced cross-entropy over the three vocabularies. At inference time, the predicted tokens are decoded by the frozen ST-VQVAE decoder into continuous SMPL-X motion (Figure 5).

![](images/85d5edce33db6cd815b6655c211dd4a37d61254a082988f00d91d9289222f8dd.jpg)  
Figure 5. SLP with sign classifier enhanced retrieval. We train an auxiliary isolated-sign classifier and extract joint-time saliency W using Integrated Gradients (IG). After temporal pooling to the ST-VQVAE token rate, we use the saliency scores to select a fixed 10-token window from the encoded isolated-sign sequence $Z _ { w } .$ The selected tokens serve as a compact retrieval prompt for autoregressive sign generation.

## 5. Experiments

We evaluate the MV-Sign framework for data and generation quality using the proposed How2Sign-MV3D dataset.

## 5.1. Experimental Setup

Datasets. The regenerated How2Sign-MV3D release follows the How2Sign train/validation/test partition [12] and contains 30,903, 1,711, and 2,325 completed motion sequences, respectively. Of the 35,241 clips in our processing manifest, 289 were excluded by input screening and 13 failed processing. Clips use the official manually realigned sentence timestamps; motion-side non-sign removal reduces untranscribed waiting intervals without additional text reannotation (Appendix F). The tokenizer and generator are trained only on How2Sign-MV3D. ASLCitizen [10] is used only for isolated sign retrieval and saliency estimation. SGNify-MV contains 57 isolated signs with groundtruth meshes and is used in the Appendix as a diagnostic rather than as the main evidence for cross-view curation on continuous signing.

Metrics. For SLP on How2Sign, we follow prior work and report DTW-JPE and DTW-PA-JPE for body and hand regions [5, 79]. DTW aligns generated and reference sequences in time; PA denotes Procrustes alignment before computing joint error. For tokenizer reconstruction, the input and reconstruction are already frame-aligned, so Table 3 and Table 4 report PA-JPE for joints and PA-VPE for vertices without DTW. PA is computed on the evaluated region: body uses the upper-body joint set, and hand scores align the left and right hands separately before averaging. Prior 3D SLP methods for How2Sign, including NSA [5] and SOKE [79], build pseudo-3D training targets and then evaluate held-out text-to-motion generation against the corresponding How2Sign motion-space protocol. We therefore treat Table 2 as an end-to-end pipeline comparison, where data curation and generator design both affect the final test set error, rather than as a pure model comparison under fixed ground-truth 3D labels. The SGNify-MV diagnostic in the Appendix reports translation-removed vertex-to-vertex error (TR-V2V, mm) on the upper body excluding the face (UBody-F), left, and right hand. We omit back-translation because recent pose-based SLP evaluation studies note that such metrics require dataset- or direction-specific pose-totext evaluators whose code and weights are rarely standardized or released, making private BLEU scores difficult to compare [28, 60]. We therefore focus on motion-space metrics and qualitative motion inspection.

Table 1. Quantitative How2Sign-MV3D results. We report DTW-PA-JPE only. Reported rows are taken from prior work and hand values are averages
<table><tr><td>Training</td><td>UBody-F ↓LHand ↓ RHand ↓</td></tr><tr><td>SOKE (3 datasets, reported)</td><td>6.82 2.35 (hands avg.)</td></tr><tr><td>SOKE (NSA&#x27;s H2S, reported)</td><td>7.92 3.07 (hands avg.)</td></tr><tr><td>SOKE (H2S-MV3D-only)</td><td>4.58 1.92</td></tr></table>

Implementation details. The ST-VQVAE is trained for 150 epochs with AdamW, learning rate $2 \times 1 0 ^ { - 4 }$ , batch size 32, latent width d = 512, temporal downsampling factor 4, and codebook sizes (96, 192, 192). The autoregressive generator is an mBART-large SOKE-style backbone with three token heads and is trained with teacher forcing. Pseudo-glosses are generated offline with GPT-5.2 using the few-shot prompt style of Zhang et al. [76], with an added ASLCitizen vocabulary constraint. Retrieval uses 10-token prompts selected from isolated sign token sequences. Inference uses greedy decoding for deterministic comparison.

## 5.2. How2Sign-MV3D Results

To test whether denoised motion improves the actual SLP training signal, we compare SOKE-style training and evaluation under different data conditions. Table 1 provides downstream evidence that training on How2Sign-MV3D yields lower motion-space errors within a SOKE-style pipeline, reducing both upper-body and hand errors relative to the reported SOKE results.

## 5.3. SLP Results

Table 2 compares our framework with prior text-to-sign and 3D SLP methods under the How2Sign motion-space protocols used in this line of work. Because How2Sign has no ground-truth 3D motion, these systems evaluate complete pipelines: each method first constructs or adopts pseudo-3D supervision, then trains a generator and evaluates held-out test samples in motion space. Our framework obtains the lowest body and hand errors among the compared pipelines, and the qualitative comparisons remain important for interpreting differences caused by pseudo-target noise.

Table 2. Quantitative SLP results. Following prior SLP work, we report DTW-PA-JPE and DTW-JPE for body and hands. Since each method relies on pseudo-3D supervision, this table compares complete pipelines under prior motion-space protocols, not isolated generator architectures under shared ground-truth. Baseline rows follow the SOKE comparison.
<table><tr><td rowspan="2">Method</td><td colspan="2">DTW-PA-JPE↓</td><td colspan="2">DTW-JPE↓</td></tr><tr><td>Body</td><td>Hands</td><td>Body</td><td>Hands</td></tr><tr><td>NAR [26]</td><td>13.94</td><td>11.80</td><td></td><td></td></tr><tr><td>Prog. Trans. [49]</td><td>14.15</td><td>11.57</td><td>14.74</td><td>30.17</td></tr><tr><td>Text2Mesh [50]</td><td>13.99</td><td>13.47</td><td>15.50</td><td>32.97</td></tr><tr><td>Adv. Train. [48]</td><td>13.78</td><td>11.17</td><td></td><td></td></tr><tr><td>T2S-GPT [70]</td><td>11.48</td><td>6.39</td><td>12.65</td><td>18.44</td></tr><tr><td>NSA [5]</td><td>7.83</td><td>7.33</td><td></td><td></td></tr><tr><td>S-MotionGPT [27]</td><td>11.23</td><td>4.39</td><td>12.41</td><td>13.74</td></tr><tr><td>SOKE [79]</td><td>6.82</td><td>2.35</td><td>7.75</td><td>10.08</td></tr><tr><td>Ours</td><td>4.58</td><td>1.86</td><td>4.98</td><td>7.95</td></tr></table>

Table 3. Tokenizer ablation. Using the same codebook size and dimension, our tokenizer reaches better reconstruction results compared to the SOKE tokenizer. DTW is not used because tokenizer reconstruction is frame-aligned.
<table><tr><td rowspan="2">Method</td><td colspan="2">PA-JPE↓</td><td colspan="2">PA-VPE↓</td></tr><tr><td>Body</td><td>Hands</td><td>All</td><td>Hands</td></tr><tr><td>SOKE (VQ-VAE)</td><td>19.37</td><td>6.65</td><td></td><td></td></tr><tr><td>SOKE (VQ-VAE, H2S-MV3D)</td><td>14.00</td><td>6.24</td><td>18.12</td><td>6.07</td></tr><tr><td>Ours (ST-VQVAE)</td><td>14.21</td><td>5.84</td><td>16.59</td><td>5.84</td></tr></table>

Table 4. Tokenizer loss ablation. Mesh supervision improves global geometry, while the hand-weighted combined loss gives the best hand reconstruction. DTW is not used because tokenizer reconstruction is frame-aligned.
<table><tr><td>Loss</td><td colspan="2">PA-JPE↓</td><td colspan="2">PA-VPE↓</td></tr><tr><td></td><td>Body</td><td>Hands</td><td>All</td><td>Hands</td></tr><tr><td>Rotation-only</td><td>18.48</td><td>6.23</td><td>21.93</td><td>6.04</td></tr><tr><td>Mesh-only</td><td>11.61</td><td>10.24</td><td>13.33</td><td>9.96</td></tr><tr><td>Rotation + mesh</td><td>14.21</td><td>5.84</td><td>16.59</td><td>5.84</td></tr></table>

## 5.4. Ablations

Table 3 reports that our ST-VQVAE improves hands and mesh reconstruction while retaining the three-stream interface required by the autoregressive decoder. The result supports the tokenizer design: prediction remains part-wise, but each token is produced after jointly observing the 43-joint skeleton. Table 4 reports the tradeoff behind the final objective. Mesh-only training reduces global vertex and bodyjoint errors because vertex-space losses are dominated by larger body motion and surface displacement, but this hurts the finer hand reconstruction needed for signing. We therefore use a combined rotation–mesh objective with extra hand weighting, accepting a modest increase in body/global reconstruction error in exchange for better hand fidelity.

Table 5. Component ablation. DTW-PA-JPE for the main generation variants and relative gains over the reported SOKE body and hands average. Components provide complementary gains rather than a strictly monotonic improvement in every region. Percentage gains are computed from the displayed scores.
<table><tr><td>Setting</td><td colspan="6">MV3D Gloss Adapter Keyframe ST-VQVAE</td><td colspan="3">DTW-PA-JPE ↓</td><td colspan="2">Gain ↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2">Body LHand RHand</td><td>Body</td><td></td><td>Hands</td></tr><tr><td>SOKE (reported)</td><td>一</td><td></td><td>一</td><td>Original</td><td>Original</td><td>6.82</td><td></td><td>2.35 avg.</td><td>一</td><td></td><td></td></tr><tr><td>SOKE (H2S-MV3D)</td><td>V</td><td>1√</td><td>1</td><td>一</td><td></td><td>4.58 4.49</td><td>1.92 1.93</td><td>1.98</td><td></td><td>32.8% 17.0% 34.2% 16.6%</td><td></td></tr><tr><td>+ adapter</td><td>√ √</td><td>√</td><td>√</td><td></td><td></td><td>4.52</td><td>1.91</td><td></td><td>1.99 1.97</td><td>33.7% 17.4%</td><td></td></tr><tr><td>+ retrieval + ST-VQVAE</td><td></td><td>√</td><td>√</td><td>√ √</td><td>√</td><td>4.58</td><td></td><td></td><td>1.91</td><td>32.8% 20.9%</td><td></td></tr><tr><td></td><td>V</td><td></td><td></td><td></td><td></td><td></td><td>1.81</td><td></td><td></td><td></td><td></td></tr></table>

Table 5 reports cumulative model components on How2Sign-MV3D. Using the reported SOKE body score of 6.82 and hands average of 2.35 as reference, How2Sign-MV3D, pseudo-gloss adaptation, keyframe retrieval, and the final ST-VQVAE tokenizer improve body DTW-PA-JPE by 32.8%, 34.2%, 33.7%, and 32.8%, respectively, and hands-average DTW-PA-JPE by 17.0%, 16.6%, 17.4%, and 20.9%. The components do not improve every region monotonically, but they provide complementary benefits: pseudo-gloss improves body alignment, keyframe retrieval concentrates the prompt budget, and ST-VQVAE improves hand-aligned metrics.

## 5.5. Qualitative and Diagnostic Analysis

Figure 6 presents qualitative evidence for the data curation stage. The comparison focuses on wrist and finger artifacts in noisy frontal-view pseudo-3D motions and shows how the refined How2Sign-MV3D target removes viewspecific failures that would otherwise propagate into production. This visual evidence is the main qualitative support for cross-view repair of How2Sign clips; SGNify-MV is kept as a controlled diagnostic in the Appendix. Figure 7 provides representative qualitative SLP results.

The component ablation in Table 5 should be read as a diagnostic of complementary denoising effects rather than as a strictly monotonic component stack. Pseudo-gloss mainly improves body/source alignment by giving the generator and retrieval module a sign-level conditioning sequence. Keyframe-aware retrieval slightly improves hands by spending the fixed prompt budget on lexical motion instead of preparation or retraction. The ST-VQVAE gives the clearest hands gain because it reconstructs body and hands jointly before factorizing into the three token streams used by the decoder.

![](images/7e6748c604bab074f726a911a1315d49b208d153e151bd4871edf875f0231890.jpg)

Figure 6. Qualitative How2Sign-MV3D results. The figure compares noisy front-view 3D motions with our cross-view-curated motions, emphasizing wrist and finger repair.  
![](images/be13d93cf8e3806461f0203ecc1f384c45495457adbc06ae786e422988b30068.jpg)  
Figure 7. Qualitative SLP results. Comparisons of generated sign sequences from our method (trained with our How2Sign-MV3D pseudo-3D motion) and the state-of-the-art SOKE [79] (trained with NSA [5] pseudo-3D motion) on How2Sign [12] test set.

## 6. Conclusion

We presented MV-Sign, a denoising framework for 3D sign language production. On the data side, the framework uses cross-view evidence to repair noisy monocular 3D sign motion. On the model side, the proposed ST-VQVAE, pseudo-gloss adapter, and keyframe-aware retrieval reduce representation, language-conditioning, and retrieval-prompt noise. How2Sign experiments provide downstream evidence that curated 3D supervision improves SOKE-style training, and the full framework achieves the strongest motion-space metrics among compared SLP pipelines.

Limitations and impact. Cross-view evidence is most useful when the frontal view is ambiguous or occluded. If the frontal view is already clean and the side view is temporally mismatched, side evidence can introduce small noise; our pipeline therefore treats side views as reliabilityweighted repair evidence rather than mandatory targets. The retrieval module estimates joint-time saliency, but the current three-stream token generator consumes an averaged signal; future joint-aware token generators could use the full saliency map. SLP systems should be developed as assistive tools and not as replacements for human interpreters.

## References

[1] Samuel Albanie, Gul Varol, Liliane Momeni, Triantafyllos¨ Afouras, Joon Son Chung, Neil Fox, and Andrew Zisserman. Bsl-1k: Scaling up co-articulated sign language recognition using mouthing cues. In European conference on computer vision, pages 35–53. Springer, 2020. 2

[2] Samuel Albanie, Gul Varol, Liliane Momeni, Hannah Bull,¨ Triantafyllos Afouras, Himel Chowdhury, Neil Fox, Bencie Woll, Rob Cooper, Andrew McParland, et al. Bbcoxford british sign language dataset. arXiv preprint arXiv:2111.03635, 2021. 2

[3] Vassilis Athitsos, Carol Neidle, Stan Sclaroff, Joan Nash, Alexandra Stefan, Quan Yuan, and Ashwin Thangali. The american sign language lexicon video dataset. In 2008 IEEE computer society conference on computer vision and pattern recognition workshops, pages 1–8. IEEE, 2008. 2

[4] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, 2018. 5

[5] Vasileios Baltatzis, Rolandos Alexandros Potamias, Evangelos Ververas, Guanxiong Sun, Jiankang Deng, and Stefanos Zafeiriou. Neural sign actors: A diffusion model for 3d sign language production from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1985–1995, 2024. 1, 2, 6, 7, 8, 21, 22

[6] Annelies Braffort, Laurence Bolot, Emilie Chetelat-Pel ´ e,´ Annick Choisier, Maxime Delorme, Michael Filhol, Jer´ emie´ Segouat, Cyril Verrecchia, Flora Badin, and Nadege Devos.\` Sign language corpora for analysis, processing and evaluation. In LREC, 2010. 2

[7] Danielle Bragg, Oscar Koller, Mary Bellard, Larwan Berke, Patrick Boudreault, Annelies Braffort, Naomi Caselli, Matt Huenerfauth, Hernisa Kacorri, Tessa Verhoef, et al. Sign language recognition, generation, and translation: An interdisciplinary perspective. In Proceedings of the 21st international ACM SIGACCESS conference on computers and accessibility, pages 16–31, 2019. 2

[8] Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Yanjun Wang, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, et al. Smpler-x: Scaling up expressive human pose and shape estimation. Advances in Neural Information Processing Systems, 36:11454–11468, 2023. 2

[9] Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. Neural sign language translation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7784–7793, 2018. 2

[10] Aashaka Desai, Lauren Berger, Fyodor O. Minakov, Vanessa Milan, Chinmay Singh, Kriston Pumphrey, Richard E. Ladner, Hal Daume III, Alex X. Lu, Naomi Caselli, and Danielle´ Bragg. Asl citizen: A community-sourced dataset for advancing isolated sign language recognition, 2023. 2, 5, 6, 17, 19, 23

[11] Lu Dong, Lipisha Chaudhary, Fei Xu, Xiao Wang, Mason Lary, and Ifeoma Nwogu. Signavatar: Sign language 3d motion reconstruction and generation. In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pages 1–10. IEEE, 2024. 2

[12] Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, and Xavier Giro-i Nieto. How2sign: a large-scale multi modal dataset for continuous american sign language. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2735–2744, 2021. 1, 2, 6, 8, 17, 21

[13] E Efthimiou, S Fotinea, T Hanke, J Glauert, R Bowden, Annelies Braffort, Christophe Collet, P Maragos, and F Goudenove. Dicta-sign: sign language recognition, generation and modelling with appllications in deaf communication. In International Conference on Language Resources and Evalua tion, 2010. 2

[14] Eleni Efthimiou, Stavroula-Evita Fotinea, Thomas Hanke, John Glauert, Richard Bowden, Annelies Braffort, Christophe Collet, Petros Maragos, and Franc¸ois Lefebvre-Albaret. The dicta-sign wiki: Enabling web communication for the deaf. In International conference on computers for handicapped persons, pages 205–212. Springer, 2012. 2

[15] Sen Fang, Chunyu Sui, Yanghao Zhou, Xuedong Zhang, Hongbin Zhong, Yapeng Tian, and Chen Chen. Sign diff: Diffusion model for american sign language produc tion. In 2025 IEEE 19th International Conference on Au tomatic Face and Gesture Recognition (FG), pages 1–11. IEEE, 2025. 2

[16] Yao Feng, Vasileios Choutas, Timo Bolkart, Dimitrios Tzionas, and Michael J Black. Collaborative regression of expressive bodies using moderation. In 2021 International Conference on 3D Vision (3DV), pages 792–804. IEEE, 2021. 2

[17] Maria-Paola Forte, Peter Kulits, Chun-Hao P Huang, Vasileios Choutas, Dimitrios Tzionas, Katherine J Kuchenbecker, and Michael J Black. Reconstructing signing avatars from video using linguistic priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12791–12801, 2023. 21, 22

[18] Jia Gong, Lin Geng Foo, Yixuan He, Hossein Rahmani, and Jun Liu. Llms are good sign language translators. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18362–18372, 2024. 1, 2

[19] Thomas Hanke, Marc Schulder, Reiner Konrad, and Elena Jahn. Extending the public dgs corpus in size and depth. In Proceedings of the LREC2020 9th workshop on the representation and processing of sign languages: Sign language resources in the service of the language community, technological challenges and application perspectives, pages 75– 82, 2020. 2

[20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2

[21] Sepp Hochreiter and Jurgen Schmidhuber. Long short-term¨ memory. Neural computation, 9(8):1735–1780, 1997. 2

[22] Seokhyeon Hong, Chaelin Kim, Serin Yoon, Junghyun Nam, Sihun Cha, and Junyong Noh. Salad: Skeleton-aware latent diffusion for text-driven motion generation and editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7158–7168, 2025. 2, 4

[23] Hezhen Hu, Weichao Zhao, Wengang Zhou, and Houqiang Li. Signbert+: Hand-model-aware self-supervised pretraining for sign language understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):11221– 11239, 2023. 1

[24] Hezhen Hu, Zhiwen Fan, Tianhao Wu, Yihan Xi, Seoyoung Lee, Georgios Pavlakos, and Zhangyang Wang. Expressive gaussian human avatars from monocular rgb video, 2024. 21

[25] Jie Huang, Wengang Zhou, Qilin Zhang, Houqiang Li, and Weiping Li. Video-based sign language recognition without temporal segmentation. In Proceedings of the AAAI conference on artificial intelligence, 2018. 2, 3

[26] Eui Jun Hwang, Jung Ho Kim, Suk Min Cho, and Jong C. Park. Non-autoregressive sign language production via knowledge distillation, 2022. 7

[27] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. In Advances in Neural Information Processing Systems, 2023. 7

[28] Zifan Jiang, Colin Leong, Amit Moryossef, Anne Gohring,¨ Annette Rios, Oliver Cory, Maksym Ivashechkin, Neha Tarigopula, Biao Zhang, Rico Sennrich, and Sarah Ebling. Meaningful pose-based sign language evaluation, 2025. 2, 6, 22

[29] Hamid Reza Vaezi Joze and Oscar Koller. Ms-asl: A largescale data set and benchmark for understanding american sign language. arXiv preprint arXiv:1812.01053, 2018. 2

[30] Shujjat Khan, Donald G. Bailey, and Gourab Sen Gupta. Pause detection in continuous sign language. International Journal of Computer Applications in Technology, 50(1/2): 75–83, 2014. 3

[31] Muhammed Kocabas, Chun-Hao P Huang, Otmar Hilliges, and Michael J Black. Pare: Part attention regressor for 3d human body estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11127– 11137, 2021. 2

[32] Kaustubh Kundu, Hrishav Bakul Barua, Lucy Robertson-Bell, Zhixi Cai, and Kalin Stefanov. Dexavatar: 3d sign language reconstruction with hand and body pose priors. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5842–5852, 2026. 2, 21

[33] Dongxu Li, Cristian Rodriguez, Xin Yu, and Hongdong Li. Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision, pages 1459–1469, 2020. 2

[34] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6):194–1, 2017. 2

[35] Xiaoben Li, Mancheng Meng, Ziyan Wu, Terrence Chen, Fan Yang, and Dinggang Shen. Human mesh recovery from arbitrary multi-view images. arXiv preprint arXiv:2403.12434, 2024. 2

[36] Zecheng Li, Wengang Zhou, Weichao Zhao, Kepeng Wu, Hezhen Hu, and Houqiang Li. Uni-sign: Toward unified sign language understanding at scale. arXiv preprint arXiv:2501.15187, 2025. 1

[37] Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, and Yu Li. One-stage 3d whole-body mesh recovery with component aware transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21159–21168, 2023. 2, 21, 22

[38] Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris Mc-Clanahan, Esha Uboweja, Michael Hays, Fan Zhang, Chuo-Ling Chang, Ming Guang Yong, Juhyun Lee, et al. Mediapipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172, 2019. 2

[39] Daniel Morales-Brotons, Thijs Vogels, and Hadrien Hendrikx. Exponential moving average of weights in deep learning: Dynamics and benefits. Transactions on Machine Learning Research, 2024. 17

[40] Mathias Muller, Zifan Jiang, Amit Moryossef, Annette Rios,¨ and Sarah Ebling. Considerations for meaningful sign language machine translation based on glosses. In Proceedings of the 61st Annual Meeting of the Association for Computa tional Linguistics (Volume 2: Short Papers), pages 682–693, 2023. 2

[41] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019. 1, 2, 3, 13, 16

[42] Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3d with transformers, 2023. 2

[43] Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. Wilor: End-to-end 3d hand localization and reconstruction in-the-wild. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12242–12254, 2025. 2, 3, 13, 21

[44] Katrin Renz, Nicolaj C. Stache, Neil Fox, Gul Varol, and¨ Samuel Albanie. Sign segmentation with changepointmodulated pseudo-labelling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 3403–3412, 2021. 3

[45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2

[46] Javier Romero, Dimitrios Tzionas, and Michael J Black. Em bodied hands: Modeling and capturing hands and bodies together. arXiv preprint arXiv:2201.02610, 2022. 2

[47] Yu Rong, Takaaki Shiratori, and Hanbyul Joo. Frankmocap: A monocular 3d whole-body pose estimation system via regression and integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Work shops, pages 1749–1759, 2021. 2

[48] Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. Adversarial training for multi-channel sign language production, 2020. 7

[49] Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. Progressive transformers for end-to-end sign language pro-

duction. In European Conference on Computer Vision, pages 687–705. Springer, 2020. 2, 7

[50] Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. Signing at scale: Learning to co-articulate signs for largescale photo-realistic sign language production. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5141–5151, 2022. 2, 7

[51] Adam Schembri, Jordan Fenlon, Ramas Rentelis, Sally Reynolds, and Kearsy Cormier. Building the british sign language corpus. Language Documentation & Conservation, 7: 136–154, 2013. 2

[52] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. International Journal of Computer Vision, 128(2):336–359, 2019. 5

[53] Xin Shen, Heming Du, Hongwei Sheng, Shuyun Wang, Hui Chen, Huiqiang Chen, Zhuojie Wu, Xiaobiao Du, Jiaying Ying, Ruihan Lu, et al. Mm-wlauslan: multi-view multi-modal word-level australian sign language recognition dataset. Advances in Neural Information Processing Systems, 37:69700–69715, 2024. 2, 23

[54] Bowen Shi, Diane Brentari, Greg Shakhnarovich, and Karen Livescu. Open-domain sign language translation learned from online video. In European Conference on Computer Vision, pages 582–599. Springer, 2022. 22

[55] Tongkai Shi, Lianyu Hu, Fanhua Shang, Jichao Feng, Peidong Liu, and Wei Feng. Pose-guided fine-grained sign language video generation. In European Conference on Computer Vision, pages 392–409. Springer, 2024. 2

[56] Qingping Sun, Yanjun Wang, Ailing Zeng, Wanqi Yin, Chen Wei, Wenjia Wang, Haiyi Mei, Chi-Sing Leung, Ziwei Liu, Lei Yang, et al. Aios: All-in-one-stage expressive human pose and shape estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1834–1843, 2024. 2, 3, 13, 21

[57] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks, 2017. 5

[58] Sihan Tan, Taro Miyazaki, and Kazuhiro Nakadai. Multilingual gloss-free sign language translation: Towards building a sign language foundation model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 553–561, 2025. 1

[59] Shengeng Tang, Jiayi He, Dan Guo, Yanyan Wei, Feng Li, and Richang Hong. Sign-idd: Iconicity disentangled diffusion for sign language production, 2024. 22

[60] Laia Tarres, Gerard I G´ allego, Amanda Duarte, Jordi Torres,´ and Xavier Giro-i Nieto. Sign language translation from in-´ structional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 5625–5635, 2023. 2, 6, 22

[61] Levente Tempfli, Stephan Huber, Oscar Koller, and Amanda Duarte. How2sign-synth3d: Markerless holistic sign language performance capture and synthetic data for dense landmark tracking. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 3501–3509, 2026. 21, 22

[62] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2

[63] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. 2

[64] Ulrich Von Agris and Karl-Friedrich Kraiss. Signum database: Video corpus for signer-independent continuous sign language recognition. In 4th Workshop on the Representation and Processing of Sign Languages: Corpora and Sign Language Technologies, pages 243–246, 2010. 2

[65] Ulrich Von Agris, Moritz Knorr, and Karl-Friedrich Kraiss. The significance of facial features for automatic sign language recognition. In 2008 8th IEEE international confer ence on automatic face & gesture recognition, pages 1–6. IEEE, 2008. 2

[66] Weijie Wang, Jichao Zhang, Chang Liu, Xia Li, Xingqian Xu, Humphrey Shi, Nicu Sebe, and Bruno Lepri. Uvmap-id: A controllable and personalized uv map generative model. In ACM MM, pages 10725–10734, 2024. 21

[67] Ryan Wong, Necati Cihan Camgoz, and Richard Bowden. Sign2gpt: Leveraging large language models for gloss-free sign language translation. arXiv preprint arXiv:2405.04164, 2024. 1, 2

[68] Huijie Yao, Wengang Zhou, Hao Feng, Hezhen Hu, Hao Zhou, and Houqiang Li. Sign language translation with iterative prototype. In Proceedings of the IEEE/CVF Interna tional Conference on Computer Vision, pages 15592–15601, 2023.

[69] Aoxiong Yin, Zhou Zhao, Weike Jin, Meng Zhang, Xingshan Zeng, and Xiaofei He. Mlslt: Towards multilingual sign language translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5109–5119, 2022. 2

[70] Aoxiong Yin, Haoyuan Li, Kai Shen, Siliang Tang, and Yueting Zhuang. T2s-gpt: Dynamic vector quantization for autoregressive sign language production from text. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3345–3356, 2024. 2, 7

[71] Kayo Yin, Amit Moryossef, Julie Hochgesang, Yoav Goldberg, and Malihe Alikhani. Including signed languages in natural language processing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7347– 7360, 2021. 1

[72] Zhengdi Yu, Shaoli Huang, Chen Fang, Toby P Breckon, and Jue Wang. Acr: Attention collaboration-based regressor for arbitrary two-hand reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12955–12964, 2023. 2

[73] Zhengdi Yu, Shaoli Huang, Yongkang Cheng, and Tolga Birdal. Signavatars: A large-scale 3d sign language holistic motion dataset and benchmark. In European Conference on Computer Vision, pages 1–19. Springer, 2024. 1, 2, 22

[74] Morteza Zahedi, Daniel Keysers, Thomas Deselaers, and Hermann Ney. Combination of tangent distance and an image distortion model for appearance-based sign language recognition. In Joint Pattern Recognition Symposium, pages 401–408. Springer, 2005. 2

[75] Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan Sun, and Yebin Liu. Pymaf-x: Towards well-aligned full-body model regression from monocular images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12287–12303, 2023. 2

[76] Han Zhang, Rotem Shalev-Arkushin, Vasileios Baltatzis, Connor Gillis, Gierad Laput, Raja Kushalnagar, Lorna C Quandt, Leah Findlater, Abdelkareem Bedri, and Colin Lea. Towards ai-driven sign language generation with nonmanual markers. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–26, 2025. 2, 5, 6, 17, 18

[77] Hao Zhou, Wengang Zhou, Yun Zhou, and Houqiang Li. Spatial-temporal multi-cue network for sign language recognition and translation. IEEE transactions on multimedia, 24: 768–779, 2021. 2

[78] Ronglai Zuo, Fangyun Wei, Zenggui Chen, Brian Mak, Jiaolong Yang, and Xin Tong. A simple baseline for spoken language to sign language translation with 3d avatars, 2024. 2

[79] Ronglai Zuo, Rolandos Alexandros Potamias, Evangelos Ververas, Jiankang Deng, and Stefanos Zafeiriou. Signs as tokens: A retrieval-enhanced multilingual sign language generator. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23806–23816, 2025. 1, 2, 4, 5, 6, 7, 8, 16, 20, 22

## A. Cross-View Motion Curation

This appendix records implementation details that are compressed in the main paper. The material here explains the engineering choices behind the multi-view motion curation, tokenizer, and generation modules to ensure full reproducibility.

## A.1. Reliability-Guided Per-View Fusion

We define two camera views $v \in \mathcal { V } _ { \mathrm { v i e w } } = \{ \mathrm { f } , \mathrm { s } \}$ for the frontal and side streams, and $h \in \{ l , r \}$ denotes the hand index. AiOS [56] yields a stable whole-body sequence $\Theta _ { t } ^ { A , v }$ , whereas WiLoR [43] predicts detailed hand parameters, namely $\theta _ { t , { \mathrm { w } } , h } ^ { W , v }$ for the wrist and $\theta _ { t , \mathrm { f i n } , h } ^ { W , v }$ for the fingers (expressed in axis-angle form on $S O ( 3 ) )$

To operationalize the reliability-guided corruption mask $m _ { t , h } ^ { v }$ introduced in the main paper, we formally derive its constituent evaluation metrics. First, the per-view global wrist rotations are extracted via:

$$
R _ { t , \mathrm { w } , h } ^ { A , v } = \mathrm { F K } _ { \mathrm { w } , h } ( \Theta _ { t } ^ { A , v } ) , \qquad R _ { t , \mathrm { w } , h } ^ { W , v } = \mathrm { E x p } ( \theta _ { t , \mathrm { w } , h } ^ { W , v } ) ,\tag{6}
$$

where $\mathrm { F K } _ { \mathrm { w } , h }$ represents forward kinematics applied to the AiOS tree, and Exp(·) maps axis-angles to rotation matrices. The hand presence validity flag is defined via an $L _ { 1 }$ -norm threshold as $\rho _ { t , h } ^ { v } = \mathbf { 1 } [ \lVert \dot { { \boldsymbol { \theta } } } _ { t , \mathrm { w } , h } ^ { W , v } \rVert _ { 1 } > \varepsilon ]$ . The orientation discrepancy and the frame-to-frame angular velocities are quantified using the standard geodesic distance on $S O ( 3 ) , d _ { \mathrm { g e o } } ( R _ { 1 } , R _ { 2 } )$ = arccos $\left( \operatorname { t r } ( R _ { 1 } ^ { \top } R _ { 2 } ) - 1 \right) / 2 )$ yielding $\begin{array} { r c l } { \omega _ { t , h } ^ { W , v } } & { = } & { d _ { \mathrm { g e o } } ( R _ { t - 1 , \mathrm { w } , h } ^ { W , v } , R _ { t , \mathrm { w } , h } ^ { W , v } ) } \end{array}$ and $\begin{array} { r l } { \omega _ { t , h } ^ { A , v } } & { { } = } \end{array}$ $d _ { \mathrm { g e o } } ( R _ { t - 1 , \mathrm { w } , h } ^ { A , v } , R _ { t , \mathrm { w } , h } ^ { A , v } )$ These individual cues are integrated into the final per-view corruption mask:

$$
\begin{array} { r l } & { m _ { t , h } ^ { v } = \mathbf { 1 } \big [ \neg \rho _ { t , h } ^ { v } \lor } \\ & { \qquad d _ { \operatorname* { g e o } } ( R _ { t , \mathrm { w } , h } ^ { W , v } , R _ { t , \mathrm { w } , h } ^ { A , v } ) > \tau _ { \mathrm { o r i } } \lor } \\ & { \qquad ( \omega _ { t , h } ^ { W , v } > \tau _ { W } \land \omega _ { t , h } ^ { A , v } < \tau _ { A } ) \big ] . } \end{array}\tag{7}
$$

We set $\varepsilon = 1 0 ^ { - 6 } , \tau _ { \mathrm { o r i } } = 1 2 0 ^ { \circ } , \tau _ { W } = 6 0 ^ { \circ }$ per frame, and $\tau _ { A } = 1 5 ^ { \circ }$ per frame; first-frame velocities are zero. If a hand has no valid WiLoR frame, its AiOS hand parameters are retained. Clips shorter than eight frames are not processed.

To safeguard the transitions of short-lived track failures, the initial mask is temporally dilated by one frame via $\begin{array} { r } { \tilde { m } _ { t , h } ^ { v } = \operatorname* { m a x } _ { | k - t | \leq 1 } m _ { k , h } ^ { v } } \end{array}$

For each corrupted interval [a, b] flagged by $\tilde { m } _ { t , h } ^ { v } = 1$ let $ { a ^ { - } } <  { a }$ and $b ^ { + } ~ > ~ b$ be the nearest valid boundary frames. The intermediate wrist and finger joint rotations are repaired using spherical linear interpolation: Let $R _ { t , h , j } ^ { v }$ denote the pre-repair rotation (global for wrist, local for fingers) of hand joint j for hand h in view v.

$$
\hat { R } _ { t , h , j } ^ { v } = \mathrm { S l e r p } \left( R _ { a ^ { - } , h , j } ^ { v } , R _ { b ^ { + } , h , j } ^ { v } ; \frac { t - a ^ { - } } { b ^ { + } - a ^ { - } } \right) , \quad t \in [ a , b ] ,\tag{8}
$$

where $j$ indexes the structural joints within the hand kinematic chain. If an interval touches the sequence boundaries, we fall back to nearest-valid forward or backward constant propagation. Concurrently, when 3D hand keypoints $X _ { t , h , j } ^ { v }$ are available, their coordinates are repaired via standard Euclidean interpolation: $\hat { X } _ { t , h , j } ^ { v } = ( 1 - \alpha _ { t } ) X _ { a ^ { - } , h , j } ^ { v } +$ $\alpha _ { t } X _ { b ^ { + } , h , j } ^ { v }$ with $\alpha _ { t } = ( t - a ^ { - } ) / ( b ^ { + } - a ^ { - } )$

To re-integrate the repaired motions, the global wrist rotations $\hat { R } _ { t , { \mathrm { w } } , h } ^ { v }$ are mapped back into the local coordinate system relative to the wrist parent joint $\mathrm { p a } ( \mathrm { w } , h )$ in the stable AiOS scaffold: $\hat { R } _ { t , \mathrm { w - l o c a l } , h } ^ { v } = ( R _ { t , \mathrm { p a } ( \mathrm { w } , h ) } ^ { A , v } ) ^ { \top } \hat { R } _ { t , \mathrm { w } , h } ^ { v } .$ The corresponding local axis-angle block $\widehat { \theta } _ { t , { \mathrm { w } } , h } ^ { v }$ is recovered via the logarithmic map Log( $\hat { R } _ { t , \mathrm { w - l o c a l } , h } ^ { v } )$ , while the repaired finger rotations are already local and directly yield $\hat { \theta } _ { t , \mathrm { f i n } , h } ^ { v }$ Finally, the complete, unified per-view SMPL-X [41] sequence $\Theta _ { t } ^ { v }$ is constructed by seamlessly blending the stable body trunk with the repaired hand parameters:

$$
\Theta _ { t } ^ { v } = \Theta _ { t , \tiny \backslash \mathrm { h a n d } } ^ { A , v } \oplus \hat { \theta } _ { t , \tiny { \mathrm { w } } } ^ { v } \oplus \hat { \theta } _ { t , \tiny \mathrm { f i n } } ^ { v } ,\tag{9}
$$

where $\Theta _ { t , \backslash \mathrm { h a n d } } ^ { A , v }$ denotes the baseline AiOS sequence without the hand blocks (retaining body pose, shape, and camera parameters, where cam trans stores the actual camera translation while trans remains a zero placeholder), and ⊕ indicates block-wise kinematic replacement.

## A.2. Alignment, Synchronization, Non-Sign Frames

Before the side view can be used as reliable verification evidence for the frontal stream, the two sequences must be spatially aligned and temporally synchronized. To compare the streams, we use the retained fused tracking coordinates $J _ { t } ^ { v } \in \mathbb { R } ^ { N \times 3 }$ ; these are not recomputed by forward kinematics from the final repaired pose.

Because the side-view global orientation and translation are recorded in a different camera coordinate system, we first apply a robust spatial alignment. We compute a global root rotation correction by projecting the average framewise root discrepancy, $\Delta R _ { t } = \bar { R _ { t , \mathrm { r o o t } } ^ { \bar { \mathrm { f } } } ( \bar { R } _ { t , \mathrm { r o o t } } ^ { \mathrm { s } } ) ^ { \top } }$ , onto $S O ( 3 )$ followed by a global translation shift to align the sequence centroids. Let $\tilde { J } _ { t } ^ { \mathrm { s } }$ denote the side-view joints after this coarse rigid alignment.

With spatial alignment established, we perform temporal synchronization. As defined in the main text, we search for the optimal temporal offset $\delta ^ { \star }$ that minimizes the Mean Per Joint Position Error (MPJPE) over a predefined temporal search window D:

$$
\delta ^ { \star } = \arg \operatorname* { m i n } _ { \delta \in \mathcal { D } } \frac { 1 } { | \mathcal { T } _ { \delta } | | \mathcal { B } | } \sum _ { t \in \mathcal { Z } _ { \delta } } \sum _ { j \in \mathcal { B } } \left\| J _ { t , j } ^ { \mathrm { f } } - \tilde { J } _ { t - \delta , j } ^ { \mathrm { s } } \right\| _ { 2 } .\tag{10}
$$

Here, $\mathcal { T } _ { \delta }$ denotes the set of valid overlapping frame indices under offset δ, where a positive δ maps synchronized sideview time t to source time $t - \delta$ . Candidate offsets are scored strictly on $\mathcal { T } _ { \delta }$ (with edge padding applied only when constructing the final synchronized sequence), and the joint subset B is restricted to the first 22 SMPL-X body joints (indices 0–21). We intentionally exclude hands from this synchronization search because hand estimates prior to refinement are highly susceptible to blur and occlusion; including them would introduce severe noise into the temporal alignment.

Even after synchronization, continuous sign recordings contain non-lexical segments $( e . g .$ , preparation, ending motion, and pauses) that should be removed. As introduced in the main text, we identify active signing frames using a combination of mouth activity, finger activity, and wrist elevation.

Non-sign activity detection and temporal editing. We compute three frontal activity cues over the SMPL-X joint groups $\mathcal { G } _ { \mathrm { m o u t h } } ~ = ~ \{ 2 2 \} ~ ( \mathrm { j a w } ) , ~ \mathcal { G } _ { \mathrm { w r i s t } } ~ = ~ \{ 2 0 , 2 1 \}$ , and $\mathcal { G } _ { \mathrm { f i n g e r } } = \{ 2 5 , . . . , 5 4 \}$ For each group g, the raw motion heat $h _ { t } ^ { g }$ is defined as the median frame-to-frame $\mathrm { S O ( 3 ) }$ geodesic distance across its joints (with $h _ { 0 } ^ { g } \ = \ 0 )$ smoothed by a three-frame moving average with edge padding. Let $m _ { g }$ denote the temporal median of $h _ { t } ^ { g }$ and $d _ { g } = \mathrm { m e d i a n } _ { t } | h _ { t } ^ { \bar { g } } - m _ { g } | + 1 0 ^ { - 8 }$ its median absolute deviation. We normalize each cue via

$$
c _ { t } ^ { g } = \mathrm { c l i p } \left( \frac { h _ { t } ^ { g } - ( m _ { g } - 0 . 5 d _ { g } ) } { 2 . 5 d _ { g } } , 0 , 1 \right) .\tag{11}
$$

The wrist and finger scores undergo additional inactivity gating. For the wrists, we infer the vertical axis direction from the median wrist–shoulder displacement; if both wrists drop below their respective elbows by a margin exceeding 0.25× the median shoulder width, we set $c _ { t } ^ { \mathrm { w r i s t } } = 0$ . Similarly, if the raw finger heat remains below $1 0 ^ { - 3 }$ for at least 18 consecutive frames, we set $c _ { t } ^ { \mathrm { f i n g e r } } = 0$ A frame is classified as active signing when all three cues exceed 0.30:

$$
a _ { t } = 1 { \Big [ } c _ { t } ^ { \mathrm { m o u t h } } \geq 0 . 3 0 \ \wedge \ c _ { t } ^ { \mathrm { f n g e r } } \geq 0 . 3 0 \ \wedge \ c _ { t } ^ { \mathrm { w r i s t } } \geq 0 . 3 0 { \Big ] }\tag{12}
$$

To preserve natural signing holds and transitions, we do not immediately truncate every inactive run $( a _ { t } ~ = ~ 0 )$ For a low-activity run starting at frame s, pruning begins only after a 23-frame buffer $( s + 2 3 )$ , and terminates up to 8 frames prior to the resumption of full signing activity $( a _ { t } = 1 )$ . Candidate inactive intervals shorter than 12 frames $( > 8 + 3 )$ are retained, while overlapping or adjacent intervals are merged. As illustrated in Fig. 8, the detected non-signing intervals are identically excised from both views, and each cut is bridged by a 4-frame synthetic transition (using Slerp for joint rotations and linear interpolation for translational/shape parameters), with sourceframe indices and synthetic-bridge masks tracked separately. Note that this kinematics-driven pruning removes prolonged idle pauses within a clip rather than recalibrating external subtitle alignments.

## A.3. Cross-View Body and Finger Repair

With the streams synchronized and non-signing frames removed, we cross-verify the sequences to repair abrupt tracking failures. Hand configurations are cross-checked between the two views and their temporal neighborhoods; the final training sequence retains the frontal body reconstruction with cross-view-refined hands.

For body joints, we operate under the assumption that true, fast signing kinematics must be consistently visible across cameras. To quantify anomalous spatial jitter, we extract the structural joint rotations $R _ { t , j } ^ { v }$ from the fused sequence $\Theta _ { t } ^ { v }$ and compute the frame-to-frame geodesic angular velocity $\alpha _ { t , j } ^ { v } \ = \ d _ { \mathrm { g e o } } ( R _ { t , j } ^ { v } , R _ { t - 1 , j } ^ { v } )$ . A short temporal spike is flagged as a failure only when it manifests abruptly in the target view but is contradicted by stable, low-jitter evidence in the opposite view v¯:

$$
b _ { t , j } ^ { v } = { \bf 1 } \left[ \alpha _ { t , j } ^ { v } > \tau _ { \mathrm { h i } } \land \alpha _ { t , j } ^ { \bar { v } } \leq \tau _ { \mathrm { l o } } \right] .\tag{13}
$$

When $b _ { t , j } ^ { v } = 1$ , the corrupted frames in the target view are inpainted via spherical linear interpolation using the nearest valid boundary frames, identical to the boundary-aware repair in Section A.1. The dual-threshold design $( \tau _ { \mathrm { h i } }$ and $\tau _ { \mathrm { l o } } )$ prevents the suppression of legitimate rapid signs.

We set $( \tau _ { \mathrm { h i } } , \tau _ { \mathrm { l o } } ) = ( 8 0 ^ { \circ } , 3 0 ^ { \circ } )$ for SMPL-X body joints $1 , \ldots , 2 1$ while leaving the root joint unchanged, and restrict inpainting to spike runs lasting at most two frames. Following spike repair, each rotation-vector component is denoised via a two-level coif3 wavelet decomposition with symmetric boundary extension. We preserve the approximation coefficients and apply soft thresholding to both detail levels using the universal threshold $\hat { \sigma } \sqrt { 2 \log ( \operatorname* { m a x } ( T , 2 ) ) }$ , where $\hat { \sigma } = \mathrm { m e d i a n } ( \vert d \vert ) / 0 . 6 7 4 5 +$ $1 0 ^ { - 8 }$ is estimated from the finest-scale detail coefficients d.

For hand articulation, the most critical residual error is the missing finger extension (e.g., a finger that should be pointing or extended collapses into a closed hand due to severe motion blur in a single view). To execute the targeted repair mechanism introduced in the main text, we first isolate the relevant degrees of freedom.

![](images/0178279d3c414e2e4b93eaebf2d4671e6a3806222f87dbd91c0f423ac78be51b.jpg)  
Figure 8. Timeline illustration of heuristic non-sign pruning and motion-blend transitions. In this frontal sequence (195 source frames), removing two detected idle intervals (23 and 16 frames) leaves 156 active source frames, and inserting two 4-frame synthetic bridges yields a 164-frame continuous output sequence.

## A.3.1. Geometric Construction of Finger Coordinate Frames

To isolate anatomical finger flexion from arbitrary local joint orientations, we construct fixed, joint-specific orthonormal coordinate frames directly from the neutral SMPL-X rest template. Let $V _ { 0 }$ denote the zero-shape template vertices and $J _ { 0 } = \mathcal { I } V _ { 0 }$ the rest joints obtained via the joint regressor $\mathcal { T }$ . For each hand, we fit a reference plane via SVD to the wrist and the four non-thumb MCP joints, orienting its unit normal n downward along the body via $- \operatorname { u n i t } ( J _ { 0 , 1 2 } - J _ { 0 , 0 } )$ to establish a consistent palmar convention.

At each finger joint $j ,$ let $y _ { j }$ be the unit vector pointing toward its distal child joint (or the corresponding fingertip template vertex for terminal phalanges). We construct the local orthonormal frame $T _ { j }$ as:

$$
\begin{array} { r l } & { p _ { j } = \mathrm { u n i t } \big ( n - y _ { j } ( n ^ { \top } y _ { j } ) \big ) , } \\ & { x _ { j } = \mathrm { u n i t } ( y _ { j } \times p _ { j } ) , \qquad z _ { j } = \mathrm { u n i t } ( x _ { j } \times y _ { j } ) , } \\ & { T _ { j } = \big [ x _ { j } y _ { j } z _ { j } \big ] \in \mathrm { S O } ( 3 ) , } \end{array}\tag{14}
$$

where the columns of $T _ { j }$ define the adjusted-frame axes in the SMPL-X rest coordinate system. A local joint rotation matrix $R _ { j } ^ { \mathrm { l o c a l } }$ transforms between the native and adjusted bases via similarity conjugation:

$$
R _ { j } ^ { \mathrm { a d j } } = T _ { j } ^ { \top } R _ { j } ^ { \mathrm { l o c a l } } T _ { j } , \qquad R _ { j } ^ { \mathrm { l o c a l } } = T _ { j } R _ { j } ^ { \mathrm { a d j } } T _ { j } ^ { \top } .\tag{15}
$$

We decompose $R _ { i } ^ { \mathrm { a d j } }$ into intrinsic Euler angles as $R _ { i } ^ { \mathrm { a d j } } =$ $R _ { z } ( \gamma ) R _ { y } ( \beta ) R _ { x } ( \check { \alpha } )$ , where the X-axis angle α (denoted below as $x _ { k , i }$ for finger k, joint i) captures primary finger flexion: a positive rotation bends the distal segment toward the palmar projection $p _ { j }$ . Placing flexion on the outer X-axis avoids the middle-axis (β) gimbal lock singularity during standard bending motions.

SMPL-X indexing and fingertips. In SMPL-X, local hand pose blocks start at joint indices 25 (left) and 40 (right), ordered as index, middle, pinky, ring, and thumb (three joints per finger). Our four-finger repair operates on the non-thumb digits in anatomical order (index, middle, ring, pinky), corresponding to block offsets (0, 3, 9, 6). Each hand and joint utilizes its own dedicated frame $T _ { j }$ . In native model finger order, the terminal fingertip vertex IDs used to define distal directions are (4933, 5058, 5286, 5169, 5361) for the left hand and (7669, 7794, 8022, 7905, 8079) for the right hand.

## A.3.2. Cross-View Finger Extension Repair

Single-view hand estimators frequently miss isolated finger extensions under motion blur or self-occlusion. To recover these cues across views, we evaluate the four non-thumb fingers $( k \in$ {index, middle, ring, pinky}). For each finger $k ,$ we extract the adjusted flexion angles $x _ { k , 1 } , x _ { k , 2 }$ from its first two joints (MCP and PIP) and map them to a normalized straightness score $S _ { k } \in [ 0 , 1 ]$

$$
S _ { k } = 1 - \mathrm { c l i p } \left( \sum _ { i = 1 } ^ { 2 } a _ { i } \frac { \left| x _ { k , i } \right| } { u _ { k , i } } , 0 , 1 \right) ,\tag{16}
$$

where $a = ( 0 . 7 , 0 . 3 )$ weights the proximal and intermediate joints, $u _ { k , 1 } \in [ 1 . 7 5 , 1 . 8 0 , 1 . 8 0 , 1 . 7 0 ]$ rad sets the fingerspecific maximum flexion bounds for the first joint, and $u _ { k , 2 } = 1 . 7 5$ rad across all four fingers.

Isolation and collapse criteria. For a given source view, we first identify the candidate finger with the largest straightness excess over the remaining three digits:

$$
k ^ { \star } = \arg \operatorname* { m a x } _ { k } E _ { k } , \qquad E _ { k } = S _ { k } - \mu ( S _ { \backslash k } ) .
$$

The winning digit $k ^ { \star }$ is confirmed as a confident isolated extension if $S _ { k ^ { \star } } > 1 . 1 5 \mu ( S _ { \backslash k ^ { \star } } )$ and the unbiased sample variance of the other three fingers satisfies $\sigma ^ { 2 } ( S _ { \backslash k ^ { \star } } ) < 0 . 2$ If confirmed, a cross-view repair is triggered in the destination view whenever the corresponding finger exhibits severe flexion collapse, defined as $S _ { k ^ { \star } } ^ { \mathrm { d s t } } < 0 . 6 5 S _ { k ^ { \star } } ^ { \mathrm { s r c } }$

Bounded angular update. Given a triggered repair, we define a target normalized flexion reduction $d = 1 . 2 E _ { k ^ { \star } } ^ { \mathrm { s r c } }$ and distribute it across all three joints $j \in \{ 1 , 2 , 3 \}$ of digit $k ^ { \star }$ using weights $q = ( 0 . 7 , 0 . 7 , 0 . 5 )$

$$
\begin{array} { r l } & { \lambda = \frac { d } { \sum _ { i = 1 } ^ { 2 } a _ { i } q _ { i } / u _ { k ^ { \star } , i } } , } \\ & { x _ { k ^ { \star } , j } ^ { \mathrm { d s t } \prime } = \mathrm { s i g n } ( x _ { k ^ { \star } , j } ^ { \mathrm { d s t } } ) \operatorname* { m a x } \big ( | x _ { k ^ { \star } , j } ^ { \mathrm { d s t } } | - \lambda q _ { j } , 0 \big ) . } \end{array}\tag{17}
$$

This clamping ensures the flexion reduction never crosses zero (preventing unnatural hyperextension overshoot) while preserving the non-flexion Euler components $( \beta , \gamma )$ before mapping back to the native basis via Equation (15).

Two-pass execution. Repair executes in two stages: (i) Cross-view pass: we evaluate the frontal view as the source first; if it triggers a repair on a given hand and frame, the reverse (side-to-front) check is skipped for that instance; otherwise, the side view is checked to repair the frontal view. (ii) Temporal propagation pass: each repaired event is propagated to its immediate neighbors (t − 1 and t + 1) if the adjacent frame also fails the 0.65 collapse threshold and its remaining three fingers remain consistent with the source event $( \Delta \mu < 0 . 1 5$ and $\Delta \sigma ^ { 2 } < 0 . 2 )$

## B. Sign-aware ST-VQVAE

## B.1. Motion Representation and Token Layout

The tokenizer uses the refined SMPL-X [41] rotation representation from the curation pipeline as both input and output. We use a selected 43-joint upper-body and hand skeleton, giving input $\boldsymbol { X } \in \mathbb { R } ^ { \bar { T } \times 4 3 \times 3 }$ in axis-angle form. After encoding and quantization, each reduced timestep produces three tokens:

$$
z _ { t } = \big ( z _ { t } ^ { ( b ) } , z _ { t } ^ { ( l ) } , z _ { t } ^ { ( r ) } \big ) , \qquad t = 1 , \ldots , T ^ { \prime } ,\tag{18}
$$

corresponding to body, left, and right hand, where $T ^ { \prime } =$ $T / 4$ after temporal downsampling. This preserves the three-stream interface used by SOKE [79]-style autoregressive generation while allowing cross-part information flow before the terminal split.

## B.2. Spatio-Temporal Hierarchy

Our tokenizer uses a three-level skeleton hierarchy. The input is a selected 43 SMPL-X [41] joint skeleton: joints 0:13 are body joints, joints 13:28 are left-hand joints, and joints 28:43 are right-hand joints. The pooling is not learnable. Each output node averages a predefined group of input joints.

After the input rotations X are projected into feature space, one pooling level takes a feature tensor $x \in$ $\mathbb { R } ^ { B \times T \times J _ { \mathrm { i n } } \times D }$ as input. One pooling level can be written as a fixed averaging matrix $W \in \mathbb { R } ^ { J _ { \mathrm { o u t } } \times J _ { \mathrm { i n } } }$

$$
W _ { o , i } = \left\{ 1 / | G _ { o } | , \begin{array} { l l } { i \in G _ { o } , } \\ { 0 , } \end{array} \right. \quad y _ { b , t , o , : } = \sum _ { i } W _ { o , i } x _ { b , t , i , : } .\tag{19}
$$

Table 6. Fixed skeleton hierarchy in ST-VQVAE. Each node is produced by average pooling over a fixed joint group. The first two levels also apply temporal average pooling with kernel size 2.
<table><tr><td></td><td>Level Size/Time</td><td>Output Nodes</td><td></td></tr><tr><td>0</td><td> $4 3  1 6$ </td><td>0: mean(0,1,2,3)</td><td>8: mean(19,20,21)</td></tr><tr><td></td><td> $T \to T / 2$ </td><td>1: mean(4,7,9,11)</td><td>9: mean(22,23,24)</td></tr><tr><td></td><td></td><td>2: mean(5,8,10,12)</td><td>10: mean(25,26,27)</td></tr><tr><td></td><td></td><td>3: mean(9,11,25)</td><td>11: mean(28,29,30)</td></tr><tr><td></td><td></td><td>4: mean(10,12,40)</td><td>12: mean(31,32,33)</td></tr><tr><td></td><td></td><td>5: mean(3,6)</td><td>13: mean(34,35,36)</td></tr><tr><td></td><td></td><td>6: mean(13,14,15)</td><td>14: mean(37,38,39)</td></tr><tr><td></td><td></td><td>7: mean(16,17,18)</td><td>15: mean(40,41,42)</td></tr><tr><td>1</td><td> $1 6  1 3$ </td><td>0: mean(prev 0,5)</td><td>7: prev 10</td></tr><tr><td></td><td> $T / 2  T / 4$ </td><td>1: mean(prev 1,3)</td><td>8: prev 11</td></tr><tr><td></td><td></td><td>2: mean(prev 2,4)</td><td>9: prev 12</td></tr><tr><td></td><td></td><td>3: prev 6</td><td>10: prev 13</td></tr><tr><td></td><td></td><td>4: prev 7</td><td>11: prev 14</td></tr><tr><td></td><td></td><td>5: prev 8</td><td>12: prev 15</td></tr><tr><td></td><td></td><td>6: prev 9</td><td></td></tr><tr><td>2</td><td></td><td></td><td></td></tr><tr><td></td><td> $1 3  3$  unchanged</td><td>body: mean(prev 0,1,2) left hand: mean(prev 3,4,5,6,7)</td><td></td></tr></table>

Here $G _ { o }$ is the joint group assigned to output node o. After the first two spatial pooling levels, we apply temporal average pooling with kernel size 2. The temporal length changes from T to $T / 2$ , and then to $T / 4$ . The last level only pools space and keeps the temporal length unchanged. The final latent skeleton has shape

$$
U \in \mathbb { R } ^ { B \times T / 4 \times 3 \times D } ,\tag{20}
$$

where the three spatial tokens are body, left hand, and right hand.

Table 6 gives the exact fixed groups.

This design keeps body and hand information in one encoder before the terminal split. At the same time, it gives separate terminal tokens for body, left hand, and right hand, which matches the group-wise quantization used below. The decoder uses the same fixed hierarchy in reverse: each pooled node is copied back to the joints in its assigned group, with temporal upsampling at the two levels that were temporally downsampled in the encoder.

We also tried other fixed hierarchies. The first type used smoother pooling, such as $4 3 \to 2 1 \to 9 \to 3$ . These variants compressed the hands earlier and gave worse results. In our setting, keeping the five finger groups separate until the last level is more important than making the spatial hierarchy smoother. The second type added cross-hand links, for example letting the left-hand branch see right-hand rotations during convolution or pooling. This also reduced reconstruction quality. We found this cross-part information more likely to act as noise for reconstruction, although it may be useful later in the generation model where the task is to predict coordinated two-hand motion.

## B.3. Group-Wise Quantization

We quantize the three terminal slots independently:

$$
\begin{array} { r } { \mathcal { C } ^ { ( p ) } = \{ e _ { k } ^ { ( p ) } \} _ { k = 1 } ^ { K _ { p } } , \qquad p \in \{ b , l , r \} , } \end{array}\tag{21}
$$

with codebook sizes:

$$
( K _ { b } , K _ { l } , K _ { r } ) = ( 9 6 , 1 9 2 , 1 9 2 ) .\tag{22}
$$

For each part,

$$
z _ { t } ^ { ( p ) } = \arg \operatorname* { m i n } _ { k \in \{ 1 , \ldots , K _ { p } \} } \left\| u _ { t } ^ { ( p ) } - e _ { k } ^ { ( p ) } \right\| _ { 2 } ^ { 2 } , \qquad \tilde { u } _ { t } ^ { ( p ) } = e _ { z _ { t } ^ { ( p ) } } ^ { ( p ) } .\tag{23}
$$

The hand codebooks are larger because sign motion contains more diverse local hand articulation than torso motion. The codebooks are trained with EMA [39] updates and code-usage regularization to reduce collapse.

## B.4. Training Objective

We expand the three loss terms used in the main-text tokenizer objective. The rotation term is a masked meansquared error over valid timesteps:

$$
\mathcal { L } _ { \mathrm { r o t } } = \frac { 1 } { | \Omega | J } \sum _ { t \in \Omega } \sum _ { j = 1 } ^ { J } \big \Vert \hat { x } _ { t , j } - x _ { t , j } \big \Vert _ { 2 } ^ { 2 } .\tag{24}
$$

For mesh supervision, both target and reconstruction are passed through SMPL-X:

$$
V _ { t } = \mathcal { M } ( X _ { t } ) , \qquad \hat { V } _ { t } = \mathcal { M } ( \hat { X } _ { t } ) .\tag{25}
$$

We compare selected upper-body and hand vertices with a SmoothL1 penalty:

$$
\mathcal { L } _ { \mathrm { m e s h } } = \frac { 1 } { \sum _ { t \in \Omega } \sum _ { q \in \mathcal { V } _ { \mathrm { m e s h } } } \omega _ { q } } \sum _ { t \in \Omega } \sum _ { q \in \mathcal { V } _ { \mathrm { m e s h } } } \omega _ { q } \rho \Bigl ( \hat { V } _ { t , q } - V _ { t , q } \Bigr )\tag{26}
$$

The weights $\omega _ { q }$ upweight hand vertices. Without this reweighting, the mesh loss is biased toward the body and arms: they occupy a larger physical scale, contain broad surface motion, and can dominate vertex-distance averages. This improves global/body reconstruction but can reduce the precision of finger articulation. Since manual articulation carries much of the sign meaning, the final tokenizer deliberately trades a small amount of body/global reconstruction accuracy for stronger hand fidelity.

## B.5. Tokenizer Training Details

We optimize the ST-VQVAE for 150 epochs using AdamW with a learning rate of $2 \times 1 0 ^ { - 4 }$ and a batch size of 32. Once trained, the tokenizer weights are frozen, and the model is used to encode each How2Sign-MV3D sequence into synchronized body, left-hand, and right-hand discrete token streams for downstream generator training and retrieval.

## C. Condition and Generation

## C.1. Pseudo-Gloss Construction

How2Sign [12] provides English sentences but not gold gloss annotations. We therefore generate pseudo-glosses offline and use them as an intermediate source representation. The pseudo-gloss is not treated as a new annotation benchmark; its purpose is to reduce the gap between natural-language word order and sign motion order before autoregressive token generation.

For few-shot prompting gloss translation, we group 20 American Sign Language English-to-gloss translation examples and 20 How2Sign English sentences into a task batch, and use GPT-5.2 API with a prompt that follows the appendix prompt style of Zhang et al. [76], which reports that few-shot prompting with GPT models is effective for pseudo-gloss construction. To better align the output glosses with our ASLCitizen [10] retrieval dictionary, our modification is to add the ASLCitizen vocabulary sheet as an allowed lexical reference so that generated pseudoglosses better match the retrieval dictionary. Although including only a single sentence per task batch reduces the dispersion of GPT-5.2’s attention and improves translation quality, the token consumption for processing the dataset increases significantly.

The resulting pseudo-gloss sequence is paired with the original English sentence during autoregressive model training.

To support reproducibility, the released code will include the prompt template, vocabulary list, and generated pseudoglosses for the experiments.

## C.1.1. External Reference Evaluation

Table 7 reports the three-model 20-shot ASLG-PC12 comparison. GPT-4o is included in light of the GPT-4o results reported by Zhang et al. [76]; their dataset and evaluation settings are not identical to ours. Among these three evaluated configurations, GPT-5.2 has the highest reference agreement, consistent with our production choice. This does not assert superiority over later models or supervised adaptation. These external evaluations use one target per request and an ASLG-style prompt, rather than replaying the vocabulary-augmented production batches.

## C.1.2. External Data and Evaluation Settings

The local ASLG-PC12 preparation starts from 87,710 rows, removes 6,619 duplicate English/gloss pairs, and splits the remaining 81,091 pairs with seed 42 into 64,872 training, 8,109 development, and 8,110 test records (80/10/10). We evaluate a fixed 200-item test subset with identical identifiers, English sentences, and references across the displayed systems. This is a local split, not a claim to use another paper’s official split. Twenty-shot or 80-shot denotes 20 or 80 English/reference-gloss demonstration pairs in each inference context, not 20 or 80 test targets. The later 80-shot systems use the saved 80-example training demonstration file (selection seed 42); each request has one test target. Test references are never included in inference prompts. The 20- shot comparisons share the same demonstrations within that experiment.

Table 7. ASLG-PC12 20-shot reference agreement on the same 200 test sentences. The prompting approach is motivated by Zhang et al. [76]; scores are our local evaluations, not a direct reproduction of their reported scores. BLEU-4 is on a 0–1 scale.
<table><tr><td>Model</td><td>BLEU-4↑</td></tr><tr><td>GPT-40</td><td>.5789</td></tr><tr><td>GPT-5.2</td><td>.6589</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>.5553</td></tr></table>

The ASLG-style prompt asks for a single gloss line matching the reference notation, including punctuation, BE, DESC-, and X-I when appropriate. The general-gloss prompt instead follows generic ASL-gloss conventions. This distinction is material: an annotation-style prompt is optimized for reference agreement, not an independent linguistic judgment. API sampling uses temperature zero; local Qwen inference uses greedy decoding with a 128-token generation cap. The saved API batch setup uses a 256- token output budget, while the GPT-5.2 20-shot configuration records a 512-token budget and low reasoning. Different prompt, output-budget, and supervision settings are explicitly identified rather than treated as a matched-compute ranking.

BLEU-4 uses add-one smoothing on aggregate clipped 1–4-gram precisions and the evaluator’s brevity penalty, on a 0–1 scale. This is the local implementation, not standard SacreBLEU. Higher scores indicate closer agreement with the dataset’s reference distribution; they do not directly measure linguistic accuracy, natural signing, or downstream motion quality.

## C.1.3. Additional Few-Shot and Supervised Adaptation Study

To further examine how scaling in-context examples and supervised fine-tuning (SFT) affect gloss agreement, we conduct an expanded benchmark on the 200-item ASLG-PC12 test set (Table 8). Because these runs use expanded supervision budgets (80-shot prompting or 5,000-sample QLoRA) and newer model checkpoints, we report them separately from the 20-shot comparison while keeping our main GPT-5.2 annotations and downstream motion experiments fixed. Table 8 includes only configurations that returned valid outputs for all 200 test items; two thinking-enabled DeepSeek runs that timed out or returned empty strings on subsets (177/200 and 135/200 completions) were excluded to ensure a uniform evaluation denominator.

QLoRA training setup. Both fine-tuned models are trained on 5,000 training and 500 development pairs from ASLG-PC12 (verified to have zero exact normalized-text overlap with the 200 test items). Base models are loaded in 4-bit NF4 with double quantization and BF16 compute, applying LoRA adapters $( r \quad = \quad 1 6 , \alpha \quad = \quad 3 2 ,$ dropout 0.05) to all attention $( \mathrm { q } / \mathrm { k } / \mathrm { v } / \mathrm { o } _ { - \mathrm { p } \Sigma \mathrm { O } } \dot { ] } )$ and MLP (gate/up/down proj) linear layers. Each sample is formatted as a zero-shot system/user prompt with the crossentropy loss masked strictly to the assistant gloss completion (max length 512). We train for 1 epoch (313 steps) using AdamW (adamw torch, $\mathrm { l r } = 2 \times 1 0 ^ { - 4 }$ , linear decay, effective batch size 16, seed 42) and evaluate the final checkpoint directly without few-shot demonstrations.

Discussion. On the same Qwen3.5-27B backbone, indomain QLoRA fine-tuning achieves substantially higher ASLG-PC12 reference agreement (.9614 BLEU-4) than 80- shot prompting (.7599). While Zhang et al. observed fewshot prompting outperforming fine-tuning under smaller data budgets on ASLLRP, our result shows that given 5,000 training pairs from a rule-based synthetic corpus like ASLG-PC12, SFT effectively internalizes corpus-specific mapping rules. However, high agreement on synthetic ASLG-PC12 glosses does not automatically translate to superior out-of-domain conditioning on continuous natural signing in How2Sign, where our frozen GPT-5.2 pipeline remains the production backbone.

## C.1.4. How2Sign Lexical Coverage

We audit the generated How2Sign pseudo-gloss sequences against the 2,300-sign ASLCitizen vocabulary (Table 9). Exact coverage measures strict token-level matches, whereas Loose coverage strips peripheral punctuation and hyphens after uppercasing. Rows with match denotes the proportion of sentences containing at least one matched dictionary sign.

All 35,263 sentences yield well-formed, single-line outputs without empty strings. Across the full dataset, 59.55% of generated tokens match the dictionary under loose matching, and 98.03% of sentences trigger at least one retrieved sign prompt. Unmatched tokens primarily stem from open-vocabulary entities, including fingerspelling markers or single-letter tokens (present in 2,817 sentences), numeric digits (1,741 sentences), proper nouns, technical terms, and compound variants (plus 33 rows containing residual lowercase characters).

## C.1.5. Reference Disagreements and Evaluation Nuances

Standard n-gram metrics such as BLEU-4 quantify surface alignment against a specific corpus convention rather than absolute linguistic validity. As illustrated by the qualitative NCSLGR comparisons in Table 10, reference glosses are sometimes highly compressed (e.g., omitting subject pronouns or path modifiers that the LLM explicitly predicts), whereas LLM predictions can occasionally introduce unsupported grammatical markers $( e . g \cdot$ ., LEARN+AGENT instead of TEACH+AGENT, or trailing discourse particles). Because natural sign language allows multiple valid realizations depending on visual context and annotation guidelines, text-only reference comparisons have inherent limits. Ultimately, because How2Sign lacks ground-truth gloss annotations, we treat LLM-generated pseudo-glosses as structured semantic intermediates for motion conditioning rather than formal linguistic transcriptions.

Table 8. Expanded ASLG-PC12 comparison on the 200-item test set. All listed configurations achieved 100% completion across the test set. Fine-tuned (QLoRA) models leverage 5,000 in-domain training pairs, whereas API and prompting baselines operate in an 80-shot in-context regime.
<table><tr><td>Model</td><td>Method BLEU-4↑</td></tr><tr><td>GPT-5.4 (2026-03-05)</td><td>80-shot, API batch .7709</td></tr><tr><td>GPT-5.5 (2026-04-23)</td><td>80-shot, API batch .8077</td></tr><tr><td>DeepSeek V4 Flash</td><td>80-shot, thinking disabled .7411</td></tr><tr><td>DeepSeek V4 Pro</td><td>80-shot, thinking disabled .8037</td></tr><tr><td>Qwen3.5-27B</td><td>80-shot, general gloss prompt .7143</td></tr><tr><td>Qwen3.5-27B</td><td>80-shot, ASLG-style prompt .7599</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>QLoRA (5k SFT, 313 steps) .9450</td></tr><tr><td>Qwen3.5-27B</td><td>QLoRA (5k SFT, 313 steps) .9614</td></tr></table>

Table 9. How2Sign pseudo-gloss formatting and ASLCitizen lexical coverage audit.
<table><tr><td>Split</td><td>Rows</td><td>Empty</td><td>Multiline</td><td> $\operatorname { A v g } .$  tokens</td><td>Exact coverage</td><td>Loose coverage</td><td>Rows with match</td></tr><tr><td>Train</td><td>31,165</td><td>0</td><td>0</td><td>11.90</td><td>59.22%</td><td>59.84%</td><td>98.10%</td></tr><tr><td>Val</td><td>1,741</td><td>0</td><td>0</td><td>12.00</td><td>57.77%</td><td>58.39%</td><td>97.30%</td></tr><tr><td>Test</td><td>2,357</td><td>0</td><td>0</td><td>12.17</td><td>56.09%</td><td>56.68%</td><td>97.62%</td></tr><tr><td>All</td><td>35,263</td><td>0</td><td>0</td><td>11.92</td><td>58.94%</td><td>59.55%</td><td>98.03%</td></tr></table>

## C.2. Gloss Adapter

The gloss adapter maps the pseudo-gloss token sequence into the hidden space of the generator. Given gloss tokens $g = ( g _ { 1 } , \dotsc , g _ { M } )$ , we compute:

$$
\begin{array} { r l } & { E _ { g } = \operatorname { E m b } _ { g } ( g ) , } \\ & { H _ { g } = \operatorname { A d a p t e r } ( E _ { g } + \operatorname { P E } _ { g } ) , } \\ & { \hat { H } _ { g } = W _ { g } H _ { g } . } \end{array}\tag{27}
$$

The adapter consists of learned gloss embeddings, learned positional embeddings, a lightweight Transformer encoder, and a linear projection. This allows the generator to condition on English semantics and pseudo-gloss structure without replacing the existing SOKE-style backbone.

## C.3. Retrieval Dictionary

The retrieval dictionary is built from isolated sign motion. Each isolated sign sequence is converted into the same SMPL-X representation and encoded by the frozen ST-VQVAE tokenizer:

$$
\begin{array} { r } { Z _ { w } \in \mathbb { N } ^ { L _ { w } \times 3 } , } \end{array}\tag{28}
$$

where $L _ { w }$ is the token length of isolated sign entry w, and the three columns are body, left-, and right-hand tokens. Each entry stores the gloss label, token matrix, and metadata used for informative-window selection. Gloss matching is performed after normalization and alias handling. If no pseudo-gloss token matches the dictionary, the model falls back to gloss-only conditioning.

## C.4. Auxiliary Isolated Sign Classifier

We train an auxiliary TCN-based isolated sign classifier on ASLCitizen [10] to extract temporal saliency from isolated motion sequences. This classifier is not intended as a sign language recognition contribution; it is only used to estimate which frames are informative within each isolated sign. All variable-length sequences are temporally resized to $t _ { \mathrm { f i x } } = 9 6$ . We use position and velocity features from 56 selected smpl kp3d joints and 43 selected poses joints, giving

$$
d _ { \mathrm { i n } } = ( 5 6 + 4 3 ) \times 6 = 5 9 4 .\tag{29}
$$

The classifier is a lightweight TCN with three temporal convolution layers:

$$
5 9 4 \to 2 5 6 \to 2 5 6 \to 2 5 6 ,
$$

using kernel sizes 5, 5, 3, BatchNorm, ReLU, and dropout 0.5 after the first two layers. Temporal average pooling is followed by a linear classifier over 2730 training classes.

We train three out-of-fold classifiers on folds BC, AC, and AB, and validate them on folds A, B, and C, respectively. Each sample is explained by the classifier that was not trained on that sample. The training uses batch size 64, 300 epochs, learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay $5 \times 1 0 ^ { - 4 }$ , label smoothing 0.1, early stopping patience 5 with minimum delta $1 0 ^ { - 4 } ,$ , and temporal scale augmentation in [0.5, 1.0] with edge padding. The trained classifiers reach above 95% top-5 accuracy, which is sufficient for our saliency extraction.

Table 10. Qualitative comparison of English inputs, NCSLGR reference glosses, and GPT-5.2 predictions, illustrating differences in compression, lexical choice, and grammatical marking conventions.
<table><tr><td>English</td><td>Dataset reference</td><td>Prediction</td><td>Structural observation</td></tr><tr><td>I would drive across.</td><td>DRIVE</td><td>IX-1P DRIVE CROSS</td><td>Reference is highly compact; pre- diction explicitly encodes subject</td></tr><tr><td>Does the teacher like chocolate?</td><td>TEACH+AGENT LIKE CHOCOLATE</td><td>CHOCOLATE REALLY LEARN+AGENT LIKE</td><td>pronoun and path. Prediction applies topic-fronting and an explicit question marker (Q), but substitutes</td></tr><tr><td>What did John buy yesterday?</td><td>JOHN BUY YESTERDAY</td><td>YESTERDAY JOHN BUY WHAT PART HOLD</td><td>LEARN+AGENT. Prediction fronts temporal ad- verb and appends explicit WH- question and aspectual markers.</td></tr></table>

For class score $S _ { c } ,$ input x, and baseline $x ^ { \prime } ,$ Integrated Gradients computes

$$
A = ( x - x ^ { \prime } ) \odot \int _ { \alpha = 0 } ^ { 1 } { \frac { \partial S _ { c } ( x ^ { \prime } + \alpha ( x - x ^ { \prime } ) ) } { \partial x } } d \alpha .\tag{30}
$$

We reduce the attribution tensor over joints and feature channels to obtain a temporal importance curve:

$$
a _ { t } = \mathrm { R e d u c e } _ { j , d } \left( A _ { t , j , d } \right) .\tag{31}
$$

The retrieval module uses the normalized temporal saliency curve after joint averaging; the full joint-time map is saved only for analysis.

## C.5. Keyframe-Aware Retrieval Window

The saliency curve is computed on video frames, but retrieval prompts are used after VQ tokenization. Since the tokenizer has a temporal compression factor of 4, we first average frame-level saliency within each 4-frame token bin. This gives a token-level saliency curve.

We then choose a short retrieval window from this curve. We use $\tau = 0 . 2 0$ to identify candidate high-saliency tokens, and then select the fixed 10-token window from the highestweighted region of the token-level saliency curve. If no token passes the threshold, we center the window on the highest-scoring token instead. The main experiments use a fixed 10-token window. This keeps the retrieval prompt length constant while moving the prompt to the most informative part of the sign.

Clip-specific selection and sign boundaries. The retrieval window is selected from each isolated clip’s own attribution curve rather than a shared Gaussian envelope, allowing early or late salient regions to anchor the window. Note that this dictionary-side prompt selection is distinct from continuous non-sign removal, and functions as a compact motion cue rather than a strict linguistic boundary detector for multi-phase signs.

We also compare this fixed-window design with softer alternatives in Table 11. E1 keeps the same crop length but lowers the trigger threshold, while E2–E5 retain the ful isolated-sign sequence and inject saliency through scaling, learned weight embeddings, or local position embeddings. These alternatives increase the amount of retrieval context, but the results favor a short explicit lexical window for the hand-aligned metrics most affected by retrieval prompts.

## C.6. Generator Training Details

We train the generator on How2Sign-MV3D motion tokens. The model follows the SOKE [79] framework and language backbone. We fully finetune it during training. The motion tokenizer is frozen. The input condition uses the pseudo-gloss sequence and ASLCitizen retrieval dictionary described in Section C.3.

The generator is trained with teacher-forced crossentropy over the three token heads. At inference time, it autoregressively predicts body, left-hand, and right-hand tokens, then decodes them through the frozen ST-VQVAE decoder. Unless otherwise specified, decoding is greedy with a maximum length of 100 tokens.

Using the How2Sign-MV3D train/validation/test split described in Section 5.1, we train the generator for 150 epochs with a batch size of 48 using AdamW $\begin{array} { r l } { ( \beta _ { 1 } } & { { } = } \end{array}$ $0 . 9 , \beta _ { 2 } = 0 . 9 9$ , zero weight decay) and a cosine learningrate schedule decaying from $2 \times 1 0 ^ { - 4 }$ to a minimum of $1 0 ^ { - 6 }$

## D. Evaluation Details

## D.1. SGNify-MV Reconstruction Diagnostic

We build SGNify-MV to test our cross-view curation pipeline under known 3D geometry. SGNify provides monocular videos and ground-truth meshes, but no sideview video. We are not aware of a public sign language benchmark that provides both multi-view video and ground-truth SMPL-X meshes. We therefore render frontal and side-view videos from the SGNify [17] ground-truth meshes in Blender.

Table 11. Retrieval prompt diagnostic on How2Sign-MV3D. E0 is the setting used in the main experiments. Full-sequence weighting does not improve hand reconstruction over the fixed 10-token crop.
<table><tr><td>Setting</td><td>Retrieval prompt</td><td colspan="3">DTW-JPE↓</td><td colspan="3">DTW-PA-JPE↓</td></tr><tr><td></td><td></td><td>Body</td><td>LHand</td><td>RHand</td><td>Body</td><td>LHand</td><td>RHand</td></tr><tr><td>E0 fixed crop,  $\tau = . 2 0$ </td><td>10-token crop</td><td>4.983</td><td>8.197</td><td>7.697</td><td>4.577</td><td>1.814</td><td>1.906</td></tr><tr><td>E1 fixed crop, τ = .14</td><td>10-token crop</td><td>5.137</td><td>8.192</td><td>8.179</td><td>4.570</td><td>1.983</td><td>1.996</td></tr><tr><td>E2 saliency scale</td><td>full sequence</td><td>5.159</td><td>8.356</td><td>8.434</td><td>4.685</td><td>1.983</td><td>1.998</td></tr><tr><td>E3 weight embedding</td><td>full sequence</td><td>5.261</td><td>8.343</td><td>8.390</td><td>4.744</td><td>1.987</td><td>1.999</td></tr><tr><td>E4 scale + embedding</td><td>full sequence</td><td>5.189</td><td>8.207</td><td>8.209</td><td>4.659</td><td>1.944</td><td>1.979</td></tr><tr><td>E5 scale + embedding + local position</td><td>full sequence</td><td>5.132</td><td>8.216</td><td>8.234</td><td>4.632</td><td>1.965</td><td>1.969</td></tr></table>

Table 12. TR-V2V results on SGNify and SGNify-MV. We report TR-V2V error (mm) for UBody-F and both hands. The first block follows the DexAvatar [32] SGNify evaluation protocol. The remaining blocks report our results using DexAvatar’s evaluation code.
<table><tr><td>Setting/Method</td><td>Input</td><td>UBody-F ↓</td><td>LHand ↓</td><td>RHand ↓</td></tr><tr><td colspan="5">Real SGNify, DexAvatar protocol</td></tr><tr><td>OSX [37]</td><td>Front</td><td>47.32</td><td>18.34</td><td>18.12</td></tr><tr><td>Neural Sign Actors [5]</td><td>Front</td><td>46.42</td><td>16.17</td><td>15.23</td></tr><tr><td>EVA* [24]</td><td>Front</td><td>40.38</td><td>13.73</td><td>13.68</td></tr><tr><td>DexAvatar [32]</td><td>Front</td><td>30.13</td><td>13.53</td><td>13.08</td></tr><tr><td colspan="5">Real SGNify</td></tr><tr><td>OSX [37]</td><td>Front</td><td>45.38</td><td>34.39</td><td>30.54</td></tr><tr><td>AiOS [56]</td><td>Front</td><td>37.42</td><td>35.67</td><td>36.88</td></tr><tr><td>AiOS+WiLoR [43, 56]</td><td>Front</td><td>34.61</td><td>13.11</td><td>12.45</td></tr><tr><td>Ours one-view</td><td>Front</td><td>34.62</td><td>12.87</td><td>12.34</td></tr><tr><td colspan="5">Rendered SGNify-MV</td></tr><tr><td>OSX [37]</td><td>Front</td><td>40.50</td><td>32.48</td><td>29.92</td></tr><tr><td>AiOS [56]</td><td>Front</td><td>38.36</td><td>38.88</td><td>42.80</td></tr><tr><td>AiOS+WiLoR [43, 56]</td><td>Front</td><td>33.50</td><td>17.36</td><td>21.38</td></tr><tr><td>Ours one-view</td><td>Front</td><td>32.77</td><td>14.94</td><td>17.86</td></tr><tr><td>Ours cross-view</td><td>Front+Side</td><td>32.80</td><td>14.95</td><td>17.88</td></tr></table>

The original SGNify [17] videos are clean 30-FPS isolated signs with slow and standard motion, yet How2Sign [12] videos are 24-FPS continuous sign language. To make the rendered benchmark closer to the failures seen in How2Sign, we render SGNify [17] 3D annotation at 24 FPS, add motion blur, and apply random temporal offsets between views. SGNify stores each frame as a separate OBJ mesh, but the topology and vertex ordering are fixed. We use this to build Blender Shape Keys, convert the discrete mesh sequence into continuous deformation, and sample different views at slightly different times. UV and Texture made by UVMap-ID [66]

TR-V2V shown in Table 12 is our main reconstruction diagnostic because it keeps the hand in the body coordinate space. This matters for signing: the same finger shape can have a different meaning when it appears near a different body part. Part-aligned hand errors are useful for local handshape analysis, but they remove this global placement information.

Table 13. Hand motion dynamics. We compare wrist and finger motion measured from GT SMPL-X meshes for SGNify and pseudo SMPL-X meshes for the other datasets. F/W Ratio is the average global finger motion divided by the average global wrist motion. Local Finger subtracts the wrist position before measuring finger movement. All motion values are speeds in the SMPL-X coordinate scale, computed as frame-to-frame 3D joint displacement multiplied by frame rate.
<table><tr><td>Dataset</td><td>Wrist</td><td>Finger</td><td>F/W Ratio</td><td>Local Finger</td></tr><tr><td>ASLCitizen</td><td>0.301</td><td>0.408</td><td>1.36</td><td>0.188</td></tr><tr><td>How2Sign</td><td>0.252</td><td>0.471</td><td>1.87</td><td>0.308</td></tr><tr><td>MM-WLAuslan</td><td>0.382</td><td>0.547</td><td>1.43</td><td>0.220</td></tr><tr><td>SGNify</td><td>0.466</td><td>0.535</td><td>1.15</td><td>0.124</td></tr><tr><td>SGNify(central)</td><td>0.383</td><td>0.510</td><td>1.33</td><td>0.220</td></tr></table>

On real-front SGNify [17], one-view repair slightly improves the hand scores. On rendered SGNify-MV, crossview repair remains close to one-view repair rather than improving it, suggesting that the side-view evidence is limited by temporal mismatch in this controlled rendered setting.

Table 13 helps explain why this diagnostic result should not be read as the full behavior on continuous How2Sign [12]. SGNify has weaker local finger dynamics than How2Sign: the finger-to-wrist ratio is 1.15 for SGNify-MV and 1.87 for How2Sign, and local finger motion is 0.124 versus 0.308. Even after trimming SGNify to its central signing interval, its local finger motion remains below How2Sign. SGNify-MV therefore provides controlled geometry and useful failure cases, but it does not fully match the faster, more finger-active continuous signing regime where How2Sign-MV3D is used for model training.

Although part-aligned MPVPE-PA is not sensitive to hand direction and position errors, we also report MPVPE-PA in Table 14 because recent SGNify tables, including How2Sign-Synth3D [61], report part-aligned MPVPE-PA. We also avoid merging published rows into our table: some SGNify values appear under different metric names across papers, so a single mixed leaderboard would be ambiguous.

Schreiben

Glas

![](images/9a15b2a97ab37df27fcb92618bae02f68b9939aae61c47d9cc40c4393f582916.jpg)  
Figure 9. SGNify-MV visualization. Examples for two SGNify signs. We show the original SGNify [17] video, ground-truth mesh sequence, and our rendered low FPS frontal/side videos with controlled temporal mismatch and motion blur.

Table 14. MPVPE-PA results on SGNify and SGNify-MV. We report MPVPE-PA (mm) for UBody-F and both hands. The first block follows the How2Sign-Synth3D [61] SGNify evaluation protocol. The remaining blocks report our results using SOKE [79]’s evaluation code.
<table><tr><td>Setting/Method</td><td>Input</td><td>UBody-F↓</td><td>LHand ↓</td><td>RHand ↓</td></tr><tr><td colspan="5">Real SGNify front, H2S-Synth3D protocol</td></tr><tr><td>OSX [37]</td><td>Front</td><td>47.32</td><td>18.34</td><td>18.12</td></tr><tr><td>SGNify [17]</td><td>Front</td><td>55.63</td><td>19.22</td><td>17.50</td></tr><tr><td>Neural Sign Actors [5]</td><td>Front</td><td>46.42</td><td>16.17</td><td>15.23</td></tr><tr><td>SignAvatars [73]</td><td>Front</td><td>38.50</td><td>15.56</td><td>13.28</td></tr><tr><td>SOKE [79]</td><td>Front</td><td>46.73</td><td>10.55</td><td>8.94</td></tr><tr><td>H2S-Synth3D [61]</td><td>Front</td><td>20.37</td><td>10.63</td><td>8.83</td></tr><tr><td colspan="5">Real SGNify</td></tr><tr><td>Ours one-view</td><td>Front</td><td>22.67</td><td>8.47</td><td>9.06</td></tr><tr><td>Rendered SGNify-MV</td><td></td><td></td><td></td><td></td></tr><tr><td>Ours one-view</td><td>Front</td><td>17.33</td><td>9.46</td><td>11.83</td></tr><tr><td>Ours cross-view</td><td>Front+Side</td><td>17.32</td><td>9.55</td><td>11.73</td></tr></table>

## D.2. Back-Translation BLEU Diagnostic

We do not report back-translation BLEU as a main metric. Recent pose-based SLP evaluation work [28, 54, 60] observes that back translation style metrics require a poseto-text translation model for the relevant language direction and dataset, but the evaluator code, weights, and upper bounds are rarely standardized or released [5, 79]; training such a model for each dataset is itself a costly extra task, and human evaluation is still needed for semantic validity [28]. This is especially problematic on open-domain ASL data: How2Sign translation remains difficult even for dedicated sign language translation systems [60], and large onlinevideo resources such as OpenASL [54] still show the difficulty of robust ASL-to-English translation in unconstrained settings.

We also ran internal diagnostics with open back translation style evaluators. A SIGN-IDD-style open SLT model [59] trained on How2Sign reached less than 2 BLEU-4 in our setup. Replacing its pose input module with our ST-VQVAE representation, connecting it through a Transformer adaptor, and decoding with T5 text models improved the best BLEU-4 to roughly 6, but the model still showed partial collapse. These results are useful diagnostics, but they are far below the reliability needed to rank SLP systems by private back translation BLEU alone.

![](images/6ef582ca35a7b9c20744a4645edf9931796bd178db421a481268323890b441d4.jpg)  
Figure 10. MM-WLAuslan-MV3D and ASLCitizen-3D visualization. Examples for two signs for each video dataset. We show the original ASLCitizen [10] and MM-WLAuslan [53] videos, with our 3D annotations below.

## D.3. ASLCitizen-3D and MM-WLAuslan-MV3D visualization

We processed ASLCitizen [10] and MM-WLAuslan [53] using our single-view/cross-view curation pipeline to visually evaluate our pipeline, and also provide additional 3D sign language annotations for future work.

## E. Front–Side Detection Availability and Reliability

We audit the raw AiOS and WiLoR outputs across the 31,474 jointly available paired clips (from the 35,241-clip manifest) prior to non-sign removal, interpolation, crossview repair, and synthetic bridging. All statistics are frameweighted, and left/right labels denote estimator-assigned anatomical hands rather than image-space sides. Because How2Sign lacks 3D ground truth and manual visibility annotations, these metrics characterize raw estimator behavior and heuristic filtering rates rather than ground-truth detection recall.

In Table 15, No box indicates frames where no hand bounding box meets the 0.3 confidence threshold (encompassing both missed detections and genuine occlusions or out-of-frame movements). For all detected boxes in this cohort, the exported pose and keypoint arrays are finite.

Beyond box detection, our native merge rule rejects a hand fit if: (i) its wrist axis-angle $L _ { 1 }$ norm is $\leq 1 0 ^ { - 6 }$ (a near-zero-rotation heuristic distinct from the detector box mask), (ii) the AiOS–WiLoR wrist orientation disagreement exceeds 120<sup>◦</sup>, or (iii) WiLoR changes by > 60<sup>◦</sup>/frame while AiOS changes by < 15<sup>◦</sup>/frame. Because these failure modes overlap, we report their union after a one-frame temporal dilation. Among finite raw fits, this rule rejects an additional 1.92%, 1.83%, 7.62%, and 14.64% of frames for front-left, front-right, side-left, and side-right hands, respectively. Note that an inaccurate AiOS wrist reference can occasionally trigger a false orientation rejection on a plausible WiLoR hand fit.

For the body, AiOS outputs dense per-frame meshes; however, side-view body fits frequently exhibit under-fitting and implausible torso tilt (see Fig. 12). Consequently, we retain the frontal body reconstruction as the kinematic backbone while performing bidirectional cross-view refinement for the hands. Specifically, frontal and side hand configurations are mutually validated alongside their tempora neighbors, allowing reliable side-view observations to correct occluded frontal hands and vice versa. Finally, in our raw-body temporal alignment diagnostic, the offset search reaches the ±8-frame boundary in 40.40% of clip pairs.

## F. Sentence Boundaries and Motion-Side Non-Sign Removal

Our curation pipeline addresses temporal alignment at three distinct levels: (i) sentence-boundary segmentation, which slices full How2Sign recordings using the official manually realigned sentence timestamps rather than the legacy How2 timing; (ii) cross-view synchronization, which aligns frontal and side motion streams; and (iii) intra-clip nonsign removal, which prunes internal idle intervals identically across both views using frontal kinematic cues (detailed in Appendix A.2) while keeping the paired English transcript unchanged.

Table 15. Raw detection availability and heuristic rejection rates. No box and Rule rejected are normalized by all decoded frames, whereas Rejected / finite uses valid finite fits as the denominator.
<table><tr><td>View</td><td>Hand</td><td>Frames</td><td>No box (%)</td><td>Rule rejected (%)</td><td>Rejected / finite (%)</td></tr><tr><td>front</td><td>left</td><td>5,508,797</td><td>2.28</td><td>4.15</td><td>1.92</td></tr><tr><td>front</td><td>right</td><td>5,508,797</td><td>0.67</td><td>2.49</td><td>1.83</td></tr><tr><td>side</td><td>left</td><td>5,508,715</td><td>8.41</td><td>15.39</td><td>7.62</td></tr><tr><td>side</td><td>right</td><td>5,508,715</td><td>23.06</td><td>34.32</td><td>14.64</td></tr></table>

![](images/ae54b22b0ab35db12fec9b2d285fc8c109f4822803f5c937ab84064bf85c51de.jpg)  
Figure 11. Qualitative example of detection absence and cross-view recovery. Six mapped frames showing raw RGB inputs, per-view estimator outputs, and the resulting processed motion.

Motivation: untranscribed waiting intervals. Even within manually realigned sentence boundaries, raw recordings frequently contain extended pauses—such as signers waiting for the next teleprompter prompt—that have no lexical counterpart in the English text. Because hiring expert ASL annotators to manually trim every sub-sentence pause across 35k clips is prohibitive, our kinematics-driven nonsign filter provides a scalable way to compact these idle intervals. Notice that our conservative buffering rules (23- frame onset delay and 8-frame lookahead) are specifically designed to remove prolonged waiting periods while preserving brief grammatical holds and transitions.

Dataset-wide pruning statistics. Table 16 summarizes the non-sign pruning statistics across the 30,447 newly processed clip pairs and 4,492 reused pairs. In the primary 30,447-clip cohort, intra-clip pruning is triggered in 18,378 clips (60.36%), removing 989,360 out of 5,325,225 decoded frontal frames (18.58% frame-weighted) and inserting 103,448 synthetic bridge frames (4 frames per cut) per view. Because excised intervals are bridged by 4-frame transitions, we track source-frame deletion indices separately from synthetic bridges and boundary padding. In Table 16, we report both the total removed source frames per clip and the longest contiguous removed run per clip (evaluated across all clips as well as edited clips only) to characterize the distribution of sustained idle pauses.

Table 16. Motion-side non-sign pruning workload and per-clip source-frame deletion statistics. Real-frame removal is frame-weighted over decoded frontal frames, while 4-frame synthetic bridges are counted separately per view. Per-clip deletion distributions are reported across all clips (All, where min = 0) and restricted to pruned clips (Edited, with ≥ 1 removed interval).
<table><tr><td></td><td></td><td></td><td colspan="3">Cohort Workload</td><td colspan="5">Per-Clip Deleted Source Frames</td></tr><tr><td>Cohort</td><td>Split</td><td>Clips</td><td>Edited clips (%)</td><td>Frames removed (%)</td><td>Bridge frames</td><td>Mean / Med (All)</td><td>Mean / Med (Edited)</td><td>Min (Edited)</td><td>Max total</td><td>Max single run</td></tr><tr><td>New</td><td></td><td>26,791</td><td>16,194 (60.45%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>train</td><td></td><td>966</td><td>18.72%</td><td>91,176</td><td>32.69 / 18</td><td>54.08 / 37</td><td>11</td><td>2,400</td><td>850</td></tr><tr><td>New</td><td>val</td><td>1,557</td><td>(62.04%) 1,218</td><td>18.57%</td><td>5,600</td><td>33.45 / 21</td><td>53.92 / 42</td><td>12</td><td>377</td><td>336</td></tr><tr><td>New</td><td>test</td><td>2,099</td><td>(58.03%) 18,378</td><td>16.79%</td><td>6,672</td><td>29.33 / 16</td><td>50.54 / 36</td><td>12</td><td>457</td><td>436</td></tr><tr><td>New</td><td>overall</td><td>30,447</td><td>(60.36%)</td><td>18.58%</td><td>103,448</td><td>32.49 / 18</td><td>53.83 / 37</td><td>11</td><td>2,400</td><td>850</td></tr><tr><td>Reused</td><td>overall</td><td>4,492</td><td>2,769 (61.64%)</td><td>18.89%</td><td>15,988</td><td>34.08 / 19</td><td>55.28 / 38</td><td>12</td><td>738</td><td>676</td></tr></table>

Scope and limitations. While combining manually realigned sentence boundaries with kinematic pause removal substantially reduces temporal redundancy, motion-activity cues operate purely on kinematics rather than lexical semantics. Consequently, this pipeline targets untranscribed idle intervals within a clip, rather than correcting potential upstream transcript errors or performing word-level forced alignment.

## F.1. Released Resources and Statistics

The release comprises three SMPL-X motion datasets, each providing the final reconstructed motion sequence per sample (omitting raw AiOS/WiLoR predictions and per-view intermediates). Specifically, the curated outputs comprise: (i) How2Sign-MV3D with 34,939 sequences (30,903 train, 1,711 validation, and 2,325 test); (ii) ASLCitizen-3D with 83,325 sequences derived from ASLCitizen; and (iii) MM-WLAuslan-MV3D with 38,580 sequences derived from MM-WLAuslan. Note that these totals reflect the successfully curated motion sequences rather than the nominal sizes of the upstream video corpora.

In addition, SGNify-MV provides rendered frontal- and side-view videos for the 57 SGNify signs, generated from the original ground-truth meshes to support controlled multi-view reconstruction evaluation. Unlike the three motion datasets above, SGNify-MV releases the rendered multi-view videos alongside our Blender rendering scripts to allow custom viewpoint generation.

Finally, we release the GPT-5.2 How2Sign pseudo-gloss annotations, the cross-view curation pipeline, and model training scripts via our GitHub repository. To respect upstream licensing, we do not redistribute raw source videos, original SGNify meshes, or SMPL-X/MANO body model files, which remain subject to their respective access terms.

![](images/a37e662e896152caa87dfb06187086d6bf0ed3d18be00bba28182ee836c0b978.jpg)  
Figure 12. Stage-by-stage pipeline visualization on clip 1xl3MOjKX84 2-8-rgb. From top to bottom: synchronized frontal and side RGB frames, native-view AiOS-only meshes, and per-view AiOS+WiLoR merged meshes (incorporating single-view reliability filtering and interpolation prior to cross-view synchronization and finger repair). Note the visible torso tilt in the side-view body meshes, which motivates retaining the frontal body backbone. The bottom rows highlight the frontal output sequence with cross-view-refined hands, illustrating local finger configurations before and after cross-view repair across five updated timestamps (with root, body, and shape held fixed for visual comparison).