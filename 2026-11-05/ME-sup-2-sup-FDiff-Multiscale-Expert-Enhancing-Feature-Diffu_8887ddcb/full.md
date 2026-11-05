# ME<sup>2</sup>FDiff: Multiscale Expert Enhancing Feature Diffusion in Residual Network for Brain Tumor Segmentation

Afrida Rahman and Bijoy Kundu

Abstract—Automated segmentation of post-treatment glioma from MRI is essential for accurate diagnosis and treatment planning, yet remains difficult. Surgery and chemoradiotherapy cause structural changes in brain like resection cavities, edema, and fibrosis, that alter imaging appearance and mimic tumor recurrence. Contemporary segmentation architectures attain high accuracy with two limitations. First, the bottleneck originates from the deepest encoder stage, so one representation must serve sub-regions that differ widely in size and appearance. Second, they produce the segmentation in a single forward pass with no mechanism for refinement. We propose ME<sup>2</sup>FDiff, a singlemodel architecture, addressing both. An attention-conditioned spectral diffusion block refines the bottleneck. It uses the imageconditioned latent diffusion process with two processes: a spatial pre-attention gate and a spectral fusion block whose timestepadaptive bandpass filter selects the informative frequency band at each noise level. A multiscale expert-enhanced feature block fuses the intermediate encoder scales into the bottleneck through expert-selected shared tokens, guaranteeing balanced utilization without an auxiliary loss. On the BraTS 2024 post-treatment glioma dataset, ME<sup>2</sup>FDiff achieves the best results on every tumor sub-region among all single-model baselines (91.23% Dice, 84.52% IOU, 2.13 mm HD95 on whole-tumor) and surpasses a three-expert ensemble on both overlap metrics at roughly onethird of its inference cost. Against recent state-of-the-art methods retrained under identical conditions, it maintains the strongest performance with the fewest parameters of any competing model. These findings demonstrate that a single, compact network can absorb the specialization, usually sought through ensembling or larger architectures.

Index Terms—BraTS2024, deep learning, medical image segmentation, MRI, expert routing, diffusion model.

## I. INTRODUCTION

LIOMAS are a prevalent malignant brain tumor and a G major cause of cancer-related mortality in adults [1]. Their infiltrative and diffusive nature yields poorly defined borders, which makes segmentation difficult. Manual delineation is labor-intensive and subject to inter-observer variability [2]. Magnetic resonance imaging (MRI) is the standard modality for glioma evaluation [3]. While pre-treatment scans show clearly enhanced tumor and edema [4], post-treatment scans are more complex, exhibiting radiation necrosis, resection cavities, and post-surgical inflammation that often resemble recurrence. Benchmark challenges like BraTS have accelerated advances in automated segmentation, with BraTS2024 focusing on the post-treatment setting [5]. Recent encoderdecoder networks [6]–[8] achieve high aggregate performance but share two limitations. First, the bottleneck originates only from the deepest encoder stage. The shallower stages hold the fine detail needed to find small structures, but they reach the decoder only through skip connections and never enter the bottleneck itself. As a result, one feature representation has to describe every subregion at once, even though they differ widely in the large, small, and sharply outlined regions. Second, the network predicts the segmentation in a single pass and has no way to correct it afterwards. This is a problem in post-treatment scans, where the boundary between tumor and treatment effects is often unclear, and a first prediction is easy to get wrong. Ensembling is the established remedy for the first limitation [9]. Several architectures are trained independently, and their predictions are combined so that the strengths of one compensate for the weaknesses of another. However, it is quite expensive; inference increases with the number of experts, and the combination is usually controlled by a predetermined mechanism, such as a sequential cascade, uniform averaging, or confidence-derived weighting, none of which can reason about the input. Diffusion models naturally address the second limitation by replacing a single deterministic forward pass with an iterative denoising process that progressively refines predictions [10]. However, they are computationally prohibitive at volumetric resolution when used naively in image space.

We introduce a single volumetric segmentation network, ME<sup>2</sup>FDiff, that meets these two limitations without recourse to multiple networks. The design follows the routing mechanism that arbitrates between separate models in an ensemble, which is equally applicable to arbitrating between encoder scales inside a single network. The diffusion process that is intractable over an image volume becomes affordable when applied to a compact mask latent space. A multimodal MRI volume is first encoded into multi-scale stage features and a deep bottleneck embedding. The Attention-Conditioned Spectral Diffusion (ACSDiff) module refines this bottleneck through an image-conditioned latent diffusion process, which is built by a spatial pre-attention gate (SPG) and a spectral diffusion conditioner (SDC). Here, the SPG sharpens the conditioning features, and the SDC module couples the diffusion-noise and semantic streams in the frequency domain under a timestepadaptive band-pass filter. Before the result is decoded to the final segmentation, the multiscale expert-enhanced feature block (ME<sup>2</sup>FB) fuses all intermediate encoder stages into the refined bottleneck through expert-selected routing. Therefore, our proposed network shows the strongest held-out test performance in its model family, with an average Dice of 0.857, at a fraction of its computational cost. The main contributions of this work are summarized as follows:

• We integrate a multiscale expert-enhanced feature block (ME<sup>2</sup>FB) into a SegResNet-VAE backbone, the first use of expert routing within a VAE-regularized residual CNN for 3D brain tumor segmentation.

• We propose ACSDiff, an attention-conditioned spectral diffusion that refines the encoder bottleneck in the frequency domain, coupled with a spatial pre-attention gate that sharpens its conditioning features.

• The combination of these components within a single network, optimized by the hybrid loss functions (Dice, Tversky, and Focal), achieves the SOTA performance at the lowest parameter count.

• Our method outperforms not only every individual backbone but also the multi-model ensembles assembled from state-of-the-art segmentation networks on BraTS2024 post-treatment gliomas.

## II. RELATED WORK

Deep learning has driven successive breakthroughs in brain tumor segmentation [11], [12]. Its strength is the ability to extract discriminative features directly from data without the need for manually created priors [13]. Convolutional architectures established the dominant paradigm on the pretreatment BraTS datasets [14], [15]. Afterwards, the U-Net encoder-decoder became the standard backbone, enriched with attention, residual, and multi-scale mechanisms [16], [17]. Convolutions capture long-range dependencies only indirectly, which led to attention-based designs [12]. Several of which are used for our method comparison: TransUNet [18] applies a Vision Transformer to CNN-extracted features and decodes them with a U-Net; SegFormer [19] pairs a hierarchical encoder with no positional encoding and a lightweight MLP decoder; MedFormer [20] reduces the quadratic cost of selfattention on 3D data with bidirectional multi-head attention; U-Mamba [21] replaces attention with a state-space model that captures long-range context in linear time; and MoE-SAM [22] adapts the Segment Anything foundation model [23] through a mixture-of-experts feature-enhancement stage. Despite their differences, these designs have one structural characteristic in common: the bottleneck is created by the deepest encoder stage alone, while shallower stages that carry the detail required to localize small sub-regions enter merely as skips.

To compensate for architecture-dependent variability, ensembling combines independently trained networks through majority voting, probability, or weighted averaging or stacking [9]. It dominates recent BraTS submissions, at the price of inference that scales with the number of experts. Mixture-ofexperts routing [11] offers a learned alternative to such fixed combination rules, though it has served chiefly to combine models rather than to allocate capacity across scales within one. A separate line of work replaces the single deterministic forward pass with iterative refinement: denoising diffusion [24] has been applied to segmentation by diffusing the mask conditioned on the image. However, this has been limited to 2D image space, where computing spatial attention affinity remains affordable but does not scale to 3D volumes. Our method draws on both by relocating each inside a single network that operates directly in 3D volume.

## III. MATERIALS AND METHODS

## A. Dataset

In this study, experiments are conducted on the BraTS 2024 Adult Glioma Post-Treatment (GLI) dataset, which, unlike earlier pre-operative BraTS editions, includes a resection cavity (RC) label and is thus representative of clinically realistic post-surgical glioma MRI. The dataset comprises 1,621 preprocessed volumes drawn from 731 unique patients, with multiple longitudinal timepoints contributing several volumes per patient. Each case provides four co-registered MRI modalities, i.e. native T1-weighted (T1n), contrastenhanced T1-weighted (T1c), T2-weighted (T2w), and T2-FLAIR (T2f) and stacked in the fixed channel order [T1n, T1c, T2w, FLAIR]. Voxel-wise annotations are provided for five classes: background (0), necrotic core (NCR, 1), edema / surrounding non-enhancing FLAIR hyperintensity (ED/SNFH, 2), enhancing tumor (ET, 3), and resection cavity (RC, 4). Resection cavity is the newly added tissue subregion, which is necessary for those patients who are receiving adjuvant radiotherapy after surgical resection. [5] (Figure 1)

1) Preprocessing: All volumes are preprocessed using a fixed pipeline prior to model training. Each modality is intensity-normalized independently using z-score normalization computed only over non-zero (brain) voxels, so that statistics are not skewed by the zero-valued background; background voxels remain at zero after normalization. Initially, the BraTS MRI scans have large dimensions i.e. 182 × 218 × 182. Volumes are subsequently cropped to the bounding box of the non-zero brain region to remove uninformative background. Later, it is resampled to a fixed isotropic grid of 128×128×128 voxels using trilinear interpolation (order 1) for the image channels and nearest-neighbor interpolation (order 0) for the segmentation masks to preserve discrete label values. To standardize the 3D resolution, all the images are cropped using the ratios [54:184, 56:184, 13:141] throughout the three spatial dimensions. Since the validation data lacked ground truth labels, the entire training dataset of 1621 samples is divided into training and validation subsets using a 70:20:10 split. This results in 1134 samples for training, 324 samples for validation, and 163 samples for testing.

## B. Baseline Methods

To situate the proposed model within the current state of the art, we establish two families of baselines of increasing strength. The first comprises individually trained segmentation architectures spanning the principal design paradigms in volumetric medical imaging: convolutional, transformer-based, and foundation-model (Section III-B1). The second comprises ensemble fusion strategies that aggregate the predictions of several such architectures at inference (Section III-B2).

![](images/54681f9821f17ec6b8357b55e883a3e1583f26850f1fa76948cb012599bbe7b3.jpg)  
Fig. 1. BraTS2024 dataset. The data set has four brain MRI scans i.e. pre-contrast T1-weighted (T1n), contrast-enhanced T1-weighted (T1ce), T2-weighted fluid-attenuated inversion recovery (T2f), T2-weighted (T2w) and four sub-regions in segmentation mask i.e. enhancing tissue (ET), surrounding non-enhancing T2/fluid-attenuated inversion recovery (FLAIR) hyperintensity (SNFH), non-enhancing tumor core (NETC), and the newly added tissue sub-region, resection cavity (RC).

1) Single-model architectures: A set of model architectures is implemented and trained independently on the four-channel (T1n, T1c, T2w, FLAIR) input. Among convolutional designs, the 3D U-Net [17] is the symmetric encoder-decoder that founded the paradigm, propagating high-resolution detail to the decoder through skip connections. SegResNetVAE [25], [26] extends a residual encoder-decoder with a variational branch that reconstructs the input from the bottleneck, regularizing the encoder representation under limited training data. AHNet [27] transfers features learned from 2D pretraining into a 3D network with anisotropic kernels, decoupling in-plane from through-plane feature extraction. MedNeXt [28] adapts the ConvNeXt block to volumetric segmentation, combining depthwise convolutions with inverted bottlenecks and large kernels. Among transformerbased designs, UNETR [29] replaces the convolutional encoder with a plain vision transformer applied to 3D patches. Swin UNETR [6] instead employs a hierarchical encoder that applies self-attention within local windows shifted between successive layers, capturing long-range context at tractable volumetric cost. Finally, SAM introduces a promptable foundation model that generates segmentation masks from points, text prompts and generalizes to unseen objects without task-specific retraining [23].

2) Ensemble Fusion Strategies: As no single architecture is uniformly optimal across sub-regions, ensembling constitutes the stronger baseline family. All strategies fuse the three best models, i.e., SegResNetVAE, SwinUNETR, and MedNeXt, selected based on their individual average Dice scores and HD95. Let $p _ { k , c } ( v )$ denote the probability assigned by expert k to class c at voxel v, and $\hat { p } _ { c } ( v )$ the fused prediction, decoded as $\hat { y } ( v ) = \arg \operatorname* { m a x } _ { c } \hat { p } _ { c } ( v )$ . Dynamic weighting [30] relaxes uniform averaging by letting each expert’s contribution vary per voxel, with weights derived analytically from its own predictive confidence and normalized across experts; cascade fusion [15] combines the experts sequentially rather than in parallel, stitching three independently trained stage networks at inference in the order WNet → TNet → ENet, so that each stage refines only within the region delineated by its predecessor. Finally, MoE gating [11] instead learns the combination, using a gating network trained over the frozen experts to predict input-conditional weights. These three strategies (cascaded refinement, learning gate, and confidence weighting) are evaluated as ensemble baseline methods against our proposed work.

## C. Proposed Method

1) Overview: The architecture of ME<sup>2</sup>FDiff is illustrated in Fig. 2. A multi-modal MRI volume is encoded by a SegRes-NetVAE encoder into a hierarchy of multi-scale stage features together with a deepest bottleneck embedding $c ^ { 0 } \in \mathbb { R } ^ { 2 5 6 \times 1 6 ^ { 3 } }$ which serves as the semantic condition. This representation is refined by two modules in sequence. The first, ACSDiff, comprises a Spatial Pre-attention Gate (SPG) that sharpens the conditioning features in the image domain. It is coupled with a Spectral Diffusion Conditioner (SDC) that fuses the diffusion noise and semantic streams in the frequency domain under a timestep-adaptive band-pass gate, and the preattended features condition the diffusion directly. The second module, Multiscale Expert Enhanced Feature Block (ME<sup>2</sup>FB), then rebuilds the refined bottleneck from all encoder scales, using expert-choice routing to determine which scale-specific evidence is emphasized at which spatial location. The resulting representation is returned to the SegResNetVAE decoder, which produces the final segmentation. The two modules are trained jointly under a composite objective comprising a latent noise-prediction term $L _ { \epsilon } = \| \epsilon - \hat { \epsilon } \| _ { 2 } ^ { 2 }$ following DDPM [31], an auxiliary segmentation term $L _ { \mathrm { s e g } } ,$ a class-weighted combination of Tversky, Dice, and Focal losses applied to the decoded prediction and the variational regularizer $L _ { \mathrm { v a e } }$ of the backbone:

$$
L _ { \mathrm { t o t a l } } = L _ { \epsilon } + \lambda _ { \mathrm { s e g } } L _ { \mathrm { s e g } } + \lambda _ { \mathrm { v a e } } L _ { \mathrm { v a e } } .\tag{1}
$$

![](images/2ee1d673c2002dc20187c03f540b6acf763491bb4bf3573ca65f84d252ca962b.jpg)  
Fig. 2. Overall model architecture of our proposed method. It has two important modules at the bottleneck of SegResNetVAE, i.e 1. Attention-Conditioned Spectral Diffusion (ACSDiff) combined with two modules: (a) Spatial Pre-attention Gate (SPG), (b) Spectral Diffusion Conditioner (SDC) with time-adaptiv Band Pass Filter (NBP filter), 2. Multiscale Expert Enhanced Block (ME<sup>2</sup>FB)

2) Attention-Conditioned Spectral Diffusion: Module ACS-Diff introduces such a mechanism by formulating mask refinement as an image-conditioned latent diffusion process and conditioning that process through two attention blocks acting in complementary domains: SPG, a spatial pre-attention gate applied to the features that supply the condition; and SDC, a spectral fusion block that couples the condition to the diffusion noise in the frequency domain.

a) Latent mask diffusion: Diffusing the mask in image space is computationally prohibitive in three dimensions. ACS-Diff instead operates on a compact latent: the ground-truth mask is compressed to a latent $\mathbf { \bar { \boldsymbol { z } } _ { 0 } } \in \mathbb { R } ^ { 5 \times 1 6 ^ { 3 } }$ and corrupted by the DDPM forward process to $z _ { t } ,$ from which a denoiser predicts the added noise ϵˆ conditioned on $c ^ { 0 }$ . Operating on a low-resolution latent rather than in image space keeps the generative refinement tractable in three dimensions while retaining full volumetric condition. At inference, the mask is unavailable, and the latent is instead sampled from Gaussian noise and denoised by a conditional DDIM reverse process.

b) Spatial Pre-attention Gate (SPG): The condition is only as informative as the features from which it is formed. SPG therefore refines each encoder stage at its native resolution using a lightweight convolutional block attention module [32], emphasizing informative channels and tumor-relevant voxels before the features are fused. Channel attention aggregates globally averaged and max-pooled descriptors through a shared reduction MLP followed by a sigmoid, while spatial attention convolves the channel-wise average and maximum maps into a single sigmoid gate over voxels:

$$
F ^ { \prime } = M _ { s } ( M _ { c } ( F ) \otimes F ) \otimes \big ( M _ { c } ( F ) \otimes F \big ) .\tag{2}
$$

where $F$ and $F ^ { \prime }$ are the input and refined stage features, $M _ { c } ( \cdot )$ and $M _ { s } ( \cdot )$ denote the channel and spatial attention maps, and ⊗ is element-wise multiplication with broadcasting. Being of linear complexity, SPG is affordable at the 128<sup>3</sup>, $6 4 ^ { 3 }$ , and $3 2 ^ { 3 }$ stage resolutions, at which self-attention would be restrictive. Both the diffusion condition and the ME<sup>2</sup>FB fusion consequently operate on saliency-reweighted features.

c) Spectral diffusion Conditioner (SDC): The denoiser must reconcile two embeddings of differing provenance: the noise embedding e, derived from a corrupted mask latent, and the semantic embedding $c ^ { 0 } ,$ , derived from the image. Their direct fusion is impaired by the domain gap between them. SDC instead bridges them in the frequency domain across N = 4 blocks, each comprising two symmetric cross-attention modules. Within a module, both streams are transformed by a three-dimensional rFFT, and an affinity is formed.

$$
\begin{array} { r } { \mathcal { M } = { \mathrm { S o f t m a x } } \Big ( \frac { 1 } { \tau } \big ( \mathcal { F } ( c ^ { 0 } ) W _ { q } \big ) ^ { \top } \big ( \mathcal { F } ( e ) W _ { k } \big ) \Big ) , } \end{array}\tag{3}
$$

where $\mathcal F ( \cdot )$ denotes the rFFT, $W _ { q }$ and $W _ { k }$ the learnable query and key projections applied in Fourier space, $\tau \_ { \mathrm { ~ a ~ } }$ learnable temperature, and M the resulting affinity map. The affinity is formed over feature channels rather than spatial tokens and is realized as a linear-complexity cross-covariance attention over feature channels [33]. It renders the operation tractable in the $1 6 ^ { 3 }$ latent resolution, whereas the channel formulation reduces this to $\mathcal { O } ( N C ^ { 2 } )$ for N tokens and $C$ channels while preserving global mixing.

d) Timestep-adaptive band-pass filter: The informative frequency band varies with the noise level: at high t only coarse structure survives corruption, whereas at low t fine boundary detail becomes recoverable. SB therefore gates the attended spectrum with a timestep-conditioned filter. To keep the gate smooth over a continuous frequency range, it is produced by a coordinate network rather than free perbin weights: each rFFT bin is described by its normalized radial frequency $r ~ \in ~ [ 0 , 1 ]$ , lifted with sinusoidal Fourier features [34] and mapped through $R = 6$ FiLM-modulated blocks [35] and a sigmoid to

$$
\mathcal { M } ^ { \prime } = g ( r , t ) \odot \big ( \mathcal { M } ( \mathcal { F } ( c ^ { 0 } ) W _ { v } ) \big ) ,\tag{4}
$$

where $g ( r , t ) ~ \in ~ ( 0 , 1 )$ is the learned gate, $W _ { v }$ the value projection, ⊙ element-wise multiplication, and $\mathcal { M } ^ { \prime }$ the gated spectrum. Shared across channels, the gate acts as a spectral modulation whose passband tracks the noise level of step t. The gated spectrum is returned to Euclidean space by the inverse rFFT, added residually to the query stream, and refined by a channel MLP; the symmetric module reverses the two streams to form the next block’s input, and after $N _ { b }$ blocks the refined noise embedding is read by the ϵ prediction head.

e) Conditioning injection: The clean latent $\hat { z } _ { 0 }$ recovered from the denoiser is injected into the bottleneck through a $1 ^ { 3 }$ convolution ${ \mathcal { P } } _ { \mathrm { { : } } }$

$$
\tilde { c } ^ { 0 } = c ^ { 0 } + \mathcal { P } ( \hat { z } _ { 0 } ) ,\tag{5}
$$

so that the diffusion output acts as a correction upon the semantic condition rather than as a replacement for it. The refined bottleneck $\tilde { c } ^ { 0 }$ is passed to ME<sup>2</sup>FB and thereafter decoded to the segmentation.

3) Multiscale Expert Enhanced Feature Block: Encoder stages at different depths encode qualitatively different information: shallow, high-resolution stages preserve local anatomical detail, whereas deeper stages capture global context. The standard SegResNet-VAE, however, constructs its bottleneck from the deepest stage alone and admits the shallower stages only as passive skip connections, so that the evidence available at the point of fusion is restricted to a single scale. ${ \bf M E ^ { 2 } F B }$ removes this restriction by combining features across stages through a mixture of experts with expert-choice routing. Each of the n intermediate stages is projected to a common width d by a $1 ^ { 3 }$ convolution and pooled to the bottleneck resolution, yielding at every bottleneck voxel a group of stage descriptors $\boldsymbol { X } \in \mathbb { R } ^ { n \times d }$ . Expert preferences are obtained from a gating function perturbed by learned noise,

$$
\begin{array} { r } { H = g ( X , \theta ) ^ { \top } + R _ { \mathrm { n o i s e } } , ~ } \\ { S _ { I } = \mathrm { S o f t m a x } ( H [ \mathrm { T o p K } ( H , k ) ] ) . } \end{array}\tag{6}
$$

where $g ( X , \theta )$ produces the expert-preference logits and $R _ { \mathrm { n o i s e } } = \varepsilon \odot$ soft $\mathrm { p l u s } ( X _ { \mathrm { n o i s e } } )$ promotes exploration and stabilizes optimization, and is disabled at inference so that routing is deterministic. Routing follows the expert-choice convention rather than each descriptor selecting its experts; each expert selects its own k descriptors, with $k = \lfloor B \cdot V \cdot n / E \rfloor$ for $V ~ = ~ 1 6 ^ { 3 }$ bottleneck voxels and E experts. Every expert therefore processes an identical share of the token set, which guarantees balanced utilization by construction and precludes expert collapse without recourse to an auxiliary balancing loss. Each expert is realized as a two-layer perceptron with expansion factor four,

$$
\hat { X } _ { I } = \operatorname { R e L U } ( X _ { I } W _ { 1 } ) W _ { 2 } , \qquad W _ { 1 } \in \mathbb { R } ^ { d \times 4 d } , \ W _ { 2 } \in \mathbb { R } ^ { 4 d \times d }\tag{7}
$$

and a descriptor selected by several experts is recombined as $\begin{array} { r } { X _ { \mathrm { f i n a l } } = \sum _ { I } S _ { I } \hat { X } _ { I } } \end{array}$ , while a residual connection carries forward unselected descriptors. The n enhanced stage descriptors at each voxel are collapsed by a learnable softmax-weighted sum, refined by self-attention over the $1 6 ^ { 3 }$ spatial tokens to inject global context, and mapped back to the bottleneck width by a $1 ^ { 3 }$ convolutional neck. The result is added residually to the bottleneck, $z = \mathrm { R e L U } ( z _ { \mathrm { b t l n } } + z _ { \mathrm { e n h } } )$

## D. Evaluation Metrics

Following the BraTS convention, performance is reported over four regions derived from the predicted label map: enhancing tumor (ET), tumor core $( \mathrm { T C } \ = \ \mathrm { E T } \ \cup \ \mathrm { N C R } )$ whole tumor $( \mathbf { W } \mathbf { T } \ = \ \mathrm { E } \mathbf { T } \ \cup \ \mathbf { N } \mathbf { C } \mathbf { R } \ \cup \ \mathrm { E } \mathbf { D } )$ , and resection cavity (RC), the last specific to the post-treatment setting and evaluated separately. Each region is assessed with the Dice Similarity Coefficient (DSC) and Intersection over Union (IoU), which measure volumetric overlap; sensitivity (Sn) and specificity (Sp), the true-positive and true-negative rates; and the 95th-percentile Hausdorff Distance (HD95), which measures boundary agreement and is robust to isolated surface outliers [6]. Higher DSC, IoU, Sn, and Sp indicate better performance, whereas HD95 is reported in millimeters, with lower values preferred. All metrics are computed per case and averaged over the held-out test set.

## IV. EXPERIMENTAL SETUP

## A. Training configuration

All models are optimized using AdamW with a peak learning rate of $1 \times 1 0 ^ { - 4 }$ [36] and a minimum learning rate of $1 \times 1 0 ^ { - 6 }$ , following a cosine annealing schedule. Models are trained for a maximum of 150 epochs with early stopping (patience of 40 epochs, monitored on validation Dice). Training uses a batch size of 1 per GPU under automatic mixed-precision (AMP) with FP16, distributed across 4 GPUs (NVIDIA A6000/A100 of Rivanna/Afton high-performance computing system) using PyTorch Distributed Data Parallel (DDP). A global random seed of 42 is used, with each distributed rank offset by its rank index (seed+rank) to ensure reproducible yet decorrelated data sampling across processes.

## B. Loss configuration

To handle the complexity of different regions and their imbalance ratios, all models are trained with a compound loss

TABLE I  
PERFORMANCE COMPARISON OF SINGLE SEGMENTATION MODELS ON THE TEST SET FOR FOUR REGIONS (ET, TC, WT, RC).
<table><tr><td rowspan="2">Models</td><td colspan="5">Dice (%) ↑</td><td colspan="5">IoU (%) ↑</td><td colspan="5">HD95 (mm) ↓</td></tr><tr><td>ET</td><td>TC</td><td>WT</td><td>NETC</td><td>RC</td><td>ET</td><td>TC</td><td>WT</td><td>NETC</td><td>RC</td><td>ET</td><td>TC</td><td>WT</td><td>NETC</td><td>RC</td></tr><tr><td>SAM</td><td>55.12</td><td>65.46</td><td>74.10</td><td>60.57</td><td>51.31</td><td>46.54</td><td>52.50</td><td>60.40</td><td>58.10</td><td>42.26</td><td>9.47</td><td>16.06</td><td>15.55</td><td>5.78</td><td>18.61</td></tr><tr><td>UNETR</td><td>58.69</td><td>78.39</td><td>87.68</td><td>51.64</td><td>60.85</td><td>50.77</td><td>68.98</td><td>79.41</td><td>47.56</td><td>52.42</td><td>6.93</td><td>5.96</td><td>4.46</td><td>5.49</td><td>8.99</td></tr><tr><td>AHNet</td><td>70.16</td><td>79.58</td><td>87.07</td><td>64.75</td><td>63.93</td><td>62.20</td><td>70.21</td><td>78.61</td><td>60.44</td><td>55.44</td><td>5.04</td><td>4.82</td><td>3.68</td><td>4.36</td><td>7.42</td></tr><tr><td>UNet</td><td>63.69</td><td>81.84</td><td>89.65</td><td>65.21</td><td>67.10</td><td>55.81</td><td>73.31</td><td>82.24</td><td>60.83</td><td>58.81</td><td>4.78</td><td>4.28</td><td>3.33</td><td>3.99</td><td>7.71</td></tr><tr><td>MedNeXt</td><td>75.11</td><td>84.33</td><td>89.19</td><td>65.73</td><td>67.23</td><td>67.58</td><td>76.48</td><td>81.88</td><td>61.36</td><td>59.02</td><td>3.58</td><td>3.55</td><td>2.86</td><td>3.85</td><td>6.44</td></tr><tr><td>SwinUNETR</td><td>76.72</td><td>85.83</td><td>90.77</td><td>69.78</td><td>73.70</td><td>69.62</td><td>78.22</td><td>83.91</td><td>65.76</td><td>65.68</td><td>3.62</td><td>3.25</td><td>2.51</td><td>3.62</td><td>5.49</td></tr><tr><td>SegResNetVAE</td><td>77.92</td><td>86.70</td><td>90.65</td><td>69.12</td><td>73.10</td><td>70.86</td><td>79.18</td><td>83.86</td><td>65.03</td><td>64.97</td><td>3.59</td><td>3.33</td><td>2.45</td><td>3.98</td><td>5.91</td></tr><tr><td>ME²FDiff</td><td>78.75</td><td>87.14</td><td>91.23</td><td>70.08</td><td>74.20</td><td>71.60</td><td>79.71</td><td>84.52</td><td>65.92</td><td>66.62</td><td>3.56</td><td>3.24</td><td>2.13</td><td>3.87</td><td>5.47</td></tr></table>

Ground Truth  
Ours (ME²FDiff)  
SegResNetVAE  
MedNeXt  
![](images/f69fe3123c63a3986e4377db622ca87af96cb64248feb1d51d558df466c1e279.jpg)  
SwinUNETR  
UNet  
Fig. 3. Qualitative comparison among the predicted masks for baseline models with ground truth.

TABLE II  
PERFORMANCE COMPARISON OF ENSEMBLE FUSION STRATEGIES WITH THREE MODELS (SEGRESNETVAE + SWINUNETR + MEDNEXT) ON THE TEST SET FOR FOUR REGIONS (ET, TC, WT, RC) WITH CORRESPONDING TRAINING TIME (TT) PER EPOCH.
<table><tr><td>Models</td><td>TT (mm:ss) ↓</td><td colspan="5">Dice (%) ↑</td><td colspan="5">IoU (%) ↑</td><td colspan="5">HD95 (mm) ↓</td></tr><tr><td></td><td></td><td>ET</td><td>TC</td><td>WT</td><td>NETC</td><td>RC</td><td>ET</td><td>TC</td><td>WT</td><td>NETC</td><td>RC</td><td>ET</td><td>TC</td><td>WT</td><td>NETC</td><td>RC</td></tr><tr><td>Dynamic Weighting</td><td>26:89</td><td>67.76</td><td>84.34</td><td>90.66</td><td>70.01</td><td>68.29</td><td>60.41</td><td>76.65</td><td>83.76</td><td>63.48</td><td>60.09</td><td>4.18</td><td>3.01</td><td>3.01</td><td>3.53</td><td>7.06</td></tr><tr><td>Cascade Fusion</td><td>30:56</td><td>78.68</td><td>85.11</td><td>90.77</td><td>65.19</td><td>65.54</td><td>71.65</td><td>77.53</td><td>83.91</td><td>60.94</td><td>57.48</td><td>3.04</td><td>3.15</td><td>2.52</td><td>3.58</td><td>6.88</td></tr><tr><td>MoE</td><td>39:87</td><td>77.59</td><td>86.45</td><td>91.18</td><td>68.79</td><td>73.31</td><td>70.63</td><td>79.09</td><td>83.59</td><td>64.22</td><td>66.36</td><td>3.20</td><td>2.92</td><td>2.17</td><td>3.34</td><td>5.21</td></tr><tr><td>ME2FDiff</td><td>10:92</td><td>78.75</td><td>87.14</td><td>91.23</td><td>70.08</td><td>74.20</td><td>71.60</td><td>79.71</td><td>84.52</td><td>65.92</td><td>66.62</td><td>3.56</td><td>3.24</td><td>2.13</td><td>3.87</td><td>5.47</td></tr></table>

![](images/179292f8355c309b937e85fb655e04f12dd4233c8a712cd0f62bef6af0d2638f.jpg)  
Fig. 4. Qualitative comparison among the predicted mask for ensemble models with ground truth.

combining three loss functions, i.e., Tversky [37], Dice [38], and Focal terms [39].

$$
\mathcal { L } _ { \mathrm { t o t a l } } = 0 . 4 \mathcal { L } _ { \mathrm { T v e r s k y } } + 0 . 3 \mathcal { L } _ { \mathrm { D i c e } } + 0 . 3 \mathcal { L } _ { \mathrm { F o c a l } } .\tag{8}
$$

where Dice provides stable overlap optimization, Tversky accounts for recall bias in small critical regions, and Focal adds voxel-level attention to hard boundaries. When combined, they operate at different granularities and cover each other’s gaps. The class weighting encodes the same priority. Rather than strict inverse-frequency weighting, which destabilizes training when a region is absent from a volume, weights are assigned in inverse rank of expected sub-region volume and confined to a narrow range: 0.5 for background, 1.0 for the largest tumor region (SNFH), 2.0 for NETC, and 2.5 for the small ET and RC. A false-negative penalty $\beta = 0 . 8$ is applied to the small ET and RC regions, biasing the Tversky term toward recall where a missed lesion is costlier than an enlarged boundary, and the Focal term uses $\gamma = 2 . 0$ and $\alpha = 0 . 2 5$ , concentrating the gradient on the ambiguous voxels that dominate posttreatment boundaries.

## V. RESULTS

We evaluate the proposed model against progressively stronger points of comparison. Section V-A reports performance against individually trained segmentation architectures; Section V-B against ensemble fusion strategies that aggregate several such architectures and therefore constitute the more demanding baseline; and Section V-C against recent state-ofthe-art methods, considering accuracy alongside model size. It also establishes the statistical reliability of the observed differences, and Section V-D isolates the contribution of each architectural component. All results are reported on the heldout test set, with Dice, IoU, Sensitivity, and HD95 computed independently for the four tumor sub-regions (ET, TC, WT, and RC) as defined in Section III-D.

## A. Comparison with Single Model Baselines

Table I reports test-set performance across the four tumor sub-regions. The SAM-based foundation models perform worst (SAM: 61.50 mean Dice), reflecting the difficulty of transferring 2D natural-image pretraining to volumetric MRI, while SwinUNETR (81.76) and SegResNetVAE (82.09) are the strongest baselines. The proposed model attains the highest Dice and IoU on every sub-region, improving over SegRes-NetVAE by 0.83, 0.44, 0.58, and 1.10 Dice points on ET, TC, WT, and RC, and raising mean Dice from 82.09 to 82.83. The largest gain is on RC, the hardest region for all baselines and the one unique to the post-treatment setting. It also achieves the best mean HD95 (3.60 mm), improving on SegResNetVAE by 0.03-0.44 mm across the four regions, with SwinUNETR the only close competitor on boundary localization (TC 3.25 vs. 3.24 mm, RC 5.49 vs. 5.47 mm). As the overlap gains are consistent across all regions and both overlap metrics whereas the HD95 differences are small and inconsistent in direction, the improvement is attributable principally to more accurate delineation of region extent rather than sharper boundary placement. The qualitative performance comparison for single models is illustrated in Figure 3.

## B. Comparison with Ensemble Baselines

Table II compares the proposed model against four ensemble fusion strategies, each aggregating the same three best-performing architectures (SegResNetVAE, SwinUNETR, MedNeXt) shown in Table I. Among the ensembles, learned gating is clearly superior to the alternatives: MoE attains the best mean Dice (82.13), ahead of Cascade Fusion (80.03) and Dynamic Weighting (77.76). The margin is most pronounced on RC, where MoE reaches 73.31 Dice against 65.54 for Cascade, evidence that confidence-derived and sequentially stitched fusion rules both degrade sharply on the hardest sub-region, whereas a learned gate does not. Crucially, the proposed model surpasses the strongest ensemble on both overlap metrics while requiring only a single forward pass. It exceeds MoE on mean Dice (82.83 vs. 82.13) and mean IoU, winning three of four sub-regions on each; the remaining case (WT for IoU) differs by only 0.07 points and is effectively tied. That a single network matches and exceeds a three-expert ensemble on volumetric overlap at approximately one-third of its inference cost. It indicates that the expert specialization such ensembles achieve through external fusion can instead be internalized within a single architecture. Each ensemble must train three separate expert networks together with its fusion stage, taking between 26.89 and 39.87 minutes (dynamic 26.89, cascade 30.56, MoE 39.87), whereas the proposed model is a single network and trains in 10.92 minutes-a 2.5-3.7× reduction. The saving carries over to deployment: inference requires a single forward pass rather than the three needed to evaluate every expert. The proposed model thus trades a small amount of boundary regularity for stronger region delineation at a fraction of the training and inference cost, a favorable trade-off where accurate estimation of tumor extent, rather than sub-millimeter boundary placement, drives clinical utility. The qualitative performance comparison for ensemble models is illustrated in Figure 4.

TABLE III  
PERFORMANCE COMPARISON OF EXISTING SOTA SEGMENTATION MODELS FOR THE WHOLE TUMOR ON TEST TEST. PARAMETERS ARE REPORTED AS MILLIONS (M) AND DICE, IOU, ACC, SN, SP ARE REPORTED AS PERCENTAGES (%).
<table><tr><td>Model</td><td>params</td><td>Dice</td><td>IOU</td><td>Acc</td><td>Sn</td><td>Sp</td></tr><tr><td>MoE-SAM</td><td>134.29</td><td>79.74</td><td>67.60</td><td>99.20</td><td>80.47</td><td>99.58</td></tr><tr><td>SegFormer</td><td>42.67</td><td>85.14</td><td>76.11</td><td>99.48</td><td>85.17</td><td>99.75</td></tr><tr><td>TransUNet</td><td>132.39</td><td>90.08</td><td>82.87</td><td>99.63</td><td>90.64</td><td>99.77</td></tr><tr><td>MedFormer</td><td>39.59</td><td>90.57</td><td>83.76</td><td>99.67</td><td>89.95</td><td>99.85</td></tr><tr><td>UMamba</td><td>31.84</td><td>90.92</td><td>84.14</td><td>99.66</td><td>91.03</td><td>99.83</td></tr><tr><td>ME2FDiff</td><td>28.82</td><td>91.23</td><td>84.65</td><td>99.67</td><td>91.24</td><td>99.83</td></tr></table>

![](images/57a92cca42169c98d71c3dfa842886885d35c2230bb80ce904f659e97fcd9563.jpg)  
Fig. 5. Compound Dice of the mean of the three BraTS tumor sub-regions (whole tumor WT, tumor core TC, enhancing tumor ET) on the held-out test set (163 cases). Each marker is the per-method mean and the horizontal bar its 95% bootstrap confidence interval (5,000 resamples); the dashed vertical line marks our mean for reference. All competing methods were retrained on the identical data split and training criteria, so differences reflect the model, not the pipeline.

## C. Comparison with State-of-the-Art Methods

We compare the proposed model against recent state-ofthe-art segmentation architectures. To ensure that observed differences reflect the model rather than the training pipeline, all competing methods were retrained under identical data splits and training criteria. Table III reports whole-tumor performance alongside parameter count, and Figure. 5 shows the compound Dice over the three BraTS sub-regions (WT, TC, ET) on the held-out test set (163 cases), with 95% bootstrap confidence intervals. The proposed model attains the best Dice (91.23), IoU (84.65), and Sensitivity (91.24) of all methods evaluated, while using the fewest parameters (28.82M). It exceeds the strongest competing method, UMamba, by 0.31 Dice and 0.51 IoU with 9.5% fewer parameters and surpasses the transformer-based TransUNet by 1.15 Dice using 4.6× fewer parameters.

The margin over MoE-SAM is largest (+11.49 Dice at 4.7× fewer parameters), consistent with the limited transferability of natural-image foundation models to volumetric MRI. Accuracy and specificity are effectively saturated across all competitive methods and therefore carry little discriminative signal. Figure. 5 confirms that this ordering is preserved when performance is aggregated across all three tumor sub-regions rather than assessed on the whole tumor alone. Across the comparison, accuracy and model size move together in the proposed model’s favor: it records the highest overlap accuracy while carrying the fewest parameters of any method evaluated. This is notable given the prevailing trend toward increasingly large architectures: the two largest models in the comparison, MoE-SAM (134.29M) and TransUNet (132.39M), are both outperformed by a model roughly one-fifth their size, supporting the view that architectural specialization, rather than raw capacity, is the principal determinant of performance on this task.

TABLE IV  
COMPONENT-WISE OVERALL CONTRIBUTION TO THE WHOLE TUMOR REGION FOR OUR PROPOSED METHOD. DICE, IOU, AND SN ARE REPORTED AS PERCENTAGES (%), AND HD95 IS REPORTED AS MILIMETERS(MM)
<table><tr><td>Configuration</td><td>Dice</td><td>IOU</td><td>Sn</td><td>HD95</td></tr><tr><td>SegResNetVAE (Baseline)</td><td>90.66</td><td>83.86</td><td>89.74</td><td>2.45</td></tr><tr><td>+ Multiscale Expert Enhanced Fea- ture Block (ME2FB)</td><td>90.81</td><td>84.07</td><td>89.91</td><td>2.42</td></tr><tr><td>+ Attention-Conditioned Spectral Diffusion (ACSDiff)</td><td>91.23</td><td>84.52</td><td>91.80</td><td>2.13</td></tr></table>

## D. Component Contribution Analysis

To isolate the contribution of each architectural component, we add modules successively to the SegResNetVAE baseline. Table IV reports whole-tumor (WT) performance, and Figure. 6 decomposes the corresponding gain on enhancing tumor (ET) over five different seeds. We report ET separately as it delineates the actively proliferating tumor that defines the treatment target volume and the primary criterion for response assessment, and is the hardest region to segment in the posttreatment setting, where genuine tumor enhancement must be distinguished from treatment-related enhancement. On WT, each component improves every metric monotonically. The ME<sup>2</sup>FB raises Dice from 90.66 to 90.81 and the ACSDiff by a further 0.42 point, giving 91.23 Dice and 84.52 IoU. The gain is most pronounced in Sensitivity, which rises by 2.06 points overall (89.74 → 91.80), indicating that the added components principally recover tumor tissue missed by the baseline, while HD95 falls from 2.45 to 2.13 mm.

![](images/f5848f2de33987a6e7cceb01accec3caba31a1a7a0476a79fc0f446e237abf50.jpg)  
Fig. 6. Component-wise contribution to Enhancing-Tumor (ET) Dice. Waterfall decomposition over the SegResNetVAE baseline. Bar heights are the cumulative gain over baseline; the bold value above each bar is the incremental improvement of that step, averaged over the training seeds. Whiskers are 95% confidence intervals of the mean and colored markers show individual seeds

The same ordering holds on ET (Figure. 6). From an ET Dice of 0.7684, ME<sup>2</sup>FB contributes +0.0087, and the ACSDiff reported, inclusive of SPG, is consistent with Table IV, a further +0.0097, reaching 0.7869: a cumulative gain of +0.0185, or +2.41% relative to the baseline. The per-seed values indicate that both components improve ET Dice on average rather than uniformly across seeds.

## VI. CONCLUSION

Ensembling remains the default answer to architecturedependent variability in tumor segmentation, at the price of several networks and a combination rule fixed in advance. ME<sup>2</sup>FDiff shows the price is avoidable: routing applied across encoder scales supplies the complementarity an ensemble buys with extra models, and refinement applied to a mask latent supplies the correction step a single forward pass otherwise lacks. On BraTS 2024 post-treatment glioma, the result is a model that is simultaneously the most accurate and the smallest we evaluated. Where it still falls short is instructive. The resection cavity, the new sub-region introduced by the posttreatment setting, resists every method tested, ours included, suggesting the remaining difficulty is one of training signals rather than architecture. Because ACSDiff’s reverse process is stochastic, sampling it repeatedly yields multiple hypotheses at no architectural cost, an avenue we leave open alongside extension to the wider BraTS 2024 cluster.

## REFERENCES

[1] M. Price, C. Ballard, J. Benedetti, C. Neff, G. Cioffi, K. A. Waite, C. Kruchko, J. S. Barnholtz-Sloan, and Q. T. Ostrom, “Cbtrus statistical report: primary brain and other central nervous system tumors diagnosed in the united states in 2017–2021,” Neuro-oncology, vol. 26, no. Suppl 6, p. vi1, 2024.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, Y. Burren, N. Porz, J. Slotboom, R. Wiest et al., “The multimodal brain tumor image segmentation benchmark (brats),” IEEE transactions on medical imaging, vol. 34, no. 10, pp. 1993–2024, 2014.

[3] S. Baig, I. Vidic, G. M. Mastorakos, R. X. Smith, N. White, S. Bash, A. M. Dale, C. R. McDonald, T. Beaumont, T. M. Seibert et al., “Segmentation of pre-and posttreatment diffuse glioma tissue subregions including resection cavities,” Neuro-Oncology Advances, vol. 6, no. 1, p. vdae140, 2024.

[4] S. Ranjbar, K. W. Singleton, L. Curtin, L. Paulson, K. Clark-Swanson, A. Hawkins-Daarud, J. R. Mitchell, P. R. Jackson, and K. R. Swanson, “Towards longitudinal glioma segmentation: Evaluating combined preand post-treatment mri training data for automated tumor segmentation using nnu-net,” medRxiv, 2023.

[5] M. C. de Verdier, R. Saluja, L. Gagnon, D. LaBella, U. Baid, N. H. Tahon, M. Foltyn-Dumitru, J. Zhang, M. Alafif, S. Baig et al., “The 2024 brain tumor segmentation (brats) challenge: glioma segmentation on post-treatment mri,” arXiv preprint arXiv:2405.18368, 2024.

[6] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, “Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images,” in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2021), ser. Lecture Notes in Computer Science, vol. 12962. Springer, Cham, 2022, pp. 272–284.

[7] W. Yao, J. Bai, W. Liao, Y. Chen, M. Liu, and Y. Xie, “From cnn to transformer: A review of medical image segmentation models,” Journal ofImaging Informatics in Medicine, vol. 37, no. 4, pp. 1529–1547, 2024.

[8] F. Isensee, P. F. Jaeger, P. Full, P. Vollmuth, and K. H. Maier-Hein, “nnunet for brain tumor segmentation,” arXiv preprint arXiv:2011.00848, 2020.

[9] T. G. Dietterich, “Ensemble methods in machine learning,” in Multiple Classifier Systems. Berlin, Heidelberg: Springer Berlin Heidelberg, 2000, pp. 1–15.

[10] Y. Yuan, W. Li, R. Song, Z. Li, H. Liu, X. Kong, Z. Dong, L. Du, Z. Sun, W. Zhang et al., “Discrete diffusion models: A unified framework from tokenization to generation,” arXiv preprint arXiv:2607.13431, 2026.

[11] P. Moeskops, M. A. Viergever, A. M. Mendrik, L. S. De Vries, M. J. Benders, and I. Isgum, “Automatic segmentation of mr brain images withˇ a convolutional neural network,” IEEE transactions on medical imaging, vol. 35, no. 5, pp. 1252–1261, 2016.

[12] J. Liang et al., “Transconver: transformer and convolution parallel network for developing automatic brain tumor segmentation in mri images,” Quantitative Imaging in Medicine and Surgery, vol. 12, no. 4, p. 2397, 2022.

[13] M. Kharaji, H. Abbasi, Y. Orouskhani, M. Shomalzadeh, F. Kazemi, and M. Orouskhani, “Brain tumor segmentation with advanced nnunet: pediatrics and adults tumors,” Neuroscience Informatics, p. 100156, 2024.

[14] R. Thillaikkarasi and S. Saravanan, “An enhancement of deep learning algorithm for brain tumor segmentation using kernel based cnn with m-svm,” Journal of medical systems, vol. 43, no. 4, p. 84, 2019.

[15] Y. Y. Wang, O. V. Glinskii, F. Bunyak, and K. Palaniappan, “Ensemble of deep learning cascades for segmentation of blood vessels in confocal microscopy images,” in 2021 IEEE Applied Imagery Pattern Recognition Workshop (AIPR). IEEE, 2021, pp. 1–7.

[16] D. Maji, P. Sigedar, and M. Singh, “Attention res-unet with guided decoder for semantic segmentation of brain tumors,” Biomedical Signal Processing and Control, vol. 71, p. 103077, 2022.

[17] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI), ser. Lecture Notes in Computer Science, vol. 9351. Springer, 2015, pp. 234–241.

[18] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, “Transunet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.

[19] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 12 077–12 090.

[20] Y. Gao, M. Zhou, D. Liu, Z. Yan, S. Zhang, and D. N. Metaxas, “A data-scalable transformer for medical image segmentation: Architecture, model efficiency, and benchmark,” in International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2022, pp. 588–598.

[21] J. Ma, F. Li, and B. Wang, “U-mamba: Enhancing long-range dependency for biomedical image segmentation,” arXiv preprint arXiv:2401.04722, 2024.

[22] R. Li, L. Wu, J. Gu, Q. Xu, W. Chen, X. Cai, and J. Bu, “ MoE-SAM: Enhancing SAM for Medical Image Segmentation with Mixtureof-Experts ,” in proceedings ofMedical Image Computing and Computer Assisted Intervention – MICCAI 2025, vol. LNCS 15961. Springer Nature Switzerland, September 2025.

[23] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Gir-´ shick, “Segment anything,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[24] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 6840–6851.

[25] A. Myronenko, “3d mri brain tumor segmentation using autoencoder regularization,” in International MICCAI brainlesion workshop. Springer, 2018, pp. 311–320.

[26] B. Kundu, Z. Qureshi, R. Chavan, and A. Rahman, “Ai-driven parametric fdg brain pet and brain tumor segmentation for multi-modal convolution neural network to differentiate tumor progression from treatment effect in gbm,” Advances in Cancer Research, vol. 168, pp. 131–157, 2025.

[27] S. Liu, D. Xu, S. K. Zhou, T. Mertelmeier, J. Wicklein, A. Jerebko, S. Grbic, O. Pauly, W. Cai, and D. Comaniciu, “3d anisotropic hybrid network: Transferring convolutional features from 2d images to 3d anisotropic volumes,” 2017. [Online]. Available: https://arxiv.org/abs/ 1711.08580

[28] S. Roy, G. Koehler, C. Ulrich, M. Baumgartner, J. Petersen, F. Isensee, P. F. Jager, and K. H. Maier-Hein, “Mednext: Transformer-driven scaling¨ of convnets for medical image segmentation,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, ser. Lecture Notes in Computer Science, vol. 14224. Springer, 2023, pp. 405–415.

[29] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, “Unetr: Transformers for 3d medical image segmentation,” in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 574–584.

[30] L. Guan and X. Yuan, “Dynamic weighting and boundary-aware active domain adaptation for semantic segmentation in autonomous driving environment,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, pp. 18 461–18 471, 2024.

[31] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 6840–6851.

[32] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional block attention module,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 3–19.

[33] A. Ali, H. Touvron, M. Caron, P. Bojanowski, M. Douze, A. Joulin, I. Laptev, N. Neverova, G. Synnaeve, J. Verbeek et al., “Xcit: Crosscovariance image transformers,” Advances in neural information processing systems, vol. 34, pp. 20 014–20 027, 2021.

[34] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” Advances in neural information processing systems, vol. 33, pp. 7537– 7547, 2020.

[35] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in AAAI Conference on Artificial Intelligence, 2018.

[36] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019.

[37] S. S. M. Salehi, D. Erdogmus, and A. Gholipour, “Tversky loss function for image segmentation using 3d fully convolutional deep networks,” in Machine Learning in Medical Imaging (MLMI). Springer, 2017, pp. 379–387.

[38] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in Proceedings of the Fourth International Conference on 3D Vision (3DV). IEEE, 2016, pp. 565–571.

[39] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988.