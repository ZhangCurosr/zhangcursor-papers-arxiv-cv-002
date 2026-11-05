# LUNGMEND: A HEALTHY-ONLY LATENT DIFFUSION SHAPE PRIOR FOR LUNG MASK CORRECTION IN SEVERE TUBERCULOSIS CT

Rezkellah Noureddine Khiati<sup>1</sup>, Pierre-Yves Brillet<sup>2</sup>, and Catalin Fetita<sup>1</sup>

<sup>1</sup>SAMOVAR, Tel´ ecom SudParis, Institut Polytechnique de Paris, Palaiseau, France ´ <sup>2</sup>Avicenne Hospital, AP-HP, Bobigny, France

## ABSTRACT

Lung segmentation models such as R231 are widely used, but they fail on severe tuberculosis (TB) CT: cavities and consolidations look like non-lung tissue and are left out of the mask. We present LungMend, a post-processing model that corrects such masks using only healthy anatomy. A latent diffusion model, trained on healthy lungs with synthetically corrupted masks, learns to recover the complete lung from the faulty mask and the bone anatomy, without any pathological data or CT intensities. LungMend recovers cavities and consolidations that R231 misses and improves accuracy on expertannotated slices. Code: noureddinekhiati/LungMend

Index Terms— Lung segmentation, latent diffusion, shape prior

## 1. INTRODUCTION

Lung segmentation is the first step of most quantitative CT pipelines. The most widely used model is the open-source U-Net R231 [1], the de facto standard in 3D Slicer. It was trained on a large and diverse set of 231 CT scans that includes consolidation and several other pathological patterns. Yet it still fails on TB, where large cavities and extensive consolidations have soft-tissue density and are excluded from the mask. We observe three recurrent failure modes: (i) loss of large contiguous regions, (ii) erosion along the chest wall where disease is subpleural, and (iii) slice-to-slice inconsistency, since R231 works in 2D and its 3D masks are fragmented along the craniocaudal axis.

Related work. Dedicated pathological lung segmentation combines intensity, texture and shape cues [2, 3, 4], but it needs pathological annotations, which are costly and never exhaustive. Shape-based post-processing, from statistical shape models [5] to autoencoder priors [6, 7, 8], projects a mask onto a single plausible shape and struggles when large regions are missing. Diffusion models [9, 10] have been used to refine segmentations [11, 12, 13], but they condition on image intensities, which are what misleads the segmenter in diseased lungs. Latent diffusion [14] makes 3D medical generation tractable [15, 16], and shape diffusion enables 3D completion in vision [17, 18, 19, 20]. To our knowledge, no prior work uses a generative shape prior learned from healthy anatomy to correct pathological lung masks.

Contributions.

• New formulation. We cast lung mask correction as shape completion with a prior learned from healthy anatomy only, so no pathological annotations are needed and the correction does not depend on the image appearance that causes R231 to fail.

• Training without pathology. We introduce a corruption model that reproduces the failure modes of texture-based segmenters, which makes it possible to train a correction model on healthy lungs alone.

• Disease-independent anchoring. We show that bone anatomy gives a reliable cue for completing large missing regions, since the rib cage is not affected by parenchymal disease.

• Validation on real TB. We evaluate on expert-annotated TB cases and show improvements over R231 and shapeprior baselines, including 3D consistency.

## 2. METHOD

Given an incomplete lung mask $m ^ { \prime }$ (e.g. the R231 output) and a bone mask b, LungMend estimates the complete lung mask mˆ . A model that has only seen healthy lungs knows what a lung should look like, so it can fill the regions where disease made the mask deviate from that shape. To do this, the lung is represented as a smooth shape rather than as image intensities, which are what misleads the segmenter, and the missing region is anchored by the bone anatomy, which disease does not alter. LungMend is trained in two stages, both on healthy lungs only (Fig. 1): Stage 1 learns a compact space of healthy shapes, and Stage 2 learns to recover a complete lung from a corrupted one. No pathological scan, mask or CT intensity is used.

Data and shape representation. Both stages use the same 57 healthy post-COVID scans from COVALUNG (CO-VALUNG dataset, Caen University Hospital, France). Each lung is cropped (bounding box +15 mm), cleaned (two largest components, hole filling) and resampled to $1 2 8 ^ { 3 }$ ; the bone mask gets the same crop. Instead of binary masks, we use the signed distance transform (SDT),

![](images/f8be06ec856bed6858f077f341a870be1f43ccb32e920da43b81fa34a8867013.jpg)  
Fig. 1. Overview of LungMend. Training (Stage 2): the SDT of a healthy lung mask is encoded by the frozen VAE encoder $E ^ { * }$ (pre-trained in Stage 1) into $\mathbf { z } _ { 0 }$ . A pathology-mimicking corruption of the mask gives $\mathbf { z } _ { \mathrm { c o r r } } ,$ and bone occupancy and distance give $\mathbf { c } _ { \mathrm { b o n e } }$ . The U-Net $\mathrm { V } _ { \theta }$ takes the noisy latent $\mathbf { z } _ { t }$ and $[ \mathbf { z } _ { \mathrm { c o r r } } , \mathbf { c } _ { \mathrm { b o n e } } ]$ (18 channels, $1 6 ^ { 3 } )$ and predicts $v _ { \theta } .$ . Inference: N samples are decoded by $D ^ { * }$ and averaged into the corrected mask and an uncertainty map. CFG denotes: classifier-free guidance.

$$
\mathrm { S D T } ( m ) = \mathrm { c l i p } \big ( d ( m ) - d ( \bar { m } ) , - C , C \big ) / C ,
$$

with $d ( \cdot )$ the Euclidean distance transform, m¯ the complement of m and $C { = } 2 0 \mathrm { m m }$ . We write $s = \mathrm { S D T } ( m )$ for a complete mask and $s ^ { \prime } = \mathrm { S D T } ( m ^ { \prime } )$ for an incomplete one. Each voxel stores its signed distance to the lung surface (positive inside), and a mask is recovered as $m = [ s > 0 ]$ . The SDT is smooth, so it can be compressed and denoised without the isolated-voxel artifacts of binary masks.

## 2.1. Stage 1: shape autoencoder

A 3D VAE [21] encodes $s ( 1 \times 1 2 8 ^ { 3 } )$ into a latent ${ \textbf { z } } ( 8 \times 1 6 ^ { 3 } )$ and decodes it to $\hat { s } ~ ( 1 \times 1 2 8 ^ { 3 } )$ . The encoder is fully convolutional, with three stride-2 convolutions (total stride 8), so the latent is a spatial grid that keeps the layout of the lung, not a single global vector. The loss is

$$
\mathcal { L } _ { \mathrm { V A E } } = \Vert \hat { s } - s \Vert _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D i c e } } ( \hat { s } , s ) + \beta \mathrm { K L } .\tag{1}
$$

The first term preserves the global distance field. $\mathcal { L } _ { \mathrm { D i c e } }$ is a soft Dice on sigmoid-relaxed SDTs [22] that sharpens the surface. KL is the Kullback–Leibler divergence between the encoder distribution $q ( \mathbf { z } | s )$ and a standard Gaussian $\mathcal { N } ( 0 , I )$ which regularizes the latent space. We use λ=1 and $\beta { = } 1 0 ^ { - 6 }$ The VAE is then frozen $( E ^ { * } , D ^ { * } ) ;$ ; having seen only healthy lungs, its latent space describes healthy shapes only.

## 2.2. Stage 2: anatomy-conditioned latent diffusion

Pathology-mimicking corruption. Without pathological ground truth, we create (faulty, true) mask pairs from healthy lungs. For each training lung m, a corrupted mask $m ^ { \prime }$ is generated on the fly with the failure modes of texture-based segmenters: region drop (a contiguous region removed from the apex, base or lateral wall), medial drop, subpleural erosion, cavities (internal holes) and truncation (a lung cut by an oblique plane). Occasionally a leak dilates the mask, so the model also learns to remove voxels. Most corruptions are mild, so the model learns to keep correct regions unchanged, and a new corruption is drawn at each epoch.

Bone conditioning. When a large region is missing, the mask alone admits many completions. We add information that does not depend on the disease: ribs and chest wall are unaffected by parenchymal lesions and tightly bound the lung. We provide the bone occupancy o and a distance-to-bone field $\delta ,$ both pooled to $1 6 ^ { 3 }$ , as $\mathbf { c } _ { \mathrm { b o n e } } = [ o , \delta ]$ (2 channels). Thin ribs largely vanish after pooling, so the smooth $\delta$ carries the distance to the chest wall.

Diffusion model. The frozen encoder gives ${ \bf z } _ { 0 } = E ^ { * } ( s )$ and $\mathbf { z } _ { \mathrm { c o r r } } = E ^ { * } ( s ^ { \prime } )$ . We add noise to the clean latent, $\mathbf { z } _ { t } ~ =$ $\alpha _ { t } \mathbf { z } _ { 0 } + \sigma _ { t } \epsilon$ , where t is the diffusion time, $\epsilon \sim \mathcal { N } ( 0 , I )$ , and $\alpha _ { t } ,$ $\sigma _ { t }$ are the signal and noise scales of a cosine schedule [23]. A 3D U-Net [24] $\mathrm { V } _ { \theta }$ receives $\mathbf { z } _ { t }$ (8 channels) and the condition $\mathbf { c } = [ \mathbf { z } _ { \mathrm { c o r r } } , \mathbf { c } _ { \mathrm { b o n e } } ] \left( 8 + 2 \right.$ channels), i.e. 18 input channels at $1 6 ^ { 3 }$ , and predicts the velocity [25] $v _ { \theta }$ (8 channels). It is trained to match the target velocity $v = \alpha _ { t } \epsilon - \sigma _ { t } \mathbf { z } _ { 0 } ;$

$$
\mathcal { L } _ { \mathrm { L D M } } = \mathbb { E } _ { t , \epsilon } \big \| v _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } ) - v \big \| ^ { 2 } .\tag{2}
$$

During training, $\mathbf { z } _ { \mathrm { c o r r } }$ is zeroed with probability 0.2 (bone channels always kept), so the same network also learns to predict from the bone alone. This prevents the model from copying its input and enables classifier-free guidance [26]: at inference we combine the two predictions as $v _ { w } = v _ { \mathrm { b o n e } } +$ w $( v _ { \mathrm { f u l l } } - v _ { \mathrm { b o n e } } )$ , where $v _ { \mathrm { f u l l } } = v _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } )$ uses the input mask and $v _ { \mathrm { b o n e } } ~ = ~ v _ { \theta } ( \mathbf { z } _ { t } , t , [ \mathbf { 0 } , \mathbf { c } _ { \mathrm { b o n e } } ] )$ uses only the bone prior. The guidance weight w sets the balance: w=1 gives the full prediction, and larger w keeps the output closer to the input mask.

Inference and uncertainty. The input mask $m ^ { \prime }$ obtained with R231 and the bone mask are cropped, and $m ^ { \prime }$ is converted to $s ^ { \prime }$ and encoded into $\mathbf { z } _ { \mathrm { c o r r } }$ . We draw N latents with DDIM [10] (50 steps, guided with w) and decode each with $D ^ { * }$ into an SDT $\hat { s } ^ { ( n ) }$ . The mean of the N decoded SDTs, $\begin{array} { r } { \bar { s } = \frac { 1 } { N } \sum _ { n } \hat { s } ^ { ( n ) } } \end{array}$ , is upsampled trilinearly to the native grid and thresholded at zero, and the two largest components form the corrected mask mˆ . The standard deviation of the same N SDTs, $u ( x ) = \operatorname { s t d } _ { n } { \hat { s } } ^ { ( n ) } ( x )$ , gives a voxel-wise uncertainty map: it is close to zero where the samples agree, for example where the input mask is kept, and high where they differ, i.e. where several completions are plausible.

## 3. EXPERIMENTS

## 3.1. Data

Training data. The VAE (Stage 1) and the diffusion model (Stage 2) are trained on the same private COVALUNG cohort: 57 healthy post-COVID patients with no consolidation or other parenchymal lesion (90/10% patient-level split). No pathological scan or mask is used at any point.

Test data. We use 20 TB cases from ImageCLEF TB [27], selected with the DeepPulmo annotations [28] as the most severe in cavity burden, since large cavities and consolidations are where R231 fails most. Cavities cover 3.1–16.1% of the lung volume (largest cavity 57–962 mL, 4–36 cavities per case) and consolidation 2.0–21.9%. Patient lists and details are provided in the code repository. The DeepPulmo lung masks are not accurate enough to serve as a reference (examples in the repository), so we use only its lesion labels. To obtain a reliable reference, a thoracic radiologist corrected the R231 mask on 200 axial slices, 10 per case. Finally, we assess LungMend qualitatively on TB CT scans of 18 patients from Avicenne Hospital, Bobigny, France.

## 3.2. Evaluation

We evaluate in two ways:

• Expert slices: Dice and average surface distance (ASD) on the 200 radiologist-corrected slices.

• Lesion recovery: using the DeepPulmo cavity and consolidation labels, the fraction of each lesion included in the predicted lung mask (recall), over the 200 slices.

In addition, we assess LungMend qualitatively on the Avicenne cohort.

Table 1. Results on the 200 expert slices of severe TB cases (mean; ASD and HD95 in mm). Best in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">Expert slices</td><td colspan="2">Lesion recall</td></tr><tr><td>Dice↑</td><td>ASD↓</td><td>HD95↓</td><td>Cav.↑</td><td>Cons.↑</td></tr><tr><td>R231 [1]</td><td>0.909</td><td>4.38</td><td>27.4</td><td>0.869</td><td>0.732</td></tr><tr><td>R231 + closing</td><td>0.929</td><td>3.79</td><td>22.9</td><td>0.934</td><td>0.868</td></tr><tr><td>Post-DAE [7]</td><td>0.768</td><td>11.87</td><td>36.5</td><td>0.968</td><td>0.942</td></tr><tr><td>Det. 3D U-Net</td><td>0.896</td><td>5.04</td><td>26.1</td><td>0.557</td><td>0.731</td></tr><tr><td>TotalSegmentator</td><td>0.919</td><td>4.29</td><td>24.7</td><td>0.951</td><td>0.723</td></tr><tr><td>VISTA3D</td><td>0.847</td><td>7.61</td><td>38.8</td><td>0.726</td><td>0.364</td></tr><tr><td>LungMend (ours)</td><td>0.986</td><td>0.57</td><td>5.5</td><td>0.993</td><td>0.981</td></tr></table>

Table 2. Ablations on the 200 expert slices (ASD in mm). A: one training component removed at a time, with the model retrained using the same protocol as the full recipe; all rows use plain sampling (1 sample, w=1). B: inference steps added one at a time to the final model. R231 input: Dice 0.909.
<table><tr><td></td><td>Dice↑</td><td>ASD↓</td></tr><tr><td>A. Training components</td><td></td><td></td></tr><tr><td>Full recipe (reference)</td><td>0.964</td><td>2.34</td></tr><tr><td>Latent → voxel-space diffusion</td><td>0.917</td><td>4.54</td></tr><tr><td>SDT → binary mask</td><td>0.795</td><td>15.02</td></tr><tr><td>No Dice term in VAE</td><td>0.935</td><td>3.68</td></tr><tr><td>No bone mask</td><td>0.929</td><td>3.47</td></tr><tr><td>No distance-to-bone</td><td>0.926</td><td>3.75</td></tr><tr><td>No lung-channel dropout</td><td>0.939</td><td>3.32</td></tr><tr><td>No random-blob corruptions</td><td>0.923</td><td>4.02</td></tr><tr><td>No medial corruption</td><td>0.938</td><td>3.16</td></tr><tr><td>No leak corruption</td><td>0.937</td><td>3.49</td></tr><tr><td>B. Inference (final model, cumulative)</td><td></td><td></td></tr><tr><td>Plain sampling (1 sample, w=1)</td><td>0.964</td><td>2.34</td></tr><tr><td>+ guidance w=1.5</td><td>0.967</td><td>2.15</td></tr><tr><td>+ average of 5 samples</td><td>0.976</td><td>0.93</td></tr><tr><td>+ smooth SDT upsampling (= LungMend)</td><td>0.986</td><td>0.57</td></tr></table>

## 3.3. Baselines and implementation

Baselines. Segmentation models applied to the CT: R231, a U-Net trained on diverse routine CT including pathological cases; TotalSegmentator [29], an nnU-Net-based model trained on 1204 CT examinations of clinical routine, with varied pathologies, scanners and sites (lung output: union of the lung lobes); and VISTA3D [30], a MONAI segmentation foundation model with a SegResNet backbone, trained on 11,454 CT volumes with anatomical and lesion annotations. Mask correction: R231 with morphological closing (10 mm ball) and hole filling; Post-DAE [7] trained on the same healthy masks and corruptions; and a deterministic 3D U-Net that takes the CT, the R231 mask and the bone mask and predicts the corrected mask, trained on the same healthy scans and corruptions. It has the same conditioning as Lung-Mend plus the CT, but no diffusion step, which isolates the effect of the generative formulation.

![](images/75ea527e8539d44f0b168c4ac10eff68e20754e222a0f9f7f21c9d31337ba8b7.jpg)  
Fig. 2. Qualitative results on severe TB. (a) ImageCLEF TB axial slices: R231+closing (red, top) and LungMend (blue, bottom) masks overlaid on the CT, with the annotation in yellow. The region added by LungMend is green; the volume added over R231 is given under each case. (b) Avicenne TB: coronal slices and 3D renderings for R231 and LungMend. LungMend recovers cavities and consolidated lung that R231 excludes, and removes the stair-step fragmentation of the R231 3D surface.

## 4. RESULTS AND DISCUSSION

Comparison with baselines. LungMend is the best method on every metric of Table 1, with the largest margin on HD95 (5.5 mm vs. 22.9 mm for the strongest baseline, R231 + closing): the baselines are accurate on most of the lung but leave large regions out, which dominates the worst-case distance. General segmentation models, even when trained on pathological CT, still decide from appearance, and more training data does not fix this (VISTA3D does worse than TotalSegmentator). Closing improves the surface but cannot recreate missing regions, and Post-DAE recovers lesions but oversegments, since it has no anatomical anchor. The deterministic U-Net has all the inputs of LungMend plus the CT, so it isolates the diffusion step: when several completions are plausible, a deterministic model returns a single estimate and loses the cavities, whereas diffusion samples plausible completions and averages them.

Number of samples and inference time. We tested several values of N and chose N=5: with fewer samples the averaged map is sparse and unstable across runs, while beyond 5 we observed no noticeable change. The cost grows linearly with N, but the samples are independent and can run in parallel on the GPU, so N=5 remains practical as a one-off post-processing step per scan.

Ablation. The ablation (Table 2, single runs) shows that the information given to the model matters more than architectural choices. Bone conditioning helps where the mask alone leaves a missing region ambiguous, and pathologymimicking corruptions are the condition for transfer: a model trained on random blobs loses the most lesion recall, because a healthy-only prior is useful only if the simulated failures resemble the real ones. Latent diffusion on the SDT beats voxel-space diffusion and binary masks. The medial and leak corruptions and the conditioning dropout have a clear effect here. At inference, guidance and averaging reduce the variability between samples, and a single sample is already competitive.

Qualitative results. Fig. 2 compares LungMend with R231 + closing on cases selected by the volume LungMend adds. The added regions are the large cavities and consolidations on the diseased side; further examples on Avicenne scans are in Fig. 2b and in the repository.

Limitations. A healthy prior may over-complete lungs with genuine volume loss (fibrotic retraction, destroyed lung), so the uncertainty map should be read together with the input mask. The gain depends on corruptions that resemble real failures, and the 128<sup>3</sup> resolution limits precision near the hilum. Lesion recall depends on the DeepPulmo labels, and the expert slices cover only part of the lung.

## 5. CONCLUSION

We presented LungMend, a post-processing model that corrects lung masks in severe TB CT using only healthy anatomy. A latent diffusion prior trained on healthy lungs, together with a corruption model that reproduces the failure modes of texture-based segmenters and a bone-anchored condition, recovers cavities and consolidations that R231 excludes, without any pathological training data or CT intensities. Repeated sampling also provides an uncertainty map that shows which regions of the corrected mask were reconstructed rather than kept from the input.

## 6. ACKNOWLEDGMENTS

This work was carried out within a ANRS-MIE project No. ANRS00885-R-PR and ANR MLQ-CT project No. ANR-23-CE17-0046

## 7. REFERENCES

[1] J. Hofmanninger et al., “Automatic lung segmentation in routine imaging is primarily a data diversity problem,” Eur. Radiol. Exp., 2020.

[2] I. Sluimer et al., “Toward automated segmentation of the pathological lung in CT,” IEEE TMI, 2005.

[3] A. Mansoor et al., “A generic approach to pathological lung segmentation,” IEEE TMI, 2014.

[4] A. P. Harrison et al., “Progressive and multi-path holistically nested networks for pathological lung segmentation,” in MICCAI, 2017.

[5] T. Heimann and H.-P. Meinzer, “Statistical shape models for 3D medical image segmentation,” Med. Image Anal., 2009.

[6] O. Oktay et al., “Anatomically constrained neural networks (ACNNs),” IEEE TMI, 2018.

[7] A. J. Larrazabal et al., “Post-DAE: anatomically plausible segmentation via denoising autoencoders,” IEEE TMI, 2020.

[8] N. Painchaud et al., “Cardiac segmentation with strong anatomical guarantees,” IEEE TMI, 2020.

[9] J. Ho et al., “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[10] J. Song et al., “Denoising diffusion implicit models,” in ICLR, 2021.

[11] J. Wolleb et al., “Diffusion models for implicit image segmentation ensembles,” in MIDL, 2022.

[12] J. Wu et al., “MedSegDiff: medical image segmentation with diffusion models,” in MIDL, 2023.

[13] M. Wang et al., “SegRefiner: model-agnostic segmentation refinement with discrete diffusion,” in NeurIPS, 2023.

[14] R. Rombach et al., “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.

[15] W. H. L. Pinaya et al., “Brain imaging generation with latent diffusion models,” in DGM4MICCAI, 2022.

[16] F. Khader et al., “Denoising diffusion probabilistic models for 3D medical image generation,” Sci. Rep., 2023.

[17] J. J. Park et al., “DeepSDF: learning continuous signed distance functions,” in CVPR, 2019.

[18] L. Zhou et al., “3D shape generation and completion through point-voxel diffusion,” in ICCV, 2021.

[19] Y.-C. Cheng et al., “SDFusion: multimodal 3D shape completion, reconstruction, and generation,” in CVPR, 2023.

[20] B. Zhang et al., “3DShape2VecSet: a 3D shape representation for neural fields,” ACM Trans. Graph., 2023.

[21] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” in ICLR, 2014.

[22] Y. Xue et al., “Shape-aware organ segmentation by predicting signed distance maps,” in AAAI, 2020.

[23] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in ICML, 2021.

[24] Ozgun Cicek, “3D U-Net: learning dense volumetric segmentation from sparse annotation,” in MICCAI, 2016.

[25] T. Salimans and J. Ho, “Progressive distillation for fast sampling of diffusion models,” in ICLR, 2022.

[26] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv:2207.12598, 2022.

[27] S. Kozlovski et al., “Overview of ImageCLEFtuberculosis 2021,” in CLEF, 2021.

[28] Z. Tan et al., “DeepPulmoTB: benchmark dataset for tuberculosis lesions in lung CT,” Heliyon, 2024.

[29] J. Wasserthal et al., “TotalSegmentator: robust segmentation of 104 anatomic structures in CT,” Radiol. Artif. Intell., 2023.

[30] Y. He et al., “VISTA3D: a unified segmentation foundation model for 3D imaging,” in CVPR, 2025.