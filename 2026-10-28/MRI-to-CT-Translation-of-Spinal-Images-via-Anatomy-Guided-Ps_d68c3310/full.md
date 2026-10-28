# MRI-to-CT Translation of Spinal Images via Anatomy-Guided Pseudo-Pairing and Adversarial Difusion

Eduardo Nieto<sup>a</sup>, Ibai Genua<sup>b</sup>, Davide Scorza<sup>b</sup> and Karen López-Linares<sup>a,c,∗</sup>

<sup>a</sup>Fundación Vicomtech, Basque Research and Technology Alliance (BRTA), Donostia - San Sebastián, Spain

<sup>b</sup>R&D Department, Cyber Surgery, Donostia - San Sebastián, Spain

<sup>c</sup>eHealth Group, Bioengineering Area, Biogipuzkoa Health Research Institute, Donostia - San Sebastián, Spain

## A R T I C L E I N F O

Keywords:   
Spine   
Translation   
MR   
CT   
Difusion   
Synthesis   
Vertebrae   
Preoperative planning

## A BS T R AC T

Background: Magnetic Resonance Imaging (MRI) is routinely used for diagnosing spinal disorders, but surgical planning often still requires Computed Tomography (CT) because of its superior visualization of vertebral bone anatomy. Avoiding an additional CT acquisition would reduce ionizing radiation exposure and simplify the preoperative workflow. MRI-to-CT synthesis therefore ofers a promising route to provide CT-like anatomical information directly from MRI.

Methods: We propose a three-stage framework for MRI-to-CT translation in spine imaging. First, a vertebral matching system uses spine segmentations to identify anatomically similar MRI and CT sagittal slices and align them by rigid registration, generating pseudo-paired training data from unpaired datasets. Second, the pseudo-paired slices are used to train an adversarial difusion model for MR-to-CT synthesis. Third, the translated sagittal slices are reassembled into 3D volumes and refined with a 3D post-processing network designed to improve inter-slice consistency and volumetric structural coherence.

Results: The approach was trained on more than 10,000 unpaired sagittal MR and CT slices and evaluated on 1,911 sagittal slices from 83 MR volumes covering all spinal regions. Evaluation included distribution-based metrics and task-specific vertebral segmentation metrics. Compared with the difusion-only baseline, the proposed strategy improved vertebral reconstruction, increasing the binary 3D Dice score from 0.666 to 0.739. The 3D post-processing network reduced visible inter-slice discontinuities after sagittal slice stacking and produced more coherent vertebral structures across axial and coronal views, supporting improved volumetric consistency of the synthetic CTs.

Conclusions: The proposed framework improves spine MRI-to-CT synthesis by combining anatomyguided pseudo-pairing with adversarial difusion-based translation and volumetric refinement. Vertebral matching was the main driver of improved anatomical alignment and vertebral fidelity, while postprocessing enhanced 3D coherence. These results support the potential of the method as a step toward CT-like spine visualization from MRI for preoperative planning, with future work focused on preserving fine intensity detail while further strengthening volumetric consistency.

## 1. Introduction

Medical imaging modalities such as Magnetic Resonance (MR) and Computed Tomography (CT) provide complementary anatomical information of the spine region. MR ofers superior soft-tissue contrast, enabling detailed visualization of intervertebral discs, the spinal cord, nerves, and surrounding soft tissues, whereas CT is particularly efective for depicting osseous structures such as vertebrae [1]. In spine surgery and preoperative planning, the combined use of both modalities enables accurate vertebral delineation, precise localization of critical neural structures, and detailed characterization of vertebral morphology, all of which are essential for decisionmaking [2, 3].

In clinical practice, MR is generally the preferred modality for assessing most spinal pathologies, including disc herniation, spinal cord lesions, and tumors, because it provides rich soft-tissue information without ionizing radiation. CT is typically reserved for applications requiring detailed evaluation of bone anatomy, such as fracture assessment, implant planning, and surgical navigation, despite the associated radiation exposure. This complementary but selective use of MR and CT motivates the development of MR-to-CT translation methods, which aim to provide CT-like bone information directly from MR acquisitions.

Recent advances in deep learning have made this objective increasingly feasible by learning cross-modal mappings between MR and CT images [4]. However, most existing MRto-CT translation methods operate exclusively on 2D slices due to computational constraints, which limits their ability to enforce inter-slice consistency and full 3D anatomical coherence. Furthermore, a significant challenge in training these models is the scarcity of large, representative paired datasets [5]. Even when paired data are available, alignment between both imaging studies can be compromised by respiratory motion, tissue deformation or diferences in patient positioning during acquisition [6, 7]. Given the higher availability of unpaired datasets, unsupervised methods have emerged as a viable alternative, showing promising results in medical image translation tasks [8, 9].

Thus, in this work, we propose a three-stage framework for MR-to-CT translation of spine images. First, a 2D vertebral matching system is implemented to generate pseudo-paired MR-CT data from diferent patients. These pseudo-pairs are then used to train a 2D SynDif architecture [10] across all spinal regions, enabling robust cross-modal synthesis despite the lack of true paired acquisitions. Finally, a 3D Generative Adversarial Network (GAN)-based postprocessing network is applied to enhance contextual consistency across adjacent slices, improving anatomical coherence in the reconstructed synthetic CT volumes.

The remainder of this paper is organized as follows. Section 2 reviews the state of the art in MR-to-CT translation and spine image synthesis. Section 3 describes the materials and methods, including the proposed three-stage framework and advanced validation strategies. Section 4 presents the experimental results and discusses the findings and their clinical implications. Finally, Section 5 summarizes the main conclusions and outlines future research directions.

## 2. State of the art

Medical image translation has attracted increasing research attention in recent years, with a rapidly expanding body of work focused on cross-modal synthesis [5]. Among the proposed approaches, deep learning methods have consistently outperformed conventional atlas, registration, and voxelbased techniques, establishing themselves as the dominant paradigm for the task [4].

Early deep learning methods relied on Variational Autoencoders (VAEs) and fully convolutional neural networks (CNNs) for image synthesis [11, 12]. However, limitations in image realism and structural fidelity motivated the widespread adoption of Generative Adversarial Networks (GAN), including paired and unpaired formulations such as Cycle-Consistent Generative Adversarial Networks (CycleGANs) [13, 14, 15]. More recently, Transformer-based architectures and Denoising Difusion Models have achieved state-of-the-art performance, surpassing earlier methods in image quality, anatomical consistency, and synthesis diversity [16, 17].

The following subsections review these methodological families, beginning with GAN-based approaches, followed by Transformer- and difusion-based methods, and concluding with a focused review on MR-to-CT translation of spine images.

## 2.1. GAN-based approaches

Generative Adversarial Networks (GANs) have become a central paradigm for medical image synthesis, relying on an adversarial objective between a generator and a discriminator to produce realistic outputs [13].

These methods rapidly became a dominant approach for MR-to-CT synthesis, with several adaptations proposed to address challenges specific to medical imaging. For instance, [18] introduced a context-aware GAN, while [19] improved image quality and reduced blurring through the incorporation of additional non-adversarial losses. Other studies focused on improving anatomical fidelity during synthesis; for instance, [20] incorporated organ segmentation constraints to better preserve shape and spatial localization.

A major limitation of early GAN-based translation methods was their reliance on paired datasets, which are dificult to obtain. To overcome this issue, unpaired GAN variants were introduced, most notably CycleGANs [21, 14, 15]. By incorporating a cycle-consistency loss, these models enforce reconstruction of the original image after forward and inverse domain translations, enabling training on unpaired MR and CT data while constraining unrealistic cross-modal mappings. This represented a major advance for MR-to-CT synthesis and led to successful applications across multiple anatomical regions, including the brain [22, 23] and pelvis [24]. By combining adversarial and cycle-consistency objectives, these architectures improve cross-modal synthesis while maintaining anatomical plausibility. Nevertheless, GANbased approaches remain limited by training instability and residual inconsistencies, which can compromise synthesis robustness and structural precision.

## 2.2. Transformers and difusion models

Recent advances in medical image translation have increasingly leveraged transformer architectures and difusion models, which have emerged as state-of-the-art alternatives to GAN-based approaches [16, 17]. These methods are particularly efective at capturing both long-range dependencies and fine local details, enabling more coherent and anatomically consistent cross-modal synthesis.

Transformer-based methods introduce self-attention mechanisms that model relationships across the entire image, allowing the network to capture global contextual information. This characteristic is especially advantageous in MR-to-CT translation, where preserving large-scale anatomical relationships is critical. Several promising approaches have been proposed in this direction, including pure Transformerbased frameworks [25] as well as hybrid architectures that combine attention mechanisms with adversarial learning [26].

Difusion models constitute another major methodological advance in medical image synthesis. These models learn the reverse of a progressive noising process, enabling highfidelity image generation through iterative denoising steps. In MR-to-CT translation, several unsupervised difusion-based approaches have shown promising performance [27, 15, 28]. Other studies have explored hybrid formulations combining difusion and adversarial objectives [29].

In general, Transformer- and difusion-based models have demonstrated superior synthesis quality compared with earlier GAN-based approaches, particularly in terms of anatomical coherence and image fidelity [30]. However, these gains are often achieved at the cost of substantially higher computational complexity and memory requirements, which remain important practical limitations for high-resolution medical imaging tasks.

## 2.3. MR-to-CT translation of spine images

Several GAN- and CycleGAN-based approaches have been proposed for MR-to-CT translation of spine images, including the works of [31, 32, 33]. While these studies demonstrated the feasibility of spine image synthesis, important challenges remain, including limited dataset size, high variability in synthesized volumes, and restriction to specific spinal regions.

To address some of these limitations, more recent studies have explored difusion-based methods. For example, [34] proposed a difusion model for sagittal lumbar spine MRto-CT translation, highlighting the potential of iterative generative frameworks for improved structural fidelity.

Comprehensive evaluations of this task have also been reported, such as the review by [35], which compared several supervised and unsupervised architectures on spine datasets. Among the evaluated methods, SynDif—an adversarial difusion framework originally introduced by [10]—stands out for its ability to leverage unpaired data while maintaining strong anatomical consistency. Notably, its translation performance was also evaluated on pelvic MR images containing the lumbar vertebrae and sacrum, further supporting its suitability for spine-related synthesis tasks.

However, challenges specific to full-spine imaging remain insuficiently addressed, particularly vertebral correspondence across unpaired datasets and slice-to-slice consistency in 3D volumes. These limitations motivated the use of SynDif as the baseline architecture and directly informed the design of the proposed framework, which incorporates a vertebral matching system and a 3D consistency enhancement step to overcome the challenges of unpaired, slice-based 2D translation.

## 3. Materials and methods

The following subsections describe the employed data and our proposed pipeline for unpaired MR to CT spine image translation.

## 3.1. Dataset collection and preparation

This study was conducted using publicly available datasets that collectively cover all spinal regions, including the cervical, thoracic, and lumbar spine.

Regarding MR data, datasets were preferentially selected to include sagittal T1-weighted acquisitions, as this plane is the clinical standard for global spine assessment and provides optimal visualization of vertebral structures.

For the cervical region, the Spine Generic Project and the Instrumented Spinal Cord T1/T2-weighted MRI Scans datasets were included [36, 37]. For the lumbar region, the SPIDER dataset was selected [38]. Publicly available MR scans covering the thoracic region or the full spine were considerably less common, as sagittal acquisitions spanning this area are less frequently available. Nevertheless, suitable cases were identified in the TotalSegmentator MRI Dataset [39]. This process yielded a total of 560 MR scans.

For CT data, datasets were selected to provide analogous regional coverage across the same spinal sections. The cervical region was represented by the PDDCA, HaN-Seg, and HNSCC-3DCT-RT datasets [40, 41, 42]. For the thoracic, lumbar, and multi-region scans, the Spine-Mets-CT-Seg,

<table><tr><td>Split</td><td>Region</td><td>MR</td><td>CT</td><td>MR Slices</td><td>CT Slices</td></tr><tr><td rowspan="6">Train</td><td>Cervical</td><td>198</td><td>340</td><td>4532</td><td>10303</td></tr><tr><td>Cervical &amp; Thoracic</td><td>13</td><td>334</td><td>297</td><td>9984</td></tr><tr><td>Thoracic</td><td>12</td><td>3028</td><td>270</td><td>82984</td></tr><tr><td>Thoracic &amp; Lumbar</td><td>1</td><td>293</td><td>24</td><td>8280</td></tr><tr><td>Lumbar</td><td>261</td><td>2412</td><td>6196</td><td>85416</td></tr><tr><td>Cervical</td><td>60</td><td>107</td><td>1371</td><td>3228</td></tr><tr><td rowspan="5">Validation</td><td>Cervical &amp; Thoracic</td><td>8</td><td>98</td><td>185</td><td>2964</td></tr><tr><td>Thoracic</td><td>7</td><td>887</td><td>139</td><td>24298</td></tr><tr><td>Thoracic &amp; Lumbar</td><td>0</td><td>81</td><td>0</td><td>2273</td></tr><tr><td>Lumbar</td><td>78</td><td>713</td><td>1860</td><td>25198</td></tr><tr><td>Cervical</td><td>33</td><td>51</td><td>739</td><td>1560</td></tr><tr><td rowspan="4">Test</td><td>Cervical &amp; Thoracic</td><td>6</td><td>55</td><td>137</td><td>1670</td></tr><tr><td>Thoracic</td><td>3</td><td>470</td><td>65</td><td>12752</td></tr><tr><td>Thoracic &amp; Lumbar</td><td>1</td><td>38</td><td>21</td><td>1078</td></tr><tr><td>Lumbar</td><td>40</td><td>354</td><td>949</td><td>12482</td></tr></table>

## Table 1

Dataset split summary after sagittal slice extraction. MR and CT columns indicate the number of subvolumes per modality, while MR Slices and CT Slices reflect the total number of 2D sagittal slices extracted from the subvolumes. The statistics are further stratified by spinal region. A vertebral region is considered present if it contains at least 50% of the possible vertebrae from that region (e.g 4 out of 7 cervical vertebrae).

TotalSegmentator CT Dataset, and CTSpine1K datasets were incorporated [43, 44, 45], producing a multisource CT dataset of 2884 scans.

After initial dataset identification, only scans containing at least five vertebrae, according to their vertebral segmentation, were retained. Some of these segmentations were already available in the original datasets, whereas the missing ones were generated using the TotalSegmentator tool [44, 39].

In addition, cases presenting severe acquisition artifacts, such as low resolution, extreme intensity areas due to the presence of metal implants or excessive image noise (like quantum noise), were also excluded after visual inspection. Hence, the final imaging cohort comprised 558 MR scans and 1136 CT scans coming from 10 distinct open-access datasets. Figure 1 summarizes the dataset generation process.

## 3.1.1. Preprocessing

The first step of the preprocessing pipeline consisted of standardizing the imaging studies to reduce the intramodality variability across MR and CT datasets. All scans were first reoriented to a LIP coordinate system and resampled to a voxel spacing of 1 mm x 1 mm x 2.5 mm. Intensity normalization was performed using min-max scaling. In addition, modality-specific preprocessing was applied: CT intensities were clipped to the range [-1000, 1000], while MR scans underwent bias field correction [46].

After this initial standardization, each scan was cropped in two stages using their spine segmentations. First, a general crop was applied to each scan to remove regions without spinal structures. After this first cropping, multiple 3D subvolumes were extracted using a sliding-window strategy, as shown in Figure 2. Specifically, the superior endplate of each vertebra was used as a reference, and 288x288xD subvolumes, where D corresponds to the original extent along the craneo-caudal axis, were extracted along the caudal direction. This patch size was selected empirically, since it provided enough anatomical coverage to include up to five vertebrae in any spinal region while avoiding excessively large empty areas.

![](images/07ca211fbe7853bd44bb83cb70ceaad8db62f367921753014eee98f1d9b620e3.jpg)  
Figure 1: Data flow diagram illustrating the full pipeline from raw data to the final dataset construction. Starting from 2844 CT scans and 560 MR scans, vertebral segmentations are obtained via TotalSegmentator, followed by manual filtering and region-based subvolume extraction. Subvolumes are then automatically filtered based on the quality of their segmentation masks, discarding those with anomalous or incomplete vertebral segmentations. Since the downstream translation model operates on 2D, sagitta slices are extracted from the subvolumes. Finally, the resulting slices are split into training, validation and test sets.

Since this strategy caused the inferior part of the scan to be excluded in most cases, a final 288x288xD subvolume was included in the extraction process, starting from the inferior end of the scan and extending in the superior direction. In this way we ensured complete coverage of the spinal column.

In order to avoid data redundancy, the sliding window was constrained so that extracted subvolumes from the same scan shared at most two complete vertebrae. This constraint was checked through the segmentation masks.

A second redundancy control criterion was also applied to intra-patient subvolumes by limiting the similarity between their spine segmentations to 90%. This similarity was computed through the number of voxels per label in each subvolume, and was defined as:

$$
S ( P _ { 1 } , P _ { 2 } ) = 1 - \frac { \sum _ { v \neq 0 } \left| n _ { v } ^ { P _ { 1 } } - n _ { v } ^ { P _ { 2 } } \right| } { \sum _ { v \neq 0 } \operatorname* { m a x } \left( n _ { v } ^ { P _ { 1 } } , n _ { v } ^ { P _ { 2 } } \right) }\tag{1}
$$

Where $S ( P _ { 1 } , P _ { 2 } )$ is the similarity score between subvolumes $P _ { 1 }$ and $P _ { 2 } , v$ denotes each vertebral label (excluding the background, $v \neq 0 )$ , and $n _ { v } ^ { P _ { i } }$ is the number of voxels belonging to vertebra � in patch $P _ { i } .$ A score close to 1 indicates that both subvolumes share a nearly identical vertebral volume distribution, and therefore cover the same spinal region, while a score close to zero indicates they cover diferent spinal regions.

Both redundancy filterings were only applied to MR subvolumes. No equivalent restriction was imposed on CT subvolumes, since redundancy in the CT pool was not detrimental for the subsequent pseudo-pair generation stage and retaining a larger number of CT candidates was beneficial for later vertebral matching.

Since the downstream translation model operates on 2D sagittal images, the final inputs to the network were the sagittal slices extracted from the resulting 3D subvolumes. Slices containing less than four vertebrae were discarded in order to exclude data with insuficient vertebral context before training.

After this process, the dataset was divided into training, validation, and test sets with target proportions of 70%, 20%, and 10%, respectively. To prevent data leakage, all subvolumes and corresponding slices from the same patient were assigned to the same dataset. In addition, the split was designed to preserve a representative distribution of vertebral region combinations across subsets. The preprocessing approach and a summary of the final dataset characteristics are shown in Figure 1 and Table 1, respectively.

## 3.2. Proposed approach

In this work, we propose a three-stage framework for MR-to-CT spine image translation. The first stage involves implementing a vertebral matching system, which processes unpaired MR and CT sagittal slices to generate pseudo-paired datasets by aligning anatomical structures. This step is crucial for addressing anatomical discrepancies and ensuring suitable inputs for the next stage. The second stage focuses on the core 2D image translation task, where we leverage the SynDif neural network architecture (described in Section 2.3). Finally, the third stage of the framework applies a post-processing network to enhance 3D contextual consistency across adjacent slices, ensuring improved anatomical coherence in the reconstructed synthetic CT volumes obtained by stacking the translated sagittal slices. This post-processing step is essential for refining the quality and structural accuracy of the synthetic CTs, and is implemented as a 3D GAN with a ResUNet generator and a PatchGAN discriminator.

In the following subsections, we provide detailed information on each stage of the framework, including the training process, model evaluation, and validation strategies.

## 3.2.1. Vertebral matching system

To enhance the performance of the MR-to-CT translation task, we introduce a vertebral matching system as the first stage of our framework. While SynDif is designed to handle unpaired data, training with randomly selected MR and CT slices can lead to mismatches, especially in terms of spine region or vertebral alignment, which can negatively impact translation accuracy. The vertebral matching system mitigates this issue by ensuring that each MR slice is paired with a CT slice that has a similar spinal structure.

![](images/e10cfd659a3b937c6c56f7b65bcd427c52c5b0f5aaace4231e9ff7ee0832c2d0.jpg)  
Figure 2: Example of MR (left) and CT (right) patches extraction after preprocessing. From each scan, multiple overlapping patches covering diferent spinal regions are extracted, as shown by the individual patch examples on the right of each scan. Unlike CT, where redundant patches from the same scan are allowed, MR patches are filtered to avoid patches with similar spine information, resulting in a significantly smaller number of extracted MR patches overall.

As both MR and CT scans contain thousands of slices, making pixel-wise comparison across all possible pairs is computationally prohibitive. To address this, we first compute a simple shape descriptor for each MR and CT sagittal slice segmentation, derived from the relative proportion of pixels occupied by each vertebrae (C1, C2... sacrum) in the complete segmentation mask. This descriptor is represented as a 1x25 array, and the cosine distance between the descriptors extracted from MR and CT slices are used to identify the most relevant MR-CT slice match candidates (two pair of images with the same vertebrae and with a similar size have closer descriptors).

While this approach helps identify slices with similar vertebral proportions, it does not guarantee identical spine shapes or vertebral tissue overlap, as the pixel distribution may still difer and the scans may be misaligned. Therefore, the descriptor is used only for an initial filtering, followed by a fixed registration, inspired by the work of Graf et al. [35], and a second comparison using a weighted Dice Similarity Score (DSC), which accounts for vertebral size and penalizes the presence of additional vertebrae. This weighted DSC is defined as:

$$
\mathrm { D S C } _ { w } = \frac { \sum _ { c \neq 0 } \mathrm { D S C } _ { c } \cdot w _ { c } } { \sum _ { c \neq 0 } w _ { c } } , \quad \mathrm { D S C } _ { c } = \frac { 2 | A _ { c } \cap B _ { c } | } { | A _ { c } | + | B _ { c } | }\tag{2}
$$

Where $A _ { c }$ and $B _ { c }$ are the binary masks of class � in each segmentation and and $w _ { c } = \lvert A _ { c } \rvert + \lvert B _ { c } \rvert$ is defined as the total number of voxels of class � across both masks. The background class (� = 0) is excluded from the metric computation.

This two comparisons ensure that not only a shape correspondence between images is established, but also tissue localization and overlap.

Regarding the number of comparisons executed for each slice, the fifty most similar candidates according to the descriptor are selected, restricting the search to those difering by at most one vertebra. For each candidate, the CT spine segmentation is aligned with the MR spine through a rigid registration based on vertebrae landmarks. These landmarks correspond to the vertebral body and spinous processes centroids in both scans, and the registration applied is a 2D Euler transformation [47].

Once aligned, both segmentations are compared through the weighted DSC, and the final match is then chosen as the one with the highest DSC among the fifty candidates. Candidates are also discarded if the DSC obtained is equal to zero. After a CT slice is matched, it is removed from the possible candidates for the following MR slices, and the rigid transform applied to the segmentation is also applied to the corresponding CT image. Since this transformation may introduce empty regions due to shifts or rotations, the resulting image is cropped to the central 192x192 region. An overview of the proposed matching approach is shown in Figure 3.

Limiting the candidates to fifty strikes a balance between computational eficiency and matching accuracy, determined through empirical testing on the test dataset. As shown in Figure 4, increasing the number of candidates improves accuracy but disproportionately increases computational time. The elbow method was applied to the eficiency curve, identifying fifty candidates as the optimal trade-of point.

However, if no match is found within the first fifty candidates because there are no slices left with the same vertebrae, the system expands the candidate pool and also allows for a greater vertebral diference.

Table 2
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">MR-CT pseudo-pairs</td><td rowspan="2">Mean DSC</td><td colspan="3">Differing vertebrae</td></tr><tr><td>0</td><td>1</td><td>&gt;1</td></tr><tr><td>Train</td><td>11319</td><td>0.6374</td><td>6459</td><td>4855</td><td>5</td></tr><tr><td>Validation</td><td>3555</td><td>0.6340</td><td>1825</td><td>1728</td><td>2</td></tr><tr><td>Test</td><td>1911</td><td>0.6180</td><td>975</td><td>928</td><td>8</td></tr></table>

Vertebral matching system results for the diferent datasets. The mean DSC value represents the average overlap between the MR and CT spines in the generated pseudo-pairs. The difering vertebrae indicates the number of vertebral mismatches between the sagittal slices of every generated pseudo-pair, with 0 meaning both slices have the same vertebrae, 1 meaning there is at least one vertebra that is not present in both slices and >1 meaning there is more than 1 vertebrae that is not present in both slices.

Following this system, the MR train, validation and test sets were matched with the most similar slices belonging to the corresponding CT set. The matching results were as shown in Table 2.

## 3.2.2. MR to CT translation using SynDif

Following the vertebral matching system introduced earlier, we proceeded to train the SynDif architecture.

SynDif consists of two main components: a non-difusive module and a difusive module. The non-difusive module generates an approximation of the target modality based on features extracted from the input MR slice. The difusive module then takes the synthetic approximation and the noisy MR slice, learning to progressively denoise and refine the image to match the target CT modality. Both modules are structured as CycleGANs, ensuring that any transformations applied maintain anatomical consistency between the input and the generated target image.

For training, we performed an ablation study to evaluate the impact of incorporating the vertebral matching system. Specifically, we compared the performance of two setups: one using MR-CT sagittal slice pairs from the same spinal region (cervical, thoracic or lumbar), without any vertebrae matching or alignment, and the other one using the vertebral matching system to create pseudo-pairs based on vertebral structures similarity. For both approaches, the same MR sagittal slices from the extracted 3D subvolumes described in 3.1 were used.

Both setups were trained until stability was observed in terms of spine-related metrics. Using the original SynDif architecture and based on the original training hyperparameters, the input sagittal slices for translation were padded from 192x192 to a size of 256x256 to match the input size used in the original implementation. Other training parameters included a batch size of one, a learning rate of $1 \times 1 0 ^ { - 4 }$ , two residual blocks for the non-difusive generator, 1000 direct difusion steps, and four denoising steps for the difusive generator.

## 3.3. 3D contextual consistency enhancement

The final stage of the proposed pipeline focuses on improving the 3D reconstruction quality of the generated synthetic CT volumes. Since the MR-to-CT translation network operates in a slice-wise 2D manner, the independently generated sagittal slices may present local discontinuities when reassembled into a full 3D volume. To enhance interslice coherence and improve anatomical continuity across adjacent vertebral structures, a dedicated post-processing network was introduced. This refinement strategy aims to preserve the high in-slice image quality achieved by SynDif while improving volumetric consistency, without requiring modifications to the core translation architecture.

To train this refinement module in a supervised manner, a representative subset corresponding to approximately 10% of the CT train subvolumes described in Figure 1 was selected as training data. For each of this 670 real CT subvolumes, a corresponding pseudo-CT simulating the discontinuity artifacts observed in the stacked synthetic CT volumes, was generated by adapting the vertebral matching system described in Section 3.2.1 to operate exclusively within the CT domain. These new volumes were generated by matching each sagittal slice from the input CT volumes to the most anatomically similar slice extracted from the remaining CT scans, preserving local vertebral appearance while intentionally disrupting inter-slice continuity across the reconstructed volume as illustrated in Figure 6. The resulting pseudo-CT volumes were used as network inputs, whereas the original anatomically coherent CT volumes served as ground-truth targets for supervised learning.

For testing, the 83 synthetic CT volumes generated from reassembling the 2D translated MR sagittal slices of the test split described in Figure 1 were used.

Regarding the architecture of the post-processing network, it was implemented as a supervised 3D GAN, with a residual encoder–decoder generator and a volumetric PatchGAN discriminator (Figure 5).

The generator was designed as a fully 3D convolutional encoder–residual–decoder architecture. The encoder progressively reduced spatial resolution through strided 3D convolutions followed by instance normalization and ReLU activations. At the bottleneck, three residual blocks were applied to learn non-linear volumetric representations of interslice inconsistencies. Spatial resolution was then restored through two decoder blocks based on trilinear upsampling followed by 3D convolutions, instance normalization, and ReLU activations. To preserve fine anatomical details, skip connections linked the encoder and decoder at matching resolutions. These skip connections were further refined through additional residual units, enabling selective preservation of anatomically consistent regions while correcting local discontinuities. Additionally, a direct skip connection from the input to the output was incorporated to preserve global structural information. The final output synthetic CT volume was generated through a 3D convolution, with output values constrained to [0,1] using a clipping operation.

![](images/439012a30350940f19eaf76f830d21aa48c01debba084827321134bfe6a10610.jpg)  
Figure 3: Vertebral matching system overview. Figure illustrates how every MR sagittal slice is paired with a slice of the CT corresponding set, trying to maximize the spine similarity and overlap between each pair through a rigid registration process. Once paired, both slices are cropped to avoid empty pairings resulting from excessive rotations or shifts from rigid registration.

![](images/a9b7135a490147efb71b8546b948092a48a7a45da6503d35ce17ba8fe54dda38.jpg)

![](images/678370ed681b0cd247a445fb23dd3d231c9ff6ee9010ce34a3e08b2fa7b4d498.jpg)  
Optimal Number of Candidates Selection (Elbow Method)

![](images/d1b03ade1646b3e441a3ce84f46e1e42ac06c452780484ab4ea9af898165c86c.jpg)

![](images/86fab3b01faac92f1255b567d8602d3444358b14517c833d15d22dbb50f15068.jpg)  
Figure 4: Vertebral matching system empirical evaluation on the test dataset. The top row shows the obtained weighted DSC, execution time and matching eficiency (DSC increase per second of execution) as a function of the number of evaluated candidates. The bottom plot applies the elbow method to find the optimal trade-of between patch spinal similarity (DSC) and computationa cost, identifying 50 candidates as the optimal selection.

The discriminator followed a 3D PatchGAN design, providing adversarial feedback at the level of overlapping volumetric patches rather than the full volume. It consisted of consecutive 3D convolutional blocks with instance normalization and LeakyReLU activations, progressively increasing the number of feature channels while reducing spatial resolution. This patch-based formulation encouraged the correction of local high-frequency discontinuities and improved robustness to regional artifacts.

During training, the generator was optimized using a weighted combination of adversarial, L1, SSIM, perceptual and inter-slice consistency losses. The interslice consistency term was defined as the L1 distance between consecutive sagittal slices, encouraging consistent transitions. The perceptual loss was computed from feature representations extracted using a pretrained 3D ResNet-18 network. The total generator objective is defined in Equation 3, where the loss weights were selected through Optuna-based hyperparameter optimization to maximize refinement performance [48].

$$
\begin{array} { r l } & { g _ { \mathrm { t o t a l \_ l o s s } } = 0 . 2 1 0 5 \cdot \mathrm { a d v \_ l o s s } + 0 . 1 1 4 5 \cdot \mathrm { 1 1 \_ l o s s } } \\ & { \phantom { g _ { \mathrm { t o t a l \_ l o s s } } = } + 0 . 1 \cdot \mathrm { i n t e r \mathrm { - } s l i c e \_ c o n s i s t e n c y \_ l o s s } } \\ & { \phantom { g _ { \mathrm { t o t a l \_ l o s s } } = } + 0 . 2 5 6 2 \cdot \mathrm { s s i m \_ l o s s } } \end{array}
$$

![](images/1c54f2d1aa435ff60a8da58efe2a2cee51019d1e65cd77bf031205a12958acdf.jpg)  
Figure 5: Overview of the post-processing net architecture. The ResUNet generator takes a synthetic CT as input and produces a corrected CT output, trying to increase its volumetric consistency and coherence. Residual skip connections concatenate encoder and decoder feature maps, enabling multi-scale feature fusion and improved anatomical detail preservation. The PatchGAN discriminator classifies local regions of the output as real or fake to provide adversarial supervision.

![](images/9e69547e93875776d2d994672ee0653ca46fcae1d93eec6e3e9b162c035a635d.jpg)  
Figure 6: Example of the simulated discontinuity artifacts generated through the vertebral matching system a on real CT volume.

$$
+ 0 . 2 6 2 9 \cdot \mathrm { p e r c \_ l o s s }\tag{3}
$$

Both networks were trained with a batch size of 1 and learning rates were set to $4 \times 1 0 ^ { - 4 }$ for the generator and $8 \times 1 0 ^ { - 5 }$ for the discriminator.

## 3.4. Validation methods

To determine the quality of the proposed approach beyond qualitative assessment, we computed probability distributionbased metrics, task-specific metrics and atlas-based metrics from full images and specific vertebrae regions.

## 3.4.1. Distribution-based metrics

Distribution-level similarity between synthetic and real CT images was evaluated using Maximum Mean Discrepancy (MMD) and Fréchet Inception Distance (FID) [49].

MMD was computed on image intensity distributions using an RBF kernel. Although the same kernel was used across comparisons, raw MMD values are not directly comparable across diferent distributions, as they depend on the data statistics and they do not have a fixed scale. In other words, it is not possible to determine whether an obtained MMD value is good or bad, as there are no fixed reference values defined. Therefore, we used the MMD between the real MR and real CT distributions, which are very diferent, as a high discrepancy value. A relative MMD (rMMD) was then defined as the ratio between the MMD between the synthetic CT and real CT distributions and the MMD between the real MR and real CT distributions.

FID was computed as a complementary feature-based realism metric using Inception-v3 embeddings. Given that the network is pretrained on natural images, FID was used only as a relative comparison measure.

## 3.4.2. Task-specific metrics

Task-specific evaluation was performed to assess the anatomical utility of the generated synthetic CT images through an auxiliary vertebral segmentation task.

Because synthetic CT in spine surgery is primarily intended to improve vertebral delineation for applications such as implant planning and pedicle screw placement, we quantified how accurately vertebral structures could be recovered from the generated images. For this purpose, Dice Similarity Coeficient (DSC) was computed between the vertebral segmentation obtained from the input MR image and the segmentation extracted from the corresponding synthetic CT, assuming preservation of vertebral spatial correspondence after translation.

Volumetric and slice level evaluation was performed using TotalSegmentator, which additionally provides an indirect measure of 3D anatomical coherence due to its reliance on spatial context.

## 3.4.3. Atlas-based metrics

To further assess vertebral anatomical fidelity, atlas-based intensity metrics were computed through the reconstructed synthetic CT vertebrae.

Using the CT training set described in Section 3.1, vertebra-specific reference intensity distributions were built by extracting Hounsfield Unit (HU) values from real CT images within the corresponding TotalSegmentator vertebral masks, excluding partially visible vertebrae at image borders. These reference distributions were then compared with those obtained from the synthetic CT reconstructions.

Comparison was performed using Earth Mover’s Distance (EMD) and Jensen-Shannon Divergence (JSD), providing complementary measures of distribution agreement in terms of global intensity shifts and histogram shape similarity, respectively.

## 4. Results and discussion

All experiments were evaluated on the independent test split introduced in Section 3.1, including 1911 sagittal MR slices from 83 3D subvolumes. Table 3 summarizes the quantitative results of the proposed pipeline and Figure 7 reflects the visual results for diferent cases across the test set. Figure 9 shows the results related to volumetric consistency through vertebral segmentations of the diferent anatomical planes.

In general, introducing the vertebral matching system prior to training improves the quantitative results over the baseline. This improvement is consistently reflected in a lower vertebra-wise FID and higher mean binary and multilabel 3D Dice scores, indicating better anatomical alignment and vertebral tissue quality. This confirms that anatomically coherent pseudo-pairs provide a stronger supervision signal than the unpaired setting.

Regarding the global alignment with the CT domain, the vertebral matching system approach reduces the rMMD compared to the baseline ([0.135] vs [0.053]), suggesting a closer match to the CT intensity distribution. However, this does not translate into an improved perceptual image quality, as measured by the Full FID, which increases slightly from 77.84 to 80.88. Given the relatively small magnitude of this diference, both approaches can be considered comparable in terms of global image quality.

Reconstructing the 3D volumes through the stacking of sagittal slices also benefits from the vertebral matching system, as shown by the 3D dice score, leading to more anatomically coherent reconstructed volumes when compared to the baseline approach.

Regarding the qualitative results, it is further confirmed that both the baseline and vertebral matching system approaches learned an acceptable representation of the CT domain, with a clear delineation for the vertebral bodies. This can be appreciated especially in the lumbar and thoracic regions which are presented in row number one and row number two in shown in Figure 7.

However, the baseline is less robust in regions with high variability across patients, making the translation of structures like the upper cervical vertebrae particularly challenging. More specifically, the C1 vertebra is sometimes merged with the C2 vertebra, leading to shape mismatches when compared to the input MR.

In addition, there are regions where both models struggle, especially the sacrum, which is sometimes merged with the L5 vertebra in certain volumes, resulting in a smoother transition when compared to the rest of the lumbar vertebrae. This may be again caused by the high variability of this region and because of its diferent anatomical characteristics when compared to the rest of the spine. An example of the baseline dificulties with the cervical region and of both models with the sacrum are shown in row number three and number four of Figure 7.

Nevertheless, the regions which show the biggest anatomical diferences across approaches are the spinous processes. While the baseline approach is more susceptible to produce bigger spinous processes and merge them across the spine, the pseudo-matching approach is able to produce more distinguishable and better delineated structures. This can be observed not only in sagittal slices, but also in the axial and coronal views, where spinous processes have more regular shapes.

Finally, the 3D post-processing network visually reduces the discontinuity artifacts observed in the axial and coronal planes of the 2D approaches. Vertebrae contours in all three anatomical planes become better defined, improving volumetric consistency, as reflected by the increase in TotalSegmentator 3D DSC from 0.564 to 0.575. However volumes also get smoother, modifying the intensity of the generated vertebrae and surrounding tissues, leading to the degradation of the realism metrics, with a slight increase in the full-image FID ([80.88] vs [87.41]) and the vertebral FID (33.64 vs 50.37). Although MMD also sufers a slight decline, its value is remains close to zero, meaning that the post-processed volumes are still aligned with the CT domain intensity distribution.

The efect of the post-processing network on volumetric anatomical coherence is further highlighted in Figure 9. While the reconstructed 3D volumes obtained from the best 2D stage already preserved the overall vertebral arrangement and delineation, local discontinuities and vertebrae mismatch remained visible after slice stacking, mainly due to intensity discrepancies between predicted and real vertebrae. The proposed volumetric refinement reduced these discontinuities and produced smoother, anatomically more coherent vertebral reconstructions.

This efect is particularly evident in the 3D overlap renderings, where the refined synthetic CT segmentations show improved correspondence with the MR vertebral segmentations. These observations are consistent with the quantitative improvement observed in the volumetric TotalSegmentator DSC.

To further analyze vertebral-level fidelity, Figure 8 reports vertebra-wise EMD and JSD distributions before and after 3D refinement. The results reveal a heterogeneous behavior across vertebral regions for both the 2D MR-to-CT translation model and the post-processing network.

![](images/160d55a3f17edceea28c845ef0a74651e0e72bf13505faab8ae5bd9ecdc4dfe1.jpg)  
Figure 7: Progressive qualitative impact of each pipeline stage. Rows show representative and challenging examples across diferent spinal regions.

<table><tr><td colspan="4">Global realism</td><td colspan="4">Vertebral fidelity</td></tr><tr><td>Method</td><td></td><td></td><td>rMMD ↓ Full FID ↓ TS 3D DSC (bin.)↑ TS 3D DSC ↑ Vertebral FID ↓</td><td></td><td></td><td>EMD↓</td><td>JSD ↓</td></tr><tr><td>SynDiff baseline</td><td>0.135</td><td>77.84</td><td> $0 . 6 6 6 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 4 3 5 { \scriptstyle \pm 0 . 1 7 }$ </td><td>41.57</td><td> $1 1 1 . 2 { \pm } 6 1 . 1 $ </td><td> $0 . 3 0 8 { \pm } 0 . 1 3$ </td></tr><tr><td>+ pseudo-matching</td><td>0.053</td><td>80.88</td><td> $0 . 7 3 9 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 5 6 4 { \pm } 0 . 1 5$ </td><td>33.64</td><td> $8 5 . 7 { \pm } 4 4 $ </td><td> $0 . 2 6 7 { \scriptstyle \pm 0 . 1 3 }$ </td></tr><tr><td> $+ 3 0$  post-processing</td><td>0.074</td><td>87.41</td><td> $0 . 7 3 2 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 5 7 5 { \scriptstyle \pm 0 . 1 3 3 }$ </td><td>50.37</td><td> $7 6 . 8 { \pm } 6 3 . 2 $ </td><td> $0 . 2 1 3 { \pm } 0 . 1 4$ </td></tr></table>

Table 3  
Incremental contribution of each pipeline stage. Each row adds a component on top of the previous one. Global realism metrics evaluate the overall image quality, while vertebral fidelity metrics asses the anatomical consistency of the generated CT. TS 3D DSC reports the multiclass Dice per vertebra, while TS 3D DSC (bin.) treats all vertebrae as a single foreground class. Values are reported in a mean ± and standard deviation (when available) format computed over the 83 volumes of the test set. ↑ indicates higher is better, ↓ lower is better.

In the cervical region, high EMD and moderate JSD values are appreciated prior to 3D refinement. However, applying the post-processing network translates into a partial improvement. While the median values are reduced, the interquartile ranges are notably increased, indicating a higher resembling of the post-processed cervical vertebrae with the real ones but with a higher variability.

For the upper thoracic region (T1-T6), similar results can be appreciated. Prior to the refinement, the median values and interquartile ranges for both EMD and JSD are lower than for the cervical region. Still, after post-processing, there is a significant improvement in both EMD and JSD. This can be clearly appreciated for T6.

In contrast, the lower thoracic region (T7-T12), exhibits significantly higher median values and interquartile ranges. In addition, 3D refinement is not able to reduce these values, increasing some of them and highlighting T7 as a challenging vertebra. These dificulties may be attributed to its lower anatomical representation in the training cohort.

As for the lumbar region, values remained similar before and after post-processing, with the best EMD and JSD values across the spine. Still, a slight degradation is observed at L1.

Finally, the sacrum remained the most challenging region, showing a significantly higher median and larger interquartile ranges for both EMD and JSD. These values further deteriorate after post-processing.

This behavior is consistent with the qualitative findings, where partial shape distortions and local smoothing persisted even after volumetric refinement. Nevertheless, the overall distribution alignment remained within an acceptable range for most vertebral levels, supporting the anatomical fidelity of the proposed framework.

![](images/be3ea7f4d24444d4c73809e5bc37c25644713c86b683fa182f20bdd28803a2e7.jpg)

![](images/de7abd18e18c73ef192e9cf8bd1daf8f5a927534e90bbecd11ee7228dfea5c49.jpg)

![](images/2b8d06022d4050740a4c2def0560494d899c153e780af9b17552b1db88c3825f.jpg)

![](images/d31d32636fe8cace5c9e7d0b6455964b89019d312c9fea3c6a3fe560226c0fbf.jpg)  
Figure 8: Vertebra-wise intensity fidelity before and after 3D refinement.

![](images/e5e878a0393750de5e1fe8b5ac1b36f3396a462ab10d227208ba3f1729d97cad.jpg)  
Figure 9: Volumetric vertebral coherence before and after 3D post-processing, seen through the vertebral segmentations in the diferent anatomical planes. The left column shows the MR ground truth segmentation, the middle column the synthetic CT segmentation obtained with the 2D translation model and the vertebral matching system approach and the right column shows the result after applying the 3D post-processing network.

Despite the overall improvements, several limitations remain. The sacrum and posterior vertebral elements continue to be the most failure-prone structures, likely due to their higher anatomical variability and comparatively lower representation in the available datasets. Similarly, thoracic performance remained less stable than cervical and lumbar regions, which may be explained by the scarcity of publicly available full-thoracic sagittal MR data. In addition, although the post-processing network improved volumetric coherence, it introduced mild smoothing in non-bone tissues, which negatively afected global full-image realism metrics.

Finally, the lack of paired MR-CT ground truth prevents the use of voxel-wise similarity metrics and limits the evaluation to distributional and downstream task-based validation. Likewise, FID should be interpreted cautiously, as its naturalimage feature extractor may not fully capture the anatomical characteristics of medical images.

## 5. Conclusions

In this work, we proposed a fully automatic pipeline for the translation of T1-weighted MR scans to CT scans. The method relies on a pseudo-pairing system which matches and aligns unpaired MR-CT sagittal slices with a similar vertebral anatomy through their spine segmentations, followed by a difusive adversarial training with the SynDif network. This pipeline was evaluated using a 2D approach followed by a post-processing network to better address spatial coherence in the predicted volumes.

Our experiments demonstrate that the 2D approach benefits significantly from the proposed vertebral matching system. This system enables more consistent spine alignment between modalities, resulting in improved structural fidelity in the synthetic CT scans, particularly in the vertebral regions. Although overall image quality metrics are slightly worse, this study was primarily oriented toward producing synthetic CT scans to support preoperative surgical planning, where anatomical accuracy in key regions is critical, with less focus on the translation of the rest of the structures.

From a volumetric perspective, the proposed 2D approach struggled to maintain anatomical coherence across axial and coronal views. Although precise and validated volumetric segmentation algorithms were able to achieve an acceptable 3D segmentation, qualitative results showed the presence of discontinuity artifacts across slices.

The post-processing net, which aimed to solve this issue, successfully eliminated the artifacts and achieved a slight improvement in the vertebral anatomical fidelity across all planes. However, it also had a negative impact on the quality of the 2D sagittal slices, reducing the accuracy of vertebral tissue intensities and overall image realism.

This trade-of suggests that future work should explore refinement strategies that preserve 2D intensity fidelity and realism while improving vertebral continuity.

Potential directions include the development of fully 3D translation networks that can better capture anatomical continuity across slices, more sophisticated post-processing techniques that reduce discontinuity artifacts while preserving fine-grained details, and the integration of additional anatomical priors or multi-modal constraints to further enhance structural fidelity. Moreover, exploring hybrid approaches that combine 2D eficiency with 3D spatial awareness may provide a practical balance between computational cost and volumetric accuracy.

Although the clinical adoption of synthetic CT generation tools may still face important regulatory and validation challenges, particularly in surgical applications where reliability and safety are critical, the proposed approach represents a promising step toward their future integration into preoperative planning workflows. In spine surgery, CT is often required for accurate assessment of bony anatomy and is commonly registered with intraoperative fluoroscopy or navigation systems. Therefore, the ability to derive CT-like information directly from preoperative MRI could potentially reduce the need for additional CT acquisitions while preserving the anatomical information required for surgical planning and image-guided procedures.

To conclude, our results highlight the potential of rigidregistration-based pseudo-pairing in MR-to-CT translation and establish a strong baseline for further developments toward more spatially coherent and clinically useful synthetic imaging.

## Acknowledgements

This work has been partially funded by the Basque Government under the HAZITEK 2024 Programme (grant number ZE-2024/00030). Ethical approval and consent requirements were addressed by the original dataset providers. Additionally, ethical approval for the current study was obtained by Comité de Ética de la Investigación con medicamentos de Euskadi (CEIm-E) on October 23, 2025, with code PS2025063.

## References

[1] A. P. Dhawan, Medical image analysis, John Wiley & Sons, 2011.

[2] G. K. Harada, Z. K. Siyaji, S. Younis, P. K. Louie, D. Samartzis, H. S. An, Imaging in spine surgery: current concepts and future directions, Spine surgery and related research 4 (2020) 99–110.

[3] R. Oulbacha, S. Kadoury, Mri to ct synthesis of the lumbar spine from a pseudo-3d cycle gan, in: 2020 IEEE 17th international symposium on biomedical imaging (ISBI), IEEE, 2020, pp. 1784–1787.

[4] S. Dayarathna, K. T. Islam, S. Uribe, G. Yang, M. Hayat, Z. Chen, Deep learning based synthesis of mri, ct and pet: Review and analysis, Medical image analysis 92 (2024) 103046.

[5] J. McNaughton, J. Fernandez, S. Holdsworth, B. Chong, V. Shim, A. Wang, Machine learning for medical image translation: A systematic review, Bioengineering 10 (2023) 1078.

[6] M. Bajger, M.-S. To, G. Lee, A. Wells, C. Chong, M. Agzarian, S. Poonnoose, Lumbar spine ct synthesis from mr images using cyclegan-a preliminary study, in: 2021 Digital Image Computing: Techniques and Applications (DICTA), IEEE, 2021, pp. 1–8.

[7] D. Vallejo-Cendrero, J. M. Molina-Maza, B. Rodriguez-Gonzalez, D. Viar-Hernandez, B. Rodriguez-Vila, J. Soto-Pérez-Olivares, J. Moujir-López, C. Suevos-Ballesteros, J. Blázquez-Sánchez, J. Acosta-Batlle, et al., Cyclegan-driven mr-based pseudo-ct synthesis for knee imaging studies, Applied Sciences 14 (2024) 4655.

[8] J. M. Wolterink, A. M. Dinkla, M. H. F. Savenije, P. R. Seevinck, C. A. T. van den Berg, I. Išgum, Deep mr to ct synthesis using unpaired data, in: Simulation and Synthesis in Medical Imaging: Second International Workshop, SASHIMI 2017, Held in Conjunction with MICCAI 2017, Québec City, QC, Canada, September 10, 2017, Proceedings 2, Springer, 2017, pp. 14–23.

[9] P. Paavilainen, S. U. Akram, J. Kannala, Bridging the gap between paired and unpaired medical image translation, in: MICCAI Workshop on Deep Generative Models, Springer, 2021, pp. 35–44.

[10] M. Özbey, O. Dalmaz, S. U. H. Dar, H. A. Bedel, Ş. Özturk, A. Güngör, T. Çukur, Unsupervised medical image translation with adversarial difusion models, IEEE Transactions on Medical Imaging 42 (2023) 3524–3539.

[11] B. Cao, H. Zhang, N. Wang, X. Gao, D. Shen, Auto-gan: selfsupervised collaborative learning for medical image synthesis, in: Proceedings of the AAAI conference on artificial intelligence, volume 34, 2020, pp. 10486–10493.

[12] D. Nie, X. Cao, Y. Gao, L. Wang, D. Shen, Estimating ct image from mri data using 3d fully convolutional networks, in: Deep Learning and Data Labeling for Medical Applications: First International Workshop, LABELS 2016, and Second International Workshop, DLMIA 2016, Held in Conjunction with MICCAI 2016, Athens, Greece, October 21, 2016, Proceedings 1, Springer, 2016, pp. 170–178.

[13] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, Advances in neural information processing systems 27 (2014).

[14] X. Gu, Y. Zhang, W. Zeng, S. Zhong, H. Wang, D. Liang, Z. Li, Z. Hu, Cross-modality image translation: Ct image synthesis of mr brain images using multi generative network with perceptual supervision, Computer Methods and Programs in Biomedicine 237 (2023) 107571.

[15] M. Zhao, F. Bao, C. Li, J. Zhu, Egsde: Unpaired image-to-image translation via energy-guided stochastic diferential equations, Advances in Neural Information Processing Systems 35 (2022) 3609–3623.

[16] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).

[17] J. Ho, A. Jain, P. Abbeel, Denoising difusion probabilistic models, Advances in neural information processing systems 33 (2020) 6840– 6851.

[18] D. Nie, R. Trullo, J. Lian, L. Wang, C. Petitjean, S. Ruan, Q. Wang, D. Shen, Medical image synthesis with deep convolutional adversarial networks, IEEE Transactions on Biomedical Engineering 65 (2018) 2720–2730.

[19] K. Armanious, C. Jiang, M. Fischer, T. Küstner, T. Hepp, K. Nikolaou, S. Gatidis, B. Yang, Medgan: Medical image translation using gans, Computerized medical imaging and graphics 79 (2020) 101684.

[20] H. Emami, M. Dong, S. P. Nejad-Davarani, C. K. Glide-Hurst, Sagan: Structure-aware gan for organ-preserving synthetic ct generation, in: Medical Image Computing and Computer Assisted Intervention– MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part VI 24, Springer, 2021, pp. 471–481.

[21] J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, Unpaired image-toimage translation using cycle-consistent adversarial networks, in: Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223–2232.

[22] H. Yang, J. Sun, A. Carass, C. Zhao, J. Lee, J. L. Prince, Z. Xu, Unsupervised mr-to-ct synthesis using structure-constrained cyclegan, IEEE transactions on medical imaging 39 (2020) 4249–4261.

[23] Y. Hiasa, Y. Otake, M. Takao, T. Matsuoka, K. Takashima, A. Carass, J. L. Prince, N. Sugano, Y. Sato, Cross-modality image synthesis from unpaired data using cyclegan: Efects of gradient consistency loss and training data size, in: Simulation and Synthesis in Medical Imaging: Third International Workshop, SASHIMI 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Proceedings 3, Springer, 2018, pp. 31–41.

[24] Y. Lei, J. Harms, T. Wang, Y. Liu, H.-K. Shu, A. B. Jani, W. J. Curran, H. Mao, T. Liu, X. Yang, Mri-only based synthetic ct generation using dense cycle consistent generative adversarial networks, Medical physics 46 (2019) 3565–3581.

[25] L. Zhong, Z. Chen, H. Shu, K. Zheng, Y. Li, W. Chen, Y. Wu, J. Ma, Q. Feng, W. Yang, Multi-scale tokens-aware transformer network for multi-region and multi-sequence mr-to-ct synthesis in a single model, IEEE transactions on medical imaging 43 (2023) 794–806.

[26] B. Zhao, T. Cheng, X. Zhang, J. Wang, H. Zhu, R. Zhao, D. Li, Z. Zhang, G. Yu, Ct synthesis from mr in the pelvic area using residual transformer conditional gan, Computerized medical imaging and graphics 103 (2023) 102150.

[27] Y. Li, H.-C. Shao, X. Liang, L. Chen, R. Li, S. Jiang, J. Wang, Y. Zhang, Zero-shot medical image translation via frequency-guided difusion models, IEEE transactions on medical imaging 43 (2023) 980–993.

[28] Y. Luo, Q. Yang, Z. Liu, Z. Shi, W. Huang, G. Zheng, J. Cheng, Target-guided difusion models for unpaired cross-modality medical image translation, IEEE Journal of Biomedical and Health Informatics (2024).

[29] Z. Wang, H. Zheng, P. He, W. Chen, M. Zhou, Difusion-gan: Training gans with difusion, arXiv preprint arXiv:2206.02262 (2022).

[30] P. Dhariwal, A. Nichol, Difusion models beat gans on image synthesis, Advances in neural information processing systems 34 (2021) 8780– 8794.

[31] E. M. V. N. Karthik, C. Laporte, F. Cheriet, Three-dimensional segmentation of the scoliotic spine from mri using unsupervised volume-based mr-ct synthesis, in: Medical Imaging 2021: Image Processing, volume 11596, SPIE, 2021, pp. 402–409.

[32] K. H. Kim, E.-C. Lee, Y. D. Yoon, D.-W. Shin, H.-W. Koo, B.-J. Lee, Translation of computed tomography images to t2-weighted magnetic resonance images of lumbar spine using generative adversarial networks on sagittal images, Scientific Reports 15 (2025) 1–11.

[33] M. Roberts, G. Hinton, A. J. Wells, J. Van Der Veken, M. Bajger, G. Lee, Y. Liu, C. Chong, S. Poonnoose, M. Agzarian, et al., Imaging evaluation of a proposed 3d generative model for mri to ct translation in the lumbar spine, The Spine Journal 23 (2023) 1602–1612.

[34] Y. Chen, N. Konz, H. Gu, H. Dong, Y. Chen, L. Li, J. Lee, M. A. Mazurowski, Contourdif: Unpaired image translation with contourguided difusion models, arXiv preprint arXiv:2403.10786 2 (2024) 4.

[35] R. Graf, J. Schmitt, S. Schlaeger, H. K. Möller, V. Sideri-Lampretsa, A. Sekuboyina, S. M. Krieg, B. Wiestler, B. Menze, D. Rueckert, et al., Denoising difusion-based mri to ct image translation enables automated spinal segmentation, European Radiology Experimental 7 (2023) 70.

[36] J. Cohen-Adad, E. Alonso-Ortiz, M. Abramovic, C. Arneitz, N. Atcheson, L. Barlow, R. L. Barry, M. Barth, M. Battiston, C. Büchel, et al., Generic acquisition protocol for quantitative mri of the spinal cord,

Nature protocols 16 (2021) 4611–4632.

[37] A. Sharafi, K. Koch, Dataset of instrumented spinal cord t1/t2-weighted mri scans, 2024. doi:10.17632/sjpx7md7cf.1.

[38] J. W. van der Graaf, M. L. van Hoof, C. F. Buckens, et al., Spiderlumbar spine segmentation in mr images: a dataset and a public benchmark, Boston: Zendo 10 (2023).

[39] J. Wasserthal, T. Akinci D’Antonoli, Totalsegmentator mri dataset: 616 mri images with segmentations for 50 anatomical regions, 2025. doi:10.5281/zenodo.11367004, [Data set].

[40] P. F. Raudaschl, P. Zafino, G. C. Sharp, M. F. Spadea, A. Chen, B. M. Dawant, T. Albrecht, T. Gass, C. Langguth, M. Lüthi, et al., Evaluation of segmentation methods on head and neck ct: auto-segmentation challenge 2015, Medical physics 44 (2017) 2020–2036.

[41] G. Podobnik, P. Strojan, P. Peterlin, B. Ibragimov, T. Vrtovec, Hanseg: The head and neck organ-at-risk ct and mr segmentation dataset, Medical physics 50 (2023) 1917–1927.

[42] T. Bejarano, M. Ornelas-Couto, I. Mihaylov, Head-and-neck squamous cell carcinoma (hnscc) patients with 3d ct taken during pre-treatment, mid-treatment, and post-treatment dataset, 2018.

[43] S. Pieper, N. Haouchine, D. B. Hackney, W. M. Wells, M. Sanhinova, T. Balboni, A. Spektor, M. Huynh, S. Tanguturi, E. Kim, J. P. Guenette, D. E. Kozono, B. Czajkowski, S. Caplan, P. Doyle, H. Kang, R. N. Alkalay, Spine metastatic bone cancer: pre and post radiotherapy ct (spine-mets-ct-seg) [dataset], 2024. doi:doi.org/10.7937/kh36-ds04.

[44] J. Wasserthal, H.-C. Breit, M. T. Meyer, M. Pradella, D. Hinck, A. W. Sauter, T. Heye, D. T. Boll, J. Cyriac, S. Yang, et al., Totalsegmentator: robust segmentation of 104 anatomic structures in ct images, Radiology: Artificial Intelligence 5 (2023) e230024.

[45] Y. Deng, C. Wang, Y. Hui, Q. Li, J. Li, S. Luo, M. Sun, Q. Quan, S. Yang, Y. Hao, et al., Ctspine1k: A large-scale dataset for spinal vertebrae segmentation in computed tomography, arXiv preprint arXiv:2105.14711 (2021).

[46] N. J. Tustison, B. B. Avants, P. A. Cook, Y. Zheng, A. Egan, P. A. Yushkevich, J. C. Gee, N4itk: improved n3 bias correction, IEEE transactions on medical imaging 29 (2010) 1310–1320.

[47] B. C. Lowekamp, D. T. Chen, L. Ibáñez, D. Blezek, The design of simpleitk, Frontiers in neuroinformatics 7 (2013) 45.

[48] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A nextgeneration hyperparameter optimization framework, in: Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 2623–2631.

[49] C. N. Valdebenito Maturana, A. L. Sandoval Orozco, L. J. García Villalba, Exploration of metrics and datasets to assess the fidelity of images generated by generative adversarial networks, Applied Sciences 13 (2023) 10637.