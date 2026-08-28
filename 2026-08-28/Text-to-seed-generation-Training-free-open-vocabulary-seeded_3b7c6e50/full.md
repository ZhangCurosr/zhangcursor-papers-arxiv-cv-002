# Text-to-seed generation: Training-free open-vocabulary seeded semantic segmentation via re-purposing diffusion as text-guided seed generator

Kumju Jo<sup>1</sup> Heesun Jung<sup>2</sup> , Sungyong Baik<sup>1,2†</sup>   
<sup>1</sup>Dept. of Artificial Intelligence, Hanyang University <sup>2</sup>Dept. of Data Science, Hanyang University {juice0630,jheesun,dsybaik}@hanyang.ac.kr

## Abstract

Open-vocabulary semantic segmentation (OVSS) aims to segment image regions corresponding to arbitrary text queries. Although the Segment Anything Model (SAM) is a powerfulfoundation modelfor segmentation, its standalone performance on OVSS remains limited. Existing methods therefore often use SAM to refine coarse masks predicted by other models, but this strategy is unreliable when the initial masks are inaccurate. In this work, we argue that more reliable segmentation can be achieved by exploiting SAM as a region expansion module guided by accurate object points (i.e., seeds) rather than inaccurate coarse masks. Inspired by classical seeded segmentation, we reformulate OVSS as text-guided seed localization followed by seedbased region expansion. To realize this idea, we propose Text-to-Seed (T2S), a training-free framework that leverages the text-to-region correspondence of Stable Diffusion to generate attention-based seedpointsfor target categories described by text. These sparse seeds are then used as point promptsfor SAM toproducefull object masks. Without taskspecific training or additional annotations, T2S achieves strong performance on standard OVSS benchmarks, demonstrating the effectiveness of combining semantic grounding with seed-driven spatial segmentation.

© 2026. This manuscript version is made available under the CC-BY-NC-ND 4.0 license https://creativecommons.org/ licenses/by-nc-nd/4.0/

## 1. Introduction

Open-vocabulary semantic segmentation (OVSS) [1–3] aims to handle open-set categories in arbitrary texts, generalizing beyond predefined categories. OVSS task demands for the generalization capability in modeling correspondences between visual and textual representations.

To tackle such challenging task, few works have attempted to employ foundation models for segmentation (particularly, Segment Anything Model (SAM) [4]). While SAM exhibits strong capability in learning spatially coherent visual representations, SAM faces challenges with text prompts and semantic understanding [5]. Thus, recent works either fine-tune SAM with vision-language models or use SAM to refine coarse segmentation masks or bounding boxes [6–8]. However, fine-tuning is susceptible to risks of suboptimal performance on unseen categories, while refinement approaches are limited by the quality of coarse segmentation masks or bounding boxes.

![](images/4e6c8bc1a3ae4aefa3e0f6d117dac278c851f01fc72913b65bc3cb9708818367.jpg)  
Figure 1. Overview of our framework, dubbed T2S. We extract seeds from attention maps of Stable Diffusion. Then, seeds are fed into SAM to perform region expansion.

Another line of works [8, 9] focuses on employing text-to-image generative model, namely Stable Diffusion (SD) [10], for its capability to spatially align visual features and textual features. SD is used to either directly estimate segmentation masks from attention maps [11–14] or generate synthetic images to prepare semantic features [8, 9, 15]. However, since SD is originally designed for image generation, these approaches are often faced with the difficulty of generating high-quality masks [16] or recognizing multiple subjects in an image.

In this work, we tackle OVSS from the perspective of seeded segmentation [17], which comprises two stages: seed initialization to localize an object and region expansion to find similar features and segment the whole object. From this perspective, we draw connections between seeded segmentation and the capabilities of SAM and SD. Notably, the spatial text-image alignment capability of SD can be used for seed initialization; while, the spatial coherence modeling proficiency of SAM for region expansion. Specifically, we leverage the attention maps in SD to localize a queried category with initial seed. This initial seed is then passed to SAM as a point prompt to obtain a high-quality mask. The overall framework, dubbed Text-to-seed (T2S), is illustrated in Fig. 1.

Experimental results highlight the outstanding performance of our proposed framework across various datasets. We note that our framework is a plug-and-play framework that is designed for SAM and SD to be seamlessly incorporated in an off-the-shelf manner, without any additional training. The strong performance and flexibility of T2S corroborate our motivation and idea that draws connections with classical seeded segmentation, re-purposing SD for seed generation and SAM for region expansion.

## 2. Related Works

Traditional semantic segmentation tasks have focused on assigning each pixel with pre-defined categories [18–21]. Despite rapid and notable progresses with deep learning, the reliance on predefined categories has limited its generalization capability. To take a step further and generalize beyond such assumption, open-vocabulary semantic segmentation (OVSS) aims to assign each pixel with any given category prompted by arbitrary text queries.

To tackle OVSS, many works have employed Vision-Language Models (VLMs) [22–28], especially CLIP, trained on a large-scale dataset of text-image pairs. However, CLIP is originally trained for modeling image-level correspondences with textual features, exhibiting poor spatial coherence and thus providing poor mask quality. Many works have tried to endow CLIP with spatial coherence modeling via fine-tuning [2, 29–32]. Yet, these works still struggle with achieving high-quality masks for arbitrary categories. Recent methods [33, 34] improve the spatial representation of frozen CLIP, with CLIPer further utilizing diffusion self-attention to capture fine-grained spatial details. However, these methods mainly improve the spatial consistency of CLIP-derived predictions without introducing an independent text-conditioned localization signal, and thus remain dependent on the semantic quality of the initial CLIP predictions.

Recent studies have shown that text-guided Diffusion Models (DMs), particularly Stable Diffusion (SD) [10], not only achieve strong text-to-image generation performance but also encode rich semantic and spatial cues in their latent representations and attention maps [35–37]. Such controllability and representational richness have enabled DMs to be extended to diverse generation-related tasks, including controllable generation through attention guidance [37, 38], attention-based image editing [36, 39, 40], fair generation with attention modulation [41], personalization using attention maps or attention modules [42–44], concept erasure [45, 46], and debiasing with editing models [47]. Motivated by these advances, several works have investigated the use of DMs for open-vocabulary semantic segmentation (OVSS). Few works [8, 9, 11–15, 48] have attempted to leverage the generative capability of SD to generate several images for a given category, extracting region features from synthetic images. The extracted region features are then compared against features of a given real image to perform region-to-region feature matching, bypassing the difficulties of patch-text correspondence modeling. Meanwhile, other works [12–15] utilize attention maps and of text-to-image correspondence modeling capability of SD to directly obtain semantic masks. More recently, [49] further combines CLIP cross-attention and Stable Diffusion selfattention with an entropy-guided random walk for mask refinement. However, direct generation of semantic masks from attention maps have often led to coarse masks.

In parallel, few works [6, 8, 50] have attempted to tackle OVSS with Segment Anything Model (SAM) [4], a recent vision foundation model for object segmentation. However, such capability is only applicable to class-agnostic masks, lacking the semantic understanding [5]. Although recent SAM models (i.e., SAM 3) [51] have improved text-prompt handling, their semantic understanding is effectively limited to simple noun phrases. This inherent constraint in semantic grounding indicates that these models still fail to independently process arbitrary open-vocabulary queries. GroundedSAM [50] introduces a workaround by performing openvocabulary object detection with Grounding DINO and then feeding a bounding box into SAM to perform segmentation. Meanwhile, few works [6, 8] have attempted to refine their coarse masks with masks produced by SAM in postprocessing. However, initial poor-quality masks can limit the extent of refinement SAM masks can give during postprocessing.

Recent methods [52, 53] further integrate SAM into CLIP-based dense inference using SAM-derived correlations or region masks. These methods perform spatial refinement using CLIP-based initial predictions or classagnostic region priors from SAM; therefore, their final performance can still depend on the quality of the initial semantic localization and region proposals. On the other hand, our proposed framework, named Text-to-Seed (T2S), better exploits the spatial coherence capabilities of SAM by drawing connections with seeded segmentation. In particular, T2S re-purposes the diffusion model as a text-guided seed generator to produce point-level accurate localization (seeds); based on these, T2S leverages SAM for region expansion rather than for conventional coarse mask or bounding box refinement.

## 3. Background

## 3.1. Diffusion Model

Stable Diffusion (SD) is one of instances of [10] a textguided diffusion model designed to generate an image x from random noise $\boldsymbol { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , based on a text embedding $\tau ( y )$ obtained from an input text prompt y of length $P$ through a text encoder τ. Diffusion models work under the assumption that z and x are linked by a Markov chain, which incrementally adds noise to $\mathbf { \boldsymbol { x } } = \boldsymbol { z } _ { \mathrm { 0 } }$ , transforming it into $z = z _ { T }$ over a predetermined number of timesteps $T \colon$

$$
q ( z _ { t } | z _ { t - 1 } ) = \mathcal { N } ( z _ { t } ; \sqrt { 1 - \beta _ { t } } z _ { t - 1 } , \beta _ { t } \mathbf { I } ) ,\tag{1}
$$

where $\beta _ { t }$ denotes the noise or variance schedule.

From this perspective, an image x can be viewed as a denoised version of z. To perform such denoising process to generate x, a network $\epsilon _ { \theta } .$ , parameterized by θ, is trained to predict the added noise $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ at each timestep t, conditioned on the text prompt y, by minimizing the following objective function:

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , t , \tau ( \boldsymbol { y } ) } \left[ \big \| \boldsymbol { \epsilon } - \boldsymbol { \epsilon } _ { \theta } ( z _ { t } , t , \tau ( \boldsymbol { y } ) ) \big \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{2}
$$

## 3.2. Aggregation of attention maps

The denoising network in Stable Diffusion is composed of residual and transformer blocks. Each transformer block includes cross-attention (CA) and self-attention (SA) layers, with a total of $L = 1 6$ transformer blocks. At the l-th transformer layer, the l-th CA layer merges information from a conditioning text prompt embedding $\boldsymbol { \tau } ( y ) \in \mathbb { R } ^ { P \times d _ { \tau } }$ and latent features $\boldsymbol { z } _ { t } ^ { l } \in \dot { \mathbb { R } } ^ { H _ { l } \times W _ { l } \times d _ { z } }$ from a previous layer, facilitating image generation aligned with the conditioning text. We can calculate the CA map $\mathcal { A } _ { l , t } ^ { \mathrm { C A } } \in \mathbb { R } ^ { H _ { l } \times W _ { l } \times P }$ from CA layer, at each timestep t as follows:

$$
\mathcal { A } _ { l , t } ^ { \mathrm { C A } } = \operatorname { s o f t m a x } \left( \frac { Q _ { l , t } ^ { z } \cdot K _ { l } ^ { \tau ^ { \top } } } { \sqrt { d _ { l } } } \right) ,\tag{3}
$$

where $Q _ { l , t } ^ { z }$ and $K _ { l } ^ { \tau }$ are the query matrix of $ { \boldsymbol { z } } _ { t } ^ { l }$ and the key matrix of $\tau ( y )$ , respectively, obtained via linear projections. $d _ { l }$ denotes the latent feature dimension at layer l. The $l -$ th SA layer, meanwhile, captures spatial similarities within $\boldsymbol { z } _ { t } ^ { l } ,$ generating an SA map $\overset { \cdot } { \mathcal { A } } _ { l , t } ^ { \mathrm { S A } } \in \dot { \mathbb { R } } ^ { H _ { l } \times W _ { l } \times H _ { l } \times W _ { l } }$ by the following:

$$
\mathcal { A } _ { l , t } ^ { \mathrm { S A } } = \operatorname { s o f t m a x } \left( \frac { Q _ { l , t } ^ { z } \cdot K _ { l , t } ^ { z } ^ { \top } } { \sqrt { d _ { l } } } \right) ,\tag{4}
$$

where $Q _ { l , t } ^ { z }$ and $K _ { l , t } ^ { z }$ are the query and key matrices of $\boldsymbol { z } _ { t } ^ { l } .$ Stable Diffusion’s U-Net architecture, with attention layers in both the encoder and decoder sections, generates

attention maps at four different resolutions: $( H _ { l } , W _ { l } ) \ \in$ $\{ ( s _ { r } , s _ { r } ) \} _ { r = 0 } ^ { 3 }$ with $s _ { r } \in \{ 8 , 1 6 , 3 2 , 6 4 \}$

The numerous attention maps generated by Stable Diffusion can be simplified by grouping them by resolution and then averaging and normalizing them to a 0-1 range:

$$
\bar { \mathcal { A } } _ { s _ { r } } = \frac { 1 } { \vert \mathbb { L } _ { s _ { r } } \vert \cdot T } \sum _ { l \in \mathbb { L } _ { s _ { r } } } \sum _ { 1 \le t \le T } \frac { \mathcal { A } _ { l , t } } { \operatorname* { m a x } ( \mathcal { A } _ { l , t } ) } ,\tag{5}
$$

The set $\mathbb { L } _ { s _ { r } }$ contains the indices of layers that produce attention maps at resolution $s _ { r } ,$ and $\bar { \mathcal { A } } _ { s _ { r } }$ denotes the aggregated attention map for that resolution. $\bar { \mathcal { A } } _ { s _ { r } } ^ { \mathrm { C A } }$ and $\bar { \boldsymbol { A } } _ { s _ { r } } ^ { \mathrm { S A } }$ represent the aggregated CA and SA maps, respectively.

## 4. Proposed Method

Motivated by classical seeded segmentation, our proposed method, Text-to-Seed (T2S) with the overview illustrated in Fig. 2(a), performs iterative seed initialization, region expansion, and classification. Overall, T2S consists of four stages. The first stage is dedicated to improving text-spatialvisual-feature alignment information from attention maps of SD (Sec. 4.1, Fig. 2(b)), serving as seed generator in the following section. The set of obtained attention maps, acting as seed generator, is then used to extract seeds that locate a queried category (Sec. 4.2, Fig. 2(c)). Then, seeds are refined to higher resolution via iterative seed generation and spreading procedure (Sec. 4.3, Fig. 2(d)). The extracted seeds are then fed into Segment Anything Model (SAM) as point prompts to perform region expansion (Sec. 4.4, Fig. 2(e)).

## 4.1. Initialization of seed generator: enhancing attention maps of stable diffusion

In seeded segmentation, precise mask generation hinges on an accurate seed initialization, locating a queried category. In this work, we aim to obtain an accurate seed initialization from Stable Diffusion (SD) [10]. SD is a diffusion model with U-Net architecture composed of self-attention (SA) layers and cross-attention (CA) layers. Guided by text prompts, SD learns to generate images $( t = 0 )$ from random Gaussian noise $( t = T )$ , where t is a denoising step and $T$ is a predefined number of denoising steps.

To extract accurate initial seeds from SD, we first aim to amplify the signal from a target category c in a given text prompt. We take motivation from recent analysis on DMs that the attention maps corresponding to End-of-Text ([EOT]) embeddings contain semantic information of objects present in the same text [54]. Upon the motivation, we up-weight [EOT] embeddings by a factor of two to strengthen the signal of a target category in the attention maps.

Next, we decide which diffusion denoising steps to collect attention maps from. We first note that SD is trained to generate images, while we aim to find text-visual-region correspondences in real images. At the early reverse diffusion denoising step (t = T), an image is random Gaussian image, generating noisy attention maps. As denoising step t approaches 0, an image becomes a real image, generating accurate attention maps. Thus, it is reasonable to start aggregating attention maps after the half of denoising process $( t = T / 2 )$ , when an image introduces noise. Fig. 3 validates our claim in that an aggregated attention map seems noisy when accumulating attention maps from early denoising steps. Thus, we obtain aggregated crossattention (ACA) maps $\tilde { \mathcal { A } } _ { s _ { r } , c _ { \mathrm { c l s } } } ^ { \mathrm { C A } } ~ \in ~ \mathbb { R } ^ { s _ { r } \times s _ { r } }$ and aggregated self-attention (ASA) maps $\tilde { \mathcal { A } } _ { s _ { r } , c _ { \mathrm { c l s } } } ^ { \mathrm { S A } } \ \in \ \mathbb { R } ^ { s _ { r } \times s _ { r } \times s _ { r } \times s _ { r } }$ from $t = 0 \mathrm { t o } t = T / 2$ , for each resolution $\{ s _ { r } \times s _ { r } \} _ { r = 0 } ^ { 3 } .$ , where $s _ { r } \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ and $c _ { \mathrm { c l s } }$ is the index of a class token in the text prompt.

![](images/b642da94036e1efbf5cbd0a2637e882be9c3094a31082682efa1757a1b47e1c4.jpg)  
(a) Overall framework

![](images/60c5a7d716703305571cb358143a6cd1af19245b406e0de61e49f5096584bd75.jpg)  
(c) Seed Generation and Spreading

![](images/281d0187f34336e0ffab96327fdf04553a9f4c4c5a5c5fec24aa149374a76820.jpg)

![](images/d3c66e50959c5acf50775b5fff5552013b8e674400d45c4ddfc15853ab31d26f.jpg)  
(b) Initialization of Seed Generator

![](images/7679e663a423c73553add36b06fc17ec5bbaf29a4f4335a94852d294dcfe5d84.jpg)  
(d) Iterative Seed Generation and Spreading  
(e) Region Expansion via SAM

Figure 2. (a) The simplified overview of our framework. (b) Initialization of seed generator is performed via Stable Diffusion (SD) with EoT token weighting, followed by aggregating attention maps. Once attention maps are aggregated, SD is no longer needed. (c) Seed generation and Spreading (SGS) block extracts and spreads seeds from aggregated attention maps to localize a given category. (d) Iterative Seed Generation and Spreading process iteratively applies SGS block that extracts both positive seeds for categories and negative seeds for irrelevant or background, which are iteratively refined to high resolution. Since this process is performed on already aggregated attention maps, it can be performed efficiently. Then, (e) Region Expansion process via SAM is performed by feeding positive seeds and negative seeds into SAM as point prompts to perform region expansion to get a final mask.  
![](images/29d28250e803658b6ce8559a2eb7c2a4d37076e8400cb34c82847d0dfc032f01.jpg)  
Figure 3. Visualization of how aggregated cross-attention (ACA) maps change with different starting denoising steps.

![](images/d9dfd682920e2dab09245bb03a2d4933baaaba553f8dd1677e3d66fb26e5242b.jpg)  
Figure 4. Visualization of cross-attention aggregation (ACA) maps at different resolutions in Stable Diffusion.

## 4.2. Seed initialization

ACA maps $\tilde { \mathcal { A } } _ { s _ { r } , c _ { \mathrm { c l s } } } ^ { \mathrm { C A } }$ quantify the correspondence between textual features and visual latent features at each attention map coordinate $( i , j )$ for each resolution $s _ { r }$ . We observe that attention tends to disperse as resolution increases, losing the precision of category localization, as shown in Fig. 4. Thus, we employ ACA maps at resolutions $8 \times 8$ and $1 6 \times 1 6$ . In particular, we first extract initial seeds from ACA maps at resolution $8 \times 8 , \tilde { A } _ { 8 , c _ { \mathrm { c l s } } } ^ { \mathrm { C A } }$ , by gathering spatial coordinates at which ACA value exceeds a threshold α:

$$
\mathbb { S } _ { 1 } = \{ ( i , j ) | \tilde { \mathcal { A } } _ { 8 , c _ { \mathrm { c l s } } } ^ { \mathrm { C A } } [ i , j ] \ge \alpha \} .\tag{6}
$$

## 4.3. Iterative seed generation and spreading

While $\mathbb { S } _ { 1 }$ may contain seeds that point at a query category, they may capture only a small portion of a category and also fail to attend to other multiple instances. We also note that higher-resolution ASA provides fine-grained attention maps, while lower-resolution ACA provides better localization. Upon this observation, we use ACA and ASA to perform an iterative seed generation and spreading to gradually refine seeds to higher resolution, obtaining more precise coordinates of seeds.

To this end, we first “spread” current extracted seeds $\mathbb { S } _ { k }$ by using ASA maps $\tilde { \mathcal { A } } _ { s _ { k } , c _ { \mathrm { c l s } } } ^ { \mathrm { S A } }$ to locate features similar to extracted seeds $\mathbb { S } _ { 1 }$ :

$$
\mathcal { M } _ { s _ { k } } ^ { \prime } = \frac { 1 } { \vert \mathbb { S } _ { k } \vert } \sum _ { ( i , j ) \in \mathbb { S } _ { k } } \tilde { \mathcal { A } } _ { s _ { k } } ^ { \mathrm { S A } } [ i , j , : , : ] .\tag{7}
$$

This aggregated attention mask $\mathcal { M } _ { s _ { k } } ^ { \prime }$ now attends to visual features that are similarly to current extracted seeds $\mathbb { S } _ { k } .$ . Then, $\mathcal { M } _ { s _ { k } } ^ { \prime }$ is upsampled via bilinear-upsampling to higher resolution to obtain its higher-resolution counterpart $\mathcal { M } _ { s _ { k + 1 } } \in \mathbb { R } ^ { s _ { k + 1 } \times s _ { k + 1 } } \mathrm { : }$

$$
\mathcal { M } _ { s _ { k + 1 } } = \mathrm { b i l i n e a r - u p s a m p l } \mathbf { e } _ { s _ { k + 1 } } ( \mathcal { M } _ { s _ { k } } ^ { \prime } ) .\tag{8}
$$

When the resolution is $1 6 \times 1 6$ , we further leverage ACA map $\tilde { \mathcal { A } } _ { 1 6 , c _ { \mathrm { c l s } } } ^ { \mathrm { C A } }$ for its better semantic correspondence information, compared to higher resolutions. To this end, we combine $\mathcal { M } _ { 1 6 }$ and $\tilde { \mathcal { A } } _ { 1 6 , c _ { \mathrm { c l s } } } ^ { \mathrm { C A } ^ { - } }$ via point-wise max operation to capture all visual cues (e.g., seeds):

$$
\mathcal { M } _ { 1 6 }  \operatorname* { m a x } ( \tilde { A } _ { 1 6 } ^ { \mathrm { C A } } , \mathcal { M } _ { 1 6 } ) .\tag{9}
$$

Through the above process, the point-wise transformation of the ACA map $\tilde { A } _ { 1 6 , c _ { \mathrm { c l s } } } ^ { \bar { \mathrm { C A } } }$ can be observed in Fig. 6. Again, we extract seeds from $\ddot { \mathcal { M } } _ { s _ { k + 1 } }$

$$
\mathbb { S } _ { k + 1 } = \{ ( i , j ) \mid \mathcal { M } _ { s _ { k + 1 } } [ i , j ] \geq \alpha \} .\tag{10}
$$

![](images/fefeb9f2c82829f5a12652bfb4ee41343eeec8d15fb76d33db1b64c331055f44.jpg)

Figure 5. Illustration of the process that iteratively aggregates masks before generating the final mask.  
![](images/9d0f1a87fe320d0414c9da9dc642c71a0f53c877705360fe120e74a8295b415c.jpg)  
Figure 6. Visualization of how aggregated self-attention mask $\mathcal { M } _ { s _ { k } }$ evolves over the iterations in iterative seed generation and spreading. Through the progress, $\mathcal { M } _ { s _ { k } }$ is shown to exhibit more fine-grained and sparse attention masks. Thus, at the end of the iterative seed generation and spreading, we can extract precise seeds at a resolution of 64 × 64, which are then passed to SAM for region expansion.

Then, we iteratively repeat the process of seed generation and seed spreading using Eq. 7, 8, 10 until the last iteration $K$ that corresponds to the highest resolution of ASA maps $( s _ { K } = 6 4 )$

## 4.4. Region expansion via SAM

In this stage, we aim to employ SAM to perform region expansion: finding a high-quality mask region, pointed by extracted seeds from the previous stage. SAM is known to give a more precise segmentation mask, when positive point prompts (targets) and negative point prompts (nontargets). Thus, similar to how we obtain positive point prompts (seeds for categories) in the previous stage, we obtain negative point prompts (seeds for background or other irrelevant features) by simply iteratively extracting seeds and seed spreading, with a small change in Eq. 10:

$$
\mathbb { S } _ { k + 1 } ^ { \mathrm { n e g } } = \{ ( i , j ) \mid \mathcal { M } _ { s _ { k + 1 } } ^ { \mathrm { n e g } } [ i , j ] \leq \beta \} ,\tag{11}
$$

where $\mathcal { M } _ { s _ { k + } } ^ { \mathrm { n e g } }$ is the negative-point-counterpart of $\mathcal { M } _ { s _ { k + 1 } }$ obtained with negative seeds from a previous resolution $\mathbb { S } _ { k } ^ { \mathrm { n e g } }$ . To extract negative point prompts, we now look for points at which cross-attention values are low.

Then, the set of positive seeds $\mathbb { S } _ { K }$ and negative seeds $\mathbb { S } _ { K } ^ { \mathrm { n e g } }$ extracted at the last iteration $K \ ( \mathbb { S } _ { \mathrm { S A M } } = \mathbb { S } _ { K } \cup \mathbb { S } _ { K } ^ { \mathrm { n e g } } )$

Table 1. Comparison with open-vocabulary semantic segmentation models on Pascal VOC [21], Cityscapes [20], ADE20K [19], COCO objects [18] and Pascal Context [55]. The performance of other models is reported as in the respective papers. Best and second best results are highlighted in bold and underlined, respectively.
<table><tr><td>Model</td><td></td><td></td><td></td><td></td><td colspan="7">mIoU</td></tr><tr><td></td><td>training-free</td><td>supervision</td><td>preprocess</td><td>SAM</td><td>VOC-20</td><td>VOC-21</td><td>Cityscapes</td><td>ADE</td><td>Objects</td><td>Pascal59</td><td>Pascal60</td></tr><tr><td>Weakly-supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReCo [56]</td><td>x</td><td>√</td><td>x</td><td>x</td><td>57.7</td><td>25.1</td><td>21.1</td><td>11.2</td><td>15.7</td><td>22.3</td><td>19.9</td></tr><tr><td>GroupViT [29]</td><td>x</td><td>√</td><td>x</td><td>x</td><td>79.7</td><td>50.4</td><td>11.1</td><td>9.2</td><td>27.5</td><td>23.4</td><td>17.7</td></tr><tr><td>MaskCLIP [57]</td><td>x</td><td>√</td><td>x</td><td>x</td><td>74.9</td><td>38.8</td><td>12.6</td><td>9.8</td><td>20.6</td><td>26.4</td><td>23.6</td></tr><tr><td>TCL [32]</td><td>x</td><td>√</td><td>x</td><td>x</td><td>83.2</td><td>55.0</td><td>24.0</td><td>17.1</td><td>31.6</td><td>33.9</td><td>30.4</td></tr><tr><td>SAM-CLIP [7]</td><td>x</td><td>√</td><td>x</td><td>√</td><td>=</td><td>60.6</td><td>=</td><td>17.1</td><td></td><td></td><td>29.2</td></tr><tr><td colspan="2">CLIP-based training-free</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCLIP [58]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>67.5</td><td>43.8</td><td>19.5</td><td>11.3</td><td>24.6</td><td>25.6</td><td>23.5</td></tr><tr><td>ProxyCLIP [59]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>80.3</td><td>61.2</td><td>38.1</td><td>20.1</td><td>39.1</td><td>39.3</td><td>38.3</td></tr><tr><td>CaR [6]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>91.6</td><td>67.5</td><td>15.9</td><td>17.8</td><td>37.5</td><td>39.7</td><td>31.5</td></tr><tr><td>NACLIP [33]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>83.0</td><td>64.1</td><td>38.3</td><td>19.1</td><td>36.2</td><td>38.4</td><td>35.0</td></tr><tr><td>CASS [60]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>87.8</td><td>65.8</td><td>39.4</td><td>20.4</td><td>37.8</td><td>40.2</td><td>36.7</td></tr><tr><td>CLIPer [34]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>89.8</td><td>72.2</td><td>42.5</td><td>25.0</td><td>44.7</td><td>44.6</td><td>39.5</td></tr><tr><td colspan="2">Diffusion-based training-free</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OVDiff [13]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>79.8</td><td>67.4</td><td>17.5</td><td>13.1</td><td>40.6</td><td>30.6</td><td>29.1</td></tr><tr><td>DiffSegmenter [12]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>=</td><td>60.1</td><td>=</td><td></td><td>37.9</td><td>=</td><td>27.5</td></tr><tr><td>FreeDA [9]</td><td>√</td><td>x</td><td>√</td><td>x</td><td>86.2</td><td>55.4</td><td>35.9</td><td>23.1</td><td>37.4</td><td>43.4</td><td>39.2</td></tr><tr><td>FreeSeg-Diff [15]</td><td>√</td><td>x</td><td>x</td><td>x</td><td></td><td>53.3</td><td></td><td></td><td>31.1</td><td>=</td><td></td></tr><tr><td>NERVE [49]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>90.1</td><td>69.7</td><td>34.1</td><td>24.0</td><td>43.3</td><td>43.4</td><td>37.7</td></tr><tr><td colspan="2">SAM-based training-free</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Grounded-SAM [50]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>81.5</td><td></td><td>26.9</td><td>15.3</td><td>41.5</td><td>44.0</td><td></td></tr><tr><td>RIM [8]</td><td>√</td><td>x</td><td>√</td><td>√</td><td></td><td>77.8</td><td></td><td></td><td>44.5</td><td></td><td>34.3</td></tr><tr><td>CaR+SAM [6]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>91.7</td><td>67.9</td><td>16.0</td><td>17.9</td><td>37.7</td><td>40.0</td><td>31.6</td></tr><tr><td>Trident [52]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>88.7</td><td>70.8</td><td>47.6</td><td>26.7</td><td>42.2</td><td>44.3</td><td>40.1</td></tr><tr><td>T2S (Ours)</td><td>√</td><td>x</td><td>x</td><td>√</td><td>92.1</td><td>74.2</td><td>47.3</td><td>27.1</td><td>45.6</td><td>47.9</td><td>41.7</td></tr></table>

is fed into SAM along with an input image x to obtain a segmentation mask m. Then, the segmented region is classified by feeding a masked image m ⊙ x into CLIP. Although we start with cross-attention maps that attend to visual attributes of a target category, iterative seed generation and spreading process on small resolutions (i.e., $8 \times 8 , 1 6 \times 1 6 , 3 2 \times 3 2 , 6 4 \times 6 4 )$ can lead to false positives. Thus, we leverage strong zero-shot classification capability of CLIP to filter out such false positives.

## 4.5. Iterative seeded semantic segmentation

Since cross-attention maps of SDs often seem to struggle with identifying multiple subjects, we propose to iteratively perform seed generation and region expansion from Sec. 4.2 to Sec. 4.4, followed by masking each found mask region out from 16 × 16 aggregated attention map $\tilde { \mathcal { A } } _ { 1 6 } ^ { \mathrm { C A } }$ , such that seeds will be extracted at different location in each iteration. We repeat this procedure until CLIP classification re jects segmented regions for more than n (n = 10 in this work) times in a row or all image regions are covered. Fig. 5 demonstrates the effectiveness of the iterative seeded semantic segmentation, where masked attention maps become more fine-grained and sparse, providing more accurate localization as the procedure proceeds.

Table 2. Ablation on our proposed module. Init denotes the Initialization of Seed Generator, SG denotes Seed Generator, ISSS denotes Iterative Seeded Semantic Segmentation from our method. All experiments are conducted on Pascal VOC 20.
<table><tr><td rowspan="2">Init Token</td><td colspan="2">SE</td><td rowspan="2">ISSS Iterative</td><td rowspan="2">VOC</td></tr><tr><td>Seed generation</td><td>Negative seed</td></tr><tr><td>weight –</td><td>√</td><td></td><td>SSS</td><td>mIoU 73.1</td></tr><tr><td>√</td><td>√</td><td>- –</td><td>- 一</td><td>76.8</td></tr><tr><td>√</td><td>√</td><td>√</td><td>–</td><td>82.3</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>92.1</td></tr></table>

## 5. Experiments

## 5.1. Dataset and Evaluation Metric

We evaluate our method Text-to-Seed (T2S) on common benchmarks: Pascal VOC 2012 [21], Pascal Context [55], Cityscapes [20], COCO object [18], ADE20K [19]. Following previous works [13], we consider the background class in Pascal VOC 2012 and Pascal Context dataset. Pascal VOC 2012 is consist of 1449 images with 20 classes for VOC-20 and 21 classes for VOC-21. Pascal Context

Input image

T2S (Ours)

CaR

Input image

T2S (Ours)  
CaR  
![](images/b2c54381729f50f093c27d6b8d3752cf1090148836194175112382577bf08bb0.jpg)  
Figure 8. Qualitative results of T2S under multi subject scenarios.

comprises 5104 images with 59 classes for Pascal59 and 60 classes for Pascal60. Cityscapes includes 500 images with 19 classes without background class. COCO object includes 5000 images with 80 classes. ADE20k comprises 2000 images with 150 classes without background class. Following standard evaluation protocol, the mean Intersection-over-Union (mIoU) metric is used to evaluate the models.

## 5.2. Implementation Details

We utilize the Stable Diffusion v1.4 to extract CA map and SA map conditioned on input text prompt. We utilize the text prompt ‘A photo of <class>’ to condition the SD model when extracting the CA map. For region classification, we utilize the CLIP model with ViT-L/14 as a backbone, as in previous works [6, 59]. We use the seed generation threshold parameter α as 0.5 and β as 0.3 to extract the seeds from initialization map at each resolution of the SD model. All experiments are performed on NVIDIA RTX 4090.

Table 3. Ablation study on the selection of the timestep at which to initiate the denoising process.
<table><tr><td></td><td>3/4 T</td><td>1/2T</td><td>1/4T</td></tr><tr><td>mIoU</td><td>90.4</td><td>92.1</td><td>89.7</td></tr></table>

## 5.3. Quantitative results

Tab. 1 reports the performance of recent OVSS methods and our method, T2S across several datasets, namely VOC-20, VOC-21, Cityscapes, ADE, Objects, Pascal59, and Pascal60. All the results of other methods are taken from the numbers reported in the respective papers, except for Grounded-SAM, which is reported in [61]. The results demonstrate the outstanding performance of our method T2S, compared to previous works that employ CLIP, Diffusion, SAM, or the combination.

While the original paper of SAM [4] suggests that SAM supports the text prompt, the official implementation of SAM does not support the text prompt. GroundedSAM [50] provides a simple workaround to the lack of the semantic modeling capability of SAM. However, GroundedSAM relies on the bounding box produced by Grounding DINO, which is trained on annotated bounding boxes. Another work [7] also fine-tunes SAM and CLIP with mask annotation for OVSS. The reliance on such rich annotation exposes the risks of poor performance on unseen categories. In parallel, other works employ SAM to refine the initial masks produced by CLIP [6] or Stable Diffusion [8]. Such frameworks rely on the quality of initial masks, unable to fully exploit the capabilities of SAM.

On the other hand, our proposed framework, T2S, better exploits the spatial coherence capabilities of SAM by utilizing it as region expansion. Furthermore, in contrast to previous works that employ SD for initial mask generation, T2S utilizes SD for seed initialization (i.e., object localization), which is less susceptible to noise and errors, compared to the whole masks.

## 5.4. Ablation Study

We conduct ablation studies to evaluate the effectiveness of the various modules introduced in our methodology. Ablation study experiments are performed on Pascal VOC 2012 with mean Intersection over Union (mIoU) as the primary metric to quantitatively assess the influence of each design choice and module.

Module Ablation. Tab. 2 presents the results of an ablation study evaluating the effectiveness of each module: namely, token weighting in the initialization of seed generator; negative seed point prompt; and iterative procedure of seed generation and spread. Using seed generation with SAM for mask generation as the baseline model, we progressively apply each module and observe the performance difference. Our basic pipeline generates point prompts from the aggregated attention maps of SD, which can be considered as applying only seed generation from Seed Generator. Then, to get more precise point prompts, we utilize token weighting process from Initialization of Seed Generator, which has indeed led to performance improvement. Extracting and feeding negative point prompts to T2S has led to substantial performance improvement. Introducing iterative procedure to better handle multiple objects has been observed to provide even more performance improvement.

Table 4. mIoU by aggregated cross-attention (ACA) map resolu tions used for seed initialization in SGS.
<table><tr><td>Resolutions</td><td>mIoU</td></tr><tr><td>8</td><td>88.2</td></tr><tr><td>8,16</td><td>92.1</td></tr><tr><td>8, 16, 32</td><td>82.4</td></tr><tr><td>8, 16, 32, 64</td><td>76.3</td></tr></table>

Table 5. mIoU by seed generation threshold α.
<table><tr><td>α</td><td>mIoU</td></tr><tr><td>0.3</td><td>87.2</td></tr><tr><td>0.4</td><td>91.7</td></tr><tr><td>0.5</td><td>92.1</td></tr><tr><td>0.6</td><td>90.3</td></tr><tr><td>0.7</td><td>84.2</td></tr></table>

Table 6. Ablation study comparing fixed iteration counts (R) and our adaptive rejection-based strategy using CLIP.
<table><tr><td></td><td>R= 1</td><td>R= 5</td><td>Ours</td></tr><tr><td>mIoU</td><td>82.3</td><td>86.8</td><td>92.1</td></tr></table>

Ablation on hyper-parameters. Tab. 3 presents the ablation study on the starting diffusion denoising step to extract aggregated attention maps. The first row of the table indicates the timestep at which the denoising process begins, with T set to 75 steps. We observe that initiating the denoising from the midpoint, at 1/2T, leads to the best performance as it allows for greater focus on the object. This is because images become close to Gaussian distribution and hence noisier, as the denoising is performed at earlier steps.

Tab. 4 illustrates the effect of the resolution of attention maps. Using higher resolution of attention maps for seed generation initialization leads to a decrease in performance. This is because attention disperses at higher resolution, losing the category localization information, as discussed in Sec. 4.1 and illustrated in Fig. 4.

Table 7. Generalization performance on ISPRS Potsdam dataset.
<table><tr><td colspan="2">CaR+SAM</td><td>T2S (Ours)</td></tr><tr><td>mIoU</td><td>81.3</td><td>83.2</td></tr></table>

Table 8. Computational cost of T2S. Runtime, GPU memory, and model-call counts are reported per image, while the average number of iterations is reported per queried class. CLIP calls correspond to mask verification.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Inference time</td><td>188 s/image (3.13 min)</td></tr><tr><td>Peak GPU memory</td><td>~20 GB</td></tr><tr><td>Average number of iterations</td><td>9.1/class</td></tr><tr><td>SAM image-encoder calls</td><td>1/image</td></tr><tr><td>SAM mask-decoder calls</td><td>311.7/image</td></tr><tr><td>CLIP mask-verification calls</td><td>21.8/image</td></tr></table>

Table 9. The analysis of computational complexity with respect to the number of objects, measured in GFLOPs.
<table><tr><td></td><td>n = 1 </td><td>n = 2 n = 3</td><td></td><td> $\overline { { n > 5 } }$ </td></tr><tr><td>Seed Generator</td><td>3623</td><td>3630</td><td>3618</td><td>3651</td></tr></table>

Table 10. Peak GPU memory and computational-cost comparison with representative training-free OVSS baselines on VOC-20. Memory usage is reported in GB, and computational cost is reported in GFLOPs.
<table><tr><td colspan="2">GPU Memory</td><td>GFLOPs</td></tr><tr><td>CaR [6]</td><td>8.6 GB</td><td>338</td></tr><tr><td>FreeDA [9]</td><td>23.4 GB</td><td>588</td></tr><tr><td>OVDiff [13]</td><td>21.0 GB</td><td>1492</td></tr><tr><td>T2S (Ours)</td><td>20.0 GB</td><td>3851</td></tr></table>

Tab. 5 reports the effect of seed generation threshold parameter α. Without extensive hyperparameter search, α = 0.5 is shown to provide the best performance. Tab. 6 represents an ablation study on the number of iterations R during Iterative Seeded Semantic Segmentation. We evaluate the performance when the number of iterations R is fixed to either 1 and 5 and compare it with our final version, where the number of iterations R is dynamically adjusted instance-wise using the rejection count n (n = 10 in our work). In other words, T2S iteratively extracts masks until the number of rejections by CLIP reaches n = 10. The outstanding performance by T2S compared to other variants suggests the importance of iteratively collecting masks.

![](images/b072ff496b75da1563a5de9ed2561d84939a2aaaff5e358894d3818262171c75.jpg)  
Figure 9. Qualitative results of our method on background classes in the Pascal59 dataset.

![](images/9b5d564a0c115a33688e0381d248aebdfed6d2086a3cecad1265ae5573e38d2f.jpg)  
Figure 10. Qualitative comparison of T2S and CaR on small objects and occlusion.

This is because aggregated cross-attention maps provide precise localization of one object at a time. The ablation study results validate that we have overcome such limitation of Stable Diffusion via iterative seed generation and expansion. However, the iterative procedure does not require the multiple passes of SD for each category, as we perform seed generation and expansion on aggregated attention maps, which need to be extracted from SD only once for each category. Tab. 8 reports the practical inference cost of T2S. T2S requires 188 s per image and approximately 20 GB of peak GPU memory, with an average of 9.1 iterations per queried class. As shown in Tab. 9, the seedgeneration cost varies by less than 1% across the evaluated object-count bins (3.62–3.65 TFLOPs), indicating that this stage is largely insensitive to the number of objects. Tab. 10 further compares the peak GPU memory usage and computational cost (GFLOPs) of T2S with those of representative training-free OVSS baselines on VOC-20. T2S maintains peak GPU memory usage comparable to that of diffusionbased baselines while incurring a higher computational cost than CaR. This overhead mainly arises from repeated SAM mask decoding and CLIP-based candidate verification, and it remains a practical limitation. Nevertheless, these iterative operations enable the discovery of multiple target instances and the rejection of false-positive candidates, both of which are integral to the T2S pipeline.

![](images/91f5697d4ce6001dc5cac647c5e54c73edaf7f5866b8870b126aebf3b509ea21.jpg)  
(c) Single-step SAM3 prompting using manual point or text prompts  
Figure 11. Qualitative comparison of T2S with simpler alternatives.

## 5.5. Qualitative Results

Fig. 8 illustrates the qualitative results of T2S under multiobject scenarios. These results show that T2S can produce high-quality masks in multi-object scenes, despite the known limitation of Stable Diffusion in modeling multiple objects. The results validate the effectiveness of our proposed iterative seeded semantic segmentation procedure in overcoming the limitation.

Meanwhile, Fig. 7 exhibits the qualitative results of T2S under Open-Vocabulary Semantic Segmentation (OVSS)

scenarios. The high-quality masks produced by T2S for arbitrary categories, particularly fictional characters, validate the effectiveness of employing Stable Diffusion to localize objects while using SAM to perform region expansion. It would be difficult to obtain the high-quality masks for such arbitrary categories, if SAM or other components are finetuned on a fixed number of categories with bounding box or mask annotations as in [7, 50]. On the other hand, T2S exploits the text-spatial-image-feature alignment capability of SD and the spatial coherence modeling ability of SAM in an off-the-shelf manner, leading to high-quality masks for arbitrary categories. In particular, SD is trained on a large collection of web-crawled image-caption pairs, while SAM is trained to segment general objects without semantic understanding. Thus, SD enables T2S to obtain an accurate localization of arbitrary objects, while SAM helps T2S segment localized categories.

T2S further demonstrates robust mask generation under complex visual conditions, including intricate background structures (Fig. 9) and small occluded objects (Fig. 10). Specifically, Fig. 10 shows that T2S recovers precise boundaries for small occluded targets, whereas CaR misses several small objects as highlighted by dotted red circles.

Fig. 11 further compares T2S with simpler alternatives that use the same underlying components without the proposed seed-generation-and-spreading formulation. Directly thresholding diffusion attention maps often produces incomplete or noisy masks, since attention maps provide localization cues rather than full object regions. Selecting top-k attention responses as point prompts improves boundary quality through SAM, but remains vulnerable to incorrect attention peaks and missed instances. Direct SAM3 prompting can generate accurate masks when reliable manual or text prompts are available, but such prompts are difficult to obtain for arbitrary named entities and multiinstance scenes. In contrast, T2S automatically derives textconditioned seed prompts through CA-based localization and ASA-based seed spreading, enabling SAM to perform region expansion without manual prompts or coarse mask inputs.

Fig. 12 complements Figs. 9 and 10 by providing qualitative evaluations across varying object scales and scene complexities. These visual comparisons confirm the overall robustness and accuracy of T2S. Fig. 13 illustrates potential failure modes, indicating that T2S may encounter difficulties under severe occlusion, faint object boundaries, or target/non-target spatial overlaps, leading to unsegmented regions or boundary leakage into adjacent objects and background regions.

## 5.6. Evaluation on Out-of-Domain images

To further stress-test the generalization capability of T2S, we evaluate T2S and one of the recent state-of-the-art models, CaR+SAM [6], on aerial and satellite images, namely the ISPRS Potsdam [62] dataset. Such aerial and satellite images substantially differ from the datasets used for training SAM or Diffusion models. The results reported in Tab. 7 show that T2S outperforms CaR+SAM, which is one of the recent state-of-the-art models, demonstrating the strong generalization capability of our proposed framework.

![](images/08014aa164eb1ba85d7a9f0731fcb1c2ba33cc38ac27682b533d77c74d49496b.jpg)

![](images/d58df3c6ebe8ffd5b3f9cedb03c105b9bc4ecc4e1ffd4ce2b6c59f040fdc2f85.jpg)  
(a) large objects  
(b) small objects

![](images/fb59c08730cbf8dacd9c0008eef9dcd7f9dee4627a15b2ec5012c9c056c8e43b.jpg)  
(c) complex scene

Figure 12. Qualitative analysis across object scales and scene complexity.  
![](images/6b7b18bee3d75a986989577c92e3d87c52f68255ed89f6f003bda0c97299bbfc.jpg)

T2S (Ours)  
Ground Truth  
![](images/edda9006a80671442c0038f407fcf922411be90efc7819900543c893c6578e91.jpg)

![](images/18068602a150fb0dd2d65a7902cf76ef99943b114df08b9e598f62070c85e48b.jpg)

(a) Successful case  
![](images/db4fb85aaff1bbec5826d4c582ad1310905eff7cfa2ea78b077ddaf2e0278b11.jpg)

![](images/9f828082dd7ecb604507ba1e216cbb47883cd2c0594a8267c2315a99edcb40ae.jpg)

![](images/9599a5b87b11a573070d2d08515f0ac8e78efb318713c78062c1687c7725128d.jpg)

Failure reason: Occlusion  
![](images/73114d31cfff33d788b524ad6d5e8a894e573ae72b800e059abdc689f8e56b74.jpg)

![](images/ec1ad9136daec58f9d70735bebd6bf34118ba92edd577cb2ee56d5ad16e1fa7a.jpg)

![](images/e97575d9c4842f0403a0ec5142c39dd3c96ccb30d4e51b306399f355b7e67929.jpg)  
Failure reason: Weak object boundaries

![](images/86631baa3ea08df10e399881a65a81ab107bc8fd95dfd6aa83018ec2796c70ec.jpg)

![](images/9a7b1afbeed8cfe90ff96ca795c9e1e3d684083f9f935437e841a28f1761e254.jpg)

![](images/9ba4a3ed4e6338208efc9f2e233e43d665741645a5bf43bb4d7b913a279fc9b2.jpg)  
Failure reason: Overlap  
(b) Failure cases  
Figure 13. Qualitative results with failure analysis. (a) Successful case shows the very fine-grained segmentation output. (b) The failure cases show the object missed by T2S (highlighted by a dotted red box). T2S may fail in scenes with severe occlusion, overlapping target and non-target regions, or weak object boundaries. These cases can lead to missed instances or leakage into nearby objects.

## 6. Conclusion

In this work, we introduce Text-to-Seed (T2S), a trainingfree approach that recasts to open-vocabulary semantic segmentation as text-guided seed localization and seed-based region expansion. By using Stable Diffusion to generate attention-based seed points and SAM to expand them into full object masks, T2S avoids the limitations of coarse-mask refinement while effectively combining semantic grounding with spatial segmentation. Results on standard OVSS benchmarks show that the proposed framework achieves strong performance without additional training or annotations. Overall, our findings demonstrate the promise of using SAM as a seed-driven expansion model and suggest a practical direction for more flexible open-vocabulary segmentation.

## 7. Acknowledgments

This work was supported by the Institute of Information and communications Technology Planning and evaluation (IITP) grants (No.RS-2025-25422680, No. RS-2020- II201373) and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2025-24533064).

## References

[1] M. Bucher, T.-H. Vu, M. Cord, and P. Perez, “Zero-shot´ semantic segmentation,” in NeurIPS, 2019. 1

[2] M. Xu, Z. Zhang, F. Wei, Y. Lin, Y. Cao, H. Hu, and X. Bai, “A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model,” in ECCV, 2022. 2

[3] G. Ghiasi, X. Gu, Y. Cui, and T.-Y. Lin, “Scaling openvocabulary image segmentation with image-level labels,” in ECCV, 2022. 1

[4] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in ICCV, 2023. 1, 2, 8

[5] M. Espinosa, C. Yang, L. Ericsson, S. McDonagh, and E. J. Crowley, “There is no samantics! exploring sam as a backbone for visual understanding tasks,” 2024. 1, 2

[6] S. Sun, R. Li, P. Torr, X. Gu, and S. Li, “Clip as rnn: Segment countless visual concepts without training endeavor,” in CVPR, 2024. 1, 2, 6, 7, 8, 9, 11

[7] H. Wang, P. K. A. Vasu, F. Faghri, R. Vemulapalli, M. Farajtabar, S. Mehta, M. Rastegari, O. Tuzel, and H. Pouransari, “Sam-clip: Merging vision foundation models towards semantic and spatial understanding,” in CVPRW, 2024. 6, 8,

[8] Y. Wang, R. Sun, N. Luo, Y. Pan, and T. Zhang, “Imageto-image matching via foundation models: A new perspective for open-vocabulary semantic segmentation,” in CVPR, 2024. 1, 2, 6, 8

[9] L. Barsellotti, R. Amoroso, M. Cornia, L. Baraldi, and R. Cucchiara, “Training-free open-vocabulary segmentation with offline diffusion-augmented prototype generation,” in CVPR, 2024. 1, 2, 6, 9

[10] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022. 1, 2, 3

[11] L. Barsellotti, R. Amoroso, L. Baraldi, and R. Cucchiara, “Fossil: Free open-vocabulary semantic segmentation through synthetic references retrieval,” in WACV, 2024. 1, 2

[12] J. Wang, X. Li, J. Zhang, Q. Xu, Q. Zhou, Q. Yu, L. Sheng, and D. Xu, “Diffusion model is secretly a training-free open vocabulary semantic segmenter,” IEEE Transactions on Image Processing, 2025. 2, 6

[13] L. Karazija, I. Laina, A. Vedaldi, and C. Rupprecht, “Diffu sion models for zero-shot open-vocabulary segmentation,” in ECCV, 2024. 6, 9

[14] K. Namekata, A. Sabour, S. Fidler, and S. W. Kim, “Emerdiff: Emerging pixel-level semantic knowledge in diffusion models,” in ICLR, 2024. 1

[15] B. T. Corradini, M. Shukor, P. Couairon, G. Couairon, F. Scarselli, and M. Cord, “Freeseg-diff: Trainingfree open-vocabulary segmentation with diffusion models,” 2024. 1, 2, 6

[16] W. Wu, Y. Zhao, H. Zhou, M. Z. Shou, and C. Shen, “Diffumask: Synthesizing images with pixel-level annotations for semantic segmentation using diffusion models,” in ICCV, 2023. 1

[17] J. Zhang, J. Zheng, and J. Cai, “A diffusion approach to seeded image segmentation,” in CVPR, 2010. 1

[18] T.-Y. Lin, M. Maire, S. J. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco:´ Common objects in context,” in ECCV, 2014. 2, 6

[19] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, “Scene parsing through ade20k dataset,” in CVPR, 2017. 6

[20] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in CVPR, 2016. 6

[21] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” International journal of computer vision, 2010. 2, 6

[22] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021. 2

[23] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in ICML, 2021.

[24] Y. Li, H. Fan, R. Hu, C. Feichtenhofer, and K. He, “Scaling language-image pre-training via masking,” in CVPR, 2023.

[25] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “Eva-clip: Improved training techniques for clip at scale,” 2023.

[26] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu, “Coca: Contrastive captioners are imagetext foundation models,” Transactions on Machine Learning Research, 2022.

[27] X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and L. Beyer, “Lit: Zero-shot transfer with locked-image text tuning,” in CVPR, 2022.

[28] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in ICCV, 2023. 2

[29] J. Xu, S. De Mello, S. Liu, W. Byeon, T. Breuel, J. Kautz, and X. Wang, “Groupvit: Semantic segmentation emerges from text supervision,” in CVPR, 2022. 2, 6

[30] J. Mukhoti, T.-Y. Lin, O. Poursaeed, R. Wang, A. Shah, P. H. Torr, and S.-N. Lim, “Open vocabulary semantic segmentation with patch aligned contrastive learning,” in CVPR, 2023.

[31] J. Xu, J. Hou, Y. Zhang, R. Feng, Y. Wang, Y. Qiao, and W. Xie, “Learning open-vocabulary semantic segmentation models from natural language supervision,” in CVPR, 2023.

[32] J. Cha, J. Mun, and B. Roh, “Learning to generate textgrounded mask for open-world semantic segmentation from only image-text pairs,” in CVPR, 2023. 2, 6

[33] S. Hajimiri, I. B. Ayed, and J. Dolz, “Pay attention to your neighbours: Training-free open-vocabulary semantic segmentation,” in WACV, 2025. 2, 6

[34] L. Sun, J. Cao, J. Xie, X. Jiang, and Y. Pang, “Cliper: Hierarchically improving spatial representation of clip for openvocabulary semantic segmentation,” in ICCV, 2025. 2, 6

[35] R. Tang, L. Liu, A. Pandey, Z. Jiang, G. Yang, K. Kumar, P. Stenetorp, J. Lin, and F. Ture, “What the daam: Interpret-¨ ing stable diffusion using cross attention,” in ACL, 2023. 2

[36] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, “Prompt-to-prompt image editing with cross-attention control,” in ICLR, 2023. 2

[37] H. Chefer, Y. Alaluf, Y. Vinker, L. Wolf, and D. Cohen-Or, “Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models,” 2023. 2

[38] W. Feng, X. He, T.-J. Fu, V. Jampani, A. R. Akula, P. Narayana, S. Basu, X. E. Wang, and W. Y. Wang, “Training-free structured diffusion guidance for compositional text-to-image synthesis,” in ICLR, 2023. 2

[39] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng, “Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing,” in ICCV, 2023. 2

[40] S. Bagon, T. Dekel, N. Tumanyan, and M. Geyer, “Plugand-play diffusion features for text-driven image-to-image translation,” in CVPR, 2022. 2

[41] D. Kim, J. Kim, and S. Baik, “Training-free debiasing of diffusion models via clip-guided denoising optimization,” arXiv preprint arXiv:2607.00817, 2026. 2

[42] Y. Sun, B. Liu, X. Chen, R. Song, and J. Fu, “Vico: Engaging video comment generation with human preference rewards,” in MM, 2024. 2

[43] S. Kim, K. Shin, H. Jung, J. Kim, and S. Baik, “Decoupled guidance: Disentangling subject and context pathways in text-to-image personalization,” arXiv preprint arXiv:2607.00766, 2026.

[44] J. Ma, J. Liang, C. Chen, and H. Lu, “Subject-diffusion: Open domain personalized text-to-image generation with out test-time fine-tuning,” in SIGGRAPH, 2024. 2

[45] J. Ahn, S. Yoon, and S. Baik, “Egloce: Training-free energy-guided latent optimization for concept erasure,” arXiv preprint arXiv:2604.09405, 2026. 2

[46] C.-P. Huang, K.-P. Chang, C.-T. Tsai, Y.-H. Lai, F.-E. Yang, and Y.-C. F. Wang, “Receler: Reliable concept erasing of text-to-image diffusion models via lightweight erasers,” in ECCV, 2024. 2

[47] J. Seo, Y. Park, C. Lee, and S. Baik, “Biasedit: A trainingfree bias-detect-and-edit framework for learning fair visual classifiers,” in WWW, 2026. 2

[48] J. H. Park, K. Jo, and S. Baik, “Seediff: Off-the-shelf seeded mask generation from diffusion models,” in AAAI, 2025. 2

[49] K. Mahatha, J. Dolz, and C. Desrosiers, “Nerve: Neighbourhood & entropy-guided random-walk for training free open-vocabulary segmentation,” in WACV, 2026. 2, 6

[50] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assem bling open-world models for diverse visual tasks,” 2024. 2, 6, 8, 11

[51] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” 2025. 2

[52] Y. Shi, M. Dong, and C. Xu, “Harnessing vision foundation models for high-performance, training-free open vocabu lary segmentation,” in ICCV, 2025. 2, 6

[53] D. Zhang, F. Liu, and Q. Tang, “Corrclip: Reconstructing patch correlations in clip for open-vocabulary semantic segmentation,” in ICCV, 2025. 2

[54] S. Li, J. van de Weijer, T. Hu, F. S. Khan, Q. Hou, Y. Wang, and J. Yang, “Get what you want, not what you don’t: Im age content suppression for text-to-image diffusion models,” in ICLR, 2024. 3

[55] R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R. Urtasun, and A. Yuille, “The role of context for object detection and semantic segmentation in the wild,” in CVPR, 2014. 6

[56] G. Shin, W. Xie, and S. Albanie, “Reco: Retrieve and co segment for zero-shot transfer,” in NeurIPS, 2022. 6

[57] C. Zhou, C. C. Loy, and B. Dai, “Extract free dense labels from clip,” in ECCV, 2022. 6

[58] F. Wang, J. Mei, and A. Yuille, “Sclip: Rethinking selfattention for dense vision-language inference,” in ECCV, 2024. 6

[59] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Proxyclip: Proxy attention improves clip for open-vocabulary segmentation,” in ECCV, 2024. 6, 7

[60] C. Kim, D. Ju, W. Han, M.-H. Yang, and S. J. Hwang, “Distilling spectral graph for object-context aware openvocabulary semantic segmentation,” in CVPR, 2025. 6

[61] B. Blumenstiel, J. Jakubik, H. Kuhne, and M. V ¨ ossing,¨ “What a mess: Multi-domain evaluation of zero-shot semantic segmentation,” in NeurIPS Track on Datasets and Benchmarks, 2023. 8

[62] F. Rottensteiner, G. Sohn, J. Jung, M. Gerke, C. Baillard, S. Benitez, and U. Breitkopf, “The isprs benchmark on urban object classification and 3d building reconstruction,” Gottingen: Copernicus GmbH ¨ , 2012. 11

[63] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. HAZIZA, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[64] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[65] J. Wang, K. Markert, M. Everingham et al., “Learning models for object recognition from natural language descriptions.” in BMVC, no. 2009, 2009.

[66] K. Barnard, P. Duygulu, D. Forsyth, N. d. Freitas, D. M. Blei, and M. I. Jordan, “Matching words and pictures,” Journal ofmachine learning research, 2003.

[67] R. Zhang, Z. Zhang, M. Li, W.-Y. Ma, and H.-J. Zhang, “A probabilistic semantic model for image annotation and multimodal image retrieval,” in ICCV, 2005.

[68] P. Duygulu, K. Barnard, J. F. de Freitas, and D. A. Forsyth, “Object recognition as machine translation: Learning a lexicon for a fixed image vocabulary,” in ECCV, 2002.

[69] J. Li and J. Z. Wang, “Automatic linguistic indexing of pictures by a statistical modeling approach,” Transactions on pattern analysis and machine intelligence, 2003.

[70] N. Srivastava and R. R. Salakhutdinov, “Multimodal learning with deep boltzmann machines,” in NeurIPS, 2012.

[71] A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. Ranzato, and T. Mikolov, “Devise: A deep visualsemantic embedding model,” in NeurIPS, 2013.

[72] A. Karpathy and L. Fei-Fei, “Deep visual-semantic alignments for generating image descriptions,” in CVPR, 2015.

[73] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014.

[74] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” Transactions on Pattern Analysis and Machine Intelligence, 2017.

[75] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” Transactions on Pattern Analysis and Machine Intelligence, 2017.

[76] M. Wysoczanska, M. Ramamonjisoa, T. Trzci´ nski, and´ O. Simeoni, “Clip-diy: Clip dense inference yields open-´ vocabulary semantic segmentation for-free,” in WACV, 2024.

[77] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in CVPR, 2022.

[78] Y. Xian, S. Choudhury, Y. He, B. Schiele, and Z. Akata, “Semantic projection network for zero- and few-label semantic segmentation,” in CVPR, 2019.

[79] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl, “Language-driven semantic segmentation,” in ICLR, 2022.

[80] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev, “Reproducible scaling laws for contrastive language-image learning,” in CVPR, 2023.

[81] H. Luo, J. Bao, Y. Wu, X. He, and T. Li, “Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation,” in ICML, 2023.

[82] Z. T. Zheng Ding, Jieke Wang, “Open-vocabulary universal image segmentation with maskclip,” in ICML, 2023.

[83] J. Ding, N. Xue, G.-S. Xia, and D. Dai, “Decoupling zero shot semantic segmentation,” in CVPR, 2022.

[84] D. Kang and M. Cho, “In defense of lazy visual ground ing for open-vocabulary semantic segmentation,” in ECCV, 2024.

[85] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Clearclip: Decomposing clip representations for dense vision-language inference,” in ECCV, 2024.

[86] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda, and D. Marculescu, “Open-vocabulary semantic segmentation with mask-adapted clip,” in CVPR, 2023.

[87] S. Jiao, Y. Wei, Y. Wang, Y. Zhao, and H. Shi, “Learn ing mask-aware clip representations for zero-shot segmen tation,” in NeurIPS, 2023.

[88] M. Xu, Z. Zhang, F. Wei, H. Hu, and X. Bai, “Side adapter network for open-vocabulary semantic segmentation,” in CVPR, 2023.

[89] J. Tian, L. Aggarwal, A. Colaco, Z. Kira, and M. Gonzalez Franco, “Diffuse attend and segment: Unsupervised zeroshot segmentation using stable diffusion,” in CVPR, 2024.

[90] C. Ma, Y. Yang, C. Ju, F. Zhang, J. Liu, Y. Wang, Y. Zhang, and Y. Wang, “Diffusionseg: Adapting diffusion towards unsupervised object discovery,” 2023.

[91] M. A. Aydın, E. M. Cırpar, E. Abdinli, G. Unal, and Y. H. Sahin, “Itaclip: Boosting training-free semantic segmentation with image, text, and architectural enhancements,” in CVPRW, 2025.

[92] R. Burgert, K. Ranasinghe, X. Li, and M. S. Ryoo, “Peeka boo: Text to image diffusion models are zero-shot segmentors,” 2022.

[93] W. Bousselham, F. Petersen, V. Ferrari, and H. Kuehne, “Grounding everything: Emerging localization properties in vision-language transformers,” in CVPR, 2023.

[94] R. Sun, H. Mai, N. Luo, T. Zhang, Z. Xiong, and F. Wu, “Structure-decoupled adaptive part alignment network for

domain adaptive mitochondria segmentation,” in MICCAI, 2023.

[95] R. Sun, N. Luo, Y. Pan, H. Mai, T. Zhang, Z. Xiong, and F. Wu, “Appearance prompt vision transformer for connectome reconstruction.” in IJCAI, 2023.

[96] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” in NeurIPS, 2014.

[97] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in ICLR, 2014.

[98] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[99] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.

[100] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015.

[101] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in ICML, 2021.

[102] W. Wu, Y. Zhao, H. Chen, Y. Gu, R. Zhao, Y. He, H. Zhou, M. Z. Shou, and C. Shen, “Datasetdm: Synthesizing data with perception annotations using diffusion models,” in NeurIPS, 2023.

[103] Q. Nguyen, T. Vu, A. Tran, and K. Nguyen, “Dataset diffusion: Diffusion-based synthetic dataset generation for pixel-level semantic segmentation,” in NeurIPS, 2023.

[104] H. Caesar, J. Uijlings, and V. Ferrari, “Coco-stuff: Thing and stuff classes in context,” in CVPR, 2018.

[105] M.-Q. Le, T. V. Nguyen, T.-N. Le, T.-T. Do, M. N. Do, and M.-T. Tran, “Maskdiff: Modeling mask distribution with diffusion probabilistic model for few-shot instance segmentation,” in AAAI, 2024.

[106] J. Xu, S. Liu, A. Vahdat, W. Byeon, X. Wang, and S. De Mello, “Open-vocabulary panoptic segmentation with text-to-image diffusion models,” in CVPR, 2023.

[107] Z. Li, Q. Zhou, X. Zhang, Y. Zhang, Y. Wang, and W. Xie, “Open-vocabulary object segmentation with diffusion models,” in ICCV, 2023.

[108] L. Sun, J. Cao, J. Xie, F. S. Khan, and Y. Pang, “iseg: An iterative refinement-based framework for training-free segmentation,” 2024.

[109] J. Yang, M. Gao, Z. Li, S. Gao, F. Wang, and F. Zheng, “Track anything: Segment anything meets videos,” 2023.

[110] Z. Wang, Z. Sha, Z. Ding, Y. Wang, and Z. Tu, “Tokencompose: Text-to-image diffusion with token-level supervision,” in CVPR, 2024.

[111] P. Marcos-Manchon, R. Alcover-Couso, J. C. SanMiguel,´ and J. M. Mart´ınez, “Open-vocabulary attention maps with token optimization for semantic segmentation in diffusion models,” in CVPR, 2024.

[112] R. Adams and L. Bischof, “Seeded region growing,” Transactions on Pattern Analysis and Machine Intelligence, 1994.

[113] N. Ikonomatakis, K. Plataniotis, M. Zervakis, and A. Venetsanopoulos, “Region growing and region merging image segmentation,” in ICDSP, 1997.

[114] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014.

[115] S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang, and P. H. S. Torr, “Conditional ran dom fields as recurrent neural networks,” in ICCV, 2015.

[116] Z. Cai and N. Vasconcelos, “Cascade r-cnn: Delving into high quality object detection,” in CVPR, 2018.

[117] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in ECCV, 2020.

[118] B. Cheng, A. Schwing, and A. Kirillov, “Per-pixel classification is not all you need for semantic segmentation,” in NeurIPS, 2021.

[119] S. Sun, W. Wang, A. Howard, Q. Yu, P. Torr, and L.-C. Chen, “Remax: Relaxing for better training on efficient panoptic segmentation,” in NeurIPS, 2023.

[120] H. Wang, Y. Zhu, H. Adam, A. Yuille, and L.-C. Chen, “Max-deeplab: End-to-end panoptic segmentation with mask transformers,” in CVPR, 2021.

[121] Q. Yu, H. Wang, S. Qiao, M. Collins, Y. Zhu, H. Adam, A. Yuille, and L.-C. Chen, “k-means mask transformer,” in ECCV, 2022.

[122] R. Li, K. Li, Y.-C. Kuo, M. Shu, X. Qi, X. Shen, and J. Jia, “Referring image segmentation via recurrent refinement networks,” in CVPR, 2018.

[123] D. Lin, G. Chen, D. Cohen-Or, P.-A. Heng, and H. Huang, “Cascaded feature network for semantic segmentation of rgb-d images,” in ICCV, 2017.

[124] G. Lin, A. Milan, C. Shen, and I. Reid, “Refinenet: Multipath refinement networks for high-resolution semantic segmentation,” in CVPR, 2017.

[125] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in CVPR, 2017.

[126] F. Yu, D. Wang, E. Shelhamer, and T. Darrell, “Deep layer aggregation,” in CVPR, 2018.

[127] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask r-´ cnn,” in ICCV, 2017.

[128] J. Tian, L. Aggarwal, A. Colaco, Z. Kira, and M. Gonzalez-Franco, “Diffuse attend and segment: Unsupervised zeroshot segmentation using stable diffusion,” in CVPR, 2024.

[129] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detec tion,” in ECCV, 2024.

[130] H. Che and V.-T. Nguyen, “Fa-seg: A fast and accu rate diffusion-based method for open-vocabulary segmen tation,” Neurocomputing, 2026.

[131] S. Mahajan, T. Rahman, K. M. Yi, and L. Sigal, “Prompt ing hard or hardly prompting: Prompt inversion for text-toimage diffusion models,” in CVPR, 2024.

[132] Y. Hao, Z. Chi, L. Dong, and F. Wei, “Optimizing prompts for text-to-image generation,” in NeurIPS, 2023.

[133] S.-Y. Yeh, Y. LI, S. Park, G. Oh, X. Wang, M. Song, Y. Yu, and S.-H. Lai, “TIPO: Text to image with text pre-sampling for prompt optimization,” in ICLR, 2026.