# Hierarchical Curriculum Learning for Surgical Video Generation: Focusing on Bilateral Axillo-Breast Approach Robotic Thyroidectomy Datasets

Youngbin Kong<sup>1,2</sup>, Seungho Lee<sup>5</sup>, Jae Kyung Shin<sup>5</sup>, Jong-hyuk Ahn<sup>6</sup>, Hyoun-Joong Kong<sup>2,3,4,⋆</sup>, and Kyu Eun Lee<sup>5,7,⋆</sup>

<sup>1</sup> Interdisciplinary Program in Medical Informatics, Seoul National University College of Medicine, Seoul, Republic of Korea

2 Department of Transdisciplinary Medicine, Seoul National University Hospital, Seoul, Republic of Korea

3 Institute of Convergence Medicine with Innovative Technology, Seoul National University Hospital, Seoul, Republic of Korea

4 Department of Medicine, Seoul National University College of Medicine, Seoul, Republic of Korea

Department of Surgery, Seoul National University Hospital, Seoul, Republic of Korea

<sup>6</sup> YONSEI WITH-YOU Surgery Clinic, Seoul, Republic of Korea <sup>7</sup> Seoul National University College of Medicine, Seoul, Republic of Korea

⋆ Corresponding authors

Abstract. High-fidelity surgical video generation is increasingly important for surgical education, simulation, and downstream AI training, as conventional simulation pipelines are limited by cost, realism, and scenario diversity. Adapting generic text-to-video difusion models to surgical procedures remains dificult, however, due to complex anatomical structures, intricate tool–tissue interactions, the inherently hierarchical nature of surgical workflows, and the scarcity of curated training data outside a few well-studied procedures. We propose a surgical video generation framework that combines hierarchical procedural annotation with curriculum transfer learning. Surgical procedures are decomposed into three levels (phase, task, and action) and a pretrained text-to-video diffusion model is progressively adapted from coarse to fine-grained surgical context, supported by a dynamic sampling strategy that handles variable-resolution and variable-length surgical clips. We instantiate the framework on Bilateral Axillo-Breast Approach robotic thyroidectomy, a procedure for which no curated benchmark currently exists. Relative to the preceding task-level model, the final action-level model reduces FVD by 20.1% and FID by 15.7%, and a matched-budget ablation indicates that these gains arise from the curriculum design rather than from additional training. A user study with six clinicians rates visual realism highly while highlighting remaining gaps in anatomical accuracy and clinical reliability. The proposed framework ofers a step toward AI-driven surgical simulation in procedures where curated data are scarce.

Keywords: Surgical video generation · Hierarchical curriculum transfer learning · Robotic thyroidectomy

## 1 Introduction

Surgical simulation and computer-assisted training systems have become integral components of modern surgical education, allowing trainees to acquire technical skills in a risk-free, standardized, and objectively assessable environment, and decoupling the resident learning curve from patient risk [3, 30]. The field has progressed from low-fidelity box trainers and cadaveric or animal models [37] to virtual reality (VR) systems that enable repeatable practice without spatiotemporal constraints, with growing evidence of its efectiveness in surgical training [26,33,35]. More recently, real surgical videos have emerged as a richer simulation source [6], motivated by evidence that video-based curricula improve resident preparation and intraoperative participation [19], and computer-assisted intervention approaches further combine deep models with surgical video to enable 3D reconstruction and immersive simulation [25,42,46]. Yet these approaches remain limited by the size and diversity of public surgical video datasets, which makes large-scale, scenario-diverse simulation dificult to extend across surgical disciplines.

Generative artificial intelligence has recently enabled high-fidelity synthesis across text, image, and video modalities [2, 14], with difusion-based video models in particular emerging as the dominant paradigm due to their training stability, robustness to mode collapse, and high sample quality [9, 14]. Modern text-to-video systems such as Sora [2] and Open-Sora [52] achieve high visual realism and temporal coherence on general-domain content, raising the prospect of AI-generated surgical videos as scalable training material, augmentation for downstream surgical AI, and a substrate for surgical planning. However, applying these generic video generation models directly to the surgical domain remains challenging. Surgical scenes contain complex anatomical structures, subtle tissue deformations, and intricate tool–tissue interactions, and surgical procedures inherently follow a hierarchical structure progressing from operational phases through intermediate tasks down to specific instrument actions. Generic textto-video models often fail to capture this nuanced hierarchy, leading to temporal inconsistencies, physically implausible tool motion, and anatomical inaccuracies. Moreover, existing surgical video generation work has concentrated on procedures for which curated public datasets such as CholecT50 [31] or Kvasir-Capsule [40] already exist, leaving procedures with distinct anatomical access or instrumentation largely unaddressed.

We focus on Bilateral Axillo-Breast Approach (BABA) robotic thyroidectomy as a representative example of such procedures. Thyroidectomy is among the most common endocrine surgeries worldwide, and BABA, a remote-access technique developed at Seoul National University Hospital [7] and later adapted to robotic surgery, provides an anatomically symmetric midline view that closely resembles the open surgical perspective [20]. BABA combines an abundance of high-quality intraoperative video with the absence of a curated public dataset, leaving it unexplored in recent surgical video generation research. This combination of rich raw footage and missing curated benchmarks makes BABA an ideal testbed for data-eficient adaptation of large pretrained video models to a new procedural domain.

Recent advances in surgical workflow analysis suggest a path forward. Surgical procedures are inherently hierarchical, ranging from the entire operation down to phases, tasks, and actions, and recent works exploit this structure for surgical video understanding [49, 50]. In parallel, curriculum learning, inspired by human learning, orders training data by dificulty and has been explored in medical image analysis [24, 39]. However, while these advances enable richer surgical understanding, they have been applied almost exclusively to surgical analysis, with very limited exploration of how hierarchical procedural structure can be reflected in surgical video generation: the gap addressed by this paper.

To this end, we introduce a surgical video generation framework based on hierarchical annotation and curriculum transfer learning, with BABA robotic thyroidectomy as the target procedure. We construct a Phase–Task– Action three-level annotation of clinical BABA videos, and on top of the Open-Sora v1.2 architecture [52], which combines a T5-XXL text encoder [34], a 3D variational autoencoder (VAE), and a Spatial-Temporal Difusion Transformer (STDiT), we fine-tune only STDiT through three progressive stages aligned with the annotation hierarchy. A dynamic bucket-based sampling strategy further enables eficient training on variable-resolution and variable-length surgical clips.

To our knowledge, this is the first work to incorporate explicit hierarchical procedural structure into surgical video generation rather than recognition. Our contributions are summarized as follows:

1. Hierarchical curriculum transfer learning. We propose a three-stage strategy that progressively adapts a pretrained video difusion backbone from coarse Phase-level to fine-grained Action-level supervision, mirroring the surgical workflow hierarchy while keeping the text encoder and VAE frozen. To align data dificulty with each stage, we repurpose Open-Sora’s dynamic bucket sampling [52] as a stage-conditioned curriculum control rather than treating it as a new mechanism.

2. Phase–Task–Action annotation for generation. We define and apply a three-level temporal annotation schema for BABA robotic thyroidectomy, providing structured text supervision tailored to generative, not discriminative, training.

3. Data-eficient adaptation to a benchmark-free procedure. We instantiate the framework on BABA, for which no curated public dataset exists, and validate it through quantitative metrics, an ablation that isolates the curriculum efect from additional training, and a clinical user study with six clinicians.

## 2 Related Work

## 2.1 General-Domain Video Generation

Video generation has progressed rapidly with the rise of difusion models [15], after early eforts dominated by GAN-based approaches [29]. Video Difusion Models (VDM) [15] first applied difusion to video generation by extending the 2D U-Net used in image difusion to a 3D U-Net, enabling unconditional and conditional generation while improving inter-frame consistency. Make-A-Video [38] introduced a strategy that augments a pretrained text-to-image model with additional spatial and temporal layers, enabling video synthesis without paired text–video data. Imagen Video [13] adopted a cascaded difusion pipeline that first generates low-resolution videos and then performs multi-stage spatial and temporal super-resolution to reach high-definition output.

More recent models emphasize temporal consistency and high-resolution generation. ART-V [45] introduces masked difusion to enable autoregressive frame generation while maintaining temporal coherence over long sequences. Lumiere [1] uses a Space-Time U-Net (STUNet) to generate all frames jointly, replacing the keyframe-then-interpolate paradigm with a holistic low-resolution generation followed by super-resolution. OpenAI’s Sora [2] produces up to 60-second highresolution (1920 × 1080) videos with complex multi-agent scenes, while Open-Sora [52] provides an open-source counterpart based on a Spatial-Temporal Diffusion Transformer (STDiT) and a 3D VAE, supporting up to 16-second videos at up to 720p with variable aspect ratios.

While these models achieve high realism on general-domain content, they do not encode the conceptual structure or domain-specific data required for highly structured procedural video, such as surgery, and thus cannot be applied directly to surgical video generation.

## 2.2 Surgical Video Generation in the Medical Domain

Surgical video generation poses unique challenges beyond those of general-domain video [48], requiring high accuracy and realism for use in medical education and simulation [28]. Early work applied neural style transfer [10], which combines content-feature reconstruction with texture synthesis, to enhance the realism of simulation-rendered images. SurReal [27] transferred the style of real surgical videos onto simulation renderings, preserving structural correctness while greatly improving visual realism. Real-time style transfer algorithms have also been proposed to improve the realism of laparoscopic surgical simulators while maintaining temporal smoothness [36], applying the appearance of real surgical images to silicone-based phantoms.

A complementary line of work focuses on 3D mesh stylization and texture synthesis. Paget et al. [32] proposed a coherent texturing framework that generates tileable variable textures from intraoperative images and maps them onto 3D mesh geometry. More recently, MeshBrush [12] performs neural mesh stylization that preserves temporal consistency for endoscopic video by learning vertex-wise textures on meshes derived from patient imaging.

Beyond texturing and stylization, an emerging direction generates surgical videos directly. Endora [21] is among the first surgical video generation frameworks targeting endoscopic simulation, combining a latent difusion model with a spatial-temporal video transformer and a DINO [4] feature prior that enhances inter-frame consistency. SurGen [6] is a text-guided difusion model for surgical video synthesis built on CogVideoX [47], generating 480p clips of 49 frames. SurgSora [5] extends this direction with a more controllable design that uses a single input frame and motion cues, comprising a Dual Semantic Injector, a Decoupled Flow Mapper, and a Trajectory Controller, and decoupling RGB-D features from optical flow. VISAGE [48] synthesizes laparoscopic surgical videos via action scene graphs whose nodes represent instruments and organs and whose edges represent the performed actions, capturing the spatial and relational dynamics of surgical activity through difusion models. MedSora [43] integrates an attention-Mamba video difusion backbone [11], an optical-flowbased representation alignment module that enhances pixel-level inter-frame attention, and a frequency-compensated video VAE designed to reduce information loss for medical features. In ophthalmic surgery, Ophora [22] demonstrates that natural-language-guided ophthalmic surgical videos can be generated using a large dataset of more than 160K video–instruction pairs and a progressive video instruction tuning scheme that transfers spatiotemporal knowledge from general-domain pretraining.

These advances have, however, concentrated on surgical domains for which curated datasets such as CholecT50 [31] and Kvasir-Capsule [40] already exist, leading to strong domain dependency and limited generalization. Procedures with distinct anatomical access and instrumentation, such as BABA robotic thyroidectomy, cannot directly benefit from these methods.

## 2.3 Hierarchical Annotation and Curriculum Learning

Surgical procedures are inherently hierarchical, ranging from the entire operation down to phases, tasks, and actions. Recent works exploit this structure for surgical workflow analysis. HecVL [49] and PeskaVLP [50] pair short clips with atomic action descriptions, longer segments with phase summaries, and full videos with overall summaries, enabling models to capture both short-range and long-range dependencies and yielding strong representations for downstream tasks such as zero-shot phase recognition, with PeskaVLP further using LLMs to refine and augment these hierarchical texts.

In parallel, curriculum learning has been explored in medical image analysis through progressive easy-to-hard schedules [39] and anti-curriculum selection of informative samples [24]. While these advances enable richer surgical understanding, they operate in a discriminative setting in which hierarchy guides representation learning for recognition rather than constraining a generative process. How hierarchical procedural structure should instead shape surgical video generation, where the model must synthesize, not classify, phase-, task-, and action-level content, remains largely unexplored.

## 3 Methodology

This section details the proposed framework for surgical video generation. We first describe the construction of the hierarchically annotated clinical dataset, then introduce the network architecture, and finally present the dynamic bucketbased sampling and three-stage curriculum transfer learning strategy.

## 3.1 Hierarchically Annotated Surgical Video Dataset

Data Collection A total of 43 BABA robotic thyroidectomy videos were collected from Seoul National University Hospital. All operations were performed using the da Vinci Xi system, and each video covers the full procedure from start to end as a continuous recording at a resolution of 1920 × 1080. This study was approved by the Institutional Review Board (IRB) of Seoul National University Hospital (IRB No. H-2207-162-1344). All patient identifiers were removed, and the data were stored and used in compliance with the institutional ethical guidelines.

Hierarchical Annotation To convert the structured surgical workflow into a form suitable for training, we adopt a Phase–Task–Action three-level Hierarchical Surgical Annotation Schema (Fig. 1). Phases are macro-level operational steps that modularize high-level surgical functions, tasks are intermediate clinical sub-goals reflecting surgeon judgment, instrument manipulation, and anatomical recognition, and actions are atomic micro-level units describing tool–motion– target interactions.

![](images/6a61358b64fd5c6593eb9efb83cfcdf438dc7e847842dfe255819cede77fb6ee.jpg)  
Fig. 1. Schematic and example of the hierarchical annotation system for surgical video. BABA robotic thyroidectomy is divided into six Phases (MID, LAT, INF, BER, SUP, END), each further decomposed into Tasks and Actions.

The Phase hierarchy is defined based on the six surgical modules of BABA robotic thyroidectomy proposed by Kim et al. [18]: midline incision to isthmectomy (MID), lateral dissection (LAT), preservation of inferior parathyroid glands (INF), recurrent laryngeal nerve and Berry ligament dissection (BER), superior pole dissection (SUP), and specimen removal and closure (END). The Task hierarchy is derived through cognitive task analysis of intermediate procedural goals within each phase. The Action hierarchy follows the action triplet representation ⟨instrument, verb, target⟩ proposed by Nwoye et al. [31].

Annotations were performed by clinicians from the breast and endocrine surgery division of Seoul National University Hospital, who recorded start/end timestamps and textual captions at each level. As an exploratory methodological study, Task and Action annotations focused on the most generalizable End Phase (specimen removal and closure).

Preprocessing and Dataset Construction Each video is segmented into clips along the Phase–Task–Action timestamps, then further split into fixedlength sequences of 120 frames at 24 FPS to balance training stability and surgical context. CSV-format metadata is generated for each sequence, mapping it to its caption and recording resolution, frame length, and aspect ratio for downstream sampling. Among the 43 cases, 4 were held out as a patient-level test set and the remaining 39 were used for training. At the Phase, Task, and Action levels, 20, 16, and 7 cases were annotated, respectively; after holding out the 4 test cases, 20, 12, and 7 cases remained for training. The per-level clip counts and training budgets are reported in Table 1.

## 3.2 Network Architecture

We adopt the Open-Sora v1.2 architecture [52], an open-source Sora-like video generation model [2], pretrained on more than 20M training samples with approximately 35,000 NVIDIA H100 GPU hours [16]. The model supports up to 16-second clips at resolutions from 144p to 720p, and contains ∼1.1B parameters. The architecture comprises three modules (Fig. 2).

![](images/66f87735d93a28ade29ca45cdf2ba024d35e96f30b65bcb993396dc0a496b54c.jpg)  
Fig. 2. Architecture of the surgical video generation model. The pipeline integrates a T5-XXL text encoder, the Open-Sora 3D VAE, and the Spatial-Temporal Difusion Transformer (STDiT3).

Text Encoder. We use T5-XXL [34] to encode surgical text prompts $( \mathrm { e . g . }$ “robotic instrument grasping tissue during lateral dissection”) into rich contextual embeddings that condition STDiT.

3D VAE. The Open-Sora v1.2 VAE compresses input videos into a lowdimensional spatiotemporal latent representation and decodes them back. It first compresses each frame spatially by a factor of 8 with a 2D VAE and then compresses the temporal dimension by a factor of 4 with a temporal VAE, producing more compact latents than a per-frame 2D VAE alone.

STDiT3. The core difusion model operates in the latent space and learns conditional generation between text embeddings and video latents. To remain computationally tractable, STDiT decouples spatial and temporal attention: each block first applies spatial attention within frames, then temporal attention across frames, achieving eficient yet expressive spatiotemporal modeling. We adopt the STDiT3-XL/2 configuration: 28 blocks, hidden size 1152, 16 attention heads, patch size (1, 2, 2), and approximately 1.12B parameters.

Preliminaries: Latent Rectified-Flow Generation Following Open-Sora [52], the frozen 3D VAE encodes an input clip x into a spatiotemporal latent $z _ { 0 } =$ $\mathcal E ( x )$ , and the frozen T5-XXL encoder maps the prompt to a conditioning embedding c of maximum length 300. STDiT is trained with a rectified-flow objective, in which the noisy latent $z _ { \tau } = ( 1 - \tau ) z _ { 0 } + \tau \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , I )$ linearly interpolates between data and noise, and STDiT regresses the velocity $z _ { 0 } - \epsilon _  \mathrm  ~ $

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { \tau , z _ { 0 } , \epsilon } \big \| v _ { \theta } ( z _ { \tau } , \tau , c ) - ( z _ { 0 } - \epsilon ) \big \| _ { 2 } ^ { 2 } .\tag{1}
$$

The timestep $\tau$ is drawn from a logit-normal distribution and shifted toward higher noise levels for higher-resolution and longer clips, following the resolutionand length-aware timestep sampling of Open-Sora [16, 52].

## 3.3 Dynamic Bucket-Based Sampling

Surgical clips vary substantially in resolution, aspect ratio, and duration. Rather than padding or cropping them to a fixed shape, we adopt the bucket-based sampling natively supported by Open-Sora [52] (Fig. 3) and repurpose its perbucket selection probabilities as a stage-conditioned curriculum control. A bucket is defined as a triplet $B = ( R , F , A )$ , where R is the resolution in pixel area, F is the number of frames, and A is the aspect ratio, and each bucket is assigned a keep probability keep ${ } _ { - } p r o b _ { B }$ and a batch size. For a video $V _ { i } .$ , the set of valid buckets is

$$
B _ { \mathrm { v a l i d } } = \{ B \mid R _ { V _ { i } } \geq 0 . 8 R _ { B } \ \wedge \ F _ { V _ { i } } > F _ { B } \} .\tag{2}
$$

Valid buckets are visited in descending order of resolution, and at each resolution R the bucket with the largest valid frame count is selected,

$$
\operatorname { S e l e c t } ( V _ { i } \mid R ) = \arg \operatorname* { m a x } _ { \substack { B \in B _ { \mathrm { v a l i d } } , R _ { B } = R } } F _ { B } ,\tag{3}
$$

and accepted if u < keep\_ ${ } _ { p r o b }$ with u $\sim \mathcal { U } [ 0 , 1 ]$ . If it is rejected, the next lower resolution is tried, and a clip for which no bucket is accepted is skipped. The accepted clip is then resized and temporally subsampled to the bucket’s shape, so that every batch contains clips of identical shape.

![](images/503a6495a9b1621890c6910486e27f6787ce31f29f12b374ac784a5f254e393c.jpg)  
√ accept if u < keep-probB = ps (B), u ∼ U [0, 1]; X try the next lower resolution  
Fig. 3. Dynamic bucket sampling over the resolution–frame lattice. A clip V<sub>i</sub> is routed to the highest-resolution valid bucket and accepted with probability keep\_prob $_ { B } \dot  \}$ otherwise the next lower resolution is tried. The accepted clip is downsampled into a shape-aligned batch. Per-stage keep\_prob (Fig. 4) acts as the curriculum control. Only a subset of buckets is shown. Based on the bucket mechanism of [52].

This dynamic strategy aligns batch tensor shapes to minimize padding, controls memory and compute by lowering keep\_prob for high-resolution/long-frame buckets, and accelerates convergence by oversampling low-resolution/short-frame buckets early on.

The per-stage keep\_prob serves as the curriculum control (Fig. 4): Stage 1 downweights high-resolution buckets to acquire global context, while Stages 2 and 3 restore them to sharpen fine detail.

## 3.4 Curriculum Transfer Learning Framework

Rather than fine-tuning a generic video difusion model directly on fast, complex Action-level clips, we adopt a three-stage curriculum (Fig. 5) that mirrors the surgical hierarchy and proceeds from coarse to fine-grained representations. Throughout all stages, the T5-XXL text encoder and 3D VAE are frozen, and only STDiT is updated, focusing learning capacity on text-conditioned spatiotemporal generation while reusing the pretrained vision–language priors.

Stage 1: Phase fine-tuning. The pretrained STDiT is first fine-tuned on Phase-level data, which contains long, slow clips capturing global surgical context. We set high keep\_prob on low-resolution buckets and low keep\_prob on high-resolution buckets, and run the longest schedule. This stage aligns the model with the general visual distribution of the surgical cavity, lighting, and tissue appearance.

![](images/fb026763d5db9e3923486247a97865a492097514a0e2acbda6262f520ab10453.jpg)  
Fig. 4. Average bucket keep probability by resolution across the three curriculum stages. Stage 1 (Phase) heavily downsamples high-resolution buckets to emphasize global context, while Stages 2 (Task) and 3 (Action) retain high-resolution clips to capture fine visual detail.

![](images/f2f1e31f222b41ae5e15bc8e60a0c3396615529afc712235983f154c87b0b003.jpg)  
Fig. 5. Three-stage curriculum transfer learning pipeline. The text encoder and 3D VAE are frozen, and STDiT is fine-tuned progressively from Phase to Task to Action levels.

Stage 2: Task fine-tuning. Starting from Stage 1 weights, the model is finetuned on Task-level data to learn intermediate procedural goals and coordinated instrument motions. Compared with Stage 1, the sampling strategy increases keep\_prob on higher-resolution buckets to capture sharper visual structure.

Stage 3: Action fine-tuning. Finally, the model is fine-tuned on the Actionlevel dataset, which contains highly specific, fast tool–tissue interactions. Because the global and intermediate-level context has already been internalized in the previous stages, even a small amount of well-curated data and a short schedule are suficient to refine fine details and physical plausibility. The bucket sampling configuration follows Stage 2.

Formally, let $p _ { s } ( B )$ be the keep probability of bucket B at curriculum stage $s \in \ \{ \mathrm { P } $ hase, Task, Action} (Fig. 4). Together with the stage-specific clips, $p _ { s }$ turns the bucket selection of Eqs. (2)–(3) into a stage-specific training distribution $\mathcal { D } _ { s }$ . Starting from the pretrained backbone $\theta _ { 0 } .$ , each stage minimizes the rectified-flow loss of Eq. (1) over its own distribution:

$$
\theta _ { s } = \arg \operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { ( x , c ) \sim \mathcal { D } _ { s } } \big [ \mathcal { L } ( \theta ) \big ] , \quad \mathrm { i n i t i a l i z e d ~ f r o m } \ \theta _ { s - 1 } .\tag{4}
$$

Parameters are optimized with AdamW $( \epsilon = 1 0 ^ { - 1 5 }$ , learning rate $1 0 ^ { - 4 }$ , gradient clip 1.0, 1,000 warmup steps) under bf16 precision with gradient checkpointing and ZeRO Stage 2, and an exponential moving average of the weights with decay 0.99 is maintained. Stage- and bucket-level data composition and time budgets are summarized in Table 1, and implementation details in Table 2.

Table 1. Dataset split and per-stage curriculum composition. “Annotated”: cases labeled at each level; “Train”: cases used after the 4-case test split. Clips, steps, and time refer to training.
<table><tr><td></td><td>Level Annotated Train Clips Steps</td><td></td><td>Time (h)</td></tr><tr><td>Phase</td><td>20</td><td>20 16,000 55,000</td><td>1,375</td></tr><tr><td>Task</td><td>16</td><td>12 3,20015,000</td><td>375</td></tr><tr><td>Action</td><td>7</td><td>7 160</td><td>28.8</td></tr><tr><td colspan="4">Corpus: 43 cases (39 train / 4 test, patient-level). &quot;Annotated&quot;: cases labeled at each level; &quot;Train&quot;: cases used after the test split.</td></tr></table>

## 4 Experiments and Results

## 4.1 Experimental Setup

We evaluated the proposed framework on a patient-disjoint test set comprising 4 of the 43 BABA robotic thyroidectomy videos described in Section 3. From each test case, 10 representative task sequences were selected and clipped to 5-second segments at 24 FPS (120 frames), yielding 40 ground-truth clips. For each case, the model generated 10 End-Phase clips at matched specifications. Implementation details and the training/evaluation environment are summarized in Table 2.

Table 2. Implementation details and training/evaluation environment. Optimization hyperparameters (learning rate, ϵ, warmup, EMA decay) are given in the text.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>Open-Sora v1.2 (STDiT3-XL/2)</td></tr><tr><td>Scheduler</td><td>rectified flow (rflow)</td></tr><tr><td>Batch size</td><td>1 per GPU</td></tr><tr><td>Precision</td><td>bf16</td></tr><tr><td>Distributed</td><td>DeepSpeed ZeRO Stage 2</td></tr><tr><td colspan="2">Gradient checkpointing yes</td></tr><tr><td>GPU</td><td>8× NVIDIA A100 80 GB</td></tr><tr><td>CPU</td><td>AMD EPYC 7402X @ 2.8 GHz</td></tr><tr><td>Framework</td><td>PyTorch 2.1.0, Open-Sora 1.2.0</td></tr><tr><td>OS / CUDA</td><td>Ubuntu 20.04.6, CUDA 11.8</td></tr></table>

For quantitative evaluation, we report Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [44], Learned Perceptual Image Patch Similarity (LPIPS) [51], FloLPIPS [8], Fréchet Video Distance (FVD) [41], and Fréchet Inception Distance (FID). Surgical video generation is inherently non-deterministic: from a given prompt the model synthesizes a plausible realization rather than a pixel-aligned copy of any particular ground-truth clip. Reference-based pixel and patch metrics (PSNR, SSIM, LPIPS, FloLPIPS) therefore penalize legitimate appearance variation and, while reported for completeness, serve only as secondary frame-level indicators. We accordingly treat the distribution-level metrics FID and FVD, together with the clinical user study, as our primary measures, as they assess whether generated frames and videos match the real surgical distribution rather than a single reference clip.

For qualitative clinical evaluation, six clinicians (one resident and five fellows) rated the generated videos using a 5-point Likert scale [23] on visual similarity, anatomical accuracy, procedural consistency of motion, and overall quality. Inspired by the evaluation dimensions of VBench [17], the questionnaire comprised seven items spanning general video quality (visual similarity, motion smoothness, imaging quality, aesthetic quality) and clinical relevance (human anatomy, clinical applicability, clinical reliability).

## 4.2 Quantitative Evaluation of Curriculum Learning

Table 3 reports per-stage performance, focusing on the Stage 2 (Task) → Stage 3 (Action) transition. On our primary distribution-level metrics, Stage 3 reduces FID by 15.7% and FVD by 20.1% relative to the final Stage 2 model, indicating improved spatial and spatiotemporal realism. The secondary frame-level metrics move in the same direction but by smaller margins (PSNR +5.6%, SSIM +1.2%, LPIPS −1.5%); FloLPIPS alone worsens slightly (+1.0%), which we attribute to its sensitivity to frame-to-frame flow diferences under non-deterministic generation rather than to degraded temporal quality, a reading corroborated by the large FVD improvement on the same clips.

Table 3. Per-stage quantitative evaluation of the curriculum transfer learning framework.
<table><tr><td>Model Stage</td><td colspan="6">Steps SSIM ↑ PSNR ↑ LPIPS ↓ FloLPIPS ↓ FVD ↓ FID ↓</td></tr><tr><td rowspan="4">Stage 2 (Task)</td><td>3k</td><td>0.307</td><td>9.77</td><td>0.717</td><td>0.726</td><td>4242.5 244.9</td></tr><tr><td>7k</td><td>0.285</td><td>9.64</td><td>0.744</td><td>0.739</td><td>4652.7 249.1</td></tr><tr><td>11k</td><td>0.264</td><td>10.18</td><td>0.680</td><td>0.709</td><td>5097.4 213.5</td></tr><tr><td>15k</td><td>0.285</td><td>10.19</td><td>0.719</td><td>0.722</td><td>4105.4 241.2</td></tr><tr><td rowspan="3">Stage 3 (Action) 0.8k</td><td>0.4k</td><td>0.299</td><td>10.41</td><td>0.708</td><td>0.734</td><td>4665.6 233.6</td></tr><tr><td></td><td>0.282</td><td>10.53</td><td>0.677</td><td>0.739</td><td>4608.1 198.5</td></tr><tr><td>1.2k</td><td>0.288</td><td>10.77</td><td>0.708</td><td>0.729</td><td>3278.4 203.4</td></tr></table>

Within Stage 2, FVD plateaued near 4100 and FID remained above 213, indicating that Task-only training had limited headroom. The frame-level scores fluctuated non-monotonically: SSIM dipped to 0.264 at 11k before recovering to 0.285 at 15k. We attribute this to the model generating sharper, more varied textures as training progresses; because such textures cannot match the groundtruth clip pixel by pixel, they can lower reference-based scores even when visual quality does not decrease. In Stage 3, the primary metrics improved overall: after an initial rise (4665.6 at 0.4k), FVD decreased to 3278.4 at 1.2k (a 29.7% reduction from the first Stage 3 checkpoint), and FID reached its minimum of 198.5 at 0.8k. These results indicate that Action-level fine-tuning improves finegrained detail and temporal coherence beyond the Stage 2 plateau.

## 4.3 Ablation Study

To verify that Stage 3’s gains stem from the curriculum rather than from additional training, we compared the full model (Stage 1 → 2 → 3) against a matched-budget baseline that spends the final 1.2k steps on Stage 2 data (Stage 1 → 2 → 2). Under an identical compute budget (Table 4), the curriculum model improves FVD by 18.0% alongside concurrent PSNR (+9.5%) and SSIM (+9.7%) gains, while LPIPS, FloLPIPS, and FID remain essentially unchanged. The simultaneous improvement in distribution-level (FVD) and pixel/structural (PSNR, SSIM) fidelity suggests that the improvement is not specific to a single metric. Notably, Stage 3 attains these gains from only 160 Action-level clips, 20 times fewer than the 3,200 Task-level clips used by the baseline, suggesting that, in this setting, curriculum ordering contributes more than additional Task-level data.

Table 4. Ablation study on the efect of curriculum transfer learning. Both models share Stage 1 (55k) and Stage 2 (15k) and difer only in the final 1.2k steps.
<table><tr><td>Schedule</td><td>SSIM ↑ PSNR ↑ LPIPS ↓ FloLPIPS ↓ FVD ↓ FID ↓</td><td></td><td></td><td></td><td></td></tr><tr><td>Stage 1→2→2 (matched budget)</td><td>0.263</td><td>9.83</td><td>0.696</td><td>0.721</td><td>3997.2 200.9</td></tr><tr><td>Stage 1 →2→3 (Ours)</td><td>0.288</td><td>10.77</td><td>0.708</td><td>0.729</td><td>3278.4 203.4</td></tr></table>

## 4.4 Qualitative Evaluation

Visual inspection complements the quantitative findings. In early Stage 1 (15k steps), the model often produced physically impossible scenes such as three robotic arms appearing simultaneously, indicating that the basic constraints of the surgical environment had not yet been learned. By 35k steps, the arm count and motion errors were largely corrected, but textures of gauze and surrounding tissue remained blurred. By 55k steps, both physical constraints and fine textures (e.g., gauze fabric pattern, tissue surface) were faithfully reproduced (Fig. 6), suggesting that Phase-level training captures global surgical context.

![](images/c2669051e2ef3c9b3ef9c184702b6aa6273f73ea9a1a7f188d3253893566d391.jpg)  
Fig. 6. Qualitative results across Stage 1 training steps (15k, 35k, 55k). Physical inconsistencies (e.g., extra robotic arms) progressively disappear and tissue textures become realistic.

We also compared Stage 2 and Stage 3 models on specific Task prompts (Fig. 7). For suturing, Stage 2 captured rough shapes of needle and thread but rendered them ambiguously across frames, whereas Stage 3 produced sharper, more consistent thread contours and a clearer grasp on needle and thread. For irrigation, Stage 2 portrayed liquid sloshing and suction but with unnatural dynamics, while Stage 3 reproduced the inflow of liquid and surface specular reflections more realistically. These qualitative diferences echo the quantitative gains in FVD and FID for Stage 3.

![](images/75f1d759a0d9fd97b32de2045252790880fcb594acb941edc8dfbe5912d97b4c.jpg)  
Fig. 7. Qualitative comparison between Stage 2 and Stage 3 on suturing and irrigation tasks. Stage 3 reproduces fine instruments (thread, needle) and fluid dynamics more faithfully.

## 4.5 Clinical User Study

Six clinicians rated the generated videos on the seven items described above. The mean scores (Fig. 8) are highest for Imaging Quality (4.33), followed by Visual Similarity (4.00) and Aesthetic Quality (4.00), suggesting that the model reproduces the overall color, texture, and sharpness of real surgical video. Motion Smoothness (3.67) and Clinical Applicability (3.67) also received favorable ratings.

In contrast, Clinical Reliability (3.33) and Human Anatomy (3.50) received lower scores with larger inter-rater variance, suggesting that while overall shapes and motions are convincing, precise anatomical structures and subtle tissue interactions are not yet at clinical-grade fidelity. These findings indicate that further refinement of soft-tissue physics and anatomical priors will be needed for safe deployment in clinical training scenarios.

![](images/2dab148478092837ecb925d8e2dfdbfbdc22b2b7459301b45a660b97911f2f2e.jpg)  
Fig. 8. User study results from six clinicians. The model performs best on imagingrelated criteria but is more limited on anatomical accuracy and clinical reliability.

## 5 Discussion

In addition to quantitative metrics, we incorporate a structured user study with practicing clinicians, enabling assessment of clinical reliability and anatomical accuracy beyond raw video quality. More importantly, our results show that a hierarchically structured curriculum extends high-fidelity generation to BABA robotic thyroidectomy without any curated benchmark, indicating that the approach is not confined to data-rich procedures targeted by prior work. By integrating hierarchical annotation with curriculum transfer learning, the framework treats surgery as a structured, hierarchical procedural domain rather than a flat sequence of frames, providing a path toward domain-specialized generative models with clinical relevance.

The quantitative results support this view: Action-level fine-tuning breaks the Stage 2 plateau, and the ablation (Table 4) shows the gain stems from the curriculum rather than extra training, as the matched-budget Stage 1 → 2 → 2 baseline lags despite using 20× more clips.

Qualitatively, the progression of Stage 1 from physically implausible scenes (15k steps) to faithful textures (55k steps) suggests that Phase-level fine-tuning provides a foundation upon which Stage 2 and Stage 3 build with comparatively few additional examples. The Stage 2-vs-Stage 3 comparisons on suturing and irrigation suggest that Action-level training particularly benefits fine instruments and fluid dynamics, where temporal coherence is critical.

The user study makes the remaining limitations concrete: Imaging Quality, Visual Similarity, and Aesthetic Quality all exceed 4.0/5.0, while Clinical Reliability and Human Anatomy remain below this threshold with substantial interrater variance. This indicates that improving anatomical fidelity and soft-tissue dynamics, rather than overall image quality, is the main remaining challenge.

## 5.1 Limitations and Future Work

This study has several limitations. First, the dataset originates from a single institution (Seoul National University Hospital) with 43 BABA robotic thyroidectomy cases, so generalization to other institutions, surgical techniques, and patient populations is not directly demonstrated. Second, we focus on the End Phase for Task and Action annotation; modeling the entire procedure with consistent transitions between phases remains future work. Third, the absence of a public benchmark for surgical video generation prevents direct quantitative comparison with prior models; our evaluation is therefore relative to internal baselines. Fourth, while we employ standard quality metrics and expert review, we do not yet measure how generated videos afect actual training outcomes. Future work will integrate the system into existing training pipelines and quantify educational impact on residents and fellows.

To address these gaps, we plan to (i) construct an internal benchmark grounded in prior evaluation protocols to enable more objective comparison, (ii) expand the dataset to multiple institutions, surgical techniques, and patient cohorts, (iii) extend the framework to generate full-procedure surgical videos with consistent inter-phase transitions, and (iv) condition the generative model on additional patient-specific signals such as preoperative CT/MRI or surgical kinematics, enabling personalized rehearsal aligned with each patient’s anatomy.

## 6 Conclusion

We presented a framework for surgical video generation that combines a Phase– Task–Action hierarchical annotation system with a three-stage curriculum transfer learning strategy on top of the Open-Sora STDiT backbone, applied to BABA robotic thyroidectomy. Quantitative metrics, qualitative analysis, an ablation study, and a clinician user study collectively show that the framework adapts a large pretrained model to domain-specific surgical content and improves finegrained visual fidelity while preserving global procedural context. Although validated on a single procedure, the approach may extend to other procedures and ofers a basis for AI-driven surgical simulation, education content generation, and downstream surgical AI systems requiring high-fidelity synthetic video.

Acknowledgments. The authors thank the clinicians of the breast and endocrine surgery division at Seoul National University Hospital for their guidance on annotation and clinical evaluation. This research was supported by the Naver Digital Bio Research Fund (Grant No. 3720250030).

## References

1. Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Liu, G., Raj, A., et al.: Lumiere: A space-time difusion model for video generation. In: SIGGRAPH Asia 2024 Conference Papers. pp. 94:1–94:11. ACM (2024). https://doi.org/10.1145/3680528.3687614

2. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators. OpenAI technical report, https://openai.com /index/video-generation-models-as-world-simulators/ (2024)

3. Cardoso, S.A., et al.: Exploring the role of simulation training in improving surgical skills among residents: a narrative review. Cureus 15(9), e44654 (2023). https: //doi.org/10.7759/cureus.44654

4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9650–9660 (2021). https://doi.org/10.1109/ICCV48922.2021.00951

5. Chen, T., Yang, S., Wang, J., Bai, L., Ren, H., Zhou, L.: SurgSora: Object-aware difusion model for controllable surgical video generation. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. LNCS, vol. 15969, pp. 521–531. Springer (2025). https://doi.org/10.1007/978-3-032-05127-1\_50

6. Cho, J., Schmidgall, S., Zakka, C., Mathur, M., Kaur, D., Shad, R., Hiesinger, W.: SurGen: Text-guided difusion model for surgical video generation. arXiv preprint arXiv:2408.14028 (2024)

7. Choe, J.H., Kim, S.W., Chung, K.W., Park, K.S., Han, W., Noh, D.Y., Oh, S.K., Youn, Y.K.: Endoscopic thyroidectomy using a new bilateral axillo-breast approach. World Journal of Surgery 31(3), 601–606 (2007). https://doi.org/10 .1007/s00268-006-0481-y

8. Danier, D., Zhang, F., Bull, D.: FloLPIPS: A bespoke video quality metric for frame interpolation. In: 2022 Picture Coding Symposium (PCS). pp. 283–287. IEEE (2022). https://doi.org/10.1109/PCS56426.2022.10018062

9. Dhariwal, P., Nichol, A.: Difusion models beat GANs on image synthesis. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 34, pp. 8780– 8794 (2021)

10. Gatys, L.A., Ecker, A.S., Bethge, M.: Image style transfer using convolutional neural networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2414–2423 (2016). https://doi.org/10.1109/ CVPR.2016.265

11. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. In: First Conference on Language Modeling (COLM) (2024)

12. Han, J.J., Acar, A., Kavoussi, N., Wu, J.Y.: MeshBrush: Painting the anatomical mesh with neural stylization for endoscopy. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. LNCS, vol. 15006, pp. 380–390. Springer (2024). https://doi.org/10.1007/978-3-031-72089-5\_36

13. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., Salimans, T.: Imagen Video: High definition video generation with difusion models. arXiv preprint arXiv:2210.02303 (2022)

14. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 33, pp. 6840–6851 (2020)

15. Ho, J., Salimans, T., Gritsenko, A.A., Chan, W., Norouzi, M., Fleet, D.J.: Video difusion models. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 35, pp. 8633–8646 (2022)

16. HPC-AI Tech: Open-Sora 1.2 report. https://github.com/hpcaitech/Open-Sor a/blob/main/docs/report\_03.md (2024)

17. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench:

Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21807–21818 (2024)

18. Kim, H., Yu, H.W., Ahn, J.H., Lee, T.S., Lee, K.E.: Defining competencies in robotic thyroidectomy: development of a model assessing an expert operator’s intraoperative performance skills and cognitive strategies. Gland Surgery 13(3), 340–350 (2024). https://doi.org/10.21037/gs-23-467

19. Kim, M.P., et al.: Video-based curriculum improves resident participation during robot-assisted surgery. Journal of Thoracic Disease 14(12), 4641–4649 (2022). ht tps://doi.org/10.21037/jtd-22-603

20. Lee, K.E., Choi, J.Y., Youn, Y.K.: Bilateral axillo-breast approach robotic thyroidectomy. Surgical Laparoscopy Endoscopy & Percutaneous Techniques 21(4), 230–236 (2011). https://doi.org/10.1097/SLE.0b013e31822d0455

21. Li, C., Liu, H., Liu, Y., Feng, B.Y., Li, W., Liu, X., Chen, Z., Shao, J., Yuan, Y.: Endora: Video generation models as endoscopy simulators. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. LNCS, vol. 15006, pp. 230–240. Springer (2024). https://doi.org/10.1007/978-3-031-72089-5\_22

22. Li, W., Hu, M., Wang, G., Liu, L., Zhou, K., Ning, J., Guo, X., Ge, Z., Gu, L., He, J.: Ophora: A large-scale data-driven text-guided ophthalmic surgical video generation model. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. LNCS, vol. 15968, pp. 425–435. Springer (2025). https://doi.org/10.1007/978-3-032-05114-1\_41

23. Likert, R.: A technique for the measurement of attitudes. Archives of Psychology 22(140), 1–55 (1932)

24. Liu, F., Tian, Y., Chen, Y., Liu, Y., Belagiannis, V., Carneiro, G.: ACPL: Anticurriculum pseudo-labelling for semi-supervised medical image classification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20697–20706 (2022)

25. Liu, Y., et al.: EndoGaussian: Real-time Gaussian splatting for dynamic endoscopic scene reconstruction. arXiv preprint arXiv:2401.12561 (2024)

26. Lucas, S.M., et al.: Training on a virtual reality laparoscopic simulator improves performance of an unfamiliar live laparoscopic procedure. The Journal of Urology 180(6), 2588–2591 (2008). https://doi.org/10.1016/j.juro.2008.08.041

27. Luengo, I., et al.: SurReal: Enhancing surgical simulation realism using style transfer. arXiv preprint arXiv:1811.02946 (2018)

28. McGuire, L.S., Kwasnicki, A., Ismail, R., Weiss, T., Charbel, F.T., Alaraj, A.: Future of visualization and simulation in neurosurgery. In: Alaraj, A. (ed.) Comprehensive Healthcare Simulation: Neurosurgery, pp. 265–282. Springer International Publishing, Cham (2018). https://doi.org/10.1007/978-3-319-75583-0\_20

29. Melnik, A., Ljubljanac, M., Lu, C., Yan, Q., Ren, W., Ritter, H.: Video difusion models: A survey. Transactions on Machine Learning Research (2024)

30. de Montbrun, S.L., MacRae, H.: Simulation in surgical education. Clinics in Colon and Rectal Surgery 25(3), 156–165 (2012). https://doi.org/10.1055/s-0032-1 322553

31. Nwoye, C.I., Yu, T., Gonzalez, C., Seeliger, B., Mascagni, P., Mutter, D., Marescaux, J., Padoy, N.: Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos. Medical Image Analysis 78, 102433 (2022). https://doi.org/10.1016/j.media.2022.102433

32. Paget, R., Harders, M., Székely, G.: A framework for coherent texturing in surgical simulators. In: 13th Pacific Conference on Computer Graphics and Applications (2005)

33. Peng, M.J., et al.: Virtual reality-based surgical planning simulator for tumorous resection in FreeForm modeling: an illustrative case of clinical teaching. Quantitative Imaging in Medicine and Surgery 14(2), 2060–2068 (2024). https: //doi.org/10.21037/qims-23-1151

34. Rafel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research 21(140), 1–67 (2020)

35. Rogister, F., Salmon, C., Ghuysen, A., Andrews, P.J., Bonnet, P., Camby, S.: Virtual reality surgical simulation as a tuition aid for understanding surgical temporal bone anatomy: trial on 15 ear, nose, and throat registrars. B-ENT 16(2), 103–108 (2020). https://doi.org/10.5152/B-ENT.2020.20017

36. Shao, M., Clark, J., Huson, D., Hardeberg, J.: Real-time style transfer for videos to enhance the realism of simulation of laparoscopic surgeries. In: 2022 10th European Workshop on Visual Information Processing (EUVIP). IEEE (2022). https://do i.org/10.1109/EUVIP53989.2022.9922706

37. Sharma, D., Agrawal, V., Bajaj, J., Agarwal, P.: Low-cost simulation systems for surgical training: a narrative review. Journal of Surgical Simulation 7(1), 33–52 (2020). https://doi.org/10.1102/2051-7726.2020.0005

38. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., Taigman, Y.: Make-A-Video: Text-tovideo generation without text-video data. In: International Conference on Learning Representations (ICLR) (2023)

39. Singh, P., Nampalle, K.B., Narayan, U.V., Raman, B.: See through the fog: Curriculum learning with progressive occlusion in medical imaging. arXiv preprint arXiv:2306.15574 (2023)

40. Smedsrud, P.H., et al.: Kvasir-Capsule, a video capsule endoscopy dataset. Scientific Data 8(1), 142 (2021). https://doi.org/10.1038/s41597-021-00920-z

41. Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: FVD: A new metric for video generation. In: ICLR 2019 Workshop on Deep Generative Models for Highly Structured Data (2019)

42. Wang, K., Yang, C., Zhao, K., Yang, X., Shen, W.: Realistic surgical simulation from monocular videos. arXiv preprint arXiv:2412.02359 (2024)

43. Wang, Z., et al.: Optical flow representation alignment Mamba difusion model for medical video generation. arXiv preprint arXiv:2411.01647 (2024)

44. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861

45. Weng, W., Feng, R., Wang, Y., Dai, Q., Wang, C., Yin, D., Zhao, Z., Qiu, K., Bao, J., Yuan, Y., Luo, C., Zhang, Y., Xiong, Z.: ART-V: Auto-regressive text-to-video generation with difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 7395–7405 (2024). https://doi.org/10.1109/CVPRW63382.2024.00735

46. Yang, S., Li, Q., Shen, D., Gong, B., Dou, Q., Jin, Y.: Deform3DGS: Flexible deformation for fast surgical scene reconstruction with Gaussian splatting. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. LNCS, vol. 15006, pp. 132–142. Springer (2024). https://doi.org/10.1007/978-3-031 -72089-5\_13

47. Yang, Z., et al.: CogVideoX: Text-to-video difusion models with an expert transformer. In: International Conference on Learning Representations (ICLR) (2025)

48. Yeganeh, Y., Lazuardi, R., Shamseddin, A., Dari, E., Thirani, Y., Navab, N., Farshad, A.: VISAGE: Video synthesis using action graphs for surgery. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024 Workshops. LNCS, vol. 15274, pp. 146–156. Springer (2025). https://doi.org/10.100 7/978-3-031-77610-6\_14

49. Yuan, K., Srivastav, V., Navab, N., Padoy, N.: HecVL: Hierarchical video-language pretraining for zero-shot surgical phase recognition. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. LNCS, vol. 15006, pp. 306– 316. Springer (2024). https://doi.org/10.1007/978-3-031-72089-5\_29

50. Yuan, K., Srivastav, V., Navab, N., Padoy, N.: Procedure-aware surgical videolanguage pretraining with hierarchical knowledge augmentation. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 37, pp. 122952–122983 (2024)

51. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 586–595 (2018)

52. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-Sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404 (2024)