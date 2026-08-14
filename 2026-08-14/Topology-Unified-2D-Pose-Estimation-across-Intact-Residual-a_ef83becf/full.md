# Topology-Unified 2D Pose Estimation across Intact, Residual and Prosthetic Limbs

Tianye Qi , Tengyue Zhang , Jiaying Ying , Tianqing Zhu , and Xin Yu

![](images/92532bdbae061932adf5cfeb3415bafb3b7e0fe5acc41d567a9035c3bb153afb.jpg)

![](images/33d12e0937a2d76559b07d66593b2e23d7037cced767a3735a36b40331360aa8.jpg)

![](images/f0dae4628891ba149d6a2061acaf6ad2063580b7b820c2405cd19af84bd02c38.jpg)  
Fig. 1. Resolving topological ambiguity, on an upper limb (top) and a lower limb (bottom). Unlike COCO [1] that hallucinates biological joints on mechanical structures, and LDPose [2] that truncates topology at the residual limb, ProPose explicitly disentangles joint semantics to accurately reconstruct the complete kinematic chain across intact, residual, and prosthetic limbs. In the top row the elbow and wrist of the prosthetic arm do not exist and carry no coordinate: COCO has no way to record that and must place them on the device, LDPose stops at the residual limb, and only ProPose ends the chain where the limb ends and resumes it at the prosthetic fingertip.

Abstract—Driven by the availability of large-scale datasets, Human Pose Estimation (HPE) plays a critical role in numerous downstream tasks. However, mainstream benchmarks exhibit severe representation bias, predominantly featuring able-bodied individuals. While a few pioneering datasets have attempted to address limb differences, their annotation protocols fail to generalize, struggling to represent specialized mechanical structures like running blades or unprosthetized residual limbs. To bridge this gap, we introduce ProPose, a large-scale benchmark featuring a novel annotation protocol that unifies the topological representation of biological limbs, diverse prostheses, and physical absences within a single framework. Because real-world prosthetic images are inherently scarce and exhibit extreme longtail distributions, we design a Real-to-Synthetic data expansion pipeline to explicitly synthesize and expand the underrepresented cases. However, simply training existing models on this enriched dataset often leads to suboptimal solutions, as they estimate each keypoint independently and might hallucinate non-existent joints on mechanical structures. To resolve this, we propose ProLoss, a structure-aware objective that enforces keypoint dependencies within a single limb to prevent unrealistic limb predictions. Extensive experiments demonstrate that our approach improves the classification accuracy of long-tail prosthetic joints by 2% to 6% without compromising spatial coordinate localization performance. This work sets a foundation for inclusive pose estimation, unlocking new possibilities for understanding the interactions between human bodies and assistive devices.

Index Terms—Pose estimation with Prostheses Body Structure-Aware Representation Prosthesis-aware Pose Estimation Benchmark

## I. INTRODUCTION

UMAN Pose Estimation (HPE) stands as a pivotal task in computer vision, serving as the cornerstone for a myriad of downstream applications. Its precise localization of anatomical keypoints is indispensable for action recognition [3], [4], physical literacy analysis [5], 3D human mesh recovery [6]–[8], and clinical gait analysis [9], [10]. Driven by the rapid advancement of deep learning, recent years have witnessed the emergence of robust state-of-the-art frameworks. Representative methods include transformer-based ViTPose [11], and real-time RTMPose [12] and YOLO-Pose series [13]–[15]. The impressive performance of these models is largely attributed to the availability of large-scale, high-quality benchmarks. Beyond foundational datasets, like MS COCO [1] and MPII [16], the field has advanced through benchmarks focusing on complex occlusion (CrowdPose [17], OCHuman [18]) and fine-grained whole-body perception (COCO-WholeBody [19]).

Despite their scale, these mainstream benchmarks suffer from a significant representation bias, predominantly featuring able-bodied individuals. A few pioneering works have attempted to bridge this gap, though they adopt different annotation protocols. LDPose [2] focuses on annotating residual limbs, which are highly valuable for studying the amputation site, but it leaves prosthetic devices unannotated. ProGait [20] includes the prosthesis by mapping mechanical components to standard biological keypoints (e.g., knee, ankle). While this strategy aligns well with anthropomorphic prostheses that mimic human geometry, biological mapping can introduce semantic ambiguities when applied to specialized prostheses with non-biological geometries, such as running blades, or to exposed residual limbs without prosthetic attachment. Consequently, it is highly desirable for a unified pose annotation protocol that can provide topological consistency across both biological limbs and diverse mechanical structures.

To bridge this critical gap, we introduce ProPose <sup>1</sup>, a large-scale benchmark specifically designed to resolve the representation bias and topological mismatch in prosthesisaware pose estimation (see Fig. 1). Advancing beyond the annotation scheme of LDPose [2], which only appends eight residual points while neglecting distal interactions, we propose an Omni-Pose Protocol that integrates two key innovations: First, to unify the topological representation across diverse prosthetic designs, we explicitly label distal extremities (fingertips, heels, and toes). This ensures that regardless of the specific mechanical configuration, a coherent kinematic chain and effective contact points are consistently captured within a single framework. Second, to endow the model with the ability to distinguish between biological limbs and mechanical devices, and to handle cases where certain joints are physically non-existent (e.g., specialized prostheses or residual limbs without prosthetic devices), we introduce a novel semantic attribute (Keypoint Type) for each joint, categorizing them as Biological, Prosthetic, or Absent.

After annotating the semantic labels and prosthetic keypoints on LDPose dataset [2], we found that the number of prosthetic keypoints is very scarce, especially for elbow and knee joints. Because the number of images of individuals with prostheses are much smaller and LDPose already includes many of the available sources, our initial web crawling efforts yielded only 1,807 new images. To overcome this critical data bottleneck, we designed a Real-to-Synthetic data expansion pipeline that leverages Nano Banana [21] for data augmentation. We reserve 437 high-quality real-world images (covering 499 unseen subjects) from this crawled subset and use them for testing. The remaining 1,370 images are processed through the pipeline for image synthesis, followed by manual review to remove generated samples containing visible artifacts. As a result, we expand our training set with 9,558 images (containing 10,345 instances).

The data expansion significantly mitigates the long-tail distribution observed in prior benchmarks. For instance, prosthetic elbows were severely underrepresented in LDPose, accounting for only 0.19% of all annotated elbow keypoints; following our augmentation pipeline, this proportion rises to 5.81%, a 31-fold increase. The proportion of prosthetic wrists has grown 8.6-fold as well, i.e., from 1.00% to 8.63%. Therefore, our ProPose dataset ensures that models can learn robust features across a wide spectrum of prosthetic devices, rather than overfitting to biological limbs.

Furthermore, directly employing standard training objectives will lead to anatomically contradictory predictions, such as hallucinating a biological wrist from a residual limb, because they treat keypoints as independent spatial entities. In practice, human-prosthetic anatomy is governed by strict structural and semantic constraints, in which one joint determines the physical validity of the others on the same limb. To eliminate unreasonable limb configurations, we propose ProLoss, a structure-aware limb-dependence function. By explicitly modeling the semantic dependencies across all joints within a single limb, ProLoss penalizes invalid type combinations, forcing a network to output logically consistent kinematic chains. Extensive experiments demonstrate that the models trained with ProLoss on our ProPose dataset achieve superior results across biological, residual and prosthetic limbs. This work effectively establishes topological consistency across diverse limb types, providing a unified framework for inclusive human pose estimation. In summary, our contributions are threefold:

• Unified Pose Representation Protocol: We propose a unified 2D human pose representation protocol, namely Omni-Pose, for diverse limb configurations. Particularly, Omni-Pose involves a semantic attribute (Keypoint Type) for each joint to determine limb types.

• Large-Scale Benchmark (ProPose): We construct Pro-Pose dataset that not only covers normal and residual limbs but also includes prosthetic limbs. Moreover, to overcome data scarcity and long-tail distributions of prosthesis images, we design a Real-to-Synthetic data expansion pipeline to enrich the quantity and diversity of prosthetic devices.

• Structure-Aware Objective (ProLoss): We propose ProLoss, a structure- and semantics-aware limbdependent loss that explicitly models intra-limb semantic dependencies. It penalizes unreasonable limb configurations, forcing the network to learn physically plausible kinematic chains.

## II. RELATED WORK

## A. Human Pose Estimation Benchmarks

a) Full-limb Benchmarks.: The rapid advancement of 2D Human Pose Estimation (HPE) has been fundamentally driven by large-scale annotated benchmarks. Foundational datasets such as MS COCO [1] and MPII [16] establish the standard for multi-person pose estimation, fueling the development of diverse data-driven architectures. To address more challenging scenarios characterized by severe occlusion and crowded scenes, CrowdPose [17] and OCHuman [18] are introduced. Furthermore, benchmarks have been extended to the temporal domain with PoseTrack21 [22] for videobased tracking, and to 3D space with Human3.6M [23] and FreeMan [24]. However, these benchmarks and models exhibit a systemic bias: they implicitly assume a canonical ablebodied topology (e.g., the standard 17-keypoint format). As a result, they often hallucinate missing limbs or fail to recognize prosthetic devices.

b) Specialized Benchmarks.: To bridge the representation gap, specialized benchmarks have recently emerged. LDPose [2] pioneers this direction by focusing on limb deficiency. It provides 28K images and extends the standard COCO topology by appending 8 keypoints to localize residual limb endpoints. Although LDPose acknowledges the presence of prosthetics, it excludes them from annotations, limiting the model’s ability to analyze the interaction between humans and assistive devices. Inclusive VidPose [25] extends the LDPose to the video domain with 313 clips totaling 327k frames. It leverages temporal context to effectively disambiguate occluded limbs from truly absent ones, but it still lacks keypoint annotations for prosthetic devices. ProGait [20] introduces a video benchmark (412 clips) that explicitly labels prosthetic joints. It maps prostheses to biological joint labels, but is constrained by its small scale and limited diversity of prosthetic devices. Thus, models trained on ProGait fail to accommodate specialized devices $( e . g .$ , running blades) or distinguish prostheses from intact limbs. These limitations underscore the necessity for a large-scale benchmark that features diverse prosthetic configurations and explicitly encodes the semantic attribute of each joint.

## B. Human Pose Estimation (HPE) Methods

HPE algorithms can often be grouped into two categories: top-down and bottom-up methods [26]. Top-down approaches first detect bounding boxes and then estimate keypoints. Representative large models include HRNet [27] and the transformer-based ViTPose [11]. However, the computational cost of top-down architectures scales linearly with the number of instances. To address this, real-time top-down models like RTMPose [12] and distillation-based DWPose [28] are proposed to balance efficiency and accuracy via optimized backbones and lightweight heads.

Bottom-up or single-stage methods prioritize inference speed by eliminating the separate cropping step. OpenPose [29] pioneers the bottom-up paradigm using Part Affinity Fields (PAFs) to efficiently associate body parts for multiperson estimation. Following this direction, models like HigherHRNet [30] address scale variation issues using highresolution feature pyramids, while DEKR [31] utilizes adaptive convolutions for direct keypoint regression. More recently, YOLO-Pose [13] embeds pose estimation heads directly into a single-stage object detector, offering end-to-end inference. Since these models are also trained on images of able-bodied individuals, they struggle recognizing prosthetic devices or hallucinate missing limbs.

## C. Data Augmentation and Synthesis

Acquiring large-scale, annotated datasets for limb-deficient individuals is inherently challenging due to the underrepresented nature of the population. To address this low-resource data challenge, synthetic data generation has been widely adopted [32]. Traditional approaches rely on 3D rendering engines to create diverse poses and environments. For instance, SURREAL [33] generates synthetic humans from 3D scans. More relevant to our domain, WheelPose [34] utilizes a synthetic engine to generate data for wheelchair users, effectively addressing severe occlusions caused by assistive equipment. However, these engine-based methods often suffer from a significant domain gap between synthetic renders and photorealistic imagery, limiting their generalization to realworld scenarios.

Recent advances in MLLMs have increasingly focused on human-centric perception and understanding, including identity association, body pose, action, and interaction reasoning [35], [36]. These advances have facilitated instructionbased human image editing, enabling models to follow textual commands while preserving identity and plausibly modifying pose, limb configuration, and appearance. Unlike 3D rendering pipelines that generate images from scratch, state-of-theart models such as Qwen-Image-Edit [37] and Gemini [21] demonstrate the ability to modify existing real-world images through precise natural language prompts. While text-to-image synthesis has also advanced considerably through latent diffusion models [38] and ControlNet [39], these methods often struggle to preserve the photorealism and subject identity when performing complex structural transformations, such as altering a posture from standing to running. In contrast, MLLMs excel at preserving these critical attributes and generating diverse, high-fidelity training samples directly from real-world data.

## III. CONSTRUCTION OF THE PROPOSED PROPOSE DATASET

We introduce the procedure of data collection, annotation and post-processing in detail, and then present key statistics of ProPose dataset. As indicated in Table I, we also compare ProPose with prior datasets.

## A. Omni-Pose Representation Protocol

Standard pose estimation protocols (e.g., COCO-17) are designed for able-bodied individuals. LDPose [2] appended 8 residual keypoints for residual limbs but neglects the existence of prosthetic devices. Although ProGait aligns prosthetic devices with the standard MS COCO topology, this biological mapping fails to capture scenarios in which specific joints are structurally absent in certain prostheses. For instance, a carbon-fiber running prosthesis lacks an anatomical ankle. Consequently, current annotation protocols fall short in their ability to unify all limb types, including intact, residual, and prosthetic limbs.

In this work, we generalize the LDPose protocol by explicitly adding 6 distal keypoints: Heel, Toe, and Fingertip (Left/Right), as well as assign a Semantic Type to each keypoint to faithfully reflect the physical conditions observed in the images (see Fig. 2). Specifically, (1) Limb Joints (including standard limb joints and the newly added distal points) can appear in all three types: c ∈ {Biological, Prosthetic, Absent}; (2) Residual Keypoints, representing the physical stump, only exist as $c \in \{ B i o l o g i c a l , A b s e n t \}$ ; and (3) Other Core

TABLE I  
COMPARISON OF HUMAN POSE DATASETS. WHILE RECENT EFFORTS HAVE INTRODUCED DATASETS FOR SPECIFIC DISABILITIES, NONE OF THE EXISTING BENCHMARKS UNIFIES INTACT (INT), RESIDUAL (RES), AND PROSTHETIC (PROS) LIMBS.
<table><tr><td>Dataset</td><td>Total Images/Videos</td><td>Total Instances</td><td>Avg. Poses per Image</td><td>Limb Types</td><td>Resolution</td><td>Annotation Source</td><td>Env. Complexity</td></tr><tr><td>MPII [16]</td><td>25k</td><td>40k</td><td>~2</td><td>Int</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>Crowd-sourced</td><td>Moderate</td></tr><tr><td>MSCOCÓ [1]</td><td>200k</td><td>250k</td><td>~2</td><td>Int</td><td> $6 4 0 \times 4 8 0 \mathrm { ~ t o ~ } 1 2 8 0 \times 7 2 0$ </td><td>Crowd-sourced</td><td>High</td></tr><tr><td>CrowdPose [17]</td><td>20k</td><td>80k</td><td>~4</td><td>Int</td><td> $1 0 8 0 \times 7 2 0$ </td><td>Crowd-sourced</td><td>High</td></tr><tr><td>OCHuman [18]</td><td>4.7k</td><td>8k</td><td>~2</td><td>Int</td><td> $1 2 8 0 \times 7 2 0$ </td><td>Crowd-sourced</td><td>High</td></tr><tr><td>PoseTrack21 [22]</td><td>66k</td><td>177k</td><td>~2.5</td><td>Int</td><td> $1 2 8 0 \times 7 2 0$ </td><td>Crowd-sourced</td><td>High</td></tr><tr><td>FreeMan [24]</td><td>11.3m</td><td>11.3m</td><td>~1</td><td>Int</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>Crowd-sourced</td><td>High</td></tr><tr><td>Human3.6M [23]</td><td>3.6m</td><td>3.6m</td><td>~1</td><td>Int</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>Experts</td><td>Low</td></tr><tr><td>LDPose [2]</td><td>28k</td><td>72.7k</td><td>~3</td><td>Int, Res</td><td> $2 0 0 \times 1 4 3 ~ \mathrm { t o } ~ 8 1 0 0 \times 4 7 5 2$ </td><td>Medical Experts</td><td>High</td></tr><tr><td>ProGait [20]</td><td>412</td><td>412</td><td>~1</td><td>Int, Pros</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>Medical Experts</td><td>Moderate</td></tr><tr><td>WheelPose [34]</td><td>2.5k</td><td>2.6k</td><td>~1</td><td>Int</td><td> $4 8 0 \times 3 6 0 ~ \mathrm { t o } ~ 1 2 8 0 \times 1 2 8 0$ </td><td>Generated</td><td>Moderate</td></tr><tr><td>ProPose(Ours)</td><td>36.4k</td><td>63.0k</td><td>~2</td><td>Int, Res, Pros</td><td> $2 0 0 \times 1 4 3 ~ \mathrm { t o } ~ 8 6 4 0 \times 8 2 5 6$ </td><td>Medical Experts</td><td>High</td></tr></table>

![](images/ca18cdbbdcb3ef932dfdfb438ef22ec54483cc241f645bc00685f721da179d9d.jpg)  
Fig. 2. Omni-Pose protocol. Omni-Pose unifies intact, residual, prosthetic and absent limb configurations within a single topology; each corner magnifies the limb of the photograph it points to. Filled markers are annotations taken from the dataset. Absent joints carry no coordinate by definition, so they are drawn as open dashed rings at their inferred position for illustration only.

Joints (e.g., head and torso) are universally Biological. This categorization captures prosthetic replacements and missing limb structures.

Visibility Protocol: Conventional visibility labels are designed to describe the observability of keypoint coordinates, typically distinguishing visible keypoints, occluded but inferable keypoints, and uncertain keypoints. However, these labels do not directly capture anatomical absence, where no valid coordinate exists but the absence of a joint can be determined. To address this limitation, we introduce an “Absent” semantic type under a deterministic visibility protocol. For example, when a residual limb is clearly visible and no prosthesis is present, the absence of subsequent joints is an observable anatomical state. We therefore assign these keypoints a visible visibility flag (v = 2) together with the “Absent” semantic type, indicating that the absence itself is visible and deterministic, even though no coordinate is annotated.

## B. Real-to-Synthetic Data Expansion Pipeline

Initial statistical analysis of the LDPose training dataset [2] reveals a severe imbalance: instances containing prosthetic elbows constituted approximately 0.2% of the total annotations. Our preliminary experiments show this scarcity leads to severe model bias after fine-tuning. However, due to the long-tail nature of prosthetic imagery, traditional web crawling alone cannot fully resolve this problem. Thus, we propose a real-to-synthetic data expansion pipeline, which comprises three phases: real-world collection, generative AI-driven data augmentation, and quality control. The pipeline can effectively address data scarcity while ensuring that the synthesized data remain realistic and useful for model training.

a) Phase 1: Data Collection.: We collect images from social media and search engines. After de-duplication and manual screening, we obtain 1,807 high-quality real-world images. We use 1,370 images as seeds for training data generation, and integrate the remaining 437 into the LDPose test set for evaluation.

b) Phase 2: Generative AI-driven Image-to-Image Synthesis.: Even with the addition of web-crawled images, the proportion of images with prosthesis is small. To enrich the training set, we employ an instruction-driven editing pipeline using Nano Banana. Here, we adopt a multi-stage strategy:

1) Contextual Outpainting: Many crawled images feature close-up shots with truncated limbs, lacking global context. We first use the model (i.e., Nano Banana) to outpaint these crops into half-body or full-body images for robust pose estimation.

2) Pose Editing and Balancing: We alter the subjects poses using detailed text prompts. To prevent the model from generating physically impossible structures, we explicitly specify the valid bending joints and the rigid, non-deformable components of the prosthetic devices. Additionally, we perform targeted oversampling, specifically generating samples containing rare prosthetic categories (e.g., prosthetic elbows and knees) to alleviate the long-tail distributions issue.

Before finalizing the image editor used in this phase, we conduct a model-selection study to determine which instruction-based editing model is suitable for prosthesisaware data expansion. For each candidate model, we manually review 100 generated outputs using three task-specific criteria: whether the human pose is clearly edited, whether the original prosthetic or residual-limb condition is preserved, and whether severe visual artifacts are present. Table II summarizes this quality-control comparison.

TABLE II  
QUALITY-CONTROL COMPARISON OF GENERATIVE MODELS FOR PROSTHESIS-AWARE DATA EXPANSION. FOR EACH MODEL, WE GENERATE 100 CANDIDATE IMAGES AND MANUALLY SCREEN THEM FOR IMAGE QUALITY. A CANDIDATE IS ACCEPTED ONLY WHEN THE POSE IS CLEARLY EDITED, THE PROSTHETIC OR RESIDUAL-LIMB CONDITION IS PRESERVED, AND NO SEVERE VISUAL ARTIFACT IS PRESENT.
<table><tr><td>Generative model</td><td>Generated</td><td>Accepted ↑</td><td>Acceptance rate</td><td>Pose edited ↑</td><td>Prosthesis preserved ↑</td><td>Artifact rate ↓</td></tr><tr><td>Gemini Flash Image / Nano Banana 2</td><td>100</td><td>80</td><td>80.0%</td><td>100.0%</td><td>96.0%</td><td>4.0%</td></tr><tr><td>Gemini 3 Pro Image / Nano Banana Pro</td><td>100</td><td>72</td><td>72.0%</td><td>100.0%</td><td>88.0%</td><td>12.0%</td></tr><tr><td>GPT Image 2</td><td>100</td><td>36</td><td>36.0%</td><td>36.0%</td><td>100.0%</td><td>8.0%</td></tr><tr><td>Seedream / Jimeng</td><td>100</td><td>16</td><td>16.0%</td><td>36.0%</td><td>56.0%</td><td>4.0%</td></tr><tr><td>Qwen-Image-Edit</td><td>100</td><td>12</td><td>12.0%</td><td>28.0%</td><td>28.0%</td><td>52.0%</td></tr><tr><td>FireRed-Image-Edit</td><td>100</td><td>8</td><td>8.0%</td><td>20.0%</td><td>20.0%</td><td>72.0%</td></tr></table>

![](images/03bc6788f94fe6a00d9b04a542ad9bb299f59eb2391935963c6a3d1b8757999b.jpg)

![](images/e5f8d297ac04d333a40aa22bac83dcd72d36ce228db6b60cc32be53e827cd8f5.jpg)

![](images/1e8dc57739e147f2fd6e0a44fbac24163537f7fb4e29757ecb6657327e2bc1c2.jpg)

![](images/d560f163f716ddce72e085b616030387cd9d7d3073ddd5ba46bf783aa6a5e28f.jpg)  
Fig. 3. Proposed generative AI-driven synthesis and dataset analysis. Statistics in (b)–(d) are computed over the training and test splits together, counting only annotated keypoints (visibility > 0). (a) Outpainting and pose editing for diverse joint types. (b) Prosthetic keypoints as a share of each joint’s annotations, split by the data source that contributed them; the multiplier in brackets is the growth of that share relative to the LDPose image subset alone. (c) Residual keypoints annotated as physically present, in the LDPose subset and in ProPose. (d) Composition of each row’s annotated keypoints by semantic type; each row is normalised independently, and the separated Residual row aggregates the eight residual keypoints, which admit only Bio and Abs.

The Gemini-based models achieve the strongest overall quality. Gemini Flash obtains the highest acceptance rate (80.0%), while Gemini 3 Pro also maintains a high acceptance rate (72.0%) with all reviewed samples showing clear pose edits. In contrast, the non-Gemini models have acceptance rates of 36.0% or lower. GPT Image 2 preserves the prosthetic device when the edit succeeds, but only 36.0% of its outputs show clear pose edits. Seedream, Qwen-Image-Edit, and FireRed-Image-Edit frequently fail to preserve prosthetic structures or introduce severe artifacts. We therefore choose the Gemini family for data extension, because it provides the best balance between controllable pose editing, prosthesis preservation, and low artifact rate.

c) Phase 3: Quality and Privacy Assurance.: To ensure data quality and adhere to ethical codes of conduct, we implement a rigorous filtering protocol:

• Cascade De-duplication: To prevent data leakage and ensure sample uniqueness, we apply a multi-stage filtering strategy. We first utilize MD5 hashing to automatically eliminate exact duplicates. Subsequently, we employ Perceptual Hashing (pHash) with a Hamming distance threshold of 5 to cluster structurally similar images. These candidate clusters are then manually reviewed to eliminate redundant near-duplicate images [40], [41].

Dual-Verification Protocol (Realism & Kinematics): Three annotators inspect each generated image, discarding those with visual artifacts or kinematic implausibility (e.g., impossible joint angles). This rigorous process removed approximately 10,000 images, ensuring the retained data strictly satisfies real-world biomechanical constraints.

• Annotation Quality Verification: All images are independently annotated by two biomechanics experts. When coordinate or semantic type annotations are inconsistent, a third expert reviews and resolves the annotations. Overall, 94.0% of samples reach direct agreement, 5.8% are resolved after third-expert review, and 0.2% are discarded due to unresolved annotation ambiguity.

![](images/efd596ed73419b8684111283ed7b8a446157e82f9b794956edce2e771a9a2d2d.jpg)  
Fig. 4. Overview of the Real-to-Synthetic data expansion pipeline. Phase 1 collects and de-duplicates web imagery and sets aside a strictly isolated real-world test set. Phase 2 uses instruction-driven editing to outpaint truncated crops into full-body images and to edit poses under kinematic constraints, with targeted oversampling of rare prosthetic categories. Phase 3 filters the generated images by cascade de-duplication, dual verification against realism and kinematics, and de-identification. Pruned LDPose images and the verified synthetic images together form the ProPose training set; the test set contains real photographs only.

• Data Pruning of LDPose images: We found that LD-Pose contains many able-bodied instances at very low resolution or with heavy occlusion. To prevent these samples from diluting the evaluation focus, we filter out low-resolution able-bodied samples, comprising around 20% of the original training and test sets. Together with the pruned LDPose dataset, our additionally crawled and generated images form our ProPose dataset.

• Copyright Compliance and De-identification: We ensure compliance with the copyright and usage rights of the original images. For images with unclear copyright or licensing status, we implement a de-identification protocol using the FaceFusion framework. Specifically, an identity-agnostic face swapping method [42] is employed to replace original faces with a synthetic non-existent identity. This process removes original biometric features while preserving pose information.

To further support reproducibility, we provide additional details of our Real-to-Synthetic data expansion pipeline. Following the model-selection results in Table II, we use the Gemini-based Nano Banana API, specifically gemini-3-pro-image-preview, for instruction-based image editing in the final data construction pipeline. During quality control, approximately 62% of the generated images are discarded due to visual artifacts, implausible prosthetic structures, or inconsistent body configurations, and 9,558 generated images are retained for the final training set. No generated image is placed in the test split, so evaluation is performed entirely on real photographs.

## C. Dataset Statistics of Our ProPose Dataset

The statistical distribution of ProPose demonstrates two critical advancements: mitigation of long-tail scarcity and finegrained semantic categorization.

a) Mitigation of Long-Tail Data Scarcity.: Our data curation reshapes the distribution of limb-deficient data. As illustrated in Fig. 3 (b), our data construction successfully boosts rare categories across training and test sets. Relative to the LDPose image subset annotated under the same protocol, the prosthetic share of elbow keypoints rises from 0.19% to 5.81% (a 31× increase) and that of wrists from 1.00% to 8.63% (8.6×), while the already better-represented lower-limb joints change by at most 1.5×. The expansion is therefore concentrated exactly where the tail was thinnest. Because the original test set likewise lacked sufficient samples to reliably evaluate these rare joints, we expand it as well: prosthetic elbow and wrist keypoints grow from 38 to 129 and from 135 to 457, respectively.

Furthermore, Fig. 3 (c) quantifies the absolute increase in residual annotations. Compared to the LDPose subset, ProPose adds at least 1,400 annotated residual keypoints at every amputation level, and 6,770 at the above-elbow level, which was previously the scarcest. This expansion offers richer signals for learning complex geometries and robust evaluation.

b) Distributions of Keypoint Semantics.: ProPose explicitly labels each keypoint as Biological, Prosthetic, or Absent. Despite our data expansion, Fig. 3 (d) reveals that a severe long-tail distribution persists across specific semantic states. For instance, prosthetic elbows and absent heels remain rare minorities compared to dominant biological joints. This semantic imbalance motivates the design of our learning objectives (Sec. IV-B).

## IV. METHODOLOGY

## A. Problem Formulation

We formulate Omni-Human Pose Estimation (Omni-HPE) as a joint learning problem of coordinate regression and semantic attribution. Given an input image $\bar { I } \ \in \ \mathbb { R } ^ { H \times W \times 3 }$ our objective is to predict a set K keypoints $\mathbf { \mathcal { P } } = \{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { K }$ their corresponding confidence $s _ { i }$ and semantics $c _ { i } { : }$

$$
\mathbf { p } _ { i } = ( \mathbf { x } _ { i } , s _ { i } , c _ { i } ) ,\tag{1}
$$

where $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { 2 }$ denotes the spatial coordinates, $s _ { i } ~ \in ~ [ 0 , 1 ]$ represents the prediction confidence, and $c _ { i } \in \mathcal { C }$ represents the semantic class and the label space $\mathcal { C } = \{ \mathtt { B i o } , \mathtt { P r o s } , \mathtt { A b s } \}$ corresponding to Biological, Prosthetic, and Absent states, respectively. Note that Bio includes both intact and residual limbs. They can be easily distinguished by their keypoint ordering.

![](images/8699dc320d29ae338aa2ddecd4cdb0a503ddb4f73a6e34b93fc36ac8e9a2cb28.jpg)  
Fig. 5. The Omni-Pose protocol applied across the ProPose dataset. The examples span the limb configurations the protocol has to describe: upper and lower limb, amputation above and below the joint, one side and both, and limbs with and without a prosthetic device. A keypoint is coloured by its semantic type, so a single skeleton can mix all three: Intact covers both healthy and residual biological joints, Prosthetic marks a joint realised by a mechanical device, and a joint labelled Absent carries no coordinate at all, which is why a kinematic chain simply ends where the limb does. The last two examples show the attribute is assigned per joint rather than per limb: the biological wrist is Absent while the fingertips of the attached prosthetic hand are Prosthetic.

For keypoints labeled as Abs, the spatial coordinate does not correspond to a physically existing anatomical or mechanical point. Therefore, coordinate supervision for these keypoints is masked out during training, while their semantic type labels are retained for classification.

## B. Proposed Learning Objective: ProLoss

Learning to estimate poses for individuals with limb differences presents two primary challenges. First, the severe class imbalance between biological and prosthetic or absent joints causes standard loss formulations to heavily bias the model toward the majority class. To address this distribution bias, we introduce a re-weighting strategy for semantic categorization. Second, standard objective functions assume an independent and identically distributed (i.i.d.) topology among keypoints. This assumption might lead to hallucinated keypoints. To jointly resolve these issues, we propose ProLoss, a structureaware objective function composed of three terms: Location Regression $( \mathcal { L } _ { r e g } )$ and Semantic Categorization $( \mathcal { L } _ { c l s } )$ , alongside a Bio-Contrastive Boundary Loss $( \mathcal { L } _ { b i o } )$

1) Distribution-Aligned Supervision:: Inspired by the strategies that address class imbalance, such as the Class-Balanced (CB) loss [43] and Focal Loss [44], we employ a decoupled, inverse-square-root weighting strategy to effectively promote the influence of rare instances.

a) Semantic Categorization Weighting.: For the semantic classification of each keypoint i, we apply a point-wise normalized weight to prevent the majority biological class from dominating the cross-entropy loss. The classification weight $\omega _ { i , c } ^ { c l s }$ for class c is inversely proportional to the square root of its frequency:

$$
\omega _ { i , c } ^ { c l s } = \frac { 1 / \sqrt { N _ { i , c } } } { \frac { 1 } { | \mathcal { C } _ { v a l i d } | } \sum _ { j \in \mathcal { C } _ { v a l i d } } ( 1 / \sqrt { N _ { i , j } } ) } ,\tag{2}
$$

where $N _ { i , c }$ is the number of training samples for class c at keypoint i, and $\mathcal { C } _ { v a l i d }$ represents the valid classes present for that specific joint.

With the semantic classification weight $\omega _ { i , c } ^ { c l s }$ , we formulate our combined objective. Since spatial coordinate regression is unaffected by the semantic long-tail distribution, we do not apply additional re-weighting to the regression branch. The localization loss $\mathcal { L } _ { \boldsymbol { r } \boldsymbol { e } \boldsymbol { g } }$ and the distribution-aligned semantic loss $\mathcal { L } _ { c l s }$ are formally defined as:

$$
\mathcal { L } _ { r e g } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbb { 1 } ( v _ { i } > 0 ~ \& ~ c _ { i } ^ { * } \neq \mathrm { a b s e n t } ) \cdot \ell ( \mathbf { p } _ { i } , \hat { \mathbf { p } } _ { i } ) ,\tag{3}
$$

$$
\mathcal { L } _ { c l s } = - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbb { 1 } ( v _ { i } > 0 ) \cdot \omega _ { i , c _ { i } ^ { * } } ^ { c l s } \sum _ { c \in \mathcal { C } } y _ { i , c } ^ { * } \log ( \hat { y } _ { i , c } ) ,\tag{4}
$$

where $\mathbb { 1 } ( \cdot )$ denotes the indicator function, equal to 1 if the specified condition is true and 0 otherwise. $v _ { i } > 0$ denotes visibility, $c _ { i } ^ { * }$ is the ground-truth semantic class, and $\ell ( \cdot , \cdot )$ represents the spatial regression loss (e.g., MSE).

2) Bio-Contrastive Boundary Loss $( { \mathcal { L } } _ { b i o } ) .$ : Limb deficiencies impose strict anatomical mutual exclusivity: Each limb branch in the kinematic structure admits only one anatomically valid terminal state. Thus, any residual points or additional joint predictions that violate biological constraints are physically infeasible. To enforce this, we directly leverage intralimb semantic mutual exclusivity. For an annotated residual keypoint $r \in \mathcal { R }$ , we define its Mutually Exclusive Set $\Omega _ { r }$ as the set of all keypoints that are biologically incompatible with r in the same limb branch:

$$
\Omega _ { r } = \{ j \in \mathcal { B } _ { L ( r ) } \mid j \neq r \ \& \ j \perp r \} ,\tag{5}
$$

where ${ \boldsymbol { B } } _ { L ( r ) }$ denotes all keypoints belonging to limb branch $L ( r )$ , and ⊥ indicates incompatibility. Our Bio-Contrastive Loss maximizes the probability of the valid residual keypoint r with respect to its mutually exclusive set. For a residual keypoint $^ { r , }$ we define its normalized biological confidence $\mathcal { P } _ { r }$ as:

$$
\mathcal { P } _ { r } = \frac { \exp ( p _ { r } ^ { \mathrm { b i o } } / \tau ) } { \exp ( p _ { r } ^ { \mathrm { b i o } } / \tau ) + \sum _ { j \in \Omega _ { r } } \mathbb { 1 } ( v _ { j } > 0 ) \cdot \exp ( p _ { j } ^ { \mathrm { b i o } } / \tau ) } ,\tag{6}
$$

where τ is a temperature hyperparameter. The resulting loss $\mathcal { L } _ { b i o }$ is then formulated as the negative log-likelihood over all the annotated residual keypoints:

$$
\mathcal { L } _ { b i o } = - \frac { 1 } { | \mathcal { R } | } \sum _ { r \in \mathcal { R } } \mathbb { 1 } \big ( \boldsymbol { v } _ { r } > 0 \mathrm { ~ } \& \mathrm { ~ } c _ { r } ^ { * } = B i o l o g i c a l \big ) \boldsymbol { \cdot } \log \left( \mathcal { P } _ { r } \right) ,\tag{7}
$$

where $c _ { r } ^ { * } = \mathtt { b i o l o g i c a l }$ ensures that the contrastive objective only operates on keypoints representing the true biological end of the residual limb. Minimizing loss $\mathcal { L } _ { b i o }$ encourages the model to increase the probability of residual limb types while suppressing intact limb predictions for their descendant joints. As a result, this loss efficiently suppresses anatomical hallucinations and improves the classification recall for both prosthetic (Pros) and absent (Abs) categories.

Overall, the total objective is formulated as:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { r e g } \mathcal { L } _ { r e g } + \lambda _ { c l s } \mathcal { L } _ { c l s } + \lambda _ { b i o } \mathcal { L } _ { b i o } .\tag{8}
$$

## V. EXPERIMENTS

## A. Implementation Details

We implement all baseline methods and our proposed framework using the MMPose codebase [45]. To ensure a fair and rigorous evaluation, identical training and testing splits are used across all methods. Since Omni-Pose involves both spatial localization and semantic classification, we evaluate models on both aspects. For geometric accuracy, we report the standard Mean Average Precision (mAP). Notably, Absent keypoints are excluded from $\mathrm { m A P }$ calculations (despite inferred visibility $v \ : = \ : 2 )$ since their locations are physically meaningless. Conversely, semantic accuracy evaluates attribute identification across all predicted types, including Absent.

a) Hyperparameter Settings.: For all experiments using ProLoss, we set $\lambda _ { r e g } = 1 , \lambda _ { c l s } = 0 . 0 5 , \lambda _ { b i o } = 0 . 0 1$ , and $\tau \ = \ 0 . 2$ . The classification-related losses are weighted to remain approximately one order of magnitude smaller than the coordinate regression loss. Empirically, we observe that AP and Type-Acc remain stable under moderate variations of these hyperparameters, indicating that ProLoss does not require task-specific tuning.

## B. Benchmark Results

We establish a comprehensive baseline on our ProPose dataset by evaluating state-of-the-art 2D pose estimators [11]– [13], [46], specifically comparing off-the-shelf pre-trained models against those fine-tuned with our ProLoss. We also evaluate fine-tuning with a standard spatial regression loss and LDLoss (visually compared in Fig. 6). As shown in Table III, models fine-tuned with ProLoss significantly outperform their pre-trained counterparts in spatial metrics. Additionally, topdown architectures demonstrate robust geometric precision, maintaining high mAP despite the increased complexity of localizing 6 extra extremity keypoints and predicting semantic types. Notably, ViTPose-L achieves a peak mAP of 90.0. However, standard models struggle to accurately classify minority joint states. While ViTPose-L performs best (92.3% for prosthetic and 85.9% for absent keypoints), others experience sharp drops, highlighting the inherent difficulty of distinguishing specialized mechanical structures from biological limbs.

To evaluate cross-dataset generalization, we manually annotated an independent test set from ProGait [20] videos. To ensure postural diversity, we sampled and manually filtered 2-3 frames from the initial and final quarters of each clip. As shown in Table IV, models maintain stable or higher metrics than on ProPose. Crucially, ProLoss-finetuned models consistently outperform pre-trained baselines. ViTPose-L remains the most robust against long-tail distributions, achieving 90.2% prosthetic accuracy, despite Swin-B marginally surpassing it on Bio joints. Ultimately, this confirms that ProPose-trained models readily generalize to novel real-world limb differences.

## C. Ablation Study

To validate our architecture and ProLoss components, we perform an ablation study using ViTPose-L on ProPose. As

TABLE III  
PERFORMANCE COMPARISON ON THE PROPOSE TEST SET. THE METRICS ARE DECOUPLED INTO GEOMETRIC PRECISION (AP AND AR ACROSS MULTIPLE THRESHOLDS) AND SEMANTIC DIAGNOSTICS (TYPE-ACC FOR TOTAL, BIO, PROS, AND ABS). THE ABS ACCURACY IS COMPUTED ONLY OVER NON-RESIDUAL KEYPOINTS, EXCLUDING ABSENT LABELS ON RESIDUAL KEYPOINTS, TO AVOID THE LARGE NUMBER OF STRUCTURALLY ABSENT RESIDUAL LABELS DOMINATING THE METRIC.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Training</td><td rowspan="2">Backbone</td><td rowspan="2">Input Size</td><td colspan="7">ProPose Test Set</td><td rowspan="2">Semantic</td></tr><tr><td></td><td colspan="4">Geometric (AP &amp; AR)</td><td colspan="2">(Type-Acc)</td></tr><tr><td rowspan="6">YoloxPose</td><td rowspan="3">Pretrain</td><td></td><td></td><td>AP</td><td>AP50</td><td>AP75</td><td>AR</td><td>AR50</td><td>AR75</td><td>Total Bio Pros Abs</td></tr><tr><td>YoloxPose-T</td><td>416×416</td><td>41.5</td><td>60.3</td><td>46.2</td><td>46.9</td><td>65.3</td><td>51.5</td><td></td></tr><tr><td>YoloxPose-S YoloxPose-M</td><td>640×640 640×640</td><td>48.8 52.2</td><td>66.2 68.7</td><td>54.8 59.1</td><td>55.1</td><td>71.3</td><td>60.6 64.8</td><td></td></tr><tr><td></td><td>YoloxPose-L 640×640</td><td></td><td>53.4 69.3</td><td>60.2</td><td>58.5 59.8</td><td>73.3 73.8</td><td>65.8</td><td></td></tr><tr><td rowspan="3">+ProLoss</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>YoloxPose-T</td><td>416×416 640×640</td><td>|41.9</td><td>67.3</td><td>43.0 53.6</td><td>49.0</td><td>72.5</td><td>50.4 82.9</td><td>89.7 53.8 27.7 90.4 64.6 52.1</td></tr><tr><td>YoloxPose-S YoloxPose-M</td><td>640×640</td><td>50.5 52.2</td><td>74.3 76.3</td><td>55.6</td><td>58.6 60.5</td><td>79.3 82.0</td><td>61.4 86.8 63.4 88.5</td><td>92.3 66.3 54.0</td></tr><tr><td rowspan="8">Swin Transformer</td><td rowspan="4">Pretrain</td><td>YoloxPose-L</td><td>640×640</td><td>53.3</td><td>76.7</td><td>56.3</td><td>61.0</td><td>81.2</td><td>63.5</td><td>89.0 92.6 69.4 56.6</td></tr><tr><td>Swin-T</td><td>256×192</td><td>74.7 95.7</td><td></td><td>83.5 78.9</td><td></td><td>96.1</td><td>85.9</td><td></td></tr><tr><td>Swin-B</td><td>256×192</td><td></td><td>|76.495.8</td><td>85.6</td><td>80.5</td><td>96.1</td><td>87.2</td><td></td></tr><tr><td>Swin-B</td><td>384×288</td><td>76.1</td><td>95.7</td><td>85.6</td><td>80.5</td><td>96.3</td><td>87.4</td><td></td></tr><tr><td>Swin-L</td><td>256×192</td><td>76.1</td><td>95.8</td><td>85.7</td><td>80.5</td><td>96.5</td><td>87.2</td><td></td></tr><tr><td>Swin-L</td><td>384×288</td><td>76.8</td><td>95.8</td><td>86.6</td><td>81.3</td><td>96.4</td><td>88.4</td><td></td></tr><tr><td rowspan="5">+ProLoss</td><td>Swin-T</td><td>256×192</td><td></td><td>|85.7 96.5</td><td>91.9</td><td>88.1</td><td>97.7</td><td>93.2 91.5</td><td>94.0 71.9 68.0</td></tr><tr><td>Swin-B</td><td>256×192</td><td>87.7</td><td>96.7</td><td>93.3</td><td>89.8</td><td>97.7</td><td>94.2</td><td>96.9 98.7 86.2 82.9</td></tr><tr><td>Swin-B</td><td>384×288</td><td>88.7</td><td>96.7</td><td>93.4</td><td>90.4</td><td>97.7</td><td>94.3 97.2</td><td>99.0 86.0 82.6</td></tr><tr><td>Swin-L Swin-L</td><td>256×192</td><td>86.4 96.7</td><td></td><td>92.2</td><td>88.8</td><td>97.5</td><td>93.7 96.7</td><td>98.6 83.6 82.7</td></tr><tr><td></td><td>384×288</td><td></td><td>|87.1 96.6</td><td>92.3</td><td>89.0</td><td>97.5</td><td>93.4</td><td>97.1 98.8 85.0 83.3</td></tr><tr><td rowspan="6">RTMPose</td><td rowspan="4">Pretrain</td><td>RTMPose-T</td><td>256×192</td><td>70.7</td><td>92.7</td><td>80.8 75.5</td><td></td><td>94.0</td><td>83.4</td><td></td></tr><tr><td>RTMPose-S RTMPose-M</td><td>256×192 256×192</td><td>73.3</td><td>94.7</td><td>83.2 78.1</td><td></td><td>95.5</td><td>85.5 87.8</td><td></td></tr><tr><td>RTMPose-L</td><td>256×192</td><td>76.1 77.2</td><td>95.8 95.9</td><td>85.6 86.8</td><td>80.8 81.9</td><td>96.3 96.8</td><td>88.9</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RTMPose-T RTMPose-S</td><td>256×192</td><td>|83.0</td><td>95.7</td><td>89.2</td><td>85.1</td><td>96.8</td><td>90.4 93.8</td><td>96.3 75.3 72.4</td></tr><tr><td rowspan="3">+ProLoss</td><td>RTMPose-M</td><td>256×192</td><td>86.1</td><td>96.6</td><td>92.3</td><td>87.9</td><td>97.4</td><td>93.0 94.7</td><td>96.8 78.5 78.0 97.8 82.5 80.3</td></tr><tr><td>RTMPose-L</td><td>256×192 256×192</td><td>88.8 89.9</td><td>96.8 96.8</td><td>93.6 94.6</td><td>90.4 91.3</td><td>97.8 97.9</td><td>94.8 95.8 95.1 96.5</td><td>98.2 85.3 83.7</td></tr><tr><td>ViT-S</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">ViTPose</td><td rowspan="3">Pretrain</td><td>ViT-B</td><td>256×192 256×192</td><td>76.3 78.1</td><td>95.7 95.7</td><td>85.6 87.7</td><td>80.6 82.6</td><td>96.1 96.8</td><td>87.3</td><td></td></tr><tr><td>ViT-L</td><td>256×192</td><td>80.5</td><td>96.9</td><td>90.8</td><td>85.1</td><td>97.6</td><td>89.7 82.3</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ViT-S ViT-B</td><td>256×192</td><td>|81.7</td><td>95.5</td><td>88.0</td><td>84.5</td><td>96.5</td><td>89.9</td><td>93.7</td><td>95.9 75.5 73.0</td></tr><tr><td rowspan="2">+ProLoss</td><td></td><td>256×192</td><td>82.0</td><td>95.5</td><td>88.9</td><td>84.7</td><td>96.8</td><td>90.2</td><td>95.0 97.5 75.5 75.8</td></tr><tr><td>ViT-L</td><td>256×192</td><td>90.0 96.7</td><td></td><td>94.5</td><td>91.9</td><td>98.0 95.7</td><td></td><td>97.0 98.2 92.3 85.9</td></tr></table>

Table V shows, results are decoupled into geometric precision and semantic diagnostics. Notably, we include the $\mathbf { A c c } _ { \mathrm { R e s - B i o } }$ metric to explicitly track the detection of physical residual stumps, preventing this rare subset from being overshadowed

by dominant biological joints.

a) Impact of Joint Feature Learning (Detach vs. Attached).: We first investigate the interaction between spatial coordinate regression and semantic type classification.

![](images/709c6fc84c4ffbc570a8726921ddd2b0ab82c1cf1aa935f90ba5c3659aef66ab.jpg)  
Fig. 6. Qualitative comparison between baseline methods and our ProLoss. Pre-trained baselines hallucinate biological limbs, while LDLoss completely ignores prostheses. Standard fine-tuning on keypoint coordinates fails to differentiate semantic types, leading to hallucinated absent joints. In contrast, our method accurately categorizes prosthetic and absent points to successfully reconstruct the kinematic structure.

Specifically, we compare a detached classification head, where gradients do not flow back to the shared backbone, against an attached, jointly trained configuration. As shown in Table V, when the classification head is detached, the model achieves an AP of 89.7% and a Pros accuracy of 85.7%. Conversely, allowing gradients from the classification loss to update the shared backbone yields simultaneous improvements in both metrics. Notably, the overall AP increases by 0.2%, alongside a substantial 4.2% surge in Pros accuracy. This firmly indicates that semantic attribute prediction acts as a beneficial auxiliary task, actively enhancing spatial feature representations rather than degrading geometric localization performance. Furthermore, it demonstrates that the standard feature embeddings extracted by the backbone inherently lack sufficient semantic information for accurate type classification without task-specific gradient feedback.

b) Effectiveness of Semantic Categorization Weighting.: Compared to the standard attached baseline without reweighting, applying Semantic Categorization Weighting exclusively to the classification head significantly improves the semantic accuracy of minority classes (Pros +1.0%, Abs +2.7%, Res-Bio +2.6%), while also boosting the overall geometric accuracy to a peak AP of 90.0%. Furthermore, we initially hypothesized that applying a frequency-based weight to the regression objective would improve the localization of rare residual keypoints. However, empirical results indicate that scaling the regression loss actually degrades geometric performance, causing the AP to drop back to 89.7%. This suggests that spatial coordinate regression is highly sensitive to artificial gradient scaling. Consequently, we discard regression weighting and apply distribution-aligned re-weighting strictly to the semantic classification branch.

c) Bio-Contrastive Boundary Loss $( \mathcal { L } _ { b i o } )$ and Training Strategy.: The inclusion of $\mathcal { L } _ { b i o }$ provides a strict logical constraint against anatomically impossible predictions. We compare two training strategies for incorporating this loss. Jointly training the entire model from scratch with $\mathcal { L } _ { b i o }$ forces the network to heavily prioritize semantic exclusivity, which slightly destabilizes spatial coordinate regression (AP drops to 89.6%). Conversely, adopting a two-stage finetuning strategy, where $\mathcal { L } _ { b i o }$ is applied only after the model has acquired stable spatial features, yields the optimal results presented in Table V.

![](images/2444f52a61f20fba04ff3c5f4c204ab5e9ad1322f7ce0c25be1f3d77a9f396e8.jpg)  
Fig. 7. Qualitative results of applying our proposed loss function to different pose estimation models. From left to right: ground truth, YOLO-Pose, ViTPose, Swin-Pose, and RTMPose trained with our loss on the ProPose dataset.

This strategy retains peak geometric performance (AP 90.0%) while dramatically boosting Abs accuracy to 85.6% and Res-Bio accuracy to 86.9% (a massive +5.8% improvement over the weighted baseline). This strongly demonstrates that $\mathcal { L } _ { b i o }$ acts as a robust kinematic filter, successfully localizing the rare true amputation boundaries.

d) Effect of Synthetic Data and ProLoss.: As shown in Table VI, adding synthetic data improves the recognition of rare semantic categories while maintaining comparable localization accuracy. For ViTPose-B, Real+Syn improves Pros accuracy from 55.1 to 63.3 and Abs accuracy from 56.6 to 58.5. A similar trend is observed for

TABLE IV  
BENCHMARKING RESULTS ON PROGAIT FOR CROSS-DATASET EVALUATION.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Training</td><td rowspan="2">Backbone</td><td rowspan="2">Input Size</td><td colspan="6">ProGait</td></tr><tr><td></td><td>Geometric (AP &amp; AR)</td><td></td><td></td><td></td><td>Semantic</td><td>(Type-Acc)</td></tr><tr><td rowspan="3">YoloxPose</td><td rowspan="2">Pretrain</td><td>YoloxPose-T</td><td>416×416</td><td>AP 44.2</td><td> $\mathsf { A P } ^ { 5 0 }$  58.7</td><td> $\mathsf { A P } ^ { 7 5 }$  55.5</td><td>AR 75.5 95.7</td><td> $\mathrm { A R } ^ { 5 0 } \ \mathrm { A R } ^ { 7 5 }$  90.6</td><td>Total Bio Pros Abs</td><td></td></tr><tr><td>YoloxPose-L</td><td>640×640</td><td>42.9 48.6</td><td>52.4 71.0</td><td>51.7</td><td>84.1</td><td>98.0 96.8</td><td></td><td></td></tr><tr><td>+ProLoss Pretrain</td><td>YoloxPose-T YoloxPose-L Swin-T</td><td>416×416 640×640 256×192</td><td>69.8 75.3</td><td>89.4</td><td>54.5 83.9</td><td>65.9 78.1</td><td>91.2 74.5 96.8 91.1</td><td>62.5 73.6</td><td>75.4 44.4 95.7 87.7 54.9 96.7</td></tr><tr><td rowspan="2">Swin Transformer</td><td></td><td>Swin-L Swin-T</td><td>384×288 256×192</td><td>73.7</td><td>97.9 96.8</td><td>93.3 90.5</td><td>79.6 98.4 78.1 97.2</td><td>94.3 92.7</td><td>一 一</td><td>一 一 一</td></tr><tr><td>+ProLoss Pretrain</td><td>Swin-L RTMPose-T</td><td>384×288 256×192</td><td>87.5 88.5 75.7</td><td>98.9 98.9 97.7</td><td>96.8 97.8 93.3</td><td>90.3 99.4 91.1 99.4 98.9</td><td>97.6 98.5 95.0</td><td>61.9 92.3 97.9</td><td>63.5 59.0 96.8 84.7 97.9</td></tr><tr><td rowspan="2">RTMPose</td><td>+ProLoss</td><td>RTMPose-L RTMPose-T</td><td>256×192 256×192</td><td>79.7 84.9</td><td>96.8 97.9</td><td>95.6 95.8</td><td>80.7 83.9 87.3</td><td>98.0</td><td>96.6</td><td>一 一 一</td></tr><tr><td>Pretrain</td><td>RTMPose-L ViT-S</td><td>256×192 256×192</td><td>89.5 75.2</td><td>99.0</td><td>97.9 91.6</td><td>99.0 99.3</td><td>96.7 98.7</td><td>85.0 90.3 97.2</td><td>94.8 72.5 99.1 80.6 99.1</td></tr><tr><td>ViTPose</td><td>+ProLoss</td><td>ViT-L ViT-S ViT-L</td><td>256×192 256×192 256×192 90.9</td><td>76.8 82.5 99.0</td><td>96.5 96.7 97.6</td><td>92.2 94.2 93.2 85.1</td><td>76.9 97.1 81.8 97.5 98.2</td><td>93.5 95.6 94.1</td><td>一 87.8</td><td>94.0 79.4 97.1</td></tr></table>

TABLE V

ABLATION STUDY OF MODEL ARCHITECTURE AND PROLOSS COMPONENTS. WE EVALUATE THE IMPACT OF GRADIENT FLOW (DETACH), SEMANTIC CATEGORIZATION WEIGHTING (SCW), AND BIO-CONTRASTIVE BOUNDARY LOSS $( { \mathcal { L } } _ { b i o } ) .$ . FOR $\mathcal { L } _ { b i o } .$ WE COMPARE JOINT TRAINING VERSUS STAGE-WISE FINETUNING (STAGED). RES-BIO EXPLICITLY EVALUATES THE CLASSIFICATION ACCURACY ON THE SPECIFIC RESIDUAL ENDPOINT KEYPOINTS.
<table><tr><td>Detach</td><td>SCW Cls Reg</td><td> $\mathcal { L } _ { b i o }$ </td><td>AP  $\mathbf { A P _ { 5 0 } }$ </td><td>Geometric Metrics</td><td> $\mathbf { A } \mathbf { P } _ { 7 5 }$ </td><td>AR</td><td>Total Bio Pros</td><td>Type-Acc (%)</td><td>Abs</td><td>Kp Acc  $\mathbf { I y p e } _ { R e s } { = } \mathbf { B i } \mathbf { 0 }$ </td></tr><tr><td>√</td><td></td><td></td><td></td><td>89.7</td><td>96.6</td><td>94.4</td><td>91.9</td><td>96.6</td><td>98.3 85.7</td><td>82.4 76.6</td></tr><tr><td></td><td>√</td><td></td><td>一</td><td>89.9 90.0</td><td>96.7 96.7</td><td>94.4 94.5</td><td>92.0 92.0</td><td>97.4 99.1</td><td>89.7 81.9</td><td>78.5 81.1</td></tr><tr><td></td><td>√</td><td></td><td></td><td>89.7 96.7</td><td>94.4</td><td>91.8</td><td>97.3 97.3</td><td>98.7 98.6</td><td>90.7 84.6 91.5 85.1</td><td>81.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $\left| \begin{array} { c } { { \checkmark } } \\ { { \checkmark } } \end{array} \right. ~ - ~ \left| \begin{array} { c } { { \mathrm { J o i n t } } } \\ { { \mathrm { S t a g e d } } } \end{array} \right.$ </td><td></td><td>89.6 96.7</td><td>94.4</td><td>91.6</td><td>97.1</td><td>98.4</td><td>91.5 84.9</td><td>81.5</td></tr><tr><td></td><td></td><td></td><td>90.0</td><td>96.7</td><td></td><td>91.9</td><td>97.0</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>94.5</td><td></td><td></td><td>98.3</td><td>91.7 85.6</td><td>86.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

RTMPose-M, where Pros and Abs accuracies increase from 63.5/67.3 to 79.8/76.0. Incorporating ProLoss further improves the challenging categories, especially residual keypoints and Pros/Abs semantic types, while keeping AP nearly unchanged. These results demonstrate that the performance gains come from both the data expansion and the structure-aware objective.

e) Fine-grained Impact ofReal-to-Synthetic Data Expansion.: To further analyze the effectiveness of our Real-to-Synthetic data expansion pipeline, we conduct a fine-grained evaluation using the ViT-Base architecture. As shown in Table VII, we compare the model trained exclusively on realworld data (Real Only) with the model trained on our enriched dataset (Real + Syn), and report both global semantic accuracy and per-joint prosthetic classification accuracy.

The results show that the synthetic data is particularly beneficial for long-tailed prosthetic joints. The global classification accuracy for Prosthetic (Pros) joints improves from 55.1% to 63.3%, while the accuracy for Absent (Abs) joints improves from 56.6% to 58.5%. More importantly, the gains are more pronounced on rare prosthetic categories. Specifically, the prosthetic classification accuracy for elbows increases from

TABLE VI  
ABLATION STUDY ON SYNTHETIC DATA AND PROLOSS.
<table><tr><td>Backbone</td><td>Data</td><td>Loss</td><td>AP</td><td>Type-Acc</td><td>Bio</td><td>Pros</td><td>Abs</td><td>Residual</td></tr><tr><td>ViTPose-B</td><td>Real</td><td>Standard</td><td>82.0</td><td>93.2</td><td>98.2</td><td>55.1</td><td>56.6</td><td>39.1</td></tr><tr><td>ViTPose-B</td><td>Real+Syn</td><td>Standard</td><td>81.9</td><td>93.6</td><td>97.8</td><td>63.3</td><td>58.5</td><td>40.7</td></tr><tr><td>ViTPose-B</td><td>Real+Syn</td><td>ProLoss</td><td>82.0</td><td>95.0</td><td>97.5</td><td>75.5</td><td>75.8</td><td>64.0</td></tr><tr><td>RTMPose-M</td><td>Real</td><td>Standard</td><td>88.9</td><td>94.5</td><td>98.6</td><td>63.5</td><td>67.3</td><td>47.1</td></tr><tr><td>RTMPose-M</td><td>Real+Syn</td><td>Standard</td><td>88.7</td><td>96.1</td><td>98.7</td><td>79.8</td><td>76.0</td><td>67.9</td></tr><tr><td>RTMPose-M</td><td>Real+Syn</td><td>ProLoss</td><td>88.8</td><td>95.8</td><td>97.8</td><td>82.5</td><td>80.3</td><td>77.4</td></tr></table>

## TABLE VII

IMPACT OF THE REAL-TO-SYNTHETIC DATA EXPANSION ON SEMANTICCLASSIFICATION AND SPATIAL LOCALIZATION. THE ENRICHED DATASETMITIGATES THE LONG-TAIL BIAS FOR PROSTHETIC (PROS) AND ABSENT(ABS) JOINTS, WHILE MAINTAINING COMPARABLE COORDINATEREGRESSION PERFORMANCE. WE FURTHER REPORT PER-JOINTPROSTHETIC CLASSIFICATION ACCURACY FOR REPRESENTATIVELONG-TAIL CATEGORIES.
<table><tr><td>Semantic Class</td><td>Real Only (Acc)</td><td>Real + Syn (Acc)</td><td>∆</td></tr><tr><td>Global: Pros (↑) Global: Abs (↑)</td><td>55.1% 56.6%</td><td>63.3% 58.5%</td><td>+8.2% +1.9%</td></tr><tr><td>Elbows (Pros) (↑)</td><td>30.2%</td><td>50.4%</td><td>+20.2%</td></tr><tr><td>Wrists (Pros) (↑)</td><td>55.6%</td><td>70.2%</td><td>+14.6%</td></tr><tr><td>Fingertips (Pros) (↑)</td><td>32.4%</td><td>40.0%</td><td>+7.6%</td></tr><tr><td>Knees (Pros) (↑)</td><td>50.5%</td><td>56.6%</td><td>+6.1%</td></tr><tr><td>Coord. Reg.</td><td>Real Only</td><td>Real + Syn</td><td>∆</td></tr><tr><td>AP (↑) AR (↑)</td><td>82.0% 81.0%</td><td>81.9% 81.3%</td><td>-0.1% +0.3%</td></tr></table>

30.2% to 50.4%, yielding a 20.2% improvement. Wrists also show a substantial gain, increasing from 55.6% to 70.2%. Distal extremities and lower-limb joints, such as fingertips and knees, also improve by 7.6% and 6.1%, respectively.

These fine-grained results indicate that the Real-to-Synthetic data expansion pipeline effectively enriches rare prosthetic configurations and mitigates the long-tail bias in prosthetic joint recognition. Meanwhile, the coordinate regression performance remains comparable, with AP changing from 82.0% to 81.9% and AR improving from 81.0% to 81.3%. This suggests that the synthetic data improves semantic recognition of rare prosthetic joints without compromising the core spatial localization capability.

f) Additional Comparative Analysis of Learning Objectives.: Due to space constraints in the original conference manuscript, the supplementary material presented additional comparisons with standard coordinate regression and the existing LDLoss. A primary concern when introducing semantic classification into pose estimation is the potential degradation of spatial regression accuracy. However, compared to the “Coordinate Regression Only” baseline, equipping the model with ProLoss does not compromise the Average Precision (AP). We also observe that directly applying LDLoss to our dataset yields suboptimal geometric performance, because LDLoss is fundamentally designed under the assumption that the kinematic chain terminates at the residual stump. While effective for unprosthetized amputations, this constraint conflicts with our Omni-Pose protocol, which aims to reconstruct the entire mechanical structure of prostheses. In contrast, ProLoss adapts to the Omni-Pose protocol by extending the network’s capability to perform semantic categorization while maintaining spatial AP.

## VI. CONCLUSION

We introduce ProPose, a comprehensive benchmark designed to resolve the representation bias and topological ambiguity in pose estimation. By proposing the Omni-Pose protocol, we effectively unify the semantic and spatial representation of intact, residual, and prosthetic limbs within a single framework. Moreover, our real-to-synthetic pipeline effectively tackles data scarcity. By enforcing intra-limb semantic dependencies, ProLoss suppresses anatomically impossible predictions. Extensive evaluations show our approach significantly improves accuracy on specialized mechanical structures and residual boundaries. We hope ProPose advances inclusive human pose estimation.

## ACKNOWLEDGMENTS

This research is funded in part by ARC-Discovery grant (DP220100800 to XY), ARC-DECRA grant (DE230100477 to XY), the Advance Queensland Industry Research Projects (AQIRP), and Follow ME PTY LTD. We thank all anonymous reviewers and ACs for their constructive suggestions.

## REFERENCES

[1] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in European conference on computer vision. Springer, 2014, pp. 740–755.

[2] J. Ying, H. Du, K. Zhang, L. Li, and X. Yu, “Ldpose: Towards inclusive human pose estimation for limb-deficient individuals in the wild,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 9865–9875.

[3] S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[4] H. Duan, Y. Zhao, K. Chen, D. Lin, and B. Dai, “Revisiting skeletonbased action recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 2969–2978.

[5] T. Guo, P. A. Logan, T. Wackwitz, and D. Martin, “Plnet-12: A visionlanguage benchmark for zero-shot physical literacy analysis across 12 fundamental movements,” in Australasian Joint Conference on Artificial Intelligence. Springer, 2025, pp. 242–254.

[6] Z. Cai, W. Yin, A. Zeng, C. Wei, Q. Sun, W. Yanjun, H. E. Pang, H. Mei, M. Zhang, L. Zhang et al., “Smpler-x: Scaling up expressive human pose and shape estimation,” Advances in Neural Information Processing Systems, vol. 36, pp. 11 454–11 468, 2023.

[7] S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa, and J. Malik, “Humans in 4d: Reconstructing and tracking humans with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 14 783–14 794.

[8] S. Wang, J. Li, T. Li, Y. Yuan, H. Fuchs, K. Nagano, S. De Mello, and M. Stengel, “Blade: Single-view body mesh estimation through accurate depth estimation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 991–22 000.

[9] S. D. Uhlrich, A. Falisse, Ł. Kidzinski, J. Muccini, M. Ko, A. S. Chaud-´ hari, J. L. Hicks, and S. L. Delp, “Opencap: Human movement dynamics from smartphone videos,” PLoS computational biology, vol. 19, no. 10, p. e1011462, 2023.

[10] J. Stenum, M. Hsu, A. Pantelyat et al., “Clinical gait analysis using video-based pose estimation: multiple perspectives, clinical populations, and measuring change. plos digital health 2024 3 e0000467.”

[11] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “Vitpose: Simple vision transformer baselines for human pose estimation,” Advances in neural information processing systems, vol. 35, pp. 38 571–38 584, 2022.

[12] T. Jiang, P. Lu, L. Zhang, N. Ma, R. Han, C. Lyu, Y. Li, and K. Chen, “Rtmpose: Real-time multi-person pose estimation based on mmpose,” arXiv preprint arXiv:2303.07399, 2023.

[13] D. Maji, S. Nagori, M. Mathew, and D. Poddar, “Yolo-pose: Enhancing yolo for multi person pose estimation using object keypoint similarity loss,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 2637–2646.

[14] Y. Tian, Q. Ye, and D. Doermann, “Yolov12: Attention-centric real-time object detectors,” arXiv preprint arXiv:2502.12524, 2025.

[15] R. Sapkota, R. H. Cheppally, A. Sharda, and M. Karkee, “Yolo26: key architectural enhancements and performance benchmarking for real-time object detection,” arXiv preprint arXiv:2509.25164, 2025.

[16] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele, “2d human pose estimation: New benchmark and state of the art analysis,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2014.

[17] J. Li, C. Wang, H. Zhu, Y. Mao, H.-S. Fang, and C. Lu, “Crowdpose: Efficient crowded scenes pose estimation and a new benchmark,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10 863–10 872.

[18] S.-H. Zhang, R. Li, X. Dong, P. Rosin, Z. Cai, X. Han, D. Yang, H. Huang, and S.-M. Hu, “Pose2seg: Detection free human instance segmentation,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 889–898.

[19] S. Jin, L. Xu, J. Xu, C. Wang, W. Liu, C. Qian, W. Ouyang, and P. Luo, “Whole-body human pose estimation in the wild,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020.

[20] X. Yin, B. Yang, W. Liu, Q. Xue, A. Alamri, G. Fiedler, and W. Gao, “Progait: A multi-purpose video dataset and benchmark for transfemoral prosthesis users,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 8984–8993.

[21] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[22] A. Doering, D. Chen, S. Zhang, B. Schiele, and J. Gall, “PoseTrack21: A dataset for person search, multi-object tracking and multi-person pose tracking,” in CVPR, 2022.

[23] Y. Zhu, N. Samet, and D. Picard, “H3wb: Human3.6m 3d wholebody dataset and benchmark,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 20 166– 20 177.

[24] J. Wang, F. Yang, B. Li, W. Gou, D. Yan, A. Zeng, Y. Gao, J. Wang, Y. Jing, and R. Zhang, “Freeman: Towards benchmarking 3d human pose estimation under real-world conditions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 978–21 988.

[25] H. Du, J. Ying, S. Wang, X. Li, K. Zhang, and X. Yu, “Inclusivevidpose: Bridging the pose estimation gap for individuals with limb deficiencies in video-based motion,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=SyQqXAdWUq

[26] W. Liu, Q. Bao, Y. Sun, and T. Mei, “Recent advances of monocular 2d and 3d human pose estimation: A deep learning perspective,” ACM Computing Surveys, vol. 55, no. 4, pp. 1–41, 2022.

[27] J. Wang, K. Sun, T. Cheng, B. Jiang, C. Deng, Y. Zhao, D. Liu, Y. Mu, M. Tan, X. Wang et al., “Deep high-resolution representation

learning for visual recognition,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 10, pp. 3349–3364, 2020.

[28] Z. Yang, A. Zeng, C. Yuan, and Y. Li, “Effective whole-body pose estimation with two-stages distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4210–4220.

[29] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh, “Openpose: Realtime multi-person 2d pose estimation using part affinity fields,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.

[30] B. Cheng, B. Xiao, J. Wang, H. Shi, T. S. Huang, and L. Zhang, “Higherhrnet: Scale-aware representation learning for bottom-up human pose estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5386–5395.

[31] Z. Geng, K. Sun, B. Xiao, Z. Zhang, and J. Wang, “Bottom-up human pose estimation via disentangled keypoint regression,” in CVPR, 2021.

[32] X. Cao, M. Xu, X. Yu, J. Yao, W. Ye, S. Huang, M. Zhang, I. Tsang, Y.-S. Ong, J. T. Kwok et al., “Analytical survey of learning with lowresource data: From analysis to investigation,” ACM Computing Surveys, vol. 58, no. 6, pp. 1–47, 2025.

[33] G. Varol, J. Romero, X. Martin, N. Mahmood, M. J. Black, I. Laptev, and C. Schmid, “Learning from synthetic humans,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 109–117.

[34] W. Huang, S. Ghahremani, S. Pei, and Y. Zhang, “Wheelpose: data synthesis techniques to improve pose estimation performance on wheelchair users,” in Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, 2024, pp. 1–25.

[35] T. Guo, C. Liu, and X. Yu, “Beyond single-view sufficiency: Cvbench for cross-view human understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 7154–7164.

[36] T. Guo, C. Liu, L. Chen, and X. Yu, “Ssmnbench: Diagnosing image-based cross-view human-object understanding via single-view sufficiency and multi-view necessity,” 2026. [Online]. Available: https://arxiv.org/abs/2606.25634

[37] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. ming Yin, S. Bai, X. Xu, Y. Chen, Y. Chen, Z. Tang, Z. Zhang, Z. Wang, A. Yang, B. Yu, C. Cheng, D. Liu, D. Li, H. Zhang, H. Meng, H. Wei, J. Ni, K. Chen, K. Cao, L. Peng, L. Qu, M. Wu, P. Wang, S. Yu, T. Wen, W. Feng, X. Xu, Y. Wang, Y. Zhang, Y. Zhu, Y. Wu, Y. Cai, and Z. Liu, “Qwen-image technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2508.02324

[38] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[39] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3836–3847.

[40] C. Zauner, “Implementation and benchmarking of perceptual image hash functions,” 2010.

[41] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25 278–25 294, 2022.

[42] R. Chen, X. Chen, B. Ni, and Y. Ge, “Simswap: An efficient framework for high fidelity face swapping,” in Proceedings of the 28th ACM international conference on multimedia, 2020, pp. 2003–2011.

[43] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9268– 9277.

[44] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.

[45] M. Contributors, “Openmmlab pose estimation toolbox and benchmark,” https://github.com/open-mmlab/mmpose, 2020.

[46] Z. Xiong, C. Wang, Y. Li, Y. Luo, and Y. Cao, “Swin-pose: Swin transformer based human pose estimation,” in 2022 IEEE 5th International Conference on Multimedia Information Processing and Retrieval (MIPR). IEEE, 2022, pp. 228–233.