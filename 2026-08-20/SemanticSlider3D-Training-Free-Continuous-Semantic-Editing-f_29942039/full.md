# SemanticSlider3D: Training-Free Continuous Semantic Editing for 3D Objects

Ru Wang<sup>∗</sup>   
ru.wang@wisc.edu   
Department of Computer Sciences   
University of Wisconsin–Madison   
Madison, Wisconsin, USA

Koichiro Niinuma kniinuma@fujitsu.com Fujitsu Research of America Pittsburgh, Pennsylvania, USA

Rahul Jain jain348@purdue.edu Department of Electrical and Computer Engineering Purdue University West Lafayette, Indiana, USA

Aakar Gupta   
agupta@fujitsu.com   
Fujitsu Research of America   
Redmond, Washington, USA

![](images/6dc6fcba398326e60b5b6c6e751a241ab3a9fcf70f6903483c2a4b9340d7391d.jpg)  
Figure 1: The pipeline of generating image conditions and editing through generation process. We use a piano as an example, with "very futuristic looking" being the target attribute specified by the user. (a) Rendering � camera views from the original 3D object; (b) generating contrastive editing prompts for both positive direction and negative direction based on user-specified attribute; (c) ranking and selecting camera views relevant to the editing prompts; (d) generating contrastive image pairs using edit-relevant views and editing prompt pairs for image editing; (e) Editing 3D objects along the semantic axis using TRELLIS with multi-view conditions and steering direction defined by contrastive image pairs.

## Abstract

Fine-grained control over continuous semantic attributes of 3D objects is essential for 3D content creation, but is not well supported by conventional 3D modeling workflows or prompt-based interac tion with existing generative AI tools. While slider-based methods have proven efective for fine-grained semantic control in 2D image generation, no equivalent approach exists for 3D. Extending these

2D methods to 3D is non-trivial due to challenges unique to 3D, including geometric integrity and cross-view coherence. We present SemanticSlider3D, a technique for continuous semantic attribute editing of 3D objects that requires no per-attribute training. Given a user-specified attribute, our pipeline constructs a semantic editing direction in the latent space of a state-of-the-art 3D generation model, presenting a diverse and coherent spectrum of 3D variations. A technical validation on a dataset of 50 3D object-attribute pairs shows our method was preferred by all five human assessors across variation range, consistency, 3D object quality, and attribute disentanglement, over a baseline combining a 2D slider with an image-to-3D model. An exploratory study with six participants demonstrates that SemanticSlider3D supported decision-making in 3D prototyping and was perceived as a valuable addition to existing workflows.

CCS Concepts

• Human-centered computing → Interactive systems and tools; • Computing methodologies → Computer vision.

Keywords

Generative AI, Semantic Editing, Creativity Support, Computer Vision

## ACM Reference Format:

Ru Wang, Rahul Jain, Koichiro Niinuma, and Aakar Gupta. 2026. SemanticSlider3D: Training-Free Continuous Semantic Editing for 3D Objects. In The 39th Annual ACM Symposium on User Interface Software and Technology (UIST ’26), November 02–05, 2026, Detroit, MI, USA. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3830398.3830718

## 1 Introduction

3D design is an essential activity in product design [58], architectural design [3], game development [36], and many other creative industries. The early stage of 3D design involves generating, refining, and iterating on ideas about the appearance of 3D models [58], which can be time-consuming and requires significant manual efort with conventional 3D modeling workflow [5, 58].

Recent advances in generative AI allow users to generate and edit high-quality 3D assets through text and image prompts [9, 40, 46, 75, 76], and such tools have been increasingly integrated into 3D prototyping workflows [19, 25, 43, 65, 71]. Through prompt-based interactions, users can rapidly create diverse alternatives and edit 3D objects to refine prototypes. However, these generative models largely operate in an end-to-end fashion, making it dificult to intervene between the prompt and the output. This is particularly problematic for editing continuous semantic attributes such as style or material finish [15, 24, 28], where users need fine-grained, incremental control over the intensity of the edit rather than a binary change [15, 18]. In practice, this leads to repetitive prompting until a generation aligns with the user’s expectation, and regeneration often changes unintended attributes [19].

There has been extensive prior work on continuous control of semantic attributes for 2D images using slider-based interfaces [15, 16, 20, 23, 24, 28, 31]. Although slider-based editing has proven efective for precise image generation and editing, it remains unclear how this approach applies to 3D, given challenges specific to 3D such as structural integrity [19] and multi-view consistency [59, 65, 68].

To close this gap, we propose SemanticSlider3D, a slider-based semantic attribute editing technique for 3D objects that requires no per-attribute training. With SemanticSlider3D, the user specifies a continuous attribute for any 3D object using a text prompt. Our pipeline selects the most representative camera views relative to the target attribute, generates contrastive image pairs aligned with the negative and positive extremes of the attribute, and leverages TRELLIS [75], a state-of-the-art image-to-3D model, to create an editing direction in the latent space. At inference time, the user sees multiple variations on a slider representing diferent intensities of the edit in both directions. Users can add intermediate points for finer granularity and perform sequential editing by anchoring new sliders on existing variants.

To validate our technique, we conducted a technical evaluation with five human assessors using a custom 3D dataset of diverse everyday objects and well-defined continuous attributes. We found that SemanticSlider3D demonstrated high variation range, reasonable variation consistency, good 3D object quality, and strong preservation of unintended attributes, and was overwhelmingly preferred by human assessors over a baseline combining a state-of-the-art image slider [28] with a state-of-the-art image-to-3D model [9]. We further conducted an exploratory user study with six participants of varying 3D prototyping experience to understand whether and how slider-based 3D editing supports 3D prototyping. We found that SemanticSlider3D facilitated decision-making in exploring and refining design ideas, helped communicate the generative AI model’s capabilities, and was perceived as a useful addition to participants 3D prototyping workflows.

Our contributions are three-fold: (1) a novel slider-based 3D semantic editing technique; (2) a technical validation with human assessors on a representative custom dataset, demonstrating the efectiveness of our technique; and (3) an exploratory user study showing the potential of SemanticSlider3D for 3D prototyping.

## 2 Related Work

## 2.1 3D Generation and Editing

2.1.1 3D Generation. Recent advances in computer vision have enabled the creation of 3D content from text, images, and other modalities [40, 41, 45, 53, 59, 73, 75]. One direction is to directly reconstruct or generate 3D content. Early approaches relied on Neural Radiance Fields (NeRF) and optimization-based pipelines to reconstruct 3D objects/scenes from sparse inputs [35, 53, 60]. Recent work has explored difusion-based models for 3D generation [49, 59], autoregressive approaches inspired by GPT-style modeling for 3D synthesis [12], and eficient representations such as 3D Gaussian Splatting (GS) for real-time rendering and editing [41]. More recently, rectified flow models have been explored as an alternative to difusion-based methods and have shown strong performance for large-scale 3D generation [9, 75]. Another direction is to leverage 2D image generation models for 3D generation, due to their strong generalization abilities. These approaches typically optimize 3D representations by distilling knowledge from pretrained image difusion models [48, 49, 69, 73]. Alternatively, some methods first generate multi-view images using difusion models and then reconstruct the corresponding 3D assets from these views[44, 66, 76]. However, since these methods rely on 2D generation, they often result in weaker geometry compared to models trained on 3D data, mainly due to inconsistencies across diferent views.

2.1.2 3D Editing. Most existing 3D editing approaches are promptbased and rely on 2D representations as intermediaries, either optimizing 3D representations via distillation from pretrained difusion models [8, 56, 64] or editing multi-view images and reconstructing the 3D assets iteratively [2, 7, 21, 33]. However, these methods sufer from multi-view inconsistencies. A few works explore directly editing 3D latent representations [1, 10, 57, 72], However, these methods ofer limited control, because edits are specified through high-level prompts without explicit control over the magnitude or structural consistency of the change. Rectified flow-based models [47, 75] provide a more promising foundation by directly modeling 3D data distributions with improved control for structured editing. However, like prior approaches, TRELLIS relies on discrete text prompts and does not support continuous control over semantic attributes. Avatar Concept Slider [26] brings continuous slider-based control to 3D, but is specific to human avatars and requires per-concept training on a 3D GS representation. No existing method supports training-free continuous semantic editing for general 3D objects. SemanticSlider3D builds on TRELLIS by introducing a pipeline for precise and continuous 3D editing.

## 2.2 3D generative authoring systems in HCI

3D generation and editing have been explored in HCI to support design and creative workflows, 3D authoring, and 3D content creation [11, 14, 18, 25, 51, 65]. Systems such as DreamCrafter [71] and GaussianShopVR [65] enable users to generate and modify 3D content using 3D generation models with text prompts and immersive interfaces. For example, GaussianShopVR allows point-level editing of 3D Gaussian Splatting scenes, combining manual selection tools with AI-driven editing to provide fine-grained control. Other works integrate 2D generative models into 3D design pipelines, such as combining 3D sketching with image generation or incorporating text-to-image models into 3D workflows (e.g., 3DALL-E [51]). DesignFromX [18] further explores user-driven design by allowing the composition of features from images of reference products. Other work has explored direct manipulation and sketch-based interfaces for 3D modeling [78] and AR/VR-based interaction systems that enable users to interact with 3D content in immersive environments [39, 65]. However, most of these systems rely on image or prompt-based interactions, where users specify edits at a high level but have limited ability to continuously control or refine specific attributes. This makes it dificult to perform fine-grained adjustments or maintain consistent control over spatial and semantic properties of the 3D scene. In this work, SemanticSlider3D address this limitation by enabling continuous control in 3D editing.

## 2.3 Semantic Sliders for 2D Image Editing

Slider-based editing of semantic attributes for 2D images has been explored in multiple recent works due to its ability to provide user-friendly, low-dimensional, continuous control over semantic attributes [16, 22, 23, 38]. These methods typically learn or identify semantic directions in the latent space, either using lightweight training (e.g., LoRA adapters) or training-free approaches that leverage pretrained models. Moving along these directions enables continuous control over attributes. Surprisingly, no such methods exist for 3D objects. This is partly because semantic attributes are easier to define and control in 2D images: Many pretrained generative models provide well-aligned latent directions that map to visual semantics. In contrast, 3D representations generally lack such disentangled semantic axes. Edits also need to preserve geometric structure and remain consistent across multiple views. This makes continuous control in 3D significantly more challenging. An obvious approach for 3D sliders would be to leverage 2D sliders for 3D editing by projecting 3D content into a 2D representation, applying semantic edits in the image domain, and mapping the results back to 3D. Such 3D-2D-3D pipelines are appealing as they reuse wellestablished editing techniques without requiring native 3D control mechanisms. We explored this pipeline as a baseline. However, we found that although this is a good starting point, the edits do not consistently translate to the intended 3D outcomes. Because each stage of this pipeline can introduce fidelity loss, making it dificult to enforce spatial relationships and physical constraints across the full pipeline. Thus, directly extending 2D slider approaches to 3D settings is insuficient. We propose SemanticSlider3D, which performs semantic editing directly in the 3D latent space to avoid this compounding fidelity loss.

## 3 SemanticSlider3D: Design Goals

Given a 3D object and a user-specified semantic attribute with a continuous range of variation, our goal is to generate variations at diferent positions along the full spectrum defined by that attribute, for example, a steak at diferent levels of doneness. To enable continuous semantic editing for 3D objects while ensuring a reasonable and efective slider-based interaction, we incorporate insights from prior literature in both 3D generation [19, 66, 75, 76] and 2D image sliders [20, 22, 23, 28, 38], and derive the following design goals (DG) for SemanticSlider3D:

DG1: Ensuring 3D Coherence. Variations generated along an semantic attribute should be visually coherent and plausible. Unlike 2D images, 3D objects involve both geometry and appearance, meaning directly lifting image variations from a 2D slider to 3D can be problematic because important structural information may be lost or inconsistent across camera views [66]. Even with multi-view 3D generation models, coherence issues, such as the Janus problem (3D objects with multiple faces or heads), can arise in the resulting edited objects [7]. Prior work has emphasized the importance of maintaining geometric and appearance quality in generated 3D objects [19, 65, 71]. We aim to ensure that each variation along the slider preserves the overall structural integrity of the origina object while applying the intended edit consistently across views.

DG2: Wide Range ofMeaningful Variations. A wide range of variations helps users explore how the attribute manifests at diferent intensities and identify the level that best matches their intent [38]. Drawing insights from 2D slider-based editing, the efective range of a slider varies depending on the specific object and attribute—some combinations produce visible changes within a narrow range while others require larger shifts before the edit becomes visually apparent [28, 38]. Therefore, the slider bounds should be adapted for each object and attribute while maximizing the diversity between the two ends.

DG3: Consistent Variation within the Slider. Variations along the slider should progress smoothly and predictably, so that each step produces visually proportional change in the attribute. A challenge with slider-based editing is that the latent space can be highly nonlinear. Users may see incremental changes across several positions and then see an abrupt jump at another[20, 24, 38], leading to expectation mismatch [38]. We aim to ensure that SemanticSlider3D produces variations that change gradually and uniformly across the slider range, so that moving the slider by a similar amount results in a similar degree of visual change.

DG4: Preserving Unintended Atributes. Editing one attribute should not change other irrelevant attributes of the 3D object. Prior work in 2D slider has proposed various methods to isolate the target edit direction from unrelated attributes (i.e. disentanglement)[20, 24, 28]. We aim to ensure that SemanticSlider3D modifies only the intended attribute while preserving the remaining characteristics of the original 3D object as much as possible.

DG2–DG4 extend challenges from 2D sliders, but manifest dif ferently in 3D (§4). DG1 is unique to 3D and is a critical challenge to solve for 3D semantic sliders.

## 4 SemanticSlider3D: Design & Implementation

Building on these design goals, we present the pipeline of SemanticSlider3D. Our technique consists of three stages: (1) generating image conditions from the 3D object for both reconstruction and attribute editing, (2) steering the generation process to produce 3D variations at diferent attribute intensities, and (3) mapping variations to perceptually consistent slider positions with adaptive bounds. We first describe the technical foundation of our technique, then detail each stage.

## 4.1 Technical Foundation

SemanticSlider3D builds on TRELLIS [75], a state-of-the-art 3D generation model that represents 3D objects using Structured LATents (SLAT)—local latent vectors defined on a sparse 3D voxel grid that jointly encode both geometry and appearance. These structured latents can be decoded into multiple output formats, including 3D Gaussians, Radiance Fields, and meshes. We selected TRELLIS after evaluating several 3D generation models for two reasons: first, its flow matching-based pipeline supports inference-time intervention without per-attribute training, which scales better than LoRA-based approaches [28] when users need to define arbitrary attributes on the fly; second, its two-stage architecture ensures high-quality 3D generation.

TRELLIS generates 3D objects through a two-stage pipeline using flow matching models [50]. The first stage generates the sparse structure (which voxels are active in a 3D voxel grid), and the second stage generates the latent vectors attached to these voxels, conditioned on text or image inputs. Multi-view renders of a 3D object (O) can be used as image conditioning $( c _ { O } )$ to generate the corresponding SLAT through the flow model’s sampling process. This generation-based conversion is key to our editing approach, because the editing happens within the sampling process itself. At each timestep, the flow model predicts a velocity $\mathbf { v } _ { \theta } ( x _ { t } , t , c )$ that guides the noisy sample $x _ { t }$ toward a structured latent consistent with the conditioning �.

While computing editing directions (vectors in the latent space that correspond to specific semantic changes) from contrastive conditions is well-established in 2D image sliders [20, 24, 28], applying this principle to a 3D structured latent space has been underexplored. We found that TRELLIS’s flow matching formulation naturally supports training-free semantic steering: Given contrastive image conditions $c _ { + }$ and �<sub>−</sub> corresponding to the positive and negative extremes of an attribute, a steering direction can be computed from the diference between their predicted velocities: ${ \bf d } _ { s } = { \bf v } _ { \theta } ( x _ { t } , t , c _ { + } ) - { \bf v } _ { \theta } ( x _ { t } , t , c _ { - } )$ . By scaling ${ \bf { d } } _ { s }$ with a factor � and adding it to the velocity predicted under the original object’s conditioning $\mathbf { v } _ { \theta } ( x _ { t } , t , c _ { O } )$ , the sampling trajectory is steered toward diferent intensities of the target attribute.

We describe how each stage of our pipeline works in the following subsections.

## 4.2 Generating Image Conditions

We first render multiple camera views from the 3D object to get initial image condition, and then select and edit attribute-relevant views to construct contrastive image pairs for steering the generation in the latent space.

4.2.1 Multi-View Rendering as Initial Condition. For a given 3D object, we sample � cameras uniformly distributed across a sphere surrounding the object using the Hammersley quasi-random sequence [74] (Fig 1a). Each camera is placed at a fixed radius of 2.0 from the object’s center with a 40° field of view, looking toward the center. We render a set of camera views $\{ p _ { i } \} _ { i = 1 } ^ { N }$ using Blender following a similar process as TRELLIS. The rendered images are encoded as multi-view conditions $\{ c _ { i } \} _ { i = 1 } ^ { N }$ via DINOv2 [55], which serve as the conditioning for reconstructing the original object through the flow model’s sampling process.

4.2.2 Generating Contrastive Editing Prompts. For a user-specified continuous semantic attribute, we use an LLM (GPT-5.2) to generate a pair of editing prompts corresponding to the positive and negative extremes of the attribute, denoted as $( t ^ { + } , t ^ { - } )$ (Fig 1b). Some attributes can be dificult to describe with a neutral statement (e.g., how traditional a sofa looks). In such cases, we also allow the user to provide only a prompt describing the change in the positive direction, and the LLM automatically generates the corresponding prompt pairs. The LLM instruction prompt is provided in the Appen dix A.1.1. These editing prompts are used to guide the generation of contrastive image pairs in the following steps.

4.2.3 Selecting Edit-Relevant Views. Rather than generating entirely new images from the editing prompts, which would introduce uncontrolled alteration beyond the target attribute, we edit the target attribute directly on the rendered camera views of the original object (DG4). This strategy ensures that the contrastive image pairs difer from the original only in the target attribute while preserving everything else. However, not all camera views contain visual information relevant to the target attribute. For example, when editing the eye size of a 3D character, views from the back do not contain the relevant attribute for efective editing. To select edit-relevant views, we use the ImageReward score [38, 77] to measure the alignment between the editing prompt and each camera view, where a higher score indicates greater relevance (Fig 1c). However, irrelevant views can occasionally receive high scores due to visual ambiguity at certain camera angles. To mitigate this, we additionally use GPT-5.2 to verify whether the target attribute is actually present in each view (Appendix A.1.2). Using this twostep pipeline, we select the � most edit-relevant views $\{ \hat { p } _ { i } \} _ { i = 1 } ^ { M }$ for constructing contrastive image pairs.

4.2.4 Generating Contrastive Image Pairs. Using GPT Image 1.5, we edit each of the � selected views with both $t ^ { + }$ and $t ^ { - }$ , appending a reassurance instruction that explicitly prevents modification of attributes irrelevant to the target edit (DG4). Because the contrastive image pairs are used to compute a steering direction through veloc ity diferences rather than to directly reconstruct a 3D object, strict multi-view consistency across the edited images is not required [24, 28]. This produces contrastive image pairs $\{ ( \hat { p } _ { i } ^ { + } , \hat { p } _ { i } ^ { - } ) \} _ { i = 1 } ^ { M }$ that difer primarily in the target attribute while preserving other visual characteristics of the original view (Fig 1d). The image pairs are then encoded as condition pairs $\{ ( \hat { c } _ { i } ^ { + } , \hat { c } _ { i } ^ { - } ) \} _ { i = 1 } ^ { M }$ via DINOv2.

## 4.3 Editing via Generation

As described in § 4.1, semantic editing can be achieved by computing a steering direction from contrastive conditions and injecting it into the sampling process. Since our contrastive conditions span � view pairs, we compute the steering direction at each timestep � as the average velocity diference across all contrastive pairs:

$$
\mathbf { d } _ { s } ( x _ { t } , t ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ \mathbf { v } _ { \theta } ( x _ { t } , t , \hat { c } _ { i } ^ { + } ) - \mathbf { v } _ { \theta } ( x _ { t } , t , \hat { c } _ { i } ^ { - } ) \right]\tag{1}
$$

Averaging across view pairs produces a more robust editing direction that captures the target attribute from multiple perspectives while canceling view-specific noise, isolating the edit to the target attribute (DG4).

To improve reconstruction quality (DG1) and further preserve unintended attributes (DG4), we apply classifier-free guidance [34] to the average velocity predicted under the original object’s multi view conditioning:

$$
\bar { \mathbf { v } } ( x _ { t } , t ) = \mathbf { v } _ { \theta } ( x _ { t } , t , \emptyset ) + w \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { v } _ { \theta } ( x _ { t } , t , c _ { i } ) - \mathbf { v } _ { \theta } ( x _ { t } , t , \emptyset ) \right]\tag{2}
$$

where ∅ is the empty condition and � the guidance strength.

The edited velocity at timestep � is then computed by adding the scaled steering direction to this guided velocity:

$$
\mathbf { v } _ { \mathrm { e d i t } } ( x _ { t } , t ) = \bar { \mathbf { v } } ( x _ { t } , t ) + \alpha \mathbf { d } _ { s } ( x _ { t } , t )\tag{3}
$$

where � controls the edit intensity: positive values steer towards the positive extreme ofthe attribute and negative values towards the negative extreme, enabling bidirectional editing along the semantic axis. $\bar { \mathbf { v } } ( x _ { t } , t )$ reconstructs the original object from its multi-view conditions, while �d $\mathbf { \Psi } _ { : } ( x _ { t } , t )$ steers the sampling trajectory toward the desired edit.

This steering is applied to both stages of TRELLIS’s pipeline, allowing the edit to afect both coarse geometry and fine appearance (DG1) (Fig 1e).

## 4.4 Mapping Latent Variations to the Slider

By adjusting the scaling factor �, we can produce diferent degrees of variation along the semantic attribute in the latent space. However, to build a usable slider-based editing experience, we need to translate these latent-space variations into visually meaningful and consistently spaced slider positions (DG2, DG3).

4.4.1 Determining Slider Bounds. To maximize the range of variation while ensuring that both ends of the slider maintain suficient generation quality, we adopt an adaptive bound search inspired by prior work on 2D sliders [38].

We first set empirical outer bounds ±� for �, determined via preliminary experiments to be large enough to cover the efective editing range for most object-attribute pairs, but not so large that the search wastes time evaluating clearly invalid generations (e.g., broken meshes). Starting from $\alpha = B$ and $\alpha = - B ,$ , we incrementally decrease |�| inward from both ends with a step size �. At each candidate �, we generate the corresponding 3D variation and evaluate it using a VLM (GPT-5.2), which checks for the presence of structural degradation or drastic appearance changes irrelevant to the target attribute across four views rendered at 90° horizontal intervals with 30° elevation (referred to as key-frames hereafter) (see Appendix A.1.3 for instruction prompt). The search stops in each direction at the first � that passes this quality check. This yields an adaptive range $[ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ]$ tailored to each object-attribute pair (DG2).

4.4.2 Perceptual Distance Estimation. Within the determined bounds $[ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ]$ , we sample � evenly spaced values of � and generate a 3D variation for each (Fig A.1a). To map these variations to perceptually meaningful slider positions, we measure the visual distance between each variation and the original object using LPIPS [79], a perceptual similarity metric closely aligned with human judgment, widely used to evaluate perceptual fidelity in 3D reconstruction pipelines [9, 33, 75].

A challenge unique to 3D slider-based editing is that the generation process does not guarantee consistent object orientation across diferent � values, meaning two variations at diferent intensities may face diferent directions. Comparing key-frames by rendering order would then measure diferences between mismatched viewpoints, which would inflate perceptual distance measurements with orientation diferences unrelated to the actual attribute change. We found that the orientation ofset between a variation and the original is almost always a multiple of 90°, so matching key-frames rendered every $9 0 ^ { \circ }$ sufices. We construct a $4 \times 4$ cost matrix of pairwise LPIPS distances between every key-frame $( \ S 4 . 4 . 1 )$ of the original object and the variation, then solve a minimum-cost bipartite matching using the Hungarian algorithm [42] to pair each original view with its closest-matching viewpoint in the variation. The signed perceptual distance for each variation is then:

$$
d ( \alpha _ { k } ) = \frac { 1 } { 4 } \mathrm { s g n } ( \alpha _ { k } ) \sum _ { j = 1 } ^ { 4 } \mathrm { L P I P S } ( f _ { j } ^ { \mathrm { o r i g } } , f _ { \sigma ( j ) } ^ { \alpha _ { k } } )\tag{4}
$$

where $f _ { j } ^ { \mathrm { o r i g } }$ is the �-th key-frame of the original object, $f _ { \sigma ( j ) } ^ { \alpha _ { k } }$ is the matched key-frame of the variation at $\alpha _ { k } .$ , and � is the optimal assignment from the Hungarian algorithm, and $s \mathrm { g n } ( \alpha _ { k } )$ assigns a positive or negative sign based on the editing direction.

4.4.3 Slider Position Mapping & Refinement. We adapt the perceptual distance-based slider mapping approach from AdaptiveSliders [38] to our 3D setting. We assign slider positions based on the signed perceptual distances: the most negative $d ( \alpha _ { k } )$ maps to −100% and the most positive to +100%, with intermediate positions assigned proportionally (Fig A.1b). This produces a mapping from � values to slider positions grounded in perceptual change rather than raw latent-space distance (DG3).

We then iteratively refine the slider by interpolating additional anchor points. Given a desired slider position (a percentage from - 100% to 100% ), we interpolate the corresponding � from the existing mapping, generate the variation, compute its perceptual distance, and insert it as a new anchor to improve subsequent interpolations (Fig A.1c). In practice, interpolated variations may not land precisely at the target percentage, as the latent space does not always support smooth linear interpolation of visual changes [24, 38]. Iterative refinement helps mitigate this.

Finally, we apply the same VLM-based quality check as in § 4.4.1 (Appendix A.1.3) to filter out visually invalid variations within the bounds. Unlike 2D image generation, where variations within a reasonable � range are typically valid, 3D generation can produce occasional artifacts (e.g., broken geometry). Invalid variations are removed from the final slider. We then search for the nearest valid replacement by incrementally shifting the slider position by 1% at a time until finding a variation that passes the quality check (DG1). Fig A.1d shows a comparison before and after slider position mapping on a “rustiness” slider created for a 3D airplane.

## 5 Technical Validation

To evaluate whether our technique satisfies the four design goals, we constructed a dataset of 50 object-attribute pairs spanning diverse 3D object categories and continuous semantic attributes, and conducted a validation experiment with five human assessors.

## 5.1 Validation Dataset

5.1.1 Object Categories. To ensure broad coverage, we selected 3D objects from categories commonly represented in established 3D object datasets [17, 27, 67], including human characters, animals, plants, furniture, transportation, electronics, food, and toys.

5.1.2 Atribute Categories. Drawing on attribute types used in prior work on 2D image sliders [28, 29, 38], we organized continuous attributes into two main categories: local attributes and overall characteristics.

Local attributes target specific parts of the object, including (1) spatial properties such as eye size or the height of a chair backrest, and (2) facial expressions such as degree of smile.

Overall characteristics afect the object as a whole, including (1) overall shape such as body shape, (2) style such as traditional vs. modern, (3) material properties such as surface roughness, (4) level of detail such as low-poly vs. photorealistic, and (5) temporal properties such as doneness of food or age of a person.

5.1.3 Dataset Construction. Two researchers collaboratively generated candidate object-attribute pairs from the above categories and reviewed each pair to ensure it represented a reasonable and editable continuous attribute for the given object. This process yielded 50 object-attribute pairs across 46 unique objects, with a few objects appearing in multiple pairs with diferent attributes. For each pair, we sourced 3D objects from CC-BY licensed assets in Objaverse [17] and from 3D-FUTURE [27]. We queried each dataset using keywords derived from the target object name based on available metadata, randomly retrieved 20 candidate objects per query, and manually selected the one that best matched the intended object with suficient geometric and appearance quality.

## 5.2 Apparatus

To assess how well SemanticSlider3D satisfies the four design goals, we compared it with a baseline as a reference. Since no existing method directly supports slider-based continuous semantic editing of 3D objects, we constructed a baseline by combining Concept

Sliders [28], a state-of-the-art 2D image slider, with SAM 3D [9], a state-of-the-art image-to-3D model, to represent the most straightforward of-the-shelf alternative: editing a 2D image along a slider and then lifting each variation to 3D independently. For both the baseline and SemanticSlider3D, we aimed to generate nine variations on the resulting slider, presenting the full spectrum of each target attribute.

5.2.1 SemanticSlider3D Setup. For each object-attribute pair, we rendered � = 40 camera views of the 3D object and selected � = 10 most relevant views for constructing contrastive image pairs, balancing generation speed and quality. We set the outer bound � = 5 (suficient to reach invalid generations for most object-attribute pairs), step size � = 0.25 (ensuring relatively fine granularity while keeping generation time reasonable), and guidance strength � = 3 (balancing condition adherence with steerability). In the slider mapping stage, we completed � = 2 rounds of refinement after sampling � = 9 initial � values. The slider variations corresponding to nine evenly spaced percentages from −100% to 100% were exported as GLB files. The entire pipeline was fully automatic and ran on a single A100 GPU.

5.2.2 Baseline: Concept Sliders + SAM 3D. Concept Sliders [28] train low-rank adaptors (LoRA) for each target attribute in a diffusion model’s weight space, learning a parameter direction that corresponds to the desired semantic concept. At inference time, the trained adaptor is loaded and a single scalar weight controls the intensity of the edit. For each object-attribute pair, we trained a Concept Slider on SDXL-Turbo [63], a fast text-to-image difusion model, using the same 10 contrastive image pairs (positive and negative) as in § 5.2.1.

Since Concept Sliders operate on a single input image, we used GPT-5.2 to select the most representative and informative camera view from all 40 rendered views. To generate variations from this input image, we applied Regularized Newton Inversion (GRNI) [62] to convert the image into the difusion model’s latent space. Following their inversion setup [62], we set the denoising strength to 0.6. We used a detailed caption of the 3D object as the inversion prompt, obtained using the captioning method described in [75].

We applied the same slider bound search and position mapping algorithm as SemanticSlider3D, with the outer bound set to � = 12, larger than the � = 8 used in prior work [38], to further ensure no valid variations at the extremes are missed. After finalizing the nine 2D slider variations, we segmented the object from the background using SAM 3 [6] with a basic object description as prompt, and lifted each segmented image to 3D using SAM 3D [9]. The resulting 3D objects were exported as GLB files. The entire baseline pipeline was fully automatic and ran on a single A100 GPU.

5.2.3 Evaluation Interface. We built a web-based evaluation interface (Fig A.2) that presents human assessors with one objectattribute pair at a time. The top left panel displays the reference 3D object and the target attribute. The bottom panel shows two rows of nine 3D variations generated by SemanticSlider3D and the baseline, labeled Slider A and Slider B. The mapping between methods and labels is randomized for each object-attribute pair so that assessors cannot identify which method they are evaluating. Assessors can rotate all objects simultaneously to inspect them from diferent angles, adjust the cell window size for closer inspection. The top right panel presents evaluation questions one at a time. For each object-attribute pair, assessors first rate Slider A on four criteria using a 7-point Likert scale: (1) range of variation, (2) quality of the 3D object, (3) variation consistency, and (4) preservation of irrelevant attributes. They then rate Slider B on the same four cri teria. On each question, the meaning of the two ends on the scale is labeled. Finally, they indicate an overall preference between the two methods (Slider A, Slider B, or Can’t Decide). In total, assessors answer nine questions per object-attribute pair before proceeding to the next.

## 5.3 Procedure

Five assessors participated in the technical validation, three from a software company and two from a local university. None of them had prior exposure to either method. Each assessor evaluated all 50 object-attribute pairs in a single session of 2 hours. Before beginning, assessors received an explanation of the task and four evaluation criteria (design goals) with image examples. To calibrate ratings, assessors first completed one practice trial, after which the researcher discussed their rationale and clarified any misunderstandings of the criteria. Assessors were also instructed to ignore minor color or lighting variations due to rendering conditions and monitor diferences.

## 5.4 Results

We report results across the four evaluation criteria corresponding to our design goals, as well as overall preference, breaking results down by the seven attribute categories. For each criterion, we report mean ratings for both methods across all 50 object-attribute pairs over the five assessors. Because a two-tailed Wilcoxon signed rank test cannot reach significance with five assessors $\left( \boldsymbol { p } \geq . 0 6 3 \right.$ by construction), we report descriptive statistics and inter-rater reliability (IRR). To assess IRR, we used $\mathrm { G w e t } ^ { \prime } s \mathrm { A C } _ { 1 }$ for the overall preference and $\mathrm { A C _ { 2 } }$ for the ordinal Likert-scale ratings, both robust to skewed rating distributions [32, 70]. For object-attribute pairs where a method received a rating of 2 or below for range of variation, we excluded that method’s variation consistency rating from analysis, as a near-zero range trivially implies smooth variation, making the consistency question meaningless. Fig A.3 details the Likert responses from the five assessors, and mean ratings are also summarized as a radar plot (Fig A.4).

5.4.1 Quality ofthe 3D Object (DG1). Overall, SemanticSlider3D received high ratings $( M e a n = 5 . 7 0 , S D = 0 . 3 8 )$ with almost perfect agreement across assessors $( A C _ { 2 } = 0 . 9 6 )$ , indicating that SemanticSlider3D produces high quality 3D Objects. The baseline was also rated positively $( M e a n = 4 . 6 6 , S D = 0 . 7 5 )$ , with substantial agreement $( A C _ { 2 } = 0 . 7 2 )$ . Our method scored highest on temporal $( M e a n = 5 . 9 8 , S D = 0 . 2 8 )$ and style attributes $( M e a n = 5 . 9 3 ,$ $S D = 0 . 4 3 )$ , and lowest on local spatial attributes $( M e a n = 5 . 1 8 ,$ $S D = 0 . 4 7 )$ , where fine-grained geometric edits occasionally introduced minor artifacts. Material attributes were rated towards the higher end for both methods (Ours: $M e a n = 5 . 6 2 , S D = 0 . 4 1 ;$ Baseline: $M e a n = 5 . 4 7 , S D = 0 . 4 8 )$

5.4.2 Range ofVariation (DG2). SemanticSlider3D received high ratings indicating wide range of variation $( M e a n = 6 . 6 0 , S D = 0 . 2 0 )$ with very high IRR $( A C _ { 2 } = 0 . 8 6 )$ , and across all categories, ranging from $6 . 3 6 ~ ( S D ~ = ~ 0 . 3 6 )$ on local spatial attributes to $6 . 8 0 \ ( S D \ =$ 0.17) on temporal attributes. Local spatial was the lowest-scoring category for our method, possibly because localized geometric changes (e.g., eye size, backrest height) are harder to steer in the latent space. Baseline ratings were on the lower end of the scale $( M e a n = 2 . 8 7 , S D = 0 . 3 2 )$ , though with low agreement $( A C _ { 2 } = 0 . 3 9 )$

5.4.3 Variation Consistency (DG3). SemanticSlider3D showed high variation consistency $( M e a n = 5 . 6 9 , S D = 0 . 2 5 )$ and almost perfect agreement among assessors $( A C _ { 2 } ~ = ~ 0 . 8 5 )$ . While the baseline’s mean was also high $( \mathrm { M e a n } { = } 5 . 2 2 , \mathrm { S D } { = } 1 . 0 7 )$ , ratings varied widely across assessors $( A C _ { 2 } ~ = ~ 0 . 5 4 )$ . Our method scored consistently high on material $( M e a n = 6 . 2 9 , S D = 0 . 3 2 )$ and temporal attributes $( M e a n = 6 . 2 0 , S D = 0 . 3 3 )$ . Assessors were moderately satisfied with overall shape $( M e a n = 4 . 3 6 , S D = 0 . 2 3 )$ . This may be because shape edits require coordinated geometric changes across many parts of the object, making it dificult to produce evenly spaced variations compared to appearance-dominant attributes like style or material, where the overall structure can remain stable while the surface properties change.

5.4.4 Preservation ofIrrelevant Atributes (DG4). SemanticSlider3D demonstrated high preservation of irrelevant attributes overall $( M e a n = 5 . 6 9 , S D = 0 . 4 3 )$ , with high IRR $( A C _ { 2 } = 0 . 7 7 )$ . The baseline received moderate ratings on this criterion $( M e a n = 4 . 2 0 , S D =$ 0.55) with low IRR $( A C _ { 2 } = 0 . 3 9 )$ . Our method scored high on temporal $( M e a n = 6 . 2 0 , S D = 0 . 4 1 )$ and style attributes $( M e a n = 6 . 2 0$ $S D = 0 . 5 8 )$ , and lower on local spatial attributes $( M e a n = 4 . 8 2 ,$ $S D = 0 . 8 0 )$ . The baseline showed low performance on facial expressions $( M e a n = 2 . 5 2 , S D = 1 . 0 0 )$ . This is because each step of the pipeline—image inversion, object segmentation, and 2D-to-3D conversion—can introduce fidelity loss, leading to unintended changes beyond the target expression. Local spatial attributes were the lower-scoring category for both methods (Baseline: $M e a n = 4 . 2 4$ $S D = 0 . 6 9 )$ , where both methods found it relatively challenging to fully isolate the edit from irrelevant features.

5.4.5 Overall Preference. All five assessors preferred Semantic-Slider3D over the baseline with substantial agreement $( A C _ { 1 } = 0 . 7 9 )$ with individual preference rates ranging from 72% to 96%, consistently across categories.

5.4.6 Summary. SemanticSlider3D’s absolute ratings were above 5.0 on all four criteria, suggesting that our method satisfies the four design goals as perceived by human assessors. Across attribute categories, our method performed strongest on temporal and style attributes, and was more limited on spatial attributes (both local and overall), where precise geometric changes are harder to steer smoothly in the latent space.

## 6 User Study

We conducted an exploratory user study with six participants who had prior experience with 3D tools. We aimed to understand how users would use SemanticSlider3D, and get their feedback on its usability and utility for 3D prototyping.

## 6.1 Apparatus

To this end, we built an interactive playground as a technology probe [37] using React [52] and Flask [30]. Users could generate 3D objects from text (TRELLIS text-to-3D) or image (SAM 3D), and create sliders using SemanticSlider3D. We used the same SemanticSlider3D setup as detailed in § 5.2.1, except with � = 4 (user-adjustable) initial samples and � = 1 refinement round to balance generation time and granularity. The interface (Fig 2) consists of the following components:

• 3D Viewport: The central workspace where users can view and manipulate 3D objects. Objects can be selected, repositioned, rotated, and scaled using on-screen gizmos or the Transform Panel (e).

• Image-to-3D (a): Users can upload a reference image (a1), seg ment a target object by clicking on it (via SAM 2 [61]), and lift the segmented object into a 3D object (a2).

• Text-to-3D (b): Users can type a text description (b1) to generate a 3D object (b2).

• Slider Panel (c): Users specify an object and a semantic attribute in text fields, then click “Create Slider” to generate the corresponding semantic slider with initial anchors points. Before generating, users can adjust how many anchor points to sample — more points produce a smoother slider but take longer to gen erate. The created slider displays variations along the semantic attribute, and users can move the slider handle to preview different variations (f). An “Add Point” feature allows users to request a new anchor point at a specified percentage. The system interpolates the corresponding variation from the existing slider mapping and generates it, progressively filling in the slider for finer exploration.

• Sidebar (d): Thumbnails of all 3D objects in the scene. Clicking a thumbnail selects the corresponding object in the viewport. Right-clicking provides options to hide/unhide.

Our playground was served on two A100 GPUs (four processes each), allowing at most two sliders to be built concurrently. Across all sliders built during the study sessions (§ 6.3), build time under default setting (� = 4, � = 1) averaged 726.3s (�� = 183.2). It varied across object-attribute pairs because failed quality checks required additional generations and mesh complexity afected per-variation generation time (43s at ∼1.2k triangles to 84s at ∼50k triangles). Table A.1 reports the full per-stage time breakdown for our 3D editing pipeline.

## 6.2 Participants

We recruited six participants (two male, four female, ages 22–31, ���� = 26, �� = 3.16) with prior exposure to 3D prototyping from two universities. Four were graduate students in computer science and two in mechanical engineering. P6 also had a background as a sculpture and animation artist. Participants had varying levels of 3D prototyping experience, ranging from 1 month to 10 years, using tools such as Autodesk Fusion, SolidWorks, Blender. In addition, P1, P2 and P3 had prior exposure to online 3D generation services such as Meshy AI <sup>1</sup>. Each participant received a \$25/hr compensation via Amazon gift cards. Participants’ demographic information and 3D prototyping experience are detailed in Table A.2

## 6.3 Study Design

6.3.1 Procedure. We conducted in-person study sessions, each lasting about 150 minutes. The study consisted of three phases.

In the introduction phase, the researcher explained the purpose of the study and walked participants through the interface, which was preloaded with six objects and sliders (e.g., a car with a head light size slider, a deer with a realisticness slider) to familiarize participants with the range of attributes the system supports. The walkthrough covered how to create sliders by specifying an object and a semantic attribute, preview variations by moving the slider, and add new anchor points. Participants then had 5 minutes to freely explore before proceeding.

In the task phase, participants completed two tasks. In Task 1, participants were asked to think of a 3D prototyping scenario with a specific goal in mind, then create two 3D objects (via text-to-3D or image-to-3D) and a slider for each. In Task 2, participants created two additional sliders based on existing objects in the scene, either by selecting the original object or by copying an anchor point variation as a new starting point. Once built, participants could browse variations without latency; adding a new anchor point took about 90s. Participants were asked to think aloud throughout. After each slider creation, the researcher asked about their subjective experience, including whether results aligned with expectations and their next steps toward their goal. During generation wait times, the researcher collected demographic information and participants’ prior experience with 3D prototyping and generative AI tools.

In the exit interview, participants responded to questions adapted from the Creativity Support Index [13] and slider-specific questions on the usefulness of diferent slider components. All items were phrased as agreement statements (e.g., “The system helped me fully explore the space of potential designs”) and rated on a 7-point Lik ert scale from 1 (strongly disagree) to 7 (strongly agree). We also collected general feedback about their strategies during 3D prototyping with SemanticSlider3D and suggestions for improvement.

6.3.2 Data Collection and Analysis. We transcribed all recordings and analyzed the transcripts using thematic analysis [4]. The first author coded all transcripts using open coding, and then iteratively generated and refined themes while reviewing participants’ artifacts. The resulting codes and themes were reviewed and discussed with the research team.

## 6.4 Findings

We detailed participants’ individual 3D prototyping tasks and goals in Appendix A.4. In this section, we report participants’ experience with our technology probe via qualitative feedback and Likert-scale responses.

6.4.1 Slider Variations for Exploration. Participants broadly appreciated the slider’s ability to generate diverse design variations along a semantic attribute, all from a single object and prompt specification (P1, P2, P3, P6). The slider was rated favorably for facilitating diverse exploration (���� = 6.17, �� = 1.17) and for supporting decisions on what other design variations to try (���� = 6.17, �� = 0.75). Participants also found that slider variations can bring unexpected inspiration to their prototyping (P1, P2, P3, P6). P3 originally wanted a very modern and futuristic floor lamp, but ended up being attracted by a retro variation (Fig 3): “because it (retro style lamp) has much more features or much more designs, instead ofjust a very simple hood (a modern variation).” P2 additionally commented that existing slider variations could trigger new exploration directions, while brainstorming the design of a prosthetic leg, “I was already just thinking about the style... but after I see diferent options provided by the (futuristic style) slider and now I think ‘oh there could be more things added to it,’ so now it inspired me to think about the length and authenticity but the slider gave the trigger.” Despite these merits, two participants (P3, P6) wanted to explore beyond the current 100% endpoint. For example, P6 would like to see more abstract versions of her sculpture (Fig 4) beyond the allowed range, suggesting that users should be granted more authority over what is considered out-of-range for attributes like artistic styles.

![](images/a7a91a89d377c1650139a92585773be33c95edf58626070e826f98bc0bc75703.jpg)  
Figure 2: The interactive playground interface used in our study. Users generate 3D objects via image-to-3D (a: upload and segment a reference image (a1) to produce a 3D object (a2)) or text-to-3D (b: type a description (b1) to generate a 3D object (b2)) The Slider Panel (c) allows users to specify an object and semantic attributes, create a slider, and add new anchor points at desired percentages; as the user moves the slider handle on created sliders, the corresponding variation is displayed in-place in the viewport (f). The Sidebar (d) displays thumbnails of all objects in the scene. The Transform Panel (e) provides controls for positioning, rotating, and scaling selected objects.

Participants also commented on the granularity of the slider’s variations. P4 valued the slider’s ability to show intermediate states that would be dificult to obtain through discrete prompting with current GenAI tools, such as seeing the full spectrum of color as an apple progresses through ripeness. However, three participants (P1, P3, P5) noted that the percentage placement of anchor points was sometimes unpredictable. This is likely due to the nonlin earity of the latent space (§ 4.4.3). P1 commented, “specifying where on the slider that (the new point) ends up going is dificult.” These observations suggest that while the slider was efective at producing diverse variations, the perceived continuity did not always meet expectations.

6.4.2 Expressing Design Intents via Sliders. While the slider facilitated diverse exploration, participants also reflected on how the text-based attribute specification shaped their ability to express design intent. Overall, participants reported being able to be ex pressive (���� = 5.67, �� = 1.03). We observed that for vague attributes (e.g., "futuristic looking"), the LLM converged on a single concrete interpretation, whereas for specific attributes, the generated prompts closely followed the user’s wording. In our study, four participants (P1, P4-P6) moved from broad concepts in Task 1 to more specific attributes in Task 2, producing outputs better matching their intent.

![](images/6aa2dd88184c70e1cb6550415b4da2e6705ee4ff007cba6cd7e979e943dab33c.jpg)  
Figure 3: P3 created a (a) “futuristic style” slider for a floor lamb generated from an image, and chose the -98% variation to further refine with a (b) “umbrella-shaped hood” slider and chose 74% variation to further edit with a (c) “pattern abstractness” slider to get a desired pattern. Green % = input object’s slider position (same in later figures).

Seeing slider results helped concretize and clarify participants’ mental images of the attributes they wished to edit (P1, P2, P4, P5). For example, P5 simply typed a single word “strong”

![](images/4bbabfd5fe935b2350dec94b3ec20766a14d0f3e78a898f69db60baba189da56.jpg)  
Figure 4: P6 created an (a) “abstractness” slider on an object generated from an image (a man hugging a ball) sourced online, with the goal of designing a sculpture with the same gesture. She then chose a variation at 42% and further explored material use with a (b) “hard vs. soft” slider where she found a good design at -100%.

in the attribute text box for a deer object, hoping to see a range of body shapes without a concrete idea of what she was looking for. However, after viewing the full spectrum, especially seeing the very skinny deer on the negative side (Fig 5b), she realized the attribute she actually wanted to adjust was maturity (age) rather than muscle mass. This suggests that the slider serves not only as an editing tool but also as a mechanism for users to calibrate their own understanding of their design intent.

![](images/2b3cd5373711c37441bd01fbd5cc6837871bdd45cf89cd44ca866274a0e9c53b.jpg)  
Figure 5: (a) P1: no entrance for bartender at 96% on a “clutteredness” slider for a bar counter; (b) P5: skinny deer at -84% on a “strong” slider.

Beyond this reflective use, participants also identified oppor tunities to enrich the attribute specification process. P1 and P5 wanted system-provided guidance on prompt refinement, and P2 wanted attribute suggestions given a 3D object, similar to features in existing creativity support tools [38, 51]. P2 and P6 sug gested using images to define attributes as an alternative to text, because this is more natural for “visual people.” Three participants (P1, P3, P6) further expressed the need for defining intermediate anchor points to allow mixing concepts, similar to Prompt-Paint’s prompt mixing [15]. P6 envisioned a material slider from water to stone: “let’s say water to stone. Maybe then the middle can be ocean or shoreline, like where the water and land meet... And then the sculpture can be something like... halfwater and... halfsand... that’s like in between.”

6.4.3 Integration into Current Workflow. In general, participants were satisfied with the artifact they created: P5 and P2 wanted to use outputs directly for 3D printing and adding variations to a VR asset library, respectively; P3 and P6 began considering physical manufacturing processes; and P1 indicated his prototypes were ready for mesh-level refinement in 3D authoring tools.

Participants discussed how the slider-based technique could fit within their existing 3D prototyping practices. All participants considered the slider useful for rapid 3D prototyping (���� = 6.33, �� = 1.21) and as a valuable addition to their current workflows (���� = 6.00, �� = 1.26).

P3, an experienced mechanical engineer, explained how our technique can support rapid prototyping: “with one prompt, Iwas able to get a slider that have multiple designs on that. But if I do text-to-image, I would have to do multiple prompting to get diferent type of images then multiple prompts to convert that image to 3D. So it’s a lengthy process... The reason we don’t do 3D-initial 3D prototyping is because the time it takes to [generate] prototype in 3D. So this slider-based tool reduce that time by a lot.” Beyond speed, P1 appreciated that the slider could quantify subjective attributes by percentages, giving designers a more concrete estimate of the amount of edit before reaching their goal, which is beneficial to the prototyping process regardless of whether slider-based editing is used downstream.

While promising for improving current workflows, some participants highlighted practical considerations around functional feasibility. For example, P1 noticed that some bar counter variations lacked an entrance for bartenders (Fig 5a), requiring further manual adjustment. However, P3 argued that such structural constraints are better addressed with conventional CAD tools, while the slider should focus on discovering visual attribute variations. This complementary view was echoed by participants’ experiences with spatial attributes (P2, P5): both noted quality degradation for fine-grained geometric edits (e.g., gaze direction), but considered this acceptable given that conventional 3D authoring tools handle such adjustments more reliably.

## 7 Discussion

In this section, we reflect on findings from both the technical validation and the user study. We discuss the unique challenges of 3D continuous editing, how the slider interaction shapes human-AI co-creation during 3D prototyping, practical tradeofs between latency and slider granularity, and tensions between system-imposed constraints and user autonomy.

## 7.1 3D Editing via 2D vs. Direct Latent Steering

Simply applying existing 2D image sliders and lifting the results to 3D seems like an intuitive solution, but our technical validation suggests that operating directly in the 3D latent space, as SemanticSlider3D does, is more efective. The baseline (Concept Sliders + SAM 3D) consistently struggled to produce meaningful variation, and its multi-step pipeline (image inversion, 2D editing, segmentation, 2D-to-3D conversion) introduced compounding fidelity loss that degraded both quality and attribute preservation, aligning with prior findings that 2D-assisted 3D generation often yields lower quality due to multi-view inconsistency [66, 75].

By contrast, SemanticSlider3D steers the generation directly in the 3D latent space, where geometry and appearance are jointly represented, avoiding the information loss of 2D-3D projection and naturally preserving multi-view consistency. However, spatial attributes (local and overall) remain challenging, as they require precise geometric changes that are hard to control through latent steering. A user-specified 3D structure prior indicating the desired geometric change at each slider point, inspired by [54], could potentially enable more controllable geometric edits. More broadly, our latent steering approach could generalize to other 3D generation pipelines that support multi-view contrastive conditioning and a flow or difusion sampler.

## 7.2 Slider-Based Human-AI Co-Creation

Our exploratory user study uncovered the potential of Semantic-Slider3D for human-AI co-creation during the prototyping process. Participants reported that slider variations brought unexpected inspiration and triggered new exploration directions that they had not initially considered. This finding echoes prior work on how generative AI can expand the design space by surfacing possibilities beyond what users would envision on their own [18, 51].

However, the current interaction between the slider and the user is largely one-directional: the system generates variations and the user evaluates them. As noted in our findings, anchor point placement did not always match participants’ expectations, primarily due to the nonlinearity or sparsity in the latent space which can’t be fixed perfectly using LPIPS normalization techniques alone if the extent of the problem is too large. One promising direction to address this is to allow users to directly modify 3D variations on the slider—for instance, manually adjusting a variation to indicate what a desired percentage on the semantic axis should look like. This would turn the slider into a bidirectional interaction, where the user’s manual edits inform the system’s understanding of the attribute and help calibrate subsequent generations. Such a human-in-the-loop approach could help the user and the model converge on a shared understanding of the editing direction more eficiently, echoing prior work [22].

## 7.3 Latency and Slider Granularity

A practical tension in SemanticSlider3D is between slider granularity and generation time. Each slider took approximately 12 minutes to generate initial anchors, and each additional anchor point requires another 90s (Table A.1). Although smaller than the pretraining requirements for existing 2D LoRA based approaches, this latency limited the number of anchor points we could provide during the study, which could afect the perceived smoothness of the slider.

This tradeof between granularity and latency is not unique to our work. Dreamcrafter addressed a similar challenge by introducing 2D proxy representations for high-latency 3D generation and editing [71]. A similar strategy could benefit SemanticSlider3D: lightweight 2D previews could give users immediate feedback on the editing direction while the full 3D variations generate in the background. Additionally, directly viewing Gaussian Splats could reduce latency by skipping mesh conversion (exporting to GLB), and running on more resource-intensive compute will make generation faster, enabling denser sampling and smoother sliders. In the meantime, future work could explore progressive refinement strategies that prioritize generating anchor points in regions where the user is actively exploring.

## 7.4 Transparency and Freedom of Control

Our study surfaced tensions between system automation and user awareness. Under the hood, SemanticSlider3D augments the user’s attribute prompt via AI models to generate contrastive editing prompts, selects edit-relevant views, and determines slider bounds— all without user visibility. While this automation reduces user burdens, it also means the system is making interpretive decisions that can silently deviate from the user’s original intent. Making these intermediate steps transparent, such as showing users the generated contrastive prompts and the selected views before generating the slider, could help users verify that the system’s interpretation matches their intent and intervene early when it does not.

Beyond transparency, participants expressed a desire for greater freedom of control. As reported in our findings, participants wanted to explore beyond predetermined bounds (P3, P6), use images to define attributes (P2, P6), and mix concepts at intermediate anchor points (P1, P3, P6). These suggestions all indicate a need for more control over the editing process. This would allow users to calibrate the tool to their own creative standards and context while still benefiting from the system’s default setting as a starting point.

## 7.5 Limitations & Future Work

Our work has limitations which should be looked at in future work. First, the VLM-based quality checks (GPT-5.2) are not always reliable. Sometimes structurally valid variations were rejected, while variations with visible artifacts occasionally passed. Allowing users to override or review flagged variations could help mitigate this. Future work should also investigate more reliable quality assessment methods by incorporating 3D structural information rather than relying solely on 2D rendered views. Second, our validation dataset focused on common everyday objects with limited coverage of artistic or abstract attributes. Future work should expand the dataset to better evaluate the technique’s applicability in creative and expressive contexts. We will open source our dataset to push this work forward. Lastly, our technical validation involved a small number of assessors. A larger pool would enable statistically significant comparisons with the baseline.

## 8 Conclusion

We presented SemanticSlider3D, a training-free technique for continuous, slider-based semantic editing of 3D objects. Our technical validation showed that SemanticSlider3D produces diverse, consistent, and high quality variations without altering irrelevant attributes, and was overwhelmingly preferred over a baseline. Our exploratory user study further showed that the technique facilitated exploration of design alternatives and was perceived as a useful addition to 3D prototyping workflows.

## References

[1] Panos Achlioptas, Ian Huang, Minhyuk Sung, Sergey Tulyakov, and Leonidas Guibas. 2023. ShapeTalk: A language dataset and framework for 3d shape edits and deformations. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 12685–12694.

[2] Amir Barda, Matheus Gadelha, Vladimir G Kim, Noam Aigerman, Amit H Bermano, and Thibault Groueix. 2025. Instant3dit: Multiview inpainting for fast editing of 3d objects. In Proceedings of the Computer Vision and Pattern Recognition Conference. 16273–16282.

[3] M Sanem Bayar and Zeeshan Aziz. 2018. Rapid prototyping and its role in supporting architectural design process. Journal of Architectural Engineering 24, 3 (2018), 05018003.

[4] Virginia Braun and Victoria Clarke. 2006. Using thematic analysis in psychology. Qualitative research in psychology 3, 2 (2006), 77–101.

[5] Bradley Camburn, Vimal Viswanathan, Julie Linsey, David Anderson, Daniel Jensen, Richard Crawford, Kevin Otto, and Kristin Wood. 2017. Design prototyping methods: state of the art in strategies, techniques, and guidelines. Design Science 3 (2017), e13.

[6] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025).

[7] Hansheng Chen, Ruoxi Shi, Yulin Liu, Bokui Shen, Jiayuan Gu, Gordon Wetzstein, Hao Su, and Leonidas Guibas. 2024. Generic 3d difusion adapter using controlled multi-view editing. arXiv preprint arXiv:2403.12032 (2024).

[8] Minghao Chen, Junyu Xie, Iro Laina, and Andrea Vedaldi. 2024. Shap-editor: Instruction-guided latent 3d editing in seconds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 26456–26466.

[9] Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. 2026. Sam 3d: 3dfy anything in images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7220–7232.

[10] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. 2024. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21476–21485.

[11] Yuzhao Chen, Runlin Duan, Rahul Jain, Yichen Hu, Chenfei Zhu, Jingyu Shi, and Karthik Ramani. 2026. Canvas3D: Empowering Precise Spatial Control for Image Generation with Constraints from a 3D Virtual Canvas. In Proceedings ofthe 31st International Conference on Intelligent User Interfaces (IUI ’26). Association for Computing Machinery, New York, NY, USA, 1362–1386. doi:10.1145/3742413. 3789142

[12] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al. 2024. Meshanything: Artist-created mesh generation with autoregressive transformers. arXiv preprint arXiv:2406.10163 (2024).

[13] Erin Cherry and Celine Latulipe. 2014. Quantifying the creativity support of digital tools through the creativity support index. ACM Transactions on Computer-Human Interaction (TOCHI) 21, 4 (2014), 1–25.

[14] Jiin Choi, Seung Won Lee, and Kyung Hoon Hyun. 2025. GenPara: Enhancing the 3D Design Editing Process by Inferring Users’ Regions of Interest with Text Conditional Shape Parameters. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. 1–21.

[15] John Joon Young Chung and Eytan Adar. 2023. Promptpaint: Steering text-to image generation through paint medium-like interactions. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology. 1–17.

[16] Hai Dang, Lukas Mecke, and Daniel Buschek. 2022. Ganslider: How users control generative models for images using multiple sliders with and without feedforward information. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems. 1–15.

[17] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2022. Objaverse: A Universe of Annotated 3D Objects. arXiv preprint arXiv:2212.08051 (2022).

[18] Runlin Duan, Chenfei Zhu, Yuzhao Chen, Yichen Hu, Jingyu Shi, and Karthik Ramani. 2025. DesignFromX: Empowering consumer-driven design space exploration through feature composition of referenced products. In Proceedings ofthe 2025 ACM Designing Interactive Systems Conference. 1040–1060.

[19] Kristen M Edwards, Brandon Man, and Faez Ahmed. 2024. Sketch2Prototype: rapid conceptual design exploration and prototyping with generative AI. Proceedings ofthe Design Society 4 (2024), 1989–1998.

[20] Yigit Ekin and Yossi Gandelsman. 2026. The Unreasonable Efectiveness of Text Embedding Interpolation for Continuous Image Steering. arXiv preprint arXiv:2603.17998 (2026).

[21] Ziya Erkoç, Can Gümeli, Chaoyang Wang, Matthias Nießner, Angela Dai, Peter Wonka, Hsin-Ying Lee, and Peiye Zhuang. 2025. Preditor3d: Fast and precise 3d shape editing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and

Pattern Recognition. 640–649.

[22] Noyan Evirgen and Xiang’Anthony’ Chen. 2022. Ganzilla: User-driven direction discovery in generative adversarial networks. In Proceedings ofthe 35th Annual ACM Symposium on User Interface Software and Technology. 1–10.

[23] Noyan Evirgen and Xiang’Anthony Chen. 2023. Ganravel: User-driven direction disentanglement in generative adversarial networks. In Proceedings ofthe 2023 CHI Conference on Human Factors in Computing Systems. 1–15.

[24] Rotem Ezra, Hedi Zisling, Nimrod Berman, Ilan Naiman, Alexey Gorkor, Liran Nochumsohn, Eliya Nachmani, and Omri Azencot. 2025. FreeSliders: Training Free, Modality-Agnostic Concept Sliders for Fine-Grained Difusion Control in Images, Audio, and Video. arXiv preprint arXiv:2511.00103 (2025).

[25] Faraz Faruqi, Ahmed Katary, Tarik Hasic, Amira Abdel-Rahman, Nayeemur Rahman, Leandra Tejedor, Mackenzie Leake, Megan Hofmann, and Stefanie Mueller. 2023. Style2Fab: functionality-aware segmentation for fabricating personalized 3D models with generative AI. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology. 1–13.

[26] Lin Geng Foo, Yixuan He, Ajmal Saeed Mian, Hossein Rahmani, Jun Liu, and Christian Theobalt. 2024. Avatar Concept Slider: Controllable Editing of Concepts in 3D Human Avatars. arXiv preprint arXiv:2408.13995 (2024).

[27] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 2021. 3d-future: 3d furniture shape with texture. International Journal ofComputer Vision 129, 12 (2021), 3313–3337.

[28] Rohit Gandikota, Joanna Materzyńska, Tingrui Zhou, Antonio Torralba, and David Bau. 2024. Concept sliders: Lora adaptors for precise control in difusion models. In European Conference on Computer Vision. Springer, 172–188.

[29] Rohit Gandikota, Zongze Wu, Richard Zhang, David Bau, Eli Shechtman, and Nick Kolkin. 2025. Sliderspace: Decomposing the visual capabilities of difusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 15994–16003.

[30] Miguel Grinberg. 2018. Flask-SocketIO documentation. Available online at: https://flask-socketio.readthedocs.io/en/latest/, last accessed on 9/10/2023.

[31] Julia Guerrero-Viu, Milos Hasan, Arthur Roullier, Midhun Harikumar, Yiwei Hu, Paul Guerrero, Diego Gutierrez, Belen Masia, and Valentin Deschaintre. 2024. Texsliders: Difusion-based texture editing in clip space. In ACM SIGGRAPH 2024 conference papers. 1–11.

[32] Kilem L Gwet. 2014. Handbook of inter-rater reliability: The definitive guide to measuring the extent ofagreement among raters. Advanced Analytics, LLC.

[33] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. 2023. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF international conference on computer vision. 19740– 19750.

[34] Jonathan Ho and Tim Salimans. 2022. Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022).

[35] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023)

[36] Georgi Hristov and Diyana Kinaneva. 2021. A workflow for developing game assets for video games. In 2021 3rd International Congress on Human-Computer Interaction, Optimization and Robotic Applications (HORA). IEEE, 1–5.

[37] Hilary Hutchinson, Wendy Mackay, Bo Westerlund, Benjamin B Bederson, Allison Druin, Catherine Plaisant, Michel Beaudouin-Lafon, Stéphane Conversy, Helen Evans, Heiko Hansen, et al. 2003. Technology probes: inspiring design for and with families. In Proceedings ofthe SIGCHI conference on Human factors in computing systems. 17–24.

[38] Rahul Jain, Amit Goel, Koichiro Niinuma, and Aakar Gupta. 2025. AdaptiveSliders: User-aligned Semantic Slider-based Editing of Text-to-Image Model Output. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems. 1–27.

[39] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. 2024. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIGGRAPH 2024 conference papers. 1–1.

[40] Heewoo Jun and Alex Nichol. 2023. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023).

[41] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

[42] Harold W Kuhn. 1955. The Hungarian method for the assignment problem. Naval research logistics quarterly 2, 1-2 (1955), 83–97.

[43] Jaewook Lee, Filippo Aleotti, Diego Mazala, Guillermo Garcia-Hernando, Sara Vicente, Oliver James Johnston, Isabel Kraus-Liang, Jakub Powierza, Donghoon Shin, Jon E Froehlich, et al. 2025. Imaginatear: Ai-assisted in-situ authoring in augmented reality. In Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology. 1–21.

[44] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. 2023. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214 (2023).

[45] Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, and Lu Sheng. 2025. Voxhammer: Training-free precise and coherent 3d editing in native 3d space. arXiv preprint arXiv:2508.19247 (2025).

[46] Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, et al. 2025. Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747 (2025).

[47] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. 2025. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[48] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. 2024. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 6517–6526.

[49] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3d: High-resolution text-to-3d content creation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 300–309.

[50] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022).

[51] Vivian Liu, Jo Vermeulen, George Fitzmaurice, and Justin Matejka. 2023. 3DALL-E: Integrating text-to-image AI in 3D design workflows. In Proceedings ofthe 2023 ACM designing interactive systems conference. 1955–1977.

[52] Meta. 2022. React - A JavaScript library for building user interfaces. Available online at: https://reactjs.org, last accessed on 9/7/2022.

[53] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

[54] Phillip Mueller, Talip Uenlue, Sebastian Schmidt, Marcel Kollovieh, Jiajie Fan, Stephan Günnemann, and Lars Mikelsons. 2025. GeoDifusion: A Training-Free Framework for Accurate 3D Geometric Conditioning in Image Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 6374– 6384.

[55] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).

[56] Francesco Palandra, Andrea Sanchietti, Daniele Baieri, and Emanuele Rodola. 2024. Gsedit: Eficient text-guided editing of 3d objects via gaussian splatting. arXiv preprint arXiv:2403.05154 (2024).

[57] Maria Parelli, Michael Oechsle, Michael Niemeyer, Federico Tombari, and Andreas Geiger. 2026. 3d-latte: Latent space 3d editing from textual instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14377–14386.

[58] Jongsu Park, Shokhikha Amalana Murdivien, Kyungwan Choi, Duhwan Mun, and Jumyung Um. 2026. Generative 3D appearance design: A survey of generation, segmentation and editing by artificial intelligence. Journal ofComputational Design and Engineering 13, 1 (2026), 1–23.

[59] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022).

[60] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10318–10327.

[61] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2025. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, Vol. 2025. 28085–28128.

[62] Dvir Samuel, Barak Meiri, Nir Darshan, Shai Avidan, Gal Chechik, and Rami Ben-Ari. 2024. Regularized Newton Raphson Inversion for Text-to-Image Difusion Models.

[63] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. 2024. Adversarial difusion distillation. In European Conference on Computer Vision. Springer, 87–103.

[64] Etai Sella, Gal Fiebelman, Peter Hedman, and Hadar Averbuch-Elor. 2023. Vox-e: Text-guided voxel editing of 3d objects. In Proceedings of the IEEE/CVF international conference on computer vision. 430–440

[65] Yulin Shen, Boyu Li, Jiayang Huang, David Yip, and Zeyu Wang. 2025. Gaussian-ShopVR: Facilitating immersive 3D authoring using Gaussian splatting in VR. In Proceedings ofthe 38th Annual ACM Symposium on User Interface Software and Technology. 1–14.

[66] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023. Mvdream: Multi-view difusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023).

[67] Stefan Stojanov, Anh Thai, and James M Rehg. 2021. Using shape to categorize: Low-shot learning with an explicit shape bias. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 1798–1808.

[68] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. 2024. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision. Springer, 1–18.

[69] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2023. Dreamgaussian: Generative gaussian splatting for eficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023).

[70] Jennifer Tjia, Chengwu Yang, Julie Flahive, Kelly Harrison, Geraldine Puerto, Vennesa Duodu, Lisa A Cooper, Olga Valdman, and Janice Sabin. 2025. A Novel Communication Rating Scale to Mitigate the Efect of Implicit Bias. JAMA Network Open 8, 9 (2025), e2532319.

[71] Cyrus Vachha, Yixiao Kang, Zach Dive, Ashwat Chidambaram, Anik Gupta, Eunice Jun, and Björn Hartmann. 2025. Dreamcrafter: Immersive editing of 3d radiance fields through flexible, generative inputs and outputs. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. 1–13.

[72] Yuanhao Wang, Cheng Zhang, Gonçalo Frazão, Jinlong Yang, Alexandru-Eugen Ichim, Thabo Beeler, and Fernando De la Torre. 2025. GarmentCrafter: Progressive Novel View Synthesis for Single-View 3D Garment Reconstruction and Editing. arXiv preprint arXiv:2503.08678 (2025).

[73] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems 36 (2023), 8406–8441.

[74] Tien-Tsin Wong, Wai-Shing Luk, and Pheng-Ann Heng. 1997. Sampling with Hammersley and Halton points. Journal ofgraphics tools 2, 2 (1997), 9–24.

[75] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025. Structured 3d latents for scalable and versatile 3d generation. In Proceedings ofthe IEEE/CVFconference on computer vision and pattern recognition. 21469–21480.

[76] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. 2024. Instantmesh: Eficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 (2024).

[77] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. Imagereward: Learning and evaluating human prefer ences for text-to-image generation. Advances in Neural Information Processing Systems 36 (2023), 15903–15935.

[78] Emilie Yu, Rahul Arora, J Andreas Baerentzen, Karan Singh, and Adrien Bousseau. 2022. Piecewise-smooth surface fitting onto unstructured 3D sketches. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–16.

[79] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In CVPR.

## A Appendix

## A.1 System Prompts

A.1.1 Contrastive Editing Prompt Generation. The following system prompt is used with GPT-5.2 to generate contrastive editing prompts for a given object-attribute pair:

You are an expert assistant specializing in generating concise, high-impact prompts for image editing APIs. Given an “Object A” and an “Attribute B”, your task is to generate a “negative\_prompt” and a “positive\_prompt” that modify the object along the spectrum of that attribute.

Instructions:

(1) Positive Prompt: Edit Object A to present the MAXI-MUM extreme or highest intensity of Attribute B.

(2) Negative Prompt: Edit Object A to present the MINI-MUM extreme or exact opposite of Attribute B.

(3) Sentiment Independence: Note that Attribute B can itself be inherently negative (e.g., “rusty”), neutral, or positive. The “positive prompt” always means MORE of the attribute, and the “negative prompt” always means LESS of it.

(4) Brevity (Critical): Image models lose focus with long prompts. Keep the editing description concise and direct (under 20 words).

Safety Guardrails (Critical):

• NEVER generate prompts that could trigger NSFW, nudity, or sexual content safety filters.

• When reducing clothing, use safe, concise descriptors like “casual summer wear” or “lightweight clothing”. Avoid “skimpy”, “revealing”, “bare”, or “underwear”.

• Modifying physical traits (“very thin”, “highly muscular”) is perfectly safe, but keep it purely descriptive and nonsuggestive.

Examples:

Object: “a blue sofa” | Attribute: “modern style”

Negative: “Make the sofa traditional, antique, and ornate style.”

Positive: “Make the sofa ultra-modern, sleek, and minimalist style.”

Object: “a woman” | Attribute: “eye size”

Negative: “Make the woman’s eyes extremely small and narrow.”

Positive: “Make the woman’s eyes dramatically large and wide open.”

Object: “a woman” | Attribute: “wearing very warm and thick clothes”

Negative: “Make the woman wear modest, casual summer clothes like a t-shirt.”

Positive: “Make the woman wear extremely warm, heavy winter coats and thick scarves.”

Output Format:

Return ONLY a valid JSON object matching this schema without any markdown formatting:

{“prompts”: [“<negative\_prompt>”, “<positive\_prompt>”]}

A.1.2 View Relevance Verification. The following system prompt is used with GPT-5.2 to verify whether a rendered camera view contains the region targeted by the editing prompt:

You are a view-relevance checker for 3D object editing. You will be given an editing prompt describing a desired change to a 3D object, and an image showing the object rendered from a particular viewpoint.

Your task is to determine whether the region or part of the object targeted by the editing prompt is clearly visible in this view.

Guidelines:

• If the edit targets a specific part (e.g., “make the eyes bigger”), check whether that part is visible and not occluded or facing away from the camera.

• If the edit targets the entire object (e.g., “change the color/style of the object”), check whether the object is suficiently visible in the image.

Output Format:

Return {“result”: true} if the targeted region is clearly visible, or {“result”: false} otherwise.

A.1.3 3D Variation Quality Check. The following system prompt is used with GPT-5.2 to evaluate whether a generated 3D variation maintains structural integrity and perceptual plausibility:

You are a sophisticated Vision-Language Model specializing in the perceptual evaluation of 3D asset renders.

Context:

• Input 1: An image containing 4 camera views (quad-view) of the “Original Object.”

• Input 2: An image containing 4 camera views (quad-view) of the “Edited Object.”

• Input 3: An “Editing Prompt” describing the intended transformation.

Your Task: Analyze all 4 views to determine if the editing process caused “unreasonable” issues. You must distinguish between acceptable stylistic interpretations and unacceptable structural failures.

1. Reasonable & Prompt-Related Variations (Be Lenient):

• If the prompted change is dificult to see, very subtle, or seems missing entirely (e.g., the prompt asked for a “bigger head” but it looks the same), it is completely fine. Do not penalize for a lack of obvious change.

• Forgive unprompted changes that are indirectly related to the prompt or follow its aesthetic theme.

• Forgive misinterpretations where a change is visually similar to the prompt (e.g., if asked for “extremely large eyes” and they are rendered so big they resemble sunglasses, this is an acceptable stylistic interpretation).

• Minor texture shifts, lighting variations, or incidental color bleeds in the general area of the edit are completely acceptable.

2. Perceptual Sanity & Structural Integrity (Be Strict):

• Apply “The Weird Test”: Does the edit result in a look that is “unreasonable” or “illogical” for this type of object compared to the original?

• Failure: Substantial unprompted changes that make no sense (e.g., a hole appearing in a car roof when only the tires were edited, or a door melting into the body).

• Failure: Structural degradation like heavily melted textures, floating or disconnected geometry, broken mesh patches, or distorted “spikes” that look like generative glitches rather than intentional design.

Evaluation Rules:

• Ignore background: Do not evaluate anything outside the foreground object.

• Multi-view consistency: If a “weird” or “broken” failure is visible in any of the 4 views, the asset fails.

• Logic over isolation: Do not fail an image just because it changed more than requested or the edit is too subtle. Only fail it if the unprompted change is illogical, broken, or perceptually “wrong.”

## Instructions:

(1) Identify the intended change based on the Editing Prompt.

(2) Scan all 4 views of the Edited Object. Compare them to the Original Object’s structural logic.

(3) Determine if any unprompted changes are “reasonable interpretations” (pass) or “illogical glitches” (fail).

(4) If the prompted change is not obvious, focus entirely on whether the rest of the object remained stable and logical.

(5) Return ONLY a valid JSON object without markdown formatting.

## Output Format:

{   
“evaluation\_reasoning”: “List spotted changes. Explain   
why   
they are either reasonable interpretations of the   
prompt or   
illogical/weird structural failures. Reference specific   
views   
(e.g., top-left) where issues occur.”,   
“passed”: true/false   
}

## A.2 Generation Time Breakdown

<table><tr><td>Stage</td><td>Time (s)</td></tr><tr><td>Multi-view rendering</td><td> $1 9 . 0 \pm 2 . 3$ </td></tr><tr><td>Prompt pairs generation</td><td> $2 . 1 \pm 0 . 7$ </td></tr><tr><td>Best-view selection + filtering</td><td> $2 1 . 2 \pm 1 4 . 4$ </td></tr><tr><td>Image pairs generation</td><td> $4 3 . 6 \pm 1 6 . 3$ </td></tr><tr><td>Image condition generation</td><td> $2 2 . 0 \pm 1 . 2$ </td></tr><tr><td>3D generation (per variation)</td><td> $6 5 . 3 \pm 1 6 . 3$ </td></tr><tr><td>GLB exporting (per final slider position)</td><td> $3 2 . 3 \pm 4 . 2$ </td></tr></table>

Table A.1: Per-stage time breakdown of our 3D edit ing pipeline (�=40, �=10). Values are reported as mean ± standard deviation in seconds. Multi-view ren dering through image condition generation are one-time costs for each slider. 3D generation repeats once per sampled variation, while GLB exporting repeats once per exported slider position.

## A.3 Participant Information

<table><tr><td>ID</td><td>Age/Gender Experience</td><td></td><td>3D Prototyping Tools</td></tr><tr><td>P1</td><td>25/M</td><td>1.5 yr</td><td>Unity, Unreal, Blender</td></tr><tr><td>P2</td><td>26/F</td><td>2 yr</td><td>Unity, Blender</td></tr><tr><td>P3</td><td>28/M</td><td>10 yr</td><td>Fusion, SolidWorks, Blender, Unity</td></tr><tr><td>P4</td><td>22/F</td><td>1 mo</td><td>SolidWorks</td></tr><tr><td>P5</td><td>24/F</td><td>6 yr</td><td>Fusion, SolidWorks</td></tr><tr><td>P6</td><td>31/F</td><td>10 yr</td><td>Fusion, SolidWorks</td></tr></table>

Table A.2: Demographic information and 3D prototyping experience of participants.

## A.4 Overview of Participant Tasks

Across the two task rounds, participants created sliders spanning diverse domains and semantic attributes.

P1 built a Western saloon scene by generating a bar counter (text-to-3D) with a “clutteredness” slider to adjust the number of bottles on the counter (Fig 5a), and a cowboy character (image-to-3D) with a “worn outfit” slider to control the degree of damage on the outfit. Together the two sliders adjust the level of visual “abandonedness” in the scene. He then created a “photorealism” slider on a variation of the bar counter to make it look more trueto-life, and created a “vibrant red” slider for a cowboy variation to recolor the character. This way he could create more realistic scenes and diferent characters.

P2 was motivated by her interest in disability representation in virtual space. She created a prosthetic leg (image-to-3D) with a “futuristic style” slider exploring diferent designs of prosthetic legs, and used a preloaded human face to create an “eyeball direction” slider to change the convergence/divergence of eyes, representing diferent visual ability. For task 2, she created a “toe abstractness” slider for a prosthetic leg variation to reduce hallucinated anatomical details (e.g., incorrect toe counts) which is common in 3D generation models, and created a “snowy” slider on the preloaded car to simulate snow accumulation over time as a form of temporal storytelling.

P3 created a floor lamp (image-to-3D) with a “futuristic style” slider to change the design of the lamp (Fig 3a), and a motorcycle with an “of-road-ness” slider to make the motorcycle more rugged for of-road use. He then created an “umbrella-shaped hood” slider for a lamp variation to change the lamp shade’s form to be more umbrella-like, with a wider curved canopy (Fig 3b), followed by a “pattern abstractness” slider on another lamp variation to experiment with diferent level of pattern abstractness for the hood design (Fig 3c)—following a top-down strategy of finalizing overall style first, then refining local form, then surface treatment.

P4 created an apple (text-to-3D) with a “ripeness” slider for a continuous progression from unripe to overripe, and a human face (image-to-3D) with an “age/wrinkles” slider to adjust the age of the face, from youthful to elderly. She then attempted to improve the face’s visual quality by creating a “realism” slider on it.

P5 created a cat (text-to-3D) with a “fur length” slider wanting to match her mental image of her own cat, and an octopus (imageto-3D) with a “realism” slider to make it more or less photorealistic. She then started fresh explorations on preloaded objects, creating a “futuristic style” slider on a car and a “strong” slider on a deer (Fig 5b), later realizing the attribute she wanted was maturity rather than muscle mass (see § 6.4).

P6 created a human sculpture with an “abstractness” slider to find the right abstractness level to communicate an intended emotion (Fig 4a). She also created a slider for “expressiveness of eyes” on a silicone soft robot to explore what eye design could give the robot more personality. She then created a “hard vs. soft” slider to get further inspiration on material choices for her sculpture prototyping (Fig 4b), and a body form slider on the robot inspired by pneumatic actuation (using compressed air to produce mechanical motion).

![](images/afe0f62bff34e3156878badef0a5d61a132f3f234c26e0672b02e92d76d7e6d2.jpg)

(d) Top: slider variations before mapping; bottom: after mapping d  
Figure A.1: The process of mapping slider positions to � values. In this example, the 3D object is an airplane, and the target attribute is "rustiness". (a) Sampling � = 9 evenly distributed initial values of �; (b) building the mapping between perceptual distance and slider positions, and using initial anchors for future refinement; (c) interpolating additional anchor points on the slider based on existing anchors to refine the slider mapping; (d) comparison result showing slider variation is smoother and more consistent after slider mapping.  
![](images/1ba4a25185d1229ab39c502105fd3ecaabd972577abd399479623b60421f3865.jpg)  
Figure A.2: The evaluation interface for the technical validation, showing the reference 3D object and target attribute (top left), two sequences of nine 3D variations generated by SemanticSlider3D and the baseline (bottom), and evaluation questions (top right). The mapping between methods and Slider A/B labels is randomized per object-attribute pair; in this example, Slider A corresponds to SemanticSlider3D and Slider B to the baseline. The target attribute is “clothing thickness (thin to thick).”

![](images/700d6c37af4706f0fa977727caafe7e0715a17c35d4c9f7fd93faf85791fbd3e.jpg)

Figure A.3: Assessors’ response to the Likert scale questions.  
![](images/ce80ad4f2a33d836f34f587fba9c9bb0983eac12afd2e7f01b16a770c0585025.jpg)  
Figure A.4: Mean assessor ratings across attribute categories.