# ObjectSplat: Improving Mesh Fidelity and Interactivity for 3D Scenes via Object-Level Mesh Splatting

Minhas Kamal, Hiranya Garbha Kumar, Mahedi Kamal, Balakrishnan Prabhakaran

State University of New York at Albany, USA

{mxkamal, hgkumar, mkamal3, bprabhakaran} @albany.edu

## Abstract

Splatting-based algorithms reconstruct photorealistic, realtime-renderable, and mesh-exportable 3D scenes from regular images, but they represent a scene as a single monolithic field. Therefore, the reconstruction has no objectlevel structure, leaving it infeasible for downstream editing or interaction. Moreover, regions that are never directly observed in the input scans are contaminated by the surrounding texture and left uncorrected, capping both mesh fidelity and novel-view synthesis. We propose a decomposebefore-reconstruct approach: we segment the instances out of every frame, consider the remaining as background and inpaint it, reconstruct each instance and the background independently with mesh splatting, and compose them into a single scene. Our method significantly improves mesh fidelity (over a 5% gain in F-score) and novel-view synthesis, while supporting object-wise modifiability and interactivity. The code will be made publicly available.

## 1. Introduction

Radiance-field methods have transformed the way photorealistic 3D scenes are reconstructed from 2D scans. Neural Radiance Fields (NeRF) [23] first demonstrated that a scene could be encoded as a continuous volumetric function and rendered from novel views with excellent realism, but its implicit formulation is slow to train and query. 3D Gaussian Splatting (3DGS) [19] replaced the neural volume with an explicit set of anisotropic Gaussians and a tile-based rasterizer, achieving comparable quality at real-time framerates. However, 3DGS represents a scene as an unstructured cloud of semi-transparent primitives that float in space and never explicitly model a surface, making its output ill-suited to the polygonal meshes that traditional rendering pipelines and game engines consume. Several subsequent works [11, 45] try to improve the surface quality of 3DGS. However, 2D Gaussian Splatting (2DGS) [16] takes a direct approach, collapsing primitives onto oriented surface-aligned disks.

Following 2DGS, triangle splatting [14] and mesh splatting [13] methods produce explicit meshes that can be imported into 3D engines.

Interactivity - selecting an object and then moving, duplicating, or removing it - requires that a scene be organized into discrete, semantically meaningful objects. This is not merely a structural preference; interaction itself creates new demands. For instance, moving an object reveals its previously unobserved surfaces, making robust novel-view synthesis critical for a seamless user experience. Holistic methods like traditional splatting, however, provide no such object structure. They reconstruct the environment as a single continuous field, a “soup” of primitives with no notion of where one object ends and another begins. Consequently, attempting to move an object can cause the holisticallyoptimized scene to break at the seams, leaving behind corrupted geometry or undefined holes. Although some existing methods attempt to recover object structures after reconstruction, they often produce contact-point artifacts and fail to solve the underlying issue: unobserved areas, now exposed by interaction, are typically filled with noise from the initial joint optimization.

We take a different approach: rather than segmenting a scene after it is reconstructed, we decompose before we reconstruct. Our starting observation concerns why holistic reconstruction limits fidelity in the first place. When a scene is fit as a single field, regions that are never directly observed (the underside of a table, the wall behind a cabinet) receive no corrective gradient and are instead filled in by whatever texture surrounds them, leaving stray, miscolored primitives that degrade the recovered geometry. Reconstructing each object in isolation removes this failure mode: with the background and every other object masked away, the optimizer has nothing to fit but the object itself, and no opportunity to place primitives where the object is not.

We do a 1:7 train-test split (sparse view) for evaluating our methods ability for novel and unseen view synthesis. Our comparison against vanilla mesh splatting show that per-object decomposition yields significant improvements in mesh fidelity (measured by F-score against ground-truth geometry) together with modest gains in photorealism. We attribute this to the per-object problem being fundamentally easier: individual objects converge in fewer than half the iterations a full scene requires during training. Finally, we study completion as a route to still-higher fidelity, and find that today’s obstacle is as much evaluation as method. Our reported metrics exclude every form of completion: diffusion background inpainting and generative amodal completion [41] are not view-consistent enough across frames to train on, and even a deterministic flat-surface completion lowers the measured F-score because the ground-truth meshes are themselves incomplete exactly where completion acts, so a recovered surface is scored against a ground truth that lacks it. Assessing completion fairly therefore needs geometry-complete ground truth, such as realistic synthetic scenes, which this setting still lacks. We report this, alongside the consistency limits of automatic multiview segmentation, to guide future work.

![](images/135c3c6ee104523b57146e33215a5d00f502ee8bad13cfe624361ce94743a98d.jpg)  
Figure 1. Demonstration of our method’s capabilities. (a) Qualitative Comparison: On a novel test view, the baseline holistic reconstruction (top) exhibits significant floating artifacts and texture degradation. In contrast, our object-level approach (bottom) produces a cleaner, more geometrically accurate result. (b) Interactivity: Our object-decomposed representation enables direct manipulation of scene elements, including scaling (purse), appearance editing (shoe), duplication (banana), and rigid transformation (hand). In our pipeline, as backgrounds are inpainted before reconstruction, objects can be moved from their original positions without leaving holes.

## In summary, our contributions are:

• A novel method for object-level reconstruction via mesh splatting, built on a decompose-before-reconstruct principle. Our approach incorporates a new optimization strategy that exploits the opacity of primitives to prune extraneous geometry, leading to substantial improvements in mesh fidelity and novel-view synthesis while producing an editable, object-centric scene.

• Empirical gains over vanilla mesh splatting in mesh fidelity and novel-view synthesis, with the observation that per-object optimization converges in under half the iterations of a holistic reconstruction.

• A systematic analysis of the key bottlenecks in decomposition-based reconstruction. We examine the critical role of cross-view consistency across the segmentation, completion, and inpainting modules, and identify fundamental limitations in current datasets and evaluation metrics that hinder progress.

Table 1. A capability comparison of competing methods. Our work is the only approach that jointly provides an objectdecomposed, mesh-accurate reconstruction with photorealistic appearance and a completed background from sparse views. Legend: ✓ = supported, ∼ = partial support (e.g., per-edit or runtime only), × = not supported.
<table><tr><td>Method</td><td></td><td>Meccu-rate</td><td>Photaisic</td><td>Oobiedoe-oosed</td><td>backround Comppeted</td><td>Spar-vew</td><td>Inutactive</td></tr><tr><td>2DGS</td><td>[16]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>SuGaR</td><td>[11]</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Gaussian Opacity Fields</td><td>[45]</td><td>√</td><td>X</td><td>X</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Mesh / Triangle Splatting</td><td>[13, 14]</td><td>√</td><td>√</td><td>X</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Gaussian Grouping</td><td>[43]</td><td>×</td><td>√</td><td>√</td><td>2</td><td>×</td><td>√</td></tr><tr><td>VR-GS</td><td>[18]</td><td>X</td><td>√</td><td>√</td><td>~</td><td>X</td><td>√</td></tr><tr><td>GScream</td><td>[39]</td><td>×</td><td>√</td><td>X</td><td>2</td><td>×</td><td>X</td></tr><tr><td>InFusion</td><td>[22]</td><td>×</td><td>√</td><td>X</td><td>~</td><td>×</td><td>√</td></tr><tr><td>MAtCha</td><td>[12]</td><td>√</td><td>√</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Object-centric 2DGS [32]</td><td></td><td>√</td><td>√</td><td>~</td><td>×</td><td>X</td><td>√</td></tr><tr><td>Split&amp;Splat</td><td>[26]</td><td>×</td><td>√</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Ours</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 2. Related Works

## 2.1. Monolithic Reconstruction with Splatting

Neural Radiance Fields [23] and 3D Gaussian Splatting (3DGS) [19] established radiance fields as a means of photorealistic, real-time novel-view synthesis, but neither yields the explicit surfaces that downstream graphics pipelines require. Two distinct strategies have since emerged to recover geometry. The first extracts a mesh from a trained 3DGS field as a post-process: SuGaR [11] regularizes Gaussians toward the surface and meshes them via Poisson reconstruction, Gaussian Opacity Fields [45] define a level-set opacity field for adaptive surface extraction, and planar- and depth-based variants [6, 46] further sharpen the recovered geometry. These methods yield accurate geometry, but they optimize shape rather than appearance: the extracted mesh is uncolored, and re-attaching texture from the underlying Gaussians afterward is non-trivial and lossy, which limits the usefulness of the output in rendering and editing pipelines that expect textured assets.

The second strategy makes the surface primitive itself the object of optimization, so that geometry and appearance are learned jointly. 2D Gaussian Splatting (2DGS) [16] collapses each primitive onto an oriented, surface-aligned disk, yielding view-consistent geometry that meshes cleanly, while triangle- and mesh-based splatting [13, 14] optimize photorealistic triangle or mesh primitives end-to-end and produce textured, engine-ready meshes directly. Crucially, however, every method in both categories reconstructs the scene as a single, monolithic field with no object-level structure (Table 1, top block). Our work builds on the second strategy, specifically the mesh-splatting instance.

## 2.2. Post-Hoc Segmentation and Shared-Field Editing

To make a reconstructed scene editable, a body of work attaches object semantics to the primitives after reconstruction training. One line lifts 2D masks (typically from a promptable segmenter such as SAM [20]) into the 3D field, either by optimally assigning masks to primitives (Flash-Splat [36]) or by distilling per-primitive identity and language features during training (Gaussian Grouping [43], SAGA [5], Feature-3DGS [48], LangSplat [29]). Because a single primitive straddling an object boundary is shared between neighbours, selecting or editing one object perturbs the other and produces artifacts at contact points; SAGD [15] identifies these boundary primitives explicitly but resolves them only by splitting within the shared field. Building on such segmentations, interactive systems let objects be moved, removed, or simulated (Point’n Move [17] for click-based manipulation, VR-GS [18] for physics-driven deformation, and text- or physics-based editors [7, 42]), yet each operates on a shared 3DGS field, so moving an object exposes background regions the reconstruction never modelled, which are then filled per edit at inference time rather than reconstructed once (Table 1, second block). Almost all of this work targets 3DGS, whose primitives lack the mesh accuracy of the 2D-splatting representations above.

Closest to our setting are two concurrent, unpublished preprints. Object-centric 2DGS [32] reconstructs a single isolated object on the mesh-accurate 2D-splatting representation, but removes—rather than completes—its background and handles only one object at a time. Split&Splat [26] instead segments a scene into multiple objects, reconstructs each independently, and merges them, making it the most similar in spirit to our pipeline; however, it operates on 3DGS, leaves the background exposed by removed objects unfilled, and does not target mesh fidelity or the sparse-view regime. Neither, moreover, provides a public implementation, precluding a direct experimental comparison. Our method differs from both by reconstructing every object and a completed background as independent mesh splats, and by doing so under sparse views.

## 2.3. Inpainting and Amodal Completion

A related line studies how to fill the regions revealed when an object is removed. In the NeRF setting, SPIn-NeRF [24] established the mask-and-inpaint paradigm, supervising the masked region with a 2D inpainter. Gaussian-domain methods extend this to splats: GScream [39] regularizes the geometry and features of the filled region for cross-view consistency, InFusion [22] learns depth completion from a diffusion prior to place new primitives at correct depths, and RefFusion [25] adapts a diffusion inpainter to a reference view and distills it into the 3D field. These methods fill the hole left by a single removal at edit time; they do not produce a persistent, fully completed background that supports arbitrary rearrangement of many objects, which is what our pipeline reconstructs up front.

A complementary question is how to recover the parts of an object that are themselves occluded. Amodal completion methods such as Amodal3R [41] hallucinate an object’s hidden geometry from its visible extent and could, in principle, further improve the fidelity and completeness of each perobject reconstruction. In practice, however, we find that current amodal methods are not sufficiently consistent across views: injecting their inconsistent completions into the perobject optimization degrades rather than improves it. We therefore treat object completion as a promising but not-yetreliable direction rather than a component of our pipeline.

## 2.4. Sparse-View Radiance-Field Reconstruction

Because dense captures are impractical in many settings, a growing body of work reconstructs radiance fields from sparse views. Regularization-based methods add geometric priors to few-shot 3DGS (FSGS [49] densifies Gaussians guided by proximity and monocular depth, and DNGaussian [21] normalizes depth at global and local scales), while pose-free and feed-forward approaches such as InstantSplat [10] and MVSplat [8] initialize geometry directly from stereo or cost-volume priors, and diffusionprior methods such as ReconFusion [40] synthesize pseudoobservations at unseen viewpoints. Most of these optimize novel-view RGB quality on a monolithic scene; MAtCha [12] is a notable exception in targeting mesh-quality geometry from few views, but it too reconstructs the scene as a whole. None pairs sparse-view robustness with per-object, mesh-accurate decomposition, the regime in which we report our largest gains (Table 1, bottom block).

![](images/4652624c5ee93e9de573a59c3d328aa07cc91b8dd7d1f4cf01816b28f77f1d2f.jpg)  
Figure 2. Overview of our pipeline. Every pixel of every posed frame is assigned to exactly one component: an object or the background. Objects and the background, which for our qualitative results is inpainted but is excluded from quantitative evaluation (see Sec. 4.4), are reconstructed independently; the components are then composed by union into a single, separable scene.

Across these four lines of work, the same gap recurs. Methods that achieve high mesh fidelity reconstruct the scene as a single monolithic field; methods that recover object structure segment an already-reconstructed field, inheriting contact-point artifacts and disocclusion holes that are filled per edit, almost always on 3DGS rather than a mesh-accurate representation, and, because the underlying reconstruction is already fixed, unable to improve its mesh fidelity; inpainting and amodal completion address those holes only one edit at a time, or not yet reliably; and sparseview methods improve few-shot reconstruction but leave the scene holistic and appearance-oriented. Consequently, no prior method in Table 1 satisfies all of these axes at once. We close this gap by inverting the usual order: instead of reconstructing a scene and then decomposing it, we decompose the multi-view observations first and reconstruct each object, alongside a completed background, as independent mesh splats that merge into a single interactive scene. We detail this method in Section 3.

## 3. Methodology

Our method reconstructs a scene not as a single field but as a collection of independently optimized objects and a background, assembled into one environment. The input is a casually captured video of a static scene together with per-frame camera poses; in practice we recover poses with global structure-from-motion [28, 35], though any posed image set suffices. We build on mesh splatting [13], a surface-splatting representation whose primitives are opaque, spherical-harmonic-shaded triangles initialized from the sparse point cloud. One property of this representation is central to our approach: although primitives may be partially transparent early in optimization, they are driven opaque as training proceeds, so a fitted surface cannot be seen through.

The organizing principle of the pipeline is a partition of image space. Across the posed frames, every pixel is assigned to exactly one component: one of the scene’s objects, or the background. Each component is reconstructed independently from the pixels it owns before all components are composed into one scene (Fig. 2). Two properties follow directly and motivate the design. First, because the components partition the images (no pixel is supervised twice, and none is left unsupervised), their union covers exactly what a holistic reconstruction would have covered, so the composed scene is free of holes by construction. Second, because each object is a self-contained component in the shared world frame, it remains separable after the fact: it can be selected, moved, or removed without disturbing its neighbors.

## 3.1. Scene Decomposition

The partition begins with a set of per-frame instance masks that are mutually exclusive: each pixel names at most one object. We obtain these masks in one of two ways, depending on what the capture provides.

When the scene ships with a ground-truth 3D reconstruction carrying per-object annotations, as in ScanNet++ [44], we rasterize the annotated geometry into every camera and read off, at each pixel, the object owning the nearest surface. This yields masks with three convenient properties: occlusion is resolved automatically by the depth test, so a chair behind a table simply cedes those pixels to the table; object identity is shared across all frames by construction, so no cross-frame association is needed; and, because each pixel stores a single owner, the masks are mutually exclusive by representation. Their only imperfection is inherited from the source scan (where it is incomplete, a few pixels are left unowned), which the background construction below absorbs. We use this path for our main results, as it isolates the contribution of the reconstruction method from any error in the segmentation.

The second regime targets the practical setting where no such annotation exists. Here we obtain masks with a video segmenter that follows objects through the capture (SAM 3 [4], prompted with the scene’s object categories), or with an off-the-shelf video object tracker [4, 9, 30] that propagates a set of initial masks across all frames, so that object identity is carried through the video rather than reinferred and re-associated per frame. This removes the dependence on ground-truth geometry, but such trackers are imperfect in two ways that matter downstream. First, their mask boundaries are not perfectly consistent across views: the same object is delineated slightly differently from frame to frame, and these inconsistencies propagate into the reconstruction. Second, and more damaging, a tracker occasionally merges distinct objects under a single identity (an object together with the surface it rests on, say), so that one object’s component is trained on pixels belonging to two scene objects at once. Its reconstruction then fuses their geometry, which breaks separability for interactivity and, because the component is now fit against inconsistent multiobject supervision, degrades mesh and even visual fidelity. We report this regime as a step toward a fully automatic pipeline.

## 3.2. Image-Space Partition and Background Completion

Given the masks, we turn each frame into training images for the components that appear in it. For every retained object we write a full-frame image in which only the object’s own pixels are kept and all others are masked out. We keep the full frame rather than a tight crop so that each object is optimized against the scene’s true cameras; it is the mask, not the framing, that restricts what it learns from. The background is then the per-frame complement of the objects: every pixel not claimed by an object trained in that frame. This makes the partition explicit (each pixel belongs to exactly one component, an object or the background) and is precisely what guarantees the composed scene is gap-free.

Removing the objects, however, leaves the background unobserved wherever an object stood and was never seen from another view. Because these pixels are simply absent from the background’s training images rather than supervised as empty, they leave a hole in the reconstructed room only where the capture never revealed the surface behind an object. For interactive use we complete such regions so that no gap is exposed when an object is relocated, using an image inpainter adapted from a pretrained latent diffusion model [34]. Following Marigold [38], we keep the VAE encoder and decoder frozen and fine-tune only the UNet; we extend its three-channel input with six additiona channels, three for the RGB input and three for the hole mask, and generate training data synthetically from indoor imagery [31] by randomly masking parts of each image. Because the inpainter operates per frame, its completions are not consistent from one view to the next, and training a reconstruction on them would inject that disagreement into the geometry. We therefore use inpainting only to render the final composed scene hole-free for interactivity and qualitative novel-view figures (Fig. 1); it is excluded from all quantitative results, which train the background on its observed pixels alone.

Completion need not stop at the background. Separability introduces a gap of its own: a surface an object rests on, such as a tabletop beneath a keyboard, is reconstructed with a hole where the occluder blocked it, exposed the moment the occluder is moved. We posit that completing such occluded object geometry should further improve reconstruction. Generative amodal completion is the natural tool, but we find that its cross-view inconsistency proves far too damaging for geometry-rich objects and collapses the perobject optimization. To test the premise without that instability, we apply a deterministic proxy on the tractable case of flat supporting surfaces: we recover the occluded region by binary hole-filling of the surface’s mask [37] and inpaint the exposed pixels with Navier–Stokes inpainting [2]. This yields a relatively hole-free supporting surface for interactive use, but we keep it out of our headline metrics: because the ground-truth meshes are themselves incomplete where these surfaces are occluded, a completed surface is scored against a ground truth that lacks it, so the measured F-score drops even where the recovered geometry is plausible. We report this analysis, and the geometry-complete data it would take to evaluate completion properly, in the appendix.

## 3.3. Independent Reconstruction and Background Randomization

Each object and the background are reconstructed by the same, unmodified mesh-splatting optimizer, differing only in the masked images they are shown. Every component is optimized against the shared cameras under an identical protocol (the same resolution and iteration budget), and we keep, per component, the checkpoint that scores best on its own held-out views. Because isolated objects converge at very different rates, and far sooner than a holistic reconstruction of the whole scene (refer to appendix for details), this per-component selection matters. Since the components are independent, they are trained in parallel.

Training a surface splat on a masked object, however, exposes a subtlety of the representation that we exploit. Mesh splatting renders a pixel p by compositing, front to back, the N triangles that cover it [13]. Each triangle $t _ { n }$ contributes its spherical-harmonic color $\mathbf { c } _ { t _ { n } }$ scaled by its opacity $o _ { t _ { n } }$ (the minimum of its three vertex opacities) and a spatial coverage window $I _ { n } ( \mathbf { p } ) \in [ 0 , 1 ]$ , attenuated by the transmittance of the triangles in front of it. Compositing the primitives over a background color b gives

$$
C ( \mathbf { p } ) = \sum _ { n = 1 } ^ { N } \mathbf { c } _ { t _ { n } } w _ { n } ( \mathbf { p } ) \ + \ T ( \mathbf { p } ) \mathbf { b }\tag{1}
$$

$$
w _ { n } ( \mathbf { p } ) = o _ { t _ { n } } I _ { n } ( \mathbf { p } ) \prod _ { i = 1 } ^ { n - 1 } ( 1 - o _ { t _ { i } } I _ { i } ( \mathbf { p } ) )\tag{2}
$$

where $\begin{array} { r } { { \cal T } ( { \bf p } ) = \prod _ { n = 1 } ^ { N } \bigl ( 1 - o _ { t _ { n } } I _ { n } ( { \bf p } ) \bigr ) } \end{array}$ is the residual transmittance reaching the background. Writing the total foreground weight as $\begin{array} { r l r } { W ( \mathbf { p } ) } & { { } = } & { \sum _ { n } w _ { n } ( \mathbf { p } ) \quad = } \end{array}$ $1 \mathrm { ~ - ~ } T ( \mathbf { p } )$ and the weight-averaged foreground color as $\begin{array} { r } { \bar { \mathbf { c } } ( \mathbf { p } ) = { \dot { W } } ( \mathbf { p } ) ^ { - 1 } \sum _ { n } \mathbf { c } _ { t _ { n } } w _ { n } ( \mathbf { p } ) } \end{array}$ , Eq. (1) is simply $C ( \mathbf { p } ) =$ $W ( \mathbf { p } ) \bar { \mathbf { c } } ( \mathbf { p } ) + \big ( 1 - \ddot { W } ( \mathbf { p } ) \big ) \mathbf { k }$ . As optimization matures the vertex opacities saturate $( o _ { t _ { n } }  1 )$ , so a covered pixel becomes opaque and its geometry can no longer be thinned by lowering opacity.

When an object is trained in isolation, every pixel in the masked-out region Ω should render as empty space, that is, as the background itself. Supervising these pixels toward b, the loss over Ω is

$$
\mathcal { L } _ { \Omega } \ = \ \sum _ { \mathbf { p } \in \Omega } \left\| C ( \mathbf { p } ) - \mathbf { b } \right\| ^ { 2 } \ = \ \sum _ { \mathbf { p } \in \Omega } W ( \mathbf { p } ) ^ { 2 } \left\| \bar { \mathbf { c } } ( \mathbf { p } ) - \mathbf { b } \right\| ^ { 2 }\tag{3}
$$

With a fixed background, Eq. (3) is minimized trivially by painting the stray primitives the background color $( { \bar { \mathbf { c } } } \ = \ \mathbf { b } )$ , which leaves constant-colored triangles littering Ω and corrupting the object. We instead resample b from a uniform distribution at every iteration. Taking the expectation over b (mean b<sup>¯</sup>, per-channel variance $\sigma _ { \mathbf { b } } ^ { 2 ^ { - } } > 0 )$

$$
\mathbb { E } _ { \mathbf { b } } \big [ \mathcal { L } _ { \Omega } \big ] \ = \ \sum _ { \mathbf { p } \in \Omega } W ( \mathbf { p } ) ^ { 2 } \Big ( \big \| \bar { \mathbf { c } } ( \mathbf { p } ) - \bar { \mathbf { b } } \big \| ^ { 2 } + \sigma _ { \mathbf { b } } ^ { 2 } \Big )\tag{4}
$$

which, because $\sigma _ { \mathbf { b } } ^ { 2 } > 0$ and the colors are bounded, is minimized only as $W ( \mathbf { p } ) \to 0$ for all $\textbf { p } \in \ \Omega \colon$ no fixed foreground color can track a background that changes every step. Since the opacities are already saturated and the coverage $I _ { n } ( \mathbf { p } )$ of a triangle over a pixel it overlaps cannot vanish, $W ( \mathbf { p } )$ cannot be driven down by making primitives fainter; the only way to satisfy Eq. (4) is to remove the primitives covering Ω. Background randomization thus turns the masked region into a pressure that deletes every primitive not supported by the object. This requires only setting the renderer’s background color b per iteration, which mesh splatting’s differentiable renderer already exposes. It is also what makes the method strong at novel-view synthesis from few captures: a holistic reconstruction sheds such stray, environment-corrupted primitives only once enough viewpoints expose them as inconsistent, whereas background randomization removes them from a single view regardless of how many are available.

This pruning is also the mechanism behind our fidelity gains, and it clarifies why decomposition helps at all. When a whole scene is fit at once, surfaces that are never directly observed (the underside of a table, the wall behind a cabinet) receive no corrective supervision and are left filled with whatever texture the surrounding geometry happened to deposit, a defect no further training removes because nothing observes it. Reconstructing an object in isolation ameliorates this failure mode: with every other surface masked away and the masked region actively pruned, the optimizer has nothing to fit but the object, and no way to place primitives where the object is not.

Finally, decomposition makes each reconstruction far easier: a single object presents much less geometric and appearance variation than a full room and reaches its best checkpoint in fewer than half the iterations a whole scene requires (refer appendix for details), so few views already constrain it well. Composition is then trivial. Because every component was fitted in the same world frame against a partition of the same photographs, their union is the scene: we concatenate the per-object and background primitives into a single splat, with no joint re-optimization and no blending. What each component contributes to a rendered pixel is decided by the renderer’s ordinary depth ordering, exactly as if the primitives had been optimized together. The result is one scene that renders as a whole, yet in which every object remains an identifiable, separable group of primitives available for downstream selection, editing, or export.

Table 2. Comparison of meshsplatting and segmented-meshsplatting across scenes in sparse view (train:test = 1:7)
<table><tr><td></td><td></td><td>Sabbfe</td><td>08b82a</td><td>39305b</td><td>69593969</td><td>Cc01452</td><td>eF1008</td><td>q6935d</td><td>e08a4</td><td>0370</td><td>o8bcçd</td><td>Average</td></tr><tr><td rowspan="4">meshsplatting (30k)</td><td>SSIM↑</td><td>0.748</td><td>0.782</td><td>0.794</td><td>0.820</td><td>0.618</td><td>0.687</td><td>0.728</td><td>0.702</td><td>0.819</td><td>0.769</td><td>0.747</td></tr><tr><td>PSNR↑</td><td>20.207</td><td>21.731</td><td>23.034</td><td>22.422</td><td>14.100</td><td>17.656</td><td>20.297</td><td>18.868</td><td>20.751</td><td>21.362</td><td>20.043</td></tr><tr><td>LPIPS↓</td><td>0.367</td><td>0.310</td><td>0.288</td><td>0.277</td><td>0.429</td><td>0.414</td><td>0.375</td><td>0.385</td><td>0.322</td><td>0.322</td><td>0.349</td></tr><tr><td>F-score↑</td><td>0.412</td><td>0.498</td><td>0.390</td><td>0.521</td><td>0.275</td><td>0.359</td><td>0.400</td><td>0.426</td><td>0.497</td><td>0.437</td><td>0.421</td></tr><tr><td rowspan="4">meshsplatting (best checkpoint)</td><td>SSIM↑</td><td>0.769</td><td>0.786</td><td>0.788</td><td>0.811</td><td>0.675</td><td>0.714</td><td>0.740</td><td>0.717</td><td>0.841</td><td>0.777</td><td>0.762</td></tr><tr><td>PSNR↑</td><td>20.545</td><td>22.168</td><td>22.742</td><td>22.024</td><td>15.931</td><td>18.557</td><td>20.577</td><td>19.219</td><td>21.415</td><td>21.421</td><td>20.460</td></tr><tr><td>LPIPS↓</td><td>0.358</td><td>0.306</td><td>0.297</td><td>0.293</td><td>0.415</td><td>0.402</td><td>0.372</td><td>0.371</td><td>0.297</td><td>0.328</td><td>0.344</td></tr><tr><td>F-score↑</td><td>0.417</td><td>0.514</td><td>0.417</td><td>0.548</td><td>0.299</td><td>0.366</td><td>0.435</td><td>0.435</td><td>0.505</td><td>0.456</td><td>0.439</td></tr><tr><td rowspan="4">segmented-meshsplatting</td><td>SSIM↑</td><td>0.778</td><td>0.785</td><td>0.786</td><td>0.810</td><td>0.687</td><td>0.716</td><td>0.749</td><td>0.712</td><td>0.847</td><td>0.759</td><td>0.763</td></tr><tr><td>PSNR↑</td><td>20.793</td><td>21.737</td><td>22.594</td><td>21.792</td><td>16.486</td><td>18.768</td><td>20.816</td><td>18.921</td><td>21.705</td><td>20.935</td><td>20.455</td></tr><tr><td>LPIPS↓</td><td>0.339</td><td>0.294</td><td>0.293</td><td>0.284</td><td>0.409</td><td>0.398</td><td>0.353</td><td>0.366</td><td>0.283</td><td>0.332</td><td>0.335</td></tr><tr><td>F-score↑</td><td>0.487</td><td>0.554</td><td>0.452</td><td>0.581</td><td>0.367</td><td>0.401</td><td>0.468</td><td>0.485</td><td>0.543</td><td>0.491</td><td>0.483</td></tr></table>

## 4. Experiments

## 4.1. Setup

Datasets. Our approach needs two things of a dataset: per-frame instance masks consistently associated with the same 3D objects across views, and a ground-truth mesh for measuring geometric fidelity. ScanNet++ [44] dataset satisfies both: its per-object 3D annotations rasterize into every frame as mutually consistent masks (Sec. 3.1), and its meshes let us measure F-score directly (scene selection in the appendix). All main results use ScanNet++. We also tried MipNeRF-360 [1], but it has no ground-truth mesh and no instance annotations, and even state-of-the-art video segmentation associates objects inconsistently across its frames, degrading the per-object optimization; we relegate its results to the appendix as evidence the method runs without ground-truth masks.

Novel-view protocol. To probe reconstruction from few views we invert the usual split: whereas vanilla mesh splatting trains on 7 of every 8 frames and tests on the 8th, we train on 1 and test on the other 7, spreading the single training view evenly through the capture. Every method sees the same 1-in-8 training frames and is scored on the same heldout 7. Because the held-out frames differ from the training view by changes in perspective, this measures novelview synthesis quality, sharpened by the scarcity of training

views.

Metrics. We report visual fidelity with PSNR, SSIM, and LPIPS [47], and geometric fidelity with F-score against the ground-truth scan.

Baselines. Our primary baseline is the original mesh splatting implementation [13], for which we report results from the best-performing checkpoint for each scene (evaluated at both a fixed 30k-iteration, as is common in related literature, and at convergence). A direct quantitative comparison to the closest concurrent works (Sec. 2.2) was precluded by the absence of public implementations or results.

Implementation. As a preprocessing step, we use COLMAP [35] to undistort the ScanNet++ images. Each scene component is then trained independently for 30k iterations using the unmodified mesh-splatting optimizer. For the final scene composition, we select the best-performing checkpoint for each individual component. Components are independent and trained in parallel across GPUs. Reported metrics use decomposition and background randomization: the background is trained on its observed pixels, and neither the diffusion inpainter nor the deterministic surface completion of Sec. 3.2 is applied. Both are used only for the qualitative interactivity results and are analyzed in the appendix.

## 4.2. Results

Table 2 reports ScanNet++ results across ten scenes. Our object-decomposed reconstruction improves geometric fidelity (F-score) over vanilla mesh splatting on every scene, at both the 30k and best-checkpoint settings, while PSNR and SSIM stay comparable and LPIPS improves on most scenes: decomposition sharpens geometry substantially and appearance modestly. The circular table in Fig. 1(a) also ablates background randomization; removing it introduces contaminated geometry and textures to the underside, confirming that the pruning it induces is a primary source of the geometric gain rather than a byproduct of decomposition alone. Under the conventional dense split (7 train, 1 test) the same comparison shows a 2% improvement in F1 scores while visual fidelity remains comparable, and the method also runs without ground-truth annotations on MipNeRF-360; both are reported in the appendix.

## 4.3. Analysis

Why decomposition helps novel-view synthesis. The gain comes from how stray, environment-corrupted object primitives are removed. A holistic reconstruction can trim them, but only given enough viewpoints to expose them as inconsistent; as views grow scarce it loses that signal, whereas background randomization prunes them from a single view, and so regardless of view count. Decomposition also eases optimization: each object is a far simpler target and converges in fewer than half the iterations of a full scene (refer to the appendix). We analyze deterministic surface completion, which is excluded from the results above, in the appendix.

## 4.4. Interactivity

Because every object is a separable group of primitives in a shared world frame, the composed scene supports downstream editing directly. Fig. 1 shows objects moved, removed, and rearranged, and the exported meshes brought into a standard game engine. For these qualitative results we fill regions revealed behind relocated objects with the diffusion inpainter of Sec. 3.2, making the scene hole-free for interaction; because its completions are not cross-viewconsistent, it is used only here and not in our quantitative evaluation (see appendix and Sec. 5).

## 5. Limitations and Future Work

Our largest bottleneck is cross-view consistency. Decomposing a scene and reconstructing its parts independently only works when each part is delineated the same way across every frame, and four modules must each supply that consistency: object segmentation, cross-frame association, and occluded-surface completion, and background inpainting. Each falls short today. Automatic video segmentation and tracking produce boundaries that drift from frame to frame and occasionally merge distinct objects, which is why our headline results fall back on ground-truth masks; even the annotation-rasterized masks of ScanNet++ are not perfectly consistent, as the source scans are incomplete (see appendix). Diffusion background inpainting and generative amodal completion [41] are likewise not multi-viewconsistent enough to train a reconstruction on, so our quantitative results exclude both. Surface completion is excluded for a second reason: the ground-truth meshes are themselves incomplete precisely in the occluded regions completion targets, so a completed surface is scored as a false positive against a ground truth that lacks it, and the measured Fscore falls even where the recovered geometry is plausible. We therefore cannot presently confirm completion’s benefit through geometric metrics, only its qualitative role in producing hole-free interactive scenes. The absence of a suitable dataset is itself a limitation on two fronts. No dataset we found supplies consistent per-frame instance masks tied to 3D objects alongside a ground-truth mesh except Scan-Net++, which constrains both evaluation and the scenes we can study; and measuring completion fairly would additionally require geometry-complete ground truth, such as realistic synthetic CAD scenes, which do not yet exist for this setting. Building such a dataset is an important direction for making completion-based methods measurable.

A second set of limitations is inherited from the underlying mesh-splatting representation. The recovered meshes are not watertight, so they cannot be driven directly by a physics engine; their textures bake in the capture’s lighting, precluding accurate relighting; and their surfaces are spikily and haphazardly triangulated, which costs visual fidelity on close inspection and yields inaccurate contact geometry for interaction. Our reconstructions also assume a static scene and depend on recovered camera poses, and per-object optimization adds cost, though it is largely offset by training components in parallel. Improving the mesh quality of the base representation would compound directly with the fidelity gains from decomposition.

## 6. Conclusion

We presented a decompose-before-reconstruct approach that reconstructs a scene as independently optimized objects and a background over a partition of image space, instantiated on mesh splatting. Its central ingredient is a background-randomized training scheme that, by exploiting the non-transparency of mesh-splatting primitives, prunes stray geometry during optimization and drives our gains in mesh fidelity and novel-view synthesis, with the largest margins when input views are few. The same decomposition yields a separable, mesh-exportable scene that is directly usable in downstream applications. Our analysis of completion shows that its benefit is presently hard to measure rather than absent: a recovered surface is penalized against ground-truth meshes that are themselves incomplete. The cross-view inconsistency of current generative methods for inpainting, completion, and segmentation remains a primary bottleneck, indicating that future gains will come from developing more consistent models and the geometry-complete datasets needed to train them.

## References

[1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022. 7, 14

[2] Marcelo Bertalmio, Andrea L Bertozzi, and Guillermo Sapiro. Navier-stokes, fluid dynamics, and image and video inpainting. In Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, pages I–I. IEEE, 2001. 5

[3] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools, 2000. 12

[4] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle, Triantafyllos Afouras, Effrosyni Mavroudi,¨ Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, RISHI HAZRA, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In The Fourteenth International Conference on Learning Representations, 2026. 5, 12

[5] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians. In Proceedings of the AAAI conference on artificial intelligence, pages 1971–1979, 2025. 3

[6] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics, 31 (9):6100–6111, 2024. 3

[7] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21476–21485. IEEE, 2024. 3

[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European conference on computer vision, pages 370–386. Springer, 2024. 4

[9] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1316– 1326. IEEE, 2023. 5

[10] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Sparse-view gaussian splatting in seconds. arXiv preprint arXiv:2403.20309, 2024. 4

[11] Antoine Guedon and Vincent Lepetit. Sugar: Surface-´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5354–5363. IEEE, 2024. 1, 2, 3

[12] Antoine Guedon, Tomoki Ichikawa, Kohei Yamashita, and´ Ko Nishino. Matcha gaussians: Atlas of charts for high-

quality geometry and photorealism from sparse views. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6001–6011. IEEE, 2025. 2, 4

[13] Jan Held, Sanghyun Son, Renaud Vandeghen, Daniel Rebain, Matheus Gadelha, Yi Zhou, Anthony Cioppa, Ming C. Lin, Marc Van Droogenbroeck, and Andrea Tagliasacchi. MeshSplatting: Differentiable rendering with opaque meshes, 2025. 1, 2, 3, 4, 6, 7, 12

[14] Jan Held, Renaud Vandeghen, Adrien Deliege, Abdullah Hamdi, Daniel Rebain, Silvio Giancola, Anthony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, et al. Triangle splatting for real-time radiance field rendering. In Thirteenth International Conference on 3D Vision, 2025. 1, 2, 3

[15] Xu Hu, Yuxi Wang, Lue Fan, Chuanchen Luo, Junsong Fan, Zhen Lei, Qing Li, Junran Peng, and Zhaoxiang Zhang. Sagd: Boundary-enhanced segment anything in 3d gaussian via gaussian decomposition. IEEE Transactions on Image Processing, 2026. 3

[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accu rate radiance fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024. 1, 2, 3

[17] Jiajun Huang, Hongchuan Yu, Jianjun Zhang, and Hammadi Nait-Charif. Point’n move: Interactive scene object manipu lation on gaussian splatting radiance fields. IET Image Processing, 18(12):3507–3517, 2024. 3

[18] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIG-GRAPH 2024 conference papers, pages 1–1, 2024. 2, 3

[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 3

[20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In 2023 IEEE/CVF international conference on computer vision (ICCV), pages 3992–4003. IEEE, 2023. 3

[21] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normal ization. In 2024 IEEE/CVF Conference on Computer Vi sion and Pattern Recognition (CVPR), pages 20775–20785. IEEE, 2024. 4

[22] Zhiheng Liu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jie Xiao, Kai Zhu, Nan Xue, Yu Liu, Ping Luo, and Yang Cao. Infusion: Inpainting 3d gaussians via learning depth completion from diffusion prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5726–5738, 2026. 2, 3

[23] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view syn-

thesis. Communications ofthe ACM, 65(1):99–106, 2021. 1, 3

[24] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G Derpanis, Jonathan Kelly, Marcus A Brubaker, Igor Gilitschenski, and Alex Levinshtein. Spin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20669–20679. IEEE, 2023. 3

[25] Ashkan Mirzaei, Riccardo De Lutio, Seung Wook Kim, David Acuna, Jonathan Kelly, Sanja Fidler, Igor Gilitschenski, and Zan Gojcic. Reffusion: Reference adapted diffusion models for 3d scene inpainting. arXiv preprint arXiv:2404.10765, 2024. 3

[26] Leonardo Monchieri, Elena Camuffo, Francesco Barbato, Pietro Zanuttigh, and Simone Milani. Split&splat: Zero-shot panoptic segmentation via explicit instance modeling and 3d gaussian splatting. arXiv preprint arXiv:2602.03809, 2026. 2, 3

[27] Ege Ozguroglu, Ruoshi Liu, D´ıdac Sur´ıs, Dian Chen, Achal Dave, Pavel Tokmakov, and Carl Vondrick. pix2gestalt: Amodal segmentation by synthesizing wholes. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3931–3940, 2024. 13

[28] Linfei Pan, Daniel Bar´ ath, Marc Pollefeys, and Johannes L.´ Schonberger. Global structure-from-motion revisited. In¨ Computer Vision – ECCV 2024, pages 58–77. Springer Nature Switzerland, 2025. 4

[29] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20051–20060. IEEE, 2024. 3

[30] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Seg-¨ ment anything in images and videos. In International Conference on Learning Representations, pages 28085–28128, 2025. 5

[31] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10892–10902. IEEE, 2021. 5, 13

[32] Marcel Rogge and Didier Stricker. Object-centric 2d gaussian splatting: Background removal and occlusionaware pruning for compact object models. arXiv preprint arXiv:2501.08174, 2025. 2, 3

[33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10674–10685. ieee, 2022. 13

[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In 2022 IEEE/CVF con-

ference on computer vision and pattern recognition (CVPR), pages 10674–10685. ieee, 2022. 5, 13

[35] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4104–4113, 2016. 4, 7

[36] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally. In European Conference on Computer Vision, pages 456–472. Springer, 2024. 3

[37] Pierre Soille et al. Morphological image analysis: principles and applications. Springer, 1999. 5

[38] Massimiliano Viola, Kevin Qu, Nando Metzger, Bingxin Ke, Alexander Becker, Konrad Schindler, and Anton Obukhov. Marigold-dc: Zero-shot monocular depth completion with guided diffusion. In 2025 IEEE/CVF International Confer ence on Computer Vision (ICCV), pages 5359–5370. IEEE, 2025. 5, 13

[39] Yuxin Wang, Qianyi Wu, Guofeng Zhang, and Dan Xu. Learning 3d geometry and feature consistent gaussian splatting for object removal. In European conference on computer vision, pages 1–17. Springer, 2024. 2, 3

[40] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21551–21561, 2024. 4

[41] Tianhao Wu, Chuanxia Zheng, Frank Guan, Andrea Vedaldi, and Tat-Jen Cham. Amodal3r: Amodal 3d reconstruction from occluded 2d images. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9181–9193. IEEE, 2025. 2, 3, 8

[42] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physicsintegrated 3d gaussians for generative dynamics. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4389–4398. IEEE, 2024. 3

[43] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In European conference on computer vision, pages 162–179. Springer, 2024. 2, 3

[44] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3d indoor scenes. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12–22. IEEE, 2023. 5, 7, 12

[45] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (ToG), 43(6):1–13, 2024. 1, 2, 3

[46] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiao-Xiao Long, and Ping Tan. Rade-gs: Rasterizing depth in gaussian splatting. ACM Transactions on Graphics, 45(2): 1–14, 2026. 3

[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep

features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision andpattern recognition, pages 586– 595. IEEE, 2018. 7

[48] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21676–21685. IEEE, 2024. 3

[49] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145–163. Springer, 2024. 3

## A. Scene Selection

For our experimental evaluation, we select a subset of 10 scenes from the ScanNet++ [44] dataset. To ensure a rigorous and fair assessment of our decomposition and reconstruction pipeline, these scenes were curated based on the following two primary criteria:

• Ground-Truth Geometry: The chosen scenes exhibit high-quality, relatively complete ground truth (GT) meshes, which is critical for accurate quantitative geometric evaluation and generation of complete 2D masks via backprojection.

• Object Composition: The environments contain a sufficient number of distinct, decomposable object instances that are highly relevant to our methodology (specifically for ScanNet++: chairs, tables, sofas, coffee tables, benches, PCs, and pool tables).

Furthermore, to specifically evaluate the usefulness of OpenCV’s [3] Navier-Stokes-based deterministic surface completion, we consider a narrower subset of 4 scenes. Because our heuristic for identifying fillable occlusion holes is intentionally conservative to avoid aggressive geometric hallucinations, we focus on scenes featuring planar surfaces (e.g., desks, tables, etc).

## B. Modality Disparity and Backprojection Mask Incompleteness

A key challenge in leveraging 3D annotations for 2D mask generation is the fundamental modality disparity between high-resolution RGB images and sensor-reconstructed 3D meshes. As ScanNet++ [44] is a real-world dataset, the GT 3D instance annotations are defined on a reconstructed mesh that is inherently imperfect, containing local geometric defects, scanning artifacts, and missing surfaces. While back-projecting these 3D annotations onto the 2D image plane, these underlying mesh defects propagate directly into the 2D domain, illustrated in Fig. 3, yielding incomplete masks that fail to cover the objects entirely. While our method’s capacity to improve mesh and visual fidelity in sparse-view settings largely absorbs these degradations, rendering the net impact on our sparse-view reconstruction minor, this modality gap nonetheless imposes an upper bound on performance that a geometrically pristine 3D annotation source would resolve.

Albeit this mask incompleteness, the GT 3D annotations remain multi-view consistent, which is critical for current splatting methods. Cross-view consistency issues get significantly exacerbated when transitioning to ML-based segmentation pipelines (e.g., using 2D video tracking models such as SAM3 [4]). Consequently, we utilize the rasterized ground-truth annotations from ScanNet++ for our primary evaluations.

![](images/219d4fbf4b8994046394a303f5d03af62d1206c3278dc92b4311e718458b058f.jpg)  
Figure 3. Limitations of 3D-to-2D mask projection due to incomplete geometry in ScanNet++. Due to missing geometry and reconstruction artifacts in the 3D meshes (visible as checkerboard background patterns indicating holes), projecting 3D segmentations to 2D viewports yields incomplete masks and false visibility boundaries. Specific failure cases include: (a) occluded underdesk regions and thin structures such as chair legs, and (b) transparent/specular surfaces (e.g., window panes) as well as dark, non-Lambertian surfaces (e.g., leather sofa folds).

## C. Per-Component Training Convergence

The variance in training dynamics, observed in Fig. 4, is heavily governed by the physical scale of the target: smaller object instances possess lower geometric and textural complexity compared to large supporting structures or dense backgrounds, thereby requiring significantly fewer optimization iterations to converge. This disparity in convergence rates makes our per-component training strategy highly advantageous, as it allows us to perform bestcheckpoint selection individually for each object, avoiding the sub-optimal compromises and over-fitting typical of uniform holistic training. We note that we select our best checkpoint after the Restricted Delaunay Triangulation (RDT) step (11K-th iteration in Mesh-Splatting [13]) which is run only once during the entire training routine and improves the connectivity of the triangle soup.

## D. Background Inpainting and Amodal Completion

## D.1. Network Architecture

To recover unobserved background regions (inpainting) previously occluded by foreground objects, as well as to generate hidden geometry for partially occluded foreground instances (amodal completion), we design a unified latent diffusion-based architecture. While the structural design of our network takes inspiration from the dense prediction framework introduced in Marigold [38], we adapt it for: background inpainting and amodal object completions.

![](images/6a657fab789b530eacd146d1fd4a8a1a99ec6b19dec28b564f80f35706da3776.jpg)  
Figure 4. Training Graph. Individual objects reach their best-scoring checkpoints in fewer iterations than the whole-scene reconstruction. Plus, objects converge at markedly different rates from one another primarily depending on their shape simplicity. This figure shows training and validation graphs (PSNR, SSIM, LPIPS, L1=m\*PSNR+n\*SSIM+o\*LPIPS) throughout the 30k optimization iterations.

![](images/b75a03249d6618547626dbc3605dd89e5e9e5d254f3f7ee1b0399585ebe9e83b.jpg)  
Figure 5. Adapted stable diffusion (SD) [34] architecture for object completion and background inpainting. We condition on incomplete rgb and mask for the denoising operation, similar to Marigold [38]. Here, we take a pretrained SDv1.5 and modify the input layer of the U-Net. The network is retrained by synthetically generated data from the Hypersim [31] dataset for both background inpainting (Fig. 6) and object completion (Fig. 7); the text encoder and the image encoder-decoder (VAE) are kept frozen. During inference, we employ a variant of the Classifier-Free Guidance approach.

To leverage generalized image priors, we start with the pre-trained Stable Diffusion v1.5 [33]. During the training phase, only the Diffusion U-Net is gets updated. For this retraining, we utilize synthetic indoor environments from the Hypersim dataset [31], which provides diverse, photorealistic scenes for learning complex geometries and lighting variations. Though background inpainting and amoda completion are significantly different tasks, we use the same architecture for both models. Background inpainting model is simply trained on randomly masked images. However, generating data for amodal completion is tricky: we use a method similar to [27]. All other auxiliary networks, including the VAE encoder, VAE decoder, and text encoders, remain strictly frozen. This selective fine-tuning approach preserves the robust foundational visual priors of the pretrained latent space, while allowing the U-Net to specialize on our tasks.

## D.2. Performance and Optimization Limitations

While our diffusion-based model is capable of generating visually plausible background and amodal completions for isolated frames, it inherently struggles to maintain strict multi-view consistency. As illustrated for background inpainting in Fig. 6, and for object completion in Fig. 7, applying the 2D generative model independently across varying perspectives results in severe spatial inconsistencies.

Masked Background

Inpainted Background

![](images/911ab1e664e1d38892ec977a80e7aadce2c708b64f8b48fe2b579d348b39d606.jpg)  
Figure 6. Multi-view inconsistency in 2D background inpainting. While the diffusion model fills the masked regions (left), its independent 2D generation produces inconsistent textures for the same physical area across different views (right). In (a) and (c) the model hallucinates grid patterns but of different color, and (b) has a flat, untextured gray surface. These severe frame-to-frame structural and textural variations create conflicting gradients for splatting methods degrading the 3D mesh optimization.

For instance, when generating the occluded regions of a hexagonal wooden table (Fig. 7), the model inconsistently predicts structural elements like missing legs, while introducing severe texture noise and geometric warping across different frames.

Because our 3D splatting optimization relies heavily on photometric consistency across overlapping views, these frame-to-frame generative variations act as contradictory supervisory signals. If injected directly into the optimization pipeline, these conflicting gradients force the model to average over inconsistent appearances, which actively degrades both the reconstructed mesh geometry and the overall texture fidelity. Consequently, while 2D generative completion serves as a valuable tool for downstream interactive scene editing and visualization, both background inpainting and amodal completion are expressly excluded from our optimization pipeline during quantitative evaluations to prevent geometric corruption.

Completed Object

![](images/aa038af446ff5d2e58b330429bce790fa703988f9e1f5dacb62e3fba6c6cc4a3.jpg)  
Figure 7. Multi-view inconsistency in 2D amodal completion. While applying independent amodal completion to occluded objects across different frames, the generative model produces inconsistent predictions. (a) and (b) has missing legs, and (c) has severe texture noise and geometric warping. These inconsistencies get propagated and corrupts the splat training.

## E. MipNeRF-360 Results

ScanNet++ supplies both consistent per-object masks and a ground-truth mesh, so it carries our main results. To show that the method still runs without ground-truth annotations, we also report MipNeRF-360 [1], whose masks are obtained through the automatic video-segmentation path of Section Methodology (Scene Decomposition). Because MipNeRF-360 provides no ground-truth masks, the crossview inconsistency of automatic segmentation causes a marginal loss of visual fidelity relative to the ground-truthmask setting; with no ground-truth mesh available, we cannot evaluate its effect on geometric fidelity. However, we present qualitative reconstruction results on various objects from the Mip-NeRF 360 dataset in Fig. 8.

## F. Surface Completion

The surface completion proposed in our methodology is designed to recover occluded regions of supporting structures (e.g., tabletops, desks) that are shadowed by foreground object instances. Fig. 9 shows results from this method. By patching these occlusions, we reconstruct a relatively

Original

Observed Views

Unobserved Views

![](images/4931b5b810092c40d9eed7fae57d556c3af1673a7667e11dbbd1239d2d23f982.jpg)  
Figure 8. Robust mesh and visual reconstruction across extreme perspective changes on Mip-NeRF 360. Our approach maintains highly consistent geometry and texture quality from both observed views (left) and completely unobserved, novel viewpoints (right). Despite significant shifts in perspective, the extracted 3D structures remain complete and artifact-free, demonstrating the generalization capability and spatial consistency of the reconstruction across diverse objects.

complete background mask suitable for downstream interactive editing. To evaluate this module, we conduct a comparative analysis on the completion subset of ScanNet++ (the four scenes containing prominent planar structures; see Sec. A), comparing our final reconstructions with and without surface completion. For the completion pipeline, we utilize OpenCV’s Navier-Stokes-based inpainting method (cv2.inpaint) over the projected planar surfaces.

As shown in Table 3, executing surface completion results in a marginal decrease in 3D F-score, while leaving 2D rendering metrics (PSNR, SSIM, and LPIPS) virtually unchanged. The invariance of the visual metrics is expected: during evaluation, the segmented foreground objects are rendered in their original positions, thereby reoccluding the completed regions and preventing them from contributing to the image-space loss. We hypothesize that the slight decline in F-score is an artifact of the evaluation target rather than a reflection of geometric degradation. Sensor-based ground-truth meshes (such as those in ScanNet++) are themselves incomplete in occluded regions, meaning that a physically plausible, reconstructed surface is penalized as a false positive when measured against a ground truth that lacks geometry in those coordinates. Fully validating the fidelity of completed surfaces requires a geometry-complete ground truth—such as realistic, synthetic CAD-modeled environments—which does not currently exist at the scale and complexity of ScanNet++. We view the development of such complete synthetic benchmarks as an essential prerequisite for the quantitative evaluation of completion-based reconstruction techniques.

![](images/2e30b8941ff6d1844d3fced0c1cfc957c978ac679d531db6b7980ad24d022968.jpg)  
Figure 9. Qualitative results of deterministic surface completion. Original reconstructed supporting surfaces exhibit occlusion holes and broken edges where foreground objects previously stood (left). The OpenCV deterministic completion method successfully repairs these planar regions (right).

Table 3. Effect of deterministic surface completion. Quantitative comparison (F-score, PSNR, SSIM, and LPIPS) on 4 selected ScanNet++ scenes, demonstrating the impact of deterministic flatsurface completion.
<table><tr><td></td><td colspan="3">without completion</td><td colspan="3">with completion</td></tr><tr><td>Scene</td><td></td><td>SSIM↑ PSNR↑ LPIPS↓</td><td>F1↑</td><td></td><td>SSIM↑ PSNR↑ LPIPS↓</td><td>F1↑</td></tr><tr><td>08bd80ce2a</td><td>0.785</td><td>21.737</td><td>0.294 0.554</td><td>0.785</td><td>21.771</td><td>0.296 0.550</td></tr><tr><td>69e5939669</td><td>0.810</td><td>21.792</td><td>0.284 0.581</td><td>0.811</td><td>21.811</td><td>0.283 0.580</td></tr><tr><td>fe5fe0a8a4</td><td>0.713</td><td>18.921</td><td>0.366 0.485</td><td>50.711</td><td>18.842</td><td>0.367 0.484</td></tr><tr><td>08bbbdcc3d 0.759</td><td></td><td>20.935</td><td>0.3320.491 0.755</td><td></td><td>20.708</td><td>0.338 0.491</td></tr></table>

## G. Dense-View Experiments

Our primary results focus on sparse views to evaluate our method’s capability for unseen view synthesis. Here, we additionally present results from dense views to demonstrate its scene reconstruction quality, which is on par with vanilla mesh splatting.

Table 4. Dense-view reconstruction results on ScanNet++. Per-scene and average quantitative metrics (SSIM, PSNR, LPIPS, and F-score) comparing vanilla mesh splatting and our decomposition method under a dense 7:1 train-to-test split.
<table><tr><td></td><td></td><td>a2bbfe</td><td>08be2a</td><td>3913d05b</td><td>65399369</td><td>cc0452</td><td>eF18708</td><td>Ff64935d</td><td>8a4</td><td>030</td><td>qq3d</td><td>Average</td></tr><tr><td></td><td>SSIM↑</td><td>0.849</td><td>0.834</td><td>0.798</td><td>0.846</td><td>0.850</td><td>0.776</td><td>0.794</td><td>0.799</td><td>0.887</td><td>0.869</td><td>0.830</td></tr><tr><td>meshsplatting (30k)</td><td>PSNR↑</td><td>24.163</td><td>25.343</td><td>23.874</td><td>24.029</td><td>22.762</td><td>22.459</td><td>23.389</td><td>22.167</td><td>24.702</td><td>25.563</td><td>23.845</td></tr><tr><td></td><td>LPIPS↓</td><td>0.262</td><td>0.279</td><td>0.305</td><td>0.273</td><td>0.285</td><td>0.385</td><td>0.333</td><td>0.319</td><td>0.256</td><td>0.240</td><td>0.294</td></tr><tr><td></td><td>F-score↑</td><td>0.466</td><td>0.599</td><td>0.471</td><td>0.603</td><td>0.338</td><td>0.411</td><td>0.509</td><td>0.547</td><td>0.605</td><td>0.560</td><td>0.511</td></tr><tr><td></td><td>SSIM↑</td><td>0.852</td><td>0.802</td><td>0.790</td><td>0.812</td><td>0.850</td><td>0.769</td><td>0.776</td><td>0.779</td><td>0.877</td><td>0.833</td><td>0.814</td></tr><tr><td>Ours</td><td>PSNR↑</td><td>23.704</td><td>23.200</td><td>23.482</td><td>22.261</td><td>23.633</td><td>21.255</td><td>22.121</td><td>21.037</td><td>23.462</td><td>23.535</td><td>22.769</td></tr><tr><td></td><td>LPIPS↓</td><td>0.242</td><td>0.298</td><td>0.312</td><td>0.304</td><td>0.279</td><td>0.374</td><td>0.341</td><td>0.323</td><td>0.248</td><td>0.271</td><td>0.300</td></tr><tr><td></td><td>F-score↑</td><td>0.525</td><td>0.555</td><td>0.492</td><td>0.585</td><td>0.412</td><td>0.466</td><td>0.492</td><td>0.458</td><td>0.502</td><td>0.547</td><td>0.503</td></tr></table>

In our dense-view experiments (detailed in Table 4), we observe that our method performs marginally below holistic mesh splatting. We hypothesize that this performance characteristic is governed by a trade-off between mask quality and view density:

• Dense-View Regime: With highly redundant viewpoints, the vanilla mesh splatting baseline can leverage the dense multi-view optimization constraints to effectively prune erroneous geometry on its own. Consequently, the relative margin of improvement from our decomposition vanishes, and the cumulative negative impact of noisy, incomplete boundaries across numerous frames begins to dominate the performance trade-off.

• Sparse-View Regime: Conversely, under severe view constraints, the vanilla baseline lacks the redundant perspective supervision necessary to self-prune erroneous geometry, resulting in severe floaters and corrupted surfaces. Under these conditions, the explicit geometric regularization and floater elimination provided by our objectlevel decomposition heavily outweigh the local errors introduced by imperfect masks, resulting in a substantial net improvement in both mesh and rendering quality without any hyper-parameter tuning.