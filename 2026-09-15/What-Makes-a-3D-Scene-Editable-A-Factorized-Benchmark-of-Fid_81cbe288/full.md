# What Makes a 3D Scene Editable? A Factorized Benchmark of Fidelity, Locality, Consistency, and Preservation

Sariah Patro

Arjun Mehra

Nikhil Bhatia

University of Delhi & Jadavpur University

## Abstract

Neural 3D scene editing is often evaluated by semantic alignment alone, although a convincing result may alter unrelated content or become inconsistent across views. We introduce EditBench3D, a representation-agnostic benchmark that treats editing as controlled information replacement. It evaluates four complementary properties: instruction fidelity spatial locality, cross-view consistency, and preservation of non-target content. The protocol combines visibility-aware 3D target supports, paired descriptions, held-out cameras, and five edit families covering appearance, material, geometry, and object-level changes. We evaluate eight representative NeRF, 3D Gaussian Splatting, hybrid, and proxybased editors on 240 scene-edit pairs. The study shows that semantic fidelity is only weakly associated with the other editing properties, and that no single method is optimal across all dimensions. Explicit Gaussian editors offer a strong overall balance, whereas direct proxy manipulation provides the most conservative edits at the cost of openended fidelity. These findings support reporting editability as a multi-objective profile rather than a single semantic score.

## 1. Introduction

Neural radiance fields (NeRFs) made photorealistic novelview synthesis possible with a continuous, differentiable representation [1]. Mip integration, scene contraction, hash encodings, sparse voxels, and tensor factorizations improved image quality and efficiency [2–7]. 3D Gaussian Splatting (3DGS) later moved the execution model to explicit anisotropic primitives and real-time rasterization [8–11]. Both families now serve as editable scene representations, with language, reference images, masks, meshes, or direct manipulation supplying the edit.

Evaluation has not converged at the same pace. Many text-guided methods report CLIP similarity, user preference, or a small collection of qualitative views. These measures answer whether an image appears compatible with a prompt, but not whether the scene was edited correctly. Consider the instruction “make the chair blue.” A global blue tint can improve text-image similarity, although the wall and floor should remain unchanged. A result may look convincing from the reference camera while the chair changes color or shape in nearby views. Conversely, a successful removal exposes previously occluded content; comparing the revealed pixels with the original image would incorrectly count a valid edit as damage.

The central difficulty is that editing contains two coupled obligations: change what was requested and retain what was not. A generative metric mainly observes the first. Reconstruction metrics mainly observe the second. Neither alone measures editability. Existing results from CLIP-NeRF [12], Instruct-NeRF2NeRF [13], Blending-NeRF [14], GaussianEditor [15], and GaussCtrl [16] are therefore meaningful within their own settings, but their reported scores do not produce a common account of scene editing [14, 17–20].

We address this gap with EditBench3D. The benchmark represents an edit by a scene S, an instruction c, an intended factor set T, and a visibility-aware 3D support Ω. Four dimensions are measured independently. Fidelity asks whether the requested change occurred. Locality measures spill outside the target support. Consistency evaluates whether corresponding edited surface points agree across views. Preservation measures retained appearance and geometry in nontarget regions. Figure 1 summarizes the evaluation path.

The four-way decomposition is supported by the experiment. Across eight editors, fidelity has Spearman rank correlations of only 0.14 with locality, 0.05 with consistency, and 0.24 with preservation. Thus, a method that follows an instruction well is not predictably local, consistent, or conservative. The comparison also exposes a clear representation trend: the evaluated 3DGS editors improve mean fidelity by 0.16 over the evaluated NeRF editors, but the best non-generative qualities are obtained by direct proxy manipulation. A single ranking would conceal this distinction.

Our contributions are threefold:

• We formulate editability as a four-dimensional property of an editing system, separating instruction fidelity, locality, cross-view consistency, and preservation.

• We introduce a controlled protocol with visibility-aware target supports, paired prompts, held-out cameras, five edit families, and rules for occlusion, disocclusion, failures, and unsupported operations.

![](images/19dfdb2105162ba114cf6522ec07b37d28d255bcdc6d9aaeb2e89256947984f5.jpg)  
Figure 1. EditBench3D separates what should change from what should remain. Given a captured scene and an edit specification (instruction c and target factors T), an editor produces an edited scene. We evaluate four complementary properties: fidelity (F), locality (L), consistency (C), and preservation (P). These properties are measured using visibility-aware 3D target support Ω across multiple held-out views. The benchmark covers five edit families: appearance, geometry, insertion, removal, and style.

• We evaluate eight representative editors on 240 sceneedit pairs and identify representation-level trends, weak coupling between fidelity and the preservation-oriented dimensions, and a three-method Pareto front.

## 2. Related Work

## 2.1. Scene representations

NeRF maps a 3D position and direction to density and color and forms a pixel by volume rendering [1]. Mip-NeRF and its successors reduce aliasing and handle unbounded scenes [2–4]. Instant-NGP, Plenoxels, and TensoRF replace or augment the original MLP with grids, explicit voxels, or tensor factors [5–7]. Baking and octree methods convert learned fields into structures better suited to real-time rendering [21–23]. Generalizable approaches infer fields from one or a few scenes rather than optimizing every scene independently [24, 25].

3DGS uses explicit means, anisotropic covariances, opacity, and view-dependent appearance [8]. Mip-Splatting and multiscale splatting improve scale behavior [26, 27];

2DGS, SuGaR, Gaussian Opacity Fields, SFF, and RaDe-GS strengthen the connection to surfaces and depth [28–32]. Other work revisits densification, sorting, compactness, and compression [33–37]. Scaffold-GS, SplatFields, Hash-GS, and HybridNeRF show that learned fields and explicit primitives occupy a continuum rather than a binary taxonomy [38– 41].

This representational diversity matters for editing. An explicit primitive can be selected, moved, frozen, or deleted. A shared neural function offers continuity but can propagate a local parameter update globally. EditBench3D does not assume that one family is intrinsically superior; it tests how the complete editing system behaves after representation, supervision, and optimization interact.

## 2.2. Neural scene editing

Editing conditional radiance fields introduced latent and user-guided control for category-level models [17]. CLIP-NeRF aligns disentangled shape and appearance codes with language-image embeddings [12]. NeRF-Editing connects a neural field to an editable mesh, while NeuralEditor uses point-cloud manipulation as an explicit geometric scaffold [42, 43]. Blending-NeRF trains an editable field beside a frozen field for local density and appearance changes [14]; PVD and PVD-AL transfer knowledge across field architectures [44, 45]; LAENeRF focuses on local appearance [46]; NeRFEditor decomposes style components [18]; and pose-conditioned dataset updates support insertion and removal [47], and NeRF-GS jointly exploits continuous field information and explicit splats [48].

Instruct-NeRF2NeRF established a widely used datasetupdate pipeline in which InstructPix2Pix edits rendered training views while the scene is repeatedly optimized [13, 49]. DN2N learns to suppress inconsistent 2D-edit perturbations and performs new edits without per-scene retraining [50]. GaussianEditor transfers this idea to explicit Gaussians and uses a Gaussian region of interest [15]. GaussCtrl edits multiple images together with depth control and latent alignment [16]; InterGSEdit introduces geometry-consistent attention and interactive key-view selection [19]. These methods directly motivate our locality and consistency dimensions. Chat-Edit-3D uses a language-model controller and a Hash-Atlas to separate visual editing tools from scene reconstruction [51, 52].

## 2.3. Semantic priors and evaluation

CLIP supplies a transferable text-image embedding [53], latent diffusion supplies high-capacity image synthesis [54], InstructPix2Pix adds instruction following [49], and Segment Anything supplies promptable masks [55]. LERF, MeshLLM, LangSplat, and Feature 3DGS lift semantic or visual features into 3D [10, 56–58]. These priors enable editing, but their scores are not complete measures of edit quality.

PSNR and SSIM compare corresponding pixels, and LPIPS measures learned perceptual distance [59, 60]. In editing, correspondence is conditional: it is expected outside an appearance target, invalid in a newly exposed region, and geometry-dependent after motion. LLFF, DTU, Tanks and Temples, and Deep Blending provide established scene captures [61–64]; COLMAP provides reproducible camera estimation [65]. We build on these assets but add the support and visibility annotations needed to interpret change.

## 3. Editing as Controlled Change

## 3.1. Problem definition

We write a scene as ${ \cal S } = ( G , A , M , L , Z )$ , where G denotes geometry, A appearance, M material, L illumination, and Z semantics. An editor receives instruction c and optional controls u and produces $S ^ { \prime } = E ( S , c , u )$ . Each benchmark item specifies the intended factor subset $T \subseteq \{ G , A , M , L , Z \}$ and spatial support Ω. A successful edit changes the intended components inside Ω and keeps independent components stable:

$$
D _ { t } ( S , S ^ { \prime } ) > 0 \forall t \in T , \qquad D _ { j } ( S , S ^ { \prime } ) \approx 0 \forall j \notin T .\tag{1}
$$

Equation 1 is a contract, not an additive score. The amount of desired change depends on the instruction, while the appropriate invariances depend on the edit family. A rigid translation should change geometry and visibility without changing material. A recoloring should change appearance while retaining geometry and illumination. Removal necessarily alters occupancy and reveals content, so original pixels cannot be used as references everywhere.

## 3.2. Visibility-aware target support

For evaluation camera v, the original and edited scenes produce colors $I _ { v } , I _ { v } ^ { \prime }$ , depths $d _ { v } , d _ { v } ^ { \prime } .$ , and projected target masks $M _ { v } , M _ { v } ^ { \prime }$ . We retain two masks because geometry edits can change silhouette and visibility. Their union $U _ { v } = M _ { v } \cup M _ { v } ^ { \prime }$ contains pixels that may legitimately change. Pixels visible only after the edit form a disocclusion set $H _ { v }$ . They are excluded from before-after preservation comparisons and evaluated against a held-out target when such a target exists.

The 3D support is constructed by lifting corrected masks through scene depth and checking visibility in other annotated views. A point receives a positive vote only when its projected depth agrees with the rendered depth within a scene-normalized tolerance. This avoids copying a foreground label onto an occluded background surface. The support stores an uncertainty band around boundaries and regions observed in too few views. Scores are computed on the confident interior and exterior; uncertainty coverage is reported instead of being silently assigned to either region.

## 3.3. Why four dimensions

Fidelity and preservation are not opposites. A no-op has perfect preservation and zero fidelity; a global redraw may have high fidelity and poor preservation. Locality and preservation also differ: locality measures where pixels change, while preservation additionally measures whether non-target geometry and stable appearance are retained. Consistency is orthogonal to both because the same off-target change can be reproduced consistently from every view. These examples rule out a single before-after distance as an adequate definition.

## 4. EditBench3D

## 4.1. Scene and edit collection

The benchmark contains 18 synthetic objects, 12 forwardfacing captured scenes, and 10 bounded multiview scenes. Synthetic objects provide controlled geometry and appearance targets. Forward-facing scenes stress partial coverage and uncertain disocclusion. Bounded captures provide wider baselines and stronger cross-view checks. Each of the 40 scenes receives six edits, for 240 scene-edit pairs. Twentyfour cameras are evaluated per pair: 12 cameras overlapping the reconstruction trajectory and 12 held out from editor optimization. The resulting test set contains 5,760 edited render targets.

Table 1. Edit families and the invariances used by the benchmark.
<table><tr><td>Family</td><td>Intended change</td><td>Primary invariance</td></tr><tr><td>Appearance</td><td>albedo/color</td><td>geometry, illumination</td></tr><tr><td>Material/style</td><td>reflectance/texture</td><td>silhouette, background</td></tr><tr><td>Rigid motion</td><td>object pose</td><td>object appearance</td></tr><tr><td>Deformation</td><td>local geometry</td><td>non-target structure</td></tr><tr><td>Insert/remove</td><td>occupancy</td><td>visible surroundings</td></tr></table>

We organize edits into five families (Table 1). Appearance substitution changes a named object’s color or texture. Material and style edits alter reflectance or local visual character while retaining silhouette. Rigid edits translate, rotate, or uniformly scale one object. Deformation edits modify shape through a mesh or point proxy. Insertion and removal change occupancy and are evaluated with explicit disocclusion masks. The benchmark is not intended to reduce all scene editing to local edits; global style and weather transformations are reported in a separate track because spatial locality is not diagnostic for them.

## 4.2. Annotations and prompts

An annotation contains the target identity, factor label, canonical instruction, two paraphrases, a description of the original state, source-view masks, fused 3D support, per-camera visibility, and uncertainty. Three annotators participate in mask construction. The first marks the object in three separated views, the second corrects propagated masks, and the third resolves topology changes and disocclusions. Items with pairwise mask IoU below 0.75 after correction are excluded.

Prompt variation is controlled rather than optimized per method. One instruction is concise, one uses a relation to identify the target, and one is descriptive without adding a new style attribute. The three prompts specify the same edit. A negative instruction describes the existing scene state and tests whether an editor changes a scene unnecessarily. Prompt strings and evaluation cameras are fixed before running any method.

## 4.3. Fidelity

Fidelity is evaluated inside $U _ { v }$ . We use paired descriptions $c ^ { + }$ and $c ^ { - }$ for the edited and original state and compute direction-aware semantic gain

$$
F _ { \mathrm { s e m } } = \frac { 1 } { | V | } \sum _ { v \in V } \left[ s ( I _ { v } ^ { \prime } \odot U _ { v } , c ^ { + } ) - s ( I _ { v } ^ { \prime } \odot U _ { v } , c ^ { - } ) \right]\tag{2}
$$

Raw gains are calibrated to [0, 1] on synthetic exact edits and no-op controls. The paired form reduces credit for generic compatibility with the object category. Synthetic edits additionally use reference LPIPS and geometric distance where appropriate. Those diagnostic values are not merged into $F .$

## 4.4. Locality

Let $Q _ { v } \ = \ 1 - \mathrm { d i l a t e } ( U _ { v } , r )$ denote the confidently nontarget region, excluding disocclusion and uncertainty. The outside change is

$$
S _ { \mathrm { o u t } } = \frac { \sum _ { v , p } Q _ { v } ( p ) \rho ( I _ { v } ^ { \prime } ( p ) - I _ { v } ( p ) ) } { \sum _ { v , p } Q _ { v } ( p ) + \epsilon } ,\tag{3}
$$

where $\rho$ is a robust channel-averaged penalty. We map it to $L = \exp ( - S _ { \mathrm { o u t } } / \tau _ { L } )$ using a fixed calibration scale. A second boundary score uses only the angularly normalized ring immediately outside $U _ { v }$ . This prevents a narrow but visible halo from disappearing in the average over a large image.

## 4.5. Cross-view consistency

Visible 3D points are reprojected between overlapping heldout views using edited depth. Forward-backward depth checks remove occlusions. Patch-level appearance error is computed at valid correspondences:

$$
C = 1 - \frac { 1 } { | \mathcal { P } | } \sum _ { ( v , w , p ) \in \mathcal { P } } \mathrm { L P I P S } _ { \mathrm { p a t c h } } \left( I _ { v } ^ { \prime } , p ; I _ { w } ^ { \prime } , \pi _ { v \to w } ( p ) \right) .\tag{4}
$$

We separate a low-order view-dependent component estimated from the original scene so that legitimate specular response is not automatically counted as inconsistency. The score is calibrated with exact synthetic targets, consistent blur controls, and independently perturbed views.

## 4.6. Preservation and entanglement

Preservation combines appearance and geometry outside the target:

$$
\begin{array} { r } { P = \frac { 1 } { 2 } \exp ( - S _ { \mathrm { o u t } } / \tau _ { P } ) + \frac { 1 } { 2 } \exp ( - D _ { \mathrm { g e o m } } ^ { \bar { \Omega } } / \tau _ { G } ) . } \end{array}\tag{5}
$$

The geometry term uses depth for captured scenes and surface distance for synthetic scenes. Edit entanglement is retained as a vector $\mathbf { e } _ { t } = [ D _ { j } ( S , S ^ { \prime } ) ] _ { j \neq t }$ rather than collapsed into a scalar. For example, appearance-depth entanglement identifies a color edit that alters shape even when its target crop is semantically correct.

## 4.7. Metric sanity checks

We generate controlled corruptions from each exact synthetic target: an outside-only tint, a boundary halo, view-specific shuffled texture, an under-edit, and global blur. The expected partial order is fixed before method evaluation. Outside tint must reduce L and $P$ without improving $F ;$ shuffled texture must reduce $C ;$ under-edit must reduce $F ;$ global blur cannot receive an artificial consistency advantage. We also verify that scores remain stable when the same surface is sampled by additional cameras. These checks determine metric settings; they are not counted as editor test samples.

## 5. Evaluation Protocol

## 5.1. Methods and common inputs

We evaluate CLIP-NeRF, Instruct-NeRF2NeRF, Blending-NeRF, CE3D, GaussianEditor, GaussCtrl, InterGSEdit, and direct proxy editing. The set spans latent control, iterative dataset updates, layered fields, a representation-agnostic atlas, explicit Gaussian selection, joint multiview diffusion, geometry-consistent attention, and deterministic proxy manipulation. It is representative rather than exhaustive.

All editors receive identical camera calibration, source images, prompts, and target annotations. NeRF methods start from a common nerfacto-style reconstruction, and Gaussian methods start from a common 3DGS reconstruction. Editor-specific components follow their official settings. Each stochastic editor is run with three fixed seeds; Table 2 reports the mean over prompts, seeds, scenes, and valid held-out views. Unsupported operations are excluded from within-family aggregation and retained in a separate coverage record. Failed runs remain in the denominator using the lowest valid score for that scene.

## 5.2. Representation controls

Pre-edit reconstruction quality can masquerade as editing quality. Every method is compared with its own unedited rendering for locality and preservation, while fidelity uses the shared edited target or paired language description. Views whose original reconstruction falls below the preregistered threshold are excluded for every editor using that reconstruction, not only for the failing method. A secondary renderer check repeats compatible edits with Instant-NGP, standard 3DGS, 2DGS, and SuGaR [5, 8, 28, 29].

## 5.3. Aggregation

All four dimensions are oriented so that higher is better and calibrated to [0, 1]. We report them separately. Pareto dominance provides a weight-free summary: method a dominates b if it is no worse on every dimension and strictly better on at least one. A method is Pareto-optimal if no other method dominates it. We use the unweighted mean of $L , C , P$ only for the descriptive plot in Fig. 2; it is not a leaderboard score.

The unit of analysis is a scene-edit pair. Frames are not treated as independent observations. Human raters inspect a stratified subset of synchronized camera orbits to identify semantic or visibility failures that automatic metrics miss. Their judgments are an audit, not a fifth term tuned to reproduce the automatic ranking.

## 6. Results

## 6.1. Overall comparison

Table 2 reports the aggregate comparison. InterGSEdit obtains the highest fidelity, 0.78, followed by GaussCtrl at 0.76 and GaussianEditor at 0.73. This ordering is consistent with increasingly explicit multiview coordination in the evaluated Gaussian editors. It does not extend to every other dimension: GaussianEditor is more local than GaussCtrl (0.82 versus 0.75), and direct proxy editing exceeds all learned editors in locality, consistency, and preservation.

![](images/557f8af032b62d4e162b940bb7e37916f6766a1050f97ad5f03b658b11311df8.jpg)  
Figure 2. Measured fidelity versus the unweighted mean of locality, consistency, and preservation. The plot is descriptive; the three non-fidelity axes remain separate in all primary analyses.

The direct proxy baseline is especially informative. Its fidelity is only 0.59, below six of the seven language-guided systems, but its locality, consistency, and preservation are 0.92, 0.91, and 0.89. A deterministic geometric operator precisely controls support and view correspondence but cannot express open-ended semantic changes as well as a diffusionguided editor. This is not a weak baseline; it exposes a different operating point.

At the other extreme, Instruct-NeRF2NeRF reaches 0.71 fidelity but records the lowest locality, consistency, and preservation: 0.55, 0.73, and 0.57. Iterative 2D supervision follows the instruction, yet dense and view-dependent updates can overwrite unrelated content. Blending-NeRF sacrifices fidelity relative to Instruct-NeRF2NeRF (0.62 versus 0.71) but improves locality by 0.21 and preservation by 0.17. Separating an editable field from the frozen scene changes the preservation trade-off more than the semantic score alone suggests.

## 6.2. Representation-level comparison

Table 3 groups the three NeRF-only and three 3DGS-only editors. The 3DGS group improves the mean on all dimensions. The largest absolute gains are fidelity and locality, both 0.16. Preservation rises from 0.640 to 0.783, and consistency from 0.767 to 0.833. Relative to the NeRF group, these correspond to gains of 26.8%, 25.4%, 8.7%, and 22.4%, respectively.

These values should not be read as a controlled causal estimate of representation alone: the methods use different optimization and supervision strategies. Nevertheless, two mechanisms are compatible with the pattern. First, explicit Gaussians allow target selection and parameter freezing at primitive level. Second, Gaussian editors in the comparison contain stronger multiview controls. EditBench3D reports the group difference while keeping the causal interpretation modest.

Table 2. EditBench3D results over 240 scene-edit pairs. Higher is better. Bold and underlined values denote the best and second-best result
<table><tr><td>Editor</td><td>Representation</td><td>Fidelity  $F$ </td><td>Locality  $L$ </td><td>Consistency  $C$ </td><td>Preservation  $P$ </td></tr><tr><td>CLIP-NeRF [12]</td><td>NeRF</td><td>0.46</td><td>0.58</td><td>0.79</td><td>0.61</td></tr><tr><td>Instruct-NeRF2NeRF [13]</td><td>NeRF</td><td>0.71</td><td>0.55</td><td>0.73</td><td>0.57</td></tr><tr><td>Blending-NeRF [14]</td><td>NeRF</td><td>0.62</td><td>0.76</td><td>0.78</td><td>0.74</td></tr><tr><td>CE3D [51]</td><td>NeRF/3DGS</td><td>0.67</td><td>0.70</td><td>0.81</td><td>0.72</td></tr><tr><td>GaussianEditor [15]</td><td>3DGS</td><td>0.73</td><td>0.82</td><td>0.76</td><td>0.80</td></tr><tr><td>GaussCtrl [16]</td><td>3DGS</td><td>0.76</td><td>0.75</td><td>0.86</td><td>0.76</td></tr><tr><td>InterGSEdit [19]</td><td>3DGS</td><td>0.78</td><td>0.80</td><td>0.88</td><td>0.79</td></tr><tr><td>Direct proxy edit [42]</td><td>mesh/NeRF</td><td>0.59</td><td>0.92</td><td>0.91</td><td>0.89</td></tr></table>

Table 3. Mean results by representation group. The comparison averages three NeRF and three 3DGS editors; it is descriptive rather than causal.
<table><tr><td>Group</td><td> $F$ </td><td>L</td><td>C</td><td> $P$ </td></tr><tr><td>NeRF (3)</td><td>0.597</td><td>0.630</td><td>0.767</td><td>0.640</td></tr><tr><td>3DGS (3)</td><td>0.757</td><td>0.790</td><td>0.833</td><td>0.783</td></tr><tr><td> $\Delta$ </td><td>+0.160</td><td>+0.160</td><td>+0.067</td><td>+0.143</td></tr></table>

CE3D occupies an intermediate point, with 0.67 fidelity, 0.70 locality, 0.81 consistency, and 0.72 preservation. Its atlas interface decouples editing from the 3D representation, but this flexibility does not automatically provide the primitive-level locality of GaussianEditor or the joint multiview constraints of InterGSEdit. The result supports evaluating the full pipeline rather than assigning a fixed editability score to a representation class.

## 6.3. The dimensions are not interchangeable

Table 4 shows Spearman correlations across the eight methods. Fidelity is nearly uncorrelated with consistency $( \rho =$ 0.05) and only weakly correlated with locality and preservation. A high semantic score therefore provides little information about whether the edit is spatially controlled or stable across views.

Locality and preservation are strongly correlated $( \rho =$ 0.98), which is expected because outside-region appearance contributes to both. They are not duplicates. Preservation also contains non-target geometry and remains meaningful when a small color difference hides a structural change. Consistency correlates moderately with locality and preservation (0.50 and 0.57): methods that avoid global rewrites tend to be more stable, but the relation is far from deterministic.

Table 4. Spearman rank correlation across the eight methods.
<table><tr><td></td><td> $F$   $L$ </td><td> $C$ </td><td> $P$ </td></tr><tr><td> $F$ </td><td>1.00 0.14</td><td>0.05</td><td>0.24</td></tr><tr><td> $L$ </td><td>0.14</td><td>1.00 0.50</td><td>0.98</td></tr><tr><td> $C$ </td><td>0.05</td><td>0.50 1.00</td><td>0.57</td></tr><tr><td> $P$ </td><td>0.24</td><td>0.98 0.57</td><td>1.00</td></tr></table>

The correlation directly tests the paper’s motivation. If fidelity were strongly coupled to the other dimensions, a factorized benchmark would add reporting overhead without changing conclusions. The observed ranks show the opposite. Instruct-NeRF2NeRF ranks fourth in fidelity but last in three other dimensions; direct proxy editing ranks secondlowest in fidelity but first in the other three. Any one-number evaluation must choose which behavior it values.

## 6.4. Pareto analysis

Only GaussianEditor, InterGSEdit, and direct proxy editing are non-dominated. GaussianEditor remains because its locality of 0.82 exceeds InterGSEdit’s 0.80, despite lower scores elsewhere. InterGSEdit supplies the strongest generative operating point, with the best fidelity and second-best consistency. Direct proxy editing supplies the conservative operating point, leading the three preservation-oriented dimensions.

InterGSEdit strictly dominates GaussCtrl on all four axes. It also dominates CE3D and Blending-NeRF. GaussCtrl dominates CE3D; GaussianEditor, GaussCtrl, and InterGSEdit each dominate Instruct-NeRF2NeRF. CLIP-NeRF is dominated by CE3D, GaussCtrl, InterGSEdit, and direct proxy editing. These relationships are invariant to positive metric weights. Ranking the three Pareto methods, however, depends on application preferences.

For asset authoring, locality and reversibility may justify a direct proxy even at lower language fidelity. For rapid open-ended editing, InterGSEdit is the stronger choice in this comparison. GaussianEditor is attractive when local control matters but a language-driven appearance change is still required. A benchmark should expose these choices, not erase them with an arbitrary weighted sum.

## 6.5. What the results say about design

Three design patterns emerge. First, explicit target control matters. Blending-NeRF improves over dense NeRF updating on locality and preservation, and GaussianEditor further benefits from assigning a region of interest to primitives. Second, multiview coordination matters independently of representation. GaussCtrl and InterGSEdit achieve the strongest consistency among learned editors. Third, stronger generative supervision is not free. The best fidelity score coexists with measurable preservation loss.

These are empirical associations, not universal laws. The method set is small, and later NeRF editors may close the group gap. Still, the results offer actionable diagnostics. An editor below the Pareto front should first identify which failure dimension causes domination. Improving an already strong fidelity score is of limited value if locality makes the method strictly worse than another editor. Conversely, optimizing only preservation can converge to a no-op. The four axes specify where progress is needed.

## 6.6. Method profiles

Table 5 gives the rank of every editor on each dimension. The changes in rank are at least as important as the absolute values. CLIP-NeRF is last in fidelity but fifth in consistency. Its low semantic alignment should not be confused with unstable rendering. Instruct-NeRF2NeRF has the opposite profile: fourth in fidelity but eighth in the three remaining columns. The method follows open-ended instructions substantially better than CLIP-NeRF, but dense iterative updates provide weak control over what survives.

Blending-NeRF ranks fourth in locality and fifth in preservation. Its explicit separation between the original and editable fields is reflected most clearly outside the target, rather than in fidelity. CE3D ranks near the middle on all four axes. This balanced profile is consistent with a flexible editing interface, although it does not reach the leading specialized method on any dimension.

The three Gaussian methods also differ. GaussianEditor ranks second in locality and preservation but only seventh in consistency. Its primitive-level region control is effective at limiting collateral change, while view coordination remains its main weakness in this comparison. GaussCtrl moves from fifth in locality to third in consistency, matching its emphasis on multiview diffusion. InterGSEdit ranks first in fidelity and second in consistency. Its locality rank is lower than GaussianEditor’s, showing that stronger cross-view coupling does not automatically sharpen the spatial boundary.

Direct proxy editing has the clearest specialization: it ranks first in locality, consistency, and preservation, but seventh in fidelity. The method reliably executes a restricted operator; it does not infer the broad visual intent available to a diffusion model. This profile is precisely why task coverage and the four scores must be read together.

Table 5. Per-dimension ranks (1 is best) derived from Table 2. Rank changes reveal different failure profiles.
<table><tr><td>Method</td><td>F</td><td>L</td><td>C</td><td>P</td></tr><tr><td>CLIP-NeRF</td><td>8</td><td>7</td><td>5</td><td>7</td></tr><tr><td>Instruct-NeRF2NeRF</td><td>4</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Blending-NeRF</td><td>6</td><td>4</td><td>6</td><td>5</td></tr><tr><td>CE3D</td><td>5</td><td>6</td><td>4</td><td>6</td></tr><tr><td>GaussianEditor</td><td>3</td><td>2</td><td>7</td><td>2</td></tr><tr><td>GaussCtrl</td><td>2</td><td>5</td><td>3</td><td>4</td></tr><tr><td>InterGSEdit</td><td>1</td><td>3</td><td>2</td><td>3</td></tr><tr><td>Direct proxy</td><td>7</td><td>1</td><td>1</td><td>1</td></tr></table>

## 7. Benchmark Validity

## 7.1. Construct validity

The benchmark is designed around observable failure modes rather than a claim that one learned metric captures human judgment. Fidelity uses a language-image model only within the target and relative to the original-state description. Locality uses before-after change only where change is not requested. Consistency uses cross-view correspondences rather than independent view quality. Preservation includes geometry so an editor cannot preserve color while deforming the background. Each dimension therefore has a distinct operational meaning.

The controlled corruptions serve as unit tests for these meanings. A global tint and a local target edit can have similar text alignment, but only the former should reduce locality. Repeating one erroneous texture in every view can be consistent but semantically wrong. Applying the correct edit in only one view can have high single-view fidelity but low cross-view consistency. Blurring every view should not be rewarded simply because low-frequency images reproject easily. A metric configuration is accepted only after these orderings hold on synthetic controls.

Some dependence remains. Locality and preservation share outside-region appearance, which helps explain their rank correlation of 0.98. We retain both because locality answers a spatial question, whereas preservation also covers geometry and can be analyzed by factor. In a future benchmark version, the geometry and appearance components of P should always accompany the aggregate preservation number.

## 7.2. Internal validity

Four safeguards reduce confounding. First, every editor receives the same source images, prompts, cameras, and target supports. Second, all NeRF methods share an initial reconstruction and all Gaussian methods share another. Third, locality and preservation compare an edited model with its own pre-edit rendering, preventing initial reconstruction differences from being labeled as edit damage. Fourth, prompt variants and random seeds are fixed before evaluation rather than selected from successful examples.

These safeguards do not make the representation comparison causal. The editors contain different selection mechanisms and diffusion constraints. A strict representation experiment would hold the edit signal and optimizer fixed while swapping only the scene representation. Our group averages answer a narrower question: how do representative released NeRF and 3DGS editing systems compare under a common output protocol? The paper states this scope explicitly.

Manual control is another potential confound. Direct proxy editing requires an object proxy and a specified transformation, whereas text-guided methods infer both target and effect. We do not hide this difference inside fidelity. The benchmark records the input modality, interaction time, and supported edit families. Direct proxy editing demonstrates the upper end of controllability for a restricted operator, not a drop-in replacement for open-ended language editing.

## 7.3. External validity

The three scene groups cover different reconstruction regimes, but they do not represent every deployment. Synthetic objects provide ground truth yet have simpler materials. Forward-facing captures contain natural appearance but incomplete geometry. Bounded multiview scenes strengthen correspondence while remaining smaller than street-scale environments. Scores should therefore be reported by scene group when new methods are added, even though the present paper centers the aggregate result.

The five edit families cover common local operations but not global relighting, weather, or cinematic stylization. Those tasks require different preservation definitions. A global style edit may legitimately change every pixel but should retain geometry and semantic identity. Extending EditBench3D to such cases should disable spatial locality and introduce factor-specific preservation rather than assigning every method a trivially low locality score.

Temporal scenes pose a further challenge. A 4D edit must remain consistent across viewpoint and time, and the target support can deform or become occluded. Dynamic NeRF and Gaussian representations [66–70] provide appropriate testbeds, but temporal identity requires a fifth dimension beyond the static protocol studied here.

## 8. Implications for Editor Design

Separate selection from synthesis. The results favor systems that identify a target in 3D before applying a strong generative prior. GaussianEditor gains locality from a primitivelevel region of interest; Blending-NeRF gains preservation by separating original and editable fields. A practical editor should expose its selection confidence and decline to update ambiguous regions. Better synthesis cannot repair a target mask that includes the wall.

Coordinate views before optimizing 3D. Instruct-NeRF2NeRF shows that a shared 3D representation does not, by itself, resolve inconsistent 2D supervision. The scene can average disagreements into blur or view-dependent artifacts. GaussCtrl and InterGSEdit obtain higher consistency by coordinating the edited views or their attention. Future systems should treat multiview agreement as a property of the edit signal, not only as an eventual consequence of reconstruction.

Preserve an explicit reference state. Local editing benefits from a frozen reference, whether it is an original field, an atlas, unselected Gaussians, or a proxy. Regularizing against the initial rendering is useful but insufficient around occlusions, because the correct pixel correspondence changes after motion or removal. Reference preservation should operate on 3D support and visibility rather than on a fixed 2D mask.

Optimize for a target operating point. No method must maximize all axes for every application. A film asset pipeline may prefer direct manipulation with strong locality and a narrow operation vocabulary. A consumer editing tool may accept modest collateral change for better language fidelity. A robotics simulator may prioritize geometry preservation and deterministic consistency. Publishing the four-dimensional profile allows these choices without declaring one universal winner.

## 9. Discussion

## 9.1. Why not one editability score?

A weighted score w $F + w _ { L } L + w _ { C } C + w _ { P } P$ is easy to rank but difficult to justify. Equal weights treat a 0.01 change in a calibrated CLIP direction as commensurate with a 0.01 change in geometric preservation. Learned weights inherit the preferences of a particular rater pool. More importantly, weights can reverse the order of the three Pareto methods. We therefore publish the vector and Pareto front as primary outcomes. Users may apply domain-specific utilities afterward, provided that component scores remain visible.

## 9.2. Failure accounting

Editing systems fail in different ways. A prompt may not identify the target; a 2D editor may change the wrong attribute; multiview updates may disagree; geometry may drift; or optimization may collapse. Unsupported tasks are not equivalent to failed attempts. We retain a separate coverage matrix and failure log so a specialized geometry editor is not penalized as though it attempted free-form stylization, while a method that claims support but produces an empty scene remains accountable.

Human inspection is most useful for assigning these causes. Raters view synchronized original and edited camera orbits rather than isolated preferred frames. They answer separate questions about instruction success, off-target change, and cross-view flicker. This audit guards against systematic metric errors, but it does not replace the four quantitative dimensions with a preference vote.

## 9.3. Reproducibility

The benchmark records camera matrices, linearized images, masks, 3D supports, prompts, editor configurations, random seeds, raw per-view metrics, wall time, peak memory, and failed runs. Aggregation begins from scene-edit pairs; the large number of rendered frames is not used to inflate the sample size. Method-specific manual input, such as masks or handles, is recorded as interaction time. Versioned annotations and fixed test prompts prevent silent benchmark drift.

## 10. Limitations and Ethical Considerations

No automatic metric proves semantic correctness. CLIPbased fidelity inherits language and data biases. Segmentation errors can make locality optimistic by absorbing spill into an oversized target. Depth-based correspondence is unreliable on transparent, reflective, or very thin surfaces. Our paired descriptions, mask adjudication, uncertainty regions, and human audit reduce these problems but do not remove them.

The aggregate table does not establish that 3DGS is causally more editable than NeRF. The compared methods differ in supervision, publication date, optimization budget, and interface. The representation averages are evidence about the tested systems, not all possible systems. Similarly, the high correlation between locality and preservation is based on eight method-level observations and should be interpreted descriptively.

The benchmark emphasizes bounded edits with identifiable support. Global relighting, weather changes, and artistic stylization require different invariances; forcing them through a locality score would be misleading. Dynamic scenes introduce temporal consistency and identity preservation beyond our static-camera protocol [66–70]. These remain natural extensions.

Scene editing can be used to falsify visual evidence or modify identifiable people and private environments. Benchmark releases should contain only licensed or consented captures, retain edit provenance, and avoid prompts that create defamatory or sensitive content. Evaluation code should preserve metadata connecting an edited scene to its source and method configuration.

## 11. Conclusion

EditBench3D evaluates neural scene editing as controlled change rather than unconstrained generation. Across 240 scene-edit pairs, instruction fidelity is weakly related to locality, consistency, and preservation; 3DGS-based editors are stronger on average in this comparison; and the best semantic and conservative operating points belong to different methods. InterGSEdit leads fidelity, direct proxy editing leads the three preservation-oriented dimensions, and GaussianEditor joins them on the Pareto front through stronger locality. A good edited view is not sufficient evidence of a good 3D edit. Progress should be reported as a profile of what changed, where it changed, whether it agrees across views, and what survived.

## References

[1] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision, pages 405–421, 2020. 1, 2

[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5855–5864, 2021. 1, 2

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5470–5479, 2022.

[4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-NeRF: Anti-aliased gridbased neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697– 19705, 2023. 2

[5] Thomas Muller, Alex Evans, Christoph Schied, and Alexan-¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4):102:1–102:15, 2022. 2, 5

[6] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5501–5510, 2022.

[7] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial radiance fields. In European Conference on Computer Vision, pages 333–350, 2022. 1, 2

[8] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139:1– 139:14, 2023. 1, 2, 5

[9] Shuangkang Fang, I Shen, Xuanyang Zhang, Zesheng Wang, Yufeng Wang, Wenrui Ding, Gang Yu, Takeo Igarashi, et al.

Dropping anchor and spherical harmonics for sparse-view gaussian splatting. arXiv preprint arXiv:2602.20933, 2026.

[10] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676–21685, 2024. 3

[11] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, and Wenming Yang. VastGaussian: Vast 3d gaussians for large scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5166–5175, 2024. 1

[12] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. CLIP-NeRF: Text-and-image driven manipulation of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3835–3844, 2022. 1, 2, 6

[13] Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-NeRF2NeRF: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19740–19750, 2023. 1, 3, 6

[14] Hyeonseop Song, Seokhun Choi, Hoseok Do, Chul Lee, and Taehyeong Kim. Blending-NeRF: Text-driven localized editing in neural radiance fields. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14383– 14393, 2023. 1, 3, 6

[15] Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and Qi Tian. Gaussianeditor: Editing 3d gaussians delicately with text instructions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20902– 20911, 2024. 1, 3, 6

[16] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip H. S. Torr, and Victor Adrian Prisacariu. Gaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing. In European Conference on Computer Vision, 2024. 1, 3, 6

[17] Steven Liu, Xiuming Zhang, Zhoutong Zhang, Richard Zhang, Jun-Yan Zhu, and Bryan Russell. Editing conditional radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5773–5783, 2021. 1, 2

[18] Chunyi Sun, Yanbin Liu, Junlin Han, and Stephen Gould. NeRFEditor: Differentiable style decomposition for 3d scene editing. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7306–7315, 2024. 3

[19] Minghao Wen, Shengjie Wu, Kangkan Wang, and Dong Liang. Intergsedit: Interactive 3d gaussian splatting editing with 3d geometry-consistent attention prior. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26136–26145, 2025. 3, 6

[20] Yufeng Wang, Shuangkang Fang, Huayu Zhang, Hongguang Li, Zehao Zhang, Xianlin Zeng, and Wenrui Ding. Uavenerf: Text-driven uav scene editing with neural radiance

fields. IEEE Transactions on Geoscience and Remote Sensing, 62:1–14, 2024. 1

[21] Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, and Paul Debevec. Baking neural radiance fields for real-time view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5875–5884, 2021. 2

[22] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. PlenOctrees for real-time rendering of neural radiance fields. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5752–5761, 2021.

[23] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. KiloNeRF: Speeding up neural radiance fields with thousands of tiny MLPs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14335– 14345, 2021. 2

[24] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelNeRF: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578–4587, 2021. 2

[25] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. MVSNeRF: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14124–14133, 2021. 2

[26] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19447–19456, 2024. 2

[27] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20923–20931, 2024. 2

[28] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, 2024. 2, 5

[29] Antoine Guedon and Vincent Lepetit. SuGaR: Surface- ´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5354–5363, 2024. 5

[30] Shuangkang Fang, Dacheng Qi, Weixin Xu, Yufeng Wang, Zehao Zhang, Xiaorong Zhang, Huayu Zhang, Zeqi Shao, and Wenrui Ding. Efficient implicit sdf and color reconstruction via shared feature field. In Asian Conference on Computer Vision, pages 250–267. Springer, 2024.

[31] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics, 43(6):271:1–271:13, 2024.

[32] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. Rade-GS: Rasterizing depth in gaussian splatting. ACM Transactions on Graphics, 45(2):19:1–19:14, 2026. 2

[33] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. In Advances in Neural Information Processing Systems, volume 37, pages 80965–80986, 2024. 2

[34] Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus Steinberger. StopThePop: Sorted gaussian splatting for view-consistent real-time rendering. ACM Transactions on Graphics, 43(4):64:1–64:17, 2024.

[35] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Hengshuang Zhao. Pixel-GS: Density control with pixel-aware gradient for 3d gaussian splatting. In European Conference on Computer Vision, pages 326–342, 2024.

[36] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719– 21728, 2024.

[37] Simon Niedermayr, Josef Stumpfegger, and Rudiger West-¨ ermann. Compressed 3d gaussian splatting for accelerated novel view synthesis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10349–10358, 2024. 2

[38] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-GS: Structured 3d gaussians for view-adaptive rendering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024. 2

[39] Marko Mihajlovic, Sergey Prokudin, Siyu Tang, Robert Maier, Federica Bogo, Tony Tung, and Edmond Boyer. SplatFields: Neural gaussian splats for sparse 3d and 4d reconstruction. In European Conference on Computer Vision, pages 313–332, 2024.

[40] Yijia Xie, Yuhang Lin, Laijian Li, Lina Liu, Xiaobin Wei, Yong Liu, and Jiajun Lv. Hash-GS: Anchor-based 3d gaussian splatting with multi-resolution hash encoding for efficient scene reconstruction. In IEEE International Conference on Robotics and Automation, pages 13964–13971, 2025.

[41] Haithem Turki, Vasu Agrawal, Samuel Rota Bulo, Lorenzo\` Porzi, Peter Kontschieder, Deva Ramanan, Michael Zollhofer,¨ and Christian Richardt. HybridNeRF: Efficient neural rendering via adaptive volumetric surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19647–19656, 2024. 2

[42] Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma, Rongfei Jia, and Lin Gao. NeRF-editing: Geometry editing of neural radiance fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18353–18364, 2022. 2, 6

[43] Jun-Kun Chen, Jipeng Lyu, and Yu-Xiong Wang. Neuraleditor: Editing neural radiance fields via manipulating point clouds. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12439–12448, 2023. 2

[44] Shuangkang Fang, Weixin Xu, Heng Wang, Yi Yang, Yufeng Wang, and Shuchang Zhou. One is all: Bridging the gap

between neural radiance fields architectures with progressive volume distillation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 597–605, 2023. 3

[45] Shuangkang Fang, Yufeng Wang, Yi Yang, Weixin Xu, Heng Wang, Wenrui Ding, and Shuchang Zhou. Progressive volume distillation with active learning for efficient NeRF architecture conversion. International Journal ofComputer Vision, 134(5):228, 2026. 3

[46] Lukas Radl, Michael Steiner, Andreas Kurz, and Markus Steinberger. LAENeRF: Local appearance editing for neural radiance fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4969– 4978, 2024. 3

[47] Ka Chun Shum, Jaeyeon Kim, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Language-driven object fusion into neural radiance fields with pose-conditioned dataset updates. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5176–5187, 2024. 3

[48] Shuangkang Fang, I-Chao Shen, Takeo Igarashi, Yufeng Wang, Zesheng Wang, Yi Yang, Wenrui Ding, and Shuchang Zhou. NeRF is a valuable assistant for 3d gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26230–26240, 2025. 3

[49] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 3

[50] Shuangkang Fang, Yufeng Wang, Yi-Hsuan Tsai, Yi Yang, Wenrui Ding, Shuchang Zhou, and Ming-Hsuan Yang. Editing 3d scenes via text prompts without retraining. IEEE Transactions on Visualization and Computer Graphics, 32(7):5386– 5400, 2026. 3

[51] Shuangkang Fang, Yufeng Wang, Yi-Hsuan Tsai, Yi Yang, Wenrui Ding, Shuchang Zhou, and Ming-Hsuan Yang. Chatedit-3d: Interactive 3d scene editing via text prompts. In European Conference on Computer Vision, pages 199–216, 2024. 3, 6

[52] Shuangkang Fang, Yufeng Wang, Yi-Hsuan Tsai, Wenrui Ding, Yi Yang, Shuchang Zhou, and Ming-Hsuan Yang. Chatedit-3d++: Interactive 3d and 4d scene editing via large language models. arXiv preprint arXiv:2608.29137, 2026. 3

[53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 3

[54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3

[55] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF

International Conference on Computer Vision, pages 4015– 4026, 2023. 3

[56] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729–19739, 2023. 3

[57] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051–20060, 2024.

[58] Shuangkang Fang, I Shen, Yufeng Wang, Yi-Hsuan Tsai, Yi Yang, Shuchang Zhou, Wenrui Ding, Takeo Igarashi, Ming-Hsuan Yang, et al. Meshllm: Empowering large language models to progressively understand and generate 3d mesh. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14061–14072, 2025. 3

[59] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. 3

[60] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. 3

[61] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics, 38(4):29:1–29:14, 2019. 3

[62] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-scale data for multiple-view stereopsis. International Journal of Computer Vision, 120(2):153–168, 2016.

[63] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4):78:1– 78:13, 2017.

[64] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. ACM Transactions on Graphics, 37(6):257:1–257:15, 2018. 3

[65] Johannes L. Schonberger and Jan-Michael Frahm. Structure-¨ from-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4104–4113, 2016. 3

[66] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural radiance fields for dynamic scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318–10327, 2021. 8, 9

[67] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5865–5874, 2021.

[68] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20310–20320, 2024.

[69] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20331–20341, 2024.

[70] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8508–8520, 2024. 8, 9