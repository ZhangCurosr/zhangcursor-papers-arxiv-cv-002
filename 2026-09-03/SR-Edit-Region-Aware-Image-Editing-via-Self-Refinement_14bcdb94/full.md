# SR-Edit: Region-Aware Image Editing via Self-Refinement

Andong Wang<sup>1,2</sup> , Zehua Chen<sup>1</sup> <sup>⋆</sup>, Yuxuan Jiang<sup>1</sup> , and Jun Zhu<sup>1</sup> <sup>⋆</sup>

<sup>1</sup> Tsinghua University, Beijing, China

<sup>2</sup> Renmin University of China, Beijing, China

andong000@ruc.edu.cn {zhc23thuml,dcszj}@mail.tsinghua.edu.cn

Abstract. With the recent rapid progress in generative models, image editing has made remarkable advances, yet achieving faithful edits that precisely modify only the target regions while strictly preserving all other regions remains challenging. Since externally provided region annotations are often dificult to obtain in practice, a growing body of work seeks to improve preservation by automatically inferring edit and non-edit regions, and then enforcing consistency on the latter. However, these approaches still sufer from inaccurate region estimation and heuristic correction strategies that distort the native inference process, making methods designed for fidelity themselves a new source of artifacts. We propose SR-Edit, an image editing framework that overcomes these issues via iterative self-refinement. Specifically, at each iteration, SR-Edit first (i) extracts progressively precise and self-consistent region separation from the model’s own predictions by lightweight post-processing, and then (ii) enforces preservation in non-edit areas through correction updates that remain aligned with the original sampling dynamics. Extensive experiments demonstrate that SR-Edit achieves superior preservation and overall image quality compared to existing editing techniques.

Keywords: Image Editing · Generative Models · h-transform theory

## 1 Introduction

Recent progress in generative models [20,34,37,49,55] has greatly expanded the capabilities of image editing. A variety of editing paradigms have been proposed, including editing with forward noising followed by conditional denoising [41], training-based editors that learn an explicit mapping from a source image and a textual editing instruction to the edited image [5, 36, 62, 63, 66], and inversionbased methods that invert an image into a difusion trajectory and then modify the denoising process to achieve the edit [19, 28, 38, 44, 64, 72].

Despite impressive results, a persistent bottleneck remains: faithful editing, i.e., producing the desired modifications while strictly preserving all other content. In real-world edits, users typically expect backgrounds, identity cues, fine textures, and geometric details outside the target concept to remain unchanged. However, most generative editing pipelines still rely on sampling dynamics that start from a noisy state and evolve under a new condition, which inherently operates on global image content [41]. As a result, even when the semantic edit succeeds, the output may exhibit unintended drift in non-edited regions, such as subtle background changes, texture inconsistencies, or identity degradation.

![](images/c8e85abae3c30af42f82b8f9ef4c8afc138147ad85946bcf2be369e426ddeff9.jpg)  
Fig. 1: Examples of edited images produced by SR-Edit. The edit regions are automatically inferred by the model and indicated by red overlays.

A common strategy to improve preservation is to explicitly separate the image into edit and non-edit regions, and then enforce consistency on the latter. When accurate user-provided masks are available, spatially constrained editing can be efective [1, 39, 52], but manual annotation is costly and often unavailable in practice. This motivates mask-free approaches that infer editable regions automatically, for example by manipulating or interpreting attention maps [18,19,46,53,56] or by computing diferences between generation trajectories under source and target prompts [11, 38].

Nevertheless, automatic region estimation remains challenging. The inferred masks are often insuficiently precise, failing to align with true pixel boundaries, and can also be temporally unstable across sampling steps [6]. Moreover, many preservation mechanisms rely on heuristic interventions such as feature fusion and reuse, or KV [57] interpolation in intermediate representations, which are not consistent with the original sampling dynamics [38, 47]. Consequently, the techniques introduced above to improve fidelity do not reliably enhance preservation. Instead, they can become a new source of imprecision and artifacts.

We propose SR-Edit, a self-refined image editing framework that addresses both failure modes: inaccurate region estimation and heuristic preservation corrections. The key idea is to use the model’s own output as self-feedback to construct a reliable pixel-space separation between edited and preserved content, and then to enforce preservation through an update rule that remains aligned with intrinsic sampling dynamics.

Concretely, SR-Edit leverages the model’s predictions to build self-consistent diference maps and converts them into a precise edit and non-edit decomposition via lightweight pixel-space post-processing [17, 54]. Unlike latent-space discrimination, pixel-space diferences directly reflect decoded visual changes and admit mature refinement operations, yielding sharper and more stable region estimates.

![](images/443aa0921bc9331ba28d48d2c59b1eb76f120bdcba879caa0bf2f546b2dd86a7.jpg)  
Fig. 2: Overview of SR-Edit. The edit instruction of the case is Change the animal’s fur color to a darker shade. We use u<sub>θ</sub> to denote the sampling vector field that drives the generative trajectory. s denotes the start time of self-correction update while n denotes the time duration. N denotes the number of self-refinement iterations.

To enforce preservation, we adopt a principled conditioning mechanism based on Doob’s h-transform theory [4, 14]. We formulate preservation as a constraint on the non-edit region by modeling it as a vanishing-noise observation, and derive an additive guidance term that enhances fidelity while preserving the structure of the sampling procedure. Under a plug-in approximation, this guidance reduces to a masked residual on the model’s clean-image prediction, which can be injected into difusion samplers as an additive drift correction and analogously into flowmatching samplers as a velocity correction.

In contrast to heuristic state interventions (e.g., direct feature replacement or blending) [47, 57], our update is derived from a distribution transform and therefore tends to preserve native inference dynamics while suppressing non-edit drift. This complements existing methods that improve fidelity to the source image through more accurate inversion [23, 44, 58] and structural conditioning approaches that anchor geometry of the source via external controls [62, 69].

SR-Edit operates in an iterative self-correction loop: region separation is inferred from current predictions, preservation guidance is applied to suppress non-edit drift, and the model produces a progressively refined result across iterations. This self-refinement gradually improves region localization and preservation while maintaining edit intent. Extensive experiments demonstrate that SR-Edit yields superior non-edit preservation and improved overall visual quality compared to prior region-aware and mask-free editing techniques [11, 38, 47].

Our contributions can be summarized as follows:

– We introduce SR-Edit, an image editing framework that improves faithfulness by jointly enhancing region identification and preservation in an iterative self-refinement process.

– We propose self-consistent region identification derived from the model’s own predictions and show that pixel-space post-processing enables more precise and stable edit and non-edit separation than latent or attentionbased discrimination.

– We develop a Doob’s h-transform formulation for region preservation and derive a practical masked guidance rule that integrates naturally with both difusion and flow-matching samplers.

– We validate that SR-Edit improves non-edit preservation and overall image quality across diverse editing scenarios, outperforming existing editing techniques.

## 2 Related Work

## 2.1 Image Editing with Generative Model Backbones

Generative model backbones. Modern generative models, such as difusion models [20, 49, 55], flow matching [2, 34, 37], and bridge models [7, 10, 33, 71], have shown a strong capability of faithfully reconstructing a target distribution with learned time-dependent scores or vector fields. Given a scalable network architecture [35, 40, 49], these generative frameworks have been able to capture complex data distributions defined by large-scale datasets, enabling high-fidelity generation across data modalities, such as image [15, 49], audio [9, 25, 30], video [21, 43,60], or time-series signals [3,8,42]. In the inference process, these frameworks usually start from a prior distribution, e.g., Gaussian noise or a clean representation, and gradually generate the target with iterative refinement steps, showing a noise-to-data [12, 24, 59] or data-to-data sampling trajectory [32, 67].

Image editing paradigm. Following the remarkable breakthroughs of generative models, a growing body of research has focused on adapting these models for editing tasks [22,26]. Image editing methods can be roughly grouped by how they incorporate an input image and enforce edit intent. Forward-and-backward approaches inject noise into the input and denoise under new conditions, using the noise level to regulate the edit magnitude [41]. Training-based instruction editing methods learn an explicit mapping from the source image and textual instruction to the edited output through supervised or synthetic training pipelines, allowing direct and scalable manipulation [5, 36, 62, 63, 66]. Inversion-based methods first invert a real image into a difusion latent trajectory and then apply prompt changes while trying to preserve reconstruction [19, 28, 38, 44, 64, 72].

Preservation of the source. Preserving the source image during edits is still dificult because noisy initialization in generative backbones can induce global drift, causing unintended changes in irrelevant regions. Existing approaches improve preservation through several typical mechanisms. [23,44,58] achieve high-fidelity inversion and reconstruction, then the conditional sampling can start from a more faithful state and thus reduce global deviation. External or automatically estimated masks separate edit regions and non-edit regions, which explicitly improve preservation of the source [1, 11]. We will further discuss the method in Sec. 2.2. Manipulating attention features [57] or injecting intermediate features can also enforce spatial alignment between the source image and edited output [6, 19, 46, 56]. Moreover, external structural constraints can be conditioned on edges, poses, segmentation, or depth, providing an explicit control to preserve original structure even when appearance changes [62, 69].

![](images/2a9ee6fa8c4277856ff027c1588acf0eaeb38477add868117c5205d8f4bb3d44.jpg)

![](images/0456b1a3777d28b81652afe64b7dbad215a39327ce676e02ee0768f4cbfe7009.jpg)

![](images/bd85e175989e5007c9922aa08e828df40dc937eb53d8ef90b11e5be1ca528f50.jpg)  
Fig. 3: Quantitative evidence of model behavior. On edit cases and humanannotated region partitions from MagicBrush test split [68], editing model [63] shows a clear pattern: edit regions concentrate strong changes in a coherent area, while nonedit regions produce low-magnitude and weakly structured changes. Left: mean pixel change inside and outside the annotated region after applying a MAD-based threshold<sup>1</sup>; the inside-to-outside ratio increases from 2.32 to 7.71, indicating much higher change energy in edit regions. Center: number of connected components among pixels covering a top fraction of total change energy, extremely large counts (e.g., 1472 vs. 100 at 0.5 coverage) indicate that non-edit diferences are highly fragmented whereas edits remain spatially concentrated. Right: fraction of selected pixels contained in the largest connected component, further confirming stronger spatial concentration for edits (e.g., 64% vs. 29% at 0.5 coverage).

## 2.2 Region-Aware Image Editing

External mask. Accurate edit and non-edit region separation ofers an important pathway to faithful image editing. Early and widely-adopted approaches rely on binary masks provided by users to explicitly specify where edits should occur [1, 39, 52]. This design provides direct spatial control and often yields strong results when accurate masks are available. However, it requires manual annotation, which limits applicability in real-world usage. Moreover, the manual mask often has an inaccurate boundary, leading to incorrect modifications on irrelevant regions and therefore degrading edit faithfulness.

![](images/2f00e1c2236d54835395817f50777bd298d94453749c6499b7c82da40bc2ee2a.jpg)  
Fig. 4: Visualization of the incremental efects of each component in the SR-Edit post-processing pipeline for region identification.

Text-driven localization. To reduce reliance on external masks, subsequent work investigates inferring editable regions from text-conditioned signals alone. DifEdit and Follow-Your-Shape [11, 38] derive edit regions by computing diferences between inference processes conditioned on the source and target prompts. Image editing frameworks based on manipulation of cross-attention maps associate textual tokens with spatial regions, enabling localized edits without external masks [18, 19, 46, 53]. However, the editable region is inferred from semantic representations instead of pixel-level evidence, and the subsequent projection from semantic diferences to image region separation is only approximate, often resulting in imprecise and unstable localization [6].

Region identification via inference behavior. More recent methods aim to localize editable regions by directly leveraging the model’s inference behavior, which is naturally aligned with the editing process. SpotEdit [47] further explores region control from the inference trajectory by leveraging single-step reconstruction and performing linear interpolation between KV features [57] of intermediate latents and references. Although efective in practice, the single-step reconstruction is typically sensitive, while the reliance on KV-level masking or interpolation introduces heuristic engineering choices that could be refined toward more principled and precise localization.

## 3 SR-Edit

At inference time, we organize editing into a short stabilization stage followed by a stack of SR-Edit blocks. Each SR-Edit block performs a lightweight probe-andrefine cycle: (i) a few-step inference produces a provisional edit $\mathbf { I } _ { \mathrm { r e f } }$ , from which we infer the edit and non-edit regions via the pixel-space pipeline in Sec. 3.1; (ii) conditioned on this newly estimated region, we run a few guidance steps using the Doob’s h-transform term in Sec. 3.2 to enforce region preservation while continuing the edit. Stacking these blocks yields an iterative self-refinement process: the model repeatedly re-estimates where changes occur and re-applies principled preservation guidance, so the region identification is continually refined along with the prediction, progressively sharpening region separation for more accurate boundaries while keeping the edit flexible.

## 3.1 Precise Region Identification

Observation. Modern training-based image editors [2, 36, 63, 66] often demonstrate reasonably good fidelity in practice: meaningful modifications tend to concentrate in the intended edit region, while non-edit areas are largely preserved, with residual deviations typically small and weakly structured (Fig. 3). This empirical behavior makes the editor output itself a useful self-feedback cue for region separation: large and spatially coherent diferences indicate edited areas, whereas scattered low-amplitude diferences are more consistent with preserved regions.

Moreover, while many editing models operate in the latent space, performing region discrimination in decoded pixel space rather than latent space enables more precise region identification and allows us to leverage mature image post-processing techniques [17, 54] to suppress scattered artifacts, since latent features are typically patch-level and their variations are not a stable proxy for pixel changes due to highly nonlinear decoding [49]. Therefore, following prior practices [11,38,47] that exploit output-driven cues for refinement and motivated by the advantages of pixel-space analysis discussed above, we explicitly treat the model’s own prediction as a stable self-feedback signal in pixel space to infer edit regions.

Region identification pipeline. Given an input image $\mathbf { I } _ { \mathrm { i n } }$ and an edited image I<sub>ref</sub> produced by few-step inference, both defined on the same pixel grid $\varOmega =$ $\{ 1 , \ldots , H \} \times \{ 1 , \ldots , W \}$ with C channels, we estimate a binary change mask M : $\varOmega  \{ 0 , 1 \}$ , where $\mathbf { M } ( \mathbf { p } ) = 1$ indicates edited pixels and $\mathbf { M } ( \mathbf { p } ) = 0$ indicates non-edited pixels. The method involves four steps: (i) constructing a diference map, (ii) binarization using Otsu’s threshold, (iii) morphological refinement, and (iv) connected-component filtering. The incremental efects of each component can be viewed in Fig. 4.

Diference map. We compute a diference map $d ( p )$ by taking the per-pixel absolute diference between the input and reference images, averaging over the C channels:

$$
d ( p ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \left| \mathbf { I } _ { \mathrm { i n } } ( p , c ) - \mathbf { I } _ { \mathrm { r e f } } ( p , c ) \right| , \qquad p \in \varOmega .\tag{1}
$$

This map $d ( p )$ quantifies the local change at each pixel, with larger values indicating greater deviation between the images.

Otsu thresholding. To binarize d without manual tuning, we apply Otsu’s separation method [45]. Its key idea is to choose a single global threshold t that best separates the histogram of d into two classes (low diferences as unchanged

![](images/1440f050a42b72f2c75bdc1a1dc93c1102a676c0566bc5bf2b3acea63c525999.jpg)  
Fig. 5: Comparison on region identification between diferent methods. We include region separation results from manual annotations in MagicBrush [68], crossattention maps [57] on token with largest L2 energy, Follow-Your-Shape [38], SpotEdit [47] and SR-Edit. All evaluated methods use Qwen-Image-Edit 2511 [63] as the backbone.

and high diferences as changed). Specifically, for each candidate threshold $t ,$ Otsu defines class probabilities $\omega _ { 0 } ( t ) , \omega _ { 1 } ( t )$ and class means $\mu _ { 0 } ( t ) , \mu _ { 1 } ( t )$ from the histogram, and selects the threshold that maximizes the between-class variance:

$$
t ^ { \star } = \arg \operatorname* { m a x } _ { t } \ \omega _ { 0 } ( t ) \omega _ { 1 } ( t ) \big ( \mu _ { 0 } ( t ) - \mu _ { 1 } ( t ) \big ) ^ { 2 } .\tag{2}
$$

We then obtain the initial mask by thresholding: $\mathbf { M } _ { 0 } ( p ) = \mathbb { I } [ d ( p ) \geq t ^ { \star } ]$ , where $\mathbb { I } [ \cdot ]$ denotes the indicator function that returns 1 if the condition holds and 0 otherwise.

Morphological refinement. The initial mask ${ { \bf { M } } _ { 0 } }$ may contain small isolated artifacts and small holes or breaks inside true change regions. We refine it using binary morphology [17, 54]. Specifically, we use a small circular kernel on the pixel grid: we first apply opening with kernel size $s _ { \mathrm { o p e n } }$ to remove small foreground speckles, and then apply closing with size $s _ { \mathrm { c l o s e } }$ to fill small holes and connect narrow gaps. This morphological refinement yields a cleaner and more spatially coherent mask ${ { \bf { M } } _ { 1 } }$

Connected-component filtering. Finally, we run connected-component labeling on ${ { \bf { M } } _ { 1 } }$ [50]. Each connected component corresponds to a maximal set of foreground pixels that are mutually reachable through neighbor-to-neighbor steps. We remove components whose area (pixel count) falls below a minimum threshold $A _ { \operatorname* { m i n } } .$ , treating them as residual noise, and keep the remaining components as the final change mask M. Our method shows improved precision in region identification compared with prior approaches, as illustrated in Fig. 5.

## 3.2 Region Preservation via Doob’s h-transform

Our method achieves region preservation by conditioning the sampling dynamics through an additive Doob’s h-transform [4, 14] guidance term. In contrast to heuristic feature fusion such as KV reuse [57], the correction is derived from a distribution transform and thus more consistent with the underlying generative process.

Difusion model. We consider a continuous-time difusion model [20] defined by the forward SDE [55]

$$
\begin{array} { r } { d { \pmb x } _ { t } = f ( { \pmb x } _ { t } , t ) d t + g ( t ) d { \pmb w } _ { t } . } \end{array}\tag{3}
$$

Here ${ \pmb w } _ { t }$ is standard Brownian motion, f is the drift, and $g ( t )$ controls the noise scale. Let $p _ { t }$ denote the marginal density of $\mathbf { \Delta } \mathbf { x } _ { t }$ at time t. For conditional editing, we condition on a control signal C and write the corresponding marginal as $p _ { t } ( { \pmb x } \mid { \mathcal { C } } )$ . Its score $\nabla _ { \pmb { x } } \log p _ { t } ( \pmb { x } \mid \mathcal { C } )$ is the conditional log-density gradient that drives the reverse-time denoising dynamics.

Observation model and Doob function. Let M<sup>¯</sup> denote the complement of the mask defined earlier, and let $\pmb { y } \triangleq \bar { \mathbf { M } } \pmb { x } _ { \mathrm { s r c } }$ . We enforce the preservation constraint in the form $\bar { \bf M } { \bf x } _ { 0 } = { \bf y }$ . We encode preservation with the vanishing-noise observation model $\mathbf { \boldsymbol { Y } } = \bar { \mathbf { M } } \mathbf { \boldsymbol { x } } _ { 0 } + \boldsymbol { \xi } , \boldsymbol { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { \mathrm { o b s } } ^ { 2 } I )$ , and take $\sigma _ { \mathrm { o b s } }  0$ . The corresponding likelihood is $p ( \pmb { y } \mid \pmb { x } _ { 0 } ) = \mathcal { N } ( \pmb { y } ; \bar { \mathrm { M } } \bar { \pmb { x } _ { 0 } } , \sigma _ { \mathrm { o b s } } ^ { 2 } I )$ . Define the Doob function

$$
h ( t , \pmb { x } ) \triangleq \mathbb { E } [ p ( \pmb { y } \mid \pmb { x } _ { 0 } ) \mid \pmb { x } _ { t } = \pmb { x } , \mathcal { C } ] .\tag{4}
$$

$\mathrm { B y }$ the Doob’s h-transform, conditioning on $\textbf { \textit { Y } } = \textbf { \textit { y } }$ yields the factorization $p _ { t } ( { \pmb x } \mid { \pmb y } , { \mathcal C } ) \propto p _ { t } ( { \pmb x } \mid { \mathcal C } ) h ( t , { \pmb x } )$ , and therefore the conditional score decomposes additively as

$$
\nabla _ { \boldsymbol { x } } \log p _ { t } ( \boldsymbol { x } \mid \boldsymbol { y } , \boldsymbol { \mathcal { C } } ) = \nabla _ { \boldsymbol { x } } \log p _ { t } ( \boldsymbol { x } \mid \boldsymbol { \mathcal { C } } ) + \nabla _ { \boldsymbol { x } } \log h ( t , \boldsymbol { x } ) .\tag{5}
$$

Consequently, any reverse-time difusion sampler that uses the reference score can be made region-preserving by augmenting its score estimation with the guidance term $\nabla _ { \boldsymbol { \mathbf { x } } } \log h ( t , \boldsymbol { \mathbf { x } } )$

Plug-in and masked Gaussian guidance. Exact evaluation of $h ( t , { \pmb x } )$ is generally intractable. Let $\widehat { \pmb { x } } _ { 0 } ( { \pmb x } _ { t } )$ be the model prediction of the clean image at time t. We adopt a plug-in approximation that replaces the latent $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ inside the likelihood by $\widehat { \pmb { x } } _ { 0 } ( { \pmb x } _ { t } )$ , which gives $h ( t , \pmb { x } _ { t } ) \approx \mathcal { N } ( \pmb { y } ; \bar { \mathrm { M } } \hat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t } ) , \sigma _ { \mathrm { o b s } } ^ { 2 } I )$ . Equivalently, up to an additive constant independent of $\mathbf { \Delta } \mathbf { x } _ { t } .$

$$
\log h ( t , \mathbf { x } _ { t } ) \approx - \frac { 1 } { 2 \sigma _ { \mathrm { o b s } } ^ { 2 } } \left. \bar { \mathbf { M } } \big ( \widehat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } ) - \mathbf { x } _ { \mathrm { s r c } } \big ) \right. _ { 2 } ^ { 2 } .\tag{6}
$$

Diferentiating yields

$$
\nabla _ { x _ { t } } \log { h ( t , { \boldsymbol x } _ { t } ) } \approx - \frac { 1 } { \sigma _ { \mathrm { o b s } } ^ { 2 } } J _ { \widehat { \pmb x } _ { 0 } } ( { \pmb x } _ { t } ) ^ { \top } \bar { \mathbf { M } } \big ( \widehat { \pmb x } _ { 0 } ( { \pmb x } _ { t } ) - { \pmb x } _ { \mathrm { s r c } } \big ) ,\tag{7}
$$

where $J _ { \widehat { \pmb { x } } _ { 0 } } ( \pmb { x } _ { t } )$ denotes the Jacobian of $\widehat { \mathbf { x } } _ { 0 }$ with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . In implementation, the local sensitivity is absorbed into a time-dependent scalar schedule, which yields the practical masked-residual form

$$
\nabla _ { \pmb { x } _ { t } } \log h ( t , \pmb { x } _ { t } ) \approx - \lambda ( t ) \bar { \bf M } \big ( \hat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t } ) - \pmb { x } _ { \mathrm { s r c } } \big ) .\tag{8}
$$

Combining Equations (5) and (8) shows that, under the plug-in approximation, the h-transform guidance reduces to a masked residual on the model prediction $\widehat { \pmb { x } } _ { 0 } ( { \pmb x } _ { t } )$ . This residual can be viewed as a time-consistent error signal, so the correction acts as a conditional energy tilt toward preservation rather than a heuristic state intervention [4, 14, 31]. Since it enters only through the drift, it does not modify the difusion coeficient $g ( t )$ and therefore typically leaves the noise schedule and sampling structure largely intact. In the ideal case where the base conditional dynamics already match the desired conditioned process, the guidance term vanishes.

Preservation guidance for flow matching Flow models [34, 37] sample by the ODE $\begin{array} { r } { \frac { d \pmb { x } _ { t } } { d t } = \pmb { u } _ { \theta } ( \pmb { x } _ { t } , t ) } \end{array}$ and do not come with a canonical Doob’s h-transform tied to an SDE. After the same plug-in reduction, preservation is encoded by the quadratic energy [16] $\begin{array} { r } { E _ { t } ( \pmb { x } _ { t } ) \overset { \Delta } { = } \frac { 1 } { 2 \sigma _ { \mathrm { { s h e } } } ^ { 2 } } \| \bar { \mathbf { M } } ( \widehat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t } ) - \pmb { x } _ { \mathrm { s r c } } ) \| _ { 2 } ^ { 2 } } \end{array}$ , whose gradient yields the same masked residual direction. Because this guidance depends only on $\widehat { \pmb { x } } _ { 0 } ( { \pmb x } _ { t } )$ and not on a difusion coeficient, it can be injected as an additive velocity correction without changing the ODE form:

$$
\frac { d \boldsymbol { x } _ { t } } { d t } = \boldsymbol { u } _ { \theta } ( \boldsymbol { x } _ { t } , t ) - \gamma ( t ) \nabla _ { \boldsymbol { x } _ { t } } E _ { t } ( \boldsymbol { x } _ { t } ) \ \approx \ u _ { \theta } ( \boldsymbol { x } _ { t } , t ) - \gamma ( t ) \lambda ( t ) \bar { \mathbf { M } } \big ( \widehat { \boldsymbol { x } } _ { 0 } ( \boldsymbol { x } _ { t } ) - \boldsymbol { x } _ { \mathrm { s r c } } \big ) ,\tag{9}
$$

which retains the flow matching transport structure while encouraging $\bar { \mathbf { M } } \widehat { \mathbf { x } } _ { 0 } = \mathbf { y }$

## 4 Experiments

## 4.1 Dataset

We evaluate on ImgEdit-Bench, a benchmark derived from the ImgEdit dataset for instruction-based image editing [65]. Since our method focuses on spatially localized editing, we focus on the single-turn setting and consider editing tasks confined to a specific area of the image, including replace, adjust, background, remove, add, and action, resulting in 497 valid cases in total.

<table><tr><td rowspan="2">Method</td><td colspan="3">Preservation</td><td rowspan="2">Semantic Overall CLIP↑</td><td rowspan="2">Judge↑</td></tr><tr><td>SSIM↑</td><td>PSNR↑ DISTS↓LPIPS↓</td><td></td></tr><tr><td>InstructPix2Pix [5]</td><td>0.67</td><td>16.58 0.20</td><td>0.46</td><td>26.68</td><td>2.69</td></tr><tr><td>+SR-Edit (Ours)</td><td>0.68</td><td>16.94 0.18</td><td>0.43</td><td>25.07</td><td>2.75</td></tr><tr><td rowspan="2">AnyEdit [66] +SR-Edit (Ours)</td><td>0.70</td><td>18.95 0.17</td><td>0.41</td><td>25.15</td><td>2.82</td></tr><tr><td>0.85</td><td>19.19 0.13</td><td>0.35</td><td>25.21</td><td>2.99</td></tr><tr><td>Qwen-Image-Edit 2511 [63]</td><td>0.61</td><td>14.92 0.23</td><td>0.46</td><td>26.16</td><td>3.84</td></tr><tr><td>+Follow-Your-Shape [38]</td><td>0.61</td><td>14.96</td><td>0.23 0.47</td><td>26.03</td><td>3.59</td></tr><tr><td>+SpotEdit [47]</td><td>0.70</td><td>16.55</td><td>0.20 0.32</td><td>26.11</td><td>3.91</td></tr><tr><td>+SR-Edit (Ours)</td><td>0.67</td><td>17.40</td><td>0.18 0.31</td><td>26.80</td><td>3.94</td></tr><tr><td>Step1X-Edit v1p2 [36]</td><td>0.68</td><td>15.96</td><td>0.20</td><td>0.39 25.89</td><td>4.00</td></tr><tr><td>+Follow-Your-Shape</td><td>0.67</td><td>15.93</td><td>0.20 0.39</td><td>25.84</td><td>4.03</td></tr><tr><td>+SpotEdit</td><td>0.75</td><td>16.77 0.17</td><td>0.31</td><td>25.91</td><td>4.08</td></tr><tr><td>+SR-Edit (Ours)</td><td>0.77</td><td>16.48</td><td>0.14 0.27</td><td>26.09</td><td>4.01</td></tr></table>

Table 1: Comparison between diferent methods on ImgEdit-Bench. Bold indicates the best result for each metric within each backbone group, and underlining indicates the second best. For the first two backbone groups, we do not mark second best results. CLIP scores are reported after multiplication by 10<sup>2</sup>.

We additionally evaluate on a widely used image-editing benchmark PIE-Bench [27], which provides human-annotated edit masks together with descriptive captions. The masks enable separate evaluation of edited and non-edited regions, while the captions allow semantic consistency between the edited image and the intended content. Together, these annotations provide complementary and more fine-grained evaluation dimensions beyond ImgEdit-Bench. As above, we exclude style edits and viewpoint transformations that cannot be meaningfully localized, resulting in 578 cases for evaluation.

## 4.2 Backbone Editors

We test our method on four open-source backbones from two editor families. For difusion-based backbones, we include InstructPix2Pix, a standard instructionfollowing difusion editor [5], and AnyEdit, a strong unified difusion image editor that shows competitive editing performance across diverse instructions [66]. For flow-based editors, Qwen-Image-Edit 2511 and Step1X-Edit v1p2 are recent flow-style editors that are commonly adopted as modern backbones for general instruction-based editing [36, 63].

## 4.3 Baselines

We compare with two representative training-free region-aware methods that automatically infer editable regions and enforce preservation in non-edit areas during inference. Follow-Your-Shape uses a trajectory divergence map to localize editable regions and applies scheduled KV [57] injection to preserve non-target content during editing [38]. SpotEdit identifies editable regions via reconstruction-based stability estimation and preserves context using KV caching with interpolation from the source image [47].

## 4.4 Metrics

We report a CLIP-based text–image similarity score to reflect whether the edited result is semantically consistent with the instruction [48]. We report PSNR, SSIM, DISTS, and LPIPS between the edited image and the source image to measure preservation and fidelity, where PSNR and SSIM emphasize structure similarity, while DISTS and LPIPS capture perceptual distance [13, 61, 70]. We also include the ImgEdit judge-model score as an automatic evaluator that is designed to align with human preference for instruction-based editing [65].

For PIE-Bench, we report SSIM, LPIPS, and MSE over the human-annotated non-edit regions for region-specific preservation evaluation, together with Structure Distance (SD) [56] for structural preservation. We further report CLIP-Whole and CLIP-Edited to assess semantic consistency at the whole-image and edit-region levels, respectively.

## 4.5 Inference Configuration

Unless otherwise specified, we use the default settings provided by each baseline model and method. For SpotEdit [47] and Follow-Your-Shape [38], when the inference step count difers from the backbone editor’s setting, we linearly

<table><tr><td rowspan="2">Method</td><td colspan="4">Preservation</td><td colspan="2">Semantic</td></tr><tr><td>SSIM↑</td><td>MSE↓</td><td>SD↓</td><td>LPIPS↓</td><td> ${ \mathrm { C L I P } } _ { W } \uparrow$ </td><td>CLIPE↑</td></tr><tr><td>InstructPix2Pix [5]</td><td>0.76</td><td>2.26</td><td>5.93</td><td>15.57</td><td>29.30</td><td>25.89</td></tr><tr><td rowspan="2">+SR-Edit (Ours) Qwen-Image-Edit 2511 [63]</td><td>0.86</td><td>1.81</td><td>4.30</td><td>10.45</td><td>29.79</td><td>26.02</td></tr><tr><td>0.87</td><td>0.88</td><td>4.91</td><td>8.22</td><td>31.45</td><td>26.99</td></tr><tr><td>+SR-Edit (Ours)</td><td>0.94</td><td>0.72</td><td>4.50</td><td>4.69</td><td>31.46</td><td>26.90</td></tr></table>

Table 2: Additional quantitative results on PIE-Bench. SSIM, LPIPS, MSE are computed over human-annotated non-edit regions. SD measures overall structural preservation, and is computed on the entire image. $\mathrm { C L I P } _ { W }$ and $\mathrm { C L I P } _ { E }$ measure CLIP similarity of target text description with the whole image and the edited region, respectively. LPIPS, MSE, SD, and CLIP scores are reported after multiplication by $\mathrm { 1 0 ^ { 2 } }$

![](images/71054384fbee08025db7cb0be05d2552adfd53230e7647caa93da4a6c3b169bc.jpg)  
Fig. 6: Qualitative comparison. Many baselines (e.g. the default setting) show weaker preservation to the source image, and Follow-Your-Shape can yield insuficiently realistic edits with residual blur. SpotEdit often introduces boundary discontinuities (edge jumps). In contrast, our method is more faithful to the input and produces realistic, high-quality edits. The instructions of the two cases are Add a seagull perched on the edge of the wooden pier and Change the object color to a soft blue. All editing methods use Qwen-Image-Edit 2511 [63] as backbone.

rescale the editing schedule to match the new number of steps. For SR-Edit, we instantiate two iterations and set few-step inference stride to 3. The post-process parameters are set to $s _ { \mathrm { o p e n } } = 3 , s _ { \mathrm { c l o s e } } = 5 , A _ { \mathrm { m i n } } = 1 0 0 0$ . We set $\sigma _ { \mathrm { o b s } } = 3 \times 1 0 ^ { - 2 }$ and employ a linearly scheduled h-transform guidance strength. We enable selfrefinement starting at 30% of the model’s inference steps.

## 5 Results

## 5.1 Quantitative Results

As shown in Tab. 1, SR-Edit consistently improves source preservation across diferent backbone models while largely maintaining semantic alignment. For example, SSIM increases from 0.70 to 0.85 on AnyEdit and from 0.68 to 0.77 on Step1X-Edit v1p2, while LPIPS is reduced from 0.41 to 0.35 and from 0.39 to $0 . 2 7 .$ , respectively. On Qwen-Image-Edit 2511, SR-Edit also achieves the best PSNR, DISTS, and LPIPS while improving CLIP from 26.16 to 26.80, showing that stronger preservation does not generally weaken the intended edit.

Table 3: Ablation results on ImgEdit-Bench. All methods use Qwen-Image-Edit 2511 [63] as the backbone. Reconstruction uses the model’s one-step reconstruction output for region estimation, while Single Iter uses only one iteration, meaning the edit region is identified once and then kept fixed. Constant replaces the default linearly increasing preservation guidance with a constant strength, and Linear Down reverses the schedule by linearly decreasing the guidance strength over the correction process.
<table><tr><td rowspan="2">Method</td><td colspan="4">Preservation</td><td rowspan="2">Semantic CLIP↑</td><td rowspan="2">Overall Judge↑</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>DISTS↓</td><td>LPIPS↓</td></tr><tr><td>SR-Edit</td><td>0.67</td><td>17.40</td><td>0.18</td><td>0.31</td><td>26.80</td><td>3.94</td></tr><tr><td>Reconstruction</td><td>0.64</td><td>16.90</td><td>0.21</td><td>0.35</td><td>26.95</td><td>3.88</td></tr><tr><td>Single Iter</td><td>0.60</td><td>16.10</td><td>0.24</td><td>0.41</td><td>25.90</td><td>3.60</td></tr><tr><td>Constant</td><td>0.62</td><td>16.56</td><td>0.22</td><td>0.38</td><td>26.97</td><td>3.94</td></tr><tr><td>Linear Down</td><td>0.61</td><td>16.37</td><td>0.24</td><td>0.40</td><td>27.03</td><td>3.87</td></tr></table>

PIE-Bench provides more direct region-level evidence for this behavior. SR-Edit improves non-edit SSIM from 0.76 to 0.86 on InstructPix2Pix and from 0.87 to 0.94 on Qwen-Image-Edit, with clear reductions in LPIPS, MSE, and structural distance. Meanwhile, semantic scores are either improved or largely preserved, indicating that SR-Edit mainly suppresses unintended changes outside the target region rather than simply producing more conservative edits.

![](images/99da0551aa27295ea6544ccca58c39128fd48f48d0c8be2d5ffdd37523de34ce.jpg)  
Fig. 7: Comparison between few-step inference and reconstruction. Few-step inference is better at almost all steps.

<table><tr><td>Setting</td><td>P</td><td>R</td><td>IoU</td><td>F1</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  Otsu</td><td>0.10</td><td>1.00</td><td>0.10</td><td>0.16</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  Open</td><td>0.69</td><td>1.00</td><td>0.69</td><td>0.76</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  Close</td><td>1.00</td><td>0.91</td><td>0.91</td><td>0.95</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  CC</td><td>0.76</td><td>1.00</td><td>0.76</td><td>0.84</td></tr></table>

Table 4: Ablation results on postprocessing pipeline of region identification. The results indicate that each component contributes in a distinct and complementary manner. For example, removing the opening module may increase recall, as fewer constraints allow more positive predictions, but this comes at the expense of precision and overall balance. These variations confirm that each module plays a specific role in improving the overall accuracy and robustness.

Table 5: Runtime and memory analysis. “Time” denotes the average inference time (s) over 100 randomly sampled cases from our ImgEdit-Bench evaluation dataset, and “VRAM” denotes the peak GPU memory usage (GB) during inference. All analyses are performed using a single NVIDIA RTX PRO 6000 GPU. “Vanilla” refers to the model’s original inference. For each model, the best and second-best results among the applicable editing methods are highlighted in bold and underlined, respectively. Vanilla results are provided as baselines and excluded from the ranking.
<table><tr><td rowspan="2">Model</td><td colspan="4">Vanilla Follow-Your-Shape SpotEdit</td><td colspan="2"></td><td colspan="2">SR-Edit</td></tr><tr><td>Time VRAM</td><td></td><td>Time</td><td>VRAM</td><td></td><td></td><td>Time VRAM Time VRAM</td><td></td></tr><tr><td>InstructPix2Pix</td><td>2.49</td><td>3.55</td><td></td><td>Not Applicable</td><td></td><td></td><td>3.41</td><td>3.55</td></tr><tr><td>AnyEdit</td><td>9.84</td><td>12.93</td><td></td><td></td><td></td><td></td><td></td><td>13.49 12.93</td></tr><tr><td>Qwen-Image-Edit 2511</td><td>53.69</td><td>59.94</td><td>104.37</td><td>62.89</td><td></td><td>24.6066.35</td><td>75.31</td><td>59.94</td></tr><tr><td>Step1X-Edit v1p2</td><td>70.81</td><td>42.41</td><td>139.01</td><td>44.88</td><td>26.7644.77</td><td></td><td>97.45</td><td>42.41</td></tr></table>

## 5.2 Qualitative Results

Fig. 6 shows that our method achieves higher fidelity to the input and more realistic, higher-quality edits than competing methods.

## 6 Ablation Study

Few-step inference versus reconstruction for region estimation. We compare fewstep inference under diferent strides with reconstruction in terms of regionidentification quality (Fig. 7). Overall, few-step inference consistently outperforms reconstruction in both IoU and F1. Moreover, reconstruction shows noticeable instability (e.g., a sharp drop around step 32 in both metrics), whereas the few-step curves remain smooth. Among the few-step settings, stride-2 yields the best performance; stride-3 is slightly lower but close; and stride-4 tends to underperform, as recall degrades more substantially with increasing steps. In practical self-refined inference, using regions inferred via reconstruction results in less faithful edits (Tab. 3).

Iterative refinement. Next, we ablate the number of iterative self-refinement blocks. Using two SR-Edit blocks improves region alignment, as the mask and preservation guidance are re-estimated from progressively refined predictions. With a single block, the mask is fixed early and is typically over-inclusive and inaccurate, which weakens non-edit preservation and may also reduce instruction adherence by over-constraining the model (Tab. 3). These results highlight the importance of iterative mask re-estimation for simultaneously maintaining preservation and edit expressiveness.

Region post-processing pipeline. We ablate each post-processing component against the full pipeline output to examine its role in region identification. Tab. 4 indicates that the four steps are complementary: removing Otsu produces an almost non-informative mask with extremely low precision; removing opening increases false positives and lowers precision; removing closing harms spatial coherence and reduces overlap, and removing connected-component filtering retains noisy fragments and degrades overall balance. Together, these results demonstrate the distinct yet complementary roles of the individual components.

Input  
![](images/e230e2a7cb19eefe43ab6cc8f0a43568e8616be64a23a0b1f373899da65431d5.jpg)

Edited  
![](images/035295df799d830ddb1f44e8db3261031fbb663a10f817500e6c98591fc4c431.jpg)

![](images/824e6064257625eb3c589efe023c79b71e636dcb33819207fee2eff51fe077eb.jpg)

![](images/baa955b43997efe1618dc5338eb20bb9c81285e226ed0f70b795bc45a5661ada.jpg)  
Fig. 8: SR-Edit sometimes introduces unintended blending between the preserved content and the target edit. The first example shows a dark artifact in the upper-right corner, while the second produces a hybrid cabin–tent structure due to incomplete replacement. The edit instructions for the two cases are Change the blurred environment in the background to an autumn forest with orange and yellow leaves on the trees and Replace the wooden cabin in the image with a large camping tent.

Guidance schedule. As shown in Tab. 3, the linearly increasing schedule consistently improves preservation over Constant and Linear Down, with minor variation in semantic and overall quality. One key factor underlying this trend is the decreasing uncertainty of $p ( \pmb { x } _ { 0 }$ $\mathbf { \Psi } _ { \pmb { x } _ { t } , \pmb { \mathcal { C } } } )$ : at noisy steps, its larger covariance yields a smaller inverse covariance and thus weaker efective preservation, while the increasing precision later in sampling

strengthens the correction. The increasing schedule we adopt captures this tendency and is therefore better aligned with the actual theoretical correction.

## 7 Computational Eficiency Analysis

As shown in Tab. 5, SR-Edit introduces negligible additional memory overhead, with peak VRAM usage remaining essentially unchanged from the vanilla setting. While it introduces a moderate increase in inference time, this additional cost is generally acceptable considering the substantial improvement in source fidelity. More importantly, SR-Edit can be used to construct large-scale, high-fidelity editing pairs, allowing its source-preserving capability to be distilled into the base model. This provides a practical path toward retaining the benefits of SR-Edit while eliminating the additional sampling overhead at inference time.

## 8 Scope and Limitations

We currently focus on localized editing, which is the primary target setting of SR-Edit. Our main observed failures arise when preservation interferes with the intended edit semantics: (1) over-estimating non-edit region, leading to abnormal mix between guidance and edit modification (Fig. 8). (2) preservation may still be slightly weaker than hard replacement, as our soft correction cannot fully reset drifted non-edit latents, allowing residual interference to propagate into the edited region and leading to semantic failures. As stronger preservation is the central goal of this work, we will continue refining the design to improve preservation while maintaining the integrity of the intended edit semantics.

## 9 Conclusion

In this paper, we present SR-Edit, a region-aware image editing method that performs faithful edits without external masks. It uses iterative self-refinement to infer and sharpen edit regions from pixel-space predictions, and applies a Doob’s h-transform–based masked residual correction to improve preservation for difusion and flow-based backbones. Experiments across multiple editors and metrics show better non-edit preservation and visual quality than representative baselines while maintaining instruction consistency. Ablation studies further verify the benefits of few-step probing for robust region estimation and the necessity of iterative mask re-estimation for balancing preservation and edit expressiveness. Overall, SR-Edit ofers a simple, training-free framework that can be applied to modern backbones to reduce unintended drift during inference.

## 10 Acknowledgment

This work is supported by the National Natural Science Foundation of China (62550004, U24A20342, U25B6003, 92570001).

## References

1. Avrahami, O., Lischinski, D., Fried, O.: Blended difusion for text-driven editing of natural images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18208–18218 (2022)

2. Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., et al.: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints pp. arXiv–2506 (2025)

3. Bolton, A., Zhou, W., Chen, Z., Iacovides, G., Mandic, D.: Refinebridge: Generative bridge models improve financial forecasting by foundation models. In: ICASSP (2026)

4. Bortoli, V.D., Thornton, J., Heng, J., Doucet, A.: Difusion schrödinger bridge with applications to score-based generative modeling. In: Beygelzimer, A., Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems (2021)

5. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18392–18402 (2023)

6. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 22560– 22570 (2023)

7. Chen, Z., He, G., Zheng, K., Tan, X., Zhu, J.: Schrodinger bridges beat difusion models on text-to-speech synthesis. arXiv preprint arXiv:2312.03491 (2023)

8. Chen, Z., Miao, Y., Wang, L., Fan, L., Mandic, D.P., Zhu, J.: Versatile cardiovascular signal generation with a unified difusion transformer. Nature Machine Intelligence 8(1), 6–19 (2026)

9. Chen, Z., Tan, X., Wang, K., Pan, S., Mandic, D., He, L., Zhao, S.: Infergrad: Improving difusion models for vocoder by considering inference in training. In: ICASSP (2022)

10. Chen, Z., Yang, Y., Yuan, B., Zheng, K., Liu, J.S., Zhu, J.: Guidedbridge: Trainingfreely improving bridge models with prior guidance. In: ICML (2026)

11. Couairon, G., Verbeek, J., Schwenk, H., Cord, M.: Difedit: Difusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427 (2022)

12. Dai, Y., Chen, Z., Jiang, Y., Ke, Q., Cai, J., Zhu, J.: Omni2sound: Towards unified video-text-to-audio generation. In: CVPR (2026)

13. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44(5), 2567–2581 (2020)

14. Doob, J.L.: Conditional brownian motion and the boundary limits of harmonic functions. Bulletin de la Société mathématique de France 85, 431–458 (1957)

15. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Rombach, R.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML (2024)

16. Feng, R., Yu, C., Deng, W., Hu, P., Wu, T.: On the guidance of flow matching. arXiv preprint arXiv:2502.02150 (2025)

17. Goyal, M.: Morphological image processing. IJCST 2(4), 59 (2011)

18. Helbling, A., Meral, T.H.S., Hoover, B., Yanardag, P., Chau, D.H.: Conceptattention: Difusion transformers learn highly interpretable features. arXiv preprint arXiv:2502.04320 (2025)

19. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)

20. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

21. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video difusion models. arXiv preprint arXiv:2204.03458 (2022)

22. Huang, Y., Huang, J., Liu, Y., Yan, M., Lv, J., Liu, J., Xiong, W., Zhang, H., Cao, L., Chen, S.: Difusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

23. Jeong, D., Kang, D., Park, J., Lee, H., Paik, J.: Structure-preserving zero-shot image editing via stage-wise latent injection in difusion models. arXiv preprint arXiv:2504.15723 (2025)

24. Jiang, Y., Chen, Z., Ju, Z., Dai, Y., Dou, W., Zhu, J.: Controlaudio: Tackling textguided, timing-indicated and intelligible audio generation via progressive difusion modeling. In: ACL (2026)

25. Jiang, Y., Chen, Z., Ju, Z., Li, C., Dou, W., Zhu, J.: Freeaudio: Training-free timing planning for controllable long-form text-to-audio generation. In: ACM MM (2025)

26. Jiang, Y., Han, M., Dai, Y., Wang, A., Zhou, T., Ye, J., Wang, D., Shi, H., Li, B., Song, J., Yu, C., Zheng, B., Dou, W., Chen, Z., Zhu, J.: Freesonic: Trainingfree temporal-aware decoupled attention for precise audio editing. In: Interspeech (2026)

27. Ju, X., Zeng, A., Bian, Y., Liu, S., Xu, Q.: Direct inversion: Boosting difusionbased editing with 3 lines of code. arXiv preprint arXiv:2310.01506 (2023)

28. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6007–6017 (2023)

29. Law, J.: Robust statistics—the approach based on influence functions (1986)

30. Leng, Y., Chen, Z., Guo, J., Liu, H., Chen, J., Tan, X., Mandic, D., He, L., Li, X.Y., Qin, T., Zhao, S., Liu, T.Y.: Binauralgrad: A two-stage conditional difusion probabilistic model for binaural audio synthesis. In: NeurIPS (2022)

31. Léonard, C.: A survey of the schr\" odinger problem and some of its connections with optimal transport. arXiv preprint arXiv:1308.0215 (2013)

32. Li, C., Chen, Z., Bao, F., Zhu, J.: Bridge-sr: Schrödinger bridge for eficient sr. In: ICASSP (2025)

33. Li, C., Chen, Z., Wang, L., Zhu, J.: Audio super-resolution with latent bridge models. In: NeurIPS (2025)

34. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)

35. Liu, H., Chen, Z., Yuan, Y., Xinhao, M., Liu, X., Mandic, D., Wang, W., Plumbley, M.: Audioldm: Text-to-audio generation with latent difusion models. In: ICML (2023)

36. Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., et al.: Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761 (2025)

37. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)

38. Long, Z., Zheng, M., Feng, K., Zhang, X., Liu, H., Yang, H., Zhang, L., Chen, Q., Ma, Y.: Follow-your-shape: Shape-aware image editing via trajectory-guided region control. arXiv preprint arXiv:2508.08134 (2025)

39. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising difusion probabilistic models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11461– 11471 (2022)

40. Ma, N., Goldstein, M., Albergo, M.S., Bofi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and difusion-based generative models with scalable interpolant transformers. In: ECCV (2024)

41. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic diferential equations. In: ICLR (2022)

42. Miao, Y., Chen, Z., Li, C., Mandic, D.: Respdif: An end-to-end multi-scale rnn difusion model for respiratory waveform estimation from ppg signals. In: ICASSP (2025)

43. Mo, S., Chen, Z., Bao, F., Zhu, J.: Difgap: A lightweight difusion module in contrastive space for bridging cross-model gap. In: ICASSP (2025)

44. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6038–6047 (2023)

45. Otsu, N., et al.: A threshold selection method from gray-level histograms. Automatica 11(285-296) (1979)

46. Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot imageto-image translation. In: ACM SIGGRAPH 2023 conference proceedings. pp. 1–11 (2023)

47. Qin, Z., Tan, Z., Wang, Z., Liu, S., Wang, X.: Spotedit: Selective region editing in difusion transformers. arXiv preprint arXiv:2512.22323 (2025)

48. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

49. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

50. Rosenfeld, A., Pfaltz, J.L.: Sequential operations in digital picture processing. Journal of the ACM (JACM) 13(4), 471–494 (1966)

51. Rousseeuw, P.J., Croux, C.: Alternatives to the median absolute deviation. Journal of the American Statistical association 88(424), 1273–1283 (1993)

52. Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., Taigman, Y.: Emu edit: Precise image editing via recognition and generation tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8871–8879 (2024)

53. Simsar, E., Tonioni, A., Xian, Y., Hofmann, T., Tombari, F.: Lime: Localized image editing via attention regularization in difusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 222–231. IEEE (2025)

54. Soille, P., et al.: Morphological image analysis: principles and applications, vol. 2. Springer (1999)

55. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic diferential equations. In: ICLR (2021)

56. Tumanyan, N., Geyer, M., Bagon, S., Dekel, T.: Plug-and-play difusion features for text-driven image-to-image translation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1921–1930 (2023)

57. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

58. Wallace, B., Gokul, A., Naik, N.: Edict: Exact difusion inversion via coupled transformations. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22532–22541 (2023)

59. Wang, J., Chen, Z., Yuan, B., Zheng, K., Li, C., Jiang, Y., Zhu, J.: Audiomog: Guiding audio generation with mixture-of-guidance. In: ICME (2026)

60. Wang, Y., Chen, Z., Chen, X., Zhu, J., Chen, J.: Framebridge: Improving imageto-video generation with bridge models. In: ICML (2025)

61. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

62. Wei, C., Xiong, Z., Ren, W., Du, X., Zhang, G., Chen, W.: Omniedit: Building image editing generalist models through specialist supervision. In: The Thirteenth International Conference on Learning Representations (2024)

63. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)

64. Yan, Z., Ma, Y., Zou, C., Chen, W., Chen, Q., Zhang, L.: Eedit: Rethinking the spatial and temporal redundancy for eficient image editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17474–17484 (2025)

65. Ye, Y., He, X., Li, Z., Lin, B., Yuan, S., Yan, Z., Hou, B., Yuan, L.: Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275 (2025)

66. Yu, Q., Chow, W., Yue, Z., Pan, K., Wu, Y., Wan, X., Li, J., Tang, S., Zhang, H., Zhuang, Y.: Anyedit: Mastering unified high-quality image editing for any idea. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 26125–26135 (2025)

67. Zhang, C., Chen, Z., Zheng, K., Zhu, J.: Voicebridge: Designing latent bridge models for general speech restoration at scale. arXiv preprint arXiv:2509.25275 (2025)

68. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36, 31428–31449 (2023)

69. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3836–3847 (2023)

70. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

71. Zhou, L., Lou, A., Khanna, S., Ermon, S.: Denoising difusion bridge models. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 (2024)

72. Zhu, T., Zhang, S., Shao, J., Tang, Y.: Kv-edit: Training-free image editing for precise background preservation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16607–16617 (2025)

# SR-Edit: Region-Aware Image Editing via Self-Refinement – Supplementary Material

## 1 Pseudo Code of Post-Processing and Self-Refinement

We provide the pseudo code of the proposed SR-Edit framework below. It consists of a pixel-space post-processing pipeline for region identification and an iterative self-refinement inference procedure.

Algorithm 1 Edit Region Identification via Pixel-Space Post-Processing   
Require: Input image $\mathbf { I } _ { \mathrm { i n } } \ \in \ \mathbb { R } ^ { H \times W \times C } ;$ provisional edited image $\mathbf { I } _ { \mathrm { r e f } } \ \in \ \mathbb { R } ^ { H \times W \times C } ;$   
opening size $s _ { \mathrm { o p e n } } ;$ closing size $s _ { \mathrm { c l o s e } } ;$ minimum area A<sub>min</sub>   
1: for all $p \in \varOmega$ do   
2: Compute the diference map $\begin{array} { r } { d ( p ) \gets \frac { 1 } { C } \sum _ { c = 1 } ^ { C } | \mathbf { I } _ { \mathrm { i n } } ( p , c ) - \mathbf { I } _ { \mathrm { r e f } } ( p , c ) | } \end{array}$   
3: end for   
4: Compute the Otsu threshold [45] t<sup>⋆</sup> ← arg max $\omega _ { 0 } ( t ) \omega _ { 1 } ( t ) \big ( \mu _ { 0 } ( t ) - \mu _ { 1 } ( t ) \big ) ^ { 2 }$   
5: for all $p \in \varOmega$ do   
6: $\mathbf { M } _ { 0 } ( p )  \mathbb { I } [ d ( p ) \geq t ^ { \star } ]$   
7: end for   
8: Construct circular structuring elements [17, 54] $B _ { \mathrm { o p e n } }$ and $B _ { \mathrm { c l o s e } }$   
9: $\mathbf { M } _ { 1 } \gets \mathrm { O p e n i n g } ( \mathbf { M } _ { 0 } , B _ { \mathrm { o p e n } } )$   
10: $\mathbf { M } _ { 1 }  \mathrm { C l o s i n g } ( \mathbf { M } _ { 1 } , B _ { \mathrm { c l o s e } } )$   
11: Compute connected components [50] $\{ \mathcal { C } _ { k } \} _ { k }$ of ${ { \bf { M } } _ { 1 } }$   
12: Initialize $\mathbf { M } ( p )  0$ for all $p \in \varOmega$   
13: for all connected components $\mathcal { C } _ { k }$ do   
14: if Area $( \mathcal { C } _ { k } ) \ge A _ { \operatorname* { m i n } }$ then   
15: for all $p \in \mathcal { C } _ { k }$ do   
16: $\mathbf { M } ( p )  1$   
17: end for   
18: end if   
19: end for   
20: return M

Algorithm 1 summarizes the former, including diference-map construction, thresholding [45], morphological refinement [17, 54], and connected-component filtering [50], while Algorithm 2 presents the latter for both difusion and flow backbones. Together, they provide a concise summary of the overall pipeline and practical guidance for implementation.

## 2 Additional Details

Quantitative analysis. For the analysis in Fig. 3, we use the human-annotated masks from MagicBrush test split [68] and further manually select only samples

Algorithm 2 Unified SR-Edit Inference for Difusion and Flow Backbones   
Require: Input image $\mathbf { I } _ { \mathrm { i n } } ;$ editing condition $\mathcal { C } ;$ terminal time $T ;$ self-correction start   
time s; correction duration n; number of self-refinement blocks $N ;$ probe stride K   
1: Encode the input image into latent $\scriptstyle { \mathbf {  { x } } } _ { \mathrm { { r c } } }$ and initialize ${ \pmb x } _ { T }$   
2: Run stabilization inference from $T$ to s   
3: for $i = 0$ to $N - 1$ do   
4: I. Few-step probe for region estimation   
5: Starting from latent state at $s - i \cdot n ,$ run few-step inference with stride K and   
decode the latent prediction to provisional image $\mathbf { I } _ { \mathrm { r e f } }$   
6: Compute pixel-space mask M = RegionIdentification $( \mathbf { I } _ { \mathrm { i n } } , \mathbf { I } _ { \mathrm { r e f } } )$   
7: Downsample the mask to latent space: M<sub>latent</sub> = Downsample(M)   
8: II. Self-correction update   
9: Define the non-edit mask as $\bar { \mathbf { M } } _ { \mathrm { l a t e n t } } = 1 - \mathbf { M } _ { \mathrm { l a } }$ tent   
10: Compute latent residual guidance [4, 14, 16]: ${ \pmb r } _ { t } = - \bar { \bf M } _ { \mathrm { l a t e n t } } \big ( \widehat { \pmb x } _ { 0 } ( { \pmb x } _ { t } ) - { \pmb x } _ { \mathrm { s r c } } \big )$   
11: Augment score estimation [20, 55] or vector field $[ 3 4 , 3 7 ]$ with $\mathbf { \nabla } _  \mathbf { \} } r _ { t }$ and schedule   
12: Perform self-correction sampling over $[ s - ( i + 1 ) \cdot n , \ s - i \cdot n ]$   
13: end for   
14: Decode final latent state to $\mathbf { I _ { \mathrm { o u t } } }$ and return $\mathbf { I _ { \mathrm { o u t } } }$

whose masks accurately match the edited region, excluding cases where the mask does not define a realizable target for a mask-free editing model (e.g., task add), yielding 198 samples in total. Raw results are reported in Tab. 6 and 7.

Table 6: Dif on level K [29,51]
<table><tr><td>K Diff (Edit)</td><td>Diff (Non)</td></tr><tr><td>0.00 0.15</td><td>0.06</td></tr><tr><td>0.50</td><td>0.13 0.05</td></tr><tr><td>1.00</td><td>0.12 0.04</td></tr><tr><td>2.00 0.10</td><td>0.02</td></tr><tr><td>3.00 0.08</td><td>0.01</td></tr></table>

Table 7: Connectivity at diferent energy coverages
<table><tr><td colspan="4">Energy CC(Edit) CC(Non) LCC(Edit) LCC(Non)</td></tr><tr><td>0.50</td><td>100.45</td><td>1472.12 0.64</td><td>0.29</td></tr><tr><td>0.60</td><td>104.85</td><td>1741.81 0.67</td><td>0.31</td></tr><tr><td>0.70</td><td>99.42</td><td>1912.12 0.70</td><td>0.36</td></tr><tr><td>0.80</td><td>87.44</td><td>1934.48 0.76</td><td>0.47</td></tr><tr><td>0.90</td><td>77.43</td><td>1664.87</td><td>0.85 0.69</td></tr></table>

## 3 Sensitivity and Robustness

Mask building. In Fig. 9, across diferent backbones and metrics, the results show a clear, reasonable negative relation between source preservation and instruction adherence, while nearby configurations vary smoothly, indicating the strong stability of SR-Edit.

Resolution and approximation. In Tab. 8, SR-Edit robustly improves fidelity across resolutions, and fixed mask parameters remain efective. Since exact Doob guidance requires unavailable true $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ (data) posterior, we compare plugin $\widehat { \mathbf { x } } _ { 0 }$ with averaged rollout estimation, which is theoretically more accurate but brings no gain (also costly), suggesting one-step estimation is suficient.

![](images/c27f5a802fd66853d3087679564a53701398c39b9b3948bda2e642bf8c3ae2a8.jpg)  
Fig. 9: Sensitivity of mask parameters on 54 configs over 100 cases randomly chosen from our ImgEdit-Bench evaluation dataset. “Prompt compliance” denotes instruction-adherence score from Judge [65] model.

<table><tr><td rowspan="2">Metric</td><td colspan="3">Resolution 0.5×</td><td colspan="2">1.0×</td><td colspan="3">1.25×</td><td colspan="2">Approx.</td></tr><tr><td>Vanilla</td><td>Uns.</td><td>Scale</td><td>Vanilla</td><td>SR-Edit</td><td>Vanilla</td><td>Uns.</td><td>Scale</td><td>Roll-3</td><td>Roll-5</td></tr><tr><td>PSNR</td><td>9.56</td><td>10.70</td><td>10.01</td><td>14.59</td><td>17.06</td><td>12.06</td><td>13.70</td><td>14.14</td><td>16.80</td><td>16.84</td></tr><tr><td>SSIM</td><td>0.25</td><td>0.40</td><td>0.32</td><td>0.54</td><td>0.70</td><td>0.39</td><td>0.62</td><td>0.65</td><td>0.61</td><td>0.58</td></tr><tr><td>CLIP</td><td>27.27</td><td>27.33</td><td>27.42</td><td>26.83</td><td>26.99</td><td>27.12</td><td>27.27</td><td>27.32</td><td>27.39</td><td>26.46</td></tr><tr><td>Judge</td><td>3.12</td><td>3.10</td><td>3.19</td><td>3.86</td><td>3.93</td><td>3.72</td><td>3.77</td><td>3.69</td><td>3.81</td><td>3.87</td></tr></table>

Table 8: Robustness analysis on Qwen-Image-Edit 2511 [63] on the same dataset as in the previous analysis. “Vanilla” denotes the model’s original inference. “Scale” rescales mask parameters by resolution factor, while “Uns.” keeps them fixed. Roll-3 and Roll-5 replace plug-in with average of 5 full-chain estimates with stride 3 and 5. CLIP scores are reported after multiplication by 10<sup>2</sup>.