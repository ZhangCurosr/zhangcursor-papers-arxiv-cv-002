# TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery

Niran Nataraj<sup>a</sup>, Maina Sogabe<sup>a,∗</sup>, Masamichi Yamashita<sup>b</sup>, Norihiko Ito<sup>c</sup> and Kenji Kawashima<sup>a</sup>

<sup>a</sup>Department of Information Physics and Computing, Graduate School of Information Science and Technology, The University of Tokyo, Tokyo, Japan

<sup>b</sup>Department of Veterinary Neurology and Oncology, Faculty of Agriculture, Tottori University, Tottori, Japan

<sup>c</sup>Department of Ophthalmology, Tokyo Medical University, Tokyo, Japan

## A R T I C L E I N F O

Keywords:   
Surgical scene reconstruction   
Instrument occlusion   
Cataract surgery   
Ophthalmic surgery   
Surgical robotics perception   
Reference-guided reconstruction   
Real-time surgical vision

## A BS T RA C T

Intraoperative scene perception in ophthalmic microsurgery is limited by instrument occlusion: when a tool enters the high-magnification field, it conceals tissue from both the surgeon and the perception components of robot-assisted systems that process the video. Generative inpainting can remove the occluder but replaces the missing image content with synthesis, which is unsuitable when visual provenance matters. Many video inpainters also depend on future frames and cannot operate online. We present TransGHOST, a causal reference-guided reconstruction pipeline whose non-generative safe mode operates in real time. Every restored value is obtained by resampling a recorded frame using geometric transforms fitted only to visible regions. The system provides per-pixel confidence and provenance and marks unsupported pixels as abstentions. We evaluate TransGHOST on a new veterinary clinical corpus of canine cataract surgery with pixel-level instrument and anatomy labels. Only the occluded region is scored, and each scoring target is verified to difer from the reference supplied to the system. Safe mode reaches 31.89 dB masked PSNR and 0.915 masked SSIM at 51.7 ms median and 85.5 ms 95th-percentile latency. On a pre-declared held-out cohort using instrument cutouts excluded from segmentation training, it reaches 29.83 dB masked PSNR, while matchedmask evaluation on validation shows a 3.45 dB advantage over LaMa. Region-aware reference selection improves masked PSNR by 3.60 dB over recency when candidate references can themselves be occluded. Reference-age analysis further defines the operating interval for the brief, repeated occlusions targeted by TransGHOST.

## 1. Introduction

During ophthalmic microsurgery, the surgeon views the operative field through a high-magnification live video feed [1]. At this scale, the view must remain continuous and geometrically stable: displacements of tens of micrometres matter, and temporary concealment of a microscopic feature can leave pathology unaddressed. Surgical instruments must nevertheless enter the field and thereby occlude the tissue under treatment. Occlusion is particularly consequential during fine dissection and instrument exchange, when no alternative view of the covered tissue is available and the surgeon’s cognitive load and task dificulty increase [2, 3]. The surgeon may consequently pause or reposition the instrument, responses that prolong operating time, and any inference about the covered tissue may itself be inaccurate [4].

The same occlusion also constrains automated systems that process the operative view. Intraoperative scene perception in robot-assisted ophthalmic surgery is built on the microscope video stream, and an instrument crossing the field removes part of the scene from any component that performs registration, tissue tracking, workflow monitoring or visual servoing on that stream. Recovering the hidden context without inventing it is therefore a perception and reconstruction problem before it is a display problem. We investigate whether that context can be recovered in real time from previously observed pixels, with the absence of synthesis established by construction and reconstruction fidelity established by measurement.

![](images/30e0f7cc6ddf8264af0b6cad69a5982c291d698279a1109e7a8a69e751e82311.jpg)  
Figure 1: TransGHOST reconstructs the masked region from an aligned causal reference. Safe output uses geometric resampling and returns confidence, abstention and provenance layers.

Cataract surgery is the setting examined here. Instruments such as a phacoemulsification tip and a hook are introduced into the anterior chamber through corneal incisions in order to sculpt, divide and rotate the lens nucleus [5]. Each incision acts as a fulcrum that constrains the motion of the instrument passing through it, and the whole working field spans only a few millimetres. In a field of that size an instrument covering the tissue under treatment is dificult to avoid, and the surgeon commonly continues the manoeuvre rather than pausing, estimating the covered region from the frames seen a moment earlier and from the motion of the surrounding tissue.

Moving the instrument aside is not a free alternative. Its range of motion is limited by the position of the incision and by the geometry of the eye, and withdrawing and reinserting an instrument in order to look interrupts the manoeuvre and disturbs the anterior chamber and the incision itself. Presenting the tissue behind the instrument without interrupting the manipulation would therefore supply visual information that is otherwise unavailable. Neither the time lost to these repeated brief occlusions nor the additional demand they place on the surgeon is routinely recorded, which motivates the controlled pre-clinical investigation reported here, whose procedure scope is described in Section 3.3.

The most direct existing approaches address only part of this problem. Augmented-reality systems in other surgical fields can prevent virtual overlays from covering the tools [6, 7], but these systems do not recover the tissue concealed by the instrument. Existing general-purpose video inpainters are unsuitable for a diferent reason: they are generative, so the content reconstructed within the occluded region is synthesised rather than observed and cannot be treated as direct visual evidence during surgery.

Our objective is to restore occluded tissue in real time without synthesising anatomy and to mark pixels that lack support from an observed reference. TransGHOST addresses this objective through safe-mode reconstruction (Figure 1), which leaves each output value unchanged or obtains it by geometrically resampling a recent recorded frame. It also provides per-pixel provenance and abstains when the reference cannot support a reconstruction. All headline fidelity, latency and provenance results refer to this output, which remains within the sub-100 ms window for live surgical guidance at both the median and the 95th percentile on the new veterinary clinical corpus.

This study makes five contributions to reference-guided reconstruction for surgical video. (i) We show that instrument occlusion in this setting can be resolved without synthesising tissue: each restored value is resampled from a recorded frame under transforms fitted only to visible regions, and it carries the reference frame and source coordinate it came from, so a downstream component can separate observed pixels from unsupported ones. (ii) We define a region-aware reference-selection policy that chooses, from a bounded causal bufer, the newest frame whose instrument mask leaves the restoration region visible, and we measure what target-region visibility is worth against recency. (iii) We construct a benchmark in which candidate references may themselves be occluded, which is the condition under which a selection rule can be measured at all. (iv) We establish an evaluation protocol comprising scoring within the masked region, separation of each scoring target from its reference, reference-free controls, a cutout-disjoint heldout cohort and an analysis of fidelity against reference age. (v) We introduce a pixel-annotated ophthalmic corpus labelled for both instruments and anatomy, which supports the semi-synthetic construction that human patient-derived ophthalmic corpora do not readily permit (Section 3.1).

The occluded-reference benchmark (iii) is necessary to evaluate the selection policy (ii) because conventional composited benchmarks supply only instrument-free candidate frames. When every candidate leaves the target region visible, an overlap-aware policy becomes indistinguishable from recency. Occluded candidates are therefore required to measure the efect of regional visibility on selection.

Scored over the occluded region alone, the system reaches 31.89 dB masked PSNR and 0.915 masked SSIM at a 95th-percentile latency of 85.5 ms. On the held-out cohort using training-excluded instrument cutouts, it reaches 29.83 dB masked PSNR at 86.1 ms for the same latency percentile.

## 2. Related Work

## 2.1. Ophthalmic datasets and real-time segmentation

Widely used public ophthalmic datasets primarily annotate anatomical structures rather than surgical instruments. DRIVE, STARE and CHASE\_DB1 serve as established benchmarks for retinal vessel and structure segmentation, but none labels forceps, cutters or lasers [8, 9]. Studies that require instrument labels often rely on proprietary collections, including datasets for cataract procedures [5] and vitrectomy [10]. Instrument localisation in eye surgery has also been addressed with single-shot detectors [11], but a bounding box does not delimit the occluded pixels that reference-guided restoration has to replace.

Several groups have adapted lightweight convolutional networks to the latency requirements of the operating room. PhacoTrainer tracks cataract instruments at 30 fps [12], and YOLACT++ jointly segments the retina and instruments at approximately 87 fps [10]. Compressed U-Nets delineate optical coherence tomography layers with sub-4 ms inference [13], and PseudoSegRT operates below 1 ms per frame for semi-supervised optical coherence tomography guidance [14]. All of these methods segment in real time, but none restores the tissue obscured by the instruments.

## 2.2. Video inpainting

Flow-guided methods propagate motion edges with optical flow to preserve temporal boundaries, as in Flow-edge Guided Video Completion [15]. The end-to-end E2FGVI model [16] instead learns flow completion and feature propagation jointly, improving PSNR and SSIM at per-frame runtimes of about 0.12 s at 240p. The transformer-based ProPainter [17] combines recurrent flow completion with sparse attention to improve long-range temporal coherence, and its code and weights are public. All three methods are generative and require future frames, making them unsuitable for live surgical use. Their reconstructed regions may depict anatomy that was never observed, while their dependence on future frames is incompatible with online operation.

TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery

## 2.3. Reference-guided de-occlusion in surgery

Reference-guided inpainting supplies an image in which the occluded region is visible, thereby reducing reliance on generative completion. Koreeda et al. [18] rendered instruments virtually transparent at 62.4 ms per frame by warping a second endoscope’s view into the hidden region. This construction does not synthesise image content but requires a second endoscope. Hasan et al. subsequently applied reference-guided tool removal to laparoscopy, initially using a pool of frames captured before the instrument entered [19]. Their later system selects, for each frame, the candidate whose visible pixels most closely match the current view near the instrument. A companion method in the same work instead accumulates a background map over time [20]. General-purpose video inpainting systems also use references, but refresh them at a fixed frame stride without testing whether the target region is visible [21].

Among the related approaches, Xie et al. [22] address instrument occlusion in minimally invasive endoscopic surgery. Their method is designed for large, sustained toolchannel occlusions and blends a generative reconstruction of the current frame with information from previously observed occlusion-free frames. This hybrid design can return a complete image when the stored background is insuficient. However, because generated content may enter the final blend and no source map is provided, observed and inferred pixels cannot be distinguished. The method does not report an abstention mechanism for unsupported regions. Trans-GHOST addresses a diferent regime of smaller, shorter and repeated occlusions in ophthalmic microsurgery. Its safe mode resamples recorded image content, provides a per-pixel provenance map and marks unsupported pixels as abstentions. On the occluded-reference benchmark, its region-aware selector improves masked PSNR by 3.60 dB over recency and 3.63 dB over the published selection criterion evaluated here.

Xie et al. validate their method on clinical endoscopic video and report a mean throughput of 12.02 frames per second. Their evaluation characterises a diferent clinical setting and occlusion regime. They identify real-time streaming as future work, and their masked metrics characterise the single-frame restoration component during training rather than the complete hybrid output. Our evaluation measures the full TransGHOST pipeline with synchronised per-frame latency percentiles, verifies that the scoring target difers from the supplied reference, and includes a pre-declared cohort using instrument cutouts excluded from segmentation training. The difering datasets and protocols make the studies complementary rather than head-to-head. TransGHOST contributes single-camera operation, a non-generative safe output, per-pixel provenance, explicit abstention and end-toend evaluation.

## 2.4. Occlusion handling for surgical perception and guidance

In other surgical fields, augmented-reality systems handle occlusion by controlling where overlays are drawn.

![](images/ba8deffa9e4a0023e8c956d402603b4d96a886bf2f3fc7f589ca77ec286119b6.jpg)  
(a)

![](images/6ecb22f71f501f0b72b12c3e35e8796fac472aeca2b040a3f2cd3ed79b31d787.jpg)  
outer corneal boundary  
(b)  
inner corneal boundary instrument

Figure 2: Dataset preparation: (a) an annotated surgical frame with labelled corneal boundaries and instruments, and (b) further annotated frames from the training corpus.

De Backer et al. [6] delineate robotic instruments in renal surgery so that augmented-reality meshes do not cover them, and Celdrán et al. [23] review comparable assistance in laparoscopy. These systems preserve the visibility of the instrument, but they do not reveal the tissue behind it. Trans-GHOST instead reconstructs that tissue within the 100 ms limit reported as acceptable for surgical video feedback in telesurgery and laparoscopy [24, 25].

Real-time occlusion handling has also been demonstrated in ophthalmology. Liu et al. [26] describe a tracking and inpainting network for augmented-reality guidance in deep anterior lamellar keratoplasty, reporting 16.26 ms for inpainting inference and 35.14 ms end to end. This result reports real-time generative inpainting in that setting. TransGHOST addresses the complementary requirement of traceable reconstruction: safe mode resamples observed image content and requires only past frames. The remaining challenge is to bring these capabilities together. Low-latency segmentation, real-time surgical inpainting, reference pools, per-frame selection and accumulated background maps have each been established, but not in a causal single-camera pipeline that records the source of every restored pixel and abstains when a reference cannot support reconstruction. TransGHOST combines these elements in such a pipeline. It maintains a bounded bufer of recent frames and selects the newest frame whose instrument mask leaves the target region visible, drawing on video compression and motion estimation [27, 28]. Safe-mode output consists only of recorded image content under geometric resampling. To our knowledge, no surgical study has measured fidelity as the reference ages or quantified preservation outside the restoration mask. The complete pipeline has a median latency of 51.7 ms, and the evaluation separately measures restoration fidelity, outside-mask preservation and abstention.

## 3. Materials and Data

## 3.1. Surgical video corpus

Reference-guided restoration requires labels for both the visible anatomy and the instruments that occlude it. Existing cataract datasets provide part of this information: CADIS [29] labels corneal anatomy and a fine-grained instrument taxonomy, whereas CATA7 [30] labels instruments. They support segmentation research but do not provide the paired clean targets and temporally separated references needed to evaluate restoration. Our study also requires material from which controlled semi-synthetic composites can be constructed. We therefore built a veterinary clinical corpus with two groups of labels: the inner and outer corneal boundaries and all surgical instruments present.

The corpus comprises 13 high-definition $( 1 9 2 0 \times 1 0 8 0 ,$ 29.97 fps) recordings of canine cataract surgery. Recordings are assigned to roles before any frame is used: six supply the training composites, one the validation set and three the held-out cohort (Section 6.6), two are sealed for future evaluation and one is unassigned. This partition governs every TransGHOST result. The two learned models trained in this work, the TransRef-inspired baseline (Section 5.1) and the refinement in Appendix A, were instead trained on the triplets of Section 3.6, which draw on eleven recordings, including the validation recording and two of the held-out recordings. Their scores on these recordings may therefore be optimistic. Each recording covers a single operative step. The ten recordings behind the results reported here cover nucleus sculpting, division and rotation together with anteriorchamber maintenance, and the instruments entering the field across them are a phacoemulsification tip, a hook and a 27- gauge blunt cannula. One recording in the held-out cohort contains a tip and a hook at the same time. Each recording was subsampled to retain clinically salient moments while reducing frame redundancy. The annotation introduced here is pixel-level and covers both anatomy and instruments, as required for reference-guided restoration.

## 3.2. Ethics and approvals

Clinical surgical images were obtained from canine patients undergoing phacoemulsification for the treatment of cataracts. The images were acquired using a surgical microscope (Leica M844 F40, Leica Microsystems) following pharmacological pupil dilation. The original surgical videos were recorded at a resolution of $1 9 2 0 \times 1 0 8 0$ pixels and a frame rate of 29.97 fps.

All surgical procedures were performed as part of routine veterinary care, and no additional invasive interventions were conducted for research purposes. The use of clinical image data for research was reviewed and approved by the Animal Clinical Research Ethics Review Committee of the Faculty of Agriculture, Tottori University (Approval No. R8-006).

To protect the confidentiality of veterinary clinical records and minimize the use of personal clinical information, the research dataset was limited to surgical image data required for this study. Accordingly, demographic information, including breed, sex, and age, was not included in the dataset provided for research.

## 3.3. Imaging field and clinical scope

The recordings show cataract surgery on the operative eye from a single high-magnification camera. The camera provides a frontal view in which the eye occupies most of the frame, with periocular tissue and the surgical drape visible at the margins and instruments entering laterally. The two anatomy classes delineate the inner and outer corneal boundaries, within which TransGHOST reconstructs tissue occluded by an instrument.

Safe-mode reconstruction does not depend on identifying the tissue hidden by the instrument. Each restored value is resampled from an earlier frame recorded by the same camera, and the geometric transforms are fitted exclusively to regions visible in the current frame.

Building on real-time ophthalmic de-occlusion [26] and reference-guided pixel transfer in surgery [18], Trans-GHOST combines causal single-camera operation with a non-generative safe output, per-pixel provenance and explicit abstention. Section 6.4 compares its restoration accuracy with that of a generative inpainter under matched masks.

## 3.4. Annotation

Annotation proceeded through the three stages described below, using the labelled frames illustrated in Figure 2.

(i) Manual seeding. We sampled a seed set of $\tilde { N } = 4 0 0$ frames from the development videos and manually annotated each frame with polygons.

(ii) Manual expansion. We manually expanded the seed to 1,760 polygon-annotated frames, the set on which the segmentation architectures of Section 5.3 are compared in Section 6.1. All counts of “hand-labelled frames” in this paper refer to this set rather than to the 400-frame seed.

(iii) Automatic expansion and compositing. We enlarge the hand-labelled set through confidence-gated pseudolabelling and semi-synthetic compositing (Section 3.5). Temporal jitter of ±1 frame adds further variation, producing the totals in Table 1. Source videos are assigned to data splits before expansion, and every derived frame remains in the same split as its source.

Class taxonomy. The annotation defines six classes, divided into anatomy and tools:

$$
C = \{ \mathbf { c o r } _ { \mathrm { o u t e r } } , \mathbf { c o r } _ { \mathrm { i n n e r } } , \mathbf { i n s } _ { m } , \mathbf { i n s } _ { s } , \mathbf { i n s } _ { t } , \mathbf { i n s } _ { \mathrm { o t h e r / r e f l e c t i o n } } \} ,
$$

where cor denotes the corneal boundaries that delimit the intra-ocular field of view and ins denotes the surgical instruments. Specular reflections are predominantly instrumentassociated in these recordings and are therefore grouped with unclassified instruments for restoration.

Class frequencies vary widely across this annotated set. The two corneal classes appear in 1,725 frames each and the main instrument class $\mathrm { i n s } _ { m }$ in 1,259, while ${ \mathrm { i n s } } _ { s } ,$ ins and ins<sub>other/reflection</sub> appear in 250, 135 and 23 frames, respectively. Because restoration requires the location of instrument pixels rather than instrument identity, the four tool classes are merged into a single instrument channel for every restoration experiment. This task-aligned representation pools all annotated tool pixels during training, while the annotation files retain the per-class taxonomy for other applications.

Corpus size after each expansion stage. The real column counts recorded frames carrying hand or confidence-gated pseudo labels and the synthetic column counts semi-synthetic composites. Source videos are assigned to a split before expansion, so no derived frame crosses a split boundary. This is the segmentation partition and is separate from the restoration evaluation sets of Section 5.4.
<table><tr><td>Split</td><td>Real</td><td>Synthetic</td><td>Total images</td></tr><tr><td>Train</td><td>20,000</td><td>40,000</td><td>60,000</td></tr><tr><td>Test</td><td>4,000</td><td>6,000</td><td>10,000</td></tr></table>

![](images/c6ba7f9afbf62abe3253e339d0c157cabb9f2c1d1761e85bf2e380fa1ba6fcfe.jpg)  
Figure 3: Training triplets from three development videos, one per instrument. Each removal mask $M _ { t } ^ { \mathrm { t r } }$ unites the instrument mask (white) with one to four random rectangles or circles (blue), which may overlap one another and the instrument. The mask is binary, and the two colours only distinguish its parts. The reference $I _ { t - 3 0 }$ is the frame 30 frames earlier, and the dotted outline marks the mask region in it. In these examples the instrument has moved away from its own masked region.

## 3.5. Dataset expansion pipeline

Automatic labelling with the trained segmenter. Given an unlabelled frame �, predictions from the trained segmentation model $f _ { \theta }$ are retained at a confidence threshold of $\tau = 0 . 9$ and converted to pseudo-label masks. This process adds $N _ { \mathrm { p s e u d o } } \approx$ 20k pseudo-labelled images to the training pool.

Semi-synthetic compositing. For each accepted mask $\mathscr { M } _ { j }$ , we select a background frame $B _ { k }$ whose own instrument mask does not overlap it, $\mathcal { M } _ { j } \cap \mathcal { M } _ { B _ { k } } = \emptyset$ , and generate a composite as

$$
I _ { j k } ^ { \mathrm { s y n } } = B _ { k } \odot ( 1 - \mathcal { M } _ { j } ) \ + \ I _ { j } \odot \mathcal { M } _ { j } .
$$

Each composite receives one randomly selected augmentation: a flip, elastic deformation, colour jitter, a rotation with $\theta \sim \mathcal { V } [ - 2 0 ^ { \circ } , 2 0 ^ { \circ } ]$ , or a scaling with $s \sim \mathcal { V } [ 0 . 8 , 1 . 2 ]$ . The process yields 40k synthetic training images and 6k synthetic test images. Together with the real frames, these produce the totals in Table 1.

## 3.6. Reference-reconstruction data triplets

For training the learned components, we build the triplets shown in Figure 3, $\langle I _ { t } , M _ { t } ^ { \mathrm { t r } } , I _ { t - \Delta _ { \mathrm { t r } } } \rangle$ . The ofset is $\Delta _ { \mathrm { t r } } = 3 0$ frames for triplets taken from recorded video and 22 frames for triplets in which recorded instrument motion is overlaid on a static background frame. As at deployment, the reference precedes the frame it restores, and the ofset introduces temporal variation between the target and its reference. The training mask $M _ { t } ^ { \mathrm { t r } }$ is the instrument mask united with one to four randomly placed, axis-aligned rectangles or circles. Rectangle sides are drawn between 5% and 33% of the frame width and height, and circle radii between 2.5% and 16.7% of the shorter side. The added shapes encourage the learned components to restore varied regions rather than a single silhouette. These triplets train the optional learned variant in Appendix A and the baseline generator of Section 5.1. The deployed safe-mode fill contains no network and requires no training data (Section 4.4).

At inference, the bounded causal bufer of Section 4.3 retains recent frames rather than sampling at a fixed stride. The regional overlap rule selects among them and can bypass a sequence of references in which the target region remains occluded.

## 4. The TransGHOST Method

## 4.1. Problem formulation and pipeline overview

TransGHOST recovers occluded anatomy through a three-stage pipeline applied to each frame $I _ { t }$ and summarised in Algorithm 1. A lightweight segmentation model produces an instance mask of all surgical tools. A regionaware reference-selection policy scans a bounded causal bufer for a recent frame in which the region now hidden was visible. A fill stage then reconstructs that region from the selected reference. All weights are fixed during inference, so the equations and timings describe inference only.

TransGHOST reconstructs the occluded region through safe mode (Section 4.4), which geometrically registers an earlier reference to the current frame and resamples it within the occluded region. Every transform is estimated outside the current instrument mask, where the current frame shows tissue rather than the instrument, and no restored pixel is drawn from a coordinate hidden in the reference. Bilinear resampling makes each restored value a weighted average of neighbouring intensities in one recorded frame, so safe mode is non-generative by construction. Safe mode is the default configuration and produces the output evaluated throughout the main paper. Appendix A evaluates an optional learned refinement as a separate design variant.

## 4.2. Instrument segmentation

The restoration stage requires a binary occluder mask rather than the identity of each instrument. The deployed segmentation model therefore collapses the four annotated tool classes $\{ \mathrm { i n s } _ { m } , \mathrm { i n s } _ { s } , \mathrm { i n s } _ { t } , \mathrm { i n s } _ { \mathrm { o t h e r / r e f l e c t i o n } } \}$ into a single instrument channel during training. This task-aligned merger pools the available examples across the class-frequency distribution described in Section 3.

Algorithm 1 TRANSGHOST processing for one frame: instrument segmentation, regional reference selection and gated   
geometric compositing.   
Input: video stream $\{ I _ { t } \} _ { t = 1 } ^ { T }$   
Parameters: segmentation confidence �, allowed classes , compositing confidence threshold $\tau _ { c } ,$ bufer length �   
Output: restored stream $\{ \hat { I } _ { t } \} _ { t = 1 } ^ { T }$ , with per-pixel confidence $c _ { t } ,$ abstention mask, and provenance map   
1: Initialise causal bufer $\varkappa  \varnothing$   
2: for t ← 1 to T do   
3: � ← INSTANCESEGMENT $( I _ { t } ; \tau , c )$ {instrument mask, Section 4.2}   
4: if $M _ { t } = \emptyset$ or $\mathcal { H } = \emptyset$ then   
5: $\hat { I } _ { t } \gets I _ { t }$ {nothing to restore, or no past frame yet}   
6: else   
7: $\begin{array} { r } { ( I _ { \mathrm { r e f } } , M _ { \mathrm { r e f } } ) \gets \arg \operatorname* { m i n } _ { ( I ^ { \prime } , M ^ { \prime } ) \in \mathcal { H } } \big ( \| M _ { t } \wedge M ^ { \prime } \| _ { 0 } \big / \| M _ { t } \| _ { 0 } , - t ^ { \prime } \big ) \{ \mathrm { E q . } ( 1 ) \} } \end{array}$   
8: $( \hat { I } _ { t } , c _ { t } ) \gets \mathsf { S }$ AFECOMPOSITE $\left( I _ { t } , M _ { t } , I _ { \mathrm { r e f } } , M _ { \mathrm { r e f } } \right)$ {Section 4.4}   
9: Retain $I _ { t } ( p )$ wherever $c _ { t } ( p ) < \tau _ { c }$ {abstention, Eq. (4)}   
10: end if   
11: Emit $\hat { I } _ { t }$ with $c _ { t } ,$ the abstention mask and the provenance map   
12: Append $( I _ { t } , M _ { t } )$ to , keeping the most recent � entries   
13: end for

Masks smaller than five pixels are discarded before the remaining masks are merged and dilated:

$$
M _ { t } ^ { \mathrm { r a w } } = \bigvee _ { k } m _ { t , k } , \qquad M _ { t } = \mathrm { d i l a t e } ( M _ { t } ^ { \mathrm { r a w } } , { \bf S } _ { 1 4 } ) ,
$$

with ${ \bf S } _ { 1 4 }$ a 14×14 pixel kernel that provides a restoration margin around detected tool boundaries. The final dilated mask is used throughout end-to-end evaluation, so the reported precision, recall, IoU and false-positive MAE characterise the complete mask passed to the compositor. Segmentation takes a median of 13.8 ms per 1920×1080 frame and 14.5 ms at the 95th percentile. These values were measured during the deployed timing evaluation on the same RTX 5000 Ada used for every other timing in this paper.

## 4.3. Reference selection

A bounded causal bufer ${ \mathcal { H } } = \{ ( I _ { t - i } , M _ { t - i } ) \}$ stores the $L = 1 2$ most recent frames and their predicted instrument masks (Figure 4). Only frames with indices earlier than � enter the bufer, preventing the selector from using future information, and the fixed length bounds both memory and the per-frame selection cost independently of sequence duration. At 29.97 fps the bufer spans 0.4 seconds, which is the reference-age interval characterised in Section 6.7.

Candidate references are ranked by the overlap between their instrument masks and the current restoration region in image coordinates:

$$
\begin{array} { r } { ( I _ { \mathrm { r e f } } , M _ { \mathrm { r e f } } ) = \underset { ( I ^ { \prime } , M ^ { \prime } ) \in \mathcal { H } } { \arg \operatorname* { m i n } } \bigg ( \frac { \| M _ { t } \wedge M ^ { \prime } \| _ { 0 } } { \| M _ { t } \| _ { 0 } } , - t ^ { \prime } \bigg ) . } \end{array}\tag{1}
$$

The two terms are ordered lexicographically. The selector first chooses the most recent reference with zero overlap. If every reference covers part of the region, it instead chooses the one with the smallest overlap. Equation (1) is a lowcost pre-registration visibility score for the short causal intervals considered here. Final pixel support is determined

causal buffer

![](images/cc299271d51153542e88b423bbfc661522b7541222601ca0e13aeeb3ae99e8bb.jpg)  
Figure 4: Regional reference selection over the causal bufer. The overlap rule selects the newest candidate whose instrument mask leaves the target region visible.

after registration: Eq. (4) checks whether every sampled source coordinate lies within the reference image and outside $M _ { \mathrm { r e f } } .$ . The selector therefore ranks candidates, while the confidence, abstention and provenance outputs govern the pixels admitted to the reconstruction. Because ranking and source validation have separate roles, only post-registration validated pixels enter the output even when the two criteria identify diferent supports. This regional criterion improves masked PSNR by 2.90 dB over the composite frame-quality score evaluated in Section 6.5. It also requires only one mask intersection per reference, whereas an appearancebased comparison evaluates image values over a region.

Reference-selection behaviour across evaluation sets. The reported protocol uses Eq. (1). On the validation and held-out sets every supplied reference is instrument-free, so the overlap term carries no information about the target region and the rule follows its recency tie-break on essentially every sample. The measurements bear this out: the heldout metrics are identical under the two rules, and on unobstructed references the two rules select the same reference on 173 of the 178 validation samples, leaving the 0.05 dB separation reported in Section 6.5 to the remaining five, with recency marginally ahead. That is small beside the gain the regional term produces when candidates can hide the region. The occluded-reference benchmark in Section 6.5 supplies that condition and yields a 3.60 dB masked-PSNR advantage.

If the selected reference still covers part of $M _ { t } ,$ the system reconstructs the supported pixels and retains the currentframe values elsewhere. The abstention mask records the remaining region explicitly, preventing pixels from one instrument from being transferred into the location of another. Section 6.5 quantifies this behaviour when the candidate pool provides partial rather than complete support.

## 4.4. Safe mode: reference-grounded compositing

Safe mode reconstructs $M _ { t }$ by locating the corresponding image content in $I _ { \mathrm { r e f } }$ and resampling it. Within $M _ { t } ,$ the current frame shows the instrument rather than the underlying tissue, so measurements from this region would track instrument motion and are excluded from every fit.

Registration proceeds through three stages, from global alignment to local correction, and the two masks enter at the stages where each is informative. The global fit is constrained by $M _ { t }$ alone, because a whole-frame criterion is dominated by the large unobstructed area and remains well conditioned when part of the reference is occluded. The reference mask $M _ { \mathrm { r e f } }$ then enters the two stages that read individual reference samples: it restricts the band in Eq. (3) and the blockwise fit to coordinates the reference can actually supply, and it sets the confidence to zero in Eq. (4) wherever the sampled source coordinate falls inside it, so that a restored pixel is drawn only from a location the reference records as tissue.

Global registration. An enhanced correlation coeficient criterion fits a $2 \times 3$ afine map $A _ { t }$ from the current frame to the reference. The fit is evaluated on a decimated image, with $M _ { t }$ excluded from the objective:

$$
A _ { t } = \arg \operatorname* { m a x } _ { A } \ \rho \big ( I _ { t } , \ I _ { \mathrm { r e f } } { \circ } A ; \ { \lnot } M _ { t } \big ) ,\tag{2}
$$

where $\rho$ is the correlation coeficient over the valid support. Equation (2) accounts for camera drift and the dominant component of tissue motion.

Residual translation. The afine fit typically leaves a residual displacement of a few pixels, motivating a dedicated correction stage. We estimate this displacement by evaluating integer and then fractional translations � on a narrow band $B _ { t }$ of visible pixels outside the mask:

$$
d _ { t } = \arg \operatorname* { m i n } _ { d } \ { \frac { 1 } { | B _ { t } ( d ) | } } \sum _ { p \in B _ { t } ( d ) } { \Big | } I _ { t } ( p ) - I _ { \mathrm { r e f } } ( A _ { t } p + d ) { \Big | } .\tag{3}
$$

The valid band at ofset � is

$$
B _ { t } ( d ) = \{ p \in B _ { t } : A _ { t } p + d \mathrm { ~ i s ~ i n ~ v i e w ~ a n d ~ o u t s i d e ~ } M _ { \mathrm { { r e f } } } \} .
$$

It contains the pixels for which the reference provides a valid sample. Normalising by $| B _ { t } ( d ) |$ | prevents an ofset from receiving a lower error merely because it moves samples outside the image. Since $B _ { t }$ lies outside $M _ { t }$ , the hidden region does not contribute to Eq. (3).

Per-block local ofsets. Local tissue deformation between frames is represented by fitting an independent translation for each �×� block using the same visible-region criterion. Blocks meeting the support criterion contribute ofsets that are interpolated into a continuous field $\delta _ { t }$ to avoid seams. The comparison with dense optical flow in Appendix C further supports estimation from visible regions outside the instrument mask.

Compositing. The reference is sampled once at the final coordinates and at native resolution, without downsampling. Writing $s _ { t } ( p ) = A _ { t } p + d _ { t } + \delta _ { t } ( p )$ for the source coordinate of pixel $p ,$ the sampled value is $\tilde { I } _ { t } ( p ) = I _ { \mathrm { r e f } } ( s _ { t } ( p ) )$ . A perpixel confidence $c _ { t }$ combines registration agreement with two validity conditions, the source coordinate lying within the reference image domain Ω and outside the reference mask:

$$
c _ { t } ( p ) = \mathrm { m i n } \big ( \rho _ { t } , 1 \big ) ^ { + } \cdot { \mathbf 1 } \big [ s _ { t } ( p ) \in \Omega \big ] \cdot { \mathbf 1 } \big [ s _ { t } ( p ) \notin M _ { \mathrm { r e f } } \big ] ,
$$

where $\rho _ { t }$ is the correlation coeficient returned by Eq. (2), clamped to [0, 1] and set to zero if the fit does not converge. Pixels below the threshold $\tau _ { c }$ form the abstention set and remain unmodified. A source coordinate outside the frame or inside the reference mask therefore has zero confidence, irrespective of photometric agreement, so support is decided by geometry and the photometric term acts as a per-frame gate.

For reproducibility, global registration uses a 0.10 decimation, 15 iterations and 24 pixels of context around the mask bounding box. Residual translation is evaluated over a 8-pixel band, within a radius of 5 and at 4 sub-pixel steps. Local ofsets use � = 48 pixel blocks, a search radius of 3 and 0.35 of native resolution. The remaining thresholds are $\tau _ { c } = 0 . 6 5$ , mask-core confidence 0.65, uncertain-mask confidence 0.35 and segmentation confidence 0.25. The output is

$$
\hat { I } _ { t } ( p ) = \left\{ \begin{array} { l l } { \tilde { I } _ { t } ( p ) } & { p \in M _ { t } \mathrm { ~ \land ~ } c _ { t } ( p ) \geq \tau _ { c } , } \\ { I _ { t } ( p ) } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

Thus, every pixel outside $M _ { t }$ retains its input value. Alongside $\hat { I } _ { t } ,$ , the pipeline returns $c _ { t } ,$ the abstention mask and a provenance map containing the reference frame and source coordinate for each replaced pixel.

## 4.5. Latency

Under the synchronised protocol of Appendix F, safe mode has a median end-to-end latency of 51.7 ms and a 95th percentile of 85.5 ms on an RTX 5000 Ada. Both are below the 100 ms delay reported as acceptable for surgical video feedback in telesurgery and laparoscopy [24, 25]. Segmentation accounts for 13.8 ms at the median and the threestage alignment for 20.3 ms. The remaining time covers native-resolution compositing, confidence and provenance maps, and frame input and output. Stage medians do not sum exactly to the end-to-end median because they occur on diferent frames.

Section 6.8 examines the tail-latency distribution in more detail.

## 5. Experimental Design

## 5.1. Baselines: reconstruction strategies for surgical video

The baseline methods difer in the context available to them and in whether they synthesise unobserved content. Single-image models such as LaMa [31] receive only the current frame and must infer the vessels and reflections behind an instrument, potentially producing frame-to-frame variation. The video methods considered here propagate information through flow or attention [32, 17]. They improve temporal coherence but use future frames and remain generative when an occlusion persists. Reference-guided methods instead receive an earlier frame in which the hidden region is visible, allowing observed content to inform the reconstruction.

Accordingly, LaMa is included as the reference-free method and is evaluated under the same masked-region protocol. The real-time comparison is restricted to causal methods, whereas video inpainters that require future frames fall outside this operating condition.

We evaluate both learned and geometric forms of reference guidance. The learned comparison is a real-timeconstrained adaptation of the dual-encoder reference transformer of TransRef [33, 34]: one transformer block is removed from each decoder stage, reducing floating-point operations by about 40 per cent, and a lightweight PatchGAN term is added during fine-tuning on the triplets of Section 3.6 (Figure 5). It operates at 256 px as a compact learned comparison with the low-latency geometric method. Section 6.4 evaluates it using the same samples, masks, references and masked metrics as the other restoration methods.

The geometric alternative deployed by TransGHOST registers the reference to the current frame and transfers recorded image content without using a network for reconstruction. A learned fill can produce content absent from the reference and must therefore be assessed empirically, whereas geometric transfer is constrained by construction to recorded content. Section 6 compares classical difusion, reference-free learned inpainting, reference-guided learned inpainting and geometric transfer.

## 5.2. Metrics

Restoration quality is measured over the occluded region using four metrics. PSNR (peak signal-to-noise ratio, ↑) measures pixel fidelity and penalises large errors quadratically. SSIM (structural similarity index, ↑) measures luminance, contrast and structural agreement. MAE (mean

## Table 2

Segmentation benchmark on the 1,760-frame hand-labelled set. Both models are scored on mask metrics at their best checkpoint, matching Figure 6. Latency is model inference alone, measured on the same GPU and separately from the deployed segmentation stage timed in Section 6.8.
<table><tr><td>Model</td><td></td><td>F1 ↑ mAP@.50 ↑ Latency (ms) ↓</td><td></td></tr><tr><td>YOLOv8-xl</td><td>0.91</td><td>0.975</td><td>8-10</td></tr><tr><td>YOLOv11</td><td>0.92</td><td>0.982</td><td>6-8</td></tr></table>

absolute error, ↓, in 0-255 intensity levels) quantifies the average absolute error, while LPIPS (learned perceptual image patch similarity, ↓) measures perceptual diferences in texture and structure. Where two systems are compared, they are scored on the same frames, masks and references, so the per-sample values are paired and are compared with a two-sided Wilcoxon signed-rank test rather than by their means alone.

## 5.3. Segmentation-model training

We fine-tuned three architectures on the 1,760-frame hand-labelled set: Detectron2 Mask R-CNN, YOLOv8 and YOLOv11. Optimiser settings were held common where the architectures permitted, with the same initial learning rate, momentum and weight decay. The training budgets difer: YOLOv8 was fine-tuned for 50 epochs at batch size 24 and YOLOv11 for 100 epochs at batch size 20, so the comparison is between the fine-tuned models rather than between architectures at a matched budget.

1. Mask R-CNN (Detectron2): the objective is $\begin{array} { r l } { \mathcal { L } } & { { } = } \end{array}$ $\lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { s m o o t h - L 1 } } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { B C E } }$ , with ROIAlign pooling at two resolutions (box 7×7 and mask 14×14). The � subscripts identify the branch on which each loss acts: classification (cls), bounding-box regression (box) and mask prediction (mask).

2. YOLOv8 and YOLOv11: both use the unified loss $\mathcal { L } = \mathcal { L } _ { \mathrm { o b j } } + \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { m a s k } }$ . Both were fine-tuned from the released segmentation checkpoints under the framework’s automatic optimiser selection, with an initial learning rate $l r _ { 0 } ~ = ~ 0 . 0 1$ , momentum 0.937, weight decay $5 \times 1 0 ^ { - 4 }$ and an IoU assignment threshold of 0.7.

Mask R-CNN under Detectron2 was not carried forward: its per-frame inference latency was the highest of the three and its segmentation average precision was the lowest under its own COCO evaluation protocol. Table 2 therefore reports the two models that remained candidates, and YOLOv11 was selected. All restoration experiments use this architecture after the retraining described in Section 6.1, on the merged instrument channel of Section 4.2.

## 5.4. Evaluation protocol

Although inpainting studies often report whole-frame scores, prior surgical de-occlusion studies have used masked scoring. Xie et al. [22] use this approach, computing structural similarity, peak signal-to-noise ratio and root mean squared error over the filled region. Whole-frame scoring is uninformative here because the system changes only pixels within the instrument mask, which occupies a few per cent of the image. More than ninety-five per cent of such a score would come from pixels that no compared method altered, allowing even an inaccurate reconstruction to retain a high whole-frame PSNR. For this reason, all restoration results in Section 6 use masked scoring.

![](images/4098aa24e9a18b623c1165e32fa7c2ddb9384be551b70bf98a81a482f0863ca8.jpg)  
Figure 5: Real-time-constrained TransRef-inspired baseline. Reference features enter through Ref-PA modules and a transfer decoder regenerates the masked stream at 256 px.

A second safeguard requires the reference supplied in every semi-synthetic sequence to difer from the clean frame used as the scoring target. A pre-evaluation check excludes any sample that violates this condition. Without it, direct copying of the target would receive a perfect score and the experiment could not distinguish reconstruction from leakage. We identify this per-sample target-reference separation, rather than masked scoring, as the element that distinguishes this protocol.

Unless stated otherwise, evaluation uses 178 nativeresolution (1920 × 1080) samples from clips excluded from training. Mask areas range from 0.1 to 4.6 per cent of the frame, with a median of 0.9 per cent. Ground-truth masks isolate the restoration stage, whereas predicted masks evaluate the deployed system. This is the default validation set, while Section 6.6 adds a pre-declared held-out cohort and Appendix F provides the metric definitions and timing protocol.

## 6. Results

Segmentation results use the annotated corpus from Section 3. Restoration results follow the protocol in Section 5.4 and are measured at native high-definition resolution (1920× $1 0 8 0 \mathrm { p x } )$ on one GPU.

![](images/26f6e6d6619a6ef17e0a04fe702d8e6f93babf102c71e1bd193bee37f5df1ae0.jpg)  
Figure 6: Mask F1 against detection confidence for (a) YOLOv8 and (b) YOLOv11, from the validation run of each fine-tuned model. Thin curves are the five classes with enough annotated frames for a stable sweep and the bold curve aggregates all classes, with its maximum given in each panel title.

## 6.1. Segmentation backbone and mask quality

We compared three segmentation architectures for accuracy and latency (Section 5.3). Figure 6 shows the mask F1 confidence sweep for YOLOv8 and YOLOv11, reproduced from the validation output of each run. On mask metrics the two models are close: F1 is 0.91 against 0.92 and $\mathrm { m A P } _ { 5 0 }$ is 0.975 against 0.982, and the training budgets behind them difer as stated in Section 5.3. The accuracy margin is therefore too small to decide the choice on its own. YOLOv11 was selected on inference latency, at 6-8 ms per frame against 8- 10 ms, and on label stability: during visual inspection the other two architectures switched labels more often between similar instrument classes.

Table 3  
Reference-guidance and alignment ablation on 178 validation samples using identical samples, masks and evaluation code.
<table><tr><td></td><td colspan="3">Ground-truth masks</td><td colspan="5">Predicted masks (deployed)</td></tr><tr><td>Fill rule or stage</td><td></td><td>PSNR↑ SSIM↑ MAE↓</td><td></td><td>PSNR ↑</td><td></td><td></td><td>SSIM ↑ MAE↓ Bnd. MAE↓</td><td>FP MAE↓</td></tr><tr><td colspan="9">Reference-free and unaligned controls evaluated with ground-truth masks</td></tr><tr><td>Occluded input (no restoration)</td><td>13.43</td><td>0.485</td><td>46.21</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Telea inpainting (no reference)</td><td>28.47</td><td>0.867</td><td>7.64</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Navier-Stokes inpainting (no reference)</td><td>28.58</td><td>0.867</td><td>7.43</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unaligned reference transfer (unaligned)</td><td>30.26</td><td>0.863</td><td>6.73</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">TransGHOST alignment components, cumulative</td></tr><tr><td>Global affine registration only</td><td>33.16</td><td>0.905</td><td>4.76</td><td>30.44</td><td>0.889</td><td>5.29</td><td>2.98</td><td>5.42</td></tr><tr><td>+ residual translation</td><td>34.71</td><td>0.926</td><td>3.93</td><td>31.51</td><td>0.913</td><td>4.42</td><td>2.47</td><td>4.46</td></tr><tr><td>+ sub-pixel refinement</td><td>34.85</td><td>0.928</td><td>3.88</td><td>31.57</td><td>0.914</td><td>4.40</td><td>2.46</td><td>4.44</td></tr><tr><td>+ per-block local offsets</td><td>35.34</td><td>0.932</td><td>3.48</td><td>31.89</td><td>0.915</td><td>4.10</td><td>2.24</td><td>3.96</td></tr></table>

![](images/10c23c4017b0fd1d752960b929a2258fbf3d05fbfac856e00e4a5e69c947e274.jpg)  
Figure 7: Instrument segmentation by the deployed model on validation frames, with per-instance confidences. The model predicts the merged instrument channel of Section 4.2.

Segmentation model used for the restoration results. Table 2 compares architectures on the multi-class annotations rather than reporting the final segmentation model. The deployed segmentation stage timed in Section 6.8 additionally assembles, merges and dilates the mask at 1920 × 1080, which accounts for its larger median. For restoration, the selected YOLOv11 backbone is retrained at a 512 px input resolution on the merged instrument channel from Section 4.2, using the semi-synthetic augmentation in Section 3.5. On its mixed validation set of real and semi-synthetic frames, this model reaches mask precision 0.962, recall 0.886 (F1 0.923) and $\operatorname* { m A P } _ { 5 0 - 9 5 } 0 . 7 5 6$ . This set is distinct from the 178- sample restoration evaluation, and Figure 7 shows representative validation predictions.

Because restoration depends on the predicted mask, we also measure segmentation on instrument cutouts excluded from training. On the cutout-disjoint cohort in Section 6.6, recall is 0.973, precision is 0.757 and IoU is 0.744. Pixel recall directly measures instrument coverage, and the heldout value corresponds to a predicted restoration mask covering 97.3 per cent of annotated instrument pixels. Precision and false-positive MAE separately characterise the added restoration margin beyond the annotation. Together, these metrics show that the selected mask operating point maintains high tool coverage on training-excluded cutouts while quantifying its boundary extension.

## 6.2. Contribution of the reference and alignment components

We evaluate restoration fidelity within the occlusion, preservation outside it and the conditions under which the system abstains, with representative safe-mode outputs shown in Figure 8.

Table 3 tests whether reference guidance improves reconstruction and how much each alignment stage contributes. Its upper block contains reference-free controls, while the lower block adds alignment components sequentially and reports results with ground-truth masks and with predicted masks end to end. Every row uses the same samples and evaluation code. The controls are reported only with ground-truth masks because they do not include an instrument-segmentation stage.

The ablation shows that alignment enables reference guidance to improve fidelity. Telea’s fast-marching method and Navier-Stokes inpainting propagate neighbouring content into the mask and achieve masked PSNR values of 28.47 dB and 28.58 dB, respectively, on the same validation samples with ground-truth masks. Direct transfer from an unaligned reference achieves 30.26 dB masked PSNR, approximately 1.68 dB above Navier-Stokes, the higher-PSNR reference-free control, while registration also raises masked SSIM by compensating for tissue motion between frames. The residual-translation correction contributes the largest gain beyond the afine fit: approximately 1.55 dB in masked PSNR with ground-truth masks. Sub-pixel refinement increases masked PSNR by approximately 0.1 dB, and blockwise translations add a further 0.5 dB, producing 35.34 dB with ground-truth masks and 31.89 dB in the deployed configuration.

Each component improves fidelity under both mask conditions, while false-positive MAE decreases from 5.42 to 3.96. The final alignment stage therefore combines higher masked fidelity with lower error in the predicted-mask extension. The approximately 3.45 dB diference between the ground-truth-mask and deployed columns shows that mask estimation has the greatest measured influence on end-toend fidelity. The 35.34 dB ground-truth-mask result also quantifies the reconstruction performance available as mask estimation improves.

TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery  
![](images/7b0307c273fc23e931c444bd3162e2c83ddcb7fde80a9ea7c4b434d6bcf44600.jpg)

![](images/726873be497532b28d787aa290db19a3f5955c614d237f7c2a894f3f90e67940.jpg)  
Figure 8: Safe-mode results on six validation samples, labelled (a) to (f), each set on its own tinted strip of four panels. Within a strip the panels are the occluded frame, the reference selected for it, the restored composite produced by the deployed configuration, and a twofold magnification of the restored region. The solid outline traces the instrument mask recorded for that sample. The dotted outline marks that same boundary at identical image coordinates on the other three panels, where the region is unobstructed in the reference and the instrument no longer appears in the output.

## 6.3. Exactness outside the masked region

Compositing operates at native resolution and modifies only masked pixels. Across all 178 samples, the mean absolute change outside the true mask is 0.0 intensity levels with ground-truth masks and 0.010 levels with predicted masks. The latter value measures the small extension of the predicted mask beyond the annotation, while pixels outside the predicted mask remain unchanged by construction.

The deployed mask covers 97.6 per cent of annotated instrument pixels on the validation set, corresponding to recall 0.976. Precision 0.783 and IoU 0.769 characterise the predicted-mask boundary, with the corresponding held-out results given in Section 6.1. False-positive MAE in Table 3 provides the associated pixel-level measurement.

## 6.4. Comparison with learned inpainters

Table 4 extends the classical controls with referencefree and reference-guided learned methods, evaluated on the same samples with masked scoring. LaMa [31] provides the reference-free baseline. The real-time-constrained TransRef-inspired generator from Section 5.1 receives the same reference as TransGHOST and provides a learned reference-guided comparison at its 256 px operating resolution.

With matched masks, TransGHOST reaches the highest PSNR and SSIM and the lowest MAE, boundary MAE and LPIPS among the compared methods in both evaluation sets. Its PSNR exceeds LaMa by 3.45 dB on validation (35.34 versus 31.89) and by 3.77 dB on the held-out cohort (33.94 versus 30.17). Masked MAE is 34 per cent lower on both sets, and masked SSIM is higher by 0.041 and 0.052. Under the masked-crop protocol in Appendix F, TransGHOST also reduces LPIPS from 0.034 for LaMa to 0.021, a 38 per cent improvement that complements its pixel-fidelity gains with stronger perceptual similarity. Because both systems score the same 178 samples under the same masks, the comparison is paired. The matched-mask diferences hold across the cohort rather than resting on a few samples: masked PSNR favours TransGHOST on 135 of the 178 samples, masked

SSIM on 141, masked MAE on 145 and boundary MAE on 139, each separated at $p < 0 . 0 0 1$

The TransRef-inspired generator reaches 29.80 dB on validation and 27.54 dB on the held-out cohort. Because its training data include the validation recording and two heldout recordings (Section 3.1), these values may overstate its performance. Matched-mask TransGHOST improves these values by 5.54 and 6.40 dB, respectively, and also achieves lower MAE, boundary error and LPIPS. Within this realtime-constrained learned comparison, the results support retaining native-resolution geometric transfer in the deployed system.

Mask matching is necessary for a valid restoration comparison because neither learned baseline predicts an instrument mask and both therefore receive the ground-truth mask. Comparing them directly with a TransGHOST configuration that predicts its own mask would confound segmentation and restoration. We therefore report matched-mask results for restoration and place the deployed configuration in a separate, labelled row.

When TransGHOST predicts its own masks, its masked MAE remains 23 per cent below LaMa on validation and 22 per cent below it on the held-out cohort, even though LaMa receives ground-truth masks. TransGHOST also has higher masked SSIM on both sets, and its validation masked PSNR equals LaMa’s to two decimal places, a diference the paired test does not resolve $( p = 0 . 3 4 )$ . Its masked SSIM and MAE advantages are separated at $p < 0 . 0 0 1$ . Boundary MAE is higher in the mean, at 2.24 against 2.00, but the paired test does not resolve that diference either $( p = 0 . 2 2 )$ and masked LPIPS is higher, at 0.038 against 0.034. With ground-truth masks, it reaches 35.34 dB, quantifying the potential for higher reconstruction fidelity through improved mask estimation, while its safe-mode output remains nongenerative and source-traceable.

Interpretation. The matched-mask advantage in masked PSNR ranges from 3.45 to 3.77 dB, while the deployed system has lower MAE and higher SSIM than LaMa. Provenance is a separate distinction that the fidelity columns do not capture. The learned methods infer the content inside the mask, whereas each safe-mode replacement is linked to a recorded frame and source coordinate. A high fidelity score alone does not establish this source relationship.

Table 4  
Masked-region fidelity on matched samples. Learned baselines use ground-truth masks. Predicted denotes deployed TransGHOST. Bnd. denotes boundary MAE. LPIPS is reported for validation.
<table><tr><td>Method</td><td>Masks</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MAE↓</td><td>Bnd. ↓</td><td>LPIPS ↓</td></tr><tr><td>Validation, 178 samples</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LaMa [31], generative, no ref.</td><td>true</td><td>31.89</td><td>0.891</td><td>5.29</td><td>2.00</td><td>0.0335</td></tr><tr><td>Real-time TransRef-inspired [33], generative, ref.</td><td>true</td><td>29.80</td><td>0.879</td><td>6.79</td><td>2.87</td><td>0.0548</td></tr><tr><td>TransGHOST safe mode, non-generative, ref.</td><td>true</td><td>35.34</td><td>0.932</td><td>3.48</td><td>1.47</td><td>0.0206</td></tr><tr><td>TransGHOST safe mode, non-generative, ref.</td><td>predicted</td><td>31.89</td><td>0.915</td><td>4.10</td><td>2.24</td><td>0.0377</td></tr><tr><td>Held-out, 101 samples</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LaMa [31], generative, no ref.</td><td>true</td><td>30.17</td><td>0.866</td><td>6.46</td><td>2.21</td><td></td></tr><tr><td>Real-time TransRef-inspired [33], generative, ref.</td><td>true</td><td>27.54</td><td>0.847</td><td>8.41</td><td>3.51</td><td></td></tr><tr><td>TransGHOST safe mode, non-generative, ref.</td><td>true</td><td>33.94</td><td>0.918</td><td>4.28</td><td>1.72</td><td></td></tr><tr><td>TransGHOST safe mode, non-generative, ref.</td><td>predicted</td><td>29.83</td><td>0.893</td><td>5.01</td><td>2.69</td><td></td></tr></table>

For both learned baselines, the generated region is composited into the original frame using the supplied mask, following LaMa’s inference procedure. Table 4 therefore focuses on masked-region fidelity, since the common compositing step fixes their outside-mask change at 0.0. The synchronised latency of the complete TransGHOST pipeline is reported separately in Section 6.8. For the deployed Trans-GHOST pipeline, which predicts its mask, the measured change outside the true mask is 0.010 intensity levels.

## 6.5. Reference selection and abstention

We isolate the efect of reference selection on a benchmark in which candidate references may themselves be occluded, a condition under which the region-aware rule improves masked PSNR by 3.60 dB over recency.

Conventional composited benchmarks obtain a known target by placing an instrument over an instrument-free frame and supplying another instrument-free frame from the same interval as the reference [19]. In this setting, every candidate has zero mask overlap with the target region, so the region-aware selector reduces to its recency tie-break by design. The occluded-reference benchmark introduces the visibility variation needed to measure the regional term directly.

The occluded-reference benchmark modifies the same samples by placing instruments in a controlled share of the candidate references. Some instruments cover the target region and provide partial support, others appear elsewhere in the frame and leave the target region visible, and the remaining references are unobstructed. This design distinguishes target-region visibility from whole-frame instrument occupancy and also includes cases in which every candidate covers part of the target region. Among 178 samples, the most recent reference is occluded over the target in 78, and 27 provide partial support in every candidate.

Table 5 evaluates the same rules with unobstructed and occluded references. With unobstructed references, the overlap rule reaches 33.37 dB masked PSNR and recency reaches

33.42 dB, so recency is ahead by 0.05 dB. No candidate hides the target region in this setting, so the regional term has nothing to discriminate: the two rules select the same reference on 173 of the 178 samples, and the 0.05 dB comes from the remaining five. The choice between them is immaterial here, and the separation is small beside the gain measured once candidates can hide the region.

With occluded references, recency reaches 26.44 dB masked PSNR because it can select a frame in which the target region is hidden. The regional overlap rule instead reaches 30.04 dB masked PSNR, a gain of 3.60 dB with the alignment stack, masks and samples held fixed.

The two rules choose the same reference on 100 of the 178 samples and produce identical output there. They difer on the remaining 78, which is also the number of samples in which the most recent reference is occluded over the target, and the overlap rule gives the higher masked PSNR on 66 of them. A two-sided Wilcoxon signed-rank test on the paired per-sample values separates the two rules at $p < 0 . 0 0 1$ . The gain therefore comes from the samples the regional term was designed to catch rather than from a shift across the whole cohort.

The composite quality score provides an intermediate comparison: its whole-frame clean-fraction term improves masked PSNR over recency by +0.70 dB, reaching 27.14 dB. The regional overlap rule reaches 30.04 dB masked PSNR because it ignores instruments that lie outside the target region, a distinction made measurable by candidates with instruments in non-overlapping locations.

The published criterion of Hasan et al. [20] selects the reference that best matches the current frame in a window around the mask. It reaches 26.41 dB masked PSNR, compared with 26.44 dB for recency and 3.63 dB below the regional overlap rule. This criterion and the overlap rule difer on 84 samples, and the overlap rule gives the higher masked PSNR on 71 of them, separated at $p < 0 . 0 0 1$ . It also compares image values over a region for every reference, whereas the overlap rule intersects two masks. In this experiment, its 95th-percentile latency is approximately 1.6× that of the overlap rule.

Reference-selection rules on unobstructed and controlled-occlusion candidate sets. Abst. denotes the proportion of reconstruction pixels marked as abstentions.
<table><tr><td rowspan="2">Selection rule</td><td colspan="2">Unobstructed references</td><td colspan="4">Occluded references</td></tr><tr><td>PSNR</td><td>Abst.</td><td>PSNR</td><td>Δ</td><td>SSIM</td><td>Abst.</td></tr><tr><td>Newest valid reference (recency)</td><td>33.42</td><td>0.000</td><td>26.44</td><td>一</td><td>0.854</td><td>0.154</td></tr><tr><td>Composite quality score</td><td>n/a</td><td>n/a</td><td>27.14</td><td>+0.70</td><td>0.862</td><td>0.134</td></tr><tr><td>Least region difference [20]</td><td>n/a</td><td>n/a</td><td>26.41</td><td>-0.03</td><td>0.857</td><td>0.145</td></tr><tr><td>Least mask overlap (ours)</td><td>33.37</td><td>0.000</td><td>30.04</td><td>+3.60</td><td>0.904</td><td>0.037</td></tr></table>

The selection-timing experiment uses a common pool of six references for every rule, while the live pipeline scans the causal bufer in Algorithm 1. The values in Table 5 therefore provide a controlled comparison between selection rules, and deployment latency is reported separately in Section 6.8.

Appearance scoring uses visible anatomy outside the region being filled, where nearby references are similar. The reference mask instead records target-region visibility directly. Appendix B shows that, for unobstructed references, recency matches oracle pre-alignment selection.

Efect of occlusion severity. We vary three aspects of benchmark severity: the frequency of instrument-bearing references, the probability that an instrument covers the target region, and the frequency with which every candidate provides only partial support. Table 6 shows consistent gains across this range: 1.34 dB when 16 per cent of references hide the region, 3.60 dB at 41 per cent and 4.40 dB at 64 per cent. Across all three settings, the overlap rule abstains between one-fifth and one-third as often as recency.

Abstention as a measured behaviour. The system abstains when no reference supports a pixel with suficient confidence. With unobstructed references, abstention covers 0.02 per cent of the region requiring reconstruction. With occluded references, it rises to 15.4 per cent under recency and 3.7 per cent under the overlap rule.

The benchmark records whether each sample contains a reference that exposes the complete target region, allowing reference-selection and support efects to be separated. In the 27 samples constructed so that every candidate covers part of the target, the overlap rule concentrates 89 per cent of all abstention and marks, on average, 21.4 per cent of their target region. In the remaining 151 samples, where a fully visible candidate exists, its mean abstention is 0.5 per cent with a median of zero, against 12.6 per cent under recency, which may select an occluded reference. Regional selection therefore confines abstention to the samples that genuinely lack support and removes it where a complete reference is available.

The regional rule also improves partial reconstruction in the 27 samples for which every candidate covers part of the target. Choosing the least-occluding reference reaches 20.9 dB masked PSNR, compared with 19.8 dB under recency, while the confidence map identifies the pixels supported by the selected reference.

The confidence threshold is stable over a broad range, with unchanged masked PSNR and abstention from 0.50 to 0.80 and only measurement-level variation at 0.90. The selected value lies within this plateau and away from the transition observed at 0.95, where abstention is 6.8 per cent rather than 3.7 per cent and masked PSNR difers by 0.37 dB. This stability arises because support is determined mainly by whether the source lies within the image and outside the reference mask, rather than by a continuously varying photometric score.

## 6.6. Held-out evaluation with training-excluded instrument cutouts

The configuration was selected using the 178 validation samples and fixed before evaluating the cohort in Table 7. The held-out cohort was processed as a pre-declared batch without fidelity tuning. After the initial latency measurement, the alignment search was optimised on validation data and evaluated once against pre-declared confirmation criteria described in Appendix D.

The held-out evaluation combines video-level separation with cutout-level separation. The 93 foreground configurations in the cohort are built from 110 distinct instrument cutouts, all drawn from the segmentation test split, and none of them occurs in segmentation training or in the training and validation composites. Every scoring target also difers from its supplied reference. The cohort therefore evaluates new background videos together with instrument cutout images excluded from model development.

The ablation follows the same monotonic pattern on the held-out cohort. The four stages improve masked PSNR by a total of 1.58 dB, reaching 29.83 dB masked PSNR and 0.893 masked SSIM on samples not used for configuration selection.

Mask generalisation is evaluated on the held-out cohort, where recall, precision and IoU are 0.973, 0.757 and 0.744, compared with validation values of 0.976, 0.783 and 0.769. Recall remains near 0.97 for training-excluded instrument cutouts, so tool coverage is retained across the cutout-disjoint cohort.

Video-level independence. The held-out cohort spans three videos and eight clips, allowing the masked-PSNR gain to be assessed across recordings. The gain is positive for every video: +1.96, +1.32 and +1.99 dB on 36, 60 and 5 samples, respectively. The clip-clustered paired bootstrap interval for this gain is [+1.04, +2.20] dB, and the videoclustered interval is [+1.32, +1.99] dB, with both intervals excluding zero.

Reference-selection gains across light, medium and heavy controlled occlusion distributions.
<table><tr><td rowspan="2">Distribution</td><td rowspan="2">Candidates hiding the region</td><td rowspan="2">Partial-support samples</td><td colspan="3">Masked PSNR</td><td colspan="2">Abstention</td></tr><tr><td>recency</td><td>overlap</td><td>gain</td><td>recency</td><td>overlap</td></tr><tr><td>Light</td><td>16%</td><td>6</td><td>31.29</td><td>32.63</td><td>+1.34</td><td>0.043</td><td>0.009</td></tr><tr><td>Medium</td><td>41%</td><td>27</td><td>26.44</td><td>30.04</td><td>+3.60</td><td>0.154</td><td>0.037</td></tr><tr><td>Heavy</td><td>64%</td><td>53</td><td>23.36</td><td>27.76</td><td>+4.40</td><td>0.219</td><td>0.068</td></tr></table>

![](images/ae9f7d474c5d149d53846570ea23c78ccda27dc6ee7ee300e3c1258d8d1bc5c9.jpg)  
Figure 9: Masked fidelity against reference age. The shaded band spans the classical reference-free controls of Table 3, and the dashed series is the fixed-cohort control of Section 6.7.

The positive per-video gains agree with the eight-clip interval, and the video-clustered interval provides the corresponding recording-level summary.

## 6.7. Fidelity as a function of reference age

Figure 9 quantifies reference-guided reconstruction over controlled reference ages. To our knowledge, surgical deocclusion studies have not previously reported this temporal fidelity curve.

A one-frame-old reference yields a masked PSNR of 33.42 dB, whereas a twelve-frame-old reference yields 28.06 dB at 0.4 seconds and 29.97 fps. This measured interval covers the brief occlusions targeted by TransGHOST, including instrument exchange, and provides an empirical criterion for selecting the reference-age limit in the reported real-time setting. The shaded band in Figure 9 marks the classical reference-free controls of Table 3, so the curve locates the reference ages over which geometric transfer exceeds difusion-based filling. The comparison with the learned reference-free baseline is a separate measurement and is reported under matched masks in Section 6.4.

The available sample count decreases at longer lags because early frames in a clip lack a suficiently old reference. To separate this change in cohort from the efect of reference age, we repeat the analysis on the 84 samples available at every lag. The fixed-cohort masked-PSNR curve gives 33.37 dB rather than 33.42 dB at lag 1 and the same value at lag 12 (Figure 9, dashed). A clip-clustered paired bootstrap estimates changes of −1.43 dB by lag 2 and −5.31 dB by lag 12, with 95 per cent intervals excluding zero at every lag, as tabulated in Appendix E.

## 6.8. Real-time viability

Real-time operation is assessed here against the interactive criterion adopted in Section 4.5: the end-to-end delay between a frame arriving and its restored output must stay within the 100 ms bound reported as acceptable for surgical video feedback in telesurgery and laparoscopy [24, 25]. Because occasional slow frames determine whether that bound holds in practice, the criterion is applied to the latency distribution rather than to its mean.

Safe mode has a median latency of 51.7 ms, a validation 95th-percentile latency of 85.5 ms, a held-out 95thpercentile latency of 86.1 ms and a validation 99th percentile of 101.4 ms on the same hardware. The criterion is met at the median and at the 95th percentile: the validation 95th percentile is approximately 15 ms below the 100 ms threshold and the median leaves roughly half the budget unused at 1920×1080. The 99th percentile lies 1.4 ms above the threshold, within the 2 to 3 ms repeat-measurement spread reported in Appendix F, so approximately one frame in a hundred reaches the bound.

Latency as a tail property. Because mean latency does not characterise occasional high-latency frames, we report the median, 95th and 99th percentiles on validation and the 95th percentile on the held-out cohort using the synchronised protocol in Appendix F (Table D1 and Appendix D).

Held-out ablation on 101 samples from three training-excluded videos using test-split instrument cutouts.
<table><tr><td>Stage</td><td>PSNR ↑</td><td>Δ</td><td>SSIM↑</td><td>MAE↓</td><td>Bnd. MAE ↓</td><td>FP MAE↓</td></tr><tr><td>Global affine registration only</td><td>28.25</td><td></td><td>0.861</td><td>6.76</td><td>3.83</td><td>6.63</td></tr><tr><td>+ residual translation</td><td>29.13</td><td>+0.89</td><td>0.883</td><td>5.81</td><td>3.26</td><td>5.50</td></tr><tr><td>+ sub-pixel refinement</td><td>29.26</td><td>+1.01</td><td>0.885</td><td>5.71</td><td>3.19</td><td>5.36</td></tr><tr><td>+ per-block local offsets</td><td>29.83</td><td>+1.58</td><td>0.893</td><td>5.01</td><td>2.69</td><td>4.34</td></tr></table>

Most latency variation arises from alignment: segmentation latency increases from a median of 13.8 ms to 14.5 ms at the 95th percentile, whereas alignment latency increases from a median of 20.3 ms to 46.3 ms at the 95th percentile and reaches a maximum of 62.6 ms. Because alignment cost scales with the warped area - from approximately 13,000 pixels for a small instrument tip to 1.35 million when the joint bounding region of two separated instruments covers most of the frame - alignment rather than segmentation determines the latency tail.

The sparse band sampling described in Appendix D reduces the 95th percentile from 96.5 to 85.5 ms and the 99th percentile from 115.4 to 101.4 ms, with a masked-PSNR change of 0.006 dB. An additional limit defines a lowerlatency operating point with a further 6 ms reduction at the 95th percentile, while the deployed configuration retains the higher-fidelity setting.

## 7. Discussion

## 7.1. Comparison with existing work

Xie et al. [22] provide one relevant clinical comparison among the reference-guided approaches discussed in Section 2. Their evaluation uses human endoscopic video, while ours uses a controlled pre-clinical ophthalmic corpus that permits compositing and masked ground-truth evaluation (Section 3.1). The methods address complementary occlusion regimes: Xie et al. target large, sustained toolchannel occlusions and use a generative term when stored background is insuficient, whereas TransGHOST targets brief, repeated occlusions and preserves pixel provenance in safe mode. The reference-age experiment quantifies this operating range directly rather than inferring it from a single reference interval.

The two designs difer in how they achieve coverage. Hybrid reconstruction prioritises a complete image by allowing generated content, while TransGHOST safe mode prioritises traceable support and records abstention explicitly. This pixel-level distinction allows intraoperative displays and downstream perception components to select the evidence appropriate to their task.

Within the established areas of reference-guided surgical de-occlusion [18, 19] and real-time ophthalmic de-occlusion [26], TransGHOST combines a non-generative safe output with causal single-camera operation, per-pixel provenance, measured abstention and evaluation that separates every scoring target from its supplied reference.

## 7.2. Relevance to robot-assisted surgical perception

Robotic assistance for intraocular microsurgery increasingly combines manipulation with intraoperative imaging [35]. Robot-assisted intraocular surgery has been performed in patients [36], virtual fixtures for a handheld micromanipulator have been generated in real time from microscope video [37], and a policy learned from expert demonstrations has navigated a tool to retinal targets in an eye phantom [38]. In cataract surgery, semi-automated lens extraction guided by optical coherence tomography has been demonstrated in post-mortem porcine eyes [39]. In such systems the controlled instrument also occludes part of the image used to guide it, so the tissue on which it acts is hidden from the components that track, register or constrain it.

Perception methods for robot-assisted surgery already contend with tool occlusion within their own models. Realtime simultaneous localisation and mapping of the retinal vasculature for robot-assisted intraocular microsurgery has been demonstrated under conditions that include tool occlusion [40]. In robotic minimally invasive surgery, dynamic reconstruction from stereo endoscopic video has used tool segmentation and temporal aggregation to recover the scene obstructed by the tool [41], and neural rendering of deformable tissue has used tool-mask-guided ray casting to handle the occlusion [42]. TransGHOST addresses the same problem at the image level, from a single camera view and without depth estimation or a scene model.

Instrument occlusion is a structured failure of intraoperative scene perception. Unlike many image degradations, the missing content may have been visible a fraction of a second earlier, and the afected region is estimated directly by segmenting the instrument. These two conditions make non-generative reconstruction from a recent frame possible.

In a robotic perception stack, provenance allows each downstream component to decide whether reference-derived pixels are admissible. A display may present the full composite, whereas a component requiring direct visual evidence can use the provenance and abstention masks to exclude unsupported pixels. A tracker or registration module can thus retain features beneath the instrument where a recorded frame supports them and treat abstained pixels as missing data, rather than receiving synthesised content that it cannot distinguish from observation. Reporting the median, 95th and 99th percentiles also supports latency budgeting for TransGHOST within perception, monitoring and visualservoing loops.

The resulting interface combines the reconstructed image with confidence, provenance and abstention layers and gives robotic components visual context together with the source of each reconstructed pixel. Integration with a robotic system and its efect on tracking or control were not evaluated in this study and remain future work.

## 7.3. Operating scope and validation

The study evaluates TransGHOST in a controlled preclinical setting, on canine cataract surgery. The held-out evaluation uses instrument cutouts excluded from training and reports positive gains for every held-out video (Section 6.6). This design separates development and evaluation across background videos and instrument cutouts while retaining exact pixel-level targets.

Masked-region fidelity is measured on semi-synthetic composites, which provide the clean target required for pixel-level scoring while retaining real surgical backgrounds and instrument cutouts. On unmodified continuous video no such target exists, because the tissue behind the instrument is never recorded in the frame being scored. Restoration fidelity is therefore not measurable on that surface, and no fidelity result is claimed for it. The occluded-reference benchmark evaluates light, medium and heavy reference obstruction and shows consistent gains of 1.34, 3.60 and 4.40 dB, respectively.

Figure 9 defines the reference-age range for brief, repeated occlusions by measuring fidelity from one to twelve frames. Segmentation is characterised jointly by recall, precision, IoU and false-positive MAE, so both tool coverage and predicted-mask extension are represented in the evaluation. The cutout-disjoint cohort then confirms positive reconstruction gains across three held-out videos and eight clips.

## 7.4. Safety and provenance

Safe mode achieves the highest measured fidelity in these experiments, but fidelity and provenance remain separate properties. A generative method could obtain the same score while producing pixels that are inferred rather than traceable to a recorded frame. Provenance must therefore be reported directly rather than inferred from an image-quality metric.

For this reason, each frame is accompanied by perpixel confidence, an abstention mask and a provenance map. Pixels without valid reference support retain their currentframe values and are marked as abstentions.

## 8. Conclusion

TransGHOST treats instrument occlusion in ophthalmic microsurgery as a reference-guided scene-reconstruction problem. We evaluate this clinically motivated problem in a controlled pre-clinical setting. The deployed safe mode uses geometric resampling of recorded image content and has a median end-to-end latency of 51.7 ms and a 95th-percentile latency of 85.5 ms. This source-traceable composite is the intraoperative output reported throughout the paper.

Safe mode is causal, uses a single camera and preserves pixels outside the predicted mask, with a mean absolute change of 0.010 intensity levels when measured outside the true mask. Each replacement is linked to its reference frame and source coordinate, while unsupported pixels remain unchanged and are marked as abstentions. The fidelity-versusreference-age curve further defines the temporal range over which a recent reference remains useful.

On masked-region evaluation, TransGHOST reaches 31.89 dB masked PSNR and 0.915 masked SSIM at a median latency of 51.7 ms and a 95th-percentile latency of 85.5 ms. On the held-out cohort, it reaches 29.83 dB masked PSNR at 86.1 ms for the same latency percentile. With masks matched, it exceeds LaMa by 3.45 dB in masked PSNR and reaches the best value on every reported fidelity metric among the compared methods. When predicting its own masks, it retains lower masked MAE and higher masked SSIM than LaMa while matching validation masked

PSNR. When references may be occluded, regional selection improves masked PSNR by 3.60 dB over recency.

Upon acceptance, the source code and model architecture will be released in a public GitHub repository, without the dataset. The corpus, evaluation sets and trained weights are available on request under the conditions in the Data availability statement.

Together, these results characterise real-time reconstruction with source traceability for brief, repeated occlusions in the evaluated ophthalmic setting.

## CRediT authorship contribution statement

Niran Nataraj: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization. Maina Sogabe: Conceptualization, Methodology, Formal analysis, Resources, Writing - original draft, Writing - review and editing, Supervision, Funding acquisition. Masamichi Yamashita: Resources, Data curation. Norihiko Ito: Resources, Data curation. Kenji Kawashima: Formal analysis, Resources, Writing - original draft, Writing - review and editing, Supervision, Project administration.

## Funding

This work was supported by JSPS KAKENHI Grant Number 21K18074.

## Declaration of competing interest

Kenji Kawashima holds stock in RiverField Inc., a surgical robotics company. Niran Nataraj, Maina Sogabe, Masamichi Yamashita and Norihiko Ito declare that they have no known competing financial interests that could have appeared to influence the work reported in this paper.

## Data availability

The source code and model architecture supporting the findings of this study will be released in a public GitHub repository upon acceptance. The release will not include the dataset.

The annotated veterinary clinical image corpus, derived evaluation sets, trained model weights and underlying surgical video recordings are available from the corresponding author on reasonable request, subject to institutional approval and applicable ethical, ownership and data-use restrictions.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used Grammarly to improve the clarity, readability, and academic presentation of the English text. After using this service, the authors reviewed and edited the manuscript as needed and take full responsibility for the content of the published article.

## References

[1] L. Ribeiro, J. Oliveira, D. Kuroiwa, M. Kolko, R. Fernandes, O. Junior, N. Moraes, H. Vasconcelos, T. Oliveira, M. Maia, Advances in vitreoretinal surgery, Journal of Clinical Medicine 11 (21) (2022) 6428. doi:10.3390/jcm11216428.

[2] C. Eckardt, E. B. Paulo, Heads-up surgery for vitreoretinal procedures: An experimental and clinical study, Retina 36 (1) (2016) 137- 147. doi:10.1097/IAE.0000000000000689.

[3] G. Mukash, P. Vartianen, M. Torkamani-Azar, Z. Karadis, M. Faraz, R. Bednarik, P. A. Karjalainen, M. Iso-Mustajärvi, A. Hussein, Evaluation of visual ergonomics in microsurgery: a real-time video processing solution, Acta Neurochirurgica 167 (1) (2025) 286. doi: 10.1007/s00701-025-06694-2.

[4] L. W. Way, L. Stewart, W. Gantert, K. Liu, C. M. Lee, K. Whang, J. G. Hunter, Causes and prevention of laparoscopic bile duct injuries: Analysis of 252 cases from a human factors and cognitive psychology perspective, Annals of Surgery 237 (4) (2003) 460-469. doi:10.1097/ 01.SLA.0000060680.92690.E9.

[5] S. Morita, H. Tabuchi, H. Masumoto, H. Tanabe, N. Kamiura, Realtime surgical problem detection and instrument tracking in cataract surgery, Journal of Clinical Medicine 9 (12) (2020) 3896. doi: 10.3390/jcm9123896.

[6] P. De Backer, C. Van Praet, J. Simoens, M. Peraire Lores, H. Creemers, K. Mestdagh, C. Allaeys, S. Vermijs, P. Piazza, A. Mottaran, C. A. Bravi, M. Paciotti, L. Sarchi, R. Farinha, S. Puliatti, F. Cisternino, F. Ferraguti, C. Debbaut, G. De Naeyer, K. Decaestecker, A. Mottrie, Improving augmented reality through deep learning: Real-time instrument delineation in robotic renal surgery, European Urology 84 (1) (2023) 86-91. doi:10.1016/j.eururo.2023.02.024.

[7] J. Hofman, P. De Backer, I. Manghi, J. Simoens, R. De Groote, H. Van Den Bossche, M. D’Hondt, T. Oosterlinck, J. Lippens, C. Van Praet, F. Ferraguti, C. Debbaut, Z. Li, O. Kutter, A. Mottrie, K. Decaestecker, First-in-human real-time AI-assisted instrument deocclusion during augmented reality robotic surgery, Healthcare Technology Letters 11 (2-3) (2024) 33-39. doi:10.1049/htl2.12056.

[8] G. Kovács, A. Fazekas, A new baseline for retinal vessel segmentation: Numerical identification and correction of methodological inconsistencies afecting 100+ papers, Medical Image Analysis 75 (2022) 102300. doi:10.1016/j.media.2021.102300.

[9] A. Hatamizadeh, H. Hosseini, N. Patel, J. Choi, C. C. Pole, C. M. Hoeferlin, S. D. Schwartz, D. Terzopoulos, RAVIR: A dataset and methodology for the semantic segmentation and quantitative analysis of retinal arteries and veins in infrared reflectance imaging, IEEE Journal of Biomedical and Health Informatics 26 (7) (2022) 3272- 3283. doi:10.1109/JBHI.2022.3163352.

[10] R. G. Nespolo, D. Yi, E. Cole, D. Wang, A. Warren, Y. I. Leiderman, Feature tracking and segmentation in real time via deep learning in vitreoretinal surgery: A platform for artificial intelligence-mediated surgical guidance, Ophthalmology Retina 7 (3) (2023) 236-242. doi: 10.1016/j.oret.2022.10.002.

[11] M. Sogabe, N. Ito, T. Miyazaki, T. Kawase, T. Kanno, K. Kawashima, Detection of instruments inserted into eye in cataract surgery using single-shot multibox detector, Sensors and Materials 34 (1) (2022) 47-54. doi:10.18494/SAM3762.

[12] H.-H. Yeh, A. M. Jain, O. Fox, K. Sebov, S. Y. Wang, PhacoTrainer: Deep learning for cataract surgical videos to track surgical tools, Translational Vision Science & Technology 12 (3) (2023) 23. doi: 10.1167/tvst.12.3.23.

[13] S. Borkovkina, A. Camino, W. Janpongsri, M. V. Sarunic, Y. Jian, Real-time retinal layer segmentation of OCT volumes with GPUaccelerated inferencing using a compressed, low-latency neural network, Biomedical Optics Express 11 (7) (2020) 3968-3984. doi: 10.1364/BOE.395279.

[14] Y. Huang, R. Asaria, D. Stoyanov, M. Sarunic, S. Bano, PseudoSegRT: Eficient pseudo-labelling for intraoperative OCT segmentation, International Journal of Computer Assisted Radiology and Surgery 18 (7) (2023) 1245-1252. doi:10.1007/s11548-023-02928-9.

[15] C. Gao, A. Saraf, J.-B. Huang, J. Kopf, Flow-edge guided video completion, in: Proceedings of the European Conference on Computer Vision (ECCV), 2020, pp. 713-729. doi:10.1007/978-3-030-58610-2\_ 42.

[16] Z. Li, C.-Z. Lu, J. Qin, C.-L. Guo, M.-M. Cheng, Towards an end-toend framework for flow-guided video inpainting, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 17541-17550. doi:10.1109/CVPR52688.2022. 01704.

[17] S. Zhou, C. Li, K. C. K. Chan, C. C. Loy, ProPainter: Improving propagation and transformer for video inpainting, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 10443-10452. doi:10.1109/ICCV51070.2023.00961.

[18] Y. Koreeda, Y. Kobayashi, S. Ieiri, Y. Nishio, K. Kawamura, S. Obata, R. Souzaki, M. Hashizume, M. G. Fujie, Virtually transparent surgical instruments in endoscopic surgery with augmentation of obscured regions, International Journal of Computer Assisted Radiology and Surgery 11 (10) (2016) 1927-1936. doi:10.1007/s11548-016-1384-5.

[19] S. M. K. Hasan, R. A. Simon, C. A. Linte, Segmentation and removal of surgical instruments for background scene visualization from endoscopic/laparoscopic video, in: Medical Imaging 2021: Image-Guided Procedures, Robotic Interventions, and Modeling, Vol. 11598, 2021, p. 115980A. doi:10.1117/12.2580668.

[20] S. M. K. Hasan, R. A. Simon, C. A. Linte, Inpainting surgical occlusion from laparoscopic video sequences for robot-assisted interventions, Journal of Medical Imaging 10 (4) (2023) 045002. doi: 10.1117/1.JMI.10.4.045002.

[21] P. Li, L. Liu, C.-B. Schönlieb, A. I. Aviles-Rivero, Optimised ProPainter for video diminished reality inpainting, in: 2024 IEEE International Symposium on Biomedical Imaging (ISBI), 2024, pp. 1-5. doi:10.1109/ISBI56570.2024.10635380.

[22] D. Xie, W. Chen, J. Zhao, X. Song, K. Wang, W. Xia, H. Liu, F. Chang, C. Li, X. Duan, Surgical instruments hyalinization: Occlusion removal in minimally invasive endoscopic surgery, Biomimetic Intelligence and Robotics 3 (3) (2023) 100105. doi:10.1016/j.birob. 2023.100105.

[23] F. J. Celdrán, J. Jiménez-Ruescas, C. Lobato, L. Salazar, J. A. Sánchez-Margallo, F. M. Sánchez-Margallo, P. González, Use of augmented reality for training assistance in laparoscopic surgery: Scoping literature review, Journal of Medical Internet Research 27 (2025) e58108. doi:10.2196/58108.

[24] A. Nankaku, M. Tokunaga, H. Yonezawa, T. Kanno, K. Kawashima, K. Hakamada, S. Hirano, E. Oki, M. Mori, Y. Kinugasa, Maximum acceptable communication delay for the realization of telesurgery, PLOS ONE 17 (10) (2022) e0274328. doi:10.1371/journal.pone. 0274328.

[25] A. Kumcu, L. Vermeulen, S. A. Elprama, P. Duysburgh, L. Platiša, Y. Van Nieuwenhove, N. Van De Winkel, A. Jacobs, J. Van Looy, W. Philips, Efect of video lag on laparoscopic surgery: Correlation between performance and usability at low latencies, The International Journal of Medical Robotics and Computer Assisted Surgery 13 (2) (2017) e1758. doi:10.1002/rcs.1758.

[26] W. Liu, J. Pan, L. Jia, S. Rao, J. Zang, Real-time tracking and inpainting network with joint learning iterative modules for ARbased DALK surgical navigation, Computer Methods and Programs in Biomedicine 272 (2025) 109068. doi:10.1016/j.cmpb.2025.109068.

[27] P.-H. Conze, P. Robert, T. Crivelli, L. Morin, Multi-reference combinatorial strategy towards longer long-term dense motion estimation, Computer Vision and Image Understanding 150 (2016) 66-80. doi: 10.1016/j.cviu.2016.04.013.

[28] B. Li, J. Han, Y. Xu, K. Rose, Optical flow based co-located reference frame for video compression, IEEE Transactions on Image Processing 29 (2020) 8303-8315. doi:10.1109/TIP.2020.3014723.

[29] M. Grammatikopoulou, E. Flouty, A. Kadkhodamohammadi, G. Quellec, A. Chow, J. Nehme, I. Luengo, D. Stoyanov, CaDIS: Cataract dataset for surgical RGB-image segmentation, Medical Image Analysis 71 (2021) 102053. doi:10.1016/j.media.2021.102053.

[30] Z.-L. Ni, G.-B. Bian, X.-H. Zhou, Z.-G. Hou, X.-L. Xie, C. Wang, Y.-J. Zhou, R.-Q. Li, Z. Li, RAUNet: Residual attention U-Net for semantic segmentation of cataract surgical instruments, in: Neural Information Processing (ICONIP), 2019, pp. 139-149. doi:10.1007/ 978-3-030-36711-4\_13.

[31] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, V. Lempitsky, Resolutionrobust large mask inpainting with Fourier convolutions, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 3172-3182. doi:10.1109/WACV51458. 2022.00323.

[32] H. Ouyang, T. Wang, Q. Chen, Internal video inpainting by implicit long-range propagation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 14559- 14568. doi:10.1109/ICCV48922.2021.01431.

[33] T. Liu, L. Liao, D. Chen, J. Xiao, Z. Wang, C.-W. Lin, S. Satoh, TransRef: Multi-scale reference embedding transformer for referenceguided image inpainting, Neurocomputing 632 (2025) 129749. doi: 10.1016/j.neucom.2025.129749.

[34] T. Liu, L. Liao, Z. Wang, S. Satoh, Reference-guided texture and structure inference for image inpainting, in: 2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 1996-2000. doi: 10.1109/ICIP46576.2022.9897592.

[35] I. I. Iordachita, M. D. de Smet, G. Naus, M. Mitsuishi, C. N. Riviere, Robotic assistance for intraocular microsurgery: Challenges and perspectives, Proceedings of the IEEE 110 (7) (2022) 893–908. doi: 10.1109/JPROC.2022.3169466.

[36] T. L. Edwards, K. Xue, H. C. M. Meenink, M. J. Beelen, G. J. L. Naus, M. P. Simunovic, M. Latasiewicz, A. D. Farmery, M. D. de Smet, R. E. MacLaren, First-in-human study of the safety and viability of intraocular robotic surgery, Nature Biomedical Engineering 2 (9) (2018) 649–656. doi:10.1038/s41551-018-0248-4.

[37] B. C. Becker, R. A. MacLachlan, L. A. Lobes, G. D. Hager, C. N. Riviere, Vision-based control of a handheld surgical micromanipulator with virtual fixtures, IEEE Transactions on Robotics 29 (3) (2013) 674–683. doi:10.1109/TRO.2013.2239552.

[38] J. W. Kim, C. He, M. Urias, P. Gehlbach, G. D. Hager, I. Iordachita, M. Kobilarov, Autonomously navigating a surgical tool inside the eye by learning from demonstration, in: 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 7351– 7357. doi:10.1109/ICRA40945.2020.9196537.

[39] C.-W. Chen, Y.-H. Lee, M. J. Gerber, H. Cheng, Y.-C. Yang, A. Govetto, A. A. Francone, S. Soatto, W. S. Grundfest, J.-P. Hubschman, T.- C. Tsao, Intraocular robotic interventional surgical system (IRISS): Semi-automated OCT-guided cataract removal, The International Journal of Medical Robotics and Computer Assisted Surgery 14 (6) (2018) e1949. doi:10.1002/rcs.1949.

[40] D. Braun, S. Yang, J. N. Martel, C. N. Riviere, B. C. Becker, Eye-SLAM: Real-time simultaneous localization and mapping of retinal vessels during intraocular microsurgery, The International Journal of Medical Robotics and Computer Assisted Surgery 14 (1) (2018) e1848. doi:10.1002/rcs.1848.

[41] Y. Long, Z. Li, C. H. Yee, C. F. Ng, R. H. Taylor, M. Unberath, Q. Dou, E-DSSR: Eficient dynamic surgical scene reconstruction with transformer-based stereoscopic depth perception, in: Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, 2021, pp. 415–425. doi:10.1007/978-3-030-87202-1\_40.

[42] Y. Wang, Y. Long, S. H. Fan, Q. Dou, Neural rendering for stereo 3D reconstruction of deformable tissues in robotic surgery, in: Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, 2022, pp. 431–441. doi:10.1007/978-3-031-16449-1\_41.

Table B1  
Reference choice with alignment, masks and samples held fixed.
<table><tr><td>Selection policy</td><td>PSNR ↑</td><td>Δ</td><td>SSIM ↑</td><td>MAE↓</td></tr><tr><td>Nearest valid reference (ours)</td><td>33.42</td><td></td><td>0.933</td><td>2.93</td></tr><tr><td>Random valid reference</td><td>31.06</td><td>-2.36</td><td>0.910</td><td>4.64</td></tr><tr><td>Oldest valid reference</td><td>29.19</td><td>-4.23</td><td>0.888</td><td>6.37</td></tr><tr><td>Oracle on pre-alignment agreement</td><td>33.35</td><td>-0.07</td><td>0.931</td><td>3.02</td></tr></table>

## A. Evaluation of optional learned refinement

We evaluated a compact U-Net with windowed reference attention as an optional refinement of the safe composite. The network operates on a 256 px region of interest around the mask and is trained on the triplets from Section 3.6. Its result is returned as a separate image layer with every altered pixel marked as synthesised.

On the 178 validation samples with ground-truth masks, evaluated in this 256 px refinement configuration rather than the native-resolution setting of Table 3, the unrefined safe composite reaches 33.44 dB masked PSNR and the learned refinement reaches 33.43 dB. The learned variant has a mean end-to-end latency of 55.8 ms and a 95th percentile of 86.2 ms on the RTX 5000 Ada used for all timing experiments. Because the refinement was trained on triplets that include the validation recording (Section 3.1), this comparison may favour it. The safe composite provides the higher measured fidelity while preserving direct source traceability, supporting its selection as the reported TransGHOST output.

## B. Reference selection as a measured variable

Table B1 varies the reference choice while holding the alignment stack, masks and samples fixed.

The spread in masked PSNR between the best and worst valid choices is 4.23 dB, confirming that reference choice afects fidelity. Recency reaches the highest masked PSNR among the evaluated policies, including the oracle that selects on highest pre-alignment agreement, which is 0.07 dB lower. Pre-alignment agreement is a proxy rather than the scored quantity, so a policy that maximises it need not maximise masked PSNR after alignment. When all available references are unobstructed, the newest valid reference is therefore the choice the measurements support.

Every reference in Table B1 is unobstructed, so mask overlap is zero and the regional criterion reduces to recency. Section 6.5 evaluates the overlap term separately on references that may contain an instrument over the target region.

Together, the experiments show how the selector adapts to candidate visibility. With unobstructed references, the regional criterion reduces to recency by design and avoids the lower fidelity of older or random choices. When references may occlude the target region, the regional term becomes active and improves on recency by 3.60 dB.

## C. Evaluation of alternative configurations

Table C1 reports alternatives measured before the latency changes in Appendix D. Their latency values can therefore be compared with one another but are higher than the deployed percentiles in Section 6.8, while the later change alters masked PSNR by 0.006 dB. This table and Appendix D record the pre-change configuration in separate timing sessions, which is why its 95th percentile appears here as 99 ms and there as 96.5 ms, a diference within the repeat spread stated in Appendix F.

The nearest single reference matches or exceeds median fusion and blockwise selection among five aligned references while achieving the lowest 99th-percentile latency. After registration, the remaining correspondence error is systematic rather than independent across references, explaining why averaging adds little information. The same designselection experiment favours direct geometric compositing over a photometric consistency threshold and partialtransparency correction: the threshold can suppress valid reconstruction at short lags, while boundary unmixing depends on additional alpha and instrument-colour estimates and has approximately three times the latency.

Local correspondence fitted to visible pixels outside the mask outperforms dense optical flow within the mask by 0.3 to 1.1 dB across the evaluated smoothing settings. The visible-region formulation avoids following instrument motion in the current frame and supports the local-ofset design used in Table 3.

Efect of block granularity. With access to ground truth, an oracle choice of translation per block improves masked PSNR by +1.5 to +2.9 dB as the block size decreases from 64 to 16 pixels, indicating potential benefit from finer local correspondence. We evaluated practical block sizes of 16, 24, 32 and 48 pixels at search resolutions of 0.35, 0.50 and 0.70 of native resolution, covering eight of the twelve combinations. The deployed setting achieves the highest measured fidelity at 31.89 dB, compared with 31.64 to 31.86 dB across the alternatives, while also avoiding the higher latency of finer settings. At block sizes below 48 pixels, the visible support becomes too sparse for stable matching. The oracle experiment therefore distinguishes the potential of groundtruth correspondence from practical selection based on visible support and supports the current operating point among the evaluated settings.

## D. Latency engineering

Alignment cost scales with the warped area, which varies by two orders of magnitude: the median is 140,000 pixels, the 95th percentile is 754,000, and the maximum is 1.35 million. Because the largest regions determine the latency tail, we apply two changes to reduce their computational cost.

The residual-translation search uses only a band of visible pixels outside the mask. The original implementation nevertheless warped the complete region at each stage to obtain these samples. We instead sample the reference directly at the band coordinates and evaluate all ofsets in one batched interpolation, making cost proportional to the number of band pixels rather than the full region area. Direct sampling also evaluates the ofset in the same reference-coordinate system in which it is ultimately applied.

Table C1  
Design-selection experiments comparing the reported configuration with measured alternatives.
<table><tr><td>Variant</td><td>PSNR ↑</td><td>Δ</td><td>p95 (ms) ↓</td><td>p99 (ms) ↓</td></tr><tr><td colspan="5">Multi-reference evaluation set, one reference per candidate lag</td></tr><tr><td>Single nearest reference (ours)</td><td>33.42</td><td></td><td>96</td><td>120</td></tr><tr><td>Median fusion of 5 references</td><td>33.39</td><td>-0.03</td><td>109</td><td>744</td></tr><tr><td>Per-block choice among 5 references</td><td>33.42</td><td>0.00</td><td>109</td><td>343</td></tr><tr><td colspan="5">Base validation set, deployed configuration</td></tr><tr><td>Deployed configuration (ours)</td><td>31.89</td><td></td><td>99</td><td>118</td></tr><tr><td>+ photometric consistency gate</td><td>31.14</td><td>-0.75</td><td>110</td><td>122</td></tr><tr><td>+ alpha unmixing at the boundary</td><td>31.81</td><td>-0.08</td><td>245</td><td>266</td></tr></table>

![](images/3b558a8f59c86cdabd8dfdf123c61806eadf6d84cd96e83cb0f20a805c444430.jpg)  
Figure D1: End-to-end latency percentiles for three implementations of the alignment search on the validation set. The median changes by several milliseconds, whereas the 99th percentile decreases by approximately one quarter.

The blockwise search compares a decimated region for every ofset. With fractional decimation alone, large regions remain far more expensive than small ones, so an absolute limit is also placed on the number of pixels compared per ofset. Regions below this limit remain unchanged, while only the largest are further coarsened after the global subpixel stage has estimated fine displacement.

Figure D1 shows that, on the validation set, sparse band sampling reduces the 95th-percentile end-to-end latency from 96.5 ms to 85.5 ms and the 99th percentile from 115.4 to 101.4 ms, with a masked-PSNR change of 0.006 dB. An additional absolute pixel limit provides a second operating point with a further 6 ms reduction at the 95th percentile, at approximately 0.06 dB lower masked PSNR. The reported default retains the higher-fidelity configuration.

Table D1 reports the initial and confirmatory measurements on the held-out cohort. The initial full configuration reached 29.83 dB masked PSNR at 102.7 ms for the 95th percentile, while the reduced configuration met the latency target at 29.25 dB masked PSNR. This comparison motivated the latency analysis and the resulting sparse-search changes, which were designed and selected on the validation set with a masked-PSNR efect of 0.006 dB.

Before the confirmatory held-out evaluation, we specified three acceptance conditions: masked PSNR within 0.10 dB of the initial result, a 95th percentile below 100 ms, and unchanged mask metrics because segmentation was not modified. All three conditions were met: the full configuration reaches 29.83 dB masked PSNR at 86.1 ms for this percentile, with masked PSNR unchanged to two decimal places. The reduced alternative decreases latency by 3 ms and reaches 29.26 dB masked PSNR. Together with the validation-selected optimisation described above, this predeclared confirmation supports the reported real-time operating point.

## E. Fidelity against reference age

Table E1 reports the numerical values underlying the curve in Section 6.7, including the sample count at each lag and the fixed-cohort control that separates reference age from the changing cohort size.

## F. Evaluation protocol details

This appendix provides the complete evaluation protocol, including the rationale for masked scoring and targetreference separation.

Rationale for masked scoring. Compositing replaces only masked pixels and leaves the remainder of the frame unchanged. Because masks occupy only a few per cent of the frame, whole-frame metrics are dominated by pixels that no method alters. All fidelity results in Section 6 are therefore computed over the instrument mask. Masked LPIPS uses the mask-centred crop defined below so that boundary seams remain visible to the metric.

Rationale for a composited benchmark. Restoration fidelity requires a known view of the tissue behind an instrument. We therefore place a real instrument cutout over an instrument-free frame from a held-out clip and use the unobstructed frame as the scoring target. The supplied reference must be a diferent image, and samples that violate this condition are excluded before evaluation. This controlled semi-synthetic construction combines real surgical imagery with an exact pixel-level target.

Table D1  
Initial and confirmatory end-to-end latency on the held-out cohort.
<table><tr><td>Configuration</td><td>PSNR ↑</td><td>p50 (ms) ↓ p95 (ms) ↓ p99 (ms) ↓</td><td></td><td></td></tr><tr><td colspan="5">Initial measurement</td></tr><tr><td>Without per-block local offsets</td><td>29.25</td><td>57.6</td><td>87.2</td><td>100.5</td></tr><tr><td>Full configuration</td><td>29.83</td><td>61.0</td><td>102.7</td><td>116.9</td></tr><tr><td colspan="5">Confirmatory measurement, sparse band sampling</td></tr><tr><td>Without per-block local offsets</td><td>29.26</td><td>52.1</td><td>83.1</td><td>90.0</td></tr><tr><td>Full configuration</td><td>29.83</td><td>55.9</td><td>86.1</td><td>101.2</td></tr></table>

Metric definitions. Masked PSNR, SSIM and MAE are computed over the true instrument mask. For LPIPS, the mask bounding box is padded by 64 pixels, clamped to the frame and resized to 256 × 256 before evaluation with LPIPS(VGG). Including context outside the mask allows the metric to capture seams at the restoration boundary. Boundary MAE uses a narrow band around the mask edge, while false-positive MAE reports reconstruction error at pixels included by the predicted mask but absent from the true mask. The two supports jointly characterise masked fidelity and predicted-mask extension.

Learned baseline implementation. LaMa uses the released Big-LaMa generator weights at a maximum image side of 1920 pixels. The real-time-constrained TransRefinspired generator uses the project-fine-tuned checkpoint at its 256 px operating resolution. It receives the same reference and ground-truth mask as TransGHOST, and its prediction is resized to the native frame before masked compositing. Both learned baselines are evaluated in inference mode on the same samples and with the same masked metrics. Table 4 reports this matched fidelity comparison, while Section 6.8 reports synchronised complete-pipeline latency for Trans-GHOST.

Latency protocol. Wall-clock timings are measured on one otherwise-idle RTX 5000 Ada at native 1920 × 1080 resolution. Eight warm-up passes are discarded, and the device is synchronised before and after each measured frame.

Repeated measurements vary by approximately 2 to 3 ms at the 95th percentile, which defines the resolution used to interpret latency diferences.

Reproducibility contact. Questions about the protocol or requests concerning the evaluation sets may be addressed to Niran Nataraj (niran@g.ecc.u-tokyo.ac.jp) or Maina Sogabe (fsogabe-vet@g.ecc.u-tokyo.ac.jp).

Reference-age analysis at 30 fps. The fixed-cohort block contains samples available at every lag. Intervals are 95 per cent clipclustered paired bootstrap estimates against lag 1.
<table><tr><td colspan="7"></td><td colspan="2">Fixed cohort</td></tr><tr><td>Lag</td><td>n</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MAE↓</td><td>Bnd. MAE ↓</td><td>PSNR ↑</td><td>∆ vs lag 1 [95% CI]</td></tr><tr><td>1</td><td>178</td><td>33.42</td><td>0.933</td><td>2.93</td><td>1.58</td><td>33.37</td><td></td></tr><tr><td>2</td><td>171</td><td>31.95</td><td>0.920</td><td>3.75</td><td>2.04</td><td>31.94</td><td>-1.43 [-2.93, -1.21]</td></tr><tr><td>3</td><td>166</td><td>31.20</td><td>0.910</td><td>4.35</td><td>2.38</td><td>31.16</td><td>-2.22 [-4.75, -1.95]</td></tr><tr><td>5</td><td>150</td><td>30.12</td><td>0.896</td><td>5.47</td><td>2.96</td><td>30.17</td><td>-3.20 [-7.02, -3.04]</td></tr><tr><td>8</td><td>120</td><td>28.86</td><td>0.884</td><td>6.85</td><td>3.65</td><td>28.93</td><td>-4.45 [-7.89, -3.96]</td></tr><tr><td>12</td><td>84</td><td>28.06</td><td>0.883</td><td>7.36</td><td>3.99</td><td>28.06</td><td>-5.31 [-9.46, -4.94]</td></tr></table>