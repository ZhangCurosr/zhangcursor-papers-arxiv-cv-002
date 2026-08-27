# THA-Flow Generative Model: Prosthesis Geometry Prediction from Preoperative CT

Yiping Wang<sup>1,\*</sup>, Jie Li<sup>2</sup>, Jingyu Shen<sup>1</sup>, Liao Wang<sup>2,\*</sup>

<sup>1</sup>Changzhou Jinse Medical Information Technology Co., Ltd., Changzhou 213000, Jiangsu, China <sup>2</sup>Shanghai Key Laboratory of Orthopaedic Implants, Department of Orthopaedic Surgery, Shanghai Ninth People’s Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China

<sup>\*</sup>Correspondence: Yiping Wang (yiping.wang@mdivi.cn) or Liao Wang (wang821127@163.com)

## Abstract

Preoperative planning for total hip arthroplasty (THA) is commonly framed as selecting a single prosthesis configuration and placement for a patient’s osseous anatomy. In practice, however, the same anatomy may admit several clinically reasonable solutions, making planning inherently a one-to-many problem that is better represented by a conditional probability distribution. We present THA-Flow, a conditional flow-matching model that generates three-dimensional prosthesis geometry directly from preoperative CT. Separate AutoencoderKL models compress preoperative bone anatomy and prosthesis geometry, while a three-dimensional UNet learns a rectified flow from Gaussian noise to the prosthesis latent space under spatial bone conditioning and optional structured prosthesis parameters. The retrospective cohort comprised 1,355 hips from 1,149 patients undergoing primary THA. Following rigid registration of postoperative CT to preoperative CT, the actual postoperative prostheses were transformed independently according to the pelvic and femoral registrations and represented as a dual-channel truncated signed distance field. The prosthesis autoencoder achieved a peak signal-to-noise ratio of 47.11 dB and a structural similarity index of 0.9964 on the validation set. Complete acetabular and femoral geometries were generated across seven major stem models representing 93.4% of the cohort. Repeated bone-conditioned sampling preserved component position, alignment, and the principal bone–prosthesis interfaces while allowing limited local geometric variation. To our knowledge, THA-Flow represents the first application of generative AI to three-dimensional surgical planning for THA.

Keywords: Total hip arthroplasty; Surgical planning; Patient-matched prosthesis; Flow-matching model

## 1 Introduction

Preoperative planning for THA requires selection of prosthesis shape and size together with its spatial placement relative to patient anatomy. Current digital workflows rely primarily on twoor three-dimensional measurements, heuristic rules, and matching against commercial prosthesis libraries[1]. CT-based systems such as AIHIP apply deep learning to pelvic and femoral segmentation and anatomical landmark detection, but component selection and placement remain downstream procedures based on measurements, explicit planning rules, and predefined prosthesis inventories[2]. These systems improve the eficiency of image processing and measurement, yet still converge on a single component configuration and placement and therefore do not represent the distribution of plausible plans for the same anatomy.

Generative models ofer a diferent formulation by permitting repeated sampling from a conditional distribution. THA-Net established the feasibility of synthesizing simulated postoperative radiographs from preoperative radiographs, with both unconditional and prosthesis-specified generation[3]. Its output, however, remains a two-dimensional synthetic image. Generating the entire postoperative image also redraws bone, soft tissue, and the prosthesis interface, so realistic image appearance does not guarantee preservation of the patient’s original osseous anatomy. Conditioning explicitly on the preoperative bone while restricting the target to three-dimensional prosthesis geometry allows the prediction to be superimposed directly on the unaltered patient

anatomy.

The primary obstacle is the construction of spatially consistent three-dimensional supervision. Hip posture difers between preoperative and postoperative CT, while femoral neck resection and metal artefacts further undermine direct registration. The postoperative prosthesis can provide voxel-level supervision only after its acetabular and femoral components have been returned independently to the preoperative coordinate space with the pelvis and femur. High-resolution threedimensional volumes also impose substantial memory and computational demands; latent-space generation reduces this burden while preserving the principal spatial structure[4, 5]. MAISI and related work have further demonstrated spatially controlled generation of large three-dimensional medical images, with rectified flow reducing the number of sampling iterations[6, 7]. Compared with synthesizing a complete postoperative CT containing metal artefacts, soft-tissue texture, and complex acquisition-specific features, generating only an implicit representation of the target prosthesis geometry provides a more clearly defined and tractable objective. The output can also be converted directly into a surface for subsequent measurement, retrieval, and geometric analysis.

This study focuses on the three-dimensional generative formulation and technical validation of THA-Flow. Its contributions are threefold: a postoperative-to-preoperative registration pipeline that excludes metal-contaminated and surgically altered regions to establish reliable supervision in preoperative space; a dual-channel TSDF representation in which acetabular and femoral components follow the pelvis and femur, respectively; and a latent rectified-flow model in which the bone latent is injected as a spatial condition from the first convolutional layer, with optional stem model and size as additional semantic controls. We evaluate generated geometry across stem designs, at the bone–prosthesis interface, and through conditional distribution consistency; latent reconstruction and velocity-field errors serve as technical checks on model training.

## 2 Results

## 2.1 Conditional Inference Architecture

In the intended planning workflow, THA-Flow takes the preoperative CT as its primary input (Fig. 1). A bone encoder first extracts anatomy-preserving spatial features, while random noise initializes the prosthesis state. When a femoral stem model or size is specified, a condition encoder converts these inputs into additional semantic constraints. The three-dimensional generator integrates the anatomy, optional product information, and evolving prosthesis state, and the prosthesis decoder then reconstructs three-dimensional geometry in the coordinate system of the preoperative CT for direct inspection against the patient’s bone.

The same model supports three clinically relevant modes. With preoperative CT alone, it produces a patient-matched candidate from the bone geometry. Adding the stem model constrains the generated shape toward the specified product family. Adding both model and size further restricts stem dimensions while preserving anatomical fit. Figure 1 shows all three outputs for the same patient. Component position, alignment, and overall extent remain anchored by the bone condition, whereas the product condition modulates local stem geometry. On the single-GPU system used in this study, median latent integration time across the three modes was 0.31 s, and median generation time including sliding-window decoding was 1.76 s per case, supporting interactive candidate generation; model loading and preprocessing were excluded.

![](images/a135538dec22944250313a3884b8c3c1e0b27490c0957c0e5ae06e6385778555.jpg)  
Figure 1: Conditional inference architecture and generation modes of THA-Flow. a, Preoperative CT is encoded as a spatial bone condition, while the optional femoral stem model and size are encoded as semantic constraints. The three-dimensional generator transports a random latent state to a prosthesis latent, which is decoded in preoperative CT space. b, Outputs for the same patient under bone-only, bone-plus-model, and bone-plus-model-and-size conditions. Green denotes the acetabular cup and femoral head; blue denotes the femoral stem, separated from the head at the neck.

The discontinuity at the neck in Fig. 1 is a deliberate consequence of the supervisory coordinate system rather than a generation failure. The actual postoperative prosthesis is registered to preoperative CT with the pelvis and femur independently. The model therefore learns the geometry associated with each bone in the pathological preoperative joint configuration, not the assembled construct after joint reduction. Keeping the cup and femoral head in one channel preserves the postoperative hip centre and head–cup spatial relationship, avoiding separate estimation of their narrow clearance and relative pose. Separating this channel from the stem at the neck decouples patient-matched prosthesis generation from joint reduction and assembly. The two channels are nevertheless generated jointly in a shared coordinate system and can therefore encode their assembly relationship implicitly. Their relative displacement at the neck also provides a direct reference for post-processing: its longitudinal component represents the leg-length correction required to move from the preoperative pathological posture to the reduced construct.

## 2.2 Patient-Matched Generation Across Stem Designs

The seven most prevalent femoral stem models accounted for 1,265 hips, or 93.4% of the cohort. One validation case from each model was used to compare the preoperative CT, the actual postoperative prosthesis registered to preoperative space, bone-only generation, and generation conditioned jointly on bone and the recorded stem model (Fig. 2). Both generative modes produced complete acetabular and femoral geometry, with stem position and alignment adapting to the proximal femur and medullary canal. Model conditioning additionally constrained stem length, proximal width, and lateral profile without disrupting anatomical fit, demonstrating that a single generator can accommodate multiple widely used product families.

A bone-only result need not reproduce the stem design selected at the actual operation, yet it can retain a plausible intramedullary position, alignment, and spatial extent and thus represent another candidate permitted by the anatomy. Conditioning on the recorded stem model shifts the generated geometry toward the corresponding design features. The postoperative prosthesis remains one clinically implemented solution rather than the sole correct answer for preoperative planning. Pairwise agreement with that solution is informative about model behaviour, but should not be the only criterion for the clinical plausibility of a generated plan.

Preoperative CT

Registered prosthesis

Bone-conditioned prediction

Bone + stem-model prediction

![](images/b828d0491159823a9328736302f30c076da913182b6ee0fbce170b6d0d9e73a5.jpg)  
Figure 2: Patient-matched generation across major femoral stem designs. Each row represents one recorded postoperative stem model: Corail, SUMMIT, Accolade TMZF, and Tri-Lock from top to bottom. Columns show the preoperative CT, the actual prosthesis registered to preoperative space, bone-only generation, and generation conditioned on bone and the recorded stem model. The two generated results use the same initial noise; neither receives stem size or other component parameters. Green and blue denote the acetabular and femoral prosthesis channels, respectively.

![](images/10acd854d622511a548e05c646410b8fd099bb4c16c9ecba9cdce91acee93e80.jpg)  
Figure 2: Patient-matched generation across major femoral stem designs (continued). Secur-Fit, Wagner Cone, and M/L Taper are shown from top to bottom. Column definitions, conditioning modes, and colour coding are identical to those in Fig. 2.

## 2.3 Bone–Prosthesis Fit and Fill

Fit and fill at the bone–prosthesis interface are central to evaluating prosthesis geometry and placement. The efective canal is the anterolateral region of the proximal medullary canal that accommodates the stem and contributes to fixation, with its posteromedial boundary defined by the femoral calcar[10]. Consecutive axial sections from the same patient show that the generated stem did not simply occupy the entire canal. Proximally, it entered the efective canal and expanded along the regions intended for contact without encroaching beyond the calcar boundary; distally, it retained a continuous, regular taper rather than deforming unnecessarily to every local variation in the canal wall (Fig. 3). Consecutive acetabular sections showed a complete cup whose outer surface closely followed the curvature of the acetabular wall. Greater fill is not inherently better, and the expected contact regions vary with stem design and fixation philosophy. A single overlap score or surface distance cannot fully characterise interface quality; review of consecutive sections and anteroposterior projections by clinical experts therefore remains important. The observed selective proximal fit and fill, respect for the calcar boundary, and regular distal geometry are consistent with the requirements for initial stability and avoidance of cortical impingement in cementless stems.

The coronal anteroposterior projections further show the medial border of the generated stem closely following the dense principal compressive trabecular system that extends inferiorly from the medial femoral neck, while the distal stem retains a regular profile. Piecewise CT intensity normalisation increased contrast across the transition from cancellous bone to cortex, allowing the canal boundary, cortex, and load-bearing trabeculae to act as distinct spatial cues. Truncated distance-field supervision concentrated learning near the prosthesis surface and reduced the influence of remote background voxels. Together, these choices encouraged region-specific fit and fill rather than prediction of only a coarse component centre, axis, or canal occupancy.

![](images/589e410cf1f5f60fcdae34984def9853ccc84c1f0a59eb2ac3d1f2a8d64c0351.jpg)  
Figure 3: Bone–prosthesis fit and fill on consecutive sections and coronal projections. The upper half shows the actual prosthesis registered to preoperative space; the lower half shows the generated prosthesis conditioned on bone and the Corail stem model. Axial sections on the left progress from the distal stem to the acetabulum, while coronal projections on the right show overall component position and alignment. The femoral channel in blue demonstrates fit within the efective canal, distal stem geometry, and respect for the calcar boundary; the acetabular channel in green demonstrates the spatial relationship between the cup and the acetabular wall.

## 2.4 Conditional Distribution Consistency

Retrospective postoperative CT records one plan implemented from among several potentially acceptable solutions, rather than a unique ground truth for the same preoperative anatomy. Voxel overlap or surface distance to that single clinical solution would count other plausible candidates as errors and is therefore insuficient for evaluating a probabilistic generator. A more appropriate criterion is conditional distribution consistency: repeated samples for one anatomy should remain concentrated in component position, alignment, principal bone–prosthesis interfaces, and overall scale, while retaining limited variation in local contour and length; across patients, the generated distribution should remain consistent with the clinical distribution of actual postoperative pros-

theses.

Repeated sampling across validation cases showed that most generated regions overlapped the registered actual prosthesis. Non-overlapping contours were concentrated near the cup rim, proximal stem boundary, and stem tip, forming a narrow distribution band (Fig. 4). Across cases, cup and stem position and stem alignment remained stable, whereas local shape diferences represented neighbouring solutions permitted by the same anatomy. Diversity and convergence are therefore complementary: random initial noise explores local variation within the conditional distribution, while spatial bone conditioning anchors the common anatomical position and principal geometric extent. For a generative planning model, this distribution-level consistency is more informative than requiring every sample to reproduce a single postoperative solution.

Short-stem case  
![](images/0c98f909aed86f8f70a22056967a73a477745283cc4de8fcec8e9d962511b932.jpg)

Long-stem case  
![](images/826883e8e43880944d3259f93323736c55e558c6bfaa4bfae9f52e55d3117aca.jpg)  
Figure 4: Within-patient bone-conditioned distribution relative to the clinical solution. A short-stem case is shown on the left and a long-stem case on the right. Green and blue solids represent the actual acetabular and femoral components registered to preoperative space. Coloured lines delineate portions of generated projections extending beyond the actual prosthesis under diferent initial noise samples; overlapping regions are not redrawn. Inference uses the preoperative bone condition only, without prosthesis parameters.

## 3 Discussion

This study reformulates THA planning from discriminative prediction of a single model or placement into conditional generation of a three-dimensional prosthesis distribution. THA-Flow produced complete acetabular and femoral geometry across cases representing seven major stem models. Component position and alignment adapted to individual anatomy, the principal bone– prosthesis interfaces remained selectively matched, and repeated samples exhibited limited local variation under the same bone constraint. These findings indicate that the model learned a patientmatched conditional geometry distribution rather than a fixed template.

This formulation difers from two established pathways. Systems such as AIHIP use deep learning for segmentation and landmark detection before applying measurements, rules, and a standard prosthesis library to derive a single plan[2]. THA-Net introduced generative modelling to THA, but generates simulated two-dimensional postoperative images[3]. THA-Flow instead confines generation to an implicit three-dimensional field around the prosthesis surface, without redrawing bone, soft tissue, or metal artefacts. The output can be overlaid directly on the original preoperative CT or recovered as a solid from its zero level set, making it a more direct geometric object for surgical planning. To our knowledge, this is the first application of generative AI to three-dimensional planning for THA.

Reliable supervision in preoperative space is fundamental to the task. Because the pelvis and femur change pose independently between examinations, separate registration prevents joint motion from being confounded with variation in prosthesis placement. The dual-channel representation returns the acetabular components with the pelvis and the stem with the femur. It separates patient-matched generation from joint reduction at the neck while preserving the postoperative hip centre and head–cup spatial relationship as references for subsequent reduction and assembly.

Strong spatial conditioning and truncated distance-field supervision jointly shape the local bone–prosthesis relationship. Introducing the bone latent from the first convolution allows the canal wall, femoral calcar, load-bearing trabeculae, and acetabular wall to constrain geometry directly. Generated stems consequently showed selective fit and fill within the efective canal rather than indiscriminate canal occupation. Because appropriate contact regions vary with stem design and fixation philosophy, neither overlap nor surface distance alone fully captures interface quality. Moreover, postoperative CT records only one of several viable clinical solutions. Repeated sampling preserved component position, alignment, and the principal interfaces while allowing limited boundary variation, supporting conditional distribution consistency rather than agreement with a single postoperative result as the more appropriate concept of generative accuracy.

Structured parameters provide an additional level of control. Specifying a stem model modifies stem length, proximal width, and lateral profile while preserving anatomical fit, indicating that discrete product information can serve as an additional control over the conditional geometry distribution.

Several known limitations remain. First, the single-centre cohort may limit generalisability across patient populations, imaging protocols, and prosthesis inventories. Second, the strong spatial condition introduced by channel-wise bone concatenation deliberately prioritises patient matching, but consequently weakens the ability of model and size conditions to preserve standard product geometry. When a specified model and size are poorly compatible with the patient’s anatomy, the generated result may represent an anatomically adapted deformation of that design rather than a faithful reproduction of its catalogue geometry. Finally, the model outputs continuous geometry rather than a directly actionable commercial model and size. Bone-only results therefore require geometric measurement or retrieval against a standard prosthesis library to identify the nearest available model and size, followed by component reduction and assembly to complete the surgical plan.

## 4 Methods

## 4.1 Two-Stage Training Architecture

Training used two stages: latent-space pretraining and conditional rectified-flow matching (Fig. 5). Separate AutoencoderKL models were first trained for bone and prosthesis geometry. The same prosthesis encoder–decoder processed the acetabular and femoral TSDF channels independently, after which their two four-channel encodings were concatenated into the eight-channel prosthesis latent y. After reconstruction was validated, both autoencoders were frozen. Linear path states y<sub>τ</sub> were sampled between y and Gaussian noise ϵ, concatenated voxel-wise with the bone latent x, and passed to a three-dimensional UNet trained to predict the prosthesis latent velocity y ϵ. Six structured prosthesis parameters were mapped by a separate encoder to condition tokens and injected through cross-attention at the deepest network level. Random condition dropout enabled one generator to learn unconditional, bone-conditioned, and partially or fully parameter-

conditioned distributions.

![](images/42f1a0a620dabcb3d730761f21c41b825a1abebc68cec9cb1b868576f13b50a3.jpg)  
Figure 5: Latent pretraining and conditional rectified-flow matching. a, Bone and prosthesis autoencoders (AutoencoderKL) learn latent representations of preoperative CT and dual-channel prosthesis TSDF, respectively, and are frozen after reconstruction training. b, The bone latent is concatenated channel-wise with the current prosthesis path state. Stem model, size, and component parameters are encoded as semantic tokens. A three-dimensional UNet learns the conditional rectified flow from Gaussian noise to prosthesis geometry by velocity matching.

## 4.2 Study Design and Ethics

This single-centre retrospective imaging study analysed CT examinations and operative records acquired during routine clinical care and involved no additional intervention. Eligible cases had temporally proximate preoperative and postoperative CT examinations from the same primary THA episode. Direct identifiers were removed before processing, and no research image contains an original accession number, patient name, or facial information.

## 4.3 Cohort Construction and Data Split

DICOM series were converted to three-dimensional NIfTI volumes, and axial and coronal digitally reconstructed projections were generated for rapid screening. Preoperative and postoperative examinations from the same operation were matched manually using examination dates, operative side, and imaging findings. Derived series, insuficient scan coverage, fractures within critical registration regions, and intentional intraoperative osseous alterations were excluded. TotalSegmentator was used only for coarse localisation of the operated hemipelvis and femur and identification of regions of interest, not to provide prosthesis supervision[8].

Of 1,483 primary THA hips initially screened, 128 were excluded during CT pairing, prosthesisrecord verification, rigid-registration quality control, or training-sample completeness checks. The final cohort comprised 1,355 hips from 1,149 patients, including 649 left and 706 right hips. The training, validation, and held-out test sets contained 1,188, 84, and 83 hips, respectively (Table 1). The test set remained sealed and was not used for model selection or figure preparation.

Table 1: Distribution of included hips across the training, validation, and held-out test sets.
<table><tr><td>Operated side</td><td>Training Validation</td><td>Test</td><td></td><td>Total</td></tr><tr><td>Left</td><td>572</td><td>46</td><td>31</td><td>649</td></tr><tr><td>Right</td><td>616</td><td>38</td><td>52</td><td>706</td></tr><tr><td>Total</td><td>1,188</td><td>84</td><td>83</td><td>1,355</td></tr></table>

Nine stem models used for model-stratified splitting covered 1,280 hips. The remaining 75 hips were not stratified: 73 lacked a reliable stem-model record and two belonged to extremely infrequent records. Validation and test cases were held out by stem-model strata to maintain coverage of major designs. Corail, SUMMIT, Accolade TMZF, and Tri-Lock together accounted for 1,106 hips, or 81.6% of the cohort (Table 2).

Table 2: Femoral stem model distribution in the included cohort.
<table><tr><td>Stem model</td><td>Hips</td><td>Stem model</td><td>Hips</td></tr><tr><td>DePuy Corail</td><td>362</td><td>Not model-stratified</td><td>75</td></tr><tr><td>DePuy SUMMIT</td><td>343</td><td>Wagner Cone</td><td>53</td></tr><tr><td>Stryker Accolade TMZF</td><td>218</td><td>Zimmer M/L Taper</td><td>25</td></tr><tr><td>DePuy Tri-Lock</td><td>183</td><td>Zimmer Wagner SL</td><td>9</td></tr><tr><td>Stryker Secur-Fit</td><td>81</td><td>DePuy S-ROM</td><td>6</td></tr></table>

## 4.4 Prosthesis Metadata and CT Verification

Femoral stem model and size, acetabular cup, femoral head, and liner information were first extracted from itemised billing records for the same admission. Product catalogue numbers in billing entries were less susceptible to free-text transcription errors and were therefore used as the primary index. Publicly available manufacturer catalogues then mapped each catalogue number to stem model, model-specific size, cup outer diameter, head diameter, head ofset, and liner ofset. Fields that could not be established reliably from catalogue numbers and manufacturer documentation remained unlabelled and were excluded from parameter conditioning by field-specific masks during training.

Descriptive statistics covered nine stem models and 70 model-specific sizes. Cup outer diameter and head diameter were recorded or verified from CT for all 1,355 hips, with ranges of 40–62 mm and 22–40 mm, respectively. Cup diameters were concentrated at 48, 50, 52, and 54 mm (306, 293, 263, and 202 hips), while head diameters were predominantly 32 and 36 mm (611 and 572 hips). Head ofset was labelled in 1,290 hips and ranged from 5 to +8.5 mm, with +5 and +1.5 mm accounting for 399 and 376 hips. Liner ofset was labelled in 1,277 hips: 1,091 had 0 mm and 186 had +4 mm. Liner material was retained only as a descriptive variable; ceramic and polyethylene liners accounted for 777 and 578 hips and were not used as generative conditions. Missing fields naturally contributed partially conditioned examples during training.

Cup outer diameter and head diameter were further verified three-dimensionally on postoperative CT. Using the billing record as an initial estimate, a 0.2-mm isotropic volume was resampled around the femoral head centre, and candidate head spheres and hemispherical cup shells were projected onto sagittal, coronal, and axial sections. NVIDIA Warp (the warp-lang Python package)[15] kernels measured high-density metal occupancy inside and outside each candidate geometry throughout the full volume while jointly optimising head centre, cup axis, cup diameter, and fitting ofset. An interactive Streamlit interface displayed all three orthogonal planes and allowed position, orientation, and dimensions to be reviewed in increments of 0.25 mm or 0.25◦. In the example in Fig. 6, a billed cup diameter of 54 mm was corrected to the CT-supported value of 56 mm. Head and liner ofsets retained their catalogue-derived values; the image-derived liner ofset was used only for cup-shell localisation and not as a conditioning label.

a Sagittal  
![](images/cde2c7e3f6ea51136a405635340f040b69f048419ed8cf830adba7a59adc7d54.jpg)

b Coronal  
![](images/8adb9807eeb5643cb845e5d693025fcf773f362d4e70be1e55e9795a27703bc5.jpg)

c Axial  
![](images/cf3c69f529c15feee1ce04324fd10b24ea824d56669acf3153917b5de6a95607.jpg)  
Blue: fitted acetabular shell

d 3D geometric fit  
![](images/482ba6f4cdc920828e5faccf965bd00a1450f2412b32c2a6887eb568208563bf.jpg)  
CT-verified $D _ { c u p } = 5 6$ mm  
Figure 6: Verification of prosthesis component parameters on postoperative CT. a–c, Candidate femoralhead spheres and hemispherical cup shells projected onto sagittal, coronal, and axial sections; fitting scores were computed over the complete three-dimensional volume. d, Definitions of cup outer diameter $D _ { \mathrm { c u p } } ,$ head diameter $D _ { \mathrm { h e a d } } ,$ , and fitted centre ofset $\delta _ { \mathrm { f i t } }$ along the cup axis. In this case, the billed cup diameter of 54 mm was corrected to a CT-measured diameter of 56 mm; $\delta _ { \mathrm { f i t } }$ was used only for cup-shell localisation.

## 4.5 Independent Pelvic and Femoral Registration

Metal artefacts from the postoperative cup and stem contaminate adjacent bone surfaces, precluding use of the complete postoperative bone surface. On the pelvic side, the inferior ischium provided an initial extent alignment, after which surfaces common to both scans, remote from cup artefacts, and unafected by surgery were sampled. On the femoral side, volumes were aligned by their proximal extent; the lesser trochanter and adjacent shaft served as the principal rotationspecific anatomy, while the femoral neck osteotomy and metal-contaminated proximal regions were excluded. Both bones were registered using rigid iterative closest point optimisation without scaling or reflection[9].

Separate transformation matrices were estimated for the acetabular and femoral sides to accommodate relative hip motion between examinations. Matrices estimated in local regions of interest were combined with crop ofsets to recover transformations in the global physical coordinates of the original CT. Registration was not accepted on the basis of the ICP root-mean-square distance alone. Anteroposterior and lateral overlays were generated automatically for manual review of key bone surfaces, sampled points, and metal contours. Cases with fractures, subtrochanteric osteotomy, or other osseous changes between the lesser trochanter and stem tip were excluded. Additional fixation hardware was permitted when it did not compromise structural correspondence in this critical region.

## 4.6 Watertight Reconstruction and Dual-Channel TSDF

Metal was identified on postoperative CT using a high threshold of approximately 2,700 HU. Morphological closing and hole filling reduced false internal cavities caused by photon-starvation darkening within the femoral head. CUDA-accelerated DifDMC reconstructed a closed, watertight surface from the metal voxels, and this solid surface defined the zero level set of the TSDF. Distances were truncated at 5 mm, with positive values inside and negative values outside. The acetabular channel contained the cup and femoral head, while the femoral channel contained the stem; the channels were separated at the stem neck.

Unlike a binary mask, a TSDF provides a continuous distance gradient on both sides of the surface while limiting the contribution of remote background voxels to the loss. A mid-coronal section displays the full stem length, and the three-dimensional solid can be recovered directly from the zero level set (Fig. 8).

![](images/a254160d9222c6deed38a1c448f35e567761cc6b958d1aa15b81296edb18b5b7.jpg)  
Figure 7: Independent rigid registration of the pelvis and femur establishes supervision in preoperative space. a,b, Preoperative and postoperative bone surfaces and ICP samples for the pelvic and femoral registrations. Pale yellow and pale green indicate preoperative and postoperative isosurfaces, red points indicate postoperative surface samples selected away from metal artefacts and surgically altered regions, and white lines indicate postoperative prosthesis contours. c, The acetabular and femoral components are returned independently to preoperative CT space using their respective rigid transformations.

![](images/d605945dd3487cab326822dab81456ede49a947fe703c59ac706499f1031a7a0.jpg)

![](images/9c3650f2029c7da7d67a8f89358e7ca7a41c188c0907e98f56971597ea27cde6.jpg)

![](images/c57b9f6c5f9f1a282beff21d2fc836d9ecca7daf7b4aabc0a78f5f936d39d344.jpg)  
Figure 8: TSDF representation of prosthesis geometry. a, Solid femoral stem extracted from postoperative CT on a mid-coronal section. b, Continuous signed distance field on the same section; the black line marks the zero level set and distances are truncated at 5 mm. c, Closed, watertight stem surface recovered from the three-dimensional zero level set.

All images were resampled to 1-mm isotropic voxels, and each spatial dimension was padded to a multiple of 32. Preoperative CT underwent piecewise linear density normalisation. The transition from cancellous bone to cortex, spanning 150–650 HU, was mapped to the full intensity interval from 1 to 0; 650–1,150 HU and 1,150–3,150 HU were mapped to 0–0.5 and 0.5–1, respectively. This mapping enhanced density transitions among the canal, trabecular bone, and cortex. Training volumes covered the intersection of the preoperative scan and registered postoperative extent, with additional margins for the periprosthetic TSDF. Spatial dimensions ranged from $1 2 8 \times 1 2 8 \times 1 9 2$ to $2 5 6 \times 3 5 2 \times 6 4 0$ , with axis-wise medians of $1 6 0 \times 1 6 0 \times 5 1 2$ . Voxel-wise operations including nearestneighbour search, artefact-aware sampling, resampling, and distance-field computation were implemented as just-in-time compiled NVIDIA Warp GPU kernels. This avoided Python-level iteration and repeated CPU–GPU transfers over large volumes and supported a unified, automated GPU pipeline for parameter verification, registration, resampling, and distance-field construction.

## 4.7 Latent Autoencoders and Reconstruction

Separate KL-regularised autoencoders were trained for preoperative bone and prosthesis TSDF volumes[11]. Both used channel levels of 32, 64, and 128, two residual blocks per level, four-fold spatial downsampling, and four latent channels. Training used $1 2 8 ^ { 3 }$ patches, whereas full volumes were encoded and decoded with Gaussian-weighted sliding windows. The encoder and decoder were fully convolutional to avoid inconsistency between patch boundaries and global selfattention.

Reconstruction objectives comprised L1 loss, weighted mean-squared error, and KL regularisation. The bone branch additionally used a PatchGAN least-squares adversarial objective and a three-dimensional MedicalNet perceptual loss[12]. The prosthesis branch used Eikonal-gradient and solid-interior constraints. Models were trained with Adam; generator and discriminator learning rates were $1 0 ^ { - 4 }$ and $2 \times 1 0 ^ { - 4 }$ , respectively, with $\beta = ( 0 . 5 , 0 . 9 )$ . Adversarial loss was disabled for the first five epochs. Each latent space was calibrated using its training-set global mean and scaling factor to approach zero mean and unit scale.

The bone and prosthesis autoencoders were selected at epochs 188 and 97, respectively. On the validation set, the bone branch achieved an L1 error of 0.007946, peak signal-to-noise ratio (PSNR) of 35.85 dB, and structural similarity index (SSIM) of 0.9802. Corresponding values for the prosthesis branch were 0.003313, 47.11 dB, and 0.9964 (Table 3). The prosthesis branch achieved an interpolation Eikonal error of 0.013177 and a reconstruction Eikonal error of 0.007241, supporting the decoding of intermediate latents into continuous distance fields.

Table 3: Best validation reconstruction performance of the bone and prosthesis autoencoders.
<table><tr><td>Branch</td><td>Epoch</td><td>L1</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>Preoperative bone</td><td>188</td><td>0.007946</td><td>35.85</td><td>0.9802</td></tr><tr><td>Prosthesis TSDF</td><td>97</td><td>0.003313</td><td>47.11</td><td>0.9964</td></tr></table>

![](images/47bb6c3909c21aaf5ab52f12a5adfb80df8e0777e3d8f66c9ba3bc4ee4640e4f.jpg)  
Figure 9: Latent reconstruction of the dual-channel prosthesis TSDF. The upper and lower rows show the acetabular and femoral channels, respectively. Columns show the original TSDF, autoencoder reconstruction, and contrast-enhanced absolute error.

## 4.8 Bone-Conditioned Rectified Flow Matching

Let y denote the eight-channel prosthesis TSDF latent, x the four-channel preoperative bone latent, and $\epsilon \sim \mathcal { N } ( 0 , I )$ . Using normalised generative time τ from noise to data, the linear path is

$$
y _ { \tau } = ( 1 - \tau ) \epsilon + \tau y , \qquad \tau \sim \mathcal { U } ( 0 , 1 ) ,\tag{1}
$$

and the model regresses the velocity $y - \epsilon$ :

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { y , x , \tau , \epsilon } \Vert v _ { \theta } ( [ y _ { \tau } \Vert x ] , \tau , c ) - ( y - \epsilon ) \Vert _ { 2 } ^ { 2 } ,\tag{2}
$$

<sup>where</sup> <sup>[</sup>·∥·<sup>]</sup> <sup>denotes</sup> <sup>channel-wise</sup> <sup>concatenation</sup> <sup>and</sup> <sup>c</sup> <sup>is</sup> <sup>the</sup> <sup>optional</sup> <sup>structured</sup> <sup>prosthesis</sup> <sup>condi-</sup> tion. This objective corresponds to single-stage rectified-flow matching[13].

The three-dimensional UNet received 12 channels and predicted eight, with channel levels of 96, 192, and 384 and two residual blocks per level. Self-attention was used only at the deepest level.

The bone latent retained the same spatial dimensions as the prosthesis path state and entered as a channel condition from the first convolution. Structured parameters were represented by six 256-dimensional tokens and injected through cross-attention at the deepest level. The parameters comprised stem model, model-specific size, cup outer diameter, head diameter, head ofset, and liner ofset.

## 4.9 Conditioning, Optimisation, and Inference

Classifier-free condition dropout[14] sampled six modes with equal probability: unconditional, all prosthesis parameters only, bone only, bone plus stem model, bone plus stem model and size, and bone plus all parameters. No classifier-free guidance amplification was applied during training. The generator was trained with AdamW at a learning rate of 10−<sup>4</sup> using automatic mixed precision and a physical batch size of one. Gradients were accumulated dynamically according to voxel count to achieve an efective batch size of approximately 48. Exponential moving averages (EMA; decay 0.999) were maintained for both the generator and parameter encoder. At inference, Gaussian noise matching the target latent shape was integrated to the data endpoint with a small number of Euler steps and decoded by the prosthesis autoencoder using sliding windows.

The generator was trained for 1,000 epochs. Mean training velocity-field MSE decreased from 1.1524 to 0.0187, and the median validation MSE for bone-only conditioning over the final 100 epochs was 0.02099. EMA weights from epoch 999 were used for final inference. The large efective batch mitigated gradient variance arising from a physical batch size of one and variable volume dimensions, while EMA reduced sampling perturbations caused by individual parameter updates during prolonged optimisation.

## 4.10 Computing Environment

Data processing, model training, inference, and visualisation were performed in a Linux container. The compute node contained two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs (97,887 MiB each), two Intel Xeon Platinum 8476C processors (96 physical and 192 logical cores in total), and 251 GiB of system memory. Each reported model-training or inference run used one GPU. The software environment comprised Python 3.13.13, PyTorch 2.11.0, CUDA 13.0, MONAI 1.5.2, ITK 5.4.6, and NVIDIA Warp 1.12.1 with GPU driver 580.173.02. NVIDIA Warp kernels supported registration sampling, distance computation, and resampling, while DifDMC provided CUDA-accelerated watertight isosurface reconstruction.

## 4.11 Evaluation of Generated Geometry

All final results were generated with EMA weights. Primary visualisations were coronal anteroposterior projections and consecutive sections in preoperative CT space, with CT in greyscale and acetabular and femoral components in green and blue. Cross-design evaluation included validation cases with Corail, SUMMIT, Accolade TMZF, Tri-Lock, Secur-Fit, Wagner Cone, and M/L Taper stems. Each case underwent bone-only generation and generation conditioned jointly on bone and the recorded stem model. The two modes used identical initial noise and received neither stem size nor other component parameters, isolating the efect of stem-model conditioning.

Bone–prosthesis interface evaluation compared corresponding axial levels over the longitudinal extent shared by the actual and generated prostheses. Coronal projections were reviewed concurrently for stem alignment, fit and fill within the efective canal, respect for the calcar boundary, and the relationship between cup and acetabular wall. Conditional distribution evaluation repeated bone-only generation while varying the initial noise and used the registered actual prosthesis as a clinical reference. Portions of the generated projection extending beyond the actual projection were overlaid as diferently coloured contours, while overlapping regions were not redrawn, thereby depicting the within-patient spatial distribution. L1, PSNR, SSIM, Eikonal error, and velocity-field MSE were used to verify latent reconstruction and training convergence, not as stand-alone measures of clinical planning quality.

## Data Availability

The raw and processed data cannot be made publicly available because of patient privacy, ethics requirements, and institutional data-governance restrictions.

## Code Availability

Source code is available at https://github.com/doidio/nonahip.

## Author Contributions

Y.W. conceived the engineering architecture, developed the software and data-processing pipeline, trained and validated the models, analysed the results, prepared the visualisations, and drafted the manuscript. J.L. performed clinical case inclusion and exclusion, verified surgical and billing records, and reviewed the manuscript. J.S. performed data annotation, data curation, and quality control, and reviewed the manuscript. L.W. defined the clinical use scenarios, conducted expert review and clinical interpretation, supervised the clinical aspects of the study, and reviewed the manuscript. All authors approved the final manuscript.

## Competing Interests

Y.W. and J.S. are employees of Changzhou Jinse Medical Information Technology Co., Ltd. J.L. and L.W. declare no competing interests.

## References

[1] Colombi, A., Schena, D. & Castelli, C. C. Total hip arthroplasty planning. EFORT Open Rev. 4, 626–632 (2019). doi:10.1302/2058-5241.4.180075.

[2] Chen, X. et al. Development and validation of an artificial intelligence preoperative planning system for total hip arthroplasty. Front. Med. 9, 841202 (2022). doi:10.3389/fmed.2022.841202.

[3] Rouzrokh, P. et al. THA-Net: a deep learning solution for next-generation templating and patientspecific surgical execution. J. Arthroplasty 39, 727–733.e4 (2024). doi:10.1016/j.arth.2023.08.063.

[4] Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. High-resolution image synthesis with latent difusion models. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition 10684–10695 (2022). doi:10.1109/CVPR52688.2022.01042.

[5] Wang, H. et al. 3D MedDifusion: a 3D medical latent difusion model for controllable and high-quality medical image generation. IEEE Trans. Med. Imaging 44, 4960–4972 (2025). doi:10.1109/TMI.2025.3585372.

[6] Guo, P. et al. MAISI: medical AI for synthetic imaging. In Proc. IEEE/CVF Winter Conf. Applications of Computer Vision 4430–4441 (2025). doi:10.1109/WACV61041.2025.00435.

[7] Zhao, C. et al. MAISI-v2: accelerated 3D high-resolution medical image synthesis with rectified flow and region-specific contrastive loss. Proc. AAAI Conf. Artif. Intell. 40, 13088–13098 (2026). doi:10.1609/aaai.v40i15.38309.

[8] Wasserthal, J. et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiol. Artif. Intell. 5, e230024 (2023). doi:10.1148/ryai.230024.

[9] Besl, P. J. & McKay, N. D. A method for registration of 3-D shapes. IEEE Trans. Pattern Anal. Mach. Intell. 14, 239–256 (1992). doi:10.1109/34.121791.

[10] Wang, Z. & Dai, K. Geometric morphology of the femoral calcar and efective cavity of the proximal femur. Chin. J. Orthop. 14, 436–440 (1994) (in Chinese).

[11] Kingma, D. P. & Welling, M. Auto-encoding variational Bayes. In Proc. International Conference on Learning Representations (2014).

[12] Chen, S., Ma, K. & Zheng, Y. Med3D: transfer learning for 3D medical image analysis. Preprint at https://arxiv.org/abs/1904.00625 (2019).

[13] Liu, X., Gong, C. & Liu, Q. Flow straight and fast: learning to generate and transfer data with rectified flow. In Proc. International Conference on Learning Representations (2023).

[14] Ho, J. & Salimans, T. Classifier-free difusion guidance. Preprint at https://arxiv.org/abs/2207. 12598 (2022).

[15] Macklin, M. Warp: a high-performance Python framework for GPU simulation and graphics. NVIDIA GPU Technology Conference (GTC) (2022). https://github.com/NVIDIA/warp.