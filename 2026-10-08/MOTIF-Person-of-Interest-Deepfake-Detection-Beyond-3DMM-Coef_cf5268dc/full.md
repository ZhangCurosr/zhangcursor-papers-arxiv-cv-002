# MOTIF: Person-of-Interest Deepfake Detection Beyond 3DMM Coefficients

Giovanni Affatato, Sara Mandelli, Paolo Bestagini, Stefano Tubaro

Dipartimento di Elettronica, Informazione e Bioingegneria, Politecnico di Milano, 20133 Milan, Italy. Corresponding Author: giovanni.affatato@polimi.it

Abstract—Video deepfakes targeting a specific individual, the Person-of-Interest (POI), are the most harmful ones, and, since a public figure is abundantly recorded, a detector can be built from genuine footage of that individual. Such detectors commonly describe a subject through a 3D Morphable Model (3DMM) and adopt its coefficients as a whole, so which part of that description carries the signal has never been measured. We dissect it, holding the encoder, the training corpus and the enrollment protocol fixed and varying only what the encoder observes. The groups of coefficients prove largely redundant, since the shape block alone recovers almost all the accuracy of the full vector, and their temporal evolution contributes a real but bounded amount. We further show that the dense surface the same fit returns, which these detectors discard, carries identity information that the coefficients do not, and that it helps precisely where they are weakest. We assemble the best configuration into MOTIF, a visual-only detector trained on real videos only, with no manipulated video and no POI-specific data. It improves on both state-of-the-art POI detectors in every dataset and manipulation of our benchmark and at two quality levels. Our experimental code will be released at polimi-ispl/MOTIF.

Index Terms—person-of-interest deepfake detection, 3D morphable models, facial geometry, contrastive learning.

## I. INTRODUCTION

Among video deepfakes, the most harmful are those that target one specific individual, referred to as the Person-of-Interest (POI), such as a politician, a celebrity or a corporate executive, since such forgeries enable disinformation and fraud. Generative AI keeps making such forgeries easier to produce and harder to recognize [1]. However, a POI is a public figure, so genuine recordings of that subject are abundant, and a detector can be built around them. Such detectors describe how the subject looks [2] or how the subject moves while speaking [3].

Some of these detectors are trained on the specific POI they protect, which requires a large amount of footage of that subject and yields a model that does not transfer to anyone else. We address the complementary approach, which learns a general representation of identity from many subjects and enrolls the POI only at inference, from a set of pristine reference videos, so that neither footage of the POI nor any manipulated video is required during training [4]–[7].

Most detectors of this kind describe a subject through a 3D Morphable Model (3DMM) [4], [6], [8], a statistical model of face geometry that represents a face by a short vector of coefficients weighting a basis of aligned three-dimensional face scans [9]. Separate groups of coefficients account for the facial shape, which belongs to the subject and is stable across recordings, for the expression and for the rigid head pose, and the same coefficients decode back into a dense surface of vertices covering the face. The appeal is twofold: the description is compact, and it keeps no pixels and hence neither the appearance of the subject nor the traces left by a generator.

These works, however, adopt the representation wholesale: a fixed vector of shape, expression and pose coefficients is fed to a temporal encoder, and the resulting system is evaluated end to end. Their ablations vary the encoder and the objective, but not the description. ID-Reveal compares three metric losses and measures the effect of its adversarial branch, which it introduces so that the encoder depends on behavior rather than on visual information alone [4], while Petmezas et al. vary the number of recurrent and attention layers and credit the result with a better modeling of the temporal dynamics of facial expressions [6]. In both cases the claim concerns the representation, while the experiments concern the model built upon it. We therefore examine the description itself, leaving the encoder aside, and explore three questions whose answers would inform anyone building on a 3DMM: (i) which groups of coefficients provide an effective signal for detection, and whether they are complementary or redundant; (ii) whether the temporal evolution of the coefficients adds anything over a time-invariant summary; and (iii) whether the dense surface the model reconstructs, discarded once the coefficients are extracted, carries identity information that the coefficients do not.

To answer them we build MOTIF (MOrphable model for Temporal Identity Forensics), a visual-only POI deepfake detector that serves first as a testbed and then, in its best configuration, as our proposed method. Its encoder is a temporal transformer trained by an identity-contrastive objective [10] on real videos of many subjects, with no manipulated video and no footage of the target POI, as sketched in Fig. 1. We retrain it from scratch for every representation we compare, holding the corpus, the encoder and the enrollment protocol fixed, so that a difference in performance is attributable to the representation alone. We vary the groups of coefficients and their temporal treatment in a factorial design, pairing each sequence with a time-invariant summary, so that the temporal evolution is isolated from the capacity of the input. We also decode the coefficients back into geometry [11] and describe the vertices of the mouth as a separate branch, evaluated alone and fused with the global one. Our contributions are:

![](images/b85d5e9f8d8f45877bc2f16d7e2ee208081efab587be6196d06ac9edd087e8a8.jpg)  
Fig. 1: MOTIF training overview.

• We present a controlled study of the 3DMM description for reference-based POI deepfake detection.

• We introduce a use of the representation that coefficientbased detectors leave unexploited, namely the signal carried by the 3D reconstruction of the mouth.

• We assemble the best configuration into MOTIF and compare it with six state-of-the-art detectors on four datasets spanning face swap, reenactment and lip-sync, at two levels of video quality.

## II. PROPOSED METHODOLOGY

## A. POI Deepfake Detection

Each target subject, the POI, is characterized by a collection of pristine reference videos $\mathcal { D } _ { \mathrm { { r e f } } } ~ = ~ \{ V _ { 1 } , . . . , V _ { N _ { \mathrm { { r e f } } } } \}$ that genuinely depict the individual. Given a test video V claimed to portray the same POI, our goal is to determine whether V is authentic or manipulated, assigning a label $y \in \{ \mathrm { r e a l } , \mathrm { f a k e } \}$ on the basis of ${ \mathcal { D } } _ { \mathrm { r e f } }$ . In line with Cozzolino et al. [4], [5], we frame detection as the thresholding of a subject-similarity score. Let E denote an encoder mapping a video to a set of subject descriptors. A test video is scored according to how well its descriptors E(V) align with those extracted from the reference set via a similarity function $S ( V , \mathcal { D } _ { \mathrm { r e f } } ) \in \mathbb { R }$

We study this problem under two constraints. First, we operate in an open-set regime, meaning that the forgery techniques employed to generate deepfakes are not known at training time. Second, we require detection to be POIagnostic: we do not use POI-specific data at training time, so that the encoder is not tailored to any particular identity.

## B. Extraction of Facial Representations

1) Global descriptor extraction: Let a video be a sequence of T frames. We process the t-th frame with a 3DMM face reconstruction model [11], which regresses compact sets of coefficients describing the observed face: the shape coefficients ${ \pmb { \alpha } } _ { t } ^ { \mathrm { i d } } \in \mathbb { R } ^ { D _ { \mathrm { i c } } }$ , which encode the neutral facial geometry of the subject, the expression coefficients $\pmb { \alpha } _ { t } ^ { \mathrm { e x p } } \in \mathbb { R } ^ { D _ { \mathrm { e x p } } }$ which encode the facial gesture being performed, the head pose ${ \boldsymbol \rho } _ { t } \in \mathbb { R } ^ { 3 }$ and the translation $ { \mathbf { d } } _ { t } \in \mathbb { R } ^ { 3 }$ , which describe the rigid motion of the head with respect to the camera. We concatenate the four retained coefficient sets into a single perframe vector

$$
\mathbf { c } _ { t } = \left[ { \alpha } _ { t } ^ { \mathrm { i d } } ; { \alpha } _ { t } ^ { \mathrm { e x p } } ; { \rho } _ { t } ; { \bf d } _ { t } \right] \in \mathbb { R } ^ { D _ { \mathrm { c } } } ,\tag{1}
$$

with $D _ { \mathrm { c } } = D _ { \mathrm { i d } } + D _ { \mathrm { e x p } } + 3 + 3$ , which we normalize using the mean and standard deviation estimated on real training videos.

Both in training and in test, we process a video V in windows of L consecutive frames with 50% overlap. The global descriptor, which observes the whole face, is the window starting at the t-th frame,

$$
\mathbf { X } _ { t } ^ { \mathrm { g l o b a l } } = \left[ \mathbf { c } _ { t } , \ldots , \mathbf { c } _ { t + L - 1 } \right] ^ { \top } \in \mathbb { R } ^ { L \times D _ { \mathrm { c } } } .\tag{2}
$$

2) Mouth descriptor extraction: The coefficients summarize the whole face into a single vector, whereas the same fit also returns a dense surface, whose vertices place the description on specific parts of the face. We therefore extract a second descriptor from the lip vertices alone.

Using 3DMMs, we reconstruct the 3D geometry of the face from each frame and retain only the lip vertices. We refer the reader to [11] for the details of the reconstruction. Let M be the index set of the lip vertices of the face model extracted through the 3DMM and let $D _ { \mathrm { v } }$ , the number of vertex coordinates, be equal to 3|M|. We then subtract the temporal mean of the lip vertices over the analysis window, which removes the subject’s neutral lip shape and retains only how the lips move, independently of what they look like at rest. We define the mouth descriptor as X<sup>mouth</sup> $\in \mathbb { R } ^ { L \times D _ { \mathrm { v } } }$

In general, we denote by $\mathcal { W } _ { b } ( V )$ the set of the window descriptors of branch b extracted from a video $V ,$ , so that $\mathcal { W } _ { \mathrm { g l o b a l } } ( V )$ collects the global face descriptors, while ${ \mathcal { W } } _ { \mathrm { m o u t h } } ( V )$ the mouth ones. We denote by $L \times D _ { b }$ the input dimension of branch b.

## C. Encoder and Training Objective

1) Temporal transformer: Both branches (global and mouth) employ the same architecture, which we define as the map $E _ { b } : \mathbb { R } ^ { L \times D _ { b } } \  \ \mathbb { S } ^ { C - 1 }$ . In practice, a linear layer tokenizes each frame into a C-dimensional token, a learnable classification (CLS) token is prepended, learnable positional embeddings are added and a stack of transformer blocks processes the sequence. We define $E _ { b } ( \mathbf { X } )$ as the $\ell _ { 2 } \cdot$ -normalized CLS embedding of the last block. Notice that attention operates over time only, so that the model can relate distant frames of the same window. The two instances of $E _ { b }$ are trained separately and share no weights, and they differ only in the input dimension $D _ { b }$ of the tokenizer.

2) Identity-contrastive objective: We train each encoder with a single identity-contrastive objective. Let B be the set of window indices in a batch and let $\ell ( a )$ be the identity label of window a. We define the set of positive samples of an anchor $a \in B$ as all the other windows of the same subject, i.e., ${ \mathcal { P } } ( a ) = \{ j \ \in \ B \ \backslash \ \{ a \} \ : \ \ell ( j ) = \ \ell ( a ) \ \}$ , while every window of a different subject is a negative sample. Let K denote the set of selected encoder depths, and let $\mathbf { z } _ { a } ^ { ( k ) } \in \mathbb { S } ^ { C - 1 }$ be the CLS embedding of window a read at layer k. We adopt the Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss [10], which we write for an anchor a and a depth k as

$$
\mathcal { L } ^ { ( k , a ) } = - \frac { 1 } { \vert \mathcal { P } ( a ) \vert } \sum _ { j \in \mathcal { P } ( a ) } \log \frac { \exp \bigl ( \langle \mathbf { z } _ { a } ^ { ( k ) } , \mathbf { z } _ { j } ^ { ( k ) } \rangle / \eta \bigr ) } { \sum _ { m \in \mathcal { B } \backslash \{ a \} } \exp \bigl ( \langle \mathbf { z } _ { a } ^ { ( k ) } , \mathbf { z } _ { m } ^ { ( k ) } \rangle / \eta \bigr ) }\tag{3}
$$

here $\eta$ is the temperature parameter and $\langle \cdot , \cdot \rangle$ denotes the inner product. The total objective aggregates over all the anchors and the selected depths with equal weight. In doing so, we make the intermediate blocks, which present features with an increasing order of abstraction, identity-discriminative as well.

We populate ${ \mathcal { P } } ( a )$ with windows drawn from different videos of the same subject, so that most positive pairs are cross-video and the objective cannot be satisfied by recordingspecific cues. Moreover, we populate B by balancing the identities, so that no subject dominates the contrastive comparison. We also use a Multi-Similarity Miner [12], which keeps only the most informative comparisons.

Furthermore, the contrastive pass operates on a temporally masked window in which contiguous spans of frames are removed, so that the two views of a positive pair are not trivially matchable. Masking is applied at training time only.

## D. Reference-Set Matching and Score Fusion

1) Reference-set matching: To enroll the POI, we compute for each branch b a set of reference embeddings $\mathcal { R } _ { b } .$ , one for every reference window,

$$
\mathscr { R } _ { b } = \left\{ \ E _ { b } ( \mathbf { X } ) \ : \ \mathbf { X } \in \mathscr { W } _ { b } ( V _ { i } ) , \ V _ { i } \in \mathscr { D } _ { \mathrm { r e f } } \ \right\} .\tag{4}
$$

A query video V yields in the same way $\mathcal { Q } _ { b } ( V ) = \{ E _ { b } ( \mathbf { X } )$ $\mathbf { X } \in \mathcal { W } _ { b } ( V ) \big \}$

The reference embeddings of a subject are usually not isotropic: they spread along the directions in which that subject varies across genuine videos, such as recording conditions and spoken content, and stay narrow along the directions in which that subject is stable. A plain cosine weights every direction equally, so it rewards agreement inside a spread that any video of the subject already covers, and it dilutes a disagreement along the narrow directions, which is where identity actually lives. We therefore rescale each direction by the inverse of its reference standard deviation, so that a query earns similarity only where the subject is consistent.

We define the covariance of $\mathcal { R } _ { b }$ to be $\Sigma _ { b }$ . We define the per-POI whitening map as

$$
\omega _ { b } ( { \bf z } ) = \frac { \boldsymbol { \Sigma } _ { b } ^ { - 1 / 2 } \mathbf { z } } { \left\| \boldsymbol { \Sigma } _ { b } ^ { - 1 / 2 } \mathbf { z } \right\| _ { 2 } } ,\tag{5}
$$

where $\lVert \cdot \rVert _ { 2 }$ denotes the Euclidean norm. We score the query as the maximum cosine similarity in the whitened space,

$$
S _ { b } ( V ; \mathcal { R } _ { b } ) = \operatorname* { m a x } _ { \mathbf { r } \in \mathcal { R } _ { b } } \operatorname* { m a x } _ { \mathbf { q } \in \mathcal { Q } _ { b } ( V ) } \big \langle \omega _ { b } ( \mathbf { r } ) , \omega _ { b } ( \mathbf { q } ) \big \rangle ,\tag{6}
$$

where r and q denote a reference and a query embedding.

2) Score fusion: The two scores of (6) live on per-subject and per-branch scales, so before combining them we calibrate them on the reference set. Specifically, we score each reference video in a leave-one-out setting, i.e., every i-th reference video is tested against the set $\mathcal { R } _ { b } ^ { ( - i ) }$ of the reference embeddings of $\mathscr { D } _ { \mathrm { r e f } } \backslash \{ V _ { i } \}$ . We denote by $\mu _ { b }$ and $\sigma _ { b }$ the mean and the standard deviation of the scores $\vdots S _ { b } ( V _ { i } ; \mathcal { R } _ { b } ^ { ( - i ) } ) \} _ { i = 1 } ^ { N _ { \mathrm { r e f } } }$ so obtained. A generic query video (not part of the reference set) is then expressed in units of the genuine variability of that subject,

$$
z _ { b } ( V ) = \frac { S _ { b } ( V ; \mathcal { R } _ { b } ) - \mu _ { b } } { \sigma _ { b } } .\tag{7}
$$

This operation places the two branches on a common scale, so that they can be combined with a straightforward arithmetic mean, where each of them can raise or lower the verdict and a query is authentic only if it is consistent with the reference set both in the description of the whole face and in that of the mouth.

## III. EXPERIMENTAL SETUP

## A. Datasets

1) Training data: We train on real videos of Vox-Celeb2 [13], an in-the-wild audio-visual corpus of interview footage. It comprises 5,994 identities, 150,480 videos and 1.13M clips. The pristine videos of FakeAVCeleb are themselves VoxCeleb2 clips, so we remove from the training list the 500 identities used to build that dataset [14]. In this way, none of the identities we train on appears among the evaluation POIs.

2) Evaluation data: We evaluate on four public deepfake datasets, namely DF-TIMIT [15], FakeAVCeleb [14], DeepSpeak [16] and KoDF [17]. Together, they span all the main manipulation families that a detector has to face, namely identity swapping, full-face reenactment or synthesis, identitypreserving lip-sync, and their combinations.

We group the manipulations in Table I according to what they do to the two quantities that a POI detector can observe, i.e., facial appearance and facial motion: (i) Face-swap (FS): the face of the target is replaced by that of a source identity, so appearance is altered globally while the driving performance remains that of the target; (ii) Reenactment (RE): identity and appearance are preserved, or synthesized as an avatar, but the motion is imported from a driving actor, i.e., the complement of the previous case; (iii) Lip-sync (LS): identity, appearance and head motion are all preserved, and only the mouth is resynthesized to match a target audio track; (iv) Faceswap + lip-sync (FS+LS): a face swap that has then been lipsynced, so both quantities are altered at once.

## B. Evaluation Metrics

We assess performance through the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve, which measures the quality of the score ranking regardless of the operating threshold, and the Balanced Accuracy (BA), which we take as the maximum of the balanced accuracy over the decision threshold. We compute both per POI, i.e., we score the real videos of a subject against its fakes, and we average the result over the subjects of a dataset. This is consistent with the POI scenario, in which the system targets a specific subject and a small set of subject-specific samples is available to calibrate the threshold, so that the reported values measure the discriminative capability of the method after adaptation to the target POI. When we report a single manipulation category, we restrict the fakes of each subject to that category, we leave its real videos unchanged and we exclude the subjects that have no fake of it. The aggregate values instead use all the fakes of a subject, regardless of the manipulation.

TABLE I: Generation methods of each evaluation dataset, grouped into the four canonical categories used in the per-category results.
<table><tr><td>Category</td><td>Dataset</td><td>Generation method</td></tr><tr><td>Face-swap (FS)</td><td>DF-TIMIT FakeAVCeleb DeepSpeak K₀DF</td><td>autoencoder-based swap FaceSwap, FSGAN FaceFusion, INSwapper, SimSwap FaceSwap, DeepFaceLab, FSGAN</td></tr><tr><td>Face-  $s w a p + l i p $  sync (FS+LS)</td><td>FakeAVCeleb</td><td>FaceSwap and FSGAN followed by Wav2Lip [18]</td></tr><tr><td>Reenactment (RE)</td><td>DeepSpeak K₀DF</td><td>LivePortrait, HelloMeme, Memo FOMM</td></tr><tr><td>Lip-sync (LS)</td><td>FakeAVCeleb DeepSpeak K₀DF</td><td>Wav2Lip [18] Diff2Lip, LatentSync Wav2Lip [18]</td></tr></table>

## C. State-of-the-Art Baselines

We compare MOTIF against publicly available general and POI-specific deepfake detectors. Among general deepfake detectors, we consider RealForensics [19], LipForensics [20] and FTCN [21]. We select them because they release pretrained weights that we can evaluate directly under our cross-dataset protocol, their common training corpus being FaceForensics++ [22], which shares no material with any dataset we test on. We further include the model of Seferbekov [23], which won the DFDC challenge and was trained on the corresponding dataset.

Regarding POI-specific detectors, we select two recently proposed and publicly available methods, namely POI-Forensics [5] and ID-Reveal [4]. Both are trained on Vox-Celeb2 [13], i.e., the corpus we train on as well, which makes them the closest available comparison to MOTIF. POI-Forensics is a multimodal framework that analyzes audio and video jointly: we evaluate it in its video-only setting so that the comparison matches our visual-only setup. ID-Reveal is visual-only by design and is the prior method closest to ours, since it also describes a subject through 3DMM coefficients.

## D. Architecture and Training Details

1) Input: We extract the per-frame coefficients with the   
3DDFA-V3 model [11], of which we keep $D _ { \mathrm { i d } } = 8 0$ shape

coefficients, $D _ { \mathrm { e x p } } = 6 4$ expression coefficients, 3 head-pose angles and 3 translation components, for a total of $D _ { \mathrm { c } } = 1 5 0$ dimensions per frame. We process each video in windows of $L = 7 5$ frames, i.e., 3 sec at 25 fps, with 50% overlap. The mouth branch reconstructs the $| { \mathcal { M } } | = 9 6 8$ lip vertices of the face model, i.e., $D _ { \mathrm { v } } = 2 { , } 9 0 4$ dimensions per frame, from which we remove the per-window temporal mean before applying the same per-dimension scaling.

2) Encoders: Both branches employ the same temporal transformer, with 12 blocks, hidden size $C \ = \ 1 9 2$ and 3 attention heads, preceded by a linear per-frame tokenizer and by learnable positional embeddings.

3) Training: We train with the NT-Xent objective at temperature $\eta ~ = ~ 0 . 1$ , applied with equal weight to the CLS embedding read at the depths $\mathcal { K } = \{ 3 , 6 , 9 , 1 2 \}$ , and we mine the pairs with a Multi-Similarity Miner $( \epsilon = 0 . 1 )$ . Batches are drawn by the identity-balanced sampler as 8 subjects $\times ~ 4$ videos × 2 windows, and the training-time view of a window is masked by contiguous spans of 15 frames covering half of it. We optimize with AdamW, at learning rate $1 0 ^ { - 4 }$ and weight decay 0.1, under a cosine schedule with 10 warmup epochs. We select the deployed checkpoint as the epoch of lowest contrastive loss on the held-out VoxCeleb2 test identities, which comprise 118 identities and 942 videos and are disjoint from the training ones by the split of the dataset.

## IV. RESULTS

## A. Dissecting the Coefficient Vector

Prior work adopts the 3DMM coefficients as a whole, so we first ask which of their groups provides an effective signal for detection, and whether the groups are complementary or redundant. We then ask whether the evolution of the coefficients over time carries information that a time-invariant description of the same subject does not.

We design the experiment as a factorial ablation over the coefficient vector of (1), which we split into a shape block ${ \bf s } _ { t } ~ = ~ \alpha _ { t } ^ { \mathrm { i d } }$ and an expression block ${ \bf e } _ { t } ~ = ~ [ \alpha _ { t } ^ { \mathrm { e x p } } ; \rho _ { t } ; { \bf d } _ { t } ]$ which collects the expression, the pose and the translation. We train the encoder with $\mathbf { s } _ { t } ,$ with $\mathbf { e } _ { t }$ or with $\mathbf { c } _ { t } .$ , and we read each of them in a dynamic variant, which keeps the perframe trajectory over the window, and in a static variant, which repeats the average of the block over the clip, obtaining the six configurations of Table II, from which we omit DF-TIMIT because every configuration achieves 100% on it.

The three inputs turn out to be largely interchangeable. The six configurations span less than two points of overall mean AUC, and the three dynamic ones lie within 0.9 points of each other. Reading the shape block alone attains 90.4% against the 90.6% of the full vector, so the expression, the pose and the translation are worth 0.2 points once the shape is read over time, whereas discarding the shape block costs 0.9.

Reading the coefficients dynamically is the better choice for every block, and it is worth 1.0 points of overall mean AUC on the shape block, 0.9 on the expression block and 1.6 on their combination. The gain is largest on reenactment, where the motion of a driving actor is precisely the manipulated quantity.

TABLE II: AUC / BA (%) per dataset and manipulation category, considering static (S) coefficients or dynamic (D) ones. Best result per column in bold.
<table><tr><td></td><td colspan="4">FakeAVCeleb</td><td colspan="4">DeepSpeak</td><td colspan="4">KoDF</td><td>Overall</td></tr><tr><td>Input</td><td>FS</td><td>FS+LS</td><td>LS</td><td>All</td><td>FS</td><td>RE</td><td>LS</td><td>All</td><td>FS</td><td>RE</td><td>LS</td><td>All</td><td>Mean</td></tr><tr><td>Shape (S)</td><td>91.6/92.2</td><td>93.1/93.5</td><td>60.7/78.3</td><td>85.3/87.0</td><td>98.5/98.4</td><td>87.6/90.9</td><td>86.7/91.5</td><td>89.7/91.3</td><td>97.6/97.4</td><td>85.7/86.8</td><td>93.0/93.5</td><td>93.1/92.0</td><td>89.4/90.1</td></tr><tr><td>Expression (S)</td><td>89.2/90.4</td><td>91.1/91.7</td><td>57.0/75.7</td><td>82.9/85.1</td><td>98.3/98.5</td><td>88.2/91.4</td><td>87.2/91.3</td><td>90.0/91.2</td><td>97.8/97.7</td><td>85.8/87.0</td><td>94.0/94.2</td><td>93.5/92.3</td><td>88.8/89.6</td></tr><tr><td>Shape + expr. (S)</td><td>90.5/91.6</td><td>91.8/92.9</td><td>59.0/78.3</td><td>83.9/86.3</td><td>98.3/98.3</td><td>87.9/90.8</td><td>86.3/91.6</td><td>89.6/91.1</td><td>97.6/97.6</td><td>86.6/87.5</td><td>96.2/95.5</td><td>93.7/92.6</td><td>89.0/90.0</td></tr><tr><td>Shape (D)</td><td>92.8/93.4</td><td>93.3/94.1</td><td>59.2/77.7</td><td>85.6/87.6</td><td>99.7/99.6</td><td>90.7/92.6</td><td>89.3/92.5</td><td>92.2/92.5</td><td>96.9/96.8</td><td>86.5/87.4</td><td>95.5/95.4</td><td>93.4/92.4</td><td>90.4/90.8</td></tr><tr><td>Expression (D)</td><td>90.2/91.0</td><td>91.7/92.8</td><td>57.9/77.0</td><td>83.8/85.6</td><td>99.6/99.7</td><td>90.2/92.8</td><td>88.9/92.6</td><td>91.8/92.8</td><td>97.5/97.3</td><td>86.4/87.3</td><td>95.5/95.9</td><td>93.6/92.5</td><td>89.7/90.3</td></tr><tr><td>Shape + expr. (D)</td><td>92.6/93.7</td><td>95.0/95.7</td><td>58.3/77.1</td><td>86.2/87.9</td><td>99.5/99.6</td><td>90.5/92.9</td><td>88.0/91.9</td><td>91.6/92.3</td><td>97.5/97.4</td><td>86.8/86.7</td><td>96.4/95.8</td><td>94.0/92.6</td><td>90.6/90.9</td></tr></table>

TABLE III: AUC / BA (%) per dataset and manipulation category, considering the global branch only (G-only), the mouth branch only (M-only) or their fusion. Best result per column is highlighted in bold.
<table><tr><td></td><td>DF-TIMIT</td><td colspan="4">FakeAVCeleb</td><td colspan="4">DeepSpeak</td><td colspan="4">KoDF</td><td>Overall</td></tr><tr><td>Config</td><td>FS</td><td>FS</td><td>FS+LS</td><td>LS</td><td>All</td><td>FS</td><td>RE</td><td>LS</td><td>All</td><td>FS</td><td>RE</td><td>LS</td><td>All</td><td>Mean</td></tr><tr><td>G-only</td><td>100.0/100.0</td><td>92.6/93.7</td><td>95.0/95.7</td><td>58.3/77.1</td><td>86.2/87.9</td><td>99.5/99.6</td><td>90.5/92.9</td><td>88.0/91.9</td><td>91.6/92.3</td><td>97.5/97.4</td><td>86.8/86.7</td><td>96.4/95.8</td><td>94.0/92.6</td><td>92.9/93.2</td></tr><tr><td>M-only</td><td>92.1/92.8</td><td>76.9/82.2</td><td>82.4/85.6</td><td>68.2/79.0</td><td>77.6/81.5</td><td>88.3/91.5</td><td>81.0/84.9</td><td>84.3/88.3</td><td>84.0/86.3</td><td>85.0/86.5</td><td>81.7/84.6</td><td>90.8/91.4</td><td>85.2/85.6</td><td>84.7/86.6</td></tr><tr><td>Fusion</td><td>100.0/100.0</td><td>91.0/92.3</td><td>93.3/94.0</td><td>67.4/79.7</td><td>86.8/87.7</td><td>99.1/99.3</td><td>88.5/91.6</td><td>87.5/91.8</td><td>90.5/91.6</td><td>97.1/97.1</td><td>90.4/90.6</td><td>98.0/97.4</td><td>95.3/94.0</td><td>93.1/93.3</td></tr></table>

Both results have a common explanation in the front end. The shape of a face does not change while a subject speaks, so its coefficients should in principle be constant in time and their clip average should be the best available estimate of that constant, yet reading the block over time is worth a point, so its variation is not estimation noise. Since we regress the coefficients independently on each frame, nothing keeps the shape coefficients of a clip consistent, and since the shape and the expression bases deform the same mesh, a single frame does not separate them. We therefore conjecture that the shape block absorbs a share of the expression dynamics during the per-frame fit, which would explain both why it gains from being read over time and why the two blocks stay so close to each other.

Along both axes, then, the description is less differentiated than its common use would suggest: the groups of coefficients largely duplicate each other, and their temporal evolution contributes real but bounded information. Most of the accuracy of the full vector is already available from the shape block alone. We nonetheless adopt the full coefficient vector c<sub>t</sub>, read dynamically, for the remainder of this article, since it is the best configuration on average.

## B. Exploiting the Reconstructed Surface

The coefficients are not the only description a 3DMM provides, since the same fit also returns a dense surface, so we ask whether that surface carries identity information that the coefficients do not. Table III reports the global and the mouth branch read alone and fused by the mean of their per-POI calibrated scores.

Read alone, the mouth branch attains an overall mean AUC of 84.7% against the 92.9% of the global branch, so a few hundred vertices of the reconstructed lips already identify a subject almost as well as the whole coefficient vector. What matters for our question, however, is not how far the surface goes on its own but whether it adds anything to the coefficients, and it does: fusing the two branches improves on the global one by 9.1 points on the lip-sync manipulation of FakeAVCeleb, by far the weakest column of the table, and by 3.6 points on the reenactment of KoDF.

These two columns have little in common as manipulations, which already suggests that what the surface supplies does not depend on the manipulation being confined to the mouth. The lip-sync columns of KoDF and FakeAVCeleb make the point directly, since they come from the same generator (Table I) and yet the global branch attains 96.4% on the first and 58.3% on the second, so that the fusion adds 1.6 points instead of 9.1: the surface adds most where the coefficients achieve least.

Across the remaining columns the fusion never loses more than 2.0 points, and it attains the best overall mean of the three configurations at 93.1%, so the information it contributes comes at a bounded cost. We adopt the fusion for the remainder of this article.

## C. Comparison with the State of the Art

Table IV positions MOTIF against the two families of baselines on all four datasets and at two quality levels, where we refer to every original dataset as High Quality (HQ) and we define its Low Quality (LQ) version as an H.264 recompression at CRF 40, which emulates the lossy re-encoding applied by video-sharing platforms.

In the HQ block our method is competitive with the general deepfake detectors, since LipForensics [20] attains the highest mean AUC and MOTIF trails it by 4.4 points, while among the POI-specific methods it leads by 10.7 points over ID-Reveal and by 15.1 over POI-Forensics. This comparison is not symmetric in terms of supervision, since the general detectors are trained on manipulated videos, whereas our method never observes one.

The picture changes under compression, where MOTIF becomes the best method overall, ahead of the second best by 9.2 points. It gives up 4.6 points of mean AUC between the two blocks, against an average of 24.6 for the general detectors, which is consistent with what the two families measure: the latter rely on low-level synthesis traces that a recompression at CRF 40 largely destroys, whereas the facial description survives it. The POI-specific baselines are comparably stable, losing 1.1 and 3.1 points, yet they start from a considerably lower level.

TABLE IV: State-of-the-art comparison per manipulation (AUC / BA, %). Bold: best POI-specific method; italic: best general deepfake detector.
<table><tr><td rowspan="2">Method</td><td>DF-TIMIT</td><td colspan="4">FakeAVCeleb</td><td colspan="4">DeepSpeak</td><td colspan="4">KoDF</td><td>Overall</td></tr><tr><td>FS</td><td>FS</td><td>FS+LS</td><td>LS</td><td>All</td><td>FS</td><td>RE</td><td>LS</td><td>All</td><td>FS</td><td>RE</td><td>LS</td><td>All</td><td>Mean</td></tr><tr><td colspan="10">High quality</td><td colspan="7"></td></tr><tr><td>RealForensics [19]</td><td>100.0/100.0</td><td>98.2/98.8</td><td>95.7/94.3</td><td>75.7/78.9</td><td>86.9/84.9</td><td>86.5/90.0</td><td>84.9/84.0</td><td>86.1/86.7</td><td>85.5/83.5</td><td></td><td>87.1/85.5</td><td>94.5/91.4</td><td>99.2/98.1</td><td>93.1/89.9</td><td>91.4/89.6</td></tr><tr><td>LipForensics [20]</td><td>97.9/98.1</td><td>96.5/96.2</td><td>96.1/94.4</td><td>95.7/94.3</td><td>96.3/93.8</td><td>84.1/89.5</td><td>93.7/93.2</td><td>97.8/97.4</td><td>93.6/92.6</td><td></td><td>88.7/88.8</td><td>91.2/88.5</td><td>99.6/99.1</td><td>93.1/90.8</td><td>95.2/93.8</td></tr><tr><td>FTCN [21]</td><td>100.0/100.0</td><td>80.5/84.9</td><td>86.8/87.4</td><td>80.8/84.8</td><td>82.6/82.7</td><td>88.7/92.2</td><td>73.4/80.5</td><td>83.2/87.0</td><td>79.4/83.3</td><td></td><td>90.3/89.8</td><td>92.0/88.5</td><td>98.2/97.5</td><td>93.5/90.8</td><td>88.9/89.2</td></tr><tr><td>Seferbekov [23]</td><td>90.0/92.5</td><td>94.3/93.5</td><td>98.0/97.1</td><td>81.1/88.1</td><td>89.2/89.0</td><td>57.2/74.8</td><td>59.3/69.6</td><td>92.9/92.9</td><td>70.0/75.7</td><td></td><td>92.9/94.4</td><td>81.7/81.0</td><td>98.4/97.9</td><td>90.9/89.6</td><td>85.0/86.7</td></tr><tr><td>POI-Forensics [5]</td><td>89.6/93.3</td><td>81.6/83.8</td><td>76.5/80.5</td><td>52.8/71.5</td><td>66.7/73.1</td><td>94.9/96.8</td><td>65.1/73.8</td><td>43.4/59.9</td><td>62.0/68.4</td><td></td><td>95.4/94.6</td><td>71.1/71.2</td><td>80.9/78.2</td><td>84.5/81.8</td><td>75.7/79.2</td></tr><tr><td>ID-Reveal [4]</td><td>97.6/97.4</td><td>68.9/75.7</td><td>76.5/79.1</td><td>59.7/72.0</td><td>67.1/72.1</td><td>93.9/95.6</td><td>72.2/77.0</td><td>64.4/73.9</td><td>72.9/75.8</td><td></td><td>93.6/91.2</td><td>67.0/67.9</td><td>87.2/82.8</td><td>83.0/80.1</td><td>80.1/81.4</td></tr><tr><td>Ours</td><td>100.0/100.0</td><td>90.8/90.9</td><td>92.8/92.2</td><td>63.7/75.5</td><td>78.8/79.4</td><td>99.4/99.5</td><td>88.0/90.0</td><td>87.3/90.3</td><td>89.4/89.6</td><td></td><td>96.9/96.1</td><td>89.7/86.2</td><td>97.2/95.0</td><td>94.8/91.1</td><td>90.8/90.0</td></tr><tr><td colspan="10">Low quality</td><td colspan="7"></td></tr><tr><td></td><td>77.0/84.4</td><td>82.2/83.5</td><td>54.5/65.5</td><td>34.6/57.9</td><td>51.3/62.1</td><td>61.1/74.7</td><td>60.6/68.4</td><td>83.5/83.6</td><td></td><td>68.8/72.0</td><td>74.4/74.8</td><td>82.0/79.6</td><td>93.3/88.6</td><td>81.4/78.8</td><td></td></tr><tr><td>RealForensics [19] LipForensics [20]</td><td>73.3/82.2</td><td>80.4/82.7</td><td>59.0/67.3</td><td>59.1/67.1</td><td>64.2/68.0</td><td>60.8/74.4</td><td>48.1/61.1</td><td>74.0/77.3</td><td></td><td>59.1/65.9</td><td>79.3/80.0</td><td>83.2/80.8</td><td>96.7/94.0</td><td>85.1/82.8</td><td>69.6/74.3 70.4/74.7</td></tr><tr><td>FTCN [21]</td><td>88.5/91.0</td><td>34.2/59.4</td><td>24.4/55.1</td><td>33.2/59.6</td><td>30.7/56.4</td><td>58.6/72.6</td><td>59.5/68.0</td><td>58.1/67.6</td><td></td><td>58.7/65.8</td><td>56.8/62.2</td><td>66.3/67.2</td><td>79.3/76.6</td><td>64.3/65.7</td><td>60.6/69.7</td></tr><tr><td>Seferbekov [23]</td><td>55.3/72.8</td><td>79.2/80.4</td><td>82.9/83.1</td><td>45.3/65.1</td><td>64.3/70.1</td><td>53.0/71.3</td><td>49.7/61.6</td><td>60.8/68.4</td><td></td><td>54.0/62.2</td><td>82.8/83.8</td><td>53.6/60.3</td><td>81.9/80.2</td><td>73.1/74.0</td><td>61.7/69.8</td></tr><tr><td>POI-Forensics [5]</td><td>88.4/92.8</td><td>81.3/83.3</td><td>76.3/80.1</td><td>53.5/71.0</td><td>66.9/72.8</td><td>95.1/96.8</td><td>61.5/70.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ID-Reveal [4]</td><td>97.8/98.1</td><td>66.5/73.6</td><td>71.7/75.7</td><td>55.9/68.8</td><td>63.2/69.3</td><td>93.4/95.2</td><td>66.8/72.5</td><td>40.9/58.2 55.0/66.3</td><td></td><td>59.2/65.8 66.8/70.3</td><td>95.3/94.5 92.4/89.8</td><td>70.0/70.3 63.6/65.5</td><td>79.7/77.0 82.7/79.1</td><td>83.8/81.2 80.3/77.9</td><td>74.6/78.1</td></tr><tr><td>Ours</td><td>100.0/100.0</td><td>86.7/87.0</td><td>88.9/88.1</td><td>60.5/71.3</td><td>75.2/75.8</td><td>98.8/98.9</td><td>76.5/79.3</td><td>71.5/76.7</td><td></td><td>78.0/78.0</td><td>96.4/95.7</td><td>82.8/79.8</td><td>94.9/91.1</td><td>91.6/87.6</td><td>77.0/78.9 86.2/85.3</td></tr></table>

## V. CONCLUSIONS

We have taken apart the 3DMM description that referencebased POI detectors adopt as a whole, and measured what each of its parts contributes to the detection. Its groups of coefficients turn out to be largely redundant, and their temporal evolution to be worth a real but bounded amount, while the dense surface that the same fit returns, and that these detectors discard, carries information the coefficients do not. Assembling the best configuration into MOTIF improves on both state-of-the-art POI detectors in every cell of our benchmark and at both quality levels. A 3DMM fitted consistently over a clip, rather than frame by frame, would separate identity from behavior more sharply, and facial regions other than the mouth remain to be explored.

## REFERENCES

[1] C. Koutlis, A. Pianese, D. Cozzolino, M. Schinas, S. Mylonas, L. Verdoliva, and S. Papadopoulos, “Video Deepfake Detection: Challenges and Recent Trends,” in Countering Disinformation in the Era of Generative AI. Springer, 2026, pp. 215–246.

[2] X. Dong, J. Bao, D. Chen, T. Zhang, W. Zhang, N. Yu, D. Chen, F. Wen, and B. Guo, “Protecting Celebrities from DeepFake with Identity Consistency Transformer,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[3] S. Agarwal, H. Farid, Y. Gu, M. He, K. Nagano, and H. Li, “Protecting World Leaders Against Deep Fakes,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019.

[4] D. Cozzolino, A. Rossler, J. Thies, M. Niesner, and L. Verdoliva, “ID-Reveal: Identity-aware DeepFake Video Detection,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[5] D. Cozzolino, A. Pianese, M. Nießner, and L. Verdoliva, “Audio-Visual Person-of-Interest DeepFake Detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2023.

[6] G. Petmezas, V. Vanian, K. Konstantoudakis, E. E. I. Almaloglou, and D. Zarpalas, “Video deepfake detection using a hybrid CNN-LSTM-Transformer model for identity verification,” Multimedia Tools and Applications, 2025.

[7] D. Salvi, V. Negroni, S. Mandelli, P. Bestagini, and S. Tubaro, “Phoneme-Level Analysis for Person-of-Interest Speech Deepfake Detection,” in IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2025.

[8] K. Shiohara, T. Yamasaki, and V. Golyanik, “ExposeAnyone: Personalized Audio-to-Expression Diffusion Models Are Robust Zero-Shot Face Forgery Detectors,” arXiv preprint arXiv:2601.02359, 2026.

[9] B. Egger, W. A. P. Smith, A. Tewari, S. Wuhrer, M. Zollhoefer, T. Beeler, F. Bernard, T. Bolkart, A. Kortylewski, S. Romdhani, C. Theobalt, V. Blanz, and T. Vetter, “3D Morphable Face Models - Past, Present, and Future,” ACM Transactions on Graphics (TOG), vol. 39, pp. 1–38, 2020.

[10] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” in International Conference on Machine Learning (ICML), 2020.

[11] Z. Wang, X. Zhu, T. Zhang, B. Wang, and Z. Lei, “3D Face Reconstruction with the Geometric Guidance of Facial Part Segmentation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[12] X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott, “Multi-Similarity Loss With General Pair Weighting for Deep Metric Learning,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[13] J. S. Chung, A. Nagrani, and A. Zisserman, “VoxCeleb2: Deep Speaker Recognition,” in Interspeech, 2018.

[14] H. Khalid, S. Tariq, M. Kim, and S. S. Woo, “FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset,” in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2021.

[15] P. Korshunov and S. Marcel, “DeepFakes: A New Threat to Face Recognition? Assessment and Detection,” arXiv preprint arXiv:1812.08685, 2018.

[16] S. Barrington, M. Bohacek, and H. Farid, “The DeepSpeak Dataset,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026.

[17] P. Kwon, J. You, G. Nam, S. Park, and G. Chae, “KoDF: A Largescale Korean DeepFake Detection Dataset,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[18] K. R. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar, “A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild,” in ACM International Conference on Multimedia (ACM MM), 2020.

[19] A. Haliassos, R. Mira, S. Petridis, and M. Pantic, “Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[20] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, “Lips Don’t Lie: A Generalisable and Robust Approach to Face Forgery Detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[21] Y. Zheng, J. Bao, D. Chen, M. Zeng, and F. Wen, “Exploring Temporal Coherence for More General Video Face Forgery Detection,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[22] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Niessner, “FaceForensics++: Learning to Detect Manipulated Facial Images,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[23] S. Seferbekov, “A prize winning solution for the DFDC Challenge,” https://github.com/selimsef/dfdc deepfake challenge, 2020.