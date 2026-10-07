# Post-Training Semantic Lifting for 3D Gaussian Splatting: Separating Detector, Lifting and Representation Error

Iván Verdugo Guerra<sup>∗</sup> Ezequiel López Rubio Jorge García González

Department of Computer Languages and Computer Science

University of Málaga

ivanver@uma.es elr@uma.es jorgegarcia@uma.es

October 2026

Preprint

Code: GitHub repository

![](images/ad191c0ee391682c14eee934ea26621d9e9588ada7e69c9c6ae4d6536b3543d9.jpg)  
Figure 1: One view of the ScanNet++ test scene 21d970d8de. Even though YOLO does not detect the tables in this view, they are still labelled, since it detects them in other views.

## Abstract

The same Gaussian of a 3D Gaussian Splatting model is seen from many views, and these views do not always agree on the class it belongs to. The Gaussian may be occluded in some of them, and the confidence of the detector is not the same from one view to another. The ground truth, on the other hand, is given as an annotated mesh, because two training runs do not produce the same Gaussians. In this work, we propose a post-training lifting method that works with one target class at a time and combines the information coming from all the views. Target and nontarget evidence are accumulated simultaneously, weighted by the visibility of each Gaussian in each view. After that, the Gaussians are filtered with two thresholds: a main threshold β selects the high-confidence seeds, and a lower one γβ adds the connected components around them. For the evaluation, the labels are transferred from the Gaussians to the mesh vertices that are both visible and annotated. With this design, we can separate three sources of error: the 2D detector, the lifting and the transfer between representations. The thresholds and the transfer operator are chosen on seven Replica validation scenes, and the method is evaluated on ten held-out ScanNet++ scenes with the same values for every scene and class. The mean mIoU on the validation scenes was 0.93 with masks from the dataset annotations and 0.65 with YOLO masks, and on the ScanNet++ test scenes it was 0.80 and 0.54. Compared with thresholding the evidence per view, as a previous version of the method did, the fraction improves the test mIoU by 0.24 and makes it possible to use a single threshold for all the classes and scenes of both datasets. Finally, the error analysis shows that most of the remaining error comes from the detector.

Keywords: 3D Gaussian Splatting, semantic segmentation, label lifting, evidence accumulation, hysteresis thresholding, error decomposition.

## 1 Introduction

3D Gaussian Splatting represents a reconstructed scene as a set of anisotropic 3D Gaussians [10, 11]. Each of them stores its own geometry and appearance, that is, its position, covariance, opacity and spherical harmonics. The class to which the Gaussian belongs, however, is not part of these parameters.

In most cases, the semantic information comes from the 2D masks that a detector produces on the images used for the reconstruction. These masks are projected back onto the Gaussians, and the main problem is that the same Gaussian is seen by several cameras that do not always agree: it may be occluded in one view, or detected with a diferent confidence in another. There is also a second dificulty. The labels are evaluated on a mesh, which is a set of fixed vertices, while the Gaussian centres form a sparse cloud. As a consequence, a labelling that looks correct on the Gaussians can still lose accuracy when it is transferred to the mesh, since the area that a primitive covers in a view depends on its anisotropic shape and on its depth.

In general, the methods that add semantics to a Gaussian scene can be divided into two groups. Some of them train semantic features into the primitives themselves, either during or after the reconstruction [12, 20, 28, 27], and others lift the labels onto the Gaussians once the model is trained [22, 16, 25]. The method that we present in this paper belongs to the second group.

In our case, the masks are the only semantic input of the method. For each camera, the means and covariances of the visible Gaussians are projected [35] and their contribution is weighted with alpha compositing in depth order. In the same pass, two quantities are accumulated for every Gaussian: the evidence that it belongs to the target class and the evidence that it belongs to anything else. The ratio between the first one and the sum of both is what we call the target evidence fraction, and it is thresholded at a fixed β. A Gaussian below that threshold but above γβ still gets the label of the target class when it belongs to the same connected component as a seed. Finally, for the evaluation, the labelled Gaussians are transferred to the mesh vertices, where they are compared with the ground truth. Figure 1 shows an example of the method on one view of a test scene.

The main aim of our evaluation is to find out in which part of the pipeline the accuracy is lost. YOLO masks [9] are not the same as the masks derived from the dataset annotations, and this is something that we can take advantage of: running the same method with both sources allows us to separate the detector from the rest of the pipeline. With the annotation masks, whatever is still lost comes either from the lifting or from the representation, because a mesh vertex may not be reachable from any Gaussian, or may be reached mostly by Gaussians of the wrong class. This second part is measured with a reference that sends the annotations through the same transfer operator.

The development of the method was carried out on office\_0 from Replica [24] and 7831862f02 from Scan-Net++ [29], and both scenes were left out of validation and test. The pair $( \beta , \gamma )$ and the operator that carries the labels to the mesh are selected with a sweep over seven Replica validation scenes, and ten held-out ScanNet++ scenes are then evaluated with the same parameters. Both mask sources are used in both datasets.

The main contributions of this work are the following:

• A semantic voting scheme in which the visibility is taken into account. Target and non-target evidence are accumulated at the same time, and a spatial hysteresis makes the result more stable.

• A PyTorch implementation by tiles of the visibility computation of the oficial Gaussian Splatting renderer [11], adapted to semantic votes weighted by confidence. All the diferences with respect to the CUDA code are documented.

• A selection protocol that fixes the pair $( \beta , \gamma )$ and the transfer operator from the validation mean and the variation between scenes, with a tolerance of 0.01 mIoU.

• An error analysis that separates 2D detection, lifting and representation, and two analyses that explain the result: the evidence fraction against the evidence per view of a previous version, which it improves by 0.24 mIoU on the test scenes, and the 3D result against the 2D quality of the detector. All of this comes with a container workflow for Docker and Apptainer and cached intermediate data to reproduce the experiments.

The rest of the paper is organised as follows. Section 2 reviews the related work, Section 3 describes the method and Section 4 its implementation. Sections 5 and 6 present the evaluation protocol and the experimental design, and Section 7 reports the results. The paper ends with the limitations of the method and the conclusions of the work, in Sections 8 and 9.

## 2 Related Work

## 2.1 Gaussian scene representations

Since the appearance of 3D Gaussian Splatting, several works have proposed changes in the geometry of the primitive. For example, 2D Gaussian Splatting replaces the ellipsoid with an oriented planar disk. A ray meets a disk at a single point, so the depth and the normal do not change with the viewpoint, and the resulting geometry is more accurate than with volumetric primitives [8]. Our implementation is based on the oficial release [11], and this is why it keeps the original Gaussians and the same training procedure.

Other methods include the semantic information in the representation itself, during the optimisation. Semantic-NeRF learns an implicit semantic field together with the radiance field [32]. This is not only done with NeRF, as Gaussian models are also trained in this way, either with language features for text queries [12, 20], or with identity and afinity features that group the primitives into objects [28, 2, 27]. In all these cases, the semantic label is learnt together with the model itself.

## 2.2 Label lifting from images to 3D

The idea of transferring image predictions to a common 3D representation is not new, and it is in fact older than Gaussian scenes. In RGB-D pipelines, the predictions of each frame are accumulated during the reconstruction with Bayesian updates and pairwise regularisation over the points [7, 18]. Closer to our work, Kundu et al. render a reconstructed mesh from virtual viewpoints and aggregate the predictions of each view on its vertices [14], and Genova et al. reuse this aggregation to train a 3D segmentation network from images only [5]. All of them have to deal with calibration, visibility and views that contradict each other, and they difer mainly in the weight given to each observation. In our case, this weight comes from the projected covariance of the Gaussian and from its contribution to the pixel.

An aggregated label usually accumulates errors from several sources: the images, the aggregation step and the representation where it ends up. To study them, Kundu et al. [14] compare the accuracy of the predictions of each view with the accuracy on the mesh where they are aggregated. Semantic-NeRF, on the other hand, is trained on labels that have been corrupted on purpose, which shows how much of that noise is absorbed by the consistency between views [32]. To measure the cost of the representation alone, we send the ground-truth annotation through the same transfer operator as the prediction. A similar idea has already been used in LiDAR segmentation, where each point takes the label of its cell and the resulting mIoU is interpreted as the upper bound imposed by the grid [31, 34], and also with superpixels in indoor RGB-D segmentation [6]. As our transfer operator has its own parameters, we report this value together with the settings that produced it.

## 2.3 Post-training label lifting on Gaussian models

The closest work to ours is CDSeg, submitted on 6 August 2026 [25]. It builds a Gaussian carrier that can be rendered, either by completing each input point with a single Gaussian and keeping its index, or by reusing the primitives of a Gaussian scene that is already optimised. After that, it records the associations between pixels and primitives given by the renderer, aggregates the mask labels by voting and applies a local filter. The setting is similar to ours, as in both cases the labels go from the images to the primitives after the reconstruction, using the visibility of the renderer. Nevertheless, the goal of both works is diferent. CDSeg is a general interface for both instance and semantic tasks, which also handles LiDAR points, and its main priorities are keeping the indices and building the carrier. We work instead with one class at a time, we weight each vote by the confidence of the detector and we let target and non-target evidence compete in the same pass. We also measure the cost of the transfer to an

annotated mesh.

The rest of the post-training methods difer mainly in the way in which the information of the images is assigned to each primitive. FlashSplat solves the assignment as a global linear problem with a background bias [22], while LUDVIG aggregates DINO, SAM or CLIP features in the scene and refines them with graph difusion [16]. Finally, PointGS densifies a point cloud into a Gaussian space, distils the semantics there and registers the labels back onto the input points [23].

## 2.4 Thresholding and spatial consistency

Applying a threshold to a score and then using spatial information to correct the result is a very common idea in computer vision, and it has been used for many years. For example, the Canny edge detector keeps a weak response when it is connected to a strong one [1]. Our spatial hysteresis does the same with two thresholds, but on a graph of Gaussians instead of on an image. Other approaches in this line are dense CRFs, which regularise the pixel labels with pairwise potentials over the whole image [13], and graph-based segmentation, which merges regions according to a predicate on the edge weights [4]. In our case, the approach is simpler, as we only use one connectivity radius and one low threshold.

## 3 Method

## 3.1 Problem formulation

Let $\mathcal { G } = \{ g _ { i } \} _ { i = \cdot } ^ { N }$ be a Gaussian model that has already been optimised for photometric reconstruction from a set of calibrated cameras V. Each Gaussian $g _ { i }$ is defined by its mean $\pmb { \mu _ { i } } \in \mathbb { R } ^ { 3 }$ , a rotation $\mathbf { R } _ { i } \in S O ( 3 )$ , a diagonal scale $\mathbf { S } _ { i }$ and an opacity $o _ { i } \in [ 0 , 1 ]$ . Its covariance, which is the one used during the optimisation [10], is obtained from the rotation and the scale as

$$
\pmb { \Sigma } _ { i } = \mathbf { R } _ { i } \mathbf { S } _ { i } \mathbf { S } _ { i } ^ { \top } \mathbf { R } _ { i } ^ { \top } .\tag{1}
$$

The segmentation is posed as a one-vs-rest problem, which is solved once for each target class of the dataset taxonomy. Given a class k, every camera provides a 2D mask that separates target and non-target pixels, with a confidence for each pixel. From these masks, the method assigns a binary label $\ell _ { i } ^ { k } \in \{ 0 , 1 \}$ to each Gaussian, and these labels are then transferred to the mesh vertices of the scene for the evaluation, as described in Section 3.7.

During the projection, two non-negative quantities are accumulated for each Gaussian: the target evidence $E _ { i } ^ { + }$ and the non-target evidence $E _ { i } ^ { - }$ , both with the same visibility weights of Eq. (11). Their sum is zero only when the Gaussian has no contributing view, that is, when it lies behind the near plane or its projected centre falls outside the image in every view where the class appears. Occlusion alone does not make the sum zero, because small contributions are not discarded at this point. The not have a confidence of their own, so decision is made on the support index set

![](images/115b43aa5140465b2644a33bd47912bc87e28067740d024c3d891f1d374585be.jpg)  
Figure 2: The six stages of the method. Stages 2 and 3 do not depend on the class, and a change of $\beta$ or $\gamma$ only runs the pipeline again from stage 5.

$$
\begin{array} { r } { S ^ { k } = \{ i \in \{ 1 , \dots , N \} : E _ { i } ^ { + } + E _ { i } ^ { - } > 0 \} . } \end{array}\tag{2}
$$

Gaussians outside $S ^ { k }$ keep $\ell _ { i } ^ { k } = 0$ . Inside it, the label of $g _ { i }$ depends on the pair $( E _ { i } ^ { + } , E _ { i } ^ { - } )$ and on the other Gaussians of $S ^ { k }$ that lie within a fixed radius of $\pmb { \mu } _ { i }$

The six stages of the method are shown in Figure $^ { 2 , }$ and the following subsections describe them in the order in which they run.

## 3.2 Two-dimensional masks and confidence

Each image provides two inputs to the method: a label map $L _ { v } ,$ whose value at each pixel is the identifier stored for that pixel, and a detector confidence map $q _ { v }$ with values in [0, 1]. If the dimensions do not match, the detector masks are resampled to the camera resolution with nearest-neighbour interpolation. The masks obtained from the datasets are written in the same format, with confidence one on every annotated pixel, so the following stages treat both sources in the same way.

For the target class k, the pixel grid $\Omega _ { v }$ of view v is split into

$$
\mathcal { P } _ { v } ^ { + } = \{ p \in \Omega _ { v } : L _ { v } ( p ) = k \} , \qquad \mathcal { P } _ { v } ^ { - } = \Omega _ { v } \setminus \mathcal { P } _ { v } ^ { + } ,\tag{3}
$$

where the identifier zero is reserved for pixels without any detection. These pixels also belong to $\mathcal { P } _ { v } ^ { - }$ , but they do

$$
q _ { v } ( p ) = \left\{ \begin{array} { l l } { \hat { q } _ { v } ( p ) , } & { L _ { v } ( p ) \neq 0 , } \\ { q _ { 0 } , } & { L _ { v } ( p ) = 0 , } \end{array} \right.\tag{4}
$$

where $\hat { q } _ { v } ( p ) = s _ { v } ( p )$ c is the value of the soft mask $s _ { v }$ at pixel p multiplied by the score c of the whole detection. In this way, the confidence changes inside the same object mask. The value $q _ { 0 } = 0 . 2 5$ is a fixed confidence that is given to the background pixels. Therefore, a pixel with a detected label keeps the confidence of its detection, whether that label is the target or not. At first, the same confidence was given to every pixel of a detection, which was simpler, and then we let it vary per pixel with the soft mask, which slightly improved the results.

## 3.3 Projection of means and covariances

Given a Gaussian $g _ { i }$ , its mean $\pmb { \mu } _ { i } \in \mathbb { R } ^ { 3 }$ is first transformed to camera coordinates $\pmb { \mu } _ { i } ^ { \mathrm { c a m } } = ( x _ { i } , y _ { i } , z _ { i } ) ^ { \top }$ with the rigid transform

$$
\pmb { \mu } _ { i } ^ { \mathrm { c a m } } = \mathbf { W } \pmb { \mu } _ { i } + \mathbf { t } ,\tag{5}
$$

and only the Gaussians in front of the near plane, that is, with $z _ { i } > 0$ , are kept. For a pinhole camera, the projected 2D centre $\pmb { \mu } _ { i } ^ { \prime } = ( u _ { i } , v _ { i } ) ^ { \top }$ is

$$
u _ { i } = f _ { x } \frac { x _ { i } } { z _ { i } } + c _ { x } , \qquad v _ { i } = f _ { y } \frac { y _ { i } } { z _ { i } } + c _ { y } ,\tag{6}
$$

where $\left( c _ { x } , c _ { y } \right)$ is the principal point given by the camera calibration.

The covariance is projected with the local afine approximation of EWA splatting [35], in the same form as in the Gaussian Splatting rasteriser [10]:

$$
\begin{array} { r } { \pmb { \Sigma _ { i } ^ { \prime } } = \mathbf { J } _ { i } \mathbf { W } \pmb { \Sigma } _ { i } \mathbf { W } ^ { \top } \mathbf { J } _ { i } ^ { \top } , } \end{array}\tag{7}
$$

where W is the rotation part of the world-to-camera transform and $\mathbf { J } _ { i }$ is the Jacobian of the projection evaluated at $\pmb { \mu } _ { i } ^ { \mathrm { c a m } }$ ，

$$
\mathbf { J } _ { i } = \left[ { \begin{array} { c c c } { f _ { x } / z _ { i } } & { 0 } & { - f _ { x } x _ { i } / z _ { i } ^ { 2 } } \\ { 0 } & { f _ { y } / z _ { i } } & { - f _ { y } y _ { i } / z _ { i } ^ { 2 } } \end{array} } \right] .\tag{8}
$$

As in the standard rasteriser [10], a low-pass filter of $0 . 3 \mathrm { p x } ^ { 2 }$ is added to the projected covariance, that is, we compute $\pmb { \Sigma } _ { i } ^ { \prime } + 0 . 3 \mathbf { I } _ { 2 }$ before the inversion. In this way, the covariance of the Gaussians smaller than a pixel can still be inverted without numerical problems. The inverse is symmetric and is stored as three conic coeficients $( A , B , C )$ , so the exponent at an ofset $( \Delta u , \Delta v )$ becomes the quadratic form $- \textstyle { \frac { 1 } { 2 } } ( A \Delta u ^ { 2 } + C \Delta v ^ { 2 } ) - B \Delta u \Delta v$ and no matrix has to be inverted per pixel. From the largest eigenvalue $\lambda _ { i }$ of the filtered covariance, we obtain the conservative radius

$$
r _ { i } = \left\lceil 3 \sqrt { \lambda _ { i } } \right\rceil ,\tag{9}
$$

which bounds, in whole pixels, the region that $g _ { i }$ can reach. The Gaussians whose projected centre falls outside the image are discarded, as Section 3.4 explains.

## 3.4 Tile rasterisation and visibility weights

As in the original rasteriser, the image is divided into square tiles of 16 pixels per side. Each remaining Gaussian is assigned to every tile that its radius intersects, and inside a tile all the relevant pairs of Gaussian and pixel are evaluated at once with tensor broadcasting.

For Gaussian i and pixel p at position $\mathbf { x } _ { p } .$ , the alpha value is the projected Gaussian multiplied by the opacity,

$$
\begin{array} { r } { \alpha _ { i , p } = o _ { i } \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } _ { p } - { \pmb \mu } _ { i } ^ { \prime } ) ^ { \top } \left( \pmb { \Sigma } _ { i } ^ { \prime } \right) ^ { - 1 } ( \mathbf { x } _ { p } - { \pmb \mu } _ { i } ^ { \prime } ) \right) , } \end{array}\tag{10}
$$

and the Gaussians that cover p are composited from front to back after sorting them by depth [17]:

$$
T _ { i , p } = \prod _ { j < i } ( 1 - \alpha _ { j , p } ) , \qquad w _ { i , p } = \alpha _ { i , p } T _ { i , p } ,\tag{11}
$$

where $j < i$ runs over the Gaussians that cover $p$ and are closer to the camera. When rendering, $T _ { i , p }$ is the transmittance that reaches $g _ { i }$ and $w _ { i , p }$ its contribution to the colour of $p .$ The accumulation of Section 3.5 uses $w _ { i , p }$ as the weight of the semantic evidence that pixel $p$ gives about Gaussian i. As a result, a Gaussian that is hidden behind closer ones receives almost no weight from that view.

This visibility computation follows, with some changes, the CUDA rasteriser released with 3D Gaussian Splatting [10, 11], but it is written in Python and PyTorch. The main diference with respect to the original implementation is in the culling step. The culling of Section 3.3 is stricter than in the reference, which keeps any Gaussian whose radius intersects a tile. We decided to change it because we observed that a few very large Gaussians, with their centres outside the frustum, collected target evidence from several views through the tiles that their support touched. For this reason, in our implementation the centre of a Gaussian must lie inside the frustum, even if its radius intersects a tile. The reference also cuts the compositing loop short. It discards a contribution once its alpha is too small to be visible, and it stops processing a pixel once its accumulated transmittance is negligible. In practice, we do not apply either of these cuts, so the nearly transparent contributions also reach Equation (13), and an opaque Gaussian centred on a pixel takes the transmittance of that pixel to zero.

## 3.5 Target and non-target evidence

Both channels are filled in the same pass over the cameras. A camera contributes to class k when a mask was matched to it and that mask contains at least one pixel of the class,

$$
\mathcal { V } ^ { k } = \{ v \in \mathcal { V } : \mathcal { P } _ { v } ^ { + } \neq \emptyset \} ,\tag{12}
$$

and using the pixel sets of Equation (3), the confidence of Equation (4) and the visibility weight of Equation (11),

$$
E _ { i } ^ { \pm } = \sum _ { v \in \mathscr { V } ^ { k } } \sum _ { p \in \mathscr { P } _ { v } ^ { \pm } } q _ { v } ( p ) w _ { i , p } .\tag{13}
$$

The weight $w _ { i , p }$ is computed only once per view and is used by both sums, so target and non-target evidence see the same occlusions and the same visibility, and the only thing that separates the two channels is the mask. A view where the class does not appear is skipped before any accumulation, so it contributes to neither channel. Otherwise, the cameras that never see the class would dilute the evidence, and the meaning of the $\beta$ threshold, introduced in Section 3.6, would change from one class to another.

The target evidence fraction of Gaussian i is

$$
\rho _ { i } ^ { k } = \frac { E _ { i } ^ { + } } { E _ { i } ^ { + } + E _ { i } ^ { - } } , \qquad i \in \mathcal { S } ^ { k } ,\tag{14}
$$

which lies in $[ 0 , 1 ]$ by construction. Note that the fraction also depends on $\bar { \mathcal { V } } ^ { k }$ . A camera that never sees class k only has non-target pixels, and if these cameras were included, in a scene where the class is rare every fraction would be pushed towards zero. This alternative policy, in which every camera enters both sums, is the one reported in Section 6.4.

![](images/df89bce317b43b637c2fad7910a07a10b62562c3476d46967f34651971244799.jpg)  
Figure 3: Hysteresis on the graph of Gaussian centres. A component takes the class only when it contains a seed.

## 3.6 Threshold and spatial hysteresis

The parameter $\beta \in ( 0 , 1 ]$ is the fraction of target evidence that a Gaussian must reach by itself. For any threshold $t \in [ 0 , 1 ]$ , we define the corresponding level set as

$$
{ \mathcal { S } } _ { t } ^ { k } = \{ i \in { \mathcal { S } } ^ { k } : \rho _ { i } ^ { k } \geq t \} .\tag{15}
$$

We added a second threshold after seeing what happened with a lower $\beta \colon$ the Gaussians of the object came back, but so did the noise around it, and hysteresis allows us to recover the first ones without the second. With $\gamma \in [ 0 , 1 )$ ， we have by construction $S _ { \beta } ^ { k } \subseteq { \mathcal { S } } _ { \gamma \beta } ^ { k }$ . The elements of $S _ { \beta } ^ { k }$ are the seeds, and the elements of $S _ { \gamma \beta } ^ { k }$ are split into the connected components of a graph that joins any two Gaussians within a distance η of each other. All the Gaussians of a component are labelled as the target class when the component contains at least one seed. When $\gamma = 0$ there is no expansion, and only the seeds get label one.

An example of this process is shown in Figure 3. The idea is the same as in the hysteresis of edge detection [1], but with the neighbourhood defined in the scene itself. As in Canny’s method, the evidence is not propagated between Gaussians, so Equation (14) does not change. It should be noted that the method gives one binary label per class and does not distinguish between instances.

The radius η was fixed on the development scenes, and the rule of Section 6.2 gave $( \beta , \gamma ) = ( 0 . 7 , 0 . 8 )$ on the seven Replica validation scenes. In the evaluation, the same pair $( \beta , \gamma )$ is used for every class and every scene.

## 3.7 Transfer from Gaussians to mesh

Since the prediction is evaluated on the mesh, it is necessary to transfer the selected Gaussians to it. The two operators that we consider look at the same neighbourhood. An annotated vertex v collects the Gaussians whose centres are within a radius τ of it,

$$
\begin{array} { r } { \mathcal { N } ( \mathbf { v } ) = \big \{ i \in \{ 1 , \dots , N \} : \| \pmb { \mu _ { i } } - \mathbf { v } \| \le \tau \big \} , } \end{array}\tag{16}
$$

and a vertex with an empty neighbourhood stays unlabelled.

![](images/a403f4e407c074fce9b690677b089ef63bb9cbfc4dda9738d9d7fdfc3495ba97.jpg)  
Figure 4: The two transfer operators at a vertex next to the border of an object. With θ below 0.5, the radius vote gives the class to vertices where the nearest Gaussian does not, so the labelled region grows.

The choice of this operator has an important efect on the results, as we will see in Section 7.7, so we consider two of them. The first operator is the nearest Gaussian. The vertex takes the label of the Gaussian of ${ \mathcal { N } } ( { \bf { v } } )$ whose centre is closest to it, with no weighting. In this way, the labelled surface follows the labelled Gaussians and nothing more.

The second operator, which we call radius vote, is a weighted vote inside the neighbourhood. In this case, each Gaussian of the neighbourhood votes with a weight that decreases with the squared distance and increases with the opacity,

$$
\nu _ { i } ( \mathbf { v } ) = { \frac { \operatorname* { m a x } ( o _ { i } , o _ { \operatorname* { m i n } } ) } { \| \mu _ { i } - \mathbf { v } \| ^ { 2 } + \epsilon } } ,\tag{17}
$$

where $o _ { \mathrm { m i n } }$ prevents an almost transparent Gaussian from losing its vote completely, and $\epsilon = 1 0 ^ { - 1 0 }$ keeps the quotient finite when a centre falls on the vertex itself. The votes are split between the Gaussians that carry the class and the ones that do not,

$$
\begin{array} { l } { { \displaystyle { \cal S } ^ { + } ( { \bf v } ) = \sum _ { i \in \mathcal { N } ( { \bf v } ) } \ell _ { i } ^ { k } \nu _ { i } ( { \bf v } ) } , } \\ { { \displaystyle { \cal S } ^ { - } ( { \bf v } ) = \sum _ { i \in \mathcal { N } ( { \bf v } ) } \left( 1 - \ell _ { i } ^ { k } \right) \nu _ { i } ( { \bf v } ) } , } \end{array}\tag{18}
$$

and the vertex takes the label of class k when the fraction of the first sum over both of them reaches a threshold θ, as can be seen in Figure 4:

$$
\frac { S ^ { + } ( \mathbf { v } ) } { S ^ { + } ( \mathbf { v } ) + S ^ { - } ( \mathbf { v } ) } \geq \theta .\tag{19}
$$

The term $S ^ { - }$ is a second form of background competition, apart from the $E _ { i } ^ { - }$ channel of Equation (13). Without it, every vertex within τ of a selected Gaussian would take the class, since in a one-vs-rest run there are no Gaussians of other classes to compete with. Note that a value of θ below 0.5 lets the class win a vertex where the background holds more weight, so the labelled region can grow beyond the selected Gaussians, up to a distance τ .

The two operators share τ , while $\theta , o _ { \mathrm { m i n } }$ and the opacity weighting only exist in the radius vote. These parameters were fixed on the development scenes, and the operator itself is chosen on the validation scenes, together with $( \beta , \gamma )$ , as Section 6.2 explains. The rule finally chose the nearest Gaussian.

As for the ground-truth transfer reference, it follows the same path, but starting from the mesh annotation. First, the annotation is transferred to the Gaussians with the radius vote, where every vertex within τ votes with uniform weight, since a vertex has no opacity, and every label of the dataset competes. Then, these Gaussian labels go back to the mesh with the same operator as the prediction. The reference and the prediction only difer in where the Gaussian labels come from, so the reference measures what the representation and the transfer lose, and it depends on the Gaussian model, on the mesh and on the transfer settings.

## 4 Implementation and reproducibility

The pipeline has been implemented taking reproducibility into account. The stages run inside container images with fixed versions, one for training, one for lifting and one for COLMAP [21]. In this way, the versions of CUDA, PyTorch [19] and COLMAP are the same in all the experiments reported here. The host launches these containers and takes care of loading the scenes, the metrics, the cache and the analytics. In general terms, the pipeline is a sequence of stages, from the preparation of the dataset to the metrics, and each stage computes its outputs only from its inputs. The $\beta$ and γ sweeps read the projected votes from the cache instead of computing them again, so the projection of Section 3.4 runs once per scene and class, independently of how many candidates the sweep has.

The elapsed time and the peak CUDA memory of every run and of each of its stages were recorded. For each stage we also store whether it computed its outputs or read them from the cache, so both cases, cache miss and cache hit, are measured on the same scene and the same hardware. These values are the ones reported in Section 7.9.

The analytics of all the runs are stored in several normalised CSV tables, indexed by run, scene, class and $\beta .$ They contain the parameters used to launch each run, the vertex counts per class of Section 5.3, the quantiles of the target evidence fraction, and the counts and metrics behind the numbers of this paper. Moreover, each run saves some execution metadata together with the results, such as the code commit, the GPU model, the driver version and the command-line arguments.

## 5 Experimental setup and evaluation protocol

## 5.1 Datasets and scene splits

We evaluate the method on two public datasets of indoor scenes, Replica and ScanNet++, which are widely used in this area. Replica contains synthetic indoor scenes. We use its semantic meshes [24, 3] and the pre-rendered sequences released with Semantic-NeRF [32, 33]. The cameras are pinhole, with a resolution of $6 4 0 \times 4 8 0$ , focal lengths $f _ { x } = f _ { y } = 3 2 0$ and the principal point at the image centre. In order to reduce the computation time, we subsample each sequence by a factor of five, so only one of every five frames is used in the lifting. The poses are the trajectories that come with the sequences, but we convert them to the COLMAP text format and add a point cloud sampled from the depth frames, so the Gaussian model is trained with those extrinsics.

ScanNet++, on the other hand, contains real indoor scans with geometry from a laser scanner and a denser semantic annotation [29, 30]. Its DSLR images are undistorted with COLMAP [21] using the reconstruction provided by the dataset, with at most 1600 pixels on the long side, and every camera of that model is used. The principal point of Equation (6) is the one given by COLMAP. With respect to the training resolution, for Replica we train the Gaussian model at the full frame size, while for ScanNet++ we train it at half the undistorted resolution.

The ground truth is not annotated in the same way in both datasets, so the preprocessing is also slightly diferent. Replica labels the faces of its mesh, whereas ScanNet++ labels the vertices directly. In Replica, a vertex takes the label of the majority of the faces that meet at it, as long as that majority reaches 0.6 of them. The vertices below this fraction are left unannotated and are never evaluated. Also, Replica renders its sequences from the labelled mesh, so the geometry, the annotation and the images are the same object, while in ScanNet++ they are three independent measurements of the room.

The pre-rendered version of Replica that we use [32] has eight scenes. We used office\_0 as the development scene during the implementation, and the other seven scenes form the validation split, where the sweeps are carried out and from which the scene means and the main tables are obtained. From ScanNet++ we take a held-out test split: 7831862f02 was used for development, and ten other scenes form the test set. These are the downloaded scenes with the most evaluated classes in their 3D annotation, breaking ties by the scene identifier. This rule only reads the annotation, and the list was fixed before evaluating any of them. The configuration selected on Replica is applied to them without retuning it per scene or per class. It should be noted that the development scenes are not included in any validation or test summary.

The names of the classes of each dataset were mapped by hand to the names used by the detector. Replica has six classes and ScanNet++ has seven, and four of them are common to both. Bench only appears in the ScanNet++ development scene, so the test scenes are evaluated on the other six. In the case of ScanNet++, whose vocabulary is much larger, one class of the detector is often mapped to several classes of the dataset. In each scene, the evaluated classes are those of the mapped vocabulary that appear in at least one 2D mask derived from the dataset, and this set is the same for both mask sources. If a class is present in a scene but no sampled camera gives a mask of it, the class is not evaluated in that scene. A class that the detector never proposes is treated as an empty prediction, so its misses are counted and not removed from the scene.

## 5.2 Two-dimensional mask sources

For the detector condition, the yolo26x-seg segmentation checkpoint from the YOLO26 family [9] is used, through the Ultralytics package [26]. Detections with a confidence below 0.75 are discarded, and the soft masks of the remaining ones are binarised at 0.5. This confidence threshold was chosen after a small sweep: at 0.5 too many weak detections entered the vote, and at 0.75 most of them were removed.

In the annotation-derived condition, these masks are replaced by masks rendered from the dataset labels, with the same format and confidence one on every annotated pixel. Replica already provides them, since the Semantic-NeRF sequences include a semantic frame next to each colour frame, so in this dataset the condition works as a 2D ground truth. ScanNet++, however, does not have these frames, and its masks are obtained by rasterising the annotated 3D mesh into each camera with nvdifrast [15], following the oficial ScanNet++ toolkit [30]. Running the same lifting under both conditions allows us to separate the errors of the detector from those of the lifting, and this is what the detector gap of Section 5.5 measures.

## 5.3 Evaluation support

The predictions are evaluated on the annotated vertices of the scene mesh that the sampled cameras have seen. With this criterion, vertices behind a wall or in a hidden part of the room are not counted as misses of the method. The way in which this visibility is decided is diferent in each dataset, as it depends on the data that each one provides. A Replica vertex is visible when it is in front of a camera, it projects inside the image and its depth agrees, within 0.05 metres, with the depth frame of the sequence at that pixel. For ScanNet++, the annotated mesh is rasterised with nvdifrast [15], and the three vertices of a triangle are marked as visible when the triangle covers at least one pixel.

## 5.4 Metrics and aggregation

For each class and scene we report the one-vs-rest IoU, as well as the precision and the recall, which are the usual metrics in semantic segmentation. A vertex counts as annotated when its label belongs to the dataset vocabulary, whether or not that class is evaluated. This means that the negatives of a class are annotated surfaces and not unlabelled space. A class with annotated positives but no predicted positive gets an IoU of zero.

Apart from these metrics, the predictions are also compared with the ground-truth transfer reference. Let $\mathrm { I o U } _ { k } ^ { \mathrm { R } }$ be the IoU obtained when the ground-truth annotations are transferred through the Gaussian model and back to the mesh. When this reference is zero, not even the annotation can reach the class through the model. For this reason, every summary only uses the pairs with a positive reference. Each scene gives one value per class, the mIoU of a scene is the mean over those classes, and the result of a split is the mean over its scenes, together with the standard deviation between scenes. Since a split has only seven or ten scenes, we also give a 95% confidence interval for that mean with a percentile bootstrap over scenes: the scenes are resampled with replacement ten thousand times, and the interval holds the central 95% of the resampled means. When two configurations are compared, the same bootstrap is applied to their diference scene by scene. For the same pairs we compute

$$
\mathrm { I o U } _ { k } ^ { \mathrm { r e l } } = \frac { \mathrm { I o U } _ { k } } { \mathrm { I o U } _ { k } ^ { \mathrm { R } } } ,\tag{20}
$$

and the relative mIoU, mIoU , is its mean. The relative IoU of a class could be greater than one when the lifting reaches annotated vertices that the transferred annotation does not reach.

## 5.5 Error decomposition

One of the main goals of this work is to know how much each part of the pipeline contributes to the final error. For this reason, we use three quantities to separate the sources of error. Let mIoU , mIoU and $\mathrm { m I o U _ { R } }$ be the mIoU obtained with the 2D masks of the detector, with the 2D masks from the dataset annotations and with the ground-truth transfer reference, respectively. We define

$$
\begin{array} { r l } & { g _ { \mathrm { d e t } } = \mathrm { m I o U } _ { \mathrm { A } } - \mathrm { m I o U } _ { \mathrm { D } } , } \\ & { g _ { \mathrm { l i f t } } = \mathrm { m I o U } _ { \mathrm { R } } - \mathrm { m I o U } _ { \mathrm { A } } , } \\ & { g _ { \mathrm { r e p } } = 1 - \mathrm { m I o U } _ { \mathrm { R } } . } \end{array}\tag{21}
$$

The three quantities depend on the settings of Section 3.7, and their sum is $1 - \mathrm { m I o U _ { D } }$ , which is the total error. As the mIoU gives the same weight to all the classes of the dataset, although some classes cover more mesh vertices than others, two errors of the same size do not necessarily correspond to the same number of mislabelled vertices.

Table 1: Configuration fixed before the validation sweep. The last four rows only apply to the radius vote.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Model Gaussian model iteration</td><td>30000</td></tr><tr><td>Masks Detector score threshold Mask binarisation</td><td>0.75 0.5</td></tr><tr><td>Voting Fallback confidence qo Contributing views</td><td>0.25 class present</td></tr><tr><td>Tile size Hysteresis</td><td>16 px</td></tr><tr><td>Connectivity radius η Transfer</td><td>0.05 m</td></tr><tr><td>Operator</td><td>chosen on validation</td></tr><tr><td></td><td></td></tr><tr><td>Radius τ</td><td>0.10 m</td></tr><tr><td>Minimum vote fraction θ</td><td>0.3</td></tr><tr><td></td><td></td></tr><tr><td>Opacity floor  $O _ { \mathrm { { m i n } } }$ </td><td>0.1</td></tr><tr><td></td><td></td></tr><tr><td>Opacity weighting</td><td></td></tr><tr><td>Background competition</td><td>enabled both directions</td></tr></table>

## 6 Experimental design

## 6.1 Frozen configuration

Apart from the operating point, the method has several other parameters, and they were fixed on two development scenes, office\_0 from Replica and 7831862f02 from ScanNet++. The most important ones were the transfer radius and the minimum vote fraction, since together they decide how the filtered Gaussians interact with the mesh.

These two parameters were swept separately, with the radius vote and with the masks derived from the dataset. First, with θ fixed at 0.5, the default value of the implementation, we ran τ over {0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20} metres. Then, with τ fixed at 0.10, we ran θ over $\{ 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 \}$ . Each candidate was scored by the mean mIoU of the two development scenes, and among the candidates whose mean was within 0.01 of the best one, we chose the one where the two scenes difered the least. This gave $\tau = 0 . 1 0$ m and $\theta = 0 . 3$ . Both sweeps are shown in Figure 11 of the appendix. It should be noted that no validation or test scene was used during this procedure.

From this point on, the parameters of Table 1 were kept fixed for the rest of the experiments. The seven Replica validation scenes were used to choose the operating point, and the ten held-out ScanNet++ scenes were evaluated with everything already fixed, without retuning anything per scene or per class.

## 6.2 Operating point selection

A candidate is a transfer operator together with a pair $( \beta , \gamma )$ , where the two operators are the ones of Section 3.7. The grid has thirteen values of $\beta ,$

$$
\begin{array} { r l } & { \beta \in \{ \ 0 . 5 0 , 0 . 7 0 , 0 . 9 0 , 0 . 9 4 , 0 . 9 5 , 0 . 9 6 , 0 . 9 7 , 0 . 9 7 5 , } \\ & { \qquad 0 . 9 8 , 0 . 9 8 5 , 0 . 9 9 , 0 . 9 9 5 , 0 . 9 9 9 \} , } \end{array}
$$

denser above 0.9, where the result changes fastest when there is no hysteresis, and five values of γ: 0, 0.5, 0.7, 0.8 and 0.9, so in total there are 130 candidates. All of them were run on the seven validation scenes, with both mask sources, and we applied the following rule to the results with the masks derived from the dataset.

The idea behind the rule is to prefer a stable candidate over a slightly better but less stable one. Among the candidates whose mean mIoU over the validation scenes is within 0.01 of the best mean, it takes the one with the smallest standard deviation between scenes:

$$
\begin{array} { r l } & { m ^ { * } = \underset { c } { \mathrm { m a x ~ } } \mathrm { m e a n } _ { s } \big ( \mathrm { m I o U } _ { s , c } \big ) , } \\ & { \quad \mathcal { E } = \big \{ c : \mathrm { m e a n } _ { s } \big ( \mathrm { m I o U } _ { s , c } \big ) \geq m ^ { * } - 0 . 0 1 \big \} , } \\ & { \quad \hat { c } = \underset { c \in \mathcal { E } } { \mathrm { a r g ~ m i n ~ } } \mathrm { s t d } _ { s } \big ( \mathrm { m I o U } _ { s , c } \big ) , } \end{array}\tag{22}
$$

where c goes over the candidates and s over the seven scenes. We break possible ties using the radius vote, then the smaller $\beta$ and finally the smaller γ. The rule chose the nearest Gaussian with $( \beta , \gamma ) = ( 0 . 7 , 0 . 8 )$ . We also apply the same rule to the radius vote alone, which gives (0.9, 0.7), and we report this point on the test scenes too, as this allows us to see the efect of the transfer operator on its own.

## 6.3 Baseline

In order to see the efect of the fraction, we compare the method with the version that we used before it. That version accumulated the same target evidence $E _ { i } ^ { + }$ and selected a Gaussian when

$$
E _ { i } ^ { + } \geq \beta | \mathcal { V } ^ { k } | ,\tag{23}
$$

that is, it thresholded the target evidence per view where the class appears. Note that we keep the name $\beta$ because it plays the same role, since it sets the threshold, but it is not the same quantity as in the method. The fraction of Equation (14) lies in [0, 1], while the evidence per view is a sum of weighted confidences over many pixels, so this $\beta$ does not have to be between zero and one. Everything else is the same as in the method: the votes, the hysteresis with the same values of $\gamma ,$ the nearest Gaussian for the transfer and the value of τ. The threshold has no natural scale here, so its grid has thirteen values between 0.003 and 1, around the values that this version used, and the rule of Equation (22) chooses its pair $( \beta , \gamma )$ on the same validation scenes. In this way, the comparison only changes the score that the threshold cuts.

## 6.4 Sensitivity ablations

The opacity weighting and the background competition of the transfer only exist in the radius vote, so the ablations change one factor at a time around the radius vote at its own selected point, (0.9, 0.7), on the validation scenes and with the masks derived from the dataset. In particular, we tested the following factors:

• disabling the hysteresis, that is, setting γ = 0.

• disabling the background competition in the transfer, in each direction separately and in both at the same time.

• disabling the opacity weighting.

• using all the matched cameras instead of only the ones that contain the class, as in Equation (12).

Every ablation uses the same scenes and classes as its baseline, so we compare them scene by scene.

## 6.5 Detector masks against the 3D result

In order to know what the fusion of views adds to the detector, we also measure, for every class and scene, the pixel IoU, precision and recall of the YOLO masks against the annotation masks over all the views, and we compare them with the 3D IoU that the method reaches from the same YOLO masks.

## 7 Results

## 7.1 Results at the selected point

Table 2 shows the results at the point selected on validation. In general, the results follow the same pattern in both datasets, although they are lower on the real scans of ScanNet++. On the ten held-out ScanNet++ scenes, the mean mIoU is 0.80 with the annotation masks, with a 95% confidence interval from 0.77 to 0.83, and 0.54 with YOLO masks. In other words, a single operating point, chosen on synthetic rooms, is still valid for the real scans of another dataset without retuning anything. With the annotation masks, precision and recall stay close to each other, so the prediction does not go beyond the objects, and it does not leave out large parts of them either.

## 7.2 Two test scenes

Figure 5 shows, as an example, two test scenes whose mIoU is close to the median. With the annotation masks, most of the errors appear as thin bands at the borders of the objects, where the mesh and the Gaussians do not agree on the position of the surface. With YOLO masks, on the other hand, the errors depend mostly on the detector. The round table of the first scene looks like a COCO dining table and is well labelled, but the cofee table next to the sofa and the desk of the second scene are blue, since YOLO does not propose them. The red Gaussians on the round table are objects that lie on it and enter the mask of the detector.

## 7.3 Against the evidence per view

Table 3 compares the method with the baseline of Section 6.3, and the fraction is better in every configuration. On the test scenes, the diference is +0.24 with the annotation masks, with an interval from +0.20 to +0.28, and the method is better in all 10 scenes. With YOLO masks the diference is +0.12, from +0.06 to +0.17.

Figure 6 shows the diference between both methods, as well as the stability that the hysteresis brings. With hysteresis, the fraction gives almost the same result along its whole grid, while the evidence per view has a narrow peak and goes from 0.34 to 0.64. The best threshold of the baseline also changes with the class, from 0.07 to 0.2. The reason why this happens is that the evidence still grows with the size of the object in the image, with its distance to the camera and with the occlusions. In addition, a background Gaussian that many views see collects target evidence at the borders of the masks, so the baseline selects too much: on the test scenes its precision is 0.64 for a recall of 0.83. The fraction, instead, compares the target evidence with all the evidence that reached the Gaussian, so these factors cancel.

## 7.4 Where the error comes from

Table 4 splits the error following Section 5.5. The detector is the largest source of error in both datasets, with 0.26 on ScanNet++, while the lifting loses 0.11. Therefore, most of what can still be gained is in the 2D masks, and with the annotation masks the method already reaches 0.876 of the reference on ScanNet++.

Figure 7 shows that the detector gap is not spread evenly over the classes. The clearest case is the one of the table class. On ScanNet++, its IoU is 0.72 with the annotation masks and 0.11 with YOLO, which gives it zero in 5 of the 9 scenes. The detector was trained on COCO, whose only table is the dining table, while most tables of ScanNet++ are desks, so YOLO recovers only 0.09 of the table pixels and there is almost nothing to lift. As could be expected, the representation term is larger on ScanNet++, and the clock is the worst case, with a reference of 0.81, since it is a small object covered by few Gaussians in real scans.

## 7.5 What the fusion of views adds to the detector

Figure 8 compares, for every class and scene, the YOLO masks in 2D with the 3D result that the method obtains from them. The 3D IoU is higher in 20 of the 23 pairs of Replica and in 27 of the 39 pairs of ScanNet++, and on ScanNet++ the mean goes from 0.45 to 0.55. In 2D, YOLO is relatively precise but it misses a large part of the objects, with a pixel precision of 0.70 and a recall of 0.51 on ScanNet++. When it misses an object in some views, the

Table 2: Results at the selected point. The mIoU is the mean over scenes with its standard deviation, and the 95% confidence interval comes from the bootstrap of Section 5.4.
<table><tr><td>Dataset</td><td>Masks</td><td>Scenes</td><td>mIoU</td><td>95% CI</td><td>Precision</td><td>Recall</td><td>Reference</td><td> $\mathrm { m I o U _ { r e l } }$ </td></tr><tr><td rowspan="2">Replica</td><td>annotation</td><td>7</td><td> $0 . 9 3 \pm 0 . 0 1$ </td><td>[0.92, 0.94]</td><td>0.96</td><td>0.97</td><td>0.97</td><td>0.956</td></tr><tr><td>YOLO</td><td>7</td><td> $0 . 6 5 \pm 0 . 1 3$ </td><td>[0.55, 0.74]</td><td>0.72</td><td>0.79</td><td>0.97</td><td>0.668</td></tr><tr><td rowspan="2">ScanNet++</td><td>annotation</td><td>10</td><td> $0 . 8 0 \pm 0 . 0 5$ </td><td>[0.77, 0.83]</td><td>0.86</td><td>0.92</td><td>0.91</td><td>0.876</td></tr><tr><td>YOLO</td><td>10</td><td> $0 . 5 4 \pm 0 . 0 9$ </td><td>[0.48, 0.60]</td><td>0.64</td><td>0.71</td><td>0.91</td><td>0.582</td></tr></table>

![](images/a5350e0ff6c5a68e6c1f9c81e9292adc4ad0de2b3c9d583a022e4a22bd167ee9.jpg)  
Figure 5: Prediction against reference in two ScanNet++ test scenes close to the median, over all their evaluated classes. The mIoU is the one of the whole scene, measured on the mesh.

Table 3: mIoU of the baseline and of the method, each one at the point that the same rule chose for it.
<table><tr><td></td><td></td><td colspan="2">Replica</td><td colspan="2">ScanNet++</td></tr><tr><td>Thresholded score</td><td>(β, γ)</td><td>annotation</td><td>YOLO</td><td>annotation</td><td>YOLO</td></tr><tr><td>Target evidence per view</td><td>(0.1, 0.5)</td><td>0.64 ±0.04</td><td>0.44 ±0.07</td><td>0.56 ±0.05</td><td>0.42 ±0.04</td></tr><tr><td>Target evidence fraction</td><td>(0.7, 0.8)</td><td>0.93 ±0.01</td><td>0.65 ±0.13</td><td>0.80 ±0.05</td><td>0.54 ±0.09</td></tr></table>

Table 4: The three terms of Equation (21) at the selected point.
<table><tr><td>Dataset</td><td>gdet</td><td>glift</td><td>grep</td></tr><tr><td>Replica</td><td>0.28</td><td>0.04</td><td>0.03</td></tr><tr><td>ScanNet++</td><td>0.26</td><td>0.11</td><td>0.09</td></tr></table>

![](images/76d3b2b63617977b628a07ce1d69826019230e61d41f1237874fc9e26f54234b.jpg)  
Figure 6: Mean validation mIoU along the threshold grid of each score, with the annotation masks. The dotted line is the selected threshold.

Gaussians of the object still collect target evidence from the views where it finds it, and a view without the class does not contribute at all. An example of this can be seen in Figure 1, with the tables of a classroom. Nevertheless, the desks, which YOLO almost never proposes, stay low in 3D as well.

![](images/b7e3dfa71c0cf3a3b7fc69dc84e0a1b250f535780ec67e00f76c35cbdc041140.jpg)  
Figure 7: IoU of each class with each mask source and for the reference, averaged over the scenes that contain the class.

![](images/d94bcd47e45682d6a14aa006db38d245507d5e77f21ec6ac6f6421f679d29f01.jpg)  
Figure 8: Pixel IoU of the YOLO masks against the 3D IoU that the method reaches from them, one marker per class and scene. Above the diagonal, the 3D result is better than the masks.

![](images/3dab7a077e21bf2d77bdcd3be5eca7b472589b4dc18af7e3c57f3dfd0aedd151.jpg)  
Figure 9: Mean validation mIoU along $\beta$ for every $\gamma ,$ with the annotation masks. The grey band holds the candidates within 0.01 of the best mean.

## 7.6 The operating point

Figure 9 shows the results of the whole validation grid. Of the 130 candidates, 18 were within 0.01 of the best mean, 0.94, and all of them use the nearest Gaussian. Among them, the rule chose (0.7, 0.8), with a mean of 0.93 and a standard deviation between scenes of 0.01.

The figure also allows us to see the efect of the hysteresis. Without it, in the dashed curves, the mean goes down from 0.93 to 0.38 as $\beta$ grows. With $\gamma = 0 . 8$ , it stays between 0.90 and 0.93 for every $\beta$ of the grid. So, the hysteresis does not raise the best result, but it makes the result almost independent of $\beta \colon$ a high $\beta$ keeps only the clearest Gaussians as seeds, and the hysteresis recovers the rest of the object through the Gaussians connected to them. The same efect is shown class by class in Figure 10 of the appendix.

## 7.7 The transfer operator

Table 5 compares the two operators, each one at its own selected point. On validation, the nearest Gaussian is better by 0.03 with the annotation masks, and its dispersion between scenes is a third of the one of the radius vote. On the test scenes, with the annotation masks, it is better by 0.01. With YOLO masks, however, the radius vote is slightly better: the diference is -0.01, with an interval from -0.03 to +0.00.

This is due to the minimum vote fraction of the radius vote. With $\theta = 0 . 3 ,$ , the class wins a vertex with less than half of the vote, so the labelled region grows up to τ around the selected Gaussians. On Replica, the clock is a good example: with the radius vote, its precision with the annotation masks is 0.64 for a recall of 1.00. Taking into account that the YOLO masks leave part of the objects out, this growth recovers some recall, and this explains the small advantage of the radius vote in that condition. The nearest Gaussian does not grow anything, so the labelled surface simply follows the lifting.

Table 5: Each transfer operator at the point that the rule chooses for it.
<table><tr><td></td><td></td><td colspan="3">Replica, validation</td><td colspan="3">ScanNet++, test</td></tr><tr><td>Operator</td><td> $( \beta , \gamma )$ </td><td>annotation</td><td></td><td>YOLO reference</td><td>annotation</td><td>YOLO</td><td>reference</td></tr><tr><td>Nearest Gaussian</td><td>(0.7, 0.8)</td><td> $0 . 9 3 \pm 0 . 0 1$ </td><td> $0 . 6 5 \pm 0 . 1 3$ </td><td>0.97</td><td> $0 . 8 0 \pm 0 . 0 5$ </td><td> $0 . 5 4 \pm 0 . 0 9$ </td><td>0.91</td></tr><tr><td>Radius vote</td><td>(0.9, 0.7)</td><td> $0 . 9 0 \pm 0 . 0 4$ </td><td> $0 . 6 3 \pm 0 . 1 2$ </td><td>0.91</td><td> $0 . 7 9 \pm 0 . 0 5$ </td><td> $0 . 5 5 \pm 0 . 0 7$ </td><td>0.84</td></tr></table>

Table 6: Ablations on the validation scenes with the annotation masks. Every row changes one factor of the radius vote at its selected point, and the diference is paired scene by scene.
<table><tr><td>Configuration</td><td>mIoU</td><td>Difference</td><td>Reference</td></tr><tr><td>Radius vote at its selected point</td><td>0.90 ±0.04</td><td></td><td>0.91</td></tr><tr><td>Hysteresis disabled</td><td>0.91 ±0.04</td><td>+0.008</td><td>0.91</td></tr><tr><td>No competition, Gaussian to mesh</td><td>0.75 ±0.05</td><td>-0.153</td><td>0.76</td></tr><tr><td>No competition, mesh to Gaussian</td><td>0.90 ±0.04</td><td>+0.000</td><td>0.76</td></tr><tr><td>No competition, both directions</td><td>0.75 ±0.05</td><td>-0.153</td><td>0.63</td></tr><tr><td>Opacity weighting disabled</td><td>0.90 ±0.03</td><td>+0.002</td><td>0.91</td></tr><tr><td>Every matched camera contributes</td><td>0.90 ±0.04</td><td>+0.001</td><td>0.91</td></tr></table>

## 7.8 Sensitivity ablations

Table 6 shows the results of the ablations. The background competition from Gaussians to mesh is the only factor that has an important efect on the result. Without it, every vertex within $\tau$ of a selected Gaussian takes the class, and the mIoU changes by -0.15. Without competition from mesh to Gaussians, the prediction does not change and only the reference goes down, to 0.76, since then the Gaussians of other surfaces keep the label of any class vertex within τ.

The rest of the factors hardly change the result. Without opacity weighting and with every matched camera, the diference stays below 0.01. We expected the cameras without the class to dilute the fraction, but a camera that sees an object also has it in its mask, so these extra cameras add non-target evidence mostly to Gaussians that do not belong to the class, which were not going to be selected in any case. Finally, disabling the hysteresis at $\beta = 0 . 9$ gives +0.01, which agrees with Section 7.6: at a moderate $\beta$ the hysteresis is not needed, and the reason why it is useful is that the result does not depend on this choice.

Table 7: Cost of each stage for a whole test scene over the 8 scenes that ran every stage from scratch. The time is the median and the memory the highest peak. The last row runs the thirteen values of $\beta$ from the cached votes.
<table><tr><td>Stage</td><td>Time</td><td>Peak memory</td></tr><tr><td>Dataset preparation</td><td>19.5 s</td><td></td></tr><tr><td>Mask generation</td><td>103 s</td><td>0.5 GB</td></tr><tr><td>Model training</td><td>617 s</td><td>1.9 GB</td></tr><tr><td>Vote accumulation</td><td>1359 s</td><td>0.8 GB</td></tr><tr><td>Threshold and hysteresis</td><td>6.4 s</td><td></td></tr><tr><td>Mesh transfer and metrics</td><td>257 s</td><td></td></tr><tr><td>Sweep over  $\beta ,$  cached</td><td>272 s</td><td></td></tr></table>

## 7.9 Cost

The last aspect that we analyse is the computational cost of the method. Table 7 shows the cost of a test scene. The median total time of a scene is 2463 s, and the vote accumulation is the slowest stage, with 1359 s, more than the training of the model itself. This is due to the fact that the visibility is computed in PyTorch. The cache is what makes the sweeps possible: once the votes are stored, the thirteen values of $\beta$ take 272 s, and no candidate projects the Gaussians again. All the experiments ran on NVIDIA A100-SXM4-40GB GPUs of the Picasso supercomputer. The results of every class and every scene can be found in Appendix A.

## 8 Limitations

As with any method, ours has some limitations, and two of them concern the method itself. First, if a class never appears in any mask, it receives no evidence and the method cannot recover it. In the same way, if a surface is covered by very few Gaussians, the reconstruction noise can make the non-target evidence dominate, and the method will not recover the target class there.

Second, the method cannot label several target classes at the same time. The same Gaussian can be selected for two classes in two separate runs, but we do not provide any procedure for the case where each Gaussian needs a single exclusive label. The cost of a scene grows with the number of classes, and the vocabulary is limited by what the detector proposes and by what the dataset annotates.

The correspondence between the dataset names and the detector names was written by hand, since each dataset has only six or seven classes. In a future version, learned embeddings could propose or check these correspondences, but they would add a source of variation that is not part of this study.

The ground-truth transfer reference is obtained with the operator of Section 3.7 and changes with it, as Section 7.7 shows for the two operators we tried. Therefore, the reference measures the cost of the representation and the transfer for one specific operator, and not for the best possible transfer.

Replica and ScanNet++ difer in many aspects apart from their scenes. Their vertices are annotated in different ways, and the visibility criteria and the origin of the camera poses are also diferent. The class sets are not the same either, with six and seven classes and only four of them in common. We subsample Replica by a factor of five, while in ScanNet++ we use every camera, and the Gaussian models of each dataset are trained at diferent image scales. In Replica the images are renders of the annotated mesh, so the masks, the Gaussians and the evaluated vertices share the same reference frame. In ScanNet++ the derived masks also agree with the mesh, but they are related to the photographs through an estimated alignment. The error of this alignment is present in every condition we measure there, and this is why we choose the operating point on Replica. The change from validation to test includes all of these diferences at once, and our design only isolates the gap due to the mask source.

In Replica, a vertex is left unannotated when its faces disagree beyond the fraction of Section 5.1. Because of this, part of the boundary between objects is not evaluated in that dataset.

Our visibility module is written in plain PyTorch instead of custom CUDA kernels, which makes the method slower than the original rasteriser. In addition, a tile with many Gaussians allocates an intermediate tensor of size $N _ { \mathrm { G a u s s i a n s } } \times N _ { \mathrm { p i x e l s } }$ , which can cause out-of-memory errors.

Finally, we only compare the method with its own conditions and its own reference, and we did not run any other published method on these scenes. Therefore, the numbers of Section 7 do not show how the method compares with published baselines.

## 9 Conclusion

In this work, we have presented a method that labels a Gaussian scene that is already trained, one target class at a time. Target and non-target evidence are accumulated at the same time, and each Gaussian receives a fraction of target evidence. A fixed threshold β selects the seeds, and the spatial hysteresis then adds every Gaussian above $\gamma \beta$ that belongs to a connected component with at least one seed. For the evaluation, the labels go to the mesh through the nearest Gaussian.

The transfer parameters were fixed on the development scenes, the seven Replica validation scenes were used to choose the transfer operator and a single pair $( \beta , \gamma ) =$ (0.7, 0.8), and the result was applied to the ten ScanNet++ test scenes without retuning anything. On the validation split, the mean mIoU was 0.93 with the annotation masks and 0.65 with YOLO masks, while on the held-out scenes it was 0.80 and 0.54.

From the analysis of the error, we can draw three main conclusions. First, the detector is the main source of error, with a gap of 0.26 on the test scenes, and most of it comes from classes that its vocabulary does not cover well, such as the desks of ScanNet++. Second, the evidence fraction is what makes a single threshold possible. Compared with the evidence per view of a previous version, it improves the test mIoU by 0.24 with the annotation masks and by 0.12 with YOLO masks, and its threshold moves from Replica to ScanNet++ almost without loss. Third, the hysteresis does not raise the best result, but it makes the result almost independent of $\beta ,$ which matters when the same value has to work in another dataset.

Regarding future work, several lines follow from the limitations above. In the first place, a comparison with other published methods on these scenes would allow us to put our results in context. In addition, the results suggest improving the 2D masks before anything else, for example with an open-vocabulary detector, as well as measuring the coverage of the representation before transferring labels.

## Author Contributions

The contribution of each author follows the CRediT roles, where I.V.G. is Iván Verdugo Guerra, E.L.R. is Ezequiel López Rubio and J.G.G. is Jorge García González.

• Conceptualization: I.V.G., E.L.R., J.G.G.

• Methodology: I.V.G., E.L.R., J.G.G.

• Software: I.V.G.

• Validation: I.V.G.

• Formal analysis: I.V.G.

• Investigation: I.V.G.

• Data curation: I.V.G.

• Writing – original draft: I.V.G.

• Writing – review & editing: I.V.G., E.L.R., J.G.G.

• Visualization: I.V.G.

• Supervision: E.L.R., J.G.G.

• Project administration: E.L.R., J.G.G.

• Resources: E.L.R., J.G.G.

• Funding acquisition: E.L.R., J.G.G.

## Acknowledgements

The experiments of this work were run on the Picasso supercomputer. For this reason, the authors thankfully acknowledge the computer resources, technical expertise and assistance provided by the SCBI, the Supercomputing and Bioinformatics center of the University of Malaga.

## Declaration on the use of AI tools

The main parts of the work were carried out by the authors, namely the method formulation and implementation, including the evidence fraction and thresholding, designing the evaluation protocol and the error decomposition, as well as the execution of the experiments and the analysis of the results.

Large language models were used in some specific tasks. First, Claude Opus 5 helped to decide how everything would be organised in diferent sections. The writing of the manuscript was then carried out by the first author, and afterwards, Claude Opus 5.5 was used to help fix English inconsistencies, while the work was revised by all the authors, who take full responsibility for the content of this paper. Finally, DeepSeek V4.1 Flash helped to write the code of the folder evaluation/scripts, which generates the figures and tables of the preprint, and Gemini 3.1 Pro helped with the code that adapts Replica and ScanNet++ to the pipeline, in the folders evaluation/replica and evaluation/scannetpp.

We are in favour of a responsible use of AI tools in research, in which their use is declared openly and the authors remain responsible for every result.

## References

[1] John Canny. A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6):679–698, 1986. https: //doi.org/10.1109/TPAMI.1986.4767851.

[2] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3D Gaussians. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1971–1979, 2025. https://doi.org/10.1609/aaai.v39i2.32193.

[3] Facebook Research. Replica dataset. https://github. com/facebookresearch/replica-dataset, 2019.

[4] Pedro F. Felzenszwalb and Daniel P. Huttenlocher. Efficient graph-based image segmentation. International Journal of Computer Vision, 59(2):167–181, 2004. https: //doi.org/10.1023/B:VISI.0000022288.19776.77.

[5] Kyle Genova, Xiaoqi Yin, Abhijit Kundu, Caroline Pantofaru, Forrester Cole, Avneesh Sud, Brian Brewington, Brian Shucker, and Thomas Funkhouser. Learning 3D semantic segmentation with only 2D image supervision. In Proceedings of the International Conference on 3D Vision, pages 361–372, 2021. https://doi.org/10.1109/ 3DV53792.2021.00046.

[6] Yang He, Wei-Chen Chiu, Margret Keuper, and Mario Fritz. STD2P: RGBD semantic segmentation using spatiotemporal data-driven pooling. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4837–4846, 2017. https://doi.org/10.1109/CVPR. 2017.757.

[7] Alexander Hermans, Georgios Floros, and Bastian Leibe. Dense 3D semantic mapping of indoor scenes from RGB-D images. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 2631–2638, 2014. https://doi.org/10.1109/ICRA.2014.6907236.

[8] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian Splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, 2024. https://doi.org/10.1145/ 3641519.3657428.

[9] Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, and Muhammet Esat Kalfaoglu. Ultralytics YOLO26: Unified real-time end-to-end vision models, 2026. arXiv:2606.03748.

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139:1–139:14, 2023. https://doi.org/10. 1145/3592433.

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian Splatting: Official reference implementation. https://github.com/ graphdeco-inria/gaussian-splatting, 2023.

[12] Justin Kerr, Chung Min Kim, Ken Goldberg, Matthew Tancik, and Angjoo Kanazawa. LERF: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[13] Philipp Krähenbühl and Vladlen Koltun. Eficient inference in fully connected CRFs with Gaussian edge potentials. In Advances in Neural Information Processing Systems, pages 109–117, 2011.

[14] Abhijit Kundu, Xiaoqi Yin, Alireza Fathi, David Ross, Brian Brewington, Thomas Funkhouser, and Caroline Pantofaru. Virtual multi-view fusion for 3D semantic segmentation. In Proceedings of the European Conference on Computer Vision, pages 518–535, 2020. https://doi. org/10.1007/978-3-030-58586-0\_31.

[15] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance diferentiable rendering. ACM Transactions on Graphics, 39(6):194:1–194:14, 2020. https: //doi.org/10.1145/3414685.3417861.

[16] Juliette Marrie, Romain Menegaux, Michael Arbel, Diane Larlus, and Julien Mairal. LUDVIG: Learning-free uplifting of 2D visual features to Gaussian Splatting scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7440–7450, 2025.

[17] Nelson Max. Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics, 1(2):99–108, 1995. https://doi.org/10.1109/ 2945.468400.

[18] John McCormac, Ankur Handa, Andrew J. Davison, and Stefan Leutenegger. SemanticFusion: Dense 3D semantic mapping with convolutional neural networks. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 4628–4635, 2017. https://doi.org/10.1109/ICRA.2017.7989538.

[19] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pages 8024–8035, 2019.

[20] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3D language Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051– 20060, 2024.

[21] Johannes L. Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4104–4113, 2016. https://doi.org/10. 1109/CVPR.2016.445.

[22] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flash-Splat: 2D to 3D Gaussian Splatting segmentation solved optimally. In Computer Vision – ECCV 2024, volume 15080 of Lecture Notes in Computer Science. Springer, 2024. https://doi.org/10.1007/978-3-031-72670-5\_ 26.

[23] Yixiao Song, Qingyong Li, Wen Wang, and Zhicheng Yan. PointGS: Semantic-consistent unsupervised 3D point cloud segmentation with 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. arXiv:2605.11520.

[24] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, et al. The Replica dataset: A digital replica of indoor spaces, 2019. arXiv:1906.05797.

[25] Wentao Sun, Yiping Chen, Zhengsen Xu, Jonathan Li, and John S. Zelek. CDSeg: A renderable Gaussian carrier for image-to-3D label transfer, 2026. arXiv:2608.05482.

[26] Ultralytics. Ultralytics YOLO. https://github.com/ ultralytics/ultralytics, 2026.

[27] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Jian Zhang. OpenGaussian: Towards point-level 3D Gaussian-based open vocabulary understanding. In Advances in Neural Information Processing Systems, 2024.

[28] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3D scenes. In Proceedings of the European Conference on Computer Vision, 2024.

[29] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3D indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12– 22, 2023.

[30] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++ toolkit. https://github. com/scannetpp/scannetpp, 2023.

[31] Yang Zhang, Zixiang Zhou, Philip David, Xiangyu Yue, Zerong Xi, Boqing Gong, and Hassan Foroosh. Polar-Net: An improved grid representation for online Li-DAR point clouds semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9601–9610, 2020. https://doi.org/10.1109/CVPR42600.2020.00962.

[32] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J. Davison. In-place scene labelling and understanding with implicit scene representation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

[33] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J. Davison. Semantic-NeRF. https://github. com/Harry-Zhi/semantic\_nerf, 2021.

[34] Zixiang Zhou, Yang Zhang, and Hassan Foroosh. Panoptic-PolarNet: Proposal-free LiDAR point cloud panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13194–13203, 2021. https://doi.org/10.1109/ CVPR46437.2021.01299.

[35] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus Gross. EWA Splatting. IEEE Transactions on Visualization and Computer Graphics, 8(3):223–238, 2002. https://doi.org/10.1109/TVCG.2002.1021576.

## A Results by class and by scene

Table 8 gives the IoU of each class at the selected point, and Table 9 gives the mIoU of every scene with both

![](images/302c16ede57ca4d721bb3fdd4c34be00f908c2c32b502508c8c36dc108d40497.jpg)  
Figure 10: IoU of each class on the validation scenes along β, with the annotation masks, with and without hysteresis. The dotted line is $\beta = 0 . 7$

operators. A class only contributes to the mean in the scenes that contain it, so the last column of Table 8 shows how much weight each row has.

Table 8: IoU per class at the selected point, averaged over the scenes that contain the class.
<table><tr><td>Dataset</td><td>Class</td><td>Annotation</td><td>YOLO</td><td>Reference</td><td>Scenes</td></tr><tr><td>Replica</td><td>chair</td><td>0.91</td><td>0.76</td><td>0.96</td><td>6</td></tr><tr><td></td><td>sofa</td><td>0.93</td><td>0.71</td><td>0.97</td><td>3</td></tr><tr><td></td><td>table</td><td>0.94</td><td>0.34</td><td>0.98</td><td>6</td></tr><tr><td></td><td>tv</td><td>0.95</td><td>0.92</td><td>0.98</td><td>2</td></tr><tr><td></td><td>plant</td><td>0.95</td><td>0.51</td><td>0.99</td><td>3</td></tr><tr><td>ScanNet++</td><td>clock</td><td>0.93</td><td>0.92</td><td>0.96</td><td>3</td></tr><tr><td></td><td>chair</td><td>0.86</td><td>0.64</td><td>0.95</td><td>10</td></tr><tr><td></td><td>table</td><td>0.72</td><td>0.11</td><td>0.87</td><td>9</td></tr><tr><td></td><td>tv</td><td>0.87</td><td>0.78</td><td>0.95</td><td>8</td></tr><tr><td></td><td>laptop</td><td>0.83</td><td>0.75</td><td>0.93</td><td>3</td></tr><tr><td></td><td>sink</td><td>0.73</td><td>0.57</td><td>0.91</td><td>7</td></tr><tr><td></td><td>clock</td><td>0.85</td><td>0.84</td><td>0.81</td><td>2</td></tr></table>

## B Development sweep

Figure 11 shows the results of the two sweeps of Section 6.1. A small radius leaves many vertices without any Gaussian, and both the prediction and the reference fall quickly below $\tau = 0 . 0 5$ m. Above 0.08 m the prediction is almost flat. When θ goes down, the prediction improves on ScanNet++, while the reference gets worse, since the class takes vertices where it only has part of the vote. This is the growth that Section 7.7 discusses.

Table 9: mIoU of every scene for each operator at its own selected point.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Scene</td><td colspan="3">Nearest Gaussian, (0.7,0.8)</td><td colspan="3">Radius vote, (0.9, 0.7)</td></tr><tr><td>annotation</td><td>YOLO</td><td>reference</td><td>annotation</td><td>YOLO</td><td>reference</td></tr><tr><td>Replica</td><td>office_1</td><td>0.95</td><td>0.44</td><td>0.98</td><td>0.90</td><td>0.44</td><td>0.91</td></tr><tr><td></td><td>office_2</td><td>0.94</td><td>0.78</td><td>0.97</td><td>0.89</td><td>0.74</td><td>0.88</td></tr><tr><td></td><td>office_3</td><td>0.91</td><td>0.65</td><td>0.96</td><td>0.85</td><td>0.59</td><td>0.85</td></tr><tr><td></td><td> $\mathsf { o f f i c e \_ 4 }$ </td><td>0.93</td><td>0.64</td><td>0.97</td><td>0.86</td><td>0.57</td><td>0.87</td></tr><tr><td></td><td> $\mathtt { r o o m \_ 0 }$ </td><td>0.92</td><td>0.52</td><td>0.98</td><td>0.95</td><td>0.54</td><td>0.96</td></tr><tr><td></td><td> $\tt r o o m _ { - } 1$ </td><td>0.93</td><td>0.84</td><td>0.99</td><td>0.93</td><td>0.84</td><td>0.96</td></tr><tr><td></td><td> $\tt r o o m _ { - } 2$ </td><td>0.93</td><td>0.68</td><td>0.97</td><td>0.93</td><td>0.67</td><td>0.94</td></tr><tr><td>ScanNet++</td><td> $0 9 \mathsf { c } 1 4 1 4 \mathsf { f } 1 \mathsf { b }$ </td><td>0.71</td><td>0.46</td><td>0.91</td><td>0.71</td><td>0.48</td><td>0.85</td></tr><tr><td></td><td>0d2ee665be</td><td>0.77</td><td>0.47</td><td>0.88</td><td>0.74</td><td>0.50</td><td>0.76</td></tr><tr><td></td><td>21d970d8de</td><td>0.90</td><td>0.68</td><td>0.93</td><td>0.89</td><td>0.66</td><td>0.89</td></tr><tr><td></td><td>25f3b7a318</td><td>0.83</td><td>0.54</td><td>0.94</td><td>0.82</td><td>0.53</td><td>0.86</td></tr><tr><td></td><td>27dd4da69e</td><td>0.76</td><td>0.39</td><td>0.90</td><td>0.76</td><td>0.46</td><td>0.79</td></tr><tr><td></td><td>3db0a1c8f3</td><td>0.79</td><td>0.53</td><td>0.91</td><td>0.76</td><td>0.57</td><td>0.85</td></tr><tr><td></td><td>3f15a9266d</td><td>0.81</td><td>0.63</td><td>0.88</td><td>0.78</td><td>0.59</td><td>0.80</td></tr><tr><td></td><td>5942004064</td><td>0.82</td><td>0.67</td><td>0.93</td><td>0.81</td><td>0.69</td><td>0.87</td></tr><tr><td></td><td>5eb31827b7</td><td>0.87</td><td>0.55</td><td>0.94</td><td>0.86</td><td>0.56</td><td>0.88</td></tr><tr><td></td><td>6115eddb86</td><td>0.76</td><td>0.48</td><td>0.92</td><td>0.78</td><td>0.49</td><td>0.87</td></tr></table>

![](images/6e7879d4d8a5a8f2f508123925c85e8918d06365cb61ce470abcc7f7fc50ba1d.jpg)

![](images/3c5c1e494bbc9563542869d936e3dd26fb492d03b5967464ce180bb605815123.jpg)  
Figure 11: Development sweep of the radius vote with the annotation masks. Solid lines are the prediction and dashed lines the reference.