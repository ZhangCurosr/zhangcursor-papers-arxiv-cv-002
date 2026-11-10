# Towards Reconstructing and Recognizing Objects behind Fragmented Occlusion using Video

Roman Pflugfelder Austrian Institute of Technology Digital Safety & Security Department Technical University of Munich School of Computation, Information and Technology Munich Center for Machine Learning (MCML) roman.pflugfelder@{ait.ac.at|tum.de}

Michael Lindenbaum Technion Faculty of Computer Science mic@cs.technion.ac.il

Daniel Cremers Technical University of Munich School of Computation, Information and Technology Munich Center for Machine Learning (MCML)

cremers@tum.de

## Abstract

Perceiving objects through fragmented occlusions, such as foliage, is a natural capability of the human visual system but remains a fundamental challenge for computer vision. This work addresses the problem of object reconstruction, or de-occlusion, in the presence ofhigh-density,fragmented occlusions. We propose a reconstruction framework that operates on a sequence of grayscale images of a moving object, acquired by a single camera, either stationary or in motion. The algorithm estimates the object’s relative positions across frames by jointly registering the images and optimizing the alignment parameters to maximize the contrast of their average, referred to as the integrated image. This integrated image serves as the reconstructed representation of the object. Experimental evaluations on both synthetic and real video sequences demonstrate the feasibility of accurate reconstruction under occlusion densities exceeding 90 %. The proposed approach is competitive with synthetic aperture techniques, which are limited to static objects and require complex camera setup with careful calibration. Furthermore, we show that the integrated image enables reliable object classification under severe occlusion when used as input to standard, off-the-shelfclassifiers.

![](images/254dbe9d2ba45ae853b1ca32f3152193ef1ab2e05acba4675bbbd36ec895862c.jpg)  
Figure 1. Reconstructing or de-occluding an object’s shape and appearance from a sequence of occluded images creates an integrated, non-occluded image of the moving object for some reference in the sequence (see the red framed image).

## 1. Introduction

Fragmented occlusion occurs when an object is partially visible, such as through foliage, exposing its surface at different shattered image regions. Even if the camera or object is moving, the object is never fully visible. In contrast to the human vision research community, we currently have very limited knowledge about vision under fragmented occlusion [7, 10, 39, 40]. It turns out that humans can effortlessly perceive such occluded objects. Despite some advancements, we still do not comprehend this remarkable human ability, and current recognition algorithms do not provide an explanation either [59]. Most algorithms are designed for the fully visible object. Even algorithms designed for recognition under partial occlusion [16, 23, 24, 37, 55, 56] would not work satisfactorily under fragmented occlusion, as they need sufficiently large visible regions to extract features. Although fragmented occlusion seems like a niche problem, the study of it will enhance our overall understanding of occlusion [45], which remains a challenge, still significantly impacting recent research [51].

As mentioned, psychologists have studied fragmented occlusion for a longer time. This area is known as dynamic perception [9, 34, 38]. They believe that our brain aggregates perceptual information over time and then reconstructs shape and appearance of such occluded objects. This paper is partially devoted to this belief. The proposed 2D reconstruction algorithm takes a sequence of occluded images and provides a generated, integrated image where the object becomes more visible (see Fig. 1). The motivation is that recognition algorithms are not applied to the single occluded images, but to this integrated image. Thus, in a sense, the paper provides both, a computational model for the human ability to perceive occluded objects and the reconstruction algorithm for making occluded objects more visible, which we call object de-occlusion.

Given the limited literature, a possible algorithm for object de-occlusion is to reconstruct the complete 3D scene with voxel coloring [44] and then to delete the occluder voxels and project back all remaining scene elements into a reference camera [49]. This approach is very sensitive to the unknown object’s depth, which causes severe visible holes in the reconstruction. Another way is to control the focal length and the aperture [11]. Such an algorithm is effective and may suit a human computational model, but it is very much limited to short distances in the micrometer range [32]. Several algorithms, denoted as the synthetic aperture approach, address the object de-occlusion by combining data from multiple cameras. Even for a moderate 70 % occlusion, many cameras are needed, where these are usually arranged in arrays [48, 49] or lines [21, 36]; e.g. 88 cameras are necessary in a foliage experiment [49]. Alternatively, a camera rig on a drone [25] or camera on a computer controlled gantry [49, 53, 54, 58] is used. Camera baselines are typically in the scale of the occluder size [26]. Such apparatuses use RGB cameras [53], thermal, event cameras [54, 58], or combinations [29, 31] and apply neural networks for better reconstruction [18, 53, 57], but the approach is generally limited by the need for static scenes and calibrated cameras, which are difficult to obtain in foliage.

This paper presents a further algorithm for object deocclusion which uses video. This approach assumes a single uncalibrated camera and exploits the visual parallax between the camera and the object in the images of the sequence. The approach also assumes that the object parts become visible as they move. Our algorithm is conceptually simple: The images are first warped, so that the occluded object becomes stationary, and then the registered images are averaged together to create a single image containing the more visible object, while other image regions become blurred. Despite occlusion, this is achieved using a parametric motion model [6] and by maximizing contrast within a window in the integrated image to robustly estimate the motion parameters. We test this robustness on real video and synthetic images, using a new simulation that mimics foliage. We found that the algorithm reconstructs objects when they are barely visible, with an occlusion density of as high as 92.3 %. We also combine this new algorithm for object de-occlusion with zero-shot object classification, resulting in a 68 % reduction in the classification error rate for such 90 % occluded images.

In summary, the contributions are as follows:

1. An algorithm that combines global registration, averaging, and contrast maximization within a window in the integrated image, demonstrating superior performance compared to the state-of-the-art for reconstructing occluded objects in both synthetic and real imagery

2. A fragmented occlusion simulation that is used for quantitatively evaluating the algorithm’s behavior

3. Object classification, robust under fragmented occlusion, and its successful evaluation

The paper is organized as follows: Sec. 2 discusses methods relevant for the proposed algorithm. Sec. 3 then describes the algorithm in detail. Sec. 4 presents the results of the experiments and discusses the findings, as well as describing the classification algorithm. Sec. 5 concludes the paper with some analogies from psychological research.

## 2. Related Work

Registering sequentially acquired data and integrating it into a single representation has been previously discussed in a few papers: some studies of motion computation with two consecutive images [20] and with event data [13, 14], for the application of de-fencing [35], for the synthetic aperture approach with a thermal camera on a flying drone [27], and for the inverse synthetic aperture approach using radar [50] or airborne thermal images [3]. Our algorithm differs from these papers by using a windowed contrast function to determine motion from the integrated image.

Joint 2D reconstruction and motion computation are already considered [22, 33, 43, 47]. These algorithms either alternate between optical flow estimation and reconstruction [33, 43], or solve both tasks simultaneously using a neural network [47]. The proposed algorithm, while potentially linked to these papers, focuses on fragmental occlusion. These earlier works fail to handle fragmental occlusion because they rely on optical flow, which is undefined for occluded pixels [4], or because of fundamental limitations of convolutional neural networks with occlusion [1].

The work most similar to this paper is described in [3]. Their reconstruction relies on the infrared domain and fails with visible images. Our approach differs from this paper in two ways: we use selective windowing on the occluded object to enhance contrast measurement, and we use a coherent motion model. The proposed algorithm can now be applied to visible images and successfully de-occludes the reference image. In [3], the authors manually set the motion parameters. We also introduce a new fragmental occlusion simulation to evaluate the algorithm. Finally, we demonstrate that classification under dense occlusion is feasible by feeding the integrated image into a standard classifier.

## 3. Method

Consider a scene of relative motion between a camera and an object. Let the image foreground contains all scene elements in front of the object, where some parts of the foreground occlude the object. As illustrated in Fig. 1, the camera captures a sequence of images $I _ { 1 } , \ldots , I _ { t } , \ldots , I _ { T }$ , for which each image $I _ { t } \in \mathbb { R } ^ { H \times W }$ is of height H and width W. The stacked images $\mathcal { T } \in \mathbb { R } ^ { H \times W \times T }$ are the input to the algorithm. One of the images in the sequence is denoted the reference image $I _ { r }$ . The result of the algorithm is a computationally generated image $\hat { I } _ { r } \in \mathbb { R } ^ { H \times \smile }$ which represents the reconstructed object at time r.

## 3.1. Underlying Computation

Motion Compensation: A classical constraint for computing the apparent motion between two images $I _ { t }$ and $I _ { t + 1 }$ is the image brightness constancy [17]

$$
I _ { t + 1 } ( u + p ( u , v , t ) , v + q ( u , v , t ) ) \approx I _ { t } ( u , v ) .\tag{1}
$$

Under most circumstances, the apparent brightness of moving objects remains constant [46]. The components $p ( u , v , t )$ and $\boldsymbol { q } ( u , v , t )$ in Eq. (1) form a so-called flow vector

$$
\mathbf { v } _ { t } ( u , v ) = \left( p ( u , v , t ) q ( u , v , t ) \right) ^ { \top }\tag{2}
$$

that describes the apparent motion in pixel $I _ { t } ( u , v )$ $\mathbf { v } _ { t } ( u , v )$ points to the corresponding location of this pixel in $I _ { t + 1 }$ . The flow vectors of all pixels specify a warping operation

$$
\mathcal { F } _ { t } ( u , v ) = \left( u v \right) ^ { \top } + \mathbf { v } _ { t } ( u , v )\tag{3}
$$

between $I _ { t }$ and $I _ { t + 1 } . \mathrm { E q . } ( 1 )$ can be rewritten using a warping operation that takes $I _ { t + 1 }$ and warps it according to $\mathcal { F } _ { t }$ to get

$$
I _ { t } ( u , v ) \approx I _ { t + 1 } ( \mathcal { F } _ { t } ( u , v ) ) = J _ { t + 1 } ( u , v ) .\tag{4}
$$

$J _ { t + 1 }$ in Eq. (4) is a motion-compensated image with pixels registered to their corresponding pixels in $I _ { t }$

Without loss of generality and for simplicity, let $r \ = \ 1$ With this reference image, brightness constancy is applicable to the images $I _ { t } \ ( 1 < t \leq T )$ ), by expanding Eq. (4):

$$
I _ { r } ( u , v ) = I _ { 1 } ( u , v ) \approx J _ { t } = I _ { t } ( \mathcal { F } _ { 1 } \circ . . . \mathcal { F } _ { t - 1 } ( u , v ) ) .\tag{5}
$$

Let $J _ { 1 } = I _ { 1 } \colon \mathrm { E q . } ( 5 )$ warps the $T - 1$ images to the reference image. These motion-compensated images and the reference image are then put together to form the image stack

$$
\mathcal { I } = ( J _ { 1 } , J _ { 2 } , \ldots , J _ { T } ) \in \mathbb { R } ^ { H \times W \times T } .\tag{6}
$$

The advantage of the stack structure is that the registered pixels are slices $( \mathcal { I } ( u , v , 1 ) , \ldots , \mathcal { I } ( u , v , T ) )$ of ${ \mathcal { I } } ,$ , which makes the successive integration of images directly applicable. Without warping, the motion compensation would need to infer the trajectories from the flow fields beforehand and then retrieve the pixel values which is memory inefficient.

Image Integration: Following common techniques in various fields (e.g. [15], [19], [49]), we propose to specify the integrated image as the average

$$
\hat { I } _ { r } ( \boldsymbol { u } , \boldsymbol { v } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { I } ( \boldsymbol { u } , \boldsymbol { v } , t ) .\tag{7}
$$

Eq. (5) implies that brightness constancy is established between the reference image and the integrated image

$$
I _ { r } ( u , v ) \approx \hat { I } _ { r } ( u , v ) , \quad ( u v ) \in \mathcal { O } \subseteq I _ { r } .\tag{8}
$$

Suppose the flow field corresponds to the relative motion between object and camera. Then, Eq. (8) is valid for the correctly compensated motion and for region O, which contains the object in the integrated image. Although averaging is sensitive to the occluder’s brightness values, O does not appear blurry as much as pixels in the rest of the integrated image. This is because the registered pixels outside O are averages of non-registered occluder and background. In contrast, the averages of pixels registered within O are closer to the unoccluded object appearance.

Image Contrast: To find the warps $\mathcal { F } _ { t } ( u , v )$ , we propose to evaluate the sample variance

$$
\mathrm { v a r } ( \hat { I } _ { r } ) \triangleq \mathrm { v a r } ( \{ ( \hat { I } _ { r } ) _ { i } \} _ { 1 } ^ { H W } )\tag{9}
$$

over the HW pixel values of the integrated image.

The pixel intensities in the integrated image results from warping and averaging and are not identical to intensities in the raw images. For warping corresponding to the motion of some object, the natural high contrast of this object is preserved also in $\hat { I } _ { r }$ . The empirical image variance is an indication of this contrast [13]. This is especially true for objects of high contrast but seems to work in general. The motion of a stationary occluder can be easily excluded because it is zero. If the sought-for occluded object is large enough, then the contrast is maximized for its nonzero motion, and enable the algorithm to estimate the required warping.

On the other hand, the less accurate the warping is, the lower the contrast of the integrated image will be, because large areas in the integrated image are blurred due to the averaging process. This makes the sample variance smaller.

This is a simple criterion but it has one important advantage: the contrast is a smooth function of the warping accuracy. This was confirmed in our experiments. Note that alternative, seemingly improved, criteria, such as the use of robust integration [49] will lose this property. We also experimented with more complex gradient based cost functions (see [14]) and found that they induce a harder search space, with many local maxima.

Fixation Window: We found that the sample variance for the correct warping may not be the maximum relative to other warpings. This is because real-world scenes often yield largely varying foreground and background intensities over the image. Consequently, the integrated sample variance is large and may overshadow the variance associated with the occluded object; see Figs. 6e to 6h in Sec. 4 and Sec. 9 in the supplemental material.

One way to mitigate this problem is to evaluate the sample variance over a smaller region or fixation window W of the integrated image, which encompasses the object or object parts. That is to calculate the variance by

$$
\begin{array} { r } { \operatorname { v a r } _ { | W | } ( \{ \hat { I } _ { r } ( u , v ) \} ) , ( u , v ) \in W \subseteq I _ { r } , W \cap \mathcal { O } \neq \emptyset . } \end{array}\tag{10}
$$

Sec. 3.3 describes the procedure for computing this window.

## 3.2. Algorithm

Using the variance to estimate the image contrast as an indication to the validity of the estimated flow fields, suggests an algorithm that alternates between computing $\bar { \boldsymbol { I } } _ { r }$ with Eq. (7) and (re-)estimating $\mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { t - 1 }$ by evaluating $\hat { I } _ { r }$ with Eq. (9). Thereby the reconstruction of the occluded object, which is within the integrated image, occurs simultaneously with the motion computation. The (re-)estimation of the flow fields is written as optimization problem with the flow fields as unknown variables, or more compactly, as parameter vector

$$
\boldsymbol { \theta } = \{ \mathbf { v } _ { 1 } ( 1 , 1 ) ^ { \top } , \mathbf { v } _ { 1 } ( 2 , 1 ) ^ { \top } , \dots , \mathbf { v } _ { T - 1 } ( H , W ) ^ { \top } \} .\tag{11}
$$

With θ, Eq. (3) can be written as parametrized warp

$$
\mathcal { F } _ { t } ( u , v ; \theta ) = \left( u v \right) ^ { \top } + \left( \theta _ { k + 1 } \theta _ { k + 2 } \right) ^ { \top }\tag{12}
$$

![](images/b09cade29c606b99745b88c45d663803fb38c47b913fc45c8fcda91330ec96ea.jpg)  
Figure 2. Stack $\mathcal { I }$ of motion compensated images (top) is combined into the integrated image (bottom). The algorithm then estimates the contrast of $\hat { I } _ { r }$ for different motions, $\mathcal { F } _ { i } , \mathcal { F } _ { j }$ and $\mathcal { F } ^ { * }$ . The correct motion $\theta ^ { * }$ and the reconstruction of the occluded object is simultaneously found with $\hat { I } _ { r }$ at its maximal contrast.

with the vector’s component index $k = 2 H W ( t - 1 ) +$ $H ( v - 1 ) + 2 ( u - 1 )$ . The optimization problem now is

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } f ( \theta ) + \lambda R ( \theta )\tag{13}
$$

with objective function

$$
f ( { \boldsymbol { \theta } } ) = \operatorname { v a r } _ { | W | } ( { \hat { I } } _ { r } ) .\tag{14}
$$

Prior knowledge of the motion is encoded by the regularization term $R ( \cdot )$ . λ is a hyperparameter. By substituting $\hat { I } _ { r }$ in Eq. (14) with Eq. (7), Eq. (6) and Eq. (5), the objective function is written as

$$
\boldsymbol { f } ( \boldsymbol { \theta } ) = \mathrm { v a r } _ { | W | } ( I _ { r } + \sum _ { t = 2 } ^ { T } I _ { t } ( \mathcal { F } _ { 1 } \circ . . . \circ \mathcal { F } _ { t - 1 } ( \boldsymbol { u } , \boldsymbol { v } ; \boldsymbol { \theta } ) ) .\tag{15}
$$

If all warps $\mathcal { F } _ { t }$ are accurate, $i . e .$ the result of the optimization $\theta ^ { * }$ is close to the true object motion, the objective function in Eq. (15) will attain its maximum value. As explained above, the correct motion estimate results in high contrast of the integrated image or a large value of $f ( \theta )$ Conversely, all other motions cause blur, leading to low contrast. Fig. 2 illustrates the algorithm: $\theta ^ { * }$ is found at $\nabla f = \mathbf { 0 }$ and negative definite Hessian $H ( f )$ ).

## 3.3. Remarks

Parametrized Motion: A completely free local motion model $\mathbf { v } _ { t } ( u , v )$ in Eq. (2) would require too many variables. This work therefore uses a parametrized motion model [20]

$$
\mathbf { v } _ { t } ( u , v ; \theta ) = \left( p ( u , v , t ; \theta ) q ( u , v , t ; \theta ) \right) ^ { \top } .\tag{16}
$$

The flow components p and $q$ describe the motion in Eq. (16) by some functions, typically low dimensional polynomials. Certainly, p and q must be complex enough to capture real motion. A second advantage of the parametrized motion models over the local model is that their computation is more robust to fragmented occlusion.

In this paper we use a motion model of coherent translation at certain depth, which is a special case of a rigid scene’s translation [46]:

$$
p ( u , v , t ; \theta ) = ( t - r ) \alpha\tag{17}
$$

$$
q ( u , v , t ; \theta ) = ( t - r ) \beta\tag{18}
$$

with two parameters

$$
\begin{array} { r } { \theta = \left( \alpha \beta \right) ^ { \top } . } \end{array}\tag{19}
$$

The translation in the scenes considered here is often close to horizontal in the image plane, which encourages the prior

$$
R ( \theta ) = - \beta ^ { 2 } .\tag{20}
$$

This simpler motion model seems sufficient for the general assumption of an upright camera and objects moving at close to constant depth on a perpendicular ground plane, and using it led to successful reconstructions with videos acquired by a camera.

Implementation Details: We used two alternative method for optimization. For visualization of $f ,$ we use grid search over intervals of θ. For more efficient search, we parallelize the problem in JAX [12]. We use gradient ascent [5] with Adam in combination with JAX’s accurate auto-differentiation. For the experiments, we set λ to $\frac { 1 } { 2 }$ and the step size to 0.1.

Window Placement: As discussed above, for highly varying brightness in outdoor images, it is crucial to estimate the variance within a window and not over the whole image. The fixation window is placed at the location of maximal saliency, and its size is set to 10 % of the image size. The work uses a simple motion based saliency map which is the absolute difference of consecutive images. For good results, the images are blurred with a Gaussian kernel.

## 4. Experimental Results

The experimental methodology follows Vaish et al.’s empirical analysis [49] and includes experiments with synthetic images and images acquired by a camera. Selected videos are provided in the supplementary material (see Sec. 11).

## 4.1. Reconstruction in Synthetic Videos

Following the scene model outlined in Sec. 3, we generate eight-bit, grayscale images using Luxor (https:

<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1> $\rho$ </td><td rowspan=2 colspan=1> $T$ </td><td></td></tr><tr><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\overline { { 1 1 . 6 7 \pm 0 . 5 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 5 . 2 4 \pm 1 2 . 3 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 . 0 \pm 0 . 0 0 } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 8 . 9 1 \pm 1 . 0 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 5 . 4 0 \pm 1 2 . 2 6 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 . 0 \pm 0 . 0 0 } }$ </td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> $\overline { { 5 1 . 2 0 \pm 1 . 1 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 3 . 3 6 \pm 0 9 . 1 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 . 0 \pm 0 . 0 0 } }$ </td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1> $\overline { { 7 0 . 2 2 \pm 1 . 5 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 4 . 4 4 \pm 1 0 . 9 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 9 9 . 8 3 \pm 2 . 4 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1> $\overline { { 8 2 . 5 7 \pm 1 . 1 4 } }$ </td><td rowspan=1 colspan=1> $2 3 . 9 8 \pm 1 0 . 6 6$ </td><td rowspan=1 colspan=1> $\overline { { 9 6 . 6 4 \pm 2 . 6 1 } }$ </td></tr><tr><td rowspan=1 colspan=1> $^ 6$ </td><td rowspan=1 colspan=1> $\overline { { 9 0 . 5 1 \pm 1 . 8 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 5 . 9 1 \pm 1 2 . 1 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 4 . 0 0 \pm 8 . 6 9 } }$ </td></tr></table>

Table 1. Synthetic video statistics: The image sequences are divided into six sets associated, each, with roughly the same occlusion density $\rho$ (mean ± std. deviation of each quantity). The sequence length $T$ and the object’s total visibility ν are also given.

//juliapackages.com/p/luxor). Fig. 1 is an example. The background behind the object and the foreground, which also comprises the occluder, are set to a midgray value of 128. The foreground is composed of randomly placed and oriented triangles that may overlap one another [8]. Uniformly varying the triangles’ size controls the occlusion density $\rho \ ( \% )$ up to a small, random deviation. This occlusion model mimics foliage reasonably. For a different model see [26].

Experimental Setup: As the object, we choose a planar ”Ape” cartoon from OpenClipArt $( \mathrm { h t t p s : } / \ \nearrow$ openclipart.org), convert it to grayscale and resize the image to fill two-thirds of the $2 5 6 ^ { 2 }$ px canvas. For each generated image sequence, the object appears at the image center for the reference time

$$
r = { \frac { 1 } { 2 } } ( T - 1 ) , \quad T = 2 t + 1 , \quad t = 1 , 2 , 3 , \ldots ,\tag{21}
$$

and the object is following a uniform translation in the image plane. We randomly varied velocity within fixed bounds to generate image sequences of different length T. The experimental methodology, but not the algorithm, had access to the unoccluded image $\bar { I } _ { r }$ , to a binary mask $M _ { r }$ indicating all the object’s pixels and to a measurement of total visibility ν (%) that expresses the fraction of pixels that are visible in at least one image of the sequence.

We generate a total of 300 sequences for six specified occlusion levels: $\rho = \{ 1 0 , 3 0 , 5 0 , 7 0 , 8 0 , 9 0 \}$ . Each sequence contains a unique occlusion pattern, specified by the triangles’ size, positions and orientations. Consequently, the actual occlusion level is random, and given in Tab. 1.

Metrics: To assess the quality of the reconstructed image, we calculate the Pearson correlation coefficient

$$
\mu _ { \hat { \mathrm { P } } } \doteq \mu _ { \mathrm { P } } ( M _ { r } \hat { I } _ { r } , M _ { r } \bar { I } _ { r } )\tag{22}
$$

between the reconstructed image $\hat { I } _ { r }$ and the known, unoccluded image $\bar { I } _ { r }$ . As the algorithm reconstructs the object

![](images/440a3eeb7b6532f14688801f77fba62850d233007ce1e735401ba2cdad30a474.jpg)

![](images/151f57deb176970c275ec874132d51161a7a709fcf98525e5171427f1c7a5592.jpg)  
(a) Results  
(b) Evaluation of µ<sub>MSE</sub> at T  
Figure 3. Experimental results: (a) The plots draw µ<sub>MSE</sub> (square), $\mu _ { \hat { \mathrm { P } } }$ (cross), µ<sub>P</sub> (circle) with confidence interval ±3σ (ordinate) vs. ρ (abscissa). (b) Again µ<sub>MSE</sub> vs. ρ, but we provide information on the sequence length T: Dark blue encodes long sequences. Longer sequences do not necessarily leader to a better reconstruction.

and no other scene element, the metric is only applied to the object region, determined by the mask $M _ { r }$

For comparison, we include the correlation coefficient

$$
\mu _ { \mathrm { P } } \doteq \mu _ { \mathrm { P } } ( M _ { r } I _ { r } , M _ { r } \bar { I } _ { r } )\tag{23}
$$

between the given occluded image $I _ { r }$ and $\bar { I } _ { r }$

We also use the mean-squared error (MSE) ratio as an additional measure of quality for the reconstruction:

$$
\mu _ { \mathrm { M S E } } = \frac { \mathbf { M S E } ( M _ { r } I _ { r } , M _ { r } \bar { I } _ { r } ) - \mathbf { M S E } ( M _ { r } \hat { I } _ { r } , M _ { r } \bar { I } _ { r } ) } { \mathbf { M S E } ( M _ { r } I _ { r } , M _ { r } \bar { I } _ { r } ) } .\tag{24}
$$

$\mu _ { \mathrm { P } }$ and MSE are described in Sec. 7 of the supp. material.

Results: Fig. 3 shows the experimental results. Note that the average correlation coefficient associated with the reconstructed images $\mu _ { \hat { \mathrm { P } } }$ is larger than 0.7 for all occlusion levels. On the other hand, as expected, the correlation coefficient associated with the original occluded images $\mu _ { \mathrm { P } }$ is lower than 0.23. The large values of $\mu _ { \hat { \mathrm { P } } } ,$ , obtained after the integration, is a strong indicator that corresponding pixels are found in the warped images of Eq. (6) - even for very dense occlusion. The MSE ratio rapidly decreases as the level of occlusion increases. Some of the degradation in reconstruction performance is due to object parts that are never seen during the sequence; see ν in Tab. 1 for sets 4-6.

Figs. 4b to 4f illustrates occluded and reconstructed images for five representative samples of the 300 synthetic videos. The quality of the reconstructed object is comparable to the result in [49]. Their plane sweeping algorithm, however, fails for $\rho > 7 5 \%$ , while the motion computation in Eq. (13) is successful even for $\rho > 9 0 \%$ . This success is also evident from the large $\mu _ { \hat { \mathrm { P } } }$ values. The gradient-based motion estimation algorithm in Eq. (13) achieves an accuracy of $\alpha , \beta \pm 0 . 0 1$ px or better. Fig. 4a lists the associated quantitative characteristics of these five sample sequences.

<table><tr><td>#</td><td>ρ</td><td>T</td><td>ν</td><td>α</td><td>β</td><td>μMSE</td><td>µp</td><td>µp</td></tr><tr><td>57</td><td>27.6</td><td>13</td><td>100</td><td>-6.0</td><td>-6.0</td><td>0.90</td><td>0.97</td><td>0.09</td></tr><tr><td>149</td><td>49.7</td><td>19</td><td>100</td><td>4.0</td><td>6.0</td><td>0.72</td><td>0.97</td><td>0.10</td></tr><tr><td>193</td><td>70.3</td><td>13</td><td>99</td><td>-6.0</td><td>-4.0</td><td>0.52</td><td>0.87</td><td>0.02</td></tr><tr><td>203</td><td>85.2</td><td>59</td><td>99.8</td><td>0.0</td><td>2.0</td><td>0.27</td><td>0.86</td><td>0.16</td></tr><tr><td>300</td><td>92.3</td><td>25</td><td>71.2</td><td>-3.0</td><td>-1.0</td><td>0.10</td><td>0.54</td><td>0.07</td></tr></table>

(a) Statistics and quantitative results

![](images/f97b1ec8090c7d6147d990713905c53dc9172ed586ae5b1fdf886114857befee.jpg)  
(b) 27.6 %

![](images/d230940a8e152456109b4e6af04870a6ff90c309e7affccdd34bcc74ee89e684.jpg)  
(c) 49.7 %

![](images/17a4299048852a0e40af07f2fdde8b88561f90eee03d00445a2409586f68db0e.jpg)  
(d) 70.3 %

![](images/fadcc0e83a95f92ce02fd32127162abc146e5eab87d1d5d6c39a6c7238075736.jpg)  
(e) 85.2 %

![](images/56bbb005a0b169187ec2100b224e8b2e962349f83f5918f89b60dfffa22ee508.jpg)  
(f) 92.3%  
Figure 4. Five synthetic sample videos: (a) quantitative data. (b) - (f) Raw occluded images (top row) and the object reconstructions for the different occlusions densities (bottom row).

7

## 4.2. Robust Recognition

Visually, it appears that the images are more comprehensible after reconstruction (see Figs. 4b to 4f). To quantify this impression, the following experiment shows that the proposed algorithm enhances the resilience of zero-shot offthe-shelf classification to fragmented occlusion.

Experimental Setup: We use the CLIP model [41] based on the pretrained ViT-B/32 architecture (https : / / huggingface . co / openai / clip - vit - base - patch32) to encode both occluded and reconstructed images, one from each of the 300 sequences. We then specify text labels, such as: $" a$ photo of a {l}", with labels $l \in L$ coming from a set L of category names, including ape and diverse outdoor categories such as deer, helicopter, swamp. These categories are inspired by [28] and adapted to our project needs. The 24 used text labels are found in Sec. 6 of the supp. material. Finally, we encode the tokenized text labels into the same CLIP space. No training is done.

Metric: Let p and q be two features of image and text respectively. The cosine distance [42] between them is

$$
d ( \mathbf { p } , \mathbf { q } ) = 1 - \frac { \mathbf { p q } ^ { \top } } { \| \mathbf { p } \| \| \mathbf { q } \| } .\tag{25}
$$

Due to CLIP’s modality gap [30], the cosine distance is typically large. The following normalized measure is contextaware and scale-invariant, and better accounts for the con-

![](images/b53a4bafbe2d2d551f0305c8d36c64ee594f18eebfda08d5a68ba113339b7e52.jpg)  
(a) Occluded

![](images/237f05e0302cd730fd831c38d4dbfc14a30cb3f775d21d364d7716c3ad52efc0.jpg)  
(b) Reconstructed

![](images/002b52ebe08522ce075953e0379c0cbb80b4d8a752eb2fc5f244e61e60a76b1c.jpg)  
(c) Cls. Error Rate  
Figure 5. Recognition with CLIP: (a) The distance D vs. the occlusion density $\rho$ for the occluded images. The mean and confidence interval of ape is shown in blue. (b) The same evaluation for the reconstructed images, shows a significant improvement. (c) The CLIP’s classification error rate vs. ρ.

trastive loss employed during CLIP training [41]

$$
D ( \mathbf { p } , \mathbf { q } _ { r } ) = \frac { 1 } { \left| L \right| } \sum _ { i } \frac { d ( \mathbf { p } , \mathbf { q } _ { r } ) } { d ( \mathbf { p } , \mathbf { q } _ { i } ) } .\tag{26}
$$

D determines whether a text label $\mathbf { q } _ { r }$ , when compared to all other text labels, is relatively distant or close to image p.

Results: Figs. 5a to 5b evaluate D for all 300 occluded and integrated images, and show the means and the 2σ confidence intervals of the correct label ape. The results confirm our first impression: The proposed algorithm substantially reduces the normalized distance D for the ape label.

Top1 classification with the occluded image fails for $\rho >$ 30 ${ \% } ,$ but, for deoccluded images, was perfect for $\rho < 9 0 \%$ (see Fig. 5c). The same results were (obviously) obtained with the cosine distance. This is useful and remarkable, when remembering that usually, CNN models trained on non-occluded images, fail under occlusion [37].

## 4.3. Foliage Experiments

Three experiments demonstrate the proposed reconstruction in challenging real-world scenes. The first sequence contains 20 images, with a walking person in a park on a sunny day. The foliage is moderate and comparable to Vaish’s foliage experiment [49]. A second sequence contains 20 images of a closer person in the boreal forest with higher occlusion density. A third boreal forest sequence contains 30 images and was captured by panning the camera while the person was walking.

Results: By employing the fixation window (Figs. 6a, 8a and 8c), the algorithm finds a maximum for the person in all three sequences. Our current window placement algorithm, is simple and assumes a static camera. For the third sequence with camera motion, we therefore placed the fixation window manually.

Fig. 7a depicts an evaluation of $f ,$ clearly demonstrating in the Park sequence the maximum value for $\alpha = 6 . 0 \mathrm { p x }$ . The associated integrated image is in Fig. 6c and its histogram in Fig. 10 of the supp. material. Note that a peak for zero motion $( \alpha = 0 )$ in f always exists, reflecting the static camera and the stationary large background and foreground; see the reconstructed static fore- and background integrated image in Fig. 6e. To search for the correct person’s motion, the optimizer in Eq. (13) therefore excludes the subspace in the vicinity of $\alpha = 0$

![](images/c5e6a2e146c295801f7a9d1410f8f7d947f1771666e7339b7123619c86416af4.jpg)  
(a) I<sub>1</sub>

![](images/cc5815a115c816099b77e6f5faaa8f09f75ec8bcd34fdfb02342ed25acc253f8.jpg)  
(b) Evaluation of d with $I _ { r } = I _ { 1 }$

![](images/f28737dc613359ca6a1976ffb427681105de6864b40d6c711a2873b7a59b91ff.jpg)

![](images/3d5708689838750f64d5110f9381071222abf8543837a3242b3cb2ffc3a79891.jpg)

(c) $\alpha = 6 . 0 \mathrm { p x }$  
![](images/a502012f4b1e78ffd78e76500926ba601b5a38815d749dc1453249b7829ef19d.jpg)  
(e) $\alpha = 0 . 0 \mathrm { p x }$

![](images/386c2f27b467348d1100cb881181942ff076bd3e2ed5fb8c857b8d05ae934251.jpg)  
(f) α = 3.0 px

(d) Evaluation of d with I<sup>ˆ</sup>  
![](images/901c089c5f6b8ce1a5ac5f8be000ecd4fd69d37f1e0ce1759ad7548c717e5c0f.jpg)  
(g) α = −5.0 px

![](images/e08504dc0a6b80c82a87bbca6b28bcf3fa06002fe4858900d87b955dcff6ed94.jpg)  
(h) α = 10.0 px

Figure 6. Park results: (a) Reference image and fixation window. (b) Applying CLIP to (a). The bar plot shows the four smallest cosine distances for label human figure and the same 24 diverse outdoor categories as used for the synthetic videos. (c) object reconstruction, (d) applying CLIP to (c). Note the correct ranking compared to (b). (e) fore- and background reconstruction. (f) - (h) integral images corresponding to some incorrect motions.  
![](images/162baf3222cb1ae35929eafe96becda2ae7bea0b43090970fd649bd5a22f10c9.jpg)  
(a) Park

![](images/aab9e4bee6ebaf89fd8702ceadd1d74fa9a788d85b38999aee4617c91aa33097.jpg)  
(b) Fence  
Figure 7. Both plots draw f (ordinate) vs. α (abscissa). (a) Two significant maxima of the person and static motions are observable. (b) The overlaps of fence slats cause misleading maxima.

![](images/fa7891251c70db7903e6e1085af7ee30ad67f81aeb937eda1fc1a3618b4a5b52.jpg)

![](images/cd6d4f456ec8ab57ea8fabef73b9240d7f2b2b51fca18e0cfcf27981d8887fdb.jpg)  
(b) α = 1.5 px

(a) I<sub>1</sub>  
![](images/25380fa174b2511212c15bbf03f99d7c1ab89cd58321a2ea922d643db03eb1dc.jpg)  
(c) I<sub>27</sub>

![](images/9474719f5d27f526455e832631a8ed67b137168dceca6b23365ff0d431889275.jpg)

![](images/5e5dd65fab2657878ccbdc7e763c56d3d370515968fb320395a132dc9f3b9324.jpg)  
(d) α = −1.75 px  
(e) α = 2.75 px  
Figure 8. Boreal forest and camera pan results: (a) and (c) ref. images and fixation windows. (b) and (d) object reconstruction. (e) foreground (tree) reconstruction related to camera panning.

We also applied the CLIP classification algorithm to the park sequence. As with the synthetic videos, the same 24 categories were considered, plus human figure instead of ape. Using the de-occluded image as input led to successful top-1 classification (Figs. 6b and 6d). Figs. 6e to 6h illustrate qualitative results for specific translations. Figs. 6c, 8b and 8d show the reconstructed persons. To enhance visibility, those images’ histograms were linearly stretched to span the full brightness range [0, 255].

In contrast, in the third boreal sequence, the camera pan introduces a global motion. If this motion is selected then the algorithm reconstructs near distant trees which move at $\alpha = 2 . 7 5 \mathrm { p x }$ (Fig. 8e). The camera pan is not a translation, so not all parts of the foreground and background are sharply reconstructed.

## 4.4. Fence Experiment

We finally tested the algorithm on a forth sequence of 30 images, with a moving car (Fiat 500) behind a garden fence. Fig. 9 shows the results. This sequence was taken with a hand-held cellular phone, demonstrating the robustness to small camera motions during image acquisition.

As depicted in Fig. 7b, f exhibits numerous local maxima that are not associated to the car’s motion. Although the integrated image at $\alpha = 2 9$ has lower contrast (see Fig. 9d), it appears to be a better reconstruction than the image found by the algorithm (see Fig. 9f). The misleading maxima are probably due to the repetitive textures and regions of uniform intensities, such as the fence slats. See also the associated histograms in Fig. 11 and the of the supp. material.

![](images/6d3518663f13b27b83d5ebeba1c19644b29e039fc1beba106e715ab22c73cbf6.jpg)  
(a) I<sub>14</sub>

![](images/276360de005b052e5bb5b31f16aee1c6cab7545d5cfbadca81aa3010f5c15191.jpg)  
(b) α = 21.5 px

![](images/c0120321374125b3c1743b433ed90e7ee796b8601f28da56378f3b7a9d35f828.jpg)  
(c) α = −8.0 px

![](images/c68dd519d8ea3e54ca5b6f315ea62d2d8e2b1af5fb0930b6961f389739e47d2a.jpg)

![](images/496afc9dcef1ad3644d61953e0b7de9f3be247740437b8598747272915c26ef7.jpg)  
(e) α = 0.0 px

(d) α = 29.0 px  
![](images/fbb0d0b2538675f18c1e36a01e2165d62a0d9c1dd6fb7e693af66132a0760222.jpg)  
(f) α = 32.5 px  
Figure 9. Fence results: (a) Reference image and fixation window. (b) - (f) Integral images for different motions. (e) fore- and background reconstruction. (f) car reconstruction.

## 4.5. Discussion of Experimental Results

The reconstruction quality is comparable to that obtained from synthetic apertures using stereo [49], but our algorithm maintains its accuracy for higher occlusion densities, exceeding 70 %. This is due to the proposed coherent motion computation, which is more robust to occlusion. We found that the quality primarily depends on the scene specific motion and occlusion patterns, rather than the video length, which is known for a long time [20].

A classification algorithm, combining the proposed reconstruction and a zero-shot CLIP classification, achieves a high, 70.0 % top1 accuracy for images with approximately 90 % occlusion. This is remarkable because applying CLIP on the occluded image leads to 2.0 % accuracy, which is close the chance level.

Interestingly, f in Eq. (15) has local maxima corresponding to all scene motions, and motion computation involves searching for all maxima. In comparison to alternative algorithms such as optical flow [17] and motion layers [52], this approach is robust to occlusion and does not require the number of motions given. The same computational process is used for both finding the motion and for reconstruction.

## 5. Conclusions

The paper proposes an algorithm to reconstruct or deocclude an object behind fragmented occlusion. It is an alternative to synthetic aperture techniques, and relies on motion between the object and the camera. Averaging registered images to create an integrated image and using sample variance as a contrast measure yields local maxima under certain conditions, as discussed in this paper. These maxima determine the object’s motion, and the resulting reconstructed object. Using the integrated image and CLIP as a straightforward zero-shot classifier, achieves an impressive average 70.0 % accuracy for images with 90 % occlusion. This is significant because no other known classifiers perform well under such conditions. The algorithm is simple but provides useful results up to 90 % occlusion density and significantly improves state-of-the-art performance. More complex motion models or robust estimation for generating the integrated image, have the potential to improve the results. Learning can be used throughout the process, but we did not use it yet to reduce the risk of hallucinations. We are currently investigating these directions.

The work focuses on psychological studies in dynamic perception. The registration in the algorithm can be seen as a fixation on the object. Interestingly, a human fixation on a point makes the rest of the retinal image blurry, a phenomenon known as Troxler’s fading [2], but also observed in the integrated image. While the common explanation to the Troxler’s fading is different, it would be interesting to see whether the algorithm provides a relevant explanation.

## References

[1] Alessandro Achille. Emergent properties of deep neural networks. PhD thesis, UCLA, 2019. 3

[2] Robert G. Alexander, Ashwin Venkatakrishnan, Jordi Chanovas, Stephen L. Macknik, and Susana Martinez-Conde. Microsaccades mediate perceptual alternations in monet’s “impression, sunrise”. Scientific Reports, 11(1): 3612, 2021. 9

[3] Rakesh John Amala Arokia Nathan, Indrajit Kurmi, and Oliver Bimber. Inverse airborne optical sectioning. Drones, 6(9), 2022. 2, 3

[4] Alper Ayvaci, Michalis Raptis, and Stefano Soatto. Sparse occlusion detection with optical flow. International Journal ofComputer Vision, 97(3):322–338, 2012. 3

[5] Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: a survey. Journal of Machine Learning Research, 18(153):1–43, 2018. 5

[6] James R. Bergen, P. Anandan, Keith J. Hanna, and Rajesh Hingorani. Hierarchical model-based motion estimation. In Computer Vision — ECCV’92, pages 237–252, Berlin, Heidelberg, 1992. Springer Berlin Heidelberg. 2

[7] Michael J. Black and P. Anandan. The robust estimation of multiple motions: Parametric and piecewise-smooth flow

fields. Computer Vision and Image Understanding, 63(1):75 – 104, 1996. 1

[8] Robert Bridson. Fast poisson disk sampling in arbitrary di mensions. In ACM SIGGRAPH 2007 Sketches, pages 22–es, New York, NY, USA, 2007. Association for Computing Ma chinery. 5

[9] Gideon Paul Caplovitz. On the spatiotemporal nature of vision, as revealed by covered bridges and puddles: A dispatch from vermont. i-Perception, 12(6):20416695211062625, 2021. PMID: 35035871. 2

[10] T. Darrell and A. Pentland. Robust estimation of a multilayered motion representation. In Proceedings of the IEEE Workshop on Visual Motion, pages 173–178, 1991. 1

[11] P. Favaro and S. Soatto. Seeing beyond occlusions (and other marvels of a finite lens aperture). In 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings., pages II–II, 2003. 2

[12] Roy Frostig, Matthew Johnson, and Chris Leary. Compiling machine learning programs via high-level tracing. In Pro ceedings of the Conference on Machine Learning and Sys tems (MLSYS), 2018. 5

[13] Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3867–3876, 2018. 2, 3

[14] Guillermo Gallego, Mathias Gehrig, and Davide Scaramuzza. Focus is all you need: Loss functions for event-based vision. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12272–12281, 2019. 2, 4

[15] Umer Hassan and Muhammad Sabieh Anwar. Reducing noise by repetition: introduction to signal averaging. Eu ropean Journal ofPhysics, 31(3):453, 2010. 3

[16] Jinpeng He, Huaixin Chen, Biyuan Liu, Sijie Luo, and Jie Liu. Enhancing yolo for occluded vehicle detection with grouped orthogonal attention and dense object repulsion. Scientific Reports, 14(1):19650, 2024. 2

[17] Berthold K.P. Horn and Brian G. Schunck. Determining optical flow. Artificial Intelligence, 17(1):185–203, 1981. 3, 8

[18] J. Hur, J. Lee, J. Choi, and J. Kim. I see-through you: A framework for removing foreground occlusion in both sparse and dense light field images. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 229–238, Los Alamitos, CA, USA, 2023. IEEE Computer Society. 2

[19] Michal Irani and Shmuel Peleg. Improving resolution by image registration. CVGIP: Graphical Models and Image Processing, 53(3):231–239, 1991. 3

[20] Michal Irani, Benny Rousso, and Shmuel Peleg. Computing occluding and transparent motions. International Journal of Computer Vision, 12(1):5–16, 1994. 2, 4, 8

[21] Neel Joshi, Shai Avidan, Wojciech Matusik, and David J. Kriegman. Synthetic aperture tracking: Tracking through occlusions. In 2007 IEEE 11th International Conference on Computer Vision, pages 1–8, 2007. 2

[22] Anil Kokaram. A statistical framework for picture reconstruction using 2d ar models. Image and Vision Computing, 22(2):165–171, 2004. Statistical Methods in Video Processing. 2

[23] A. Kortylewski, J. He, Q. Liu, and A. L. Yuille. Compositional convolutional neural networks: A deep architecture with innate robustness to partial occlusion. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8937–8946, Los Alamitos, CA, USA, 2020. IEEE Computer Society. 2

[24] Ketan Kotwal, Tanay Deshmukh, and Preeti Gopal. Latent enhancing autoencoder for occluded image classification. In 2024 IEEE International Conference on Image Processing (ICIP), pages 894–900, 2024. 2

[25] Indrajit Kurmi, David C. Schedl, and Oliver Bimber. Airborne optical sectioning. Journal of Imaging, 4(8), 2018. 2

[26] I. Kurmi, D. C. Schedl, and O. Bimber. A statistical view on synthetic aperture imaging for occlusion removal. IEEE Sensors Journal, 19(20):9374–9383, 2019. 2, 5

[27] Indrajit Kurmi, David C. Schedl, and Oliver Bimber. Fast automatic visibility optimization for thermal synthetic aperture visualization. IEEE Geoscience and Remote Sensing Letters, 18(5):836–840, 2021. 2

[28] Connor Lee, Matthew Anderson, Nikhil Ranganathan, Xingxing Zuo, Kevin Do, Georgia Gkioxari, and Soon-Jo Chung. Caltech aerial rgb-thermal dataset in the wild. In Computer Vision – ECCV 2024, pages 236–256, Cham, 2025. Springer Nature Switzerland. 6

[29] Si-Qi Li, Yue Gao, and Qiong-Hai Dai. Image de-occlusion via event-enhanced multi-modal fusion hybrid network. Machine Intelligence Research, 19(4):307–318, 2022. 2

[30] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y. Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. ArXiv, abs/2203.02053, 2022. 6

[31] Wei Liao, Xiang Zhang, Lei Yu, Shijie Lin, Wen Yang, and Ning Qiao. Synthetic aperture imaging with events and frames. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17714–17723, 2022. 2

[32] Chao Liu, Srinivasa G. Narasimhan, and Artur W. Dubrawski. Matting and depth recovery of thin structures using a focal stack. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4782–4790, 2017. 2

[33] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to see through obstructions with layered decomposition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):8387– 8402, 2022. 2

[34] J Daniel McCarthy, Gennady Erlikhman, and Gideon Paul Caplovitz. The maintenance and updating of representations of no longer visible objects and their parts. Prog Brain Res, 236:163–192, 2017. 2

[35] Yadong Mu, Wei Liu, and Shuicheng Yan. Video de-fencing. IEEE Transactions on Circuits and Systems for Video Technology, 24(7):1111–1121, 2014. 2

[36] Rakesh Nathan, Indrajit Kurmi, and Oliver Bimber. Through-foliage tracking with airborne optical sectioning. Journal ofRemote Sensing, 2022, 2022. 2

[37] Elad Osherov and Michael Lindenbaum. Increasing cnn robustness to occlusions by reducing filter support. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 550–561, 2017. 2, 7

[38] E. M. Palmer, P. J. Kellman, and T. F. Shipley. A theory of dynamic occluded and illusory object perception. Journal of Experimental Psychology, 135(4):513–541, 2006. 2

[39] Julian Pegoraro and Roman Pflugfelder. The problem of fragmented occlusion in object detection. In Austrian Joint Computer Vision and Robotics Workshop (ACVRW), 2020. 1

[40] Roman Pflugfelder and Jonas Auer. Person localisation un der fragmented occlusion. In IEEE AVSS, 2021. 1

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748– 8763. PMLR, 2021. 6, 7

[42] G. Salton, A. Wong, and C. S. Yang. A vector space model for automatic indexing. Commun. ACM, 18(11):613–620, 1975. 6

[43] Thomas Schoenemann and Daniel Cremers. A coding-cost framework for super-resolution motion layer decomposition. IEEE Transactions on Image Processing, 21(3):1097–1110, 2012. 2

[44] S.M. Seitz and C.R. Dyer. Photorealistic scene reconstruction by voxel coloring. In Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 1067–1073, 1997. 2

[45] Stefano Soatto and Alessandro Chiuso. Modeling visual rep resentations: Defining properties and deep approximations. In ICLR (Poster), 2016. 2

[46] Emanuele Trucco and Alessandro Verri. Introductory Tech niques for 3-D Computer Vision. Prentice Hall PTR, USA, 1998. 3, 5

[47] Stavros Tsogkas, Fengjia Zhang, Allan Jepson, and Alex Levinshtein. Efficient flow-guided multi-frame de-fencing. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1838–1847, 2023. 2

[48] V. Vaish, B. Wilburn, N. Joshi, and M. Levoy. Using plane + parallax for calibrating dense camera arrays. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., pages I–I, 2004. 2

[49] V. Vaish, M. Levoy, R. Szeliski, C.L. Zitnick, and Sing Bing Kang. Reconstructing occluded surfaces using synthetic apertures: Stereo, focus and robust measures. In 2006 IEEE Computer Society Conference on Computer Vision and Pat tern Recognition (CVPR’06), pages 2331–2338, 2006. 2, 3, 4, 5, 6, 7, 8, S2

[50] Risto Vehmas and Nadav Neuberger. Inverse synthetic aper ture radar imaging: A historical perspective and state-of-theart survey. IEEE Access, 9:113917–113943, 2021. 2

[51] Kirill Vishniakov, Zhiqiang Shen, and Zhuang Liu. ConvNet vs transformer, supervised vs CLIP: Beyond ImageNet accuracy. In Proceedings ofthe 41st International Conference on Machine Learning, pages 49545–49557. PMLR, 2024. 2

[52] J.Y.A. Wang and E.H. Adelson. Representing moving images with layers. IEEE Transactions on Image Processing, 3 (5):625–638, 1994. 8

[53] Y. Wang, T. Wu, J. Yang, L. Wang, W. An, and Y. Guo. Deoccnet: Learning to see through foreground occlusions in light fields. In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 118–127, Los Alamitos, CA, USA, 2020. IEEE Computer Society. 2

[54] Lei Yu, Xiang Zhang, Wei Liao, Wen Yang, and Gui-Song Xia. Learning to see through with events. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):8660– 8678, 2023. 2

[55] Guanqi Zhan, Weidi Xie, and Andrew Zisserman. A tri-layer plugin to improve occluded detection. In BMVC, 2022. 2

[56] Shanshan Zhang, Di Chen, Jian Yang, and Bernt Schiele. Guided Attention in CNNs for Occluded Pedestrian Detection and Re-identification. International Journal of Computer Vision, 129(6):1875–1892, 2021. 2

[57] Shuo Zhang, Zeqi Shen, and Youfang Lin. Removing foreground occlusions in light field using micro-lens dynamic filter. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pages 1302– 1308. International Joint Conferences on Artificial Intelligence Organization, 2021. Main Track. 2

[58] Xiang Zhang, Wei Liao, Lei Yu, Wen Yang, and Gui-Song Xia. Event-based synthetic aperture imaging with a hybrid network. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14230–14239, 2021. 2

[59] Zhengxia Zou, Keyan Chen, Zhenwei Shi, Yuhong Guo, and Jieping Ye. Object detection in 20 years: A survey. Proceed ings ofthe IEEE, 111(3):257–276, 2023. 2

# Towards Reconstructing and Recognizing Objects behind Fragmented Occlusion using Video

Supplementary Material

## 6. Text Labels

For our robust recognition experiment in Sec. 4.2, we used the following text labels coming from a set of 24 diverse outdoor categories $L = \{ { \tt g o a t }$ , helicopter, swamp, drone, fireplace, train, camp, snowmobile, boat, cave, tent, rock, tree, lake, river, bridge, house, street, deer, fox, motorcycle, horse, car, airplane}.

![](images/b67791a45c257f858bfaaab0942e70baf2eef3cbe0dcff6bc47b6067d9744975.jpg)

![](images/80f8d99072b614c15e29c5272e120e3eeade7611b537cdcb09506d0059c58f47.jpg)  
(b) $\alpha = 0 . 0 \mathrm { p x }$

(a) $\alpha = - 5 . 0$ px  
![](images/be079a98620e8aa94df6cc7e0e14e521006e1ef4c58f212b01d91b88537eb81e.jpg)  
(c) $\alpha = 3 . 0 \mathrm { p x }$

![](images/844333f26dd68c52362a8956df83ec7c6b376a06c6ce5e639b10849e2aaa3470.jpg)  
(d) $\alpha = 6 . 0 \mathrm { p x }$

![](images/a0fa0ca6163dd6c56aa6e060827ef52ebfbd6f1e73d604cbf4cc70b4f2abb2f4.jpg)  
(e) $\alpha = 1 0 . 0 \mathrm { p x }$  
Figure 10. Park sequence histograms.

## 7. Measures

The mean-squared error (MSE) between two images X and Y is defined

$$
\operatorname { M S E } ( X , Y ) = \frac { 1 } { N } \sum _ { i , j } ( X ( i , j ) - Y ( i , j ) ) ^ { 2 } .\tag{27}
$$

The Pearson correlation coefficient between images X and $Y$ is computed by vectorizing X and $Y ,$ , i.e. $x =$

vec $( X ) , y = \operatorname { v e c } \left( Y \right)$ and then by computing their mean values $m _ { x }$ and $m _ { y }$ . The coefficient is then written as

$$
\mu _ { \mathrm { P } } = \frac { \sum ( x - m _ { x } ) ( y - m _ { y } ) } { \sqrt { \sum ( x - m _ { x } ) ^ { 2 } \sum ( y - m _ { y } ) ^ { 2 } } } .\tag{28}
$$

## 8. Histograms

We selected two sequences from Sec. 4 to show the histograms from integral images.

Fig. 10 shows the histograms of the integrated images depicted in Fig. 6 - Park sequence. The histograms’ spreads translate into the variances plotted in Fig. 7a. The integrated image at $\alpha = 0$ has largest variance (spread), the integrated image at $\alpha ~ = ~ 6 . 0$ has second-largest variance (spread) which relates to the person’s motion.

Moreover, Fig. 11 shows the histograms of the integrated images depicted in Fig. 9 - Fence sequence.

![](images/adbf0b84c2d86fd64234bf8f0f12eb47ee9a49b6f0e57c787a4914d6fe41b84b.jpg)  
(a) $\alpha = - 8 . 0 \mathrm { p x }$

![](images/02d4305f1516131d34a515fc259e57ee308733c22596226bf41a636a9d61de89.jpg)  
(b) α = 0.0 px

![](images/3e9df0160beb9ad524980ed0b0184778ba91163bac64b4be5cee082e96ee613a.jpg)  
(c) α = 21.5 px

![](images/739ee0e0dc4cd819f921043fd7a71a39987f9f03ed322d78c4704db4ddee4f0a.jpg)  
(d) $\alpha = 2 9 . 0 \mathrm { p x }$

![](images/1e46ade2d08e230357e9fa3ed28de2f7ebf1032925b8d2b725b1662c3e93abe0.jpg)  
(e) $\alpha = 3 2 . 5 \mathrm { p x }$  
Figure 11. Fence sequence histograms.

![](images/6c23ffd4cee70c9b60e8eeaa9f189de8d433eab7dce5fd4e3002296887a3046c.jpg)

![](images/6828ab79429c14637c458e8d0aea547fb42bfb3649575a1ad8ddce9e7ab3835b.jpg)

![](images/8bf3a2cbf3713cf4761e5ae16b036a6fc2e5e926057132239027f11c4f40d79d.jpg)  
(a) α = −5.0 px, β = 4.0 px

![](images/d3a62fbece0efc742ec0711ec589ed5adc5965e919ef28c785e77408f1bcce54.jpg)  
(b) α = −5.0 px, β = 5.0 px

![](images/6ea61cffffae7fc636da1304d8256b4eef77cda84681f1ac75028aac502bd45b.jpg)

![](images/6aedc2dc57476fae670feba910f8daba6034186dd791a02c7145bd6820d0ca9d.jpg)

![](images/db5571d619e8b40cb672285ef19c0a42041a1308feb18c9ba74c797a5ead3be0.jpg)  
(c) α = −5.0 px, β = 6.0 px

![](images/64d4cd3ab071f181385a5b7873d9d2c4497077e0741205cc6bcabd9e3d2dc99d.jpg)  
(d) α = 0.0 px, β = 0.0 px  
Figure 12. Evaluation of different motions. Top: The variance of (a) and (c) is larger or equal to (b), although (b) shows higher contrast than (a) and (c) and is the integrated image of the correct box’s motion. Bottom: This is verified by the histogram spreads. (d) illustrates the static occluder. The histogram spread for (d) is maximal (255) due to the chosen brightness values for the background and occluder.

## 9. Variance: Failure Case

Motion computation in Eq. (13) successfully works for all 300 synthetic image sequences in Sec. 4. However, when applied to real image sequences, the integrated sample variance becomes significantly larger than the variance associated with the occluded object. Consequently, the function f fails to attain its maximum value. This raises the question of the conditions under which motion computation may fail.

To study such conditions, we generated a sequence of 21 synthetic images, showing an artificial box under occlusion density $\rho ~ = ~ 4 9 . 3$ % (see Fig. 13). The box’s brightness value is mid-gray (128), the background white (255), and the occluder black (0). Fig. 12 illustrates the results. Motion computation fails under these conditions, because the range of the average brightness values within the integrated image includes the brightness value of the box object. This can be nicely seen by the histograms’ spreads for the different motion parameters α and β in Eq. (19).

We conducted another experiment with the Park sequence, computing the sample variance for different fixation windows. These windows vary in size and capture the object or parts of it. Fig. 14 shows the results of this experiment. The sample variance is similar across window sizes for incorrect object motions. However, for small windows with larger object regions, the sample variance suddenly increases for the correct motion $\alpha = 6 . 0$

We conclude that motion computation fails if the object brightness values are a subset of the integrated image. This is because there are no distinctive brightness values to increase variance, so f does not attain a maximum for the correct motion. This is a fundamental limitation of variance in our algorithm. This may also explain Vaish et al. ’s observation that shape-from-focus (SfF) performs better. SfF exploits texture by incorporating image gradients instead of brightness values, which leads to better results [49].

![](images/c33e7335a4a3bc0c3f9694b1f711f7afe9713bacd7e735174417d8bc76818ab7.jpg)  
(a) I<sup>¯</sup><sub>r</sub>

![](images/578da28211c5f6a0f90315242338cc15b8f98f11534bf249030b937e6e237bff.jpg)  
(b) I<sub>r</sub>

![](images/bdad36eea2294d65fdcb8acb3e979c1a71b0f6bc1d12f47473cd6406ebd381ed.jpg)  
(c) f  
Figure 13. Box experiment: (a) Ground truth reference image. (b) Sequence’s reference image at reference r = 11. (c) 2D visualization of f - β (abscissa) vs. α (ordinate): Colors encode the variance. The motion at $( \alpha \beta ) ^ { \top } = { \bf 0 }$ is found by the algorithm (+). f attains no maximum for the box’s motion $\left( - 5 5 \right) ^ { \top }$

## 10. Multiple Objects

This experiment tests the algorithm’s behavior with six cartoon styles: Ape, Man, Crabby, Biker, Rake, and Fish. Unlike the synthetic experiment with a single Ape cartoon in Sec. 4, a fixed occlusion pattern is used for 90 generated sequences (15 per object and level).

In this experiment, we also use the peak signal-to-noise ratio (PSNR), a widely used image quality measure, in addition to the provided metrics:

![](images/76769511b554eb0cdb7d76fdc392cf3fa6468ad40ae9f0a3ec9cec55fea7c42b.jpg)

(a) Fixation windows  
![](images/c17a7a359c36d0f789c12a283fe003ee1a3c4ab52cfed0f55c4df085fa58f087.jpg)  
(b) Sample variance vs. window diagonal (px)  
Figure 14. Evaluation of the fixation window size: (a) Windows of varying size. (b) Sample variance computed in the integrated image for the different windows. Note that the sample variance for the correct motion is significantly higher than for the incorrect motions when the fixation window closely catches the object.

$$
\mu _ { \mathrm { P S N R } } = 1 0 \log \frac { 2 5 5 ^ { 2 } } { \mathrm { M S E } ( M _ { r } \hat { I } _ { r } , M _ { r } \bar { I } _ { r } ) } .\tag{29}
$$

We found that the algorithm successfully reconstructs cartoon objects in all 90 sequences (see Figs. 15a to 15d). Evaluation of µ<sub>PSNR</sub> shows no correlation between intensity or shape and reconstruction quality (see Fig. 15f). The reconstruction mainly depends on the deviation of object brightness from the occluder brightness (see additionally Fig. 15g). This deviation is computed as $\mathrm { R M S E } = \sqrt { \mathrm { M S E } }$ between object and occluder intensities. Smaller deviations result in a higher $\mu _ { \mathrm { P S N R } }$ because occluder intensities have a lesser impact on the averaging, which demonstrates the well known sensitivity of averaging (or the mean) to outliers.

![](images/2337cb6fcf00ade3f43f10f28ab1a0486cae61d731b02f04a32ab3335c32e099.jpg)  
(a) 48 %  
(b) 67.8 %  
(c) 87.8 %  
(e) 90.8 %  
(d) 90.1 %

![](images/9b085236c38f291950eb4978ab2e41c538afe6b48ea84b0d3130aeee1defcc91.jpg)  
(f) µ<sub>PSNR</sub>

![](images/2f8b6d86bc546a5b4dc04104361c13dd2d84eea723ea974bb9c1aca29380044c.jpg)  
(g) Color deviation  
Figure 15. Experiment with multiple objects: (a) - (e) For varying occlusion densities, the images show the ground truth I<sup>¯</sup><sub>r</sub> (top), raw data I<sub>r</sub> (middle) and reconstructed images I<sup>ˆ</sup><sub>r</sub> (bottom). (f) Plot of PSNR (dB) vs. occlusion density. Colors are the same as in (g) Deviation (RMSE) between object and occluder intensities. Colors encode the six objects.

## 11. Videos, Data and Code

We provide the following four videos to aid readers in understanding the algorithm in Sec. 3: park-sweep.mp4, boreal-sweep.mp4, pan-sweep.mp4, fence-sweep.mp4. The videos show the integrated images and illustrate a sweep through the motion parameter α in Eq. (19).

For each real sequence and the Box experiment (see Sec. 9), we additionally provide two zip files: one with raw input image sequences and the other with intermediate results (histograms, integrated images). We do not provide data for the synthetic experiment and the multiple objects in Sec. 10, because adding this data exceeds the supplemental material size limit. Upon acceptance, we will provide this data and the algorithm implementation to the reader.