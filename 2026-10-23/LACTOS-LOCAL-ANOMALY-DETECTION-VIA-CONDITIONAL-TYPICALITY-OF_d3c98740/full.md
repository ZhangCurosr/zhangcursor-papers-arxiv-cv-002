# LACTOS: LOCAL ANOMALY DETECTION VIA CONDITIONAL TYPICALITY OF SUPERPIXELS

Jihoon Hong<sup>1∗</sup>, Alireza Kheirandish<sup>1∗</sup>, Rohan Sanda<sup>2,3</sup> & Sara Fridovich-Keil<sup>1</sup>

<sup>1</sup>Georgia Institute of Technology, Atlanta, GA, USA

<sup>2</sup>Stanford University, Stanford, CA, USA

<sup>3</sup>Cohere

{jhong392,akheirandish3,sfk}@gatech.edu rsanda@stanford.edu

## ABSTRACT

Local anomaly detection is the task of identifying spatially confined anomalous or out-of-distribution (OOD) regions within an otherwise in-distribution (ID) image. Most existing methods are designed for a specific context, with a strong focus on industrial defect detection, where they achieve excellent performance on well-established benchmarks. However, we observe that these methods can exhibit significant performance drops in other contexts, especially when the inlier distribution is multimodal. To account for this inlier complexity, we propose LAC-ToS, a principled local OOD detection method based on statistical typicality and made tractable through two innovations: (i) grouping pixels into superpixels and (ii) using a computationally efficient proxy to estimate typicality. We evaluate LACToS across three diverse applications spanning medical imaging, human faces, and Xray security screening. LACToS performs more robustly than prior methods across these benchmarks. The code is released here: https://github.com/voilalab/LACToS.

## 1 INTRODUCTION

Local anomaly detection is the task of identifying out-of-distribution (OOD) regions that are spatially confined within an otherwise in-distribution (ID) image. It is central to applications such as industrial defect inspection, medical image analysis, and security screening, where an anomalous region may occupy only a small fraction of the image yet indicate a defect, tumor, or weapon. Unlike imagelevel OOD detection, which assigns a single anomaly score to an entire image, localized pixel-level anomaly detection involves generating a dense heatmap that indicates which pixels or regions of each outlier image are inconsistent with the inlier distribution. This makes the problem espe-

![](images/ed7e00f475206a037a6d82b20c119ec558199e1e495f8f6b94e82b30507190be.jpg)  
Figure 1: Using a diffusion model pretrained on ID images only, LACToS is able to identify locally OOD regions in images across various tasks, including tumors in CT scans, anomalies such as scars on faces, and harmful objects such as scissors in X-ray baggage scans.

cially challenging: a local OOD detector must be sensitive to subtle local deviations while remaining robust to natural variation in normal image content.

Prior work on localized anomaly detection has largely followed representation-based and reconstruction-based approaches. Representation-based methods extract local features from normal (inlier) training images, and score test patches using their distance from the learned distribution or their likelihood under it (Cohen and Hoshen, 2020; Defard et al., 2021; Roth et al., 2022). Reconstructionbased methods instead reconstruct, inpaint, or denoise the target image under a prior for inlier images, and use the discrepancy between the input and its reconstruction as an anomaly signal (Mousakhan et al., 2024; Wu et al., 2024). Although both approaches have achieved strong empirical performance, especially for industrial defect detection (Bergmann et al., 2019), their anomaly scores are often heuristic: representation-based methods depend on the choice of feature and distance metric, while reconstruction-based methods typically collapse the distribution of inlier completions into a point estimate. Conformal and risk-control methods can provide statistical guarantees for calibrated segmentation outputs (Shen and Liu, 2025), but they generally rely on a large amount of OOD calibration data. Moreover, we empirically observe that these methods do not always generalize reliably beyond industrial defect detection.

We present LACToS, a statistical method for local OOD detection that is designed to address these limitations of existing methods. Our approach is fundamentally reconstruction-based, but uses the key insight that the distribution of the reconstructions contains valuable information that is lost when reconstruction error is computed using a single sample or an aggregated point estimate. To leverage this richer distributional information towards improved local anomaly detection, we propose an anomaly metric based on statistical typicality, a measure of how representative a sequence of observations is for a given distribution. Our use of typicality is inspired by previous typicality-based OOD detection methods (Nalisnick et al., 2019; Osada et al., 2023; Barkley et al., 2026). However, these works are focused on image-level OOD detection where an entire image is classified as OOD, and to the best of our knowledge, using typicality for local anomaly detection has not yet been explored.

We begin with the typicality score of Nalisnick et al. (2019), and adapt it to identify local rather than whole-image anomalies (Section 4.1). Instead of asking whether an entire image is typical with respect to the unconditional prior distribution, we ask whether each pixel is typical with respect to the inpainting posterior conditioned on all other pixels in the image. However, this substitution induces three major challenges: (i) failure to detect larger local anomalies that span multiple contiguous pixels, (ii) loss of statistical validity, because typicality testing requires sampling a sequence of i.i.d. observations, and (iii) prohibitive computational cost of performing posterior sampling separately for each pixel.

LACToS tackles these challenges through two innovations: we partition images into superpixels and replace pixel-level typicality with superpixel-level typicality (Section 4.2), and we use a single denoising posterior as a common proxy for all inpainting posteriors. Our use of superpixels addresses challenges (i) and (ii), because a local OOD feature is more likely to be captured by a superpixel than a single pixel, and the individual pixels in each superpixel can be treated as independent observations to evaluate superpixel-level typicality. Our use of denoising rather than inpainting addresses challenge (iii), by dramatically reducing the number of posterior samples needed to estimate typicality. Through these innovations, we show that LACToS achieves strong quantitative and qualitative performance across a wide range of local OOD detection tasks including detecting tumors in CT scans, detecting unusual facial accessories in photographs, and detecting weapons in X-ray baggage scans.

To summarize, we propose LACToS, a statistical approach to local OOD detection in images using an inlier diffusion model and typicality testing. Our method (1) is grounded in statistical typicality, and (2) is competitive with, and often more precise than, existing methods across diverse local OOD detection tasks with complex inlier distributions.

## 2 PRELIMINARIES

In LACToS, we treat a diffusion model as an oracle denoiser that samples images consistent with its training distribution and a noisy input image. We provide a brief introduction to diffusion models and their use in posterior sampling in Appendix A. Here, we provide context on local OOD detection methods proposed in prior work, some of which inspired our use of statistical typicality in LACToS.

OOD detection is studied at both the image and pixel (local) levels. Image-level OOD detection assigns a single anomaly score to an entire image, with the goal of identifying images that are not drawn from the inlier distribution seen during training. In contrast, we focus on pixel-level OOD detection, in which the goal is to produce a dense anomaly heatmap to localize anomalous regions within each image. Formally, given an image $x ^ { \star }$ , the goal is to estimate the ground truth anomaly mask $m ^ { \star }$ . Previous works (Mousakhan et al., 2024; Wyatt et al., 2022; Defard et al., 2021) define an anomaly score $f ( x ^ { \star } , j )$ that captures how likely each pixel $j \in \mathcal I$ is to be OOD, where $\mathcal { I }$ is an index set of size $W \stackrel { \cdot } { \times } H$ over all pixels. Then, they compute an anomaly heatmap A with elements $A _ { j } = f ( x ^ { \star } , j )$ for each pixel, fix some threshold $\tau ,$ , and set $\hat { m } = \mathbb { 1 } \{ A \geq \tau \}$ . Many of these works, especially those using diffusion models, are reconstruction-based: they generate an in-distribution reconstruction of the input using posterior sampling and use discrepancies such as $\ell _ { 1 } , \ell _ { 2 } ,$ SSIM (Wang et al., 2004), LPIPS (Zhang et al., 2018), or feature-space distances as anomaly scores (Zavrtanik et al., 2021; Wu et al., 2024; Mousakhan et al., 2024; Pirnay and Chai, 2022). Although effective, these methods often collapse the distribution of plausible reconstructions into a first-order statistic, such as the mean reconstruction error, which does not capture the full reconstruction distribution and can reduce robustness to complex inlier distributions.

Meanwhile, distribution-aware statistics have been explored across several image-level OOD detection methods. For example, EigenScore (Shoushtari et al., 2025) uses the eigenvalue spectrum of the posterior covariance, and DDPM-OOD (Graham et al., 2023) aggregates standardized reconstruction statistics. There are also typicality-based methods that compare the average negative log-likelihood of the samples with the entropy of the inlier distribution (Nalisnick et al., 2019; Osada et al., 2023; Barkley et al., 2026). Specifically, let $( x ^ { ( 1 ) } , \dots , x ^ { ( N ) } )$ be independent draws from the inlier distribution $p .$ The (ϵ, N)-typical set is defined as

$$
\mathcal T _ { \epsilon , N } ^ { p } = \left\{ ( \boldsymbol x ^ { ( 1 ) } , \ldots , \boldsymbol x ^ { ( N ) } ) \ : \left| \ : - \frac { 1 } { N } \log p ( \boldsymbol x ^ { ( 1 ) } , \ldots , \boldsymbol x ^ { ( N ) } ) - \mathbb H ( p ) \right| \le \epsilon \right\} ,\tag{1}
$$

where $\mathbb { H } ( p ) = \mathbb { E } _ { x \sim p } [ - \log p ( x ) ]$ is the entropy of $p .$ By the Asymptotic Equipartition Property (Cover et al., 1991),

$$
- { \frac { 1 } { N } } \log p ( X ^ { ( 1 ) } , \ldots , X ^ { ( N ) } ) \xrightarrow [ N \to \infty ] { } \mathbb { H } ( p ) .\tag{2}
$$

For image-level OOD detection, Nalisnick et al. (2019) estimate the deviation from typicality as

$$
\hat { \epsilon } ( x ^ { ( 1 ) } , \ldots , x ^ { ( N ) } ) : = \left| - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log p ( x ^ { ( n ) } ) - \mathbb { H } ( p ) \right| ,\tag{3}
$$

and classify a set of samples as OOD when ϵˆ exceeds a threshold. Typicality has been shown to be a more effective image-level OOD detection metric than likelihood, especially in high dimensions (Nalisnick et al., 2019; Osada et al., 2023).

## 3 MOTIVATION

The local OOD detection problem has been extensively studied in the context of industrial defect detection. Prior works such as PatchCore (Roth et al., 2022), EfficientAD (Batzner et al., 2024), GLASS (Chen et al., 2024), and diffusion-based DDAD (Mousakhan et al., 2024) achieve excellent performance on widely used benchmarks such as MVTec AD (Bergmann et al., 2019) and VisA (Zou et al., 2022). However, we observe that these methods do not extend robustly to other OOD detection tasks, such as tumor detection, facial anomaly detection, and security screening using X-ray images, as shown in Table 1. One potential cause of this brittleness is the complexity of the inlier distribution. Industrial defect-detection datasets often have very structured inlier distributions, as products without defects are visually near-homogeneous, whereas CT scans, images of human faces, and X-ray baggage scans are far more heterogeneous even in the absence of anomalies. Moreover, defects in products are context-agnostic: a defective feature, such as a crack or fracture, is anomalous regardless of where in the image or product it occurs. In contrast, a dark region in a CT scan may be innocuous normal anatomy in one location, or indicative of a tumor in another.

Specifically, when the conditional distribution over a local region given the rest of the image is multimodal, methods that rely on a first-order statistic may erroneously identify samples from a different inlier mode as anomalous. To illustrate this phenomenon through a toy experiment, we construct four inlier distributions with increasing complexity levels, and train a diffusion model on each inlier distribution. Each image contains four shapes, one per quadrant. In the upper-right quadrant, every inlier image contains a circle, which allows us to define all other shapes placed in that quadrant as local anomalies. The other three quadrants can contain a wider variety of shapes, as shown in the top row of Figure 2, with the candidate shapes at each level occurring with equal probability. From level 1 to level 4, the set of shapes that can be placed in these three quadrants progressively grows, which increases the number of modes of the inlier distribution. We then introduce local anomalies in test images by replacing the top-right circle with a square, and evaluate two baseline methods, DDAD (Mousakhan et al., 2024) and PatchCore (Roth et al., 2022), on how well they can correctly identify only the square as an anomaly. The bottom row of Figure 2 demonstrates that at a per-image true positive rate (TPR) of 95%, the false positive rate (FPR) increases with inlier multimodality for both baseline methods. This trend holds across the TPR values shown in Figure 7. For more information on the experimental setup, please refer to Section B.

![](images/bf896fcde2b5743d0056725044ca6d5c760763e3b58b1f185731432d7044539c.jpg)

## 4 METHOD

To tackle this problem, we propose LAC-ToS, a typicality-based pixel-level OOD detection method using diffusion models, which is much more robust to inlier multimodality. In Section 4.1, we first describe an idealized extension of an existing typicality-based metric for image-level OOD detection to pixel-level anomaly localization, and several issues that limit its direct application. Then, in Section 4.2 and Section 4.3 we introduce two key techniques, superpixels and a denoising proxy, that help overcome these issues and allow for an effective and tractable approximation to pixel-level typicality. In Section 4.4, we additionally incorporate extracted pixel-level features which can provide more informative statistical context than color alone.

Throughout, we use $x ^ { \star }$ to denote the test image and subscripts to identify specific pixels or superpixels. Similarly, we use x (again with appropriate subscripts) to de

![](images/aa72a38057cc5940ab7f099218a01288dcbcaa8fd779cb2867c45ef02ecee6b5.jpg)  
Figure 2: Multimodality Toy Experiment. (Top) Images from the inlier distributions across different complexity levels. The top-right corner always contains a circle, while other corners can contain different shapes, with a wider variety of shapes at higher levels. (Bottom) False positive rates at true positive rate of 95%, evaluated on images that contain anomalous squares in the top-right corner.

note the corresponding random variable under the prior or posterior, with Monte Carlo realizations indexed by superscripts.

## 4.1 TYPICALITY-BASED ANOMALY SCORE

A natural extension of the existing typicality-based OOD detection metric in Equation (3) from images to pixels is conditional typicality. It is defined as

$$
f ^ { \operatorname* { i n p } } ( x ^ { \star } , j ) : = \big | - \log p ( x _ { j } ^ { \star } | x _ { - j } ^ { \star } ) { - } \mathbb { H } ( p ( x _ { j } | x _ { - j } ^ { \star } ) ) \big | , \qquad j \in \mathcal { I } ,\tag{4}
$$

where the superscript inp stands for inpainting, $\boldsymbol { x } _ { j } ^ { \star }$ and $x _ { - j } ^ { \star }$ are the values of pixel $j$ and all remaining pixels in the test image and all other pixels, respectively, and $p ( x _ { j } | x _ { - j } ^ { \star } )$ is the distribution of priorconsistent values for pixel $j$ when all other pixel values are fixed at their observed values $x _ { - j } ^ { \star } .$ Intuitively, we define a pixel in $x ^ { \star }$ to be OOD if it does not belong to the typical set with respect to this conditional data distribution $p ( x _ { j } | x _ { - j } ^ { \star } )$ , given all other pixel values.

Unfortunately, this direct extension of typicality to conditional distributions in Equation (4) is not a practical score for anomaly detection for the following reasons. (P1) Due to the spatial locality of pixel intensities and the smoothness of natural images, the value of a pixel given all others is often strongly constrained by nearby pixels. With conditional distributions that are concentrated on a tiny set determined by the neighboring pixels, the typicality score is unlikely to capture OOD features that span multiple adjacent pixels. (P2) In addition, a single pixel value is a sequence of length 1, which is insufficient for the theoretical guarantees of typical sets (Equation (2)) to hold. Indeed, the typicality test of Nalisnick et al. (2019) has been shown to exhibit poor empirical performance on single samples (Chauhan et al., 2022). (P3) It is also computationally demanding to compute the anomaly heatmap A by approximating Equation (4), as $f ^ { \mathrm { i n p } } ( x ^ { \star } , j )$ has to be evaluated for every pixel $j \in \mathcal I$ . Using N posterior samples for a Monte Carlo estimate $\hat { p } ( x _ { j } | x _ { - j } ^ { \star } )$ of $p ( x _ { j } | x _ { - j } ^ { \star } )$ for each pixel would require a total of $N \times | \mathcal { I } |$ | posterior samples, which is computationally prohibitive at practical image resolution H, W. (P4) Finally, raw RGB values often do not capture sufficient information about local image contents, including semantic information, which can lead to subpar performance in anomaly detection tasks.

## 4.2 SUPERPIXEL-LEVEL TYPICALITY

Our first key insight is that superpixels (Ren and Malik, 2003; Achanta et al., 2012; Stutz et al., 2018), groups of spatially contiguous pixels that share similar color, can help us address (P1) and (P2), by grouping OOD pixels together into small groups that are easier to detect and can be treated as a sequence to estimate typicality. The latter requires Assumption 4.1, which captures the intuition that, within each superpixel, individual pixel values are independent and identically distributed. An example of superpixels is visualized in Figure 3.

![](images/a5009f61abf91513a3f224cdc60e7ebbf67d23f5eb79c137b106c465a43432ce.jpg)

Concretely, we define $\mathcal { S } ( x ^ { \star } ) = \{ S _ { k } \subset \mathcal { I } \mid k \in \mathcal { K } \}$ to be a finegrained superpixel partition of the pixels in ${ \mathcal { I } } ,$ , which can be obtained using off-the-shelf superpixel algorithms (Achanta et al., 2012; Van den Bergh et al., 2012; Liu et al., 2011; Li and Chen, 2015). Here k indexes superpixels and $S _ { k }$ is the set of pixels in superpixel

Figure 3: A fine-grained superpixel decomposition of an example image.

k. Let $x _ { S _ { k } }$ denote image pixels in superpixel $k ,$ and $x _ { - S _ { k } }$ denote all other pixels.

Assumption 4.1. [Conditionally i.i.d. pixel values within each superpixel] Given $x ^ { \star }$ and $\boldsymbol { \mathcal { S } } ( \boldsymbol { x } ^ { \star } )$ , there exists a single-pixel distribution $q _ { k } ^ { \mathrm { i n p } }$ associated with each superpixel $k \in \mathcal { K }$ such that

$$
x _ { j } \mid x _ { - S _ { k } } ^ { \star } \stackrel { i i d } { \sim } q _ { k } ^ { \mathrm { i n p } } \quad \forall j \in S _ { k } .
$$

Under Assumption 4.1, we define the superpixel-level typicality score $g ^ { \mathrm { i n p } }$ as

$$
g ^ { \mathrm { i n p } } ( x ^ { \star } , k ) = \frac { 1 } { | S _ { k } | } \Big | - \log p ( x _ { S _ { k } } ^ { \star } | x _ { - S _ { k } } ^ { \star } ) - \mathbb { H } ( p ( x _ { S _ { k } } | x _ { - S _ { k } } ^ { \star } ) ) \Big | = \left| - \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \log { q _ { k } ^ { \mathrm { i n p } } ( x _ { j } ^ { \star } ) } - \mathbb { H } ( q _ { k } ^ { \mathrm { i n p } } ) \right| .\tag{5}
$$

where $p ( x _ { S _ { k } } | x _ { - S _ { k } } ^ { \star } )$ is the conditional distribution of the superpixel given all other pixel values, and the last equality holds due to the conditional independence assumption (Assumption 4.1) on the pixels in $\bar { S } _ { k }$ . We can then compute the pixel-level anomaly heatmap A by estimating the typicality score in Equation (5) using Monte Carlo sampling. For each superpixel k, this involves drawing M samples $x ^ { ( 1 ) } , \ldots , x ^ { ( M ) }$ from the inpainting distribution $p _ { k } ^ { \mathrm { i n p } } : = p ( x \mid x _ { - S _ { k } } ^ { \star } )$ , extracting pixels $\{ x _ { j } ^ { ( m ) } \} _ { j \in S _ { k } , m \in [ M ] }$ , fitting a histogram $\hat { q } _ { k } ^ { \mathrm { i n p } }$ to estimate $q _ { k } ^ { \mathrm { i n p } }$ , and computing the estimate $\hat { g } ^ { \mathrm { i n p } } ( x ^ { \star } , k )$ Finally, for pixel-level scores it is natural to duplicate the same score over all pixels $j \in S _ { k }$ , i.e. $\hat { f } ^ { \mathrm { i n p } } ( x ^ { \star } , j ) = \hat { g } ^ { \mathrm { i n p } } ( x ^ { \star } , k )$ for all $j \in S _ { k }$ , and apply Gaussian smoothing to smooth out the hard boundaries between superpixels.

Estimating the superpixel-level typicality score $g ^ { \mathrm { i n p } }$ rather than the pixel-level typicality score $f ^ { \mathrm { i n p } }$ helps address these problems. (P1) By inpainting a group of similar pixels instead of a single pixel, $g ^ { \mathrm { i n p } }$ can better identify local OOD features that span multiple pixels than $f ^ { \mathrm { i n p } }$ . (P2) Under Assumption 4.1, we are given a sequence of $| S _ { k }$ | independent observations $\{ x _ { j } ^ { \star } \} _ { j \in S _ { k } }$ for $q _ { k } ^ { \mathrm { i n p } }$ , which facilitates computing a typicality score for each superpixel. (P3) In terms of computational cost, the number of anomaly scores to evaluate is reduced from the number of pixels $| \mathcal { I }$ | to the number of superpixels $| \kappa |$ . We can also use a much smaller number of samples $\bar { M } \ll \dot { N }$ to compute the histogram $\hat { q } _ { k }$ for $q _ { k } ^ { \mathrm { i n p } }$ , since each posterior sample from $p _ { k } ^ { \mathrm { i n p } }$ translates to $\lvert S _ { k } \rvert$ samples from $q _ { k } ^ { \mathrm { i n p } }$

![](images/b8f9c0db8f29c832d9f7c53f5cb12e4c0fc29d4f2c2c5e348c191486df886e5a.jpg)  
Figure 4: LACToS Overview. (1) Posterior Sampling. We sample noise independently from an isotropic Gaussian distribution and add it to the test image $x ^ { \star }$ to create noised versions of $x ^ { \star }$ denoted by $y ^ { m }$ . Then, using diffusion-based posterior sampling methods, we sample reconstructions $x ^ { m } \sim p ( \dot { x } | \dot { y } ^ { m } )$ ). (2) Feature Extraction. For $x ^ { \star }$ and each sample $\boldsymbol { x } ^ { m }$ , we pass the image through a pretrained ResNet model, extract activations, and compress them using an encoder to create a feature vector of dimension $C _ { f } ^ { \prime } + C$ for each pixel, where $C$ is the original channel dimension. (3) Histogram Fitting. For a given superpixel k, we gather the features of the reconstructed pixels in the superpixel, and construct $\hat { q } _ { k } ^ { \mathrm { d e n } } , \mathrm { \dot { a } } \hat { C _ { f } ^ { \prime } } + C$ -dimensional histogram. (4) Typicality Computation. Finally, using the fitted histogram, we compute the typicality of the pixel features in $x ^ { \star }$ and report the anomaly heatmap A.

## 4.3 DENOISING POSTERIOR

Although superpixel-level typicality reduces the required number of posterior samples from $| { \mathcal { I } } | N _ { }$ to $| \mathcal { K } | M$ , the cost remains prohitive for fine superpixel partitions. For instance, Figure 3 shows a $2 5 6 \times 2 5 6$ image partitioned into 2,600 superpixels (approximately 25 pixels per superpixel on average), which would require a total of at least 2600 posterior samples using a diffusion model. This cost could be reduced by partitioning the image into fewer, larger superpixels, but these may struggle to localize small anomalies and fail to satisfy Assumption 4.1.

We overcome this problem (P3) by estimating $q _ { k } ^ { \mathrm { i n p } }$ for all $k \in \mathcal { K }$ simultaneously using a small number of posterior samples. Instead of using different samples from each of the inpainting conditional distributions $\{ p _ { k } ^ { \mathrm { i n p } } \} _ { k \in \mathcal K }$ , we use the same M samples from the denoising conditional distribution, defined as

$$
p ^ { \mathrm { d e n } } ( x ) : = \mathbb { E } _ { \eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) } \big [ p ( x \mid x ^ { \star } + \eta ) \big ] .\tag{6}
$$

For each $S _ { k }$ , we pool the pixel values across these samples to form $\{ x _ { j } ^ { m } \} _ { j \in S _ { k } , m \in [ M ] }$ , which we use to fit a histogram for $q _ { k } ^ { \mathrm { i n p } }$ , and compute the anomaly score.

Intuitively, $p ^ { \mathrm { d e n } }$ serves as a shared relaxation of the superpixel-specific inpainting posteriors $\{ p _ { k } ^ { \mathrm { i n p } } \} _ { k \in \mathcal K }$ . In each $p _ { k } ^ { \mathrm { i n p } }$ , the complement $x _ { - S _ { k } } ^ { \star }$ is fully observed while $x _ { S _ { k } } ^ { \star }$ is left unobserved. For the denoising posterior $p ^ { \mathrm { d e n } }$ , there is a tension between information on $x _ { - S _ { k } } ^ { \star }$ and lack of information on $x _ { S _ { k } } ^ { \star }$ , controlled by the noise level σ. With increasing σ, we lose more information about $x _ { S _ { k } } ^ { \star }$ which is desirable, but we also lose information about the context $x _ { - S _ { k } } ^ { \star }$ . The following theorem informally bounds the error introduced by replacing the inpainting posterior with the denoising posterior. Appendix section D states and proves the formal result and gives bounds without Assumption 4.1.

Theorem 4.1. (Informal). Let $g ^ { \mathrm { i n p } } ( x ^ { \star } , k )$ and $g _ { \sigma } ^ { \mathrm { d e n } } ( x ^ { \star } , k )$ be the anomaly scores obtained using the ideal inpainting distribution and the proxy denoising distribution with a noise distribution of ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } I )$ , respectively. Under appropriate assumptions, the error can be bounded as

$$
\left| g ^ { \mathrm { i n p } } ( x ^ { \star } , k ) - g _ { \sigma } ^ { \mathrm { d e n } } ( x ^ { \star } , k ) \right| \leq \alpha _ { k } \sigma + \beta _ { k } \sigma ^ { - \gamma _ { k } }
$$

for some positive constants $\alpha _ { k } , \beta _ { k }$ , and $\gamma _ { k }$ .

## 4.4 FEATURE AUGMENTATION

One final problem remains (P4): RGB values alone may not distinguish local ID and OOD features. To address this issue, (P4) we follow the convention of previous works (Mousakhan et al., 2024; Roth et al., 2022; Defard et al., 2021) and augment RGB values with additional features. Specifically, we extract the activations of the last three layers of a pretrained ResNet (He et al., 2016) ϕ, as later layers of convolutional neural networks are known to encode higher-level visual structure and have been widely used as transferable representations for downstream vision tasks (Razavian et al., 2014; Zeiler and Fergus, 2014). The high activation dimension, denoted as $C _ { f }$ , can increase histogram estimation error when few samples are available. We therefore use an autoencoder trained on each ID dataset to compress the features to dimension $C _ { f } ^ { \prime } ( \ll C _ { f } ) $ ; in our experiments we use $C _ { f } ^ { \prime } = 3$

Algorithm Summary. Our complete algorithm, illustrated in Figure 4, can be summarized as follows. First, we generate M samples from the denoising conditional distribution $p ^ { \mathrm { d e n } }$ through posterior sampling using a diffusion model. Second, each of the samples is passed through ϕ to extract features, which are then compressed using the trained encoder E. Third, the pixel values and the compressed features are concatenated, resulting in

$$
u _ { j } ^ { m } : = ( x _ { j } ^ { m } , \pmb { \mathcal { E } } ( \phi ( x ^ { m } ) _ { j } ) ) \in \mathbb { R } ^ { C + C _ { f } ^ { \prime } }
$$

for each pixel j. Fourth, the augmented vectors $u _ { j } ^ { m }$ that belong to each superpixel $S _ { k }$ are collected to form $\{ u _ { j } ^ { m } \} _ { j \in S _ { k } , m \in [ M ] }$ and used to construct a histogram ${ \hat { q } } _ { k } ,$ an estimate of the $C + C _ { f } ^ { \prime }$ -dimensional distribution $q _ { k } ^ { \mathrm { i n p } }$ . The fourth step is repeated over all superpixels, after which we can compute the anomaly map by computing the typicality of $x ^ { \star }$ for each superpixel (Equation (5)), and finally apply spatial Gaussian smoothing to refine pixel-level anomaly scores across superpixel boundaries.

## 5 EVALUATION

Datasets. We evaluate LACToS on three local anomaly detection datasets spanning medical imaging, human faces, and X-ray security screening (Table 4). In each case, the diffusion model in LACToS is trained only on ID images, and all methods are evaluated on a held-out test set. Each test set includes an equal number of ID images and images containing local OOD features.

CHAOS (Kavur et al., 2019) provides abdominal CT scans of healthy patients. We train on 3,070 inlier slices and construct a test set containing 100 healthy test slices and 100 locally anomalous slices acquired by adding synthetic tumor to each. Following common practice in CT image analysis (Wang et al., 2021; Thummerer et al., 2023; Su et al., 2019), we apply the provided body mask at evaluation to exclude background pixels. FFHQ (Karras et al., 2019) is a dataset of high-quality face images, which we treat as the inlier distribution. We train on a random subset of $5 { , } 0 0 { \bar { 0 } }$ images, and manually curate an OOD test set of 50 photographs of movie and television characters with prosthetics or atypical facial features (e.g., scars, metal implants), verified not to overlap with FFHQ. The ID test set contains 50 randomly selected held-out FFHQ images. DVXray (Ma et al., 2024) is a dataset of X-ray luggage scans, with an ID set of weapon-free bags and an OOD set containing weapons. We train on 5,000 inlier images and restrict the OOD test set to the scissors category, where targets are small and spatially confined. For randomly sampled scissor images, we used SAM 3 (Carion et al., 2026) to convert the dataset’s bounding-box annotations into segmentation masks, manually discarding images with visibly inaccurate masks for a final OOD test set of 72 images. We construct an ID test set that also contains 72 images, none of which contains a weapon.

Baselines & Models. We compare LACToS against six strong baselines developed for industrial defect detection, spanning representation- and reconstruction-based methods. PatchCore (Roth et al., 2022) builds a coreset-subsampled memory bank of ID patch features from a frozen ImageNet pretrained backbone and scores test patches by nearest-neighbor distance. SimpleNet (Liu et al., 2023) learns a feature adapter and a discriminator that separates adapted ID features from synthetic anomalies generated by injecting Gaussian noise in feature space. SuperSimpleNet (Rolih et al., 2024) extends SimpleNet with architectural and training improvements for fast, reliable surface defect detection. EfficientAD (Batzner et al., 2024) combines a lightweight student–teacher model with an autoencoder to detect local structural anomalies and violations of global logical constraints through feature discrepancies. GLASS (Chen et al., 2024) trains a discriminator to separate ID features from synthetic anomalies generated using gradient ascent with truncated projection in feature space and local texture overlays in image space. DDAD (Mousakhan et al., 2024) conditions a denoising diffusion model on the test image to produce an ID reconstruction and scores anomalies by feature-space discrepancy.

EfficientAD  
Table 1: Pooled quantitative performance in local anomaly detection. AUROC and AP are computed over all pixels in each dataset jointly. AUPRO is computed over the dataset using per-region overlap for FPR ≤ 0.3. Higher is better for all metrics. The best result is shown in bold and the second-best is underlined.
<table><tr><td>Dataset</td><td>Metric</td><td>PatchCore</td><td></td><td>SimpleNet SuperSimpleNet EfficientAD</td><td></td><td>GLASS</td><td>DDAD</td><td>LACToS (Ours)</td></tr><tr><td rowspan="3">CHAOS</td><td>AUROC</td><td>0.9297</td><td>0.7464</td><td>0.3786</td><td>0.6664</td><td>0.8318</td><td>0.9707</td><td>0.9620</td></tr><tr><td>AP</td><td>0.0590</td><td>0.0193</td><td>0.0051</td><td>0.0105</td><td>0.0410</td><td>0.3159</td><td>0.4318</td></tr><tr><td>AUPRO</td><td>0.6811</td><td>0.4191</td><td>0.0332</td><td>0.2513</td><td>0.5530</td><td>0.8676</td><td>0.8764</td></tr><tr><td rowspan="3">FFHQ</td><td>AUROC</td><td>0.8491</td><td>0.7162</td><td>0.7647</td><td>0.6200</td><td>0.6707</td><td>0.7469</td><td>0.8515</td></tr><tr><td>AP</td><td>0.0798</td><td>0.0537</td><td>0.0772</td><td>0.0405</td><td>0.0345</td><td>0.0563</td><td>0.1177</td></tr><tr><td>AUPRO</td><td>0.4094</td><td>0.3174</td><td>0.4047</td><td>0.2365</td><td>0.2440</td><td>0.3083</td><td>0.5320</td></tr><tr><td rowspan="3">DVXray</td><td>AUROC</td><td>0.9640</td><td>0.9234</td><td>0.8688</td><td>0.9184</td><td>0.8944</td><td>0.8567</td><td>0.9607</td></tr><tr><td>AP</td><td>0.0545</td><td>0.0254</td><td>0.0165</td><td>0.0235</td><td>0.0155</td><td>0.0118</td><td>0.0568</td></tr><tr><td>AUPRO</td><td>0.8141</td><td>0.6577</td><td>0.5351</td><td>0.6592</td><td>0.6062</td><td>0.5183</td><td>0.8179</td></tr></table>

Image  
LACToS  
DDAD  
PatchCore  
GLASS  
![](images/e000ad909978c43b78edbc5c9286b93449efc1c71532d5c2fb8541dce1ad67a2.jpg)  
Figure 5: Anomaly heatmaps generated by each method on example images from each test dataset. Rows 1-2 (CHAOS): detect liver tumors in CT scans. Rows 3-4 (FFHQ): detect anomalous facial features. Rows 5-6 (DVXray): detect weapons (scissors) in X-ray baggage scans.

LACToS requires an unconditional diffusion model and a posterior sampler. For all benchmarks, we use the guided-diffusion U-Net of Dhariwal and Nichol (2021) with diffusion posterior sampling (DPS) (Chung et al., 2022) at inference time. Full experimental details are provided in Section E and in our code, which will be released upon publication.

Results. Each method produces a heatmap of anomaly scores per test image. Following standard practice, we pool pixel-level anomaly scores across all test images and evaluate three complementary pixel-level detection metrics. AUROC captures the relative ordering of heatmap values against the ground-truth anomaly mask, but can reward low-contrast heatmaps that flag many ID pixels as nearly as anomalous as the true OOD pixels. AP summarizes precision across recall levels, weighting each precision value by the corresponding increase in recall. AUPRO is the normalized area under the per-region overlap curve up to a false positive rate of 0.3, where per-region overlap is the fraction of correctly detected pixels within each connected ground-truth anomalous region, averaged equally across regions. When different anomalies have different sizes, AUPRO weights all anomalies equally rather than overweighting large anomalies.

Table 1 reports these metrics, which indicate that LACToS is competitive with baselines across all tasks, though the best method differs across metrics. The example heatmaps in Figure 5 also suggest that LACToS tends to be more precise: it correctly marks OOD pixels but rarely flags ID pixels, at the cost of occasionally missing part or all of the OOD region. The baselines instead produce blurrier heatmaps that recover the OOD region but also flag many ID pixels.

Ablation Studies. To verify the robustness of LACToS, we conduct ablation studies on several hyper-parameters, which we describe in detail in Section G. These include the number of posterior samples M and the average size of each superpixel. We observe that the performance of LACToS is mostly stable across different settings, demonstrating its robustness to hyperparameter choices. In addition, we perform an ablation on the use of autoencoder-based features, which shows that the features can substantially improve performance compared to typicality based solely on RGB color, though this improvement is dataset-dependent.

## 6 DISCUSSION

We present LACToS, a framework for general-purpose local anomaly detection. We leverage a pretrained diffusion model to represent the inlier distribution, and combine it with a statistical approach that scores image regions by their typicality (or rather, atypicality) under the conditional distribution induced by their surrounding context. Empirically, LACToS shows strong performance across diverse local OOD detection tasks ranging from medical imaging to security screening.

We note that since LACToS is more robust to inlier multimodality, it is also less likely to flag features from minority inlier modes as anomalous, especially when using a larger M. We demonstrate this behavior through a toy experiment, and refer interested readers to Section C. It is possible that some features annotated as anomalies in the benchmarks we evaluate may be similar to features present with low frequency in the inlier (training) datasets, causing LACToS to classify them as inliers and thus generate false negative predictions according to the dataset anomaly labels. We therefore advise practitioners that, in order to maximize the accuracy of LACToS, it is important to ensure that the training set does not contain the anomalies one desires to detect.

Limitations and future work. Since LACToS targets general-purpose local anomaly detection, specialized approaches can outperform LACToS for specific highly structured datasets, such as those common in industrial defect detection. Please refer to table 7 for evaluation on MVTec AD Bergmann et al. (2019). LACToS inherits the computational cost of diffusion posterior sampling, which is significantly higher per-image than feedforward representation-based methods such as PatchCore. LACToS also relies on Assumption 4.1, which may be violated when superpixels are large or when the anomalous region spans multiple superpixels with heterogeneous pixel statistics. LACToS requires several hyperparameters, such as the noise variance in the denoising posterior sampling, the quantization granularity and smoothing bandwidth in the histogram estimation, the number of posterior samples, the granularity of superpixels, and the spatial smoothing variance of the final heatmap. While we provide ablation studies showing LACToS is fairly robust to these hyperparameters, future work may find mechanisms to remove or automatically set optimal parameter values across tasks. Future work may also replace our histogram estimation with a learned or parametric density model, or extend our LACToS framework to video and 3D data types where spatiotemporal or volumetric superpixels could provide longer sequences for improved typicality estimation.

## 7 AKNOWLEDGEMENT

This work was supported in part by the HIVES program at Georgia Tech Research Institute (GTRI).

## REFERENCES

Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk. Slic superpixels compared to state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence, 34(11):2274–2282, 2012.

Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12 (3):313–326, 1982.

Brett Barkley, Preston Culbertson, and David Fridovich-Keil. SCOPED: Score-curvature out-of-distribution proximity evaluator for diffusion. International Conference on Learning Representations, 2026.

Kilian Batzner, Lars Heckler, and Rebecca König. Efficientad: Accurate visual anomaly detection at millisecondlevel latencies. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 127–137. IEEE, 2024.

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9592–9600, 2019.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2026. URL https://arxiv.org/abs/2511.16719.

Kushal Chauhan, Pradeep Shenoy, Manish Gupta, Devarajan Sridharan, et al. Robust outlier detection by de-biasing vae likelihoods. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9881–9890, 2022.

Qiyu Chen, Huiyuan Luo, Chengkan Lv, and Zhengtao Zhang. A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization. In European Conference on Computer Vision, pages 37–54. Springer, 2024.

Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.

Niv Cohen and Yedid Hoshen. Sub-image anomaly detection with deep pyramid correspondences. arXiv preprint arXiv:2005.02357, 2020.

Thomas M Cover, Joy A Thomas, and John Kieffer. Elements of information theory, volume 2. wiley New York, 1991.

Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. Padim: a patch distribution modeling framework for anomaly detection and localization. In International conference on pattern recognition, pages 475–489. Springer, 2021.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Mark S Graham, Walter HL Pinaya, Petru-Daniel Tudosiu, Parashkev Nachev, Sebastien Ourselin, and Jorge Cardoso. Denoising diffusion models for out-of-distribution detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2948–2957, 2023.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal ofMachine Learning Research, 6(4), 2005.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.

Ali Emre Kavur, M. Alper Selver, Oguz Dicle, Mustafa Barı¸s, and N. Sinem Gezer. CHAOS - Combined˘ (CT-MR) Healthy Abdominal Organ Segmentation Challenge Data, April 2019. URL https://doi.org/ 10.5281/zenodo.3362844.

Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in neural information processing systems, 35:23593–23606, 2022.

Zhengqin Li and Jiansheng Chen. Superpixel segmentation using linear spectral clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1356–1363, 2015.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Ming-Yu Liu, Oncel Tuzel, Srikumar Ramalingam, and Rama Chellappa. Entropy rate superpixel segmentation. In CVPR 2011, pages 2097–2104. IEEE, 2011.

Zhikang Liu, Yiming Zhou, Yuansheng Xu, and Zilei Wang. Simplenet: A simple network for image anomaly detection and localization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20402–20411, 2023.

Bowen Ma, Tong Jia, Mingyuan Li, Songsheng Wu, Hao Wang, and Dongyue Chen. Toward dual-view x-ray baggage inspection: A large-scale benchmark and adaptive hierarchical cross refinement for prohibited item discovery. IEEE Transactions on Information Forensics and Security, 19:3866–3878, 2024. doi: 10.1109/TIFS.2024.3372797.

Fernand Meyer and Serge Beucher. Morphological segmentation. Journal of visual communication and image representation, 1(1):21–46, 1990.

Arian Mousakhan, Thomas Brox, and Jawad Tayyub. Anomaly detection with conditioned denoising diffusion models. In DAGM German Conference on Pattern Recognition, pages 181–195. Springer, 2024.

Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, and Balaji Lakshminarayanan. Detecting out-of-distribution inputs to deep generative models using typicality. arXiv preprint arXiv:1906.02994, 2019.

Genki Osada, Tsubasa Takahashi, Budrul Ahsan, and Takashi Nishide. Out-of-distribution detection with reconstruction error and typicality-based penalty. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 5551–5563, 2023.

Jonathan Pirnay and Keng Chai. Inpainting transformer for anomaly detection. In International conference on image analysis and processing, pages 394–406. Springer, 2022.

Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, and Stefan Carlsson. CNN features off-the-shelf: An astounding baseline for recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 806–813, 2014.

Ren and Malik. Learning a classification model for segmentation. In Proceedings ninth IEEE international conference on computer vision, pages 10–17. IEEE, 2003.

Blaž Rolih, Matic Fucka, and Danijel Sko ˇ caj. Supersimplenet: Unifying unsupervised and supervised learningˇ for fast and reliable surface defect detection. In International Conference on Pattern Recognition, pages 47–65. Springer, 2024.

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14318–14328, 2022.

Cheng Shen and Yuewei Liu. Conformal segmentation in industrial surface defect detection with statistical guarantees. Mathematics, 13(15):2430, 2025.

Shirin Shoushtari, Yi Wang, Xiao Shi, M Salman Asif, and Ulugbek S Kamilov. Eigenscore: Ood detection using covariance in diffusion models. arXiv preprint arXiv:2510.07206, 2025.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. arXiv preprint arXiv:2111.08005, 2021.

David Stutz, Alexander Hermans, and Bastian Leibe. Superpixels: An evaluation of the state-of-the-art. Computer Vision and Image Understanding, 166:1–27, 2018.

Kuan-Hao Su, Harry T Friel, Jung-Wen Kuo, Rose Al Helo, Atallah Baydoun, Christian Stehning, Adina N Crisan, Melanie S Traughber, Ajit Devaraj, David W Jordan, et al. Ute-mdixon-based thorax synthetic ct generation. Medical physics, 46(8):3520–3531, 2019.

Adrian Thummerer, Erik Van der Bijl, Arthur Galapon Jr, Joost JC Verhoeff, Johannes A Langendijk, Stefan Both, Cornelis (Nico) AT van den Berg, and Matteo Maspero. Synthrad2023 grand challenge dataset: Generating synthetic ct for radiotherapy. Medical physics, 50(7):4664–4674, 2023.

Michael Van den Bergh, Xavier Boix, Gemma Roig, Benjamin De Capitani, and Luc Van Gool. Seeds: Superpixels extracted via energy-driven sampling. In European conference on computer vision, pages 13–26. Springer, 2012.

Tonghe Wang, Yang Lei, Yabo Fu, Jacob F Wynne, Walter J Curran, Tian Liu, and Xiaofeng Yang. A review on medical imaging synthesis using deep learning and its clinical applications. Journal of applied clinical medical physics, 22(1):11–36, 2021.

Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Di Wu, Shicai Fan, Xue Zhou, Li Yu, Yuzhong Deng, Jianxiao Zou, and Baihong Lin. Unsupervised anomaly detection via masked diffusion posterior sampling. arXiv preprint arXiv:2404.17900, 2024.

Julian Wyatt, Adam Leach, Sebastian M Schmon, and Chris G Willcocks. Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 650–656, 2022.

Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. Reconstruction by inpainting for visual anomaly detection.ˇ Pattern Recognition, 112:107706, 2021.

Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European conference on computer vision, pages 818–833. Springer, 2014.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. Spot-the-difference selfsupervised pre-training for anomaly detection and segmentation. In European conference on computer vision, pages 392–408. Springer, 2022.

## A BACKGROUND

## A.1 DIFFUSION MODELS

Diffusion models are generative models that have gained enormous popularity for their high sample quality Ho et al. (2020); Dhariwal and Nichol (2021). To sample from a desired data distribution $p ,$ these models first define a noising process (also called the forward process) as a stochastic differential equation (SDE) (Song et al., 2020) of the form

$$
d x = f ( x , t ) d t + g ( x , t ) d w , \qquad t = 0 \to 1 ,\tag{7}
$$

where $w$ is a standard Wiener process (Brownian motion) and $f , g$ are the drift and diffusion coefficients, respectively, which are determined by a chosen noise schedule. If we denote the marginal distribution of x at time t as $p _ { t }$ , this process transforms clean data samples $x \sim p _ { 0 } : = p$ across time until the distribution becomes pure noise at $t = 1 , \mathrm { i . e . } \ p _ { 1 } = \mathcal { N } ( 0 , I )$ . Once the forward process is defined, there exists, under regularity conditions, a corresponding backward SDE with drift $\tilde { f }$ and diffusion $\tilde { g }$ (Anderson, 1982; Song et al., 2020):

$$
d x = \tilde { f } ( x , t ) d t + \tilde { g } ( x , t ) d \bar { w } , \qquad t = 1 \to 0 ,\tag{8}
$$

where w¯ is also a Wiener process but with t indexed backwards. This backward SDE shares the same marginal distributions $p _ { t }$ as the marginal distributions induced by the forward process for all $t \in [ 0 , 1 ]$ . Since the drift coefficient $\tilde { f }$ of the backward SDE has the form

$$
\tilde { f } ( x , t ) = f ( x , t ) - g ( x , t ) ^ { 2 } \nabla _ { x } \log p _ { t } ( x ) ,\tag{9}
$$

a diffusion model is optimized on either a score-matching loss to predict $\nabla _ { x } \log p _ { t } ( x )$ Hyvärinen and Dayan (2005); Song et al. (2020) or on a flow-matching loss to predict $\tilde { f } ( x , t )$ directly Lipman et al. (2022), and used to iteratively sample from $p _ { t }$ through discretization of the backward SDE.

## A.2 POSTERIOR SAMPLING USING DIFFUSION MODELS

To condition generation on inference-time measurements using a pretrained diffusion model, prior works modify the reverse sampling process to approximately sample from the posterior distribution capturing simultaneous consistency with the prior encoded by the diffusion model and the likelihood induced by the measurement model Song et al. (2021); Kawar et al. (2022); Chung et al. (2022); Wang et al. (2022).

These methods decompose the conditional score using Bayes’ rule as

$$
\nabla _ { x } \log { p _ { t } ( x | y ) } = \nabla _ { x } \log { p _ { t } ( x ) } + \nabla _ { x } \log { p _ { t } ( y | x ) } ,\tag{10}
$$

where $\nabla _ { x } \log p _ { t } ( x )$ is the unconditional score, which is provided by the pretrained diffusion model, and $\nabla _ { x } \log p _ { t } ( y | x )$ is the likelihood score, which is typically approximated to promote consistency with $y$ while maintaining computational tractability. These works have shown strong empirical performance on various imaging inverse problems such as inpainting, denoising, deblurring, superresolution, and tomography; in all of these tasks the condition is given in the form of measurements $y = \mathcal { A } ( x ) + \eta$ for linear or non-linear observation model $\mathcal { A }$ and noise $\eta .$

## B TOY EXPERIMENT 1

Here we provide further details and results for the toy experiment conducted in Section 3. For each level of inlier complexity, we train a dedicated diffusion model on a set of inlier images that always contain a circle at the upper-right corner. The other 3 corners of the inlier images contain a shape randomly selected from the set of candidate shapes, with probabilities defined in Table 2. As the level increases, more shapes are equally probable at these three corners, thereby isolating the effect of increasing local multimodality in the inlier distribution. Some example images in each training set are shown in Figure 6.

Table 2: Toy Experiment 1 Inlier Distribution Specification.
<table><tr><td>Level</td><td>Set of Candidate Shapes</td><td>Probability</td></tr><tr><td>Level 1</td><td>{Circle, Square}</td><td>[0.5,0.5]</td></tr><tr><td>Level 2</td><td>{Circle, Square, Triangle}</td><td>[0.3, 0.3, 0.3]</td></tr><tr><td>Level 3</td><td>{Circle, Square, Triangle, Ellipse}</td><td>[0.25, 0.25, 0.25, 0.25]</td></tr><tr><td>Level 4</td><td>{Circle, Square, Triangle, Ellipse, Star}</td><td>[0.2, 0.2, 0.2, 0.2, 0.2]</td></tr></table>

Example Images from Training Set

![](images/c246488e9441c87bb4722f823ec5e136f58a59f903525c5d8abb44eb87835711.jpg)  
Figure 6: Example Images from Training Sets. The shape orientations are random, with positions sampled with jitter around fixed anchors in each quadrant. The upper-right corner, marked with a red box, always contains a circle. With increasing level, each of the other three corners has increasing modality, as more shapes are equally probable under the inlier distribution.

![](images/492fa3af094cad278221d384276e54f0299554266157ef78cf2fad6e0211683e.jpg)  
Figure 7: LACToS Toy Experiment 1: Robustness Under Multimodality. Average per-image False Positive Rates at True Positive Rate ∈ {80%, 90%, 95%, 99%}, evaluated on images that contain squares on the upper-right corner.

Using these diffusion models, we evaluate how the multimodality of the inlier distribution affects the performance of different local anomaly detection methods. We generate 50 test images that contain a square on the upper-right corner, and place other inlier shapes at the other corners. We label the pixels in the square as positive (anomalous), and all others as negative (non-anomalous). For DDAD Mousakhan et al. (2024), each test image is noised then denoised, which is used in conjunction with ResNet embeddings of the image to construct the anomaly heatmap. We also evaluate a variant of DDAD that uses more samples by repeating the noising-then-denoising process 20 times with different random seeds, averaging the denoised samples, and using the resulting average to construct the anomaly heatmap. For PatchCore Roth et al. (2022), a feed-forward model is trained for each inlier distribution, and a heatmap is generated by evaluating the test image with the feed-forward model. Finally, for LACToS, we follow the process described in Section 4, while varying the sample count from 1 to 20. Since the images are grayscale, we use the same set of parameters as for CHAOS, the dataset containing abdominal CT scans which are also in grayscale, for all methods.

To effectively isolate how likely different methods are to erroneously predict inlier shapes as anomalous, we report per-image averaged false- positive rate (FPR) at fixed values of per-image true-positive rate (TPR) ∈ {80%, 90%, 95%, 99%}. Figure 7 shows that the FPR of DDAD (Mousakhan et al., 2024) and PatchCore (Roth et al., 2022) increases with inlier multimodality, whereas the FPR of LACToS decreases rapidly with sample size, and stays negligible compared to the baseline methods when 20 samples are used, even when the inlier distribution is increasingly multimodal. This demonstrates that LACToS is more robust to inlier multimodality.

## C TOY EXPERIMENT 2

We conduct a second toy experiment to better understand how LACToS and baseline methods treat ID but under-represented features, compared to dominant ID features and strictly OOD features. We hypothesize that, due to the distribution-aware nature of statistical typicality, LACToS considers these minority ID features to be closer to majority ID features, whereas baseline methods that use a first-order statistic may consider them to be closer to OOD. To demonstrate this concretely, we train 5 diffusion models on training distributions whose specifications are provided in Table 3.

Table 3: Toy Experiment 2 Inlier Distribution Specification.
<table><tr><td>Level</td><td>Set of Candidate Shapes</td><td>Probability</td></tr><tr><td>Level 1</td><td>{Circle, Square}</td><td>{0.5, 0.5}</td></tr><tr><td>Level 2</td><td>{Circle, Square}</td><td>{0.6, 0.4}</td></tr><tr><td>Level 3</td><td>{Circle, Square}</td><td>{0.7, 0.3}</td></tr><tr><td>Level 4</td><td>{Circle, Square}</td><td>{0.8, 0.2}</td></tr><tr><td>Level 5</td><td>{Circle, Square}</td><td>{0.9,0.1}</td></tr></table>

Similar to the toy experiment in Section B, the upper-right corner still only admits a circle under the inlier distribution, so a square there is a strict OOD feature. On the other hand, the inlier probability of a square being placed in the other 3 corners is progressively lowered from 0.5 to 0.1. Example images in each training set are shown in Figure 8.

To verify the hypothesis, we measure two quantities. First, we measure the AUROC over pixels that belong to ID squares and circles, where squares are labeled positive and circles are labeled negative. A higher AUROC here means the method treats ID, but potentially under-represented, squares differently from ID circles, and vice versa. Second, we measure the AUROC over pixels that belong to squares, where OOD squares (those at the upper-right corner) are labeled positive and ID squares (those at other corners) are labeled negative. Higher AUROC here means that the method treats the under-represented ID squares differently from truly OOD squares, and vice versa. In Figure 9, we observe that LACToS is less likely to distinguish ID squares from ID circles and more likely to distinguish under-represented ID squares from truly OOD squares compared to DDAD. Both trends become more apparent with increasing number of sampled reconstructions. Together, these results verify our hypothesis, and inform practitioners that LACToS is a more reliable choice when one wishes to detect features that are not included in the training set without falsely flagging uncommon inlier features.

Example Images from Training Set

$$
\frac { 1 } { 2 } \times \frac { 1 } { 3 } = \frac { 1 } { 2 } \times \frac { 1 } { 3 }
$$

Figure 8: Example Images from Training Sets. Positions of the shapes are sampled with jitter around fixed anchors, with random orientations. The upper-right corner, marked with a red box, always contains a circle. With increasing level, the probability of a square at the other three corners decreases, making it more difficult to distinguish squares there from the square in the upper-right corner.

<table><tr><td>DDAD 1 sample LACToS 1 sample</td><td>LACToS 5 samples</td><td>LACToS 20 samples</td></tr><tr><td>DDAD 20 sample meanLACToS 2 samples</td><td>LACToS 10 samples</td><td>LACToS 40 samples</td></tr></table>

![](images/ffc68721966b5a6458c597eee1b302bb3b47e987f148a29882569af8eaa6d26f.jpg)

![](images/d11ef17a6993fb3486fb1c037db58ca27eabce862878b6cc8e410235d05e8d3b.jpg)  
Figure 9: LACToS Toy Experiment 2. AUROCs across different inlier square probabilities, over methods and different sample sizes. (Left) AUROC is computed over pixels that belong to inlier squares and circles, with those pixels from squares labeled positive, and negative otherwise. Higher AUROC means the method is more likely to distinguish an inlier square (minority) from inlier circles (majority). (Right) AUROC is computed over pixels that belong to squares, with those pixels from the square in the upper-right corner (truly OOD) labeled positive, and negative otherwise (minority inlier). Higher AUROC means the method better distinguishes an inlier but uncommon square from a strictly out-of-distribution square.

## D THEORY

Here we provide a formal version of Theorem 4.1. Fix a test image $x ^ { \star }$ and a superpixel $S _ { k } \in$ $\boldsymbol { \mathcal { S } } ( \boldsymbol { x } ^ { \star } )$ . Let $\mathcal { V } : = \mathbb { Z } _ { 2 5 6 } ^ { C }$ , where $\mathbb { Z } _ { 2 5 6 } = \{ 0 , \dots , 2 5 5 \}$ denotes the set of possible 8-bit values for one color channel. Thus, V is the single-pixel state space, under the finite quantized histogram setting. We consider features as an empirical extension, and do not include them in the discussion here. Under Assumption 4.1, the ideal inpainting posterior $p _ { k } ^ { \mathrm { i n p } } = p ( x \mid x _ { - S _ { k } } ^ { \star } )$ induces the single-pixel distribution

$$
q _ { k } ^ { \operatorname* { i n p } } : = p ( x _ { j } \mid x _ { - S _ { k } } ^ { \star } ) , \qquad q _ { k } ^ { \operatorname* { i n p } } : \mathcal { V } \to [ 0 , 1 ] , \quad j \in S _ { k } .\tag{11}
$$

For $\eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ , write $y : = x ^ { \star } + \eta$ . For a realization $y ^ { m }$ of $y ,$ , let $y ^ { m } : = x ^ { \star } + \eta ^ { m }$ denote the noised version of $x ^ { \star }$ used to sample $x ^ { m }$ , and define the pooled single-pixel denoising marginal for superpixel $S _ { k }$ as

$$
q _ { k , y ^ { m } } ^ { \mathrm { d e n } } : = \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } p ( x _ { j } \mid y ^ { m } ) , \qquad q _ { k , y ^ { m } } ^ { \mathrm { d e n } } : \mathcal { V } \to [ 0 , 1 ] .\tag{12}
$$

The denoising surrogate distribution used by the score is then

$$
\begin{array} { r } { q _ { k } ^ { \mathrm { d e n } , \sigma } : = \mathbb { E } _ { \eta } \left[ q _ { k , y } ^ { \mathrm { d e n } } \right] . } \end{array}\tag{13}
$$

The corresponding ideal and denoising scores are

$$
g ^ { \mathrm { i n p } } ( x ^ { \star } , k ) : = \left| - \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \log q _ { k } ^ { \mathrm { i n p } } ( x _ { j } ^ { \star } ) - | S _ { k } | \mathbb { H } ( q _ { k } ^ { \mathrm { i n p } } ) \right| ,\tag{14}
$$

$$
g _ { \sigma } ^ { \mathrm { d e n } } ( x ^ { \star } , k ) : = \left| - \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \log q _ { k } ^ { \mathrm { d e n } , \sigma } ( x _ { j } ^ { \star } ) - | S _ { k } | \mathbb { H } ( q _ { k } ^ { \mathrm { d e n } , \sigma } ) \right| .\tag{15}
$$

For a realization $y ^ { m }$ of $y ,$ , define the auxiliary distribution

$$
q _ { k , y ^ { m } } ^ { \mathrm { a u x } } : = \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } p ( x _ { j } \mid x _ { - S _ { k } } ^ { \star } , y _ { S _ { k } } ^ { m } ) , \qquad q _ { k , y ^ { m } } ^ { \mathrm { a u x } } : \mathcal { V } \to [ 0 , 1 ] .\tag{16}
$$

This is the ideal inpainting conditional after additionally revealing the noisy superpixel values, pooled over pixels in $S _ { k }$ . It is used only to separate the two sources of error: loss of context from observing the noised version of $x _ { - S _ { k } } ^ { \star }$ instead of $x _ { - S _ { k } } ^ { \star }$ , and the leakage of information about $x _ { S _ { k } } ^ { \star }$ from observing a noisy instead of completely masked superpixel. We will refer to the former as context $g a p .$ , and the latter as self-leakage.

The following assumption formalizes the bias–variance-like tradeoff induced by using denoising instead of inpainting. A denoising conditional observes a noisy version of the entire image, whereas the ideal inpainting conditional observes the clean complement $x _ { - S _ { k } } ^ { \star }$ and does not observe $x _ { S _ { k } } ^ { \star }$ . As σ increases, the noisy observation outside $S _ { k }$ becomes a worse proxy for the clean outside context and widens that context gap, which we capture as a linear function of $\sigma$ in the form of $\alpha _ { k } \sigma$ . In the opposite direction, when $\sigma$ is too small, the noisy values inside $S _ { k }$ reveal too much about $x _ { S _ { k } } ^ { \star } ,$ allowing the denoising posterior to copy the held-out region rather than predict it from context; this self-leakage is represented by the $\beta _ { k } \sigma ^ { - \gamma _ { k } }$ term.

Assumption D.1. [Denoising–inpainting tradeoff] There exist constants $\alpha _ { k } , \beta _ { k } , \gamma _ { k } > 0$ and $0 <$ $\underline { { \sigma } } _ { k } \le \overline { { \sigma } } _ { k }$ such that, for every $\sigma \in [ \underline { { \sigma } } _ { k } , \overline { { \sigma } } _ { k } ]$

$$
\mathbb { E } _ { \eta } \left[ \left. q _ { k , y } ^ { \mathrm { d e n } } - q _ { k , y } ^ { \mathrm { a u x } } \right. _ { 1 } \right] \leq \alpha _ { k } \sigma ,\tag{17}
$$

$$
\mathbb { E } _ { \eta } \left[ \left. q _ { k , y } ^ { \mathrm { a u x } } - q _ { k } ^ { \mathrm { i n p } } \right. _ { 1 } \right] \leq \beta _ { k } \sigma ^ { - \gamma _ { k } } .\tag{18}
$$

We add another assumption, which is a mild regularity condition ensuring that small distributional errors translate into small score errors. Since the score contains both a log-likelihood term and an entropy term, probabilities near zero can make the functional unstable. The lower bound by $\delta _ { k }$ rules out this degenerate case on the finite single-pixel state space. In practice, the computed histogram-based scores can enforce such a floor by smoothing or clipping the estimated probabilities.

Assumption D.2 (Score regularity). There exists $\delta _ { k } \in ( 0 , 1 )$ such that, for every $\sigma \in [ \underline { { \sigma } } _ { k } , \overline { { \sigma } } _ { k } ]$

$$
q _ { k } ^ { \mathrm { i n p } } ( v ) \geq \delta _ { k } \quad \mathrm { a n d } \quad q _ { k } ^ { \mathrm { d e n } , \sigma } ( v ) \geq \delta _ { k } \qquad \mathrm { f o r ~ a l l } \ v \in \mathcal { V } .\tag{19}
$$

We now state the formal version of Theorem 4.1.

Theorem D.1 (Formal denoising score approximation). Under Assumptions D.1 and D.2, for every $\sigma \in [ \underline { { \sigma } } _ { k } , \overline { { \sigma } } _ { k } ]$

$$
\begin{array} { r } { \left| g ^ { \mathrm { i n p } } ( x ^ { \star } , k ) - g _ { \sigma } ^ { \mathrm { d e n } } ( x ^ { \star } , k ) \right| \leq L _ { \delta _ { k } } \left( \alpha _ { k } \sigma + \beta _ { k } \sigma ^ { - \gamma _ { k } } \right) , } \end{array}\tag{20}
$$

where

$$
L _ { \delta _ { k } } : = \big ( \delta _ { k } ^ { - 1 } + 1 + \log ( \delta _ { k } ^ { - 1 } ) \big ) .\tag{21}
$$

Equivalently, after absorbing $L _ { \delta _ { k } }$ into the constants, there exist positive constants $\widetilde { \alpha } _ { k }$ and $\widetilde { \beta } _ { k }$ such that

$$
\begin{array} { r } { \left| g ^ { \mathrm { i n p } } ( x ^ { \star } , k ) - g _ { \sigma } ^ { \mathrm { d e n } } ( x ^ { \star } , k ) \right| \leq \widetilde { \alpha } _ { k } \sigma + \widetilde { \beta } _ { k } \sigma ^ { - \gamma _ { k } } . } \end{array}\tag{22}
$$

Proof. First, we begin by inspecting the error of the anomaly scores. For any two distributions $q , r : \mathcal { V } \to [ 0 , 1 ]$ satisfying $q ( v ) , r ( v ) \geq \delta _ { k }$ for every $v \in \mathcal V$

$$
\left| \left| - \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \log q ( x _ { j } ^ { \star } ) - \mathbb { H } ( q ) \right| - \left| - \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \log r ( x _ { j } ^ { \star } ) - \mathbb { H } ( r ) \right| \right|
$$

$$
\leq \left| { \frac { 1 } { | S _ { k } | } } \sum _ { j \in S _ { k } } { \big ( } \log r ( x _ { j } ^ { \star } ) - \log q ( x _ { j } ^ { \star } ) { \big ) } + { \big ( } \mathbb { H } ( r ) - \mathbb { H } ( q ) { \big ) } \right|
$$

$$
\leq \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \left| \log r ( x _ { j } ^ { \star } ) - \log q ( x _ { j } ^ { \star } ) \right| + \left| \mathbb { H } ( q ) - \mathbb { H } ( r ) \right| ,\tag{23}
$$

by applying reverse triangle inequality, and then triangle inequality. Since $t \mapsto - \log t$ has derivative bounded in absolute value by $\delta _ { k } ^ { - 1 }$ on $[ \delta _ { k } , 1 ]$

$$
\frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } \left| \log r ( x _ { j } ^ { \star } ) - \log q ( x _ { j } ^ { \star } ) \right| \leq \delta _ { k } ^ { - 1 } \| q - r \| _ { 1 } .\tag{24}
$$

Similarly, for $h ( t ) = - t$ log t, the derivative h<sup>′</sup>(t) = −1 − log t satisfies

$$
| h ^ { \prime } ( t ) | \leq 1 + \log ( \delta _ { k } ^ { - 1 } ) \qquad { \mathrm { f o r } } t \in [ \delta _ { k } , 1 ] .\tag{25}
$$

Applying the mean-value theorem coordinate-wise gives

$$
\begin{array} { r } { | \mathbb H ( q ) - \mathbb H ( r ) | \le \left( 1 + \log ( \delta _ { k } ^ { - 1 } ) \right) \| q - r \| _ { 1 } . } \end{array}\tag{26}
$$

Combining (23)–(26),

$$
\left| \left| - { \frac { 1 } { | S _ { k } | } } \sum _ { j \in S _ { k } } \log q ( x _ { j } ^ { \star } ) - \mathbb { H } ( q ) \right| - \left| - { \frac { 1 } { | S _ { k } | } } \sum _ { j \in S _ { k } } \log r ( x _ { j } ^ { \star } ) - \mathbb { H } ( r ) \right| \right| \leq L _ { \delta _ { k } } \| q - r \| _ { 1 } .\tag{27}
$$

It remains to control $\| q _ { k } ^ { \mathrm { d e n } , \sigma } - q _ { k } ^ { \mathrm { i n p } } \| _ { 1 }$ . By the definition of $q _ { k } ^ { \mathrm { d e n } , \sigma }$ and convexity of the $\ell _ { 1 }$ norm,

$$
\begin{array} { r } { \left. q _ { k } ^ { \mathrm { d e n } , \sigma } - q _ { k } ^ { \mathrm { i n p } } \right. _ { 1 } = \left. \mathbb { E } _ { \eta } \left[ q _ { k , y } ^ { \mathrm { d e n } } - q _ { k } ^ { \mathrm { i n p } } \right] \right. _ { 1 } } \\ { \leq \mathbb { E } _ { \eta } \left[ \left. q _ { k , y } ^ { \mathrm { d e n } } - q _ { k } ^ { \mathrm { i n p } } \right. _ { 1 } \right] . } \end{array}\tag{28}
$$

For each realization $y ^ { m }$ of $y ,$ , insert $q _ { k , y ^ { m } } ^ { \mathrm { a u x } }$ and apply the triangle inequality:

$$
\left\| q _ { k , y ^ { m } } ^ { \mathrm { d e n } } - q _ { k } ^ { \mathrm { i n p } } \right\| _ { 1 } \leq \left\| q _ { k , y ^ { m } } ^ { \mathrm { d e n } } - q _ { k , y ^ { m } } ^ { \mathrm { a u x } } \right\| _ { 1 } + \left\| q _ { k , y ^ { m } } ^ { \mathrm { a u x } } - q _ { k } ^ { \mathrm { i n p } } \right\| _ { 1 } .\tag{29}
$$

Taking expectations and using Assumption D.1 yields

$$
\begin{array} { r } { \left. q _ { k } ^ { \mathrm { d e n } , \sigma } - q _ { k } ^ { \mathrm { i n p } } \right. _ { 1 } \leq \alpha _ { k } \sigma + \beta _ { k } \sigma ^ { - \gamma _ { k } } . } \end{array}\tag{30}
$$

Finally, apply (27) with $q = q _ { k } ^ { \mathrm { d e n } , \sigma }$ and $r = q _ { k } ^ { \mathrm { i n p } }$ . This proves (20). The absorbed-constant form (22) follows by setting $\widetilde { \alpha } _ { k } = L _ { \delta _ { k } } \alpha _ { k }$ and $\widetilde { \beta } _ { k } = L _ { \delta _ { k } } \beta _ { k }$ □

Corollary D.1 (Bound-optimal denoising noise). Under the conditions of Theorem D.1, the upper bound in (20) is minimized over $\sigma \in [ \underline { { \sigma } } _ { k } , \overline { { \sigma } } _ { k } ]$ by

$$
\sigma _ { k } ^ { \mathrm { o p t } } : = \operatorname* { m i n } \left\{ \overline { { \sigma } } _ { k } , \operatorname* { m a x } \left\{ \underline { { \sigma } } _ { k } , \left( \frac { \gamma _ { k } \beta _ { k } } { \alpha _ { k } } \right) ^ { 1 / ( \gamma _ { k } + 1 ) } \right\} \right\} .\tag{31}
$$

In particular, if

$$
\left( \frac { \gamma _ { k } \beta _ { k } } { \alpha _ { k } } \right) ^ { 1 / \left( \gamma _ { k } + 1 \right) } \in [ \underline { { \sigma } } _ { k } , \overline { { \sigma } } _ { k } ] ,\tag{32}
$$

then the bound is minimized at the interior value

$$
\sigma _ { k } ^ { \mathrm { o p t } } = \left( \frac { \gamma _ { k } \beta _ { k } } { \alpha _ { k } } \right) ^ { 1 / \left( \gamma _ { k } + 1 \right) } ,\tag{33}
$$

and the minimum value of the right-hand side of (20) is

$$
L _ { \delta _ { k } } ( \gamma _ { k } + 1 ) \gamma _ { k } ^ { - \gamma _ { k } / ( \gamma _ { k } + 1 ) } \alpha _ { k } ^ { \gamma _ { k } / ( \gamma _ { k } + 1 ) } \beta _ { k } ^ { 1 / ( \gamma _ { k } + 1 ) } .\tag{34}
$$

Proof. The multiplicative factor $L _ { \delta _ { k } }$ in (20) is positive and therefore does not affect the choice of σ. It is enough to minimize

$$
\alpha _ { k } \sigma + \beta _ { k } \sigma ^ { - \gamma _ { k } } .\tag{35}
$$

The derivative is

$$
\alpha _ { k } - \gamma _ { k } \beta _ { k } \sigma ^ { - \gamma _ { k } - 1 } ,\tag{36}
$$

which vanishes exactly when

$$
\sigma = \left( \frac { \gamma _ { k } \beta _ { k } } { \alpha _ { k } } \right) ^ { 1 / \left( \gamma _ { k } + 1 \right) } .\tag{37}
$$

The second derivative is

$$
\gamma _ { k } ( \gamma _ { k } + 1 ) \beta _ { k } \sigma ^ { - \gamma _ { k } - 2 } > 0 ,\tag{38}
$$

so this critical point is the unique unconstrained minimizer. Restricting $\sigma$ to $[ \underline { { \sigma } } _ { k } , \overline { { \sigma } } _ { k } ]$ gives (31). When the unconstrained minimizer lies inside the interval, substituting (33) into $L _ { \delta _ { k } } ( \alpha _ { k } \sigma + \beta _ { k } \sigma ^ { - \gamma _ { k } } )$ gives (34). 口

This corollary should be interpreted as selecting the bound-optimal noise level rather than the necessarily optimal empirical denoising noise. The expression balances the two errors in Assumption D.1: increasing σ worsens the context gap, while decreasing σ worsens self-leakage. At an interior optimum, these two effects satisfy $\bar { \alpha _ { k } \sigma _ { k } ^ { \mathrm { o p t } } } = \gamma _ { k } \beta _ { k } ( \sigma _ { k } ^ { \mathrm { o p t } } ) ^ { - \gamma _ { k } }$

We now analyze the typicality score without Assumption 4.1. The signed statistic remains centered under the ID model, while dependence affects its concentration through the covariance of the per-pixel information values.

Fix $S _ { k }$ and $x _ { - S _ { k } } ^ { \star }$ , and write $n _ { k } : = | S _ { k } |$ |. Since the single-pixel marginals may now differ, replace $q _ { k } ^ { \mathrm { i n p } }$ by their pooled marginal

$$
\bar { q } _ { k } ^ { \mathrm { i n p } } ( v ) : = \frac { 1 } { n _ { k } } \sum _ { j \in S _ { k } } p ( x _ { j } = v \mid x _ { - S _ { k } } ^ { \star } ) , \qquad v \in \mathcal { V } .\tag{39}
$$

This reduces to $q _ { k } ^ { \mathrm { i n p } }$ under Assumption 4.1. Retaining the probability floor of Assumption D.2 for $\bar { q } _ { k } ^ { \mathrm { i n p } }$ , define

$$
\begin{array} { r l } & { Z _ { k , j } : = - \log \bar { q } _ { k } ^ { \mathrm { i n p } } ( x _ { j } ) - \mathbb { H } ( \bar { q } _ { k } ^ { \mathrm { i n p } } ) , } \\ & { T _ { k } ^ { \mathrm { i n p } } : = \displaystyle \frac { 1 } { n _ { k } } \sum _ { j \in S _ { k } } Z _ { k , j } , \qquad G _ { k } ^ { \mathrm { i n p } } : = | T _ { k } ^ { \mathrm { i n p } } | . } \end{array}\tag{40}
$$

Here $G _ { k } ^ { \mathrm { i n p } }$ is the per-pixel counterpart of the score in (14), using the pooled marginal.

By the definition of $\bar { q } _ { k } ^ { \mathrm { i n p } }$ , the expected average information is H $( \bar { q } _ { k } ^ { \mathrm { i n p } } )$ , and hence $\mathbb { E } _ { p _ { k } ^ { \mathrm { i n p } } } [ T _ { k } ^ { \mathrm { i n p } } ] = 0$ Dependence enters through the total variance

$$
\begin{array} { r l } & { \Gamma _ { k } ^ { \mathrm { I D } } : = \mathrm { V a r } _ { p _ { k } ^ { \mathrm { i n p } } } \left( \displaystyle \sum _ { j \in S _ { k } } Z _ { k , j } \right) } \\ & { \quad \quad = \displaystyle \sum _ { j \in S _ { k } } \mathrm { V a r } _ { p _ { k } ^ { \mathrm { i n p } } } ( Z _ { k , j } ) + 2 \sum _ { \stackrel { i , j \in S _ { k } } { i < j } } \mathrm { C o v } _ { p _ { k } ^ { \mathrm { i n p } } } ( Z _ { k , i } , Z _ { k , j } ) . } \end{array}\tag{41}
$$

Since $\mathrm { V a r } _ { p _ { k } ^ { \mathrm { i n p } } } ( T _ { k } ^ { \mathrm { i n p } } ) = \Gamma _ { k } ^ { \mathrm { I D } } / n _ { k } ^ { 2 }$ , Chebyshev’s inequality gives

$$
\mathbb { P } _ { p _ { k } ^ { \mathrm { i n p } } } \big ( G _ { k } ^ { \mathrm { i n p } } \ge \tau \big ) \le \frac { \Gamma _ { k } ^ { \mathrm { I D } } } { n _ { k } ^ { 2 } \tau ^ { 2 } } , \qquad \tau > 0 .\tag{42}
$$

To express this in terms of effective sample size, let

$$
v _ { k , \mathrm { I D } } ^ { 2 } : = \frac { 1 } { n _ { k } } \sum _ { j \in S _ { k } } \mathrm { V a r } _ { p _ { k } ^ { \mathrm { i n p } } } ( Z _ { k , j } ) , \qquad n _ { k , \mathrm { e f f } } ^ { \mathrm { I D } } : = \frac { n _ { k } ^ { 2 } v _ { k , \mathrm { I D } } ^ { 2 } } { \Gamma _ { k } ^ { \mathrm { I D } } } \quad ( \Gamma _ { k } ^ { \mathrm { I D } } > 0 ) .\tag{43}
$$

The bound becomes $v _ { k , \mathrm { I D } } ^ { 2 } / ( n _ { k , \mathrm { e f f } } ^ { \mathrm { I D } } \tau ^ { 2 } )$ . Under Assumption 4.1, the cross-covariances vanish and $n _ { k , \mathrm { e f f } } ^ { \mathrm { I D } } = n _ { k } ; $ a positive sum of cross-covariances reduces the effective sample size and weakens concentration.

The same variance dependence appears in OOD detection. For an OOD conditional distribution $R _ { k }$ , let $\Delta _ { k } : = | \mathbb { E } _ { R _ { k } } [ T _ { k } ^ { \mathrm { i n p } } ] |$ and $\begin{array} { r } { \Gamma _ { k } ^ { \mathrm { O O D } } : = \operatorname { V a r } _ { R _ { k } } ( \sum _ { j \in S _ { k } } Z _ { k , j } ) } \end{array}$ . If $G _ { k } ^ { \mathrm { i n p } } < \tau < \Delta _ { k }$ , then $| T _ { k } ^ { \mathrm { i n p } } - |$ $\begin{array} { r } { \mathbb { E } _ { R _ { k } } [ T _ { k } ^ { \mathrm { i n p } } ] | \geq \Delta _ { k } - \tau } \end{array}$ . Chebyshev’s inequality therefore yields

$$
\mathbb { P } _ { R _ { k } } ( G _ { k } ^ { \mathrm { i n p } } < \tau ) \le \frac { \Gamma _ { k } ^ { \mathrm { O O D } } } { n _ { k } ^ { 2 } ( \Delta _ { k } - \tau ) ^ { 2 } } , \qquad 0 < \tau < \Delta _ { k } .\tag{44}
$$

Finally, let $G _ { k , \sigma } ^ { \mathrm { d e n } }$ be the score in (15) divided by $n _ { k }$ , with its denoising marginal evaluated at the image being scored. If Assumptions D.1 and D.2 hold uniformly over the considered superpixel realizations, with common constants and $\bar { q } _ { k } ^ { \mathrm { i n p } }$ replacing $q _ { k } ^ { \mathrm { i n p } }$ , the proof of Theorem D.1 gives

$$
\begin{array} { r } { | G _ { k } ^ { \mathrm { i n p } } - G _ { k , \sigma } ^ { \mathrm { d e n } } | \leq e _ { k } ( \sigma ) , \qquad e _ { k } ( \sigma ) : = L _ { \delta _ { k } } \big ( \alpha _ { k } \sigma + \beta _ { k } \sigma ^ { - \gamma _ { k } } \big ) . } \end{array}\tag{45}
$$

Applying (42) and (44) at the resulting shifted thresholds gives

$$
\mathbb { P } _ { p _ { k } ^ { \mathrm { i n p } } } \big ( G _ { k , \sigma } ^ { \mathrm { d e n } } \geq \tau \big ) \leq \frac { \Gamma _ { k } ^ { \mathrm { I D } } } { n _ { k } ^ { 2 } ( \tau - e _ { k } ( \sigma ) ) ^ { 2 } } , \qquad \tau > e _ { k } ( \sigma ) ,\tag{46}
$$

$$
\mathbb { P } _ { R _ { k } } \big ( G _ { k , \sigma } ^ { \mathrm { d e n } } < \tau \big ) \le \frac { \Gamma _ { k } ^ { \mathrm { O O D } } } { n _ { k } ^ { 2 } ( \Delta _ { k } - \tau - e _ { k } ( \sigma ) ) ^ { 2 } } , \qquad 0 < \tau < \Delta _ { k } - e _ { k } ( \sigma ) .\tag{47}
$$

Under Assumption 4.1, the effective sample size equals the number of pixels in the superpixel. Without this assumption, the signed statistic remains centered under the ID model, but positive within-superpixel covariance reduces the effective sample size and weakens the concentration and detection bounds. The denoising approximation retains the same form, with the effect of dependence captured by the variance terms.

## E EXPERIMENTAL DETAILS

## E.1 DATASET SUMMARY

Table 4 summarizes the train/test composition, masking, and image resolution for each dataset. MVTec AD and CHAOS are evaluated at 256×256 following standard practice; FFHQ is 256×256; DVXray is $6 0 0 \times 8 0 0$ in its native form, and we report per-pixel mean metrics so resolution does not bias cross-dataset comparison.

Table 4: Dataset composition. Training images contain only inlier (anomaly-free) examples. Test images contain local anomalies and are paired with pixel-level ground-truth masks.
<table><tr><td>Dataset</td><td># train (inlier)</td><td># test (ID + OOD)</td><td>Eval mask</td><td>Resolution</td></tr><tr><td>CHAOS</td><td>3,070</td><td>100 + 100</td><td>body silhouette</td><td>256×256</td></tr><tr><td>FFHQ</td><td>5,000</td><td> $5 0 + 5 0$ </td><td></td><td> $2 5 6 \times 2 5 6$ </td></tr><tr><td>DVXray (scissors)</td><td>5,000</td><td>72 + 72</td><td></td><td> $6 0 0 \times 8 0 0$ </td></tr><tr><td>MVTec AD (15 categories)</td><td>3,629</td><td>1,258</td><td></td><td> $2 5 6 \times 2 5 6$ </td></tr></table>

## E.2 DIFFUSION MODELS AND RECONSTRUCTION

For MVTec AD, we use DDAD’s checkpoints, which are ϵ-prediction UNet diffusion models, trained separately for each category. For all other benchmarks, we use the guided-diffusion UNet of Dhariwal and Nichol (Dhariwal and Nichol, 2021) trained from scratch on each inlier set, and produce reconstructions via DPS (Chung et al., 2022).

## E.3 LACTOS HYPERPARAMETERS

For sample reconstruction using DPS (Chung et al., 2022), we utilize the identity forward model and zero-mean Gaussian noise with variance $\sigma _ { \mathrm { n o i s e } }$ . DPS also has a scale parameter that controls the posterior guidance strength.

Once M reconstructions are computed and the corresponding features are acquired, we rescale RGB values from [0, 255] and autoencoder outputs from [0, 1] to a common [0, 255] range. Then, we partition the reconstructions and feature maps into superpixels containing an average of $S _ { \mathrm { a v g } }$ pixels each, using marker-controlled watershed segmentation (Meyer and Beucher, 1990). For each superpixel, we quantize the rescaled values into $B _ { \mathrm { R G B } }$ and $B _ { \mathrm { f e a t } }$ bins per dimension, for RGB values and autoencoder outputs respectively. While it is ideal to construct $C + C _ { f } ^ { \prime }$ -dimensional histograms, the number of histogram bins grows exponentially with the dimension. Therefore, we construct two empirical histograms per superpixel, one over RGB values and one over feature values. We smooth each histogram with a Gaussian kernel of $\sigma _ { \mathrm { h i s t } }$ to mitigate empty bins under limited samples, and use the histograms to compute the RGB and feature typicality scores for each superpixel. The two scores are summed with a weight parameter λ multiplied to the feature typicality to balance the trade-off between the two scores. Finally, we smooth the anomaly map that contains the same value over all pixels in each superpixel to acquire the pixel-level anomaly map. All parameters are selected by performing a grid search over a small subset containing 20 test images, and the configuration with the highest AUROC is used to evaluate over the full test set. The specific parameters used for each benchmark can be found in Table 5.

Table 5: Dataset-specific hyperparameters used for LACToS.
<table><tr><td>Parameter</td><td>CHAOS</td><td>FFHQ</td><td>DVXray</td></tr><tr><td>DPS  $\sigma _ { \mathrm { n o i s e } }$ </td><td>0.1</td><td>0.01</td><td>2.0</td></tr><tr><td>DPS scale</td><td>0.4</td><td>0.1</td><td>1.0</td></tr><tr><td> $S _ { \mathrm { a v g } }$ </td><td>100</td><td>20</td><td>10</td></tr><tr><td>M</td><td>5</td><td>20</td><td>20</td></tr><tr><td>σhist</td><td>0.5</td><td>1.0</td><td>0</td></tr><tr><td> $\left( { B _ { \mathrm { R G B } } } , { B _ { \mathrm { f e a t } } } \right)$ </td><td>(16,64)</td><td>(16,32)</td><td>(16, 16)</td></tr><tr><td>λ</td><td>1.0</td><td>0.25</td><td>0</td></tr></table>

## E.4 BASELINES

For baselines, we use the official implementations of PatchCore, SimpleNet, SuperSimpleNet, EfficientAD, GLASS, and DDAD, with each method’s default backbone: WideResNet-50 for PatchCore; ImageNet-pretrained ResNet/WideResNet for the SimpleNet variants.

For fair comparison, we perform parameter tuning for each baseline method using the same set of 20 test images. Note that we do not tune the parameters of SimpleNet, SuperSimpleNet, and GLASS as this would require retraining; we use the default parameters in the respective manuscripts or github repositories. For PatchCore, k, the number of nearest neighbors in the coreset to compare each test patch, is swept across $k \in \{ 1 , 2 , 3 , 5 , 7 , 9 , 1 5 \}$ . For EfficientAD, α, which balances the strengths of the student-teacher detector and the autoencoder detector, is swept across $\alpha \in \{ 0 , 0 . 1 , 0 . 2 , \cdot \cdot \cdot , 1 . 0 \}$ For DDAD, a 2D grid of $( w , v ) \in \{ 1 , 2 , 4 , 6 , 8 \} \times \{ 0 . 5 , 1 , 2 , 4 \}$ is searched, where w controls the guidance strength during denoising and v balances the strengths of pixels and extracted features. The selected parameter values are described in Table 6.

Table 6: Dataset-specific hyperparameters used for baseline methods.
<table><tr><td>Parameter</td><td>CHAOS</td><td>FFHQ</td><td>DVXray</td></tr><tr><td>PatchCore</td><td> $k = 1$ </td><td> $k = 1$ </td><td> $k = 1$ </td></tr><tr><td>EfficientAD</td><td> $\alpha = 0$ </td><td> $\alpha = 0 . 7$ </td><td> $\alpha = 0 . 2$ </td></tr><tr><td>DDAD</td><td> $( w , v ) = ( 2 , 1 )$ </td><td> $( w , v ) = ( 2 , 4 )$ </td><td> $( w , v ) = ( 4 , 4 )$ </td></tr></table>

## E.5 FEATURE EXTRACTOR AND AUTOENCODER

The feature extractor ϕ is an ImageNet-pretrained ResNet-101 He et al. (2016), frozen at all stages. We extract activations from the last three residual stages, upsample them to the input resolution, and concatenate them. To reduce histogram error under limited samples, these features are compressed to $C _ { f } ^ { \prime } = 3$ dimensions by a per-pixel autoencoder trained on each inlier dataset.

## F ADDITIONAL RESULTS

We provide additional visualizations of each benchmark in Figure 10, Figure 11, and Figure 12.

Image  
LACToS  
DDAD  
PatchCore  
SimpleNet  
Super SimpleNet  
EfficientAD  
GLASS  
![](images/f67b6132da5070cd733f1e1e958342482a93145cd0c8e844bdc0b68be0f6d4f1.jpg)  
Figure 10: CHAOS Additional Visualizations. 10 randomly selected test images and the heatmaps for each method.

Image  
LACToS  
DDAD  
PatchCore  
SimpleNet  
Super SimpleNet  
EfficientAD  
GLASS  
![](images/1b37cfad5872508bfddd0fc29e583b2d98e50214fbd856ed67bf815a35be37fa.jpg)  
Figure 11: FFHQ Additional Visualizations. 10 randomly selected test images and the heatmaps for each method.

Image  
LACToS  
DDAD  
PatchCore  
SimpleNet  
Super SimpleNet  
EfficientAD  
GLASS  
![](images/a061f567783db25cede4a2d54bd185b449f0008f6daf7655b9b7eba12a1023fb.jpg)  
Figure 12: DVXray Additional Visualizations. 10 randomly selected test images and the heatmaps for each method.

In Table 7, we report the AUROC, AP, and AUPRO of all methods for each of the object category in the MVTec dataset.

Table 7: Results on MVTec AD. We report pixel-level AUROC, AP, and AUPRO for each object or texture category. AUROC and AP pool every pixel of all test images of the category (anomalous and anomaly-free); AUPRO is the region-based per-region overlap integrated up to an FPR of 0.3. Each method occupies three rows, one per metric. Higher is better for all metrics.
<table><tr><td>Method</td><td>Metric</td><td>Carpet</td><td>Grid</td><td>Leather</td><td>Tile</td><td>Wood</td><td>Bottle</td><td>Cable</td><td>Capsule</td></tr><tr><td rowspan="4">SimpleNet</td><td rowspan="4">AUROC AP AUPRO</td><td>0.9765</td><td>0.9656</td><td>0.9898</td><td>0.9326</td><td>0.9177</td><td>0.9775</td><td>0.9688</td><td>0.9862</td></tr><tr><td>0.3650</td><td>0.2440</td><td>0.3257</td><td>0.5470</td><td>0.3673</td><td>0.6315</td><td>0.6151</td><td>0.3729</td></tr><tr><td>0.9047</td><td>0.8757</td><td>0.9630</td><td>0.7781</td><td>0.7866</td><td>0.8945</td><td>0.8741</td><td>0.9093</td></tr><tr><td></td><td>0.9771</td><td>0.9878</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">SuperSimpleNet</td><td rowspan="4">AUROC AP AUPRO</td><td>0.9832</td><td></td><td></td><td>0.9383</td><td>0.9208</td><td>0.9779</td><td>0.9728</td><td>0.9805</td></tr><tr><td>0.4548</td><td>0.2383</td><td>0.3492</td><td>0.5109</td><td>0.3867</td><td>0.6202</td><td>0.6151</td><td>0.3973</td></tr><tr><td>0.9222</td><td>0.9236</td><td>0.9686</td><td>0.8107</td><td>0.8551</td><td>0.8982</td><td>0.8746</td><td>0.8753</td></tr><tr><td>0.9877</td><td>0.9729</td><td>0.9903</td><td>0.9496</td><td>0.9301</td><td>0.9813</td><td>0.9831</td><td>0.9873</td></tr><tr><td rowspan="3">PatchCore</td><td rowspan="3">AUROC AP AUPRO</td><td>0.5770</td><td>0.2326</td><td>0.3533</td><td>0.4947</td><td>0.4016</td><td>0.6842</td><td>0.6258</td><td>0.3935</td></tr><tr><td>0.9295</td><td>0.8932</td><td>0.9574</td><td>0.8212</td><td>0.8312</td><td>0.9253</td><td>0.9153</td><td>0.9227</td></tr><tr><td>0.9822</td><td>0.9795</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">DDAD</td><td rowspan="3">AP AUPRO</td><td>0.6422</td><td>0.2710</td><td>0.9878 0.4124</td><td>0.9790 0.7853</td><td>0.9374 0.5285</td><td>0.9860</td><td>0.9784 0.6641</td><td>0.8693 0.2313</td></tr><tr><td></td><td>0.9291</td><td>0.9811</td><td>0.9117</td><td>0.8604</td><td>0.8047</td><td></td><td>0.6845</td></tr><tr><td>0.8994</td><td>0.9833</td><td>0.9916</td><td>0.9816</td><td></td><td>0.9003</td><td>0.8835</td><td></td></tr><tr><td rowspan="3">EfficientAD</td><td>AP AUPRO</td><td>0.9831 0.7511</td><td>0.4328</td><td>0.5421</td><td>0.7505</td><td>0.9680 0.6273</td><td>0.9884 0.7852</td><td>0.9863 0.7593</td><td>0.9915 0.4603</td></tr><tr><td></td><td>0.9489</td><td>0.9144</td><td>0.9840</td><td>0.9249</td><td>0.9207</td><td>0.9550</td><td>0.9009</td><td>0.9445</td></tr><tr><td>AUROC</td><td>0.9952 0.8108</td><td>0.9908 0.4528</td><td>0.9952</td><td>0.9963</td><td>0.9761</td><td>0.9922</td><td>0.9569</td><td>0.9898</td></tr><tr><td rowspan="3">LACToS (Ours)</td><td>AUPRO AUROC AP</td><td>0.9832</td><td>0.9606</td><td>0.5518 0.9815</td><td>0.9640 0.9770</td><td>0.7196 0.9467</td><td>0.8966 0.9652</td><td>0.4997 0.8470</td><td>0.4798 0.9460</td></tr><tr><td></td><td>0.9822</td><td>0.9712</td><td>0.9935</td><td>0.9307</td><td>0.9099</td><td>0.9839</td><td>0.9739</td><td>0.9703</td></tr><tr><td>AUPRO</td><td>0.4489 0.9209</td><td>0.3813 0.8934</td><td>0.6339 0.9540</td><td>0.7085 0.8165</td><td>0.4139 0.6801</td><td>0.8151</td><td>0.5891</td><td>0.4118</td></tr><tr><td>Method</td><td>Metric AUROC</td><td>Hazelnut</td><td>Metal Nut</td><td>Pill</td><td>Screw</td><td>Toothbrush</td><td>0.8877 Transistor</td><td>0.8712 Zipper</td><td>0.8757 Average</td></tr><tr><td rowspan="3">SimpleNet</td><td>AP AUPRO</td><td>0.9741</td><td>0.9747</td><td>0.9839</td><td>0.9887</td><td>0.9823</td><td></td><td>0.9790</td><td>0.9715</td></tr><tr><td></td><td>0.3978</td><td>0.8049</td><td>0.7402</td><td>0.2325</td><td>0.3377</td><td>0.9746</td><td></td><td>0.4714</td></tr><tr><td></td><td>0.9297</td><td>0.8082</td><td></td><td>0.9464</td><td>0.8404</td><td>0.6181 0.9354</td><td>0.4712 0.9091</td><td>0.8837</td></tr><tr><td rowspan="3">SuperSimpleNet</td><td>AUROC</td><td></td><td></td><td>0.9009</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AP</td><td>0.9792</td><td>0.9855</td><td>0.9841</td><td>0.9873</td><td>0.9851</td><td>0.9618</td><td>0.9813</td><td>0.9735</td></tr><tr><td>AUPRO</td><td>0.5013</td><td>0.8630</td><td>0.7763</td><td>0.2242</td><td>0.4923</td><td>0.6081</td><td>0.4395</td><td>0.4985</td></tr><tr><td rowspan="3">PatchCore</td><td></td><td>0.9447</td><td>0.9004</td><td>0.9326</td><td>0.9434</td><td>0.8473</td><td>0.9130</td><td>0.9304</td><td>0.9027</td></tr><tr><td>AUROC</td><td>0.9851</td><td>0.9659</td><td>0.9895</td><td>0.9908</td><td>0.9827</td><td>0.9742</td><td>0.9813</td><td>0.9768</td></tr><tr><td>AP</td><td>0.5019</td><td>0.7060</td><td>0.7824</td><td>0.2585</td><td>0.3272</td><td>0.6312</td><td>0.4444</td><td>0.4943</td></tr><tr><td rowspan="3">DDAD</td><td>AUPRO</td><td>0.9559</td><td>0.8557</td><td>0.9517</td><td>0.9572</td><td>0.8613</td><td>0.9431</td><td>0.9244</td><td>0.9097</td></tr><tr><td>AUROC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AP</td><td>0.9832 0.5405</td><td>0.9840 0.8580</td><td>0.9461</td><td>0.8490 0.0202</td><td>0.9729 0.5126</td><td>0.9436 0.6341</td><td>0.9751 0.5388</td><td>0.9569 0.5377</td></tr><tr><td rowspan="3">EfficientAD</td><td>AUPRO</td><td>0.9279</td><td>0.9072</td><td>0.6220 0.5272</td><td>0.5060</td><td>0.6514</td><td>0.9233</td><td>0.9056</td><td>0.8266</td></tr><tr><td>AUROC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AP</td><td>0.9856 0.5492</td><td>0.9847 0.8220</td><td>0.9885 0.7474</td><td>0.9945 0.4125</td><td>0.9882 0.5202</td><td>0.9775 0.8170</td><td>0.9892 0.5594</td><td>0.9855 0.6358</td></tr><tr><td rowspan="3">GLASS</td><td>AUPRO</td><td>0.9457</td><td>0.9337</td><td>0.9606</td><td>0.9692</td><td>0.9315</td><td>0.9455</td><td>0.9615</td><td>0.9427</td></tr><tr><td>AUROC</td><td></td><td></td><td></td><td></td><td>0.9629</td><td></td><td></td><td>0.9474</td></tr><tr><td>AP</td><td>0.9863 0.6267</td><td>0.8987 0.5778</td><td>0.9727 0.8330</td><td>0.9315 0.2391</td><td>0.2321</td><td>0.5712 0.1719</td><td>0.9948 0.7725</td><td>0.5886</td></tr><tr><td rowspan="3"></td><td>AUPRO</td></table>

## G ABLATION STUDIES

In this section, we ablate our method on the different choices of hyper-parameters.

Sample efficiency. Table 8 reports the AUROC, AP, and AUPRO of diffusion-based methods DDAD (Mousakhan et al., 2024) and LACToS as the number of denoising samples M varies from 1 to 20. LACToS exhibits improved performance with increasing M on CHAOS and DVXray compared to DDAD, which relies on point estimates of reconstruction errors. This is consistent with the statistical properties of typicality: more samples yield better estimations of $\hat { q } _ { k } ^ { \mathrm { i n p } }$ , which improves the reliability of the entropy and log likelihood estimates. At M = 1, LACToS degenerates toward a reconstruction-error-like score.

Superpixel size. Table 9 shows the effect of the number of superpixels’ average size in the performance of LACToS, which is mostly stable but with different trends across benchmarks.

AutoEncoder features. The sensitivity of the metrics under varying λ is illustrated in Table 10. λ = 0 corresponds to using RGB typicality only. In most datasets, including feature typicality significantly improves performance.

Table 8: Ablation on sample size M in diffusion-based methods. AUROC, AP, and AUPRO are reported for each dataset.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="2"> $M = 1$ </td><td colspan="2"> $M = 5$ </td><td colspan="2"> $M = 1 0$ </td><td colspan="2"> $M = 2 0$ </td></tr><tr><td>DDAD</td><td>LACToS</td><td>DDAD</td><td>LACToS</td><td>DDAD</td><td>LACToS</td><td>DDAD</td><td>LACToS</td></tr><tr><td rowspan="3">CHAOS</td><td>AUROC</td><td>0.9707</td><td>0.9552</td><td>0.9748</td><td>0.9620</td><td>0.9749</td><td>0.9615</td><td>0.9745</td><td>0.9620</td></tr><tr><td>AP</td><td>0.3159</td><td>0.2536</td><td>0.3496</td><td>0.4318</td><td>0.3468</td><td>0.4236</td><td>0.3489</td><td>0.4338</td></tr><tr><td>AUPRO</td><td>0.8676</td><td>0.8524</td><td>0.8827</td><td>0.8764</td><td>0.8825</td><td>0.8713</td><td>0.8814</td><td>0.8729</td></tr><tr><td rowspan="3">FFHQ</td><td>AUROC</td><td>0.7469</td><td>0.8446</td><td>0.7512</td><td>0.8521</td><td>0.7559</td><td>0.8503</td><td>0.7583</td><td>0.8515</td></tr><tr><td>AP</td><td>0.0563</td><td>0.0937</td><td>0.0543</td><td>0.1197</td><td>0.0552</td><td>0.1119</td><td>0.0564</td><td>0.1177</td></tr><tr><td>AUPRO</td><td>0.3083</td><td>0.4638</td><td>0.3065</td><td>0.5157</td><td>0.3202</td><td>0.5239</td><td>0.3151</td><td>0.5320</td></tr><tr><td rowspan="3">DVXray</td><td>AUROC</td><td>0.8567</td><td>0.9347</td><td>0.8518</td><td>0.9541</td><td>0.8523</td><td>0.9585</td><td>0.8495</td><td>0.9607</td></tr><tr><td>AP</td><td>0.0118</td><td>0.0248</td><td>0.0119</td><td>0.0465</td><td>0.0119</td><td>0.0530</td><td>0.0118</td><td>0.0568</td></tr><tr><td>AUPRO</td><td>0.5183</td><td>0.7385</td><td>0.4896</td><td>0.7964</td><td>0.4880</td><td>0.8055</td><td>0.4890</td><td>0.8179</td></tr></table>

Table 9: Ablation on the average size of superpixels. The larger the average size, the smaller the total number of superpixels. AUROC, AP, and AUPRO on CHAOS, FFHQ, and DVXray.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="6">Average size of superpixels</td><td rowspan="2">200</td></tr><tr><td>10</td><td>20</td><td>30</td><td>50</td><td>70</td><td>100</td></tr><tr><td rowspan="3">CHAOS</td><td>AUROC</td><td>0.9418</td><td>0.9504</td><td>0.9535</td><td>0.9578</td><td>0.9595</td><td>0.9620</td><td>0.9593</td></tr><tr><td>AP</td><td>0.1271</td><td>0.1599</td><td>0.1857</td><td>0.2434</td><td>0.2643</td><td>0.4318</td><td>0.3677</td></tr><tr><td>AUPRO</td><td>0.7992</td><td>0.8274</td><td>0.8385</td><td>0.8499</td><td>0.8571</td><td>0.8764</td><td>0.8600</td></tr><tr><td rowspan="3">FFHQ</td><td>AUROC</td><td>0.7930</td><td>0.8515</td><td>0.8502</td><td>0.8480</td><td>0.8462</td><td>0.8438</td><td>0.8383</td></tr><tr><td>AP</td><td>0.0774</td><td>0.1177</td><td>0.1145</td><td>0.1107</td><td>0.1095</td><td>0.1065</td><td>0.0998</td></tr><tr><td>AUPRO</td><td>0.3890</td><td>0.5320</td><td>0.5263</td><td>0.5180</td><td>0.5101</td><td>0.5068</td><td>0.4863</td></tr><tr><td rowspan="3">DVXray</td><td>AUROC</td><td>0.9607</td><td>0.9611</td><td>0.9593</td><td>0.9501</td><td>0.9468</td><td>0.9463</td><td>0.9330</td></tr><tr><td>AP</td><td>0.0568</td><td>0.0570</td><td>0.0537</td><td>0.0495</td><td>0.0458</td><td>0.0418</td><td>0.0342</td></tr><tr><td>AUPRO</td><td>0.8179</td><td>0.8122</td><td>0.8125</td><td>0.8040</td><td>0.7872</td><td>0.7817</td><td>0.7466</td></tr></table>

Table 10: Ablation on the feature-weighting parameter λ. $\lambda = 0$ corresponds to using RGB features only. AUROC, AP, and AUPRO are reported on CHAOS, FFHQ, and DVXray.
<table><tr><td>Dataset</td><td>Metric</td><td> $\lambda = 0$ </td><td> $\lambda = 0 . 2 5$ </td><td> $\lambda = 0 . 5$ </td><td> $\lambda = 0 . 7 5$ </td><td> $\lambda = 1 . 0$ </td><td> $\lambda = 1 . 5$ </td><td> $\lambda = 2$ </td><td> $\lambda = 4$ </td><td>Feature only</td></tr><tr><td rowspan="4">CHAOS</td><td>AUROC</td><td>0.8418</td><td>0.9638</td><td>0.9632</td><td>0.9625</td><td>0.9620</td><td>0.9613</td><td>0.9609</td><td>0.9601</td><td>0.9592</td></tr><tr><td>AP</td><td>0.0729</td><td>0.3893</td><td>0.4318</td><td>0.4356</td><td>0.4318</td><td>0.4192</td><td>0.4056</td><td>0.3645</td><td>0.2733</td></tr><tr><td>AUPRO</td><td>0.5689</td><td>0.8764</td><td>0.8778</td><td>0.8772</td><td>0.8764</td><td>0.8755</td><td>0.8748</td><td>0.8735</td><td>0.8717</td></tr><tr><td>AUROC</td><td>0.7605</td><td>0.8515</td><td>0.8488</td><td>0.8459</td><td>0.8437</td><td>0.8406</td><td>0.8386</td><td>0.8347</td><td>0.8293</td></tr><tr><td rowspan="3">FFHQ</td><td>AP</td><td>0.0677</td><td>0.1177</td><td>0.1115</td><td>0.1075</td><td>0.1041</td><td>0.0997</td><td>0.0975</td><td>0.0944</td><td>0.0916</td></tr><tr><td>AUPRO</td><td>0.3688</td><td>0.5320</td><td>0.5353</td><td>0.5345</td><td>0.5331</td><td>0.5305</td><td>0.5287</td><td>0.5252</td><td>0.5211</td></tr><tr><td>AUROC</td><td>0.9607</td><td>0.9589</td><td>0.9522</td><td>0.9454</td><td>0.9397</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">DVXray</td><td>AP</td><td>0.0568</td><td>0.0535</td><td>0.0425</td><td>0.0336</td><td>0.0285</td><td>0.9305</td><td>0.9235</td><td>0.9057</td><td>0.8088</td></tr><tr><td></td><td></td><td>0.7980</td><td>0.7686</td><td>0.7432</td><td>0.7223</td><td>0.0232 0.6905</td><td>0.0205 0.6675</td><td>0.0165</td><td>0.0115</td></tr><tr><td>AUPRO</td><td>0.8179</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.6122</td><td>0.4791</td></tr></table>