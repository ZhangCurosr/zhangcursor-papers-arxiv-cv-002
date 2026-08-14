# Reconstructing Historical Manuscripts through MSI: The Potential of Contrast in Assessing Image Quality and Legibility

Anna Breger<sup>1[0000</sup>−<sup>0001</sup>−<sup>8878</sup>−<sup>5743]</sup>

<sup>1</sup>Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, UK

Abstract. Digital restoration of historical manuscript images aims to improve readability while preserving the authenticity of cultural heritage documents. However, evaluating quality of restored manuscripts remains challenging, where readability is often subjective and expert annotations are scarce. This study investigates the suitability of contrast-based image quality measures to assess quality and legibility of reconstructed manuscript images from multi-spectral imaging. Two experiments were conducted with publicly-available data sets, facilitating manual quality scores by experts and full-reference image quality measures as reference evaluations. The results show that potential contrast achieves the highest correlation with expert ratings, while contrast-to-noise ratio demonstrates the strongest agreement with full-reference quality measures. Overall, contrast-based measures consistently outperform general image quality measures, demonstrating their potential as objective indicators of manuscript legibility and reconstruction quality.

Keywords: Historical Manuscript Data · IQA · Potential Contrast

## 1 Introduction

Digital restoration of historical manuscripts remains a central topic in document image analysis due to severe degradations that afect many cultural heritage collections. Common degradations such as ink fading, bleed-through and paper aging significantly reduce legibility and hinder both human interpretation and downstream processing tasks such as transcription and recognition. In recent years, substantial progress has been made with data-driven and deep learning based approaches for document analysis and enhancement [14] and through task-aware objectives have also improved readability of hand-written texts, see e.g. [13]. Moreover, multi-spectral imaging (MSI) has emerged as a powerful noninvasive tool for the analysis and restoration of historical manuscripts by acquiring images across ultraviolet, visible, and near-infrared wavelengths. Together with advanced post-processing, MSI has demonstrated its power in revealing information that has otherwise not been visible, see e.g. [8],[12],[9].

Automated evaluation of text legibility remains a challenging and largely open problem [6], although highly needed for task-driven restoration frameworks and larger amounts of data. Common image quality assessment (IQA) measures are primarily designed for natural images and often fail to capture legibility in degraded documents. In order to understand suitability of IQA measures, dedicated evaluation protocols and comprehensive data sets would be needed for existing as well as newly developed approaches. However, available data sets that enable the assessment of IQA methods for historical manuscript legibility remain very limited. Some data sets have been developed, but never published, e.g. [19], others had been made available in the past but disappeared over time, e.g. [20]. The few available data sets include the subjective assessment framework SALAMI, that provides expert-based legibility annotations for manuscript images [5], as well as the Parchment data set, that includes artificially degraded parchment patches and their untreated references, in its original form [11] and modified [4]. Both data sets are based on MSI of the degraded manuscripts.

Potential Contrast (PC) has been proposed in [21] as a task-dependent IQA measure that estimates the maximum achievable contrast between pixel classes under arbitrary intensity transformations. Unlike traditional contrast measures, it explicitly accounts for the possibility that relevant information may become visible only after suitable grayscale transformations, making it particularly relevant for cultural heritage applications by incorporating the possibility that the user might adjust the contrast in the viewing process. Its formulation allows user-guidance through annotated pixel sample sets and has been shown to be efective in identifying informative structures in degraded document images, e.g. [10]. Recently, it has been modified to be independent from the underlying data type, cf. [17], and throughout our experiments we will employ this Normalized Potential Contrast (NPC).

Despite its promising properties, the suitability of (N)PC as an IQA measure for historical manuscript legibility has not yet been systematically investigated. In particular, its relationship to text legibility and its behavior compared to other IQA measures when judging manuscript enhancement quality remains unclear. Given the increasing reliance on objective measures for evaluating restoration pipelines, we aim to provide first insights here. We design two experiments with the described data sets, analyzing the rank correlation of NPC values and expert ratings as well as reference-based quality evaluation across common degradations such as ink fading and bleed-through. For comparison we also report the contrast measure contrast-to-noise ratio (CNR), simple image statistics such as the root mean square contrast (RMSC) and entropy, as well as common no-reference (NR) IQA measures BRISQUE [15], NIQE [16] and PIQE [22]. CNR can be understood as a direct complementing contrast measures by using the same sample masks in the computation as NPC. BRISQUE, NIQE and PIQE have been developed for natural images and do not require any reference information.

## 2 Data and Methods

## 2.1 Experiment 1: SALAMI Data [3][5]

The SALAMI dataset (Subjective Assessments of Legibility in Ancient Manuscript Images), published in 2020, contains grayscale manuscript images (900  900 pixels) derived from MSI data of 48 historical manuscripts, where 5 output reconstructions of 50 distinct image regions are shared publicly. The resulting 250 images had been annotated region-wise by 20 experts with philology and paleography background regarding legibility, yielding score maps for each image. The data set serves as a benchmark for developing and evaluating quantitative measures of legibility and digital restoration quality in historical manuscript imaging. A first IQA analysis with the data set was published by the data collectors in [6], reporting the Spearman Rank Correlation Coeficients (SRCC) of several quality measures and the mean score maps, including simple image statistics, text detection and NR IQA measures, over several window sizes and dedicated text areas. (N)PC had not been included in the study as it requires defined foreground and background pixels, which also holds true for CNR.

For our complementing experiments we choose from the 50 distinct images the ones that contain writing over the whole image and where at least one of the provided 5 versions of each image allows automated text and background extraction via direct thresholding yielding objective binary annotation masks. This selection process results in 55 test images (11 images with 5 versions each), in which we compute 10000 random patches of the size 200 400 to account for the region-wise score maps, respectively. In Figure 1 we show two examples of random patches and corresponding mean score maps.

## 2.2 Experiment 2: Parchment Data [4] and Random Reconstructions

The Parchment data set, published in 2019, and available at [2], is a modified subset of the original data set [11] published in 2017. The image data is based on multispectral acquisitions of 18th-century parchment manuscript fragments written with iron-gall ink, imaged before and after controlled artificial degradation treatments. It holds 22 parchment patches (720 720) with diferent treatment, including untreated control samples, and provides registered multispectral image data with 21 spectral bands from 400 to 950 nm. The degradation procedures simulate common forms of manuscript deterioration, enabling direct evaluation of the degraded images with their intact counterparts serving as a ground truth.

Here, we chose 4 patches (208R, 305R, 309R, 602V) of the provided data set, restricting the experiment to patches with severe degradations/illegible parts after treatment, as well as requiring to have some signal information of the degraded bits in at least one of the 21 MSI bands, i.e. enforcing possible reconstruction of the illegible notation, see examples in Figure 2. Furthermore, we cropped the images to the degraded parts, e.g. if the lower half became illegible after the treatment, but the upper half remained perfectly intact, then only the lower half was evaluated to ensure feasible results representing the degraded regions. Binary masks for text and background were created manually on small regions therein, employing a heuristically chosen MSI band with suficient remaining information, mimicking a real use case without available ground truth. For each image 15 minutes were set to create the masks in the software GIMP, imitating a research situation with constraint timelines.

Next, we use the MSI data of the treated patches to create with random orthogonal projections a set of 100000 reconstructed grayscale outputs per image to be evaluated for quality beyond the 25 provided outputs per image.

Dimension reduction with orthogonal projections. Let $x \in \mathbb { R } ^ { d }$ be a $h i g h -$ dimensional image. A random lower-dimensional representation is given by $p x \in$ $V ,$ , where $V \subset \mathbb { R } ^ { d }$ denotes a k-dimensional linear subspace with $k < d$ and $p \in$ R<sup>d</sup>×<sup>d</sup> an orthogonal projection distributed according to the unique orthogonally invariant probability measure on the Grassmannian. The dimension d is reduced to k in practice via elements $q \in \mathbb { R } ^ { k \times d }$ of the Stiefel manifold with $q ^ { T } q = p$

Random projections can be computed eficiently via the QR composition, cf. [7], and it has been shown that they give a good representation of the whole space whilst preserving important properties, cf. [1]. Here, we do have $d = 2 1$ MSI bands and reduce the images directly to a grayscale output, i.e. $k = 1$ , in which case it corresponds to the uniform distribution on the sphere.

With ground truth images available, we employ 3 full-reference (FR) IQA measures that have shown stable behavior to assess structural information or, in particular, text legibility, as references of quality by comparing the random reconstruction to the untreated ground truth data. Namely, we employ HaarPSI [18] based on Haar wavelets, Pearson correlation [4] and the multi-scale SSIM [23]. As suggested in the original paper [4], we implement the evaluation invariant to polarity. Then, the SRCC is computed between the tested quality measures and these 3 reference measures. As an additional experiment, we determine for each of the four images the highest-ranked MSI band among the 21 available bands for every tested quality measure. Please find the results in the supplementary material.

![](images/3f096d0acda57f323da677908fff80b49a891a0789dded0b6d8fc738f982161d.jpg)  
(a) npc = 0.56

![](images/8406589ef8ac2a60d86bd02a1a0aeb25d77ed630b31127b7626cd004cbd4d0a5.jpg)  
(b) npc = 0.13

![](images/a22d1c7261553c8d7d38e58ebbd50026432f5deeae453346bf54d91630b48410.jpg)  
(c) mean score map

![](images/3726f71b6a022e0f970d0c4fbe0b997cff3ab62425672db82e03a313cf289df3.jpg)  
(d) mean score map

Fig. 1. Visual examples of two random patches (a)(b) in the SALAMI data set with corresponding NPC values and mean score maps (c)(d) stored as uint8 images, where white (255) denotes the best and black (0) the worst score. The values of NPC are in [0,1], with 1 denoting the best judgement. The left example shows good matching between the NPC and the expert ratings with an average score over the map of 126.54, the right example shows a pitfall of $\mathrm { N P C , }$ judging the quality poorly although it is rated with an average score of 188.95 by the experts.

![](images/39b8296ca7566d63e6800febcbdff7868f42b68041a5eb28531171ed62ecfc2a.jpg)  
(a) cnr = 1.51

![](images/64ad1ed4c5a455517b8fdb83778b457358ed00193a02a23e134e9dd54b766f15.jpg)  
(b) cnr = 2.28

![](images/f473eeeb029589f8d874a603c2ad063b3758e9f11a477a5cac28d1e868fcf96c.jpg)  
(c) cnr = 3.03

![](images/d6877ffd6c733c53817b647f111ba22519db7483ce20023b1d41175dea3860da.jpg)  
(d) npc = 0.40

![](images/0fa7d605a2342d77d21ce8062a59d8d5de3f6453bfe247d23e28f4e7b6a93687.jpg)  
(e) npc = 0.59

![](images/55b55040185d22a3819fe8375a42b65574e7040d62129e93671fb28ac7539438.jpg)  
(f) npc = 0.77  
Fig. 2. Visual examples of random reconstructions of the MSI data of fol.305R in the Parchment data set. The images are ordered in each row in regards to the quality assessment by CNR (top) and NPC (bottom), with increasing quality values from the left to the right, where the right gives the highest quality value, respectively, within the 100000 random reconstructions.

## 3 Results

In the quantitative results we will state the absolute Spearman Rank Correlation Coeficient (SRCC) of the measures’ quality evaluation and the references, indicating how well the tested measures align with the reference judgement. In the qualitative analysis we exemplify results of the highest achieving IQA measures.

## 3.1 Experiment 1

In Table 1 the correlation results of the IQA measures and mean score maps are shown. The highest result was achieved by NPC with a mean correlation of 0.6784 over the 10000 random patches and 0.6561 for the full image comparison.

In Figure 1, we visualize 2 random patches with their NPC score and the corresponding mean score maps. The first example shows a patch where the NPC value and score map align well, whereas in the second example NPC fails.

<table><tr><td rowspan=1 colspan=1>IQA method</td><td rowspan=1 colspan=2>|BRISQUE|CNR</td><td rowspan=1 colspan=1>ENTROPY</td><td rowspan=1 colspan=1>NPC</td><td rowspan=1 colspan=1>NIQE</td><td rowspan=1 colspan=1>PIQE</td><td rowspan=1 colspan=1>|RMSC</td></tr><tr><td rowspan=1 colspan=1>SRCC mean10kvariance10k</td><td rowspan=1 colspan=1>0.05620.0021</td><td rowspan=1 colspan=1>0.53320.0039</td><td rowspan=1 colspan=1>0.48050.0009</td><td rowspan=1 colspan=1>0.68060.0005</td><td rowspan=1 colspan=1>0.20310.0005</td><td rowspan=1 colspan=1>0.16300.0002</td><td rowspan=1 colspan=1>0.58960.0012</td></tr><tr><td rowspan=1 colspan=1>SRCC full</td><td rowspan=1 colspan=1>0.1709</td><td rowspan=1 colspan=1>0.4627</td><td rowspan=1 colspan=1>0.5136</td><td rowspan=1 colspan=1>0.6561</td><td rowspan=1 colspan=1>0.2341</td><td rowspan=1 colspan=1>0.1558</td><td rowspan=1 colspan=1>0.6150</td></tr></table>

Table 1. Absolute SRCC values for the tested IQA measures and the mean annotation scores over all 55 images with the 10000 random patches of size 200 × 400 (mean and variance) as well as the full images. The highest correlation value is highlighted in bold.

## 3.2 Experiment 2

We report in Table 2 the absolute SRCC over 100000 random reconstructions for each of the 4 images between the tested IQA measures and the chosen set of FR-IQA measures, i.e. Haar-PSI, Pearson Correlation and MS-SSIM, serving as a reference judgement set. Moreover, in Figure 2, we show 3 random reconstructions ordered by IQA values from CNR and NPC, the two measures that obtained the highest correlation results. It is important to note that NPC is invariant to linear brightness/contrast changes and we display in Figure 3 a visual result when adjusting the image with intensity transformations.

<table><tr><td rowspan=1 colspan=1>IQA</td><td rowspan=1 colspan=1>||BRISQUE|</td><td rowspan=1 colspan=1>CNR</td><td rowspan=1 colspan=1>ENTROPY</td><td rowspan=1 colspan=1>NPC</td><td rowspan=1 colspan=1>NIQE</td><td rowspan=1 colspan=1>PIQE</td><td rowspan=1 colspan=1>RMSC</td></tr><tr><td rowspan=1 colspan=1>Image 1Mean</td><td rowspan=1 colspan=1>.08,.10,.120.10</td><td rowspan=1 colspan=1>.84,.83,.910.86</td><td rowspan=1 colspan=1>.64,.55,.470.55</td><td rowspan=1 colspan=1>.76,.75,.860.79</td><td rowspan=1 colspan=1>.16,.27,.240.22</td><td rowspan=1 colspan=1>.01,.23,.240.16</td><td rowspan=1 colspan=1>.01,.03,.040.03</td></tr><tr><td rowspan=2 colspan=1>Image 2Mean</td><td rowspan=2 colspan=1>.13,.18,.180.16</td><td rowspan=2 colspan=1>.84,.95,.940.91</td><td rowspan=2 colspan=1>.90,.55,.750.73</td><td rowspan=1 colspan=1>.85,.94,.94</td><td rowspan=2 colspan=1>.58,.32,.520.47</td><td rowspan=2 colspan=1>.17,.01,.010.06</td><td rowspan=2 colspan=1>.77,.34,.590.57</td></tr><tr><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=2 colspan=1>Image 3Mean</td><td rowspan=2 colspan=1>.06,.06,.090.07</td><td rowspan=2 colspan=1>.86,.96,.880.90</td><td rowspan=2 colspan=1>.66,.48,.640.59</td><td rowspan=2 colspan=1>.79,.76,.79.78</td><td rowspan=1 colspan=1>.68,.50,.67</td><td rowspan=2 colspan=1>.42, .32, .430.39</td><td rowspan=2 colspan=1>.55,.34,.530.48</td></tr><tr><td rowspan=1 colspan=1>0.61</td></tr><tr><td rowspan=1 colspan=1>Image 4Mean</td><td rowspan=1 colspan=1>.11,.09,.000.07</td><td rowspan=1 colspan=1>.53,.94,.58.68</td><td rowspan=1 colspan=1>.23,.53,.070.27</td><td rowspan=1 colspan=1>.50,.82,.500.61</td><td rowspan=1 colspan=1>.50,.47,.490.49</td><td rowspan=1 colspan=1>.29,.40,.330.34</td><td rowspan=1 colspan=1>.22,.49,.060.26</td></tr></table>

Table 2. Absolute SRCC values of the tested IQA measures and the reference measures HaarPSI, Pearson Correlation and MS-SSIM, respectively as well as the mean, over all 100000 random reconstructions. The highest correlation values are highlighted in bold.

![](images/6be89d93f84483f78f5de34ca05591b9308aff69cef40096fd4dc0fb7b54ea36.jpg)  
(a) npc = 0.77

![](images/6ee9b00dee217444206ec3a102f66fc9d27ed6b4be7899e0b1fbdd74c8af579a.jpg)  
(b) npc = 0.77  
Fig. 3. NPC is invariant to linear brightness/contrast changes in the image. This property does not hold for the FR IQA measures it has been compared to in Experiment 2, influencing the correlation result.

## 4 Discussion and Limitations

For Experiment 1 we can observe in Table 1 that NPC clearly yields the highest correlation to the mean score maps derived from the experts’ ratings, for the random patches as well as the full images, with the patch-wise computation yielding a higher result. This behavior is expected, since the mean score maps were created region-wise and the NPC computation over the whole image might not reflect local quality changes. The low variance confirms stability across the random patches. It is notable that the 3 contrast measures (NPC, RMSC and CNR) yield the 3 best results, indicating that indeed contrast notions can be helpful to identify image restoration quality as well as legibility. Nevertheless, NPC by no means acts perfectly, which is reflected in the overall correlation value of 0.68 and possible mismatches as displayed in Figure 1.

For Experiment 2, which has a more complex design since manual annotations are not available in the employed data set, we can observe in Table 2 that CNR outperforms all tested measures, followed by NPC. This diferent behavior in comparison to Experiment 1 is not surprising because NPC is invariant to intensity transformations, which aligns well with quality ratings where humans might adjust the intensity when viewing, but does not align necessarily well with FR IQA measures that rely on computation in the intensity scaling that is given. In Figure 3 we show the visual diference that can be obtained while NPC remains the same. This property can be useful in practice when it is possible to adjust the brightness/contrast of an image output, e.g. with an image viewer. Figure 2 demonstrates that both, CNR and NPC, produce meaningful orders of image quality.

Limitations of this study include that expert-annotated quality scores were not available for the Parchment dataset; therefore, 3 FR-IQA measures were used as references instead. While these measures capture meaningful aspects of image quality, they do not necessarily represent human expert judgment. In addition, NPC and CNR depend on manually created masks, introducing subjectivity into the evaluation. Finally, larger and more diverse datasets are needed to improve robustness and generalizability and to capture a wider range of manuscript degradations. Further research shall study the behavior of recent deep-learning based and other document-specific image quality measures and include diferent image transformations to better assess their suitability for legibility evaluation in historical manuscripts. Note that Experiment 2 adds to the study [6], where results with complementing image quality measures may be found.

## 5 Summary

In conclusion, the results indicate that contrast-based measures are valuable tools for assessing image restoration quality and legibility in historical manuscripts, tested here on images derived from MSI data. NPC showed the strongest performance when compared to human-derived quality maps, while CNR achieved the best results against full-reference image quality measures. Although both measures demonstrate great potential for legibility evaluation, limitations in the study are given. Future research shall explore additional quality measures and larger datasets to improve the reliability and generalizability of the study.

Acknowledgments. A.B. acknowledges funding by the Cambridge Centre for Data-Driven Discovery and Accelerate Programme (grant LEAH/011) through a donation by Schmidt Sciences.

Disclosure of Interests. The author has no competing interests to declare.

## References

1. Breger, A., et al.: On orthogonal projections for dimension reduction. Journal of Mathematical Imaging and Vision 62(3) (2020)

2. Brenner, S.: On the use of artificially degraded manuscripts for quality assessment of readability enhancement methods - dataset & code (2019). https://doi.org/ 10.5281/zenodo.2650152

3. Brenner, S.: SALAMI - Subjective Assessments of Legibility in Ancient Manuscript Images (2020). https://doi.org/10.5281/zenodo.4270352

4. Brenner, S., Sablatnig, R.: On the use of artificially degraded manuscripts for quality assessment of readability enhancement methods. In: Proceedings of the ARW & OAGM Workshop 2019. Verlag der Technischen Universität Graz (2019)

5. Brenner, S., Sablatnig, R.: Subjective assessments of legibility in ancient manuscript images - the salami dataset. In: Pattern Recognition. ICPR International Workshops and Challenges (2021)

6. Brenner, S., et al.: Estimating human legibility in historic manuscript images - a baseline. In: Document Analysis and Recognition - Proceedings of ICDAR (2021)

7. Chikuse, Y.: Statistics on Special Manifolds, Lecture Notes in Statistics, vol. 174. Springer-Verlag, New York, NY (2003)

8. Dufy, C.: Multi-spectral imaging at the british library. In: 3rd Digital Heritage International Congress. pp. 1–4 (2018)

9. Easton, R.L., et al.: Standardized system for multispectral imaging of palimpsests. vol. 7531. International Society for Optics and Photonics, SPIE (2010)

10. Faigenbaum-Golovin, S., et al.: Multispectral imaging reveals biblical-period inscription unnoticed for half a century. PLOS ONE 12(6), 1–10 (2017)

11. Giacometti, A., et al.: Evaluating multispectral image processing methods for the analysis of primary historical texts. Digital Scholarship in the Humanities 32 (2017)

12. Janke, A., et al.: A second look at multispectral data of late medieval music manuscripts. Manuscript Studies 9(1), 90–117 (2024)

13. Khamekhem Jemni, S., et al.: Enhance to read better: A multi-task adversarial network for handwritten document image enhancement. Pattern Recognition (2022)

14. Lombardi, F., Marinai, S.: Deep learning for historical document analysis and recognition-a survey. J Imaging 6(10) (2020)

15. Mittal, A., et al.: No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing 21(12), 4695–4708 (2012)

16. Mittal, A., et al.: Making a "completely blind" image quality analyzer. IEEE Signal Processing Letters 20(3), 209–212 (2013)

17. Peaslee, W., et al.: Potential contrast: Properties, equivalences, and generalization to multiple classes. In: Proc. of 33rd EUSIPCO Conf. (2025), https://github. com/wallacepeaslee/Multiple-Class-Normalized-Potential-Contrast

18. Reisenhofer, R., et al.: A Haar wavelet-based perceptual similarity index for image quality assessment. Signal Processing: Image Communication 61 (2018)

19. Shahkolaei, A., et al.: Mhdid: A multi-distortion historical document image database. In: IEEE Workshop ASAR (2018)

20. Shahkolaei, A., et al.: Subjective and objective quality assessment of degraded document images. Journal of Cultural Heritage 30, 199–209 (2018)

21. Shaus, A., et al.: Potential contrast – a new image quality measure. In: IS&T Electronic Imaging Symposium. pp. 52–58 (2017)

22. Venkatanath, N., et al.: Blind image quality evaluation using perception based features. In: International Conference on SPC. IEEE (2015)

23. Wang, Z., et al.: Multi-scale structural similarity for image quality assessment. In: The 37th Asilomar Conference on Signals, Systems & Computers. IEEE (2003)

# Supplementary Material to Reconstructing Historical Manuscripts through MSI: The Potential of Contrast in Assessing Image Quality and Legibility

Anna Breger<sup>1[0000</sup>−<sup>0001</sup>−<sup>8878</sup>−<sup>5743]</sup>

<sup>1</sup>Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, UK

## A. 1. Formulas

Spearman Rank Correlation To assess the alignment of the manual ratings and the IQA measures’ evaluations, we employ the Spearman Rank Correlation (SRCC), see $\mathrm { e . g . }$ [5]. In contrast to the Pearson Correlation Coeficient, which computes the linear relationship of two variables, the SRCC computes how strongly the ranks of the entries of a vector $x \in \mathbb { R } ^ { n }$ (here, containing the manual quality ratings by the expert graders) and $y \in \mathbb { R } ^ { n }$ (here, containing the image evaluation values provided by an IQA measure), correlate. The SRCC can be written as

$$
\operatorname { S R C C } ( x , y ) = 1 - { \frac { 6 \sum _ { i = 1 } ^ { n } ( \operatorname { R } ( x _ { i } ) - \operatorname { R } ( y _ { i } ) ) ^ { 2 } } { n ( n ^ { 2 } - 1 ) } } ,
$$

where $\mathrm { R } ( x _ { i } )$ denotes the rank of the i-th entry of the vector $x ,$ corresponding here to the rating of the i-th image.

Contrast-to-noise ratio The contrast-to-noise ratio (CNR) for an image $I \subset$ R<sup>m</sup>×<sup>n</sup> and two image regions $A \subset I$ and $B \subset I$ can be written as

$$
\mathrm { C N R } = \frac { \left| \mu _ { A } - \mu _ { B } \right| } { \sigma } ,
$$

where $\mu _ { A }$ denotes the mean signal intensity in the region A, analogously for $B ,$ and $\sigma$ the background noise. Here, A corresponds to the foreground text, B to the background and we employ the standard deviation of the background to compute $\sigma ,$ i.e. the intensity variation of the parchment/paper background. A high CNR then indicates a high separability between the writing and the manuscript substrate, and a low CNR indicates that text and background might be similar in intensity or that the manuscript substrate obscures the writing.

(Normalized) Potential Contrast Due to its complexity we refer to the original papers for the definition of potential contrast (PC), cf. [7], and normalized potential contrast (NPC), cf. [6]. Note that we employ in all experiments the same binary text and background masks for the computation of NPC and CNR.

## A.2. Additional Experiments with Parchment Data [3][1]

In the main paper, 100000 random reconstructions of the parchment MSI data were used to create a large test set with varying levels of reconstruction quality. Here, complimenting that experiment, we provide more detailed insights by analyzing the quality evaluation of the provided MSI data for each of the 4 images. In particular, for the chosen quality measures we compute the quality values of all 21 bands and compare which bands has been chosen as the best in regards to each quality measure. Again, the full-reference IQA measures HaarPSI, Pearson Correlation and MS-SSIM serve as the point of reference. In Table 1 the results are summarized and chosen bands are displayed in Figure 1. We can see that CNR and NPC were able to reliably pick the best band as chosen by the FR-IQA measures, and on the other hand, that the common IQA measures BRISQUE, NIQE and PIQE failed for that task. In the visualization we can observe that the choices of those measures, demonstrated with BRISQUE, are inconsistent and sometimes even choose a band with hardly any remaining signal information.

<table><tr><td rowspan=1 colspan=1>IQA</td><td rowspan=1 colspan=8>FR-IQA |BRISQUE|CNR|ENTROPY| NPC|NIQE |PIQE|RMSC</td></tr><tr><td rowspan=1 colspan=1>Image 1</td><td rowspan=1 colspan=1>(15, 15, 15)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>Image 2</td><td rowspan=1 colspan=1>(19, 19, 19)</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>17</td></tr><tr><td rowspan=1 colspan=1>Image 3</td><td rowspan=1 colspan=1>(18, 17, 18)</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>Image 4</td><td rowspan=1 colspan=1>(3, 19, 3)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>10</td></tr></table>

Table 1. MSI bands that were rated best by the tested quality measures and the FR-IQA measures HaarPSI, Pearson Correlation and MS-SSIM over the 21 MSI bands per image. Closest match (majority vote) to the FR-IQA choice is highlighted in bold. Note that bands have been ordered according to their scanning wavelengths, i.e. closer band numbering indicates a more similar acquisition wavelength.

Moreover, we tested if the untreated ground truth image would be correctly identified as best image when added to the MSI stack, see Table 2. For all tested quality measures CNR and NPC were the only measures that correctly identified the ground truth as the best quality for all 4 images. Note that the ground truth data was not used to create the binary annotation masks for CNR and NPC. FR-IQA measures have been excluded, since the reference for their computation is the ground truth image.

<table><tr><td rowspan=1 colspan=8>IQA |BRISQUE|CNR|ENTROPY| NPC|NIQE |PIQE|RMSC</td></tr><tr><td rowspan=1 colspan=1>Image 1</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Image 2</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Image 3</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Image 4</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td></tr></table>

Table 2. Measures are marked if they were able to identify the ground truth as the best quality image when added to the 21 MSI bands.

The results in the additional experiments support the trend in the experiments of the main paper, showing that NPC and CNR produce meaningful results when applied as quality measures for legibility, whereas common measures such as BRISQUE fail to provide consistent results for that task.

![](images/d4bb46b71200be2e9be0cad432c30599f6fe290332bf35dd782081f054069ddd.jpg)  
(a) FR-IQA

![](images/97f747764063e2fa9a7bca548e1fb00b51a07928adf34c1b80f7368ded9e9a70.jpg)  
(b) CNR

![](images/b46dec5d532cdd0161c350d1d8d7e46eb38cf5776e173572971a09fa110ff206.jpg)  
(c) NPC

![](images/94889a4950d1c03a1492605438efcde8f391bba6a0121409efdfbd1572e5bc7b.jpg)  
(d) BRISQUE  
Fig. 1. Visualization of results in Table 1 with best band picked by majority vote FR-IQA references (left), CNR (2nd col), NPC (3rd col) and BRISQUE (right). Images are cropped to the degraded parts as used in the analysis.

## A.3. Additional Figures

## SALAMI Data [2][4]

In Figure 2 we display 2 full example images with corresponding mean score maps from the expert annotations. Examples of random patches (size 200 400) are displayed in the main paper (Figure 1). In Figure 3 we show the binary masks derived by thresholding for the computation of CNR and NPC. Note that the patch-wise computation resembles a more realistic experimental set-up since text annotations for the whole image are usually too time-consuming.

![](images/9fe2afc0b757c0ff5fcbf5343d5f924c9975ef08bba5606800ca023744c8d8dd.jpg)  
Fig. 2. Example images of the SALAMI data set with their corresponding provided mean score maps.

![](images/bd951b206a9b271c4d729dc24b080db75053eaf7cbbf30c80e69375bd01cd386.jpg)

![](images/9e87b925a4568f91675c41adcfb804719658b2d3de7e737a32ffff542f5d887b.jpg)

![](images/42c5bc8f0f67b077f52ac93766faf230219cd697f12d15057f2d1a9f483d5b88.jpg)  
Fig. 3. Example reference image (left) of the SALAMI data set and derived binary masks for text (middle) and random background (right).

## Parchment Data

In Figure 5 we display the untreated ground truth images next to the treated (i.e. degraded) references and the manual annotations used for the computation of CNR and NPC. In Figure 4 we extend Figure 2 in the main paper, adding PIQE and BRISQUE to the visual evaluation, where we can observe that the provided ordering fails to represent visual quality and legibility.

![](images/11b354ba45d6f71d2ba09613d6a8bd66fa77af2d026ef01c17cafca1276c0243.jpg)  
(a) cnr = 0.77

![](images/be6b01889f7576671caac04bd8a0e0a2fbf8dcb57002a151398f6acabb60f3ef.jpg)  
(b) cnr = 1.51

![](images/411ce7d72e643ce03c5680430d75ff79b8b2f116654379bc01ea98c758302c53.jpg)  
(c) cnr = 2.28

![](images/eb56e852d1bd3bb47a463304c6168ca3d47e2619d198ae509c1ec140100524ab.jpg)  
(d) cnr = 3.03

![](images/32f46712f8d639f4290f3184f01c9e9e3c907a274a4b67dc1a0813000f6a8581.jpg)  
(e) npc = 0.20

![](images/53d6ae376143137d7ac77368b61ab12ae6856c5473ee82640188605c398fbe47.jpg)  
(f) npc = 0.40

![](images/35e925a70db728a4871bbde7daa8ca8c57b140d4ec1cc3bce4df49a5f0b02030.jpg)

![](images/a23da8a3fe61b65134138e69b9a3c154250347ed51953447d69aabd7422ae62b.jpg)  
(g) npc = 0.59  
(h) $\mathrm { n p c } = 0 . 7 7$

![](images/fde317ba4bd272deb6768e56ab6c8a320b571e2decdece56446bd84523cb2de7.jpg)  
(i) piqe = 43.45

![](images/bf8a095733c99a594eeb03e75dba7452ed2ab429f7a6e97f3aa5003ed567c297.jpg)

![](images/4f52045211c945dd9ec0da5997e6cfa5ea4df3a9472b51c33c8f94a7175a5c7d.jpg)  
(j) piqe = 30.31

![](images/791e6b0c5b41b4907fe74ff31992c7d34e680f52d7c0ba9b1b4c50da80ae1b16.jpg)  
(l) piqe = 4.03  
(k) piqe = 17.17

![](images/2071995e821fccbc065d7e2cafdfc951a297467db0a525d7c9d45aade77af2f4.jpg)  
(m) brisque = 32.67

![](images/ee583a10d83513f819184e7938b07901b4355026dc85db258921f4a9775dde08.jpg)  
(n) brisque = 25.30

![](images/e16c3a1e6e3efe8d8292ed52552bd0e51a98eee7aa25c191f935e0300ad7e886.jpg)  
(o) brisque = 17.92

![](images/a93044fccb02d6e7439d34aed696b77e4b0ee495e5feb421c28d7a74d16eca6a.jpg)  
(p) brisque = 10.53

Fig. 4. Visual examples of random reconstructions of the MSI data of fol.305R. The images are ordered in each row in regards to the quality assessment by CNR (top), NPC (2nd row), PIQE (3rd row) and BRISQUE (bottom), with increasing quality values from the left to the right, where the right gives the highest quality value, respectively, within the 100000 random reconstructions. Note that it has been taken into account that PIQE and BRISQUE state quality in decreasing order.

![](images/69bce7719a50608f80377b1aaaebf32ce6ee8c1d675029ec0fd140a7fc1bd022.jpg)

![](images/4b2020f8d88f4ba8c6262d329942548112127e98952ae1e1fe3b0ee75dcc807e.jpg)

![](images/785946060c224a40febf27830ad94a47abd2d079e092ed6cda9f28811f5635fe.jpg)

![](images/ff29166cdf2df2a13d14b269fc16a49f11f652733e6c6b18761cf7a2fdbb1741.jpg)  
(a) Ground Truth

![](images/000f0a02a899b404bfee37bbc8807d4a47c3196289632080aad6c5bfb81cbd4d.jpg)  
(b) Treated Reference

![](images/05ef8d9decbd540fb949deba7cd88fe2e84b650dba581b99ba35ad08d4b6a8f2.jpg)  
(c) Manual Annotation  
Fig. 5. Ground truth (untreated) images, treated references and manual annotations of the 4 employed images in the Parchment data set: fol.208r, fol. 305r, fol.309r, fol.602v. Note that only the degraded parts were used for the analysis, see Figure 1 for the cropped versions of the images. The manual annotations are visualized with text in white and background in gray.

## References

1. Brenner, S.: On the use of artificially degraded manuscripts for quality assessment of readability enhancement methods – dataset & code (2019). https://doi.org/ 10.5281/zenodo.2650152, https://doi.org/10.5281/zenodo.2650152

2. Brenner, S.: SALAMI - Subjective Assessments of Legibility in Ancient Manuscript Images. https://doi.org/10.5281/zenodo.4270352 (2020). https://doi.org/10. 5281/zenodo.4270352

3. Brenner, S., Sablatnig, R.: On the use of artificially degraded manuscripts for quality assessment of readability enhancement methods. In: Pichler, A., Roth, P.M., Sablatnig, R., Stübl, G., Vincze, M. (eds.) Proceedings of the ARW & OAGM Workshop 2019. pp. 140–145. Verlag der Technischen Universität Graz (2019). https://doi.org/10.3217/978-3-85125-663-5-28

4. Brenner, S., Sablatnig, R.: Subjective assessments of legibility in ancient manuscript images - the salami dataset. In: Del Bimbo, A., Cucchiara, R., Sclarof, S., Farinella, G.M., Mei, T., Bertini, M., Escalante, H.J., Vezzani, R. (eds.) Pattern Recognition. ICPR International Workshops and Challenges. pp. 68–82. Springer International Publishing, Cham (2021)

5. Fieller, E.C., Hartley, H.O., Pearson, E.S.: Tests for rank correlation coeficients. I. Biometrika 44(3-4), 470–481 (12 1957). https://doi.org/10.1093/biomet/44. 3-4.470, https://doi.org/10.1093/biomet/44.3-4.470

6. Peaslee, W., Breger, A., Schönlieb, C.B.: Potential contrast: Properties, equivalences, and generalization to multiple classes. In: Proceedings of 33rd European Signal Processing Conference EUSIPCO 2025 (2025), https://arxiv.org/abs/2505. 01388

7. Shaus, A., Faigenbaum-Golovin, S., Sober, B., Turkel, E., Piasetzky, E.: Potential contrast – a new image quality measure. vol. 2017, pp. 52–58 (01 2017)