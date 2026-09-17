# Toward Markerless Video-based Tremor Analysis: Objective Quantification of Pathological Tremor in Mouse Preclinical Models

Yota Koshimoto<sup>1</sup>, Akihiro Tsukahara<sup>1</sup>, Yasuhiro Moriwaki<sup>1</sup>, and Mariko Isogawa<sup>1</sup>

Keio University, Yokohama, Japan {koshimoto.yota, mariko.isogawa}@keio.jp

Abstract. Tremor is a movement disorder characterized by involuntary, rhythmic oscillations of body parts and is a hallmark of several neurological conditions, including Parkinson’s disease and essential tremor. Elucidating its underlying mechanisms relies heavily on mouse models, which ofer genetic manipulability and translational relevance to human neural circuitry. Accordingly, these models are indispensable for studying tremor pathophysiology. So far, electromyography and accelerometers have been used as methods to quantitatively observe tremors in mice. However, these methods have several drawbacks, such as high costs and complex setups. In particular, the invasive surgical implantation of devices causes significant stress to the animals. Although RGBbased methods ofer non-invasive and cost-efective alternatives, they often lack the sensitivity required to detect subtle tremors. Therefore, this paper addresses these challenges by achieving mouse tremor severity estimation using conventional RGB cameras only. To address the challenging task of isolating tremor-related vibrations while the mouse itself is also in motion, our pipeline incorporates segmentation-based preprocessing to extract the mouse region and a Tremor Score Estimation Module that captures subtle tremors with high sensitivity. In the experiments, we assessed tremors in unrestrained mice using a non-invasive method with two standard cameras. The results demonstrated a strong correlation with accelerometer measurements and confirmed that the method accurately captured the intensity-dependent characteristics of tremors. The project page is available at https://isogawalab.github. io/Video-based-Tremor-Analysis-Project/.

Keywords: Mice tremor · Animal pose tracking · Quantitative evaluation.

## 1 Introduction

Parkinson’s disease (PD) has become an increasingly significant global health concern as the aging population continues to expand, underscoring the urgent need for efective therapeutic strategies. Tremor, defined as involuntary and rhythmic oscillatory movement, is one of the cardinal motor symptoms of PD. To investigate its underlying mechanisms and evaluate potential treatments, genetically manipulable PD model mice are widely used to reproduce key pathological and behavioral features of the disease. Accordingly, precise and objective quantification of murine tremor is essential for accurately assessing therapeutic eficacy and facilitating the translation of preclinical findings into clinical applications.

However, conventional tremor assessment relies on visual observation by human evaluators [12, 20], leading to issues of subjectivity and inconsistency. Furthermore, such manual evaluation methods are time-consuming and sufer from poor reproducibility. To ensure objectivity, previous studies have employed invasive, contact-based approaches such as accelerometers [7, 13, 17], electromyography sensors [2, 6, 9], and marker-based motion capture systems [11]. While quantitative, these methods restrict natural movement and may compromise behavioral observations.

To address this issue, alternative non-invasive methods using piezoelectric film sensors [1] and force plate-based systems [5, 16] have been proposed. However, these solutions require specialized equipment and high costs. Using inexpensive and widely used conventional RGB cameras is one possible option. However, the existing implementation [16] has two major limitations in tremor detection. First, because the method depends solely on movement velocity, it lacks the sensitivity to detect the subtle oscillations that characterize tremors. Second, its reliance on a top-down perspective makes it dificult to capture tremors occurring in the anti-gravity (vertical) direction. Collectively, these constraints compromise the overall precision of tremor quantification.

To overcome these limitations, this paper proposes a framework for estimating mouse tremor severity using only RGB videos. Specifically, our framework extracts body part positions from an animal pose estimation model using a horizontal camera perspective. We incorporate a Move Detection Module to isolate involuntary oscillations from voluntary locomotion and prevent the false detection of movement as a tremor. Additionally, we propose a Tremor Score Estimation Module specifically to sensitively quantify microscopic oscillations into a tremor score. By focusing on movement along the vertical axis, our approach is uniquely capable of capturing tremors occurring in the anti-gravity direction, which were previously undetectable from top-down views. This quantitative assessment of subtle tremors facilitates the investigation of the relationship between drug dosage and tremor severity, thereby aiding in the development of therapeutic strategies for Parkinson’s disease.

In summary, our contributions are as follows.

– We propose a novel, non-invasive framework that quantifies tremor severity with high precision using only side-view RGB videos.

– We construct an original dataset of RGB videos synchronized with accelerometer data, enabling rigorous quantitative evaluation of video-based tremor assessment.

– We propose two novel modules: a Move Detection Module that distinguishes voluntary locomotion from involuntary tremor oscillations, and a Tremor Score Estimation Module that directly quantifies wide-range tremor severity from coordinates of a body part.

## 2 Related Work

## 2.1 Sensor Based Mouse Tremor Measurement

Invasive Methods. To quantitatively assess mouse tremors, various methods have been proposed. Sensor-based approaches utilizing accelerometers [7, 13, 17] or electromyography [2,6,9] require surgical implantation of devices. While these methods provide objective physiological data, they necessitate animal restraint and device attachment, which disrupt natural behavior.

In recent years, a marker-based motion capture system has emerged as an innovative approach for tremor analysis in mice [11]. This system involves surgical implantation of multiple reflective markers beneath the skin, with subsequent 3D motion tracking using synchronized high-speed cameras. This multi-sensor implantation system enables comprehensive visualization of whole-body tremor dynamics, allowing for precise anatomical localization of tremor activity. However, this approach retains significant limitations: first, the necessity of specialized surgical instrumentation, and second, the inherently invasive nature of subcutaneous sensor implantation procedures in mouse models.

Non-Invasive Methods. Acknowledging the inherent limitations of conventional tremor assessment, a study incorporating piezoelectric film-based tremor measurement with advanced image analysis methods [1] was conducted. It presents a quantitative evaluation of microtremors in unrestrained mouse subjects. However, this method requires specialized equipment, such as a piezoelectric sensor.

Recent studies employ markerless pose estimation to analyze tremor dynamics through overhead video capture [16]. While this approach attempts to quantify tremor using RGB data, it faces two fundamental limitations. First, the method relies on inter-frame movement velocity, which lacks the sensitivity to distinguish the subtle oscillations characteristic of tremors from voluntary motion. Second, the exclusive use of a top-down perspective constrains motion capture predominantly to the horizontal plane. Because tremor severity is often reflected in oscillatory movements during anti-gravity postural maintenance, insuficient representation of vertical components may reduce measurement accuracy.

While our method shares the objective of previous research to achieve objective and quantitative tremor assessment, it addresses the previously unexplored problem of estimating anti-gravity tremor severity using side-view video data acquired from RGB cameras. It allows tremor measurements to be made in a simple filming environment consisting of two conventional cameras, with minimal interference with the subject’s natural behavior and no reliance on subjective observer assessment.

![](images/b7dc716dadb2465f813bb02f6300eb9c383e7f18bd0dc991c330ee460ef562a3.jpg)  
Fig. 1: Overview of the proposed framework.

## 2.2 Video-based Animal Behavior Analysis with Pose Estimation

The advent of deep learning-based markerless pose estimation frameworks, particularly DeepLabCut (DLC) [15] and SLEAP [18], has significantly advanced quantitative behavioral analysis in mouse models. These models enable automated tracking of anatomical keypoints from standard video, facilitating objective assessment of motor phenotypes without physical instrumentation.

Recent applications demonstrate the utility of pose estimation for quantifying motor deficits in neurological disease models. For instance, skeletal keypoints in Parkinson’s disease models have been tracked using DLC [3], enabling objective classification of motor behaviors and eliminating manual scoring subjectivity. Similarly, unsupervised clustering of pose dynamics [10] successfully distinguished L-DOPA-induced dyskinesia from normal grooming behaviors. Furthermore, locomotor ataxia has been quantified by analyzing spatial variability and inter-joint coordination [14], revealing disease-specific gait signatures.

In this work, we extend the application of Animal Pose Estimation to the domain of video-based tremor analysis. Accordingly, this work proposes a novel tremor assessment method leveraging animal pose estimation models to capture tremor dynamics in a simple recording environment without restricting natural behavior, enabling the demonstration of tremor severity estimation from RGB videos.

## 3 Method

Figure 1 illustrates our framework. Given dual-view video sequences I<sup>front</sup> and I<sup>back</sup> of a freely moving mouse, our goal is to estimate tremor severity O as a scalar score. Our framework consists of three components: Sec. 3.1 Multi-view Selection to choose the optimal viewing angle, Sec. 3.2 Move Detection Module to separate locomotion from tremor, and Sec. 3.3 Tremor Score Estimation Module to quantify tremor intensity. We detail each component below.

## 3.1 Multi-view Selection Strategy

To ensure robust tracking regardless of the mouse’s orientation, our system captures video from two horizontal perspectives. Since the mouse frequently changes direction, causing self-occlusion, one view typically provides clearer visibility of the target body parts than the other. First, to isolate the subject from the complex background, we employ SAM 2 [19] for segmentation-based preprocessing, generating a segmentation mask of the mouse in each frame $\mathbf { I } _ { i }$ . From the masked region, the head position $( x _ { i } , y _ { i } )$ and its associated tracking confidence $c _ { i }$ are extracted using an animal pose estimation model. The head is selected as the primary tracking target due to its prominent tremor expression and lower occlusion risk compared to limbs. Subsequently, the trajectory is partitioned into $M$ non-overlapping segments, denoted as $\lbrace V _ { m } \rbrace _ { m = 1 } ^ { M }$ , each with a fixed duration of T seconds. For each segment, the view with the higher mean confidence $\bar { c } _ { m }$ is selected for subsequent analysis.

## 3.2 Move Detection Module

Distinguishing involuntary tremor from voluntary locomotion is crucial for accurate assessment. To address this, we propose a module that filters out intervals dominated by active movement using the trajectory selected in Sec. 3.1. Let the trajectory of the selected view for the segment $V _ { m }$ be denoted as a sequence of coordinates $\mathbf { P } _ { m } = \{ ( x _ { 0 } , y _ { 0 } ) , ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { N - 1 } , y _ { N - 1 } ) \}$ , where N is the number of frames within the segment duration T. We calculate the cumulative movement distance $d _ { m }$ as the sum of Euclidean distances between consecutive frames.

If $d _ { m }$ is below a predefined threshold, the segment is identified as a stationary state and is preserved for tremor scoring. Conversely, segments with excessive movement are discarded to prevent locomotion artifacts. In this study, we empirically set the segment duration $T = 3$ seconds (corresponding to $N = 9 0$ frames at 30 fps) and the movement threshold to 400 pixels, corresponding to approximately 80 mm using our pixel-to-millimeter calibration factor of 0.2 mm per pixel estimated from held-out pilot data.

## 3.3 Tremor Score Estimation Module

This module quantifies tremor severity within segments identified as stationary. The primary challenge in tremor quantification is to suppress non-pathological noise while sensitively capturing subtle tremors. To achieve this, we employ peak prominence analysis rather than absolute amplitude, focusing on the vertical (Yaxis) displacement. Let $\mathbf { Y } _ { m } = \{ y _ { 0 } , y _ { 1 } , \dots , y _ { N - 1 } \}$ be the Y-coordinate sequence for a segment $V _ { m }$ . For a local maximum at index $p ,$ we define its left and right reference boundaries, $L ( p )$ and $R ( p )$ , as the nearest indices where the signal height equals or exceeds $y _ { p } ,$ or the sequence endpoints:

$$
\begin{array} { l } { L ( p ) = \operatorname* { m a x } \{ i < p \mid y _ { i } \geq y _ { p } \mathrm { ~ o r ~ } i = 0 \} } \\ { R ( p ) = \operatorname* { m i n } \{ i > p \mid y _ { i } \geq y _ { p } \mathrm { ~ o r ~ } i = N - 1 \} } \end{array}\tag{1}
$$

The prominence $P ( p )$ is then determined by the vertical distance from the peak to the higher of the two local minima (bases) within these boundaries:

$$
P ( p ) = y _ { p } - \operatorname* { m a x } \left( \operatorname* { m i n } _ { i \in \left[ L ( p ) , p \right] } y _ { i } , \operatorname* { m i n } _ { i \in \left[ p , R ( p ) \right] } y _ { i } \right)\tag{2}
$$

Unlike absolute peak height, this metric considers the relative depth of surrounding valleys, providing a more robust measure of oscillatory intensity by filtering out minor positional fluctuations. The global tremor score O for the entire observation period is then calculated as the total count of peaks whose prominence exceeds a predefined threshold τ. In this study, we set $\tau = 1 . 0$ pixel, corresponding to approximately 0.2 mm according to our calibration factor.

## 4 Experimental Settings

Dataset. We collected an original dataset comprising 56 sequences of RGB videos (paired front/back views of 28 wild-type mice). We used wild-type mice divided into four groups based on harmaline dosage: 0 mg/kg (saline, n=6), 5 mg/kg (n=8), 10 mg/kg (n=7) and 20 mg/kg (n=7). Mice were placed on a custom-built suspension platform. As the onset of prominent tremors was typically observed at 2.5 minutes, we analyzed the subsequent 5-minute interval to ensure the evaluation of the active tremor phase. Videos were recorded at a resolution of $1 2 8 0 \times 7 2 0$ pixels at 30 fps. We calibrated the pixel-to-millimeter scale using a checkerboard with a known square size placed near the center of the platform. Both front- and back-view cameras were positioned at the same distance and configured with the same image resolution, yielding a local scale of 0.2 mm per pixel at the measurement plane. This scale can be re-estimated for diferent experimental setups using the same calibration procedure. To obtain ground-truth motion data, we utilized a wireless 3-axis accelerometer (MVP-RF8-S-V170, MicroStone $\mathrm { C o . }$ , Ltd.) attached to the suspension platform.

Evaluation Metric. Following established accelerometer-based tremor assessment protocols [4], we used the peak Power Spectral Density (PSD) of the accelerometer signal as the reference measure of tremor intensity. We evaluated the proposed video-based tremor score by computing its correlation with the accelerometer-derived PSD.

Baseline Method. To evaluate the performance of our framework, we compared our method with two baselines: (1) Ni et al. [16]: we employed a prior approach that estimates mouse tremor intensity from RGB video. This method involves tracking the animal’s center in overhead video via a pose estimation model and deriving a score from the peak value of the movement’s Power Spectral Density (PSD). To ensure a fair comparison using our dataset, we applied this pipeline to a side-view perspective. (2) Human-subjective scoring: A manual assessment was conducted based on the clinical scoring criteria established in Zhang et al. [20]. For each analyzed segment, an experienced evaluator scored the tremor severity, and the total sum of these scores across the measurement period was calculated to represent the subjective tremor intensity.

Table 1: Comparison of correlation coeficient
<table><tr><td>Methods</td><td>Correlation coefficient</td></tr><tr><td>Ni et al. [16]</td><td>-0.27</td></tr><tr><td>Human-subjective evaluation</td><td>0.82</td></tr><tr><td>Ours</td><td>0.86</td></tr><tr><td>Ours w/o multi-view strategy</td><td>0.82</td></tr><tr><td>Ours w/o seg. preprocessing</td><td>0.75</td></tr><tr><td>Ours w/o Move Detection Module</td><td>0.78</td></tr></table>

![](images/86a3cf95ad0aafc4730d7342c9ed8e7f64fe82e80689f2b736663768cdd2474e.jpg)  
Fig. 2: Sensitivity analysis of hyperparameters.

![](images/554c7f9da9c16dc21f008b1b3fe019fa9528dc4da15f09919c743d5444fc3932.jpg)  
Fig. 3: Relation between dosage and PSD.

![](images/23425511f8734e3c5c1fcb1757d81e2375660a7f24b1d7c5054dfdd20d286a08.jpg)  
Fig. 4: Relation between dosage and tremor score.

Implementation Details. For pose estimation, we employed DeepLabCut with a ResNet-50 backbone [8]. The model was trained on 480 frames (20 frames manually annotated from each of the 24 videos) to ensure robust tracking.

## 5 Experiments and Results

To verify the efectiveness of our framework, we conducted comparative experiments using the peak Power Spectral Density (PSD) from accelerometer data as the ground-truth for tremor intensity.

Comparison with Baseline Methods. We compared our method with a velocity-based automated baseline [16] and human-subjective scoring. In Table 1, the RGB baseline showed a negative correlation $( r = - 0 . 2 7 )$ with the accelerometer ground truth, as simple cumulative displacement is easily dominated by locomotive noise and lacks sensitivity to subtle tremors. In contrast, while human-subjective scoring achieved a high correlation $( r = 0 . 8 2 )$ , it remains inherently limited by its discrete, coarse nature and the prohibitive time-cost of manual frame-by-frame analysis, making it impractical for large-scale studies.

![](images/95519c7170898807a2cba8c774fb6b98ebc948d4f50ebbf4a6b76475928ae48a.jpg)  
Fig. 5: An example head displacement during tremor.

Ablation Study. We evaluated the contribution of each module to the estimation accuracy (Table 1). The performance declined when key components were removed, with the correlation coeficient dropping to $r = 0 . 7 5$ without the segmentation-based preprocessing and to $r = 0 . 7 8$ without the Move Detection Module. These results underscore the necessity of isolating the mouse region and filtering out locomotion artifacts to capture subtle tremor dynamics. Additionally, excluding the multi-view selection strategy resulted in a decrease to $r = 0 . 8 2$ , confirming its role in maintaining robust tracking against self-occlusion.

Sensitivity Check on Hyperparameters. We assessed the framework’s robustness against hyperparameter variations. As shown in Fig. 2, the correlation coeficient remained stable above 0.80 across various settings for both the movement threshold and prominence threshold τ. This stability indicates that our method is not overly sensitive to specific parameter tuning, ensuring reliable performance across diferent environments and tremor intensities.

Pharmacological Validation. In animal studies for Parkinson’s disease research, tremor severity under pharmacological manipulation is a key indicator for evaluating drug eficacy and disease mechanisms. Therefore, we evaluated the practical utility of our framework by replicating the physiological dose-response relationship of harmaline. We investigated whether the tremor severity scores estimated by our method increase with dosage during harmaline administration, where tremor severity is known to exhibit a dose-dependent increase [1]. A one-way ANOVA was performed to examine the efect of harmaline dosage (saline, 5, 10 and 20 $\mathrm { m g / k g ) }$ on tremor severity, revealing a significant main efect $( p \ : < \ : 0 . 0 5 )$ . As shown in Fig. 3, post-hoc tests confirmed significantly higher tremor intensity in the 10 and 20 mg/kg groups. Our vision-based tremor scores (Fig. 4) exhibited a highly concordant trend, correctly capturing the dosedependent increase. These results indicate that our non-invasive approach serves as a reliable alternative to invasive sensors for drug eficacy evaluation.

Anti-Gravity (Vertical) Tremor. As shown in Fig. 5, the mouse exhibits rhythmic vertical head oscillations during tremors. These anti-gravity movements are inherently dificult to observe from a conventional top-down view. Thus, our side-view configuration is essential to explicitly visualize these vertical displacements and comprehensively evaluate tremor severity.

## 6 Conclusion

This study proposes a non-invasive tremor measurement framework using standard RGB cameras. Leveraging mouse pose estimation and prominence-based peak detection, the method extracts subtle tremor oscillations while suppressing behavioral noise. Results demonstrate a strong correlation with accelerometerbased ground truth $( r = 0 . 8 6 )$ , achieving accuracy comparable to expert manual scoring while eliminating its time burden. Furthermore, successfully replicating dose-dependent tremor patterns highlights the system’s utility for pharmacological evaluation. Our framework overcomes the limitations of contact-based sensors and subjective observation, providing a scalable platform to accelerate drug discovery and mechanistic research. Notably, its ability to analyze tremors during resting states underscores its broad translational potential, extending its applicability to Parkinsonian tremor and other pathological movement disorders.

Ethics Approval Statement. This study was conducted with the approval of the Ethics Committee of Keio University (Approval No. A2022-341).

Acknowledgements. This work was partially supported by the KGRI Challenge Grant, JSPS KAKENHI Grant Number 25H01159, and the grant from the Smoking Research Foundation.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Ajima, A., Yoshida, T., Yaguchi, K., Itohara, S.: Objective detection of microtremors in netrin-g2 knockout mice. Journal of Neuroscience Methods 351, 109074 (2021). https://doi.org/10.1016/j.jneumeth.2021.109074

2. Bekar, L., Libionka, W., Tian, G., Xu, Q., Torres, A., Wang, X., Lovatt, D., Williams, E., Takano, T., Schnermann, J., Bakos, R., Nedergaard, M.: Adenosine is crucial for deep brain stimulation–mediated attenuation of tremor. Nature Medicine 14(1), 75–80 (2008). https://doi.org/10.1038/nm1693

3. Bidgood, R., Zubelzu, M., Ruíz-Ortega, J.A., Morera-Herreras, T.: Automated procedure to detect subtle motor alterations in the balance beam test in a mouse model of early parkinson’s disease. Scientific Reports 14 (2024). https://doi. org/10.1038/s41598-024-51225-1

4. Carlsen, E.M.M., Amrutkar, D.V., Sandager-Nielsen, K., Perrier, J.F.: Accurate and afordable assessment of physiological and pathological tremor in rodents using the accelerometer of a smartphone. Journal of Neurophysiology 122(3), 970–974 (2019). https://doi.org/10.1152/jn.00281.2019

5. Fowler, S., Birkestrand, B., Chen, R., Moss, S., Vorontsova, E., Wang, G., Zarcone, T.: A force-plate actometer for quantitating rodent behaviors: illustrative data on locomotion, rotation, spatial patterning, stereotypies, and tremor. Journal of Neuroscience Methods 107(1-2), 107–124 (2001). https://doi.org/10.1016/ S0165-0270(01)00359-4

6. Günther, H., Brunner, R., Klußmann, F.W.: Spectral analysis of tremorine and cold tremor electromyograms in animal species of diferent size. Pflügers Archiv 399(3), 180–185 (1983). https://doi.org/10.1007/BF00656712

7. Hallberg, H., Carlsson, L., Elg, R.: Objective quantification of tremor in conscious unrestrained rats, exemplified with 5-hydroxytryptaminemediated tremor. Journal of Pharmacological Methods 13(3), 261–266 (1985). https://doi.org/10.1016/ 0160-5402(85)90026-9

8. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

9. Hosoi, N., Shibasaki, K., Hosono, M., Konno, A., Shinoda, Y., Kiyonari, H., K, I., Muramatsu, S., Ishizaki, Y., Hirai, H., Furuichi, T., Sadakata, T.: Deletion of class ii adp-ribosylation factors in mice causes tremor by the nav1.6 loss in cerebellar purkinje cell axon initial segments. Journal of Neuroscience 39(32), 6339–6353 (2019). https://doi.org/10.1523/JNEUROSCI.2002-18.2019

10. Hsu, A.I., Yttri, E.A.: B-SOiD, an open-source unsupervised algorithm for identification and fast prediction of behaviors. Nat Commun 12(1), 5188 (2021). https://doi.org/10.1038/s41467-021-25420-x

11. Ignatowska-Jankowska, B.M., Swaminathan, L.I., Turkki, T.H., Sakharuk, D., Gurkan Ozer, A., Kuck, A., Uusisaari, M.Y.: Accurate tracking of locomotory kinematics in mice moving freely in three-dimensional environments. eNeuro 12(6) (2025). https://doi.org/10.1523/ENEURO.0045-25.2025

12. Jolicoeur, F.B., Rivest, R., Drumheller, A.: Hypokinesia, rigidity, and tremor induced by hypothalamic 6-ohda lesions in the rat. Brain Research Bulletin 26(2), 317–320 (1991). https://doi.org/10.1016/0361-9230(91)90245-f

13. Kistler, W.M., de Jeu, M.T.G., Elgersma, Y., Gissen, R.S.V.D., Hensbroek, R., Luo, C., Koekkoek, S.K.E., Hoogenraad, C.C., Hames, F.P.T., Gueldenagel, M., Sohl, G., Wilekke, K., de Zeeuw, C.I.: Analysis of cx36 knockout does not support tenet that olivary gap junctions are required for complex spike synchronization and normal motor performance. Annals of the New York Academy of Sciences 978(1), 391–404 (2002). https://doi.org/10.1111/j.1749-6632.2002.tb07582.x

14. Machado, A.S., Marques, H.G., Duarte, D.F., Darmohray, D.M., Carey, M.R.: Shared and specific signatures of locomotor ataxia in mutant mice. eLife 9, e55356 (Jul 2020). https://doi.org/10.7554/eLife.55356

15. Mathis, A., Mamidanna, P., Cury, K.M., Abe, T., Murthy, V.N., Mathis, M.W., Bethge, M.: DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nature neuroscience 21(9), 1281–1289 (2018). https://doi. org/10.1038/s41593-018-0209-y

16. Ni, C.L., Lin, Y.T., Lu, L.Y., Wang, J.H., Liu, W.C., Kuo, S.H., Pan, M.K.: Tracking motion kinematics and tremor with intrinsic oscillatory property of instrumental mechanics. Bioengineering & Translational Medicine 8(2), e10432 (2023). https://doi.org/10.1002/btm2.10432

17. Park, Y., Park, H., Lee, C.J., Choi, S., Jo, S., Choi, H., Kim, Y., Shin, H., Llinas, R.R., Kim, D.: Ca(v)3.1 is a tremor rhythm pacemaker in the inferior olive. Proceedings of the National Academy of Sciences 107(23), 10731–10736 (2010). https://doi.org/10.1073/pnas.1002995107

18. Pereira, T.D., Tabris, N., Matsliah, A., Turner, D.M., Li, J., Ravindranath, S., Papadoyannis, E.S., Normand, E., Deutsch, D.S., Wang, Z.Y., McKenzie-Smith, G.C., Mitelut, C.C., Castro, M.D., D’Uva, J., Kislin, M., Sanes, D.H., Kocher, S.D., Wang, S.S.H., Falkner, A.L., Shaevitz, J.W., Murthy, M.: SLEAP: A deep learning system for multi-animal pose tracking. Nature Methods 19(4), 486–495 (2022). https://doi.org/10.1038/s41592-022-01426-1

19. Ravi, N., Gabeur, V., Hu, Y., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024). https://doi.org/10.48550/ arXiv.2408.00714

20. Zhang, R.X., Xu, J.T., Zhong, H.J., Cai, Y.L., Zhuang, Y.P., Xie, Y.T., He, X.X.: Gut microbiota from essential tremor patients aggravates tremors in mice. Front. Microbiol. 14, 1252795 (2023). https://doi.org/10.3389/fmicb.2023.1252795