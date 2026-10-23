# REINFORCING INTRA-OPERATIVE TUMOUR LOCALISATION WITH WEAKSUPERVISION FROM PRE-OPERATIVE APPEARANCES

Sidhart Sami<sup>1,3</sup>, Ali Sharjeel<sup>1,3</sup>, Sri Atluri<sup>1,2</sup>, Miao Zhao<sup>1,2</sup>, Caroline Roney<sup>1,2</sup>, Eirini Marouli<sup>2,4</sup>, Shaheer U. Saeed<sup>1,2,∗</sup>

<sup>1</sup>Centre for Bioengineering, School of Engineering and Materials Science, Queen Mary University of London, London, United Kingdom <sup>2</sup>Digital Environment Research Institute, Queen Mary University of London, London, United Kingdom <sup>3</sup>National University of Computer and Emerging Sciences (FAST-NUCES), Karachi, Pakistan <sup>4</sup>William Harvey Research Institute, Faculty of Medicine and Dentistry, Queen Mary University of London, London, United Kingdom \*Email: shaheer.saeed@qmul.ac.uk

## ABSTRACT

Limited tumour visibility on intra-operative images complicates surgical navigation, including liver hepatocellular carcinoma (HCC) targeting on computed tomography (CT) imaging. This makes labelled datasets difficult to curate for deep learning, and transferring pre-operative delineations through registration remains challenging due to intra-operative deformation and lack of paired pre- and intra-operative data. We propose to transfer tumour appearances learned from pre-operative magnetic resonance (MR) images to guide locali sation on intra-operative CT through weak supervision. A classifier is trained to classify tumour presence on pre-operative images. This is applied to windows on intra-operative images, and despite the domain gap, it assigns higher tumour-presence probabilities on average to tumour-containing windows. These weak cross-domain signals guide a reinforcement learning agent to localise tumours on intra-operative images. We compare direct transfer, few-shot adaptation and synthetic transfer (trained using unpaired MR and CT), which translates CT into synthetic MR before localisation. On an external HCC cohort, synthetic transfer outperformed other tested methods and CT-only baselines. Our framework enables cross-modality tumour localisation without paired MR-CT training images.

Index Terms— Cancer, Weak Supervision, Reinforcement

## 1. INTRODUCTION

Multi-modal imaging is central to modern clinical oncology, where complementary imaging modalities are acquired at different stages of care to maximise diagnostic accuracy and interventional precision [1]. A prominent example is the combined use of pre-operative magnetic resonance (MR) imag ing and intra-operative computed tomography (CT) imaging in oncology, where each modality serves a distinct clinical role [2]. In liver oncology, pre-operative MR imaging provides soft-tissue contrast that supports tumour detection and delineation [3]. Intra-operative CT is rapidly acquired for navigation, but tumour visibility can be limited [4, 5]. Lesions such as hepatocellular carcinoma (HCC) may therefore be difficult to distinguish from surrounding liver parenchyma [4].

Transferring tumour localisation information from preoperative to intra-operative imaging is therefore clinically important but technically challenging. Image registration offers a means of transferring pre-operative tumour delineations, but its application faces several challenges. Firstly, the limited availability of publicly accessible paired MR-CT images constrains the training of deep learning methods that require paired data [6, 7]. Secondly, intra-operative liver displacement, respiratory motion and patient repositioning introduce non-rigid deformations that cannot be adequately captured by rigid alignment [6]. Thirdly, some unpaired deep learningbased deformable registration approaches rely on corresponding anatomical labels to guide alignment [6]; obtaining reliable tumour labels in intra-operative CT is challenging for large datasets as lesion visibility is low, requiring expensive expert annotations [4]. Directly learning HCC localisation on intra-operative CT is similarly constrained by the substantial expert effort required to curate labelled training data. These challenges limit both registration-based label transfer and direct learning of localisation for lesions with low CT visibility, such as HCC. For example, recent approaches achieve lesion Dice scores of only 0.16-0.19 when MR and CT images are registered using tumour labels [4]. These low tumour Dice scores contrast with reported anatomical structure Dice scores of 0.72, highlighting that broader anatomical alignment does not necessarily ensure accurate tumour label transfer [4, 8].

Rather than transferring patient-specific tumour coordinates through registration, we propose to transfer learned tumour appearance information through weak supervision. A tumour presence classifier is first trained on MR images with binary labels indicating HCC tumour presence or absence. This classifier is then applied on local windows of intra-operative CT images, where it is able to generate the tumour-presence classification probability (from logits). Our central hypothesis is that, despite the modality gap, the classifier retains sufficient discriminatory information for tumour-containing windows to receive higher tumour-presence classification probabilities on average than windows without tumour. These classification probabilities are a ‘weak’ signal of cancer presence likelihood [9, 10, 11], with higher values indicating higher cancer presence likelihood within the window. Despite the domain gap to CT, the windows with cancer lead to higher probabilities on average compared to windows without cancer based on preliminary results from bootstrapped averages. The task of localising cancer in the CT image then becomes finding the areas where windows lead to higher classification probabilities on average. We formulate this localisation using reinforcement learning (RL), where an agent moves a window across the CT image, and receives a classification probability-derived reward signal guiding the search towards areas of higher cancer likelihood. Window-level classification probabilities are spatially aggregated using maximum probability retention to generate a tumour score map, thresholded to obtain a segmentation.

We investigate three strategies for deriving the weak supervision signal. Firstly, the MR-trained classifier is simply applied to the CT images, in a ‘zero-shot’ setting. Secondly, the MR-trained classifier is fine-tuned on a few CT examples, in a ‘few-shot’ setting. Lastly, ‘synthetic transfer’, uses a cycle generative adversarial network (GAN) to translate the CT image into synthetic MR before localisation, where the Cycle-GAN itself is trained using unpaired CT and MR images only. It is noteworthy that our proposed framework circumvents the need for paired data or tumour segmentation labels.

Our contributions are summarised: 1) a weakly supervised framework for transferring pre-operative tumour appearance information to intra-operative localisation, without requiring paired images or tumour masks in the zero-shot setting; 2) an RL-based tumour search strategy that uses the pre-operative image trained classifier’s classification probabilities as weak signals to guide tumour localisation on intra-operative images; 3) comparative evaluation of zero-shot, few-shot and synthetic image transfer strategies for localising HCC in intra-operative CT using a pre-operative MR-trained tumour classifier.

## 2. METHODS

## 2.1. Tumour-presence classifier training

We first train a classifier to recognise tumour presence in preoperative MR images. Let $x \in \mathcal { X } = \mathbb { R } ^ { a \times b \times c \times d }$ denote an MR image input, where a, b and c are its spatial dimensions and d is the number of channels, and let $y \in \mathcal { V } = [ 0 , 1 ]$ denote its tumour-presence label. Training labels are binary, with $y = 1$ indicating tumour presence and $y = 0$ indicating tumour absence. We define the classifier as:

![](images/e07b27cb9660be771400cc3f10216a5ea8a4b4547009f4d6e9759d523e4b3a28.jpg)  
Figure 1: Summary of the proposed method.

$$
f ( \cdot ; w ) : \mathcal { X } \to \mathcal { Y } ,\tag{1}
$$

where w denotes its trainable parameters and $f ( x ; w )$ is the predicted tumour-presence probability. The classifier is trained using binary cross-entropy loss $L ( x , y ; w ) = - y \log f ( x ; w ) -$ $( 1 - y ) \log ( 1 - f ( x ; w ) )$ , with optimisation following:

$$
\boldsymbol { w } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { w } } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) } \left[ L ( \boldsymbol { x } , \boldsymbol { y } ; \boldsymbol { w } ) \right] ,\tag{2}
$$

where the expectation is over labelled MR training inputs.

This objective requires only tumour-presence labels rather than voxel-wise tumour delineations.

## 2.2. Window-based weak supervision

To apply the same trained classifier $f ( \cdot ; w ^ { * } )$ to intra-operative CT, we extract local windows from each image, similar to [9, 12]. Let z denote the full CT image and $z _ { p }$ denote a fixedsize window centred at spatial position $p$ within the image. Each window is processed to match the input dimensions and preprocessing requirements of the classifier. Its tumourpresence classification probability is given by $f ( z _ { p } ; w ^ { * } )$ .

Although the modality gap may affect the absolute classification probabilities, we hypothesise that tumour-containing windows retain higher values on average than windows without tumour [12, 9]. We therefore use these classification probabilities as relative weak signals for localisation. Different approaches to transfer this weak signal to CT are evaluated, as outlined in Sec. 2.4.

Classification probabilities from the windows visited during localisation are aggregated into a voxel-wise tumour score map m using maximum probability retention. The localisation process is outlined in Sec. 2.3. Each visited window contributes its classification probability to every voxel within its spatial extent. For each voxel v, the map retains the highest probability among all visited windows containing that voxel::

$$
m ( v ) = \operatorname* { m a x } _ { t : v \in z _ { p _ { t } } } f ( z _ { p _ { t } } ; w ^ { * } ) ,\tag{3}
$$

where t indexes visited windows and $v \in z _ { p _ { t } }$ indicates that voxel v lies within the spatial extent of the window. The map m has the same spatial dimensions as the CT image z, with $m ( v ) \in [ 0 , 1 ]$ defined only at voxels covered by at least one visited window. The resulting map is converted into a tumour segmentation by thresholding, with the threshold τ selected through a grid search (see Tab. 2b).

## 2.3. Reinforcement learning for tumour localisation

We formulate window navigation as a sequential decisionmaking problem. At step t, the agent occupies position $p _ { t }$ and observes the current CT window:

$$
s _ { t } = \left( z _ { p _ { t } } \right)\tag{4}
$$

where $z _ { p _ { t } } \in \mathcal X$ is the classifier input and $p _ { t }$ specifies the three-dimensional window centre.

The action $a _ { t }$ translates the window using $p _ { t + 1 } = p _ { t } + a _ { t }$ by a fixed step size δ along one of the three image axes:

$$
a _ { t } \in \{ ( \pm \delta , 0 , 0 ) , ( 0 , \pm \delta , 0 ) , ( 0 , 0 , \pm \delta ) \}\tag{5}
$$

The agent receives a scalar reward $r _ { t } \in [ - 1 , 1 ]$ based on the change in tumour-presence classification probability:

$$
r _ { t } = f ( z _ { p _ { t + 1 } } ; w ^ { * } ) - f ( z _ { p _ { t } } ; w ^ { * } ) .\tag{6}
$$

Positive rewards encourage movement towards windows with higher tumour-presence classification probabilities. The classifier remains fixed during policy optimisation.

The navigation policy $\pi ( a _ { t } \mid s _ { t } ; \theta )$ , parameterised by $\theta ,$ assigns a probability to each valid action given the current state. Its parameters are trained to maximise the expected discounted return:

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t } \right] ,\tag{7}
$$

where $T$ is the episode length and $\gamma \in ( 0 , 1 )$ is the discount factor. Episodes begin at random valid locations and terminate after T steps. At inference, classification probabilities from the visited windows are aggregated using Equation 3.

## 2.4. Enhancing the classifier-derived weak signal

## 2.4.1. Zero-shot transfer

The MR-trained classifier $f ( \cdot ; w ^ { * } )$ remains fixed, and no labelled CT examples are used for classifier adaptation. This is applied directly to CT windows to obtain tumour-presence classification probabilities.

![](images/84ff52ed76c8e51873b79ab60a4e014b3f92274924e5baca040c40dce5c3f3d4.jpg)  
Figure 2: CycleGAN synthetic transfer (top: CT; bottom: MR) - average SSIM: 0.79 - ground truth segmentation in green

## 2.4.2. Few-shot adaptation

The MR-trained classifier is fine-tuned using a small set of CT inputs with binary tumour-presence labels. Fine-tuning uses the same binary cross-entropy objective, initialised with $w ^ { * }$ . The adapted classifier is then fixed and used to obtain classification probabilities for CT windows during RL-based localisation.

## 2.4.3. Synthetic image transfer

An image translation function $g : \mathcal { X } _ { \mathrm { C T } }  \mathcal { X } _ { \mathrm { M R } }$ is learned using an unpaired CycleGAN, without corresponding patient pairs or tumour labels. Training jointly learns g and a reverse mapping $h : \mathcal { X } _ { \mathrm { M R } } \to \mathcal { X } _ { \mathrm { C T } }$ . Adversarial losses encourage translated images to resemble the target modality, while cycle consistency encourages reconstruction of the inputs:

$$
L _ { \mathrm { c y c } } = \mathbb { E } _ { z } \left[ \| h ( g ( z ) ) - z \| _ { 1 } \right] + \mathbb { E } _ { x } \left[ \| g ( h ( x ) ) - x \| _ { 1 } \right] .\tag{8}
$$

The training objective combines the adversarial losses with $\lambda L _ { \mathrm { c y c } }$ , where λ controls the cycle-consistency contribution.

At inference, navigation is performed on $g ( z )$ , with the MR-trained classifier $f ( \cdot ; w )$ providing window-level classification probabilities for reward calculation and spatial aggregation. The resulting map is assigned to the corresponding CT coordinates (with anatomical correspondence retained owing to the spatial preservation enabled by the pixel-wise distance minimisation of the cycle-consistency loss [13] - as also noted in preliminary results presented in Fig. 2).

## 3. EXPERIMENTS

## 3.1. Data

MR data: 334 patients from LLD-MMR [14], ATLAS [15], CHAOS [16]. All volumes were resampled to 1.0 mm isotropic spacing, cropped to 128×128×128 voxels, clipped at the 0.5th and 99.5th intensity percentiles, and z-score normalised.

CT data: 313 patients from WAW-TACE [17], CHAOS-CT [16], and an internal dataset from [REC=14/LO/1264, IRAS=158321], including both pre- and intra-operative data. All volumes were windowed to -100 to +400 HU, resampled to 1.5 mm isotropic spacing, and z-score normalised.

Both datasets were partitioned into development and validation sets using a 70%/30% split at the patient-level. Evaluation is reported on TCIA HCC-TACE-Seg (105 patients) [18], where volumes had radiologist voxel-wise tumour annotations as ground truth. We focus on HCC which is conventionally challenging to identify on intra-operative CT images [4].

## 3.2. Networks and hyperparameters

The classifier used SE-ResNet3D [19] (learning rate $1 0 ^ { - 3 }$ batch size 8). The CycleGAN used 3D encoder–decoder generators and multi-scale patch discriminators [13, 20]. The RL agent used PPO [21] with a 3D convolutional network (learning rate $3 \times 1 0 ^ { - 4 }$ , discount factor 0.99). Hyperparameters provided in repository. On a single Nvidia Quadro GV100 GPU, the classifier converged in 8 hours, the CycleGAN in 16 hours, and the RL tumour localisation in 48 hours.

## 3.3. Experimental setup

We compared zero-shot, few-shot and synthetic transfer with recent methods [4, 6]. Few-shot adaptation fine-tuned the classifier on 40 development CT volumes; synthetic transfer trained the CycleGAN on the MR and CT development sets before localisation on synthetic MR. Registration methods used paired evaluation data specified in [4], rather than our CT-only evaluation cohort. To ablate MR transfer, we trained CT-only RL baselines on the CT development set using rewards from tumour labels (full supervision; FS) or a CT-only classifier (weak supervision; WS). We included an FS nn-UNet trained on segmentation-labelled training data, as implemented in [22]. Sensitivity analyses assessed window size and segmentation threshold τ . We report patient-level Dice ± standard deviation, with t-tests for statistical comparisons. Localisation success rate is the proportion of individual discontiguous tumours across patients achieving Dice $> 0 . 1 0$ , following [4].

## 4. RESULTS

## 4.1. Comparisons

Table 1 shows that synthetic transfer achieved the highest mean Dice score $( 0 . 3 1 \pm 0 . 1 0 ) $ and success rate (93%), with signifi cantly higher Dice than all other methods (all $p < 0 . 0 1 )$ . No statistically significant differences in Dice were found between few-shot transfer and CT-only nn-UNet FS baseline $( p > 0 . 1 1 )$ . Registration results are provided as contextual comparisons because they were obtained on a different evaluation cohort in the respective cited works. Qualitative results in Fig. 3.

![](images/70dd71e724c0c4fe9960e0f1ad38ffd3105c18eaead0c32c78a1b3981867420a.jpg)  
Figure 3: Qualitative results (top: images; bottom: ground truth segmentation in green and predicted in red - smoothed).

Table 1: Comparison with recent state-of-the-art.
<table><tr><td>Method</td><td>Mean ± SD</td><td>Success Rate</td></tr><tr><td>zero-shot transfer</td><td> $0 . 1 6 { \pm } 0 . 0 6$ </td><td>0.67</td></tr><tr><td>few-shot transfer</td><td>0.19±0.09</td><td>0.72</td></tr><tr><td>synthetic transfer</td><td>0.31±0.10</td><td>0.93</td></tr><tr><td>labelled deformable registration [4]</td><td>0.19±0.06</td><td>0.71</td></tr><tr><td>unlabelled deformable registration [4, 6]</td><td>0.13±0.09</td><td>0.56</td></tr><tr><td>CT-only FS baseline</td><td>0.07±0.05</td><td>0.49</td></tr><tr><td>CT-only WS baseline</td><td>0.08±0.06</td><td>0.58</td></tr><tr><td>CT-only nn-UNet FS baseline [22]</td><td>0.24±0.12</td><td>0.81</td></tr></table>

## 4.2. Sensitivity analysis

We evaluated 16 candidate values per hyperparameter on the validation splits and report test results for the five bestperforming settings. $\mathrm { ~ A ~ 3 2 ~ } \times 3 2 \times 1 6$ voxel window achieved the highest mean Dice $( 0 . 3 1 \pm 0 . 1 0 )$ , closely followed by $3 2 \times 3 2 \times 3 2$ (Table 2a). Mean Dice increased with threshold τ from 0.90 to 0.98, then decreased slightly at 0.99 (Table 2b).

Table 2: Sensitivity analysis for hyperparameters.
<table><tr><td rowspan=1 colspan=2>(a) Window size.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(b) Threshold τ.</td></tr><tr><td rowspan=1 colspan=1>Window size</td><td rowspan=1 colspan=1> $\overline { { \mathrm { M e a n } \pm \mathrm { S D } } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>Mean ± SD</td></tr><tr><td rowspan=1 colspan=1> $1 6 \times 1 6 \times 8$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 2 4 \pm 0 . 0 9 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1> $\overline { { 0 . 2 0 { \pm } 0 . 0 9 } }$ </td></tr><tr><td rowspan=2 colspan=1> $3 2 \times 3 2 \times 1 6$  $3 2 \times 3 2 \times 3 2$ </td><td rowspan=1 colspan=1> ${ \bf 0 . 3 1 } { \bf \pm 0 . 1 0 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1> $0 . 2 6 { \pm } 0 . 1 1$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 3 0 { \pm } 0 . 0 7$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.95</td><td rowspan=2 colspan=1> $0 . 2 7 { \pm } 0 . 1 0$  ${ \bf 0 . 3 1 } { \pm } { \bf 0 . 1 0 }$ </td></tr><tr><td rowspan=2 colspan=1> $4 8 \times 4 8 \times 2 4$  $6 4 \times 6 4 \times 3 2$ </td><td rowspan=1 colspan=1> $0 . 2 8 { \pm } 0 . 1 1$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.98</td></tr><tr><td rowspan=1 colspan=1> $0 . 2 8 { \pm } 0 . 0 8$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1> $0 . 2 9 { \pm } 0 . 0 8$ </td></tr></table>

## 5. CONCLUSION

Synthetic transfer significantly outperformed other evaluated methods, suggesting that reducing the modality gap strengthens MR-trained classifier guidance. Comparatively poorer CT-only results highlight the value of MR-derived information. Despite a 93% localisation success rate, the Dice score of 0.31 indicates that HCC boundary delineation remains challenging and could be explored using alternative supervision.

We developed a weakly supervised framework that uses an MR-trained classifier to guide liver tumour localisation on CT without paired MR–CT training images. This demonstrates the potential of transferring learned tumour appearances across modalities through weak supervision.

## Open-source implementation

The code has been open-sourced at: github.com/SidhartSami/cross-modality-weak-supervision.

## Acknowledgements

This work was supported by the International Alliance for Cancer Early Detection, an alliance between Cancer Research UK [EDDAPA-2024/100014], the University of Cambridge, the OHSU Knight Cancer Institute, the German Cancer Research Center (DKFZ), the University College London, the University of Manchester, and the Dana Farber Cancer Institute.

## Compliance with Ethical Standards

This research study was conducted retrospectively using human subject data made available in open access. The internal dataset from REC=14/LO/1264, IRAS=158321 complies with the Helsinki Declaration of 1975, as revised in 2000, and was obtained with informed patient consent.

## References

[1] Kevin M. Boehm et al., “Harnessing multimodal data integration to advance precision oncology,” Nature Reviews Cancer, 2022.

[2] Alejandro Forner et al., “Hepatocellular carcinoma,” The Lancet, 2018.

[3] Victoria Chernyak et al., “Liver imaging reporting and data system (li-rads) version 2018: imaging of hepatocellular carcinoma in at-risk patients,” Radiology, 2018.

[4] Budhaditya Mukhopadhyay et al., “Towards segmenting the invisible: An end-to-end registration and segmentation framework for weakly supervised tumour analysis,” in AIBIO, ECAI 2025, 2026, Communications in Computer and Information Science.

[5] Lewis R. Roberts et al., “Imaging for the diagnosis of hepatocellular carcinoma: A systematic review and metaanalysis,” Hepatology, 2018.

[6] Yunguan Fu et al., “Deepreg: a deep learning toolkit for medical image registration,” JOSS, 2020.

[7] Xiaohuan Cao et al., “Region-adaptive deformable registration of CT/MRI pelvic images via learning-based image synthesis,” IEEE TIP, 2018.

[8] James Neuberger et al., “Guidelines on the use of liver biopsy in clinical practice from the British Society of Gastroenterology, the Royal College of Radiologists and the Royal College of Pathologists,” Gut, 2020.

[9] Shaheer U. Saeed et al., “Competing for pixels: a selfplay algorithm for weakly-supervised semantic segmentation,” IEEE TPAMI, 2024.

[10] Catalina Tan et al., “Spars: Self-play adversarial reinforcement learning for segmentation of liver tumours,” in MIUA, 2025.

[11] Shaheer U. Saeed et al., “Reasoning in machine vision by learning fast and slow thinking,” Nature Comms, 2026.

[12] Martynas Pocius et al., “Weakly supervised localisation of prostate cancer using reinforcement learning for biparametric mr images,” in 2024 IEEE ISBI, 2024.

[13] Jun-Yan Zhu et al., “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in ICCV, 2017.

[14] Meng Lou et al., “Sdr-former: A siamese dual-resolution transformer for liver lesion classification using 3d multiphase imaging,” Neural Networks, 2025.

[15] Felix Quinton et al., “A tumour and liver automatic seg-´ mentation (atlas) dataset on contrast-enhanced magnetic resonance imaging for hepatocellular carcinoma,” Data, 2023.

[16] A. Emre Kavur et al., “Chaos - combined (ct-mr) healthy abdominal organ segmentation challenge,” Medical Image Analysis, 2021.

[17] Krzysztof Bartnik et al., “Waw-tace: A hepatocellular carcinoma multiphase ct dataset with segmentations, radiomics features, and clinical data,” Radiology: Artificial Intelligence, 2024.

[18] TCIA Clinical Research Group, “Radiogenomic and segmentation dataset of hepatocellular carcinoma treated with tace (hcc-tace-seg),” 2023.

[19] Jie Hu et al., “Squeeze-and-excitation networks,” in IEEE CVPR, 2018.

[20] Yuta Hiasa et al., “Cross-modality image synthesis from unpaired data using CycleGAN: Effects of gradient consistency loss and training data size,” in SASHIMI - MIC-CAI Workshop, 2018.

[21] John Schulman et al., “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[22] Ilaria Manghi et al., “Deep learning for automatic segmentation of hepatocellular carcinoma in contrast enhanced ct scans,” Sci Rep, 2025.