# Multi-Tool Image Editing Attribution in Facial Forgery

Sheng Liu<sup>1,2</sup>, Qiang Sheng<sup>1,2</sup>, Danding Wang<sup>1,2</sup>, Yu Li<sup>1</sup>, Chenming Zhou<sup>1,2</sup>, and Juan Cao<sup>1,2</sup>

<sup>1</sup>Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China

<sup>2</sup>University of Chinese Academy of Sciences, Beijing, China

As generative AI tools become increasingly powerful and easy to use, people can easily edit portrait images with a prompt, necessitating the task of image editing attribution, which predicts the involved editing tools from the given image. Existing attribution methods hold the single-tool assumption and can only attribute a specific editing tool, but struggle to handle the more complex and increasingly common multi-tool editing scenarios, where artifacts left by diferent editing tools are composite and overlapped. To address this gap, we explore Multi-Tool Image Editing Attribution (MIEA), which aims to identify multiple editing tools involved in a multi-tool edited facial image. To simulate the real-life editing operations on facial images, we then construct a new dataset, MultiEdit, which contains 500k+ edited facial images and covers six types of editing tools that support face swapping (Deepfake) and various facial enhancements. Inspired by the findings from data analysis, we design DPEC, a multi-tool attribution method that can capture distinguishable, locality-aware editing tool traces from both spatial and frequency domains with the support of an error-based curriculum learning strategy. Experiments show DPEC outperforms nine methods for facial images edited in at most five steps.

Code, Data & Extended Materials: https://github.com/ICTMCG/MIEA Venue: ACM Multimedia 2026

Date: September 2, 2026

![](images/a4c9a507ebb7bcbdeaa99ee13eff3e816bb251c8a67f2ecf47456e9a67ec74f5.jpg)

## 1. Introduction

With the proliferation of powerful generative AI techniques like generative adversarial networks (Goodfellow et al., 2014, Yang and Lim, 2019) and difusion models (Ho et al., 2020), the creation and manipulation of digital images has become increasingly accessible and afordable for ordinary people. Using convenient AI image tools and software, users can produce vivid images that are hard for humans to distinguish with only a prompt (Cooke et al., 2024), a phenomenon evidenced by its popularity on social media: 71% of social media images are now AI-generated or AI-edited <sup>1</sup>.

Meanwhile, such widespread adoption of AI image tools has led to a growing risk of misuse. Fake facial images produced by AI have been used for fake news fabrication (Dufour et al., 2024, Liu et al., 2025), celebrity reputation damage (Stokel-Walker, 2023), and academic fraud (Davydiuk et al., 2025), posing severe challenges to online information ecosystems. Although researchers have made remarkable progress in AI-generated content detection (Wang et al., 2025, Yang et al., 2019) and attribution (Yang et al., 2022, 2023), we argue that the recent paradigm shift from single-tool use to combinational multi-tool use in AI image editing presents a new challenge that existing solutions can hardly address. While some AI editing tools claim to accomplish complex edits in a single step, the final edited images circulated online are often created through multiple editing operations involving various tools (Yan et al., 2025). On one hand, even general-purpose image editing tools are rarely fully capable in all functions, encouraging users to apply diferent tools for incremental image adjustments to achieve desired efects (Joseph et al., 2024, Zhou et al., 2025). On the other hand, an image may also be edited by multiple people independently, using diverse tools for diferent purposes. Consequently, multi-tool editing results in a mixture of diferent AI tool artifacts within a single image, violating the basic assumption that only one type of tool trace exists on an image. This renders existing methods inefective in practice, as they can only identify the use of one tool, and their single-tool detection sensitivity degrades significantly when other editing operations are applied afterward (we will demonstrate this empirically in Sec. 3). If we perform forensic analysis of maliciously edited images using single-tool attribution methods, we cannot recover the full set of involved editing tools and may only detect benign edits while missing the actual malicious operations, allowing malicious actors to evade accountability. Therefore, to achieve accurate image forensics, it is critical to reconstruct the entire editing chain. By identifying every tool employed, we can demystify the black-box editing workflow, increase the transparency of the editing process, and reduce the room for hidden malicious manipulation.

![](images/b134d112866ca75c9b5ad0ee14e638709c8691f4be04f4ac166089eb26b9bcec.jpg)  
Figure 1: Multi-Tool Image Editing and Attribution. A polished portrait often undergoes multiple editing operations from diferent tools, and the whole editing process remains unseen and undisclosed. Existing single-tool attribution methods focus on extracting specific features produced by a single tool, which, to their best, only produces an incomplete prediction. In this paper, we explore how to build a multi-tool attributor that can identify all involved tools in the image editing process from a given edited result image.

To address these challenges, we carry out this work across three core dimensions: task formulation, dataset construction, and method design. We first formulate a new task, Multi-Tool Image Editing Attribution (MIEA) that aims to identify multiple editing tools involved in a multi-tool edited image (Fig. 1). We then construct a large-scale facial editing dataset, MultiEdit, which contains over 500k edited facial images and covers six representative types of editing tools. Based on detailed analysis of tool-specific patterns, we propose Dual-domain Patchwise Enhancement with Curriculum learning (DPEC), a multi-tool attribution method that captures distinguishable, locality-aware editing tool traces from both spatial and frequency domains, supported by an error-based curriculum learning strategy. Moreover, DPEC is designed to recover all tools in the image editing chain, efectively preventing early editing traces from going undetected due to being overwritten or degraded by subsequent editing operations. Experiments show that DPEC outperforms existing methods in MIEA and achieves competitive performance in the conventional detection task. In summary, our main contributions are:

• New task: We propose a challenging task, Multi-Tool Image Editing Attribution, that attributes a multitool-edited image to the multiple tools involved in its whole editing process, instead of only detecting the final editing tool.

• New method: We propose a multi-tool attribution method DPEC that can extract distinguishable features of editing tools from multi-edited images via dual-domain patchwise feature enhancement and error-based curriculum learning. By uncovering the editing chain despite trace-masking from sequential edits, DPEC enables more transparent and precise image forensics.

• New dataset: We construct MultiEdit, a large-scale dataset covering 500k+ facial images and 6 common editing tools, on which we experimentally verify the efectiveness of DPEC.

## 2. Related Works

Face Forgery Generation & Detection The escalating realism of synthetic media has sparked an arms race between face forgery generation and detection techniques. Traditional editing tools like Photoshop and GIMP, which rely on manual pixel-level operations (e.g., splicing, liquify), often leave detectable inconsistencies (Wang et al., 2019, Zhang et al., 2023). With rapid advancement of deep generative models, particularly Autoencoders (Bounareli et al., 2025), Generative Adversarial Networks (GANs) (Ling et al., 2021, Jiang et al., 2025, Yang and Lim, 2019), and more recently, Difusion Models (Huang et al., 2023, Wei et al., 2025) have significantly elevated the visual fidelity of various manipulation techniques, including deepfake, face reenactment, and attribute manipulation. Accordingly, numerous deep learning-based approaches have emerged as the mainstream for manipulation detection. Specifically, Xception (Chollet, 2017), TSA-Net (Yan et al., 2022) and constrained convolutional layers (Bayar and Stamm, 2018) leverage both spatial and frequency-domain features to amplify subtle manipulation traces, while F<sup>3</sup>-Net (Qian et al., 2020) employs frequency-component partitioning for detection purposes. Recent works have further focused on cross-attention-based detectors (e.g., NPR (Tan et al., 2024) and FatFormer (Liu et al., 2024)) to cope with diverse forgery patterns in face manipulation. These eforts are supported by benchmark datasets such as ImgEdit-Bench (Ye et al., 2025), yet standardized evaluation protocols for multi-tool attribution still remain underdeveloped.

Model Attribution The core task of model attribution is identifying the specific tool or model behind an edit, primarily relying on proactive watermarks (Zhang et al., 2018a, Singh and Singh, 2024) and detecting unique model fingerprints (Yang et al., 2022). Proactive embedded fingerprints (invisible watermarks) are alternatively engineered during model training/inference for more direct provenance tracing (Zhang et al., 2018a, Singh and Singh, 2024), while model fingerprint extraction uses deep learning classifiers (Ganguly et al., 2022, Yang et al., 2023, Liu et al., 2024) to learn unique artifacts embedded by models like GANs.

Table 1: Selected tools and their editing types. Empty cell means type not supported or with unsatisfactory efects.
<table><tr><td rowspan="2">Tool</td><td colspan="3">Editing Type</td></tr><tr><td>Deepfake</td><td>Shape Beauty Skin Beauty</td><td>Makeup</td></tr><tr><td>FaceFusion</td><td>√</td><td></td><td></td></tr><tr><td>DiffSwap</td><td>√</td><td></td><td></td></tr><tr><td>SHMT</td><td></td><td>√</td><td>√</td></tr><tr><td>CSD-MT</td><td></td><td>√</td><td>√</td></tr><tr><td>OpenCV</td><td>√</td><td>√</td><td>√</td></tr><tr><td>FLUX.1</td><td>√</td><td>√</td><td>√</td></tr></table>

Typical contrastive learning based attribution methods, such as RepMix (Bui et al., 2022) and DNA-Det (Yang et al., 2022), usually assume that an image is fingerprinted by only one model, while multi-tool editing may mask and overwrite earlier traces, thereby rendering these methods inefective. For example, a filter adjuster B may destroy the traces left by the emotion-editing model A. On the subject of multi-tool editing, Fakechain (Heo and Woo, 2025) discovered the impacts of mixed traces on detection methods. MSA (Tan et al., 2025) aims to identify involved editing tools between two sequentially edited images, a special case in MIEA. Moreover, its requirement for paired sequentially edited images is too dificult to meet, as in-the-wild images hardly carry editing history records, which limits the task’s practicality, while our proposed task and method aim to identify multiple editing tools used based on the final edited image only.

## 3. Dataset Construction

## 3.1. Selected Editing Tools

To investigate the traces left by diferent image editing tools when modifying a portrait image in a sequential manner, we selected six image editing tools based on five distinct model/algorithm frameworks: OpenCV<sup>2</sup>, FaceFusion<sup>3</sup>, DifSwap (Zhao et al., 2023), CSD-MT (Sun et al., 2024b), SHMT (Sun et al., 2024a), and FLUX.1-Kontext-[dev] (Labs et al., 2025).

Deepfake Tools FaceFusion is an industry-leading face manipulation platform that integrates multiple advanced model architectures for face swapping, lip-syncing and so on. DifSwap, on the other hand, is a difusion-based framework designed for high-fidelity and highly controllable face swapping following a conditional inpainting methodology.

Face Beautify Tools OpenCV is selected as the representative non-generative method, and we implemented three types of face beauty algorithms based on its oficial tutorial<sup>4</sup>. CSD-MT (Sun et al., 2024b) is an unsupervised makeup transfer method that decouples facial content and makeup style via frequency decomposition. SHMT (Sun et al., 2024a) builds upon LDM (Rombach et al., 2022), proposing a self-supervised hierarchical makeup transfer method and enabling precise, natural makeup transfer while preserving the source facial content. FLUX.1-Kontext-[dev] (Labs et al., 2025) focuses exclusively on editing tasks. The model enables iterative text-guided editing, excels at character preservation across diverse scenes, and allows both precise local and global edits.

Considering the typical designed use scenarios of these tools and their practical editing efects, we select the editing types where these tools excel in the editing process (shown in Table 1). For each editing type, we define a set of editing targets covering frequent daily editing operations for better simulation (shown in Table 2).

Table 2: Involved editing types and their editing targets
<table><tr><td>Editing Type</td><td>Editing Target</td></tr><tr><td>Deepfake</td><td>Identity change</td></tr><tr><td>Skin</td><td>Acne/wrinkle removal Skin whitening Skin smoothing</td></tr><tr><td>Shape</td><td>Face slim Eyes/nose/mouth size adjustment Mouth length adjustment Lips thickness adjustment</td></tr><tr><td>Makeup</td><td>Blush Eyeshadow Lip color adjustment Full-face makeup</td></tr></table>

## 3.2. Multi-Tool Editing Workflow

To cover as many combinations of editing tools and types as possible in real-world scenarios, we considered editing types, tools, targets, and sequential positions of tools.

Algorithm 1 shows a general editing process of k beautification steps. Specifically, in line with the practical application logic of deepfake technology—where it does not perform face-swapping after facial enhancement is completed—we restrict deepfake technology to only appear in the first step of the multistep editing process. During the actual generation process, we set k = 4 and adopt the method of random sampling without replacement to select the editing tool t used in each step. To control the data quality and evaluate the edited images, we use the SSIM (Wang et al., 2004) metric to measure the similarity between the two images before and after a single editing operation, and employ Qwen3-VL (Qwen Team, 2025) to determine whether the image has achieved the editing goal. An editing process is considered successful if and only if the SSIM score falls between 0.4 and 0.98 and Qwen3-VL judges that the target editing goal has been achieved in all steps.

## 3.3. Dataset Overview

For authentic real facial images, we selected 200 real human portrait images from the DEFACTO-Face subset (Mahfoudi et al., 2019) as the source set S. For each image in S, we enumerate all possible permutations (i.e., unique tool-use sequences) specified in Algorithm 1, and then filter out unsuccessful editing cases. Ultimately, each image is processed with 648 valid plans, yielding a total of 510k edited images in MultiEdit. The statistical details of MultiEdit are summarized in Table 3.

## 3.4. Data Analysis

Since diferent editing tools may leave diferent traces in their editing areas, a multi-tool editing process can lead to more complex trace mixtures and overwriting issues. To guide the following method design, we perform analysis on MultiEdit samples to discover the editing impacts in single-tool detectability (Q1), domain diversity (Q2), regional efect (Q3), and inter-tool interaction (Q4).

Algorithm 1: Multi-tool face editing process   
Input:   
Portrait set $S = \left\{ I _ { 1 } , \cdots , I _ { n } \right\}$   
A portrait image $I \in S$   
Beautify steps parameter $k$   
Deepfake Tools $D = \left\{ \begin{array} { l l } { \begin{array} { r l r } \end{array} } \end{array} \right.$ {FaceFusion, DifSwap}   
Beauty Tools   
$M = \{ { \mathrm { O p e n C V } } ,$ CSD-MT, SHMT, FLUX}   
Beauty Types $E = \cdot$ {Shape, Skin, Makeup}   
Tool Target $\mathrm { T } { = } \{ T _ { m } = \left\{ { \bar { t } } _ { m , 1 } , \cdots , t _ { m , n } \right\} \}$ m∈M   
Result: Edited Images $I _ { E } = \left\{ I _ { 1 } , I _ { 2 } , \cdots , I _ { k } \right\}$   
1 $r _ { \mathrm { f a c e } }$ ← rand in $S \cup$ {None};   
2 if $\underline { r } _ { \mathrm { f a c e } }$ ≠ None then   
3 $I _ { \mathrm { s w a p } }$ ← randomly select from $S \setminus \{ I \} ;$   
4 $I _ { 1 } \gets$ replace face of I with face from $I _ { \mathrm { s w a p } } ;$   
5 else   
6 $I _ { 1 }  I ;$   
7 end   
8 $I _ { E }  [ I _ { 1 } ] ;$   
9 for $i \gets 2$ to k do   
10 $( \mathbf { m } , \mathbf { e } , \mathbf { t } ) \gets$ randomly select from $\left( M , E , T _ { m } \right)$   
11 $I _ { i } \gets$ apply $\left( \mathbf { m } , \mathbf { e } , \mathbf { t } \right)$ on $I _ { i - 1 } ;$   
12 Add $I _ { i }$ to $I _ { E } ;$   
13 end   
14 return $I _ { E } ;$

Q1: Does multi-tool editing make tools harder to detect? Previous detection methods (Qian et al., 2020, Chollet, 2017) and attribution methods (Bui et al., 2022, Yang et al., 2022) show good performance in detecting tools in single-tool edited images. To explore whether multi-tool editing makes it harder for these methods to detect specific tools, we conducted experiments where the training set comprised real images and images subjected to single-step editing solely by a specific tool, and the model was trained to perform binary classification to distinguish real from fake images. For the test set, images were produced by applying subsequent multi-step editing with other tools on the basis of the aforementioned single-step edited images, and the model was then adopted to execute the same real-versus-fake binary classification on these test images. We calculate the recall of each face beauty tool and report the mean recall across editing tools at each number of subsequent editing steps. As shown in Figure 2, as the number of subsequent editing steps increases, the recall of both detection and attribution methods shows a significant decline, indicating that the multi-tool editing indeed makes it harder to detect specific tools.

Q2: What editing patterns do editing tools exhibit in spatial and frequency domains? To investigate the patterns exhibited by diferent editing tools, we analyze the spatial and frequency domain similarities between the images before and after editing by specific tools on diferent facial regions. Let the flattened original image patch be denoted as x and the corresponding edited image patch as y. For spatial domain, we compute the L2 distance by pixel-wise subtraction between the edited and the original image patches:

Table 3: Statistics of MultiEdit. A permutation (Perm) is a unique tool-use editing sequence. We removed implausible permutations to better simulate real-life scenarios.
<table><tr><td rowspan=1 colspan=2>Number of Samples</td><td rowspan=1 colspan=1>517,540</td></tr><tr><td rowspan=3 colspan=1>Number ofEditing Steps</td><td rowspan=3 colspan=1>1 (w. 5 Perms)2 (w. 17 Perms)3 (w. 36 Perms)4 (w. 48 Perms)5 (w. 24 Perms)</td><td rowspan=1 colspan=1>43,168</td></tr><tr><td rowspan=1 colspan=1>129,385</td></tr><tr><td rowspan=1 colspan=1>129,385129,38586,127</td></tr><tr><td rowspan=1 colspan=1>Number of SamplesPer Editing Tool</td><td rowspan=1 colspan=1>FaceFusionDiffSwap (Zhao et al., 2023)CSD-MT (Sun et al., 2024b)SHMT (Sun et al., 2024a)FLUX.1 (Labs et al., 2025)OpenCV</td><td rowspan=1 colspan=1>172,684172,184323,551323,479323,388323,432</td></tr><tr><td rowspan=3 colspan=1>Number of SamplesPer Editing Type</td><td rowspan=3 colspan=1>Shape BeautyMakeupSkin BeautyDeepfake</td><td rowspan=1 colspan=1>86,267345,019</td></tr><tr><td rowspan=1 colspan=1>345,024</td></tr><tr><td rowspan=1 colspan=1>344,868</td></tr></table>

![](images/3833e8459255ca5b2bff32b47f78beb435bb29adf2d467cd854bfe9658844d5c.jpg)  
Figure 2: Recalls of two methods $( \mathbf { F } ^ { 3 }$ -Net and DNA-Det) on detecting single editing tool. The number of subsequent editing steps indicates how many editing operations were applied after the target tool was used.

$$
D _ { s p a t i a l } ( \mathbf { x } , \mathbf { y } ) = \| \mathbf { x } - \mathbf { y } \| _ { 2 } ,\tag{1}
$$

For frequency-domain similarity, the 2D Discrete Cosine Transform (DCT) transforms patched image data into orthogonal frequency components:

$$
\begin{array} { r } { \mathbf { X } = 2 \mathrm { D } { \cdot } \mathrm { D C T } ( \mathbf { x } ) , \quad \mathbf { Y } = 2 \mathrm { D } { \cdot } \mathrm { D C T } ( \mathbf { y } ) . } \end{array}\tag{2}
$$

Therefore, we calculate cosine similarity without the DC component to obtain the frequency-domain distance of two regions. Let $\mathbf { \boldsymbol { x } } _ { A C }$ and $\mathbf { Y } _ { A C }$ denote the frequency vectors with the DC component removed. The cosine similarity is calculated using the dot product:

$$
\mathrm { S i m } _ { f r e q } ( \mathbf { X } _ { A C } , \mathbf { Y } _ { A C } ) = \frac { \mathbf { X } _ { A C } \cdot \mathbf { Y } _ { A C } } { \| \mathbf { X } _ { A C } \| _ { 2 } \| \mathbf { Y } _ { A C } \| _ { 2 } } .\tag{3}
$$

Finally, the frequency-domain distance between the two regions is obtained as:

$$
D _ { f r e q } ( \mathbf { x } , \mathbf { y } ) = 1 - \mathrm { S i m } _ { f r e q } ( \mathbf { X } _ { A C } , \mathbf { Y } _ { A C } ) .\tag{4}
$$

Accordingly, we concatenate $D _ { s p a t i a l }$ and $D _ { f r e q }$ from diferent facial regions and plot the ridge plot to visualize both distributions for diferent editing tools. As illustrated in Figure 3, the spatial domain distributions for CSD-MT, SHMT, and DF overlap considerably, yet their frequency domain profiles are clearly distinct. Similarly, OpenCV and FLUX display comparable behaviors in the frequency domain while exhibiting highly divergent distributions in the spatial domain. These findings demonstrate that editing tools sharing similar statistical patterns in one domain can still be diferentiated in the other. Consequently, the spatial and frequency domains ofer mutually complementary information for the MIEA task.

![](images/bc7a9d7c8b79bd69f42d647294d7ad24a99766ce161453862d0e647e2255c907.jpg)  
Figure 3: Ridge graph of the spatial and frequency domains among editing tools. We analyze the spatial and frequency domain diferences between the images before and after editing by specific tools on diferent facial regions. In the spatial domain, CSD-MT, SHMT, and DeepFake show similar distributions, while totally difering in the frequency domain. Likewise, the distributions of OpenCV and FLUX are similar in the frequency domain while varying in the spatial domain.

Q3: Does a single editing tool exhibit the same editing pattern among regions? For MIEA, identifying the editing tools involved in the editing process is the core, and the key to this problem is to capture the distinguishable traces left by diferent editing tools. To explore the regional characteristics of diferent editing tools, we conduct analysis on the frequency domain based on distribution distance to discover the consistency of editing intensity across facial regions. Inspired by Qian et al. (2020) and Cox et al. (1997), we split the DCT coeficients into three bands (low, mid, high), select low band and mid band for further analysis by Manhattan distance, and conduct Kruskal–Wallis H test (K-W test) to determine whether there are statistically significantly diferent distributions of $D _ { f r e q }$ between edited patches and original patches among edited facial regions for each tool. Results are shown in Figure 4. The K-W test shows that in the low and mid-frequency bands, all tools exhibit statistically significant inter-region diferences, indicating that one tool can also alter diferent facial regions to varying degrees.

Q4: What influences does the multi-tool editing process bring? An intuitive thought is that as the number of editing steps increases, the edited image gradually deviates from the original image, leading to a continuous increase in human perceptual loss. From this perspective, we apply the Learned Perceptual Image Patch

![](images/69021062e48a3efb42aaec4e4e11ef85ddbb9bdba0a409f9bd8f31f5c94ec368.jpg)

![](images/4167562a097c922f59ddf6fda802182433be83a221dced876da75b9777019fc5.jpg)

![](images/0427e4d3e500239c18dc97fd4dfe3791814962de19c9087c11b79acea38a8c4d.jpg)

![](images/63f24280971642e8e3943e9487526599da657aa383e24ddc4e452c4d17aa664b.jpg)

![](images/4a2cd98f2dbc7130a33a7268634061f246406381926db82c31c847e0109cdc37.jpg)  
Figure 4: Grouped box plots of low and mid band of $D _ { f r e q }$ distributions between edited and original patches across four facial regions for each editing tool. ∗ indicates significance levels from the Kruskal-Wallis H test with the hypothesis that distributions among regions are not consistent.

Similarity (LPIPS) (Zhang et al., 2018b) metric to measure the human perceptual loss between the original image and the edited images with diferent numbers of editing steps. LPIPS captures high-level semantic and perceptual features to align its evaluations more closely with human visual perception, with a lower score indicating better consistency of visual features before and after editing. In Figure 5 a), from step 2 to step 5, the LPIPS score does not increase continuously, which goes against the intuition above. This counterintuitive phenomenon may be because the subsequent round of editing overwrites the relevant information of the previous round, causing the impact of the previous editing to be gradually diluted and overwritten.

![](images/0633f6b95e7b2a767ff66992e87cbe30fdfac7f1e8d54ea2c08be4b46e5bab43.jpg)  
Figure 5: Box Plots of LPIPS Score. a) LPIPS Score between the original image and edited images with diferent numbers of editing steps; b) LPIPS Score between the images before and after editing by specific tools. DF=Deepfake Tools. Note that for results produced by deepfake tools, we calculate the LPIPS score between the target face image (to be swapped) and the generated result.

## 4. Proposed Method: DPEC

## 4.1. Problem Formulation & Method Overview

We formulate multi-tool image editing attribution as a multi-label classification task: Given an edited image $I _ { s } ,$ an attribution model $\mathcal { M }$ predicts which tool(s) in an n-tool set $\tau$ were applied to $I _ { s }$ editing. ℳ outputs an n-dimensional vector for $_ { 5 } ^ { I _ { s } , }$ each entry of which corresponds to a tool in $\tau$ and denotes the probability of its being used for $I _ { s }$ editing.<sup>5</sup>

![](images/cdf87a845cac298544b2130000471f5fa3f5fa7c00384ee6425ed3d7ad07dc34.jpg)  
Figure 6: Overview of DPEC. DPEC consists of a) Patchwise Feature Enhancement (PFE) module and b) Error-based Curriculum Learning (ECL) module. PFE applies Slide Window DCT and a patchwise cross-attention to capture distinguishable, locality-aware features in both spatial and frequency domains. ECL adjusts data sampling weights of each subset of diferent dificulties during training based on error rate feedback, enabling the model to learn progressively from simple to hard samples. Based on the fused feature, a multi-label classifier finally predicts which tools in the candidates manipulated the given edited image I<sub>S</sub>.

As presented in Figure 6, our proposed DPEC is a fine-grained attribution framework to figure out the involvement of editing tools in a facial image, consisting of the Patchwise Feature Enhancement (PFE) module and the Error-based Curriculum Learning (ECL) strategy. The PFE module is designed to extract distinguishable, locality-aware features generated by diferent editing tools patch by patch from both spatial and frequency domains, while the ECL strategy enables the model to progressively learn to capture individual characteristics of diferent tools, starting from easy samples undergoing fewer editing steps to hard samples that underwent more editing steps, by adaptively adjusting the proportion of easy and hard samples based on error feedback.

## 4.2. Patchwise Feature Enhancement

Based on the findings from Q2 and Q3 in Sec. 3.4, the PFE module is designed to select distinctive patches with the most discriminative features for diferent editing tools, via patch-level feature co-attention across both spatial and frequency domains. This mechanism enables position-aware patch selection, feature extraction, and dual-domain feature enhancement, all tailored to the characteristics of each editing tool, to efectively capture both intra- and inter-patch features. Formally, given an input image $I _ { s } \in \mathbb { R } ^ { H \times W \times C }$ , we apply Slide-Window DCT (SWDCT) (Qian et al., 2020) with window size $h \times w$ and step d to obtain $I _ { f }$ :

$$
I _ { f } = \left\{ I _ { f _ { 1 } } , I _ { f _ { 2 } } , \cdots , I _ { f _ { N } } \right\} \underbrace { S W D C T } _ { I _ { s } , } I _ { s } ,\tag{5}
$$

where $I _ { f _ { i } } \in \mathbb { R } ^ { h \times w \times C }$ denotes the DCT coeficients of the i-th patch, and $\begin{array} { r } { N = \left( \frac { H - h } { d } + 1 \right) \times \left( \frac { W - w } { d } + 1 \right) } \end{array}$ . Then, we extract spatial features $F _ { s }$ and frequency features $F _ { f }$ from $I _ { s }$ and $I _ { f }$ using two separate backbones. Considering the diferent editing intensities across facial regions, we add learnable position encoding to flattened $F _ { s }$ and $F _ { f }$ to enhance the position awareness of features, denoted as $F _ { s p }$ and $F _ { f p . }$ , respectively. The cross attention is computed in two directions: spatial features attending to frequency features and vice versa. Thus, the output features of PFE $F _ { s } ^ { \prime }$ and $F _ { f } ^ { \prime }$ are updated as:

$$
\mathrm { A t t n } { \left( Q , K , V \right) } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } \right) V ,\tag{6}
$$

$$
F _ { s } ^ { I } = F _ { s } + \mathrm { R e s h a p e } \left( \mathrm { A t t n } ( Q _ { F _ { s p } } , K _ { F _ { f p } } , V _ { F _ { f p } } ) \right) ,\tag{7}
$$

$$
F _ { f } ^ { \prime } = F _ { f } + { \mathrm { R e s h a p e } } \left( { \mathrm { A t t n } } ( Q _ { F _ { f p } } , K _ { F _ { s p } } , V _ { F _ { s p } } ) \right) ,\tag{8}
$$

where $Q _ { F } , K _ { F }$ , and $V _ { F }$ denote query, key, and value projections of $F ,$ and $d _ { k }$ is the dimension of the key vectors.

## 4.3. Error-Based Curriculum Learning

Based on the observation from Q4 that simply increasing the number of editing steps will not afect the edited perceptual distance linearly, we apply an Error-Based Curriculum Learning (ECL) strategy to progressively train the model from easy samples (small number of editing steps with more clean traces) to hard samples (large number of editing steps with more non-linearly mixed efects) by adjusting the proportion of easy and hard samples based on error feedback. Specifically, we first divide the training dataset into M subsets according to the number of editing steps, where subset $S _ { i }$ contains images edited with i steps. During training, we maintain a dynamic sampling weight $P = \{ p _ { 1 } , \cdots , p _ { M } \}$ over these subsets. At the end of each epoch, based on validation performance, we compute the error rate $e _ { i }$ for each subset $S _ { i }$ . To adjust sampling probability, we follow the Self-Paced Learning (SPL) (Jiang et al., 2015) methodology and apply the polynomial soft weighting strategy (Gong et al., 2018). Formally, the training objective of the model with SPL regularizer is defined as:

$$
\operatorname* { m i n } _ { \mathbf { w } ; \mathbf { p } \in [ 0 , 1 ] ^ { \mathrm { N } } } \mathbb { E } ( \mathbf { w } , \mathbf { p } ; \lambda ) \sum _ { i = 1 } ^ { M } p _ { i } e _ { i } + Q ( \mathbf { p } ; \lambda ) .\tag{9}
$$

where w denotes model parameters, $p _ { i }$ is the weight assigned to the i-th subset (initialized to a uniform distribution), $e _ { i }$ is the average error for subset $i ,$ and λ refers to the age parameter incremented epoch by epoch.

The above learning objective is often optimized with the Alternative Optimization Strategy (Jiang et al., 2015). Based on the error rates $e _ { i \cdot }$ , we update the sampling probabilities $p _ { i }$ for each subset $S _ { i }$ as follows. Given the polynomial SPL regularizer set:

$$
Q = \left\{ f \bigl ( \mathbf { p } ; \lambda \bigr ) = \lambda \left( \frac { 1 } { t } \sum _ { i = 1 } ^ { M } p _ { i } ^ { t } - \sum _ { i = 1 } ^ { M } p _ { i } \right) \bigg | t > 1 \right\} ,\tag{10}
$$

the closed-form optimal solution for $p _ { i }$ can be derived as:

$$
p _ { i } ^ { * } = \left\{ \left( 1 - \frac { e _ { i } } { \lambda } \right) ^ { 1 / \left( t - 1 \right) } \quad e _ { i } < \lambda , \right.\tag{11}
$$

Table 4: Performance of models at diferent editing steps. Acc-S=Strict Accuracy. Acc-T=Toolwise Accuracy.
<table><tr><td>#Editing Step(s)</td><td colspan="2">1</td><td colspan="2">2</td><td colspan="2">3</td><td colspan="2">4</td><td colspan="2">5</td><td colspan="2">Mean</td></tr><tr><td>Method</td><td>Acc-S</td><td>Acc-T</td><td>Acc-S</td><td>Acc-T</td><td>Acc-S</td><td>Acc-T</td><td>Acc-S</td><td>Acc-T</td><td>Acc-S</td><td>Acc-T</td><td>Acc-S</td><td>Acc-T</td></tr><tr><td>ResNet18(He et al., 2016)</td><td>0.7982</td><td>0.9557</td><td>0.7123</td><td>0.9344</td><td>0.5271</td><td>0.8944</td><td>0.4096</td><td>0.8590</td><td>0.3485</td><td>0.8444</td><td>0.5372</td><td>0.8924</td></tr><tr><td>ResNet50(He et al., 2016)</td><td>0.8889</td><td>0.9766</td><td>0.8647</td><td>0.9715</td><td>0.7190</td><td>0.9416</td><td>0.5967</td><td>0.9116</td><td>0.5249</td><td>0.8941</td><td>0.7070</td><td>0.9366</td></tr><tr><td>Xception(Chollet, 2017)</td><td>0.8513</td><td>0.9689</td><td>0.7008</td><td>0.9372</td><td>0.5435</td><td>0.8995</td><td>0.3611</td><td>0.8409</td><td>0.2567</td><td>0.8008</td><td>0.5153</td><td>0.9061</td></tr><tr><td>SwinT(Liu et al., 2021)</td><td>0.9444</td><td>0.9878</td><td>0.8308</td><td>0.9646</td><td>0.6059</td><td>0.9190</td><td>0.4224</td><td>0.8778</td><td>0.3161</td><td>0.8537</td><td>0.5965</td><td>0.9150</td></tr><tr><td>F³-Net(Qian et al., 2020)</td><td>0.9753</td><td>0.9950</td><td>0.9524</td><td>0.9904</td><td>0.9131</td><td>0.9826</td><td>0.8550</td><td>0.9705</td><td>0.8094</td><td>0.9614</td><td>0.8963</td><td>0.9790</td></tr><tr><td>Qwen3-VL</td><td>0.0498</td><td>0.6702</td><td>0.0561</td><td>0.5110</td><td>0.0005</td><td>0.4788</td><td>0.0000</td><td>0.4664</td><td>0.0000</td><td>0.3989</td><td>0.0184</td><td>0.4864</td></tr><tr><td>Qwen3-VL-CLS</td><td>0.3990</td><td>0.8411</td><td>0.1532</td><td>0.7402</td><td>0.1668</td><td>0.7570</td><td>0.1700</td><td>0.7200</td><td>0.1764</td><td>0.7331</td><td>0.1852</td><td>0.7466</td></tr><tr><td>NPR(Tan et al., 2024)</td><td>0.7731</td><td>0.9027</td><td>0.3894</td><td>0.8537</td><td>0.3344</td><td>0.8235</td><td>0.2145</td><td>0.7705</td><td>0.2294</td><td>0.7684</td><td>0.3250</td><td>0.8152</td></tr><tr><td>FatFormer(Liu et al., 2024)</td><td>0.9514</td><td>0.9902</td><td>0.9070</td><td>0.9812</td><td>0.8066</td><td>0.9608</td><td>0.7029</td><td>0.9395</td><td>0.6108</td><td>0.9209</td><td>0.7853</td><td>0.9564</td></tr><tr><td>DPEC</td><td>0.9938</td><td>0.9988</td><td>0.9806</td><td>0.9961</td><td>0.9309</td><td>0.9861</td><td>0.8918</td><td>0.9779</td><td>0.8740</td><td>0.9742</td><td>0.9293</td><td>0.9856</td></tr></table>

$p _ { i } ^ { * }$ decreases with respect to $e _ { i }$ and increases with respect to λ. It holds that lim $\mathsf { l } _ { e _ { i } \to 0 } p _ { i } ^ { * } = 1$ $\begin{array} { r } { \operatorname* { l i m } _ { e _ { i } \to \infty } p _ { i } ^ { * } = 0 . } \end{array}$ $\begin{array} { r } { \operatorname* { l i m } _ { \lambda \to 0 } p _ { i } ^ { * } = 0 } \end{array}$ , and $\begin{array} { r } { \operatorname* { l i m } _ { \lambda \to \infty } p _ { i } ^ { * } = 1 } \end{array}$ . That is, as the accuracy increases and the training progresses, the ECL strategy will increasingly add hard samples into subsequent training.

## 5. Experiments

## 5.1. Experimental Settings

Evaluation Metrics We employ Strict Accuracy (Acc-S) and Toolwise Accuracy (Acc-T) to respectively reflect the sample and tool-level performance. Acc-S measures the accuracy of predicting the exact tools in the image editing process, while Acc-T shows the proportion of correctly identified editing tools. For example, given an image edited by three tools, if the prediction= {1, 0, 1} and label= {1, 1, 1}, Acc-S will be 0 and Acc-T will be 0.67.

Dataset We split MultiEdit into training and test sets in a 9:1 ratio. To avoid potential face ID bias, the original faces of training and testing samples do not overlap.

Baselines Since no existing models are perfectly tailored for our task, we selected models with widely-used architectures and state-of-the-art methods for image forgery detection and attribution as baselines, including ResNet18 (He et al., 2016), ResNet50 (He et al., 2016), Xception (Chollet, 2017), NPR (Tan et al., 2024), $\mathrm { F } ^ { 3 } { \mathrm { - N e t } }$ (Qian et al., 2020), SwinT (Liu et al., 2021), and FatFormer (Liu et al., 2024). Additionally, we employ vanilla Qwen3-VL-30B (Qwen Team, 2025) for zero-shot inference, and replace its unembedding layer with an MLP classifier (dubbed Qwen3-VL-CLS), so as to evaluate the performance of large vision-language models on this task.

Implementation Details For PFE, we utilize the first three layers of Xception (Chollet, 2017) as the backbone for dual-domain feature extraction. The SWDCT window size is set to 16 × 16 with a step size of 2. For ECL, we set the number of subsets M = 5 corresponding to images with 1 to 5 editing steps, and use a polynomial SPL regularizer with t = 3. For ECL, the initial sampling weight for each subset is set to 0.2 and the model age parameter λ is initialized to 0.5 and increased by 1 after each epoch. Further details are provided in the supplementary materials.

## 5.2. Main Results

DPEC outperforms all baselines on edited images of all dificulty levels Specifically, DPEC achieves a mean strict accuracy of 92.93% and a mean toolwise accuracy of 98.56%, significantly surpassing the baselines and SOTA methods. Also, its performance on the most dificult cases (5 editing steps) is superior, with 87.40% strict accuracy. This highlights the superiority of DPEC in addressing the challenges posed by multi-tool image editing attribution.

Moreover, consistent with the results in (Heo and Woo, 2025), as the number of editing steps increases, all models sufer varying degrees of performance degradation. This is primarily because the traces left by diferent editing tools on the image overlap and mask each other more severely. Further, methods are tailored to extract features from single-tool edited images and attribute all observed features exclusively to a single tool, naturally restricting their capacity to analyze the complex and cumulative efects.

## 5.3. Performance on Binary Detection

Although DPEC is designed to detect the involvement of editing tools in the post-detection stage, it can also be extended naturally to detect fake images. To evaluate the traditional detection performance, we conducted experiments where we replaced all deepfake-tool-edited images in the training set with images from FF++ and collected the prediction results associated with the deepfake tag among the multi-label prediction outcomes to determine whether the image is fake or not. As shown in Table 5, DPEC achieves performance competitive with those baselines on both FF++ (in-dataset) and Celeb-DF (cross-dataset). This also demonstrates that DPEC can efectively distinguish between real and edited images, which is crucial for practical digital forensics and media authenticity.

Table 5: Cross-dataset binary detection performance. For baseline methods, we directly apply binary classification heads after their backbone networks and train the whole model on the respective datasets. For DPEC, we replace all Deepfake-tool-edited images with those from the respective datasets and choose the toolwise prediction result (Deepfake tool) to determine whether the image is fake or not.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Train On</td><td colspan="2">Test Dataset (ACC)</td></tr><tr><td>FF++</td><td>Celeb-DF (Li et al., 2020)</td></tr><tr><td>ResNet50(He et al., 2016)</td><td rowspan="6"></td><td>0.887</td><td>0.615</td></tr><tr><td>Xception(Chollet, 2017)</td><td>0.919</td><td>0.695</td></tr><tr><td>F³-Net(Qian et al., 2020) FF+ +(Rössler et al., 2019)</td><td>0.926</td><td>0.788</td></tr><tr><td>NPR(Tan et al., 2024)</td><td>0.925</td><td>0.726</td></tr><tr><td>DPEC</td><td>0.922</td><td>0.786</td></tr><tr><td></td><td></td><td></td></tr></table>

## 5.4. Performance on Compressed Images

In real-world scenarios, the edited images may be compressed and resized when being uploaded to social media platforms, which can further obscure the traces left by editing tools. To evaluate the robustness of DPEC under such conditions, we apply common compression and resizing operations to the test set and assess the model’s performance. The results show that while there is a decline in accuracy compared to uncompressed images, DPEC still maintains a significant performance advantage over baseline methods, demonstrating its robustness in practical applications.

![](images/28aec33589c305238ad80e015982ac0046ee740b13e34942b50bf11b0fd0adfa.jpg)  
Figure 7: Performance comparison on compressed images.

## 5.5. Ablation Study

Module-Level: Efectiveness of PFE and ECL To validate the efectiveness of each component in DPEC, we conduct ablation studies by removing the PFE module, the ECL strategy, or both, respectively. As shown in G1 of Table 6, removing either PFE or ECL results in a significant drop in performance, demonstrating their crucial roles in feature extraction and progressive learning.

Table 6: Strict accuracies (Acc-S) of DPEC and the ablative variants at the module (G1) and transform (G2) levels. Results with 3 to 5 editing steps are reported for brevity. The model without PFE uses the vanilla backbone for feature extraction. The patch size for image splitting is set to match the sliding window size of SWDCT.
<table><tr><td colspan="2">#Editing Steps</td><td>3</td><td>4</td><td>5</td></tr><tr><td colspan="2">DPEC</td><td>0.9309</td><td>0.8918</td><td>0.8740</td></tr><tr><td rowspan="3">G1: Modules</td><td>w/o PFE</td><td>0.9121</td><td>0.8325</td><td>0.8234</td></tr><tr><td>w/o ECL</td><td>0.9270</td><td>0.8872</td><td>0.8568</td></tr><tr><td>w/o ECL &amp; PFE</td><td>0.8278</td><td>0.7968</td><td>0.7297</td></tr><tr><td rowspan="2">G2: Transforms</td><td>Patched-DWT</td><td>0.8223</td><td>0.7857</td><td>0.7438</td></tr><tr><td>Patched-DCT</td><td>0.8875</td><td>0.8226</td><td>0.8097</td></tr></table>

Transform-Level: Efectiveness of SWDCT To assess the impact of SWDCT in the PFE module, we replace it with standard DCT and DWT and evaluate the model’s performance. As shown in G2 of Table 6, the original model that adopts SWDCT outperforms the variants by approximately 14.7% and 7.1%, respectively. This superiority remains consistent across diferent editing steps, demonstrating that the SWDCT is more efective in providing distinguishable frequency-domain features for MIEA.

## 5.6. Further Analysis

Impact of the Size of Editing Areas To investigate whether large-area editing operations at the later steps impact the performance, we analyze the performance on the samples with editing lengths of 4 and 5. Based on whether the total editing area in the first/last two steps exceeds 50% of the face region, we categorize them into three groups (i.e., large-to-small, large-to-large, and small-to-large) and calculate the respective tool attribution performance.<sup>6</sup> From Table 7, we see that when the large-area editing operation appears in later steps, identifying tools used earlier becomes harder. This is intuitive, as large-area editing operations are more likely to overwrite or obscure the traces left by previous editing tools, making the attribution more challenging. Despite this, our method still outperforms the best baseline $\mathrm { F } ^ { 3 } .$ -Net with $\mathsf { A c c } . . \mathsf { S } = 8 1 . 3 2 \%$ on small-to-large samples.

Table 7: Performance on diferent subsets. The last row shows the largest performance drop in each column.
<table><tr><td colspan="2">Size of Edited Area</td><td rowspan="2">Acc-S</td><td rowspan="2">Acc-T</td><td rowspan="2">Recall Step 1 / 2</td></tr><tr><td>First 2 Steps Last 2 Steps</td><td></td></tr><tr><td>Large</td><td>Small</td><td></td><td>0.9563 0.9781</td><td>0.9970 / 0.9593</td></tr><tr><td>Large</td><td>Large</td><td>0.9437 0.9713</td><td></td><td> $0 . 9 8 3 4 / 0 . 9 5 9 2$ </td></tr><tr><td>Small</td><td>Large</td><td>0.8512 0.9256</td><td></td><td> $0 . 9 1 0 7 / 0 . 9 4 0 4$ </td></tr><tr><td>Δ(↓)</td><td></td><td>0.1051 0.0525</td><td></td><td> $\left. 0 . 0 8 7 3 \right/ 0 . 0 1 8 9$ </td></tr></table>

Decisive ROI Visualization To discover which facial regions play a decisive role when attributing diferent editing tools, we employ Grad-CAM (Selvaraju et al., 2017) to visualize the regions that DPEC focuses on when identifying each editing tool. The visualizations reveal that DPEC efectively attends to relevant facial areas corresponding to the specific edits made by each tool. Also, the ROIs are distinct for diferent tools, showing the efectiveness of the designed PFE module, selecting the most tool-relevant representative image patches to attribute for each editing tool. However, an interesting observation is that, for some tools like OpenCV, the model probably focuses on non-facial regions such as the neck and hair. The reason could be that the same facial skin regions are also afected by other editing tools, making it challenging to isolate the efect of a single tool solely based on facial areas. Thus, the model may leverage cues from adjacent non-facial regions to enhance attribution accuracy.

![](images/6442322473f6288e5f46fd708d77923e39d824131d09f13ea826e00f55603651.jpg)  
a) Original b) Edited c) CSD-MT d) SHMT e) OpenCV f) FLUX.1 g) Deepfake  
Figure 8: Decisive ROI Visualization for Diferent Editing Tools. The red-highlighted regions indicate the areas that DPEC focuses on when attributing each tool. a) Original images; b) Multi-tool edited images; c)-g) Grad-CAM heatmaps for diferent editing tools. Editing targets are marked with colored boxes for clarity.

## 6. Conclusion

In this study, we introduced a new task, Multi-tool Image Editing Attribution, which identifies multiple editing tools involved in single-image editing, making the image editing process more transparent and supporting accountability for image authenticity. We first constructed a 500k-scale dataset, MultiEdit, which covers various combinations based on six editing tools. Motivated by the analysis of MultiEdit, we proposed DPEC, a multi-tool attribution method that consists of two branches that leverage Patchwise Feature Enhancement to efectively learn distinguishable features of individual tools in two domains. Also, we designed an Error-based Curriculum Learning strategy to progressively train the model from easy samples to hard samples to capture the unique and significant tool characteristics in a mixture of editing traces. Extensive experiments showed the superiority of DPEC in multi-tool attribution. In the future, we plan to further explore tool-use sequence prediction and test our method on general natural images.

## References

Belhassen Bayar and Matthew C Stamm. 2018. Constrained Convolutional Neural Networks: A New Approach Towards General Purpose Image Manipulation Detection. IEEE Transactions on Information Forensics and Security, 13(11):2691–2706.

Stella Bounareli, Christos Tzelepis, Vasileios Argyriou, Ioannis Patras, and Georgios Tzimiropoulos. 2025. DiffusionAct: Controllable Difusion Autoencoder for One-shot Face Reenactment. Preprint, arXiv:2403.17217.

Tu Bui, Ning Yu, and John Collomosse. 2022. Repmix: Representation mixing for robust attribution of synthesized images. In European Conference on Computer Vision, pages 146–163. Springer.

François Chollet. 2017. Xception: Deep Learning with Depthwise Separable Convolutions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1251–1258.

Di Cooke, Abigail Edwards, Sophia Barkof, and Kathryn Kelly. 2024. As Good As A Coin Toss: Human detection of AI-generated images, videos, audio, and audiovisual stimuli. Preprint, arXiv:2403.16760.

I.J. Cox, J. Kilian, F.T. Leighton, and T. Shamoon. 1997. Secure spread spectrum watermarking for multimedia. IEEE Transactions on Image Processing, 6(12):1673–1687.

Nadiia Davydiuk, Elisha Krieg, Jens Gaitzsch, Patrick M McCall, Günter K Auernhammer, Mu Yang, Joseph B Tracy, Sara Bals, Wolfgang J Parak, Nicholas A Kotov, et al. 2025. The rising danger of AI-generated image in nanomaterials science and what we can do about it. Nature Nanotechnology, pages 1–4.

Nicholas Dufour, Arkanath Pathak, Pouya Samangouei, Nikki Hariri, Shashi Deshetti, Andrew Dudfield, Christopher Guess, Pablo Hernández Escayola, Bobby Tran, Mevan Babakar, and Christoph Bregler. 2024. AMMeBa: A Large-Scale Survey and Dataset of Media-Based Misinformation In-The-Wild. Preprint, arXiv:2405.11697.

Shreyan Ganguly, Aditya Ganguly, Sk Mohiuddin, Samir Malakar, and Ram Sarkar. 2022. ViXNet: Vision Transformer with Xception Network for deepfakes based video and image forgery detection. Expert Systems with Applications, 210:118423.

Maoguo Gong, Hao Li, Deyu Meng, Qiguang Miao, and Jia Liu. 2018. Decomposition-based evolutionary multiobjective optimization to self-paced learning. IEEE Transactions on Evolutionary Computation, 23(2):288–302.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, pages 2672–2680.

Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. 2022. Accelerate: Training and inference at scale made simple, eficient and adaptable. https://github.com/huggingface/accelerate.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778.

Minji Heo and Simon S. Woo. 2025. FakeChain: Exposing Shallow Cues in Multi-Step Deepfake Detection. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, page 855–866. ACM.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Ziqi Huang, Kelvin C.K. Chan, Yuming Jiang, and Ziwei Liu. 2023. Collaborative Difusion for Multi-Modal Face Generation and Editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Kaiwen Jiang, Shu-Yu Chen, Feng-Lin Liu, Hongbo Fu, and Lin Gao. 2025. Towards High-Quality and Disentangled Face Editing in a 3D GAN. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, and Alexander Hauptmann. 2015. Self-paced curriculum learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29.

KJ Joseph, Prateksha Udhayanan, Tripti Shukla, Aishwarya Agarwal, Srikrishna Karanam, Koustava Goswami, and Balaji Vasan Srinivasan. 2024. Iterative multi-granular image editing using difusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 8107–8116.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. 2025. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. Preprint, arXiv:2506.15742.

Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. 2020. Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3204–3213.

Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, and Sanja Fidler. 2021. EditGAN: High-Precision Semantic Image Editing. Advances in Neural Information Processing Systems, 34:16331– 16345.

Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Jingdong Wang, and Yao Zhao. 2024. Forgeryaware adaptive transformer for generalizable synthetic image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10770–10780.

Xuannan Liu, Zekun Li, Peipei Li, Huaibo Huang, Shuhan Xia, Xing Cui, Linzhi Huang, Weihong Deng, and Zhaofeng He. 2025. MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs. Preprint, arXiv:2406.08772.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10012–10022.

Gaël Mahfoudi, Badr Tajini, Florent Retraint, Frederic Morain-Nicolier, Jean Luc Dugelay, and Marc Pic. 2019. DEFACTO: Image and Face Manipulation Dataset. In 2019 27th European Signal Processing Conference, pages 1–5.

Daniel Moreira, Aparna Bharati, Joel Brogan, Allan Pinto, Michael Parowski, Kevin W. Bowyer, Patrick J. Flynn, Anderson Rocha, and Walter J. Scheirer. 2018. Image Provenance Analysis at Scale. IEEE Transactions on Image Processing, 27(12):6109–6123.

Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. 2020. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In European Conference on Computer Vision, pages 86–103. Springer.

Qwen Team. 2025. Qwen3-VL: Sharper Vision, Deeper Thought, Broader Action.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695.

Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. 2019. FaceForensics++: Learning to Detect Manipulated Facial Images. In International Conference on Computer Vision.

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision, pages 618–626.

Himanshu Kumar Singh and Amit Kumar Singh. 2024. Digital Image Watermarking Using Deep Learning. Multimedia Tools and Applications, 83(1):2979–2994.

Chris Stokel-Walker. 2023. A Journalist Believes He Was Banned From Midjourney After His AI Images Of Donald Trump Getting Arrested Went Viral. Accessed: 2026-04-01.

Zhaoyang Sun, Shengwu Xiong, Yaxiong Chen, Fei Du, Weihua Chen, Fang Wang, and Yi Rong. 2024a. SHMT: Self-supervised Hierarchical Makeup Transfer via Latent Difusion Models. In Advances in Neural Information Processing Systems.

Zhaoyang Sun, Shengwu Xiong, Yaxiong Chen, and Yi Rong. 2024b. Content-Style Decoupling for Unsupervised Makeup Transfer without Generating Pseudo Ground Truth. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. 2024. Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28130– 28139.

Zhiya Tan, Xin Zhang, and Joey Tianyi Zhou. 2025. Modelship Attribution: Tracing Multi-Stage Manipulations Across Generative Models. Preprint, arXiv:2506.02405.

Junke Wang, Zhenxin Li, Chao Zhang, Jingjing Chen, Zuxuan Wu, Larry S Davis, and Yu-Gang Jiang. 2025. Fighting Malicious Media Data: A Survey on Tampering Detection and Deepfake Detection. Proceedings of the IEEE.

Sheng-Yu Wang, Oliver Wang, Andrew Owens, Richard Zhang, and Alexei A Efros. 2019. Detecting Photoshopped Faces by Scripting Photoshop. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10072–10081.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612.

Mengting Wei, Tuomas Varanka, Yante Li, Xingxun Jiang, Huai-Qian Khor, and Guoying Zhao. 2025. Towards Consistent and Controllable Image Synthesis for Face Editing. Preprint, arXiv:2502.02465.

Shiqian Yan, Yuanyuan Han, Xiaoya Fan, and Zhong Wang. 2022. Image Tamper Detection Based on Two-Stream Attention Faster R-CNN. In 2022 7th International Conference on Signal and Image Processing, pages 305–310. IEEE.

Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. 2025. GPT-ImgEval: A Comprehensive Benchmark for Diagnosing GPT4o in Image Generation. Preprint, arXiv:2504.02782.

Chao Yang and Ser-Nam Lim. 2019. Unconstrained Facial Expression Transfer using Style-based Generator. Preprint, arXiv:1912.06253.

Tianyun Yang, Juan Cao, Danding Wang, and Chang Xu. 2025. Model Synthesis for Zero-Shot Model Attribution. IEEE Transactions on Multimedia, pages 1–13.

Tianyun Yang, Ziyao Huang, Juan Cao, Lei Li, and Xirong Li. 2022. Deepfake Network Architecture Attribution. In Proceedings of the 36th AAAI Conference on Artificial Intelligence.

Tianyun Yang, Danding Wang, Fan Tang, Xinying Zhao, Juan Cao, and Sheng Tang. 2023. Progressive Open Space Expansion for Open-Set Model Attribution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15856–15865.

Xin Yang, Yuezun Li, Honggang Qi, and Siwei Lyu. 2019. Exposing GAN-synthesized Faces Using Landmark Locations. In Proceedings of the ACM Workshop on Information Hiding and Multimedia Security, page 113–118, New York, NY, USA. Association for Computing Machinery.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. 2025. ImgEdit: A Unified Image Editing Dataset and Benchmark. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph Stoecklin, Heqing Huang, and Ian Molloy. 2018a. Protecting Intellectual Property of Deep Neural Networks with Watermarking. In Proceedings of the 2018 Asia Conference on Computer and Communications Security, pages 159–172.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018b. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595.

Yushu Zhang, Zhibin Fu, Shuren Qi, Mingfu Xue, Xiaochun Cao, and Yong Xiang. 2023. PS-net: A learning strategy for accurately exposing the professional photoshop inpainting. IEEE Transactions on Neural Networks and Learning Systems, 35(10):13874–13886.

Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou, and Jiwen Lu. 2023. DifSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8568–8577.

Zijun Zhou, Yingying Deng, Xiangyu He, Weiming Dong, and Fan Tang. 2025. Multi-turn consistent image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15792– 15801.

## Supplementary Material

## We organize our supplementary material as follows:

Sec. A Limitations and Future Work

Sec. B Implementation Details:

• Sec. B.1 Details of using Qwen3-VL as an edit judge

• Sec. B.2 Details of the DPEC design

• Sec. B.3 Details of the training process

Sec. C Edited Image Cases

Sec. D Additional Analysis:

• Sec. D.1 Key hyperparameter sensitivity

• Sec. D.2 Failure case analysis

• Sec. D.3 Attempt at sequence recovery

## A. Discussion on Limitations and Future Work

To address the potential attribution issue brought by the recent transition from single-tool image generation or editing using GANs and difusion to multi-tool image editing, we constructed the infrastructure for multi-tool image editing attribution in this paper by formulating the task, building a large-scale multi-tool-edited image dataset, implementing nine baselines, and providing a new method. Our work could be regarded as an extension of deepfake detection and attribution to catch up with the recent progress on generative AI, and also a type of sister work of the conventional image provenance analysis that finds images that provide materials for a manipulated one (Moreira et al., 2018). Despite this, as a preliminary study, we identify the following limitations:

1) The dataset does not consider general-purpose image editing. Although the latest image editing tools are capable of general editing, the MultiEdit dataset only considers human face editing. We focus on face samples because face editing is of high frequency, often related to multiple edits, and has been used to fabricate fake news about celebrities multiple times. This would make our dataset more aligned with real-world cases. We also admit that this could bring some convenience to our analysis due to the constraints of human facial structure. We plan to extend the dataset to edited images in general domains.

2) We did not perform tool-use order-level predictions. We formulate the task as a multi-label classification task regardless of the tool-use order. During our research, we made an initial attempt but finally put this aside as no reasonable results were observed (will show in Sec. D.3). We speculate that the efects of each editing tool may partially follow commutativity and thus be indistinguishable in terms of the order. Further experiments on the feasibility of order prediction and its impact on set-level prediction are worth deeper exploration, and we leave it to future work.

3) We mainly follow a closed-set setting that limits the attribution models’ extensibility. Following the research line of deepfake attribution (Yang et al., 2022, 2023, 2025), we start with a fundamental closed-set setting, i.e., the attribution model can only attribute the editing of an image to pre-set editing tool candidates. For an image edited by an unseen tool, the model cannot provide a reliable prediction. After validating the attribution possibility through this work, we plan to extend the closed-set setting to an open-set one, making it more applicable to complex real-world scenarios.

## B. Implementation Details

## B.1. Qwen3-VL Evaluation Instructions

To determine whether a specific editing operation has been successfully applied to an image, we use Qwen3- VL to assess the success of the editing, as stated in Section 3.2. The instruction template employed is shown in Figure S1 below. We further conducted manual sampling verification to assess Qwen3-VL’s competence for this task. Results indicate over 90% consistency between Qwen3-VL’s judgments and human evaluations, confirming its suitability as a reliable identifier.

Now, you are an image editing efect evaluation expert, and please help me complete the following   
task:   
You will be given:   
1. Original image (Image 1)   
2. Edited image (Image 2)   
3. Description of the editing target. e.g., Add blush to the man, enlarge the eyes of the person, etc.]   
Please compare Image 1 and Image 2, focusing on checking whether the above-mentioned editing   
operation has been successfully and accurately applied to the image.   
The criteria for your judgment are: whether the changes described in the editing target are clearly   
and completely presented in Image 2 and conform to the core target of the operation (if the editing   
target is partially implemented, incorrectly implemented, or not implemented at all, it shall be   
deemed as a failure).   
PLEASE OUTPUT A SINGLE KEY:VALUE PAIR IN STRICT JSON FORMAT, DO NOT OUTPUT ANY   
OTHER INFORMATION!!!!   
The key is result, value is Success or Fail.   
Two output examples are given below:   
{result: "Success"}   
{result: "Fail"}  
Figure S1: Prompt template for Qwen3-VL-based image editing efect evaluation

## B.2. Details of DPEC Design

DPEC mainly incorporates two modules: Patchwise feature enhancement (PFE) and Error-based Curriculum Learning (ECL) training strategy, shown in Section 4 of the main text. For the PFE module, we apply Slide-Window Cosine Discrete Transform (SWDCT) to transform the input image from the spatial to frequency domain. And a window size of 16×16 and a step size of 2 were employed in the sliding window process. In the cross attention process, the number of heads is set to 8, and the hidden size in cross attention $d _ { k }$ is set to 768. For ECL, we set the order t of the polynomial in Equation (9) to 3 and the age parameter in Equation (10) is initially set to 0.5, and is incremented by 1 after each training epoch.

## B.3. Details of Training Process

In all training processes, we use the Accelerate library (Gugger et al., 2022) for distributed training, and the learning rate is initialized to 3e-4. And we apply a hybrid learning rate scheduling strategy that combines linear warmup and cosine annealing. During the warmup phase (when the current step is less than the warmup steps), the learning rate increases linearly with the number of steps. The scaling factor is calculated $\begin{array} { r } { \mathsf { a s } \frac { ( T + 1 ) } { T _ { 0 } } } \end{array}$ , where T and $T _ { 0 }$ denote current epoch and warmup epochs, respectively. After the warmup phase, the learning rate decays following a cosine annealing pattern. First, the progress is calculated as the ratio of the steps elapsed since warmup to the total remaining steps after warmup (ranging from 0 to 1). Then, a cosine-based decay curve is generated using the formula $W = 0 . 5 * \left( 1 + { c o s } \left( \pi * T - T _ { 0 } \right) \big / \big ( T _ { a l l } - T _ { 0 } \big ) \right)$ , where $T _ { a l l }$ refers to total training epochs. Finally, the decay factor W ∗ 0.99 + 0.01 is applied multiplicatively to the optimizer’s initial learning rates.

For Qwen3-VL and Qwen3-VL-CLS models, we select the Qwen3-VL-30B (Qwen Team, 2025) as the base model and use the instruction in Figure S2 for zero-shot inference and training. In the training process of Qwen3-VL-CLS, we apply the low-rank approximation (LoRA) (Hu et al., 2022) to train both text and image encoders and use binary cross-entropy loss for the multi-label classification loss.

![](images/c5a9d8f3ed4560f8e492208d30558f8a21f176d1ae0c490d7bd8fa8c41fa1bb5.jpg)  
Figure S2: Prompt template for Qwen3-VL-based attribution

## C. Edited Image Cases

Figures S3 and S4 exhibit samples in MultiEdit. During image editing, we select 40 images in total as reference images for the tools CSD-MT and SHMT. More edited samples are provided in the supplementary material. In some cases, the image edited with multiple tools may feature relatively heavy makeup, resembling stage makeup rather than everyday looks. Because the task evaluation should consider more diverse scenarios, not limited to daily natural looks, we retained these samples for the sake of data diversity.

e) Step4

a) Original b) Step1 c) Step2  
![](images/ce4078c3970b44d51d96a1b4b297c3ab07e6dda60bcecec1766867dc39d08210.jpg)  
Figure S3: Multi-tool edited examples on real faces.

## D. Additional Analysis

## D.1. Results on Hyperparameter Sensitivity

We conducted ablation studies on the key operation and hyperparameters of PFE and ECL, and the results are shown in Table S1. It illustrates that our method exhibits a certain degree of hyperparameter insensitivity. Models deviating from the optimal hyperparameter still maintain high performance on images edited by more than two tools.

## D.2. Failure Case Analysis

Figure S5 depicts a group of sequentially edited images that DPEC failed to detect some of the editing tools. In this sequential editing process, the traces of the preceding small-area editing tools are obfuscated by later

a) Original b) Step1:DF c) Step2

![](images/cda0c4e83369aa5cdda50108b173475ac9d4ed6511872f26743c9314ee8c34ee.jpg)  
Figure S4: Multi-tool edited examples on deepfake faces.

large-area edits, resulting in missed detection on preceding editing tools. This is somewhat aligned with the results in Table 6 of the main text that the size of the pixel-level diferences between two consecutive edited images would influence the detectability of the traces left by previous tools.

## D.3. An Attempt at Tool-Use Sequence Recovery

As we stated in the main text, we formulate multi-tool image attribution as a multi-label classification task regardless of the tool-use order. That is, we perform a set-level attribution rather than a sequence-level one. In this section, we made a preliminary attempt at sequence-level attribution (i.e., tool-use sequence recovery for an edited image) to know whether the editing order is predictable (or whether the tool-use does not hold commutativity). Unlike previous work modeling this task as an autoregressive prediction task, since images are exhibited as static media and thus non-sequential inherently, we change the output of the model DPEC to

Table S1: Ablation on hyperparameters selection in DPEC. Slide Window Step = Size × Ratio. The row in red means the best hyperparameters and corresponding performances.
<table><tr><td colspan="2">Module</td><td colspan="3"># Editing Steps 3</td></tr><tr><td colspan="3">Params.</td><td>4</td><td>5</td></tr><tr><td rowspan="10">SWDCT Size;Ratio ECL</td><td rowspan="3">8×8;1/8</td><td>0.8832</td><td>0.7832</td><td>0.7256</td></tr><tr><td>8×8;1/4 0.8798</td><td>0.7796</td><td>0.7185</td></tr><tr><td>8×8;1/2 0.8837</td><td>0.8012</td><td>0.7115</td></tr><tr><td>16×16;1/8</td><td>0.9309</td><td>0.8918</td><td>0.8740</td></tr><tr><td>16×16;1/4</td><td>0.9387</td><td>0.8848</td><td>0.8675</td></tr><tr><td>16×16;1/2</td><td>0.9247</td><td>0.8564</td><td>0.8703</td></tr><tr><td>32×32;1/8</td><td>0.9198</td><td>0.8756</td><td>0.8564</td></tr><tr><td>32×32;1/4</td><td>0.9227</td><td>0.8530</td><td>0.8501</td></tr><tr><td>32×32;1/2</td><td>0.9017</td><td>0.8534</td><td>0.8567</td></tr><tr><td>64×64;1/8</td><td>0.8917</td><td>0.8321</td><td>0.8032</td></tr><tr><td rowspan="5">Increase</td><td>64×64;1/4</td><td>0.8546</td><td>0.8062</td><td>0.8042</td></tr><tr><td>64×64;1/2</td><td>0.8347</td><td>0.7865</td><td>0.7564</td></tr><tr><td>0.5</td><td>0.9423</td><td>0.8716</td><td>0.8392</td></tr><tr><td>1.0</td><td>0.9309</td><td>0.8918</td><td>0.8740</td></tr><tr><td>1.5</td><td>0.9323 0.9218</td><td>0.8716 0.8567</td><td>0.8692</td></tr><tr><td colspan="2">rate of λ</td><td>2.0 2.5</td><td>0.9134 0.8874</td><td></td><td>0.8548 0.8632</td></tr></table>

![](images/b2f2acbd05a3f63732850f0dfd39879195038dec69ce9417f79fabd0c0db1ec9.jpg)  
Figure S5: A failure case where some tools are in editing process but not detected. The missed tools are in red.

$$
M = [ \stackrel { [ 0  } { \vdots } \stackrel {  0 } { \vdots } \stackrel {  \cdots } \stackrel {  1 } { \ddots } \stackrel {  \vdots } \stackrel {  ] } { \vdots } ] _ { N \times T } ,\tag{S1}
$$

where N denotes the maximum editing steps and T denotes the total number of used editing tools. $M [ i , j ] = 1$   
means that the model predicts $T o o l _ { j }$ is used at the i-th editing step.

For evaluation, we apply a sequence-level exact match score (EM) that requires both the tool set and order to be correct. The model achieves 1.67% in EM on the images edited by 5 tools and 12.64% in EM on average among all test samples. These results show that the tool-use sequence recovery is dificult for the current model, and further exploration is expected.