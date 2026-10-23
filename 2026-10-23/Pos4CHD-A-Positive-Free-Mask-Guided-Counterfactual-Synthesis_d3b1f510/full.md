# Pos4CHD: A Positive-Free Mask-Guided Counterfactual Synthesis Framework for Fetal Congenital Heart Defect Analysis

Wenjie Xuan, Yapeng Li, Hang Su, Chen Cheng, Juhua Liu, Member, IEEE, Sheng Zhao, Ziming Huang, and Bo Du, Senior Member, IEEE

Abstract— The scarcity of positive cases remains a ma-<sup>CHD</sup> jor bottleneck in developing automated diagnostic systemsunterfactual Synthesis for fetal Congenital Heart Defects (CHDs), particularly for rare subtypes. Although normal cases are accessible with effort, positive cases are hard to collect due to low inci-<sub>like</sub> <sub>VSD?</sub> dence rates and privacy concerns, causing severe class imbalance or absence of positive training samples. Existing works resort to synthesizing positives with conditional generative models, but they are task-specific and require posi-<sup>al</sup> <sup>4CH</sup> <sup>view</sup> <sub>tive</sub> <sub>training</sub> <sub>samples,</sub> <sub>hindering</sub> <sub>application</sub> <sub>for</sub> <sub>fetal</sub> <sub>CHDs.(Trained</sub> <sub>w</sub> <sub>Positives)</sub> CHD?<sub>ct</sub> <sub>on</sub> <sub>inter-</sub> To this end, we propose a positive-free mask-guided coun-<sup>icular</sup> <sup>septum.</sup> terfactual generation framework for fetal echocardiography,<sub>es Weak</sub> <sub>Condition Weak</sub> <sub>Spatial</sub> <sub>Alignment</sub> namely Pos4CHD, which enables realistic, high-quality synthesis of pseudo-positive samples from semantic masks,.e., FetalFlex with only normal samples at hand. Technically, we first develop FE-Gen, a mask-guided generative model tailored for<sub>Generative</sub> <sub>Model</sub> fetal heart ultrasound synthesis that mitigates histogram(Zero-shot) drift and enhances image-mask spatial alignment. Building on FE-Gen, Pos4CHD employs a mask-to-image generation paradigm for CHD synthesis. It first manipulates semantic masks to simulate fetal-heart structural anomalies and, Ours then renders anatomically consistent images. Extensive experiments on in-house datasets, together with expert<sub>Generative</sub> <sub>Model</sub> evaluations, demonstrate the visual fidelity (−12.97 FID)(Zero-shot) and anatomical consistency (+3.96% DSC) of synthesized images. Further validations across diverse real-world CHD diagnosis tasks exhibit promising results and consistent increments, including zero-shot classification (achieving 93.25% ACC on EbA and 72.92% ACC on TAPVC), lesion detection (+6.95% mAP50 on VSD), and anatomy segmentation (+10.49% DSC on fetal heart) compared with baselines. Codes and models are publicly available on GitHub.

Index Terms— Fetal echocardiography, diffusion model, counterfactual synthesis, data augmentation

## I. INTRODUCTION

This work was supported in part by the National Key Research and Development Program of China under Grant 2023YFC2705701, in part by the National Natural Science Foundation of China under Grants 62225113, 625B2132 and U23B2048, in part by the Science and Technology Major Project of Hubei Province under Grants 2025BCB026, and in part by the China Birth Defect Intervention and Rescue Foundation and Beijing ByteDance Charity Foundation under 2026CFBD-CKZN-005. Corresponding author: Juhua Liu (e-mail: liujuhua@whu.edu.cn)

Wenjie Xuan, Yapeng Li, Hang Su, Juhua Liu and Bo Du are with the School of Computer Science, Wuhan University, Wuhan, China. (<sup></sup>: {dreamxwj, yapengli, hangsu, liujuhua, dubo}@whu.edu.cn).

Chen Cheng, Sheng Zhao and Ziming Huang are with Maternal and Child Health Hospital of Hubei Province, Wuhan, China. (<sup></sup>: {chengchen 777, zhao1015568495}@163.com, huangziming@hbfy.com).

![](images/7669e1a369bfaa510581c4acade46ad4650ad68599d810f920d75faaed0a0fd1.jpg)  
Fig. 1. Paradigm comparison for (a) class/text-, (b) layout-, (c) maskguided medical counterfactual synthesis. Since CHDs are structural anomalies, we employ a mask-guided paradigm for fetal CHD synthesis without positive training samples, providing dense structural guidance, fine spatial alignment, and interpretability for downstream analysis.

leading causes of infant mortality, which affects approximately 0.8∼1.0% of live births [1], [2]. Fetal echocardiography, i.e., fetal heart ultrasound, is the primary tool for prenatal examination due to its radiation-free, low-cost, and real-time nature. Clinicians typically examine fetal heart anatomical structures using standardized cardiac views [3] like the Four-Chamber (4CH) view, Three-Vessel and Trachea (3VT) view, etc., which provide essential and vital references for cardiac function analysis. This practice helps early detection and diagnosis of various CHD subtypes, including Ebstein’s Anomaly (EbA), Total Anomalous Pulmonary Venous Connection (TAPVC), Ventricular Septal Defect (VSD), etc., contributing to improved worldwide pregnancy healthcare [4].

However, compared with existing AI systems for MRI, CT, and X-rays [5], AI-assisted fetal CHD detection and analysis remains underdeveloped due to severe data scarcity, which stems from: (1) the acquisition of standardized fetal cardiac views requires expertise and experience, (2) varied view quality across operators and examinations, and (3) cardiac anatomical annotation is labor-intensive and requires expertise, resulting in few high-quality public datasets. More importantly, given the wide range of CHD subtypes (≥27), although normal cases are accessible through clinical efforts, positive cases for rare CHDs are often out of reach owing to low incidence rates (<1/10, 000) [2] and privacy constraints. This positive-sample scarcity hinders the training of downstream models, leading to severe class imbalance and potential generalization issues [6].

Recent advances in generative models, e.g., diffusion models, offer a promising solution to medical data scarcity via image synthesis. Existing studies focused on improving image fidelity [7], [8] and anatomical plausibility [9], [10] across different imaging modalities, and adapting generative foundation models for medical applications [11]–[13]. Conditional generative models, e.g., Stable Diffusion (SD) [14] and ControlNet [15], further enable task-specific conditional guidance to synthesize images with desired properties. Practices (see Fig. 1) include class labels or text prompts [16], [17] for anatomical description, layouts [18] for lesion position and size, segmentation masks [19], [20] for structure and shape, with optional reference images [21], [22] to provide extra context for local variations. Such methods can produce paired image-condition data for augmentation purposes and benefit various downstream tasks including classification, segmentation, and multi-center generalization [10], [23]. Recent counterfactual synthesis explores generating positive cases, such as tumors [24], thyroid cancers [25], breast lesions [26], etc.

Despite these advances, directly applying such approaches to fetal CHD synthesis remains challenging. First, most works are designed for specific tasks and limited scenarios. We need to redesign a framework for fetal CHD synthesis, where the complicated cardiac anatomy and heterogeneous CHD subtypes raise the complexity. Second, current methods [24]– [26] require positive samples to learn lesion-specific textures and structures, implicitly assuming the availability of positive cases. But this assumption is violated because of the absence of fetal CHD samples, particularly for rare subtypes. Third, existing works lack fine-grained control over synthesized images. As shown in Fig. 1, given that CHDs are mostly structural anomalies, the class labels and text prompts offer limited spatial control, where the anatomical plausibility of synthesized positives depends on generative priors and is hardly guaranteed. Though bounding boxes offer locations and enable simulation of position and size variations, they still fall short in depicting morphology and fine-grained variations, thus insufficient to distinguish anatomical differences across CHD subtypes. Therefore, we raise the question: can we design a unified framework to synthesize realistic and anatomyconsistent samples of diverse CHD subtypes for downstream analysis, with only negative samples at hand?

To this end, we propose Pos4CHD, a positive-free maskguided counterfactual synthesis framework for fetal CHD analysis. Firstly, considering the absence of publicly available datasets, we collect ∼8k fetal 4CH views and ∼500 expertannotations covering 14 key anatomical structures (Sec. III-C.1), and train FE-Gen (Sec. III-C.2), a mask-conditioned generator for fetal echocardiography, which features: 1) learnable semantic queries that enhance the spatial alignment between synthesized images and conditioning masks, and 2) a regularization term that mitigates the histogram drift when adapting diffusion models for ultrasound with improved image fidelity. Building on FE-Gen, Pos4CHD (Sec. III-B) utilizes a mask-to-image generation strategy to synthesize fetal CHDs from normal samples. Specifically, it first manipulates cardiac masks to simulate structural anomalies with respect to clinical prior knowledge, realizing rule-based counterfactual mask generation for fetal CHDs. Then, FE-Gen renders corresponding echocardiographic images guided by masks, yielding spatially aligned pseudo-positive samples. These synthesized samples are subsequently utilized to augment downstream training data after quality checking. Experiments and expert evaluations demonstrate the image fidelity and anatomical consistency of FE-Gen (Sec. IV-B.1). Extensive experiments on realworld scenarios, including zero-shot EbA/TAPVC classification, VSD detection, and fetal cardiac anatomy segmentation, verify the effectiveness of Pos4CHD, which not only enables training models without real positives but also consistently outperforms the baseline (Sec. IV-B.2). Notably, models trained solely on synthesized data can generalize well to real samples. Our contributions are summarized as follows:

1) We develop Pos4CHD, a positive-free mask-to-image counterfactual generation framework to tackle the scarcity of fetal CHD cases, especially rare subtypes, which offers pseudo-positive image-mask pairs for diverse CHDs through rule-based counterfactual mask manipulation and mask-guided fetal ultrasound synthesis.

2) We collect normal fetal 4CH-view datasets with rich pixel-level anatomical annotation. Based on the collected data, we train FE-Gen, a mask-guided generative model for fetal echocardiography, realizing improved ultrasound fidelity and advanced image-mask alignment.

3) Experiments together with expert evaluations demonstrate the effectiveness of our synthesized samples for data augmentation across various downstream tasks, bringing consistent improvements on zero-shot CHD classification, lesion detection, and fetal heart anatomy segmentation. Detailed ablations and discussions provide valuable insights to the community on how to obtain and utilize the synthesized positive samples with our method.

## II. RELATED WORK

## A. Medical Image Generation

Medical image generation has been widely explored to alleviate the scarcity, limited accessibility, and high acquisition costs of data across different modalities [5]. Following early GAN-based approaches, diffusion models have become the primary choice for high image fidelity and training stability nowadays. Recent studies moved beyond visual realism toward medical-specific challenges, including anatomical fidelity [9], [10], [20], few-shot adaptation [12], [13], and privacy preservation. Recently, controllable generation has gained increasing attention for downstream clinical applications. By incorporating conditions like patient metadata [16], modalities and styles [27], pathological states [28], and anatomical layouts or masks [19], [24], conditional generative models enable targeted synthesis of samples with desired characteristics. Such synthetic samples are privacy-friendly, valuable for mitigating class imbalance, enriching data diversity, and improving the robustness and generalization of medical AI models.

## B. Ultrasound Image Generation

Ultrasound image generation attracts growing attention due to its broad clinical use and distinctive imaging characteristics. Recent studies incorporated physical priors [7] and structural constraints [29], [30] to improve fidelity. Controllable ultrasound generation has also been explored for downstream applications, such as augmenting image-mask pairs for segmentation [31] and enriching data diversity [32], [33].

Fetal ultrasound generation is a crucial research direction. FU-LoRA [34] and DiFUSAL [35] fine-tuned SD for textconditioned fetal ultrasound synthesis to enhance plane classification and gestational age prediction. FVEDM [36] combines image inpainting with semantic masks and edge guidance to introduce localized variations to 4CH views, thereby enriching appearance diversity for CHD classification. FetalFlex [18] employs masked inpainting with anatomical layouts and global prompts for multi-plane fetal ultrasound synthesis. Although similar to FetalFlex in motivation, we adopt a fine-grained mask-to-image paradigm for ultrasound synthesis to achieve precise spatial control, and we mitigate the histogram shift to improve fidelity. Besides, Pos4CHD targets the more challenging fetal CHD synthesis for downstream analysis, where fine-grained anatomical fidelity is critical for distinguishing heterogeneous subtypes with complex cardiac structures.

## C. Counterfactual Synthesis

Counterfactual synthesis produces hypothetical images by altering specific attributes of an existing sample while preserving its remaining characteristics, corresponding to what-$i f$ interventions as illustrated in Fig. 1. Such counterfactual samples are valuable for model robustness evaluation and interpretation, anomaly detection, and data augmentation. This paper focuses on augmenting positive cases. Existing works typically rely on conditional generation or image editing [21], [22] to synthesize positives, guided by class labels or text prompts [37], [38], or spatial conditions like edges, layouts, and masks [39], [40] in Fig. 1. However, disease-specific synthesis usually requires pathological samples or supervision to learn appearance and structural characteristics associated with the target disease. For example, Tiger [25] performs textguided synthesis to augment rare thyroid cancer subtypes but requires cancer samples to train the generative model. This dependency restricts applicability to diseases like fetal CHDs, where positive cases are scarce and infeasible to collect. Therefore, Pos4CHD targets subtype-specific fetal CHD synthesis, characterized by structural anomalies, using only normal samples. Combining rule-based mask manipulation and maskguided ultrasound synthesis, Pos4CHD enables fine-grained modeling and synthesis of heterogeneous CHD subtypes with verifiable semantic masks for clinical applications.

## III. METHODOLOGY

## A. Preliminaries

1) Diffusion Models: The diffusion probabilistic model has been the primary choice for image generation nowadays. It defines a Markov chain, where images are first corrupted by a series of Gaussian noise (forward process), and then it samples synthesized images from a learned distribution by starting from random Gaussian noise (backward process). Supposing a sample $\pmb { x } _ { 0 } \sim q ( \pmb { x } )$ , the forward diffusion process adds increasing gaussian noise ϵ to the clean data $\scriptstyle { \pmb x } _ { 0 }$ according to timestep $t \in T$ , which is defined as $q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) { = } \mathcal { N } ( \pmb { x } _ { t } ; \alpha _ { t } \pmb { x } _ { t - 1 } , \beta _ { t } \pmb { I } )$ where $\alpha _ { t } , \beta _ { t } \ \in \ [ 0 , 1 ]$ denote the noise schedule for image corruption and satisfy $\alpha _ { t } ^ { 2 } + \beta _ { t } ^ { 2 } { = } 1$ for variance preservation. This process can be written and re-parameterized as onestep noising, $\mathbf { \Delta x } _ { t } = \sqrt { \overline { { \alpha } } _ { t } } \mathbf { \Delta x _ { 0 } } + \sqrt { 1 - \overline { { \alpha } } _ { t } } \epsilon , \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I )$ , where $\scriptstyle { \overline { { \alpha } } } _ { t } = \prod _ { 1 } ^ { t } \alpha _ { t }$ . The backward sampling process starts from a random Gaussian noise x , which is reversed back towards the original data distribution $q ( { \pmb x } )$ via iterative denoising, formatted as $\begin{array} { r } { p _ { \theta } ( { \pmb x } _ { 0 : T } ) { = } p ( { \pmb x } _ { T } ) \prod _ { t = 1 } ^ { l ^ { \prime } } p _ { \theta } ( { \pmb x } _ { t - 1 } \vert { \pmb x } _ { t } ) } \end{array}$ , where $p _ { \theta } ( { \pmb x } _ { t - 1 } | { \pmb x } _ { t } ) { = } \mathcal { N } ( { \pmb x } _ { t - 1 } ; \mu _ { \theta } ( { \pmb x } _ { t } , t ) , \beta _ { t } I )$ . Since the exact calculation of $\hat { \mu } _ { \boldsymbol { \theta } }$ is intractable, it is approximated by a network $\hat { \epsilon } _ { \theta } ( x _ { t } , t )$ , optimized by ELBO, and formatted as,

$$
\hat { \mu } _ { \boldsymbol { \theta } } ( { \pmb x } _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } ( { \pmb x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \overline { { \alpha } } _ { t } } } \hat { \epsilon } _ { \boldsymbol { \theta } } ( { \pmb x } _ { t } , t ) ) .\tag{1}
$$

Therefore, the final training objective can be further simplified as a denoising target, namely ϵ-prediction, formatted as,

$$
\begin{array} { r } { \mathcal { L } _ { d e n o i s e } = \mathbb { E } _ { \pmb { x } _ { 0 } , t , \epsilon } | | \epsilon - \hat { \epsilon } _ { \theta } ( \pmb { x } _ { t } , t ) | | _ { 2 } ^ { 2 } . } \end{array}\tag{2}
$$

To reduce the computational cost, SD [14] moves the diffusion process to a latent space constructed by VAE, where the image latent $\scriptstyle { \pmb { x } } = { \pmb { \mathcal { E } } } ( I )$ and the reconstructed image $\scriptstyle { \hat { I } } = { \mathcal { D } } ( { \boldsymbol { \mathbf { x } } } )$ where an encoder $\mathcal { E } ( \cdot )$ and decoder $\mathcal { D } ( \cdot )$ are shown in Fig. 4, namely Latent Diffusion Models. This paradigm has achieved impressive results in natural image generation tasks, and researchers have explored its applications in medical image generation for the long-standing issue of data scarcity.

2) Spatially Controllable Image Generation: The representative ControlNet [15] introduces adapters to SD to enable taskspecific spatial control over image synthesis, e.g., edges, semantic masks, and depths, which shows impressive results on customized image generation and application. Supposing one encoder block $F ( \cdot ; \theta )$ of SD, it accepts the input feature x and output ${ \pmb y } { = } F ( { \pmb x } ; \theta )$ . ControlNet makes a trainable copy $F ( \cdot ; \theta ^ { \prime } )$ to inject condition c into the model with zero-initialized convolutions $Z ( \cdot ; w )$ for stability, which is formulated $\mathbf { a s } ,$

$$
{ \pmb y } _ { c } = F ( { \pmb x } ; { \pmb \theta } ) + \lambda * Z ( F ( { \pmb x } + Z ( { \pmb c } ; w _ { 1 } ) ; { \pmb \theta } ^ { \prime } ) ; w _ { 2 } ) ,\tag{3}
$$

where $_ { { \pmb { y } } _ { c } }$ is the output feature modulated by spatial conditions. The conditioning scale λ is introduced to adjust control strength during inference. The training objective for conditional generative models is formatted as,

$$
\mathcal { L } _ { d e n o i s e } = \mathbb { E } _ { { \mathbf { x } } _ { 0 } , t , { \mathbf { c } } , { \epsilon } } | | { \epsilon } - \hat { { \epsilon } } _ { \boldsymbol { \theta } } ( { \mathbf { x } } _ { t } , t , { \mathbf { c } } ) | | _ { 2 } ^ { 2 } .\tag{4}
$$

![](images/583986f9b5747e1491a8e57cc681224c168011a78c43a474d17684d8843c783d.jpg)  
Fig. 2. Overview of our Pos4CHD framework, which synthesizes counterfactual positive samples for fetal CHD analysis. It comprises five steps: (1) mask collection from normal samples, (2) rule-based counterfactual mask generation, (3) pseudo-positive image synthesis with our proposed FE-Gen, (4) quality check and data filtering, and (5) downstream utilization including classification, detection, and segmentation.

## B. Pos4CHD: A Positive-free Mask-to-image Counterfactual Generation Framework for Fetal CHD Analysis

Given the scarcity or absence of fetal CHD cases, directly learning disease-specific appearance and structures from positives is infeasible. Thus, we formulate CHD synthesis as maskto-image counterfactual generation, where semantic masks explicitly encode CHD-related anatomical changes and guide the synthesis of positive samples. This scheme supports generation of diverse CHD subtypes within a unified framework.

Our design is motivated by three observations: (1) Most CHDs are presented as cardiac structural anomalies. Semantic masks therefore provide an explicit and interpretable representation of cardiac anatomy and morphology, as a unified proxy for CHD analysis; (2) Semantic masks provide dense, finegrained spatial control and can be manipulated easily, flexibly, and reliably, which enables CHD simulation with anatomical consistency according to clinical priors; (3) Counterfactual masks offer explicit references for verifying the anatomical correctness of synthesized samples while supplying pixel-level supervision for downstream tasks at no extra annotation cost.

The pipeline of Pos4CHD is shown in Fig. 2, which consists of five steps: (1) normal mask collection, (2) counterfactual mask manipulation, (3) pseudo-positive image synthesis, (4) quality check and data filtering, and (5) downstream utilization. The implementation of each step is depicted as follows.

We begin with a set of normal fetal cardiac masks, denoted as $\mathbf { \bar { \mathcal { M } } } = \{ \mathbf { m } _ { i } \} _ { i = 1 } ^ { N }$ , obtained through manual annotation, pretrained segmentation models, or atlas animations. These masks serve as the basis for counterfactual sample synthesis. Given a target fetal CHD subtype c∈C, we first design morphological transformation rules $R ( \cdot , c ; \psi _ { c } )$ based on clinical prior knowledge, thereby simulating plausible counterfactual masks with subtype-specific characteristics through operations like translations, editing, free-form deformation, etc., where $\psi _ { c }$ denotes category-specific transformation hyperparameters. Specifically, this paper focuses on three representative heterogeneous subtypes, i.e., EbA, TAPVC, and VSD. Their key morphological properties [41] and corresponding manipulation rules are summarized in Fig. 2. Next, we apply customized transformation $R ( \cdot , c ; \psi _ { c } )$ to the normal mask m, yielding counterfactual cardiac mask m. This clinical-knowledgedriven counterfactual mask manipulation is formulated as,

$$
\begin{array} { r } { \overline { { \bf m } } = R ( { \bf m } , c ; \psi _ { c } ) . } \end{array}\tag{5}
$$

Given the counterfactual mask m, we employ a maskguided conditional generative model $\mathcal { G } ( \boldsymbol { \epsilon } , \mathbf { m } )$ to generate spatially aligned, realistic fetal echocardiographic images, where the synthesis of the pseudo-positive image I is formulated as,

$$
\overline { { I } } = \mathcal { G } ( \epsilon , \overline { { \mathbf { m } } } ) , \ \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) ,\tag{6}
$$

where we employ a customized generator, FE-Gen (see Sec. III-C), for advanced image fidelity and image-mask alignment for fetal ultrasound synthesis. Following the pipeline in Fig. 2, we can produce diverse candidate pseudo-positive image-mask pairs (I, m) for CHD subtypes, owing to the randomness of mask manipulation, augmentations, and image generation.

Then, we inspect and filter low-quality synthesized samples (I, m) manually or with pretrained segmentation models to construct a high-quality synthetic dataset $\overline { { \mathcal { D } } } { = } \{ ( \overline { { I } } _ { i } , \overline { { \mathbf { m } } } _ { i } ) \} _ { i = 1 } ^ { N ^ { \prime } }$ of pseudo-positives with pixel-level annotations. This is because the stochastic nature of diffusion sampling may introduce artifacts [42] or unfaithful CHD samples, particularly in medical scenarios requiring precise anatomical fidelity, as validated in Sec. IV-C.4. The resulting image-mask pairs can be readily converted into class labels or bounding boxes, supporting diverse real-world CHD analysis tasks.

## C. FE-Gen: A Mask-guided Controllable Generation Model for Fetal Echocardiography

As discussed above, Pos4CHD enables CHD synthesis usingon<sup>d</sup> n<sup>c</sup> only negative cases, but its performance relies on the quality of mask-to-image generator $\mathcal { G } ( \cdot )$ , where two key factors matter: (1) image fidelity, which decides how closely the synthesized fetal echocardiographic images match the real ones in visual appearance and data distribution; and (2) spatial alignment between each synthesized image and conditioning mask, which determines the faithfulness of CHD-specific characteristics and mask accuracy. Poor spatial alignment would introduce noisy<sub>i</sub>n<sup>g</sup> labels, thereby degrading downstream performance and data<sub>e</sub>n<sup>d</sup> efficiency. Therefore, an effective generator should produce realistic fetal heart ultrasound with precise image-mask alignment. The implementation details are as follows.

1) Data Collection: Due to the lack of public fetal echocardiography data, we collect an in-house dataset of fetal heart ultrasound from the hospital, namely WHU-4CH-8k. This dataset contains 7, 974 fetal heart 4CH views from 2, 855 patients acquired between 2017 and 2023, mostly ranging from 21-23 and 28-30 gestational weeks, after plane quality checks [43]. This unlabeled dataset is utilized to train the fetal ultrasound generator. We further select 517 high-quality samples and annotate 14 key anatomical structures with clinical experts, namely WHU-4CH-Seg517. As shown in Fig. 3, the rich semantic annotations on 4CH views provide anatomical and spatial guidance to the generator, supporting simulation of CHD subtypes associated with these structures. More details, including data splits and resolution, are provided in Sec. IV-A.

2) FE-Gen Architecture: Following recent practices [8], [12], [18], a straightforward approach is to adapt SD for fetal ultrasound synthesis. However, we notice two limitations during implementation: (1) Histogram distribution drift: gen-� = { � , � , � }� Positions� =Descriptions{� , � , ···, � } erated images tend to be low-contrast and blurry, with his-Spatial Pose Embedding(a) togram distributions deviating from real ultrasound as showno<sup>n</sup> n<sup>gKeypoint</sup> <sup>Random</sup> g in Fig. 6, which remains under-explored in previous literature;n<sup>d</sup> b<sup>eL</sup> <sub>n</sub><sup>de</sup> (2) Sub-optimal image-mask alignment: Recent studies [18]′�(�) neglect quantitative measurement of spatial alignment betweenSkeleton Embedding�� � generated images and conditions, and conventional binary orTokenizer <sup>Latent</sup> RGB embeddings for spatial guidance may interfere withPrompt“ ··· ”<�1><�2> <��> anatomy semantics and degrade spatial control precision. To address these issues, we propose FE-Gen, which is tailored forText Embedding ···�\*1 �\*2 �\*N high-fidelity and spatially consistent fetal ultrasound synthesis <sup>Text</sup> Text with two key designs: 1) learnable semantic queries, and 2) histogram distribution regularization. Details are as follows.<sup>Keypoint</sup> <sup>Tokens</sup>

Learnable semantic queries. Considering plain color em-<sup>></sup> beddings lack explicit semantic meaning and may cause ambiguity, particularly between categories with similar colors, we instead utilize learnable semantic queries inspired by SP-

![](images/e4dadafeaf7992053d47514e2b2ac4295f376d35d5a8888b0e7fd6098b7b3e47.jpg)  
Fig. 3. Example fetal cardiac 4CH view with 14 anatomical annotations.

![](images/acbdfa546cbd4b3a51af2769080e5df4c76f70b1b2f1a344f35046225136d469.jpg)  
Fig. 4. Architecture of FE-Gen: (a) Learnable semantic queries for spatial alignment, and (b) Histogram regularization for ultrasound fidelity.

Ctrl [44]. As shown in Fig. 4(a), each anatomical category $k \in$ K is initialized with a fixed random vector $e _ { k } ^ { 0 } \in \mathbb { R } ^ { 1 \times \bar { C } }$ , where C is the embedding dimension. An encoding module $\mathbb { E } ( \cdot ; \phi )$ which consists three stacked linear blocks parameterized by ϕ, transforms it into a learnable semantic query $\boldsymbol { e } _ { k } = \mathbb { E } ( e _ { k } ^ { 0 } ; \phi )$ $\boldsymbol { e } _ { k } \in \mathbb { R } ^ { 1 \times C }$ . The condition mask $\mathcal { M } \ \bar { \in } \ \bar { \mathbb { R } ^ { H \times W \times C } }$ is then constructed by assigning the learned queries $\{ e _ { k } \} _ { k = 1 } ^ { K }$ to each pixel according to category-index mask $\mathcal { M } _ { i d x } \ = \ \{ m _ { i , i } ^ { \prime } \ =$ $k \} _ { H \times W \times 1 }$ , formulated as, $\mathcal { M } = \{ m _ { i , j } = e _ { k } | m _ { i , j } ^ { \prime } = k , m _ { i , j } ^ { \prime } \in$ $\mathcal { M } _ { i d x } \} _ { H \times W \times C }$ . These semantic queries $\{ e _ { k } \} _ { k = 1 } ^ { K }$ are learned rolNet <sub>adaptively during the training of the diffusion model.</sub>

oss ps <�1> <�2> <��>Histogram distribution regularization. To mitigate the hisnv nv nv nv nv z<sup>ed</sup> io<sup>n</sup>togram drift as shown in Fig. 5 and Fig. 6 when adapting SD Z<sup>er</sup> Z<sup>er</sup> Ze<sup>r</sup> Z<sup>er</sup> Z<sup>er</sup> V<sup>is</sup> A<sup>t</sup>and ControlNet for ultrasound synthesis, we introduce an L1- o<sup>in</sup> m<sup>a</sup>based sorted loss for distribution regularization. Specifically, <sup>K H</sup> � � �since diffusion models generate images through iterative denoising, we first use a DDIM scheduler [45] to estimate the clean image latent $\scriptstyle { \hat { \mathbf { x } } } _ { 0 }$ <sup>ht</sup>from the current noisy latent $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and le predicted noise $\hat { \epsilon } _ { \theta }$ at timestep t, which is formulated as,

$$
\hat { \mathbf { x } } _ { 0 } = \frac { \mathbf { x } _ { t } - \sqrt { 1 - \overline { { \alpha } } _ { t } } \hat { \epsilon } _ { \theta } ( \mathbf { x } _ { t } , t ) } { \sqrt { \overline { { \alpha } } _ { t } } } ,\tag{7}
$$

where $\hat { \epsilon } _ { \theta } ( x _ { t } , t )$ is the predicted noise at timestep $t ,$ and $\alpha _ { t }$ is decided by the noise scheduler. Then, the histogram regularization $\mathcal { L } _ { h i s t }$ is calculated by a sorted L1 loss between the clean latent $\scriptstyle { \mathbf { { \vec { x } } } } _ { \mathbf { 0 } }$ and the estimated ${ \hat { \mathbf { x } } } _ { 0 }$ , formulated as,

$$
\mathcal { L } _ { h i s t } = \frac { 1 } { N } \sum _ { i = 1 } \lvert \hat { x } _ { 0 } ^ { ( i ) } - x _ { 0 } ^ { ( i ) } \rvert , ~ x ^ { ( i ) } \in \mathrm { s o r t } ( x )\tag{8}
$$

where $\operatorname { s o r t } ( \pmb { x } ) = [ x ^ { ( 1 ) } , \cdots , x ^ { ( N ) } ] , x ^ { ( i ) }$ denotes sorted pixel values and $x ^ { ( i ) } \leq x ^ { ( j ) }$ , if $i \leq j$ . Validations see Sec. IV-C.

3) Multi-step Training: We employ a three-step fine-tuning strategy to adapt SD to fetal heart ultrasound synthesis, including: (1) SD-VAE adaptation, (2) SD-UNet fine-tuning, and (3) FE-Gen training, inspired by existing adaptation paradigms [8], [12]. Fig. 4 shows the whole training process.

Step-1: SD-VAE Adaptation. For latent diffusion models, the reconstruction quality of VAE decides the upper bound for image fidelity. SD-VAE, pretrained on natural images, is underrepresented for fetal echocardiography and loses fine details, consistent with observations [46] when adapting SD to specific domain. Therefore, we fine-tune VAE decoder on WHU-4CH-8k with the reconstruction loss $L _ { r e c }$ with MSE, perceptual loss $L _ { p }$ with LPIPS, and GAN loss $L _ { G A N }$ , where

$$
\mathcal { L } _ { V A E } = L _ { r e c } + \lambda _ { 1 } L _ { p } + \lambda _ { 2 } L _ { G A N } ,\tag{9}
$$

where loss weights $\{ \lambda _ { i } \} _ { i = 1 } ^ { 2 }$ set empirically following [14]. We freeze the VAE encoder to keep the latent space unchanged. Table V verifies this adaptation with improved performance.

Step-2: SD-UNet Fine-tuning. Since SD, pretrained on natural images, cannot directly produce reasonable and realistic fetal echocardiography and exhibits large distribution gaps, we fine-tune SD-UNet with LoRA [47] on WHU-4CH-8k for ultrasound synthesis, using a combination of denoising loss and histogram regularization, formulated as,

$$
\begin{array} { r } { \mathcal { L } _ { U N e t } = \mathcal { L } _ { d e n o i s e } + \omega _ { h } \cdot \mathcal { L } _ { h i s t } , } \end{array}\tag{10}
$$

where $\omega _ { h }$ denotes the regularization weight. We use a fixed text prompt $\mathbf { c } _ { t }$ as “A realistic ultrasound image, fetal cardiac, standard four-chamber view.”. Table V validates the effectiveness of $\mathcal { L } _ { h i s t }$ in image fidelity. The fine-tuned SD can generate fetal ultrasound images that better match the real data distribution, serving as the basis for controllable generation.

Step-3: FE-Gen Training. We then train FE-Gen on imagemask pairs from WHU-4CH-Seg517, initialized with the finetuned SD-VAE and SD-UNet, to realize mask-guided fetal heart ultrasound generation, optimized with the same objective as Eq. 10. The embedding module $\mathbb { E } ( \cdot ; \phi )$ and learnable semantic queries $\{ \mathbf { e } _ { k } \} _ { k = 1 } ^ { K }$ are jointly optimized during the training. Empirically, jointly fine-tuning SD-UNet LoRA with FE-Gen yields better performance. Sec. IV-C further investigates the effects of training data scale, classifier-free guidance, and conditioning scale, providing practical guidelines for model configuration. Notably, FE-Gen introduces no additional inference cost compared with the baseline ControlNet.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Datasets: Due to the lack of publicly available datasets for fetal heart ultrasound, we collect two in-house datasets with clinician assistance as described in Sec. III-C.1, including: 1) WHU-4CH-8k, split into 6,466 training and 1,508 test images for fine-tuning SD-VAE and SD-UNet, and 2) WHU-4CH-Seg517, with 14 anatomical annotations, split into 417 training and 100 test images for training FE-Gen.

To evaluate synthesized data on real-world CHD analysis tasks, we select three heterogeneous CHD subtypes—EbA, TAPVC, and VSD—with different incidence rates and prognoses as representatives. We collect extra real positive cases for each subtype with the assistance of experts and consider three distinct settings for evaluation, including:

(1) Zero-shot classification: For rare CHDs with poor prognoses and insufficient training positives, we construct two realworld test sets for EbA (0.53% of CHDs) and TAPVC (1.50% of CHDs) referring to clinical reports. CLS-EbA-test84 and CLS-TAPVC-test80 consist of 42/40 normal and 42/40 positive cases from 5 and 7 patients separately. It represents scenarios where no real positives are available for training.

(2) Lesion detection: For more prevalent CHD subtypes like VSD (35.57% of CHDs), where a few positives can be collected with effort, we construct DET-VSD-908, with a training set (760 negative/ 28 positive) and a test set (60 negative/ 60 positive) following [48], to evaluate the effectiveness of synthesized samples under severe class imbalance scenarios.

(3) Anatomy segmentation: We evaluate synthesized normal samples for fetal cardiac anatomy segmentation using WHU-4CH-Seg517, where predicted masks can be used for biometric assessment and indicate possible pathologies. To avoid data leakage, data synthesis and augmentation use only automatically detected semantic masks from the DET-VSD-908 training set. No test images or masks are used to train generative models to ensure fair evaluations.

2) Metrics: We evaluate fetal echocardiography generation from three aspects: (1) Reconstruction: PSNR and SSIM for VAE reconstruction quality; (2) Image fidelity: FID and KID measure feature distributions, LPIPS evaluate perceptual similarity, and KL divergence, Skewness, and Standard Deviation quantify histogram distribution differences. We further compute CLIP-Score using pretrained FetalCLIP [49] with fixed prompts to assess image-text distribution; (3) Image-mask alignment: Dice Similarity Coefficient (DSC) of synthesized images, estimated by an nnUNet model pre-trained on the WHU-4CH-Seg517 training set. For downstream tasks, we use standard metrics: Accuracy, Recall, Precision, AUC for classification; mAP50 for detection; Precision, Recall, DSC, mIoU for segmentation. Each experiment is repeated three times or with 5-fold cross-validation for robust evaluation.

3) Implementation: All experiments are conducted with Py-Torch on four NVIDIA 3090 GPUs. FE-Gen is built on SD v1.5 and ControlNet. For SD-VAE adaptation, we fine-tune the decoder for 20 epochs with a batch size of 4 and a learning rate of 1e−5. For SD-UNet adaptation, we adopt a LoRA rank of 8 for fine-tuning 2,000 epochs with a learning rate of 1e−4 and a batch size of 16. The noise offset is set to 0.1 [50] to alleviate contrast flaws, and we randomly drop 30% prompts to support classifier-free guidance (CFG). For FE-Gen, ControlNet is initialized from the UNet encoder and trained for 2,000 epochs with a batch size of 16 and a learning rate of 1e−5. The regularization weight $\omega _ { h }$ is set to 0.1 (see Sec. IV-D.1). During inference, we adopt CFG=2.0 and conditioning scale λ=1.0 (see Sec. IV-D.3). For downstream tasks, we use ConvNeXt for zero-shot classification, Yolo-v5s for lesion detection, and nnUNet for fetal heart anatomy segmentation following [48], [51] with default settings. Sec. IV-B.2 provides more details. All experiments are conducted with an image resolution of 512×512 unless otherwise specified.

## B. Experimental Results

1) Image Generation: We compare FE-Gen with representative image synthesis methods in Table I: (1) SD-adaptation methods, i.e., DreamBooth [52] and LoRA [47], and (2) spatially controllable generators, i.e., T2I-Adapter [53] and ControlNet [15]. Our baseline is ControlNet (CtrlNet). Notably, text-guided-only methods lack explicit spatial control and cannot follow masks. Our method achieves competitive FID DreamBoothand the lowest LPIPS, and surpasses CtrlNet by 3.96% DSC, T2I-Adapterdemonstrating superior image-mask alignment for subsequent Ourscounterfactual synthesis. Our method also achieves the lowest m<sup>-</sup>KL divergence, while others exhibit low standard deviations Eor large skewness gaps, suggesting reduced image contrast. t<sup>-</sup>Qualitative comparisons in Fig. 5, histogram distributions in Fig. 6, and FetalCLIP [49] feature distributions in Fig. 7 show that our synthesized data exhibit the closest distribution to real ones. Furthermore, Fig. 8 reports the user study results with 6 clinicians with an average 12-year experience in fetal ultrasound. For 20 real and 20 synthesized images of each type (Normal, VSD, EbA, and TAPVC), clinicians are asked to classify them as real or synthesized independently. The overall average accuracy is 46.4%, comparable to the 50% chance level, indicating it is hard to distinguish synthesized samples from real data, reflecting our high perceptual fidelity.

TABLE I  
COMPARISON OF FETAL ULTRASOUND SYNTHESIS METHODS. THE BEST AND SECOND-BEST RESULTS ARE DENOTED BY bold AND UNDERLINE.50152.89 <sup>75.38</sup> C <sub>I</sub>DData Scale Data Scale Data Scalegram Distribution <sub>Real</sub> <sub>Images</sub>
<table><tr><td rowspan="2">Model</td><td rowspan="2">Condition</td><td colspan="8">Metrics</td></tr><tr><td>KL↓</td><td>Skewness</td><td>STD</td><td>LPIPS↓</td><td>KID↓</td><td>FID↓</td><td>CLIPScore</td><td>DSC↑</td></tr><tr><td>Real Data</td><td></td><td>0.0000</td><td>0.64</td><td>50.89</td><td>0.0000</td><td>0.0000</td><td>0.00</td><td>35.25</td><td>80.75</td></tr><tr><td>DreamBooth</td><td>text</td><td>0.4181</td><td>0.07</td><td>49.81</td><td>0.5742</td><td>0.0822</td><td>170.71</td><td>38.82</td><td>11.58</td></tr><tr><td>SD-LoRA</td><td>text</td><td>0.1497</td><td>0.81</td><td>40.26</td><td>0.5727</td><td>0.0382</td><td>147.42</td><td>31.24</td><td>10.72</td></tr><tr><td>T2I-Adapter</td><td>mask</td><td>0.1169</td><td>1.04</td><td>40.26</td><td>0.4962</td><td>0.0911</td><td>180.57</td><td>36.69</td><td>72.05</td></tr><tr><td>ControlNet</td><td>mask</td><td>0.3047</td><td>0.49</td><td>35.76</td><td>0.4570</td><td>0.0575</td><td>161.10</td><td>37.64</td><td>75.68</td></tr><tr><td>Ours</td><td>mask</td><td>0.0350</td><td>0.54</td><td>46.32</td><td>0.4347</td><td>0.0423</td><td>148.13</td><td>36.83</td><td>79.52</td></tr></table>

DreamBooth  
SD-LoRA  
T2I-Adapter w overlapped mask  
ControlNet w overlapped mask  
Ours w overlapped mask  
![](images/201db33cccd10af9bf8b647e07efe7ea84b82039b46acecae2c73ef83e467b42.jpg)  
Fig. 5. Comparisons of fetal heart ultrasound images generated by different methods with overlapped semantic masks for visualization.<sup>SD-LoRA</sup>

![](images/8b8fcc27e439b70e46e90c16c0b6aa28ebd4e24869d3eec0e2ddec1b4c1b48c5.jpg)  
Fig. 6. Histogram distribution comparison of synthesized images.

![](images/f71d2d652b77b651dc883b7dddd71c7dafb3cf9f088bb19e5ae10f3a75aa587a.jpg)  
Fig. 7. Data distribution visualization via t-SNE with FetalCLIP.

![](images/f4a430e69c567dced3016a00e4ff8b166e078ebb3dfa274da8a0955d9fc208f6.jpg)  
Fig. 8. Expert evaluation results on real or fake classification with real and our synthesized samples .

2) Downstream Tasks: We evaluate synthesized samples on three CHD analysis tasks: zero-shot EbA/TAPVC classification, VSD detection, and fetal cardiac anatomy segmentation.

Zero-shot classification. We synthesize EbA and TAPVC samples following Fig. 2 and train ConvNeXt for 200 epochs on 200 positives (Syn.) and 200 negatives (Real or Syn.), and evaluate on CLS-EbA-test84 and CLS-TAPVC-test80. As listed in Table II, models trained entirely on synthesized data generalize well to real test images and even outperform those trained with real negatives, possibly because models overfit low-level features for distinguishing real and synthetic data rather than CHD-specific features. We also outperform CtrlNet, achieving 93.25% ACC on EbA and 72.92% ACC on TAPVC, owing to superior fidelity and spatial alignment.

Lesion detection. For VSD detection, we generate pseudopositives from masks of the WHU-4CH-Seg517 training set, apply mask deformation, and filter low-quality samples, yielding ∼276 synthesized positives for augmentation. We train Yolo-v5s for 250 epochs. As shown in Table III, Pos4CHD improves the average mAP50 by 6.95% over the real data, also 3.41% higher than CtrlNet. FE-Gen retains 66.2% (276/417) generated samples after quality control, much higher than the 26.4% of CtrlNet, exhibiting better data efficiency.

TABLE II  
ZERO-SHOT CLASSIFICATION FOR EBA AND ${ \mathsf { T A P V C } } .$ REAL AND SYN DENOTE REAL AND SYNTHESIZED DATA. Bold FOR THE BEST RESULT.
<table><tr><td rowspan="2">Model</td><td colspan="2">Train Set</td><td rowspan="2">ACC(%)</td><td rowspan="2">P(%)</td><td rowspan="2">R(%)</td><td rowspan="2">AUC(%)</td></tr><tr><td>Neg.</td><td>Pos.</td></tr><tr><td colspan="8">EbA Classification</td></tr><tr><td rowspan="2">CtrlNet</td><td>Real</td><td>Syn</td><td> $\overline { { 5 8 . 7 3 \pm 1 . 8 2 } }$ </td><td> $\overline { { 8 8 . 8 1 \pm 9 . 7 9 } }$ </td><td> $2 0 . 6 4 \pm 5 . 9 9$ </td><td> $\overline { { 8 3 . 8 6 \pm 9 . 8 4 } }$ </td></tr><tr><td>Syn</td><td>Syn</td><td> $9 1 . 2 7 { \pm } 1 . 8 2$ </td><td> $9 4 . 1 1 \pm 1 . 0 9$ </td><td> $8 8 . 0 9 { \scriptstyle \pm 4 . 7 6 }$ </td><td> $\mathbf { 9 6 . 0 9 } \pm 2 . 6 0$ </td></tr><tr><td rowspan="2">Ours</td><td>Real</td><td>Syn</td><td> $7 1 . 4 3 \pm 1 . 1 9$  </td><td> $\overline { { { \bf 1 0 0 . 0 0 } _ { \pm 0 . 0 0 } } }$ </td><td> $4 2 . 8 6 \pm 2 . 3 8$ </td><td> $8 3 . 7 3 { \scriptstyle \pm 8 . 7 8 }$ </td></tr><tr><td>Syn</td><td>Syn</td><td> $\mathbf { 9 3 . 2 5 } \pm 0 . 6 9$ </td><td> $9 3 . 6 1 \pm 1 . 3 0$ </td><td> $\mathbf { 9 } 2 . 6 6 \pm 0 . 1 7$ </td><td> $9 5 . 0 3 { \scriptstyle \pm 0 . 9 2 }$ </td></tr><tr><td colspan="7"> $\overline { { \mathrm { { T A P V C ~ C l a s s i f i c a t i o n } } } }$ </td></tr><tr><td rowspan="2">CtrlNet</td><td>Real</td><td>Syn</td><td>53.33±2.60</td><td> $9 5 . 2 4 \pm 8 . 2 5$ </td><td> $\overline { { 7 . 5 \pm 6 . 6 1 } }$ </td><td> $7 5 . 1 2 \pm 9 . 3 2$ </td></tr><tr><td>Syn</td><td>Syn</td><td> $6 7 . 0 8 \pm 5 . 6 4$ </td><td> $7 3 . 3 9 \pm 3 . 6 9$ </td><td> $5 3 . 3 3 \pm 1 5 . 2 7$ </td><td> $7 2 . 7 9 \pm 4 . 3 4$ </td></tr><tr><td rowspan="2">Ours</td><td>Real</td><td>Syn</td><td> $6 2 . 9 2 \pm 1 . 4 4$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm \mathrm { 0 . 0 0 }$ </td><td> $2 5 . 8 3 \pm 2 . 8 8$ </td><td> $7 4 . 1 1 \pm 2 . 7 4$ </td></tr><tr><td>Syn</td><td>Syn</td><td> $7 2 . 9 2 { \scriptstyle \pm 2 . 6 0 }$ </td><td> $7 0 . 2 7 \pm 3 . 7 1$ </td><td> $\mathbf { 8 0 . 0 0 } \pm 2 . 5 0$ </td><td> $7 7 . 5 4 \pm 5 . 8 9$ </td></tr></table>

TABLE III

RESULTS OF VSD DETECTION. RE A L AND SY N DENOTE REAL AND SYNTHESIZED DATA. Bold FOR THE BEST RESULT.
<table><tr><td rowspan="2">Model</td><td colspan="2">Train Set</td><td rowspan="2">AVG</td><td rowspan="2">mAP50(%) IVS</td><td rowspan="2">VSD</td></tr><tr><td>Neg</td><td>Pos</td></tr><tr><td></td><td>Real</td><td>Real</td><td> $7 1 . 4 3 \pm 0 . 4 0$ </td><td> $\overline { { 6 9 . 6 0 \pm 2 . 8 2 } }$ </td><td> $\overline { { 7 3 . 3 0 \pm 2 . 9 6 } }$ </td></tr><tr><td>CtrlNet</td><td>Real</td><td> $\overline { { \mathrm { R e a l } + S \mathrm { y n } } }$ </td><td> $\overline { { 7 4 . 9 7 \pm 0 . 6 5 } }$ </td><td> $7 3 . 5 5 { \scriptstyle \pm 2 . 9 2 }$ </td><td> $7 6 . 3 2 \pm 2 . 2 1$ </td></tr><tr><td>Ours</td><td>Real</td><td>Real+Syn</td><td> $\mathbf { 7 8 . 3 8 \bot 0 . 5 0 }$  </td><td> $7 5 . 6 8 \pm 2 . 2 5$  </td><td> $\mathbf { 8 0 . 4 3 \substack { \pm 1 . 8 5 } }$ </td></tr></table>

TABLE IV

FETAL HEART ANATOMY SEGMENTATION RESULTS OF DIFFERENT DATA
<table><tr><td>Scale</td><td>Aug</td><td>P(%)</td><td>R(%)</td><td>DSC(%)</td><td>mIoU(%)</td></tr><tr><td rowspan="2">0</td><td>Baseline</td><td></td><td></td><td></td><td></td></tr><tr><td>CtrlNet</td><td> $7 5 . 3 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $7 3 . 9 3 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $7 3 . 6 7 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $6 0 . 0 2 { \scriptstyle \pm 0 . 5 4 }$ </td></tr><tr><td rowspan="2"></td><td>Ours</td><td> $\mathbf { 7 6 . 9 1 } \pm \mathbf { 0 . 8 0 }$ </td><td> ${ \bf 7 6 . 7 2 { \scriptstyle \pm 0 . 5 8 } }$ </td><td> ${ \bf 7 5 . 9 4 } _ { \pm 0 . 3 2 }$ </td><td> ${ \bf 6 3 . 1 5 } _ { \pm 0 . 4 8 }$ </td></tr><tr><td>Baseline</td><td> $\overline { { 7 1 . 0 1 \pm 1 . 9 7 } }$ </td><td> $\overline { { 6 8 . 9 9 \pm 1 . 4 6 } }$ </td><td> $\overline { { 6 6 . 9 4 \pm 1 . 6 4 } }$ </td><td> $\overline { { 5 4 . 4 3 \pm 1 . 8 9 } }$ </td></tr><tr><td rowspan="2">20</td><td>CtrlNet</td><td> $7 7 . 9 0 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $7 6 . 6 2 \pm 0 . 1 8$ </td><td> $7 6 . 1 5 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $6 3 . 5 1 \pm 0 . 2 4$ </td></tr><tr><td>Ours</td><td> $\mathbf { 7 9 . 1 5 } \pm \mathbf { 0 . 2 2 }$ </td><td> ${ \bf 7 8 . 0 6 } { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 7 . 4 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> ${ \bf 6 5 . 4 3 } \pm 0 . 3 8$  1</td></tr><tr><td rowspan="2">50</td><td>Baseline</td><td> $7 7 . 9 3 \pm 0 . 4 1$ </td><td> $7 5 . 3 7 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $\overline { { 7 4 . 6 1 \pm 0 . 6 8 } }$ </td><td> $\overline { { 6 2 . 9 1 \pm 0 . 8 0 } }$ </td></tr><tr><td>CtrlNet</td><td> $7 9 . 5 8 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $7 8 . 5 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $7 8 . 1 2 \pm 0 . 2 5$ </td><td> $6 6 . 1 9 { \scriptstyle \pm 0 . 2 9 }$ </td></tr><tr><td rowspan="2"></td><td>Ours</td><td> $\mathbf { 7 9 . 9 3 } \pm \mathrm { 0 . 3 3 }$ </td><td> $\mathbf { 7 9 . 3 2 } \pm \mathrm { 0 . 2 6 }$ </td><td> ${ 7 8 . 4 8 \pm 0 . 2 7 }$ </td><td> ${ \bf 6 6 . 8 8 } \pm 0 . 3 4$  </td></tr><tr><td>Baseline</td><td> $\overline { { 8 0 . 8 4 \pm 0 . 1 8 } }$ </td><td> $7 8 . 8 9 \pm 0 . 2 3$ </td><td> $7 8 . 5 9 \pm 0 . 1 4$ </td><td> $6 7 . 3 8 \pm 0 . 2 7$ </td></tr><tr><td rowspan="2">100</td><td>CtrlNet</td><td> $8 0 . 5 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> ${ \bf 8 0 . 0 7 } \pm 0 . 2 6$ </td><td> $7 9 . 3 8 { \pm } 0 . 0 8$ </td><td> $6 7 . 9 8 \pm 0 . 1 1$ </td></tr><tr><td>Ours</td><td> ${ \bf 8 1 . 0 2 } _ { \pm 0 . 1 6 }$ </td><td> $7 9 . 9 9 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $\mathbf { 7 9 . 4 3 } \pm \mathrm { 0 . 2 7 }$ </td><td> ${ \bf 6 8 . 2 1 } \pm 0 . 3 1$  </td></tr><tr><td rowspan="2">200</td><td>Baseline</td><td> $\overline { { { \bf 8 1 . 6 9 } _ { \pm 0 . 2 5 } } }$ </td><td> $\overline { { 8 0 . 1 0 { \pm 0 . 3 7 } } }$ </td><td> $\overline { { 7 9 . 8 2 \pm 0 . 1 7 } }$ </td><td> $\overline { { 6 8 . 7 6 \pm 0 . 2 7 } }$ </td></tr><tr><td>CtrlNet</td><td> $8 1 . 4 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> ${ \bf 8 0 . 8 5 \pm 0 . 1 9 }$ </td><td> $\mathbf { 8 0 . 1 4 } \pm 0 . 1 4$ </td><td> ${ \bf 6 9 . 1 2 } _ { \pm 0 . 2 1 }$ </td></tr><tr><td rowspan="2"></td><td>Ours</td><td> $8 1 . 4 8 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 0 . 8 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 0 . 1 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> ${ \bf 6 9 . 1 2 } _ { \pm 0 . 2 4 }$  </td></tr><tr><td>Baseline</td><td> $\overline { { 8 2 . 5 4 \pm 0 . 1 9 } }$ </td><td> $\overline { { 8 1 . 1 9 \pm 0 . 1 9 } }$ </td><td> $\mathbf { 8 0 . 8 9 \bot } 0 . 1 7$ </td><td> $7 0 . 1 4 \pm 0 . 2 4$ </td></tr><tr><td rowspan="2">400</td><td>CtrlNet</td><td> $8 2 . 2 8 { \pm } 0 . 3 2$ </td><td> $8 1 . 4 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $8 0 . 8 5 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $\mathbf { 7 0 . 1 7 \pm 0 . 1 1 }$ </td></tr><tr><td>Ours</td><td> $8 2 . 0 1 { \scriptstyle \pm 0 . 1 7 }$ </td><td> ${ \bf 8 1 . 5 2 } _ { \pm 0 . 2 5 }$ </td><td> $8 0 . 7 2 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 0 . 0 4 \pm 0 . 1 3$ </td></tr></table>

Anatomy Segmentation. We evaluate Pos4CHD-synthesized normal cases on fetal cardiac segmentation. We train nnUNet for 200 epochs on WHU-4CH-Seg517 training set and explore its effects under various data scales in Table IV. We outperform CtrlNet under most settings. When only 20 real samples are available, adding synthesized data improves DSC by 10.49%. The performance gain diminishes as the real data scales over 200 empirically, where further adding real data brings marginal increments. We filter out low-quality synthetic samples before training using a pretrained nnUNet with $\mathrm { D S C } { \le } 8 0 \%$ for rejection. Our method retains 57.5% (437/760) synthesized samples, higher than the 35.4% utility of CtrlNet.

## C. Ablation Studies

1) Effects of regularization ${ \mathcal { L } } _ { \mathrm { h i s t } } .$ As shown in Table V and Fig. 6, the proposed histogram regularization effectively reduces the distribution gap between synthesized and real images during SD-UNet fine-tuning and FE-Gen training. It leads to lower FID and improves image fidelity, while the baseline shows much lower standard deviation than real data.

![](images/85d8b5f656827f73c2521e3ef30f1d1e1610fdf184b299634078b5cabfe14184.jpg)  
Fig. 9. Comparison of synthesized CHD samples with real cases. TABLE V

ABLATION STUDIES FOR ULTRASOUND GENERATION.
<table><tr><td>Settings</td><td colspan="6"></td></tr><tr><td></td><td colspan="6">Metrics</td></tr><tr><td colspan="6">Step-1: SD-VAE Adaptation</td><td></td></tr><tr><td></td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>KL↓</td><td>KID↓</td><td>FID↓</td></tr><tr><td>Real Data</td><td>1.0000</td><td>inf</td><td>0.0000 0.0325</td><td>0.0000</td><td>0.0000</td><td>0.00</td></tr><tr><td>SD-VAE w fine-tune</td><td>0.9306 0.9496</td><td>35.18 37.72</td><td></td><td>0.0180</td><td>0.0048</td><td>10.77</td></tr><tr><td> $\overline { { S t e p  – 2 \colon S D  – U N e t ~ F i n e \ – t u n i n g } }$ </td><td></td><td></td><td>0.0168</td><td>0.0010</td><td>0.0008</td><td>4.83</td></tr><tr><td colspan="7"></td></tr><tr><td></td><td>KL↓</td><td>Skew</td><td>STD</td><td>LPIPS↓</td><td>KID↓</td><td>FID↓</td></tr><tr><td>Real Data</td><td>0.0000</td><td>0.76</td><td>51.52</td><td>0.0000</td><td>0.0000</td><td>0.00</td></tr><tr><td>SD-UNet (LoRA)</td><td>0.1159</td><td>0.77</td><td>42.97</td><td>0.5872</td><td>0.0311</td><td>49.54</td></tr><tr><td> $\underline { { w \mathcal { L } _ { h i s t } } }$ </td><td>0.0194</td><td>0.73</td><td>55.18</td><td>0.6055</td><td>0.0153</td><td>37.48</td></tr><tr><td colspan="7">Step-3: FE-Gen Training</td></tr><tr><td></td><td>KL↓</td><td>Skew</td><td>STD</td><td>KID↓</td><td>FID↓</td><td>DSC↑</td></tr><tr><td>Real Data</td><td>0.0000</td><td>0.64</td><td>50.89</td><td>0.0000</td><td>0.00</td><td>80.75</td></tr><tr><td>ControlNet</td><td>0.3047</td><td>0.55</td><td>35.76</td><td>0.0575</td><td>161.10</td><td>75.68</td></tr><tr><td>Ours</td><td>0.0350</td><td>0.54</td><td>46.32</td><td>0.0423</td><td>148.13</td><td>79.52</td></tr><tr><td>w/o  $\mathcal { L } _ { h i s t }$ </td><td>0.2822</td><td>0.48</td><td>35.10</td><td>0.0846</td><td>177.65</td><td>79.23</td></tr><tr><td>w/o  $s t e p { - } 2 ~ f i n e t u n e$ </td><td>0.0216</td><td>0.55</td><td>48.55</td><td>0.0618</td><td>162.96</td><td>79.10</td></tr><tr><td>w/o learned query</td><td>0.0312</td><td>0.68</td><td>46.30</td><td>0.0428</td><td>148.97</td><td>74.55</td></tr></table>

2) Effects of learnable semantic query: Table V also validates the effectiveness of learnable semantic queries, improving DSC by 4.97%, which is critical for downstream applications. Fig. 5 also shows better image-mask alignments.

3) Effects of training strategy: Table V ablates the performance of multi-step fine-tuning described in Sec.III-C.3. We validate that fine-tuning SD-VAE and SD-UNet both benefit the overall image fidelity of our FE-Gen.

4) Effects of human filtering & mask augmentation: Human filtering removes low-quality images that fail to exhibit reasonable CHD characteristics (see Fig. 14), thereby reducing noisy annotations for training. Mask augmentation improves mask diversity, leading to better sample diversity (see Fig. 13). Both strategies improve mAP50 in Table VI, and their combination increases 6.95% on average and 7.13% for VSD.

## D. Discussions

1) Discussions on regularization weight ω : Considering the trade-off between image fidelity (FID) and spatial alignment (DSC) in Fig. 10(a), we set the weight $\omega _ { h } { = } 0 . 1$ in Eq.8. 2) Discussions on training data scales forFE-Gen: Fig. 10(b) indicates that, increasing the data scale consistently reduces FID and improves DSC, with diminishing marginal gains, revealing the importance of training data scale.

3) Discussions on key synthesis hyperparameters: As shown in Fig. 11, the CFG scale exhibits a trade-off between FID and DSC, and we set it to 2.0 for better spatial alignment. Increasing conditioning scale λ consistently lowers FID and improves DSC, with the best performance achieved at $\lambda = 1 . 0 .$

(b) IVS  
(b) Training Data Scale(b) Training Data Scale<sup>40</sup>  
TABLE VI  
VALIDATION OF DATA FILTERING AND MASK AUGMENTATION.
<table><tr><td>Dataset</td><td>Human Filter</td><td>Mask Aug</td><td>AVG</td><td>mAP50 VSD</td><td>IVS</td></tr><tr><td>Real</td><td>一</td><td>一</td><td>71.43</td><td>73.30</td><td>69.60</td></tr><tr><td rowspan="4">+Syn</td><td>×</td><td>×</td><td>73.50</td><td>70.37</td><td>76.67</td></tr><tr><td>√</td><td>X</td><td>77.63</td><td>76.03</td><td>79.27</td></tr><tr><td>X</td><td>√</td><td>77.17</td><td>79.93</td><td>74.43</td></tr><tr><td>√</td><td>√</td><td>78.38</td><td>80.43</td><td>75.50</td></tr></table>

(a) Histogram Regularization Weight (a) Histogram Regul

![](images/01991102ec7d7ba04f3c2b07da25e13efd912b7bb567b6ead59b88fb6f7126d3.jpg)

![](images/550c831a60c4fc93ed5c637ad6b3d8eb83688261a34b330679dcc309ab58213a.jpg)  
Fig. 10. Discussions on (a) regularization weight, (b) training data scale.<sup>0.5</sup>

![](images/83e2688d8e3a65f2eec8e2b6068ea6a91c342a044c37a9c2acefea28e3aaebbb.jpg)

![](images/d46d6e1a1999716d7b5e93156f357ef85f9bf03309690091ebb6ab6690458ecc.jpg)

Fig. 11. Discussions on (a) CFG scale and (b) conditioning scale.  
![](images/e933644b4eedbfb98ed4ce56fa68c4188b132f047dd4049fbc83ba21d5ac1fce.jpg)

![](images/5ba5bf868a24bd48616604aaf850f08dfbb66a42ad2a894b1b91c813e3b68e94.jpg)

![](images/8335786e757552a8a1dbfa799aac2eecfecd0349401746e7065038d20b20e056.jpg)  
84.33<sub>79.67</sub> 76.50Fig. 12. Discussions on synthesized data scales for VSD detection.

74.9378.384) Discussions on augmentation data scale: Using all 417 <sup>80.43</sup>masks from the WHU-4CH-Seg517 training set, we synthesize pseudo-VSD samples by Pos4CHD as ×1 scale, and Data Scale Data Scale Data Scalerepeat this process to construct larger scales. As shown in Fig. 12, the performance improves with more synthesized samples and gradually saturates at approximately ×3 scale.

5) Discussions on synthesized sample diversity: As shown in Fig. 13, Pos4CHD can synthesize diverse samples for CHD subtypes and normal cases, which arises from: (1) structural variations introduced by stochastic mask augmentation and counterfactual manipulation, and (2) appearance variations induced by the randomness of image generation.

6) Analysis offailure cases: Fig. 14 shows failure cases that do not faithfully exhibit CHD-specific characteristics. Most occur in pseudo-VSD samples, where the IVS remains visually continuous despite the defect specified by the counterfactual mask, attributed to: (1) the inherent stochasticity of diffusion sampling and (2) the difficulty of precise spatial control over subtle, localized structures. Although our method improves spatial alignment over the baseline, precise synthesis of subtle structures remains an important direction for future work.

Aug Mask-Ⅰ  
Synthesized Images  
Aug Mask-Ⅱ  
Synthesized Images  
![](images/30fd2097f2f5a4e10ec055482d081931e5715920f3d336134cd8588f7fb7eb9a.jpg)

Fig. 13. Diverse images synthesized from the same source mask.  
![](images/bd8503ee3ffac605e9a97560c714bd51a2c8de42bdfbca90fb1e2b68dc47d7aa.jpg)  
Fig. 14. Examples of failure cases of Pos4CHD.

## V. CONCLUSION

This paper presents Pos4CHD, a positive-free mask-guided counterfactual generation framework for fetal echocardiography, to alleviate CHD data scarcity. We first introduce FE-Gen for ultrasound generation, which mitigates histogram shift via regularization and improves spatial image-mask alignments via learnable semantic queries. Based on FE-Gen, Pos4CHD transforms normal masks into counterfactual ones and gen-<sup>EbA Average</sup> <sup>ACC=</sup> <sup>35.8%</sup>erates corresponding images, yielding pseudo-CHD image-TAPVC Average ACC= 45.8%mask pairs for downstream analysis. Extensive experiments on <sub>VSD</sub> Average ACC= 55.0%synthesized samples exhibit consistent improvements on realworld tasks including zero-shot classification, lesion detection, <sup>Accuracy</sup>and anatomy segmentation, demonstrating the potential of syn-Normal Average ACCthesized data for automated fetal CHD diagnosis and analysis.

## ACKNOWLEDGMENT

VSD Average ACCy <sup>ControlNet</sup><sub>Ours</sub>The numerical calculations are done on the supercomputing Accuracya<sup>bi</sup>system of the Supercomputing Center of WHU. We used <sup>Accuracy</sup>P<sup>r</sup>GPT solely for grammar checking and manuscript polishing throughout preparation, and all edits were revised by authors.

## REFERENCES

<sup>Co</sup><sub>b</sub>i<sup>l</sup>[1] J. Carvalho et al., “Isuog practice guidelines (updated): Fetal cardiac i<sup>li</sup>o<sup>b</sup>screening.” Ultrasound Obstet Gynecol, vol. 61, no. 6, pp. 788–803, 2023.

<sup>P</sup> T2I-Adapter[2] Y. Liu et al., “Global birth prevalence of congenital heart defects 1970– <sup>ControlNet</sup>m2017: Updated systematic review and meta-analysis of 260 studies,” Int. E J. Epidemiology, vol. 48, no. 2, pp. 455–463, 2019.

<sub>-</sub>S[3] A. J. Moon-Grady et al., “Guidelines and recommendations for performance of the fetal echocardiogram: An update from the american society of echocardiography,” Journal of the American Society of Echocardiography, vol. 36, no. 7, pp. 679–723, 2023.

<sup>DreamBooth</sup>[4] H. Y. Sun, “Prenatal diagnosis of congenital heart defects: Echocardiography,” Translational Pediatrics, vol. 10, no. 8, p. 2210, 2021.

SD-LoRA<sup>ControlNet</sup>m<sup>-</sup>[5] C. Niu, P. Wu, B. D. Man, and G. Wang, “Foundation models for <sub>ControlNet</sub>-<sup>2</sup><sub>E</sub> medical imaging: Status, challenges, and directions,” IEEE Transactions <sup>Ours</sup>d<sup>i</sup>S<sup>N</sup>on Medical Imaging, pp. 1–1, 2026.

[6] M. C. Fiorentino, F. P. Villani, M. Di Cosmo, E. Frontoni, and S. Moccia, “A review on deep-learning algorithms for fetal ultrasound-image analysis,” Medical Image Analysis, vol. 83, p. 102629, 2023.

[7] M. Dom´ınguez, Y. Velikova, N. Navab, and M. F. Azampour, “Diffusion as sound propagation: Physics-inspired model for ultrasound image generation,” in Proceedings of Medical Image Computing and Computer Assisted Intervention, vol. LNCS 15004, October 2024.

[8] P. Guo et al., “Medical ai for synthetic imaging,” in IEEE/CVF Winter Conference on Applications of Computer Vision, 2025, pp. 4430–4441.

[9] Y. Xu et al., “Medsyn: text-guided anatomy-aware synthesis of highfidelity 3-d ct images,” IEEE Transactions on Medical Imaging, vol. 43, no. 10, pp. 3648–3660, 2024.

[10] J. Mao et al., “Medsegfactory: Text-guided generation of medical image-mask pairs,” in IEEE/CVF International Conference on Computer Vision, 2025, pp. 21 525–21 535.

[11] K.-W. Yung et al., “Anatodiff: Synthesizing anatomically truthful radiographs with limited training images,” IEEE Transactions on Medical Imaging, vol. 45, no. 6, pp. 2750–2763, 2026.

[12] H. Wu, Z. Zhao, Y. Zhang, Y. Wang, and W. Xie, “Mrgen: Segmentation data engine for underrepresented mri modalities,” in IEEE/CVF International Conference on Computer Vision, 2025, pp. 19 903–19 913.

[13] L. Zhang et al., “Generative ai enables medical image segmentation in ultra low-data regimes,” Nature Communications, vol. 16, no. 1, p. 6486, 2025.

[14] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 674–10 685.

[15] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 3813–3824.

[16] Y. Wang et al., “Toward general text-guided multimodal brain mri synthesis for diagnosis and medical image analysis,” Cell Reports Medicine, vol. 6, no. 6, 2025.

[17] F. Perez-Garc´ ´ıa et al., “Radedit: Stress-testing biomedical vision models via diffusion image editing,” in European Conference on Computer Vision, 2024, pp. 358–376.

[18] Y. Duan et al., “Fetalflex: Anatomy-guided diffusion model for flexible control on fetal ultrasound image synthesis,” Medical Image Analysis, vol. 105, p. 103725, Oct. 2025.

[19] C. Zhao et al., “Maisi-v2: Accelerated 3d high-resolution medical image synthesis with rectified flow and region-specific contrastive loss,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 15, 2026, pp. 13 088–13 098.

[20] N. Konz, Y. Chen, H. Dong, and M. A. Mazurowski, “Anatomicallycontrollable medical image generation with segmentation-guided diffusion models,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2024, pp. 88–98.

[21] H. Min, T. You, H. Lee, Y. Cho, and S. Cho, “Instructx2x: An interpretable local editing model for counterfactual medical image generation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2025, pp. 279–288.

[22] M. B. Alaya, D. M. Lang, B. Wiestler, J. A. Schnabel, and C. I. Bercea, “Mededit: Counterfactual diffusion-based image editing on brain mri,” in Simulation and Synthesis in Medical Imaging, 2025, pp. 167–176.

[23] A. Oliveras et al., “Anatomically guided latent diffusion for highresolution 3d chest ct synthesis,” Scientific Reports, 2026.

[24] Q. Chen et al., “Towards generalizable tumor synthesis,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11 147–11 158.

[25] F. Dai et al., “Improving ai models for rare thyroid cancer subtype by text-guided diffusion models,” Nature Communications, vol. 16, no. 1, p. 4449, 2025.

[26] S. Chen et al., “Subtyping breast lesions via generative augmentationbased long-tailed recognition in ultrasound,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2025, pp. 519–529.

[27] C. Zhan et al., “ MedM2G: Unifying Medical Multi-Modal Generation via Cross-Guided Diffusion with Visual Invariant ,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Los Alamitos, CA, USA, Jun. 2024, pp. 11 502–11 512.

[28] F. F. Abir, S. S. Jennifer, N. Yousefi, and L. J. Brattain, “Hybrid diffusion model for breast ultrasound image augmentation,” in IEEE 23rd International Symposium on Biomedical Imaging, 2026, pp. 1–4.

[29] S. Qiao et al., “A pseudo-siamese feature fusion generative adversarial network for synthesizing high-quality fetal four-chamber views,” IEEE

Journal of Biomedical and Health Informatics, vol. 27, no. 3, pp. 1193– 1204, 2023.

[30] Y. Wen, K. M. Curran, X. Wang, N. A. Healy, and J. J. Healy, “Synthesizing breast cancer ultrasound images from healthy samples using latent diffusion models,” Journal of Medical Imaging, vol. 13, no. 2, p. 024002, 2026.

[31] D. Stojanovski, U. Hermida, P. Lamata, A. Beqiri, and A. Gomez, “Echo from noise: Synthetic ultrasound image generation using diffusion models for real image segmentation,” in International Workshop on Advances in Simplifying Medical Ultrasound, 2023, pp. 34–43.

[32] G. Van De Vyver et al., “Generative augmentations for improved cardiac ultrasound segmentation using diffusion models,” Scientific Reports, vol. 15, no. 1, p. 38013, 2025.

[33] M. Yazdani et al., “Flow matching for medical image synthesis,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2025, pp. 216–226.

[34] F. Wang, K. Whelan, G. Silvestre, and K. M. Curran, “Generative diffusion model bootstraps zero-shot classification of fetal ultrasound images in underrepresented african populations,” in International Workshop on Preterm, Perinatal and Paediatric Image Analysis, 2024, pp. 143–154.

[35] M. Arjemandi, S. Hassan, H. Wang, S. Valappil, and M. Yaqub, “Diffusion-based fetal ultrasound synthesis with active learning,” in International Workshop on Advances in Simplifying Medical Ultrasound, 2025, pp. 130–139.

[36] S. Qiao et al., “Fetal ultrasound four-chamber view editing synthesis via denoising diffusion model,” Expert Systems with Applications, vol. 295, p. 128939, 2026.

[37] Y. Yeganeh et al., “Latent drifting in diffusion models for counterfactual medical image synthesis,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 7685–7695.

[38] S. Liu, F. Wang, Z. Ren, C. Lian, and J. Ma, “Controllable counterfactual generation for interpretable medical image classification,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2024, pp. 143–152

[39] B. Li, W. Peng, M. Li, E. Adeli, and K. M. Pohl, “Integrating anatomical priors into a causal diffusion model,” IEEE Transactions on Medical Imaging, vol. 45, no. 7, pp. 3934–3947, 2026.

[40] S. Wu, S. Kurugol, and A. Tsai, “Improving the radiographic image analysis of the classic metaphyseal lesion via conditional diffusion models,” Medical Image Analysis, vol. 97, p. 103284, 2024.

[41] ISUOG, “Abnormal 4-chamber view,” Visual Encyclopedia of US in Obstetrics and Gynecology. [Online]. Available: https://www.isuog.org/ education/visuog/obstetrics/heart/abnormal-4-chamber-view.html

[42] S. Wang et al., “Generative artificial intelligence in medical imaging: Foundations, progress, and clinical translation,” Research, vol. 8, p. 1029, 2025.

[43] U. B. of Chinese Medical Doctor Association et al., “Chinese expert consensus on assessment standards of obstetric ultrasound training program (2022 edition),” Chin J Ultrason, vol. 31, no. 5, p. 10, 2022.

[44] W. Xuan, J. Zhang, J. Liu, B. Du, and D. Tao, “Rethink sparse signals for pose-guided text-to-image generation,” in IEEE/CVF International Conference on Computer Vision, 2025, pp. 15 896–15 906.

[45] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021.

[46] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “Textdiffuser: Diffusion models as text painters,” Advances in Neural Information Processing Systems, vol. 36, pp. 9353–9387, 2023.

[47] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.

[48] F. Li et al., “Application of artificial intelligence in vsd prenatal diagnosis from fetal heart ultrasound images,” BMC Pregnancy and Childbirth, vol. 24, no. 1, p. 758, 2024.

[49] F. Maani et al., “Fetalclip: A visual-language foundation model for fetal ultrasound image analysis,” npj Digital Medicine, 2026.

[50] S. Lin, B. Liu, J. Li, and X. Yang, “Common diffusion noise schedules and sample steps are flawed,” in IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 5392–5399.

[51] M. M. K. Sarker et al., “Leveraging harmonic convolutions for automated standard plane detection in fetal heart ultrasound videos,” Medical Image Analysis, p. 103758, 2025.

[52] N. Ruiz et al., “Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 500–22 510.

[53] C. Mou et al., “Learning adapters to dig out more controllable ability for text-to-image diffusion models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 5, 2024, pp. 4296–4304.