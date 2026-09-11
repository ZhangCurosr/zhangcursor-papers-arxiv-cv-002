# Multi-Modal Controlled Coherent Motion Generation

Yifei Liu<sup>1,2</sup>, Qiong Cao<sup>2</sup>, Hongwei Yi<sup>3</sup>, Huaiguang Jiang<sup>1</sup>, and Changxing Ding<sup>1</sup> <sup>⋆</sup>

<sup>1</sup> South China University of Technology

ft\_lyf@mail.scut.edu.cn, hihuagong2021@scut.edu.cn, chxding@scut.edu.cn

<sup>2</sup> Joy Future Academy

mathqiong2012@gmail.com

<sup>3</sup> Peking University

hongweiyi@pku.edu.cn

Abstract. It is natural for us to walk and talk simultaneously. This paper tackles the challenge of replicating such natural behaviors in 3D avatar motion generation driven by concurrent multi-modal inputs, e.g., a text description “a man is walking” alongside a speech audio. Existing methods, constrained by the scarcity of aligned multi-modal data, typically combine motions from individual modalities sequentially or through weighted sum. However, they often result in mismatched or unrealistic movements. To overcome these limitations, we propose MOCO, a novel difusion-based framework capable of processing multiple simultaneous inputs—including speech audio, text descriptions, and trajectory data—to generate coherent and lifelike motions without requiring aligned multimodal data. Our key innovation lies in decoupling the motion generation process. During each denoising step, the difusion model independently generates motions for each modality from the input noise and assembles the body parts according to predefined spatial rules. The resulting combined motion is then difused and serves as the input noise for the subsequent denoising step. This iterative approach enables each modality to refine its contribution within the context of the overall motion, progressively harmonizing movements across modalities. Consequently, the generated motions become increasingly natural and fluid with each iteration, achieving coherent and synchronized behaviors. We evaluate our approach using a purpose-built multi-modal benchmark. Experimental results demonstrate that MOCO outperforms existing baselines, advancing the field of multi-modal motion generation for 3D avatars. The code will be released on https://feifeifeiliu.github.io/MOCO.

Keywords: Human Motion Generation · Difusion Models · Multi-Modal

## 1 Introduction

Imagine watching a virtual talk show where the host delivers engaging dialogue complemented by expressive gestures, natural body movements, and precise

⋆ Corresponding author.

movement paths. The host walks across the stage following a scripted trajectory, uses hand gestures to emphasize points based on their speech, and shifts posture in response to both the conversation’s flow and predefined text instructions—all occurring in perfect harmony. This level of realism transforms the viewing experience, making interactions feel genuine and immersive. Achieving such lifelike behavior is no small feat, yet it is essential for enhancing user engagement in applications ranging from virtual reality to interactive gaming and beyond.

Driving a 3D avatar to perform such lifelike motions involves managing multiple control signals, such as text descriptions, speech audio, and trajectory data. Multi-modal signals may be provided concurrently, for instance, a text prompt like “a man is walking” alongside a speech audio clip. However, most prior works primarily focus on single-modality control, such as text-to-motion [20, 59] or speech-to-gesture [18, 71]. Recent studies [76, 82, 83] have explored designing unified models capable of addressing multi-modal signals by leveraging datasets from diferent generation tasks. Nevertheless, these models typically process only one modality at a time, combining motions conditioned on diferent inputs in a limited and sequential manner when multiple control signals are present.

![](images/441d5aa53a2b934d34171b0d285866c5052f24f3f18d1981c6d95dac24d73df3.jpg)  
Fig. 1: Examples of Multi-Modal Controlled Motion Generation. Given multiple control signals of diferent modalities—including text descriptions, speech audio, and trajectory data—our MOCO framework generates realistic and coherent holistic body motion. This includes both body movements and detailed features such as facial expressions and hand gestures, all closely aligned with the provided conditions. To clearly illustrate this, we highlight two clips with temporal zoom, showcasing the natural integration of speech gestures and lower-body movements in our generated motions. Additional visual results are provided in Fig. 4 and the supplementary materials.

The primary challenge in achieving simultaneous multi-modal control of mo tion generation is the scarcity of aligned multi-modal data. While collecting additional multi-modal data could help, it requires significant resources. In addition, the activity regions in speech-to-gesture datasets are often limited, making it hard to train models that can generate trajectory-controlled speech gestures. Some eforts address this issue by combining the predictions of text-driven model and audio-driven model through weighted sum [12, 70]. [37] suggested using speech scripts as pseudo text labels to create aligned text-audio-motion datasets, replacing scripts with movement descriptions during inference. However, these approaches face inherent limitations: for weighted sum, the predominance of standing poses in speech-to-gesture datasets creates imbalanced model corrections; for pseudo-labels, the use of speech scripts as training labels limits generalization to diverse motion descriptions during inference. To overcome these challenges, we propose a novel difusion-based framework, named Multi-MOdal Controlled COherent Motion Synthesis (MOCO). MOCO exploits a natural spatial decomposition: speech audio primarily drives upper-body dynamics—–gestures and facial expressions–—while text descriptions mainly influence lower-body movements such as walking and stance shifts (see Appendix A for supporting statistics). To achieve this, MOCO is first trained on multiple datasets, ensuring that the model can independently generate motions conditioned on either text or speech inputs. At each denoising step, the model generates motions for each modality separately from the input noise and assembles the body parts according to predefined spatial rules, i.e. combining audio-driven upper-body motion with text-driven lower-body motion. This combined motion is then difused and used as the input noise for the next denoising step. The separation ensures that each body part’s motion is highly aligned with its corresponding input condition, while the iterative process conditions each generation step on the current state of the combined motion. This allows each modality to refine its contribution within the context of the overall movement. Consequently, with each iteration, the motions generated for diferent body parts become increasingly harmonized, resulting in natural movements that exhibit coherent and synchronized behaviors. Furthermore, this decoupled generation process enables our framework to incorporate trajectory control into co-speech motion generation. We can leverage trajectory data to generate text-driven motion and combine it with audio-driven motion, producing speech gestures that closely align with the given trajectory.

To facilitate the evaluation of this novel task, we develop a multi-modal benchmark comprising 1,000 test clips which are generated from 40 fundamental text descriptions of body movements $( e . g .$ , “walk forwards” and “step back and sit down”) and 694 audio clips from eight diferent speakers. Each test clip integrates two text prompts describing a movement with two speech audio clips. We rigorously evaluate our approach against baseline methods using both textto-motion and speech-to-gesture metrics. Experimental results demonstrate that our method outperforms existing baselines, advancing the field of multi-modal controlled motion generation for 3D avatars.

## 2 Related Work

## 2.1 Multi-Modal Conditioned Motion Generation

In recent years, human motion generation has received significant attention, largely driven by advancements in dataset collection. Various scenarios have been explored depending on the input conditions, including action labels [21], text descriptions [6, 14, 20, 23, 26, 27, 35, 59, 63, 75, 78], speech audio [18, 31, 40, 42, 44, 50, 53, 71, 80], music [34, 55, 60], scene context [24, 47], spatial signal [58, 66], and even the motion of another person [11, 22, 46, 48, 57, 62]. Beyond single-modality control, research eforts have been extended to handle integrating multi-modal signals. For instance, some works considered speaker identity, speech audio, and transcripts to generate conversational gestures [2,3,68,73], while other works jointly leverage text and scene inputs for motion generation [10,15,64,72]. Moreover, some works focus on integrating various datasets to train unified motion models that enhance scalability and applicability across multiple scenarios [13, 28, 76, 81, 83].

A critical limitation persists, however: current models depend heavily on aligned multi-modal training data, limiting their ability to handle novel input combinations (e.g., text with audio, audio with trajectory) unseen in training. To address this, FreeTalker [70] and SynTalker [12]—the latter concurrent with the first version of this paper [45]—propose combining predictions from textdriven and audio-driven models through weighted sum. Liu et al. [37] suggested using speech scripts as pseudo text labels (e.g., “A male speaker is saying: ‘I am shocked by what you have done.’ ”) to create aligned text-audio-motion datasets for training and replace these scripts with movement descriptions during inference. However, the weighted sum method poses challenges, as the speech-driven model exerts significantly stronger correction forces than the text-driven model. This imbalance often generates motions that overly favor speech conditions while suppressing text conditions (Appendix E). Regarding pseudo labels, the reliance on speech transcripts as supervisory signals creates a domain gap between the transcript distribution and the motion-description distribution, thereby constraining generalization to diverse motion descriptions.

## 2.2 Difusion-based Methods in Motion Generation

The difusion model has been recognized as one of the most advanced generative paradigms and has also gained significant attraction in human motion generation. Its applications can be categorized into two primary streams according to the space where the denoising process is implemented. One representative method in the first stream is the Human Motion Difusion Model (MDM) [59], which operates directly in the original motion space to perform denoising [1,29,32,36,84]. This approach ofers several advantages, including high editability and strong controllability, enabling operations such as concatenation and combination within the original motion space during the denoising process. The second stream is exemplified by the Motion Latent Difusion model (MLD) [14], which conducts denoising in the latent space of a Variational Autoencoder (VAE) [8, 16, 52].

Utilizing the VAE latent space improves eficiency in representing complex motion data and significantly reduces the computational load required for difusion processes, resulting in faster inference speeds. In this work, we select MDM as the foundational model due to its superior editability [5, 49, 54].

## 2.3 Compositional Motion Generation

A complementary line of work generates complex motion by composing multiple actions or sub-motions. Spatial composition assembles motions of diferent body parts: SINC [5] combines multiple text-driven actions through body-part masks in a VAE-based, ofline manner. Temporal composition instead sequences multiple actions over time and focuses on seamless transitions between them, as in TEACH [4], T2LM [33], and FlowMDM [9]. STMC [49] unifies both axes, bringing per-denoising-step body-part composition into the difusion framework with a DifCollage-style [79] mechanism for temporally overlapping actions. All these methods compose homogeneous, text-only conditions; we instead build on the per-step composition paradigm and extend it to heterogeneous audio, text, and trajectory control, placing MOCO at the intersection of compositional and multi-modal motion generation.

## 3 Method

In this section, we begin with a brief introduction to the Motion Difusion Model (MDM) [59], which serves as our foundational model (Sec. 3.1). Then, we describe our framework’s data representations and important model components (Sec. 3.2). Next, we explain our multi-modal decoupled denoising strategy for holistic motion generation (Sec. 3.3). Finally, we address a more complex scenario where the trajectory condition is included and utilize a Large Language Model for motion planning (Sec. 3.4). An overview of our framework can be found in Fig. 2.

## 3.1 Preliminary: Motion Difusion Model

MDM is a difusion-based motion generation model [59]. The same as other difusion models, it regards difusion as a Markov noising process $( x _ { t } ) _ { t = 0 } ^ { T }$ that starts from a training sample $x _ { 0 }$ . As the time step t increases, the distribution of x\_T approaches a standard normal distribution. Besides, MDM models the conditional distribution $p ( x _ { 0 } \mid c )$ by reversing the difusion process through iterative denoising on $x _ { T }$ . To achieve this, it minimizes a loss function that penalizes the diference between the original and denoised samples. Moreover, it performs sampling from $p ( x _ { 0 } \mid c )$ iteratively, predicting $x _ { 0 }$ at each timestep t and computing $x _ { t - 1 }$ until t reaches 0.

MDM adopts the classifier-free guidance [25] to adjust its adherence to the conditioning signal c. Specially, its denoiser G is trained on both conditioned and unconditioned data by randomly setting c as $\mathcal { D }$ , which allows $G ( x _ { t } , t , \theta )$ to approximate the unconditional data distribution. During sampling, it adjusts the strength of c using a scaling factor s:

![](images/79ff187b2de8d68f5d00ae94179c2e3342c349868e68c0ff43c093bd8709173e.jpg)  
Fig. 2: Overview of MOCO. At each denoising step t , the input conditions and noisy data are fed into their respective denoisers to predict clean motion, which is then difused for the next iteration. Specifically, the upper-body motion conditioned on speech audio and the lower-body motion conditioned on text description are combined to form the overall body motion. The blue arrows in the figure highlight two key points. One indicates that the denoising process of v\_0 is completed before body motion denoising. The other shows that after the denoising process, the detailed facial and hand movements, and the combined body motion are integrated together to produce the final holistic motion.

$$
\boldsymbol { G } ^ { s } \left( \boldsymbol { x } _ { t } , t , \boldsymbol { c } \right) = \boldsymbol { G } \left( \boldsymbol { x } _ { t } , t , \infty \right) + \boldsymbol { s } \cdot \left( \boldsymbol { G } \left( \boldsymbol { x } _ { t } , t , \boldsymbol { c } \right) - \boldsymbol { G } \left( \boldsymbol { x } _ { t } , t , \infty \right) \right) ,\tag{1}
$$

where $G ^ { s }$ denotes the denoiser with the classifier-free guidance.

## 3.2 Data Representation and Model Architecture

Data Representation. Our framework incorporates four data modalities: motion, text, audio, and trajectory. The motion data is represented as $m =$ $\{ m ^ { n } \} | _ { n = 1 } ^ { \dot { N } } \in \mathcal { R } ^ { N \times { 4 9 1 } }$ , where N is the number of frames. Specifically, the motion data for each frame is denoted as $m ^ { n } = \{ b ^ { n } , d ^ { n } \}$ , with $\bar { b } \in \mathcal { R } ^ { 2 0 5 }$ representing the body pose [49], and $d \in \mathcal { R } ^ { 2 8 6 }$ capturing detailed facial expression and hand movements. The text embeddings are encoded using the CLIP model [51] and are denoted as $c _ { t e x t } \in \mathcal { R } ^ { 5 1 2 }$ . Audio features are extracted via the Wav2Vec2 model [7] and represented as $c _ { a u d i o } \in \mathcal { R } ^ { N \times 7 6 8 }$ . Finally, the trajectory data is encoded as $c _ { t r a j } \ \stackrel { - } { \in } \mathcal { R } ^ { N \times 2 }$ , representing the position on the XY-plane for each frame.

Model Design. Our framework includes four transformer-based denoisers: one for text-to-motion (T2M), one for speech-to-gesture (S2G), one for trajectoryto-velocity (T2V), and one for speech-to-details (S2D) that synthesizes facial expressions and hand poses:

$$
\hat { b } _ { 0 } = G _ { \mathrm { T 2 M } } ( b _ { t } , t , c _ { t e x t } ) , \quad \hat { b } _ { 0 } = G _ { \mathrm { S 2 G } } ( b _ { t } , t , c _ { a u d i o } ) ,\tag{2}
$$

$$
\hat { v } _ { 0 } = G _ { \mathrm { T 2 V } } ( v _ { t } , t , c _ { t r a j } ) , \quad \hat { d } _ { 0 } = G _ { \mathrm { S 2 D } } ( d _ { t } , t , c _ { a u d i o } ) .\tag{3}
$$

We denote the sampling with classifier-free guidance for each denoiser as $G _ { u } ^ { s } ,$ where u ∈ {T2M, S2G, T2V, S2D}.

For the T2M denoiser conditioned on a text embedding $c _ { t e x t } .$ , we follow prior works [14, 59] by treating $c _ { t e x t }$ as a token and applying self-attention to incorporate the semantic information in $c _ { t e x t }$ into the motion generation process. In contrast, the other denoisers are conditioned on sequential data; therefore, we utilize cross-attention to model the relationships between the input sequences and the generated motion. Additionally, the T2M and T2V denoisers are trained on HumanML3D [20], while the S2G and S2D denoisers are trained on the BEAT2 [40] dataset. All denoisers adhere to the objective function and difusion paradigm described in Sec. 3.1. The computational complexity of each component is reported in Appendix F.

![](images/313fa054063c4c166c7d5185fe2cac160d5b016f9ba5109fa5882a4c3c52b65e.jpg)  
Fig. 3: Examples of synchronous and asynchronous conditions. Synchronous conditions occur when all condition signals are provided within the same time interval. In contrast, asynchronous conditions involve multiple conditions, each corresponding to diferent time intervals.

## 3.3 Multi-Modal Decoupled Denoising

In this section, we introduce our multi-modal decoupled denoising strategy for generating motion in scenarios where text and speech audio are provided synchronously—that is, within the same time interval—as shown in Fig. 3 (a).

Our key observation is that the speech audio naturally guides upper-body gestures (including head and arm poses), while text descriptions mainly influence lower-body movements (including spine and leg poses) like walking or shifting stance. In a difusion context, this implies that the joint conditional probability $p ( b _ { t - 1 } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t } )$ can be approximated by the product of two probabilities: $p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { t e x t } } , b _ { t } )$ and $p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a u d i o } } , b _ { t } )$ . Mathematically, this can be represented as follows:

$$
p ( b _ { t - 1 } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t } ) \approx p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { t e x t } } , b _ { t } ) \cdot p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a u d i o } } , b _ { t } ) .\tag{4}
$$

Here, $b _ { t }$ represents the motion at denoising step t , which is composed of the upperbody motion $b _ { t , \mathrm { u p p e r } }$ and the lower-body motion $b _ { t , \mathrm { l o w e r } }$ . A detailed derivation of $\mathrm { E q . \ ( 4 ) }$ is provided in Appendix B.

Based on this observation, we can decouple the multi-modal controlled denoising process into distinct streams. The text-driven stream $p ( b _ { t - 1 } \mid c _ { \mathrm { t e x t } } , b _ { t } )$ can be formulated as follows:

$$
\hat { b } _ { 0 } ^ { \mathrm { t e x t } } = G _ { \mathrm { T 2 M } } ^ { s } ( b _ { t } , t , c _ { \mathrm { t e x t } } ) , \quad b _ { t - 1 } ^ { \mathrm { t e x t } } = \sqrt { \alpha _ { t - 1 } } \hat { b } _ { 0 } ^ { \mathrm { t e x t } } + \sqrt { 1 - \alpha _ { t - 1 } } \epsilon ,\tag{5}
$$

where $\begin{array} { r } { \alpha _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ represents the cumulative product of $( 1 - \beta _ { s } )$ up to timestep t , and $\beta _ { s }$ is the variance schedule controlling the amount of noise added at each timestep. The variable $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ is Gaussian noise sampled from a standard normal distribution [59]. Similarly, the audio-driven stream $p ( b _ { t - 1 } \mid c _ { \mathrm { a u d i o } } , b _ { t } )$ is defined as:

$$
\hat { b } _ { 0 } ^ { \mathrm { a u d i o } } = G _ { \mathrm { S 2 G } } ^ { s } ( b _ { t } , t , c _ { \mathrm { a u d i o } } ) , \quad b _ { t - 1 } ^ { \mathrm { a u d i o } } = \sqrt { \alpha _ { t - 1 } } \hat { b } _ { 0 } ^ { \mathrm { a u d i o } } + \sqrt { 1 - \alpha _ { t - 1 } } \epsilon .\tag{6}
$$

Finally, the body motion $b _ { t - 1 }$ at step t is computed as follows:

$$
b _ { t - 1 } = I \odot b _ { t - 1 } ^ { \mathrm { t e x t } } + ( 1 - I ) \odot b _ { t - 1 } ^ { \mathrm { a u d i o } } ,\tag{7}
$$

where $I \in \mathbb { R } ^ { 2 0 5 }$ is a binary vector with entries set to 1 for the lower body and 0 for the upper body; \odot denotes element-wise multiplication.

The decoupled denoising allows each body part’s motion to be precisely guided by its corresponding input condition, ensuring high fidelity to the control signals. Furthermore, each stream adjusts its output based on the current overall motion, promoting coordination among the body parts. As this process continues, the motions generated for diferent body parts become increasingly synchronized, resulting in natural and coherent full-body movements.

## 3.4 Trajectory and Motion Planning

The above subsection handles the multi-modal decoupled denoising for synchronous conditions. We now extend our MOCO framework to tackle more complex, challenging yet more realistic scenarios, such as incorporating trajectory control and leveraging Large Language Model (LLM) for motion planning.

Trajectory Control. Following the approach of Petrovich et al. [49], we represent the global transition of body pose using the velocity vector $v = [ \dot { r } _ { x } , \dot { r } _ { y } , \dot { \theta } ]$ where $\dot { r } _ { x }$ and $\dot { r } _ { y }$ are the linear velocities of the pelvis in the x and y directions, respectively, and $\dot { \theta }$ is the angular velocity about the body’s vertical $( \mathrm { Z } )$ axis. Given the trajectory data $c _ { \mathrm { t r a j } }$ , we first predict $\hat { v } _ { 0 }$ using Eq. (3). To enhance prediction accuracy, we incorporate loss guidance into our method. During each denoising step for predicting the velocity vector, we compute $\hat { v } _ { 0 }$ using Eq. (3) and apply loss guidance as follows:

$$
L _ { \mathrm { g u i d a n c e } } = F K ( \hat { v } _ { 0 } ) - c _ { \mathrm { t r a j } } ,\tag{8}
$$

where FK represents the diferentiable Forward Kinematics function [65] that converts linear and angular velocities into the trajectory. We follow the methodology of InterControl [65] and optimize $L _ { \mathrm { g u i d a n c e } }$ with respect to $\hat { v } _ { 0 }$ using the second-order LBFGS optimizer [38]. This optimization ensures that the predicted global transitions closely match the provided trajectory data.

Once $\hat { v } _ { 0 }$ is predicted, we substitute the velocity component in $\hat { b } _ { 0 }$ with $\hat { v } _ { 0 }$ during each iteration of its generation. This substitution guides the generation process to adapt the remaining elements of $\hat { b } _ { 0 }$ to align with $\hat { v } _ { 0 }$ , thereby ensuring consistency with the provided trajectory data.

Table 1: Example of an LLM-generated motion timeline.
<table><tr><td>Condition</td><td>Start Time</td><td>End Time</td><td>Body Part</td></tr><tr><td>c1 = kneel down</td><td>f i</td><td>f f f f f</td><td>spine, legs</td></tr><tr><td>c2 = stand up from ground</td><td>f f fだ</td><td></td><td>spine, legs</td></tr><tr><td>c3 = run forward</td><td></td><td></td><td>spine, legs</td></tr><tr><td>c4 = audio clip 1</td><td></td><td></td><td>head, arms</td></tr><tr><td>c5 = audio clip 2</td><td></td><td></td><td>head, arms</td></tr></table>

Motion Planning. In real-world applications, models could face audio conditions paired with complex text descriptions, which can challenge their ability to handle intricate semantics. One approach to address this challenge is decomposing complex textual descriptions into basic motion units and generating them separately [56]. However, this decomposition often introduces more complex conditions—such as asynchronous conditions, as illustrated in Fig. 3.

To address these challenges, we propose an LLM-based motion planning pipeline. Our pipeline begins by using an LLM to generate a “motion timeline” from the input text and audio conditions. Particularly, we carefully design a prompt that instructs the LLM to decompose complex text into elementary motion units, insert appropriate pose transitions, and specify the start and end times and the body parts involved for each condition. More details are provided in Appendix C. For example, when given the text condition “kneel down then walk forward” along with two audio clips, the LLM produces a motion timeline comprising multiple intervals, as detailed in Tab. 1.

Notably, the transition condition $c _ { 2 }$ is automatically generated by the LLM, which ensures a smooth stance changes and avoids abrupt movements. Users, however, have the flexibility to customize the timeline. We then adopt the strategy of STMC [49], generating body masks according to a predefined timeline. At each denoising step t, the body pose across the entire timeline is generated as follows:

$$
\hat { b } _ { 0 } = \sum _ { j = 1 } ^ { J } I _ { j } \odot G _ { j } ^ { s } \left( b _ { t , f _ { j } ^ { s } : f _ { j } ^ { e } } , t , c _ { j } \right) ,\tag{9}
$$

where $I _ { j }$ represents a binary mask corresponding to the motion generated by the j-th condition, and $G _ { j } ^ { s } \in \{ G _ { \mathrm { T 2 M } } ^ { s } , G _ { \mathrm { S 2 G } } ^ { s } \}$ denotes the denoiser utilized for the j-th condition. The operator \odot stands for element-wise multiplication.

Similarly, the facial and hand movements at step t are computed by:

$$
\hat { d } _ { 0 } = \sum _ { j = 1 } ^ { J } G _ { \mathrm { S 2 D } } \left( d _ { t , f _ { j } ^ { s } : f _ { j } ^ { e } } , t , c _ { j } \right) .\tag{10}
$$

Since the text-to-motion dataset, HumanML3D, lacks facial and hand movement data, the denoiser $G _ { \mathrm { S 2 D } }$ is only trained on the speech-to-gesture dataset. Therefore, in intervals without audio clips, we maintain the model’s temporal continuity by setting the conditional input $c _ { j }$ to the empty condition $\varnothing .$ Finally, we employ DifCollage [79] to ensure smoother transitions at the interval boundaries.

## 4 Experiments

## 4.1 Datasets

Task-Specific Datasets. The HumanML3D dataset is a large Text-to-Motion dataset created by amalgamating motion sequences from the Human-Act12 and AMASS datasets [20]. It consists of 14,616 motions and 44,970 descriptions composed of 5,371 distinct words, totaling 28.59 hours of motion data. To align the data representation—specifically, to use SMPL-X parameters for representing joint rotations—we utilize only the AMASS portion of HumanML3D because it has an oficial SMPL-X version. The BEAT2 dataset is a large-scale Speech-to-Gesture dataset specifically designed for research in speech-to-gesture generation [40]. It contains synchronized recordings of speech audio and corresponding 3D motion capture data of human gestures, totaling 60 hours of data from 25 speakers. In addition to audio and motion data, the dataset includes valuable annotations such as text transcriptions and emotional states. Here, we select speakers with IDs lower than 10, resulting in a total of 10.15 hours of training data.

Multi-Modal Benchmark. We create a multi-modal benchmark of 1,000 test clips to evaluate our proposed task efectively. Each test clip is automatically constructed and contains two text descriptions and two audio clips. To create these clips, we manually collect 40 texts focusing on lower-body movements that commonly occur during speech delivery or conversation. We then split the audio from the BEAT2 test set into clips using a Voice Activity Detector (VAD). To serve as ground truth for computing evaluation metrics (Sec. 4.2), we selected motion samples from AMASS and BEAT2 corresponding to each text and audio clip. This structured construction enables controlled and repeatable comparisons under concurrent multi-modal conditions and isolates the challenge of resolving conflicting cross-modal signals. Accordingly, the benchmark is intended to probe compositional consistency rather than broad coverage of open-ended semantics. More details about the benchmark are provided in Appendix H.

Table 2: Comparison with baselines. We highlight the best performance in bold and use underlining to indicate the second-best performance.
<table><tr><td></td><td colspan="4">Text2Motion</td><td colspan="3">Speech2Gesture</td></tr><tr><td></td><td>FID+ ↓</td><td>R1 ↑</td><td>M2T ↑</td><td>M2M ↑</td><td>FID-A↓</td><td>BC↑</td><td>L1div ↑</td></tr><tr><td>GT (Ground Truth)</td><td>0.000</td><td>40.0</td><td>0.781</td><td>1.000</td><td></td><td>-</td><td></td></tr><tr><td>Weighted Sum [70]</td><td>1.335</td><td>6.8</td><td>0.546</td><td>0.537</td><td>2.17</td><td>2.20</td><td>4.08</td></tr><tr><td>Pseudo-Text [37]</td><td>1.593</td><td>2.2</td><td>0.511</td><td>0.503</td><td>2.22</td><td>2.55</td><td>6.43</td></tr><tr><td>SynTalker [12]</td><td>0.985</td><td>9.8</td><td>0.601</td><td>0.603</td><td>6.60</td><td>2.95</td><td>9.12</td></tr><tr><td>MOCO</td><td>0.862</td><td>24.6</td><td>0.649</td><td>0.639</td><td>3.83</td><td>2.72</td><td>8.62</td></tr></table>

## 4.2 Metrics

We evaluate our method using two categories of metrics: text-to-motion and speech-to-gesture [20, 40, 49]. For text-to-motion, FID+ assesses realism by measuring the distribution diference between real and generated motions using five random 5-second clips per test sample. R1 evaluates alignment by recording the frequency of correct text prompts appearing in the top-1 retrieved text. M2T (motion-to-text) and M2M (motion-to-motion) measure alignment through cosine similarity between embeddings of generated motions and ground truth texts or motions. In the speech-to-gesture category, FID-A similarly measures the realism of motion generated based on speech audio. Beat Consistency (BC) evaluates how well gestures synchronize with the rhythm and beats of the speech, while L1 Diversity (L1Div) quantifies gesture diversity by calculating the average L1 distance between multiple gesture clips. This comprehensive set of metrics thoroughly evaluates our method across key dimensions.

## 4.3 Comparison with Baselines

Baseline Setting. To efectively evaluate the performance of our method, we propose a series of baselines inspired by reasonable settings and prior works. These include: Weighted Sum, a method that follows [70] by combining the predictions of text- and audio-conditioned models through weighted sum; Pseudo-Text, a method that follows [37] by using pseudo text descriptions of a speaker’s speech as the text condition during training; and SynTalker [12], an open-source multi-modal motion generation method performing weighted sum in body-part-specific latent space. Notably, because these baselines cannot handle asynchronous conditions, we align the text time intervals with the audio time intervals to construct synchronous conditions for all baseline methods. The results are reported in Tab. 2.

Result Analysis. As shown in the table, our proposed method, MOCO, exhibits robust performance across both sets of metrics, delivering competitive results in both text-to-motion and speech-to-gesture tasks simultaneously. This underscores the efectiveness of MOCO in generating condition-aligned motions when multi-modal conditions are provided concurrently.

Weighted Sum and Pseudo-Text achieves good results on speech-to-gesture metrics but poor performance on text-to-motion metrics, indicating their limited ability to handle multi-modal data. We explain this further in Appendix E.

SynTalker also notices the importance of separating diferent body parts, resulting in an improvement in text-to-motion metrics compared to Weighted Sum and Pseudo-Text. However, its insuficient decoupling of conditions and suboptimal joint latent space—negatively impacted by repetitive patterns in the speech-to-gesture data—results in limited expressiveness and unstable movements. This is reflected in its less competitive performance in R1 and FID-A.

Beyond the baselines above, STMC [49] is particularly relevant, as its perstep composition paradigm inspired our multi-modal decoupled denoising. Since STMC is text-only and does not natively accept speech audio, we adapt it to our concurrent setting and conduct a dedicated comparison, including a user study (Appendix G.1 and Appendix G.5). There, STMC attains competitive text-to-motion scores, yet MOCO produces more natural and better-synchronized motion, as confirmed by quantitative metrics, additional naturalness metrics, and the subjective user study.

## 4.4 Ablation Study

To assess the impact of key designs within MOCO, we conduct an ablation study presented in Tab. 3. This study systematically examines the efects of body masking (Body Mask), weight sharing (Share Weight), and combination at each step (Per-Step Comb.) on the model’s performance across text-to-motion and speech-to-gesture metrics. In addition, we introduce the Transition Smoothness Ratio (TSR) to measure temporal coherence at segment boundaries, defined as the ratio between the mean frame-to-frame speed within transition windows and the mean speed in non-transition regions.

Body Masking. In Variants 1 and 2, we test our hypothesis that speech audio guides upper-body motion (head and arms) while text descriptions influence lower-body movements (spine and legs). In Variant 1, we expand the body mask to include the spine along with the head and arms (Body Mask = head, arms, spine). This modification results in a worse FID+ and a slight decrease in R1, indicating a decline in text-to-motion performance. Moreover, it does not produce significant improvements in speech-to-gesture metrics, suggesting that including the spine in the body mask fails to enhance gesture generation and instead compromises text-driven motion performance.

Variant 2 further adjusts the body mask to include the legs and spine (Body Mask = legs, spine), which significantly degrades text-to-motion metrics and Beat Consistency. This mainly stems from a mismatch: the texts in our multimodal benchmark describe diverse actions such as “walk,” “sit,” and “turn right,” whereas the speech-to-gesture data are mostly standing gestures. Controlling lower-body motion with audio makes it hard to align with these texts, while controlling upper-body motion with text complicates beat alignment. Although Variant 2 yields a notable improvement in FID-A—reflecting a bias in the speech-to-gesture data toward standing-in-place motions—its overall performance still degrades.

Table 3: Ablation study on key designs within MOCO. We highlight the best performance in bold and use underlining to indicate the second-best performance achieved by our method.
<table><tr><td>Method</td><td>Share Weight</td><td>Body Mask 1 - I</td><td>Per-Step Comb.</td><td>Text2Motion FID+ ↓ R1 ↑ M2T ↑</td><td></td><td>|FID-A ↓ BC ↑ L1div ↑</td><td>Speech2Gesture</td><td></td><td>TSR ↓</td></tr><tr><td>GT</td><td>-</td><td></td><td>=</td><td>0.000</td><td>40.0 0.781</td><td></td><td></td><td></td><td></td></tr><tr><td>MOCO</td><td>x</td><td>head, arms</td><td>√</td><td>0.862</td><td>24.6 0.649</td><td>3.83</td><td>2.72</td><td>8.62</td><td>0.90</td></tr><tr><td>Variant 1</td><td>x</td><td>head, arms, spine</td><td>√</td><td>0.921</td><td>22.4 0.634</td><td>3.86</td><td>2.81</td><td>8.35</td><td>0.90</td></tr><tr><td>Variant 2</td><td>x</td><td>spine, legs</td><td>√</td><td>1.234</td><td>7.9 0.554</td><td>2.75</td><td>2.19</td><td>5.11</td><td>0.95</td></tr><tr><td>Variant 3</td><td>√</td><td>head, arms</td><td>√</td><td>0.866</td><td>22.1 0.656</td><td>4.24</td><td>2.68</td><td>8.95</td><td>0.85</td></tr><tr><td>Variant 4</td><td>x</td><td>head, arms</td><td>x</td><td>0.832</td><td>24.4 0.645</td><td>3.99</td><td>2.27</td><td>8.60</td><td>1.17</td></tr></table>

In contrast, our original method (Body Mask = head, arms) efectively balances the influences of both text and audio inputs. By assigning the upper body to be guided by audio and the lower body by text, we achieve superior results across both text-to-motion and speech-to-gesture metrics. This demonstrates the advantage of our approach in producing coherent and contextually appropriate motions that align well with the provided conditions.

Weight Sharing. In Variant 3, we enable weight sharing (Share Weight = ✓), following previous multi-modal methods [12,37,70], while keeping the body mask and transition method unchanged. Compared to the full MOCO model (without weight sharing), enabling weight sharing results in worse performance across several metrics, including R1 and FID-A. This decline suggests that sharing weights between modalities may limit the model’s ability to capture modalityspecific nuances, thereby reducing its efectiveness in generating accurate and realistic motions for both text-to-motion and speech-to-gesture tasks.

Combination at Each Step. Variant 4 performs body-part combination only at the final denoising step t = 0 (Per-Step Comp. = ✗). Lacking iterative coordination to aggregate holistic motion information, this baseline exhibits temporal discontinuities (reflected by high TSR) and reduced body coherence. A user study (Sec. 4.6) and qualitative results in the supplementary videos further confirm that MOCO produces more natural and synchronized motions.

## 4.5 Qualitative Analysis

To clearly illustrate the overall performance of MOCO, we visualize four generated samples and their corresponding conditions in Fig. 4. The lighter color of the mesh and the text description background indicates the start of each sequence, while the darker color indicates the end. These results exhibit natural speechdriven upper-body gestures synchronized with diverse lower-body motions such as jogging, walking, and sitting, demonstrating MOCO’s ability to generate coherent and realistic motions that closely align with the given multimodal signals. A detailed analysis of these examples is provided in the figure caption. Additional qualitative results, including comparisons with baselines, are included in the supplementary materials.

![](images/896a7fbd6f21329b15d0447251ec2d804838e51b63061d2fba32aaaefbb28a6a.jpg)  
Fig. 4: Qualitative results. We visualize four samples generated by MOCO. Darker colors represent later points in time. The results demonstrate that MOCO is capable of generating coherent and realistic motions that highly align with the given multi-modal control signals. Figures (a), (b), and (d) present natural speech gestures coordinated with various lower-body movements as specified by the text inputs, such as jogging, walking in a circle, turning right, running, and so on. Figure (c) displays natural speech movements while sitting down. Figure (d) reveals a limitation of MOCO. When standing up or sitting down, the foot should remain stationary. However, the foot highlighted in the red box slides, leading to unrealistic results.

Table 4: User study results comparing MOCO against baseline methods.
<table><tr><td rowspan="2">Baseline</td><td colspan="3">Better Text Following (%)</td><td colspan="3">Better Beat Synchronization (%)</td></tr><tr><td>Neither</td><td>Baseline</td><td>MOCO (Ours)</td><td>Neither</td><td>Baseline</td><td>MOCO (Ours)</td></tr><tr><td>Pseudo-Text</td><td>1.0</td><td>0.0</td><td>99.0</td><td>13.0</td><td>12.5</td><td>74.5</td></tr><tr><td>Weighted Sum</td><td>0.0</td><td>0.0</td><td>100.0</td><td>13.5</td><td>3.5</td><td>83.0</td></tr><tr><td rowspan="2">Baseline</td><td colspan="3">Better Body Coherence (%)</td><td colspan="3">Better Temporal Fluidity (%)</td></tr><tr><td>Neither</td><td>Baseline</td><td>MOCO (Ours)</td><td>Neither</td><td>Baseline</td><td>MOCO (Ours)</td></tr><tr><td>Combine Once (Variant 4)</td><td>12.0</td><td>17.0</td><td>71.0</td><td>12.6</td><td>42.6</td><td>44.8</td></tr></table>

## 4.6 User Study

Tab. 4 presents the results of a user study comparing MOCO with three methods. Specifically, we evaluate MOCO against Pseudo-Text and Weighted Sum, introduced in Sec. 4.3, and Combine Once (Variant 4 in Sec. 4.4), to assess overall performance in text following and audio beat synchronization. Additional experimental details are provided in Appendix G.4.

As shown in the table, MOCO achieves significant advantages over both Pseudo-Text and Weighted Sum, demonstrating the efectiveness of our decoupled denoising strategy in generating motion aligned with multi-modal conditions. Furthermore, when compared to Combine Once, our method was rated significantly higher in both body coherence and temporal fluidity. This indicates that MOCO does more than merely combine diferent body parts controlled by separate conditions; it ensures that each body part aligns with its corresponding condition while enhancing coordination among all body parts. To further strengthen these findings, we additionally conduct a larger-scale pairwise preference study with 51 participants comparing MOCO against STMC, where MOCO is preferred with statistical significance on audio synchronization and naturalness $\left( p < 0 . 0 5 \right)$ ; full protocol and results are reported in the supplementary material.

## 5 Discussion and Limitations

MOCO achieves efective multi-modal motion generation, but its fixed bodypart assignment can be suboptimal when control signals conflict. For instance, when text descriptions specify arm movements (e.g., “wave hands”), this rigid separation leaves the upper body solely governed by audio, preventing the model from following text instructions involving arm motions. To mitigate this limitation, MOCO allows users to configure body-part control through the motion timeline (see Appendix D), enabling text to drive upper-body motion when needed.

Our multi-modal benchmark is also deliberately focused on compositional consistency under controlled settings rather than open-ended semantic coverage, enabling precise and repeatable evaluation of cross-modal integration but not claiming generalization to arbitrary text–audio pairs. Future work may explore adaptive body-part assignment or residual blending that dynamically handles spatially overlapping actions, moving beyond fixed spatial rules toward contextaware fusion while keeping the interpretability of decoupled generation.

## 6 Conclusion

This study presents MOCO, a novel difusion-based framework to generate realistic and coherent holistic body motions from multi-modal control signals, including text descriptions, speech audio, and trajectory data. Our key innovation lies in a decoupled denoising process where, during each denoising step, the model independently generates motions for each modality and assembles them according to predefined spatial rules. This approach ensures that the generated motion is closely aligned with each condition while producing realistic and coherent wholebody movements. Experimental results demonstrate that our approach delivers state-of-the-art performance both qualitatively and quantitatively, advancing the field of multi-modal controlled motion generation.

Acknowledgments. This work was supported by the National Natural Science Foundation of China under Grants 62476099 and 62076101, Guangdong Basic and Applied Basic Research Foundation under Grants 2024B1515020082 and 2023A1515010007, and the TCL Young Scholars Program.

## References

1. Alexanderson, S., Nagy, R., Beskow, J., Henter, G.E.: Listen, denoise, action! audiodriven motion synthesis with difusion models. ACM Transactions on Graphics 42(4), 1–20 (2023)

2. Ao, T., Gao, Q., Lou, Y., Chen, B., Liu, L.: Rhythmic gesticulator: Rhythm-aware co-speech gesture synthesis with hierarchical neural embeddings. ACM Transactions on Graphics 41(6), 1–19 (2022)

3. Ao, T., Zhang, Z., Liu, L.: Gesturedifuclip: Gesture difusion model with clip latents. ACM Transactions on Graphics 42(4), 1–18 (2023)

4. Athanasiou, N., Petrovich, M., Black, M.J., Varol, G.: Teach: Temporal action composition for 3d humans. In: International Conference on 3D Vision (3DV). pp. 414–423. IEEE (2022)

5. Athanasiou, N., Petrovich, M., Black, M.J., Varol, G.: Sinc: Spatial composition of 3d human motions for simultaneous action generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9984–9995 (2023)

6. Bae, J., Hwang, I., Lee, Y.Y., Guo, Z., Liu, J., Ben-Shabat, Y., Kim, Y.M., Kapadia, M.: Less is more: Improving motion difusion models with sparse keyframes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025)

7. Baevski, A., Zhou, Y., Mohamed, A., Auli, M.: wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems (NeurIPS) 33, 12449–12460 (2020)

8. Barquero, G., Escalera, S., Palmero, C.: Belfusion: Latent difusion for behaviordriven human motion prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2317–2327 (2023)

9. Barquero, G., Escalera, S., Palmero, C.: Seamless human motion composition with blended positional encodings. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 457–469 (2024)

10. Cen, Z., Pi, H., Peng, S., Shen, Z., Yang, M., Zhu, S., Bao, H., Zhou, X.: Generating human motion in 3d scenes from text descriptions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1855–1866 (2024)

11. Cen, Z., Pi, H., Peng, S., Shuai, Q., Shen, Y., Bao, H., Zhou, X., Hu, R.: Ready-toreact: Online reaction policy for two-character interaction generation. In: Interna tional Conference on Learning Representations (ICLR) (2025)

12. Chen, B., Li, Y., Ding, Y.X., Shao, T., Zhou, K.: Enabling synergistic full-body control in prompt-based co-speech motion generation. In: Proceedings of the ACM International Conference on Multimedia. pp. 6774–6783 (2024)

13. Chen, C., Zhang, J., Lakshmikanth, S.K., Fang, Y., Shao, R., Wetzstein, G., Fei-Fei, L., Adeli, E.: The language of motion: Unifying verbal and non-verbal language of 3d human motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025)

14. Chen, X., Jiang, B., Liu, W., Huang, Z., Fu, B., Chen, T., Yu, G.: Executing your commands via motion difusion in latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18000–18010 (2023)

15. Cong, P., Wang, Z., Dou, Z., Ren, Y., Yin, W., Cheng, K., Sun, Y., Long, X., Zhu, X., Ma, Y.: Laserhuman: language-guided scene-aware human motion generation in free environment. arXiv preprint arXiv:2403.13307 (2024)

16. Dai, W., Chen, L.H., Wang, J., Liu, J., Dai, B., Tang, Y.: Motionlcm: Realtime controllable motion generation via latent consistency model. In: European Conference on Computer Vision (ECCV). pp. 390–408. Springer (2024)

17. Fan, Y., Lin, Z., Saito, J., Wang, W., Komura, T.: FaceFormer: Speech-driven 3D facial animation with transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18770–18780 (2022)

18. Ginosar, S., Bar, A., Kohavi, G., Chan, C., Owens, A., Malik, J.: Learning individual styles of conversational gesture. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3497–3506 (2019)

19. Guo, C., Mu, Y., Javed, M.G., Wang, S., Cheng, L.: Momask: Generative masked modeling of 3d human motions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1900–1910 (2024)

20. Guo, C., Zou, S., Zuo, X., Wang, S., Ji, W., Li, X., Cheng, L.: Generating diverse and natural 3d human motions from text. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5152–5161 (2022)

21. Guo, C., Zuo, X., Wang, S., Zou, S., Sun, Q., Deng, A., Gong, M., Cheng, L.: Action2motion: Conditioned generation of 3d human motions. In: Proceedings of the ACM International Conference on Multimedia. pp. 2021–2029 (2020)

22. Gupta, P., Verma, S., Grama, A., Bera, A.: Unified multi-modal interactive & reactive 3d motion generation via rectified flow. In: International Conference on Learning Representations (ICLR) (2026)

23. Harithas, S., Sridhar, S.: Motionglot: A multi-embodied motion generation model. In: International Conference on Learning Representations (ICLR) (2025)

24. Hassan, M., Choutas, V., Tzionas, D., Black, M.J.: Resolving 3d human pose ambi guities with 3d scene constraints. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2282–2292 (2019)

25. Ho, J., Salimans, T.: Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022)

26. Hong, S., Kim, C., Yoon, S., Nam, J., Cha, S., Noh, J.: Salad: Skeleton-aware latent difusion for text-driven motion generation and editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7158–7168 (2025)

27. Hosseyni, S.R., Rahmani, A.A., Seyedmohammadi, S.J., Seyedin, S., Mohammadi, A.: Bad: Bidirectional auto-regressive difusion for text-to-motion generation. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2025)

28. Hu, L., Ye, Y., Xia, S.: Hmvlm: Human motion-vision-language model via moe lora. arXiv preprint arXiv:2511.01463 (2025)

29. Huang, S., Wang, Z., Li, P., Jia, B., Liu, T., Zhu, Y., Liang, W., Zhu, S.C.: Difusionbased generation, optimization, and planning in 3d scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16750–16761 (2023)

30. Huang, Y., Yang, H., Luo, C., Wang, Y., Xu, S., Zhang, Z., Zhang, M., Peng, J.: Stablemofusion: Towards robust and eficient difusion-based motion generation framework. In: Proceedings of the ACM International Conference on Multimedia. pp. 224–232 (2024)

31. Jiang, Y., Liao, Q., Lu, Z.: Smoothsync: Dual-stream difusion transformers for jitter-robust beat-synchronized gesture generation from quantized audio. arXiv preprint arXiv:2601.04236 (2026)

32. Karunratanakul, K., Preechakul, K., Suwajanakorn, S., Tang, S.: Guided motion difusion for controllable human motion synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2151–2162 (2023)

33. Lee, T., Baradel, F., Lucas, T., Lee, K.M., Rogez, G.: T2lm: Long-term 3d human motion generation from multiple sentences. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1867–1876 (2024)

34. Li, R., Yang, S., Ross, D.A., Kanazawa, A.: Ai choreographer: Music conditioned 3d dance generation with aist++. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 13401–13412 (2021)

35. Li, Z., Yuan, W., He, Y., Qiu, L., Zhu, S., Gu, X., Shen, W., Dong, Y., Dong, Z., Yang, L.T.: Lamp: Language-motion pretraining for motion generation, retrieval, and captioning. In: International Conference on Learning Representations (ICLR) (2025)

36. Liang, H., Zhang, W., Li, W., Yu, J., Xu, L.: Intergen: Difusion-based multi-human motion generation under complex interactions. International Journal of Computer Vision pp. 1–21 (2024)

37. Ling, Z., Han, B., Wong, Y., Kangkanhalli, M., Geng, W.: Mcm: Multi-condition motion synthesis framework for multi-scenario. arXiv preprint arXiv:2309.03031 (2023)

38. Liu, D.C., Nocedal, J.: On the limited memory bfgs method for large scale optimization. Mathematical programming 45(1), 503–528 (1989)

39. Liu, H., Iwamoto, N., Zhu, Z., Li, Z., Zhou, Y., Bozkurt, E., Zheng, B.: Disco: Disentangled implicit content and rhythm learning for diverse co-speech gestures synthesis. In: Proceedings of the ACM International Conference on Multimedia. pp. 3764–3773 (2022)

40. Liu, H., Zhu, Z., Becherini, G., Peng, Y., Su, M., Zhou, Y., Iwamoto, N., Zheng, B., Black, M.J.: Emage: Towards unified holistic co-speech gesture generation via masked audio gesture modeling. arXiv preprint arXiv:2401.00374 (2023)

41. Liu, H., Zhu, Z., Iwamoto, N., Peng, Y., Li, Z., Zhou, Y., Bozkurt, E., Zheng, B.: Beat: A large-scale semantic and emotional multi-modal dataset for conversational gestures synthesis. In: European Conference on Computer Vision (ECCV). pp. 612–630. Springer (2022)

42. Liu, P., Song, L., Huang, J., Liu, H., Xu, C.: Gesturelsm: Latent shortcut based co-speech gesture generation with spatial-temporal modeling. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025)

43. Liu, X., Wu, Q., Zhou, H., Xu, Y., Qian, R., Lin, X., Zhou, X., Wu, W., Dai, B., Zhou, B.: Learning hierarchical cross-modal association for co-speech gesture generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10462–10472 (2022)

44. Liu, Y., Cao, Q., Wen, Y., Jiang, H., Ding, C.: Towards variable and coordinated holistic co-speech motion generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1566–1576 (2024)

45. Liu, Y., Cao, Q., Yi, H., Jiang, H., Ding, C.: Multi-modal controlled coherent motion synthesis. OpenReview (2024), https://openreview.net/forum?id=i5Gxilzk0u

46. Liu, Y., Chen, C., Yi, L.: Interactive humanoid: Online full-body motion reaction synthesis with social afordance canonicalization and forecasting. ArXiv abs/2312.08983 (2023), https://api.semanticscholar.org/CorpusID: 266209846

47. Ma, S., Cao, Q., Zhang, J., Tao, D.: Contact-aware human motion generation from textual descriptions. arXiv preprint arXiv:2403.15709 (2024)

48. Peng, Y., Song, J.T., Jung, S., Liu, R., Liu, H., Chu, X., Liu, R., Wu, E., Koike, H., Kitani, K.: Dyadit: A multi-modal difusion transformer for socially favorable dyadic gesture generation. arXiv preprint arXiv:2602.23165 (2026)

49. Petrovich, M., Litany, O., Iqbal, U., Black, M.J., Varol, G., Peng, X.B., Rempe, D.: Multi-track timeline control for text-driven 3d human motion generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1911–1921 (2024)

50. Qi, X., Zhang, H., Wang, Y., Pan, J., Liu, C., Sun, M., Xue, W., Zhang, S., Han, S., Liu, Q., et al.: Cocogesture: Towards coherent co-speech 3d gesture generation in the wild. Information Fusion p. 103613 (2025)

51. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML). pp. 8748–8763. PMLR (2021)

52. Sampieri, A., Palma, A., Spinelli, I., Galasso, F.: Length-aware motion synthesis via latent difusion. In: European Conference on Computer Vision (ECCV). pp. 107–124. Springer (2024)

53. Sha, X., Zhang, L., Mashita, T., Chiba, N., Uranishi, Y.: 3dgespolicy: Phonemeaware holistic co-speech gesture generation based on action control. arXiv preprint arXiv:2601.18451 (2026)

54. Shafir, Y., Tevet, G., Kapon, R., Bermano, A.H.: Human motion difusion as a generative prior. arXiv preprint arXiv:2303.01418 (2023)

55. Siyao, L., Yu, W., Gu, T., Lin, C., Wang, Q., Qian, C., Loy, C.C., Liu, Z.: Bailando: 3D dance generation by actor-critic GPT with choreographic memory. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11050–11059 (2022)

56. Sun, J., Zhang, Q., Duan, Y., Jiang, X., Cheng, C., Xu, R.: Prompt, plan, perform: Llm-based humanoid control via quantized imitation learning. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 16236–16242. IEEE (2024)

57. Sun, M., Xu, C., Jiang, X., Liu, Y., Sun, B., Huang, R.: Beyond talking–generating holistic 3d human dyadic motion for communication. International Journal of Computer Vision 133(5), 2910–2926 (2025)

58. Tevet, G., Raab, S., Cohan, S., Reda, D., Luo, Z., Peng, X.B., Bermano, A.H., van de Panne, M.: Closd: Closing the loop between simulation and difusion for multi-task character control. In: International Conference on Learning Representations (ICLR) (2025)

59. Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., Bermano, A.H.: Human motion difusion model. arXiv preprint arXiv:2209.14916 (2022)

60. Tseng, J., Castellon, R., Liu, K.: Edge: Editable dance generation from music. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 448–458 (2023)

61. Wang, H., Zhu, W., Miao, L., Xu, Y., Gao, F., Tian, Q., Wang, Y.: Aligning human motion generation with human perceptions. In: International Conference on Learning Representations (ICLR). vol. 2025, pp. 77626–77656 (2025)

62. Wang, Y., Wang, S., Zhang, J., Fan, K., Wu, J., Xue, Z., Liu, Y.: Timotion: Temporal and interactive framework for eficient human-human motion generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7169–7178 (2025)

63. Wang, Y., Li, M., Liu, J., Leng, Z., Li, F.W., Zhang, Z., Liang, X.: Fg-t2m++: Llms-augmented fine-grained text driven human motion generation. International Journal of Computer Vision pp. 1–17 (2025)

64. Wang, Z., Chen, Y., Jia, B., Li, P., Zhang, J., Zhang, J., Liu, T., Zhu, Y., Liang, W., Huang, S.: Move as you say interact as you can: Language-guided human motion generation with scene afordance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 433–444 (2024)

65. Wang, Z., Wang, J., Lin, D., Dai, B.: Intercontrol: Generate human motion interactions by controlling every joint. arXiv preprint arXiv:2311.15864 (2023)

66. Xie, Y., Jampani, V., Zhong, L., Sun, D., Jiang, H.: Omnicontrol: Control any joint at any time for human motion generation. arXiv preprint arXiv:2310.08580 (2023)

67. Xing, J., Xia, M., Zhang, Y., Cun, X., Wang, J., Wong, T.T.: Codetalker: Speechdriven 3d facial animation with discrete motion prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12780–12790 (2023)

68. Xu, Z., Lin, Y., Han, H., Yang, S., Li, R., Zhang, Y., Li, X.: Mambatalk: Eficient holistic gesture synthesis with selective state space models. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)

69. Yang, S., Wu, Z., Li, M., Zhang, Z., Hao, L., Bao, W., Cheng, M., Xiao, L.: Difusestylegesture: Stylized audio-driven co-speech gesture generation with difusion models. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI-23). pp. 5860–5868 (2023)

70. Yang, S., Xu, Z., Xue, H., Cheng, Y., Huang, S., Gong, M., Wu, Z.: Freetalker: Controllable speech and text-driven gesture generation based on difusion models for enhanced speaker naturalness. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 7945–7949. IEEE (2024)

71. Yi, H., Liang, H., Liu, Y., Cao, Q., Wen, Y., Bolkart, T., Tao, D., Black, M.J.: Generating holistic 3d human motion from speech. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 469–480 (2023)

72. Yi, H., Thies, J., Black, M.J., Peng, X.B., Rempe, D.: Generating human interaction motions in scenes with text control. arXiv preprint arXiv:2404.10685 (2024)

73. Yoon, Y., Cha, B., Lee, J.H., Jang, M., Lee, J., Kim, J., Lee, G.: Speech gesture generation from the trimodal context of text, audio, and speaker identity. ACM Transactions on Graphics 39(6), 1–16 (2020)

74. Zhang, J., Zhang, Y., Cun, X., Zhang, Y., Zhao, H., Lu, H., Shen, X., Shan, Y.: Generating human motion from textual descriptions with discrete representations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14730–14740 (2023)

75. Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., Liu, Z.: Motiondiffuse: Text-driven human motion generation with difusion model. arXiv preprint arXiv:2208.15001 (2022)

76. Zhang, M., Jin, D., Gu, C., Hong, F., Cai, Z., Huang, J., Zhang, C., Guo, X., Yang, L., He, Y., et al.: Large motion model for unified multi-modal motion generation. In: European Conference on Computer Vision (ECCV). pp. 397–421. Springer (2024)

77. Zhang, M., Li, H., Cai, Z., Ren, J., Yang, L., Liu, Z.: Finemogen: Fine-grained spatio-temporal motion generation and editing. Advances in Neural Information Processing Systems (NeurIPS) 36, 13981–13992 (2023)

78. Zhang, P., Liu, P., Kim, H., Garrido, P., Chaudhuri, B.: Kinmo: Kinematic-aware human motion understanding and generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025)

79. Zhang, Q., Song, J., Huang, X., Chen, Y., Liu, M.Y.: Difcollage: Parallel generation of large content with difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10188–10198. IEEE (2023)

80. Zhang, X., Li, J., Zhang, J., Dang, Z., Ren, J., Bo, L., Tu, Z.: Semtalk: Holistic co-speech motion generation with frame-level semantic emphasis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025)

81. Zhang, Z., Wang, Y., Mao, W., Li, D., Zhao, R., Wu, B., Song, Z., Zhuang, B., Reid, I., Hartley, R.: Motion anything: Any to motion generation. arXiv preprint arXiv:2503.06955 (2025)

82. Zhou, Z., Wan, Y., Wang, B.: A unified framework for multimodal, multi-part human motion synthesis. ArXiv abs/2311.16471 (2023), https://api.semanticscholar. org/CorpusID:265466120

83. Zhou, Z., Wang, B.: Ude: A unified driving engine for human motion generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5632–5641 (2023)

84. Zhu, L., Liu, X., Liu, X., Qian, R., Liu, Z., Yu, L.: Taming difusion models for audiodriven co-speech gesture generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10544–10553 (2023)

## A Body-Part Motion Statistics

A core design principle of MOCO is the spatial decomposition of multi-modal conditioning: text descriptions govern lower-body motion, while speech audio drives upper-body dynamics. This appendix provides a quantitative characterization of the motion distributions associated with each conditioning modality within the scope of our target task, ofering an empirical basis for understanding why this decomposition is well-suited to the concurrent text-and-speech control scenario.

## Task-Scoped Text Conditions and Motivation for Generated Samples

Our target application involves concurrent control by both speech audio and text descriptions. In this setting, text conditions are expected to specify body-level actions that are spatially complementary to co-speech gestures—that is, actions primarily activating the lower body without conflicting with concurrent upperbody gesture generation. We therefore focus our analysis on a curated set of 40 text descriptions centered on locomotion and postural transitions (see Appendix H), which represent the intended operating domain of the text conditioning signal in MOCO. These descriptions are representative of the text conditions under which MOCO is designed to operate, and do not aim to characterize the full diversity of text-driven motion.

To obtain the motion distribution for these curated descriptions, we generate 1,000 sequences using a pretrained text-to-motion model (MDM [59]) trained on HumanML3D [20], as these specific lower-body–oriented descriptions do not have corresponding ground-truth annotations in existing datasets. This approach allows us to directly estimate the body-part activation patterns associated with the text conditions employed in our multi-modal benchmark.

## Setup

Both data sources are represented using the unified 205-dimensional SMPL-X body parameterization. Body parts are defined via fixed joint indices, grouped into five regions: left arm, right arm, head, spine, and legs. For each motion sequence, we compute the temporal variance of each body-part region as a proxy for motion activity, and normalize across all five parts so that they sum to 100% per sequence. We further aggregate upper-body parts (head, left arm, right arm) and lower-body parts (spine, legs), and define the lower-body ratio as:

$$
r _ { \mathrm { l o w e r } } = \frac { \mathrm { V a r } _ { \mathrm { l o w e r } } } { \mathrm { V a r } _ { \mathrm { u p p e r } } + \mathrm { V a r } _ { \mathrm { l o w e r } } } ,\tag{11}
$$

where $\mathrm { V a r } _ { \mathrm { u p p e r } }$ and $\mathrm { V a r } _ { \mathrm { l o w e r } }$ denote the aggregated temporal variance of upperand lower-body joints, respectively. This normalization eliminates confounds from sequence length and global motion scale, enabling a fair comparison across datasets.

Statistics are computed over two sources:

– Generated (text-driven): 1,000 sequences produced by a pretrained textto-motion difusion model conditioned on our curated set of 40 locomotionand posture-oriented descriptions.

– BEAT2 (speech-driven): 5,000 sequences sampled from the BEAT2 training set [40].

## Results

Quantitative results are reported in Tab. 5. Under our task-scoped text conditions, the generated motions exhibit a lower-body variance share of 62.0%, indicating that locomotion- and posture-oriented descriptions predominantly activate the legs and spine. In contrast, BEAT2 speech-gesture sequences are strongly upperbody dominant, with 85.1% of motion variance concentrated in the head and arms—consistent with the expressive, gesture-rich nature of co-speech motion. These complementary distributions provide empirical support for our proposed spatial decomposition.

Table 5: Body-part motion statistics for text-driven generated sequences and BEAT2 speech-gesture data. Temporal variance is computed per body-part region and normalized within each sequence.
<table><tr><td>Source</td><td>Upper</td><td>Lower MeanVar MeanVar</td><td>Lower / Upper</td><td>Lower-Body Ratio</td></tr><tr><td>Generated (text-driven)</td><td>0.00387</td><td>0.00663</td><td>1.71×</td><td>62.0%</td></tr><tr><td>BEAT2 (speech-driven)</td><td>0.00587</td><td>0.00164</td><td>0.28×</td><td>14.9%</td></tr></table>

## B Theoretical Analysis for Decoupled Denoising

Our proposed MOCO relies on the assumption that the joint conditional probability $p ( b _ { t - 1 } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t } )$ can be approximated by $p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { t e x t } } , b _ { t } )$ $p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a u d i o } } , b _ { t } )$ , expressed as:

$$
\begin{array} { r l } & { p ( b _ { t - 1 } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t } ) } \\ & { ~ \approx p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { t e x t } } , b _ { t } ) \cdot p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a u d i o } } , b _ { t } ) , } \end{array}\tag{12}
$$

where $b _ { t }$ denotes the motion at denoising step $t ,$ composed of upper-body motion $b _ { t , \mathrm { u p p e r } }$ and lower-body motion $b _ { t , \mathrm { l o w e r } } .$

$$
p ( b _ { t - 1 } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t } )
$$

$$
= p ( b _ { t - 1 , \mathrm { l o w e r } } , b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } ) , \mathrm { w h e r e } c _ { \mathrm { a l l } } = \{ c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t } \}
$$

$$
= p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { a l l } } ) \cdot p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } , b _ { t - 1 , \mathrm { l o w e r } } )\tag{13}
$$

$$
\approx p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { a l l } } ) \cdot p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } )\tag{14}
$$

$$
\approx p ( b _ { t - \mathrm { 1 , l o w e r } } \ | \ c _ { \mathrm { a l l } } \setminus \{ c _ { \mathrm { a u d i o } } \} ) \cdot p ( b _ { t - \mathrm { 1 , u p p e r } } \ | \ c _ { \mathrm { a l l } } \setminus \{ c _ { \mathrm { t e x t } } \} )\tag{15}
$$

$$
= p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { t e x t } } , b _ { t } ) \cdot p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a u d i o } } , b _ { t } ) .
$$

The first approximation occurs in the transition from Eq. (13) to $\mathrm { E q . ~ ( 1 4 ) }$ Here, we approximate:

$$
\begin{array} { r l } & { p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } , b _ { t - 1 , \mathrm { l o w e r } } ) } \\ & { ~ = p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t , \mathrm { u p p e r } } , b _ { t , \mathrm { l o w e r } } , b _ { t - 1 , \mathrm { l o w e r } } ) } \\ & { ~ \approx p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { t e x t } } , c _ { \mathrm { a u d i o } } , b _ { t , \mathrm { u p p e r } } , b _ { t , \mathrm { l o w e r } } ) } \\ & { ~ = p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } ) . } \end{array}
$$

This approximation assumes that $b _ { t }$ already encapsulates suficient information about $b _ { t - 1 }$ , allowing us to neglect the influence of $b _ { t - 1 , \mathrm { l o w e r } }$ when estimating $b _ { t - 1 , \mathrm { u p p e r } }$ . This simplification is justified by the proximity of the difusion steps and the strong correlation between the states at steps t and $t - 1$

The second approximation occurs in the transition from Eq. (14) to Eq. (15), where we decouple modality-specific influences:

$$
\begin{array} { r l } & { p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { a l l } } ) \approx p ( b _ { t - 1 , \mathrm { l o w e r } } \mid c _ { \mathrm { a l l } } \setminus \{ c _ { \mathrm { a u d i o } } \} ) , } \\ & { p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } ) \approx p ( b _ { t - 1 , \mathrm { u p p e r } } \mid c _ { \mathrm { a l l } } \setminus \{ c _ { \mathrm { t e x t } } \} ) . } \end{array}
$$

This approximation leverages the observation that text input $\left( c _ { \mathrm { t e x t } } \right)$ primarily influences lower-body movements $( e . g .$ , walking or shifting stance), while audio input $\left( { \boldsymbol { c } } _ { \mathrm { a u d i o } } \right)$ predominantly afects upper-body movements $( e . g .$ , gestures or facial expressions). By excluding $c _ { \mathrm { a u d i o } }$ from the conditioning set for $b _ { t - 1 }$ ,lower and $c _ { \mathrm { t e x t } }$ for $b _ { t - 1 , \mathrm { u p p e r } }$ , we ensure the conditioning focuses on the most relevant modality for each body part.

## C Leveraging LLM for Motion Planning

In summary, our prompt instructing the LLM to generate a motion timeline consists of several key components:

1. Purpose: Clearly state the objective of generating a motion timeline based on text and audio conditions.

2. Output Formats: Specify the formats that the LLM should adhere to when producing output.

3. Guidelines for Generating Timelines: Outline several essential rules that the LLM must follow.

4. Examples of Timelines: Provide a pair of example timelines, showcasing both an efective and a less efective version.

Additionally, to enhance the precision and completeness of the LLM’s reasoning, we append the phrase “please reason step by step” at the end of the dialogue.

To illustrate this process more clearly, we provide a complete dialogue record with GPT-4o mini in motion-planning-dialogue.pdf. In this dialogue, the input conditions are processed by the LLM as follows:

<table><tr><td>Condition</td><td>Start Time End Time Body Part</td><td></td><td></td></tr><tr><td>sit down on a chair</td><td>0.0</td><td>5.0</td><td>legs, spine</td></tr><tr><td>stand up from chair</td><td>5.0</td><td>7.0</td><td>legs, spine</td></tr><tr><td>turn around</td><td>7.0</td><td>11.5</td><td>legs, spine</td></tr><tr><td>walk forward</td><td>11.5</td><td>17.5</td><td>legs, spine</td></tr><tr><td>5 stewart 0 1 1,$785440$911328</td><td>5.0</td><td>12.85</td><td>head, arms</td></tr><tr><td>5 stewart 0 1 1,$918560$1050080</td><td>17.5</td><td>24.3</td><td>head, arms</td></tr></table>

## Explanation:

– Text Description: “sit down on a chair”

• Start and End Time in Overall Motion: # 0.0 # 5.0

• Controlled Body Parts: $\#$ legs # spine

– Audio Input File: speak:5\_stewart\_0\_1\_1

• Start and End Frames in Audio File: \$785440\$911328

• Start and End Time in Overall Motion: # 5.0 # 12.85

• Controlled Body Parts: $\#$ head $\#$ arms

Notable, the LLM may occasionally produce calculation errors, such as miscalculating the duration of the second audio segment. However, these issues can be easily resolved through code refinement. The motion generated by our MOCO based on the refined timeline is provided in motion-planning-result.mp4.

## D Conflict Resolution and Flexibility in Multi-Modal Control

MOCO ’s default configuration assigns upper-body control to audio signals and lower-body control to text descriptions. However, there are scenarios where users may wish to explicitly control specific upper-body motions, such as waving hands while talking. To clarify how our method handles this case, we provide the following example:

In this example, a conflict arises between the upper-body motion cues from the text description “waves hands” and the speech audio during the overlapping period from 2.578 seconds to 8.0 seconds. To resolve such conflicts, our method relies on a predefined timeline that gives precedence to the speech audio in governing upper-body motion during this interval. Consequently, the ‘waves hands” motion is suppressed between 2.578 and 8.0 seconds.

<table><tr><td>Condition</td><td colspan="2">Start Time End Time</td><td>Body Part</td></tr><tr><td>a man walks then waves hands</td><td>0.0</td><td>8.0</td><td>legs, spine</td></tr><tr><td>speak:1_wayne_0_103_103,$9248$99808</td><td>2.578</td><td>8.238</td><td>left arm, right arm, head</td></tr><tr><td>speak:1wayne_0103103,$107552$196064</td><td>9.722</td><td>15.254</td><td>left arm, right arm, head</td></tr></table>

While this illustrates a limitation of our default setup, the flexibility of MOCO ofers a way to address such conflicts by enabling finer-grained user control. If users wish to control upper-body motions using text during speech, we can simply adjust the timeline as follows:

<table><tr><td>Condition</td><td colspan="2">Start Time End Time</td><td>Body Part</td></tr><tr><td>a man walks</td><td>0.0</td><td>8.0</td><td>legs, spine</td></tr><tr><td>wave hands</td><td>7.0</td><td>11.0</td><td>left arm, right arm</td></tr><tr><td>speak:1 wayne 0 103 103,$9248$99808</td><td>2.578</td><td>8.238</td><td>left arm, right arm, head</td></tr><tr><td>speak:1wayne0103103,$107552$196064</td><td>9.722</td><td>15.254</td><td>left arm, right arm, head</td></tr></table>

After modifying the timeline, the text command “wave hands” controls the arms from 7.0 to 11.0 seconds. Since the speech audio spans from 2.578 to 15.254 seconds, there is an overlap between 7.0 and 11.0 seconds during which both audio and text attempt to control the arms. According to our conflict resolution rules, the condition with fewer controlled body parts takes precedence. In this case, the text command “wave hands” governs the arms during the overlapping period.

This example demonstrates the flexibility of our method. Although our default configuration prioritizes audio for the upper body and text for the lower body, users can easily reconfigure these priorities through timeline adjustments to achieve their desired behavior. The corresponding visualization is provided in the supplementary materials as flexibility.mp4.

## E Limitations of Weighted Sum in Multi-Modal Motion Generation

To understand why the Weighted Sum method achieves good results in speech-togesture metrics but poor performance in text-to-motion metrics, we conducted the following experiments.

Given both speech and text inputs, we used only the text-to-motion model to update the motion. At each denoising step t, we computed the diference dif<sub>t</sub> between the speech-to-gesture model’s prediction—based on the speech input and the current motion from the text-to-motion model—and the current motion from the text-to-motion model. This diference quantifies how much the speech-to-gesture model perceives a mismatch between the speech condition and the current motion. Conversely, when we used only the speech-to-gesture model to update the motion, the calculated diference indicated how much the text-to-motion model perceived a mismatch between the text condition and the current motion. A larger diference suggests a greater mismatch and that the model will update the motion more aggressively.

![](images/3f73321fbeeecce043486a413655525572983a50225f74a71731b0e9d0c88228.jpg)  
(a) T, text =“standing”

![](images/cec5dc2b15de906f88390e5af72cab55b006389c5b32232963794c84ead12f39.jpg)  
(b) A, text=“standing"

![](images/4757ce7ea64fdb066eda95a40b43cc3291c4f6216e3195c277b5fc2039b17bf9.jpg)  
(c) T, text=“sitting”

![](images/93d5752c2ae003fb35f295dfaedaa99244c4b19c8513194dc82f7b53358dcd3e.jpg)  
(d) A, text=“sitting”  
Fig. 5: Comparison of diferences calculated by the speech-to-gesture and text-to-motion models during motion updates. “T” denotes using a text-to-motion model to update motion, while “A” denotes using a speech-to-gesture model to update motion. The results show that the speech-to-gesture model computes larger diferences than the text-to-motion model, indicating it adjusts the motion more aggressively based on the conditions. This explains why the weighted sum method attains strong performance on speech-to-gesture metrics but poor performance on text-to-motion metrics. Additionally, when the text condition is “sitting,” the speech-to-gesture model calculates larger diferences in the legs than in the arms. This is counterintuitive and could be attributed to data bias in the speech-to-motion dataset. Best viewed in color.

We recorded these diferences in both scenarios and divided them into whole body, arms, and legs for clearer illustration. Comparing Fig. 5 (a) and (b), as well as Fig. 5 (c) and (d), we found that the diferences calculated by the speech-togesture model are larger than those by the text-to-motion model. This indicates that the speech-to-gesture model adjusts the motion more aggressively based on its conditions than the text-to-motion model does. This explains why, when using the weighted sum method, the generated result closely resembles that produced entirely by the speech-to-gesture model.

Furthermore, by comparing Fig. 5 (a) and (c), which have diferent text conditions, we observe that when the text condition is “sitting,” the diferences calculated by the speech-to-gesture model in the legs are larger than in the arms. This is counterintuitive since speech is typically associated with upperbody gestures rather than lower-body movements. Conversely, when the text condition is “standing,” the diferences in the legs are smaller than in the arms, aligning with expectations. This phenomenon may be attributed to data bias in the speech-to-motion dataset, where most motions are performed in standing positions.

These observations reveal the limitations of weighted sum in multi-modal motion generation and suggest the validity of our proposed decoupled denoising process.

## F Computational Complexity

Table 6: Complexity of each denoiser of MOCO.
<table><tr><td colspan="2">Parameters Model Size FLOPs Inference Time (M)</td><td>(MB) (G)</td><td> $\mathrm { ( m s / f r a m e ) }$ </td></tr><tr><td rowspan="3"> $G _ { \mathrm { T 2 M } }$   $G _ { \mathrm { S 2 G } }$   $G _ { \mathrm { T 2 V } }$ </td><td>27.01</td><td>103.02 5.19</td><td>2.26</td></tr><tr><td>36.86</td><td>140.62 6.72</td><td>4.30</td></tr><tr><td>0.34</td><td>1.31 0.06 6.74</td><td>6.20</td></tr><tr><td> $G _ { \mathrm { S 2 D } }$ </td><td>36.94</td><td>140.94</td><td>4.37</td></tr></table>

Our framework, MOCO, comprises four transformer-based denoisers: $G _ { \mathrm { T 2 M } }$ for text-to-motion (T2M), $G _ { \mathrm { S 2 G } }$ for speech-to-gesture (S2G), $G _ { \mathrm { T 2 V } }$ for trajectoryto-velocity (T2V), and $G _ { \mathrm { S 2 D } }$ for speech-to-details (S2D), which manages facial expressions and hand poses. To clearly illustrate the computational complexity of MOCO, we present various metrics, including the number of parameters, model size, $\mathrm { F L O P s }$ , and inference time on a single NVIDIA 4090 GPU, as shown in Tab. 6.

As indicated in the table, our framework is overall lightweight and suficiently fast. Specifically, the speech-to-gesture denoiser $G _ { \mathrm { S 2 G } }$ and the speech-to-details denoiser $G _ { \mathrm { S 2 D } }$ are relatively larger than the other denoisers due to additional cross-attention parameters. In contrast, the trajectory-to-velocity denoiser $G _ { \mathrm { T 2 V } }$ is the most lightweight module, featuring fewer hidden state dimensions and transformer layers because the task it handles involves low-dimensional data. However, the introduction of a guidance mechanism for more accurate predictions results in $G _ { \mathrm { T 2 V } }$ having the longest inference time.

Finally, to generate the motion sequences for a 35-second demo video consisting of nine clips under diferent conditions and with a total duration of 54 seconds, our method completed the body motion generation task in only 3.72 seconds. This fast generation time highlights the potential of our approach for real-time applications.

For a like-for-like comparison against competing methods, we additionally report end-to-end generation speed (seconds per sequence) under the concurrent multi-modal setting in Tab. 7 (Appendix G.1): MOCO is the fastest at $0 . 6 9 \mathrm { s } / \mathrm { s e q }$ , ahead of STMC (0.75 s/seq) and the StableMoFusion-backbone variant $( 0 . 8 6 \mathrm { s / s e q } )$ , confirming that MOCO attains its naturalness and synchronization advantages without incurring additional inference cost.

## G Additional Experiments

## G.1 Comparison with STMC and Stronger Backbones

This section presents experiments comparing MOCO against two additional baselines, with results reported in Tab. 7. We include STMC [49] because its singlemodality-per-step composition inspired our multi-modal decoupled denoising, and StableMoFusion [30] to examine whether a backbone that is stronger on text-to-motion transfers into improved motion quality in our concurrent control setting. To adapt STMC to the multi-modal setting while faithfully preserving its original design, we use self-attention to process both text and audio conditions, share parameters across denoisers, and rely on an LLM to automatically annotate the assignment of body parts. In addition to the text-to-motion (FID+, R1) and speech-to-gesture (FID-A, BC) metrics used in the main paper, we further report two naturalness-oriented metrics: MotionCritic (MC) [61], a learned motionquality score aligned with human perception, and $\mathrm { J i t t e r _ { s p i n e } }$ , the jerk of the spine joints at the boundary between the audio-controlled upper body and the text-controlled lower body, which quantifies how smoothly the two independently denoised regions are stitched together.

Table 7: Comparison with per-step composition (STMC) and a stronger difusion backbone (StableMoFusion). Best among generative methods is in bold.
<table><tr><td>Method</td><td>Naturalness MC↑</td><td> $\mathrm { J i t t e r } _ { \mathrm { s p i n e } } \downarrow$ </td><td>Text2Motion FID+↓ R1↑</td><td></td><td>Speech2Gesture FID-A↓ BC↑</td><td>Speed  $\mathrm { s / s e q \downarrow }$ </td></tr><tr><td>GT</td><td>-1.95</td><td>0.59</td><td>0.000</td><td>40.0</td><td></td><td></td></tr><tr><td>MOCO (Ours)</td><td>-3.39</td><td>0.60</td><td>0.862</td><td>24.6 3.83</td><td>2.72</td><td>0.69</td></tr><tr><td>STMC [49]</td><td>-3.84</td><td>0.63</td><td>0.684</td><td>27.5 4.44</td><td>2.22</td><td>0.75</td></tr><tr><td>StableMoFusion [30]</td><td>-3.91</td><td>0.69</td><td>0.752</td><td>27.2</td><td>4.83 2.92</td><td>0.86</td></tr></table>

Table 8: Evaluation of trajectory control under classifier-free guidance (CFG), Omni Control (OC) [66], and post-hoc L-BFGS optimization, on location (m) and orientation (rad). OmniControl is a dedicated spatial-control method used to replace our trajectoryto-velocity module $G _ { \mathrm { T 2 V } } ;$ the first three rows therefore use G<sub>T2V</sub>.
<table><tr><td colspan="2">Method</td><td colspan="2">Location</td><td colspan="2">Orientation</td></tr><tr><td>CFG OC L-BFGS</td><td></td><td>Average Difference Difference</td><td>Goal</td><td>Average Difference Difference</td><td>Goal</td></tr><tr><td>X</td><td>X</td><td>x</td><td>0.56 1.21</td><td>0.71</td><td>1.26</td></tr><tr><td>√</td><td>x x</td><td>0.57</td><td>1.32</td><td>0.83</td><td>1.51</td></tr><tr><td>√</td><td>x √</td><td>0.11</td><td>0.20</td><td>0.71</td><td>1.28</td></tr><tr><td>x</td><td>√ x</td><td>0.44</td><td>0.79</td><td>0.53</td><td>0.97</td></tr><tr><td>x</td><td>x √</td><td>0.08</td><td>0.13</td><td>0.59</td><td>1.10</td></tr><tr><td>x</td><td>√</td><td>V 0.14</td><td>0.20</td><td>0.48</td><td>0.87</td></tr></table>

We report the results transparently. On pure text-to-motion fidelity, STMC is in fact slightly stronger than MOCO, and StableMoFusion is likewise competitive on these metrics. However, MOCO achieves better speech-to-gesture fidelity, boundary smoothness, and MotionCritic score. These results suggest that, rather than being uniformly best on every isolated single-modality metric, MOCO delivers the most natural and best-synchronized motion under joint text-andspeech control, as further corroborated by the pairwise user study in Appendix G.5. We note that MC should be interpreted only as an auxiliary indicator, because it is trained predominantly on text-driven motion, whereas MOCO produces dense, expressive co-speech upper-body gestures that may lie partly outside its training distribution.

## G.2 Evaluation of Trajectory Control

Tab. 8 evaluates trajectory control methodologies by assessing the efects of classifier-free guidance (CFG), OmniControl (OC) [66], and L-BFGS optimization on both location (meters) and orientation (radians). Here, OmniControl is a dedicated spatial-control method that replaces our lightweight trajectory-tovelocity module $G _ { \mathrm { T 2 V } }$ , whereas the other configurations build on $G _ { \mathrm { T 2 V } }$ . For each category, two metrics are reported: Average Diference, the mean deviation between the generated trajectory and the ground truth (GT), and Goal Diference, the discrepancy at the final point relative to the GT.

The post-hoc L-BFGS optimization is the single most efective component: applied on top of $G _ { \mathrm { T 2 V } }$ , it drastically reduces the location error while also modestly improving orientation. In contrast, CFG does not help and can even interfere with the optimization, slightly worsening both location and orientation. Replacing $G _ { \mathrm { T 2 V } }$ with OmniControl improves orientation, and OC+L-BFGS achieves the overall best orientation—but at the cost of a larger location error and additional inference time. We therefore retain $G _ { \mathrm { T 2 V } } + \mathrm { L } \mathrm { - B F G S }$ as our default for its better location accuracy and eficiency, and note that adopting a dedicated spatial controller is a worthwhile direction when orientation fidelity is prioritized.

We further emphasize, in the interest of transparency, that the absolute orientation accuracy of MOCO is still not fully satisfactory. We attribute this to a deliberate design trade-of: MOCO adopts a motion representation chosen to facilitate temporal stitching, which is prone to accumulated error along the trajectory—hence the orientation error remains comparatively large even after optimization.

## G.3 Single Modality Performance

To demonstrate MOCO’s performance in single-modality scenarios, we trained it from scratch on HumanML3D for text-to-motion and on BEAT2 for speechto-gesture, respectively, ensuring a fair comparison. The results, presented in Tab. 9 and Tab. 10, show that in the HumanML3D text-to-motion benchmark (Tab. 9), our model achieves performance comparable to the widely-used MDM. This outcome is expected since our text-to-motion denoiser, $G _ { \mathrm { T 2 M } }$ , is based on MDM. In the BEAT2 speech-to-gesture benchmark (Tab. 10), MOCO attains competitive performance compared to state-of-the-art methods.

Table 9: Quantitative results of text-to-motion generation on the HumanML3D test set.
<table><tr><td>Method</td><td>R-Precision Top 1 Top 2 Top 3</td><td>FID↓ MM Dist↓ Diversity↑ MM↑</td><td></td></tr><tr><td>Ground Truth</td><td>0.511 0.703 0.797</td><td>0.002 2.974</td><td>9.503</td></tr><tr><td>T2M-GPT [74]</td><td>0.491 0.680 0.775</td><td>0.116 3.118</td><td>9.761 1.856</td></tr><tr><td>MDM [59]</td><td>0.611</td><td>0.544 5.566</td><td>9.559 2.799</td></tr><tr><td>FineMoGen [77]</td><td>0.5040.690 0.784</td><td>0.151 2.998</td><td>9.263 2.696</td></tr><tr><td>MoMask [19]</td><td>0.521 0.713 0.807</td><td>0.045 2.958</td><td>1.241</td></tr><tr><td>LMM-Tiny [76]</td><td>0.496 0.685 0.785</td><td>0.415 3.087</td><td>9.176 1.465</td></tr><tr><td>LMM-Large [76]</td><td>0.525 0.719 0.811</td><td>0.040 2.943</td><td>9.814 2.683</td></tr><tr><td>SynTalker [12]</td><td>0.375 0.564 0.681</td><td>4.385 4.499</td><td>9.374</td></tr><tr><td></td><td></td><td>0.530</td><td></td></tr><tr><td>MOCO (Ours)</td><td>0.434 0.6180.720</td><td>3.563</td><td>9.856 2.663</td></tr></table>

Table 10: Quantitative results of speech-to-gesture generation on the BEATX test set.
<table><tr><td>Method</td><td>FGD↓ BC</td><td>Diversity↑ MSE↓ LVD↓</td><td></td></tr><tr><td>FaceFormer [17]</td><td>一</td><td></td><td>7.787 7.593</td></tr><tr><td>CodeTalker [67]</td><td></td><td></td><td>8.026 7.766</td></tr><tr><td>S2G [18]</td><td>28.154.683</td><td>5.971</td><td></td></tr><tr><td>Trimodal [73]</td><td>12.415.933</td><td>7.724</td><td></td></tr><tr><td>HA2G [43]</td><td>12.326.779</td><td>8.626</td><td></td></tr><tr><td>DisCo [39]</td><td>9.417 6.439</td><td>9.912</td><td></td></tr><tr><td>CaMN [41]</td><td>6.644 6.769</td><td>10.86</td><td></td></tr><tr><td>DiffStyleGesture [69]</td><td>8.8117.241</td><td>11.49</td><td></td></tr><tr><td>TalkShow [71]</td><td>6.2096.947</td><td>13.47</td><td>7.791 7.771</td></tr><tr><td>EMAGE [40]</td><td>5.512 7.724</td><td>13.06</td><td>7.680 7.556</td></tr><tr><td>ProbTalk [44]</td><td>6.170 8.099</td><td>10.43</td><td>8.990 8.385</td></tr><tr><td>SynTalker [12]</td><td>6.4137.971</td><td>12.72</td><td></td></tr><tr><td>MOCO (Ours)</td><td>5.543 7.089</td><td>14.05</td><td>7.285 7.573</td></tr></table>

## G.4 Details of User Study

To construct the questionnaire for the user study, we first generated 20 videos using each method, including MOCO (Ours), Pseudo-Text, Weighted Sum, and Combine Once. The videos generated by MOCO were then concatenated with the other videos, resulting in 60 pairs of comparison videos. The user study involved 10 participants, each of whom evaluated all 60 pairs of comparison videos. Participants were asked to select their preferred video or indicate that neither was better based on one of the following criteria: text following, beat synchronization, body coherence, and temporal fluidity. The first 40 video comparisons involved MOCO versus Pseudo-Text and MOCO versus Weighted Sum, focusing on text following and beat synchronization. The last 20 video comparisons involved MOCO versus Combine Once, focusing on body coherence and temporal fluidity. The statistical results are presented in the main paper.

## G.5 Extended Pairwise-Preference Study against STMC.

The study above compares MOCO against the weighted-sum, pseudo-text, and combine-once baselines. To further corroborate the metric-level comparison with STMC reported in Appendix G.1 through human perception, we additionally conducted a larger-scale pairwise-preference study against STMC [49] (the same adapted STMC baseline as in Appendix G.1). This study involved 51 participants drawn from a bachelor’s-level (undergraduate) background, none of whom were involved in the project; each evaluated 60 comparison videos. We report participant background explicitly here for transparency, as the demographic composition of evaluators can influence perceptual judgments. For each video pair, participants chose the preferred clip—or “no preference”—along three criteria: text alignment, audio synchronization, and naturalness.

The results are summarized in Tab. 11. MOCO is preferred over STMC on all three criteria, with the margin reaching statistical significance for audio synchronization (47.1% vs. 39.4%, p = 0.02) and naturalness (46.1% vs. 39.8%, $p = 0 . 0 3 )$ , using a two-sided binomial test with ties excluded. For text alignment the preference favors MOCO but does not reach significance (45.9% vs. 42.0%, p = 0.11), which is consistent with the metric-level observation in Appendix G.1 that STMC is competitive on text-to-motion metrics. Taken together, these perceptual results reinforce our central claim: under concurrent text-and-speech control, MOCO produces motion that human viewers judge to be better synchronized and more natural, even where it does not dominate on isolated text-to-motion metrics.

Table 11: Extended pairwise-preference user study comparing MOCO against STMC [49] under concurrent text-and-speech control. Each cell reports the fraction of comparisons in which participants preferred MOCO, expressed no preference, or preferred STMC. The p-value is from a two-sided binomial test on the preference between the two methods (ties excluded); <sup>∗</sup> denotes statistical significance at $p < 0 . 0 5$
<table><tr><td>Criterion</td><td>Prefer MOCO</td><td>No Preference</td><td>Prefer STMC</td><td>p-value</td></tr><tr><td>Text Alignment</td><td>45.9%</td><td>12.1%</td><td>42.0%</td><td>0.11</td></tr><tr><td>Audio Synchronization</td><td>47.1%</td><td>13.5%</td><td>39.4%</td><td>0.02*</td></tr><tr><td>Naturalness</td><td>46.1%</td><td>14.1%</td><td>39.8%</td><td>0.03*</td></tr></table>

## H Details of Multi-Modal Benchmark

To efectively evaluate our proposed task, we developed a multi-modal benchmark comprising 1,000 test clips, following the methodology outlined in [49]. Each test clip is automatically generated and includes two text descriptions and two audio clips.

For the text descriptions, we manually curated a set of 40 texts focusing on lower-body movements commonly associated with speech delivery or conversation. These descriptions provide the necessary context for evaluating the corresponding movements within the benchmark. Regarding the audio clips, we selected recordings from the BEAT2 dataset, specifically choosing eight speakers with speaker IDs below 10. These audio files were segmented into clips using a Voice Activity Detector (VAD), resulting in 694 audio clips with an average duration of 9.14 seconds.

The 1,000 test clips were generated through an automated process that utilizes the curated text descriptions and audio clips. For each test clip, two text descriptions are randomly selected and assigned random durations. Subsequently, two neighboring audio clips are randomly chosen. The start times for both the text and audio intervals are determined randomly, allowing the sequence to commence with either text or audio. This process results in the creation of four intervals that correspond to the selected text descriptions and audio clips.

Optional text descriptions are listed here:

walk in a circle clockwise   
walk in a circle counterclockwise   
3 walk in a quarter circle to the left   
walk in a quarter circle to the right   
5 turn 180 degrees to the left on the left foot   
6 turn 180 degrees to the left on the right foot   
turn left   
8 turn right   
9 walk forwards   
10 walk backwards   
11 slowly walk forwards   
12 slowly walk backwards   
13 quickly walk forwards   
14 quickly walk backwards   
15 run   
16 jogs forwards   
17 jogs backwards   
18 slowly walk in a circle   
19 perform a squat   
20 sit down   
21 turn around then sit down in a chair   
22 sit down then get back up and walk back   
23 sit down for a moment   
24 step back and sit down   
25 sit down indian style   
26 take a step to their right and sit down   
27 sit criss cross   
28 sit down on the ground and cross their legs   
29 squat down   
30 sit on a high object   
31 sit on a barstool and rest their legs on the stool   
32 take a large step and sits on a stool

33 get down on their knees   
34 sit on the ground with his legs extended in front of him   
35 walk up to a backwards chair and sit down on it with legs   
outstretched   
36 sit down and adjust themselves   
37 sit down and swap their legs crossing back and forth   
38 sit and lie down on a lounge chair   
39 sit down and lean on the chair   
40 sits very still in the chair