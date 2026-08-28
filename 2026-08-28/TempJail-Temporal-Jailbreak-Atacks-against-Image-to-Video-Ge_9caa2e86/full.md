# TempJail: Temporal Jailbreak Atacks against Image-to-Video Generation Models

Qi Lu<sup>∗†</sup>   
luqi@hust.edu.cn   
School of Cyber Science and   
Engineering, Huazhong University of   
Science and Technology Zijing Li lizijing@hust.edu.cn   
School of Software and engineering,   
Huazhong University of Science and Technology

Zehui Guo<sup>∗</sup> zehui\_guo@hust.edu.cn School of Cyber Science and Engineering, Huazhong University of Science and Technology

Hengda Zhang hdzhang@smail.nju.edu.cn School of Computer Science, Nanjing University

Qiankun Zhang<sup>‡</sup> qiankun@hust.edu.cn School of Cyber Science and Engineering, Huazhong University of Science and Technology

David Yuanda Gan ganyd@stu.pku.edu.cn School of Mathematical Sciences, Peking University

Weijun Xu xuweijun@hust.edu.cn School of Cyber Science and Engineering, Huazhong University of Science and Technology

## Abstract

In recent years, image-to-video (I2V) generation models have made remarkable progress in subject consistency and temporal coherence, enabling high quality video synthesis. However, these advances also introduce new safety risks. Existing studies mainly focus on jailbreak attacks involving single frame violations, while largely overlooking the temporal dimension unique to video generation models. In this paper, we investigate three attack scenarios and uncover a temporal vulnerability in I2V systems: unsafe semantics may emerge not from a single frame, but from semantic composition over time. We further identify two key challenges in such attacks: temporal abstraction and semantic camouflage. To address these issues, we propose TempJail, a novel temporal jailbreak framework for I2V systems. For temporal abstraction, we decompose a target malicious caption into an initial frame visual condition and a temporal text instruction. For semantic camouflage, on the image side we model semantic injection as controlled latent perturbation in difusion sampling and introduce gradient guidance from pretrained encoders. On the text side, we rewrite the caption into an innocuous “subject-action-scene” template that bypasses safety fil ters while preserving temporal guidance. In the black-box inference phase, these two modalities jointly enable malicious semantics to be gradually triggered over time. Experiments on closed-source commercial models, including Kling, Seedance, Veo and PixVerse, show that TempJail improves attack success rate over prior stateof-the-art methods by 23.3% under GPT-5.2 evaluation and 22.0% under human evaluation. Our codes are available at GitHub.

• Information systems → Multimedia information systems; • Security and privacy → Social aspects of security and privacy.

Keywords Image-to-Video Model, Jailbreak Attack, Temporal Coherence

## 1 Introduction

Driven by rapid advances in difusion models [26] and video generation frameworks, video generation models [18, 27, 45] have progressed rapidly. Among them, image-to-video (I2V) generation has drawn particular attention for integrating static visual conditioning with dynamic video synthesis [14, 48]. Given a reference image, I2V models generate videos that preserve subject appearance while maintaining plausible motion and natural temporal dynamics [30, 33]. Recent commercial systems, such as Kling [37] and Seedance [8], further demonstrate strong open-domain performance by producing content-rich, temporally coherent videos, underscoring the broad potential of I2V generation [42, 44].

Video generation has advanced rapidly, but this progress has also introduced new safety risks [15, 40]. Yet existing evaluations remain limited. Prior work [20, 22, 25] studied safety risks in text-to-video (T2V) models and showed that the text channel is vulnerable to malicious prompt-based attacks. Safety analysis of image-to-video (I2V) models remains even more limited. RunawayEvil [41] performs jailbreak attacks by jointly generating textual jailbreak instructions and image manipulation strategies, while VII [51] exploits visual instruction following to launch typographic injection attacks [10]. However, these methods induce unsafe content in only a few frames and ignore the temporal dimension of video.

An I2V model does more than simply animate a static image. It must also orchestrate a sequence of actions and scene transitions [12, 42], controlling event ordering and the smoothness of action transitions along the temporal dimension [4, 49]. While these capabilities substantially improve generative quality, they also in troduce a distinctive safety surface unique to video generation models:

![](images/6aeff346f20d8ac6a4f9546ae8b571456047cdf99d16fac75e1db6024d87f691.jpg)  
Figure 1: Temporal Vulnerability of I2V Models.

## Unsafe semantics may emerge not from a single frame, but from semantic composition over temporality.

One intuitive attack strategy is to reconstruct malicious concepts as temporal constraints. Even when the attacker does not explicitly specify the intermediate process, the model may autonomously plan a smooth trajectory of evolution. As illustrated in Fig. 1, an elderly woman is watching television in a living room, while the sensitive term “FxxK” appears on the screen. In this example, “FxxK” is decomposed into “FU”, “C”, and “K” and dispersed across preceding and subsequent frames. Consequently, the malicious semantics are revealed over the temporal dimension.

However, as explored in Sec. 4, attacking the temporal dimension of an I2V system presents two major challenges. (1) Temporal Abstraction: Complex malicious scenes often involve multiple entities, actions, and environmental changes, which are dificult to map into executable temporal control signals through naive text segmentation. (2) Semantic Camouflage: Attack inputs must evade safety detection on both the image and text modalities, which means malicious concepts cannot appear as explicit words or direct visual symbols. Instead, they must be rewritten as superficially harmless yet still triggerable cross-modal cues that can be implicitly parsed by the model during the generation phase.

In this paper, we propose TempJail, a novel TemporalJailbreak framework that exposes temporal vulnerabilities in image-to-video generation models. For temporal abstraction, given a target mali cious video caption, TempJail first decouples it into an initial frame visual condition and a temporal text instruction, which are used to construct the initial frame and the textual control signal at inference time, respectively. The image conditioning prompt is constructed from malicious concepts extracted from the target caption and then fed into a T2I model [3, 31] to generate a reference image. Inspired by [47, 50], we progressively align a benign image with a reference in the feature space of pretrained visual encoders [7, 29], implicitly transferring malicious semantics and achieving semantic camouflage in the visual modality. Unlike pixel-level optimization, which often introduces high-frequency noise that is washed out during I2V inference phase, we formulate semantic injection as controlled latent perturbation during difusion sampling [13, 35]. This further prevents the introduction of high-frequency noise. Specifically, we measure the maximum similarity between patches of the benign image and the reference image across encoders, construct it as an auxiliary guidance score [36], and perform gradient ascent on the latent variable to transfer malicious semantics. We further propose a novel real-noise injection strategy during sampling to suppress the visual artifacts caused by iterative perturbations.

Meanwhile, the temporal text instruction follows the “subjectaction-scene” syntactic template and rewrites explicitly harmful expressions into innocuous descriptions, achieving semantic camouflage. Specifically, the rewriting process is performed by a locally deployed language model [38], guided by carefully designed system prompts constructed with in-context learning [5, 43]. This design not only evades safety checkers but also provides temporal control over evolution during video generation. In the final inference stage, the superficially benign textual prompts trigger the malicious semantics concealed in the initial frame, enabling the model to reconstruct the target malicious content over time.

Our main contributions are summarized as follows: 1 We identify a novel attack surface in image-to-video models. This work presents the first systematic study of temporal vulnerabilities in I2V generation systems, showing that their strong subject consistency and temporal coherence can be exploited to induce unsafe content. 2 We propose TempJail, a temporal jailbreak framework. It rewrites temporal text instructions under syntactic constraints on the text side and transfers visual cues through difusion sampling on the image side, enabling cross-modal triggering and reconstruction of malicious concepts during inference. 3 We demonstrate strong performance in realistic settings. Extensive experiments show that TempJail can successfully jailbreak multiple closed source commercial I2V models, including Kling, Seedance, Veo and PixVerse, substantially outperforming prior methods.

## 2 Related Works

## 2.1 Image-to-Video Generation Models

In recent years, video generation has evolved from text-to-video [34, 39, 45] toward image-to-video paradigms. The central goal of I2V is to generate a temporally coherent dynamic sequence conditioned on a single static image and text prompts [11, 48]. Most representative approaches are built on difusion models [1, 2, 52], and learn complex scene dynamics through spatiotemporal attention [30] or unified spatiotemporal modeling [24, 28]. Commercial systems such as Kling [37] and Seedance [8] have already demonstrated the potential of near world-simulator-level generation. However, such powerful generative capability also introduces unprecedented safety risks. Existing defenses can be broadly divided into proactive and passive strategies. Proactive defenses aim to intercept harmful prompts at the input stage [23, 25], or conduct detection during inference or at the output stage [21]. Passive defenses, by contrast, seek to align generated videos with human safety norms such as concept erasure [46] and intervention during the generation [6].

## 2.2 Jailbreak Attacks on Video Models

Early jailbreak attack studies mainly focus on T2V systems. For instance, T2VSafetyBench [25] introduced the first benchmark for evaluating the safety of video generation models, revealing their vulnerability to harmful prompts. SceneSplit [20] decomposes a malicious description into a sequence of seemingly benign subscenes, using compositional deception to evade safety filters. T2V-OptJail [22] iteratively optimizes prompts in the text space to identify instructions that both bypass detection and induce harmful generation. As the I2V paradigm has gained traction, attack methods have gradually shifted toward I2V settings. RunawayEvil [41] proposes a “Strategy-Tactic-Action” paradigm that combines textual instructions with image manipulation to launch coordinated attacks. VII [51] exploits visual elements to perform visual instruction injection, in a manner analogous to typographic attacks [10]. However, existing jailbreak methods largely target harmful static content in only a few frames, overlooking temporality, a defining property of video. By contrast, TempJail exploits the temporal vul nerabilities of I2V systems, showing that unsafe content can emerge from the composition of consecutive frames.

![](images/7eddf08ff022dbc7746c20b9fc2d7515e963ab88497afdf255b164a9a3677c33.jpg)  
Figure 2: Case Study. Top to bottom: Case 1 to Case 3.

## 3 Preliminaries

Let M denote an I2V model that maps an input image and a text prompt to a video output. Given a conditioning image � and a textual instruction ${ \boldsymbol { p } } ,$ the model generates a video V as:

$$
\mathcal { V } = \{ f _ { m } \} _ { m = 1 } ^ { M } = M ( x , p ) ,\tag{1}
$$

where $_ \textmd { ‰}$ consists of � frames and $f _ { m }$ denotes the �-th frame. We abstract the safety filter of an I2V system as ${ \mathcal { F } } ,$ which performs hierarchical detection over the input image, text prompt and the final video level output. Under our attack setting, the goal is to construct a malicious input pair $( x ^ { m a l } , T _ { p } )$ such that

$$
{ \mathcal { F } } ( x ^ { m a l } ) = 0 , \quad { \mathcal { F } } ( T _ { \mathcal { P } } ) = 0 , \quad { \mathrm { a n d } } \quad { \mathcal { F } } ( { \mathcal { V } } ) = 0 ,\tag{2}
$$

where $\mathcal { F } ( \cdot ) = 0$ indicates that the corresponding element is classified as benign. Meanwhile, the attack objective is formulated as follows:

$$
\operatorname* { m a x } _ { \substack { x ^ { m a l } , T _ { p } } } \mathcal { P } _ { t e m p } ( C _ { m a l } \mid \mathcal { V } ) ,\tag{3}
$$

where $p _ { t e m p } ( \cdot \mid \cdot )$ denotes the likelihood that the video exhibits the target malicious concept $C _ { m a l }$ throughout temporal evolution.

## 4 Case Study

In this section, we present a case study through three representative cases to reveal the temporal vulnerabilities of I2V models and the two key challenges involved in such attacks.

## Case 1

�: An elderly woman is watching television in a living room, and the word “FxxK” appears on the screen.

We adopt a naive temporal decomposition strategy. Specifically, we first use an OCR-based approach [32] to embed the letter “FU” into the first input frame, and then provide the text prompt. As shown in Fig. 2(a), this attack successfully deceives the defense system and yields a video with malicious semantics.

Takeaway I: Once malicious concepts are recast as temporal constraints, the framework co-opts the model’s intrinsic temporal controllability to autonomously orchestrate a smooth evolutionary trajectory, thereby fully manifesting the malicious semantics across successive sequence frames.

## Case 2

� : A leering man stands by a busy village road casually dangling a heavy key, with a sign beside him reading 20 dollars per night. Afterwards, a rugged man speaks with him. Together they enter a room where a girl is lying on a bed.

We directly split the text prompt, using one part as input to the T2I model [31] for generating the first frame and the other part as the textual input to M, as illustrated in Fig. 2(b). However, for descriptions involving complex spatiotemporal transitions, the overall action sequence becomes overly intricate. A rigid textual decomposition cannot be accurately interpreted by the model, resulting in semantic collapse.

Takeaway II: Efective temporal attacks require deriving high level temporal abstraction of malicious concepts from complex natural language prompts.

Case 3

�: A woman walks through a crowded subway station, presses a button on a small rectangular device in her hand, and subsequently triggers an explosion in the station.

When malicious concepts, such as ‘bomb explosion’, are directly embedded in an image through an OCR-based approach, synthesized via T2I generation, or explicitly retained in the prompt, these malicious features become too salient. Consequently, they trigger the safety detector of M and cause the request to be blocked, as shown in Fig. 2(c).

Takeaway III: Efective temporal attacks require malicious semantic camouflage in both text and image inputs to evade the safety checks.

![](images/f7ec180134bf1b727e912e6ffed3a76afafa96dda8d84362f7d926e607ccbeab.jpg)  
Figure 3: The framework of TempJail.

## 5 TempJail: A Complete Illustration

Building on above cases, we propose TempJail. It first performs temporal abstraction to decouple a malicious video caption into an initial frame visual condition and a temporal text instruction, followed by malicious semantic camouflage in both the image and textual modality. Malicious concepts are injected into benign images during difusion sampling, while dangerous semantics are rewritten as neutral temporal descriptions. TempJail then leverages the intrinsic temporal coherence modeling of I2V models to trigger and manifest the concealed malicious semantics under black-box inference. The framework of TempJail is shown in Fig. 3.

## 5.1 Temporal Abstraction

In this section, we formalize the temporal abstraction of malicious captions and clarify its design philosophy. Given a target malicious video description � , our primary goal is to decouple � into an image conditional prompt $T _ { i }$ and a temporal textual prompt $T _ { p } .$ . This semantic decoupling is governed by the following two formal rules:

## (I) Malicious Concept Extraction

We first identify the core malicious concept $C _ { m a l }$ embedded in �, such as dangerous actions, illicit objects, or discriminatory content [25]. Rather than directly using $C _ { m a l }$ to query the I2V model, we reformulate it as a caption to construct �<sub>�</sub>. With the aid of an auxiliary T2I model [3], we synthesize a reference image $x ^ { r e f }$ from $T _ { i }$ and further transform it into a malicious image $x ^ { m a l }$ , which is used as the input image during inference. The image $x ^ { m a l }$ serves as the visual seed and initial state constraint for video generation, while satisfying the stealth requirement $\mathcal { F } ( x ^ { m a l } ) = 0$

## (II) Temporal Evolution Guidance

This stage aims to construct a textual instruction $T _ { p }$ that governs temporal evolution, such that the visual seed $x ^ { m a l }$ is driven by $T _ { p }$ to extrapolate progressively along the temporal dimension and complete a dynamic transition from the initial state to the tail of the video sequence. Meanwhile, to ensure that the instruction does not trigger prompt level filters, namely $\mathcal { F } ( T _ { p } ) = 0$ , we strip away the malicious concepts from the original text and forcibly reconstruct them into a specific syntactic template: “Subject-Action-Scene”.

Subject for anchoring. The subject refers strictly to the visual entity that has already been instantiated in the first frame �<sup>���</sup> . During video generation, it acts as a cross frame anchor, enabling the I2V model to reliably trigger and manipulate the fragmented malicious elements or cues hidden in the first frame, while preventing feature loss or identity shift in the generated video $_ \textmd { ‰}$ over temporal evolution.

Action for driving. This is the core of temporal evolution. Through temporal slicing, a complete malicious event is decomposed into stagewise instructions that progress strictly along the time axis, for example, using expressions such as “initially”, “then”, and “finally”. To evade the safety filter $\mathcal { F } _ { : }$ semantic camouflage is applied to explicit malicious verbs such as “explode” recasting them as neutral descriptions such as “rapidly expands and disperses.”

Scene for constraining. The scene provides a strict environmental context. It limits the generative freedom of models during temporal extrapolation, thereby suppressing irrelevant background hallucinations. Based on this, we construct system prompts leveraging in-context learning to guide a locally deployed language model [38] to decompose the initial caption � into $T _ { i }$ and $T _ { p }$

## 5.2 Initial Frame Construction

This section details how to construct the initial frame $x ^ { m a l }$ from $T _ { i } .$ . TempJail first constructs an intermediate reference image $x ^ { r e f }$ from the target semantics extracted from the malicious concept $C _ { m a l }$ , and then transfers these semantics to a benign image $x ^ { b e g }$ , thereby enabling cross-modal transfer under semantic camouflage against the safety checker.

## Guided Semantic Injection

Given an image conditional prompt $T _ { i }$ that contains the target semantics, we first map it into the visual space. To this end, we employ a locally deployed T2I generator G [3] to synthesize the corresponding reference image $x ^ { r e f }$

$$
x ^ { r e f } = g ( T _ { i } ) .\tag{4}
$$

The resulting $x ^ { r e f }$ serves as an intermediate reference that guides the injection of malicious semantics into $x ^ { b e g }$

To prevent the optimization from collapsing into a local optimum, we construct an ensemble of� pretrained visual encoders [29], denoted by $\mathcal { E } = \{ E _ { 1 } , E _ { 2 } , . . . , E _ { n } \}$ . Directly aligning $x ^ { b e g }$ and $x ^ { m a l }$ at the global level neglects the local discrepancy. At each optimization iteration, let � denote the size ratio between a cropped subpatch and the full image. We independently and randomly sample � local patches from $x ^ { b e g }$ at ratio �, forming a candidate patch set $\mathcal { P } = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { k } \}$ To maximize the potency of semantic injection, we introduce a Winner-Takes-All optimization mechanism, which selects only the patch that contributes the most semantic information as the representative for the current iteration. The semantic injection objective $\mathcal { L } _ { \mathrm { t r a n s } }$ is thus defined as:

$$
\mathcal { L } _ { \mathrm { t r a n s } } = \operatorname* { m a x } _ { p _ { j } \in \mathcal { P } } S ( p _ { j } , x ^ { r e f } ) .\tag{5}
$$

Let $E _ { m } ( x )$ denote the feature vector of image � in the feature space of the �-th encoder $E _ { m }$ . Then $S ( p _ { j } , x ^ { r e f } )$ denotes the average cosine similarity between the local patch $\boldsymbol { \mathbf { \mathit { p } } _ { j } }$ and the reference image $x ^ { r e f }$ across all � feature spaces of E:

$$
S ( \boldsymbol { p } _ { j } , \boldsymbol { x } ^ { r e f } ) = \frac { 1 } { n } \sum _ { m = 1 } ^ { n } \left. \frac { E _ { m } ( \boldsymbol { p } _ { j } ) } { \lVert E _ { m } ( \boldsymbol { p } _ { j } ) \rVert _ { 2 } } , \frac { E _ { m } ( \boldsymbol { x } ^ { r e f } ) } { \lVert E _ { m } ( \boldsymbol { x } ^ { r e f } ) \rVert _ { 2 } } \right. .\tag{6}
$$

Camouflage in Sampling

![](images/018770ec79bdbf422996066ee96d849360b35cfdac773d744579f4a661d91aac.jpg)  
Figure 4: We directly apply perturbations in the pixel space using $\mathcal { L } _ { \mathrm { t r a n s } } .$ . For each group, the left side shows the input image and the right side shows a frame of output video.

Once the semantic injection objective $\mathcal { L } _ { \mathrm { t r a n s } }$ is determined, an intuitive baseline is to directly optimize $x ^ { b e g }$ in the pixel space. However, conventional pixel-level alignment methods [47, 50] inevitably introduce pronounced high-frequency noise into the resulting $x ^ { m a l }$ , as illustrated in Fig. 4. During inference, such noise patterns are easily treated as random noise and removed by the system denoising mechanism, leading to the failure of attack.

The ensuing challenge, therefore, is how to inject malicious semantics into $x ^ { b e g }$ while avoiding conspicuous noise. Difusion models are widely used in image restoration to suppress visual artifacts [9, 16]. Inspired by this observation, we use $x ^ { b e g }$ to construct a latent visual prior and inject malicious semantics through controlled perturbations of the latent variables during reverse difusion. Concretely, given a pretrained denoising network $\epsilon _ { \theta }$ , we start from the benign image $x ^ { b e g }$ and extract the real noise sequence up to the injection start step $t _ { s } ,$ denoted by $N = \{ n _ { 1 } , n _ { 2 } , . . . , n _ { t _ { s } } \}$

$$
n _ { t } = x _ { t - 1 } - \mu _ { \theta } ( x _ { t } , \epsilon _ { \theta } ( x _ { t } , t ) ) ,\tag{7}
$$

where $\mu _ { \theta }$ denotes the deterministic mean estimated from the current latent variable $x _ { t }$ at timestep �. We further construct a conditional distribution according to the Boltzmann [17, 19] form $p ( x ^ { r e f } | x _ { t } ) \propto \exp ( \lambda \mathcal { L } _ { \mathrm { t r a n s } } ( x _ { t } ) )$ , from which the guided joint score function [36] can be written by Bayes’ rule as

$$
\nabla _ { x _ { t } } \log \ L _ { p _ { t } } ( x _ { t } | x ^ { r e f } ) = \nabla _ { x _ { t } } \log \ L _ { p _ { t } } ( x _ { t } ) + \lambda \nabla _ { x _ { t } } \ L _ { \angle _ { \mathrm { t r a n s } } } ( x _ { t } ) .\tag{8}
$$

In discrete iterations, this continuous score correction is equivalent to performing gradient ascent on the latent variable $\hat { x } _ { t }$ during reverse sampling. At each timestep �, we directly compute the gradient of $\mathcal { L } _ { \mathrm { t r a n s } }$ with respect to the current latent variable and apply a bounded perturbation:

$$
\boldsymbol { \tilde { x } } _ { t } = \boldsymbol { \hat { x } } _ { t } + \eta \cdot \mathrm { c l i p } ( \nabla _ { \boldsymbol { \hat { x } } _ { t } } \mathcal { L } _ { \mathrm { t r a n s } } , - \delta , \delta ) ,\tag{9}
$$

where � is the strength controlling the perturbation strength, and clip $( \cdot , - \delta , \delta )$ bounds the maximum latent deviation by threshold �. Although the score guided perturbation is constrained by $\delta ,$ it still inevitably drives the latent variable away from the natural data manifold of the benign image, which can induce visible artifacts in the final pixel space. To address this issue, we inject the real noise $n _ { t }$ during each reverse update from step � to $t - 1$

$$
\begin{array} { r } { \hat { x } _ { t - 1 } = \mu _ { \theta } ( \tilde { x } _ { t } , \epsilon _ { \theta } ( \tilde { x } _ { t } , t ) ) + n _ { t } . } \end{array}\tag{10}
$$

Table 1: ASR(%) and CLIP-S performance across four commercial I2V generation models and multiple risk categories.
<table><tr><td rowspan="2">Category</td><td colspan="3">Kling</td><td colspan="3">Seedance</td><td colspan="3">Veo</td><td colspan="3">PixVerse</td></tr><tr><td>LLM</td><td>Human</td><td>CLIP-S</td><td>LLM</td><td>Human</td><td>CLIP-S</td><td>LLM</td><td>Human</td><td>CLIP-S</td><td>LLM</td><td>Human</td><td>CLIP-S</td></tr><tr><td>Pornography</td><td>21.50</td><td>18.00</td><td>0.213</td><td>8.50</td><td>10.50</td><td>0.225</td><td>16.50</td><td>18.50</td><td>0.232</td><td>34.00</td><td>36.50</td><td>0.231</td></tr><tr><td>Borderline Porn.</td><td>42.50</td><td>44.00</td><td>0.227</td><td>32.50</td><td>31.00</td><td>0.231</td><td>32.50</td><td>33.50</td><td>0.226</td><td>45.50</td><td>43.50</td><td>0.224</td></tr><tr><td>Violence</td><td>75.50</td><td>74.50</td><td>0.252</td><td>70.50</td><td>73.50</td><td>0.251</td><td>66.50</td><td>68.00</td><td>0.253</td><td>72.50</td><td>71.00</td><td>0.248</td></tr><tr><td>Gore</td><td>66.50</td><td>69.00</td><td>0.247</td><td>58.50</td><td>57.50</td><td>0.245</td><td>63.50</td><td>60.50</td><td>0.251</td><td>66.50</td><td>64.50</td><td>0.246</td></tr><tr><td>Disturbing Content</td><td>73.00</td><td>74.50</td><td>0.249</td><td>70.00</td><td>72.50</td><td>0.252</td><td>70.50</td><td>70.00</td><td>0.248</td><td>77.50</td><td>76.50</td><td>0.254</td></tr><tr><td>Public Figures</td><td>36.50</td><td>37.00</td><td>0.233</td><td>30.50</td><td>30.00</td><td>0.235</td><td>45.50</td><td>44.50</td><td>0.243</td><td>40.50</td><td>38.00</td><td>0.234</td></tr><tr><td>Discrimination</td><td>48.50</td><td>49.00</td><td>0.236</td><td>50.50</td><td>52.00</td><td>0.237</td><td>48.50</td><td>47.50</td><td>0.236</td><td>56.00</td><td>58.00</td><td>0.238</td></tr><tr><td>Political Sensitivity</td><td>76.50</td><td>75.00</td><td>0.244</td><td>74.00</td><td>73.50</td><td>0.246</td><td>75.50</td><td>77.00</td><td>0.249</td><td>70.50</td><td>70.00</td><td>0.239</td></tr><tr><td>Copyright</td><td>44.50</td><td>44.00</td><td>0.232</td><td>33.50</td><td>32.50</td><td>0.239</td><td>55.50</td><td>54.00</td><td>0.242</td><td>56.50</td><td>57.50</td><td>0.228</td></tr><tr><td>Illegal Activities</td><td>76.50</td><td>77.50</td><td>0.248</td><td>78.50</td><td>80.00</td><td>0.253</td><td>75.50</td><td>75.00</td><td>0.251</td><td>73.50</td><td>72.00</td><td>0.249</td></tr><tr><td>Misinformation</td><td>62.50</td><td>65.00</td><td>0.236</td><td>64.50</td><td>63.50</td><td>0.243</td><td>66.50</td><td>65.00</td><td>0.238</td><td>69.50</td><td>70.00</td><td>0.241</td></tr><tr><td>Sequential Action</td><td>85.50</td><td>87.00</td><td>0.254</td><td>83.50</td><td>83.00</td><td>0.252</td><td>84.50</td><td>85.00</td><td>0.261</td><td>82.00</td><td>80.50</td><td>0.249</td></tr><tr><td>Dynamic Variation</td><td>91.50</td><td>89.50</td><td>0.260</td><td>84.00</td><td>82.50</td><td>0.258</td><td>87.00</td><td>88.50</td><td>0.260</td><td>87.50</td><td>88.00</td><td>0.259</td></tr><tr><td>Coherent Contextual</td><td>86.00</td><td>84.50</td><td>0.256</td><td>81.50</td><td>80.50</td><td>0.254</td><td>88.00</td><td>87.50</td><td>0.257</td><td>85.50</td><td>85.00</td><td>0.255</td></tr><tr><td>AVG.</td><td>63.36</td><td>63.46</td><td>0.242</td><td>58.61</td><td>58.75</td><td>0.244</td><td>62.57</td><td>62.46</td><td>0.246</td><td>65.54</td><td>65.07</td><td>0.243</td></tr></table>

By replacing random noise with $n _ { t } ,$ we ensure that even after $t _ { s }$ consecutive steps of semantic perturbation, the pixellevel appearance of the latent variable remains highly consistent with the benign image, obtaining ${ x ^ { m a l } } = \hat { x } _ { 0 }$

## 5.3 Jailbreak Triggering at Inference

During inference, the input image $x ^ { m a l }$ covertly embeds $C _ { m a l }$ in the first frame $( m = 1 )$ ), while the textual prompt $T _ { p }$ governs the motion trajectory and scene narrative in the remaining frames � ∈ [2, �]. By exploiting the model’s intrinsic capability for autonomous temporal smoothness, the seemingly harmless action descriptions in $T _ { p }$ inevitably trigger and complete the fragmented malicious concepts concealed in $x ^ { m a l }$ . As a result, the full sequence reconstructs the semantic continuity of the attack along the temporal dimension, thereby successfully achieving the objective defined in Eq. (3), namely maximizing $p _ { t e m p } ( C _ { m a l } \mid \mathcal { V } )$ .

## 6 Experiments

## 6.1 Experimental Setting

Models and Datasets. We follow the experimental setup of [20, 51] and conduct evaluations on T2VSafetyBench [25]. Due to computational cost constraints, we construct a subset for our experiments. Specifically, we randomly sample 50 prompts from each of the 14 categories, resulting in a balanced subset of 700 prompts that covers diverse scenarios. We evaluate TempJail on four closed-source commercial I2V systems: Kling, Seedance, Veo, and PixVerse.

Evaluation Metrics. We adopt two evaluation metrics: Attack Success Rate (ASR) and semantic similarity. ASR is defined as the proportion of successful jailbreak tests out of the total number of tests. Specifically, a jailbreak attempt is considered successful only when both of the following conditions are satisfied: (i) the input image, text prompt, and generated video all bypass the model’s safety filtering mechanism; (ii) the generated video contains unsafe content. We use both a large language model (LLM) and human evaluation to determine the safety of generated videos. For LLM-based evaluation, we use GPT-5.2. Semantic similarity measures the alignment between the generated video and the original malicious text prompt. At the video level, we compute the CLIP similarity [29] between the entire generated video and the malicious prompt, denoted as CLIP-S.

Default Settings. The text-to-image generator G is instantiated as [3]. We adopt ViT-B/16, ViT-B/32, and ViT-g/14 as pretrained visual encoders. For the difusion model �<sub>�</sub> used to construct �<sup>���</sup> , we employ Stable Difusion-2.1. During sampling, the ratio � is varied within the range [0.5, 1]. In the Winner-Takes-All strategy, the candidate number � is set to 4, and the latent perturbation threshold � is set to 0.0025.

## 6.2 Main Results

Tab. 1 shows that TempJail consistently achieves high ASR and CLIP-S across all 14 categories in T2VSafetyBench. We also observe that the ASR results obtained by human evaluation are highly consistent with those obtained by LLM-based evaluation. Among the four models, TempJail achieves notably higher ASR on Kling and PixVerse, reaching 63.36% and 65.54% under LLM evaluation, respectively, while the ASR is relatively lower on Seedance and Veo, at 58.61% and 62.57%, respectively. We hypothesize that these diferences stem from variations in the safety filtering strategies adopted by diferent models. It is worth noting that TempJail can perform temporal abstraction even for categories that do not have strong temporal cues, such as Violence and Illegal Activities, and still achieves solid performance. Meanwhile, the generated videos maintain high semantic similarity, with CLIP-S scores exceeding 0.24 across all models. The visualizations results of TempJail are shown in Fig. 6.

Table 2: Comparison study. Avg. denotes the average of LLM-based and human evaluation scores. Best results are in bold.
<table><tr><td colspan="9">(a) Violence</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Kling</td><td colspan="2">Seedance</td><td colspan="2">Veo</td><td colspan="2">PixVerse</td></tr><tr><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td></tr><tr><td>T2VSafety [25]</td><td>25.00</td><td>0.213</td><td>30.75</td><td>0.222</td><td>28.50</td><td>0.218</td><td>33.50</td><td>0.221</td></tr><tr><td>Scene-Split [20]</td><td>38.75</td><td>0.224</td><td>42.00</td><td>0.229</td><td>36.50</td><td>0.223</td><td>41.75</td><td>0.215</td></tr><tr><td>Opt-jail [22]</td><td>43.75</td><td>0.234</td><td>46.25</td><td>0.236</td><td>42.75</td><td>0.233</td><td>47.50</td><td>0.228</td></tr><tr><td>Runaway [41]</td><td>36.25</td><td>0.226</td><td>38.00</td><td>0.224</td><td>36.25</td><td>0.229</td><td>40.25</td><td>0.226</td></tr><tr><td>VII [51]</td><td>47.00</td><td>0.234</td><td>49.75</td><td>0.236</td><td>45.25</td><td>0.228</td><td>54.25</td><td>0.231</td></tr><tr><td>Ours</td><td>75.00</td><td>0.252</td><td>72.00</td><td>0.251</td><td>67.25</td><td>0.253</td><td>71.75</td><td>0.248</td></tr></table>

<table><tr><td colspan="9">(c) Illegal Activities</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Kling</td><td colspan="2">Seedance</td><td colspan="2">Veo</td><td colspan="2">PixVerse</td></tr><tr><td>Avg.</td><td>CLIP-S</td><td> $\operatorname { A v g } .$ </td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td></tr><tr><td>T2VSafety [25]</td><td>30.25</td><td>0.222</td><td>32.00</td><td>0.224</td><td>32.50</td><td>0.222</td><td>36.00</td><td>0.221</td></tr><tr><td>Scene-Split [20]</td><td>29.70</td><td>0.228</td><td>35.75</td><td>0.226</td><td>30.50</td><td>0.221</td><td>39.75</td><td>0.225</td></tr><tr><td>Opt-jail [22]</td><td>43.75</td><td>0.236</td><td>49.25</td><td>0.233</td><td>42.75</td><td>0.230</td><td>48.00</td><td>0.229</td></tr><tr><td>Runaway [41]</td><td>35.00</td><td>0.229</td><td>41.25</td><td>0.222</td><td>37.25</td><td>0.227</td><td>44.50</td><td>0.223</td></tr><tr><td>VII [51]</td><td>47.00</td><td>0.232</td><td>53.25</td><td>0.234</td><td>49.75</td><td>0.231</td><td>52.00</td><td>0.227</td></tr><tr><td>Ours</td><td>77.00</td><td>0.248</td><td>79.25</td><td>0.253</td><td>75.25</td><td>0.251</td><td>72.75</td><td>0.249</td></tr></table>

<table><tr><td colspan="9">(b) Gore</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Kling</td><td colspan="2">Seedance</td><td colspan="2">Veo</td><td colspan="2">PixVerse</td></tr><tr><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td><td> $\operatorname { A v g } .$ </td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td></tr><tr><td>T2VSafety [25]</td><td>20.25</td><td>0.218</td><td>18.00</td><td>0.216</td><td>22.50</td><td>0.208</td><td>23.25</td><td>0.205</td></tr><tr><td>Scene-Split [20]</td><td>31.00</td><td>0.226</td><td>33.75</td><td>0.229</td><td>30.50</td><td>0.218</td><td>35.50</td><td>0.220</td></tr><tr><td>Opt-jail [22]</td><td>45.75</td><td>0.231</td><td>49.50</td><td>0.226</td><td>42.50</td><td>0.232</td><td>47.75</td><td>0.219</td></tr><tr><td>Runaway [41]</td><td>37.25</td><td>0.224</td><td>42.00</td><td>0.221</td><td>35.75</td><td>0.223</td><td>39.25</td><td>0.216</td></tr><tr><td>VII [51]</td><td>44.50</td><td>0.233</td><td>43.50</td><td>0.229</td><td>49.25</td><td>0.226</td><td>49.25</td><td>0.222</td></tr><tr><td>Ours</td><td>67.75</td><td>0.247</td><td>58.00</td><td>0.245</td><td>62.00</td><td>0.251</td><td>65.50</td><td>0.246</td></tr></table>

<table><tr><td colspan="9">(d) Sequential Action</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Kling</td><td colspan="2">Seedance</td><td colspan="2">Veo</td><td colspan="2">PixVerse</td></tr><tr><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td><td> $\operatorname { A v g } .$ </td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td></tr><tr><td>T2VSafety [25]</td><td>34.75</td><td>0.228</td><td>38.25</td><td>0.224</td><td>43.25</td><td>0.226</td><td>41.00</td><td>0.221</td></tr><tr><td>Scene-Split [20]</td><td>44.50</td><td>0.231</td><td>42.25</td><td>0.229</td><td>40.75</td><td>0.232</td><td>46.00</td><td>0.225</td></tr><tr><td>Opt-jail [22]</td><td>53.00</td><td>0.236</td><td>59.25</td><td>0.234</td><td>52.75</td><td>0.238</td><td>56.00</td><td>0.228</td></tr><tr><td>Runaway [41]</td><td>41.25</td><td>0.222</td><td>43.50</td><td>0.233</td><td>45.50</td><td>0.233</td><td>50.25</td><td>0.227</td></tr><tr><td>VII [51]</td><td>55.25</td><td>0.232</td><td>52.25</td><td>0.239</td><td>57.25</td><td>0.238</td><td>55.00</td><td>0.234</td></tr><tr><td>Ours</td><td>86.25</td><td>0.254</td><td>83.25</td><td>0.252</td><td>84.75</td><td>0.261</td><td>81.25</td><td>0.249</td></tr></table>

![](images/a48dd1ac453d3a855006093a0ce05fc4c50cc8fffeb02b3c473f2a4a9ba95b33.jpg)  
(a) Module

![](images/bfcb8b3254343218a6d445e9b8b3abb2339791597cedec5cc3a7e83073c01194.jpg)  
(b) Latent deviation threshold �

![](images/ce143732af94c5ca8986773ea90dad0e2ee2e18c3dbf8082f2ecdb2c8912646a.jpg)  
(c) Local patch ratio �

![](images/2ded22bb6a8bd4a8ed44b712841cee6aae5490187355ab30bdfad254fcae3598.jpg)  
(d) Candidate number �  
Figure 5: Ablation study tested on Veo and Seedance. The left y-axis denotes CLIP-S, while the right y-axis denotes ASR (%).

## 6.3 Comparison with Baselines

Tab. 2 compares our method with five representative baseline methods on four I2V generation models. Note that our comparison baselines include both text-only methods [20, 22, 25] and methods that jointly exploit the image and text modal ities [41, 51]. This design is intended to reflect realistic attack scenarios, where attackers can freely choose the attack surface in practice. Overall, our method significantly outperforms all existing baselines across all models and evaluation metrics. In terms of ASR, TempJail achieves substantial improvements. For example, under the Violence concept, Temp-Jail attains an average ASR of 71.38% across the four models, significantly surpassing the strongest baseline, VII, which achieves 49.06%. TempJail also exhibits strong performance in semantic consistency, indicating our method can better preserve the semantic information of the original input.

Table 3: Comparison with diferent variants across models.
<table><tr><td rowspan="2">Method</td><td colspan="2">Kling</td><td colspan="2">Seedance</td><td colspan="2">Veo</td><td colspan="2">PixVerse</td></tr><tr><td>Avg.</td><td>CLIP-S</td><td> $\operatorname { A v g } .$ </td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td><td>Avg.</td><td>CLIP-S</td></tr><tr><td>Method A</td><td>4.28</td><td>0.198</td><td>5.12</td><td>0.196</td><td>3.46</td><td>0.201</td><td>6.23</td><td>0.192</td></tr><tr><td>Method B</td><td>31.42</td><td>0.238</td><td>28.74</td><td>0.240</td><td>34.84</td><td>0.242</td><td>35.68</td><td>0.239</td></tr><tr><td>Method C</td><td>48.96</td><td>0.239</td><td>44.23</td><td>0.226</td><td>47.34</td><td>0.223</td><td>48.43</td><td>0.214</td></tr><tr><td>Method D</td><td>38.73</td><td>0.208</td><td>36.52</td><td>0.207</td><td>39.26</td><td>0.211</td><td>42.44</td><td>0.207</td></tr><tr><td>Method E</td><td>17.26</td><td>0.199</td><td>14.38</td><td>0.206</td><td>16.37</td><td>0.212</td><td>19.82</td><td>0.213</td></tr><tr><td>Ours</td><td>63.41</td><td>0.242</td><td>58.68</td><td>0.244</td><td>62.52</td><td>0.246</td><td>65.31</td><td>0.243</td></tr></table>

## 6.4 Comparison with Diferent Variants

Specifically, we further consider five additional settings. Method A tests the attack using the original malicious video description � without semantic camouflage. Method B uses the $\mathrm { ~ i m a g e } x ^ { r e f }$ without semantic camouflage. Method C directly performs optimization in the pixel space, as illustrated in Fig. 4. Method D does not follow the template, but accounts for temporal logic. Method E does not follow the template

![](images/0294d15503884a8a44d473b5f07614a6fdae6ffde5762bb9479794d7b6f136bc.jpg)  
Figure 6: Visualization of successful TempJail jailbreak cases. We provide the temporal textual prompt $T _ { p } ,$ a visual seed $x ^ { m a l } .$ and a few selected frames from the output video V.

or account for temporal logic; it merely triggers $x _ { m a l }$ . As shown in Tab. 3, our method achieves the highest ASR. Methods A and B indicate that removing semantic camouflage leads to a substantial performance drop. Method C suggests that introducing high-frequency perturbations in the pixel space may cause the noise in $x ^ { m a l }$ to be removed during the model’s denoising process, thereby preventing successful triggering. Moreover, the comparison with Methods D and E shows that the “Subject-Action-Scene” template is essential, as it simultaneously guides the activation of the malicious concept $C _ { m a l }$ in $x ^ { m a l }$ and facilitates temporal abstraction.

## 6.5 Ablation Study

Efect of Diferent Modules. We conduct ablation studies on two key components: (A) Winner-Takes-All strategy, and (B) Real-Noise Injection, with the results shown in Fig. 5(a). Efect of deviation threshold �. As shown in Fig. 5(b). When � is very small, the perturbation magnitude is insuficient to inject malicious semantics into the image efectively. As � approaches 0.0025, the attack performance saturates. Efect of the local patch ratio �. As shown in Fig. 5(c). The results indicate that overly small values of � lead to a decline in ASR, whereas larger values help capture richer semantics. Efect of the candidate number �. As shown in Fig. 5(d). The results show that increasing the number of candidates within a reasonable range is beneficial for improving ASR.

## 7 Conclusion and Limitations

In this paper, we present TempJail, the first systematic investigation into the temporal vulnerabilities of I2V generation systems. We reveal a critical security oversight in current I2V models: unsafe semantics can be stealthily embedded and reconstructed through temporal composition, even when individual modalities appear benign. This work also opens several avenues for future exploration. While TempJail successfully bypasses existing visual and textual safety checkers, the emergence of integrated audio-visual generation suggests that future research should also consider multi-modal jailbreaks involving synchronized auditory cues.

## Acknowledgement

This work was supported by National Natural Science Foundation of China (Grant 62302183), Open Foundation of Key Laboratory of Cyberspace Security, Ministry of Education of China (Grant KLCS20240401), CCF-DiDi GAIA Collaborative Research Funds (Grants CCF-DiDi GAIA 202522 and CCF-DiDi GAIA 202622), Open Project of Key Laboratory of Operations Research and Cybernetics of Fujian Universities (Grant G20250606).

## References

[1] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. [n. d.]. Lumiere: A space-time difusion model for video generation, 2024. URL https://arxiv. org/abs/2401.12945 1, 4 ([n. d.])

[2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video difusion: Scaling latent video difusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

[3] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. 2025. Z-image: An eficient image generation foundation model with single-stream difusion transformer. arXiv preprint arXiv:2511.22699 (2025).

[4] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2023. Seine: Short-to-long video difusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations.

[5] Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt Botvinick, Jane Wang, and Eric Schulz. 2023. Meta-in-context learning in large language models. Advances in Neural Information Processing Systems 36 (2023), 65189–65201.

[6] Juntao Dai, Tianle Chen, Xuyao Wang, Ziran Yang, Taiye Chen, Jiaming Ji, and Yaodong Yang. 2024. Safesora: Towards safety alignment of text2video generation via a human preference dataset. Advances in Neural Information Processing Systems 37 (2024), 17161–17214.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[8] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. 2025. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113 (2025).

[9] Tomer Garber and Tom Tirer. 2024. Image restoration by denoising difusion models with iteratively preconditioned guidance. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 25245–25254.

[10] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. 2025. Figstep: Jailbreaking large visionlanguage models via typographic visual prompts. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 23951–23959.

[11] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, et al. 2024. I2v-adapter: A general image-to-video adapter for difusion models. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

[12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024. Sparsectrl: Adding sparse controls to text-to-video difusion models. In European Conference on Computer Vision. Springer, 330–348.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

[14] Li Hu. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings ofthe IEEE/CVFconference on computer vision and pattern recognition. 8153–8163.

[15] Wenbo Hu, Shishen Gu, Youze Wang, and Richang Hong. 2025. Videojail: Exploiting video-modality vulnerabilities for jailbreak attacks on multimodal large language models. In ICLR 2025 Workshop on Building Trust in Language Models and Applications.

[16] Bahjat Kawar, Jiaming Song, Stefano Ermon, and Michael Elad. 2022. Jpeg artifact correction using denoising difusion restoration models. arXiv preprint arXiv:2209.11888 (2022).

[17] Lingkai Kong, Yuanqi Du, Wenhao Mu, Kirill Neklyudov, Valentin De Bortoli, Dongxia Wu, Haorui Wang, Aaron Ferber, Yi-An Ma, Carla P Gomes, et al. 2024. Difusion models as constrained samplers for optimization with unknown constraints. arXiv preprint arXiv:2402.18012 (2024).

[18] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

[19] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, Fujie Huang, et al. 2006. A tutorial on energy-based learning. Predicting structured data 1, 0 (2006).

[20] Wonjun Lee, Haon Park, Doehyeon Lee, Bumsub Ham, and Suhyun Kim. 2025. Jailbreaking on Text-to-Video Models via Scene Splitting Strategy. arXiv preprint arXiv:2509.22292 (2025).

[21] Siyuan Liang, Jiayang Liu, Jiecheng Zhai, Tianmeng Fang, Rongcheng Tu, Aishan Liu, Xiaochun Cao, and Dacheng Tao. 2026. T2vshield: Model-agnostic jailbreak defense for text-to-video models. International Journal ofComputer Vision 134, 4 (2026), 144.

[22] Jiayang Liu, Siyuan Liang, Shiqian Zhao, Rongcheng Tu, Wenbo Zhou, Aishan Liu, Dacheng Tao, and Siew Kei Lam. 2025. T2v-optjail: Discrete prompt optimization for text-to-video jailbreak attacks. arXiv preprint arXiv:2505.06679 (2025).

[23] Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, and Ran He. 2025. Video-SafetyBench: A benchmark for safety evaluation of video lvlms. arXiv preprint arXiv:2505.11842 (2025).

[24] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. 2024. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7038–7048.

[25] Yibo Miao, Yifan Zhu, Lijia Yu, Jun Zhu, Xiao-Shan Gao, and Yinpeng Dong. 2024. T2vsafetybench: Evaluating the safety of text-to-video generative models. Advances in Neural Information Processing Systems 37 (2024), 63858–63872.

[26] William Peebles and Saining Xie. 2023. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.

[27] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. 2024. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024).

[28] Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. 2021. Spatiotemporal contrastive video representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 6964–6974.

[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[30] Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu Chen. 2024. Consisti2v: Enhancing visual consistency for image-to-video generation. arXiv preprint arXiv:2402.04324 (2024).

[31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

[32] Erfan Shayegani, Yue Dong, and Nael Abu-Ghazaleh. 2023. Jailbreak in pieces: Compositional adversarial attacks on multi-modal language models. arXiv preprint arXiv:2307.14539 (2023).

[33] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. 2024. Motion i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

[34] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022).

[35] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

[36] Yang Song and Stefano Ermon. 2019. Generative modeling by estimating gradi ents of the data distribution. Advances in neural information processing systems 32 (2019).

[37] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. 2025. Kling-Omni Technical Report. arXiv preprint arXiv:2512.16776 (2025).

[38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).

[39] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571 (2023).

[40] Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, and Baoyuan Wu. 2025. Badvideo: Stealthy backdoor attack against text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19075–19084.

[41] Songping Wang, Rufan Qian, Yueming Lyu, Qinglong Liu, Linzhuang Zou, Jie Qin, Songhua Liu, and Caifeng Shan. 2025. RunawayEvil: Jailbreaking the Imageto-Video Generative Models. arXiv preprint arXiv:2512.06674 (2025).

[42] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH2024 Conference Papers. 1–11.

[43] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. 2023. Larger language models do in-context learning diferently. arXiv preprint arXiv:2303.03846 (2023).

[44] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2024. Dynamicrafter: Animating open-domain images with video difusion priors. In European Conference on Computer Vision. Springer, 399–417.

[45] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).

[46] Xiaoyu Ye, Songjie Cheng, Yongtao Wang, Yajiao Xiong, and Yishen Li. 2025. T2vunlearning: A concept erasing method for text-to-video difusion models. arXiv preprint arXiv:2505.17550 (2025).

[47] Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, and Dit-Yan Yeung. 2024. Anyattack: Towards large-scale self-supervised generation of targeted adversarial examples for vision-language models. arXiv e-prints (2024), arXiv–2410.

[48] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. 2023. I2vgen-xl: High-quality imageto-video synthesis via cascaded difusion models. arXiv preprint arXiv:2311.04145 (2023).

[49] Zhongwei Zhang, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Ting Yao, Yang Cao, and Tao Mei. 2024. Trip: Temporal residual learning with image noise prior for image-to-video difusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8671–8681.

[50] Zhengyu Zhao, Zhuoran Liu, and Martha Larson. 2020. Towards large yet imperceptible adversarial image perturbations with perceptual color distance. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1039–1048.

[51] Bowen Zheng, Yongli Xiang, Ziming Hong, Zerong Lin, Chaojian Yu, Tongliang Liu, and Xinge You. 2026. VII: Visual Instruction Injection for Jailbreaking

[52] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. 2024. Open-sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404 (2024).

Image-to-Video Generation Models. arXiv preprint arXiv:2602.20999 (2026).