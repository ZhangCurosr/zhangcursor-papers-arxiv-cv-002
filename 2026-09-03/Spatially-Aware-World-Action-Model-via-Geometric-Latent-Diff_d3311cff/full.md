# Spatially Aware World Action Model via Geometric Latent Diffusion

Javier Alejandro Lopetegui Gonzalez, Paul Pacaud and Cordelia Schmid

https://jlopetegui98.github.io/projects/sa wam.html

Abstract— World Action Models (WAMs) leverage the capabilities of large-scale pretrained video diffusion models to jointly predict future observations and actions, inheriting rich visual and physical priors from internet-scale video. This has made them a promising paradigm for robot policy learning, yet the prevailing models operate exclusively on RGB observations and do not leverage 3D information. To bridge this gap, we introduce a Spatially Aware World Action Model (SA-WAM), which repurposes a pretrained video model for joint action, RGB, and depth prediction, enabling 3D-aware world modeling and action prediction within a single diffusion backbone. We use a nonlinear encoding that maps the unbounded depth signal into the bounded input domain expected by the frozen VAE tokenizer. This allows us to reuse the tokenizer without 3D-specific fine-tuning, incorporating geometric information without sacrificing the pretrained priors. SA-WAM achieves state-of-the-art results on the RoboCasa and LIBERO-Plus benchmarks, while simultaneously improving future-state predictions. Furthermore, SA-WAM outperforms strong baselines in real-world evaluation using a UR5 robotic arm, with strong gains in randomized environments. We analyze the correlation between world model prediction quality and rollout success, providing insights into WAM performance and avenues for its improvement.

## I. INTRODUCTION

Vision-Language-Action (VLA) models have achieved impressive generalization by leveraging large-scale pre-training on vision, language and robot data [1]–[6]. However, VLA models are typically initialized from Vision-Language Models trained predominantly on static image and text data. Video generative models, trained on dynamic sequences, appear as a potentially stronger prior for modeling temporal evolution, motion and physical interactions relevant for robotic policies. Early approaches exploit these priors by generating future video observations and decoding actions through inverse dynamics [7]–[9]. Video-Action Models (VAMs) instead condition action prediction on video latents from a pretrained backbone [10], but typically require multi-stage video-and-action training. World Action Models (WAMs) or Unified World Models (UWMs) unify video and action generation in a shared generative architecture [11]– [15], showing that jointly learning actions and future states can improve policy performance.

However, current WAMs mainly operate on RGB observations and do not leverage any 3D information. In contrast, 3D-aware VLA policies have shown the value of 3D information in robotic policies, specifically for manipulationintensive tasks where object geometry plays a critical role [16]–[20]. Bringing this spatial awareness to WAMs remains largely unexplored.

![](images/030beda16315834b4ad93fc709ca45f5a037398a2ec443af8a576bd310889739.jpg)  
Fig. 1: State-of-the-art results on the RoboCasa benchmark. SA-WAM reaches 76.6% success rate on RoboCasa with only 50 demonstrations per task, improving over the matched Cosmos-Policy baseline by 9.5 points and outperforming prior approaches trained with 6–20× more data.

We propose the Spatially Aware World Action Model (SA-WAM). By incorporating explicit geometry into world–action modeling, our approach significantly enhances policy performance; see Figure 1. Following a latent-frame injection strategy for adapting latent video diffusion models to visuomotor policy learning [14], we augment the Diffusion Transformer (DiT) latent sequence with depth signal (Figure 2). Depth is encoded as additional latent frames and processed by the same frozen pretrained VAE tokenizer as RGB observations. The key challenge is to encode this unbounded metric depth into the frozen VAE feature space while preserving geometric precision without using modality-specific 3D encoders. Motivated by nonlinear parameterizations used in diffusion-based depth estimation [21,22], SA-WAM employs nonlinear depth normalization. The main intuition is to compress the far-field while preserving near-field resolution, where manipulation precision matters most.

Our main contributions are summarized as follows:

(i) We introduce SA-WAM, which integrates explicit 3D geometric information into World Action Models. We use nonlinear normalization of the depth to avoid training dedicated 3D encoders.

(ii) SA-WAM achieves state-of-the-art results on Robo-

![](images/858acc9d7e359d1ea4a98c624957209de6675c0ca084894beb19875eecf94d44.jpg)  
Fig. 2: SA-WAM architecture. SA-WAM injects geometry into the WAM latent sequence as tokenizer-compatible frames paired with each RGB view, while the proprioception (q<sub>t</sub>) and action chunk (a<sub>t</sub>) are inserted directly into dedicated positions in the latent sequence. The conditioning signals include the task description (task), one RGB frame per view (v<sup>c</sup>) interleaved with its corresponding 3D modality (g<sup>c</sup>) and the proprioceptive information at time t (q<sub>t</sub>). The action chunk, future-state RGB $( \nu _ { t ^ { \prime } } ^ { c } )$ , and future depth $( g _ { t ^ { \prime } } ^ { c } )$ frames undergo the denoising process. The decoding phase is optional at inference time.

Casa [23] with only 50 demonstrations per task (Figure 1) and on LIBERO-Plus [24]. Moreover, it obtains strong realworld results on a UR5 robotic arm platform, demonstrating its sample efficiency and robustness to randomized environments.

(iii) We investigate different normalization strategies for depth, showing how log-scale normalization leads to stronger policy performance while remaining better matched to the frozen video tokenizer’s distribution. Additionally, we analyze how geometric information improves WAM world modeling quality and study the correlation between future prediction quality and policy rollout success.

## II. RELATED WORK

a) World Models for Robot Policy Learning: World models have long been used in robot learning to predict future states for planning and control [25,26]. More recent video-based robot policies leverage generative video models as learned priors for robot behavior. Methods such as UniPi [7] and Dreamitate [9] synthesize future visual trajectories and recover the corresponding actions through inverse dynamics or action decoding. In contrast, Video-Action Models such as Mimic-Video [10] augment a pretrained video backbone with a dedicated action-prediction head and follow a two-stage training procedure: first training the video model, then learning the action module conditioned on the video latent representations. More recent World Action Models (WAMs) and Unified World Models (UWMs), including VideoPolicy [8], UWM [12], DreamZero [15], Motus [27], and Cosmos-Policy [14], jointly model future states and actions within a shared generative architecture. Cosmos-Policy is particularly relevant to our work because it repurposes the latent sequence of a pretrained video diffusion transformer to represent robot-specific modalities, such as proprioception and actions. This design preserves the foundational video model architecture without introducing modality-specific modules, while exploiting video pretraining without explicitly generating the complete future video sequence. These methods demonstrate the value of visual foresight for policy learning, but they primarily operate on RGB observations.

b) 3D-Aware Robot Policies: Explicit 3D observations have been widely used to improve manipulation policies by providing direct access to object geometry and spatial relations. PolarNet [28] predicts actions from languageconditioned point-cloud representations, while 3D-VLA [16] connects 3D perception, reasoning, and action through a generative world model. SpatialVLA [17] introduces egocentric 3D positional encodings and adaptive spatial action grids, whereas PointVLA [29] injects point-cloud features into a pretrained VLA through a lightweight module. PointACT [19] further couples hierarchical point-cloud features with action tokens through multi-scale interaction. PointMapPolicy [20] instead represents dense 3D coordinates as image-aligned PointMaps, allowing explicit geometry to be processed using standard visual encoders. Similarly, SA WAM uses a dense representation of the metric depth as its 3D modality and maps it into latent frames using the same frozen video tokenizer as RGB.

c) Metric Geometry Encoding for Pretrained Visual Models: Geometric information can be represented through dedicated modality-specific latent spaces or adapted to the representation domain of existing visual models. Diffusion-Depth [30] learns a compact depth-specific latent space using a lightweight autoencoder, enabling diffusion directly over depth representations. Alternatively, metric-depth methods commonly employ non-linear parameterizations to better distribute representational capacity across large depth ranges. DMD [21] uses log-scale depth to jointly model indoor and outdoor scenes, while SwinMTL [31] applies logarithmic scaling to emphasize the densely represented near-depth range. DepthFM [22] encodes log-normalized depth within the bounded input range of a pretrained latent autoencoder, showing improved performance over linear normalization. GRIN [32] similarly performs pixel-level diffusion in logdepth space to improve precision across large distance ranges. SA-WAM transfers this principle to WAMs: we encode depth using a nonlinear normalization that makes metric geometry compatible with a frozen video tokenizer while allocating greater resolution to manipulation-relevant regions.

## III. METHOD

## A. 3D Modality Injection for World Action Modeling

SA-WAM adapts a pretrained latent video diffusion model into a spatially aware World Action Model that jointly predicts robot actions and visual-geometric world evolution. Conditioned on the task instruction, current proprioception, and multi-view RGB-D observations, the model jointly denoises an action chunk together with the corresponding future RGB-D observations. Following [14], robot proprioception and actions occupy dedicated positions in the latent sequence. In contrast, RGB and depth observations are encoded into latent frames using the same frozen video tokenizer. This design repurposes the pretrained video model for unified world and action generation without introducing dedicated action heads or geometry-specific encoders, as illustrated in Figure 2.

Let $\mathcal { V } = \{ w , l , r \}$ denote the wrist, left, and right cameras, respectively. For each camera $c \in \mathcal { V }$ , let $I _ { t } ^ { c }$ denote the RGB observation and $d _ { t } ^ { c }$ its corresponding metric depth map at time t. We denote the frozen VAE encoder by $E ,$ and define the RGB and geometric latent frames as

$$
\nu _ { t } ^ { c } = E ( I _ { t } ^ { c } ) , \qquad g _ { t } ^ { c } = E ( I _ { d , t } ^ { c } ) ,\tag{1}
$$

where $I _ { d , t } ^ { c }$ is a tokenizer-compatible three-channel representation of the metric depth $d _ { t } ^ { c }$ , defined below.

Let ℓ denote the task instruction, $q _ { t }$ the robot proprioception, and $a _ { t }$ the action chunk starting at time t. The conditioning is $\mathcal { C } _ { t } = ( \ell , q _ { t } , \{ \nu _ { t } ^ { c } , g _ { t } ^ { c } \} _ { c \in \mathcal { V } } )$ . The model jointly denoises $a _ { t }$ and the future observation latents $\{ \nu _ { t ^ { \prime } } ^ { c } , g _ { t ^ { \prime } } ^ { c } \} _ { c \in \mathcal { V } }$ at $t ^ { \prime } = t + H$ , where H denotes the prediction horizon. Thus, RGB and depth occupy separate latent-frame positions while sharing the same pretrained tokenizer.

Reusing the frozen video tokenizer for depth requires mapping metric depth, which is positive and unbounded in principle, to the bounded input range expected by the VAE encoder. We therefore define a camera-specific normalization $f$ that transforms metric depth into a tokenizer-compatible image representation. We compare three choices: linear, inverse-depth, and log-scale normalization.

a) Depth Injection: For each camera c, let $d _ { t } ^ { c } ( u , \nu ) > 0$ denote the metric depth at pixel $\left( u , \nu \right)$ . Since the wrist and external cameras cover different depth ranges, we estimate robust camera-specific bounds from the training data. Let $\mathcal { D } _ { c }$ contain all valid depth values observed by camera c across the training set, and define $d _ { \operatorname* { m i n } } ^ { c } = P _ { 2 } ( \mathcal { D } _ { c } )$ and $d _ { \operatorname* { m a x } } ^ { c } = P _ { 9 8 } ( { \mathcal { D } } _ { c } )$ , where $P _ { p }$ denotes the p-th percentile. Each depth observation is then clamped as $d _ { t } ^ { c , \star } ( u , \nu ) =$ clip $( d _ { t } ^ { c } ( u , \nu ) , d _ { \operatorname* { m i n } } ^ { c } , d _ { \operatorname* { m a x } } ^ { c } )$

All depth encodings follow the same procedure. A monotone normalizer $f ( \cdot ; a , b ) : [ a , b ] \to [ 0 , 1 ]$ is applied to the clamped metric depth, after which its output is rescaled to the [−1,1] tokenizer input range and replicated across three channels:

$$
\begin{array} { r l } & { \hat { d } _ { t } ^ { c } ( u , \nu ) = 2 f \big ( { d } _ { t } ^ { c , \star } ( u , \nu ) ; { d } _ { \operatorname* { m i n } } ^ { c } , { d } _ { \operatorname* { m a x } } ^ { c } \big ) - 1 , } \\ & { I _ { d , t } ^ { c } ( u , \nu ) = \big ( \hat { d } _ { t } ^ { c } ( u , \nu ) , \hat { d } _ { t } ^ { c } ( u , \nu ) , \hat { d } _ { t } ^ { c } ( u , \nu ) \big ) \in [ - 1 , 1 ] ^ { 3 } . } \end{array}\tag{2}
$$

The resulting depth representation is encoded by the frozen VAE according to (1).

The three encoding strategies differ only in the choice of $f \colon$

$$
f ( z ; a , b ) = \left\{ \begin{array} { l l } { \displaystyle \frac { z - a } { b - a } , } & { \mathrm { l i n e a r , } } \\ { \displaystyle \frac { 1 / z - 1 / b } { 1 / a - 1 / b } , } & { \mathrm { i n v e r s e - d e p t h , } } \\ { \displaystyle \frac { \log \left( z / a \right) } { \log \left( b / a \right) } , } & { \mathrm { l o g - s c a l e . } } \end{array} \right.\tag{3}
$$

Their local sensitivity $| \partial f / \partial z |$ is constant for linear normalization, proportional to $1 / \dot { z } ^ { 2 }$ for inverse-depth normalization, and proportional to $1 / z$ for log-scale normalization. Linear normalization therefore distributes representational resolution uniformly across the metric range. Inverse-depth normalization concentrates it strongly on nearby geometry, whereas log-scale normalization provides an intermediate behavior, allocating greater resolution to the near field while retaining useful precision at larger distances. Each mapping is invertible over the retained depth interval, allowing predictions to be mapped back to metric depth for evaluation.

## B. Training Objective

We initialize the DiT backbone from the open-source Cosmos-Predict2 model [33], a pretrained video model, specifically trained for physical AI, that provides strong dynamic priors for robot policy learning. We use the corresponding Wan2.1 spatiotemporal VAE tokenizer, which remains frozen during training. Let $x _ { 0 }$ denote the clean target sequence that contains the action chunk together with the tokenizer latents of the future RGB and geometric frames. Conditioning information $\mathcal { C } _ { t }$ includes the task instruction, current proprioception, and current multi-view RGB-D observations. Following the EDM formulation [34], x is corrupted with Gaussian noise $n \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ , and the denoiser is optimized with $\mathcal { L } _ { \mathrm { E D M } } = \mathbb { E } \big [ \lambda ( \sigma ) \| D _ { \theta } ( x _ { 0 } + n ; \sigma , \mathcal { C } _ { t } ) - x _ { 0 } \| _ { 2 } ^ { 2 } \big ]$ where $\lambda ( \sigma )$ is the noise-dependent weighting. The same objective jointly supervises the action, future RGB and future depth slots, allowing all modalities to be learned within a single generative sequence without dedicated prediction heads or modality-specific losses.

## IV. SIMULATION EXPERIMENTS

## A. Experimental setup

a) Simulation Benchmarks: For our evaluation we use three simulation benchmarks, RoboCasa [23], LIBERO [35] and LIBERO-Plus [24].

RoboCasa [23] is a large-scale simulation benchmark for everyday kitchen manipulation, built on RoboSuite/MuJoCo with a Franka Panda arm. We evaluate on its 24 atomic tasks, including pick-and-place, articulated-object interaction, appliance control, and faucet manipulation. Each task provides two third-person cameras, a wrist-mounted camera, and proprioception. We report average task success rate over three random seeds.

LIBERO [35] is a simulation benchmark for robotic manipulation using a Franka Panda arm. We evaluate on its four standard suites: SPATIAL, OBJECT, GOAL, and LONG. The first three isolate spatial, object, and goal-conditioned generalization, while LONG evaluates longer-horizon task execution. Furthermore, we evaluate on LIBERO-Plus [24], which extends the same suites with controlled perturbations along seven dimensions (object layout, camera viewpoint, robot initial state, language instruction, lighting, background texture, and sensor noise). For fair comparison against baselines, we apply the same noise perturbation to depth as to RGB when evaluating sensor noise. Following its official zero-shot protocol, we train only on the original LIBERO demonstrations and evaluate directly on LIBERO-Plus, without any adaptation to the perturbed conditions.

b) Training Details: We fine-tune the 2B-parameter Cosmos-Predict2 DiT and keep the Wan2.1 VAE tokenizer frozen. For RoboCasa and LIBERO, training is run on 40 NVIDIA H100 GPUs with an effective batch size of 800 and 960, respectively. We optimize the EDM objective with AdamW using a peak learning rate of $1 0 ^ { - 4 }$ . After an initial warm-up, the learning rate is linearly decayed until 30,000 steps, at which point it is reduced by an additional factor of 5 and kept constant for the remainder of training. Training runs for 45,000 iterations on RoboCasa and 40,000 iterations on LIBERO. During training we sample noise using a hybrid approach, from the base CosmosPredict2 [36] log-normal distribution $p _ { \mathrm { b a s e } } ( \sigma )$ with probability 0.7 and from U(1,85) with probability 0.3 [14]. The model predicts action chunks of length H=32, of which the first 16 steps are executed open-loop.

## B. Ablations on the depth normalization approach

We conduct all ablations on the same eight RoboCasa tasks: PnPSinkToCounter, PnPCounterToSink, PnP MicrowaveToCounter, PnPCounterToMicrowave, OpenDrawer, TurnOnSinkFaucet, TurnOffSink Faucet and CoffeeServeMug. This subset spans pickand-place, articulated-object interaction, faucet manipulation, and mug-serving behaviors. We first analyze how different

TABLE I: Ablations on the eight-task RoboCasa subset. Metric depth reconstruction is reported as AbsRel (%, ↓). Overall values aggregate all camera views, while rangespecific values are reported for the wrist camera only. Policy success rate (SR, %) is averaged over 3 evaluation seeds × 50 trials per task. † denotes the configuration retained for subsequent experiments.

(a) Metric reconstruction
<table><tr><td>Normalization</td><td>Overall</td><td>&lt;0.3 m</td><td>0.3-1 m</td><td>&gt;1m</td></tr><tr><td>Linear</td><td>1.47</td><td>6.68</td><td>0.80</td><td>1.78</td></tr><tr><td>Inverse</td><td>1.41</td><td>0.62</td><td>4.16</td><td>13.19</td></tr><tr><td>Log</td><td>0.63</td><td>1.25</td><td>1.00</td><td>3.19</td></tr></table>

(b) Policy-level ablation
<table><tr><td>Normalization</td><td></td><td>SR (%)</td></tr><tr><td>RGB</td><td></td><td>52.7</td></tr><tr><td>RGB-D</td><td>Linear</td><td>56.5</td></tr><tr><td>RGB-D</td><td>Inverse</td><td>59.5</td></tr><tr><td>RGB-D</td><td>Log</td><td>61.7†</td></tr></table>

depth encodings preserve metric information through the tokenizer and then evaluate downstream policy performance.

a) Metric reconstruction analysis: We first focus on understanding the impact of each normalization approach on the ability of the frozen tokenizer to handle the geometric representation. We evaluate the normalization approaches discussed in Section III-A: linear, inverse and log-scale. Given a depth observation, we perform a closed encodedecode round trip and evaluate the reconstructed depth and report the Absolute Relative Error (AbsRel) with respect to the input. The corresponding overall results for the three cameras and the depth-range breakdown for the wrist camera are reported in Table Ia.

Linear and inverse normalization exhibit complementary failure modes. Linear depth preserves medium- and farrange geometry well, but substantially loses precision in the near field, reaching 6.68% AbsRel below 0.3 m. Inversedepth normalization shows the opposite behavior: it is highly accurate in the near field (0.62%), but it degrades sharply at medium and far distances (4.16% and 13.19%, respectively). Log normalization provides a more balanced profile, achieving the lowest overall error (0.63%) while maintaining comparatively low error across the wrist-camera depth ranges.

b) Policy-level ablation: We train all configurations under the same protocol, using 50 demonstrations per task and 6,000 gradient steps for faster iteration. As shown in Table Ib, all geometric approaches substantially improve over RGB-only method. Log-scale depth normalization improves over linear normalization by 5.2 points and over inverse normalization by 2.2, consistent with the more balanced metric precision observed above.

We therefore retain log-scale normalization for depth. It achieves the highest policy success while providing balanced metric precision across the workspace. Unless stated otherwise, all subsequent experiments use this configuration, denoted as SA-WAM.

TABLE II: Category-wise success rate (SR, %) on the 24 RoboCasa tasks. The categories contain 8 pick-and-place tasks, 6 open/close tasks, 7 turn/toggle tasks, and 3 coffee tasks. Category averages are computed from available per-task results. The final column reports the average over all 24 tasks. Best overall results in bold.
<table><tr><td>Method</td><td>Train. demos / task</td><td>PnP</td><td>Open/Close</td><td>Turn/Toggle</td><td>Coffee</td><td>Avg.</td></tr><tr><td>π0 [4]</td><td>300</td><td>一</td><td>一</td><td>一</td><td>一</td><td>62.5</td></tr><tr><td>VideoPolicy [8]</td><td>300</td><td></td><td></td><td></td><td>一</td><td>66.0</td></tr><tr><td>UWM [12]</td><td>1000</td><td>35.6</td><td>82.0</td><td></td><td></td><td>60.8</td></tr><tr><td>FLARE [37]</td><td>1000</td><td></td><td></td><td>一</td><td></td><td>71.3</td></tr><tr><td>GR00T-N1.5 [38]</td><td>300</td><td>53.8</td><td>85.7</td><td>61.7</td><td>54.0</td><td>64.1</td></tr><tr><td> $\mathrm { G R 0 0 T - N 1 . 5 + H A M L E T } \ [ 3 9 ]$ </td><td>300</td><td>48.5</td><td>87.7</td><td>69.7</td><td>64.0</td><td>66.4</td></tr><tr><td> $\mathrm { G R 0 0 T  – N 1 . 6 – f t } + \mathrm { W o r l d 2 A c t } \ [ 4 0 ]$ </td><td>300</td><td></td><td></td><td></td><td></td><td>72.6</td></tr><tr><td>Cosmos-Policy [14]</td><td>50</td><td>51.8</td><td>91.1</td><td>66.3</td><td>61.6</td><td>67.1</td></tr><tr><td>SA-WAM (ours)</td><td>50</td><td>68.5</td><td>92.6</td><td>77.5</td><td>64.7</td><td>76.6</td></tr></table>

TABLE III: Weighted-average success rate (SR, %) on LIBERO-Plus under zero-shot evaluation. Results are aggregated across the seven official perturbation axes and weighted by the number of episodes in each axis. Best result in bold.
<table><tr><td>Method</td><td>π0 [4]</td><td>π0-FAST [41]</td><td>OpenVLA- OFT [42]</td><td>RIPT-VLA [43]</td><td>π0.5 [5]</td><td>HoloBrain- 0-GD [44]</td><td>ABot-M0 [45]</td><td>Fast-WAM [46]</td><td>Cosmos- Policy [14]</td><td>SA-WAM (ours)</td></tr><tr><td>Avg. SR</td><td>53.6</td><td>61.6</td><td>69.6</td><td>68.4</td><td>84.6</td><td>74.0</td><td>80.5</td><td>50.0</td><td>81.4</td><td>86.6</td></tr></table>

## C. Comparison to the State of the Art

In this section we compare SA-WAM to existing robot policy baselines on the RoboCasa and LIBERO-Plus benchmarks.

a) RoboCasa results: Table II compares SA-WAM with strong VLA policies [4,38], video- and world-model approaches [8,12,14], and recent future-modeling or posttraining methods [37,39,40]. We report the average success rate across three seeds on the 24 RoboCasa tasks, together with the category-level breakdown when available. SA-WAM achieves the highest overall success rate at 76.6% using only 50 demonstrations per task, outperforming methods trained with substantially larger downstream data budgets. In particular, compared to Cosmos-Policy [14] under the same demonstration budget, SA-WAM improves by 9.5 points overall and performs consistently better across all task categories. The largest improvements occur on PnP and Turn/Toggle tasks, with gains of 16.7 and 11.2 points, respectively, indicating that explicit geometric conditioning is particularly beneficial for manipulations requiring accurate object localization and spatial interaction.

Figure 3 shows one representative pick-and-place task where SA-WAM completes the manipulation, whereas Cosmos-Policy fails. We observe that SA-WAM produces future predictions that remain better aligned with the corresponding simulator rollout; see IV-E for a quantitative evaluation.

b) LIBERO and LIBERO-Plus results: On standard LIBERO, SA-WAM achieves an average success rate of 98.4%, remaining on par with the strongest policies on this largely saturated benchmark. Table III compares SA-WAM with recent VLAs [4,5,41]–[45] and World Action Models [14,46] on the more challenging LIBERO-Plus benchmark under zero-shot evaluation. SA-WAM achieves the highest weighted-average success rate at 86.6%, exceeding the strongest VLA baseline, $\pi _ { 0 . 5 } ,$ by 2.0 points. Moreover, it improves over the strong Cosmos-Policy baseline by 5.2 points, indicating that explicit geometric conditioning improves WAM robustness to perturbations.

## D. World Modeling Quality Analysis

We analyze whether adding explicit geometry improves the visual world-modeling quality of SA-WAM. Predicted future RGB frames are compared with ground-truth simulator renders using PSNR [47], which measures pixel-level reconstruction fidelity; SSIM [48], which measures structural similarity; and LPIPS [49], which measures perceptual distance in a learned feature space. Higher PSNR and SSIM indicate better predictions, whereas lower LPIPS is better. Metrics are averaged across the 24 RoboCasa tasks and reported separately for the fixed-camera views (averaged over the left and right cameras) and the wrist camera. We additionally report masked metrics computed over the union of the robot arm, gripper, and target-object segmentation masks, thereby isolating prediction quality in manipulationrelevant regions [50]. As shown in Table IV, SA-WAM improves over Cosmos-Policy across all reported metrics. The gains are most pronounced for the dynamic wrist view, where close-range interaction and camera motion make future prediction particularly challenging. Improvements also persist under masked evaluation, indicating that they extend to the robot and manipulated object rather than arising only from background or global image statistics. The qualitative examples in Figure 3 complement these metrics: SA-WAM produces RGB predictions that remain more consistent with its realized rollout. Together with the corresponding gains in task success, these results support the hypothesis that, in WAM, stronger future-state modeling is associated with better policy performance.

TABLE IV: World-model RGB prediction quality on RoboCasa. F and W denote the averaged fixed-camera views and the wrist camera, respectively; subscript m denotes masked region. ↑ higher is better; ↓ lower is better.
<table><tr><td></td><td colspan="4">PSNR ↑</td><td colspan="4">SSIM ↑</td><td colspan="4">LPIPS ↓</td></tr><tr><td>Model</td><td>F</td><td>W</td><td> $\mathrm { F } _ { m }$ </td><td> $\mathrm { W } _ { m }$ </td><td>F</td><td>W</td><td> $\mathrm { F } _ { m }$ </td><td> $\mathrm { W } _ { m }$ </td><td>F</td><td>W</td><td> $\mathrm { F } _ { m }$ </td><td> $\mathrm { W } _ { m }$ </td></tr><tr><td>Cosmos-Policy</td><td>15.78</td><td>14.64</td><td>12.56</td><td>15.68</td><td>.525</td><td>.563</td><td>.400</td><td>.581</td><td>.167</td><td>.417</td><td>.219</td><td>.381</td></tr><tr><td>SA-WAM</td><td>15.97</td><td>15.16</td><td>12.82</td><td>16.08</td><td>.528</td><td>.566</td><td>.406</td><td>.585</td><td>.161</td><td>.354</td><td>.207</td><td>.321</td></tr></table>

![](images/cdb49a7b89b6650a1368ef0ab305093cf92c1586cc2476a716f52f5f5f75440f.jpg)  
Fig. 3: RoboCasa pick-and-place task: Pick the condiment bottle from the counter and place it in the sink. We show the wrist view before and during grasping, and at the final rollout stage. The upper block shows Cosmos-Policy and the lower block shows SA-WAM, with predictions compared against their corresponding simulator rollouts; for SA-WAM we also show depth predictions. Red boxes mark rollout inconsistencies, and the green box marks task completion.

## E. Prediction Quality versus Rollout Success

Next, we quantitatively study the correlation between future prediction consistency and rollout outcome. We focus on the manipulation-heavy pick-and-place (PnP) tasks for this analysis. We leverage the future depth prediction in SA-WAM to calculate an object-centric geometric error from the wrist camera by comparing the model-predicted future with the realized simulator rollout at a fixed short horizon after the grasp attempt. Specifically, we use the groundtruth simulator mask of the target object, back-project the predicted and ground-truth depth values within this mask into 3D, and measure the resulting object-region 3D prediction error. Figures 4(a) and (b) show that this object-level error is strongly associated with rollout outcome. Successful rollouts tend to keep the predicted object aligned with the realized object, whereas failed rollouts show larger divergence around the manipulation event; see Figure 4(a). Across PnP rollouts, successful episodes have lower 3D prediction error than failed ones; a threshold on this error yields $\mathsf { A U C } \approx 0 . 8 8$ detecting 80% of failures at a 13% false-alarm rate as we can see in Figure 4(b). These results suggest that task-relevant geometric future-prediction error can serve as a quantitative diagnostic of WAM failure modes.

![](images/d9d35385dfe2350b7ee2e09c5695ad097749bbb97f756372f22826052974a8f5.jpg)  
(a)

![](images/9132c8d3313e27f502294555d9b8061fa5d08e9f2493062a0ab7a39b1c9a9551.jpg)  
(b)  
Fig. 4: Future-prediction quality correlates with rollout success. (4a) Object-centric 3D prediction error is lower for successful rollouts. (4b) A threshold on this error detects failures with $\mathrm { A U C } \approx 0 . 8 8$ and an operating point of 80% detection at 13% false positive rate.

![](images/17163cd62d0811025616715b17885eca56ff13064787432f4a360d9bc1f5fb19.jpg)  
(a) Clean environments

![](images/bca73fbfcc0381c5d76b543f7896116b6b13a9fdeccb291b42c4ebd98e7b3018.jpg)  
(b) Randomized environments

Fig. 5: Real-world UR5 results. Task-completion scores over 10 trials per category for π , Cosmos-Policy, and SA-WAM in the clean and randomized settings. SA-WAM retains the strongest performance under randomization, consistent with depth providing a complementary spatial cue in the presence of visually confusing distractors.  
![](images/f699d283d5cc1f79c3dd307719490f67141816ca5a4d99f3c9a571638439620f.jpg)

(a) Task: take the pink mug and put it on the middle part of the hanger.  
![](images/ab5b67ae063f2a696c10f96b19c3b61a0603ca6d81a83cba9ec8a92a474f8232.jpg)  
(b) Task: put the grapes in the yellow plate, then put the banana in the pink plate.  
Fig. 6: Qualitative examples in randomized environments for the UR5 experiments: Two manipulation examples under domain randomization where SA-WAM reaches task completion while Cosmos-Policy fails to disambiguate visually similar distractors. We highlight rollout errors with red squares, and task completion with green squares.

## V. REAL-WORLD EXPERIMENTS

Real-world UR5 setup. We evaluate SA-WAM in a realworld tabletop manipulation setting, comparing it with Cosmos-Policy [14] and $\pi _ { 0 }$ [4]. The platform consists of a 6-DoF UR5 arm equipped with an RG6 parallel gripper, see Figure 6. The workspace is observed by a fixed Orbbec Femto Mega RGB-D camera, which provides time-aligned RGB images and metric depth.

Robot states and actions are represented in end-effector (EE) space. The state contains the current EE pose and gripper state, while each action specifies an absolute EE waypoint and gripper command. Both are represented as 8-D vectors, $( x , y , z , q _ { x } , q _ { y } , q _ { z } , q _ { w } , \mathrm { g r i p p e r } )$ , where $( x , y , z )$ denotes the EE position, $( q _ { x } , q _ { y } , q _ { z } , q _ { w } )$ its orientation as a unit quaternion, and the final scalar denotes the normalized gripper

aperture.

Actions correspond to semantic key-step waypoints rather than dense control-rate deltas or joint-space commands. The training dataset contains 20 demonstrations for each of 10 tasks, grouped into five categories: single-step, put-in-plates, put-in-box, mug-to-hanger, and stack-cups. For evaluation, we exclude the single-step category, as it consists of isolated primitive behaviors that also appear as subgoals within the composite tasks. Episodes contain 3–11 key steps, each providing an RGB-D observation together with the corresponding 8-D EE state and absolute EE action.

Training details. We train Cosmos-Policy and SA-WAM for 2,400 iterations on eight H100 GPUs with an effective batch size of 192. The learning-rate schedule consists of a 300- step warm-up followed by a linear decay to zero. The model predicts chunks of three key-step actions, of which the first two are executed at inference.

UR5 results. Figure 5 reports results on the four composite manipulation categories mentioned above. Each category is evaluated under clean and randomized conditions. In the clean setting, object positions differ from the demonstrations, while the remaining scene follows the training distribution. The randomized setting additionally introduces distractor objects and visual perturbations.

We perform 10 trials per category in each setting, corresponding to 20 trials per category overall. For multi-stage tasks, partial credit is assigned according to task progress: each completed subgoal contributes 0.5 in two-stage tasks and 0.25 in four-stage tasks. We compare SA-WAM with Cosmos-Policy [14] and $\pi _ { 0 }$ [4]. For $\pi _ { 0 } .$ , we initialize from the official pretrained weights and fine-tune the full model on the same key-step dataset with action horizon H = 3, using the LeRobot library [51].

Figure 5(a) shows the evaluation results in the clean setting. SA-WAM matches or exceeds both baselines in all four categories, reaching an aggregate completion score of 90.0%, compared with 75.0% for Cosmos-Policy and 21.3% for $\pi _ { 0 } .$ . Under randomization, SA-WAM retains 77.5%, whereas Cosmos-Policy and $\pi _ { 0 }$ decrease to 48.8% and 11.3%, respectively, as shown in Figure 5(b). This trend is consistent with explicit depth improving robustness when appearance alone is insufficient to distinguish the target from distractors.

Figure 6 shows representative randomized-environment cases in which SA-WAM succeeds and Cosmos-Policy fails on the same episode. The scenes contain layout changes and visually confusing distractors. These examples are consistent with depth providing a complementary spatial cue when appearance alone is ambiguous.

## VI. CONCLUSION

We introduced SA-WAM, a spatially aware World Action Model that injects explicit 3D information into the DiT latent-frame sequence. By leveraging a log-scale normalization, SA-WAM encodes depth into the input range expected by a frozen VAE tokenizer, enabling geometry-aware worldaction modeling without a dedicated 3D encoder. SA-WAM achieves state-of-the-art results on RoboCasa and LIBERO-Plus simulation benchmarks. Moreover, SA-WAM’s strong performance transfers to the real-world setting with a UR5 robotic arm. SA-WAM improves RGB prediction quality over the Cosmos-Policy baseline on the RoboCasa benchmark, highlighting the value of explicit spatial grounding for WAM-based robot policies.

Limitations. Although our approach improves over state-ofthe-art WAMs, the problem of predicting inconsistent futures and actions is still present. Geometrically consistent training and verification are possible avenues for future improvement. Moreover, further work is needed to improve the inference efficiency of WAMs.

## ACKNOWLEDGMENT

This work was granted access to HPC resources of IDRIS under the allocation AD011017145 made by GENCI. It was funded in part by the French government under management of Agence Nationale de la Recherche as part of the “France 2030” program, reference ANR-23-IACL-0008 (PR[AI]RIE-PSAI project) and the ANR project VideoPredict (ANR-21- FAI1-0002-01). Cordelia Schmid would like to acknowledge the support by the Korber European Science Prize. The ¨ authors thank Peteris Kulits, Federica Spinola and Zeeshan Khan for their valuable contributions to this project.

## REFERENCES

[1] B. Zitkovich, T. Yu, S. Xu, P. Xu, et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in CoRL, 2023.

[2] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, et al., “OpenVLA: An Open-Source Vision-Language-Action Model,” CoRL, 2024.

[3] D. Ghosh, H. Walke, K. Pertsch, K. Black, et al., “Octo: An Open-Source generalist robot policy,” in RSS, 2024.

[4] K. Black, N. Brown, D. Driess, A. Esmail, et al., “π<sub>0</sub>: A Vision-Language-Action flow model for general robot control,” arXiv, 2024.

[5] K. Black, N. Brown, J. Darpinian, K. Dhabalia, et al., “π <sub>.</sub> : a Vision-Language-Action model with Open-World Generalization,” in CoRL, 2025.

[6] P. Intelligence, B. Ai, A. Amin, R. Aniceto, et al., “π<sub>0.7</sub>: a steerable generalist robotic foundation model with emergent capabilities,” arXiv, 2026.

[7] Y. Du, S. Yang, B. Dai, H. Dai, et al., “Learning universal policies via Text-Guided video generation,” in NeurIPS, 2023.

[8] J. Liang, P. Tokmakov, R. Liu, S. Sudhakar, et al., “Video generators are robot policies,” arXiv, 2025.

[9] J. Liang, R. Liu, E. Ozguroglu, S. Sudhakar, et al., “Dreamitate: Realworld visuomotor policy learning via video generation,” CoRL, 2024.

[10] J. Pai, L. Achenbach, V. Montesinos, B. Forrai, et al., “mimic-video: Video-action models for generalizable robot control beyond VLAs,” ICLR, 2026.

[11] S. Li, Y. Gao, D. Sadigh, and S. Song, “Unified video action model,” in RSS, 2025.

[12] C. Zhu, R. Yu, S. Feng, B. Burchfiel, et al., “Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets,” in RSS, 2025.

[13] Y. Shen, F. Wei, Z. Du, Y. Liang, et al., “VideoVLA: Video generators can be generalizable robot manipulators,” in NeurIPS, 2025.

[14] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, et al., “Cosmos policy: Fine-Tuning video models for visuomotor control and planning,” in ICLR, 2026.

[15] S. Ye, Y. Ge, K. Zheng, S. Gao, et al., “World action models are zero-shot policies,” arXiv, 2026.

[16] H. Zhen, X. Qiu, P. Chen, J. Yang, et al., “3D-VLA: A 3D Vision-Language-Action generative world model,” in ICML, 2024.

[17] D. Qu, H. Song, Q. Chen, Y. Yao, et al., “SpatialVLA: Exploring spatial representations for visual-language-action model,” RSS, 2025.

[18] P. Li, Y. Chen, H. Wu, X. Ma, et al., “BridgeVLA: Input-Output alignment for efficient 3D manipulation learning with Vision-Language models,” in NeurIPS, 2025.

[19] S. Chen, P. Pacaud, and C. Schmid, “PointACT: Vision-Language-Action models with Multi-Scale Point-Action interaction,” RSS, 2026.

[20] X. Jia, Q. Wang, A. Wang, H. Wang, et al., “PointMapPolicy: Structured point cloud processing for multi-modal imitation learning,” NeurIPS, 2025.

[21] S. Saxena, J. Hur, C. Herrmann, D. Sun, and D. J. Fleet, “Zero-Shot metric depth with a Field-of-View conditioned diffusion model,” in ECCV 2024 Workshop on Learning to Understand Diverse Scenes (Wild3D), 2024.

[22] M. Gui, J. Schusterbauer, U. Prestel, P. Ma, et al., “DepthFM: Fast generative monocular depth estimation with flow matching,” in arxiv, 2024.

[23] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, et al., “RoboCasa: Large-Scale simulation of household tasks for generalist robots,” in RSS, 2024.

[24] S. Fei, S. Wang, J. Shi, Z. Dai, et al., “LIBERO-Plus: In-depth robustness analysis of vision-language-action models,” arXiv, 2025.

[25] D. Ha and J. Schmidhuber, “Recurrent world models facilitate policy evolution,” in NeurIPS, 2018.

[26] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, et al., “Learning latent dynamics for planning from pixels,” in ICML, 2019.

[27] H. Bi, H. Tan, S. Xie, Z. Wang, et al., “Motus: A unified latent action world model,” in CVPR, 2026.

[28] S. Chen, R. G. Pinel, C. Schmid, and I. Laptev, “PolarNet: 3D point clouds for Language-Guided robotic manipulation,” in CoRL, 2023.

[29] C. Li, Y. Zhu, J. Wen, Y. Peng, et al., “PointVLA: Injecting the 3D World into Vision-Language-Action Models,” RA-L, 2025.

[30] Y. Duan, X. Guo, and Z. Zhu, “Diffusiondepth: Diffusion denoising approach for monocular depth estimation,” in ECCV, 2024.

[31] P. Taghavi, R. Langari, and G. Pandey, “SwinMTL: A shared architecture for simultaneous depth estimation and semantic segmentation from monocular camera images,” in IROS. IEEE, 2024.

[32] V. Guizilini, P. Tokmakov, A. Dave, and R. Ambrus, “GRIN: Zero-Shot metric depth with Pixel-Level diffusion,” in 3DV, 2025.

[33] NVIDIA, “Cosmos-predict2: World simulation model for physical ai,” 2025. [Online]. Available: https://github.com/nvidia-cosmos/ cosmos-predict2

[34] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” NeurIPS, 2022.

[35] B. Liu, Y. Zhu, C. Gao, Y. Feng, et al., “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” NeurIPS, 2023.

[36] N. Agarwal, A. Ali, M. Bala, Y. Balaji, et al., “Cosmos world foundation model platform for Physical AI,” arXiv, 2025.

[37] R. Zheng, J. Wang, S. Reed, J. Bjorck, et al., “FLARE: Robot learning with implicit world modeling,” in CoRL, 2025.

[38] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da,˜ et al., “GR00T N1: An open foundation model for generalist humanoid robots,” arXiv, 2025.

[39] M. Koo, D. Choi, T. Kim, K. Lee, et al., “Hamlet: Switch your visionlanguage-action model into a history-aware policy,” in ICLR, 2026.

[40] A. Dinh Vuong, T. Van Vo, A. Sohail, H. Ding, et al., “World2Act: Latent Action Post-Training from World Model Dynamics,” arXiv, 2026.

[41] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, et al., “Fast: Efficient action tokenization for vision-language-action models,” RSS, 2025.

[42] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” RSS, 2025.

[43] S. Tan, K. Dou, Y. Zhao, and P. Krahenb ¨ uhl, “Interactive post-training¨ for vision-language-action models,” arXiv, 2025.

[44] X. Lin, T. Lin, Y. Du, H. Xie, et al., “HoloBrain-0 technical report,” arXiv, 2026.

[45] Y. Yang, S. Zeng, T. Lin, X. Chang, et al., “ABot-M0: VLA foundation model for robotic manipulation with action manifold learning,” arXiv, 2026.

[46] T. Yuan, Z. Dong, Y. Liu, and H. Zhao, “Fast-WAM: Do world action models need test-time future imagination?” arXiv, 2026.

[47] Q. Huynh-Thu and M. Ghanbari, “Scope of validity of PSNR in image/video quality assessment,” Electronics letters, 2008.

[48] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, 2004.

[49] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.

[50] J. Bardhan, P. Drozdik, J. Sivic, and V. Petrik, “Persistent robot world models: Stabilizing multi-step rollouts via reinforcement learning,” arXiv, 2026.

[51] R. Cadene, S. Aliberts, F. Capuano, M. Aractingi, et al., “LeRobot: An open-source library for end-to-end robot learning,” arXiv, 2026.