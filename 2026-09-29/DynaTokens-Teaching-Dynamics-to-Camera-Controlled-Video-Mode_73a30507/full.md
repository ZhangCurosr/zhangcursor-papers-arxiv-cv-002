# DynaTokens: Teaching Dynamics to Camera-Controlled Video Models at Test Time

Ziqi Ma Hongqiao Chen Georgia Gkioxari California Institute of Technology

## Abstract

Video generation must account for two sources of motion, one induced by the observer’s camera path and the other caused by scene dynamics. An ideal cameracontrolled video model should account for both motions: let users move the camera while evolving the scene dynamics. While current models handle camera-induced motion well in static settings, they struggle for dynamic scenes: objects are static, move incorrectly, or degrade in generation quality. We introduce DynaTokens, a lightweight set of learnable scene-specific tokens that teach dynamics to an existing camera-controlled world model. Our method is motivated by a simple asymmetry between the two sources of motion: whereas camera motion affects the generated view globally, object dynamics are spatially localized. Through cross-attention, DynaTokens trains the learnable tokens from a few example trajectories for a scene while keeping the base model frozen, and enables dynamics under new query camera paths. DynaTokens achieves a better simultaneous dynamics-camera tradeoff on VBench2 and WorldScore evaluations than LoRA, block finetuning, and specialized trainable-layer baselines. Analyses of token attention, ablations, and motion temporality suggest that matching the trainable interface to the structure of the learning target is important for effective adaptation. Project website: https: //glab-caltech.github.io/dynatokens/

## 1 Introduction

Current camera-controlled video models [23, 25, 2, 22, 30, 10] can generate navigable “worlds” from a image or text prompts, opening new possibilities for gaming, immersive media, and robotics. However, despite their ability to follow camera trajectories, these models struggle with dynamics [31, 18]: the generated worlds often looks static, generate incorrect motion, or degrade in visual quality when the scene should evolve.

Recent work has shown that test-time training can adapt video foundation models to support new capabilities, including long-horizon generation [7], scientific discovery [32], and reasoning [1]. However, enabling scene dynamics in camera-controlled video models remains challenging because current architectures inject camera information directly into the generative flow process – via PRoPE [15], Plücker ray embeddings [33], or action conditioning [23, 10] – entangling camera control with dynamics generation. As a result, supervision on dynamics can inadvertently degrade camera control, and naive fine-tuning methods such as LoRA [11] perform poorly. This observation motivates a new approach to model adaptation. We introduce DynaTokens, a test-time training method that selectively teaches scene dynamics while preserving the camera-control behavior.

DynaTokens builds on a simple principle: in video generation, camera control is global, influencing every patch across every frame, whereas scene dynamics can be spatially local, often confined to the objects that move or change. To exploit this distinction, DynaTokens introduces learnable tokens that steer the video latents toward localized dynamics without altering the global scene generation. By decoupling local dynamic control from global camera control, this design enables dynamic scene generation while preserving camera controllability.

DynaTokens is lightweight: a few scene-specific tokens are added and learned while keeping the base model frozen. It can be trained from only a few samples (≈ 15) and generalize to unseen camera trajectories, producing videos under novel camera controls. Importantly, these few training trajectories need not be high quality: they can be short, contain background artifacts or motion discontinuity. Through test-time training of the tokens, DynaTokens can overcome these limitations, generating temporally continuous videos with prompt-aligned dynamics for new trajectories.

We show quantitatively and qualitatively that DynaTokens effectively enables dynamics without compromising camera control, surpassing current capabilities of state-of-the-art camera-controlled world models on VBench2 [35] dynamic-related categories and WorldScore [8]. Baseline methods such as LoRA, block finetuning, or Test-Time Training layers struggle. Beyond dynamic motion categories, we show that DynaTokens can also enable physics and stylistic changes.

We provide analysis on the DynaTokens design to understand the disentangling of dynamics with camera control. We also probe the temporality aspects of dynamic tokens to better understand how they elicit dynamic behavior of base models. Our analyses on test-time training of dynamic tokens provide deeper insight into architecture and representation of world models, paving the way for future development of large-scale world models. We also illuminate how to effectively adapt these models at test time to enable new capabilities. Additionally, DynaTokens offers a path toward video data generation. Existing efforts in collecting dynamic, camera-annotated data rely on synthetic environments, such as games [6]. DynaTokens, via efficient test-time training, opens up generation of camera-controlled videos in diverse settings.

Our main contributions are as follows:

• We develop DynaTokens, which significantly improves dynamic scene generation for cameracontrolled video models via test-time training, evaluated on VBench2 and WorldScore.

• DynaTokens is a new architecture design which decouples dynamics from camera control, enables much better performance than alternatives like LoRA and TTT layers.

• We identify a structural mismatch between existing adaptation methods and the underlying transformations they model: LoRA-type methods apply global parameter updates, whereas object dynamics is spatially localized. We provide analysis on how DynaTokens overcomes this limitation via a design that aligns with the learning target.

## 2 Related work

Camera conditioning in video models. Camera-controlled video models are flow-based video generation models that take in camera control, along with image and text, as conditioning signals. Camera conditioning can be achieved in 4 ways: noise warping [4, 22] which warps the noise based on camera movement, PRoPE [23, 31] which encodes camera frustum in positional embedding, action conditioning [23, 10] which encodes discrete camera actions, and plücker ray embedding [25, 22], usually channel-wise concatenated. We design DynaTokens to minimally affect camera conditioning and selectively “teach” dynamics.

Dynamics in camera-controlled video models. Camera-controlled video models [23, 25, 22, 10] struggle with dynamics due to a combination of 3D prior [2, 22] and the lack of dynamics in training data [17]. Methods that tackle dynamics can fall into two categories: (1) Improving existing general purpose world models. These methods mostly focus on loosening 3D constraints in memory, such as [6, 9, 31], which addresses out-of-frame dynamics. (2) Alternative formulations of 4D-aware generation models, such as BulletTime [29] which allows both camera and time control, but is limited to short and small-range camera movements. The recent Atlas model [26] enables camera-controlled dynamics generation, but requires multiple input views.

Test-time training. Test-time training methods train foundation models at test time to surpass base model capabilities in challenging tasks, such as the ARC challenge [1], scientific discovery [32], and long video generation [7]. Architecturally, test-time training methods either use LoRA [1] or specialized architectures like Test-time Training (TTT) layers [34, 7]. We design DynaTokens, a new test-time training architecture for world models, to enable localized dynamic learning.

![](images/fd2fc1387a68afffa86f251c92511ec8d95a5c0f5f92566e6265d1b4b52beb15.jpg)  
Figure 1: DynaTokens enables camera-controlled video models to learn dynamics via test-time training. (a) DynaTokens introduces learnable tokens that are cross-attended by the video patch tokens in every transformer block. (b) For each scene, we train DynaTokens on a small set of camera trajectories (≈15), enabling inference on novel, unseen trajectories. (c) Visualization of the attention weights for a dynamic token, corroborating the localized dynamics hypothesis.

## 3 Method

DynaTokens introduces scene-specific tokens into a pretrained camera-controlled video generation model. During training, the base model is kept frozen, and only the tokens are updated, enabling correct dynamics when exploring different camera trajectories for the scene. We motivate the design of dynamic tokens, then explain our architecture, test-time training setup, and training objective.

## 3.1 Motivating example

Consider a simple setting where the world is a 2D coordinate frame, in which a filled unit circle, $P \subseteq \mathbb { R } ^ { 2 }$ , is present, similar to the setting in [16]. The observer frame starts identical as the world frame. The goal of a video generative process is to recover the world at time t from the observer frame under both observer and scene motion.

![](images/33210365b808d694923c1e86925cf3fb1bc999b4b35f67530b9d1463df4ed485.jpg)  
Figure 2: Motivating example.

$\psi _ { t }$ which represents the “observer-to-world” mapping: $\psi _ { t } ( \mathbf { x } _ { \mathrm { o b s } } ) = \mathbf { x } _ { \mathrm { w o r l d } } { \mathrm { ~ f o r ~ } } \mathbf { x } _ { \mathrm { o b s } } \in \mathbb { R } ^ { 2 }$ , where $\mathbf { x } _ { \mathrm { w o r l d } }$ is a point’s coordinate in the world frame and $\mathbf { x _ { \mathrm { o b s } } }$ its coordinate in the observer frame. Assume $\psi _ { t } : ( x , y )  ( x + t , y )$ (observer moves right). If the scene is static, in the observer frame, this is equivalent to $\psi _ { t } ^ { - 1 }$ applied to object P: $P _ { t } = \psi _ { t } ^ { - 1 } ( P ) = \{ ( x + t , y ) : ( x , y ) \in P \} - P$ moves left. In the observer frame, the scene is represented as $f ( x , y , t ) = \mathbb { 1 } _ { \{ ( x + t ) ^ { 2 } + y ^ { 2 } \leq 1 \} }$ (top right of Fig 2).

Dynamic scene: Consider $P$ to be a dynamic object, evolving with $\phi _ { t } ( P ) \subseteq \mathbb { R } ^ { 2 }$ . Unlike $\psi _ { t }$ which has global effect on the whole frame, $\phi _ { t }$ is only applied to the local subset of object points, $P .$ Assume $\phi _ { t }$ represents uniform-speed “up” movement: $\phi _ { t } : ( x , y )  ( x , y + t )$ in the world frame. In the observer frame, this is equivalent to $P _ { t } = \psi _ { t } ^ { - 1 } ( \phi _ { t } ( P ) ) = \{ ( x - t , y + t ) : ( x , y ) \in P \} - P$ moves left and up. The world from the observation frame can be represented as $f _ { \mathrm { d y n a } } ( x , y , t ) = 1 _ { \{ ( x + t ) ^ { 2 } + ( y - t ) ^ { 2 } \leq 1 \} }$ shown in Fig 2 bottom right.

Localized dynamics hypothesis. In a generative context, we care about the effect of object dynamics on the scene in the observer frame, $f ,$ which describes the “static world” and $f _ { \mathrm { d y n a } }$ which describes the “dynamic world”. We define operator D to capture its effect, $f _ { \mathrm { d y n a } } = D f$

The motivating example above illustrates an important observation: at any given time, the effect of $D$ is limited to $P ^ { * } { \bf s }$ movement. Thus, we can decompose D into a dynamics function and localized gating function: $( D f ) ( x , y , t ) = m ( x , y , t ) \cdot ( G f ) ( x , y , t )$ , where $m ( x , y , t )$ is a binary gating function that localizes the dynamics and G applies the dynamics. In our example, m $( x , y , t ) =$ $\mathbb { 1 } _ { \{ ( x + t ) ^ { 2 } + ( y - t ) ^ { 2 } \leq 1 \vee ( x + t ) ^ { 2 } + y ^ { 2 } \leq 1 \} }$ , and G can be defined as $( G f ) ( x , y , t ) = f ( x , y - t , t )$

Similar to the motivating example of a circle in a plane, object dynamics in the real world are also localized: “a dog running” should affect only the dog region while preserving the rest of the scene. In other words, $\bar { \cal D ( f ) } = \bar { m _ { \odot } } \bar { G } ( f )$ . We build upon a base model that obeys camera control but largely keeps the scene static, i.e., generates $f ,$ and turns it into $f _ { \mathrm { d y n a } }$ which captures both camera and scene dynamics. We do this by recovering D, which we decompose into a localized gating function m which identifies where dynamics should occur, and a dynamics operator G that enables the motion.

## 3.2 Our approach: dynamic tokens

To achieve this, we design a lightweight add-on architecture for pretrained camera-controlled video models, enabling test-time learning of scene dynamics while preserving camera control. This presents two requirements: (1) the architecture needs to have enough representation capacity to “steer” the structure of the generated video latents while being lightweight; (2) the design should enable dynamics and camera control simultaneously.

We introduce DynaTokens, as illustrated in Fig. 1(a). Dynamic tokens are added to transformer blocks of a video generative model. The input video patch tokens interact with the learned dynamic tokens via cross attention. The output is added back to the model, along with the two paths of self-attention, with and without camera control (often instantiated via PRoPE [15]). We apply this to every double stream transformer block. The projection matrix that mixes multiple heads of cross-attention is initialized at zero to ensure minimal perturbation at initialization. We keep the base model frozen, and learn only the dynamic tokens at test time.

For each block, we introduce K dynamic tokens $\mathcal { D } = \{ d _ { j } \in \mathbb { R } ^ { m } \} _ { j = 1 } ^ { K } . \mathrm { I f } \ \mathcal { H } = \{ h _ { i } \in \mathbb { R } ^ { n } \} _ { i = 1 } ^ { N }$ are the video tokens input to the block, the output of cross-attention is represented as:

$$
h _ { i } ^ { \prime } = h _ { i } + \sum _ { j } a _ { i j } \cdot v ( d _ { j } ) , \qquad a _ { i j } = { \frac { \exp \bigl ( q ( h _ { i } ) ^ { \top } k ( d _ { j } ) \bigr ) } { \sum _ { l } \exp \bigl ( q ( h _ { i } ) ^ { \top } k ( d _ { l } ) \bigr ) } } .\tag{1}
$$

The attention weights, $a _ { i j }$ , act as a gating function across the video frames, and enable application of dynamics only on relevant patches, essentially serving the role of m in our localized dynamics motivation above. Since the MLP after the attention block is applied channel-wise $( i . e . ,$ , not mixing different patch tokens), this locality is preserved across transformer blocks. We train K = 16 dynamic tokens of dimension $m = 6 4$ in our experiments.

In Fig. 1(c), we visualize this gating mechanism in a real video by showing the attention weights applied on frames of a video generation with prompt: “The dog is on the left of the table, then it runs to the front of the table”. The attention weights localize the dynamic regions, following the target object (dog) as it moves over time. We provide qualitative and quantitative analysis of attention map localization in §D in the Appendix.

## 3.3 Test-time training of dynamic tokens

Setup. Dynamic tokens are trained at test time to enable dynamic video generation. While largescale data curation with rich scene and camera motion is difficult, DynaTokens’ lightweight design enables learning only from a few videos (≈15), which are easier to curate. We design a curation process leveraging VLMs and video interpolation, detailed in §G of the Appendix, which yields videos with dynamics. These videos might contain artifacts, and we only curate them for a few camera trajectories. Trained on these few trajectories, DynaTokens enables correct dynamics on new camera trajectories. Our setup is illustrated in Fig. 1(b).

However, sample curation alone is insufficient: fine-tuning or LoRA-adapting a pretrained cameracontrolled video model on the same data yields poor results, as we show in our experiments. In contrast, DynaTokens learns to generate the right dynamics under novel, unseen camera trajectories.

Training objective. We build on pretrained camera-controlled video models. As a representative architecture, we consider autoregressive video diffusion models, which generate videos chunkby-chunk through diffusion. We instantiate our method on HY-WorldPlay [23], a state-of-the-art camera-controlled video model.

Training video models is challenging: even with discretized camera actions, a single scene can admit over $1 0 ^ { \breve { 1 } 9 }$ possible trajectories over a 5-second clip. From only a handful of training trajectories $( \approx 1 5 )$ we aim to teach the add-on dynamic tokens to capture scene motion and generalize to unseen camera paths. To reduce exposure bias, where the model conditions on its own previous-chunk predictions that drift from the training setting, we leverage a diffusion-forcing-type [5] objective,

similar to [23, 22]:

$$
\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { l , t } \left[ \left. v _ { \theta } ( x _ { l } ^ { ( t ) } , \tilde { x } _ { 1 : l - 1 } ^ { ( t ) } , t ) - \left( x _ { l } - x _ { l } ^ { ( 0 ) } \right) \right. ^ { 2 } \right]\tag{2}
$$

where t is the flow timestep and l is the chunk index. We provide ablations on our design choices and hyperparameters in §5.

## 4 Experiments

We demonstrate the impact of DynaTokens: (1) whereas all current state-of-the-art camera-controlled video models struggle greatly with dynamics, DynaTokens can effectively enable dynamics within such models via test-time training; (2) alternative test-time training methods, such as LoRA [11] or TTT layer [7], fail, while DynaTokens enables learning dynamics while maintaining camera control. We evaluate on a broad range of videos with dynamic changes, including motion ordering, dynamic spatial relations, text-based motion accuracy, physics, and style transformations.

Benchmarks and Metrics. Since there are no existing dynamics-focused benchmarks for cameracontrolled world models, we repurpose existing video benchmarks for dynamics-related evaluation: VBench2 [35] and WorldScore [8]. We evaluate two aspects: dynamics correctness and camera control correctness.

VBench2 is a text-to-video benchmark with VLM judges. While VBench2 holistically evaluates video models, we use its dynamics-related categories, “Dynamic Spatial Relations” (DSR, e.g., “a dog is on the left of the table, then the dog runs to the front of the table”) and “Motion Order Understanding” (MOU, e.g., “a person is cooking, then they suddenly start organizing the pantry”). The original evaluation assumes a static camera, thus we make slight modifications to accommodate camera movement, such as not evaluating on frames where the main object is out-of-view due to camera movement. For camera, we calcaulte an aggregate score over rotation and translation error like WorldScore [8], leveraging state-of-the-art camera estimation method, ViPE [12]. We evaluate on a random subset of 20 prompts across the two categories, each for 6 different camera trajectories, on a total of 120 videos for VBench. Standard errors are reported in the Appendix.

WorldScore provides an input image and text prompt, and evaluates on optical-flow-based heuristics for dynamics. We focus on “Motion Accuracy” (MA) and “Camera Control” (Cam) scores. The original motion evaluation assumes a static camera, thus we make slight modifications to accommodate the moving camera, detailed in §F of the Appendix. We evaluate on 15 scenes selected randomly from the benchmark, each with 6 different camera trajectories, on a total of 90 videos for WorldScore. Standard errors are reported in the Appendix.

## 4.1 Comparing with state-of-the-art camera-controlled models

State-of-the-art camera-controlled models struggle with dynamics. While methods like Hydra [6] and LiveWorld [9] improve out-of-frame dynamics by modifying their memory, they still struggle with in-frame dynamics. We show that DynaTokens, via test-time training, effectively enables dynamics.

We compare with a suite of general camera-controlled video models – WorldPlay [23], Lingbot [25], and Lyra2 [22] – as well as dynamics-focused methods – Hydra [6] and LiveWorld [9]. For WorldPlay, we evaluate their camera-post-trained checkpoint. We also compare with Genie3 [10] qualitatively on a small set of samples due to the UI-only access and queue time. Refer to §F for more details.

Table 1(a) shows results on VBench2 “Dynamic Spatial Relation” (DSR), “Motion Order Understanding” (MOU), WorldScore “Motion Accuracy” (MA), and cameral control (Cam) on both benchmarks. Current models have clear failure modes, as shown in Fig. 3 (top):

• Static scenes: this is a clear failure mode for both WorldPlay and Lyra2 in almost all examples.

• Incorrect motion: Lingbot tends to generate wrong motions, such as the bird flying away or the cat not jumping. Genie 3 similarly generates wrong motions.

• Duplicate subject: LiveWorld splits one object into multiple, such as the 2 birds.

• Poor visual quality: Hydra often outputs blurry scenes (bird example).

More qualitative examples can be found in §A of the Appendix, and in the project webpage. These failures showcase that current camera-controlled models, when evaluated zero-shot, struggle with

A bird is behind an apple, then it flies to the right of the apple. A cat is playing with a toy, then it suddenly jumps onto a shelf.

![](images/282af9caf0a70a1f234cb33e019e68bad543cf915ba7150635bf1b36d43eba45.jpg)  
Figure 3: Top: Comparison of DynaTokens with state-of-the-art camera-controlled video models. DynaTokens consistently enables dynamics, whereas existing models struggle, either keeping the object static, show incorrect motion, or produce low-quality videos. Bottom: Comparison across different test-time training strategies. Only DynaTokens is able to correctly learn dynamics while maintaining camera control. The camera control is overlayed on the frames using keyboard schema, “wasd” means translation and arrows mean rotation.

dynamics. As shown in Table 1(a), DynaTokens improves dynamics via test-time training, resulting in a 28% improvement in dynamic spatial relations, 35% in motion ordering on VBench2, and a 21% improvement in WorldScore motion accuracy. Additionally, DynaTokens maintains camera control on both benchmarks.

<table><tr><td rowspan="2"></td><td colspan="3">VBench2</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>Lyra2</td><td>0.54</td><td>0.17</td><td>0.87</td><td>1.11</td><td>0.70</td></tr><tr><td>Lingbot</td><td>0.68</td><td>0.25</td><td>0.82</td><td>4.37</td><td>0.64</td></tr><tr><td>WorldPlay</td><td>0.56</td><td>0.07</td><td>0.88</td><td>1.95</td><td>0.69</td></tr><tr><td>Hydra</td><td>0.51</td><td>0.18</td><td>0.61</td><td>2.51</td><td>0.56</td></tr><tr><td>LiveWorld</td><td>0.44</td><td>0.35</td><td>0.82</td><td>4.49</td><td>0.70</td></tr><tr><td>DynaTokens</td><td>0.96</td><td>0.69</td><td>0.87</td><td>5.33</td><td>0.69</td></tr></table>

(a)

<table><tr><td rowspan="2"></td><td colspan="3">VBench2</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>DynaTokens</td><td>1.00</td><td>0.72</td><td>0.88</td><td>4.46</td><td>0.67</td></tr><tr><td>LoRA</td><td>0.61</td><td>0.22</td><td>0.83</td><td>2.79</td><td>0.56</td></tr><tr><td>LoRA-no PRoPE</td><td>0.41</td><td>0.06</td><td>0.84</td><td>3.62</td><td>0.67</td></tr><tr><td>Fine-tuning</td><td>0.59</td><td>0.67</td><td>0.83</td><td>2.66</td><td>0.54</td></tr><tr><td>TTT Layer</td><td>0.41</td><td>0.11</td><td>0.88</td><td>3.79</td><td>0.66</td></tr></table>

(b)  
Table 1: (a) Comparing DynaTokens with state-of-the-art camera-controlled video models. We evaluate Dynamic Spatial Relation (DSR) and Motion Order Understanding (MOU) on VBench2 and Motion Accuracy (MA) on WorldScore, as well as camera. Current state-of-the-art models struggle greatly with dynamics. DynaTokens, via test-time training, effectively learns dynamics while maintaining camera. (b) DynaTokens outperforms alternative test-time training methods, LoRA, finetuning, and TTT layer for dynamics. Standard errors are in Table 4 and Table 5 in Appendix.

We additionally perform human evaluation, shown on the right. We perform pairwise ranking between DynaTokens and each baseline. For every baseline, we show DynaTokens’ win rate against it across 100 rankings done by 5 independent annotators. Dyna-Tokens shows clear advantage of 71%-95.5% win rate for dynamics. DynaTokens outperforms all other baselines except for WorldPlay on camera, with which it is a close match at 48%.

![](images/25954b03b829580066b8310ada199014928606a6f9abd1234935e91cc76dc2f0.jpg)

## 4.2 Comparison with other test-time methods

We have shown DynaTokens to effectively “teach” camera-controlled models dynamics while main taining camera via test-time training in the previous section. Now, we further show that alternative test-time training methods are not as effective. We compare DynaTokens with LoRA, modified LoRA to preserve the PRoPE branch (LoRA-no PRoPE), fine-tuning last model blocks, and TTT layer for video models introduced by [7].

Table 1(b) shows results on VBench2 (DSR, MOU) and WorldScore. For each method, we use WorldPlay as the base model, and train on the same trajectories. We evaluate on 3 seen and 3 unseen camera trajectories for each scene to calculate the average score for dynamics and camera control. We test-time train each method on 12 distinct scenes and evaluate a total of 72 videos.

As shown in Table 1, DynaTokens shows clear advantage in dynamics learning across both benchmarks and all metrics, corroborating our hypothesis that disentangled design for localized dynamic learning enables effective learning while maintaining camera control. In the qualitative examples shown in Fig. 3(bottom), all other methods fail to fully learn dynamics, such as making the person “get up and stretch”, or making the kangaroo “jump to the right of the box”. More qualitative examples can be found in §A in the Appendix and the project webpage.

LoRA, although high in representation capacity, cannot effectively disentangle dynamics from camera, and holistically “steers” the generation output, thus unable to effectively learn dynamics and yields worse camera control. We show additional analysis on LoRA and LoRA-no PRoPE in §5. Fine-tuning the last transformer blocks is also not sufficient for learning dynamics. TTT Layer, although preserving camera, shows distorted objects, as shown in the human example of Fig. 3, and is unable to effectively learn dynamics. More qualitative examples can be found in the appendix, Fig. 11 and Fig. 12 in the Appendix. We provide further analysis on the failure of LoRA and its effect on PRoPE in §5.

![](images/86e6cd41c8c6007f64068a0862baf63f459395779fb772776b68d0944d18b218.jpg)

In addition to superior performance in both dynamics and camera, DynaTokens is also faster: as shown in Fig. 4, DynaTokens only adds 7.9% inference time compared to the base model (WorldPlay), while showing great improvement in dynamics. In comparison, LoRA has a much higher overhead of 57.7% inference time, despite having fewer parameters (30M = 0.3% of base model 8.6B parameters) than DynaTokens (240M, 2.7% base model parameters). Inference is evaluated on 4 A100 GPUs.

## 4.3 Generalization to Unseen Camera Paths

The space of all possible camera paths is large – given the duration of videos we generate (up to 93 frames), there are more than $1 0 ^ { 1 9 }$ total possible camera trajectories. While trained only trains on ≈ 15 camera paths, DynaTokens achieves generalization to diverse camera poses. Table 2 compares DynaTokens’ performance on seen and unseen camera paths on VBench2. Each evaluated unseen and seen path differ by 39.4 degrees geodesic distance on average, and differ completely in the first 28 frames’ action, demonstrating sufficient spread in the exploration trajectories.

<table><tr><td>Trajectories</td><td>Dynamics</td><td>Camera</td></tr><tr><td>Seen</td><td>0.83</td><td>0.85</td></tr><tr><td>Unseen</td><td>0.81</td><td>0.86</td></tr></table>

Table 2: DynaTokens demonstrates strong generalization, evaluated on VBench2.

## 4.4 Enabling diverse dynamics: physics and stylistic change

We evaluate whether DynaTokens can handle diverse dynamics such as physics and style transformations. While DynaTokens is designed to learn localized dynamics, we stress-test our approach for global style transformations to probe its capability.

![](images/fedc3ceb2d7befac6cafa57213d9a4415f6d4ef31d8dfbd49e2d6c0c9f3f3c54.jpg)  
Figure 5: DynaTokens enables diverse dynamics, including physics and stylistic changes, whereas current state-of-the-art camera-controlled video models like Lingbot and Genie 3 struggle.

We show qualitative examples on Physics IQ [19] and StevoBench [18]. As shown on the top of Fig. 5, the best open-source state-of-the-art model, Lingbot, either fails to evolve physics (ball not rolling) or completely fails to turn the camera. Genie 3, a closed-source state-of-the-art camera-controlled model, also fails to evolve physics correctly. DynaTokens successfully learns to evolve physics in these settings. We evaluate qualitatively since Physics IQ’s evaluation is based on pixel-wise alignment (IoU, MSE) under a static-camera ground truth, which cannot evaluate camera-controlled models. More examples can be found in Fig. 13 in the Appendix.

As shown on the bottom of Fig. 5, DynaTokens is capable of stylistic changes such as changing a cityscape into black-and-white line art, or transforming a room into a spaceship, while allowing the viewer to navigate the scene with camera control. Lingbot fails to transform the scene in both cases.

## 5 Ablations and Analyses

Why does LoRA fail to capture dynamics? Table 1 shows that LoRA fails to simultaneously enable dynamics and preserve camera. We provide further analysis.

(1) Non-localized effect. While DynaTokens enables localized gating, LoRA is unable to represent such localized functions since its update is “global” to all patches: $\begin{array} { r } { \breve { F } ( h _ { i } ) = ( W + B A ^ { \top } ) \dot { h } _ { i } } \end{array}$ . The matrix multiplication formulation results in “patch mixing”, and introduces global changes that interfere with camera. (2) Preserving PRoPE. For models that use PRoPE, the QKV matrix updates from LoRA result in a shift in the PRoPE product, as detailed in §H of the Appendix. To alleviate this, we also experiment with LoRA-no PRoPE, which removes LoRA on the PRoPE path. However, LoRA-no PRoPE drastically worsens dynamics while only slightly improving camera compared to LoRA, as shown in Table 1(b).

Ablating the design of DynaTokens. We ablate design of DynaTokens, and evaluate alternatives such as adding dynamic tokens directly to the patch tokens (additive), only using one token, changing the token dimensions, and eliminating context noising (using clean historical frames) during training.

Table 3 shows that the current design achieves the best performance overall on VBench2 dynamics tasks, considering both dynamics and camera metrics. Training with a single dynamic token, while slightly better on camera, compromises dynamics metrics greatly. This is likely because the softmax attention weight, which acts as localized gating, is no longer meaningful. No context noising, although slightly better on the simpler Motion Order Understanding tasks, causes a significant drop in Dynamic Spatial Relations, which require later chunks of the video to produce spatially-correct motions despite the discrepancy between the model’s self prediction at inference time with teacher forcing training.

<table><tr><td rowspan=1 colspan=7></td><td rowspan=1 colspan=1>DSR MOU Cam</td></tr><tr><td rowspan=6 colspan=7>DynaTokensSingle tokenReduce token dim (32)Increase token dim (128)No context noisingAdditive</td><td rowspan=1 colspan=1>1.00  0.72  0.88</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.57  0.56  0.89</td></tr><tr><td rowspan=1 colspan=3>token dim</td><td rowspan=1 colspan=1>2)</td><td rowspan=1 colspan=1>0.65  0.72  0.88</td></tr><tr><td rowspan=1 colspan=2>en </td><td rowspan=1 colspan=4>dim (128)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.60  0.72  0.87</td></tr><tr><td rowspan=1 colspan=1>0.67  0.83  0.84</td></tr><tr><td rowspan=1 colspan=1>0.91  0.67  0.85</td></tr></table>

Table 3: Ablations on dynamic token design evaluated on VBench2. The current design yields best holistic performance on dynamics and camera. DSR: Dynamic Spatial Relation. MOU: Motion Order Understanding.

Probing the temporal aspect of DynaTokens. We probe whether the base model is capable of continuing dynamics after the motion is kick-started by only applying DynaTokens to the first 1/8, 1/4 and 1/2 of the full video. As shown on the right, even only using dynamic tokens for the first 1/8 can yield over 64% successful dynamic trajectories, and using it for the first half yields successful dynamics 88% of the time. This suggests that once the motion is “kick-started”, the base model has the capability of continuing it via autoregressive chunk-by-chunk generation.

![](images/1d17950a6be353aa30fd1f162a122a7ad01178cb689a24afcea0656aa0a366b3.jpg)

We also evaluate the application DynaTokens on longer videos, showcasing that the learned dynamics is not limited to the training duration, shown in Fig. 19 in the Appendix.

## 6 Conclusions and Discussions

Dynamic scenes are a key challenge in camera-controlled world models. We present DynaTokens, a lightweight add-on module that can be test-time-trained to enable camera-controlled video models to learn localized dynamics for a given scene. This design is able to “selectively” teach dynamics while enabling good camera control, whereas other designs such as LoRA fail. We provide analysis on why the design of dynamic tokens can effectively disentangle local dynamics with global camera control, and show analyses with attention weights to demonstrate the localized gating effect of our design. DynaTokens, via test-time training, greatly surpasses current state-of-the-art camera-controlled video models on VBench2 dynamic categories and WorldScore, and can extend to physical dynamics and style transfer. Our analysis on the dynamic token design sheds light on the representation of dynamics and camera control for video models. DynaTokens also opens up data generation for dynamic-rich, camera-controlled videos in diverse settings, paving the way for generating dynamic, navigable worlds in the future. For more info, visit our project page https://glab-caltech.github.io/ dynatokens/.

## 7 Ackowledgements

We thank Damiano Marsili, Aadarsh Sahoo, Jiacheng Liu, and Jhan Liufu for discussions. This project was funded by the Packard Fellowship, the Powell Foundation, an Amazon Research Award and a Google Faculty Award. We also thank Google for providing us with Gemini credits.

## References

[1] Ekin Akyürek, Mehul Damani, Adam Zweiger, Linlu Qiu, Han Guo, Jyothish Pari, Yoon Kim, and Jacob Andreas. The surprising effectiveness of test-time training for few-shot learning. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 942–963. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/ akyurek25a.html.

[2] Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B. Lindell, Zan Gojcic, Sanja Fidler, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra: Generative 3d scene reconstruction via video diffusion model self-distillation. In International Conference on Learning Representations (ICLR), 2026.

[3] W. G. C. Bandara and Vishal M. Patel. Attention prompt tuning: Parameter-efficient adaptation of pre-trained models for action recognition. 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pages 1–10, 2024. URL https://api.semanticscholar.org/CorpusID:271115318.

[4] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, Michael Ryoo, Paul Debevec, and Ning Yu. Go-with-the-flow: Motion-controllable video diffusion models using real-time warped noise. In CVPR, 2025. Licensed under Modified Apache 2.0 with special crediting requirement.

[5] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2025.

[6] Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026.

[7] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17702–17711, 2025.

[8] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 27713–27724, October 2025.

[9] Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models, 2026. URL https://arxiv.org/abs/2603.07145.

[10] Google DeepMind. Genie 3: A new frontier for world models. https://deepmind.google/ blog/genie-3-a-new-frontier-for-world-models/, 2025. Accessed: 2026-03-03.

[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=nZeVKeeFYf9.

[12] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, and Sanja Fidler. Vipe: Video pose engine for 3d geometric perception. In NVIDIA Research Whitepapers arXiv:2508.10934, 2025.

[13] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European conference on computer vision, pages 709–727. Springer, 2022.

[14] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon/.

[15] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[16] Hansen Jin Lillemark, Benhao Huang, Fangneng Zhan, Yilun Du, and Thomas Anderson Keller. Flow equivariant world models: Memory for partially observed dynamic environments, 2026.

[17] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

[18] Ziqi Ma, Mengzhan Liufu, and Georgia Gkioxari. Out of sight, out of mind? evaluating state evolution in video world models, 2026. URL https://arxiv.org/abs/2603.13215.

[19] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles?, 2025. URL https://arxiv.org/abs/2501. 09038.

[20] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, volume 2025, pages 28085–28128, 2025.

[21] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos, 2025.

[22] Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, Sanja Fidler, Jiahui Huang, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra 2.0: Explorable generative 3d worlds. arXiv preprint arXiv:2604.13036, 2026.

[23] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world model. arXiv preprint, 2025.

[24] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, Xiao Hu, Xiaohua Hu, Boyuan Jiang, Fangyuan Kong, Hang Li, Jie Li, Qingyu Li, Shen Li, Xiaohan Li, Yan Li, Jiajun Liang, Borui Liao, Yiqiao Liao, Weihong Lin, Quande Liu, Xiaokun Liu, Yilun Liu, Yuliang Liu, Shun Lu, Hangyu Mao, Yunyao Mao, Haodong Ouyang, Wenyu Qin, Wanqi Shi, Xiaoyu Shi, Lianghao Su, Haozhi Sun, Peiqin Sun, Pengfei Wan, Chao Wang, Chenyu Wang, Meng Wang, Qiulin Wang, Runqi Wang, Xintao Wang, Xuebo Wang, Zekun Wang, Min Wei, Tiancheng Wen, Guohao Wu, Xiaoshi Wu, Zhenhua Wu, Da Xie, Yingtong Xiong, Yulong Xu, Sile Yang, Zikang Yang, Weicai Ye, Ziyang Yuan, Shenglong Zhang, Shuaiyu Zhang, Yuanxing Zhang, Yufan Zhang, Wenzheng Zhao, Ruiliang Zhou, Yan Zhou, Guosheng Zhu, and Yongjie Zhu. Kling-omni technical report, 2025. URL https://arxiv.org/abs/2512.16776.

[25] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[26] World Labs Team. Atlas: A world model for spatial intelligence. World Labs Blog, 2026. https://www.worldlabs.ai/blog/atlas.

[27] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras, 2021.

[28] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In European Conference of Computer Vision (ECCV), page 36–54, 2024. ISBN 978-3- 031-72666-8.

[29] Yiming Wang, Qihang Zhang, Shengqu Cai, Tong Wu, Jan Ackermann, Zhengfei Kuang, Yang Zheng, Frano Rajic, Siyu Tang, and Gordon Wetzstein. Bullettime: Decoupled control of timeˇ and camera pose for video generation. arXiv preprint arXiv:2512.05076, 2025.

[30] Zehan Wang, Tengfei Wang, Haiyu Zhang, Xuhui Zuo, Junta Wu, Haoyuan Wang, Wenqiang Sun, Zhenwei Wang, Chenjie Cao, Hengshuang Zhao, et al. Worldcompass: Reinforcement learning for long-horizon world models. arXiv preprint, 2026.

[31] Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Sri Siddarth Chakaravarthy P, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, and Animesh Garg. Mosaicmem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026.

[32] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, et al. Learning to discover at test time. arXiv preprint arXiv:2601.16175, 2026.

[33] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In International Conference on Learning Representations (ICLR), 2024.

[34] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, and Hao Tan. Test-time training done right. In International Conference on Learning Representations (ICLR), 2026.

[35] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

## A Additional Qualitative Results

Fig. 7 & 8 & 9 & 10 show additional qualitative comparisons between DynaTokens and current stateof-the-art camera-controlled models. Fig. 13 shows additional examples of physical dynamics, and Fig. 14 shows that DynaTokens can enable dynamics of multiple objects. Fig. 11 & 12 show additional comparisons between DynaTokens and other test-time training methods. For interactive video visualization, please visit our project webpage https://glab-caltech.github.io/dynatokens/.

## B Evaluation: Standard Errors

<table><tr><td rowspan="2"></td><td colspan="3">VBench2</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>Lyra2</td><td>0.54 (0.071)</td><td>0.17 (0.042)</td><td>0.87 (0.017)</td><td>1.11 (0.226)</td><td>0.70 (0.015)</td></tr><tr><td>Lingbot</td><td>0.68 (0.071)</td><td>0.25 (0.058)</td><td>0.82 (0.017)</td><td>4.37 (0.605)</td><td>0.64 (0.023)</td></tr><tr><td>WorldPlay</td><td>0.56 (0.070)</td><td>0.07 (0.032)</td><td>0.88 (0.009)</td><td>1.95 (0.273)</td><td>0.69 (0.013)</td></tr><tr><td>Hydra</td><td>0.51 (0.066)</td><td>0.18 (0.050)</td><td>0.61 (0.019)</td><td>2.51 (0.520)</td><td>0.56 (0.029)</td></tr><tr><td>LiveWorld</td><td>0.44 (0.061)</td><td>0.35 (0.062)</td><td>0.82 (0.016)</td><td>4.49 (0.817)</td><td>0.70 (0.013)</td></tr><tr><td>DynaTokens</td><td>0.96 (0.030)</td><td>0.69 (0.061)</td><td>0.87 (0.011)</td><td>5.33 (0.531)</td><td>0.69 (0.015)</td></tr></table>

Table 4: Comparison of DynaTokens with camera-controlled video models on VBench2 and World-Score with standard errors across test samples. DynaTokens’ advantage is significant.

<table><tr><td rowspan="2"></td><td colspan="3">VBench2</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>Dynatoken</td><td>1.00 (0.000)</td><td>0.72 (0.082)</td><td>0.88 (0.011)</td><td>4.46 (0.538)</td><td>0.67 (0.060)</td></tr><tr><td>LoRA</td><td>0.61 (0.113)</td><td>0.22 (0.101)</td><td>0.83 (0.008)</td><td>2.79 (0.523)</td><td>0.56 (0.062)</td></tr><tr><td>LoRA-no PRoPE</td><td>0.41 (0.123)</td><td>0.06 (0.056)</td><td>0.84 (0.008)</td><td>3.62 (0.337)</td><td>0.67 (0.056)</td></tr><tr><td>Finetuning</td><td>0.59 (0.123)</td><td>0.67 (0.114)</td><td>0.83 (0.021)</td><td>2.66 (0.358)</td><td>0.54 (0.064)</td></tr><tr><td>TTT Layer</td><td>0.41(0.110)</td><td>0.11 (0.076)</td><td>0.88 (0.007)</td><td>3.79 (0.384)</td><td>0.66 (0.061)</td></tr></table>

Table 5: Comparison of DynaTokens with other test-time training methods, including LoRA, block finetuning, and TTT layer, with standard errors across evaluation examples. DynaTokens’ advantage is significant.

Table 4 and Table 5 report standard errors for each method across the test set. DynaTokens’ advantage is significant both compared to baseline world models and compared to alternative test-time training methods.

## C Additional Comparison with VPT/APT

We perform additional comparison with prefix-tuning methods, Visual Prompt Tuning (VPT) [13] (and its video version APT [3]) on a subset of evaluation scenes. The implementation of VPT/APT in the camera-controlled video generation setting is the same. When trained on the same data, VPT/APT struggles to learn dynamics effectively, stays close to the static settings for many (especially unseen) trajectories, and sometimes generates artifacts such as duplicating objects. Quantitatively, DynaTokens shows a significant advantage in Table 6.

<table><tr><td rowspan="2"></td><td colspan="3">VBench</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>DynaTokens</td><td>1.00</td><td>0.75</td><td>0.86</td><td>5.15</td><td>0.69</td></tr><tr><td>VPT/APT</td><td>0.50</td><td>0.17</td><td>0.84</td><td>2.35</td><td>0.67</td></tr></table>

Table 6: Comparison of DynaTokens and VPT/APT. Naive prefix tuning methods cannot disentangle dynamics from camera control, and struggles to learn dynamics effectively.

The comparison with VPT and APT illustrates an important design aspect of DynaTokens. While VPT and APT were originally designed for recognition (VPT for images, and APT a more efficient version for videos), applying them to camera-controlled video generation means inserting trainable “data tokens” alongside the original data tokens. While DynaTokens also inserts new tokens, the mechanism of token interaction with the base model is very different. Unlike VPT/APT, DynaTokens does not participate in the global self attention alongside the data tokens, where camera control takes place. As shown in Fig. 1(a) in the main paper, the camera conditioning happens via PRoPE in the self-attention. VPT/APT-style methods will interfere with camera conditioning (the added tokens are entangled in the camera and data path), thus failing to disentangle object movement from camera movement. This entanglement is exactly the source of issues with camera-controlled video models and the motivation behind DynaTokens.

<table><tr><td></td><td>IoU</td><td>IoU-d1</td></tr><tr><td>DynaTokens</td><td>0.342</td><td>0.511</td></tr><tr><td>LoRA</td><td>0.006</td><td>0.017</td></tr><tr><td>Text</td><td>0.011</td><td>0.024</td></tr><tr><td>Chance</td><td>0.010</td><td>0.022</td></tr></table>

Table 7: IoU and IoU-d1 (relaxed to count all distnace ≤1 neighbors as correct) of DynaTokens attention map, attention map delta before and after LoRA, the attention map from text keywords corresponding to the moving object (such as the word “dog”), as well as a random attention mask pattern. DynaTokens has a significantly higher overlap with the SAM mask of the moving object, while LoRA and text perform similarly to random chance.

## D Additional Analysis of DynaTokens’ Attention Map Localization

We quantify DynaTokens’ attention localization by measuring the overlap between attention maps with SAM [20] masks via IoU and IoU-d1 (relaxing the IoU by counting all distance ≤1 neighbors as correct). We note that while the attention map illustrates DynaTokens’ localization to the motion, its purpose is not to segment the exact object geometry in video frames. Additionally, DynaTokens work in latent patches which are coarse spatially and temporally (a latent patch is 16x16 pixels across 4 physical frames), and thus we expect coarse overlap rather than accurate, high-IoU boundary alignment. We compare with the attention map delta caused by LoRA, as well as the attention map from the text keyword of the moving object. We additionally show the expected IoU of a random attention mask pattern, denoted by “Chance”. As shown in Table 7, DynaTokens has a significantly higher overlap with the SAM mask of the moving object, while LoRA and text perform similarly to random chance.

We further calculate Ripley’s L statistic which measures whether points in 2D have a clustered distribution pattern (higher means more clustered). DynaTokens scores 5.95, LoRA 4.44, text keyword 2.17 (L(3)-3), which shows a clear lead by DynaTokens.

Fig. 15 & 16 & 17 show additional attention map visualizations with DynaTokens and compares them to LoRA and text-based attention maps. Fig. 18 further shows attention map localization for a challenging scenario where 4 objects are moving (4 ducks moving in different directions). These visualizations further support our localized dynamics hypothesis that motivates our DynaTokens design.

## E Robustness to Training Trajectories

We perform additional ablations on a subset of evaluation scenes to evaluate the robustness of DynaTokens to the quantity and quality of training trajectories. In Table 8, we perform an additional study by training on 3/6/12 trajectories on a subset of 30 evaluation videos. Notably, even when training on as few as 3 trajectories, DynaTokens shows a considerable advantage on MOU (0.50 vs. 0.35 for the best baseline, Live-World), which is a positive result. Other metrics

<table><tr><td rowspan="2"># trajectories</td><td colspan="3">VBench</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>3</td><td>0.67</td><td>0.50</td><td>0.80</td><td>4.02</td><td>0.67</td></tr><tr><td>6</td><td>0.75</td><td>0.67</td><td>0.80</td><td>4.68</td><td>0.68</td></tr><tr><td>12</td><td>1.00</td><td>0.75</td><td>0.86</td><td>4.81</td><td>0.68</td></tr><tr><td>15</td><td>1.00</td><td>0.75</td><td>0.87</td><td>5.15</td><td>0.68</td></tr></table>

Table 8: Studying the effect of the number of training trajectories. Even when training on as few as 3 trajectories, DynaTokens shows a considerable advantage on MOU (0.50 vs. 0.35 for the best baseline, LiveWorld).

(DSR, MOU, MA) beat baselines even at 6 trajectories. This shows that DynaTokens is effective even with a small number of training trajectories.

In Table 9, we perform an additional “swapping” analysis on a subset of 30 evaluation videos: we swap up to 33% of good training trajectories with corrupted trajectories (containing background inconsistencies or sudden motion changes). We observe DynaTokens’ performance remains high, above baselines even with 33% of training trajectories corrupted by artifacts. This is likely due to the DynaTokens formulation retaining the base model prior of coherent video generation.

<table><tr><td rowspan="2"># artifact swap</td><td colspan="3">VBench</td><td colspan="2">WorldScore</td></tr><tr><td>DSR</td><td>MOU</td><td>Cam</td><td>MA</td><td>Cam</td></tr><tr><td>0 (original)</td><td>1.00</td><td>0.75</td><td>0.87</td><td>5.15</td><td>0.68</td></tr><tr><td>1 (7%)</td><td>1.00</td><td>0.75</td><td>0.86</td><td>4.99</td><td>0.66</td></tr><tr><td>5 (33%)</td><td>1.00</td><td>0.67</td><td>0.86</td><td>4.50</td><td>0.66</td></tr></table>

Table 9: Studying the effect of the quality of training trajectories by swapping k good trajectories with k trajectories with artifacts. Even with 33% trajectories swapped to corrupted, DynaTokens’ performance remains high.

## F Benchmark Evaluation Details

For VBench2 evaluation, we use the “Dynamic Spatial Relations” and “Motion Order Understanding” categories. We use VBench’s original VLM evaluation framework with minor modifications to accommodate camera movement, since the original framework is designed for static camera. For example, VBench2 determines object spatial relation by the first and last frame. Since cameracontrolled models might occasionally turn the camera away from the main object (e.g., camera turns left while object moves right and out of frame), we add an “object presence” detection and only apply this check when the main object is present in the last frame. We also remove the questions solely based on the first frame, since all methods we test take the same first frame as input. We use Gemini 2.5 Flash instead of Llava for better spatial relation understanding. To avoid Gemini bias, we additionally use GPT-5 for verification, which yields very similar results: the mean difference is 0.025, which is less than 10% of the observed advantage of DynaTokens over the best baseline (Lingbot) on dynamic spatial relations. For methods that generate highly distorted or unrecognizable subjects, we add “recognizable and not distorted” to the evaluation prompt. We use a state-of-the-art camera estimation method, ViPE [12] to estimate camera, and calculate camera score based on a geometric mean of rotation and translation errors, where rotation error is denoted by geodesic distance, similar to WorldScore [8]. Since the translation is dependent on the unit for each model, we apply scaling to correct for the unit conversion.

The WorldScore benchmark’s dynamic evaluation is largely based on optical flow, and assumes a static camera (thus low background flow). To adapt it to the moving camera setting, we adapt the Motion Accuracy and Camera Control metrics. Two other metrics, Motion Magnitude and Motion Smoothness, are both dominated by camera-induced flow rather than scene dynamics and are thus not suitable for this task. Motion Accuracy uses SEA-RAFT [28] dense flow and SAM2 [21] object masks to score whether the prompted subject moves more than its surroundings; we (i) set the object-region flow to zero when SAM2 fails to localize the subject, so lost or distorted subjects are penalized rather than collapsing into a whole-frame statistic, and (ii) replace the background maximum with the background median, since under camera motion the background flow distribution is heavy-tailed and its maximum is unstable. Camera Control recovers the trajectory via DROID-SLAM [27] and scores the clipped geometric mean of rotation geodesic error and scale-aligned translation $\ell _ { 2 }$ error, with $C ^ { * } = \arg \operatorname* { m i n } _ { C } \| t _ { \mathrm { g t } } - C t _ { \mathrm { p r e d } } \| _ { 2 }$ ; we constrain $C \in [ c _ { \operatorname* { m i n } } , c _ { \operatorname* { m a x } } ]$ to avoid the loophole where under-translating prediction is rescaled to match the ground-truth magnitude.

For evaluation of the baseline camera-controlled models, we apply conversion from action string (which WorldPlay takes as input) to camera matrices for Lingbot, Lyra2, LiveWorld and Hydra. Since Hydra only takes video input, and our setting does not have history, we repeat the initial frame as the input video.

## G Test-Time Training Details

Data Curation. We generate video samples with dynamics to train dynamic tokens. Given an initial frame $I _ { 0 } ,$ text prompt τ , and a target camera trajectory π partitioned into N segments, $\pi _ { 1 : 1 } , \ldots , \pi _ { 1 : N } .$ we construct an imperfect video supervision $( \bar { I } _ { 0 } , \tau , \dot { \pi } ) \stackrel { - } { \mapsto } V$

(i) Dynamics synthesis. A VLM (Gemini 3.1 Pro) first enhances τ by making the dynamics more explicit. Then, an image-to-video model conditioned on $I _ { 0 }$ and the enhanced caption is used to produce a static-camera clip $V _ { \mathrm { d y n } }$ . The same VLM then scores $V _ { \mathrm { d y n } }$ on physical plausibility and camera staticity, refining the caption until passes.

(ii) Camera projection. We sample $N { + 1 }$ uniformly spaced state frames $\{ S _ { 0 } , \ldots , S _ { N } \}$ from $V _ { \mathrm { d y n } }$ For each i, we run HY-WorldPlay on $( S _ { i } , \tau _ { \mathcal { O } } , \pi _ { 1 : i } )$ with the neutral prompt $\tau _ { \mathcal { O } } = ^ { 6 6 } \mathtt { A }$ static scene.”, recovering a camera-aligned view $\hat { S } _ { i }$ at the boundary of $\pi _ { 1 : i }$ . For instance, with $\pi { = } ^ { \infty } \mathrm { f o r w a r d } { - } 6$ , right-7, forward- $- 6 \ "$ and $N { = } 3$ , we run HY-WorldPlay three times: on $S _ { 1 }$ with prefix $\pi _ { 1 : 1 } = ^ { \ast } \mathbf { f }$ orwar ${ \tt d } - 6 "$ , on $S _ { 2 }$ with $\pi _ { 1 : 2 } { = } ^ { \ast } \mathbf { f }$ orward-6, $\tt r i g h t - 7 ^ { \cdot }$ , and on $S _ { 3 }$ with $\pi _ { 1 : 3 } { = } \pi$ This yields $\hat { S } _ { 1 } , \hat { S } _ { 2 } , \hat { S } _ { 3 }$ that each capture the scene state $S _ { i }$ as observed from the camera position reached after executing $\pi _ { 1 : i }$

(iii) Interpolation and stitching. Consecutive keyframes $( \hat { S } _ { i - 1 } , \hat { S } _ { i } )$ are bridged by a dual-endpoint image-to-video interpolator, Kling o3 [24], guided by a transition caption that the VLM writes from the two endpoints and the camera-action label. The per-segment clips are concatenated into the final video V, which is passed to a VLM for review.

It is important to note that, although our curated videos contain both scene and camera motion, stitching can introduce artifacts in background appearance and motion continuity. For example, the scene layout or object appearance may change across frames, and temporal discontinuities may arise. Qualitative examples can be found in Fig. 6. However, since the base model remains frozen and dynamic tokens contribute only through localized, additive cross-attention, this imperfect supervision can teach scene dynamics without corrupting the pretrained prior. Model inference calls take 16 minutes on 8 A100 GPUs (with shared prefix skipping), and API call for Gemini and Kling cost \$1.52 for all training trajectories. Our DynaTokens design can train on few trajectories, enabling camera-controlled models to produce correct dynamics for new, unseen camera trajectories.

We show examples of curated data below. They contain background inconsistencies since each keyframe’s projection might not “hallucinate” unseen space in the same way.

![](images/20fb735f58307f76fe78f611837704718c0df52d28832b8af97c429527c0a5bd.jpg)  
Figure 6: Examples of curated training examples. Curating one sample takes 20 GPU minutes (A100) and cost \$1.52 in API calls (Kling and Gemini). The curated data contains correct dynamics but may include background inconsistencies. Our test-time training approach resolves such inconsistencies by leveraging model priors.

Training. We train $K = 1 6$ dynamic tokens of dimension $m = 6 4$ in our experiments, for videos ranging from 61 to 93 frames. We use the Muon [14] optimizer with learning rate 3e-3 with weight decay and L2 regularization. The training memory is around 22GB on 4 GPUs, for which we use 40GB or 80GB A100s. For context noising, we add high noise (step [500, 985)) to the context chunks. Training takes 80 minutes for 15 trajectories (total 900 frames). To put this in context, just doing inference using Lyra2, a baseline video model, on our 6 evaluation trajectories, takes 66 A100 minutes. This suggests the test-time adaptation, which unlocks inference on diverse unseen trajectories, is not a huge overhead compared to the per-trajectory inference cost.

Training limitations. DynaTokens is a lightweight module added to pretrained video models. Dynamic tokens enable dynamics by effectively disentangling it from camera control. However, if the base model is poor at certain dynamics, learning with dynamic tokens should not expect to fully fix these issues, such as complex physics.

## H Additional Analysis on LoRA’s Failure

We provide more detailed derivation on LoRA’s effect on PRoPE:

$$
q _ { i } = ( W _ { Q } + \Delta W _ { Q } ) h _ { i } , \qquad k _ { j } = ( W _ { K } + \Delta W _ { K } ) h _ { j }\tag{3}
$$

$$
\begin{array} { c c } { { \tilde { q } _ { i } = D ( c _ { i } ) q _ { i } , } } & { { \tilde { k } _ { j } = D ( c _ { j } ) k _ { j } , } } \\ { { } } & { { } } \\ { { s _ { i j } = \left. D ( c _ { i } ) W _ { Q } h _ { i } , D ( c _ { j } ) W _ { K } h _ { j } \right. + \left. D ( c _ { i } ) \Delta W _ { Q } h _ { i } , D ( c _ { j } ) W _ { K } h _ { j } \right. } } \\ { { } } & { { } } \\ { { + \left. D ( c _ { i } ) W _ { Q } h _ { i } , D ( c _ { j } ) \Delta W _ { K } h _ { j } \right. + \left. D ( c _ { i } ) \Delta W _ { Q } h _ { i } , D ( c _ { j } ) \Delta W _ { K } h _ { j } \right. } } \end{array}\tag{4}
$$

Only the first term corresponds to the original PRoPE value.

We also provide more analysis on why LoRA-No PRoPE fails. While it does not significantly reduce the number of free parameters compared to normal LoRA, this might be causing great discrepancy between the PRoPE vs. non-PRoPE stream (which was designed to be symmetric in the original architecture), thus hindering learning. This shows that LoRA, with its global entanglement across spatial patches, struggles to only learn dynamics without affecting camera control. Dynamic tokens, on the other hand, minimizes the effect on PRoPE by being local - unaffected patches’ keys and values stay mostly stable.

## I Societal Impact

We improve generative video models. We acknowledge that generative models can be used in ways that raise ethical concerns, including the creation of misleading synthetic media, amplification of societal biases present in training data, and potential misuse for surveillance or harmful automation. Our work is intended solely for scientific research purposes and focuses on advancing the understanding and capability of generative modeling systems. We do not release systems or artifacts designed for deceptive or malicious use, and we encourage responsible deployment practices, including transparency about generated content, careful dataset curation, and evaluation for fairness and safety.

![](images/7aceb747c1e118d97dcfc0106a2ab39c6c607b0f739a987b7eea19fbefd48b85.jpg)  
Figure 7: Additional qualitative examples of comparison with state-of-the-art camera-controlled video models.

![](images/f0337ab4feb30a75b08ccd8565ba541e2f587f9b1b0eabe20742122460029b32.jpg)  
Figure 8: Additional qualitative examples of comparison with state-of-the-art camera-controlled video models.

A fox is on the right of a shoe, then the fox runs to the behind of the shoe.  
![](images/95bab09269d2491a61f2d1451607ef6f7dc800a1ab80e81a6d12e24467d857b3.jpg)  
Figure 9: Additional qualitative examples of comparison with state-of-the-art camera-controlled video models.

The bus moves forward along the snowy road, carefully navigating through the winter conditions. It follows a steady path, making stops as it approaches the designated bus stop area.

![](images/32d52b8ba56b808ecea040e9bf5846c3bc215d8613728fcbb924400157e96033.jpg)  
Figure 10: Additional qualitative examples of comparison with state-of-the-art camera-controlled video models.

The dandelion's seeds gently detach and float away on the breeze, spiraling gracefully into the air. The stem sways slightly back and forth as the wind moves through the meadow.

![](images/24500c699e475378c05d9b464d91edf90d9038a6d89562c13aca7c9c18d75f23.jpg)  
Figure 11: Additional qualitative examples of comparison across test-time methods.

A dog is on the left of a table, then the dog runs to the front of the table  
![](images/73c21adc1b4ce78bbfb92e9b661e14040d707925b99079bc2acf2efbf848faa3.jpg)  
Figure 12: Additional qualitative examples of comparison across test-time methods.

![](images/69ada599607ce9416b9d375be17c897eb2593ee03bd056ae0be56f2ba1c0dcab.jpg)  
Figure 13: Additional qualitative examples for physical dynamics.

Four white ducks start grouped together. They separate and walk diagonally toward four distinct corners.  
![](images/f8ce9801effc43cb04a3634db58393935e4feb95ba8674bc627af01ac417a699.jpg)

The red ball rolls towards the left down the ramp, the blue ball rolls towards the right down the ramp.  
![](images/a247e32049c640693be7d3041ad40edb6b69ff3f66723d395a5e4002ed5cdbf3.jpg)

Figure 14: DynaTokens can enable dynamics of multiple objects.  
![](images/9414487725cad46fe23e189bfbceebbbc23fe7af4a4e230b0dac9f8009027e81.jpg)  
Figure 15: Attention map visualization of DynaTokens, LoRA, and the text keyword of the moving object (dog).

![](images/5fa1886cdd6869821a0a9160674acfff35578fab9b6af9438e33e4fe1236772b.jpg)  
Figure 16: Attention map visualization of DynaTokens, LoRA, and the text keyword of the moving object (kangaroo).

![](images/76a8d2f8d745d2b74db44107b2bf9302e85ba2d5ca75ef36982f7f836bb13739.jpg)  
Figure 17: Attention map visualization of DynaTokens, LoRA, and the text keyword of the moving object (person).

![](images/747261063e7c80f47220bdb871f5fc4cb75fea49010461e0d4e255bc6699eaf9.jpg)  
Figure 18: Attention map visualization of DynaTokens for a complex scene with four moving objects (four ducks moving in different directions).

![](images/88a0c72f745e3022e5925a447df5cfcbf720325d22870e65b0b5ccd23334f18c.jpg)  
Figure 19: Evaluation on longer videos.