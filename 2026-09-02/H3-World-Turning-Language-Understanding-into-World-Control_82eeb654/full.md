# H3-World: Turning Language Understanding into World Control

Danze Chen<sup>1,2,♣</sup> Zeqing Wang<sup>1,2,♣</sup> Ziyue Lin<sup>3</sup> Xingyi Yang<sup>3,∗</sup> Yeying Jin<sup>1,2,∗,♢</sup>

<sup>1</sup> Tencent <sup>2</sup> National University of Singapore <sup>3</sup> The Hong Kong Polytechnic University

We present H3-WORLD, an efficient framework that turns the 33B MiniMax-H3 video generator into an interactive world model. Our key finding is that, as large video generators become more capable, language is emerging as a natural interface for control. MiniMax-H3, for example, already supports zero-shot control of character behavior and camera motion through natural-language instructions. Building on this, H3-WORLD turns this coarse language interface into precise, temporally grounded world control, without introducing dedicated action modules. Specifically, we represent each action as a structured combination of character and camera instructions, and align them with the corresponding temporal video latents. To make the control temporally precise, we further introduce temporal attention routing, which restricts each instruction to its intended time interval and reduces control leakage across actions. Importantly, H3-WORLD directly reuses the semantic representations learned during large-scale video pretraining and requires only lightweight adaptation. With only 8,000 gameplay samples, 10,000 LoRA optimization steps, and 0.199% trainable parameters, H3- WORLD achieves effective character and camera control while preserving strong generation quality. It also generalizes to unseen scenarios. These results show that the control capabilities emerging in large video generators can be efficiently transformed into interactive world control.

Project: https://danzer1xxxxchan.github.io/H3-World Code: https://github.com/Danzer1xxxxChan/H3-World Model: https://huggingface.co/DANNY621/H3-World

![](images/212c3ecd9961766d14a293ae8f3cb738d72b8a64cec2157d9ffbb0220e482f30.jpg)  
Figure 1: H3-WORLD enables character and camera control across diverse visual environments.

## 1 Introduction

Language is becoming more than a way to describe generated worlds. As video generators grow more capable, language is also beginning to shape how those worlds evolve—how agents move, how cameras behave, and how scenes change over time [1–3]. In this sense, language is emerging as a high-level abstraction layer for visual dynamics. This shift is especially relevant to world models, which learn how environments evolve under actions and connect intelligent agents with their surroundings [4–7]. Meanwhile, advances in diffusion transformers and large-scale video generation have made pretrained video models a natural foundation for visually rich world simulation [1, 3, 8–14]. Such models already support increasingly rich forms of interactive content, game simulation, and learned real-world simulation [2, 15–19].

Yet generation is not control. A pretrained video model does not, by itself, expose the precise action interface required for interactive world modeling. Existing systems therefore introduce new control pathways through discrete action embeddings, conditional modules, camera geometry, or direct backbone adaptation [20–28]. These pathways are typically learned from temporally aligned action-video trajectories. They require additional supervision and adaptation on top of the pretrained generator. They can also increase computation and storage, and extensive adaptation may disturb capabilities acquired during pretraining. More fundamentally, this paradigm treats control as something that must be learned largely on top of the pretrained model.

But large video generators may already contain part of this bridge. Modern video models learn rich representations of objects, agents, motion, and interaction from large-scale video data [1, 3, 9, 11, 14]. MiniMax-H3, for example, can already follow coarse textual instructions for character and camera motion [1]. As shown in Figure 2, such instructions produce roughly correct responses with realistic visual dynamics, even without action-conditioned training. This observation is central to our work: language already acts as a coarse control interface for strong text-to-video models.

“The man quickly walks forward while the camera pans left rapidly."  
![](images/039d6af9a27f3d215fad5a4f87e25219f4ed97d8aecdbd6df2c98f42a0d01bcb.jpg)  
Figure 2. Vanilla MiniMax-H3 already exhibits coarse zero-shot control through textual motion instructions.

Based on this observation, we propose H3-WORLD, an efficient framework that turns the 33B-parameter MiniMax-H3 into an interactive world model. Our idea is simple. Instead of learning a new control representation from scratch, we express control in the language space the model already understands. Accordingly, H3-WORLD converts character and camera actions into compositional textual instructions and injects them through the native text pathway of MiniMax-H3. To make these controls precise over time, we align each instruction with its corresponding video latent interval. We further introduce temporal attention routing to preserve this alignment and prevent control leakage across time. As a result, H3-WORLD directly reuses the pretrained language and generative capabilities of MiniMax-H3 without introducing a dedicated action-specific control module. This design requires only lightweight LoRA adaptation [29, 30].

Surprisingly, this lightweight adaptation is sufficient to produce effective interactive control. With only 8,000 gameplay samples and 0.199% trainable parameters, H3-WORLD learns precise character and camera control while preserving the generative capabilities of the pretrained model. More importantly, the resulting control transfers beyond the training distribution, including unseen action compositions and distinct initial observations. Together, these results suggest that much of the structure needed for interactive control can already exist inside a sufficiently capable video generator. By grounding low-level actions into its native language interface, H3-WORLD provides a direct path from pretrained video generation to interactive world modeling.

In summary, our contributions are as follows.

• We show that large video generators already support coarse language-based control, providing a strong basis for interactive world modeling.

• We introduce H3-WORLD, which turns the language understanding of large video models into world control. It represents character and camera actions as textual instructions and uses temporal attention routing for precise control, without dedicated action-specific modules.

• With only 8,000 gameplay samples and 0.199% trainable parameters, H3-WORLD achieves effective control while preserving generation quality and generalizing to unseen actions and visual scenarios.

## 2 Related Work

## 2.1 Action-Conditioned Video World Models

Interactive world models generate visual futures in response to user actions. Earlier neural and diffusion game simulators condition visual dynamics on recorded actions [20–22], while Genie learns a latent action interface from video [2]. Recent systems extend stronger video generators toward open worlds, device control, causal rollout, persistent history, and real-time inference [16–19, 23–28, 31]. Related models study reactive characters, cross-game interaction, and explicit state-aware behavior [32–35]. These methods typically establish control through learned embeddings, feature modulation, camera geometry, or dedicated modules. Such interfaces require additional control learning and may limit robustness and generalization [36, 37]. In contrast, H3-WORLD injects fine-grained controls directly into the textual prompt, allowing them to be interpreted through the language understanding already acquired during video pretraining. This directly reuses the pretrained semantic prior, requires adapting only 0.199% of the parameters [1, 29, 30], and preserves strong generalization to unseen scenarios.

## 2.2 Language Actions and Temporal Grounding

Recent work uses language to express richer interventions and world-model actions [15, 38–42]. Incantation is most closely related in representing actions with per-latent-frame natural language and local text crossattention [41]. Beyond world models, controllable video generation uses trajectories, camera poses, and compositional conditions to direct object and viewpoint motion [43–47]. These works establish semantic conditioning as a useful control interface. H3-WORLD considers a different architectural setting in which semantic conditions and video latents coexist in MiniMax-H3’s packed single-stream self-attention. The model jointly denoises the full future horizon without a separate text cross-attention layer, so our focus is the temporal grounding of language actions within this bidirectional sequence.

## 3 Method

## 3.1 Overview

Given an initial observation $I _ { 0 } ,$ a static semantic condition s, and a scheduled action sequence $\mathbf { a } _ { 1 : K } ,$ H3- WORLD generates the corresponding future video latents $\mathbf { V } _ { 1 : K }$ . We build on MiniMax-H3 [1], a pretrained bidirectional audio-video foundation model that jointly denoises the complete future horizon. Its pretrained parameters θ remain frozen, and ϕ denotes the lightweight adaptation parameters learned for action control. The conditional generation process is written as

$$
\widehat { \mathbf { V } } _ { 1 : K } \sim p _ { \theta , \phi } \left( \mathbf { V } _ { 1 : K } \ | \ I _ { 0 } , s , \mathbf { a } _ { 1 : K } \right) .\tag{1}
$$

(a) Build the packed H3-World sequence  
![](images/a1ddcc238edb3dc6154b9175c16d221fc84fce29f3a67fb18452ff7bd7c0d186.jpg)

![](images/f702b8239d9ecb13115de6b08110f5ac3128f6a69af1360c93cd2a5d2ffdb224.jpg)  
Figure 3. Overview of H3-WORLD. (a) Latent-aligned action prompts are independently encoded and packed with the static semantic condition, initial observation, and video latents. (b) MiniMax-H3 processes the packed sequence through single-stream self-attention. (c) LoRA adapts the attention projections under single-egress routing, which connects each action span directly to its matched video latent and retains bidirectional attention among video latents.

MiniMax-H3 processes text, image, audio, and video tokens in a shared sequence. We retain its native audio stream and focus the following notation on visual action control.

As shown in Fig. 3, H3-WORLD contains three components. The semantic action interface converts character and camera controls into compositional textual instructions. Latent-aligned temporal binding associates each instruction with a video latent interval. Single-egress routing maintains this association throughout the H3 backbone, while LoRA learns the corresponding action-conditioned visual dynamics.

## 3.2 Semantic Action Interface

External controls are represented as discrete control states. Each state contains eight recorded character and camera keys and one binary camera-speed flag. During training, the speed flag is derived from the estimated camera yaw rate. At inference, it is specified directly by the user. Each native H3 video latent represents a short interval of RGB frames. We aggregate the recorded key states within that interval into one latent-level state, marking a key as active when it occurs in any frame of the interval. Opposing keys are cancelled before prompt construction.

These control states provide compact and temporally precise commands, while their symbols carry limited information about the corresponding visual transitions. We expose the compositional structure of the control space by separating character control from camera control. For the k-th video latent interval, the

![](images/ec04261f9a711e822799a5c9d4d8d229eafb308333ab57b0cc3b89faaab0dc46.jpg)  
Figure 4. Training coverage of the action space. Of 135 valid character–camera pairs, 83 occur in 291,264 prompts and 52 remain unseen. The top 20 and 40 pairs account for 71.4% and 95.4% of prompts.

action is written as

$$
\begin{array} { r } { \mathbf { a } _ { k } = \left( u _ { k } , c _ { k } \right) , \qquad u _ { k } \in \mathcal { U } , \qquad c _ { k } \in \mathcal { C } , } \end{array}\tag{2}
$$

where U contains the character-control commands and C contains the camera-control commands. The resulting action space contains nine character clauses and sixteen camera clauses.

We map each control pair to a short textual instruction

$$
p _ { k } = { \mathcal { T } } _ { \mathrm { c h a r } } ( u _ { k } ) \parallel { \mathcal { T } } _ { \mathrm { c a m } } ( c _ { k } ) ,\tag{3}
$$

where ∥ denotes clause concatenation. For example, a backward-left character command combined with a slow right-pan camera command produces the prompt “the man walks backward and strafes left, camera pans right slowly.” The character and camera clauses follow a shared grammatical structure across the dataset. This representation places external controls in the native text-conditioning space of H3 and preserves the factorization of the original action space.

The action space is compact, but the training data cover only a sparse and highly imbalanced subset of the valid combinations. Figure 4 summarizes the joint distribution of character and camera clauses in the training set. The Cartesian product contains $9 \times \dot { 1 } 6 = 1$ 144 combinations, of which 135 are structurally valid under our control specification. The training data covers 83 valid combinations and leaves 52 valid combinations unseen. The observed support therefore satisfies

$$
{ \mathcal { A } } _ { \mathrm { t r a i n } } \subsetneq { \mathcal { A } } _ { \mathrm { v a l i d } } \subsetneq { \mathcal { U } } \times { \mathcal { C } } .\tag{4}
$$

The distribution is concentrated among a small number of frequent pairs. The 20 most frequent combinations account for 71.4% of all action prompts, and the 40 most frequent combinations account for 95.4%. This distribution provides a natural setting for evaluating compositional control. A joint command may be absent from the training set even when its character and camera clauses occur in other combinations.

The semantic action interface specifies the requested visual transition. Time-varying interaction also requires a correspondence between each instruction and a particular portion of the generated video. We establish this correspondence in the H3 token sequence.

## 3.3 Latent-Aligned Temporal Binding

A video-level prompt provides a shared semantic condition for an entire clip and offers limited temporal resolution when the requested action changes within the generation horizon. We assign an independent action prompt to every video latent interval. Each prompt $p _ { k }$ is encoded by the shared H3 encoder $\mathcal { E }$ and processed by a shared two-layer token refiner R

$$
\mathbf { A } _ { k } = \mathcal { R } \left( \mathcal { E } ( p _ { k } ) \right) .\tag{5}
$$

The token refiner uses block-diagonal attention over the scheduled prompts. Tokens within the same action span communicate bidirectionally, and different action spans are processed as separate sequences. This layout uses a common representation space for all actions and preserves the temporal identity of each instruction before it enters the video backbone.

The initial observation follows two complementary encoding paths. The multimodal H3 encoder jointly processes the static semantic condition s and the initial observation $I _ { 0 }$ to produce static semantic tokens S. The visual VAE maps $I _ { 0 }$ to a first-frame condition $\mathbf { C } _ { 0 }$ that preserves fine-grained appearance. The target video is encoded into latent intervals $\mathbf { V } _ { 1 : K }$ . During training, these intervals contain the noised target latents used by the native H3 denoising objective.

We pack the visual conditions and generation targets into a single sequence

$$
{ \bf { X } } = [ { \bf { S } } ; { \bf { A } } _ { 1 } ; \ldots ; { \bf { A } } _ { K } ; { \bf { C } } _ { 0 } ; { \bf { V } } _ { 1 } ; \ldots ; { \bf { V } } _ { K } ; { \bf { P } } ] ,\tag{6}
$$

where P denotes masked padding tokens. Each action span $\mathbf { A } _ { k }$ is paired with the k-th target video latent under the native temporal partition of the H3 VAE. A video latent interval may represent several RGB frames after VAE decoding.

We assign mirrored temporal positions to the action spans. Let $\tau ( \cdot )$ denote the temporal coordinate used by the H3 positional encoding. The position of each action span follows the temporal order of its matched video latent and remains in the text-side positional range. Every matched pair therefore has the same relative temporal offset

$$
\tau ( \mathbf { A } _ { k } ) = \tau ( { \mathbf { V } } _ { k } ) - \Delta , \qquad \Delta > 0 .\tag{7}
$$

This construction preserves the text-before-video ordering of pretrained H3 and provides a consistent temporal alignment cue for every action and latent pair.

Temporal alignment alone does not restrict information flow in bidirectional self-attention. An action span could still communicate directly with unmatched video latents. We therefore introduce a routing structure that constrains the direct path from each action span to the video stream.

## 3.4 Single-Egress Routing and Low-Rank Adaptation

We construct a deterministic single-egress routing mask for each packed sequence. When an action span $\mathbf { A } _ { k }$ serves as an attention key, it can be read by tokens in the same action span and by its matched video latent $\mathbf { V } _ { k } .$ Static tokens, first-frame condition tokens, native audio tokens, other action spans, and unmatched video latents cannot read $\mathbf { A } _ { k } .$ When $\mathbf { A } _ { k }$ serves as an attention query, it retains access to the static context, first-frame condition, native audio context, its own tokens, and the matched video latent. Access to other action spans and unmatched video latents is masked.

All video latent spans retain the original bidirectional attention pattern of H3. The visual effect associated with $\mathbf { A } _ { k }$ first enters the video stream through $\mathbf { V } _ { k }$ and can subsequently propagate through video-to-video attention. This structure preserves full-horizon information exchange for motion continuity and scene consistency. It also gives every scheduled action a unique direct entry point into the visual stream. We refer to this property as single-egress action routing.

LoRA [29] learns the action-conditioned transformations along the permitted routes. We apply low-rank updates to the QKV and output projections in the H3 self-attention blocks and to the two-layer token refiner. The backbone, H3 encoder, visual VAE, and all remaining components remain frozen. The routing mask and span partition introduce no learnable parameters, and training retains the native H3 denoising objective.

## 4 Experiments

## 4.1 Experimental Setup

Data. We construct the training and evaluation sets from ABot-World-Explorer-500h [28]. The training set contains 7,872 gameplay clips, and a separate set of 128 clips is held out for evaluation. Each clip contains 124 frames at 24 fps with a resolution of $\overline { { 8 3 2 } } \times 4 8 0$ . Following the temporal partition described in Section 3.3, every clip provides 37 latent-aligned action prompts.

Training and inference. We train the rank-32 LoRA parameters for 10,000 optimization steps with a learning rate of $1 \times 1 0 ^ { - 4 }$ . At inference, we generate 124 frames with 50 denoising steps. The pretrained H3 comparisons use the same generation settings with the LoRA updates disabled.

Evaluation protocol. We evaluate action control with two paired protocols. For controlled interventions, the initial observation, generation seed, and sampling configuration are fixed while only the scheduled action changes. For held-out clips, we condition H3-WORLD on the initial observation and recorded action sequence, then compare the generated and ground-truth videos by their motion patterns.

Qualitative figures use uniformly sampled frames. The green border marks the initial observation, and the keyboard overlay shows the scheduled control state. Pressed keys are highlighted in orange. When a quantitative diagnostic is required, we estimate dense optical flow using the Farneback method [48] and accumulate the mean horizontal flow over each video. Positive and negative values indicate leftward and rightward camera motion, respectively.

## 4.2 Pretrained Action Prior and Adaptation

We examine whether adaptation enables temporally specified control beyond the coarse motion response already available in pretrained H3. Figure 5 uses a controlled camera schedule that pans left sharply for the first 15 temporal latents and right sharply for the remaining 22. The switch coincides with a latent block boundary. This schedule requires the model to follow both directions and assign each one to its requested interval, while the reversal within one clip controls for scene drift. We compare three settings: (1) frozen H3 with one global motion instruction appended to the scene prompt; (2) the complete per-latent action interface with every LoRA update set to zero; and (3) the trained H3-WORLD checkpoint. We report cumulative horizontal flow separately for the frames before and after the switch.

![](images/0192e0532dd63d3e3df09cd1f44c090c6204e379ca59a390ba83171afeabdb68.jpg)  
Figure 5. Action response before and after adaptation. The scheduled camera pan switches from left to right after latent 15. Check marks and crosses summarize action control and temporal control for each condition.

The three conditions exhibit distinct responses. Global prompting produces cumulative horizontal flow of 0.0 before the switch and −17.3 afterward. Frozen H3 therefore shows a small rightward response in the second half, while the scheduled leftward motion is absent. The zero-LoRA per-latent interface remains nearly static, with flows of −0.1 and 0.0 and mean absolute horizontal flow of 0.003. In contrast, H3-WORLD produces +52.7 before the switch and −106.0 afterward, following both scheduled directions. Reversing the instruction order gives the same pattern: global prompting yields −11.9 and +24.1, the zero-LoRA interface remains unresponsive, and H3-WORLD yields −58.7 and +121.0.

The constant-action control clarifies the role of adaptation. With one camera direction throughout the clip, global prompting and H3-WORLD yield nearly identical directional separation, 301.8 and 300.5. Frozen H3 can therefore respond to a coarse motion instruction. For a changing schedule, however, its global text representation is shared across the full horizon and does not bind each direction to a latent interval. The zero-LoRA condition shows that supplying span-specific instructions alone is insufficient. LoRA adaptation enables the backbone to use these temporal bindings and follow both parts of the schedule.

## 4.3 Action Controllability

Direct action conditioning. We first compare the text-based action interface of H3-WORLD with two direct action-conditioning variants. Each variant encodes the latent-level keyboard state as a learned vector and applies it to the corresponding video-token features. The additive-bias variant follows the actionconditioning mechanism used in ReactiveGWM [32], which projects the action state and adds it to the video representation. The FiLM variant instead applies feature-wise scale and shift after AdaLN [49]. Figure 6 compares the three interfaces with the ground-truth video from the same held-out clip and recorded action sequence. The additive and FiLM variants produce weak or inconsistent changes as the recorded controls vary. In contrast, H3-WORLD produces coordinated character and camera changes throughout the sequence. This diagnostic supports grounding actions in the pretrained text pathway for action control.

![](images/236a664c145caa34b302c7658ee5e266e01d1f7c53af369fe706c1332bcf2a3f.jpg)  
Figure 6. Action-conditioning interfaces on a held-out clip. GT provides the recorded reference motion. Direct additivebias and FiLM conditioning yield limited or inconsistent responses to the recorded controls, while text-based H3-WORLD produces coordinated character and camera changes.

Following recorded controls. We next evaluate the learned text-based interface on held-out gameplay clips. Figure 7 pairs each clip with a H3-WORLD generation conditioned on its first frame and recorded action sequence. Each clip admits multiple plausible futures, so we assess whether the generation exhibits the recorded character and camera motion while retaining the initial scene content. Across the five clips, H3-WORLD exhibits the main character displacement and viewpoint changes indicated by the recorded controls while preserving the scene layout and subject appearance. This indicates that the learned interface transfers recorded action semantics to held-out gameplay observations.

![](images/383b0429305ee23c9daa8294672a6ff1deee1edbd396dde77cd538306f46ab6b.jpg)  
Figure 7. Following recorded controls on held-out clips. Ground-truth videos appear above the corresponding H3- WORLD generations produced from the same initial observation and action sequence.

Controlled action intervention. We then fix the initial observation and generation seed and change only the action instruction. Figure 8 includes stationary and forward motion, left and right strafing, and slow and fast camera pans in both directions. Opposite strafe and pan commands produce distinct lateral evolution. The fast camera commands also produce visibly larger changes than their slow counterparts. The paired construction shows that the generated differences arise from the requested action.

Character Behavior  
Camera Control  
![](images/0c4ff85ce2a072a57754877e68540bab685f9e56cc3ebf0553275504de3ad651.jpg)  
Figure 8. Controlled action comparison. The initial observation, seed, and sampling configuration are fixed across rows. Changing the action produces distinct character and camera motion, including stronger fast pans.

## 4.4 Generalization

![](images/9d6b120cd635c9c14c0fc24734f1c998f9c9203dc941c303461765ac5b201167.jpg)  
Figure 9. Compositional action generalization. Each group contrasts a seen action with an unseen composition of observed character and camera clauses on held-out gameplay (top) and out-of-distribution (bottom) observations.

Compositional action generalization. Of the 135 structurally valid action combinations, 83 occur in training. We evaluate an unseen pair whose character and camera clauses each occur in other training pairs, but never together. This setting tests whether the semantic interface composes familiar control primitives beyond the observed joint support. Figure 9 contrasts this unseen forward-motion and camera pan–tilt command with a seen reference on held-out gameplay and out-of-distribution observations. H3-WORLD follows both the character and camera components of the unseen command in both settings while preserving the scene layout and subject appearance. This result shows that compositional action control extends beyond the joint combinations observed during training.

![](images/bb79c8129f2169521141cac6cdee5dbf12c336043b406baa682ddc703c81afb9.jpg)  
Figure 10. Visual generalization to diverse initial observations. Each scene is evaluated with one character-control command and one camera-control command using the same learned action interface.

Visual generalization. We further evaluate the learned action interface on six initial observations that differ substantially from the gameplay training set. The scenes cover third-person and first-person viewpoints, indoor and outdoor environments, fantasy and science-fiction content, and diverse rendering styles.

As shown in Figure 10, the same interface produces the requested character or camera response across these visual domains. The generations preserve the scene layout, subject identity, and visual style of each initial observation. These results support the use of lightweight adaptation for retaining the broad generative capabilities of pretrained H3 while learning interactive control from a small gameplay dataset.

## 5 Conclusion

We present H3-WORLD, an efficient adaptation of MiniMax-H3 for interactive world modeling. H3-WORLD expresses character and camera commands as compositional text instructions, aligns them with video latent intervals, preserves the alignment through single-egress routing, and learns the associated visual dynamics through low-rank adaptation. The results show that this design supports recorded and intervened control, generalizes to unseen action pairs and visually distinct initial observations, and reuses the control capabilities already present in a pretrained video generator. These findings show that temporally grounded language conditioning can provide an effective route from pretrained video generation to low-level world control.

However, there are still few limitations in this work. Firstly, our current study focuses on short-horizon generation, with compositional generalization and visual transfer evaluated mainly through representative examples; more systematic evaluation across action combinations, scenes, and random seeds is needed to quantify control reliability. Secondly, the current model generates fixed-length segments and does not yet provide persistent world state, real-time interaction, planning, or policy learning. Extending the interface in these directions will be important for interactive world models that operate over longer horizons.

## Acknowledgements

We would like to express our sincere gratitude to Ruidong Wang and Murphy Zhao for their invaluable support throughout this project.

## References

[1] MiniMax. MiniMax H3. https://github.com/MiniMax-AI/MiniMax-H3, 2026. Open-weight omni-modal audio-video generation model. Accessed: 2026-08-28.

[2] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first international conference on machine learning, 2024.

[3] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[4] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3):440, 2018.

[5] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pages 2555–2565. PMLR, 2019.

[6] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020.

[7] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021.

[8] William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182. IEEE, 2023.

[9] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

[10] Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.

[11] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.

[12] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7310–7320. IEEE, 2024.

[13] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

[14] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[15] Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023.

[16] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive openworld game video generation. In International Conference on Learning Representations, volume 2025, pages 37546–37593, 2025.

[17] Ruili Feng, Han Zhang, Zhilei Shu, Zhantao Yang, Longxiang Tang, Zhicai Wang, Andy Zheng, Jie Xiao, Zhiheng Liu, Ruihang Chu, et al. The matrix: Infinite-horizon world generation with real-time moving control. Advances in Neural Information Processing Systems, 38:87318–87344, 2026.

[18] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.

[19] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

[20] Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidler. Learning to simulate dynamic environments with gamegan. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1228–1237. IEEE, 2020.

[21] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In International Conference on Learning Representations, volume 2025, pages 73754–73776, 2025.

[22] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.

[23] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactorly: Creating new games with generative interactive videos. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11590–11599. IEEE, 2025.

[24] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[25] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, et al. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026.

[26] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

[27] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[28] Fan Jiang, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu, Chiyu Wang, Yunpeng Zhang, Wenlin Liu, Yun Wang, Xue Zheng, Rui Sun, et al. Abot-world-0: Infinite interactive world rollout on a single desktop gpu. arXiv preprint arXiv:2607.19191, 2026.

[29] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[30] Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4207–4216. IEEE, 2023.

[31] Etched Decart, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. URL: https://oasis-model. github. io, 2(3):6, 2024.

[32] Zeqing Wang, Danze Chen, Zhaohu Xing, Zizhao Tong, Yinhan Zhang, Xingyi Yang, and Yeying Jin. Reactivegwm: Steering npc in reactive game world models. arXiv preprint arXiv:2605.15256, 2026.

[33] Zizhao Tong, Yeying Jin, Hongfeng Lai, Zeqing Wang, Zhaohu Xing, Kexu Cheng, Haoran Xu, Zhao Pu, Shangwen Zhu, Ruili Feng, et al. Scope: Simulating cross-game operations in playable environments for fps world models. arXiv preprint arXiv:2605.23345, 2026.

[34] Zijun Lin, Zeqing Wang, Cheston Tan, Bihan Wen, and Yeying Jin. Stateplay: State-aware game world models for mechanics-consistent generation. arXiv preprint arXiv:2607.26754, 2026.

[35] Zhiyang Deng, Boran Zhang, Danze Chen, and Yeying Jin. Worldmind: Decoupled game world model for state-aware npc behavior. arXiv preprint arXiv:2608.21439, 2026.

[36] Qi Li, Xingyi Yang, and Xinchao Wang. Badwam: When world-action models dream right but act wrong. arXiv preprint arXiv:2607.15207, 2026.

[37] Linghui Shen, Mingyue Cui, and Xingyi Yang. Badworld: Adversarial attacks on world models. arXiv preprint arXiv:2606.16519, 2026.

[38] Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, Linfeng Zhang, et al. Hunyuan-gamecraft-2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025.

[39] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, and Kaipeng Zhang. Yume1. 5: A textcontrolled interactive world generation model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7752–7761, 2026.

[40] Zhexiao Xiong, Yizhi Song, Hao Kang, Qing Yan, Liming Jiang, Jenson Yang, Zhoujie Fu, Stathi Fotiadis, Angtian Wang, Zichuan Liu, et al. Actworld: From explorable to interactive world model via actionaware memory. arXiv preprint arXiv:2606.17730, 2026.

[41] Shangwen Zhu, Qianyu Peng, Zhao Pu, Zhilei Shu, Xiangrui Ke, Zhaohu Xing, Zizhao Tong, Zeqing Wang, Xinyu Cui, Zian Zheng, et al. Incantation: Natural language as the action interface for multi-entity video world models. arXiv preprint arXiv:2605.18601, 2026.

[42] Liangyang Ouyang, Ruicong Liu, Xuangeng Chu, Kaipeng Zhang, and Yoichi Sato. Helloworld: Enabling socially interactive characters in video world models. arXiv preprint arXiv:2608.05070, 2026.

[43] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36:7594–7611, 2023.

[44] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

[45] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

[46] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with user-directed camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024.

[47] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.

[48] Gunnar Farnebäck. Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis, pages 363–370. Springer, 2003.

[49] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.