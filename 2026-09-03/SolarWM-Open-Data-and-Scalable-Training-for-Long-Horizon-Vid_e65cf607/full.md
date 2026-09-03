# SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models

Junchao Huang<sup>1,2</sup>, Guian Fang<sup>3</sup>, Shengju Qian<sup>4,†</sup>, Xianghao Kong<sup>5</sup>, Zhuoran Zhao<sup>5,6</sup> Wei Huang<sup>7</sup>, Yihua Du<sup>6</sup>, Zixin Zhang<sup>6</sup>, Justin Cui<sup>8</sup>, Yuchao Gu<sup>7</sup>, Yukang Chen<sup>7</sup>, Xinting Hu Tianyu He<sup>9</sup>, Shaoshuai Shi, Zhuotao Tian<sup>2</sup>, Xin Wang, Mike Zheng Shou<sup>3</sup>, Li Jiang<sup>1,2,‡</sup>

<sup>1</sup>CUHK-SZ <sup>2</sup>SLAI <sup>3</sup>NUS <sup>4</sup>CUHK <sup>5</sup>HKUST <sup>6</sup>HKUST-GZ <sup>7</sup>NVIDIA <sup>8</sup>UCLA <sup>9</sup>MSRA # junchaoh.cs@gmail.com Dataset § Code <sup>†</sup> Project Lead <sup>‡</sup> Corresponding Author

Abstract We introduce SolarWM, a fully open foundation for building interactive video world models from data preparation through long-horizon inference. Training across heterogeneous data sources and video backbones is challenging: datasets difer in temporal scale, camera geometry, visual quality, motion, and captioning styles, while video generators use distinct representations and architectures. Naive data mixing and model-specific implementations therefore produce inconsistent supervision and make results dificult to reproduce and compare. SolarWM addresses this coupling with a reconfigurable multi-source data engine and a backbone-native adaptation framework. The engine converts 1.43 million canonical clips from 10 datasets into a unified, frame-aligned contract covering visual observations, metric camera geometry, captions, quality metadata, selection decisions, and provenance, while decoupling source processing from mixture construction. Under shared camera-conditioning, training, and inference interfaces, we instantiate four 5B–33B models based on Wan2.2, LTX-2.5, and MiniMax-H3 while preserving their native representations and objectives. A unified three-stage recipe combines bidirectional adaptation, teacherforced autoregressive initialization, and distribution matching distillation. The resulting causal models enable real-time interaction over rollouts ranging from minutes to hours after being trained on only 5s sequences. By releasing the resulting data, pipeline, recipes, weights, and framework, SolarWM provides a reproducible and extensible foundation for interactive world-model research.

Website (Dataset & Code & Model): https://junchao-cs.github.io/SolarWM-Web/

![](images/da3ae2708946a1c3fab76a8bfae6622f2e56d685699ccc68245d87e567e5e838.jpg)  
Figure 1 | SolarWM: an open foundation for interactive, long-horizon video world models.

## 1. Introduction

“Imagination will often carry us to worlds that never were.” — Carl Sagan

Interactive video world models imagine future visual observations conditioned on camera motion, player actions, or semantic instructions, transforming passive video generators into environments that can be interactively explored and controlled [Bruce et al., 2024, Mao et al., 2025, Valevski et al., 2024, Zhang et al., 2026b]. Recent systems have demonstrated increasingly realistic interaction in game environments and open visual domains, highlighting their potential for simulation, embodied learning, and interactive content creation [DreamX Team et al., 2026, Gao et al., 2026, He et al., 2025, Huang et al., 2025b, Kong et al., 2026, Wang et al., 2026a, Zhang et al., 2026a, Zhu et al., 2026a]. However, the transition from short-clip generation to interactive long-horizon rollout requires models to maintain visual quality and temporal coherence while remaining responsive to control [AlayaWorld Team et al., 2026, Gao et al., 2026, Jiang et al., 2026, Wang et al., 2026b]. Achieving this capability requires coherent supervision across heterogeneous data sources and efective adaptation of diverse video-generation backbones without compromising their pretrained capabilities.

These two requirements are closely coupled. On the data side, existing datasets difer in temporal scale, visual quality, motion distribution, captioning styles, and camera conventions [Hu et al., 2026, Ju et al., 2024, Li et al., 2025, Ling et al., 2023, Wang et al., 2025, Zheng et al., 2025]. Naively combining them can introduce inconsistent supervision, causing models that perform well on individual sources to deteriorate under multi-source training. On the model side, video generators difer substantially in architecture, including their latent representations, attention structures, and conditioning mechanisms [Rui et al., 2026, Zhao et al., 2026a]. Although a shared adaptation strategy is desirable for scalability, overlooking this heterogeneity can compromise the pretrained capabilities; conversely, backbone-specific adaptation limits systematic comparison and extensibility. Existing open systems often overlook this coupling, relying on restricted data sources and largely model-specific implementations [Li et al., 2026, Rui et al., 2026, Zhao et al., 2026a, Zhu et al., 2026a]. Consequently, the field still lacks a unified and reproducible foundation that coordinates multi-source data construction and backbone adaptation while providing a standardized and eficient training protocol across backbones.

We introduce SolarWM, a fully open and unified interactive video world-model foundation comprising a reconfigurable multi-source data infrastructure and a scalable backbone-native adaptation framework. At the data level, SolarWM converts heterogeneous sources into a common training contract with consistent temporal, geometric, semantic, and quality supervision. At the model level, SolarWM provides shared interfaces for camera conditioning, optimization, and rollout while preserving the native representations of individual video generators. We further establish a unified three-stage training recipe for heterogeneous backbones that combines simplicity with high eficiency, comprising bidirectional model training, autoregressive (AR) adaptation [Chen et al., 2024, Gu et al., 2026], and distribution matching distillation (DMD) [Huang et al., 2026b, Yin et al., 2024, Zhuang et al., 2026].

The data infrastructure contains approximately 1.43 million canonical clips from 10 source datasets, corresponding to over 25 TB of physical data. The corpus spans real-world, synthetic, and game environments across diverse scene layouts, camera motions, temporal scales, and motion patterns. Each source is converted into a canonical representation that aligns visual observations, camera geometry, language supervision, and quality metadata. We release the complete processed corpus together with the end-to-end processing pipeline, enabling reproducible reconstruction, extension, and reuse. A key design principle of the infrastructure is to construct a reconfigurable and comprehensive training resource for world models by decoupling computationally intensive source preprocessing from training-mixture construction. Researchers can reproduce the released mixture or define alternative recipes by modifying filtering criteria, sampling ratios, and source weights without rerunning sourcelevel processing.

Built on the canonical corpus produced by this infrastructure, we instantiate a scalable SolarWM model family comprising SolarWM-wan2.2-5B, SolarWM-wan2.2-14B, SolarWM-ltx-2.5-22B, and SolarWM-minimax-h3-33B [HaCohen et al., 2026, MiniMax, 2026, Wan et al., 2025], with model sizes spanning 5B to 33B parameters. The backbone-native adaptation introduces only the interfaces required for camera conditioning and long-horizon rollout while preserving each generator’s native representations and optimization objectives. This design yields an extensible and directly comparable model family that supports high-fidelity, camera-controllable video generation across real-world, synthetic, and game environments.

Experiments further reveal that strong long-horizon world models can be obtained with a simple yet highly eficient training procedure. First, our models achieve state-of-the-art performance without specialized ODE or consistency-distillation (CD) initialization [Huang et al., 2026b, Yin et al., 2025, Zhu et al., 2026b]. Second, most optimization should be performed during bidirectional training; AR adaptation then converges rapidly, while DMD requires even fewer optimization steps. Third, after training on only 5s sequences, the resulting models support open-ended rollouts over minutes-to-hours horizons without additional long-sequence fine-tuning or attention-sink mechanisms. These key findings show that scalable long-horizon generation can be achieved without a complex training pipeline or computationally intensive training on long videos, provided that data construction, backbone adaptation, and training stages are properly aligned.

Our main contributions are summarized as follows:

• A fully open and unified foundation. We introduce SolarWM, a fully open and unified interactive video world-model foundation integrating a reconfigurable multi-source data infrastructure with a scalable backbone-native adaptation framework across data construction, model training, and long-horizon inference.

• A reconfigurable multi-source data infrastructure. We process approximately 1.43 million clips from 10 datasets covering real-world, synthetic, and game environments, and release the corpus and processing pipeline to support reproducible and flexible training-mixture construction.

• A scalable backbone-native adaptation framework and model family. We train four 5B–33B models across heterogeneous backbones using a unified data recipe and a simple, eficient threestage training procedure comprising bidirectional training, AR adaptation, and DMD. The models achieve state-of-the-art performance and support open-ended rollouts over minutes-to-hours horizons after training solely on 5s sequences.

## 2. Open-Source Release

We will release the complete processed corpus including all 1.43 million canonical clips from 10 source datasets, together with their annotations, metadata, intermediate records, quality assessments, selection decisions, and provenance information, as well as the end-to-end data-processing pipeline. Because source-level preprocessing is decoupled from mixture construction, researchers can define alternative training recipes by changing filtering criteria, sampling ratios, source weights, or split definitions, without rerunning the computationally intensive source preprocessing. We will also release the training recipes, model weights, and implementations of bidirectional training, AR adaptation and rollout, and DMD training and inference. Collectively, these artifacts provide a complete and reproducible foundation for interactive video world-model research, enabling flexible data-mixture design, controlled cross-backbone comparison, and systematic extension to new data and models.

## 3. Related Work

## 3.1. Interactive Video World Models

Interactive video world models extend passive video generation into visual simulation: given an initial observation, they roll out future observations in response to actions, camera motion, or semantic instructions. Early systems focused on compact action spaces: Genie learns latent actions from Internet videos [Bruce et al., 2024], DIAMOND models Atari environments [Alonso et al., 2024a], and GameNGen simulates DOOM from recorded state–action trajectories [Valevski et al., 2024]. Minecraft later became a common testbed for autoregressive and streaming interaction, as illustrated by MineWorld [Guo et al., 2025], Memory Forcing [Huang et al., 2025a], and the Matrix-Game series [He et al., 2025, Zhang et al., 2025]. Their progress establishes video generators as interactive simulators, but many remain tied to a particular environment, collection process, or control vocabulary.

Recent work extends the setting to open-domain visual exploration. Yume, WorldPlay, and AlayaWorld support camera- or keyboard-controlled world extension with mechanisms for long-range consistency [AlayaWorld Team et al., 2026, Mao et al., 2025, Sun et al., 2025]. LingBot-World, ABot-World-0, DreamX-World, SANA-WM, Genie 3, LIVE, and recent Matrix-Game variants further improve visual quality, rollout length, interaction, or deployment eficiency [DreamX Team et al., 2026, Gao et al., 2026, Huang et al., 2026a, Jiang et al., 2026, Parker-Holder and Fruchter, 2025, Riemann Dynamics, 2026a, Robbyant Team et al., 2026, Wang et al., 2026b, Zhu et al., 2026a]. Together, these systems show that an interactive world model depends not only on the base video generation model, but also on the data, adaptation, causal-training, and inference pipelines surrounding it.

## 3.2. Training Frameworks and Open Research Stacks

Most high-quality video generators are pretrained with bidirectional temporal attention, whereas interactive rollout requires causal prediction from generated history. Difusion Forcing provides a causal sequence formulation [Chen et al., 2024]; later methods convert bidirectional video models through staged distillation, train on model-generated context to reduce exposure bias, and enable fewstep sampling through ODE, consistency, flow-map, or distribution-matching objectives [Gu et al., 2026, Huang et al., 2026b, Yin et al., 2024, 2025, Zhao et al., 2026b, Zhu et al., 2026b]. Astrolabe studies forward-process reinforcement learning for post-training distilled autoregressive video models [Zhang et al., 2026d]. KVPO further proposes ODE-native GRPO for aligning streaming autoregressive video models through semantic exploration over historical KV caches [Zhang et al., 2026c]. These methods provide the main ingredients for long-horizon causal generation, but they do not by themselves provide a unified training stack that can be applied reproducibly across heterogeneous video backbones.

Beyond these algorithmic advances, only a limited number of systems release training implementations. DIAMOND, Yume, minWM, BiWM, ForgeWM, SANA-WM, WorldPlay, and AlayaWorld expose code for some or all training stages [AlayaWorld Team et al., 2026, Alonso et al., 2024a, Li et al., 2026, Mao et al., 2025, Rui et al., 2026, Sun et al., 2025, Zhao et al., 2026a, Zhu et al., 2026a], whereas most systems release only checkpoints, inference code, or demonstrations. Even when training code is available, the released package is not always directly reproducible or reconfigurable: processed data, source-to-training construction, exact selection and mixture recipes, or checkpoint-matched optimization configurations may still be missing. Reproducing a model or adapting the pipeline to a new dataset or backbone can therefore require reconstructing substantial parts of the original system. Moreover, existing multi-backbone releases typically cover only two or three backbone families, leaving it unclear whether a training design transfers broadly or depends on model-specific engineering. Table 1 provides a systematic comparison of representative interactive video world models based on their oficial project pages, code repositories, and released artifacts. Specifically, it examines the availability of model weights, inference and training code, processed training data, complete data-selection records, executable source-to-training pipelines, exact data and optimization recipes, and support for multiple video-backbone families [Alaya Lab, 2026, Alonso et al., 2024b, Amap CVLab, 2026, Decart and Etched, 2024, DreamX Team, 2026, ForgeWM Team, 2026, LynnReal AI, 2026, Microsoft Research, 2026, NVIDIA Research, 2026, Parker-Holder and Fruchter, 2025, Riemann Dynamics, 2026b, Robbyant Team, 2026a,b, ShengShu AI MIN Lab, 2026, Skywork AI, 2026, Tencent Hunyuan, 2026, Yume Team, 2026].

Table 1 | Release matrix for representative interactive video world models. Release status was verified from oficial artifacts as of August 18, 2026; SolarWM entries indicate commitments for this release. ✓: public; △: reduced, or temporarily unavailable; —: not identified. “Pipeline”: executable source-to-training workflow; “Full Rec.”: complete annotations, scores, provenance, and decisions; “Exact Recipe”: executable data and optimization configurations. Statistics cover public or committed payloads; “Multi-BB” denotes multiple video-backbone families.
<table><tr><td>System</td><td>Domain</td><td>Weights Infer. Train</td><td></td><td></td><td>Data</td><td>Full Rec.</td><td></td><td>Pipeline Exact Recipe Multi-BB</td><td></td><td>Link</td></tr><tr><td colspan="9">Game- and action-centric systems</td><td></td></tr><tr><td>DIAMOND</td><td>Atari / CSGO</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td>Project</td></tr><tr><td>Oasis</td><td>Minecraft</td><td>Δ</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>MineWorld</td><td>Minecraft</td><td>Δ</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>GitHub</td></tr><tr><td>Matrix-Game 2.0</td><td>Game / synthetic</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>Matrix-Game 3.0</td><td>Game / real-world</td><td>Δ</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>ABot-World-0 ForgeWM</td><td>Multi-source</td><td>Δ</td><td>√ √</td><td></td><td>✓ 2.74 TB, 30k clips ✓ 95 GB, 40k clips</td><td>△</td><td></td><td>√</td><td></td><td>Project GitHub</td></tr><tr><td colspan="9">Game √</td><td></td></tr><tr><td colspan="9">Multi-domain and general-purpose systems</td><td></td></tr><tr><td>Google Genie 3</td><td>Multi-domain</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>LingBot-World</td><td>Multi-domain</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>LingBot-World 2.0</td><td>Multi-domain</td><td>Δ</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>Matrix-Game 3.5</td><td>Multi-domain</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>GitHub</td></tr><tr><td>Yume</td><td>Multi-domain</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>Δ</td><td>∆</td><td></td><td>GitHub</td></tr><tr><td>DreamX-World 1.0</td><td>Multi-domain</td><td>Δ</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Project</td></tr><tr><td>SANA-WM</td><td>Multi-domain</td><td>Δ</td><td>√</td><td>√</td><td>△ 235 GB, 1.6k clips</td><td>Δ</td><td>∆</td><td>∆</td><td></td><td>Project</td></tr><tr><td>HY-WorldPlay 1.5</td><td>Multi-domain</td><td>Δ</td><td>√</td><td>√</td><td></td><td></td><td>∆</td><td>∆</td><td>√(2)</td><td>GitHub</td></tr><tr><td>minWM</td><td>Multi-domain</td><td>√</td><td>√</td><td>√</td><td>Δ 527 GB, 19k clips</td><td></td><td>√</td><td>√</td><td>√(2)</td><td>GitHub</td></tr><tr><td>BiWM</td><td>Multi-domain</td><td></td><td>√</td><td>√</td><td>✓ 16.9 GB, 25k clips</td><td></td><td>∆</td><td>∆</td><td>√(3)</td><td>GitHub</td></tr><tr><td>AlayaWorld v1.1</td><td>Multi-domain</td><td>√</td><td>√</td><td>√</td><td>Δ 1.5 GB, 18k clips</td><td></td><td>Δ</td><td>Δ</td><td></td><td>GitHub</td></tr><tr><td>SolarWM (Ours)</td><td>Multi-domain</td><td>√</td><td>√</td><td>√</td><td>✓ 25.85 TB, 1426k clips</td><td>√</td><td>√</td><td>√</td><td>√(4)</td><td>Project</td></tr></table>

SolarWM addresses both gaps with a compact, unified, and fully open training framework. The same three-stage recipe—bidirectional camera-conditioned adaptation, teacher-forced AnyFlow autoregressive initialization, and DMD-based causal training—is designed for all four models in the SolarWM family: SolarWM-wan2.2-5B, SolarWM-wan2.2-14B, SolarWM-ltx-2.5-22B, and SolarWM-minimaxh3-33B. Rather than assembling a separate training system for each model, SolarWM shares the data, camera-conditioning, optimization, and rollout interfaces while retaining only the backbone-native representations and objectives required by each generator. This demonstrates that the simple recipe is a reusable cross-backbone framework, rather than a model-specific training procedure.

We will open-source the complete stack for all four model instantiations, including the implementation and configuration of every training stage, backbone adapters, processed corpus and complete selection records, executable source-to-training pipeline, exact data-mixture and optimization recipes, checkpoint-matched settings, model weights, and inference code. The released artifacts will therefore support both end-to-end reproduction of each of the four models and controlled reconfiguration across data filters, mixtures, backbones, schedules, and stage settings. The value of SolarWM lies not only in simplifying causal world-model training, but also in making a four-model, recipe-complete research stack available for direct reproduction, comparison, and extension.

![](images/fccd0cbecafda3a393b3a0ee3a2f3f580b5cc02f9d95dfff78ab003653ceddfc.jpg)  
Figure 2 | Overview of the SolarWM-5B training pipeline and hour-scale inference results

## 3.3. Data for Interactive World Modeling

Data for interactive world modeling draws on several overlapping data traditions. Large video–text corpora such as WebVid-10M [Bain et al., 2021], InternVid [Wang et al., 2024], and MiraData [Ju et al., 2024] provide visual and semantic diversity but generally lack frame-aligned geometric control. Camera-oriented datasets such as DL3DV-10K, RealCam-Vid, SpatialVID, and MultiCamVideo provide complementary pose, depth, or trajectory supervision [Bai et al., 2025, Ling et al., 2023, Wang et al., 2025, Zheng et al., 2025]. Multi-domain collections including OmniWorld, Sekai, MIND, and ABOT-World broaden the range of scenes and interaction settings [Jiang et al., 2026, Li et al., 2025, Ye et al., 2026, Zhou et al., 2025].

These sources are complementary but not directly interchangeable. They difer in temporal scale, resolution, visual quality, camera evidence, coordinate conventions, metric scale, captions, and motion distributions. Their released media and annotations also do not necessarily specify the processing that produces model-ready samples. The challenge is therefore to construct a consistent and auditable training contract, not simply to concatenate more datasets.

SolarWM addresses this gap with the open data engine in Section 5. It normalizes heterogeneous sources into a frame-aligned representation and separates source processing from mixture construction. We will release the processed corpus, annotations, quality and provenance records, accepted and rejected samples, executable pipeline, and exact recipes. This allows users to reproduce our mixtures or reconfigure filters, source weights, temporal views, and model readers without rerunning the annotation pipeline.

## 4. Training Pipeline

We train the world model in three stages: bidirectional adaptation, teacher-forced AnyFlow autoregressive initialization, and DMD-based causal training. The first stage adapts the pretrained video generator to camera-conditioned world data under bidirectional attention. The second switches to causal attention and combines teacher forcing with the AnyFlow loss [Gu et al., 2026], directly producing a few-step autoregressive initializer. This stage is fast because it mainly activates causal prediction in the well-trained bidirectional model rather than relearning appearance and motion. By making the autoregressive initializer few-step from the outset, TF-AnyFlow removes the need for the additional Causal ODE and Causal Consistency Distillation (CD) initialization stages used by Causal Forcing [Zhu et al., 2026b] and Causal Forcing++ [Zhao et al., 2026b], respectively. The final DMD [Yin et al., 2024] stage then trains on model-generated trajectories to align the causal generator with its inference-time distribution.

## 4.1. Bidirectional Adaptation

Let $\mathbf { z } _ { 0 }$ denote a clean video latent and let c collect the text, image, and camera conditions. Given a noisy latent $\mathbf { z } _ { t } ,$ , the model predicts the native flow or velocity target ${ \bf u } _ { t }$ of its backbone. The first stage minimizes

$$
\mathcal { L } _ { \mathrm { b i d } } = \mathbb { E } _ { \mathbf { z } _ { 0 } , t , \epsilon } \left[ \| f _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } ) - \mathbf { u } _ { t } \| _ { 2 } ^ { 2 } \right] ,\tag{1}
$$

using bidirectional attention over the complete training window. This stage adapts the pretrained visual prior to camera-conditioned world data while retaining unrestricted temporal context. Camera control is injected through fused-PRoPE (Section 6.1): projective rotations act directly on the query, key, and value tensors in the existing self-attention path, requiring neither a separate control branch nor an additional attention pass. The resulting checkpoint provides both the initialization for causal training and the fixed bidirectional reference for DMD.

## 4.2. Teacher-Forced AnyFlow Initialization

We next introduce causality while keeping the optimization target fully supervised. The latent sequence is divided into ordered blocks. When predicting the current block, the model can attend only to the current noisy state and clean ground-truth history; future blocks are hidden. For a sampled pair of noise levels $( t , r )$ , the training objective is

$$
\mathcal { L } _ { \mathrm { T F } \cdot \mathrm { A F } } = \mathbb { E } \left[ \sum _ { k } \ell _ { \mathrm { A F } } \left( f _ { \phi } ; \mathbf { z } _ { t } ^ { k } , t , r , \mathbf { z } _ { 0 } ^ { < k } , \mathbf { c } \right) \right] ,\tag{2}
$$

where $\ell _ { \mathrm { A F } }$ supervises the flow map between arbitrary noise levels, directly enabling few-step autoregressive generation.

Initialized from the bidirectional model, this stage is a short causal adaptation rather than a new round of representation learning. The AnyFlow objective exposes the model directly to the finite noise transitions used by few-step sampling, while teacher forcing supplies a stable clean history. Their combination yields a few-step autoregressive checkpoint that can be passed directly to DMD, without a separate Causal ODE or Causal CD stage.

## 4.3. DMD-based Causal Training

Teacher-forced training still observes clean history, while inference conditions on the model’s own predictions. We address this exposure gap with DMD-based causal training. A causal student generates trajectories under the same temporal rule used at inference with a detached rollout-and-replay procedure [Zhuang et al., 2026] to allow KV Cache gradients. A frozen bidirectional teacher estimates the target distribution, while a trainable fake-distribution model follows the student’s evolving rollout distribution. Their diference provides the distribution-matching direction for the student.

![](images/5a444a360c7f7ce5dfa6891f00c00ad788c1b5db63aefdc76fe6d9be450edb35.jpg)  
Figure 3 | Overview of the Solar open data engine.

Writing $G _ { \psi }$ for the causal generator, $s _ { \mathrm { r e a l } }$ for the teacher score, and $s _ { \mathrm { f a k e } }$ for the score of the generated distribution, the DMD update takes the general form

$$
\nabla _ { \psi } \mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } \left[ J _ { G _ { \psi } } ^ { \top } w ( t ) \left( s _ { \mathrm { f a k e } } ( \mathbf { z } _ { t } , t , \mathbf { c } ) - s _ { \mathrm { r e a l } } ( \mathbf { z } _ { t } , t , \mathbf { c } ) \right) \right] ,\tag{3}
$$

where $J _ { G _ { \psi } }$ maps the distribution-matching signal back to the generator parameters and �(�) controls its scale across noise levels. The fake-distribution model is optimized on generated samples, whereas the bidirectional teacher remains fixed. This separation preserves a stable quality reference while allowing the training signal to track the changing student distribution.

## 5. Open Data Engine

The data component of SolarWM is designed as an engine rather than as a fixed list of training clips. Its central principle is to process and annotate every canonical clip from every source before applying any training-time selection criteria. A clip that does not satisfy our released default recipe is therefore retained in a separate rejected partition, together with its annotations, metric values, and machine-readable rejection reasons. The released high and xhigh partitions are two convenient quality tiers, not an irreversible definition of valid data. Users can change thresholds, disable a filtering criterion, rebalance sources, or construct a backbone-specific temporal view without rerunning the expensive camera-estimation, captioning, and quality-assessment pipelines.

Our current corpus contains 1.43M canonical clips derived from 10 source datasets and organized into 14 independently addressable processing partitions, which we refer to as dataset owners. The 10 source datasets are ABOT-World [Jiang et al., 2026], DL3DV [Ling et al., 2023], MiraData [Ju et al., 2024], RealCam [Zheng et al., 2025], SpatialVID [Wang et al., 2025], Sekai-Game and Sekai-Walking [Li et al., 2025], MIND [Ye et al., 2026], MultiCamVideo [Kling Team, 2026], and OmniWorld [Zhou et al., 2025]. At the owner level, DL3DV is divided into two temporal views, DL3DV-10s and DL3DV-60s, while three additional Clean Plate owners are derived from MiraData, Sekai-Walking, and SpatialVID. This organization yields 14 dataset owners in total, each of which can be filtered, weighted, and selected independently during training-recipe construction. As illustrated in Figure 3, the Solar Open Data Engine exposes the complete path from source ingestion to training-recipe construction. The remainder of this section describes the unified data contract, processing stages, selection policies, and resulting corpus.

## 5.1. Unified Data Representation

We normalize every source into a unified, frame-aligned sample representation. Each sample is assigned a stable identifier, and all temporal modalities, including video, camera poses, intrinsics,

captions, and audio when available, are aligned to the same, deterministically selected frame interval. Unified sample schema. Every sample follows the same logical contract:

$$
S _ { i } = ( V _ { i } , P _ { i } , K _ { i } , C _ { i } , m _ { i } , q _ { i } , \pi _ { i } ) ,\tag{4}
$$

where $V _ { i }$ is the video, $P _ { i } \in \mathbb { R } ^ { N \times 4 \times 4 }$ contains metric camera-to-world transforms, $K _ { i } \in \mathbb { R } ^ { N \times 4 }$ stores per-frame $( f _ { x } , f _ { y } , c _ { x } , c _ { y } ) , C _ { i }$ is the dense caption, $m _ { i }$ contains source and media metadata, $q _ { i }$ is the complete metric record, and $\pi _ { i }$ stores provenance and lineage. The schema records the source owner, original identifier, temporal interval, processing versions, and the outcome of every stage. This makes samples from diferent sources interchangeable to the training reader while keeping each derived artifact traceable to its input.

We separate three namespaces that are often conflated in dataset releases. The physical corpus contains the canonical samples and annotations; a logical recipe contains split membership, tier policy, source weights, and repeat factors; and a model view contains backbone-specific windows or precomputed latents. Changing a recipe therefore does not duplicate the underlying videos, and changing a VAE does not alter the underlying data selection.

## 5.2. Metric-Scale Camera Annotation

We follow the robust camera-annotation design of SANA-WM [Zhu et al., 2026a], selecting the annotation path according to the geometry available from each source. For video-only sources, Pi3X estimates a temporally consistent but scale-ambiguous scene structure, while MoGe-2 provides perframe metric-depth anchors. Their depths are aligned and fused before being passed to a modified VIPE SLAM backend. VIPE then estimates the 6-DoF camera trajectory and performs bundle adjustment with independently optimizable per-frame $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ , initialized by GeoCalib. This per-frame formulation is important for clips containing zoom or focal-length drift, for which a single intrinsic matrix for the entire video is insuficient.

When a source provides ground-truth or COLMAP camera poses, we preserve the supplied trajectory rather than replacing it with a SLAM estimate. Pi3X is used only to connect the predicted scene structure to the trajectory’s metric gauge: predicted and reference camera centers are aligned with a robust Umeyama Sim(3) fit, which is re-estimated from the lowest-residual 80% of frames. The pipeline also supports sources with metric depth but no camera trajectory, using the depth directly as the scale anchor for VIPE.

All modes emit a frame-aligned metric camera-to-world (c2w) trajectory $P _ { i } \in \mathbb { R } ^ { N \times 4 \times 4 }$ and per-frame pixel-space intrinsics $K _ { i } \in \mathbb { R } ^ { N \times 4 }$ . Translations remain in metric units; no corpus-wide translation normalization is applied. Source-specific coordinate conversion is performed once at ingestion and recorded in the sample metadata. We require video frames, poses, and intrinsics to share exactly the same frame indices. Each sample is further checked for finite geometry, positive focal lengths, field of view in a plausible range, focal consistency $| f _ { x } - f _ { y } | / ( \frac { 1 } { 2 } ( f _ { x } + f _ { y } ) + \epsilon )$ , and temporal scale stability std(�)/mean(�). Missing or non-finite camera evidence fails closed whenever the corresponding recipe uses a camera gate.

## 5.3. LTX Clean Plate Processing

Dynamic people and vehicles are a frequent source of ambiguity for camera-controlled world models: they introduce motion that is neither explained by the camera nor consistently controllable. We therefore produce three additional Clean Plate derivative owners with the LTX-2.3 Clean Plate IC-LoRA pipeline. Figure 4 provides a visual comparison of the source clips and their cleaned counterparts.

![](images/966923585f836a3b30bcd36947890f0095e25526fb062c5daa847fa9648d0c2a.jpg)  
Figure 4 | LTX Clean Plate processing. Representative frames before (left) and after (right) applying the LTX-2.3 Clean Plate pipeline. The transformation removes dynamic people and vehicles while preserving the static scene layout and the source camera trajectory, yielding cleaner supervision for camera-controlled world-model training.

For each accepted source interval, the transformation removes people and vehicles while retaining scene layout and camera motion. We use eight denoising steps with strength 1.0 and the Clean Plate prompt. Inputs are processed at 1248×704, and outputs are saved at 1280×720 and 16 fps.

Clean outputs are not assumed to inherit the quality of their source clips. Removal can introduce texture artifacts, temporal discontinuities, or altered motion statistics, so captions, visual metrics, semantic metrics, and camera diagnostics are recomputed for every output window. Metric scale is restored from deterministic source-frame correspondences; invalid or ambiguous scale reconstruction fails closed. The current release contains 543k clean clips: 298k SpatialVID-Clean clips (73k at 81 frames and 224k at 160 frames), 135k MiraData-Clean clips, and 109k Sekai-Walking-Clean clips. These are independent recipe owners, rather than silent replacements for the original datasets.

## 5.4. Dense Captioning and Multi-Axis Annotation

Kimi-K2.6 captions. All 1.43M canonical clips are captioned by the same Kimi-K2.6 production pipeline. Its prompt asks the model to inspect the entire clip and describe only the persistent environment and stable scene content. Dynamic entities, actions, camera motion, shot terminology, and speculative or media-related wording are excluded from the training caption, thereby reducing the risk that the text condition leaks camera-control information. The prompt and response contract is summarized in Table 2.

Only dense\_caption becomes a text condition; the remaining fields are stored as structured annotations and can be used by source-specific recipe policies. Transition predictions remain marked unverified, and the presence of a caption alone does not afect tier assignment.

Quality and motion metrics. No single score captures the properties needed for interactive worldmodel training. We consequently retain a vector of complementary measurements:

• camera integrity: finite pose/intrinsic tests, horizontal and vertical field of view, focal divergence, and scale coeficient of variation;

• visual quality: saturation statistics and the mean of DOVER’s technical and aesthetic scores;

• motion: FFmpeg VMAF Motion and dense correspondence magnitude from UniMatch/GMFlow;

• temporal consistency: PySceneDetect cut count and the Kimi-K2.6 transition field; and

• semantic suitability: Kimi-K2.6 quality, entity density, scene type, and explicit flags such as blurry, ui\_overlay, watermark, and near\_static.

Table 2 | Kimi-K2.6 prompt and structured-output contract. The response is a strict JSON object with exactly six top-level fields.
<table><tr><td>Item</td><td>Type or allowed values</td><td>Prompt requirement and released representation</td></tr><tr><td>Visual input</td><td>1 fps; at most 64 frames</td><td>Sample the entire clip with ffmpeg round-up; maximum image edge 768 pixels, JPEG qscale 3, and no audio.</td></tr><tr><td>Caption target</td><td>One English paragraph; 60– 150 words</td><td>Describe persistent architecture, terrain, vegetation, water, furniture, materials, lighting, weather, atmosphere, and stable spatial relations.</td></tr><tr><td>Caption exclusions</td><td>Hard negative instruction</td><td>Exclude people and animals, dynamic or uncertain vehi- cles, actions, camera motion, shot/viewpoint terms, media terminology, speculation, and invented details.</td></tr><tr><td>Response protocol</td><td>Strict JSON; no extra keys</td><td>Temperature 0, seed 0, thinking=false; return only the JSON object, without Markdown or explanatory text.</td></tr><tr><td>dense_caption</td><td>String satisfying the caption contract</td><td>Becomes meta.caption and prompt.txt (caption fol- lowed by one newline).</td></tr><tr><td>vlm_entity_density Integer in {1,2,3}</td><td></td><td>Peak combined density of visible people, vehicles, and animals: none, sparse, or dense; stored as metrics.vlm_entity_density.</td></tr><tr><td>vlm_quality</td><td>Number in {1.0, 2.0, 3.0, 4.0, 5.0}</td><td>Overall visual usability from unusable to clean and richly describable; stored as metrics.vlm_quality.</td></tr><tr><td>reject_flags</td><td>Deduplicated subset of eight flags</td><td>Allowed values are text_heavy, watermark, ui_overlay, blurry, near_static, low_light, nsfw, and single_color; stored as</td></tr><tr><td>scene_type</td><td>real_world, rendered, game, animation, or mixed</td><td>metrics.vlm_reject_flags. Classifies the visible world and is stored as metrics.vlm_scene_type.</td></tr><tr><td>scene_transition</td><td>Object with exactly label, count, timestamps_sec, and evidence</td><td>The label is none, possible, or definite; con- tinuous fast motion is not a transition. Stored as metrics.vlm_scene_transition.</td></tr></table>

All measurements, including ones unused by our default selection, remain in the released record. This is important because diferent source domains have diferent natural score distributions: a universal motion or aesthetic threshold would remove useful camera trajectories from one domain while retaining artifacts in another.

## 5.5. Source-Aware Filtering and Quality Tiers

We freeze a versioned policy per dataset owner. The policy produces three disjoint physical labels: xhigh, high, and rejected. A sample is xhigh if it satisfies both the kept rule and the stricter promotion rule; otherwise, a sample satisfying the kept rule is high. All other samples are retained as rejected with their failure reasons. Every metric named by a rule must be present, finite, and correctly typed.

For compactness, Table 3 uses the following notation. � is the frame count; � is Kimi-K2.6 quality; � is its reject-flag set; � is the Kimi transition count; � is entity density; � is mean DOVER; � is VMAF Motion; � is UniMatch motion; � is mean saturation; and � is the PySceneDetect cut count. �<sub>120</sub> requires both fields of view in [25, 120] degrees, focal divergence $\leq 0 . 2 0$ , and scale coeficient of variation ≤ 2; �<sub>125</sub> changes only the upper field-of-view bound to 125 degrees. � denotes the stricter post-Clean geometry and lineage check, and � denotes a non-empty caption. All intervals are inclusive. An “xhigh extra” is applied in addition to the kept rule.

Table 3 | Frozen source-aware selection policies. The table is executable policy documentation rather than a qualitative summary.
<table><tr><td rowspan=1 colspan=1>Dataset owner</td><td rowspan=1 colspan=1>Kept rule</td><td rowspan=1 colspan=1>xhigh extra</td></tr><tr><td rowspan=1 colspan=1>DL3DV-10s</td><td rowspan=1 colspan=1> $C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; Q \geq 4 ; D \geq . 2 0$ </td><td rowspan=1 colspan=1> $Q = 5 ; D \ge . 3 0$ </td></tr><tr><td rowspan=1 colspan=1>DL3DV-60s</td><td rowspan=1 colspan=1> $C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; Q \geq 4 ; D \geq . 2 0$ </td><td rowspan=1 colspan=1> $Q = 5 ; D \ge . 3 0$ </td></tr><tr><td rowspan=1 colspan=1>MIND</td><td rowspan=1 colspan=1> $C _ { 1 2 5 } ; R = \varnothing ; T = 0 ; Q \ge 4$ </td><td rowspan=1 colspan=1>Q=5</td></tr><tr><td rowspan=1 colspan=1>ABOT</td><td rowspan=1 colspan=1> $C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; R = \varnothing ; T = 0 ; Q \ge 4$ </td><td rowspan=1 colspan=1>Q=5</td></tr><tr><td rowspan=1 colspan=1>OmniWorld</td><td rowspan=1 colspan=1> $C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; R = \varnothing ; T = 0 ; Q \ge 4 ; D \ge . 3 0 ; V \in [ . 5 , 1 0 0 ] ; U \in [ 3 , 2 5 0 ]$ </td><td rowspan=1 colspan=1> $Q = 5 ; U \leq 2 0 0$ </td></tr><tr><td rowspan=1 colspan=1>Sekai-Game</td><td rowspan=1 colspan=1> $U \leq 3 5 0 ; D \geq . 2 0 ; Q \in [ 3 , 5 ] ; \mathtt { b l u r r y } \notin R$ </td><td rowspan=1 colspan=1> $Q \in [ 4 , 5 ] ; R = \emptyset$ </td></tr><tr><td rowspan=1 colspan=1>MiraData</td><td rowspan=2 colspan=1> $A ; N \ge 8 1 ; C _ { 1 2 0 } ; S \in ~ [ 0 , 1 8 0 ] ; Q \ge 4 ; D \ge . 4 0 ;$ resolution = 1280×720; R = ∅; $T = 0 ; V \in [ . 5 , 5 0 ] ; U \in [ 3 , 1 2 0 ]$ </td><td rowspan=2 colspan=1> $Q = 5 ; U \le 1 0 0$ </td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RealCam</td><td rowspan=2 colspan=1> $N \geq 8 1 ; C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; T = 0 ; Q \geq 4 ; V \in [ . 5 , 1 0 0 ] ; D \geq . 3 0 ; U \in [ 3 , 2 0 0 ] ;$  $\{ \mathsf { b l u r r y } , \mathsf { u i \_ o v e r l a y } \} \cap R = \emptyset$ </td><td rowspan=2 colspan=1> $Q \quad = \quad 5 ; R \quad \subseteq \quad$ {near_static}</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>MultiCamVideo</td><td rowspan=1 colspan=1> $C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; R = \emptyset ; T = 0 ;$ Q≥ 4; D≥ .40</td><td rowspan=1 colspan=1>Q=5</td></tr><tr><td rowspan=1 colspan=1>Sekai-Walking</td><td rowspan=1 colspan=1> $A ; N \geq 8 1 ; C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; T = 0 ; Q \geq 4 ; V \in [ . 5 , 5 0 ] ; D \geq . 3 5 ; U \in [ 3 , 1 2 0 ] ;$  $R \subseteq \{ { \mathrm { w a t e r m a r k } } , { \mathrm { n e a r } } _ { - } { \mathrm { s t a t i c } } \}$ </td><td rowspan=1 colspan=1> $Q = 5 ; R = \emptyset$ </td></tr><tr><td rowspan=1 colspan=1>MiraData-Clean</td><td rowspan=2 colspan=1> $\mathrm { e x a c t k e p t { - s o u r c e } \ l i n e a g e ; ~ } A \mathrm { ; ~ } N \geq 8 1 \mathrm { ; ~ } G \mathrm { ; ~ } \mathrm { r e s o l u t i o n _ { \mathrm { s r c } } = r e s o l u t i o n _ { \mathrm { o u t } } = 1 2 8 0 \times 7 2 0 ; ~ }$  $S \in [ 0 , 1 8 0 ] ; Q \geq 4 ; D \geq . 4 0 ; V \in [ . 5 , 5 0 ] ; U \in [ 3 , 1 2 0 ] ; R = \varnothing ; T = 0 ; J \in [ 0 , 1 ]$ </td><td rowspan=2 colspan=1>Q = 5; D≥ .45; U≤ 100</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Sekai-Walking-Clean</td><td rowspan=2 colspan=1> $A ; N \geq 8 1 ; G ; S \in [ 0 , 1 8 0 ] ; T = 0 ; Q \geq 4 ; D \geq . 3 5 ; U \in [ 3 , 1 2 0 ] ; V \in [ . 5 , 5 0 ] ; E \neq 3 ;$  $R \subseteq \{ { \mathrm { w a t e r m a r k } } , { \mathrm { n e a r } } _ { - } { \mathrm { s t a t i c } } \}$ </td><td rowspan=2 colspan=1> $Q = 5 ; R = \emptyset$ </td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SpatialVID</td><td rowspan=1 colspan=1> $A ; N \geq 8 1 ; C _ { 1 2 0 } ; S \in [ 0 , 1 8 0 ] ; T = 0 ; Q \geq 4 ; V \in [ . 5 , 5 0 ] ; D \geq . 3 5 ; U \in [ 3 , 1 2 0 ] ;$  $J = 0 ; R \subseteq \{ \mathrm { w a t e r m a r k , n e a r \_ s t a t i c } \}$ </td><td rowspan=1 colspan=1> $Q = 5 ; R = \emptyset$ </td></tr><tr><td rowspan=1 colspan=1>SpatialVID-Clean</td><td rowspan=2 colspan=1> $A ; N \geq 8 1 ; G ; S \in [ 0 , 1 8 0 ] ; T = 0 ; Q \geq 4 ; V \in [ . 5 , 5 0 ] ; D \geq . 3 5 ; U \in [ 3 , 1 2 0 ] ; J = 0 ;$  $E \neq 3 ; R \subseteq \{ { \mathrm { w a t e r m a r k } } , { \mathrm { n e a r } } _ { - } { \mathrm { s t a t i c } } \}$ </td><td rowspan=2 colspan=1> $Q = 5 ; R = \emptyset$ </td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

The table intentionally contains asymmetric policies. For example, Sekai-Game preserves high-motion game trajectories without imposing the generic camera-geometry or saturation gates, whereas clean owners require post-transformation geometry checks. Likewise, some metrics are retained only as annotations for a given source. This avoids retroactively presenting every computed metric as a selection gate. It also distinguishes the current Kimi-K2.6 tier policy from earlier internal recipes that used diferent semantic-score scales or fewer visual gates.

## 5.6. Open Recipes and Split Construction

The physical WebDataset release stores every canonical row under one of kept-high, kept-xhigh, or rejected, while recipe indices provide logical training and evaluation membership. Splits are constructed using stable sample identifiers and are verified to contain no overlapping identifiers. We also publish independent 100-sample test views for each owner (1,400 rows in total) for data inspection and reader validation.

A recipe may reference a row repeatedly without copying it. For example, our 81f short recipe spans ten owners and contains 600,320 physical training rows, but source balancing applies a repeat factor of six to ABOT, MiraData, and Sekai-Game, producing 870,210 virtual occurrences per epoch; its heldout test view contains 1,000 rows. Because selection, balancing, and temporalization are expressed in indices, users can audit the released recipe or create alternatives such as a metric-camera-only subset, less aggressive motion filtering, a diferent source mixture, or a longer temporal curriculum. Rejected rows remain useful for such experiments because they carry the same annotations and explicit rejection provenance as kept rows.

Table 4 | Canonical clip inventory by dataset owner. High and xhigh are disjoint tiers; kept is their sum.
<table><tr><td>Dataset owner</td><td>All</td><td>High</td><td>xhigh</td><td>Rejected</td></tr><tr><td>ABOT</td><td>30,966</td><td>127</td><td>30,715</td><td>124</td></tr><tr><td>DL3DV-10s</td><td>120,924</td><td>54,528</td><td>60,396</td><td>6,000</td></tr><tr><td>DL3DV-60s</td><td>10,077</td><td>3,578</td><td>6,065</td><td>434</td></tr><tr><td>MIND</td><td>533</td><td>117</td><td>402</td><td>14</td></tr><tr><td>MiraData</td><td>140,877</td><td>3,683</td><td>17,806</td><td>119,388</td></tr><tr><td>MiraData-Clean</td><td>135,224</td><td>12,865</td><td>5,740</td><td>116,619</td></tr><tr><td>MultiCamVideo</td><td>123,117</td><td>89,587</td><td>5,369</td><td>28,161</td></tr><tr><td>OmniWorld</td><td>19,632</td><td>3,773</td><td>13,552</td><td>2,307</td></tr><tr><td>RealCam</td><td>45,697</td><td>16,154</td><td>16,074</td><td>13,469</td></tr><tr><td>Sekai-Game</td><td>2,550</td><td>537</td><td>1,410</td><td>603</td></tr><tr><td>Sekai-Walking</td><td>22,990</td><td>6,054</td><td>12,976</td><td>3,960</td></tr><tr><td>Sekai-Walking-Clean</td><td>109,248</td><td>46,831</td><td>33,660</td><td>28,757</td></tr><tr><td>SpatialVID</td><td>365,345</td><td>100,815</td><td>127,180</td><td>137,350</td></tr><tr><td>SpatialVID-Clean</td><td>298,514</td><td>133,149</td><td>73,450</td><td>91,915</td></tr><tr><td>Total</td><td>1,425,694</td><td>471,798</td><td>404,795</td><td>549,101</td></tr></table>

Figure 5 | Distribution of canonical clips by clip length and quality tier.  
![](images/768a25995882701174d18f099551e0a710263e6dd33210762123ab8922753f35.jpg)

## 5.7. Final Corpus and Temporal Views

Table 4 and Figure 5 summarize the released corpus from complementary source and temporal perspectives. The owner-level accounting in Table 4 shows that 876k of the 1.4M canonical clips are retained: 471k in the high tier and 404k in xhigh. The remaining 549k fully processed clips are published in the rejected partition. In physical storage, the complete corpus comprises 29k shards and approximately 25.85 TB.

Figure 5 instead aggregates the same corpus by raw clip length, making the tier composition of each temporal range directly visible. The majority of clips fall in the 153–956-frame interval, which contains 858k clips overall and 553k kept clips. All 78k clips shorter than 81 frames belong to the rejected partition, whereas the long tail of clips with at least 957 frames contains 82k kept clips.

From these physical clips, we construct model-ready temporal views for short- and long-horizon generation. The release separates the annotated source corpus from temporal windowing, datamixture policies, and backbone-specific latent representations, with each layer indexed reproducibly. This design allows researchers to adapt temporal coverage, filtering criteria, source mixtures, and backbone-specific requirements without rebuilding the underlying corpus.

## 6. SolarWM Model Family

SolarWM is organized as one interactive world-model framework rather than as four isolated model releases. The framework fixes a common data recipe, a shared processing and annotation contract, a camera-geometry interface, and a consistent training and inference workflow. Diferent videogeneration backbones are then instantiated under this interface so that their behavior can be studied under comparable data and system conditions. The shared interface does not erase backbone-specific semantics: each route retains the temporal representation, attention layout, conditioning details, and native optimization objective required by its base model.

The resulting family contains four interactive video world models: SolarWM-wan2.2-5B, SolarWMwan2.2-14B, SolarWM-ltx-2.5-22B, and SolarWM-minimax-h3-33B. They cover four independent video-generation routes—Wan 2.2-5B, Wan 2.2-14B, LTX-2.5, and MiniMax-H3—under one SolarWM contract. This organization makes two comparisons possible at the same time. The common data and camera interface isolates diferences between backbones, while the route-specific adapters preserve the mechanisms that make each backbone usable for interactive generation.

Table 5 | Backbone-level diferences within the SolarWM model family. All routes share the same data, camera, and training/inference interfaces while retaining backbone-specific parameter and conditioning contracts.
<table><tr><td>Model</td><td>Retained Components</td><td>Image Conditioning</td><td>Audio Handling</td><td>VAE DS (T / S)</td></tr><tr><td>SolarWM- wan2.2-5B</td><td>5B</td><td>Native TI2V latent path; no separate y</td><td>None</td><td>4×/16x</td></tr><tr><td>SolarWM- wan2.2-14B</td><td>High-noise expert; full-timestep route</td><td>Official y: 4-channel mask + 16-channel image latent</td><td>None</td><td>4×/8x</td></tr><tr><td>SolarWM- ltx-2.5-22B</td><td>14.74B video path; 1.61B Gemma connector frozen</td><td>Native first-frame latent</td><td>3.69B audio stream and 2.58B AV cross-attention removed</td><td>8×/32x</td></tr><tr><td>SolarWM- minimax-h3-33B</td><td>33B Omni Transformer</td><td>Qwen image/caption rows + VisualVAE first-image anchor</td><td>Audio target rows retained; audio loss disabled</td><td>17n + 5 → 5n + 2 /16x</td></tr></table>

## 6.1. Unified Camera Conditioning with Fused-PRoPE

Camera control is a shared requirement of the SolarWM family. Each camera trajectory is converted into a geometric condition aligned with the video frames and injected into the video attention computation. This condition contains the camera poses, intrinsics, frame alignment, and the associated normalization transforms. A route adapter maps this common condition into the representation expected by each backbone while preserving the same underlying camera motion.

SolarWM uses fused projective rotary positional embeddings (fused-PRoPE), as in MosaicMem [Yu et al., 2026]. At a high level, the procedure is the same for all four models. The backbone first applies its native video RoPE to represent the temporal and spatial positions of video tokens. Camera poses and intrinsics then determine projective rotations that are applied to the query, key, and value tensors. A single self-attention operation follows, after which the matching output transform is applied before the native output projection. Camera motion is therefore introduced through the attention computation itself, rather than by appending an unrelated condition token or a separate camera branch. This procedure is shared across all four routes; backbone-specific details are deferred to their respective paragraphs below.

## 6.2. Comparison of the Four SolarWM Models

The four model routes and their backbone-specific parameter, conditioning, and latent-compression contracts are summarized in Table 5.

The common SolarWM interface is designed to preserve, rather than replace, the capabilities of each video-generation base. We therefore retain each backbone’s native temporal grid, text representation, attention mask, training objective, and other settings wherever they are essential to its behavior, while exposing them through the same data and camera contracts. The following paragraphs summarize these backbone-specific choices before we introduce the shared training methodology.

SolarWM-wan2.2-5B is the TI2V member of the family. The model represents a clip with a 48-channel Wan2.2 VAE latent and does not take a separate image condition tensor �. SolarWM keeps this native interface: the first latent is used as the clean image anchor and is excluded from the video loss, while the remaining latents are denoised as video targets under the shared camera condition.

SolarWM-wan2.2-14B starts from the Wan 2.2 I2V-A14B model, whose released checkpoint contains separate high-noise and low-noise experts. SolarWM uses the high-noise expert for initialization and trains it as one dense model over all timesteps; the two-expert routing rule and its noise boundary are not retained. Its image interface is also distinct from the 5B route. We construct the oficial � by encoding a video whose first frame is the input image and whose later frames are zero, then concatenate a four-channel first-frame mask and the resulting 16-channel image latent to the noisy 16-channel target. Thus � is an explicit input, and the first target latent remains noisy and supervised.

SolarWM-ltx-2.5-22B is released as an audio-visual checkpoint, but SolarWM uses it as a videoonly backbone. The audited checkpoint assigns 3.7B parameters to the audio stream and 2.6B to bidirectional audio–video cross-attention; both components are removed. SolarWM retains the 13.1B video core and loads the 1.6B Gemma video connector frozen, giving a 14.7B retained video route driven by cached Gemma4 features. The route keeps LTX’s native first-frame latent convention and rectified-flow objective, and inserts the shared camera condition only in video self-attention.

SolarWM-minimax-h3-33B keeps MiniMax-H3’s multimodal packing rather than turning H3 into a video-only DiT. Each packed sequence contains Qwen caption/image conditions, a one-frame VisualVAE image anchor, audio rows, and target-video rows. The audio rows are retained because they are small; they are filled with an oficial AudioVAE encoding of silence and assigned zero audio loss, so only target-video rows are supervised. The Qwen conditioner, VisualVAE, and AudioVAE are frozen, while the H3 Omni Transformer is trained with its native temporal, masking, and camera semantics. Its camera injection also preserves the pretrained head split: dimensions [0:96) retain the native content representation and MM-RoPE, whereas only dimensions [96:128) receive camera-relative-pose PRoPE on the VisualVAE anchor and target-video rows.

## 7. Experiments

We evaluate two stages of the SolarWM training pipeline. We first examine the bidirectional pretrained models across all four backbone routes on 10-second OOD generations initialized from images created with GPT Image 2 or Krea, testing how consistently the shared data and camera contract transfers across heterogeneous video generators and unseen visual domains. We then examine distilled causal generation under both in-domain and OOD settings, spanning 10-second, minute-scale, and hour-scale horizons. Together, these settings expose model behavior across heterogeneous backbones, input domains, training stages, and rollout horizons.

## 7.1. Configurations

We use checkpoints produced by the training pipeline in Section 4. All reported SolarWM causal results are generated at 16 fps with four sampling steps and no attention sink. Every sequence is initialized from one image and conditioned on a text description and a frame-aligned camera trajectory. Bidirectional generation produces 10-second clips with bidirectional temporal attention. Causal generation instead conditions each new unit on model-generated history to reach the reported 10-second, minute-scale, and hour-scale horizons. The scene prompt remains fixed throughout each sequence, while the prescribed camera trajectory provides the only time-varying external control. We use the native latent representation, conditioning interface, and sampling objective of each backbone, as described in Section 6.

## 7.2. Bidirectional Pretrained Models

We begin with the model family after bidirectional pretraining and before causal adaptation or distillation. Each model receives an OOD first frame created with GPT Image 2 or Krea, a fixed scene caption, and a frame-aligned camera trajectory, and generates the remaining frames. Evaluating this

![](images/47fc243ee91fdb80c8d875cdadc9dd8cf4425e07d4a9df06b323f0b67250c171.jpg)  
Figure 6 | OOD qualitative results of the bidirectional pretrained model family. Each backbone route generates a 10-second video from an OOD first frame and a prescribed camera trajectory.

stage separately tests whether the unified data and camera-conditioning contract generalizes beyond the training domains while preserving the visual prior of each native backbone.

As shown in Figure 6, the four backbone routes are evaluated on representative OOD inputs under translation, rotation, and mixed 6-DoF camera paths.

## 7.3. Distilled Causal Generation

We next examine distilled causal generation across familiar held-out inputs, externally synthesized initial images, and increasingly long autoregressive horizons.

## 7.3.1. In-Domain Validation Results

We draw the in-domain evaluation examples from a held-out validation pool. The pool is in-domain in the operational sense that its samples are drawn from the same source families and processed by the same canonical pipeline as the training mixture, while no validation sample is used for optimization.

For the selected examples, we report two complementary settings. First, Figure 7 presents 10- second third-person rollouts in which a visible subject is observed from an external viewpoint. These examples test whether the model can follow the prescribed camera trajectory while maintaining the

![](images/8be0f1882dd83fa9b891e3deb48b68788a10ff49d3dd56ec2c4dcb944abd897d.jpg)  
Figure 7 | In-domain third-person qualitative results generated by the SolarWM-wan2.2-5B-fast causal student. Each sequence is initialized from a real first frame and follows a prescribed camera trajectory. All subsequent frames are generated autoregressively, testing subject consistency and camera response from an external viewpoint.

![](images/afd336aed3e14254efc7f3817fd79ebaa963843153e0268e948edaf71f491ab8.jpg)  
Figure 8 | In-domain minute-scale qualitative results generated by the SolarWM-wan2.2-5B-fast causal student. Each uninterrupted sequence is initialized from its real first frame and follows a given camera trajectory; all subsequent frames are generated autoregressively.

subject’s appearance and its spatial relationship to the surrounding scene. Second, Figure 8 presents uninterrupted minute-scale rollouts spanning representative real-world, synthetic, and game-domain scenes under translation, rotation, and mixed 6-DoF camera paths.

## 7.3.2. Out-of-Domain Initialization

A useful interactive world model should not require every explorable world to originate from its training collection. We therefore construct a separate stress test in which each first frame is synthesized with GPT Image 2 or Krea. The prompts deliberately cover visual conditions uncommon in the curated corpus, such as stylized illustration, surreal geometry, unusual material combinations, dramatic lighting, and imaginative indoor or outdoor environments.

For every initial image, we apply several camera trajectories, including forward motion, lateral translation, turning, orbit-like motion, and compositions of translation and rotation. Using multiple trajectories from the same image separates robustness to appearance shift from responsiveness to the control signal. We first report 10-second OOD rollouts in this subsection; the minute-scale OOD stress

![](images/192b016346678038587d2138455dcac2e88c694f46f1ed63d4e4d871006f8f5b.jpg)  
Figure 9 | OOD generalization from externally synthesized initial images, generated by the SolarWM-wan2.2-5B-fast causal student. Initial frames are paired with independently specified camera trajectories. Only the first image is externally provided; every subsequent frame in each 10-second rollout is generated autoregressively. The examples span photorealistic, stylized, and deliberately unusual scenes outside the released validation set.

test is presented in Section 7.3.3.

As illustrated in Figure 9, the model turns a broad range of externally synthesized still images into navigable video while largely retaining the initial style, palette, and dominant scene elements. The examples are particularly informative when camera motion reveals regions that were not visible in the input: all newly exposed content must be inferred by the world model, rather than copied from the image generator.

## 7.3.3. Long Video Generation

We evaluate continuous autoregressive generation at minute-scale and hour-scale horizons. Each long video starts from a single image, keeps the scene prompt fixed, and follows a predetermined camera trajectory throughout the complete run. We do not restart from the input image, inject reference frames, independently generate and splice short clips, or use an attention sink.

Minute-scale rollouts. We evaluate uninterrupted minute-scale generation from synthesized first frames under camera trajectories that combine forward motion, turning, lateral translation, and pauses. These compound trajectories test both sustained response to a control and transitions between successive camera commands without resetting the model.

As shown in Figure 10, the model navigates five visually distinct scenes while preserving their principal layout and appearance. Across the sampled timelines, newly revealed stairs, rooms, vegetation, terraces, clifs, and waterways remain compatible with the initial observation, and the views continue to evolve with the prescribed camera motion throughout the minute-scale rollout. These results demonstrate OOD appearance generalization and sustained camera control well beyond the 5-second sequences used during training.

![](images/c3d3f879766809eefbc45656928d996bb3d6037280dbd2e5438721e972a4cb67.jpg)  
Figure 10 | Minute-scale OOD autoregressive rollouts generated by the SolarWM-wan2.2-5B-fast causal student. Frames are sampled from uninterrupted generations initialized by a single OOD image. The scene prompt remains fixed for the entire run.

![](images/d1931f57ebc0dc49d24b1c696ce69416f4bbd508cf7bf9a946a5c35190ca48fa.jpg)  
Figure 11 | Hour-scale world rollout generated by the SolarWM-wan2.2-5B-fast causal student. Sparse frames are sampled throughout a single uninterrupted autoregressive session initialized with a real first frame from the held-out validation pool, spanning the initial observation to the 60-minute endpoint. The scene prompt remains fixed and the model follows a predetermined camera trajectory.

Hour-scale rollouts. We further evaluate uninterrupted hour-scale generation from real first frames selected from the held-out validation pool. The model is trained only on 5-second sequences. During evaluation, it follows the given camera trajectory. Consequently, all later frames are generated autoregressively by repeatedly conditioning on model-generated history, making the generation substantially longer than the training horizon.

As shown in Figure 11, the model sustains diverse scenes through uninterrupted hour-scale rollouts. Across the sampled timeline, the generated views continue to respond to camera control while retaining the principal scene layout and visual identity established by the initial observation. The 60-minute endpoints remain recognizable and visually coherent.

## 8. Conclusion

We presented SolarWM, an open foundation for building interactive video world models across heterogeneous data sources and video backbones. SolarWM combines a reconfigurable multi-source data engine with backbone-native adaptation and a unified three-stage training recipe, yielding four camera-controllable models spanning 5B–33B parameters. The resulting causal models support real-time interaction and continuous hour-scale rollouts after training on only 5-second sequences. By releasing the data, processing pipeline, training recipes, model weights, and training & inference framework, we hope to provide a reproducible and extensible basis for future world-model research.

## References

Alaya Lab. Alayaworld. Oficial code repository, 2026. URL https://github.com/AlayaLab/Al ayaWorld. Repository state inspected at commit ce3dad3cefc3fad2599877bf1b8ef82eb9aec9a2 on August 18, 2026.

AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, Ruicong Liu, Xiaojie Xu, Xuangeng Chu, Zhen Li, Zhengyuan Lin, Zhixiang Wang, Zian Meng, and Zihui Gao. Alayaworld: Interactive long-horizon world modeling – full technical report (v1.1). arXiv preprint arXiv:2608.13492, 2026. URL https://arxiv.org/abs/2608.13492.

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Difusion for world modeling: Visual details matter in atari. arXiv preprint arXiv:2405.12399, 2024a. URL https://arxiv.org/abs/2405.12399.

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diamond: Difusion for world modeling. Oficial code repository, 2024b. URL https://github.com/eloialonso/diamond. Repository state inspected at commit 5bcd1599755b4f2fae8e5e079e02f0728e174965 on August 18, 2026.

Amap CVLab. Abot-world-0. Oficial code, model, and dataset repository, 2026. URL http s://github.com/amap-cvlab/ABot-World. Repository state inspected at commit d91c2ca43643bda5ed4b8c1831fa85a41fe3d1bb on August 18, 2026.

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. URL https://arxiv.org/abs/25 03.11647.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE/CVF International Conference on Computer Vision, 2021. URL https://arxiv.org/abs/2104.00650.

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jef Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments. arXiv preprint arXiv:2402.15391, 2024. URL https://arxi v.org/abs/2402.15391.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion forcing: Next-token prediction meets full-sequence difusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Decart and Etched. Open oasis. Oficial code repository, 2024. URL h t t p s : / / g i t h u b . c o m / e t c h e d- a i / o p e n - o a s is. Repository state inspected at commit f59deef2c019c212bd0c5a3a5b986a51f3701847 on August 18, 2026.

DreamX Team. Dreamx-world. Oficial code and model repository, 2026. URL https: //github.com/AMAP- ML/DreamX- World. Repository state inspected at commit a1f4c6e5e45600718e5236955f2e0702e53fc275 on August 18, 2026.

DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, Geng Li, Jifan Li, Ruimin Lin, Qingfeng Shi, Bingze Song, Lei Sun, Jing Tang, Ruitian Tian, Jun Wang, Jiahong Wu, Pengfei Zhang, Shen Zhang, and Jiashu Zhu. Dreamx-world 1.0: A general-purpose interactive world model. arXiv preprint arXiv:2606.16993, 2026. URL https://arxiv.org/abs/2606.16993.

ForgeWM Team. Forgewm. Oficial code repository, 2026. URL https://github.com/asdfo123/ ForgeWM. Repository state inspected at commit ca94df4c533b3ae44f0bc95ae4bbd9734885cd72 on August 18, 2026.

Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, and Hao Ouyang. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026. URL https://arxiv.org/abs/2607.075 34.

Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. Anyflow: Any-step video difusion model with on-policy flow map distillation, 2026. URL https: //arxiv.org/abs/2605.13724.

Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025. URL https://arxiv.org/abs/2504.08388.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. Ltx-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Cyrus Wu, Wei Li, Xuchen Song, Yang Liu, Eric Li, and Yahui Zhou. Matrix-game 2.0: An open-source, real-time, and streaming

interactive world model. arXiv preprint arXiv:2508.13009, 2025. URL https://arxiv.org/ab s/2508.13009.

Bin Hu, Yanwen Ma, Jiehui Huang, Ziliang Zhang, Haoning Wu, Ruicheng Zhang, Yaokun Li, Zijun Wang, Yuechen Zhang, Chun-Mei Tseng, et al. Physeditworld: A large-scale dataset toward physics-editable world models. arXiv preprint arXiv:2606.26694, 2026.

Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft, 2025a. URL https://arxiv.org/abs/2510.03198.

Junchao Huang, Xinting Hu, Shaoshuai Shi, Zhuotao Tian, and Li Jiang. Edit360: 2d image edits to 3d assets from any angle. In ICCV, 2025b.

Junchao Huang, Ziyang Ye, Xinting Hu, Tianyu He, Guiyu Zhang, Shaoshuai Shi, Jiang Bian, and Li Jiang. Live: Long-horizon interactive video world modeling, 2026a. URL https://arxiv.or g/abs/2602.03747.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026b.

Fan Jiang, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu, Chiyu Wang, Yunpeng Zhang, Wenlin Liu, Yun Wang, Xue Zheng, Rui Sun, Junfeng Ni, Hongyu Pan, Zhongxu Sun, Fei Yu, Zengye Ge, Mengmeng Du, Nianfei Fan, Mingchao Sun, Yu Liu, Yongchang, Yanqing Zhu, Jiahang Wang, Ning Ying, Yuze Xuan, Di Yang, Zhicheng Liu, Zhe Gao, Tingbing Xu, Jiacheng Sui, Wenjin Yang, Junnan Lai, Shufeng Liu, Yuan Liu, Zheng Zhou, Yingliang Peng, Dawei Cao, Kaifeng Sheng, Yuxiang Cai, Fei Lu, Mu Xu, and Ning Guo. Abot-world-0: Infinite interactive world rollout on a single desktop gpu. arXiv preprint arXiv:2607.19191, 2026. URL https://arxiv.org/abs/2607.19191.

Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. arXiv preprint arXiv:2407.06358, 2024. URL https://arxiv.org/abs/2407.06358.

Kling Team. Multicamvideo-dataset. Oficial dataset release, 2026. URL https://huggingface. co/datasets/KlingTeam/MultiCamVideo-Dataset. Accessed August 19, 2026.

Xianghao Kong, Zeyu Zhang, Yuwei Guo, Zhuoran Zhao, Songchun Zhang, and Anyi Rao. Composing concepts from images and videos via concept-prompt binding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14800–14810, June 2026.

Xinye Li, Lingshuai Lin, Lei Wang, Liuzhou Zhang, Jialin Cui, Qingshan Li, Guanchu Wang, Qingbin Liu, Xi Chen, Jiang Bian, and Wai Lam. Forgewm: Progressive causal training for few-step action-conditioned video world models. arXiv preprint arXiv:2608.14022, 2026. URL https: //arxiv.org/abs/2608.14022.

Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Zhixiang Wang, Yuwei Wu, Tong He, Jiangmiao Pang, Yu Qiao, Yunde Jia, and Kaipeng Zhang. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025. URL https://arxiv.org/abs/25 06.15675.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. arXiv preprint arXiv:2312.16256, 2023. URL https://arxiv.org/abs/2312.16256.

LynnReal AI. Biwm: Bidirectional autoregressive video world models. Oficial code repository, 2026. URL https://github.com/LynnReal-AI/BiWM. Repository state inspected at commit 626a0714ab139f74ee76826f411a9f6076096add on August 18, 2026.

Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025. URL https://arxiv.org/abs/2507.17744.

Microsoft Research. Mineworld. Oficial code repository, 2026. URL h t t p s : / / g i t h u b . c o m / m i c r o s o f t / m i n e w o r ld. Repository state inspected at commit 9f49efbcc68a5f0cf60e78f0abc566eac409282f on August 18, 2026.

MiniMax. Minimax h3: An open model breaking the boundaries between tasks and modalities. Oficial model release article, 2026. URL https://www.minimax.io/blog/minimax-h3. Published July 31, 2026.

NVIDIA Research. Sana-wm. Oficial code, model, and example-data repository, 2026. URL h t t p s : / / g i t h u b . c o m / N V l a b s / S a na. Repository state inspected at commit 5498e5b2070ef8c808ee7eab7f7c7707b53cd8ea on August 18, 2026.

Jack Parker-Holder and Shlomi Fruchter. Genie 3: A new frontier for world models. Google DeepMind technical article, 2025. URL https://deepmind.google/discover/blog/genie-3-a-new -frontier-for-world-models/. Published August 5, 2025; accessed August 18, 2026.

Riemann Dynamics. Matrix-game 3.5: Enhancing real-time streaming interactive world models with patch memory. Technical report, Riemann Dynamics, July 2026a. URL https://matrix-gam e-v3-5.github.io/paper/Matrix-Game-3.5.pdf. Technical report.

Riemann Dynamics. Matrix-game 3.5. Oficial code and model repository, 2026b. URL https: //github.com/Riemann-Dynamics/Matrix-Game-3.5. Repository state inspected at commit fa6d2b628ac9b0f1657dc24689536d74bfeb51da on August 18, 2026.

Robbyant Team. Lingbot-world 2.0. Oficial code and model repository, 2026a. URL https: //github.com/Robbyant/lingbot-world-v2. Repository state inspected at commit 2648877f763a06cc743bcd919936da4d25f12e7b on August 18, 2026.

Robbyant Team. Lingbot-world. Oficial code and model repository, 2026b. URL https: //github.com/Robbyant/lingbot- world. Repository state inspected at commit a43bec7f8091c83e9b30b16b912f6fc906236fa6 on August 18, 2026.

Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026. URL https://arxiv.org/abs/2601.20540.

Shaohao Rui, Xiaofeng Mao, Zhanyu Zhang, Peijia Lin, Yansong Zhu, Yibo Zhang, Haibin Wan, and Weijie Ma. Biwm: Advancing open-source interactive video world models with bidirectional autoregression. arXiv preprint arXiv:2606.10135, 2026. URL https://arxiv.org/abs/2606 .10135.

ShengShu AI MIN Lab. minwm: Full-stack open-source video world model framework. Oficial code repository, 2026. URL https://github.com/shengshu-ai/minWM. Repository state inspected at commit df522a26cd4409d3e3e8f269cc98eac069b5df47 on August 18, 2026.

Skywork AI. Matrix-game. Oficial code repository, 2026. URL h t t p s : / / g i t h u b . c o m / S k y w o r k A I / M a t r i x- G a m e. Repository state inspected at commit 71c3cd7f741311f8100f6cf9cde942b6c1378d11 on August 18, 2026.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025. URL https://arxiv.org/abs/2512.14614.

Tencent Hunyuan. Hy-world 1.5 (worldplay). Oficial code repository, 2026. URL https:// github.com/Tencent-Hunyuan/HY-WorldPlay. Repository state inspected at commit 1588e1336e842b03b0a7860c654ebd7c46bb065e on August 18, 2026.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Difusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024. URL https://arxiv.org/abs/2408.14837.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, Xiao-Xiao Long, Hao Zhu, Zhaoxiang Zhang, Xun Cao, and Yao Yao. Spatialvid: A large-scale video dataset with spatial annotations. arXiv preprint arXiv:2509.09676, 2025. URL https://arxiv.org/abs/2509.09676.

Xinyu Wang, Chongbo Zhao, Fangneng Zhan, and Yue Ma. Liveedit: Towards real-time difusion-based streaming video editing. arXiv preprint arXiv:2606.26740, 2026a.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2307.06942.

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026b. URL https://arxiv.org/abs/2604.08995.

Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. Mind: Benchmarking memory consistency and action control in world models. arXiv preprint arXiv:2602.08025, 2026. URL https://arxiv.org/abs/2602 .08025.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6613–6623. IEEE, 2024.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22963–22974. IEEE, 2025.

Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Sri Siddarth Chakaravarthy P, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, and Animesh Garg. Mosaicmem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026. URL https://arxi v.org/abs/2603.17117.

Yume Team. Yume: An interactive world generation model. Oficial code, model, and data-processing repository, 2026. URL https://github.com/stdstu12/YUME. Repository state inspected at commit 111c3fab7fb020d1e261a68be6ec78a3fecc8d5b on August 18, 2026.

Guiyu Zhang, Yabo Chen, Xunzhi Xiang, Junchao Huang, Zhongyu Wang, and Li Jiang. Symphomotion: Joint control of camera motion and object dynamics for coherent video generation. arXiv preprint arXiv:2604.03723, 2026a.

Ruicheng Zhang, Guangyu Chen, Zunnan Xu, Zihao Liu, Zhizhou Zhong, Mingyang Zhang, Jun Zhou, and Xiu Li. Robostereo: Dual-tower 4d embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639, 2026b.

Ruicheng Zhang, Kaixi Cong, Jun Zhou, Zhizhou Zhong, Zunnan Xu, Shuiyang Mao, Wei Liu, and Xiu Li. Kvpo: Ode-native grpo for autoregressive video alignment via kv semantic exploration. arXiv preprint arXiv:2605.14278, 2026c.

Songchun Zhang, Zeyue Xue, Siming Fu, Jie Huang, Xianghao Kong, Y Ma, Haoyang Huang, Nan Duan, and Anyi Rao. Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models, 2026d. URL https://arxiv.org/abs/2603.17051.

Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, and Yahui Zhou. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025. URL https://arxiv.org/abs/2506.18701.

Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, and Jun Zhu. minwm: A full-stack open-source framework for real-time interactive video world models. arXiv preprint arXiv:2605.30263, 2026a. URL https://arxiv.org/abs/2605.30263.

Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive difusion distillation for real-time interactive video generation, 2026b. URL https://arxiv.org/abs/2605.15141.

Guangcong Zheng, Teng Li, Xianpan Zhou, and Xi Li. Realcam-vid: High-resolution video dataset with dynamic scenes and metric-scale camera movements. arXiv preprint arXiv:2504.08212, 2025. URL https://arxiv.org/abs/2504.08212.

Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, Mingyu Liu, Dingning Liu, Jiange Yang, Zhoujie Fu, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Kaipeng Zhang, and Tong He. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025. URL https://arxiv.org/abs/2509.12201.

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. Sana-wm: Eficient minute-scale world modeling with hybrid linear difusion transformer. arXiv preprint arXiv:2605.15178, 2026a. URL https://arxiv.org/abs/2605.15178.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive difusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026b.

Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, Haoran Li, Haoyang Huang, Zeyue Xue, and Nan Duan. Self gradient forcing: Native long video extrapolation, 2026.