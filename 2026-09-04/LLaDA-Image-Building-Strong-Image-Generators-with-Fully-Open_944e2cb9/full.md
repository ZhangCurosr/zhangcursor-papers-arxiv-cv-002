# LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes

Chuyan Chen, Haoxing Chen<sup>∗</sup>, Kun Chen<sup>∗</sup>, Zhenglin Cheng<sup>♦</sup>, Long Cui, Ruishan Fang, Zhangxuan Gu, Zhicheng Huang<sup>∗</sup>, Zhenzhong Lan<sup>†</sup>, Yuanting Lei, Haoquan Li, Jianguo Li<sup>†</sup>, Rongchuan Li, Sidu Li, Tao Lin<sup>†</sup>, Deyuan Liu, Jiacheng Liu<sup>♦</sup>, Lin Liu, Yuxuan Lou, Zhisheng Lu, Yuxin Ma, Shuheng Shen, Peng Sun<sup>♦</sup>, Chaoyang Wang, Hongjun Wang, Xiaomei Wang<sup>∗</sup>, Yongxin Wang<sup>∗</sup>, Chengzhang Wu, Hongru Wu, Jun Xie<sup>♦</sup>

AGI Research Center, Inclusion AI

## Abstract

We introduce LLaDA-Image, a unified framework that pairs a 6B Diffusion Transformer (DiT) trained from scratch with a frozen vision-language understanding module built on the LLaDA2.0-Mini diffusion language model backbone. Instead of relying heavily on paired image–text data from the beginning, we first build a strong visual generative prior through image-only pre-training and mid-training. The generation pipeline comprises 220M samples, 98% of which are real images. For efficient and scalable optimization, we use parameter-free RMSNorm throughout the DiT together with the Muon optimizer. The resulting unified model produces highly photorealistic images while accurately following fine-grained editing instructions. We further distill LLaDA-Image into LLaDA-Image Turbo, enabling fast inference in 2–4 sampling steps. On Qwen-Image-Bench, LLaDA-Image achieves overall scores of 53.53 and 53.38 on the English and Chinese tracks, respectively, setting a new state-of-the-art among open-source models on both tracks. To support further research on capable and efficient generative models, we release our model weights, training code, and detailed recipes.

GitHub https://github.com/inclusionAI/LLaDA-Image

0 HuggingFace https://huggingface.co/collections/inclusionAI/llada-image

Scope. This report asks a practical systems question: what does it take to train a strong image generator from scratch with a fully open recipe? We trace the complete path from image-only visual-prior learning through language alignment and joint generation–editing to TwinFlow distillation. For each stage, we document the data construction, model interfaces, optimization choices, training schedule, and evaluation used to produce the released checkpoints. Our goal is to make the resulting design trade-offs inspectable and the recipe reusable. We do not claim that every component is universally optimal, nor do we attempt an exhaustive comparison of all architectures or data mixtures.

![](images/05e9954d7d1699716adc8680ee86579053d0228a861506aae7d51337ed3ea2d3.jpg)  
Figure 1: Qwen-Image-Bench Results. LLaDA-Image achieves open-source SOTA on both English and Chinese tracks; models are ordered by their two-track average.

# Real or Generated? A Reader Challenge

Which images, if any, are real photographs? Inspect the details, make your selections, and lock in your answer before turning the page.

![](images/bbb665d4bde17dfd09ffbe114c6aa301c7d03a5111bea09c945dc90a86a6404a.jpg)  
Your task: identify every real image in the grid—if there are any. The answer appears on the next page.

# The Reveal: None of Them Are Real

Every candidate in the challenge was produced by LLaDA-Image; no real images were included.

Answer: every candidate was generated. The apparent photographs, rendered text, and stylized scenes were all synthesized by the same unified model.

![](images/d4f1d9c1d6a75b8845f20dc9361ca7ac3050cae7b0900836d289b2c10bc609f1.jpg)  
Disclosure. All results displayed on these two pages are generated outputs, spanning text-to-image synthesis, bilingual text rendering. This informal reader challenge is a qualitative demonstration, not a controlled perceptual study.

## Cross-Model Qualitative Comparison

张阴天海边小镇中的旅行女性人像摄影  
用 偏远 远一些的中景构图。一位二十 多岁 的  
轻白人女性沿着靠海的人行道行走，两只在 风 衣口 袋 里 ，人 物并没有正对 镜 头在 走 动过 程 中 自 然 侧过脸。 她 留 着 浅<sup>金</sup><sub>发</sub> 的 <sup>发，</sup><sub>乱，</sub> 海 风 将 大 量 发 丝 向<sup>型</sup><sub>脸</sub> 不呈现 精 心整理的波浪。 椭圆，额头较宽，颧骨轻微  
显 下 。 眉 毛 <sup>为</sup><sub>色</sub> 自 然 浅 棕 色 毛不浓密。 密。 <sup>眼睛呈灰蓝</sup><sub>方。鼻梁较</sub> 中等 小， 视  
向道路前 高， 尖具 有真 实嘴唇 唇偏 薄 嘴 角 自 然 闭 合 。皮肤 白 皙  
<sup>雀 ，</sup>持自然纹理，并非光滑陶瓷质感<sub>色棉质风衣、灰色针织衫和浅色</sub> 斑 自然理，光滑陶质 眼 下存在 淡淡 青 灰 阴 影 ，皮 感 。她穿深 肤色 牛 仔裤。 。景是灰蓝色海面、石墙、低矮住宅、  
泊的小渔船和被风吹动的草丛，阴天 天  
坦而柔和，人物与环境都有清晰空间关系写

Input Prompt

![](images/ce63682f8c9c4b056a6f44d68fe36c6eef04d69baa757979446513ac1dc2d48e.jpg)  
LLaDA-Image Turbo

![](images/cc06cd6bd48f0fac7a903ad5abfeb6bf86a902ae755f1fa904050185d7374265.jpg)  
Z-Image Turbo

![](images/aa8d152a537c167ec51b8781032f0272df729e61b2099934d3d6ea63242e976e.jpg)  
SenseNova U1.5 Preview

![](images/c9388614bba99719883b2b6ef7ee856d256f86b56de6a50d3cef075892235ea3.jpg)  
Qwen-Image 2512

![](images/e2feaec4b377bfa52b3f64b1fd3e784fc5be3a29616915bf1f669a3e81a694a8.jpg)  
GPT-Image 2

幅垂直构图的中国传统浅绛山水画轴，画面描绘春雨过后的江南山村。近景是一段湿润的石岸，几株柳树从画面左下方向上生 ，细 长 柳枝 自 然 下 垂 ，嫩叶以极淡的 青<sup>点染</sup><sub>青草</sub> 九 树 下散落着灰褐色石块和刚刚冒出。 真。 中 景是一座横跨溪流的石拱桥，桥布 数 间 白墙灰瓦的江南民居，屋檐和 墙被 薄 雾部分遮挡，溪水以大片留白与几道墨水纹表现。远处低矮山峦沿画面向上层递进，以湿润淡墨晕染，山腰笼 罩 着细腻气。右上角用略带行书意味的楷书分两列排题写，右列从上至下写<sub>“</sub>沾衣欲湿杏花雨<sub>”</sub>，左列从上至下写<sub>“</sub>吹面不寒杨柳风<sub>”</sub>，两列字距疏朗，排列整齐，墨色自然。纸本呈柔和的暖米黄色，可见细微纸纹、墨色 渗绫边，整体色调清淡雅致，具有真实传统中国画的湿润空气感。

Input Prompt

![](images/551711f6c25fac00ce10fc8b36466071fe935fe767b3d7cc5e155d60ecbc906f.jpg)  
LLaDA-Image Turbo

![](images/9520db785e84d75ac7233566130b5879f9d62db0ea73b60d6aa57c8b67b83330.jpg)  
Z-Image Turbo

![](images/cf049cbd5013220020140ef6cf1f18118b0d8cfe196a015886653521fa9d0ed7.jpg)  
SenseNova U1.5 Preview

![](images/cc90dadd9e0a5c06e1c55ba81430c338322579439dc391ba780262ca3c8243f6.jpg)  
Qwen-Image 2512

![](images/7cd70db7610c0d56a4b63c626a9b7998ed6ce4a755b97360ce648ffdfac84f37.jpg)  
GPT-Image 2

![](images/5be6bee7b96bcba007cb78082af70c2d296387c2249d52c11f6ad9547f968c3e.jpg)  
Boogu-Image 0.1 Turbo

![](images/8c3c149baf4dd2bd570a01dfc79713492040c1cb758ea1d2a7af1a26e2be40f3.jpg)  
Gemini 3.1 Flash Image

![](images/71ce74947585240d31e2327214453d2febee421eb213b028940f24cf25cf57b8.jpg)  
Boogu-Image 0.1 Turbo

![](images/b9d1b17482e57658f5da11010fe70a27a3898de0ca6681e1eed604bfd059b915.jpg)  
Gemini 3.1 Flash Image

# Instruction-Guided Image Editing

LLaDA-Image follows diverse editing instructions while preserving unedited content.

<sub>Case</sub> <sub>#1</sub> 把图片的风格改为吉卜力风格

![](images/918be5ae5f69057d0e2ce8934bd41b9f0b1e7a3b6231288b8cc01bdf6fab41e6.jpg)

![](images/3c0a25e73262753ecfe597d8e1edd0e306d0d9837580ad310e769febb3ca33bc.jpg)  
Case #2 <sup>把</sup>“Stay / fresh” <sup>改成</sup>“Keep / fresh”

![](images/a2eab8cf2e16367704600afc12d44960c78ca6c44ecd31744bb4912ee1d81248.jpg)

![](images/82885d0268e6bab22779dacaf4a716d73847fa1011919f6181c81d94f4d21f28.jpg)  
<sub>Case</sub> <sub>#3</sub> 把图中女性身上红色的包去掉

![](images/9706d1dbce9a6a3337190612e00c5c0c494162d896e614ddc754d3fadebb4dfe.jpg)

![](images/de8ce0999d26d7c7be4c4efc1094b904eb318319bb9572f63f5a3c8336511ea0.jpg)  
<sub>Case#4</sub> 对这个手绘图进行上色，其中小女孩的肤色为肉色，头发改成黑色，小熊为棕色

![](images/a676976f7bde2d494ea5b48c966e23bd5a8accd565caf8d78d5a17e6d8b6879c.jpg)

![](images/9520f62979d082d71389e935237f6c053090e4241abb96f1f274ca0983517353.jpg)  
<sub>Case</sub> <sub>#5</sub> 让她单手比一个<sub>“V”</sub>字手势，保持背景不变  
<sub>Case</sub> <sub>#6</sub> 把这个背景换成绿色的大草原，草原上还有蒙古包

![](images/252373479187b7d015b38a2955e9706f821789f933004b7c1b741631ba5e8332.jpg)

![](images/8652f1d93e3d8f6b4283ef30c570c52d56a1272a559d4241797acc810e1b0d47.jpg)  
<sub>Case</sub> <sub>#7</sub> 在这个地毯上去添加一个茶几

![](images/8525c5f9175d67367d42b09dd12eb6db675a4c770156982a0d414e2bfb958345.jpg)

![](images/1bf804d85e2bfce3d3e3bb385f3a567dcbd6a67742317f46b3e20fb5116c643e.jpg)  
<sub>Case#8</sub> 把这个蓝色的冰淇淋换成草莓味圣代

![](images/d5989d4bf293560656ad86c2873b3212c6e87de28b51a7b9a5ee04338b6350da.jpg)

![](images/df4d2acff2a867435406194e2981bd03c0d2a0b3166aa924547d35d1cad22c8c.jpg)

![](images/bfce744a3a9dba747b65b258d622838b16f06f4fc9c5b9961d64173d2b65b178.jpg)

![](images/702b25a38bed313bd2e6d930f719ab42c4f3b6ca853970456154fcc03e6326ef.jpg)

## Contents

Introduction 7   
2 Model Design 8   
2.1 dLLM-based Vision-Language Model 9   
2.2 Understanding-to-Generation Connector 9   
2.3 Single-stream Diffusion Transformer 10   
3 Data Collection and Preparation 10   
3.1 Data Composition . 10   
3.2 Data Processing 11   
Training Recipe 11   
4.1 CoT Supervised Fine-tuning . 12   
4.2 Image-only Pre-training 12   
4.3 Image-only Mid-training 14   
4.4 Supervised Fine-tuning . 15   
4.4.1 Text-to-Image Training 15   
4.4.2 Image Editing Training . 16   
4.4.3 Checkpoint Merging 16   
4.5 Few-step Distillation 16   
4.5.1 TwinFlow for Few-step Distillation . 16   
5 Experiments 17   
5.1 Text-to-Image Generation 17   
5.1.1 General Image Generation Performance 17   
5.2 Image Editing Generation . 22   
6 Conclusion and Future Directions 23   
A Aspect-Ratio Bucket Configurations 29   
B CoT Supervised Fine-tuning and Reinforcement Learning 30   
C Additional Qualitative Comparison 32   
D Image Captioning and Filtering 35   
E Prompts Used in The Report 38

## 1 Introduction

Image generation systems are evolving from specialized text-to-image models into general-purpose visual creation systems (Inclusion AI et al., 2026; Wu et al., 2025b; Deng et al., 2025). They must understand complex and multilingual instructions, synthesize photorealistic content, preserve reference information during editing, render text accurately, and remain usable under practical inference budgets. Although proprietary systems have made impressive progress, their data, model, and training recipes remain largely inaccessible. For open models, the goal is therefore not generation quality alone: a useful system must combine broad capabilities with an attainable data budget, stable large-scale training, and efficient deployment.

These requirements create linked bottlenecks across data, model design, and deployment. Prevailing training paradigms (Rombach et al., 2022; Esser et al., 2024) couple visual-prior learning with language alignment from the outset, requiring paired captions during the most compute-intensive stages. Yet captions are expensive and lossy; at low training resolutions, details mentioned in a caption may disappear after downsampling. Synthetic image–text pairs can accelerate early convergence, but synthetic-heavy mixtures may propagate artifacts and limit long-run realism. Meanwhile, a unified model must connect multimodal understanding to generation while preserving reference-image evidence for editing, and deployment requires compressing long diffusion trajectories without quality loss. Addressing these challenges together requires co-designing supervision, architecture, the training pipeline, and distillation.

$$
\begin{array} { r } { \mathrm { \rVert ~ L M G C ~ C T S F T } } \\  \mathrm { \rVert ~ U n d e r s t a n d i n g ~ \rVert ~ \left[ \begin{array} { l } { \mathrm { ~ I m g e < o m b y ~ P T ~ } } \\ { 2 5 6 ^ { 2 } \mathrm { ~ P i x e } \mathrm { B r e } } \end{array} \right] \sim \left[ \begin{array} { l } { \mathrm { ~ I m a g e < o m b y ~ M T ~ } } \\ { 5 1 2 ^ { 2 } \mathrm { ~ A R b u c k e } } \end{array} \right] \sim \left[ \begin{array} { l } { \mathrm { ~ I m s t o - i n a n g e ~ S F T } } \\ { 5 1 2 ^ { 2 } \cdots 1 0 2 4 ^ { 2 6 } } \end{array} \right] \sim \left[ \begin{array} { l } { \mathrm { ~ I m i t c ~ G e n - E d i t ~ } } \\ { \mathrm { ~ I m 2 t + I 2 1 ~ } } \\ { \mathrm { ~ I m 4 t i ~ } \cdots \mathrm { ~ I m ~ a g e ~ \pi ~ } } \end{array} \right] \sim \left[ \begin{array} { l } { \mathrm { ~ L \lambda a D \lambda A - i m a g e ~ } } \\ { \mathrm { ~ M u l t i * s t e } } \end{array} \right] \sim \left[ \begin{array} { l } { \mathrm { ~ \left[ L a D \lambda A - I m a g e ~ T u e t o ~ \right] ~ } } \\ { 2 \mathrm { ~ i \lambda a D \lambda _ 2 - i s e p s ~ } } \end{array} \right] } \end{array}
$$

Figure 2: Training and release path. CoT SFT prepares the frozen understanding backbone (dashed); the solid path trains the generator and yields the Base and TwinFlow-distilled Turbo checkpoints. PT and MT denote pre-training and mid-training, respectively.

We present LLaDA-Image, a 6B-parameter Diffusion Transformer trained from scratch, together with the complete model and training stack behind it. The system first learns its visual prior through image-only pre-training and mid-training, then progressively introduces language alignment, refinement, and joint generation–editing training. A dLLM-based understanding module and a single-stream DiT support understanding, generation, and editing within one framework, while TwinFlow produces the few-step LLaDA-Image Turbo variants. The principal characteristics of the system are summarized as follows:

• Data-efficient image-only pre-training and mid-training. We decouple visual-prior learning from language alignment and avoid large-scale paired supervision in the early training stages. A frozen dLLMbased vision-language model derives the condition from the same image region that the DiT learns to generate, ensuring condition–target compatibility without an external caption and enabling informative crops of high-resolution images to be resized only mildly for $2 5 6 ^ { 2 }$ training. The complete image-generation pipeline processes approximately 220M generation-training samples, with image-only training accounting for more than 90% of the total, keeping the overall data requirement moderate.

• Unified dLLM-based understanding, generation, and editing. A dLLM-based VLM jointly interprets text, visual context, and structured reasoning traces. A Residual Query Adapter and lightweight Transformer connector expose generation-relevant hidden states to a pure single-stream DiT. Text-to-image prompts and editing instructions share this semantic pathway, while SigLIP-VQ features and the clean VAE latent provide complementary semantic and pixel-level reference signals for editing. This design brings multimodal understanding, text-to-image generation, and reference-preserving editing together in one checkpoint.

• A stable, real-data-dominant progressive training pipeline. We progress from $2 5 6 ^ { 2 }$ image-only pretraining to aspect-ratio-bucketed mid-training, paired language alignment at $5 1 2 ^ { 2 }$ and $1 0 2 4 ^ { 2 }$ , targeted refinement on text-rich and portrait data, and finally joint generation–editing training. The real-image share remains above 70% throughout supervised fine-tuning. Although this real-data-dominant recipe can improve more slowly on early benchmarks than synthetic-heavy alternatives, it yields stronger realism at convergence and a higher long-horizon capability ceiling. Parameter-free RMSNorm throughout the DiT and the Muon optimizer further stabilize the full training pipeline.

• Efficient 2–4-step inference with LLaDA-Image Turbo. Building on distribution-matching distillation (Yin et al., 2024) and self-adversarial flow training, TwinFlow (Cheng et al., 2026) distills the original multi-step model into LLaDA-Image Turbo. The resulting variants require only 2–4 sampling steps.

![](images/72190ef0bd4c2d10e6dbf8f9e6aa00d1f7082947a99786ade06eec44298e82bd.jpg)  
Figure 3: Overview of the LLaDA-Image architecture. Learnable query tokens aggregate information from the input tokens through cross-attention in the RQA. The resulting query representations are concatenated with the original inputs and encoded by LLaDA 2.0 Mini, after which the connector maps the hidden states into the DiT conditioning space. The FLUX.2 VAE supplies image latents, while editing additionally conditions the generator on SigLIP-VQ features and clean VAE latents from the reference image. All modal tokens are concatenated and processed by the single-stream DiT to predict the flow-matching velocity.

• Open weights, code, and training recipes. We release the LLaDA-Image and LLaDA-Image Turbo checkpoints, training and inference code, and detailed recipes for data construction, progressive training, unified generation and editing, and TwinFlow distillation.

Taken together, these components yield two complementary deployment profiles within a single model family. The original LLaDA-Image checkpoint serves as the full multi-step model for high-fidelity generation and unified editing, whereas LLaDA-Image Turbo is its distilled 2–4-step deployment variant for substantially lower inference cost. Evaluations on Qwen-Image-Bench, LongText-Bench, CVTG-2K, and GEdit-Bench demonstrate broad visual quality and prompt alignment, balanced Chinese–English text rendering, and competitive instruction-guided editing without separate task-specific backbones. More broadly, the results show that a capable visual creation system can be trained with a moderate data budget dominated by imageonly and real data, and then efficiently adapted for deployment through few-step distillation. We release four checkpoints in total: the standard Base and Turbo checkpoints, together with an FP8 variant of each. We also release the training and inference code and the progressive recipes to make this path reproducible.

## 2 Model Design

As illustrated in Fig. 3, LLaDA-Image consists of three principal components: a diffusion large language model (dLLM)-based vision-language model (VLM) for multimodal understanding, a connector that projects VLM representations into the generator’s conditioning space, and a diffusion transformer (DiT) for image synthesis (Peebles & Xie, 2023). This modular architecture cleanly decouples high-level semantic interpreta tion from pixel-space generation, assigning each to a dedicated component while using the connector as an explicit bridge. Crucially, the unified architecture seamlessly supports both text-to-image generation and image editing. For editing, the framework reuses the shared text-conditioning pathway and DiT backbone while engaging an additional reference-image pathway directly within the DiT—critically bypassing the VLM entirely.

## 2.1 dLLM-based Vision-Language Model

The understanding component is a VLM built upon the LLaDA 2.0 Mini dLLM backbone, including its SigLIP-VQ vision encoder (Inclusion AI et al., 2026). It processes the text prompt or editing instruction to steer generation, while also being capable of consuming visual tokens during image-only self-conditioning. Unlike conventional text encoders that merely encode the input prompt into static embeddings, the VLM provides a unified interface capable of joint reasoning over multimodal contexts. Crucially, the reference image used during editing is deliberately excluded from this comprehension stage; instead, it is injected directly into the DiT via the dedicated pathway detailed in Sec. 2.3.

Multimodal input construction. Let t denote the text tokenizer and embedding layers, v the SigLIP-VQ vision encoder, and g the VLM backbone. Given an input text sequence $\mathbf { s } _ { \mathrm { t e x t } }$ and an optional image y, we obtain the corresponding text and image token representations as:

$$
\begin{array} { r } { \mathbf { c } _ { \mathrm { t e x t } } = \pmb { t } ( \mathbf { s } _ { \mathrm { t e x t } } ) , \qquad \mathbf { c } _ { \mathrm { i m g } } = \pmb { v } ( \mathbf { y } ) . } \end{array}\tag{1}
$$

Depending on the task, the active token representations in Eq. (1) are concatenated in a predefined order to form a unified sequence c:

• Text-to-Image Generation: c contains solely the text condition $\pmb { \mathrm { c } } _ { \mathrm { t e x t } }$

• Image-Only Pre-training: c comprises the auxiliary prompt alongside the sampled visual tokens $\mathbf { c } _ { \mathrm { i m g } } .$

• Image Editing: c processes only the editing instruction, as the reference image completely bypasses the VLM.

By decoupling these pathways, the VLM remains a general-purpose multimodal understanding module. The bridge that adapts these representations for visual synthesis is the understanding-to-generation connector, described next.

## 2.2 Understanding-to-Generation Connector

The VLM and DiT operate in distinct representation spaces optimized for different objectives: the former emphasizes high-level semantic understanding, whereas the latter requires conditioning signals that directly guide visual denoising trajectories. To bridge this gap, we introduce an understanding-to-generation interface consisting of two generation-specific modules: a Residual Query Adapter (RQA) preceding the VLM prefill, and a Transformer Connector following it. Both modules function independently of the core VLM, serving specifically to extract and project VLM representations into the DiT conditioning space.

Residual query adaptation. The frozen VLM does not directly expose the fine-grained visual priors required for high-fidelity image generation. To avoid the cost of fine-tuning the entire VLM and preserve its pre-trained understanding capabilities, we adopt the lightweight RQA introduced in IOMM (Sun et al., 2026c). Let q denote a set of learnable query tokens and $\mathbf { \Delta } \mathbf { \textit { q } } _ { \psi }$ represent the RQA parameterized by ψ. Prior to VLM prefilling, these learnable queries cross-attend to the full multimodal input sequence c:

$$
\mathbf { q } _ { \mathrm { r e s } } = { \pmb q } _ { \psi } ( \mathbf { q } _ { 0 } , { \pmb c } ) .\tag{2}
$$

Instead of replacing the original input, the residual queries produced by Eq. (2) are appended to c. This augmented sequence is then processed by the frozen VLM backbone g in a single prefill forward pass:

$$
{ \bf h } _ { \mathrm { v l m } } = g ( \mathrm { c o n c a t } ( { \bf c } , { \bf q } _ { \mathrm { r e s } } ) ) .\tag{3}
$$

Together, Eq. (2) and Eq. (3) prompt the frozen VLM to extract and expose multimodal context that is maximally beneficial for generative synthesis.

Condition-space alignment. The prefill hidden states $\mathbf { h } _ { \mathrm { v l m } }$ remain within the VLM representation space, which typically differs from the DiT conditioning space in both feature dimension and geometric structure. We therefore process $\mathbf { h } _ { \mathrm { v l m } }$ with a connector $c _ { \phi }$ composed of a shallow stack of Transformer blocks. This module projects the features and aligns them directly with the DiT condition space:

$$
\mathbf { h } _ { \mathrm { c o n d } } = c _ { \phi } ( \mathbf { h } _ { \mathrm { v l m } } ) .\tag{4}
$$

The connector mapping in Eq. (4) complements the RQA: the former translates VLM features into a format optimized for DiT consumption, while the latter elicits generation-centric information during the VLM prefill. Together, they establish a unified conditioning pipeline across text-to-image generation and image only training. For image editing, this shared pathway handles the instruction text, while an independent reference-image stream feeds directly into the DiT.

## 2.3 Single-stream Diffusion Transformer

Recipe #1. We replace every normalization layer in the DiT with parameter-free RMSNorm (Zhang & Sennrich, 2019). This simple design choice substantially improves optimization stability over long training horizons.

The generation component is a pure single-stream Transformer-based DiT. Given a noised image latent $\mathbf { x } _ { t } ,$ timestep t, and condition encoding $\mathbf { h } _ { \mathrm { c o n d } }$ , we embed the image and condition tokens into a shared sequence and process them jointly through the same stack of Transformer blocks. In contrast to architectures that maintain separate condition and image streams, every block in our DiT directly models interactions between semantic conditions and evolving visual tokens through joint self-attention. The predicted flow field is

$$
\widehat { \mathbf { v } } _ { t } = F _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { h } _ { \mathrm { c o n d } } ) = F _ { \theta } \big ( \mathbf { x } _ { t } , t , c _ { \phi } \big ( g \big ( \mathrm { c o n c a t } \big ( \mathbf { c } , q _ { \psi } ( \mathbf { q } _ { 0 } , \mathbf { c } ) \big ) \big ) \big ) \big ) ,\tag{5}
$$

In $\operatorname { E q . } ( 5 ) , F _ { \theta }$ denotes the DiT. This single-stream formulation provides a direct path for multimodal semantics to influence image synthesis at every layer, while retaining a simple and homogeneous Transformer backbone.

Reference-image conditioning for editing. As shown in the lower editing pathway of Fig. 3, image editing introduces a reference-image stream in parallel with, rather than inside, the common VLM–connector pathway. Let $\mathbf { y } _ { \mathrm { r e f } }$ denote the clean reference image. We first extract its semantic representation with SigLIP-VQ, and then process the resulting features using a DiT-specific reference branch $\widehat { b } _ { \omega } \colon$

$$
\mathbf { f } _ { \mathrm { r e f } } = v ( \mathbf { y } _ { \mathrm { r e f } } ) , \qquad \mathbf { h } _ { \mathrm { r e f } } = b _ { \omega } ( \mathbf { f } _ { \mathrm { r e f } } ) .\tag{6}
$$

The branch $b _ { \omega }$ in $\operatorname { E q . }$ (6) consists of a dedicated feature embedder followed by two Transformer layers. It adapts the SigLIP-VQ representation to the DiT token space without injecting the reference image into the VLM. The resulting reference tokens are unified with the text condition produced by the connector,

$$
{ \bf h } _ { \mathrm { j o i n t } } = \mathrm { c o n c a t } ( { \bf h } _ { \mathrm { c o n d } } , { \bf h } _ { \mathrm { r e f } } ) ,\tag{7}
$$

The joint tokens in Eq. (7) then participate with the visual tokens in all subsequent single-stream Transformer layers of the DiT.

Semantic features alone cannot fully preserve the low-level appearance of regions that should remain unchanged. We therefore additionally encode the clean reference image with the FLUX.2 VAE (Black Forest Labs, 2026) and concatenate its latent with the noised target latent $\mathbf { x } _ { t }$ before the DiT input embedder:

$$
\begin{array} { r } { \boldsymbol { x } _ { \mathrm { r e f } } = \mathrm { V A E } ( \boldsymbol { \mathbf { y } } _ { \mathrm { r e f } } ) , \qquad \boldsymbol { x } _ { t } ^ { \mathrm { e d i t } } = \mathrm { c o n c a t } ( \boldsymbol { x } _ { t } , \boldsymbol { x } _ { \mathrm { r e f } } ) , \qquad \boldsymbol { \widetilde { \mathbf { v } } } _ { t } ^ { \mathrm { e d i t } } = F _ { \boldsymbol { \theta } } \left( \boldsymbol { x } _ { t } ^ { \mathrm { e d i t } } , t , \boldsymbol { \mathbf { h } } _ { \mathrm { j o i n t } } \right) . } \end{array}\tag{8}
$$

The semantic pathway in Eq. (6) and the pixel pathway in Eq. (8) are complementary: the former supplies high-level guidance about the reference content, whereas the clean VAE latent provides native pixel-level evidence for regions that do not need to be modified. Their combination improves instruction following while substantially strengthening identity, structure, texture, and background consistency with the reference image, all without introducing a separate editing backbone.

## 3 Data Collection and Preparation

We collect a large-scale image corpus from web sources and curated image collections. Most samples are real images, while synthetic images form a smaller part of the supervised fine-tuning (SFT) data and are used mainly for text rendering. The corpus provides image-only data for pre-training and mid-training, and paired image–text data for SFT.

Recipe #2. Real image-only data can improve final generation quality and support visual-prior learning at scale without paired captions.

## 3.1 Data Composition

We organize the SFT data into four broad content groups: nature, design, people, and synthetic content. Of the 220M samples used for generation training, 98% are real images, while image-only samples account for more than 90% of the total. Pre-training and mid-training use only real images, while the real-image share of the paired image–text data used for SFT remains above 70%. Using real image-only data during pre-training and mid-training provides a scalable way to expose the model to diverse visual content. Fig. 4 summarizes the SFT-weighted data composition.

![](images/f3414a0abd95cb364c0c118af48710172f20b029f69811ee295e97e6d739c94e.jpg)  
Figure 4: Estimated SFT-weighted content distribution, with examples illustrating each content group.

## 3.2 Data Processing

We filter the collected images in three stages: metadata, aesthetics, and quality filtering. Metadata filtering keeps only valid images with a total pixel count greater than 1024<sup>2</sup> and a file-size-to-pixel ratio of at least 0.15 bytes per pixel. Aesthetics filtering removes images with ArtiMuse (Cao et al., 2025) scores below 60, and quality filtering removes images with DeQA-Score (You et al., 2025) below 4.0. After filtering, pretraining uses random square crops resized to 256<sup>2</sup>, whereas mid-training and subsequent SFT stages use the aspect-ratio buckets described in Sec. 4.3.

We use both Qwen3.6-35B-A3B (Qwen Team, 2026) and Qwen3-VL-235B-A22B-Instruct (Bai et al., 2025) to caption the filtered images from image content alone. The captioning prompt requires a faithful and complete description of visible subjects, attributes, actions, spatial relations, scenes, and visual styles. The models also transcribe visible text while preserving its original language and spelling.

We keep the resulting image–text pairs only when image quality and caption agreement are sufficient. Qwen3.6-35B-A3B compares each caption with its source image and removes captions that hallucinate objects, attributes, relations, or text, or that inaccurately transcribe visible text. We further discard malformed outputs, refusal responses, degenerate repetitions, private information, and watermark signals. The remaining captions provide image–text supervision for SFT. Additional details are provided in App. D.

For image-editing data, we apply image-quality filtering similar to that used for the SFT data, followed by an edit-instruction consistency check. We keep only source–target pairs whose instructions describe the visible changes in the correct direction and contain no unsupported details.

## 4 Training Recipe

The overall training and release path is summarized in Fig. 2. Inspired by recent studies Inclusion AI et al. (2026); Han et al. (2026) demonstrating that strong comprehension capabilities can facilitate generation, we first conduct Chain-of-Thought (CoT) supervised fine-tuning to enhance the backbone’s performance. The

details of our training recipe are provided below. Tab. 1 and Tab. 2 summarize the independent training pipelines for understanding and generation, respectively. Across all generation stages, we use the Muon optimizer (Liu et al., 2025a).

Table 1: Detailed configuration for understanding training. CoT SFT denotes CoT Supervised Fine-tuning.
<table><tr><td>Configuration</td><td>CoT SFT</td></tr><tr><td>Training Process</td><td></td></tr><tr><td>Training Samples Training Epochs Resolution</td><td>~2.6M packed seq. (16,384 tok.) 4  $5 1 2 ^ { 2 }$ </td></tr><tr><td>Global Batch Size</td><td>512</td></tr><tr><td>Data Distribution</td><td></td></tr><tr><td>Supervision Data Composition</td><td>Masked answer tokens (block diffusion, b=32)</td></tr><tr><td>Hyperparameters</td><td> $\mathrm { G e n : U n d : T e x t } = 9 { : } 9 { : } 2$ </td></tr><tr><td></td><td></td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 5 )$ </td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 5 } \to 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR Schedule</td><td>Cosine decay</td></tr><tr><td>LR Warmup</td><td>Linear, 1%</td></tr><tr><td>Weight Decay</td><td></td></tr><tr><td></td><td>0.1</td></tr><tr><td>Grad. Norm Clip</td><td>1.0</td></tr></table>

## 4.1 CoT Supervised Fine-tuning

Data Process and Construction. We tokenize all corpora offline and greedily bin-pack them into sequences of exactly $L = 1 6 { , } 3 8 4$ tokens, and we VQ-encode each image once at a $5 1 2 ^ { 2 }$ target resolution under variableaspect-ratio cropping. A block-diagonal attention mask built from the cumulative lengths keeps packed samples mutually invisible, so packing raises utilization without cross-sample leakage. We mix generation, understanding and text-only data at a 9:9:2 ratio.

Training Objective. We keep the block-diffusion formulation (Inclusion AI et al., 2026) with a block size of $b = 3 2$ . For each supervised region we draw a masking ratio $\rho = \cos ( r \pi / 2 )$ with $r \sim \mathcal { U } ( 0 , 1 )$ and replace that fraction of the response tokens by the mask token. The system prompt, the question and all visual tokens are never masked and never supervised, so the loss is a cross-entropy over the masked positions of the assistant turn: the <think> trace together with the answer. One mask covers only part of the response, so we consume each batch twice, under the sampled mask and under its complement, as two optimizer steps. Every response token is then supervised within the pair, while each forward pass still reads a partially clean context. We normalize the loss per block: within each block we divide the loss of every supervised token by the number of supervised tokens in that block, then average over the blocks that carry supervision and over the packed sequences in the batch.

Optimization. We train with AdamW (Loshchilov & Hutter, 2019) under FSDP2 full sharding with mixed precision and gradient checkpointing. The learning rate warms up linearly and then decays by cosine to its floor; since the complementary pass takes its own step, the schedule spans twice the number of data iterations.

## 4.2 Image-only Pre-training

The most compute-intensive stage of visual generation training conventionally relies on large-scale image– text pairs. Following Image-Only Training for UMMs (IOMM) (Sun et al., 2026c), we instead bootstrap the visual generative component from unlabeled images alone. The key observation is that an image already contains the high-level semantics required to supervise its generation: a frozen vision-language model (VLM) can extract these semantics and use them as a self-conditioning signal. This stage therefore learns a strong visual prior before any explicit image–text alignment, while keeping the pre-trained VLM intact.

Recipe #3. Large-scale image–text pairs are not required to pre-train LLaDA-Image: images alone provide sufficient supervision for learning a strong visual generative prior.

Resolution-aware image sampling. We conduct image-only pre-training at a resolution of $2 5 6 ^ { 2 }$ . When training with conventional image–text pairs, the caption usually describes the complete image; consequently, the entire image must be resized to the training resolution to preserve image–caption alignment. For a high-resolution source image, aggressive downsampling to $2 5 6 ^ { 2 }$ can remove fine-grained content referenced by the caption, introduce visual distortions, and ultimately create a mismatch between the image and its text condition. Image-only pre-training removes this constraint because its condition is derived from the sampled image itself. We randomly crop a region from each source image whose native crop resolution is either $\bar { 2 5 6 ^ { 2 } }$ or moderately larger, and then resize the crop to $2 5 6 ^ { 2 }$ with only a small downsampling ratio. The VLM extracts the semantic condition directly from this processed crop, ensuring that the condition supplied to the diffusion transformer remains faithful to the actual training target. This sampling strategy exposes the model to substantially more local detail while avoiding information loss and distortion caused by aggressively compressing an entire high-resolution image.

Masked self-conditioning. Given a training image ${ \bf y } ,$ let x denote its clean latent in the generator’s representation space. Following the notation introduced in Sec. 2.1, v denotes the frozen SigLIP-VQ vision encoder and g denotes the frozen VLM. We first encode the image into a sequence of patch features $\mathbf { c } _ { \mathrm { i m g } } =$ ${ \pmb v } ( { \bf y } ) \in \mathbb { R } ^ { N \times D }$ , where N is the number of image tokens and D is their feature dimension. These features are paired with ${ \mathfrak { c } } _ { \mathrm { a u x } } = t ( { \mathfrak { s } } _ { \mathrm { a u x } } )$ , where $\pmb { \mathscr { s } } _ { \mathrm { a u x } }$ is a fixed auxiliary prompt such as Generate an image identical to the reference image. Directly conditioning on all image features would reveal an almost complete description of the reconstruction target and encourage a trivial identity mapping. We therefore define an image-token masking ratio $\rho _ { \mathrm { i m g } }$ and sample a binary token mask $\mathbf { m } \in \{ 0 , 1 \} ^ { N }$ with keep probability $1 - \rho _ { \mathrm { i m g } } .$

$$
\begin{array} { r } { \widetilde { \mathbf { c } } _ { \mathrm { i m g } } = \mathbf { c } _ { \mathrm { i m g } } \odot \mathbf { m } , \qquad \mathbf { c } = \mathrm { c o n c a t } \big ( \mathbf { c } _ { \mathrm { a u x } } , \widetilde { \mathbf { c } } _ { \mathrm { i m g } } \big ) , } \end{array}\tag{9}
$$

where m is broadcast along the feature dimension. The construction in Eq. (9) converts pre-training from dense reconstruction into a sparse-to-dense prediction task: the generator must recover missing content from the visible patches and thus learn contextual and compositional visual structure. This stage changes only how the multimodal input sequence c is constructed; its subsequent transformation follows the common model architecture described in Sec. 2.1 and Sec. 2.2. Specifically, the RQA, frozen VLM, and connector apply Eq. (2), Eq. (3), and Eq. (4), respectively, to produce $\bar { \mathbf { h } _ { \mathrm { c o n d . } } }$ , which conditions the DiT through Eq. (5). Thus, image-only training reuses exactly the same RQA–VLM–connector pathway as downstream conditional generation.

Flow-matching objective. We optimize the visual generator with flow matching (Lipman et al., 2023). For Gaussian noise $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $t \stackrel { \cdot } { \sim } \mathcal { U } ( 0 , 1 )$ , we construct the interpolation ${ \bf x } _ { t } = ( \breve { 1 } - t ) { \bf x } + t { \bf z }$ . The generator $F _ { \theta }$ predicts the corresponding constant velocity field, and is trained with

$$
\mathcal { L } _ { \mathrm { F M } } ( \theta , \phi , \psi ) = \mathbb { E } _ { \mathbf { y } , \mathbf { x } , \mathbf { z } , t , \mathbf { m } } \left[ \| F _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { h } _ { \mathrm { c o n d } } ) - ( \mathbf { z } - \mathbf { x } ) \| _ { 2 } ^ { 2 } \right] .\tag{10}
$$

When optimizing Eq. (10), only the visual generator $F _ { \theta } ,$ residual query adapter ${ \pmb q } _ { \psi } ,$ and connector $c _ { \phi }$ are updated; v and g remain frozen throughout this stage. Combining masked reconstruction with parameterefficient self-conditioning allows the model to absorb large-scale visual knowledge from inexpensive imageonly data and provides a robust initialization for the subsequent alignment stages.

Table 2: Detailed configurations for generation training. PT, MT, and SFT denote image-only pre-training, image-only mid-training, and supervised fine-tuning, respectively. SFT comprises a $5 1 2 ^ { 2 }$ text-alignment stage and a $5 1 2 ^ { 2 }  1 0 2 4 ^ { 2 }$ resolution-transition stage, followed by refinement and editing; the final column reports few-step distillation. Centered entries apply to all stages they span. The 220M samples are counted cumulatively across the image-generation pipeline and refer exclusively to generation training.
<table><tr><td rowspan="2">Configuration</td><td rowspan="2">PT</td><td rowspan="2">MT</td><td colspan="3">SFT</td><td rowspan="2">Distillation</td></tr><tr><td> $5 1 2 ^ { 2 } \ { \bf A l i g n . \ 5 1 } 2 ^ { 2 }  { \mathrm { 1 0 } } 2 4 ^ { 2 }$ </td><td>Refine.</td><td>Editing</td></tr><tr><td>Training Process</td><td colspan="6"></td></tr><tr><td>Training Samples</td><td colspan="6">220M total</td></tr><tr><td>Resolution</td><td rowspan="2"> $2 5 6 ^ { 2 }$  24,576</td><td rowspan="2"> $5 1 2 ^ { 2 }$  (bucketed) 6,400</td><td rowspan="2"> $5 1 2 ^ { 2 }$  (bucketed)</td><td colspan="2" rowspan="2"> $1 0 2 4 ^ { 2 } ( \mathrm { b u c k e t e d } )$ </td><td rowspan="2"> $1 0 2 4 ^ { 2 }$  (bucketed) 256</td></tr><tr><td>Global Batch Size</td><td>4,608 2,048</td></tr><tr><td>Sampling Steps</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2" rowspan="2">2,880 2,688 50</td><td rowspan="2">2-4</td></tr><tr><td>Data Distribution</td></tr><tr><td>Supervision</td><td colspan="2">Image-only</td><td colspan="2">Image-Text Pairs</td><td></td><td>Image-Text Pairs</td></tr><tr><td>Data Ratio</td><td colspan="2">Image-only</td><td colspan="2">T2I only</td><td>T2I:I2I = 1:1</td><td>T2I:I2I = 1:1</td></tr><tr><td>Hyperparameters</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Optimizer</td><td>Muon</td><td></td><td>Muon</td><td></td><td></td><td>Muon</td></tr><tr><td>Learning Rate</td><td> $4 \times 1 0 ^ { - 4 } \qquad 2 \times 1 0 ^ { - 4 }$ </td><td></td><td> $5 \times 1 0 ^ { - 5 }$ </td><td></td><td> $3 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Weight Decay</td><td>0.0</td><td></td><td></td><td>0.0</td><td></td><td>0.0</td></tr><tr><td>Grad. Norm Clip</td><td>1.0</td><td></td><td></td><td>1.0</td><td></td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EMA Ratio</td><td></td><td>一</td><td></td><td>0.9995</td><td></td><td>0.995</td></tr></table>

## 4.3 Image-only Mid-training

We continue to use the same image-only training formulation as in pre-training, including image-derived self-conditioning, image-token masking, and the flow-matching objective, while increasing the nominal pixel budget from $2 5 \bar { 6 } ^ { 2 }$ to 512<sup>2</sup>. The condition therefore remains inherently aligned with the processed training image, allowing us to expose the DiT to finer spatial structure without introducing caption mismatch. We also begin using the aspect-ratio-bucketed sampling strategy detailed below, so $5 1 2 ^ { \overset { \sim } { 2 } }$ denotes the target pixel budget rather than requiring every image to be resized to a fixed square.

This stage primarily serves as a bridge between image-only pre-training and paired text-to-image alignment. As discussed above, captions generally describe the complete source image, requiring the full image to be downsampled when image–text pairs are used. Excessive compression can remove fine-grained details referenced by the caption and thereby weaken image–text compatibility. More importantly, directly switching from 256<sup>2</sup> image-only training to high-resolution image–text pair training would change both the spatial distribution and the source of supervision at the same time, producing an unnecessarily abrupt shift in the training regime. Image-only mid-training decouples these two changes: the model first adapts to the 512<sup>2</sup> pixel budget under the familiar image-derived condition, and only then transitions to textual supervision during SFT. This staged schedule provides a smoother transition in both resolution and conditioning modality while preserving the visual prior learned during pre-training.

Aspect-ratio-bucketed variable-resolution training. We introduce variable-resolution training during image-only mid-training and retain it throughout all subsequent SFT stages. The data loader maintains a predefined collection of resolution buckets with different aspect ratios but approximately the same pixel budget. When the dataset yields an image, the data loader assigns it to the bucket whose aspect ratio is closest to that of the source image. It then downsamples the image while preserving its original aspect ratio. Because the resized image must cover the selected bucket, one side can remain slightly larger than the target; we simply crop this small overflow. Since the selected bucket already closely matches the source aspect ratio, only a few boundary pixels are typically removed and the semantic content is essentially unaffected. The detailed assignment rule, preprocessing procedure, and complete bucket configurations are provided in App. A.

This strategy offers three practical benefits. First, although different data parallel ranks may receive different spatial shapes, the nearly constant per-image pixel budget keeps their memory consumption and computation closely matched, avoiding rank-specific memory peaks and stragglers. Second, preserving the source aspect ratio prevents geometric distortion from forcibly resizing every image to a square, while the minimal crop avoids the substantial content loss caused by a square center crop. Finally, exposing the model to multiple spatial shapes directly equips it to generate images at diverse aspect ratios.

## 4.4 Supervised Fine-tuning

Logit-normal timestep sampling. We adopt logit-normal timestep sampling (Esser et al., 2024) instead of a uniform distribution, which allocates the same number of training examples to all noise levels, although their importance during inference is not uniform. Under our interpolation convention ${ \bf x } _ { t } = ( 1 - t ) { \bf x } + { \bf \bar { \it t } } { \bf z } ,$ larger t corresponds to a higher-noise state. Predictions in this region are more sensitive and less stable: the image structure has not yet been established, and errors made early in the denoising trajectory can propagate through all subsequent steps. At small t, by contrast, most visual content has already stabilized and the remaining denoising correction is comparatively limited. We therefore devote a larger fraction of the SFT optimization budget to high-noise states instead of sampling the high- and low-noise regions equally.

Specifically, we draw one standard Gaussian variable for each training sample and transform it with a sigmoid function,

$$
\epsilon \sim \mathrm { \mathcal { N } } ( 0 , 1 ) , \qquad t = \mathrm { s i g m o i d } \big ( P _ { \mathrm { m e a n } } + P _ { \mathrm { s t d } } \epsilon \big ) ,\tag{11}
$$

In $\mathrm { E q . } ( 1 1 ) , P _ { \mathrm { m e a n } }$ controls the center of the distribution in logit space and $P _ { \mathrm { s t d } }$ controls its spread. We set $\tilde { P _ { \mathrm { { m e a n } } } } = 0 . 8$ and $P _ { \mathrm { s t d } } = 0 . 8$ to shift more probability mass toward larger t while retaining support across the complete denoising trajectory. This logit-normal sampling strategy concentrates supervision on the more challenging high-noise regime, improving the stability of the early denoising process without repeatedly oversampling low-noise states that are already comparatively well resolved.

## 4.4.1 Text-to-Image Training

The preceding image-only pre-training and mid-training stages equip the diffusion transformer (DiT) with a strong visual prior and basic generation capability without requiring paired captions. However, the final text-to-image system must generate images from textual instructions at inference time. We therefore introduce image–text pairs during supervised fine-tuning to align the semantic representations extracted from text with the visual generation space learned in the image-only stages. This transition brings the training condition closer to the model’s actual usage while preserving the generative prior acquired from substantially larger image-only corpora.

Recipe #4. Compared with training predominantly on synthetic data, using a large proportion of real images leads to slower early convergence on standard benchmarks, but substantially improves the realism of the model’s final generated images after sufficient training.

$5 1 2 ^ { 2 } .$ -pixel alignment. We begin with an alignment stage using aspect-ratio buckets containing approximately $5 1 2 ^ { 2 }$ pixels per image. Here, $5 1 2 ^ { 2 }$ is only the square bucket rather than a fixed training shape; non-square images are assigned to buckets with comparable pixel counts while retaining their aspect ratios. Because image-only mid-training has already adapted the model to this pixel budget, the resolution distribution remains approximately fixed while the supervision changes from image-derived conditions to paired text. The primary purpose of this stage is therefore not to relearn image synthesis from scratch, but to associate textual concepts, attributes, and spatial relations with the corresponding visual content while preserving the well-formed visual prior of the DiT.

Scaling to $1 0 2 4 ^ { 2 }$ pixels. After establishing image–text alignment with the $5 1 2 ^ { 2 }$ -pixel buckets, we continue training with a second set of aspect-ratio buckets containing approximately 1024<sup>2</sup> pixels per image. The progressive pixel-budget schedule avoids introducing high-resolution generation and text alignment simultaneously, yielding a smoother transition to the final operating resolution. At this stage, the model learns to preserve semantic alignment while producing finer textures, sharper boundaries, and more spatial detail. The exact bucket resolutions used at both pixel budgets are listed in App. A.

Targeted refinement and data composition. In the refinement stage, we increase the proportions of textintensive and portrait-centric examples to strengthen two particularly demanding capabilities: accurate text rendering and high-fidelity human generation. At the same time, we deliberately limit the amount of synthetic data throughout all SFT stages, with the real-image share consistently remaining above 70% of the training mixture. Although this real-data-dominant recipe can converge more slowly than a synthetic-heavy mixture early in training, we find that its advantage becomes increasingly apparent over longer training: the generated images exhibit substantially stronger realism and the model reaches a higher overall capability ceiling.

## 4.4.2 Image Editing Training

Many open-source systems provide text-to-image generation and image editing through separate model variants (Wu et al., 2025a; Qwen Team, 2025c; Z-Image Team et al., 2025; Meituan LongCat Team et al., 2025). In contrast, we aim to support both capabilities within a single unified model, avoiding task-specific checkpoints and allowing generation and editing to benefit from a shared visual prior.

Mixed-task continued training. During the editing stage, we jointly train on text-to-image (T2I) and image to-image editing (I2I) examples with a 1:1 sampling ratio. Editing data alone has a narrower distribution than the large-scale T2I corpus, and continued training exclusively on it can cause the model to overfit the editing objective, weaken its open-ended generation quality, and forget part of the visual prior acquired in earlier stages. Mixing T2I examples into every stage of editing training acts as capability replay: it continuously regularizes the model toward broad semantic coverage, visual quality, and text-to-image alignment. At the same time, the I2I examples teach precise edit instruction following and reference-content preservation. The balanced mixture therefore reduces task interference and allows one checkpoint to acquire editing ability without sacrificing its original text-to-image capability. More importantly, the two tasks share the same generative backbone, so improvements in visual priors and instruction understanding can transfer between generation and editing instead of being isolated in two model branches.

Reference-conditioned editing. The editing data flow augments the standard T2I path with a parallel reference image stream. The editing instruction follows the common RQA–VLM–connector pathway, whereas the reference image does not enter the VLM. Its semantic features are extracted by SigLIP-VQ and processed by a DiT-specific embedder followed by two Transformer layers. These reference tokens are then unified with the other condition and visual tokens before the main single-stream DiT layers. In parallel, the clean reference image is encoded into a VAE latent and concatenated with the noised target latent x<sub>t</sub>. This second signal directly exposes native pixel information from regions that should remain unchanged, improving structural and visual consistency between the edited result and its reference. The complete model pathway is detailed in Sec. 2.3; its semantic, joint, and pixel-level conditions are defined in Eq. (6), Eq. (7), and Eq. (8), respectively.

## 4.4.3 Checkpoint Merging

Although our data loader controls the sampling ratios among different data sources at the global level, the empirical data composition seen by the model can still vary across individual training steps. Such local variation induces small fluctuations in the learned parameters. We observe the same phenomenon when monitoring generation benchmarks throughout training: once optimization approaches a performance plateau, scores from nearby checkpoints continue to oscillate, even though their overall capabilities are comparable. Selecting a single checkpoint may therefore retain step-specific bias and make the final result unnecessarily sensitive to short-term sampling noise.

Motivated by weight averaging (Izmailov et al., 2018), we apply checkpoint merging to the converged checkpoints. Aggregating model parameters from multiple nearby training states suppresses transient, step-dependent deviations while preserving the shared capabilities learned across them. In practice, this substantially smooths the variation observed among individual checkpoints and produces a more stable and robust final model across different evaluation benchmarks.

## 4.5 Few-step Distillation

## 4.5.1 TwinFlow for Few-step Distillation

We distill the multi-step LLaDA-Image into LLaDA-Image Turbo, a few-step generator. We use Twin-Flow (Cheng et al., 2026), which builds on distribution-matching distillation (Yin et al., 2024; Liu et al., 2026). DMD2 minimizes the reverse KL divergence between the student distribution and the target distribution using a frozen real-score model and an online fake-score model. The latter must track the non-stationary distribution produced by the student, so the fake-score and generator objectives are optimized alternately, with the fake-score estimator assigned the faster time scale. Following TwinFlow’s signed-time construction, we represent both roles with one shared DiT backbone instead of introducing a separate fake-score network. Positive times $+ t \in [ 0 , 1 ]$ are reserved for the generator update, whereas the corresponding negative times $- t \in [ - 1 , 0 ]$ train the fake-score estimator. The sign is a role indicator: +t and −t denote different learning roles.

Let $F _ { \mathrm { b a s e } }$ be a frozen copy of the original multi-step model and $F _ { \theta }$ the model being distilled. Starting from $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , we derive the objective using the direct one-step mapping from z to a clean sample:

$$
\widehat { \mathbf { x } } = \mathbf { z } - { \cal F } _ { \boldsymbol { \theta } } \big ( \mathbf { z } , + 1 , \mathbf { h } _ { \mathrm { c o n d } } \big ) .\tag{12}
$$

Under the flow-matching convention used in Eq. (12), $F _ { \theta }$ predicts $\mathbf { z } - \mathbf { x } ,$ so this update directly maps noise to data. To learn the fake score, we detach the sample x from Eq. (12) and sample $t \sim \mathcal { U } ( 0 , 1 )$ and fresh noise $\mathbf { z } ^ { \prime } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ to form

$$
\widetilde { \mathbf { x } } _ { t } = ( 1 - t ) \mathbf { s g } ( \widehat { \mathbf { x } } ) + t \mathbf { z } ^ { \prime } .\tag{13}
$$

Using the noised fake sample in Eq. (13), the negative-time branch is optimized with

$$
\mathcal { L } _ { \mathrm { f a k e } } ( \theta ) = \mathbb { E } \left[ \left| \left| F _ { \theta } ( \widetilde { \mathbf { x } } _ { t } , - t , \mathbf { h } _ { \mathrm { c o n d } } ) - \left( \mathbf { z } ^ { \prime } - \mathbf { s g } ( \widehat { \mathbf { x } } ) \right) \right| \right| _ { 2 } ^ { 2 } \right] ,\tag{14}
$$

In Eq. $( 1 4 ) , \thinspace \mathrm { { s g } ( \cdot ) }$ denotes stop-gradient. This loss fits the −t branch to the evolving student distribution, with gradients flowing only through the negative-time prediction.

For distribution matching, we retain gradients through the sample in Eq. (12) and re-noise it as

$$
\widehat { \mathbf { x } } _ { t } = ( 1 - t ) \widehat { \mathbf { x } } + t \mathbf { z } ^ { \prime } .\tag{15}
$$

A velocity prediction on the noised sample in Eq. (15) yields the score $\mathbf { \widehat { s } } _ { t } = - [ \widehat { \mathbf { x } } _ { t } + ( 1 - t ) \widehat { \mathbf { v } } _ { t } ] / t$ . Both models are evaluated on $\widehat { \mathbf { x } } _ { t }$ with $\mathbf { h } _ { \mathrm { c o n d } } \colon$ applying this conversion to the frozen $F _ { \mathrm { b a s e } } \mathrm { a t } \mathrm { + } t$ gives ${ \bf s } _ { \mathrm { r e a l } } ,$ whereas applying it to $F _ { \theta }$ at −t gives $\pmb { \mathrm { s } } _ { \mathrm { f a k e } }$ . No separate auxiliary score network is introduced. The DMD objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D M D } } ( \theta ) = \mathbb { E } \big [ w ( t ) \langle s \mathrm { g } ( s _ { \mathrm { f a k e } } - s _ { \mathrm { r e a l } } ) , \widehat { \mathbf { x } } _ { t } \rangle \big ] , } \end{array}\tag{16}
$$

In Eq. (16), w(t) is the timestep weight. With both scores detached, $\operatorname { E q } .$ (16) updates only the positive-time generator, whereas Eq. (14) updates the negative-time fake-score branch. Alternating the two realizes DMD optimization within a shared model.

In practice, we adopt three implementation choices. First, motivated by the “one-input, dual-output” design of Duality Models (DuMo) (Sun et al., 2026b), we attach two output heads to the shared DiT backbone. The fake-score head produces the negative-time prediction used to compute ${ \bf s } _ { \mathrm { f a k e } } ,$ whereas the DMD head parameterizes the positive-time generator optimized by Eq. (16). Thus, calls to $F _ { \theta } \mathrm { a t } - t$ and +t in the notation above select the fake-score and DMD heads, respectively. At inference, we retain only the DMD head and discard the fake-score head, so the auxiliary branch introduces no inference-time overhead. Second, we use four-step backward simulation to align the training inputs with the student’s inference trajectory. Third, we update the generator and fake-score branch at a 1:2 schedule. Compared with the 1:5 schedule in DMD2 (Yin et al., 2024), this reduces fake-score updates and improves training efficiency. After distillation, we further refine the model with TBSM (Sun et al., 2026a)-based reinforcement tuning.

## 5 Experiments

## 5.1 Text-to-Image Generation

## 5.1.1 General Image Generation Performance

Unless otherwise specified, the baseline results in this section are the raw scores reported by the corresponding models. For consistent evaluation, all benchmark results for LLaDA-Image Turbo are obtained with four sampling steps. We exclude results obtained with prompt enhancement, explicit thinking or reasoning, or other test-time prompt rewriting pipelines, so that the comparison reflects the generation model under its standard inference setting.

Compared systems and sources. The proprietary comparison set comprises GPT-Image 2 (OpenAI, 2026), GPT-Image 1.5 (OpenAI, 2025b), GPT-Image 1 (including its High setting) (OpenAI, 2025a), Nano-Banana 2.0 (Google DeepMind, 2026), Nano-Banana Pro (Google DeepMind, 2025c), Qwen-Image 2.0 Pro (Zhao et al., 2026), Seedream 5.0 (ByteDance Seed, 2026), Seedream 4.5 (ByteDance Seed, 2025b), Seedream 4.0 (ByteDance Seed, 2025a), FLUX.2 Max and Pro (Black Forest Labs, 2026), Imagen 4.0 and Imagen 4.0 Ultra (Google DeepMind, 2025a), and Kling-Image 2.1 (Kuaishou, 2025). The remaining baselines include LLaDA 2.0 Uni (Inclusion AI et al., 2026), SenseNova U1.5 Preview (SenseNova Team, 2026), Lumina-DiMOO (Xin et al., 2025a), the Qwen-Image family (Wu et al., 2025a; Qwen Team, 2025a), the LongCat-Image family (Meituan

LongCat Team et al., 2025), the Boogu-Image 0.1 Base and Turbo variants (Chen et al., 2026), InternVL-U (Tian et al., 2026), the Z-Image family (Z-Image Team et al., 2025), Seedream 3.0 (Gao et al., 2025), GLM-Image (Z.ai, 2026), Lumina-Image 2.0 (Qin et al., 2025), LLaDA-o (You et al., 2026), HunyuanImage 3.0 (Tencent Hunyuan Foundation Model Team, 2025), NextFlow (Zhang et al., 2026), Janus Pro (Chen et al., 2025b), OmniGen2 (Wu et al., 2025b), Emu3-Gen (Wang et al., 2024b), MMaDA (Yang et al., 2025), Lumina-mGPT 2.0 (Xin et al., 2025b), X-Omni (Geng et al., 2025), BAGEL (Deng et al., 2025), FLUX.1 [Dev] (Black Forest Labs, 2024), Kolors 2.0 (Kuaishou Kolors Team, 2025), HiDream-I1 Full (Cai et al., 2025), and BLIP3-o (Chen et al., 2025a).

Qwen-Image-Bench. Qwen-Image-Bench (Li et al., 2026) is a creator-oriented benchmark that uses 1,000 stratified prompts to assess five complementary aspects of practical image generation: visual quality, aes thetics, prompt alignment, real-world fidelity, and creative generation. Its English and Chinese tracks also allow us to examine whether a model maintains the same breadth of capabilities across languages rather than specializing in a single prompt distribution. Fig. 1 summarizes the overall comparison, while Tab. 3 and Tab. 4 provide the dimension-level breakdowns. LLaDA-Image obtains overall scores of 53.53 on the English track and 53.38 on the Chinese track. These results exceed the next-best open-source baseline, Z-Image Turbo, by 1.87 and 0.67 points, respectively, establishing a new open-source state of the art on both tracks without prompt enhancement or a test-time thinking pipeline. The gains are broad: LLaDA-Image ranks first among the compared open-source models in Quality, Aesthetics, and Alignment in both languages, as well as in Creative Generation on the English track.

Table 3: Comparison of Text-to-Image Generation Performance on Qwen-Image-Bench (EN).
<table><tr><td>Type Model</td><td>Quality</td><td>Aesthetics</td><td>Alignment</td><td>Real-world Fidelity</td><td>Creative Generation</td><td>Overall</td></tr><tr><td>GPT-Image 2</td><td>59.09</td><td>68.48</td><td>65.78</td><td>59.40</td><td>75.34</td><td>65.23</td></tr><tr><td>GPT-Image 1.5</td><td>55.78</td><td>62.87</td><td>61.39</td><td>55.86</td><td>67.06</td><td>60.42</td></tr><tr><td>Nano-Banana 2.0</td><td>54.86</td><td>62.63</td><td>61.11</td><td>54.66</td><td>64.49</td><td>59.59</td></tr><tr><td>Nano-Banana Pro</td><td>55.30</td><td>61.38</td><td>60.30</td><td>55.91</td><td>64.54</td><td>59.33</td></tr><tr><td>Qwen-Image 2.0 Pro</td><td>55.16</td><td>60.36</td><td>57.86</td><td>53.06</td><td>63.59</td><td>57.90</td></tr><tr><td>Seedream 5.0</td><td>54.01</td><td>59.96</td><td>58.63</td><td>53.86</td><td>63.64</td><td>57.80</td></tr><tr><td>Seedream 4.5</td><td>54.05</td><td>60.11</td><td>57.44</td><td>51.55</td><td>59.82</td><td>56.95</td></tr><tr><td>Seedream 4.0</td><td>54.40</td><td>59.26</td><td>56.75</td><td>52.68</td><td>57.64</td><td>56.48</td></tr><tr><td>Clos-uce FLUX.2 Max</td><td>53.99</td><td>58.77</td><td>57.31</td><td>50.69</td><td>57.79</td><td>56.14</td></tr><tr><td>FLUX.2 Pro</td><td>52.88</td><td>58.71</td><td>57.59</td><td>48.45</td><td>57.36</td><td>55.61</td></tr><tr><td>GPT-Image 1</td><td>52.50</td><td>55.77</td><td>55.35</td><td>48.25</td><td>57.29</td><td>54.24</td></tr><tr><td>Imagen 4.0 Ultra</td><td>51.16</td><td>55.64</td><td>53.75</td><td>46.00</td><td>51.32</td><td>52.42</td></tr><tr><td>Imagen 4.0</td><td>50.63</td><td>53.93</td><td>52.56</td><td>45.13</td><td>48.56</td><td>51.08</td></tr><tr><td>Kling-Image 2.1 LLaDA-Image</td><td>49.04</td><td>50.94</td><td>50.47</td><td>44.29</td><td>46.23</td><td>48.89</td></tr><tr><td>Z-Image Turbo Boogu-Image 0.1 Turbo</td><td>53.22 51.25</td><td>58.22 54.87</td><td>54.77 53.61</td><td>43.90</td><td>51.09</td><td>53.53</td></tr><tr><td>Opeo-uce</td><td>51.30</td><td>53.91</td><td>53.37</td><td>44.08 46.52</td><td>49.20 48.90</td><td>51.66</td></tr><tr><td>HunyuanImage 3.0</td><td>50.76</td><td>54.66</td><td>53.16</td><td>45.33</td><td>48.33</td><td>51.61</td></tr><tr><td>Qwen-Image 2512</td><td>51.84</td><td>54.40</td><td>51.44</td><td></td><td></td><td>51.35</td></tr><tr><td>Boogu-Image 0.1 Base</td><td>50.38</td><td></td><td></td><td>47.80</td><td>47.75</td><td>51.32</td></tr><tr><td>LLaDA-Image Turbo</td><td></td><td>53.90</td><td>52.62</td><td>46.54</td><td>47.52</td><td>51.00</td></tr><tr><td></td><td>51.89</td><td>55.22</td><td>52.49</td><td>42.08</td><td>45.75</td><td>50.98</td></tr><tr><td>Z-Image</td><td>49.42</td><td>53.88</td><td>53.41</td><td>43.44</td><td>46.93</td><td>50.52</td></tr><tr><td>SenseNova U1.5 Preview</td><td>49.14</td><td>52.02</td><td>52.46</td><td>45.22</td><td>47.23</td><td>49.93</td></tr><tr><td>Qwen-Image</td><td>48.45</td><td>51.18</td><td>50.04</td><td>43.45</td><td>45.37</td><td>48.48</td></tr><tr><td>GLM-Image</td><td>49.86</td><td>49.98</td><td>47.49</td><td>44.25</td><td>44.67</td><td>47.86</td></tr></table>

LongText-Bench. LongText-Bench (Geng et al., 2025) evaluates whether a model can render long text strings legibly and correctly in both English and Chinese. Compared with short-word or single-region tests, it is more sensitive to omitted, repeated, substituted, and malformed characters as the amount of requested text increases. As reported in Tab. 5, LLaDA-Image achieves 0.923 and 0.913 on the English and Chinese subsets, respectively. The small cross-language gap indicates balanced bilingual behavior, although the scores remain below the strongest dedicated text-rendering models. Because this benchmark primarily emphasizes transcription correctness, we use it together with CVTG-2K rather than treating it as a standalone measure of the overall visual quality of text-centric images.

Table 4: Comparison of Text-to-Image Generation Performance on Qwen-Image-Bench (CN).
<table><tr><td>Type Model</td><td>Quality</td><td>Aesthetics</td><td>Alignment</td><td>Real-world Fidelity</td><td>Creative Generation</td><td>Overall</td></tr><tr><td>GPT-Image 2</td><td>58.65</td><td>67.53</td><td>65.85</td><td>57.38</td><td>75.23</td><td>64.69</td></tr><tr><td>Nano-Banana 2.0</td><td>54.77</td><td>61.08</td><td>62.40</td><td>54.28</td><td>67.05</td><td>59.82</td></tr><tr><td>GPT-Image 1.5</td><td>55.14</td><td>60.88</td><td>61.72</td><td>53.95</td><td>66.35</td><td>59.65</td></tr><tr><td>Nano-Banana Pro</td><td>55.67</td><td>60.26</td><td>61.25</td><td>54.07</td><td>66.23</td><td>59.45</td></tr><tr><td>Qwen-Image 2.0 Pro</td><td>54.39</td><td>58.67</td><td>59.28</td><td>51.83</td><td>64.94</td><td>57.84</td></tr><tr><td>Seedream 5.0</td><td>52.55</td><td>58.40</td><td>58.90</td><td>51.92</td><td>65.29</td><td>57.22</td></tr><tr><td>Clo-urce Seedream 4.5</td><td>54.41</td><td>58.72</td><td>57.31</td><td>51.69</td><td>60.64</td><td>56.78</td></tr><tr><td>Seedream 4.0</td><td>54.01</td><td>58.81</td><td>56.64</td><td>51.05</td><td>58.15</td><td>56.21</td></tr><tr><td>FLUX.2 Max</td><td>53.64</td><td>56.85</td><td>57.35</td><td>49.35</td><td>56.50</td><td>55.33</td></tr><tr><td>FLUX.2 Pro</td><td>52.30</td><td>56.94</td><td>57.01</td><td>47.29</td><td>56.18</td><td>54.57</td></tr><tr><td>GPT-Image 1</td><td>52.34</td><td>55.09</td><td>56.28</td><td>48.14</td><td>55.78</td><td>54.07</td></tr><tr><td>Imagen 4.0 Ultra</td><td>50.90</td><td>54.25</td><td>54.02</td><td>45.59</td><td>51.14</td><td>51.99</td></tr><tr><td>Imagen 4.0 Kling-Image 2.1</td><td>50.16 49.11</td><td>52.68 50.15</td><td>51.64 49.18</td><td>44.84 44.74</td><td>47.94</td><td>50.29</td></tr><tr><td>LLaDA-Image</td><td></td><td></td><td></td><td></td><td>44.67</td><td>48.26</td></tr><tr><td>Z-Image Turbo</td><td>52.92 52.30</td><td>56.92 55.24</td><td>54.87 53.76</td><td>44.86 45.38</td><td>52.10</td><td>53.38</td></tr><tr><td>Qwen-Image 2512</td><td>51.76</td><td>54.74</td><td>52.72</td><td>47.00</td><td>52.32 50.19</td><td>52.71</td></tr><tr><td>Z-Image</td><td>50.53</td><td>54.47</td><td>54.45</td><td>44.70</td><td>49.92</td><td>52.06</td></tr><tr><td>Opeo-ouce Boogu-Image 0.1 Turbo</td><td>51.24</td><td>53.50</td><td>53.54</td><td>46.11</td><td>48.91</td><td>51.78 51.53</td></tr><tr><td>Boogu-Image 0.1 Base</td><td>50.41</td><td>53.14</td><td>52.91</td><td>45.42</td><td>48.62</td><td></td></tr><tr><td>HunyuanImage 3.0</td><td>50.35</td><td>53.57</td><td>52.00</td><td>44.31</td><td></td><td>50.96</td></tr><tr><td>LLaDA-Image Turbo</td><td>51.29</td><td>52.02</td><td>52.32</td><td>42.91</td><td>49.12</td><td>50.81</td></tr><tr><td>SenseNova U1.5 Preview</td><td>49.19</td><td>51.87</td><td>52.61</td><td>45.11</td><td>47.14</td><td>50.27</td></tr><tr><td>Qwen-Image</td><td>48.44</td><td>52.25</td><td>50.72</td><td>43.16</td><td>48.80</td><td>50.25</td></tr><tr><td></td><td></td><td>50.64</td><td>47.90</td><td>44.69</td><td>47.30 45.23</td><td>49.23</td></tr><tr><td>GLM-Image</td><td>49.26</td><td></td><td></td><td></td><td></td><td>48.19</td></tr></table>

CVTG-2K. CVTG-2K (Tai et al., 2025) focuses on complex visual text generation in practical scenarios such as advertisements, posters, memes, and street views. Each prompt contains two to five text regions, making the benchmark useful for measuring both rendering accuracy and robustness as the number of textual elements grows. We report word accuracy for each region count and its average, normalized edit distance (NED) for transcription fidelity, and CLIPScore (Hessel et al., 2021) for image–prompt alignment. Tab. 6 shows that LLaDA-Image obtains an average word accuracy of 0.875, the second-highest among all compared models, together with an NED of 0.945 and a CLIPScore of 0.818. Its word accuracy remains stable as the number of text regions increases, decreasing from 0.892 on the two-region subset to 0.857 on the five-region subset.

Table 5: Comparison of Long-Text Rendering Ability on LongText-Bench.
<table><tr><td>Model</td><td>LongText-Bench-EN ↑</td><td>LongText-Bench-ZH ↑</td></tr><tr><td>Qwen-Image 2512</td><td>0.956</td><td>0.965</td></tr><tr><td>Boogu-Image 0.1 Base</td><td>0.952</td><td>0.969</td></tr><tr><td>Boogu-Image 0.1 Turbo</td><td>0.944</td><td>0.977</td></tr><tr><td>Qwen-Image</td><td>0.943</td><td>0.946</td></tr><tr><td>Z-Image</td><td>0.935</td><td>0.936</td></tr><tr><td>Z-Image Turbo</td><td>0.917</td><td>0.926</td></tr><tr><td>LLaDA-Image</td><td>0.923</td><td>0.913</td></tr><tr><td>LLaDA-Image Turbo</td><td>0.899</td><td>0.919</td></tr><tr><td>Seedream 3.0</td><td>0.896</td><td>0.878</td></tr><tr><td>X-Omni</td><td>0.900</td><td>0.814</td></tr><tr><td>GPT-Image 1 [High]</td><td>0.956</td><td>0.619</td></tr><tr><td>BAGEL</td><td>0.373</td><td>0.310</td></tr><tr><td>OmniGen2</td><td>0.561</td><td>0.059</td></tr><tr><td>FLUX.1 [Dev]</td><td>0.607</td><td>0.005</td></tr><tr><td>Kolors 2.0</td><td>0.258</td><td>0.329</td></tr><tr><td>HiDream-I1 Full</td><td>0.543</td><td>0.024</td></tr><tr><td>BLIP3-0</td><td>0.021</td><td>0.018</td></tr><tr><td>Janus Pro</td><td>0.019</td><td>0.006</td></tr></table>

Table 6: Comparison of Text Rendering Ability on CVTG-2K Benchmark.
<table><tr><td rowspan="2">Model</td><td colspan="5">Word Accuracy ↑</td><td rowspan="2">NED↑</td><td rowspan="2">CLIPScore ↑</td></tr><tr><td>2 regions</td><td>3 regions</td><td>4 regions</td><td>5 regions</td><td>average</td></tr><tr><td>SenseNova U1.5 Preview</td><td>0.890</td><td>0.887</td><td>0.902</td><td>0.869</td><td>0.887</td><td>0.943</td><td>0.809</td></tr><tr><td>LLaDA-Image</td><td>0.892</td><td>0.878</td><td>0.882</td><td>0.857</td><td>0.875</td><td>0.945</td><td>0.818</td></tr><tr><td>Z-Image</td><td>0.901</td><td>0.872</td><td>0.865</td><td>0.851</td><td>0.867</td><td>0.937</td><td>0.797</td></tr><tr><td>LongCat-Image</td><td>0.912</td><td>0.873</td><td>0.855</td><td>0.831</td><td>0.865</td><td>0.936</td><td>0.785</td></tr><tr><td>Boogu-Image 0.1 Base</td><td>0.852</td><td>0.863</td><td>0.866</td><td>0.863</td><td>0.863</td><td>0.927</td><td>0.801</td></tr><tr><td>Z-Image Turbo</td><td>0.887</td><td>0.866</td><td>0.862</td><td>0.834</td><td>0.858</td><td>0.928</td><td>0.804</td></tr><tr><td>LLaDA-Image Turbo</td><td>0.846</td><td>0.843</td><td>0.835</td><td>0.840</td><td>0.840</td><td>0.938</td><td>0.818</td></tr><tr><td>Qwen-Image 2512</td><td>0.841</td><td>0.843</td><td>0.842</td><td>0.827</td><td>0.838</td><td>0.918</td><td>0.788</td></tr><tr><td>Qwen-Image</td><td>0.837</td><td>0.836</td><td>0.831</td><td>0.816</td><td>0.829</td><td>0.912</td><td>0.802</td></tr><tr><td>Boogu-Image 0.1 Turbo</td><td>0.792</td><td>0.814</td><td>0.832</td><td>0.822</td><td>0.819</td><td>0.910</td><td>0.791</td></tr><tr><td>InternVL-U</td><td>0.729</td><td>0.660</td><td>0.618</td><td>0.549</td><td>0.623</td><td>0.804</td><td>0.816</td></tr><tr><td>Seedream 3.0</td><td>0.628</td><td>0.596</td><td>0.604</td><td>0.561</td><td>0.592</td><td>0.853</td><td>0.782</td></tr><tr><td>Lumina-DiMOO</td><td>0.723</td><td>0.646</td><td>0.571</td><td>0.505</td><td>0.590</td><td>0.805</td><td>0.831</td></tr><tr><td>FLUX.1 [Dev]</td><td>0.608</td><td>0.553</td><td>0.466</td><td>0.431</td><td>0.496</td><td>0.687</td><td>0.740</td></tr><tr><td>BAGEL</td><td>0.498</td><td>0.391</td><td>0.332</td><td>0.291</td><td>0.356</td><td>0.657</td><td>0.779</td></tr></table>

Legacy diagnostic benchmarks. As discussed in the Boogu-Image 0.1 technical report (Chen et al., 2026), GenEval and DPG-Bench do not reliably track the actual capabilities of modern text-to-image models: their rankings can diverge substantially from human-preference rankings, and their increasingly saturated scores make small differences difficult to interpret. We therefore do not treat either benchmark as primary evidence of overall generation quality. Nevertheless, both benchmarks have been widely reported in previous work. For historical continuity and comparison with the existing literature, we include their results below as supplementary references.

GenEval. GenEval (Ghosh et al., 2023) provides an object-centric diagnostic of compositional text-to-image generation. It decomposes prompt following into single-object generation, two-object composition, counting, color, spatial position, and attribute binding, thereby exposing failures that may be hidden by a single holistic alignment score. As shown in Tab. 7, LLaDA-Image reaches an overall score of 0.85. It attains 1.00 for single-object generation, 0.98 for two-object composition, and a tied-highest Attribute Binding score of 0.84 among the compared models. Its lower Counting score of 0.53 limits the aggregate result and exposes a specific compositional weakness. We report these category scores as a fine-grained reference under this traditional evaluation protocol, rather than interpreting the overall ranking as a faithful measure of general generation quality.

Table 7: Comparison of Text-to-Image Generation Ability on GenEval Benchmark.
<table><tr><td>Model</td><td>Single Object</td><td>Two Object</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attribute Binding</td><td>Overall↑</td></tr><tr><td>LLaDA 2.0 Uni</td><td>1.00</td><td>0.98</td><td>0.73</td><td>0.92</td><td>0.90</td><td>0.84</td><td>0.89</td></tr><tr><td>SenseNova U1.5 Preview</td><td>1.00</td><td>0.97</td><td>0.85</td><td>0.90</td><td>0.84</td><td>0.78</td><td>0.89</td></tr><tr><td>Lumina-DiMOO</td><td>1.00</td><td>0.94</td><td>0.85</td><td>0.89</td><td>0.85</td><td>0.76</td><td>0.88</td></tr><tr><td>Qwen-Image</td><td>0.99</td><td>0.92</td><td>0.89</td><td>0.88</td><td>0.76</td><td>0.77</td><td>0.87</td></tr><tr><td>LongCat-Image</td><td>0.99</td><td>0.98</td><td>0.86</td><td>0.86</td><td>0.75</td><td>0.73</td><td>0.87</td></tr><tr><td>Boogu-Image 0.1 Base</td><td>0.99</td><td>0.95</td><td>0.80</td><td>0.84</td><td>0.85</td><td>0.68</td><td>0.85</td></tr><tr><td>InternVL-U</td><td>0.99</td><td>0.94</td><td>0.74</td><td>0.91</td><td>0.77</td><td>0.74</td><td>0.85</td></tr><tr><td>LLaDA-Image</td><td>1.00</td><td>0.98</td><td>0.53</td><td>0.93</td><td>0.78</td><td>0.84</td><td>0.85</td></tr><tr><td>Seedream 3.0</td><td>0.99</td><td>0.96</td><td>0.91</td><td>0.93</td><td>0.47</td><td>0.80</td><td>0.84</td></tr><tr><td>Boogu-Image 0.1 Turbo</td><td>1.00</td><td>0.97</td><td>0.85</td><td>0.76</td><td>0.86</td><td>0.60</td><td>0.84</td></tr><tr><td>Z-Image</td><td>1.00</td><td>0.94</td><td>0.78</td><td>0.93</td><td>0.62</td><td>0.77</td><td>0.84</td></tr><tr><td>NextFlow</td><td>0.98</td><td>0.92</td><td>0.73</td><td>0.90</td><td>0.77</td><td>0.69</td><td>0.83</td></tr><tr><td>Z-Image Turbo</td><td>1.00</td><td>0.95</td><td>0.77</td><td>0.89</td><td>0.65</td><td>0.68</td><td>0.82</td></tr><tr><td>LLaDA-Image Turbo</td><td>1.00</td><td>0.99</td><td>0.42</td><td>0.93</td><td>0.78</td><td>0.80</td><td>0.82</td></tr><tr><td>BAGEL</td><td>0.99</td><td>0.94</td><td>0.81</td><td>0.88</td><td>0.64</td><td>0.63</td><td>0.82</td></tr><tr><td>Janus Pro</td><td>0.99</td><td>0.89</td><td>0.59</td><td>0.90</td><td>0.79</td><td>0.66</td><td>0.80</td></tr><tr><td>OmniGen2</td><td>1.00</td><td>0.95</td><td>0.64</td><td>0.88</td><td>0.55</td><td>0.76</td><td>0.80</td></tr><tr><td>Qwen-Image 2512</td><td>1.00</td><td>0.92</td><td>0.57</td><td>0.90</td><td>0.47</td><td>0.66</td><td>0.75</td></tr><tr><td>HunyuanImage 3.0</td><td>1.00</td><td>0.92</td><td>0.48</td><td>0.82</td><td>0.42</td><td>0.63</td><td>0.72</td></tr><tr><td>Lumina-mGPT 2.0</td><td>0.99</td><td>0.87</td><td>0.44</td><td>0.85</td><td>0.44</td><td>0.54</td><td>0.69</td></tr><tr><td>FLUX.1 [Dev]</td><td>0.98</td><td>0.81</td><td>0.74</td><td>0.79</td><td>0.22</td><td>0.45</td><td>0.66</td></tr><tr><td>MMaDA</td><td>0.99</td><td>0.76</td><td>0.61</td><td>0.84</td><td>0.20</td><td>0.37</td><td>0.63</td></tr><tr><td>Emu3-Gen</td><td>0.98</td><td>0.71</td><td>0.34</td><td>0.81</td><td>0.17</td><td>0.21</td><td>0.54</td></tr></table>

DPG-Bench. DPG-Bench (Hu et al., 2024) evaluates dense prompt following using 1,065 long, structurally complex descriptions. Its fine-grained evaluation decomposes a prompt into global, entity, attribute, relation, and other constraints, which complements GenEval’s concise object-centric templates with a more realistic test of verbose instruction interpretation. Tab. 8 reports an overall score of 87.48 for LLaDA-Image. The model remains consistently strong across the Entity, Attribute, Relation, and Other dimensions, with scores of 92.31, 92.37, 92.55, and 91.96, respectively. This balanced profile places it ahead of several strong baselines, including LLaDA-o, LongCat-Image, and HunyuanImage 3.0. We nevertheless include these scores as a reference for comparison with prior work, rather than as a definitive measure of the model’s practical generation capability.

Table 8: Comparison of Text-to-Image Generation Ability on DPG Benchmark.
<table><tr><td>Model</td><td>Global</td><td>Entity</td><td>Attribute</td><td>Relation</td><td>Other</td><td>Overall↑</td></tr><tr><td>LLaDA-Image Turbo</td><td>92.23</td><td>91.86</td><td>93.35</td><td>91.01</td><td>91.53</td><td>88.55</td></tr><tr><td>Boogu-Image 0.1 Turbo</td><td>88.54</td><td>91.67</td><td>92.19</td><td>93.20</td><td>93.83</td><td>88.35</td></tr><tr><td>Qwen-Image</td><td>91.32</td><td>91.56</td><td>92.02</td><td>94.31</td><td>92.73</td><td>88.32</td></tr><tr><td>Seedream 3.0</td><td>94.31</td><td>92.65</td><td>91.36</td><td>92.78</td><td>88.24</td><td>88.27</td></tr><tr><td>Z-Image</td><td>93.39</td><td>91.22</td><td>93.16</td><td>92.22</td><td>91.52</td><td>88.14</td></tr><tr><td>LLaDA 2.0 Uni</td><td>91.14</td><td>93.55</td><td>91.98</td><td>92.17</td><td>93.18</td><td>87.76</td></tr><tr><td>LLaDA-Image</td><td>90.86</td><td>92.31</td><td>92.37</td><td>92.55</td><td>91.96</td><td>87.48</td></tr><tr><td>SenseNova U1.5 Preview</td><td>85.81</td><td>91.97</td><td>90.95</td><td>93.48</td><td>92.53</td><td>87.26</td></tr><tr><td>Qwen-Image 2512</td><td>89.04</td><td>91.91</td><td>92.39</td><td>90.85</td><td>93.07</td><td>87.20</td></tr><tr><td>Lumina-Image 2.0</td><td>86.63</td><td>91.97</td><td>90.20</td><td>94.85</td><td>84.80</td><td>87.20</td></tr><tr><td>Boogu-Image 0.1 Base</td><td>89.33</td><td>90.64</td><td>92.22</td><td>93.72</td><td>93.32</td><td>87.13</td></tr><tr><td>LLaDA-0</td><td>92.91</td><td>93.30</td><td>90.40</td><td>91.75</td><td>92.79</td><td>87.04</td></tr><tr><td>LongCat-Image</td><td>89.10</td><td>92.54</td><td>92.00</td><td>93.28</td><td>87.50</td><td>86.80</td></tr><tr><td>HunyuanImage 3.0</td><td>92.12</td><td>92.53</td><td>89.13</td><td>92.13</td><td>91.92</td><td>86.10</td></tr><tr><td>Lumina-DiMOO</td><td>81.46</td><td>92.08</td><td>88.98</td><td>94.31</td><td>82.00</td><td>86.04</td></tr><tr><td>NextFlow</td><td>92.40</td><td>90.05</td><td>90.51</td><td>92.72</td><td>91.14</td><td>86.00</td></tr><tr><td>InternVL-U</td><td>90.39</td><td>90.78</td><td>90.68</td><td>90.29</td><td>88.77</td><td>85.18</td></tr><tr><td>BAGEL</td><td>88.94</td><td>90.37</td><td>91.29</td><td>90.82</td><td>88.67</td><td>85.07</td></tr><tr><td>Z-Image Turbo</td><td>91.29</td><td>89.59</td><td>90.14</td><td>92.16</td><td>88.68</td><td>84.86</td></tr><tr><td>Janus Pro</td><td>86.90</td><td>88.90</td><td>89.40</td><td>89.32</td><td>89.48</td><td>84.19</td></tr><tr><td>FLUX.1 [Dev]</td><td>74.35</td><td>90.00</td><td>88.96</td><td>90.87</td><td>88.33</td><td>83.84</td></tr><tr><td>OmniGen2</td><td>88.81</td><td>88.83</td><td>90.18</td><td>89.37</td><td>90.27</td><td>83.57</td></tr><tr><td>Emu3-Gen</td><td>85.21</td><td>86.68</td><td>86.84</td><td>90.22</td><td>83.15</td><td>80.60</td></tr><tr><td>MMaDA</td><td>77.81</td><td>78.48</td><td>81.74</td><td>84.79</td><td>63.20</td><td>69.97</td></tr></table>

## 5.2 Image Editing Generation

Compared systems and sources. The editing comparison covers SenseNova U1.5 Preview (SenseNova Team, 2026), FireRed-Image-Edit (Super Intelligence Team, 2026), Qwen-Image-Edit 2511 and Qwen-Image-Edit 2509 (Qwen Team, 2025c;b), Seedream 4.5 and 4.0 (ByteDance Seed, 2025b;a), Nano-Banana Pro and Nano-Banana (Google DeepMind, 2025c;b), LongCat-Image-Edit (Meituan LongCat Team et al., 2025), Z-Image-Edit (Z-Image Team et al., 2025), Step1X-Edit v1.2 (Liu et al., 2025b), FLUX.2 [Dev] (Black Forest Labs, 2026), and LLaDA 2.0 Uni (Inclusion AI et al., 2026).

GEdit-Bench. We evaluate general instruction-based image editing on the English and Chinese tracks of GEdit-Bench (Liu et al., 2025b), which contains 606 real-world editing cases spanning 11 task categories. The benchmark separately measures semantic consistency and instruction following (G<sub>SC</sub>), perceptual quality and naturalness (G ), and their overall assessment (G ). This decomposition is particularly relevant to a unified generation-and-editing model: a successful output must execute the requested change while retaining the visual plausibility and unedited content of the reference image. As reported in Tab. 9, LLaDA-Image achieves overall scores of 7.336 and 7.294 on GEdit-Bench-EN and GEdit-Bench-CN, respectively. These results confirm that a single unified checkpoint can support bilingual image editing while retaining its text-to-image capability. The component scores also reveal that semantic consistency is more competitive than perceptual quality, particularly on the English track (8.043 versus 7.182). Closing this perceptual-quality gap relative to specialized editing models remains an important direction for future refinement.

Table 9: Comparison of Image Editing Performance on GEdit-Bench.
<table><tr><td rowspan="2">Model</td><td colspan="3">GEdit-Bench-EN</td><td colspan="3">GEdit-Bench-CN</td></tr><tr><td>Gsc ↑</td><td>GPQ ↑</td><td>Go ↑</td><td>Gsc ↑</td><td>GPQ ↑</td><td>Go ↑</td></tr><tr><td>SenseNova U1.5 Preview</td><td></td><td></td><td>8.172</td><td></td><td></td><td>8.051</td></tr><tr><td>FireRed-Image-Edit</td><td>8.363</td><td>8.245</td><td>7.943</td><td>8.287</td><td>8.227</td><td>7.887</td></tr><tr><td>Qwen-Image-Edit 2511</td><td>8.297</td><td>8.202</td><td>7.877</td><td>8.252</td><td>8.134</td><td>7.819</td></tr><tr><td>Seedream 4.5</td><td>8.268</td><td>8.167</td><td>7.820</td><td>8.254</td><td>8.167</td><td>7.800</td></tr><tr><td>Nano-Banana Pro</td><td>8.102</td><td>8.344</td><td>7.738</td><td>8.135</td><td>8.306</td><td>7.799</td></tr><tr><td>LongCat-Image-Edit</td><td>8.128</td><td>8.177</td><td>7.748</td><td>8.141</td><td>8.117</td><td>7.731</td></tr><tr><td>Seedream 4.0</td><td>8.143</td><td>8.124</td><td>7.701</td><td>8.159</td><td>8.074</td><td>7.692</td></tr><tr><td>Z-Image-Edit</td><td>8.11</td><td>7.72</td><td>7.57</td><td>8.03</td><td>7.80</td><td>7.54</td></tr><tr><td>Qwen-Image-Edit 2509</td><td>7.974</td><td>7.714</td><td>7.480</td><td>7.988</td><td>7.679</td><td>7.467</td></tr><tr><td>FLUX.2 [Dev]</td><td>7.835</td><td>8.064</td><td>7.413</td><td>7.697</td><td>8.046</td><td>7.278</td></tr><tr><td>Nano-Banana</td><td>7.396</td><td>8.454</td><td>7.291</td><td>7.540</td><td>8.424</td><td>7.399</td></tr><tr><td>LLaDA-Image</td><td>8.043</td><td>7.182</td><td>7.336</td><td>7.707</td><td>7.594</td><td>7.294</td></tr><tr><td>LLaDA-Image Turbo</td><td>7.533</td><td>7.200</td><td>7.024</td><td>7.252</td><td>7.292</td><td>6.898</td></tr><tr><td>LLaDA 2.0 Uni</td><td>6.68</td><td>7.52</td><td>6.61</td><td>6.63</td><td>7.67</td><td>6.66</td></tr></table>

## 6 Conclusion and Future Directions

This report presents LLaDA-Image, a 6B Diffusion Transformer trained from scratch for unified text-toimage generation and instruction-guided editing. Its open recipe combines image-only pre-training and mid-training, a dLLM-based VLM with RQA–connector conditioning, a real-data-dominant progressive curriculum, and TwinFlow distillation. Of 220M generation samples, 98% are real and more than 90% support image-only training; real images remain above 70% in paired SFT. SigLIP-VQ and clean VAE latents preserve reference information, while parameter-free RMSNorm and Muon stabilize training. LLaDA-Image sets the open-source state of the art on both Qwen-Image-Bench tracks, with strong bilingual text rendering and editing; LLaDA-Image Turbo reduces inference to 2–4 steps. We release the weights, code, and recipes to make this path inspectable and reusable.

Remaining challenges include broader world knowledge and more reliable long and multi-region text rendering. Broader knowledge coverage would improve the faithful depiction of specialized subjects, culturally specific concepts, and prompts whose visual realization depends on factual context. Text rendering can also become less stable as the number of regions, the length of the requested content, and the complexity of the layout increase. Training currently reaches a nominal 1024<sup>2</sup> pixel budget, so native 2K generation remains an extension rather than an evaluated capability. Future work will strengthen grounding and typography, broaden the coverage of knowledge-intensive concepts, extend the curriculum to native 2K, and explore agentic generation through closed-loop planning, evaluation, and editing.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/abs/2511.21631.

Black Forest Labs. FLUX.1 [dev] Model Card. https://huggingface.co/black-forest-labs/FLUX.1-dev, 2024.

Black Forest Labs. FLUX.2: Next-Generation Image Generation. https://bfl.ai/models/flux-2, 2026.

ByteDance Seed. Seedream 4.0. https://seed.bytedance.com/en/seedream4 0, 2025a.

ByteDance Seed. Seedream 4.5. https://seed.bytedance.com/en/seedream4 5, 2025b.

ByteDance Seed. Seedream 5.0 Pro. https://seed.bytedance.com/en/seedream5 0 pro, 2026.

Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer. arXiv preprint arXiv:2505.22705, 2025. URL https://arxiv.org/abs/2505.22705.

Shuo Cao, Nan Ma, Jiayang Li, Xiaohui Li, Lihao Shao, Kaiwen Zhu, Yu Zhou, Yuandong Pu, Jiarui Wu, Jiaquan Wang, Bo Qu, Wenhai Wang, Yu Qiao, Dajuin Yao, and Yihao Liu. Artimuse: Fine-grained image aesthetics assessment with joint scoring and expert-level understanding. arXiv preprint arXiv:2507.14533, 2025. URL https://arxiv.org/abs/2507.14533.

Guoxuan Chen, Chufeng Xiao, Haoran Yang, Siyue Xie, Binxiao Huang, Ming Zhang, Cheuk Him Chau, Xinyu Fu, Yingzhao Lian, et al. Boogu-Image-0.1: Boosting Open Agentic Multimodal Generation via Understanding under a Minimal Budget. arXiv preprint arXiv:2607.13125, 2026. URL https://arxiv.org/ abs/2607.13125.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset. arXiv preprint arXiv:2505.09568, 2025a. URL https://arxiv.org/abs/2505.09568.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are We on the Right Way for Evaluating Large Vision-Language Models? Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/abs/2403. 20330.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling. arXiv preprint arXiv:2501.17811, 2025b. URL https://arxiv.org/abs/2501.17811.

Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models. In Proceedings of the IEEE/CVF International Conference on Computer Vi sion (ICCV), 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Cheng SimpleVQA Multimodal Factuality Evaluation for Multimodal Large Language Models ICCV 2025 paper.html.

Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. Twinflow: Realizing one-step generation on large models with self-adversarial flows. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=fBc9v8CVvm.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. URL https://arxiv.org/abs/2505.14683.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi,¨ Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings ofthe 41st International Conference on Machine Learning, 2024. URL https: //arxiv.org/abs/2403.03206.

Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 Technical Report. arXiv preprint arXiv:2504.11346, 2025. URL https://arxiv.org/abs/2504.11346.

Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again. arXiv preprint arXiv:2507.22058, 2025. URL https://arxiv.org/abs/2507. 22058.

Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. arXiv preprint arXiv:2310.11513, 2023. URL https://arxiv.org/abs/ 2310.11513.

Google DeepMind. Imagen 4. https://deepmind.google/models/imagen/, 2025a.

Google DeepMind. Image Editing in Gemini Just Got a Major Upgrade. https://blog.google/ products-and-platforms/products/gemini/updated-image-editing-model/, 2025b.

Google DeepMind. Introducing Nano Banana Pro. https://blog.google/innovation-and-ai/products/ nano-banana-pro/, 2025c.

Google DeepMind. Nano Banana 2: Combining Pro Capabilities with Lightning-Fast Speed. https://blog. google/innovation-and-ai/technology/ai/nano-banana-2/, 2026.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Guan HallusionBench An Advanced Diagnostic Suite for Entangled Language Hallucination and CVPR 2024 paper.html.

Junlin Han, Shengbang Tong, David Fan, Minghao Chen, Philip Torr, Filippos Kokkinos, and Mike Lewis. Towards physics of multimodal pretraining: Knowledge flow, modality synergy, early unification, and recipes. arXiv preprint arXiv:2608.05000, 2026. URL https://arxiv.org/abs/2608.05000.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, 2021. URL https://arxiv.org/abs/2104.08718.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment. arXiv preprint arXiv:2403.05135, 2024. URL https://arxiv.org/ abs/2403.05135.

Inclusion AI, Tiwei Bie, Haoxing Chen, Tieyuan Chen, Zhenglin Cheng, Long Cui, Kai Gan, Zhicheng Huang, Zhenzhong Lan, Haoquan Li, et al. LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation with Diffusion Large Language Model. arXiv preprint arXiv:2604.20796, 2026. URL https: //arxiv.org/abs/2604.20796.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, 2018. URL https://arxiv.org/abs/1803.05407.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. URL https://link.springer.com/chapter/10.1007/978-3-319-46493-0 15.

Kuaishou. Kling AI Image 2.1. https://kling.ai/app/image, 2025.

Kuaishou Kolors Team. Kolors 2.0. https://kolors.kuaishou.com/, 2025.

Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, et al. Vl-rewardbench: A challenging benchmark for vision-language generative reward models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Li VL-RewardBench A Challenging Benchmark for Vision-Language Generative Reward Models CVPR 2025 paper.html.

Niantong Li, Guangzheng Hu, Weixu Qiao, Ying Ba, Qichen Hong, Shijun Shen, Jinlin Wang, Fan Zhou, Jianye Kang, Xin Shang, et al. Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation. arXiv preprint arXiv:2605.28091, 2026. URL https://arxiv.org/abs/2605.28091.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://arxiv.org/ abs/2210.02747.

Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Hongsheng Li, and Steven Hoi. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/ 2511.22677.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025a. URL https://arxiv.org/abs/2502.16982.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1X-Edit: A Practical Framework for General Image Editing. arXiv preprint arXiv:2504.17761, 2025b. URL https://arxiv.org/abs/2504.17761.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is Your Multi-Modal Model an All-around Player? In Proceedings ofthe European Conference on Computer Vision (ECCV), 2024a. URL https://arxiv.org/abs/2307.06281.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models. Science China Information Sciences, 2024b. URL https://link.springer.com/article/10.1007/ s11432-024-4235-6.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1711.05101.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. arXiv preprint arXiv:2310.02255, 2023. URL https://arxiv.org/abs/2310.02255.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Findings of the Associationfor Computational Linguistics: ACL 2022, 2022. URL https://aclanthology.org/2022.findings-acl.177/.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A Dataset for VQA on Document Images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021. URL https://openaccess.thecvf.com/content/WACV2021/html/Mathew DocVQA A Dataset for VQA on Document Images WACV 2021 paper.html.

Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Info-\` graphicVQA. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022. URL https://openaccess.thecvf.com/content/WACV2022/html/Mathew InfographicVQA WACV 2022 paper.html.

Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. LongCat-Image Technical Report. arXiv preprint arXiv:2512.07584, 2025. URL https://arxiv.org/abs/2512.07584.

OpenAI. Introducing Our Latest Image Generation Model in the API. https://openai.com/index/ image-generation-api/, 2025a.

OpenAI. The New ChatGPT Images Is Here. https://openai.com/index/new-chatgpt-images-is-here/, 2025b.

OpenAI. Introducing ChatGPT Images 2.0. https://openai.com/index/introducing-chatgpt-images-2-0/, 2026.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching CLIP to Count to Ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Paiss Teaching CLIP to Count to Ten ICCV 2023 paper.html.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2023. URL https://arxiv.org/abs/2212.09748.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, et al. WE-MATH: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning? In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025. URL https://aclanthology.org/2025.acl-long.983/.

Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, et al. Lumina-Image 2.0: A Unified and Efficient Image Generative Framework. arXiv preprint arXiv:2503.21758, 2025. URL https://arxiv.org/abs/2503.21758.

Qwen Team. Qwen-Image-2512 Model Card. https://huggingface.co/Qwen/Qwen-Image-2512, 2025a.

Qwen Team. Qwen-Image-Edit-2509 Model Card. https://huggingface.co/Qwen/Qwen-Image-Edit-2509, 2025b.

Qwen Team. Qwen-Image-Edit-2511 Model Card. https://huggingface.co/Qwen/Qwen-Image-Edit-2511, 2025c.

Qwen Team. Qwen3.6-35B-A3B Model Card. https://huggingface.co/Qwen/Qwen3.6-35B-A3B, 2026.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution¨ image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. URL https://arxiv.org/abs/2112.10752.

SenseNova Team. SenseNova-U1.5-8B-MoT (Preview). https://github.com/OpenSenseNova/SenseNova-U1/ blob/main/docs/u1.5 preview.md, 2026.

Peng Sun, Zhenglin Cheng, Deyuan Liu, Jun Xie, Xinyi Shang, and Tao Lin. Three-body scattering for generative modeling. arXiv preprint arXiv:2607.18198, 2026a. URL https://arxiv.org/abs/2607.18198.

Peng Sun, Xinyi Shang, Tao Lin, and Zhiqiang Shen. Duality models: An embarrassingly simple one-step generation paradigm. arXiv preprint arXiv:2602.17682, 2026b. URL https://arxiv.org/abs/2602.17682.

Peng Sun, Jun Xie, and Tao Lin. Rethinking UMM visual generation: Masked modeling for efficient imageonly pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026c. URL https://arxiv.org/abs/2603.16139.

Super Intelligence Team. FireRed-Image-Edit-1.0 Technical Report. arXiv preprint arXiv:2602.13344, 2026. URL https://arxiv.org/abs/2602.13344.

Ying Tai, Nikai Du, Rui Xie, Zhennan Chen, Qian Wang, Zhengkai Jiang, Kai Zhang, and Jian Yang. Investigating text insulation and attention mechanisms for complex visual text generation. arXiv preprint arXiv:2503.23461, 2025. URL https://arxiv.org/abs/2503.23461.

Tencent Hunyuan Foundation Model Team. HunyuanImage 3.0 Technical Report. arXiv preprint arXiv:2509.23951, 2025. URL https://arxiv.org/abs/2509.23951.

Changyao Tian, Danni Yang, Guanzhou Chen, Erfei Cui, Zhaokai Wang, Yuchen Duan, Penghao Yin, Sitao Chen, Ganlin Yang, Mingxin Liu, et al. InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing. arXiv preprint arXiv:2603.09877, 2026. URL https://arxiv.org/abs/2603.09877.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset. Advances in Neural Information Processing Systems (NeurIPS), 2024a. URL https://proceedings.nips.cc/paper files/paper/ 2024/hash/ad0edc7d5fa1a783f063646968b7315b-Abstract-Datasets and Benchmarks Track.html.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-Token Prediction Is All You Need. arXiv preprint arXiv:2409.18869, 2024b. URL https://arxiv.org/abs/2409.18869.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324, 2025a. URL https://arxiv.org/abs/2508.02324.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. OmniGen2: Towards Instruction-Aligned Multimodal Generation. arXiv preprint arXiv:2506.18871, 2025b. URL https://arxiv.org/abs/2506.18871.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023. URL https://arxiv.org/abs/2312.14135.

xAI. Grok-1.5 Vision Preview. https://x.ai/news/grok-1.5v, 2024.

Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding. arXiv preprint arXiv:2510.06308, 2025a. URL https://arxiv.org/abs/2510.06308.

Yi Xin, Juncheng Yan, Qi Qin, Zhen Li, Dongyang Liu, Shicheng Li, Victor Shea-Jay Huang, Yupeng Zhou, Renrui Zhang, Le Zhuo, et al. Lumina-mGPT 2.0: Stand-Alone AutoRegressive Image Modeling. arXiv preprint arXiv:2507.17801, 2025b. URL https://arxiv.org/abs/2507.17801.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. MMaDA: Multimodal Large Diffusion Language Models. arXiv preprint arXiv:2505.15809, 2025. URL https://arxiv. org/abs/2505.15809.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fr¨ edo Durand, and William T.´ Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, volume 37, pp. 47455–47487, 2024. URL https://papers.nips.cc/paper files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html.

Zebin You, Xiaolu Zhang, Jun Zhou, Chongxuan Li, and Ji-Rong Wen. LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model. arXiv preprint arXiv:2603.01068, 2026. URL https://arxiv.org/abs/ 2603.01068.

Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/You Teaching Large Language Models to Regress Accurate Image Quality Scores CVPR 2025 paper.html.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, et al. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Yue MMMU A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for CVPR 2024 paper.html.

Z-Image Team, Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer. arXiv preprint arXiv:2511.22699, 2025. URL https: //arxiv.org/abs/2511.22699.

Z.ai. GLM-Image Model Card. https://huggingface.co/zai-org/GLM-Image, 2026.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Advances in Neural Information Processing Systems, 2019. URL https://arxiv.org/abs/1910.07467.

Huichao Zhang, Liao Qu, Yiheng Liu, Hang Chen, Yangyang Song, Yongsheng Dong, Shikun Sun, Xian Li, Xu Wang, Yi Jiang, et al. NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation. arXiv preprint arXiv:2601.02204, 2026. URL https://arxiv.org/abs/2601.02204.

Bing Zhao, Chenfei Wu, Deqing Li, Hao Meng, Jiahao Li, Jie Zhang, Jingren Zhou, Junyang Lin, Kaiyuan Gao, et al. Qwen-Image-2.0 Technical Report. arXiv preprint arXiv:2605.10730, 2026. URL https://arxiv. org/abs/2605.10730.

## A Aspect-Ratio Bucket Configurations

We introduce aspect-ratio buckets during image-only mid-training and retain them throughout supervised fine-tuning at two target pixel budgets. The terms “512<sup>2</sup>-pixel” and “1024<sup>2</sup>-pixel” refer to the approximate number of pixels in an image, not to a fixed square training resolution. Tab. 10 lists the exact configurations. The aspect ratio is defined as width divided by height, and each spatial configuration is reported as a (W, H) tuple in pixels.

Let an input image have width W, height H, and aspect ratio $a = W / H$ . The data loader maintains the set of K resolution buckets $\boldsymbol { \mathcal { B } } = \{ ( W _ { k } , H _ { k } ) \} _ { k = 1 } ^ { K }$ and assigns the image to the bucket with the smallest absolute difference in aspect ratio:

$$
k ^ { * } = \underset { 1 \leq k \leq K } { \arg \operatorname* { m i n } } \left| \frac { W } { H } - \frac { W _ { k } } { H _ { k } } \right| .\tag{17}
$$

After selecting $k ^ { * }$ via Eq. (17), we resize the image without changing its aspect ratio. For the selected bucket $\left( W _ { k ^ { * } } , H _ { k ^ { * } } \right)$ , the resize-to-fill scale is

$$
s = \operatorname* { m a x } \left( \frac { W _ { k ^ { * } } } { W } , \frac { H _ { k ^ { * } } } { H } \right) , \qquad ( W ^ { \prime } , H ^ { \prime } ) = ( s W , s H ) .\tag{18}
$$

In practice, the transformation in Eq. (18) is a downsampling step because training images are sufficiently large. One resized dimension matches its bucket dimension, whereas the other can be slightly larger. We crop only this excess region to obtain the exact target shape. Because $k ^ { * }$ has nearly the same aspect ratio as the source image, the overflow is typically only a few pixels; consequently, the crop removes little content and does not materially alter the image.

The buckets within each stage have approximately equal areas $W _ { k } H _ { k }$ . Consequently, different data-parallel ranks can process different aspect ratios while receiving similar numbers of image tokens. This prevents a single rank from producing a disproportionate memory peak or becoming a training-time straggler. Compared with forcing every sample into a square, the procedure also avoids geometric distortion and aggressive square cropping, while teaching the model to generate a broad range of output aspect ratios.

Table 10: Aspect-ratio bucket configurations at the 512<sup>2</sup>-pixel and 1024<sup>2</sup>-pixel budgets. Each tuple reports (W, H) in pixels.
<table><tr><td>Aspect Ratio (W /H)</td><td>5122-pixel (W, H)</td><td> ${ \bf 1 0 2 4 ^ { 2 } - p i x e l } \left( W , H \right)$ </td></tr><tr><td>Portrait (W / H &lt; 1)</td><td></td><td></td></tr><tr><td>0.25</td><td>(256,1024)</td><td>(512, 2048)</td></tr><tr><td>0.26</td><td>(256,992)</td><td>(512, 1984)</td></tr><tr><td>0.27</td><td>(256,960)</td><td>(512, 1920)</td></tr><tr><td>0.28</td><td>(256,928)</td><td>(512, 1856)</td></tr><tr><td>0.32</td><td>(288,896)</td><td>(576, 1792)</td></tr><tr><td>0.33</td><td>(288,864)</td><td>(576, 1728)</td></tr><tr><td>0.35</td><td>(288,832)</td><td>(576, 1664)</td></tr><tr><td>0.40</td><td>(320,800)</td><td>(640, 1600)</td></tr><tr><td>0.42</td><td>(320,768)</td><td>(640, 1536)</td></tr><tr><td>0.48</td><td>(352,736)</td><td>(704, 1472)</td></tr><tr><td>0.50</td><td>(352,704)</td><td>(704, 1408)</td></tr><tr><td>0.52</td><td>(352, 672)</td><td>(704, 1344)</td></tr><tr><td>0.57 0.60</td><td>(384,672)</td><td>(768, 1344)</td></tr><tr><td>0.68</td><td>(384, 640) (416, 608)</td><td>(768, 1280) (832, 1216)</td></tr><tr><td>0.72</td><td>(416,576)</td><td>(832, 1152)</td></tr><tr><td>0.78</td><td></td><td>(896, 1152)</td></tr><tr><td>0.82</td><td>(448,576)</td><td>(896, 1088)</td></tr><tr><td>0.88</td><td>(448,544)</td><td></td></tr><tr><td>0.94</td><td>(480,544)</td><td>(960, 1088) (960, 1024)</td></tr><tr><td></td><td>(480,512)</td><td></td></tr><tr><td>Square (W/H = 1)</td><td></td><td></td></tr><tr><td>1.00</td><td>(512,512)</td><td>(1024,1024)</td></tr><tr><td>Landscape (W / H &gt; 1) 1.07</td><td></td><td></td></tr><tr><td>1.13</td><td>(512,480)</td><td>(1024,960)</td></tr><tr><td>1.21</td><td>(544, 480)</td><td>(1088,960)</td></tr><tr><td>1.29</td><td>(544, 448)</td><td>(1088,896)</td></tr><tr><td>1.38</td><td>(576,448)</td><td>(1152,896)</td></tr><tr><td>1.46</td><td>(576, 416)</td><td>(1152,832)</td></tr><tr><td></td><td>(608,416)</td><td>(1216,832)</td></tr><tr><td>1.67</td><td>(640,384)</td><td>(1280,768)</td></tr><tr><td>1.75 2.00</td><td>(672,384)</td><td>(1344,768)</td></tr><tr><td>2.09</td><td>(704,352)</td><td>(1408,704)</td></tr><tr><td>2.40</td><td>(736,352)</td><td>(1472,704)</td></tr><tr><td></td><td>(768,320)</td><td>(1536,640)</td></tr><tr><td>2.50</td><td>(800,320)</td><td>(1600,640)</td></tr><tr><td>2.89</td><td>(832,288)</td><td>(1664,576)</td></tr><tr><td>3.00</td><td>(864,288)</td><td>(1728,576)</td></tr><tr><td>3.11</td><td>(896,288)</td><td>(1792,576)</td></tr><tr><td>3.62</td><td>(928,256)</td><td>(1856,512)</td></tr><tr><td>3.75</td><td>(960,256)</td><td>(1920,512)</td></tr><tr><td>3.88</td><td>(992, 256)</td><td>(1984,512)</td></tr><tr><td>4.00</td><td>(1024,256)</td><td>(2048,512)</td></tr></table>

## B CoT Supervised Fine-tuning and Reinforcement Learning

CoT supervised fine-tuning (Sec. 4.1) is the only stage that updates the understanding backbone, so we verify its effect separately before the generation stages build on it.

Benchmarks. We evaluate across 18 multimodal understanding benchmarks, focusing on three core capabilities: general VQA, reasoning, and OCR/document understanding:

• General Tasks: MMStar (Chen et al., 2024), MMBench (Liu et al., 2024a) (English and Chinese dev splits), HallusionBench (Guan et al., 2024), RealWorldQA (xAI, 2024) and SimpleVQA (Cheng et al., 2025).

• Reasoning Tasks: MMMU (Yue et al., 2024) (val), MathVista (Lu et al., 2023) (mini), We-Math (Qiao et al., 2025), and MathVision (Wang et al., 2024a) (mini).

• OCR&Chart Tasks: ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021) (val), InfoVQA (Mathew et al., 2022), OCRBench (Liu et al., 2024b), and AI2D (Kembhavi et al., 2016) (test).

• Other Multimodal Tasks: CountBenchQA (Paiss et al., 2023), VL-RewardBench (Li et al., 2025), and V<sup>∗</sup> (Wu & Xie, 2023).

Table 11: Multimodal understanding results before and after CoT supervised fine-tuning. The LLaDA 2.0 Uni column reproduces the scores reported in its technical report (Inclusion AI et al., 2026); the CoT SFT column is evaluated by us under the protocol described in App. B.
<table><tr><td>Benchmark</td><td>LLaDA 2.0 Uni</td><td>+ CoT SFT</td></tr><tr><td>General Tasks</td><td></td><td></td></tr><tr><td>MMStar</td><td>64.1 81.5</td><td>64.1 86.1</td></tr><tr><td>MMBenchdev-EN  $\mathrm { M M B e n c h } _ { \mathrm { d e v - C N } }$ </td><td>81.2</td><td>86.2</td></tr><tr><td>HallusionBench RealWorldQA</td><td>50.2 66.7</td><td>51.3 64.1</td></tr><tr><td>SimpleVQA</td><td>44.0</td><td>42.1</td></tr><tr><td>Reasoning Tasks</td><td></td><td></td></tr><tr><td>MMMUval</td><td>50.1</td><td>51.3 68.4</td></tr><tr><td>MathVistamini MathVisionmini</td><td>68.1 26.7</td><td>24.3</td></tr><tr><td>We-Math</td><td>29.3</td><td>30.9</td></tr><tr><td>OCR &amp; Chart Tasks</td><td></td><td></td></tr><tr><td>ChartQA</td><td>80.1</td><td>82.8</td></tr><tr><td>DocVQAval</td><td>89.5</td><td>88.2</td></tr><tr><td>InfoVQA</td><td>70.1</td><td>70.4</td></tr><tr><td>OCRBench</td><td>75.7</td><td>77.6</td></tr><tr><td>AI2Dtest</td><td>82.0</td><td>80.4</td></tr><tr><td>Other Tasks</td><td></td><td></td></tr><tr><td>CountBenchQA</td><td>86.0</td><td>84.2</td></tr><tr><td>VL-RewardBench</td><td>47.8</td><td>55.7</td></tr><tr><td></td><td></td><td></td></tr><tr><td>V*</td><td>61.8</td><td>61.8</td></tr></table>

## C Additional Qualitative Comparison

一尊汉代陶制侍女俑独立摆放在博物馆展台上，人物双手收于身前，长袍垂直落下，造型简洁克制，面部五官以古朴方式塑造，表面残留少量褪色的朱红、黑色和彩绘颜料陶土呈暖灰褐色，可见细小裂纹、土沁与局部缺损。柔和自然光从左侧落下，背景为温暖的灰米色墙面，突出人物轮廓与历史痕迹，真实考古摄影，垂直构图

![](images/b94c3ba8f6d4096b2f251cc6425c33d706c88c42bf98a8f2ec7dea6e4b38ab05.jpg)  
Input Prompt  
LLaDA-Image Turbo

![](images/bb07d654ed0e79b4c67ec58718f2735436cbb3da4a6595419f99ff6e97235414.jpg)  
SenseNova U1.5 Preview

![](images/21a281aef7cc58192f03b2a0bddc2110b0fd48fa23de9a0a35ae2531f60f9329.jpg)  
Qwen-Image 2512

明<sub>1368-1644</sub>年掐丝珐琅海棠式双耳瓶，浅中摆放。器物通体呈海棠四瓣花形，由口足轮廓一致，形似四个铜瓶接连合成。瓶四瓣花形，边缘鎏金。颈部两侧对称装饰两只立体鎏金螭龙耳，龙身卷曲，龙首探向瓶口。颈部与腹部之间有一圈凸起的变形莲瓣纹饰带。瓶身主体以蓝绿色为地，表面布满掐丝珐琅工艺烧制的垂拱式夔龙纹，纹饰线条细密，色彩包含蓝、绿、红、黄等，间以金色丝线勾勒， 感古朴。瓶底为四瓣花形圈足，足边亦见鎏金痕迹。整体器型端庄，工艺繁复，色彩斑斓。

Input Prompt

![](images/471fd874b20e3e02b53fc54fbd59028ff57147de3f8cd460c2e463fdb3032ca0.jpg)  
LLaDA-Image Turbo

![](images/ac800fe0b4c9638341fb9e748ee43b12765500f38ec4f698d8e25216d4956690.jpg)  
SenseNova U1.5 Preview

![](images/496ae83a4e86c743ee782697c7da113daeccdcfef2fdde24eb5b5af5a2c32d37.jpg)  
Qwen-Image 2512

![](images/de4f79c83f704b068b92736713d3b4e3771970b1574ff1d7b69cbf545b840f75.jpg)  
Z-Image Turbo

![](images/c921d1a7767d7ae57b7734da8077c4ff42aca785836a2e77d6b3a31575409c47.jpg)  
Z-Image Turbo  
GPT-Image 2

![](images/c25f22d06ee9b160556b445c058522c7d2eb90560b0156c9e7cc074e4d565854.jpg)

![](images/d9d6454ea3398ed5677305cd864ed3054627da39efbc5ad474dd824f1035809c.jpg)  
GPT-Image 2

![](images/e96ee4d7332f0d7ce1a4b312fb07b5158826bc273349dfa2f985f4df706b49f1.jpg)  
Boogu-Image 0.1 Turbo

![](images/125a6579e46eb7223c0bd784b4053daa47cb413628d4312974850d7db9fb0011.jpg)  
Gemini 3.1 Flash Image

![](images/2307c7597542ce5cf8e31ad078555781df37a02b83331007b3864e5308c32b83.jpg)  
Boogu-Image 0.1 Turbo

![](images/e632180d9e93e2fb409eef6597fabbe050987a171a148a0bceab9a17b7a5bbad.jpg)  
Gemini 3.1 Flash Image

# Additional Qualitative Comparison (Continued)

张冰 川观 景 区 的 性旅 行游 <sup>游客</sup>十岁<sub>，站</sub>午 主 体 一 位 三 <sup>右</sup><sub>木</sub><sup>容</sup><sub>被</sub>嘴<sup>暖</sup><sub>偏</sub> 哑 光  
夹 T恤、 血、 深 灰  
<sup>，</sup><sub>巨</sub> 个 个小 型 登缓 向 山  
川表 色， 可 可以 然 <sup>裂</sup><sub>与</sub> 隙<sup>，</sup><sub>蓝</sub> 山 为 <sup>石</sup><sub>于</sub> <sup>积雪</sup><sub>峰附</sub>  
。 透，  
平台周围还有护栏和少 ， 整体规  
感明确， 同时保持普通游客旅行纪 照的  
然感。

Input Prompt

![](images/d437d6e82cac0b263d8386e8e005dfc9b348ca65c2c81431cabd338bc766b485.jpg)

![](images/aa7aec6388a66b8c4b7ffd892e3ec87dd0efcffe4647e3019f5da9df6b7b4607.jpg)

![](images/a4ccb071c9bfa24f8f5d66003f065d605df3cb3e35a6022c8d9a0b09807434bc.jpg)  
LLaDA-Image Turbo  
SenseNova U1.5 Preview

![](images/3fb8c834a47d670891212297ddd7dc88e9d70086d92303ea3165d347870a6ce7.jpg)  
Qwen-Image 2512  
Z-Image Turbo

![](images/a4707de790e1a436c5e2776c97f8a00cc5168a8bebefcf0f0074f640346e2f9c.jpg)  
GPT-Image 2

张写实风格的欧洲山地小镇风光摄影作品，拍摄于清晨。画面主体是一座依山而建的地中海小镇，成排的浅黄色、米白色和暖橙色石砌房屋顺着山坡层层向上延伸，屋顶多为红褐色瓦片，窗户尺寸不一，部分窗台摆放着绿植和花盆。狭窄弯曲的石板街巷在房屋之间穿行，几段台阶沿着坡地向上连接不同高度的平台。画面下方是小镇边缘的露台和护墙，远处能看到蓝灰色群山和一小片平静水面。晨光从侧面照亮建筑立面，墙面有清楚但柔和的明暗变化，整体色彩自然克制，空气清新，像真实旅行中拍到的欧洲山城景象。

Input Prompt

![](images/1a67ce37f4c142791f337dffcfdc68705e0780ac890e229d1e6da02409df4221.jpg)  
LLaDA-Image Turbo

![](images/3243e8ce201ee746f033916b949cf928ba286115f5b7bb8293e7e122c5999029.jpg)  
Z-Image Turbo

![](images/63f3e9d4f5499711fde07d95b7fba92e4e201dce1e2e7ae3fccfb66c200be894.jpg)  
SenseNova U1.5 Preview

![](images/449d06377dc92c382c97307c390c00ce18b6908c098caaa1a479803bb0d86ab7.jpg)  
Qwen-Image 2512

![](images/8cddaa52d3ce53acbb6ed72beb81992131c69d84516cf3dadf8b164e45cc549f.jpg)  
GPT-Image 2

![](images/1628f7925d16066b0042a30049f755ba7d4aad02b87b1ad1d4a606df1918fbac.jpg)  
Boogu-Image 0.1 Turbo

![](images/bcebff61ee3c5c47e5d50475f838fc88ed6da4b00b79280766358c3b15942d64.jpg)  
Gemini 3.1 Flash Image

![](images/332000197a781fda7d7e62e72f34cc452766aff27e12e5668a9268e649bf0ec6.jpg)  
Boogu-Image 0.1 Turbo

![](images/658dca7519c3b7a5319e009837bebc47fdbf66e159b27756f26aa6810aa21b5f.jpg)  
Gemini 3.1 Flash Image

# Additional Qualitative Comparison (Continued)

张写实风格的江南水乡城镇风光摄影作品，画面以一条蜿蜒水道为主线，两侧分布着白墙灰瓦的传统民居和临水店铺，建筑高低错落，沿着微微起伏的地势延伸。前景是一座低矮石拱桥，桥下有一艘窄长木船缓慢划过，船上坐着游客。中景是临河石板路、 木窗、黑色檐口、晾晒的衣物和零散绿植，几家店铺门前挂着简单招牌，其中一块写着<sub>”</sub>茶馆<sub>”</sub>。远处可以看到更密集的屋顶、树木和薄雾中的小桥。整体光线柔和自然，画面既有水乡层次，也有真实生活气息。

![](images/e7f9b92f2638a2b74c3821a9f9b231dc95699f5b5142902ee9ab41d3b1bcca5b.jpg)  
Input Prompt

![](images/2c404c2ecb8d1d50b299f9c9ee9503b8dbae3ffdde73a6267912642f2f0f6bad.jpg)  
LLaDA-Image Turbo  
Z-Image Turbo

![](images/0c7c9c0f16d4150d511156a5baec1ccfacef70a762d1a9dce887d412f35b72da.jpg)  
SenseNova U1.5 Preview

![](images/96fe02582da0beff0d2a7e2adf7da3d1ec7acb312a32b5135077a814416255e1.jpg)  
Qwen-Image 2512

![](images/d949774419a01a16ca84eaecbcfddef85985570c3017bb4ab7929ba0066dea18.jpg)  
GPT-Image 2

![](images/9c7622739b7592192227438b1a30329bce5966ce6c995f7e82ca4e0d2539c707.jpg)  
Input Prompt  
LLaDA-Image Turbo

张写实风格的海岸悬崖广角风景摄影作<sup>品，</sup><sub>风侵</sub> 采 用高处俯瞰视角。前景是一片被 海裸露出灰褐色岩石，几条狭窄步道沿着悬崖边缘蜿蜒向远处。 中 景是数百米高的巨大海蚀悬崖，岩 壁呈深灰与褐色，垂直切入深蓝 色海面，局部能看到明显岩层和被海浪冲刷形成的凹槽。海水不断拍击崖脚，形成细长白色浪带和大片泡沫。远处连续数个岬角逐渐<sup>消失在蓝灰色空气透视中，天空开阔，大片</sup>白云与蓝色天空交错。画面不出现人物和建<sub>筑，突出海岸线的巨大尺度、风力感和开阔</sub> 色天空交错。画面不出现人物和建岸线的巨大尺度、风力感和开阔间。

![](images/817629a43495f7e3dcff2d1e4782b89327367d7cced0cb0c41d3be4ec7ae081f.jpg)  
Z-Image Turbo

![](images/d9903bc9f2cf207a61cfdcc7f7168437a93c0cf08575b901579a969aa8c41773.jpg)

![](images/0c6d3092d77dd9d3abd54f2332d255e9159d52d52f96f4c8904387ffabc4dd8b.jpg)  
SenseNova U1.5 Preview  
Qwen-Image 2512

![](images/3e020d466faca688eaae3cfec96ff37d029d55bfde898ada48ae3d60ac96b9cf.jpg)  
GPT-Image 2  
Gemini 3.1 Flash Image  
Boogu-Image 0.1 Turbo

![](images/5ac311b2c702cf191918ce3a987a5d37283f87917dfc33f751c1d323267643fa.jpg)

![](images/f966d38ed53794b3bdb39139ff7efdbaa38e104612960125f64661dcb38d3053.jpg)

![](images/74f1962db71d34f23b96a3c883a2e3db80140b864ef9e7fca0f25ced9bbe4921.jpg)  
Boogu-Image 0.1 Turbo

![](images/af6a01b6020bcd5540ef62863c5c8500aa6add924da6c89a0a74e7d2c4b81e0d.jpg)  
Gemini 3.1 Flash Image

## D Image Captioning and Filtering

For images selected for paired supervision, the captioner describes the main subjects, attributes, actions, spatial relations, scene, and visual style based only on the image.

After captioning, structural checks remove failed or incomplete outputs, refusal responses, and degenerate repetitions. Qwen3.6-35B-A3B then assesses both image quality and caption faithfulness from the image, caption, and visible-text transcription. We remove samples containing watermark signals or private information. We also reject captions that hallucinate visual content or inaccurately describe or transcribe visible text.

## Sample of Captioning Prompt.

## Chinese Captioning Prompt.

你是一位文本生成图片（<sub>T2I</sub>）模型的数据标注专家。你的目标是为图片生成忠实、具体、信息密度高、贴近真实用户输入习惯的中文训练标注。仅根据图片内容进行标注。

## detailed prompt <sup>规则</sup>

生成一段用于<sub>T2I</sub> 训练的中文描述。描述应忠实、具体、信息密度高。不要写成鉴赏、总结或新闻稿。

覆盖以下要素（不要求固定顺序，自然组织语言即可）：

<sub>•</sub> 主体：人物<sub>/</sub>物体<sub>/</sub>动物<sub>/</sub>食物<sub>/</sub>建筑<sub>/</sub>商品等，包括数量（仅在明确可辨时）、外观、颜色、材质、服装、姿态、动作

<sub>•</sub> 场景：室内<sub>/</sub>室外、地点类型、背景元素

<sub>•</sub> 构图与视角：特写、半身、全身、俯拍、仰拍、平视、近景、远景、居中、对称、留白、拼贴、分栏等，只在明显时描述

<sub>•</sub> 风格与媒介：实拍照片、产品摄影、插画、漫画、<sub>3D</sub>渲染、截图、<sub>UI</sub>设计、海报、信息图、表情包等，只在明显时描述

<sub>•</sub> 光线、色彩、材质：如暖黄灯光、冷色调、木质、金属质感、磨砂、透明等

<sub>•</sub> 可见文字（重要）：图中所有需要渲染的文字必须保留原文，不翻译、不改写，并用中文双引号<sub>“...”</sub>包裹，并标注位置和视觉属性（字号、颜色、字体粗细）。多行文字应逐行描述。如：顶部居中大号红色粗体<sub>”</sub>春节快乐<sub>”</sub>，底部右对齐小号白色细体<sub>”2026”</sub>。<sub>render text</sub> 字段另外做精确逐字转写用于校验

## 写作要求：

<sub>•</sub> 中文，直接描述图中可见内容，开头方式应多样，可以从主体名词、视角描述、场景环境、媒介类型等不同角度切入

<sub>•</sub> 像是直接告诉生成模型<sub>”</sub>要画什么<sub>”</sub>的指令，而不是对图片的解说或鉴赏

<sub>•</sub> 只写有视觉证据支撑的内容，不编造

<sub>•</sub> 长度应随画面信息量变化：简单主体图可短至<sub>150</sub> 字，复杂场景<sub>/</sub>密集文字图应长至<sub>500-800</sub> 字，不要所有图都写成相近长度

• detailed prompt <sup>和</sup>short prompt <sup>作为描述性文本，不使</sup>用emoji<sup>、</sup>markdown <sup>格式或换行符</sup>

<sub>• render text</sub> 是精确转写字段，完全如实还原图上文字的原始内容（包括其中的<sub>emoji</sub>、符号、格式标记等）

## short prompt <sup>规则</sup>

模拟一个真实用户在文生图工具中会输入的简短提示词：

<sub>•</sub> 只关注画面中最主要的几个要素，不需要覆盖全部内容

<sub>•</sub> 像真实用户会输入的提示词，简洁直接

<sub>•</sub> 大致<sub>10-50</sub> 字，不加句号

<sub>•</sub> 例：<sub>”</sub>穿红裙子的女生在海边<sub>””</sub>白底产品图蓝色运动鞋<sub>””</sub>赛博朋克城市夜景霓虹灯<sub>”</sub>

## render text <sup>规则</sup>

<sub>•</sub> 逐字转写画面中清晰可见的文字，保留原始语言

<sub>•</sub> 看不清或模糊的不写，无文字给空列表

输出格式（仅输出<sub>JSON</sub>，无任何额外文字）

{ "tag"<sub>: [</sub>"1-3个类别标签，优先从：fashion/portrait/food/scenery/product/beauty/home/pet/ car/fitness/parenting/poster/ui/infographic/text doc/meme/tutorial/logo/typography/packaging/ other 中选择"<sub>],</sub>

"render text"<sub>: [</sub>"逐字转写的文字片段"<sub>],</sub>

"short prompt"<sub>:</sub> "模拟用户输入的简短提示"<sub>,</sub>

"detailed prompt"<sub>:</sub> "客观信息密集的中文长描述" }

本批次额外硬性要求

<sub>•</sub> 本条必须输出中文。

<sub>•</sub> 图中可见文字必须保留原文，不翻译、不改写、不归一化。

<sub>•</sub> 在d<sub>etai</sub>l<sub>e</sub>d <sub>prompt</sub> 中提到任何图中文字时，必须使用中文双引号<sub>“...”</sub>包裹。

<sub>• render text</sub>字段必须逐字转写图中文字原文；不要翻译。

<sub>•</sub> 仅输出<sub>JSON</sub>，不要输出解释文字。

## English Captioning Prompt.

You are a data annotation expert for text-to-image (T2I) models. Your task is to produce faithful, specific, information-dense English training annotations that resemble prompts written by real users. Base every annotation only on the image content.

## detailed prompt rules

Write an English description for T2I training. The description must be faithful, specific, and informationdense. Do not write an aesthetic review, summary, or news report.

Cover the following elements when they are visible. Use a natural order rather than a fixed template.

• Subjects: people, objects, animals, food, buildings, products, and other main elements, including count when clearly discernible, appearance, color, material, clothing, pose, and action.

• Scene: indoor or outdoor setting, location type, and background elements.

• Composition and viewpoint: close-up, half-body or full-body framing, overhead or low-angle view, eye-level view, near or distant view, centered or symmetric composition, negative space, collage, or column layout, but only when visually evident.

• Style and medium: photograph, product photography, illustration, comic, 3D rendering, screenshot, UI design, poster, infographic, meme, or another clearly visible medium.

• Lighting, color, and material: for example, warm lighting, a cool palette, wood, metal, matte surfaces, or transparency.

• Visible text: preserve all text that must be rendered in its original wording without translation or rewriting. Enclose it in double quotation marks and describe its position and visual properties, including size, color, and weight. Describe multiple lines separately. For example: large red bold text “SUMMER SALE” at the top center and small white light-weight text “2026” at the bottom right. The render text field separately records an exact transcription for verification.

## Writing requirements:

• Write in English and directly describe visible content. Vary the opening by starting from the main subject, viewpoint, setting, or medium as appropriate.

• Write like a direct instruction telling a generation model what to draw, rather than commentary or an aesthetic review.

• Include only details supported by visual evidence. Do not invent content.

• Let the length follow the amount of visible information. Use a shorter description for a simple subject and a longer one for a complex scene or text-dense image. Do not force all captions to similar lengths.

• detailed prompt and short prompt are descriptive text. Do not use emoji, Markdown formatting, or line breaks in either field.

• render text is a verbatim transcription field. Preserve the original visible text, including emoji, symbols, and formatting marks.

## short prompt rules

Simulate a short prompt that a real user might enter into a T2I system.

• Focus only on the most important elements rather than covering the entire image.

• Keep it concise, direct, and natural for a user prompt.

• Use roughly 5–30 words and do not add a final period.

• Examples: “a woman in a red dress by the sea”, “blue running shoe product photo on a white background”, “cyberpunk city at night with neon lights”.

## render text rules

• Transcribe clearly legible text verbatim and preserve its original language.

• Omit illegible or blurred text. Return an empty list if no text is visible.

## Output format (JSON only, with no additional text)

{ "tag": ["1--3 category labels, preferably selected from: fashion/portrait/food/scenery/ product/beauty/home/pet/car/fitness/parenting/ poster/ui/infographic/text doc/meme/tutorial/ logo/typography/packaging/other"],

"render text": ["verbatim visible-text fragment"],

"short prompt": "short prompt resembling a user request",

"detailed prompt": "objective, information-dense English description" }

Additional mandatory requirements for this batch

• The output for this item must be in English.

• Preserve visible text in its original wording. Do not translate, rewrite, or normalize it.

• When detailed prompt mentions visible text, enclose it in double quotation marks.

• The render text field must transcribe visible text verbatim. Do not translate it.

• Output JSON only, with no explanatory text.

## Unified Filtering Prompt.

你是<sub>T2I</sub> 训练数据质量审核员。根据图片内容，判断以下四个字段并输出<sub>JSON</sub>。

1. has private info (bool)

判断图中是否包含个人私密信息。

<sub>• true</sub>：身份证、银行卡、证件照（含真人面部的证件）、私人聊天中的手机号或微信号、快递单上的个人姓名<sub>+</sub>地址<sub>+</sub>电话。

<sub>•</sub> f<sub>alse</sub>：餐厅地址、普通人像照片。

判断标准：是否为个人非公开的可识别信息。

2. has watermark (bool)

看图判断：图片上是否有社交媒体平台叠加的水印（如角落的平台<sub>logo</sub>、半透明的账号<sub>ID</sub>等）。

<sub>• true</sub>：图上可见平台水印

<sub>• false</sub>：图上没有平台水印

3. rt usable (enum: yes / partial / no / na)

对比图片中实际可见的文字与<sub>render text</sub> 的转写内容，判断转写质量。

<sub>• yes</sub>：转写与图上实际文字一致，可直接用作渲染目标。

<sub>• partial</sub>：大部分正确但有少量错字或遗漏。

<sub>• no</sub>：存在严重错字或编造了图上不存在的文字。

<sub>• na</sub>：图上本身没有文字。

4. caption faithfulness (enum: faithful / minor issue / hallucinated)

对比图片内容与detailed caption，判断描述是否忠实。

• faithful<sup>：描述与</sup>画<sup>面一致。</sup>

<sub>• minor issue</sub>：存在小偏差（颜色、数量、位置略有出入）但整体正确，不影响训练。

<sub>• hallucinated</sub>：描述中包含图中明确不存在的物体、人物或文字。

只输出<sub>JSON</sub>，不输出任何其他文字。

当前样本

short: {short}

detailed: {detailed}

render text: {render text}

## E Prompts Used in The Report

This section provides the prompts for the text-to-image examples presented in the report. We retain their original wording and ordering for reproducibility.

## Prompt #1

一张城市公园中的自然街拍式人像摄影，拍摄于阴天之后刚刚放晴的下午。一位年轻东亚女性沿着树木茂密的步道缓慢行走，人物约占画面高度的一半，身体略微侧向镜头，刚好在经过一棵大树时回头看向身后，动作自然随意。她留着黑色中长发，头发整体柔顺，但被潮湿空气影响后有轻微蓬松和细碎毛躁，耳边有几缕头发贴近脸颊。脸型柔和偏圆，下颌收束自然，并非尖细瓜子脸。眉毛颜色较深，毛流完整，眉峰很轻。眼睛偏细长，眼睑厚度真实，瞳孔深棕色，视线直接落向镜头附近，但并不是刻意凝视，带有一点自然的好奇感。下眼睑有轻微阴影，鼻翼附近能看到非常淡的泛红。鼻梁柔和自然，鼻尖略圆。嘴唇厚薄适中，嘴角自然放松，唇纹细腻。皮肤并不追求无瑕，面颊有轻微肤色变化，额头和鼻梁有真实的柔和高光。她穿一件长度到膝盖附近的深绿色棉质风衣，衣服没有完全扣起，里面是一件灰白色薄针织上衣，下身搭配深蓝色牛仔裤。风衣面料稍厚，肩部、手肘和腰部形成明显自然褶皱。背景是公园内层层叠叠的树木、长椅、潮湿的石板路和低矮灌木，部分叶片仍挂着雨珠。远处有几个散步的人影，但经过景深处理后十分柔和。雨后阳光从树木之间穿过，形成局部明暗变化，在人物头发和肩膀上留下不规则亮边。整体像一张真实城市生活摄影。

English translation. A natural, candid street-style portrait photograph in a city park, taken on an afternoon when the sky has just cleared after overcast weather. A young East Asian woman walks slowly along a densely wooded path. She occupies about half the height of the frame, with her body angled slightly toward the camera, and turns to look back just as she passes a large tree; her movement is natural and casual. She has medium-length black hair that is generally smooth, but the humid air has made it slightly voluminous with fine flyaways; a few strands near her ears cling close to her cheeks. Her face is soft and somewhat round, with a naturally tapering jaw rather than a sharply pointed, V-shaped face. Her eyebrows are dark, with complete, clearly visible hair growth and only a very subtle arch. Her eyes are relatively long and narrow, with realistically thick eyelids and dark-brown irises. Her gaze falls directly near the camera, but she is not staring deliberately; it carries a hint of natural curiosity. There are slight shadows beneath her lower eyelids and extremely faint redness near the sides of her nose. Her nasal bridge is soft and natural, and the tip of her nose is slightly rounded. Her lips are of medium fullness, the corners of her mouth are naturally relaxed, and the fine lip lines are visible. Her skin is not intended to look flawless: there are slight variations in complexion across her cheeks and realistic, soft highlights on her forehead and nasal bridge. She wears a dark-green cotton trench coat reaching to around her knees. It is not fully buttoned, revealing a thin grayish-white knitted top underneath, paired with dark-blue jeans. The trench-coat fabric is somewhat thick, forming clearly visible, natural folds at the shoulders, elbows, and waist. The background contains layered park trees, benches, a damp flagstone path, and low shrubs, with raindrops still hanging from some leaves. Several distant walkers are visible, but are rendered very softly by the depth of field. Post-rain sunlight filters through the trees, creating localized changes between light and shade and leaving irregular rim highlights on the woman’s hair and shoulders. The overall image should resemble an authentic photograph of everyday urban life.

## Prompt #2

一张写实风格、像是手机后置镜头拍摄的、光线柔和的室内人像照片，捕捉一位年轻白人女性的温馨瞬间。画面主体位于中央，女性留着浓密波浪长发，发色呈现温暖的赤褐色，显得通透且富有光泽。女性面部特征清晰，皮肤白皙，鼻梁和脸颊上分布着明显的雀斑。女性面带灿烂的微笑，露出洁白整齐的牙齿，眼神温柔而自然地注视着镜头。她身穿一件白色长袖衬衫，外搭一件浅灰色针织背心，下身穿着一条淡粉色长裙。她左手手腕上佩戴一只银色金属表带手表，站在桌旁或坐在桌后，姿态自然端庄，用一只手指向相框，起到展示相框内容的作用，动作自然、清晰可见。桌子上放置着一个巨大的长方形相框，相框位于画面视觉中心，尺寸非常醒目，正面朝向镜头展示。相框风格精美雅致，边框具有细腻的装饰感与真实材质质感。相框中的内容清晰可见，内部文字为超大号粗体字体，醒目、居中、完整展示。相框内清晰写着：<sub>“LLaDA-Image is</sub><sub>Released!”</sub>。文字占据相框内部主要视觉区域，字形工整清楚，边缘锐利，清晰易读，不模糊、不变形、不被遮挡。画面左侧前景放置一个透明玻璃花瓶，里面插着一束干枯的粉色玫瑰花，花茎细长，叶片呈灰绿色，为画面增添复古而自然的质感。背景是一面纯白色墙壁，没有任何装饰，以突出人物主体和桌上的相框。桌面呈现出深色且有光泽的质感，为深色大理石表面，并反射微弱柔和的光线。整体色调以柔和的灰色、粉色和白色为主，营造出温暖、宁静、自然的氛围。光线均匀柔和，没有强烈阴影，属于典型的室内自然光或柔光摄影风格。整体画面呈现真实、细腻、温馨的人像摄影质感，人物清晰，对焦准确，浅景深适中，主体突出，构图重点明确落在女性与桌上巨大相框的互动展示上。

English translation. A realistic, softly lit indoor portrait photograph that looks as though it was taken with a phone’s rear camera, capturing a warm moment with a young white woman. The subject is centered in the frame. The woman has thick, long, wavy hair in a warm auburn shade that appears translucent and lustrous. Her facial features are clear; she has fair skin and prominent freckles across the bridge of her nose and cheeks. She wears a radiant smile that reveals neat, white teeth, and looks gently and naturally at the camera. She is dressed in a white long-sleeved shirt under a light-gray knitted vest, with a pale-pink long skirt. She wears a watch with a silver metal band on her left wrist. Standing beside the table or sitting behind it in a natural, poised posture, she points toward the picture frame with one hand to present its contents; this gesture should look natural and be clearly visible. A huge rectangular picture frame sits on the table at the visual center of the image. Its conspicuous size immediately attracts attention, and its front faces the camera. The frame is refined and elegant, with delicately decorative detailing and a realistic material texture. Its contents are clearly visible, with exceptionally large, bold text displayed prominently, centrally, and in full. The frame clearly reads: “LLaDA-Image is Released!” The text occupies most of the visual area inside the frame; the lettering is neat and clear, with sharp edges, easy to read, neither blurred nor distorted, and not obscured. In the left foreground, a transparent glass vase holds a bunch of dried pink roses with slender stems and gray-green leaves, adding a vintage yet natural texture to the scene. The background is a completely plain white wall with no decoration, emphasizing the woman and the picture frame on the table. The tabletop is dark and glossy, made of dark marble, and reflects faint, soft light. The overall palette is dominated by soft gray, pink, and white, creating a warm, tranquil, and natural atmosphere. The illumination is even and gentle, with no harsh shadows, characteristic of indoor natural-light or soft-light photography. The overall image should have the realistic, delicate, and warm quality of portrait photography, with the woman clearly rendered, accurate focus, a moderate shallow depth of field, a prominent subject, and the composition explicitly centered on the woman’s interaction with and presentation of the huge picture frame on the table.

## Prompt #3

一张带有轻复古感的小红书风格街拍人像摄影作品，拍摄于老城区的咖啡馆外墙边。主体是一位二十多岁的年轻东亚女生，靠着一面浅米色旧墙站立，一只手拿着拍立得相机，另一只手提着纸袋。她的脸型偏柔和长椭圆，额头和下颌过渡自然，颧骨位置不高。黑色中长发自然散落，发尾有轻微弧度，头顶发丝细软，耳边有两缕碎发。眉毛自然偏浓，毛流清楚。眼睛偏细长，双眼皮较浅，瞳孔看向手里的相机，神态带一点安静和专注。眼下阴影很轻，右侧眼角附近有一颗很淡的小痣。鼻梁顺滑，鼻尖略圆，鼻翼附近保留自然毛孔。嘴唇厚薄适中，唇色接近自然豆沙色，唇纹清晰。皮肤偏哑光，有轻微肤色不均和真实质感，不显油润。她穿一件奶白色针织短袖开衫，下身搭配浅咖色高腰半裙，脚穿棕色乐福鞋和白色短袜，整体穿搭年轻又有一点文艺复古感。背景是旧墙、木窗、咖啡馆小圆桌和门口黑板，黑板上面写着<sub>“</sub>今日特调<sub>”</sub>四个字，下面写着<sub>“</sub>冰美式<sub>”</sub>三个字。傍晚柔和天光与店内暖光混合，整体画面轻松、时髦、青春，具有很强的社交平台分享感。

English translation. A Xiaohongshu-style street portrait with a lightly vintage character, photographed beside the exterior wall of a cafe in an old city district. The subject is a young East Asian woman in her twenties, standing against an aged, light-beige wall. She holds an instant camera in one hand and carries a paper bag in the other. Her face has a soft, elongated oval shape, with natural transitions across the forehead and jaw and cheekbones that are not set high. Her medium-length black hair falls naturally, with a slight curve at the ends, fine and soft strands at the crown, and two wispy locks beside her ears. Her eyebrows are naturally rather full, with individual hairs clearly visible. Her eyes are relatively long and narrow, with shallow double-eyelid creases, and her pupils look toward the camera in her hand; her expression is quiet and focused. There are only very faint shadows beneath her eyes and a subtle small mole near the outer corner of her right eye. Her nasal bridge is smooth, the tip of her nose is slightly rounded, and natural pores remain visible around the sides of her nose. Her lips are of medium fullness, with a natural muted-rose color and clearly visible lip lines. Her skin has a mostly matte appearance, with slight unevenness in tone and a realistic texture, rather than looking oily. She wears a cream-colored, short-sleeved knitted cardigan with a light-brown high-waisted midi skirt, brown loafers, and short white socks. The overall outfit feels youthful with a slightly artistic, vintage quality. The background contains the old wall, a wooden window, a small round cafe table, and a blackboard by the entrance. The blackboard must show the four Chinese characters “<sup>今日特</sup>调” on the upper part and the three Chinese characters “<sup>冰美式</sup>” beneath them. Soft evening daylight mixes with warm light from inside the cafe. The overall image is relaxed, fashionable, and youthful, with a strong sense of being made for sharing on social media.

## Prompt #4

一张垂直构图的街头纪实摄影作品，拍摄于清晨刚开始营业的菜市场。画面主体是一位中年男性摊主，皮肤被阳光晒得略深，头发短而微乱，穿着浅灰色短袖<sub>T</sub>恤和深蓝色防水围裙，正低头整理摊位上的蔬菜。摊位上整齐摆放着绿色的黄瓜、红色番茄、紫色茄子、带泥的胡萝卜和成捆的小葱，部分蔬菜表面还带有细小水珠，显得新鲜。前景左下角是一只装着生菜的塑料筐，边缘略微虚化。画面右侧可以看到一位戴草帽的顾客正在挑选青椒，只露出半个身影。背景中是一排排简易金属支架和蓝绿色遮阳棚，棚下悬挂着白色灯泡和写有价格的手写纸牌，上面写着<sub>“</sub>小番茄<sub>5</sub>元<sub>/</sub>斤<sub>”</sub>几个字。清晨的自然光从市场入口处斜射进来，在蔬菜表面形成明亮高光，同时保留阴影区域的真实层次。整体画面色彩丰富但不过度饱和，呈现出普通城市早晨的烟火气息。

English translation. A vertically composed street-documentary photograph taken at a produce market just as it begins operating in the early morning. The main subject is a middle-aged male stallholder with slightly sun-darkened skin and short, mildly tousled hair. Wearing a light-gray short-sleeved T-shirt and a dark-blue waterproof apron, he looks down while arranging the vegetables on his stall. Neatly displayed on the stall are green cucumbers, red tomatoes, purple eggplants, mud-streaked carrots, and bundles of scallions. Tiny droplets of water remain on some of the vegetables, making them appear fresh. In the lower-left foreground is a plastic crate filled with lettuce, its edges slightly out of focus. On the right side of the frame, a customer in a straw hat is selecting green peppers, with only half of their figure visible. The background contains rows of simple metal supports and blue-green awnings. White light bulbs and handwritten paper price signs hang beneath the awnings; one sign must read “<sup>小番茄</sup>5<sup>元</sup>/<sup>斤</sup>”. Early-morning natural light enters diagonally from the market entrance, forming bright highlights on the vegetables while preserving realistic tonal depth in the shadowed areas. The overall image is colorful but not excessively saturated, conveying the lively, everyday atmosphere of an ordinary urban morning.

一张傍晚海边栈道上的旅行合照摄影作品，主体是两位年轻东亚男性和一位年轻东亚女性站在木栈道护栏前拍合照。三人并排而立，身体朝向镜头，风把头发和衣角轻轻吹起，整体像傍晚散步时请旁人帮忙拍下的一张照片。左侧男生脸型偏长圆，黑色短发被海风吹得稍微凌乱，眉毛浓度中等，眼睛偏长，鼻尖略圆，嘴唇较薄，皮肤偏哑光。他穿浅卡其色风衣和深灰色长裤。中间女生脸型偏柔和鹅蛋形，黑色长发扎成低马尾，耳边有两缕细碎发丝，眼睛偏细长，鼻梁柔和，嘴唇略丰满，皮肤自然干净。她穿白色针织上衣和浅蓝色牛仔裤。右侧男生黑色短发较整齐，眉毛较粗，眼睛为内双，鼻梁中等高度，嘴角轻微上扬。他穿深蓝色夹克和米白色<sub>T</sub>恤。背景是灰蓝色海面、远处灯塔、礁石和被风吹动的低矮灌木。阴天傍晚的天光柔和均匀，整体安静、真实、带旅行感。

English translation. A travel group photograph taken on a seaside boardwalk in the evening. The subjects are two young East Asian men and one young East Asian woman posing for a group photo in front of the wooden boardwalk railing. The three stand side by side facing the camera as the wind gently lifts their hair and the edges of their clothing. The image should look like a photograph taken by a passerby for them during an evening stroll. The man on the left has a somewhat elongated, rounded face; short black hair made slightly messy by the sea breeze; eyebrows of medium density; relatively long eyes; a slightly rounded nose tip; thin lips; and mostly matte skin. He wears a light-khaki trench coat and dark-gray trousers. The woman in the center has a soft, oval face and long black hair tied in a low ponytail, with two fine, wispy strands beside her ears. Her eyes are relatively long and narrow, her nasal bridge is soft, her lips are slightly full, and her skin looks naturally clear. She wears a white knitted top and light-blue jeans. The man on the right has fairly neat short black hair, thick eyebrows, tapered double eyelids, a nasal bridge of medium height, and slightly upturned corners of the mouth. He wears a dark-blue jacket and an off-white T-shirt. The background consists of a gray-blue sea, a distant lighthouse, reefs, and low shrubs stirred by the wind. The overcast evening light is soft and even, and the overall image feels quiet, authentic, and evocative of travel.

## Prompt #6

一张垂直构图的室内中景照片，捕捉了一对年轻东亚情侣在餐厅庆祝生日的温馨瞬间。画面中心，一位留着黑色长直发的年轻女子身穿深灰色毛呢大衣，内搭米白色高领针织衫，面带温柔微笑注视镜头，双手托举着一个精致的粉色圆形生日蛋糕。在她身后右侧，一位留着黑色短发的年轻男子身穿浅灰色连帽卫衣，左臂亲密地环绕在女子肩后，同样微笑着看向镜头。女子手中的蛋糕表面覆盖着粉色奶油，装饰有数只金边白翅的蝴蝶造型糖饰、大小不一的白色珍珠状装饰球，以及一支正在燃烧的细长生日蜡烛。前景的深色木质圆桌上放置着一盘意大利面，旁边摆放着两个透明玻璃水杯和一个插着燃烧白色粗蜡烛的金色烛台。背景中可见铺着深绿色桌布的餐桌、一瓶红酒、一瓶插着粉色花朵的小花瓶以及一盆绿植，墙面采用木质护墙板装饰。整体光线柔和温暖，营造出浪漫且充满生活气息的氛围。

English translation. A vertically composed, indoor medium shot capturing a warm moment as a young East Asian couple celebrates a birthday in a restaurant. At the center of the frame, a young woman with long, straight black hair wears a dark-gray wool coat over an off-white turtleneck sweater. Smiling gently and looking at the camera, she holds up an elegant, round pink birthday cake with both hands. Behind her on the right, a young man with short black hair wears a light-gray hoodie. His left arm is wrapped affectionately behind the woman’s shoulders, and he likewise smiles toward the camera. The cake in the woman’s hands is covered with pink frosting and decorated with several butterfly-shaped sugar ornaments featuring gold-edged white wings, white pearl-like decorative balls in varying sizes, and one slender birthday candle that is burning. On the dark wooden round table in the foreground is a plate of pasta, beside two transparent glass tumblers and a gold candlestick holding a lit, thick white candle. Visible in the background are a dining table covered with a dark-green tablecloth, a bottle of red wine, a small vase containing pink flowers, and a potted green plant; the walls are finished with wooden wainscoting. The overall lighting is soft and warm, creating a romantic atmosphere rich in the feeling of everyday life.

<sub>Prompt#7</sub> 一张图书馆儿童区里的自然人像摄影作品，主体是一位七岁左右的东亚小女孩，坐在靠窗的矮桌前翻看一本绘本。她的脸型偏长圆，额头平整，面颊仍然饱满，下颌线柔和，下巴略圆。黑色长发扎成松散低马尾，额前留着少量细碎刘海，耳边有两缕细发自然垂下。眉毛细而自然，眉峰很轻。眼睛偏细长，双眼皮较浅，深棕色瞳孔向下看着书页，睫毛自然，不浓密夸张。眼下有极轻的阴影和淡淡卧蚕。鼻梁低而顺滑，鼻尖圆润。嘴唇小巧，上唇略薄，下唇稍厚，嘴角自然放松。皮肤整体自然偏哑光，面颊和鼻侧有非常轻微的肤色变化。她穿浅粉色针织开衫，内搭白色圆领 恤，下身为深灰色百褶裙。桌面上放着几支彩色铅笔和一本打开的绘本，封面写着 森林朋友 。背景是儿童书架、软垫和窗外树影，上午自然光柔和明亮，整体安静、真实。

English translation. A natural portrait photograph in the children’s area of a library. The subject is an East Asian girl about seven years old, seated at a low table by a window and leafing through a picture book. Her face is somewhat elongated and rounded, with a smooth forehead, still-full cheeks, a soft jawline, and a slightly rounded chin. Her long black hair is tied into a loose low ponytail, with a small amount of fine, wispy fringe across her forehead and two thin strands falling naturally beside her ears. Her eyebrows are fine and natural, with a very subtle arch. Her eyes are relatively long and narrow, with shallow double-eyelid creases.

Her dark-brown pupils look downward at the pages, and her eyelashes are natural rather than excessively thick or exaggerated. Beneath her eyes are extremely faint shadows and a subtle fullness of the lower eyelids. Her nasal bridge is low and smooth, and her nose tip is rounded. Her lips are small, with a slightly thin upper lip and a somewhat fuller lower lip, and the corners of her mouth are naturally relaxed. Her skin has an overall natural, mostly matte appearance, with extremely slight variations in complexion across the cheeks and sides of the nose. She wears a light-pink knitted cardigan over a white crew-neck T-shirt and a dark-gray pleated skirt. Several colored pencils and an open picture book rest on the tabletop; the cover must read “<sup>森</sup> <sup>林朋友</sup>”. The background contains children’s bookshelves, cushions, and tree shadows visible outside the window. Soft, bright morning daylight illuminates the scene, and the overall image is quiet and authentic.

## Prompt #8

一张居家环境中的儿童生活摄影作品，主体是一位四岁左右的小女孩，盘腿坐在客厅地毯上搭积木。她拥有柔和偏圆的脸型，额头饱满，颧骨几乎不明显，面颊圆润，下巴短而圆。深棕黑色短发长度到耳下，发尾自然内扣，额前留着略微参差的薄刘海，头顶有几根细软发丝翘起。眉毛细而浅，眉头毛流自然。眼睛为偏圆的杏仁形，双眼皮褶皱较浅，深棕色瞳孔正看向手里准备叠上的红色积木，神情认真。鼻梁较低，鼻尖圆润，鼻翼自然。嘴唇小巧，上唇略薄，下唇饱满一点，嘴巴轻轻张开，呈现孩子专注时不自觉的表情。皮肤呈自然偏哑光的细腻质感，面颊有轻微自然红润，额头和鼻侧没有明显油光。她穿浅米色针织上衣和浅绿色棉质长裤，脚上只穿白色袜子。地毯上散落着木质积木、绘本和一只毛绒兔子，背景是浅色布艺沙发、落地窗和绿植，午后自然光从窗边照入，整体温暖、真实、生活化。

English translation. A slice-of-life photograph of a child in a home setting. The subject is a girl about four years old, sitting cross-legged on the living-room carpet and building with blocks. She has a soft, somewhat round face, a full forehead, nearly imperceptible cheekbones, rounded cheeks, and a short, round chin. Her short, dark brown-black hair reaches just below her ears and naturally curves inward at the ends. A thin, slightly uneven fringe crosses her forehead, and several fine, soft strands stand up at the crown. Her eyebrows are fine and light, with natural hair growth at their inner ends. Her eyes are relatively round and almond-shaped, with shallow double-eyelid creases. Her dark-brown pupils are focused on the red block in her hand that she is about to stack, and her expression is serious and attentive. Her nasal bridge is low, her nose tip is rounded, and the sides of her nose look natural. Her lips are small, with a slightly thin upper lip and a somewhat fuller lower lip. Her mouth is gently open, showing the unconscious expression of a child absorbed in concentration. Her skin has a delicate, natural, mostly matte texture, with a faint natural flush in her cheeks and no obvious shine on her forehead or the sides of her nose. She wears a light-beige knitted top and light-green cotton trousers, with only white socks on her feet. Wooden blocks, picture books, and a plush rabbit are scattered across the carpet. The background contains a light-colored fabric sofa, a floor-to-ceiling window, and green plants. Afternoon natural light enters from the window, and the overall image feels warm, authentic, and true to everyday life.

## Prompt #9

一张冬日傍晚游乐场里的儿童旧照片风格摄影作品，画面主体是一位五岁左右的东亚小男孩，站在旋转木马入口旁，一只手摆出剪刀手，另一只自然下垂，抬头看向镜头。孩子脸型偏圆，额头较宽，面颊柔软饱满，下巴短而圆。黑色短发细软，从毛线帽边缘露出来，额前有几缕细碎头发被冷风轻轻吹乱，但整体没有油亮感。眉毛颜色较浅，整体偏直。眼睛偏圆，双眼皮浅，深棕色瞳孔正对镜头，眼神里带有期待和一点兴奋，下眼睑有轻微卧蚕。鼻梁低而自然，鼻尖圆润，鼻翼略宽，因为天气寒冷，鼻尖和双颊带有一点淡淡的自然红润。嘴唇偏小，下唇稍厚，嘴巴轻轻张开，露出几颗儿童牙齿。皮肤呈自然偏哑光质感，面部受冷空气影响有柔和的红润变化。孩子穿一件米白色羽绒外套，内搭浅灰色高领针织衫，下身是浅蓝色牛仔背带长裤，脚上穿红白相间的运动鞋，头上戴着一顶深红色针织毛线帽，脖子上围着一条浅卡其色围巾。背景是冬日傍晚的旋转木马、暖黄色灯泡、略微褪色的彩灯装饰、铁栏杆和远处穿着厚外套、手里拿着气球的游客，入口灯牌上写着 儿童乐园 。地面是略显冰冷的深灰色水泥地，边缘还能看到几片被风吹到角落里的枯叶，空气中带有明显冬季傍晚的清冷感。整张画面像家庭相册中保存多年的冬日游乐场旧照片，暖黄色胶片偏色明显但不过度，灯光有轻微晕开效果，整体饱和度偏低，带有细腻颗粒、轻微扫描灰尘和一点边缘柔化，呈现真实自然的年代感。右下角显示橙红色日期戳<sub>“021202”</sub>。

English translation. An old-photo-style image of a child at an amusement park on a winter evening. The main subject is an East Asian boy about five years old, standing beside the entrance to a carousel. He raises one hand in a V sign while the other hangs naturally, and looks up toward the camera. The child has a somewhat round face, a broad forehead, soft full cheeks, and a short, round chin. His fine, soft black hair peeks out from the edge of his knitted hat. Several wispy strands across his forehead are lightly tousled by the cold wind, but his hair has no overall oily sheen. His eyebrows are light in color and mostly straight. His eyes are relatively round, with shallow double-eyelid creases. His dark-brown pupils face the camera, and his gaze conveys anticipation and a little excitement; there is a slight fullness beneath his lower eyelids. His nasal bridge is low and natural, his nose tip is rounded, and the wings of his nose are slightly wide. Because of the cold weather, the tip of his nose and both cheeks show a faint, natural flush. His lips are small, with a somewhat thicker lower lip, and his mouth is gently open, revealing several baby teeth. His skin has a natural, mostly matte texture, with soft changes in facial redness caused by the cold air. The child wears an off-white down jacket over a light-gray turtleneck sweater, light-blue denim overalls, red-and-white sneakers, a dark-red knitted hat, and a light-khaki scarf around his neck. The background shows a carousel on a winter evening, warm-yellow light bulbs, slightly faded colored-light decorations, iron railings, and distant visitors wearing thick coats and holding balloons. The illuminated entrance sign must read “<sup>儿童乐园</sup>”. The ground is cold-looking dark-gray concrete, with several dead leaves visible along the edges where the wind has blown them into corners, and the air has the distinct chill of a winter evening. The entire image should resemble an old winter-amusement-park photograph preserved for many years in a family album. It has a pronounced but not excessive warm-yellow film color cast, slight halation around the lights, generally low saturation, fine grain, a little scanned dust, and slight edge softness, producing an authentic and natural period quality. An orange-red date stamp reading “02 12 02” appears in the lower-right corner.

一张写实自然风光摄影作品，表现一片由雪山融水补给的钙华湖群。远处背景是一列雪山，山顶积雪明亮，靠近山脊的地方可见被风吹出的雪纹，山腰为灰黑色裸岩和零星深绿色针叶林。中景主体是多个彼此相连的浅水钙华池，池壁由乳白色和浅米色矿物沉积形成，轮廓弯曲自然。水体颜色非常丰富，从浅蓝、青绿、碧绿到接近透明的浅色过渡明显，部分水池底部可见金黄色、浅褐色的矿物沉积纹路。几道细流从高处雪山方向汇入池群，在池间形成小型跌水和白色泡沫。前景是一片湿润的浅色石灰质地面，上面有不规则水道、低矮苔藓、几簇黄绿色草丛以及被矿物染成浅褐色的细小枝干。画面左侧有一片松林从山脚延伸至湖边，右侧则是较开阔的山坡，分布着灌木和浅灰岩石。水面倒映天空与远山，但因为流水和风的影响并不完全平整。天空为高原地区常见的浅蓝色与大片白云组合，云影投射在部分池面和山坡上，形成清晰的明暗变化。整体画面包含雪山、钙华池、流水、林地、草坡和矿物地表，细节充分，真实而富有层次。

English translation. A realistic natural-landscape photograph depicting a group of travertine lakes fed by meltwater from snow-covered mountains. In the distant background is a range of snowy mountains with bright snow on their peaks. Wind-sculpted patterns in the snow are visible near the ridgelines, while the mountainsides consist of gray-black exposed rock and scattered dark-green coniferous forest. The principal midground subject is a series of interconnected, shallow travertine pools. Their walls are formed from milky-white and light-beige mineral deposits and have naturally curving outlines. The water displays a rich range of colors, with clearly visible transitions from light blue through cyan-green and emerald green to pale, nearly transparent tones. Golden-yellow and light-brown patterns of mineral deposits can be seen on the bottoms of some pools. Several narrow streams flow toward the pool system from the higher snowy mountains, creating small cascades and white foam between the pools. The foreground consists of damp, pale limestone terrain crossed by irregular channels, with low moss, several clumps of yellow-green grass, and fine twigs stained light brown by minerals. On the left side of the image, a pine forest extends from the mountain foot to the lakeshore, while the right side contains a more open slope scattered with shrubs and light-gray rocks. The water reflects the sky and distant mountains but is not perfectly smooth because of flowing water and wind. The sky combines the pale blue typical of high-altitude regions with broad white clouds. Cloud shadows fall across some pool surfaces and slopes, creating clearly defined variations between light and dark. The overall image includes snowy mountains, travertine pools, flowing water, woodland, grassy slopes, and mineral terrain, with abundant detail, realism, and rich spatial depth.

## Prompt #11

一张写实风光摄影作品，展示一处喀斯特峡谷中的地下河出口。画面主体是一座巨大的石灰岩崖壁，岩壁中央开有一个半圆形巨型洞口，洞口内部昏暗，地下河从洞中流出，形成一条青绿色的河流向画面前方延伸。洞口周围岩壁呈灰白色与深灰色相间，表面布满溶蚀形成的凹凸纹理、裂隙和垂直水痕，局部附着深绿色苔藓和蕨类植物。中景河岸两侧生长着浓密植被，包括灌木、藤蔓、小乔木和大片阔叶植物，层次非常丰富。前景是一片湿润的卵石河滩，散布着浅灰色、褐色和青灰色圆石，石头之间积着浅浅水洼，反射着天空和周围绿植。河水表面在靠近洞口处较为平静，向前流动时逐渐出现细小波纹和局部白色浪花。画面右侧还能看到一股小瀑布从崖壁裂缝中落下，汇入主河道。天空只在峡谷上方露出一条狭窄的浅蓝色区域，其余大部分被高耸崖壁和树冠遮挡。散射光从峡谷上方洒入，使岩石、水流和植物都保有真实质感。整体氛围清凉、潮湿、结构复杂，充分体现喀斯特峡谷和地下河系统的地貌特征。

English translation. A realistic landscape photograph showing the outlet of an underground river in a karst canyon. The main subject is an immense limestone cliff with a gigantic semicircular cave opening at its center. The interior of the cave is dim, and an underground river flows out from it, forming a blue-green river that extends toward the foreground. Around the cave opening, the cliff alternates between gray-white and dark gray rock. Its surface is covered with uneven dissolution textures, fissures, and vertical water stains, with dark-green moss and ferns attached in places. Dense vegetation grows along both midground riverbanks, including shrubs, vines, small trees, and broad expanses of broadleaf plants, creating exceptionally rich layering. The foreground is a damp pebble beach scattered with rounded stones in light gray, brown, and blue-gray. Shallow puddles collect between the stones, reflecting the sky and surrounding greenery. The river surface is relatively calm near the cave opening; as the water flows forward, fine ripples and localized white splashes gradually appear. On the right side of the image, a small waterfall can also be seen dropping from a fissure in the cliff and joining the main river channel. Only a narrow strip of pale-blue sky is visible above the canyon; most of it is obscured by the towering cliffs and tree canopies. Diffuse light enters from above the canyon, allowing the rocks, flowing water, and plants to retain realistic textures. The overall atmosphere is cool, humid, and structurally complex, fully expressing the geomorphological characteristics of a karst canyon and underground-river system.

<sub>Prompt #12</sub> 一张高清风景摄影作品，展现了中国江南水乡的宁静景色，画面呈现出典型的<sub>“</sub>小桥流水人家<sub>”</sub>风貌。前景左下角是一丛茂密的绿色灌木，叶片间点缀着几片枯黄的叶子，增加了画面的层次感和自然气息。中景是一条宽阔平缓的河流，河水呈灰绿色，水面有细微的波纹，倒映着岸边的绿树和天空。河流左侧岸边错落分布着几座白墙黛瓦的传统中式民居，房屋依水而建，具有典型的江南建筑特色，屋檐下隐约可见红色的灯笼装饰。河流右侧是一条沿河铺设的蜿蜒步道，步道旁种植着高大茂盛的垂柳和其他阔叶树木，翠绿的枝叶繁茂，部分枝条垂向水面。在河流的中远景处，一座灰色的石拱桥横跨两岸，连接着左右两边的景色。背景是连绵起伏的青山，山上覆盖着浓密的森林，呈现出深浅不一的绿色，山峦在远处与灰白色的天空相接。整体光线柔和均匀，为阴天漫射光，没有强烈的阴影，营造出一种清幽、静谧且略带湿润的氛围。

English translation. A high-definition landscape photograph presenting the tranquil scenery of a Jiangnan water town in China and the archetypal character of “small bridges, flowing water, and waterside homes.” In the lower-left foreground is a dense cluster of green shrubs, with several withered yellow leaves interspersed among the foliage to add visual depth and a natural quality. The midground contains a broad, gently flowing river. Its water is gray-green, with fine ripples across the surface reflecting the green trees along the banks and the sky. Several traditional Chinese residences with white walls and dark-gray tiled roofs are irregularly distributed along the left bank. Built beside the water, the houses display typical Jiangnan architectural features, and red lantern decorations are faintly visible beneath their eaves. On the right side of the river, a winding path follows the bank. Tall, lush weeping willows and other broadleaf trees grow beside it, with abundant bright-green foliage and some branches hanging down toward the water. In the middle-to-far distance, a gray stone arch bridge spans the river, connecting the scenery on the two banks. The background consists of rolling green mountains covered in dense forest and rendered in varied shades of green; in the distance, the mountains meet a gray-white sky. The overall lighting is soft and even, consisting of diffuse overcast light with no harsh shadows, creating a secluded, serene, and slightly humid atmosphere.

<sub>rompt #13</sub> 一张垂直构图的写实摄影作品，捕捉了威尼斯狭窄运河中的经典景象。画面主体是一艘黑色的贡多拉（<sub>Gondola</sub>），正沿着水道向画面深处驶去。船尾站立着一位男性船夫，背对镜头，身穿经典的蓝白横条纹水手衫和深色长裤，头戴深色帽子，正手持长桨划船。贡多拉内部可见红色的软垫座椅和金色的金属装饰细节。运河水面呈深绿色，波光粼粼，清晰地倒映着两侧建筑的色彩和天空的光线，尤其是中央黄色建筑的金黄色倒影在水面上拉得很长。画面左侧是一排红褐色的多层建筑，墙面斑驳，显露出岁月的痕迹，窗户配有深绿色的百叶窗，二楼有一个带有白色栏杆的小阳台，墙根处可见裸露的红砖。画面右侧是另一栋红褐色建筑，墙面同样有剥落的痕迹，设有拱形的门洞和窗户。视线向远处延伸，可以看到一座白色的石拱桥横跨在运河上，连接着两岸。背景中央矗立着一栋醒目的黄色建筑，拥有白色的窗框和阳台，与周围暖色调的建筑形成对比。光线柔和，呈现出黄昏或清晨的暖色调，阳光从侧面照射在建筑上，营造出宁静、浪漫且充满历史感的氛围。

English translation. A vertically composed, photorealistic photograph capturing a classic scene along a narrow canal in Venice. The main subject is a black gondola traveling along the waterway into the depth of the frame. A male gondolier stands at the stern with his back to the camera, wearing a classic blue-and-white horizontally striped sailor shirt, dark trousers, and a dark hat, and propelling the boat with a long oar. Inside the gondola, red upholstered seats and gold-colored metal decorative details are visible. The canal water is deep green and glitters with reflected light, clearly mirroring the colors of the buildings on both sides and the light from the sky; in particular, the golden reflection of the yellow building in the center stretches far across the water. On the left is a row of reddish-brown multistory buildings whose mottled walls show the passage of time. Their windows have dark-green shutters; a small second-floor balcony has white railings, and exposed red brick can be seen near the base of the wall. On the right is another reddish-brown building with similarly peeling walls, arched doorways, and arched windows. Looking farther into the distance, a white stone arch bridge spans the canal and connects the two banks. An eye-catching yellow building with white window frames and balconies rises in the center of the background, contrasting with the surrounding warm-toned architecture. The light is soft and warmly toned, suggesting dusk or early morning. Side lighting falls across the buildings, creating a tranquil, romantic atmosphere rich in historical character.

## Prompt #14

一张写实城市风光摄影作品，拍摄于雨后刚刚放晴的傍晚。画面从一处较高的人行天桥向下俯视，前景是被雨水打湿的金属栏杆和几片粘在地面的梧桐叶，栏杆表面残留细小水珠。中景是一条宽阔城市主干道，深灰色柏油路面仍然湿润，车道上的白色标线、公交车和私家车都在积水中形成拉长的倒影，路口处有骑自行车和撑伞的行人正在等待信号灯。道路两侧是高度不同的写字楼、住宅楼和沿街商铺，玻璃幕墙反射着雨后浅蓝灰色天空，部分较老建筑的外墙能看到空调机位、晾衣架和褪色招牌。远处一座高架桥横跨画面，桥下亮起第一批暖黄色路灯，更远处是密集城市天际线。云层正在从西侧散开，一束偏暖的夕阳从建筑缝隙中照入，使湿润路面、玻璃幕墙和树冠出现细碎亮光。整体色彩自然克制，既有雨后空气的清透感，也保留交通、人群、建筑和街道设施构成的真实城市复杂度。

English translation. A photorealistic urban landscape photograph taken on an evening when the sky has just cleared after rain. The scene looks downward from an elevated pedestrian overpass. In the foreground are rain-soaked metal railings and several plane-tree leaves stuck to the ground, with tiny water droplets still clinging to the railings. The middle ground contains a broad urban arterial road whose dark-gray asphalt remains wet. The white lane markings, buses, and private cars cast elongated reflections in standing water, while cyclists and umbrella-carrying pedestrians wait for the traffic light at the intersection. Office towers, residential buildings, and street-level shops of varying heights line both sides of the road. Glass curtain walls reflect the pale blue-gray post-rain sky; on the exteriors of some older buildings, air-conditioner mounting bays, clothes-drying racks, and faded signs are visible. In the distance, an elevated roadway crosses the frame, with the first warm-yellow streetlights glowing beneath it, and a dense city skyline farther beyond. Clouds are breaking apart in the west, and a warm shaft of sunset light enters through gaps between the buildings, creating scattered glints on the wet roadway, glass facades, and tree canopies. The overall colors are natural and restrained, conveying both the crystalline clarity of the air after rain and the authentic urban complexity formed by traffic, crowds, buildings, and street infrastructure.

## Prompt #15

一张清晨老城区屋顶视角的写实城市摄影作品。前景是几排灰黑色瓦屋顶和红褐色砖墙，屋檐上能看到雨槽、电视天线和成排空调外机，部分平屋顶上摆放着水箱、晾衣架和盆栽。狭窄街巷隐藏在建筑之间，只能从缝隙中看到早餐摊升起的淡淡蒸汽和几辆缓慢经过的电动车。中景逐渐过渡到八九十年代建设的浅灰和米黄色住宅楼，阳台密集，窗户样式不统一。更远处则是一片明显更现代的城市核心区，高层写字楼和住宅塔楼从旧城区后方升起，玻璃表面反射清晨浅金色光线。低处仍残留少量薄雾，使不同距离的建筑产生自然空气透视。天空为浅蓝色，几条淡薄云带横向延伸。画面重点不是宏伟天际线，而是通过瓦屋、生活设施、旧住宅和现代高楼同时出现，表现城市长期生长形成的丰富层次。

English translation. A photorealistic urban photograph viewed across the rooftops of an old city district in the early morning. The foreground contains several rows of gray-black tiled roofs and reddish-brown brick walls. Gutters, television antennas, and rows of outdoor air-conditioning units can be seen along the eaves, while water tanks, clothes-drying racks, and potted plants sit on some flat roofs. Narrow lanes are hidden between the buildings; only through the gaps can one glimpse faint steam rising from breakfast stalls and several electric scooters passing slowly. The middle ground gradually transitions to pale-gray and beige residential blocks built in the 1980s and 1990s, with densely packed balconies and mismatched window styles. Farther away lies a distinctly more modern urban core, where high-rise office buildings and residential towers rise behind the old neighborhood, their glass surfaces reflecting the pale golden light of morning. A small amount of thin mist lingers at lower elevations, creating natural aerial perspective among buildings at different distances. The sky is pale blue, crossed horizontally by several faint, slender bands of cloud. The visual emphasis is not on a grand skyline, but on the rich layers created by tiled houses, everyday residential fixtures, older apartment buildings, and modern towers appearing together, expressing the city’s long-term organic growth.

<sub>Prompt #16</sub> 一张高清风景摄影作品，展现了中国黄山（<sub>Huangshan</sub>）冬季雪后的壮丽景色。画面前景被茂密的松树占据，深绿色的针叶上覆盖着厚实、蓬松的白雪，树枝向四周伸展，形成自然的画框。中景是陡峭险峻的灰白色花岗岩山峰，岩石纹理清晰，表面斑驳地覆盖着积雪，几株苍劲的松树顽强地生长在悬崖峭壁之上。在两座山峰之间的峡谷深处，云雾缭绕，白色的浓雾如丝绸般缠绕在山腰，营造出朦胧幽深的意境。在云雾掩映的山腰处，隐约可见一座依山而建的传统中式建筑，拥有深色的屋顶和木质结构，与周围险峻的自然环境融为一体。背景是连绵起伏的山峦，逐渐隐没在灰白色的阴沉天空和漫天的雾气中。整体色调以冷峻的灰白、墨绿和深褐为主，光线柔和且漫射，没有强烈的阴影，呈现出一种静谧、肃穆且充满东方水墨画意境的氛围。

English translation. A high-definition landscape photograph presenting the magnificent scenery of Huangshan, China, after a winter snowfall. Dense pine trees occupy the foreground, their dark-green needles covered in thick, fluffy white snow. Their branches extend in all directions to form a natural frame. The middle ground features steep, precipitous gray-white granite peaks with clearly visible rock textures and irregular patches of snow across their surfaces. Several vigorous pine trees grow tenaciously on the cliffs. Deep in the gorge between two peaks, clouds and mist swirl, with dense white fog wrapping around the mountainsides like silk and creating an ethereal sense of secluded depth. Partly concealed by mist on the mountainside, a traditional Chinese building constructed against the slope can be faintly seen. Its dark roof and wooden structure blend into the surrounding rugged natural environment. In the background, undulating mountain ranges gradually disappear into a gloomy gray-white sky and pervasive fog. The overall palette is dominated by austere gray-white, deep ink green, and dark brown. The light is soft and diffuse, with no harsh shadows, producing a tranquil, solemn atmosphere imbued with the poetic character of an East Asian ink landscape painting.

## Prompt #17

一张写实风格的风光摄影作品，画面主体是一座从左向右横向贯穿整个画面的桥梁，桥体几乎与画面下边缘平行，形成非常明确的水平视觉结构。桥梁位于画面中部偏下的位置，两端分别连接左右两侧的河岸或山体，不做斜向延伸，也不采用强烈透视消失点，整体以正侧面视角完整展示桥梁的长度和结构。桥体采用浅灰色混凝土与深色钢结构结合的现代设计，桥面平直修长，下方由数个间距均匀的桥墩支撑，桥墩垂直落入水中，结构比例真实。桥面上有少量汽车、公交车和行人经过，人物和车辆尺寸较小，用来体现桥梁的实际尺度。护栏、路灯、伸缩缝和桥面边缘都保留清楚的工程细节，但不要让车辆遮挡桥体主体。桥下是一条宽阔而平缓的河流，水面呈灰蓝色，微风形成细密波纹，桥墩和桥体在水面留下断断续续的倒影。前景靠近岸边的位置可以看到浅色石滩、芦苇、低矮灌木和几块被水冲刷得较圆润的岩石，使画面下方不显得空旷。河岸两侧分布着树木、步道和零散建筑，远处则是一层较低的城市天际线或起伏山脉，让桥梁保持绝对视觉主体。天空占据画面上半部分，采用阴天转晴后的浅蓝灰色调，云层有明显厚薄变化，局部阳光从云隙中照到桥面和水面，在桥体边缘形成柔和亮部。整体色彩自然克制，不使用夸张霞光，不追求奇幻感。画面强调桥梁横向展开的长度感、稳定感和空间秩序，桥体从画面左侧一直延伸到右侧，正侧面完整可见，构图平衡、开阔、真实。

English translation. A photorealistic landscape photograph whose principal subject is a bridge running horizontally across the entire frame from left to right. The bridge is almost parallel to the lower edge of the image, creating a very clear horizontal visual structure. It is positioned slightly below the center of the frame, with its two ends connecting to the riverbanks or mountains on the left and right. It must not extend diagonally or use a strong perspective vanishing point; instead, a direct side view should present the bridge’s full length and structure. The bridge has a modern design combining light-gray concrete with dark steel construction. Its deck is straight, long, and slender, supported below by several evenly spaced piers descending vertically into the water, with structurally realistic proportions. A small number of cars, buses, and pedestrians cross the bridge; the people and vehicles are small in scale to convey the bridge’s true dimensions. Engineering details such as guardrails, streetlights, expansion joints, and the edges of the deck remain clearly visible, but the vehicles must not obscure the main bridge structure. Beneath the bridge flows a broad, gently moving river with gray-blue water. A light breeze produces fine ripples, and the bridge and its piers cast intermittent reflections on the surface. Near the bank in the foreground are a pale stony shore, reeds, low shrubs, and several rocks rounded by water erosion, preventing the lower portion of the image from feeling empty. Trees, walking paths, and scattered buildings line both riverbanks, while a low city skyline or undulating mountain range appears in the distance, allowing the bridge to remain the unequivocal visual focus. The sky occupies the upper half of the image and has a pale blue-gray tone after an overcast sky has begun to clear. The cloud layers vary noticeably in thickness, and patches of sunlight pass through gaps in the clouds onto the bridge deck and water, forming soft highlights along the bridge’s edges. The overall colors are natural and restrained, without exaggerated sunset glow or a fantastical appearance. The image emphasizes the bridge’s horizontally extended length, stability, and spatial order. The bridge spans continuously from the left edge of the frame to the right, fully visible in direct side profile, in a balanced, expansive, and realistic composition.

## Prompt #18

一张拍摄于中国秦岭森林中的野生动物摄影作品，主体是一只中国金丝猴停在粗壮树枝上，身体微微侧向镜头，双手抓住树干，头部转向前方，神态警觉而安静。它拥有标志性的金色至橙黄色毛发，脸部带有淡蓝色调的皮肤，眼睛明亮，鼻部轮廓清晰，面部表情真实自然。毛发非常丰富，脸周、肩部和背部的毛发更长，能看到蓬松且有层次的真实质感。背景是中国北方山林环境，树干、苔藓、树叶和柔和的远景森林层次清楚但不杂乱。早晨薄云天气下的自然散射光照亮金丝猴，让毛发和脸部结构清晰呈现。真实自然生态摄影，高分辨率，长焦镜头拍摄感，细节丰富，真实毛发纹理，画面安静而有野外气息。

English translation. A wildlife photograph taken in the forests of China’s Qinling Mountains. The subject is a golden snub-nosed monkey perched on a thick tree branch, its body turned slightly toward the camera, both hands gripping the tree trunk, and its head turned forward with an alert yet calm expression. It has the species’ signature golden to orange-yellow fur, pale blue-toned facial skin, bright eyes, a clearly defined nose, and a realistic, natural facial expression. Its coat is exceptionally abundant, with longer hair around the face, shoulders, and back, revealing a fluffy, layered, lifelike texture. The background depicts a northern Chinese mountain-forest environment, where tree trunks, moss, leaves, and softly rendered distant forest layers are clearly distinguished without appearing cluttered. Natural diffuse light under thin morning clouds illuminates the monkey, clearly revealing its fur and facial structure. Authentic natural-history photography, high resolution, the visual feel of a telephoto-lens capture, rich detail, realistic fur texture, and a quiet image imbued with the atmosphere of the wild.

## Prompt

<sub>#19</sub> 一张拍摄于中国四川山地竹林中的野生动物纪实摄影作品，主体是一只成年大熊猫，正安静地坐在一片湿润的竹林地面上，前爪抱着一根新鲜竹子低头啃食。它的黑白毛发厚实蓬松，脸部圆润，黑色眼圈边缘自然，鼻子湿润，嘴边毛发因为咀嚼竹叶而微微散开。毛发质感真实，能够看到不同区域毛发的长短和方向变化，白色毛发不是纯白，带有轻微灰调和自然阴影，黑色毛发有真实层次。环境是四川常见的山地竹林，背景有绿色竹叶、潮湿泥土、苔藓和少量岩石，空气略带湿润感。阴天自然光柔和均匀地落在大熊猫身上，没有夸张高光，整体光线安静自然。真实野生动物摄影，高分辨率，自然色彩，纪实风格，皮毛细节清晰，整体像国家公园里真实拍到的一张照片。

English translation. A documentary wildlife photograph taken in a mountainous bamboo forest in Sichuan, China. The subject is an adult giant panda sitting quietly on the damp bamboo-forest floor, holding a fresh stalk of bamboo in its forepaws and lowering its head to eat. Its black-and-white fur is thick and fluffy; its face is round, the edges of its black eye patches are natural, its nose is moist, and the hair around its mouth is slightly splayed from chewing bamboo leaves. The fur texture is lifelike, showing variations in hair length and growth direction across different areas. The white fur is not pure white but carries a slight gray cast and natural shading, while the black fur has realistic tonal depth. The setting is a mountain bamboo forest typical of Sichuan, with green bamboo leaves, damp soil, moss, and a small number of rocks in the background, and a subtly humid quality to the air. Soft, even natural light from an overcast sky falls on the giant panda without exaggerated highlights, creating quiet and natural illumination throughout. Authentic wildlife photography, high resolution, natural colors, documentary style, clearly resolved fur details, and the overall appearance of a photograph genuinely captured in a national park.

## Prompt #20

一幅垂直构图的中国传统水墨山水画轴，画面描绘秋日山谷中的溪流与松林。近景是数株苍劲古松，树干以浓墨皴擦出粗糙树皮，枝叶层层展开。中景有一条浅溪从山石间穿过，水面以留白表现，岸边点缀几块灰褐色岩石和少量枯黄草木。远景群山层层递进，以淡墨晕染出薄雾笼罩的空间感。画面右上角以行楷竖排题写<sub>“</sub>空山新雨后，天气晚来秋<sub>”</sub>，在其左侧有小字写着<sub>“</sub>丙午年秋谢俊<sub>”</sub>，文字清晰完整，墨色略干，旁边钤有一方朱红色印章。画作绘制在略微泛黄的纸本上，边缘能够看到细小折痕和自然氧化痕迹。四周以米黄色绫边装裱，整体色彩以墨黑、灰褐、浅赭和少量青绿色为主，光线柔和均匀，呈现真实古画展陈的沉静质感。

English translation. A vertically composed hanging scroll in the style of traditional Chinese ink landscape painting, depicting a stream and pine forest in an autumn valley. In the foreground stand several vigorous ancient pines, their rough bark rendered with dense-ink texture strokes and their branches and needles unfolding in successive layers. In the middle ground, a shallow stream passes between mountain rocks; the water surface is represented by reserved blank paper, and several gray-brown rocks and a small amount of withered yellow vegetation dot the banks. In the distance, mountain ranges recede layer by layer, with pale-ink washes creating a sense of space enveloped in thin mist. In the upper-right corner, the inscription “<sup>空山新雨后，</sup>天<sup>气晚来秋</sup>” is written vertically in running-standard script. To its left, smaller characters read “<sup>丙午年秋谢俊</sup>”. All characters are clear and complete, the ink is slightly dry, and a vermilion seal is stamped beside them. The painting is executed on slightly yellowed paper, with tiny creases and traces of natural oxidation visible along the edges. It is mounted on all sides with beige silk borders. The overall palette consists mainly of ink black, gray-brown, pale ocher, and a small amount of blue-green. Soft, even lighting conveys the tranquil material presence of an authentic antique painting on display.

## Prompt #21

一幅书法作品特写，采用平铺视角拍摄，画面完全被米黄色的、带有金色丝线的宣纸背景填满，没有任何边框或留白。纸张表面呈现出自然的纤维纹理和细微的颗粒感，色调温暖且均匀。画面主体为两列竖向排列的黑色行草书法汉字，墨色浓郁，笔触苍劲有力，墨迹在纸面上有自然的晕染和飞白效果，显示出毛笔书写的动态感。右侧一列从上至下书写着 功不唐捐 四个大字，左侧一列从上至下书写着 玉汝于成 四个大字。最左侧中下方是落款，从上到下用小字写着<sub>“</sub>政淋题于丙午年秋<sub>”</sub>，落款正下方红色小猫头像印章清晰可见。字体结构舒展，笔画间有牵丝连带，墨色浓淡变化丰富，既有饱满的墨块，也有干枯的笔锋。整体风格古朴典雅，光线柔和均匀，无明显阴影，突出了书法艺术的质感和纸张的肌理。

English translation. A close-up photograph of a work of calligraphy, shot from a flat, directly overhead viewpoint. The frame is completely filled by a beige xuan-paper background threaded with gold-colored silk fibers, with no border or blank margin. The paper surface displays natural fibrous texture and a subtle grain, in a warm, even tone. The main subject consists of two vertical columns of black Chinese characters in running-cursive calligraphy. The ink is rich, the brushwork vigorous and forceful, and the ink naturally feathers across the paper with areas of dry-brush white streaking, conveying the dynamic motion of brush writing. The right-hand column reads, from top to bottom, the four large characters “<sup>功不唐捐</sup>”; the left-hand column reads, from top to bottom, the four large characters “<sup>玉汝于成</sup>”. At the middle-to-lower left is the signature, written vertically in smaller characters as “<sup>政</sup>淋<sup>题于丙午年秋</sup>”. Directly beneath the signature, a red seal depicting a small cat’s head is clearly visible. The character structures are open and expansive, with fine linking strokes connecting the brushwork. The ink varies richly in density, including both full, saturated masses and dry, depleted brush tips. The overall style is classically simple and elegant. Soft, even light with no obvious shadows emphasizes the material quality of the calligraphy and the texture of the paper.

## Prompt #22

一张垂直构图的高清摄影作品，主体为米开朗基罗（<sub>Michelangelo</sub>）著名雕塑《大卫》（<sub>David</sub>）的白色大理石半身像。雕塑置于画面中央，安放在一个覆盖着深红色织物的圆形台座上，背景是纯色的深红色墙面，墙面上方有横向的装饰线条。大卫的头部微微向右侧转，目光凝视右前方，神情专注而严肃，卷曲的头发雕刻精细，颈部肌肉线条清晰可见，左肩部分残缺。在雕塑的头顶上，巧妙地插着几朵鲜花：左侧有两朵盛开的白色玫瑰，花瓣层层叠叠，右侧有三朵白色雏菊，花蕊呈黄色，绿色的花茎从发丝间伸出，与白色的大理石形成鲜明对比。光线柔和且均匀，从正面照射，在雕塑的右侧和下方投下淡淡的阴影，突显了大理石的质感和花朵的娇嫩。整体色调以红色和白色为主，营造出一种古典艺术与自然生机相结合的氛围。

English translation. A vertically composed, high-definition photograph whose subject is a white marble bust of Michelangelo’s renowned sculpture David. The sculpture is centered in the frame on a round pedestal covered with deep-red fabric. The background is a solid deep-red wall with horizontal decorative molding near its top. David’s head turns slightly to the right, his gaze directed toward the front right with a focused, solemn expression. His curly hair is finely carved, the musculature of his neck is clearly visible, and part of his left shoulder is missing. Several fresh flowers are cleverly tucked into the top of the sculpture’s head: two fully bloomed white roses with layer upon layer of petals on the left, and three white daisies with yellow centers on the right. Their green stems rise from between the locks of hair, contrasting sharply with the white marble. The lighting is soft and even, falling from the front and casting faint shadows to the sculpture’s right and below, emphasizing both the marble’s texture and the flowers’ delicacy. The overall palette is dominated by red and white, creating an atmosphere that combines classical art with the vitality of nature.

<sub>Prompt #23</sub> 清康熙<sub>1662-1722</sub>年松花石龙凤硯，清康熙松花石龙凤砚，包含绿色石砚与紫色石砚盒。左侧为椭圆形绿色石砚，砚面打磨光滑，中央留有一粗椭圆形墨堂，表面可见细微墨痕，砚面通体分布深浅不一的绿色天然横纹。砚池呈新月形，砚面周边浮雕仿古凤纹，凤首位于池左上方，双翼向左右覆盖环绕砚周，周棱因凤羽纹呈连弧状，池右下方浮雕凤爪。右侧为紫色椭圆形砚盒，盒身与盒盖借砚身扣合，盒体由紫绿两层色石琢制，盒身为紫色，盒盖表面为暗紫色层中夹灰绿色层。盒盖周边镂空雕刻仿古子母夔龙纹，大龙在右，幼龙居左，大龙之首雕于上边正中，幼龙之首雕于右下角，二者上下遥望，大龙龙身下垂，小龙龙身上翘，环绕盒盖周边。向内剔去暗紫色层露出灰绿色层，雕出仿古变形带状夔龙纹，呈椭圆形镂空绕接于暗紫色子母夔龙纹，龙身阴刻方回纹、勾云纹、龙鳞等纹饰。纹饰下嵌有玻璃，可透视砚面。盒背周缘起一圈宽棱，表面有一层褐黄色涂料。整体置于浅灰背景上，光线均匀。

English translation. A Songhua-stone inkstone with dragon-and-phoenix decoration from the Kangxi reign of the Qing dynasty, 1662–1722; a Qing Kangxi Songhua-stone dragon-and-phoenix inkstone comprising a green stone inkstone and a purple stone inkstone case. On the left is an oval green stone inkstone with a smoothly polished surface. A broad oval grinding surface is reserved at the center, with faint traces of ink visible upon it, and natural horizontal green striations of varying lightness and darkness extend throughout the stone. The ink reservoir is crescent-shaped. Archaistic phoenix motifs are carved in relief around the edge of the inkstone: the phoenix’s head is positioned at the upper left of the reservoir, its two wings spread left and right to encircle the inkstone’s perimeter, the rim forming a sequence of connected arcs in response to the feather pattern, and a phoenix claw is carved in relief at the lower right of the reservoir. On the right is an oval purple inkstone case whose base and lid clasp around the body of the inkstone. The case is carved from stone with two layers of color, purple and green: the base is purple, while the lid’s surface consists of a dark-purple layer interspersed with a gray-green layer. Around the lid, openwork carving forms an archaistic paired parent-and-young kui-dragon design, with the large dragon on the right and the young dragon on the left. The large dragon’s head is carved at the center of the upper edge, and the young dragon’s head at the lower-right corner; they look toward each other from above and below. The large dragon’s body curves downward while the smaller dragon’s body turns upward, encircling the perimeter of the lid. Farther inward, the dark-purple layer has been cut away to reveal the gray-green layer, in which an archaistic, transformed ribbon-like kui-dragon motif is carved. This oval openwork design winds around and connects with the dark-purple parent-and-young kui-dragon motif. The dragons’ bodies are incised with square-spiral patterns, hooked-cloud patterns, scales, and other ornament. Glass is inlaid beneath the decoration, allowing the inkstone surface to be seen through it. A broad raised ridge runs around the back edge of the case, whose surface bears a layer of brownish-yellow coating. The entire object is placed against a light-gray background under even lighting.

## Prompt #24

明錢穀张复合画水程图（二）册高郵，一幅明代钱谷张复合绘制的《水程图（二）》册页，描绘了高邮的水乡风光。画面采用散点透视，前景为宽阔平静的水面，左下角有一处小岛，岛上生长着几株柳树，岸边停泊着几艘小船。中景展示了蜿蜒的河岸，岸边错落有致地分布着多座民居，屋顶多为黄色或灰色，房屋周围绿树环绕。一座石拱桥横跨在河道之上，连接着两岸。背景是连绵起伏的山丘，山脚下有一座宏伟的城楼，城楼上方可见城墙蜿蜒延伸，远处还有一座高耸的宝塔。画面右侧的山坡上也有房屋点缀。整幅画作以水墨淡彩为主，笔触细腻，色彩淡雅，营造出宁静悠远的意境。画面右侧中部有竖排汉字<sub>“</sub>高郵<sub>”</sub>。左下角有一方红色印章。整体构图疏密有致，展现了古代水陆交通的繁忙景象与水乡的秀美风光。

English translation. Water Route Map (II), Gaoyou, an album leaf jointly painted by Qian Gu and Zhang Fu in the Ming dynasty, depicting the waterside scenery of Gaoyou. The image employs shifting perspective. In the foreground is a broad, tranquil expanse of water, with a small island in the lower-left corner. Several willow trees grow on the island, and several small boats are moored along its shore. The middle ground presents a winding riverbank, where numerous residences with mostly yellow or gray roofs are arranged in a varied yet orderly manner and surrounded by green trees. A stone arch bridge spans the waterway, connecting the two banks. The background consists of undulating hills. At the foot of the hills stands a magnificent gate tower, above which a city wall can be seen winding into the distance; farther away rises a tall pagoda. Houses also dot the hillside on the right side of the image. The work is executed primarily in ink and light colors, with delicate brushwork and an understated palette that creates a tranquil, far-reaching mood. At the middle right of the image, the vertical Chinese characters “<sup>高郵</sup>” appear. A red seal is positioned in the lower-left corner. The overall composition balances dense and open areas, presenting both the bustle of ancient transportation by water and land and the graceful beauty of the waterside region.

## Prompt #25

一张垂直构图的极近距离美食摄影作品，聚焦于一块刚刚切开的厚切炸猪排。画面中央，一双深棕色木筷夹起一块猪排切片，炸衣呈均匀的金黄色，粗糙的面包糠颗粒清晰可见，边缘带有几处略深的焦黄色区域。切面中的猪肉厚实而柔嫩，颜色为自然的浅粉白色，纤维细密，内部保留着明显汁水，在切口边缘形成细小反光。下方是一只深色陶瓷餐盘，盘中整齐排列着剩余的猪排切片，旁边堆放着大量切得极细的浅绿色卷心菜丝，部分菜丝自然卷曲。盘子右侧放着一小勺亮黄色芥末和一滩浓稠的深棕色猪排酱，酱汁边缘带有自然流动痕迹。背景虚化，可见一碗白米饭和一只盛着味噌汤的黑色漆碗。光线从侧上方照射，使炸衣颗粒产生细密阴影，突出酥脆外层与湿润肉质之间的强烈质感对比。

English translation. A vertically composed, extreme close-up food photograph focusing on a freshly sliced, thick-cut tonkatsu. At the center of the image, a pair of dark-brown wooden chopsticks lifts one slice of the pork cutlet. The breaded crust is an even golden yellow, with the coarse panko crumbs clearly visible and several slightly darker, toasted golden areas along the edges. The pork exposed by the cut is thick and tender, naturally pale pinkish white, with fine fibers and visibly retained juices that form tiny highlights along the cut edge. Below is a dark ceramic plate on which the remaining pork-cutlet slices are arranged neatly, beside a generous mound of extremely finely shredded pale-green cabbage, some strands curling naturally. On the right side of the plate are a small spoonful of bright-yellow mustard and a pool of thick, dark-brown tonkatsu sauce, whose edges show natural traces of flowing. The blurred background reveals a bowl of white rice and a black lacquer bowl filled with miso soup. Light enters from the upper side, casting fine shadows among the breading particles and emphasizing the strong textural contrast between the crisp exterior and the moist meat.

## Prompt #26

一张写实风格的美食摄影作品，采用中近距离、略高于桌面的拍摄视角，主体是一份正在烤制中的韩式烤肉。画面中央是一只嵌入餐桌的不锈钢圆形烤盘，烤盘表面铺着数片厚薄不一的五花肉和牛肉片，肉片边缘已经卷起并呈现金黄与焦褐色过渡，中间仍保留粉棕色与油脂透明感，表面能看到明显的烤纹、油脂渗出后的微亮光泽，以及高温烤制时形成的细小焦边。部分肉片上点缀着蒜片、洋葱块和青椒片，烤盘边缘还能看到几朵正在加热的杏鲍菇和金针菇。烤盘中央微微冒起热气，局部可见油脂滴落后产生的细小泡沫和焦化痕迹，营造出真实的现烤状态。画面前景左侧摆放着一只白色小碗，里面是翠绿色生菜叶和紫苏叶，叶片新鲜舒展，表面带有轻微水珠。前景右侧是一只浅色蘸料碟，分别装有棕色麻油盐、红色韩式辣酱和切碎的蒜末。桌面上还分布着多只韩式小菜碟，包括橙红色泡菜、浅黄色腌萝卜片、凉拌豆芽、拌菠菜和几片白色洋葱丝，颜色丰富但不杂乱。画面右上角有一双金属筷子正夹起一片刚烤好的五花肉，肉片柔软下垂，边缘微焦，能够清楚看到肥瘦相间的层次。背景略微虚化，可以看到深色木纹桌面、不锈钢排烟管、一只装着米饭的不锈钢饭碗，以及一杯浅黄色啤酒。整体光线明亮而偏暖，重点突出烤肉表面的油脂光泽、热气、焦香质感和韩式烤肉店热闹真实的用餐氛围。

English translation. A realistic food photograph taken from a medium-close viewpoint slightly above tabletop height, featuring Korean barbecue as it cooks. At the center is a round stainless-steel grill inset into the dining table. Several slices of pork belly and beef of varying thickness cover its surface; their edges have curled and transition from golden to charred brown, while their centers retain pinkish-brown tones and translucent fat. Distinct grill marks, the faint sheen of rendered fat, and tiny charred rims created by high-temperature grilling are visible on the meat. Some slices are garnished with sliced garlic, chunks of onion, and pieces of green pepper, while several king oyster mushrooms and enoki mushrooms can be seen heating around the edge of the grill. A small amount of steam rises from the center, and in places tiny bubbles and caramelized traces caused by dripping fat are visible, creating the authentic appearance of meat being grilled to order. In the left foreground sits a small white bowl filled with vivid-green lettuce and perilla leaves; the leaves are fresh and unfurled, with fine droplets of water on their surfaces. In the right foreground is a light-colored dipping-sauce dish holding, separately, brown sesame oil with salt, red Korean chili paste, and minced garlic. Several small dishes of Korean banchan are also distributed across the table, including orange-red kimchi, pale-yellow pickled radish slices, seasoned bean sprouts, seasoned spinach, and several shreds of white onion; the colors are rich without appearing cluttered. In the upper-right corner, a pair of metal chopsticks lifts a freshly grilled slice of pork belly. The meat hangs softly, its edge lightly charred, and its alternating layers of fat and lean meat are clearly visible. The slightly blurred background shows a dark wood-grain tabletop, a stainless-steel exhaust duct, a stainless-steel rice bowl filled with rice, and a glass of pale-yellow beer. The overall lighting is bright and warm, emphasizing the oily sheen, rising heat, and char-grilled texture of the meat, as well as the lively, authentic dining atmosphere of a Korean barbecue restaurant.

## Prompt #27

一张垂直构图、高清晰度的美食摄影照片，近距离俯拍展示了一桌热气腾腾的潮汕牛肉火锅。画面中心是一个置于嵌入式电磁炉上的圆形不锈钢火锅，锅内清澈的牛骨汤正在剧烈沸腾，表面翻滚着密集的气泡，锅中可见一片柠檬和白色的萝卜片，一把不锈钢漏勺斜靠在锅沿右侧，上方升腾着白色的蒸汽。前景横向排列着三个长方形浅色竹木托盘，左侧托盘上整齐码放着红白相间的肥牛卷，纹理清晰；中间托盘上铺着色泽鲜红、质地紧实的嫩牛肉片，正中央点缀着一株翠绿的香菜；右侧托盘上则摆放着厚切的粉褐色牛舌片，同样装饰有香菜。背景左侧的木质桌面上，叠放着两个带有蓝色边缘的黄色陶瓷小碗，后方立着一个红色的方形纸巾盒，盒身印有白色二维码和文字，旁边还有一个米黄色的圆柱形调料罐。背景右上角模糊可见一盘新鲜的黄芽白菜。整体光线明亮均匀，呈现出餐厅室内暖色调的氛围，焦点集中在前景的肉片纹理和锅内的沸腾细节上，景深自然，色彩鲜艳诱人。

English translation. A vertically composed, high-definition food photograph showing, in a close overhead view, a steaming table of Chaoshan beef hot pot. At the center is a round stainless-steel hot pot set on an inset induction cooker. The clear beef-bone broth is boiling vigorously, with dense bubbles rolling across the surface; a slice of lemon and slices of white radish are visible in the pot. A stainless-steel skimmer rests diagonally against the right rim, while white steam rises above it. Three rectangular, light-colored bamboo trays are arranged horizontally across the foreground. On the left tray, red-and-white rolls of fatty beef are stacked neatly, with their marbling clearly visible. The center tray is covered with vividly red, firm-textured slices of tender beef, garnished at the exact center with a sprig of bright-green cilantro. The right tray holds thick-cut, pinkish-brown slices of beef tongue, likewise garnished with cilantro. On the wooden tabletop in the left background, two yellow ceramic bowls with blue rims are stacked together. Behind them stands a red square tissue box printed with a white QR code and text, next to a beige cylindrical condiment jar. A plate of fresh yellow-heart napa cabbage is faintly visible, out of focus, in the upper-right background. The lighting is bright and even, creating the warm-toned atmosphere of a restaurant interior. Focus is concentrated on the texture of the meat slices in the foreground and the boiling details inside the pot, with natural depth of field and vivid, appetizing colors.

竖版的摄影海报，真实质感的自然摄影风格，一只白色家鹅在深蓝色湖面上高速向镜头游来，主体占据画面左下至中央区域，鹅身体微微倾斜，白色羽毛湿润蓬松、纹理清晰，橙黄色宽大鹅嘴正对镜头并微微张开，神态生动、有点滑稽又充满冲劲；鹅快速划水，在身体周围激起大量透明水花和飞溅水珠，水珠被高速快门凝固在空中，细小水滴清晰锐利，形成强烈的动态感。湖水呈浓郁的宝石蓝、深海蓝与藏青色渐变，水面布满柔和波纹、反射和高光，背景只有纯净开阔的水面，不出现岸边、建筑或其他动物。采用近距离低机位抓拍视角，镜头几乎贴近水面，与鹅保持平视甚至略微俯视的感觉，鹅头位于画面中央偏左，身体从左下方延伸进入画面，尾部靠近左侧边缘，形成斜向前冲的视觉动势；画面右侧和右上方保留大面积平静的深蓝色负空间，使主体与背景形成明显的不对称构图。前景鹅与水花清晰锐利，远处湖面略微柔化，浅景深但整体环境仍具有真实摄影质感。自然日光照明，偏冷色调，鹅的白色羽毛与橙色鹅嘴在蓝色湖水中形成鲜明的冷暖色彩对比，高光自然，阴影柔和，曝光真实，具有专业野生动物摄影、抓拍摄影、杂志封面摄影的视觉品质。在画面右上方负空间加入极简主义英文排版，使用粗体白色无衬线字体，大号文字分成三行排列：<sub>“Stopexplainingyourself.”</sub>，文字左对齐，行距紧凑，视觉重量明显；主标题下方加入两行较小的白色<sup>英文副标</sup>题<sup>：</sup>“They heard you. They just liked the drama.”<sup>，再下方加入一行更小的中文</sup>“<sup>停</sup>止<sup>解</sup>释<sup>自己</sup>”<sup>。</sup>所有文字保持干净、现代、克制，与蓝色水面形成高对比，同时不遮挡鹅与飞溅水花。

English translation. A vertical photographic poster in a natural-photography style with realistic texture. A white domestic goose swims rapidly toward the camera across a deep-blue lake, occupying the area from the lower left to the center of the image. Its body tilts slightly; its white feathers are wet, fluffy, and clearly textured. Its broad orange-yellow bill points straight at the camera and is slightly open, giving the goose a vivid expression that is a little comical yet full of momentum. As it paddles quickly, the goose throws up abundant transparent splashes and airborne droplets around its body. A high-speed shutter freezes the water in midair, rendering even the tiny droplets crisp and sharp and producing a powerful sense of motion. The lake transitions through rich jewel blue, deep-ocean blue, and navy, with soft ripples, reflections, and highlights across its surface. The background contains only clean, open water, with no shore, buildings, or other animals. Use a close-range, low-angle candid viewpoint, with the lens almost touching the water and viewing the goose at eye level or from very slightly above. Place the goose’s head just left of center; its body enters the image from the lower left, with its tail near the left edge, creating a diagonal, forward-surging visual movement. Preserve a large expanse of calm, deep-blue negative space on the right and upper-right, creating a distinctly asymmetric composition between subject and background. Keep the foreground goose and splashes crisp and sharp while softening the distant lake slightly. Use shallow depth of field, but retain the realistic photographic quality of the overall environment. Illuminate the scene with natural daylight in a cool palette. The goose’s white feathers and orange bill should form a vivid warm–cool color contrast against the blue water, with natural highlights, soft shadows, and realistic exposure, achieving the visual quality of professional wildlife photography, candid action photography, and magazine-cover photography. In the negative space in the upper right, add minimalist English typography in bold white sans-serif type. Set the large text over three lines: “Stop explaining yourself.” Align it left, use tight leading, and give it clear visual weight. Beneath the main title, add the smaller white English subtitle over two lines: “They heard you. They just liked the drama.” Below that, add an even smaller line of Chinese text, “<sup>停</sup>止<sup>解释自己</sup>”. Keep all text clean, modern, and restrained, with high contrast against the blue water, while ensuring that it does not obscure the goose or the splashing water.

<sub>Prompt #29</sub> 一张手机拍摄的博物馆展品说明牌照片，展品旁的标准说明牌格式，包含文物名称年代材质尺寸和简要描述，设计简约底色为深灰或黑色配白色文字。顶部居中的画面顶部文字层以中等大小字清晰可见：<sub>”</sub>青花瓷盘（明代<sub>1403-1424</sub>）<sub>”</sub> 中央的字幕条以小字居中清晰显示：<sub>”</sub>材质：瓷器产地：景德镇尺寸：口径<sub>45.2cm</sub> 足径<sub>28.5cm</sub> 高<sub>8.3cm</sub> 馆藏编号：故瓷<sub>001256”</sub>。底部注释栏以左对齐小字呈现：<sub>”</sub>故宫文物博物院陶瓷馆请勿触摸展品展厅温度：<sub>20</sub>±<sub>2</sub><sup>◦</sup><sub>C</sub> 湿度：<sub>45%-55%</sub> 来源：清宫旧藏入藏时间：<sub>1925</sub>年<sub>”</sub> 展签底部的导览信息区以小号字体标注着：<sub>”</sub>语音讲解编号：<sub>TC-0256</sub> 时长：<sub>3</sub>分<sub>28</sub>秒<sub>”</sub>

English translation. A smartphone photograph of a museum exhibit label, using the standard format of an information plaque placed beside an artifact. It includes the artifact’s name, period, material, dimensions, and a brief description. The minimalist design uses white text on a dark-gray or black background. Centered at the top, a medium-sized top text layer is clearly visible and reads: ”<sup>青花瓷盘（明代</sup>1403-1424<sup>）</sup>”. At the center, a caption strip displays the following clearly in small, centered type: ”<sup>材质：瓷器产地：景德镇</sup>尺<sup>寸：</sup> <sup>口径</sup>45.2cm <sup>足径</sup>28.5cm <sup>高</sup>8.3cm <sup>馆藏编号：故瓷</sup>001256”. In the bottom notes area, the following appears in small, left-aligned type: ”<sup>故宫文物博物院陶瓷馆请勿触摸</sup>展品展<sup>厅温度：</sup>20±2<sup>◦</sup>C <sup>湿度：</sup>45%-55% <sup>来源：清</sup> <sup>宫旧藏入藏时间：</sup>1925<sup>年</sup>”. In the visitor-guide information area at the bottom of the label, small type reads: ”<sup>语音讲解编号：</sup>TC-0256 <sup>时长：</sup>3<sup>分</sup>28<sup>秒</sup>”.

<sub>Prompt#30</sub> 竖版东方美学艺术海报，融合传统中国工笔画、古典宫廷建筑细密插画与当代平面设计语言。整体背景为大面积温润的米白色、象牙白宣纸质感，带有细腻自然的纸张纤维与轻微颗粒纹理，画面克制，大量留白。画面中央由左下向右上贯穿一条巨大而厚重的黑色墨刷笔触，形成强烈的斜向对角线构图。笔触边缘带有明显的干笔飞白、断裂墨迹、毛笔拖拽感和破碎缺口，局部墨块向外飞散，富有书法般的力量感。大面积米白留白与浓黑笔触形成极强视觉反差。在黑色笔触内部嵌入一组沿中轴层层展开的故宫建筑群，整体呈现紫禁城的秩序感与皇家气象。包括朱红宫墙、金黄色琉璃瓦、汉白玉台基、宽阔石阶、宫门、殿宇、回廊与庭院，建筑由下向上逐层展开，形成清晰的纵深与节奏。建筑细节精密，具有传统界画与工笔重彩插画风格。建筑之间点缀整齐而繁茂的古树，树冠色彩以深青蓝、孔雀蓝、青绿与金黄色为主，象征秋日宫苑景观。局部穿插乳白色传统祥云纹样，卷云造型细腻典雅。庭院、台阶与宫门前分布少量古代人物，人物极小，身着传统服饰，在宫殿之间行走、驻足、交谈，为恢弘宫阙注入人文气息。整体色彩以象牙白、墨黑、朱砂红、琉璃黄、孔雀蓝、青绿为主，层次高级，兼具传统中国审美与现代设计海报的精致感。主体景观严格被限制在黑色笔触内部，边缘局部被飞白切断，强化<sub>“</sub>从墨中浮现<sub>”</sub>的视觉效果。构图极端不对称，主体沿斜向聚集，四周保持大面积干净留白。左上角加入极小而克制的中文排版，使用纤细金色高雅字体，两行排列：<sub>“</sub>故宫博物院<sub>” “</sub>紫禁城<sub>”</sub> 旁边搭配一个极简几何圆形徽记，低调精致，不喧宾夺主。底部中央加入极小字号、字距疏朗的中文文字：<sub>“</sub>宫阙秋深<sub>”</sub>两侧加入细线分隔，右下角加入四个极简轮廓几何符号。整体呈现高端文化展览海报、博物馆视觉识别、现代东方平面设计风格，中国风编辑海报，故宫建筑工笔插画，宣纸纹理，飞白墨刷构图，大面积负空间，极精细细节，竖版构图

English translation. A vertical art poster with an Eastern aesthetic, blending traditional Chinese meticulous gongbi painting, finely detailed illustration of classical imperial architecture, and contemporary graphicdesign language. The overall background consists of a large expanse of warm off-white and ivory-white xuan-paper texture, with delicate natural paper fibers and a subtle grain. Keep the image restrained and leave abundant negative space. A huge, heavy black ink-brush stroke runs through the center from the lower left to the upper right, creating a powerful diagonal composition. Its edges show pronounced dry-brush reserve white, broken ink marks, the dragging texture of the brush, and fractured gaps; localized chunks of ink scatter outward, conveying calligraphic force. The broad off-white negative space and dense black stroke create an extremely strong visual contrast. Embed within the black stroke a group of Forbidden City buildings unfolding layer by layer along the central axis, conveying the order of the Forbidden City and the grandeur of imperial authority. Include vermilion palace walls, golden-yellow glazed roof tiles, white-marble terraces, broad stone stairways, palace gates, halls, corridors, and courtyards. Arrange the architecture in successive tiers rising from bottom to top to create clear depth and rhythm. Render the architectural details with precision, in the style of traditional ruled-line architectural painting and richly colored gongbi illustration. Between the buildings, place orderly, luxuriant ancient trees whose crowns are predominantly deep blue-green, peacock blue, turquoise green, and golden yellow, evoking an imperial garden in autumn. Interweave localized creamy-white traditional auspicious-cloud motifs, with delicately elegant scrolling-cloud forms. Distribute a small number of ancient figures in the courtyards, on the steps, and before palace gates. The figures should be tiny, dressed in traditional clothing, walking, pausing, and conversing among the palaces, adding human presence to the magnificent architecture. Use a principal palette of ivory white, ink black, cinnabar red, glazed-tile yellow, peacock blue, and blue-green. The result should feel sophisticated and layered, combining traditional Chinese aesthetics with the refinement of a modern design poster. Strictly confine the main landscape inside the black brushstroke, with parts of its edges interrupted by dry-brush white gaps, reinforcing the visual effect of the scene “emerging from the ink.” Make the composition extremely asymmetric, with the subject clustered along the diagonal and large areas of clean negative space maintained on all sides. In the upper left, add extremely small, restrained Chinese typography in a slender, elegant gold typeface, arranged in two lines: “<sup>故宫博物院</sup>” and “<sup>紫禁城</sup>”. Pair it with a minimalist geometric circular emblem that is understated and refined and does not compete with the subject. At the bottom center, add very small Chinese text with generous tracking: “<sup>宫阙秋</sup>深”. Place fine dividing lines on both sides, and add four minimalist outlined geometric symbols in the lower right. The overall result should resemble a high-end cultural-exhibition poster, museum visual identity, and modern Eastern graphic design: a Chinese-style editorial poster, detailed gongbi illustration of Forbidden City architecture, xuan-paper texture, dry-brush ink composition, abundant negative space, extremely fine detail, and vertical composition.

## Prompt #31

横版城市文旅插画海报，现代极简排版结合东方国风风景插画，整体画面为高端、干净、雅致的旅游城市宣传海报风格。背景为大面积温润的米白色、浅米灰色纯净底色，带有轻微纸张质感，整体留白充足，版式整洁，视觉高级。画面主体由超大英文单词<sub>“HANGZHOU”</sub> 构成，字母占据画面中央大部分区域，采用粗壮、狭长、规整的无衬线大写字体，字母内部不是纯色，而是作为 镂空取景窗 ，每个字母中都填充精致的杭州风景插画。字母内部呈现连续但又各自独立的江南景观，包括西湖水面、古典亭台楼阁、石拱桥、宝塔、荷花荷叶、柳枝、园林水榭、曲折步道、湖边树木、远山、茶园梯田，以及现代高楼建筑群。每个字母都像一个微缩景观切片，内部构图精致丰富，形成<sub>“</sub>城市风景被装进文字里<sub>”</sub>的视觉效果。在<sub>“HANGZHOU”</sub>各字母内部，重点体现杭州代表性元素：古典宝塔高耸，立于湖山之间；石拱桥横跨水面，桥上有细小行人；西湖边有亭台、水榭、小船、曲桥；湖面上点缀荷叶与淡粉色荷花；局部出现茶园梯田与采茶人物；一侧融入现代杭州城市天际线与高楼群，体现传统与现代并存；柳枝从部分字母顶部自然垂下，增强江南柔美气质；个别字母中出现圆门、庭院、松树、湖岸步道等中式园林元素。画面最上方横向排布一条细长的杭州全景插画带，像一条城市长卷。长卷中从左到右依次出现：远山、古塔、拱桥、湖面、亭台、现代城市天际线、湖上列车或轻轨、另一侧古塔与垂柳，形成传统与现代并置的城市轮廓线。天空极简，仅保留淡淡的云朵、几只飞鸟，以及左上方一轮低饱和暖红色圆日，营造安静、诗意、东方审美的氛围。整体插画风格细腻、平面化、装饰性强，介于国风绘本、旅游宣传插画、文创海报与高级信息图形设计之间。色彩柔和克制，采用低饱和配色，以浅青绿、灰蓝、湖蓝、豆绿色、浅卡其、淡粉、米白、青灰为主，避免高对比和浓烈艳色。水面清透平静，远山朦胧，建筑精巧，植物枝叶柔和细密，整体呈现温润、轻盈、安静、诗意的江南气质。构图上强调字体海报设计感： 是绝对视觉中心；顶部的横向城市长卷作为辅助信息层；背景大面积留白，画面呼吸感强；整体对称中带有轻微变化，版式平衡而现代；文字与插画高度融合，兼具平面设计感与叙事性。画面风格要求：高端城市宣传海报、杭州文旅视觉、国风城市插画、现代排版设计、文字景观融合、极简留白、雅致清新的东方配色、精致细节、平面插画质感、横版海报。

English translation. A horizontal illustrated urban-culture-and-tourism poster combining modern minimalist typography with an Eastern, Chinese-style landscape illustration. The overall image should have the highend, clean, and elegant look of a promotional poster for a tourist city. Use a large expanse of warm off-white and pale beige-gray as a pristine background, with a subtle paper texture, ample negative space, a tidy layout, and a sophisticated visual finish. Construct the main subject from the enormous English word “HANGZHOU,” occupying most of the center of the image in bold, condensed, regular sans-serif capitals. The interiors of the letters should not be solid color; instead, treat them as “cutout viewing windows,” each filled with a refined illustration of Hangzhou scenery. Within the letters, depict Jiangnan landscapes that are continuous yet individually distinct, including the surface of West Lake, classical pavilions and towers, stone arch bridges, pagodas, lotus flowers and leaves, willow branches, waterside garden pavilions, winding paths, lakeside trees, distant mountains, terraced tea plantations, and clusters of modern high-rises. Each letter should resemble a miniature landscape cross-section with an intricate, rich internal composition, creating the visual effect of “city scenery placed inside the text.” Within the letters of “HANGZHOU,” prominently feature representative Hangzhou elements: a tall classical pagoda standing amid lake and mountains; a stone arch bridge spanning the water, with tiny pedestrians crossing it; pavilions, waterside halls, small boats, and curved bridges beside West Lake; lotus leaves and pale-pink lotus blossoms scattered across the lake; terraced tea fields and tea pickers in selected areas; a modern Hangzhou skyline and clusters of high-rises integrated on one side to express the coexistence of tradition and modernity; willow branches hanging naturally from the tops of some letters to enhance the gentle Jiangnan character; and Chinese-garden elements such as moon gates, courtyards, pine trees, and lakeside paths inside individual letters. Across the very top of the image, arrange a slender horizontal band of panoramic Hangzhou illustration like a long city handscroll. From left to right, this scroll should show distant mountains, an ancient pagoda, an arch bridge, the lake, pavilions, a modern urban skyline, a train or light rail over the lake, and another ancient pagoda with weeping willows, forming a city silhouette in which the traditional and the modern are juxtaposed. Keep the sky extremely minimal, retaining only faint clouds, a few flying birds, and a low-saturation warm-red circular sun in the upper left, creating a quiet, poetic atmosphere with an Eastern aesthetic. The overall illustration style should be delicate, flat, and highly decorative, positioned between a Chinese-style picture book, tourism-promotion illustration, cultural-creative poster, and high-end infographic design. Use a soft, restrained, low-saturation palette dominated by pale blue-green, gray-blue, lake blue, muted bean green, light khaki, pale pink, offwhite, and blue-gray, avoiding high contrast and intense colors. Make the water clear and calm, the distant mountains hazy, the architecture finely rendered, and the foliage soft and detailed, giving the whole image the warm, light, quiet, poetic character of Jiangnan. Emphasize the feel of typographic poster design in the composition: “HANGZHOU” must be the absolute visual center; the horizontal city handscroll at the top serves as a secondary information layer; a large amount of background negative space gives the image room to breathe; the composition should be broadly symmetric with slight variations, creating a balanced, modern layout; and text and illustration should be deeply integrated, combining graphic-design clarity with narrative richness. Required visual style: high-end city-promotion poster, Hangzhou cultural-tourism identity, Chinese-style urban illustration, modern typographic design, integration of text and landscape, minimalist negative space, elegant and fresh Eastern colors, refined details, flat-illustration texture, and a horizontal poster format.

## Prompt #32

竖版现代东方文化海报，温暖象牙白纸张背景。右上角一支真实毛笔正在绘制一道浅青绿色与湖蓝色混合的长弧形水墨笔触。整条笔触既是毛笔留下的颜料，也是苏州河道与水巷。沿蓝绿色笔触从左下向右上布置精致微缩苏州景观。左下前景是一片白墙黛瓦江南民居，建筑紧邻水巷，两岸有石板路和垂柳，一座拱形石桥跨过水面，小木船从桥洞中穿过。向中央逐渐出现苏州园林，月洞门、太湖石、亭台、水榭、回廊和松树组成精致的园林微缩景观。中段以虎丘塔作为主要城市文化地标，塔身从青绿色树木中升起。随后传统园林逐渐过渡至现代苏州城市轮廓，可以克制加入现代高楼和金鸡湖城市意象，但现代建筑不能压过古城气质。最后所有景观逐渐缩小，并在右上角融入毛笔笔尖。整体为低饱和湖蓝、竹青、黛绿、暖白、淡灰和少量砖红，气氛细腻、安静、文雅。指定文字顶部：<sub>SUZHOU CHINA</sub> 副标题：<sub>A City Between</sub>Gardens and Water <sup>中部：</sup>Canal Whispers <sup>小字：</sup>Bridges, gardens and quiet lanes unfold beside the water.<sup>底部：</sup>Gardens of Time <sup>下一行：</sup>White walls, dark roofs and ancient canals preserve the rhythm of Jiangnan.<sup>最后：</sup>Poetry in Every Turn <sup>下一行：</sup>Cross a bridge, enter a garden, and discover another view. <sup>只允许出现</sup>以上指定文字，全部准确清晰渲染。

English translation. A vertical modern Eastern cultural poster on a warm ivory-white paper background. In the upper right, a realistic calligraphy brush is painting a long, arcing ink stroke that blends pale blue-green with lake blue. The entire stroke is both the pigment left by the brush and the waterways and canal lanes of Suzhou. Arrange exquisite miniature Suzhou scenery along the blue-green stroke from the lower left to the upper right. In the lower-left foreground, show a group of Jiangnan residences with white walls and dark-tiled roofs standing immediately beside a canal lane. Include stone-paved paths and weeping willows on both banks, an arched stone bridge spanning the water, and a small wooden boat passing through the bridge opening. Moving toward the center, gradually introduce Suzhou gardens: moon gates, Taihu rocks, pavilions, waterside halls, covered corridors, and pine trees forming an intricate miniature garden landscape. In the middle section, use Tiger Hill Pagoda as the principal urban-cultural landmark, its tower rising from blue-green trees. Then gradually transition from the traditional gardens to the modern Suzhou skyline. Modern high-rises and visual references to Jinji Lake may be added with restraint, but the modern architecture must not overpower the character of the old city. Finally, make all scenery gradually diminish in scale and merge into the brush tip in the upper right. Use an overall low-saturation palette of lake blue, bamboo green, dark green, warm white, pale gray, and small amounts of brick red, creating a delicate, quiet, and refined atmosphere. Use only the following specified text, rendered accurately and clearly. At the top: SUZHOU CHINA. Subtitle: A City Between Gardens and Water. In the middle: Canal Whispers. Small text: Bridges, gardens and quiet lanes unfold beside the water. At the bottom: Gardens of Time. On the next line: White walls, dark roofs and ancient canals preserve the rhythm of Jiangnan. Finally: Poetry in Every Turn. On the next line: Cross a bridge, enter a garden, and discover another view. No text other than the specified text may appear.

## Prompt #33

竖版潮流编辑海报，夜晚街头摄影与高饱和实验平面设计结合。背景是一条深蓝色夜间街道，一台略显老旧的白色自动售货机独自立在街角，机器内部发出冷白与浅青色灯光，周围环境整体被处理为强烈蓝色调。地面略微湿润，可以看到冷蓝反光。一罐橙黄色汽水刚刚从机器出口滚出来，成为画面唯一明显暖色视觉焦点。自动售货机后方叠加一个巨大亮黄色圆形几何图形。顶部使用巨大黄色高对比衬线标题：<sub>”Small Joys”</sub>下方中文：<sub>”</sub>小小快乐<sub>”</sub> 底部中文：<sub>”</sub>快乐不必有正当理由<sub>”</sub> 左下：<sub>”</sub>今天也值得<sub>” ”</sub>奖励自己<sub>”</sub> 右侧：<sub>”</sub>随时供应<sub>” ”</sub>快乐不断<sub>”</sub> 底部细小英文：<sub>”A LITTLEJOY IS STILLJOY”</sub> 外围使用深海军蓝宽边框下方加入白色细线和三个向右白色三角形。整体使用深蓝、亮黄、橙色强烈撞色，颗粒胶片、粗网点、丝网印刷和潮流独立杂志视觉。只允许出现以上双引号内的中文和英文。

English translation. A vertical trend-focused editorial poster combining nighttime street photography with highly saturated experimental graphic design. The background is a deep-blue street at night, where a slightly old white vending machine stands alone on a street corner. Its interior emits cool-white and pale-cyan light, and the surrounding environment is treated in an intense blue palette. The ground is slightly wet, showing cool-blue reflections. A can of orange-yellow soda has just rolled out of the machine’s dispensing slot, becoming the image’s only clearly warm-colored visual focal point. Overlay a huge, bright-yellow circular geometric shape behind the vending machine. At the top, use an enormous yellow high-contrast serif title: ”Small Joys”. Below it, place the Chinese text: ”<sup>小小快乐</sup>”. At the bottom, place the Chinese text: ”<sup>快乐不必有</sup>正<sup>当理由</sup>”. In the lower left: ”<sup>今</sup>天<sup>也值得</sup>” and ”<sup>奖励自己</sup>”. On the right: ”<sup>随时供应</sup>” and ”<sup>快乐</sup> <sup>不断</sup>”. At the bottom, in very small English type: ”A LITTLE JOY IS STILL JOY”. Surround the image with a wide, deep-navy border. At the bottom, add a fine white line and three right-pointing white triangles. Use a forceful color clash of deep blue, bright yellow, and orange, together with grainy film texture, coarse halftone dots, screen printing, and the visual language of a fashionable independent magazine. Only the Chinese and English text inside the quotation marks above may appear.

## Prompt #34

横版极简硬科幻电影概念海报， 宽幅构图，融合复古科幻小说封面、现代主义电影海报和粗颗粒丝网印刷质感。整体仅使用深黑、温暖奶油白、暗金黄色以及极少量冷蓝色。整个画面采用明显的左右不对称构图。左侧约 的区域完全由超大字体占据，右侧约 用于恒星、宇航员与外星文明视觉。左侧使用巨大、粗壮、狭窄的奶油米白色无衬线字体，三行排列：<sub>”SAVE””THE””SUN”</sub>文字几乎顶到画面上下边缘，字号极大、字距紧密，形成如建筑结构般的巨大文字墙。部分字母被右侧恒星圆盘侵入和遮挡，使字体与图像不是简单并排，而是彼此咬合。右侧放置一颗占据几乎整个画面高度的巨大恒星。恒星并非写实太阳照片，而是一枚由暗金、米白和粗颗粒印刷纹理构成的巨大圆盘，右侧边缘仍然明亮，左侧部分逐渐被黑色侵蚀，形成恒星正在失去能量的视觉隐喻。巨大恒星与左侧文字局部重叠，例如太阳边缘切入<sub>”SUN”</sub>最后几个字母。恒星前方下部放置一名极小的孤独宇航员，人物背对镜头，站在狭窄的飞船观察平台上，仰望巨大恒星。宇航员尺寸非常小，用尺度差异强化宇宙的压迫感。宇航员旁边加入一个完全不同于人类科技的非对称外星机械结构，由金属几何体、多面体、支架和奇异接口组成，不设计成恐怖怪物，而是让两个文明的科技同时面对同一个问题。在右上角留出少量黑色负空间，准确排版：<sub>”PROJECT HAIL MARY”</sub> 其<sup>下：</sup>”ONE STAR. TWO WORLDS. ONE CHANCE.” <sup>右下角较小字体：</sup>”THE SUN IS DYING. THE ANSWERIS LIGHT-YEARS AWAY.” <sup>最下方：</sup>”SCIENCE IS THE ONLY WAY HOME.” <sup>所有文字必须完整、锐利、拼</sup>写准确。禁止伪文字、乱码、额外标志和水印。整体强调巨大字体、宇宙尺度、孤独感、科学合作、复古硬科幻印刷质感。

English translation. A horizontal minimalist hard-science-fiction film concept poster in a 16:9 widescreen composition, blending the look of a vintage science-fiction novel cover, a modernist film poster, and coarsegrained screen printing. Use only deep black, warm cream white, dark golden yellow, and a very small

## Prompt #35

amount of cool blue. Give the entire image a clearly asymmetric left–right composition. Approximately 55% of the left side is occupied completely by oversized typography, while approximately 45% on the right is reserved for imagery of a star, an astronaut, and an alien civilization. On the left, use enormous, bold, condensed, cream-beige sans-serif letters arranged over three lines: ”SAVE”, ”THE”, and ”SUN”. The text should nearly touch the upper and lower edges of the image, with an extremely large font size and tight tracking, forming an immense typographic wall like an architectural structure. Allow the stellar disk on the right to intrude upon and obscure parts of some letters so that text and image do not simply sit side by side but interlock. On the right, place a gigantic star occupying almost the full height of the image. The star must not look like a realistic photograph of the Sun; instead, make it a huge disk composed of dark gold, cream white, and coarse-grained print textures. Its right edge remains bright, while its left side is gradually eroded by black, creating a visual metaphor for a star losing its energy. Let the giant star partially overlap the text on the left; for example, its edge may cut into the last few letters of ”SUN”. In the lower area before the star, place a tiny, solitary astronaut with their back to the camera, standing on a narrow spacecraft observation platform and gazing up at the immense star. Keep the astronaut extremely small, using the disparity in scale to intensify the oppressive immensity of space. Beside the astronaut, add an asymmetric alien mechanical structure entirely unlike human technology, composed of metallic geometric solids, polyhedra, supports, and unfamiliar interfaces. Do not design it as a frightening monster; instead, show the technologies of two civilizations confronting the same problem together. Leave a small area of black negative space in the upper right and typeset the following accurately: ”PROJECT HAIL MARY”. Beneath it: ”ONE STAR. TWO WORLDS. ONE CHANCE.” In smaller type in the lower right: ”THE SUN IS DYING. THE ANSWER IS LIGHT-YEARS AWAY.” At the very bottom: ”SCIENCE IS THE ONLY WAY HOME.” All text must be complete, sharp, and spelled correctly. Prohibit pseudo-text, garbled characters, additional logos, and watermarks. Overall, emphasize monumental typography, cosmic scale, solitude, scientific cooperation, and the print texture of vintage hard science fiction.

横版荒诞现代编辑海报， 构图，复古办公室摄影、幽默动物摄影与独立杂志拼贴设计结合。浅灰白色办公室背景，画面右侧偏中位置站着一匹真实的成年马，体型健壮，毛色自然，身体完整地进入画面。马以一种平静而略显荒诞的方式出现在办公室中，四蹄稳稳站在地面上，身体微微侧向镜头，头部自然转向前方，目光看向镜头，神态自信、倔强、理直气壮，带有一种<sub>“</sub>我今天就是不想工作<sub>”</sub>的冷静态度。马保持真实的身体比例、骨骼结构与肌肉质感，站姿自然，不做夸张拟人化动作。它的脖子上挂着一张办公室工牌，工牌通过挂绳套在颈部，尺寸相对偏小，形成轻微幽默感。工牌清晰可见，但不过分抢眼，可以写上简单英文<sub>”STAFF”</sub> 。背景极简，仅保留一张办公桌和一台关闭的笔记本电脑，办公室空间略显空旷，进一步强化<sub>“</sub>马居然是这里的员工<sub>”</sub>这种荒诞反差。左侧上方使用巨大黑色高对比衬线字体：<sub>”Not Today”</sub> 下面小型中文：<sub>”</sub>今天不行<sub>”</sub> 中部加入橙色斜体英文：<sub>”Maybe Tomorrow?”</sub> 左下排布中文：<sub>”</sub>不是没能力<sub>” ”</sub>只是今天不<sup>想发挥</sup>” <sup>下方小号英文：</sup>”REST IS ALSO PART OF THE PLAN” <sup>最后一行：</sup>”<sup>先坐一会</sup>” <sup>整体主要使</sup>用<sup>米白、</sup>钴蓝、橙黄和黑色，构图为左侧文字、右侧主体，具有复古半色调印刷、杂志扫描颗粒、荒诞高级编辑视觉。只能出现以上双引号内指定的中文和英文，不得生成任何其他语言、伪文字、乱码、随机字符、水印或品牌标志。

English translation. A horizontal absurdist modern editorial poster in a 16:9 composition, combining vintage office photography, humorous animal photography, and independent-magazine collage design. Against a pale gray-white office background, position a real adult horse slightly right of center. It has a sturdy build and natural coat color, and its entire body is visible within the frame. The horse appears in the office in a calm yet mildly absurd manner: all four hooves stand firmly on the floor, its body is angled slightly toward the camera, its head turns naturally forward, and its gaze meets the camera. Its expression is confident, stubborn, and unapologetically self-assured, conveying the calm attitude of “I simply do not feel like working today.” Preserve realistic body proportions, skeletal structure, and muscular texture. Give it a natural standing pose without exaggerated anthropomorphic gestures. Hang an office ID badge from its neck on a lanyard. The badge should be relatively small, creating subtle humor; it must be clearly visible without becoming overly prominent and may carry the simple English word ”STAFF”. Keep the background minimal, retaining only a desk and a closed laptop. Make the office feel somewhat empty, further reinforcing the absurd contrast that “the horse is actually an employee here.” In the upper left, use enormous black high-contrast serif type: ”Not Today”. Beneath it, in small Chinese text: ”<sup>今</sup>天<sup>不行</sup>”. In the middle, add orange italic English text: ”Maybe Tomorrow?” In the lower left, arrange the Chinese text: ”<sup>不是没能力</sup>” and ”<sup>只是今</sup>天<sup>不想发挥</sup>”. Below that, add small English text: ”REST IS ALSO PART OF THE PLAN”. On the final line: ”<sup>先坐一会</sup>”. Use primarily off-white, cobalt blue, orange-yellow, and black. Compose the image with text on the left and the subject on the right, and give it vintage halftone printing, scanned-magazine grain, and a sophisticated absurdist editorial look. Only the specified Chinese and English text inside the quotation marks above may appear; do not generate any other language, pseudo-text, garbled text, random characters, watermarks, or brand logos.

竖版中国文学主题艺术海报，围绕李清照一剪梅中的<sub>“</sub>月满西楼<sub>”</sub>进行视觉创作。整体融合中国古典楼阁建筑、秋夜意境、旧式木刻印刷、碑拓质感、丝网印刷与现代编辑设计，呈现安静、怀旧、含蓄而庄重的东方文学气质。背景是一张具有明显纤维质感的浅奶油色旧宣纸，纸面温润、略微泛黄，可以看到自然的纸纤维、灰色细小斑点、旧纸颗粒与轻微磨损痕迹。画面不是写实摄影，而是具有明显的平面印刷感和旧版画质感。画面上半部分中央偏右悬挂一轮极其巨大的金黄色满月，月亮占据显著视觉面积，呈完整圆形，颜色为温暖而浓郁的金黄色。月面具有轻微斑驳、颜料不均、旧印刷颗粒与自然色块变化，不做逼真的月球坑洞，也不产生真实摄影式照明，而是一枚具有象征意味的巨大暖色圆盘。画面中央略偏左位置矗立一座高而狭窄的中国传统多层楼阁式古塔，采用低机位仰视角度。古塔明显窄于背后的满月，使金色圆月从塔身两侧大面积露出。古塔采用深靛青紫色印刷剪影表现，但保留丰富建筑细节：多层飞檐、向上翘起的檐角、斗拱结构、窗格、栏杆、层层收窄的塔身、顶部细长塔刹，以及少量细长结构线。建筑边缘并非完全锐利，而带有木版印刷般的细小缺口、干墨和颗粒。画面下半部分由一组斜向延伸的传统瓦屋顶构成主要前景。黛瓦屋檐从左下向右侧斜向穿过画面，层层瓦片、屋脊和飞檐结构清晰，但全部被处理为靛青、灰紫与淡紫色的粗糙印刷剪影。屋檐周围生长大量秋季树木和枝叶。树冠与枝条具有细密自然的植物结构，同时被压缩成类似旧版画的平面色块。部分枝叶从画面左右边缘伸入，与楼阁和满月形成遮挡关系。画面右侧增加一座较小的传统亭阁或楼阁，隐藏在浓密树木之间，用于丰富层次，但不能抢夺中央古塔的视觉中心。整体构图形成非常明确的视觉关系：巨大金色满月→ 狭长靛青古塔→ 斜向古建筑屋顶→ 秋季枝叶。顶部和两侧保留较多奶油色负空间，下半部分则逐渐被淡紫、灰紫与深靛青建筑和植物填满，形成上轻下重的传统海报结构。主色严格控制为复古低饱和东方配色：大面积浅奶油宣纸色，接近<sub>#E7E0B8</sub> 下半部分大面积淡薰衣草紫，接近<sub>#AFAAD0</sub> 建筑、枝叶与主要文字使用深靛青紫，接近<sub>#575991</sub> 次级建筑使用柔和灰紫色，接近<sub>#928FC0</sub>局部使用中等紫罗兰色，接近<sub>#7473A8</sub> 满月使用明显集中的暖金黄色少量文字可以使用与满月呼应的金黄色作为强调整体不要艳丽紫色，也不要现代霓虹感，强调旧印刷中的紫、蓝、奶油色与金黄色关系。印刷与材质采用粗粝的双色或多色丝网印刷、<sub>Risograph</sub> 孔版印刷、传统木刻版画和碑拓质感。纸张上具有：轻微套色错位、半色调颗粒、颜料堆积、干墨断裂、墨色深浅不均、细小漏印点、灰尘颗粒和自然老化痕迹。建筑瓦片、树皮、树叶和斗拱仍然保留精细结构，不要因为复古印刷效果而完全变成模糊色块。大号中文字体尤其重要：字体应该像旧式木刻标题字或碑刻拓印字，字形方正、紧凑、直立，笔画厚重、横平竖直，末端钝而有力，仅有克制的干墨侵蚀。不要草书，不要行书，不要夸张毛笔飞白，不要长笔锋，不要书法飘带感。文字排版所有需要真正出现在海报中的文字，都必须严格按照双引号中的内容渲染。左上角放置一个尺寸很大的单独汉字： 月 使用深靛青色木刻式方正字体，字形紧凑、直立、厚重，表面有少量旧印刷磨损。上方中央使用小号宋体或明体横向排版：<sub>”</sub>雁字回时，月满西楼<sub>”</sub> 文字纤细、规整，与巨大标题形成明显字号反差。在右上方奶油色负空间内预留一块宽约画面<sub>38%</sub>的横向标题区域。严格使用一整行：<sub>”</sub>满西楼<sub>”</sub>三个字必须横向并列在同一条基线上，绝对不能换行，不能上下堆叠。字体采用巨大、厚重、方正的木刻印刷字体，深靛青紫色。笔画横平竖直，字形紧凑，带有轻微干墨颗粒与旧版画磨损，但主体结构必须清晰稳定。不要使用草书或自由毛笔字。画面中央偏右使用单列纤细、规整的竖排宋体文字：<sub>”</sub>轻解罗裳，独上兰舟。<sub>”</sub>文字不要过大，与建筑和月亮保持明显的视觉层级。左下方使用一行较小的金黄色中文作为局部视觉强调：<sub>”</sub>云中谁寄锦书来<sub>”</sub> 与巨大金色满月形成色彩呼应。整体效果是一张李清照文学主题的新中式复古编辑海报：巨大的金色满月压在画面上方，深靛青古塔从淡紫色屋顶与秋树之间向月亮升起，建筑与枝叶像旧时代木刻印刷在泛黄宣纸上；文字克制而具有碑拓感，在怀旧印刷质地中呈现<sub>“</sub>雁字回时，月满西楼<sub>”</sub>的清冷、思念与秋夜意境。

English translation. A vertical art poster themed around Chinese literature, visually interpreting “<sup>月满西楼</sup>” from Li Qingzhao’s poem “<sup>一剪梅</sup>”. Blend classical Chinese pavilion and tower architecture, an autumn-night mood, old-fashioned woodblock printing, the texture of stone-rubbing impressions, screen printing, and modern editorial design to create a quiet, nostalgic, restrained, and dignified Eastern literary sensibility. The background is a sheet of aged, pale-cream xuan paper with a pronounced fibrous texture. Its surface is warm and slightly yellowed, showing natural paper fibers, tiny gray specks, old-paper grain, and slight signs of wear. The image must not be realistic photography; it should have a clear flat-print character and the texture of an antique print. In the upper half, slightly right of center, suspend an extremely large golden-yellow full moon. The moon occupies a prominent visual area, forms a complete circle, and has a warm, saturated golden-yellow color. Its surface shows subtle mottling, uneven pigment, aged-print grain, and natural variations in blocks of color. Do not depict realistic lunar craters or photographic lighting; instead, render it as a huge, symbolic warm-colored disk. Slightly left of center, erect a tall, narrow, traditional Chinese multi-story pavilion-style pagoda, viewed from a low upward-looking angle. The pagoda is visibly narrower than the full moon behind it, allowing broad areas of the golden disk to remain exposed on both sides of the tower. Render the pagoda as a deep indigo-purple printed silhouette while preserving rich architectural detail: multiple tiers of projecting eaves, upturned corners, dougong bracket sets, lattice windows, railings, a body narrowing tier by tier, a slender finial at the top, and a few fine structural lines. Its edges should not be perfectly sharp, but should show tiny gaps, dry ink, and grain like a woodblock print. Form the main foreground in the lower half from a group of traditional tiled roofs extending diagonally. Dark-tiled eaves run diagonally from the lower left toward the right side of the image. The layered tiles, ridgelines, and projecting-eave structures remain clear, but all are treated as rough printed silhouettes in indigo, gray-purple, and pale purple. Surround the eaves with abundant autumn trees, branches, and foliage. The crowns and branches should retain delicate, natural botanical structure while being compressed into flat blocks of color resembling an antique print. Let some branches and leaves enter from the left and right edges and overlap the pavilion and full moon. Add a smaller traditional pavilion or tower on the right, hidden among dense trees, to enrich the depth, but do not let it compete with the central pagoda as the visual focus. Establish a very explicit compositional relationship: enormous golden full moon → slender indigo pagoda → diagonally arranged ancient tiled roofs → autumn branches and leaves. Preserve ample cream negative space at the top and on both sides, while gradually filling the lower half with pale-purple, gray-purple, and deep-indigo architecture and vegetation, producing a traditional poster structure that is light above and visually heavy below. Strictly control the principal palette as a vintage, low-saturation Eastern color scheme: a large expanse of pale-cream xuan-paper color, approximately #E7E0B8; a broad area of pale lavender in the lower half, approximately #AFAAD0; deep indigo-purple for the architecture, branches, foliage, and main typography, approximately #575991; soft gray-purple for secondary architecture, approximately #928FC0; medium violet in selected areas, approximately #7473A8; and a distinctly concentrated warm golden yellow for the full moon. A small amount of text may use golden yellow as an accent echoing the moon. Avoid vivid purple and any modern neon feeling; emphasize the relationship among purple, blue, cream, and gold found in aged printing. For printing and material qualities, use rough two-color or multicolor screen printing, Risograph stencil printing, traditional woodblock-print texture, and the texture of stone-rubbing impressions. The paper should show slight color-registration misalignment, halftone grain, accumulated pigment, broken dry ink, uneven ink density, tiny unprinted pinholes, dust particles, and natural aging marks. Preserve fine structures in roof tiles, tree bark, leaves, and dougong; do not allow the vintage-print effect to reduce everything to blurred blocks of color. The large Chinese typography is especially important: it should resemble old woodblock title type or carved inscription characters reproduced by rubbing, with square, compact, upright forms; heavy strokes; level horizontals and vertical uprights; blunt, forceful terminals; and only restrained dry-ink erosion. Do not use cursive script, semi-cursive script, exaggerated flying-white brushwork, long brush tips, or flowing calligraphic ribbons. For typography, every piece of text that actually appears on the poster must be rendered strictly as written inside the quotation marks. In the upper left, place one very large standalone Chinese character: ”<sup>月</sup>”. Use a deep-indigo, square woodblock-style typeface with a compact, upright, heavy form and a small amount of aged-print wear on its surface. At the upper center, horizontally typeset in small Songti or Mingti type: ”<sup>雁字回时，月满西楼</sup>”. Keep the lettering slender and regular, creating a clear contrast in scale with the enormous title. Reserve a horizontal title area approximately 38% of the image width within the cream negative space in the upper right. Use strictly one unbroken line: ”<sup>满西楼</sup>”. The three characters must be arranged horizontally side by side on the same baseline and must never wrap or stack vertically. Use enormous, heavy, square woodblock-print lettering in deep indigo-purple, with level horizontal and vertical strokes, compact character forms, slight dry-ink grain, and worn antique-print texture, while keeping the primary structures clear and stable. Do not use cursive or freely brushed calligraphy. Slightly right of center use a single vertical column of slender, orderly Songti text: ”<sup>轻解罗裳，独上兰舟。</sup>” Keep it modest in size and clearly subordinate to the architecture and moon. At the lower left, use one line of smaller golden-yellow Chinese text as a localized visual accent: ”<sup>云中谁寄锦书来</sup>”. This should echo the color of the enormous golden moon. The overall result is a new-Chinese-style vintage editorial poster themed around Li Qingzhao’s literature: an immense golden full moon presses over the upper image; a deep-indigo pagoda rises toward it from pale-purple roofs and autumn trees; architecture and foliage appear woodblock-printed on yellowed xuan paper; and restrained typography with the texture of a stone rubbing conveys the cool solitude, longing, and autumn-night mood of “<sup>雁字回时，月满西楼</sup>” through nostalgic print textures.

偏方形食物科普海报， 构图，复古植物学图鉴、自然史插画与现代食品编辑设计结合。背景为温暖象牙白色旧纸张，具有轻微纸纤维、泛黄和细颗粒印刷质感。画面中央放置最大主体：一串成熟的小番茄连着绿色藤蔓，果实大小略有不同，表皮呈自然鲜红和橙红色，高光柔和，使用精致植物学水彩方式描绘，不是摄影，不是卡通。主体周围不规则分布四幅小型科学插画：左上是一株带叶、花朵和未成熟绿色果实的小番茄植株；上方中央是几颗不同成熟程度的小番茄，从绿色、橙黄色逐渐变化到鲜红色；右侧是一颗从中间切开的番茄横截面，清楚表现果肉、籽腔、种子与汁液结构；左下是一小组不同形状的小番茄，包括圆形、椭圆形、梨形果实。右上使用巨大黑色中文标题：<sub>”</sub>小番茄<sub>”</sub>下方黑色衬线英文：<sub>”CherryTomato”</sub>左侧放置简洁信息：<sub>”</sub>口味：<sub>” ”</sub>酸甜<sub>·</sub> 清爽<sub>·</sub> 多汁<sub>”</sub> 右下放置：<sub>”</sub>主要营养：<sub>” ”</sub>维生素<sub>C” ”</sub>番茄红素<sub>” ”</sub>膳食纤维<sub>”</sub> 底部加入一句很小的文字：<sub>”</sub>一颗小小的红色果实<sub>”</sub>整体颜色以番茄红、橙红、叶片绿、嫩黄和米白为主，复古自然博物学海报，科学食物插画，精细水彩，旧植物图谱，少量文字，大面积留白。

English translation. A nearly square educational food poster in a 1:1 composition, combining a vintage botanical field guide, natural-history illustration, and modern food-editorial design. Use warm ivory-white aged paper as the background, with subtle paper fibers, yellowing, and fine-grained print texture. Place the largest subject at the center: a cluster of ripe cherry tomatoes attached to a green vine. The fruits vary slightly in size, their skins naturally bright red and orange-red with soft highlights. Render them as refined botanical watercolor, not as photography and not as cartoons. Distribute four small scientific illustrations irregularly around the main subject. In the upper left, show a cherry-tomato plant with leaves, flowers, and unripe green fruit. At the upper center, show several cherry tomatoes at different stages of ripeness, transitioning from green through orange-yellow to vivid red. On the right, show a cross-section of a tomato cut through the middle, clearly depicting the flesh, seed cavities, seeds, and juice structure. In the lower left, show a small group of cherry tomatoes of different shapes, including round, oval, and pear-shaped fruit. In the upper right, use an enormous black Chinese title: ”<sup>小番茄</sup>”. Beneath it, set the black serif English text: ”Cherry Tomato”. On the left, place concise information: ”<sup>口味：</sup>” and ”<sup>酸甜</sup>· <sup>清爽</sup>· <sup>多汁</sup>”. In the lower right, place: ”<sup>主要营养：</sup>”, ”<sup>维生素</sup>C”, ”<sup>番茄红素</sup>”, and ”<sup>膳食纤维</sup>”. At the bottom, add one very small line of text: ”<sup>一</sup> <sup>颗小小的红色果实</sup>”. Use primarily tomato red, orange-red, leaf green, tender yellow, and off-white. The result should resemble a vintage natural-history poster, scientific food illustration, fine watercolor, and an old botanical plate, with little text and abundant negative space.