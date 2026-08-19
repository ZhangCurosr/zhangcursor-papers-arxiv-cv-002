# SE-MoLoRA: Shared-Expert LoRA Adapters for Domain-Specific Photographic Assessment

Bishwash Khanal bishwash.b.khanal@jyu.fi   
Faculty of Information Technology   
University of Jyväskylä, Finland

Anlan Zhang anlanz@adobe.com Adobe Research, San Jose, CA, USA

Sasu Tarkoma sasu.tarkoma@helsinki.fi   
Department of Computer Science   
University of Helsinki, Finland   
Tommi Mikkonen tommi.j.mikkonen@jyu.fi   
Faculty of Information Technology   
University of Jyväskylä, Finland   
Abhishek Kumar abhishek.k.kumar@jyu.fi   
Faculty of Information Technology   
University of Jyväskylä, Finland

## Abstract

Vision-language models can describe images fluently, but they often fail to provide ac tionable photographic critique because semantic content and aesthetic judgment remain entangled. We propose SE-MoLoRA, a modular parameter-eficient adaptation framework for domain-specific photographic assessment. The method separates general photographic knowledge from specialist residual judgments using an always-active shared LoRA expert and routed adapters for composition, lighting, and technical quality. A lightweight query router selects the relevant specialist, enabling targeted critique without training separate full models. A rank-64 shared adapter captures broad photographic vocabulary, while rank-32 specialists learn domain-specific residuals with an orthogonal regularization penalty that encourages disentangled representations. Training data is obtained by distilling the Reddit Photo Critique Dataset into domain-labeled critique samples. On held-out critique generation, SE-MoLoRA improves BERTScore-F1 from 0.2317 to 0.4215 over monolithic LoRA and is preferred in 84.6% of pairwise comparisons, while using fewer active parameters than separate specialist models. SVD-based ablation study shows that shared-specialist decomposition and orthogonal regularization reduce expert overlap. These results demonstrate that modular adaptation improves controllability and specificity in multimodal photographic cri tique.

## 1 Introduction

Recent Vision-Language Models (VLMs) have achieved strong general-purpose visual understanding, yet their ability to provide meaningful aesthetic critique remains limited Jiang & Chen (2025); Wang et al. (2025); Huang et al. (2024b). For example, a model may describe “a beautiful sunset over the ocean”, but it fails to tell a photographer whether the horizon is tilted or the composition lacks a clear anchor point. This gap between general visual understanding and aesthetic understanding is not merely an issue of data volume but a structural limitation of monolithic architectures: the recognition of what appears in an image is entangled with how well the scene has been captured Yuksekgonul et al. (2023). This entanglement produces well-documented biases such as the halo efect, where models prefer semantically appealing subjects over sound technical judgment Gulati et al. (2025).

Existing approaches to automated aesthetic assessment each address only part of this problem. Score-based methods reduce aesthetic judgment to a single scalar rating, providing no actionable insight for photographers Talebi & Milanfar (2018). Monolithic VLM fine-tuning combines all aesthetic domains in a single parameter space, which entangles knowledge representations and causes catastrophic forgetting Ghosh et al. (2024); Li et al. (2024b). Training separate full-sized models for each domain is prohibitively expensive, as memory and latency scale linearly with the number of domains Houlsby et al. (2019). Mixture-of-Experts (MoE) architectures ofer a modular alternative, but they fail on vague user intent and are computationally expensive to train Shazeer et al. (2017). These limitations highlight the need for an approach that is modular, parameter-eficient, and equipped with a reliable routing mechanism.

To address the above-mentioned challenges, in this study, we propose Shared-Expert Mixture-of-LoRA (SE-MoLoRA), a modular parameter-eficient framework that separates domain-agnostic photographic knowledge from specialist aesthetic reasoning through a hierarchical adapter design. A permanently active shared adapter anchors general visual understanding, while conditionally routed specialists provide targeted critique for composition, lighting, and technical quality. Trained on community critiques distilled into domain-specific supervision from the Reddit Photo Critique Dataset Nieto et al. (2022), SE-MoLoRA produces more specific and disentangled critiques than monolithic fine-tuning while using fewer active parameters than separate specialist models (Section 4). Our contributions are twofold:

1. We introduce SE-MoLoRA, a parameter-eficient shared-specialist LoRA architecture for photographic critique, where an always-active shared adapter captures general photographic language and routed specialist adapters model composition, lighting, and technical residuals.

2. We show that modular shared-specialist adaptation improves critique alignment and domain specificity over zero-shot and monolithic LoRA baselines, while requiring fewer active parameters than separate specialist models.

## 2 Related Work

## 2.1 Vision-Language Models for Aesthetic Assessment

Vision-language models (VLMs) have evolved from domain-specific systems into broadly capable multimodal assistants Li et al. (2024a), integrating strong visual encoders with large language backbones to improve visual question answering, scene interpretation, and reasoning-intensive tasks Gao et al. (2023); Bai et al. (2025); Achiam et al. (2023); Team et al. (2023). Despite these advances, current VLMs remain strong generalists rather than reliable specialists: they identify objects and produce fluent descriptions, but these strengths do not transfer to aesthetic visual understanding automatically Jiang & Chen (2025); Wang et al. (2025); Huang et al. (2024b).

Photographic critique demands a diferent kind of visual reasoning. The model must explain why a framing choice helps or harms an image and how lighting supports or undermines subject emphasis. A model may correctly describe a portrait as backlit yet fail to analyze whether that backlighting preserves facial details or causes avoidable highlight clipping. These failures indicate that visual description and expert aesthetic reasoning are related but separate capabilities. A compounding concern is semantic-aesthetic bias, where models associate aesthetic quality with semantic content rather than technical properties Gulati et al. (2025), causing visually appealing subjects to receive disproportionately positive evaluations even when technical quality is poor.

Visual instruction tuning Liu et al. (2023) partially addresses this gap, but naive adaptation can introduce catastrophic forgetting, degrading broader visual reasoning while improving the target task Ghosh et al. (2024); Li et al. (2025). These observations motivate an architecture that preserves the base model’s visual grounding while injecting domain-specific critique capability through modular adapters, rather than monolithic fine-tuning.

## 2.2 Parameter-Eficient Adaptation and Modular Routing

Low-Rank Adaptation (LoRA) Hu et al. (2022) has become the dominant parameter-eficient fine-tuning strategy, augmenting frozen models with trainable low-rank decomposition matrices that approximate full fine-tuning at negligible inference overhead once merged. The framework has since been extended in several directions: quantization-aware variants Dettmers et al. (2023) for memory eficiency, DoRA Liu et al. (2024) for expressivity through magnitude-direction decoupling, and LoRA+ Hayou et al. (2024) for improved convergence. An important consideration is rank selection: increasing rank improves adaptation capacity but raises overfitting risk Zhang et al. (2023), motivating higher-rank shared components for cross-domain knowledge and lower-rank specialists for domain-specific residuals.

To route inputs to specialized adapters, we draw on Mixture-of-Experts (MoE) architectures, which enable conditional computation by gating a subset of experts per input Jacobs et al. (1991); Shazeer et al. (2017); Fedus et al. (2022). Recent adapter-centric implementations extend MoE from token-level feed-forward modules to modular adaptation layers, routing over LoRA blocks rather than dense experts Pfeifer et al. (2021); Huang et al. (2024a); Feng et al. (2024). While classical MoE routing operates at the token level, adapter-based routing occurs at the task or query level, making it suitable for domain-oriented applications where specialization is determined by overall input context.

The DeepSeek MoE architecture Dai et al. (2024) introduces shared experts alongside routed experts, providing always-active representations while routed experts specialize in domain-specific patterns. This concept has been adapted to multi-task LoRA learning Yang et al. (2025) and is particularly relevant when domains exhibit substantial overlap, as in photographic critique. SE-MoLoRA combines these ideas: it adopts the shared-expert concept from DeepSeekMoE but implements it entirely through parameter-eficient LoRA adapters, enabling MoE-style specialization without MoE-scale compute.

## 2.3 Residual Learning and Expert Disentanglement

Specialist adapters must learn distinct features to justify their modular cost. In modular adaptation, domain experts should learn incremental diferences on top of shared knowledge rather than full mappings from scratch, following the principle that optimizing residual mappings is easier than optimizing un-referenced mappings He et al. (2016). Chaining multiple LoRA adapters through iterative residual updates has been shown to approximate full-parameter fine-tuning while maintaining computational eficiency Xia et al. (2024). This motivates a hierarchical adaptation: a frozen base model preserves general multimodal capabilities, a shared adapter captures cross-domain photographic knowledge, and specialist adapters learn domain-specific residuals. This hierarchy also addresses the stability-plasticity dilemma Kirkpatrick et al. (2017), where freezing the base model preserves prior capabilities while adapters provide integration of new domains.

However, if multiple specialists learn similar representations, modular capacity is wasted and every expert collapses into a shared expert. This risk is acute for photographic critique, where aesthetic attributes and semantic content are frequently entangled. Orthogonal regularization addresses this by penalizing cosine similarity between adapter weight matrices during training, inspired by concept-level disentanglement in VLMs Li et al. (2024b) and DisLoRA Yifei et al. (2025), which shows that enforcing orthogonal task-specific directions in LoRA updates prevents representational collapse. A further challenge is spectral dominance, where a shared expert’s activations may overtake specialist contributions during inference, addressable through linear weighting or model merging techniques like TIES Yadav et al. (2023) and DARE Yu et al. (2024). Our training pipeline directly addresses both: orthogonal regularization prevents specialist convergence, while asymmetric adapter weighting preserves specialist contributions at inference.

## 2.4 Image Aesthetic Assessment

Automated Image Aesthetic Assessment (IAA) has shifted from score prediction Talebi & Milanfar (2018); Murray et al. (2012) through attribute-based approaches predicting factors like composition, lighting, and depth of field Chang et al. (2017); Jin et al. (2019), to multimodal frameworks for critique generation. Recent approaches finetune foundational VLMs for aesthetic critique, often leveraging newly curated datasets Hentschel et al. (2022); Huang et al. (2024b); Zhou et al. (2024); Wu et al. (2024), while others investigate reasoning strategies like chain-of-thought prompting and reinforcement learning to improve grounded critique Jiang & Chen (2025); Liu et al. (2025). Architectures such as PhotoEye Qi et al. (2025) combine multiple vision encoders with language models, showing that aesthetic assessment benefits from multiple viewpoints. The Reddit Photo Critique Dataset (RPCD) Nieto et al. (2022) captures community feedback from hobbyists to professional photographers, providing diverse critique styles particularly useful for LoRA-based fine-tuning. Unlike PhotoEye, which fuses multiple vision encoders with full-parameter fine-tuning, SE-MoLoRA achieves modular specialization through lightweight adapters on a single frozen backbone, targeting controllable, domain-specific critique generation under parameter-eficient constraints.

## 2.5 Overall positioning of SE-MoLoRA

Prior MoE-LoRA methods that primarily target multi-task or general instruction adaptation, SE-MoLoRA is designed for domain-specific photographic critique, where domains are semantically overlapping rather than cleanly separated. This motivates three design choices: an always-present shared photographic adapter, lowrank residual specialists, and explicit regularization to reduce expert overlap. The contribution is therefore not only the use of multiple LoRA adapters, but the decomposition of photographic critique into shared aesthetic language plus routed residual expertise.

## 3 Methodology

## 3.1 LLM-Distilled RPCD Dataset

The first requirement is domain-labeled training data. We build on the Reddit Photo Critique Dataset (RPCD) Nieto et al. (2022), which pairs approximately 74,000 images with 220,000 free-form community critiques. Because raw comments are noisy and often span multiple aesthetic domains, we apply LLM-based distillation to decompose them into domain-specific training samples.

All comments associated with a given post are concatenated and processed with a constrained instruction prompt to decompose the critique into four expert domains. The seven original PCCD attributes Jin et al. (2019) are grouped by thematic similarity: general\_impression and subject\_of\_photo form the shared domain; composition maps to the composition domain; color\_lighting maps to the lighting domain; and focus, depth\_of\_field, and use\_of\_camera form the technical domain. This process yields approximately 47,000 domain-labeled training samples.

## 3.2 SE-MoLoRA Architecture

SE-MoLoRA decomposes a single monolithic adapter into a permanently active shared expert and a set of conditionally activated specialist experts. The architecture is built on top of Qwen3-VL-8B Bai et al. (2025), which serves as the frozen base model.

As shown in Figure 1, the system consists of three layers: (1) the frozen base model $W _ { \mathrm { b a s e } } .$ , (2) a shared LoRA adapter $\Delta W _ { s }$ (rank-64) that is always active and captures domain-agnostic photographic vocabulary, and (3) three specialist LoRA adapters (rank-32 each) for composition, lighting, and technical analysis, of which at most one is activated per query. The total weight during inference is:

$$
W _ { \mathrm { o u t } } = W _ { \mathrm { b a s e } } + \lambda _ { s } \Delta W _ { s } + \lambda _ { e } \Delta W _ { e ^ { * } } \quad \mathrm { w h e r e } \quad e ^ { * } = \mathrm { R o u t e } ( q )\tag{1}
$$

where $\lambda _ { s }$ and $\lambda _ { e }$ control the relative influence of the shared and specialist adapters, and $e ^ { * }$ is the specialist selected by the router for query q.

The shared expert is trained on the general\_impression and subject\_of\_photo domains, aligning the model to professional photography terminology without introducing aesthetic bias. It also serves as a fallback for ambiguous queries that cannot be routed to a specialist. The specialist experts each focus on distinct aesthetic attributes: the composition expert targets framing, leading lines, and visual balance; the lighting expert targets exposure, white balance, and color grading; and the technical expert targets focus accuracy, depth of field, noise, and lens artifacts.

![](images/50adbcb316c7e1b151f42b31334e0b049fa6a033b40b4f9e32c316892cdde7a3.jpg)  
Figure 1: The SE-MoLoRA architecture. A shared adapter handles general photographic knowledge, while specialized adapters for composition, lighting, and technical aspects are conditionally activated based on the router’s decision.

## 3.3 Training Strategy

Three-stage pipeline. Training all adapters jointly risks the same domain entanglement the architecture aims to eliminate. We therefore adopt a three-stage residual learning pipeline. In Stage1, the shared expert adapter is trained on approximately 15K samples using standard supervised fine-tuning with causal language modeling. In Stage2, the trained shared adapter is permanently merged into the base model weights, producing an enhanced base model $W _ { \mathrm { n e w } } = W _ { \mathrm { b a s e } } + \Delta W _ { s }$ . This merging step ensures that specialist gradients in Stage3 flow only through their own low-rank matrices rather than through the shared adapter, preventing vanishing gradient and interference issues. In Stage3, each specialist adapter is trained independently on 5K domain-specific samples on top of $W _ { \mathrm { n e w } }$ . Specialists never observe each other’s gradients, minimizing cross-domain interference.

Orthogonal regularization. Despite domain-separated training data, specialist adapters can still learn overlapping representations. To enforce disentanglement, we add an orthogonal regularization term in the standard loss function (Equation 2) that penalizes cosine similarity between specialist weight updates (Equation 3):

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { S F T } } + \lambda _ { \mathrm { o r t h o } } \cdot \mathcal { L } _ { \mathrm { o r t h o } }\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { o r t h o } } = \sum _ { l \in L } \frac { \vert \mathrm { v e c } ( \Delta W _ { \mathrm { c u r r } } ^ { ( l ) } ) \cdot \mathrm { v e c } ( \Delta W _ { \mathrm { r e f } } ^ { ( l ) } ) \vert } { \vert \Delta W _ { \mathrm { c u r r } } ^ { ( l ) } \vert F \vert \Delta W \mathrm { r e f } ^ { ( l ) } \vert F }\tag{3}
$$

where $\Delta W _ { \mathrm { c u r r } }$ is the adapter being trained and $\Delta W _ { \mathrm { r e f } }$ are the weights of previously trained specialists. Specialists are trained sequentially: the composition expert is trained first, then the lighting expert uses the composition weights as reference, and the technical expert uses both.

Spectral dominance mitigation. Due to the rank asymmetry between the shared (rank-64) and specialist (rank-32) adapters, the shared adapter’s larger singular values can dominate the final output, drowning out specialist contributions. We address this with asymmetric weighting: $\lambda _ { s } = 0 . 5$ and $\lambda _ { e } = 1 . 5$ in Equation 1, which rebalances the relative influence during inference.

## 3.4 Routing Mechanism

At inference time, the system must decide which specialist to activate based on the user’s query. Unlike token-level MoE routing Shazeer et al. (2017), our routing operates at the query level to determine overall user intent. We develop and compare two architectures:

Text-only router. A fine-tuned DistilBERT model (67M parameters) performs four-class classification over the expert domains. The training data combines combinatorially generated prompt-label pairs from domain-specific vocabulary with naturalistic queries from the PCCD dataset Jin et al. (2019) to balance keyword coverage with linguistic variability. This router is fast and lightweight but cannot leverage visual cues when user prompts are vague.

Multimodal router. To handle ambiguous queries where the image itself contains diagnostic cues (e.g., “What do you think?” paired with an overexposed image), we introduce a CLIP ViT-B/32-based multimodal classifier. Frozen CLIP encoders extract image and text embeddings in parallel, which are normalized, concatenated, and passed through a trainable MLP classification head (260K trainable parameters). Training labels are derived from PCCD attribute scores: images with disproportionately low scores in a specific attribute are labeled as requiring that domain’s specialist.

## 4 Experiments

## 4.1 Implementation Details

All experiments use a single NVIDIA A100 (40GB) with bfloat16 precision, Unsloth’s memory-optimized training, and 8-bit AdamW. The shared expert (rank-64) trains on 15K samples for 5 epochs; each specialist (rank-32) trains on 5K domain-specific samples for 3 epochs. The base model is Qwen3-VL-8B-Instruct Bai et al. (2025).

## 4.2 Critique Quality and Domain Specificity

We first evaluate whether modular adaptation improves critique alignment over monolithic training. Table 1 compares SE-MoLoRA against the zero-shot base model and a monolithic LoRA baseline trained on the same combined data. The shared expert achieves a BERTScore F1 of 0.4215, roughly ten times the base model (0.0379) and nearly double the monolithic baseline (0.2317). All three specialists likewise outperform both baselines across every metric, with composition (0.2999) and lighting (0.2990) showing the largest gains over the monolithic adapter. The base model produces negative BERTScore F1 on all specialist domains, indicating that its verbose outputs are less aligned with reference critiques than even the rescaled baseline. All improvements are statistically significant $( p < 0 . 0 0 1$ , paired t-test).

To complement reference-based evaluation, we conduct blind pairwise comparisons using GPT-4o as an LLM-as-judge (Figure 2). Against the base model, all four experts achieve decisive win rates. Against the monolithic baseline, the shared expert wins 84.6% of comparisons, demonstrating that monolithic fine-tuning over-specializes and cannot suppress domain-specific knowledge when tasked with general critique. Specialist experts achieve win rates of 50.0–54.8% against a rank-96 monolithic adapter despite having half the rank, suggesting that the modular decomposition itself, not additional capacity, drives the quality improvement.

While these aggregate comparisons confirm quality gains, they do not reveal whether specialists produce qualitatively distinct outputs. To test this, we perform a switch test: identical images are presented to the system with each expert activated in turn. As shown in Figure 3, each expert critiques with visibly distinct domain-specific vocabulary, confirming functional specialization.

## 4.3 Expert Disentanglement

Improved critique metrics do not guarantee that specialists encode distinct knowledge. To verify structural disentanglement, we analyze SVD-based cosine similarity of learned weight updates $\Delta W = B A$ . Table 2 presents entanglement values across four configurations of the orthogonal penalty $\lambda _ { \mathrm { o r t h o } }$

<table><tr><td>Expert</td><td>Architecture</td><td>BERTScore F1 (↑)</td><td>SBERT (Cos Sim) (↑)</td><td>ROUGE-L F1 (↑)</td></tr><tr><td rowspan="3">Shared</td><td>Base Model</td><td> $0 . 0 3 7 9 \pm 0 . 0 7 5 1$ </td><td> $0 . 6 1 0 5 \pm 0 . 1 1 4 0$ </td><td> $0 . 0 9 2 1 \pm 0 . 0 3 0 9$ </td></tr><tr><td>Monolithic</td><td> $0 . 2 3 1 7 \pm 0 . 0 8 3 1$ </td><td> $0 . 6 3 3 7 \pm 0 . 1 0 1 3$ </td><td> $0 . 1 7 8 7 \pm 0 . 0 6 2 9$ </td></tr><tr><td>SE-MoLoRA</td><td> $\mathbf { 0 . 4 2 1 5 \ : \pm { 0 . 1 0 0 0 } }$ </td><td> $\mathbf { 0 . 6 8 5 1 \pm 0 . 1 0 6 1 }$ </td><td> $\mathbf { 0 . 3 3 3 9 \ : \pm { \ : 0 . 1 0 1 8 } }$ </td></tr><tr><td rowspan="3">Composition</td><td>Base Model</td><td> $- 0 . 0 2 1 1 \pm 0 . 0 4 7 8$ </td><td> $0 . 5 2 7 7 \pm 0 . 1 2 3 2$ </td><td> $0 . 0 8 1 0 \pm 0 . 0 2 7 0$ </td></tr><tr><td>Monolithic</td><td> $0 . 2 3 4 8 \pm 0 . 0 7 7 3$ </td><td> $0 . 5 3 6 2 \pm 0 . 1 2 6 5$ </td><td> $0 . 1 8 4 5 \pm 0 . 0 5 3 0$ </td></tr><tr><td>SE-MoLoRA</td><td> $\mathbf { 0 . 2 9 9 9 \ : \pm { \ : 0 . 0 7 4 5 } }$ </td><td> $\mathbf { 0 . 5 8 5 1 } \pm \mathbf { 0 . 1 3 3 6 }$ </td><td> $\mathbf { 0 . 2 1 4 5 \ : \pm { \ : 0 . 0 5 7 3 } }$ </td></tr><tr><td rowspan="3">Lighting</td><td>Base Model</td><td> $- 0 . 0 7 4 1 \pm 0 . 0 4 5 6$ </td><td> $0 . 5 1 5 5 \pm 0 . 1 2 3 0$ </td><td> $0 . 0 4 1 4 \pm 0 . 0 1 6 8$ </td></tr><tr><td>Monolithic</td><td> $0 . 1 7 4 1 \pm 0 . 0 9 3 8$ </td><td> $0 . 5 2 5 3 \pm 0 . 1 1 4 5$ </td><td> $0 . 1 4 2 7 \pm 0 . 0 5 2 0$ </td></tr><tr><td>SE-MoLoRA</td><td> $\mathbf { 0 . 2 9 9 0 \ : \pm { \ : 0 . 1 0 6 0 } }$ </td><td> $\mathbf { 0 . 5 4 1 0 \ : \pm { \ : 0 . 1 4 2 8 } }$ </td><td> $\mathbf { 0 . 2 0 1 4 \ : \pm { \ : 0 . 0 9 0 5 } }$ </td></tr><tr><td rowspan="3">Technical</td><td>Base Model</td><td> $- 0 . 0 4 8 8 \pm 0 . 0 4 9 6$ </td><td> $0 . 4 4 6 7 \pm 0 . 1 2 4 8$ </td><td> $0 . 0 6 7 7 \pm 0 . 0 2 3 8$ </td></tr><tr><td>Monolithic</td><td> $0 . 2 1 3 2 \pm 0 . 0 8 2 3$ </td><td> $0 . 3 9 1 9 \pm 0 . 1 3 6 7$ </td><td> $0 . 1 3 5 5 \pm 0 . 0 5 6 6$ </td></tr><tr><td>SE-MoLoRA</td><td> $\mathbf { 0 . 2 4 6 1 \ : \pm { \ : 0 . 0 7 9 6 } }$ </td><td> $\mathbf { 0 . 4 9 9 2 \ : \pm { \ : 0 . 1 3 4 5 } }$ </td><td> $\mathbf { 0 . 1 7 2 7 \ : \pm { \ : 0 . 0 5 7 1 } }$ </td></tr></table>

Table 1: Automated metric evaluation comparing the zero-shot Base model, a Monolithic LoRA baseline, and our routed SE-MoLoRA architecture. Best results are highlighted in bold. A paired Student’s t-test confirmed statistical significance $( p < 0 . 0 0 1 )$ .

![](images/e6c9b5e7c613a0194635cc6d4ca25df5f353baeae50b5369a554c5aa346e7b9c.jpg)  
Figure 2: LLM-as-a-Judge Evaluation Win Rates of the SE-MoLoRA architecture against the zero-shot base model (Top) and the monolithic adapter (Bottom).

Without orthogonal regularization, specialist pairs exhibit high entanglement (≈ 0.52). Applying $\lambda _ { \mathrm { o r t h o } } = 0 . 1$ reduces this to ≈ 0.42, but increasing the penalty beyond this yields diminishing returns, establishing an entanglement floor. This floor reflects a fundamental interdependence of photographic critique domains: low-level visual features such as edge detection are shared across composition, focus, and lighting analysis.

The shared expert acts as an entanglement sink, absorbing common features as evidenced by consistently lower shared-specialist similarity (≈ 0.34) compared to specialist-specialist pairs (≈ 0.42). Removing the

![](images/59f763a7ea636f285eb0f57d22d04913e3ef122b854696871bef4ee0b14c86de.jpg)

Technical Expert: The image exhibits noticeable grain, likely from high ISO settings, which detracts from overall image quality. The sky shows signs of over-processing with excessive blue enhancement, creating an unnatural appearance. The water lacks the expected texture and detail, appearing overly smooth and flat. The image would benefit from noise reduction techniques and more natural sky and water rendering.

Shared Expert: The photograph features a building with architectural details including a dome and spires, captured in what appears to be a European setting with a sky backdrop.

Lighting Expert: The color palette is well-executed, with the warm orange tones of the building lights creating an appealing contrast against the cool blue sky and water. The lighting effectively highlights the architectural details and creates an inviting atmosphere.

Composition Expert: The composition effectively uses the river as a leading line that guides the viewer's eye toward the building, creating a strong visual flow. The framing successfully captures the building's full profile, though the inclusion of the river in the foreground may be considered excessive by some viewers. The image appears to have been taken from a slightly elevated position, which could be improved by shooting from a lower vantage point to create a more dramatic perspective. The overall composition is well-executed and visually appealing.

Figure 3: Qualitative comparison between responses from each expert for the same image (Sample Taken from RPCD Nieto et al. (2022)). Domain-specific vocabulary for each expert is highlighted in purple.
<table><tr><td rowspan="3">Expert Pair</td><td rowspan="3">No Shared Expert during training</td><td colspan="4">Shared Expert present during training</td></tr><tr><td>λ = 0.0</td><td>λ = 0.1</td><td>λ = 0.5</td><td>λ = 1.0</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Comp. vs Light.</td><td>0.6468</td><td>0.5395</td><td>0.4365</td><td>0.4358</td><td>0.4392</td></tr><tr><td>Comp. vs Tech.</td><td>0.6241 0.6380</td><td>0.5169 0.5251</td><td>0.4229 0.4238</td><td>0.4193 0.4230</td><td>0.4233</td></tr><tr><td>Light. vs Tech.</td><td></td><td></td><td></td><td></td><td>0.4229</td></tr><tr><td>Shared vs Comp.</td><td>0.4335</td><td>0.3512</td><td>0.3477</td><td>0.3477</td><td>0.3477</td></tr><tr><td>Shared vs Light.</td><td>0.4314</td><td>0.3426</td><td>0.3419</td><td>0.3420</td><td>0.3408</td></tr><tr><td>Shared vs Tech.</td><td>0.4223</td><td>0.3461</td><td>0.3418</td><td>0.3392</td><td>0.3422</td></tr></table>

Table 2: SVD-based cosine similarity between expert adapters. Lower values indicate greater disentanglement.

shared expert entirely (first column) increases all pairwise similarities substantially, confirming its role in promoting disentanglement.

## 4.4 Routing Accuracy

Figure 5 presents confusion matrices for both routers. On the held-out test set, the text-only router achieves higher classification accuracy than the multimodal router (93.42% vs. 86.95%), as routing labels are heavily correlated with textual vocabulary.

However, on downstream aesthetic benchmarks with naturalistic queries (Table 3), the multimodal router consistently outperforms the text-only router. On Q-Bench, the multimodal router achieves 57.65% vs. 55.51%; on PhotoBench, 53.97% vs. 50.66%. This reversal occurs because benchmark prompts are generally vague, and the text-only router lacks visual grounding to detect latent image flaws, defaulting to the shared expert. The multimodal router detects visual outliers from the image, routing generic queries to suitable specialists at the cost of higher inference latency (3.60 ± 0.45 vs. 2.17 ± 0.18 ms/sample).

![](images/99c15146e68d61df5af5306cec842f52dbc4b3bb01f70744273529feee004306.jpg)

Figure 4: SVD-based expert disentanglement across diferent $\lambda _ { o r t h o }$ constraints.  
![](images/71bb92eb08a516bd6f6fd15aedb3e21c487a1f68f445c862711520298f80e99b.jpg)

![](images/1da7812409056f41b07663963a8b89db30cbc90975fe0f4b9a100bc5f64b7cf5.jpg)

Figure 5: Confusion matrix for the text-only and multimodal router.
<table><tr><td>Benchmark</td><td>Text Router</td><td>Multimodal Router</td></tr><tr><td>Q-Bench</td><td>55.51%</td><td>57.65%</td></tr><tr><td>PhotoBench</td><td>50.66%</td><td>53.97%</td></tr></table>

Table 3: Router comparison on zero-shot aesthetic benchmarks.

## 4.5 Ablation: Impact of the Shared Expert

To isolate the shared expert’s contribution, we compare the full SE-MoLoRA pipeline against a variant where specialist adapters are trained directly on the base model without the shared expert. Using GPT-4o as LLM-as-judge (Figure 6), specialists trained with the shared expert produce consistently higher win rates across all domains, generating critiques with proper visual grounding that bridge general visual context with domain-specific analysis. The disentanglement data in Table 2 complements this finding: without the shared expert, specialist pairs are substantially more entangled (≈ 0.63 vs. ≈ 0.42), confirming that the shared expert absorbs cross-domain features and prevents specialists from wasting capacity on redundant representations. The shared expert particularly benefits the technical domain, which otherwise shows the lowest gains across all metrics.

![](images/42c7e85a327461233c7f90e35541effe58335b7a8f7edd3b3d76c0911316e4f2.jpg)  
Figure 6: Ablation results showing the impact of the shared expert on win rates with LLM-as-judge.

## 4.6 Comparison with State-of-the-Art

We next examine how SE-MoLoRA performs on established aesthetic benchmarks. Table 4 compares against published results on Q-Bench and PhotoBench. SE-MoLoRA outperforms the zero-shot base model but underperforms state-of-the-art methods such as PhotoEye Qi et al. (2025). This gap reflects a deliberate design trade-of. First, both benchmarks are zero-shot multiple-choice QA tasks, and SE-MoLoRA’s prompt routed architecture does not reliably trigger appropriate specialists with generic benchmark queries. Second, methods like PhotoEye employ continuous multi-view vision fusion from diverse foundation models (CLIP, DINOv2, SAM, CoDETR) with full-parameter fine-tuning on 450K images, whereas SE-MoLoRA freezes the base encoder and adapts only through lightweight LoRA modules. SE-MoLoRA targets a diferent operating point: controllable, domain-specific critique generation under parameter-eficient constraints, rather than benchmark-optimal zero-shot QA.

<table><tr><td>Model</td><td>Accuracy (%)</td><td>Model</td><td>Accuracy (%)</td></tr><tr><td>Random guess*</td><td>37.80</td><td>GPT-4o (2024-08-06)*</td><td>64.12</td></tr><tr><td>GPT-4o (2024-08-06)*</td><td>73.04</td><td>UNIAA* Zhou et al. (2024)</td><td>36.87</td></tr><tr><td>AesExpert* Huang et al. (2024b)</td><td>64.15</td><td>AesExpert* Huang et al. (2024b)</td><td>60.01</td></tr><tr><td>Q-Instruct* Wu et al. (2024)</td><td>67.09</td><td>Q-Instruct* Wu et al. (2024)</td><td>43.86</td></tr><tr><td>PhotoEye* Qi et al. (2025)</td><td>74.50</td><td>PhotoEye* Qi et al. (2025)</td><td>73.92</td></tr><tr><td>Qwen3-VL-8B</td><td>50.79</td><td>Qwen3-VL-8B</td><td>31.72</td></tr><tr><td>SE-MoLoRA</td><td>57.65</td><td>SE-MoLoRA</td><td>53.97</td></tr></table>

Table 4: Comparison of SE-MoLoRA with State-of-the-Art image aesthetic methods on Q-Bench (left) and PhotoBench (right). Results indicated by <sup>∗</sup> were reported in Qi et al. (2025).

Qualitative analysis on adversarial images (Figure 7) reveals that SE-MoLoRA experts are less prone to overrating aesthetically pleasing but technically flawed images compared to the base model, indicating reduced halo bias. However, specialists also tend to over-criticize technically sound images, revealing a trade-of between reduced semantic-aesthetic bias and increased critical behavior.

## 5 Conclusion

Our experiments show that decomposing photographic critique into modular LoRA adapters yields measurably better domain-specific vocabulary than monolithic fine-tuning, even when the monolithic baseline has double the adapter capacity. The shared expert proves essential: it absorbs cross-domain features that would otherwise entangle specialists, reducing pairwise cosine similarity from ≈ 0.63 to ≈ 0.42 and improving

![](images/aa0ce562e7886d015444131950855ef549a56edcb28a6ee002f3fdb42020783f.jpg)

Base Model: The image employs a natural, organic framing through foreground foliage, particularly the blurred green leaves and stems on the left and right which partially obscure the view and create a sense of peeking into a hidden space

Composition Expert: The framing could be improved by positioning the subject more centrally in the frame, as the current composition places the mouse off-center with excessive negative space on the left side.

![](images/006b98c59e9bee73d2e30eac2ed1bd0b77e9e33b7b1bf261f3c5fa12120e27a8.jpg)

Base Model: The image exhibits shallow depth of field, with only a narrow plane in focus. The central flower bud is the sharpest point, while surrounding buds and background elements are progressively blurred.

Technical Expert: The depth of field is too shallow, with only a small portion of the flower in sharp focus while the rest of the subject falls into blur. A smaller aperture would increase the depth of field to bring more of the flower into acceptable focus, though this may require adjusting shutter speed or ISO to maintain proper exposure.

![](images/ccfe74ce279de94969bf3919d57a7fef77150417e5cff54e4c6ff684ee4a6852.jpg)

Base Model: The river acts as a natural horizontal band that anchors the composition, while the distant mountain ridgeline reinforces this horizontal structure. The sky occupies the upper third, providing a balanced weight above the land.

Composition Expert: The current composition lacks a clear focal point, with the foreground elements competing with the scenic background for visual attention.

![](images/ff4c7c778eccfcfd64235de0075f16d026c44a64192555611b7f8f999755142e.jpg)

Base Model: The image exhibits soft focus across the entire frame. There is no discernible plane of sharpness, suggesting either a shallow depth of field with the focus point misaligned, or a general lack of optical acuity.

Technical Expert: The image exhibits noticeable grain, which may be intentional for aesthetic effect but could be reduced through noise reduction techniques. The overall sharpness appears adequate, though the grain structure is prominent enough to be distracting.

Figure 7: Qualitative comparison of SE-MoLoRA and base model on adversarial images (Sample Taken from RPCD Nieto et al. (2022)).

visual grounding across all domains. For real-world deployment, the multimodal router provides more robust specialist selection on vague queries, though at higher latency than the text-only alternative.

Limitations and future work. The current system activates only one specialist per query to keep inference cost predictable and to make domain attribution interpretable. Multi-specialist activation is left for future work. Another avenue for future work is replacing the hard router with soft classification over multiple experts.

All evaluation rely on automated metrics and LLM-based judgments rather than human user studies, and conducting evaluations with professional photographers would validate whether disentangled critiques are perceived as more actionable. The architecture is evaluated on a single base model (Qwen3-VL-8B) and three aesthetic domains; scaling to additional domains and base models remains unexplored. Finally, the SE-MoLoRA framework is not limited to photographic critique and could be applied to other domains where specialist knowledge exhibits varying degrees of interdependence.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Kuang-Yu Chang, Kung-Hung Lu, and Chu-Song Chen. Aesthetic critiques generation for photos. In 2017 IEEE International Conference on Computer Vision (ICCV), pp. 3534–3543, 2017.

Damai Dai, Chengqi Deng, Chenggang Zhao, R.x. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.k. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1280–1297, Bangkok, Thailand, August 2024. Association for Computational Linguistics.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: eficient finetuning of quantized llms. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: scaling to trillion parameter models with simple and eficient sparsity. J. Mach. Learn. Res., 23(1), January 2022. ISSN 1532-4435.

Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, and Hao Wang. Mixture-of-LoRAs: An eficient multitask tuning method for large language models. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.), Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 11371–11380, Torino, Italia, May 2024. ELRA and ICCL.

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-eficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.

Sreyan Ghosh, Chandra Kiran Reddy Evuru, Sonal Kumar, Ramaneswaran S, Deepali Aneja, Zeyu Jin, Ramani Duraiswami, and Dinesh Manocha. A closer look at the limitations of instruction tuning. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 15559–15589. PMLR, 21–27 Jul 2024.

Aditya Gulati, Moreno D’Incà, Nicu Sebe, Bruno Lepri, and Nuria Oliver. Beauty and the bias: Exploring the impact of attractiveness on multimodal large language models. Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, 8(2):1154–1168, Oct. 2025.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: eficient low rank adaptation of large models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016.

Simon Hentschel, Konstantin Kobs, and Andreas Hotho. Clip knows image aesthetics. Frontiers in Artificial Intelligence, Volume 5 - 2022, 2022. ISSN 2624-8212.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-eficient transfer learning for NLP. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2790–2799. PMLR, 09–15 Jun 2019.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Eficient cross-task generalization via dynamic loRA composition. In First Conference on Language Modeling, 2024a.

Yipo Huang, Xiangfei Sheng, Zhichao Yang, Quan Yuan, Zhichao Duan, Pengfei Chen, Leida Li, Weisi Lin, and Guangming Shi. Aesexpert: Towards multi-modality foundation model for image aesthetics perception. In Proceedings of the 32nd ACM International Conference on Multimedia, MM ’24, pp. 5911–5920, New York, NY, USA, 2024b. Association for Computing Machinery. ISBN 9798400706868.

Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geofrey E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991.

Ruixiang Jiang and Chang Wen Chen. Multimodal llms can reason about aesthetics in zero-shot. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, pp. 6634–6643, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400720352.

Xin Jin, Le Wu, Geng Zhao, Xiaodong Li, Xiaokun Zhang, Shiming Ge, Dongqing Zou, Bin Zhou, and Xinghui Zhou. Aesthetic attributes assessment of images. In Proceedings of the 27th ACM International Conference on Multimedia, MM ’19, pp. 311–319, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450368896.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. Found. Trends. Comput. Graph. Vis., 16(1-2):1–214, May 2024a. ISSN 1572-2740.

Junzhuo Li, Bo Wang, Xiuze Zhou, and Xuming Hu. Dynamic expert specialization: Towards catastrophic forgetting-free multi-domain moe adaptation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 18489–18504, 2025.

Tang Li, Mengmeng Ma, and Xi Peng. DEAL: Disentangle and Localize Concept-level Explanations for VLMs. In Proceedings of the European Conference on Computer Vision (ECCV), 2024b.

Boyang Liu, Yifan Hu, Senjie Jin, Shihan Dou, Gonglei Shi, Jie Shao, Tao Gui, and Xuanjing Huang. Unlocking the essence of beauty: Advanced aesthetic reasoning with relative-absolute policy optimization. arXiv preprint arXiv:2509.21871, 2025.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: weight-decomposed low-rank adaptation. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Naila Murray, Luca Marchesotti, and Florent Perronnin. Ava: A large-scale database for aesthetic visual analysis. In 2012 IEEE conference on computer vision and pattern recognition, pp. 2408–2415. IEEE, 2012.

Daniel Vera Nieto, Luigi Celona, and Clara Fernandez-Labrador. Understanding aesthetics with language: a photo critique dataset for aesthetic assessment. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Jonas Pfeifer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. AdapterFusion: Non-destructive task composition for transfer learning. In Paola Merlo, Jorg Tiedemann, and Reut Tsarfaty (eds.), Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 487–503, Online, April 2021. Association for Computational Linguistics.

Daiqing Qi, Handong Zhao, Jing Shi, Simon Jenni, Yifei Fan, Franck Dernoncourt, Scott Cohen, and Sheng Li. The photographer’s eye: Teaching multimodal large language models to see and critique like photographers. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24807–24816, 2025.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017.

Hossein Talebi and Peyman Milanfar. Nima: Neural image assessment. IEEE transactions on image processing, 27(8):3998–4011, 2018.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Guolong Wang, Heng Huang, Zhiqiang Zhang, Wentian Li, Feilong Ma, and Xin Jin. Aestest: Measuring aesthetic intelligence from perception to production. arXiv preprint arXiv:2511.06360, 2025.

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, Geng Xue, Wenxiu Sun, Qiong Yan, and Weisi Lin. Q-instruct: Improving low-level visual abilities for multi-modality foundation models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 25490–25500, 2024.

Wenhan Xia, Chengwei Qin, and Elad Hazan. Chain of lora: Eficient fine-tuning of language models via residual learning. arXiv preprint arXiv:2401.04151, 2024.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Rafel, and Mohit Bansal. Ties-merging: resolving interference when merging models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, and Miki Haseyama. Adaptive shared experts with lora-based mixture of experts for multi-task learning. arXiv preprint arXiv:2510.00570, 2025.

She Yifei, Xinhao Wei, and Yulong Wang. DisLoRA: Task-specific low-rank adaptation via orthogonal basis from singular value decomposition. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13740–13755, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: absorbing abilities from homologous models as a free lunch. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In International Conference on Learning Representations, 2023.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-eficient fine-tuning. In The Eleventh International Conference on Learning Representations, 2023.

Zhaokun Zhou, Qiulin Wang, Bin Lin, Yiwei Su, Rui Chen, Xin Tao, Amin Zheng, Li Yuan, Pengfei Wan, and Di Zhang. Uniaa: A unified multi-modal image aesthetic assessment baseline and benchmark. arXiv preprint arXiv:2404.09619, 2024.