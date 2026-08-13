# SCOPE-Router: Cost-Aware Open-Set VLM Routing for Execution-Oriented Tasks

Tao Yu<sup>1,</sup> <sup>2∗</sup>, Yifei Qu<sup>1</sup> <sup>∗</sup> <sup>†</sup>, Zhiqing Cui<sup>1</sup> <sup>∗</sup> <sup>†</sup>, Pengfei Zhou<sup>3‡</sup>, Zhongtian Luo<sup>1,</sup> <sup>†</sup>, Yujia Yang<sup>2</sup>, Shenghua Chai<sup>1,</sup> <sup>†</sup>, Haopeng Jin<sup>4</sup>, Zhenghao Zhang<sup>2</sup>, Xinming Wang<sup>1,</sup> <sup>2</sup>, Hongzhu Yi<sup>2‡</sup>, Wangbo Zhao<sup>3</sup>, Zhenglin Wan<sup>3</sup>, Yan Huang<sup>1,</sup> <sup>2‡</sup>, Yeshani<sup>4</sup>, Jinwen Luo<sup>4</sup>, Yang You<sup>3</sup>

<sup>1</sup>CASIA <sup>2</sup>UCAS <sup>3</sup>NUS <sup>4</sup>Tencent

Model routing aims to select the most suitable model from a candidate pool for each query, balancing quality and cost. Existing VLM routing research is limited to traditional VQA evaluation, lacks systematic calibration optimization for open-set scenarios, and employs training objectives that dilute multi-positive signals via softmax normalization without incorporating cost. We address these limitations with three contributions: (1) VLM-ExecRouterBench, the first execution-oriented VLM routing benchmark covering Code, Agentic, and Search domains with 11 candidate models spanning nearly two orders of magnitude in pricing; (2) SCOPE-Router, a dual-tower router that matches queries to model behavior profiles constructed via hybrid calibration (random/diagnostic/diversity sampling), enabling new models to join routing without retraining; (3) CRM+RCCR, an architectureagnostic cost-aware objective that encodes cost preference into continuous relevance targets through per-pair independent scoring, eliminating multi-positive dilution while regularizing queries with similar routing preferences to be closer in the routing space. Empirically, SCOPE-Router achieves the best Rank Score on all three benchmarks, surpassing the runner-up by 1.84 points under OOD settings and by 6.75 points under doubly OOD open-set evaluation. When applied to four diverse routers, CRM+RCCR improves Rank Score by 1.25–6.21 points.

Date: Aug. 12, 2026   
Code: https://github.com/yutao1024/SCOPE-Router Data: https://huggingface.co/datasets/Kirito-Lab/VLM-ExecRouterBench

## 1 Introduction

The rapid advancement of LLMs and VLMs has given rise to a heterogeneous model ecosystem. From open-source models with fewer than 8B parameters to flagship commercial APIs, models differ substantially in capability, cost, and latency. No single model achieves optimal accuracy at minimal cost across all tasks [1, 2, 3], motivating model routing: selecting the most suitable model per query to balance quality and cost [4, 5, 6]. In practice, VLMs are increasingly deployed not through raw API calls for simple question answering, but within execution pipelines, including code generation, tool-calling agents, and multi-step web retrieval, where routing decisions directly impact both task success and deployment cost.

Recent years have seen progress in LLM/VLM routing. Methodologically, existing work spans classifierbased predictive routing, contrastive learning matching [7], cluster-based generalization [8], and reinforcement learning orchestration [9]. On the evaluation side, VL-RouterBench first extends routing evaluation to vision-language models, covering 14 datasets and 17 candidate VLMs; MMR-Bench [10] further introduces budget-aware multimodal reasoning evaluation. However, existing work faces limitations along three dimensions:

![](images/9d498552f75a5497dc56e7b2992b0b23ca2ac07d32745ee509a9a93173d1f8f7.jpg)  
Figure 1. Overview of the proposed SCOPE-Router framework. Left: router data preparation constructs unified execution samples, correctness and cost matrices, and query-aware model profiles using the hybrid calibration set. Middle: during router training, the query and model-profile branches are projected into a shared routing space and jointly optimized using Cost-aware Relevance Matching (CRM) and Routing-Consistency Contrastive Regularization (RCCR). Right: during inference, the router selects the most suitable candidate according to query–profile similarity; a new model can be incorporated by running only the calibration set to construct and cache its profile, without retraining the router.

Limited domain coverage. Current VLM routing benchmarks cover only traditional VQA tasks—visual reasoning, OCR, chart understanding—where models directly output text answers. Yet VLMs are increasingly applied to tasks that require active execution: code generation demands precise program logic verified by unit tests, tool-calling tasks require multi-step reasoning with external tool interaction, and web search involves planning retrieval strategies and synthesizing multi-source information. These execution tasks impose different capability requirements and exhibit stronger inter-model complementarity, yet no routing benchmark incorporates them.

Insufficient open-set capability. New models emerge continuously in practice, but traditional routing methods treat models as fixed class labels, requiring retraining upon each addition. Existing open-set approaches (UniRoute, ICL-Router [11]) lack systematic optimization of calibration data selection: UniRoute randomly partitions validation data for model representation, while ICL-Router filters samples by a single signal. Under limited calibration budgets, the absence of targeted selection strategies leads to low discriminability in model profiles.

Training objectives decoupled from cost. RouterDC’s contrastive objective partitions positive/negative examples based solely on correctness, ignoring cost differences; VL-RouterBench’s softmax CE loss dilutes multi-positive signals through normalization; UniRoute only incorporates cost as a post-hoc inference penalty, leaving the training process itself cost-unaware.

To address these limitations, we make the following contributions (see Figure 1 for an overview):

(1) VLM-ExecRouterBench: the first execution-oriented VLM routing benchmark. We extend evaluation from traditional VQA to three execution domains: code generation (Code), tool-calling agents (Agentic), and multi-step web search (Search). Each sample is standardized into a Routing Input / Execution Context /

Verification Rule triplet, uniformly covering three distinct execution protocols. The candidate pool comprises 11 models from five major providers, with output pricing spanning nearly two orders of magnitude.

(2) SCOPE-Router: an open-set dual-tower router with hybrid calibration. We formulate open-set routing as query-to-profile matching: for each model, we construct a Query-Aware Profile that fuses behavioral statistics and semantic directions, and compute matching scores via lightweight dual-tower MLPs in a shared routing space. New models need only run once on a small calibration set to generate a profile and immediately join routing, without retraining. To maximize profile discriminability under limited budgets, we design a hybrid calibration strategy allocating budgets at 50%/30%/20% to random sampling (distributional coverage), diagnostic sampling (driven by model disagreement and cost spread), and diversity sampling (uniform semantic coverage). The three strategies are complementary, with the full combination outperforming any single strategy.

(3) CRM+RCCR cost-aware training objective. We propose Cost-aware Relevance Matching (CRM), which defines a continuous relevance target that encodes cost preference directly into supervision and adopts per-pair independent sigmoid binary cross-entropy (BCE) in place of softmax cross-entropy (CE) to avoid multi-positive signal dilution. This is complemented by Routing-Consistency Contrastive Regularization (RCCR), which pulls queries with similar routing patterns closer in the routing space, enabling the router to generalize across queries that share model suitability. The objective is architecture-agnostic and can directly replace existing router losses. Extensive experiments on RouterDC, ZOOTER [12], CosineCls, and VLC all show consistent Rank Score improvements without any structural modification.

We conduct systematic experiments on VLM-ExecRouterBench, VL-RouterBench, and MMR-Bench. SCOPE-Router achieves the best Rank Score on all three benchmarks and maintains its lead under OOD generalization settings and open-set evaluation. Ablation studies verify the complementarity of CRM and RCCR, the effectiveness of the hybrid calibration strategy, and the cross-architecture transferability of the proposed training objective.

## 2 Related Work

LLM Routing Benchmarks. RouterBench formulated LLM routing as cost–quality optimization. LLM-RouterBench [13] and RouterArena further expanded model pools and evaluation dimensions for text routing. VL-RouterBench extended the paradigm to VLMs with 14 datasets, 17 models, and a unified Rank Score metric. However, existing VLM benchmarks remain confined to traditional VQA tasks. Execution-based tasks such as code generation, tool-use agentic reasoning, and web retrieval exhibit stronger inter-model complementarity yet have not been incorporated into any routing benchmark. Our VLM-ExecRouterBench is the first to cover these domains.

Model Representation and Open-Set Routing. Early routers map queries to fixed model IDs, requiring retraining for new models. Recent open-set approaches use per-cluster error rates, anchor pass/fail patterns, or public-signal GNNs [14], but each sacrifices sample-level granularity, cost information, or behavioral fidelity. Our profiles are derived from first-hand inference on a carefully designed hybrid calibration set, fusing per-sample correctness, cost, and semantic directions to capture both what a model can solve and at what price.

Router Training Objectives. RouterDC uses dual contrastive losses but partitions positives purely by correctness. VL-RouterBench pairs cost-weighted soft labels with softmax CE, but normalization dilutes gradients when multiple models answer correctly. UniRoute treats cost only at inference time. We propose CRM+RCCR: Cost-aware Relevance Matching encodes cost preference into continuous relevance targets and eliminates multi-positive dilution, while Routing-Consistency Contrastive Regularization pulls queries with similar routing patterns closer in the routing space, enabling the router to benefit from shared modelsuitability structure.

## 3 Method

## 3.1 Problem Formulation

Model routing comprises three stages: data construction, router training, and routing inference.

Data Construction. Given task queries $\{ x _ { i } \} _ { i = 1 } ^ { N }$ (text and/or images) and a candidate VLM pool $\mathcal { M } =$ $\{ m _ { 1 } , \hdots , m _ { K } \}$ , we execute every model on every sample, yielding a correctness matrix $Y \in \overline { { \{ 0 , 1 \} ^ { N \times K } } }$ $( Y _ { i , m } = 1$ if model m answers sample i correctly) and a cost matrix $C \in \mathbb { R } _ { + } ^ { N \times K } \left( C _ { i , m } \right.$ is the inference cost computed from token usage and per-million-token pricing).

Router Training. Given Y and C, the router learns a scoring function $s _ { \theta } ( x , m ) \colon \mathcal { X } \times \mathcal { M } \to \mathbb { R }$ . High-scoring models should be both likely correct and relatively cheap. The training objective must encode quality and cost preferences.

Routing Inference. Given a new query $x _ { i } ,$ the router selects $\hat { m } _ { i } = \arg \operatorname* { m a x } _ { m _ { k } \in \mathcal { M } } s _ { \theta } ( x _ { i } , m _ { k } )$ . Evaluation uses the Rank Score:

$$
S _ { \beta } = \frac { ( 1 + \beta ) \cdot A \cdot \widetilde { C } } { \beta \cdot A + \widetilde { C } } ,\tag{1}
$$

where A is routing accuracy, $\widetilde { C } = \frac { \log _ { 2 } c _ { \mathrm { m a x } } - \log _ { 2 } \mathrm { c o s t } } { \log _ { 2 } c _ { \mathrm { m a x } } - \log _ { 2 } c _ { \mathrm { m i n } } }$ is the log-normalized cost score, and $\beta$ weights accuracy vs. cost (details in the Evaluation Metrics paragraph).

## 3.2 Data Generation

Existing VLM routing benchmarks draw exclusively from traditional VQA-style datasets. We extend evaluation to execution-oriented domains (Code, Agentic, Search) by generating standardized samples from existing domain-specific sources (see the Datasets section for detailed dataset descriptions and Appendix A for data-generation procedures and implementation details).

Each sample is converted into a three-part structure: (1) Routing Input: the query visible to the router; (2) Execution Context: the full context given to each candidate VLM; (3) Verification Rule: the validation logic that judges model output. This design separates what the router sees from what the model executes.

Matrix Construction. For each generated sample $x _ { i } ,$ all models execute the Execution Context and the Verification Rule judges the results, producing the correctness matrix Y and cost matrix C.

Data Splits. The full dataset is partitioned 70%/10%/20%: training, validation, and held-out test.

## 3.3 Open-Set Routing and Hybrid Calibration Sets

Open-set routing enables new models to integrate without retraining. Instead of treating models as fixed class labels, we build a behavioral profile vector for each model. The router learns to match queries to profiles— a new model need only be executed once on a small calibration set to generate its profile and immediately participate in routing.

Hybrid Calibration Set Selection. The calibration set $S _ { \mathrm { c a l i b } } \subset S _ { \mathrm { t r a i n } } ( | S _ { \mathrm { c a l i b } } | = 1 0 2 4 )$ is selected via a hybrid strategy allocating the budget at a 50%/30%/20% ratio across three complementary mechanisms, executed sequentially.

Random Sampling (50%). Uniform sampling without replacement, ensuring unbiased coverage of the training distribution.

Diagnostic Sampling (30%). From remaining samples, this component maximizes discriminative power. For each candidate sample i:

$$
d _ { i } = w _ { \mathrm { d i s } } \cdot \underbrace { 4 p _ { i } ( 1 - p _ { i } ) } _ { \mathrm { d i s a g r e e m e n t } } + w _ { \mathrm { c o s t } } \cdot \underbrace { \mathrm { c o s t } _ { - } s \mathrm { p r e a d } _ { i } ^ { \mathrm { n o r m } } } _ { \mathrm { c o s t s p r e a d } } ,\tag{2}
$$

where $\begin{array} { r } { p _ { i } = \frac { 1 } { K } \sum _ { m = 1 } ^ { K } Y _ { i , m } } \end{array}$ is the fraction of models in the reference pool $\mathcal { M }$ answering correctly $( w _ { \mathrm { d i s } } = 0 . 7 $ $w _ { \mathrm { c o s t } } = 0 . 3 )$ . The disagreement term peaks at $p _ { i } = 0 . 5$ (maximum inter-model disagreement) and equals zero when all models agree. The cost spread is defined as the difference between the maximum and minimum inference cost among models that answer correctly, normalized by the 95th percentile across all samples for robustness. Samples are stratified into difficulty buckets (hard $p _ { i } \leq 0 . 2 5 \bar { ~ / ~ }$ medium (0.25, 0.75] / easy $> 0 . 7 5$ , allocated at $2 0 \% / 6 0 \% / 2 0 \% )$ and drawn with weights proportional to $\exp ( \hat { d } _ { i } / { \tau _ { \mathrm { s a m p } } } )$ within each bucket, where $\hat { d } _ { i }$ is the min-max normalized score within each bucke $( \tau _ { \mathrm { s a m p } } = 0 . 6 )$

Diversity Sampling (20%). From the remaining unselected samples, we apply MiniBatchKMeans clustering on their pre-computed query embeddings (text and vision encoder features fused via normalize-concat), with the number of clusters set equal to the target sample count. The sample closest to each cluster centroid is selected, ensuring uniform coverage of the embedding space. Empty clusters (rare) are filled by selecting the farthest points from the global centroid.

Model Behavior Profile Construction. Given the calibration set (S = 1024 samples), we construct a Query-Aware Profile $\boldsymbol { \mathsf { p } } _ { m }$ for each model m.

Behavioral Vector.

$$
\mathbf { p } _ { m } ^ { \mathrm { b e h a v } } = [ \mathbf { y } _ { m } ; ~ \tilde { \mathbf { c } } _ { m } ; ~ \mathbf { y } _ { m } \odot ( 1 - \tilde { \mathbf { c } } _ { m } ) ; ~ \mathbf { s } _ { m } ] ,\tag{3}
$$

where $\mathbf { y } _ { m } \in \{ 0 , 1 \} ^ { S }$ is per-sample correctness, $\tilde { \mathbf { c } } _ { m } \in [ 0 , 1 ] ^ { S }$ is min-max normalized cost, the third term is the “value” vector (high weight for samples answered correctly at low cost), and $\mathbf { s } _ { m } \in \mathbb { R } ^ { 8 }$ is a summary statistics vector containing accuracy, mean cost (raw and normalized), mean cost on correct/incorrect subsets, mean value score, and correct/incorrect fractions.

Semantic Vector. Query embeddings from the calibration set are L2-normalized, then aggregated by model performance:

$$
\begin{array} { r } { \mathbf { p } _ { m } ^ { \mathrm { s e m } } = \bigl [ \bar { \mathbf { e } } _ { m } ^ { + } ; \bar { \mathbf { e } } _ { m } ^ { - } ; \bar { \mathbf { e } } _ { m } ^ { v } \bigr ] , } \end{array}\tag{4}
$$

where

$$
\bar { \mathbf { e } } _ { m } ^ { + } = \frac { \sum _ { i } Y _ { i , m } \cdot \hat { \mathbf { e } } _ { i } } { \sum _ { i } Y _ { i , m } } , \quad \bar { \mathbf { e } } _ { m } ^ { - } = \frac { \sum _ { i } ( 1 - Y _ { i , m } ) \cdot \hat { \mathbf { e } } _ { i } } { \sum _ { i } ( 1 - Y _ { i , m } ) } ,\tag{5}
$$

with $\hat { \mathbf { e } } _ { i } = \mathbf { e } _ { i } / \lVert \mathbf { e } _ { i } \rVert _ { 2 }$ . The three directional vectors encode what the model is good at, bad at, and efficient at in semantic space.

The final profile is ${ \bf p } _ { m } = [ { \bf p } _ { m } ^ { \mathrm { b e h a v } } ; ~ { \bf p } _ { m } ^ { \mathrm { s e m } } ] \in \mathbb { R } ^ { 3 S + 8 + 3 D }$ , where D is the query embedding dimension (with default encoders ${ \tt B G E - M 3 } \left[ 1 5 \right] + { \tt D I N O v 2 - l a r g e ^ { \prime } [ 1 6 ] } , { \cal D } = 2 0 4 8 )$

## 3.4 Router Architecture

We propose SCOPE-Router (Selective Calibration for Open-set Profile-Enhanced Routing)—a dual-tower architecture projecting queries and model profiles into a shared routing space.

Query Encoder. Query $x _ { i }$ is encoded via text and vision encoders with normalize-concat fusion to form $\mathbf { e } _ { i }$ After standardization, a lightweight QueryMLP projects to the routing space:

$$
\begin{array} { r } { \mathbf q _ { i } = \ell _ { 2 } \big ( \mathrm { Q u e r y M L P } ( \mathrm { s t a n d a r d i z e } ( \mathbf e _ { i } ) ) \big ) , } \end{array}\tag{6}
$$

where $\ell _ { 2 } ( \mathbf { v } ) = \mathbf { v } / \| \mathbf { v } \| _ { 2 }$ denotes L2 normalization.

Profile Encoder. Model profiles $\boldsymbol { \mathsf { p } } _ { m }$ are likewise standardized and projected via ProfileMLP:

$$
\hat { \bf p } _ { m } = \ell _ { 2 } \big ( \mathrm { P r o f i l e M L P } ( \mathrm { s t a n d a r d i z e } ( { \bf p } _ { m } ) ) \big ) .\tag{7}
$$

Both MLPs are 2-layer (hidden dimension 128 with ReLU and dropout; output dimension 64). L2-normalized outputs produce the matching score:

$$
\begin{array} { r } { s _ { i , m } = \mathbf q _ { i } ^ { \top } \hat { \pmb p } _ { m } / \tau , } \end{array}\tag{8}
$$

where τ is a temperature parameter. At inference, the router selects arg max<sub>m</sub> $s _ { i , m }$

## 3.5 Cost-Aware Training Objective

Standard routing training faces two problems: (1) hard-label softmax discards information about other correct models; (2) cost preference is only applied post-hoc. We propose a unified solution.

Relevance Target. For each query–model pair:

$$
R _ { i , m } = \mathbb { 1 } \big [ Y _ { i , m } = 1 \big ] \cdot \exp \big ( - \lambda \cdot \alpha \cdot \big ( C _ { i , m } - C _ { i } ^ { \mathrm { m i n } } \big ) \big ) ,\tag{9}
$$

where $\begin{array} { r } { C _ { i } ^ { \mathrm { m i n } } = \operatorname* { m i n } _ { m ^ { \prime } : Y _ { i , m ^ { \prime } } = 1 } C _ { i , m ^ { \prime } } } \end{array}$ is the cost of the cheapest correct model. The cheapest correct model receives $R = 1 .$ , more expensive correct models receive $R \in ( 0 , 1 )$ , and incorrect models receive $R = 0$ . For all-failed samples $( \sum _ { m } Y _ { i , m } = 0 )$ , we set $R _ { i , : } = \mathbf { 0 } ;$ they contribute to CRM as all-negative examples but are excluded from RCCR via U. Edge cases in diagnostic sampling and profile aggregation (empty correct/incorrect sets) default to zero.

CRM Loss. We predict each pair independently via binary cross-entropy rather than competing models via row-wise softmax:

$$
\mathcal { L } _ { \mathrm { C R M } } = \frac { 1 } { B K } \sum _ { i = 1 } ^ { B } \sum _ { m = 1 } ^ { K } \mathrm { B C E } \big ( \boldsymbol { \sigma } ( s _ { i , m } ) , R _ { i , m } \big ) ,\tag{10}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. Each pair’s gradient is independent, so positive signals are not diluted in multi-positive scenarios.

Routing-Consistency Contrastive Regularization (RCCR). Let $\tilde { \mathbf { r } } _ { i } = R _ { i , : } / \| R _ { i , : } \| _ { 1 }$ be the L1-normalized relevance vector. We compute pairwise sample similarity $w _ { i j } = \tilde { \mathbf { r } } _ { i } ^ { \top } \tilde { \mathbf { r } } _ { j }$ and row-normalize into $\hat { w } _ { i j }$ . Define $\mathcal { U } = \left. i : \| R _ { i , : } \| _ { 1 } > 0 \right.$ and $\begin{array} { r } { \hat { \mathcal { V } } = \left\{ i \in \mathcal { U } : \sum _ { j \neq i } \hat { w _ { i j } } > 0 \right\} } \end{array}$ :

$$
\mathcal { L } _ { \mathrm { R C C R } } = - \frac { 1 } { | \mathcal { V } | } \sum _ { \overset { \substack { i \in \mathcal { V } } } { j \in \mathcal { U } } } \hat { w } _ { i j } \log \frac { \exp ( \mathbf { q } _ { i } ^ { \top } \mathbf { q } _ { j } / \tau _ { s } ) } { \displaystyle \sum _ { \substack { \ell \in \mathcal { U } , \ell \neq i } } \exp ( \mathbf { q } _ { i } ^ { \top } \mathbf { q } _ { \ell } / \tau _ { s } ) } ,\tag{11}
$$

where $\tau _ { s }$ is a temperature parameter.

Total Loss.

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C R M } } + \mu \cdot \mathcal { L } _ { \mathrm { R C C R } } , } \end{array}\tag{12}
$$

with $\mu = 0 . 1$ by default. Since this objective operates solely on $Y , C ,$ and the router’s output logits, it is architecture-agnostic and can be applied to any trainable router as a replacement loss.

## 4 VLM-ExecRouterBench

## 4.1 Datasets

Our routing benchmark spans three task domains with 34k samples (374k sample-model pairs) and 5.1B inputoutput tokens. All datasets use the unified Routing Input / Execution Context / Verification Rule structure (described in the Data Generation section), split 70%/10%/20%.

Code. MBPP [17], BigCodeBench [18], APPS [19], and LiveCodeBench [20] cover code synthesis from introductory to competition-level, verified by unit tests.

Agentic. MathVista [21], ChartQA [22], MMMU [23], OCRBench [24], DocVQA [25], AI2D [26], and Real-WorldQA [27] are reformulated under an agentic setting requiring multi-step tool calls.

Search. BrowseComp-Plus [28] is a deep-research benchmark derived from BrowseComp with a fixed, humanverified corpus. Models perform multi-step retrieval over the provided document collection to answer complex queries.

![](images/23122a6e1f116c012438c6b05640083c18eeffba062cfe625523a14eb805141b.jpg)  
Figure 2. Data composition of VLM-ExecRouterBench.

## Candidate Model Pool.

<table><tr><td>#</td><td>Model</td><td>Provider</td><td>Input ($)</td><td>Output ($)</td></tr><tr><td>1</td><td>Qwen3-VL-8B-Instruct</td><td>Qwen / Alibaba</td><td>0.08</td><td>0.50</td></tr><tr><td>2</td><td>Gemini 2.5 Flash Lite</td><td>Google</td><td>0.10</td><td>0.40</td></tr><tr><td>3</td><td>Qwen3.5-35B-A3B</td><td>Qwen / Alibaba</td><td>0.14</td><td>1.00</td></tr><tr><td>4</td><td>MiniMax M3</td><td>MiniMax</td><td>0.30</td><td>1.20</td></tr><tr><td>5</td><td>Gemini 3 Flash Preview</td><td>Google</td><td>0.50</td><td>3.00</td></tr><tr><td>6</td><td>GPT-5.4 mini</td><td>OpenAI</td><td>0.75</td><td>4.50</td></tr><tr><td>7</td><td>Claude Haiku 4.5</td><td>Anthropic</td><td>1.00</td><td>5.00</td></tr><tr><td>8</td><td>Gemini 3.5 Flash</td><td>Google</td><td>1.50</td><td>9.00</td></tr><tr><td>9</td><td>GPT-5.4</td><td>OpenAI</td><td>2.50</td><td>15.00</td></tr><tr><td>10</td><td>Claude Sonnet 4.6</td><td>Anthropic</td><td>3.00</td><td>15.00</td></tr><tr><td>11</td><td>Claude Opus 4.6</td><td>Anthropic</td><td>5.00</td><td>25.00</td></tr></table>

Table 1. Candidate model pool. Prices are per 1M tokens.

We select 11 representative VLMs as the candidate pool M, spanning diverse capability tiers and cost ranges (Table 1), including Qwen3-VL-8B [29], Qwen3.5-35B-A3B [30], Gemini 2.5/3/3.5 Flash series [31, 32, 33], GPT-5.4 mini/5.4 [34, 35], Claude Haiku 4.5/Sonnet 4.6/Opus 4.6 [36, 37, 38], and MiniMax M3 [39]. Each model executes the full Execution Context on all samples, yielding $Y \in \{ 0 , 1 \} ^ { N \times 1 1 }$ and $C \in \mathbb { R } _ { + } ^ { N \times 1 1 }$

<table><tr><td rowspan="2">Router</td><td colspan="4">VLM-ExecRouterBench</td><td colspan="4">VL-RouterBench</td><td colspan="4">MMR-Bench</td></tr><tr><td>|Avg. Acc. ↑</td><td></td><td></td><td></td><td>Avg. Cost ↓ Rank Score ↑ Rank ↓ | Avg. Acc. ↑ Avg. Cost ↓ Rank Score ↑ Rank ↓ | Avg. Acc. ↑ Avg. Cost ↓ Rank Score ↑ Rank ↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Training-free Baselines</td><td colspan="3"></td></tr><tr><td>Oracle</td><td>95.25</td><td>59.53</td><td>91.77</td><td>0</td><td>95.60</td><td>0.37</td><td>93.68</td><td>0</td><td>91.67</td><td>10.41</td><td>75.67</td><td>0</td></tr><tr><td>Strongest</td><td>86.86</td><td>308.24</td><td>66.75</td><td>13</td><td>78.01</td><td>2.72</td><td>68.88</td><td>11</td><td>74.25</td><td>81.30</td><td>50.64</td><td>14</td></tr><tr><td>Cheapest</td><td>67.74</td><td>18.94</td><td>69.78</td><td>12</td><td>62.43</td><td>0.14</td><td>64.63</td><td>14</td><td>42.25</td><td>0.25</td><td>43.20</td><td>15</td></tr><tr><td></td><td> $3 3 . 7 0 \pm 3 . 7 3$ </td><td></td><td></td><td></td><td>End-to-End Routers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"> $7 9 . 5 5 \pm 0 . 5 0 $  |70.24 ± 0.62 0.61 ± 0.16</td><td colspan="3"></td></tr><tr><td>RouterDC</td><td> $\lvert 7 9 . 1 7 \pm 0 . 4 6$   $\lvert 7 9 . 1 1 \pm 0 . 8 4$ </td><td> $3 8 . 6 5 \pm 5 . 0 4$ </td><td> $7 9 . 1 6 \pm 0 . 8 4$ </td><td></td><td> $\overline { { 7 7 . 5 2 \pm 1 . 0 4 } } 1 . 0 4 \pm 0 . 2 6$ </td><td></td><td> $6 9 . 9 0 \pm 0 . 3 4$   ${ \underline { { 7 4 . 5 9 \pm 1 . 0 5 } } }$ </td><td></td><td></td><td> $\lvert 6 1 . 3 1 \pm 0 . 5 8 7 . 3 7 \pm 1 . 9 8$   $\ H _ { 6 2 . 2 2 \pm 0 . 8 8 } \ 7 . 3 3 \pm 1 . 7 8$ </td><td> $5 6 . 0 5 \pm 0 . 7 7$   $5 6 . 7 5 \pm 0 . 8 9$ </td><td>76</td></tr><tr><td>ZOOTER</td><td> $\vert 7 4 . 3 2 \pm 1 . 2 3$ </td><td>27.55 ± 4.87</td><td> $7 5 . 4 7 \pm 1 . 2 0$ </td><td>2-396</td><td> $| 7 4 . 6 5 \pm 1 . 3 4 0 . 9 3 \pm 0 . 1 5$ </td><td></td><td>72.58 ± 1.03</td><td>9273</td><td></td><td> $\lvert 6 1 . 6 5 \pm 1 . 2 1 3 . 6 2 \pm 0 . 6 3$ </td><td> $5 7 . 7 7 \pm 1 . 0 2$ </td><td>53</td></tr><tr><td>VLC</td><td> $\left| 7 5 . 8 3 \pm 1 . 2 0 \right.$ </td><td> $\overline { { 3 0 . 9 3 \pm 0 . 8 0 } }$ </td><td> $7 6 . 6 6 \pm 1 . 1 2$ </td><td></td><td> $7 8 . 0 9 \pm 1 . 1 7 1 . 2 3 \pm 0 . 0 3$ </td><td></td><td> $7 4 . 3 3 \pm 0 . 5 1$ </td><td></td><td></td><td> $\overline { { 6 5 . 0 6 \pm 1 . 0 3 5 . 5 0 \pm 0 . 1 3 } }$ </td><td> $5 9 . 5 9 \pm 0 . 7 9$ </td><td></td></tr><tr><td colspan="10"></td><td colspan="3"></td></tr><tr><td>KNN</td><td></td><td>40.93</td><td>71.41</td><td>11</td><td>Feature-level Routers 66.26</td><td></td><td>67.13</td><td>12</td><td>58.66</td><td>7.07</td><td>54.11</td><td>10</td></tr><tr><td>PRkNN</td><td>70.81 75.10</td><td>40.72</td><td>75.36</td><td>10</td><td>70.68</td><td>0.38 0.41</td><td>71.09</td><td>8</td><td>59.72</td><td>7.30</td><td>54.85</td><td>9</td></tr><tr><td>OVR</td><td>85.27</td><td>407.16</td><td>55.14</td><td>15</td><td>75.49</td><td>2.18</td><td>68.92</td><td>10</td><td>69.01</td><td>55.43</td><td>51.98</td><td>12</td></tr><tr><td>K-means</td><td>85.69</td><td>362.20</td><td>60.73</td><td>14</td><td>70.01</td><td>2.21</td><td>64.62</td><td>15</td><td>70.12</td><td>61.59</td><td>51.63</td><td>13</td></tr><tr><td>Linear</td><td></td><td>79.85 ± 1.11 101.15 ± 10.01</td><td> $7 6 . 1 5 \pm 1 . 0 7$ </td><td></td><td>68.32 ± 7.45 1.32 ± 0.33</td><td></td><td> $6 5 . 6 8 \pm 5 . 6 9$ </td><td>13</td><td></td><td> $\overline { { 1 . 0 . 6 0 \pm 1 . 0 5 ~ 5 3 . 7 6 \pm 4 . 8 9 } }$ </td><td> $5 3 . 0 5 \pm 0 . 9 3$ </td><td>11</td></tr><tr><td>MLP</td><td>75.88 ± 0.53</td><td> $2 8 . 4 5 \pm 3 . 5 5$ </td><td> $7 6 . 8 7 \pm 0 . 5 5$ </td><td></td><td>77.49 ± 0.56 1.13 ± 0.13</td><td></td><td> $7 4 . 2 3 \pm 0 . 2 2$ </td><td></td><td> $\overline { { 6 0 . 8 8 \pm 0 . 4 1 } } 7 . 4 5 \pm 0 . 8 9$ </td><td></td><td> $5 5 . 6 9 \pm 0 . 4 2$ </td><td></td></tr><tr><td>UniRoute-KM</td><td>76.57 ± 0.76</td><td> $4 8 . 1 6 \pm 1 3 . 1 3$ </td><td> $7 6 . 2 6 \pm 1 . 0 3$ </td><td></td><td>75.88 ± 0.73 0.82 ± 0.21</td><td></td><td> $7 4 . 0 4 \pm 1 . 0 8$ </td><td>45</td><td> $\ H 6 0 . 8 8 \pm 0 . 6 1 0 . 5 1 \pm 0 . 1 4$ </td><td></td><td> $5 9 . 7 2 \pm 0 . 6 0$ </td><td></td></tr><tr><td>UniRoute-LM</td><td> $7 8 . 5 5 \pm 1 . 6 1$ </td><td> $4 2 . 5 8 \pm 8 . 2 9$ </td><td> $7 8 . 4 0 \pm 1 . 5 5$ </td><td>8574</td><td> $7 5 . 1 4 \pm 0 . 9 6 0 . 6 8 \pm 0 . 1 4$ </td><td></td><td> $7 3 . 8 6 \pm 1 . 0 4$ </td><td>6</td><td> $\overline { { 5 8 . 2 3 \pm 1 . 1 6 } } 0 . 2 7 \pm 0 . 0 5$ </td><td></td><td> $\overline { { 5 7 . 9 1 \pm 1 . 0 5 } }$ </td><td>8241</td></tr><tr><td>SCOPE-Router (Ours)81.65 ± 1.06</td><td></td><td> $4 6 . 2 0 \pm 1 1 . 6 2$ </td><td> ${ \bf 8 0 . 9 4 \pm 1 . 2 2 }$ </td><td>1</td><td> ${ \bf 7 9 . 1 1 \pm 1 . 0 2 0 . 9 8 \pm 0 . 2 7 }$ </td><td></td><td> ${ \bf 7 6 . 1 8 \pm 1 . 4 4 }$ </td><td>1</td><td> $\overline { { 1 6 2 . 9 0 \pm 0 . 8 1 } } \overline { { 0 . 6 5 \pm 0 . 1 7 } }$ </td><td></td><td> ${ \bf 6 1 . 2 3 \pm 0 . 7 5 }$ </td><td></td></tr></table>

Table 2. Performance of router methods on VLM-ExecRouterBench, VL-RouterBench, and MMR-Bench. Avg. Acc. denotes average accuracy (%), Avg. Cost denotes average cost (\$/10K samples), and methods are ranked by Rank Score.

Routing Methods. We evaluate four categories of routing methods (detailed descriptions in Appendix C): Training-free baselines. Oracle is the theoretical upper bound and selects the cheapest correct model; Strongest always selects the globally best-performing model; Cheapest always selects the globally cheapest model.

Feature-level routers. These methods use embeddings from frozen encoders to make routing decisions via non-parametric methods, clustering, or lightweight classifiers. They include KNN [40], PRkNN, Kmeans [41], Linear [42], MLP [43], OVR [44], and UniRoute. We adopt UniRoute as the open-set baseline because its routing mechanism is modality-agnostic; other open-set routers rely on text-only features and require non-trivial redesign for multimodal settings.

End-to-end routers. These methods learn the routing decision function end-to-end and include CosineCls, RouterDC, ZOOTER, and VLC.

SCOPE-Router (Ours). Our method is a calibration-profile-based dual-tower matching router trained with CRM+RCCR (see the Method section).

Evaluation Metrics. For each router, we report Average Accuracy (fraction of test samples routed correctly), Average Cost (mean inference cost in \$/10K samples), and Rank Score $S _ { \beta } \left( \mathrm { E q . } \left( 1 \right) , \bar { \beta } = 0 . 1 \right)$ . Rank Score serves as the primary ranking criterion. Formal definitions of all metrics are provided in Appendix D. We also sweep λ on VL-RouterBench to trace each router’s accuracy–cost Pareto frontier (Figure 4).

## 5 Experiments

## 5.1 Main Results

We evaluate all routing methods on three benchmarks: (1) VLM-ExecRouterBench (ours); (2) VL-RouterBench; and (3) MMR-Bench [10]. For each benchmark, all routers are trained on its corresponding training split. We sweep the cost-sensitivity coefficient $\lambda \in \{ 0 , 1 0 , 1 0 0 , 1 0 0 0 , 1 0 0 0 0 , + \infty \}$ for each method and report results at the λ achieving the best validation Rank Score. Training details and per-method hyperparameters are in Appendix C.

Table 2 presents results across all three benchmarks. SCOPE-Router ranks first in Rank Score on all three benchmarks, the only method to achieve this across benchmarks spanning traditional VQA, executionoriented tasks, and multimodal reasoning.

![](images/7e2ecbcc346616fa826190853e2c8d120fe5fb740d8dcfab0af779f4c168d6fc.jpg)

![](images/96783ec0751c7cf7d985db97f08cca668f31c03daf99583ba4d65b933ed3ace1.jpg)

![](images/1755598ccc52d97b3626d0b39ade5dc506037371deecaac8e0670691813f3fe0.jpg)  
Figure 3. Transferability of our cost-aware loss across different end-to-end routers on VL-RouterBench and VLM-ExecRouterBench. Applying our loss consistently improves the overall routing performance across RouterDC, ZOOTER, CosineCls, and VLC.

On VLM-ExecRouterBench, SCOPE-Router obtains a Rank Score of 80.94 ± 1.22, followed by CosineCls (79.55 ± 0.50). Compared with the Strongest baseline, it reduces cost by 85% while sacrificing only 5.21pp in accuracy. The top end-to-end routers (CosineCls, RouterDC) outperform MLP on execution tasks—a pattern not observed on VL-RouterBench—suggesting that execution domains’ higher inter-task heterogeneity benefits more expressive matching. A per-dataset breakdown in Appendix Table 5 reveals that SCOPE-Router’s overall advantage stems from consistent competitiveness across all three domains rather than dominance in any single one: it achieves the highest accuracy on Search (68.18%) and DocVQA (97.92%), while maintaining strong cost efficiency on Code and Agentic tasks where cost-unaware methods (OVR, Kmeans) achieve higher accuracy but at 3–8× the cost.

On VL-RouterBench, SCOPE-Router scores 76.18 ± 1.44 vs. the runner-up RouterDC (74.59 ± 1.05). The mean±std intervals overlap, so the ranking difference on this benchmark alone is not conclusive; however, SCOPE-Router simultaneously achieves the highest accuracy (79.11 ± 1.02%) and reduces cost by 64% vs. Strongest, indicating a favorable quality–cost tradeoff.

![](images/80a4b56cb7c6b2f1f3052a9686d38231ac48f69f15f78b0b6f8cac1d0225d4f9.jpg)  
Figure 4. Accuracy–cost trade-offs on VL-RouterBench. Curves trace each router’s Pareto front across λ; points nearer the upper-left are better.

Here MLP (74.23 ± 0.22) performs comparably to end-to-end routers (RouterDC 74.59, VLC 74.33), indicating that frozen encoders already provide strong discriminative features for traditional VQA tasks.

On MMR-Bench, SCOPE-Router achieves 61.23 ± 0.75, surpassing UniRoute-KM (59.72 ± 0.60) by 1.51 points with mean±std intervals that do not overlap, at merely 0.8% of the Strongest baseline’s cost (0.65 vs. 81.30).

## 5.2 Ablation Studies

Loss Function Ablation. We ablate the training objective on VL-RouterBench (Table 3). CRM alone (w/o RCCR) achieves 75.50, a 1.0-point gain over pure softmax (74.51), confirming that independent per-model scoring avoids multi-positive dilution. RCCR is effective only under CRM: adding RCCR to softmax hurts performance (74.51 → 73.41), whereas adding it to CRM improves Rank Score from 75.50 to 76.18 (+0.68). This reveals that CRM’s unnormalized scoring framework is a prerequisite for RCCR’s distributional regularization to be beneficial—the two components are complementary rather than independently additive.

<table><tr><td>Configuration</td><td></td><td></td><td>| Avg. Acc. ↑ Avg. Cost ↓ Rank Score ↑</td></tr><tr><td>Full model</td><td>79.11</td><td>0.98</td><td>76.18</td></tr><tr><td>w/o RCCR</td><td>77.90</td><td>0.89</td><td>75.50</td></tr><tr><td>w/o CRM</td><td>75.82</td><td>0.97</td><td>73.41</td></tr><tr><td>w/o CRM, RCCR</td><td>76.13</td><td>0.76</td><td>74.51</td></tr></table>

Table 3. Loss function ablation on VL-RouterBench. RCCR denotes Routing-Consistency Contrastive Regularization.
<table><tr><td rowspan="2">Router</td><td colspan="4">In-Distribution</td><td colspan="4">Out-of-Distribution</td></tr><tr><td>| Avg. Acc. ↑</td><td>Avg. Cost↓</td><td>Rank Score ↑</td><td></td><td>Rank ↓ | Avg. Acc. ↑</td><td>Avg. Cost ↓ Rank Score ↑</td><td></td><td>Rank↓</td></tr><tr><td>UniRoute-KM</td><td>73.58</td><td>31.78</td><td>74.51</td><td>2</td><td>76.95</td><td>3.21</td><td>78.60</td><td>2</td></tr><tr><td>UniRoute-LM</td><td>73.05</td><td>31.03</td><td>74.06</td><td>3</td><td>75.86</td><td>2.82</td><td>77.56</td><td>3</td></tr><tr><td>SCOPE-Router (Ours)</td><td>78.44</td><td>82.93</td><td>75.96</td><td>1</td><td>84.21</td><td>19.84</td><td>85.35</td><td>1</td></tr></table>

Table 4. Open-set routing results with 5 of 11 candidate models held out under in-distribution and out-of-distribution settings.

Generalizability of the Loss Function. CRM+RCCR applied as a replacement loss to RouterDC, ZOOTER, CosineCls, and VLC improves Rank Score on both benchmarks (Figure 3), with gains ranging from +1.25 to +6.21 points. Gains are larger on VLM-ExecRouterBench, where execution tasks’ stronger inter-model complementarity amplifies the benefit of eliminating multi-positive dilution. This confirms the objective’s cross-architecture generalizability.

OOD Generalization. We select 5 datasets from VLM-ExecRouterBench (MBPP, LiveCodeBench, ChartQA, OCRBench, RealWorldQA) as OOD test sets, with the remaining data split 9:1 into training and validation sets (Table 1). All 11 models remain in the candidate pool; only the task distribution shifts. SCOPE-Router achieves the highest Rank Score (88.14), outperforming the runner-up K-means (86.30) by 1.84 points at only 21.8% of its cost. Its OOD score exceeds the in-distribution score (80.94), consistent with higher Oracle performance in the OOD set (97.15 vs. 91.77), indicating stronger inter-model complementarity in these held-out tasks.

Open-Set Routing. We conduct open-set experiments on VLM-ExecRouterBench by holding out 5 of 11 candidate models as unseen (Gemini 2.5 Flash Lite, MiniMax M3, GPT-5.4, Gemini 3.5 Flash, Claude Opus 4.6), excluded from profile construction and router training. At evaluation time, their profiles are generated from calibration samples and added to the candidate pool. On in-distribution data, SCOPE-Router achieves Rank Score 75.96 (Table 4), outperforming UniRoute-KM by 1.45 points. Under doubly OOD evaluation—unseen models on the same 5 held-out datasets (Table 4)—SCOPE-Router’s advantage expands to +6.75 Rank Score and +7.26 accuracy over UniRoute-KM, demonstrating that profile-based matching is more robust than cluster-based methods under distribution shifts.

MLP Hidden Dimension. The optimum is 128 dimensions (Table 2), though scores span only 0.62 points across {64–4096}, indicating low sensitivity.

Freezing vs. Unfreezing the Query Embedder. Freezing the query embedder outperforms unfreezing on VL-RouterBench (76.18 vs. 75.68; Table 3) while reducing training cost; we adopt freezing as default.

Calibration Strategy Ablation. Among single strategies (Table 4), Diversity achieves the best Rank Score (75.69); the full triple combination achieves 76.18, confirming complementarity.

Text and Vision Encoder Selection. Across 25 encoder combinations (Figure 1), Rank Scores span <0.8 points (75.54–76.33), confirming robustness to encoder choice.

## 6 Conclusion

We present VLM-ExecRouterBench, SCOPE-Router, and CRM+RCCR, advancing VLM routing along evaluation coverage, open-set capability, and cost-aware training. SCOPE-Router achieves the best Rank Score on all three benchmarks, leads by 1.84 points under OOD and 6.75 points under doubly OOD open-set evaluation. CRM+RCCR consistently improves four architecturally diverse routers by +1.25 to +6.21 points. Limitations are discussed in Appendix F.

## References

[1] Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm routing system, 2024. URL https://arxiv.org/abs/2403.12031.

[2] Zhehao Huang, Baijiong Lin, Jingyuan Zhang, Jingying Wang, Yuhang Liu, Ning Lu, Tao Li, and Xiaolin Huang. Vl-routerbench: A benchmark for vision-language model routing, 2026. URL https: //arxiv.org/abs/2512.23562.

[3] Yifan Lu, Rixin Liu, Jiayi Yuan, Xingqi Cui, Shenrun Zhang, Hongyi Liu, and Jiarong Xing. Routerarena: An open platform for comprehensive comparison of llm routers, 2025. URL https://arxiv.org/abs/ 2510.00202.

[4] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data, 2025. URL https: //arxiv.org/abs/2406.18665.

[5] Seamus Somerstep, Felipe Maia Polo, Allysson Flavio Melo de Oliveira, Prattyush Mangal, Mírian Silva, Onkar Bhardwaj, Mikhail Yurochkin, and Subha Maity. Carrot: A cost aware rate optimal router, 2025. URL https://arxiv.org/abs/2502.03261.

[6] Pengfei Zhou, Zhiwei Tang, Yixing Ma, Jiasheng Tang, Yizeng Han, Zhenglin Wan, Fanqing Meng, Wei Wang, Bohan Zhuang, Wangbo Zhao, et al. Agent-as-a-router: Agentic model routing for coding tasks. arXiv preprint arXiv:2606.22902, 2026.

[7] Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang. Routerdc: Query-based router by dual contrastive learning for assembling large language models, 2024. URL https://arxiv.org/abs/ 2409.19886.

[8] Wittawat Jitkrittum, Harikrishna Narasimhan, Ankit Singh Rawat, Jeevesh Juneja, Congchao Wang, Zifeng Wang, Alec Go, Chen-Yu Lee, Pradeep Shenoy, Rina Panigrahy, Aditya Krishna Menon, and Sanjiv Kumar. Universal model routing for efficient llm inference, 2025. URL https://arxiv.org/abs/ 2502.08773.

[9] Cheng Qian, Zuxin Liu, Shirley Kokane, Akshara Prabhakar, Jielin Qiu, Haolin Chen, Zhiwei Liu, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, and Huan Wang. xrouter: Training cost-aware llms orchestration system via reinforcement learning, 2025. URL https://arxiv.org/abs/ 2510.08439.

[10] Haoxuan Ma, Guannan Lai, and Han-Jia Ye. Mmr-bench: A comprehensive benchmark for multimodal llm routing, 2026. URL https://arxiv.org/abs/2601.17814.

[11] Chenxu Wang, Hao Li, Yiqun Zhang, Linyao Chen, Jianhao Chen, Ping Jian, Peng Ye, Qiaosheng Zhang, and Shuyue Hu. Icl-router: In-context learned model representations for llm routing, 2025. URL https://arxiv.org/abs/2510.09719.

[12] Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models, 2023. URL https: //arxiv.org/abs/2311.08692.

[13] Hao Li, Yiqun Zhang, Zhaoyan Guo, Chenxu Wang, Shengji Tang, Qiaosheng Zhang, Yang Chen, Biqing Qi, Peng Ye, Lei Bai, Zhen Wang, and Shuyue Hu. Llmrouterbench: A massive benchmark and unified framework for llm routing, 2026. URL https://arxiv.org/abs/2601.07206.

[14] Jingjun Xu, Hongji Pu, Tao Feng, Haozhen Zhang, Jiaxuan You, and Ge Liu. Routeprofile: Graph-based profiling for cold-start llm routing, 2026. URL https://arxiv.org/abs/2605.00180.

[15] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multilingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2024.

[16] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.

[17] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

[18] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, Simon Brunner, Chen Gong, Thong Hoang, Armel Randy Zebaze, Xiaoheng Hong, Wen-Ding Li, Jean Kaddour, Ming Xu, Zhihan Zhang, Prateek Yadav, Naman Jain, Alex Gu, Zhoujun Cheng, Jiawei Liu, Qian Liu, Zijian Wang, Binyuan Hui, Niklas Muennighoff, David Lo, Daniel Fried, Xiaoning Du, Harm de Vries, and Leandro Von Werra. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions, 2025. URL https://arxiv.org/abs/2406.15877.

[19] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with apps, 2021. URL https://arxiv.org/abs/2105.09938.

[20] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/2403.07974.

[21] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024. URL https://arxiv.org/abs/2310.02255.

[22] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning, 2022. URL https://arxiv.org/abs/ 2203.10244.

[23] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2024. URL https://arxiv.org/abs/2311.16502.

[24] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), December 2024. ISSN 1869-1919. doi: 10.1007/ s11432-024-4235-6. URL http://dx.doi.org/10.1007/s11432-024-4235-6.

[25] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images, 2021. URL https://arxiv.org/abs/2007.00398.

[26] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016. URL https://arxiv.org/abs/1603.07396.

[27] xAI. Grok-1.5 vision preview. https://x.ai/news/grok-1.5v, April 2024. Introduces the RealWorldQA benchmark.

[28] Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent, 2025. URL https://arxiv.org/abs/ 2508.06600.

[29] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

[30] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog? id=qwen3.5.

[31] Google DeepMind. Gemini 2.5 Flash Model Card. Model Card, 2025. URL https://storage.googleapis. com/deepmind-media/Model-Cards/Gemini-2-5-Flash-Model-Card.pdf.

[32] Google DeepMind. Gemini 3 Flash Model Card. Model Card, December 2025. URL https://deepmind. google/models/model-cards/gemini-3-flash/.

[33] Google DeepMind. Gemini 3.5 Flash Model Card. Model Card, May 2026. URL https://deepmind. google/models/model-cards/gemini-3-5-flash/.

[34] OpenAI. Introducing GPT-5.4, March 2026. URL https://openai.com/index/introducing-gpt-5-4/.

[35] OpenAI. Introducing GPT-5.4 mini and nano, March 2026. URL https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/.

[36] Anthropic. System card: Claude Haiku 4.5. System Card, October 2025. URL https://www.anthropic. com/claude-haiku-4-5-system-card.

[37] Anthropic. System card: Claude Sonnet 4.6. System Card, February 2026. URL https://www.anthropic. com/claude-sonnet-4-6-system-card.

[38] Anthropic. System card: Claude Opus 4.6. System Card, February 2026. URL https://www.anthropic. com/claude-opus-4-6-system-card.

[39] Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, and Pengyu Zhao. Minimax sparse attention, 2026. URL https://arxiv.org/abs/2606.13392.

[40] Evelyn Fix and Joseph L. Hodges. Discriminatory analysis—nonparametric discrimination: Consistency properties. Technical Report Project 21-49-004, Report No. 4, USAF School of Aviation Medicine, Randolph Field, Texas, February 1951.

[41] James B. MacQueen. Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, volume 1, pages 281–297. University of California Press, 1967.

[42] Daniel McFadden. Conditional logit analysis of qualitative choice behavior. In Paul Zarembka, editor, Frontiers in Econometrics, pages 105–142. Academic Press, New York, 1974.

[43] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by backpropagating errors. Nature, 323(6088):533–536, 1986. doi: 10.1038/323533a0.

[44] Ryan M. Rifkin and Aldebaro Klautau. In defense of one-vs-all classification. J. Mach. Learn. Res., 5: 101–141, 2004.

[45] Yubo Zhang, Xueqing Wang, Manhui Lin, Yue Zhang, Penglongyi Deng, Ting Sun, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Changda Zhou, Hongen Liu, Suyin Liang, Cheng Cui, Yi Liu, Dianhai Yu, and Yanjun Ma. Pp-ocrv6: From 1.5m to 34.5m parameters, surpassing billion-scale vlms on ocr tasks, 2026. URL https://arxiv.org/abs/2606.13108.

[46] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

[47] OpenAI. GPT-5.5 System Card. https://openai.com/index/gpt-5-5-system-card/, April 2026. Accessed: 2026-07-29.

## Appendix

A Data Generation 15   
B Source Datasets 16   
B.1 Code 16   
B.2 Agentic 16   
B.3 Search . 17   
C Implementation Details 17   
D Evaluation Metrics 18   
E Additional Results 18   
F Limitations 20

## A Data Generation

VLM-ExecRouterBench provides a reproducible routing benchmark built from standardized executions of a shared candidate VLM pool. The final benchmark contains 33,966 samples and 373,626 sample–model executions, covering 12 datasets and 11 candidate VLMs. To support reuse and independent extension, we will release the data-generation pipeline together with the benchmark data, including the source conversion scripts, executor prompts, tool schemas, verification code, and saved execution trajectories. Figure 5 summarizes the end-to-end construction process across the three execution domains.

Unified task interface. Each source example is first converted into a common execution-oriented schema with three components. The Routing Input contains only the pre-execution information available to the router. The Execution Context contains the complete prompt, visual inputs, tool specification, or retrieval context sent to the candidate VLM. The Verification Rule specifies the task-dependent checker used to map a model response to a binary correctness label. This design keeps the router input lightweight while evaluating each VLM under the full execution protocol.

Execution backends and tools. The three benchmark domains share the same schema but use different execution backends.

Code. Code examples are converted into a unified executor prompt containing the problem statement and available public tests. Candidate VLMs return code solutions, which are checked by unit tests, input–output tests, or official benchmark evaluators depending on the source dataset.

Agentic. Multimodal document and visual reasoning examples are executed through a lightweight multimodal agent loop. The agent receives the image and question directly, can call image tools before answering, and must return a final natural-language response. The available tools include image inspection for dimensions, crop and zoom operations for isolating local visual regions, and an OCR[45] tool for reading image text. OCR is implemented with a PaddleOCR-backed HTTP service that returns recognized text blocks, confidence scores, and bounding boxes. This tool interface makes document, chart, diagram, and OCR-heavy VQA tasks closer to realistic interactive VLM use, rather than a single opaque image-to-answer call.

Search. BrowseComp-Plus examples are executed with a bounded fixed-corpus retrieval tool. Each model may issue retrieval calls against the provided corpus and then must return an exact final answer string. Retrieval is implemented with a dense retriever based on Qwen3-Embedding-8B[46]: query embeddings are generated with the retrieval instruction prefix used for web-search-style passage retrieval, L2-normalized, and scored against precomputed corpus embeddings by inner product. Each retrieval call returns the top five passages with their ranks, similarity scores, document identifiers, and available fields such as title and URL. Passage text is capped at 2,400 characters before being returned to the model, which keeps the search context bounded while still preserving enough evidence for multi-hop answer synthesis.

Trajectory logging. In addition to final correctness labels, the generation pipeline records execution trajectories for downstream analysis. For multimodal agent tasks, the trajectory contains the model turns, tool schemas, issued tool calls, crop/zoom or OCR results, tool-budget state, and final answer. For Search tasks, the trajectory records retrieval queries, retrieved document identifiers, snippets, and final answers. These traces are not used by the router at inference time, but they make the benchmark useful for studying model tool-use behavior, failure modes, and future trajectory-aware routing methods.

Verification and matrix construction. Responses are verified using the strongest available automatic checker for each source: executable tests for code, normalized exact/choice matching for short-answer visual tasks, GPT-5.5[47] semantic judging only when rule-based matching is insufficient, and answer matching against the BrowseComp-Plus references for search. After verification, every sample–model pair contributes one entry to the correctness matrix $Y \in \{ 0 , 1 \} ^ { N \times K }$ . Logged token usage and the corresponding input/output prices define the cost matrix $C \in \mathbb { R } _ { + } ^ { N \times K }$ The official train/dev/test split contains $2 3 , 7 7 6 / 3 , 3 9 6 / 6 , 7 9 4$ samples; the full dataset composition is visualized in the main paper.

## B Source Datasets

## B.1 Code

• MBPP contributes 974 entry-level Python programming tasks focused on basic algorithmic and functionalprogramming skills. Each problem is paired with unit tests, making it a clean source for evaluating whether a routed model can produce executable and semantically correct code rather than merely plausible text.

• BigCodeBench contributes 1,139 more realistic Python programming tasks with richer function specifications and hidden test suites. Compared with MBPP, it introduces broader library usage, longer prompts, and more implementation-heavy requirements, making it useful for distinguishing models that differ in practical coding ability.

• APPS contributes 4,507 competitive-programming problems described by natural-language statements and evaluated through input–output tests. It emphasizes algorithmic reasoning under stdin/stdout-style execution, where solutions must generalize beyond visible examples and handle edge cases.

• LiveCodeBench contributes 400 recent programming-contest-style problems, covering both function-call and standard-input formats. Its temporally newer tasks reduce overlap with older training distributions and provide a sharper test of current code-generation robustness.

## B.2 Agentic

• MMMU contributes 10,813 multimodal questions spanning college-level disciplines and heterogeneous visual formats such as diagrams, tables, charts, and scientific figures. It is included to evaluate expert-level visual reasoning that requires both domain knowledge and deliberate multimodal understanding.

• DocVQA contributes 5,347 document-image question-answering examples. It targets visually grounded reading, layout understanding, and information extraction from scanned or rendered documents, covering a capability that differs substantially from natural-image recognition.

• ChartQA contributes 4,411 questions over charts and plots. It tests whether a model can interpret visual encodings, compare quantities, and perform chart-specific reasoning, making it a useful diagnostic for models that can read text but struggle with structured visual data.

• AI2D contributes 3,049 grade-school science diagram questions. The dataset stresses diagram parsing, symbolic visual elements, arrows, object-relation reasoning, and educational science knowledge, complementing document and chart understanding with diagram-centric reasoning.

• OCRBench contributes 983 text-centric visual understanding examples. It evaluates OCR and downstream reasoning over text embedded in images, capturing failure modes where a model’s final answer is bottlenecked by visual text recognition.

• MathVista contributes 980 visually grounded mathematical reasoning problems. It combines figures, tables, diagrams, and quantitative reasoning, allowing the benchmark to probe fine-grained differences in multimodal math capability.

• RealWorldQA contributes 737 questions about real-world images. It provides open-ended perception and commonsense reasoning over natural scenes, balancing the more structured document, chart, and diagram datasets in the Agentic domain.

## B.3 Search

• BrowseComp-Plus contributes 626 fixed-corpus deep-search questions. It forms the Search domain: models must retrieve evidence from a bounded corpus and synthesize the final answer, allowing routing evaluation to include retrieval-grounded reasoning rather than only direct question answering.

## C Implementation Details

Query and profile representation. Unless otherwise stated, we use BGE-M3[15] as the text encoder and DINOv2-large[16] as the vision encoder. Text and vision embeddings are individually normalized and then concatenated. SCOPE-Router builds a calibration-derived behavior profile for each candidate VLM. A QueryMLP projects the incoming query representation to a 64-dimensional routing vector, while a ProfileMLP projects each model profile to the same 64-dimensional space; the router scores their match in this shared routing space. This keeps the router independent of model identity embeddings and allows newly onboarded VLMs to be introduced through calibration profiles.

Calibration. The default calibration budget is 1024 training samples. Hybrid calibration combines random coverage, diagnostic samples selected by model disagreement and cost-sensitive utility, and diversity samples selected in the frozen embedding space. The default mixture uses 50% random, 30% diagnostic, and 20% diversity samples. Profiles are built from calibration correctness, cost, value scores, and aggregated task-level behavior statistics.

Training. SCOPE-Router maps each query representation and each calibration-derived model profile into a shared 64-dimensional routing space. The default QueryMLP and ProfileMLP use 128-dimensional hidden layers with dropout 0.5. The router uses a dot-product scorer and cost-aware supervision with λ = 10. We optimize with AdamW using a fixed learning rate of 0.001, weight decay 0.003, and batch size 512. Training runs for at most 100 epochs with early stopping patience 20, selecting checkpoints by development Rank Score. In the main setting, the query encoder is frozen, so training updates the QueryMLP and ProfileMLP. We use this configuration for the main experiments unless an ablation explicitly changes the hidden dimension, calibration strategy, or encoder update policy.

Open-set evaluation. For open-set model experiments, held-out VLMs are removed from router training and are also excluded from calibration-set selection. They are allowed to join the candidate pool only after behavior profiles are constructed on the fixed calibration set. This models the practical deployment setting where a newly released VLM is profiled once on the calibration set before serving traffic. In strict unseen-pool evaluation, the test-time candidate set contains only held-out VLMs, so the router must choose among models whose training labels were never observed.

## D Evaluation Metrics

Accuracy and cost. Average accuracy is the fraction of routed samples answered correctly by the selected VLM. Average cost is computed from the token-level execution cost of the selected VLM and reported in dollars per 10K samples in the result tables.

Rank Score. Following VL-RouterBench, Rank Score harmonically averages accuracy and log-normalized cost. Given the full-sample cost range $[ c _ { \mathrm { m i n } } , c _ { \mathrm { m a x } } ]$ and router average cost ${ \bar { C } } ,$ the normalized cost score is

$$
C _ { \mathrm { n o r m } } = 1 0 0 \times \frac { \log _ { 2 } ( c _ { \mathrm { m a x } } ) - \log _ { 2 } ( \bar { C } ) } { \log _ { 2 } ( c _ { \mathrm { m a x } } ) - \log _ { 2 } ( c _ { \mathrm { m i n } } ) } .\tag{13}
$$

The factor of 100 aligns the cost term with the percentage scale of average accuracy. We then combine average accuracy A<sup>¯</sup> and normalized cost using the weighted harmonic mean:

$$
S ( \beta ) = \frac { ( 1 + \beta ) \cdot \bar { A } \cdot C _ { \mathrm { { n o r m } } } } { \beta \cdot \bar { A } + C _ { \mathrm { { n o r m } } } } ,\tag{14}
$$

where $\beta > 0$ controls the preference between quality and cost. We set $\beta = 0 . 1$ by default to prioritize accuracy. Oracle selects the cheapest correct model per sample; Strongest always selects the globally most accurate model; Cheapest always selects the globally cheapest model.

## E Additional Results

Tables 1–5 and Figures 1–4 provide the appendix results referenced in the main paper and the additional model-level diagnostics. The additional results support four main observations.

First, Table 1 shows that SCOPE-Router remains robust on VLM-ExecRouterBench under the same OOD dataset split and evaluation protocol used in the main paper, where held-out datasets are used only for testing. Although several feature-level routers can obtain high accuracy by selecting expensive models, SCOPE-Router achieves the best Rank Score among learned routers by preserving high accuracy while avoiding the cost inflation of OVR and K-means. This is the intended behavior of profile-based open-set routing: the router uses model profiles to identify useful specialists without collapsing to the strongest or most expensive model.

Second, Tables 2–4 indicate that the method is not highly sensitive to oversized routing networks, but it does depend on the quality of the calibration set. A hidden dimension of 128 gives the best overall tradeoff in our VL-RouterBench ablation, while larger hidden dimensions do not yield consistent gains. Freezing the query embedder performs on par with fine-tuning, suggesting that most of the routing signal comes from aligning frozen query embeddings with model behavior profiles rather than from adapting the encoder itself. For calibration, the full hybrid strategy combining random, diagnostic, and diversity sampling outperforms each single-source or two-source variant, supporting the use of a calibration set that covers both representative queries and discriminative model-failure cases.

Third, Table 5 shows that router behavior varies substantially across execution domains. Document, chart, OCR, code, and search tasks favor different model choices and therefore expose different failure modes for learned routers. SCOPE-Router is especially useful in this setting because it routes using both query features and calibration-derived model profiles, rather than learning a closed-set classifier over a fixed model identity space. This helps explain why methods that are competitive on one dataset can degrade on another when the cost–accuracy frontier changes.

Table 1. OOD generalization results with all 11 candidate models available. Rank is determined by Rank Score.
<table><tr><td>Router</td><td>Avg. Acc. ↑ Avg. Cost ↓ Rank Score ↑ Rank ↓</td><td></td><td></td><td></td></tr><tr><td colspan="5">Training-free Baselines</td></tr><tr><td>Oracle</td><td>96.87</td><td>7.68</td><td>97.15</td><td>0</td></tr><tr><td>Strongest</td><td>89.45</td><td>224.98</td><td>75.29</td><td>15</td></tr><tr><td>Cheapest</td><td>75.86</td><td>2.82</td><td>77.56</td><td>14</td></tr><tr><td colspan="5">End-to-End Routers</td></tr><tr><td>CosineCls</td><td>78.13</td><td>5.07</td><td>79.72</td><td>9</td></tr><tr><td>RouterDC</td><td>78.53</td><td>5.23</td><td>80.10</td><td>7</td></tr><tr><td>ZOOTER</td><td>77.42</td><td>5.32</td><td>79.04</td><td>11</td></tr><tr><td>VLC</td><td>77.31</td><td>9.24</td><td>78.94</td><td>12</td></tr><tr><td colspan="5">Feature-level Routers</td></tr><tr><td>KNN</td><td>76.63</td><td>4.34</td><td>78.29</td><td>13</td></tr><tr><td>PRkNN</td><td>81.31</td><td>6.06</td><td>82.71</td><td>4</td></tr><tr><td>OVR</td><td>87.37</td><td>76.06</td><td>83.93</td><td>3</td></tr><tr><td>K-means</td><td>88.74</td><td>58.67</td><td>86.30</td><td>2</td></tr><tr><td>Linear</td><td>82.37</td><td>58.97</td><td>80.76</td><td>5</td></tr><tr><td>MLP</td><td>78.31</td><td>5.09</td><td>79.88</td><td>8</td></tr><tr><td>UniRoute-KM</td><td>78.07</td><td>13.09</td><td>79.66</td><td>10</td></tr><tr><td>UniRoute-LM</td><td>78.88</td><td>12.04</td><td>80.42</td><td>6</td></tr><tr><td>SCOPE-Router (Ours)</td><td>87.10</td><td>12.79</td><td>88.14</td><td>1</td></tr></table>

Table 2. Effect of MLP hidden dimension on VL-RouterBench.
<table><tr><td>Hidden Dim</td><td>Avg. Acc. ↑ Avg. Cost↓</td><td></td><td>Rank Score ↑</td></tr><tr><td>64</td><td>77.37</td><td>0.77</td><td>75.56</td></tr><tr><td>128</td><td>79.11</td><td>0.98</td><td>76.18</td></tr><tr><td>256</td><td>78.23</td><td>0.82</td><td>76.08</td></tr><tr><td>512</td><td>78.27</td><td>0.87</td><td>75.90</td></tr><tr><td>1024</td><td>78.05</td><td>0.82</td><td>75.91</td></tr><tr><td>2048</td><td>78.52</td><td>0.88</td><td>76.08</td></tr><tr><td>4096</td><td>78.68</td><td>0.97</td><td>75.83</td></tr></table>

Table 3. Effect of freezing or fine-tuning the query embedder on VL-RouterBench and VLM-ExecRouterBench. The base setting freezes the query embedder and only trains QueryMLP.
<table><tr><td rowspan="2">Variant</td><td colspan="3">VL-RB</td><td colspan="3">VLM-ExecRB</td></tr><tr><td>|Acc. ↑</td><td>Cost↓</td><td>Score ↑ | Acc. ↑</td><td></td><td>Cost↓</td><td>Score ↑</td></tr><tr><td>Frozen</td><td>79.11</td><td>0.98</td><td>76.18</td><td>81.65</td><td>46.20</td><td>80.94</td></tr><tr><td>Fine-tuned</td><td>78.36</td><td>0.94</td><td>75.68</td><td>81.76</td><td>44.85</td><td>81.14</td></tr></table>

Table 4. Calibration-strategy ablation on VL-RouterBench. R, D, and Div denote Random, Diagnostic, and Diversity sampling, respectively.
<table><tr><td>Variant</td><td>Avg. Acc. ↑ Avg. Cost ↓ Rank Score ↑</td></tr><tr><td></td><td>0.90 75.31</td></tr><tr><td>R only 77.73 D only</td><td>77.34 0.80</td></tr><tr><td></td><td>75.37 0.99 75.69</td></tr><tr><td>Div only 78.59 R+D</td><td></td></tr><tr><td>78.87 0.96</td><td>75.78</td></tr><tr><td>R+Div 78.44</td><td>0.88 76.00 0.90 75.87</td></tr><tr><td>D+Div 78.37 R+D+Div (Full) 79.11</td><td>0.98 76.18</td></tr></table>

Finally, Figure 1, Figure 2, Figure 3, and Figure 4 provide diagnostics for the representation and candidatemodel pool, while Figure 5 complements the earlier data-generation description with a visual overview of the three execution domains. The encoder ablation shows that routing quality depends on the joint text–vision representation. The per-dataset heatmap further shows that candidate VLMs are not uniformly strong across datasets. The radar plot summarizes the same heterogeneity at the level of task families, while the accuracy–cost plot shows that global model strength and inference cost are not aligned. These visualizations motivate the benchmark design: open-set routing is meaningful precisely because model strengths are heterogeneous across task families, and cost-efficient deployment requires selecting among those strengths rather than ranking models by a single global accuracy number.

## F Limitations

SCOPE-Router extends naturally to newly added VLMs, but each new model must still be profiled on a labeled calibration set before it can be routed reliably. As a result, open-set performance depends on how well the calibration examples expose the capabilities and failure modes of the incoming model. In addition, the current router makes a single task-level decision and uses the selected model for the entire execution. This matches the benchmark setting, where each sample is represented by one correctness and cost entry per candidate VLM, but it does not adapt to changing requirements within long-horizon Agentic or Search trajectories. For instance, an early step may require fine-grained OCR or visual localization, whereas a later step may require retrieval synthesis or answer normalization. Future work will study more robust model profiling from compact calibration sets and dynamic routing policies that can revise model choices across intermediate execution steps.

![](images/3bf342efbc33284ecd5198da9be5815d2837e056e14fce136ba29061cd73f6e3.jpg)  
Figure 1. Text and vision encoder selection across 25 encoder combinations. Rank Scores span less than 0.8 points, demonstrating robustness to encoder choice.

Model Performance Across Datasets  
![](images/1fb7bc97f92f913e27b59383e78bbb934e9fc377fd3a6df5e7b0997e473de7a7.jpg)  
Figure 2. Per-dataset accuracy of each candidate model on VLM-ExecRouterBench. The heatmap exposes complementary model strengths across code, multimodal visual reasoning, and search-style datasets.

Table 5. Performance comparison of router methods on VLM-ExecRouterBench across datasets. Avg. Acc. is average accuracy (%), Avg. Cost is average cost (\$/10K samples). The best and second-best results among learned routers are highlighted in bold and underlined, respectively, excluding Oracle and baseline methods.
<table><tr><td colspan="3" rowspan="2">Group Dataset</td><td rowspan="2">Metric</td><td colspan="3">Baselines</td><td colspan="4">Hard Label</td><td colspan="9">Soft Label (λ = 10)</td></tr><tr><td colspan="3">Oracle Strong. Cheap.|</td><td colspan="2">KNN PRkNN</td><td colspan="2">OVR K-means|</td><td colspan="2">Linear MLP CosCls</td><td colspan="2">RDC</td><td colspan="2">VLC Zoot.</td><td>U-KM U-LM SCOPE</td><td></td><td></td></tr><tr><td rowspan="8">Aentic</td><td rowspan="2">DocVQA</td><td>Avg. Acc. ↑ Avg. Cost ↓</td><td>99.53 3.77</td><td>96.22 246.03</td><td>92.62 3.01</td><td>93.00 3.49</td><td>94.13 5.13</td><td>97.54 86.96</td><td></td><td>93.66| 94.42 38.03 57.08</td><td>93.19</td><td></td><td>94.04 4.31</td><td>93.66</td><td>93.09 3.67</td><td>92.81</td><td>94.32 95.84</td><td>97.92</td></tr><tr><td>Avg. Acc. ↑</td><td></td><td>97.00 88.00</td><td>80.50|</td><td>85.50</td><td>84.00</td><td>89.50</td><td>87.00|</td><td>85.00</td><td>3.40 89.50</td><td>91.00</td><td>4.82 88.00</td><td>88.00</td><td>3.15 86.00</td><td>26.77 81.00</td><td>39.99 84.50</td><td>14.31 82.00</td></tr><tr><td>OCRBench ChartQA</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>95.63</td><td>5.78 221.49 89.51</td><td>2.45 76.07|</td><td>2.84 78.58</td><td>3.30 84.70</td><td>81.00 88.52</td><td>26.93 87.65|</td><td>79.33 80.11</td><td>4.25 84.37</td><td>3.59 87.87</td><td>3.69 87.32</td><td>3.43 86.89</td><td>3.04 83.83</td><td>7.21 80.11</td><td>10.65 83.06</td><td>5.48 87.87</td></tr><tr><td>AI2D</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>15.73 99.84</td><td>269.70 93.74</td><td>2.43 77.37|</td><td>3.11 80.42</td><td>3.73 82.99</td><td>65.47 91.81</td><td>25.18 93.26|</td><td>32.35 89.57</td><td>3.81 82.18</td><td>85.39</td><td>4.68 85.07</td><td>4.75 85.07</td><td>4.59 3.26 84.11</td><td>4.25 82.66</td><td>11.01 82.99</td><td>5.57 85.07</td></tr><tr><td>MathVista</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>7.19 97.52</td><td>165.90 88.61</td><td>6.60 63.86|</td><td>7.42 73.27</td><td>6.75 82.18</td><td>101.97 90.10</td><td>93.96 93.07</td><td>68.15 83.17</td><td>7.91 82.18</td><td>81.68</td><td>8.53 81.68</td><td>8.56 81.68</td><td>8.56 7.73 80.20</td><td>11.48 77.72</td><td>8.54 82.18</td><td>8.56 82.18</td></tr><tr><td>MMMU</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>14.22 96.06</td><td>266.49 86.71</td><td>16.69 60.91|</td><td>18.66 63.41</td><td>12.77 67.72</td><td>170.23 82.40</td><td>159.04 83.60|</td><td>108.23 75.41</td><td>16.29 66.37</td><td>15.48 73.74</td><td>15.48 74.34</td><td>15.56 73.78</td><td>15.14 72.77</td><td>16.71 70.17</td><td>15.47 72.30</td><td>17.59 74.43</td></tr><tr><td>RealWorldQA</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>21.06 98.59</td><td>335.52 79.58</td><td>19.39 59.15|</td><td>19.69 58.45</td><td>27.38 61.27</td><td>211.32 72.54</td><td>196.39 76.06</td><td>119.19 79.58</td><td>18.91 64.08</td><td>35.15 70.42</td><td>34.70 71.83</td><td>33.43</td><td>28.96</td><td>32.26</td><td>34.44</td><td>38.10</td></tr><tr><td>Agentic Avg.</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>9.74 97.28</td><td>400.87 89.85</td><td>3.16 72.59|</td><td>5.05 75.01</td><td>5.40 78.71</td><td>178.59 87.88</td><td>221.78 87.73</td><td>352.91 82.45</td><td>6.90 78.10</td><td>8.73 82.47</td><td>8.62 82.43</td><td>61.27 81.73</td><td>61.97 6.96 3.89 80.50</td><td>71.13 13.24 78.90</td><td>69.01 12.99 80.86</td><td>71.13 13.24 83.18</td></tr><tr><td rowspan="6">Code</td><td></td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>13.92 94.37</td><td>281.17 92.96</td><td>10.51 70.42</td><td>11.08 71.83</td><td>14.38 80.28</td><td>141.10 91.55</td><td>116.04 92.96</td><td>90.14 87.32</td><td>10.94 84.51</td><td>17.95 84.51</td><td>17.89</td><td>17.06</td><td>14.70</td><td>21.84 85.92</td><td>26.26 87.32</td><td>21.58 90.14</td></tr><tr><td>LiveCodeBench</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>5.49 99.51</td><td>53.90 92.68</td><td>5.33 74.63|</td><td>7.39 76.10</td><td>12.47 75.61</td><td>72.86</td><td>54.76</td><td>35.39</td><td>12.61 82.93</td><td>12.89 74.63</td><td>85.92 12.96</td><td>70.42 4.77</td><td>70.42 5.33</td><td>12.96</td><td>15.80 86.83</td><td>30.76 87.32</td></tr><tr><td>MBPP</td><td>Avg. Cost ↓ Avg. Acc. ↑</td><td>6.10 89.04</td><td>55.82 82.40</td><td>2.86 49.30</td><td>5.33 53.15</td><td>5.63 61.66</td><td>96.10 192.55 77.16</td><td>93.17 59.55</td><td>86.83 20.01 73.43</td><td>14.10 68.30</td><td>2.86 70.28</td><td>76.59 8.46</td><td>74.63 2.58</td><td>79.51 5.03</td><td>84.88 15.77 70.51</td><td>16.60 70.05</td><td>31.99</td></tr><tr><td>APPS</td><td>Avg. Cost ↓</td><td>41.31 74.35</td><td>122.09</td><td>28.09</td><td>40.35</td><td>50.15</td><td>174.52</td><td>81.93 141.40</td><td>78.41</td><td>54.52</td><td>54.82</td><td>69.23 54.22</td><td>49.42 30.90</td><td>49.42 27.08</td><td>54.91</td><td>58.34</td><td>79.84 106.83</td></tr><tr><td>BigCodeBench</td><td>Avg. Acc. ↑ Avg. Cost ↓</td><td>12.21</td><td>53.48 40.93</td><td>47.83| 1.61</td><td>45.65 4.73</td><td>47.39 3.41</td><td>55.65 93.10</td><td>55.65 59.73</td><td>47.83 13.95</td><td>49.57 3.79</td><td>46.96 3.79</td><td>50.00 3.36</td><td>47.39 11.50</td><td>28.26 2.40</td><td>54.35 26.78</td><td>53.91 26.48</td><td>53.04 36.57</td></tr><tr><td>Code Avg.</td><td>Avg. Acc. ↑ Avg. Cost ↓</td><td>88.42 29.25</td><td>79.62 94.90</td><td>53.96 18.65</td><td>56.30 27.36</td><td>62.32 33.62</td><td>77.13 158.21</td><td>79.77 110.82</td><td>71.85 56.52</td><td>68.18 37.71</td><td>67.74 36.22</td><td>67.96 36.62</td><td>53.96 22.01</td><td>51.47 18.47</td><td>70.75 42.10</td><td>70.75 44.48</td><td>76.98 79.78</td></tr><tr><td>Aaer rch</td><td>BrowseComp+</td><td>Avg. Acc. ↑ Avg. Cost ↓</td><td>84.09 2203.19 3599.38</td><td>41.67</td><td>15.15| 359.94</td><td>52.27 1379.06</td><td>62.12</td><td>64.39 1171.33 13658.61</td><td>65.15| 12839.69</td><td>58.33 1004.43 635.84</td><td>65.91</td><td>65.15</td><td>61.36 639.89 892.86 679.74 637.13</td><td>65.15</td><td>62.12</td><td>43.18 1167.60 678.14</td><td>66.67</td><td>68.18 687.35</td></tr><tr><td></td><td>All Datasets</td><td>Avg. Acc. ↑ Avg. Cost ↓ Rank Score ↑</td><td>95.25 59.53 91.77</td><td>86.86 308.24 66.75</td><td>67.74| 18.93 69.78</td><td>70.81 40.93 71.41</td><td>75.10 40.72 75.36</td><td>85.27 407.16 55.14</td><td>85.69| 362.20 60.73</td><td>79.85 101.16 76.15</td><td>75.88 28.45 76.87</td><td>79.17 33.71 79.55</td><td>79.11 38.65 79.16</td><td>75.83 30.93 76.66</td><td>74.32 27.55 75.47</td><td>76.57 48.16 76.26</td><td>78.55 42.58 78.40</td><td>81.65 46.20 80.94</td></tr></table>

![](images/3da48d781340612d2b55b5f4aaf6880e34dc6c623ac93d0f99ae284b29b1a3c0.jpg)  
Figure 3. Model strength profiles aggregated by dataset group. The radar plot highlights that candidate VLMs specialize differently across code, multimodal visual reasoning, and search-style tasks.

![](images/6ad241abca0a6ff7775a8250c5de1e69fd20866062134517e3bacb4a9beafba8.jpg)  
Figure 4. Accuracy–cost distribution of individual candidate VLMs and the oracle reference computed from the final routing matrix. The plot illustrates that higher global accuracy often comes with substantially higher execution cost, motivating cost-aware routing.

![](images/a90acc71166bb6beff82ee626828d6fc419882ca26b85410a3af4ba814cd4b99.jpg)  
Figure 5. Overview of the VLM-ExecRouterBench data-generation pipeline. Heterogeneous Code, Agentic, and Search source tasks are standardized into executor inputs with the appropriate modalities and optional tool schemas. Candidate VLMs produce responses through the shared executor: Code tasks use one-shot code generation without environment interaction, while Agentic and Search tasks may produce saved tool trajectories. Task-specific verifiers then convert each execution into correctness and cost entries. The resulting JSONL records and aligned correctness/cost matrices form the released router dataset.