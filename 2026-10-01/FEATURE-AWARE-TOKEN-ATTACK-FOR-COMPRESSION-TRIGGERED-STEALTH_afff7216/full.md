# FEATURE-AWARE TOKEN ATTACK FOR COMPRESSION-TRIGGERED STEALTHY FAILURES IN LARGE VISION-LANGUAGE MODELS

Shilinlu Yan1 Bowen Chen1 Yuechen Zhang2 Zhenhong Zhou³ Li Sun1 Sen Su1,4

¹Beijing University of Posts and Telecommunications

2Jiangnan University

3Nanyang Technological University

4Chongqing University of Posts and Telecommunications

{lulu\_land, cbcbw, lsun,susen}@bupt.edu.cn orange.zhangyc05@gmail.comzhenhong001@e.ntu.edu.sg

## ABSTRACT

Visual-token compression improves the efficiency of large vision-language models, but can expose failures that full-token evaluation misses. We study adversarial images that preserve full-token correctness yet induce errors after compression, even when both inference paths succeed on the clean image. Creating such failures is challenging because perturbing token importance can also damage the visual content needed for full-token inference. We propose Feature-Aware Token Attack (FATA), which couples attention suppression with cosine-based feature preservation on a fixed set of salient clean-image tokens. In the primary LLaVA-1.5-7B setting, FATA uses only vision-encoder gradients, without access to the deployed compressor, token budget, or downstream task. Across four visually dependent task subsets and four compressors under a controlled reconstruction protocol, FATA achieves SR = 96.3% full-token accuracy retention and CBR = 22.1% conditional blinding, compared with 89.8% and 15.7% for CAA. Ablations support the role of both objectives in balancing compressed-path failure against full-token preservation. FATA also has the lowest measured detection rate among four attacks across three evaluated detectors at a 5% false-positive rate. These findings motivate assessing adversarial robustness jointly across full-token and compressed inference.

## 1 INTRODUCTION

Visual-token compression reduces the inference cost of large vision-language models (LVLMs) as image resolution and visual context grow (An et al., 2026; Song et al., 2026). Recent methods select informative tokens or merge redundant features while retaining much of the original task accuracy (Yang et al., 2025; Zhang et al., 2025b; Shang et al., 2025; Tong et al., 2026). Because these decisions determine which visual evidence remains available to the language model (Chen et al., 2024), their robustness to input perturbations is essential to reliable compressed inference.

We study a failure in which a perturbed image remains correctly understood with the full token sequence but yields an incorrect answer after compression (As shown in Figure 1). When both paths succeed on the clean image, this divergence exposes a vulnerability that full-token evaluation alone would miss. An attack targeting this vulnerability must therefore meet a dual requirement: induce compressed-path failure while preserving full-token correctness.

Prior compression-aware attacks approach this problem through feature disruption (CAGE) or tokenimportance manipulation (CAA) (Zhang et al., 2026b;a). Related attacks show that vision-encoder access alone can suffice to disrupt visual features (Wang et al., 2024; Mei et al., 2026). Selective failure poses an additional challenge: encoder attention offers a surrogate for token importance, but perturbing it can also damage the features needed for full-token inference. Conversely, preserving features alone provides little pressure to induce compressed-path failure. This coupling motivates our central question: how can an attack exploit compression sensitivity while preserving full-token behavior, using only vision-encoder access?

![](images/3d38ad5fbd9d2606b6a0a54f5c3158cc3c1acb0e2afc1b36929e84ea4a278d92.jpg)

![](images/2b028b684f406dd671ca448cdac23519229e9ae417de7b57cdba7cfc03217c7d.jpg)

![](images/52d76822761aefd8f213e4d94454e8375aca2cbee2c62615ec316bdb8a48e8a0.jpg)  
Figure 1: FATA: attention suppression with feature preservation. A illustrates the compressiontriggered failure setting; B presents the two attack objectives; C compares full-token retention and conditional blinding; D shows how answers to the same perturbed TextVQA image vary across compression paths.

In this paper, we propose Feature-Aware Token Attack (FATA), which couples two objectives on a fixed set of tokens selected by clean encoder attention. An attention-suppression objective lowers attention to these positions as a surrogate for compression sensitivity, while a feature-preservation objective maintains cosine alignment with their clean feature vectors. Sharing the target set ties the attack pressure to the content being protected. In the primary LLaVA-1.5-7B setting, FATA optimizes only the input perturbation through a frozen vision encoder, without access to the downstream language model, questions, answers, compressor, or token budget. The same perturbed image is then evaluated across compression rules and budgets without re-optimization.

Across four visually dependent task subsets and four compressors under a controlled reconstruction protocol, FATA achieves SR = 96.3% full-token accuracy retention and CBR = 22.1% conditional blinding rate, compared with 89.8% and 15.7% for CAA. Ablations show why both objectives matter: attention suppression alone substantially degrades full-token accuracy, while feature preservation alone yields weaker conditional blinding. These results support jointly controlling attention and feature distortion to expose compression sensitivity while preserving full-token behavior.

The contributions of this paper are summarized as follows.

0 Feature-aware attack design. We introduce FATA, coupling attention suppression with explicit feature preservation on fixed clean-image targets to induce compression-triggered failures under vision-encoder-only access.

Paired robustness evaluation. We evaluate the balance between full-token retention and conditional compressed-path failure across four task settings and four compressors, with loss ablations and model-specific extensions under stated access assumptions.

Detection across model stages. We introduce a Multi-Level Activation Trajectory Detector (ML-ATD) to monitor the vision encoder, projector, and language model, and compare three detection approaches at a fixed low false-positive rate.

## 2 RELATED WORK

Visual Token Compression Inputs at high resolution can produce long visual prefixes, increasing prefill latency and memory use (Chen et al., 2024; Arif et al., 2025). Visual token compression reduces these costs at the encoder output, visual projector, or decoder (Zhang et al., 2024; Li et al., 2025; Ye et al., 2025). Methods that require no training retain dominant, salient, or diverse tokens (Yang et al., 2025; Zhang et al., 2025b; Alvar et al., 2025). Other approaches merge nearby features or use early decoder attention and information flow across layers to remove redundancy (Shang et al. 2025; Chen et al., 2024; Tong et al., 2026). However, visual token compression can also increase vulnerability to adversarial perturbations that change which tokens are retained for inference (Zhang et al., 2026b;a).

Adversarial Perturbations under Compression Earlier methods disrupt encoder features, alignment between modalities, or transferable representations (Wang et al., 2024; Lu et al., 2023; Zhang et al., 2025a). Recent works use targets generated by diffusion models, encoder objectives independent of downstream tasks, or guidance from projector features (Guo et al., 2024; Mei et al., 2026; Cao et al., 2026). These approaches do not model which tokens a compressor retains. For visual token compression, CAGE targets tokens likely to survive unknown budgets (Zhang et al., 2026b). Though CAA and our work both focus on inducing failures after compression while preserving full token behavior (Zhang et al., 2026a), our method further incorporates an explicit semantic preservation loss to better achieve this goal.

Adversarial Detection Adversarial detectors distinguish adversarial inputs from clean inputs. Feature Squeezing assesses prediction consistency under input transformations, while CIDER measures changes in similarity between modalities under denoising (Xu et al., 2017; 2024). Internal-state detectors use feature distribution deviations or activations associated with safety (Lee et al., 2018; Jiang et al., 2025). JailDAM uses adaptive memory to update unsafe knowledge during inference and improve detection of unseen jailbreak strategies (Nian et al., 2025). Alongside input and feature distribution analysis, inspired by HiddenDetect, ML-ATD uses an attack reference direction estimated from Base examples and monitors activations in the visual encoder, projector, and language model.

## 3 METHOD

## 3.1 OVERVIEW

![](images/2cce3d591c2ff75c3691a5b7ef3021feade9d7225c98e6bcb89856451baf841d.jpg)  
Figure 2: FATA overview. Fixed clean targets guide input-only PGD with attention and semantic losses (A-C), followed by full-token and compressed-path evaluation of the same image (D).

Figure 2 separates attack optimization (A-C) from compression-response evaluation (D). The frozen CLIP encoder uses last-layer clean CLS-to-patch attention to fix the Top-M target tokens and stores their patch features as stop-gradient anchors (A). For each adversarial iterate, $\mathcal { L } _ { \mathrm { a t t n } }$ suppresses attention over those same positions, while $\mathcal { L } _ { \mathrm { s e m } }$ preserves their feature directions through cosine similarity to the clean anchors (B). PGD backpropagates their weighted sum through CLIP and updates only $\delta ;$ the encoder weights, target set, and anchors stay fixed, and the compressor and LLM remain outside the attack loop (C).

Panel D evaluates the same optimized image along two paths. LLaVA's vision tower shares the encoder weights but reads penultimate-layer features H, distinct from the last-layer attack features. The full-token path passes H directly to the projector. The compressed path selects or merges K representatives $Z ,$ then uses similarity lookup from H to reconstruct an N-slot sequence $\widetilde { H }$ from those representatives. Both paths use the same projector, LLM, and question. This comparison exposes compression-triggered errors despite largely preserved full-token performance. Reconstruction fills all $\hat { N }$ positions with representative features.

## 3.2 FATA

FATA is formulated as a gray-box method on LLaVA with gradient access to the visual encoder $E ,$ while the compressor, token budget, and downstream language model are unknown during optimization. Given a clean image x, FATA optimizes a perturbation δ under $\| \delta \| _ { \infty } \leq \epsilon$ to obtain $x _ { \mathrm { a d v } } = \mathrm { c l i p } ( x + \delta , 0 , 1 )$ . Let $( \mathbf { s } ^ { c } , \mathbf { H } ^ { c } ) = E ( x )$ denote the last-layer CLS attention scores over visual tokens and their corresponding features. Attention scores are averaged across heads, and the top M visual tokens form a fixed target set.

$$
\mathcal { M } = \mathrm { T o p K } ( \mathbf { s } ^ { c } , M ) , \qquad M = 6 4 .\tag{1}
$$

The visual tokens in M are referred to as target tokens. For the adversarial image $x _ { \mathrm { a d v } } .$ FATA

```latex
Algorithm 1 FATA optimization overview
Require: Clean image x, encoder $E , \epsilon , \alpha , T , \lambda ,$ target count M
1: $( \mathbf { s } ^ { c } , \mathbf { H } ^ { c } )  E ( x )$
2: M ← TopK(sc, M)
3: $\delta ^ { ( 0 ) } \sim \mathcal { U } ( - \epsilon , \epsilon )$
4: for $t = 0 , \ldots , T - 1 ,$ do
5: $\boldsymbol { x } _ { a } ^ { t } \gets \mathrm { c l i p } ( \boldsymbol { x } + \delta ^ { ( t ) } , 0 , 1 )$
6: $( \mathbf { s } ^ { a } , \mathbf { H } ^ { a } )  E ( x _ { a } ^ { t } )$
7: $\textstyle { \mathcal { L } } _ { \mathrm { a t t n } } \gets \sum _ { i \in { \mathcal { M } } } s _ { i } ^ { \alpha }$
8: $\begin{array} { r } { \mathcal { L } _ { \mathrm { s e m } }  1 - \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \cos ( \mathbf { h } _ { i } ^ { a } , \mathbf { h } _ { i } ^ { c } ) } \end{array}$
9: $\mathcal { L } _ { \mathrm { F A T A } }  \mathcal { L } _ { \mathrm { a t t n } } + \lambda \mathcal { L } _ { \mathrm { s e m } }$
10: $\delta ^ { ( t + 1 ) } \gets \Pi _ { \parallel \delta \parallel _ { \infty } \leq \epsilon } ( \delta ^ { ( t ) } -$ α sign(∇δLFATA))
11: end for
12: return $\mathrm { c l i p } ( x + \delta ^ { ( T ) } , 0 , 1 )$
```

minimizes the following attention loss over the target tokens.

$$
\mathcal { L } _ { \mathrm { a t t n } } = \sum _ { i \in \mathcal { M } } s _ { i } ^ { a } ,\tag{2}
$$

where $\mathbf { s } ^ { a }$ denotes the adversarial attention scores. Minimizing ${ \mathcal { L } } _ { \mathrm { a t t n } }$ suppresses attention to the target tokens, with the aim of reducing their selection priority during visual token compression.

To preserve full token behavior, FATA aligns the feature directions of the adversarial target tokens with their clean features. Let $\mathbf { h } _ { i } ^ { c }$ and $\mathbf { h } _ { i } ^ { a }$ denote the clean and adversarial features of target token ¿. The semantic loss is

$$
\mathcal { L } _ { \mathrm { s e m } } = 1 - \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { i \in \mathcal { M } } \cos ( \mathbf { h } _ { i } ^ { a } , \mathbf { h } _ { i } ^ { c } ) .\tag{3}
$$

Both losses use the same fixed target set. The final objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F A T A } } = \mathcal { L } _ { \mathrm { a t t n } } + \lambda \mathcal { L } _ { \mathrm { s e m } } . } \end{array}\tag{4}
$$

Algorithm 1 applies $\ell _ { \infty } \mathrm { P G D }$ to $\mathcal { L } _ { \mathrm { F A T A } }$ with $\delta ^ { ( 0 ) } \sim \mathcal { U } ( - \epsilon , \epsilon ) , \lambda = 1 , \epsilon = 2 / 2 5 5 , \alpha = 0 . 5 / 2 5 5 ,$ and $T = 1 0 0$ steps. Here, $\Pi _ { \| \delta \| _ { \infty } \leq \epsilon }$ projects the perturbation onto the $\ell _ { \infty }$ ball of radius €. Optimization details and sensitivity to α and λ are provided in Appendix A.

## 3.3 EVALUATION METRICS

Full token behavior and compression induced failure are evaluated with SR, CER, and CBR. Let $\operatorname { A c c } _ { m \ @ K }$ denote the task accuracy of method m at token budget $K ,$ and let $\operatorname { A c c } _ { m } \ @ \operatorname { f u l l }$ denote its accuracy under full token inference. SR measures full token performance retention and is then defined as

$$
\mathrm { S R } = \frac { \mathrm { A c c } _ { \mathrm { F A T A @ f u l l } } } { \mathrm { A c c } _ { \mathrm { C l e a n @ f u l l } } } .\tag{5}
$$

For LLaVA, the practical token budget $K _ { \mathrm { p r a c } }$ is selected separately for each dataset-compressor pair using clean accuracy and is fixed across attacks. Qwen3.5 and InternVL3.5 use predefined task-specific retention fractions, as detailed in Appendix B.

$$
K _ { \mathrm { p r a c } } = \operatorname* { m i n } \left\{ K : \frac { \mathrm { A c c } _ { \mathrm { C l e a n @ } K } } { \mathrm { A c c } _ { \mathrm { C l e a n @ f u l l } } } \geq 0 . 8 \right\} .\tag{6}
$$

For attack $m ,$ the compressed error rate (CER) measures overall task error at $K _ { \mathrm { p r a c } }$

$$
\mathrm { C E R } _ { m } = 1 - \mathrm { A c c } _ { m  @ K _ { \mathrm { p r a c } } } .\tag{7}
$$

It uses task-credit accuracy and does not subtract the error on Clean inputs. CBR measures the fraction of examples on which attack m becomes incorrect at the practical budget, conditioned on both Clean and attack m being correct at full tokens and Clean remaining correct under the same compressor and practical budget. Binary correctness follows the task-specific rules in Appendix C For FATA, $m = \mathrm { F A T A }$ 9

$$
\mathrm { C B R } = \frac { \left| \left\{ x : \mathrm { C l e a n @ f u l l ~ c o r r e c t ~ } \wedge \ m @ \mathrm { f u l l ~ c o r r e c t } \wedge \right. } \\ { \left. \mathrm { C l e a n @ { K _ { \mathrm { p r a c } } } ~ c o r r e c t } \wedge \ m @ K _ { \mathrm { p r a c } } \mathrm { i n c o r r e c t } \right\} \right| } { \left| \left\{ x : \mathrm { C l e a n @ f u l l ~ c o r r e c t } \wedge \ m @ \mathrm { f u l l ~ c o r r e c t } \wedge \right\} \right| } .\tag{8}
$$

Scoring rules and metric aggregation across the three models are detailed in Appendix C

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

The main evaluation uses LLaVA-1.5-7B (Liu et al., 2024) with four visual token compressors, VisionZIP, VisPruner, PruMerge, and FlowCut (Yang et al., 2025; Zhang et al., 2025b; Shang et al. 2025; Tong et al., 2026). The evaluation covers four 1,000-image subsets, TextVQA-Open, VQAv2- Open, ScienceQA-MC, and VQAv2-MC (Singh et al., 2019; Goyal et al., 2017; Lu et al., 2022). Before inference, one representative question is selected for each unique image. The subset is then constructed by retaining examples that are answered correctly with the original image but incorrectly when the image is replaced by a black image. Further details of subset construction are provided in Appendix D. The retained token budgets are K ∈ {576, 192, 128, 64, 32, 16}, and FATA is compared with Base(attention-only), CAA (Zhang et al., 2026a), and CAGE (Zhang et al., 2026b) under the common perturbation bound $\epsilon = 2 / 2 5 5$

For each dataset and compressor, clean and adversarial inputs are evaluated on the same valid examples at every retained token budget using the metrics defined in Section 3.3. $K _ { \mathrm { p r a c } }$ is selected from clean results only and then fixed for attack evaluation, as defined in Eq. 6.

## 4.2 LLAVA-1.5-7B RESULTS

## 4.2.1 MAIN RESULTS

Table 1 reports results on LLaVA-1.5-7B (Liu et al., 2024) and shows that FATA preserves most full token performance while causing larger degradation under visual token compression.

At K = 576, FATA achieves an average accuracy of 89.7%, close to the 93.2% Clean average, while Base and CAGE fall to 61.6% and 34.4%, respectively. This result indicates that FATA preserves full token behavior before compression is applied.

At matched compression budgets, the accuracy difference between Clean and FATA increases from 3.5 points at K = 576 to 5.5, 7.4, 9.3, 12.5, and 15.6 points as the retained token budget decreases to 192, 128, 64, 32, and 16, respectively. At $K = 1 6$ , TextVQA-Open exhibits the largest dataset level difference, at 23.8 points. The widening gap shows that FATA induces larger accuracy degradation after visual token compression

Table 1: Accuracy (%) across four benchmarks and visual-token budgets.
<table><tr><td rowspan="2">Method</td><td colspan="4">TextVQA-Open</td><td rowspan="2"></td><td colspan="4">VQAv2-Open</td><td colspan="4">ScienceQA-MC</td><td colspan="4">VQAv2-MC</td></tr><tr><td>Clean Base CAA CÂGE FATA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Clean Base CAA CAGE FATA Clean Base CAA CAGE FATA Clean Base CAA CAGE FATA</td><td></td><td></td><td></td></tr><tr><td></td><td colspan="10"></td><td colspan="7"></td></tr><tr><td>None</td><td>92.0 39.9 77.4 19.8</td><td></td><td></td><td>86.6</td><td></td><td>88.2 66.1 78.9 28.3</td><td></td><td>86.0</td><td>94.5 57.8 89.2 40.6 90.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>98.0 82.5 89.6 48.7 96.2</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Retain 192 Tokens (Kmodel = 192)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZIP</td><td>89.1</td><td>35.7 82.5</td><td>16.2</td><td>83.3</td><td>86.6</td><td>61.8 56.2</td><td>25.8</td><td>84.2</td><td></td><td>91.3 54.5</td><td>86.3</td><td>38.2</td><td>86.3</td><td>96.9</td><td>79.1 94.8</td><td>45.8</td><td>95.1</td></tr><tr><td>VisPruner</td><td>90.6</td><td>36.1 83.9</td><td>17.9</td><td>83.3</td><td>87.4</td><td>65.1 84.9</td><td>26.6</td><td>84.3</td><td>91.1</td><td>56.5</td><td>87.1</td><td>40.2</td><td>88.7</td><td>96.8 79.8</td><td>95.1</td><td>45.0</td><td>95.0</td></tr><tr><td>FlowCut</td><td>88.1</td><td>38.0 81.9</td><td>18.5</td><td>82.9</td><td>86.2</td><td>63.2 84.6</td><td>24.1</td><td>83.3</td><td>87.7</td><td>56.5</td><td>86.1</td><td>39.3</td><td>87.0</td><td>96.0 77.1</td><td>94.6</td><td>45.5</td><td>94.1</td></tr><tr><td>PruMerge</td><td>84.5</td><td>29.8 77.7</td><td>16.3</td><td>60.4</td><td>84.1</td><td>58.5 83.3</td><td>25.5</td><td>79.1</td><td>89.4</td><td>50.3</td><td>86.9</td><td>38.1</td><td>74.9</td><td>94.8 75.3</td><td>65.3</td><td>42.8</td><td>90.7</td></tr><tr><td>Average</td><td>88.1</td><td>34.9 81.5</td><td>17.2</td><td>77.5</td><td>86.1</td><td>62.2 77.3</td><td>25.5</td><td>82.7</td><td>89.9</td><td>54.5</td><td>86.6</td><td>39.0</td><td>84.2</td><td>96.1 77.8</td><td>87.5</td><td>44.8</td><td>93.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Retain 128 Tokens (Kmodel</td><td></td><td></td><td></td><td>128)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZIP</td><td>86.7</td><td>32.3 81.6</td><td>13.5</td><td>80.2</td><td>86.1</td><td>58.5 58.4</td><td>22.6</td><td>82.7</td><td>88.9</td><td>53.5</td><td>85.5</td><td>37.1</td><td>85.3</td><td>95.5</td><td>76.3 93.3</td><td>43.0</td><td>92.9</td></tr><tr><td>VisPruner</td><td>88.2</td><td>34.8 81.8</td><td>15.9</td><td>81.3</td><td>86.0</td><td>62.3</td><td>84.2 23.4</td><td>82.9</td><td>89.4</td><td>56.3</td><td>85.7</td><td>39.3</td><td>87.1</td><td>95.5</td><td>76.3</td><td>94.7 42.9</td><td>93.7</td></tr><tr><td>FlowCut</td><td>84.9</td><td>32.8 79.4</td><td>15.5</td><td>78.4</td><td>84.0</td><td>58.5</td><td>82.3 19.6</td><td>80.7</td><td>85.6</td><td>56.6</td><td>83.6</td><td>39.5</td><td>84.6</td><td>94.0</td><td>72.6 92.7</td><td>39.7</td><td>90.5</td></tr><tr><td>PruMerge</td><td>81.8</td><td>23.9 76.6</td><td>12.9</td><td>46.0</td><td>80.6</td><td>54.4 79.9</td><td>23.5</td><td>69.8</td><td>88.5</td><td>47.6</td><td>85.9</td><td>36.2</td><td>71.5</td><td>92.6</td><td>71.2 67.6</td><td>41.1</td><td>83.0</td></tr><tr><td>Average</td><td>85.4</td><td>30.9 79.9</td><td>14.5</td><td>71.5</td><td>84.2</td><td>58.4 76.2</td><td></td><td>79.0</td><td>88.1</td><td></td><td></td><td></td><td></td><td></td><td>87.1</td><td>41.7</td><td></td></tr><tr><td>Retain 64 Tokens (Kmodel</td><td></td><td></td><td></td><td></td><td></td><td></td><td>22.3</td><td></td><td></td><td>53.5</td><td>85.2</td><td>38.0</td><td>82.1</td><td>94.4</td><td>74.1</td><td></td><td>90.0</td></tr><tr><td>VisionZIP|</td><td>81.2</td><td>25.8 75.7</td><td>11.4</td><td>70.9</td><td>80.3</td><td>51.4 58.8</td><td>20.1</td><td></td><td></td><td>64) 52.0</td><td>82.8</td><td>35.6</td><td>83.1</td><td>90.7</td><td>68.4 87.7</td><td></td><td>86.3</td></tr><tr><td>VisPruner</td><td>82.6</td><td>29.7 75.6</td><td>13.1</td><td>74.6</td><td>80.3</td><td>55.8</td><td>78.6 20.9</td><td>77.3 75.3</td><td>85.3 85.9</td><td>53.4</td><td>83.8</td><td>39.7</td><td>84.1</td><td>92.2</td><td>71.4 89.8</td><td>37.0 35.6</td><td>88.7</td></tr><tr><td>FlowCut</td><td>76.8</td><td>24.6 72.4</td><td>11.2</td><td>66.8</td><td>74.1</td><td>47.7</td><td>72.6</td><td>69.7</td><td></td><td>51.8</td><td>80.6</td><td>37.1</td><td>76.7</td><td>85.5</td><td>60.8</td><td>85.9</td><td>79.8</td></tr><tr><td>PruMerge</td><td>77.7</td><td>20.0 73.6</td><td>10.4</td><td>38.2</td><td>77.8</td><td>47.7</td><td>14.5 20.5</td><td>62.5</td><td>82.4 84.8</td><td>46.4</td><td>85.5</td><td>34.7</td><td>66.4</td><td>90.6</td><td>65.7 66.4</td><td>31.0 38.0</td><td>78.5</td></tr><tr><td>Average</td><td>79.6</td><td>25.0 74.3</td><td>11.5</td><td>62.6</td><td>78.1</td><td>75.8</td><td></td><td>71.2</td><td></td><td></td><td></td><td></td><td>77.6</td><td>89.8</td><td></td><td>35.4</td><td></td></tr><tr><td>Retain 32 Tokens (Kmodel</td><td></td><td></td><td></td><td></td><td></td><td>50.6 71.4</td><td>19.0</td><td></td><td>84.6</td><td>50.9 二</td><td>83.2</td><td>36.8</td><td></td><td></td><td>66.6</td><td>82.4</td><td>83.3</td></tr><tr><td>67.6 37.4 51.5</td><td></td><td>17.0 64.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>32)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZIP| VisPruner</td><td>69.0</td><td>22.9 68.2</td><td>8.8 10.0</td><td>54.7 60.1</td><td>70.5</td><td>44.0 68.3</td><td>15.1 16.7</td><td>61.2 64.6</td><td>81.6 79.3</td><td>47.2 51.0</td><td>79.9 75.6</td><td>32.2 39.8</td><td>75.2 74.1</td><td>82.3 84.7</td><td>54.9 79.8 58.4 83.2</td><td>32.3 32.4</td><td>74.9 79.0</td></tr><tr><td>FlowCut</td><td>72.7 63.0 16.7</td></table>

Figure 3 shows when failures occur across attacks using the same examples answered correctly on clean inputs at full tokens. Among the evaluated methods, FATA has the fewest failures before compression (5.3%) and the most emerging after compression (23.3%), while 71.5% of cases remain correct after compression. CAA also limits failures before compression (8.8%) but has fewer failures emerging after compression (18.7%). Compared with FATA, Base and CAGE have more failures before compression, at

![](images/1c57975a971b34c9b1e6df8599058d4530e71d582af98e7de6537420016a4f5b.jpg)  
Figure 3: Failure-stage decomposition over 14,316 shared Clean-full-correct cases.

35.6% and 64.5%, respectively. These percentages share the Clean-full-correct denominator. CBR conditions on Clean full-token, attacked full-token, and Clean compressed correctness, then measures the attack error rate under the same compressor and practical budget.

Table 2 reports SR, CER, and CBR to characterize the balance between full token retention and failure after compression at $K _ { \mathrm { p r a c } }$ . CER is computed as $1 0 0 \% - \mathrm { A c c @ } K _ { \mathrm { p r a c } }$ from the displayed accuracies in Table 1 and budgets in Table 11, then averaged over the 16 settings and rounded to two decimal places.

Compared with Base, FATA raises SR from 66.7% to 96.3% while reducing CER from 54.78% to 30.38%

Table 2: Overall stealth-lethality trade-off
<table><tr><td>Attack</td><td>SR↑</td><td>CER↑</td><td>CBR↑</td></tr><tr><td>Base (Attn.-only)</td><td>66.7</td><td>54.78</td><td>32.9</td></tr><tr><td>CAA</td><td>89.8</td><td>24.65</td><td>15.7</td></tr><tr><td>CAGE</td><td>36.6</td><td>75.58</td><td>43.1</td></tr><tr><td>FATA</td><td>96.3</td><td>30.38</td><td>22.1</td></tr></table>

and CBR from 32.9% to 22.1%, balancing full token behavior and compression induced failure. CAGE reaches higher CER and CBR, 75.58% and 43.1%, respectively, but its SR drops to 36.6%, while CAA keeps 89.8% SR but produces a lower CBR of 15.7%. These results indicate that Base and CAGE exhibit higher CER and CBR alongside greater full-token degradation, while FATA preserves full-token performance more effectively. Compared with CAA, FATA increases SR from 89.8% to 96.3% and CBR from 15.7% to 22.1%, offering a better trade-off. Appendix E provides qualitative examples of changes in CLIP attention rankings and compression-dependent answers.

## 4.2.2 ABLATION STUDY

Table 3 separates the roles of the two losses in Eq. 4. The attention loss $\mathcal { L } _ { \mathrm { a t t n } }$ alters the visual evidence that survives compression, whereas the semantic loss $\mathcal { L } _ { \mathrm { s e m } }$ preserves the full token representation. Applying ${ \mathcal { L } } _ { \mathrm { a t t u } }$ alone gives the highest CBR of 32.9% but reduces SR to 66.7%, showing that attention disruption strengthens compression induced failure but damages full token behavior without semantic preservation. Using $\mathcal { L } _ { \mathrm { s e m } }$ alone gives the opposite effect, reaching 98.5% SR but only 14.5% CBR, which indicates that semantic preservation protects full token inference but leads to fewer compression-induced failures. The full FATA objective maintains 96.3% SR and reaches 22.1% CBR, combining the full token behavior supported by $\mathcal { L } _ { \mathrm { s e m } }$ with the compression sensitivity introduced by $\mathcal { L } _ { \mathrm { a t t n } }$ . Appendix F shows the corresponding variation across the 16 experimental settings.

Table 3: Dual-objective ablation of FATA.
<table><tr><td>Variant</td><td>Objective</td><td></td><td>SR↑ CBR↑</td></tr><tr><td>Attention-only</td><td> $\mathcal { L } _ { \mathrm { a t t } 1 }$ </td><td>66.7</td><td>32.9</td></tr><tr><td>Semantic-only</td><td> $\mathcal { L } _ { \mathrm { s e m } }$ </td><td>98.5</td><td>14.5</td></tr><tr><td>FATA Full</td><td> $\mathcal { L } _ { \mathrm { a t t n } } + \lambda \mathcal { L } _ { \mathrm { s e m } }$ </td><td>96.3</td><td>22.1</td></tr></table>

## 4.3 CROSS-MODEL EVALUATION

## 4.3.1 QWEN3.5-9B

Since the number of visual tokens in Qwen3.5(Qwen Team, 2026) depends on input resolution, compression budgets are specified as fractions of visual tokens. The evaluation uses a $1 / 3$ token budget, while the practical token budget is set to $1 / 9$ for open tasks and $1 / 1 8$ for multiple choice tasks. Model-specific adaptations are detailed in Appendix G.1.

Table 4: Qwen3.5-9B macro accuracy (%) under full, 1/3, and practical token budgets. Values are averaged over four datasets and the four compressors.
<table><tr><td>Visual-token budget</td><td>Clean</td><td>FATA</td><td>Damage (pp)</td><td>Amplification (pp)</td></tr><tr><td>Full</td><td>89.55</td><td>88.92</td><td>0.62</td><td></td></tr><tr><td>Retain  $1 / 3$ </td><td>80.66</td><td>72.47</td><td>8.19</td><td>7.56</td></tr><tr><td>Practical (Open: 1/9; MC: 1/18)</td><td>65.10</td><td>48.48</td><td>16.62</td><td>15.99</td></tr></table>

Figure 4 shows the corresponding CBR values for each dataset and compressor. At the practical token budget, FATA accuracy decreases by 40.44 percentage points, from 88.92% to 48.48%, while Clean accuracy decreases by 24.45 points, from 89.55% to 65.10%, as reported in Table 4. The larger decrease for FATA shows that compression leads to greater accuracy loss with FATA on Qwen3.5.

![](images/932d2100030fa1752acda4e7222a22464ce919bd8cae657cac29da59fb08b860.jpg)  
Figure 4: Qwen3.5 practical-budget CBR by compressor and dataset.

Table 5 reports CBR and SR on Qwen3.5-9B. SR

remains high at an average of 99.3% across the four datasets, while CBR averages 46.43% across the 16 dataset and compressor combinations. These results support the same pattern observed on LLaVA, with FATA retaining full token behavior while exhibiting failures after compression.

Table 5: Qwen3.5-9B full-token SR and practical-budget conditional CBR (%) across the four compressors.
<table><tr><td>Compression Method</td><td>TextVQA-Open SR=99.22%</td><td>VQAv2-Open SR=98.67%</td><td>ScienceQA-MC SR=100.21%</td><td>VQAv2-MC SR=98.96%</td></tr><tr><td></td><td colspan="4">CBR at Practical Budget (Open: 1/9; MC: 1/18)</td></tr><tr><td>VisionZIP</td><td>56.58</td><td>44.70</td><td>53.28</td><td>35.85</td></tr><tr><td>VisPruner</td><td>33.79</td><td>34.66</td><td>40.79</td><td>31.66</td></tr><tr><td>FlowCut</td><td>85.37</td><td>48.83</td><td>53.07</td><td>37.21</td></tr><tr><td>PruMerge</td><td>66.67</td><td>41.13</td><td>46.20</td><td>33.02</td></tr><tr><td>Average</td><td>60.60</td><td>42.33</td><td>48.34</td><td>34.43</td></tr></table>

## 4.3.2 INTERNVL3.5-8B-HF

To accommodate the dynamic tiling and token organization of InternVL3.5-8B-HF (Wang et al.   
2025), TA-FATA adapts the FATA loss while retaining attention disruption and semantic preservation.   
The complete objective, fixed hyperparameters, and evaluation details are given in Appendix G.2.

Table 6 reports the accuracy of TA-FATA across token budgets. At full tokens, the four-task average accuracy is 90.52% for Clean and 85.28% for TA-FATA, while the corresponding values at the 1/3 budget are 85.96% and 75.59%. At the practical budget, TA-FATA accuracy decreases by 32.81 percentage points from full tokens to 52.47%, while Clean accuracy decreases by 24.08 points to 66.44%. As on LLaVA and Qwen3.5, visual token compression leads to greater accuracy loss with TA-FATA than with Clean inputs.

Table 6: InternVL3.5-8B-HF accuracy (%) under full, 1/3, and practical token budgets. Compressed rows average the four retained compressors; the practical budget is 1/9 for open-ended tasks and 1/18 for multiple-choice tasks.
<table><tr><td rowspan="2">Compression Method</td><td colspan="8">InternVL3.5-8B-HF Accuracy (%)</td></tr><tr><td colspan="2">TextVQA-Open</td><td colspan="2">VQAv2-Open</td><td colspan="2">ScienceQA-MC</td><td colspan="2">VQAv2-MC</td></tr><tr><td></td><td>Clean</td><td>FATA</td><td>Clean</td><td>FATA</td><td>Clean</td><td>FATA</td><td>Clean</td><td>FATA</td></tr><tr><td colspan="9">Full Visual Tokens</td></tr><tr><td>No Compression</td><td>87.6</td><td>81.2</td><td>79.1</td><td>70.0</td><td>98.7</td><td>96.6</td><td>96.7</td><td>93.3</td></tr><tr><td colspan="9"></td></tr><tr><td>VisionZIP</td><td>81.6</td><td>64.6</td><td>Retain 1/3 of Visual Tokens 77.8</td><td>66.7</td><td>97.1</td><td>91.9</td><td>96.0</td><td>91.2</td></tr><tr><td>VisPruner</td><td>66.0</td><td>44.0</td><td>75.9</td><td>62.2</td><td>95.2</td><td>88.6</td><td>94.5</td><td>89.6</td></tr><tr><td>FlowCut</td><td>78.4</td><td>62.0</td><td>75.8</td><td>61.9</td><td>95.0</td><td>88.8</td><td>94.5</td><td>88.5</td></tr><tr><td>PruMerge</td><td>82.3</td><td>65.3</td><td>76.9</td><td>66.1</td><td>93.5</td><td>88.1</td><td>94.8</td><td>90.0</td></tr><tr><td>Average</td><td>77.1</td><td>59.0</td><td>76.6</td><td>64.2</td><td>95.2</td><td>89.4</td><td>95.0</td><td>89.8</td></tr><tr><td colspan="9"></td></tr><tr><td></td><td></td><td>31.9</td><td>Practical Budget (Open: 1/9; MC: 1/18)</td><td></td><td>68.1</td><td>57.4</td><td>84.5</td><td></td></tr><tr><td>VisionZIP VisPruner</td><td>51.8 41.6</td><td>22.8</td><td>71.1</td><td>53.1 48.2</td><td>67.9</td><td>57.5</td><td>84.6</td><td>72.8 72.7</td></tr><tr><td>FlowCut</td><td>49.2</td><td>26.8</td><td>66.2</td><td>47.6</td><td>64.5</td><td>54.8</td><td>80.0</td><td>69.4</td></tr><tr><td>PruMerge</td><td>47.3</td><td>33.4</td><td>66.5</td><td>56.9</td><td>67.9</td><td>60.7</td><td>81.9</td><td>73.5</td></tr><tr><td>Average</td><td>47.5</td><td>28.7</td><td>69.9 68.4</td><td>51.5</td><td>67.1</td><td>57.6</td><td>82.8</td><td>72.1</td></tr></table>

Table 7: InternVL3.5-8B-HF full-token SR and practical-budget conditional CBR (%) across four compressors.
<table><tr><td>Compression Method</td><td>TextVQA-Open SR=93.29%</td><td>VQAv2-Open SR=88.11%</td><td>ScienceQA-MC SR=97.87%</td><td>VQAv2-MC SR=96.48%</td></tr><tr><td></td><td colspan="4">CBR at Practical Budget (Open: 1/9; MC: 1/18)</td></tr><tr><td>VisionZIP</td><td>62.18</td><td>28.00</td><td>41.39</td><td>23.00</td></tr><tr><td>VisPruner</td><td>72.48</td><td>36.07</td><td>41.18</td><td>23.54</td></tr><tr><td>FlowCut</td><td>67.64</td><td>36.61</td><td>43.88</td><td>26.57</td></tr><tr><td>PruMerge</td><td>60.12</td><td>23.96</td><td>37.55</td><td>21.92</td></tr><tr><td>Average</td><td>65.61</td><td>31.16</td><td>41.00</td><td>23.76</td></tr></table>

Table 7 reports exact-match SR and conditional CBR on InternVL3.5-8B-HF. Macro SR is 93.94%, compared with 96.3% on LLaVA and 99.3% on Qwen3.5. Macro CBR over the 16 dataset and compressor combinations is 40.38%, higher than LLaVA's 22.1% but lower than Qwen3.5's 46.43%. InternVL3.5 therefore incurs a greater cost to full-token preservation while exhibiting a conditional blinding rate between the other two models. Its exact-match SR uses binary correctness, while Table 6 reports canonical task-credit accuracy, so the latter cannot directly reproduce SR.

Across LLaVA, Qwen3.5, and InternVL3.5, FATA and its adaptations retain most full token performance while exhibiting greater accuracy loss under compression than Clean.

## 5 DETECTION ANALYSIS

Beyond the performance of FATA in full token retention and failure after compression, this section evaluates the detectability of FATA inputs using input and internal-state detectors

Table 8 reports the area under the receiver operating characteristic curve (AUROC), the area under the precision recall curve (AUPR), and true positive rate at a 5% false positive rate (TPR@5FPR) for FATA, CAA, CAGE, and Base using Feature Squeezing (Xu et al., 2017), Mahalanobis-Max (Lee et al., 2018), and a Multi-Level Activation Trajectory Detector (ML-ATD) inspired by HiddenDetect (Jiang et al., 2025). Among the three detectors, Feature Squeezing compares model predictions before and after input transformations by measuring changes in their scores against ground-truth answers. Mahalanobis-Max measures feature space deviation from normal samples; ML-ATD tracks changes in activations across the visual encoder, projector, and LLM. Additional detector construction and evaluation details are provided in Appendix H.

Table 8: Cross-attack detection performance with clean and random negatives. Higher values indicate easier detection; protocol differences and limitations are detailed in Appendix H.
<table><tr><td rowspan="2"></td><td colspan="3">Feature Squeezing</td><td colspan="3">Mahalanobis-Max</td><td colspan="3">ML-ATD</td></tr><tr><td>Attack AUROC</td><td>AUPR</td><td>TPR@5FPR AUROC</td><td></td><td>AUPR</td><td>TPR@5FPR AUROC</td><td></td><td>AUPR</td><td>TPR@5FPR</td></tr><tr><td>FATA</td><td>0.560</td><td>0.382</td><td>0.024</td><td>0.695</td><td>0.509</td><td>0.134</td><td>0.828</td><td>0.724</td><td>0.497</td></tr><tr><td>CAGE</td><td>0.797</td><td>0.655</td><td>0.147</td><td>0.936</td><td>0.882</td><td>0.647</td><td>0.913</td><td>0.854</td><td>0.694</td></tr><tr><td>CAA</td><td>0.564</td><td>0.385</td><td>0.027</td><td>0.833</td><td>0.765</td><td>0.468</td><td>0.802</td><td>0.721</td><td>0.533</td></tr><tr><td>Base</td><td>0.665</td><td>0.494</td><td>0.064</td><td>0.813</td><td>0.660</td><td>0.278</td><td>0.952</td><td>0.918</td><td>0.801</td></tr></table>

FATA has the lowest TPR@5FPR among the four attacks for each detector. Feature Squeezing gives similarly low recall for FATA and CAA (2.4% and 2.7%), whereas Mahalanobis-Max separates them more clearly (13.4% and 46.8%). Under ML-ATD, FATA reaches 49.7%, compared with 53.3% for CAA, 69.4% for CAGE, and 80.1% for Base; CAA has slightly lower AUROC and AUPR. Table 17 in Appendix H reports the relative reductions from Base to FATA.

## 6 CONCLUSION

FATA combines attention disruption and semantic preservation to shift damage from full-token to compressed inference. It retains 96.3% of LLaVA full-token performance with 22.1% practicalbudget conditional blinding, and model-adapted Qwen3.5 and InternVL3.5 tests show that the trigger extends beyond LLaVA. These results call for evaluating token compression by adversarial behavior as well as clean efficiency and accuracy.

Limitations. FATA assumes visual-encoder gradients. Its strongest Qwen3.5 and InternVL3.5 results use task- or architecture-aware extensions, with adaptation-dependent full-token cost. Stealth is task-based, and ML-ATD is not an optimal defense. Black-box transfer, broader tasks, and stronger online detectors remain open.

## AI USE STATEMENT

Generative AI tools, including ChatGPT/Codex and Claude-based coding assistants, were used to assist with implementation and debugging, dataset-processing workflows, feedback on experimental procedures, and interpretation of quantitative and qualitative results. They also supported literature search, Chinese-English translation, manuscript drafting and editing, and preparation of figures, tables, and references. AI-assisted revisions were checked against the manuscript's equations and tables, dataset-selection records, and source publications. The authors retain responsibility for the experimental evidence, methodological choices, and final text, including all AI-assisted content.

## ETHICS STATEMENT

This work examines a security risk in efficient vision-language inference. Attacks that preserve full-token behavior could be misused to evade screening and disrupt systems after compression is enabled. Our evaluation uses existing image-question benchmarks and model checkpoints to characterize this risk, and includes detection experiments to inform mitigation.Deployment safety and robustness across populations and applications require further evaluation. Reuse of benchmark images and model checkpoints remains subject to their original licenses and privacy obligations; the attack methods are intended for controlled, authorized robustness evaluation.

## REPRODUCIBILITY STATEMENT

Algorithm 1 and Appendix A specify the FATA optimization procedure. Appendix B documents practical-budget selection and the model-specific retention ratios. Appendix Cdefines the evaluation metrics. Appendix D records the benchmark configuration and attack settings, with the unique-image selection, deduplication, and scoring protocol in Appendix D.1. The Qwen3.5 and InternVL3.5 adaptations are given in Appendix G, and detector fitting, scoring, and evaluation protocols are described in Appendix H.

## REFERENCES

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversitybased visual token pruning for large multimodal models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9392–9401. IEEE, 2025.

Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, Yifei Shen, Yuanhan Zhang, Kaichen Zhang, Wenkang Zhang, Zheng Cheng, Nansen Zhang, Chunsheng Wu, Chunjiang Ge, Zimin Ran, Dehua Song, Chunyuan Li, Shikun Feng, Ming Hu, Zhangquan Chen, Junbo Niu, Bo Li, Ziyong Feng, Ziwei Liu, Zongyuan Ge, and Jiankang Deng. LLaVA-OneVision-2: Towards next-generation perceptual intelligence. arXiv preprint arXiv:2605.25979, 2026. doi: 10.48550/arXiv.2605.25979. URL https : // arxiv.org/abs/2605.25979.

Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution visionlanguage models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39(2), pp. 1773–1781, 2025.

Yiming Cao, Yanjie Li, Kaisheng Liang, and Bin Xiao. Enhancing targeted adversarial attacks on large vision-language models via intermediate projector. IEEE Transactions on Information Forensics and Security, 2026.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6904–6913, 2017.

Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, and Qing Guo. Efficient generation of targeted and transferable adversarial examples for vision-language models via diffusion models. IEEE Transactions on Information Forensics and Security, 20:1333–1348, 2024.

Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, and Xiangyu Yue. Hiddendetect: Detecting jailbreak attacks against multimodal large language models via monitoring hidden states. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14880–14893, 2025.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Advances in neural information processing systems, 31, 2018.

Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, 133(10):6794–6812, 2025.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296. IEEE, 2024.

Dong Lu, Zhiqiang Wang, Teng Wang, Weili Guan, Hongchang Gao, and Feng Zheng. Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 102–111. IEEE, 2023.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, 2022.

Hefei Mei, Zirui Wang, Shen You, Minjing Dong, and Chang Xu. Veattack: Downstream-agnostic vision encoder attack against large vision language models. In International Conference on Learning Representations, volume 2026, pp. 18135–18161, 2026.

Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, and Yue Zhao. Jaildam: Jailbreak detection with adaptive memory for vision-language model. arXiv preprint arXiv:2504.03770, 2025.

Qwen Team. Qwen3.5-9b. Hugging Face model card, 2026. Qwen/Qwen3.5-9B.

Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22857–22867. IEEE, 2025.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8317–8326, 2019.

Jiafei Song, Fengwei Zhou, Jin Qu, Wenjin Jason Li, Tong Wu, Gengjian Xue, Zhikang Zhao, Daomin Wei, Yichao Lu, and Bailin Na. EvoComp: Learning visual token compression for multimodal large language models via semantic-guided evolutionary labeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. URL https://arxiv.org/abs/2604.17087.

Jintao Tong, Wenwei Jin, Pengda Qin, Anqi Li, Yixiong Zou, Yuhong Li, Yuhua Li, and Ruixuan Li. Flowcut: Rethinking redundancy via information flow for efficient vision-language models. Advances in Neural Information Processing Systems, 38:94946–94973, 2026.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, andefficiency,2025. URL https://arxiv.org/abs/2508.18265.

Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, and Linli Xu. Break the visual perception: Adversarial attacks targeting encoded visual tokens of large vision-language models. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 1072–1081, 2024.

Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. arXiv preprint arXiv:1704.01155, 2017.

Yue Xu, Xiuyuan Qi, Zhan Qin, and Wenjie Wang. Cross-modality information check for detecting jailbreaking in multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 13715–13726, 2024.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19792–19802. IEEE, 2025.

Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24972–24982. IEEE, 2025.

Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, and Dit-Yan Yeung. Anyattack: Towards large-scale self-supervised adversarial attacks on vision-language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19900–19909. IEEE, 2025a.

Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20857–20867. IEEE, 2025b.

Xiaomei Zhang, Zhaoxi Zhang, Leo Yu Zhang, Yanjun Zhang, Guanhong Tao, and Shirui Pan. Less is more-until it breaks: Security pitfalls of vision token compression in large vision-language models. arXiv preprint arXiv:2601.12042, 2026a.

Xinwei Zhang, Hangcheng Liu, Li Bai, Hao Wang, Qingqing Ye, Tianwei Zhang, and Haibo Hu. On the adversarial robustness of large vision-language models under visual token compression. arXiv preprint arXiv:2601.21531, 2026b.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024.

## APPENDIX CONTENTS

Appendix A. Optimization Details . . 13   
Appendix B. Model-Specific Visual-Token Budgets . 14   
Appendix C. Detailed Evaluation Definitions . 16   
Appendix D. Experimental and Implementation Details . 17   
Appendix E. Qualitative Compression-Triggered Cases . 19   
Appendix F. Additional Ablation Results. .22   
Appendix G. Additional Cross-Model Results . 22   
Appendix H. Detection Details and Cross-Attack Analysis . .27

## A OPTIMIZATION DETAILS

## A.1 DETAILED OPTIMIZATION PROCEDURE

Algorithm 2 presents the detailed optimization procedure corresponding to Algorithm 1. The target set M and its clean feature references are computed once and held fixed throughout optimization. At each iteration, attention scores and patch features are recomputed from the perturbed image to evaluate both losses on $\mathcal { M }$ Gradients are propagated through the frozen encoder to update only the input perturbation.

```latex
Algorithm 2 FATA optimization in detail
Require: Clean image x, visual encoder $E ,$ budget $\epsilon ,$ step size $\alpha .$ iterations $T ,$ weight λ, target count M
1: Extract clean attention $\mathbf { s } ^ { c }$ and patch features $\mathbf { \bar { H } } ^ { c }$ from $E ( x )$
2: Fix $\mathcal { M } \gets \mathrm { T o p K } ( \mathbf { s } ^ { c } , M )$
3: Initialize $\delta ^ { ( 0 ) } \sim \mathcal { U } ( - \epsilon , \epsilon )$
4: for $t = 0 , \ldots , T - 1$ do
5: $x _ { a } ^ { t } \gets \mathrm { c l i p } ( x + \delta ^ { ( t ) } , 0 , 1 )$
6: Extract s $\bar { \mathbf { \xi } } ^ { \bar { a } } , \mathbf { H } ^ { a } \gets E ( \bar { x } _ { a } ^ { t } )$
7: $\textstyle { \mathcal { L } } _ { \mathrm { a t t n } } \gets \sum _ { i \in { \mathcal { M } } } s _ { i } ^ { a }$
8: $\begin{array} { r } { \mathcal { L } _ { \mathrm { s e m } }  1 - \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \cos ( \mathbf { h } _ { i } ^ { a } , \mathbf { h } _ { i } ^ { c } ) } \end{array}$
9: $\mathcal { L } _ { \mathrm { F A T A } }  \mathcal { L } _ { \mathrm { a t t n } } + \lambda \mathcal { L } _ { \mathrm { s e m } }$
10: $g ^ { ( t ) } \gets \nabla _ { \delta } \mathcal { L } _ { \mathrm { F A T A } }$
11: $\mathring { \delta } ^ { { ( t + 1 ) } } \gets \Pi _ { \| \delta \| _ { \infty } \leq \epsilon } ( \delta ^ { ( t ) } - \alpha \mathrm { s i g n } ( g ^ { ( t ) } ) )$
12: end for
13: return $\mathrm { c l i p } ( x + \delta ^ { ( T ) } , 0 , 1 )$
```

## A.2 SENSITIVITY TO PGD HYPERPARAMETERS

Both hyperparameter sweeps evaluate FATA on LLaVA-1.5-7B using TextVQA-Open and VisionZIP.

Evaluation data. Tables 9 and 10 report experiments on unfiltered TextVQA-Open data used during method development, before applying the subset-selection procedure in Appendix D.1. Table 1 uses the filtered evaluation subset, which retains examples answered correctly with the original image but incorrectly with a black image at full tokens. This subset emphasizes reliance on visual evidence and exhibits more pronounced compression-triggered attack effects. The different sample populations therefore explain the difference in absolute accuracy between these sweeps and the TextVQA-VisionZIP results in Table 1, even when the model, compressor, and hyperparameters coincide.

We use random-start $\ell _ { \infty }$ PGD with $\epsilon = 2 / 2 5 5 , T = 1 0 0$ and $M = 6 4$ fixed clean target tokens, and evaluate $K \in \{ 5 7 6 , 1 9 2 , 1 2 8 , 6 4 , 3 2 , 1 6 \}$ , with $K = 5 7 6$ denoting full-token inference. We fix $\lambda = 1$ when varying the step size α and $\alpha = \mathrm { { \bar { 0 } } . 5 / 2 5 5 }$ when varying the semantic-preservation weight λ. Step sizes are reported in units of $1 / 2 5 5$ . Parameter selection considers full-token preservation and compressed accuracy over this trajectory.

Table 9: FATA accuracy on unfiltered TextVQA-Open across token budgets and step sizes α.

PGD step size. $\mathrm { A t } \alpha = 0 . 5 / 2 5 5$ , Table 9 reports 95.0% full-token accuracy and the lowest mean compressed accuracy, 84.1%, over $K \in \{ 1 9 2 , 1 2 8 , 6 4 , 3 2 , 1 6 \}$ Reducing α to $0 . 2 5 / 2 5 5$ improves full-token accuracy to $9 6 . 0 \%$ but raises the compressed-budget mean to 84.8%. $\mathrm { A t } \alpha = 0 . 7 5 / 2 5 5$ , the corresponding values are 94.3% and 84.4%. We choose $\alpha ~ = ~ 0 . 5 / 2 5 5$ for its stronger compression-triggered effect with high full-token accuracy. Its entries coincide with the shared $\lambda = 1$ setting in Table 10.

<table><tr><td>K</td><td>.25</td><td>.5</td><td>.75</td><td>1</td><td>1.25</td><td>1.5</td><td>1.75</td></tr><tr><td>576 192</td><td>96.0 94.1</td><td>95.0 92.7</td><td>94.3 93.4</td><td>94.5 93.8</td><td>94.2 93.8</td><td>95.1 92.9</td><td>94.3 94.0</td></tr><tr><td>128</td><td>93.0</td><td>91.5</td><td>92.1</td><td>92.1</td><td>92.8</td><td>91.7</td><td>92.7</td></tr><tr><td>64</td><td>88.9</td><td>87.6</td><td>88.4</td><td>88.5</td><td>89.4</td><td>88.8</td><td>88.7</td></tr><tr><td>32 16</td><td>78.8 69.4</td><td>80.1 68.8</td><td>80.7 67.3</td><td>80.6 69.4</td><td>82.0 68.1</td><td>81.4 69.2</td><td>80.9 71.0</td></tr></table>

Semantic-preservation weight. Table 10 exposes the expected trade-off. Increasing λ steadily improves full-token accuracy from 94.1% at $\lambda = 0 . 5$ to 96.0% at $\lambda = 8 ,$ but it also raises the K = 16 accuracy from 65.4% to $7 0 . 7 \%$ , indicating a weaker compression-triggered effect. The selected λ = 1 recovers 0.9 points of full-token accuracy over $\lambda = 0 . 5$ while retaining $6 8 . 8 \%$ accuracy at $K = 1 6$ and 80.1% at $K = 3 2$ . Moving to $\lambda \geq 2$ provides at most another 1.0 point at full tokens, but generally increases compressed-budget accuracy; at $\lambda = 8 ,$ for instance, the $\bar { K } = 3 2$ and $K = 1 6$ accuracies rise to 81.0% and 70.7%. We choose λ = 1 to limit full-token damage while retaining the compression-triggered effect that weakens at larger weights.

Table 10: FATA accuracy on unfiltered TextVQA-Open across token budgets and semantic weights λ.
<table><tr><td>K</td><td> $\lambda = . 5$ </td><td>1</td><td>2</td><td>4</td><td>8</td></tr><tr><td>576</td><td>94.1</td><td>95.0</td><td>95.2</td><td>95.6</td><td>96.0</td></tr><tr><td>192</td><td>92.0</td><td>92.7</td><td>93.3</td><td>94.2</td><td>94.3</td></tr><tr><td>128</td><td>91.8</td><td>91.5</td><td>93.1</td><td>93.3</td><td>92.2 90.1</td></tr><tr><td>64 32</td><td>87.8 77.4</td><td>87.6 80.1</td><td>87.5 81.1</td><td>89.0</td><td>81.0</td></tr><tr><td>16</td><td>65.4</td><td>68.8</td><td>68.8</td><td>81.4 70.3</td><td>70.7</td></tr></table>

Together, the sweeps support $\alpha = 0 . 5 / 2 5 5$ and $\lambda = 1$ as a balance between full-token preservation and compression-triggered failure across budgets.

## B MODEL-SPECIFIC VISUAL-TOKEN BUDGETS

## B.1 LLAVA-1.5-7B

LLaVA-1.5-7B receives a fixed sequence of 576 visual patch tokens from its vision encoder. We use $K \ = \ 5 7 6$ for full-token inference and evaluate the five compressed budgets $K \in$ $\{ 1 9 2 , 1 2 8 , 6 4 , 3 2 , 1 6 \}$ . Relative to the full sequence, these settings retain $1 / 3 , 2 / 9 , \mathsf { \bar { 1 } / 9 } , 1 / 1 8$ and 1/36 of the visual tokens, respectively. Here, $K$ denotes the number of selected or merged visual representatives. In the reconstruction-based LLaVA path in Figure 2D, these representatives reconstruct an $N = 5 7 6$ -slot sequence before the projector.

The practical budget is selected separately for each of the 16 dataset-compressor pairs. Let $\kappa =$ {16, 32, 64, 128, 192} denote the candidate budgets. Clean accuracy at each budget is measured relative to full-token clean accuracy:

$$
R _ { \mathrm { c l e a n } } ^ { ( d , c ) } ( K ) = \frac { \operatorname { A c c } _ { \mathrm { C l e a n } } ^ { ( d , c ) } ( K ) } { \operatorname { A c c } _ { \mathrm { C l e a n } } ^ { ( d , c ) } ( 5 7 6 ) } .\tag{9}
$$

Both accuracies are measured on the same examples using identical prompts and the same task scorer. Open-ended tasks use normalized VQA scores, and multiple-choice tasks use option-letter accuracy Retention ratios are computed from unrounded task accuracies.

Using this ratio, we select the smallest candidate budget $K _ { \mathrm { p r a c } } ^ { ( d , c ) }$ that retains at least 80% of full-token clean accuracy:

$$
K _ { \mathrm { p r a c } } ^ { ( d , c ) } = \operatorname* { m i n } \left\{ K \in \mathcal { K } : R _ { \mathrm { c l e a n } } ^ { ( d , c ) } ( K ) \geq 0 . 8 0 \right\} .\tag{10}
$$

The candidates are compared by their numeric values, from 16 upward. The uncompressed point $K = 5 7 6$ is excluded from K because its retention ratio relative to itself is necessarily one.

Table 11 reports the practical budgets obtained by applying Eq. 10 to each of the 16 datasetcompressor pairs.

Table 11: Clean-only practical token budgets $K _ { \mathrm { p r a c } } ^ { ( d , c ) }$ for LLaVA-1.5-7B. Each entry is the smallest $K \in \{ 1 6 , 3 2 , 6 4 , 1 2 \dot { 8 } .$ 192} that retains at least 80% of the corresponding full-token clean accuracy.
<table><tr><td>Compressor</td><td>TextVQA-Open</td><td>VQAv2-Open</td><td>ScienceQA-MC</td><td>VQAv2-MC</td></tr><tr><td>VisionZIP</td><td>64</td><td>64</td><td>32</td><td>32</td></tr><tr><td>VisPruner</td><td>64</td><td>64</td><td>32</td><td>32</td></tr><tr><td>PruMerge</td><td>64</td><td>64</td><td>32</td><td>32</td></tr><tr><td>FlowCut</td><td>64</td><td>64</td><td>32</td><td>64</td></tr></table>

The selected budgets are fixed for all attacks. All four compressors use $K = 6 4$ on the open-ended tasks and $K = { \bar { 3 } } 2$ on ScienceQA-MC. On VQAv2-MC, FlowCut requires $K = 6 4$ for its clean accuracy at $K = 3 2$ falls below 80% of its full-token value; the other three compressors use $K = 3 2$

The visual-token retention ratio measures the fraction of visual tokens retained at the selected budget. For LLaVA, it is

$$
r _ { \mathrm { p r a c } } ^ { ( d , c ) } = \frac { K _ { \mathrm { p r a c } } ^ { ( d , c ) } } { 5 7 6 } .\tag{11}
$$

$K = 6 4$ and $K = 3 2$ correspond to retention fractions of $1 / 9$ and $1 / 1 8$ , respectively. CBR is evaluated at these budgets using the definition in Appendix C.

## B.2 QWEN3.5 AND INTERNVL3.5

The number of visual tokens in Qwen3.5 and InternVL3.5 varies across images because of their image preprocessing and token organization. For each image i with $N _ { i }$ visual tokens, the compression budget is specified as a retained fraction r of $N _ { i }$

The main comparisons on Qwen3.5 and InternVL3.5 report full visual tokens, $1 / 3$ retention, and a practical budget specified for each task. The $1 / 3$ fraction matches the proportion retained by LLaVA at $K = 1 9 2$ . Before adversarial evaluation, the practical fractions are fixed at $1 / 9$ for TextVQA-Open and VQAv2-Open and $1 / 1 8$ for ScienceQA-MC and VQAv2-MC, corresponding to LLaVA's 64/576 and 32/576 settings, respectively. These fractions apply to all four compressors, with the visual token budgets and sample counts for all three models summarized in Table 12.

Table 12: Model-specific budgets and sample counts used in evaluation. Practical budgets are given separately for open-ended and multiple-choice tasks.
<table><tr><td>Model</td><td>Full</td><td>Practical Open</td><td>Practical MC</td><td>Valid images per dataset</td></tr><tr><td>LLaVA-1.5-7B</td><td>576</td><td>64</td><td>32 or 64</td><td>1,000</td></tr><tr><td>Qwen3.5-9B</td><td> $N _ { i }$ </td><td> $r = 1 / 9$ </td><td> $r = 1 / 1 8$ </td><td>1,000</td></tr><tr><td>InternVL3.5-8B-HF</td><td> $N _ { i }$ </td><td> $r = 1 / 9$ </td><td> $r = 1 / 1 8$ </td><td>998 or 1,000</td></tr></table>

## C DETAILED EVALUATION DEFINITIONS

The metrics in Section 3.3 use the model-specific budgets in Table 12. Accuracy enters the equations as a fraction in [0, 1] and is reported as a percentage. The metrics in Section 3.3 are evaluated using the budgets and sample counts summarized in Table 12.

Task scores and binary correctness. For a dataset with n valid examples, let $v _ { i } \in [ 0 , 1 ]$ denote the task credit and $c _ { i } \in \{ 0 , 1 \}$ the binary correctness of prediction i. The corresponding accuracies are

$$
{ \mathrm { A c c } } ^ { \mathrm { t a s k } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } v _ { i } , \qquad { \mathrm { A c c } } ^ { \mathrm { E M } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } c _ { i } .\tag{12}
$$

Table 13 gives the task-specific rules. Open answers use the benchmark's normalization of punctuation, articles, and number words before matching.

Table 13: Task-level scoring rules for reported accuracy and binary correctness. CBR uses the binary decisions in the final column.
<table><tr><td>Task</td><td>Task credit  $v _ { i }$ </td><td>Binary correctness  $c _ { i }$ </td></tr><tr><td>VQAv2-Open</td><td>TextVQA-Open and Normalized VQA credit against the human 1 for a normalized exact match to an ac- reference answers, with partial credit al- cepted reference answer; 0 otherwise. lowed.</td><td></td></tr><tr><td>ScienceQA-MC</td><td>notated answer index converted to a letter; 0 otherwise.</td><td>1 if the parsed letter (A-F) matches the an- The same option-letter decision as  $v _ { i } .$ </td></tr><tr><td>VQAv2-MC</td><td>fixed target derived from the modal human answer; 0 otherwise.</td><td>1 if the parsed letter (A-D) matches the The same option-letter decision as  $v _ { i } .$ </td></tr></table>

Metric computation. The accuracy tables for all three models report $\operatorname { A c c } ^ { \mathrm { t a s k } }$ . SR is computed using this accuracy for LLaVA and Qwen3.5 and $\operatorname { A c c } ^ { \mathrm { E M } }$ for InternVL3.5, and is averaged over the four datasets.

CER is reported only for LLaVA and is the complement of $\operatorname { A c c } ^ { \mathrm { t a s k } }$ at $K _ { \mathrm { p r a c } } ,$ following Eq. 7. It retains partial task credit and measures overall error without conditioning on Clean correctness or isolating attack-induced failures. The summary is the equal-weight mean over the 16 dataset– compressor settings, calculated from the displayed accuracies, expressed as a percentage, and rounded to two decimal places, as specified alongside Table 2.

CBR uses the binary correctness decisions $c _ { i }$ for all three models, following Eq. 8. For attack m, its denominator requires $c _ { i } = 1$ for Clean at full tokens, attack m at full tokens, and Clean under the evaluated compressor and practical budget; the numerator additionally requires $c _ { i } = 0$ for attack m under that same compression setting. Macro CBR is the unweighted mean over the 16 dataset-compressor pairs, whereas pooled CBR is computed by summing their numerator counts and denominator counts before division.

## D EXPERIMENTAL AND IMPLEMENTATION DETAILS

Main benchmark. The main victim is LLaVA-1.5-7B with a CLIP ViT-Large visual encoder. We test VisionZIP, VisPruner, PruMerge, and FlowCut at $K \in \{ 5 7 6 , 1 9 2 , 1 2 8 , 6 4 , 3 2 , 1 6 \}$ . Each of TextVQA-Open, VQAv2-Open, ScienceQA-MC, and VQAv2-MC contains 1,000 evaluation samples. Base and FATA use $\epsilon = \bar { 2 } / 2 5 5 , \alpha = 0 . 5 / 2 5 5$ , and 100 steps; CAA uses the same $\epsilon = 2 / 2 5 5$ comparison budget with $\alpha = 1 / 2 5 5$ and 100 steps; CAGE is run under $\epsilon = 2 / 2 5 5$

## D.1 EVALUATION SUBSET CONSTRUCTION

The procedure below documents the released dataset reconstruction. The reported experiments retain their archived evaluation mappings; in particular, the original VQAv2-MC options cannot be reconstructed from the released builder's seed alone.

Selection unit and representative question. Each subset uses unique images as the selection unit. Images are deduplicated using source identifiers and image content. For each image, the first eligible question in source order is fixed before model inference. If the resulting pair fails the selection criteria, no alternative question is selected for that image.

Selection based on model responses. LLaVA decodes each candidate greedily with at most 32 new tokens, using the same prompt and question for the original image and a $3 3 6 \times 3 3 6$ black image. Both conditions use all 576 visual tokens and greedy decoding with at most 32 new tokens. We retain an image only if the answer is incorrect with a $3 3 6 \times 3 3 6$ black image and correct with the original image. Let $q _ { i }$ denote the representative question assigned to image $x _ { i } ,$ b the black image, and $c ( \hat { y } , y )$ the correctness rule used during subset construction. The evaluation subset is

$$
\begin{array} { r } { \mathcal { D } _ { j } ^ { \mathrm { e v a l } } = \mathrm { F i r s t } _ { 1 0 0 0 } \{ ( x _ { i } , q _ { i } , y _ { i } ) \in \mathcal { U } _ { j } : c ( f _ { 5 7 6 } ( b , q _ { i } ) , y _ { i } ) = 0 \land c ( f _ { 5 7 6 } ( x _ { i } , q _ { i } ) , y _ { i } ) = 1 \} , } \end{array}\tag{13}
$$

where $\mathcal { U } _ { j }$ contains the candidate question-image pairs and reference answers for dataset j, ordered by source occurrence after image deduplication. $\mathrm { F i r s t _ { 1 0 0 0 } }$ selects the first 1,000 pairs satisfying both conditions. Selection uses no attack predictions or results under token compression.

TextVQA-Open. We use the TextVQA validation split. During selection, open answers are lowercased, punctuation and articles are removed, and number words from zero through five are mapped to digits. A prediction passes if it exactly matches a normalized reference answer or contains that answer within the generated phrase.

VQAv2-Open. We use the VQAv2 validation split and exclude yes/no questions. Selection uses the same answer normalization and matching rule as TextVQA-Open.

VQAv2-MC. The released reconstruction uses the VQAv2 validation split, excludes yes/no questions and invalid option sets, and pairs the modal human answer with three distinct distractors. Its configured global seed is $s _ { 0 } ~ = ~ 2 0 2 6 0 9 0 4$ . For each question, the UTF-8 string formed by $\mathtt { v q a v 2 - m c - o p t i o n s - v 1 } , s _ { 0 }$ in decimal, and question\_id, separated by null bytes, is hashed with SHA-256. The first eight digest bytes, interpreted as an unsigned big-endian integer, seed a local Python random . Random instance for distractor draws and option shuffling. Thus $s _ { 0 }$ is a fixed configuration constant, while each question's random seed is derived deterministically from it and the question ID. The reconstruction excludes images selected for VQAv2-Open. For the archived experiments, the original option-generation seed and RNG state are unavailable; exact reproduction uses the saved questions, option order, and target letters in the archived mapping.

ScienceQA-MC. The candidate pool combines the ScienceQA train, validation, and test splits for evaluation only. We discard records without images or with invalid metadata or choices. Image deduplication also removes exact duplicates across splits. The optional hint is prepended to the question, the original choices are labeled A-F, and the annotated answer index determines the target letter.

Relation to reported accuracy. The reported evaluations use archived question-image mappings with their saved answer specifications; the released reconstruction above does not redefine those samples. Evaluation uses each model's prompt and formal task scorer, so clean full-token accuracy can be below 100%. Task scoring and metric computation are detailed in Appendix C.

## D.2 PERFORMANCE ACROSS TOKEN BUDGETS

![](images/1cfefb639f87235bfa43da99636a61673efac75049b0c22c48f504ee8232f183.jpg)  
Figure 5: Average task performance versus visual-token budget. The exact per-method values are reported in Table 1 in the main paper.

Figure 5 visualizes the results in Table 1 of the main paper for each dataset. The point at K = 576 reports full-token accuracy, and each compressed-budget point averages accuracy over VisionZIP VisPruner, PruMerge, and FlowCut.

TextVQA-Open shows the largest Clean-FATA gap at every compressed budget. Under severe compression, both open-ended tasks reach lower Clean and FATA accuracies than the multiple-choice tasks. Across all four datasets, the Clean–FATA gap widens as the token budget decreases.

<table><tr><td rowspan=5 colspan=1>Question: What color is the bike?Ground truth: silver</td><td rowspan=1 colspan=1>Compressor</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>K=192</td><td rowspan=1 colspan=1>K=128</td><td rowspan=1 colspan=1>K=64</td><td rowspan=1 colspan=1>K=32</td></tr><tr><td rowspan=1 colspan=1>VisionZIP</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>Silver</td></tr><tr><td rowspan=1 colspan=1>VisPruner</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>Silver</td></tr><tr><td rowspan=1 colspan=1>PruMerge</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1>Silver</td></tr><tr><td rowspan=1 colspan=1>FlowCut</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Silver</td><td rowspan=1 colspan=1>Blue</td><td rowspan=1 colspan=1>Black C/F</td></tr></table>

![](images/5de3ec9b8cedacbb891a760c85aadb28887af70b9bb71911f6993bbfa0da8d67.jpg)

## E QUALITATIVE COMPRESSION-TRIGGERED CASES

In the LLaVA examples, selection maps show last-layer CLIP CLS-to-patch attention on a shared monotonic log scale that preserves token rankings. This visual-encoder attention is independent of the language input. Each Clean Top-K panel partitions all K clean tokens into those present in the adversarial Top-K set (gold) and those displaced (red).

## E.1 LLAVA-1.5-7B

<table><tr><td rowspan=5 colspan=1>Question: what is the brand namefirst word?Ground truth: chateau</td><td rowspan=1 colspan=1>Compressor</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>K=192</td><td rowspan=1 colspan=1>K=128</td><td rowspan=1 colspan=1>K=64</td><td rowspan=1 colspan=1>K=32</td></tr><tr><td rowspan=1 colspan=1>VisionZIP</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Travemunde</td><td rowspan=1 colspan=1>Merlot</td><td rowspan=1 colspan=1>Merlot</td></tr><tr><td rowspan=1 colspan=1>VisPruner</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Merlot</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Travaux</td><td rowspan=1 colspan=1>Patek philippe</td></tr><tr><td rowspan=1 colspan=1>PruMerge</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Merlot</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Gerger</td></tr><tr><td rowspan=1 colspan=1>FlowCut</td><td rowspan=1 colspan=1>Chateau</td><td rowspan=1 colspan=1>Travemunde</td><td rowspan=1 colspan=1>Travemunde</td><td rowspan=1 colspan=1>Meridian</td><td rowspan=1 colspan=1>None</td></tr></table>

<table><tr><td rowspan=5 colspan=1>Question: What color are the kidsskis?Ground truth: B: grayOptions: A. black; B. gray;C. pink; D. brown</td><td rowspan=1 colspan=1>Compressor</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>K=192</td><td rowspan=1 colspan=1>K=128</td><td rowspan=1 colspan=1>K=64</td><td rowspan=1 colspan=1>K=32</td></tr><tr><td rowspan=1 colspan=1>VisionZIP</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>A: black</td><td rowspan=1 colspan=1>A: blackC/F</td><td rowspan=1 colspan=1>A: black</td><td rowspan=1 colspan=1>A: black</td></tr><tr><td rowspan=1 colspan=1>VisPruner</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>C: pink</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>D: brown</td></tr><tr><td rowspan=1 colspan=1>PruMerge</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>A: black</td><td rowspan=1 colspan=1>A: blackCF</td><td rowspan=1 colspan=1>A: blackC/F</td><td rowspan=1 colspan=1>B: gray</td></tr><tr><td rowspan=1 colspan=1>FlowCut</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>B: gray</td><td rowspan=1 colspan=1>A: black</td><td rowspan=1 colspan=1>A: black</td><td rowspan=1 colspan=1>A: black</td></tr></table>

Figure 6: LLaVA-1.5-7B cases from TextVQA-Open (a), VQAv2-Open (b), VQAv2-MC (c), and ScienceQA-MC (d). Tables report FATA predictions across compressors and token budgets, with the shared legend below (a) applying to all panels.

## E.2 QWEN3.5-9B

Selection maps visualize the $\ell _ { 2 }$ norm of post-merger visual features on a shared logarithmic scale. The retention panels track the corresponding global Top-K sets; each compressor follows its own selection rule. Full-token predictions remain correct for Clean and FATA in all four cases. Compressed responses depend on both the compressor and retention fraction. In VQAv2-Open, VisionZIP recovers the correct FATA answer at retention 1/9 after errors at the two larger fractions.

![](images/c520c805dcea135ef716f9eade91ab782266ffc2ef9a973828c1ac22cbab52ad.jpg)  
Figure 7: Qwen3.5-9B cases in the same task order as Figure 6. Tables report FATA answers using the shared colour conventions.

## E.3 INTERNVL3.5-8B-HF

InternVL processes local image tiles and a thumbnail, whose CLS-to-patch scores and token footprints are projected onto the original image. In VQAv2-MC, clean support for the shirt comes mainly from thumbnail tokens. At practical retention, all four compressors lose their target-overlapping clean anchors and return incorrect FATA answers.

![](images/a2f15648900b3f6ac3d433eee3835151bbc107b1ff007b5c02f60a3c76f05ecb.jpg)  
Figure 8: InternVL3.5-8B-HF cases across the four tasks, with global native-priority Top-K panels and the colour conventions of Figure 6

## F ADDITIONAL ABLATION RESULTS

Figure 9 shows how the loss variants behave within individual dataset and compressor settings. In panel (a), attention-only SR spans roughly 40– 90%, whereas semantic-only values cluster near 100%. Full FATA keeps all plotted SR values above 90%, showing that semantic preservation also reduces variation across settings. Panel (b) shows a range for full FATA CBR, from roughly 9% to 55%. Most settings remain below 30%, with two near 40% and 55%. The trajectories from semantic-only to full FATA include both increases and decreases in CBR, so the aggregate gain does not hold uniformly across settings. Together, the panels show consistent full-token retention under the joint objective and a more variable compression-triggered effect.

![](images/c658a44208c578315936c52c6996a290a088343ff775f81b36f8b32aec88ef33.jpg)

![](images/2c722fe007b87b544ad51bef718e419f1b11e818cfcb11e4dc2867ef40de2d99.jpg)  
Figure 9: Loss ablation across 16 dataset-compressor settings, with paired lines and mean diamonds; shading marks full FATA. Error bars show ±1 sample standard deviation across settings

## G ADDITIONAL CROSS-MODEL RESULTS

This appendix details the model-specific objectives and supplementary analyses for the cross-model experiments in Section 4.3. Visual-token budgets and scoring rules are specified in Appendices B and C, respectively.

## G.1 QWEN3.5 TASK-AWARE FATA ADAPTATION

The Qwen3.5 adaptation uses task-level losses to preserve full-token predictions while inducing errors under compression, following FATA's design principle. This white-box adaptation uses ground-truth answers and downstream Qwen3.5 gradients.

Task-aware objective. Let $x _ { a }$ be the perturbed input, y the ground-truth answer, $m _ { t }$ the compressor selected at PGD step t, and $r _ { t }$ the retained-token fraction used at that step. In the released Qwen implementation, x denotes the processor's pixel-value tensor and $\boldsymbol { x } _ { a } = \mathrm { c l i p } ( \boldsymbol { x } + \delta , - 1 , 1 )$ . The four compressors are cycled round-robin, and the budget schedule emphasizes the practical operating point while revisiting $1 / 3$ retention. We write the compression-side attack score as

$$
\mathcal { A } _ { t } = w _ { \mathrm { t a s k } } \mathcal { L } _ { \mathrm { t a s k } } ^ { m _ { t } , r _ { t } } + w _ { \mathrm { r a n k } } \mathcal { D } _ { \mathrm { r a n k } } + w _ { \mathrm { c o m p } } \mathcal { D } _ { \mathrm { c o m p } } ,\tag{14}
$$

where $\mathcal { L } _ { \mathrm { t a s k } } ^ { m _ { t } , r _ { t } }$ is the teacher-forced ground-truth cross-entropy under compressed inference, written as a score to be maximized. The locked coefficients are $w _ { \mathrm { t a s k } } = 1 . 5 , w _ { \mathrm { r a n k } } = 0 . 2 5$ , and $w _ { \mathrm { c o m p } } = 0 . 0 5$

Rank and compressor auxiliaries. The N merged tokens in pooler\_output are denoted by $z _ { i } ( x )$ . Rank the clean tokens by descending $\| z _ { i } ( x ) \| _ { 2 } .$ For each of the medium and practical budgets, $K = \operatorname* { m a x } ( 1 , \operatorname { r o u n d } ( N r ) )$ with $r = 1 / 3$ and $r = 1 / 9$ (Open) or $1 / 1 8 ( \mathrm { M C } )$ , respectively, take

$$
{ w _ { K } = \mathrm { m i n } \{ \mathrm { m a x } ( 8 , \mathrm { r o u n d } ( 0 . 0 5 N ) ) , K , N - K \} }\tag{15}
$$

tokens on each side of the cutoff: the core set contains clean ranks $K - w _ { K } + 1 , \dotsc , K$ , and the decoy set contains ranks $K + 1 , \ldots , K + w _ { K }$ . Let C and D be the respective deduplicated unions over the two budgets. Both sets are fixed before optimization and reused at every step.

Let $\overline { { v } } _ { S } = | S | ^ { - 1 } \sum _ { i \in S } v _ { i }$ and $n _ { i } = \| z _ { i } ( x _ { a } ) \| _ { 2 }$ , and let M contain VisionZIP, VisPruner, FlowCut, and PruMerge. The released implementation defines the maximization scores as

$$
{  { \mathcal D } _ { \mathrm { r a n k } } } = - \mathrm { s o f t p l u s } ( 0 . 0 5 + \overline { { n } } _ { C } - \overline { { n } } _ { D } ) ,\tag{16}
$$

$$
{ \mathcal { D } } _ { \mathrm { c o m p } } = - { \frac { 1 } { 4 } } \sum _ { m \in \mathcal { M } } \mathrm { s o f t p l u s } ( 0 . 0 5 + \overline { { s } } _ { m , C } - \overline { { s } } _ { m , D } ) .\tag{17}
$$

Thus, both margins are 0.05, and softplus is applied after taking the core and decoy means. The negative signs convert the implemented boundary losses into scores to maximize. Unlike the task term,

these auxiliaries use the fixed two-budget targets; the compressor term averages all four surrogates at every step.

For completeness, the surrogate token scores are

$$
\begin{array} { r l r l } & { s _ { \mathrm { V i s i o n Z I P } } = 0 . 7 5 a + 0 . 2 5 v , \qquad } & & { s _ { \mathrm { V i s P r u n e r } } = 0 . 5 a + 0 . 5 b , } \\ & { s _ { \mathrm { F l o w C u t } , i } = ( a _ { i } + c _ { i } ) \| z _ { i } \| _ { 1 } , \qquad } & & { s _ { \mathrm { P r u M e r g e } } = 0 . 7 a + 0 . 3 p . } \end{array}\tag{18}
$$

Here all features are evaluated at $x _ { a }$ . Define Norm $\begin{array} { r } { \mathrm { \Lambda } _ { 1 } ( u ) = u / ( \sum _ { i } u _ { i } + 1 0 ^ { - 8 } ) , a = \mathrm { N o r m } _ { 1 } ( n ) } \end{array}$ and $b = \operatorname { N o r m } _ { 1 } ( d )$ , where $d _ { i } = 1 - ( \sum _ { i } \cos ( z _ { i } , z _ { j } ) - 1 ) / ( N - \overline { { 1 } } ) \overline { { } }$ . The distinctiveness score is $v _ { i } = 1 - \operatorname* { m a x } _ { j \in T _ { \mathrm { d o m } } } \cos ( z _ { i } , z _ { j } )$ , where $T _ { \mathrm { d o m } }$ contains the $\operatorname* { m a x } ( 1 , \lfloor N / 4 \rfloor )$ current tokens with largest feature norms. $\begin{array} { r } { \mathrm { W i t h                              } \bar { z } = N ^ { - 1 } \sum _ { i } z _ { j } , c = \mathrm { N o r m } _ { 1 } ( ( [ \cos ( z _ { i } , \bar { z } ) ] _ { + } ) _ { i } ) } \end{array}$ . Finally, $p = b$ for $N \leq 1 0 2 4$ and $p = \mathrm { N o r m _ { 1 } } ( ( 1 / N , \dots , 1 / N ) )$ otherwise. The PruMerge surrogate receives no spatial grid in this call, and all four surrogate temperatures equal one.

Full-token preservation is imposed with two complementary terms. The first keeps the ground-truth answer likely at full tokens,

$$
\mathcal { L } _ { \mathrm { f u l l } } = \mathrm { C E } \big ( p _ { \theta } ( \cdot \mid x _ { a } , \mathrm { F u l l } ) , y \big ) ,\tag{19}
$$

and the second distills the clean full-token predictive distribution,

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = D _ { \mathrm { K L } } \left( p _ { \theta } ( \cdot  { | } x , \mathrm { F u l l } )  { | | } p _ { \theta } ( \cdot  { | } x _ { a } , \mathrm { F u l l } ) \right) .\tag{20}
$$

Both are computed with teacher forcing during optimization; autoregressive generation is reserved for evaluation. The preservation branch is

$$
\mathcal { P } = w _ { \mathrm { f u l l } } \mathcal { L } _ { \mathrm { f u l l } } + w _ { \mathrm { d i s t i l l } } \mathcal { L } _ { \mathrm { d i s t i l l } } , \qquad &  w _ { \mathrm { f u l l } } = w _ { \mathrm { d i s t i l l } } = 1 .\tag{21}
$$

The two branches summarize the ascent and preservation signs as

$$
\mathcal { T } _ { t } ( \delta ) = \mathcal { A } _ { t } - \mathcal { P } .\tag{22}
$$

The actual update separately normalizes each term's gradient before weighting, rather than differentiating this fixed weighted sum. With $\mathcal { N } ( g ) = g / ( | | g \bar { | } | _ { 2 } + 1 0 ^ { - 8 } )$ , using the L2 norm over the entire gradient tensor, its direction is

$$
\begin{array} { r } { g _ { t } = 1 . 5 \mathcal { N } ( \nabla _ { \delta } \mathcal { L } _ { \mathrm { t a s k } } ^ { m _ { t } , r _ { t } } ) + 0 . 2 5 \mathcal { N } ( \nabla _ { \delta } \mathcal { D } _ { \mathrm { r a n k } } ) + 0 . 0 5 \mathcal { N } ( \nabla _ { \delta } \mathcal { D } _ { \mathrm { c o m p } } ) } \\ { - \mathcal { N } ( \nabla _ { \delta } \mathcal { L } _ { \mathrm { f u l l } } ) - \mathcal { N } ( \nabla _ { \delta } \mathcal { L } _ { \mathrm { d i s t i l l } } ) . } \end{array}\tag{23}
$$

Each gradient includes the derivative mask of the input clipping operation. Random-start $\ell _ { \infty }$ PGD then updates

$$
\delta ^ { ( t + 1 ) } = \Pi _ { [ - \epsilon , \epsilon ] } \left[ \delta ^ { ( t ) } + \alpha \mathrm { ~ s i g n } \left( g _ { t } \right) \right] , \qquad x _ { a } ^ { ( t + 1 ) } = \mathrm { c l i p } ( x + \delta ^ { ( t + 1 ) } , - 1 , 1 ) ,\tag{24}
$$

with $\epsilon = 2 / 2 5 5 , \alpha = 0 . 5 / 2 5 5$ , and $T = 1 0 0$ steps in the processor input domain. The final attack uses no additional gradient-projection operator. One adversarial image is generated per input and shared across all four compressors and evaluation budgets.

The compression branch aims to alter which visual evidence survives compression. The preservation branch uses full-token cross-entropy and clean-logit distillation to preserve task semantics under Qwen3.5's dynamic tokenization.

Dataset-level amplification. Table 14 complements the aggregate results in Table 4 with a datasetlevel breakdown. Here $D _ { r }$ denotes the Clean-FATA accuracy difference at budget $r ,$ and $A _ { r } =$ $D _ { r } - D _ { \mathrm { F u l l } }$ . Both are calculated before rounding. Practical-budget amplification is positive across all four datasets.

## G.2 INTERNVL3.5 ARCHITECTURE-AWARE FATA ADAPTATION

The InternVL3.5 adaptation combines compression-side disruption with preservation of the uncompressed representation. Its archived V5 objective includes VisionZIP, VisPruner, DivPrune, PruMerge, and FlowCut; the reported comparisons show four of these compressors, excluding DivPrune.

Table 14: Qwen3.5 compression-specific damage and amplification by dataset (percentage points). Positive amplification means that compression increases the attack-induced accuracy gap relative to full-token inference.
<table><tr><td>Dataset</td><td> $D _ { \mathrm { F u l l } }$ </td><td> $D _ { 1 / 3 }$ </td><td> $D _ { \mathrm { P r a c } }$ </td><td> $A _ { 1 / 3 }$ </td><td> $A _ { \mathrm { P r a c } }$ </td></tr><tr><td>TextVQA-Open</td><td>0.70</td><td>5.22</td><td>8.60</td><td>4.52</td><td>7.90</td></tr><tr><td>VQAv2-Open</td><td>1.00</td><td>10.00</td><td>14.57</td><td>9.00</td><td>13.57</td></tr><tr><td>ScienceQA-MC</td><td>-0.20</td><td>9.47</td><td>23.70</td><td>9.67</td><td>23.90</td></tr><tr><td>VQAv2-MC</td><td>1.00</td><td>8.05</td><td>19.60</td><td>7.05</td><td>18.60</td></tr></table>

Features and post-compression disruption. For an image with J tiles, let $H ( x ) \in \mathbb { R } ^ { J \times 2 5 6 \times d _ { h } }$ contain the post-shuffle features before the projector, and let $\breve { P } ( x ) \in \mathbb { R } ^ { J \times 2 5 6 \times d _ { p } }$ contain the projector outputs. Thus $N = 2 5 6 J ;$ all tiles, including a thumbnail when present, contribute patch tokens, with CLS excluded. For compressor c and retention fraction $r ,$ the attack pass produces $\bar { Z } _ { c , r } ( x )$ with $K _ { c , r }$ output tokens. The post-compression score compares corresponding output positions:

$$
\mathcal { D } _ { \mathrm { p c d } } ^ { c , r } = 1 - \frac { 1 } { K _ { c , r } } \sum _ { j = 1 } ^ { K _ { c , r } } \cos \left( z _ { j } ^ { c , r } ( x _ { a } ) , z _ { j } ^ { c , r } ( x ) \right) .\tag{25}
$$

The cosine is computed along the feature dimension and then averaged over tokens, without pooling beforehand. This attack pass disables reconstruction; PruMerge returns selected centers without merging in this mode. The comparison follows output order rather than matching source-token identities. Clean outputs are fixed, while adversarial selections are recomputed at each step.

Cutoff targets and retained sets. At the practical fraction $r _ { * } ,$ the total budget is $K \_ { } =$ max(1, round $( N r _ { * } ) )$ . Each tile first receives $\lfloor \bar { K } / J \rfloor$ tokens, with the remainder assigned to earlier tiles. For a tile budget $k _ { t }$ , sort the clean family scores $s _ { c , t , i } ( x )$ in descending order and set

$$
b _ { t } = \operatorname* { m i n } \{ 6 4 , \operatorname* { m a x } ( 1 6 , \operatorname { r o u n d } ( 0 . 2 5 k _ { t } ) ) , k _ { t } - 1 , 2 5 5 - k _ { t } \} .\tag{26}
$$

The ordered lists $B _ { c , t } ^ { + }$ and $\boldsymbol { B } _ { c , t } ^ { - }$ contain ranks $k _ { t } - b _ { t } + 1 , \ldots , k _ { t }$ and $k _ { t } + 1 , \ldots , k _ { t } + b _ { t }$ , respectively. Each has $b _ { t }$ entries: 16 for Open and 13 or 14 for MC at the practical fractions. They are fixed from the clean image throughout optimization, without combining different budgets. Ties use the descending argsort order, with no extra tie rule. Separately, ${ \mathcal { U } } _ { c , t }$ contains the source-token indices actually selected by clean compressor c in tile t at $r _ { * }$ . Its size is the actual number of selected representatives, with no additional cap. These retained sets are also fixed and need not equal the boundary lists or their union.

The ranking score pairs entries at the same position in the two lists:

$$
\mathcal { D } _ { \mathrm { r a n k } } ^ { c } = - \frac { 1 } { J } \sum _ { t = 1 } ^ { J } \frac { 1 } { b _ { t } } \sum _ { j = 1 } ^ { b _ { t } } \left[ s _ { c , t , \boldsymbol { \mathscr { B } } _ { c , t } ^ { + } [ j ] } ( \boldsymbol { x } _ { a } ) - s _ { c , t , \boldsymbol { \mathscr { B } } _ { c , t } ^ { - } [ j ] } ( \boldsymbol { x } _ { a } ) \right] _ { + } , \qquad [ \boldsymbol { u } ] _ { + } = \operatorname* { m a x } ( \boldsymbol { u } , \boldsymbol { 0 } ) .\tag{27}
$$

The margin is $\mu = 0$ : this is a paired ReLU penalty expressed as a score to maximize. All reported tiles have nonempty boundary lists; an empty list is omitted from the tile mean.

Family scores. Write A for the number of attention heads and $\alpha _ { a , i }$ for penultimate-layer CLS attention after summing each four-patch group into token i. With $\begin{array} { r } { u _ { i } = A ^ { - 1 } \sum _ { a } \alpha _ { a , i } } \end{array}$ , the VisionZIP and VisPruner rank scores are $\textstyle \sum _ { a } \alpha _ { a , i }$ and $u _ { i } ,$ respectively. PruMerge uses final-layer CLS query–key attention, while FlowCut combines normalized attention, value similarity, and value magnitude:

$$
\begin{array} { r } { s _ { \mathrm { P r u M e r g e } , i } = \mathrm { s o f t m a x } _ { j = 0 , \ldots , N } \left( q _ { 0 } ^ { \top } k _ { j } / \sqrt { d _ { k } } \right) _ { i } , } \end{array}
$$

$$
e _ { i } = \frac { 1 } { A } \sum _ { a } \mathrm { s o f t m a x } _ { j = 0 , . . . , N } \left( v _ { a , 0 } ^ { \top } v _ { a , j } \right) _ { i } ,
$$

$$
s _ { \mathrm { F l o w C u t } , i } = [ \mathrm { N o r m } ( \boldsymbol { u } ) _ { i } + \mathrm { N o r m } ( \boldsymbol { e } ) _ { i } ] \left\| \frac { 1 } { A } \sum _ { a } \boldsymbol { v } _ { a , i } \right\| _ { 1 } ,\tag{28}
$$

where $\begin{array} { r } { \mathrm { N o r m } ( w ) = w / ( \sum _ { i = 1 } ^ { N } w _ { i } + 1 0 ^ { - 8 } ) } \end{array}$ . In these two archived rank helpers, index 0 is the first tile's CLS and indices $1 , \ldots , N$ span the image-wide patch stream; actual compressor passes remain tile-local. The Q/K/V features are aligned by concatenating the four patch-group slots and repeating CLS to the matching width $d _ { k }$ or $d _ { v }$ . FlowCut uses penultimate-layer values, with no scale or temperature in its value softmax; CLS is dropped before Norm. DivPrune uses $s _ { \mathrm { D i v P r u n e } , t , i } =$ $\mathrm { m i n } _ { j \in \mathcal { U } _ { \mathrm { D i v P r u n e } , t } } [ 1 - \cos ( h _ { t , i } , h _ { t , j } ) ]$ , allowing self-matches. Evaluating these scores on clean features fixes the cutoff lists; their adversarial values enter Eq. 27.

Diversity and relation/value terms. For c ∈ {DivPrune, VisPruner}, the diversity score increases nearest-neighbor similarity within the fixed retained set:

$$
\mathcal { D } _ { \mathrm { d i v } } ^ { c } = \frac { 1 } { J } \sum _ { t } \frac { 1 } { | \mathcal { U } _ { c , t } | } \sum _ { i \in \mathcal { U } _ { c , t } } \operatorname* { m a x } _ { j \in \mathcal { U } _ { c , t } \setminus \{ i \} } \hat { h } _ { t , i } ( x _ { a } ) ^ { \top } \hat { h } _ { t , j } ( x _ { a } ) , \quad \hat { h } = \frac { h } { \operatorname* { m a x } ( \| h \| _ { 2 } , 1 0 ^ { - 1 2 } ) } .\tag{29}
$$

Self-pairs are excluded, and tiles with fewer than two selected tokens are omitted from the mean. The pairwise cosine distance is $1 - \hat { h } _ { i } ^ { \top } \hat { h } _ { j } ;$ Eq. 29 averages each selected token's largest non-self similarity.

The relation term uses FlowCut's selected-token value magnitude. Let $V _ { a , i } ( x _ { a } )$ be the penultimatelayer value of global patch token i in head a, define $\begin{array} { r } { \nu _ { i } = \| A ^ { - 1 } \sum _ { a } V _ { a , i } ( x _ { a } ) \| } \end{array}$ 1, and map the tile-local retained sets into their image-wide union $\mathcal { U } _ { \mathrm { F l o w C u t } }$ . Then

$$
\mathcal { D } _ { \mathrm { r e l } } = - \frac { | \mathcal { U } _ { \mathrm { F l o w C u t } } | ^ { - 1 } \sum _ { i \in \mathcal { U } _ { \mathrm { F l o w C u t } } } \nu _ { i } } { \operatorname* { m a x } \left( N ^ { - 1 } \sum _ { i = 1 } ^ { N } \nu _ { i } , 1 0 ^ { - 6 } \right) } .\tag{30}
$$

Head averaging precedes the feature L1 norm. Normalization divides the selected-token mean by the mean over all adversarial patch tokens, with a denominator floor of $1 0 ^ { - 6 }$ . Gradients propagate through both means.

Attention and redundancy targets. Let $a _ { t , i }$ average CLS attention over heads and the last three vision layers, then average each four-patch group into a post-shuffle token. In each tile, the clean ranking fixes the top $q = \mathrm { r o u n d } ( 2 5 6 \cdot 6 4 / 5 7 6 ) = 2 8$ targets $\mathcal { T } _ { t } .$ the next 28 decoys $\mathcal { E } _ { t }$ , and width-8 lists $\bar { \mathcal { A } } _ { \rho , t } ^ { \bar { \pm } }$ on either side of rank round $( 2 5 6 \rho )$ for $\rho \in \mathcal { R } = \{ 1 / 3 , 1 / 9 , 1 / 1 8 \}$ . Let 〈· average over tiles and corresponding list positions. The attention score is

$$
\begin{array} { r l } & { \mathcal { D } _ { \mathrm { a t t n } } = \mathrm { ~ - ~ } \langle a _ { \mathcal { T } } ( x _ { a } ) \rangle - \langle [ a _ { \mathcal { T } } ( x _ { a } ) - a _ { \mathcal { E } } ( x _ { a } ) ] _ { + } \rangle } \\ & { \mathrm { ~ - ~ } \frac { 1 } { 3 } \displaystyle \sum _ { \rho \in \mathcal { R } } \langle [ a _ { \mathcal { A } _ { \rho } ^ { + } } ( x _ { a } ) - a _ { \mathcal { A } _ { \rho } ^ { - } } ( x _ { a } ) ] _ { + } \rangle . } \end{array}\tag{31}
$$

The redundancy score compares the same targets with the other 228 projected tokens within their tile:

$$
\mathcal { D } _ { \mathrm { r e d } } = \frac { 1 } { J q } \sum _ { t } \sum _ { i \in \mathcal { T } _ { t } } \operatorname* { m a x } _ { j \not \in \mathcal { T } _ { t } } \cos ( p _ { t , i } ( x _ { a } ) , p _ { t , j } ( x _ { a } ) ) .\tag{32}
$$

These target/complement sets differ from the compressor-specific retained sets $\mathcal { U } _ { c , t }$

Semantic preservation and pooling. The semantic hierarchy consists of target tokens, tile means, and an image mean of the projector output. Pooling is arithmetic averaging: pool $( \{ p _ { i } \} _ { i \in S } ) =$ $\begin{array} { r } { | S | ^ { - 1 } \sum _ { i \in S } p _ { i } } \end{array}$ . Define $\bar { p } _ { t } = \bar { 2 5 6 } ^ { - 1 } \sum _ { i } \bar { p _ { t , i } }$ and $\begin{array} { r } { \bar { p } = \bar { N } ^ { - 1 } \sum _ { t , i } { p _ { t , i } } } \end{array}$ . The three levels receive equal weight:

$$
\begin{array} { c l } { \displaystyle \mathcal { L } _ { \mathrm { s e m } } = \frac { 1 } { 3 } \Bigg [ \frac { 1 } { J q } \sum _ { t } \sum _ { i \in \mathcal { T } _ { t } } \big ( 1 - \cos ( p _ { t , i } ( x _ { a } ) , p _ { t , i } ( x ) ) \big ) } \\ { \displaystyle + \frac { 1 } { J } \sum _ { t } \big ( 1 - \cos ( \bar { p } _ { t } ( x _ { a } ) , \bar { p } _ { t } ( x ) ) \big ) + 1 - \cos ( \bar { p } ( x _ { a } ) , \bar { p } ( x ) \big ) \Bigg ] . } \end{array}\tag{33}
$$

Tile and image means include all full-token patch positions and any thumbnail tile, without CLS. Equal tile sizes make the mean of tile means identical to the mean over all N tokens. PCD, semantic, and projected-redundancy cosines use a feature-norm floor of $1 0 ^ { - 8 }$ . Pooling applies to the two semantic means; PCD compares compressed tokens individually.

Complete objective and optimization. For Open tasks, the nonzero budget weights are $\omega _ { d } ( 1 / 9 ) =$ $1 , \omega _ { d } ( 1 / 3 ) = 0 . 4$ , and $\omega _ { d } ( 2 / 9 ) = 0 . 2 5$ . For MC tasks they are $\omega _ { d } ( 1 / 1 8 ) = 1 , \omega _ { d } ( 1 / 9 ) = 0 . 4$ , and $\omega _ { d } ( 1 / 3 ) \stackrel { \cdot } { = } 0 . 2$ Only PCD uses the normalized multibudget combination:

$$
\mathcal { D } _ { \mathrm { p c d } } = \frac { \sum _ { r } \omega _ { d } ( r ) | \mathcal { C } _ { r } | ^ { - 1 } \sum _ { c \in \mathcal { C } _ { r } } \mathcal { D } _ { \mathrm { p c d } } ^ { c , r } } { \sum _ { r } \omega _ { d } ( r ) } .\tag{34}
$$

Here $\mathcal { C } _ { r _ { * } }$ contains all five attack compressors; secondary budgets exclude DivPrune. Rank averages the five practical-budget scores, diversity averages DivPrune and VisPruner, and relation uses FlowCut. The attention and redundancy terms use their fixed targets above. The maximization form is

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \| \delta \| _ { \infty } \leq \epsilon } } & { ~ \mathcal { T } _ { \mathrm { I n t e r n V L } } = 6 \mathcal { D } _ { \mathrm { p c d } } + 3 \left( 2 0 \mathcal { D } _ { \mathrm { r a n k } } \right) + 2 \mathcal { D } _ { \mathrm { d i v } } + 2 \mathcal { D } _ { \mathrm { r e l } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + 4 0 \mathcal { D } _ { \mathrm { a t t n } } + 1 . 5 \mathcal { D } _ { \mathrm { r e d } } - 0 . 0 5 \mathcal { L } _ { \mathrm { s e m } } . } \end{array}\tag{35}
$$

These scores reverse the minimized implementation losses, up to additive constants for PCD, diversity, and redundancy. The single weighted objective is differentiated directly, without separately normalizing or projecting term gradients. Random-start PGD uses

$$
\begin{array} { r } { \delta ^ { ( t + 1 ) } = \Pi _ { [ - \epsilon , \epsilon ] } \left[ \delta ^ { ( t ) } + \alpha \ \mathrm { s i g n } \left( \nabla _ { \delta } \mathcal { T } _ { \mathrm { I n t e r n V L } } \right) \right] , \qquad x _ { a } ^ { ( t + 1 ) } = \mathrm { c l i p } ( x + \delta ^ { ( t + 1 ) } , 0 , 1 ) , } \end{array}\tag{36}
$$

with $\epsilon = 2 / 2 5 5 , \alpha = 0 . 5 / 2 5 5$ , and 100 steps. The random seed for zero-based sample index i is $4 2 + i .$

Additional retention levels. Table 15 supplements Table 6 with additional retention fractions applied uniformly across all four datasets. Damage is the Clean-FATA accuracy difference, and amplification is its increase relative to full-token inference. Both are calculated before rounding.

Table 15: InternVL3.5 accuracy and amplification at additional retention fractions. Each row applies the same fraction to all four datasets and averages over the four compressors.
<table><tr><td>Budget</td><td>Clean</td><td>FATA</td><td>Damage (pp)</td><td>Amplification (pp)</td></tr><tr><td> $2 / 9$ </td><td>80.77</td><td>68.49</td><td>12.28</td><td>7.05</td></tr><tr><td> $1 { \dot { / } } 9$ </td><td>70.78</td><td>56.20</td><td>14.58</td><td>9.34</td></tr><tr><td>1/18</td><td>60.07</td><td>47.74</td><td>12.33</td><td>7.09</td></tr><tr><td>1/36</td><td>49.97</td><td>41.90</td><td>8.06</td><td>2.83</td></tr></table>

The Clean–FATA accuracy gap peaks at $1 / 9$ and narrows as retention falls to $1 / 3 6$

Retained-set diagnostics. For either diagnostic m and a fixed compressor c, macro averaging first gives equal weight to valid images within each dataset, then equal weight to the four datasets:

$$
\bar { m } _ { c } = \frac { 1 } { 4 } \sum _ { d = 1 } ^ { 4 } \left( \frac { 1 } { n _ { d } } \sum _ { i \in \mathcal { I } _ { d } } m _ { i , d , c } \right) .\tag{37}
$$

Here $\mathcal { T } _ { d }$ is the valid-image set and $n _ { d } = \vert \mathcal { T } _ { d } \vert$ . In the order TextVQA-Open, VQAv2-Open, ScienceQA-MC, and VQAv2-MC, the counts are 998, 1000, 1000, 1000. No weighting by token count is used. For Table 16, tile-local selected indices are mapped into the concatenated image-wide token stream before computing each image's diagnostics. Both diagnostics use the final adversarial input at the practical budget $( 1 / 9$ for Open and $\bar { 1 } / 1 8$ for MC).

Table 16: InternVL3.5 retained-set diagnostics at the practical budget. For each image, $S _ { \mathrm { c l e a n } }$ and $S _ { \mathrm { F A T A } }$ contain the selected source-token indices, using representative indices for merging methods. Flip rate is $| S _ { \mathrm { c l e a n } } \setminus S _ { \mathrm { F A T A } } | / | S _ { \mathrm { c l e a n } } |$ , the fraction of clean-selected tokens removed after attack; Jaccard is $| \dot { S } _ { \mathrm { c l e a n } } \cap \dot { S } _ { \mathrm { F A T A } } | / | \dot { S } _ { \mathrm { c l e a n } } \cup S _ { \mathrm { F A T A } } | .$ Values are fractions, macro-averaged by Eq. 37.

<table><tr><td>Compressor</td><td>Flip rate</td><td>Jaccard</td></tr><tr><td>VisionZIP</td><td>0.854</td><td>0.079</td></tr><tr><td>VisPruner</td><td>0.925</td><td>0.040</td></tr><tr><td>FlowCut</td><td>0.890</td><td>0.059</td></tr><tr><td>PruMerge</td><td>0.617</td><td>0.240</td></tr></table>

The retained-set statistics are consistent with substantial changes to the compression decision across the retained compressors, while the family-aware surrogates cover complementary attention-, diversity-, and relation-sensitive selection behavior. These diagnostics do not establish that any single surrogate is solely responsible for the task-level drop.

InternVL3.5 data integrity. The TextVQA-Open exclusions summarized in Appendix B arise from truncated source images 371 and 759, which are omitted consistently from Clean and FATA aggregation. The final formal validation reports no duplicate or missing conditions, no NaN or Inf values, and no perturbation-budget violations. The small set of parser status exceptions is inherited from the same fixed evaluation pipeline and is scored as zero credit without re-running inference.

## H DETECTION DETAILS AND CROSS-ATTACK ANALYSIS

## H.1 ML-ATD CONSTRUCTION

HiddenDetect (Jiang et al., 2025) characterizes jailbreak behavior through directions derived from refusal-related hidden-state changes. Inspired by HiddenDetect, ML-ATD adapts this approach by replacing the refusal-related reference with an attack direction estimated from Base examples and extending feature readout to the visual encoder, projector, and language model. Figure 10 shows reference fitting, feature extraction, trajectories within each stage, and the calculation of detection metrics. Each stage is evaluated in its own feature space.

![](images/4999154bdac900cddad381348a209fc0efe61ffdc217404445077c01b08d68db.jpg)  
Figure 10: ML-ATD reference fitting, feature readout, and stage-wise detection. Visual, Projector, and LLM trajectories are scored and evaluated separately, with stage metrics averaged only for reporting.

Fixed per-tap references (A). Reference and test images use the same feature extraction procedure, with mean pooling followed by $\ell _ { 2 }$ normalization at each observation point, or tap, l, giving fe(x). The negative references, denoted by clean\_clip+random\_clip, consist of clean images and images generated by adding random perturbations to clean images, without filtering by prediction correctness. Base adversarial examples provide the positive references. The mean feature vectors of the positive and negative references at each tap define the fixed direction

$$
\mathbf { d } _ { \ell } = \frac { \boldsymbol { \mu } _ { \ell , \mathrm { B a s e } } - \boldsymbol { \mu } _ { \ell , \mathrm { n e g } } } { \lVert \boldsymbol { \mu } _ { \ell , \mathrm { B a s e } } - \boldsymbol { \mu } _ { \ell , \mathrm { n e g } } \rVert _ { 2 } } .\tag{38}
$$

Reference fitting uses only the negative references and Base examples, excluding FATA examples.

Monitored feature readout (B). A single forward pass through the frozen LLaVA model provides visual patch features from layers 6, 12, 18, and 24, the projector input and output, and LLM activations from layers 8, 16, 24, and 32. Mean pooling covers all patches for visual features and only the image-token span for LLM features. The projector input and output each use mean pooling and $\ell _ { 2 }$ normalization, with separate reference directions and calibration statistics. In the compressed inference path shown in Figure 10, token selection or merging produces $K$ representatives $\dot { Z , }$ which reconstruct an N-slot sequence $\widetilde { H }$ before the projector. The LLM receives the projected visual sequence together with the question.

Standardized projections (C). Each normalized feature is centered on its negative mean, projected onto its own attack direction, and standardized

$$
u _ { \ell } ( x ) = \left( \widetilde { \mathbf { f } } _ { \ell } ( x ) - \pmb { \mu } _ { \ell , \mathrm { n e g } } \right) ^ { \top } \mathbf { d } _ { \ell } , \qquad z _ { \ell } ( x ) = \frac { u _ { \ell } ( x ) - m _ { \ell } ^ { - } } { s _ { \ell } ^ { - } } ,\tag{39}
$$

where $m _ { \ell } ^ { - }$ and $s _ { \ell } ^ { - }$ denote the mean and standard deviation of ue over the negative reference samples. For each stage $r ^ { \mathrm { ^ { \circ } } } \in \{ V , P , L \} , \mathbf { z } _ { r } ( x )$ contains the standardized projections in layer order, with depth normalized to [0, 1] within that stage. Schematic in panel C illustrates how these values vary across layers.

Stage scores and reported metrics (D). ML-ATD integrates the standardized projections over the normalized depth interval using the trapezoidal rule. The resulting area $T _ { r } ( x )$ gives the detection score for stage r after normalization by its mean $a _ { r } ^ { - }$ and standard deviation $b _ { r } ^ { - }$ over the negative reference samples,

$$
T _ { r } ( x ) = { \mathrm { T r a p z } } _ { [ 0 , 1 ] } { \big ( } \mathbf { z } _ { r } ( x ) { \big ) } , \qquad S _ { r } ( x ) = { \frac { T _ { r } ( x ) - a _ { r } ^ { - } } { b _ { r } ^ { - } } } .\tag{40}
$$

Evaluation uses the scores $S _ { V } ( x ) , S _ { P } ( x )$ , and $S _ { L } ( x )$ separately to compute AUROC, AUPR, and TPR@5FPR against the attack and negative labels. For each stage, TPR@5FPR is obtained from its test-set ROC curve at a 5% false positive rate. For each metric $Q ,$ the reported value is the mean across the three stages,

$$
\overline { { { Q } } } = \frac { Q _ { V } + Q _ { P } + Q _ { L } } { 3 } .\tag{41}
$$

## FATA-Base comparison.

Table 17: Detection performance on FATA and Base. Higher values mean easier detection; green arrows show the relative reduction from Base to FATA.
<table><tr><td>Detector</td><td>Attack</td><td>AUROC</td><td>AUPR</td><td>TPR@5FPR</td></tr><tr><td>Feature Squeezing</td><td>FATA Base</td><td>0.560↓15.8% 0.665</td><td>0.382↓22.7% 0.494</td><td>0.024↓62.5% 0.064</td></tr><tr><td>Mahalanobis-Max</td><td>FATA Base</td><td>0.695↓14.5% 0.813</td><td>0.509 ↓22.9% 0.660</td><td>0.134↓51.8% 0.278</td></tr><tr><td>ML-ATD</td><td>FATA Base</td><td>0.828↓13.0% 0.952</td><td>0.724↓21.1% 0.918</td><td>0.497↓37.9% 0.801</td></tr></table>

Table 17 isolates the FATA and Base rows of the main-text comparison in Table 8; the green arrows report relative reductions from Base to FATA. At a 5% false positive rate, FATA reduces TPR by 62.5% under Feature Squeezing (from 6.4% to 2.4%), by 51.8% under Mahalanobis-Max (from 27.8% to 13.4%), and by 37.9% under ML-ATD (from 80.1% to 49.7%, with the reduction computed before rounding). Thus, FATA is less detectable than Base across all three detectors at this operating point.

## H.2 DETECTION EVALUATION

Baseline settings and aggregation. The results in Tables 8 and 17 use two negatives per positive attack example. Feature Squeezing (FS) evaluates inputs at $K = 5 7 6$ using the unthresholded

MaxFS score (max\_fs\_score\_diff), which measures changes in task credit against groundtruth answers before and after input transformations. Mahalanobis-Max uses the per-image maximum of standardized CLS and mean-pooled patch distances in CLIP space, with 128-dimensional PCA, clean fitting at indices 0–499, and evaluation at 500–999. Both baselines use seed 0, without attack-specific score selection.

FS averages 16 dataset-compressor combinations with identical full-token scores across compressors, making this equivalent to averaging four datasets. Mahalanobis-Max combines matching records across compressors before averaging four datasets. ML-ATD reports an equal-weight mean over four compressors (VisionZIP, VisPruner, PruMerge, and FlowCut), four datasets, and the three stages. Its detection runs use K = 64 for open-ended tasks and K = 32 for multiple-choice tasks.

Input consistency versus feature deviation. The four-attack comparison in Table 8 shows that detection performance depends on the type of deviation each detector measures. In particular, Mahalanobis-Max orders CAGE > CAA > Base > FATA on all three metrics. Although FS gives similar detection performance for FATA and CAA, Mahalanobis-Max detects FATA less readily. This lower detectability in feature space is consistent with FATA's semantic preservation objective.

Evaluation sample coverage. CAA evaluation on ScienceQA includes 990 attack examples for FS and 496 for Mahalanobis-Max, with 1,980 and 992 aligned negatives, respectively. Other conditions use nominal counts.