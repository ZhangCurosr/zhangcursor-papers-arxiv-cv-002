# FastJEV: Understanding Redundancy for Compact JEV Inference

Jie Ma, Jie Gao, Yihang Liu, Zhike Qiu, Junle Li, Chongyi Zhuang, Jiayi Ji, Xiaoshuai Sun

Xiamen University

JEV models make multimodal decisions by directly scoring candidates. Although the common context is encoded once, candidate evaluation can still repeat matching token histories, duplicate inference states, and execute the full backbone. In this paper, we study these sources of redundancy and present FastJEV for compact candidate evaluation. We jointly organize history reuse and state storage, since sharing computation requires preserving states for later branches. We first introduce shared context anchoring to reuse recurrent initial states and omit unused final recurrent caches. We extend this reuse through candidate prefix sharing, retaining the intermediate states needed by subsequent branches. To further reduce the depth of these paths, we apply decision guided pruning based on relative score changes measured on a small unlabeled set. Our method retains full context encoding and all candidates without additional training. We evaluate FastJEV across three OmniJev model sizes on five public benchmarks and reconstructed LIBERO-10 ofline questions. At the selected pruning budgets, the complete method reduces candidate depth by 43.75% to 45.83%, while retaining 93.66% to 97.52% of the original task scores on average across the six evaluation sets. Through controlled experiments, we show how candidate overlap and branching structure afect the execution cost of history reuse. In our implementation, candidate prefix sharing can reduce repeated computation while increasing latency. These findings motivate designing sharing granularity and execution schedules together for eficient JEV inference.

Date: October 9, 2026 Correspondence: jiema100@stu.xmu.edu.cn, xssun@xmu.edu.cn Code: https://github.com/JieMaMagic/FastJEV

![](images/37ce61af9b6525aa31fcb79cbd1e46f0e76b6f368accf496ba7418115268bbde.jpg)

## 1 Introduction

Multimodal models connect visual perception with language understanding, supporting tasks such as recognition, question answering, and instruction following (Radford et al., 2021; Liu et al., 2023b; Li et al., 2023a). Their capabilities also extend to embodied settings, where visual observations and language instructions guide action selection (Zitkovich et al., 2023; Team, 2025). Across these settings, a central role of multimodal models is to turn observations into task decisions. JEV makes this role explicit by directly scoring candidate choices without autoregressive answer generation (Xu et al., 2026). This approach places decision computation at the center of inference. Understanding how that computation can be organized compactly is an important step toward eficient multimodal decision models.

Research on eficient inference has explored both computation reuse and selective reduction. One line of work shares common prefixes and cached states to reduce repeated computation and duplicate storage (Kwon et al., 2023; Zheng et al., 2024). Another reduces computation according to estimated redundancy, either by pruning layers (Men et al., 2025; Song et al., 2024) or by shortening visual token sequences (Chen et al., 2024). More selective strategies use output sensitivity to bypass visual attention operations while preserving feed-forward updates (Ma et al., 2026). Together, these directions motivate studying how computation can be shared and reduced within multimodal decision models.

In direct candidate scoring, a common multimodal context can be encoded once and reused across alternatives (Xu et al., 2026). Candidate paths can still repeat matching histories, duplicate inference states, and execute the full backbone. These costs are closely related. Reusing a history removes repeated computation but requires retaining states for its later branches. Reducing depth changes the computation along both shared paths and their continuations, and its efect must be assessed through the final candidate scores. How can we organize candidate histories, inference states, and execution depth into a compact structure while preserving decision quality?

To address this question, we present FastJEV, which connects shared context anchoring, candidate prefix sharing, and decision guided pruning. We first use the context as a common anchor, sharing its recurrent initial states across candidate paths and omitting unused final recurrent caches. Candidate prefix sharing extends this reuse through a trie of matching token histories. Each shared history is evaluated once, and states needed by later branches are retained. We then select layers for pruning by measuring how skipping each layer changes relative candidate scores. A small unlabeled calibration set provides a fixed layer ranking for each model. For each pruning budget, we apply the selected mask to every candidate path and reuse it across tasks. FastJEV retains full context encoding and all answer candidates without additional training.

We evaluate FastJEV using the 0.8B, 2B, and 4B models of OmniJev on five public benchmarks (Kembhavi et al., 2016; Lu et al., 2022; Li et al., 2023b; Fu et al., 2024; Team, 2025) and reconstructed LIBERO-10 ofline questions (Liu et al., 2023a). At the selected pruning budgets, the complete method reduces candidate depth by 43.75% to 45.83%. These configurations retain 93.66% to 97.52% of the original task scores on average across the six evaluation sets. Controlled experiments show how history sharing reduces repeated candidate computation and how state management reduces live state storage. They further reveal that execution cost depends on candidate overlap and branching structure. With independent batched candidate evaluation, shared context anchoring and pruning achieve speedups of 1.053× to 1.258× at the default six-layer pruning budget on one NVIDIA L20 GPU. These findings provide a structural view of JEV inference and clarify when compact candidate evaluation improves execution eficiency.

Our contributions are summarized as follows.

• We provide a structural analysis of JEV candidate evaluation that connects repeated token histories, inference state storage, and execution depth.

• We present FastJEV, which combines shared context anchoring, candidate prefix sharing, and decision guided pruning for compact candidate evaluation. The method retains full context encoding and all candidates without additional training.

• We evaluate FastJEV across three model sizes and six evaluation sets. At the selected pruning budgets, the complete method reduces candidate depth by 43.75% to 45.83%, with mean task score retention of 93.66% to 97.52%. Controlled analyses examine how candidate overlap and branching structure afect execution cost.

## 2 Related Work

## 2.1 Efficient Multimodal Inference

Multimodal inference can produce decisions through representation matching (Radford et al., 2021) or autoregressive generation (Liu et al., 2023b; Li et al., 2023a). Direct candidate scoring provides another formulation, in which a shared context supports the comparison of explicit alternatives (Xu et al., 2026). These formulations expose diferent sources of inference cost. For generative vision-language models, selective execution often targets the visual inputs. Attention importance can guide visual token pruning (Chen et al., 2024), while output sensitivity can guide selective skipping of visual attention (Ma et al., 2026). FastJEV studies the computation required to compare candidates after a common context has been encoded. It retains full context processing and all candidates, and organizes the remaining computation through history sharing, state reuse, and depth reduction.

## 2.2 Prefix and State Sharing

Prefix reuse reduces repeated computation when multiple requests or branches share an input history. Blockbased KV cache management supports shared storage across requests and decoding branches (Kwon et al., 2023). Tree-based prefix indexing extends cache reuse to structured language model programs (Zheng et al.,

2024). In hybrid backbones, recurrent layers introduce state representations beyond attention KV caches (Yang et al., 2025). Recent work compresses the recurrent state information needed for speculative verification and reconstructs the accepted state (Ghantasala, 2026). FastJEV connects history reuse and state management within a candidate scoring request. The common context supplies a shared recurrent anchor, and a candidate trie extends reuse to matching sufix histories. State retention follows the tree dependencies, preserving continuation states and omitting unused final recurrent caches. This organization links the computation shared by candidates to the states needed for their subsequent evaluation.

## 2.3 Structured Pruning

Structured pruning reduces computation by removing coupled parameter groups or complete layers. One approach uses gradient information to prune coupled structures, followed by parameter-eficient recovery (Ma et al., 2023). Layer-level approaches estimate redundancy from changes in hidden representations (Men et al., 2025) or changes in predictive loss after block removal (Song et al., 2024). Output sensitivity also guides visual attention skipping through divergence between vocabulary distributions, while feed-forward updates are retained (Ma et al., 2026). FastJEV follows this sensitivity-based perspective, with relative candidate scores as the decision signal. We rank layers using a small unlabeled calibration set and bypass selected complete blocks only during candidate evaluation. The same selected layers apply to shared histories and their continuations, connecting depth reduction with prefix and state reuse. Full context encoding is retained, and the method requires no additional training.

## 3 Method

Overview. FastJEV organizes candidate evaluation into a compact structure, as illustrated in Figure 1. We connect the reuse of states and matching histories with a reduction in candidate execution depth. Shared context anchoring (Section 3.2) retains full context encoding and shares recurrent initial state storage across candidate paths. However, starting from the same context does not eliminate repeated computation over matching candidate prefixes. Candidate prefix sharing (Section 3.3) therefore extends reuse beyond the initial context by evaluating matching token histories only once. This sharing removes duplicate prefix computation, but each remaining segment still traverses the full backbone. Decision guided pruning (Section 3.4) then reduces candidate depth by bypassing layers selected according to their efect on final scores.

## 3.1 Preliminaries on JEV

For visual tasks, OmniJev combines a visual encoder, a language backbone, and a decision head (Xu et al., 2026). The visual encoder maps images to features that the language backbone processes with the question and answer candidates. The decision head then assigns a score to each candidate. We use OmniJev, whose Qwen3.5 backbone combines full attention and Gated DeltaNet layers (Yang et al., 2025).

Given images I, a question q, and candidates $\mathcal { A } = \{ a _ { i } \} _ { i = 1 } ^ { N }$ , OmniJev forms a shared prefix $P$ and candidate sufixes $R _ { i }$ . The prefix contains the shared visual and text context. Each sufix contains the tokens needed to score one candidate, including its boundary markers. Each sufix retains the token position used to read out the question representation. The full hybrid backbone encodes $P$ once and produces a prefix cache $\mathcal { C } _ { P }$ . For candidate i, we compute the hidden outputs from the prefix cache as

$$
H _ { i } = F _ { \theta } ( R _ { i } ; { \mathcal { C } } _ { P } ) ,\tag{1}
$$

where $F _ { \theta }$ denotes the language backbone with pretrained weights $\theta ,$ and $H _ { i }$ contains the hidden outputs of branch i. The original readout $g _ { \theta }$ combines question and candidate representations with language model features to produce

$$
\mathbf { z } = g _ { \boldsymbol \theta } ( H _ { 1 } , \dots , H _ { N } ) .\tag{2}
$$

For multiple choice, $\mathbf { z } \in \mathbb { R } ^ { N + 1 }$ contains the candidate logits and one abstention logit. The latter is produced by the head without an additional backbone branch.

This structure separates shared context encoding from the work needed to score each candidate. The hybrid implementation reuses prefix computation but copies its recurrent states when creating branches. Matching

![](images/9994668d2f40c0bcdc877671259a4753fb7ebed91fa5542f2f64ff398b4de7bc.jpg)  
Figure 1 Overview of FastJEV, illustrated with an ERQA example (Team, 2025) on OmniJev (Xu et al., 2026). (a) Shared context anchoring reuses the encoded context and shares recurrent initial state storage. (b) Candidate prefix sharing merges matching token histories and retains continuation caches at forks. (c) Decision guided pruning bypasses selected candidate layers using a fixed mask determined through ofline sensitivity assessment. (d) The original scoring procedure produces candidate logits in their original order and an abstention logit. Full context encoding and all candidate tokens are retained. Phrase boxes group consecutive tokens, and layer positions are schematic.

token histories within candidate sufixes can also incur repeated computation. Every branch also runs all L language layers. We use this separation to organize candidate histories, state storage, and execution depth around a fully encoded context.

## 3.2 Shared Context Anchoring

On the hybrid backbone, all candidates begin with the same encoded history. We treat the complete prefix cache as a common anchor and extend its reuse from prefix computation to recurrent state storage. The anchor is the cache already produced by the original full backbone. It requires no learned parameters or additional context encoding.

Let $S _ { P } ^ { \ell }$ be the recurrent state produced by the prefix at Gated DeltaNet layer ℓ. We set the initial state of candidate i at layer ℓ as

$$
S _ { i , 0 } ^ { \ell } = S _ { P } ^ { \ell } .\tag{3}
$$

We implement this common initial condition through a single read-only bufer. Each branch reads the bufer and independently updates its own working state. Sharing is valid when the visual and token history, positions, and masks are identical. The initial value and branch updates match the original recurrence in exact arithmetic. Convolution states and attention KV caches retain separate storage for each branch.

The anchor also clarifies which states must persist during scoring. The prefix cache is retained because all candidate batches depend on it. Once a candidate path ends and no continuation depends on it, its final recurrent cache has no further use. We therefore omit its allocation and writeback, while still executing every recurrent update needed for scoring. This includes the last update of each executed recurrent layer. Intermediate caches needed to continue a candidate path are retained.

Context anchoring removes repeated state storage but still processes candidate sufixes separately. When these sufixes contain matching histories, their token computations ofer a further opportunity for reuse. We next extend sharing to these histories while retaining the state needed at each fork.

## 3.3 Candidate Prefix Sharing

Candidates may share an initial token sequence beyond the common context P. Evaluating this sequence separately repeats the same transformation at each language layer. We organize candidate sufixes into a prefix tree, or trie, and evaluate each shared node once per executed layer. The tree starts from the common context anchor and carries the encoded history forward as candidate paths diverge.

History Consistent Merging. Let ${ \boldsymbol { r } } _ { i , t }$ be the token at position t of candidate sufix $R _ { i } ,$ , including its boundary markers. Let $\mathbf { p } _ { i , t }$ contain its three original position IDs, and let $v _ { i , t - 1 }$ denote the tree node for its preceding history. All sufixes start from the same root, which represents the common context $P .$ . We assign the token to a node using the key

$$
\kappa _ { i , t } = \big ( v _ { i , t - 1 } , r _ { i , t } , \mathbf { p } _ { i , t } \big ) .\tag{4}
$$

Two occurrences share a node only when their keys are identical. The parent node identifies the complete preceding path, so equal tokens after diferent histories remain separate. Position IDs are retained from the unmerged input rather than reassigned after merging. All sufixes use the same starting position in the branch layout, so tree depth determines these position IDs.

Anchored Tree Execution. We execute the tree as path segments that end at a fork or a candidate boundary. Each segment starts from the cache for its parent history. This cache contains the attention keys and values, convolution states, and recurrent states of the retained layers. Segments with the same parent can be batched and share its recurrent state as their initial anchor. Each segment then updates its own working state. Segments with descendants retain their final caches for continuation. Only leaf segments may omit final recurrent cache storage.

The inherited cache and the causal mask within each segment restrict each token to $P ,$ its ancestors, and itself. For a fixed set of executed layers, merged occurrences retain the same visible history and positions as in separate candidate evaluation. Their hidden outputs are therefore unchanged in exact arithmetic during inference. Finite precision can introduce diferences when batch shapes change.

Candidate Preserving Readout. The decision head requires a separate entry for every candidate. We retain a mapping from each token occurrence in $R _ { i }$ to its tree node. This mapping recovers the question and candidate representations required by $g _ { \theta }$ . For the language model features, we project each required predictor node once and normalize over the full vocabulary. We gather the token log probabilities along each candidate path to construct its original language model features. The gathered representations and features enter the unchanged decision head in the original candidate order. The head produces the candidate logits and the abstention logit. Candidates that share a complete tree path remain separate entries in the readout.

## 3.4 Decision Guided Pruning

Prefix sharing reduces repeated evaluation of matching token histories, but leaves the depth of each candidate path unchanged. We further reduce this depth by selecting layers according to their efect on final scores. The resulting mask applies to all candidate paths, while shared context encoding retains the complete backbone.

Decision Sensitivity. The decision head combines several features. Changes in intermediate representations alone do not specify their efect on candidate scores. We therefore assess each layer through the output of the complete scoring process in Eq. (2).

Let D be a fixed calibration set of inputs $x = ( I , q , { \mathcal { A } } )$ . Layer selection does not use the answer labels of these inputs. For each $x \in \mathcal { D }$ , let $\mathbf { z } ( x )$ be the original logits and $\mathbf { z } ^ { ( - \ell ) } ( x )$ the logits obtained by skipping only layer ℓ in candidate branches. The prefix still uses every layer in both runs. Both vectors include the abstention logit and are measured before probability postprocessing. We write their diference as $\Delta _ { \ell } ( x ) = \mathbf { z } ^ { ( - \ell ) } ( x ) - \mathbf { z } ( x )$

A common shift of all logits leaves their diferences and softmax probabilities unchanged. We remove this shift using $\mathrm { c t r } ( v ) = v - \mathrm { m e a n } ( v ) { \bf 1 }$ , where 1 is the all-ones vector. For an m-dimensional vector, let

![](images/4d2c53d5ec2301092b1b94cbdca5a7a72dff864edec0078ba09d25728cf9a976.jpg)  
Figure 2 Decision sensitivity across layers in OmniJev. Each point is the mean relative score perturbation $J _ { \ell }$ over 32 multiple-choice calibration inputs when a single candidate block is skipped. The six lowest-scoring layers are highlighted for the default pruning budget. Layer indices start at one, and the vertical axis uses a logarithmic scale. Context encoding retains all layers.

rm $\mathfrak { s } ( v ) = \| v \| _ { 2 } / \sqrt { m } .$ . We measure layer sensitivity by

$$
d _ { \ell } ( x ) = \frac { \mathrm { r m s } ( \mathrm { c t r } ( \Delta _ { \ell } ( x ) ) ) } { \mathrm { m a x } ( \epsilon , \mathrm { r m s } ( \mathrm { c t r } ( { \bf z } ( x ) ) ) ) ) } ,\tag{5}
$$

where $\epsilon = 1 0 ^ { - 8 }$ . The denominator accounts for score scale diferences across inputs. This criterion evaluates changes in relative scores, including changes that may leave the highest scoring candidate unchanged.

Calibration Guided Selection. The sensitivity $d _ { \ell } ( x )$ measures the efect of skipping layer ℓ for one input. To select a fixed pruning mask, we average this quantity over D and define the layer score as

$$
J _ { \ell } = \frac { 1 } { | \mathcal { D } | } \sum _ { x \in \mathcal { D } } d _ { \ell } ( x ) ,\tag{6}
$$

where |D| is the number of calibration inputs. A smaller $J _ { \ell }$ indicates a smaller average perturbation of relative candidate scores on D. Given a pruning budget k, we form M from the k layers with the smallest $J _ { \ell } ,$ with ties resolved by layer index. We score layers individually and select all k layers from the resulting ranking. We select M once for each model and reuse it for subsequent inputs. This calibration requires forward passes only and leaves the model weights unchanged.

Context Preserving Execution. Once M is fixed, we apply it only to candidate branches. Shared segments and their continuations use the same mask. The prefix continues to execute all layers and supplies the context cache for every retained block. Let $H _ { i } ^ { \ell }$ denote the hidden outputs of branch i after layer $\ell ,$ with $H _ { i } ^ { 0 }$ denoting its input embeddings. For $\ell = 1 , \ldots , L$ , we compute these outputs as

$$
\begin{array} { r } { H _ { i } ^ { \ell } = \left\{ \begin{array} { l l } { H _ { i } ^ { \ell - 1 } , } & { \ell \in \mathcal { M } , } \\ { \mathcal { B } _ { \ell } ( H _ { i } ^ { \ell - 1 } ; \mathcal { C } _ { P } ^ { \ell } ) , } & { \ell \notin \mathcal { M } , } \end{array} \right. } \end{array}\tag{7}
$$

where $B _ { \ell }$ is the original language block and $\mathcal { C } _ { P } ^ { \ell }$ is its prefix cache. A selected block is replaced by an identity mapping, skipping both its attention or recurrent module and its feed-forward network.

At inference, FastJEV encodes the common context with the full backbone and uses its cache as the anchor for candidate evaluation. Matching candidate histories share computation through the trie. Each segment executes the retained blocks and preserves the states needed by its descendants. The node mapping then recovers the features of every candidate for the original readout. Together, these steps connect history sharing, state reuse, and depth reduction within a single scoring request. All answer candidates remain available, and all model weights are retained for full context encoding.

Empirical Justification. Figure 2 visualizes the calibration scores $J _ { \ell }$ for three OmniJev model sizes. Low sensitivity occurs in parts of the later backbone, while the final block has the highest sensitivity at every scale. Thus, sensitivity does not decrease uniformly with depth. This pattern motivates selecting layers by measured changes in relative candidate scores.

## 4 Experiments

## 4.1 Experimental Setup

Models and data. We use the released 0.8B, 2B, and 4B models of OmniJev v1.1 (Xu et al., 2026). The public benchmarks comprise AI2D test (Kembhavi et al., 2016), ScienceQA-IMG test (Lu et al., 2022), POPE (Li et al., 2023b), BLINK validation (Fu et al., 2024), and ERQA (Team, 2025). They contain 3,088, 2,017, 9,000, 1,901, and 400 questions, respectively. We also construct an ofline question set from LIBERO-10 demonstrations (Liu et al., 2023a). It contains 188 frames from 20 evaluation episodes across ten tasks, with eight question types per frame. The 1,504 questions follow the public OmniJev question schema and available templates. Their labels use simulator states and fixed trajectory rules. The original question pool is unavailable, so LIBERO results measure ofline question answering accuracy on this reconstructed set.

Configurations. T denotes candidate prefix sharing, S denotes shared context anchoring, and P denotes decision guided pruning. Full FastJEV denotes T+S+P, and S+P is the ablation without T. Original uses the released inference path, which already computes the common visual and text prefix once. All configurations retain the original candidates and decision heads. We compute one sensitivity ranking per model on 32 OK-VQA inputs (Marino et al., 2019), each with 64 fixed candidates drawn from the training answer vocabulary. Layer ranking uses neither the calibration answer labels nor the target evaluation labels. For each pruning budget, we select a fixed mask from this ranking and reuse it across tasks without further training. The default budget skips six candidate blocks, reducing candidate depth by 25% for 0.8B and 2B and by 18.75% for 4B. Larger budgets extend the same ranking. The common prefix still executes every block, and all weights remain resident

Evaluation protocol. We report accuracy on each complete evaluation split. Public multiple-choice benchmarks select the highest-scoring provided option. Binary questions use the original sigmoid head with a probability threshold of 0.5. The LIBERO protocol includes abstention in the choice argmax and counts it as an error. Mean retention is the unweighted mean of six accuracy ratios to the matched Original. Ratios are computed before rounding and are not clipped at 100%.

Cost measurement. Performance is measured with exclusive access to one NVIDIA L20 GPU using a BF16 backbone, SDPA, and the same PyTorch convolution backend. Public benchmarks use 64 fixed questions for three timing rounds and 16 questions for memory measurement. LIBERO uses 20 frames for three timing rounds and ten frames for memory measurement. Each frame is evaluated through two requests covering its eight questions. Latency includes image loading and preprocessing, context encoding, candidate execution, and scoring, with CUDA synchronization at the boundaries. It excludes model loading, warmup, configuration installation, and result serialization. We report the largest PyTorch peak allocated memory across the measured inputs, including model weights. After warmup, configurations are timed in an interleaved order determined by a fixed random seed.

## 4.2 Compact Candidate Evaluation across Tasks

Quality under candidate depth reduction. Table 1 evaluates full FastJEV at four pruning budgets for each model scale. Pruning 45.83% of candidate layers gives mean task score retention of 97.52% for 0.8B and 94.92% for 2B. For 4B, pruning 43.75% retains 93.66% of the Original score on average. These results show that substantial candidate depth reduction is possible within the shared structure. Increasing pruning beyond 56% lowers mean retention to between 73.81% and 78.76% across the tested configurations. The six-block rows use the default-budget experiment, while the larger budgets use their matched sweep references.

Quality at the default budget. At six skipped blocks, mean retention is 100.97%, 98.98%, and 100.21% for 0.8B, 2B, and 4B, respectively. The absolute accuracy change is below one percentage point in 13 of the 18 comparisons. For 4B, AI2D changes from 84.33% to 84.29%, BLINK remains at 64.33%, and ERQA improves from 44.75% to 46.50%. LIBERO accuracy decreases by 2.66 and 1.46 percentage points for 2B and 4B. Appendix A.2 reports episode bootstrap intervals and POPE F1 scores. Section 4.6 examines task sensitivity at larger budgets.

Table 1 Decision quality of OmniJev and FastJEV, grouped by model scale. All FastJEV rows use T+S+P. Pruning applies to candidate evaluation, while context encoding uses all layers. Dataset columns report accuracy (%). Mean retention averages six accuracy ratios to the matched Original. Bold pruning and retention entries mark the largest tested budget with mean retention of at least 90%. SQA denotes ScienceQA-IMG, and LIBERO denotes reconstructed LIBERO-10 ofline QA.
<table><tr><td></td><td></td><td>Pruning</td><td colspan="6">Accuracy (%)</td><td></td></tr><tr><td>Method</td><td>Pruned layers</td><td>(%)</td><td>AI2D</td><td>SQA</td><td>POPE</td><td>BLINK</td><td>ERQA</td><td>LIBERO</td><td>Mean ret. (%)</td></tr><tr><td colspan="10">0.8B (24 layers)</td></tr><tr><td>OmniJev</td><td>-</td><td>0.00</td><td>62.27</td><td>69.01</td><td>84.76</td><td>43.98</td><td>28.75</td><td>55.45</td><td>100.00</td></tr><tr><td>FastJEV</td><td>6</td><td>25.00</td><td>62.37</td><td>69.31</td><td>87.79</td><td>43.40</td><td>29.50</td><td>55.65</td><td>100.97</td></tr><tr><td>FastJEV</td><td>8</td><td>33.33</td><td>62.37</td><td>68.86</td><td>88.17</td><td>42.92</td><td>31.00</td><td>56.38</td><td>101.85</td></tr><tr><td>FastJEV</td><td>11</td><td>45.83</td><td>58.19</td><td>66.63</td><td>88.50</td><td>41.45</td><td>28.00</td><td>54.92</td><td>97.52</td></tr><tr><td>FastJEV</td><td>14</td><td>58.33</td><td>31.31</td><td>41.30</td><td>50.00</td><td>38.77</td><td>31.75</td><td>49.53</td><td>76.17</td></tr><tr><td colspan="10">2B (24 layers)</td></tr><tr><td>OmniJev</td><td>-</td><td>0.00</td><td>75.00</td><td>83.79</td><td>88.83</td><td>52.24</td><td>36.75</td><td>58.24</td><td>100.00</td></tr><tr><td>FastJEV</td><td>6</td><td>25.00</td><td>74.90</td><td>83.84</td><td>88.80</td><td>51.13</td><td>37.00</td><td>55.59</td><td>98.98</td></tr><tr><td>FastJEV</td><td>8</td><td>33.33</td><td>74.84</td><td>83.94</td><td>89.40</td><td>51.50</td><td>36.50</td><td>56.78</td><td>99.28</td></tr><tr><td>FastJEV</td><td>11</td><td>45.83</td><td>70.24</td><td>77.84</td><td>89.33</td><td>49.40</td><td>35.00</td><td>54.12</td><td>94.92</td></tr><tr><td>FastJEV</td><td>14</td><td>58.33</td><td>50.52</td><td>52.55</td><td>75.86</td><td>41.50</td><td>32.00</td><td>35.57</td><td>73.81</td></tr><tr><td colspan="10">4B (32 layers)</td></tr><tr><td>OmniJev</td><td>-</td><td>0.00</td><td>84.33</td><td>94.30</td><td>90.77</td><td>64.33</td><td>44.75</td><td>65.23</td><td>100.00</td></tr><tr><td>FastJEV</td><td>6</td><td>18.75</td><td>84.29</td><td>94.10</td><td>90.64</td><td>64.33</td><td>46.50</td><td>63.76</td><td>100.21</td></tr><tr><td>FastJEV</td><td>10</td><td>31.25</td><td>83.78</td><td>93.85</td><td>90.53</td><td>64.07</td><td>42.25</td><td>62.83</td><td>98.16</td></tr><tr><td>FastJEV</td><td>14</td><td>43.75</td><td>81.96</td><td>92.56</td><td>90.46</td><td>64.70</td><td>40.00</td><td>50.20</td><td>93.66</td></tr><tr><td>FastJEV</td><td>18</td><td>56.25</td><td>68.30</td><td>76.10</td><td>90.00</td><td>59.50</td><td>33.00</td><td>29.65</td><td>78.76</td></tr></table>

From compactness to execution cost. At the default six-block budget, full FastJEV lowers total peak GPU memory in six of the 18 comparisons. It is faster than Original on POPE at all three model scales and slower on the other five datasets. On LIBERO, peak memory decreases by 15.70%, 8.43%, and 9.12% for 0.8B, 2B, and 4B, respectively, with higher latency at each scale. These measurements include the efects of state management, segmented execution, and readout scheduling. At the larger budgets selected in Table 1, the geometric mean speedups over the six sets are 0.938, 0.939, and 0.953 for the three model scales. Table 7 reports all default-budget measurements. The following ablations examine how structural compactness relates to these execution costs.

## 4.3 Component Interaction and Efficient Execution

Interactions within the complete structure. We compare all eight combinations of T, S, and P with the same recurrent kernel. The baseline C evaluates every candidate layer, copies the initial recurrent states, and retains the final caches. It difers from the released Original. Table 2 presents the 4B results on AI2D and LIBERO, with P fixed to six skipped blocks when enabled. With T and P fixed, S lowers total peak memory by 104.8 MiB on AI2D and 224.6 MiB on LIBERO, with unchanged accuracy. With T and S fixed, P reduces latency on both datasets, while LIBERO accuracy changes from 64.89% to 63.76%. S and P thus reduce state storage and executed depth within the shared candidate paths. The comparison with and without T also changes segmentation and vocabulary projection scheduling, as detailed in Appendix A.

Understanding compactness and efficiency. T merges repeated candidate histories, S shares the states from which child paths continue, and P shortens the computation along these paths. The complete structure combines these operations to reduce repeated histories, initial-state copies, and executed candidate layers.

Table 2 Factorial comparison with 4B using a common recurrent kernel. C copies initial states, retains final caches, and executes full candidate depth. P skips six candidate blocks when enabled. C difers from the released Original in Table 1. LIBERO latency is per frame. Memory denotes the total allocated peak. Bold marks the best value in each column.
<table><tr><td></td><td colspan="3">AI2D</td><td colspan="3">LIBERO-10 QA</td></tr><tr><td>Mode</td><td>Acc. (%) ↑</td><td>ms↓</td><td>MiB↓</td><td>Acc. (%) ↑</td><td>ms↓</td><td>MiB↓</td></tr><tr><td>C</td><td>84.23</td><td>132.51</td><td>9454.1</td><td>64.96</td><td>324.58</td><td>10381.8</td></tr><tr><td>T</td><td>84.20</td><td>175.74</td><td>9525.0</td><td>64.89</td><td>838.51</td><td>9689.8</td></tr><tr><td>S</td><td>84.23</td><td>130.82</td><td>9261.6</td><td>64.96</td><td>319.20</td><td>9508.0</td></tr><tr><td>P</td><td>84.36</td><td>125.42</td><td>9453.0</td><td>63.90</td><td>296.14</td><td>10356.8</td></tr><tr><td>T+S</td><td>84.20</td><td>174.98</td><td>9419.0</td><td>64.89</td><td>833.27</td><td>9477.9</td></tr><tr><td>T+P</td><td>84.29</td><td>163.45</td><td>9524.0</td><td>63.76</td><td>725.58</td><td>9685.9</td></tr><tr><td>S+P</td><td>84.36</td><td>125.05</td><td>9260.5</td><td>63.90</td><td>294.16</td><td>9483.9</td></tr><tr><td>T+S+P</td><td>84.29</td><td>161.41</td><td>9419.2</td><td>63.76</td><td>721.34</td><td>9461.3</td></tr></table>

(a) History reuse  
![](images/308581ccb09dcca6eeadcacd655ecd5ec82eed7dc2f2c332870b99a57fb7d598.jpg)  
Logical tokens removed (%)

(b) State sharing  
![](images/043b37259ac8966e6b6f6d466f3b58b4c9051f385c5bf3506b7ebbea22bdb233.jpg)  
Tracked state storage (MiB)

(c) Execution cost  
![](images/371919ef392ef9390bdce507a220608d6069094133f6e6a31eb0f8df3ec55124.jpg)  
Figure 3 Structural compactness and execution cost with 4B. (a) Logical candidate token reduction across the six complete evaluation sets. SQA-IMG denotes ScienceQA-IMG, and LIBERO QA is our reconstructed ofline set. (b) Observed live state storage for four inputs, with T and six-layer P fixed. Copy / Keep and Share / Drop follow Table 3. (c) Full-request latency and total peak GPU memory of T+S+P relative to S+P across all 96 artificial workload conditions. Colors indicate candidate length, and marker shapes indicate branching structure. Ratios below one indicate lower cost. Artificial workloads evaluate execution cost only.

Figure 3 relates history and state compactness to execution cost. S+P removes history merging and evaluates candidate sufixes directly in batches. Across six datasets and three model scales, it achieves 1.053× to 1.258× speedups over Original in all 18 comparisons. Its 2B LIBERO accuracy changes from 58.24% to 55.85%. Three independent processes on 4B AI2D and LIBERO preserve the same latency ordering, with $\mathrm { S } { \mathrm { + P } }$ fastest, followed by Original and T+S+P. Appendix A.1 reports the means and standard deviations. We next examine the storage savings from S and the conditions under which the shared execution of T reduces cost.

## 4.4 State Storage within the Shared Structure

We fix T and the six-layer pruning mask and vary two state policies. Initial recurrent states are copied or shared, and final recurrent caches at leaves are retained or omitted. All four configurations retain internal trie states for continuation. We use the AI2D inputs with the minimum and maximum segment counts in the timing subset and the first two frames in the LIBERO timing list.

Table 3 counts live states and caches at program observation points, deduplicates their underlying storage, and reports the maximum observed value. Share/Drop lowers this peak in all four cases. For LIBERO A, it falls from 544.1 MiB with Copy/Keep to 253.7 MiB. All 16 runs produce the same candidate choices and probabilities as the complete method on their corresponding input.

Table 3 Observed peak live state and cache storage (MiB) with 4B. A and B denote two representative inputs per dataset. T and six-layer P are fixed. Bold marks the lowest storage per input.
<table><tr><td colspan="2">State policy</td><td colspan="2">AI2D</td><td colspan="2">LIBERO-10 QA</td></tr><tr><td>Initial</td><td>Leaf</td><td>A</td><td>B</td><td>A</td><td>B</td></tr><tr><td>Copy</td><td>Keep</td><td>351.2</td><td>345.7</td><td>544.1</td><td>545.1</td></tr><tr><td>Copy</td><td>Drop</td><td>343.3</td><td>341.7</td><td>530.2</td><td>531.2</td></tr><tr><td>Share</td><td>Keep</td><td>303.3</td><td>321.7</td><td>460.2</td><td>461.2</td></tr><tr><td>Share</td><td>Drop</td><td>161.3</td><td>263.2</td><td>253.7</td><td>306.2</td></tr></table>

Table 4 Layer selection accuracy (%) with 4B and six skipped blocks in $\mathrm { T } { \mathrm { + } } \mathrm { S } { \mathrm { + } } \mathrm { P } . $ . Random reports the mean and sample standard deviation over seeds 17, 29, and 43. Bold marks the best accuracy.
<table><tr><td rowspan="2">Selection</td><td colspan="2">Accuracy (%) ↑</td></tr><tr><td>AI2D</td><td>BLINK</td></tr><tr><td>Random</td><td>79.59 ± 2.14</td><td>61.56 ± 2.93</td></tr><tr><td>Tail</td><td>83.94</td><td>63.97</td></tr><tr><td>Block Influence</td><td>79.89</td><td>64.97</td></tr><tr><td>Greedy Teacher KL</td><td>84.42</td><td>64.12</td></tr><tr><td>Output sensitivity</td><td>84.29</td><td>64.33</td></tr></table>

Table 5 Selected artificial 4B workloads with fixed source images and questions. Sharing is the fraction of candidate content held in common, excluding mandatory boundary tokens. Each condition uses four source inputs and five timing rounds. Speedup is S+P latency divided by T+S+P latency. Bold numbers mark the lower latency or peak memory in each pair. Artificial candidates are used to analyze execution cost.
<table><tr><td colspan="3">Workload</td><td colspan="2">Latency (ms) ↓</td><td></td><td colspan="2">Peak memory (MiB) ↓</td></tr><tr><td>Candidates</td><td>Tokens</td><td>Shared (%)</td><td>S+P</td><td>T+S+P</td><td>Speedup ↑</td><td>S+P</td><td>T+S+P</td></tr><tr><td>2</td><td>8</td><td>0</td><td>110.54</td><td>140.13</td><td>0.789</td><td>9436.6</td><td>9549.0</td></tr><tr><td>8</td><td>32</td><td>50</td><td>129.95</td><td>153.10</td><td>0.849</td><td>9646.7</td><td>9449.2</td></tr><tr><td>64</td><td>128</td><td>0</td><td>1511.67</td><td>1595.58</td><td>0.947</td><td>24742.3</td><td>12466.2</td></tr><tr><td>64</td><td>128</td><td>75</td><td>1511.75</td><td>552.62</td><td>2.736</td><td>24615.1</td><td>11260.5</td></tr></table>

Accumulated over the LIBERO A request, initial-state copy allocations fall from 1,584 MiB to zero, and kernel final-state allocations fall from 1,254 to 380 MiB. The remaining final states support the continuation of internal segments. Initial-state sharing removes duplicate copies, while omitting leaf caches further reduces final-state allocation. These measurements isolate the storage efect of S with the candidate histories and pruning mask held fixed. The adjacent Table 4 compares layer selection criteria, discussed in Section 4.6.

## 4.5 History Reuse and Execution Structure

Redundancy in real candidate sets. We count logical candidate token occurrences before padding with the 4B model. T merges nodes only when their complete histories, positions, and execution conditions agree. Across the complete evaluation sets, token counts fall by 29.64% on AI2D, 26.16% on ScienceQA-IMG, 34.88% on BLINK, 26.74% on ERQA, and 62.84% on LIBERO. POPE has no mergeable candidate histories under this protocol.

History merging also divides execution into segments. LIBERO requires an average of 18.73 candidate segment calls per frame. Profiling three frames gives 17 to 19 candidate backbone calls for T+S+P, compared with two for S+P. On the timing subset, their latencies are 721.34 and 294.16 ms per frame, respectively. The reduction in logical histories therefore coexists with additional calls between segments.

Candidate scale and shared content. We vary candidate counts, sufix lengths, shared content, and branching structure while fixing source images and questions. The scan covers 96 conditions for 4B and 17 representative conditions for each smaller model. These artificial candidates measure execution cost and are excluded from accuracy evaluation. With S and P fixed, the trie implementation lowers latency in 27 of the 96 conditions for 4B and lowers total peak memory in 63. It lowers latency in five conditions for each smaller model. Table 5 shows four representative workloads. With 64 candidates of 128 tokens and 75% shared content, logical token occurrences fall from 8,192 to 2,207. T+S+P achieves a 2.736× speedup over S+P and lowers total peak memory by 54.25%. The complete structure can thus improve both costs when candidate evaluation contains suficient shared computation.

![](images/13194021066bd810cc6c6c8b26ea515c7f9dbe8a121560f3dec6e7ffa8f56edc.jpg)

Table 6 Quality and execution cost of the 4B depth sweep over six datasets. Retention averages six accuracy ratios to the matched Original. Speedup divides Original latency by variant latency. Peak ratio divides variant total peak memory by Original peak memory. Both cost ratios use geometric means over the six datasets.  
Figure 4 Accuracy and latency of OmniJev 4B on AI2D and BLINK under fine-grained candidate pruning. Labels denote 0, 2, 4, 6, 8, or 10 skipped blocks, with T and S enabled. Stars mark Original measured in this sweep, open circles mark T+S without pruning, and orange points mark the six-block configuration. All points use complete evaluation splits.
<table><tr><td>Mode</td><td>Skip</td><td>Ret.(%)</td><td>Speedup↑</td><td>Peak ratio ↓</td></tr><tr><td>T+S+P</td><td>10</td><td>98.16</td><td>0.890</td><td>0.994</td></tr><tr><td>T+S+P</td><td>14</td><td>93.66</td><td>0.953</td><td>0.993</td></tr><tr><td>T+S+P</td><td>18</td><td>78.76</td><td>1.011</td><td>0.993</td></tr><tr><td>S+P</td><td>10</td><td>98.30</td><td>1.188</td><td>0.971</td></tr><tr><td>S+P</td><td>14</td><td>93.87</td><td>1.236</td><td>0.971</td></tr><tr><td>S+P</td><td>18</td><td>79.44</td><td>1.280</td><td>0.970</td></tr></table>

These costs also reflect the readout implementation. Trie readout projects at most 128 required nodes to the vocabulary at a time, while S+P projects all predictor positions in a candidate batch together. With the same 64 candidates of 128 tokens but no shared content, total peak memory still falls by 49.62%, while only 1.54% of logical token occurrences are merged through common boundary markers. The memory reduction measures the combined efect of trie execution and its readout schedule, rather than history merging alone.

The role of branching. We compare 24 pairs of flat and two-level tries with matched candidate counts, sufix lengths, and unique token counts. The two-level trie is slower in 22 pairs, with latency ratios relative to the flat trie ranging from 0.950 to 1.499. For eight candidates of 32 tokens, both structures contain 144 unique tokens. Increasing the segment count from two to four changes latency from 153.10 to 218.79 ms. The eficiency of shared execution thus depends on both the amount of reused history and its segmentation.

## 4.6 Reducing the Depth of Shared Paths

Fine-grained pruning behavior. Figure 4 plots accuracy against latency on AI2D and BLINK with 4B. We retain T and S and vary the number of skipped candidate blocks from zero to ten in steps of two. Increasing pruning lowers latency on both tasks, while accuracy changes non-monotonically. This local variation motivates examining larger budgets across all six datasets.

Exploring the pruning range. We use the fixed sensitivity rankings to examine how much candidate depth can be removed. For 4B, we first screen nine budgets on 128 fixed questions per public dataset and all 188 LIBERO frames. For T+S+P and S+P separately, we select the largest tested budget with at least 90% mean retention and its two neighboring budgets. Their union gives 10, 14, and 18 skipped blocks, which we evaluate on all six complete splits. For 0.8B and 2B, the complete-split sweep tests 6, 8, 11, and 14 skipped blocks. The six-block rows in Table 1 retain the default-budget results, and the larger-budget rows use the sweep results. This analysis uses the evaluation sets to characterize compression limits. The 90% criterion selects budgets from measured task scores, while the layer rankings remain fixed without target labels.

Task sensitivity under pruning. The larger-budget results in Table 1 show that tasks respond diferently to pruning. At ten skipped blocks, full FastJEV retains at least 94.41% on every dataset with 4B. At fourteen blocks, POPE and BLINK retain 99.66% and 100.57%, while ERQA and LIBERO retain 89.39% and 76.96%. Four datasets reach 90% retention with T+S+P, and five with S+P. For 0.8B and 2B at eleven skipped blocks, all six datasets exceed 90%, with minima of 93.45% and 92.61%. These diferences show that the pruning range depends on both task and model scale.

Execution cost under deeper pruning. Table 6 compares quality and cost for both 4B configurations. Latency and memory are measured against Original within the same process using the protocol in Section 4.1. Cost ratios are normalized within each dataset and aggregated by their geometric mean. At fourteen skipped blocks, mean retention is 93.66% for T+S+P and 93.87% for S+P. Their speedups over Original are 0.953× and 1.236×, with total peak memory ratios of 0.993 and 0.971. Fourteen blocks is the largest fully evaluated budget meeting the mean retention criterion for both configurations. For 0.8B and 2B at eleven skipped blocks, speedups are 0.938× and 0.939×, with peak memory ratios of 0.985 and 1.023. Pruning reduces computation along candidate paths, while context encoding and the execution of these paths continue to contribute to request latency.

Selection at a fixed budget. We compare output sensitivity with Block Influence based on hidden-state similarity (Men et al., 2025), tail pruning, three random masks, and greedy teacher KL selection. All methods use the same six-block budget, calibration source, and T+S+P executor. Random masks use seeds 17, 29, and 43, and report mean accuracy and sample standard deviation. Table 4 reports complete-split accuracy on AI2D and BLINK. Output sensitivity reaches 84.29% on AI2D, close to greedy teacher KL at 84.42% and above Block Influence, tail pruning, and the random mean. On BLINK, it reaches 64.33%, below Block Influence at 64.97% and above the other criteria. The two tasks favor diferent criteria, while output sensitivity remains competitive on both. It provides the fixed ranking reused across tasks and pruning budgets in our experiments.

## 4.7 Discussion

Does more sharing always mean faster inference? Prefix reuse is a natural design for JEV inference, where candidates are evaluated against a common context. Existing context reuse avoids repeatedly encoding this shared input (Xu et al., 2026). Extending reuse to matching candidate histories further reduces repeated computation. However, our OmniJev experiments show that this extension can increase inference latency. Independent batched candidate evaluation can execute faster despite retaining repeated histories. Shared context anchoring with decision guided pruning (S+P) retains common context reuse. The comparison therefore examines the benefit of additional sharing within candidate sufixes. The finding highlights a gap between reducing repeated work and reducing execution time.

Why can less computation take more time? In our trie executor, candidate paths are divided into segments at forks, and continuation states connect these segments. Frequent branching can leave little computation within each segment, while traversal, kernel launches, and state handling still incur costs. Independent candidate evaluation keeps complete sufixes in regular batches. Sharing therefore changes both the amount of computation and the opportunities to batch it. Its execution benefit depends on whether the saved computation outweighs the cost of segmentation and state handling. Pruning further reduces computation within each segment without removing its boundaries, which may make these costs more prominent.

What does this imply for JEV design? Full FastJEV connects history reuse, state storage, and candidate depth in one compact structure. Its comparison with this ablation shows why the extent of sharing alone is insuficient to assess execution eficiency. The useful granularity of sharing also depends on branching structure and the computation remaining after pruning. This motivates future schedules that preserve shared histories while grouping compatible continuations into larger batches. Our findings suggest evaluating prefix sharing together with its execution schedule when designing eficient JEV models.

## 5 Conclusion

In this paper, we presented FastJEV for compact candidate evaluation in JEV models. Shared context anchoring and candidate prefix sharing jointly organize state storage and matching token histories. Decision guided pruning then reduces candidate depth according to its efect on final scores. This structure retains full context encoding and all candidates without additional training. Across three OmniJev v1.1 scales, the complete method removes about 44% to 46% of candidate layers. These configurations retain over 93% of the original task scores on average across six evaluation sets. Controlled experiments show that the complete structure reduces repeated histories and state storage. Combining shared context anchoring with decision guided pruning improves eficiency when candidate sufixes are evaluated independently in batches. The execution benefit of history sharing depends on candidate overlap and branching structure. These findings provide a structural view of JEV inference and a basis for further study of compact candidate evaluation.

## References

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXI, volume 15139 of Lecture Notes in Computer Science, pages 19–35. Springer, 2024. doi: 10.1007/978-3-031-73004-7\_2. URL https://doi.org/10.1007/978-3-031-73004-7\_2.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: multimodal large language models can see but not perceive. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXIII, volume 15081 of Lecture Notes in Computer Science, pages 148–166. Springer, 2024. doi: 10.1007/978-3-031-73337-6\_9. URL https://doi.org/10.1007/978-3-031-73337-6\_9.

Sneha Murthy Ghantasala. Treewy: Speculative verification for gated deltanet hybrids, 2026. URL https://doi.org/ 10.48550/arXiv.2608.20961.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part IV, volume 9908 of Lecture Notes in Computer Science, pages 235–251. Springer, 2016. doi: 10.1007/978-3-319-46493-0\_15. URL https://doi.org/10.1007/978-3-319-46493-0\_15.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with pagedattention. In Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace, editors, Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 611–626. ACM, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/3600006.3613165.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR, 2023a. URL https://proceedings.mlr.press/v202/li23q.html.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 292–305. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.EMNLP-MAIN.20. URL https://doi.org/10.18653/v1/2023.emnlp-main.20.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: benchmarking knowledge transfer for lifelong robot learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023a. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets\_and\_Benchmarks.html.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New

Orleans, LA, USA, December 10 - 16, 2023, 2023b. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/ 2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html.

Jie Ma, Zhike Qiu, Jiayi Ji, Xiaoshuai Sun, and Rongrong Ji. Look less, reason more: Block-wise attention skipping for eficient multimodal llms, 2026. URL https://doi.org/10.48550/arXiv.2606.08511.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/ paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A visual question answering benchmark requiring external knowledge. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 3195–3204. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR.2019.00331. URL http://openaccess.thecvf.com/content\_CVPR\_2019/html/Marino\_OK-VQA\_A\_ Visual\_Question\_Answering\_Benchmark\_Requiring\_External\_Knowledge\_CVPR\_2019\_paper.html.

Xin Men, Mingyu Xu, Qingyu Zhang, Qianhao Yuan, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, volume ACL 2025 of Findings of ACL, pages 20192–20204. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.FINDINGS-ACL.1035. URL https://doi.org/10.18653/v1/2025.findings-acl.1035.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021. URL http://proceedings.mlr.press/v139/radford21a.html.

Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, and Jae-Joon Kim. SLEB: streamlining llms through redundancy verification and elimination of transformer blocks. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, pages 46136–46155. PMLR / OpenReview.net, 2024. URL https://proceedings.mlr.press/v235/song24f.html.

Gemini Robotics Team. Gemini robotics: Bringing AI into the physical world, 2025. URL https://doi.org/10. 48550/arXiv.2503.20020.

Tianrun Xu, Hongbang Fan, Jiahao Lin, Zilin Zhu, Zhenxin Diao, Longteng Guo, and Jing Liu. Omnijev: an omni-modal system one decision model, 2026. URL https://github.com/tinnel123666888/OmniJev. Beijing Zhongguancun Academy; Institute of Automation, Chinese Academy of Sciences; Zevo.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=r8H7xhYPwz.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. Sglang: Eficient execution of structured language model programs. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ 724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong T. Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: vision-language-action models transfer web knowledge to robotic control. In Jie Tan, Marc Toussaint, and Kourosh Darvish, editors, Conference on Robot Learning, CoRL 2023, 6-9 November 2023, Atlanta, GA, USA, volume 229 of Proceedings of Machine Learning Research, pages 2165–2183. PMLR, 2023. URL https://proceedings.mlr.press/v229/zitkovich23a.html.

## Supplementary Material

This supplement provides complete experimental results and extended pruning analyses for FastJEV. All experiments use OmniJev v1.1. LIBERO results refer to the reconstructed ofline question set described in the main paper.

## A Additional Experimental Results

Table 7 reports the complete quality, latency, and peak GPU memory measurements for the six-block main evaluation. It retains all three model scales and six datasets from Table 1. FastJEV uses T+S+P in every comparison. The measurements follow Section 4.1, including full model weights in peak memory. The larger-budget sweep uses its own matched Original measurements and is reported separately in Table 6.

Trie readout projects at most 128 required nodes to the vocabulary at a time, while the independent branch path projects all predictor positions in a candidate batch together. The measured cost diferences between T+S+P and S+P therefore include history merging, segment scheduling, and vocabulary projection scheduling. Their individual cost contributions are not isolated by these comparisons.

The complete method lowers total peak memory in six of the 18 comparisons. At all three model scales, it is faster than Original on POPE and slower on the other five datasets. POPE has no candidate histories that can be merged under this protocol, so its speedup reflects other diferences in candidate execution. The LIBERO memory reductions of 15.70%, 8.43%, and 9.12% occur with higher latency. Section 4.5 examines segmentation and readout scheduling alongside logical history reduction.

## A.1 Independent Timing Runs

We repeat the 4B measurements on AI2D and LIBERO in three independent processes with fixed inputs and configurations. Each process uses five timing rounds on AI2D and three on LIBERO. We report the mean and sample standard deviation of the three process means. On AI2D, latency is 144.98 ± 0.56 ms for Original, 124.42 ± 0.26 ms for S+P, and 161.56 ± 1.59 ms for T+S+P. On LIBERO, the corresponding values are 336.27 ± 2.45, 295.09 ± 2.16, and 723.27 ± 12.29 ms per frame. All three processes preserve the same latency ordering on both datasets.

Profiling three 4B LIBERO frames gives 4,963 to 5,027 kernel launches within candidate computation for S+P and 41,647 to 47,271 for T+S+P. These counts complement the backbone call analysis in Section 4.5 and expose the finer execution granularity of the trie path.

## A.2 Additional Decision Quality Measures

At the default six-block budget, POPE F1 scores for Original and T+S+P are 86.05% and 88.29% for 0.8B, 89.00% and 89.00% for 2B, and 90.52% and 90.50% for 4B. On reconstructed LIBERO-10 QA, the accuracy changes for 2B and 4B are −2.66 and −1.46 percentage points. Using 2,000 paired bootstrap samples over the 20 evaluation episodes, their 95% confidence intervals are [−4.12, −1.27] and [−2.76, −0.07] percentage points. The resampling unit is an episode, preserving the grouping of its frames and questions.

## A.3 Understanding Compact Candidate Evaluation

Shared histories and execution granularity. T, S, and P act on connected parts of candidate evaluation. T organizes matching histories into shared paths, S reduces the state copies needed to execute their continuations, and P shortens the depth of these paths. Figure 3 of the main paper measures the resulting reductions in repeated histories and live state storage. The branching experiment further shows that token reuse alone does not determine execution cost. With matched unique token counts, the two-level trie is slower than the flat trie in 22 of 24 pairs. Additional segments introduce more calls between shared paths and their continuations. The eficiency of T therefore depends on both the amount of shared work and how that work is organized. S+P retains state sharing and pruning while evaluating candidate sufixes directly in batches, providing a complementary view of their execution eficiency.

Table 7 Complete accuracy and inference cost at the default six-block budget across three OmniJev model sizes. FastJEV uses all three components (T+S+P). Bold marks the best value in each pair at the displayed precision. Latency is per question, except for LIBERO-10 QA, which is per frame of eight questions. Peak GPU memory includes model weights and uses the maximum PyTorch allocated peak over the measured inputs. LIBERO-10 QA is our reconstructed ofline set.
<table><tr><td></td><td colspan="2">Accuracy (%) ↑</td><td colspan="2">Latency (ms) ↓</td><td colspan="2">Peak memory (MiB) ↓</td></tr><tr><td>Dataset</td><td>Original</td><td>FastJEV</td><td>Original</td><td>FastJEV</td><td>Original</td><td>FastJEV</td></tr><tr><td colspan="7">0.8B</td></tr><tr><td>AI2D</td><td>62.27</td><td>62.37</td><td>98.09</td><td>105.54</td><td>1930.9</td><td>1934.8</td></tr><tr><td>ScienceQA-IMG</td><td>69.01</td><td>69.31</td><td>100.17</td><td>106.60</td><td>1907.9</td><td>1890.8</td></tr><tr><td>POPE</td><td>84.76</td><td>87.79</td><td>91.26</td><td>75.30</td><td>1832.1</td><td>1877.8</td></tr><tr><td>BLINK</td><td>43.98</td><td>43.40</td><td>119.35</td><td>123.10</td><td>2020.2</td><td>2075.1</td></tr><tr><td>ERQA</td><td>28.75</td><td>29.50</td><td>111.84</td><td>133.41</td><td>2133.5</td><td>2219.4</td></tr><tr><td>LIBERO-10 QA</td><td>55.45</td><td>55.65</td><td>192.44</td><td>487.10</td><td>2312.8</td><td>1949.8</td></tr><tr><td colspan="7">2B</td></tr><tr><td>AI2D</td><td>75.00</td><td>74.90</td><td>111.99</td><td>120.30</td><td>4549.1</td><td>4644.0</td></tr><tr><td>ScienceQA-IMG</td><td>83.79</td><td>83.84</td><td>111.14</td><td>121.44</td><td>4523.6</td><td>4603.7</td></tr><tr><td>POPE</td><td>88.83</td><td>88.80</td><td>104.29</td><td>89.27</td><td>4464.8</td><td>4590.4</td></tr><tr><td>BLINK</td><td>52.24</td><td>51.13</td><td>163.29</td><td>167.12</td><td>4633.2</td><td>4930.3</td></tr><tr><td>ERQA</td><td>36.75</td><td>37.00</td><td>141.95</td><td>163.36</td><td>4753.9</td><td>5220.9</td></tr><tr><td>LIBERO-10 QA</td><td>58.24</td><td>55.59</td><td>235.84</td><td>538.90</td><td>5012.7</td><td>4590.4</td></tr><tr><td colspan="7">4B</td></tr><tr><td>AI2D</td><td>84.33</td><td>84.29</td><td>145.33</td><td>161.41</td><td>9454.0</td><td>9419.2</td></tr><tr><td>ScienceQA-IMG</td><td>94.30</td><td>94.10</td><td>137.39</td><td>156.90</td><td>9387.3</td><td>9340.2</td></tr><tr><td>POPE</td><td>90.77</td><td>90.64</td><td>126.82</td><td>105.92</td><td>9192.5</td><td>9322.5</td></tr><tr><td>BLINK</td><td>64.33</td><td>64.33</td><td>246.36</td><td>253.48</td><td>9629.2</td><td>9792.3</td></tr><tr><td>ERQA</td><td>44.75</td><td>46.50</td><td>202.21</td><td>238.19</td><td>9788.3</td><td>10163.1</td></tr><tr><td>LIBERO-10 QA</td><td>65.23</td><td>63.76</td><td>336.46</td><td>721.34</td><td>10410.2</td><td>9461.3</td></tr></table>

State storage and total memory. The state experiment fixes T and the pruning mask while changing state storage policies. For LIBERO A, sharing initial states and omitting unused leaf caches lowers observed live state storage from 544.1 to 253.7 MiB without changing the candidate probabilities. This isolates a storage benefit within the same candidate structure. Total GPU peak memory also includes model weights, activations, attention caches, and readout bufers. All model weights remain resident because context encoding retains the full backbone. The reduction in candidate state storage therefore accounts for only part of the total memory footprint. The readout scheduling diference described above also contributes to the cost comparison between trie and independent execution.

Takeaway. The complete structure jointly reduces repeated histories, state copies, and executed candidate layers. Its latency benefit depends on the amount of shared work and the granularity of execution.

## A.4 Qualitative Comparisons

Figure 5 and Figure 6 compare individual decisions from OmniJev and FastJEV across the six evaluation sets. Both use the 4B v1.1 model. FastJEV combines all three components and skips 14 of the 32 candidate layers, while retaining full context encoding. We show the input images, question, ground truth, and selected answers alongside the highest candidate probabilities. We select examples answered correctly by FastJEV, including cases where both models are correct and cases where their answers difer. Aggregate performance is reported in Table 1.

Table 8 Selected candidate blocks across calibration set sizes for OmniJev v1.1. Each entry lists the six blocks selected for pruning. All three calibration draws yield the same set in every entry. Layer indices start at one and are listed in ascending order. Context encoding retains all blocks.
<table><tr><td>Calibration inputs</td><td>0.8B</td><td>2B</td><td>4B</td></tr><tr><td>8</td><td>{16, 17, 18, 19, 20, 21}</td><td>{15, 17, 18, 19, 21, 22}</td><td>{21, 23, 24, 25, 26, 27}</td></tr><tr><td>16</td><td>{16, 17, 18, 19, 20, 21}</td><td>{15, 17, 18, 19, 21, 22}</td><td>{21, 23, 24, 25, 26, 27}</td></tr><tr><td>32</td><td>{16, 17, 18, 19, 20, 21}</td><td>{15, 17, 18, 19, 21, 22}</td><td>{21, 23, 24, 25, 26, 27}</td></tr><tr><td>64</td><td>{16, 17, 18, 19, 20, 21}</td><td>{15, 17, 18, 19, 21, 22}</td><td>{21, 23, 24, 25, 26, 27}</td></tr><tr><td>128</td><td>{16, 17, 18, 19, 20, 21}</td><td>{15, 17, 18, 19, 21, 22}</td><td>{21, 23, 24, 25, 26, 27}</td></tr></table>

Preserved decisions and changing scores. The diagram and science examples retain the correct answer after candidate depth is reduced by 43.75%. Their probability distributions change even when the selected answer remains the same. The object presence and counting examples show changes from an incorrect to a correc answer. In the counting example, the probability of the correct answer decreases slightly, but its rank rises because the competing answer decreases more. This example illustrates the role of relative candidate scores in the final decision.

Correct decisions across visual tasks. The remaining examples extend this comparison to object interaction and robot state recognition. FastJEV selects the marked faucet handle in ERQA and correctly identifies that the robot gripper holds an object in the reconstructed LIBERO question. Together, these examples illustrate correct candidate decisions across diagrams, scene images, and robot observations with reduced candidate depth.

## B Extended Pruning Analysis

## B.1 Stability of Layer Selection

Decision guided pruning estimates layer sensitivity from a small set of unlabeled inputs. We examine whether changing the size of this set changes the layers selected for pruning. We evaluate the 0.8B, 2B, and 4B OmniJev models using three disjoint calibration draws from OK-VQA. Each draw contains 128 inputs and defines nested subsets of 8, 16, 32, and 64 inputs. All three models use the same calibration draws. Each input contains 64 candidate answers, and no answer labels are used for layer selection. We fix the pruning budget at six candidate blocks and retain all blocks for context encoding. Table 8 reports the selected blocks for each calibration set size. For each model, the selected set remains unchanged across all five sizes and all three draws. These sets also match the corresponding masks used in the main experiment. Although the models select diferent blocks, each model recovers a consistent set from the tested calibration subsets.

## B.2 Pruning Budgets and Depth Redundancy

This section supplements the pruning results in Figure 4 and Table 6 of the main paper. We describe budget selection and reference measurements, then discuss candidate depth redundancy.

Budget selection and reference measurements. The layer ranking is fixed from calibration before the budget scan. For 4B, the initial screen evaluates nine budgets using 128 fixed questions per public dataset and all 188 LIBERO frames. The largest tested budget meeting 90% mean retention and its neighbors determine the budgets evaluated on complete splits. This scan characterizes compression on the reported evaluation sets. Both the fine-grained sweep and the larger-budget sweep use their own matched Original measurements. The same verified Original accuracies normalize all points within the larger-budget sweep. Latency and peak memory are measured against Original within the same process for each evaluated budget and dataset.

For 0.8B and 2B, the complete-split scan evaluates 6, 8, 11, and 14 skipped blocks. At 11 blocks, all six datasets exceed 90% retention, with minima of 93.45% and 92.61%, respectively. Table 1 of the main paper

<table><tr><td>phytoplankton</td><td>98.3%</td></tr><tr><td>mackerel</td><td>0.8%</td></tr><tr><td>large shark</td><td>0.3%</td></tr></table>

Question  
GT no

(a) AI2D  
![](images/d04dde2375ce379be009cd6059d92a34db8f9aa5dfccd051fc592f6a07ba7481.jpg)  
Question  
Which of these is the lowest in the food chain in this diagram?

GT phytoplankton  
(b) ScienceQA-IMG  
![](images/50f034d8de86fb8cd9f3fcb6e79dd2962d851cf06f38b590c201b139998c8118.jpg)  
What is the capital of Nevada?

GT Carson City
<table><tr><td>FastJEV (ours) Carson City</td><td>Carson City L</td><td>OmniJev (original)</td><td>√</td></tr><tr><td>Carson City</td><td>83.4%</td><td>Carson City</td><td>100.0%</td></tr><tr><td>Las Vegas</td><td>11.7%</td><td>Las Vegas</td><td>&lt;0.1%</td></tr><tr><td>Reno</td><td>3.8%</td><td>Abstain</td><td>&lt;0.1%</td></tr></table>

(c) POPE  
![](images/18b73c728d25d5ae755a709b59c118b5bbf5415d65e57e1eb8abb032944e8937.jpg)  
Question  
Is there a person in the image?  
Figure 5 Selected correct FastJEV answers on AI2D, ScienceQA-IMG, and POPE using OmniJev 4B. FastJEV uses shared context anchoring, candidate prefix sharing, and pruning of 14 candidate layers. Each panel places the image on the left and the question and answer comparison on the right. GT denotes the evaluation label. Bars show up to three highest probabilities from each model without renormalization. Checks and crosses indicate agreement with GT.

Wrist view  
(d) BLINK  
![](images/d8cf6c3bc79fa94de7608754d5dd5636dc75b72c37b7fd9755df0d74f7162385.jpg)  
(e) ERQA

## Question

How many white pillows are in this image?

GT 4
<table><tr><td colspan="2">FastJEV (ours) 4</td><td colspan="2">OmniJev (original) √ 3</td><td colspan="2">X</td></tr><tr><td>4</td><td></td><td>35.6%</td><td>3</td><td></td><td>59.8%</td></tr><tr><td>3</td><td></td><td>31.4%</td><td>4</td><td></td><td>38.4%</td></tr><tr><td></td><td></td><td>21.6%</td><td>5</td><td></td><td>1.7%</td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Question

![](images/ec37b0c26cdaed309d31a794a864add60302d9634b4e65bbf28429708957f5bc.jpg)  
Which point (yellow, blue, red, orange) should I grasp the faucet in order to turn it on?

GT Blue.
<table><tr><td colspan="2">FastJEV (ours) Blue.</td><td colspan="2">OmniJev (original) √ Yellow.</td><td colspan="2">x</td></tr><tr><td>Blue.</td><td></td><td>28.9%</td><td>Yellow.</td><td></td><td>60.5%</td></tr><tr><td>Yellow.</td><td></td><td>27.0%</td><td>Blue.</td><td></td><td>28.8%</td></tr><tr><td>Red.</td><td></td><td>25.5%</td><td>Red.</td><td></td><td>10.1%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

(f) LIBERO-10 QA  
![](images/162c463d162ae9999311abd61d4e7a0e7ff66e9a11927626e0de640ef1f3a967.jpg)

## Question

![](images/3b79c0b4da4b7d846fe66265b2ae8402483d5f4d1c7f956fbc4cb7c6208f8be0.jpg)

Robot camera view. The gripper is currently closed on an object.

GT yes
<table><tr><td rowspan=1 colspan=2>yes</td><td rowspan=1 colspan=1>55.3%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>no</td><td rowspan=1 colspan=1>74.1%</td></tr><tr><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=2>44.7%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.9%</td></tr></table>

Figure 6 Selected correct FastJEV answers on BLINK, ERQA, and reconstructed LIBERO-10 ofline QA under the same OmniJev 4B configuration. All input views are shown, including both robot cameras. Answers display the selected candidate text. Appended choice lists and response-format instructions are omitted from displayed questions. The LIBERO ground truth uses the frozen state-derived label. Bars retain the original model probabilities and are not normalized over the displayed subset.

reports the default-budget results at six blocks and the sweep results at larger budgets.

Understanding depth redundancy. Context encoding remains at full depth throughout these sweeps. The retained quality at roughly 44% to 46% candidate depth reduction therefore supports reducing candidate computation after the context has been fully encoded. At a fixed pruning budget, T+S+P and S+P use the same layer mask and show similar retention curves. For example, their mean retention difers by only 0.21 percentage points at 14 skipped blocks with 4B. This similarity suggests that the task diferences in this sweep are mainly associated with the candidate depth removed under both execution structures. The larger retention losses on ERQA and LIBERO indicate a narrower pruning range for these tasks than for POPE and BLINK.

Takeaway. Candidate depth can be reduced substantially while preserving full context encoding. The task curves identify where deeper pruning begins to reduce decision quality within the tested range.

## C Limitations and Future Work

FastJEV provides a structural perspective on redundancy in JEV candidate evaluation. The current implementation introduces scheduling overhead when shared paths are split into short segments. Our analysis is based on OmniJev v1.1. Further work is needed to reduce this overhead and examine how the findings extend to more JEV models.

Our pruning strategy uses decision sensitivity to reduce candidate depth, while token pruning reduces sequence length. These two dimensions ofer complementary opportunities for compression. Future work can explore their combination and evaluate its efects on decision quality and execution cost.