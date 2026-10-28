# Latent Visual Contrastive Reasoning with Multi-Branch Exploration

Kegeng Tang<sup>1</sup>, Jingbo Wang<sup>1</sup>, Yangming Zhang<sup>2</sup>, Huanshu Zhang<sup>3</sup>, Shaogang Ren<sup>1</sup>, and Zihao Wang<sup>1</sup>\*

<sup>1</sup>Department of Computer Science & Engineering, University of Tennessee at Chattanooga, Chattanooga, TN, USA

<sup>2</sup>Department of Computer Science, University of Texas at Arlington, Arlington, TX, USA

<sup>3</sup>Department of Electrical Engineering, The Pennsylvania State University, University Park, PA, USA

{hcw575, ydk297, sren9, zihao.wang}@tennessee.edu, yangming.zhang@uta.edu, hpz5226@psu.edu

\*Corresponding author: zihao.wang@tennessee.edu

Abstract—Multimodal large language models have made sub stantial progress in visual question answering, but complex visual reasoning often requires locating and comparing multiple pieces of question-relevant visual evidence. Existing methods that move intermediate reasoning into the visual latent space usually follow a single linear reasoning trajectory, which can be limiting when multiple visual hypotheses or competing evidence regions are relevant. We propose Latent Visual Contrastive Reasoning, a branch-aware latent reasoning framework that generates multiple question-conditioned latent visual branches as soft hypotheses over image tokens. To train these branches, our method uses boundingbox evidence annotations to build target evidence representations and introduces a soft contrastive branch-evidence alignment objective. This objective encourages evidence-relevant branches to align with annotated visual regions while contrasting them against negative evidence. We instantiate our method on Qwen2.5- VL backbones and train it with standard supervised fine-tuning. Experiments on perception-intensive visual reasoning benchmarks show that the proposed method achieves overall improvements over the corresponding backbone and single-path latent reasoning baselines at both 3B and 7B scales.

## I. INTRODUCTION

Multimodal large language models (MLLMs) have advanced vision-language understanding by connecting visual encoders with large language models [1]. This paradigm has been further strengthened by frozen vision-language bootstrapping, visual instruction tuning, and stronger general-purpose multimodal modeling [2]–[4]. Despite these advances, complex visual question answering requires more than recognizing objects or directly mapping an image-question pair to an answer. Models must identify question-relevant evidence, compare candidate regions, and reason over fine-grained visual details [5], [6].

A common way to improve reasoning is to elicit chain-ofthought (CoT) rationales in the language space [7]. For multimodal reasoning, this encourages the model to decompose a visual question into intermediate textual steps before answering [8]. Although textual reasoning can improve interpretability and accuracy, it remains fundamentally language-centric. The image is typically encoded once and treated as a static condition, while intermediate thoughts are represented as discrete text tokens rather than visual evidence.

To better incorporate visual information, recent methods augment MLLMs with explicit visual operations, such as region selection, cropping, zooming, tool invocation, or program execution. Visual programming methods compose perception modules and symbolic programs for compositional visual reasoning [9]. Tool-augmented approaches allow language models to invoke external visual tools or execute programs during inference [10], [11]. VISUAL COT further provides visual chain-of-thought annotations with bounding-box evidence for multimodal reasoning [12]. These approaches enable models to dynamically focus on image regions and re-inject localized observations into later reasoning steps. However, they often depend on external modules, hand-designed operations, or repeated image re-encoding, and the final reasoning trajectory is still largely expressed in natural language.

Latent Visual Reasoning (LVR) offers a promising alternative by moving multimodal reasoning from the language space into the visual embedding space [13]. Instead of generating only textual thoughts, LVR trains the language model to produce latent visual states that reconstruct query-relevant visual tokens and interleaves them with text generation, allowing the model to reason over visual semantics without verbalizing all intermediate evidence. While effective for fine-grained visual understanding and perception-intensive question answering, existing latent visual reasoning methods still largely follow a single linear reasoning trajectory [13]. This formulation can be limiting for questions involving multiple plausible objects, visually similar candidates, or relational evidence across regions, where competing visual hypotheses should be preserved before final prediction [6], [12]. Forcing such alternatives into a single latent path may blur distinct hypotheses and make the learned representation less discriminative.

To address these limitations, we propose Latent Visual Contrastive Reasoning (LVCR), a branch-aware latent reasoning framework for visual question answering. Instead of following a single latent reasoning trajectory, LVCR represents multiple question-conditioned visual hypotheses in parallel. Each branch softly attends to image tokens and encodes a candidate evidence state, which is injected into the language model as an additional latent visual reasoning token. This allows answer generation to condition on multiple evidence configurations within the model’s latent space, without external visual tools or repeated image re-encoding.

To make the branch space evidence-aware, we use boundingbox annotations from VISUAL COT as weak supervision [12]. We introduce a differentiable bbox-guided contrastive alignment objective [14] that pulls the evidence-weighted branch representation toward annotated answer-relevant visual evidence [15] while contrasting it against negative visual regions.

Our contributions are summarized as follows:

• We develop a latent reasoning framework (LVCR) that represents multiple question-conditioned visual hypotheses instead of a single latent reasoning trajectory.

• We propose a soft bbox-guided contrastive branchevidence alignment method that aligns latent branches with answer-relevant visual evidence while contrasting them against negative evidence regions.

• We introduce a lightweight branch injection and ranking mechanism that feeds multiple latent visual branches into the decoder as reasoning tokens, enabling branchaware answer generation through question-relevant branch ordering.

• We demonstrate the effectiveness of LVCR on perceptionintensive visual reasoning benchmarks at both 3B and 7B scales, showing improvements over backbone models and single-path latent reasoning baselines.

## II. RELATED WORK

## A. Multimodal LLM for Visual Reasoning

Multimodal large language models (MLLMs) have become a dominant paradigm for vision-language understanding by connecting visual encoders with large language models. Early representative systems such as Flamingo show that large language models can be adapted to interleaved image-text inputs and perform few-shot vision-language tasks [1]. PaLI further demonstrates the effectiveness of jointly scaling languageimage models across multilingual and multitask settings [16]. Subsequent work improves multimodal alignment by using frozen visual encoders and lightweight modality bridges [2], [17]. Instruction-tuned MLLMs further enhance open-ended visual dialogue and multimodal task following [3], [18]. Recent general-purpose MLLMs further strengthen visual recognition, object localization, document understanding, and multimodal instruction following through improved visual encoders, instruction data, and scaling recipes [4], [19]–[22]. These advances have enabled strong performance on visual question answering and compositional visual reasoning benchmarks [5], [6]. However, many visual reasoning problems require more than global image understanding: models often need to identify question-relevant evidence, compare visually similar candidates, and reason over fine-grained attributes or spatial relations. Recent studies have also extended multimodal foundation models to specialized scientific and medical reasoning scenarios [23], [24].

## B. Chain-of-Thought and Region-Grounded Visual Reasoning

Chain-of-thought prompting improves complex reasoning in language models by eliciting intermediate textual rationales before producing final answers [7]. This idea has also been extended to multimodal settings, where models generate rationales conditioned on both images and text to support answer prediction [8]. Interactive visual chain-of-thought prompting further decomposes knowledge-based visual reasoning into perception, reasoning, and confirmation stages [25]. While such methods improve interpretability and often help reasoning performance, their intermediate reasoning process is primarily represented in the language space.

To better expose visual evidence, another line of work introduces explicit region-grounded, interactive, or tool-augmented reasoning. Visual programming methods compose off-the-shelf vision modules and symbolic operations to solve compositional visual tasks without task-specific training [9]. Tool-augmented systems use language models to generate programs or invoke external visual experts for multi-step visual reasoning [10], [11]. Interactive systems further connect language models with external visual foundation models for perception, editing, and reasoning [26], [27]. In parallel, region-grounded MLLMs support spatial-coordinate inputs and outputs for referential dialogue, localization-aware visual reasoning, and fine-grained grounding [28]–[31]. Recent spatially grounded VLMs further emphasize region-level and 3D-aware spatial reasoning over objects and relations [32], [33]. Visual CoT provides largescale multimodal reasoning data with bounding-box evidence and step-by-step rationales, encouraging models to focus on answer-relevant image regions [12]. These studies demonstrate the importance of grounding reasoning in localized visual evidence, but many of them rely on explicit textual rationales, external visual tools, hand-designed operations, or repeated visual processing.

## C. Latent-Space Reasoning and Visual Alignment

Recent work has begun to investigate whether intermediate reasoning can be represented in continuous latent spaces rather than only as discrete text tokens. In language reasoning, continuous-thought methods reuse hidden states as latent reasoning states instead of verbalizing every intermediate step [34]. In multimodal reasoning, Latent Visual Reasoning (LVR) proposes to generate latent visual states that reconstruct query-relevant visual tokens, thereby enabling autoregressive reasoning directly in the visual embedding space [13], [35]. Recent multimodal continuous-thought work further explores latent-space reasoning over visual and textual representations [36], [37]. This direction is closely related to broader efforts in representation learning and vision-language alignment. Contrastive learning objectives learn representations by pulling positive pairs together and separating them from negative samples [14]. Large-scale vision-language pretraining methods further show that contrastive alignment between visual and textual representations can produce transferable multimodal representations [15], [38], [39]. Together, latent-space reasoning and contrastive alignment provide useful foundations for learning visually grounded intermediate representations. However, how to structure and supervise multiple latent visual hypotheses for fine-grained multimodal reasoning remains relatively underexplored.

![](images/ff17447b4f5988f094d305075b99920b19e2140d6b3317eb8a5ad12bf420d121.jpg)  
Fig. 1. Overall framework of LVCR. Given an image-question pair, LVCR encodes the image and question into visual and textual tokens, and proposes multiple question-conditioned latent branches as candidate visual hypotheses over the image. These branches are contrastively aligned with answer-relevant visual evidence and separated from irrelevant regions. LVCR then ranks the branches using relevance scores computed by cosine similarity with the question representation, and injects the selected branch tokens into the MLLM decoder, which combines them with the original multimodal context to generate the final answer.

## III. METHOD

Fig. 1 shows the overall pipeline (from left to right) of the proposed LVCR. Given an image–question pair, LVCR first encodes the image into visual tokens and the question into a textual representation using a pretrained vision-language backbone. It then constructs K question-conditioned latent branches, where each branch softly attends to the visual tokens and aggregates them into a candidate evidence state, thereby maintaining multiple latent visual hypotheses before answer decoding. During training, LVCR uses answer-relevant bounding-box annotations as localized evidence supervision: the annotated region is converted into a positive evidence representation, and a soft contrastive branch-evidence alignment loss pulls evidence-matched branches toward this representation while contrasting them against negative visual regions. The ranked branches are finally injected into the language decoder as latent reasoning tokens, and the whole model is optimized with a compact objective combining answer prediction and bbox-guided contrastive evidence alignment.

## A. Question-Conditioned Branch Proposal

Let I denote the input image. The vision-language backbone encodes I into a visual token sequence

$$
V = \{ v _ { i } \} _ { i = 1 } ^ { N } , \qquad v _ { i } \in \mathbb { R } ^ { d } ,\tag{1}
$$

and produces a question representation $q ~ \in ~ \mathbb { R } ^ { d }$ . LVCR introduces K learnable branch queries $\{ e _ { k } \} _ { k = 1 } ^ { K }$ to parameterize a set of latent visual branches. For each branch k, we compute a question-conditioned relevance score for each visual token:

$$
s _ { k , i } = f _ { \mathrm { b r } } ( v _ { i } , q , e _ { k } ) ,\tag{2}
$$

where $e _ { k }$ is the k-th learnable branch query, and $f _ { \mathrm { b r } } ( \cdot )$ is a learnable scoring function used to compute branch-specific attention scores over visual tokens conditioned on the question. The scores are normalized over the visual tokens to obtain the branch attention weights:

$$
a _ { k , i } = \frac { \exp ( s _ { k , i } ) } { \sum _ { j = 1 } ^ { N } \exp ( s _ { k , j } ) } .\tag{3}
$$

The latent representation of branch k is then computed as the attention-weighted visual state

$$
b _ { k } = \sum _ { i = 1 } ^ { N } a _ { k , i } v _ { i } .\tag{4}
$$

Collecting all branches gives

$$
\begin{array} { r } { B = \{ b _ { k } \} _ { k = 1 } ^ { K } . } \end{array}\tag{5}
$$

Because the attention weights depend jointly on the visual token $v _ { i } ,$ the question representation $q ,$ and the branch query $\textstyle e _ { k } ,$ , different branches can specialize to different question-conditioned visual evidence patterns while remaining differentiable. The next section describes how localized evidence annotations are converted into supervision for these latent branches.

## B. BBox-Guided Evidence Representation

During training, we assume access to answer-relevant bounding-box annotations that indicate the visual evidence required to answer each question. These annotations can come from any dataset or supervision source that provides localized evidence regions, and our formulation is not tied to a specific dataset. We use the annotated regions to construct a target evidence representation for contrastive supervision.

Let $m _ { i } \in \{ 0 , 1 \}$ indicate whether visual token v falls inside the annotated bounding box. The target evidence representation is obtained by pooling the visual tokens inside the bounding box:

$$
r ^ { + } = \frac { \sum _ { i = 1 } ^ { N } m _ { i } v _ { i } } { \sum _ { i = 1 } ^ { N } m _ { i } } .\tag{6}
$$

The representation $r ^ { + }$ serves as the positive visual evidence for the current image-question pair. Intuitively, a good branch proposal module should produce at least one branch that is strongly aligned with this evidence representation.

## C. Contrastive Branch-Evidence Alignment

The goal of contrastive branch-evidence alignment is to guide latent branches toward answer-relevant visual evidence without forcing the annotated region to be assigned to a single branch. Since different branches may capture partially relevant evidence, we use a soft evidence-matching mechanism to distribute supervision across all branches. Given the proposed branches B and the evidence representation $r ^ { + }$ , we compute a soft matching weight for each branch:

$$
w _ { k } = \frac { \exp ( \sin ( b _ { k } , r ^ { + } ) / \tau _ { s } ) } { \sum _ { j = 1 } ^ { K } \exp ( \sin ( b _ { j } , r ^ { + } ) / \tau _ { s } ) } ,\tag{7}
$$

where sim $( \cdot , \cdot )$ denotes cosine similarity and $\tau _ { s }$ is a soft assignment temperature. The weight $w _ { k }$ measures how strongly the k-th branch is associated with the annotated evidence, assigning larger weights to evidence-relevant branches while still keeping all branches differentiable.

We then construct a soft positive branch representation by aggregating all candidate branches according to their evidencematching weights:

$$
\bar { b } ^ { + } = \sum _ { k = 1 } ^ { K } w _ { k } b _ { k } .\tag{8}
$$

This soft aggregation avoids a non-differentiable hard branch selection and enables credit assignment across multiple poten tially relevant branches.

The contrastive branch-evidence alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { N C E } } = - \log \frac { \exp ( s ^ { + } / \tau ) } { \exp ( s ^ { + } / \tau ) + \sum _ { k = 1 } ^ { K } \exp ( s _ { k } ^ { - } / \tau ) } ,\tag{9}
$$

where

$$
s ^ { + } = \sin ( \bar { b } ^ { + } , r ^ { + } ) , \qquad s _ { k } ^ { - } = \sin ( b _ { k } , r ^ { - } ) .\tag{10}
$$

Here, $\tau$ is the contrastive temperature and $r ^ { - }$ denotes negative evidence representations sampled from low-overlap candidate regions or other images in the mini-batch.

This objective pulls the softly aggregated positive branch representation toward the annotated evidence while contrasting individual branches against negative evidence regions. As a result, the proposal module learns visually grounded branch representations without requiring an explicit branch selector or a manually designed diversity regularizer.

## D. Differentiable Branch Credit Assignment

The soft aggregation in Eq. (8) can also be interpreted as a differentiable credit-assignment mechanism over the latent branch set. Let

$$
c _ { k } = \sin ( b _ { k } , r ^ { + } )\tag{11}
$$

denote the evidence compatibility of branch $k .$ The normalized weight $w _ { k }$ is a softmax function of $c _ { k } .$ , so every branch participates in the positive representation ${ \bar { b } } ^ { + }$ , but branches that are more compatible with the annotated evidence receive larger coefficients. Consequently, the contrastive objective does not require a discrete branch label or a non-differentiable selection step. Instead, the branch proposal module can gradually redistribute evidence responsibility among the K branches during optimization.

This formulation is useful because the supervision specifies an answer-relevant image region but does not specify how that region should be decomposed across latent hypotheses. A hard assignment would force the training procedure to choose a single branch even when several branches capture complementary parts of the same evidence. In LVCR, the soft assignment preserves this ambiguity: strongly matched branches dominate the evidence-aligned representation, while partially matched branches can still receive learning signals through both their contribution to ${ \bar { b } } ^ { + }$ and the normalization over branch similarities. At the same time, the negative term in Eq. (9) discourages branches from indiscriminately matching visually irrelevant regions. The resulting training signal therefore combines positive evidence attraction with branch-level contrast against negative visual content.

## E. Answer Prediction

The proposed branches are injected back into the visionlanguage model as additional latent visual reasoning tokens. Concretely, we reserve K special <lvcr> token positions in the input sequence, placed between a <lvcr\_start> marker and the answer. All K branches are injected and the language model’s own multi-head self-attention integrates the branch information without any separate fusion module. The decoder then predicts the final answer auto-regressively:

$$
p ( \boldsymbol { y } \mid I , \boldsymbol { q } ) = \operatorname { D e c o d e r } ( \boldsymbol { y } \mid \boldsymbol { q } , b _ { 1 } , \ldots , b _ { K } , V ) .\tag{12}
$$

The answer prediction loss is the standard negative loglikelihood:

$$
\mathcal { L } _ { \mathrm { a n s } } = - \sum _ { t = 1 } ^ { T } \log p ( y _ { t } \mid y _ { < t } , I , q ) .\tag{13}
$$

At inference time, we rank the K branches by their cosine similarity to the question representation $q$ before injection. The branches are injected in descending order of similarity, so that the most question-relevant branch occupies the first <lvcr> position. This ordering prioritizes the strongest branch signal while preserving all branches for complementary visual context.

## F. Ranked Latent Sequence Construction

The branch representations are continuous vectors and therefore do not have an intrinsic left-to-right order. Before they are inserted into the language model at inference time, LVCR converts the unordered branch set into a deterministic latent reasoning sequence. We first compute a question-relevance score for every branch,

$$
\rho _ { k } = \sin ( b _ { k } , q ) ,\tag{14}
$$

and define a permutation π such that

$$
\rho _ { \pi ( 1 ) } \geq \rho _ { \pi ( 2 ) } \geq \cdot \cdot \cdot \geq \rho _ { \pi ( K ) } .\tag{15}
$$

The ordered latent branch sequence is then

$$
B ^ { \mathrm { r a n k } } = \left[ b _ { \pi ( 1 ) } , b _ { \pi ( 2 ) } , \ldots , b _ { \pi ( K ) } \right] .\tag{16}
$$

This sequence is placed between the two boundary tokens to form the latent reasoning scaffold

$$
S _ { \mathrm { L V C R } } = \left[ < \mathrm { l v c r } _ { - } s \mathrm { t a r t } > , B ^ { \mathrm { r a n k } } , < \mathrm { l v c r } _ { - } \mathrm { e n d } > \right] .\tag{17}
$$

The scaffold is concatenated with the original multimodal context before answer decoding. Importantly, ranking changes only the positions of the branch states; it does not discard any branch. Thus, the first latent position receives the branch most directly aligned with the question, while later positions retain alternative or complementary visual hypotheses. The decoder can attend to all K branch states together with the original image and question tokens, allowing the final prediction to combine the strongest evidence hypothesis with additional contextual cues.

During training, the branch states remain fully differentiable and are optimized jointly through answer prediction and contrastive evidence alignment. During inference, the evidence annotation and the contrastive target are removed, and the question-conditioned ranking provides the only ordering operation. This creates a simple transition from supervised branch organization during training to annotation-free latent reasoning at test time.

## G. Training Objective

We train LVCR to answer the question correctly while guiding its latent branches toward answer-relevant visual evidence. Accordingly, the final objective combines the standard answer prediction loss with bbox-guided contrastive branchevidence alignment:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { a n s } } + \lambda _ { \mathrm { n c e } } \mathcal { L } _ { \mathrm { N C E } } ,\tag{18}
$$

where $\lambda _ { \mathrm { n { c e } } }$ controls the strength of the contrastive alignment objective.

The answer loss remains the primary learning signal, while the NCE loss regularizes the latent branches to capture localized visual evidence related to the answer. This yields a compact end to-end training formulation that couples language supervision with visual evidence alignment.

Algorithm 1 Training and inference procedure of LVCR.   
Require: Image I, question Q, branch number $K ;$ training addition   
ally uses bounding box R and answer y   
Ensure: Predicted answer $\hat { y }$   
1: Encode I and Q to obtain visual tokens $V = \{ v _ { i } \} _ { i = 1 } ^ { N }$ and question   
representation q   
2: for $k = 1 , \dots , K$ do   
3: Compute $s _ { k , i } = f _ { \mathrm { { b r } } } ( v _ { i } , q , e _ { k } )$ over all visual tokens   
4: Normalize $\{ s _ { k , i } \} _ { i = : } ^ { N }$ <sub>1</sub> and aggregate $\begin{array} { r } { b _ { k } = \sum _ { i } a _ { k , i } v _ { i } } \end{array}$   
5: end for   
6: if training then   
7: Map R to the visual-token mask $\{ m _ { i } \}$ and construct positive   
evidence $r ^ { + }$   
8: Sample negative evidence $r ^ { - }$ and compute soft branch weights   
$\{ w _ { k } \}$   
9: Form $\begin{array} { r } { \bar { b } ^ { + } = \sum _ { k } w _ { k } b _ { k } } \end{array}$ and compute $\mathcal { L } _ { \mathrm { N C E } }$   
10: Insert $\left\{ b _ { k } \right\}$ into the reserved <lvcr> positions and compute   
$\mathcal { L } _ { \mathrm { a n s } }$   
11: Update trainable parameters using $\mathcal { L } _ { \mathrm { a n s } } + \lambda _ { \mathrm { n c e } } \mathcal { L } _ { \mathrm { N C E } }$   
12: else   
13: Rank branches by sim $( b _ { k } , q )$ in descending order   
14: Insert all ranked branches into the reserved <lvcr> positions   
15: Autoregressively decode the answer yˆ   
16: end if

## H. Implementation Details

LVCR is implemented as a lightweight latent-reasoning module on top of the representations already produced by the pretrained multimodal backbone. The visual sequence V and question representation q in Section III-A share the hidden dimension $d ,$ and the K branch queries are trainable parameters optimized jointly with the language model and the remaining LVCR parameters. The branch proposal module therefore operates on the encoded visual tokens directly rather than introducing a second visual encoder or an additional image-processing stage.

For localized supervision, each annotated bounding box is mapped to the visual-token grid to construct the binary mask $\{ m _ { i } \} _ { i = 1 } ^ { N }$ used in Eq. (6). The positive evidence representation is pooled only from the tokens selected by this mask, while negative evidence is obtained from low-overlap candidate regions or from other images in the mini-batch, as described in Section III-C. The bounding-box annotation is used only to construct the training-time contrastive target; neither bounding boxes nor the contrastive loss are required at inference time.

The latent branches are inserted through the reserved reasoning scaffold <lvcr\_start>, K <lvcr> positions, and <lvcr\_end>. The branch vectors occupy the K reserved latent positions and are processed together with the original multimodal context by the decoder’s existing self-attention layers. Consequently, LVCR does not require an additional branch-fusion network. During training, we optimize the language model and LVCR parameters while keeping the vision encoder and multimodal merger frozen; the numerical optimization settings are given in Section IV-A.

Algorithm 1 summarizes the complete training and inference paths. Training uses localized evidence only to shape the latent branch space, whereas inference constructs, ranks, and injects the branches from the image–question pair alone. This separation is important because evidence supervision improves the internal reasoning representation without introducing an annotation requirement at deployment time.

## I. Computational Complexity and Efficiency

LVCR adds computation primarily through branch proposal and the insertion of a small number of latent tokens. Let N denote the number of visual tokens, d the hidden dimension, K the number of branches, and L the sequence length seen by the decoder before LVCR tokens are inserted. Evaluating the branch scoring function over all branch–token pairs and aggregating the branch representations requires O(KN d) computation when the scoring operation is linear in $d ,$ together with $O ( K N )$ attention weights. The soft branch-evidence assignment and contrastive similarities add only O(Kd) computation once the positive and negative evidence representations have been formed. Ranking the branches at inference time costs O(K log K).

For answer generation, LVCR increases the decoder sequence length from L to $L + K$ . Under standard self-attention, the corresponding per-layer attention term changes from $O ( L ^ { 2 } d )$ to $O ( ( L + K ) ^ { 2 } d )$ , giving an additional $O ( ( 2 L K + K ^ { 2 } ) d )$ attention term. In our configurations, K is small $( K = 4$ for 3B and $K = 8$ for 7B), so the number of added latent positions is fixed and much smaller than the full multimodal context. More importantly, all branches are constructed from the visual tokens produced by a single image encoding. LVCR therefore avoids iterative cropping, external visual-tool calls, and repeated vision-encoder passes. The bounding-box pooling and NCE computation are training-only operations and introduce no inference-time annotation or evidence-processing cost. This analysis characterizes the structural overhead of LVCR rather than claiming a particular wall-clock speedup.

## J. Design Discussion

a) Why multiple latent branches?: A single latent state must compress all question-relevant visual information into one representation. This is potentially restrictive when the answer depends on several plausible objects, multiple spatial regions, or competing fine-grained cues. LVCR instead maintains a small set of branch states before decoding. Because each branch has its own learnable query and question-conditioned attention distribution, the model can represent different candidate evidence configurations in parallel rather than averaging them immediately into a single trajectory. The decoder can then integrate complementary branch information jointly with the original visual and textual context.

b) Why soft evidence assignment?: The bounding box specifies which visual region is relevant to the answer, but it does not provide a semantic label indicating which latent branch should explain that region. A hard one-to-one assignment would therefore introduce an arbitrary branch identity and would discard useful gradients for other partially aligned branches. The soft weights in Eq. (7) treat branch identity as latent: strongly aligned branches receive larger supervision while all branches remain differentiable. This lets localized evidence organize the branch space without requiring a separate branch selector or a manually specified matching rule.

c) Why rank before injection?: The set of candidate branches has no inherent semantic order, whereas the injected <lvcr> positions form an ordered sequence for the language model. Ranking by cosine similarity with the question representation provides a deterministic, question-dependent ordering that places the most directly relevant branch first while retaining the remaining branches as complementary context. This design avoids the information bottleneck of selecting only one hypothesis and removes the permutation ambiguity of injecting an unordered branch set. Section IV-C evaluates these alternatives and shows that ranked multi-branch injection is more effective than either top-1 or unranked injection.

## IV. EXPERIMENTS

## A. Experimental Setup

a) Training data.: We use VISUAL COT as the primary training data [12]. This large-scale VQA dataset contains question-answer examples with bounding-box evidence annotations. After formatting and filtering, our training set contains 417,453 image-question-answer examples. Each example contains one annotated bounding box, which we use to construct the positive visual evidence representation for the contrastive branch-evidence alignment loss.

b) Model and training.: We build LVCR on top of Qwen2.5-VL-Instruct backbones and report results for both the 3B and 7B variants [4]. For 3B models, LVCR uses K = 4 latent branches, while for 7B models it uses $K = 8 ;$ in both cases, the number of inserted <lvcr> tokens is set equal to K. The answer prediction loss is optimized jointly with the contrastive branch-evidence alignment loss using $\lambda _ { \mathrm { n c e } } ~ = ~ 0 . 1$ , contrastive temperature $\tau \ = \ 0 . 0 7$ , and soft assignment temperature $\tau _ { s } = 0 . 2$ . We fine-tune the language model and LVCR parameters while keeping the vision encoder and multimodal merger frozen. Training uses bfloat16 precision, gradient checkpointing, DeepSpeed ZeRO-2, AdamW with learning rate $1 \times 1 0 ^ { - 5 }$ , weight decay 0.1, warmup ratio 0.03, and a cosine learning-rate schedule. We train the 3B models on two NVIDIA A100 80GB GPUs and the 7B models on four NVIDIA A100 80GB GPUs.

c) Inference.: At inference time, we insert the LVCR latent reasoning scaffold before answer generation using a <lvcr\_start> token, K <lvcr> tokens, and a <lvcr\_end> token. Before injection, we rank all K branches by their cosine similarity to the question representation and place the most question-relevant branch in the first <lvcr> position. All ranked branches are therefore available to the decoder before it emits the final textual answer. We also evaluate ablations without branch ranking and with top-1 injection in Section IV-C0b, where top-1 denotes using only the highest-ranked branch under the same cosine-similarity criterion. All benchmark evaluations use deterministic decoding.

d) Evaluation benchmarks.: We evaluate LVCR on perception-intensive multiple-choice visual question answering benchmarks, including $V ^ { * }$ Bench, MMVP, and BLINK. $V ^ { * }$ Bench evaluates fine-grained visual detail search and relative spatial reasoning, and we report the overall score $( V ^ { * } )$ as well as its Direct Attributes $( V _ { \mathrm { D . A . } } ^ { * } )$ and Relative Position $( V _ { \mathrm { R . P . } } ^ { * } )$ subsets [40]. MMVP evaluates whether MLLMs can distinguish fine-grained visual differences under subtle image perturbations [41]. For BLINK, we report results on Counting, IQ Test, Jigsaw, Relative Reflectance, and Spatial Relation, which cover perception-intensive skills such as counting, abstract visual reasoning, reflectance comparison, and spatial understanding [42]. For all benchmarks, the model is prompted to answer directly with the option letter, and accuracy is computed by exact matching against the ground-truth option after standard answer extraction. We compare LVCR with the corresponding Qwen2.5-VL-Instruct backbones and additional multimodal reasoning baselines under the same evaluation protocol.

e) Baselines.: We compare LVCR with baselines at both 3B and 7B scales. At the 3B scale, we include Qwen2.5- VL-3B as the backbone baseline [4], Vision-R1 as a recent visual reasoning baseline [43], a baseline trained with standard supervised fine-tuning (SFT) on the same instruction-format data, denoted as Vanilla SFT, and LVR as the single-path latent visual reasoning baseline [13]. At the 7B scale, we compare against Qwen2.5-VL-7B, PAPO [44], Vision-R1, PixelReasoner [45], Vanilla SFT, and LVR [13]. PAPO, Vision-R1, and PixelReasoner provide comparisons with recent multimodal reasoning methods. All methods are evaluated using the same prompting, deterministic decoding, and answer-extraction protocol.

## B. Main Results and Analysis

Tables I and II report the main results at the 3B and 7B scales. Overall, LVCR achieves competitive performance across the evaluated perception-intensive visual reasoning benchmarks. At the 3B scale, LVCR obtains the best or tied-best results on five out of nine metrics, including the overall $V ^ { * }$ score, $V _ { \mathrm { D . A . } } ^ { * }$ Jigsaw, Relative Reflectance, and Spatial Relation. Compared with the Qwen2.5-VL-3B backbone, LVCR improves the overall $V ^ { * }$ score and both $V ^ { * }$ subsets, suggesting that branchaware latent reasoning is beneficial for fine-grained visual search and attribute-level perception. Compared with the single path LVR baseline, LVCR improves most metrics, especially on $V ^ { * } , V _ { \mathrm { R . P . } } ^ { * }$ , Jigsaw, and Spatial Relation. This indicates that representing multiple question-conditioned visual hypotheses can provide more useful region-level evidence than relying on a single latent reasoning trajectory.

At the 7B scale, LVCR remains strong against a broader set of baselines, including PAPO, Vision-R1, PixelReasoner, Vanilla SFT, and LVR. LVCR achieves the best or tied-best results on $V _ { \mathrm { D . A . } } ^ { * }$ , MMVP, Counting, IQ-Test, and Jigsaw, and obtains second-best performance on several other subsets. The gains over the Qwen2.5-VL-7B backbone are particularly clear on MMVP and BLINK subsets, where the model needs to distinguish subtle visual differences, count multiple objects, or reason over spatial and perceptual relations. Compared with LVR, LVCR further improves performance on MMVP, Counting, IQ-Test, Jigsaw, and Spatial Relation, showing that the proposed multi-branch formulation brings consistent benefits beyond single-path latent visual reasoning. These results support the motivation that complex visual reasoning often involves multiple plausible evidence regions, and that preserving several latent visual hypotheses before answer generation can improve perception-intensive reasoning.

![](images/75c819ee57a5e67898ae3d944511c36fa29411f2a6e7c8e4f8314ec30455c94c.jpg)  
Fig. 2. Ablation study on the NCE loss coefficient using LVCR-3B with $K { = } 4$ . We compare the default setting with $\lambda _ { \mathrm { N C E } } = 0 . 1$ against a variant trained without the NCE loss, i.e., $\lambda _ { \mathrm { N C E } } = 0 .$ Removing the NCE loss leads to consistent performance drops on $V ^ { * }$ , MMVP, and BLINK, indicating that contrastive branch-evidence alignment helps latent branches capture questionrelevant visual cues.

The comparison with Vanilla SFT also suggests that the improvements do not simply come from additional supervised fine-tuning. Although Vanilla SFT improves some individual metrics, its performance is less stable across benchmarks and model scales. For example, at the 3B scale, Vanilla SFT improves Spatial Relation but underperforms the backbone on the overall $V ^ { * }$ score and relative-position reasoning. This suggests that standard supervised fine-tuning alone may not reliably improve fine-grained visual reasoning. In contrast, LVCR combines answer supervision with bbox-guided contrastive branch-evidence alignment, which encourages the latent branches to capture answer-relevant visual regions while still supporting final answer prediction.

The results also reveal that LVCR is not uniformly best on every benchmark. Vision-R1 performs strongly on MMVP and $V _ { \mathrm { R . P . } } ^ { * }$ at the 3B scale, while PixelReasoner achieves the best overall $V ^ { * }$ and $V _ { \mathrm { R . P . } } ^ { * }$ results at the 7B scale. These cases suggest that different reasoning methods may emphasize different visual capabilities, such as pixel-level reasoning, visual robustness, or spatial localization. Nevertheless, LVCR provides broad improvements over the backbone and single-path latent reasoning baselines across many perception-intensive subsets. This demonstrates that branch-aware latent visual reasoning is a useful and complementary mechanism for improving multimodal visual reasoning.

<table><tr><td>Model</td><td> $\overline { { V ^ { * } } }$ </td><td> $\overline { { V _ { \mathrm { D . A . } } ^ { * } } }$ </td><td> $\overline { { V _ { \mathrm { R . P . } } ^ { * } } }$ </td><td>MMVP</td><td>Counting</td><td>IQ-Test</td><td>Jigsaw</td><td>Relative Reflect</td><td>Spatial Relation</td></tr><tr><td>Qwen2.5-VL-3B</td><td>73.8</td><td>80.9</td><td>63.2</td><td>60.0</td><td>65.8</td><td>24.7</td><td>53.3</td><td>42.5</td><td>81.1</td></tr><tr><td>Vision-R1</td><td>71.7</td><td>72.2</td><td>71.1</td><td>66.0</td><td>60.0</td><td>20.0</td><td>44.7</td><td>32.8</td><td>76.2</td></tr><tr><td>Vanilla SFT</td><td>70.7</td><td>80.0</td><td>56.6</td><td>59.0</td><td>67.5</td><td>22.0</td><td>52.0</td><td>42.5</td><td>84.6</td></tr><tr><td>LVR</td><td>70.7</td><td>79.1</td><td>57.9</td><td>60.7</td><td>64.2</td><td>28.7</td><td>52.7</td><td>40.3</td><td>81.1</td></tr><tr><td>LVCR (Ours)</td><td>75.4</td><td>82.6</td><td>64.5</td><td>61.0</td><td>62.5</td><td>28.0</td><td>54.0</td><td>42.5</td><td>84.6</td></tr></table>

TABLE I  
MAIN RESULTS FOR 3B-SCALE MODELS. ALL NUMBERS ARE ACCURACY (%). VISION-R1 AND LVR ARE REPRODUCED ON THE 3B BACKBONE USING THEIR OFFICIAL IMPLEMENTATIONS. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD, AND THE SECOND-BEST RESULT IS UNDERLINED.

<table><tr><td>Model</td><td> $\overline { { V ^ { * } } }$ </td><td> $\overline { { V _ { \mathrm { D . A . } } ^ { * } } }$ </td><td> $\overline { { V _ { \mathrm { R . P . } } ^ { * } } }$ </td><td>MMVP</td><td>Counting</td><td>IQ-Test</td><td>Jigsaw</td><td>Relative Reflect</td><td>Spatial Relation</td></tr><tr><td>Qwen2.5-VL-7B</td><td>78.5</td><td>81.7</td><td>73.7</td><td>66.7</td><td>66.7</td><td>26.0</td><td>52.0</td><td>38.8</td><td>87.4</td></tr><tr><td>PAPO</td><td>36.1</td><td>25.2</td><td>52.6</td><td>54.3</td><td>66.7</td><td>29.3</td><td>52.0</td><td>39.6</td><td>88.8</td></tr><tr><td>Vision-R1</td><td>70.2</td><td>70.4</td><td>69.7</td><td>46.7</td><td>51.7</td><td>26.7</td><td>27.3</td><td>44.8</td><td>66.4</td></tr><tr><td>PixelReasoner</td><td>80.1</td><td>81.7</td><td>77.6</td><td>67.0</td><td>65.0</td><td>23.3</td><td>52.7</td><td>40.3</td><td>88.1</td></tr><tr><td>Vanilla SFT</td><td>79.1</td><td>82.6</td><td>73.7</td><td>65.7</td><td>67.5</td><td>26.7</td><td>45.3</td><td>33.6</td><td>88.8</td></tr><tr><td>LVR</td><td>79.8</td><td>82.3</td><td>76.1</td><td>70.5</td><td>67.3</td><td>27.7</td><td>52.7</td><td>42.5</td><td>86.2</td></tr><tr><td>LVCR (Ours)</td><td>79.6</td><td>82.6</td><td>76.3</td><td>72.3</td><td>70.0</td><td>29.3</td><td>54.7</td><td>42.5</td><td>88.1</td></tr></table>

TABLE II  
MAIN RESULTS FOR 7B-SCALE MODELS. ALL NUMBERS ARE ACCURACY (%). FOR PAPO, VISION-R1, AND PIXELREASONER, WE EVALUATE THEIR PUBLICLY RELEASED 7B CHECKPOINTS. FOR LVR, WE RETRAIN THE MODEL FOLLOWING ITS OFFICIAL TRAINING RECIPE. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD, AND THE SECOND-BEST RESULT IS UNDERLINED.

![](images/27836f1afbfcf01a2625cce9b7ba46c8d06c8bdd8da988fbbd14aea98e6c8696.jpg)  
Fig. 3. Accuracy (%) of three branch injection strategies on $V ^ { * }$ , MMVP, and BLINK (aggregated). Ranked denotes our default LVCR strategy; Top-1 injects only the highest-ranked branch; Unranked injects all branches without ordering. Results are reported using LVCR-3B with K=4.

## C. Ablation Studies

a) Ablation on contrastive alignment.: We first study the effect of the contrastive branch-evidence alignment objective. To isolate its contribution, we retrain LVCR-3B by setting the coefficient of the NCE loss to zero, while keeping the remaining training configuration unchanged. As shown in Figure 2, removing the NCE loss consistently degrades performance across the evaluated benchmarks. In particular, the model without NCE loss drops from 75.4 to 72.3 on $V ^ { * }$ and from 54.0 to 52.9 on BLINK, while showing a smaller decrease on MMVP.

These results suggest that the NCE loss provides useful supervision for organizing latent branches around questionrelevant visual evidence. Without contrastive branch-evidence alignment, different latent branches are still trained through the answer prediction objective, but they receive weaker guidance on which visual regions or hypotheses they should specialize in. As a result, the learned branches may become less discriminative and less consistently aligned with the visual cues required by the question. The improvement brought by the NCE loss is especially visible on $V ^ { * }$ and BLINK, which require finegrained visual grounding and compositional perception. This supports our design choice of using contrastive supervision to encourage more informative and evidence-aware latent visual branches.

b) Inference strategy.: We ablate how latent branches are organized during inference. Our default LVCR strategy ranks all K branches by their cosine similarity to the question representation and injects them in descending order of relevance. We compare it with two alternatives: injecting only the top-1 ranked branch (Top-1) and injecting all branches without ranking (Unranked).

As shown in Figure 3, Ranked consistently achieves the best performance across $V ^ { * }$ , MMVP, and BLINK. On $V ^ { * }$ , Ranked obtains 75.4% accuracy, outperforming Top-1 by 1.1 points and Unranked by 3.2 points. On BLINK, the improvement is more pronounced, with Ranked exceeding Top-1 and Unranked by 1.2 and 3.8 points, respectively. The gap between Ranked and Top-1 suggests that relying on a single branch creates an information bottleneck, as complementary visual cues from other branches are discarded. Meanwhile, the weaker performance of Unranked despite using all branches indicates that relevance-based ordering provides useful structural guidance for latent visual reasoning. These results motivate ranked multi-branch injection as the default inference strategy.

## V. CONCLUSION

In this paper, we proposed LVCR, a branch-aware latent reasoning framework for perception-intensive multimodal reasoning. LVCR generates multiple latent visual branches and uses bounding-box supervision with a soft contrastive objective to align each branch with answer-relevant visual evidence. At inference time, LVCR ranks these branches by question relevance and injects the selected latent tokens into the decoder for answer prediction. Experiments at both 3B and 7B scales show that LVCR achieves consistent improvements over competitive baselines. These results suggest that structured multi-branch latent reasoning is a promising direction for visually grounded latent-space reasoning.

## REFERENCES

[1] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23 716–23 736, 2022.

[2] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[3] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892–34 916, 2023.

[4] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[5] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “Vqa: Visual question answering,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2425–2433.

[6] D. A. Hudson and C. D. Manning, “Gqa: A new dataset for real-world visual reasoning and compositional question answering,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6700–6709.

[7] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24 824–24 837, 2022.

[8] Z. Zhang, A. Zhang, M. Li, H. Zhao, G. Karypis, and A. Smola, “Multimodal chain-of-thought reasoning in language models,” arXiv preprint arXiv:2302.00923, 2023.

[9] T. Gupta and A. Kembhavi, “Visual programming: Compositional visual reasoning without training,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14 953–14 962.

[10] D. Surís, S. Menon, and C. Vondrick, “Vipergpt: Visual inference via python execution for reasoning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 888–11 898.

[11] P. Lu, B. Peng, H. Cheng, M. Galley, K.-W. Chang, Y. N. Wu, S.-C. Zhu, and J. Gao, “Chameleon: Plug-and-play compositional reasoning with large language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 43 447–43 478, 2023.

[12] H. Shao, S. Qian, H. Xiao, G. Song, Z. Zong, L. Wang, Y. Liu, and H. Li, “Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning,” Advances in Neural Information Processing Systems, vol. 37, pp. 8612– 8642, 2024.

[13] B. Li, X. Sun, J. Liu, Z. Wang, J. Wu, X. Yu, H. Chen, E. Barsoum, M. Chen, and Z. Liu, “Latent visual reasoning,” arXiv preprint arXiv:2509.24251, 2025.

[14] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[15] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[16] X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer et al., “Pali: A jointly-scaled multilingual language-image model,” arXiv preprint arXiv:2209.06794, 2022.

[17] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49 250–49 267, 2023.

[18] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 296–26 306.

[19] S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan et al., “Cambrian-1: A fully open, visioncentric exploration of multimodal llms,” Advances in Neural Information Processing Systems, vol. 37, pp. 87 310–87 356, 2024.

[20] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini et al., “Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 91–104.

[21] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.

[22] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[23] H. Zhang, K. Tang, L. Kang, S. D. Campbell, Z. Wang, and D. H. Werner, “A comprehensive review of large language models for nanophotonics: From surrogate modeling to autonomous design,” arXiv preprint arXiv:2608.18279, 2026.

[24] K. Tang, J. Wang, S. Ren, and Z. Wang, “CT-∆Bench: A benchmark for longitudinal 3d medical imaging difference reporting with vision language models,” arXiv preprint arXiv:2608.11534, 2026.

[25] Z. Chen, Q. Zhou, Y. Shen, Y. Hong, Z. Sun, D. Gutfreund, and C. Gan, “Visual chain-of-thought prompting for knowledge-based visual reasoning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 1254–1262.

[26] C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan, “Visual chatgpt: Talking, drawing and editing with visual foundation models,” arXiv preprint arXiv:2303.04671, 2023.

[27] Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z. Liu, C. Liu, M. Zeng, and L. Wang, “Mm-react: Prompting chatgpt for multimodal reasoning and action,” arXiv preprint arXiv:2303.11381, 2023.

[28] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, Q. Ye, and F. Wei, “Grounding multimodal large language models to the world,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 51 575–51 598.

[29] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, “Shikra: Unleashing multimodal llm’s referential dialogue magic,” arXiv preprint arXiv:2306.15195, 2023.

[30] H. You, H. Zhang, Z. Gan, X. Du, B. Zhang, Z. Wang, L. Cao, S.-F. Chang, and Y. Yang, “Ferret: Refer and ground anything anywhere at any granularity,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 57 153–57 180.

[31] Z. Li, Q. Xu, D. Zhang, H. Song, Y. Cai, Q. Qi, R. Zhou, J. Pan, Z. Li, V. Tu et al., “Groundinggpt: Language enhanced multi-modal grounding model,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 6657–6678.

[32] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia, “Spatialvlm: Endowing vision-language models with spatial reasoning capabilities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 455–14 465.

[33] A.-C. Cheng, H. Yin, Y. Fu, Q. Guo, R. Yang, J. Kautz, X. Wang, and S. Liu, “Spatialrgpt: Grounded spatial reasoning in vision-language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 135 062–135 093, 2024.

[34] S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, and Y. Tian, “Training large language models to reason in a continuous latent space,” arXiv preprint arXiv:2412.06769, 2024.

[35] J. Wang, K. Tang, Z. Wang, and S. Ren, “Scdm: Scalable causal discovery in nonlinear temporal systems with meta-learning,” arXiv preprint arXiv:2610.29992, 2026.

[36] T.-H. Pham and C. Ngo, “Multimodal chain of continuous thought for latent-space reasoning in vision-language models,” arXiv preprint arXiv:2508.12587, 2025.

[37] G. Zheng, P. Bo, L. Liu, Z. Cong, K. Tang, and C. Zhang, “Corosam: enhancing sam with frequency and orientation awareness for coronary artery segmentation in x-ray angiography,” in 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2025, pp. 3349–3356.

[38] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.

[39] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[40] P. Wu and S. Xie, “v<sup>∗</sup>: Guided visual search as a core mechanism in multimodal llms,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 084–13 094.

[41] S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, and S. Xie, “Eyes wide shut? exploring the visual shortcomings of multimodal llms,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9568–9578.

[42] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna, “Blink: Multimodal large language models can see but not perceive,” in European Conference on Computer Vision. Springer, 2024, pp. 148–166.

[43] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Z. Xu, X. Tang, Y. Hu, and S. Lin, “Vision-r1: Incentivizing reasoning capability in multimodal large language models,” arXiv preprint arXiv:2503.06749, 2025.

[44] Z. Wang, X. Guo, S. Stoica, H. Xu, H. Wang, H. Ha, X. Chen, Y. Chen, M. Yan, F. Huang et al., “Perception-aware policy optimization for multimodal reasoning,” arXiv preprint arXiv:2507.06448, 2025.

[45] H. Wang, A. Su, W. Ren, F. Lin, and W. Chen, “Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning,” arXiv preprint arXiv:2505.15966, 2025.