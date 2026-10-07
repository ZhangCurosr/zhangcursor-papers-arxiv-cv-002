# Emoception: Selective Affective Layer Fine-Tuning of Video Vision Transformers for Player Arousal Change Recognition from Gameplay Footage

Yi Xia<sup>1,\*</sup>, Ibrahim Khan<sup>2</sup>, Mury Fajar Dewantoro<sup>1</sup>, Wenwen Ouyang<sup>3</sup>, and Ruck Thawonmas<sup>2</sup>

<sup>1</sup>Graduate School of Information Science and Engineering, Ritsumeikan University, Osaka, Japan,

yi.xia@ice.ci.ritsumei.ac.jp, fajar.dewantoro.mury@ice.ci.ritsumei.ac.jp

<sup>2</sup>College of Information Science and Engineering, Ritsumeikan University, Osaka, Japan,

khan@fc.ritsumei.ac.jp, ruck@is.ritsumei.ac.jp

<sup>3</sup>Alumni Association of Carnegie Mellon University, Pittsburgh, PA, USA, wenweno@alumni.cmu.edu <sup>\*</sup>Corresponding author: yi.xia@ice.ci.ritsumei.ac.jp

Abstract—This paper proposes Selective Affective Layer Fine-Tuning (SALFT), an efficient adaptation framework for Video Vision Transformers in player arousal recognition from gameplay. To bypass computationally expensive full fine-tuning, SALFT introduces a selection criterion based on the L2-norm change in layer parameters after brief adaptation, directly measuring representational shifts and providing a more stable basis than gradient-based alternatives. Evaluated via five-fold crossvalidation on the Arousal Video Game AnnotatIoN dataset, SALFT achieves performance comparable to full fine-tuning across all games without statistically significant degradation (p > 0.05), while updating only ≈8% of parameters (over 92% reduction). Notably, in one game, SALFT consistently outperforms both full fine-tuning and the best baseline across all metrics and folds, reaching the theoretical minimum p-value (p = 0.0625, exact two-sided Wilcoxon signed-rank test). Additionally, we introduce an interpretability method to trace attention patterns, enhancing model transparency. These results establish SALFT as an effective and efficient approach for affective game computing. GitHub repo for code, data, and supplementary materials: https://github.com/Yi-Xia-2010/Emoception-SALFT

Index Terms—Affective computing; Arousal modelling; Video Games

## I. INTRODUCTION

Recognizing players’ emotional dynamics during gameplay is an emerging focus in affective computing [1]. In particular, modeling arousal changes, a key dimension of affect, offers insight into player engagement and cognitive states. Recent studies have explored subject-agnostic arousal recognition in games using game footage video (GFV) [2], [3] or GFV combined with gameplay telemetry [4], without requiring physiological signals.

Building on this line of work, we target subject-agnostic arousal change recognition from GFV and propose Selective

Affective Layer Fine-Tuning (SALFT), a layer-freezing finetuning framework for the Video Vision Transformer (ViViT) [5]. While ViViT effectively captures spatiotemporal patterns, its high complexity makes full fine-tuning computationally expensive and prone to overfitting with limited data. SALFT mitigates these issues by introducing a novel, task-oriented selection criterion, the L2-norm change in layer parameters, to identify and fine-tune only the most responsive transformer block, significantly improving parameter efficiency.

We evaluate SALFT on the Arousal Video Game Annotation (AGAIN) dataset [6], which contains gameplay videos from nine games across three genres. Using rigorous 5-fold crossvalidation with temporal block splitting and comprehensive statistical evaluation, we compare SALFT against multiple baselines including full ViViT fine-tuning, ResNet+LSTM, and exploratory LoRA implementations. Results show that SALFT achieves performance comparable to full fine-tuning on most games, while fine-tuning only a small fraction of parameters, and significantly outperforms it in specific cases (e.g., Gun). To complement these findings, we introduce an efficient interpretability method for ViViT that produces finegrained spatiotemporal relevance maps, revealing how the model attends to gameplay dynamics when predicting arousal changes.

Emoception, a blend of emotion and inception, captures the conceptual foundation of our work: understanding how the player’s affective states, particularly arousal, emerge and evolve during gameplay. This reflects our focus on investigating temporally embedded emotional cues within game footage, as interpreted through the representational capacity of a spatiotemporal video model. Our contributions are as follows:

• We propose SALFT, an efficient adaptation framework for ViViT that introduces a novel, directly measurable criterion (L2-norm change) for selecting the most taskresponsive layer to fine-tune, improving both parameter efficiency and performance in arousal recognition.

• A rigorous 5-fold cross-validation evaluation on the

AGAIN dataset demonstrates that SALFT consistently matches or numerically exceeds full fine-tuning performance across all nine games with only ∼8% trainable parameters. Comparisons with strong baselines, including a pixel-based ResNet+LSTM, confirm its competitiveness.

• We propose an interpretability method for ViViT that generates efficient, fine-grained spatiotemporal relevance maps, offering insights into model decision-making for predicting arousal changes.

## II. RELATED WORK

## A. Subject-Agnostic Affective Computing in Games

A trend in affective game computing is the development of subject-agnostic models that infer player emotion from in-game signals alone, avoiding intrusive physiological sensors [1]. Seminal research using gameplay footage showed that deep networks can predict player arousal directly from raw game frames [2] and audio-visual streams [3]. More recently, large-scale annotated datasets such as AGAIN [6] have advanced gameplay telemetry analysis. Building on this, the study [4] demonstrates the potential of multimodal fusion through specialized architectures like ViViT that integrate video and context. Separately, [7] showcases the power of Multimodal Large Language Models to process visual and text prompts for inferring player engagement. These studies highlight video’s dual role as both an affective cue and a stimulus, motivating our focus on efficient strategies for subject-agnostic arousal recognition from gameplay footage.

## B. Efficient Adaptation of Video Transformers

Transfer learning with large pre-trained video transformers such as ViViT [5] has proven effective for downstream tasks like player arousal recognition [4], but full fine-tuning remains computationally expensive and susceptible to overfitting. To mitigate this, strategies such as explicit regularization [8]–[10] and Parameter-Efficient Fine-Tuning (PEFT) have been proposed. PEFT methods reduce trainable parameters through various means, including addition-based techniques like Adapters [11] and Soft Prompts [12], [13], reparameterization approaches such as LoRA [14] and its variants [15], [16], and selection-based methods that fine-tune a subset of parameters [17]–[19].

Recent selective fine-tuning approaches face practical limitations. SubTuning requires conducting multiple experiments to construct a fine-tuning profile for block selection, where each experiment fine-tunes only one block while freezing the others; this process is computationally intensive [20]. AIM-Fair relies on synthetic data generation [21]. Feedforwardbased Parameter Selection (FPS) ranks parameters by the product of magnitudes and input activations [22]. While FPS efficiently identifies salient parameters through a deterministic, forward-pass-only metric, its selection is static and decoupled from the optimization dynamics that drive adaptation during fine-tuning. Furthermore, the lack of an official implementation precludes rigorous reproducibility.

Most closely related to our work is surgical fine-tuning [19], which efficiently adapts a single contiguous block selected via heuristic criteria like Relative Gradient Norm (RGN) or Signal-to-Noise Ratio (SNR). However, as shown in the original work, these heuristics do not consistently select the block that maximizes final accuracy compared to an exhaustive per-block validation strategy (“Cross-Val”), revealing a reliability gap. Motivated by the need for a more direct and stable selection signal than the optimization-sensitive gradient heuristics, we propose SALFT. Our key innovation is a task-oriented selection heuristic: the L2-norm change in layer parameters after brief fine-tuning. This metric directly quantifies how much a layer’s parameters adjust in response to the affective recognition task, offering a more stable and interpretable criterion for layer responsiveness. Our approach enhances arousal recognition while maintaining the efficiency of single-run, data-parsimonious adaptation.

## C. Explainability in Video Transformers

While visual attribution methods were largely pioneered for CNNs, evolving from the seminal Class Activation Mapping (CAM) [23] to influential successors like Grad-CAM [24] and others [25]–[27], their architectural specificity necessitated the development of new approaches tailored for Transformers. For Transformers, these methods range from class-agnostic attention visualization like Rollout [28] to class-specific attribution. The latter typically either propagate relevance backward through the network, as in the Layer-wise Relevance Propagation (LRP)-based work of Chefer et al. [29], or fuse attention with gradients at the layer level, as in Grad-SAM [30]. Complementing these attribution methods, other works like Video Transformer Concept Discovery [31] focus on discovering higher-level semantic concepts.

Recognizing player arousal from subtle, transient gameplay cues requires fine-grained spatiotemporal attribution, a need unmet by concept-based methods like VTCD [31]. To this end, we propose a hybrid method for ViViT that integrates the gradient-weighting principle of Grad-SAM [30] with the multi-layer rollout framework of Chefer et al. [29]. This design mitigates the prohibitive computational cost of directly adapting LRP to ViViT while still producing pixel-level spatiotemporal maps for detailed attribution.

## III. METHODOLOGY

## A. Task Reformulation and data preprocessing

In the original AGAIN dataset study [6], arousal change prediction was formulated as a binary classification task. Training samples were generated by applying a threshold to the magnitude of arousal change, retaining only segments with significant increases or decreases while discarding stable periods. The resulting model was trained to predict whether arousal would increase or decrease. Although this formulation represented a significant step toward modeling arousal changes in game contexts, it limits applicability to real-time scenarios where arousal may remain stable.

To better reflect real-world scenarios, the work in [4] reformulated the task as a three-class classification problem, predicting whether arousal would increase, decrease, or remain stable. Unlike the aforementioned threshold-based approach, change direction was determined by directly comparing the mean arousal values of consecutive windows. Furthermore, 1- second sliding windows were employed to expand the dataset and capture fine-grained temporal dynamics.

Following [4], we adopt the same preprocessing strategy while focusing exclusively on video data. For each time point t, a sample consists of two consecutive 3-second windows (t to $t + 6 )$ , yielding a 6-second segment. All samples are retained regardless of change magnitude. For each segment, 32 uniformly spaced frames are extracted, resized to 224×224 pixels, and fed into the ViViT-based modeling pipeline to match its input specification. Threshold selection and tuning, as conducted in [3], [6], are beyond the scope of this study and thus not considered. Following preprocessing, the number of samples per arousal change class across all games is shown in the last three columns of Table I.

## B. Selective Affective Layer Fine-Tuning

Our SALFT framework consists of two consecutive stages: (1) exploratory full fine-tuning to identify the layer whose parameters deviate most from the pretrained initialization, and (2) second-stage selective fine-tuning in which only the identified layer remains trainable while all others are frozen. Unlike Surgical Fine-Tuning which relies on indirect, gradient-based heuristics (RGN/SNR) to approximate layer importance, our criterion directly measures the actual parameter change induced by the task. This provides a more stable and interpretable signal for selecting the most responsive layer. Throughout the entire fine-tuning framework, the standard cross-entropy loss is consistently employed for model optimization. The procedure is illustrated in Algorithm 1 and detailed below.

First-Stage: Exploratory Full Fine-Tuning:

1) Initial Setup. Let $\Theta ^ { ( 0 ) } = \{ \pmb { \theta } _ { 1 } ^ { ( 0 ) } , \dots , \pmb { \theta } _ { L } ^ { ( 0 ) } \}$ denote the pretrained parameters of all layers $l ~ \in ~ \{ 1 , \ldots , L \}$ where each $\theta _ { l }$ includes both weights and biases. We fine-tune the entire model on the downstream corpus, saving a checkpoint at the end of each epoch e. Finetuning continues until the pattern of layer-wise parameter changes is deemed stable according to a correlationbased convergence criterion. Since the classifier head, is re-initialized to match the downstream class set, its parameters are expected to change accordingly. Consequently, we exclude the classifier layer, L, from the displacement computations and stability analysis; all subsequent operations focus exclusively on the remaining $L - 1$ non-classifier layers.

2) Layer-wise parameter aggregation. For each nonclassifier layer l, which may comprise multiple subcomponents (e.g., query, key, value projections and feedforward blocks), we aggregate the parameters at epoch e by flattening and concatenating all parameters within the layer into a single vector:

$$
\begin{array} { r } { \mathbf { v } _ { l } ^ { ( e ) } = \mathrm { c o n c a t } \big ( \mathsf { v e c } ( \theta _ { l , 1 } ^ { ( e ) } ) , ~ \ldots , ~ \mathsf { v e c } ( \theta _ { l , K _ { l } } ^ { ( e ) } ) \big ) \in \mathbb { R } ^ { d _ { l } } , } \end{array}\tag{1}
$$

where $K _ { l }$ is the number of sub-modules in layer l and $d _ { l }$ the resulting dimensionality.

Algorithm 1 Selective Affective Layer Fine-Tuning   
Require: Pretrained parameters $\Theta ^ { ( 0 ) }$ ; correlation threshold   
$\tau _ { \rho } = 0 . 8 0$   
Ensure: Final model with only one non-classifier layer and   
the classifier layer fine-tuned   
//First-Stage: Exploratory Full Fine-Tuning   
1: Initialize epoch counter: $e \gets 0$   
2: Initialize layer-wise distance vector: $\pmb { \Delta } ^ { ( 0 ) }  \mathbf { 0 }$   
3: repeat   
4: $e  e + 1$   
5: Fine-tune all layers for one epoch using current train  
ing data   
6: for $l = 1$ to $L - 1$ do   
7: ${ \bf v } _ { l } ^ { ( e ) }  { \bf C }$ ONCATFLATTEN $( \pmb { \theta } _ { l , * } ^ { ( e ) } )$   
8: $\bar { \Delta _ { l } ^ { ( e ) } }  \| \mathbf { v } _ { l } ^ { ( e ) } - \mathbf { v } _ { l } ^ { ( 0 ) } \| _ { 2 }$   
9: end for   
10: $\Delta ^ { ( e ) }  [ \Delta _ { 1 } ^ { ( e ) } , \dotsc , \Delta _ { L - 1 } ^ { ( e ) } ]$   
11: $( \rho , p ) \gets \mathrm { S P E A R M A N } \big ( \Delta ^ { ( e - 1 ) } , \Delta ^ { ( e ) } \big )$   
12: until $\rho \ge \tau _ { \rho }$ and $p < 0 . 0 5$   
13: $e ^ { \star }  e$   
14: $l ^ { \star } \gets \arg \operatorname* { m a x } _ { l } \Delta _ { l } ^ { ( e ^ { \star } ) }$   
//Second-Stage: Selective Fine-Tuning   
15: Reset model parameters to $\Theta ^ { ( 0 ) }$   
16: Freeze all layers except $l ^ { \star }$ and L   
17: while not converged do   
18: Fine-tune only layers $l ^ { \star }$ and $L$   
19: end while   
return The selectively fine-tuned model

3) Layer-wise displacement. For each layer, we compute the $\ell _ { 2 }$ distance between its parameter vector at epoch e and the corresponding original parameter vector, defined as

$$
\Delta _ { l } ^ { ( e ) } = \left. \mathbf { v } _ { l } ^ { ( e ) } - \mathbf { v } _ { l } ^ { ( 0 ) } \right. _ { 2 } ,\tag{2}
$$

where ${ \bf v } _ { l } ^ { ( e ) }$ denotes the parameter vector of layer l at epoch $e ,$ and ${ \bf v } _ { l } ^ { ( 0 ) }$ denotes the original pretrained parameter vector of the same layer. We then form a displacement vector using the displacement values (2) as follows:

$$
\Delta ^ { ( e ) } = \big ( \Delta _ { 1 } ^ { ( e ) } , \Delta _ { 2 } ^ { ( e ) } , \dots , \Delta _ { L - 1 } ^ { ( e ) } \big ) .\tag{3}
$$

4) Stability check. To assess the stabilization of the layerwise displacement pattern, we compute the Spearman rank correlation coefficient $\rho ^ { ( e ) }$ between $\Delta ^ { ( e - 1 ) }$ and $\Delta ^ { ( e ) }$ . We consider the pattern to be stabilized when the correlation is both statistically significant and exceeds a predefined threshold $\tau _ { \rho } ,$ as specified by the following condition:

$$
\rho ^ { ( e ) } \geq \tau _ { \rho } = 0 . 8 0 \quad \mathrm { a n d } \quad p \mathrm { - v a l u e < 0 . 0 5 . }\tag{4}
$$

This criterion indicates that the trend of displacements across layers remains consistent between successive epochs. The threshold $\tau _ { \rho }$ is motivated by commonly accepted interpretations of Spearman correlation coefficients, where values above 0.70 or 0.80 are generally regarded as reflecting strong or stable associations [32]– [34]. Note that this criterion is used solely to assess the stability of ranking trends; it does not imply linearity or causation. We define $e ^ { \star } = e$ as the last epoch before this condition is first met, and terminate fine-tuning at epoch $e ^ { \star }$

5) Layer selection. The layer with the largest displacement at epoch $e ^ { \star }$ is selected:

$$
l ^ { \star } = \arg \operatorname* { m a x } _ { l } \Delta _ { l } ^ { ( e ^ { \star } ) } .\tag{5}
$$

Second-Stage: Selective Fine-Tuning: We re-initialize the model to $\Theta ^ { ( 0 ) }$ , then perform a second fine-tuning run in which only the selected layer $l ^ { \star }$ and classifier head L are trainable; all other layers are frozen. The optimizer state is reset to avoid interference from the previous fine-tuning stage. This strategy concentrates the model’s learning capacity on the most taskrelevant layer while preserving the generalization ability of the pretrained model.

## C. Model interpretability

To interpret the spatiotemporal decision-making of ViViT models, we propose an attribution method inspired by the approaches of Chefer et al. [29] and Grad-SAM [30]. Chefer et al.’s approach computes layer-wise relevance via a backward propagation process using the computationally intensive LRP algorithm, which is then combined with class-specific gradients and aggregated across multiple layers to produce refined attributions. Grad-SAM combines gradients with attention and simplifies multi-layer attribution by averaging the attribution maps across layers. Our method aims to balance effectiveness and computational efficiency by integrating these ideas.

Our work, therefore, synthesizes these insights: we replace the expensive LRP relevance with the model’s raw attention maps A, integrating them into the sophisticated, non-averaging aggregation framework of Chefer et al. and adapting this process for ViViT’s spatiotemporal architecture. This results in a lightweight signal that maintains high explanatory fidelity at a fraction of the computational cost.

In the following, we detail our attribution method adapted to the ViViT architecture. For each Transformer encoder layer l, we extract the raw multi-head self-attention maps $\mathbf { A } ^ { l } \in$ $\mathbb { R } ^ { H \times N \times N }$ , where H is the number of heads and N the number of tokens. Averaging over heads yields

$$
\bar { \mathbf { A } } ^ { l } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathbf { A } _ { h } ^ { l } \in \mathbb { R } ^ { N \times N } .
$$

A backward pass is initialized from the target class logit $y _ { c }$ to compute the class-specific gradients $\frac { \partial y _ { c } } { \partial \mathbf { A } ^ { l } }$ for each layer l, which are head-averaged to obtain $\bar { \mathbf { G } } ^ { l }$

We define the layer-wise attention relevance matrix at layer l as

$$
\mathbf { R } ^ { l } = \mathrm { R e L U } \big ( \bar { \mathbf { A } } ^ { l } \odot \bar { \mathbf { G } } ^ { l } \big ) + \mathbf { I } ,\tag{6}
$$

where $\odot$ denotes the Hadamard product and identity matrix I accounts for residual connections. Note that $\mathbf { R } ^ { l }$ here is an intermediate relevance matrix capturing gradient-weighted attention patterns within layer $l ,$ which is distinct from the relevance computed via LRP defined in [29]. To ensure numerical stability and preserve token-wise relevance, each $\mathbf { R } ^ { l }$ is row-normalized so that each row sums to one before recursive aggregation. Starting from $\begin{array} { r } { \mathbf { C } ^ { L } = \mathbf { I } } \end{array}$ at the output layer $L ,$ cumulative relevance matrix is propagated backward via

TABLE I: Games in the AGAIN dataset with number of samples per arousal change class. The column “Abbrev.” stands for the abbreviated game identifier; “Inc.”, “Dec.”, and “Unchg.” denote arousal Increase, Decrease, and Unchanging, respectively.
<table><tr><td>Game</td><td>Abbrev.</td><td>Genre</td><td>Inc.</td><td>Dec.</td><td>Unchg.</td></tr><tr><td>ApexSpeed</td><td>apex</td><td>Racing</td><td>6629</td><td>4446</td><td>2375</td></tr><tr><td>Solid</td><td>solid</td><td>Racing</td><td>6520</td><td>4333</td><td>1947</td></tr><tr><td>TinyCars</td><td> $\mathtt { t i n y }$ </td><td>Racing</td><td>5944</td><td>4402</td><td>2279</td></tr><tr><td>Heist</td><td> $\mathtt { f p s }$ </td><td>Shooter</td><td>6561</td><td>3986</td><td>2337</td></tr><tr><td>Shootout</td><td>gallery</td><td>Shooter</td><td>7049</td><td>3121</td><td>2236</td></tr><tr><td>TopDown</td><td>topdown</td><td>Shooter</td><td>6914</td><td>3963</td><td>2724</td></tr><tr><td>The Endless</td><td>endless</td><td>Platformer</td><td>6610</td><td>3591</td><td>3010</td></tr><tr><td>Pirates</td><td>platform</td><td>Platformer</td><td>6156</td><td>3748</td><td>2593</td></tr><tr><td>RunN&#x27;Gun</td><td>gun</td><td>Platformer</td><td>6438</td><td>3863</td><td>2855</td></tr></table>

$$
\mathbf { C } ^ { l - 1 } = \mathbf { C } ^ { l } \cdot \mathrm { n o r m a l i z e } ( \mathbf { R } ^ { l } ) , \quad l = L , \ldots , 1 .\tag{7}
$$

The final relevance matrix $\mathbf { C } ^ { 0 }$ encodes the token-level relevance scores with respect to the target class, serving as the basis for the model’s explanation.

Unlike 2D Vision Transformers, ViViT operates on spatiotemporal tubelets, where each token corresponds to a 2D spatial patch extended over a fixed number of consecutive frames. We derive the spatiotemporal relevance map $\mathbf { M } _ { \mathrm { r e l } } \in$ $\mathbb { R } ^ { T \times H \times W }$ by selecting the [CLS] row from $\mathbf { C } ^ { 0 }$ and reshaping it into a spatiotemporal grid, with $T$ denoting the number of tubelets and $H \times W$ the spatial token layout. Each temporal slice is then replicated over its corresponding frames to produce per-frame relevance heatmaps, which are upsampled to $2 2 4 \times 2 2 4$ and overlaid on the corresponding input video frames, providing a fine-grained visualization of the model’s spatiotemporal attribution.

## IV. EXPERIMENT

## A. Dataset

This study employs the ‘clean’ subset of the AGAIN dataset [6], which comprises 995 two-minute gameplay sessions (from an original total of 1,116) from 122 participants. Each session consists of video recordings, in-game telemetry, and time-continuous self-reported arousal annotations. Data were collected from nine custom-developed games spanning racing, shooter, and platformer genres (Figure 1), using the abbreviated identifiers provided in the dataset files (the first three columns of Table I). The annotations were initially unbounded and subsequently normalized to the [0,1] range. Unlike previous approaches combining video and telemetry modalities [4], this work focuses exclusively on video and its associated arousal traces, given that the visual modality provides more informative cues for this task.

![](images/c1eeefd23543816e9ad423743e933fabd52142e962dccb77872504728fdef530.jpg)  
(a) apex

![](images/729bea01b02473dc56b45b7f2ec201b5bc9880e7bba80f5f8058419d1646f4fa.jpg)  
(b) solid

![](images/1d4a4e8a707a16204539b6b7caa6dbe80bd2a9ed13047d97f7285700c6a3497e.jpg)  
(c) tiny

![](images/3724592967f179670524bb034b56adc7e6ff1dcbaf827036a3f22844e041ed28.jpg)

![](images/4177141392ae9530a49e9f9636b3c03726d40f56c96b24f2dfdc19cf9f90006b.jpg)  
(d) fps

![](images/01dfb48a3b7f79b6ad924de635fe4766707ef9a8d9e0bb5eaf575b5410cecc60.jpg)

![](images/ab9483d240c1d7a586315829d4e3200b4f2a6af966bdcff63c892d4296d237e3.jpg)  
(g) endless

(e) gallery  
(f) topdown  
![](images/78270f4eccb33f838e6cbe0d3ca99be2bc75d4ea69660439f4513f670a0922c1.jpg)  
(h) platform

![](images/117bb91cebc883cebea361605bdacf73b4eb547b99ec63f73aa0ade7b0205726.jpg)  
(i) gun  
Fig. 1: Representative screenshots of each game in the AGAIN dataset.

## B. Data Split

To ensure rigorous evaluation while addressing the specific spatiotemporal characteristics of gameplay data, we employed a Block-Stratified 5-Fold Cross-Validation strategy. Our design rationale is threefold:

• Addressing Interaction-Driven Covariate Shift: Unlike passive media consumption, gameplay footage exhibits substantial variance due to diverse player strategies (e.g., aggressive vs. stealthy playstyles). A strict subject-level split risks evaluating the model on unseen interaction patterns, potentially conflating distinct playstyles with affect labels. By distributing user data across folds, we expose the model to the full manifold of gameplay behaviors, encouraging the learning of generalizable visual-state mappings rather than overfitting to specific interaction subsets.

• Validating Affect-Content Independence: A primary concern in random splitting is identity leakage. However, our analysis demonstrates that User Identity is a poor predictor of affect in this domain. As shown in the GitHub repo, t-SNE visualizations reveal that feature embeddings from different players are highly intermixed, indicating no distinct “user signatures” in the gameplay video. Furthermore, the high Intra-User Label Entropy (mean=0.75) confirms that individual sessions contain a diverse spectrum of arousal states. This high withinsubject variance forces the model to learn content-based features rather than relying on trivial subject-specific cues.

• Preventing Short-Term Data Leakage: We extract samples using a 1-second sliding window. Random splitting would introduce severe data leakage due to the high overlap between adjacent windows. To rigorously isolate this risk, we introduced non-overlapping temporal blocks as the atomic grouping unit. This ensures that explicit data overlap from the sliding window process is eliminated, maintaining strict disjointness between training and test sets at the sample level.

Our data partitioning protocol consists of three stages de signed to prevent temporal leakage and maintain class balance. First, temporal micro-blocking segments each gameplay session into non-overlapping 6-second blocks, with all 1-second sliding windows within a block sharing a group identifier. This ensures that samples from the same local temporal context remain together. Second, a stratified 5-fold cross-validation scheme is applied at the block level, preserving the distribution of arousal classes in each fold while keeping all windows from a given block within a single fold. This eliminates intra-block leakage and disrupts session-level temporal continuity. Finally, a nested split reserves 20% of blocks for testing, and divides the remaining 80% into training (72%) and validation (8%) sets, maintaining the same block-level grouping constraints. This protocol rigorously isolates temporal dependencies while providing a balanced evaluation across affective states.

## C. Experimental Setup

We adopt the ViViT-B/16×2 backbone [5], pretrained on Kinetics-400, and replace the classification head to match the target classes while keeping the backbone unchanged. Experiments are conducted on the AGAIN dataset comprising nine games (Section IV-A), where the model is independently fine-tuned and evaluated on each game to assess generalization across diverse gameplay scenarios. Fine-tuning is performed on an NVIDIA A100 GPU using the AdamW optimizer with a batch size of 4 and a learning rate of $5 \times 1 0 ^ { - 5 }$ ; other optimizer parameters are kept at their defaults. All results are reported from a single run due to computational constraints, with a fixed random seed (42) to ensure reproducibility. Spearman’s rank correlations are computed using SciPy’s spearmanr, with significance assessed at $\alpha \ = \ 0 . 0 5$ . Code, data, and supplementary materials are publicly available (see abstract).

## D. Baseline and Comparison Strategy

To evaluate the proposed fine-tuning framework, we compare it against several representative baselines. For a fair comparison, all methods are fine-tuned for 14 epochs, and we report the test performance achieved by the model checkpoint with the best validation performance. The 14-epoch limit is derived from the gradual unfreezing schedule [18], which we adapted to align with the ViViT model’s 14-layer architecture (embedding, 12 encoder layers, and classifier); however, following prior work reporting its inferior performance [19], this schedule was excluded as a baseline.

Our baseline selection also reflects practical computational constraints. For instance, while the Surgical Fine-Tuning paper [19] employs a rigorous cross-validation protocol (“Cross-Val”) for selecting the optimal layer block, it requires training a separate model for each candidate block, a cost-prohibitive for our large-scale video model across nine domains. This inefficiency directly motivates SALFT’s one-pass selection protocol. The reliability of this efficient approach is supported by a preliminary study (c.f. the GitHub repo), which shows that our L2 criterion selects the same layer as the Cross-Val.

Random Forest. Following the original AGAIN dataset paper [6], we implement a random forest model trained on game content features. Although our task differs by focusing on arousal changes prediction from GFV, this classical baseline provides a reference point. Prior work [4] reported a performance decline under the new task but evaluated only a single game; here, we extend the analysis to all nine games.

ViViT Full Fine-tuning. As a strong baseline, we fine-tune all parameters of the pretrained ViViT model. This standard approach in video understanding often achieves competitive results due to full model adaptability. Previous work [4] applied this to GFV, yielding strong performance for arousal change recognition. However, it incurs higher computational cost and risk of overfitting, especially with small or domainshifted datasets.

ResNet+LSTM (Full Fine-tuning). Motivated by [35], to benchmark against a different and widely used architecture for spatiotemporal modeling, we include a 3D-ResNet-50+LSTM model, which is fully fine-tuned on our task. This baseline allows us to evaluate the relative performance of the ViViT backbone and our method within a different architectural family.

SALFT on ResNet+LSTM. To demonstrate the generality of our proposed SALFT framework beyond the ViViT architecture, we also apply its two-stage procedure (exploratory full tuning followed by selective layer tuning) to the ResNet+LSTM model. This tests whether the task-aware layer selection criterion is effective and beneficial for a substantially different network structure.

ViViT with LoRA. To compare against a modern parameter-efficient fine-tuning (PEFT) method, we implement Low-Rank Adaptation (LoRA) [14] on the ViViT backbone. We configure LoRA to adapt the query, key, value, and final dense projection layers within each Transformer block. Starting from a standard configuration (learning rate $5 \times 1 0 ^ { - 4 }$ rank $r = 1 6 ,$ scaling factor α = 32), we observed that LoRA failed to converge (predicting only a single class). Even after increasing the learning rate to $1 \times 1 0 ^ { - 3 }$ , rank to $r = 6 4 .$ and α to 128, the model still did not converge. This baseline is included to illustrate the challenge of applying generic PEFT methods to our gameplay video task, and thereby highlights the advantage of SALFT’s task-aware, layer-wise selection strategy which successfully adapts the model.

## E. Evaluation.

Model performance is evaluated using three metrics: the weighted-average $F _ { 1 }$ -score (primary), the macro-average $F _ { 1 } -$ score, and accuracy. The weighted $F _ { 1 }$ -score is prioritized as it accounts for class frequency, providing a performance measure that better reflects the imbalanced label distribution. The macro $F _ { 1 }$ -score complements this by treating all classes equally, highlighting performance on minority classes. Accuracy is reported for completeness and direct comparison with prior work.

All results are reported as the mean and its 95% confidence interval across the five cross-validation folds. Performance differences are assessed via exact two-sided Wilcoxon signedrank tests separately for each game and metric, comparing

ViViT SALFT against (1) full fine-tuning ViViT and (2) the best-performing baseline. Given $N = 5$ , traditional significance $( p \ < \ 0 . 0 5 )$ is mathematically unattainable; thus, the theoretical minimum $( p \ : = \ : 0 . 0 6 2 5 )$ is utilized to denote consistent empirical superiority across all folds. Full statistics are provided in the GitHub repo.

## F. Comparison of Efficiency

In addition to predictive performance, we evaluate the efficiency of our proposed SALFT framework using hardwareagnostic metrics. This analysis compares SALFT applied to the ViViT model against the standard full fine-tuning of ViViT, focusing on the first fold of the 5-fold cross-validation. The exploratory full-model fine-tuning stage required by SALFT to determine layer selection is contained within this fold, making it a representative and fair basis for efficiency comparison. Other baselines (e.g., Random Forest, ResNet+LSTM) are excluded from this section due to their fundamentally different architectures and computational profiles, which would preclude a direct and meaningful comparison of fine-tuning efficiency.

1) Parameter Efficiency: To compare these two approaches, we quantify parameter efficiency by the absolute number of trainable parameters. We also report this value as a percentage of the total parameters of the base model. These metrics directly reflect storage cost (e.g., checkpoint size) and impact memory usage for optimizer states, providing key indicators of storage footprint and deployment practicality.

2) Computational Efficiency Analysis: We analyze computational efficiency by plotting the weighted $F _ { 1 }$ score per epoch on the validation set against cumulative computational cost measured in Floating Point Operations (FLOPs). For each method under comparison, the cost per epoch $( C o s t _ { e p o c h } )$ is estimated by measuring the FLOPs per fine-tuning step using torch.profiler after a warm-up phase, then multiplying by the number of steps per epoch S:

$$
C o s t _ { e p o c h } = F L O P s _ { s t e p } \times S .\tag{8}
$$

Here, $F L O P s _ { s t e p }$ is averaged over 10 iterations, each comprising a forward pass, backward pass, and optimizer update. We exclude the computational cost of validation inference, as it is negligible compared to fine-tuning. Validation involves only a forward pass, whereas fine-tuning requires both forward and backward passes, which are substantially more computationally intensive. Furthermore, the number of validation steps is typically smaller than the number of training steps, further reducing its relative impact. For our two-stage SALFT framework, the cumulative computational cost includes both the exploratory and selective fine-tuning stages, enabling a fair comparison with the baseline. Efficiency is illustrated by plotting weighted $F _ { 1 }$ against cumulative FLOPs, where methods achieving higher weighted $F _ { 1 }$ at lower FLOPs are deemed more efficient.

## G. Interpretability Methods Evaluation

1) Baselines: We evaluate the effectiveness of our proposed interpretability method on our arousal change recognition task using the fine-tuned model with SALFT from the first fold of the 5-fold cross-validation. Our approach is evaluated against three baselines:

• Rollout [28]: Vanilla attention rollout across all layers.

• Grad-CAM [24]: Grad-CAM adapted for the ViViT architecture, using the final encoder layer’s frame token embeddings as the feature map, with gradients computed based on the [CLS] token’s output.

• Grad-SAM [30]: Computes the element-wise product of attention maps and their ReLU-clipped gradients, averaged across all layers and heads; here, we adapt it specifically to the ViViT architecture.

In our evaluation, we exclude the class of LRP-based methods, such as those analyzed in [29]. Their adaptation to the ViViT architecture incurs prohibitive computational costs that lead to out-of-memory errors with our available hardware. This practical challenge highlights the need for more computationally efficient interpretability techniques, which is a primary motivation for our interpretability method.

2) Perturbation Analysis Protocol: For the quantitative evaluation of our explanation method, we follow the general methodology for perturbation analysis presented in [29]. For each sample, this analysis is performed against two distinct targets: (1) the model’s predicted class, to evaluate the fidelity of the explanation to the model’s internal decision-making process; and (2) the ground-truth class, to assess the plausibility of the explanation by examining its ability to highlight genuinely class-discriminative features. The evaluation is conducted on the test set of each game dataset.

The protocol involves sequentially masking tubelet embeddings over 10 discrete steps (from 10% to 90%) by zeroing them out after the initial embedding layer but prior to the transformer encoder. This follows two complementary schemes:

• Positive Perturbation (Deletion): The most relevant tubelets are masked first. A faithful explanation should identify crucial tokens, leading to a steep drop in model confidence upon their removal.

• Negative Perturbation (Preservation): The least relevant tubelets are masked first. In this case, model confidence should be largely maintained until important tokens are removed.

The primary metric is the Area Under the Curve (AUC) of the resulting confidence-vs-perturbation plot. For this analysis, the confidence is the model’s output probability for the initially predicted class. A higher-fidelity explanation is thus characterized by a lower AUC for positive perturbation and a higher AUC for negative perturbation.

## V. RESULTS AND DISCUSSION

## A. Overall Performance Comparison

As summarized in Table II, the proposed SALFT framework delivers strong performance across the nine AGAIN games. When compared to full fine-tuning, SALFT achieves numerically higher weighted F1 scores in 6 games, macro F1 in 5 games, and accuracy in 7 games, while updating only about 8% of parameters. It attains the best scores across all three metrics in 5 games. The only case where full fine-tuning shows a modest, consistent lead is in Tiny. These results demonstrate that selectively adapting a minimal, task-relevant subset of parameters is sufficient to match or exceed the performance of updating all parameters, with no instances of degradation reaching the theoretical minimum p-value (p = 0.0625). The balanced results across weighted and macro F1 further indicate that this efficiency is achieved without compromising robustness to class imbalance, confirming SALFT as an effective and efficient adaptation strategy for video transformers.

The baseline comparisons yield two critical insights. First, the limited effectiveness of the random forest baseline reinforces the necessity of deep spatiotemporal representation learning for this video-based task. Second, and more critically, SALFT’s performance profile highlights its distinct advantages. It matches or surpasses full fine-tuning while updating only approximately 8% of parameters, demonstrating that a direct, task-aware selection criterion can precisely identify a sufficient and minimal adaptation subspace. This efficiency and reliability stand in contrast to generic parameter-efficient methods.

Specifically, LoRA failed to converge despite extensive hyperparameter tuning, consistently collapsing to a singleclass prediction even when increasing the rank and adjusting optimization settings, suggesting that standard hyperparameter adjustments are insufficient. We hypothesize that this failure is partly due to the substantial domain shift from Kinetics-400 to gameplay videos, as the unique spatiotemporal dynamics and complex visual elements of gaming environments are likely not fully captured by the pretrained features. In addition, the arousal prediction task depends on both explicit action semantics and long-range spatiotemporal dynamics, making optimization more sensitive to representation quality and prone to degenerate solutions. This challenge is further amplified in video models such as ViViT, where spatial and temporal features are tightly coupled, potentially requiring more expressive updates. Under these conditions, LoRA’s uniform low-rank updates may be insufficient, as they constrain adaptation to a fixed low-dimensional subspace. In contrast, SALFT adopts a task-informed strategy that selectively enables unrestricted updates on responsive layers, allowing flexible adaptation while maintaining efficiency. Furthermore, its success on hybrid architectures such as ResNet+LSTM suggests that this layer selection strategy can generalize beyond transformer-based models.

In the first stage of our two-stage fine-tuning framework, the optimal layer is reliably identified within two epochs of full fine-tuning across all nine games. For the ViViT backbone, selection consistently converges on the first encoder layer (encoder.layer.0), whereas for the ResNet+LSTM architecture, it identifies the final ResNet-50 layer (backbone.7) as most responsive. We attribute this architectural divergence in layer selection to the nature of the task, which requires capturing subtle spatiotemporal dynamics in gameplay videos for arousal prediction. In ViViT, early encoder layers operate close to the embedding stage and directly process low-level spatiotemporal patterns, making them a natural point for adapting to gameplay dynamics. In contrast, in the ResNet+LSTM architecture, spatial features are extracted by the ResNet backbone and then aggregated temporally by the LSTM, making the final ResNet layer a key interface for spatiotemporal representation learning. This suggests that, driven by the task’s demand for unified spatiotemporal modeling, SALFT inherently prioritizes layers where spatial and temporal information are most tightly integrated. The strong downstream performance of SALFT, achieving results comparable to full fine-tuning with only 8% of parameters, further suggests that this procedure can identify a minimal yet sufficient parameter subspace for adaptation.

TABLE II: Values are percentages with 95% confidence intervals in parentheses. Best result per game per metric is highlighted in bold. Cases where ViViT SALFT yields empirical improvements reaching the theoretical minimum p-value (0.0625 for N = 5, exact two-sided Wilcoxon signed-rank test) are denoted by <sup>†</sup> (vs. ViViT Full) and <sup>‡</sup> (vs. the best competing baseline).
<table><tr><td>Method</td><td>Apex W-F1</td><td>Apex M-F1</td><td>Apex Acc</td><td>Endless W-F1</td><td>Endless M-F1</td><td>Endless Acc</td><td>FPS W-F1</td><td>FPS M-F1</td><td>FPS Acc</td></tr><tr><td>Random Forest</td><td>51.96 (1.16)</td><td>43.00 (1.26)</td><td>56.68 (0.78)</td><td>53.91 (2.18)</td><td>46.86 (2.32)</td><td>59.43 (1.59)</td><td>56.03 (2.85)</td><td>46.38 (2.92)</td><td>60.82 (2.45)</td></tr><tr><td>ResNet+LSTM Full</td><td>57.27 (1.43)</td><td>53.22 (1.47)</td><td>58.07 (1.35)</td><td>58.27 (2.40)</td><td>54.89 (2.79)</td><td>58.47 (1.69)</td><td>62.38 (1.79)</td><td>57.73 (2.06)</td><td>63.15 (1.60)</td></tr><tr><td>ResNet+LSTM SALFT</td><td>56.27 (1.65)</td><td>52.23 (1.09)</td><td>56.90 (2.04)</td><td>56.39 (2.71)</td><td>52.08 (3.52)</td><td>57.75 (1.29)</td><td>61.21 (2.23)</td><td>55.98 (2.55)</td><td>61.77 (2.64)</td></tr><tr><td>ViViT Full</td><td>58.05 (1.84)</td><td>54.44 (2.23)</td><td>58.20 (2.16)</td><td>54.54 (1.26)</td><td>50.58 (0.73)</td><td>54.93 (1.79)</td><td>62.37 (2.90)</td><td>57.46 (3.10)</td><td>63.03 (2.86)</td></tr><tr><td>ViViT SALFT(Ours)</td><td>58.45 (0.97)</td><td>54.00 (1.69)</td><td>59.13 (0.99)</td><td>55.58 (1.90)†</td><td>51.49 (2.48)</td><td>56.34 (2.04)</td><td>62.75 (1.93)</td><td>57.99 (2.20)</td><td>63.58 (1.84)</td></tr><tr><td>Method</td><td>Gallery W-F1</td><td>Gallery M-F1</td><td>Gallery Acc</td><td>Gun W-F1</td><td>Gun M-F1</td><td>Gun Acc</td><td>Platform W-F1</td><td>Platform M-F1</td><td>Platform Acc</td></tr><tr><td>Random Forest</td><td>41.83 (2.34)</td><td>25.38 (1.47)</td><td>56.98 (1.83)</td><td>42.18 (3.46)</td><td>33.30 (2.35)</td><td>51.33 (2.88)</td><td>45.49 (2.67)</td><td>36.84 (1.95)</td><td>52.89 (2.62)</td></tr><tr><td>ResNet+LSTM Full</td><td>51.52 (1.77)</td><td>44.63 (2.45)</td><td>52.68 (2.48)</td><td>47.90 (2.81)</td><td>44.19 (3.27)</td><td>48.41 (2.58)</td><td>49.96 (2.70)</td><td>45.61 (3.32)</td><td>50.89 (2.69)</td></tr><tr><td>ResNet+LSTM SALFT</td><td>51.77 (1.30)</td><td>44.27 (1.26)</td><td>53.09 (2.44)</td><td>45.65 (1.99)</td><td>42.54 (1.58)</td><td>45.90 (2.29)</td><td>47.50 (3.59)</td><td>43.38 (3.50)</td><td>48.34 (3.84)</td></tr><tr><td>ViViT Full</td><td>55.29 (1.04)</td><td>47.23 (1.55)</td><td>57.86 (0.99)</td><td>54.15 (3.41)</td><td>50.84 (3.25)</td><td>54.95 (2.92)</td><td>54.37 (2.33)</td><td>50.90 (2.60)</td><td>54.98 (2.34)</td></tr><tr><td>ViViT SALFT(Ours)</td><td>56.69 (2.24)</td><td>49.58 (2.74)†</td><td>58.58 (2.36)</td><td>57.27 (1.80)†‡</td><td>54.08 (1.81)†‡</td><td>57.78 (1.54)†</td><td>56.39 (3.37)</td><td>53.08 (3.51)</td><td>57.02 (3.21)†‡</td></tr><tr><td>Method</td><td>Solid W-F1</td><td>Solid M-F1</td><td>Solid Acc</td><td>Tiny W-F1</td><td>Tiny M-F1</td><td>Tiny Acc</td><td>Topdown W-F1</td><td>Topdown M-F1</td><td>Topdown Acc</td></tr><tr><td>Random Forest</td><td>49.91 (2.87)</td><td>38.12 (1.86)</td><td>55.58 (2.33)</td><td>50.98 (0.62)</td><td>44.63 (0.78)</td><td>54.66 (0.75)</td><td>53.39 (2.00)</td><td>44.62 (1.40)</td><td>58.79 (1.76)</td></tr><tr><td>ResNet+LSTM Full</td><td>56.79 (2.10)</td><td>51.40 (2.63)</td><td>57.17 (2.07)</td><td>49.29 (3.27)</td><td>46.01 (4.13)</td><td>50.08 (3.26)</td><td>60.10 (2.02)</td><td>56.08 (1.25)</td><td>60.13 (2.50)</td></tr><tr><td>ResNet+LSTM SALFT</td><td>57.25 (2.08)</td><td>51.89 (2.81)</td><td>57.66 (2.01)</td><td>49.79 (2.19)</td><td>45.88 (2.67)</td><td>50.62 (2.48)</td><td>58.55 (2.55)</td><td>53.80 (2.87)</td><td>59.39 (1.84)</td></tr><tr><td>ViViT Full</td><td>59.43 (2.11)</td><td>54.30 (1.27)</td><td>59.80 (2.93)</td><td>56.35 (1.66)</td><td>53.39 (1.81)</td><td>56.81 (1.50)</td><td>58.91 (2.60)</td><td>54.53 (2.18)</td><td>59.04 (3.13)</td></tr><tr><td>ViViT SALFT(Ours)</td><td>60.58 (2.05)</td><td>54.53 (2.03)</td><td>61.41 (1.90)</td><td>55.60 (2.59)</td><td>52.11 (3.55)</td><td>56.47 (2.19)</td><td>59.74 (1.66)</td><td>54.95 (1.87)</td><td>60.38 (1.52)</td></tr></table>

## B. Analysis of Generalization Across Games

The robust performance of SALFT across diverse games, achieving parity with full fine-tuning in 8 out of 9 cases, demonstrates its strong generalization. This success indicates that our task-aware selection criterion effectively identifies a broadly adaptable parameter subspace within the pre-trained backbone, which remains effective despite variations in genre, visual style, or perspective.

Two instructive exceptions merit closer examination. The game Tiny, with its extremely low-poly, isometric visuals and repetitive scenery, presents a stark domain shift where full fine-tuning retains a marginal advantage. This aligns with findings that models pre-trained on real-world imagery can struggle with simplified visual styles [36], and suggests that adapting to such sparse styles might benefit from broader parameter updates. In contrast, for Endless, which features pixel-art graphics and a highly repetitive background, the full fine-tuning of the ResNet+LSTM architecture achieves the strongest performance among all methods. This indicates that the local feature inductive bias of convolutional networks may align better with the structured patterns of pixel art for this task.

Overall, these results highlight that a targeted, task-informed adaptation strategy can efficiently overcome domain shifts caused by visual diversity. They also reveal that the choice of backbone architecture interacts with specific visual styles. The primary remaining challenge for vision-only models, however, lies in capturing emotional cues that are inherently non-visual. As noted in affective computing research, the core difficulty is in establishing a robust association between visual features and abstract emotional categories, especially when the primary triggers are non-visual [37].

TABLE III: Trainable parameters and per-step computational cost (FLOPs) of different fine-tuning methods (batch size of 4). Lower values (↓) indicate better efficiency.
<table><tr><td>Method</td><td>Trainable Params ↓</td><td>FLOPs per Step (G) ↓</td></tr><tr><td>Full fine-tuning</td><td>88.65M (100%)</td><td>6402.90</td></tr><tr><td>Second-Stage Selective Fine-Tuning</td><td>7.09M (7.998%)</td><td>4449.00</td></tr></table>

## C. Efficiency Comparison

1) Parameter Efficiency: As shown in Table III, our Second-Stage Selective Fine-Tuning updates fewer than 8% of model parameters, compared to 100% in full fine-tuning. This substantial reduction significantly lowers memory and storage costs. Despite the smaller parameter set, our method achieves comparable task performance across 5-fold cross-validation (Table II), indicating that selectively updating a well-chosen layer enables more efficient and effective adaptation.

2) Computational Efficiency: Table III also shows the FLOPs per fine-tuning step, measured following the procedure described in Section IV-F2, with a batch size of 4. FLOPs are reported in giga-FLOPs $( 1 \mathrm { G } = 1 0 ^ { 9 }$ operations), providing a hardware-independent measure of computational cost. Our method requires only 4449.00 GFLOPs per step, reducing computational cost by 30.52% compared to full fine-tuning (6402.90 GFLOPs). This reduction aligns with the smaller proportion of updated parameters, confirming the efficiency of our approach.

Figure 2 plots weighted $F _ { 1 }$ against cumulative fine-tuning FLOPs for the first fold, using gallery as a representative example (see the GitHub repo for the other eight games). The plot illustrates the trade-off between effectiveness and computational cost. Despite comprising two fine-tuning stages, our method achieves its final performance at a lower cumulative computational cost. In the exploratory first stage, training is limited to only 2 epochs to determine the layer selection, keeping this initial cost manageable. In the selective second stage, only the chosen subset of layers is fine-tuned, substantially reducing the FLOPs per epoch relative to full fine-tuning. As observed in Figure 2, SALFT typically reaches its performance plateau with less overall computation. This efficiency gain, combined with the fixed layer selection that can be reused across folds (in our 5-fold CV) or for similar tasks, accelerates model development and iteration while maintaining deployment compatibility since the final model architecture is unchanged.

![](images/90cfd73998399f3011cefce160ad00ab7d6c3b3228e019ef976cf07662f72c7b.jpg)  
Fig. 2: Weighted $F _ { 1 }$ against cumulative fine-tuning FLOPs for gallery as a representative case (full fine-tuning vs. SALFT). Analysis is based on the first fold of 5-fold crossvalidation. Complete results for all nine games are provided in the GitHub repo.

## D. Interpretability Analysis

To understand how the fine-tuned model with SALFT (first fold of 5-fold cross-validation) responds to gameplay videos, we perform an interpretability analysis using the method described in Section III-C. We use samples from Solid as a representative case due to its clear and bright visuals, which facilitate effective visualization. We selected video segments correctly classified by the model according to ground-truth labels. Figure 3 and Figure 4 show simplified examples of heatmaps overlaid on frames from Solid, illustrating attention patterns for arousal-increasing and arousal-decreasing segments. Warmer colors denote regions of higher relevance, indicating where the model focuses when predicting player arousal changes. Full visualizations, including segments with unchanged arousal and examples from other games, are available in the GitHub repo.

We observed that for arousal-increasing segments (Fig. 3), the model placed greater attention on semantically meaningful regions in the latter video frames, such as enemy characters, sudden movements, bright flashes, or other salient visual events. This suggests that the model associates arousal increases with contextually relevant and potentially exciting stimuli. The model appears to exhibit a degree of semantic understanding, prioritizing emotionally salient elements over superficial cues such as static heads-up display components or background textures.

In contrast, during arousal-decreasing or arousalunchanging segments (Fig. 4), when the gameplay lacks dynamic or emotionally salient events, the model’s attention was concentrated on visually calm or repetitive regions, such as static background elements, idle states, non-hostile navigation, or low-activity scenes. This indicates that in the absence of strong emotional cues, the model defaults to focusing on static elements of the scene, reflecting its sensitivity to the overall emotional context across different affective states.

Attention heatmaps from the SALFT-fine-tuned ViViT model exhibited game-specific patterns, demonstrating the model’s ability to dynamically adjust focus according to each game’s visual and gameplay context. In racing games, attention is targeted at the road ahead and potential collision zones. In shooter games, focus is centered on the crosshair, weapon barrel, enemies, and shooting-related effects. In platformers, attention is concentrated on the player character, enemies, and interactive elements essential for gameplay. Such adaptability reflects the model’s sensitivity to salient visual cues and suggests robustness in handling diverse gameplay scenarios. A comparison with the full fine-tuned model’s attention patterns is provided in the GitHub repo.

We further conducted a perturbation-based evaluation to quantitatively assess the attribution results, as described in Section IV-G. Using the SALFT-fine-tuned model from the first fold, we computed AUC scores on the corresponding test set of the solid game as a representative example (Table IV). The proposed interpretability method performs notably well, achieving favorable AUC scores under both positive and negative perturbations, indicating that the highlighted regions are critical to the model’s decisions. Across the nine games (full results provided in the GitHub repo), our method outperforms all other baselines. These quantitative findings, based on the first-fold evaluation, corroborate our visual analysis, confirming the high faithfulness of the generated explanations.

TABLE IV: Positive and negative perturbation AUC results (shown as percentages) for the predicted and target classes on the solid test set. Lower (↓) is better for positive perturbation and higher (↑) for negative. Best results are in bold.
<table><tr><td>Type</td><td>Class</td><td>Rollout</td><td>Grad-CAM</td><td>Grad-SAM</td><td>Our Method</td></tr><tr><td rowspan="2">Positive ↓</td><td>Predicted</td><td>46.96</td><td>47.47</td><td>49.82</td><td>24.01</td></tr><tr><td>Target</td><td>45.81</td><td>44.54</td><td>48.69</td><td>23.13</td></tr><tr><td rowspan="2">Negative ↑</td><td>Predicted</td><td>86.86</td><td>87.91</td><td>84.67</td><td>93.91</td></tr><tr><td>Target</td><td>80.75</td><td>84.64</td><td>78.65</td><td>87.72</td></tr></table>

## E. Limitations

First, the generalizability of our layer selection strategy requires further validation across more diverse games and domains, such as those represented in recently released multimodal affective game corpora [38], [39]. Second, as a visiononly method, its performance is inherently limited in visually sparse environments, motivating the future integration of multimodal data—an opportunity enabled by the same emerging datasets. Third, to isolate the effects of our layer selection strategy, we fixed the threshold for defining arousal changes; consequently, a full sensitivity analysis of this threshold’s impact on performance is left for future work. Fourth, our focus on a single fine-tuned layer may not capture more complex feature interactions, and future work could explore multi-layer or dynamic tuning strategies. Finally, while SALFT ViViT consistently matches or numerically exceeds both full fine-tuning ViViT and the best-performing baseline across all games, only a subset of these improvements achieves the theoretical minimum p-value, an inherent constraint of the 5- fold cross-validation setup.

up | solid | Idx: 11  
![](images/0db670cee4a0b4f92f579583521d3d123c0fdfc8dcb5c6ac2266c2a40bc0c92f.jpg)  
Fig. 3: Interpretation example for arousal-increasing class in solid (SALFT-fine-tuned model, first fold).

![](images/71dd3a6ccbee573f158215caca4d6e3bbfcf9c4b9df680955fec9ea29070b683.jpg)  
Fig. 4: Interpretation example for arousal-decreasing class in solid (SALFT-fine-tuned model, first fold).

## VI. CONCLUSIONS

This paper introduced Selective Adaptive Layer Fine-Tuning (SALFT), a parameter-efficient framework for recognizing player arousal from gameplay videos. By fine-tuning only the most responsive transformer layers identified via an L2-norm criterion, SALFT reduces trainable parameters by approximately 92% (from 88.65M to 7.09M) and training FLOPs by 30.5%. As validated by exact Wilcoxon signed-rank tests, SALFT ViViT maintains performance statistically comparable to both full fine-tuning ViViT and the best-performing baseline across all nine games, succeeding where the generic PEFT method, LoRA, failed to converge. Interpretability analyses reveal that SALFT dynamically attends to semantically meaningful game elements across different genres. These results position SALFT as a practical and robust solution for resource-constrained affective computing in gaming, offering an effective balance between performance, efficiency, and interpretability.

[1] G. N. Yannakakis and D. Melhart, “Affective game computing: A survey,” Proceedings of the IEEE, vol. 111, no. 10, pp. 1423–1444, 2023.

[2] K. Makantasis, A. Liapis, and G. N. Yannakakis, “From pixels to affect: A study on games and player experience,” in 2019 8th International Conference on Affective Computing and Intelligent Interaction (ACII). IEEE, 2019, pp. 1–7.

[3] ——, “The pixels and sounds of emotion: General-purpose representations of arousal in games,” IEEE Transactions on Affective Computing, vol. 14, no. 1, pp. 680–693, 2023.

[4] Y. Xia, X. Li, S. Chen, and R. Thawonmas, “Now you see me: Recognizing the player’s arousal changes in the game through game footage videos and game context features,” in Computers and Games, M. Hartisch, C.-H. Hsueh, and J. Schaeffer, Eds. Cham: Springer Nature Switzerland, 2025, pp. 159–169.

[5] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Luciˇ c, and C. Schmid,´ “Vivit: A video vision transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 6836–6846.

[6] D. Melhart, A. Liapis, and G. N. Yannakakis, “The arousal video game annotation (again) dataset,” IEEE Transactions on Affective Computing, vol. 13, no. 4, pp. 2171–2184, 2022.

[7] D. Melhart, M. Barthet, and G. N. Yannakakis, “Can large language models capture video game engagement?” arXiv preprint arXiv:2502.04379, 2025.

[8] L. Xuhong, Y. Grandvalet, and F. Davoine, “Explicit inductive bias for transfer learning with convolutional networks,” in International conference on machine learning. PMLR, 2018, pp. 2825–2834.

[9] H. Gouk, T. M. Hospedales, and M. Pontil, “Distance-based regularisation of deep networks for fine-tuning,” 2021.

[10] N. Zhou, J. Chen, and D. Huang, “Dr-tune: Improving fine-tuning of pretrained visual models by distribution regularization with semantic calibration,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, 2023, pp. 1547–1556.

[11] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 09–15 Jun 2019, pp. 2790–2799. [Online]. Available: https://proceedings.mlr.press/v97/houlsby19a.html

[12] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts for generation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli, Eds. Online: Association for Computational Linguistics, Aug. 2021, pp. 4582–4597. [Online]. Available: https://aclanthology.org/2021.acl-long.353/

[13] X. Liu, K. Ji, Y. Fu, W. Tam, Z. Du, Z. Yang, and J. Tang, “P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 61–68. [Online]. Available: https://aclanthology.org/2022.acl-short.8/

[14] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9

[15] Z. Liu, S. Kundu, A. Li, J. Wan, L. Jiang, and P. Beerel, “AFLoRA: Adaptive freezing of low rank adaptation in parameter efficient finetuning of large models,” in Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 161–167. [Online]. Available: https://aclanthology.org/2024.acl-short.16/

[16] N. Ding, X. Lv, Q. Wang, Y. Chen, B. Zhou, Z. Liu, and M. Sun, “Sparse low-rank adaptation of pre-trained language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 4133–4145.

[17] D. Guo, A. Rush, and Y. Kim, “Parameter-efficient transfer learning with diff pruning,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli, Eds. Online:

Association for Computational Linguistics, Aug. 2021, pp. 4884–4896. [Online]. Available: https://aclanthology.org/2021.acl-long.378/

[18] J. Howard and S. Ruder, “Universal language model fine-tuning for text classification,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), I. Gurevych and Y. Miyao, Eds. Melbourne, Australia: Association for Computational Linguistics, Jul. 2018, pp. 328–339. [Online]. Available: https://aclanthology.org/P18-1031/

[19] Y. Lee, A. S. Chen, F. Tajwar, A. Kumar, H. Yao, P. Liang, and C. Finn, “Surgical fine-tuning improves adaptation to distribution shifts,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=APuPRxjHvZ

[20] G. Kaplun, A. Gurevich, T. Swisa, M. David, S. Shalev-Shwartz, and E. Malach, “Less is more: Selective layer finetuning with subtuning,” arXiv preprint arXiv:2302.06354, 2023.

[21] Z. Zhao, Z. Liu, Y. Cao, S. Gong, and I. Patras, “Aim-fair: Advancing algorithmic fairness via selectively fine-tuning biased models with contextual synthetic data,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 28 748–28 758.

[22] K. Yang, W.-L. Wei, and J.-C. Lin, “Fps: Feedforward-based parameter selection for efficient fine-tuning,” arXiv preprint arXiv:2510.27359, 2025.

[23] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning deep features for discriminative localization,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition, 2016, pp. 2921– 2929.

[24] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 618–626.

[25] A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, “Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks,” in 2018 IEEE winter conference on applications of computer vision (WACV). IEEE, 2018, pp. 839–847.

[26] P.-T. Jiang, C.-B. Zhang, Q. Hou, M.-M. Cheng, and Y. Wei, “Layercam: Exploring hierarchical class activation maps for localization,” IEEE Transactions on Image Processing, vol. 30, pp. 5875–5888, 2021.

[27] S. Belharbi, S. Murtaza, M. Pedersoli, I. Ben Ayed, L. McCaffrey, and E. Granger, “CoLo-CAM: Class activation mapping for object colocalization in weakly-labeled unconstrained videos,” Pattern Recognition, vol. 162, p. 111358, 2025.

[28] S. Abnar and W. Zuidema, “Quantifying attention flow in transformers,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 4190–4197. [Online]. Available: https: //aclanthology.org/2020.acl-main.385/

[29] H. Chefer, S. Gur, and L. Wolf, “Transformer interpretability beyond attention visualization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 782–791.

[30] O. Barkan, E. Hauon, A. Caciularu, O. Katz, I. Malkiel, O. Armstrong, and N. Koenigstein, “Grad-sam: Explaining transformers via gradient self-attention maps,” in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, ser. CIKM ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 2882–2887. [Online]. Available: https://doi.org/10.1145/3459637. 3482126

[31] M. Kowal, A. Dave, R. Ambrus, A. Gaidon, K. G. Derpanis, and P. Tokmakov, “Understanding video transformers via universal concept discovery,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 946–10 956.

[32] J. D. Evans, Straightforward statistics for the behavioral sciences. Thomson Brooks/Cole Publishing Co, 1996.

[33] “A guide to appropriate use of correlation coefficient in medical research,” Malawi medical journal, vol. 24, no. 3, pp. 69–71, 2012.

[34] P. Schober, C. Boer, and L. A. Schwarte, “Correlation coefficients: appropriate use and interpretation,” Anesthesia & analgesia, vol. 126, no. 5, pp. 1763–1768, 2018.

[35] K. Pinitas, D. Renaudie, M. Thomsen, M. Barthet, K. Makantasis, A. Liapis, and G. N. Yannakakis, “Predicting player engagement in tom clancy’s the division 2: A multimodal approach via pixels and gamepad actions,” in Proceedings of the 25th International Conference on Multimodal Interaction, ser. ICMI ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 488–497. [Online]. Available: https://doi.org/10.1145/3577190.3614203

[36] M. Jung, H. Yang, and K. Min, “Improving deep object detection algorithms for game scenes,” Electronics, vol. 10, no. 20, p. 2527, 2021.

[37] D. Wu, D. Yang, Y. Zhou, and C. Ma, “Bridging visual affective gap: Borrowing textual knowledge by learning from noisy image-text pairs,” in Proceedings of the 32nd ACM International Conference on Multimedia, ser. MM ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 602–611. [Online]. Available: https://doi.org/10.1145/3664647.3680875

[38] A. Rashed, S. Shirmohammadi, and M. Hefeeda, “Descriptor: Multimodal dataset for player engagement analysis in video games (multipeng),” IEEE Data Descriptions, vol. 2, pp. 17–25, 2025.

[39] M. Barthet, M. Kaselimi, K. Pinitas, K. Makantasis, A. Liapis, and G. N. Yannakakis, “Gamevibe: a multimodal affective game corpus,” Scientific Data, vol. 11, no. 1, p. 1306, 2024.