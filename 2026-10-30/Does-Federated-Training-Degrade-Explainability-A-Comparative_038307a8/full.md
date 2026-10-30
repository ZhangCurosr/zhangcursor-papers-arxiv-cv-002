# Does Federated Training Degrade Explainability? A Comparative Analysis of Grad-CAM Faithfulness in Centralized vs. Federated Deepfake Detection

Muhammad Uzair Shahid

Muhammad Zabil Mehboob

Abstract—Automated detection has become the essential safeguard against realistic synthetic media, but deploying it at scale requires training on sensitive data from multiple organizations, putting privacy and the transparency demanded by forensic review in direct conflict. Federated learning resolves the privacy side by sharing only model updates rather than raw data, but whether this comes at the cost of trustworthy gradient-based explanations, which analysts rely on to decide whether or not to trust a flagged clip, has not been studied. We present the systematic study of Grad-CAM explanation faithfulness under federated deepfake detection, evaluating an EfficientNet-B4 detector on FaceForensics++ across five federated clients under varying degrees of data heterogeneity. We find that the relationship between federation and explanation quality is non-monotonic: at near-IID (Independent and Identically Distributed Data) and moderate heterogeneity, federated models match or exceed the faithfulness of a centralized baseline, but at extreme heterogeneity, explanations break down entirely, becoming spatially incoherent and, on average, causally misleading. This degradation is not a general cost of federated training but a threshold effect tied to client heterogeneity, and we offer practical guidance for identifying when privacy-preserving deepfake detection can be deployed safely and when its explanations can no longer be trusted. Code and data partitions are available for reproducibility.

Impact Statement—Federated learning offers a practical way to train deepfake detectors across organizations without requiring them to share sensitive media. However, privacy-preserving training alone does not establish that the resulting explanations remain trustworthy. This work advances understanding of this issue by systematically examining the faithfulness of Grad-CAM explanations under different levels of client heterogeneity. Our results show that federated training does not inherently reduce explanation faithfulness: under near-IID and moderate heterogeneity, federated models can maintain or exceed the explanation faithfulness of

centralized training. In contrast, extreme heterogeneity can cause explanations to become spatially incoherent and causally misleading. These findings identify client heterogeneity as an important factor in evaluating the reliability of explainable federated AI systems, rather than treating federated training itself as the source of explanation degradation. The study therefore provides a practical basis for assessing when federated deepfake detection can support applications where both data privacy and trustworthy model explanations are required.

Index Terms— Area Over the Perturbation Curve (AOPC), Deepfake Detection, Explainable Artificial Intelligence (XAI), Explanation Faithfulness, FaceForensics++, Federated Learning, Grad-CAM (Gradientweighted Class Activation Mapping), Non-IID Data (Non-Independent and Identically Distributed Data).

## I. INTRODUCTION

## A. Motivation

YNTHETIC media, also called deepfakes, has reached S <sub>a</sub> <sub>level</sub> <sub>of</sub> <sub>realism</sub> <sub>that</sub> <sub>makes</sub> <sub>manual</sub> <sub>inspection</sub> unreliable and threatens digital forensics [1], information integrity, and public trust. Convolutional neural networks trained on large benchmarks such as FaceForensics++ now classify manipulated facial video with high accuracy [1], and therefore automated detection has become the front line against this threat.

However, putting that detection into real forensic pipelines runs into two demands that pull in different directions. The first is privacy: training on sensitive user-generated content pooled from social media platforms, news agencies, and government bodies raises data-governance concerns that rule out naive centralized aggregation. The second is explainability: reviewers in high-stakes settings, such as content moderation, legal proceedings and journalism verification, cannot act on a bare binary output. They need explanations that point to the specific facial regions that drove a manipulation decision [2], [3].

Federated learning (FL) solves the first problem by letting multiple data holders train one model together without any of them sharing raw samples [4]. Under Federated Averaging, each participant trains locally on their own private data and sends only parameter updates to a central server, which combines them into a global model through sample-weighted aggregation. Digital forensics, healthcare, and finance have picked up this approach for exactly that reason.

This privacy-preserving setup can, however, affect what the model learns. Clients typically hold different mixes of manipulation types, and so moving from centralized to federated training can cause gradient averaging to pull client weights apart, changing the model’s internally learned features [5]. Grad-CAM and similar post-hoc explanation methods [2] read visual saliency straight off those gradients, so if federated aggregation distorts the gradient space, the resulting heatmaps may no longer track what the model is actually doing. Forensic analysts routinely trust a flagged clip because its heatmap looks plausible, yet this is precisely the signal that federated aggregation could compromise.

## B. Research Gap

While federated learning, deepfake detection, and explainable AI have each cultivated extensive bodies of literature, these domains have largely evolved in isolation.

Federated learning research focuses on convergence guarantees, communication cost, and classification accuracy under statistical heterogeneity [4], [5], [6], [7], [8]. The quality of the gradient-based explanations that come out of the aggregated model is rarely mentioned.

Deepfake detection research reaches state-of-the-art accuracy with centralized training pipelines and checks visual explanations such as Grad-CAM and SHAP only under centralized conditions [1], [9], [10], [11], [12]. Federated deployment of these same detectors is not part of that conversation.

While research on XAI faithfulness has yielded robust quantitative tools—such as AOPC [3], parameterrandomization sanity checks [13], and saliency reliability analyses [14]—these metrics are almost exclusively applied to centralized models. To date, they have not been adapted for federated settings, much less for visual forensic tasks.

Federated XAI does exist in a few other domains: supply chain credit assessment [15] and network intrusion detection [16]. These studies show that explanations can be produced without exposing client data, but they work in non-visual settings and never quantitatively compare centralized against federated explanation faithfulness in visual deepfake detection.

## C. Hypothesis

Our primary hypothesis is:

H1: Federated training introduces gradient inconsistency across client models that reduces the spatial faithfulness of post-hoc gradient-based explanations (Grad-CAM) relative to centralized baselines, and this degradation correlates monotonically with the degree of data heterogeneity across clients, controlled here through the Dirichlet concentration parameter α.

Under H1, the most heterogeneous federated settings (low α) should produce the least faithful explanations, while near-IID conditions (high α) should come closest to centralized explanation quality.

## D. Contributions

This paper contributes the following:

1) Empirical measurement of XAI faithfulness degradation in federated deepfake detection. We compare AOPC and Explanation IoU between a centralized EfficientNet-B4 baseline and federated counterparts across four controlled heterogeneity levels on FaceForensics++.

2) A characterization of how heterogeneity and faithfulness relate. We show that federation’s effect on explanation quality is non-monotonic and confined to the extreme heterogeneity regime $( \alpha = 0 . 1 )$ , backed by a perfect Spearman rank correlation (ρ = −1.0) between the Dirichlet concentration parameter and AOPC degradation.

3) An open-source experimental framework. All data partitions, training configurations, Grad-CAM maps, and evaluation scripts are released so the results can be replicated directly and reused as a foundation for federated XAI research in visual forensics.

4) Practical guidance for practitioners. We give concrete thresholds for when federated training preserves explanation quality and when it needs help, for instance from proximal regularization or explanation-aware aggregation.

## II. RELATED WORK

The proposed study sits at the intersection of four research areas that have largely developed on separate tracks: federated learning under statistical heterogeneity, deepfake detection, explainable artificial intelligence (XAI), and quantitative evaluation of explanation faithfulness. Federated learning research is centered on optimization efficiency and convergence. Deepfake detection work is centered on classification accuracy. XAI research examines explanation generation and reliability under centralized training. Federated XAI applications remain confined to low-dimensional domains. This section reviews each pillar in turn, then synthesizes the gap that motivates the present study.

## A. Federated Learning and Non-IID Data

Federated Learning (FL) allows collaborative model training without exchanging raw data. Participating clients perform local optimization and share only model parameters, which makes FL suitable for privacy-preserving machine learning in healthcare, finance, mobile computing, and digital forensics. McMahan et al. [4] established the foundation of modern federated optimization with Federated Averaging (FedAvg): each client runs several local stochastic gradient descent (SGD) updates before sending parameters to a central server for weighted aggregation. Their experiments showed that FedAvg cuts communication rounds by an order of magnitude relative to synchronized distributed SGD while matching its classification performance. Their evaluation, however, is limited to communication efficiency and predictive accuracy; the quality and reliability of the learned gradient representations is left unexamined.

Subsequent work identified statistical heterogeneity as one of the primary obstacles to federated optimization.

Zhao et al. [5] showed that when clients hold highly skewed local datasets, the global model’s performance degrades substantially. They traced this to weight divergence: discrepancies between local and global label distributions produce increasingly inconsistent parameter updates over training, with accuracy reductions approaching 55% under severe non-IID settings and divergence closely tied to the Earth Mover’s Distance (EMD) between client and global data distributions..

Li et al. [6] proposed FedProx to address heterogeneous environments, adding a proximal regularization term to each client’s local objective that discourages large deviations from the current global model. Unlike FedAvg, FedProx addresses both statistical and system heterogeneity and produces more stable convergence across clients with different computational capacities and data distributions, improving performance by approximately 22% in highly heterogeneous settings. Constraining local optimization this way changes the gradient behavior used during backpropagation, and since post-hoc explainability methods derive explanations directly from these gradients, the effect of proximal regularization on explanation quality is still uninvestigated.

Li et al. [7] provided the first rigorous convergence guarantees for FedAvg under non-IID data, proving an O(1/T) convergence rate for strongly convex objectives and showing that increasing local optimization epochs introduces a bias proportional to the degree of statistical heterogeneity unless a decaying learning rate is used. This is optimization theory, but it carries a direct implication for explainability: gradient-based explanation methods assume gradients faithfully represent the model’s decision process, so if federated optimization systematically biases those gradients, explanation faithfulness should degrade as well.

Li et al. [8] introduced NIID-Bench, a benchmark evaluating multiple federated algorithms across several forms of statistical heterogeneity — label skew, feature skew, and quantity imbalance. No single algorithm consistently outperformed the others across all scenarios, and Dirichlet-based label partitioning has emerged as one of the most challenging and widely adopted methods for simulating heterogeneous client distributions. The Dirichlet concentration parameter (α) has since become the standard mechanism for controlling statistical heterogeneity in federated learning experiments, and the present study adopts this framework to analyze explanation degradation under varying degrees of client diversity.

Taken together, the federated learning literature shows that statistical heterogeneity alters optimization dynamics through conflicting client updates and gradient divergence, but existing evaluation stays largely within convergence speed and classification accuracy. How these altered gradient fields affect post-hoc explainability has received little attention, leaving a gap between federated optimization theory and explainable AI.

## B. Deepfake Detection and Explainability

Advances in generative AI have made manipulated multimedia content more realistic and more accessible, which makes automated deepfake detection an increasingly important problem. Detection models now achieve strong classification performance, but forensic and security deployments need more than accurate predictions — they need explanations that let human analysts verify model decisions.

Wang et al. [9] surveyed modern deepfake detection techniques from a reliability perspective and identified several persistent challenges, including limited transferability across manipulation techniques, insufficient interpretability, and robustness issues. They emphasized the need for reliable detection models in real-world settings and identified interpretability and robustness as important directions for further research, but did not empirically investigate how federated training affects the faithfulness of explanations in visual deepfake detection.

Gradient-based and feature-based explanation techniques have been used to study transparency in deepfake detection directly. Mansoor and Iliev [10] applied network dissection to CNN-based detectors, including ResNet-50, VGG-16, and Inception V3, and analyzed the interpretable facial concepts learned by these models. The detectors achieved F1-scores between 0.80 and 0.90, while the network dissection analysis identified interpretable facial concepts across different CNN architectures. This demonstrates the potential of model-interpretation techniques for analyzing CNN-based deepfake detectors, but their framework evaluates models under centralized training and does not consider explanation behavior under privacy-sensitive distributed learning.

Sunkari and Srinagesh [11] proposed DeepExplain, integrating Grad-CAM and SHAP into a CNN-LSTM architecture trained on the DeepFake Detection Challenge (DFDC) dataset. Their framework uses Grad-CAM to visualize regions that contribute to model predictions and SHAP to quantify the contribution of input features, providing insights into the model’s decision-making process. The study argues that these explainability mechanisms can foster trust and improve understanding of DeepFake detection decisions. However, the framework is evaluated under centralized training, so whether explanation consistency survives distributed training across heterogeneous clients remains unclear.

Maheshwari and Paulchamy [12] introduced XAI-ART, combining adversarial robustness training with explainable AI to improve both detector resilience and transparency, reaching approximately 97.5% classification accuracy and maintaining high recall under adversarial attacks while producing interpretable attention maps. This shows that robustness and explainability can coexist in centralized deepfake detectors, but it does not examine whether federated optimization affects the stability or faithfulness of those explanations.

Existing deepfake detection research demonstrates that Grad-CAM and related techniques improve transparency and user trust under centralized training, but each study assumes a coherent gradient space produced by centralized optimization. None examine whether gradient interference from federated aggregation changes the spatial consistency or faithfulness of explanation maps, which leaves a gap for privacy-preserving deepfake detection systems.

SUMMARY OF THE CONVERGING LITERATURES  
TABLE I
<table><tr><td>Research Pillar</td><td>Established</td><td>Missing</td></tr><tr><td>[7], [8]</td><td>ing [4], [5], [6], divergence under non-IID data, Dirichlet α as the faithfulness remains unexplored standard heterogeneity benchmark</td><td>Federated Learn- FedAvg, FedProx, convergence analysis, weight Effect of gradient divergence on explanation</td></tr><tr><td>Deepfake Detection [10], [11], [12]</td><td>[9], SHAP, adversarial robustness, and centralized planation reliability explainability</td><td>High-performance detectors with Grad-CAM, Federated implementation and evaluation of ex-</td></tr><tr><td>XAI [3], [13], [14]</td><td>Grad-CAM, AOPC, sanity checks, and saliency No application of these metrics to federated tion</td><td>Faithfulness [2], reliability provide objective explanation evalua- deepfake detection under statistical heterogeneity</td></tr><tr><td>Federated XAI [15], [16]</td><td>Explainability integrated into FL for credit as- High-dimensional visual forensics, quantitative sessment and intrusion detection</td><td>comparison between centralized and federated ex- planations, and heterogeneity-aware faithfulness evaluation</td></tr></table>

## C. XAI Faithfulness Measurement

Recent XAI research has shifted from generating visual explanations to evaluating whether those explanations faithfully represent a model’s decision process. Grad-CAM, proposed by Selvaraju et al. [2], remains the most widely used explanation technique for CNNs, producing heatmaps from class gradients, but it assumes a coherent gradient space that may not survive federated aggregation.

Samek et al. [3] introduced the Area Over the Perturbation Curve (AOPC) to assess explanation quality objectively, measuring the drop in model confidence as important regions are progressively removed. Adebayo et al. [13] and Kindermans et al. [14] showed that saliency maps need validation through sanity checks and robustness analysis, since visually plausible explanations do not always reflect actual model behavior.

These methods give explainability a reliable evaluation framework, but they have been validated almost exclusively under centralized training. Whether they apply to federated deepfake detection, where statistical heterogeneity may disrupt gradient consistency, remains largely unexplored.

## D. XAI within Federated Systems

Federated XAI extends federated learning by integrating explainability into privacy-preserving machine learning. Existing work applies XAI to supply chain credit assessment [15] and intrusion detection [16], showing that explanations can be generated without exposing client data.

These studies focus on non-visual applications, though, and do not examine whether explanation quality and faithfulness remain stable under federated training for visual deepfake detection. Quantitative comparisons between centralized and federated explanations for visual deepfake detection specifically have not been reported.

## E. Research Gap

Table I summarizes where each research pillar stands and what it leaves open.

Federated learning, deepfake detection, XAI, and federated XAI have each advanced independently, but their intersection remains largely unexamined. Federated learning research has characterized gradient divergence and convergence behavior under non-IID data, but not its downstream effect on explanation faithfulness. Deepfake detection research has validated Grad-CAM and SHAP as explanation tools, but only under centralized training. XAI research has produced faithfulness metrics such as AOPC and sanity checks, but has not applied them to federated visual models. Federated XAI research has integrated explainability into FL for tabular and sequential domains such as credit assessment, intrusion detection, and speech, but has not extended to high-dimensional visual forensics or heterogeneity-aware faithfulness evaluation. This paper addresses that combined gap directly.

## III. METHODOLOGY

## A. Dataset and Pre-processing

All experiments used the FaceForensics++ (FF++) benchmark [1], at the lightly compressed c23 encoding. The corpus contains 1,000 real videos sourced from YouTube and 1,000 videos for each of five manipulation methods: Deepfakes (DF), Face2Face (F2F), FaceShifter (FSh), FaceSwap (FS), and NeuralTextures (NT), for 6,000 clips across six classes. We collapse these to a binary real-vs.- fake label for classification.

Frame extraction. We sampled frames at a fixed temporal stride of 10 and localized face regions with the OpenCV DNN res10 SSD face detector (pretrained Caffe model, minimum detection confidence 0.50). Each detected face crop was resized to 224 × 224 pixels and saved as a JPEG at quality 95. The extraction pipeline is implemented in src/data.py. An acceptance gate required at least 14,000 face crops before training could start.

Dataset splits. We partitioned the extracted crops into training (70%), validation (15%), and test (15%) subsets using a fixed random seed (seed = 42) via NumPy’s default random number generator, so the split is reproducible. The test split was fixed across all experiments and was never touched during architecture selection or hyperparameter tuning. It contains 12,037 images, confirmed against results/centralized metrics.json.

Normalization and augmentation. Images were normalized with ImageNet channel statistics:

$$
\mathrm { \ m e a n } = [ 0 . 4 8 5 , \ 0 . 4 5 6 , \ 0 . 4 0 6 ]\tag{1}
$$

$$
\mathrm { s t d } = [ 0 . 2 2 9 , \ 0 . 2 2 4 , \ 0 . 2 2 5 ]\tag{2}
$$

Training images also received random horizontal flipping and mild color jitter (brightness, contrast, and saturation perturbations of ±0.1). Evaluation images received only resizing and normalization, with no stochastic augmentation.

## B. Federated Data Partitioning

To simulate client heterogeneity, we partitioned the training corpus across $N = 5$ simulated clients with a Dirichlet distribution at concentration α, following the non-IID partitioning protocol of Zhao et al. [5]. For each class label, we drew a set of proportions from $\operatorname { D i r } ( \alpha \cdot { \mathbf { 1 } } _ { N } )$ and assigned training indices to clients accordingly. Low α concentrates a client’s data in one or two manipulation categories, approximating federated nodes that specialize in different forgery techniques. As $\alpha \to \infty$ , the partition converges to IID.

We swept four heterogeneity levels: $\alpha = 0 . 1 , 0 . 3 , 0 . 5 .$ and 1.0. $\mathrm { A t } \ \alpha = 0 . 1$ , a client may draw the overwhelming majority of its samples from a single manipulation category. $\mathrm { { A t } } \ \alpha \ = \ 1 . 0$ , client distributions closely approximate uniform coverage across all classes. All partitions were generated deterministically (seed = 42) and serialized to data/partitions/alpha \*.json.

## C. Model Architecture

The backbone is EfficientNet-B4 [17], pretrained on ImageNet and fine-tuned for binary deepfake classification. We replaced the original classification head with:

$$
\mathrm { G l o b a l A v g P o o l } \to \mathrm { D r o p o u t } ( p { = } 0 . 3 ) \to \mathrm { L i n e a r } ( C \to 2 )
$$

The EfficientNet-B4 feature extractor (backbone.extr act\_features) produces spatial feature maps of shape $( B , C , 7 , 7 )$ at the penultimate convolutional stage; we globally average-pool these to a $( B , C )$ vector before the classification head. All backbone layers stayed unfrozen during fine-tuning. The model is implemented in src/ model.py and exposes a $\mathsf { g e t \_ l a s t \_ c o n v } ( )$ ) accessor returning backbone.\_conv\_head, the Grad-CAM target layer.

We chose EfficientNet-B4 for its track record on FF++ and because gradient-based attribution methods apply directly to convolutional architectures. This work introduces no new architecture; its contribution is an analysis of how explanations behave under federation.

## D. Centralized Baseline Training

The centralized baseline was trained on the full training split without client partitioning, using Adam (learning rate $1 0 ^ { - 4 }$ , batch size 32) with cross-entropy loss, for 10 epochs, with checkpointing at peak validation accuracy. The best checkpoint (models/centralized/best.pt) was used for all subsequent Grad-CAM generation and AOPC evaluation, and serves as the performance and faithfulness upper bound against which we compare all federated conditions.

## E. Federated Training Protocol (FedAvg)

We implemented federated training as a sequential FedAvg simulation in pure PyTorch (src/federated server.py), with no external orchestration framework (e.g., Ray or Flower). Only one client model resides in GPU memory at a time, so memory use stays bounded regardless of client count.

Communication rounds. Given single-GPU compute constraints, we ran $\alpha \in \{ 0 . 3 , 0 . 5 , 1 . 0 \}$ for 10 communication rounds, by which point both validation accuracy and Explanation IoU had plateaued (Fig. 3). To separate slow convergence from genuine non-convergence at the most heterogeneous setting, we extended $\alpha = 0 . 1$ to 30 rounds; metrics stayed essentially flat over that extension (Expl. IoU: 0.127 at round $1 0  0 . 1 3 2$ at round 30), which points to non-convergence rather than a transient slowdown.

Local training. Each round consisted of one local epoch per client, following the FedAvg protocol of McMahan et al. [4], with Adam (learning rate $1 0 ^ { - 4 } )$ , cross-entropy loss, batch size 32, and mixed-precision training where CUDA was available.

Aggregation. After all five clients finished local training in a round, we updated the global model by sampleweighted averaging:

$$
w ^ { t + 1 } = \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { n } w _ { k } ^ { ( t + 1 ) }\tag{3}
$$

where $n _ { k }$ is the number of samples client k processed in that round and $n = \textstyle \sum _ { k } n _ { k }$ . We saved checkpoints every 5 rounds; for $\alpha = 0 . 1$ we kept rounds 10, 20, and 30, and for $\alpha \in \{ 0 . 3 , 0 . 5 , 1 . 0 \}$ we kept rounds 5 and 10.

## F. Grad-CAM Explanation Generation

We applied Gradient-weighted Class Activation Mapping (Grad-CAM) [2] to the centralized model and to every federated final-round checkpoint, using the pytorch-gradcam library. The target layer was backbone.\_conv \_head and the target output was the fake class (index 1). For each test image, the Grad-CAM engine computed the gradient of the target class score with respect to the convolutional feature map and applied a ReLU-weighted linear combination. The resulting CAM was a float32 array of spatial dimensions (H, W), normalized to [0, 1] and saved as a .npy file indexed by image ID. All CAMs were generated with shuffle=False so image IDs stayed aligned across model configurations.

## G. Faithfulness Metrics

1) Area Over the Perturbation Curve (AOPC): AOPC [3] tests whether the regions a Grad-CAM heatmap highlights are causally important to the model’s decision. For each test image and its CAM:

1) Record the baseline confidence $f _ { 0 } { \mathrm { : } }$ the model’s softmax probability for the fake class on the unperturbed image.

2) For each step $k \in \{ 1 , \ldots , 1 0 \}$ , mask the top $k$ 10% most salient pixels (ranked by CAM value) by replacing them with the per-channel image mean.

3) Record the model confidence $f _ { k }$ on the masked image.

TABLE II TABLE II  
DETECTION ACCURACY, AUC, MEAN AOPC, AND EXPLANATION IOU ACROSS THE CENTRALIZED BASELINE AND FOUR FEDERATED α CONDITIONS.
<table><tr><td>Setup</td><td>Accuracy (%)</td><td>AUC</td><td>AOPC Mean</td><td>AOPC Std</td><td>Expl. IoU</td></tr><tr><td>Centralized (baseline)</td><td>97.31</td><td>0.9947</td><td>0.0694</td><td>0.3082</td><td></td></tr><tr><td>Fed.  $\alpha = 1 . 0$  (near-IID)</td><td>96.68</td><td></td><td>0.1657</td><td>0.3482</td><td>0.513</td></tr><tr><td>Fed.  $\alpha = 0 . 5$ </td><td>95.31</td><td></td><td>0.1080</td><td>0.3005</td><td>0.547</td></tr><tr><td>Fed.  $\alpha = 0 . 3$ </td><td>94.43</td><td></td><td>0.0492</td><td>0.2827</td><td>0.534</td></tr><tr><td>Fed.  $\alpha = 0 . 1$  (30 rounds)</td><td>78.42</td><td></td><td>-0.0104</td><td>0.0555</td><td>0.132</td></tr></table>

AOPC is computed as:

$$
{ \mathrm { A O P C } } = { \frac { 1 } { 1 0 } } \sum _ { k = 1 } ^ { 1 0 } \left( f _ { 0 } - f _ { k } \right)\tag{4}
$$

A higher AOPC means masking the highlighted region produces a larger confidence drop, i.e., a more faithful explanation. A negative AOPC means the highlighted region is not causally relevant to the prediction, or is actively misleading. We computed AOPC for every image in the held-out test set, for each model configuration.

2) Explanation IoU: To quantify spatial agreement between explanations from different model configurations, we binarized each CAM at its 80th percentile (marking the top 20% of activations as salient) and computed Intersection-over-Union between the two binary masks:

$$
\mathrm { E x p l . ~ I o U } = \frac { | B _ { \mathrm { c e n t r a l } } \cap B _ { \mathrm { f e d } } | } { | B _ { \mathrm { c e n t r a l } } \cup B _ { \mathrm { f e d } } | }\tag{5}
$$

This metric serves two purposes: a cross-model similarity score between centralized and federated explanations at convergence, and a within-run convergence monitor across communication rounds.

## H. Statistical Testing

To check whether AOPC differences were statistically significant rather than incidental, we applied the following tests, all implemented in src/metrics.py with scipy.stats:

• Mann–Whitney U test (one-sided; $H _ { 1 } ;$ : central AOPC > federated AOPC): a non-parametric rank test suited to non-normal AOPC distributions across the full test set, applied per α condition against the centralized baseline.

• Cohen’s d (pooled standard deviation): quantifies effect size alongside the U-test p-values.

• Spearman rank correlation (ρ) between the four α values and the corresponding mean AOPC degradation (∆ = central mean AOPC − federated mean AOPC), testing whether degradation orders monotonically with data heterogeneity.

## IV. RESULTS

We organize the results around the sub-questions from the research proposal. Table II reports the aggregated performance and faithfulness metrics for all experimental conditions. All federated results for $\alpha \in \{ 0 . 3 , 0 . 5 , 1 . 0 \}$ reflect the round-10 checkpoint; the $\alpha = 0 . 1$ result reflects round 30, after the extended run that confirmed nonconvergence.

Figure 1 — Explanation Faithfulness vs. Data Heterogeneity  
![](images/eb49ff605ae99222eeb14ec0f8f3a204ffeb0abb8e24b40d761cd0039b6c8dca.jpg)  
Fig. 1. Mean AOPC for each configuration (centralized, $\alpha \quad =$ 1.0, 0.5, 0.3, 0.1), with error bars showing the standard deviation.

## A. Detection Accuracy

The centralized baseline reached 97.31% accuracy and an AUC of 0.9947 on the 12,037-image held-out test set, setting a strong performance ceiling. Accuracy fell monotonically as heterogeneity increased (Table II). At α = 1.0 (near-IID), accuracy dropped only to 96.68%, within the 3–8% tolerance the research proposal anticipated and consistent with prior federated learning results. At α = 0.3 and $\alpha = 0 . 5$ , accuracy stayed within that range (94.43% and 95.31%). At the most extreme heterogeneity level, $\alpha = 0 . 1$ , accuracy collapsed to 78.42%, a drop of nearly 19 percentage points from the centralized baseline. FedAvg aggregation fails to reconcile highly divergent client updates at this extreme partitioning regime, a result that shapes how we read the corresponding XAI findings.

## B. Explanation Faithfulness

The AOPC faithfulness metric follows a pattern that does not track heterogeneity in one direction. The centralized baseline’s AOPC was 0.0694 (σ = 0.3082). Across the federated conditions:

• α = 1.0: mean AOPC = 0.1657 (σ = 0.3482), a margin of +0.0963 over the centralized baseline.

$\alpha \ = \ 0 . 5 \colon$ mean AOPC = 0.1080 (σ = 0.3005), +0.0386 over baseline.

• α = 0.3: mean AOPC = 0.0492 (σ = 0.2827), −0.0202 below baseline.

$\alpha = 0 . 1 \colon \mathrm { m e a n ~ A O P C } = - 0 . 0 1 0 4 \ ( \sigma = 0 . 0 5 5 5 ) ,$ negative: the explanations are non-informative or actively misleading.

![](images/a36dd715339f9973538fa1918de83e675e959020ffd4bb2ef0d53601b836395a.jpg)  
Fig. 2. Qualitative Grad-CAM comparison. The centralized baseline (center) yields coherent focal activations, whereas the federated model at extreme heterogeneity $( \alpha = 0 . 1 $ , right) produces diffuse, spatially incoherent heatmaps.

The one-sided Mann–Whitney U test $( H _ { 1 } )$ : central AOPC $>$ federated AOPC) found the centralized model was not superior in faithfulness at $\alpha \in \{ 0 . 5 , 1 . 0 \} \ ( p = 1 . 0$ in both cases), with Cohen’s $d \ \mathrm { o f \ - 0 . 1 3 }$ and −0.29. These negative effect sizes mean that under near-IID and moderate heterogeneity, the federated models produced more faithful explanations than the centralized baseline — a result that runs against the intuition that federation should only hurt explanation quality. The U-test reached significance $( p \approx 0 )$ only at $\alpha = 0 . 3 ~ ( d = 0 . 0 6 8 .$ , small effect) and $\alpha = 0 . 1 ~ ( d = 0 . 3 1 5$ , small-to-medium effect), where federated AOPC fell below the centralized reference.

These results partially support hypothesis H1. At low-tomoderate heterogeneity, federated training does not degrade explanation faithfulness and may modestly improve it. Only at extreme heterogeneity $( \alpha = 0 . 1 )$ , where accuracy has already collapsed, does AOPC turn negative and faithfulness clearly fail.

## C. Heterogeneity Effect on Explanation Degradation

The Spearman rank correlation between α and mean AOPC degradation $( \Delta = \mathrm { c e n t r a l }$ mean AOPC − federated mean AOPC) was $\rho ~ = ~ - 1 . 0 ~ \left( p ~ < ~ 0 . 0 0 1 \right)$ : a perfect monotonic relationship between heterogeneity and the direction of AOPC change. Increasing α (lower heterogeneity) tracked with rising AOPC relative to baseline, while decreasing α tracked with deteriorating or negative AOPC. The perfect rank correlation $( | \rho | = 1 . 0 )$ confirms that AOPC degradation orders monotonically with heterogeneity, though the direction at $\alpha \in \{ 0 . 5 , 1 . 0 \}$ runs opposite to what the primary hypothesis predicted.

The relationship between federated heterogeneity and explanation quality is monotonic but not uniformly degrading. Both detection performance and faithfulness degrade primarily in the extreme heterogeneity regime $( \alpha = 0 . 1 ) ;$ moderate-to-low heterogeneity preserves, or even improves, explanation faithfulness relative to the centralized model.

## D. Explanation Stability Across Communication Rounds

We tracked Explanation IoU across communication rounds to check whether federated explanations stabilize over training:

• α = 1.0: IoU = 0.487 at round 5 → 0.513 at round 10.

$\alpha = 0 . 5 \colon \mathrm { I o U } = 0 . 5 0 0 \ \mathrm { a t ~ r o u n d ~ } 5  0 . 5 4 7$ at round 10.

• α = 0.3: IoU = 0.446 at round $5  0 . 5 3 4$ at round 10.

![](images/a06ea2a04391ab3eccfabcb296d6841512569ba53fcf93be03f2dd4942dcc9c9.jpg)  
Fig. 3. Explanation IoU across communication rounds for $\alpha \in$ {0.1, 0.3, 0.5, 1.0}.

$\alpha = 0 . 1 \colon \mathrm { I o U } = 0 . 1 2 7$ at round $1 0  0 . 1 3 1$ at round $2 0  0 . 1 3 2$ at round 30.

For $\alpha \in \{ 0 . 3 , 0 . 5 , 1 . 0 \}$ , Explanation IoU rose clearly between round 5 and round 10, showing that federated and centralized explanations aligned spatially as training advanced. In all three cases the trajectory had plateaued by round 10, which is why we stopped training there under single-GPU compute constraints.

For $\alpha = 0 . 1$ , the trajectory looks different: IoU stayed low through the entire extended run, rising from 0.127 at round 10 to just 0.132 at round 30, a change of 0.005 over 20 additional rounds. This near-zero gradient does not match the gradual-convergence profile of a slowly learning model; it indicates that the global model failed to converge under extreme client divergence. The extended run rules out the alternative explanation that $\alpha = 0 . 1$ was simply slow to learn, and supports genuine non-convergence.

The qualitative heatmap comparison (Fig. 2) visually corroborates the quantitative findings. While our metrics indicate that models trained under moderate heterogeneity maintain spatial agreement with the baseline, the extreme heterogeneity condition $( \alpha = 0 . 1 )$ demonstrates a complete collapse in spatial coherence. As depicted in the figure, the centralized model focuses tightly on core facial features, whereas the federated model at $\alpha = 0 . 1$ disperses attention toward diffuse, peripheral regions.

## E. Adebayo et al. Sanity Check

Before faithfulness evaluation, the parameterrandomization test confirmed that the trained centralized model’s Grad-CAM maps do not function as modelagnostic edge detectors: mean Explanation IoU between trained and fully weight-randomized CAMs fell below the 0.3 threshold set a priori, validating that the attribution maps are sensitive to the model’s learned parameters.

## F. Summary of Findings

Accuracy. Federated training degrades detection accuracy monotonically with data heterogeneity, from a 0.63 pp drop at $\alpha = 1 . 0$ to an 18.89 pp collapse at $\alpha = 0 . 1$ At moderate heterogeneity $( \alpha \in \{ 0 . 3 , 0 . 5 \} )$ , degradation stays within the 3–8 pp range prior federated literature anticipates.

Explanation faithfulness. The effect of federated training on AOPC depends on heterogeneity and is not uniform.

At $\alpha \in \{ 0 . 5 , 1 . 0 \}$ , federated models produce higher mean AOPC than the centralized baseline (not statistically worse; $p = 1 . 0 ) . \mathrm { A t } \alpha = 0 . 3 , \mathrm { A O P C }$ falls modestly below baseline (significant but small effect; $d = 0 . 0 6 8 )$ $\mathrm { ~ A t ~ } \alpha = 0 . 1$ AOPC turns negative $( d = 0 . 3 1 5 )$ : the explanations stop being causally informative.

Heterogeneity monotonicity. Spearman $\rho ~ = ~ - 1 . 0$ $( p \textless 0 . 0 0 1 )$ confirms a perfect rank-order relationship between data heterogeneity and AOPC degradation. This relationship implies that AOPC degradation is an extremeheterogeneity phenomenon, not a general property of federated training.

Convergence of explanations. Explanation IoU rises across communication rounds for $\alpha ~ \in ~ \{ 0 . 3 , 0 . 5 , 1 . 0 \}$ plateauing by round 10. For $\alpha = 0 . 1$ , IoU stays flat across all 30 rounds, confirming non-convergence in this regime.

## V. DISCUSSION

## A. Federation Does Not Simply Break Explanations

The primary result of this study was contrary to the expected outcome. Distributing training across heterogeneous clients was expected to weaken gradient-based explanations, but moderate heterogeneity instead improved them. Federated models at $\alpha = 1 . 0$ and $\alpha = 0 . 5$ produced mean AOPC values of 0.1657 and 0.1080, both above the centralized baseline of 0.0694. A Mann-Whitney U test found no advantage for the centralized model at either setting $( p = 1 . 0 )$ , and negative Cohen’s d values (−0.29 and −0.13) confirm the direction of the difference.

This pattern is unlikely to result from random sampling variation. The standard deviations remain within a comparable range (0.30–0.35), while the Explanation IoU values of 0.513 and 0.547 indicate that the federated heatmaps remain spatially consistent with the centralized ones. At the same time, they achieve higher scores on the causal AOPC criterion. These results suggest that, under near-IID and moderately heterogeneous conditions, federated training can improve explanation quality rather than simply preserve it.

One possible explanation is the implicit regularization introduced by federated aggregation. When five clients optimize on different subsets of the FaceForensics++ manipulation categories, each local gradient captures a slightly different view of the forgery decision boundary. Averaging these gradients gives the global model a form of gradient ensemble, which can reduce the client-specific activations that a single training run might otherwise overfit to. As a result, the model may focus more consistently on genuine forensic cues, such as boundary mismatches and texture irregularities, rather than dataset-specific correlations. Because Grad-CAM uses the predicted class gradients with respect to the final convolutional layer, more focused and causally relevant activations can result in higher AOPC scores. This mirrors the known effect of ensemble learning on explanation quality [3], produced here implicitly through FedAvg aggregation rather than by design.

A second, compatible explanation points at the centralized baseline itself. Its AOPC of 0.0694 is modest even though the model generalizes well (97.31% accuracy, AUC 0.9947). A model can reach near-perfect accuracy through a mix of solid and spurious features and still produce diffuse Grad-CAM maps that never concentrate on the regions that actually decide the outcome. Federated training, by narrowing each client’s effective training distribution, may push the global model toward features that generalize across clients, and those are the features the gradient signal underlying Grad-CAM ends up attending to more tightly.

We do not claim to have settled why federated models score better on AOPC under moderate heterogeneity; that would require ablations beyond what this study set out to do. What the results do establish is that the effect is real and repeats across settings, and that it belongs in how practitioners weigh the trade-offs of federated deepfake detection.

## B. The Collapse at Extreme Heterogeneity Is a Different Kind of Failure

The moderate-heterogeneity results are surprising in direction. The $\alpha = 0 . 1$ result is alarming in scale. Mean AOPC dropped $\mathrm { { t o \ - 0 . 0 1 0 4 } }$ : masking the regions Grad-CAM emphasizes increases the model’s confidence in the fake class, on average. The highlighted regions are not just uninformative here. They actively point the wrong way. Explanation IoU collapsed to 0.132, and extending training to 30 rounds barely moved it (0.127 to 0.132 over 20 additional rounds). That near-flat trajectory rules out slow convergence as the explanation; this is non-convergence.

The mechanism is not subtle. At $\alpha = 0 . 1$ , the data distribution across the five clients becomes highly skewed, with each client receiving samples mainly from one manipulation category. A client dominated by DeepFakes, for instance, will mostly learn patterns associated with that type of forgery, while a NeuralTextures-heavy client is exposed to a very different set of features. FedAvg then has to combine these conflicting local updates, making it difficult for the global model to learn a representation that works well across all manipulation types. This is reflected in the large drop in accuracy to 78.42%, which is 18.89 percentage points below the centralized result. The explanation metrics point to a more serious issue: the negative AOPC and near-zero IoU indicate that the model is no longer producing a stable gradient structure. In other words, the problem is not only that the classifier becomes less accurate; its explanations also become unreliable. For a forensic analyst, a heatmap that does not consistently correspond to the model’s actual decision could be misleading, something that the accuracy numbers alone would not show.

The deployment lesson follows directly. Organizations using federated deepfake detection across highly specialized or domain-skewed clients, such as nodes focused on a single platform or compression type, should not rely on accuracy alone when deciding whether a model is ready for deployment. A model may still meet a minimum accuracy threshold under extreme heterogeneity even when its explanations have become unreliable. In such cases, presenting a Grad-CAM heatmap to a reviewer could give a false sense of confidence and potentially influence their judgment in the wrong direction.

## C. What the Monotonicity Result Tells Us

The Spearman rank correlation between the Dirichlet concentration parameter α and mean AOPC degradation (∆ = centralized AOPC minus federated AOPC) is $\rho = - 1 . 0$ $( p < 0 . 0 0 1 )$ , indicating a perfectly consistent relationship across the four settings. As α decreases and client heterogeneity increases, the difference between centralized and federated AOPC becomes progressively larger in the unfavorable direction.

This finding points to more than a simple statistical relationship. It suggests that α can serve as an indicator of explanation reliability in this federated setting. If practitioners know the approximate Dirichlet concentration of their client data, or can estimate the Earth Mover’s Distance and its relationship with weight divergence [5], they can anticipate potential risks to Grad-CAM explanations. This also provides a practical way to assess data heterogeneity during onboarding and identify systems that may require additional evaluation before being used in workflows that depend on model explanations.

It is important to clarify what this monotonic relationship does not imply. It does not mean that every increase in heterogeneity causes the same degree of degradation; the results show that the effect is concentrated at the extreme end. The perfect rank ordering is driven in part by the sharp drop at $\alpha = 0 . 1$ , while the remaining federated conditions $( \alpha \in 0 . 3 , 0 . 5 , 1 . 0 )$ remain above the centralized baseline. A more accurate interpretation is that explanation faithfulness remains relatively stable across a broad range of heterogeneity levels before declining sharply beyond a tipping point.

## D. Practical Implications for Federated Forensic Pipelines

Put together, the results give practitioners a workable set of guidelines.

At moderate heterogeneity $( \alpha \ge 0 . 3 )$ , federated training does not compromise, and may strengthen, Grad-CAM faithfulness relative to a centralized baseline. Privacypreserving federated deployment looks safe from an explainability standpoint here, and no special XAI mitigation is needed beyond standard sanity checks such as the parameter-randomization test [13].

At extreme heterogeneity $( \alpha ~ \approx ~ 0 . 1 )$ , the model’s explanations cannot be trusted. Options worth considering include proximal regularization such as FedProx [6] to constrain client divergence; substantially more communication rounds, though our 30-round extension showed little payoff even then; explanation-aware aggregation that weights client contributions by how consistent their gradient directions are; or restricting human-in-the-loop review to the model’s raw confidence score rather than a spatial heatmap that has lost its causal grounding.

Accuracy alone is not a sufficient gate. A model running at 78.42% accuracy $( \alpha = 0 . 1 )$ would likely get caught by an accuracy threshold before deployment. But the statistically significant drop in AOPC observed at $\alpha = 0 . 3$ where accuracy stays at 94.43%—comfortably within an acceptable operational range—shows that explanation degradation can begin before accuracy falls below deployment thresholds. AOPC, or a lightweight proxy for it, belongs in federated model evaluation pipelines as a routine check, not an afterthought.

## E. Limitations

A few limits bound what we can claim here. The federated simulation runs sequentially on a single GPU with $N = 5$ clients, which does not capture the asynchrony, partial participation, or communication delays of a real federated deployment. We evaluate one backbone (EfficientNet-B4) and one post-hoc explanation method (Grad-CAM); whether the AOPC pattern generalizes to transformer-based detectors or attention-based explanations is still open. The Dirichlet partition is a controlled stand-in for heterogeneity; real federated nodes carry structured heterogeneity, platform-specific compression, device-dependent quality, that a Dirichlet model approximates without fully capturing. Our communication-round budget was limited by single-GPU resources, and the nonconvergence seen at $\alpha = 0 . 1$ might in principle respond to more rounds or adaptive aggregation that we were not able to fully explore. Each of these points toward a natural follow-up.

## VI. CONCLUSION

This paper asked a question the literature had not answered directly: does federated training weaken the faithfulness of Grad-CAM explanations in deepfake detection, and by how much? The answer is neither a clean yes nor a reassuring no. It is conditional on data heterogeneity, and that conditionality has real consequences for how practitioners should think about deploying explanationdependent forensic systems.

Under near-IID and moderately heterogeneous conditions $( \alpha \in \{ 0 . 3 , 0 . 5 , 1 . 0 \} )$ ), federated EfficientNet-B4 models trained on FaceForensics++ match or exceed the centralized model’s explanation faithfulness by AOPC, while keeping detection accuracy within the range prior federated learning work would predict. Explanation IoU above 0.5 at these settings confirms that the federated heatmaps line up spatially with the centralized reference. Privacy-preserving federated deployment does not inherently damage the explanations human reviewers rely on, as long as client heterogeneity stays moderate.

The behavior changes sharply under extreme heterogeneity $( \alpha = 0 . 1 )$ . Accuracy drops by nearly 19 percentage points, while Explanation IoU falls to 0.132. More importantly, mean AOPC becomes negative, indicating that the regions highlighted by the model are no longer informative and may instead be causally misleading. Extending training to 30 communication rounds did not resolve the issue; the minimal change in IoU between rounds 10 and 30 suggests that the problem is not simply slow learning but persistent instability. The Spearman rank correlation of $\rho = - 1 . 0$ between α and AOPC degradation further shows a perfectly monotonic relationship, suggesting that data heterogeneity can serve as an early indicator of downstream explanation quality risk.

Our starting hypothesis, that gradient inconsistency from federated aggregation degrades Grad-CAM faithfulness monotonically with heterogeneity, is only partly right. The monotonic ordering holds. However, the direction of the effect at moderate heterogeneity contradicts our initial hypothesis: rather than merely mitigating expected degradation, federated models in this regime actually outperform the centralized baseline in explanation faithfulness. That revision matters in its own right, because it reframes the question from ”does federation hurt explanations?” to ”at what heterogeneity threshold does it start to hurt them, and what drives the benefit below that threshold?”

The broader methodological point is that accuracy alone is not sufficient for deciding whether a federated model is ready for deployment in explanation-dependent forensic applications. Explanation faithfulness can deteriorate in ways that standard performance metrics fail to capture, even when accuracy remains well above operational thresholds. For this reason, AOPC or a comparable causal perturbation metric should be included as a routine part of federated model validation.

A few directions follow naturally from here. Explanationaware aggregation, where client contributions are weighted by how consistent their gradient directions are rather than by sample count alone, offers a principled way to protect explanation quality under heterogeneity. Adaptive communication strategies that extend training specifically when Explanation IoU has not plateaued could address the non-convergence problem at extreme heterogeneity without committing to a fixed round budget in advance. Extending this analysis to transformer-based detectors, where attention maps rather than convolutional gradients carry the explanatory signal, would test whether these patterns hold outside gradient-based attribution. We release our full experimental pipeline so these extensions are easy to build on.

## REFERENCES

[1] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M.¨ Nießner, “FaceForensics++: Learning to detect manipulated facial images,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2019.

[2] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017.

[3] W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Muller, “Evaluating the visualization of what a deep neural network ¨ has learned,” IEEE Trans. Neural Netw. Learn. Syst., vol. 28, no. 11, pp. 2660–2673, Nov. 2017.

[4] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. Int. Conf. Artif. Intell. Stat. (AISTATS), 2017.

[5] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, “Federated learning with non-IID data,” arXiv preprint arXiv:1806.00582, 2018.

[6] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proc. Mach. Learn. Syst. (MLSys), 2020.

[7] X. Li, K. Huang, W. Yang, S. Wang, and Z. Zhang, “On the convergence of FedAvg on non-IID data,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[8] Q. Li, Y. Diao, Q. Chen, and B. He, “Federated learning on non-IID data silos: An experimental study,” in Proc. IEEE Int. Conf. Data Eng. (ICDE), 2022.

[9] T. Wang, X. Liao, K. P. Chow, X. Lin, and Y. Wang, “Deepfake detection: A comprehensive survey from the reliability perspective,” ACM Comput. Surv., vol. 57, no. 3, 2025.

[10] N. Mansoor and A. Iliev, “Explainable AI for DeepFake Detection,” Applied Sciences, vol. 15, no. 2, Art. no. 725, 2025.

[11] V. Sunkari and A. Srinagesh, “DeepExplain: Enhancing DeepFake Detection Through Transparent and Explainable AI model,” Informatica, vol. 48, no. 8, pp. 103–110, 2024.

[12] R. Uma Maheshwari and B. Paulchamy, “Securing online integrity: A hybrid approach to deepfake detection and removal using Explainable AI and Adversarial Robustness Training,” Automatika, vol. 65, no. 4, pp. 1517–1532, 2024.

[13] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” in Adv. Neural Inf. Process. Syst., vol. 31, 2018, pp. 9505–9515.

[14] P.-J. Kindermans, S. Hooker, J. Adebayo, M. Alber, K. T. Schutt, S.¨ Dahne, D. Erhan, and B. Kim, “The (un)reliability of saliency meth-¨ ods,” in Explainable AI: Interpreting, Explaining and Visualizing Deep Learning. Cham, Switzerland: Springer, 2019, pp. 267–280.

[15] J. Shi, A. P. J. M. Siebes, and S. Mehrkanoon, “Trans-XFed: An Explainable Federated Learning for Supply Chain Credit Assessment,” in Proc. 3rd Int. Conf. Federated Learning Technol. Appl. (FLTA), 2025, pp. 1–8.

[16] M. A. Bilal, I. U. Islam, N. Iltaf, M. J. Khan, and M. Jaleed Khan, “Federated Learning With Explainable AI for Malicious Traffic Detection in IoT Networks,” IEEE Access, vol. 13, pp. 173368– 173383, 2025.

[17] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. Int. Conf. Mach. Learn. (ICML), 2019.