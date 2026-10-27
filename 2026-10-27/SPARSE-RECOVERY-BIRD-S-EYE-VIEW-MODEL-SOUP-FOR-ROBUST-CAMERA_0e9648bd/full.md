# SPARSE RECOVERY BIRD’S-EYE VIEW MODEL SOUP FOR ROBUST CAMERA-BASED 3D OBJECT DETECTION

Jinwoo Chung<sup>1</sup>, Sangho An<sup>1</sup>, Hyunsung Park<sup>2</sup>, and Jangho Kim<sup>1∗</sup>

Abstract— Camera-based bird’s-eye view (BEV) perception methods such as Fast-BEV have demonstrated promising results with fast inference speed and competitive accuracy. However, their robustness to real-world corruptions remains insufficiently explored, posing risks for autonomous driving deployment where environmental variations are inevitable. In this paper, we propose Sparse Recovery Bird’s-Eye View Model Soup (SR-BEVS), a sparse recovery learning approach that simultaneously improves clean accuracy and corruption robustness without requiring additional data or architectural modifications. Our method identifies pruning candidates based on layer-wise weight magnitudes, applies random pruning to create diverse sparse models, and retrains each model with a randomly sampled learning rate. By averaging these diverse recovered models into a single dense model, we achieve enhanced robustness against various corruptions while introducing zero inference overhead. We further provide a mode connectivity analysis demonstrating that our pruning-based diversification preserves the connected low-loss basin necessary for stable weight averaging. Experiments on the nuScenes dataset demonstrate up to 33.8% improvement in corruption robustness and a 1.8% gain in clean accuracy over the Fast-BEV baseline, with consistent improvements across sensor noise, weather distortions, and illumination changes. Our simple yet effective approach may be applicable to other BEV models.

## I. INTRODUCTION

Camera-based Bird’s-Eye View (BEV) perception offers a cost-effective approach for autonomous driving. Fast-BEV [1] achieves real-time inference through efficient view transformation and a streamlined architecture, enabling deployment on embedded platforms with competitive detection accuracy. Importantly, this efficiency stems from architectural design rather than simply scaling up model size, avoiding the higher latency, memory footprint, and power consumption that would otherwise hinder practical deployment.

However, robustness to real-world environmental variations is equally critical for safe autonomous driving. Perception failures under adverse weather, illumination changes, or sensor degradation can directly compromise safety-critical decision making. As shown in Fig. 1, a model trained solely on clean data is highly vulnerable to diverse weather conditions and input noise, with substantial performance gaps across all corruption types. Fig. 2 further illustrates this vulnerability: although Fast-BEV achieves accurate detection on clean inputs, its performance degrades drastically under corrupted conditions such as heavy fog, with the mAP dropping from 0.277 to as low as 0.017 under severe Gaussian noise. Despite its importance, the robustness of Fast-BEV under corruption scenarios has not been systematically evaluated, posing potential risks for real-world deployment.

![](images/3b69122998420eea47be1ffd96b4d247e9daac841ac2a5472c83e66978e4bb1d.jpg)  
(a) mAP Improvement in M0  
(b) NDS Improvement in M0  
Fig. 1: The mAP (a) and NDS (b) performance improvement rates (%) of Model Soup and SR-BEVS compared to the Fast-BEV baseline on the M0 model [1]. Results are averaged across severity levels for each of the seven corruption types, plus the nuScenes clean data [2]. The results show that our SR-BEVS consistently outperforms the two baselines.

To address this robustness gap without increasing inference cost—a key constraint for autonomous driving—weight averaging of multiple fine-tuned checkpoints (model soup [3]) is appealing because it yields a single deployable model without architectural changes. However, applying it to Fast-BEV brings only marginal gains: due to the shared training recipe and strong camera-to-BEV inductive bias, independently fine-tuned checkpoints converge to highly similar solutions (Fig. 3), leaving little diversity for averaging to exploit. While one could force diversity via stronger perturbations or substantially different training dynamics, doing so risks leaving the same lowloss basin; once mode connectivity is broken, naive weight averaging/interpolation can cross high-loss regions and even hurt performance [4], [5]. Thus, the core requirement is to increase diversity without breaking mode connectivity, enabling stable averaging with meaningful robustness gains.

We address this challenge by proposing Sparse Recovery Bird’s-Eye View Model Soup (SR-BEVS), which maintains mode connectivity within the same loss basin to enable stable averaging while deliberately increasing diversity so that candidate models learn distinct representations. We provide a dedicated analysis of mode connectivity in Sec. III, showing that SR-BEVS is carefully designed to inject diversity without leaving the connected low-loss region. Inspired by the weight averaging principle of model soup [3] and the well-known generalization benefits of connection removal in dropout [6], SR-BEVS performs a one-time random pruning step to induce distinct sparsity patterns across candidates, followed by a short recovery fine-tuning stage to restore accuracy.

![](images/dccc736a087345af7b5faec81980c51abab6f2059ca413736ce6d69aba43787c.jpg)  
Fig. 2: Qualitative comparison under clean and corrupted inputs. Fast-BEV performs well on clean data but produces incorrect predictions under heavy fog, whereas SR-BEVS maintains accurate detection even under severe corruption, yielding more stable and spatially consistent bounding boxes across the scene.

![](images/097a38e7c709896ff74907df6ba75dc1c2864dcdc662ba56c771e7b49bd28ac2.jpg)  
(a)

![](images/e952c32b1e16df57d5d9ca3e12a66b5e60baecff0712dd634a3bcf6c119a9de5.jpg)  
(b)  
Fig. 3: Weight diversity comparison between Model Soup and SR-BEVS. (a) L2 distance (higher is better) and (b) cosine similarity (lower is better) between candidate model pairs. Standard Model Soup candidates exhibit high similarity, while SR-BEVS produces substantially more diverse models.

Importantly, we prune low-magnitude weights to avoid introducing excessive perturbations. This choice is primarily motivated by the long-standing observation in pruning literature [7], [8], [9] that many small-magnitude parameters contribute little to the prediction function, so removing them can effectively preserve the model’s overall performance while still creating controlled structural differences across candidates. By eliminating these relatively unimportant connections, we encourage complementary feature learning under different sparse patterns while keeping each candidate close to the original solution—thereby promoting diversity and preserving mode connectivity. Finally, we average the recovered candidates into a single dense model, maintaining or improving clean accuracy while substantially reducing performance degradation under corruptions, with no inference overhead. As shown in Fig. 1, SR-BEVS consistently outperforms both the Fast-BEV baseline and standard Model Soup across all corruption types, achieving up to 33.8% improvement in corruption robustness and 1.8% gain in clean accuracy.

Our contributions are summarized as follows:

• We propose a simple yet effective sparse recovery learning method that improves both accuracy and robustness without additional inference cost. Our approach requires no architectural modifications and is validated on two Fast-BEV configurations, providing a practical trainingtime recipe for robustness improvement.

• Our method achieves significant improvements in both corruption robustness and clean accuracy over the Fast-BEV baseline, with an absolute NDS improvement from 0.077 to 0.103 under severe Gaussian noise (+33.8% relative gain).

• We demonstrate that pruning-induced diversity, combined with mode connectivity within the same loss basin, enables effective weight averaging that goes beyond what standard model soup can achieve, providing practical insights for robust deployment.

## II. RELATED WORK

Camera-based BEV Perception. LSS [10] pioneered 2Dto-3D transformation via depth estimation, followed by improvements like BEVDet [11], BEVDepth [12], BEVFusion [13] and Bevstereo [14]. Transformer-based methods (BEV-Former [15], DETR3D [16]) offer high accuracy but demand heavy computation. Fast-BEV [1] uses pre-computed Look-

Up-Tables to reach 52.6 FPS without depth prediction or transformers, balancing speed and accuracy for practical deployment. We select Fast-BEV as our baseline.

Robustness in Autonomous Driving Autonomous vehicles encounter sensor noise, weather effects, and lighting changes in practice. ImageNet-C [17] standardized corruption testing for 2D images, but 3D detection from cameras lacks similar benchmarks. Current methods need data augmentation or adversarial training, yet cannot anticipate every real-world corruption. We enhance Fast-BEV using clean data alone, testing on 7 corruption types at 17 severity levels.

Model Soup and Weight Averaging. Model soup [3] averages the weights of multiple fine-tuned models to improve accuracy and robustness without increasing inference cost. Prior work on loss surfaces [4], [5] showed that naive weight interpolation between models in different loss basins can cross high-loss regions and degrade performance, highlighting the importance of mode connectivity for stable averaging. A subsequent study on transfer learning [18] further demonstrated that models fine-tuned from the same pretrained initialization tend to remain in the same loss basin, enabling safe linear interpolation with low loss throughout the path. However, candidates differing only in learning rate or schedule tend to converge to highly similar solutions, limiting the gains from averaging. Diverse weight averaging [19] addressed this limitation by showing that securing diversity alongside connectivity is critical for achieving meaningful improvements in out-of-distribution robustness.

Model Pruning. Model pruning is a representative approach to model compression, aiming to accelerate edge deployment by reducing parameter count and/or improving inference efficiency [7], [8], [20]. Depending on the granularity, pruning is commonly categorized into unstructured pruning and structured pruning. In practice, real hardware speedups are more readily obtained with structured pruning (e.g., removing channels or filters), whereas aggressive structured pruning often leads to substantial accuracy degradation and is difficult to apply at high pruning ratios [21], [22]. By contrast, unstructured pruning can remove a large fraction of weights with relatively modest accuracy loss, but its latency benefits typically require specialized sparse kernels or API support to translate sparsity into actual speedups [23], [24]. Among various pruning criteria, magnitude pruning is one of the most widely used: weights (or filters) with small absolute values are treated as less important and are preferentially removed. In this work, we adopt random unstructured pruning over low-magnitude weights not for compression, but to induce diversity across candidate models while keeping perturbations controlled. Pruning with a large ratio or removing influential parameters can significantly distort the optimization landscape, potentially breaking mode connectivity and harming accuracy. Our design instead deletes weights that are likely to have limited influence on predictions, thereby enabling different sparse patterns to encourage complementary representations while remaining within the same connected low-loss basin. As a result, we preserve mode connectivity for stable weight averaging and improve robustness without introducing any inference overhead.

![](images/964ac63c09f70cd3aee0f5cf167afe97e67421afb3e591c933dcb0cc58bf4a4a.jpg)  
Fig. 4: Linear interpolation between SR-BEVS candidate model pairs on the M0 configuration. The horizontal axis denotes the interpolation coefficient α in $\theta ( \alpha ) ~ = ~ ( 1 ~ -$ $\alpha ) \theta _ { a } + \alpha \theta _ { b } .$ . Both mAP (left) and NDS (right) remain stable across the entire path, indicating the absence of a loss barrier between candidates.

## III. METHOD

## A. Backbone Architecture: Fast-BEV

We adopt Fast-BEV [1] as our baseline for its efficiency. Fast-BEV transforms N = 6 camera images $\mathbf { I } \in \mathbb { R } ^ { H \times W \times 3 }$ to BEV space through Fast-Ray transformation, which assumes uniform depth along camera rays. Unlike depth-based or transformer-based view transformations that require learned depth prediction or expensive attention operations, Fast-Ray relies entirely on fixed camera geometry, making it highly suitable for deployment on resource-constrained platforms.

The key components are:

(1) Look-Up-Table (LUT). pre-computes 3D-to-2D projection indices offline using fixed camera intrinsic and extrinsic parameters, eliminating runtime projection calculations. Since these parameters are constant once the perception system is built, the projection index is computed only once and stored as a static table for efficient querying during inference.

(2) Multi-View to One-Voxel. directly projects all camera views to the same dense voxel feature, avoiding the expensive sparse voxel aggregation that would otherwise be needed when maintaining separate voxel features per camera.

(3) Space-to-Channel (S2C). operator reshapes 4D voxel tensor $\textbf { V } ~ \in ~ \mathbb { R } ^ { X \times Y \times Z \times C }$ to 3D BEV tensor ${ \bf V } ^ { \prime } \in  \Psi$ $\mathbb { R } ^ { X \times Y \times ( Z \cdot C ) }$ , avoiding expensive 3D convolutions.

We use single-scale image features with temporal fusion that aggregates 4 consecutive frames, resulting in 24 images (6 cameras × 4 timestamps) per sample. While Fast-BEV achieves competitive accuracy with fast inference, our evaluation reveals significant performance degradation under corruptions. We address this through our proposed SR-BEVS training strategy.

## B. Sparse Recovery Bird’s-Eye View Model Soup

To enhance both corruption robustness and clean accuracy, we propose a training strategy that creates diverse models through controlled pruning and recovery, then averages them into a single dense model for deployment. We adopt magnitude-based pruning principles, which have been shown to effectively identify less important weights [25], preserve network capacity [26], and maintain performance under various sparsity levels [27].

Algorithm 1 SR-BEVS: Sparse Recovery Bird’s-Eye View   
Model Soup   
Require: Pretrained Fast-BEV model $\theta _ { 0 }$ , pruning candidate   
ratio $p ,$ number of candidates $N$   
Ensure: Final robust model $\theta _ { \mathrm { f i n a l } }$   
1: // Stage 1: Candidate Pool Generation   
2: for each layer l in $\theta _ { 0 }$ do   
3: Compute magnitude threshold $\tau _ { l }$ at the p-th percentile   
of $| \mathbf { W } _ { l } |$   
4: $\mathcal { C } _ { l }  \{ w _ { i } \in \mathbf { W } _ { l } : | w _ { i } | < \tau _ { l } \}$   
5: end for   
6: // Stage 2: Pruning and Recovery   
7: for $k = 1$ to $N$ do   
8: $\theta _ { k } \gets \theta _ { 0 }$ // Clone pretrained weights   
9: Sample pruning ratio $p _ { k }$ ∼ Uniform(0, 1)   
10: Sample learning rate $\alpha _ { k }$ ∼ Uniform $( \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } )$   
11: for each layer l do   
12: Randomly select $p _ { k }$ fraction of weights from $\mathcal { C } _ { l }$   
13: Set selected weights to zero in $\theta _ { k }$   
14: end for   
15: Fine-tune $\theta _ { k }$ for $E$ epochs with learning rate $\alpha _ { k }$   
// No sparsity mask   
16: end for   
17: // Stage 3: Weight Averaging   
18: $\begin{array} { r } { \theta _ { \mathrm { f i n a l } }  \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \theta _ { k } } \end{array}$ ▷ All $\theta _ { k }$ are dense after recovery   
19: return $\theta _ { \mathrm { f i n a l } }$

Stage 1: Candidate Pool Generation. For each layer l in the pretrained Fast-BEV model $\theta _ { 0 }$ , we identify pruning candidates based on weight magnitudes:

$$
\mathcal { C } _ { l } = \left\{ w _ { i } \in \mathbf { W } _ { l } : | w _ { i } | < \tau _ { l } \right\}\tag{1}
$$

where $\tau _ { l }$ is the threshold chosen such that the pruning candidate ratio $p$ of the weights in layer l (e.g., the lowest 20% when $p = 0 . 2 )$ fall below it. This layer-wise selection ensures that every layer contributes proportionally to the candidate pool, preventing any single layer from being excessively pruned. The pool $\mathcal { C } _ { l }$ consists of the least important weights, whose removal introduces controlled perturbation while preserving the model’s proximity to the original solution.

Stage 2: Model Diversification via Pruning and Recovery. From the shared pretrained checkpoint $\theta _ { 0 } .$ , we create N diverse models through double randomization. For each model $k \in \{ 1 , . . . , N \}$ , we: (1) randomly sample a pruning ratio $p _ { k } ; ( 2 )$ randomly select $p _ { k }$ proportion of weights from the candidate pools {C<sub>l</sub>} for pruning—this double randomization ensures each model has a unique sparsity pattern; (3) prune the selected weights by setting them to zero, creating sparse model $\theta _ { k } ;$ (4) fine-tune $\theta _ { k }$ with a randomly sampled learning rate $\alpha _ { k }$

Crucially, pruning is applied only once before training begins and no sparsity mask is enforced during fine-tuning. The pruned weights are free to recover non-zero values through gradient updates, so each candidate becomes a fully dense model after recovery. Because each candidate was forced to recover from a different damage pattern, the recovery process drives each model toward distinct feature representations and decision boundaries while remaining close to the original pretrained solution.

![](images/97e070873c10dc4a412c79efbf09f6072cdfebf936883adc8effdf609676e345.jpg)

![](images/e73884e6f200cb57329a05314baca8b48b308717345d6946a0e5510284a342c8.jpg)  
(a) Model Soup  
(b) SR-BEVS (Ours)  
Fig. 5: 2D loss landscape visualization on the M0 configuration. Candidate models are projected onto the two principal directions of the weight space. (a) Standard Model Soup candidates cluster tightly near the pretrained model. (b) SR-BEVS candidates are more spread out while remaining within the same low-loss basin, demonstrating increased diversity without breaking mode connectivity.

Stage 3: Weight Averaging for Deployment. The final model is obtained through uniform parameter averaging:

$$
\theta _ { \mathrm { f i n a l } } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \theta _ { k }\tag{2}
$$

Since every candidate is a fully dense model after recovery, $\theta _ { \mathrm { f i n a l } }$ has exactly the same architecture and parameter count as the original Fast-BEV baseline, introducing zero inference overhead. An overview of our training pipeline is summarized in Algorithm 1. When these diverse models with different learned pathways are averaged, their complementary strengths combine while individual vulnerabilities are mitigated, resulting in a model that is more robust to various corruptions. As we verify in Sec. III-C, this averaging is safe because all candidates remain within the same low-loss basin.

## C. Mode Connectivity Analysis

For weight averaging to improve performance, candidate models must reside in the same low-loss region or at least be connected through paths that do not cross high-loss areas. Prior work refers to this condition as mode connectivity [4], [5] and has shown that models fine-tuned from the same pretrained initialization often satisfy it [18]. We verify this property for SR-BEVS both quantitatively and qualitatively through loss barrier analysis.

a) Loss barrier measurement protocol: Given two candidate models $\theta _ { a }$ and $\theta _ { b } ,$ we construct a linear interpolation path $\theta ( \alpha ) = ( 1 - \alpha ) \theta _ { a } + \alpha \theta _ { b }$ for $\alpha \in [ 0 , 1 ]$ and evaluate performance on the validation set at each interpolation point.

If performance collapses in the middle of the path, this indicates that the two models lie in different basins and averaging is unsafe. Conversely, if performance remains stable throughout, the models are connected within the same basin and weight averaging $\begin{array} { r } { \bar { \theta } = \frac { 1 } { K } \sum _ { k } \theta _ { k } } \end{array}$ can be expected to stay in the low-loss region.

b) Empirical verification: Fig. 4 shows the linear interpolation results between SR-BEVS candidate pairs on the M0 configuration. Both mAP and NDS remain stable across the entire interpolation path with no significant loss barrier observed. Furthermore, the 2D loss landscape visualizations in Fig. 5 confirm that SR-BEVS candidates are distributed within the same low-loss basin. In the standard Model Soup setting (Fig. 5a), candidates cluster tightly near the pretrained model due to limited diversity. In contrast, SR-BEVS candidates (Fig. 5b) are more spread out yet still remain within the same basin, indicating that our pruning-based diversification successfully increases inter-model distance without breaking mode connectivity.

c) Implications for averaging gain: The low loss barrier confirmed above is a necessary condition that ensures weight averaging does not drift into high-loss regions. However, connectivity alone is not sufficient for meaningful improvement. When candidates are too similar, as in standard model soup where only learning rates differ, averaging provides limited gain. SR-BEVS satisfies both conditions simultaneously: (i) mode connectivity ensures safe averaging (Fig. 4, Fig. 5), and (ii) pruning-induced diversity ensures that averaging is beneficial by combining complementary feature representations (Fig. 3). This dual satisfaction of connectivity and diversity explains the consistent improvements of SR-BEVS over both the baseline and standard model soup.

## IV. EXPERIMENTS

## A. Experimental Setup

Dataset. We conduct experiments on the nuScenes dataset [2], utilizing only the camera modality for 3D object detection. The dataset provides synchronized images from 6 surround-view cameras covering 360° around the vehicle: three front-facing cameras (CAM FRONT, CAM FRONT LEFT, CAM FRONT RIGHT) and three rear-facing cameras (CAM BACK, CAM BACK LEFT, CAM BACK RIGHT). Each camera captures RGB images at 1600 × 900 resolution. We use the standard split with 28,130 training samples (700 scenes) and 6,019 validation samples (150 scenes). For temporal context, we aggregate 4 consecutive frames at 3-frame intervals, resulting in 24 images (6 cameras × 4 timestamps) per sample. The dataset provides 3D bounding box annotations for 10 categories within a 100 m × 100 m BEV range.

Implementation Details. We implement our method based on Fast-BEV with ResNet [28] backbones (R18 for M0, R34 for M2). Following the Fast-BEV model family design, we evaluate two model configurations: M0 (R18 backbone) and M2 (R34 backbone) from their proposed M0- M5 series, which offer a good balance between efficiency and performance for our pruning experiments. All models are trained on 3 NVIDIA RTX 3090 GPUs. Input images are resized to 256 × 704. For our robust sparse ensemble, we set the pruning candidate ratio to $p = 0 . 2 ,$ , identifying the bottom 20% of weights by magnitude in each layer as pruning candidates, from which random subsets are selected to create diverse sparsity patterns. We create N=3 models using the baseline models. Learning rates are uniformly sampled from [5e-6, 5e-4] for each model. Models are fine-tuned for 2 epochs (M0) or 4 epochs (M2) using AdamW [29] optimizer with batch sizes of 16 (M0) or 12 (M2) samples per GPU.

Corruption Evaluation. To systematically evaluate robustness in 3D detection, we designed a new corruption benchmark. This benchmark is inspired by the widely used ImageNet-C [17] for 2D images, but is specifically tailored to include corruptions commonly encountered in real-world driving scenarios. Specifically, we test on seven corruption types with varying severity levels, totaling 17 different corrupted validation sets: (1) Gaussian Noise (5%, 10%, 15% intensity): simulates sensor noise in low-light conditions; (2) Gaussian Blur (σ=1, 2, 3): models camera focus issues and motion blur; (3) Shot Noise (5%, 10% intensity): represents Poisson noise from camera sensors; (4) Fog (mild, heavy): simulates adverse weather conditions; (5) Contrast (low, high): captures varying lighting from overcast to bright sunlight; (6) Brightness (5%, 10%, 20% increase): models overexposure and bright environments; (7) Occlusion (10%, 20% coverage): simulates partial view blockage from obstacles.

Each corruption is applied to all 6 camera views simultaneously to reflect realistic multi camera degradation. We evaluate on the full validation set (6,019 samples) for each corruption configuration.

Metrics. We evaluate performance using the nuScenes metrics: mAP (mean Average Precision) for detection accuracy and NDS (nuScenes Detection Score), a composite score that includes mAP and errors in key properties like translation, scale, and orientation to reflect detection quality.

Baselines. We compare four methods that share the same architecture and parameter count at inference. Fast-BEV refers to the original pretrained model without any modi fication. Fast-BEV (FT) applies additional fine-tuning with the same training budget as our recovery stage to isolate the effect of extra training from that of pruning-induced diversity. Model Soup trains three models with different randomly sampled learning rates and averages their weights using (2). SR-BEVS averages three dense models recovered from differently pruned variants as detailed in Sec. III-B.

Training and Inference Cost. Since the final model $\theta _ { \mathrm { f i n a l } }$ has the identical architecture and parameter count as a single Fast-BEV model, SR-BEVS introduces zero additional inference overhead in terms of latency, memory, and power consumption. The extra cost is limited to the training phase, where generating N candidates requires N independent recovery fine-tuning runs of E epochs each. This cost scales linearly with N, but as shown in our ablation (Table IV), performance gains exhibit diminishing returns beyond N=3, making it a practical default choice.

TABLE I: Robustness evaluation on nuScenes validation set under diverse corruptions and severity levels. Fast-BEV (FT) uses the same additional fine-tuning budget as SR-BEVS recovery.
<table><tr><td></td><td></td><td>Method</td><td colspan="2">Gauss Noise 10%</td><td colspan="2"></td><td colspan="2">Gauss Blur</td><td colspan="2">Shot Noise 10%</td><td colspan="2"> $\mathrm { ~ \bf ~ M ~ } ^ { \mathrm { F o g } } \mathrm { ~ \bf ~ H ~ }$ </td><td colspan="2">Contr. H</td><td colspan="2">Bright. 10%</td><td colspan="2">Occl. 10%</td></tr><tr><td></td><td rowspan="2"></td><td></td><td>5%</td><td></td><td>15%</td><td> $\sigma 1$ </td><td> $\sigma 2$ </td><td> $\sigma 3$ </td><td>5%</td><td></td><td></td><td></td><td>L</td><td></td><td>5%</td><td>20%</td><td></td><td>20%</td></tr><tr><td>mAP</td><td>Fast-BEV Fast-BEV (FT)</td><td>0.189 0.190</td><td>0.079 0.081</td><td>0.017</td><td>0.227</td><td>0.125 0.051</td><td>0.221 0.222</td><td>0.219</td><td>0.234</td><td>0.173</td><td>0.266</td><td>0.197</td><td>0.276</td><td>0.276</td><td>0.275</td><td>0.226</td><td>0.180</td></tr><tr><td rowspan="6">M0</td><td rowspan="3"></td><td>Model Soup</td><td>0.190</td><td>0.080</td><td>0.017 0.016</td><td>0.227 0.229</td><td>0.126 0.127</td><td>0.052 0.053</td><td>0.219 0.222 0.220</td><td>0.234 0.236</td><td>0.174 0.175</td><td>0.267 0.266</td><td>0.198 0.198</td><td>0.276 0.278</td><td>0.277 0.278</td><td>0.275 0.277</td><td>0.227 0.228</td><td>0.180 0.182</td></tr><tr><td>SR-BEVS</td><td>0.198</td><td>0.087</td><td>0.018</td><td>0.231</td><td>0.129</td><td>0.056</td><td></td><td>0.247</td><td>0.192</td><td>0.270</td><td>0.202</td><td></td><td>0.282</td><td>0.281</td><td>0.281 0.233</td><td>0.187</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.230</td><td>0.228</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fast-BEV</td><td>0.325</td><td>0.181</td><td>0.077</td><td>0.373</td><td>0.290</td><td>0.153</td><td>0.360</td><td>0.359</td><td>0.378</td><td>0.325</td><td>0.403</td><td>0.339</td><td>0.412</td><td>0.411</td><td>0.409</td><td>0.369</td><td>0.324</td></tr><tr><td>Fast-BEV (FT) NDS</td><td>0.326</td><td>0.181</td><td>0.080</td><td>0.374</td><td>0.290</td><td>0.153</td><td>0.360</td><td>0.360</td><td>0.378</td><td>0.326</td><td>0.403</td><td>0.349</td><td>0.413</td><td>0.412</td><td>0.409</td><td>0.370</td><td>0.325</td></tr><tr><td rowspan="3"></td><td>Model Soup SR-BEVS</td><td>0.326 0.337</td><td>0.181 0.187</td><td>0.085 0.103</td><td>0.375 0.381</td><td>0.291 0.298</td><td>0.152 0.186</td><td>0.360</td><td>0.360</td><td>0.378</td><td>0.327</td><td>0.403</td><td>0.340</td><td>0.412</td><td>0.412</td><td>0.410 0.414</td><td>0.370</td><td>0.323 0.330</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.371</td><td>0.370</td><td>0.389</td><td>0.341</td><td>0.406</td><td>0.342</td><td>0.417</td><td>0.416</td><td></td><td>0.377</td><td></td></tr><tr><td>Fast-BEV</td><td>0.218</td><td>0.104</td><td>0.031</td><td>0.275</td><td>0.155</td><td>0.054</td><td>0.254</td><td>0.252</td><td>0.290</td><td>0.231</td><td>0.318</td><td>0.240</td><td>0.325</td><td>0.324</td><td>0.322</td><td>0.274</td><td>0.230</td></tr><tr><td rowspan="7">M2</td><td rowspan="7">mAP</td><td>Fast-BEV (FT)</td><td>0.219 0.221</td><td>0.105</td><td>0.031 0.032</td><td>0.277 0.279</td><td>0.157 0.161</td><td>0.055</td><td>0.256</td><td>0.254 0.256</td><td>0.291</td><td>0.231</td><td>0.319 0.322</td><td>0.240</td><td>0.327</td><td>0.326 0.330</td><td>0.324 0.328</td><td>0.276 0.280</td><td>0.232 0.236</td></tr><tr><td>Model Soup</td><td>0.229</td><td>0.106 0.114</td><td>0.038</td><td>0.287</td><td></td><td>0.058</td><td>0.259</td><td></td><td>0.293</td><td>0.233</td><td>0.327</td><td>0.245</td><td>0.330</td><td></td><td>0.333</td><td>0.284</td><td>0.238</td></tr><tr><td>SR-BEVS</td><td></td><td></td><td></td><td></td><td>0.174</td><td>0.067</td><td>0.268</td><td>0.267</td><td>0.301</td><td>0.248</td><td></td><td>0.255</td><td>0.335</td><td>0.335</td><td></td><td></td><td></td></tr><tr><td>Fast-BEV</td><td>0.370</td><td>0.228</td><td>0.151</td><td>0.416</td><td>0.321</td><td>0.171</td><td>0.403</td><td>0.399</td><td>0.426</td><td>0.380</td><td>0.450</td><td>0.381</td><td>0.453</td><td>0.453</td><td>0.450</td><td>0.414</td><td>0.374</td></tr><tr><td>Fast-BEV (FT)</td><td>0.372 0.375</td><td>0.230</td><td>0.152</td><td>0.417</td><td>0.323</td><td>0.172</td><td>0.405</td><td>0.401</td><td>0.427</td><td>0.381</td><td>0.450</td><td>0.382</td><td>0.455</td><td>0.455</td><td>0.452</td><td>0.416</td><td>0.376</td></tr><tr><td>Model Soup</td><td></td><td>0.232</td><td>0.153</td><td>0.419</td><td>0.325</td><td>0.191</td><td>0.408</td><td>0.404</td><td>0.432</td><td>0.384</td><td>0.455</td><td>0.388</td><td>0.459</td><td>0.459</td><td>0.457</td><td>0.418</td><td>0.382</td></tr><tr><td>SR-BEVS</td><td>0.383</td><td>0.254</td><td>0.161</td><td>0.429</td><td>0.342</td><td>0.203</td><td>0.413</td><td>0.411</td><td>0.435</td><td>0.391</td><td>0.460</td><td>0.396</td><td>0.462</td><td>0.461</td><td>0.460</td><td>0.423</td><td>0.385</td></tr></table>

TABLE II: Detection performance on nuScenes validation set. Fast-BEV (FT) denotes additional fine-tuning with the same training budget as SR-BEVS recovery.
<table><tr><td rowspan="2">Method</td><td colspan="2">M0 (R18)</td><td colspan="2">M2 (R34)</td></tr><tr><td>mAP↑</td><td>NDS↑</td><td>mAP↑</td><td>NDS↑</td></tr><tr><td>Fast-BEV</td><td>0.277</td><td>0.411</td><td>0.326</td><td>0.455</td></tr><tr><td>Fast-BEV (FT)</td><td>0.278</td><td>0.412</td><td>0.327</td><td>0.457</td></tr><tr><td>Model Soup</td><td>0.278</td><td>0.412</td><td>0.330</td><td>0.457</td></tr><tr><td>SR-BEVS</td><td>0.282</td><td>0.417</td><td>0.332</td><td>0.460</td></tr></table>

## B. Quantitative Results

Clean Data Performance. Table II shows that our method consistently outperforms both baselines across all configurations. For M0, we achieve 0.282 mAP and 0.417 NDS, surpassing Fast-BEV (0.277 mAP/0.411 NDS) and standard model soup (0.278 mAP/0.412 NDS). Similarly, M2 reaches 0.332 mAP and 0.460 NDS, exceeding baseline (0.326 mAP/0.455 NDS) and soup (0.330 mAP/0.457 NDS). While standard model soup provides marginal gains, our pruning-based approach delivers more substantial improvements, demonstrating that sparsity-induced diversity is more effective than simple fine-tuning diversity for model ensembling.

Corruption Robustness. We evaluate robustness against critical corruptions commonly encountered in autonomous driving scenarios. We test seven corruption types that directly impact camera-based perception: sensor noise (Gaussian, Shot), motion artifacts (Blur), weather conditions (Fog), lighting variations (Contrast, Brightness), and partial occlusions. As shown in Table I, our method consistently improves performance across all corruption types and severity levels for both M0 and M2 models. For M0, the most significant improvements occur under severe Gaussian noise (15%: NDS 0.077→0.103, +33.8%) and heavy fog (mAP 0.173→0.192, +11.0%). M2 demonstrates even stronger gains with Gaussian blur (σ = 3: NDS 0.171→0.203, +18.7%) and heavy fog (mAP 0.231→0.248, +7.4%). These consistent improvements across diverse corruptions demonstrate that SR-BEVS enhances model resilience to real-world visual degradations critical for safe autonomous driving deployment.

## C. Qualitative Results

We select two corruption scenarios frequently encountered in real-world driving, heavy fog and severe Gaussian noise, and analyze the representative failure modes exhibited by each method.

a) Severe Gaussian noise:false positives: Under strong Gaussian noise, the baseline tends to produce false positive detections by misinterpreting local high-frequency noise patterns as objects such as pedestrians in regions where no actual objects exist. This is a typical robustness failure where noise artifacts are mistaken for object cues. SR BEVS substantially reduces these false positives under the same conditions. Weight averaging suppresses the extreme confidence scores and bounding box predictions of individual candidates, driving the ensemble toward more conservative and stable predictions.

b) Heavy fog: missed detections: Under fog conditions, the overall contrast is severely degraded and object contours and texture cues become ambiguous. As a result, the baseline fails to detect certain objects that are present in the scene. SR-BEVS reliably detects these objects under the same conditions, maintaining a coherent BEV representation even in low-visibility environments. This can be attributed to the complementary predictions of the candidate models: when one candidate lacks sufficient cues to detect an object, other candidates that have learned alternative feature pathways can compensate, and their averaged prediction recovers the missed detection. These qualitative observations suggest that SR-BEVS simultaneously mitigates the two dominant error modes under corrupted inputs: (i) missed detections caused by low visibility and (ii) false positives triggered by noise patterns. This is consistent with the mode connectivity analysis in Sec. III-C, where the low loss barrier enables safe averaging and the prediction complementarity across diverse candidates translates into tangible robustness gains.

![](images/b4c2685a06d3bafd304cbbf3795c81be4e79424b40751eaba14690d356924936.jpg)  
Fig. 6: Qualitative comparison of 3D detection results under severe corruptions on the M0 configuration. Top row: under strong Gaussian noise, Fast-BEV and Model Soup misclassify noise patterns as objects such as pedestrians (green boxes), while SR-BEVS avoids these incorrect predictions. Bottom row: under heavy fog, Fast-BEV and Model Soup miss objects present in the scene, whereas SR-BEVS maintains reliable detection consistent with the ground truth.

TABLE III: Effect of pruning candidate ratio p on model performance (M0, clean data)
<table><tr><td>Pruning Candidate Ratio (p)</td><td> $\mathbf { m A P \uparrow }$ </td><td> $\mathbf { N D S } \uparrow$ </td></tr><tr><td>Baseline</td><td>0.277</td><td>0.411</td></tr><tr><td> $p = 0 . 1$ </td><td>0.280</td><td>0.414</td></tr><tr><td> $p = 0 . 2$ </td><td>0.282</td><td>0.417</td></tr><tr><td> $p = 0 . 3$ </td><td>0.279</td><td>0.413</td></tr></table>

## D. Ablation Study

We conduct ablation experiments on the M0 configuration to analyze the impact of each design choice in SR-BEVS. Unless otherwise stated, all results are reported on the nuScenes validation set using clean data.

Optimal Pruning Candidate Ratio. Table III investigates the impact of the pruning candidate ratio p on model performance. We identify the bottom p% weights by magnitude in each layer as pruning candidates, then randomly select subsets from this pool for actual pruning. A pruning candidate ratio of $p = 0 . 2$ achieves the best balance with 0.282 mAP and 0.417 NDS, outperforming both $p = 0 . 1$ (0.280 mAP, 0.414 NDS) and $p = 0 . 3$ (0.279 mAP, 0.413 NDS). $\mathrm { A t } \ p = 0 . 1$ , the candidate pool is too small to generate sufficiently distinct sparsity patterns across candidates, limiting the diversity that drives averaging gains. At $p =$ 0.3, the pool includes weights of higher magnitude whose removal introduces excessive perturbation, making it difficult for models to fully recover within the short fine-tuning budget, and performance drops back toward the baseline. The $p = 0 . 2$ setting strikes an effective balance between inducing enough structural perturbation to encourage diverse recovery pathways and preserving sufficient network capacity for the pruned weights to regain meaningful values during finetuning.

Number of Candidate Models. Table IV examines how the number of candidate models N affects the final averaged model. With $N = 1$ , the method reduces to a single prunedand-recovered model without any averaging, which already slightly outperforms the Fast-BEV baseline due to the regularization effect of the pruning-recovery process. Increasing to $N = 2$ brings an additional gain from weight averaging, and $N = 3$ further improves performance to 0.282 mAP and 0.417 NDS. Extending to $N = 5$ yields a marginal additional improvement of 0.284 mAP and 0.419 NDS, suggesting diminishing returns beyond three candidates. We use $N = 3$ in our main experiments as it provides a favorable trade-off between training cost and performance, since each additional candidate requires a separate fine-tuning run.

TABLE IV: Effect of the number of candidate models N on model performance (M0, clean data)
<table><tr><td>#Candidates (N)</td><td>mAP ↑</td><td>NDS ↑</td></tr><tr><td>1</td><td>0.278</td><td>0.412</td></tr><tr><td>2</td><td>0.280</td><td>0.414</td></tr><tr><td>3</td><td>0.282</td><td>0.417</td></tr><tr><td>5</td><td>0.284</td><td>0.419</td></tr></table>

TABLE V: Weight Diversity Analysis and Performance Comparison
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Weight Diversity</td><td colspan="2">Clean</td><td colspan="2">Corrupted (Avg.)</td></tr><tr><td>Cos.Sim.↓</td><td>L2 Dist.↑</td><td>mAP</td><td>NDS</td><td>mAP</td><td>NDS</td></tr><tr><td rowspan="2">M0</td><td>Soup</td><td>0.914</td><td>1.706</td><td>0.278</td><td>0.412</td><td>0.191</td><td>0.324</td></tr><tr><td>SR-BEVS</td><td>0.912</td><td>5.064</td><td>0.282</td><td>0.417</td><td>0.197</td><td>0.333</td></tr><tr><td rowspan="2">M2</td><td>Soup</td><td>0.900</td><td>5.795</td><td>0.330</td><td>0.457</td><td>0.233</td><td>0.373</td></tr><tr><td>SR-BEVS</td><td>0.893</td><td>24.389</td><td>0.332</td><td>0.460</td><td>0.241</td><td>0.381</td></tr></table>

Model Diversity Analysis. We analyze the relationship between weight diversity and performance to validate our core hypothesis that pruning-induced diversity is the key driver behind the improvements of SR-BEVS. As shown in Table V, SR-BEVS achieves substantially higher weight diversity compared to standard model soup across both model configurations. For M0, the L2 distance between candidate pairs increases approximately 3× from 1.706 to 5.064, while cosine similarity decreases from 0.914 to 0.912. The effect is even more pronounced for M2, where the L2 distance increases approximately 4× from 5.795 to 24.389 and cosine similarity drops from 0.900 to 0.893. This greater diversity in M2 is consistent with the larger model having more parameters available for pruning, which amplifies the structural differences created by distinct sparsity patterns.

Importantly, the enhanced diversity translates directly into performance gains, and the gains are more pronounced on corrupted data than on clean data. For M0, the corruptedaverage NDS improves by 0.009 (0.324→0.333) compared to a clean NDS improvement of 0.005 (0.412→0.417). This pattern holds for M2 as well, with corrupted NDS improving by 0.008 versus 0.003 on clean data. This asymmetry suggests that diverse feature representations are particularly beneficial under distribution shift, where the complementary strengths of differently trained candidates provide broader coverage against varied corruption types. The consistent correlation between higher diversity and better robustness validates our hypothesis that sparse recovery learning creates ensembles whose diversity is both meaningful and beneficial.

## V. CONCLUSION

We investigated robustness limitations of camera-based BEV perception and found that Fast-BEV, while accurate on clean inputs, degrades sharply under realistic corruptions. To close this gap without increasing inference cost, we proposed Sparse Recovery Bird’s-Eye View Model Soup (SR-BEVS), a training-time strategy that makes weight averaging effective by jointly preserving mode connectivity and injecting pruning-induced diversity. The recovered candidate models are averaged into a single dense network with identical architecture and zero inference overhead. Extensive experiments on nuScenes demonstrate that SR-BEVS consistently outperforms the Fast-BEV baseline and standard model soup, improving clean accuracy while substantially reducing performance degradation under corruptions (up to 33.8% robustness gain under severe noise). While our validation is currently limited to Fast-BEV and synthetic corruptions, SR-BEVS offers a practical training-time recipe for improving robustness in deployment-constrained camera-based BEV perception.

## REFERENCES

[1] Y. Li, B. Huang, Z. Chen, Y. Cui, F. Liang, M. Shen, F. Liu, E. Xie, L. Sheng, W. Ouyang et al., “Fast-bev: A fast and strong bird’s-eye view perception baseline,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 8665–8679, 2024.

[2] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 621–11 631.

[3] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith et al., “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” in International conference on machine learning. PMLR, 2022, pp. 23 965–23 998.

[4] T. Garipov, P. Izmailov, D. Podoprikhin, D. P. Vetrov, and A. G. Wilson, “Loss surfaces, mode connectivity, and fast ensembling of dnns,” Advances in neural information processing systems, vol. 31, 2018.

[5] F. Draxler, K. Veschgini, M. Salmhofer, and F. Hamprecht, “Essentially no barriers in neural network energy landscape,” in International conference on machine learning. PMLR, 2018, pp. 1309–1318.

[6] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” The journal of machine learning research, vol. 15, no. 1, pp. 1929–1958, 2014.

[7] S. Han, J. Pool, J. Tran, and W. Dally, “Learning both weights and connections for efficient neural network,” Advances in neural information processing systems, vol. 28, 2015.

[8] M. Zhu and S. Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression,” arXiv preprint arXiv:1710.01878, 2017.

[9] T. Gale, E. Elsen, and S. Hooker, “The state of sparsity in deep neural networks,” 2019.

[10] J. Philion and S. Fidler, “Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d,” in European conference on computer vision. Springer, 2020, pp. 194–210.

[11] J. Huang, G. Huang, Z. Zhu, Y. Ye, and D. Du, “Bevdet: Highperformance multi-camera 3d object detection in bird-eye-view,” arXiv preprint arXiv:2112.11790, 2021.

[12] Y. Li, Z. Ge, G. Yu, J. Yang, Z. Wang, Y. Shi, J. Sun, and Z. Li, “Bevdepth: Acquisition of reliable depth for multi-view 3d object detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 2, 2023, pp. 1477–1485.

[13] T. Liang, H. Xie, K. Yu, Z. Xia, Z. Lin, Y. Wang, T. Tang, B. Wang, and Z. Tang, “Bevfusion: A simple and robust lidar-camera fusion framework,” Advances in Neural Information Processing Systems, vol. 35, pp. 10 421–10 434, 2022.

[14] Y. Li, H. Bao, Z. Ge, J. Yang, J. Sun, and Z. Li, “Bevstereo: Enhancing depth estimation in multi-view 3d object detection with temporal stereo,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 2, 2023, pp. 1486–1494.

[15] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai, “Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[16] Y. Wang, V. C. Guizilini, T. Zhang, Y. Wang, H. Zhao, and J. Solomon, “Detr3d: 3d object detection from multi-view images via 3d-to-2d queries,” in Conference on robot learning. PMLR, 2022, pp. 180–191.

[17] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” arXiv preprint arXiv:1903.12261, 2019.

[18] B. Neyshabur, H. Sedghi, and C. Zhang, “What is being transferred in transfer learning?” Advances in neural information processing systems, vol. 33, pp. 512–523, 2020.

[19] A. Rame, M. Kirchmeyer, T. Rahier, A. Rakotomamonjy, P. Gallinari, and M. Cord, “Diverse weight averaging for out-of-distribution generalization,” Advances in Neural Information Processing Systems, vol. 35, pp. 10 821–10 836, 2022.

[20] J. Kim, J. Yoo, Y. Song, K. Yoo, and N. Kwak, “Finding efficient pruned network via refined gradients for pruned weights,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 9003–9011.

[21] S. An, J. Kim, K. Lee, J. Huh, C. Kwak, Y. Lee, M. Jin, and J. Kim, “Sparse structure exploration and re-optimization for vision transformer,” in The 41st Conference on Uncertainty in Artificial Intelligence, 2025.

[22] K. Shim, J. Yun, and S. Choi, “Snp: Structured neuron-level pruning to preserve attention scores,” in European Conference on Computer Vision. Springer, 2024, pp. 90–104.

[23] J. Kim, J. Shin, S. An, and J. Kim, “Exploring diverse sparse network structures via dynamic pruning with weight alignment,” in Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 1324–1333.

[24] T. Lin, S. U. Stich, L. Barba, D. Dmitriev, and M. Jaggi, “Dynamic model pruning with feedback,” arXiv preprint arXiv:2006.07253, 2020.

[25] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning filters for efficient convnets,” arXiv preprint arXiv:1608.08710, 2016.

[26] J. Frankle and M. Carbin, “The lottery ticket hypothesis: Finding sparse, trainable neural networks,” arXiv preprint arXiv:1803.03635, 2018.

[27] D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, and J. Guttag, “What is the state of neural network pruning?” Proceedings ofmachine learning and systems, vol. 2, pp. 129–146, 2020.

[28] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[29] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.