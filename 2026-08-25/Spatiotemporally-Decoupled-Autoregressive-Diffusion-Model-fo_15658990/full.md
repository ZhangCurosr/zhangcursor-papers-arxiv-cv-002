# Spatiotemporally Decoupled Autoregressive Diffusion Model for Human Motion Generation

Chengqun Yang, Liang Xu, Yanping Li, Fulong Liu, Jingnan Gao, Weili Zeng, Yichao Yan MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University {ycq0191, liangxu, ypli2024, lfl925884898, gjn0310, zwl666, yanyichao}@sjtu.edu.cn

Abstract—Text-driven human motion synthesis has made substantial development with two core modules of motion representation and generative architecture. For representation, Vector Quantization (VQ)-based methods compress motion data into discrete tokens while latent-based models operate directly in continuous space. However, both of these representations exhibit significant limitations. VQ-based methods suffer from inherent information loss, which compromises the quality, diversity, and generalization of generated motions, while continuous representation on holistic whole-body motion hinders part-level flexibility. For architecture, diffusion and autoregressive diffusion models have demonstrated their superiority, yet the fine-grained controllability over individual body parts is also limited. Thus, we propose a unified spatiotemporally decoupled framework named DeMoDiff, which jointly redesigns representation and architecture. To enhance representation extraction capabilities and offer greater part-level controllability, we present a spatialtemporal VAE that encodes each body joint rather than compressing the whole-body motion into a single latent space. Then, we incorporate spatial-temporal masking and attention mechanisms into an autoregressive diffusion generator, achieving both generative capability and controllable editability. Extensive experiments on the HumanML3D and KIT-ML datasets demonstrate that our model achieves state-of-the-art reconstruction performance and compelling motion generation results. Moreover, our framework demonstrates strong temporal and spatial editing capabilities, further validating its effectiveness. Our project page: https://rex0191.github.io/DeMoDiff/.

Index Terms—Text-driven human motion generation, Autoregressive diffusion model, Spatial-temporal decoupling.

## I. INTRODUCTION

Text-driven human motion generation [1]–[5] is a rapidly advancing frontier in computer vision, focusing on synthesizing realistic and controllable 3D human motion sequences that semantically align with the given language prompt. Existing approaches for human motion generation typically comprise two essential components: motion representation and generative architecture.

Regarding motion representation, mainstream approaches primarily fall into two dominant paradigms of Vector Quantization (VQ)-based [3], [4], [6]–[8] and latent-based representations [2], [9], [10]. Generative architectures are inherently entangled with motion representations. For instance, VQ-based methods typically employ autoregressive models to predict discrete tokens [3], [4], [7]. Conversely, diffusion [1], [2], [11], [12] and autoregressive diffusion models [9], [10] have demonstrated their superiority, yet the fine-grained controllability over individual body parts is also limited. Although continuous regression avoids the quantization-induced precision loss of VQ-based methods, it introduces a more challenging problem of directly regressing articulated and high-dimensional human motions, making fine-grained manipulation infeasible.

![](images/fd9efe69c38593c527d34a09bec94dbf9776fad47f215161a78758a6ed1d435c.jpg)  
Fig. 1: Comparison of motion representation (A1-3) and generation model (B1-3). Our proposed DeMoDiff is composed of spatial-temporal VAE and spatial-temporal autoregressive diffusion model for enhanced motion representation, generation, and spatiotemporal editing capabilities.

To overcome the limitations of existing methods, we introduce a unified spatiotemporally decoupled framework with two specific modules as illustrated in Fig. 1, ensuring highfidelity generation and flexible controllability. For motion representation, we propose a lightweight spatial-temporal VAE that encodes each body joint rather than compressing the whole-body motion into a single latent space. The decoupled paradigm substantially enhances representation extraction performance and offers greater part-level controllability. For the generative architecture, we incorporate spatial-temporal masking and attention mechanisms into an autoregressive diffusion generator, achieving both generative capability and controllable editability.

We conduct extensive experiments on HumanML3D [13] and KIT-ML [14] datasets to validate the superiority of our proposed spatiotemporally Decoupled autoregressive Motion Diffusion model (DeMoDiff), for text-driven motion generation and motion editing. We also offer in-depth analysis and visualizations of our method, indicating that further advancements in autoregressive diffusion technology hold the potential to enhance its performance in the future.

In summary, our main contributions are as follows:

• We novelly compress human motion into a continuous 2D latent space based on a spatial-temporal VAE, with improved reconstruction and generation metrics.

• A temporal-spatial masking strategy and diffusion mechanism adapted to continuous 2D latent of temporal-spatial representation, supporting fine-grained temporal-spatial control and motion generation.

• Our method demonstrates superior performance in several motion-related tasks, especially in joint-level editing.

## II. RELATED WORK

Text-driven Human Motion Generation. Text-driven human motion generation [13], [15], [16] aims to synthesize realistic 3D human motions from natural languages, which poses significant challenges in fine-grained semantic mapping between texts and body movements, intricate body part control, and elaborate textual descriptions.

Vector Quantization (VQ)-based methods [6]–[8] encode motion data into discrete tokens for subsequent autoregressive motion synthesis. However, the sequential generation design inherently impedes the ability to model long-horizon temporal relationships. Most recent works [3], [4] employ bidirectional masked generation models similar to BERT [17], which allows the model to predict all masked tokens simultaneously and enhances the context modeling capability. Despite the impressive performance, VQ-based approaches suffer from inevitable information loss. In this paper, we introduce a lightweight spatial-temporal VAE to encode motion into latent vectors with better motion reconstruction performance. Denoising diffusion models, which have achieved significant success in image synthesis, have now become powerful tools for generating human motion. Some attempts [1], [11], [12] operate on raw motion representation, while MLD [2] uses a Transformerbased VAE to compress the entire motion sequence to latent vectors. Optimizing the denoising process in the latent space enhances training and sampling efficiency yet limits the model’s ability to capture temporal and spatial dynamics. Different from previous diffusion-based methods, we compress the raw motion into temporal and spatial latent space with a CNN-based encoder and decoder, and we autoregressively generate temporal and spatial motion latent during the inference stage with part-level control.

Human Motion Representation. Human motion representation is critical for motion-related tasks. Previous works have extensively explored single human motion representations from the perspectives of both dimensional design and compact encoding. From a dimensional perspective, HumanML3D [13] presents a 263-d representation that includes root angular velocity, root linear velocities, root height, local joint positions, velocities, rotations, and foot-ground contact. MotionStreamer [10] further introduces a 272-d representation to avoid data duplication and intricate post-processing of converting to SMPL parameters. MARDM [9] excludes the redundant dimensions of the 263-d format, resulting in a more essential and compact representation. MotionPatches [18] divides and sorts skeleton joints based on body parts similar to image patches for ViT [19] pre-training. To achieve better joint encoding capability and joint-level spatial relationship, MoGenTS [3] splits the 263-d representation along the joint dimension. From a compression perspective, many works [3], [4], [6]–[8] represent motion as a discrete codebook. Some approaches [9], [10] mapping the motion sequence into a temporal latent sequence, while MLD [2] compresses the entire motion sequence into a single latent vector. In this paper, we follow the dimensional setting of MoGenTS [3] for spatialtemporal decoupling and part-level editing, and subsequently encode motion into a 2D latent representation.

## III. THE DEMODIFF MODEL

The architecture overview of our proposed DeMoDiff model is demonstrated in Fig. 2. We provide a detailed explanation of spatial-temporal VAE for motion representation and spatialtemporal autoregressive diffusion architecture for continuous human motion generation.

## A. Spatial-Temporal VAE.

The architecture of our proposed spatial-temporal VAE is shown in Fig. 2 (A). Following the previous method [3], we organize the motion sequence into a 2D structure (in both temporal and joint dimensions) and process it as a 2D map, serving as the input to the VAE. For a motion sequence ${ \bf X } _ { 2 d } =$ $\{ x _ { 2 d } ^ { i , k } \} _ { i = 1 , k = 1 } ^ { T , J }$ with length $T$ and joint number $^ { J , }$ we employ a 2D convolutional network to encode the 2D motion features $x _ { 2 d }$ into parameters of latent probability distributions, which are decoded to motion sequence. The network is optimized by a loss considering both reconstruction quality and latent space regularization, as:

$$
\mathcal { L } _ { 2 d } = \mathcal { L } _ { n l l } ^ { 2 d } + \lambda _ { k } ^ { 2 d } \cdot \mathcal { L } _ { K L } ^ { 2 d } + \lambda _ { v } \cdot \mathcal { L } _ { v } ,\tag{1}
$$

where $\mathcal { L } _ { n l l } ^ { 2 d }$ is negative log-likelihood (NLL) loss, $\lambda _ { k } ^ { 2 d }$ and $\lambda _ { v }$ are the hyper-parameters for the Kullback Leibler (KL) loss $\mathcal { L } _ { K L } ^ { 2 d }$ and joint velocity loss $\mathcal { L } _ { v }$ . More details are shown in the supplementary material.

## B. Spatial-Temporal Autoregressive Diffusion

In this section, we present Spatial-Temporal Autoregressive Diffusion, a diffusion-based framework for motion generation. Given a text input and a 2D latent vector, Spatial-Temporal Autoregressive Diffusion uses an AdaLN [20] transformer network to predict the condition at the masked locations. Then the condition is set to the diffusion head for generating the predicted latent. There are three attentions performed in this transformer: spatial-temporal attention, spatial attention, and temporal attention.

Attention mechanism. For the spatial-temporal attention, CLIP [21] is employed to extract the text feature, resulting in a text latent $\mathbb { O } _ { \mathrm { t e x t } }$ . The text features are injected into the transformer via AdaLN [20], while the 2D latent vectors are directly augmented with 2D positional encodings [3].

![](images/962c5ed887e361e8c901f5ea5d636ec8055eeb922392b937927ea80d84fc0a41.jpg)  
Fig. 2: Overview of DeMoDiff. (A) Spatial-Temporal VAE encodes motion into 2D latent and decodes it for reconstruction. (B) Spatial-Temporal Autoregressive Diffusion includes Spatial-Temporal Attention, Spatial Attention and Temporal Attention and is trained via mask modeling, using a diffusion head to predict motion latent. (C) During inference, Spatial-Temporal Autoregressive Diffusion iteratively predicts the latent representations, which are decoded into the motion sequence.

After the position encoding P, we flatten the 2D latent vector to a 1D structure. We perform spatial-temporal attention on this sequence to preserve spatial-temporal relationships.

Spatial attention focuses solely on spatial dimensions by treating time frames as separate batches, which computes relationships between joints within each time batch.

Temporal attention is similar to spatial attention, focusing solely on temporal dimensions by treating joints as separate batches, which computes time-step relationships within each joint batch.

Since we utilize the bidirectional model, the three attention modules can be applied respectively; however, we choose the integration version in which the three modules are applied sequentially for better performance.

Masking and training strategy. Our masking strategy adopts the same mask ratio schedule following [3], [22]. This ratio is computed as:

$$
\gamma ( \tau ) = \cos ( \frac { \pi \tau } { 2 } ) .\tag{2}
$$

During training, τ is uniformly sampled from $( 0 , 1 ) \ ( \tau \ \sim$ ${ \bf U } ( 0 , 1 ) )$ to generate a mask ratio $\gamma ( \tau )$ . Temporal masking then randomly masks $\gamma ( \tau ) \times t \times j$ latent, and spatial masking randomly masks $\gamma ( \tau ) \times j$ latent for each individual frame, both following this ratio.

For enhanced disturbance in masked prediction, BERT’s remasking mechanism [17] is incorporated: A latent vector $z _ { 2 d }$ selected for masking is replaced by $\mathbb { O } _ { \mathrm { m a s k } }$ (80% chance), a random noise $\mathcal { N }$ (10% chance), or kept as is (10% chance).

During training, a random mask is applied on the latent sequence $\mathbf { Z } _ { 2 d }$ which is then processed by the transformer. After the transformer processing, we obtain the intermediate latent $\mathbf { C } = \{ c _ { 2 d } ^ { u m } \}$ from unmasked latent vectors and the text latent $\mathbb { O } _ { \mathrm { t e x t } } .$ which serve as the condition for the diffusion head (a small MLP) to predict masked latent $\hat { \bf Z } _ { 2 d } = \{ z _ { 2 d } ^ { m } \}$ Following [23], [24], the loss function is defined as:

$$
\mathcal { L } = \mathbb { E } _ { \epsilon , t } [ | | \epsilon - \epsilon _ { \theta } ( z _ { 2 d } ^ { m } | t , c _ { 2 d } ^ { u m } ) | | ^ { 2 } ] ,\tag{3}
$$

where t denotes the timestep of the noise schedule.

During inference, all latent vectors are initially set as mask latents. Text is first encoded via CLIP to obtain a semantic embedding, which is then fed into an AdaLN [20] modulation mechanism, which dynamically generates modulation parameters (scaling and shifting factors) for the AdaLN [20] transformer, thereby injecting text-conditioned signals into the transformer’s normalization layers to guide its processing. The transformer, under the guidance of these text-derived conditions, predicts a set of conditional latents. These latents are input to diffusion MLPs, which further predict the predicted latents. This newly generated latent set is then inserted back into the original sequence, and the aforementioned process is iteratively repeated until the complete latent sequence is predicted. Unlike previous works [9], [10], we utilize DDIM [25] to reduce the denoising step in our inference stage.

## IV. EXPERIMENTS

## A. Datasets.

We adopt two widely used text-to-motion datasets, i.e., HumanML3D [13] and KIT-ML [14], to evaluate our proposed DeMoDiff model. HumanML3D includes 14,616 motion sequences and 44,970 corresponding text descriptions, while KIT-ML contains 3,911 motions and 6,278 texts. Following prior approaches, HumanML3D uses 23,384/1,460/4,383 samples for training, validation, and testing, respectively; KIT-ML employs 4,888/300/830 samples for the same splits. The human motions are extracted into features with 263 dimensions follows the standard procedures established by previous research [4], [7], [13]. Detailed explanation of these metrics are provided in the supplementary material.

TABLE I: Evaluation on the HumanML3D dataset (upper half) and KIT-ML dataset (lower half). We repeat the evaluation for 20 times and report the average with 95% confidence interval. ↓ means the lower, the better, while ↑ represents the opposite. → indicates the closer to the real motion the better. We use bold to indicate the best result and underscore for the second best.
<table><tr><td>Datasets</td><td>Methods</td><td>FID ↓</td><td>Top1 ↑</td><td> $\mathrm { T o p } 2 \uparrow$ </td><td>Top3 ↑</td><td>MM-Dist↓</td><td>Diversity →</td></tr><tr><td rowspan="6">HumanML3D</td><td>Ground Truth</td><td> $0 . 0 0 2 ^ { \pm . 0 0 0 }$ </td><td> $0 . 5 1 1 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 0 3 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 9 7 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 7 4 ^ { \pm . 0 0 8 }$ </td><td> $9 . 5 0 3 ^ { \pm . 0 6 5 }$ </td></tr><tr><td>MDM [1]</td><td> $0 . 5 4 4 ^ { \pm . 0 4 4 }$ </td><td> $0 . 3 2 0 ^ { \pm . 0 0 5 }$ </td><td> $0 . 4 9 8 ^ { \pm . 0 0 4 }$ </td><td> $0 . 6 1 1 ^ { \pm . 0 0 7 }$ </td><td> $5 . 5 6 6 ^ { \pm . 0 2 7 }$ </td><td> $\mathbf { 9 . 5 5 9 ^ { \pm . 0 8 6 } }$ </td></tr><tr><td>MLD [2]</td><td> $0 . 4 7 3 ^ { \pm . 0 1 3 }$ </td><td> $0 . 4 8 1 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 7 3 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 7 2 ^ { \pm . 0 0 2 }$ </td><td> $3 . 1 9 6 ^ { \pm . 0 1 0 }$ </td><td> $9 . 7 2 4 ^ { \pm . 0 8 2 }$ </td></tr><tr><td>MotionDiffuse [11]</td><td> $0 . 6 3 0 ^ { \pm . 0 0 1 }$ </td><td> $0 . 4 9 1 ^ { \pm . 0 0 1 }$ </td><td> $0 . 6 8 1 ^ { \pm . 0 0 1 }$ </td><td> $0 . 7 8 2 ^ { \pm . 0 0 1 }$ </td><td> $3 . 1 1 3 ^ { \pm . 0 0 1 }$ </td><td> $9 . 4 1 0 ^ { \pm . 0 4 9 }$ </td></tr><tr><td>ReMoDiffuse [12]</td><td> $\mathbf { 0 . 1 0 3 ^ { \pm . 0 0 4 } }$ </td><td> $0 . 5 1 0 ^ { \pm . 0 0 5 }$ </td><td> $0 . 6 9 8 ^ { \pm . 0 0 6 }$ </td><td> $0 . 7 9 5 ^ { \pm . 0 0 4 }$ </td><td> $2 . 9 7 4 ^ { \pm . 0 1 6 }$ </td><td> $\overline { { 9 . 0 1 8 } } ^ { \pm . 0 7 5 }$ </td></tr><tr><td>DeMoDiff (Ours)</td><td> $0 . 1 6 4 ^ { \pm . 0 1 2 }$ </td><td> $\mathbf { 0 . 5 1 7 ^ { \pm . 0 0 3 } }$ </td><td> $\mathbf { 0 . 7 1 2 ^ { \pm . 0 0 4 } }$ </td><td> $\overline { { { \bf 0 . 8 0 8 } } } ^ { \pm . 0 0 2 }$ </td><td> $\mathbf { 2 . 9 0 7 ^ { \pm . 0 1 3 } }$ </td><td> $9 . 3 9 9 ^ { \pm . 0 9 6 }$ </td></tr><tr><td rowspan="6">KIT-ML</td><td>Ground Truth</td><td> $0 . 0 3 1 ^ { \pm . 0 0 4 }$ </td><td> $0 . 4 2 4 ^ { \pm . 0 0 5 }$ </td><td> $0 . 6 4 9 ^ { \pm . 0 0 6 }$ </td><td> $0 . 7 7 9 ^ { \pm . 0 0 6 }$ </td><td> $2 . 7 8 8 ^ { \pm . 0 1 2 }$ </td><td> $1 1 . 0 8 ^ { \pm . 0 9 7 }$ </td></tr><tr><td>MDM [1]</td><td> $0 . 4 9 7 ^ { \pm . 0 2 1 }$ </td><td> $0 . 1 6 4 ^ { \pm . 0 0 4 }$ </td><td> $0 . 2 9 1 ^ { \pm . 0 0 4 }$ </td><td> $0 . 3 9 6 ^ { \pm . 0 0 4 }$ </td><td> $9 . 1 9 1 ^ { \pm . 0 2 2 }$ </td><td> $1 0 . 8 5 ^ { \pm . 1 0 9 }$ </td></tr><tr><td>MLD [2]</td><td> $0 . 4 0 4 ^ { \pm . 0 2 7 }$ </td><td> $0 . 3 9 0 ^ { \pm . 0 0 8 }$ </td><td> $0 . 6 0 9 ^ { \pm . 0 0 8 }$ </td><td> $0 . 7 3 4 ^ { \pm . 0 0 7 }$ </td><td> $3 . 2 0 4 ^ { \pm . 0 2 7 }$ </td><td> $1 0 . 8 0 ^ { \pm . 1 1 7 }$ </td></tr><tr><td>MotionDiffuse [11]</td><td> $1 . 9 5 4 ^ { \pm . 0 6 2 }$ </td><td> $0 . 4 1 7 ^ { \pm . 0 0 4 }$ </td><td> $0 . 6 2 1 ^ { \pm . 0 0 4 }$ </td><td> $0 . 7 3 9 ^ { \pm . 0 0 4 }$ </td><td> $2 . 9 5 8 ^ { \pm . 0 0 5 }$ </td><td> $\mathbf { 1 1 . 1 0 ^ { \pm . 1 4 3 } }$ </td></tr><tr><td>ReMoDiffuse [12]</td><td> $\mathbf { 0 . 1 5 5 ^ { \pm . 0 0 6 } }$ </td><td> $0 . 4 2 7 ^ { \pm . 0 1 4 }$ </td><td> $0 . 6 4 1 ^ { \pm . 0 0 4 }$ </td><td> $0 . 7 6 5 ^ { \pm . 0 5 5 }$ </td><td> $\mathbf { 2 . 8 1 4 ^ { \pm . 0 1 2 } }$ </td><td> $1 0 . 8 0 ^ { \pm . 1 0 5 }$ </td></tr><tr><td>DeMoDiff (Ours)</td><td> $0 . 2 6 3 ^ { \pm . 0 1 7 }$ </td><td> $\overline { { { \bf 0 . 4 3 2 } } } ^ { \pm . 0 0 8 }$ </td><td> $\overline { { { \bf 0 . 6 6 0 } } } ^ { \pm . 0 0 7 }$ </td><td> $\overline { { { \bf 0 . 7 8 5 } } } ^ { \pm . 0 0 6 }$ </td><td> $2 . 8 2 7 ^ { \pm . 0 1 9 }$ </td><td> $1 0 . 8 6 ^ { \pm . 1 1 2 }$ </td></tr></table>

Text: A person walks forward, sits.  
![](images/be38bc6e92dfdf076dd609a7ab25ed98b40bc88438423112da7ee2636f7b4d4a.jpg)  
Text: A person who is running, stops, bends over and looks down while taking small steps, then resumes running

![](images/7275770d7d8be5ed7f6b2a9312b56660f04945dc19bfaa8970693432f77d6eb5.jpg)  
Fig. 3: Visual comparison of generated results. The color from light blue to dark blue indicates the motion sequential order.

TABLE II: Evaluation of motion compression methods on HumanML3D dataset. MPJPE and PAMPJPE are measured in millimeters. ACCL indicates acceleration error.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Generation</td><td colspan="3">Reconstruction</td></tr><tr><td>FID ↓</td><td>Diversity →</td><td>MPJPE ↓</td><td>PA-MPJPE↓</td><td>ACCL ↓</td></tr><tr><td>TM2T [6]</td><td>0.320</td><td>9.705</td><td>57.5</td><td>42.3</td><td>7.8</td></tr><tr><td>T2M-GPT [7]</td><td>0.119</td><td>9.705</td><td>58.0</td><td>41.1</td><td>7.0</td></tr><tr><td>MoMask [4]</td><td>0.019</td><td>9.662</td><td>29.5</td><td>20.3</td><td>5.3</td></tr><tr><td>MoGenTS [3]</td><td>0.005</td><td>9.547</td><td>16.4</td><td>8.0</td><td>3.0</td></tr><tr><td>VPoser-t [26]</td><td>1.430</td><td>8.336</td><td>75.6</td><td>48.6</td><td>9.3</td></tr><tr><td>ACTOR [27]</td><td>0.341</td><td>9.569</td><td>65.3</td><td>41.0</td><td>7.0</td></tr><tr><td>MLD [2]</td><td>0.017</td><td>9.554</td><td>14.7</td><td>8.9</td><td>5.1</td></tr><tr><td>DeMoDiff (Ours)</td><td>0.005</td><td>9.413</td><td>9.4</td><td>2.7</td><td>2.4</td></tr></table>

for HumanML3D and 251 dimensions for KIT-ML as in [13], encompassing global elements like root velocity, root height, and foot contact, as well as local joint information (22 joints for HumanML3D and 21 joints for KIT-ML).

## B. Evaluation Metrics.

We evaluate both Spatial-Temporal VAE and motion generation within our framework. The evaluation process strictly

We compare our model with previous diffusion-based textto-motion works, including the continuous regression-based methods. From the results in Tab. I, our method achieves performance that is highly competitive with previous stateof-the-art approaches on both the HumanML3D and KIT-ML datasets, reflecting the effectiveness of our design. The full comparison results and analysis are provided in supplementary material. DeMoDiff achieves better text-motion alignment, as evidenced by higher R-precision and lower MM-Dist. The relatively lower FID of ReMoDiffuse is mainly attributed to its retrieval module, whose retrieved motion segments may deviate from the distribution of dataset-native motions, increasing distributional mismatch despite improvements in semantic alignment. DeMoDiff consistently achieves comparable metrics, demonstrating its effectiveness in generating coherent and high-quality human motion. We also demonstrate

TABLE III: Ablation study of Spatial-Temporal VAE on the HumanML3D dataset.
<table><tr><td>Different VAE</td><td>FID ↓</td><td>Top1 ↑</td><td>Top2 ↑</td><td>Top3 ↑</td><td>MM-Dist ↓</td><td>Diversity →</td><td>MPJPE↓</td></tr><tr><td>1D Temporal (16 latent dim)</td><td> $0 . 0 1 4 4 ^ { \pm . 0 0 0 2 }$ </td><td> $0 . 5 0 7 1 ^ { \pm . 0 0 2 4 }$ </td><td> $0 . 6 9 8 4 ^ { \pm . 0 0 2 5 }$ </td><td> $0 . 7 9 3 6 ^ { \pm . 0 0 2 0 }$ </td><td> $3 . 0 0 9 2 ^ { \pm . 0 0 9 6 }$ </td><td> $9 . 4 8 3 9 ^ { \pm . 1 0 6 0 }$ </td><td> $2 8 . 0 ^ { \pm . 1 }$ </td></tr><tr><td>Ours (16 latent dim)</td><td> $0 . 0 0 5 4 ^ { \pm . 0 0 0 1 }$ </td><td> $\mathbf { 0 . 5 1 4 2 ^ { \pm . 0 0 2 6 } }$ </td><td> $0 . 7 0 1 5 ^ { \pm . 0 0 0 2 5 }$ </td><td> $\underline { { 0 . 7 9 8 3 } } ^ { \pm . 0 0 2 4 }$ </td><td> $\mathbf { 2 . 9 7 4 0 ^ { \pm . 0 0 8 5 } }$ </td><td> $9 . 5 4 0 7 ^ { \pm . 0 7 3 6 }$ </td><td> $9 . 4 ^ { \pm . 1 }$ </td></tr><tr><td>Ours (32 latent dim)</td><td> $0 . 0 0 4 8 ^ { \pm . 0 0 0 0 }$ </td><td> $0 . 5 1 2 6 ^ { \pm . 0 0 2 3 }$ </td><td> $\mathbf { 0 . 7 0 4 2 ^ { \pm . 0 0 1 8 } }$ </td><td> $\overline { { { \bf 0 . 7 9 8 8 } } } ^ { \pm . 0 0 1 9 }$ </td><td> $2 . 9 7 5 7 ^ { \pm . 0 0 7 3 }$ </td><td> $\mathbf { 9 . 5 0 8 1 ^ { \pm . 0 7 1 0 } }$ </td><td> $9 . 8 ^ { \pm . 1 }$ </td></tr><tr><td>Ours (64 latent dim)</td><td> $\mathbf { 0 . 0 0 4 1 ^ { \pm . 0 0 0 1 } }$ </td><td> $\overline { { 0 . 5 1 1 1 } } ^ { \pm . 0 0 2 0 }$ </td><td> $0 . 7 0 1 9 ^ { \pm . 0 0 1 8 }$ </td><td> $0 . 7 9 5 1 ^ { \pm . 0 0 1 6 }$ </td><td> $\overline { { 2 . 9 8 0 6 } } ^ { \pm . 0 0 6 7 }$ </td><td> $9 . 4 8 9 6 ^ { \pm . 0 7 1 6 }$ </td><td> $\mathbf { 7 . 7 ^ { \pm . 1 } }$ </td></tr></table>

![](images/2ddf42ac49f03bf3e772e5653d9db9847471f5cad23afb14f6e56d8343bfa297.jpg)

TABLE IV: Ablation study of the attention mechanisms choices on the HumanML3D dataset.  
![](images/50a48398bd8c02584e11bcee759255bf782d58746c09da9183baed8cf171d86e.jpg)  
MLD  
MoMask  
Fig. 4: Comparison of different motion compression methods. The dark blue represents ground-truth, the light blue represents predicted results ( Note that the global position of SMPL is the global position of motion. We have not performed any additional alignment processing ).

<table><tr><td>Methods</td><td>FID ↓</td><td>Top1 ↑</td><td> $\mathrm { T o p } 2 \uparrow$ </td><td>Top3 ↑</td><td></td><td>MM-Dist ↓ Diversity →</td></tr><tr><td>1D Temporal VAE + temporal attention</td><td> $0 . 6 6 5 ^ { \pm . 0 1 7 }$ </td><td> $0 . 4 5 2 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 3 3 ^ { \pm . 0 0 4 }$ </td><td> $0 . 7 3 5 ^ { \pm . 0 0 3 }$ </td><td> $3 . 3 6 1 ^ { \pm . 0 1 1 }$ </td><td> $9 . 1 1 4 ^ { \pm . 0 7 0 }$ </td></tr><tr><td>w/o all attention</td><td> $3 7 . 7 6 9 ^ { \pm . 1 2 2 }$ </td><td> $0 . 0 8 6 ^ { \pm . 0 0 3 }$ </td><td> $0 . 1 5 6 ^ { \pm . 0 0 4 }$ </td><td> $0 . 2 1 2 ^ { \pm . 0 0 3 }$ </td><td> $7 . 5 1 1 ^ { \pm . 0 1 1 }$ </td><td> $3 . 3 7 4 ^ { \pm . 0 8 3 }$ </td></tr><tr><td>w/o temporal&amp;spatial attention</td><td> $1 . 3 1 5 ^ { \pm . 0 3 4 }$ </td><td> $0 . 4 4 3 ^ { \pm . 0 0 7 }$ </td><td> $0 . 6 3 9 ^ { \pm . 0 0 7 }$ </td><td> $0 . 7 4 8 ^ { \pm . 0 0 6 }$ </td><td> $3 . 2 6 9 ^ { \pm . 0 1 3 }$ </td><td> $8 . 9 0 5 ^ { \pm . 1 4 7 }$ </td></tr><tr><td>w/o spatial attention</td><td> $0 . 7 8 3 ^ { \pm . 0 1 5 }$ </td><td> $0 . 4 6 3 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 6 1 ^ { \pm . 0 0 4 }$ </td><td> $0 . 7 6 5 ^ { \pm . 0 0 7 }$ </td><td> $3 . 2 1 2 ^ { \pm . 0 2 2 }$ </td><td> $9 . 2 3 2 ^ { \pm . 1 2 7 }$ </td></tr><tr><td>w/o temporal attention</td><td> $0 . 7 5 2 ^ { \pm . 0 2 6 }$ </td><td> $0 . 4 6 6 ^ { \pm . 0 0 5 }$ </td><td> $\overline { { 0 . 6 5 7 } } ^ { \pm . 0 0 4 }$ </td><td> $0 . 7 6 1 ^ { \pm . 0 0 4 }$ </td><td> $3 . 1 6 8 ^ { \pm . 0 1 8 }$ </td><td> $9 . 1 9 3 ^ { \pm . 0 9 5 }$ </td></tr><tr><td>DeMoDiff (Full Model)</td><td> $\overline { { { \bf 0 . 1 6 4 } } } ^ { \pm . 0 1 2 }$ </td><td> $\overline { { \mathbf { 0 . 5 1 7 } } } ^ { \pm . 0 0 3 }$ </td><td> $\mathbf { 0 . 7 1 2 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 0 . 8 0 8 ^ { \pm . 0 0 2 } }$ </td><td> $\overline { { { \bf 2 . 9 0 7 } } } ^ { \pm . 0 1 3 }$ </td><td> $\mathbf { 9 . 3 9 9 ^ { \pm . 0 9 6 } }$ </td></tr></table>

DeMoDiff (Ours)  
T2M-GPT

## C. Ablation Study.

visualization comparison results in Fig. 3. More visualization results of DeMoDiff are also provided in the supplementary material.

The core idea of our motion representation model is Spatial-Temporal VAE, which differs from previous quantizationbased methods. Therefore, we compare with both previous continuous methods and quantization-based methods for a fair comparison. As reported in Tab. II, the reconstruction accuracy of our method surpasses that of previous methods by a significant margin for the HumanML3D datasets. We also present visualization results with the SMPL body model in Fig. 4. It can be observed that our method outperforms other approaches in both motion similarity and global positional alignment. This reveals that the effect of our Spatial-Temporal VAE is much better than that of previous approaches.

To verify the efficacy of the proposed framework, we conducted ablation experiments on Spatial-Temporal VAE and Spatial-Temporal Autoregressive Diffusion. For the VAE, we first establish a baseline model incorporating a 1D Temporal motion VAE. Then, we change the VAE to our Spatial-Temporal VAE, examining its impact on both motion representation and motion generation capabilities. We also evaluate the influence of different VAE latent sizes. Regarding Spatial-Temporal Autoregressive Diffusion, we assessed the impact of various attention choices on human motion generation.

Spatial-Temporal VAE. As shown in Tab. III, we conducted a comparative analysis between our Spatial-Temporal VAE and the 1D Temporal VAE, alongside an investigation into the effects of varying VAE latent dimension sizes. Our experiments reveal that a larger latent dimension in the VAE leads to improvements in reconstruction metrics. This can be attributed to the fact that a higher-dimensional latent space provides a more abundant representational capacity for data distribution, enabling it to capture the detailed features of input data more precisely.

Spatial-Temporal Autoregressive Diffusion. We analyze the impact of different design choices of the attention mechanisms, as shown in Tab. IV. These attention mechanisms for Spatial-Temporal Autoregressive Diffusion were evaluated both in their individual applications and in sequential stacked configurations. We first removed all attention modules from the model, and then incrementally stacked the temporal-spatial attention, temporal attention, and spatial attention modules in a step-by-step manner until the complete model architecture was reconstructed. We also evaluate the metrics of 1D temporal VAE and temporal attention. Experimental results demonstrate that the model achieves a significant performance improvement when all three attention modules are integrated, which validates the effectiveness of the proposed model design.

## D. Temporal and Spatial Editing.

Since we utilize bidirectional attention in our framework, our method supports both temporal editing and spatial editing

Original Text: A person is walking on a circle. Edit Text: A person raises his hands.

![](images/6a891c9641fcaaa13ab435e4009adb111ef55ea30e9fa6424a17ecf6e0aa02b2.jpg)

![](images/e72746d38e3017cf727b5e5881214f08cb5fe477a6c85b62c2db87d0b80f5517.jpg)  
Temporal Editing

![](images/4a45d28bedcc27c24cd7d063972b93ac0443e800bb832fc9059124d4ad2ff3ba.jpg)  
Spatial Editing

Fig. 5: Visualization of editing results. The color from light blue to dark blue indicates the motion sequence order. The green color indicates the edited regions.

without requiring any editing-specific fine-tuning, which differs from previous diffusion-based works [1], [2], [9]–[12]. For temporal motion editing, we treat all latents corresponding to the temporal dimensions that need to be edited as mask latents and then generate motions following our standard generation procedure, which conditions on the unmasked latents and the editing textual instructions. For spatial editing, similar to the temporal editing step, we only treat the latents of the spatial dimension to be edited as mask latents. We demonstrate the visualization results in Fig. 5

## V. CONCLUSION

We present DeMoDiff as a novel framework for text-tomotion generation that integrates a spatiotemporally decoupled autoregressive diffusion model to directly predict temporal and spatial motion latents. By introducing the spatial-temporal VAE and the spatial-temporal autoregressive diffusion model, DeMoDiff enables sampling in both temporal and spatial latent spaces, with the ability for temporal and spatial editing. Our method demonstrates its competitiveness in motion generation while providing greater flexibility.

## REFERENCES

[1] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit Haim Bermano, “Human motion diffusion model,” in ICLR, 2023.

[2] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu, “Executing your commands via motion diffusion in latent space,” in CVPR, 2023.

[3] Weihao Yuan, Weichao Shen, Yisheng HE, Yuan Dong, Xiaodong Gu, Zilong Dong, Liefeng Bo, and Qixing Huang, “Mogents: Motion generation based on spatial-temporal joint modeling,” in NeurIPS, 2024.

[4] Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng, “Momask: Generative masked modeling of 3d human motions,” in CVPR, 2024.

[5] Sai Shashank Kalakonda, Shubh Maheshwari, and Ravi Kiran Sarvadevabhatla, “Action-gpt: Leveraging large-scale language models for improved and generalized action generation,” in ICME, 2023.

[6] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng, “Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts,” in ECCV, 2022.

[7] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Shan Ying, “Generating human motion from textual descriptions with discrete representations,” in CVPR, 2023.

[8] Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, and Wanli Ouyang, “Motiongpt: Finetuned llms are general-purpose motion generators,” in AAAI, 2024.

[9] Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang, “Rethinking diffusion for text-driven human motion generation: Redundant representations, evaluation, and masked autoregression,” in CVPR, 2025.

[10] Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, and Jingbo Wang, “Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space,” in ICCV, 2025.

[11] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu, “Motiondiffuse: Text-driven human motion generation with diffusion model,” TPAMI, 2024.

[12] Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu, “Remodiffuse: Retrievalaugmented motion diffusion model,” in ICCV, 2023.

[13] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng, “Generating diverse and natural 3d human motions from text,” in CVPR, 2022.

[14] Matthias Plappert, Christian Mandery, and Tamim Asfour, “The KIT motion-language dataset,” Big Data, 2016.

[15] Mathis Petrovich, Michael J Black, and Gul Varol, “Temos: Generating¨ diverse human motions from textual descriptions,” in ECCV, 2022.

[16] Mathis Petrovich, Michael J Black, and Gul Varol, “Tmr: Text-to-motion¨ retrieval using contrastive 3d human motion synthesis,” in ICCV, 2023.

[17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in NAACL, 2019.

[18] Qing Yu, Mikihiro Tanaka, and Kent Fujiwara, “Exploring vision transformers for 3d human motion-language models with motion patches,” in CVPR, 2024.

[19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[20] Yunhui Guo, Chaofeng Wang, Stella X Yu, Frank McKenna, and Kincho H Law, “Adaln: a vision transformer for multidomain learning and predisaster building information extraction from images,” Journal of Computing in Civil Engineering, 2022.

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021.

[22] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman, “Maskgit: Masked generative image transformer,” in CVPR, 2022.

[23] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He, “Autoregressive image generation without vector quantization,” in NeurIPS, 2024.

[24] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[25] Jiaming Song, Chenlin Meng, and Stefano Ermon, “Denoising diffusion implicit models,” in ICLR, 2021.

[26] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black, “Expressive body capture: 3d hands, face, and body from a single image,” in CVPR, 2019.

[27] Mathis Petrovich, Michael J. Black, and Gul Varol, “Action-conditioned¨ 3d human motion synthesis with transformer VAE,” in ICCV, 2021.