# MULTI-PATH MAMBA DIFFUSION FOR 3D POINT CLOUD SEMANTIC SCENE COMPLETION

Li Liang<sup>⋆</sup> Naveed Akhtar<sup>†</sup> Jordan Vice<sup>⋆</sup> Anwaar Ulhaq <sup>‡</sup> Ajmal Saeed Mian<sup>⋆</sup>

<sup>⋆</sup> The University of Western Australia, <sup>†</sup> The University of Melbourne, <sup>‡</sup> Central Queensland University

## ABSTRACT

3D semantic scene completion plays a critical role in enabling various downstream tasks within autonomous systems by estimating both missing geometric structures and semantic information in scene data. In this work, we introduce a unique neural framework that integrates state space modeling with diffusion-based generative approaches to enhance 3D point cloud semantic scene completion. We propose MPath Mamba diffusion network, leveraging a skip-triple Mamba structure, dimensional decomposition residuals, and multi-directional dilations within multi-path data to efficiently process longsequence data for feature extraction. The proposed method operates in the conditioned latent space of a variational autoencoder, enhanced by a state-space mechanism to improve 3D point cloud representation and reconstruction. Extensive experiments conducted on the SemanticKITTI benchmark demonstrate that our approach achieves highly competitive performance for 3D point cloud semantic scene completion.

Index Terms— 3D point cloud, 3D semantic scene completion, diffusion models, state space modeling.

## 1. INTRODUCTION

3D scene understanding is a cornerstone of autonomous driving, combining geometric and semantic analysis to facilitate reliable perception and informed decision-making. LiDAR sensors play an essential role in this domain, providing highprecision, long-range measurements that surpass the capabilities of cameras in various scenarios. However, achieving accurate semantic scene completion from 3D point clouds in large-scale outdoor environments continues to pose a significant challenge. This difficulty arises from the inherent sparsity of LiDAR data and the complexity of real-world scenes containing occlusions, which hinder the development of robust and comprehensive solutions for semantic understanding.

Recent studies [1, 2, 3, 4] have significantly advanced the precision and efficiency of 3D point cloud semantic scene completion. For instance, JS3C-Net [2] integrates a segmentation network with a scene completion module. However, its dependence on initial segmentation introduces a potential for error propagation, which can adversely impact overall performance. S3CNet [3] addresses this by employing sparse convolutions and a multi-view fusion strategy to enhance semantic scene completion. Similarly, SCPNet [4] improves single-frame representations through dense-to-sparse knowledge distillation. However, SCPNet’s two-stage framework—comprising scene completion followed by segmentation, substantially increases the model size, a challenge further exacerbated by the absence of effective downsampling mechanisms in the technique.

This study tackles this challenge by utilizing advancements in generative diffusion modeling [5] and state-space models [6]. The proposed neural network presented in Fig. 1 operates within the conditioned latent space of a Variational Autoencoder (VAE). The proposed Multi-Path (MPath) Mamba Diffusion model, effectively captures rich contextual information by employing triple-directional feature dependencies with varying dilation rates within a multi-path data structure. By incorporating a combination of multi-scale fusion and feature extraction blocks with an efficient downsampling strategy, the proposed approach significantly improves computational efficiency while simultaneously improving 3D semantic scene completion performance. Through this work, we contribute:

• A computationally efficient neural network design, integrating state-space and generative diffusion structures within the latent space of a VAE for 3D point cloud semantic scene completion.

• The MPath Mamba diffusion model, utilizing skiptriple Mamba layers with varied dilation rates within a multi-path structure to enhance 3D scene completion.

• A comprehensive evaluation of our technique on the SemanticKITTI dataset demonstrates that MPath Mamba achieves highly competitive performance for 3D point cloud semantic scene completion.

## 2. RELATED WORK

This work contributes to 3D point cloud semantic scene completion by proposing a diffusion-based network integrated with state-space modeling. We provide a review of related work in three key domains: diffusion models, state-space models, and 3D point cloud semantic scene completion.

![](images/b98c788c3c0d2191c8015ee1c71059d6db225b86a8f29b297fe66a0e454cdef1.jpg)  
Fig. 1: Schematics of the proposed approach. It consists of two main components: (A) 3D scene completion network and (B) a 3D semantic segmentation network. The scene completion network operates within the latent space of a Variational Autoencoder (VAE), incorporating two conditioning sub-networks, Multi-Scale Fusion Blocks (MSFusion), and MPath Mamba diffusion denoising network. A modular segmentation network performs 3D semantic segmentation.

Diffusion Models. Diffusion models [7] work on the principle of learning a sampler for the original data distribution. In the 3D domain, they have become increasingly prominent in generative tasks, such as 3D shape generation [8] and 3D occupancy prediction [9]. Despite their effectiveness, these models often require high computational costs due to frequent function and gradient evaluations [10]. Latent diffusion models [5] address this challenge by performing the diffusion process within the latent space of an autoencoder, thereby significantly reducing the computational overhead. While much of the existing 3D research has centered on object-level generation, Sparseocc [9] expanded this to scene synthesis by introducing a 3D sparse diffusion model that utilizes decomposed sparse kernels and a restructured transformer head.

State-Space Models. Initially developed within the field of control theory [11], state-space models have proven effective in capturing long-range dependencies and have recently emerged as compelling alternatives to CNNs and Transformers in natural language processing and computer vision. In this context, S4 [12] utilizes diagonal parameter structures for efficient sequence modeling, which has inspired subsequent advancements such as HTTYH [13] and S4D [14], both of which strike a balance between computational efficiency and high performance. Building upon these foundations, S5 [15] improves efficiency through parallel scanning. These developments have been further extended through the Mamba framework [6], which has been successfully applied to various tasks, including point cloud processing [16]. It is noteworthy that Liu et al. [16] proposed a point cloud backbone based on a state space model with an octree-based ordering strategy, which is an entirely different contribution than our proposal of MPath Mamba for scene completion.

3D Point Cloud Semantic Scene Completion. 3D semantic scene completion methods can be categorized into imagebased [1], and point cloud-based [17], with most relying on CNNs or transformers as their backbone. Rist et al. [17] proposed Deep Implicit Functions (DIFs) using a 2D multiresolution grid to represent the geometry and semantics of 3D scenes, while SCPNet [4] leverages dense knowledge distillation and label rectification to improve single-frame representations. CNN-based methods are limited by their constrained receptive fields, whereas transformers extend receptive fields but are memory-intensive. The Mamba framework [6] addresses these challenges by expanding the receptive field with minimal memory overhead, making it particularly suitable for 3D point cloud semantic scene completion tasks. This is one of the key inspirations behind our unique technique.

## 3. METHOD

The objective of 3D Point Cloud Semantic Scene Completion is to simultaneously infer the geometric structure and semantic information of a scene from sparse and incomplete point cloud data. The input point cloud, denoted as $P \in \mathbb { R } ^ { N \times 3 } ;$ , is discretized into a 3D voxel grid with dimensions $L \times W \times H ,$ corresponding to the spatial extent of the scene. Each voxel is assigned a semantic label $c \in { 0 , 1 , . . . , C - 1 }$ , indicating whether it is empty or occupied by an object belonging to one of C the predefined categories.

## 3.1. Proposed Network

Figure 1 illustrates the proposed approach, centered on the Multi-Path Mamba (MPath Mamba) Diffusion Denoising

Network. Built on a Variational Autoencoder (VAE), the framework incorporates two conditioning networks to generate compact voxel representations, reducing computational complexity while maintaining performance. A Multi-Scale Fusion (MSFusion) Block improves contextual understanding through a combination of global and local convolutional kernels and conditional features, while the MPath Mamba network integrates downsampling, multi-scale feature extraction (MSFeature), MPath Mamba block, MLP, and upsampling to achieve its objectives.

## 3.2. 3D Scene Completion Network

The 3D scene completion network forms the core of our overall approach. This network is marked as Stage A in Fig. 1 to delineate it from the segmentation network, which is Stage B. The scene completion and semantic segmentation tasks are modular in our technique.

As depicted in Fig. 1, the proposed MSFusion and MPath Mamba Diffusion Denoising Network are designed to operate within the latent space of a VAE, utilizing both the VAE and conditioning networks to enhance efficiency and performance. Inspired by [18], the VAE is trained using a combination of cross-entropy and Lovasz-Softmax losses to reduce the blur from voxelization. For a voxelized scene v $\in L \times W \times H$ the encoder E maps v to a latent representation $z = \mathbb { E } ( v )$ , and the decoder D reconstructs it as $\tilde { v } = \mathbb { D } ( z )$ . The latent space $z ~ \in ~ \mathbb { R } ^ { l \times w \times h \times c }$ is compressed by a factor of $f = 4 .$ . This 3D latent space enables efficient compression and reconstruction for the diffusion model. The conditioning networks are aligned with the encoder’s architecture for seamless integration with the VAE latent space.

The MSFusion block efficiently extracts multi-scale features by substituting large convolutional kernels with sequential $3 \times 3 \times 3$ convolutions, drawing inspiration from SCP-Net [4] and VGG [19]. Two layers employ $\mathrm { ~ a ~ 5 ~ } \times \mathrm { ~ 5 ~ } \times \mathrm { ~ 5 ~ }$ kernel, and three layers leverage a $7 \times 7 \times 7$ kernel, reducing computational cost while maintaining capacity. For a feature map with C channels, the computational cost of a $5 \times 5 \times 5$ convolution is reduced from $1 2 5 C ^ { 2 } : 0 5 4 C ^ { 2 }$ , and for a $7 \times 7 \times 7$ kernel, from $3 4 3 C ^ { 2 }$ to $8 1 C ^ { 2 }$ . This design efficiently captures local and global contexts, balancing performance and computational efficiency for large-scale 3D scene understanding.

The down-sampling block in the MPath Mamba network is composed of three $3 \times 3 \times 3$ convolutional layers, each incorporating instance normalization. LeakyReLU activation is applied to the first and last layers to optimize feature extraction. A residual connection bypasses the first two layers and merges with the third, ensuring effective gradient flow and representation learning. The MLP adopts the same structure but without downsampling, supporting residue transfer between encoding and decoding. The upsampling block employs transposed convolution for resolution enhancement, followed by the downsampling block’s structure for feature re-

![](images/1b5d1709fd247606d278acb46ed34692c86b6467968298a591a068cfcd7d67a4.jpg)  
Fig. 2: MPath Mamba flow. Data is divided into multiple paths based on dilation rates, with each path providing three directional components as input to MPath Mamba.

construction.

The MSFeature block enhances signal processing within the Mamba framework by capturing spatial relationships in 3D features before flattening them into a 1D sequence for subsequent processing. It employs Dimensional Decomposition Residual (DDR) blocks [20] with varying dilation rates to effectively capture scene layout information. The DDR block improves the efficiency of 3D CNN computations by decomposing convolution operations into $1 \times 1 \times k , 1 \times k \times 1$ and $k \times 1 \times 1$ layers. This reduces computational cost from $C ^ { i n } \times C ^ { o u t } \times k ^ { \mathrm { 3 } }$ to $C ^ { i n } \times C ^ { o u t } \times 3 k$ , cutting parameters by approximately one-third compared to standard $3 \times 3 \times 3$ kernels while preserving essential spatial details.

The proposed MPath Mamba Block is a key component of the MPath Mamba Denoiser, utilizing a Triple Mamba configuration inspired by [26] to capture both direct and indirect feature connections, enhancing contextual information. It incorporates MPath Skip Triple Mamba (MSTM) layers with varying dilation rates across forward, reverse, and random directions within a multi-path structure. As shown in Fig. 2, an example with a dilation rate of 1 demonstrates how each MSTM layer calculates feature dependencies; the same procedure is applicable when dilation rates of 0 and 3 are employed. effectively capturing comprehensive feature relationships. The output of the $i ^ { \mathrm { { t h } } }$ MSTM layer is defined as:

$$
\Omega _ { i } = \Omega _ { i } ^ { f } ( z _ { d } ) + \Omega _ { i } ^ { r } ( z _ { d } ) + \Omega _ { i } ^ { s } ( z _ { d } ) ,\tag{1}
$$

$$
\Omega _ { M P a t h } = \sum _ { i } \Omega _ { i } , i \in \left\{ 0 , 1 , 3 \right\} ,\tag{2}
$$

where $z _ { d }$ represents the input multi-path features, while $f , \ r ,$ and s denote the forward, reverse, and random directions, respectively. The block’s final output, $\Omega _ { f i n a l } ,$ is computed as $\Omega _ { f i n a l } \ = \ \mathrm { M L P } ( \mathrm { L N } ( \Theta _ { i n } ) ) + \Theta _ { i n }$ , where $\Theta _ { i n } = \Omega _ { M P a t h } ( \mathrm { L N } ( \phi _ { \mathrm { i n i t i a l } } ) ) + \phi _ { \mathrm { i n i t i a l } }$ . Here, ϕ<sub>initial</sub> represents the input features of the MPath Mamba block, and MLP and LN refer to Multi-Layer Perceptrons and Layer Normalization. By utilizing varying dilation rates across the three directions within multi-path data, the block captures diverse feature dependencies, enhancing connectivity and improving performance in 3D semantic scene completion.

Table 1: Quantitative results on the SemanticKITTI [21] test set are from the online server. ”Param.” represents parameters in millions, and bold indicates the best results in each column.
<table><tr><td></td><td></td><td></td><td></td><td colspan="10">mottcoycle othtre-eh. biceycle (39,2%) (%00) (%£00) (%91 0) (%00%)</td><td>sidwalk parlking</td><td></td><td>ohet-nd.</td><td>buidig</td><td></td><td>veton</td><td></td><td></td><td></td><td>tua-isn</td></tr><tr><td>Method</td><td>IoU</td><td>mIoU</td><td>Param.</td><td>car.</td><td></td><td></td><td>tuck ■</td><td></td><td>person (0L%)</td><td>bicycit</td><td>motcrcist (0%%)</td><td>(%500) oad</td><td>1530%)</td><td>(712%)</td><td>(11°3%)</td><td>(0.6%)</td><td>(1%I)</td><td>fence (300%)</td><td>(333%)</td><td>unk (0%15&#x27;%)</td><td>terain (9%1%)</td><td>01</td><td>(%80&#x27;0)</td></tr><tr><td>LMSCNet [22]</td><td>56.7</td><td>17.6</td><td>0.4M</td><td>30.9</td><td>0.0</td><td>0.0</td><td>1.5</td><td>0.8</td><td>0.0</td><td>0.0</td><td>0.0</td><td>64.8</td><td>29.0</td><td>34.7</td><td>4.6</td><td>38.1</td><td></td><td>21.3</td><td>41.3</td><td>19.9</td><td>32.1</td><td>15.0</td><td>0.8</td></tr><tr><td>UDNet [23]</td><td>59.4</td><td>19.5</td><td></td><td>33.9</td><td>0.8</td><td>0.4</td><td>3.8</td><td>4.4</td><td>0.5</td><td>0.3</td><td></td><td>0.3</td><td>62.0</td><td>28.2</td><td>35.1</td><td>9.1</td><td>39.5</td><td>24.4</td><td>40.9</td><td>23.2</td><td>32.3</td><td>18.8</td><td>13.1</td></tr><tr><td>Local-DIFs [17]</td><td>57.7</td><td>22.7</td><td></td><td>34.8</td><td>3.6</td><td>2.4</td><td>4.4</td><td>4.8</td><td>2.5</td><td>1.1</td><td>0.0</td><td>67.9</td><td>40.1</td><td>42.9</td><td></td><td>11.4</td><td>40.4</td><td>29.0</td><td>42.2</td><td>26.5</td><td>39.1</td><td>21.3</td><td>17.5</td></tr><tr><td>SSA-SC [24]</td><td>58.8</td><td>23.5</td><td>41.0M</td><td>36.5</td><td>13.9</td><td>4.6</td><td>5.7</td><td>7.4</td><td>4.4</td><td>2.6</td><td>0.7</td><td>72.2</td><td>37.4</td><td>43.7</td><td></td><td>10.9</td><td>43.6</td><td>30.7</td><td>43.5</td><td>25.6</td><td>41.8</td><td>14.5</td><td>6.9</td></tr><tr><td>JS3C-Net [2]</td><td>56.6</td><td>23.8</td><td>3.1M</td><td>33.3</td><td>14.4</td><td>8.8</td><td>7.2</td><td>12.7</td><td>8.0</td><td>5.1</td><td>0.4</td><td>64.7</td><td>34.9</td><td>39.9</td><td></td><td>14.1</td><td>39.4</td><td>30.4</td><td>43.1</td><td>19.6</td><td>40.5</td><td>18.9</td><td>15.9</td></tr><tr><td>SSC-RS [25]</td><td>59.7</td><td>24.2</td><td>23.0M</td><td>36.4</td><td>10.1</td><td>5.1</td><td>5.3</td><td>11.2</td><td>4.7</td><td>2.4</td><td>0.9</td><td>73.1</td><td>38.6</td><td>44.4</td><td>17.4</td><td></td><td>44.6</td><td>30.8</td><td>44.1</td><td>26.0</td><td>41.9</td><td>15.0</td><td>7.2</td></tr><tr><td>S3CNet [3]</td><td>45.6</td><td>29.5</td><td></td><td>31.2</td><td>41.5</td><td>45.0</td><td>6.7</td><td>16.1</td><td>45.9</td><td>35.8</td><td>16.0</td><td>42.0</td><td>17.0</td><td>22.5</td><td>7.9</td><td></td><td>52.2</td><td>31.3</td><td>39.5</td><td>34.0</td><td>21.2</td><td>31.0</td><td>24.3</td></tr><tr><td>Ours</td><td>55.6</td><td>29.0</td><td>29.4M</td><td>40.1</td><td>22.5</td><td>19.8</td><td>15.5</td><td>20.4</td><td>14.2</td><td>12.9</td><td>0.5</td><td>70.9</td><td>45.7</td><td>45.5</td><td>25.2</td><td></td><td>36.4</td><td>35.5</td><td>43.1</td><td>33.7</td><td>45.2</td><td>19.6</td><td>4.4</td></tr></table>

## 3.3. 3D Semantic Segmentation Network

The 3D Semantic Segmentation Network is designed to process and segment the outputs from the 3D completion network. Its architecture incorporates multi-path blocks inspired by SCPNet [4] and DDR blocks [20], enabling efficient capture of both local and global contextual features. The network is designed to process and segment the outputs from the 3D completion network. Its architecture incorporates multipath blocks inspired by SCPNet [4] and DDR blocks [20], enabling efficient capture of both local and global contextual features. In addition, feature extractor-(1) employs multi-path blocks [4] to extract robust 3D spatial and contextual information, while feature extractor-(2) utilizes a 3D feature extraction network that incorporates multi-path blocks [4] and DDR blocks [20], along with a 2D image feature extraction network [24]. This combined approach integrates 3D spatial features and 2D spatial features to provide more robust contextual representations.

## 3.4. Training Objective

The training objective, designed for the complexity of the task, uses a composite loss function. The MPath Mamba denoising network follows the diffusion denoising framework, minimizing the expected squared error between the predicted output and target data, expressed as:

$$
L _ { D M } = \mathbb { E } _ { x , \epsilon \sim N ( 0 , 1 ) , t } \left[ \left\| \epsilon - \epsilon _ { \theta } \left( x _ { t } , t \right) \right\| _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where $\epsilon _ { \theta } \left( x _ { t } , t \right)$ represents an equally weighted sequence of denoising autoencoders, with $t \in \{ 1 , 2 , . . , T \}$ . The model is trained to predict the denoised version of the input $x _ { t }$ , which corresponds to the noisy version of the original input x at time step t. For 3D semantic segmentation, the objective function is defined as:

$$
\mathcal { L } = \mathcal { L } _ { C E } + \beta \mathcal { L } _ { L o v a s z } ,\tag{4}
$$

where the balancing coefficient $\beta ,$ which determines the relative contributions of different loss components, is set to 1

in the proposed method. The cross-entropy loss $\mathcal { L } _ { C E }$ and Lovasz-Softmax loss $\mathcal { L } _ { L o v a s z }$ are defined as:

$$
\mathcal { L } _ { C E } = - \sum _ { i } y _ { i } \log ( \hat { y _ { i } } ) , \mathcal { L } _ { L o v a s z } = \frac { 1 } { | C | } \sum _ { c = 1 } ^ { C } J \left( e ( c ) \right) ,\tag{5}
$$

where $\hat { y } _ { i }$ and $y _ { i }$ represent the predicted output and ground truth for the $i ^ { \mathrm { { t h } } }$ element, J is the Lovasz extension of IoU, and $e ( c )$ is the error vector for class $C$ in the set of classes. The VAE objective combines $\mathcal { L } _ { C E } , \mathcal { L } _ { L o v a s z }$ , and the Kullback-Leibler (KL) divergence to ensure effective reconstruction and regularization.

## 4. EXPERIMENTS

The proposed network is evaluated on the SemanticKITTI dataset [27], a widely recognized benchmark for 3D semantic scene completion. This is the only outdoor dataset used to evaluate existing 3D point cloud semantic scene completion methods. Ablation studies are also performed to assess the contributions of each component of our framework.

Datasets. The SemanticKITTI benchmark [27] provides densely annotated urban driving scenes. The dataset includes voxelized scenes within a $5 1 . 2 m \times 5 1 . 2 m \times 6$ .4m volume, represented as $2 5 6 \times 2 5 6 \times 3 2$ voxel grids with a resolution of 0.2 m. It comprises 10 training sequences, 1 validation sequence, and 11 testing sequences, covering 20 classes.

Evaluation Metrics. The proposed framework is evaluated using standard metrics for semantic scene completion [17], focusing on both geometric reconstruction and semantic segmentation accuracy. Geometric performance is assessed using Intersection over Union (IoU), defined as $\begin{array} { r } { I o U _ { i } = \frac { T \breve { P } _ { i } } { T P _ { i } + F N _ { i } + F P _ { i } } } \end{array}$ , where $T P _ { i } , F P _ { i }$ , and $F N _ { i }$ denote the true positives, false positives, and false negatives for the $i ^ { \mathrm { { t h } } }$ class, respectively. Semantic segmentation accuracy is evaluated across all classes using Mean Intersection over Union (mIoU), calculated as $\begin{array} { r } { m I o \mathbf { \bar { U } } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } I o U _ { c } , } \end{array}$ , where C represents the total number of classes.

![](images/3d47b5fe1673fcb4d09b8ac56113773c4f6c711d2f3cbe93b7a1010cebc6501c.jpg)  
Fig. 3: Qualitative results on the SemanticKITTI validation set. Referencesoutputs of our method, LMSCNet, SSA-SC, and SSC-RS.

## els. arXiv pr4.1. Results

The proposed method is compared against several state-ofthe-art approaches in semantic scene completion, including LMSCNet [22], UDNet [23], Local-DIFs [17], SSA-SC [24], JS3C-Net [2], SSC-RS [25], and S3CNet [3].

Table 1 shows the Semantic-KITTI test results for 3D point cloud semantic scene completion. The proposed method <sub>324–333.</sub>achieves competitive performance, surpassing most state-ofthe-art methods for mIoU, except S3CNet [3]. Our method performs comparably to S3CNet in segmentation while out-<sub>ence: CVPR</sub>performing it considerably in completion. We note that another method, SCPNet [4], also reports highly competitive results. However, that method also entails a very high <sub>Recurrent U-Net. arXiv preprint arXiv:2403.08885.</sub>computational cost that far exceeds our NVIDIA 4090 GPU resources. Our method ensures robust performance across diverse object scales, excelling in both large-scale (e.g., park-<sub>priors. In The IEEE/CVF Conference: WACV, 3781–3790.</sub>ing) and small-scale objects (e.g., cars, trucks) while keeping the compute requirements reasonable.

onstruction: A Survey of deep learning methods. Elsevier:<sup>We also utilize the pre-trained models and official im-</sup> Computers & Graphics, 94: 164–190.plementations of LMSCNet [22], SSA-SC [24], and SSC-Geiger, A.; Lenz, P.; and Urtasun, R. 2012. Are we ready forRS [25] for qualitative comparisons on the SemanticKITTI autonomous driving? the kitti vision benchmark suite. In Thevalidation set. As illustrated in Fig. 3, the proposed method IEEE/CVF Conference: CVPR, 3354–3361. IEEE.demonstrates excellent performance in segmenting planar cat-<sub>Glasser, W. 1985. Control theory. Harper and Row New</sub>egories (e.g., roads, terrain) and consistently provides accu-<sub>York.</sub>rate scene completions, effectively avoiding the distorted reconstructions observed in the compared methods.

## 4.2. Ablation Study

parameterization and initialization of diagonal state spaceSystematic experiments are conducted to evalu te he impact models. NeurIPS, 35: 35971–35983.of various components within the proposed framework, as Gu, A.; Goel, K.; and Ré, C. 2021. Efficiently modelingdetailed in Table 2. Performance variations were analyzed long sequences with structured state spaces. arXiv preprintby removing spec fic blo ks from the framework. The abarXiv:2111.00396.lation study underscores the significance of MSFusion and Gu, A.; Johnson, I.; Timalsina, A.; Rudra, A.; and Ré, C.MPath Mamba i improving the performance of 3D semantic 2022b. How to train your hippo: State space models withscene completi n. The results demonstrate that the integrageneralized orthogonal basis projections. arXiv preprinttion of components is crucial for capturing complex spatial <sup>arXiv:2206.12037.</sup>and semantic relationships. The configuration without MPath

Columns from the left represent input data, ground truth, and Table 2: Ablation study on the SemanticKITTI validation set for 3D semantic Scene Completion (IoU and mIoU). w/o <MPath Mamba> indicates the use of <Triple Mamba>.
<table><tr><td>Method</td><td>3D Semantic Scene Completion IoU mIoU</td></tr><tr><td>w/o MSFusion</td><td>46.5 26.2</td></tr><tr><td>w/o MSFeature</td><td>48.7 28.4</td></tr><tr><td>w/o MPath Mamba</td><td>49.4 27.8</td></tr><tr><td>Full Model</td><td>51.2 29.1</td></tr></table>

Mamba denotes the use of a Triple Mamba block.

## 5. CONCLUSION

<sup>I.</sup> <sup>2019a.</sup> <sup>Rgbd</sup> <sup>based</sup> <sup>dimensional</sup> <sup>decomposition</sup> <sup>residu</sup>In this paper, we propose a multi-path Mamba denoising diffusion network for efficient and robust 3D point cloud semantic scene completion. The proposed MPath Mamba network combines variational autoencoders with dual conditioning to generate compact latent representations, ensuring efficiency and performance. For this purpose, it employs skip triple dilations to capture direct and indirect feature relationships within multi-path data, resulting in the enhancement of spatial and semantic representations. Evaluations on the SemanticKITTI dataset validate the effectiveness and robustness of the approach.

## C.; Li, Y.; Zhao, X.; and Gao, Y. 6. ACKNOWLEDGMENTS

pletion. In AAAI, volume 34, 11402–11409.Dr. Naveed Akhtar holds an ARC Discovery Early Career Re-Li, W.; Hong, X.; and Fan, X. 2024. Spikemba: Multi-modsearcher Award (DE230101058), and Professor Ajma Saee spiking saliency mamba for temporal video grounding. arXMian hold an ARC Future Fellowship (FT210100268), both preprint arXiv:2404.01174.funded by the Australian Government.

## <sub>3d semantic sc</sub>References

Li, Y.; Yu, Z.; Choy, C.; Xiao, C.; Alvarez, J. M.; Fidler, <sub>[1] Shuran Song, Fisher Yu, Andy Zeng, Angel X Chang,</sub> ng, C.; and Anandkumar, A. 2023c. Voxformer: SparManolis Savva, and Thomas Funkhouser, “Semantic xel transformer for camera-based 3d semantic scene coscene completion from a single epth image,” in CVPR, tion. In The IEEE/CV2017, pp. 1746–1754.

[2] Xu Yan, Jiantao Gao, Jie Li, Ruimao Zhang, Zhen Li, Rui Huang, and Shuguang Cui, “Sparse single sweep lidar point cloud segmentation via learning contextual shape priors from scene completion,” in AAAI, 2021, vol. 35, pp. 3101–3109.

[3] Ran Cheng, Christopher Agia, Yuan Ren, Xinhai Li, and Liu Bingbing, “S3cnet: A sparse semantic scene completion network for lidar point clouds,” in CoRL, 2021, pp. 2148–2161.

[4] Zhaoyang Xia, Youquan Liu, Xin Li, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, and Yu Qiao, “Scpnet: Semantic scene completion on point cloud,” in CVPR, 2023, pp. 17642–17651.

[5] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer, “High-resolution im-¨ age synthesis with latent diffusion models,” in CVPR, 2022, pp. 10684–10695.

[6] Albert Gu and Tri Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[7] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020.

[8] Jaehyeok Shim, Changwoo Kang, and Kyungdon Joo, “Diffusion-based signed distance fields for 3d shape generation,” in CVPR, 2023, pp. 20887–20897.

[9] Pin Tang, Zhongdao Wang, Guoqing Wang, Jilai Zheng, Xiangxuan Ren, Bailan Feng, and Chao Ma, “Sparseocc: Rethinking sparse latent representation for vision-based semantic occupancy prediction,” in CVPR, 2024, pp. 15035–15044.

[10] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean, “Carbon emissions and large neural network training,” arXiv preprint arXiv:2104.10350, 2021.

[11] William Glasser, Control theory, Harper and Row New York, 1985.

[12] Albert Gu, Karan Goel, and Christopher Re, “Efficiently´ modeling long sequences with structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.

[13] Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, and Christopher Re, “How to train your hippo: State´ space models with generalized orthogonal basis projections,” arXiv preprint arXiv:2206.12037, 2022.

[14] Albert Gu, Karan Goel, Ankit Gupta, and Christopher Re, “On the parameterization and initialization of diag-´ onal state space models,” NeurIPS, vol. 35, pp. 35971– 35983, 2022.

[15] Jimmy TH Smith, Andrew Warrington, and Scott W Linderman, “Simplified state space layers for sequence modeling,” arXiv preprint arXiv:2208.04933, 2022.

[16] Jiuming Liu, Ruiji Yu, Yian Wang, Yu Zheng, Tianchen Deng, Weicai Ye, and Hesheng Wang, “Point mamba: A novel point cloud backbone based on state space model with octree-based ordering strategy,” arXiv preprint arXiv:2403.06467, 2024.

[17] Christoph B Rist, David Emmerichs, Markus Enzweiler, and Dariu M Gavrila, “Semantic scene completion using local deep implicit functions on lidar data,” IEEE TPAMI, vol. 44, no. 10, pp. 7205–7218, 2021.

[18] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in IEEE 3DV, 2016, pp. 565–571.

[19] Karen Simonyan and Andrew Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.

[20] Jie Li, Yu Liu, Dong Gong, Qinfeng Shi, Xia Yuan, Chunxia Zhao, and Ian Reid, “Rgbd based dimensional decomposition residual network for 3d semantic scene completion,” in CVPR, 2019, pp. 7693–7702.

[21] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen¨ Gall, “Semantickitti: A dataset for semantic scene understanding of lidar sequences,” in ICCV, 2019, pp. 9297–9307.

[22] Luis Roldao, Raoul de Charette, Raoul Verroust-Blondet, Anne, Raoul Verroust-Blondet, Anne, and Anne Verroust-Blondet, “Lmscnet: Lightweight multiscale 3d semantic completion,” in 3DV, 2020, pp. 111– 119.

[23] Hao Zou, Xuemeng Yang, Tianxin Huang, Chujuan Zhang, Yong Liu, Wanlong Li, Feng Wen, and Hongbo Zhang, “Up-to-down network: Fusing multi-scale context for 3d semantic scene completion,” in IROS. IEEE, 2021, pp. 16–23.

[24] Xuemeng Yang, Hao Zou, Xin Kong, Tianxin Huang, Yong Liu, Wanlong Li, Feng Wen, and Hongbo Zhang, “Semantic segmentation-assisted scene completion for lidar point clouds,” in IROS. IEEE, 2021, pp. 3555– 3562.

[25] Jianbiao Mei, Yu Yang, Mengmeng Wang, Tianxin Huang, Xuemeng Yang, and Yong Liu, “Ssc-rs: Elevate lidar semantic scene completion with representation separation and bev fusion,” in IROS. IEEE, 2023, pp. 1–8.

[26] Yijun Yang, Zhaohu Xing, and Lei Zhu, “Vivim: a video vision mamba for medical video object segmentation,” arXiv preprint arXiv:2401.14168, 2024.

[27] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall, “Semantickitti: A dataset for semantic scene understanding of lidar sequences,” in ICCV, 2019, pp. 9297–9307.