# Single-Stream Multi-Feature Fusion with Temporal Robustness for Gait Emotion Recognition

Shirong Lyu<sup>1</sup>, Silu Quan<sup>2</sup>, Yixuan Ding<sup>1</sup>, and Chengpeng Wang<sup>3(B)</sup>

<sup>1</sup> College of Computer and Information Science, Southwest University, Chongqing, China

<sup>2</sup> College of Physical Science and Technology, Southwest University, Chongqing, China

<sup>3</sup> Wisesoft Inc., Chengdu, China wcp12.4@gmail.com

Abstract. 3D skeleton-based gait emotion recognition faces high annotation costs, data scarcity, and poor generalization on heterogeneous data. This paper proposes SV-GCN, a single-stream multi-feature fusion framework with temporal invariance. We introduce intra-frame relative motion features to eliminate frame-rate sensitivity and embed heterogeneous cues at shallow layers, enabling early fusion without multistream complexity. For variable-length sequences, we design a global mask-guided valid-frame spatio-temporal graph convolution module, introducing frame-rate insensitivity for the first time in this domain. On the E-Gait dataset, our method achieves performance comparable to state-of-the-art while demonstrating strong generalization across varying sequence lengths and frame rates, ofering a viable pathway for pretraining on large-scale skeleton-based action recognition datasets. The code is available at https://github.com/lsr51/SV-GCN.

Keywords: 3D skeleton · gait emotion recognition · spatio-temporal graph convolutional.

## 1 Introduction

Emotion recognition holds significant application value in human-computer interaction, behavioral analysis, and intelligent surveillance. Research indicates that a substantial portion of afective cues are conveyed through non-verbal signals such as gait and gestures.

Unlike facial expressions, which are susceptible to illumination, distance, and observer efects [20], gait provides more reliable afective signals in long-distance scenarios due to its natural and unconscious nature. Advances in 3D pose estimation [21] have enabled skeleton-based gait emotion recognition. Early studies established kinematic-level correlations between gait and emotion [11–13]; for instance, positive emotions often correspond to increased arm swing and faster stride. With deep learning, models like LSTM [22] and GCN [23] have been introduced, significantly improving accuracy.

![](images/e6a461b051889c4223ed083e32d091d7f919195e759993cd364de5fbc13ca0a0.jpg)  
Fig. 1: Sample heterogeneity and feature diversity make it dificult to fully utilize information and model gait emotion recognition.

However, deploying existing gait emotion recognition models for practical applications faces severe challenges, primarily being insuficient generalization. Fig. 1 illustrates two key factors:

Feature heterogeneity and fusion. STEP [15] simply concatenates afective features with network-learned features, hindering efective utilization. BPM-GCN [18] treats afective features as constraints (Afective Constraint) for knowledge distillation, but this indirect guidance is limited, and extending the dualstream network to multi-stream increases complexity. Afective mapping [16] imposes only an overall constraint. These methods fail to embed afective features into local joints, preventing shallow-layer feature fusion and weakening fine-grained guidance of afective features on local representations.

Data scarcity and heterogeneous inconsistency. High annotation costs and scarce data are compounded by inconsistencies across datasets (sequence length, frame rate, gait cycle). While some studies employ generative augmentation [15] or semi-supervised learning [16] to address sample scarcity, leveraging large-scale skeleton-based action recognition datasets for pre-training remains unexplored. The E-Gait dataset [15] contains two subsets with diferent frame rates and lengths, yet existing models [15, 16, 18] directly ignore these diferences, forcibly padding sequences to fixed lengths—causing temporal distortion, feature contamination, and inaccurate batch normalization statistics.

To address the lack of temporal invariance and ineficient multi-feature fusion in existing GCNs, we propose a multi-feature fusion network robust to frame rate and sequence length. First, we introduce a frame-rate insensitivity joint motion feature that captures coordinated body movements, replacing conventional inter-frame absolute velocity [18]. Second, we embed multiple features (including afective features) into a unified space on a per-joint basis, enabling early fusion via a single-stream ST-GCN [4] in shallow layers. This avoids parameter explosion and overfitting while learning richer local patterns. Third, we adopt a global masking mechanism (inspired by speech/NLP [24,25]) to handle variable-length sequences, ensuring reduction operations use only valid frames and preventing padding contamination. We replace BatchNorm with mask-aware GroupNorm to avoid statistical drift. Meanwhile, we systematically leverage the diferences between the two subsets by conducting comparative experiments on separated and combined configurations of the E-Gait dataset. Our main contributions are as follows:

– We propose the Shallow Multi-feature Embedding and Fusion module (SMEF), which designs frame-rate insensitivity intra-frame relative motion features and embeds afective features into joints. This achieves the first early fusion of multiple features, overcoming existing methods’ limitation of treating affective features only as global constraints.

We propose the Valid Frame Spatio-Temporal Graph Convolutional Network (VF-STGCN), introducing frame-rate insensitivity to gait emotion recognition for the first time. Its global masking mechanism only operates on valid frames, avoiding bias and contamination.

We propose the Single-stream Variable-length Graph Convolutional Network (SV-GCN), which is robust to heterogeneous data with varying lengths and frame rates, and, for the first time, achieves fine-tuning of a gait emotion model using a model pre-trained on large-scale datasets.

## 2 Related Work

## 2.1 Skeleton-Based Action Recognition

Early deep learning approaches utilized RNNs and LSTMs [22] to model skeletal motion [3]. Subsequently, Graph Convolutional Networks (GCNs) became dominant due to their ability to explicitly model human body structure. Yan et al. [4] pioneered Spatio-Temporal Graph Convolutional Networks (ST-GCN), laying the foundation for many follow-up works [5, 6]. In recent years, Transformerbased methods [7] have also been introduced to capture global dependencies. Widely used datasets include NTU RGB+D [8] and Kinetics [9], which serve as important benchmarks for evaluating model generalization. For a comprehensive review, refer to [10].

## 2.2 Gait Emotion Recognition

Early research validated the feasibility of gait-based emotion recognition using handcrafted features combined with classical methods such as SVM [11]. With the advancement of deep learning, Randhavane et al. [14] fused handcrafted features with LSTMs for emotion recognition. Following the introduction of ST-GCN [4], a series of GCN-based methods emerged. Bhattacharya et al. [15] created the E-Gait dataset and proposed STEP; they later proposed a semisupervised method [16]. Other representative works include BPM-GCN [18] and MSA-GCN [19]. Due to dataset scarcity and overfitting concerns, Transformerbased methods remain largely unexplored in this domain.

![](images/376d179d9d67126ea4d0b1a6f7706046741bf22ef502454edbb47328fc3d19c2.jpg)  
Fig. 2: Overview of the proposed SV-GCN framework. The SMEF module (red) performs multi-feature embedding and fusion at joint level; the VF-STGCN module (green) handles variable-length sequences guided by a global valid frame mask. N, C, T, J denote batch size, channels, max frames, and number of joints, respectively. Yellow boxes indicate mask-aware modules.

## 3 Method

This section presents SV-GCN, a single-stream multi-feature fusion framework with a valid frame mechanism for robust gait emotion recognition. As illustrated in Fig. 2, the framework consists of two key components: the Single-stream Multifeature Embedding and Fusion (SMEF) module, which integrates heterogeneous features at the joint level with adaptive weighting, and the Valid Frame Spatio-Temporal Graph Convolutional (VF-STGCN) module, which employs a global masking strategy to handle variable-length sequences while preserving statistical stability. Together, these designs enhance the model’s robustness to temporal variations and multi-source heterogeneity.

## 3.1 Single-stream Multi-feature Embedding and Fusion Module

This module (SMEF) employs a single-stream network to perform joint-based local representation and fusion of diferent types of features at the input stage.

The three-dimensional joint coordinates $f _ { \mathrm { C o o r d } } ( x , y , z )$ serve as the primary foundational pose features.

Afective features describe geometric relationships among joints, such as angles, distances, and areas. These are typically defined by 2–3 joints. Although they are not attributes of a single node, they have a clear dominant node in spatial relationships (except for area features). For example, angle features are dominated by the middle joint. Drawing on the afective feature sets defined in papers [16, 18], we associate each joint with the most relevant afective features, with the mapping relationship shown in Fig. 3. For limb joints, the angle feature is assigned to the middle joint; for endpoints (head, hands, feet), we design specific angle features as indicated in blue and green in the figure.

For three adjacent joints on a limb connection, the included angle is calculated as the afective feature of the middle joint. Given three-dimensional joint coordinates $J _ { 1 } , J _ { 2 } , J _ { 3 } \in \mathbb { R } ^ { 3 } \ ( \mathrm { F i g . ~ 3 } ,$ second column), the angle feature $f _ { \theta }$ is expressed by (1).

![](images/ea5d8505e5eba0a27311c9d3275af9188daa1028c5b11bf120511e707ec22572.jpg)  
Fig. 3: Afective feature embeddings correspond to joint points

$$
\begin{array} { l } { { u = \displaystyle \frac { J _ { 1 } - J _ { 2 } } { \| J _ { 1 } - J _ { 2 } \| _ { 2 } } , \quad \displaystyle v = \displaystyle \frac { J _ { 3 } - J _ { 2 } } { \| J _ { 3 } - J _ { 2 } \| _ { 2 } } , } } \\ { { f _ { \theta } = \operatorname { a r c c o s } ( \operatorname* { m i n } \{ \operatorname* { m a x } \{ u \cdot v , - 1 \} , 1 \} ) \in [ 0 , \pi ] } } \end{array}\tag{1}
$$

To capture dynamic information and enhance the model’s robustness to data with varying frame rates, we introduce features based on relative motion rather than using absolute velocities that depend on time intervals. Let $f _ { \Delta t } ^ { ( j ) } \ \in \mathbb { R } ^ { 1 0 }$ denote the motion feature [linear vel. (4D: x, y, z, mag), linear acc. $( \mathrm { 4 D } \colon x , y , z ,$ mag), angular vel. $\left( \mathrm { 1 D } = f _ { \theta } \right)$ , angular acc. (1D)] of node $j$ between adjacent frames. By calculating the relative motion state of each joint with respect to the root node within the same frame, we eliminate the direct influence of temporal variables, thereby providing a time-scale-invariant description of coordinated body-part movements. The relative motion feature $f _ { \mathrm { r e l } } ^ { ( j ) } \in \mathbb { R } ^ { 1 0 }$ of node j with respect to the root node 0 is computed as in (2) and (3), where $\epsilon = 1 0 ^ { - 8 }$ prevents division by zero, ⊙ is element-wise multiplication, the logarithm stabilizes gradients, and σ represents the relative direction.

$$
f _ { \mathrm { r e l } } ^ { ( j ) } = \sigma \odot \ln \left( \frac { \lVert f _ { \Delta t } ^ { ( j ) } + \epsilon \rVert } { \lVert f _ { \Delta t } ^ { ( 0 ) } + \epsilon \rVert } \right)\tag{2}
$$

$$
\sigma \triangleq \mathrm { s i g n } ( f _ { \Delta t } ^ { ( j ) } ) \odot \mathrm { s i g n } ( f _ { \Delta t } ^ { ( 0 ) } ) \in \{ - 1 , + 1 \} ^ { 1 0 }\tag{3}
$$

The above pose features, motion features, and afective features (totaling 14 dimensions) are embedded and concatenated along the channel dimension within the network to form the initial fused feature $\mathbf { \bar { \boldsymbol { X } } } \in \mathbb { R } ^ { N \times C \times T \times J }$ , where $N , C , T , J$ denote batch size, number of channels, time steps, and number of joints, respectively.

We introduce a channel attention mechanism to allow the network to adaptively learn the importance of diferent feature channels. Considering that the features are located in the shallow layers of the network and their numerical scales may vary significantly, we first apply LayerNorm and linear projection to the features, as shown in (4), where Proj(·) denotes a linear transformation layer combined with a ReLU activation function.

$$
\begin{array} { r l } & { X _ { \mathrm { e m b e d } } = \left[ \mathrm { P r o j } \big ( \mathrm { L N } \big ( X _ { ( : , [ \mathrm { C o o r d } ] , : , : ) } \big ) \big ) , \right. } \\ & { \qquad \mathrm { P r o j } \big ( \mathrm { L N } \big ( X _ { ( : , [ f _ { r e l } [ 0 : \tau ] ] , : , : ) } \big ) \big ) , \ } \\ & { \qquad \mathrm { P r o j } \big ( \mathrm { L N } \big ( X _ { ( : , [ f _ { \theta } , f _ { r e l } [ 8 : 9 ] ] , : , : ) } \big ) \big ) \big ] } \end{array}\tag{4}
$$

Apply LayerNorm to $X _ { \mathrm { e m b e d } } .$ , then average pool over $T$ and $^ { J , }$ and generate channel-wise attention weights $w _ { c } ^ { i }$ using a two-layer MLP with sigmoid activation. By employing the mask (7), only valid frames are considered. The weighted feature output is given by (6).

$$
w _ { c } ^ { i } = \sigma \left( \mathrm { L i n e a r _ { 2 } } \left( \mathrm { R e L U } \left( \mathrm { L i n e a r _ { 1 } } \left( \frac { 1 } { J L _ { i } } \sum _ { t = 0 } ^ { L _ { i } - 1 } \sum _ { j = 0 } ^ { J - 1 } X _ { i , : , t , j } \right) \right) \right) \right)\tag{5}
$$

$$
X _ { \mathrm { a t t n } } = X _ { i , c , t , j } ^ { \mathrm { E m b e d } } \cdot w _ { c } ^ { i } \cdot M _ { i , t }\tag{6}
$$

## 3.2 Adaptive Alignment Valid Frame ST-GCN

To efectively handle variable-length gait sequences while avoiding overfitting, we introduce a global masking mechanism based on ST-GCN and propose the Valid Frame ST-GCN module (VF-STGCN). This design ensures that all temporal aggregation operations are applied only to valid data regions. A binary mask $M \in \langle 0 , 1 \} ^ { N \times T }$ indicates the valid length $L _ { i }$ of sample $i ,$ broadcast to $M _ { 4 \mathrm { d } } \in$ {0, 1}<sup>N×C×T</sup> <sup>×J</sup> with $M _ { 4 \mathrm { d } } = M _ { n , t } \ ( \forall _ { c , j } )$ . The valid computation domain Ω is defined as in (7). All temporal reduction operations are constrained to Ω.

$$
M _ { i , t } = \left\{ \begin{array} { l l } { 1 , } & { t < L _ { i } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} , \right. \qquad \varOmega = \{ ( n , c , t , j ) \mid M _ { n , c , t , j } = 1 \}\tag{7}
$$

However, in the presence of masking, the statistics of BatchNorm fluctuate with the valid count, leading to training instability. Therefore, we employ Masked GroupNorm (M-GN) as a replacement. M-GN divides the channels into G groups and independently computes the normalization statistics based solely on the valid frames within the current sample and the current group. For the g-th group, the mean $\mu _ { g }$ and variance $\sigma _ { g } ^ { 2 }$ are calculated as shown in (8), where valid\_coun $\begin{array} { r } { \bar { \boldsymbol { \mathbf { \rho } } } _ { g } = \sum _ { n , c , t , j } M _ { n , g , c , t , j } ^ { ( g ) } } \end{array}$ . This eliminates the influence of inter-sample length variation on the statistical distribution.

$$
\mu _ { g } = \frac { \displaystyle \sum _ { n , c , t , j } X _ { n , g , c , t , j } ^ { ( g ) } \cdot M _ { n , g , c , t , j } ^ { ( g ) } } { \mathrm { v a l i d } _ { - } \mathrm { c o u n t } _ { g } } , \quad \sigma _ { g } ^ { 2 } = \frac { \displaystyle \sum _ { n , c , t , j } \left( X _ { n , g , c , t , j } ^ { ( g ) } - \mu _ { g } \right) ^ { 2 } \cdot M _ { n , g , c , t , j } ^ { ( g ) } } { \mathrm { v a l i d } _ { - } \mathrm { c o u n t } _ { g } }\tag{8}
$$

To prevent invalid frames from propagating through the adjacency matrix A, valid frames are extracted and concatenated before graph convolution. The valid features of all samples are concatenated into a compact continuous tensor, on which standard graph convolution is performed, and the results are redistributed back according to the original sample order, as described in (9). Here, $T _ { \mathrm { t o t a l } } =$ $\textstyle \sum _ { i = 1 } ^ { N } L _ { i }$ denotes the total length of all valid frames in the batch.

$$
\begin{array} { r l } & { X _ { i } ^ { \mathrm { v a l i d } } = X _ { i } [ : , : L _ { i } , : ] \in \mathbb { R } ^ { C \times L _ { i } \times J } } \\ & { X ^ { \mathrm { c o m p a c t } } = \mathrm { C o n c a t } \left( X _ { 1 } ^ { \mathrm { v a l i d } } , X _ { 2 } ^ { \mathrm { v a l i d } } \dots , X _ { N } ^ { \mathrm { v a l i d } } \right) \in \mathbb { R } ^ { C \times T _ { \mathrm { t o t a l } } \times J } } \\ & { Y ^ { \mathrm { c o m p a c t } } , A ^ { \prime } = G \left( X ^ { \mathrm { c o m p a c t } } , A \right) } \\ & { Y _ { i } [ : , : L _ { i } , : ] = Y ^ { \mathrm { c o m p a c t } } [ : , t _ { i - 1 } : t _ { i } , : ] } \end{array}\tag{9}
$$

Furthermore, for downsampling operations with stride $s > 1$ (used in TCN and residual connections), the valid frame length is updated to $L _ { i } ^ { \prime } \overset { \cdot } { = } \left\lceil L _ { i } / s \right\rceil$ , and the temporal mask is correspondingly adjusted via pooling.

$$
\sum _ { Y _ { n , c } = 1 } ^ { T } \sum _ { j = 1 } ^ { J } M _ { n , t } \cdot X _ { n , c , t , j }\tag{10}
$$

$$
Z _ { n } = \operatorname { F C } \left( Y _ { n } \right)\tag{11}
$$

$$
\mathcal { L } _ { \mathrm { C E } } ^ { ( n ) } = - \log \left( \frac { \exp ( z _ { n , y _ { n } } ) } { \sum _ { c = 1 } ^ { C } \exp ( z _ { n , c } ) } \right)\tag{12}
$$

The network backbone consists of three stacked groups of VF-STGCN modules, each containing three blocks with channel dimensions of 64, 128, and 256, respectively. Downsampling between groups is performed via temporal convolution with stride $s = 2 ;$ the temporal kernel sizes are set to 11, 7, and 3 successively to accommodate short-sequence samples. The final features are aggregated by masked global average pooling, as shown in (10), where $\begin{array} { r } { L _ { n } = \sum _ { t = 1 } ^ { T } \bar { M } _ { n , t } } \end{array}$ denotes the number of valid frames of sample n. The pooled features are then mapped through a fully-connected layer and optimized using the standard cross-entropy loss, as described in (11) and (12).

## 4 Experiments

## 4.1 Datasets

The Emotion-Gait (E-Gait) dataset [15] is the only popular publicly available dataset, comprising 2,177 real and 1,000 synthetic 3D skeletal samples. Each sample consists of 16 joints with 3-dimensional coordinates. Among the real gaits, 1,835 samples are taken from the Edinburgh Locomotion MOCAP Database [1], where each running or walking gait contains 240 frames over 4 seconds. The remaining 342 samples are walking data collected by the authors, with frame counts varying between 18 and 75. We leverage the inherent diferences between the two subsets (EG-1835 and EG-342) to evaluate generalization under heterogeneous source conditions. The real gaits were labeled with four emotion classes (happy, sad, angry, and neutral) by the same domain experts. Following [18], for the 1,835 samples, we applied fixed-stride (t=5) downsampling on the 240-frame sequences to obtain samples of length T=48. For the 342 samples, we padded all sequences to a uniform length of 75 frames by repeating the last frame and truncating to the first 48 frames, aligning the temporal dimension across subsets. The data were split into training and test sets in a 9:1 ratio.

Table 1: Accuracy Comparison of Feature Embeddings
<table><tr><td></td><td>Coords.</td><td>Joint Affective Inter Feat.</td><td>CA Vel.</td><td>Acc(%)</td></tr><tr><td>1</td><td>√</td><td>√</td><td rowspan="3"></td><td>86.96 87.86</td></tr><tr><td>2</td><td>√</td><td>√</td><td>90.76</td></tr><tr><td>3 4</td><td>√ √</td><td>√ V</td><td>92.39</td></tr></table>

Table 2: Accuracy Comparisons between Fixed-length and Variable-length Datasets
<table><tr><td>Dataset</td><td></td><td>Base  $\mathrm { { V F } _ { \ V e l . } ^ { \ I n t r a } }$ </td><td>Kernel|Acc(%)</td><td></td></tr><tr><td rowspan="3">EG-1835</td><td>1</td><td>√ √ √</td><td rowspan="3">√ √ √</td><td rowspan="3">90.76 88.59 89.13 89.67</td></tr><tr><td>23</td><td>√ √</td></tr><tr><td>4</td><td>√ √</td></tr><tr><td rowspan="3">EG-1835 +EG-342</td><td>5 6</td><td>√ √</td><td></td><td rowspan="3">88.07 √</td></tr><tr><td>7</td><td>√ √ √</td><td>√</td><td>88.53 89.00</td></tr><tr><td>8</td><td>√ √</td><td>√</td><td>89.45</td></tr></table>

## 4.2 Implementation Details

The default experimental settings include an initial learning rate of 0.008, weight decay of $1 \times 1 0 ^ { - 4 }$ , a batch size of 12, and a total of 200 training epochs. We employ the Adam optimizer. To mitigate gradient instability during the initial training phase, we introduce a learning rate warm-up mechanism: over the first 7 epochs, the learning rate linearly increases from 0.0008 to 0.008. After the warmup phase, an exponential decay strategy is applied with a decay factor of $\gamma =$ 0.99. We randomly apply data augmentation strategies such as temporal scaling, temporal shifting, Gaussian noise-based coordinate jittering, and spatial rotation with certain probabilities to enhance sample diversity and prevent overfitting. All experiments are conducted on a single NVIDIA GeForce 3090 GPU.

## 4.3 Ablation Studies

We adopt the code from [15] as baseline and report accuracy averaged over three runs.

Analysis of multi-feature embedding. Table 1 evaluates the efectiveness of the proposed multi-feature embedding module on the EG-1835 subset, where all samples have uniform length and frame rate. Rows 1–3 progressively incorporate joint coordinates, afective features, and movement features, showing clear accuracy gains. Row 4 further introduces channel-wise attention, leading to the best performance.

![](images/f4f589c8409808375ac0a19020370f01e392f79086d39d7d47a09236e9b56620.jpg)  
(a) EG-1835

![](images/1c77b7d4528d350d4e5202508810bf0fc5eb677e269469f6fde87f7d26e0f093.jpg)  
(b) EG-1835 + EG-342  
Fig. 4: Comparison of Confusion Matrices for Models Trained on Diferent Datasets

Analysis of the impact of sequence-length and frame-rate consistency on model training. Table 2 compares two data configurations: EG-1835 (uniform) and EG-1835+EG-342 (variable-length). The label-to-sample counts in EG-1835 are (0:1048, 1:454, 2:254, 3:79), while EG-342 has (0:112, 1:33, 2:78, 3:119). The baseline uses joint coordinates, afective features, and inter-frame absolute velocity, with BatchNorm and a fixed kernel size of 9. Rows 2–4 and 6–8 progressively replace components with the proposed VF-STGCN module, intraframe velocity features, and kernel sizes of 9, 7, 5. After adopting VF-STGCN, BatchNorm is replaced with GroupNorm (32 channels per group), with dropout 0.2.

As shown in Table 2, adding the variable-length EG-342 subset increases training dificulty and lowers overall accuracy (row 1 vs. row 5). However, in this challenging setting, the proposed modifications consistently improve accuracy (rows 6–8 vs. row 5), demonstrating their efectiveness. Fig. 4 shows confusion matrices for the configurations in rows 4 and 8. The model trained only on EG-1835 achieves high overall accuracy but performs poorly on the minority Neutral class (Fig. 4a), while incorporating EG-342 yields more balanced predictions across all classes (Fig. 4b).

## 4.4 Comparison with SOTAs

Based on the configuration in Table 2 (row 8), we convert the large-scale skeletonbased action recognition dataset NTU RGB+D [8] from 25 joints to 16 joints, select 43,515 samples for pretraining, our method achieves competitive accuracy against state-of-the-art approaches, as summarized in Table 3. Compared to the baseline STEP [15], our method yields a significant improvement. Notably, while dual-stream methods like 2s-AGCN [5], TNTC [28], and BPM-GCN [18] rely on higher complexity or interaction modules, our simple single-stream network with early multi-feature fusion attains comparable performance, demonstrating the efectiveness of the proposed design.

![](images/ba1f6686cf719ccf36739ad0cf534c807cfd092f9072447be478d05127a7fdf9.jpg)  
Fig. 5: Gait sequence alignment from heterogeneous data (only 5 frames are shown). Frame # indicates the sequence number of the corresponding frame in its original sample sequence.

Table 3: Accuracy Comparisons of Diferent Methods on the Emotion-Gait Dataset
<table><tr><td>Methods</td><td>Acc (%)</td></tr><tr><td>LSTM [26] G-GCSN [27]</td><td>75.10</td></tr><tr><td></td><td>81.50</td></tr><tr><td>ProxEmo [17]</td><td>82.40</td></tr><tr><td>STEP [15]</td><td>83.15</td></tr><tr><td>2s-AGCN [5]</td><td>84.40</td></tr><tr><td>TNTC [28]</td><td>85.97</td></tr><tr><td>ST-Gait++ [29]</td><td>87.50</td></tr><tr><td>BPM-GCN [18]</td><td>90.37</td></tr><tr><td>SV-GCN(Ours)</td><td>89.91</td></tr></table>

## 4.5 Visualization

The two subsets of the E-Gait dataset exhibit significant diferences in gait patterns. EG-1835 contains diverse motion modes (walking, jogging, running) and spans about 4 seconds with four gait cycles, while EG-342 was collected under the specific scenario of “walk while thinking of the four diferent emotions,” lasting only about 1 second with one gait cycle. Using the method from [2], we rendered the joint data into dynamic 3D videos (Fig. 5). During preprocessing, EG-1835 was downsampled every 5 frames, whereas EG-342 was not. As shown in the figure, EG-1835 exhibits more pronounced pose variations over time. The limited valid frames and low inter-frame variability in EG-342 weaken classification performance, making the model especially prone to confusing ‘angry’ with ‘sad’.

## 5 Conclusion

This paper addresses key limitations in gait emotion recognition—ineficient multi-feature fusion, frame-rate sensitivity, and padding interference—by proposing a unified framework that enables efective training on heterogeneous data, consisting of a shallow multi-feature embedding and fusion module (SMEF) for joint-level afective feature integration and a valid-frame spatio-temporal graph convolution module (VF-STGCN) with global masking for robust variable-length sequence handling. Experiments on the E-Gait dataset demonstrate that the proposed SV-GCN achieves competitive performance, and its robustness on heterogeneous subsets suggests strong potential for transfer learning from large-scale skeleton-based action recognition datasets, ofering a viable pathway to alleviate the poor generalization of gait emotion models.

## References

1. I. Habibie, D. Holden, J. Schwarz, J. Yearsley, and T. Komura, A recurrent variational autoencoder for human motion synthesis, In Proceedings of the British Machine Vision Conference (BMVC) (2017).

2. W. Zhu, X. Ma, Z. Liu, L. Liu, W. Wu, and Y. Wang, MotionBERT: A unified perspective on learning human motion representations, In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023).

3. J. Liu, A. Shahroudy, D. Xu, and G. Wang, Hierarchical recurrent neural network for skeleton based action recognition, In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1110–1118 (2015).

4. S. Yan, Y. Xiong, and D. Lin, Spatial temporal graph convolutional networks for skeleton-based action recognition, In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) (2018).

5. L. Shi, Y. Zhang, J. Cheng, and H. Lu, Two-stream adaptive graph convolutional networks for skeleton-based action recognition, In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12 026–12 035 (2019).

6. Z. Liu, H. Zhang, Z. Chen, Z. Wang, and W. Ouyang, Disentangling and unifying graph convolutions for skeleton-based action recognition, In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 143–152 (2020).

7. C. Zheng, S. Zhu, M. Mendieta, T. Yang, C. Chen, and Z. Ding, 3D human pose estimation with spatial and temporal transformers, In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11 656–11 665 (2021).

8. A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, NTU RGB+D: A large scale dataset for 3D human activity analysis, in Proc. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1010–1019 (2016).

9. J. Carreira and A. Zisserman, Quo vadis, action recognition? A new model and the Kinetics dataset, In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6299–6308 (2017).

10. M. Liu, H. Liu, Q. Hu, B. Ren, J. Yuan, J. Lin, and J. Wen, 3D skeleton-based action recognition: A review, arXiv:2506.00915 (2025).

11. G. Venture, H. Kadone, T. Zhang, J. Grèzes, A. Berthoz, and H. Hicheur, Recognizing emotions conveyed by human gait, International Journal of Social Robotics, pp. 621–632 (2014).

12. B. Li, C. Zhu, S. Li, and T. Zhu, Identifying emotions from non-contact gaits information based on Microsoft Kinects, IEEE Transactions on Afective Computing, pp. 585–591 (2016).

13. M. Chiu, J. Shu, and P. Hui, Emotion recognition through gait on mobile devices, In: IEEE PerCom Workshops (2018).

14. T. Randhavane, U. Bhattacharya, P. Kabra, K. Kapsaskis, K. Gray, D. Manocha, and A. Bera, Learning gait emotions using afective and deep features, In: Proceedings of the 15th ACM SIGGRAPH Conference on Motion, Interaction and Games, pp. 1–10 (2022).

15. U. Bhattacharya, T. Mittal, R. Chandra, T. Randhavane, A. Bera, and D. Manocha, STEP: Spatial temporal graph convolutional networks for emotion perception from gaits, In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pp. 1342–1350 (2020).

16. U. Bhattacharya, C. Roncal, T. Mittal, R. Chandra, K. Kapsaskis, K. Gray, A. Bera, and D. Manocha, Take an emotion walk: Perceiving emotions from gaits using hierarchical attention pooling and afective mapping, In Proceedings of the European Conference on Computer Vision (ECCV), pp. 145–163 (2020).

17. V. Narayanan, B. M. Manoghar, V. S. Dorbala, D. Manocha, and A. Bera, Prox-Emo: Gait-based emotion learning and multi-view proxemic fusion for socially-aware robot navigation, In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2020).

18. Y. Zhai, G. Jia, Y.-K. Lai, J. Zhang, J. Yang, and D. Tao, Looking into gait for perceiving emotions via bilateral posture and movement graph convolutional networks, IEEE Transactions on Afective Computing (2024).

19. Y. Yin, L. Jing, F. Huang, G. Yang, and Z. Wang, MSA-GCN: Multiscale adaptive graph convolution network for gait emotion recognition, Pattern Recognition (2024).

20. J.-M. Fernández-Dols and M.-A. Ruiz-Belda, Expression of emotion versus expressions of emotions, In: Everyday Conceptions of Emotion, pp. 505–522 (1995).

21. R. B. Neupane, K. Li, and T. F. Boka, A survey on deep 3D human pose estimation, Artificial Intelligence Review, vol. 58, no. 1 (2025).

22. S. Hochreiter and J. Schmidhuber, Long short-term memory, Neural Comput., vol. 9, no. 8, pp. 1735–1780 (1997).

23. T. N. Kipf and M. Welling, Semi-supervised classification with graph convolutional networks, In Proceedings of the International Conference on Learning Representations (ICLR) (2017).

24. A. Vaswani et al., Attention is all you need, In Advances in Neural Information Processing Systems (NeurIPS), pp. 5998–6008 (2017).

25. D. Bahdanau, K. Cho, and Y. Bengio, Neural machine translation by jointly learning to align and translate, In Proceedings of the International Conference on Learning Representations (ICLR) (2015).

26. T. Randhavane, U. Bhattacharya, K. Kapsaskis, K. Gray, A. Bera, and D. Manocha, Learning perceived emotion using afective and deep features for mental health applications, In: IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct) (2019).

27. Y. Zhuang, L. Lin, R. Tong, J. Liu, Y. Iwamot, Y.-W. Chen, G-gcsn: Global graph convolution shrinkage network for emotion perception from gait, In Proceedings of the Asian Conference on Computer Vision (ACCV) (2020).

28. C. Hu, W. Sheng, B. Dong, and X. Li, Tntc: Two-stream network with transformerbased complementarity for gait-based emotion recognition, In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2022).

29. L. M. Lima, W. D. L. Costa, E. T. Martinez, V. Teichrieb, ST-Gait++: Leveraging spatio-temporal convolutions for gait-based emotion recognition on videos, In the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2024).

![](images/bde86067db3e60f16ceaf9ce8585e8a7545afbf4943cc2e47df93800ed23d5b6.jpg)  
Fig. 6: Multi-feature embedding channel attention

## 6 Supplementary Material

We performed a visual analysis of the attention weight vector in Eq. (5) of the main text, and the results are shown in Fig. 6. Channels 0-2 correspond to the three-dimensional (x, y, z) joint-coordinate features; the bright yellow regions indicate the importance of these basic pose features. Channels 11-13 correspond to the afective features we introduced (which are mainly composed of joint angles) as well as angular-velocity and angular-acceleration features. The prominent yellow highlighting indicates that joint rotation has a strong representational efect on afect. Channels 3-10 represent the linear-velocity and linear-acceleration features of the joints. Although their weights appear as relatively dark blue in the visualization, the values remain around 0.4, confirming the efectiveness of motion features in the overall feature composition. Furthermore, considering that replacing BatchNorm with GroupNorm reduces the number of valid samples available for computing statistics, we did not compute channel-wise attention independently for each joint, so as to avoid introducing noise due to insuficient sample size.