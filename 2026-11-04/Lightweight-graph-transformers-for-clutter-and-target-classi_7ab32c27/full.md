# Lightweight graph transformers for clutter and target classification in automotive radar

Adrian Gheorghiu, Tunc Alkanat, Ashish Pandharipande

Abstract— Radar is a core sensor modality for scene perception to achieve higher levels of autonomy in automotive driving. A common occurrence in automotive radars is clutter - detections of non-existent moving objects, that can adversely impact target detection and classification performance and subsequent driving actions. We propose lightweight tiny graph transformer network (TGTNet) models for classifying clutter from stationary and moving targets in the scene. Performance evaluation on the public RadarScenes show that our proposed TGTNet models achieve

![](images/6c9aef4e41552152a3add30a9dddde7fb5cc6344eb5ad5ccadfd882c292da9e5.jpg)

similar classification performance in precision and recall metrics in comparison to state of the art models, with one to two orders of magnitude lower model size and significantly faster inference.

Index Terms— Automotive radar, Point clouds, Radar clutter, Graph neural networks, Automotive driving.

## I. INTRODUCTION

Automotive radars play an integral role in advanced driver assistance systems (ADAS) and autonomous driving systems (ADS), by providing range, Doppler, and angular information of targets in the driving scene. Their low cost and environmentally robust sensing capabilities make radars a core sensor in various automotive driving functionalities, such as adaptive cruise control, lane assist, and autonomous emergency braking. However, despite their higher robustness in comparison to sensors like cameras and LiDARs, automotive radars are susceptible to signal artifacts, such as clutter.

Radar clutter refers to erroneous detections in the radar system due to a non-existent object. As depicted in Fig. 1 for an illustrative scene, clutter can occur due to indirect signal propagation paths between the target and the radar, or erroneous ambiguity resolution in the radar processing chain. Specifically, ghost detections are a subclass of clutter detections that are caused by multipath signal reflections. These ghost targets often occur when radar signals bounce off large reflective surfaces such as buildings, metallic guardrails, other vehicles, or in environments like tunnels, and then arrive at the radar receiver via indirect paths [1], [2].

Clutter detections impact the overall radar detection performance and can mislead decision-making processes, e.g., leading to faulty braking or steering maneuvers. Hence, robust detection and mitigation of clutter targets are essential for highly reliable radar sensing and perception.

In this work, we propose tiny graph transformer network (TGTNet) models to classify clutter, stationary, and moving targets using radar point cloud data. The first class of TGTNet models use a single point cloud frame, while the second class uses temporal augmentation. Our main contributions are:

i. We propose TGTNet, lightweight point transformer models for efficient radar point cloud clutter segmentation. We propose the Superpoint Block, a point cloud neural network module that leverages energy model clustering to extract latent features. Model size is reduced further by employing a lowrank vector attention mechanism.

ii. Two temporal augmentations are incorporated in TGTNet to exploit time-series point cloud information.

iii. All proposed models are evaluated against state of the art models using the RadarScenes public dataset that contains clutter and target annotations [3], [4]. Qualitative and quantitative results are presented to demonstrate that the proposed models show comparable classification performance while having one to two orders of magnitude lower model sizes.

## II. RELATED LITERATURE

One class of works for ghost target detection is based on signal processing approaches. A simple model for ghost target detection based on the deviation of the angle of the motion vector from a box model was presented in [5]. The existence of multipath was modeled as a binary composite hypothesis test and a generalized likelihood ratio test detector was proposed in [6] to determine whether indirect paths exist in a range-Doppler cell. Both of these theoretical approaches were evaluated with limited simulations. In [1], the authors observe that ghost detections are inherently a geometrical phenomenon, and propose a simple rule-based method for detecting clutter by employing searches within the detections at the current timestep and a buffer of the previous timesteps.

![](images/1fc9ac62ae9c38a8788b2ad5db5927fa88c8459132f76e8ff6ac71f6a0d69bf6.jpg)  
Fig. 1: Illustration depicting radar point cloud with clutter, stationary, and moving targets.

A multistage rule-based multipath ghost object labeling technique leveraging LiDAR is proposed in [7]. Region proposals of multipath ghost reflections are created by reflecting real objects off reflective surfaces, and then candidate ghost reflections are chosen based on a Mahalanobis distance threshold. Performance is reported on the Radar Ghost dataset [2].

As data-driven point cloud processing methods gained traction, deep learning methods for clutter handling have emerged. The Ghost Radar Dataset in [2] is currently the only publicly available dataset that provides manually labeled ghost objects. The annotations are, however, limited to a single type of multipath reflection that is caused by a single reflective surface. Along with the dataset, two instance segmentation methods were presented that use PointNet++ [8] as the semantic segmentation backbone. The first method utilizes a similarity group proposal network [9] in an end-to-end training scenario, while the second method applies the well-known DBSCAN algorithm [10] to the semantic segmentation outputs.

In [11], a modified PointNet++ [8] variant with two abstraction levels was proposed for moving object segmentation. Training and testing were done on the RadarScenes dataset [3] by aggregating points in a 500 ms time window. The architecture was designed to process point clouds with a fixed size of 3072 points. If the point cloud has fewer points, it is oversampled; if it has more points, it is undersampled by selecting the points with the highest Doppler values.

An automatically labeled public dataset with clutter detections was proposed in [4], along with a modified deep learning architecture for ghost object detection. The automatic labeling method was applied to the RadarScenes dataset, which provided annotations for moving objects. First, the background detections found in close proximity to the “real” moving object were also labeled as “real” to account for radar imprecision. All the remaining detections that have the ego-motioncompensated radial speed above a certain threshold were marked as “clutter”. The authors then proposed a modified version of the PointNet++ architecture from [11] and tested it on both the new dataset and the private dataset from their previous work [1] to validate the label generation method. The model input was reduced to a smaller fixed size of 1280 points. Point clouds were aggregated in a queue of fixed size, with the oldest points discarded when new points come in. The point clouds were then brought to the same coordinate space using ego motion information, and training was done by oversampling points such that every sample in the batch had the same number of points. This method and the one in [11] are both specifically tailored to radar point clouds, making them suitable baselines for our models.

The Point Transformer was proposed in [12]. The Gaussian Radar Transformer (GRT), a new point transformer segmentation model, was presented in [13], employing a decoupled point attention mechanism that facilitates the flow of gradients. To enhance the model’s ability to learn spatial relations, attentive downsampling and upsampling layers were proposed, utilizing a simplified translation-sensitive attention mechanism that eliminates the need for key, query, and value separation. The model was trained and tested without any temporal aggregation by merging frames from all sensors into a single point cloud. Using this architecture, state-of-the-art performance was achieved on the RadarScenes moving object segmentation task.

Generally, deep learning approaches to clutter detection in literature use variations of PointNet++ with various methods for point cloud aggregation, without consideration to model complexity which is an important aspect for automotive edge implementations. In this work, we extend our previous work [14] and develop models that take advantage of recent advances in graph neural networks and transformer architectures that have thus far been overlooked. We efficiently extend our models proposed in [14] to time domain capabilities, and present additional qualitative visualizations as well as new quantitative results on a second dataset, the Radar Ghost Dataset [2], showcasing the wider applicability of our models.

## III. PROPOSED LIGHTWEIGHT TRANSFORMER MODELS

Point Transformers [12] are characterized by their application of an attention mechanism on localized structures which are defined by neighborhoods within the point cloud. Given an unordered point cloud $\mathcal { P }$ consisting of Cartesian positions $p \in \mathbb { R } ^ { N \times c }$ and endowed with features $x \in \mathbb { R } ^ { N \times d }$ , a Point Transformer uses a learnable function θ to process the relative position between a point and its neighbors and extract a positional encoding

$$
\delta = \theta ( p _ { i } - p _ { j } )\tag{1}
$$

that is used in the attention mechanism.

The scaled dot-product attention of the original Transformer is changed in favor of vector attention. As a result, the Point Transformer attention layer is defined as

$$
h _ { i } = \sum _ { j \in { \cal { N } } ( i ) } \rho \left( \gamma \left( \varphi ( x _ { i } ) - \psi ( x _ { j } ) + \delta \right) \right) \odot \left( \alpha ( x _ { j } ) + \delta \right) ,\tag{2}
$$

where $\varphi , \psi ,$ and α are linear projections of the query, key, and value, respectively [12]. $\gamma$ is a learnable mapping function (e.g., a multi-layer perceptron, MLP), $\rho$ is a normalization function (e.g., softmax), and $\mathcal { N }$ is a neighborhood function.

For our architecture, we take inspiration from [13] and use the Gaussian activation function

$$
G ( x ) = \exp \left( { \frac { - x ^ { 2 } } { 2 } } \right)\tag{3}
$$

as the normalization function, for its property of facilitating gradient flow. Additionally, this activation function facilitates the identification of similarities between points, as the attention weights between dissimilar features will be weighed down, while weights between similar features will be larger. This choice improves the expressivity of our small models.

To facilitate learning long-range dependencies for isolated points, the neighborhood function is chosen as the k-nearest neighbors (kNN). The attention mechanism is also simplified to exclude the positional encoding for the values, and the additional mapping $\gamma$ is discarded. The output is then passed through an output projection with bias $\omega ,$ resulting in the Gaussian Attention Layer

$$
h _ { i } = \omega \left( \sum _ { j \in \mathrm { k N N } ( i ) } G \left( \varphi ( x _ { i } ) - \psi ( x _ { j } ) + \delta \right) \odot \alpha ( x _ { j } ) \right) ,\tag{4}
$$

which is followed by a GELU activation [15] and placed between two pointwise linear layers, also followed by GELU activations, forming the Gaussian Transformer Block (GTB). The output projection $\omega$ concentrates the bias of the attention layer into one single projection and serves as a post-processing step for the attention mechanism.

Most clutter detections are found behind real objects and are caused by reflections of the real detections. These properties make the vector attention with element-wise feature subtraction found in the GTB suitable for learning feature dependencies and similarities that are indicative of clutter.

The MLP θ that extracts the positional encoding δ from the relative position uses two linear layers with a batch normalization layer [16] applied across all encodings in the batch. The first linear layer keeps the dimension of the positions $c ,$ and the output is passed through the batch normalization layer and a ReLU activation. The second linear layer projects the positional encoding to the dimension d of the attention layer.

## A. Radar point cloud clustering

To facilitate learning local features, we additionally introduce a new module, the Superpoint Block, which extracts features from a clustered point cloud input. The base building block of the Superpoint Blocks is the Superpoint Transformer. The Superpoint transformer takes as input a clustered point cloud and extracts the mean and standard deviation of the positions and features, resulting in $N _ { S }$ superpoint positions $\dot { [ p _ { s } , p _ { s } ^ { \mathrm { ( s t d ) } } ] }$ and $N _ { S }$ superpoint features $[ \bar { x } _ { s } ^ { \mathrm { ~ \doteq ~ } } x _ { s } ^ { \mathrm { ( s t d ) } } ]$ , where s represents a unique cluster index. The superpoint positions and features extracted from the clusters are then treated as a point cloud and passed through a GTB. Finally, the output superpoints are applied a gather operation, which assigns to each original point the relevant output superpoint that represents the same cluster to which the point belongs. Used in conjunction with a skip connection, the Superpoint Transformer extracts useful features for each point based on the cluster mean and spread as well as the relations to nearby clusters of similar spread. Based on the Superpoint Transformer, two Superpoint Block versions are then proposed, as seen in Fig. 2.

![](images/805e194c5665e4f473bd257114a0adebcefe3e72a34a6fb948a7bdf9584c20bc.jpg)  
Fig. 2: Diagram of the regular and light Superpoint Blocks. The mean and standard deviation are extracted for both the positions and features, with the Superpoint Transformer operating on a point cloud with four positional dimensions. The pointwise linear layers are followed by ReLU activations.

The first block version applies a pointwise linear layer that first projects the features to the block dimension, followed by a GTB. The output of the GTB is then passed through a Superpoint Transformer. The features of the two outputs are concatenated with the output positions of the Superpoint Transformer, then passed through a pointwise linear layer that reduces the feature dimension to the block dimensions, and finally passed through a dropout layer.

The second block is a lighter version of the first block that forgoes the initial linear layer and GTB. Other than that, the features and positions are passed through the Superpoint Transformer exactly like the first module.

To cluster the point clouds, we branch out from the pervasive DBSCAN algorithm [10] and its variants, which are often used in radar point cloud clustering. In that sense, we partition the point cloud using a global energy model similar to the one used in [17]. Therefore, the energy model is defined on top of the kNN of the point cloud and considers the Cartesian positions as a signal $f \in \mathbb { R } ^ { N \times c }$ . The objective is to find a piecewise constant signal g that approximates f by minimizing

![](images/ba3414e9c30795f4b6b555030db98161a115cb0f8dfbaa726fb6779e62ac1cf8.jpg)  
Fig. 3: Diagram of TGTNet. All pointwise linear layers except the last one are followed by GELU activations.

the energy model

$$
g ^ { \star } = \underset { g \in \mathbb { R } ^ { N \times c } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { N } \vert \vert g _ { i } - f _ { i } \vert \vert ^ { 2 } + \lambda \sum _ { i = 1 } ^ { N } \sum _ { j \in \mathbf { k } \mathrm { N N } ( i ) } [ g _ { i } \neq g _ { j } ] ,\tag{5}
$$

where the [·] operator is known as the Iverson bracket. This energy model consists of two parts: one that ensures fidelity to the original signal through the squared Euclidean norm, and a regularizer with regularization strength λ that determines the number of distinct values in the signal. The partition is then defined as the connected components of the kNN graph for which the approximated signal $g ^ { \star }$ is constant. Despite the direct minimization of the energy model not being computationally tractable, it can be efficiently approximated by the iterative $\ell _ { \mathrm { 0 } } { \mathrm { - c u t } }$ pursuit algorithm in [18].

Our approach diverges from DBSCAN due to its sensitivity to hyperparameter selection and its limitations when handling clusters with varying densities. Additionally, DBSCAN often classifies isolated detections as noise, resulting in the loss of potentially valuable information. In contrast, the proposed energy-based clustering method is specifically designed to accommodate the sparsity and irregularity of radar point clouds. This, in turn, ensures that our Superpoint Block learns important local relations between elements with simple shapes. To augment TGTNet with the Superpoint Blocks, the blocks are prepended to TGTNet. The models augmented with the normal and light versions of the Superpoint Block are then called S-TGTNet and SL-TGTNet, respectively.

## B. Single snapshot models

Our architecture, the Tiny Graph Transformer Network (TGTNet), takes the form of a simplified U-Net [19] with only one level of abstraction. The architecture is designed with translation invariance in mind and can operate on point clouds of varying sizes. Therefore, the input point cloud is passed through a pointwise linear layer, then through a downsampling layer. The downsampled point cloud is passed through a Gaussian Transformer Block (GTB) and another pointwise linear layer before being upsampled again through the upsampling block with a skip connection from the first linear layer. The upsampled point cloud is finally passed through a pointwise classification head MLP with two layers.

To downsample the point cloud, we employ the farthestpoint sampling (FPS) algorithm, first proposed for use on point clouds in PointNet++ [8], and set the downsampling ratio to halve the number of points after downsampling. To facilitate learning useful features for the downsampled point clouds, we again take inspiration from [13] and their attentive downsampling layer. The attentive downsampling layer presented in [13] uses a simplified attention mechanism that operates on the downsampled point cloud by aggregating the features of the 9-nearest neighbors in the original point cloud. Instead of using three separate matrices for extracting query, key, and value representations, the authors draw inspiration from [20] and process the features using a shared pointwise MLP. They then calculate the weights by applying a second MLP to the processed weights, concatenated with the Cartesian positions.

Due to the lack of positional invariance of the aforementioned attentive downsampling layer combined with the high parameter count from the use of two input MLPs, we propose our own attentive downsampling layer. To this end, our layer reverts back to the full attention in (4) and applies it on the downsampled point cloud by aggregating points from the original point cloud. In order to introduce more knowledge about the local regions of the downsampled points, the attention layer output for each point is concatenated with the standard deviation of the Cartesian coordinates in the aggregated region before being multiplied by the output projection ω.

The upsampling layer uses the same idea as the downsampling layer. This time, the query features originate from the skip connection of the first pointwise linear layer, while the key and value features originate from the downsampled point cloud. The only other difference is that the resulting queries, keys, and values must share the same dimension; however, the points they originate from have different feature dimensions, so a common dimension needs to be chosen for the features to be projected to. As a result, we decided to project everything to the dimensions of the skip connections.

To further decrease the memory footprint of our model, we also test modified versions of the attention layer and Gaussian Transformer Block architectures that learn low-rank (LoRa) adaptations for a subset of the linear layers. Inspired by LoRa adaptations [21] and the recent advancements in LLMs brought by DeepSeek-v3 [22], we propose the use of learning compressed representations in our attention layers and label the models that use this attention as $\mathrm {  { ~ \hat { ~ } { ~ L ~ o R a } ~ } } ^ { 6 6 }$ models.

To this end, the LoRa variants of the base models have all attentive layers modified, while augmentation blocks used for models have their own separate LoRa qualifier. For the Gaussian Transformer Block and our proposed attentive downsampling and upsampling layers, the linear projection matrices in (4) that extract the query, key, and value are transformed into low-rank projections. As a result, a matrix $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ is transformed into two subsequent projection matrices $A \in$ $\mathbb { R } ^ { k \times d _ { 2 } }$ and $B \in \mathbb { R } ^ { d _ { 1 } \times k }$ , with k being the compressed dimension. Furthermore, the keys and values are extracted using the same shared compression matrix, but different decompression matrices. In the case of the Gaussian Transformer Block, the linear layers that encapsulate the attention layer are also transformed into low-rank projections.

TABLE I: Comparison of our TGTNet model for different sizes with a baseline.
<table><tr><td rowspan="2">Setup</td><td rowspan="2">LoRa</td><td rowspan="2">Model size</td><td rowspan="2">Inf. time (ms)</td><td colspan="3">Moving object Precision</td><td colspan="2">Clutter</td><td colspan="2">F1</td><td colspan="2">Stationary</td><td colspan="2">Mean Precision</td><td></td></tr><tr><td></td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td></td><td>Precision</td><td>Recali</td><td>F1</td><td></td><td>Recall</td><td>F1</td></tr><tr><td>GRT [13]</td><td></td><td>20.07 MB</td><td>25.47</td><td>90.76</td><td>91.47</td><td>91.12</td><td>95.42</td><td>95.32</td><td>95.37</td><td>99.88</td><td>99.85</td><td>99.86</td><td>95.35</td><td>95.55</td><td>95.45</td></tr><tr><td>TGTNet (dim 32)</td><td>x</td><td>80.25 KB</td><td>6.83</td><td>89.49</td><td>88.81</td><td>89.15</td><td>94.50</td><td>94.27</td><td>94.38</td><td>99.82</td><td>99.86</td><td>99.84</td><td>94.60</td><td>94.32</td><td>94.46</td></tr><tr><td>TGTNet (dim 32)</td><td>V</td><td>40.29 KB</td><td>7.10</td><td>88.33</td><td>89.29</td><td>88.81</td><td>94.66</td><td>93.85</td><td>94.25</td><td>99.83</td><td>99.83</td><td>99.83</td><td>94.28</td><td>94.33</td><td>94.30</td></tr><tr><td>TGTNet (dim 16)</td><td>x</td><td>21.19 KB</td><td>6.93</td><td>85.39</td><td>87.49</td><td>86.43</td><td>93.70</td><td>92.05</td><td>92.87</td><td>99.79</td><td>99.78</td><td>99.79</td><td>92.96</td><td>93.11</td><td>93.03</td></tr><tr><td>TGTNet (dim 16)</td><td>√</td><td>11.21 KB</td><td>7.39</td><td>85.33</td><td>87.15</td><td>86.23</td><td>93.21</td><td>91.95</td><td>92.58</td><td>99.79</td><td>99.77</td><td>99.78</td><td>92.78</td><td>92.96</td><td>92.86</td></tr></table>

In all cases, we empirically chose the compressed dimension to be the input dimension divided by 8. For layers that process input features of different dimension sizes, we chose the larger input dimension of the two.

## C. Time-series snapshot models

To improve the performance of our small, optimized models and make them comparable to larger models, we test two methods for incorporating multiple radar frames as input.

First, we test temporal feature extraction using the Point-GRU method presented in [23]. For this, we adapt the Point-GRU equations to utilize the point transformer attention as the base building block for computing the update and reset gate point clouds, as well as the candidate point cloud. Due to experiencing divergence during training when using Gaussian normalization for the attention coefficients, we use standard softmax normalization for the PointGRU unit instead. Additionally, since temporal relations between points happen across small spatial distances, the kNN neighborhood is switched for the radius neighborhood:

$$
h _ { i } = \sum _ { j \in \mathrm { r a d i u s } ( i ) } \operatorname { s o f t m a x } { \left( \varphi ( x _ { i } ) - \psi ( x _ { j } ) + \delta \right) } \odot \alpha ( x _ { j } ) .\tag{6}
$$

The point-attention-rnn is then defined by applying (6) to every point in the current point cloud $\mathcal { P } _ { t } = ( P _ { t } , X _ { t } )$ by aggregating the points in the current point cloud and the point cloud of the previous state $\boldsymbol { S } _ { t - 1 } = ( P _ { t - 1 } , S _ { t - 1 } )$ . As a result, the PointGRU equation uses three different point-attention-rnn layers, each with its own set of parameters:

$$
\begin{array} { r } { Z _ { t } = \sigma ( \mathrm { p o i n t - a t t e n t i o n - r n n } _ { \mathrm { u } } ( ( P _ { t } , X _ { t } ) , ( P _ { t - 1 } , S _ { t - 1 } ) ) ) , } \end{array}\tag{7}
$$

$$
R _ { t } = \sigma ( \mathrm { p o i n t - a t t e n t i o n - r n n } _ { \mathrm { r } } ( ( P _ { t } , X _ { t } ) , ( P _ { t - 1 } , S _ { t - 1 } ) ) ) ,\tag{8}
$$

$$
\begin{array} { r } { \hat { S } _ { t - 1 } = \mathrm { p o i n t - a t t e n t i o n - r n n } _ { \mathrm { s } } ( ( P _ { t } , X _ { t } ) , ( P _ { t - 1 } , S _ { t - 1 } ) ) , } \end{array}\tag{9}
$$

$$
\tilde { S } _ { t } = \mathrm { t a n h } ( W _ { \tilde { s } } [ X _ { t } , R _ { t } \odot \tilde { S } _ { t - 1 } ] + b _ { \tilde { s } } ) ,\tag{10}
$$

$$
S _ { t } = Z _ { t } \odot \tilde { S } _ { t - 1 } + ( 1 - Z _ { t } ) \odot \tilde { S } _ { t } .\tag{11}
$$

Here, σ is the sigmoid function and tanh is the hyperbolic tangent function. This layer is used to extract temporally rich features, which are then passed onto our model to segment the last frame’s point cloud, an architecture which we call GRU-TGTNet. It is worth noting that (9) also deviates from the original paper [23], and is changed to additionally take the features of the current point cloud $X _ { t }$ as input. This modification is made to condition the GRU twice on the current point cloud, thus helping reinforce the importance of the most recent point cloud.

Second, we test a simpler method for incorporating the time domain, TGTNet-T, which involves including the index of the frame number in the sequence as a feature. This method processes all frames independently in the downsampling part, then aggregates them to the same coordinate space before they are passed through the Gaussian Transformer Block. Finally, for upsampling, only the point cloud of the last frame is passed as a skip connection.

## IV. PERFORMANCE EVALUATION

## A. Dataset and evaluation metrics

Our models were trained and evaluated on the RadarScenes dataset [3]; the method in [4] was used to for ground truth class labels. We utilize the original dataset split from [3] in conjunction with the additional test and validation split from [13]. Specifically, sequences 6, 42, 58, 85, 99, and 122 are assigned to the validation set, while the remaining sequences from the original validation set are used for the test set. The RadarScenes dataset comprises radar scans from four frequency modulated continuous waveform (FMCW) radar sensors positioned on either side of the car’s front bumper, covering different angles. To guarantee that a single radar snapshot captures the entire field of view of the car, the scans were merged from all four sensors in a single frame. The point features we trained our models on are the Cartesian positions in car coordinates, $x _ { c c } , y _ { c c } ,$ the ego-motion compensated radial velocity, $\hat { v } _ { r , s c } ,$ and the radar cross-section (RCS), σ. Therefore, a point comprises of position $\textit { p } = \ [ x _ { c c } , y _ { c c } ]$ and features $\boldsymbol { x } = [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , \sigma ]$ . Some models are also tested using the extended feature set used in [4], which additionally contains the distance $r _ { s c }$ and azimuth $\phi _ { s c }$ in sensor coordinates, the timestamp t in seconds relative to the latest frame, and the sensor id $s _ { i d }$ , resulting in $\boldsymbol { x } = [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , \sigma , r _ { s c } , \phi _ { s c } , t , s _ { i d } ]$ The choice of including the Cartesian positions into the features makes the models lose their translation invariance property, but improves performance.

We additionally train and report the performance of our models on the Radar Ghost Dataset, for which we use features $x \ = \ [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , A , r _ { s c } , \phi _ { s c } ]$ , where A is the radar amplitude in dB.

TABLE II: Comparison of our Superpoint Block models. Best values are shown in red, second best in blue. The reduced feature set is $\boldsymbol { x } = [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , \sigma ]$ , while the extended feature set is $\boldsymbol { x } = [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , \sigma , r _ { s c } , \phi _ { s c } , t , s _ { i d } ]$
<table><tr><td rowspan="2">Setup</td><td rowspan="2">LoRa†</td><td rowspan="2">Model size</td><td rowspan="2">Inf. time (ms)</td><td colspan="2">Moving object</td><td colspan="2"></td><td colspan="2">Clutter</td><td colspan="2">Stationary</td><td colspan="2"></td><td>Mean</td><td></td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recali</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>S-TGTNet* (dim 16)</td><td>x</td><td>47.11 KB</td><td>11.99</td><td>90.32</td><td>90.05</td><td>90.18</td><td>94.82</td><td>94.76</td><td>94.79</td><td>99.85</td><td>99.87</td><td>99.86</td><td>95.00</td><td>94.89</td><td>94.94</td></tr><tr><td>SL-TGTNet* (dim 16)</td><td>x</td><td>14.88 KB</td><td>10.56</td><td>88.85</td><td>85.63</td><td>87.21</td><td>92.95</td><td>93.20</td><td>93.07</td><td>99.75</td><td>99.87</td><td>99.81</td><td>93.85</td><td>92.90</td><td>93.36</td></tr><tr><td>S-TGTNet* (dim 16)</td><td>√</td><td>27.78 KB</td><td>12.80</td><td>90.11</td><td>89.04</td><td>89.57</td><td>94.27</td><td>94.91</td><td>94.59</td><td>99.85</td><td>99.86</td><td>99.85</td><td>94.74</td><td>94.60</td><td>94.67</td></tr><tr><td>SL-TGTNet* (dim 16)</td><td>√</td><td>13.92 KB</td><td>11.10</td><td>87.56</td><td>88.82</td><td>88.18</td><td>94.74</td><td>90.62</td><td>92.63</td><td>99.70</td><td>99.86</td><td>99.78</td><td>94.00</td><td>93.10</td><td>93.53</td></tr><tr><td>S-TGTNet (dim 16)</td><td>√</td><td>28.03 KB</td><td>12.01</td><td>90.92</td><td>89.39</td><td>90.15</td><td>94.80</td><td>92.67</td><td>93.72</td><td>99.71</td><td>99.89</td><td>99.80</td><td>95.14</td><td>93.98</td><td>94.56</td></tr><tr><td>SL-TGTNet (dim 16)</td><td>√</td><td>16.52 KB</td><td>10.12</td><td>86.86</td><td>90.13</td><td>88.47</td><td>95.37</td><td>92.57</td><td>93.95</td><td>99.82</td><td>99.82</td><td>99.82</td><td>94.01</td><td>94.17</td><td>94.08</td></tr><tr><td>S-TGTNet (dim 32)</td><td>√</td><td>103.03 KB</td><td>12.49</td><td>90.97</td><td>91.49</td><td>91.23</td><td>95.82</td><td>95.06</td><td>95.44</td><td>99.86</td><td>99.88</td><td>99.87</td><td>95.55</td><td>95.48</td><td>95.51</td></tr><tr><td>S-TGTNet# (dim 32)</td><td>√</td><td>103.03 KB</td><td>12.49</td><td>90.75</td><td>91.09</td><td>90.92</td><td>95.44</td><td>95.25</td><td>95.35</td><td>99.87</td><td>99.86</td><td>99.86</td><td>95.35</td><td>95.40</td><td>95.38</td></tr><tr><td>SL-TGTNet (dim 32)</td><td>V</td><td>50.16 KB</td><td>10.44</td><td>89.05</td><td>89.95</td><td>89.50</td><td>95.15</td><td>92.48</td><td>93.80</td><td>99.76</td><td>99.86</td><td>99.81</td><td>94.65</td><td>94.10</td><td>94.37</td></tr></table>

\* Reduced feature set. <sup>#</sup> DBSCAN clustering <sup>†</sup> Refers to Superpoint Block. All base models are LoRa.

1) Model and training details: For training, we chose to train all models for 50 epochs on RadarScenes and 30 epochs on the Radar Ghost Dataset, and then select the model with the highest F1 score for clutter. For our tests, we employ a combination of the Lovasz-Softmax loss [24] and a weighted´ cross-entropy loss. The Lovasz-Softmax loss is calculated by´ applying the Lovasz continuous extension to the Jaccard index,´ and using the resulting extension on the probability distribution output of the model. Therefore, for predicted labels yˆ and ground truth labels y, our loss is defined as

$$
\mathcal { L } ( \hat { y } , y ) = \mathcal { L } _ { L o v \bar { a } s z } ( \hat { y } , y ) + \mathcal { L } _ { C E } ( \hat { y } , y , w _ { c } ) ,\tag{12}
$$

where $w _ { c }$ is the vector of the class weights. For RadarScenes, this vector is set empirically to 0.5, 8, and 8 for the stationary, moving object, and clutter classes, respectively. For the Radar Ghost Dataset, we employ the method described in [25] and calculate the weights based on the effective number of samples. Due to the Lovasz-Softmax loss being a tractable´ surrogate for the mean intersection-over-union (mIoU) metric, we apply it only to clutter and moving object detections to prevent large gradients caused by background class errors in the early stages of training. Moreover, the loss is applied to each individual point cloud from the minibatch and then averaged, thereby ensuring the sample mIoU is maximized. For optimization, we use the Adam optimizer [26] with a learning rate of 0.0005 and a batch size of 32. The learning rate is reduced after each epoch according to a cosine annealing scheduler [27] with no restarts.

For our model, we tested two dimensions of size 16 and 32. The overall dimension is kept by the downsampling layer and the Gaussian Transformer Block, while the pointwise layer that follows the transformer block doubles it, and the upsampling layer reduces it back down to its original size. For the temporal models, we standardize the positions to the [−1, 1] range and normalize the other features by subtracting their training set mean and dividing by the standard deviation. The Cartesian positions are normalized together so as not to skew the neighborhood computations. The kNN neighborhoods that the layers operate on are set to k = 9 for the upsampling and downsampling layers and k = 16 for the Gaussian transformer block layers, which are the same parameters used in [13].

For the superpoint blocks, we chose the block dimension to be the same as the dimension of the base model it augments, thus 16 or 32. The clustering hyperparameters are selected to facilitate the extraction of geometric primitives while ensuring the number of superpoints is kept to a minimum. Therefore, we settled on the kNN number of neighbors to be k = 10 and the regularization strength λ = 1. The Superpoint block output dropout probability is set to 0.1.

For the temporal augmentations, we chose to use six frames for all methods. The GRU uses a dimension of 16 for the embeddings along with a radius of 5 for the neighborhood function, limited to at most 32 neighbors.

To evaluate model performance, we report precision, recall and F1 metrics for the entire test set for all of the classes. The reported mean F1 scores are macro-averaged, treating all classes with equal importance regardless of dataset imbalance. We also compare the inference time of the models, as well as their size (a small memory footprint is desired for real-time inference in ADAS [28]), which is reported for 32-bit floating point parameters. When measuring inference time, all models are tested under the same conditions and with the same input point cloud, taking into account the frame aggregation method employed by each model. Using a single RTX A6000 GPU, a total of 100 runs are averaged to compute the inference time.

## B. Discussion of results

1) Quantitative results: The base TGTNet architecture is first evaluated against the GRT for different combinations of hyperparameters on the reduced feature set to analyze how the proposed size reduction techniques affect performance, with results reported in Table I. It can be observed that using LoRa layers instead of regular layers has a negligible effect on performance while simultaneously halving the memory footprint. However, despite LoRa layers requiring fewer operations for the matrix multiplications, there is, on average, a less than 1ms (5%) increase in inference time due to the matrix multiplications executing sequentially. A greater performance difference is observed when changing the model size directly. It can be seen that the performance degrades by only 1–2% across metrics when using the smaller model (dim 16 vs. dim 32), while the memory footprint is four times smaller. Such a performance drop can be justifiable when resources are scarce. Compared to the GRT, the best-performing TGTNet models only have a 2% loss in moving object F1 performance and a 1% loss in clutter F1 performance, while being 200 times smaller and having almost a 4 times faster inference time.

Table II shows tests for Superpoint block augmentations using LoRa models as the base. It can be noticed that the regular Superpoint Block generally performs better than the light Superpoint Block. SL-TGTNet achieves only a 1% improvement for moving object F1 compared to TGTNet, while S-TGTNet achieves a 4% moving object F1 increase and a 1% clutter F1 increase. This occurs because the normal Superpoint Block provides an additional full GTB that adds expressivity to the model. By testing LoRa variants of the Superpoint Blocks independently from the base model, we observe that LoRa offers no performance degradation. Regarding the model size, an increase of 1% F1 for moving objects and an almost 2% increase in F1 for clutter objects can be noticed, increases which are similar to those observed in the base models.

TABLE III: Comparison of our temporal methods with baselines. Best values are shown in bold. The reduced feature set is $\boldsymbol { x } = [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , \sigma ] ,$ , while the extended feature set is $\boldsymbol { x } = [ x _ { c c } , y _ { c c } , \hat { v } _ { r , s c } , \sigma , r _ { s c } , \phi _ { s c } , t , s _ { i d } ]$
<table><tr><td>Setup</td><td>LoRa†</td><td>Model size</td><td>Inf. time (ms)</td><td>Precision</td><td>Moving object Recall</td><td>F1</td><td>Precision</td><td>Clutter Recall</td><td>F1</td><td>Precision</td><td>Stationary Recall</td><td>F1</td><td>Precision</td><td>Mean Recall</td><td>F1</td></tr><tr><td>[4]</td><td></td><td>1.82 MB</td><td>16.34</td><td>92.58</td><td>90.59</td><td>91.57</td><td>94.92</td><td>93.91</td><td>94.41</td><td>99.78</td><td>99.92</td><td>99.85</td><td>95.76</td><td>94.80</td><td>95.28</td></tr><tr><td>[11]</td><td></td><td>1.82 MB</td><td>18.23</td><td>88.48</td><td>88.59</td><td>88.54</td><td>78.75</td><td>93.74</td><td>85.60</td><td>99.79</td><td>98.83</td><td>99.31</td><td>89.01</td><td>93.72</td><td>91.15</td></tr><tr><td>GRU-TGTNet*</td><td></td><td>27.15 KB</td><td>30.76</td><td>87.90</td><td>88.49</td><td>88.19</td><td>94.25</td><td>93.36</td><td>93.80</td><td>99.82</td><td>99.84</td><td>99.83</td><td>93.99</td><td>93.90</td><td>93.94</td></tr><tr><td>TGTNet-T (dim 16)</td><td></td><td>11.52 KB</td><td>10.26</td><td>88.39</td><td>89.83</td><td>89.10</td><td>94.72</td><td>93.78</td><td>94.25</td><td>99.85</td><td>99.83</td><td>99.84</td><td>94.32</td><td>94.48</td><td>94.40</td></tr><tr><td>TGTNet-T (dim 32)</td><td></td><td>40.91 KB</td><td>9.94</td><td>90.21</td><td>91.26</td><td>90.73</td><td>95.66</td><td>94.68</td><td>95.17</td><td>99.86</td><td>99.87</td><td>99.87</td><td>95.24</td><td>95.27</td><td>95.26</td></tr><tr><td>S-TGTNet-T (dim 32)</td><td>x</td><td>180.56 KB</td><td>16.79</td><td>91.64</td><td>92.13</td><td>91.89</td><td>95.95</td><td>95.49</td><td>95.72</td><td>99.88</td><td>99.88</td><td>99.88</td><td>95.82</td><td>95.83</td><td>95.83</td></tr><tr><td>S-TGTNet-T (dim 32)</td><td>√</td><td>103.15 KB</td><td>16.75</td><td>91.45</td><td>92.35</td><td>91.90</td><td>96.05</td><td>94.91</td><td>95.48</td><td>99.86</td><td>99.88</td><td>99.87</td><td>95.79</td><td>95.71</td><td>95.75</td></tr><tr><td>S-TGTNet-T# (dim 32)</td><td>√</td><td>103.15 KB</td><td>16.75</td><td>90.45</td><td>91.90</td><td>91.17</td><td>95.74</td><td>95.37</td><td>95.55</td><td>99.89</td><td>99.85</td><td>99.87</td><td>95.36</td><td>95.71</td><td>95.53</td></tr><tr><td>SL-TGTNet-T (dim 16)</td><td>√</td><td>16.58 KB</td><td>13.61</td><td>89.20</td><td>91.05</td><td>90.11</td><td>95.74</td><td>91.30</td><td>93.46</td><td>99.73</td><td>99.88</td><td>99.80</td><td>94.89</td><td>94.07</td><td>94.46</td></tr></table>

\* reduced feature set <sup>#</sup> DBSCAN clustering <sup>†</sup> Refers to Superpoint Block. All base models are LoRa.

TABLE IV: Comparison of our models on the Radar Ghost Dataset. Best values are shown in bold.
<table><tr><td rowspan="2">Setup</td><td rowspan="2">LoRa</td><td rowspan="2">Model size</td><td rowspan="2">Inf. time (ms)</td><td colspan="3">Real</td><td rowspan="2">Ghost</td><td colspan="2"></td><td rowspan="2"></td><td colspan="2">Background</td><td rowspan="2"></td><td colspan="2">Mean</td></tr><tr><td>Precision</td><td>Recall</td><td>F1 Precision</td><td>Recall</td><td>F1 Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall F1</td></tr><tr><td>PointNet++</td><td></td><td>3.74MB</td><td>11.26</td><td>94.67</td><td>84.20</td><td>89.13</td><td>43.72</td><td>66.79</td><td>52.85</td><td>99.81</td><td>99.93</td><td>99.87</td><td>79.40</td><td>83.64</td><td>80.62</td></tr><tr><td>TGTNet (dim 16)</td><td>x</td><td>21.19 KB</td><td>6.93</td><td>85.69</td><td>85.53</td><td>85.61</td><td>56.01</td><td>73.46</td><td>63.56</td><td>99.71</td><td>99.60</td><td>99.66</td><td>80.47</td><td>86.20</td><td>82.94</td></tr><tr><td>TGTNet (dim 32)</td><td>x</td><td>80.25 KB</td><td>6.83</td><td>89.82</td><td>81.09</td><td>85.23</td><td>54.00</td><td>70.39</td><td>61.11</td><td>99.61</td><td>99.77</td><td>99.69</td><td>81.14</td><td>83.75</td><td>82.01</td></tr><tr><td>TGTNet (dim 16)</td><td>√</td><td>11.21 KB</td><td>7.39</td><td>96.07</td><td>85.00</td><td>90.20</td><td>54.77</td><td>79.69</td><td>64.92</td><td>99.70</td><td>99.86</td><td>99.78</td><td>83.51</td><td>88.18</td><td>84.97</td></tr><tr><td>S-TGTNet (dim 16)</td><td>√</td><td>28.03 KB</td><td>12.01</td><td>94.87</td><td>80.26</td><td>86.96</td><td>52.66</td><td>68.14</td><td>59.41</td><td>99.55</td><td>99.88</td><td>99.71</td><td>82.36</td><td>82.76</td><td>82.03</td></tr><tr><td>TGTNet-T (dim 16)</td><td>√</td><td>11.52 KB</td><td>10.26</td><td>92.41</td><td>88.72</td><td>90.53</td><td>58.64</td><td>76.08</td><td>66.23</td><td>99.79</td><td>99.80</td><td>99.80</td><td>83.62</td><td>88.20</td><td>85.52</td></tr><tr><td>S-TGTNet-T (dim 32)</td><td>√</td><td>103.15 KB</td><td>16.75</td><td>95.44</td><td>86.16</td><td>90.57</td><td>61.28</td><td>72.60</td><td>66.46</td><td>99.69</td><td>99.89</td><td>99.79</td><td>85.47</td><td>86.22</td><td>85.61</td></tr></table>

Additional tests are done by augmenting the base model with different methods that embed temporal information into the model, as can be seen in Table III. The first method, GRU-TGTNet, involves adding a graph GRU module as a temporal feature extraction block before the model. The second method, TGTNet-T, retains the same parameter set as the base model and utilizes the downsampling layer as a shared layer across different frames before aggregating them together. It can be noticed that, despite its simplicity, TGTNet-T achieves 1% higher clutter and moving object F1 scores compared to the more complex and slower GRU feature extraction method.

Further performance can be achieved by combining Superpoint Block feature extraction with parallel downsampling, as can be observed from the results of the TGTNet-T models. Every frame is processed separately by the same shared Superpoint Block and then passed onto the parallel downsampling network. This method offers an advantage of more than 1% F1 for the clutter class and achieves the same or higher F1 for moving objects, compared to the best metrics obtained by the baselines, while having a 20 times smaller memory footprint, and maintaining similar inference time. Moreover, it can be observed that the regular Superpoint Block brings a 1% increase in F1 for moving objects in the temporal model, while the clutter detection performance is similar.

Tables II and III also feature comparisons of DBSCAN with energy model clustering, demonstrating that the latter yields superior mean F1 scores across temporal and nontemporal model variants without increasing computational cost.

Table IV shows results for our model against a modified version of PointNet++. This version keeps the exact same architecture as the “semantic scene labelling” architecture described in the original PointNet++ paper [8], with downsampling set to 0.5. It can be noticed that our smallest TGTNet model outperforms PointNet++ by almost 9% in Real object F1 and 20% in ghost F1. Surprisingly, modifications that increase the model’s complexity appear to reduce its performance. We argue that this happens because of the dataset’s small size, which encourages overfitting.

2) Qualitative results: For qualitative results, we compare the two best-performing baselines, the GRT [13] and [4] against our smallest model, the LoRa dim 16 TGTNet, and one of the best-performing models, the LoRa dim 32 S-TGTNet-T.

As shown in Fig. 4 (a), S-TGTNet-T accurately classifies all structured clutter detections, unlike the larger models. The smaller 11KB TGTNet variant is also close to the larger models in its classification.

In Fig. 4 (b), the detections surrounding the vehicle marked in the image in red are ambiguous. Our smaller model, TGTNet, incorrectly estimates the spatial structure of the detections while also omitting clutter detections. S-TGTNet-T, however, correctly predicts all types of detections and their shapes, but is more conservative with its stationary and clutter prediction, thus estimating a larger moving object footprint than the ground truth and the larger models.

## V. CONCLUSIONS AND FUTURE WORK

We presented lightweight graph transformer models for clutter and target classification in radar point clouds. Different lightweight variants of the proposed TGTNet model, along with different temporal augmentations, were developed and evaluated. Among these models, applying parallel downsampling and merging the downsampled point clouds was found to be particularly effective, despite its simplicity. Adding superpoint feature extraction blocks further kept the model size small while achieving precision, recall and F1 classification scores compared to the much larger baseline models.

Finally, based on the performance that temporal augmentations bring, future work could involve utilizing pre-trained point cloud flow estimator networks to estimate the past positions of the points in the current point cloud, yielding more informative neighborhoods. Since the proposed models use 32- bit floating-point parameters, methods such as quantizationaware training could prove useful in further reducing memory footprint and computational cost. Future work will also focus on further optimizing the proposed models for deployment on automotive-grade processors.

![](images/44da11949e24398f71998f210052267e43c6db1d97a38063c8a838f5867ba926.jpg)

![](images/ce3995d76e816b8257fd10cc53f25c7ab7396d8f54045790a6c9c862de8761da.jpg)

![](images/863805b405a4f973f70e8ca5bdf6567b7b255a87f540ece60b0c8527c2ba602d.jpg)

![](images/e01860985956c04f0a93997426d3868cbc5dbcc8c0b42dc753328dab791d5b3d.jpg)  
(a)

![](images/11f245a4ae10f4c8646006c131988671d827b9198aa3e59959b7c11330541ed0.jpg)

![](images/889fbbea5f800c5512df594c395969031f83e254decd4816e54a2868be094853.jpg)

![](images/0c87a853103e9ecb029705b1c69d3a09d219d647c45ca67c80c0d5216b9c81e0.jpg)

![](images/8efd0e93346c4df3ed39ff32288e154de96893caf37b98d6506adf039beac399.jpg)

![](images/e820561897a3b40dcbb2dd86c047905983186ce842ee1c17405edf972c72e7e7.jpg)

![](images/441e412eae87d75be8bd6effe6cb57c3115cbeeff2f6e83ae3c1a98bee3319da.jpg)  
(b)

![](images/c57a6aeccd09492f1d421a75b4d501f0f9713d998110f4e6b6ab3478ab268a30.jpg)  
Fig. 4: Comparison of our models against baselines for two scenes (a) and (b). Stationary detections are shown in blue •, moving object detections in orange •, and clutter in green •. TGTNet is chosen as the LoRa variant with dimension 16, and S-TGTNet-T is chosen as the LoRa variant with dimension 32 and extra input features.

## REFERENCES

[1] J. Kopp, D. Kellner, A. Piroli, and K. Dietmayer, “Fast rule-based clutter detection in automotive radar data,” in 2021 IEEE Intl. Intelligent Transportation Systems Conf. (ITSC). IEEE, 2021, pp. 3010–3017.

[2] F. Kraus, N. Scheiner, W. Ritter, and K. Dietmayer, “The radar ghost dataset–an evaluation of ghost objects in automotive radar data,” in IEEE Intl. Conf. Intelligent Robots and Systems. IEEE, 2021, pp. 8570–8577.

[3] O. Schumann, M. Hahn et al., “RadarScenes: A real-world radar point cloud data set for automotive applications,” in 2021 IEEE 24th Intl. Conf. Information Fusion (FUSION). IEEE, 2021, pp. 1–8.

[4] J. Kopp, D. Kellner, A. Piroli, and K. Dietmayer, “Tackling clutter in radar data-label generation and detection using PointNet++,” in IEEE Intl. Conf. on Robotics and Automation. IEEE, 2023, pp. 1493–1499.

[5] F. Roos, M. Sadeghi, J. Bechter, N. Appenrodt, J. Dickmann, and C. Waldschmidt, “Ghost target identification by analysis of the Doppler distribution in automotive scenarios,” in 2017 18th International Radar Symposium (IRS), 2017, pp. 1–9.

[6] L. Zheng, J. Long, M. Lops, F. Liu, X. Hu, and C. Zhao, “Detection of ghost targets for automotive radar in the presence of multipath,” IEEE Transactions on Signal Processing, vol. 72, pp. 2204–2220, 2024.

[7] S. Danino and I. Bilik, “Automatic multipath annotation for conventional automotive radar datasets,” IEEE Sensors Journal, vol. 24, no. 8, pp. 13 500–13 517, 2024.

[8] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep hierarchical feature learning on point sets in a metric space,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[9] W. Wang, R. Yu, Q. Huang, and U. Neumann, “SGPN: Similarity group proposal network for 3d point cloud instance segmentation,” in Proc. of IEEE Conf. Comp. Vision & Pattern Recognition, 2018, pp. 2569–2578.

[10] M. Ester, H.-P. Kriegel, J. Sander, X. Xu et al., “A density-based algorithm for discovering clusters in large spatial databases with noise,” in kdd, vol. 96, 1996, pp. 226–231.

[11] O. Schumann, “Machine learning applied to radar data: classification and semantic instance segmentation of moving road users,” Ph.D. dissertation, Dissertation, Dortmund, Technische Universitat, 2021.¨

[12] H. Zhao, L. Jiang, J. Jia, P. H. Torr, and V. Koltun, “Point transformer,” in Proc. of the IEEE Intl. Conf. Comp. Vision, 2021, pp. 16 259–16 268.

[13] M. Zeller, J. Behley, M. Heidingsfeld, and C. Stachniss, “Gaussian radar transformer for semantic segmentation in noisy radar data,” IEEE Robotics and Automation Letters, vol. 8, no. 1, pp. 344–351, 2022.

[14] A. Gheorghiu, T. Alkanat, and A. Pandharipande, “Graph transformers for automotive radar clutter and target classification at the edge,” in IEEE International Conference on Acoustics, Speech & Signal Processing (ICASSP), 2026, to appear.

[15] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016.

[16] S. Ioffe and C. Szegedy, “Batch normalization: accelerating deep network training by reducing internal covariate shift,” in Proc. 32nd Intl. Conf. Machine Learning - Volume 37. JMLR.org, 2015, p. 448–456.

[17] L. Landrieu and M. Simonovsky, “Large-scale point cloud semantic segmentation with superpoint graphs,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 4558–4567.

[18] L. Landrieu and G. Obozinski, “Cut pursuit: Fast algorithms to learn piecewise constant functions on general weighted graphs,” SIAM Journal on Imaging Sciences, vol. 10, no. 4, pp. 1724–1766, 2017.

[19] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.

[20] B. Yang, S. Wang, A. Markham, and N. Trigoni, “Robust attentional aggregation of deep feature sets for multi-view 3d reconstruction,” Intl. Journal of Computer Vision, vol. 128, no. 1, pp. 53–73, 2020.

[21] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.

[22] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., “Deepseek-v3 technical report,” arXiv preprint arXiv:2412.19437, 2024.

[23] H. Fan and Y. Yang, “PointRNN: Point recurrent neural network for moving point cloud processing,” arXiv preprint arXiv:1910.08287, 2019.

[24] M. Berman, A. R. Triki, and M. B. Blaschko, “The lovasz-softmax loss:´ A tractable surrogate for the optimization of the intersection-over-union measure in neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4413–4421.

[25] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2019, pp. 9268–9277.

[26] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, Y. Bengio and Y. LeCun, Eds., 2015.

[27] I. Loshchilov and F. Hutter, “SGDR: Stochastic Gradient Descent with Warm Restarts,” in ICLR (Poster), 2017.

[28] F. Dellinger, T. Boulay, D. M. Barrenechea, S. El-Hachimi, I. Leang, and F. Burger, “Multi-task network pruning and embedded optimization¨ for real-time deployment in ADAS,” preprint arXiv:2101.07831, 2021.