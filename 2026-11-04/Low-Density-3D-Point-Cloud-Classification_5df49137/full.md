# Low-Density 3D Point Cloud Classification

Ahmed Baha Ben Jmaa

Efrei Research Lab

Paris Pantheon-Assas University´ , France

ahmedbaha.benjmaa@outlook.fr

Faten Chaieb

Efrei Research Lab

Paris Pantheon-Assas University´ , France

faten.chakchouk@efrei.fr

Abstract—3D point cloud classification, a task in computer vision, has recently gained substantial interest due to its extensive applications across various domains such as augmented and virtual reality, robotics, and autonomous driving. However, classifying 3D point clouds becomes challenging in real-world scenarios where the density levels vary. These variations are due to limitations inherent to current sensing technologies and variable environmental conditions, which often prevent the capture of detailed and comprehensive geometric representations. Existing classification methods, which have demonstrated impressive results on dense point clouds, may be affected and could experience reduced performance when dealing with low-density point clouds. Addressing this challenge, this paper introduces LD-PointNet++, an architecture optimized for the classification of low-density point clouds. To pave the way for LD-PointNet++, we conduct the first benchmarking study that systematically evaluates stateof-the-art point cloud classification methods across varying densities, revealing significant performance declines in low-density environments. Experimental results on the ModelNet benchmark dataset demonstrate that LD-PointNet++ not only outperforms these methods in low-density scenarios but also demonstrates consistent accuracy above 90% across different density levels.

Index Terms—Low-Density 3D Point Cloud, 3D Point Cloud Classification, PointNet++

## I. INTRODUCTION

Nowadays, point cloud, a geometric representation of threedimensional (3D) data, is gaining increasing interest in emerging computer vision applications [1], [2]. It is finding more and more uses in domains including augmented and virtual reality, robotics, and autonomous driving, thanks to rapid advancements in acquisition techniques such as laser scanners and depth cameras [2]. A 3D point cloud represents the direct result of capturing the spatial distribution of objects and offers the advantage of being both simple and precise. It provides rich geometric information, allowing for a faithful reproduction of the real world.

However, real-world scenarios often present challenges in acquiring point clouds, primarily due to variable environmental conditions and sensor limitations. These factors frequently result in low-density point clouds, characterized by sparse points that occur due to increased distances between the sensor and the 3D object/scene, or due to the constraints of the capture devices. These limitations compromise the ability to capture fine details of 3D objects/scenes, which are essential for computer vision tasks. For example, emerging applications like autonomous driving, which require efficient processing methods for point clouds, encounter a reduction in point density as the spatial separation from the sensor to the object extends.

Consequently, classifying 3D point clouds with low density poses several challenges. Indeed, a limited number of points in a point cloud can lead to an incomplete or even incomprehensible geometric representation of the 3D object/scene, resulting in decreased classification accuracy. Although stateof-the-art methods for point cloud classification have shown impressive results in scenarios where point clouds are very dense, they fail to maintain comparable performance in lowdensity scenarios. In response to this limitation, this paper introduces LD-PointNet++, a model specifically designed for low-density point cloud classification. The main objective is to address the challenges posed by variations in point cloud density, ensuring consistent and accurate classification outcomes across different scenarios.

Contributions. Our key contributions are summarized as follows: (1) We present the first benchmarking study of state-ofthe-art methods for low-density 3D point cloud classification through a robustness study on input density. This study reveals that the performance of these methods significantly degrades in low-density scenarios, providing crucial insights into their limitations. (2) We propose LD-PointNet++, a novel variant specifically optimized for low-density 3D point cloud classification. LD-PointNet++ not only consistently outperforms state-of-the-art methods in low-density scenarios but also maintains an accuracy above 90% across varying densities. We demonstrate that no previous method achieves such high accuracy with low density.

Paper Organization. In the following, Section II presents a benchmarking study of 3D point cloud classification methods, focusing on their robustness to density variations through experimental evaluations. Section III introduces LD-PointNet++, explaining its working principle in low-density scenarios. Section IV focuses on evaluating the effectiveness of LD-PointNet++ in classifying low-density 3D point clouds. Finally, Section V concludes the paper.

## II. UNVEILING THE CHALLENGE: LOW-DENSITY POINT CLOUD CLASSIFICATION - BENCHMARKING STUDY

In this section, we investigate the robustness of 3D point cloud classification methods to density variation. Real-world scenarios often involve incomplete point cloud acquisition due to factors such as acquisition conditions and sensor characteristics. Consequently, point density can vary across different regions of the 3D object/scene, which poses challenges for the analysis and processing of point clouds. Low-density 3D point clouds, in particular, present a challenge for classification models as they may contain insufficient data to accurately classify the 3D object/scene. Understanding how 3D point cloud classification methods perform under such conditions is crucial.

## A. Methods Overview

In this section, we provide a concise overview of the working principle of point cloud classification methods.

Pointwise MLP Methods. PointNet [3] is the pioneering model that processes point clouds directly as input to the neural network. It uses multiple Multilayer Perceptron (MLP) layers to learn feature for each point, and finally aggregates them into a global feature using a max pooling layer. Point-Net learns each point independently, therefore it does not capture local structure and interactions between points. To address these limitations, PointNet++ [4] is introduced as a hierarchical network that captures local features for each point from its neighborhood. The idea is to learns features by considering the local context of each point. To achieve this, PointNet++ defines a number of abstraction levels. Each abstraction level includes three distinct layers: a sampling layer, a grouping layer, and a PointNet layer. At the last abstraction level, features are aggregated using max pooling to obtain a learnable representation. This feature vector is then passed through a fully connected classifier network to obtain the score for each class. Following PointNets, many pointwise MLP methods have been introduced, either based on or inspired by PointNet++ [1], [5], [6].

Convolution-Based Methods. These methods define convolutional kernels for point cloud. Several definitions of convolution operators have been proposed to adapt convolutional neural networks to point cloud [1], [7]–[10]. In general, for the majority of approaches, for a point cloud $\mathbf { \bar { \mathcal { P } } } ~ = ~ \{ p _ { i } \in \mathbb { R } ^ { 3 } \} _ { \substack { i < N } } ~ \in ~ \mathbb { R } ^ { N \times 3 }$ , the input feature map in a convolutional layer, denoted as ${ \mathcal F } ,$ can be expressed as $\pmb { \mathcal { F } } = \left\{ f _ { i } \in \mathbb { R } ^ { D } \right\} _ { i < N } \in \mathbb { R } ^ { N \times D }$ . The convolution of $\mathcal { F }$ by a kernel g at a point $ { \dot { p } } \in \mathbb { R } ^ { 3 }$ with neighborhood $\mathcal { N } _ { p }$ is defined as:

$$
( { \mathcal { F } } * g ) ( p ) = \sum _ { p _ { i } \in N _ { p } } g \left( p _ { i } - p \right) f _ { i } .\tag{1}
$$

In this definition, two parts are important: the definition of the neighborhood $\mathcal { N } _ { p }$ and the definition of the kernel function $g .$ The two types of neighborhoods commonly used in the literature are the spherical neighborhood (defined by a radius) and the K-Nearest Neighbors (KNN) neighborhood [1]. The kernel function $g$ is a crucial element in (1), differentiating state-of-the-art methods. This function takes as its input the positions of neighbors centered around the point p $( q _ { i } = p _ { i } - p )$ and produces a weight matrix in $\mathbb { R } ^ { O }$ , where O represents the dimension of the output feature vector.

![](images/8dd2d3f000264a3e63308e59f98c8c4546bbff8eebdb6e7e0ed2a83f2c8ac1b8.jpg)  
Fig. 1: Robustness to sampling density variation.

Graph-Based Methods. Graph-based methods [1], [11], [12] are generally composed of two parts: (1) graph construction and (2) feature learning and clustering. Each point in the point cloud is considered as a vertex (node). From the neighborhood of each point, edges are generated. Formally, to create a directed graph $G = ( \mathcal { P } , \mathcal { E } )$ from a point cloud ${ \mathcal P } _ { : }$ , we connect each vertex i to every vertex j within its local spatial neighborhood. Learning is subsequently conducted in either spatial or spectral domains. In this way, these methods are grouped into two classes: Methods based on spatial structure analysis techniques and Methods based on spectral graph theory [1].

## B. Experimental Setup and Results

In this section, we assess the robustness of existing 3D point cloud classification methods to variations in density. We focus on the most representative methods, including PointNet++ [4], Point-TnT (Point Transformer-in-Transformer) [13], DGCNN [11], and PointMLP [6]. This study does not include other recent methods mentioned in [1] because they inherently require a large minimum number of points in the point cloud to function. To conduct the evaluation, we train and evaluate candidate methods using the ModelNet40 variant of the ModelNet dataset [14], which is widely adopted as a benchmark for point cloud classification. This dataset comprises 12,311 CAD models categorized into 40 object classes. The standard data split is followed, resulting in a training set of 9,843 objects and a test set containing 2,468 objects. Each input point cloud consists of 1024 points, with each point represented by attributes $( x , y , z )$ denoting 3D Euclidean coordinates. The evaluation involves training these methods on the complete dataset and subsequently testing them on subsets of points with varying densities. This methodology—training on dense point clouds and testing on sparse ones—is designed to reflect real-world scenarios, where dense data might be available during the training phase but not during application. This approach is well-supported in the literature and aligns with practical applications as outlined in [4]. Density variations are achieved by randomly sampling the dataset, as proposed in [4], while preserving the original distribution pattern of the point cloud.

![](images/691972dc373cf2fe15ad05e135da2f41bec3d1d934216353ef4439863a14fa8a.jpg)  
Fig. 2: Comparison of t-SNE visualization for different models under experimental setups ES1 and ES2.

In Fig. 1, we present the test accuracy as a function of the sampling ratio. The sampling ratio is defined as $\begin{array} { l } { { \frac { N } { k } } } \end{array} = \begin{array} { l } { { \frac { 1 0 2 4 } { k } } } \end{array}$ , where N is the number of original points, and k represents the number of sampled points $( k \in$ {64, 96, 128, 160, 192, 224, 256, 512, 1024}).

To assess the resilience of these methods for low-density 3D point cloud classification, we conducted experiments using a sampling ratio corresponding to 64 points per point cloud during the test phase, thereby simulating low-density scenarios. Subsequently, we focus on the following two Experimental Setups (ES): ES 1 - Training and testing using the entire point cloud (1024 points). ES 2 - Training using the entire point cloud (1024 points) and testing using only a few points (64 points). For the evaluation of model performance, Table I presents the following classification metrics: accuracy, precision, recall, and F1-score, computed on the test set across all experimental configurations. Finally, we apply t-SNE (t-distributed Stochastic Neighbor Embedding) [15] to the learned features of the different models under the two experimental setups. The results are summarized in Fig. 2.

## C. Findings and Analysis

How robust are state-of-the-art point cloud classification models to variations in input density? The results presented in Fig. 1 demonstrate the sensitivity of the evaluated point cloud classification models to density variation. The performance of the models compared in this study dramatically degrades when dealing with low-density 3D point cloud, especially with less than 160 input points, corresponding to a sampling ratio of 6.4. This degradation in accuracy is huge and may have implications in real-world applications where point clouds can be low-density. For example, PointMLP, considered one of the top-performing models in dense point

TABLE I: Performance of 3D point cloud classification models in two setups: full training and testing on 1024 points (ES1), and training on 1024 points with testing on 64 points (ES2).

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metrics</td><td colspan="2">Experimental Setup</td></tr><tr><td>ES1</td><td>ES2</td></tr><tr><td rowspan="4">PointNet++ [4]</td><td>Accuracy</td><td>92.42%</td><td>35.45% (↓ 56.97%)</td></tr><tr><td>Precision</td><td>92.55%</td><td>61.51% (↓ 31.04%)</td></tr><tr><td>Recall</td><td>92.42%</td><td>35.45% (↓ 56.97%)</td></tr><tr><td>F1-score</td><td>92.44%</td><td>34.66% (↓ 57.78%)</td></tr><tr><td rowspan="4">Point-TnT [13]</td><td>Accuracy</td><td>92.79%</td><td>38.17% (↓ 54.62%)</td></tr><tr><td>Precision</td><td>92.79%</td><td>62.04% (↓ 30.75%)</td></tr><tr><td>Recall</td><td>92.79%</td><td>38.17% (↓ 54.62%)</td></tr><tr><td>F1-score</td><td>92.68%</td><td>35.26% (↓ 57.42%)</td></tr><tr><td rowspan="4">DGCNN [11]</td><td>Accuracy</td><td>93.27%</td><td>4.62% (↓ 88.65%)</td></tr><tr><td>Precision</td><td>93.59%</td><td>6.28% (↓ 87.31%)</td></tr><tr><td>Recall</td><td>93.27%</td><td>4.62% (↓ 88.65%)</td></tr><tr><td>F1-score</td><td>93.39%</td><td>2.35% (↓ 91.04%)</td></tr><tr><td rowspan="4">PointMLP [6]</td><td>Accuracy</td><td>94.08%</td><td>4.29% (↓ 89.79%)</td></tr><tr><td>Precision</td><td>94.07%</td><td>10.37% (↓ 83.70%)</td></tr><tr><td>Recall</td><td>94.08%</td><td>4.29% (↓ 89.79%)</td></tr><tr><td>F1-score</td><td>94.04%</td><td>4.51% (↓ 89.50%)</td></tr></table>

cloud classification, has the most significant sensitivity to input density variations. With a degradation margin of up to 89.79%, PointMLP’s performance degrades significantly in low-density scenarios. On the other hand, PointNet++ shows some resilience compared to other models at different sampling rates. However, even PointNet++ struggles to achieve satisfactory results, achieving only 34.45% accuracy when operating with 64 points per input point cloud.

How do state-of-the-art point cloud classification models perform in low-density 3D point cloud classification? The robustness analysis focused on evaluating the models in two experimental configurations, ES1 and ES2 (see Table I). The results obtained in ES1, which serve as a reference for evaluating the performance of dense point cloud classification, show that DGCNN and PointMLP achieve reasonably high performance on all metrics: DGCNN and PointMLP respectively reach an accuracy of 93.27% and 94.08%. However, the performance of these models drops dramatically when tested with only a limited number of points (64 points) while being trained on the entire point cloud (1024 points). This decrease in accuracy and other metrics indicates that the models fail to generalize in low-density scenarios. This supports the hypothesis that state-of-the-art point cloud classification methods, which excel in dense point cloud scenarios, face significant challenges in maintaining comparable performance in low-density scenarios. The analysis of t-SNE visualizations of the features of different models under ES1 and ES2 further reinforces this observation (see Fig. 2). The good feature separation for DGCNN and PointMLP in the dense point cloud scenario (ES1) shows that these models effectively capture the data variability in dense point clouds. However, the poor feature separation observed for all four models in the lowdensity scenario (ES2) indicates their limitations in separating features in low-density point clouds. This finding aligns with the performance degradation observed in the quantitative evaluation, again confirming the difficulty of the models in handling low-density scenarios.

![](images/b348b756a9c8fc0807c751a4bb5960121db434faff56ad042702a19d7a63508a.jpg)  
Fig. 3: Illustration of LD-PointNet++ architecture.

This robustness study indicates that state-of-the-art models face significant challenges in maintaining their performance when dealing with low-density point clouds. Critical implications for real-world point cloud classification applications may arise. This emphasizes the need to improve the generalization capabilities of these models for more robust point cloud classification systems suitable for a wider range of applications.

## III. LD-POINTNET++: A PROPOSED METHOD FOR LOW-DENSITY 3D POINT CLOUD CLASSIFICATION

## A. Problem Formulation

Low-density 3D point cloud classification is a classification task that assigns a specific label to an object/scene represented by a point cloud with reduced density. The process of developing a deep learning model for this task involves training with dense 3D point clouds and then inferring on low-density point clouds during the testing phase. This methodology ensures the model’s ability to generalize and accurately classify point clouds with varying densities. Thus, the problem is formulated as follows. The input data during the training phase consists of a dense 3D point cloud, denoted as $\mathcal { P } _ { t r a i n } = \{ p _ { 1 } , . . . , p _ { n } \}$

During the testing phase, the model is presented with a lowdensity 3D point cloud, simulating a less accurate acquisition process, denoted as $\mathcal { P } _ { t e s t } = \{ p _ { 1 } , . . . , p _ { m } \}$ , where $m \ < \ n$ The main goal is to design a classification function, denoted as $f _ { c } ,$ capable of classifying low-density point clouds. $f _ { c }$ must convert each input point cloud into a corresponding probability for each category, while demonstrating the ability to generalize effectively across variable data densities.

## B. Model Overview

LD-PointNet++, as illustrated in Fig. 3, consists of two modules, Feature Extraction and Classification. The feature extraction module learns a set of local descriptors based on sampled points and their neighboring points. It comprises three distinct types of blocks: sampling, grouping, and PointNet. It learns a feature from an input point cloud. Once the features are extracted, the feature vector is subsequently mapped to classes using fully connected layers. Let $\mathcal P = \left\{ p _ { i } \in \mathbb R ^ { 3 } \right\} _ { i < N }$ represent a point cloud consisting of N points, where p<sub>i</sub> denotes a point with coordinates $( x , y , z )$ . The model is formulated as described in (2):

$$
{ { f } _ { i } } = { { \mathcal { A } } _ { j = 1 } ^ { K } } \left( { { h } _ { \theta } } ( { { p } _ { i } } , { { p } _ { i j } } ) \right)\tag{2}
$$

where $p _ { i }$ corresponds to a sampled point from $\mathcal { P } , \ f _ { i }$ is a feature vector representing the local region centered around the point $p _ { i } , p _ { i j }$ is a neighboring point of $p _ { i }$ which encodes the local information of $p _ { i } , h _ { \theta }$ represents a non-linear mapping function with learnable parameters denoted by $\theta ,$ and A refers to a symmetric aggregation operation over the K neighboring points of $p _ { i }$ (MAX function, in our case). In (2), three parts are important: (1) Sampled point $( p _ { i } ) \colon$ These points serve as representative points of the local region. (2) Neighborhood of a sampled point $( p _ { i j } ) \colon$ This corresponds to the function provided by the second block of the feature extraction module. (3) Feature learning operation (h<sub>θ</sub> and ${ \mathcal { A } } ) { \mathrm { : } }$ These operations are carried out using PointNet, which constitutes the third block of the feature extraction module.

Local Region Selection. To describe the local region around a point $p _ { i } .$ , we need to describe how to select the neighboring points of $p _ { i }$ . This task involves defining and parameterizing the neighborhood. The appropriate choice of a neighborhood definition depends on the characteristics of the point cloud. In the existing literature, two neighborhood definitions are commonly used [1]: Spherical neighborhood and KNN. The spherical neighborhood is defined by the radius and also a maximum number of neighbors limit. Unlike kNN, this type of neighborhood maintains the local region fixed, making the features more generalizable in space. This is advantageous for tasks that depend on recognizing local patterns [4]. Therefore, we adopt the definition of the spherical neighborhood as a versatile definition. The parameters of the neighborhood depend on the structure and density of the points in the data. The choice of these parameters is done empirically.

PointNet for Global-Local Feature Extraction. To extract features from the local region, we employ a PointNet [3]. As illustrated in Fig. 3, this block encodes a local region into a feature vector by taking M local regions of neighboring points as input and generating M feature vectors. In summary, after dividing the point cloud into M local regions centered around a sampled point, we encode the neighbors of the sampled point using PointNet into a feature vector. Once PointNet is applied to the local regions, we have a set of feature vectors representing all the local regions. We apply PointNet again at this stage to this group of vectors to finally obtain a learnable representation of the point cloud.

## C. Large Neighborhoods → Significant Improvements: Maximizing Low-Density 3D Point Cloud Classification

In computer vision, the receptive field is crucial for developing neural networks and enhancing their performance. Originally applied with success in 2D vision, it refers to the region of input data that influences the output of a neural unit, thereby marking the ”visible” area of the image for that unit. Increasing the size of the receptive field significantly enhances the network’s ability to capture features. Extending this concept to the 3D vision increases the capability of neural networks to interpret complex spatial structures, which is essential for understanding and extracting more informative features from 3D point cloud. In 2D, techniques such as the use of deeper hierarchical architectures or increasing convolution kernel sizes are employed to extend the receptive field. Transitioning these techniques from 2D vision to 3D point cloud analysis requires adapting these strategies, considering the unique properties of 3D point clouds such as their irregularity and density variability, which can impact the local features.

To effectively increase the receptive field in neural networks dealing with 3D point clouds, two main strategies can be considered, analogous to those in 2D: (1) the use of hierarchical architectures, and (2) expanding the neighborhood (the local region) of each point by increasing the radius in the case of spherical neighborhoods. While hierarchical structures were adopted in the original PointNet++, our LD-PointNet++ focuses on neighborhood expansion, as illustrated in Table II. This approach improves neural network performance, allowing for the consideration of a broader context, crucial for accurate classification in low-density areas. The main modification to consider the context of low-density point clouds is the expansion of the spherical neighborhood. Indeed, the classification model for low-density point cloud can learn incomplete representations of the underlying structures due to the lack of points in the point cloud. By enlarging the neighborhood, more neighboring points are included, increasing the number of points considered and enabling a more comprehensive understanding of the overall structure of the 3D object/scene and local context. Consequently, the model can make more informed decisions, thereby enhancing performance.

TABLE II: Comparison of PointNet++ and LD-PointNet++ for low-density 3D point cloud classification.
<table><tr><td>Property</td><td>PointNet++</td><td>LD-PointNet++</td></tr><tr><td>Enlarged Neighborhood</td><td>x</td><td>√</td></tr><tr><td>Hierarchical Architecture</td><td>√</td><td>x</td></tr><tr><td>Technique to Scale the Receptive Field</td><td>Hierarchical architecture</td><td>Larger neighborhood</td></tr><tr><td>Designed for Low-Density</td><td>x</td><td>√</td></tr></table>

In regions with sparse points, the neighborhood encompasses a broader area, ensuring sufficient points are considered for classification. Conversely, in denser regions, the model can focus on a smaller area for classification (see Fig. 4).

The use of a large neighborhood for low-density point cloud classification offers the advantage of naturally expanding the receptive field of neural networks. As a result, each neuron’s receptive field extends over a wider area of the input point cloud, including a larger number of neighboring points, which allows capturing more extensive features, both global and local, and better understanding the overall structure of the 3D object (see Fig. 5). Formally, the volume V(R) of a spherical neighborhood with radius R around a point p in a point cloud P is defined by the formula for the volume of a sphere:

$$
V ( R ) = \frac { 4 } { 3 } \pi R ^ { 3 }\tag{3}
$$

Thus, increasing R results in a cubic increase in the volume V(R), which geometrically expands the area that each neuron can access and analyze. The impact on the receptive field can be expressed by (3). This directly broadens the network’s capacity to analyze and understand the global structure of the 3D object/scene, facilitating a more precise classification, especially in low-density regions. In practice, this means that each neuron has access to a larger portion of the point cloud, enabling it to capture more extensive features, both global and local. Moreover, by increasing the radius R, the probability that a point p is isolated, i.e., without neighbors in his neighborhood, significantly decreases. This is particularly crucial in low-density regions. By reducing the isolation of points, the neural network can better grasp the underlying structures, even in sparse regions, thereby facilitating a more precise and robust classification of the point cloud.

![](images/8cffdee1fa0f1a8888985a3c6cfc277e8ace33f030efd28072d8d8d2e772a8f1.jpg)  
Fig. 4: Large neighborhood effect in low-density 3D point cloud classification

![](images/fa681054f6ee3c01a0ca07c9ac47c04ac036a7a891191e7c3c4490c676f25da3.jpg)  
Fig. 5: LD-PointNet++ expands receptive field in low-density 3D point cloud, improving feature capture and 3D structure recognition.

The choice of neighborhood radius is critical and must be adapted to the specific characteristics of each dataset. An optimal radius must be large enough to capture the necessary contextual data but not so vast that it includes irrelevant points that could dilute the significance of local structures. Achieving this balance requires empirical testing and adjustments based on the specific characteristics of each dataset.

## IV. EXPERIMENTAL EVALUATION OF LD-POINTNET++

## A. Experimental Setup

We conducted our experiments using the ModelNet dataset [14] to train and test our model. During the training phase, we use a fixed set of 1024 points per object, whereas for testing, we reduced the points to 64 per object, as discussed in Section II. The ModelNet dataset consists of two variants: ModelNet40, detailed in Section II, and ModelNet10, containing 4899 CAD models from 10 distinct object categories. We adhered to the standard data split procedure [4]. The training process involved training the network from scratch using the Adam optimizer on an Nvidia GeForce RTX 3090. We trained the model over 200 epochs with a batch size of 24. The initial learning rate was 0.001, and we implemented a decay strategy, reducing the learning rate by 0.0001. Furthermore, we used the cross-entropy loss function. We employed a spherical neighborhood with a radius of 0.8, allowing for a maximum of 64 neighbor points within each local region. Subsequently, each PointNet in the architecture consists of three fully connected layers, with the first PointNet having widths [128, 128, 256], and the second having widths [256, 512, 1024]. We used the same classification module as [4].

## B. Performance Evaluation in Low-Density Context

This section is dedicated to the evaluation of the classification process conducted by LD-PointNet++ in the context of low-density scenarios. In order to gain a comprehensive understanding of the performance, both quantitative and qualitative assessments were employed.

Quantitative Evaluation. To quantitatively evaluate the performance of LD-PointNet++, we compare the different metrics, precision, recall, and F1-score, under two experimental setups (ES1 and ES2): one where the model is tested with 1024 points and another with 64 points, as depicted in Fig. 6.

![](images/dfbf6bba0c634294f3513786a1490157cd9ee89c224b7165c87f4addeaddebda.jpg)  
Fig. 6: Classification results of LD-PointNet++ on the ModelNet40 dataset under experimental setups ES1 and ES2.

Qualitative Evaluation. To demonstrate the effectiveness of the learned representation of point cloud, we employed t-SNE visualization to assess the separability of the features. We applied t-SNE to the learned features in the two experimental setups, ES1 and ES2. The results are summarized in Fig. 7.

![](images/bfffa874d7f46733a558298aa40828e9b1deb3f967c268659672e3548af0ed08.jpg)  
(a) ES 1

![](images/2aa619869ba7ec7d93a11084c2a44cfabfb3f1563376f427a1ab81b7115f1a36.jpg)  
(b) ES 2  
Fig. 7: T-SNE visualization of the features learned by LD-PointNet++ under experimental setups ES1 and ES2.

## C. Comparison with State-of-the-Art

In this section, we present a comparative analysis of our proposed LD-PointNet++ model against existing state-of-theart methods for low-density 3D point cloud classification. As per standard practice, we present in Table III the overall accuracy (OA) and the mean class accuracy (mAcc) based on a test set where each point cloud consists of 64 points.

TABLE III: Comparison of LD-PointNet++ with state-of-theart for low-density 3D point cloud classification on Model-Net40 dataset (Training with 1024 points, Testing with 64 points).
<table><tr><td>Method</td><td>OA</td><td>mAcc</td></tr><tr><td>PointNet++ [4]</td><td>35.39 %</td><td>37.35 %</td></tr><tr><td>Point-TnT [13]</td><td>38.16 %</td><td>34.39 %</td></tr><tr><td>DGCNN [11]</td><td>4.61 %</td><td>5.05 %</td></tr><tr><td>PointMLP [6]</td><td>4.29 %</td><td>7.75 %</td></tr><tr><td>LD-PointNet++ (Ours)</td><td>90.10 %</td><td>86.43 %</td></tr></table>

## D. Discussion

How effective is LD-PointNet++ in low-density 3D point cloud classification? The results presented in the previous sections demonstrate the effectiveness of LD-PointNet++ in low-density 3D point cloud classification. Although it does not achieve the highest accuracy in dense point cloud scenarios (see Fig. 1), LD-PointNet++ shows very promising results in low-density scenarios. The success of our method is due to the use of a large neighborhood, which naturally expands the receptive field and efficiently captures both global and local features. LD-PointNet++’s robustness in the low-density scenario is confirmed by analyzing classification metrics as shown in Fig. 6. Even with a reduced number of input points (ES2), LD-PointNet++ achieves an F1-score of 90.16%, only 1.96% less than when using the full set of input points (ES1). This demonstrates the consistent and reliable performance of LD-PointNet++ in low-density 3D point cloud classification. Additionally, we visualize the learned features using t-SNE to show feature separability. As depicted in Fig. 7, even for the low-density scenario (ES2), LD-PointNet++ maintains good separation between classes, supporting the quantitative results.

How does LD-PointNet++ compare to state-of-the-art methods in low-density scenarios? As shown in Table III, LD-PointNet++ clearly outperforms all other methods in the low-density classification scenario, especially in terms of overall accuracy (OA) for the same 64 point input: an improvement of 54.71% compared to PointNet++ and up to 85.81% compared to PointMLP. This highlights LD-PointNet++’s ability to excel in very low-density point cloud scenarios.

How robust is LD-PointNet++ to variations in input density? The robustness of LD-PointNet++ to input density variations is also studied, as shown in Fig. 1. LD-PointNet++ outperforms other methods across various input density ratios, particularly in low-density scenarios, with a precision margin of up to 85.81% compared to PointMLP. In addition, LD-PointNet++ achieves almost constant accuracy, exceeding 90%, for all sampling ratios. To the best of our knowledge, there is no previous method that can achieve the same level of classification accuracy with such a low level of input point density.

## V. CONCLUSION

In this paper, we introduced LD-PointNet++, a novel variant specifically designed for low-density 3D point cloud classification, featuring a larger neighborhood. Through a robustness study, we evaluated the performance of existing 3D point cloud classification methods, highlighting their limitations in lowdensity contexts. The experiments conducted on the ModelNet dataset demonstrated that LD-PointNet++ outperforms existing methods, consistently achieving a classification accuracy of over 90%, regardless of point cloud density. For instance, even with a reduced subset of 64 points, LD-PointNet++ achieved an accuracy of 90.1%. While LD-PointNet++ may not reach the highest accuracy in dense point cloud scenarios, it stands out when evaluated with a reduced number of points.

## REFERENCES

[1] Y. Guo, H. Wang, Q. Hu, H. Liu, L. Liu, and M. Bennamoun, “Deep learning for 3d point clouds: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 12, pp. 4338–4364, 2021.

[2] B. Fei, W. Yang, W.-M. Chen, Z. Li, Y. Li, T. Ma, X. Hu, and L. Ma, “Comprehensive review of deep learning-based 3d point cloud completion processing and analysis,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 12, pp. 22 862–22 883, 2022.

[3] R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “PointNet: Deep learning on point sets for 3d classification and segmentation,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 77–85.

[4] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” in Advances in Neural Information Processing Systems, vol. 2017-December, 2017, p. 5100 – 5109.

[5] G. Qian, Y. Li, H. Peng, J. Mai, H. Hammoud, M. Elhoseiny, and B. Ghanem, “PointNeXt: Revisiting pointnet++ with improved training and scaling strategies,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 23 192–23 204.

[6] X. Ma, C. Qin, H. You, H. Ran, and Y. Fu, “Rethinking network design and local geometry in point cloud: A simple residual MLP framework,” in International Conference on Learning Representations, 2022.

[7] B.-S. Hua, M.-K. Tran, and S.-K. Yeung, “Pointwise convolutional neural networks,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 984–993.

[8] H. Thomas, C. R. Qi, J.-E. Deschaud, B. Marcotegui, F. Goulette, and L. Guibas, “KPConv: Flexible and deformable convolution for point clouds,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 6410–6419.

[9] J. Zhang, M.-Y. Yu, R. Vasudevan, and M. Johnson-Roberson, “Learning rotation-invariant representations of point clouds using aligned edge convolutional neural networks,” in 2020 International Conference on 3D Vision (3DV), 2020, pp. 200–209.

[10] Z. Zhang, B.-S. Hua, and S.-K. Yeung, “Riconv++: Effective rotation invariant convolutions for 3d point clouds deep learning,” International Journal of Computer Vision, vol. 130, no. 5, pp. 1228–1243, 2022.

[11] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph cnn for learning on point clouds,” ACM Transactions on Graphics (TOG), 2019.

[12] G. Te, W. Hu, Z. Guo, and A. Zheng, “RGCNN: Regularized graph CNN for point cloud segmentation,” in ACM international conference on Multimedia, 2018.

[13] A. Berg, M. Oskarsson, and M. O’Connor, “Points to patches: Enabling the use of self-attention for 3d shape recognition,” in 2022 26th International Conference on Pattern Recognition (ICPR), 2022, pp. 528– 534.

[14] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, “3d shapenets: A deep representation for volumetric shapes,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1912–1920.

[15] L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” Journal of Machine Learning Research, vol. 9, no. 86, pp. 2579–2605, 2008.