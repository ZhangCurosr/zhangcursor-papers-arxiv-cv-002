# Robust Multi-Model Fitting through Learning Neighbor Regions

Chang Nie, Guangming Wang, Zhe Liu, Member, IEEE, and Hesheng Wang, Senior Member, IEEE,

Abstract—Multi-model fitting involves fitting multiple models accurately in a noisy environment. It is the basis for computer vision tasks such as scene reconstruction and mixed reality. However, its performance is often limited by insufficient feature utilization, inefficient optimization, model overlap, and the nondifferentiable pipelines. To overcome these limitations, we introduce a robust coarse-to-fine framework called Learning Neighbor Regions (LNR). Recognizing that substantial computational resources are wasted on numerous bad minimum sets, we propose the coarse-level module. This module utilizes a neural network to extract and analyze geometric feature of both local pointwise relationships and global contextual information in minimum sets, outputting confidence to pre-select a small number of good minimum sets, thereby enhancing overall efficiency before solving hypotheses. To address model overlap, LNR encodes neighbor region features for each hypothesis in its fine-level module. These region features consist of geometric features of neighboring data points, which can be used by multiple regions simultaneously. This design allows the neural network to individually refine and score each hypothesis. Importantly, LNR is trained to learn directly from data point features rather than from the hypothesis parameters, thus avoiding differentiating the sampling process and the model solvers. Extensive experiments on four classic multi-model fitting tasks demonstrate that LNR achieves stateof-the-art performance. The analysis suggests that LNR can be easily adapted to various robust multi-model fitting tasks. The code will be available at https://github.com/IRMVLab/LNR.

Index Terms—Multi-Model Fitting, Robust Estimation, Vanishing Point Estimation, Two-view Plane Segmentation, Two-view Motion.

## I. INTRODUCTION

R <sup>OBUST</sup> <sup>multi-model</sup> <sup>fitting</sup> <sup>is</sup> <sup>an</sup> <sup>essential</sup> <sup>and</sup> <sup>widely</sup>applied task in computer vision. It involves identifying and fitting multiple models within a dataset, often complicated by significant noise. Improvements in this area have broad implications, benefiting numerous applications such as robot navigation [1]–[4], 3D scene reconstruction [5]–[8], autonomous driving [9]–[13], and mixed reality [14]–[17]. To address this task, a straightforward idea is to extend the singlemodel methods, like RANdom SAmple Consensus (RANSAC) [18], to multi-model. For example, Sequential RANSAC [19] finds multiple instances by iteratively removing inliers. This greedy strategy resembles clustering, where each data point is assigned to a single model. Consequently, it may overlook the model overlap, where data points belong to both models. This limitation can lead to an underestimation of certain models, particularly when instances are not clearly separated.

![](images/228cda0ac40eab0a69e839f5c12c03c5e132e6055730ba9ade3a0f3b0a25339f.jpg)  
Fig. 1. LNR addresses the challenges of the two-step framework. LNR uses neural network in the coarse module to fully extract geometric features and select good minimum sets for efficiency. In the fine-level module, LNR encodes region features for each hypothesis independently to deal with model overlap. This framework learns on data points rather than the hypothesis parameters, avoiding sampling process and model solvers

Most recent methods adopt a two-step process. Firstly, they generate a hypothesis pool from minimum data sets [20]– [24]. These methods often employ iterative sampling techniques to expand the hypothesis pool, attempting to discard less promising ones. However, because each new model is built upon previous iterations, errors can accumulate due to model overlap throughout the process. Numerous noise also causes inefficient optimization on bad hypotheses. Secondly, a compound model is selected from the generated hypotheses [25]–[31]. Techniques include energy optimization [25], [27] or hyper-graph partitioning [28], [30]. These methods often rely on complex cost functions to handle model overlap. These intricate cost functions, however, might not fully exploit the underlying geometric information inherent in the data, potentially limiting their effectiveness across diverse and complex scenarios. These limitations motivate the exploration of alternative approaches that can more effectively utilize geometric features, such as integrating deep learning methods [22], [32]. However, challenges arise in incorporating learning-based methods due to the non-differentiable nature of traditional sampling and model-fitting processes.

To address the challenges of insufficient features utilization, inefficient optimization, model overlap, and the nondifferentiable pipelines as shown in Fig. 1, we propose Learning Neighbor Regions (LNR) for robust multi-model fitting. To avoid error accumulation from iterative sampling, LNR simultaneously generates a large number of minimum sets through random sampling. This randomized approach ensures a broad coverage of potential models and enhances generalization. Meanwhile, to fully utilize the geometric features and manage the computational cost associated with numerous bad minimum sets, LNR introduces a coarse-level module. This module analyzes the geometric features of each minimum set to select good minimum sets before solving for the hypotheses.

For confusion caused by model overlap and the barrier of non-differentiable pipelines, LNR encodes neighbor region features for each hypothesis. This allows individual data points to contribute to multiple region features simultaneously, providing rich information to deal with various noises and model overlap. A neural network then analyzes these region features to refine and score the hypotheses. Importantly, LNR focuses on features derived from the data points, rather than the hypothesis parameters, avoiding the need to differentiate the sampling process or model solvers. Finally, to determine the number of models without prior knowledge, LNR employs non-maximum suppression (NMS), a post-processing technique commonly used in object detection, to output the final multi-model fitting results.

We evaluate LNR on four established multi-model fitting tasks: multi-line fitting, vanishing point estimation, two-view plane segmentation, and two-view motion estimation. Experimental results demonstrate that LNR achieves state-of-the-art performance across these tasks.

The main contributions of LNR are as follows:

• We propose Learning Neighbor Regions (LNR), a novel framework for robust multi-model fitting. For challenges of non-learning methods, this learning-based method fully analyzes region features of hypotheses for better performance, effectively avoiding differentiation through sampling and solving processes.

• LNR introduces a coarse-level hypotheses generation module. It leverages random minimum set sampling for robustness and employs confidence-based selection to prioritize good minimum sets for coarse-hypothesis generation. Furthermore, a fine-level module encodes region features of hypotheses to handle noise and model overlap, which are then used for hypotheses refinement and assessment.

• Extensive evaluations across four classic robust multimodel fitting tasks show that LNR achieves state-ofthe-art performance in multi-line fitting, vanishing point estimation, two-view plane segmentation, and two-view motion tasks. Analysis indicates broad applicability of LNR to various robust multi-model fitting tasks.

## II. RELATED WORK

## A. Extension of Single-Model Robust Estimation

Robust multi-model fitting attracts extensive exploration due to its significance. Traditional single-model estimation tasks are often addressed using RANSAC-based methods [33], [34]. Given the success of these single-model approaches, researchers have naturally sought to extend them to the more complex scenario of multi-model fitting. For instance, Sequential RANSAC [19] iteratively detects models by clustering inliers, preserving the robustness and generalization of RANSAC. In contrast, MultiRANSAC [35] attempts to fit multiple models by sampling K models simultaneously to avoid data reuse issues. However, this approach requires K as a prior knowledge and faces computational inefficiency as K grows, since finding multiple good models in one iteration is challenging.

## B. Two Steps Frameworks

Nowadays, most methods [20], [26], [27], [36] often adopt a two-step framework: hypothesis generation followed by instance selection. First, a large number of minimum sets are sampled. Then, these minimum sets will generate a hypothesis pool, where the hypotheses are filtered by various numerical optimization methods. Second, many methods select multiple instances from the hypothesis pool to form a compound model. Following the idea of RANSAC, the selected compound model is generally required to have the most inliers.

In the hypothesis pool generation step, some methods aim to improve efficiency by generating only necessary hypotheses. For example, Chin et al. [37] employ residual ordering information to guide the sampling process. Purkait et al. [36] propose a hyperedge-guided sampling strategy based on the concept of clustering. MF-Net [38] directly classifies the minimum sets as valid or invalid through a neural network. T-Linkage [39] performs clustering via preference sets. However, these clustering-style techniques ignore the reuse of data. Prog-X [20] removes bad instances through the design of proposal validation and optimizing modules to improve efficiency. CONSAC extends the NG-RANSAC [40] method from single-model robust estimation to multi-model fitting tasks. Specifically, CONSAC simultaneously feeds generated hypotheses information along with data points into a network. Then the network produces the probabilities of data points, which can guide the sampling for this iteration. PARSAC [32] improves its efficiency by parallel computing. But the number of instances is needed as a priori information, which limits the application of PARSAC.

In the selecting instances step, several approaches aim to evaluate the qualities of hypotheses [25]–[27]. PEARL [25] addresses geometric multi-model fitting by viewing it as an optimization problem. This approach employs a global energy function that considers both geometric errors and the intrinsic regularity of clusters. Amayo et al. [26] explore the soft assignments of points to the geometric model by minimizing the overall assignment energy. In addition, there are methods [28] that treat instances as vertices and points as hyperedges, solving the evaluation problem through hyper-graph partitioning. MSH [28] regards the multi-model fitting task as a mode-seeking problem on a hyper-graph.

![](images/92e7056c03a934b5e39c2b9dace2e983e9d6c333b1054a415ee9ce2253706d8d.jpg)  
Fig. 2. The pipeline of the proposed LNR. Using the multi-line fitting task as an example. Different colored lines indicate their scores and categories. Two points can form a minimum set to determine a line. LNR consists of three main components: 1. Geometric features are extracted from the data points (Sec. III-A). 2. The quality of randomly sampled minimum sets is evaluated. The good minimum sets are solved into coarse-hypotheses (Sec. III-B). 3. For each coarse-hypotheses, encoding its region feature, which is used for refinement and assessment to get fine-hypotheses (Sec. III-C). Then a Hypothesis-based Non-Maximum Suppression (HNMS) outputs a compound model as the final robust multi-model fitting results (Sec. III-D).

Building on the above methods, we further remodel the two-step framework into a deep learning framework as LNR. It extracts geometric features and selects good minimum sets in the coarse-level to improve efficiency. At the finelevel, LNR encodes region features for scoring and refining these hypotheses, avoiding differentiating the hypotheses and addressing model overlap.

## III. METHOD

We design a coarse-to-fine framework for robust multimodel fitting. Fig. 2 provides an overview of our proposed method, which comprises three key stages: geometric feature extraction (Sec. III-A), coarse hypothesis generation (Sec. III-B), and fine-level refinement (Sec. III-B).

## A. Geometric Feature Extraction

For a given set of input data points, denoted as $\chi = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ contaminated by noise, we first extract geometric features F that capture both local and global information. As shown in Fig. 3, raw data points undergo Fourier feature mapping [41] to enhance spatial frequency encoding. Shared multilayer perceptrons (MLPs) (implemented via 1×1 convolutions) then generate point-wise features for each data point. Subsequently, a max-pooling operation aggregates these point-wise features to generate global features, representing the holistic properties of the input data. These point-wise and global features are concatenated to form a 256-dimensional geometric feature vector F for each point. This combined representation enables subsequent modules to leverage both local point-specific details and global contextual information.

![](images/00d927a18a4bc6e104e5f84e236860f62975d8397a473dbd1e597fd58ffb2d70.jpg)  
Fig. 3. The architecture of the neural network in geometric feature extraction. Most raw data can be represented as unordered N ×c data points, making the input data N × c. The symbol ⊕ denotes element-wise vector addition, while “C” denotes vector concatenation. The network output is 256- dimensional, comprising 128-dimensional point-wise features concatenated with 128-dimensional global features.

## B. Coarse-level Hypotheses Generation Module

After extracting geometric features, LNR generates coarsehypotheses through random sampling akin to RANSAC. Specifically, LNR draws n minimum sets $\mathbb { M } \ = \ \{ M _ { j } \} _ { j = 1 } ^ { n }$ from χ, where each set $M _ { j }$ contains the minimum number of points required for model estimation (e.g., two points for line fitting). The number of minimum sets, n, is intentionally large to ensure a comprehensive exploration of potential hypotheses.

Among the sampled minimum sets M, a large number of sets are incorrect. To select a small number of good minimum sets to improve efficiency, avoiding numerous of bad minimum sets consuming resources, LNR introduces a coarse-level module to estimate confidence of each sampled minimum set. This module uses a neural network, illustrated in Fig. 4, to fully analyzes the geometric features of the minimum sets to assign confidence. Specifically, the geometric features of the points within a minimum set, $F _ { M }$ , are concatenated to form a minimum set feature. To preserve original data information, the original features are also incorporated at each point. Features from all minimum sets, $\mathbb { F } _ { M } .$ , are then processed together. Then, the LNR analyzes these features using shared MLPs and generates a confidence $\mathcal { F } _ { c o n f } ( M _ { j } )$ for each minimum set. A higher confidence indicates a minimum set of better quality. Only the top $n _ { g o o d }$ minimum sets, $\mathbb { M } _ { g o o d } ,$ with the highest confidence scores are then processed by a minimum solver S. This solver produces coarse-hypotheses $H _ { c o a r s e } = \{ S \left( M _ { j } \right) \vert M _ { j } \in \mathbb { M } _ { g o o d } , j = 1 , 2 , . . . , n _ { g o o d } \}$ . Thus, even though a large number of minimum sets are initially sampled, only a small fraction of the minimum sets will be solved, greatly improving the overall efficiency.

![](images/8cc3d639a2a102cb522fe701a029355a9f30d586f64d3b8611b08f764ed658e1.jpg)  
Fig. 4. The architecture of the neural network in coarse-level module. The input data consists of the features of each point in the n minimum sets. The size m of the minimum set varies with the task. The features of each point are concatenated with c-dimensional original features and the extracted 256-dimensional geometric features. The tensor concatenation module concatenates the features of all the points of a minimum set.

![](images/7712f421bbe78d20d80deecfce32f79d2a378ae837b9678685a4d95a66aea1e9.jpg)  
Fig. 5. Encoding region features in fine-level module. Each coarsehypothesis is encoded region features. The data points within a distance R of the hypothesis are delimited as region points. Geometric features associated with these region points are concatenated as region features, which are learned by the neural network for refinement. Learning the region features of the hypothesis rather than the hypothesis parameters avoids differentiating the sampling process and the model solvers.

## C. Fine-level Module

To refine and assess each coarse-hypothesis, LNR employs a fine-level module. This module prioritizes local information over global context, as local details are more relevant for refining individual hypotheses [18], [33], [42]. In contrast, while global information is comprehensive, it can introduce many irrelevant details that may hinder the refinement process for a specific hypothesis.

As depicted in Fig. 5, the fine-level module begins by encoding a region feature for each hypothesis. For each hypothesis, a neighbor region is defined as the area within a range R in the data point space. The definition of this range R and the distance metric used vary depending on the task. For example, in multi-line fitting, it is a Euclidean distance; for vanishing point estimation, it is a directional distance; for twoview plane segmentation, it is re-projection error; and for twoview motion, it is normalized epipolar constraint distance. The geometric features F of the data points within these neighbor regions are then encoded as the region feature $F _ { r e g i o n }$ . Since the number of points within each region can vary, regions with fewer points are padded with zeros to ensure consistent input size for a data batch. Inspired by MQ-Net [38], LNR concatenates residuals from each point in the neighbor region to the hypothesis.

![](images/a64f833bbbfe86b5493bc373c09375e8149e32cc4cf0ddc69368457c77983afb.jpg)  
Fig. 6. The architecture of the neural network in the fine-level module. The input data consists of the encoded features of the neighbor region points, forming a batch. Based on the features input to the coarse-level module, the residuals from each point in the neighbor region to its hypothesis are also concatenated. The symbol ⊕ denotes element-wise vector addition. The Refinement Head outputs only the increments of the hypothesis parameters, which need to be added to the original hypotheses to obtain the refined hypotheses.

These enriched features are then fed into a network with a refinement head and a score head. The refinement head predicts increments to the hypothesis parameters. These increments are added to the coarse-hypotheses $H _ { c o a r s e }$ to yield the refined fine-hypotheses $H _ { f i n e }$ . The score head, with a single output channel, assesses the quality of each hypothesis, providing a more robust evaluation than simple inlier counting (verified in the ablation study (Sec. IV-F). This fine-level module thus refines the initial coarse-hypotheses by focusing on local region information and learning to score and adjust hypothesis parameters.

## D. Testing Post-Processing and Training Loss

During testing, the goal is to obtain a set of non-redundant, high-quality hypotheses that represent the underlying multimodel structure. To achieve this, LNR employs Hypothesisbased Non-Maximum Suppression (HNMS) to filter the finehypotheses $H _ { f i n e }$ . HNMS, inspired by the method of Lin et al. [43] and Non-Maximum Suppression in object detection [44], iteratively selects the best hypotheses based on their fine-level scores and suppresses geometrically similar and redundant hypotheses. For the calculation of similarity between hypotheses, (1) the multi-line fitting task uses the distance between two endpoints and the angle between the two lines to compute the similarity. (2) For the vanishing point estimation task, the VPs can be projected onto the Gaussian sphere [43], where proximity on the sphere indicates similarity, as shown in Fig. 7. (3) In the two-view plane segmentation and two-view motion tasks, which are “point-to-model assignment” problems, the distance from each point to the hypothesis is encoded as a vector. The similarity between homographies and fundamental matrices is then computed based on the similarity between these vectors. The final set of selected hypotheses from HNMS constitutes the multi-model fitting result.

![](images/6fc4102ad73684cea0807b4b31a5128f74615a899d80629d53c7983577102d5f.jpg)  
Fig. 7. The similarity between vanishing points. The vanishing points are projected onto a Gaussian sphere and the distances on the sphere are used to compute the similarity between the vanishing points.

The training of LNR involves optimizing three loss functions: Confidence Loss $( \mathcal { L } _ { C o n f } )$ , Score Loss $( \mathcal { L } _ { S c o r e } )$ , and Refinement Loss $( \mathcal { L } _ { R e f i n e } ) $ , as illustrated in Fig. 8. The Confidence Loss, applied in the coarse-level module, and the Score Loss, applied in the fine-level module, both use Binary Cross Entropy (BCE) loss for binary classification. For Confidence Loss, minimum sets are labeled as “good” (1) or “bad” (0) based on the similarity of their solved hypotheses to ground truth instances $y _ { c o n f _ { j } } ~ \in ~ \{ 0 , 1 \}$ . This allows the coarse-level module to learn to select promising minimum sets without relying on explicit degeneracy tests. Therefore, Binary Cross Entropy (BCE) loss is used:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } _ { C o n f } = - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \left[ y _ { c o n f _ { j } } \cdot \log \left( \mathcal { F } _ { c o n f } ( M _ { j } ) \right) \right. } } } \\ { { \displaystyle ~ \left. + \left( 1 - y _ { c o n f _ { j } } \right) \cdot \log \left( 1 - \mathcal { F } _ { c o n f } ( M _ { j } ) \right) \right] } . } \end{array}\tag{1}
$$

For the Score Loss, the label $y _ { s c o r e _ { j } }$ also comes from the similarity between the hypothesis $h _ { j }$ and the ground truth instance $I _ { g t }$ . The difference is that the confidence of the minimum sets M is predicted in the coarse-level module. While the score of the hypothesis $h _ { j }$ is predicted in the finelevel module. Therefore, the scoring part is also a binary classification problem with BCE loss:

$$
\begin{array} { c l l } { \mathcal { L } _ { s c o r e } = - \displaystyle \frac { 1 } { n _ { g o o d } } \sum _ { j = 1 } ^ { n _ { g o o d } } [ y _ { s c o r e _ { j } } \cdot \log { ( \mathcal { F } _ { s c o r e } ( h _ { j } ) ) } } \\ { + \left( 1 - y _ { s c o r e _ { j } } \right) \cdot \log { ( 1 - \mathcal { F } _ { s c o r e } ( h _ { j } ) ) } ] . } \end{array}\tag{2}
$$

The Refinement Loss utilizes Smooth L1 loss to regress the offsets needed to refine the hypothesis parameters. For the multi-line fitting task, the predicted refinement offsets for the two endpoints of the hypothetical line segment are denoted as $\Delta x _ { 1 } , \Delta y _ { 1 } , \Delta x _ { 2 } , \Delta y _ { 2 }$ . These offsets are then applied to the coordinate $x _ { 1 _ { h } } , y _ { 1 _ { h } } , x _ { 2 _ { h } } , y _ { 2 _ { h } }$ , forming the regression residuals between ground truth and hypothesis:

$$
\begin{array} { r l } & { r _ { x _ { 1 } } = x _ { 1 _ { g t } } - ( x _ { 1 _ { h } } + \Delta x _ { 1 } ) , r _ { y _ { 1 } } = y _ { 1 _ { g t } } - ( y _ { 1 _ { h } } + \Delta y _ { 1 } ) , } \\ & { r _ { x _ { 2 } } = x _ { 2 _ { g t } } - ( x _ { 2 _ { h } } + \Delta x _ { 2 } ) , r _ { y _ { 2 } } = y _ { 2 _ { g t } } - ( y _ { 2 _ { h } } + \Delta y _ { 2 } ) . } \end{array}\tag{3}
$$

![](images/4f8ec57b67ecc406672197c494014684d8bbccb5864c7215cda61abb86047c88.jpg)  
Fig. 8. The architecture of the neural network in fine-level module. The LNR optimizes three loss functions: Confidence Loss, Score Loss, and Refinement Loss. Both Confidence Loss and Score Loss utilize Binary Cross Entropy Loss (BCE Loss). The coarse-level module identifies the most similar ground truths for loss after solving the hypotheses for the minimum sets of all sampled minimum sets. The fine-level module refines coarse-hypotheses by finding the most similar ground truths for the coarse-hypotheses. To enhance training efficiency, BCE Loss is applied only to positive samples. The Refinement Loss uses Smooth L1 Loss for regression.

The smooth L1 [45] regression loss can represent this regression:

$$
\mathcal { L } _ { R e f i n e } = \sum _ { j = 1 } ^ { n _ { g o o d } } \sum _ { q \in ( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ) } \mathrm { S m o o t h } L 1 ( r _ { q } ) .\tag{4}
$$

The regression residuals of vanishing point estimation is

$$
\begin{array} { r l } & { r _ { x } = x _ { g t } - ( x _ { h } + \Delta x ) , } \\ & { r _ { y } = y _ { g t } - ( y _ { h } + \Delta y ) , } \\ & { r _ { z } = z _ { g t } - ( z _ { h } + \Delta z ) . } \end{array}\tag{5}
$$

The smooth L1 regression loss can represent this regression:

$$
\mathcal { L } _ { R e f i n e } = \sum _ { j = 1 } ^ { n _ { g o o d } } \sum _ { q \in ( x , y , z ) } \mathrm { S m o o t h } L 1 ( r _ { q } ) .\tag{6}
$$

For the two-view plane segmentation task, the predicted offset is $\Delta H$ . Then the regression residual is:

$$
r _ { H } = H _ { g t } - ( H _ { h } + \Delta H ) .\tag{7}
$$

Thus the smooth L1 regression loss of this task can represent:

$$
\mathcal { L } _ { R e f i n e } = \sum _ { j = 1 } ^ { n _ { g o o d } } \mathrm { S m o o t h } L 1 ( r _ { H } ) .\tag{8}
$$

For the two-view motion task, it is similar to the two-view plane segmentation task. The predicted offset is $\Delta F$ . Then the regression residual is:

$$
r _ { F } = F _ { g t } - ( F _ { h } + \Delta F ) .\tag{9}
$$

Thus the smooth L1 regression loss of this task is:

$$
\mathcal { L } _ { R e f i n e } = \sum _ { j = 1 } ^ { n _ { g o o d } } \mathrm { S m o o t h } L 1 ( r _ { F } ) .\tag{10}
$$

Thus, the total loss function is:

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { C o n f } + \mathcal { L } _ { S c o r e } + \mathcal { L } _ { R e f i n e } .\tag{11}
$$

In this way, this multi-task loss can jointly optimizes all modules towards a common goal, improving overall performance. Moreover, the training of this multi-task loss function allows the loss of the fine-level module to optimize coarselevel module as well. This means that the coarse-level module produces coarse-hypotheses $H _ { c o a r s e }$ that are more consistent with the requirements of the fine-level module.

## IV. EXPERIMENTS

This section details the experiments conducted to evaluate the performance of LNR across four distinct and classic computer vision tasks: multi-line fitting, vanishing point estimation, two-view plane segmentation, and two-view motion estimation. Multi-line fitting serves as a foundational task for robust multi-model fitting algorithms. It offers a clear visualization of an algorithm to various noise scales and its ability to manage model overlap, as it can manipulate the degree of overlap between models. Vanishing point estimation, conversely, showcases algorithm performance in complex, real-world scenarios. Finally, the two-view plane segmentation and two-view motion estimation tasks assess the capability of LNR in processing correspondences between image pairs. Crucially, these two tasks represent classic instances of “point assignment to model” problems.

## A. Settings

The LNR framework undergoes training for 100 epochs. Post-processing is not incorporated during the training phase. All experiments are executed on an RTX 3090 GPU, with the framework implemented using PyTorch. The computational efficiency of LNR is O(n).

## B. Case Study 1: Multi-Line Fitting

This task evaluates basic multi-model fitting capabilities using synthetic 2D point data $\chi = \{ [ x _ { i } , y _ { i } ] | i = 1 , 2 , . . . , N \} \in$ $\mathbb { R } ^ { N \times c }$ , where $c = 2$ indicates that the data points only contain the coordinates. As a line can be defined by two points, each minimum set consists of two data points. For comparison, several classic methods are also evaluated in the same scenes with the same settings.

A synthetic dataset is generated to systematically evaluate performance, encompassing scenes with 1 to 10 lines, each scene containing 12,000 pictures. Of these, 10,000 pictures are designated for training and the remaining 2,000 for testing. All data points are constrained within a $1 \times 1$ picture. Lines, ranging from 1 to 10 per picture, are randomly generated with a minimum length of 0.3 times the longest distance of the picture. Each line segment comprises 40 to 100 points, randomly distributed and perturbed by Gaussian noise ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with σ varying between 0.007 and 0.008. Outliers, constituting 40% to 60% of the total data points, are added uniformly and randomly. Note that outliers close to line segments may become inliers due to random scattering.

TABLE I  
THE QUANTITATIVE RESULT OF MULTI-LINE FITTING. THE AVERAGE AUC@0.5<sup>◦</sup> ↑ ON 1-10 LINE SCENARIOS WITH FIVE RUNS ARE REPORTED. THE BEST INDICATORS ARE BOLDED.
<table><tr><td>Method</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Seq. RANSAC [19]</td><td>91.98 65.76 58.16 53.80 51.95 50.66 50.01 49.28 49.02 48.55</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>J-linkage [46]</td><td>93.1474.98 67.9866.87 66.02 65.6865.17 64.97 64.63 63.97</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T-linkage [39]</td><td>94.31 77.90 69.97 68.01 67.18 66.61 66.10 65.71 65.1064.29</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CONSÁC [22]</td><td>96.1491.29 86.63 84.2181.61 79.8378.30 76.3475.5174.84</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>96.73 93.41 89.99 87.71 84.89 83.32 81.47 79.91 78.59 77.51</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/47cb312c2a4c173965afb05cf13aad2e5f407847188fb7ccedce56f99da1a622.jpg)  
Fig. 9. The qualitative results of LNR on multi-line fitting. The red points represent inliers, while the black points are outliers. The ground truth is represented by green lines, while the fitted lines are represented by blue lines.

Performance assessment relies on the Area Under the Curve (AUC) metric, as detailed in Kluger et al. [22]. AUC calculation generating a recall curve for all line segments in the test set, considering a maximum angular error of $0 . 5 ^ { \circ }$ . AUC can comprehensively assess the accuracy and recall of a method.

TABLE I summarizes the quantitative performance of LNR and comparative methods across scenes with increasing line counts. LNR consistently outperforms other methods across all scenarios, demonstrating its effective feature learning for accurate instance generation. As the number of lines increases, a general performance decrease is observed across all methods, attributable to heightened model overlap and noise. However, the performance of alternative methods degrades significantly, highlighting their limitations in managing model overlap. Conversely, our proposed method exhibits a minimal performance reduction, underscoring its robust resistance to interference and effective handling of model overlap, properties attributed to its effective region feature encoding.

Qualitative evaluations, shown in Fig. 9 (a) and (b) using an existing dataset [46], illustrate that our proposed method can accurately fit lines. Moreover, Fig. 9 (c) demonstrates the efficacy of our proposed method in fitting a single line, indicating its capability in single-model robust estimation tasks. Furthermore, even with increased line counts and noise levels, as shown in Fig. 9 (d), (e), and (f), LNR maintains accurate multi-instance fitting. These qualitative and quantitative results collectively suggest that our proposed method effectively leverages region features to address the challenges posed by model overlap.

## C. Case Study 2: Vanishing Point Estimation

In this task, data points are represented as $\chi \quad =$ $\left\{ \left[ x _ { 1 } ^ { i } , y _ { 1 } ^ { i } , x _ { 2 } ^ { i } , y _ { 2 } ^ { i } \right] | i = 1 , \bar { 2 } , . . . . , N \right\}$ , where $\chi ~ \in ~ \mathbb { R } ^ { N \times c }$ and c = 4, representing the endpoints of line segments. Evaluation settings are consistent with CONSAC [22]. For comparison, several methods are compared. Notably, to test robustness and generalization, LNR is trained exclusively on the NYU-VP dataset and subsequently tested on the YUD+ and YUD datasets without further training.

TABLE II  
THE QUANTITATIVE RESULT OF VANISHING POINT ESTIMATION TASK. THE AVERAGE AUC IN % AND RELATED STANDARD DEVIATIONS (STD.) OVER FIVE RUNS ARE REPORTED. THE GENERAL ROBUST FITTING METHODS ARE COMPARED. THE “GENERAL” IS “YES” FOR A GENERAL ROBUST FITTING METHOD, AND “NO” FOR A VANISHING POINT-SPECIFIC METHOD. THE “DL” IS “YES” FOR A DEEP LEARNING METHOD. THE BEST INDICATORS ARE BOLDED AND SUB-OPTIMAL INDICATORS ARE UNDERLINED.
<table><tr><td rowspan="2">Method</td><td rowspan="2">General</td><td rowspan="2">DL</td><td colspan="4">NYU-VP</td><td colspan="4">YUD+</td><td colspan="4">YUD</td><td rowspan="2">Speed (Hz)↑</td></tr><tr><td>@5°↑</td><td>std.↓</td><td>@10°↑</td><td>std.↓</td><td> $\overline { { \ @ 5 ^ { \circ } \dag } }$ </td><td>std.↓</td><td> $\overline { { \ @ 1 0 ^ { \circ } \dag } }$ </td><td>std.↓</td><td>@5°↑</td><td>std.↓</td><td>@10°↑</td><td>std.↓</td></tr><tr><td>J-Linkage [46]</td><td>Yes</td><td>No</td><td>42.22</td><td>0.82</td><td>56.60</td><td>0.83</td><td>60.42</td><td>1.16</td><td>72.39</td><td>0.85</td><td>68.71</td><td>2.26</td><td>81.10</td><td>1.55</td><td>0.52</td></tr><tr><td>T-Linkage [39]</td><td>Yes</td><td>No</td><td>43.03</td><td>0.71</td><td>57.79</td><td>0.72</td><td>59.45</td><td>0.84</td><td>71.75</td><td>0.55</td><td>66.11</td><td>1.43</td><td>79.54</td><td>0.92</td><td>1.28</td></tr><tr><td>Prog-X [20]</td><td>Yes</td><td>No</td><td>49.28</td><td>0.20</td><td>60.73</td><td>0.22</td><td>60.03</td><td>0.71</td><td>68.54</td><td>0.70</td><td>60.11</td><td>0.78</td><td>68.48</td><td>0.74</td><td>29.42</td></tr><tr><td>CONSAC [22]</td><td>Yes</td><td>Yes</td><td>50.61</td><td>0.39</td><td>64.29</td><td>0.40</td><td>63.76</td><td>0.68</td><td>74.45</td><td>0.76</td><td>71.34</td><td>0.48</td><td>82.81</td><td>0.30</td><td>0.31</td></tr><tr><td>Lin et al. [43]</td><td>No</td><td>Yes</td><td>55.87</td><td></td><td>69.53</td><td></td><td>59.57</td><td></td><td>71.34</td><td></td><td>72.25</td><td></td><td>84.98</td><td></td><td>5.50</td></tr><tr><td>PARSAC [32]</td><td>Yes</td><td>Yes</td><td>51.68</td><td>0.12</td><td>64.61</td><td>0.13</td><td>65.49</td><td>0.35</td><td>74.76</td><td>0.42</td><td>75.92</td><td>0.15</td><td>86.39</td><td>0.09</td><td>84.13</td></tr><tr><td>Ours</td><td>Yes</td><td>Yes</td><td>57.54</td><td>0.08</td><td>71.93</td><td>0.10</td><td>67.96</td><td>0.28</td><td>77.21</td><td>0.32</td><td>76.39</td><td>0.13</td><td>87.12</td><td>0.08</td><td>10.31</td></tr><tr><td>Ours-tiny</td><td>Yes</td><td>Yes</td><td>51.52</td><td>0.11</td><td>64.46</td><td>0.12</td><td>65.53</td><td>0.31</td><td>74.91</td><td>0.34</td><td>75.92</td><td>0.14</td><td>86.40</td><td>0.10</td><td>46.96</td></tr></table>

![](images/39881bf4d990573af56db622a68b5ed45e88313c0df9c9bef5f3145412b9b360.jpg)  
Fig. 10. The qualitative results of CONSAC and LNR on the vanishing point estimation task in various scenes. The inlier line segments that belong to different vanishing points are distinguished by different colors. The major errors are indicated by red circles.

The quantitative results, presented in TABLE II, demonstrate strong average AUC performance of LNR. Notably, LNR, as a general algorithm, surpasses the performance of Lin et al. [43], a method specifically designed for vanishing point estimation. This outcome underscores the efficacy of the finelevel module of LNR in refining and assessing hypotheses. Furthermore, LNR achieves a processing speed of 10 Hz, outperforming most compared methods and delivering top-tier results, benefiting from GPU-accelerated parallel computing. Moreover, a reduced configuration, Ours-tiny, achieves competitive real-time performance, demonstrating the efficiency of the multi-task loss training framework for robust multi-model fitting. While PARSAC [32] exhibits marginally faster speed, it requires prior knowledge of the number of hypotheses, a limitation in practical scenarios. Our proposed method, relying on accurate scoring, effectively fits multiple instances through HNMS without needing to predefine the number of hypotheses.

TABLE II also reveals the sustained superior performance of LNR on the YUD+ and YUD datasets despite being trained only on NYU-VP. This highlights the exceptional generalization capability of LNR, validating the effectiveness of its region feature encoding. This suggests that our proposed method effectively classifies neighboring region points for refinement and assessment, enabling the selection of superior hypotheses from well-classified region points.

Qualitative evaluations in Fig. 10 visualize vanishing point estimation performance across diverse scenes. Comparison with CONSAC<sup>1</sup> [22], a state-of-the-art method, reveals that LNR more accurately determines line directions, resulting in fitting outcomes closer to the ground truth, especially for shorter lines. This directional accuracy contributes to the precise vanishing point estimation of LNR, consistently demonstrated across varied scenes. In contrast, CONSAC exhibits a tendency to misclassify and miss lines. The superior performance of LNR is attributed to the effective quality assessment scores and refined parameters of its generated hypotheses.

Score visualization for vanishing point estimation, presented in Fig. 12, further underscore the capability of LNR to accurately determine line directions. The hypothetical vanishing points of the high scores assigned by the LNR are similar to the locations of the ground truth. Moreover, the LNR assigns low scores to bad hypotheses, providing an excellent suppression for these hypotheses to be selected. Such a property can make it easier for the correct hypotheses to be selected.

![](images/513898e4594687739df46db6acd185e2fd15e8cbb7ac8d2703218a6e2ea9cfd5.jpg)

![](images/f2bbc7a512ab1128fe347a93de40ed900985b01e28695b879c6e6a5eba382d7d.jpg)  
Fig. 11. The qualitative results of the method of Barath et al. [21] and LNR on the two-view plane segmentation task in various scenes. Different colors indicate which of the planes the correspondences belong to. The small image and the large image are a pair of images. Their correspondences are used as input to the task. The major errors are indicated by red circles.  
(b) Ground Truth Similarities  
(c) Predicted Scores  
Fig. 12. Score visualization for the vanishing point estimation task. (a) Line segments belonging to different vanishing points are indicated by different colors. All hypothetical vanishing points are projected on a Gaussian sphere (b) and (c). In (b), the linear similarities of the hypotheses to the ground truth are indicated by different colors. In (c), the scores for the hypotheses evaluated by LNR are also indicated by different colors. The same vanishing points in (b) and (c) are marked by the same color circles. LNR assigns high scores to correct hypotheses while greatly suppressing the scores of incorrect hypotheses, better than the linear similarities.

## D. Case Study 3: Two-view Plane Segmentation

Plane segmentation in two views involves detecting homographies between two images of the same scene to segment multiple 3D planes. For training, the HEB dataset [47] is utilized, which provides 226,260 ground truth homographies. Given that a single image in HEB may contain multiple homographies, we merge HEB into a multi-homography dataset for training. Evaluation is conducted on the AdelaideRMF-H dataset [48], employing the average misclassification error (ME) and its standard deviation as evaluation metrics, consistent with Prog-X [20]. The input data, representing corresponding points across two views, is formulated as $\chi =$ $\{ \left[ x _ { 1 } ^ { i } , \bar { y _ { 1 } ^ { i } } , x _ { 2 } ^ { i } , \bar { y _ { 2 } ^ { i } } \right] | i = 1 , 2 , . . . . , N \}$ , where $\chi ~ \in ~ \mathbb { R } ^ { N \times c }$ and c = 4. LNR is evaluated against several competing methods.

TABLE III  
THE QUANTITATIVE RESULT OF TWO-VIEW PLANE SEGMENTATION AND TWO-VIEW MOTION ON ADELAIDERMF DATASET. THE AVERAGE MISCLASSIFICATION ERRORS (AVG. IN %, 5 RUNS) AND THEIR STANDARD DEVIATIONS (STD.). THE BEST INDICATORS ARE BOLDED AND SUB-OPTIMAL INDICATORS ARE UNDERLINED.
<table><tr><td rowspan="2">Method</td><td colspan="2">Two-view plane segmentation</td><td colspan="3">Two-view motion</td></tr><tr><td>avg. ↓ std. ↓</td><td>Speed (Hz)↑</td><td>avg. ↓</td><td>std. ↓</td><td>Speed (Hz)↑</td></tr><tr><td>Prog-X [20]</td><td>6.68 5.84</td><td>0.82</td><td>10.72</td><td>8.70</td><td>0.92</td></tr><tr><td>PARSAC [32]</td><td>7.24 5.21</td><td>12.31</td><td>8.73</td><td>4.35</td><td>60.13</td></tr><tr><td>CONSAC [22]</td><td>5.23 6.47</td><td>0.11</td><td></td><td></td><td></td></tr><tr><td>Barath et al. [21]</td><td>3.12 3.53</td><td>1.59</td><td>5.33</td><td>4.42</td><td>5.15</td></tr><tr><td>Ours</td><td>2.91 3.14</td><td>4.79</td><td>5.75</td><td>4.13</td><td>12.47</td></tr><tr><td>Ours-tiny</td><td>3.12 3.50</td><td>10.21</td><td>6.27</td><td>4.32</td><td>30.26</td></tr></table>

The quantitative results for two-view plane segmentation, summarized in TABLE III. LNR achieves state-of-the-art performance, where the mean ME and standard deviation are slightly better than Barath et al. [21]. Consistent with the vanishing point estimation task, the Ours-tiny configuration of LNR provides notable speed enhancements. These results underscore the effectiveness of the encoded region features in both refining and scoring hypotheses, further illustrating its proficiency in handling image-based correspondences.

Qualitative evaluations, visualized in Fig. 11, confirm the effectiveness of LNR in estimating 3D planes across diverse scenes. As one of the state-of-the-art methods, the method of Barath et $a l . ^ { 2 }$ [21] is also visualized. Compared to Barath et al. [21], LNR is closer to the ground truth. For example, in contrast to Barath et al. [21] which misses planes in columns 1 and 4 of Fig. 11, LNR successfully segments all planes.

![](images/d2c1c1356486c05d4c889f8dfeab32af550cee931209df7cff9d5029319b3c6b.jpg)  
Fig. 13. The qualitative results of the method of Barath et al. [21] and LNR on the two-view motion task in various scenes. Different colors indicate which of the fundamental matrix the correspondences belong to. The small image and the large image are a pair of images. Their correspondences are used as input to the task. The major errors are indicated by red circles.

This improvement is attributed to the ability of LNR to weight neighbors within a broader region, effectively mitigating noise interference. Furthermore, results in the last column of Fig. 11 indicate the applicability of LNR to single-model robust fitting scenarios. In these situations, Barath et al. [21] incorrectly segments a single plane into two. In contrast, LNR accurately identifies the single plane, demonstrating the compatibility of our proposed method for the single model.

## E. Case Study 4: Two-view Motion

For two-view motion estimation, LNR is trained on the HOPE-F Dataset [32], which comprises 4000 image pairs with keypoint features. Evaluation is performed using the AdelaideRMF-F dataset [48], with the average ME and its standard deviation serving as the evaluation metrics. Input data representation $\boldsymbol { \chi } \in \mathbb { R } ^ { N \times c }$ remains consistent with the two-view plane segmentation task, where c = 4. our proposed method is compared with several other methods.

Quantitative results for two-view motion estimation, presented in TABLE III. LNR achieves state-of-the-art performance. Qualitative results, shown in Fig. 13, further illustrate the superior ability of LNR to solve the “point assignment to model” problem compared to the state-of-the-art method of Barath et al.<sup>2</sup> [21]. This performance advantage is supported by LNR’s efficient classification of data points based on geometric features.

## F. Ablation Study

This section details an ablation study designed to evaluate the contribution of each module within the LNR framework. The experimental settings and evaluation metrics remain consistent with those previously established.

The generalization of LNR across different numbers of instances is shown by Ours-5 and Ours-6 in TABLE IV. Specifically, Ours-5 and Ours-6 denote models trained exclusively on scenes containing 5 and 6 lines, respectively.

TABLE IV

THE ABLATION STUDY OF MULTI-LINE FITTING. THE AVERAGE  
AUC@0.5<sup>◦</sup> ↑ ON 1-10 LINE SCENARIOS WITH FIVE RUNS ARE REPORTED.“5” AND “6” INDICATE THAT THE METHODS ARE TRAINED ONLY IN “5”AND “6” LINES SCENARIOS, TESTED DIRECTLY ON OTHER SCENARIOS.  
THE BEST INDICATORS ARE BOLDED AND SUB-OPTIMAL INDICATORS AREUNDERLINED.

<table><tr><td>Method</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Ours</td><td>96.73</td><td>93.41</td><td>89.99</td><td>87.71</td><td>84.89</td><td>83.32</td><td>81.47</td><td>79.91</td><td>78.59</td><td>77.51</td></tr><tr><td>Ours-5</td><td>92.33</td><td>91.82</td><td>90.06</td><td>87.75</td><td>84.89</td><td>82.58</td><td>80.31</td><td>78.62</td><td>77.28</td><td>76.14</td></tr><tr><td>Ours-6</td><td>91.25</td><td>91.47</td><td>89.83</td><td>87.90</td><td>85.47</td><td>83.32</td><td>81.41</td><td>79.67</td><td>78.37</td><td>77.20</td></tr><tr><td>CONSAC [22]</td><td>96.14</td><td>91.29</td><td>86.63</td><td>84.21</td><td>81.61</td><td>79.83</td><td>78.30</td><td>76.34</td><td>75.51</td><td>74.84</td></tr><tr><td>CONSAC-5 [22]</td><td>91.25</td><td>90.42</td><td>86.59</td><td>84.23</td><td>81.61</td><td>79.12</td><td>77.41</td><td>75.32</td><td>74.13</td><td>73.25</td></tr><tr><td>CONSAC-6 [22]</td><td>90.32</td><td>90.01</td><td>86.21</td><td>84.15</td><td>81.64</td><td>79.83</td><td>77.57</td><td>75.43</td><td>74.23</td><td>73.38</td></tr></table>

These models are then directly tested on scenes with different line counts. Performance metrics exceeding those of the standard Ours model are highlighted in bold. Intriguingly, Ours-5 achieves performance levels closely comparable to Ours. Even more notably, Ours-5 slightly surpasses the performance of LNR trained on scenes with 3 or 4 lines. This unexpected outcome suggests that training with a higher instance count, specifically 5 lines in this case, potentially introduces greater data variability or noise. This increased variability may foster a more robust model, enhancing the resilience of LNR to interference. Similar to Ours-5, Ours-6 also exhibits a slight performance advantage over Ours when evaluated on scenes with 4 and 5 lines. We performed the same setup on CONSAC [22]. CONSAC-5 and CONSAC-6 exhibits the same findings, but its performance gains are smaller. Collectively, these results underscore the robustness and strong generalization capabilities inherent in our proposed method.

The effects of multi-task loss training are examined in TABLE V (a). One approach considered is to directly predict hypotheses and refine them for numerous sampled minimum sets, bypassing the generation of coarse hypotheses altogether. However, removing the coarse-level module leads to a significant decline in performance. This degradation likely occurs because the network struggles to effectively learn from the substantially increased data volume associated with directly processing all minimum sets. Furthermore, the presence of excessive noisy data negatively affects the performance of the fine-level module in both scoring and refinement stages. While pre-training the coarse-level module is another possible strategy, it proves less effective than multi-task loss training. Multi-task training offers an advantage because the loss signals from the fine-level module are propagated back to optimize the coarse-level module, a mechanism observed in prior researches [49], [50]. This back-propagation enables the coarselevel module to prioritize the selection of coarse hypotheses that are more compatible with the fine-level module, rather than simply adhering to pre-defined labels. This synergistic interaction between the coarse-level and fine-level modules is facilitated by their shared input of encoded region features during training.

TABLE V  
THE ABLATION STUDY RESULTS OF LNR FOR VANISHING POINT ESTIMATION ON THE NYU-VP DATASET. THE AUC IN % IS REPORTED. THE SETTINGS ARE CONSISTENT WITH TABLE II. THE BEST INDICATORS ARE BOLDED.
<table><tr><td></td><td>Method</td><td>@5°↑</td><td>@10°↑</td></tr><tr><td>(a)</td><td>Ours (w/o coarse-level) Ours (pre-trained coarse-level) Ours (full, multi-task loss trained coarse-level)</td><td>49.26 55.14 57.54</td><td>63.12 69.42 71.93</td></tr><tr><td>(b)</td><td>Ours (w/o fine-level, coarse-level +inlier count) Ours (w/o fine-level, coarse-level +inlier count+inlier refinement) Ours (coarse-level +inlier count+fine-level refinement) Ours (coarse-level +fine-level count+inlier refinement)</td><td>44.21 49.73 52.41 53.94</td><td>58.68 63.61 67.21 68.62</td></tr><tr><td>(c)</td><td>Ours (full, coarse-level +fine-level count+fine-level refinement) CONSAC-(Time:3.19s) Ours (full, LNR)-(Time:0.09s) CONSAC+LNR-(Time:3.32s)</td><td>57.54 49.85 57.54 57.63</td><td>71.93 64.07 71.93 72.03</td></tr><tr><td>(d)</td><td>Ours (PointNet without Embedding) Ours (PointNet+Basic Embedding) Ours (full, PointNet+Fourier Embedding)</td><td>57.31 57.42 57.54</td><td>71.69 71.81 71.93</td></tr><tr><td>(e)</td><td>Ours (LSQ refinement) Ours (full, NN refinement)</td><td>53.89 57.54</td><td>68.57 71.93</td></tr></table>

The different coarse-to-fine approaches are evaluated in TABLE V (b). This evaluation involves substituting the finelevel module with more conventional methodologies. The results indicate that employing a simple inlier count as a replacement yields inferior performance. This finding reinforces the conclusion from MQ-Net that “consensus maximization does not favor the best model”. Moreover, using inlier refinement as an alternative to the fine-level module also underperforms. Inlier refinement appears to be more susceptible to variations in data quality. These findings collectively validate the effectiveness and necessity of the dedicated fine-level module within the LNR framework.

Combining the different sampling methods are presented in TABLE V (c). LNR, in its standard configuration, employs an unbiased random sampling strategy to generate a large number of minimum sets, contrasting with probability-based sampling techniques like CONSAC. Substituting the random sampling with CONSAC sampling leads to improved performance. However, this performance gain comes at the cost of increased computational time, making the overall process considerably slower.

The effects of Fourier feature mapping is tested in

TABLE V (d). Positional embedding, through Fourier feature mapping, converts geometric information into highdimensional vectors. This transformation empowers the neural network to more effectively learn and represent subtle, high-frequency geometric details. Consequently, the network becomes better equipped to discern and learn the nuanced differences between individual data points.

Different hypothesis refinement approaches are tested in the TABLE V (e). The neural network refinement outperforms LSQ refinement. This is because the neural network effectively identifies and discards noisy points, enhancing the use of significant data points.

The effects of Fourier feature mapping is tested in TABLE V (d). The results demonstrate that neural networkbased refinement surpasses traditional Least Squares (LSQ) refinement. This performance advantage can be attributed to the ability of neural network to effectively identify and disregard noisy data points, thereby prioritizing and leveraging the more informative and significant data points during the refinement process.

## V. DISCUSSION

Traditional multi-model fitting methods, grounded in numerical computation, offer inherent interpretability and clearly defined application scopes. However, these approaches often struggle to fully exploit the geometric features of the data, potentially limiting their performance. Conversely, while deep learning techniques have demonstrated remarkable performance gains, end-to-end methods [43], [51] that directly map input data points to multi-models can operate as “black boxes”, lacking transparency and potentially hindering their reliable deployment in practical applications.

In contrast, our proposed method seeks to bridge this gap by enhancing performance while maintaining interpretability. LNR leverages neural networks to effectively analyze the geometric features within the data. Crucially, for interpretability, LNR adopts the established two-step strategy: first, it gathers high-quality minimum sets of data points. Second, it applies minimum solvers to generate multi-models. This is achieved by training LNR to learn from data points within neighboring regions, rather than directly from hypotheses. This strategic choice naturally avoids the need to explicitly differentiate the sampling process and the model solvers, thus integrating traditional frameworks into the learning pipeline. Furthermore, Experiments indicate that learning-based hypothesis evaluation is significantly more efficient than most traditional rule-based evaluation techniques. Therefore, we hope to introduce a new idea for addressing multi-model fitting problems in learning fashions.

In addition, LNR can be easily transferred to various robust multi-model fitting tasks due to the following reasons:

• Our proposed method accepts generic, unordered data points as input, encompassing various data modalities including image correspondences, 3D point clouds, and 2D pixel points, all uniformly represented as point sets.

• The network input focuses solely on the geometric features of these data points. While hypotheses and neighbor regions guide the utilization of these features, the loss function operates on the geometric features rather than the hypotheses. As a result, Our proposed method maintains compatibility with a broad spectrum of hypothesis solvers and sampling processes, whether differentiable or nondifferentiable.

## VI. CONCLUSION

In this paper, we introduce LNR, a coarse-to-fine framework of Learning Neighbor Regions (LNR) to achieve robust multimodel fitting. For the inefficiency caused by sampling numerous minimum sets, LNR introduces a coarse-level module to select good minimum sets by analyzing geometric features, solving the coarse-hypotheses. For the model overlap and nondifferentiable problem, LNR encodes neighbor region features for each coarse-hypothesis to refine and score these hypotheses individually. Since the region features consist of data points rather than hypotheses, the LNR avoids differentiating the sampling process and the model solvers. Furthermore, the multi-task loss function optimizes the coarse-to-fine process towards a consistent goal. In the experiments, the multi-line fitting task highlights the powerful generalization of LNR across different scenes. Notably, LNR excels in vanishing point estimation, two-view plane segmentation and two-view motion tasks, achieving state-of-the-art performance. LNR can effectively process correspondences and address the “point-tomodel assignment” problem. Significantly, LNR framework is not task-specific. It can be easily transferred to various robust multi-model fitting tasks.

## REFERENCES

[1] L. Kong, S. Xie, H. Hu, L. X. Ng, B. Cottereau, and W. T. Ooi, “Robodepth: Robust out-of-distribution depth estimation under corruptions,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2024.

[2] Z. Jian, Z. Lu, X. Zhou, B. Lan, A. Xiao, X. Wang, and B. Liang, “Putn: A plane-fitting based uneven terrain navigation framework,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 7160–7166.

[3] Z. Jian, Z. Liu, H. Shao, X. Wang, X. Chen, and B. Liang, “Path generation for wheeled robots autonomous navigation on vegetated terrain,” IEEE Robotics and Automation Letters, vol. 9, no. 2, pp. 1764– 1771, 2023.

[4] B. Li, D. Zou, Y. Huang, X. Niu, L. Pei, and W. Yu, “Textslam: Visual slam with semantic planar text features,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 1, pp. 593–610, 2023.

[5] D. Rempe, T. Birdal, A. Hertzmann, J. Yang, S. Sridhar, and L. J. Guibas, “Humor: 3d human motion model for robust pose estimation,” in Proceedings of the IEEE/CVF international conference on computer vision (ICCV), 2021, pp. 11 488–11 499.

[6] H. Guo, S. Peng, H. Lin, Q. Wang, G. Zhang, H. Bao, and X. Zhou, “Neural 3d scene reconstruction with the manhattan-world assumption,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5511–5520.

[7] J. Liu, P. Ji, N. Bansal, C. Cai, Q. Yan, X. Huang, and Y. Xu, “Planemvs: 3d plane reconstruction from multi-view stereo,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8665–8675.

[8] W. Yin, J. Zhang, O. Wang, S. Niklaus, S. Chen, Y. Liu, and C. Shen, “Towards accurate reconstruction of 3d scene shape from a single monocular image,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 6480–6494, 2022.

[9] D. Jin, S. Karmalkar, H. Zhang, and L. Carlone, “Multi-model 3d registration: Finding multiple moving objects in cluttered point clouds,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 4990–4997.

[10] Q. Yang, Y. Ma, L. Li, Y. Gao, J. Tao, Z. Huang, and R. Jiang, “A fast vanishing point detection method based on row space features suitable for real driving scenarios,” Scientific reports, vol. 13, no. 1, p. 3088, 2023.

[11] S. Liu, Y. Zhou, and Y. Zhao, “Vapid: A rapid vanishing point detector via learned optimizers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12 859–12 868.

[12] W. Yang, B. Fang, and Y. Y. Tang, “Fast and accurate vanishing point detection and its application in inverse perspective mapping of structured road,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 48, no. 5, pp. 755–766, 2016.

[13] C.-K. Chang, J. Zhao, and L. Itti, “Deepvp: Deep learning for vanishing point detection on 1 million street view images,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 4496–4503.

[14] Z. Shi, N. Carlini, A. Balashankar, L. Schmidt, C.-J. Hsieh, A. Beutel, and Y. Qin, “Effective robustness against natural distribution shifts for models with different training data,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2024.

[15] X. Jiang and J. Ma, “Robust Model Reasoning and Fitting via Dual Sparsity Pursuit,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2024.

[16] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,” IEEE Transactions on Visualization and Computer Graphics, 2024.

[17] C. Ebner, S. Mori, P. Mohr, Y. Peng, D. Schmalstieg, G. Wetzstein, and D. Kalkofen, “Video see-through mixed reality with focus cues,” IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 5, pp. 2256–2266, 2022.

[18] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.

[19] E. Vincent and R. Laganiere, “Detecting planar homographies in an´ image pair,” in Proceedings of the 2nd International Symposium on Image and Signal Processing and Analysis (ISPA)., 2001, pp. 182–187.

[20] D. Barath and J. Matas, “Progressive-x: Efficient, anytime, multimodel fitting algorithm,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 3780–3788.

[21] D. Barath, D. Rozumnyi, I. Eichhardt, L. Hajder, and J. Matas, “Finding Geometric Models by Clustering in the Consensus Space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 5414–5424.

[22] F. Kluger, E. Brachmann, H. Ackermann, C. Rother, M. Y. Yang, and B. Rosenhahn, “CONSAC: Robust multi-model fitting by conditional sample consensus,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 4634– 4643.

[23] X. Lu, Y. Zhou, and S. Shen, “Event-based motion segmentation by cascaded two-level multi-model fitting,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 4445–4452.

[24] L. Magri, F. Leveni, and G. Boracchi, “Multilink: Multi-class structure recovery via agglomerative clustering and model selection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1853–1862.

[25] H. Isack and Y. Boykov, “Energy-based geometric multi-model fitting,” International journal of computer vision, vol. 97, no. 2, pp. 123–147, 2012.

[26] P. Amayo, P. Pinies, L. M. Paz, and P. Newman, “Geometric multi-´ model fitting with a convex relaxation algorithm,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 8138–8146.

[27] D. Barath and J. Matas, “Multi-class model fitting by energy minimization and mode-seeking,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 221–236.

[28] H. Wang, G. Xiao, Y. Yan, and D. Suter, “Mode-seeking on hypergraphs for robust geometric model fitting,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 2902–2910.

[29] M. Farina, L. Magri, W. Menapace, E. Ricci, V. Golyanik, and F. Arrigoni, “Quantum multi-model fitting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 640–13 649.

[30] Y. Feng, S. Ji, Y.-S. Liu, S. Du, Q. Dai, and Y. Gao, “Hypergraphbased multi-modal representation for open-set 3d object retrieval,” IEEE

Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 4, pp. 2206–2223, 2023.

[31] E. Levinkov, A. Kardoost, B. Andres, and M. Keuper, “Higher-order multicuts for geometric model fitting and motion segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 608–622, 2022.

[32] F. Kluger and B. Rosenhahn, “PARSAC: Accelerating Robust Multi-Model Fitting with Parallel Sample Consensus,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024, pp. 2804–2812.

[33] O. Chum, J. Matas, and J. Kittler, “Locally optimized RANSAC,” in Joint Pattern Recognition Symposium, 2003, pp. 236–243.

[34] P. H. Torr, S. J. Nasuto, and J. M. Bishop, “Napsac: High noise, high dimensional robust estimation-it’s in the bag,” in British Machine Vision Conference (BMVC), 2002, p. 3.

[35] M. Zuliani, C. S. Kenney, and B. Manjunath, “The multiransac algorithm and its application to detect planar homographies,” in IEEE International Conference on Image Processing 2005, vol. 3, 2005, pp. III–153.

[36] P. Purkait, T.-J. Chin, A. Sadri, and D. Suter, “Clustering with hypergraphs: the case for large hyperedges,” IEEE transactions on pattern analysis and machine intelligence, vol. 39, no. 9, pp. 1697–1711, 2016.

[37] T.-J. Chin, J. Yu, and D. Suter, “Accelerated hypothesis generation for multistructure data via preference analysis,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 4, pp. 625–638, 2011.

[38] D. Barath, L. Cavalli, and M. Pollefeys, “Learning to find good models in RANSAC,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 15 744–15 753.

[39] L. Magri and A. Fusiello, “T-linkage: A continuous relaxation of jlinkage for multi-model fitting,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 3954– 3961.

[40] E. Brachmann and C. Rother, “Neural-guided RANSAC: Learning where to sample model hypotheses,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (CVPR), 2019, pp. 4322– 4331.

[41] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” Advances in neural information processing systems (NeurIPS), vol. 33, pp. 7537–7547, 2020.

[42] D. Barath, J. Noskova, M. Ivashechkin, and J. Matas, “MAGSAC++, a fast, reliable and accurate robust estimator,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2020, pp. 1304–1312.

[43] Y. Lin, R. Wiersma, S. L. Pintea, K. Hildebrandt, E. Eisemann, and J. C. van Gemert, “Deep vanishing point detection: Geometric priors make dataset variations vanish,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 6103– 6113.

[44] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2016, pp. 779–788.

[45] R. Girshick, “Fast R-CNN,” in Proceedings of the IEEE international conference on computer vision (ICCV), 2015, pp. 1440–1448.

[46] R. Toldo and A. Fusiello, “Robust multiple structures estimation with j-linkage,” in 10th European Conference on Computer Vision (ECCV), 2008, pp. 537–547.

[47] D. Barath, D. Mishkin, M. Polic, W. Forstner, and J. Matas, “A¨ Large-Scale Homography Benchmark,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 21 360–21 370.

[48] H. S. Wong, T.-J. Chin, J. Yu, and D. Suter, “Dynamic and hierarchical multi-structure geometric model fitting,” in 2011 International Conference on Computer Vision (ICCV), 2011, pp. 1044–1051.

[49] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “LoFTR: Detectorfree local feature matching with transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2021, pp. 8922–8931.

[50] J. Ma, Z. Zhao, X. Yi, J. Chen, L. Hong, and E. H. Chi, “Modeling task relationships in multi-task learning with multi-gate mixture-of-experts,” in Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 2018, pp. 1930–1939.

[51] T. Chen, X. Ying, J. Yang, R. Wang, R. Guo, B. Xing, and J. Shi, “VPDETR: End-to-End Vanishing Point DEtection TRansformers,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 2, 2024, pp. 1192–1200.