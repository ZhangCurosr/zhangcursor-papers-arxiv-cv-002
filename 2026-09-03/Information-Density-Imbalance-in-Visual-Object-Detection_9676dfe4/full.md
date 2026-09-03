# Information Density Imbalance in Visual Object Detection

Ziwei Zhao , Yanxi Lu, Yuwei Hu, Shiyang Su, Mingxuan Wang, Chenyue Zhou, Jiayi Chen, Hehan Li, Xiaoshuai Hao, Andi Zhang, Yanbiao Ma\*

Abstract—In object detection, the number of instances is typically used to determine whether a dataset exhibits a long-tailed distribution, implicitly assuming that the model will perform poorly on categories with fewer instances. This assumption has led to extensive research on category bias in datasets with imbalanced instance numbers. However, even in datasets where instance numbers are relatively balanced, models still exhibit category bias, indicating that instance count alone cannot explain this phenomenon. In this work, we first introduce the concept and measurement of information density. We then observe a significant negative correlation between a category’s information density and its accuracy, and we investigate how the training process impacts this relationship. Empirical studies suggest that information density imbalance may be a potential source of category bias. To preliminarily validate the potential of information density, we made simple improvements to three advanced object detection loss functions using this concept. Experiments on the Pascal VOC, COCO-LT, and LVIS datasets demonstrate that information density can significantly reduce model bias while effectively enhancing the overall performance of existing loss functions. This study provides a new perspective for understanding the generalized bias phenomenon in object detection models and offers new tools for designing fairer loss functions and training strategies.

✦

Index Terms—Fairness of DNNs, Model Bias, Unbalanced Object Detection, Data-Centric AI, Visual Datasets.

## 1 INTRODUCTION

O <sup>BJECT</sup> <sup>detection</sup> <sup>is</sup> <sup>a</sup> <sup>key</sup> <sup>task</sup> <sup>in</sup> <sup>the</sup> <sup>field</sup> <sup>of</sup> <sup>computer</sup>vision, widely applied in various practical scenarios such vision, widely applied in various practical scenarios such as autonomous driving, medical image analysis, and surveillance systems [1], [2], [3], [4]. Despite significant progress in recent years, the issue of dataset imbalance remains a persistent challenge [5], [6], [7], [8], [9]. Traditional assumptions often attribute the poor performance of models on certain categories to the scarcity of instances, a view that has been widely accepted and has led to many instance-quantity-based improvement methods [10], [11], [12], [13], [14], [15], [16]. However, our quantitative research suggests that this assumption may be overly simplistic. As shown in Fig. 1, empirical analysis on the Pascal VOC dataset [17] indicates that the Pearson correlation coefficient between the number of category instances and corresponding detection accuracy is only -0.171. Such a weak correlation prompts us to hypothesize that other deeper factors may be influencing the performance of object detection models.

In the field of image classification, multiple studies [14], [18], [19], [20], [21] have observed that classification models still exhibit significant bias towards certain categories even when the sample quantities are perfectly balanced. To uncover the underlying mechanisms, [18], [19], [21] have found a significant correlation between the geometric properties of perceptual manifolds in deep neural networks and model bias. [20] discovered that the decay rate of the feature spectrum of the embedding distribution could be a source of model bias. However, in the task of object detection, category bias induced by non-long-tailed distributions has been relatively overlooked. Therefore, we aim to explore and quantify the deeper factors within datasets that influence model bias, beyond merely the number of instances.

![](images/4b6e3a4bd349a64f78b8cc69455bff88b995234c236e7f3b01e866129ceaf952.jpg)  
Fig. 1. The left vertical axis represents the number of instances per class. The right vertical axis shows the performance of Faster R-CNN with an R-50-FPN backbone across all classes, trained on the Pascal VOC dataset. The model was trained using the settings described in Section 5.2. The red text box displays the Pearson correlation coefficient between class performance and the number of instances.

Intuitively, if a category has highly diverse instances, it becomes more difficult for the model to learn this category. Conversely, if a category’s diversity is low, the model can learn it well with fewer instances. Additionally, a more intuitive factor is the size of the instances: when a category contains instances of larger size, the model can detect them more easily [22]. However, quantifying instance diversity is the first challenge in verifying this intuition. Inspired by the neural manifolds generated during visual processing in the human brain [23], [24], [25], we propose methods for measuring instance diversity and calculating the total area of instances in Section 3.1. Our findings indicate that both instance diversity and the total area of instances are highly correlated with category accuracy, validating our hypothesis.

Furthermore, we introduce the concept of information density, defined as the ratio of instance diversity to the total area of instances (Section 3.2). Information density represents the amount of information per unit area, reflecting the concentration of information. Surprisingly, there is a significant negative correlation between information density and category detection accuracy (see Fig. 2 and Fig. 3). Moreover, the correlation between information density and category accuracy strengthens throughout training (see Fig. 4), indicating that current optimization objectives overlook the model bias induced by imbalances in information density. Information density effectively quantifies the combined impact of category diversity and instance area, providing a new perspective for understanding the generalized bias phenomenon in object detection models. This concept also offers a theoretical foundation for designing more equitable loss functions and training strategies.

To demonstrate the potential of information density, we improved three advanced loss functions for object detection in Section 4.1: Seesaw Loss [26], Equalized Focal Loss [27], and C2AM Loss [28]. Additionally, to dynamically update information density during training, we proposed a novel low-cost training strategy in Section 4.2. Comprehensive evaluations on non-longtailed (Pascal VOC [17]) and long-tailed (COCO-LT [13], LVIS v1.0 [15]) datasets show that incorporating information density into the loss function significantly alleviates model bias and improves overall performance. In summary, our research emphasizes the importance of understanding the factors influencing object detection performance beyond instance quantity. Future research can further explore how to optimize model training based on information density to enhance model performance in handling categories with high diversity and small instance areas.

## 2 RELATED WORK

In recent years, significant progress has been made in object detection technology. Mainstream research has focused on developing general object detection models, while a few studies have been proposed to improve overall performance by mitigating bias in object detection models [26], [27], [28], [29], [30], [31], with Focal Loss [32] and Seesaw Loss [26] being two typical examples. To specifically address the issue of model bias, researchers have introduced the long-tailed object detection task. This task has both pros and cons; while it artificially constructs long-tailed object detection datasets to amplify model bias, providing a challenging scenario [13], [15], it simultaneously leads researchers to overlook the bias exhibited by models in more balanced object detection scenarios. This widely observed model bias cannot be fully explained by the imbalance in instance numbers, and there is a lack of research on this phenomenon in the field of object detection.

In the following, we will first introduce the current state and progress of research in the field of long-tailed object detection, and then discuss existing methods for measuring the difficulty of class learning. In our discussion of class difficulty measurement, we will comprehensively review relevant advancements, not limited to object detection tasks.

## 2.1 Long-Tailed Object Detection

In the research of long-tailed object recognition, the main approaches include data re-sampling [33], [34], specialized loss function design [14], [35], architectural improvements [36], [37], decoupled training [38], [39], and data augmentation [40], [41].

Data re-sampling is a common method to address imbalanced datasets by increasing the sampling frequency of tail class samples to balance the data distribution [33]. Common re-sampling strategies include Class-aware sampling [42] and Repeat factor sampling (RFS) [15]. These methods can be employed at different stages of training to achieve a multi-stage training process. Specialized loss function design is another technical approach to tackling long-tailed challenges. For instance, EQL [29] reduces suppression on tail classes by truncating the negative gradients from head classes. The subsequent EQLv2 [30] further improves this approach through a gradient balancing mechanism. Other methods, such as Seesaw Loss [26], Equalized Focal Loss [27], ACSL [43], and LOCE [44], reduce excessive suppression of tail classes by dynamically adjusting classification logits or suppressing overconfident scores. C2AM [28] observed that the severe imbalance in weight norms across classes leads to pathological decision boundaries, and therefore proposes learning fairer decision boundaries by adjusting the ratio of weight norms.

Current research mainly focuses on these two directions. In addition, module improvement emphasizes modifying the structure of detectors to address long-tailed distribution issues. For example, BAGS [45] and Forest R-CNN [46] mitigate the impact of head classes on tail classes by grouping all classes based on valuable prior knowledge. Decoupled training [39] has found that long-tailed distributions do not significantly affect the learning of high-quality features, thus some methods freeze the feature extractor parameters during the classifier learning phase, adjusting only the classifier [13], [47], [48]. Data augmentation, as a means of introducing additional sample variability, has been shown to provide further improvements in long-tailed detection tasks. Recently proposed methods such as Simple Copy-Paste [49], FDC [47], FASA [50], and FUR [7] supplement the insufficiency of tail-class samples by performing data augmentation in both image and feature spaces [40].

## 2.2 Methods for Measuring Class Difficulty

The study of class difficulty is most relevant to our work. Most research addressing class bias has focused on scenarios with sample imbalance, where rebalancing strategies based on sample size can be somewhat effective. However, recent studies have reported that even when sample sizes are perfectly balanced, classification models still exhibit significant performance disparities across different classes. Investigating the root causes of model bias in scenarios where sample sizes are balanced is crucial for improving model fairness and understanding learning mechanisms. However, research on this issue is still limited. From a geometric perspective, DSB [18], CR [19], and IDR [21] conceptualize the data classification process as the disentangling and separating of different perceptual manifolds. These three studies respectively reveal that the geometric properties of perceptual manifolds—volume, curvature, and intrinsic dimensionality—are significantly correlated with class performance. [20] discovered that differences in the spectral features of classes could be a source of class bias. Unfortunately, in the field of object detection, there has been no research exploring the underlying causes of model bias. Our work is the first to directly report on the widespread bias present in object detection models and to attempt to explore the potential mechanisms underlying this bias.

## 3 INFORMATION DENSITY IMBALANCE

In Section 3.1, we first introduce how to calculate the total area of instances in a category, and then propose a measurement to estimate the diversity of instances in a category. In Section 3.2, we present two findings: the significant negative correlation between instance diversity and category average precision, and the significant positive correlation between the total area of instances and category accuracy. Based on these observations, we propose the concept of information density to unify these two factors. Finally, in Section 3.3, we explore the impact of learning on the correlation between information density and category average precision.

## 3.1 Measurement of Instance Diversity and Total Area

## Definition and Measurement ofTotal Area ofInstances

The total area of instances in a category can be obtained by summing the ground truth areas of all instances. Given a set of ground truth bounding boxes $B ~ = ~ \{ B _ { 1 } , B _ { 2 } , \ldots , B _ { m } \}$ for all instances in a category, where $B _ { i }$ represents the ground truth bounding box of instance $i ,$ the area $A ( B _ { i } )$ of each bounding box can be calculated using the following formula:

$$
A ( B _ { i } ) = ( x _ { m a x } - x _ { m i n } ) \cdot ( y _ { m a x } - y _ { m i n } ) ,
$$

where $( x _ { m a x } , y _ { m a x } )$ and $( x _ { m i n } , y _ { m i n } )$ are the coordinates of the top-right and bottom-left corners of the bounding box $B _ { i } ,$ respectively. The total area of the category can be obtained by summing the areas of all instances:

$$
A ( B ) = \sum _ { i = 1 } ^ { m } A ( B _ { i } ) .
$$

Next, we introduce how to measure the diversity of instances within a category.

## Definition and Measurement ofInstance Diversity

In the nervous system, when neurons are stimulated by physical features of the same category, a perceptual manifold is formed [24], [51]. The formation of the perceptual manifold helps the nervous system differentiate and process objects with different features within the same category. Recent studies [18], [21], [52] have shown that the response of deep neural networks to images is similar to human vision, following the manifold distribution law, meaning that the embeddings of a category of natural images are distributed near a low-dimensional perceptual manifold embedded in a high-dimensional space. Continuous sampling along one dimension of the manifold corresponds to continuous changes in physical features. Therefore, the volume of the perceptual manifold mapped by deep neural networks can measure the diversity of a category.

Given a category, suppose the embeddings of all instances are $X = [ x _ { 1 } , \ldots , \bar { x } _ { m } ] \in \bar { \mathbb { R } } ^ { p \times m }$ , where $p$ is the dimension of the embeddings, and m is the number of instances. It is important to note that the embeddings used to calculate diversity need to be extracted from the classification module of the object detection model, not the regression module. The volume of the perceptual manifold corresponding to the embedding set $X$ can be measured by the span of the subspace spanned by the eigenvectors of the covariance matrix of X. However, directly calculating the sample covariance matrix in high-dimensional situations is often inaccurate, so we use the Ledoit-Pech´ e nonlinear shrinkage method [´ 53] to improve the estimation results.

First, the sample covariance matrix is estimated as:

$$
\Sigma ( X ) = \frac { 1 } { m } ( X - \bar { X } ) ( X - \bar { X } ) ^ { T } ,
$$

where $\bar { X }$ is the mean vector of the data:

$$
{ \bar { X } } = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } x _ { i } .
$$

Next, the sample covariance matrix $\Sigma ( X )$ is decomposed into eigenvalues $\lambda _ { i } , i = 1 , . . . , p$ and the corresponding eigenvector matrix $V { : }$

$$
\Sigma ( X ) = V \Lambda V ^ { T } ,
$$

where $\Lambda$ is a diagonal matrix with the eigenvalues $\lambda _ { i } , i = 1 , \ldots , p$ on the diagonal.

According to the Marcenko-Pastur distribution in random ˇ matrix theory [54], a nonlinear transformation is applied to the eigenvalues. Specifically, the parameter $c = p / m$ is calculated of the Marcenko-Pastur distribution. According to the Marˇ cenko-ˇ Pastur distribution, the theoretical support interval of the eigenvalues is

$$
\lambda _ { \pm } = ( 1 \pm { \sqrt { c } } ) ^ { 2 } .
$$

Then, the following nonlinear transformation is applied to each eigenvalue:

$$
d _ { i } = \operatorname* { m a x } ( \lambda _ { i } , \lambda _ { - } ) , i = 1 , \ldots , p .
$$

The purpose of this nonlinear transformation is to avoid the undue influence of excessively small eigenvalues on the covariance matrix estimation, ensuring that all eigenvalues are at least above the theoretical lower bound, thus improving the stability of the estimation. Finally, the covariance matrix is reconstructed using the nonlinearly transformed eigenvalues $d _ { i }$ and the eigenvector matrix $V { : }$

$$
\Sigma _ { L P } = V d i a g ( d _ { i } , i = 1 , \dots , p ) V ^ { T } .
$$

Furthermore, we formally define the instance diversity of a category as

$$
V o l ( X ) = \frac { 1 } { 2 } \log _ { 2 } \operatorname * { d e t } ( I + \Sigma _ { L P } ) ,
$$

where det $\left( I + \Sigma _ { L P } \right)$ is the determinant of the matrix $I + \Sigma _ { L P }$ The determinant, equivalent to the product of its eigenvalues, represents the span of the subspace spanned by the eigenvectors. The use of the logarithmic transformation is to enhance the stability and smoothness of the computation results. This method of measuring instance diversity provides a tool for the quantitative study of the generalized imbalance phenomenon in object detection models.

## 3.2 Imbalance of Information Density and Model Bias

We trained standard Faster R-CNN [55] models with R-50-FPN and R-101-FPN [22], [56] backbones on Pascal VOC [17] and MS COCO [57], using cross-entropy (CE), SeeSaw, and Focal loss functions. We used each instance’s ground truth as the Region of Interest (ROI) and extracted classification features. We then computed instance diversity for each category and analyzed the Pearson correlation between instance diversity and category average precision. The experimental results, as shown in Fig. 2, indicate a significant negative correlation between category diversity and accuracy, suggesting that the model’s learning difficulty increases with higher intra-category variability. Conversely, we found a significant positive correlation between the total area of instances in a category and detection accuracy. Categories with larger instance areas are more accurately detected by the model. These results indicate that intra-category variability and instance area significantly impact the model’s learning performance.

![](images/5f008984ac912824e5d8f5ba322c7f01ed26d912a905f5dfd0e5d5530b943463.jpg)

![](images/26d3928a5746e261636652d7e6ece7576c6fa7762fbebc71d09f0e1846fd553d.jpg)  
Fig. 2. The Pearson correlation coefficients between the average precision (AP) per category on the Pascal VOC dataset and the following factors: instance diversity, total instance area, instance count, and information density. Note that all four factors are measured at the category level.

Algorithm 1 Information Density Calculation   
1: Input: Set of bounding boxes $B ~ = ~ \{ B _ { 1 } , B _ { 2 } , \ldots , B _ { m } \}$   
feature embeddings $X = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } ] \in \mathbb { R } ^ { p \times m }$   
2: Compute Total Area $A ( B ) { \mathrm { : } }$   
3: for i = 1 to m do   
4: $A ( B _ { i } )  ( x _ { \operatorname* { m a x } } ^ { i } - x _ { \operatorname* { m i n } } ^ { i } ) \times ( y _ { \operatorname* { m a x } } ^ { i } - y _ { \operatorname* { m i n } } ^ { i } )$   
5: end for   
6: $\begin{array} { r } { A ( B )  \sum _ { i = 1 } ^ { m } A ( B _ { i } ) } \end{array}$   
7: Compute Diversity $V o l ( X ) { \mathrel { : } }$   
8: Compute sample covariance matrix: $\Sigma ( X ) \  \ { \frac { 1 } { m } } ( X \ -$   
$\bar { X } ) ( \dot { X } - \bar { X } ) ^ { T }$ , where $\textstyle { \bar { X } } = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } x _ { i }$   
9: Perform eigenvalue decomposition on $\Sigma ( X ) { \mathrm { : } }$   
$\Sigma ( X ) = \bar { V } \Lambda V ^ { T }$   
10: Compute Marcenko-Pastur distribution parameter:ˇ   
$\textstyle { \mathcal { s } } \gets { \frac { p } { m } }$   
11: Compute theoretical support interval: $\lambda _ { \pm }  ( 1 \pm \sqrt { c } ) ^ { 2 }$   
12: for $i = 1 \mathrm { t o } p$ do   
13: $d _ { i } \gets \operatorname* { m a x } ( \lambda _ { i } , \lambda _ { - } )$   
14: end for   
15: Reconstruct covariance matrix with transformed eigenvalues:   
$\Sigma _ { \mathrm { L P } }  V \mathrm { d i a g } ( d _ { i } ) V ^ { T }$   
16: Compute diversity: $\begin{array} { r } { V o l ( X ) \gets \frac { 1 } { 2 } \log _ { 2 } \operatorname* { d e t } ( I + \Sigma _ { \mathrm { L P } } ) } \end{array}$   
17: Compute Information Density $I { \boldsymbol { : } }$   
18: $I \gets \frac { \prime { v o l } ( X ) } { A ( B ) }$   
19: Output: I

Further, we combined these two factors, instance diversity and total area, to define information density:

$$
I = \frac { V o l ( X ) } { A ( B ) } ,
$$

where $X \ = \ [ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } ] \ \in \ \mathbb { R } ^ { p \times m }$ represents the set of embeddings for all instances in a category, and $V o l ( X )$ denotes instance diversity. A(B) represents the total area of all ground truth bounding boxes. Clearly, information density I represents the amount of information per unit area, reflecting the richness of information. Higher information density implies that the model needs to process more diverse features within a relatively small region, increasing the complexity and difficulty of learning.

![](images/552cd0d894d03ed8989e04f8572c2110d3b766e57e8cf2b28e856b0ff2839a2b.jpg)  
Fig. 3. Taking Faster R-CNN trained with CE loss as an example, the horizontal axis represents the average precision of each category, and the vertical axis represents the information density of each category. The blue text box reports the Pearson correlation coefficient between the average precision of each category and its information density.

We were surprised to find that information density I exhibits an even more significant negative correlation with category average precision (see Fig. 2 and Fig. 3), and this metric has a physical meaning. Information density I effectively quantifies the combined impact of category diversity and instance area, providing a unified measurement standard.

This empirical study suggests directions for precisely mitigating model bias. For example, given two categories with the same number of instances but different levels of diversity, the category with higher diversity should obviously receive more attention. However, traditional re-weighting methods based on instance count fail to achieve this. In contrast, the information density can more accurately guide the model to handle different

![](images/1820ba0700b5e9ba94efc508ce845cd636153839665cc1fe5a69fc794c3ef461.jpg)

![](images/668dc454db4aa6921827a357933faf2cc543e7ad70d6bdb903595249a55b5ba8.jpg)  
Fig. 4. The Pearson correlation coefficient between category information density and average precision (AP) per category over epochs.

categories more fairly.

## 3.3 Training Increases the Correlation Between Category Information Density and Average Precision

Interestingly, our experiments also reveal that the correlation between information density and category average precision gradually strengthens as training progresses (as shown in Fig. 4). This phenomenon may be attributed to the model’s tendency to initially learn the more prominent and common features within the dataset during the early stages of training. At this stage, the model has yet to fully understand and distinguish complex features, resulting in a weaker influence of information density on class accuracy. As training continues, the model progressively learns and extracts more intricate and nuanced features. For categories with higher information density, the model needs to identify and differentiate a greater number of distinct features within a given area, which increases the difficulty of learning.

This trend persists until the end of training, where the significant correlation between information density and category accuracy remains. This indicates that the existing optimization objectives insufficiently address categories with high information density. To alleviate this issue and improve the overall performance of the model, we will introduce targeted improvements to the existing object detection loss functions in the next section. These improvements aim to better balance the learning difficulty across different categories, particularly by giving more attention to categories with high information density. Through this approach, we hope to reduce model bias and achieve better performance in handling complex and highly diverse categories.

## 4 METHOD

In this section, we first demonstrate how to embed information density into three advanced object detection loss functions (See-Saw, EFL, and C2AM Loss). Then, we propose a novel training strategy to achieve more efficient dynamic updates of information density during the learning process.

## 4.1 Embedding Information Density into Loss Function

## 4.1.1 Integration of Information Density into SeeSaw Loss

The original purpose of SeeSaw Loss [26] is to reduce the negative gradient exerted by head class samples on tail classes, thereby

increasing the model’s attention to tail classes. The form of Seesaw loss is:

$$
L _ { s e e s a w } ( z ) = - \sum _ { i = 1 } ^ { C } y _ { i } \log ( \sigma _ { i } ) ,
$$

with

$$
\sigma _ { i } = \frac { e ^ { z _ { i } } } { \sum _ { j \neq i } ^ { C } S _ { i j } e ^ { z _ { j } } + e ^ { z _ { i } } } .
$$

Here, $z = [ z _ { 1 } , z _ { 2 } , \dots , z _ { C } ]$ and $\sigma = [ \sigma _ { 1 } , \sigma _ { 2 } , \ldots , \sigma _ { C } ]$ are the predicted logits and probabilities of the classifier, respectively. $y _ { i }$ is the true label of the sample. $S _ { i j }$ is the balancing factor between different classes, composed of the mitigation factor and the compensation factor, specifically $S _ { i j } ~ = ~ M _ { i j } \cdot C _ { i j }$ . The mitigation factor $M _ { i j }$ is defined as the ratio of the number of instances in class $j$ to class $i .$ When the number of instances in class i is greater than that in class $j ,$ the negative gradient of class j is suppressed. The compensation factor focuses on misclassified samples; when a sample from class i is incorrectly identified as class j, the negative gradient generated for class $j$ increases.

Since we observed that the model performs poorly on classes with high information density, we want the model to pay more attention to these classes. Assuming the information densities of $C$ classes are $I _ { 1 } , I _ { 2 } , \ldots , I _ { C }$ , we normalize the information density to the [0, 1] range as follows:

$$
I _ { i } = \frac { e ^ { I _ { i } / ( \bar { I } \cdot \sqrt { C } ) } } { \sum _ { j = 1 } ^ { C } e ^ { I _ { j } / ( \bar { I } \cdot \sqrt { C } ) } } \cdot C + 1 , \quad \bar { I } = \sum _ { i = 1 } ^ { C } I _ { i } .
$$

Furthermore, we propose modifying the balancing factor $S _ { i j }$ to:

$$
S _ { i j } = \left( \frac { I _ { i } } { I _ { j } } \right) ^ { k } \cdot M _ { i j } \cdot C _ { i j } ,
$$

where $I _ { i }$ represents the information density of class $i ,$ and $k$ is a hyperparameter. The gradient of $L _ { \mathrm { s e e s a w } } ( z )$ with respect to the negative class $j$ on $z _ { j }$ is:

$$
\frac { \partial L _ { \mathrm { s e s a w } } ( z ) } { \partial z _ { j } } = \left( \frac { I _ { i } } { I _ { j } } \right) ^ { k } \cdot M _ { i j } \cdot C _ { i j } \cdot \frac { e ^ { z _ { j } } } { e ^ { z _ { i } } } \cdot \sigma _ { i } .
$$

When $I _ { i }$ is greater than $I _ { j }$ , the model needs to give more attention to class $i ,$ thus increasing the negative gradient for class j; conversely, it suppresses the negative gradient for class j. Other details remain consistent with the original Seesaw loss. We refer to the modified loss function as Information Density Guided Seesaw Loss (IDG-Seesaw).

## 4.1.2 Information Density Guided Focal Loss (IDG-FL)

Since Focal Loss cannot differentially handle different classes, Equalized Focal Loss (EFL) [27] proposes a class-specific focusing factor to address the imbalance of positive and negative samples across different classes. The EFL loss function is defined as:

$$
E F L ( p _ { t } ) = - \sum _ { i = 1 } ^ { C } \alpha _ { t } \left( \frac { \gamma _ { b } + \gamma _ { v } ^ { i } } { \gamma _ { b } } \right) ( 1 - p _ { t } ) ^ { \gamma _ { b } + \gamma _ { v } ^ { i } } \log ( p _ { t } ) ,
$$

where the meanings of $\alpha _ { t }$ and $p _ { t }$ are consistent with those in focal loss. The focusing factor $\gamma ^ { i }$ consists of class-independent and class-dependent parameters: $\gamma ^ { i } = \gamma _ { b } + \gamma _ { v } ^ { i } = \gamma _ { b } + s ( 1 - g ^ { i } )$ Increasing the focusing factor reduces the loss, so the weighting term $\left( \frac { \gamma _ { b } + \gamma _ { v } ^ { i } } { \gamma _ { b } } \right)$ in EFL is used to enhance the absolute magnitude of the loss for each class. However, EFL still does not consider the model bias caused by the imbalance of information density between classes. To better address this issue, we propose an improved method that incorporates information density into EFL. Specifically, we modify the weighting term $\left( \frac { \gamma _ { b } + \gamma _ { v } ^ { i } } { \gamma _ { b } } \right)$ to:

$$
\left( \frac { \gamma _ { b } + \gamma _ { v } ^ { i } } { \gamma _ { b } } \right) \cdot I _ { i } ,
$$

where

$$
I _ { i } = \frac { e ^ { I _ { i } / ( \bar { I } \cdot \sqrt { C } ) } } { \sum _ { j = 1 } ^ { C } e ^ { I _ { j } / ( \bar { I } \cdot \sqrt { C } ) } } \cdot C + 1 , \quad \bar { I } = \sum _ { i = 1 } ^ { C } I _ { i } .
$$

By introducing information density $I _ { i } ,$ we can more effectively account for the imbalance of information density between classes, thereby reducing model bias. Apart from this modification, other details remain consistent with the original Equalized Focal Loss (EFL).

## 4.1.3 Information Density Guided C2AM Loss (IDG-C2AM)

In long-tailed object detection tasks, Category-Aware Angular Margin Loss (C2AM) [28] discovered that the norm distribution of classifier weights is highly imbalanced, leading to severe compression of the decision space for classes with smaller weight norms. To address this issue, C2AM proposed that classes with greater diversity should occupy larger decision spaces. However, due to the inability to directly quantify class diversity, C2AM chose to smooth the ratios between class weight norms instead of the diversity ratios and used these ratios to adjust the optimization objective, thus reasonably partitioning the decision space. Specifically, the C2AM loss function is formulated as follows:

$$
L _ { C 2 A M } = - \log ( \frac { e ^ { s \cdot \cos ( \theta _ { i } ) } } { e ^ { s \cdot \cos ( \theta _ { i } ) + \sum _ { j = 1 , j \neq i } ^ { C } e ^ { s \cdot \cos ( \theta _ { j } + m _ { i j } ) } } } ) ,
$$

where

$$
m _ { i j } = \operatorname* { m a x } ( 0 , \frac { \alpha } { \pi } \log ( \frac { \left\| W _ { i } \right\| _ { 2 } } { \left\| W _ { j } \right\| _ { 2 } } ) ) .
$$

Fortunately, our work addresses the challenge of quantifying class diversity by introducing the concept of information density.

Here, we use information density to improve C2AM. Specifically, we modify $m _ { i j }$ to:

$$
m _ { i j } = \operatorname* { m a x } ( 0 , \frac { \alpha } { \pi } ( \frac { I _ { i } } { I _ { j } } ) ) ,
$$

where

$$
I _ { i } = \frac { e ^ { I _ { i } / ( \bar { I } \cdot \sqrt { C } ) } } { \sum _ { j = 1 } ^ { C } e ^ { I _ { j } / ( \bar { I } \cdot \sqrt { C } ) } } \cdot C + 1 , \quad \bar { I } = \sum _ { i = 1 } ^ { C } I _ { i } .
$$

In this formula, the information density $I _ { i }$ is used to replace the weight norm ratios in the original method, thus more accurately reflecting the impact of class diversity on the decision space.

With this, we have detailed how to embed information density into loss functions for object detection. However, we still face an engineering challenge: the information density of classes changes with the model parameters, requiring dynamic updates. Calculating information density requires embeddings of all instances in a class, but we can only obtain a small number of samples within each batch during each iteration. If we were to re-extract embeddings of all instances in the dataset during every iteration, it would interrupt training and incur significant time costs. In the next section, we propose a novel training framework to address this issue.

## 4.2 Low-Cost Dynamic Update of Information Density 4.2.1 Dynamic Update Strategy

The phenomenon of feature slow drift [18] indicates that as training progresses, the distance between embeddings of the same sample at different times becomes smaller, making it possible to approximate the latest embedding with a previous version. Inspired by this, we propose a straightforward solution: store all instance embeddings generated during training in a queue, with the queue length equal to the total number of instances in the dataset. After each training epoch, update all embeddings in the queue. At the end of each epoch, use the embeddings in the queue to calculate and update the information density of all categories. We refer to this approach as the original strategy. While the original strategy avoids repeatedly extracting instance embeddings from the entire dataset, it increases storage space requirements.

Considering that calculating the class diversity in information density essentially requires the covariance matrix of instance embeddings, we propose a new strategy that significantly reduces storage space. The core idea of the new strategy is to use multiple local sample covariance matrices to calculate the global sample covariance matrix, which we prove in Proof 1. The specific steps are as follows:

(1) Initialize a queue to store instance embeddings. The length of the queue can be adjusted according to the GPU memory size. Suppose the object detection dataset contains $C$ categories with a total of $\bar { N }$ instances, and the queue length is d. If $d <$ $N .$ , it means the queue cannot hold all instance embeddings. In this work, we set the queue length to 50,000, which can store 50,000 instance embeddings.

(2) At the beginning of a training epoch, first store the instance embeddings generated in each batch into the queue until it is full (i.e., storing $d$ instance embeddings). Then, use the embeddings in the queue to calculate the local covariance matrix and mean for each category. Continuously update the queue, and once all old embeddings in the queue are updated, calculate the local covariance matrix and mean for each category again. By the end of an epoch, we can calculate $\lfloor N / d \rfloor + 1$ local sample covariance matrices $\Sigma _ { i } ^ { k }$ and means $\bar { \mu } _ { i } ^ { k }$ for each category $\bar { i } = 1 , \ldots , C , k = 1 , \ldots , \bar { \lfloor N / d \rfloor } + 1$ (3) At the end of an epoch, use the stored local covariance matrices to calculate and update the information density for each category. Taking category i as an example, first calculate the global mean:

Algorithm 2 Dynamic Update of Information Density   
1: Input: Number of classes C, total number of instances ${ \overline { { N } } } ,$   
queue length d (e.g., 50,000), number of epochs E   
2: Initialize queue $Q$ with maximum length d   
3: for each epoch $e = 1$ to E do   
4: for each batch of data do   
5: Extract instance embeddings for each class and store in   
queue $Q$   
6: if Queue $Q$ is full then   
7: Compute local covariance matrix $\Sigma _ { i } ^ { k }$ and mean $\mu _ { i } ^ { k }$ for   
each class i   
8: Update queue $Q$ with new embeddings   
9: end if   
10: end for   
11: After processing all batches, compute additional local co  
variance matrices and means as needed   
12: for each class $i = 1$ to $C$ do   
13: Compute global mean $\mu _ { i } { : }$   
$\mu _ { i }  \frac { 1 } { N _ { i } } \sum _ { k = 1 } ^ { \lfloor N / d \rfloor + 1 } n _ { i } ^ { k } \mu _ { i } ^ { k }$   
14: Compute global covariance matrix $\Sigma _ { i } \colon$   
$\Sigma _ { i }  \frac { 1 } { N _ { i } } \binom { \lfloor N / d \rfloor + 1 } { k = 1 } n _ { i } ^ { k } \Sigma _ { i } ^ { k }$   
$+ \sum _ { k = 1 } ^ { \lfloor N / d \rfloor + 1 } n _ { i } ^ { k } ( \mu _ { i } ^ { k } - \mu _ { i } ) ( \mu _ { i } ^ { k } - \mu _ { i } ) ^ { T } \Big )$   
15: Estimate the information density $I _ { i }$ using Algorithm 1.   
16: end for   
17: end for   
18: Output: Information densities $I _ { i }$ for each class i

$$
\mu _ { i } = \frac { 1 } { N _ { i } } \sum _ { k = 1 } ^ { \lfloor N / d \rfloor + 1 } n _ { i } ^ { k } \mu _ { i } ^ { k } ,
$$

where $N _ { i }$ is the total number of instances in category $i ,$ and $n _ { i } ^ { k }$ is the number of instances in the local sample. Then, calculate the global covariance matrix:

$$
\Sigma _ { i } = \frac { 1 } { N _ { i } } \left( \sum _ { k = 1 } ^ { \lfloor N / d \rfloor + 1 } n _ { i } ^ { k } \Sigma _ { i } ^ { k } + \sum _ { k = 1 } ^ { \lfloor N / d \rfloor + 1 } n _ { i } ^ { k } ( \mu _ { i } ^ { k } - \mu _ { i } ) ( \mu _ { i } ^ { k } - \mu _ { i } ) ^ { T } \right)
$$

The proof of this formula is provided in Proof 1. By integrating local covariance matrices to obtain the global covariance matrix, we significantly reduce the additional storage space required to update the information density. Further, the diversity of category i is estimated as $\begin{array} { r } { \mathrm { V o l } _ { i } = \frac { 1 } { 2 } \log _ { 2 } \operatorname* { d e t } ( I + \Sigma _ { i } ) } \end{array}$ and the information density is $I = V o l _ { i } \bar { / } A _ { i }$ , where $A _ { i }$ is the total area of instances in category i.

Proof. 1: Integrating Local Covariance Matrices to Obtain the Global Covariance Matrix. Assume we have a dataset containing $N$ instances, and we divide these instances into K batches, each containing $n _ { k }$ instances. For the k-th batch, let the instances be $\{ x _ { k 1 } , x _ { k 2 } , . . . , x _ { k n _ { k } } \}$ . The mean vector and local covariance matrix for this batch are defined as follows:

$$
\mu _ { k } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } x _ { k i } ,
$$

$$
\Sigma _ { k } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } ( x _ { k i } - \mu _ { k } ) ( x _ { k i } - \mu _ { k } ) ^ { T } .
$$

The global covariance matrix is the covariance matrix of all batches, defined as:

$$
\mu = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } x _ { k i } ,
$$

$$
\Sigma = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } ( x _ { k i } - \mu ) ( x _ { k i } - \mu ) ^ { T } .
$$

First, calculate the global mean $\mu { : }$

$$
\mu = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } { x _ { k i } } = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } { n _ { k } \mu _ { k } } .
$$

Then, split $( x _ { k i } - \mu )$ in the global covariance matrix Σ into $\left( \boldsymbol { x } _ { k i } - \boldsymbol { \mu _ { k } } \right)$ and $( \mu _ { k } - \mu ) \colon$

$$
\Sigma = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } [ ( x _ { k i } - \mu _ { k } + \mu _ { k } - \mu ) ( x _ { k i } - \mu _ { k } + \mu _ { k } - \mu ) ^ { T } ] .
$$

Expanding this, we get:

$$
\begin{array} { c } { \Sigma = \displaystyle \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } [ ( x _ { k i } - \mu _ { k } ) ( x _ { k i } - \mu _ { k } ) ^ { T } + ( x _ { k i } - \mu _ { k } ) ( \mu _ { k } - \mu ) ^ { T } } \\ { + ( \mu _ { k } - \mu ) ( x _ { k i } - \mu _ { k } ) ^ { T } + ( \mu _ { k } - \mu ) ( \mu _ { k } - \mu ) ^ { T } ] . } \end{array}
$$

According to the properties of the covariance matrix, the first term is the local covariance matrix $\Sigma _ { k }$ , and the expectation values of the second and third terms are zero. The fourth term can be calculated as:

$$
\sum _ { i = 1 } ^ { n _ { k } } ( \mu _ { k } - \mu ) ( \mu _ { k } - \mu ) ^ { T } = n _ { k } ( \mu _ { k } - \mu ) ( \mu _ { k } - \mu ) ^ { T } .
$$

Finally, the expression for the global covariance matrix is:

$$
\Sigma = \frac { 1 } { N } \left( \sum _ { k = 1 } ^ { K } n _ { k } \Sigma _ { k } + \sum _ { k = 1 } ^ { K } n _ { k } ( \mu _ { k } - \mu ) ( \mu _ { k } - \mu ) ^ { T } \right) .
$$

This formula demonstrates that the global covariance matrix can be calculated by taking a weighted sum of the local covariance matrices and adding the difference terms between local means and the global mean. This integration method effectively utilizes the unbiasedness and independence of the local covariance matrices, ensuring the accuracy of the global covariance matrix. □

## 4.2.2 Storage Space Comparison

Assume the object detection dataset contains N instances, each with an embedding dimension of $p ,$ and there are $C$ categories.

![](images/5cb47ed06070858dd7eb92c47dfc87f1e036ba2fb0f518ee28a76936770ab114.jpg)

![](images/d4dbb3cef0763ef0bae9b8ed2803308b1d220c3117bad0eeeb1a4016229ad3de.jpg)  
Fig. 5. The function of the storage space ratio R as it varies with the queue length d on the Pascal VOC and MS COCO datasets.

The queue length is set to $d .$ The storage space required by the original strategy is:

$$
S _ { \mathrm { o r i g i n a l } } = N \times p .
$$

The storage space required by the new strategy is:

$$
S _ { \mathrm { n e w } } = d \times p + C \times \left( \lfloor N / d \rfloor + 1 \right) \times p ^ { 2 } .
$$

where $C \times ( \lfloor N / d \rfloor + 1 ) \times p ^ { 2 }$ represents the space needed to store the local covariance matrices. To analyze when the new strategy saves more space, we define the storage space ratio R:

$$
R = { \frac { S _ { \mathrm { n e w } } } { S _ { \mathrm { o r i g i n a l } } } } = { \frac { d \times p + C \times ( \lfloor N / d \rfloor + 1 ) \times p } { N } } .
$$

When $R \ < \ 1 .$ , the new strategy saves more storage space. To visually compare the storage space requirements of the new and original strategies, we take the Pascal VOC, MS COCO, and LVIS v1.0 datasets as examples and plot the function graph of the storage space ratio R as it varies with the queue length d in Fig. 5. It can be observed that in most cases, the storage space required by the new strategy is less than that of the original strategy. By visualizing our proposed storage space ratio, it becomes easy to choose the optimal queue length d, thereby saving approximately 60% of the storage space. Given two examples $\{ N = 5 5 8 0 0 , p =$ 128, C = 20} and $\{ N = 6 0 5 6 3 8 , p = 1 2 8 , C = 8 0 \}$ , corresponding to the Pascal VOC and MS COCO datasets respectively, we use Listing 1 to select the optimal queue lengths d of 11517 and 68182. For Example 1, the original strategy requires an additional 27.25 MB of memory, while the new strategy only requires 11.87 MB, saving approximately 56.44% of memory. For Example 2, the new strategy can reduce memory usage from 295.72 MB to 78.29 MB, saving approximately 73.52% of memory.

The new training framework significantly reduces storage space utilization by merging the local sample covariance matrices, ensuring that the calculated value of information density remains unchanged. This innovative strategy not only provides an efficient solution for dynamically updating information density but also offers a low-cost storage solution for future research. In the experimental section, we demonstrate the significant improvement effects of information density on multiple loss functions. Additionally, loss functions improved with information density can be combined with other types of methods to further enhance the performance of object detection models (see Section 5).

```python
Listing 1 Selecting the optimal queue length d.
1 import matplotlib.pyplot as plt
2 import numpy as np
3
4 # Define parameters
5 N = 55800 # Total number of instances
6 p = 128 # Embedding dimension
7 C = 20 # Number of categories
8
9 # Define the range of queue lengths
10 d_values = np.linspace(1000, N, 100)
11 R_values = list((d_values + C <sub>*</sub> (np.floor(N /
d_values) + 1) <sub>*</sub> p) / N)
12
13 min_R_values = min(np.abs(R_values))
14 optimal_d = d_values[R_values.index(min_R_values)
]
15 print(’The optimal d is:’, optimal_d)
```

## 5 EXPERIMENTS

We conducted a comprehensive evaluation of the universal and significant improvements that information density brings to existing object detection loss functions on three object detection benchmark datasets. The experiments were divided into two parts. The first part was conducted on the non-long-tailed object detection dataset Pascal VOC. The second part was conducted on the longtailed object detection datasets COCO-LT and LVIS v1.0.

## 5.1 Datasets and Evaluation Metrics

Pascal VOC [17] includes two versions, 2007 and 2012, comprising a total of 20 classes. Following standard practice [58], we trained on the train+val sets of VOC2007 and VOC2012 and tested on the test set of VOC2007. The COCO-LT [13] dataset is a long-tailed subset of MS COCO [57], and they share the same validation set. Consistent with previous work [26], we divided the 80 classes in COCO-LT into four groups based on the number of training instances per class: fewer than 20 images, 20∼400 images, 400∼8000 images, and 8000 or more images. We report the Average Precision (AP) for each class on Pascal VOC, and on COCO-LT, we report the accuracy of the four groups as $A P _ { 1 } ^ { b }$ <sup>b</sup>, AP<sup>b</sup>, 2 $A P _ { 3 } ^ { b } ,$ and $A P _ { 4 } ^ { b }$ . The mean average precision is reported as $m A { \ddot { P } } ^ { b }$ . LVIS v1.0 [15] contains 1,203 categories, with the training set consisting of 100k images (approximately 1.3M instances) and the validation set containing 19.8k images. Based on the frequency of occurrence in the training set, all categories are divided into three groups: rare (1∼10 images), common (11∼100 images), and frequent (more than 100 images). In line with EFL [27], we report not only the widely used object detection metric $A P ^ { b }$ across IOU thresholds (from 0.5 to 0.95) but also the bounding box AP for frequent $( A P _ { f } )$ , common $( A P _ { c } )$ and rare (AP<sub>r</sub>) categories separately.

TABLE 1  
Average precision (%) of Faster R-CNN with R-50-FPN backbone trained using various loss functions on the Pascal VOC.
<table><tr><td>Class Seesaw IDG-Seesaw EFL IDG-FL</td></tr><tr><td>cat 89.5</td><td>89.0</td><td>88.3</td><td>C2AM IDG-C2AM 88.7 88.1</td><td>87.8 81.3</td></tr><tr><td>car</td><td>79.8</td><td>82.5</td><td>78.9 80.5 85.5</td><td>80.8</td></tr><tr><td>horse</td><td>85.9</td><td>86.7 86.3 85.8</td><td></td><td>87.0 88.1</td></tr><tr><td>bus</td><td>84.8</td><td>82.9 87.5</td><td>81.9 83.8</td><td>84.5 85.2</td></tr><tr><td>bicycle</td><td>86.1</td><td>84.5</td><td></td><td>85.2 84.6</td></tr><tr><td>dog</td><td>88.4</td><td>88.2 85.3</td><td>86.0</td><td>86.5 87.4</td></tr><tr><td>person</td><td>79.1</td><td>79.7 77.1</td><td>80.1</td><td>77.8 79.5</td></tr><tr><td>train</td><td>84.2</td><td>85.9 82.9</td><td>84.4</td><td>85.3 87.0</td></tr><tr><td>motorbike</td><td>83.6</td><td>84.7 80.1</td><td>82.7</td><td>82.0 83.2</td></tr><tr><td>COw</td><td>82.9</td><td>83.5 79.8</td><td>81.6</td><td>80.7 82.1</td></tr><tr><td>aeroplane</td><td>69.7 72.8 ↑3.1</td><td></td><td>72.1 74.4 ↑2.3</td><td>71.8 74.3 ↑2.5</td></tr><tr><td>tvmonitor</td><td>75.8</td><td>79.3 76.4 78.7</td><td>77.2</td><td>74.1 75.7 75.6 76.5</td></tr><tr><td>sheep</td><td>76.5</td><td>74.8</td><td>76.9</td><td>79.7 81.0</td></tr><tr><td>bird</td><td>79.3</td><td>81.5 75.2</td><td>76.8</td><td>75.1</td></tr><tr><td>diningtable sofa</td><td>74.0 78.7</td><td>76.0 71.3 79.1</td><td>73.2</td><td>73.4 79.5 80.8</td></tr><tr><td>boat</td><td>66.4 68.2 ↑1.8</td><td>72.7</td><td>70.5 62.4 64.9 ↑2.5</td><td>64.2 66.5 ↑2.2</td></tr><tr><td>bottle</td><td>53.6</td><td>57.9 ↑4.3</td><td>50.8 54.6 ↑3.8</td><td>54.8 58.2 ↑3.4</td></tr><tr><td>chair</td><td>65.8</td><td>68.5 ↑2.7</td><td>61.1 63.7 ↑2.6</td><td>62.3 65.1 ↑2.8</td></tr><tr><td>pottedplant</td><td>52.7</td><td>57.3 ↑4.6</td><td>49.1 53.2 ↑4.1</td><td>51.5 54.8 ↑3.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Total</td><td>76.9</td><td>78.6 ↑1.7</td><td>74.6 76.0 ↑1.4</td><td>76.2 77.7 ↑1.5</td></tr></table>

## 5.2 Implementation Details

We implemented the Faster R-CNN [55] detector using the MMDetection [59] toolbox and adopted ResNet-50 and ResNet-101 [56] with an FPN [22] structure as the backbone networks. During training, we set the batch size to 16 and the initial learning rate to 0.02, consistent with Seesaw [26], EFL [27], and C2AM [28]. We trained the model using an SGD optimizer with a momentum of 0.9 and a weight decay rate of 0.0001 for 24 epochs. The learning rate was reduced to 0.002 and 0.0002 at the end of the 16th and 22nd epochs, respectively. In all experiments, we applied random horizontal flipping and multi-scale jittering for data augmentation. We did not use any test-time augmentations.

## 5.3 Evaluation on Pascal VOC

We first trained baseline models using Seesaw, EFL, and C2AM losses. Then, we trained the proposed IDG-Seesaw, IDG-FL, and IDG-C2AM under the same settings. The experimental results on Pascal VOC are presented in Table 1 and 2. It can be observed that even for Seesaw, EFL, and C2AM, which are designed to mitigate model bias, our improvements led to an overall increase in accuracy. Specifically, when the backbone is R-50-FPN, IDG-Seesaw, IDG-FL, and IDG-C2AM improve the overall performance by 1.7%, 1.4%, and 1.5%, respectively, compared to the original methods. When using ResNet-101 as the backbone, the improved methods enhance the overall performance by 1.2%, 1.5%, and 1.3%, respectively.

TABLE 2  
Average precision (%) of Faster R-CNN with R-101-FPN backbone trained using various loss functions on the Pascal VOC.
<table><tr><td>Class</td></tr><tr><td>Seesaw IDG-Seesaw EFL IDG-FL cat 91.4</td><td>91.8</td><td>91.2</td><td>90.5</td><td>C2AM IDG-C2AM 89.8 88.0 83.3</td></tr><tr><td>car</td><td>82.2</td><td>83.0</td><td>81.1 82.4 86.6</td><td>82.0 87.6</td></tr><tr><td>horse</td><td>86.5</td><td>85.6 84.9</td><td>87.1 85.7</td></tr><tr><td>bus</td><td>83.6</td><td>84.4 88.3</td><td>85.3 86.5 85.5 85.5</td></tr><tr><td>bicycle</td><td>87.0 90.8</td><td>85.8 90.2 90.3</td><td>86.2 87.1 86.7</td></tr><tr><td>dog</td><td>90.6 84.9</td><td>82.8 84.6</td><td>78.2 79.8</td></tr><tr><td>person</td><td>84.6 88.0</td><td>85.4 85.9</td><td>85.5 86.1</td></tr><tr><td>train</td><td>86.4 87.7</td><td>84.1</td><td>82.6 83.6</td></tr><tr><td>motorbike</td><td>87.4 81.8</td><td>85.0 80.1 81.9</td><td>81.0 81.9</td></tr><tr><td>COw</td><td>82.0 73.1</td><td>70.0 72.6 ↑2.6</td><td>72.5 74.7 ↑2.2</td></tr><tr><td>aeroplane tvmonitor 75.2</td><td>75.5 ↑2.4 76.8</td><td>69.9 72.2</td><td>77.0 78.5</td></tr><tr><td></td><td>80.4 81.7</td><td>78.2 80.9</td><td>75.5 77.4</td></tr><tr><td>sheep bird</td><td>74.1 76.4</td><td>72.3 74.1</td><td>80.4 81.2</td></tr><tr><td>diningtable</td><td>75.5 75.9</td><td>73.3 75.1</td><td>74.3 76.1</td></tr><tr><td>sofa</td><td>77.2 78.0</td><td>72.5 74.8</td><td>79.7 80.6</td></tr><tr><td>boat</td><td>63.3 65.3 ↑2.0</td><td>64.2 67.2 ↑3.0</td><td>65.2 67.9 ↑2.7</td></tr><tr><td>bottle</td><td>52.4 55.6 ↑3.2</td><td>51.5 54.0 ↑2.5</td><td>55.6 58.6 ↑3.0</td></tr><tr><td>chair</td><td>62.3 65.6 ↑3.3</td><td>61.9 64.2 ↑2.3</td><td>63.5 66.0 ↑2.5</td></tr><tr><td>pottedplant</td><td>53.9 57.8 ↑3.9</td><td>50.0 53.5 ↑3.5</td><td>52.1 55.8 ↑3.7</td></tr><tr><td>Total</td><td>77.5 78.7 ↑1.2</td><td>75.8 77.3 ↑1.5</td><td>77.0 78.3 ↑1.3</td></tr></table>

More importantly for this work, our methods significantly improve the performance of poorly performing categories. We selected five representative underperforming categories for observation: aeroplane, boat, bottle, chair, and potted plant. When the backbone is R-50-FPN, our improvements led to performance gains of 3.1%, 1.8%, 4.3%, 2.7%, and 4.6% for Seesaw loss in these five categories, respectively. When using R-101-FPN as the backbone, our improvements enhanced Seesaw loss performance by 2.4%, 2.0%, 3.2%, 3.3%, and 3.9% in these categories, respectively. Particularly, for the two worst-performing categories, bottle, and potted plant, our improvements were the most pronounced. The experimental results demonstrate that although Seesaw, EFL, and C2AM losses are designed to address model bias, our methods can further focus on and identify the underperforming categories. This further proves that the proposed information density can more accurately measure the learning difficulty of categories.

## 5.4 Evaluation on COCO-LT and LVIS v1.0

The comparison results on COCO-LT are summarized in Table 3. When using the R-50-FPN backbone, IDG-Seesaw, IDG-FL, and IDG-C2AM improved mAP by 1.0%, 1.3%, and 1.1%, respectively, compared to Seesaw, EFL, and C2AM. More importantly, our improvements led to increases in $4 P _ { 1 } ^ { b }$ (bounding box detection precision for rare categories) by 5.2%, 4.7%, and 5.5%, respectively. When using the R-101-FPN backbone, our modifications further enhanced $A P _ { 1 } ^ { b }$ for Seesaw, EFL, and C2AM by 5.8%, 5.5%, and 5.6%, respectively. The performance improvements for rare categories were even more pronounced compared to overall performance gains.

TABLE 3  
Evaluation results on COCO-LT. The $m A P ^ { b } , A P _ { 1 } ^ { b } , A P _ { 2 } ^ { b } , A P _ { 3 } ^ { b }$ , and $A P _ { 4 } ^ { b } ~ ( \% )$ for each method are reported, with red arrows indicating performance improvements.
<table><tr><td>Dataset</td><td colspan="5">COCO-LT</td></tr><tr><td></td><td> $m A P ^ { b }$ </td><td> $A P _ { 1 } ^ { b }$ </td><td> $A P _ { 2 } ^ { b }$ </td><td>APb</td><td> $A P _ { 4 } ^ { b }$ </td></tr><tr><td colspan="6">R-50-FPN</td></tr><tr><td>Cross-Entropy (CE)</td><td>24.5</td><td>0</td><td>14.6</td><td>29.6</td><td>32.9</td></tr><tr><td>Seesaw [26]</td><td>23.9</td><td>3.0</td><td>14.5</td><td>28.4</td><td>32.3</td></tr><tr><td>IDG-Seesaw</td><td>24.9 ↑1.0</td><td>8.2 ↑5.2</td><td>16.7</td><td>28.6</td><td>32.1</td></tr><tr><td>EFL [27]</td><td>25.0</td><td>3.8</td><td>16.3</td><td>29.5</td><td>32.5</td></tr><tr><td>IDG-FL</td><td>26.3 ↑1.3</td><td>8.5 ↑4.7</td><td>19.3</td><td>29.7</td><td>32.8</td></tr><tr><td>C2AM [28]</td><td>24.7</td><td>2.8</td><td>15.6</td><td>29.4</td><td>32.3</td></tr><tr><td>IDG-C2AM</td><td>25.8 ↑1.1</td><td>8.3 ↑5.5</td><td>17.8</td><td>29.8</td><td>32.6</td></tr><tr><td colspan="6">R-101-FPN</td></tr><tr><td>Cross-Entropy (CE)</td><td>26.0</td><td>0</td><td>16.4</td><td>31.4</td><td>34.2</td></tr><tr><td>Seesaw [26]</td><td>24.9</td><td>3.2</td><td>14.5</td><td>30.0</td><td>33.4</td></tr><tr><td>IDG-Seesaw</td><td>26.1 ↑1.2</td><td>9.0 ↑5.8</td><td>16.8</td><td>30.7</td><td>33.5</td></tr><tr><td>EFL [27]</td><td>25.4</td><td>3.6</td><td>16.5</td><td>30.2</td><td>32.8</td></tr><tr><td>IDG-FL</td><td>26.5 ↑1.1</td><td>9.1 ↑5.5</td><td>19.4</td><td>30.6</td><td>32.0</td></tr><tr><td>C2AM [28]</td><td>25.1</td><td>2.9</td><td>15.6</td><td>30.2</td><td>32.7</td></tr><tr><td>IDG-C2AM</td><td>26.3 ↑1.2</td><td>8.5 ↑5.6</td><td>18.1</td><td>30.7</td><td>32.5</td></tr></table>

Table 4 presents the experimental results on LVIS v1.0. Our approach consistently demonstrated superior performance gains for rare categories, once again highlighting our method’s focus on addressing categories that are traditionally underperforming. By improving performance for these categories, we significantly reduced model bias. Specifically, when using the R-50-FPN backbone, IDG-Seesaw, IDG-FL, and IDG-C2AM achieved improvements in the $A P _ { r }$ metric by 3.1%, 3.5%, and 3.4%, respectively, compared to the original methods. With the R-101-FPN backbone, our method increased the $A P _ { r }$ metric by 4.3%, 3.2%, and 2.9%, respectively. These experimental results collectively indicate that information density can more accurately measure class difficulty, thereby promoting the model’s focus on learning from the underperforming classes.

## 5.5 Effectiveness in Reducing Model Bias

To more clearly demonstrate the effectiveness of our method in mitigating model bias, we used the variance in Average Precision (AP) across classes as a measure of model bias. The comparative results on COCO-LT and LVIS v1.0 are presented in Fig. 6. It can be observed that under two different backbones, our method significantly reduces model bias. For example, on COCO-LT, IDG-Seesaw (R-50-FPN) reduces model bias by 37.23% compared to Seesaw (R-50-FPN). On LVIS v1.0, IDG-FL (R-101-FPN) reduces model bias by 45.89% compared to EFL (R-101-FPN).

The significant improvements to Seesaw, EFL, and C2AM suggest that mitigating model bias by addressing information density imbalance is a parallel approach to addressing gradient imbalance or decision space imbalance. That is, alleviating gradient imbalance does not necessarily address model bias caused by information density imbalance. Therefore, we urge other researchers to consider the model bias introduced by information density imbalance when designing object detection models.

TABLE 4  
Evaluation results on LVIS v1.0. The $m A P ^ { b } , A P _ { r } , A P _ { c } ,$ and $A P _ { f }$ (%) for each method are reported, with red arrows indicating performance improvements.
<table><tr><td>Dataset</td><td colspan="4">LVIS v1.0</td></tr><tr><td></td><td> $m A P ^ { b }$ </td><td> $A P _ { r }$ </td><td> $A P _ { c }$ </td><td> $A P _ { f }$ </td></tr><tr><td colspan="5">R-50-FPN</td></tr><tr><td>SCE [55]</td><td>19.3</td><td>1.1</td><td>16.1</td><td>30.9</td></tr><tr><td>BCE [55] EQL [29]</td><td>19.5 21.8</td><td>1.6 3.6</td><td>16.6 21.1</td><td>30.6 30.5</td></tr><tr><td>DropLoss [60]</td><td>21.8</td><td>5.2</td><td>21.8</td><td>29.1</td></tr><tr><td>Seesaw [26]</td><td>24.8</td><td>14.8</td><td>22.7</td><td>31.6</td></tr><tr><td>IDG-Seesaw</td><td>25.9 ↑1.1</td><td>17.9 ↑3.1</td><td>23.5</td><td>31.9</td></tr><tr><td>EFL [27]</td><td>26.0</td><td>18.6</td><td>24.5</td><td>30.8</td></tr><tr><td>IDG-FL</td><td>27.1 ↑1.1</td><td>22.1 ↑3.5</td><td>25.4</td><td>31.0</td></tr><tr><td>C2AM [28]</td><td>25.4</td><td>15.6</td><td>24.2</td><td>30.9</td></tr><tr><td>IDG-C2AM</td><td>26.6 ↑1.2</td><td>19.0 ↑3.4</td><td>25.1</td><td>31.4</td></tr><tr><td colspan="5">R-101-FPN</td></tr><tr><td>SCE [55]</td><td>20.9</td><td>1.0</td><td>18.2</td><td>32.7</td></tr><tr><td>BCE [55] EQL [29]</td><td>21.4 23.4</td><td>2.0 4.5</td><td>19.3</td><td>32.3</td></tr><tr><td>DropLoss [60]</td><td>23.5</td><td>5.9</td><td>22.9 23.9</td><td>32.3 30.7</td></tr><tr><td>Seesaw [26]</td><td>26.6</td><td>14.9</td><td>25.2</td><td>33.3</td></tr><tr><td>IDG-Seesaw</td><td>28.0 ↑1.4</td><td>19.2 ↑4.3</td><td>26.5</td><td>33.5</td></tr><tr><td>EFL [27]</td><td>26.3</td><td>19.3</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>23.2</td><td>32.6</td></tr><tr><td>IDG-FL</td><td>27.3 ↑1.0</td><td>22.5 ↑3.2</td><td>24.7</td><td>32.1</td></tr><tr><td>C2AM [28]</td><td>26.5</td><td>18.1</td><td>25.5</td><td>31.2</td></tr><tr><td>IDG-C2AM</td><td>27.8 ↑1.3</td><td>21.0 ↑2.9</td><td>26.8</td><td>31.7</td></tr></table>

## 6 CONCLUSION AND FUTURE WORK

This work investigates the issue of generalized bias in deep learning models for object detection tasks, which cannot be explained solely by the number of instances. We proposed the concept of information density to measure the difficulty of detecting different categories. Experiments revealed a significant negative correlation between a category’s information density and its accuracy. Subsequently, we utilized information density to improve three advanced object detection loss functions. The experimental results demonstrate that our improvements are most pronounced for categories with poor performance. Comprehensive empirical research indicates that information density provides a more accurate evaluation of a category’s learning difficulty, aiding models in focusing on learning underrepresented classes. In the future, we believe that information density could advance object detection tasks in the following directions:

(1) Information density can guide data augmentation methods for object detection datasets. For categories with high information density, traditional data augmentation techniques like

![](images/4ad9aeb244bc5ca19203195ee751f36f5c9978b05ef8f1c5943b29baaf091a54.jpg)

![](images/681da9e2887a6d673d84f5cfc95bb08cbb21593d1c0164201fbb54e9587c68cd.jpg)  
Fig. 6. The effectiveness of our method in mitigating model bias on COCO-LT and LVIS v1.0.

Copy-Paste or data expansion using generative models can be employed.

(2) Model Capability Exploration: Information density can be used to explore the capability limits of different object detection models or backbones. Higher information density implies that the model must handle larger amounts of information per unit, which may require larger models to tackle the task effectively. Quantifying the capability limits of each model will help engineers choose models with appropriate parameter sizes for different scale tasks, balancing deployment costs and performance requirements.

(3) Understanding and Improving Model Architecture: The high correlation between a category’s information density and its accuracy provides new insights into understanding the learning mechanisms of object detection models and improving model architectures. From the perspective of information compression, the “capacity” of a model should be related to its performance. Therefore, when designing model architectures, additional adaptive modules or branches can be added to further refine and handle categories with high information density, indirectly increasing the model’s “capacity” for specific categories.

Additionally, aside from information density, we believe there are other underlying factors affecting model bias or performance. We hope to draw attention from other researchers to further enhance the interpretability of fairness in object detection models.

## REFERENCES

[1] L. Jiao, F. Zhang, F. Liu, S. Yang, L. Li, Z. Feng, and R. Qu, “A survey of deep learning-based object detection,” IEEE access, vol. 7, pp. 128 837– 128 868, 2019.

[2] L. Liu, W. Ouyang, X. Wang, P. Fieguth, J. Chen, X. Liu, and M. Pietikainen, “Deep learning for generic object detection: A survey,”¨ International journal of computer vision, vol. 128, pp. 261–318, 2020.

[3] Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, “Object detection in 20 years: A survey,” Proceedings of the IEEE, vol. 111, no. 3, pp. 257–276, 2023.

[4] M. Qi, S. Mao, Y. Zhang, J. Gu, S. Gou, L. Jiao, and Y. Zhang, “Dasce: Long-tailed data augmentation based sparse class-correlation exploitation,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4497–4511, 2025.

[5] L. Yang, H. Jiang, Q. Song, and J. Guo, “A survey on long-tailed visual recognition,” International Journal of Computer Vision, vol. 130, no. 7, pp. 1837–1872, 2022.

[6] S. Li, L. Song, X. Wu, Z. Hu, Y.-m. Cheung, and X. Yao, “Multi-class imbalance classification based on data distribution and adaptive weights,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 10, pp. 5265–5279, 2024.

[7] Y. Ma, L. Jiao, F. Liu, S. Yang, X. Liu, and P. Chen, “Geometric prior guided feature representation learning for long-tailed classification,” International Journal of Computer Vision, pp. 1–18, 2024.

[8] K. Joseph, S. Khan, F. S. Khan, and V. N. Balasubramanian, “Towards open world object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 5830–5840.

[9] K. Oksuz, B. C. Cam, S. Kalkan, and E. Akbas, “Imbalance problems in object detection: A review,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 10, pp. 3388–3415, 2020.

[10] S. Alshammari, Y.-X. Wang, D. Ramanan, and S. Kong, “Long-tailed recognition via weight balancing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 6897– 6907.

[11] E. Zhang, C. Li, C. Geng, and S. Chen, “All-around neural collapse for imbalanced classification,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4460–4470, 2025.

[12] Y. Cao, J. Kuang, M. Gao, A. Zhou, Y. Wen, and T.-S. Chua, “Learning relation prototype from unlabeled texts for long-tail relation extraction,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 2, pp. 1761–1774, 2023.

[13] T. Wang, Y. Li, B. Kang, J. Li, J. Liew, S. Tang, S. Hoi, and J. Feng, “The devil is in classification: A simple framework for long-tail instance segmentation,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16. Springer, 2020, pp. 728–744.

[14] S. Sinha, H. Ohashi, and K. Nakamura, “Class-difficulty based methods for long-tailed visual recognition,” International Journal of Computer Vision, vol. 130, no. 10, pp. 2517–2531, 2022.

[15] A. Gupta, P. Dollar, and R. Girshick, “Lvis: A dataset for large vocabulary instance segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5356–5364.

[16] N. Chang, Z. Yu, Y.-X. Wang, A. Anandkumar, S. Fidler, and J. M. Alvarez, “Image-level or object-level? a tale of two resampling strategies for long-tailed detection,” in International conference on machine learning. PMLR, 2021, pp. 1463–1472.

[17] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes challenge: A retrospective,” International Journal of Computer Vision, vol. 111, no. 1, pp. 98–136, Jan. 2015.

[18] Y. Ma, L. Jiao, F. Liu, Y. Li, S. Yang, and X. Liu, “Delving into semantic scale imbalance,” in The Eleventh International

Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=07tc5kKRIo

[19] Y. Ma, L. Jiao, F. Liu, S. Yang, X. Liu, and L. Li, “Curvature-balanced feature manifold learning for long-tailed classification,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 15 824–15 835.

[20] C. Kaushik, R. Liu, C.-H. Lin, A. Khera, M. Y. Jin, W. Ma, V. Muthukumar, and E. L. Dyer, “Balanced data, imbalanced spectra: Unveiling class disparities with spectral imbalance,” arXiv preprint arXiv:2402.11742, 2024.

[21] Y. Ma, L. Jiao, F. Liu, L. Li, W. Ma, S. Yang, X. Liu, and P. Chen, “Unveiling and mitigating generalized biases of dnns through the intrinsic dimensions of perceptual manifolds,” arXiv preprint arXiv:2404.13859, 2024.

[22] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2117–2125.

[23] Q. Li, B. Sorscher, and H. Sompolinsky, “Representations and generalization in artificial and brain neural networks,” Proceedings of the National Academy of Sciences, vol. 121, no. 27, p. e2311805121, 2024.

[24] J. J. DiCarlo and D. D. Cox, “Untangling invariant object recognition,” Trends in cognitive sciences, vol. 11, no. 8, pp. 333–341, 2007.

[25] Y. Ma, L. Jiao, F. Liu, S. Yang, X. Liu, and L. Li, “Orthogonal uncertainty representation of data manifold for robust long-tailed learning,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 4848–4857.

[26] J. Wang, W. Zhang, Y. Zang, Y. Cao, J. Pang, T. Gong, K. Chen, Z. Liu, C. C. Loy, and D. Lin, “Seesaw loss for long-tailed instance segmentation,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9695–9704.

[27] B. Li, Y. Yao, J. Tan, G. Zhang, F. Yu, J. Lu, and Y. Luo, “Equalized focal loss for dense long-tailed object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 6990–6999.

[28] T. Wang, Y. Zhu, Y. Chen, C. Zhao, B. Yu, J. Wang, and M. Tang, “C2am loss: Chasing a better decision boundary for long-tail object detection,” in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, 2022, pp. 6980–6989.

[29] J. Tan, C. Wang, B. Li, Q. Li, W. Ouyang, C. Yin, and J. Yan, “Equalization loss for long-tailed object recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 662–11 671.

[30] J. Tan, X. Lu, G. Zhang, C. Yin, and Q. Li, “Equalization loss v2: A new gradient balance approach for long-tailed object detection,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1685–1694.

[31] J. H. Cho and P. Krahenb ¨ uhl, “Long-tail detection with effective class-¨ margins,” arXiv preprint arXiv:2301.09724, 2023.

[32] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[33] H. Sun, J. Li, and X. Zhu, “A novel expandable borderline smote oversampling method for class imbalance problem,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 5, pp. 2183–2199, 2025.

[34] Z. Zhang and T. Pfister, “Learning fast sample re-weighting without reward data,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 725–734.

[35] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9268– 9277.

[36] X. Wang, L. Lian, Z. Miao, Z. Liu, and S. X. Yu, “Long-tailed recognition by routing diverse distribution-aware experts,” arXiv preprint arXiv:2010.01809, 2020.

[37] R. Razavi-Far, M. Farajzadeh-Zanajni, B. Wang, M. Saif, and S. Chakrabarti, “Imputation-based ensemble techniques for class imbalance learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 5, pp. 1988–2001, 2021.

[38] B. Zhou, Q. Cui, X.-S. Wei, and Z.-M. Chen, “Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 9719–9728.

[39] B. Kang, S. Xie, M. Rohrbach, Z. Yan, A. Gordo, J. Feng, and Y. Kalantidis, “Decoupling representation and classifier for long-tailed recognition,” arXiv preprint arXiv:1910.09217, 2019.

[40] X. Cheng, F. Shi, Y. Zhang, H. Li, X. Liu, and S. Chen, “Frame: Feature rectification for class imbalance learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 3, pp. 1167–1181, 2025.

[41] S. Park, Y. Hong, B. Heo, S. Yun, and J. Y. Choi, “The majority can help the minority: Context-rich minority oversampling for long-tailed classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 6887–6896.

[42] L. Shen, Z. Lin, and Q. Huang, “Relay backpropagation for effective learning of deep convolutional neural networks,” in Computer Vision– ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part VII 14. Springer, 2016, pp. 467–482.

[43] T. Wang, Y. Zhu, C. Zhao, W. Zeng, J. Wang, and M. Tang, “Adaptive class suppression loss for long-tail object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 3103–3112.

[44] C. Feng, Y. Zhong, and W. Huang, “Exploring classification equilibrium in long-tailed object detection,” in Proceedings of the IEEE/CVF International conference on computer vision, 2021, pp. 3417–3426.

[45] Y. Li, T. Wang, B. Kang, S. Tang, C. Wang, J. Li, and J. Feng, “Overcoming classifier imbalance for long-tail object detection with balanced group softmax,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 10 991–11 000.

[46] J. Wu, L. Song, T. Wang, Q. Zhang, and J. Yuan, “Forest r-cnn: Largevocabulary long-tailed object detection and instance segmentation,” in Proceedings of the 28th ACM international conference on multimedia, 2020, pp. 1570–1578.

[47] Y. Ma, L. Jiao, F. Liu, S. Yang, X. Liu, and P. Chen, “Feature distribution representation learning based on knowledge transfer for long-tailed classification,” IEEE Transactions on Multimedia, 2023.

[48] S. Zhang, Z. Li, S. Yan, X. He, and J. Sun, “Distribution alignment: A unified framework for long-tail visual recognition,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2361–2370.

[49] G. Ghiasi, Y. Cui, A. Srinivas, R. Qian, T.-Y. Lin, E. D. Cubuk, Q. V. Le, and B. Zoph, “Simple copy-paste is a strong data augmentation method for instance segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2918–2928.

[50] Y. Zang, C. Huang, and C. C. Loy, “Fasa: Feature augmentation and sampling adaptation for long-tailed instance segmentation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 3457–3466.

[51] J. B. Tenenbaum, V. d. Silva, and J. C. Langford, “A global geometric framework for nonlinear dimensionality reduction,” science, vol. 290, no. 5500, pp. 2319–2323, 2000.

[52] N. Lei, D. An, Y. Guo, K. Su, S. Liu, Z. Luo, S.-T. Yau, and X. Gu, “A geometric understanding of deep learning,” Engineering, vol. 6, no. 3, pp. 361–374, 2020.

[53] Z. Burda and A. Jarosz, “Cleaning large-dimensional covariance matrices for correlated samples,” Physical Review E, vol. 105, no. 3, p. 034136, 2022.

[54] D. L. Donoho, M. Gavish, and I. M. Johnstone, “Optimal shrinkage of eigenvalues in the spiked covariance model,” Annals ofstatistics, vol. 46, no. 4, p. 1742, 2018.

[55] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards realtime object detection with region proposal networks,” Advances in neural information processing systems, vol. 28, 2015.

[56] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[57] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer, 2014, pp. 740–755.

[58] K. Tong and Y. Wu, “Rethinking pascal-voc and ms-coco dataset for small object detection,” Journal of Visual Communication and Image Representation, vol. 93, p. 103830, 2023.

[59] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu et al., “Mmdetection: Open mmlab detection toolbox and benchmark,” arXiv preprint arXiv:1906.07155, 2019.

[60] T.-I. Hsieh, E. Robb, H.-T. Chen, and J.-B. Huang, “Droploss for longtail instance segmentation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 2, 2021, pp. 1549–1557.