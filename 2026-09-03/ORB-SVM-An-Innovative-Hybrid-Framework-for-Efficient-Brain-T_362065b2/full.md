# ORB-SVM: An Innovative Hybrid Framework for Efficient Brain Tumor Detection from MRI Scans

Amirhosein Azarpour<sup>∗</sup>

Department of Computer Science

Shahid Beheshti University

Tehran, Iran

amirazarpour07@gmail.com

Abstract—Brain cancer remains one of the most significant challenges in modern medicine, where the accuracy of early stage diagnosis is a decisive factor in patient survival and treatment efficacy. Although Magnetic Resonance Imaging (MRI) is the established gold standard for visualizing neurological structures, the interpretation of these high dimensional scans is often complicated by subjective variability among practitioners and the inherent noise present in complex medical images. While contemporary approaches frequently rely on high parameter deep learning architectures, such models often involve significant computational costs and require extensive data for effective training. This study introduces a hybrid framework that utilizes the Oriented FAST and Rotated BRIEF (ORB) algorithm for precise feature extraction and a Support Vector Machine (SVM) for classification [1], [2]. The proposed approach achieves a substantial data reduction of approximately 99.5%, which effectively minimizes the influence of non informative background data while preserving critical diagnostic patterns essential for tumor identification. By balancing feature sparsity with a robust kernel based classifier, this methodology addresses the limitations of over parameterized systems while maintaining high diagnostic integrity. Experimental evaluations conducted on the Br35H dataset demonstrate that the framework attains a classification accuracy of 97.5%. The findings suggest that the integration of localized feature representation and optimized classification provides a reliable and resource efficient alternative for medical image analysis, offering a structured solution that maintains performance without the need for extensive computational overhead.

Index Terms—ORB, Support Vector Machine, Brain Tumor, Data Compression, Medical image processing, MRI images

## I. INTRODUCTION

Brain neoplasms represent a persistent and formidable challenge within modern oncology, where diagnostic precision and timely intervention are decisive factors in patient prognosis. This research proposes a robust diagnostic methodology that integrates Oriented FAST and Rotated BRIEF (ORB) for feature extraction with a Support Vector Machine (SVM) for classification [3]. Beyond its technical framework, the significance of this approach lies in its potential to mitigate the inherent inconsistencies associated with manual radiological interpretation. By automating the feature identification process, the proposed system aims to reduce the subjective variability often encountered during the clinical evaluation of complex MRI scans.

The implications of this framework extend to the broader field of medical image analysis, offering a reliable alternative to traditional diagnostic workflows. The automation of both feature extraction and subsequent classification enhances the consistency of the results, thereby minimizing the reliance on subjective judgments and reducing the ambiguity often associated with pathological patterns in neuroimaging. Furthermore, the modular architecture of this methodology suggests potential adaptability to other oncological domains or neurological disorders, provided the inherent variability of medical imaging datasets is carefully addressed.

The synergy of the selected feature representation and classification techniques achieves high accuracy while maintaining operational efficiency, thus contributing to the ongoing integration of machine learning in clinical environments. This research positions itself as a balanced contribution to a field often characterized by the tension between cautious skepticism and rapid technological adoption. Future developments involve the expansion of training datasets and the potential for clinical integration to validate the model’s efficacy in real-world diagnostic settings. While the application of this specific hybrid framework to MRI based tumor detection represents a significant advancement, a measured perspective is maintained regarding its long term maturation in complex clinical infrastructures.In contrast to high parameter architectures such as Convolutional Neural Networks (CNNs), which typically require intensive pixel-level analysis, the proposed methodology adopts a selective feature oriented strategy [4]. While deep learning models often necessitate substantial computational resources due to the high dimensionality of MRI data, the current approach focuses on extracting only the most salient descriptors. This reduction in the computational burden ensures that the system remains resource efficient and suitable for the operational constraints of clinical environments, where diagnostic processing must be both reliable and timely [5], [6]. Experimental validation on a diverse MRI dataset yielded a classification accuracy of 97.5%. Given the global prevalence of brain tumors—where meningiomas alone constitute approximately one-third of diagnosed cases—the necessity for scalable and accurate diagnostic tools is evident. Magnetic Resonance Imaging (MRI) remains the fundamental modality for neurological imaging due to its superior soft tissue contrast. The technology operates on the principle of nuclear magnetic resonance, where radiofrequency pulses excite protons within the body’s tissues. Upon the cessation of the pulse, the energy released as protons return to their equilibrium state is measured. Most of these signals are derived from hydrogen protons, which are abundant in water and fatty tissues; their behavior within a magnetic field provides the data necessary to visualize the brain’s internal structures.

The diagnostic process is methodical: the patient is subjected to a uniform magnetic field, typically 1.5 Tesla, causing protons to align and precess. A radiofrequency pulse subsequently disrupts this alignment, and the signals captured during the relaxation phase are processed. The variations in the rates at which different tissues return to equilibrium generate the contrast observed in the final images (Fig. 1). This visual narrative is primarily shaped by T1 and T2 relaxation processes, which determine the brightness and texture of the tumorous tissue relative to the surrounding healthy anatomy. Spatial encoding, achieved through magnetic gradients, enables the localization of these signals, culminating in a 2D image through Fourier transformations.

![](images/1c463b9080da7d11993d093d63c76b8ce5292c0196b82f89a48f119327750ab2.jpg)  
Fig. 1. Examples of T1 weighted, T2 weighted and PD weighted MRI scans. Adapted with permission by Kieran Maher using Graphic Converter.

## II. RELATED WORKS

Traditional machine learning methodologies have long established a robust foundation in medical image classification, with Support Vector Machines (SVM) serving as a prominent benchmark for supervised learning tasks [2]. Research by Paredes et al., for instance, utilized localized image patches integrated with k-nearest neighbor classification, demonstrating significant accuracy in early diagnostic models [7]. Subsequently, Parveen and Sathik employed a distinct methodology by combining discrete wavelet transforms with fuzzy C-means clustering for pneumonia detection in thoracic radiographs [8]. Furthermore, Caicedo et al. integrated Scale-Invariant Feature Transform (SIFT) descriptors with SVM classifiers, reporting highly precise results [9]. While these classical approaches significantly advanced the field, they were characterized by a heavy reliance on manual feature engineering and occasionally exhibited limited robustness when faced with variations in image quality or multi scale inconsistencies [7]–[9].

Chandrasekhara et al. proposed a SIFT-SVM framework for prostate cancer detection in MRI scans, extracting SIFT features and performing classification through a multiclass SVM model [10]. The results validated SIFT’s resilience against affine transformations and confirmed its compatibility with SVM in addressing complex classification challenges [10].

Yadav and Jadhav evaluated three distinct strategies for pneumonia detection in chest X-rays: linear SVM with ORB features, transfer learning via VGG16 and InceptionV3, and capsule networks [11]. Their analysis indicated a performance advantage for transfer learning, particularly in scenarios with limited datasets, where convolutional neural networks (CNNs) identified more intricate patterns compared to traditional de scriptors [11]. Similarly, Kermany et al. achieved competitive accuracy in classifying optical coherence tomography images through InceptionV3 transfer learning, demonstrating performance levels comparable to those of experienced clinical experts [12].

## III. MATERIAL AND METHODS

## A. Dataset

The dataset for this work consists of 3000 MRI scans, including two classes of healthy and cancerous brains. These scans originate from hospital patients and cover a range of MRI modalities.The dataset employed in this research is Br35H: Brain Tumor Detection 2020, which is publicly available on the Kaggle platform [13].

## B. Preprocessing

All MRI images are preprocessed prior to feature extraction. Images are resized to 256 × 256 pixels and normalized to [0, 1] range to ensure consistency across different scanners. The dataset of 3000 scans is stratified to maintain class balance and split into training (80%, 2400 images) test (20%, 600 images) subsets.

## C. ORB Algorithm Principle (Orient Fast and Rotated Brief)

Oriented Fast and Rotated Brief (ORB) is based on the famous fast feature detection and brief feature descriptor [1], [14], [15].

## D. Feature Point Detection

Feature points in an image mark locations that carry meaningful visual information places like edges, bright spots in otherwise dark areas, or dark spots in bright regions. The ORB algorithm tackles the problem of identifying these points quickly, using an efficient detection strategy. At the heart of ORB is the FAST method (Features from Accelerated Segment Test), which looks for pixels that noticeably differ from their neighbors. Essentially, a pixel is considered a feature point if its intensity stands out compared to most of the surrounding pixels, as formulated in (1):

$$
N = \sum _ { x \in \mathrm { c i r c l e } ( p ) } | I ( x ) - I ( p ) | > \varepsilon _ { d }\tag{1}
$$

In this framework, $I ( x )$ denotes the grayscale intensity of a point along the circumference surrounding the candidate pixel $p , I ( p )$ represents the intensity of the central pixel, and $\varepsilon _ { d }$ is the intensity difference threshold. A pixel p is considered a feature point if the number of surrounding pixels exceeding this threshold, N, surpasses a predefined proportion, typically three quarters of the circle.

The FAST detection procedure involves selecting a candidate pixel P with intensity M and defining a threshold V, often set to 20% of M, to assess significant intensity differences. Sixteen pixels are sampled along a circle of radius three centered on P. The candidate is identified as a corner if at least L consecutive pixels on the circle are either all brighter than $M + V$ or all darker than $M - V$ . Commonly, L is set to 12, meaning that the candidate is accepted as a feature point if this condition is satisfied; otherwise, it is rejected. The visual representation of this extraction process is shown in Fig. 2.

To enhance computational efficiency, a preliminary test is applied: four pixels at 90 degree intervals around the candidate are first examined. If at least three of these pixels display sufficient intensity difference from the center, the complete evaluation proceeds; otherwise, the candidate is discarded.

![](images/90fec04cc059d0b9435e0f3163f5c5918d020b5c6ae61ff50c89b85da65f08f4.jpg)  
Fig. 2. Schematic diagram of the extraction of FAST feature points.

Since the FAST algorithm does not provide orientation information for feature points, the ORB algorithm introduces two strategies to enhance invariance. First, a three layer Gaussian image pyramid is constructed to improve scale invariance. Second, the gray centroid method is applied to assign a rotation angle to each feature point. Specifically, a coordinate system is established with the feature point as the origin, and the centroid of the neighborhood S is calculated.

The moments of the neighborhood S are computed as shown in (2):

$$
M _ { p q } = \sum _ { x , y } x ^ { p } y ^ { q } I ( x , y ) ,\tag{2}
$$

where $I ( x , y )$ represents the grayscale intensity of the image, and $x , y \in [ - r , r ]$ ], with r denoting the radius of the feature point neighborhood.

The centroid C of the neighborhood is then derived using these moments, as given by (3):

$$
C = \left( \frac { M _ { 1 0 } } { M _ { 0 0 } } , \frac { M _ { 0 1 } } { M _ { 0 0 } } \right) .\tag{3}
$$

The orientation θ of the feature point is finally calculated as formulated in (4):

$$
\theta = \arctan \left( \frac { M _ { 0 1 } } { M _ { 1 0 } } \right) .\tag{4}
$$

To ensure rotation invariance, it is essential that the coordinates x and y are confined within the circular neighborhood of radius r, i.e., $x , y \in [ - r , r ]$ [14].

## E. Calculate Feature Point Descriptors

ORB utilizes the improved BRIEF algorithm to compute the descriptor of a feature point, addressing the primary limitation of BRIEF, namely its lack of rotation invariance. The core idea is to select N point pairs in a specific pattern around a feature point P and combine the comparison results of these N pairs to form the descriptor.

The BRIEF descriptor is based on the assumption that a small image neighborhood can be represented using relatively few intensity comparisons. For an $S \times S$ image patch P, the binary test τ is defined in (5) as:

$$
\tau ( p ; x , y ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } p ( x ) < p ( y ) , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{5}
$$

where $p ( x )$ is the pixel intensity at location $\boldsymbol { x } = ( u , v ) ^ { T }$ in the image patch centered at $P .$ Selecting $N _ { d }$ such point pairs $( x , y )$ uniquely defines the binary criterion. The BRIEF descriptor is thus a binary string of $N _ { d }$ bits, which is formulated in (6):

$$
f _ { N _ { d } } ( \boldsymbol { p } ) : = \sum _ { i = 1 } ^ { N _ { d } } 2 ^ { i - 1 } \tau ( P ; x _ { i } , y _ { i } ) .\tag{6}
$$

The descriptor length can be set to 128, 256, 512 bits, etc., allowing a trade off between speed, storage efficiency, and recognition performance. Since BRIEF considers only single pixels, it is sensitive to noise. To mitigate this, ORB computes each test using a $5 \times 5$ sub-window within a $3 1 \times 3 1$ neighborhood. The schematic diagram of this descriptor calculation is illustrated in Fig. 3.

BRIEF is undirected and lacks rotation invariance. ORB addresses this by assigning an orientation to each descriptor. For any n binary tests at positions $( x _ { i } , y _ { i } )$ , a $2 \times n$ matrix $S$ is defined. Using the neighborhood direction θ and the corresponding rotation matrix $R _ { \theta }$ , the corrected version of the neighborhood $S _ { \theta }$ is given by (7):

$$
S _ { \theta } = R _ { \theta } S .\tag{7}
$$

Thus, the steered BRIEF descriptor becomes:

$$
g _ { n } ( p , \theta ) : = f _ { N _ { d } } ( p ) \mid ( x _ { i } , y _ { i } ) \in S _ { \theta } .\tag{8}
$$

After obtaining the steered BRIEF, a greedy search is performed to select 256 pixel block pairs with the lowest correlation, resulting in the final descriptor [14].

![](images/e82c9daa479605b9d2d2811596620687afb7f07e14a38ec9bf803337a8d93a54.jpg)  
Fig. 3. Schematic diagram of the descriptor calculation

## F. The Flowchart of ORB Algorithm

A flowchart of the ORB algorithm is presented in Fig. 4:

![](images/f3a08f910d62fab686d3e10827c7ab4847dadfa443d96960b81c50aa50b1f7e8.jpg)  
Fig. 4. ORB feature extraction and the matching flowchart.

## G. Methodology of Bag of Words (BoW)

The Bag of Words (BoW) model, originally established in Natural Language Processing (NLP), is utilized in this framework for computer vision to represent MRI scans as a collection of discrete ”visual words” rather than continuous pixel level data [16]. By prioritizing feature frequency over global spatial arrangements, the BoW approach transforms high dimensional local descriptors into a structured and compact feature vector.

Following the ORB feature extraction phase, an extensive set of descriptors is generated. To manage this highdimensional data, K-Means clustering is implemented to categorize similar descriptors into $k$ distinct clusters (where $k ~ = ~ 1 5 0$ in this study). Each cluster centroid $( c _ { j } )$ serves as a ”visual word” within the constructed vocabulary. The optimization objective is to minimize the Sum of Squared Errors (SSE), as defined in (9):

$$
J = \sum _ { j = 1 } ^ { k } \sum _ { i = 1 } ^ { n } \| x _ { i } ^ { ( j ) } - c _ { j } \| ^ { 2 }\tag{9}
$$

where $x _ { i }$ represents the local descriptors and $c _ { j }$ denotes the $j \mathrm { - t h }$ cluster centroid.

For each individual image, descriptors are mapped to the nearest visual word using the Euclidean distance metric, as expressed in (10):

$$
w = \arg \operatorname* { m i n } _ { j } \| f - c _ { j } \|\tag{10}
$$

The occurrence frequency of each visual word is subsequently aggregated to form the final BoW vector. This methodology achieves a substantial data reduction of approximately 99.5%, distilling the complex image information into a concise representation of its most salient pathological patterns.

These normalized vectors, processed via a standard scaling pipeline, serve as the input for the Support Vector Machine (SVM) classifier. While the BoW model omits explicit spatial context, its capacity to compress high dimensional raw data into a refined representation provides a robust and resource efficient mechanism for brain tumor detection.

## H. Support Vector Machines (SVM)

In this phase, the normalized BoW vectors comprising 150 dimensional feature sets processed via StandardScaler are utilized as the primary inputs for a Support Vector Machine (SVM) classifier [2]. The SVM is configured to establish an optimal hyper plane that maximizes the geometric margin between classes while simultaneously minimizing the structural risk of classification error.

Given the training dataset $\left( \mathbf { x } _ { i } , y _ { i } \right)$ , where $\mathbf { x } _ { i } \in \mathbb { R } ^ { 1 5 0 }$ and $y _ { i } \in \{ - 1 , + 1 \}$ , the SVM optimizes the objective function presented in (11):

$$
\operatorname* { m i n } _ { \mathbf { w } , b , \pm } \frac { 1 } { 2 } \| \mathbf { w } \| ^ { 2 } + C \sum _ { i = 1 } ^ { n } \xi _ { i }\tag{11}
$$

subject to the following constraints:

$$
y _ { i } ( \mathbf { w } ^ { T } \phi ( \mathbf { x } _ { i } ) + b ) \geq 1 - \xi _ { i } , \quad \xi _ { i } \geq 0\tag{12}
$$

where the regularization parameter $C$ governs the trade off between margin maximization and the tolerance for misclassification.

To address the non linear distribution within the BoW feature space, a Radial Basis Function (RBF) kernel is employed, as defined in (13):

$$
K ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) = \exp ( - \gamma \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } )\tag{13}
$$

The primary hyperparameters, $C$ and $\gamma ,$ , were utilized with their default configurations $( C = 1 . 0 , \gamma = 1 / n _ { \mathrm { f e a t u r e s } } )$ . Empirically, this configuration yielded the highest classification accuracy of 97.5% across all evaluated parameter variations. The fundamental principle of this classification mechanism is visually demonstrated in Fig. 5.

![](images/63b631ae5cd6b68a28ee15baa4d9cdd836b82c60cd5c8909d681ef981b3e0f1d.jpg)  
Fig. 5. Conceptual representation of a binary classification via a maximummargin SVM hyper-plane.

## I. Methodological Framework

The end-to-end operational sequence of the proposed diagnostic system, ranging from initial image acquisition to final tumor classification, is systematically illustrated in the flowchart provided in Fig. 6.

![](images/a2980d5fd8e93231c0c875b5a7bc327cdcabbd98e117479b273fc08ca36658d6.jpg)  
Fig. 6. Flowchart depicting the sequential stages of the proposed ORB-SVM diagnostic methodology.

## IV. PROPOSED METHODOLOGY FOR BRAIN TUMOR DETECTION

This section delineates a hybrid diagnostic framework for brain tumor detection from MRI scans, integrating Oriented FAST and Rotated BRIEF (ORB) for feature extraction with a Support Vector Machine (SVM) classifier. The initial stage of the methodology involves the application of the ORB algorithm to identify and describe salient keypoints within each image. These keypoints correspond to critical structural cues and intensity variations that distinguish pathological tissue from healthy anatomical structures.

To enhance computational efficiency while maintaining diagnostic integrity, the high dimensional descriptors generated by ORB are subsequently compressed. This selective dimensionality reduction focuses on preserving essential morphological patterns and filtering out redundant information. This process ensures that the resulting feature set remains highly representative of the underlying pathology, facilitating a more robust classification process.

In the final stage, the refined feature set is utilized to train and evaluate an SVM classifier. The SVM establishes an optimal decision boundary to differentiate between malignant and benign instances based on the extracted visual descriptors. By combining the precision of ORB in capturing localized visual structures with the SVM’s capacity for highdimensional classification, the proposed framework improves both diagnostic accuracy and resource efficiency.

## V. RESULTS

In this section, we evaluate the performance of the proposed ORB-SVM framework. The model’s effectiveness is summarized using a confusion matrix and key performance metrics including Accuracy, Precision, and Sensitivity.

The following metrics, as summarized in Table I, demonstrate the model’s robustness:

TABLE I  
PERFORMANCE METRICS OF THE ORB-SVM FRAMEWORK.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Formula</td><td rowspan=1 colspan=1>Calculation</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>TP+TN</td><td rowspan=1 colspan=1>291+294</td><td rowspan=1 colspan=1>97.50%</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1> $\frac { \overline { { T P + F P } } } { \overline { { T P + F P } } }$ </td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>97.00%</td></tr><tr><td rowspan=1 colspan=1>Recall (Sensitivity)</td><td rowspan=1 colspan=1>TP $\underline { { \overline { { T P + F N } } } }$ </td><td rowspan=1 colspan=1>291297</td><td rowspan=1 colspan=1>~98%</td></tr></table>

The ORB-SVM framework demonstrates high efficiency. By reducing the input data by 99.5% through ORB feature extraction and BoW encoding, the model achieves a high accuracy of 97.5% while maintaining a significantly lower computational footprint compared to deep learning architectures.

![](images/a1bf5c9c0afe61d98cc49392e11d99b06c212e43ea367985900f5040b70ac1f2.jpg)  
Fig. 7. Model’s confusion matrix

The confusion matrix shown in Fig. 7 illustrates the distribution of correctly and incorrectly classified instances across the two classes, providing a comprehensive overview of the model’s predictive behavior.

## VI. COMPARISON WITH OTHER MODELS

To evaluate the effectiveness of the proposed ORB-SVM framework, we compare its performance against several established deep learning architectures on the same brain tumor detection task. The experimental results, summarized in Table II, present a comprehensive comparison of model complexity and classification accuracy.

TABLE II  
COMPARISON OF MODEL COMPLEXITY AND CLASSIFICATION ACCURACY.
<table><tr><td>Model</td><td>Architecture</td><td>Parameters</td><td>Accuracy</td></tr><tr><td>Proposed ORB-SVM</td><td> $\mathrm { O R B } + \mathrm { B o W } + \mathrm { S V M }$ </td><td>12,349</td><td>97.50%</td></tr><tr><td>Custom CNN</td><td>3× Conv2D + 2× Dense</td><td>3,284,159</td><td>~97.16%</td></tr><tr><td>Xception</td><td>Depthwise Separable Conv</td><td>20,863,529</td><td>~98.0%</td></tr><tr><td>DenseNet169</td><td>Dense Connections</td><td>12,642,880</td><td>~96.0%</td></tr><tr><td>InceptionResNetV2</td><td>Inception + Residual</td><td>54,338,273</td><td>~66.0%</td></tr><tr><td>ResUNet</td><td>Encoder-Decoder + Attention</td><td>31,575,649</td><td>~93.8%</td></tr></table>

The experimental results reveal several important findings. The proposed ORB-SVM framework achieves 97.50% accuracy with merely 12,349 parameters, which is approximately 1,689 times smaller than Xception (20.86M parameters) and 4,400 times smaller than InceptionResNetV2 (54.34M parameters). This represents a 99.94% less parameters while maintaining competitive accuracy.

Notably, InceptionResNetV2, despite containing 54.34M parameters, achieves only 66% accuracy 31.5% lower than the proposed method. This substantial performance gap suggests that generic deep learning architectures may overfit to noise in medical images, while the ORB feature extraction captures clinically relevant patterns more effectively.

The custom 3 layer CNN achieves slightly lower accuracy (97.16%) than our method and requires 266× more parameters. This trade-off becomes particularly relevant in resource constrained clinical environments where computational efficiency is paramount.

## VII. SUMMARY AND CONCLUSIONS

This research investigated a novel diagnostic framework for brain tumor detection by integrating Oriented FAST and Rotated BRIEF (ORB) feature extraction with a Support Vector Machine (SVM) classifier. This synergy provides a critical balance between computational efficiency and diagnostic fidelity, enabling the system to differentiate between pathological and healthy tissues without compromising the subtle structural details essential for clinical accuracy.

Experimental evaluations demonstrated that the framework achieves a classification accuracy of 97.5%. Furthermore, the substantial dimensionality reduction of approximately 99.5% significantly optimizes the feature space, reducing the risk of overfitting and ensuring the model’s robustness. Due to its minimal requirement for high end computational infrastructure, this methodology remains highly feasible for deployment in medical institutions with limited technical resources, offering a scalable alternative to parameter heavy architectures.

Despite these promising results, certain limitations remain. The current study utilized a specific dataset, and the inherent variability in tumor morphology across different stages and types necessitates further investigation. Future work will focus on expanding the diversity of the training datasets to evaluate the framework’s generalizability and to test the model on multi class datasets. Additionally, exploring hybrid architectures that integrate ORB descriptors with deep learning components, such as Convolutional Neural Networks (CNNs), may further enhance feature representation and improve diagnostic precision in complex clinical scenarios.

## REFERENCES

[1] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), Barcelona, Spain, 2011, pp. 2564–2571.

[2] M. Hearst, S. Dumais, E. Osuna, J. Platt, and B. Scholkopf, “Support vector machines,” IEEE Intelligent Systems and their Applications, vol. 13, no. 4, pp. 18–28, Jul. 1998.

[3] A. S. Beevi, S. Ratheesha, K. Saidalavi, and J. J. Chakola, “Feature extraction based on ORB-AKAZE for echocardiogram view classification,” International Journal of Electrical and Computer Engineering Systems, vol. 14, no. 4, pp. 393–400, 2023.

[4] W. Vianna, “Study and development of a computer-aided diagnosis system for classification of chest X-ray images using convolutional neural networks pre-trained for ImageNet and data augmentation,” arXiv preprint arXiv:1806.00839, 2018.

[5] S. Sazal, A. Ahmmed, H. Rana, and et al., “A method for tumour detection on brain MRI image by implementing SVM,” International Journal of Computer Applications, vol. 130, no. 11, pp. 4–8, 2015.

[6] P. Afshar, K. N. Plataniotis, and A. Mohammadi, “Brain tumor type classification via capsule networks,” in Proc. 25th IEEE Int. Conf. Image Processing (ICIP), Athens, Greece, 2018, pp. 3129–3133.

[7] R. Paredes, D. Keysers, T. M. Lehmann, B. Wein, H. Nev, and E. Vidal, “Classification of medical images using local representations,” in Bildverarbeitung fur die Medizin ¨ . Berlin, Heidelberg: Springer, 2002, pp. 71–74.

[8] N. Parveen and M. M. Sathik, “Detection of pneumonia in chest X-ray images,” Journal of X-ray Science and Technology, vol. 19, no. 4, pp. 423–428, 2011.

[9] J. C. Caicedo, A. Cruz, and F. A. Gonzalez, “Histopathology image classification using bag of features and kernel functions,” in Proc. Conf. Artificial Intelligence in Medicine in Europe (AIME). Berlin, Heidelberg: Springer, 2009, pp. 126–135.

[10] S. P. R. Chandrasekhara, M. G. Kabadi, and Srivinay, “A novel SIFT-SVM approach for prostate cancer detection,” Journal of Computer Science, vol. 16, no. 12, pp. 1742–1752, 2020.

[11] S. S. Yadav and S. M. Jadhav, “Deep convolutional neural network based medical image classification for disease diagnosis,” Journal of Big Data, vol. 6, no. 1, p. 113, 2019.

[12] D. S. Kermany, M. Goldbaum, W. Cai, C. C. S. Valentim, H. Liang, K. Zhang, and et al., “Identifying medical diagnoses and treatable diseases by image-based deep learning,” Cell, vol. 172, no. 5, pp. 1122– 1131, 2018.

[13] A. Hamada, “Brain tumor detection (br35h),” Kaggle Dataset Repository, 2020.

[14] Y. Xie, Q. Wang, Y. Chang, and X. Zhang, “Fast target recognition based on improved ORB feature,” Applied Sciences, vol. 12, no. 2, p. 786, 2022.

[15] Q. Chang, X. Chen, X. Li, W. Wang, and J. Miyazaki, “Faster than fast: Accelerating oriented FAST feature detection on low-end embedded GPUs,” ACM Transactions on Embedded Computing Systems, vol. 24, no. 3, pp. 1–22, Apr. 2025.

[16] K. Juluru, H.-H. Shih, K. N. K. Murthy, and P. Elnajjar, “Bag-of-words technique in natural language processing: A primer for radiologists,” Radiographics, vol. 41, no. 5, pp. 1420–1426, 2021.