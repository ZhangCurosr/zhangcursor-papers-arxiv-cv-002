# Rethinking Data Imbalance in Class Incremental Surgical Instrument Segmentation

Shifang Zhao<sup>a,1</sup>, Long Bai<sup>a,b,1,∗</sup>, Kun Yuan<sup>b,c,d</sup>, Feng Li<sup>b</sup>, Jieming Yu<sup>a</sup>, Wenzhen Dong<sup>a</sup>, Guankun Wang<sup>a</sup>, Mobarakol Islam<sup>e</sup>, Nicolas Padoy<sup>c,d</sup>, Nassir Navab<sup>b</sup>, Hongliang Ren<sup>a,∗</sup>

<sup>a</sup>Department of Electronic Engineering, The Chinese University of Hong Kong, Hong Kong, Hong Kong <sup>b</sup>Chair for Computer Aided Medical Procedures, Technical University of Munich, Munich, Germany <sup>c</sup>University of Strasbourg, CNRS, INSERM, ICube, Strasbourg, France <sup>d</sup>IHU Strasbourg, Strasbourg, France <sup>e</sup>UCL Hawkes Institute, University College London, London, United Kingdom

## Abstract

In surgical instrument segmentation, the increasing variety of instruments over time poses a significant challenge for existing neural networks, as they are unable to efectively learn such incremental tasks and sufer from catastrophic forgetting. When learning new data, the model experiences a sharp performance drop on previously learned data. Although several continual learning methods have been proposed for incremental understanding tasks in surgical scenarios, the issue of data imbalance often leads to a strong bias in the segmentation head, resulting in poor performance. Data imbalance can occur in two forms: (i) class imbalance between new and old data, and (ii) class imbalance within the same time point of data. Such imbalances often cause the dominant classes to take over the training process of continual semantic segmentation (CSS). To address this issue, we propose SurgCSS, a novel plugand-play CSS framework for surgical instrument segmentation under data imbalance. Specifically, we generate realistic surgica backgrounds through inpainting and blend instrument foregrounds with the generated backgrounds in a class-aware manner to balance the data distribution in various scenarios. We further propose the Class Desensitization Loss by employing contrastive learning to correct edge biases caused by data imbalance. Moreover, we dynamically fuse the weight parameters of the old and new. models to achieve a better trade-of between the biased and unbiased model weights. To investigate the data imbalance problem in surgical scenarios, we construct a new benchmark for surgical instrument CSS by integrating four public datasets: EndoVis 2017,[ EndoVis 2018, CholecSeg8k, and SAR-RAPR50. Extensive experiments demonstrate the efectiveness of the proposed framework, achieving significant performance improvement against existing baselines. Our method demonstrates excellent potential for clinical applications. The code is publicly available at github.com/Zzsf11/SurgCSS.

Keywords: Continual learning, Robotic surgery, Data imbalance, Instrument segmentation, Data synthesis

## 1. Introduction<sup>.</sup>

Image segmentation of surgical instruments is crucial for advancing fully autonomous robotic surgery, as it provides critical information to understand dynamic and complex surgical scenes (Chen et al., 2024; Nwoye et al., 2023; Maier-Hein et al., 2022; Cheng et al., 2025; Yu et al., 2024b; Ranem et al., 2024). Recently, deep learning-based methods have shown promising results in solving increasingly challenging tasks (Garcia-Peraza-Herrera et al., 2021; Colleoni et al., 2022; Lou et al., 2023; Liu et al., 2021; Alabi et al., 2025; Yu et al., 2024a). However, collecting and aggregating a large-scale dataset for surgical scenarios is dificult in practice due to high storage costs, licensing, and privacy concerns, and the frequent updating of instruments (Zia et al., 2023; Razzak et al., 2018; Wang et al., 2023b; Bai et al., 2023). Most approaches suffer from catastrophic forgetting, where the performance has a significant degradation when the past data are not available. To address this challenge, Continual Learning (CL) methods aim to enable models to learn new classes or tasks by training on new data while alleviating the catastrophic forgetting problem (Chaudhry et al., 2018; Zhao et al., 2024; Li and Hoiem, 2017; Douillard et al., 2020; Li et al., 2025; Ayromlou et al., 2024; Wang et al., 2023c). Nevertheless, previous methods suffer significant performance degradation in robotic surgical scenarios due to biased incremental learning updates, as illustrated in Figure 1, which shows a pronounced bias in the segmentation head weights for new and old classes.

Why is continual segmentation more challenging in robotic surgical scenarios? We believe this stems from the combination of three factors: (i) the data distribution for instrument classes is highly imbalanced due to variations in surgical workflow, with diferent phase durations and erratic occurrences; (ii) the imbalance is exacerbated as the new class has a high probability of occurrence and is highly concentrated, whereas the old class has a low probability of occurrence and is more dispersed; and (iii) the visual similarity of instruments, with minimal diferences near class boundaries, further complicates segmentation. Even worse, the unavailability of exemplar samples due to licensing and privacy constraints makes the task even more challenging. Additionally, attributes specific to surgical scenes lead to increased dificulty. Unlike the relatively stable illumination in general settings, surgical illumination is highly directional and often positioned to highlight specific areas. The anatomical complexity also leads to various illumination degradation scenes. The diversity of surgical instruments and the similar visual characteristics also introduce further complexity. Besides, in surgical scenes, camera viewpoints are often highly constrained, leading to noise and reduced image quality, which further increases the dificulty of applying segmentation models.

To address the class imbalance challenges, classical techniques have been widely explored in machine learning to mitigate imbalanced distributions. Recent studies (He et al., 2021) have analyzed the relationship between class incremental learning and class imbalanced learning, demonstrating that techniques from class imbalanced learning are similar to class incremental learning methods. (Wu et al., 2019; Liu et al., 2022; Zhao et al., 2020) have attributed the performance degradation caused by sample imbalance to biases in the final fully connected layer. These biases are mitigated by incorporating a bias correction layer trained using a balanced validation set (Wu et al., 2019) or normalizing the output weights (Zhao et al., 2020). While these methods provide some improvement, they fall short in the context of continual learning for surgical instrument segmentation, where data imbalance is further exacerbated by privacy restrictions and the inherent complexity of surgical workflows. Current continual semantic segmentation approaches still lack efective strategies to improve performance under imbalanced distribution scenarios, which are typical in surgical environments.

In medical image analysis, data scarcity is often addressed by generating synthetic data, enabling knowledge sharing without exposing patient-level information (Colleoni and Stoyanov, 2021; Colleoni et al., 2022). Similarly, continual learning utilizes generative models to replay synthesized data (Shin et al., 2017; Wu et al., 2018). However, updating generative models incrementally introduces challenges, including high resource consumption and reduced system stability. In this case, Xu (Xu et al., 2024) blended and augmented the foreground to the synthetic background to avoid generating model updates. These methods provide the flexibility to create a class-balanced dataset using any preferred instrument, while also automatically generating the instrument annotations. However, they simply crop patches without instruments from the images and use these patches to train a generative model. In this case, (i) the generated backgrounds are often distorted and may contain unforeseen artifacts and inconsistencies, (ii) the model can only learn foreground information based on segmentation annotations from such data, but due to the distorted backgrounds, it is challenging for the model to extract efective discriminative information between the foreground and the background, and (iii) this approach limits the model to generated background patches, neglecting the efective utilization of the scarce real samples available. Therefore, although this method is straightforward, there is still significant room for improvement.

In this work, we propose SurgCSS, a novel plug-andplay approach specifically designed to mitigate the degradation caused by class imbalance in the continual segmentation of robotic surgical scenes. We identify a significant bias in the segmentation head, analyze its root causes, and propose a twostage pipeline to resolve this issue. Specifically, we decouple the instrument foreground from the surgical background and construct a small class-balanced set using a class-aware blending technique, which is crucial in overcoming data scarcity in surgical scenarios. Additionally, we design the Class Desensitization Loss (CDL) function to reduce sensitivity to class boundary uncertainty, thereby minimizing the impact of visual similarity between classes. After training the model on the small set, we apply a dynamic weight fusion (DWF) strategy to merge the representation capabilities of the trained and untrained models efectively. Our approach efectively addresses the class imbalance between new and old data, as well as the class imbalance within the same time point. We summarize our contributions as follows:

• We propose a novel plug-and-play framework to address performance degradation in class incremental surgical instrument segmentation, specifically targeting imbalanced distributions under data-scarce surgical scenarios.

• We introduce a class-aware blending with an inpainting mechanism to decouple the instrument foreground from the high-fidelity surgical background, efectively tackling the data scarcity issue. The CDL is further proposed to minimize confusion between visually similar classes and correct segmentation boundary deviations.

• We develop a DWF strategy that combines the strengths of both incremental and rebalance models, ensuring detailed class representations are preserved while boosting overall segmentation accuracy.

• By integrating four existing public datasets, we establish a new large-scale benchmark for class incremental segmentation of surgical instrument segmentation. Through extensive comparisons and ablation studies on this benchmark, we demonstrate the potential of our method for clinical applications.

## 2. Related Work

## 2.1. Class-Incremental Continual Learning

Class-Incremental Continual Learning aims for a model to learn distinguishing features for new classes while preventing catastrophic forgetting of previously learned classes. This requires balancing the need for learning flexibility with maintaining memory stability (Wang et al., 2024). In Continual Semantic Segmentation, a specific challenge is background shift, which happens when pixels labeled as background correspond to classes the model has already seen or will see later. Early methods like MiB (Cermelli et al., 2020) addressed this by calibrating the loss for background pixels using prior model predictions. PLOP (Douillard et al., 2021) used a multi-scale pooling approach for distillation. SSUL (Cha et al., 2021) used auxiliary data to improve distillation and prevent forgetting. Other approaches focused on preventing feature changes; RCIL (Zhang et al., 2022) and EWF (Xiao et al., 2023) worked to reduce feature plasticity problems by combining parameters from the new and old models. CIT (Ge et al., 2024) converts the output of existing semantic segmentation models into a form that does not depend on the specific class. IPSeg (Yu et al., 2025) uses image posterior probabilities to align the optimization process across diferent stages and reduce negative efects from optimizing each stage separately. MoDA (Yang et al., 2024) helps the SAM encoder create features that are clearly separated for diferent task types, providing accurate task-specific information for continual learning. EIR (Yin et al., 2025) reduces background changes in new images by combining stored examples with the new images. CoMBO (Fang et al., 2025) adds an extra part to handle learning new classes while preserving the original information used for distillation. These recent methods represent the current advances in general continual segmentation scenarios.

![](images/8cc42682e7fef90f4807703ca941acddce71db92e9e3e90cc70782dd76b6be9d.jpg)

![](images/a7f5eca577b19fac55daa86303e2bb29a090075da3a7f2d99a57c6357ec105a4.jpg)  
Figure 1: Experimental results on a surgical dataset confirm the presence of bias in the final layer, as observed through weight visualization of the segmentation head in task “13-1". Heads 1-13 correspond to the old class, while heads 14-18 represent incremental classes, one by one (Best viewed in color). We visualize the bias within each class-specific output channel of the final convolutional layer using two methods: a summation average of activations (left) and a t-SNE embedding of the feature space (right).

Continual segmentation is also being explored in medical image analysis. (Xu et al., 2024) uses synthetic data with class-aware temperature normalization to enhance model rigidity while preserving privacy in robotic surgery. (Zhang et al., 2023) propose replacing the conventional output layer with a set of lightweight, class-specific heads. (Ji et al., 2023) demonstrates that continually training, followed by freezing the encoder, combined with incrementally added decoders, can efectively extract suficiently representative image features for new classes, allowing them to be validly segmented. (Elskhawy et al., 2020) introduces an approach that disentangles the feature space into task-specific and task-invariant features. (Zhu et al., 2024) employs lightweight low-rank adaptation (LoRA) to eficiently extend pre-trained segmentation models for segmenting new organs. (Sadegheih et al., 2025) introduces a mixture-of-experts mechanism and dual knowledge distillation for brain lesion segmentation. (Anand et al., 2024) proposes the construction of a common label space to facilitate incremental learning across diferent datasets. (Guo et al., 2025) introduces segmentation models that can handle a comprehensive set of fine-grained whole-body anatomies from diverse, partially labeled datasets in CT segmentation.

However, previous continual segmentation methods that use distillation may struggle with visual similarities between diferent instruments or anatomical structures, particularly near class boundaries. Methods focused on parameter fusion or regularization may not inherently address the severe class imbalance. Additionally, while data limitations represent a significant challenge in medical continual learning tasks, previous research in medical image analysis has primarily focused on modifying model architectures to improve performance, often neglecting the critical issue of data scarcity and imbalance. In this paper, we further explore the specific challenges and solutions related to surgical instrument semantic segmentation, a domain marked by imbalanced data, privacy concerns, and the complexity of surgical workflows.

## 2.2. Learning with Class Imbalance

Class imbalance in deep learning is a significant challenge arising from the disproportionate representation of classes in training datasets, which often leads to biased models favoring the majority class. To address this issue, various techniques have been proposed, broadly categorized into datalevel, algorithm-level, and hybrid methods. Data-level methods (Hensman and Masko, 2015; Lee et al., 2016) focus on modifying the dataset to reduce class imbalance by adjusting the training data distribution, either through duplicating samples from the minority class or removing samples from the majority class. In contrast, algorithm-level approaches (Lin, 2017; Wang et al., 2016) involve modifying the learning process or the loss function to account for class imbalance. Recently, researchers have increasingly focused on addressing class imbalance in continual learning. For instance, (He et al., 2021) examined the relationship between class imbalance and performance degradation in continual learning. (Wu et al., 2019) proposed a bias correction method that targets the fully connected layer, addressing the bias caused by class imbalance and efectively narrowing the performance gap between old and new classes. Furthermore, (Liu et al., 2022) tackled class imbalance in longtailed datasets by introducing a learnable weight-scaling layer to mitigate bias. This approach is particularly relevant to continual learning scenarios and aligns closely with our task of semantic segmentation for surgical instruments.

## 2.3. Image Synthesis

Surgical image scarcity, driven by privacy concerns and workflow complexities, challenges deep learning application development. To address this, synthetic datasets blend opensource surgical instrument images with real or simulated backgrounds, enabling targeted data generation (Wang et al., 2022; Garcia-Peraza-Herrera et al., 2021). However, these methods face limitations such as manual data collection costs and limited source availability. Generative models, particularly GANs, have improved data augmentation by synthesizing surgical images (Zafino et al., 2021; Shin et al., 2018; Hamghalam et al., 2020; Lee et al., 2019; Rivoir et al., 2021), and unpaired image-to-image translation has enhanced image quality (Pfeiffer et al., 2019). Recently, difusion models (DMs) have demonstrated superior performance, employing semantic maps for high-quality surgical image synthesis (Zhou et al., 2024) and latent consistency-distilled approaches for unpaired translation (Venkatesh et al., 2024). Advanced methods now generate high-resolution surgical videos from text prompts alone (Cho et al., 2024). Compare to other data augmentation methods like copypaste (Ghiasi et al., 2021) and cutpaste (Li et al., 2021), image synthesis methods ofer the capability to generate entirely new visual contexts and object appearances, create variations that extend beyond basic geometric or color transformations, and potentially produce more realistic and diverse data that better reflects the complexity of surgical environments. Our inpainting-based approach further addresses data limitations by generating realistic, class-balanced datasets.

## 3. Methodology

The proposed two-stage method addresses a continual learning scenario relevant to real-world clinical applications, where models must adapt incrementally to new classes after deployment. The training process is divided into two stages, as illustrated in Figure 2. In the incremental stage, similar to existing methods, the process begins with a model that has already been trained on prior classes at step $t - 1$ . The goal is to incrementally update the model to recognize a new class at step t while preserving its performance on previously learned classes. We introduce an additional rebalance stage to address the class imbalance problem caused during incremental training. In this stage, the model from the incremental stage undergoes a posttraining process specifically designed to further enhance performance. Unlike conventional approaches, the rebalance stage balances weight updates while keeping the backbone parameters frozen. This is achieved using a synthetic dataset for finetuning, ensuring that core feature extraction remains unaltered. By introducing the rebalance stage, our method efectively addresses challenges in class-incremental learning, significantly improving robustness and stability in robotic surgical settings.

## 3.1. Problem Formulation: Incremental Stage

Let us define the dataset at each step as $\mathcal { D } _ { t } = \{ x _ { i } , y _ { i } \}$ , where $x _ { i }$ represents the data and y is the associated label. The label space for the training task $T$ is given as $C _ { t } \cup \{ c _ { b } \}$ , where $C _ { t }$ contains all classes involved in the task, and $c _ { b }$ corresponds to the background. Typically, classification loss is handled using cross-entropy loss. Due to the disjoint nature of class sets across tasks, i.e., $C _ { i } \cap C _ { j } = \mathcal { O }$ , the objectives difer significantly between tasks, often causing catastrophic forgetting. To mitigate this, knowledge distillation is a widely adopted method. In continual learning, distillation is generally categorized into two types: feature-based distillation and logits-based distillation, which can be expressed as:

$$
\mathcal { L } _ { F D } = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { ( x _ { i } , y _ { i } ) \sim \mathcal { D } _ { t } } \Vert \Psi _ { o l d } ( x _ { i } ) - \Psi _ { n e w } ( x _ { i } ) \Vert ^ { 2 }\tag{1}
$$

$$
\mathcal { L } _ { L D } = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { ( x _ { i } , y _ { i } ) \sim \mathcal { D } _ { t } } K L ( \Phi _ { o l d } ( \Psi _ { o l d } ( x _ { i } ) ) , \Phi _ { n e w } ( \Psi _ { n e w } ( x _ { i } ) ) )\tag{2}
$$

$\Psi _ { o l d } / \Phi _ { o l d }$ and $\Psi _ { n e w } / \Phi _ { n e w }$ denote the feature extractor/classifier from the previous old model and incremental new model, respectively, and D is the corresponding dataset.

The softmax cross-entropy loss is used as the classification loss, which is computed as follows:

$$
\mathcal { L } _ { C E } = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { ( x _ { i } , y _ { i } ) \sim \mathcal { D } _ { t } } - y _ { i } \log { [ \Phi _ { n e w } ( x _ { i } ) ] }\tag{3}
$$

$\Phi _ { n e w } ( x _ { i } )$ is the output probability of new classifier.

This approach introduces a significant bias toward new classes due to the imbalance between the abundant samples of new classes and the limited exemplars from old classes. During training, the current model starts with the weights of the previous model, with an additional output channel added to handle segmentation for the new class. These class-specific parameters are highly prone to overfitting on the current class. By examining the weights of the segmentation head for both new and old classes, as illustrated in Figure 1, it is evident that the bias in class predictions results from uneven weight updates. Our work aims to address and correct the bias in the segmentation head.

![](images/173830b720b2ddcce9d4350a8a68052fa563770af25390558f30eb46f7bc07bf.jpg)  
Figure 2: Framework of Our Two-Stage Class-Incremental Continual Segmentation Method. Following the incremental training stage, similar to previous ap proaches, we introduce (a) an additional rebalance stage using a synthetic balanced dataset. During this stage, (b) a Class Desensitization Loss is applied to reduce class bias introduced in the incremental stage. A Dynamic Weight Fusion, shown in the purple dotted box, is performed between the biased model and the unbiased model to integrate knowledge derived from more diverse real-world data.

## 3.2. Proposed Method: Rebalance Stage

Building on our finding that the segmentation head is heavily biased, we propose a simple and efective bias correction method. Our approach introduces additional training stages, as illustrated in Figure 2. The basic idea is to alleviate the bias by learning a balanced distribution of both old and new classes (shown in Fig 2). First, we train the feature extractor and segmentation head using the baseline method. In the rebalance stage, we freeze the feature extractor and fine-tune the segmentation head on a synthetic, class-balanced set with a class-desensitizing loss, using a few epochs to address the imbalance between new and old classes through balanced updates. After fine-tuning, we apply a DWF between the original and fine-tuned weights to integrate prior knowledge, enhancing the model’s generalization ability. This section details the generation of the synthetic class-balanced set, the class-desensitizing loss, and the DWF method.

## 3.2.1. Inpaint and Blending

A balanced dataset is essential for rebalancing and removing bias. However, in incremental surgical segmentation, acquiring such a dataset is challenging. We pioneer the integration of inpainting and blending techniques into the continual learning setting. Specifically, we utilize an inpainting model to generate background images. The foreground instrument is cropped separately and then overlaid onto the generated background images to address data limitations. We carefully control the sample class distribution to avoid class imbalance, efectively addressing issues of data scarcity and reducing the need for costly, time-intensive labeling.

In surgical settings, backgrounds are highly valuable, but due to privacy concerns and operational constraints, obtaining and storing a large number of background images is not feasible. Available samples are often coupled with specific foreground categories, resulting in an unbalanced dataset. Although recent eforts have used generative models for data synthesis, these models often struggle with high-quality generation due to the detailed nature of surgical backgrounds and the similarity of many surgical instruments in the foreground (Xu et al., 2024). By using inpainting, we can reuse existing backgrounds and flexibly create a balanced dataset. Compared to direct generative models, inpainting provides better detail preservation as it retains more of the original information.

Given the remarkable performance demonstrated by difusion models recently, we chose to use latent difusion models (LDM) to generate the background images (Rombach et al., 2022). However, robotic surgical images difer significantly from natural images, so directly applying existing models is not ideal due to domain gaps. To address this, we fine-tune the pre-trained LDM on our surgical dataset. Specifically, we first fine-tune the pre-trained LDM autoencoder, and then train the U-Net component. During training, we mask out the surgical background regions in the images, allowing the model to learn the background inpainting process. Subsequently, during inference, the inpainting model is used to remove the foreground surgical instruments from the given data samples, generating clean background images. The inpainting data samples are il lustrated in Figure 3. The dataset is denoted as ${ \mathcal { D } } _ { r } = \{ { \mathcal { B } } , { \mathcal { F } } \}$ where B represents the background obtained through inpainting, and $\mathcal { F } = \{ f ^ { i } \}$ denotes the foreground instrument, with each $f ^ { i }$ corresponding to diferent class i.

![](images/900d3ffa6e587af0f44e678a77f746d96a412857a905d878f6dbda3f43df02da.jpg)  
Figure 3: Construction of a rebalance dataset through inpainting: During training, the model performs a reconstruction task on background-masked regions. At inference time, it is used to remove existing foreground instruments to isolate the background.

In the surgical setting, various environmental disturbances can lead to performance deviations between training and test datasets. Insuficient model robustness under these conditions may result in prediction errors, potentially compromising procedural safety (Garcia-Peraza-Herrera et al., 2021; Wang et al., 2023a). To address this, we enhance both backgrounds and foregrounds through augmentations such as rotation, scaling, and flipping. In this procedure, we allow up to a specified number of surgical instruments to appear simultaneously in a single background image. We then blend the augmented foreground and background images to generate composite images along with corresponding multi-class masks.

## 3.2.2. Class Desensitization Loss

After obtaining clean background images, our next challenge is to use these backgrounds for model training in a way that allows for balanced updates, reducing bias, and improving performance. Additionally, it is also a challenge to avoid introducing a gap between synthetic and real data. As demonstrated in FIG-URE 1, the average summation and t-SNE have shown that the segmentation heads have a strong bias toward certain classes.

Motivated by contrastive learning (Michieli and Zanuttigh, 2021; Wu et al., 2023), we design the Class Desensitization Loss (CDL) as our objective function, employing contrastive learning with class-aware blending to mitigate the bias introduced in incremental stages. Contrastive learning supports representation learning for both old and new classes by enhancing intra-class compactness and inter-class separation. By enabling the model to learn more robust and discriminative feature representations, CDL also addresses biases that might arise from the synthetic-real domain shift by acting as a corrective mechanism.

We use our existing backgrounds and instrument foregrounds to perform class-aware blending. First, we randomly select an instrument foreground and blend it into the background according to the blending method described above. Then, we use the existing model to predict the label of this foreground. As previously discussed, biased models tend to misclassify certain classes as other similar classes, which may lead to errors in category prediction when applied to the initial blended image. Based on the predicted label, we select a foreground of the predicted class and blend it again. The resulting samples are used for training and loss computation, as shown in Fig 2(b). The overall algorithm of CDL is shown in Algorithm 1.

Algorithm 1 Class Desensitization Loss   
Input: Model F; rebalanced dataset $\mathcal { D } _ { \mathrm { r e b a l a n c e } } ~ = ~ \{ \mathcal { B } , \mathcal { F } \} ;$ blending   
number N   
Output: Class Desensitization Loss $\mathcal { L } _ { \mathrm { C D } }$   
1: Randomly select a foreground object $f _ { 0 } ^ { i }$ and a background $b _ { 0 }$ from   
dataset D   
2: Blend $f _ { 0 } ^ { i }$ with $b _ { 0 }$ to create a blended image $I _ { 0 }$ and assign it label   
$Y _ { 0 }$   
3: Use model $\mathcal { F }$ to predict the label $\hat { Y } _ { 0 }$ for the foreground in $I _ { 0 }$   
4: if $\hat { Y } _ { 0 }$ is the background class then   
5: Set the new blending class $C \gets i$   
6: else if $\hat { Y } _ { 0 } \neq Y _ { 0 }$ then   
7: Set the new blending class $C \gets c ,$ where c is the incorrectly   
predicted class   
8: end if   
9: for $j = 1$ to N do   
10: Randomly select a foreground object $f _ { j } ^ { C }$   
11: Blend $f _ { j } ^ { C }$ into $I _ { j - 1 }$ to create a blended image $I _ { j }$ with label $Y _ { j }$   
12: end for   
13: Compute the loss $\mathcal { L } _ { \mathrm { C D } } ( \mathcal { F } ( \mathrm { I } _ { j } ) , Y _ { j } )$

The samples contain diferent classes that can confuse the model, which can be addressed by the CDL using contrastive learning, defined as:

$$
\mathcal { L } _ { C D } = \frac { 1 } { 2 } \left( y \cdot \Vert \mathbf { x } _ { 1 } - \mathbf { x } _ { 2 } \Vert _ { 2 } ^ { 2 } + ( 1 - y ) \cdot \operatorname* { m a x } ( 0 , m - \Vert \mathbf { x } _ { 1 } - \mathbf { x } _ { 2 } \Vert _ { 2 } ^ { 2 } ) \right)
$$

$$
\mathcal { L } _ { C E } = - \sum _ { c = 1 } ^ { C } Y \log ( \hat { Y } )\tag{4}
$$

(5)

$$
\mathcal { L } _ { B C E } = - [ Y \log ( \hat { Y } ) + ( 1 - Y ) \log ( 1 - \hat { Y } ) ]\tag{6}
$$

where $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 }$ represent the features of the confused classes, computed via global averaging. The variable y serves as the label to indicate the positive or negative class, Y and $\hat { Y }$ are the predicted probabilities and ground truth, and m represents the margin for the contrastive diference. This loss operates alongside the conventional pixel-wise cross-entropy loss, $L _ { C E } ,$ and the binary cross-entropy loss, $L _ { B C E }$ , ofering a complementary approach to improve segmentation performance. The combination is intentional: while pixel-wise cross-entropy loss focuses on predicting the correct label for each sample, CDL targets effective data representation by considering the relationships between diferent samples.

Thus, the training objective is to optimize the following combined loss:

$$
\mathcal { L } = \alpha \mathcal { L } _ { C D } + \beta ( \mathcal { L } _ { C E } + \mathcal { L } _ { B C E } )\tag{7}
$$

![](images/f905c8efd391d2b555b88390e812bc7e12c1b333c13abf7b7918d314c8656967.jpg)  
Figure 4: Illustration of the class distribution in the collected dataset, ordered from the most frequent to the least frequent class.

where $\alpha , \beta$ represent the respective weights for each loss component.

## 3.2.3. Dynamic Weight Fusion

The primary concept of our Dynamic Weight Fusion (DWF) is to enhance the model’s generalization capability while maintaining balanced representations. Although the incremental model may exhibit a significant bias toward certain classes, it often encodes more detailed knowledge about these classes. This is because the training dataset during the incremental stage is typically more diverse than the one used in the rebalance stage. To efectively retain general knowledge while preserving an unbiased representation, we propose the DWF strategy to further improve performance. Specifically, after the rebalance stage, we have two models: one trained during the incremental stage $( \theta _ { \mathrm { o r i } } )$ and another after the rebalancing process $( \theta _ { \mathrm { r e } } )$ . We introduce an additional parameter λ, which serves as a weight fusion factor, enabling us to combine the parameters of these two models in a specified ratio. This ratio acts as a balancing factor. The operation for dynamic weight fusion can thus be expressed as:

$$
\theta _ { \mathrm { f u s e d } } = \lambda \cdot \theta _ { \mathrm { o r i } } + ( 1 - \lambda ) \cdot \theta _ { \mathrm { r e } }\tag{8}
$$

This formulation ensures that the model benefits from both the detailed knowledge retained in $\theta _ { \mathrm { o r i } }$ and the balanced representation provided by $\theta _ { \mathrm { r e } } .$ allowing for improved performance while mitigating biases. While λ serves as a control ratio for parameter magnitudes, we also introduce a constraint on the selection of fusion parameters to mitigate the bias present in $\theta _ { \mathrm { o r i } }$ . To address this, we define $\alpha ,$ a dynamic threshold parameter, based on the most variable parameters after rebalancing. These parameters are likely to represent general knowledge learned from a large dataset, which is at risk of being lost during rebalancing. The threshold α is computed as follows:

$$
\alpha = \mathrm { m i n } ( \mathrm { t o p } \mathrm { - } k ( | \theta _ { \mathrm { o r i } } - \theta _ { \mathrm { r e } } | ) ) ,\tag{9}
$$

![](images/aade03c6becb0f10503862bcfaaa41ff057c00d5ad0e887ca7c004c8f1a3e82a.jpg)  
Figure 5: Demonstration of visually similar samples across diferent classes in our dataset. We present several surgical instruments with similar appearances, including Bipolar Forceps, Prograsp Forceps, Grasping Retractor, and Grasper.

where $\mathrm { t o p } { - } k ( | \theta _ { \mathrm { o r i } } - \theta _ { \mathrm { r e } } | )$ identifies the k-largest diferences between $\theta _ { \mathrm { o r i } }$ and $\theta _ { \mathrm { r e } } ,$ capturing the parameters most afected by rebalancing. The overall strategy can be expressed as:

$$
\theta _ { \mathrm { f u s e d } } = \left\{ \begin{array} { l l } { \theta _ { \mathrm { o r i } } , } & { \mathrm { i f ~ } | \theta _ { \mathrm { o r i } } - \theta _ { \mathrm { r e } } | < \alpha , } \\ { ( 1 - \lambda ) \theta _ { \mathrm { o r i } } + \lambda \theta _ { \mathrm { r e } } , } & { \mathrm { i f ~ } | \theta _ { \mathrm { o r i } } - \theta _ { \mathrm { r e } } | \ge \alpha , } \end{array} \right.\tag{10}
$$

indicating that weight fusion is applied selectively to param eters likely to be significantly influenced by class imbalance. This targeted fusion ensures the preservation of general knowledge while addressing the issues introduced during the incremental training stage.

## 4. Experiments

## 4.1. Implementation

## 4.1.1. Datasets

We collected a new dataset to assess our method. The motivation behind collecting the new dataset stems from the limitations identified in previous datasets, which include a restricted number of instrument classes and the challenge of class imbalance in real-world surgical workflows. In surgical environments, some instruments are used more frequently than others, leading to highly imbalanced data distributions. Addressing this imbalance is crucial for evaluating the robustness of Continual Learning (CL) models under realistic conditions, where the model must adapt to new classes without access to past data (a scenario prone to catastrophic forgetting).

Our dataset is merged from four public datasets: EndoVis 2017 Dataset (Allan et al., 2019), EndoVis 2018 Dataset (Allan et al., 2020), CholecSeg8k (Hong et al., 2020), and SAR-RAPR50 (Psychogyios et al., 2023). The EndoVis 2017 and 2018 datasets primarily focus on the segmentation of robotic instruments in minimally invasive surgeries, ofering detailed pixel-wise annotations for da Vinci robotic tools, including articulating parts such as shafts, jaws, and wrists, as well as surgical devices like ultrasound probes and suturing instruments. CholecSeg8k provides segmentation data for key instruments used in laparoscopic cholecystectomy, including graspers and electrocautery tools, alongside complex anatomical backgrounds. SAR-RARP50 contains segmentation annotations for robotic instruments in the context of radical prosta tectomy surgeries. Our newly collected dataset expands the number of instrument classes compared to previous datasets, as shown in Figure 4. The integrated dataset contains 18 instrument classes, with a highly imbalanced distribution. It includes 66 videos with a total of 26,638 frames, of which 19,933 are in the training set and 6,705 are in the test set. The resolution ranges from 1920 to 800 pixels, originating from RGB endoscopic videos. Each frame was annotated instance by instance, with separate annotated masks provided for each object. Additionally, the dataset includes several visually similar classes, resembling real robotic surgical scenes. FIGURE 5 presents examples of categories with similar appearances.

## 4.1.2. Experiments Setting

In our surgical instrument segmentation scenario, we assume that the training data from the time point(t=0) is no longer accessible during the subsequent incremental learning stages. We define four tasks: 13-1 (5 steps), 13-5 (1 step), 9-3 (3 steps), and 17-1 (1 step), all set under the overlapped protocol. “x-y" denotes that x is the number of old classes and y is the number of incremental classes introduced in each step. At each step, the model has access to the new dataset, but previous datasets are not available for further training. Similar to the overlapped protocol described earlier, the training set at each stage may include images containing pixels belonging to classes that are yet to be learned in future steps, with these pixels labeled as background. This setting allows us to evaluate the efectiveness of the incremental learning approach when handling new instrument classes while dealing with the limitations imposed by inaccessible prior data.

## 4.1.3. Implementation Details

We used the DeepLabv3+ architecture (Chen et al., 2018) with the ResNet-101 backbone (He et al., 2016) for segmentation tasks. We compared our proposed method against the following baselines and state-of-the-art solutions: EWC (Kirkpatrick et al., 2017), LwF (Li and Hoiem, 2017), LwF-MC (Rebufi et al., 2017), ILT (Michieli and Zanuttigh, 2019), MiB (Cermelli et al., 2020), RCIL (Zhang et al., 2022), CAT-SD (Xu et al., 2024), MBS (Park et al., 2024), NeST (Xie et al., 2024), PLOP (Douillard et al., 2021), and EWF (Xiao et al., 2023). Consistent with prior approaches, in-place activated batch normalization was set in the backbone. All input images were resized to 224×224 for uniformity. Using the SGD optimizer, we set the initial learn rate as 0.06 and 0.006 for the continual learning steps, and the batch size is set to 12. For the incremental stage, we set the training to 20 epochs per step. Taking the 13-1 (5 steps) setting as an example, PLOP requires approximately 6 h for training, and other baseline methods also require similar training time. The rebalance stage is set to 1 step with 3 epochs, which takes around 0.5 h. The training of the rebalance stage requires around 9 GB of VRAM. All experiments are conducted on two NVIDIA RTX 3090 GPUs.

The model testing and weight initialization followed the same approach as outlined earlier. Specifically, the bestperforming model from each step was selected based on the highest mean Intersection over Union (IoU) achieved on the validation set. For each continual learning step, the model was initialized with the weights of the previously trained model, ensuring that knowledge from earlier steps was retained. During the continual learning step, the old model acted as a teacher, providing logits for distillation loss calculation, while the model was optimized using both cross-entropy loss and distillation losses. This approach enabled us to learn new classes incrementally while maintaining performance in previously learned classes.

## 4.2. Experimental Results

We conduct extensive experiments and evaluations to vali date our proposed approach and investigate the efectiveness of privacy-preserving continual learning in robotic surgery. Our method is compared against several state-of-the-art continual learning approaches, which can be categorized into four groups. One category includes approaches based on penalty computation, such as EWC (Kirkpatrick et al., 2017), which employ strategies to compute the importance of parameters for old classes, safeguarding those critical parameters to prevent forgetting. The second category consists of methods based on knowledge distillation, originally designed for continual image classification, including LwF (Li and Hoiem, 2017) and LwF-MC (Rebufi et al., 2017). These methods have demonstrated impressive performance when applied to semantic segmentation tasks. The third category focuses on approaches specifically designed for continual semantic segmentation, such as ILT (Michieli and Zanuttigh, 2019), which combines both feature- and output-level knowledge distillation, and MiB (Cermelli et al., 2020), which relies solely on output-level knowledge distillation to address the background shift problem. Recently, some weight fusion methods have been proposed to im prove knowledge transfer and enhance model robustness, including RCIL (Zhang et al., 2022) and EWF (Xiao et al., 2023). Additionally, we record the results of the fine-tuning approach (FT) as the lower bound for segmentation performance, since FT updates the model with new classes without any additional constraints, leading to the most severe catastrophic forgetting. The upper bound is represented by the results from the offline training approach (Ofline), where images and annotations for all classes are available during training. To ensure a fair comparison, we implement all competitors using the same base segmentation network, as well as identical computing environments and common hyperparameter settings.

In TABLE 1 and TABLE 2, we present a statistical comparison of our method with existing approaches across tasks 13- 1, 13-5, 9-3, and 17-1 on our surgical dataset, as well as task 13-5 on the four-component dataset. Our observations indicate that, regardless of the method employed, old class forgetting remains a challenge when learning new classes. Furthermore, learning new classes can be negatively impacted when methods impose excessive constraints. While weight fusion methods maintain base class performance, they limit the plasticity required for learning new classes. By addressing the class bias between base and new classes, our method not only preserves the knowledge of base classes but also enhances their performance. For instance, on the 13-1 setting, our algorithm achieves performance gains of 27.5% and 6.15% mIoU for PLOP and

Table 1: Performance comparison across diferent continual learning tasks (13-1, 13-5, 9-3, and 17-1, “x-y" denotes that x is the number of old classes and y is the number of incremental classes introduced in each step). Results are reported for methods without continual learning (WO CL), including the old model at time point t = 0 and naive fine-tuning (FT), and those with continual learning (W CL). The metrics presented are for base, new, and all classes, with the ofline training serving as an upper bound.
<table><tr><td rowspan="2">Method</td><td colspan="3">13-1 (5 steps)</td><td colspan="3">13-5 (1 step)</td><td colspan="3">9-3 (3 steps)</td><td colspan="3">17-1 (1 step)</td></tr><tr><td> $B a s e$ </td><td> $N e w$ </td><td>All</td><td> $B a s e$ </td><td> $N e w$ </td><td>All</td><td> $B a s e$ </td><td> $N e w$ </td><td>All</td><td> $B a s e$ </td><td> $N e w$ </td><td>All</td></tr><tr><td>Old Model (t=0)</td><td>61.29</td><td>0.00</td><td>45.16</td><td>61.29</td><td>0.00</td><td>45.16</td><td>60.62</td><td>0.00</td><td>31.90</td><td>56.93</td><td>0.00</td><td>53.93</td></tr><tr><td>FT</td><td>0.00</td><td>17.67</td><td>8.53</td><td>0.00</td><td>32.39</td><td>13.71</td><td>0.00</td><td>10.99</td><td>9.52</td><td>0.00</td><td>5.49</td><td>8.77</td></tr><tr><td>EWC (Kirkpatrick et al., 2017)</td><td>0.00</td><td>10.98</td><td>5.30</td><td>0.00</td><td>52.60</td><td>22.25</td><td>0.00</td><td>7.65</td><td>8.75</td><td>0.00</td><td>5.47</td><td>8.73</td></tr><tr><td>LwF (Li and Hoiem, 2017)</td><td>1.48</td><td>21.82</td><td>11.30</td><td>14.04</td><td>14.23</td><td>20.11</td><td>3.56</td><td>27.42</td><td>18.35</td><td>23.14</td><td>4.64</td><td>23.31</td></tr><tr><td>LwF-MC (Rebuffi et al., 2017)</td><td>0.32</td><td>17.48</td><td>9.05</td><td>10.07</td><td>13.49</td><td>14.35</td><td>5.63</td><td>19.31</td><td>16.02</td><td>10.62</td><td>15.18</td><td>15.98</td></tr><tr><td>ILT (Michieli and Zanuttigh, 2019)</td><td>0.92</td><td>21.70</td><td>11.23</td><td>16.10</td><td>23.03</td><td>23.16</td><td>10.77</td><td>20.04</td><td>19.01</td><td>35.51</td><td>7.13</td><td>35.77</td></tr><tr><td>MiB (Cermelli et al., 2020)</td><td>7.98</td><td>25.21</td><td>17.11</td><td>13.56</td><td>19.41</td><td>19.52</td><td>30.79</td><td>7.86</td><td>24.98</td><td>35.04</td><td>7.45</td><td>36.79</td></tr><tr><td>RCIL (Zhang et al., 2022)</td><td>50.08</td><td>4.86</td><td>40.64</td><td>26.22</td><td>48.53</td><td>42.66</td><td>0.38</td><td>22.95</td><td>22.64</td><td>52.32</td><td>8.41</td><td>50.59</td></tr><tr><td>CAT-SD (Xu et al., 2024)</td><td>44.70</td><td>6.29</td><td>37.32</td><td>49.50</td><td>11.24</td><td>41.45</td><td>4.95</td><td>5.55</td><td>9.67</td><td>55.31</td><td>9.00</td><td>52.58</td></tr><tr><td>MBS (Park et al., 2024)</td><td>30.85</td><td>7.64</td><td>24.40</td><td>46.93</td><td>40.54</td><td>45.15</td><td>35.92</td><td>18.16</td><td>34.12</td><td>46.43</td><td>49.41</td><td>46.59</td></tr><tr><td>NeST (Xie et al., 2024)</td><td>29.61</td><td>24.08</td><td>28.08</td><td>39.59</td><td>30.05</td><td>37.89</td><td>18.56</td><td>8.26</td><td>13.41</td><td>45.64</td><td>9.44</td><td>43.63</td></tr><tr><td>PLOP (Douillard et al., 2021)</td><td>12.09</td><td>19.69</td><td>18.49</td><td>44.14</td><td>23.38</td><td>41.47</td><td>4.56</td><td>18.37</td><td>14.90</td><td>39.11</td><td>8.32</td><td>41.06</td></tr><tr><td>PLOP+Ours</td><td>52.50</td><td>18.74</td><td>45.99</td><td>55.02</td><td>23.37</td><td>48.93</td><td>34.63</td><td>11.94</td><td>28.33</td><td>57.19</td><td>19.25</td><td>54.43</td></tr><tr><td>Improvement</td><td>+40.41</td><td>-0.95</td><td>+27.50</td><td>+10.88</td><td>-0.01</td><td>+7.46</td><td>+30.07</td><td>-6.43</td><td>+13.43</td><td>+18.08</td><td>+10.93</td><td>+13.37</td></tr><tr><td>EWF (Xiao et al., 2023)</td><td>54.82</td><td>5.12</td><td>43.98</td><td>50.90</td><td>26.83</td><td>43.22</td><td>35.96</td><td>0.00</td><td>23.77</td><td>57.15</td><td>0.05</td><td>53.93</td></tr><tr><td>EWF+Ours</td><td>62.37</td><td>8.70</td><td>50.13</td><td>60.83</td><td>20.18</td><td>45.62</td><td>63.96</td><td>18.80</td><td>46.68</td><td>59.01</td><td>8.88</td><td>55.33</td></tr><tr><td>Improvement</td><td>+7.55</td><td>+3.58</td><td>+6.15</td><td>+9.93</td><td>-6.65</td><td>+2.40</td><td>+28.00</td><td>+18.80</td><td>+22.91</td><td>+1.86</td><td>+8.83</td><td>+1.40</td></tr><tr><td>Offline</td><td>58.00</td><td>60.43</td><td>60.77</td><td>58.00</td><td>60.43</td><td>60.77</td><td>58.00</td><td>60.43</td><td>60.77</td><td>58.00</td><td>60.43</td><td>60.77</td></tr></table>

![](images/4fef14d8ee6935e1a4a744bbfe908c6ff46d2e2dd70dd5be33d5b2996bdfdfdb.jpg)  
Figure 6: Visual comparison with diferent state-of-the-art methods (Task 13-1 on our surgical dataset).

EWF, respectively, by efectively mitigating bias. This mitigation further improves the performance of the base class by 40.41% and 7.55%, respectively, while the slight decrease in the performance of the new class may be attributed to the reduction in predictable pixels. The diferences in improvements between the two baseline methods are primarily because PLOP (Douil lard et al., 2021) relies on feature distillation to limit representation capacity, while EWF (Xiao et al., 2023) employs a more robust weight fusion technique. The distinction in their principles leads to diferences in plasticity, which results in varying degrees of performance improvement. Notably, as continual learning progresses, the model updates increasingly favor new classes, resulting in greater bias. This trend further highlights the efectiveness of our method in addressing such challenges. FIGURE 1 provides evidence for the bias mitigation capabilities of our proposed framework. The increased uniformity in the distribution of the summation average and t-SNE embeddings demonstrates a significant reduction in class bias.

Table 2: Performance comparison across diferent datasets for the 13-1 continual learning task. Results are reported for methods without continual learning (WO CL), including the old model at time point t = 0 and naive fine-tuning (FT), and those with continual learning (W CL). The metrics include performance on base, new, and all classes, with ofline training serving as the upper bound.
<table><tr><td rowspan="2">Method</td><td colspan="3">CholecSeg8k</td><td colspan="3">EndoVis 2017</td><td colspan="3">EndoVis 2018</td><td colspan="3">SAR-RAPR50</td></tr><tr><td>Base</td><td>New</td><td>All</td><td>Base</td><td>New</td><td>All</td><td>Base</td><td>New</td><td>All</td><td>Base</td><td>New</td><td>All</td></tr><tr><td>Old Model (t=0)</td><td>62.53</td><td>0.00</td><td>44.66</td><td>83.82</td><td>0.00</td><td>55.88</td><td>73.33</td><td>0.00</td><td>61.10</td><td>61.69</td><td>0.00</td><td>52.97</td></tr><tr><td>FT</td><td>0.00</td><td>0.00</td><td>31.60</td><td>0.00</td><td>0.00</td><td>11.00</td><td>0.00</td><td>0.00</td><td>12.02</td><td>0.00</td><td>75.05</td><td>24.49</td></tr><tr><td>EWC (Kirkpatrick et al., 2017)</td><td>0.00</td><td>0.00</td><td>41.26</td><td>0.00</td><td>0.00</td><td>16.63</td><td>0.21</td><td>0.97</td><td>12.91</td><td>0.00</td><td>74.82</td><td>23.45</td></tr><tr><td>LwF (Li and Hoiem, 2017)</td><td>0.00</td><td>78.39</td><td>58.30</td><td>0.00</td><td>0.44</td><td>11.78</td><td>0.00</td><td>0.05</td><td>12.81</td><td>0.00</td><td>66.23</td><td>22.69</td></tr><tr><td>LwF-MC (Rebuffi et al., 2017)</td><td>0.00</td><td>18.40</td><td>37.86</td><td>0.00</td><td>0.00</td><td>11.03</td><td>0.00</td><td>0.00</td><td>12.14</td><td>0.00</td><td>19.56</td><td>16.55</td></tr><tr><td>ILT (Michieli and Zanuttigh, 2019)</td><td>0.00</td><td>59.66</td><td>51.94</td><td>0.00</td><td>0.00</td><td>11.26</td><td>0.00</td><td>0.00</td><td>12.24</td><td>0.00</td><td>36.75</td><td>18.70</td></tr><tr><td>MiB (Cermelli et al., 2020)</td><td>47.41</td><td>15.80</td><td>31.60</td><td>22.00</td><td>2.75</td><td>11.00</td><td>14.49</td><td>6.21</td><td>12.42</td><td>16.15</td><td>20.93</td><td>20.93</td></tr><tr><td>RCIL (Zhang et al., 2022)</td><td>66.63</td><td>31.19</td><td>65.31</td><td>44.07</td><td>21.47</td><td>41.91</td><td>38.50</td><td>22.34</td><td>43.85</td><td>51.03</td><td>0.00</td><td>49.75</td></tr><tr><td>CAT-SD (Xu et al., 2024)</td><td>42.22</td><td>53.63</td><td>64.44</td><td>33.74</td><td>3.26</td><td>30.36</td><td>22.55</td><td>2.13</td><td>30.35</td><td>37.06</td><td>33.73</td><td>43.82</td></tr><tr><td>MBS (Park et al., 2024)</td><td>27.20</td><td>34.53</td><td>41.52</td><td>35.54</td><td>10.83</td><td>37.12</td><td>18.93</td><td>10.16</td><td>23.12</td><td>27.96</td><td>18.32</td><td>23.21</td></tr><tr><td>NeST (Xie et al., 2024)</td><td>31.23</td><td>29.53</td><td>40.95</td><td>28.79</td><td>13.37</td><td>29.62</td><td>18.02</td><td>12.61</td><td>25.49</td><td>28.28</td><td>23.77</td><td>26.42</td></tr><tr><td>PLOP (Douillard et al., 2021)</td><td>0.00</td><td>68.52</td><td>55.65</td><td>14.59</td><td>6.18</td><td>21.80</td><td>13.16</td><td>6.90</td><td>24.15</td><td>0.00</td><td>43.28</td><td>18.61</td></tr><tr><td>PLOP+Ours</td><td>59.70</td><td>45.69</td><td>67.86</td><td>46.28</td><td>67.98</td><td>60.84</td><td>63.68</td><td>25.55</td><td>62.91</td><td>11.55</td><td>12.66</td><td>23.82</td></tr><tr><td>Improvement</td><td>+59.70</td><td>-22.83</td><td>+12.21</td><td>+31.69</td><td>+61.80</td><td>+39.04</td><td>+50.52</td><td>+18.65</td><td>+38.76</td><td>+11.55</td><td>-30.62</td><td>+5.21</td></tr><tr><td>EWF (Xiao et al., 2023)</td><td>63.46</td><td>67.46</td><td>76.46</td><td>57.65</td><td>31.25</td><td>52.73</td><td>51.67</td><td>41.99</td><td>56.82</td><td>49.53</td><td>0.01</td><td>48.99</td></tr><tr><td>EWF+Ours</td><td>67.12</td><td>70.04</td><td>78.57</td><td>50.31</td><td>48.05</td><td>55.36</td><td>58.90</td><td>49.07</td><td>63.00</td><td>51.68</td><td>0.02</td><td>50.84</td></tr><tr><td>Improvement</td><td>+3.66</td><td>+2.58</td><td>+2.11</td><td>-7.34</td><td>+16.80</td><td>+2.63</td><td>+7.23</td><td>+7.08</td><td>+6.18</td><td>+2.15</td><td>+0.01</td><td>+1.85</td></tr><tr><td>Offline</td><td>64.68</td><td>74.97</td><td>79.59</td><td>64.47</td><td>65.33</td><td>59.10</td><td>50.11</td><td>56.55</td><td>49.56</td><td>77.91</td><td>67.15</td><td>58.76</td></tr></table>

Table 3: Ablation study on the Class Desensitization Loss and dynamic weight fusion in our Rebalance Stage.
<table><tr><td>Setting</td><td> $\mathcal { L } _ { C D }$ </td><td>WF</td><td>13-1 13-5</td><td>9-3</td><td>17-1</td></tr><tr><td>Incremental Stage</td><td>=</td><td>-</td><td>43.98 43.22</td><td>23.77</td><td>53.93</td></tr><tr><td rowspan="3">Rebalance Stage</td><td>x</td><td>x</td><td>45.88 45.29</td><td>41.38</td><td>50.09</td></tr><tr><td>√</td><td>X</td><td>49.83 43.06</td><td>45.75</td><td>54.64</td></tr><tr><td></td><td></td><td>50.13 45.62</td><td>46.68</td><td>55.33</td></tr></table>

These findings underscore the robustness and efectiveness of our approach in mitigating bias, preserving base class knowledge, and improving overall performance across diverse tasks. Fig 6 shows the visual comparison with the ofline and partial competitors. We observe that our method consistently provides better predictions for Vessel Sealer, Other, Ultrasound Probe, Grasping Retractor Sealer, Large Needle Driver, Catheter, and Grasper, each represented by diferent colors. Additionally, our predictions are comparable to those of the bestperforming ofline method, as evidenced by fewer instances of over-predicting and under-predicting, as well as reduced confusion among diferent instrument classes.

## 4.3. Ablation Study

In this part, we demonstrate and analyze the efectiveness of the Class Desensitization Loss and its blending method. We also provide an analysis of our Dynamic Weight Fusion Strategy. We use 13-1 (5 steps) for the ablation experiments. TA-BLE 3 presents the ablation study on the impact of each proposed module, including the rebalance stage, Class Desensitization Loss, and Dynamic Weight Fusion. With the inclusion of the rebalance stage, performance improves by 1.90%, 2.07%, and 17.61% in the 13-1, 13-5, and 9-3 tasks, respectively. This demonstrates that a simple bias adjustment step can significantly enhance the performance of continual segmentation models. However, in the 17-1 task, a slight decrease in performance is observed, primarily due to the fewer continual learning steps, where incremental classes have not introduced significant class bias. This efect is mitigated by the Class Desensitization Loss, which results in a 4.55% improvement in the 17-1 task by clarifying the class boundary. We also perform a quality analysis to ensure that the bias has been mitigated, as demonstrated by the t-SNE feature in FIGURE 7, accompanied by qualitative cluster metrics in TABLE 4. The DB Index, CH Index, and WCSS have significantly improved, showing a more uniform distribution, which serves as evidence of the eficiency. Additionally, the Dynamic Weight Fusion strategy proves to be a versatile method for combining previous models, enhancing the generalization ability on real samples. Ultimately, our method strikes a balance between enabling the learning of new classes and maintaining the ability to distinguish base classes.

Table 4: Ablation study on the Class Desensitization Loss using cluster metrics on the t-SNE embedding.
<table><tr><td> $\mathcal { L } _ { C D }$ </td><td>DB Index ↓</td><td>CH Index ↑</td><td>WCSS↓</td></tr><tr><td>x</td><td>3.56</td><td>1.13</td><td>31445.56</td></tr><tr><td>√</td><td>2.13</td><td>3.02</td><td>11366.84</td></tr></table>

## 4.3.1. Class-aware Blending

In TABLE 5, we demonstrate the performance of diferent blending methods and sample numbers. We compare random blending with class-aware blending, verifying the efectiveness of our algorithm in alleviating model bias caused by class imbalance through the use of synthetic class samples. The results show that smaller sample sizes (e.g., 100 samples) can negatively impact model performance, particularly for new classes. However, as the number of samples increases, the performance improves significantly, with the class-aware method outperforming random blending. The class-aware blending method leads to a better balance between base and new classes, resulting in enhanced overall performance.

![](images/4cc004d4dd4e6300ef5fef67143e0d6a00c5befe2632f9f53473c75441e67eb6.jpg)  
Figure 7: Ablation study on the Class Desensitization Loss via the t-SNE visualization on the feature space.

![](images/e16b9a0de66813ff087562da96127bdd5775684d6289e3400fb76614054e880c.jpg)  
Figure 8: Visualization samples of our synthetic background inpainting.

TABLE 6 illustrates that directly cropping the background from the sample would reduce the efectiveness of the Class Desensitization Loss enhancement. Moreover, in TABLE 7, we explore the synthesis quality of four generative models (DDPM, Ho et al., 2020, StyleGAN-XL, Sauer et al., 2022, FLUX.1-Fill-dev, Labs, 2024, and LDM, Rombach et al., 2022) applied to our inpainting task and finally select LDM (Rombach et al., 2022) as our inpainting model. LDM exhibits strong performance compared to other methods, as other methods mostly rely on cropped patches instead of using the latent space, thus losing completeness. FIGURE 8 presents visualization examples of our inpainting solution, and FIGURE 9 pro vides the qualitative comparison of diferent synthetic methods. Additionally, the performance of the generation method signif icantly degrades due to its low quality. In contrast, our inpainting method demonstrates a substantial performance improvement. Besides, TABLE 8 shows the performance comparison under diferent loss weight configurations, specifically varying the values of α and $\beta .$ When $\alpha = 1 . 0$ and $\beta = 1 . 0$ , the model performs best on base classes and overall classes, leading to an improved performance of 3.96%.

Table 5: Comparisons of our ablation study on diferent blending methods and sample numbers.
<table><tr><td>Method</td><td>Number</td><td>Base classes</td><td>New classes</td><td>All classes</td></tr><tr><td>Random</td><td>100 300 500 1000</td><td>44.20 48.39 50.83 54.08</td><td>2.64 5.43 7.69</td><td>36.02 39.66 41.93</td></tr><tr><td>Class-aware</td><td>100 300 500 1000</td><td>46.65 49.38 58.77 62.02</td><td>7.33 7.02 4.50 6.21 8.49</td><td>44.05 38.78 40.09 46.97 49.83</td></tr></table>

Table 6: Comparisons of diferent background synthetic methods. Crop refers to directly cropping the background from samples.
<table><tr><td>Method</td><td>Base classes</td><td>New classes</td><td>All classes</td></tr><tr><td>Crop</td><td>56.10</td><td>6.47</td><td>45.21</td></tr><tr><td>Generate (DDPM, Ho et al., 2020)</td><td>44.02</td><td>5.79</td><td>36.49</td></tr><tr><td>Inpaint (LDM, Rombach et al., 2022)</td><td>62.37</td><td>8.70</td><td>50.13</td></tr></table>

In summary, our class-aware blending strategy combines foreground and background through a simple overlay. While this approach may introduce certain data inconsistencies and biases due to the lack of efective modeling between foreground and background, it has already proven efective in improving the performance of downstream segmentation tasks. Our work does not focus on developing generative models specifically for surgical applications. Instead, the generative model can serve as a simple and efective solution to address the downstream continual segmentation task in the surgical domain. Furthermore, our generation-based approach can also enhance privacy protection and improve data scalability (Wang et al., 2022, 2023a).

## 4.3.2. Fusion Strategy

In TABLE 9, we demonstrate the performance of diferent fusion strategies and their corresponding parameter selections. The Replace method involves directly selecting the top-k parameter values from the previous model to rebalance the current model. However, this approach results in a performance decline, as it can reintroduce biased parameters into the model. For example, when top- $\begin{array} { r } { \mathbf { \nabla } \cdot k \ = \ 1 \% } \end{array}$ , the performance drops to 45.88% for all classes, with a notable decrease in base class accuracy (55.98%). As k increases (e.g., top-k = 50%), the performance continues to degrade further (43.80% overall), illustrating the detrimental efect of bringing biased parameters back into the model. In contrast, our Weighted Fusion strategy balances the knowledge from the previous model while maintaining the model’s state unbiased. This method shows a clear improvement over the Replace strategy. Specifically, when using top-k = 10% and $\lambda = 0 . 0 1$ , the model achieves the highest overall performance at 50.13%, with a 0.3% improvement over the baseline (49.83%). The weighted fusion approach also allows for better retention of both base and new class performance, as evidenced by the slight improvements in both categories. Notably, limiting the number of fused parameters (e.g., top-k = 10%) helps prevent performance degradation, keeping the model less prone to bias. These results emphasize the effectiveness of the weighted fusion strategy in enhancing model performance, ofering a robust solution for continual learning scenarios where knowledge from previous models must be in corporated without introducing bias.

Table 7: Synthesis quality comparison of diferent generation methods.
<table><tr><td>Method</td><td>FID↓ IS↑</td><td>PSNR ↑</td></tr><tr><td>DDPM (Ho et al., 2020)</td><td>266.77 0.99</td><td>11.25</td></tr><tr><td>StyleGAN-XL (Sauer et al., 2022)</td><td>138.46 1.09</td><td>12.16</td></tr><tr><td>FLUX.1-Fill-dev (Labs, 2024)</td><td>158.60 1.86</td><td>11.06</td></tr><tr><td>LDM (Ours, Rombach et al., 2022)</td><td>48.26 1.30</td><td>12.40</td></tr></table>

Table 8: Comparisons of our ablation study on the weight of the Class Desensitization Loss (Task 13-1 on our surgical dataset).
<table><tr><td rowspan=1 colspan=1>α  β</td><td rowspan=1 colspan=1>Base classes</td><td rowspan=1 colspan=1>New classes</td><td rowspan=1 colspan=1>All classes</td></tr><tr><td rowspan=1 colspan=1>1.00.0</td><td rowspan=1 colspan=1>56.54</td><td rowspan=1 colspan=1>7.76</td><td rowspan=1 colspan=1>45.92</td></tr><tr><td rowspan=1 colspan=1>0.01.0</td><td rowspan=1 colspan=1>61.33</td><td rowspan=1 colspan=1>9.90</td><td rowspan=1 colspan=1>49.65</td></tr><tr><td rowspan=1 colspan=1>1.0 1.0</td><td rowspan=1 colspan=1>61.98</td><td rowspan=1 colspan=1>8.74</td><td rowspan=1 colspan=1>49.84</td></tr><tr><td rowspan=1 colspan=1>1.0 2.0</td><td rowspan=1 colspan=1>59.98</td><td rowspan=1 colspan=1>5.79</td><td rowspan=1 colspan=1>47.74</td></tr><tr><td rowspan=1 colspan=1>2.01.0</td><td rowspan=1 colspan=1>61.65</td><td rowspan=1 colspan=1>7.82</td><td rowspan=1 colspan=1>49.36</td></tr><tr><td rowspan=1 colspan=1>2.02.0</td><td rowspan=1 colspan=1>57.22</td><td rowspan=1 colspan=1>4.64</td><td rowspan=1 colspan=1>45.48</td></tr></table>

Table 9: Comparisons of our ablation study on the fusion parameter selection top-k and merged weight λ (Task 13-1 on our surgical dataset).
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>top-k  λ</td><td rowspan=1 colspan=1>Base classes</td><td rowspan=1 colspan=1>New classes</td><td rowspan=1 colspan=1>All classes</td></tr><tr><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1>-     -</td><td rowspan=1 colspan=1>62.02</td><td rowspan=1 colspan=1>8.49</td><td rowspan=1 colspan=1>49.83</td></tr><tr><td rowspan=4 colspan=1>Replace</td><td rowspan=4 colspan=1>1%    -10%30%50%   -</td><td rowspan=4 colspan=1>55.9856.3854.7654.69</td><td rowspan=1 colspan=1>9.31</td><td rowspan=1 colspan=1>45.88</td></tr><tr><td rowspan=1 colspan=1>5.87</td><td rowspan=3 colspan=1>45.2544.0043.80</td></tr><tr><td rowspan=1 colspan=1>5.32</td></tr><tr><td rowspan=1 colspan=1>5.05</td></tr><tr><td rowspan=5 colspan=1>Weighted</td><td rowspan=3 colspan=1>10% 0.110% 0.330% 0.05</td><td rowspan=1 colspan=1>62.41</td><td rowspan=1 colspan=1>8.13</td><td rowspan=1 colspan=1>49.97</td></tr><tr><td rowspan=1 colspan=1>62.34</td><td rowspan=1 colspan=1>8.11</td><td rowspan=1 colspan=1>49.92</td></tr><tr><td rowspan=1 colspan=1>61.34</td><td rowspan=1 colspan=1>7.94</td><td rowspan=1 colspan=1>49.19</td></tr><tr><td rowspan=1 colspan=1>50% 0.1</td><td rowspan=1 colspan=1>58.56</td><td rowspan=1 colspan=1>7.16</td><td rowspan=1 colspan=1>47.09</td></tr><tr><td rowspan=1 colspan=1>10% 0.01</td><td rowspan=1 colspan=1>62.37</td><td rowspan=1 colspan=1>8.70</td><td rowspan=1 colspan=1>50.13</td></tr></table>

## 5. Discussion and Conclusion

In this work, we propose a novel two-stage pipeline to address the critical challenge of class imbalance in continual segmentation for robotic surgical scenes, a challenge exacerbated by privacy constraints and the complexity of surgical workflows. Our SurgCSS framework includes the class-aware blending with inpainting to generate a class-balanced dataset by separating instrument foregrounds from surgical backgrounds, alongside the CDL based on contrastive learning to reduce con fusion between similar classes. Additionally, a DWF strategy combines incremental and rebalance models to enhance class representation and improve overall segmentation accuracy. Experimental results demonstrate that (i) there is significant performance degradation due to class imbalance when directly applying conventional segmentation techniques; (ii) existing continual learning methods, while alleviating catastrophic forgetting, fail to address class imbalance efectively in surgical contexts; and (iii) our two-stage pipeline successfully balances performance across old and new classes, outperforming state-ofthe-art continual learning methods with average 11.82% improvement in robotic surgical scenarios.

![](images/e4ee32f967200e587994f43d6f607fe6119170696d66e714bacc92f226192ebd.jpg)  
Figure 9: Qualitative comparison of diferent generation methods.

This is a fundamental work focusing on resolving the combined issues of class imbalance, catastrophic forgetting, and privacy concerns in the context of robotic surgical segmentation. The framework can be extended to other medical imaging tasks (e.g., organ segmentation or lesion detection), as these tasks often share similar challenges such as imbalanced class distributions and insuficient data. To address these challenges, our proposed inpainting synthetic data generation method and Class Desensitization Loss ofer flexible, generalized solutions that reduce class bias and improve model robustness across various applications in the medical domain. However, our proposed framework still has some limitations: (i) Although the final deployed segmentation model remains identical to the previous model without introducing additional parameters, it still requires ofline data generation and extra training for the generative model, which significantly increases the training cost. (ii) Any defects or artifacts in the synthetic data can negatively affect the model’s performance, particularly when distinguishing between visually similar instrument classes. (iii) The generative model, trained on the background characteristics of the initial dataset, may require retraining or substantial updates to efectively generate data for significantly diferent surgical do mains, leading to extra computational costs. Therefore, future work should rigorously validate the feasibility, robustness, and applicability of the proposed continual learning approach and generative framework in real-world clinical environments, including addressing lightweight deployment and real-time inference challenges in actual hospital settings. Additionally, our continual learning approach can be efectively applied to adapt visual foundation models, such as SAM (Kirillov et al., 2023), to various downstream medical applications, mitigating catastrophic forgetting of the originally learned computer vision patterns.

## Acknowledgment

This work was supported by Hong Kong Research Grants Council GRF 14204524, GRF 14203323, GRF 14216022, GRF 14211420, CRF C4026-21GF; Guangdong Basic & Applied Research Fund Regional Joint Fund Project 2021B1515120035 (B.02.21.00101); Shenzhen-Hong Kong-Macau Technology Research Programme (Type C) STIC Grant SGDX20210823103535014 (202108233000303).

## Declaration of Using Generative AI and AI-assisted Technologies

The authors have used ChatGPT to improve the readability and language of the work. The process of using ChatGPT was

done with human oversight and control. The authors have carefully reviewed and edited the results. The authors have not used ChatGPT to provide citations.

## References

Alabi, O., Vercauteren, T., Shi, M., 2025. Multitask learning in minimally invasive surgical vision: A review. Medical Image Analysis , 103480.

Allan, M., Kondo, S., Bodenstedt, S., Leger, S., Kadkhodamohammadi, R., Luengo, I., Fuentes, F., Flouty, E., Mohammed, A., Pedersen, M., et al., 2020. 2018 robotic scene segmentation challenge. arXiv preprint arXiv:2001.11190 .

Allan, M., Shvets, A., Kurmann, T., Zhang, Z., Duggal, R., Su, Y.H., Rieke, N., Laina, I., Kalavakonda, N., Bodenstedt, S., et al., 2019. 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426 .

Anand, D., Das, B., Dangeti, V., Jerald, A., Mullick, R., Patil, U., Sharma, P., Sudhakar, P., 2024. Label sharing incremental learning framework for independent multi-label segmentation tasks. arXiv preprint arXiv:2411.11105 .

Ayromlou, S., Tsang, T., Abolmaesumi, P., Li, X., 2024. Ccsi: Continual class-specific impression for data-free class incremental learning. Medical Image Analysis , 103239.

Bai, L., Islam, M., Ren, H., 2023. Revisiting distillation for continual learning on visual question localized-answering in robotic surgery, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 68–78.

Cermelli, F., Mancini, M., Bulo, S.R., Ricci, E., Caputo, B., 2020. Modeling the background for incremental learning in semantic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9233–9242.

Cha, S., Yoo, Y., Moon, T., et al., 2021. Ssul: Semantic segmentation with unknown label for exemplar-based classincremental learning. Advances in neural information processing systems 34, 10919–10930.

Chaudhry, A., Dokania, P.K., Ajanthan, T., Torr, P.H., 2018. Riemannian walk for incremental learning: Understanding forgetting and intransigence, in: Proceedings of the European conference on computer vision (ECCV), pp. 532–547.

Chen, L.C., Zhu, Y., Papandreou, G., Schrof, F., Adam, H., 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation, in: Proceedings of the European conference on computer vision (ECCV), pp. 801–818.

Chen, Z., Zhang, Z., Guo, W., Luo, X., Bai, L., Wu, J., Ren, H., Liu, H., 2024. Asi-seg: Audio-driven surgical instrument segmentation with surgeon intention understanding, in: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE. pp. 13773–13779.

Cheng, Z., Guo, J., Zhang, J., Qi, L., Zhou, L., Shi, Y., Gao, Y., 2025. Mamba-sea: A mamba-based framework with globalto-local sequence augmentation for generalizable medical image segmentation. arXiv preprint arXiv:2504.17515 .

Cho, J., Schmidgall, S., Zakka, C., Mathur, M., Kaur, D., Shad, R., Hiesinger, W., 2024. Surgen: Text-guided diffusion model for surgical video generation. arXiv preprint arXiv:2408.14028 .

Colleoni, E., Psychogyios, D., Van Amsterdam, B., Vasconcelos, F., Stoyanov, D., 2022. Ssis-seg: Simulation-supervised image synthesis for surgical instrument segmentation. IEEE Transactions on Medical Imaging 41, 3074–3086.

Colleoni, E., Stoyanov, D., 2021. Robotic instrument segmentation with image-to-image translation. IEEE Robotics and Automation Letters 6, 935–942.

Douillard, A., Chen, Y., Dapogny, A., Cord, M., 2021. Plop: Learning without forgetting for continual semantic segmen tation, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4040–4050.

Douillard, A., Cord, M., Ollion, C., Robert, T., Valle, E., 2020. Podnet: Pooled outputs distillation for small-tasks incremental learning, in: Computer vision–ECCV 2020: 16th European conference, Glasgow, UK, August 23–28, 2020, proceedings, part XX 16, Springer. pp. 86–102.

Elskhawy, A., Lisowska, A., Keicher, M., Henry, J., Thomson, P., Navab, N., 2020. Continual class incremental learning for ct thoracic segmentation, in: MICCAI Workshop on Domain Adaptation and Representation Transfer, Springer. pp. 106– 116.

Fang, K., Zhang, A., Gao, G., Jiao, J., Liu, C.H., Wei, Y., 2025. Combo: Conflict mitigation via branched optimization for class incremental segmentation. arXiv preprint arXiv:2504.04156 .

Garcia-Peraza-Herrera, L.C., Fidon, L., D’Ettorre, C., Stoyanov, D., Vercauteren, T., Ourselin, S., 2021. Image compositing for segmentation of surgical tools without manual annotations. IEEE transactions on medical imaging 40, 1450–1460.

Ge, J., Zhang, B., Liu, A., Phan, M.H., Chen, Q., Shu, Y., Zhao, Y., 2024. Cit: Rethinking class-incremental semantic segmentation with a class independent transformation. arXiv preprint arXiv:2411.02715 .

Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.Y., Cubuk, E.D., Le, Q.V., Zoph, B., 2021. Simple copy-paste is a strong data augmentation method for instance segmentation, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2918–2928.

Guo, D., Ji, Z., Su, Y., Zheng, D., Guo, H., Wang, P., Yan, K., Wang, Y., Yu, Q., Li, Z., et al., 2025. A continual learningdriven model for accurate and generalizable segmentation

of clinically comprehensive and fine-grained whole-body anatomies in ct. arXiv preprint arXiv:2503.12698 .

Hamghalam, M., Lei, B., Wang, T., 2020. High tissue contrast mri synthesis using multi-stage attention-gan for segmentation, in: Proceedings of the AAAI conference on artificial intelligence, pp. 4067–4074.

He, C., Wang, R., Chen, X., 2021. A tale of two cils: The connections between class incremental learning and class imbalanced learning, and beyond, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3559–3569.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770– 778.

Hensman, P., Masko, D., 2015. The impact of imbalanced training data for convolutional neural networks. Degree Project in Computer Science, KTH Royal Institute of Technology .

Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851.

Hong, W.Y., Kao, C.L., Kuo, Y.H., Wang, J.R., Chang, W.L., Shih, C.S., 2020. Cholecseg8k: a semantic segmentation dataset for laparoscopic cholecystectomy based on cholec80. arXiv preprint arXiv:2012.12453 .

Ji, Z., Guo, D., Wang, P., Yan, K., Lu, L., Xu, M., Wang, Q., Ge, J., Gao, M., Ye, X., et al., 2023. Continual segment: Towards a single, unified and non-forgetting continual segmentation model of 143 whole-body organs in ct scans, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21140–21151.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al., 2023. Segment anything, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4026.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al., 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114, 3521–3526.

Labs, B.F., 2024. Flux. https://github.com/ black-forest-labs/flux.

Lee, H., Park, M., Kim, J., 2016. Plankton classification on imbalanced large scale database via convolutional neural networks with transfer learning, in: 2016 IEEE international conference on image processing (ICIP), IEEE. pp. 3713– 3717.

Lee, K., Choi, M.K., Jung, H., 2019. Davincigan: Unpaired sur gical instrument translation for data augmentation, in: International Conference on Medical Imaging with Deep Learning, PMLR. pp. 326–336.

Li, C.L., Sohn, K., Yoon, J., Pfister, T., 2021. Cutpaste: Selfsupervised learning for anomaly detection and localization, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9664–9674.

Li, W., Zhang, Y., Zhou, H., Yang, W., Xie, Z., He, Y., 2025. Clms: Bridging domain gaps in medical imaging segmentation with source-free continual learning for robust knowl edge transfer and adaptation. Medical Image Analysis 100, 103404.

Li, Z., Hoiem, D., 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence 40, 2935–2947.

Lin, T., 2017. Focal loss for dense object detection. arXiv preprint arXiv:1708.02002 .

Liu, J., Guo, X., Yuan, Y., 2021. Graph-based surgical in strument adaptive segmentation via domain-common knowledge. IEEE Transactions on Medical Imaging 41, 715–726.

Liu, X., Hu, Y.S., Cao, X.S., Bagdanov, A.D., Li, K., Cheng, M.M., 2022. Long-tailed class incremental learning, in: European Conference on Computer Vision, Springer. pp. 495– 512.

Lou, A., Tawfik, K., Yao, X., Liu, Z., Noble, J., 2023. Minmax similarity: A contrastive semi-supervised deep learning network for surgical tools segmentation. IEEE Transactions on Medical Imaging 42, 2832–2841.

Maier-Hein, L., Eisenmann, M., Sarikaya, D., März, K., Collins, T., Malpani, A., Fallert, J., Feussner, H., Giannarou, S., Mascagni, P., et al., 2022. Surgical data science–from concepts toward clinical translation. Medical image analysis 76, 102306.

Michieli, U., Zanuttigh, P., 2019. Incremental learning techniques for semantic segmentation, in: Proceedings of the IEEE/CVF international conference on computer vision workshops, pp. 0–0.

Michieli, U., Zanuttigh, P., 2021. Continual semantic segmentation via repulsion-attraction of sparse and disentangled latent representations, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1114–1124.

Nwoye, C.I., Yu, T., Sharma, S., Murali, A., Alapatt, D., Vardazaryan, A., Yuan, K., Hajek, J., Reiter, W., Yamlahi, A., et al., 2023. Cholectriplet2022: Show me a tool and tell me the triplet—an endoscopic vision challenge for surgical action triplet detection. Medical Image Analysis 89, 102888.

Park, G., Moon, W., Lee, S., Kim, T.Y., Heo, J.P., 2024. Mitigating background shift in class-incremental semantic segmentation, in: European Conference on Computer Vision, Springer. pp. 71–88.

Pfeifer, M., Funke, I., Robu, M.R., Bodenstedt, S., Strenger, L., Engelhardt, S., Roß, T., Clarkson, M.J., Gurusamy, K., Davidson, B.R., et al., 2019. Generating large labeled data sets for laparoscopic image processing tasks using unpaired image-to-image translation, in: Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part V 22, Springer. pp. 119–127.

Psychogyios, D., Colleoni, E., Van Amsterdam, B., Li, C.Y., Huang, S.Y., Li, Y., Jia, F., Zou, B., Wang, G., Liu, Y., et al., 2023. Sar-rarp50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv preprint arXiv:2401.00496 .

Ranem, A., González, C., dos Santos, D.P., Bucher, A.M., Othman, A.E., Mukhopadhyay, A., 2024. Continual atlasbased segmentation of prostate mri, in: Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 7563–7572.

Razzak, M.I., Naz, S., Zaib, A., 2018. Deep learning for medical image processing: Overview, challenges and the future. Classification in BioApps: Automation of decision making , 323–350.

Rebufi, S.A., Kolesnikov, A., Sperl, G., Lampert, C.H., 2017. icarl: Incremental classifier and representation learning, in: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001–2010.

Rivoir, D., Pfeifer, M., Docea, R., Kolbinger, F., Riediger, C., Weitz, J., Speidel, S., 2021. Long-term temporally consistent unpaired video translation from simulated surgical 3d data, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 3343–3353.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695.

Sadegheih, Y., Kumari, P., Merhof, D., 2025. Modalityindependent brain lesion segmentation with privacy-aware continual learning. arXiv preprint arXiv:2503.20326 .

Sauer, A., Schwarz, K., Geiger, A., 2022. Stylegan-xl: Scaling stylegan to large diverse datasets, in: ACM SIGGRAPH 2022 conference proceedings, pp. 1–10.

Shin, H., Lee, J.K., Kim, J., Kim, J., 2017. Continual learning with deep generative replay. Advances in neural information processing systems 30.

Shin, H.C., Tenenholtz, N.A., Rogers, J.K., Schwarz, C.G., Senjem, M.L., Gunter, J.L., Andriole, K.P., Michalski, M., 2018. Medical image synthesis for data augmentation and anonymization using generative adversarial networks, in: Simulation and Synthesis in Medical Imaging: Third International Workshop, SASHIMI 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Proceedings 3, Springer. pp. 1–11.

Venkatesh, D.K., Rivoir, D., Pfeifer, M., Speidel, S., 2024. Surgical-cd: Generating surgical images via unpaired image translation with latent consistency difusion models. arXiv preprint arXiv:2408.09822 .

Wang, A., Islam, M., Xu, M., Ren, H., 2022. Rethinking surgical instrument segmentation: A background image can be all you need, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 355–364.

Wang, A., Islam, M., Xu, M., Ren, H., 2023a. Generalizing surgical instruments segmentation to unseen domains with oneto-many synthesis, in: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE. pp. 4608–4614.

Wang, G., Bai, L., Wu, Y., Chen, T., Ren, H., 2023b. Rethinking exemplars for continual semantic segmentation in endoscopy scenes: Entropy-based mini-batch pseudo-replay. Computers in Biology and Medicine 165, 107412.

Wang, G., Ren, T.A., Lai, J., Bai, L., Ren, H., 2023c. Domain adaptive sim-to-real segmentation of oropharyngeal organs. Medical & Biological Engineering & Computing 61, 2745– 2755.

Wang, L., Zhang, X., Su, H., Zhu, J., 2024. A comprehensive survey of continual learning: theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence .

Wang, S., Liu, W., Wu, J., Cao, L., Meng, Q., Kennedy, P.J., 2016. Training deep neural networks on imbalanced data sets, in: 2016 international joint conference on neural networks (IJCNN), IEEE. pp. 4368–4374.

Wu, C., Herranz, L., Liu, X., Van De Weijer, J., Raducanu, B., et al., 2018. Memory replay gans: Learning to generate new categories without forgetting. Advances in neural information processing systems 31.

Wu, H., Wang, Z., Zhao, Z., Chen, C., Qin, J., 2023. Continual nuclei segmentation via prototype-wise relation distillation and contrastive learning. IEEE Transactions on Medical Imaging .

Wu, Y., Chen, Y., Wang, L., Ye, Y., Liu, Z., Guo, Y., Fu, Y., 2019. Large scale incremental learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 374–382.

Xiao, J.W., Zhang, C.B., Feng, J., Liu, X., van de Weijer, J., Cheng, M.M., 2023. Endpoints weight fusion for class incremental semantic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7204–7213.

Xie, Z., Lu, H., Xiao, J.w., Wang, E., Zhang, L., Liu, X., 2024. Early preparation pays of: New classifier pre-tuning for class incremental semantic segmentation, in: European Conference on Computer Vision, Springer. pp. 183–201.

Xu, M., Islam, M., Bai, L., Ren, H., 2024. Privacypreserving synthetic continual semantic segmentation for robotic surgery. IEEE Transactions on Medical Imaging .

Yang, J., Wu, Y., Cen, J., Huang, W., Wang, H., Zhang, J., 2024. Continual learning for segment anything model adaptation. arXiv preprint arXiv:2412.06418 .

Yin, H., Feng, T., Lyu, F., Shang, F., Liu, H., Feng, W., Wan, L., 2025. Beyond background shift: Rethinking instance replay in continual semantic segmentation. arXiv preprint arXiv:2503.22136 .

Yu, J., Bai, L., Wang, G., Wang, A., Yang, X., Gao, H., Ren, H., 2024a. Adapting sam for surgical instrument tracking and segmentation in endoscopic submucosal dissection videos. arXiv preprint arXiv:2404.10640 .

Yu, J., Wang, A., Dong, W., Xu, M., Islam, M., Wang, J., Bai, L., Ren, H., 2024b. Sam 2 in robotic surgery: An empirical evaluation for robustness and generalization in surgical video segmentation. arXiv preprint arXiv:2408.04593 .

Yu, X., Fang, Y., Zhao, Y., Wei, Y., 2025. Ipseg: Image posterior mitigates semantic drift in class-incremental segmentation. arXiv preprint arXiv:2502.04870 .

Zafino, P., Marzullo, A., Moccia, S., Calimeri, F., De Momi, E., Bertucci, B., Arcuri, P.P., Spadea, M.F., 2021. An opensource covid-19 ct dataset with automatic lung tissue classification for radiomics. Bioengineering 8, 26.

Zhang, C.B., Xiao, J.W., Liu, X., Chen, Y.C., Cheng, M.M., 2022. Representation compensation networks for continual semantic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7053–7064.

Zhang, Y., Li, X., Chen, H., Yuille, A.L., Liu, Y., Zhou, Z., 2023. Continual learning for abdominal multi-organ and tumor segmentation, in: International conference on medical image computing and computer-assisted intervention, Springer. pp. 35–45.

Zhao, B., Xiao, X., Gan, G., Zhang, B., Xia, S.T., 2020. Maintaining discrimination and fairness in class incremental learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13208–13217.

Zhao, Y., Bai, L., Zhang, Z., Wu, Y., Islam, M., Ren, H., 2024. Transferring knowledge from high-quality to low-quality mri for adult glioma diagnosis. arXiv preprint arXiv:2410.18698

Zhou, Y., Towning, R., Awad, Z., Giannarou, S., 2024. Image synthesis with class-aware semantic difusion models for surgical scene segmentation. arXiv preprint arXiv:2410.23962

Zhu, V., Ji, Z., Guo, D., Wang, P., Xia, Y., Lu, L., Ye, X., Zhu, W., Jin, D., 2024. Low-rank continual pyramid vision transformer: Incrementally segment whole-body organs in ct with light-weighted adaptation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 371–381.

Zia, A., Bhattacharyya, K., Liu, X., Berniker, M., Wang, Z., Nespolo, R., Kondo, S., Kasai, S., Hirasawa, K., Liu, B., et al., 2023. Surgical tool classification and localization: results and methods from the miccai 2022 surgtoolloc challenge. arXiv preprint arXiv:2305.07152 .