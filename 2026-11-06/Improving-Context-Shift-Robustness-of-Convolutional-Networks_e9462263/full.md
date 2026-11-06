# Improving Context-Shift Robustness of Convolutional Networks via Context-Regularized Cross-Entropy

Jinen Setpal Elmore Family School of Electrical and Computer Engineering Purdue University jsetpal@purdue.edu

Chaoyue Liu Elmore Family School of Electrical and Computer Engineering Purdue University cyliu@purdue.edu

## Abstract

Image classification models often encounter objects in diverse and unpredictable contexts at test time, which may differ substantially from those seen during training. This phenomenon, termed context shift, exposes a key fragility in neural networks: when models rely on non-causal contextual cues for prediction, shifts in context can lead to significant performance degradation. In this paper, we propose a novel training framework to improve the robustness of convolutional neural networks under context shift. Our approach leverages spatial feature attribution to guide models toward making predictions that rely less on contextual regions and more on object-relevant features. As a first step, recognizing a theoretical limitation of existing feature attribution methods we introduce a new variant, ContrastiveCAMs, which produces more faithful attribution maps of model predictions. Building on ContrastiveCAMs, we further propose Context-Regularized Cross-Entropy (CR-CE), a modification of the standard cross-entropy loss that regularizes the model’s attention by suppressing the influence of contextual regions, thereby improving robustness to context shift. We evaluate the effectiveness of our approach on several medium-to-large scale datasets (Waterbirds, Spawrious, ImageNet/ImageNet-BG, Hard-ImageNet), and report consistent improvements in context-shift robustness.

## 1 Introduction

The vast applications of convolutional neural networks in safety-critical domains such as medical imaging [38, 33], forensic investigation [26] and self-driving [21] make trustworthy and robust predictions essential for large-scale adoption. It has been demonstrated that existing neural networks, despite achieving remarkable performance on large-scale datasets, heavily rely on supporting context (i.e., background area within the images) to determine their predictions [41, 23, 30]. This sensitivity to surrounding context can degrade performance when the same object appears in a different environment [46, 15, 3, 4], whether due to novel object-context combinations in the test data or adversaries manipulating target-agnostic regions to mislead the model.

We highlight this concern in Figure 1 using the Waterbirds-100 dataset [37] which only has the object-context combinations of waterbird-waterfront and landbird-land. When a waterbird is shown in the context of land, the model predicts the class landbird with high confidence, due to the influence of the context dominating over the influence of the object. The converse also occurs when a landbird appears in a waterfront context. This concerning lack of robustness has been observed across safetycritical applications spanning the diagnosis of skin cancer [27], pneumothorax [36], as well as in the generalization capabilities of robot foundation models [48].

![](images/892f023b85b37270f60fde4022065309f2468f25c018f951f427cdff1e4d8206.jpg)  
Figure 1: Context Contributions Can Mislead Predictions. Spatial feature attribution maps reveal that cross-entropy trained models rely heavily on context, causing misclassifications in novel environments. Our approach encourages focus on the object thereby improving context-shift robustness.

In this paper, we aim to improve the context robustness of convolutional neural networks by leveraging spatial feature attributions. We begin by noting a theoretical limitation of existing feature attribution methods: they permit an arbitrary constant shift applied uniformly across all output logits, which undermines their ability to provide faithful explanations. As a result, the attributed contributions from context and object regions may not accurately reflect their true influence, as they can be distorted by this constant shift. We address this limitation by proposing a new variant ContrastiveCAMs, based on a prior method, HiResCAMs [10]. We further theoretically confirm that ContrastiveCAMs provide faithful explanations to the neural networks final predictions.

To address sensitivity to context shifts, we propose a new loss function, Context-Regularized Cross-Entropy (CR-CE), which leverages the proposed feature attribution method, ContrastiveCAMs, to encourage the model to focus on the primary object by suppressing contributions from the surrounding context. This approach requires object-context segmentation during training (crucially, with no segmentation required during inference), which can be readily obtained using existing pretrained segmentation models such as SAM3 [5] and Masked AutoEncoders (MAE) [13, 18]. Notably, the segmentation need not be perfect. In our experiments on the ImageNet dataset, we use MAEs to generate object-context segmentation. Experimental results demonstrate that our method effectively reduces the context contribution to model predictions and improves convolutional networks robustness to context shifts consistently across a broad class of datasets. Our key contributions are summarized as follows:

• We identify a theoretical limitation of existing spatial feature attribution methods that hinders explanation faithfulness. We address this limitation by proposing a new method ContrastiveCAMs.

• We propose a new loss function, Context-Regularized Cross Entropy (CR-CE), that encourages model predictions to focus on the object region rather than the context.

• Experimental evaluations demonstrate that the proposed training approach effectively attracts the focus of feature attribution towards the object region, thus enhancing context-shift robustness.

Notation. We denote vectors using bold lowercase letters (e.g., v), matrices using uppercase letters (e.g., M), and tensors using bold uppercase letters (e.g., T), with partial indexing implying selection of the subtensor across the remaining subsequent dimensions (e.g., $\mathbf { T } _ { i } \in \mathbb { R } ^ { b \times c } \operatorname { f o r } \mathbf { T } \in \mathbf { \mathbb { R } } ^ { a \times b \times c } )$ . We use the operator ⊙ to represent elementwise multiplication, and define $[ K ] : = \{ 1 , 2 , \dots , K \}$

## 1.1 Related Work

Feature Attribution Methods in Convolutional Networks. Class Activation Mappings (CAMs) [49] is a prominent family of methods that identify attributions of each (spatial) feature, including GradCAM [39], GradCAM++ [6], Score-CAM [45], and HiResCAM [10]. These methods typically compute feature attributions with respect to the model’s output logits, prior to the final softmax activation. However, due to the contrastive nature of the softmax function [28, 31], this can lead to inaccurate attributions for the post-softmax predictions. DiffGradCAM [31] shares a similar approach to ContrastiveCAMs. But it is based on GradCAM and its connection to the model’s prediction is not theoretically clear.

Context-Shift Robustness. One approach to improving robustness is through explanation-guided methods, which leverage feature attribution or saliency maps [40] to direct the model’s attention toward relevant objects and away from contextual distractions in the images [14]. This is typically achieved by augmenting the standard cross-entropy loss with additional regularization terms that penalize discrepancies between the model’s feature attribution map and a target explanation [35, 29, 34, 11]. The regularization term of Right for the Right Reasons (RRR) [35] penalizes the gradients of log-prediction w.r.t. the irrelevant input features. GALS [29] regularizes the Grad-CAM of the model’s predictions toward a target feature attribution map generated using training labels and auxiliary vision-language models. The regularization term of EBL [34] penalizes the positive attributions in contextual regions of the feature attribution map. ABNs [11] introduce an auxiliary attention module to find key regions of images to assist convolutional networks in making predictions. In contrast, our method directly modifies the cross-entropy loss rather than introducing an additional regularization term, so that robustness to context shifts is built in intrinsically.

An orthogonal approach is worst-group generalization [2, 7, 37]. Rather than optimizing for overall accuracy, this line of work partitions the dataset into groups defined by object-context combinations and seeks to balance performance across groups by improving the worst-performing group.

## 2 Motivation

In image classification tasks, deep neural networks make their predictions based on both the object and the context, i.e., the backgrounds in images. It has been observed that irrelevant, or even inversely correlated, context can make a substantial contribution to the model predictions, lowering model’s confidence on correct predictions, or even overturning the prediction into incorrectness with high confidence [47, 15, 30]. This reliance renders the model less robust to context shifts, such as when a trained model is deployed in a different environment, and more vulnerable to adversarial manipulation; for instance, when an attacker deliberately replaces the context with one commonly associated with other classes. In contrast, humans are able to distinguish between key features and spurious context; where deemed necessary, they can focus on the object region, neglecting the context, so as to accurately classify the relevant object in question.

As an illustration of this issue, consider the Waterbirds dataset [37], which consists of two classes of birds: waterbirds and landbirds. During training, most birds appear in their natural habitat: waterbirds are present by waterfronts, whereas landbirds are present surrounded by land. However, during testing, a context shift may occur, where a landbird (or a waterbird) appears in a waterfront context (or a land context, respectively). Neural networks often make incorrect predictions on images with context shift. For example, with a fine-tuned ResNet-50, the context region on average accounts for 46% of the model’s prediction score compared to the influence from the object. Moreover, the model’s accuracy drops from 98% for normal images down to 41% on images with context shift, which is even worse than the random guess accuracy of 50%. Using HiResCAMs [10], a member of the CAM (short for Class Activation Mapping) family that produces a heatmap highlighting the contribution from spatial locations of the image to the network output [49], we can visualize the context’s contributions to the model prediction, see Figure 1 for example. We see that when the image context is correlated with a different class, the misleading context makes large contributions to the prediction resulting in misclassifications with high confidence. This shows the pervasiveness of the context robustness concern, supported by broader statistics across other datasets in Table 1.

Our observation is that the neural networks, together with its optimization problem, do not possess an intrinsic mechanism to distinguish objects from context. To see this, consider a neural network f (before the softmax), such as a ResNet-50, composed of a convolutional backbone $g$ and a bias-free linear classifier h: $f = h \circ g .$ . Given an input image X, denote the feature embedding $g ( \mathbf { X } ) = \mathbf { A } \in \mathbb { R } ^ { d _ { 0 } \times d _ { 1 } \times d _ { 2 } }$ and the final logit $f ( \mathbf { X } ) \in \mathbb { R } ^ { K }$ , where K is the number of classes, $d _ { 0 }$ is the number of features (a.k.a. channels) and $d _ { 1 }$ and $d _ { 2 }$ are the spatial dimensions of the final convolutional layer. HiResCAMs, defined as follows,

$$
\mathcal { A } _ { k } = \sum _ { j = 1 } ^ { d _ { 0 } } ( \nabla _ { \mathbf { A } _ { j } } f _ { k } ) \odot \mathbf { A } _ { j } , \quad \forall k \in [ K ] ,\tag{1}
$$

have the following relation with the network output logits [10]:

$$
f _ { k } ( { \mathbf { X } } ) = \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( { A _ { k } } ( { \mathbf { X } } ) \right) _ { i j } , \quad \forall k \in [ K ] .\tag{2}
$$

Remark 2.1. Other CAM methods, such as GradCAM [39] and GradCAM++ [6], do not possess the above relation with output logits.

Let $H \in \mathbb { B } ^ { d _ { 1 } \times d _ { 2 } }$ be the object indicator function, i.e., $H _ { i j } = 1$ if the $( i , j )$ -patch corresponds to the object and $H _ { i j } = 0$ if the $( i , j )$ )-patch corresponds to the context. Then, the network’s final output logits can equivalently be written as:

$$
f _ { k } ( \mathbf { X } ) = \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } ( H \odot \mathcal { A } _ { k } ( \mathbf { X } ) ) _ { i j } + \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } ( ( 1 - H ) \odot \mathcal { A } _ { k } ( \mathbf { X } ) ) _ { i j } , \quad \forall k \in [ K ] ,\tag{3}
$$

where the first and second terms correspond to contributions from the object and context regions respectively. Here, we can see that the context contributes to the prediction in the same way as the object regions, and as seen in the demonstration in Figure 1, the network does not learn to suppress the context contributions.

In this paper, we seek to alleviate this issue of sensitivity to context shift with a focus on convolutional neural networks. Specifically, we will utilize a variant of HiResCAMs to design a new loss function that is capable of suppressing context contributions, encouraging the models to make predictions by primarily focusing on the objects thus improving their robustness to context shift.

## 3 From HiResCAMs to ContrastiveCAMs

In this section, we first present a fundamental issue of HiResCAMs in the multiclass setting, which degrades the faithfulness of feature attribution maps. We then propose a variant, ContrastiveCAMs, which produces accurate, class-specific attribution maps even in the multiclass setting.

HiResCAMs Admit Spurious Shifts. A key drawback of HiResCAMs emerge as a consequence of applying softmax activation on the network’s output logits $f$ to obtain the final predictions $\bar { \tilde { f } } = \sigma ( f )$ which are typically interpreted as class probabilities.

Proposition 3.1 (Redundancy in the softmax function).

$$
\sigma ( { \bf x } ) = \sigma ( { \bf x } + a { \bf 1 } _ { K } ) , \qquad \forall { \bf x } \in \mathbb { R } ^ { K } , \ a \in \mathbb { R } .\tag{4}
$$

Proof. The proof is deferred to Appendix A.

The above proposition shows that, given a prediction ${ \tilde { f } } ,$ the logit vector $f$ is not unique: if each logit is shifted by the same value, the final probability predictions $\tilde { f }$ remain unchanged. Therefore, the value of each individual logit $f _ { k }$ alone does not determine the final prediction. This redundancy has also been observed recently by Oh and Noh [28], Piland et al. [31], and discussed in the context of GradCAMs. Here, we show that this redundancy is inherited by HiResCAMs, and further pronounced in terms of degrees of freedom, via Eq. (2).

Theorem 3.2. Let $\mathcal { A } \in \mathbb { R } ^ { K \times d _ { 1 } \times d _ { 2 } }$ be a HiResCAM. Given an arbitrary matrix $M \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ , define a new HiResCAM as:

$$
\begin{array} { r } { \overline { { \mathcal { A } } } _ { k } : = \mathcal { A } _ { k } + M , \qquad \forall k \in [ K ] . } \end{array}\tag{5}
$$

Then, the two HiResCAMs correspond to the same network prediction $\tilde { f } ( { \mathbf { X } } )$

$$
\tilde { f } ( \mathbf { X } ) = \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \mathcal { A } _ { : i j } \right) = \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \overline { { \mathcal { A } } } _ { : i j } \right) .\tag{6}
$$

From this theorem, we can see that the feature attributions rendered by HiResCAMs can be shifted arbitrarily by a matrix $M \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ . As a result, feature attributions can spuriously correspond to any

![](images/1fbc50c790e39f6cbb04d2ff2d8fbe47b1e28b2200feab224a630341851896da.jpg)  
Figure 2: Illustration of Theorem 3.2. Shifting A by arbitrary matrix M results in new explanations A. Although the explanation changed significantly, it corresponds to an identical final prediction ${ \tilde { f } } .$

given region without affecting the final prediction. This is illustrated in Figure 2. These additional degrees of freedom hinder the effectiveness of regularizing against object-context segmentation during training, since the explanation can be modified without influencing the overall prediction.

To address this issue, we propose to remove the redundant degrees in the feature attribution maps using a contrastive variant of HiResCAMs, termed ContrastiveCAMs.

Definition 3.3 (k-ContrastiveCAMs). Given a class of interest $k \in [ K ]$ and a set of HiResCAMs A for a given image X, k-ContrastiveCAMs are defined as follows:

$$
\mathcal { C } _ { k } : = \left\{ \mathcal { C } _ { ( k , k ^ { \prime } ) } : k ^ { \prime } \in [ K ] \right\} , \qquad \mathrm { w h e r e } \qquad \mathcal { C } _ { ( k , k ^ { \prime } ) } : = \mathcal { A } _ { k } - \mathcal { A } _ { k ^ { \prime } } .\tag{7}
$$

The aggregated version of k-ContrastiveCAM is defined as

$$
\mathcal { C } _ { k } ^ { a g } : = \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } \mathcal { C } _ { ( k , k ^ { \prime } ) } = \mathcal { A } _ { k } - \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } \mathcal { A } _ { k ^ { \prime } } .\tag{8}
$$

Here, $\mathcal { C } _ { ( k , k ^ { \prime } ) }$ is also referred as (k, k<sup>′</sup>)-ContrastiveCAM. Contrary to HiResCAMs $\mathcal { A } _ { k }$ , which focus on single-logit explanations, ContrastiveCAMs provide spatial feature attributions that explain why the model favors one class over a competing class, making them class-discriminative.

Remark 3.4. For binary classification tasks where there is a single output logit and sigmoid applied to obtain the final predictions, HiResCAMs are intrinsically contrastive.

Remark 3.5. The idea of subtracting the average CAM has been applied in DiffGradCAMs [31]. However, DiffGradCAMs are built upon GradCAMs, whereas our ContrastiveCAMs are built upon HiResCAMs. With this distinction, both the following key properties hold for ContrastiveCAMs but not for DiffGradCAMs (c.f. Eq.(2) and Remark 2.1).

Proposition 3.6 (Uniqueness of ContrastiveCAMs). Let A and $\overline { { A } }$ be two HiResCAMs corresponding to probability predictions $\tilde { f } ( \mathbf { X } ) \in \mathbb { R } ^ { K }$ such that:

$$
\begin{array} { r } { \overline { { \mathcal { A } } } _ { k } = \mathcal { A } _ { k } + M , \qquad \forall k \in [ K ] , } \end{array}\tag{9}
$$

with $M \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ being an arbitrary matrix. Then, the ContrastiveCAMs C and ${ \overline { { \mathcal { C } } } } ,$ derived from A and $\overline { { A } }$ respectively, are the same:

$$
\begin{array} { r } { \mathcal { C } = \overline { { \mathcal { C } } } , \qquad \mathrm { a n d } \qquad \mathcal { C } ^ { a g } = \overline { { \mathcal { C } } } ^ { a g } . } \end{array}\tag{10}
$$

Thus ContrastiveCAMs remove the extra degree of freedom, resulting in faithful feature attribution maps. ContrastiveCAMs also have a close connection to the model’s class probabilities. Specifically, any input-dependent change to probability predictions $\tilde { f } \left( \mathrm { e . g . } \right.$ , as a result of updating model parameters) is reflected precisely by a proportionate change to ContrastiveCAMs.

Proposition 3.7 (Correctness of ContrastiveCAM Explanations). Given an input X and a class index $k \in [ K ]$ , the prediction $\tilde { f } _ { k } ( { \mathbf { X } } )$ is determined by the k-ContrastiveCAMs:

$$
\tilde { f } _ { k } ( \mathbf { X } ) = \left( \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } \right) \right) ^ { - 1 } , \quad \forall k \in [ K ] .\tag{11}
$$

Moreover, if for some $k \in [ K ] , i \in [ d _ { 1 } ] , j \in [ d _ { 2 } ] , \left( { \mathcal { C } } _ { ( k , k ^ { \prime } ) } \right) _ { i j } = 0 , \forall k ^ { \prime } \in [ K ]$ , then the (i, j)-patch does not influence the final prediction.

Comparing Eq. (11) with Eq. (2), one can easily find that, given any class index $k ,$ the corresponding HiResCAM only determines the output logit value $f _ { k }$ but not the final prediction ${ \tilde { f } } _ { k } ;$ on the other hand, ContrastiveCAMs directly determines the final prediction ${ \tilde { f } } _ { k } ,$ which makes ContrastiveCAMs a more faithful tool to explain the model’s prediction process. Moreover, the above proposition also shows that, iff the ContrastiveCAMs are zero in a certain spatial region, the contribution from this region of the image to the prediction is indeed zero. HiResCAMs do not share this property.

## 4 Learning with Context-Regularized Cross-Entropy Loss

We now introduce a learning framework designed to be robust to context shifts. In particular, we propose a novel loss function, called Context-Regularized Cross-Entropy (CR-CE), which inhibits the context from contributing to the model prediction. This regularization is guided using the feature attributions produced by ContrastiveCAMs.

To effectively regularize against context contributions during training, prior knowledge of the contextual regions in the input images is generally required. Such information can be obtained in two main ways: (i) through segmentation annotations provided by the dataset (e.g., PASCAL VOC [16], MS-COCO [22]), or (ii) by leveraging pretrained object detection or segmentation models (e.g., Segment Anything Model [5], Masked AutoEncoders [18]) to estimate bounding boxes or object masks (see experiments on ImageNet, Spawrious datasets in Section 5). In practice, the first option is often unavailable or prohibitively expensive. We adopt the second approach which is more accessible, as pretrained models can provide approximate context localization with minimal effort. Note that, our framework does not require precise object segmentation or localization, and rough estimates are often sufficient to effectively regularize contextual influence. In Appendix C.4, we show numerical evidence for the robustness of our framework to noisy object-context segmentation.

Given an image X, denote $H \in \mathbb { B } ^ { d _ { 1 } \times d _ { 2 } }$ as the binary mask that distinguishes object regions from context. Specifically, for each $( i , j )$ -patch where $i \in \lbrack { d _ { 1 } } ] , j \in [ d _ { 2 } ] , H _ { i j } \overset { = } = 1$ if the patch corresponds to the object, and $H _ { i j } = 0$ if it corresponds to the context. We define CR-CE as follows:

Definition 4.1 (Context-Regularized Cross-Entropy Loss). Given a training dataset D and convolutional network $f ,$ , the CR-CE loss is defined as:

$$
{ \mathcal { L } } _ { \mathrm { C R C E } } ( { \mathcal { D } } ) = \mathbb { E } _ { ( \mathbf { X } , \mathbf { y } , H ) \sim { \mathcal { D } } } [ \ell ( \mathbf { X } , \mathbf { y } , H ) ] , \ \mathrm { w i t h }
$$

$$
\ell ( \mathbf { X } , \mathbf { y } , H ) = \log ( \sum _ { k = 1 } ^ { K } \exp ( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } - H _ { i j } ( \mathcal { C } _ { ( k _ { t } , k ) } ) _ { i j } + \underbrace { ( 1 - H _ { i j } ) \vert ( \mathcal { C } _ { ( k _ { t } , k ) } ) _ { i j } \vert } _ { \mathrm { c o n t e x t r e g u l a r i z a t i o n } } ) ) .\tag{12}
$$

where $k _ { t }$ is the index of the ground-truth class. For simplicity of notation, we do not explicitly write the dependence of H and ContrastiveCAMs C on the input image X.

The context regularization term in the CR-CE loss regularizes the absolute values of the Contrastive-CAMs $| ( \overline { { C } } _ { ( k _ { t } , k ) } ) _ { i j } |$ at context regions, driving them towards zero, thus minimizing the influence of context in the final prediction ${ \tilde { f } } \ ( \mathrm { c . f . }$ Proposition 3.7). We also extend context regularization to binary classification settings; we defer this discussion to Appendix B.

Comparison with standard cross-entropy loss. The only difference between the standard crossentropy (CE) loss and CR-CE lies in the context regularization term. In particular, if the absolute values of the ContrastiveCAM, $| ( \mathcal { C } _ { ( k _ { t } , k ) } ) _ { i j } |$ , are replaced with their negated values $, - ( \mathcal { C } _ { ( k _ { t } , k ) } ) _ { i j }$ , the

CR-CE loss reduces to the standard cross-entropy loss, noting that cross-entropy loss on one sample $\ell _ { C E } ( \mathbf { X } , \mathbf { y } ) = - \log \tilde { f } _ { k _ { t } }$ as well as the relation in Proposition 3.7.

Despite this seemingly minor modification, the effect on model behavior is substantial. The standard cross-entropy loss implicitly encourages the context to cooperate with the object regions in making predictions. While this is beneficial when the context is consistent with the object, it can lead to an over-reliance on contextual cues and thus reduce robustness under context shift. In contrast, CR-CE explicitly penalizes excessive reliance on context, encouraging the model to base its predictions primarily on the object itself. As a result, when the context is altered or adversarially manipulated, models trained with CR-CE exhibit improved robustness.

Further, for models using a linear classifier head (e.g., ResNet [17], MobileNet [19], EfficientNet [42]), ContrastiveCAM can be computed in closed form, see Eqs. (1) and (7); this results in a minor computational overhead when CR-CE is used in place of cross-entropy.

## 5 Experimental Results

In this section, we numerically evaluate the effectiveness of training with the proposed CR-CE loss function in both promoting object-focused learning and improving robustness to context shifts.

## 5.1 Datasets and Experimental Setup

Datasets. We evaluate our method using the following four datasets:

• Waterbirds-100 dataset [37] consists of images of waterbirds and landbirds. In the training set, each class of bird appears in the context of their natural habitat–waterfront and land, respectively. However, in the test set, context shifts may occur, with birds appearing in the opposite environment.

• Spawrious dataset [23] consists of AI-generated images of four classes of dog breeds appearing in various contexts. The test set contains combinations of dog breed and context that are not seen during training, thereby assessing models’ robustness to context shift.

• ImageNet [8] with ImageNet-BG [30] as test set. We train the model using ImageNet-1k and test it using ImageNet-BG, the latter of which features images with manually replaced backgrounds, designed to decouple object recognition from context cues.

• Hard-ImageNet [25] is a subset of ImageNet, consisting of 15 selected classes of images that have high context influence. It is used to investigate the context’s influence on the classification by comparing the model outputs with and without the objects of the images.

For all the datasets above, context shifts only occur in the test set, which is not accessed during training. Ground-truth object-context masks are available for the training sets of Waterbirds-100 and Hard-ImageNet. For Spawrious and ImageNet where object-context segmentation is not available, we use existing segmentation models such as SAM3 [5] and Masked AutoEncoders (MAE) [18, 13] to generate these masks. Our approach allows for imperfect segmentation, enabling the effective use of off-the-shelf vision models when ground truth masks are unavailable.

Experimental Setup. We follow the methodology in the benchmark datasets [37, 23, 30, 25] and fine-tune ResNet-50 models pre-trained on ImageNet-1k. In our experiments, we also evaluate a modified ResNet architecture where the final downsampling convolution layer’s stride is set to (1, 1), thus doubling the resolution of ContrastiveCAMs, i.e., $\bar { d } _ { 1 } = d _ { 2 } = 1 4$ . The higher-resolution attribution maps provide more precise localization of object regions. More experiments with different architectures can be found in Appendix C.5. We compare our proposed CR-CE framework with the following prior approaches:

• The standard training procedure with cross-entropy loss (CE), which does not implement any context-shift robustness technique. It serves as a baseline for comparison.

• Explanation-guided methods, which improve context-shift robustness by guiding models’ feature attributions, i.e., the heatmaps, towards the target object region. Our proposed method falls in this category. We compare with existing methods: attention branch networks (ABN) [11], GALS [29], and training with energy-based loss (EBL) [34].

![](images/a6586d0c3c7c9f577bb3ac94992c02e049e452713933f8b45af3e1107a221b65.jpg)  
Figure 3: Qualitative Evaluation. Models trained using cross-entropy rely heavily on context: in the Hard-ImageNet (left) and ImageNet-BG (right) examples shown above, the baseball player and bee are predicted using the playing field (61.4% of the prediction) and the flower (55.4% of the prediction) respectively. On the other hand, models trained using CR-CE leverage object-specific regions and localize the target without the use of object-context segmentation during inference.

Table 1: Context Contribution Ratios. On the test set of each context-shift robustness benchmark we consider, CR-CE trained models primarily rely on object-specific regions to determine predictions.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="4">Context Contribution Ratio R% (↓)</td></tr><tr><td>Waterbirds-100</td><td> $\operatorname { S p a w r i o u s } ^ { \ddagger }$ </td><td>ImageNet-BG</td><td>Hard-ImageNet</td></tr><tr><td rowspan="2"> $7 \times 7$ </td><td>CE (Baseline)</td><td> $4 6 . 0 8 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $4 4 . 7 9 2 0 . 3 $ </td><td> $3 1 . 8 3 { \scriptstyle \pm 0 }$ </td><td> $6 8 . 5 3 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $\mathbf { 1 2 . 8 6 { \scriptstyle \pm 0 . 0 9 } }$ </td><td> $\mathbf { 1 4 . 8 9 { \scriptstyle \pm 0 . 2 } }$ </td><td> $\mathbf { 1 4 . 8 5 { \scriptstyle \pm 0 } }$ </td><td>35.85±0.26</td></tr><tr><td rowspan="2"> $1 4 \times 1 4$ </td><td>CE (Baseline)</td><td> $5 8 . 7 5 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $4 7 . 3 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $2 8 . 2 7 { \scriptstyle \pm 0 . 2 7 }$ </td><td>66.5±0.07</td></tr><tr><td>CR-CE (Ours)</td><td> $\mathbf { 9 . 5 4 { \scriptstyle \pm 0 . 2 8 } }$ </td><td> $\mathbf { 1 0 . 9 8 { \scriptstyle \pm 0 . 1 2 } }$ </td><td> ${ \bf 9 . 3 2 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> ${ \bf 3 1 . 1 2 } _ { \pm 0 . 1 }$ </td></tr></table>

<sup>†</sup> No std. dev. is reported; we use pre-trained weights from torchvision without modification.  
<sup>‡</sup> We report metrics on the hardest subtask and difficulty (M2M-hard) within the Spawrious benchmark.

• Other methods, including Context Masking training [1] and Causal Feature Alignment [44]. Both methods use object-context masks to transform the input/hidden features in order to discourage context reliance. Particularly, Context Masking only uses the object regions of images for training by masking out all context regions.

For fairness of comparison, we use identical object-context masks across approaches. For all our metrics, we report the mean and standard deviation computed over 3 independent runs. Details of the experimental setup are described in Appendix E.

## 5.2 CR-CE Suppresses Context Contributions

First, we evaluate the proportion of context contributions using the following metric:

$$
R ( \mathcal { C } ^ { a g } , H ) = \frac { \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } | ( \mathcal { C } _ { k _ { t } } ^ { a g } ) _ { i j } | ( 1 - H _ { i j } ) } { \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } | ( \mathcal { C } _ { k _ { t } } ^ { a g } ) _ { i j } | } ,\tag{13}
$$

where $k _ { t } \in [ K ]$ is the target class index. This metric represents the portion of the absolute value of the ContrastiveCAM heatmap from the context region. A comparison of the proposed CR-CE training framework with the standard CE training is shown in Table 1, with Figure 3 and Figure 5 presenting examples. Comparisons with the other methods assessed in this work are included in Table 5. We observe that training with CR-CE effectively guides ContrastiveCAMs onto the object regions and reduces the model’s reliance on the context region.

Additional evaluation of the effectiveness of CR-CE in suppressing context contributions using the Hard-ImageNet evaluation suite can be found in Appendix C.2.

Table 2: Context-Shift Robustness on Waterbirds-100, ImageNet, and Spawrious datasets. Accuracy across various benchmarks, with the best context-shifted performance highlighted in bold and the highest in-distribution performance underlined. Models trained with CR-CE generalize effectively to novel environments at a minor cost to in-distribution performance.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="2">Waterbirds-100</td><td colspan="2">ImageNet/ImageNet-BG</td><td colspan="3">Spawrious</td></tr><tr><td>No Shift (↑)</td><td>Shifted (↑)</td><td>No Shift (↑)</td><td>Shifted (↑)</td><td>Easy (↑)</td><td>Medium (↑)</td><td>Hard (↑)</td></tr><tr><td rowspan="6"> $7 \times 7$ </td><td>CE (Baseline)</td><td> $9 8 . 6 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $4 1 . 7 4 { \scriptstyle \pm 4 . 1 5 }$ </td><td>82.92†</td><td> $7 4 . 8 3 ^ { \dagger }$ </td><td> $9 0 . 8 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $7 9 . 4 6 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $8 1 . 5 2 { \scriptstyle \pm 1 . 0 3 }$ </td></tr><tr><td>GALS</td><td> $9 7 . 5 4 { \pm } 1 . 0 1 $ </td><td> $5 4 . 4 5 { \scriptstyle \pm 6 . 7 5 }$ </td><td> $8 2 . 6 8 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $7 3 . 8 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $9 3 . 4 7 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $8 4 . 5 8 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $7 8 . 7 4 { \scriptstyle \pm 0 . 9 7 }$ </td></tr><tr><td>EBL</td><td> $9 6 . 3 5 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 1 . 3 1 { \scriptstyle \pm 4 . 7 0 }$ </td><td> $8 2 . 8 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $7 4 . 7 9 2 0 . 0 1$ </td><td> $9 2 . 4 3 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $8 7 . 2 3 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $8 1 . 2 1 { \scriptstyle \pm 0 . 7 4 }$ </td></tr><tr><td>CFA</td><td> $9 3 . 4 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 2 . 7 1 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $8 2 . 7 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 3 . 8 5 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $8 9 . 1 4 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $8 5 . 6 8 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $8 1 . 7 7 { \scriptstyle \pm 1 . 7 3 }$ </td></tr><tr><td>Context Masking</td><td> $9 1 . 1 6 { \scriptstyle \pm 3 . 1 8 }$ </td><td> $6 1 . 6 7 { \scriptstyle \pm 6 . 1 7 }$ </td><td> $8 0 . 2 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 2 . 8 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $9 2 . 0 8 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $8 5 . 0 3 { \scriptstyle \pm 2 . 3 7 }$ </td><td> $7 8 . 4 6 { \scriptstyle \pm 1 . 1 2 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $9 4 . 3 8 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $\mathbf { 7 3 . 3 0 { \pm } _ { 4 . 4 3 } }$ </td><td> $8 1 . 8 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> ${ \bf 7 5 . 7 5 { \scriptstyle \pm 0 . 0 8 } }$ </td><td> $\mathbf { 9 5 . 5 5 } { \scriptstyle \pm 0 . 1 6 }$ </td><td> $\mathbf { 9 1 . 2 5 { \scriptstyle \pm 0 . 1 8 } }$ </td><td> $\mathbf { 8 7 . 4 9 } 2 0 . 1 4$ </td></tr><tr><td rowspan="7"> $1 4 \times 1 4$ </td><td>CE (Baseline)</td><td> $9 8 . 4 8 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $3 8 . 4 0 { \scriptstyle \pm 1 . 7 9 }$ </td><td> $8 2 . 3 3 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $7 4 . 7 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $8 8 . 9 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 3 . 3 8 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $7 9 . 7 6 { \scriptstyle \pm 1 . 0 1 }$ </td></tr><tr><td>ABN</td><td> $9 7 . 3 9 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $5 1 . 8 7 { \scriptstyle \pm 6 . 1 4 }$ </td><td> $7 9 . 9 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $6 9 . 0 7 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $8 9 . 6 5 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $7 2 . 8 1 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $6 3 . 9 6 { \scriptstyle \pm 6 . 5 4 }$ </td></tr><tr><td>GALS</td><td> $9 7 . 3 9 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $6 3 . 2 1 { \scriptstyle \pm 4 . 9 0 }$ </td><td> $8 2 . 4 3 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $7 5 . 2 1 \pm 0 . 0 2$ </td><td> $9 5 . 0 3 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $8 7 . 6 3 { \scriptstyle \pm 3 . 8 0 }$ </td><td> $8 4 . 0 8 { \scriptstyle \pm 1 . 8 7 }$ </td></tr><tr><td>EBL</td><td> $9 6 . 3 3 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $5 1 . 4 9 { \scriptstyle \pm 5 . 3 1 }$ </td><td> $8 2 . 7 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $7 5 . 7 9 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $9 3 . 4 6 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $8 9 . 0 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $7 9 . 3 6 { \scriptstyle \pm 2 . 8 3 }$ </td></tr><tr><td>CFA</td><td> $9 3 . 0 9 2 0 . 7 4 $ </td><td> $7 7 . 5 1 { \scriptstyle \pm 3 . 8 9 }$ </td><td> $7 9 . 5 4 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $7 3 . 4 8 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $8 7 . 2 2 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $8 5 . 6 8 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $7 5 . 8 8 { \scriptstyle \pm 1 . 1 2 }$ </td></tr><tr><td>Context Masking</td><td> $9 2 . 2 7 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $6 4 . 3 9 { \scriptstyle \pm 1 . 7 9 }$ </td><td> $8 0 . 7 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 5 . 4 9 2 0 . 1 0$ </td><td> $9 0 . 9 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 5 . 7 0 { \scriptstyle \pm 2 . 9 9 }$ </td><td> $8 0 . 8 0 { \scriptstyle \pm 0 . 7 1 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $9 4 . 3 6 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $\mathbf { 7 9 . 1 3 { \scriptstyle \pm 0 . 7 7 } }$ </td><td> $8 1 . 2 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $\mathbf { 7 6 . 8 4 } \pm \mathbf { 0 . 1 2 }$ </td><td> $\mathbf { 9 6 . 4 7 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> $\mathbf { 9 2 . 4 1 } { \scriptstyle \pm 0 . 2 9 }$ </td><td> $\mathbf { 8 8 . 6 6 } \pm \mathrm { 0 . 5 2 }$ </td></tr></table>

No std. dev. is reported; we use pre-trained weights from torchvision without modification.  
<sup>‡</sup> The reported metrics correspond to the many-to-many spurious correlation setting; the complete benchmark is assessed in Appendix C.1.

## 5.3 Reduced Context-Contributions Improve Robustness to Context Shifts

Next, we demonstrate the improved performance of CR-CE trained models under context shift by comparing them against the standard cross-entropy training, existing explanation-guided learning approaches [29, 11, 34] and other methods [44, 1]. We report the numerical results in Table 2.

We observe that across all datasets, the CR-CE training framework consistently improves the robustness to context shift and further outperforms models trained using cross-entropy loss and the other compared methods. This demonstrates that the proposed method successfully suppresses contribution from irrelevant or misleading context, yielding correct and robust model predictions. We also note that, when the context is relevant (as in the setting without context shift), there is a slight degradation in the prediction accuracy across tasks. This is expected because when the test dataset and training dataset have the same set of object-context combinations, the model can often infer the correct class with the help of the context. Indeed, such correlations are often not desirable if context-shift robustness is the target.

Additionally, we observe that some existing explanation-guided methods may underperform the standard cross-entropy baseline on complex classification tasks, such as ImageNet, while outperforming it on simpler tasks such as the Waterbirds-100 dataset. We hypothesize that this behavior is largely due to their reliance on classical feature attribution methods (e.g. Grad-CAM and HiResCAM), which suffer from the unfaithfulness issue theoretically described in Theorem 3.2. In the binary classification setting of Waterbirds-100, HiResCAM coincides with our ContrastiveCAMs (see Remark 3.4) and does not exhibit this limitation. More discussion of explanation-guided methods on this matter can be found in Appendix C.3.

Notably, we emphasize that our approach does not use object-context segmentation during inference; the promising performance of our approach suggests that the CR-CE trained models have the ability to distinguish objects from context while making predictions.

Finally, we also note that the generalization gap between in-distribution and context-shifted images shrinks as explanation granularity is increased (comparing the $7 \times 7$ and 14 × 14 CAM resolutions).

## 6 Conclusion

In this work, we advance explanation faithfulness and context robustness in convolutional networks. We introduce ContrastiveCAMs, a feature attribution method that addresses limitations of existing CAM methods and has a strong connection to model predictions. We further propose a new loss function, Context-Regularized Cross-Entropy (CR-CE), which penalizes contributions from context regions, thereby improving models’ robustness to context shifts. Our experimental evaluations show that our method effectively attracts the feature attributions towards the object region, and enhances context-shift robustness.

Limitations. We identify two limitations of this work. First, the method does not naturally extend to transformer architectures, such as Vision Transformers (ViTs) [9]. This is because CAM-based spatial feature attribution methods rely on the local structure induced by convolutional layers, whereas attention mechanisms [43] mix patch tokens in a way that loses explicit spatial locality. Second, the proposed CR-CE training requires additional object-context segmentation information (albeit only during training), which requires additional cost to obtain in practice. We note that existing explanation-guided methods also rely on this information.

## Acknowledgment

This research used both the DeltaAI advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2320345) and the State of Illinois, and the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

## References

[1] Ananthu Aniraj, Cassio F Dantas, Dino Ienco, and Diego Marcos. Masking strategies for background bias removal in computer vision models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4397–4405, 2023.

[2] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

[3] Pedro RAS Bassi, Sergio SJ Dertkigil, and Andrea Cavalli. Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization. Nature Communications, 15(1):291, 2024.

[4] Christopher Boland, Keith A Goatman, Sotirios A Tsaftaris, and Sonia Dahdouh. There are no shortcuts to anywhere worth going: Identifying shortcuts in deep learning models for medical image analysis. In Medical Imaging with Deep Learning, pages 131–150. PMLR, 2024.

[5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pages 138846–138923, 2026.

[6] Aditya Chattopadhay, Anirban Sarkar, Prantik Howlader, and Vineeth N Balasubramanian. Gradcam++: Generalized gradient-based visual explanations for deep convolutional networks. In 2018 IEEE Winter Conference on Applications ofComputer Vision (WACV). IEEE, March 2018. doi: 10.1109/wacv.2018.00097. URL http://dx.doi.org/10.1109/WACV.2018.00097.

[7] Mathieu Chevalley, Charlotte Bunne, Andreas Krause, and Stefan Bauer. Invariant causal mechanisms through distribution matching. In International Conference on Learning Representations, 2022.

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. IEEE, 2009.

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[10] Rachel Lea Draelos and Lawrence Carin. Use hirescam instead of grad-cam for faithful explanations of convolutional neural networks. arXiv preprint arXiv:2011.08891, 2020.

[11] Hiroshi Fukui, Tsubasa Hirakawa, Takayoshi Yamashita, and Hironobu Fujiyoshi. Attention branch network: Learning of attention mechanism for visual explanation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10705–10714, 2019.

[12] Irena Gao, Shiori Sagawa, Pang Wei Koh, Tatsunori Hashimoto, and Percy Liang. Out-ofdomain robustness via targeted augmentations. In International conference on machine learning, pages 10800–10834. PMLR, 2023.

[13] Shanghua Gao, Zhong-Yu Li, Ming-Hsuan Yang, Ming-Ming Cheng, Junwei Han, and Philip Torr. Large-scale unsupervised semantic segmentation. IEEE transactions on pattern analysis and machine intelligence, 45(6):7457–7476, 2022.

[14] Yuyang Gao, Siyi Gu, Junji Jiang, Sungsoo Ray Hong, Dazhou Yu, and Liang Zhao. Going beyond xai: A systematic survey for explanation-guided learning. ACM Computing Surveys, 56 (7):1–39, 2024.

[15] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[16] Bharath Hariharan, Pablo Arbeláez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In 2011 international conference on computer vision, pages 991–998. IEEE, 2011.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[18] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

[19] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1314–1324, 2019.

[20] Aya Abdelsalam Ismail, Hector Corrada Bravo, and Soheil Feizi. Improving deep learning interpretability by saliency guided training. Advances in Neural Information Processing Systems, 34:26726–26739, 2021.

[21] Jinkyu Kim and John Canny. Interpretable learning for self-driving cars by visualizing causal attention. In Proceedings of the IEEE international conference on computer vision, pages 2942–2950, 2017.

[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[23] Aengus Lynch, Gbetondji Jean-Sebastien Dovonon, Jean Kaddour, and Ricardo Silva. Spawrious: A benchmark for fine control of spurious correlation biases. In Workshop on Spurious Correlation and Shortcut Learning: Foundations and Solutions, 2025.

[24] Mazda Moayeri, Phillip Pope, Yogesh Balaji, and Soheil Feizi. A comprehensive study of image classification model sensitivity to foregrounds, backgrounds, and visual attributes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19087–19097, 2022.

[25] Mazda Moayeri, Sahil Singla, and Soheil Feizi. Hard imagenet: Segmentations for objects with strong spurious cues. Advances in Neural Information Processing Systems, 35:10068–10077, 2022.

[26] Jamuna S Murthy and GM Siddesh. Ai based criminal detection and recognition system for public safety and security using novel criminalnet-228. In International Conference on Frontiers in Computing and Systems, pages 3–20. Springer, 2023.

[27] Meike Nauta, Ricky Walsh, Adam Dubowski, and Christin Seifert. Uncovering and correcting shortcut learning in machine learning models for skin cancer diagnosis. Diagnostics, 12(1):40, 2021.

[28] Yoojin Oh and Junhyug Noh. Beyond softmax: Dual-branch sigmoid architecture for accurate class activation maps. arXiv preprint arXiv:2511.05590, 2025.

[29] Suzanne Petryk, Lisa Dunlap, Keyan Nasseri, Joseph Gonzalez, Trevor Darrell, and Anna Rohrbach. On guiding visual attention with language specification. Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[30] Łukasz Piekarek and Kamil Szyc. Imagenet-bg: A toolkit and dataset for evaluating vision model robustness against background variations. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 231–240, October 2025.

[31] Jacob Piland, Christopher Sweet, and Adam Czajka. Diffgradcam: A class activation map using the full model decision to solve unaddressed adversarial attacks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1201–1210, 2026.

[32] Boris T Polyak. Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics, 4(5):1–17, 1964.

[33] Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, et al. Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. arXiv preprint arXiv:1711.05225, 2017.

[34] Sukrut Rao, Moritz Böhle, Amin Parchami-Araghi, and Bernt Schiele. Studying how to efficiently and effectively guide models with explanations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1922–1933, 2023.

[35] Andrew Slavin Ross, Michael C. Hughes, and Finale Doshi-Velez. Right for the right reasons: Training differentiable models by constraining their explanations. In Proceedings ofthe Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17, pages 2662–2670, 2017. doi: 10.24963/ijcai.2017/371. URL https://doi.org/10.24963/ijcai.2017/371.

[36] Khaled Saab, Sarah Hooper, Mayee Chen, Michael Zhang, Daniel Rubin, and Christopher Ré. Reducing reliance on spurious features in medical image classification with spatial specificity. In Machine Learning for Healthcare Conference, pages 760–784. PMLR, 2022.

[37] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks. In International Conference on Learning Representations, 2020.

[38] DR Sarvamangala and Raghavendra V Kulkarni. Convolutional neural networks in medical image understanding: a survey. Evolutionary intelligence, 15(1):1–22, 2022.

[39] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pages 618–626, 2017.

[40] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In Workshop at International Conference on Learning Representations, 2014.

[41] Sahil Singla and Soheil Feizi. Salient imagenet: How to discover spurious features in deep learning? In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=XVPqLyNxSyh.

[42] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.

[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[44] Rahul Venkataramani, Parag Dutta, Vikram Melapudi, and Ambedkar Dukkipati. Causal feature alignment: Learning to ignore spurious background features. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4666–4674, 2024.

[45] Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, and Xia Hu. Score-cam: Score-weighted visual explanations for convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 24–25, 2020.

[46] Zifan Wang. On the Feature Alignment of Deep Vision Models Explainability and Robustness Connected at Hip. PhD thesis, Carnegie Mellon University, 2023.

[47] Kai Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. Noise or signal: The role of image backgrounds in object recognition. arXiv preprint arXiv:2006.09994, 2020.

[48] Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Heng Tao Shen, and Jingkuan Song. Shortcut learning in generalist robot policies: The role of dataset diversity and fragmentation. In Conference on Robot Learning, pages 3239–3266. PMLR, 2025.

[49] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2921–2929, 2016.

## A Proofs

Proposition 3.1 (Redundancy in the softmax function).

$$
\sigma ( { \bf x } ) = \sigma ( { \bf x } + a { \bf 1 } _ { K } ) , \qquad \forall { \bf x } \in \mathbb { R } ^ { K } , \ a \in \mathbb { R } .\tag{14}
$$

Proof.

$$
\sigma ( \mathbf x + a \mathbf 1 _ { K } ) = { \frac { 1 } { \sum _ { k = 1 } ^ { K } e ^ { \mathbf x _ { k } + a } } } ( e ^ { \mathbf x _ { 1 } + a } , e ^ { \mathbf x _ { 2 } + a } , \cdots , e ^ { \mathbf x _ { K } + a } ) = { \frac { 1 } { \sum _ { k = 1 } ^ { K } e ^ { \mathbf x _ { k } } } } ( e ^ { \mathbf x _ { 1 } } , e ^ { \mathbf x _ { 2 } } , \cdots , e ^ { \mathbf x _ { K } } ) = \sigma ( \mathbf x ) .\tag{15}
$$

□

Theorem 3.2. Let $\mathcal { A } \in \mathbb { R } ^ { K \times d _ { 1 } \times d _ { 2 } }$ be a HiResCAM. Given an arbitrary matrix $M \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ , define a new HiResCAM as:

$$
\begin{array} { r } { \overline { { \mathcal { A } } } _ { k } : = \mathcal { A } _ { k } + M , \qquad \forall k \in [ K ] . } \end{array}\tag{16}
$$

Then, the two HiResCAMs correspond to the same network prediction $\tilde { f } ( { \mathbf { X } } )$

$$
\tilde { f } ( \mathbf { X } ) = \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \mathcal { A } _ { : i j } \right) = \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \overline { { \mathcal { A } } } _ { : i j } \right) .\tag{17}
$$

Proof. The proof for this statement follows from the relation between the two HiResCAMs. For a given shift $M ,$ the predictions from the corresponding HiResCAM A can be expressed as:

$$
\sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \overline { { \mathcal { A } } } _ { : i j } \right) = \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } ( \mathcal { A } _ { : i j } + M _ { i j } ) \right) .\tag{18}
$$

Let $\begin{array} { r } { a = \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } M _ { i j } } \end{array}$ . We can equivalently write the above statement as:

$$
= \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \mathcal { A } _ { : i j } + a \mathbf { 1 } _ { K } \right) .\tag{19}
$$

Applying Proposition 3.1, we have:

$$
\sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \overline { { \mathcal { A } } } _ { : i j } \right) = \sigma \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \mathcal { A } _ { : i j } \right) = \tilde { f } ( \mathbf { X } ) .\tag{20}
$$

Proving the desired statement.

Proposition 3.6 (Uniqueness of ContrastiveCAMs). Let A and $\overline { { A } }$ be two HiResCAMs corresponding to probability predictions $\tilde { f } ( \mathbf { X } ) \in \mathbb { R } ^ { K }$ such that:

$$
\begin{array} { r } { \overline { { \mathcal { A } } } _ { k } = A _ { k } + M , \qquad \forall k \in [ K ] , } \end{array}\tag{21}
$$

with $M \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ being an arbitrary matrix. Then, the ContrastiveCAMs C and ${ \overline { { \mathcal { C } } } } ,$ , derivedfrom A and $\overline { { A } }$ respectively, are the same:

$$
\begin{array} { r } { \mathcal { C } = \overline { { \mathcal { C } } } , \qquad \mathrm { a n d } \qquad \mathcal { C } ^ { a g } = \overline { { \mathcal { C } } } ^ { a g } . } \end{array}\tag{22}
$$

Proof. First, consider k-ContrastiveCAMs. We have that:

$$
\overline { { \mathcal { C } } } _ { k } = \{ \overline { { \mathcal { C } } } _ { \left( k , k ^ { \prime } \right) } : k ^ { \prime } \in \left[ K \right] \}\tag{23}
$$

$$
= \{ \mathcal { A } _ { k } + M - \mathcal { A } _ { k ^ { \prime } } - M : k ^ { \prime } \in [ K ] \}\tag{24}
$$

$$
= \{ \mathcal { A } _ { k } - \mathcal { A } _ { k ^ { \prime } } : k ^ { \prime } \in [ K ] \}\tag{25}
$$

$$
= \{ \mathcal { C } _ { ( k , k ^ { \prime } ) } : k ^ { \prime } \in [ K ] \} = \mathcal { C } _ { k } , \quad \forall k \in [ K ] .\tag{26}
$$

Similarly, in the case of aggregated k-ContrastiveCAMs, we have:

$$
\overline { { \mathcal { C } } } _ { k } ^ { a g } = \overline { { \mathcal { A } } } _ { k } - \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } \overline { { \mathcal { A } } } _ { k ^ { \prime } }\tag{27}
$$

$$
= \mathcal { A } _ { k } + M - \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } ( \mathcal { A } _ { k ^ { \prime } } + M )\tag{28}
$$

$$
= \mathcal { A } _ { k } + M - \frac { K \cdot M } { K } - \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } \mathcal { A } _ { k ^ { \prime } }\tag{29}
$$

$$
\mathbf { \Omega } = \mathcal { A } _ { k } - \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } \mathcal { A } _ { k ^ { \prime } } = \mathcal { C } _ { k } ^ { a g } , \quad \forall k \in [ K ] .\tag{30}
$$

Eqs. (26) and (30) together prove the desired equalities.

Proposition 3.7(Correctness of ContrastiveCAM Explanations). Given an input X and a class index $k \in [ K ]$ , the prediction $\tilde { f } _ { k } ( { \mathbf { X } } )$ is determined by the k-ContrastiveCAMs:

$$
\tilde { f } _ { k } ( \mathbf { X } ) = \left( \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } \right) \right) ^ { - 1 } , \quad \forall k \in [ K ] .\tag{31}
$$

Moreover, if for some $k \in [ K ] , i \in [ d _ { 1 } ] , j \in [ d _ { 2 } ] , \left( { \mathcal { C } } _ { ( k , k ^ { \prime } ) } \right) _ { i j } = 0 , \forall k ^ { \prime } \in [ K ]$ , then the (i, j)-patch does not influence the final prediction.

Proof. For a class index $k \in [ K ]$ , the prediction probability given a logit vector f is expressed as:

$$
\tilde { f } _ { k } = \sigma _ { k } ( f ) = \frac { e ^ { f _ { k } } } { \sum _ { k ^ { \prime } = 1 } ^ { K } e ^ { f _ { k ^ { \prime } } } } .\tag{32}
$$

We define the logit vector in terms of the elementwise difference to a candidate class k:

$$
\mathbf { d } : = f - f _ { k } \implies f = f _ { k } + \mathbf { d } .\tag{33}
$$

Based on this definition, class probabilities can equivalently be computed as:

$$
{ \tilde { f } } _ { k } = { \frac { e ^ { f _ { k } } } { \sum _ { k ^ { \prime } = 1 } ^ { K } e ^ { f _ { k ^ { \prime } } } } } = { \frac { e ^ { f _ { k } } } { \sum _ { k ^ { \prime } = 1 } ^ { K } e ^ { f _ { k } + \mathbf { d } _ { k ^ { \prime } } } } } = { \frac { e ^ { f _ { k } } } { e ^ { f _ { k } } \sum _ { k ^ { \prime } = 1 } ^ { K } e ^ { \mathbf { d } _ { k ^ { \prime } } } } } = { \frac { 1 } { \sum _ { k ^ { \prime } = 1 } ^ { K } e ^ { \mathbf { d } _ { k ^ { \prime } } } } } .\tag{34}
$$

This re-contextualizes softmax as a direct function of the differences between class logits. We can further deconstruct the difference by logit values:

$$
\mathbf { d } _ { k ^ { \prime } } = f _ { k ^ { \prime } } - f _ { k } = \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } { \mathcal { A } } _ { k ^ { \prime } i j } - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } { \mathcal { A } } _ { k i j } , \quad \forall k , k ^ { \prime } \in [ K ] .\tag{35}
$$

Applying Definition 3.3, we have:

$$
\mathbf { d } _ { k ^ { \prime } } = \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } ( \mathcal { A } _ { k ^ { \prime } } - \mathcal { A } _ { k } ) _ { i j } = - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } , \quad \forall k , k ^ { \prime } \in [ K ] .\tag{36}
$$

Substituting $\mathbf { d } _ { k ^ { \prime } }$ from Eq. (36) into Eq. (34), we have:

$$
\begin{array} { r l } & { \tilde { f } _ { k } ( \mathbf { X } ) = \frac { 1 } { \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \Big ( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \big ( \mathcal { C } _ { ( k , k ^ { \prime } ) } \big ) _ { i j } \Big ) } } \\ & { = \left( \displaystyle \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \big ( \mathcal { C } _ { ( k , k ^ { \prime } ) } \big ) _ { i j } \right) \right) ^ { - 1 } , \quad \forall k \in [ K ] . } \end{array}\tag{37}
$$

(38)

We can thus compute class probabilities as a direct function of ContrastiveCAMs.

Next, we show that zero-patches do not influence predictions. For a given class $k \in [ K ]$ , denote the set of zero-patches as $\mathcal { Z } _ { k }$ , and its complement as $\dot { \mathcal { Z } } _ { k } ^ { \prime }$ :

$$
\mathcal { Z } _ { k } = \left\{ \left( i , j \right) \in [ d _ { 1 } ] \times [ d _ { 2 } ] : \left( \mathcal { C } _ { \left( k , k ^ { \prime } \right) } \right) _ { i j } = 0 , \forall k ^ { \prime } \in [ K ] \right\} , \quad \mathrm { a n d } \quad \mathcal { Z } _ { k } ^ { \prime } = \left( [ d _ { 1 } ] \times [ d _ { 2 } ] \right) \setminus \mathcal { Z } _ { k } .\tag{39}
$$

Applying this partition to Eq. (38), we have that:

$$
\tilde { f } _ { k } ( \mathbf { X } ) = \left( \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } \right) \right) ^ { - 1 }\tag{40}
$$

$$
= \left( \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j \in \mathcal { Z } _ { k } ^ { \prime } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } - \sum _ { i , j \in \mathcal { Z } _ { k } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } \right) \right) ^ { - 1 }\tag{41}
$$

$$
= \left( \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j \in \mathcal { Z } _ { k } ^ { \prime } } \left( \mathcal { C } _ { ( k , k ^ { \prime } ) } \right) _ { i j } - \sum _ { i , j \in \mathcal { Z } _ { k } } 0 \right) \right) ^ { - 1 }\tag{42}
$$

$$
= \left( \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \left( - \sum _ { i , j \in \mathcal { Z } _ { k } ^ { \prime } } \left( \mathcal { C } _ { \left( k , k ^ { \prime } \right) } \right) _ { i j } \right) \right) ^ { - 1 } .\tag{43}
$$

Note that, for some $k \in [ K ] , \left( \mathcal { C } _ { \left( k , k ^ { \prime } \right) } \right) _ { i j } = 0$ for all $k ^ { \prime } \in [ K ]$ is equivalent to $( \mathcal { A } _ { k } ) _ { i j } = ( \mathcal { A } _ { k ^ { \prime } } ) _ { i j }$ for all $k ^ { \prime } \in [ K ]$ , which further implies that $\left( \mathcal { C } _ { \left( k , k ^ { \prime } \right) } \right) _ { i j } = 0$ for all $k , k ^ { \prime } \in [ K ]$ . This means that $\mathcal { Z } _ { k } = \mathcal { Z } _ { k ^ { \prime } }$ for all k, $k ^ { \prime } \in \left[ K \right]$ . Therefore, Eq. (43) holds for all $k \in [ K ]$ , implying all the logit outputs are not affected by the patch $( i , j )$ □

Remark 3.4. For binary classification tasks where there is a single output logit and sigmoid applied to obtain thefinal predictions, HiResCAMs are intrinsically contrastive.

Proof. For $K = 2 , \operatorname { i . e } .$ ., binary classification tasks, one can obtain class probabilities using two setups: ${ \bf ( i ) } \bar { f } \in \mathbb { R } ^ { 2 }$ with softmax σ activation, or (ii): $f ^ { \prime } \in \mathbb { R }$ with sigmoid $\phi$ activation.

Consider ContrastiveCAMs $\mathcal { C } \in \mathbb { R } ^ { 2 \times d _ { 1 } \times d _ { 2 } }$ for softmax activation, and the corresponding HiResCAMs $\mathcal { A } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ for sigmoid activation. We will prove that HiResCAMs in the sigmoid setup are equivalent to ContrastiveCAMs in the softmax setup.

For the softmax case, we apply Proposition 3.7 and obtain class probabilities as:

$$
p _ { f } ( y = 1 | \mathbf { X } ) = \tilde { f } _ { 1 } ( \mathbf { X } ) = \left( \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( 1 , 1 ) } \right) _ { i j } \right) + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( 1 , 2 ) } \right) i j \right) \right) ^ { - 1 } .\tag{44}
$$

$\mathcal { C } _ { ( k , k ) } = \mathbf { 0 } _ { d _ { 1 } \times d _ { 2 } } , \forall k \in [ K ]$ , hence we have:

$$
p _ { f } ( y = 1 | \mathbf { X } ) = \left( 1 + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } ( \mathcal { C } _ { ( 1 , 2 ) } \big ) _ { i j } \right) \right) ^ { - 1 }\tag{45}
$$

$$
= \left( 1 + \exp \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( 2 , 1 ) } \right) _ { i j } \right) \right) ^ { - 1 } ,\tag{46}
$$

and similarly,

$$
p _ { f } ( y = 2 | \mathbf { X } ) = \left( 1 + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \left( \mathcal { C } _ { ( 2 , 1 ) } \right) _ { i j } \right) \right) ^ { - 1 } .\tag{47}
$$

For the sigmoid case, we apply Proposition B.1 and obtain class probabilities as:

$$
p _ { f ^ { \prime } } ( y = 1 | \mathbf { X } ) = 1 - \phi ( f ^ { \prime } ) = \phi ( - f ^ { \prime } ) = \left( 1 + \exp \left( \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } A _ { i j } \right) \right) ^ { - 1 } ,\tag{48}
$$

and

$$
p _ { f ^ { \prime } } ( y = 2 | \mathbf { X } ) = \phi ( f ^ { \prime } ) = \left( 1 + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } A _ { i j } \right) \right) ^ { - 1 } .\tag{49}
$$

Comparing Eq. (46) with Eq. (48) and Eq. (47) with Eq. (49), we observe that HiResCAMs’ connection to the model prediction is identical to that of Contrastive $\mathrm { \Delta A M s , }$ hence HiResCAMs in the binary setting are intrinsically contrastive. □

Proposition B.1 (Correctness of HiResCAMs in Binary Setting). Given an input X, the prediction $\tilde { f } ( { \mathbf { X } } )$ ofa binary classifier is determined by the corresponding HiResCAM:

$$
\tilde { f } ( \mathbf { X } ) = \left( 1 + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \mathcal { A } _ { i j } \right) \right) ^ { - 1 } .\tag{50}
$$

Proof. In the binary classification setting, $f ( \mathbf { X } ) \in \mathbb { R }$ , and $\tilde { f } = \phi ( f )$ , where $\phi$ denotes sigmoid activation. Applying Eq. (2), we have:

$$
\tilde { f } ( \mathbf { X } ) = \phi ( f ) = ( 1 + \exp ( - f ) ) ^ { - 1 } = \left( 1 + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } A _ { i j } \right) \right) ^ { - 1 } .\tag{51}
$$

## B Context-Regularized Binary Cross-Entropy

In the binary classification setting, the neural network outputs a single logit, $f ( \mathbf { X } ) \in \mathbb { R }$ , to which the sigmoid function, denoted $\phi : \mathbb { R }  [ 0 , 1 ]$ , is applied to produce the corresponding probability predictions: $\tilde { f } = \phi ( f )$ . The corresponding class labels are denoted $y \in \{ 0 , 1 \}$ . Additionally, HiResCAMs have shape $\mathcal { A } \in \mathbb { R } ^ { d _ { 1 } \times \bar { d } _ { 2 } }$ , and their connection to logits in Eq. (2) extends to class probabilities. This follows from the bijectivity of sigmoid activation:

Proposition B.1 (Correctness of HiResCAMs in Binary Setting). Given an input X, the prediction $\tilde { f } ( { \mathbf { X } } )$ ofa binary classifier is determined by the corresponding HiResCAM:

$$
\tilde { f } ( \mathbf { X } ) = \left( 1 + \exp \left( - \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } \mathcal { A } _ { i j } \right) \right) ^ { - 1 } .\tag{52}
$$

We leverage this property to apply context regularization in the binary setting:

Definition B.2 (Context-Regularized Binary Cross-Entropy Loss). Given a training dataset D and convolutional network $f ,$ CR-BCE is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C R B C E } } ( \mathcal { D } ) = \mathbb { E } _ { ( \mathbf { X } , y , H ) \sim \mathcal { D } } [ \ell ( \mathbf { X } , y , H ) ] , \quad \mathrm { w i t h } } \\ { \ell ( \mathbf { X } , y , H ) = \log \left( 1 + \exp \left( \displaystyle \sum _ { i , j = 1 } ^ { d _ { 1 } , d _ { 2 } } ( - 1 ) ^ { y } \mathcal { A } _ { i j } H _ { i j } + \underbrace { ( 1 - H _ { i j } ) | \mathcal { A } _ { i j } | } _ { \mathrm { c o n t e x t r e g u l a r i z a t i o n } } \right) \right) . } \end{array}\tag{53}
$$

This definition is motivated by noting that the per-sample binary cross-entropy loss $\ell _ { B C E }$ can be equivalently expressed as:

$$
\ell _ { B C E } ( \mathbf { X } , y ) = - y \log ( \tilde { f } ) - ( 1 - y ) \log ( 1 - \tilde { f } ) = \log ( 1 + \exp ( ( - 1 ) ^ { y } f ) ) ,\tag{54}
$$

when sigmoid activation is used to obtain class probabilities from logits.

Table 3: Spawrious Context-Robustness Benchmark. CR-CE outperforms the standard crossentropy training, as well as GALS and ABN, across all difficulty levels and context shift settings of the Spawrious dataset.
<table><tr><td rowspan="2">Spurious Shift Type</td><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="3">Task Difficulty</td><td rowspan="2">Average (↑)</td></tr><tr><td>Easy (↑)</td><td>Medium (↑)</td><td>Hard (↑)</td></tr><tr><td rowspan="9"> $_ \mathrm { O n e - t o - O n e }$ </td><td></td><td>CE (Baseline)</td><td> $9 3 . 3 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $8 3 . 1 7 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $8 6 . 5 8 { \scriptstyle \pm 1 . 8 8 }$ </td><td>87.69</td></tr><tr><td rowspan="5"> $7 \times 7$ </td><td>GALS</td><td> $9 3 . 4 2 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $8 4 . 3 9 { \scriptstyle \pm 2 . 1 3 }$ </td><td> $8 7 . 1 1 { \scriptstyle \pm 0 . 2 7 }$ </td><td>88.31</td></tr><tr><td>EBL</td><td> $9 3 . 3 1 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $8 3 . 9 1 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $8 6 . 4 4 { \scriptstyle \pm 1 . 4 5 }$ </td><td>87.89</td></tr><tr><td>CFA</td><td> $9 0 . 8 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $7 9 . 4 6 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $8 1 . 7 7 { \scriptstyle \pm 1 . 7 3 }$ </td><td>84.03</td></tr><tr><td>Context Masking</td><td> $9 2 . 3 6 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $8 0 . 9 7 { \scriptstyle \pm 1 . 5 9 }$ </td><td> $8 3 . 6 7 { \scriptstyle \pm 3 . 7 1 }$ </td><td>85.66</td></tr><tr><td>CR-CE (Ours)</td><td> $\mathbf { 9 5 . 4 7 { \scriptstyle \pm 0 . 2 4 } }$ </td><td> $\mathbf { 9 0 . 5 1 } { \scriptstyle \pm 0 . 1 6 }$ </td><td> $\mathbf { 9 1 . 3 5 { \scriptstyle \pm 0 . 3 4 } }$ </td><td>92.44</td></tr><tr><td>CE (Baseline)</td><td> $9 2 . 8 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $8 0 . 3 6 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $8 7 . 7 4 { \scriptstyle \pm 0 . 3 7 }$ </td><td>86.98</td></tr><tr><td rowspan="6"> $1 4 \times 1 4$ </td><td>ABN</td><td> $8 8 . 0 1 { \scriptstyle \pm 1 . 4 9 }$ </td><td> $8 0 . 4 7 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $8 0 . 0 4 { \scriptstyle \pm 2 . 1 3 }$ </td><td>82.84</td></tr><tr><td>GALS</td><td> $9 4 . 7 2 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $8 5 . 7 4 { \scriptstyle \pm 3 . 9 6 }$ </td><td> $9 0 . 5 7 { \scriptstyle \pm 0 . 6 5 }$ </td><td>90.35</td></tr><tr><td>EBL</td><td> $9 4 . 2 0 { \scriptstyle \pm 0 . 2 8 } $ </td><td> $8 2 . 2 0 { \scriptstyle \pm 3 . 3 3 }$ </td><td> $8 7 . 5 9 { \scriptstyle \pm 1 . 5 4 }$ </td><td>88.00</td></tr><tr><td>CFA</td><td> $8 8 . 9 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 3 . 3 8 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $7 5 . 8 8 { \scriptstyle \pm 1 . 1 2 }$ </td><td>79.41</td></tr><tr><td>Context Masking</td><td> $9 3 . 0 7 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $8 1 . 7 9 { \scriptstyle \pm 2 . 2 9 }$ </td><td> $8 3 . 9 1 { \scriptstyle \pm 1 . 9 5 }$ </td><td>86.26</td></tr><tr><td>CR-CE (Ours)</td><td> $\mathbf { 9 6 . 3 5 { \scriptstyle \pm 0 . 1 5 } }$ </td><td> $\mathbf { 9 2 . 2 4 { \scriptstyle \pm 0 . 3 0 } }$ </td><td> $\mathbf { 9 2 . 8 9 { \scriptstyle \pm 0 . 2 0 } }$ </td><td>93.83</td></tr><tr><td rowspan="9">Many-to-Many</td><td rowspan="5"> $7 \times 7$ </td><td>CE (Baseline)</td><td> $9 0 . 8 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $7 9 . 4 6 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $8 1 . 5 2 { \scriptstyle \pm 1 . 0 3 }$ </td><td>83.95</td></tr><tr><td>GALS</td><td> $9 3 . 4 7 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $8 4 . 5 8 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $7 8 . 7 4 { \scriptstyle \pm 0 . 9 7 }$ </td><td>85.60</td></tr><tr><td>EBL</td><td> $9 2 . 4 3 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $8 7 . 2 3 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $8 1 . 2 1 { \scriptstyle \pm 0 . 7 4 }$ </td><td>86.96</td></tr><tr><td>CFA</td><td> $8 9 . 1 4 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $8 5 . 6 8 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $8 1 . 7 7 { \scriptstyle \pm 1 . 7 3 }$ </td><td>85.65</td></tr><tr><td>Context Masking</td><td> $9 2 . 0 8 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $8 5 . 0 3 { \scriptstyle \pm 2 . 3 7 }$ </td><td> $7 8 . 4 6 { \scriptstyle \pm 1 . 1 2 }$ </td><td>85.19</td></tr><tr><td>CR-CE (Ours)</td><td> $\mathbf { 9 5 . 5 5 } { \scriptstyle \pm 0 . 1 6 }$ </td><td> $\mathbf { 9 1 . 2 5 { \scriptstyle \pm 0 . 1 8 } }$ </td><td> $\mathbf { 8 7 . 4 9 } { \scriptstyle \pm 0 . 1 4 }$ </td><td>91.43</td></tr><tr><td>CE (Baseline)</td><td> $8 8 . 9 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 3 . 3 8 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $7 9 . 7 6 { \scriptstyle \pm 1 . 0 1 }$ </td><td>80.71</td></tr><tr><td rowspan="5"> $1 4 \times 1 4$ </td><td>ABN</td><td> $8 9 . 6 5 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $7 2 . 8 1 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $6 3 . 9 6 { \scriptstyle \pm 6 . 5 4 }$ </td><td>75.47</td></tr><tr><td>GALS</td><td> $9 5 . 0 3 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $8 7 . 6 3 { \scriptstyle \pm 3 . 8 0 }$ </td><td> $8 4 . 0 8 { \scriptstyle \pm 1 . 8 7 }$ </td><td>88.91</td></tr><tr><td>EBL</td><td> $9 3 . 4 6 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $8 9 . 0 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $7 9 . 3 6 { \scriptstyle \pm 2 . 8 3 }$ </td><td>87.29</td></tr><tr><td>CFA</td><td> $8 7 . 2 2 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $8 5 . 6 8 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $7 5 . 8 8 { \scriptstyle \pm 1 . 1 2 }$ </td><td>82.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Context Masking CR-CE (Ours)</td><td></td><td> $9 0 . 9 6 { \scriptstyle \pm 0 . 2 4 }$   $\mathbf { 9 6 . 4 7 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> $8 5 . 7 0 { \scriptstyle \pm 2 . 9 9 }$   $\mathbf { 9 2 . 4 1 { \scriptstyle \pm 0 . 2 9 } }$ </td><td> $8 0 . 8 0 { \scriptstyle \pm 0 . 7 1 }$   $\mathbf { 8 8 . 6 6 } \pm \mathrm { 0 . 5 2 }$ </td><td>85.82 92.51</td></tr></table>

Table 4: Hard-ImageNet Benchmark. Predictions on models trained with CR-CE are least likely to persist when the object is absent from the image. Object-focus increases proportional to the CAM resolution, with some performance cost incurred when the target is present alongside the context.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td rowspan="2">No Shift Accuracy (↑)</td><td colspan="3">Accuracy under Target Removal</td><td colspan="2">Rel. Foreground Sensitivity</td></tr><tr><td> $\mathrm { G r a y } \ \mathbf { M a s k } \ ( \downarrow )$ </td><td> $\mathrm { G r a y ~ B B O X } \left( \downarrow \right)$ </td><td> $\mathrm { T i l e } \left( \downarrow \right)$ </td><td> $\rho = 0 . 2 5 \left( \uparrow \right)$ </td><td> $\rho = 0 . 5 \ : ( \uparrow )$ </td></tr><tr><td rowspan="5"> $7 \times 7$ </td><td>CE (Baseline)</td><td> $9 5 . 8 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 2 . 4 { \pm } 0 . 1 3$ </td><td> $7 9 . 3 8 { \pm } 0 . 2 $ </td><td> $7 7 . 1 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $- 0 . 2 1 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $- 0 . 3 6 { \pm } 0 . 0 1$ </td></tr><tr><td>GALS</td><td> $9 5 . 9 1 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $8 0 . 9 3 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $7 6 . 6 2 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $7 5 . 6 9 2 0 . 7 3$ </td><td> $- 0 . 1 8 { \pm } 0 . 0 1$ </td><td> $- 0 . 3 1 { \scriptstyle \pm 0 . 0 3 }$ </td></tr><tr><td>EBL</td><td> $9 5 . 8 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $7 7 . 6 4 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $7 3 . 2 0 { \scriptstyle \pm 2 . 0 7 }$ </td><td> $7 0 . 7 4 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $- 0 . 1 1 { \pm } 0 . 0 4$ </td><td> $- 0 . 1 9 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>CFA</td><td> $9 6 . 0 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $8 2 . 1 3 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $7 9 . 2 4 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 6 . 6 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $- 0 . 2 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $- 0 . 3 5 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $9 1 . 4 2 { \scriptstyle \pm 0 . 2 8 } $ </td><td> $\mathbf { 5 0 . 0 9 } { \scriptstyle \pm 0 . 8 9 }$ </td><td> $\mathbf { 4 5 . 7 8 { \scriptstyle \pm 0 . 4 1 } }$ </td><td> ${ \bf 4 6 . 9 8 { \scriptstyle \pm 0 . 2 } }$ </td><td> $\mathbf { + 0 . 1 0 { \scriptstyle \pm 0 . 0 2 } }$ </td><td> $\mathbf { + 0 . 1 6 { \scriptstyle \pm 0 . 0 1 } }$ </td></tr><tr><td rowspan="6"> $1 4 \times 1 4$ </td><td>CE (Baseline)</td><td> $9 5 . 8 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 1 . 2 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $7 8 . 3 6 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $7 7 . 6 4 \pm 0 . 2 $ </td><td> $- 0 . 2 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $- 0 . 3 4 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>ABN</td><td> $9 5 . 2 4 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $8 0 . 8 4 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 3 . 4 7 { \scriptstyle \pm 1 . 6 2 }$ </td><td> $7 3 . 3 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $- 0 . 3 4 { \pm } 0 . 0 2$ </td><td> $- 0 . 4 4 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>GALS</td><td> $9 5 . 8 2 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 8 . 0 0 { \scriptstyle \pm 3 . 5 4 }$ </td><td> $7 4 . 2 7 { \scriptstyle \pm 4 . 3 5 }$ </td><td> $7 3 . 8 2 { \scriptstyle \pm 3 . 7 8 }$ </td><td> $- 0 . 1 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $- 0 . 2 3 { \scriptstyle \pm 0 . 1 4 }$ </td></tr><tr><td>EBL</td><td> $9 5 . 2 4 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 0 . 3 1 { \scriptstyle \pm 4 . 4 6 }$ </td><td> $6 7 . 1 6 { \scriptstyle \pm 3 . 4 5 }$ </td><td> $6 5 . 9 1 { \scriptstyle \pm 5 . 3 3 }$ </td><td> $- 0 . 0 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $- 0 . 0 5 { \scriptstyle \pm 0 . 1 4 }$ </td></tr><tr><td>CFA</td><td> $9 6 . 0 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $8 1 . 4 2 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 8 . 2 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $7 6 . 7 6 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $- 0 . 1 8 { \pm } 0 . 0 2$ </td><td> $- 0 . 3 6 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $8 8 . 5 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $\mathbf { 3 7 . 0 7 } \pm \mathbf { 0 . 4 8 }$ </td><td> $\mathbf { 3 2 . 4 4 } { \scriptstyle \pm 0 . 1 5 }$ </td><td> $\mathbf { 3 2 . 7 6 { \scriptstyle \pm 0 . 7 8 } }$ </td><td> $\mathbf { + 0 . 1 7 { \scriptstyle \pm 0 . 0 1 } }$ </td><td> $\mathbf { + 0 . 2 3 { \scriptstyle \pm 0 . 0 1 } }$ </td></tr></table>

## C Additional Experimental Results

## C.1 Spawrious Benchmark on Various Context-Shift Configurations

The Spawrious dataset comprises of four dog breeds spread across six contexts, with three difficulty levels based on the object-context combinations available during training, and two different configurations of context shift: (i) one-to-one shifts, where each class is correlated with one context, and (ii) many-to-many shifts, where groups of classes are correlated with groups of contexts, the latter being observed broadly in practice [23]. We include results on both one-to-one and many-to-many shifts in Table 3. Consistent with our findings in Section 5.3, we observe that CR-CE trained models outperform the baseline and existing explanation-guided approaches in context-shifted accuracy across all difficulty levels.

Table 5: Context-Contribution Ratios. We report the context contribution ratios across each of the assessed approaches. CR-CE trained models place a greater emphasis on object-specific regions to determine predictions and learn to ignore cues from context regions through the process of training.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="4">Context Contribution Ratio R% (↓)</td></tr><tr><td>Waterbirds-100</td><td> $\operatorname { S p a w r i o u s } ^ { \ddagger }$ </td><td>ImageNet-BG</td><td>Hard-ImageNet</td></tr><tr><td rowspan="6"> $7 \times 7$ </td><td>CE (Baseline)</td><td> $4 6 . 0 8 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $4 4 . 7 9 2 0 . 3 $ </td><td> $3 1 . 8 3 ^ { \dagger }$ </td><td> $6 8 . 5 3 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td>GALS</td><td> $3 6 . 2 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $2 8 . 0 6 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $2 9 . 7 5 { \scriptstyle \pm 0 . 5 }$ </td><td> $6 3 . 6 4 { \scriptstyle \pm 2 . 0 3 }$ </td></tr><tr><td>EBL</td><td> $5 6 . 0 2 { \scriptstyle \pm 5 . 1 7 }$ </td><td> $2 1 . 3 7 { \scriptstyle \pm 1 . 4 5 }$ </td><td> $2 5 . 3 5 { \scriptstyle \pm 2 . 0 6 }$ </td><td> $5 7 . 2 4 { \scriptstyle \pm 3 . 9 1 }$ </td></tr><tr><td>CFA</td><td> $1 7 . 0 6 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $4 3 . 1 { \scriptstyle \pm 2 . 6 9 }$ </td><td> $3 2 . 4 9 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $5 7 . 2 2 { \scriptstyle \pm 0 . 5 0 }$ </td></tr><tr><td>Context Masking</td><td> $2 2 . 4 8 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $4 8 . 1 1 { \scriptstyle \pm 2 . 2 2 }$ </td><td> $3 2 . 8 6 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $7 0 . 6 3 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $\mathbf { 1 2 . 8 6 { \scriptstyle \pm 0 . 0 9 } }$ </td><td> $\mathbf { 1 4 . 8 9 { \scriptstyle \pm 0 . 2 } }$ </td><td> $\mathbf { 1 4 . 8 5 { \scriptstyle \pm 0 . 0 0 } }$ </td><td> $\mathbf { 3 5 . 8 5 { \scriptstyle \pm 0 . 2 6 } }$ </td></tr><tr><td rowspan="6"> $1 4 \times 1 4$ </td><td>CE (Baseline)</td><td> $5 8 . 7 5 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $4 7 . 3 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $2 8 . 2 7 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $6 6 . 5 { \scriptstyle \pm 0 . 0 7 }$ </td></tr><tr><td>ABN</td><td> $3 4 . 9 2 { \scriptstyle \pm 1 9 . 3 8 }$ </td><td> $4 9 . 8 9 { \scriptstyle \pm 1 6 . 9 9 }$ </td><td> $3 3 . 3 7 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $6 9 . 0 6 { \scriptstyle \pm 4 . 0 2 }$ </td></tr><tr><td>GALS</td><td> $3 5 . 7 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $1 7 . 7 7 { \scriptstyle \pm 2 . 0 7 }$ </td><td> $2 4 . 1 4 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $5 6 . 8 7 { \scriptstyle \pm 9 . 3 7 }$ </td></tr><tr><td>EBL</td><td> $4 8 . 2 1 { \scriptstyle \pm 2 . 4 4 }$ </td><td> $2 9 . 8 2 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $2 2 . 9 4 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $4 8 . 6 3 { \scriptstyle \pm 9 . 6 8 }$ </td></tr><tr><td>CFA</td><td> $1 6 . 0 8 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $4 6 . 6 6 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $2 8 . 4 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $4 7 . 8 9 { \scriptstyle \pm 1 . 5 6 }$ </td></tr><tr><td>Context Masking</td><td> $2 5 . 2 4 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $3 8 . 4 4 { \scriptstyle \pm 1 . 9 1 }$ </td><td> $2 7 . 6 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $6 9 . 3 5 { \scriptstyle \pm 0 . 2 }$ </td></tr><tr><td></td><td>CR-CE (Ours)</td><td> $\mathbf { 9 . 5 4 { \scriptstyle \pm 0 . 2 8 } }$ </td><td> $\mathbf { 1 0 . 9 8 { \scriptstyle \pm 0 . 1 2 } }$ </td><td> ${ \bf 9 . 3 2 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> $\mathbf { 3 1 . 1 2 } _ { \pm 0 . 1 }$ </td></tr></table>

No standard deviation is reported; we use pre-trained weights from torchvision without modification.  
‡ We report metrics on the hardest subtask and difficulty (M2M-hard) within the Spawrious benchmark.

## C.2 Evaluating Context Dependence using Hard-ImageNet

The Hard-ImageNet benchmark [25] assesses model reliance on context by comparing performance on images with and without the target object. Ideally, the model should predict the true class in the presence of the object, and a random class in the absence of the object. In other words, the prediction should not be based on the context alone. The Hard-ImageNet suite also reports Relative Foreground Sensitivity (RFS) [24], which compares accuracy reduction when gaussian noise (with magnitude denoted $\rho )$ is added to either the object or the context region; higher RFS signifies greater object-dependence.

Evaluation of CR-CE on the Hard-ImageNet and comparison with other methods are reported in Table 4. We observe that the CR-CE framework meaningfully outperforms each of the compared methods on key robustness metrics assessed by the Hard-ImageNet benchmark. Example images of Hard-ImageNet demonstrating the effectiveness of CR-CE on suppressing context dependence can be find in Figure 3 and Figure 5, see the classes Baseball Player, Snorkel, and Howler Monkey.

Remark C.1. Hard-ImageNet includes a lot of context-heavy images where object regions comprise only 13.96% of the image on average, leading to a bigger reduction of accuracy in the accuracyrobustness tradeoff.

## C.3 Comparison with Explanation-Guided Learning (EGL) Methods

Typically, explanation-guided learning frameworks [14] construct a multi-task optimization objective with a per-sample loss of the following form:

$$
\ell _ { \mathrm { E G L } } = \ell _ { \mathrm { C E } } + \lambda \cdot \ell _ { \mathrm { E G } } , \quad \mathrm { w i t h ~ g r a d i e n t } \quad \nabla \ell _ { \mathrm { E G L } } = \nabla \ell _ { \mathrm { C E } } + \lambda \nabla \ell _ { \mathrm { E G } } .\tag{55}
$$

Here, $\ell _ { \mathrm { E G } }$ is a secondary explanation-guided goal designed to suppress context contributions. The specific form of this term varies between approaches [38, 34, 20, 29, 11].

For a given $( i , j )$ -patch corresponding to context, gradients of the form in Eq. (55) create a tug-of-war dynamic: the first term seeks to minimize $\ell _ { \mathrm { C E } }$ at the cost of $\ell _ { \mathrm { E G } }$ , and vice versa. To manage this, the additional hyperparameter λ is introduced and must be carefully tuned to prevent poor classification accuracy. In contrast, CR-CE replaces the mechanism in cross-entropy that increases contributions from context regions, preventing competing gradients (taken w.r.t. the specific patch), hence resulting in a more pronounced object-focus effect.

![](images/9d2acb3c303c576b7a2c35e971dda24ed59980f411c8984d73896c009a6f1375.jpg)  
Figure 4: Superficially Aligned Attribution Maps. On a ResNet-50 trained with the GALS framework, we observe that HiResCAMs suggest a focus on the object regions; however, the model predicts the context-correlated class observed during training. ContrastiveCAMs faithfully reveal that the context contribution is in fact the primary driver of the prediction. This highlights the essential role that explanation faithfulness plays in ensuring improvements to context robustness in the explanation-guided learning paradigm.

The advantage of this is evident in the Waterbirds-100 dataset. Since it is a binary classification task, the GradCAMs used by GALS coincide with ContrastiveCAMs used by CR-CE. The only difference between the two approaches lies in how the feature attribution method is incorporated in the model training procedure. In this direct comparison, CR-CE continues to outperform GALS, see Table 2.

Explanation Correctness is Key to Robustness Improvements. In Table 2, we observe that some prior explanation-guided methods may even have an inferior performance than the CE training baselines, especially when the tasks become complicated. To investigate this performance drift, we measure the context-contribution ratios for those approaches in Table 5. We observe that models trained by those explanation-guided methods often have greater context dependence, indicating the ineffectiveness of these methods in guiding the feature attributions towards object regions. We hypothesize that this behavior is largely due to their reliance on classical feature attribution method (e.g. Grad-CAM and HiResCAM), which suffer from the unfaithfulness issue theoretically described in Theorem 3.2. Figure 4 shows typical examples where prior explanation-guided methods may be successful in guiding those classical CAMs to focus on the object regions but fail in doing so for the more faithful ContrastiveCAMs.

## C.4 CR-CE is Tolerant to Imperfect Object-Context Segmentation

In our experiments with Spawrious and ImageNet datasets in Table 2, we already used AI-predicted object-context segmentation, obtained from SAM3 [5] and MAEs [18], which are approximations and are inevitably imperfect. Here, we further assess CR-CE’s tolerance to noise by adding additional noise to these object-context masks. Specifically, we perform random affine transformations (up to 5<sup>◦</sup> of rotation, ±7.5% translation, and ±30% scaling) on each of those masks H. This augmentation results in ≈ 50% mIoU of the original object-context masks to their corrupted counterparts. We train CR-CE using these corrupted masks, and report our results in Table 6. We can see that, even with these noisy object-context segmentation, CR-CE can still predict well with minor degradation in the performance.

## C.5 CR-CE is Effective Across Architectures and Scales

Here we provide additional experimental validations of the CR-CE training framework on different network architectures, including EfficientNetV2 [42] and ResNet-18. The results are reported in Table 7 and 8 respectively, demonstrate the effectiveness of CR-CE across variations in architectures and scales.

Table 6: Context-Shift Robustness using Noisy Masks (NM). Despite the additional noise introduced to object-context masks, CR-CE remains effective at improving context-shift robustness. CR-CE retains its utility in scenarios where accurate masks are challenging or expensive to obtain.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="2">Waterbirds-100</td><td colspan="2">ImageNet/ImageNet-BG</td><td colspan="3">Spawrious</td></tr><tr><td>No Shift (↑)</td><td>Shifted (↑)</td><td>No Shift (↑)</td><td>Shifted (↑)</td><td>Easy (↑)</td><td>Medium (↑)</td><td>Hard (↑)</td></tr><tr><td rowspan="3">7×7</td><td>CE (Baseline)</td><td>98.66±0.10</td><td>41.74±4.15</td><td>82.92†</td><td>74.83†</td><td>90.86±0.54</td><td>79.46±0.48</td><td>81.52±1.03</td></tr><tr><td>CR-CE (Ours) + NM</td><td>92.10±0.22</td><td>72.61±1.26</td><td>81.14±0.02</td><td>74.89±0.07</td><td>93.07±0.32</td><td>87.75±0.42</td><td>82.19±0.16</td></tr><tr><td>CR-CE (Ours)</td><td>94.38±1.21</td><td>73.30±4.43</td><td>81.85±0.03</td><td>75.75±0.08</td><td>95.55±0.16</td><td>91.25±0.18</td><td>87.49±0.14</td></tr><tr><td rowspan="3">14 × 14</td><td>CE (Baseline)</td><td>98.48±0.09</td><td>38.40±1.79</td><td>82.33±0.10</td><td>74.72±0.06</td><td>88.98±0.14</td><td>73.38±0.87</td><td>79.76±1.01</td></tr><tr><td>CR-CE (Ours) + NM</td><td>90.67±2.42</td><td>76.12±3.24</td><td>79.83±0.06</td><td>75.24±0.02</td><td>93.07±0.51</td><td>85.64±0.17</td><td>80.59±0.84</td></tr><tr><td>CR-CE (Ours)</td><td>94.36±0.12</td><td>79.13±0.77</td><td>81.25±0.03</td><td>76.84±0.12</td><td>96.47±0.06</td><td>92.41±0.29</td><td>88.66±0.52</td></tr></table>

<sup>†</sup> No std. dev. is reported; we use pre-trained weights from torchvision without modification.

Table 7: Context-Shift Robustness On EfficientNetV2. CR-CE improves context-shift robustness on varied convolutional backbones. Following EfficientNetV2’s [42] training recipe, we adopt a larger batch size of 4096 and increase the learning rate to 10<sup>−2</sup> for ImageNet training.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="2">Waterbirds-100</td><td colspan="2">ImageNet/ImageNet-BG</td><td colspan="3">Spawrious</td></tr><tr><td>No Shift (↑)</td><td>Shifted (↑)</td><td>No Shift (↑)</td><td>Shifted (↑)</td><td>Easy (↑)</td><td>Medium (↑)</td><td>Hard (↑)</td></tr><tr><td rowspan="2">7×7</td><td>CE (Baseline)</td><td>91.08±0.05</td><td>39.31±3.01</td><td>83.71†</td><td>73.58†</td><td>92.62±0.41</td><td>81.49±0.29</td><td>77.50±0.91</td></tr><tr><td>CR-CE (Ours)</td><td>94.31±1.93</td><td>78.55±5.01</td><td>83.36±0.16</td><td>74.46±0.08</td><td>93.86±0.62</td><td>84.23±0.67</td><td>88.29±0.22</td></tr><tr><td rowspan="2">14 × 14</td><td>CE (Baseline)</td><td>84.64±0.14</td><td>38.88±1.31</td><td>84.00±0.07</td><td>74.53±0.07</td><td>93.11±0.42</td><td>80.27±2.06</td><td>79.30±1.17</td></tr><tr><td>CR-CE (Ours)</td><td>94.02±1.15</td><td>80.43±5.01</td><td>83.67±0.03</td><td>75.20±0.04</td><td>94.84±0.33</td><td>86.98±0.04</td><td>88.85±0.55</td></tr></table>

<sup>†</sup> No std. dev. is reported; we use pre-trained weights from torchvision without modification.

## D About Data Augmentation with Object-Context Mixing

A straightforward yet often effective approach to improving robustness to context shifts is to augment the training dataset with synthetic images that combine objects with diverse contexts [12]. This approach is orthogonal to the methods considered in this paper, including explanation-guided methods, as it leaves the training procedure unchanged and simply introduces additional augmented training samples. Therefore, it can be readily combined with our proposed method. However, its effectiveness may be limited by the finite set of object–context combinations observed during training, potentially leaving the model vulnerable to unseen context shifts at evaluation time, particularly adversarial ones. In contrast, our proposed training framework suppresses the influence of context regions, not limited by the artificial object–context combinations seen during training, potentially providing greater robustness to out-of-distribution scenarios.

Nevertheless, we empirically evaluate this approach by augmenting each training image with an additional image in which the object is paired with a context uniformly sampled from the nontarget classes, thereby doubling the size of the training dataset. We report the effect of pairing this augmentation with both CE and CR-CE in Table 9. We can see that the data augmentation technique performs well in simple tasks, such as Waterbirds, but becomes worse when the task becomes harder.

## E Training Details

Hyperparameter Settings. For all methods, we use Momentum SGD [32] with a momentum parameter 0.9. The hyperparameter settings shown in Table 10 are applied across all methods. GALS [29], ABN [11] and Energy loss [34] each have an additional hyperparameter λ which weights the regularization strength of spatial feature attributions, which we keep at their default value of 1.

Computational Resources. We train each of our models using a single NVIDIA H100 GPU. The most efficient loss function, cross-entropy, took a maximum of 5.5 hours to train ResNet-50 on ImageNet, and a minimum of 20 minutes on the Waterbirds dataset. Training with CR-CE took a maximum of 6.1 hours per ResNet-50 run on the ImageNet dataset, and a minimum of 20 minutes on the Waterbirds dataset. Each of the other methods approximately took between 8 and 12 hours to train a ResNet-50 on ImageNet. Each of the methods used a maximum of 65 GB of VRAM during training on the ImageNet dataset, with gradient accumulation applied as necessary to stay within the VRAM limit of the GPU. We trained a total of 582 models: 3 independent runs for each approach and backbone on 9 different training sets (6 for Spawrious, 1 each for Waterbirds, ImageNet-BG, and Hard-ImageNet), with 2 CAM resolutions for all but ABNs. ABN increases the CAM resolution as part of its architecture design; we do not reduce it to $7 \times 7 .$

Table 8: Context-Shift Robustness On ResNet-18. CR-CE improves context-shift robustness on the smaller ResNet-18 backbone. We observe a more pronounced trade-off between in-distribution and context-shifted accuracy due to the smaller size of the backbone. In contrast, this trade-off is minimized by the larger capacity of the EfficientNetV2 backbone, see Table 7.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="2">Waterbirds-100</td><td colspan="2">ImageNet/ImageNet-BG</td><td colspan="3">Spawrious</td></tr><tr><td>No Shift (↑)</td><td>Shifted (↑)</td><td>No Shift (↑)</td><td>Shifted (↑)</td><td>Easy (↑)</td><td>Medium (↑)</td><td>Hard (↑)</td></tr><tr><td rowspan="2"> $7 \times 7$ </td><td>CE (Baseline)</td><td> $7 9 . 9 6 { \scriptstyle \pm 1 2 . 0 3 }$ </td><td> $3 8 . 0 0 { \scriptstyle \pm 3 . 7 9 }$ </td><td>73.49†</td><td> $5 8 . 3 8 ^ { \dagger }$ </td><td> $8 0 . 8 9 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $5 9 . 3 4 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 2 . 1 6 { \scriptstyle \pm 1 . 2 7 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $8 8 . 4 8 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $\mathbf { 5 3 . 5 5 } { \scriptstyle \pm 1 0 . 4 4 }$ </td><td> $7 2 . 4 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> ${ \bf 6 2 . 6 1 { \scriptstyle \pm 0 . 0 2 } }$ </td><td>90.77±0.07</td><td>75.93±0.56</td><td>68.99±0.29</td></tr><tr><td rowspan="2"> $1 4 \times 1 4$ </td><td>CE (Baseline)</td><td> $9 7 . 5 9 { \pm } 0 . 3 6$ </td><td> $3 0 . 4 1 { \scriptstyle \pm 8 . 9 0 }$ </td><td> $7 5 . 0 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 0 . 4 3 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $7 4 . 8 4 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $4 6 . 1 4 { \scriptstyle \pm 2 . 4 5 }$ </td><td> $3 7 . 8 1 { \scriptstyle \pm 1 . 1 4 }$ </td></tr><tr><td>CR-CE (Ours)</td><td> $8 8 . 2 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $\mathbf { 5 9 . 9 5 } { \scriptstyle \pm 7 . 9 7 }$ </td><td>71.44±0.06</td><td> ${ \bf 6 3 . 3 7 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $\mathbf { 9 1 . 6 4 } { \scriptstyle \pm 0 . 2 6 }$ </td><td> $\mathbf { 7 9 . 4 2 { \scriptstyle \pm 0 . 2 3 } }$ </td><td> $\mathbf { 7 0 . 6 8 { \scriptstyle \pm 1 . 4 4 } }$ </td></tr></table>

<sup>†</sup> No std. dev. is reported; we use pre-trained weights from torchvision without modification.

Table 9: Context-Shift Robustness using Object-Context Augmentation. Generating objectcontext combinations using copy-paste augmentation only improves robustness in the Waterbirds setting, when object-context augmentation simulates the test set. In larger and more realistic contextshift scenarios, augmenting contexts does not produce enough signal to suppress context contributions.
<table><tr><td rowspan="2">CAM Size</td><td rowspan="2">Approach</td><td colspan="2">Waterbirds-100</td><td colspan="2">ImageNet/ImageNet-BG</td><td colspan="3">Spawrious</td></tr><tr><td>No Shift (↑)</td><td>Shifted (↑)</td><td>No Shift (↑)</td><td>Shifted (↑)</td><td>Easy (↑)</td><td>Medium (↑)</td><td>Hard (↑)</td></tr><tr><td rowspan="4">7×7</td><td>CE (Baseline)</td><td>98.66±0.10</td><td> $4 1 . 7 4 { \scriptstyle \pm 4 . 1 5 }$ </td><td>82.92†</td><td>74.83†</td><td> $9 0 . 8 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td>79.46±0.48</td><td> $8 1 . 5 2 { \scriptstyle \pm 1 . 0 3 }$ </td></tr><tr><td>CE + OCM</td><td>96.01±0.11</td><td> $9 0 . 6 4 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $8 0 . 3 8 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 2 . 8 0 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $7 9 . 6 5 { \scriptstyle \pm 0 . 3 9 }$ </td><td>49.15±2.01</td><td>42.99±1.01</td></tr><tr><td>CR-CE (Ours)</td><td> $9 4 . 3 8 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $7 3 . 3 0 { \scriptstyle \pm 4 . 4 3 }$ </td><td> $8 1 . 8 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 5 . 7 5 \pm 0 . 0 8$ </td><td> $\mathbf { 9 5 . 5 5 { \scriptstyle \pm 0 . 1 6 } }$ </td><td>91.25±0.18</td><td> $\mathbf { 8 7 . 4 9 } \pm \mathbf { 0 . 1 4 }$ </td></tr><tr><td>CR-CE (Ours) + OCM</td><td>95.35±1.02</td><td> $\mathbf { 9 2 . 6 3 { \scriptstyle \pm 0 . 9 7 } }$ </td><td> $7 9 . 5 2 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 3 . 7 9 2 0 . 0 1$ </td><td> $9 4 . 4 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td>87.95±0.61</td><td>76.96±0.60</td></tr><tr><td rowspan="4"> $1 4 \times 1 4$ </td><td>CE (Baseline)</td><td>98.48±0.09</td><td> $3 8 . 4 0 { \scriptstyle \pm 1 . 7 9 }$ </td><td> $8 2 . 3 3 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $7 4 . 7 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $8 8 . 9 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td>73.38±0.87</td><td> $7 9 . 7 6 { \scriptstyle \pm 1 . 0 1 }$ </td></tr><tr><td>CE + OCM</td><td>95.51±0.22</td><td> $9 2 . 7 3 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $8 1 . 0 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 3 . 7 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $7 0 . 6 9 { \scriptstyle \pm 2 . 4 7 }$ </td><td>40.84±0.74</td><td>32.17±5.06</td></tr><tr><td>CR-CE (Ours)</td><td> $9 4 . 3 6 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $7 9 . 1 3 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $8 1 . 2 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 6 . 8 4 \pm 0 . 1 2$ </td><td>96  $\bf { 4 7 \pm 0 . 0 6 }$ </td><td>92.41±0.29</td><td>88.66±0.52</td></tr><tr><td>CR-CE (Ours) + OCM</td><td> $9 4 . 8 8 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $\mathbf { 9 3 . 3 6 { \scriptstyle \pm 0 . 8 8 } }$ </td><td> $7 8 . 8 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 4 . 7 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $9 5 . 3 0 { \scriptstyle \pm 0 . 2 4 }$ </td><td>88.55±0.36</td><td> $7 7 . 5 1 { \scriptstyle \pm 0 . 4 1 }$ </td></tr></table>

No std. dev. is reported; we use pre-trained weights from torchvision without modification.

Table 10: Hyperparameters are dataset-specific and kept consistent across approaches. Hard-ImageNet and ImageNet-1k are trained for fewer epochs since we fine-tune ImageNet-trained models. We increase the batch size as number of classes (2, 4, 15, and 1000 respectively) increases.
<table><tr><td>Dataset</td><td>Epochs</td><td>Batch Size</td><td>LR</td><td>Momentum</td><td>Weight Decay</td><td>Scheduler</td></tr><tr><td>Waterbirds-100</td><td>150</td><td>64</td><td> $1 0 ^ { - 3 }$ </td><td rowspan="4">0.9</td><td> $1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Spawrious</td><td>150</td><td>128</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Hard-ImageNet</td><td>100</td><td>256</td><td> $5 \cdot 1 0 ^ { - 3 } ~ $ </td><td> $1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>ImageNet-1k / ImageNet-BG</td><td>20</td><td>512</td><td> $5 \cdot 1 0 ^ { - 4 } $ </td><td> $1 0 ^ { - 5 }$ </td><td>Cosine Annealing</td></tr></table>

Reproducibility. The source code for this project can be accessed at https://codeberg.org/ jinensetpal/context-regularized-cross-entropy/.

Asset Licenses. We used the following existing assets, listed with their respective licenses:

• The Waterbirds dataset [37] is licensed under MIT.

• The Spawrious dataset [23] is licensed under CC0-1.0.

• The ImageNet dataset [8] is licensed only for non-commercial purposes.

• The ImageNet-BG dataset [30] is published under a public domain license.

• The Hard-ImageNet dataset [25] is published under CC-0.

• Model weights for the Masked AutoEncoder (MAE) [18] trained on the Large-Scale Unsupervised Segmentation (LUSS) [13] dataset is licensed under CC BY-NC 4.0.

• Model weights for ABNs [11] is licensed under MIT.

• SAM3 [5] is published under a custom SAM License which permits the academic use of their tool.

• The torchvision library is published under a BSD 3-Clause License.

![](images/3d813faa68449675460ccffc0501ed3d9ef421a80fd139b7f810888e495c9008.jpg)  
Figure 5: ImageNet-BG and Hard-ImageNet. Extended qualitative evaluations of object-context dependence by cross-entropy and CR-CE trained models. Images belonging to the Snorkel and Howler Monkey classes correspond to the Hard-ImageNet validation set, whereas the rest correspond to the ImageNet-BG test set. Across each of the examples, we observe the context dependence in models trained using cross-entropy loss is significantly mitigated by training with CR-CE.