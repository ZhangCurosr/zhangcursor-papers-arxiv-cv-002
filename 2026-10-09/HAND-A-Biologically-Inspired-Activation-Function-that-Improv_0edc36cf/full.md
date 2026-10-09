# HAND: A Biologically-Inspired Activation Function that Improves Generalisation and Sample Efficiency in Image Classification

Michael W. Spratling

michael.spratling@uni.lu

Heiko H. Schutt¨

heiko.schutt@uni.lu

University of Luxembourg, Department of Behavioural and Cognitive Sciences, L-4366 Esch-sur-Alzette, Luxembourg.

## Abstract

DNNs exhibit robustness and generalisation issues not seen in humans. They are also far less data-efficient learners, requiring considerably more training samples to accurately classify novel exemplars. Inductive bias could help with these issues by providing in-built mechanisms to improve generalisation, and hence, reduce reliance on learning from data. We incorporate a biologically-inspired inductive bias into a new activation function, HAND (Homeostasis, Accelerating Nonlinearity, and Divisive-nomalisation), and show its effectiveness with CNNs trained on image classification. Using HAND a ConvNeXt-tiny required 25 training epochs to reach the same accuracy on ImageNet1k as the unmodified model achieved after 200 epochs. Consistent with the effects of an inductive bias, the performance gap reduced with training time and increased data augmentation. When the volume of training data was reduced and unevenly distributed between classes (Long-tailed ImageNet) the improvements in accuracy were even larger and did not reduce with increased training time. Generalisation performance with the common-corruptions data, and the ability to reject samples from unknown classes, were uneffected or improved by HAND. Results generalised across CNN architectures and training data-sets. HAND can, therefore, reduce the required training time and/or the required volume and variety of training data, helping to improve sample efficiency.

Keywords: generalisation, data efficiency, inductive bias, competition, class imbalance, robustness, divisive-normalisation, image classification, convolutional neural networks

Code: https://codeberg.org/mwspratling/HAND

## 1 Introduction

A deep neural network (DNN) of sufficient size can approximate any arbitrary mapping from a set of inputs to a set of outputs (Cybenko., 1989; Hornik, 1991; Hornik et al., 1989; Kidger & Lyons, 2020; Zhang et al., 2017). Hence, given sufficient training samples and time to search for the weights that produce the desired mapping, any image classification problem can be learnt without the need for strong, built-in, inductive biases. Inductive bias may, therefore, have limited influence on the final, optimised, solution learned from large data-sets. However, it can narrow down the range of putative solutions to guide learning towards a good solution in less time and with fewer training samples. Such behaviour would address multiple aims of frugal AI (Girdhar et al., 2025).

Despite these potential advantages, most work in deep learning has aimed to reduce inductive biases: relying on scaling-up (the size of the training data-set, the model, and the computational resources) to bring about improvements in performance. Similarly, most work in frugal AI is focused on enhancing the efficiency of models at inference time rather than the efficiency with which models are learnt (indeed most frugal AI methods increase training inefficiency). In opposition to this overwhelming trend, some have advocated for the introduction of new biologically-inspired inductive biases into DNNs (Celeghin et al., 2023; Hassabis et al., 2017; Linsley et al., 2025; Malhotra et al., 2020; Roy et al., 2024; Zador, 2019; Zhu et al., 2020). However, the only well-known examples of work in this direction modify the first stage of a DNN to simulate the function of the retina (Kim et al., 2015; Stracke et al., 2025), the lateral geniculate nucleus (Bertoni et al., 2022), or V1 (Dapello et al., 2020; Evans et al., 2022; Strisciuglio et al., 2020). The effects of such “front-ends” are limited as they only change one stage out of many, and these effects could alternatively be produced by using an unmodified architecture with appropriate data-augmentations. More effective inductive biases are yet to be proposed.

Here we propose a in-built inductive bias inspired by a fundamental constraint in vision: that only one feature can occupy each individual location in an image. This implies that there should be competition between neurons which represent different interpretations of the same image patch. Neurophysiological evidence suggests that this is the case in the cerebral cortex (Carandini & Heeger, 2011; Lehky et al., 2005; Olshausen & Field, 1997; Tolhurst et al., 2009; Vinje & Gallant, 2000), and computational neuroscience suggests multiple different mechanisms by which this could be achieved (Beyeler et al., 2019; Foldi¨ ak,´ 1990; Grossberg, 1987; Harpur & Prager, 1994; Hoyer, 2003; Rozell et al., 2008; Song et al., 2000; Spratling, 2014; Spratling & Johnson, 2001, 2004; Zylberberg et al., 2011). While a number of attempts to add competition to convolutional neural networks (CNNs) have been made in the past (Section 2), it remains missing from popular, contemporary, models. We hypothesised that introducing competition into CNNs would enable better discrimination between distinct patterns, and hence, improve classification accuracy. Furthermore, we expected competition to reduce redundancy by encouraging neurons to represent distinct patterns. Current DNNs contain a high proportion of redundant neurons that serve no useful purpose (Dalvi et al., 2020). This fact is exploited in many methods of network pruning (Nowak et al., 2023; Sehwag et al., 2020; Wang et al., 2022), which aim to produce more efficient inference models by removing redundant neurons and/or connections from a pre-trained model. Such post-hoc pruning is inefficient as it relies on a large, redundant, model being trained first. Using competition to reduce redundancy should make more efficient use of the available neurons, without the need for pruning.

To test these hypotheses we designed a new activation function (HAND) that uses divisive-normalisation: a simple competition mechanism that is a canonical computation in neurophysiology (Albrecht & Geisler, 1991; Carandini & Heeger, 1994; Carandini & Heeger, 2011; Heeger, 1991, 1992b; Heeger & Zemlianova, 2020; Wainwright et al., 2001). Our version of divisive-normalisation is applied across the channels to produce competition between neurons sharing the same RF location. HAND was used to replace the standard activation function in different CNN architectures. We evaluated the performance of HAND-based networks to their unmodified baselines using three metrics: accuracy classifying the standard test dataset (“clean accuracy”), accuracy classifying the common corruptions data (“corrupt accuracy”), and ability to reject out-of-distribution samples (“OOD rejection”). We found that HAND typically led to improvements in all three performance metrics. This improvement was largest when the number of parameter updates was limited (i.e., using few training epochs), or the training data was limited in variety (i.e., using few training-time augmentations), or the training data was limited in volume (using reduced and imbalanced data): exactly those conditions in which a useful inductive bias should be most effective. Divisive-normalisation therefore seems to act as an inductive bias that increases data efficiency while maintaining or improving generalisation and robustness.

## 2 Related Work

Prior work has proposed many different mechanisms that modify a neuron’s response based on the (previous or current) activity of other neurons. Only the existing literature that is most prominent or most relevant to the current work is discussed here.

## 2.1 Normalisation Methods: Subtraction and Division

Normalisation layers have become common building-blocks in contemporary DNNs. These methods aim to standardise the mean and standard-deviation of neural activations in order to make training faster and more stable (Santurkar et al., 2018). All these methods thus have the same formulation:

$$
z = { \frac { y - { \bar { y } } } { \sqrt { \sigma + { \frac { 1 } { n } } \sum _ { i \in \mathbb { P } } ( y _ { i } - { \bar { y } } ) ^ { 2 } } } } \qquad { \bar { y } } = { \frac { 1 } { n } } \sum _ { i \in \mathbb { P } } y _ { i }\tag{1}
$$

Where $y$ and z are scalars representing the activity of a single neuron before and after normalisation, respectively; σ is a small non-negative constant used to prevent division-by-zero errors; and P is the set of n values used to calculate the statistics. P differs between the different normalisation methods. Specifically, if a feature-map is represented by a tensor with dimensions (B,C,H,W), then for BatchNorm (Ioffe & Szegedy, 2015) the statistics are calculated over (B,H,W) meaning that a separate mean and variance is calculated for each channel. For LayerNorm (Ba et al., 2016) the statistics are calculated over (C,H,W) meaning that a separate mean and variance is calculated for each sample in the batch. For InstanceNorm (Ulyanov et al., 2017) the statistics are calculated over (H,W). GroupNorm (Wu & He, 2018) is intermediate between LayerNorm and InstanceNorm meaning that a separate mean and variance is calculated for each predefined sub-set of channels of each sample in the batch. An early version of InstanceNorm calculated the statistics in a local, 9x9, Gaussian-weighted spatial region (Jarrett et al., 2009), rather than across the whole height and width of the feature map. These normalisation methods are typically followed by a learned affine transformation, which is omitted from the above equation.

## 2.2 Divisive-normalisation Methods: Division Only

Divisive-normalisation methods divide activity by a factor that is a weighted-sum of input values which have been raised to a power. Hence, a standard implementation is given by (Carandini & Heeger, 2011; Heeger, 1992b):

$$
z = { \frac { y } { \sigma + { \bar { y } } } } \qquad { \bar { y } } = \sum _ { i \in \mathbb { P } } w _ { i } y _ { i } \qquad y = x ^ { \alpha }\tag{2}
$$

Where x, y and z are scalars representing the activity of a single neuron before, at an intermediate step, and after normalisation, respectively; $\sigma$ is a small non-negative semi-saturation constant; α is a non-negative parameter; and P defines the set of neighbouring neural activations that are used to calculate the weighted sum, y¯. As for the methods described in Section 2.1, a major difference between previous implementations of divisive-normalisation is the extent of the neighbourhood, P, defining the normalisation pool of each neuron.

Some methods calculate the weighted sum over both channels and space (C,H,W). However, this summation is restricted to a local spatial region (Cirincione et al., 2022; Ren et al., 2017). Cirincione et al. (2022) used this form of divisive-normalisation as a front-end (part of an extended VOneBlock) to a ResNet applied to image classification with the ImageNetTiny data-set. They found this improved accuracy on the common corruptions data. Ren et al. (2017) found improved clean accuracy on CIFAR10 and CIFAR100 when this form of divisive-normalisation was used in multiple layers of a ResNet. Another method calculates the weighted sum over space only (H,W), but within a local neighbourhood. Hernandez-C´ amara et al. (´ 2023) applied this form of divisive-normalisation in certain layers of a U-Net applied to image semantic segmentation, and found that this improved generalisation performance due to increased contrast and illumination tolerance. Other implementations have calculated the weighted sum over channels only (C), but such that only a sub-set of channels contribute to the weighted sum. Krizhevsky et al. (2012) restricted the use of this form of divisive-normalisation to only the first two layers of an AlexNet. Others implemented modified versions of this method in more layers of AlexNet and found improved clean test accuracy on ImageNet1k (Miller et al., 2022; Pan et al., 2021), and improved performance with noisy test data (Miller et al., 2022).

Our proposed method sums across all channels to calculate y¯. Another major difference between our implementation and all others (except Miller et al. (2022)) is that we include the accelerating nonlinearity by setting $\alpha = 2$ . Previous work has mostly ignored this (i.e., it has set α = 1), despite neuroscientific evidence supporting a value of α of approximately two.

## 2.3 Other Mechanisms of Competition

A number of alternative mechanisms have been used to produce competition in DNNs. For example, a k-Winners-Take-All (k-WTA) activation function was found to improve adversarial robustness (Xiao et al., 2019). However, the improvement was modest and was caused by k-WTA destroying network gradient information. Hence, this method suffers from the limitations of other defences that rely on obfuscated gradients (Athalye et al., 2018; Carlini et al., 2019). Setting to zero al but the k highest magnitude (positive and negative) values in each channel (i.e., k-WTA applied across space separately in each channel), made a CNN learn more shape-biased representations (Li et al., 2023). Inhibition between neurons, in proportion the overlap between their weights, introduced into selected layers of ResNet50 produced small improvements in generalisation and larger improvements in robustness to weak adversarial attacks (Gao & Spratling, 2025).

## 3 Methods

## 3.1 HAND: Homeostasis, Accelerating Nonlinearity, and Divisive-nomalisation

We implement our new activation function as follows:

$$
z = { \frac { y } { \sigma + { \bar { y } } } } \qquad { \bar { y } } = w \sum _ { i \in \mathbb { P } } y _ { i } \qquad y = \operatorname { R e L U } ( \operatorname { B a t c h N o r m } ( x ) ) ^ { 2 }\tag{3}
$$

Where $\sigma$ and w are learnt, scalar, non-negative parameters shared by all neurons in a feature-map, and the other symbols are as defined for Eq. (2). Compared to the standard implementation of divisive-normalisation, Eq. (2), we make a number of changes which are described below.

Our aim was to implement competition between neurons that represent alternative interpretations of the same visual feature. We reasoned that this would require competition between all neurons at the same location in a feature map, and hence, set P to span across all channels. This is in contrast to previous versions (Section 2.2) that have limited the se P used for each neuron to a sub-set of neighbouring channels and/or a spatial neighbourhood. Preliminary experiments found that combining the activities of each neuron in the normalisation pool using separate, learnt, weights did not result in improved performance compared to using equal weighting. Hence, we reduced the number of learnt parameters, by summing the activities in the pool and multiplying this result by a single weighting factor, w. Furthermore, it was found that the same w could be used for all output neurons without a reduction in performance, hence, a single weight could be learnt for the entire layer. Similarly, it was found that a single semi-saturation constant, σ, could be used for the entire layer. Hence, our implementation introduces only two learnt parameters per layer. Modern CNNs can have a large number of channels. Hence, we initialised w to a small value (equal to $\frac { 1 } { \mathrm { { n u m . } c h a n n e l s } } )$ to prevent initial values of y¯ being large and suppressing all activity. The value of σ was initialised to one. Learning typically resulted in the value of w increasing, so that the effects of normalisation became stronger over time.

Increased separation between active and inactive neurons is produced by the accelerating non-linearity. The exponent, α, was fixed to a value of two which is the value typically estimated for this parameter from biological data. Preliminary experiments found that learning α could further improve clean and corrupt accuracy, but at the expense of reduced OOD rejection performance. Divisive-normalisation alone simply scales all activations equally rather than producing competition. This is because each of the neurons at the same location in a feature map are being normalised by the same value (σ and y¯ are the same for all such neurons). However, divisive-normalisation does stabilize learning and encourages neurons to learn to respond to distinct patterns. This occurs because gradients that encourage a stronger output from a neuron will also encourage weaker outputs from other neurons at the same location in order to reduce y¯.

One concern when introducing competition into a neural network is that it can increase the number dead neurons: ones that never respond as they always loose the competition. Even without competition mechanisms dead neurons are a concern in modern DNNs (Lu et al., 2020; Pinto et al., 2025). To address this concern we applied BatchNorm to the inputs to the activation function. This BatchNorm did not use the affine transformation. From a biological-perspective BatchNorm can be interpreted as performing a type of homeostasis: keeping each neuron within its stable operating conditions.

We replaced the standard activation function used in three CNN architectures with HAND. Some additional modifications were also made to avoid instabilities resulting from interactions between HAND and normalisation methods already implemented in some architectures. For ConvNeXt (Liu et al., 2022) HAND replaced GELU in each block, and in the preceding convolutional layers the bias terms were removed, as was the affine transformation in the preceding LayerNorm. For ResNet (He et al., 2016) HAND replaced ReLU in the first two activation functions in each block and the BatchNorms preceding these activation functions were removed. For AlexNet (Krizhevsky et al., 2012) HAND replaced the ReLUs following each convolutional layer, and no other modifications were made.

## 3.2 Training Settings

Performance was assessed for CNNs trained on ImageNet1k (Russakovsky et al., 2015) and the long-tailed version of this data: ImageNetLT. For ImageNetLT the standard image sub-sets were used<sup>a</sup>, meaning that the number of samples per class ranged from 5 to 1280 (corresponding to an imbalance ratio of 256). The training set-ups used were based on the standard pytorch recipes<sup>b</sup> for the architectures we used. Specifically, for training AlexNet we used the SGD optimiser with a momentum of 0.9, weight decay of 1e-4, and a batch size of 128. The initial learning rate was 0.01 and this was reduced by a factor of 10 after $\textstyle { \frac { 1 } { 3 } }$ and $\frac { 2 } { 3 }$ of the total training epochs. For ResNet50 the set-up was the same except the batch size was 512 and the initial learning rate was 0.1 which was reduced $\textstyle { \frac { 1 } { 4 } } , \ { \frac { 1 } { 2 } }$ and $\frac 3 4$ of the way through training. For training ConvNeXt we used the AdamW optimiser (Loshchilov & Hutter, 2019) with weight decay of 0.05, except for the normalisation layers where no weight decay was used. The batch size was 512 and the peak learning rate was 0.001. The One-Cycle learning schedule (Smith & Topin, 2018) was used with the warm-up period set to be 20 epochs or $\frac { 3 } { 1 0 }$ of the total training epochs, whichever was the smallest. Cross-entropy (CE) loss was used in all cases. Comparable training time information was obtained by training AlexNet on a single NVIDIA Tesla V100 GPU with 16GB of memory, while other networks were trained using four NVIDIA Tesla A100 GPUs with 40GB.

Two different training data augmentation schemes were used. The basic augmentations consisted of RandomResized Crop to 224 pixels and random horizontal flipping. The strong augmentation scheme included the basic augmentations plus TrivialAug (Muller & Hutter,¨ 2021), RandErase with probability 0.1 (Zhong et al., 2017), mixup (Zhang et al., 2018), and cutmix (Yun et al., 2019). When using this scheme we also used label smoothing (Szegedy et al., 2015) in accordance with the pytorch standard training recipe.

To ensure the results generalised to other data-sets some additional experiments were performed with SVHN (Netzer et al., 2011), CIFAR10 (Krizhevsky, 2009), CIFAR100 (Krizhevsky, 2009), TinyImageNet (TinyIN), and long-tailed versions of CIFAR10 and CIFAR100. The long-tailed versions were created, as in previous work (Cao et al., 2019; Liu et al., 2019; Wang et al., 2021), by removing training samples unequally from each class. Different degrees of imbalance were produced so that the ratio of the number of samples in the classes with the largest and smallest numbers was 100 (designated -LT100) or 10 (“-LT10”). These additional experiments were performed with ConvNeXt-tiny with the standard stem (which performs convolution with 4x4 kernels and a stride of 4) replaced by one more appropriate to small sized images (3x3 kernels and a stride of 1). This was trained using the same training hyper-parameters as specified above, except the batch size was 128.

Some experiments with long-tailed data were performed with Logit-adjusted (LA) loss (Menon et al., 2021). LA loss is a variation of CE loss where the logits are modified by a term that is proportional to the relative number of training samples in each class. Hence,

$$
\mathcal { L } _ { L A } = - \log z _ { l }\tag{4}
$$

where l is the index corresponding to the correct class (i.e., the ground-truth class label), and z is the result of applying the softmax function to the adjusted logits:

$$
z _ { j } = \frac { \exp ( y _ { j } + \log ( p _ { j } ) ) } { \sum _ { i = 1 } ^ { n } \exp ( y _ { i } + \log ( p _ { i } ) ) }\tag{5}
$$

The term $p _ { j }$ is the proportion of training samples in class $j ,$ , i.e. $p _ { j } = s _ { j } / \sum _ { i = 1 } ^ { n } s _ { i }$ where $s _ { j }$ is the number of samples in class $j , \mathbf { y } \in \mathbb { R } ^ { n }$ are the logits, and n is the number of classes. A number of similar losses have been proposed which use alternative methods to adjust the logits (Cao et al., 2019; Ren et al., 2020; Tan et al., 2020). However, LA loss has been found to produce better results than these alternatives and many other methods of dealing with class imbalance (Menon et al., 2021; Zhao et al., 2024). In a few situations learning failed (there was either training collapse or no improvement in performance on the validation set) when using LA loss with the default training setup described above. Specifically, for the experiments reported in the third panel of Table 1 training failed for ResNet50 with both ReLU and HAND, and for the standard versions of ConvNeXt-tiny and ConvNeXt-base. In these situations the initial learning rate was reduced by a factor of 0.1 which allowed training to complete successfully.

## 3.3 Performance Metrics

Performance was evaluated against a number of different criteria. Firstly, the percentage of samples correctly classified from the ImageNet1k validation set (the “clean” accuracy). Note, that this means that ImageNetLT-trained networks were tested with balanced data. Secondly, the ability of trained networks to generalise to input distribution shifts was assessed by determining classification accuracy with the common corruptions data-set: ImageNet-C (Hendrycks & Dietterich, 2019). This data-set contains 18 different corruptions including different types of noise, blurring, synthetic weather conditions, and digital corruptions. As is typical in the literature, performance was evaluated by averaging performance over all the corruptions at all degrees of intensity.

A third performance metric was used to assess the ability of a network to distinguish known from unknown classes. This was evaluated using the Area Under the Receiver Operating Characteristic curve (AUROC) as this is a common choice in the literature (Chen et al., 2023; Cheng et al., 2023; Kirchheim et al., 2022; Lee et al., 2022; Xu-Darme et al., 2023; Yang et al., 2022; Yang et al., 2023). AUROC is calculated separately for each unknown class data-set, evaluating how distinct the confidence scores produced by samples from the standard test-set are from the confidence scores produced in response to samples from the unknown class data-set. The method for determining confidence was Maximum Logit Score (MLS) (Hendrycks et al., 2022; Vaze et al., 2022). MLS defines the confidence that a sample is of a known class as the maximum response of the network output before any activation function is applied. AUROC was calculated using the seven data-sets specified in (Spratling, 2025), and the average AUROC across all seven data-sets was reported. These seven data-set containing unknown classes were Textures (Cimpoi et al., 2014), the iNaturalist 2021 validation set (Van Horn et al., 2018), the ImageNet-O data-set (Hendrycks et al., 2021), and four data-sets containing synthetic images: (1) images containing random blobs, as used in (Hendrycks et al., 2019); (2) images in which each pixel intensity value was independently and randomly selected from a uniform distribution; (3) the images of the standard (clean) test set after a random permutation of all pixels; (4) the images of the clean test set after randomising the phase, in the Fourier domain, of each image.

Additional experiments to evaluate generalisation in ImageNet-trained models were performed with the ModelvsHuman benchmark (Geirhos et al., 2021). This employs 17 data-sets, including multiple types of image corruption (Geirhos et al., 2018), edge-filtered images and silhouettes, Stylized-ImageNet and texture-shape cue conflict images (Geirhos et al., 2019), and ImageNet-Sketch (Wang et al., 2019). As well as evaluating networks in terms of the accuracy in classifying these images, this benchmark also evaluates the consistency between the classification decisions made by a DNN and human subjects who have been tested using the same images (“error consistency”).

The additional experiments performed using alternative training data-sets were evaluated in equivalent ways. Generalisation was assessed using MNIST-C (Mu & Gilmer, 2019) (for SVHN-trained models), and CIFAR10-C, CIFAR100-C, and TinyImageNet-C (Hendrycks & Dietterich, 2019) (for CIFAR10, CIFAR100, and TinyIN-trained models, respectively). For each training data-set, OOD Rejection was evaluated using the seven test-sets specified in Spratling (2025).

Each experiment was repeated three times (using a different random weight initialisation and random presentation order of training samples), and the average performance across these three repeats is reported. To identify conditions where the difference in performance was statistically significant, performance produced by corresponding networks using HAND and the standard activation function were compared using the two-sample t-test (with p < 0.05).

## 4 Results

## 4.1 Effects of Training Duration and Data Augmentation

Using HAND as the activation function resulted in better performance particularly for short training times (Fig. 1). On average a ConvNeXt-tiny incorporating HAND trained for 25 epochs on ImageNet1k with basic training data augmentation (Fig. 1, top row) had clean accuracy of 73.76%, corrupt accuracy of 36.77%, and OOD rejection AUROC of 91.21%. An unmodified ConvNeXt-tiny achieved 71.24%, 35.77%, and 89.20% average performance on these metrics when trained in the same way. The unmodified network needed to be trained for 200 epochs to reach the same clean accuracy as the HAND network achieved after 25 epochs. Specifically, at 200 epochs the unmodified networks metrics were: 73.72%, 35.31%, and 91.36%. Replacing GELU with HAND in ConvNeXt-tiny increases the time taken for each epoch. Hence, for a fair comparison the results are plotted in terms of training time. The time required for the GELU-based network to reach the same clean accuracy as the HAND-based network achieved after 25 epochs was 5.7 times longer in terms of clock-time and 8 times longer in terms of training epochs.

At longer training times, the gap in performance narrows, as is expected if HAND acts as an inductive bias (see Section 1). However, even at 200 training epochs the difference in clean accuracy is significant. Increasing the variety of training data, by using the strong augmentation scheme (Section 3.2), reduced the advantages produced by HAND and by 200 epochs had eliminated them (Fig. 1, bottom row). As discussed in Section 1, this is entirely consistent with the effects of inductive bias. Comparing the corresponding graphs in the two rows of Fig. 1, it can be seen that stronger training data augmentations improved generalisation performance, as measured by the clean and corrupt accuracy, while reducing OOD rejection performance, and that this is the case for networks using both activation functions.

![](images/8e8bb339f31141150c9a45e3a380e117a7886475537ebe6bb515ef3af1ff6e14.jpg)

b  
![](images/c58d06540e7cba3240b970969e940985c0a808a37161eca8da5b42600f7e19f9.jpg)

c  
![](images/123d35c4da8bdce5347eea8741669ac960e50ea3478769d53cef0f903bbded38.jpg)

d  
![](images/a667818ed99a301968666aa87ef862ab3c09de8b68ad2806b3a2cdf5aecbcc1d.jpg)

e  
![](images/294aac3cb8d660ad37547fe6bd2d052643b0b30b45b597107e726e3e0ce02399.jpg)

f  
![](images/7cc30da3311b746b3823acafea05943bc453db01f45a437cf1a239fe15ec4c89.jpg)  
Figure 1: Effects of training duration for ConvNeXt-tiny trained on ImageNet1k. The plotted points correspond to 10, 25, 50, 100, and 200 training epochs. Each shows performance of separate DNNs trained for that number of epochs, not the evaluations of checkpoints taken during the training of the same DNN. The solid lines show the mean values over three trials, and the shaded regions (which can be too narrow to be visible) indicate the minimum and maximum values recorded in any of the three trials. Stars indicate conditions where the difference in performance is statistically significant.

## 4.2 Effects of Data Scarcity and Imbalance

ImageNetLT is a version of ImageNet1k from which training samples have been removed, making training data more scarce. The number of samples removed from each class is unequal, making the training data imbalanced. When training with this data-set, the improvements produced by HAND were larger and were not reduced by a longer training duration (Fig. 2). A ConvNeXt-tiny incorporating HAND trained for 25 epochs on ImageNetLT with basic training data augmentation had clean and corrupt accuracy unmatched by the unmodified network even after training for 200 epochs (Fig. 2a and b). The OOD rejection performance of the HAND-based network also exceed that of the GELU-based network after 50 training epochs (Fig. 2c).

Logit-adjusted (LA) loss (Menon et al., 2021) is a loss function that is effective for training networks with imbalanced data. Repeating the previous experiment using LA instead of CE loss produced the results shown in the second row of Fig. 2. Comparing these results with those for CE loss in the first row of Fig. 2 it can be seen that LA loss boosts clean and corrupt accuracy for the standard ConvNeXt-tiny. However, despite LA loss being a recent state-of-the-art method for dealing with long-tailed training data (Zhao et al., 2024), the improvements produced by LA are small compared to those obtained by using HAND. Furthermore, the clean and corrupt accuracy produced with HAND is also further boosted by using LA loss, meaning that the advantage over the standard, GELU-based, ConvNeXt-tiny is maintained.

## 4.3 Generalisation to Other CNN Architectures and Data-sets

To ensure that the results presented above generalise from ConvNeXt-tiny, experiments were performed using two other CNN architectures and a larger version of ConvNeXt trained on ImageNet1k (Table 1, top panel) and ImageNetLT (Table 1, third panel). It can be seen that the performance enhancement produced by HAND at short training durations with ImageNet1k and at longer training durations with ImageNetLT was consistent across CNN architectures and sizes. Furthermore, the more comprehensive evaluation provided by ModelvsHuman compared to common corruptions also showed a boost in generalisation performance from using HAND, and better alignment between the errors made by a HAND-based DNN and humans. The one exception was ResNet50 trained on ImageNetLT, where the results were mixed, suggesting HAND is less compatible with ResNet than ConvNeXt. We also found (results not shown) that HAND did not enhanced the performance of the vision transformer (Dosovitskiy et al., 2020). This was inline with our expectations, as visual transformers mix information from different parts of the whole image so that neurons do not have localised RFs in the same way as CNNs do.

a  
![](images/6aca63bdc9f2bba364f516ef1d6f2748b0f7628d22070d82bb01e019bf3267a2.jpg)

b  
![](images/7179c5971647483c110d6107e1365069b3f4ec59173192ccada0e783475ec34b.jpg)

c  
![](images/46946010d8610f350ee813c7a32d2fb9864f9119c9173e67d944632df0f7c057.jpg)

d  
![](images/c5409fabe3dff124f4f1011c675b90981b73eb112feb93f6e878fab91c5b2dfe.jpg)

e  
![](images/8904b9ae07f7c51499fc376be0bc250ff36cd067ab214fc8526c366dabebdbf2.jpg)

f  
![](images/8b6ac81bfd0e68e9af3cb7ba8b9831effb14ab9bedad83e9683c4b3b6634b00b.jpg)  
Figure 2: Effects of training duration for ConvNeXt-tiny trained using basic augmentations on ImageNetLT. The plotted points correspond to 10, 25, 50, 100, and 200 training epochs. Each shows performance of separate DNNs trained for that number of epochs, not the evaluations of checkpoints taken during the training of the same DNN. The solid lines show the mean values over three trials, and the shaded regions (which can be too narrow to be visible) indicate the minimum and maximum values recorded in any of the three trials. Stars indicate conditions where the difference in performance is statistically significant.

To ensure that the results presented in the preceding sections also generalise from ImageNet1k to other data-sets, experiments were performed with other balanced training data-sets (Table 1, second panel) and imbalanced training data (Table 1, bottom panel). In all cases, substituting the standard activation function for HAND produced similar or improved performance on all metrics. For SVHN the effects of HAND were mixed. However, this is an easy to learn data-set and performance has plateaued long before 25 training epochs. Hence, this result shows the effects of HAND after plenty of training epochs, where the effects of an inductive bias are expected to be weak. For the harder data-sets (CIFAR10, CIFAR100, and TinyIN) HAND improves performance on all metrics. HAND also improved performance on all metrics for the additional long-tailed data-sets, and this improvement was almost always statistically-significant.

## 4.4 Ablation Study

HAND consists of three components: Homeostasis (implemented using BatchNorm), an Accelerating Nonlinearity (implemented by squaring the output of ReLU), and Divisive-normalisation. To evaluate the contribution of each of these components, two architectures were trained with ablated versions of HAND that were each missing one component (see Table 2). BatchNorm was found to be essential for AlexNet, and produced improvements in the clean accuracy and OOD rejection performance with ConvNeXt-tiny. Removing the squaring reduces performance for both architectures on all metrics. The divisive-normalisation component was essential for stable learning with AlexNet, but reduced overall performance for ConvNeXt-tiny: however, this reduction was not statistically-significant for any of the metrics used. These results suggest that the accelerating non-linearity is the component that contributes most strongly to the improved performance produced by HAND, but that for some architectures the other components are essential to stabilise learning. Divisive normalisation has previously been found to perform an important role in maintaining the stability of neural activity during inference in recurrent networks (Morone et al., 2026; Rawat et al., 2024). Here, we find that it plays a role in training stability in feed-forward networks as well.

Table 1: Comparison of HAND with standard activation functions across different CNN architectures and data-sets. Experiments performed with balanced training data-sets (top two panels) used 25 training epochs, CE loss, and basic data augmentations. Experiments performed using imbalanced training data-sets (lower two panels) used 100 training epochs, LA Loss, and strong data augmentations. Bold text indicates the best performance on each metric for each combination of training data-set and network architecture. Coloured text indicates that the difference in performance is statistically significant.
<table><tr><td>Training</td><td>Architecture</td><td>Activation</td><td>Clean</td><td>Corrupt</td><td>ModelvsHuman</td><td></td><td>OOD rejection</td><td></td></tr><tr><td rowspan="2">Data-set</td><td rowspan="2">AlexNet</td><td rowspan="2">Function HAND</td><td rowspan="2">Accuracy (%)  ${ \bf 5 1 . 0 5 \pm \theta 0 . 1 5 }$ </td><td rowspan="2">Accuracy (%)  ${ \bf 2 0 . 7 8 \pm \ 0 . 1 4 }$ </td><td>Accuracy (%)</td><td>Error Consistency</td><td></td><td>AUROC (%)</td></tr><tr><td> $\mathbf { 4 1 . 6 7 \pm { \ : \ : 0 . 2 5 } }$ </td><td> $\mathbf { 0 . 1 6 5 \pm { \ : 0 . 0 0 8 } }$ </td><td>84.72±0.68</td><td></td></tr><tr><td rowspan="2">ImageNet1k</td><td rowspan="2">ResNet50</td><td>ReLU</td><td> $4 6 . 2 6 \pm \ : 0 . 2 3$ </td><td> $1 9 . 7 5 \pm \ : 0 . 2 6$ </td><td></td><td> $3 9 . 0 3 \pm \ : 0 . 5 7$ </td><td> $0 . 1 5 9 \pm \ : 0 . 0 0 6$ </td><td>78.46± 4.22</td><td></td></tr><tr><td>HAND ReLU</td><td> ${ \bf 6 7 . 3 5 \pm \theta 0 . 4 8 }$   $6 5 . 5 1 \pm \ : 1 . 1 2$ </td><td></td><td> $\mathbf { 3 1 . 0 6 \pm } \mathbf { \ : 0 . 4 4 }$ </td><td> ${ \bf 5 0 . 3 3 \pm \theta 0 . 3 2 }$ </td><td> $\mathbf { 0 . 1 8 6 \pm { \ : \ : 0 . 0 0 5 } }$ </td><td>87.13± 2.27</td><td></td></tr><tr><td></td><td>ImageNet1k ConvNeXt-tiny</td><td>HAND</td><td> $\mathbf { 7 3 . 7 6 \pm { \ : \ : 0 . 1 3 } }$ </td><td></td><td> $2 9 . 2 0 \pm \ : 0 . 8 9$   ${ \bf 3 6 . 7 7 \pm \theta . 6 7 }$ </td><td> $4 9 . 3 3 \pm \ : 0 . 5 1$   $\mathbf { 5 5 . 0 0 \pm \theta \eta 0 . 4 6 }$ </td><td> $0 . 1 8 5 \pm \ : 0 . 0 0 1$   $\mathbf { 0 . 1 9 2 \pm { \ : \ : 0 . 0 0 5 } }$ </td><td> $\mathbf { 8 8 . 5 2 \pm { \ : \ : 0 . 4 4 } }$   ${ \bf 9 1 . 2 1 \pm \mathrm { ~ 0 . 5 2 } }$ </td><td></td></tr><tr><td>ImageNet1k</td><td>ConvNeXt-base</td><td>GELU HAND</td><td> $7 1 . 2 4 \pm \ : 0 . 1 6$   $\mathbf { 7 4 . 9 4 \pm \theta . 0 6 }$ </td><td></td><td> $3 5 . 7 7 \pm \ : 0 . 1 0$   ${ \bf 4 0 . 2 1 \pm \mathrm { ~ 0 . 1 0 ~ } }$ </td><td> $5 2 . 4 3 \pm \ : 0 . 3 5$   ${ \bf 5 6 . 7 3 \pm \theta 0 . 8 6 }$ </td><td> $0 . 1 8 2 \pm \ : 0 . 0 0 2$   $\mathbf { 0 . 1 9 0 \pm { \ : \ : 0 . 0 1 1 } }$ </td><td> $\mathbf { 9 2 . 0 1 \pm { \ : 0 . 2 2 } }$ </td><td> $8 9 . 2 0 \pm \ : 1 . 6 2 $ </td></tr><tr><td>ImageNet1k</td><td></td><td>GELU HAND</td><td>72.94± 0.19  $\mathbf { 9 5 . 7 5 \pm { \ : \ : 0 . 0 6 } }$ </td><td></td><td> $3 9 . 0 2 \pm \ : 0 . 1 2$   $5 7 . 2 4 \pm \ : 1 . 7 8$ </td><td> $5 3 . 8 3 \pm \ : 0 . 8 4$  n/a</td><td> $0 . 1 7 9 \pm \ : 0 . 0 0 3$  n/a</td><td>90.27± 0.89 94.51± 0.79</td><td></td></tr><tr><td>SVHN</td><td>ConvNeXt-tiny</td><td>GELU HAND</td><td> $9 4 . 4 7 \pm \ : 0 . 0 2$   ${ \bf 9 1 . 3 8 \pm \mathrm { ~ 0 . 2 3 ~ } }$ </td><td></td><td> $\mathbf { 5 9 . 2 0 \pm { \ : 2 . 3 0 } }$   $\mathbf { 7 5 . 8 6 \pm { \ : \ : 0 . 6 0 } }$ </td><td>n/a n/a</td><td>n/a n/a</td><td> $\mathbf { 9 6 . 5 5 \pm { \ : 0 . 7 4 } }$  91.30± 1.01</td><td></td></tr><tr><td>CIFAR10</td><td>ConvNeXt-tiny</td><td>GELU HAND</td><td>88.49± 0.43  ${ \bf 6 9 . 3 1 \pm \mathrm { ~ 0 . 9 8 ~ } }$ </td><td></td><td> $7 5 . 5 3 \pm \ : 0 . 2 7$   $\mathbf { 4 9 . 6 2 \pm { \ : 0 . 2 5 } }$ </td><td>n/a n/a</td><td>n/a n/a</td><td>89.82± 0.82 81.42± 1.36</td><td></td></tr><tr><td>CIFAR100</td><td>ConvNeXt-tiny</td><td>GELU HAND</td><td>62.14± 0.69  ${ \bf 5 7 . 7 2 \pm \mathrm { ~ 0 . 1 7 } }$ </td><td></td><td>46.09± 0.21</td><td>n/a n/a</td><td>n/a</td><td>79.42± 1.72 73.80± 1.37</td><td></td></tr><tr><td>TinyIN</td><td>ConvNeXt-tiny</td><td>GELU</td><td>50.62± 0.84</td><td></td><td> ${ \bf 2 5 . 2 9 \pm \theta 0 . 0 9 }$  19.25± 0.37</td><td>n/a</td><td>n/a n/a</td><td>63.54± 4.14</td><td></td></tr><tr><td>ImageNetLT</td><td>AlexNet</td><td>HAND ReLU</td><td>27.69± 0.66 16.59± 0.55</td><td></td><td>12.00± 0.09 8.30± 0.26</td><td> $\mathbf { 3 4 . 7 0 \pm { \ : \ : 0 . 6 6 } }$   $2 4 . 4 0 \pm \ : 0 . 9 5$ </td><td> $\mathbf { 0 . 1 2 8 \pm ~ 0 . 0 0 2 }$   $0 . 0 8 3 \pm \ : 0 . 0 0 6$ </td><td>77.31± 1.41</td><td>60.89± 0.74</td></tr><tr><td>ImageNetLT</td><td>ResNet50</td><td>HAND ReLU</td><td>31.90± 0.54 33.23± 1.15</td><td></td><td> ${ \bf 1 4 . 0 2 \pm \theta 0 . 1 8 }$   $1 3 . 7 8 \pm \ : 0 . 5 6$ </td><td> $3 6 . 2 3 \pm \ : 0 . 5 7$   $\mathbf { 3 7 . 7 0 \pm { \ : \ : 0 . 3 6 } }$ </td><td> $0 . 1 2 9 \pm \ : 0 . 0 0 4$   $\mathbf { 0 . 1 3 0 \pm { \ : \ : 0 . 0 0 8 } }$ </td><td>68.50± 7.38</td><td>82.20± 0.52</td></tr><tr><td>ImageNetLT</td><td>ConvNeXt-tiny</td><td>HAND GELU</td><td> ${ \bf 5 0 . 4 8 \pm \mathrm { ~ 0 . 5 7 } }$  23.15± 0.87</td><td></td><td> ${ \bf 2 3 . 0 4 \pm \mathrm { ~ 0 . 3 7 } }$  10.32± 0.33</td><td> ${ \bf 4 7 . 9 3 \pm \delta 1 . 0 7 }$  29.80± 1.51</td><td> $\mathbf { 0 . 1 6 8 \pm { \ : 0 . 0 0 3 } }$   $0 . 1 0 1 \pm \ : 0 . 0 0 6$ </td><td>84.45± 0.48 75.80± 1.99</td><td></td></tr><tr><td>ImageNetLT</td><td>ConvNeXt-base</td><td>HAND GELU</td><td> ${ \bf 5 1 . 4 9 \pm \mathrm { ~ 0 . 3 8 ~ } }$   $3 1 . 5 2 \pm \ : 0 . 3 5$ </td><td></td><td> $\mathbf { 2 4 . 6 2 \pm { \ : 0 . 7 3 } }$   $1 4 . 4 5 \pm \ : 0 . 0 9$ </td><td> ${ \bf 4 8 . 6 7 \pm \theta 0 . 5 9 }$   $3 7 . 1 0 \pm \ : 0 . 2 0$ </td><td> $\mathbf { 0 . 1 6 6 \pm { \ : \ : 0 . 0 0 7 } }$   $0 . 1 2 5 \pm \ : 0 . 0 0 6$ </td><td> ${ \bf 8 5 . 6 7 \pm \delta 1 . 1 1 }$  77.81± 1.38</td><td></td></tr><tr><td>CIFAR10 -LT100</td><td>ConvNeXt-tiny</td><td>HAND GELU</td><td> ${ \bf 7 8 . 6 4 \pm \delta \ 0 . 1 7 }$   $6 8 . 3 3 \pm \ : 0 . 5 2$ </td><td></td><td> ${ \bf 6 6 . 6 2 \pm \theta 0 . 4 1 }$   $6 1 . 0 2 \pm \ : 0 . 6 4$ </td><td>n/a n/a</td><td>n/a n/a</td><td> $\mathbf { 7 3 . 5 5 \pm 4 . 2 5 }$   $6 3 . 5 7 \pm \ : 2 . 0 5$ </td><td></td></tr><tr><td>CIFAR10 -LT10</td><td>ConvNeXt-tiny</td><td>HAND GELU</td><td> $\mathbf { 9 0 . 6 6 \pm { \ : \ : 0 . 0 5 } }$   $8 8 . 0 6 \pm \ : 0 . 3 3$ </td><td></td><td> $\mathbf { 7 9 . 9 9 \pm { \ : \ : 0 . 2 4 } }$   $7 8 . 8 7 \pm \ : 0 . 4 8$ </td><td>n/a n/a</td><td>n/a n/a</td><td> $\mathbf { 8 0 . 2 1 \pm : 5 . 5 4 }$   $7 4 . 8 1 \pm \ : 0 . 9 5$ </td><td></td></tr><tr><td>CIFAR100 -LT100</td><td>ConvNeXt-tiny</td><td>HAND GELU</td><td> $\mathbf { 4 7 . 5 4 \pm { \ : \ : 0 . 9 0 } }$   $3 4 . 9 0 \pm \ : 1 . 0 0$ </td><td></td><td> ${ \bf 3 7 . 2 7 \pm \theta 0 . 9 8 }$   $2 9 . 0 6 \pm \ : 0 . 9 7$ </td><td>n/a n/a</td><td>n/a n/a</td><td> $7 5 . 7 3 \pm \ : 5 . 4 6$   $5 0 . 9 5 \pm \ : 3 . 0 7$ </td><td></td></tr><tr><td>CIFAR100 -LT10</td><td>ConvNeXt-tiny</td><td>HAND GELU</td><td> ${ \bf 6 4 . 6 8 \pm \theta 0 . 5 9 }$   $5 7 . 4 5 \pm \ : 0 . 7 5$ </td><td></td><td> $\mathbf { 5 1 . 5 1 \pm \ : 0 . 5 9 }$   $4 6 . 9 5 \pm \ : 0 . 7 2$ </td><td>n/a n/a</td><td>n/a n/a</td><td> ${ \bf 8 1 . 9 8 \pm \ : 2 . 2 5 }$   $7 2 . 3 5 \pm \ : 1 . 0 5$ </td><td></td></tr></table>

Table 2: Results of the ablation study. Training performed with ImageNet1k for 25 epochs with basic data augmentation. Results are also included for a version of AlexNet in which BatchNorm was applied prior to ReLU, to provide a fairer baseline with which to compare our results. Bold text indicates the best performance on each metric for each network architecture.
<table><tr><td>Architecture</td><td>Activation Function</td><td>Clean Accuracy (%)</td><td>Corrupt Accuracy (%)</td><td>OOD rejection AUROC (%)</td></tr><tr><td rowspan="6">AlexNet</td><td>HAND</td><td> ${ \bf 5 1 . 0 5 \pm \theta 0 . 1 5 }$ </td><td> ${ \bf 2 0 . 7 8 \pm \ 0 . 1 4 }$ </td><td> ${ \bf 8 4 . 7 2 \pm \mathrm { ~ 0 . 6 8 } }$ </td></tr><tr><td>HAND w/o BatchNorm</td><td></td><td>learning failed</td><td></td></tr><tr><td>HAND w/o squaring</td><td>48.41± 0.51</td><td> $1 8 . 5 1 \pm \ : 0 . 0 5$ </td><td> $8 1 . 6 4 \pm \ : 1 . 4 2$ </td></tr><tr><td>HAND w/o div-norm</td><td></td><td>learning failed</td><td></td></tr><tr><td>BatchNorm + ReLU</td><td> $4 9 . 7 6 \pm \ : 0 . 0 3$ </td><td> $1 9 . 5 4 \pm \ : 0 . 1 2$ </td><td> $8 1 . 9 4 \pm \ : 2 . 8 2$ </td></tr><tr><td>ReLU</td><td> $4 6 . 2 6 \pm \ : 0 . 2 3$ </td><td> $1 9 . 7 5 \pm \ : 0 . 2 6$ </td><td> $7 8 . 4 6 \pm \ : 4 . 2 2$ </td></tr><tr><td rowspan="5">ConvNeXt-tiny</td><td>HAND</td><td> $7 3 . 7 6 \pm \ : 0 . 1 3$ </td><td> $3 6 . 7 7 \pm \ : 0 . 6 7$ </td><td> $9 1 . 2 1 \pm \ : 0 . 5 2$ </td></tr><tr><td>HAND w/o BatchNorm</td><td> $7 1 . 8 7 \pm \ : 0 . 0 9$ </td><td>36.27± 0.29</td><td> $8 9 . 8 4 \pm \ : 0 . 8 2$ </td></tr><tr><td>HAND w/o squaring</td><td> $7 1 . 7 7 \pm \ : 0 . 1 4$ </td><td>32.89± 0.38</td><td>90.97± 0.71</td></tr><tr><td>HAND w/o div-norm</td><td> $\mathbf { 7 4 . 1 1 \pm { \ : 0 . 1 9 } }$ </td><td> ${ \bf 3 8 . 2 0 \pm \theta \ 0 . 2 6 }$ </td><td> $\mathbf { 9 1 . 2 4 \pm { \ : 0 . 9 0 } }$ </td></tr><tr><td>GELU</td><td> $7 1 . 2 4 \pm \ : 0 . 1 6$ </td><td> $3 5 . 7 7 \pm \ : 0 . 1 0$ </td><td>89.20± 1.62</td></tr></table>

## 4.5 Comparison with Previous Divisive-Normalisation Implementations

Existing normalisation layers (Section 2.1), and almost all previous implementations of divisive-normalisation applied to DNNs (Section 2.2), do not include an accelerating nonlinearity. Our ablation results show that when this component is missing the improvements in performance over unmodified networks are small, which is consistent with the results produced by previous methods that have been inspired by work on divisive-normalisation.

The previous implementations of divisive-normalisation that have been tested with ImageNet1k have been incorporated into AlexNet. These methods have not been shown to scale-up to modern architectures, nor have they been rigorously evaluated in terms of generalisation and OOD rejection. Miller et al. (2022), the only previous method to use an accelerating nonlinearity, reports a clean accuracy after 90 training epochs of 59.61%. However, this was produced using an AlexNet with 96, 256, 384, 384, and 256 channels in its convolutional layers. Our results in earlier sections were produced using the standard pytorch implementation of AlexNet that contains 64, 192, 384, 256, and 256 channels. Using HAND with the larger version of AlexNet and the same training setup as used in Miller et al. (2022) we obtain almost identical clean accuracy: 59.60%. Pan et al. (2021) claim higher clean accuracy (≈64%). However, they not only used the version of AlexNet with more convolutional filters, but also added two additional fully-connected layers. We did not test HAND with this bespoke architecture, but note that our ConvNeXt-tiny models with HAND can produce far better clean accuracy (73.76%) than reported by Pan et al. (2021) using an architecture containing far fewer parameters.

## 5 Discussion

The bias-variance trade-off (Geman et al., 1992) tells us that machine learning systems containing more inductive bias should require fewer training samples and should generalise more successfully. However, this principal does not tell us what are the appropriate inductive biases for different tasks. Here, we show that competition between neurons representing the same image patch, implemented using divisive-normalisation, is an appropriate inductive bias for image classification tasks. Our results show that this method, implemented as a new activation function (HAND), produces large improvements in performance, especially when using few training epochs, or limited training-time augmentations, or few training samples for certain classes.

The success of our implementation of divisive normalization relies on the inclusion of the accelerating non-linearity and the homeostasis mechanism preceding the normalization. We observe that the squaring operation and the BatchNorm are both more important for performance than the divisive normalization, which also explains the relatively weak performance of previous implementations. Both accelerating non-linearity (Albrecht & Geisler, 1991; Albrecht et al., 2002; Heeger, 1992a) and homeostasis mechanisms that keep neurons within their dynamic range (Davis, 2013; Davis & Bezprozvanny, 2001; Turrigiano & Nelson, 2000) are well supported by neuroscientific data, too. Thus, the improvements we include are still strongly inspired by biological neural networks and may further improve similarity between DNNs and biological neural networks.

Increasing inductive bias may have an important role to play in addressing several issues with deep learning, such as the exponentially increasing computational requirements (Sevilla et al., 2022) which undermine AI sovereignty and democracy and lead to unsustainable increases in environmental impact (Strubell et al., 2020; Thompson et al., 2020). In addition, the need for ever more training data reduces the ability to curate it: increasing the risk that the data contains, and hence the model learns, prejudices. Furthermore, this approach to improving AI seems to be reaching its limits, if claims that AI is running out of training data are to be believed (Jones, 2024).

A limitation of increasing inductive bias is that the resulting system is less general: it incorporates constraints that are specific to certain tasks and inappropriate for others. However, the fact that a single image location can only contain one image feature, and hence have one interpretation, seems to be a prior relevant to all vision tasks. Therefore, it seems likely that HAND could also improve performance in other vision-based applications. Verifying this is left to future work. We have successfully integrated HAND into existing CNNs. However, better performance might be possible with architectures optimised especially to fully exploit competition. Further work might also consider improved methods of implementing competition. For example, stronger methods of competition, that are more effective at suppressing responses that are poorer matches to the input, might further improve the ability to reject samples from unknown classes.

## Acknowledgements

The simulations were performed using the Luxembourg national supercomputer MeluXina and the HPC facilities of the University of Luxembourg (Varrette et al., 2022). The authors gratefully acknowledge the LuxProvide and University of Luxembourg teams for their expert support.

## References

Albrecht, D. G. & Geisler, W. S. (1991). Motion selectivity and the contrast-response function of simple cells in the visua cortex. Vis. Neurosci. 7: 531–46. DOI: 10.1017/S0952523800010336

Albrecht, D. G. et al. (2002). Visual cortex neurons of monkeys and cats: temporal dynamics of the contrast response function. J. Neurophysiol. 88: 888–913. DOI: 10.1152/jn.2002.88.2.888.

Athalye, A. et al. (2018). Synthesizing robust adversarial examples. Proc. Int. Conf. Mach. Learn. Vol. 80. Proc. Mach. Learn. Res. arXiv:1707.07397, pp. 284–93.

Ba, J. L. et al. (2016). Layer normalization. arXiv:1607.06450.

Bertoni, F et al. (2022). LGN-CNN: a biologically inspired CNN architecture. Neural Netw. 145: 42–55.

Beyeler, M. et al. (2019). Neural correlates of sparse coding and dimensionality reduction. PLoS Comput. Biol. 15: 1–33. DOI: 10.1371/journal.pcbi.1006908.

Cao, K. et al. (2019). Learning imbalanced datasets with label-distribution-aware margin loss. Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:1906.07413. Red Hook, NY, USA: Curran Associates Inc., pp. 1567–78.

Carandini, M. & Heeger, D. J. (1994). Summation and division by neurons in primate visual cortex. Science 264: 1333–6.

Carandini, M. & Heeger, D. J. (2011). Normalization as a canonical neural computation. Nat. Rev. Neurosci. 13: 51–62. DOI: 10.1038/nrn3136.

Carlini, N. et al. (2019). On evaluating adversarial robustness. arXiv:1902.06705.

Celeghin, A. et al. (2023). Convolutional neural networks for vision neuroscience: significance, developments, and outstanding issues. Front. Comput. Neurosci. 17: DOI: 10.3389/fncom.2023.1153572.

Chen, Y. et al. (2023). WDiscOOD: out-of-distribution detection via whitened linear discriminant analysis. Proc. Int. Conf. Comput. Vision. arXiv:2303.07543.

Cheng, Z. et al. (2023). Average of pruning: improving performance and stability of out-of-distribution detection. arXiv:2303.01201.

Cimpoi, M. et al. (2014). Describing textures in the wild. Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

Cirincione, A. et al. (2022). Implementing divisive normalization in CNNs improves robustness to common image corruptions. Proc. Conf. Advs. Neural Info. Proc. Sys. SVRHM Workshop. URL: https://openreview.net/forum?id= KAAbo44qhJV.

Cybenko., G. (1989). Approximations by superpositions of sigmoidal functions. Mathematics of Control, Signals, and Systems 2: 303–14.

Dalvi, F. et al. (2020). Analyzing redundancy in pretrained transformer models. arXiv:2004.04010.

Dapello, J et al. (2020). Simulating a primary visual cortex at the front of CNNs improves robustness to image perturbations. Proc. Conf. Advs. Neural Info. Proc. Sys. Vol. 34, pp. 13073–87. DOI: 10.1101/2020.06.16.154542.

Davis, G. W. (2013). Homeostatic signaling and the stabilization of neural function. Neuron 80: 718–28. DOI: 10.1016/j. neuron.2013.09.044.

Davis, G. W. & Bezprozvanny, I. (2001). Maintaining the stability of neural function: a homeostatic hypothesis. Annu. Rev. Physiol. 63: 847–69. DOI: 10.1146/annurev.physiol.63.1.847.

Dosovitskiy, A. et al. (2020). An image is worth 16x16 words: transformers for image recognition at scale. Proc. Int. Conf. Learning Representations. arXiv:2010.11929.

Evans, B. et al. (2022). Biological convolutions improve DNN robustness to noise and generalisation. Neural Netw. 148: 96–110. DOI: 10.1016/j.neunet.2021.12.005.

Foldi¨ ak, P. (1990). Forming sparse representations by local anti-Hebbian learning.´ Biol. Cybern. 64: 165–70.

Gao, B. & Spratling, M. W. (2025). Filter competition results in more robust convolutional neural networks. Neurocomputing 617: 128972. ISSN: 0925-2312. DOI: 10.1016/j.neucom.2024.128972.

Geirhos, R. et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. Proc. Int. Conf. Learning Representations. arXiv:1811.12231.

Geirhos, R. et al. (2018). Generalisation in humans and deep neural networks. Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:1808.08750.

Geirhos, R. et al. (2021). Partial success in closing the gap between human and machine vision. Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:2106.07411.

Geman, S. et al. (1992). Neural networks and the bias/variance dilemma. Neural Comput. 4: 1–58.

Girdhar, N. et al. (2025). A comprehensive review of frugal artificial intelligence: challenges, applications, and the road to sustainable AI. Soft Computing 29: 4823–56. DOI: 10.1007/s00500-025-10854-y.

Grossberg, S. (1987). Competitive learning: from interactive activation to adaptive resonance. Cogn. Sci. 11: 23–63.

Harpur, G. F. & Prager, R. W. (1994). A fast method for activating competitive self-organising neural networks. Proc. Int. Symposium on Artif. Neural Netw. Pp. 412–8.

Hassabis, D. et al. (2017). Neuroscience-inspired artificial intelligence. Neuron 95: 245–58. DOI: 10.1016/j.neuron.2017.06. 011.

He, K. et al. (2016). Deep residual learning for image recognition. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. arXiv:1512.03385, pp. 770–8.

Heeger, D. J. (1991). Nonlinear model of neural responses in cat visual cortex. In: Computational Models of Visual Processing. Cambridge, MA: MIT Press, pp. 119–33.

— (1992a). Half-squaring in responses of cat striate cells. Vis. Neurosci. 9: 427–43. DOI: 10.1017/S095252380001124X.

— (1992b). Normalization of cell responses in cat striate cortex. Vis. Neurosci. 9: 181–97. DOI: 10.1017/S0952523800009640.

Heeger, D. J. & Zemlianova, K. O. (2020). A recurrent circuit implements normalization, simulating the dynamics of V1 activity. Proc. Natl. Acad. Sci. U.S.A. 117: 22494–505. DOI: 10.1073/pnas.2005417117.

Hendrycks, D. & Dietterich, T. G. (2019). Benchmarking neural network robustness to common corruptions and perturbations. Proc. Int. Conf. Learning Representations. arXiv:1903.12261.

Hendrycks, D. et al. (2019). Deep anomaly detection with outlier exposure. Proc. Int. Conf. Learning Representations. arXiv:1812.04606.

Hendrycks, D. et al. (2021). Natural adversarial examples. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. arXiv:1907.07174.

Hendrycks, D. et al. (2022). Scaling out-of-distribution detection for real-world settings. Proc. Int. Conf. Mach. Learn. Vol. 162. Proc. Mach. Learn. Res. arXiv:1911.11132, pp. 8759–73. URL: https : // proceedings . mlr. press / v162 / hendrycks22a.html.

Hernandez-C´ amara, P. et al. (2023). Neural networks with divisive normalization for image segmentation.´ Pattern Recognit. Lett. 173: 64–71. ISSN: 0167-8655. DOI: 10.1016/j.patrec.2023.07.017.

Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. Neural Netw. 4: 251–7. DOI: 10.1016/ 0893-6080(91)90009-T.

Hornik, K. et al. (1989). Multilayer feedforward networks are universal approximators. Neural Netw. 2: 359–66. DOI: 10.1016/0893-6080(89)90020-8.

Hoyer, P. O. (2003). Modeling receptive fields with non-negative sparse coding. Neurocomputing 52–4: 547–52.

Ioffe, S. & Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. arXiv:1502.03167.

Jarrett, K. et al. (2009). What is the best multi-stage architecture for object recognition? Proc. Int. Conf. Comput. Vision. IEEE, pp. 2146–53. DOI: 10.1109/ICCV.2009.5459469.

Jones, N. (2024). The AI revolution is running out of data. what can researchers do? Nature 636: 290–2. DOI: 10.1038/ d41586-024-03990-2.

Kidger, P. & Lyons, T. (2020). Universal approximation with deep narrow networks. Proc. Annu. Conf. Learn. Theory. arXiv:1905.08539.

Kim, J et al. (2015). Convolutional neural network with biologically inspired ON/OFF ReLU. Proc. Int. Conf. Neural Info. Proc. Vol. 9492. Lecture Notes in Comp. Sci, Springer. DOI: 10.1007/978-3-319-26561-2 38.

Kirchheim, K. et al. (2022). PyTorch-OOD: a library for out-of-distribution detection based on pytorch. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops, pp. 4351–60. DOI: 10.1109/CVPRW56347.2022.00481.

Krizhevsky, A. et al. (2012). ImageNet classification with deep convolutional neural networks. Proc. Conf. Advs. Neural Info. Proc. Sys. Vol. 25. Curran Associates, Inc., pp. 1097–105.

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Tech. rep. University of Toronto.

Lee, J. et al. (2022). Gradient-based adversarial and out-of-distribution detection. Proc. Int. Conf. Mach. Learn. Workshop on New Frontiers in Adversarial Machine Learning. arXiv:2206.08255.

Lehky, S. R. et al. (2005). Selectivity and sparseness in the responses of striate complex cells. Vision Res. 45: 57–73.

Li, T. et al. (2023). Emergence of shape bias in convolutional neural networks through activation sparsity. Proc. Conf. Advs. Neural Info. Proc. Sys. Pp. 71755–66. DOI: 10.52202/075280-3141.

Linsley, D. et al. (2025). Better artificial intelligence does not mean better models of biology. Trends Cogn. Sci. DOI: 10.1016/j.tics.2025.11.016.

Liu, Z. et al. (2022). A convnet for the 2020s. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. arXiv:2201.03545.

Liu, Z. et al. (2019). Large-scale long-tailed recognition in an open world. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Pp. 2537–46.

Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. Proc. Int. Conf. Learning Representations. arXiv:1711.05101.

Lu, L. et al. (2020). Dying reLU and initialization: theory and numerical examples. Communications in Computational Physics 28: 1671–706. DOI: 10.4208/cicp.oa-2020-0165.

Malhotra, G. et al. (2020). Hiding a plane with a pixel: examining shape-bias in CNNs and the benefit of building in biological constraints. Vision Res. 174: 57–68. DOI: 10.1016/j.visres.2020.04.013.

Menon, A. K. et al. (2021). Long-tail learning via logit adjustment. Proc. Int. Conf. Learning Representations. arXiv:2007.07314. URL: https://openreview.net/forum?id=37nvvqkCo5.

Miller, M. et al. (2022). Divisive feature normalization improves image recognition performance in alexnet. Proc. Int. Conf. Learning Representations. URL: https://openreview.net/forum?id=aOX3a9q3RVV.

Morone, F. et al. (2026). Stabilization of recurrent neural networks through divisive normalization. Proc. Natl. Acad. Sci. U.S.A. 123: e2601841123. DOI: 10.1073/pnas.2601841123.

Mu, N. & Gilmer, J. (2019). MNIST-C: a robustness benchmark for computer vision. arXiv:1906.02337.

Muller, S. G. & Hutter, F. (2021). TrivialAugment: tuning-free yet state-of-the-art data augmentation. ¨ Proc. Int. Conf. Comput. Vision. arXiv:2103.10158.

Netzer, Y. et al. (2011). Reading digits in natural images with unsupervised feature learning. Proc. Conf. Advs. Neural Info. Proc. Sys. Workshop on Deep Learning and Unsupervised Feature Learning.

Nowak, A. I. et al. (2023). Fantastic weights and how to find them: where to prune in dynamic sparse training. Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:2306.12230.

Olshausen, B. A. & Field, D. J. (1997). Sparse coding with an overcomplete basis set: a strategy employed by V1? Vision Res. 37: 3311–25.

Pan, X. et al. (2021). Brain-inspired weighted normalization for CNN image classification. Proc. Int. Conf. Learning Representations. Workshop on How Can Findings About The Brain Improve AI Systems. DOI: 10.1101/2021.05.20. 445029.

Pinto, L. et al. (2025). Towards the theory for mitigating gradient issues and dead neurons in deep learning through a modified gaussian activation function. Neural Netw. 196: DOI: 10.1016/j.neunet.2025.108353.

Rawat, S. et al. (2024). Unconditional stability of a recurrent neural circuit implementing divisive normalization. Proc. Conf. Advs. Neural Info. Proc. Sys. Vol. 37. Curran Associates, Inc., pp. 14712–50. DOI: 10.52202/079017-0470.

Ren, J. et al. (2020). Balanced meta-softmax for long-tailed visual recognition. Proc. Conf. Advs. Neural Info. Proc. Sys. Vol. 33. arXiv:2007.10740. Curran Associates, Inc., pp. 4175–4186. URL: https://proceedings.neurips.cc/paper files/ paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf.

Ren, M. et al. (2017). Normalizing the normalizers: comparing and extending network normalization schemes. Proc. Int. Conf. Learning Representations. arXiv:1611.04520.

Roy, A. et al. (2024). Editorial: what AI and neuroscience can learn from each other – open problems in models and theories. Cogn. Comput. 16: 2331–3. DOI: 10.1007/s12559-024-10324-x.

Rozell, C. J. et al. (2008). Sparse coding via thresholding and local competition in neural circuits. Neural Comput. 20: 2526–63. DOI: 10.1162/neco.2008.03-07-486.

Russakovsky, O. et al. (2015). ImageNet large scale visual recognition challenge. Int. J. Comput. Vis. 115: 211–52. DOI: 10.1007/s11263-015-0816-y.

Santurkar, S. et al. (2018). How does batch normalization help optimization? Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:1805.11604.

Sehwag, V. et al. (2020). HYDRA: pruning adversarially robust neural networks. arXiv:2002.10509.

Sevilla, J. et al. (2022). Compute trends across three eras of machine learning. Proc. Int. Joint Conf. on Neural Netw. Pp. 1–8. DOI: 10.1109/IJCNN55064.2022.9891914.

Smith, L. N. & Topin, N. (2018). Super-convergence: very fast training of neural networks using large learning rates. arXiv:1708.07120.

Song, S. et al. (2000). Competitive Hebbian learning through spike-timing dependent synaptic plasticity. Nat. Neurosci. 3: 919–26.

Spratling, M. W. (2014). Classification using sparse representations: a biologically plausible approach. Biol. Cybern. 108: 61–73. DOI: 10.1007/s00422-013-0579-x.

— (2025). A comprehensive assessment benchmark for rigorously evaluating deep learning image classifiers. Neural Netw. 192: arXiv:2308.04137.

Spratling, M. W. & Johnson, M. H. (2001). Dendritic inhibition enhances neural coding properties. Cereb. Cortex 11: 1144–9. DOI: 10.1093/cercor/11.12.1144.

— (2004). Neural coding strategies and mechanisms of competition. Cogn. Syst. Res. 5: 93–117. DOI: 10.1016/j.cogsys. 2003.11.002.

Stracke, L. et al. (2025). Vision at night: exploring biologically inspired preprocessing for improved robustness via color and contrast transformations. Proc. Int. Conf. Comput. Vision. Workshop on Responsible Imaging. arXiv:2509.24863.

Strisciuglio, N. et al. (2020). Enhanced robustness of convolutional networks with a push-pull inhibition layer. Neura Comput. Appl. 32: 17957–71. DOI: 10.1007/s00521-020-04751-8.

Strubell, E. et al. (2020). Energy and policy considerations for modern deep learning research. Proc. AAAI Conf. on Artif. Intell. Vol. 34. 09, pp. 13693–6. DOI: 10.1609/aaai.v34i09.7123.

Szegedy, C. et al. (2015). Rethinking the inception architecture for computer vision. arXiv:1512.00567.

Tan, J. et al. (2020). Equalization loss for long-tailed object recognition. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. arXiv:2003.05176, pp. 11662–71.

Thompson, N. C. et al. (2020). The computational limits of deep learning. arXiv:2007.05558.

Tolhurst, D. J. et al. (2009). The sparseness of neuronal responses in ferret primary visual cortex. J. Neurosci. 29: 2355–70.

Turrigiano, G. G. & Nelson, S. B. (2000). Hebb and homeostasis in neuronal plasticity. Curr. Opin. Neurobiol. 10: 358–64. DOI: 10.1016/s0959-4388(00)00091-x.

Ulyanov, D. et al. (2017). Instance normalization: the missing ingredient for fast stylization. arXiv:1607.08022.

Van Horn, G. et al. (2018). The inaturalist species classification and detection dataset. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. arXiv:1707.06642.

Varrette, S. et al. (2022). Management of an Academic HPC & Research Computing Facility: The ULHPC Experience 2.0. Proc. ofthe 6th ACM High Performance Computing and Cluster Technologies Conf. (HPCCT). Fuzhou, China: Association for Computing Machinery (ACM).

Vaze, S. et al. (2022). Open-set recognition: a good closed-set classifier is all you need? Proc. Int. Conf. Learning Representations. arXiv:2110.06207.

Vinje, W. E. & Gallant, J. L. (2000). Sparse coding and decorrelation in primary visual cortex during natural vision. Science 287: 1273–6. : 10.1126/science.287.5456.1273.

Wainwright, M. J. et al. (2001). Natural image statistics and divisive normalization: modeling nonlinearities and adaptation in cortical neurons. In: Statistical Theories ofthe Brain. Cambridge, MA: MIT Press, pp. 203–22.

Wang, H. et al. (2019). Learning robust global representations by penalizing local predictive power. Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:1905.13549.

Wang, X. et al. (2021). Long-tailed recognition by routing diverse distribution-aware experts. Proc. Int. Conf. Learning Representations. URL: https://openreview.net/forum?id=D9I3drBz4UC.

Wang, Y. et al. (2022). Exploring optimal substructure for out-of-distribution generalization via feature-targeted model pruning. arXiv:2212.09458.

Wu, Y. & He, K. (2018). Group normalization. arXiv:1803.08494.

Xiao, C. et al. (2019). Enhancing adversarial defense by k-winners-take-all. arXiv:1905.10510.

Xu-Darme, R. et al. (2023). Interpretable out-of-distribution detection using pattern identification. URL: https://halcea.archives-ouvertes.fr/cea-03951966.

Yang, J. et al. (2022). OpenOOD: benchmarking generalized out-of-distribution detection. Proc. Conf. Advs. Neural Info. Proc. Sys. arXiv:2210.07242.

Yang, T. et al. (2023). MixOOD: improving out-of-distribution detection with enhanced data mixup. ACM Transactions on Multimedia Computing, Communications, and Applications. DOI: 10.1145/3578935.

Yun, S. et al. (2019). Cutmix: regularization strategy to train strong classifiers with localizable features. Proc. Int. Conf. Comput. Vision, pp. 6023–32. DOI: 10.1109/ICCV.2019.00612.

Zador, A. M. (2019). A critique of pure learning and what artificial neural networks can learn from animal brains. Nat. Commun. 10: DOI: 10.1038/s41467-019-11786-6.

Zhang, C. et al. (2017). Understanding deep learning requires rethinking generalization. Proc. Int. Conf. Learning Representations. arXiv:1611.03530.

Zhang, H. et al. (2018). Mixup: beyond empirical risk minimization. Proc. Int. Conf. Learning Representations. arXiv:1710.09412.

Zhao, L. et al. (2024). Logit normalization for long-tail object detection. Int. J. Comput. Vis. 132: 2114–34. DOI: 10.1007/ s11263-023-01971-y.

Zhong, Z. et al. (2017). Random erasing data augmentation. arXiv:1708.04896.

Zhu, Y. et al. (2020). Dark, beyond deep: a paradigm shift to cognitive AI with humanlike common sense. Engineering 6: 310–45. DOI: 10.1016/j.eng.2020.01.011.

Zylberberg, J. et al. (2011). A sparse coding model with synaptically local plasticity and spiking neurons can account for the diverse shapes of V1 simple cell receptive fields. PLoS Comput. Biol. 7: e1002250. DOI: 10.1371/journal.pcbi.1002250.