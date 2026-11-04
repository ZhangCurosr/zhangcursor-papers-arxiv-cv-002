# Polyp Matching Using Deep Learning

Kristian Dalsbø Hindberg<sup>1,∗</sup> Puneet Sharma<sup>2</sup> Benedicte Schelde-Olesen<sup>3,4</sup> Ulrik Deding<sup>3,4</sup> Jakob Frederik Frøkjær Justsen<sup>3,4</sup> Jan-Matthias Braun<sup>5,6</sup> Esmaeil S. Nadimi<sup>5,6</sup> All on behalf of the AICE Consortium<sup>1–6</sup>

<sup>1</sup>Department of Physics and Technology, Machine Learning Group, UiT – The Arctic University of Norway, Tromsø, Norway

<sup>2</sup>Department of Automation and Process Engineering, Machine Learning Group, UiT – The Arctic University of Norway, Tromsø, Norway

<sup>3</sup>Department of Surgery, Odense University Hospital, Odense, Denmark

<sup>4</sup>Department of Clinical Research, University of Southern Denmark, Odense, Denmark

<sup>5</sup>Faculty of Engineering, University of Southern Denmark, Odense, Denmark

<sup>6</sup>Applied AI and Data Science (AID), Maersk Mc-Kinney Moller Institute, Odense, Denmark

<sup>∗</sup>Corresponding author: kristian.d.hindberg@uit.no

## Abstract

A key challenge in colon capsule endoscopy (CCE) analysis is accurately identifying whether multiple images depict the same polyp, which is critical to avoid over-counting and unnecessary clinical interventions. Additionally, this prevents prolonged follow-up colonoscopies where endoscopists search for duplicate polyps without precise localization. This study presents a novel contrastive learning-based approach to polyp matching using a Siamese neural network architecture. The model was trained on a dataset from the Danish CareForColon2015 trial. The dataset comprises 5,103 image pairs of confirmed same polyp and 5,103 pairs of diferent polyps, annotated by experienced clinicians. The proposed model employs ResNet variants (ResNet18, ResNet34, ResNet50), DenseNet121 and EfficientNet B0/B3 as backbones and utilizes contrastive learning with cosine vector distance metric for classification. Experimental results demonstrate high classification accuracies of up to 92% and AUC values close to 0.98. These results suggest that automated polyp matching has the potential to reduce polyp over-counting, improve patient risk stratification, and reduce unnecessary follow-up colonoscopies.

Keywords: Colon capsule endoscopy, polyps, endoscopy, comparison, re-localization

## 1 Background

Colon capsule endoscopy (CCE) is increasingly employed for imaging and inspection of the colon [1, 2, 3, 4, 5, 6, 7, 8, 9]. In a CCE procedure, the patient swallows a capsule camera, which then traverses the gastrointestinal tract and exits via the anus. Prior to CCE, the patient goes through a procedure to clean their intestines [10, 11]. CCE lacks therapeutic capabilities, and when significant polyps are found in a CCE video, their removal requires a follow-up colonoscopy [12]. However, if the clinician is inaccurately informed of a higher number of polyps than actually present, additional time during colonoscopy may be spent searching for non-existent polyps. Furthermore, since the number of polyps identified is part of the risk allocation the patients receive, double counting of polyps may ultimately result in unnecessary follow-up, causing additional costs and discomfort or anxiety to patients [13, 4, 14].

There are several colon capsule/pill cameras on the market that are quite similar [15, 16]. Data of the current study are derived from CCE procedures conducted using second generation colon capsules (PillCam™ Colon 2, Medtronic, USA) [17]. This pill camera is 31.5 mm long and has a diameter of 11.6 mm. Each head of the pill is equipped with a camera that can function independently of the other. Each camera head has a view angle of 172<sup>◦</sup> to all sides, which in total mounts to almost 360<sup>◦</sup> view. Together, the two camera heads achieve a maximum frame rate of 35 frames per second. Typically, each camera head captures 5,000–10,000 images.

In a clinical setting, the camera capsule is known to exhibit bidirectional movement, occasionally traveling backwards between diferent colonic segments. Consequently, polyps and other features may be visualized multiple times during a CCE investigation [18, 14, 19]. To prevent the same polyp from being counted more than once, it is essential to have a reliable method for determining whether two or more images of polyps correspond to the same polyp. This task is called “polyp matching” and is essential in CCE, and the integration of artificial intelligence (AI) into this process is crucial [20].

There are several factors that make polyp matching based on solely images a challenging task [21]. First, a single polyp can appear significantly diferent when viewed from varying angles or distances. Second, the capsule’s onboard lighting can create diverse illumination conditions, including specular reflections. Third, the cleanliness of the colon can change over time afecting image consistency. In a setting where a clinician needs to determine whether two polyp identifications are of the same polyp, a list of relevant factors will be assessed. These include the time interval between images, the estimated location within the colon, the size and type of the polyps [21, 19]. However, this is a challenging task where there will be made errors by clinicians, and especially by non-expert clinicians [18].

In this paper, we focus on a fundamental question: “Given two images, both assumed to contain a polyp, are they showing the same polyp?”. To the best of our knowledge, this is the first attempt towards addressing this research question for CCE [21]. Within conventional colonoscopy, the typical situation is a bit diferent where the scope first is inserted in the rectum and moved to the start of colon (cecum) [22]. Clinical practice varies slightly regarding what is done during this insertion process. Some clinicians remove (at least small) polyps detected during the insertion process, during which they do a less thorough colon inspection compared to during the scope retraction process. Other clinicians only remove detected polyps during the retraction process from start (cecum) to end of colon (anus). This approach risks missing polyps detected during the insertion process. As the scope is normally inserted fully and then retracted without larger back-and-forth movements in the colon, the clinician will normally not be unsure about whether a detected polyp was revisited (except this aspect of polyps detected during scope insertion). However, for conventional colonoscopy there has been work done on tracking polyps and identifying the correct number of polyps. Parolari et al. [23, 24] use an architecture of ResNet and transformers to track, cluster and count polyps across consecutive frames. Similarly, Intrator et al. [25] present a method for polyp re-identification in colonoscopy videos using self-supervised learning, combining short-term tracking with appearance-based re-identification. However, such an approach is not applicable to CCE where images of the same polyp may be separated by several hours due to the capsule’s passive and bidirectional movement, which makes tracking impossible.

The main contributions of this paper are: (i) the first deep learning framework for polyp matching in CCE; (ii) a well-curated dataset of 5,103 same-polyp image pairs from the CareForColon2015 clinical trial; (iii) a Siamese network with pairwise contrastive loss achieving approximately 92% accuracy and AUC of 0.98; and (iv) a comparative evaluation of six CNN backbone architectures (ResNet18/34/50, DenseNet121, EficientNet B0/B3) confirming that performance is robust to backbone choice.

The paper is organized as follows: Section 2 describes the dataset and methodology, Section 3 presents the experimental results, and Sections 4 and 5 provide discussion and conclusions.

## 2 Methodology

This section introduces the dataset and the methodology of this study.

## 2.1 Dataset

In order to address the research question, we need at least two images known to be of the same polyp. Ideally, one would like to have images separated in time of the same polyp and seen from multiple directions. However, no such curated public dataset exists. In this study, we used the Danish CareForColon2015 trial [26, 27] as the main data source. From this dataset, clinicians from the Department of Surgery of Odense University Hospital have for a large set of polyps exported up to five images of the same polyp from the same camera pill passing (so there are no revisit images of polyps). The five exported images show 1. First partial, 2. First full, 3. Best full, 4. Last full, 5. Last partial, image of the same polyp. This means that 1 and 5 might just show parts of the polyp or the polyp might be very far away in the image, while 2–4 show the whole polyp. This did not give us images of the same polyp separated significantly in time. However, for each polyp we can use three pairs (2-3, 2-4, 3-4) of images showing the same full polyp. We used these pairs as training data for the “same polyp” class. For the “diferent polyps” class, random sets of images from diferent polyps are chosen, as described below.

In a real clinical setting, one would of course only use images of the same patient when doing inference. This can be done also in the training setting presented here, but this has at least two issues connected to it. Firstly, this restricts the analysis to patients with more than one polyp. This might mean that we undersample smaller polyps, which is expected to be on average the case for patients with a single polyp. Secondly, we know that there are some cases where the annotator has mislabeled images as being of diferent polyps, while they are really of the same polyp.

We have for the current study access to 2,780 polyps from 853 patients (The full dataset is constantly growing and currently consists of about 6,000 polyps, but all these have not been exported to this fiveimage-format described above). From this data set, we only use polyps that have been assigned all five images (partial, full etc), to be sure that we are working with images that show the whole polyp (images 2–4). This results in 1,701 polyps from 722 patients. From 1,701 polyps, each with three pairs, we get 5,103 image pairs in the “same polyp” class. From the very large number of possible pairs of diferent polyp images, we drew a random subset of 5,103 pairs for the “diferent polyp” class. As mentioned above, as the same polyp might wrongly be labeled as diferent polyps for a given patient, we do not use images from assumed diferent polyps from the same patient. I.e., a pair of non-same-polyp images are from two diferent patients. In the end, we have a balanced training set of 5,103 sample pairs for each of the two classes.

Figure 1 shows examples of these 1–5 images. The cleansing quality and presence of artifacts (like bubbles) can vary. We can note that depending on the angle or distance and the presence of artifacts in diferent images of the same polyp, the task of matching polyps can become very challenging.

All data splits were made at the patient level: all polyps, and hence all image pairs, of a given patient belong to exactly one of the training, validation or test sets, so there is no data leakage between sets. The splits are described in Section 2.3 and summarized in Table 1.

Table 1: Overview of the datasets used in this study. As all polyps of a given patient will remain in the same data set or fold split, the number of polyps within a fold will vary slightly. The set of “Test pairs” of “Split 2” is the same as the “missing” 15% not used in the CV of Split 1.
<table><tr><td></td><td>Number</td><td>Fraction of total</td></tr><tr><td>Patients in total</td><td>722</td><td>1.00</td></tr><tr><td>Polyps in total</td><td>1701</td><td>1.00</td></tr><tr><td>Polyp pairs</td><td>5103</td><td>1.00</td></tr><tr><td>Split 1: Cross-validation</td><td></td><td></td></tr><tr><td>- Polyp pairs</td><td>4338</td><td>0.85</td></tr><tr><td>Split 2: Final full model</td><td></td><td></td></tr><tr><td>- Training pairs</td><td>3572</td><td>0.70</td></tr><tr><td>- Validation pairs</td><td>766</td><td>0.15</td></tr><tr><td>- Test pairs</td><td>765</td><td>0.15</td></tr></table>

Appendix A documents the size and morphological distribution of this dataset. As the dataset is not intentionally filtered, it should resemble a general CCE population. It also includes information about the typical transit times across a polyp.

## 2.2 Network architecture

In this section, we discuss the suggested Siamese deep neural network for comparing images.

To assess whether two images, both containing polyps, are of the same polyp, we trained a Siamese deep neural network [28] based on contrastive learning [29, 30, 31, 32]. Figure 2 shows the network architecture for both training and inference mode. Contrastive learning is a class of metric learning approaches where a model learns to produce an embedding space in which similar inputs are mapped close together and dissimilar inputs far apart. It is well-suited for problems defined in terms of pairwise similarity rather than membership in a fixed set of classes. A closely related alternative is triplet loss [33], which requires one image to be designated as an anchor and the remaining two as a positive and a negative. This introduces an asymmetry that does not fit the polyp matching task naturally, where neither image in a pair has a privileged role. The pairwise contrastive loss therefore maps more directly onto both the structure of the training data, which consists of labelled same/diferent image pairs, and the clinical task of making a binary judgment between two candidate images.

![](images/ed56175d2df0959c711745309dd90239917303121317da61dace04b96a88db20.jpg)  
Figure 1: Rows 1–5: Each row shows images of a single polyp. Row 6: Example images of five diferent polyps. The figure also shows varying degree of cleansing quality, artifacts and partially visible polyps. The polyps are marked by a green ellipse.

![](images/5a199cb6eda220a9a3acbb403aec6a1f2f234b41cbc7d36841b3781d57889d1b.jpg)  
Figure 2: Network architecture of the proposed Siamese neural network with ResNet backbone and contrastive loss. Each input image (256x256 pixels, resized to 224x224 before processing) is independently fed through a shared backbone, producing a 128-dimensional embedding vector. During training, the cosine distance between the two embedding vectors is used to compute the contrastive loss, which pulls same-polyp pairs together and pushes diferent-polyp pairs apart. During inference, the cosine distance is compared against a learned threshold, where pairs below the threshold are classified as the same polyp and pairs above as diferent polyps.

A Siamese network consists of two identical networks that have the same structure and shared weights. When doing inference, two separate images are fed through each of these two “arms” of the Siamese network. The outputs of the two networks are then fed into an embedding layer of size 128, which results in a final embedding vector of length 128 for each image. The distance between these two embedding vectors forms the basis of the final classification, where a large distance is an indication of the images not being of the same polyp.

The ResNet family [34] was chosen as the main backbone due to its well-documented performance across medical imaging tasks, the availability of ImageNet-pretrained weights, and availability of diferent model sizes (number of residual blocks). In this study we used ResNet18, ResNet34 and ResNet50 as the backbone.

The cosine vector distance $D _ { W } ^ { \mathrm { c o s } } ( A , B )$ was employed for measuring vector distances between embedding vectors A and B. This is defined as 1 minus the cosine similarity cos(A, B), which gives

$$
D _ { W } ^ { \cos } ( A , B ) = 1 - \cos ( A , B ) = 1 - { \frac { A \cdot B } { \| A \| \| B \| } } ,\tag{1}
$$

where · is the vector dot product and ∥·∥ is the L2 norm.

We used a contrastive loss function [35]. This attempts to push images of diferent polyps further apart, while images of the same polyp should have as low as possible vector distance [36]. This loss is defined as

$$
L _ { C } = ( 1 - Y ) \cdot D _ { W } ^ { 2 } + Y \cdot \operatorname* { m a x } ( 0 , m - D _ { W } ) ^ { 2 } ,\tag{2}
$$

where $Y$ is $0 / 1$ for same/diferent polyp image pairs and $D _ { W }$ is some distance measure between the two embedding vectors. The margin m is there to stop the network from focusing on already large vector diferences for images pairs of diferent polyps. That is, if the vector distance is larger than m for a pair of diferent polyp images, this pair will not contribute to the loss function. We employed the commonly chosen cosine vector distance margin of $m = 0 . 5$ , which corresponds to cosine similarity of 0.5, which again corresponds to an angular separation of 60<sup>◦</sup>.

## 2.3 Training

In this section, we present the training approaches and parameters of our experiments.

The raw input images are 256x256 pixels. Even though ResNets can take many input dimensions, we resized the images to 224x224 pixels to fit the input dimensions of the pretrained ResNets (PyTorch with weights IMAGENET1K\_V1) we used. As the resizing factor is small, we do not expect to lose any significant detail level. The images were scaled to have zero mean and 1 standard deviation for each color channel. We used on-the-fly data augmentation techniques [37] to increase the efective sample size. We used random rotation (all angles), random horizontal and vertical flips (p=0.5), and color jittering (contrast=brightness=0.2).

In all experiments we ran the training for up to 100 epochs and enforced early stopping with a patience value of 20 and minimum loss decrease of 0.001 to avoid overfitting. This means that if a model trained for 40 epochs before early stopping is triggered, the model state at 20 epochs was chosen as the best model.

During the training phase, the threshold value for the cosine distance was set to 0.15. After the training had completed or early stopping had been triggered, the optimal threshold value was estimated for each fold. This was done via running a grid search over possible threshold values on the validation set, and keeping the threshold value that maximized accuracy.

As both classes are equally represented in the validation set, maximizing accuracy is equivalent to maximizing the Youden index and will normally give a balanced confusion matrix, making explicit ROC-based threshold selection redundant. This optimal validation-set-based threshold was then used for doing inference on the test set.

Prior to any cross-validation (CV) training, we split of a set of patients and their polyps into a test set. This random test set constitutes 15% of the total number of polyps (which also corresponds to approximately 15% of the patients), and was not used in any way until the final performance evaluation. The remaining 85% was used as the “CV data set”. Diferent-polyp pairs were only formed from patients within the same set.

We ran 10-fold CV where we ran 10 fully independent training runs. For each of the 10 folds, we had a test set of 10% of the CV data set, and the remaining 90% was split into 80% training and 20% validation. From the results of the 10 folds, we report mean and standard deviations of performance measures, and for each fold there is an estimated optimal threshold parameter that is recorded.

Following the CV, we performed a final training run on the whole dataset. Here we used the earlier split of 15% test set as the final test set, while the training and validation sets were 70% and 15% of the full data set, respectively. The validation set was used for early stopping and threshold selection. The performance measures of this final test set are reported as the final results of this study. We used the same other training parameters as for the CV training.

The networks were trained on The University of Southern Denmark’s Ucloud cluster solution on a NVIDIA H100 with 192 GB RAM. We used the PyTorch framework via NVIDIA’s optimized container release 25.11. Table 2 shows the training parameters used in this study.

Table 2: Parameters used in Siamese deep neural net training. The “train/val split” for CV is within the 90% of the CV data set used in that fold. The remaining 10% of each fold is used to evaluate the test set performance of that CV split.
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Backbones</td><td>ResNet 18/34/50</td><td>Image resize</td><td>256x256 to 224x224</td></tr><tr><td>Max epochs</td><td>100</td><td>Augmentation</td><td>On-the-fly</td></tr><tr><td>Batch size</td><td>64</td><td>- Random rotation</td><td>Uniform, 0°–360°</td></tr><tr><td>Early stopping</td><td></td><td>- Vertical/horizontal flip</td><td>p=0.5</td></tr><tr><td>- Patience</td><td>20</td><td>- Contrast jitter</td><td>0.2</td></tr><tr><td>- min loss decrease</td><td>0.001</td><td>- Brightness jitter</td><td>0.2</td></tr><tr><td>Loss</td><td>Contrastive</td><td>Cross-validation</td><td></td></tr><tr><td>Optimizer</td><td>Adam</td><td>- Folds</td><td>10</td></tr><tr><td>- Learning rate</td><td>0.0001</td><td>- Train/val split</td><td>80/20</td></tr><tr><td>- Weight decay</td><td>0.0001</td><td></td><td></td></tr></table>

## 3 Results

This section presents the results of the 10-fold CV training and the results of the final models based on the full dataset.

For the CV analysis, we present the estimated mean of the performance measures and their estimated standard deviation for all three backbones. As described above, the confusion matrices are expected to on average be symmetric, so we do not present them here. Each training run was at max run for 100 epochs and early stopping was enforced (and occurred in all CV runs). As we have run 10 folds and three ResNets backbones, the CV part consisted of 30 trained models. Table 3 shows the held-out test set results of various performance measures for the diferent ResNet backbones.

Table 3: Cross-validation test set performance measures for diferent ResNets. Mean values and ± 1 standard deviation across the 10 folds. AUC is area under the receiver operating characteristic curve.
<table><tr><td>Metric</td><td>ResNet18</td><td>ResNet34</td><td>ResNet50</td></tr><tr><td>Accuracy</td><td> $0 . 9 2 4 3 \pm 0 . 0 1 3 6$ </td><td> $0 . 9 2 3 1 \pm 0 . 0 0 9 6$ </td><td> $0 . 9 1 6 7 \pm 0 . 0 0 8 5$ </td></tr><tr><td>Precision</td><td> $0 . 9 2 5 3 \pm 0 . 0 1 7 9$ </td><td> $0 . 9 1 8 1 \pm 0 . 0 1 3 1$ </td><td> $0 . 9 1 4 8 \pm 0 . 0 1 6 8$ </td></tr><tr><td>Recall</td><td> $0 . 9 2 3 4 \pm 0 . 0 1 4 8$ </td><td> $0 . 9 2 9 2 \pm 0 . 0 1 2 5$ </td><td> $0 . 9 1 9 4 \pm 0 . 0 1 1 2$ </td></tr><tr><td>Specificity</td><td> $0 . 9 2 5 2 \pm 0 . 0 1 9 1$ </td><td> $0 . 9 1 7 0 \pm 0 . 0 1 4 3$ </td><td> $0 . 9 1 4 0 \pm 0 . 0 1 9 5$ </td></tr><tr><td>F1 score</td><td> $0 . 9 2 4 2 \pm 0 . 0 1 3 5$ </td><td> $0 . 9 2 3 6 \pm 0 . 0 0 9 5$ </td><td> $0 . 9 1 7 0 \pm 0 . 0 0 7 9$ </td></tr><tr><td>AUC</td><td> $0 . 9 7 8 0 \pm 0 . 0 0 5 1$ </td><td> $0 . 9 7 7 5 \pm 0 . 0 0 5 6$ </td><td> $0 . 9 7 5 9 \pm 0 . 0 0 5 3$ </td></tr><tr><td>Optimal threshold Early stopped epoch</td><td> $0 . 2 3 6 0 \pm 0 . 0 1 5 0$   $1 2 . 7 \pm \ : 5 . 9$ </td><td> $0 . 2 1 9 0 \pm 0 . 0 1 0 4$   $1 1 . 1 \pm \ : 3 . 9$ </td><td> $0 . 2 1 7 0 \pm 0 . 0 1 4 2$ </td></tr></table>

The final test set consists of 113 patients with 255 polyps. As each polyp creates three same-polyp pairs, there are $2 5 5 \cdot 3 = 7 6 5$ same-polyp pairs in the final test set (and the same number of diferent-polyp pairs). Table 4 shows the performance measures of the final models. Figure 3 shows the final training curves, with the best model marked (following early stopping). It can be seen that early stopping was triggered for all three.

Table 4: Test set performance measures for diferent ResNets. AUC is area under the receiver operating characteristic curve.
<table><tr><td>Metric</td><td>ResNet18</td><td>ResNet34</td><td>ResNet50</td></tr><tr><td>Accuracy</td><td>0.9193</td><td>0.9225</td><td>0.9206</td></tr><tr><td>Precision</td><td>0.9237</td><td>0.9275</td><td>0.9184</td></tr><tr><td>Recall</td><td>0.9141</td><td>0.9167</td><td>0.9232</td></tr><tr><td>Specificity</td><td>0.9245</td><td>0.9284</td><td>0.9180</td></tr><tr><td>F1 score</td><td>0.9188</td><td>0.9221</td><td>0.9208</td></tr><tr><td>AUC</td><td>0.9754</td><td>0.9768</td><td>0.9804</td></tr><tr><td>Optimal threshold</td><td>0.25</td><td>0.23</td><td>0.22</td></tr><tr><td>Early stopped epoch</td><td>9</td><td>18</td><td>46</td></tr></table>

![](images/4df0be3c96fd72ccac785267fdb6abc05f6965691036d5a2d58c28a63ffb7327.jpg)

![](images/ca009352c60e4cba7a7a426b179c8813b1929799cc49bfc4b09cb3af06d8da1b.jpg)

![](images/4402a8e21f3de3d0e1e38549b16c8b6de762f2c53d5b5eca7b06cdfc133492b7.jpg)  
Figure 3: Training and validation loss using cosine distance metrics, and ResNet of size 18, 34 and 50 from left to right. Best model following triggering of early stopping marked in red.

Even though the final models of this paper are selected following early stopping, we repeated the training processes without early stopping and let them run for 500 epochs. This was to make sure we did not miss an unexpected, high-epoch loss decrease. Figure 4 shows these 500-epoch training curves, and it can be seen that all curves have an increasing validation loss as the epoch number gets high. This does not mean that other training setups might not produce better results.

![](images/7c254dead7effe3eaeaf64304f3304ff45c0bfedc6a1183f496c468d63319789.jpg)

![](images/c54921b73279a24da61c5b5730a8bdb1b3b2440ab56f6ddc71a09e59e4b213f4.jpg)

![](images/ca56ca8b36267d52f8219c7f3ea23903bfbf9dd187210778f20a01aa0b805479.jpg)  
Figure 4: Training and validation loss when running for 500 epochs. The other hyperparameters are as in Table 2, except that no early stopping is used. ResNet of size 18, 34 and 50 are shown from left to right, respectively.

The runtime varied depending on the size of the ResNet used, with mean values of 9.4, 11.8 and 15.3 seconds/epoch for ResNet18, ResNet34 and ResNet50, respectively. As early stopping occurred for all runs, the actual training time for the CV was 2 hours and 54 minutes to train 10-fold CV of three ResNet backbones. The training time for the final models was 25 minutes for the three ResNet backbones.

To test whether other well-established models perform better than the ResNets, we ran the full analysis also using DenseNet121 [38], EficientNet [39] B0 and B3 as backbones. The results of these analyses are found in Appendix B. As can be seen, the diferences in results seem to be so small that it is hard to claim that one backbone is clearly better than the rest.

Appendix C displays the cosine vector distance distribution for the various set-ups and backbones.

## 4 Discussion

This study presents the first deep learning-based approach to polyp matching in CCE, achieving classification accuracy of up to 92.25% (ResNet34) and AUC values up to 0.9804 (ResNet50). These results compare favorably with clinician inter-observer agreement on polyp detection in CCE, which has been reported to be imperfect even among experts [18], suggesting that automated matching support could meaningfully reduce errors in clinical reading.

Across the six backbone architectures evaluated, performance was consistently high, with the ResNets slightly outperforming EficientNet and DenseNet variants. The similarity in performance across architectures suggests that the contrastive learning framework itself, rather than any specific backbone, is the primary driver of the results, and that the framework generalizes across CNN architectures.

The intended clinical use of this tool is as a decision support component within a CCE analysis pipeline. After a polyp detection algorithm flags candidate polyp images [20], the matching algorithm can be applied to any pair of images, outputting a binary same/diferent classification. Relevant questions in a clinical deployment would be: should all possible polyp image pairs be compared, even those separated by only a very short time? If not all pairs, how does one select which pairs to compare? In practice, this algorithm would serve as a supporting tool alongside other information such as time diference, polyp size, polyp type, and estimated position in the colon. Running inference on a pair of images typically takes about 40 milliseconds per compared pair on a NVIDIA RTX 2000 Ada Generation Laptop GPU. This means that it is feasible to compare a large number of polyp pairs even on normal GPUs, which facilitates integration of this matching procedure into existing workflows.

The main strength of the study is that we have a large set of images known to be of the same polyp, from which we have created a large and well-curated dataset. Additional strengths are that the strict patient-level data splitting ensures that no patient contributes to more than one of the train, validation, or test sets, and that the consistent performance across six diferent backbone architectures suggests that the results are not an artifact of any particular model choice.

However, there are important limitations to acknowledge. Most importantly, the presented setup does not include pairs of images known to be of the same polyp but separated significantly in time, which will typically be the case in a real clinical setting. A typical polyp image pair to be compared in clinical use is therefore expected to on average be less similar than what is shown in Figure 1, due to changes in illumination, viewpoint, and bowel preparation over time. As a consequence, the presented performance numbers are likely to be overestimates compared to real-world clinical performance. We have also only included images that show the whole polyp, which will not always be the case in practice. Furthermore, all diferent-polyp pairs were drawn from two diferent patients. In a real clinical setting the compared polyps will always be from the same patient. Within-patient pairs share to some extent bowel cleansing quality and mucosal appearance, and are therefore likely harder to separate, which may also contribute to an overestimation of performance. Finally, we only included polyps for which the operator was able to export five images. Whether this introduces any bias into the dataset is not clear, but as the same-polyp and diferent-polyp pairs are constructed from the same set of polyps, this is unlikely to be a significant issue for clinical use.

## 5 Conclusion

We propose a novel deep learning framework that addresses a critical challenge in CCE. The presented neural network takes two CCE images known to both contain polyps and tries to determine whether they are of the same polyp. Having such an automated tool available when evaluating the results of a CCE investigation would be helpful to avoid over-counting the number of detected polyps. This will both ensure more accurate patient risk allocation and ultimately correct patient follow-up and save procedure time when a clinician at a later time goes in and removes polyps.

Future work will focus on creating a dataset of images of the same polyps, but clearly separated in time. Furthermore, we plan to apply explainable AI methods to identify what is causing the wrong classifications of the presented method. It is also of interest to quantify how cleaning/image quality afects classification performance. Finally, it seems natural to expand from comparing two single images to comparing two sets of (neighboring) polyp images.

The presented approach is a first-of-a-kind feasibility study that shows the potential of using deep neural networks for polyp matching in colon capsule endoscopy, with the potential to save time and improve care in colon capsule endoscopy.

## Ethics declaration

The data used in this paper is from the CareForColon2015 trial. This was approved by The Regional Health Research Ethics Committee (Registration number: S-20190100), the Danish data protection agency (Ref. 19/29858), and has ClinicalTrials.gov registration number NCT04049357. All patients provided written consent to participate in the study. All work was done in accordance with the Declaration of Helsinki.

## Funding

All authors were funded by the AICE project (101057400), which is a HORIZON EUROPE project funded by the European Union.

## Availability of data and materials

Data access requests should be made to Prof. Gunnar Baatrup at Odense University Hospital (OUH).

## Acknowledgements

This research is part of AICE project (number 101057400) funded by the European Union, and it is part-funded by the United Kingdom government. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them.

## References

[1] Cristiano Spada, Cesare Hassan, Mariachiara Campanale, and Guido Costamagna. Colon capsule endoscopy. Techniques in Gastrointestinal Endoscopy, 17(1):19–23, 2015. doi: 10.1016/j.tgie.2015. 02.005.

[2] David Friedel, Rani Modayil, and Stavros Stavropoulos. Colon capsule endoscopy: Review and perspectives. Gastroenterology Research and Practice, pages 1–6, 2016. doi: 10.1155/2016/9643162.

[3] Naoki Hosoe, Kenji J. L. Limpias Kamiya, Yukie Hayashi, Tomohisa Sujino, Haruhiko Ogata, and Takanori Kanai. Current status of colon capsule endoscopy. Digestive Endoscopy, 33(4):529–537, 2021. doi: 10.1111/den.13769.

[4] Anastasios Koulaouzidis, Konstantinos Dabos, Michael Philipper, Ervin Toth, and Martin Keuchel. How should we do colon capsule endoscopy reading: a practical guide. Therapeutic Advances in Gastrointestinal Endoscopy, 14:26317745211001983, 2021. doi: 10.1177/26317745211001983.

[5] Sarah Moen, Fanny E. R. Vuik, Ernst J. Kuipers, and Manon C. W. Spaander. Artificial intelligence in colon capsule endoscopy—a systematic review. Diagnostics, 12(8):1994, 2022. doi: 10.3390/ diagnostics12081994.

[6] Emma Altobelli, Paolo Matteo Angeletti, Paolo Angelo Varesini, Zuleyka Bianchi, and Francesco Masedu. Colon capsule endoscopy as a promising diagnostic tool in colorectal cancer: A systematic review and network meta-analysis. Diagnostics, 15(17):2157, 2025. doi: 10.3390/diagnostics15172157.

[7] Ian Io Lei, Pablo Cortegoso Valdivia, Wojciech Marlicz, Karolina Skonieczna-Żydecka, Ramesh Arasaradnam, Rami Eliakim, and Anastasios Koulaouzidis. Systematic meta-review: diagnostic accuracy of colon capsule endoscopy for colonic neoplasia with umbrella meta-analysis. Therapeutic Advances in Gastrointestinal Endoscopy, 18:26317745251370845, 2025. doi: 10.1177/ 26317745251370845.

[8] Ian Io Lei, Ramesh Arasaradnam, Wojciech Marlicz, and Anastasios Koulaouzidis. Comment on Altobelli et al. colon capsule endoscopy as a promising diagnostic tool in colorectal cancer: A systematic review and network meta-analysis. Diagnostics 2025, 15, 2157. Diagnostics, 15(23):2938, 2025. doi: 10.3390/diagnostics15232938.

[9] Emma Altobelli, Paolo Matteo Angeletti, and Francesco Masedu. Reply to Lei et al. comment on “Altobelli et al. colon capsule endoscopy as a promising diagnostic tool in colorectal cancer: A systematic review and network meta-analysis. Diagnostics 2025, 15, 2157”. Diagnostics, 15(23):2939, 2025. doi: 10.3390/diagnostics15232939.

[10] Vicente Pons Beltrán, Cristina Carretero, Begoña González-Suárez, Iñaqui Fernandez-Urien, and Miguel Muñoz-Navas. Intestinal preparation prior to capsule endoscopy administration. World journal of gastroenterology, 14(37):5773–5775, 2008. doi: 10.3748/wjg.14.5773.

[11] C. Spada, C. Hassan, J.P. Galmiche, H. Neuhaus, J.M. Dumonceau, S. Adler, O. Epstein, G. Gay, M. Pennazio, D.K. Rex, R. Benamouzig, R. de Franchis, M. Delvaux, J. Devière, R. Eliakim, C. Fraser, F. Hagenmuller, J.M. Herrerias, M. Keuchel, F. Macrae, M. Munoz-Navas, T. Ponchon, E. Quintero, M.E. Riccioni, E. Rondonotti, R. Marmo, J.J. Sung, H. Tajiri, E. Toth, K. Triantafyllou, A. Van Gossum, and G. Costamagna. Colon capsule endoscopy: European Society of Gastrointestinal Endoscopy (ESGE) guideline. Endoscopy, 44(5):527–536, 2012. doi: 10.1055/s-0031-1291717.

[12] M. Jung. The “dificult” polyp: Pitfalls for endoscopic removal. Digestive Diseases, 30(Suppl. 2): 74–80, 2012. doi: 10.1159/000341898.

[13] Alex Ghanouni, Andrew Plumb, Paul Hewitson, Claire Nickerson, Colin J. Rees, and Christian von Wagner. Patients’ experience of colonoscopy in the English Bowel Cancer Screening Programme. Endoscopy, 48(3):232–240, 2016. doi: 10.1055/s-0042-100613.

[14] Thomas Bjørsum-Meyer, Anastasios Koulaouzidis, and Gunnar Baatrup. The optimal use of colon capsule endoscopes in clinical practice. Therapeutic Advances in Chronic Disease, 13: 20406223221137501, 2022. doi: 10.1177/20406223221137501.

[15] R. Scallan. State of the art inside view, the camera pill. Technology and Health Care, 24(4):471–481, 2016. doi: 10.3233/THC-161149.

[16] Rodrigo Gounella, Talita Conte Granado, Oswaldo Hideo Ando Junior, Daniel Luís Luporini, Mario Gazziro, and João Paulo Carmo. Endoscope capsules: The present situation and future outlooks. Bioengineering, 10(12):1347, 2023. doi: 10.3390/bioengineering10121347.

[17] Lucian Negreanu, Ruxandra Babiuc, Andreea Bengus, and Roxana Sadagurschi. Pillcam Colon 2 capsule in patients unable or unwilling to undergo colonoscopy. World journal of gastrointestinal endoscopy, 5(11):559–567, 2013. doi: 10.4253/wjge.v5.i11.559.

[18] Maria Magdalena Buijs, Rasmus Kroijer, Morten Kobaek-Larsen, Cristiano Spada, Ignacio Fernandez-Urien, Robert JC Steele, and Gunnar Baatrup. Intra and inter-observer agreement on polyp detection in colon capsule endoscopy evaluations. United European Gastroenterology Journal, 6(10):1563–1568, 2018. doi: 10.1177/2050640618798182.

[19] Ian Io Lei, Anastasios Koulaouzidis, Gunnar Baatrup, Mark Samaan, Ioanna Parisi, Mark McAlindon, Ervin Toth, Aasma Shaukat, Ursula Valentiner, Konstantinos John Dabos, Ignacio Fernandez, Alexander Robertson, Benedicte Schelde-Olesen, Nicholas Parsons, and Ramesh P. Arasaradnam. Rationalizing polyp matching criteria in colon capsule endoscopy: an international expert consensus through RAND (modified DELPHI) process. Therapeutic Advances in Gastroenterology, 17:1–12, 2024. doi: 10.1177/17562848241242681.

[20] Esmaeil S. Nadimi, Jan-Matthias Braun, Benedicte Schelde-Olesen, Smith Khare, Vinay C. Gogineni, Victoria Blanes-Vidal, and Gunnar Baatrup. Towards full integration of explainable artificial intelligence in colon capsule endoscopy’s pathway. Scientific Reports, 15:5960, 2025. doi: 10.1038/s41598-025-89648-z.

[21] Ian Io Lei, Ramesh Arasaradnam, and Anastasios Koulaouzidis. Polyp matching in colon capsule endoscopy: Pioneering CCE-colonoscopy integration towards an AI-driven future. Journal of Clinical Medicine, 13(23):7034, 2024. doi: 10.3390/jcm13237034.

[22] Douglas K Rex, Philip S Schoenfeld, Jonathan Cohen, Irving M Pike, Douglas G Adler, M Brian Fennerty, John G Lieb 2nd, Walter G Park, Maged K Rizk, Mandeep S Sawhney, Nicholas J Shaheen, Sachin Wani, and David S Weinberg. Quality indicators for colonoscopy. American Journal of Gastroenterology, 110(1):72–90, 2015. doi: 10.1038/ajg.2014.385.

[23] Luca Parolari, Andrea Cherubini, Lamberto Ballan, and Carlo Bifi. Temporally-aware supervised contrastive learning for polyp counting in colonoscopy. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2025, pages 543–553, 2026. doi: 10.1007/978-3-032-05127-1\_52.

[24] Luca Parolari, Andrea Cherubini, Lamberto Ballan, and Carlo Bifi. Towards polyp counting in fullprocedure colonoscopy videos. In 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), pages 1–5, 2025. doi: 10.1109/ISBI60581.2025.10981023.

[25] Yotam Intrator, Natalie Aizenberg, Amir Livne, Ehud Rivlin, and Roman Goldenberg. Self-supervised polyp re-identification in colonoscopy. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2023, pages 590–600, 2023. doi: 10.1007/978-3-031-43904-9\_57.

[26] Lasse Kaalby, Ulrik Deding, Morten Kobaek-Larsen, Anne-Line Volden Havshoi, Erik Zimmermann-Nielsen, Marianne Kirstine Thygesen, Rasmus Kroijer, Thomas Bjørsum-Meyer, and Gunnar Baatrup. Colon capsule endoscopy in colorectal cancer screening: a randomised controlled trial. BMJ Open Gastroenterology, 7(1):e000411, 2020. doi: 10.1136/bmjgast-2020-000411.

[27] G Baatrup, T Bjørsum-Meyer, L Kaalby, B Schelde-Olesen, M Kobaek-Larsen, A Koulaouzidis, R Kroijer, I Al-Najami, N Buch, A Høgh, N Qvist, MK Thygesen, and U Deding. Choice of colon capsule or colonoscopy versus default colonoscopy in FIT positive patients in the Danish screening programme: a parallel group randomised controlled trial. Gut, 74(10):1616–1623, 2025. doi: 10.1136/ gutjnl-2024-333687.

[28] Jane Bromley, James Bentz, Leon Bottou, Isabelle Guyon, Yann Lecun, Clif Moore, Eduard Sackinger, and Rookpak Shah. Signature verification using a “Siamese” time delay neural network. International Journal of Pattern Recognition and Artificial Intelligence, 7(4):669–688, 1993. doi: 10.1142/S0218001493000339.

[29] S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 1, pages 539–546, 2005. doi: 10.1109/CVPR.2005.202.

[30] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1597–1607, 2020. URL https://proceedings.mlr.press/v119/chen20j.html.

[31] Olga Ilina, Vadim Ziyadinov, Nikolay Klenov, and Maxim Tereshonok. A survey on symmetrical neural network architectures and applications. Symmetry, 14(7):1391, 2022. doi: 10.3390/sym14071391.

[32] Kai Zhang, Shouliang Qi, Jiumei Cai, Dan Zhao, Tao Yu, Yong Yue, Yudong Yao, and Wei Qian. Content-based image retrieval with a convolutional Siamese neural network: Distinguishing lung cancer and tuberculosis in CT images. Computers in biology and medicine, 140:105096, 2022. doi: 10.1016/j.compbiomed.2021.105096.

[33] Florian Schrof, Dmitry Kalenichenko, and James Philbin. FaceNet: A unified embedding for face recognition and clustering. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 815–823, 2015. doi: 10.1109/CVPR.2015.7298682.

[34] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.

[35] R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), volume 2, pages 1735–1742, 2006. doi: 10.1109/CVPR.2006.100.

[36] Tete Xiao, Xiaolong Wang, Alexei A. Efros, and Trevor Darrell. What should not be contrastive in contrastive learning. In International Conference on Learning Representations (ICLR), 2021. URL https://openreview.net/forum?id=CZ8Y3NzuVzO.

[37] Suorong Yang, Weikang Xiao, Mengchen Zhang, Suhan Guo, Jian Zhao, and Furao Shen. Image data augmentation for deep learning: A survey. arXiv preprint, 2023. doi: 10.48550/arXiv.2204.08610.

[38] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. doi: 10.1109/CVPR.2017.243.

[39] Mingxing Tan and Quoc Le. EficientNet: Rethinking model scaling for convolutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 6105–6114, 2019. URL https://proceedings.mlr.press/v97/tan19a.html.

## Appendix A: Size and morphology of polyps

Figures A.1 and A.2 show density and count plots of the polyp sizes of this study. Figure A.3 compares the polyp sizes of patients with a single polyp to those with more than one polyp.

A summary of the recorded morphology is displayed in Table A.1.

## Polyp transit times

Table A.2 shows the typical transit times (time from first to last sight of a polyp of a single passing). The fractions indicate how many of the polyp transit times are equal to or shorter than the given number of seconds.

![](images/492b07364ede068073eefd2dac7f7554e9eab3b4059e0c1f62df9071d162deab.jpg)  
Figure A.1: Density plot of polyp size.

![](images/c236b1ea745df0900ccc11fb9ba79b613905041b3cf34306fb05d84a33b68bbd.jpg)  
Figure A.2: Count plot of diferent polyp sizes.

Table A.1: Morphology of included polyps.
<table><tr><td colspan="2">Morphology Number of polyps</td></tr><tr><td>Broad-based/sessile</td><td>1375</td></tr><tr><td>Pedunculated</td><td>246</td></tr><tr><td>Sessile w/mucusol cap</td><td>15</td></tr><tr><td>Flat</td><td>14</td></tr><tr><td>Tumor</td><td>12</td></tr><tr><td>Other</td><td>27</td></tr><tr><td>Missing</td><td>12</td></tr></table>

![](images/0aff03d8d0ce76422af8aa1eb7f8e7a08dda204551df18ec8d2307159a0873ca.jpg)  
Figure A.3: Polyp size for patients with a single polyp versus those with more than one polyp.

Table A.2: Polyp transit times.
<table><tr><td>Seconds</td><td>Cumulative fraction</td></tr><tr><td>0</td><td>0.118</td></tr><tr><td>1</td><td>0.269</td></tr><tr><td>2</td><td>0.342</td></tr><tr><td>3</td><td>0.389</td></tr><tr><td>4</td><td>0.440</td></tr><tr><td>5</td><td>0.467</td></tr><tr><td>6</td><td>0.493</td></tr><tr><td>7</td><td>0.510</td></tr><tr><td>16</td><td>0.604</td></tr><tr><td>33</td><td>0.701</td></tr><tr><td>72</td><td>0.800</td></tr><tr><td>181</td><td>0.900</td></tr></table>

## Appendix B: Results using alternative backbones

The hyperparameters that difered from the ResNet setup (Table 2) were:

• For EficientNet B3 the images were resized from 256x256 to 300x300, the native input size of EficientNet B3.

• EficientNet B3 and DenseNet121 used learning rate $5 \times 1 0 ^ { - 5 }$ while the rest used $1 \times 1 0 ^ { - 4 }$

Tables B.1 and B.2 show the cross-validation and final test set performance measures, respectively, corresponding to Tables 3 and 4 for the ResNet backbones.

Table B.1: Cross-validation test set performance measures for diferent backbones. Mean values and $\pm \nobreakspace 1 \nobreakspace$ standard deviation across the 10 folds. AUC is area under the receiver operating characteristic curve.
<table><tr><td>Metric</td><td>DenseNet121</td><td>EfficientNet B0</td><td>EfficientNet B3</td></tr><tr><td>Accuracy</td><td> $0 . 9 1 7 2 \pm 0 . 0 0 9 2$ </td><td> $0 . 9 0 1 7 \pm 0 . 0 1 3 0$ </td><td> $0 . 9 0 9 6 \pm 0 . 0 1 4 3$ </td></tr><tr><td>Precision</td><td> $0 . 9 2 1 8 \pm 0 . 0 1 4 4$ </td><td> $0 . 8 9 8 4 \pm 0 . 0 1 9 6$ </td><td> $0 . 9 0 9 4 \pm 0 . 0 1 9 3$ </td></tr><tr><td>Recall</td><td> $0 . 9 1 2 1 \pm 0 . 0 1 1 6$ </td><td> $0 . 9 0 6 3 \pm 0 . 0 1 2 9$ </td><td> $0 . 9 1 0 3 \pm 0 . 0 2 2 8$ </td></tr><tr><td>Specificity</td><td> $0 . 9 2 2 4 \pm 0 . 0 1 5 4$ </td><td> $0 . 8 9 7 1 \pm 0 . 0 2 2 0$ </td><td> $0 . 9 0 8 8 \pm 0 . 0 2 2 2$ </td></tr><tr><td>F1 score</td><td> $0 . 9 1 6 8 \pm 0 . 0 0 9 0$ </td><td> $0 . 9 0 2 2 \pm 0 . 0 1 2 3$ </td><td> $0 . 9 0 9 6 \pm 0 . 0 1 4 6$ </td></tr><tr><td>AUC</td><td> $0 . 9 7 3 7 \pm 0 . 0 0 4 2$ </td><td> $0 . 9 6 7 3 \pm 0 . 0 0 9 4$ </td><td> $0 . 9 6 9 1 \pm 0 . 0 0 6 2$ </td></tr><tr><td>Optimal threshold</td><td> $0 . 2 3 1 0 \pm 0 . 0 2 0 7$ </td><td> $0 . 2 2 3 0 \pm 0 . 0 1 7 3$ </td><td> $0 . 2 2 1 0 \pm 0 . 0 2 2 6$ </td></tr><tr><td>Early stopped epoch</td><td> $1 2 . 5 \pm 5 . 1$ </td><td> $2 0 . 0 \pm 8 . 8$ </td><td> $1 2 . 1 \pm 2 . 1$ </td></tr></table>

Table B.2: Test set performance measures for diferent backbones. AUC is area under the receiver operating characteristic curve.
<table><tr><td>Metric</td><td>DenseNet121</td><td>EfficientNet B0</td><td>EfficientNet B3</td></tr><tr><td>Accuracy</td><td>0.9160</td><td>0.9010</td><td>0.9128</td></tr><tr><td>Precision</td><td>0.9029</td><td>0.8979</td><td>0.9238</td></tr><tr><td>Recall</td><td>0.9323</td><td>0.9049</td><td>0.8997</td></tr><tr><td>Specificity</td><td>0.8997</td><td>0.8971</td><td>0.9258</td></tr><tr><td>F1 score</td><td>0.9174</td><td>0.9014</td><td>0.9116</td></tr><tr><td>AUC</td><td>0.9745</td><td>0.9708</td><td>0.9736</td></tr><tr><td>Optimal threshold</td><td>0.20</td><td>0.20</td><td>0.22</td></tr><tr><td>Early stopped epoch</td><td>9</td><td>14</td><td>12</td></tr></table>

Figures B.1 and B.2 show the training curves with early stopping and when running for 500 epochs, respectively, corresponding to Figures 3 and 4.

![](images/c1eff6c9b67da957493f75231104aff647a2ed6cd6cb573bdd8256bb9a155431.jpg)

![](images/5d5bc0ee88c39a2496c7018a21c9b93ff0cdfc06e051d5bbac06a7fba72c03b4.jpg)

![](images/e65ec04acb7b5e16355c15430fc2f0d96d6f5a8698f93037807ca79c9e4559d6.jpg)  
Figure B.1: Training and validation loss using cosine distance metrics, and DenseNet121, EficientNet B0 and EficientNet B3 from left to right. Best model following triggering of early stopping marked in red.

![](images/a85e8fb9213c0428708fdc865088fb7ab2fcacca5dfe9a4e6d0339177bf31ff4.jpg)

![](images/4085f518374fcb9d30d664b33fc2ecc3d5d3b60aeeea98122e06f6b7b5c3503e.jpg)

![](images/c85229ed28818dac0e392ab58cd0a047b2d0ccd8f3d69d3925aff1dd9b3d32db.jpg)  
Figure B.2: Training and validation loss when running for 500 epochs. DenseNet121, EficientNet B0 and EficientNet B3 from left to right.

## Appendix C: Distribution of cosine vector distances

Figures C.1 and C.2 show the distribution of cosine vector distances for the “same polyp” and “diferent polyp” classes for the diferent backbones for the test sets of the cross-validation setup and for the final full model test set, respectively.

![](images/4a9fe0d8945d38e4adc530740d96cc6e35527dcffb67a9c6910f9a692caa56a0.jpg)  
Figure C.1: Distribution of cosine vector distances of the 10 test sets of the cross-validation for all six backbones.

![](images/70d64c5c5f49d9c48103fc54ff916bce678710e5fbd617fa155a9df271bcb1e9.jpg)

![](images/577fb69ed341e6b6283ef9bb2953ce1d5960cc4a4b036d86ec1baab3a35bed73.jpg)

![](images/b5df29133a77865399f234e2f6355762c8a0aba1561b8d5bbbfe68c914b9a6eb.jpg)

![](images/3c03d31cd71c92a8795d26170bbce5c746f7476bd6cb5ba6fa58e6a0c2cba3f1.jpg)

![](images/f3d5378921fd16a06f2772947a0391a7f47f1ba8da15c23196e1131b6f5e8104.jpg)

![](images/56ae3e7f5f44025369bfbd795a8719520b2a716ab8bd159cb911c10f7e353563.jpg)  
Figure C.2: Distribution of cosine vector distances of the final test set for all six backbones.