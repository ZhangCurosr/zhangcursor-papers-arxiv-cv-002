# INTERPRETABLE DEEP LEARNING MODELS FOR BREAST CANCER DIAGNOSIS USING MEDICAL IMAGING

Sami Alnawali<sup>∗</sup> Information Systems Department College of Computer Science and Information Technology King Faisal University, Alahsaa, Saudi Arabia salnawali@kfu.edu.sa

Abdul Hadi<sup>†</sup> Department of Robotics & Artificial Intelligence Shaheed Zulfikar Ali Bhutto Institute of Science and Technology Islamabad, Pakistan abdulhaadii146@gmail.com

Akhtar Jamil Department of Computer Science National University of Computer and Emerging Sciences Islamabad, Pakistan akhtar.jamil@nu.edu.pk

## Muhammad Atif Saeed<sup>†</sup>

Department of Artificial Intelligence & Data Science National University of Computer and Emerging Sciences Islamabad, Pakistan atif.saeed@isb.nu.edu.pk

Azka Atiq<sup>†</sup> Department of Artificial Intelligence & Data Science National University of Computer and Emerging Sciences Islamabad, Pakistan azka.atiq@isb.nu.edu.pk

Ahmad Din Department of Artificial Intelligence & Data Science National University of Computer and Emerging Sciences Islamabad, Pakistan ahmad.din@nu.edu.pk

## ABSTRACT

Breast cancer continues to be a leading cause of mortality among women globally, highlighting the necessity for precise and comprehensible diagnostic tools. Although deep learning demonstrates significant potential, several current methodologies emphasize accuracy at the expense of interpretability, hence limiting their clinical applicability. To tackle this issue, we established a thorough framework that assesses ten advanced transfer learning models ResNet50, VGG19, InceptionV3, DenseNet121, Xception, MobileNetV2, EfficientNetV2S, EfficientNet-B7, Vision Transformer, and Swin Transformer, using the BreakHis dataset for automated breast cancer classification. All models underwent pretraining on ImageNet and were subsequently fine-tuned using bespoke classification heads, strengthened by comprehensive preprocessing and data augmentation methods including rotation, flipping, zooming, normalization, and contrast enhancement to increase generalization. To improve transparency, explainable AI (XAI) techniques such as Grad-CAM, LIME, SHAP, and LRP were utilized to display distinguishing histological features, emphasizing nuclei-rich and glandular areas aligned with clinical significance. Experimental results indicated that DenseNet121 attained superior performance, achieving a test accuracy of 99.4% and an AUC-ROC of 0.9997 across various magnifications, whilst EfficientNet also exhibited robust performance with an accuracy of 98.5% and continuously low false positive rates. Transformer-based models as Swin Transformer and ViT demonstrated strong generalization in intricate scenarios, while the lightweight MobileNetV2 provided competitive performance for implementation in resource-constrained settings. This study demonstrates the efficacy of integrating advanced transfer learning with explainable artificial intelligence to provide clinically relevant and interpretable frameworks for breast cancer detection. Future work will extend this framework to include multi-modal datasets, detailed tumor subtyping, and more advanced explainable artificial intelligence techniques to augment diagnostic reliability and facilitate clinical adoption.

Keywords Breast cancer · Histopathology · BreakHis · Transfer learning · Vision transformers · Explainable AI

## 1 Introduction

Breast cancer is among the most common and lethal malignancies impacting women globally. The World Health Organization reports over 2.3 million new cases and over 680,000 deaths per year, establishing it as a primary cause of cancer-related mortality in women[1]. Timely and precise identification of breast cancer is crucial for enhancing treatment efficacy, survival probabilities, and the overall quality of life for patients[2]. Traditional diagnostic methods, including mammography, histological examination, and ultrasound imaging, while useful, are prone to inter-observer variability, necessitate professional interpretation, and may overlook small abnormalities in the early stages[3]. These constraints have catalyzed the integration of artificial intelligence (AI) and deep learning (DL) methodologies in medical imaging to aid radiologists in achieving precise and prompt diagnosis[4], [5].

Deep learning, especially convolutional neural networks (CNNs), has demonstrated exceptional efficacy in visual recognition tasks, encompassing medical image classification, segmentation, and detection[6]. Numerous research have utilized CNN architectures, including VGG, ResNet, Inception, and DenseNet, for breast cancer classification, with remarkable accuracy[1]. Nonetheless, these models frequently necessitate substantial quantities of annotated training data and considerable computational resources, posing challenges in the medical field where labeled data is limited and costly to obtain[7]. Furthermore, deep learning models are often criticized for their opacity and lack of interpretability, presenting a substantial obstacle to clinical implementation[8]. Comprehending the rationale behind a model’s decision-making process is essential in healthcare applications, as erroneous predictions can lead to severe repercussions[9].

To mitigate data scarcity and enhance model generalisability, transfer learning has emerged as an effective technique that facilitates the utilization of pretrained models on extensive datasets (e.g., ImageNet) for domain-specific applications such as breast cancer classification[10], [11]. Transfer learning decreases computational demands and training duration while improving performance on limited, specialized medical datasets. The use of Explainable Artificial Intelligence (XAI) techniques, including Grad-CAM, LIME, SHAP, and XRAI, has created new opportunities for visualizing model behavior, enhancing clinician trust, and streamlining regulatory approval[7], [12].

This paper presents a comprehensive and interpretable deep learning framework that integrates various transfer learning models with explainable artificial intelligence methodologies for robust breast cancer classification[8]. We assess ten cutting-edge models, ResNet50, VGG19, InceptionV3, DenseNet121, Xception, MobileNetV2, EfficientNetV2S, EfficientNet-B7, Vision Transformer (ViT), and Swin Transformer, across four prevalent breast cancer datasets[13]. To improve model performance and generalisation, we employ customised data augmentation methods, normalisation, and patch-based image preprocessing[4]. Additionally, we incorporate various XAI techniques to deliver both local and global interpretability, assisting physicians in comprehending the reasoning behind the model’s predictions[3].

The subsequent sections of this work are structured as follows: Section 1 delineates the pertinent literature and recent progress in deep learning for breast cancer diagnosis. Section 2 delineates the datasets employed, encompassing their structure, clinical context, and preparation procedures. Section 3 delineates the methodology, encompassing model architectures, transfer learning procedures, data augmentation techniques, and explainability methods. Section 4 delineates the experimental configuration and assessment criteria. Section 5 examines the outcomes derived from various models and datasets. Section 6 ultimately summarises the paper by presenting insights, limitations, and prospective future directions.

## 2 Literature Review

Breast cancer remains a major global health issue and one of the leading causes of cancer-related deaths among women. Early and precise detection is still crucial for improving prognosis, providing timely treatment, and enhancing survival rates[6]. In recent years, deep learning (DL) algorithms have demonstrated exceptional potential for automating breast cancer diagnosis by understanding complicated patterns from medical pictures. However, a number of difficulties remain that prevent widespread clinical implementation[13]. Key drawbacks include a lack of model generalisability across varied patient groups, biases caused by uneven or insufficient datasets, and a lack of transparency in decision-making. Many contemporary DL models behave like "black boxes," making it difficult for clinicians to trust and interpret their predictions[6]. To fill these gaps, we present a complete system that combines a variety of pre-trained transfer learning modelsj, including EfficientNetV2S, InceptionResNetV2, and XceptionNet, among others, with explainable AI (XAI) techniques such as LIME and Faster Score-CAM[5]. This integration intends to improve classification performance while also providing interpretable insights into the model’s predictions, allowing for better informed and reliable clinical decision-making.

Silva-Aravena et al. [2023][14] introduced a hybrid machine learning and explainable artificial intelligence system for the early prevention of breast cancer, utilizing risk factor analysis on an Indonesian dataset of 400 persons (200 cases and 200 controls). The system utilized an XGBoost ensemble, compared with logistic regression, SVMs, and random forests, following the CRISP-ML(Q) framework to guarantee transparency and quality[7]. Preprocessing encompassed feature encoding and stratified division of lifestyle and reproductive variables, resulting in 85% accuracy. SHAP explanations identified high-fat diet, absence of breastfeeding, and familial predisposition as significant risk factors; but, the study’s limited, self-reported dataset constrains generalisability.

Nasir et al. [2025][15] performed a comparative deep learning analysis on the Breast Cancer Wisconsin Diagnostic Dataset (569 cases, 33 tumor characteristics), assessing CNNs, RNNs, MLPs, and hybrid models (CNN+LSTM, CNN+MLP), in addition to pretrained VGG16, ResNet, and EfficientNet[4]. Preprocessing encompassed normalization, feature correlation analysis, and stratified partitioning, while data insufficiency was mitigated through augmentation (rotation, flipping, contrast adjustment), SMOTE, and GAN-based synthesis. Grad-CAM and SHAP were utilized for interpretability, with CNN+LSTM attaining the highest accuracy (98.2%), illustrating the efficacy of integrating spatial and sequential feature learning. Nonetheless, the study was constrained by the size of the dataset, the lack of image inputs, and diminished generalizability to clinical imaging[7].

Patho-Net [16] introduced a GRU-augmented U-Net segmentation framework for breast cancer histopathology, utilizing the BreakHis dataset of 2,048 pictures at 100× magnification. The pipeline utilized Reinhard color normalization with automated reference selection, EANLM noise reduction, and I-GLCM-based feature extraction, culminating in GRU-based classification[3]. It attained an accuracy of 98.9%, precision and recall of 98.6%, and specificity of 99%, with Grad-CAM and XRAI augmenting interpretability. Notwithstanding its robust performance, the model was confined to binary classification and depended on a resource-intensive preprocessing procedure, hence limiting cross-domain generalization.

Kabir et al. [2025] [17] introduced a deep learning framework for the classification of breast cancer utilizing the BUSI ultrasound dataset, which comprises 780 pictures categorized as benign, malignant, and normal. Four pretrained CNNs—EfficientNetV2S, EfficientNetV2M, InceptionResNetV2, and XceptionNet[1] were fine-tuned utilizing scaling, flipping, rotation, zooming, and stratified splitting techniques. To enhance interpretability, Faster Score-CAM and LIME were utilized, emphasizing critical diagnostic areas. EfficientNetV2S attained superior performance (91.02% accuracy, precision 0.92, recall 0.91, F1-score 0.91); however, dataset size, class imbalance, and absence of external validation limited generalizability[18].

Talukder [2025] [19] proposed an improved DenseNet169 model for breast cancer histopathology, validated on the BreakHis (40×–400×), BACH, and BHI datasets. Preprocessing encompassed resizing (256×256), sharpening, normalization, and label encoding, along with augmentation by ±20<sup>◦</sup> rotation, shifting, and flipping. Class Activation Mapping (CAM) and saliency maps were utilized for interpretability. The model attained 99.5% accuracy on BreakHis 40×, 98.8% on 100×, 97.3% on 200×, 96.9% on 400×, 94.8% on BACH, and 88% on BHI; nonetheless, difficultie remained in categorizing benign cases at elevated magnifications, highlighting the necessity for clinical validation.

Sharmi et al. [2025] [20] proposed an explainable framework for breast cancer diagnosis utilizing thermal images from the DMR dataset, comprising 2,010 samples. The pipeline integrated Attention U-Net for segmentation, K-Means clustering for region extraction, and EfficientNet-B7 for classification, incorporating preprocessing processes such as resizing (224×224), masking, and array conversion[5]. LIME was utilized for interpretability based on superpixels. The model attained a training accuracy of 96.5%, a validation accuracy of 91.7%, a recall of 91.9%, a specificity of 91.4%, a precision of 89.9%, and an F1-score of 90.9%. However, its generalizability was constrained by class imbalance, demographic bias, and insufficient cross-validation[12].

An ensemble-based transfer learning method[21] for breast cancer diagnosis utilizing ultrasound imagery was presented in IEEE Access [2024]. The research employed a dataset including 780 photos classified into benign, malignant, and normal categories, obtained from UCI and Kaggle. The authors employed an ensemble stacking technique to improve classification, integrating pretrained CNNs (InceptionV3, VGG16, VGG19) with machine learning classifiers, including MLPs and SVMs[8]. Preprocessing encompassed resizing, cropping, normalization, and data supplementation to enhance image quality and model resilience. The optimal ensemble model, utilizing InceptionV3 with stacking, attained an accuracy of 85.77% and an AUC of 0.947, underscoring its diagnostic efficacy[22]. Nonetheless, the approach failed to incorporate explainable AI (XAI) methodologies such as Grad-CAM or SHAP, which are essential for clinical transparency, and its scalability to extensive, heterogeneous ultrasound datasets has yet to be investigated[9].

## 3 Research Gap

Although substantial advancements have been made in deep learning for breast cancer classification, current research frequently prioritizes accuracy over interpretability, so limiting medical applicability. The integration of XAI approaches (e.g., Grad-CAM, LIME, SHAP) is superficial and inconsistent among architectures, while dependence on a singular dataset limits generalizability. Histopathology datasets, like BreakHis and BACH, exhibit class imbalance, while mammography datasets such as CBIS-DDSM and INbreast are inadequately employed for cross-model comparisons. Moreover, there is a paucity of benchmarks involving both CNNs and transformer-based models (ResNet, EfficientNet, MobileNet, Xception, ViT, Swin Transformer), and the synergistic impact of augmentation and preprocessing on interpretability remains little examined. This study fills these gaps by doing a thorough assessment of transfer learning models utilizing several XAI methodologies to establish a transparent, precise, and clinically dependable diagnostic framework.

## 4 Proposed Methodology

## 4.1 Dataset

In this paper, we will use the BreakHis (Breast Cancer Histopathological Image Classification) dataset which is one of the most popular benchmarks of 7,909 images of 82 patients with 40, 100, 200, and 400 magnifications of both benign and malignant tumor cells. In order to guarantee methodological rigor and prevent data leakage, a patient-level strict splitting strategy was utilized, and pictures of a patient were only used in training (70%), validation (20%), or testing (10%) sets, which ensured that no overlap occurred between splits and simulated clinical conditions in real-life. The data has an inherent imbalance of classes and this was overcome by using weighted sampling methodology in training. All the images were subjected to a standardized processing procedure that included resizing (224x224 or 299x299 based on the needs of the model), ImageNet normalization, and augmentation as rotation, flipping, zooming, and contrast adjustment to enhance generalization. Also, evaluations of the magnification specifics were performed to measure the robustness of various magnifications of images. In spite of these measures, there can still be intra-patient and inter-magnification similarities in the dataset, thus, can bring subtle correlations, this is a limitation that is noted in order to make the results transparently reported and carefully interpreted.

## 4.2 Data Augmentation

This study utilizes customized data augmentation techniques to enhance generalization and reduce overfitting in deep learning models for breast cancer classification, addressing the lack of labeled medical images and class imbalance[23]. Augmentations have been proposed to maintain diagnostic integrity while improving robustness: rotation (±20<sup>◦</sup>) to establish rotational invariance, horizontal/vertical flipping to enhance spatial diversity[3], shearing to replicate distortions from scanning or tissue preparation, zooming (±20%) to represent scale variations across magnifications[23]., and contrast adjustment to accommodate discrepancies in staining and lighting conditions[24]. Bilinear interpolation was employed during scaling and translating to maintain smooth transitions and reduce aliasing problems[12]. These augmentations collectively mimic real-world variability, allowing models to learn invariant, clinically significant characteristics that enhance classification performance across various imaging settings.

## 4.3 Transfer Learning

Transfer learning has become a potent approach in medical image analysis, especially in contexts with scarce labelled data. Transfer learning utilises deep neural networks pretrained on extensive datasets like ImageNet to generate robust, generalisable characteristics that may be refined for specific applications, such as breast cancer prediction. This work utilised many advanced pretrained models, including ResNet, VGG, DenseNet, EfficientNet, Xception, MobileNet, and vision transformer architectures, to evaluate their efficacy in breast cancer detection and classification. The models were modified by substituting their final classification layers and fine-tuning specific layers on target datasets to enhance performance for the particular diagnostic task.

## 4.3.1 DenseNet121

This study utilized DenseNet-121, a densely connected convolutional neural network pre-trained on ImageNet, as the foundation for classifying breast cancer histopathology images. The architecture utilizes extensive connectivity, allowing each layer to receive inputs from all prior levels, thus enhancing gradient flow, optimizing parameter efficiency, and alleviating vanishing gradient problems. The backbone consists of four dense blocks (6, 12, 24, and 16 layers, growth rate = 32) interspersed with transition layers utilizing 1×1 convolutions and 2×2 average pooling, culminating in global average pooling[25], [26]. A bespoke classifier head was incorporated, comprising a fully connected layer (1024→512) with ReLU activation, dropout (p=0.5) for regularization, and a concluding dense layer (512→2) generating logits for binary classification (benign versus malignant)[2]. Input photos were downsized to 224×224, normalized, then augmented by random rotations, shifts, scaling, and flips to improve generality. The model underwent fine-tuning utilizing Label Smoothing Cross-Entropy loss (ε=0.1), optimized via AdamW (weight decay = 0.01) with differential learning rates, and trained employing Cosine Annealing Warm Restarts[25], [26]. Supplementary regularization encompassed RandAugment and gradual layer unfreezing. Training was conducted with a batch size of 32 over 50 epochs, incorporating early stopping (patience = 15, monitored on validation F1-score).

## 4.3.2 EfficientNet

This study utilized EfficientNet, a scalable convolutional neural network created by Google AI, owing to its exceptional balance between accuracy and computational efficiency in medical image classification[10], [27]. The model uses the compound scaling method, which uniformly adjusts network depth, width, and input resolution, and is constructed using MBConv (Mobile Inverted Residual Bottleneck) blocks integrated with squeeze-and-excitation (SE) modules for channel-wise recalibration. We specifically fine-tuned EfficientNet-B0 and its larger variants (EfficientNetV2S and EfficientNet-B7), which were pretrained on ImageNet, by substituting the classifier head with a bespoke binary classification architecture[28]. The backbone generated 1280 feature maps, which underwent global average pooling, succeeded by dropout (p = 0.5), a fully connected layer (1280 → 512, ReLU), an additional dropout, and a concluding linear layer (512 → 2) to yield logits for benign and malignant classifications. The training utilized Cross Entropy Loss with label smoothing and softmax activation during inference. To mitigate overfitting and enhance resilience, input photos were downsized (224×224 or 380×380), normalized, and subjected to augmentation by random flipping, zooming, shifting, and rotation with Keras’ ImageDataGenerator[28]. The model underwent training with a progressive unfreezing of the pretrained backbone and varying learning rates, resulting in roughly 4 million trainable parameter post fine-tuning.

## 4.3.3 ConvNeXt

Using the BreakHis dataset, we optimized ConvNeXt, a state-of-the-art CNN architecture that draws inspiration from ResNet and Vision Transformers, for the purpose of breast cancer histological picture categorization. ConvNeXt maintains CNN inductive biases, such as locality and translation equivariance, while integrating transformer-inspired design elements, including extensive kernel depthwise convolutions, inverted bottlenecks, Layer Normalization, GELU activation, stochastic depth, AdamW optimization, and cosine learning rate scheduling[29], [30]. We employed the ConvNeXt-Tiny/Base variation pretrained on ImageNet-21k and substituted the classification head with a bespoke architecture using LayerNorm, GELU activation, dropout (p=0.5), and fully connected layers (768 → 512 → 2) for the binary classification of benign and malignant instances. Input photos were reduced to 224×224 pixels, normalized with ImageNet statistics, and augmented by RandAugment (n=2, m=10), random cropping, color jittering, flipping, and rotation, while test-time augmentation (TTA) was utilized for enhanced inference robustness[30]. Training utilized label-smoothing cross-entropy (smoothing=0.1), the AdamW optimizer (weight decay=0.01), cosine annealing for learning rate scheduling, and mixed-precision training, alongside a gradual unfreezing of backbone layers to modify pretrained features[31]. The model comprised around 28.6 million trainable parameters.

## 4.3.4 ResNet

This study utilized the fine-tuning of a ResNet-50 backbone, pretrained on ImageNet, for the classification of breast cancer utilizing the BreaKHis dataset. The input photos were downsized to 224×224 RGB, normalized using ImageNet statistics, and augmented by flipping, rotation, zooming, and RandAugment[32]. The backbone (about 23.5 million parameters) was initially fixed and subsequently unfixed with varying learning rates, while the custom classifier head (approximately 1.05 million parameters) comprised fully connected layers of 2048→512→2 with ReLU activation and a dropout rate of 0.5[33]. The training employed CrossEntropyLoss with label smoothing, the Adam optimizer, and cosine annealing with warm restarts, while the assessment metrics comprised accuracy, precision, recall, specificity, F1-score, and AUC-ROC[32]. To improve interpretability, Grad-CAM, LIME, SHAP, and LRP were combined for visualizing decision boundaries.

## 4.3.5 InceptionV3

The study used InceptionV3, a deep convolutional neural network architecture designed for effective multi-scale feature extraction, as a transfer learning model for breast cancer classification[34]. InceptionV3 enhances the original GoogLeNet by integrating factorized convolutions, auxiliary classifiers, asymmetric filters (1×7 and 7×1), and batch normalization, hence facilitating computational efficiency and superior generalization. The model employs Inception modules that concurrently process various convolution filters (1×1, 3×3, 5×5), effectively capturing both local and global representations[1]. In our approach, the pretrained backbone was initialized using ImageNet weights, and the final classification head was substituted with a dense block (2048→512→2) with ReLU activation and dropout (p=0.5), in addition to a modified auxiliary classifier. Histopathology photos were downsized to 299×299, normalized using ImageNet statistics, and augmented with RandAugment, including cropping, flipping, rotating, zooming, and color jittering[34]. Training utilized the AdamW optimizer, label smoothing cross-entropy, cosine annealing warm restarts, mixed precision, and a gradual unfreezing of backbone layers for stable fine-tuning[1].

## 4.3.6 VGG19

This research used VGG-19, pretrained on ImageNet, as a transfer learning model for the classification of breast cancer. The backbone, consisting of 16 convolutional and 5 pooling layers (about 20 million parameters), was initially frozen and subsequently fine-tuned using differential learning rates and a gradual unfreezing approach[35]. The initial head was substituted with a bespoke classifier ( 102.8M parameters) comprising Linear (25088→4096→4096→512→2) layers, ReLU activations, and several dropout layers (p=0.5) for regularization purposes. The input photos were downsized to 224×224, normalized, and augmented using RandAugment, along with cropping, flipping, rotation, and color jittering[10]. The training employed Label Smoothing Cross-Entropy (smoothing=0.1), the AdamW optimizer, cosine annealing warm restarts, AMP mixed-precision, and early halting, resulting in about 122.8 million trainable parameters optimized for histopathology picture classification[35].

## 4.3.7 MobileNetV2

This research utilized MobileNetV2, a lightweight convolutional neural network optimized for processing efficiency by depthwise separable convolutions and inverted residual bottlenecks with linear projections[31]. The model accepts 224×224 RGB inputs, commencing with a 3×3 convolution (32 filters, stride 2), succeeded by a series of 17 bottleneck layers incorporating expansion through 1×1 convolutions, depthwise filtering, and projection layers. Each block integrates ReLU6 activations and residual shortcuts when the input and output dimensions are congruent[1]. The backbone generates 1280 feature maps, which are subsequently processed by a global average pooling layer and a bespoke classifier head (Linear 1280→512, ReLU, Dropout 0.5, Linear 512→2) for binary classification. MobileNetV2, with around 3.5 million parameters, attains a balance between precision and low complexity, rendering it appropriate for medical imaging applications where efficiency and scalability are paramount[36], [24].

## 4.3.8 Vision Transformer (ViT)

This study utilized a Vision Transformer (ViT-B/16) pretrained on ImageNet-21k and subsequently fine-tuned for breast cancer categorization[37]. Input images (3×224×224) were segmented into 16×16 non-overlapping patches, resulting in 196 tokens embedded in 768-dimensional vectors, together with an additional learnable class token and positional embeddings[38]. The sequence underwent processing through 12 transformer encoder blocks, each consisting of multi-head self-attention (12 heads), MLP layers (hidden=3072, GELU), LayerNorm, residual connections, and dropout (p=0.1). The ultimate representation underwent LayerNorm, followed by a bespoke classifier head incorporating LayerNorm, Linear (768→512), GELU, Dropout, and Linear (512→2) for binary classification[39]. The architecture comprised over 86 million parameters, with around 400,000 derived from the custom head, facilitating efficient finetuning on histopathological pictures while maintaining the ViT’s robust feature extraction and contextual modeling abilities[36], [40].

## 4.3.9 Swin Transformer (Shifted Window Transformer)

This research used the Swin Transformer Tiny, a hierarchical vision transformer pretrained on ImageNet-1K, for the classification of breast cancer images[41], [40]. Input images (3×224×224) were divided into 4×4 patches and transformed into 768-dimensional tokens, which underwent four steps of shifted window-based multi-head selfattention (W-MSA, SW-MSA), residual connections, LayerNorm, and MLPs with GELU activations. Patch merging systematically diminished resolution while augmenting channel depth (768→1536→3072)[41]. The backbone output, a 3072-dimensional feature vector, was consolidated using global average pooling and forwarded to a bespoke classifier head consisting of Linear (768→512), ReLU, Dropout (0.5), Linear (512→256), ReLU, Dropout (0.5), and Linear (256→2) for the prediction of benign versus malignant outcomes[29]. The architecture had approximately 28.3 million parameters, with around 0.5 million being trainable during the initial fine-tuning phase, integrating transformer-based global context modeling with computational efficiency appropriate for high-resolution medical imaging[29].

## 4.3.10 Xception

The study utilized Xception (Extreme Inception), a deep convolutional neural network pretrained on ImageNet, modified for binary breast cancer classification[16]. Xception substitutes conventional convolutions with depthwise separable convolutions integrated with residual connections, facilitating efficient feature extraction. The network processes 299×299 RGB inputs via an entering flow of convolutional and separable convolutional layers (32–728 channels), succeeded by a middle flow including 8 repeating blocks (728 channels), and concluding with an exit flow that expands to 2048 channels[42]. The backbone output undergoes global average pooling, dropout (p=0.5), and a concluding linear layer (2048→2) for the classification of benign and malignant cases. The training utilized the AdamW optimizer, cosine annealing warm restarts, and label-smoothed cross-entropy loss, rendering the model both computationally efficient and highly expressive for histopathology picture processing[31].

## 4.4 Explainable Artificial Intelligence (XAI)

The black-box aspect of deep learning models presents major obstacles to their clinical adoption, where trust and transparency are of the utmost importance, notwithstanding their impressive accuracy in breast cancer detection. Explainable Artificial Intelligence (XAI) methodologies seek to mitigate this constraint by offering human-comprehensible insights into model predictions[43]. This study utilised XAI technologies, including Grad-CAM, LIME, SHAP, and XRAI, to visualise and elucidate the decision-making processes of diverse deep learning models. These tools not only improve model transparency but also aid doctors in verifying the regions or features most significant in diagnosis results.

## 4.4.1 Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM (Gradient-weighted Class Activation Mapping) was utilized to improve the interpretability of deep learning models by producing class-discriminative heatmaps that emphasize the picture regions most significant for classification[37, 44]. It calculates the gradients of the projected class score relative to the feature maps of the final convolutional layer, use global average pooling to obtain channel weights, and generates a ReLU-activated weighted sum to create the final heatmap. The heatmap is subsequently overlaid on the original image, clearly highlighting areas essential for decision-making[28, 31]. This study employed Grad-CAM across various architectures (ResNet50, DenseNet121, Xception, EfficientNetV2, Swin Transformer) to assess the models’ attention to aberrant tissue patterns. Its post hoc nature, necessitating no architectural alterations, renders it especially appropriate for medical imaging applications where trust and transparency are important[29].

## 4.4.2 LIME (Local Interpretable Model-Agnostic Explanations)

The study used LIME (Local Interpretable Model-Agnostic Explanations) to produce localized, case-specific elucidations of breast cancer classification models. In contrast to gradient-based approaches, LIME is model-agnostic, approximating intricate predictions with a straightforward interpretable surrogate by perturbing inputs into superpixels and analyzing output fluctuations[9]. Each image underwent masking or modification, and a local linear model was applied to assess the significance of various image segments. LIME was utilized on models such as ResNet50, Efficient-Net, DenseNet121, and Swin Transformer, emphasizing the areas most critical for benign or malignant predictions. This offered clear, instance-specific interpretability, augmenting Grad-CAM by elucidating local decision limits that global saliency maps would neglect, thus enhancing clinical confidence in AI-assisted breast cancer diagnosis[45].

## 4.4.3 SHAP (SHapley Additive Explanations)

SHAP was used to measure feature significance through principles of cooperative game theory, assigning the contribution of each input region to the model’s output. By estimating Shapley values, SHAP equitably allocates prediction influence among picture superpixels or patches[46]. This work employed SHAP on models such as ResNet50, EfficientNet, DenseNet121, and Swin Transformer, generating heatmaps that emphasized areas influencing malignant or benign predictions. In contrast to gradient-based techniques, SHAP delivers globally consistent and theoretically robust explanations, furnishing doctors with insights into the influence of each input feature on decision outcomes[24].

## 4.4.4 Integrated Gradients (IG)

Integrated Gradients were used to attribute model predictions by calculating the path integral of gradients between a baseline (e.g., a black images) and the input image. This method offers a systematic methodology for allocating contribution scores to each pixel by averaging gradients along the interpolation trajectory[46]. This study employed IG on CNN and transformer-based architectures to elucidate subtle tissue patterns affecting benign-malignant classification. Its efficacy is rooted in generating smoother and more accurate attributions than raw gradients, rendering it particularly suitable for medical imaging, where detailed interpretability is essential[43].

## 4.4.5 Layer-wise Relevance Propagation (LRP)

LRP was utilized as a backpropagation-based explanation technique, redistributing the prediction score backwards through the network to identify the input areas that were most responsible for the decision[47]. Using relevance conservation concepts, LRP assigns pixel-level contributions that aggregate to the model’s output, producing comprehensive

and intuitive visual representations. LRP was incorporated into CNNs (ResNet, DenseNet, Xception) for the analysis of breast cancer images, emphasizing significant histological features. LRP’s ability to deliver detailed pixel attributions improves clinical confidence by guaranteeing that diagnostic predictions correspond with medically relevant structures.

## 5 Experimental Details

## 6 Results and Discussion

## 6.1 Datasets

The BreakHis dataset is a collection of histopathological pictures created by the P&D Lab in Brazil, comprising 7,909 microscopic images of breast tissue samples from 82 patients, classified as benign or malignant. Images are obtained at four magnification levels: 40X, 100X, 200X, and 400X, utilising an Olympus BX-50 microscope. The dataset has eight subtypes: four benign (adenosis, fibroadenoma, phyllodes tumour, tubular adenoma) and four malignant (ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma). The dataset exhibits imbalance and significant intra-class variability, rendering it appropriate for training and assessing deep learning models in fine-grained classification tasks. Each image is in RGB format and possesses a resolution of 700×460 pixels.

## 6.2 Experimental Results

Table 1: Performance of DenseNet-121 on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>89.2</td><td>90.7</td><td>95.5</td><td>93.0</td><td>70.3</td><td>0.936</td></tr><tr><td>Test (All Mag.)</td><td>99.4</td><td>99.9</td><td>99.5</td><td>99.6</td><td>99.2</td><td>0.999</td></tr><tr><td>Test (40×)</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>1.000</td></tr><tr><td>Test (100×)</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>1.000</td></tr><tr><td>Test (200×)</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>1.000</td></tr><tr><td>Test (400×)</td><td>97.5</td><td>99.4</td><td>97.7</td><td>98.5</td><td>96.4</td><td>0.998</td></tr></table>

DenseNet-121 performed well on the BreakHis dataset, showing the ability to generalize across magnifications. On the validation set, it achieved 89.24% accuracy, 90.72% precision, 95.46% recall, 70.31% specificity, an F1-score of 93.03%, and an AUC-ROC of 0.9362, indicating high sensitivity but some misclassification of benign cases(Table 1 & Figure 1). On the test set (at all magnifications), the model achieved near-perfect results with 99.41% accuracy, 99.86% precision, 99.45% recall, 99.17% specificity, 99.65% F1-score, and 0.9997 AUC-ROC, indicating strong generalization and balanced performance across both malignant and benign classes (Figure 2 & 3). The model reliably extracted discriminative histopathological characteristics at individual magnification settings (40X, 100X, and 200X) with 100% accuracy across all measures (Figure 4). These findings show the effectiveness of DenseNet-121’s densely connected architecture in reducing vanishing gradients, encouraging feature reuse, and providing highly reliable breast cancer classification performance on histopathology pictures.

![](images/402e7d73714a7ead7a6a8fec52939a9d3e00b98f6305b66f6a7230ed7cd044b5.jpg)

![](images/3a1f73ee7b251f87fad80dfff3f16ae42e5ad7f31731abdec62ff1d7bf37643e.jpg)

![](images/df5f1f3f46df47b9f07696be6584c5cf77150ac3484291d794ddb843c150d3a8.jpg)

![](images/dcca0fad734544ac0528fd016ffb200073ef4c45620ab609606d83aedde87e2f.jpg)  
Figure 1: Performance metrics comparison of DenseNet121 on the BreakHis dataset across validation and test sets at different magnifications (40X, 100X, 200X, 400X).

![](images/f6a6fddb09e92c0593682a8258e9111de383f759a61784e7ecef7cd5eab0c148.jpg)  
(a) 40X

![](images/ed063bcc0301c7861208bd38453b101ccb77f47db89f9d768c012e4c1bf834d4.jpg)  
(b) 100X

![](images/de1e1a17e7fa104b44dcff6b0310bf60378ee96eb75209e9444015ddb129d153.jpg)  
(c) 200X

![](images/0b06f38bc33c6d7a945321f88e1078c57b5431232f5134acfc47a5c219ee760b.jpg)  
(d) 400X  
Figure 2: Confusion matrices of the DenseNet-121 model across 40X, 100X, 200X, and 400X magnifications on the BreakHis dataset.

![](images/443e7064c95102cf8d38aaec2d38f72c3f2bcc8c8c216c435a6224c7c58d313b.jpg)  
(a) Validation ROC

![](images/04cdbbf6f31c4d982f21d97c09624fe25605c4d7972c713ef92f1899f2738353.jpg)  
(b) Validation Confusion Matrix

![](images/212a81ba88d1476f3b5817dd33bfdae10e7000c210cfd6bc733c16fdaf81305a.jpg)  
(c) Test ROC

![](images/907a0005743d25663c7bacffc38833635b74afd56aa1dad7c244c42d1fd8d980.jpg)  
(d) Test Confusion Matrix  
Figure 3: Validation and test performance of the DenseNet-121 model on the BreakHis dataset across all magnifications.

![](images/d02f1558b57f1fe4b76d0d38d56a22b70b9a4320b75dc09f53203f5ae8c4ef07.jpg)  
(a) ROC 40X

![](images/9bf4667470ed4136de4e498327e95c581e5d45e248dc8a7ed380d36eec3f333c.jpg)  
(b) ROC 100X

![](images/12934a46b6bbebe3d9ed0a78476c3cf2e947bdff56da610f19abd6d85e625b56.jpg)  
(c) ROC 200X

![](images/855100c52525e93dd3afec0ed9a7fc88a848a5897405991238a60f2d91b5246a.jpg)  
(d) ROC 400X

![](images/e6ac8ea2e35c9ffba258f3d1d83e3f0d8eb9c860c10ea8a8a68a52338db80fec.jpg)  
(e) PR 40X

![](images/99ce786415b337440cba4024f51cf495a0c9dcb764523415844b636530facefb.jpg)  
(f) PR 100X

![](images/3ab2ac3fd71b36787e1fafc3aa5a689174f3238723560a2fb29342f678fdd213.jpg)  
(g) PR 200X

![](images/8424a737dc0b7e73445deeff1a601ffebe7ce0fd9264df977e2d52e41e400a53.jpg)  
(h) PR 400X  
Figure 4: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the DenseNet-121 model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

Table 2: Performance of EfficientNet on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>93.1</td><td>96.1</td><td>94.3</td><td>95.2</td><td>89.7</td><td>0.979</td></tr><tr><td>Test (All Mag.)</td><td>86.2</td><td>83.6</td><td>86.0</td><td>84.8</td><td>86.3</td><td>0.940</td></tr><tr><td>Test (40×)</td><td>85.9</td><td>87.8</td><td>80.4</td><td>83.9</td><td>90.6</td><td>0.946</td></tr><tr><td>Test (100×)</td><td>86.3</td><td>86.4</td><td>84.3</td><td>85.4</td><td>88.1</td><td>0.941</td></tr><tr><td>Test (200×)</td><td>85.5</td><td>83.2</td><td>83.2</td><td>83.2</td><td>87.3</td><td>0.959</td></tr><tr><td>Test (400×)</td><td>84.3</td><td>78.7</td><td>86.7</td><td>82.5</td><td>82.4</td><td>0.915</td></tr></table>

On the BreakHis dataset, EfficientNet performed well though with a little bit of variation. With a recall of 94.31%, a specificity of 89.74%, an F1-score of 95.18%, and an area under the curve (AUC-ROC) of 0.979 on the validation set (at all magnifications), it demonstrated strong generalizability and a balanced sensitivity-specificity trade-off (Table 2). Nevertheless, a decline in performance to 86.16% accuracy, 83.6% precision, 85.99% recall, 86.29% specificity, 84.78% F1-score, and 0.9397 AUC-ROC was seen on the test set (at all magnifications), indicating difficulties with generalization (Figure 5). 40X (85.9% accuracy, 94.64% AUC), 100X (86.27% accuracy, 94.13% AUC), 200X (85.52% accuracy, 95.87% AUC), and 400X (84.28% accuracy, 91.5% AUC) were all results that were consistent but below validation (Figure 6 & 7). These results demonstrate that EfficientNet is efficient and has good feature extraction capabilities, but they also show that it is sensitive to variance in magnification and that test performance is slightly lower than validation (Figure 8). This suggests that there may be overfitting in this dataset and that more optimization is needed.

![](images/22d859c623f4e356250263b311bf0b7f049e3849d0162af51034b49d8315a550.jpg)

![](images/e5d0a03e3bdfb6a7966ef2c5f9738d7e9514c4076cc9be38c88c45dd472ed8e1.jpg)

![](images/2c4c208943c0cc1f2362165e094fd15b6ce4b03d442b1b85f889ff479ff141d5.jpg)

![](images/a33aae4fefe4a7edd3a55e2c8d5f5f195f977fc4a88a9283cd59b30f8b04138c.jpg)  
Figure 5: Model performance metrics on the BreakHis dataset across validation and test sets at different magnifications.

![](images/9ec0d1b281669ac0153d5189acc20cf9cc94fcb38e91b88d9272b30dd905c1e3.jpg)  
(a) 40X

![](images/0fc99044c4d43bfb433da55886efdcbec0f63eae30b7c3f2ed2cc8278bd66a44.jpg)  
(b) 100X

![](images/40dffe604e6ea19dd136134a170a7e32828a74cbdc718b4b5625b7dff739da7a.jpg)  
(c) 200X

![](images/88891cb9ed7f76b7095cc92bdf29f523f8606662a60c50f720090b8e0a139520.jpg)  
(d) 400X  
Figure 6: Confusion matrices of the EfficientNet model across 40X, 100X, 200X, and 400X magnifications on the BreakHis dataset.

![](images/59745053f9780ea909112c1be2c6b089f745843025acd9979ad87546d777817d.jpg)  
(a) Validation ROC

![](images/533e3b283d15c661a39f07495e405230921052cc4b1a66a2ed59f9c0c4e8b0b5.jpg)  
(b) Validation Confusion Matrix

![](images/598c8b726bce093e14910fdb0e8317aed6e11fcd42a70f9cdf903eca3682076f.jpg)  
(c) Test ROC

![](images/ce257b4a673979473286f552ff695f5cf7b378295987dd6b613caa97ce26a033.jpg)  
(d) Test Confusion Matrix  
Figure 7: Validation and test performance of the EfficientNet model on the BreakHis dataset across all magnifications.

![](images/78940f43e5feb3511b72a79071bcd4d201c45158ede3c3c37f76ec2339994fbd.jpg)  
(a) ROC 40X

![](images/dfffacfe570c84c9f70a2743eb650e2308119ec77054f5b006f1d50708f52912.jpg)  
(b) ROC 100X

![](images/27f889326cd37ce0cc213343fbbc2c247cf0ba123284baf3300aae117cddc084.jpg)  
(c) ROC 200X

![](images/aca223213eb0046dfb138cccb669f560ac384c97bf7a7d2b3f645b41066ac97f.jpg)  
(d) ROC 400X

![](images/8631155429d9eff7c34f95bae9ef501312d5c94d88f635c08e3f00fa31d4f7d7.jpg)  
(e) PR 40X

![](images/20b434e7c83989360dee00a6b48bf59d64b9516ca95ae17c6394f55ff5088d3a.jpg)  
(f) PR 100X

![](images/9fcfc5986e1bc2811b896ac5ae988fff5a5cc74b367447f0223e5cf46f482ba6.jpg)  
(g) PR 200X

![](images/b32dd19b93acf7f4fd1be8fba1b58db1a133e0220b92c6ab33ed18cae63e1fbf.jpg)  
(h) PR 400X  
Figure 8: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the EfficientNet model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

Table 3: Performance of ConvNeXt on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>80.9</td><td>80.6</td><td>91.1</td><td>85.6</td><td>64.2</td><td>0.881</td></tr><tr><td>Test (All Mag.)</td><td>77.1</td><td>85.1</td><td>89.1</td><td>87.0</td><td>2.4</td><td>0.653</td></tr><tr><td>Test (40×)</td><td>76.1</td><td>83.6</td><td>89.5</td><td>86.4</td><td>0.0</td><td>0.450</td></tr><tr><td>Test (100×)</td><td>79.1</td><td>84.7</td><td>92.1</td><td>88.3</td><td>2.7</td><td>0.708</td></tr><tr><td>Test (200×)</td><td>79.7</td><td>88.3</td><td>88.8</td><td>88.6</td><td>8.7</td><td>0.823</td></tr><tr><td>Test (400×)</td><td>76.4</td><td>85.2</td><td>88.0</td><td>86.6</td><td>3.5</td><td>0.780</td></tr></table>

ConvNeXt showed moderate efficacy on the BreakHis dataset, with the validation set achieving 80.92% accuracy, 80.62% precision, 91.14% recall, 64.23% specificity, 85.56% F1-score, and 0.8806 AUC-ROC, indicating strong sensitivity but a diminished capacity to accurately detect benign instances (Table 3). In the test set (across all magnifications), the results exhibited a fall to 77.14% accuracy, 85.12% precision, 89.06% recall, 2.42% specificity, 87.04% F1-score, and 0.6533 AUC-ROC, indicating a pronounced bias towards malignant categorization owing to the exceedingly low specificity (Figure 9, 10 & 11)). ConvNeXt showed consistently high recall (≥.88%) across various magnifications but had inadequate specificity, especially at 40X (0%) and only slight enhancements at 100X (2.7%), 200X (8.7%), and 400X (3.45%), despite F1-scores ranging from 86% to 88% (Figure 12). The results indicate that although ConvNeXt effectively detects malignant samples owing to its hierarchical convolutional-transformer hybrid architecture, its failure to balance false positives undermines its diagnostic reliability in breast cancer classification.

![](images/127ab4cb47556910d9e2d52a87081428c6e54109edf6cb897f0d68d5aa8aa342.jpg)

![](images/0dbafb9f89c48678de7790a8a8341d8a4f5da662ac6725a0b6a06787bfbbeb26.jpg)

![](images/005c0f6f2de281f694284465546e0efad63e5eeb0ec4bd0f173e65100c0e6a2e.jpg)

![](images/e6b0b5925db59062f02a95ee4d8a50635e804adc02929942f1ed86df03a19985.jpg)  
Figure 9: Performance metrics of the ConvNeXt model across validation and test sets at multiple magnifications.

![](images/9b720c70e5cacfba196e65c1e8a346b3dccc311d02e4a3094bc29912bac3de40.jpg)  
(a) 40X

![](images/c323278060b36ca9273445ba9e4f352a5f6d4a418279dcf1b3544e1e5afc776a.jpg)  
(b) (b) 100X

![](images/b247f12a9575c3e057793488195c8b9bd5947acc15220cfb527873cc2ba682b5.jpg)  
(c) (c) 200X

![](images/a75449d364d1a046bba32785434ae43c68188b25fffb096d97a0f23fad575fb8.jpg)  
(d) (d) 400X  
Figure 10: Confusion matrices of the ConvNeXt model across 40X, 100X, 200X, and 400X magnifications on the BreakHis dataset.

![](images/d125bbfe990c3ba3a3da715ea448167976ff2d8f99ed67fec18ee1fbe36069e0.jpg)  
(a) Validation ROC

![](images/5bf35a37f37d5d1de73aecc00fe423890496e3fda490ca3aec43648fffefd822.jpg)  
(b) Validation Confusion Matrix

![](images/81dbc97236aaf5d13545d55a56429de1c40cfcb012a9fc8e27d55f5b26786128.jpg)  
(c) Test ROC

![](images/ac28dd0109641c2de97d3caf255cc49583d8db96cba4d016a088fba3c15bc199.jpg)  
(d) Test Confusion Matrix  
Figure 11: Validation and test performance of the ConvNeXt model on the BreakHis dataset across all magnifications.

![](images/e0422a00f4d2e27a9d2e6b150a5f1508a8a2a94e3199614d7f2a6b7702a61773.jpg)  
(a) ROC 40X

![](images/f4067915d4ed0659427ed5f25256c534a814e0bf3deaba0c9f9378dec96e90b3.jpg)  
(b) ROC 100X

![](images/462e68eac482b5c206e30b304fe736ac243e16af2291f7a84ff11f26a7ac8ae4.jpg)  
(c) ROC 200X

![](images/546696684cd5f55c53ad99bcdef1a95aa6ba3785a3ea78721e17f1d74372b26a.jpg)  
(d) ROC 400X

![](images/7ed73f69986e9d63ab620886cb0c23a9f0b8f3d1eaa1f43c34a53095fd777ffc.jpg)  
(e) PR 40X

![](images/50956b94326e66a77a82e5e13c979a399f2becc4a8c358b1bb26b7c03c999933.jpg)  
(f) PR 100X

![](images/ad71aad2aac0734c8fda2512a8e86439011c15425385c37de0700a57ffada503.jpg)  
(g) PR 200X

![](images/8b0bb54c83ef7f9e3c6efb124a9425ce3b4d568991b26db4fbe5784c25b308ac.jpg)  
(h) PR 400X  
Figure 12: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the ConvNeXt model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

Table 4: Performance of InceptionV3 on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>94.3</td><td>96.2</td><td>95.9</td><td>96.1</td><td>89.9</td><td>0.980</td></tr><tr><td>Test (All Mag.)</td><td>84.6</td><td>79.5</td><td>88.4</td><td>83.7</td><td>81.5</td><td>0.933</td></tr><tr><td>Test (40×)</td><td>87.2</td><td>86.0</td><td>86.0</td><td>86.0</td><td>88.2</td><td>0.937</td></tr><tr><td>Test (100×)</td><td>81.2</td><td>77.0</td><td>85.9</td><td>81.3</td><td>76.9</td><td>0.924</td></tr><tr><td>Test (200×)</td><td>85.1</td><td>80.4</td><td>86.3</td><td>83.3</td><td>84.1</td><td>0.938</td></tr><tr><td>Test (400×)</td><td>86.9</td><td>78.3</td><td>95.9</td><td>86.2</td><td>80.2</td><td>0.943</td></tr></table>

Inception V3 showed strong results on the BreakHis validation set, achieving 94.29% accuracy, 96.22% precision, 95.91% recall, 89.96% specificity, 96.06% F1-score, and 0.9803 AUC-ROC, indicating exceptional generalization and a well-balanced sensitivity–specificity ratio (Table 4). Nonetheless, performance on the test set (across all magnifications) diminished to 84.56% accuracy, 79.49% precision, 88.36% recall, 81.47% specificity, 83.69% F1-score, and 0.933 AUC-ROC, suggesting a propensity to misclassify benign instances (Figure 13, 14 & 15). The model attained accuracies of 87.18% at 40X magnification (AUC 0.9366), 81.18% at 100X (AUC 0.9243), and 85.07% at 200X (AUC 0.9383), with recall continuously surpassing precision, indicating effective detection of malignant cases but a higher incidence of false positives (Figure 16). These results underscore Inception V3’s efficacy in extracting multi-scale histopathological features, while simultaneously revealing its vulnerability to magnification-related variability and diminished specificity on previously unreported test data.

![](images/906cce5a940263648869acb8136bde70d3f9b403274be7b2d7531349716a6ec6.jpg)

![](images/f008dc25c02120075b42255c56a686ac0ae1e928c1e85d76f5db1de3879e5739.jpg)

![](images/fbfe73d91972b527f9188fa595206bfaf2fef7674bc3e6243c0030d6d1462f08.jpg)

![](images/a97186f42d62b8881e2ba8ecdd13f33529b65b01692bccea646ec456d3dff6d2.jpg)  
Figure 13: Performance of the Inception model shows high validation accuracy (94.5%, AUC = 0.98) and stable test results (81–87% accuracy, AUC = 0.93–0.95), with recall generally exceeding precision across magnifications.

![](images/0b04f1340c1d3b7dec59f3e919af8f4fff3736518233c8c5adf61866a80b401a.jpg)  
(a) 40X

![](images/674af947b706f2810e1d01b167e4ea106a6cfc21748f53bf4c2bb7b2f032d68f.jpg)  
(b) 100X

![](images/490081e4cf5ac82b02e63b4e25c7d263e727847abca20183fc334767a39d93bb.jpg)  
(c) 200X

![](images/b2f11de6d499e406d6ff61c652cc01027bec1e10b113497b1cf8cf1f32a6b2a0.jpg)  
(d) 400X  
Figure 14: Confusion matrices of the Inception model across 40X, 100X, 200X, and 400X magnifications

![](images/d375769ca0be13aae7ef551a59df4b625d0c7e747b86aa87de1f0f2170381cd6.jpg)  
(a) Validation ROC

![](images/4887261248ea68375b5d8d3d054354422132fcbb7e4f1f8e9f4debba749c8067.jpg)  
(b) Validation Confusion Matrix

![](images/e779a764dfc367b07f57dbf21acf30895b9f7c79c88b32d0bc3caad68d210fdb.jpg)  
(c) Test ROC

![](images/6eb3e6e2a18bf8b544e824227702b3ce26c82913ff2fba903ea6f9e7df62a0e5.jpg)  
(d) Test Confusion Matrix  
Figure 15: Validation and test performance of the Inception model on the BreakHis dataset across all magnifications.

![](images/3a81aeff346e44bf5ec83b515498c7d20b005c8222282c0990c0e5c9e253396b.jpg)  
(a) ROC 40X

![](images/d09086070433f5b7e9b65ddda080531ae6aae2be32bffad3d58a54fb9d2de894.jpg)  
(b) ROC 100X

![](images/b939762b58f5b294a51f0bd65c74254ef820c2d057cebfc1e077b353603bb976.jpg)  
(c) ROC 200X

![](images/ce171cdf2e3ae8a3dd32b5b09459c72ff664e979084920a93442370a5b0188b1.jpg)  
(d) ROC 400X

![](images/a5e6d558ae0d6b35e3aaf4d3564aadd7fb3e52a2dc5670314842deb1f290f602.jpg)  
(e) PR 40X

![](images/97589d1e6d4457d10b9e138dc7135856cc590bf0a8e2f513d67c20460610a6f9.jpg)  
(f) PR 100X

![](images/5e449afa0bfe4af579ca31fa66994258223728438202298f495f1121e53aadbb.jpg)  
(g) PR 200X

![](images/2603346cc4a883a6e81b74d35993c38fa3a5b0a4f4fa34630b7c654c012779b4.jpg)  
(h) PR 400X  
Figure 16: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the Inception model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

Table 5: Performance of MobileNetV2 on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>91.8</td><td>95.9</td><td>92.7</td><td>94.3</td><td>89.5</td><td>0.975</td></tr><tr><td>Test (All Mag.)</td><td>84.7</td><td>83.5</td><td>81.9</td><td>82.7</td><td>86.9</td><td>0.933</td></tr><tr><td>Test (40×)</td><td>81.6</td><td>84.0</td><td>73.8</td><td>78.6</td><td>88.2</td><td>0.923</td></tr><tr><td>Test (100×)</td><td>84.7</td><td>82.0</td><td>86.8</td><td>84.3</td><td>82.8</td><td>0.928</td></tr><tr><td>Test (200×)</td><td>86.4</td><td>85.0</td><td>83.2</td><td>84.0</td><td>88.9</td><td>0.956</td></tr><tr><td>Test (400×)</td><td>84.3</td><td>81.6</td><td>81.6</td><td>81.6</td><td>86.3</td><td>0.927</td></tr></table>

MobileNetV2 showed strong efficacy on the BreakHis validation set, attaining 91.84% accuracy, 95.93% precision, 92.70% recall, 89.53% specificity, 94.29% F1-score, and 0.9746 AUC-ROC, so affirming effective generalization during training (Table 5). In the test set (across all magnifications), the findings diminished to 84.66% accuracy, 83.54% precision, 81.95% recall, 86.87% specificity, 82.73% F1-score, and 0.9332 AUC-ROC, indicating a marginal reduction in sensitivity while maintaining balanced specificity (Figure 17, 18 & 19). At several magnifications, performance exhibited variability: 40X (81.62% accuracy, 73.83% recall, AUC 0.9229) shown diminished sensitivity, whereas 100X (84.71%, AUC 0.9284), 200X (86.43%, AUC 0.9558), and 400X (84.28%, AUC 0.9271) yielded more consistent outcomes, with recall and specificity ranging from 82% to 89% (Figure 20). The results demonstrate that MobileNetV2, utilizing lightweight depthwise separable convolutions, achieves competitive accuracy and efficiency; however, its diminished capacity relative to deeper architectures leads to decreased recall and heightened magnification sensitivity in test data.

Model Performance Metrics Comparison

![](images/e25574cbe69a154acee125e549316af4e08e00b6cc71b39a1cd387b1dd18916b.jpg)

![](images/dc470e88bf79a6ef2ac221bd616e7f41c05e3d4993bc8085597c49cdab3b38e2.jpg)

![](images/c075998c92785aee5aa4133c9b13768cfb837c6da50bed480c165c96986debd8.jpg)

![](images/ed229e82a5a28bfa5a53348e1b47a74bb58934cbe8780b1e7ca6227e2a6f6175.jpg)  
Figure 17: MobileNet performance across validation and test sets at multiple magnifications

![](images/cf8536b9f24877075b7cd6ad60d8506681f56f78558fdefa3fc9231122629da1.jpg)  
(a) 40X

![](images/9a8e80ca29c8828c4bc6b4dd25985e3de31d5a1ca9211b0a972c7e904cf710b6.jpg)  
(b) 100X

![](images/cbed16cc1e79c71fe6340e8cb10eeb70d89341468a787d1a0fa8bc13f90442d7.jpg)  
(c) 200X

![](images/9a6e2dc03c11bd3255fef4598fa984b6b88d72a80085c2fbe46c169024a2c02e.jpg)  
(d) 400X  
Figure 18: Confusion matrices of the MobileNetV2 model across 40X, 100X, 200X, and 400X magnifications on the BreakHis dataset.

![](images/1a0400ca01ed50f97bb6164c71b6c0889f41288deab1cfe9e9ded701a55d6207.jpg)  
(a) Validation ROC

![](images/5ddb43540ac7ace6ada2da7588d5e2f49b1211e5f4128105e8ea18cfe0e7ba33.jpg)  
(b) Validation Confusion Matrix

![](images/27db27d451e1e2575aacbc539ff87443c9bbb81cb5f2f829de5b74f31641e18a.jpg)  
(c) Test ROC

![](images/6a5501a0661cb1f9153007ee21be335689f47ce3dd2ae4a8b8b0c91166b8a5e6.jpg)  
(d) Test Confusion Matrix  
Figure 19: Validation and test performance of the MobileNetV2 model

![](images/a37ea7e7b56ea8baef5a64ee3492df35e86e76529af626381ba72e183d0f8a9a.jpg)  
(a) ROC 40X

![](images/2db10128a7348e2ae0cc3785a0ef66fe7f5e81ecfd0a067fce7b91f709234395.jpg)  
(b) ROC 100X

![](images/f33764f78522488f5b3017b8c7660ea2c6e4fdefed1c6bde6206ba6ee85df0f0.jpg)  
(c) ROC 200X

![](images/afb1b36f71d38b80a38056aed9165e374f12662990caecb0e88ed1df5ec52c4e.jpg)  
(d) ROC 400X

![](images/d3a964e5ef24de019d93411705c224baf9b3481caa901e2d30794019a6d4294f.jpg)  
(e) PR 40X

![](images/779e069b574a4e8acefa54cd39d498a18306196af54605cfcaa7f11c0ef83ebf.jpg)  
(f) PR 100X

![](images/fd70a5c701ba14106afc1f6df502180e16b4362d5c384a3924bf453c8b6017d7.jpg)  
(g) PR 200X

![](images/945cc9448198c13bee6f20a8bea04e97ed318187308ec57816e001e70dd5a374.jpg)  
(h) PR 400X  
Figure 20: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the MobileNetV2 model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

Table 6: Performance of ResNet50 on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>93.2</td><td>96.2</td><td>94.4</td><td>95.3</td><td>90.2</td><td>0.978</td></tr><tr><td>Test (All Mag.)</td><td>83.6</td><td>79.1</td><td>86.2</td><td>82.5</td><td>81.5</td><td>0.919</td></tr><tr><td>Test (40×)</td><td>87.2</td><td>87.4</td><td>84.1</td><td>85.7</td><td>89.8</td><td>0.914</td></tr><tr><td>Test (100×)</td><td>82.4</td><td>80.6</td><td>82.6</td><td>81.6</td><td>82.1</td><td>0.925</td></tr><tr><td>Test (200×)</td><td>83.3</td><td>76.4</td><td>88.4</td><td>81.9</td><td>79.4</td><td>0.928</td></tr><tr><td>Test (400×)</td><td>80.8</td><td>73.3</td><td>86.7</td><td>79.4</td><td>76.3</td><td>0.907</td></tr></table>

ResNet50 achieved excellent generalization on the BreakHis validation set, achieving 93.24% accuracy, 96.24% precision, 94.39% recall, 90.17% specificity, 95.30% F1-score, and 0.9784 AUC-ROC, indicating an adequate sensitivity–specificity balance during training (Table 6). In the test set (across all magnifications), performance decreased to 83.60% accuracy, 79.08% precision, 86.22% recall, 81.47% specificity, 82.50% F1-score, and 0.9185 AUC-ROC, signifying diminished precision and an increase in false positives (Figure 21, 22 & 23). At various magnifications, the model exhibited optimal performance at 40X, achieving 87.18% accuracy, 87.38% precision, 84.11% recall, 89.76% specificity, and an AUC of 0.9135. In contrast, the results at 100X yielded 82.35% accuracy with a 92.51 AUC, and at 200X, 83.26% accuracy with a 92.82 AUC, demonstrating commendable recall but diminished precision (Figure 24). Performance was suboptimal at 400X (80.79% accuracy, 73.28% precision, 86.73% recall, AUC 0.9065), indicating difficulties in collecting intricate elements at elevated magnification. ResNet50 exhibited dependable malignant diagnosis with excellent recall; however, diminished precision and variability across magnifications underscore its sensitivity to scale variations in histopathological pictures.

Model Performance Metrics Comparison  
![](images/899721395a736203034e396279e09efd2f73adbf4c5ebb084e791e2cedf94cdc.jpg)

![](images/5dd24b96ccb8412a0156aa42c514603ff8643392e360dd2cced61f608d75f0fc.jpg)

![](images/11981a107d427127470d01b001e3e3da03c590a9fa5141361f1077df0bcdd1b4.jpg)

![](images/4b3d49daa42171c27c96039f46187d5b6461b9a042ad9fbe455b0e833fdfa45e.jpg)  
Figure 21: Performance metrics of ResNet across validation and test sets at multiple magnifications.

![](images/ceaca4b0474387181dcb9de8991db2f6724cf2457e7b2b0792183e040e97752e.jpg)  
(a) 40X

![](images/e1236b387930d422fcb28649c7e7d6ce077dbac1d3fa7aec39e07c0978f959da.jpg)  
(b) 100X

![](images/d4fc57e9dc2f5a5afe5338086b7839c3ad69323704e39b6aa08b2259748b0738.jpg)  
(c) 200X

![](images/b15287f9be0012757df67b7eb9963b732508681913fc2a010eb6b41efd5e398d.jpg)  
(d) 400X  
Figure 22: Confusion matrices of the ResNet model across 40X, 100X, 200X, and 400X magnifications

![](images/0023e2d0fc7d1fff558025d193fc6d7c45eb0e45f684401299c57bea84297b32.jpg)  
(a) Validation ROC

![](images/00e69e8553b6879cbfd6195c87784369a9f0dba04fcc3dfdbddaac7ae598a6ad.jpg)  
(b) Validation Confusion Matrix

![](images/8fb95543134ead8459cde31bfb91e8ebeb6583e46ed1cd1ebe83319c09831356.jpg)  
(c) Test ROC

![](images/324e740cc82e31d11afff9d00cd9d8b44e556b3204c3110bd0fb21c0a568cb56.jpg)  
(d) Test Confusion Matrix  
Figure 23: Validation and test performance of the ResNet model on the BreakHis dataset across all magnifications.

![](images/a026c54c011c5d3b7041f5b24317d6534a48511d8d380f16a92e92efd0f9773f.jpg)  
(a) ROC 40X

![](images/84a4962be2157af0755b00553baa31066804097c0fd19866787a1d15f461585d.jpg)  
(b) ROC 100X

![](images/ab92b62a141f9caeaff427b987496fe2c621d77553fc504dff59495041addd3f.jpg)  
(c) ROC 200X

![](images/cdeb7b199df69a21bbdd4f95b837e6420a7842434b901f0829863299b25e7eff.jpg)  
(d) ROC 400X

![](images/98dbf1fd973125d81670e46270537bdc7159b4d21e0a9380757b87db77214014.jpg)  
(e) PR 40X

![](images/48759495107ea0212596c0b7bd2b5e97cc8dc596b4bbce0d9e890db5b80461ef.jpg)  
(f) PR 100X

![](images/fb0f9d82cd27477a545d8d307c538cf8d7f99e197d41c73c795a87413444d2eb.jpg)  
(g) PR 200X

![](images/2e6e5d354ee50102f6d3ba5232beec1b6b0533ed2e30c8ec97818cab9065541b.jpg)  
(h) PR 400X  
Figure 24: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h).

Table 7: Performance of Swin Transformer Tiny on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Test (All Mag.)</td><td>79.5</td><td>73.6</td><td>84.6</td><td>78.7</td><td>75.3</td><td>0.860</td></tr><tr><td>Test (40×)</td><td>85.0</td><td>81.6</td><td>86.9</td><td>84.2</td><td>83.5</td><td>0.929</td></tr><tr><td>Test (100×)</td><td>79.2</td><td>74.6</td><td>85.1</td><td>79.5</td><td>73.9</td><td>0.874</td></tr><tr><td>Test (200×)</td><td>77.8</td><td>69.8</td><td>85.3</td><td>76.8</td><td>72.2</td><td>0.834</td></tr><tr><td>Test (400×)</td><td>77.7</td><td>70.1</td><td>83.7</td><td>76.3</td><td>73.3</td><td>0.821</td></tr></table>

The Swin Transformer Tiny attained an accuracy of 79.45%, precision of 73.55%, recall of 84.56%, specificity of 75.29%, F1-score of 78.67%, and an AUC-ROC of 0.8595 across all magnifications on BreakHis, indicating robust malignancy identification but moderate precision and specificity (Table 7). Performance peaked at 40X magnification, achieving 85.04% accuracy, 81.58% precision, 86.92% recall, 83.46% specificity, an F1 score of 84.16%, and an AUC of 0.9285. In contrast, findings at 100X yielded 79.22% accuracy and an AUC of 87.36, while 200X produced 77.83% accuracy and an AUC of 83.38, both demonstrating commendable recall (>85%) but diminished precision (<75%) (Figure 25, 26 & 27). At 400X magnification, accuracy decreased to 77.73%, with precision at 70.09%, recall at 83.67%, specificity at 73.28%, and an AUC-ROC of 0.8212, suggesting challenges in discerning fine-grained patterns at elevated magnification levels. The Swin Transformer exhibited strong recall and stable F1-scores across many scales; however, its significantly lower accuracy and specificity indicate a propensity for false positives, despite its efficacy in detecting cancerous regions.

Swin Transformer Tiny Model Performance Metrics Comparison

![](images/6e1d7344eb8fd603c0fb4d3e642cb8897d068510f7539a77d1c489059490a7df.jpg)

![](images/064ebf07a240c5816b3c29a8a8d993046aa84259ce5b7d8e72fda0e8882d059f.jpg)

![](images/bb169fb525aa5eb57baa49eae387c5ecdea82faadcf7d7ee8f430e414e22eb07.jpg)

![](images/a87a7d81bf879b94666114b8e6ddb761a977ad529bd2557677c0c3789427e00d.jpg)  
Figure 25: Performance metrics of the Swin Transformer Tiny across different magnifications

![](images/67d0a1b76edc51bfe6c47008b164555eac33bc9c9ac85c1138bb33a78d3f1436.jpg)

![](images/257f396e465e60b172f1254cb5adea9c7d67b2f64df71cd444c4dabe44ae43bf.jpg)  
Figure 26: Training curves of Swin Transformer Tiny

![](images/3cebeb430e5a6ae76ba637f46d06813af7993a973e411c9086ec483f57fd9a8a.jpg)

![](images/a32da7c36621457378950a70839295c2ffa41db2276d293ef207818c0cd32045.jpg)  
(a) (a) 40X

![](images/37c6f71fed8d6a1128073dd6b599144509abc2f02a420b215e462853b3d717d6.jpg)  
(b) (b) 100X

![](images/ff07c62cb8588bd9b07c1a6d8cc5a2ef80a90f1dd8ea833c647c723c05acbcf5.jpg)  
(c) (c) 200X

![](images/09382cae484da67a02c69f4d9abad490ff701b0d13137c35956a85ed0873b6de.jpg)  
(d) (d) 400X  
Figure 27: Confusion matrices of the Swin Transformer Tiny model across 40X, 100X, 200X, and 400X magnifications

![](images/0fd6dad4f9ec9535232ef5cdbd02b9ad4d68f36bf99fb5d9efb6cecc34a6d097.jpg)  
Figure 28: Performance metrics of the VGG19 model across validation and test sets at multiple magnifications.

Table 8: Performance of VGG19 on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>93.1</td><td>95.9</td><td>94.6</td><td>95.2</td><td>89.3</td><td>0.975</td></tr><tr><td>Test (All Mag.)</td><td>81.3</td><td>76.5</td><td>84.1</td><td>80.1</td><td>79.0</td><td>0.905</td></tr><tr><td>Test (40×)</td><td>80.3</td><td>75.6</td><td>84.1</td><td>79.6</td><td>77.2</td><td>0.899</td></tr><tr><td>Test (100×)</td><td>78.8</td><td>74.5</td><td>84.3</td><td>79.1</td><td>73.9</td><td>0.914</td></tr><tr><td>Test (200×)</td><td>84.2</td><td>76.8</td><td>90.5</td><td>83.1</td><td>79.4</td><td>0.938</td></tr><tr><td>Test (400×)</td><td>80.4</td><td>74.8</td><td>81.6</td><td>78.1</td><td>79.4</td><td>0.892</td></tr></table>

VGG19 had outstanding results on the validation set, achieving 93.12% accuracy, 95.93% precision, 94.55% recall, 89.32% specificity, 95.23% F1-score, and 0.975 AUC-ROC, so proving its strong generalization throughout training (Table 8). Test performance declined to 81.26% accuracy, 76.46% precision, 84.09% recall, 78.96% specificity, 80.09% F1-score, and 0.9053 AUC-ROC, signifying diminished precision and an increase in false positives (Figure 28, 29 & 30). At certain magnifications, the results were consistent: 40X (80.34% accuracy, AUC 0.8987), 100X (78.82%, AUC 0.9138), 200X (84.16%, recall 90.53%, AUC 0.9378), and 400X (80.35%, AUC 0.892). The model exhibited strong recall at various magnifications (≥.84%), achieving the optimal balance at 200X; however, precision was comparatively lower (74–77%), indicating a predisposition towards malignant predictions (Figure 31). VGG19 exhibited significant sensitivity and dependable detection of malignant cases; nevertheless, diminished accuracy and unpredictability in magnification underscore the limitations of its deeper, sequential CNN design in comparison to contemporary alternatives.

![](images/d742c870059af123b7e2add4b981a60db799c7bbe8d7f6ddc5546ccb31e73ff7.jpg)  
(a) 40X

![](images/92f6770c9e37e0912267df84e8db543d59ec7b3e3f4ee12f731a0b99f3c45a6d.jpg)  
(b) 100X

![](images/187f79a066425934c9c82249fb46a605a0e2c2500c3059371b55c0523e7cf701.jpg)  
(c) 200X

![](images/32a3fdd99874b959365c13cb6562199fcd08e6200a7ba0f79304b425f6193bfb.jpg)  
(d) 400X  
Figure 29: Confusion matrices of the VGG19 model across 40X, 100X, 200X, and 400X magnifications on the BreakHis dataset.

![](images/1f3d0d7587218295d7e6b80b5c1692ea132b5dd60b2019ac7d4edf05b24c0138.jpg)  
(a) Validation ROC

![](images/5c9ddb1bb73995dc40743894ee6fc1311fd4f84312641e2dabeb44624f98744e.jpg)  
(b) Validation Confusion Matrix

![](images/92ff12e186c3d77b3e3757da9405ae3b16872b6d16487d41b472a58da9d01b04.jpg)  
(c) Test ROC

![](images/7e6f951d17b5c5c089f4bf577e11ee0541c2733bbd0c6143b97b09c684961d2a.jpg)  
(d) Test Confusion Matrix  
Figure 30: Validation and test performance of the VGG19 model on the BreakHis dataset across all magnifications.

![](images/d5c67b4f1c21a493adee88514cb5e02a9b71b4fe6771710d788bb91620fe078f.jpg)  
(a) ROC 40X

![](images/ab45a40350baca3a312796c7a98b624ed87deb312212f411daadbe3852e7a05b.jpg)  
(b) ROC 100X

![](images/3f4b5ed850d0f18ae6c30ba02d553f851d5afd8a738f57615b31e35665f1181c.jpg)  
(c) ROC 200X

![](images/10a177d24410025ccf3ade6388a1befa52ba2ac22c219589a976feaa39621f17.jpg)  
(d) ROC 400X

![](images/a30d670943198f742dd85b88af8d92626771c8919c801c7946ea1f4a45f6a24e.jpg)  
(e) PR 40X

![](images/6ce496cee4d646a2636776c09a1c3da85b3f94e76e1f5ea6e518d40b6876ffd3.jpg)  
(f) PR 100X

![](images/f5def07869fa98ed130f630d42a2b2f097f4b7bec6aad99c09e9e12fddb849fa.jpg)  
(g) PR 200X

![](images/f45ec01bb4b03a95c648f3e4f098bdeb2e1de905b6c1657a86cd1de41d71852c.jpg)  
(h) PR 400X  
Figure 31: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the VGG19 model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

The Vision Transformer (ViT) showed an accuracy of 86.47%, precision of 90.89%, recall of 90.46%, specificity of 75.85%, F1-score of 90.68%, and an AUC-ROC of 0.8918 on the validation set, showing robust sensitivity and balanced performance throughout training (Table 9). In the test set (across all magnifications), performance decreased to 77.64% accuracy, 71.14% precision, 84.32% recall, 72.20% specificity, 77.17% F1-score, and 0.8578 AUC-ROC, signifying diminished precision and an elevated false-positive rate (Figure 32, 33 & 34). At particular magnifications, ViT exhibited optimal performance at 200X, achieving 81.0% accuracy, 73.87% precision, 86.32% recall, 76.98% specificity, 79.61% F1-score, and an AUC of 0.8928. In contrast, both 40X and 400X attained approximately 79% accuracy with recall exceeding 83%, while performance was least effective at 100X, yielding 72.55% accuracy and an AUC of 0.7885 (Figure 35). The results underscore ViT’s proficiency in recall across various magnifications, affirming its capacity to identify malignant patients; nevertheless, diminished precision and magnification-dependent variability indicate difficulties in discerning intricate histological patterns relative to deeper CNN-based models.

Table 9: Performance of Vision Transformer (ViT) on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>86.5</td><td>90.9</td><td>90.5</td><td>90.7</td><td>75.9</td><td>0.892</td></tr><tr><td>Test (All Mag.)</td><td>77.6</td><td>71.1</td><td>84.3</td><td>77.2</td><td>72.2</td><td>0.858</td></tr><tr><td>Test (40×)</td><td>79.1</td><td>71.0</td><td>91.6</td><td>80.0</td><td>68.5</td><td>0.856</td></tr><tr><td>Test (100×)</td><td>72.6</td><td>68.4</td><td>78.5</td><td>73.1</td><td>67.2</td><td>0.789</td></tr><tr><td>Test (200×)</td><td>81.0</td><td>73.9</td><td>86.3</td><td>79.6</td><td>77.0</td><td>0.893</td></tr><tr><td>Test (400×)</td><td>79.0</td><td>71.9</td><td>83.7</td><td>77.4</td><td>75.6</td><td>0.862</td></tr></table>

![](images/c82467f7f7112a354e34586859bb9798261b91660f43dee8a6f847dc6a94602a.jpg)

![](images/79c495688df0c230eb9f8a112f3e0191ea8913d6312de246e31e3d422e8c6708.jpg)

![](images/cde34cfba218aba13b526f69071d83b3c333a91dcb2f30925773e78654ea7144.jpg)

![](images/0ac9738f790d43a5d46c0bde8a3495f62f987ef9c07b50d7bc68a3490e2ff451.jpg)  
Figure 32: ViT model performance across validation and test sets at multiple magnifications. Validation achieved strong results (Accuracy = 0.86, Precision = 0.91, Recall = 0.90, F1 = 0.91, AUC-ROC = 0.89).

![](images/5ffcfee90ce37f27f3ed7d754a4ec7cda7d8a20a280b307580d420ee47edd762.jpg)  
(a) 40X

![](images/6c16ef0c1c41bd660e27ad5e1daecf9fb9de59bbc6357bfd148fe47a30b1a1bd.jpg)  
(b) 100X

![](images/d4c1b93b5563a4304c4defe09be18460ad3cda10abee20d6cd70100cf6b4cbf0.jpg)  
(c) 200X

![](images/414091a0138f1663627b2ca667fe6d868c635c3e92d45bd419e8c68a8eedb6a4.jpg)  
(d) 400X  
Figure 33: Confusion matrices of the Vision Transformer (ViT) model across 40X, 100X, 200X, and 400X magnifications on the BreakHis dataset.

![](images/28a0ff0e9a57e6fa3cdec6d3d50589093db3c5676158824b1a55ed2079a4f2b2.jpg)  
(a) Validation ROC

![](images/09b5d0191f53f01857352d7dd3f5fe9d921db5f11c1df2275a71e4a57bff2ce7.jpg)  
(b) Validation Confusion Matrix

![](images/a58ee992d2568f848c118a417ff95b545d1c5aa1a5794e3de0542f4e8eaf2db4.jpg)  
(c) Test ROC

![](images/80231f5efeaa2ed86d7270d3f7491715bc74d37cfd197b192e684e73c2c9cf4c.jpg)  
(d) Test Confusion Matrix  
Figure 34: Validation and test performance of the Vision Transformer (ViT) model on the BreakHis dataset across all magnifications.

![](images/7032b5a3c84d0d75e278940c6f74dd1ef707f3b588afdd3d1d5f7dd2be4d1e7a.jpg)  
(a) ROC 40X

![](images/ae10badedc529f667c77ca201e5c8ffa444b4ddb1428ae5fe1a3e7244c0dd6fa.jpg)  
(b) ROC 100X

![](images/0064fa7bacaa8152b4dcbb28a12d851082802a873e9faf9d058d4cb884f2406c.jpg)  
(c) ROC 200X

![](images/e5ec52a8b0759ca44a6edc054c12fc70be58e7e7939ca99f0d2adf68510470e4.jpg)  
(d) ROC 400X

![](images/ce7345ef2852d7c91860364bb0ad34d07030d9cd7a232dc18c77c7c85337129c.jpg)  
(e) PR 40X

![](images/34b6a7bda243a8a76654fe33dbdd52552af5bc1c222c0ebcf1160454ec7bc5c9.jpg)  
(f) PR 100X

![](images/7459c90837441ad1e56939504f1acd71a1b6217d1e26ab9fc4c0fb6d9e0b4f2e.jpg)  
(g) PR 200X

![](images/165a32c6cddd9a91304543fd0ece9f22b54d0c00d4093a1b6ca0dda26c366424.jpg)  
(h) PR 400X  
Figure 35: Receiver Operating Characteristic (ROC) curves (a–d) and Precision–Recall (PR) curves (e–h) for the Vision Transformer (ViT) model evaluated on the BreakHis dataset across 40X, 100X, 200X, and 400X magnifications.

Table 10: Performance of Xception on the BreakHis dataset
<table><tr><td>Dataset</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Specificity (%)</td><td>AUC</td></tr><tr><td>Validation (All Mag.)</td><td>94.5</td><td>95.7</td><td>96.9</td><td>96.3</td><td>88.3</td><td>0.985</td></tr><tr><td>Test (All Mag.)</td><td>85.8</td><td>80.0</td><td>91.2</td><td>85.2</td><td>81.5</td><td>0.939</td></tr><tr><td>Test (40×)</td><td>87.2</td><td>82.9</td><td>90.7</td><td>86.6</td><td>84.3</td><td>0.945</td></tr><tr><td>Test (100×)</td><td>84.3</td><td>80.9</td><td>87.6</td><td>84.1</td><td>81.3</td><td>0.935</td></tr><tr><td>Test (200×)</td><td>87.3</td><td>80.2</td><td>93.7</td><td>86.4</td><td>82.5</td><td>0.961</td></tr><tr><td>Test (400×)</td><td>83.8</td><td>74.0</td><td>95.9</td><td>83.6</td><td>74.8</td><td>0.928</td></tr></table>

The Xception model showed remarkable performance on the validation set, achieving 94.52% accuracy, 95.65% precision, 96.87% recall, 88.25% specificity, 96.25% F1-score, and 0.9849 AUC-ROC, showing a robust sensitivityspecificity balance during training (Table 10). In the test set (across all magnifications), performance dropped to 85.84% accuracy, 80.0% precision, 91.21% recall, 81.47% specificity, 85.24% F1-score, and 0.9392 AUC-ROC, indicating elevated recall but diminished precision attributable to false positives. Results were comparable across magnifications: 40X (87.18% accuracy, 86.61% F1, AUC 0.9452), 100X (84.31% accuracy, 84.13% F1, AUC 0.9352), 200X (87.33% accuracy, 86.41% F1, AUC 0.9609), and 400X (83.84% accuracy, 83.56% F1, AUC 0.9278) (Figure 36). Recall was consistently elevated (≥.87%, culminating at 93.68% at 200X and 95.92% at 400X), affirming Xception’s efficacy in detecting malignant cases; nonetheless, its diminished precision throughout test sets underscores a trade-off between sensitivity and false-positive rates.

![](images/e3a7d9c7f8ddd55d8a69ebb4e31db7d7d19839a7cf4f97de085423291fdc8f95.jpg)  
Figure 36: Performance metrics comparison of the Xception model across validation and test sets at multiple magnifica tions (40X, 100X, 200X, 400X).

![](images/992aab696a34460686d84694847c3c7d31e97c14db9668e4a16130833cd1d640.jpg)

![](images/f2b26bef3a389bd119237dec7fc52437997d1a1b7e97db3e51406600dabcfb8e.jpg)  
Figure 37: Comparison of ten deep learning models on the test set (all magnifications) across accuracy, precision, recall, specificity, F1, and AUC-ROC.

![](images/f1ab8209893b80004dfe51a3dae9fa01b6bd6319d8d8cd937675e5b56d9c36a1.jpg)  
Figure 38: Top-5 performing models across evaluation metrics on the test set (all magnifications).

Among all models, DenseNet-121 showed outstanding results, attaining near-perfect generalization on the BreakHis test set with an accuracy of 99.41%, precision of 99.86%, recall of 99.45%, specificity of 99.17%, F1-score of 99.65%, and an AUC-ROC of 0.9997, while achieving 100% performance at 40X, 100X, and 200X magnifications. Other CNN-based models achieved competitive performance: Xception showed impressive performance with an accuracy of 85.84%, a recall of 91.21%, and an AUC-ROC of 0.9392, showing superior sensitivity across magnifications (recall ≥.90% at 200X and 400X) (Figure 37). Inception V3 and ResNet50 demonstrated comparable performance (accuracy 84–86%, AUC 0.92–0.94) with uniform recall across magnifications, albeit with marginally reduced precision, suggesting a propensity for false positives. EfficientNet and MobileNetV2 preserved efficiency while yielding consistent outcomes (accuracy approximately 84–86%, AUC approximately 0.93–0.95), albeit with some variability in recall contingent upon magnification (Figure 38). VGG19, despite exhibiting robust validation performance (93.12% accuracy, 0.975 AUC), demonstrated inferior generalization on test sets (about 81% accuracy, AUC around 0.91), highlighting the constraints of its antiquated sequential design. Conversely, transformer-based models like the Vision Transformer (ViT) and Swin Transformer Tiny exhibited inferior performance relative to CNNs, achieving test accuracies of approximately 77–79% and lower AUC values ( 0.85–0.89). Nevertheless, they demonstrated relatively high recall (>83%) across various magnifications, indicating potential in malignant case detection, albeit with diminished precision and specificity. ConvNeXt showed the lowest performance overall, achieving a test accuracy of 77.14% and an exceedingly low specificity of 2.42%, despite a high recall rate of 89%, which suggests a significant bias towards malignant diagnoses.

The results show that DenseNet-121 had the most robust and generalizable performance, whilst Xception, Inception V3, and EfficientNet achieved a commendable balance between accuracy and sensitivity. Transformer-based models showed potential for high recall but encountered difficulties with precision and generalization in histopathology data. These findings underscore the advantages of densely connected CNNs in breast cancer histopathology image classification, while emphasizing the necessity for additional improvement of vision transformers to address domainspecific heterogeneity.

## 7 Explainability

The DenseNet121 model achieved outstanding performance on the BreakHis dataset (test accuracy: 0.994, F1 score: 0.997, AUC-ROC: 1.0), with prediction confidence consistently above 0.98. To interpret its decisions, multiple XAI techniques were employed. Grad-CAM revealed progressive localization of discriminative regions, with deeper layers focusing on nuclei and glandular structures. Integrated Gradients consistently highlighted nuclei-dense areas and boundaries across different baselines, ensuring attribution robustness (Figure 39). LIME identified superpixel-level regions supporting benign predictions, while LRP decomposed evidence into positive contributions from epithelial and nuclei structures and negative relevance in background areas. Saliency maps further confirmed fine-grained sensitivity to relevant tissue features. Collectively, these results demonstrate that DenseNet121 not only achieves state-of-the-art performance but also grounds its predictions in clinically meaningful histopathological patterns.

![](images/8906c2e374733f5521671177cd0e948f4ad8a54391488cb4ebfb580ee0e4d192.jpg)  
LIME Analysis - Predicted: Benign (0.988)

![](images/39bc8cd1342c16e56a6ff2abfcf790fc6b4dee9c41b6f5cb186cc4bc14a2dd39.jpg)

![](images/db651afcf84ced91f6578dca9423424a6ec5db8adac19fa7e8718cb2675bbfc6.jpg)

![](images/9cde2faa8c87ff9418cd246ef056faf11b41022bff235a7f36d0c397f2d69c15.jpg)

![](images/a6489a9218f04df10fb6025177388091a8a86a5b7e135050cc54b937b00b23b8.jpg)

![](images/2226dd595e2927fe52f1d84f7d13680225eea14501c2689b0bcd5aa0e8110f10.jpg)

![](images/bc66c8a5e715ee7a4067091f7dcfe2692f464916a94ac1410003dd07cfa6b1d2.jpg)

![](images/4d3e29c361410be04c7e8e5099cddc97ae3e8aa69cd3de9147cc1ac78c31a02e.jpg)  
02

(a) Grad-CAM  
![](images/d9516641c1305600ca22fabf55967d874bb985e2b6a244ef04ab7921de9e131e.jpg)

![](images/4ee2ebd5b89bc3e57db857fa1178ca9644d2c5c6c155bec56d362b048b43b2db.jpg)

![](images/1e80efd27611ef1af29aaf1631c280d2ee6ab59d2a28801fb9af2aaa59d720e9.jpg)

![](images/c7b8dc4495074d63bc72a19834ee93f0180b9d71142290a7d0a2432ffa1b482f.jpg)  
Model Performance: Test Accuracy: 0.858 Precision: 0.800 Predicted: Malignant Confidence: 0.956

(b) Integrated Gradients  
![](images/8faef5997df0e046152694b7188a83e2dec30eebddbe696c44bc5c69dadb39b4.jpg)

![](images/4a2af147c430b5ffd1a5b41fcce96bc6a7494c0db1850c2fe08ed69cb0cdbf2e.jpg)

![](images/2b8289de1b3bb09e7dee38e0f8a551aeece2f9ac85a02ddc46a7cd057afdee4e.jpg)

![](images/deb4999790abdc3245146ee2bd4f177b08f3d74da0a30a4a7fb929ab2d9d12fe.jpg)

![](images/b6347f5886f651e8f21e75233a780780a79dd37d4da6b5364fe5384b313d2a9c.jpg)

![](images/f89b1b100ff68a0bbf3a5bdedbc97b939cb7091f6afbd2780b87d31c56702c39.jpg)

![](images/c536ff5c3136c7f786c2b8283bc3f1570aa4636ae0dd1155385a83b14be2bc8d.jpg)

![](images/fb9b41840003175e0adfcb7690b5f5c51fe73bedfefe28e593add6b3e4ed4dc2.jpg)  
(c) LIME

![](images/25429988ae97befde054f418760a33357cd3e62a8d434ae32d90f6cc4a9ba799.jpg)

![](images/b6f29d3af228a67b9b4e6f46e3672ae9ee00c9bb031d5e81705ee9fef54e1120.jpg)

![](images/c0c8a881a5c06ee36c558cea6a6338311e7d5df92c4c8935952d4ae6235f9c4b.jpg)  
Model Performance: Test Accuracy: 0.994 F1 Score: 0.997 Predicted: Benign Confidence: 0.988  
(d) LRP  
Figure 39: Explainable AI (XAI) visualizations of DenseNet121 predictions on the BreakHis dataset using (a) Grad-CAM, (b) Integrated Gradients, (c) LIME, and (d) Layer-wise Relevance Propagation (LRP).

The EfficientNet model, trained on the BreakHis dataset, attained a test accuracy of 0.862, recall of 0.860, precision of 0.836, F1-score of 0.848, and AUC-ROC of 0.940, with benign predictions exhibiting high confidence levels (0.956–0.969) (Figure 40). Grad-CAM representations at layers 2, 4, 6, and 8 demonstrated an increasingly finer concentration on nuclei-rich and glandular structures, whereas Integrated Gradients consistently highlighted epithelial boundaries and clustered nuclei across various baselines. LIME validated localized decision boundaries via superpixellevel attributions, while Saliency Maps emphasized pixel-level gradients indicative of tissue shape. The XAI approache collectively demonstrate that EfficientNet, while slightly less accurate than DenseNet121, relies on clinically significant histopathological traits, hence ensuring interpretability and robust performance.

![](images/e756a02345da0a66c9e89ef66ae2f53b03fbf6152f3d155a485b4c3434c3877a.jpg)  
EfficientNet Performance: Test Accuracy: 0.862 AUC-ROC: 0.940 Predicted: Benign Confidence: 0.969

![](images/d695ac0bf0165d77c44d925dd7e9326ad247fa605ca740a58d8958449bde6e79.jpg)

![](images/8a148c27112eaa5d97c09619983929251d76c4bfb01fd5516ad3f3d7aae7155c.jpg)

![](images/fc8433f3e5286bbb417e26a3e1fd0c9e3c0ae0f5a08b7263adc777b23d35cb29.jpg)

![](images/1f4c1880e16aed942126d52b78e497fe6c8e3989439ea248a705a4c1b56ab87c.jpg)

![](images/f986d94d52879db808f94a846e62ff8637d503cd66b51f3ec995362691c586f0.jpg)  
EfficientNet Performance: Test Accuracy: 0.862 Recall: 0.860 Predicted: Benign Confidence: 0.969

![](images/348e09bf22419c8fb2ba5466f81d9d60771fb39bb137cb81da30ffc18d7da6d9.jpg)

![](images/248e196db74a8a9abc99c090a989821b87e4226f594aba64893fb5e1af0d1870.jpg)

![](images/38ed244ebcdce1bf58c0a0f52cedcbbd2b5cc3b20b1d0404ac295e9d76d71447.jpg)

![](images/149d2f4d5fd0cb9a116fe1a4a95a55484e1429320b0290edfafb0814ececd30e.jpg)

![](images/75773d98320268ccba5f1cf7881c080677e60ce05e4333ccf2aa00e690beb0a3.jpg)  
EfficientNet Performance: Test Accuracy: 0.862 Precision: 0.836 Predicted: Benign Confidence: 0.969

(a) Grad-CAM  
(b) Integrated Gradients  
![](images/2790bda1ccaaa5a4979fc37f60afc9140dd95d5fa5b9229737ad2abcc42c233d.jpg)

LIME Analysis - Predicted: Benign (0.969  
![](images/ed8e79721b06a4683503207ab392f1ef15f96bb687542cbdb261e8d8e521d003.jpg)

![](images/81476474856df39513fe705f9e0063cf2496c5114445077989f1a9b6962bff9e.jpg)

![](images/7f48e41a1b8c6f62782af7fb63e8f397039dd302f874a420b0f41c5257dbe66b.jpg)

![](images/05ad88009f71ce85fc779e87a3b84304b156ee0033b93816648030e62d88a5ad.jpg)

![](images/5bc052224b1dd2e8ab06e02f3befae1d8b3a1062ac262e2488c295bfddc0aa56.jpg)  
(c) LIME

![](images/2dafa76c499b84907ab85450a212912c08886932fa99d38175fb0eda7de9f29b.jpg)

![](images/4b56bbf96c05b1cf57bf5990ac96fb40be96e567cb6683ca08f1238b86644f3c.jpg)

![](images/13cf4eaefd4c66898124a134c5f9c5bdb6915520e6072944560eb0aa4fb1e0d1.jpg)  
(d) Saliency Map  
Figure 40: XAI visualizations of EfficientNet predictions on the BreakHis dataset using (a) Grad-CAM, (b) Integrated Gradients, (c) LIME, and (d) Saliency Maps.

![](images/1fe88f049011e794a4446fee884fff839f6d333e5fd96ffb822470cbb35c0966.jpg)

![](images/e47ba7b8894c75a720d62609fc3fc28ae6a10264a0d7cb17efd92b14a1b2a77a.jpg)

![](images/d2d78d20458763ef70e1e1e65a9954b1655801c93cc57ec6c70520038af04a25.jpg)

![](images/36f4ecff2046d507c4016992ee69bf5e3c6b9b68920f71ccdfdc9fdbc40b463e.jpg)

![](images/802ce9b5e938936e355c584e2ceed8187ce8b7695a6e34ea1cc548bc0d05f470.jpg)

![](images/a548fe6afddd6ae313dff622547afa534677da0d7ed8203500ccf24593a90874.jpg)

![](images/5f2301178ba66b623124707c3ea879d1d853a23d82cd8032cee5941ad21f24d2.jpg)

![](images/4c32088866f44c5cf6153ba98634f0360a8617bac9a7b4ccd1c2fae815100a2d.jpg)

![](images/48cf165acc47149b2cf6922dd41fb2a2153e34217a5354b954abd5aa736b299f.jpg)

![](images/f6d8c6b919a7eb6b8efa714a90c99f32d9f9729bc11cc0c62a405540f7422f0d.jpg)  
(a) Grad-CAM

(b) Integrated Gradients  
![](images/73561401bdc0f03435ec6b7ac8c10aa44a3f564b6f5b3804d40f418fc4e8fe06.jpg)

![](images/34c6ac940e8c1bf0989463f705dde86d23c4724f2f256be29435faa09dc9aeb0.jpg)

![](images/4e7f4b2525ec143dd5f40a142b41fa499696e362d25f2e5b212863d57b7080bc.jpg)

![](images/f594e06a5c6fb31f98c78989284397f96c9fd99c0bfdac8651bc8f667c20a484.jpg)

![](images/51aed5e9a530d9d357e34d0b956ac775ba985d62cd4f779595b75e12df7b376a.jpg)

![](images/b9b1e8ac78755a32c66e9b6a1f7b2cffa15205ed8c34d13b6023c0a4d06996c8.jpg)

![](images/33a160acf327b9f884e9860933458add61408901dc12d0f45d023883f5aca638.jpg)

![](images/9a71d91b683c20200d89176fcf1925dd24bed5453ad80b7071d2021375e843b7.jpg)

![](images/565d162356d76d284a715bceabd6c0d200e5acfdec3679bb4e657e474c5aa2a0.jpg)  
(c) LIME

![](images/b630673f7ee475e34a54d3844797aea9b4a82b41cb206dd8797554f8499468c3.jpg)

![](images/83bc04111f2c06d10d597f29768f685db2dc47e35bfbf510dc68543451fc3c22.jpg)  
(d) LRP  
Figure 41: XAI visualizations of InceptionV3 predictions on the BreakHis dataset using (a) Grad-CAM, (b) Integrated Gradients, (c) LIME, and (d) Layer-wise Relevance Propagation (LRP).

![](images/a1bfa0d2d7a01a1cf2e36f92f3a6e934e55c526b745e06fc5041e5cedf53960f.jpg)

![](images/3841c5df8048dafa4c2327cd8d9edec3242c188b67a238dec8b8e842dbd9c134.jpg)  
edicted B

![](images/48148465cfeba5e8ec2442761e74e5bc5af479c5bf1dc90fe9627d0024da2d3e.jpg)

![](images/97fce92c21082a2655a5a7c167582f95bf5a6c815998b8bd231fa3cd6bbb7d56.jpg)

![](images/1e914c9130c5ba4f60f2f069e5a17c4f268e0d29d66448ad9c2c2ab843082764.jpg)

![](images/3c06cfd1f3d0c9f7df2cea80c36eadceb87309f283d28f8d1f03ace706d05e71.jpg)

![](images/a88231eca1a9d54e10e1537aa235c62d255cc7e7185ae6a5df54fdc2f8cc3d1a.jpg)  
(a) Grad-CAM

(b) Integrated Gradients  
![](images/34f5a667700e1ee7a17da042dd9d239b57a3346f098dc0b4408ff75edba031dd.jpg)  
(c) LIME  
Figure 42: XAI visualizations of MobileNetV2 predictions on the BreakHis dataset using (a) Grad-CAM, (b) Integrated Gradients, (c) LIME, and (d) LRP.

## 7.1 Statistical Significance, Robustness, and Generalization Analysis

The overall performance ranks derived from the classification experiments are well supported by the statistical validation results. Among all evaluated architectures, DenseNet121 consistently emerged as the best-performing model, achieving an average test accuracy of 99.4% (95% CI: 98.6-99.7%), an AUC-ROC of 0.999 (95% CI: 0.9963–1.0000), precision of 99.9%, recall of 99.5%, F1-score of 99.6%, Cohen’s κ of 0.986, and an MCC of 0.984. The high agreement coefficients show the high level of consistency between predicted labels and true labels, and is the reason why DenseNet121 achieved the lowest Friedman rank (1.00) and statistically outperformed all other models $( p < 0 . 0 0 1 )$ . Furthermore, it is magnifications sensitive under 1.08% across all the magnifications of 40×, 100×, 200× and 400× levels, showing excellent robustness and generalization capability in different tissue magnifications.

Table 11: Comprehensive statistical performance summary of all models on the BreakHis test set.
<table><tr><td>Model</td><td>Arch.</td><td>Acc. (%)</td><td>Prec. (%)</td><td>Recall (%)</td><td>Spec. (%)</td><td>F1 (%)</td><td>AUC</td><td>κ</td><td>MCC</td><td>F-Rank</td></tr><tr><td>DenseNet121</td><td>CNN</td><td>99.4 [98.6, 99.7]</td><td>99.9</td><td>99.5</td><td>99.2</td><td>99.6</td><td>0.999 [0.9963, 1.0000]</td><td>0.986</td><td>0.984</td><td>1.00</td></tr><tr><td>EfficientNet</td><td>CNN</td><td>86.2 [83.6, 88.4]</td><td>83.6</td><td>86.0</td><td>86.3</td><td>84.8</td><td>0.940 [0.9192, 0.9608]</td><td>0.685</td><td>0.700</td><td>3.62</td></tr><tr><td>Xception</td><td>CNN</td><td>85.8 [83.2, 88.1]</td><td>80.0</td><td>91.2</td><td>81.5</td><td>85.2</td><td>0.939 [0.9180, 0.9600]</td><td>0.676</td><td>0.689</td><td>3.50</td></tr><tr><td>InceptionV3</td><td>CNN</td><td>84.6 [81.9, 86.9]</td><td>79.5</td><td>88.4</td><td>81.5</td><td>83.7</td><td>0.933 [0.9111, 0.9549]</td><td>0.648</td><td>0.665</td><td>4.00</td></tr><tr><td>MobileNetV2</td><td>CNN</td><td>84.7 [82.0, 87.0]</td><td>83.5</td><td>81.9</td><td>86.9</td><td>82.7</td><td>0.933 [0.9111, 0.9549]</td><td>0.651</td><td>0.673</td><td>4.12</td></tr><tr><td>ResNet50</td><td>CNN</td><td>83.6 [80.9, 86.0]</td><td>79.1</td><td>86.2</td><td>81.5</td><td>82.5</td><td>0.919 [0.8950, 0.9430]</td><td>0.625</td><td>0.645</td><td>5.25</td></tr><tr><td>VGG19</td><td>CNN</td><td>81.3 [78.4, 83.9]</td><td>76.5</td><td>84.1</td><td>79.0</td><td>80.1</td><td>0.905 [0.8792, 0.9308]</td><td>0.573</td><td>0.599</td><td>7.50</td></tr><tr><td>Swin Transformer</td><td>ViT/Swin</td><td>79.5 [76.5, 82.2]</td><td>73.6</td><td>84.6</td><td>75.3</td><td>78.7</td><td>0.860 [0.8293, 0.8907]</td><td>0.532</td><td>0.564</td><td>8.00</td></tr><tr><td>ViT</td><td>ViT/Swin</td><td>77.6 [74.6, 80.4]</td><td>71.1</td><td>84.3</td><td>72.2</td><td>77.2</td><td>0.858 [0.8271, 0.8889]</td><td>0.488</td><td>0.530</td><td>8.75</td></tr><tr><td>ConvNeXt</td><td>CNN</td><td>77.1 [74.0, 79.9]</td><td>85.1</td><td>89.1</td><td>2.4</td><td>87.0</td><td>0.653 [0.6110, 0.6950]</td><td>0.477</td><td>-0.180</td><td>9.25</td></tr></table>

EfficientNet, Xception, InceptionV3 and MobileNetV2 were the second tier of models, with an accuracy of 84.6% to 86.2% and AUC ranging between 0.933 and 0.940. They were not far off their statistical measures from their predictive

![](images/003baa6f4724c936b9430de7edb40438a4bb03896d9d9a7022cf1474aaa51812.jpg)  
Figure 43: Statistical comparison of the 10 models tested in the patient-stratified BreakHis test set, with test accuracy and AUC-ROC and their 95% confidence intervals quantifying predictive uncertainty and comparative performance.

performance. The best agreement metrics with DenseNet $( \kappa = 0 . 6 8 5$ and $\mathbf { M C C } = 0 . 7 0 0 )$ and the lowest magnification variability $( \sigma = \pm 0 . 7 5 \% )$ , EfficientNet earned the top ranking (Friedman rank = 3.62), with high performance at all magnifications. Likewise, Xception achieved a high sensitivity (91.2%) and an MCC of 0.689, meaning that it had a high recall value and good predictive consistency. The moderate agreement between both models (InceptionV3 and MobileNetV2, $\kappa = 0 . 6 5 )$ and similar AUC values (0.933) helps to justify the models’ statistically similar performance when compared pairwise.

The statistical results also account for the relatively poor performance of ResNet50 and VGG19. Both models had satisfactory accuracy (83.6% and 81.3%) and relatively low agreement coefficient (0.625 and 0.573), and moderate MCC values (0.645 and 0.599), which showed less reliable classification boundaries and less robustness in variable magnification levels. Because larger magnification sensitivities, $\sigma = \pm 2 . 3 6 \% \mathrm { a n d } \pm 1 . 9 9 \%$ on image magnification were suggested with their larger effects of dependence on image scale variations, they had lower Friedman rankings.

The transformer based architectures (Swin Transformer and Vision Transformer (ViT)) had relatively poor accuracy with 79.5% and 77.6%, respectively, and AUC values less than 0.86. There is moderate agreement between these results and their moderate agreement statistics $( \kappa = 0 . 5 3 2$ and 0.488) and lower MCC scores (0.564 and 0.530). In addition, both architectures had the largest magnification sensitivities $( \sigma = \pm 2 . 9 9 \%$ and ±3.18%), and very poor Friedman rankings (8.00 and 8.75) which signify that they were not stable and showed the least robust results for different image resolutions. The Mann–Whitney U test also found that CNN architectures were significantly more accurate $( p = 0 . 0 0 3 2 )$ and more accurate for the AUC $( p = 0 . 0 0 4 7 )$ than the transformers, indicating the advantage of convolutional inductive biases for relatively small histopathological datasets like BreakHis.

![](images/fb955143a222de5d6166752b65fc66af130d69fa61918aaa565498023447f57f.jpg)  
Figure 44: Friedman and Nemenyi non-parametric analyses comparing the ten architectures across four magnification levels that display statistically significant differences and average performance ranking of the evaluated models.

Lastly, ConvNeXt was the worst-performing architecture in terms of accuracy (77.1%), AUC (0.653) and MCC (-0.180) although it had a decent F1 score (87.0%). The negative MCC value and low value of $\kappa ( 0 . 4 7 7 )$ suggest poor discriminator capability and prediction bias to the majority class. This is also supported by its low Friedman ranking (9.25) and the high dispersion in the AUC values at different magnifications. The combined results of these statistical analyses provide further validation and corroboration of the experimental results: Conventional classification metrics validate the performance hierarchy as well as rigorous significance testing, agreement measures, confidence intervals and robustness analyses, which significantly adds to the credibility and reproducibility of the results reported.

![](images/0886e01e58d621b1d733140c8f4d381563b255278f4b5a7e5197b659c810e4f3.jpg)  
Figure 45: Magnification sensitivity analysis showing the robustness and stability of the model performance at different histopathological magnifications (40×, 100×, 200× and 400×) using the accuracy, AUC-ROC and deviation from accuracy metrics.

## 8 Limitations

Although different transfer learning architectures were comprehensively benchmarked on BreakHis and various explainable AI techniques were used, there are still limitations. Firstly, the study was based on a single histopathological data set and a binary classification scenario to limit the generalizability of the result to other imaging modalities, multi-class classification of subtypes, tumour grading, and staging for clinical scenarios. All statistical measures were conducted in ways that ensure the superiority of DenseNet121, and these tests and the perfect scores at several magnifications in itself should be treated with a grain of salt due to near-perfect performance (99.4% accuracy, AUC-ROC of 0.999, $\kappa = 0 . 9 8 6$ and $\mathsf { M C C } = 0 . 9 8 4 )$ . Histopathological datasets, like BreakHis, may have nuances within and between magnifications that can create implicit correlations and result in inflated performance metrics. Moreover, the difference between the positive validation and test performance, as well as the very high agreement scores indicate that some degree of learning of the validation set may exist, in spite of the statistically significant improvement in performance. Furthermore, the XAI methods used are mostly qualitative and do not measure explanation faithfulness and clinical utility, quantitatively. Lastly, the complexity of the memory and computational hardware of large transformerbased architectures limited the exploration of a wide range of hyperparameters and external experiments. However, additional testing on independent multi-institutional data and quantitative assessment of interpretability would be required before confidently implementing for clinical use.

## 9 Conclusion & Future Work

In this work, we benchmarked 10 state-of-the-art transfer learning architectures in order to classify breast cancer on the BreakHis dataset: CNN-based models (ResNet50, DenseNet121, EfficientNetV2, MobileNetV2, VGG19, Xception, ConvNeXt and InceptionV3) and transformer-based models (Vision Transformer and Swin Transformer). Both experimental and statistical analyses showed that DenseNet121 yielded the highest accuracy (99.41%), AUC-ROC (0.9997), Cohen’s κ (0.986), and MCC (0.984) and was statistically significantly better than all models compared $( p < 0 . 0 0 1 )$ . It was also shown to be robust in terms of magnification in tests at all the image resolutions and was found to be only slightly sensitive to variations in magnification $( \sigma = \pm 1 . 0 8 \% )$ . Moreover, Grad-CAM and LIME provided clinically relevant visual explanations for DenseNet121, EfficientNet and Swin Transformer, consistently identifying nuclei-dense and glandular regions that are crucial for diagnostic decisions. Together, these results highlight the potential of combining the strengths of transfer learning architectures with explainable AI approaches to achieve reliable clinical decision-making support systems for breast cancer diagnosis that are accurate, transparent, and clinically relevant.

However, the results are to be viewed with some caution as the study is based upon one histopathological dataset and binary classification. The proposed framework will be validated with independent multi-institutional and multi-modal data sets in future, extended to subtyping and grading tasks, explored on the hybrid CNN-Transformer architectures, and validated quantitatively and clinically to increase the generalization capabilities, reliability, and clinical utility.

## Author contributions statement

Author 1 analyzed the results and reviewed the manuscript draft. Authors 2 and 4 contributed to the overall methodology, original draft preparation, and data analysis. Author 3 conducted the experiments and generated the results. Authors 5 and 6 were responsible for project administration, results evaluation. All authors reviewed, edited, and approved the final version of the manuscript.

## Data Availability

The datasets analysed during the current study are publicly available. This study utilised the BreakHis (Breast Cancer Histopathological Image Classification) dataset, which is accessible through the public repository at: https: //web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

No new datasets were generated during this study. All processed data supporting the findings of this research are available from the corresponding author upon reasonable request.

## Ethics Statement

This study used the publicly available BreakHis dataset consisting of de-identified histopathological images. No direct human participation, patient recruitment, or intervention was conducted by the authors. The dataset was originally collected following relevant clinical and ethical guidelines, and all images are anonymised. Therefore, additional ethical approval and informed consent were not required for this secondary analysis.

## Acknowledgments

This work was supported by the Deanship of Scientific Research, Vice Presidency for Graduate Studies and Scientific Research, King Faisal University, Saudi Arabia [Grant No.XXXXXX].

## References

[1] Marwa M Emam, Essam H Houssein, Nagwan Abdel Samee, Manal Abdullah Alohali, and Mosa E Hosney. Breast cancer diagnosis using optimized deep convolutional neural network based on transfer learning technique and improved coati optimization algorithm. Expert Systems with Applications, 255:124581, 2024.

[2] Weimin Wang, Yufeng Li, Xu Yan, Mingxuan Xiao, and Min Gao. Breast cancer image classification method based on deep transfer learning. In Proceedings ofthe International Conference on Image Processing, Machine Learning and Pattern Recognition, pages 190–197, 2024.

[3] Jacinta Potsangbam and Salam Shuleenda Devi. Classification of breast cancer histopathological images using transfer learning with densenet121. Procedia Computer Science, 235:1990–1997, 2024.

[4] Madhusudan G Lanjewar, Kamini G Panchbhai, and Lalchand B Patle. Fusion of transfer learning models with lstm for detection of breast cancer using ultrasound images. Computers in Biology and Medicine, 169:107914, 2024.

[5] Tharunya Arravalli, Krishnaraj Chadaga, H Muralikrishna, Niranjana Sampathila, D Cenitta, Rajagopala Chadaga, and KS Swathi. Detection of breast cancer using machine learning and explainable artificial intelligence. Scientific Reports, 15(1):26931, 2025.

[6] Md Romzan Alom, Fahmid Al Farid, Muhammad Aminur Rahaman, Anichur Rahman, Tanoy Debnath, Abu Saleh Musa Miah, and Sarina Mansor. An explainable ai-driven deep neural network for accurate breast cancer detection from histopathological and ultrasound images. Scientific Reports, 15(1):17531, 2025.

[7] Neha Rani, Deepak Kumar Gupta, and Samayveer Singh. Multi-class classification of breast cancer abnormality using transfer learning. Multimedia Tools and Applications, 83(30):75085–75100, 2024.

[8] SR Sannasi Chakravarthy, N Bharanidharan, V Vinoth Kumar, TR Mahesh, Mohammed S Alqahtani, and Suresh Guluwadi. Deep transfer learning with fuzzy ensemble approach for the early detection of breast cancer. BMC Medical Imaging, 24(1):82, 2024.

[9] Ritesh Maurya, Nageshwar Nath Pandey, Malay Kishore Dutta, and Mohan Karnati. Fccs-net: Breast cancer classification using multi-level fully convolutional-channel and spatial attention-based transfer learning approach. Biomedical Signal Processing and Control, 94:106258, 2024.

[10] Melwin D Souza, G Ananth Prabhu, Varuna Kumara, and KM Chaithra. Earlynet: a novel transfer learning approach with vgg11 and efficientnet for early-stage breast cancer detection. International Journal of System Assurance Engineering and Management, 15(8):4018–4031, 2024.

[11] Mohammad Mobin Mirimoghaddam, Jafar Majidpour, Fakhereh Pashaei, Hossein Arabalibeik, Esmaeil Samizadeh, Nema Mohmadian Roshan, and Tarik A Rashid. Her2gan: Overcome the scarcity of her2 breast cancer dataset based on transfer learning and gan model. Clinical Breast Cancer, 24(1):53–64, 2024.

[12] Belgacem Bouallegue, Yasser M Abd El-Latif, Hosam El-Sofany, and Islam ATF Taj-Eddin. An enhanced approach for predicting breast cancer using different deep learning algorithms and explainable ai techniques in an iot environment. International Journal ofIntelligent Systems, 2025(1):8884481, 2025.

[13] Dhayanithi Jaganathan, Sathiyabhama Balasubramaniam, Vidhushavarshini Sureshkumar, and Seshathiri Dhanasekaran. Revolutionizing breast cancer diagnosis: a concatenated precision through transfer learning in histopathological data analysis. Diagnostics, 14(4):422, 2024.

[14] Fabián Silva-Aravena, Hugo Núñez Delafuente, Jimmy H Gutiérrez-Bahamondes, and Jenny Morales. A hybrid algorithm of ml and xai to prevent breast cancer: A strategy to support decision making. Cancers, 15(9):2443, 2023.

[15] Faizan Nasir, Shanur Rahman, Nazim Nasir, FAIZAN NASIR, SHANUR RAHMAN, and NAZIM NASIR. Breast cancer detection using convolutional neural networks: A deep learning-based approach. Cureus, 17(5), 2025.

[16] Kalappanaickenpatty Suriaprakasam Manojee and Athiappan Rajiv Kannan. Patho-net: enhancing breast cancer classification using deep learning and explainable artificial intelligence. American Journal ofCancer Research, 15(2):754, 2025.

[17] Md Zobayer Ibna Kabir. Breast cancer classification using pre-trained cnns with explainable ai for enhanced decision support. In 2025 International Conference on Electrical, Computer and Communication Engineering (ECCE), pages 1–6. IEEE, 2025.

[18] Vinayak Gupta and Richa Sharma. Enhancing breast cancer prediction with xai-enabled boosting algorithms. In 2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT), pages 1–5. IEEE, 2024.

[19] Md Alamin Talukder. An improved xai-based densenet model for breast cancer detection using reconstruction and fine-tuning. Results in Engineering, 26:104802, 2025.

[20] Jamalur Ridha, Khairun Saddami, Muhammad Riswan, Roslidar Roslidar, et al. An explainable artificia intelligence framework for breast cancer detection. Indonesian Journal of Electronics, Electromedical Engineering, and Medical Informatics, 7(2):298–311, 2025.

[21] Kuncham Sreenivasa Rao, Panduranga Vital Terlapu, D Jayaram, Kalidindi Kishore Raju, G Kiran Kumar, Rambabu Pemula, M Venu Gopalachari, and S Rakesh. Intelligent ultrasound imaging for enhanced breast cancer diagnosis: Ensemble transfer learning strategies. IEEE Access, 12:22243–22263, 2024.

[22] Raafat M Munshi, Lucia Cascone, Nazik Alturki, Oumaima Saidani, Amal Alshardan, and Muhammad Umer. A novel approach for breast cancer detection using optimized ensemble learning framework and xai. Image and Vision Computing, 142:104910, 2024.

[23] Salma Fayaz, Syed Zubair Ahmad Shah, Nusrat Mohi ud din, Naillah Gul, and Assif Assad. Advancements in data augmentation and transfer learning: A comprehensive survey to address data scarcity challenges. Recent advances in computer science and communications (Formerly: Recent patents on computer science), 17(8):14–35, 2024.

[24] Sara Sumaiyya, G Shyamala, NR Latha, and GB Pallavi. Breast cancer detection with mobilenetv2 and explainable ai. In 2024 4th International Conference on Sustainable Expert Systems (ICSES), pages 1237–1243. IEEE, 2024.

[25] Gurusiddayya Hiremath, Jose Alex Mathew, and Naveen Kumar Boraiah. Hybrid statistical and texture features with densenet 121 for breast cancer classification. International Journal of Intelligent Engineering & Systems, 16(2), 2023.

[26] Raj Kumar Pattanaik, Satyasis Mishra, Mohammed Siddique, Tiruveedula Gopikrishna, and Sunita Satapathy. Breast cancer classification from mammogram images using extreme learning machine-based densenet121 model. Journal ofSensors, 2022(1):2731364, 2022.

[27] Sakinat Oluwabukonla Folorunso, Joseph Bamidele Awotunde, Y Pandu Rangaiah, and Roseline Oluwaseun Ogundokun. Efficientnets transfer learning strategies for histopathological breast cancer image analysis. International Journal ofModeling, Simulation, and Scientific Computing, 15(02):2441009, 2024.

[28] Majid Behzadpour, Bengie L Ortiz, Ebrahim Azizi, and Kai Wu. Breast tumor classification using efficientnet deep learning model. arXiv preprint arXiv:2411.17870, 2024.

[29] Ojonugwa Oluwafemi Ejiga Peter, Daniel Emakporuena, Bamidele Dayo Tunde, Maryam Abdulkarim, and Abdullahi Bn Umar. Transformer-based explainable deep learning for breast cancer detection in mammography: The mammoformer framework. arxiv.org, 2025.

[30] Shakil Khan, Hamdadur Rahman, Md Ismail Hossain Siddiqui, Zishad Hossain Limon, Mahbub Alam Khan, Rezaul Haque, Md Redwan Ahmed, Ahmed Wasif Reza, and Shamim Hasnat Ripon. Ensemble-based explainable approach for rare medicinal plant recognition and conservation. In 2025 10th International Conference on Information and Network Technologies (ICINT), pages 88–93. IEEE, 2025.

[31] Awika Ariyametkul, Sudarshan Tamang, and May Phu Paing. Explainable ai (xai) for breast cancer diagnosis. In 2024 16th Biomedical Engineering International Conference (BMEiCON), pages 1–5. IEEE, 2024.

[32] R Reenadevi, T Sathiya, and B Sathiyabhama. Breast cancer histopathological image classification using augmentation based on optimized deep resnet-152 structure. Annals of the Romanian Society for Cell Biology, 25(6):5866–5874, 2021.

[33] Syed Usama Khalid Bukhari, Asmara Syed, Syed Safwan Khalid, and Syed Sajid Hussain Shah. The histologica diagnosis of breast cancer by employing scale invariant resnet 18 with spatial supervised technique. medRxiv, pages 2021–09, 2021. DOI:10.1145/3492323.3495596.

[34] Wasyihun Sema Admass, Yirga Yayeh Munaye, and Ayodeji Olalekan Salau. Integration of feature enhancement technique in google inception network for breast cancer detection and classification. Journal of Big Data, 11(1):78, 2024.

[35] S Kanimozhi and S Priyadarsini. Breast cancer histopathological image classification using cnn and vgg-19. In 2024 Third International Conference on Intelligent Techniques in Control, Optimization and Signal Processing (INCOS), pages 1–6. IEEE, 2024.

[36] Md Humaion Kabir Mehedi, Moumita Khandaker, Shaneen Ara, Md Ashraful Alam, MF Mridha, and Zeyar Aung. A lightweight deep learning method to identify different types of cervical cancer. Scientific Reports, 14(1):29446, 2024.

[37] Marwa Naas, Hiba Mzoughi, Ines Njeh, and Mohamed BenSlima. An explainable ai for breast cancer classification using vision transformer (vit). Biomedical Signal Processing and Control, 108:108011, 2025.

[38] Aymen M Al-Hejri, Archana Harsing Sable, Riyadh M Al-Tam, Mugahed A Al-Antari, Sultan S Alshamrani, Kaled M Alshmrany, and Wedad Alatebi. A hybrid explainable federated-based vision transformer framework for breast cancer prediction via risk factors. Scientific Reports, 15(1):1–23, 2025.

[39] Hossam Magdy Balaha, Khadiga M Ali, Dibson Gondim, Mohammed Ghazal, and Ayman El-Baz. Harnessing vision transformers for precise and explainable breast cancer diagnosis. In International Conference on Pattern Recognition, pages 191–206. Springer, 2025.

[40] Gelan Ayana and Se-woon Choe. Vision transformers-based transfer learning for breast mass classification from multiple diagnostic modalities. Journal ofElectrical Engineering & Technology, 19(5):3391–3410, 2024.

[41] Md Redwan Ahmed, Hamdadur Rahman, Zishad Hossain Limon, Md Ismail Hossain Siddiqui, Mahbub Alam Khan, Al Shahriar Uddin Khondakar Pranta, Rezaul Haque, SM Masfequier Rahman Swapno, Young-Im Cho, and Mohamed S Abdallah. Hierarchical swin transformer ensemble with explainable ai for robust and decentralized breast cancer diagnosis. Bioengineering, 12(6):651, 2025.

[42] Farhan Farabi, Farhan Hossen, Farhan Monsur, Mir Araf Hossain, and Md Mohibul Hasan. Explainable breast cancer detectionfrom Histopathology images using transfer learning and XAI. PhD thesis, Brac University, 2023.

[43] Tamilarasi Kathirvel Murugan, Pritikaa Karthikeyan, and Pavithra Sekar. Efficient breast cancer detection using neural networks and explainable artificial intelligence. Neural Computing and Applications, 37(5):3759–3776, 2025.

[44] Kaushik Raghavan, Sivaselvan B, and Kamakoti v. Attention guided grad-cam: an improved explainable artificial intelligence model for infrared breast cancer detection. Multimedia Tools and Applications, 83(19):57551–57578, 2024.

[45] Adyasha Sahu, Pradeep Kumar Das, and Sukadev Meher. An efficient deep learning scheme to detect breast cancer using mammogram and ultrasound breast images. Biomedical Signal Processing and Control, 87:105377, 2024.

[46] Ahmed Alhussen, Mohd Anul Haq, Arfat Ahmad Khan, Rakesh Kumar Mahendran, and Seifedine Kadry. Xairacapsnet: Relevance aware capsule network-based breast cancer detection using mammography images via explainability o-net roi segmentation. Expert Systems with Applications, 261:125461, 2025.

[47] Sharia Arfin Tanim, Gazi Mohammad Imdadul Alam, Tahmid Enam Shrestha, Maruful Islam, Fariha Jahan, and Kamruddin Nur. Breast cancer diagnosis with xai-integrated deep learning approach. In 2024 International Conference on Innovation and Intelligencefor Informatics, Computing, and Technologies (3ICT), pages 659–665. IEEE, 2024.