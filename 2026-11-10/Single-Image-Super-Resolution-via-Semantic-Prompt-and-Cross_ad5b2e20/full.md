# Single Image Super-Resolution via Semantic Prompt and Cross-Modal Training on Content Area

Xiao Tan

Xidian University, Xi’an 710071, China

E-mail address: 21171213976@stu.xidian.edu.cn

## Abstract

Enhancing image quality through super-resolution is crucial in image and video processing. While contemporary deep learning models, such as CNNs and Transformers, excel in pixellevel enhancements, they often overlook the rich semantic context within images. To address this, we introduce a novel approach that synergizes pixel intensity with semantic information for superior detail recovery. Our method involves extracting semantic data, including textual descriptors and spatial positioning, and encoding this information. We also present a Cross-Modal Feature Fusion module for generating semantic feature maps via multimodal training, and a Prompt Traction Head to augment the backbone network with semantic insights, thereby refining image details. Our empirical results across various datasets validate the superiority of our approach.

Keywords: Image Super-Resolution; Multimodality; Computer Vision; Deep Learning

## 1 Introduction

Image super-resolution, a classic computer vision task, upscales low-resolution images to highresolution ones without information loss [1]. It finds applications in image compression transmission, detail enhancement, and defect restoration [2]. Recently, Convolutional Neural Network (CNN) and transformer-based methods have been proposed [3, 4, 5, 6, 7, 8, 9, 10, 11], exploiting their feature extraction capabilities for image detail restoration. However, these methods overlook the rich semantic information within images, which, if exploited, could significantly improve output quality.

This paper proposes a novel approach that leverages semantic image information using models like YOLO [12] and BLIP [13]. The extracted information is incorporated into the super-resolution model’s training process as a prompt. Additionally, an attention mechanism is employed to apply the encoded information to specific image areas, reflecting the regional properties of image semantics.

Figure 1 illustrates the diference between our approach and traditional super-resolution methods. Traditional CNN-based methods (Figure 1(a)) extract features at diferent scales, which are then fused and upsampled to restore image details [6]. Transformer-based methods (Figure 1(b)) divide the image into patches, transform them into fixed-length vectors, and process them through selfattention mechanisms [14, 15]. Our approach (Figure 1(c)) incorporates semantic extraction and cross-modal feature fusion modules, extracting textual semantic information from the image and encoding it into feature vectors that serve as prompts for the super-resolution model.

![](images/8b93cdde8ece092bea25b66be360534605bc9205d4c5d07885bbc6da9d852a88.jpg)

![](images/de48600ac1385e5fb137b627bcd9e10fb3f1c338b7d432a897430d62f9d66863.jpg)  
(b) Transformer-based image super-resolution model.

(a) Classic CNN-based image super-resolution models.  
![](images/4b42b9341c9270f9f85ad5ece632ac9874640e3f50b773427685df02cd723714.jpg)  
(c) Proposed image super-resolution model.

![](images/7b5fb88b2459a47c305c7415e27bd1ea7d66e7a37324b92d004cfecaa2fd190d.jpg)  
(d) Key content areas for diferent image types.  
Figure 1: Comparison of diferent image super-resolution model architectures.

Semantic information varies across image categories due to content diferences (Figure 1(d)). For natural images, the main content may consist of objects, people, animals, etc., while document images primarily consist of text with substantial blank areas between paragraphs [16]. Therefore, we employ an Image Caption model [17] and an Object Detection model [18] for natural images, and an OCR model [19] for document images.

We conducted extensive experiments to validate our approach, comparing it with several stateof-the-art super-resolution algorithms. The results demonstrate that our method efectively utilizes image semantic information, improving various performance metrics and visual efects of the model.

In summary, this paper contributes to image super-resolution by proposing a novel method that utilizes image semantic information for model training, improving model performance. We design diferent schemes for extracting semantic information for diferent types of images and establish associations between the extracted semantic information and the corresponding image area. Extensive experiments validate the proposed method, demonstrating its efectiveness in utilizing semantic information and improving various performance metrics and visual efects.

## 2 Related Work

## 2.1 Single Image Super-Resolution

Deep learning-based methods, such as SRCNN [6], which first applied convolutional neural networks (CNN) to image super-resolution tasks, are commonly used in this field. Subsequent methods [3, 4, 5, 6, 7, 8, 9, 10, 11] improved image reconstruction quality by optimizing convolutional modules and incorporating adversarial learning [20, 21]. Recent Transformer-based networks [9, 10, 11] have shown excellent results due to their powerful self-attention mechanisms.

Image super-resolution methods have been applied to specific image types, such as medical [22], hyperspectral [23], and scene images [24]. These methods often struggle with generalization due to their optimization for specific image types. Our method overcomes this limitation by using diferent semantic extraction modules for diferent image types, leading to stronger generalization performance.

## 2.2 Text Encoding

Text encoding in Natural Language Processing (NLP) transforms textual data into a numerical format for machine learning algorithms [25]. Key techniques include Bag-of-Words (BoW) Encoding [26], Word Embeddings (e.g., Word2Vec [27], GloVe [28], FastText [29]), and Transformers and Pretrained Language Models (e.g., BERT [30] and GPT [31]). We utilize BERT [30] for its superior performance in text encoding tasks.

## 2.3 Feature Fusion

Feature fusion integrates features from diferent modalities to improve cross-modal task performance. Techniques include early fusion [32], which combines features at a low level, late fusion [33], which combines features at a higher semantic level, and hybrid fusion [34], which integrates features at multiple stages.

## 2.4 Image Semantic Extraction

Our method utilizes a combination of Object Detection, Image Captioning, and Optical Character Recognition (OCR) to extract semantic information from images. For natural images, we use the YOLOv5 model [35] for object detection and BLIP [13] to extract deeper semantic information. For document images, PaddleOCR [36] extracts semantic information with high accuracy.

## 3 Methodology

## 3.1 Defining the Problem

The primary objective of Image Super-Resolution (SR) is to augment the resolution of a lowresolution (LR) image, thereby deriving a high-resolution (HR) image replete with intricate details. From a mathematical perspective, the super-resolution problem can be defined as follows:

Assume an input LR image $\mathbf { X } \in \mathbb { R } ^ { m \times n }$ , where m and n denote the spatial dimensions. The aim is to create an HR image $\hat { \mathbf Y } \in \mathbb R ^ { p \times q }$ , where p and q are the upsampled dimensions.

This relationship between the LR image and HR image can be formally expressed as:

$$
\mathbf { X } = { \mathcal { D } } ( \mathbf { Y } ) + \mathbf { N } ,
$$

where $\mathbf { Y } \in \mathbb { R } ^ { p \times q }$ signifies the original HR image, (Y) symbolizes the degradation operator which models the downsampling process, and N represents the additive noise. The objective of the image super-resolution algorithm is to ensure that Y<sup>ˆ</sup> closely approximates Y, thus achieving a high degree of similarity.

The quality of the reconstructed HR image can be assessed using various image quality metrics such as the peak signal-to-noise ratio (PSNR) or the structural similarity index (SSIM) [37].

However, due to the inherent ill-posed nature of the problem, a unique solution is elusive, thereby making the super-resolution problem a challenging one. To overcome this issue, a variety of approaches have been suggested, encompassing both traditional optimization-based methods and methods based on deep learning.

In this study, we put forth a super-resolution method that integrates semantic information extracted from images as cues during the model training phase. To ascertain its eficacy, we juxtapose our approach with cutting-edge methods on benchmark datasets. The reconstructed HR images are assessed using quantitative metrics. Experimental results reveal that by capitalizing on semantic prompts, our method surpasses the performance of existing algorithms.

## 3.2 Overview of the Method

Our proposed method for image super-resolution strives to utilize both pixel features and semantic features to achieve results that are not only more accurate but also visually appealing. The fundamental concept is to incorporate the extracted semantic information as cues during the model training phase, thereby enabling the model to be directed by the context and content of the images.

![](images/3a7e8373cf889f516887efe5b87881e985fe864aa7a45dca1baef964c451ec1d.jpg)  
Figure 2: The comprehensive framework of the proposed method. The snowflakes in the figure denote that the module is in a frozen state, while the flames indicate that the module is undergoing training.

The overall workflow of our method is illustrated in Figure 2. It comprises a series of components:

• Super-Resolution Backbone

• Semantic Extraction & Visual Extraction

• Cross-Modal Feature Fusion

• Prompt Traction Head

The Super-Resolution Backbone is crucial for preserving image details and textures during the super-resolution process. It utilizes advanced algorithms and deep learning to map lowresolution to high-resolution spaces. We adopt the ESRGAN generator from [5] as our backbone, a UNET-like architecture that is highly compatible with complex image data. ESRGAN is extensively employed in various computer vision tasks such as image segmentation, making it a dependable choice for super-resolution. Its forte lies in enhancing image details and high-frequency information, which are vital for improving image quality in super-resolution tasks.

The Semantic Extraction & Visual Extraction component is primarily responsible for extracting the semantic information and visual features of an image. In our method, we use a pre-trained VGG16 [38] to extract visual features of images for subsequent cross-modal feature fusion. For image semantic information extraction, we use BLIP [13] and YOLOv5 [35] for semantic extraction of natural images, and PaddleOCR [36] for semantic extraction of document images. The extracted semantic information consists of two parts: text description and region location coordinates. The combination of text description and location information forms a natural language representation of key regions in the image, which can be encoded and utilized for model training through language modeling.

The Cross-Modal Feature Fusion component is used to align and optimize visual and textual modal features, enabling the model to learn more profoundly from multimodal features. Moreover, our method is designed to efectively utilize semantic region position information, which includes encoding location in the form of coordinates and involving location features in training. By doing so, the model is able to consider diferent regions in the image separately, thus improving the performance. In terms of concrete implementation, we leverage the self-attention mechanism proposed by Transformer to design a multimodal attention module that considers position information for image super-resolution tasks.

The Prompt Traction Head component guides and optimizes the output of a super-resolution network by inputting the prompt feature obtained from multimodal feature fusion. Specifically, we use the prompt feature as a kind of exciters to increase or suppress the values of the image feature maps, allowing the details of the feature maps to be better displayed. The processed feature maps are then processed by the upsampling module to obtain the enlarged image.

## 3.3 Semantic Processing

This section delves into the generation and processing of semantic information in our proposed approach. We utilize a fully trained, domain-specific model to automatically extract semantic information from images. This extracted information is then transformed into textual data for feature extraction, deploying a well-established language model, as detailed in [31, 30]. Recognizing the importance of localized semantic content within an image, we also consider the spatial distribution of key elements.

It’s crucial to acknowledge that semantic information varies across diferent image types. In natural images, semantic information is typically the types of objects in the image and their spatia relationships. However, for document images, it refers to the textual content within the image. Hence, we have specifically defined the input and output interfaces for the semantic extraction module, thereby enhancing the flexibility of our approach. The module receives an image as input and outputs a textual representation of the image’s semantics and the coordinates of significant regions. We have developed specific semantic extraction modules for both natural and document images, which have been utilized in our subsequent experiments.

As depicted in Figure 3(a), for natural images, our process begins with YOLOv5 [35] to detect objects, determining their category and spatial information. The image is then processed by BLIP [13] to create a textual description, using the detected objects’ categories as prompts to guide text generation. Our object detection model excels at recognizing diverse object categories. Thus, when selecting prompt words, we prioritize categories based on the area they occupy in the image, opting for those with the greatest coverage. This strategy enhances the creative capability of the image captioning model.

For document images, as shown in Figure 3(b), we use the PaddleOCR model [36] to identify text content and its corresponding spatial coordinates. As document images primarily contain textual elements arranged in coherent paragraphs, we argue that these paragraphs inherently carry the semantic essence of such images. Based on this premise, we directly apply an OCR model to extract both the textual content and the precise positioning of paragraph regions. This methodology ensures consistency in our semantic extraction across various image types.

![](images/25785ab3b23aa8dcb057d1a10c2f055b52d851a48fb6dbfb935bd0beadf08122.jpg)

(a) For natural images, we employ YOLOv5 to identify the category and position of objects in the image, and utilize BLIP to generate a textual description of the image based on the detected objects’ names.  
![](images/24af1c0ec9d1fe5f3ef27a48057533718f4017bcde51b1dcd4d9ed538b590a9d.jpg)  
(b) For document images, we directly apply PaddleOCR to extract the positions and text content of paragraphs in the image.  
Figure 3: Semantic Information Processing for Diferent Types of Images.

## 3.4 Cross-Modal Feature Fusion

The Cross-Modal Feature Fusion module is designed to integrate features from diferent sources, namely visual and semantic domains. The inputs to this module are detailed as follows:

• Visual Features: These are the features extracted from the image using a convolutional neural network (CNN). The visual features are high-dimensional representations that capture the visual content of the image, including textures, edges, and color information. The dimensionality and the level of abstraction of these features depend on the depth of the CNN and the layer from which they are extracted.

• Positional Masks: Positional masks are binary or continuous maps that provide spatial information about regions of interest within the image. These masks are used to highlight or suppress specific areas during the feature fusion process, allowing the model to focus on salient parts of the image that are more relevant to the task at hand.

• Semantic Features: Semantic features are derived from the textual information associated with the image. These features are obtained using a language model such as BERT, which encodes the textual semantics into a high-dimensional vector space. The semantic features capture the contextual information and the relationships between diferent elements described in the text. Positional information is also encoded and added to the semantic features to ensure that the spatial relationships within the image are preserved during the fusion process.

Once these inputs are prepared, the Cross-Modal Feature Fusion module proceeds to perform a series of operations to merge the visual and semantic features efectively.

## 3.4.1 Semantic Feature Preparation

In image semantic extraction, the primary focus often revolves around analyzing specific localized regions. However, for model training eficiency, it is vital to integrate positional information into the semantic features. To this end, we have developed an innovative position encoding module that efectively incorporates positional context into semantic features. Additionally, in Equation 10, we have utilized cosine similarity as a crucial part of the loss function. This strategic decision enables the model to skillfully uncover the relationship between visual and semantic cues while concurrently maintaining the coherence of positional insights across diverse modalities.

The comprehensive workflow is detailed in Figure 4(a). The textual semantics extracted from the image are encoded using BERT [30], a state-of-the-art language model. Simultaneously, we use an eficient matrix embedding layer to encode positional data, adeptly converting coordinates into high-dimensional vectors. Unlike traditional positional encoding methods, such as those used in Vision Transformer (VIT) [39], our approach represents information in the form of two-dimensional coordinates. When processing visual features, the first step involves generating a mask derived from positional data. Then, the positional features are smoothly fused with their corresponding visual counterparts based on the underlying mask. The generation of local visual features within specific regions is achieved using the Masked Global Average Pooling method [40].

Upon completion of semantic feature generation as depicted in Figure 4(a), the output of the image semantic extraction module is a multi-dimensional vector. This vector encapsulates the semantic information derived from the image, which has been processed through the BERT language model. The dimensions of the output vector are determined by the architecture of BERT and the specific implementation details, typically resulting in a fixed-size embedding that represents the semantic content. For instance, if BERT-Base is used, the output dimension would be 768, corresponding to the hidden size of the BERT-Base model. This semantic feature vector is then ready to be fused with the visual features, which have been similarly encoded into a high-dimensional space reflective of the positional information and visual content of the image.

It is important to note that the dimensionality of the semantic features must be compatible with the visual features for efective fusion. If there is a discrepancy in dimensions, a transformation operation, such as a fully connected (FC) layer, may be employed to align the semantic features with the visual features. This ensures that the subsequent position mask attention module can efectively perform its task, leveraging both semantic and visual cues to enhance the model’s understanding of the image content.

## 3.4.2 Position Mask Attention

Traditional multimodal feature fusion primarily relies on Transformer-based processing, as demonstrated by previous works like Transformer [41]. Given that feature extractors for various modalities all adopt the Transformer architecture, a straightforward approach involves summing these features and further processing them to facilitate multimodal training. However, when dealing with convolutional neural networks, the fusion of distinct modal features becomes a more complex challenge [42]. Merely adjusting dimensions and performing summations often fail to achieve the desired synergistic efect. Therefore, we incorporate an attention mechanism [43] to unite diverse modal features while simultaneously distinguishing between critical and non-critical regions within the image.

![](images/9e1828c87b829ecdc579ffadff8177573275a23d51fa84f5ee3db9f127b45f5c.jpg)

(a) Semantic feature generation. GAP operations are performed on visual features based on the coordinates of the content area. Positional information is encoded and added to multimodal features. To maintain semantic consistency between visual and textual features, a cosine loss function is used for constraints.  
![](images/6d74ead95096d5862c486b3b9cced0c3a1a8c8b09316da9ee4815d3d9090c9e9.jpg)  
(b) Cross-modal feature fusion. By integrating the position information of the content area, attention operations are performed on features of diferent modalities for feature fusion. Content and non-content regions are added after parallel operations.  
Figure 4: Structure of Position Mask Attention.

As shown in Figure 4(b), this module takes visual features, position masks, and semantic features as input and proceeds to outline discrete regions within the visual features with the help of masks.

$$
F _ { m a s k } = F _ { v i s u a l } \times X _ { m a s k }\tag{1}
$$

$$
F _ { m a s k } ^ { - 1 } = F _ { v i s u a l } \times X _ { m a s k } ^ { - 1 }\tag{2}
$$

Given the size disparity within the visual features, we first initiate a 2 downsampling of both $F _ { m a s k }$ and $F _ { m a s k } ^ { - 1 }$ , followed by a reflective transformation applied to all features. In the subsequent equation, the variables $\theta , \delta ,$ and λ are employed as learnable parameters.

$$
K _ { \alpha } , V _ { \alpha } = F C ( \theta , F _ { m a s k } )\tag{3}
$$

$$
K _ { \beta } , V _ { \beta } = F C ( \delta , F _ { m a s k } ^ { - 1 } )\tag{4}
$$

$$
Q = F C ( \lambda , F _ { s e m a n t i c } )\tag{5}
$$

Next, we use the semantic features as queries and perform attention operations on the key and non-key regions, distinctly facilitating the fusion of diferent modal features.

$$
F = \sum _ { i } ^ { \alpha , \beta } U p s a m p l e ( \omega _ { i } ( Q ^ { * } \otimes K _ { i } ^ { T } ) \times V _ { i } )\tag{6}
$$

$$
Q ^ { * } = \left\{ \begin{array} { l l } { { \ Q } } & { { i = \alpha } } \\ { { - 1 \times Q } } & { { i = \beta } } \end{array} \right.\tag{7}
$$

To conclude the process, the fused features undergo dimensional transformation and upscaling, resulting in the summation of features from diferent regions. It is worth emphasizing that we designate Q as the semantic features related to critical regions. In our efort to make the features in non-critical regions distinctly separable from those in critical regions, we use a negated version of Q (-Q) for the attention operation within non-critical regions.

## 3.5 Prompt Traction Head

The Prompt Traction Head is tasked with directing and optimizing the super-resolution network’s output by utilizing prompt features derived from multimodal feature fusion. This section delves into the specific operations within the Prompt Traction Head, including the enhancement of image feature details using prompts and the subsequent process of image upscaling.

In particular, prompt features serve as exciters that modulate the values within image feature maps, thereby bolstering the depiction of feature details. These feature maps are then subjected to additional processing via an upsampling module, resulting in the final upscaled image.

The operations of the Prompt Traction Head can be expressed through the following equations:

$$
F _ { \mathrm { e x c i t e d } } = C o n v ( F _ { \mathrm { p r o m p t s } } ) + F _ { \mathrm { s r } } \cdot \delta ( F _ { \mathrm { p r o m p t s } } )\tag{8}
$$

$$
F _ { \mathrm { u p s c a l e d } } = \mathrm { U p s a m p l e } ( F _ { \mathrm { e x c i t e d } } )\tag{9}
$$

Here, δ denotes the tanh function and Conv symbolizes the point convolution operation. $F _ { \mathrm { s r } }$ symbolizes the feature map produced by the super-resolution backbone, while $F _ { \mathrm { p r o m p t s } }$ signifies the prompt feature post cross-modal fusion. The incorporation of prompts within this module enables a more nuanced manipulation of the feature maps, ensuring that the image’s fine-grained details are efectively preserved and amplified during the super-resolution process. This method yields visually appealing and precise super-resolved images.

![](images/fdfa1e885cca8f1a539ade18cead024d3355049a222a4c13a76bb73cf267f605.jpg)  
Figure 5: The structural diagram of the Prompt Traction Head, where δ denotes the tanh function and Conv symbolizes the point convolution operation.

As depicted in Figure $5 ,$ our design is characterized by its simplicity and eficacy. Initially, a point convolution is applied to map $F _ { \mathrm { p r o m p t s } }$ once, followed by an element-wise addition to merge it with $F _ { \mathrm { s r } }$ . This operation aims to integrate the image’s semantic information with the original features, thereby augmenting the expressive capacity of these features. Simultaneously, $F _ { \mathrm { p r o m p t s } }$ undergoes a tanh transformation to generate parameter values for modulating the excitation intensity of $F _ { \mathrm { s r } }$ This modulation process, which involves element-wise multiplication, enables the amplification or suppression of the integrated semantic information’s features. Ultimately, a straightforward deconvolution operation is employed to carry out the upsampling, culminating in the creation of an upscaled image.

## 3.6 Metric and Loss Function

In our methodology, we deploy a carefully curated set of metrics and loss functions, which serve as the guiding light for the super-resolution process. We incorporate traditional image quality metrics, such as PSNR and SSIM [37], to provide a quantitative assessment of the fidelity of the super-resolved images.

$$
{ \mathcal { L } } ( x , y , F _ { v i s } , F _ { s e m } , F _ { s r } ) = L 1 ( x , y ) + \alpha \cdot { \mathcal { S } } ( F _ { s e m } , F _ { v i s } ) + \beta \cdot { \mathcal { V } } ( F _ { s r } , F _ { v i s } )\tag{10}
$$

$$
L 1 ( x , y ) = \| x - y \| _ { 1 }\tag{11}
$$

$$
S ( F _ { s e m } , F _ { v i s } ) = c o s i n e ( F _ { s e m } - G A P ( F _ { v i s } ) )\tag{12}
$$

$$
\mathcal { V } ( F _ { s r } , F _ { v i s } ) = \sqrt { ( F _ { s r } - M a x P o o l _ { \times s c a l e } ( F _ { v i s } ) ) ^ { 2 } }\tag{13}
$$

We have exercised considerable care in the design of our loss function. As depicted in the formula above, our loss function integrates three critical components: pixel loss, semantic similarity loss, and visual feature similarity loss. In this equation, α and $\beta$ denote weight coeficients, $F _ { v i s }$ corresponds to the visual features of the generated image, $F _ { s e m }$ signifies the semantic features of the generated image, and $F _ { s r }$ represents the features derived from super-resolution backbone downsampling. The pixel loss employs an L1 loss [44] function to optimize the distance between the generated image and the real image. Semantic similarity loss is harnessed to ensure the consistency between image features and semantic features. Visual feature similarity loss is utilized to maintain the expressive resemblance between visual feature extractors and the super-resolution backbone at the feature level. It is noteworthy that, due to the inconsistency in the size of the feature maps generated by the two networks, we employ a MaxPool operation to standardize the size, thereby facilitating the execution of calculations.

## 3.7 Discussion on Computational Costs

Our semantic processing method is efective for image understanding but computationally demanding. The YOLOv5 model ofers a good speed-accuracy tradeof for object detection but is resourceintensive, especially with high-resolution images on limited-GPU systems.

The BLIP model for textual descriptions also requires significant GPU resources due to its advanced vision-language architecture. It can be taxing when producing detailed captions.

Extracting text and coordinates from document images with PaddleOCR can become costly for high-resolution, text-dense images, leading to increased processing times and memory demands.

Additionally, BERT’s deep bidirectional architecture for text feature extraction further adds to the computational load. Its thorough language analysis is resource-heavy, potentially slowing down real-time applications or large dataset processing.

## 4 Experimental Procedures

## 4.1 Configuration of the Experiment

Datasets. Our experiment utilizes two datasets: the DF2K dataset (DIV2K [45] combined with Flickr2K [46]), and the Baidu Document Image dataset (BDDI) [47]. These datasets serve as the training sets, enabling us to examine the eficacy of our method on both natural and document images. For our super-resolution network, we employ the generator component of ESRGAN, incorporating a UNET-inspired network architecture that has proven to be efective in handling complex image data. To provide a comparative analysis, we adjusted the number of network layers and channels in ESRGAN [5], PAN [4], REALSR [3], SWINIR [10], and HAT [11] to maintain a similar parameter count across all models. It is important to note that none of these models were pre-trained using any other datasets.

Implementation Details. We configured the language model to have a maximum text input length of 128 and a text feature dimensionality of 768. The dimensionality remains the same after position information has been encoded. In the cross-modal attention mechanism, we downsample the feature map and map it through the MLP layer, specifying a feature dimension of 256 after mapping. We discovered through experimentation that optimal model performance can be achieved by adjusting the mask magnitude in the content region masking operation, which we set to 0.3. To investigate the model’s performance at diferent image scales, we conducted image super-resolution experiments at 2 , 3 , and 4 . The performance of the model was evaluated using PSNR and SSIM metrics [37]. All computations in the experiments were performed using two NVIDIA V100 GPUs.

## 4.2 Enhanced Performance Comparison with State-of-the-Art Methods

## 4.2.1 Evaluation on Natural Image Datasets

Table 1: Comparative Results on Natural Image Datasets. Our method surpasses other techniques in numerous benchmarks, particularly leveraging ESRGAN as the backbone while achieving significant improvements over it.
<table><tr><td>Method</td><td>Scale</td><td>Set5</td><td></td><td>Set14</td><td>BSD100</td><td>Urban100</td><td></td><td>Manga109</td></tr><tr><td>ESRGAN</td><td>×2</td><td>37.96</td><td>0.9593 33.87</td><td>0.9135</td><td>31.62 0.8902</td><td>32.27</td><td>0.9307</td><td>39.56 0.9648</td></tr><tr><td>PAN</td><td>×2</td><td>38.34</td><td>0.9625</td><td>34.36 0.9221</td><td>32.00 / 0.9001</td><td>33.42</td><td>0.9402</td><td>39.84 0.9726</td></tr><tr><td>RealSR</td><td>×2</td><td>38.19</td><td>0.9622</td><td>34.38 0.9223</td><td>31.98 /0.8967</td><td>33.89</td><td>0.9395</td><td>39.79 0.9721</td></tr><tr><td>SwinIR</td><td>×2</td><td>38.37</td><td>0.9617</td><td>34.42 0.9228</td><td>32.02 /0.9000</td><td>33.26</td><td>/0.9387</td><td>39.76 /0.9710</td></tr><tr><td>HAT</td><td>×2</td><td>38.59</td><td>0.9629</td><td>34.47 0.9223</td><td>32.12 0.9009</td><td>33.44</td><td>0.9418</td><td>40.02 0.9729</td></tr><tr><td>Ours</td><td>×2</td><td>38.62</td><td>0.9636 34.55</td><td>0.9249</td><td>32.10 0.9024</td><td>33.60</td><td>0.9425</td><td>40.14 0.9730</td></tr><tr><td>ESRGAN</td><td>×3</td><td>35.24</td><td>0.9347</td><td>30.76 0.8557</td><td>29.44 0.8607</td><td>29.84</td><td>0.8803</td><td>36.59 0.9471</td></tr><tr><td>PAN</td><td>×3</td><td>35.41</td><td>0.9351</td><td>30.80 /0.8571</td><td>29.44 /0.8610</td><td>29.95</td><td>/0.8816</td><td>36.67 /0.9488</td></tr><tr><td>RealSR</td><td>×3</td><td>35.39</td><td>0.9360 30.79</td><td>0.8560</td><td>29.49 0.8606</td><td>29.89</td><td>/0.8805 36.70</td><td>/0.9491</td></tr><tr><td>SwinIR</td><td>×3</td><td>36.11</td><td>0.9445 31.52</td><td>0.8671</td><td>29.66 /0.8611</td><td>30.49</td><td>/0.8837</td><td>36.90 /0.9531</td></tr><tr><td>HAT</td><td>×3</td><td>36.20</td><td>0.9444 31.57</td><td>0.8669</td><td>29.71</td><td>0.8613 30.55</td><td>0.8844</td><td>36.98 0.9529</td></tr><tr><td>Ours</td><td>×3</td><td>36.15</td><td>0.9441 31.63</td><td>0.8674</td><td>29.71 0.8615</td><td>30.57</td><td>0.8849 37.05</td><td>0.9537</td></tr><tr><td>ESRGAN</td><td>×4</td><td>32.73</td><td>0.8915 27.72</td><td>0.7794</td><td>27.64 0.7406</td><td>26.48</td><td>0.8158</td><td>31.62 0.9146</td></tr><tr><td>PAN</td><td>×4</td><td>32.58</td><td>0.8881</td><td>27.73 0.7770</td><td>27.66 0.7498</td><td>26.51</td><td>0.8172</td><td>31.77 0.9173</td></tr><tr><td>RealSR</td><td>×4</td><td>32.59</td><td>0.8879</td><td>27.86 0.7782</td><td>27.66</td><td>/0.7462 26.49</td><td>/0.8161</td><td>31.79 0.9169</td></tr><tr><td>SwinIR</td><td>×4</td><td>32.66</td><td>0.8889</td><td>28.59 0.7850</td><td>27.69</td><td>0.7422 27.50</td><td>0.8260</td><td>31.85 0.9184</td></tr><tr><td>HAT</td><td>×4</td><td>32.77</td><td>0.8912</td><td>28.80 0.7805</td><td>27.71</td><td>0.7475 27.63</td><td>0.8227</td><td>32.07 0.9222</td></tr><tr><td>Ours</td><td>×4</td><td>32.84</td><td>0.8947</td><td>28.87 0.7830</td><td>27.76</td><td>0.7481 27.43</td><td>0.8269</td><td>32.13 0.9234</td></tr></table>

Our model was meticulously trained on the expansive DF2K dataset and subsequently subjected to comprehensive evaluations on several benchmark datasets, including Set5 [48], Set14 [49], BSD100 [50], Urban100 [51], and Manga109 [52]. The results, detailed in Table 1, clearly attest to our model’s preeminence over a spectrum of state-of-the-art methods. Our approach is particularly noteworthy for its exceptional performance relative to ESRGAN, highlighting the eficacy of integrating multimodal data to bolster feature extraction capabilities. This enhancement allows for more nuanced texture details to be discerned from the images, as evidenced by the results.

In Figure 6, we present a detailed visual comparison that showcases our model’s adeptness at image detail restoration. The visualizations vividly convey the model’s advanced ability to reconstruct color fidelity and intricate textures, outstripping its peers. This superior performance can be attributed to the model’s augmented representational capacity, derived from the integration of content-aware attention mechanisms. These mechanisms enable the model to focus selectively on the most pertinent features within the image, thereby preserving vital details with exceptional accuracy.

Furthermore, our analysis indicates that our model maintains its high performance even when tasked with processing images of small dimensions. This robustness is likely due to the model’s capacity to leverage semantic information within the image, prompting it to generate and reconstruct richer content. The result is a more authentic and detailed representation of the image’s nuances.

## 4.2.2 Enhanced Performance on Document Image Super-Resolution

A rigorous set of experiments was performed using the BDDI dataset [47], and the results are systematically detailed in Table 2. The data clearly demonstrates that our proposed model achieves

![](images/b0a80da3d69000c14e00c96fab0a8a53cc486cc9c4e7fe8e31580aa46db9c1fc.jpg)  
Figure 6: Visualization of enhanced image details for select test images, accompanied by captions generated by the semantic extraction module. The displayed PSNR values quantify the image quality.

Table 2: Comparative Performance on Document Image Super-Resolution
<table><tr><td>Method</td><td>Scale</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>ESRGAN</td><td>×2</td><td>25.53</td><td>0.9441</td></tr><tr><td>PAN</td><td>×2</td><td>25.97</td><td>0.9489</td></tr><tr><td>RealSR SwinIR</td><td>×2</td><td>26.64</td><td>0.9563</td></tr><tr><td>HAT</td><td>×2</td><td>29.13</td><td>0.9691</td></tr><tr><td>Ours</td><td>×2 ×2</td><td>29.89 31.05</td><td>0.9743 0.9773</td></tr><tr><td>ESRGAN</td><td>×3</td><td>23.97</td><td>0.9382</td></tr><tr><td>PAN</td><td>×3</td><td>24.92</td><td>0.9411</td></tr><tr><td>RealSR</td><td>×3</td><td>25.30</td><td>0.9528</td></tr><tr><td>SwinIR</td><td>×3</td><td>27.61</td><td>0.9603</td></tr><tr><td>HAT</td><td>×3</td><td>27.92</td><td>0.9627</td></tr><tr><td>Ours</td><td>×3</td><td>28.79</td><td>0.9687</td></tr><tr><td>ESRGAN</td><td>×4</td><td>23.11</td><td>0.9361</td></tr><tr><td>PAN</td><td>×4</td><td>23.61</td><td>0.9437</td></tr><tr><td>RealSR</td><td>×4</td><td>23.59</td><td>0.9432</td></tr><tr><td>SwinIR</td><td>×4</td><td>24.90</td><td>0.9427</td></tr><tr><td>HAT</td><td>×4</td><td>25.08</td><td>0.9469</td></tr><tr><td>Ours</td><td>×4</td><td>25.41</td><td>0.9503</td></tr></table>

superior performance metrics, notably in Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM), surpassing other contemporary models.

Additionally, Figure 7 illustrates the enhanced edge recovery of intricate text details achieved by our method. An in-depth investigation into the Locally Adaptive Modulation (LAM) characteristic [53] reveals that our model not only enhances text region activation but also adeptly manages pixel activation in adjacent areas, thus sharpening the focus on salient content within the images.

It is noteworthy that the dataset contains instances of text blurring due to small image dimensions, which present challenges in model training. Nevertheless, our findings indicate that the application of a multimodal training approach yields images with significantly improved clarity and sharpness in text regions, as opposed to training without it. This enhancement is likely attributable to the model’s deeper comprehension of image content, facilitated by leveraging pre-trained models from the NLP domain within the multimodal framework. Consequently, our approach is capable of producing superior results even when the original images are blurred.

## 4.3 Ablation Studies

## 4.3.1 Efectiveness of Each Component

To thoroughly examine the contribution of individual components to our methodology’s performance, we executed a series of ablation studies, the outcomes of which are presented in Table 3.

![](images/877159e619efb4fb3a50f0822d357ac3d3748eef48c41ec48f0ba9936b4fccd1.jpg)  
Figure 7: Visualization and analysis of the Locally Adaptive Modulation (LAM) efect on text detail enhancement in the BDDI dataset. Our approach significantly improves pixel activation in text-heavy areas.

These investigations scrutinized three pivotal aspects of our framework, with performance evaluations conducted on both the BSD100 and BDDI datasets.

The exclusion of the position encoding module ( PE) led to a discernible decrease in performance metrics across both datasets, with a more marked impact on document images. This observation suggests that the positional cues provided by the encoding module are especially critical for accurately resolving smaller text regions within images. The replacement of the content mask attention ( CMA) with a straightforward feature summation also resulted in diminished performance metrics, reinforcing the eficacy of our content-focused multimodal feature fusion strategy in enhancing image super-resolution tasks. The removal of the semantic extraction module ( MT), which efectively reduced our approach to a standard image super-resolution model, caused a substantial drop in performance metrics. This highlights the value of leveraging techniques from diverse domains through multimodal training to significantly improve outcomes, even for low-level vision tasks. Such insights may have far-reaching implications for a variety of applications, including image denoising, as noted by recent studies [54].

The findings from these ablation studies provide an in-depth understanding of the importance of each component in our approach and emphasize the advantages of adopting a multimodal strategy to advance the capabilities of various computer vision tasks.

## 4.3.2 Exploring the Efects of Content Area Mask Magnitude on Model Performance

This section delves into the efect of varying the magnitude of content area masks on the eficacy of our model. The content area mask is a pivotal element in our experiments and is quantified by distinct magnitudes. Table 4 encapsulates the outcomes of our comparative analysis, wherein we experimented with magnitudes ranging from 0.0 to 1.0, while maintaining a consistent upscaling factor of 2. The model’s performance was appraised using two separate datasets.

Table 3: Ablation study results. PE denotes the exclusion of the position encoding module, CMA represents the substitution of the content mask attention, and MT signifies the absence of multimodal training.
<table><tr><td>Method</td><td>Scale BSD100</td><td>BDDI</td></tr><tr><td>Baseline ∅PE</td><td>×2 32.10 0.9024 ×2 31.91</td><td>31.05 0.9773 28.71 0.9682 0.9679</td></tr><tr><td>∅CMA ∅MT</td><td>×2 31.69 ×2</td><td>0.8982 0.8936 28.03 31.62 0.8902 25.53</td></tr><tr><td>Baseline</td><td>×3 29.71</td><td>0.9441 0.8615 28.79 0.9687</td></tr><tr><td>∅PE</td><td>×3 29.58 0.8610</td><td>27.16 0.9604</td></tr><tr><td>∅CMA</td><td>×3 29.63</td><td>0.8613 26.67 0.9553</td></tr><tr><td>∅MT</td><td>×3 29.44</td><td>0.8607 23.97 0.9382</td></tr><tr><td>Baseline</td><td></td><td></td></tr><tr><td>∅PE</td><td>×4 27.76</td><td>0.7481 25.41 0.9503</td></tr><tr><td></td><td>×4 27.71</td><td>0.7467 25.13 0.9479</td></tr><tr><td>∅CMA</td><td>×4 27.66</td><td>0.7459 24.89 0.9446</td></tr><tr><td>∅MT</td><td>×4 27.64 / 0.7406</td><td>23.11 0.9361</td></tr></table>

Table 4: Comparative analysis of model performance with varying content area mask magnitudes. The data indicates that a moderate increase in magnitude can enhance model eficacy.
<table><tr><td>Magnitude</td><td>Upscaling Factor</td><td>BSD100</td><td>BDDI</td></tr><tr><td>0.0</td><td>×2</td><td>31.94 / 0.9003</td><td>30.62 0.9711</td></tr><tr><td>0.3</td><td>×2</td><td>32.10 /0.9024</td><td>31.05 0.9773</td></tr><tr><td>0.5</td><td>×2</td><td>31.54 / 0.8895</td><td>30.22 /0.9696</td></tr><tr><td>0.7</td><td>×2</td><td>31.88 /0.8936</td><td>29.32 0.9603</td></tr><tr><td>1.0</td><td>×2</td><td>31.47 0.8872</td><td>26.33 0.9481</td></tr></table>

A key observation is that a magnitude of 0.3 yields the most substantial enhancement in model performance, as evidenced by the superior scores in image quality and associated metrics. This suggests that an optimal, moderate increment in the magnitude of the content area mask, exemplified by a value of 0.3, can significantly bolster the capabilities of our model. The results underscore the importance of fine-tuning the magnitude of the content area mask to refine and augment the performance of our model. Additional experimentation with this variable may yield further insights, potentially contributing to advancements in image processing and restoration methodologies.

## 4.3.3 Influence of Loss Weighting on Multimodal Performance

Our investigation extends to the influence of loss weights on our model’s multimodal performance. Loss weights represent the relative significance assigned to the diferent components of loss during the training of our model. We specifically examine the impact of varying the semantic similarity loss weight, while maintaining the pixel loss and visual feature similarity loss weights at empirical values of 1.0 and 0.0001, respectively.

The summary of outcomes from our comparative experiments is presented in Table 5. We experimented with loss weight values ranging from 0.0 to 0.0001, while the upscaling factor remained constant at 2. The model’s performance was evaluated on two distinct datasets, BSD100 and

Table 5: Comparative experiment outcomes based on varying loss weights. "Weight" refers to the semantic similarity loss weight, with pixel loss and visual feature similarity loss weights fixed at 1.0 and 0.0001 respectively.
<table><tr><td>Weight</td><td>Scale</td><td>BSD100</td><td>BDDI</td></tr><tr><td>0.0</td><td>×2</td><td>31.25 0.8879</td><td>25.73 /0.9467</td></tr><tr><td>1.0</td><td>×2</td><td>26.13 0.8122</td><td>21.39 0.8951</td></tr><tr><td>0.1</td><td>×2</td><td>29.37 0.8890</td><td>24.98 0.9381</td></tr><tr><td>0.01</td><td>×2</td><td>31.68 0.8883</td><td>29.11 0.9695</td></tr><tr><td>0.001</td><td>×2</td><td>32.10 0.9024</td><td>31.05 0.9773</td></tr><tr><td>0.0001</td><td>×2</td><td>31.89 0.8991</td><td>30.04 0.9711</td></tr></table>

BDDI, focusing on the quality of scale enhancement. The empirical data indicates that the choice of semantic similarity loss weight markedly influences the model’s eficacy. A decline in the weight of this loss component correlates with enhanced upscaling results, with an optimal weight of 0.001 yielding the most favorable performance. This underscores the importance of a calibrated approach to loss component weighting in our model to achieve optimal results. It should be emphasized that the ideal loss weight configuration may vary across diferent datasets and tasks. Therefore, finetuning the loss weights is an essential step towards enhancing performance across various modalities.

## 5 Conclusion

This study presents a novel approach to image super-resolution that efectively capitalizes on the semantic content within images to augment the super-resolution process. We have developed a dedicated semantic extraction and processing module, complemented by a sophisticated cross-modal feature fusion module, which is adeptly customized for various image categories. The integration of positional encoding allows for the establishment of precise, localized connections between the extracted semantic elements and their corresponding regions in the image. Our comprehensive experimental analysis validates the superiority of our method, as evidenced by significant improvements in performance metrics and the quality of the visual results. This research makes a substantial contribution to the domain of image super-resolution by introducing an innovative technique that exploits the intrinsic semantics of images to refine model training and enhance the precision of image detail recovery.

## References

[1] S. Anwar, S. Khan, N. Barnes, A deep journey into super-resolution: A survey, ACM Computing Surveys 53 (3) (2020) 1–34. doi:10.1145/3390462.

[2] A. M. Raid, W. M. Khedr, M. A. El-dosuky, W. Ahmed, JPEG image compression using discrete cosine transform – a survey, arXiv preprint arXiv:1405.6147 (2014).

[3] X. Ji, Y. Cao, Y. Tai, C. Wang, J. Li, F. Huang, Real-world super-resolution via kernel estimation and noise injection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020, pp. 466–467.

[4] H. Zhao, X. Kong, J. He, Y. Qiao, C. Dong, Eficient image super-resolution using pixel attention, in: Computer Vision – ECCV 2020 Workshops, Vol. 12536 of Lecture Notes in Computer Science, Springer, 2020, pp. 56–72. doi:10.1007/978-3-030-67070-2\_3.

[5] X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, C. C. Loy, ESRGAN: Enhanced superresolution generative adversarial networks, in: Computer Vision – ECCV 2018 Workshops, Vol. 11133 of Lecture Notes in Computer Science, Springer, 2018, pp. 63–79. doi:10.1007/ 978-3-030-11021-5\_5.

[6] C. Dong, C. C. Loy, K. He, X. Tang, Image super-resolution using deep convolutional networks, IEEE Transactions on Pattern Analysis and Machine Intelligence 38 (2) (2016) 295–307. doi: 10.1109/TPAMI.2015.2439281.

[7] T. Dai, J. Cai, Y. Zhang, S.-T. Xia, L. Zhang, Second-order attention network for single image super-resolution, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11065–11074. doi:10.1109/CVPR.2019.01132.

[8] B. Niu, W. Wen, W. Ren, X. Zhang, L. Yang, S. Wang, K. Zhang, X. Cao, H. Shen, Single image super-resolution via a holistic attention network, in: Computer Vision – ECCV 2020, Vol. 12357 of Lecture Notes in Computer Science, Springer, 2020, pp. 191–207.

[9] W. Li, X. Lu, S. Qian, J. Lu, X. Zhang, J. Jia, On eficient transformer-based image pre-training for low-level vision, in: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI), 2023, pp. 1089–1097. doi:10.24963/ijcai.2023/121.

[10] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, R. Timofte, SwinIR: Image restoration using swin transformer, in: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021, pp. 1833–1844. doi:10.1109/ICCVW54120.2021.00210.

[11] X. Chen, X. Wang, J. Zhou, Y. Qiao, Activating more pixels in image super-resolution transformer, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 22367–22377. doi:10.1109/CVPR52729.2023.02142.

[12] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You only look once: Unified, real-time object detection, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779–788. doi:10.1109/CVPR.2016.91.

[13] J. Li, D. Li, C. Xiong, S. Hoi, BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation, in: Proceedings of the 39th International Conference on Machine Learning (ICML), Vol. 162 of Proceedings of Machine Learning Research, PMLR, 2022, pp. 12888–12900.

[14] A. M. Ali, B. Benjdira, A. Koubaa, W. El-Shafai, Z. Khan, W. Boulila, Vision transformers in image restoration: A survey, Sensors 23 (5) (2023) 2385. doi:10.3390/s23052385.

[15] A. de Santana Correia, E. L. Colombini, Attention, please! a survey of neural attention models in deep learning, Artificial Intelligence Review 55 (8) (2022) 6037–6124. doi:10.1007/ s10462-022-10148-x.

[16] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, P. Dollár, Microsoft COCO: Common objects in context, in: Computer Vision – ECCV 2014, Vol. 8693 of Lecture Notes in Computer Science, Springer, 2014, pp. 740–755. doi:10.1007/978-3-319-10602-1\_48.

[17] M. Stefanini, M. Cornia, L. Baraldi, S. Cascianelli, G. Fiameni, R. Cucchiara, From show to tell: A survey on deep learning-based image captioning, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (1) (2023) 539–559. doi:10.1109/TPAMI.2022.3148210.

[18] Z. Zou, K. Chen, Z. Shi, Y. Guo, J. Ye, Object detection in 20 years: A survey, Proceedings of the IEEE 111 (3) (2023) 257–276. doi:10.1109/JPROC.2023.3238524.

[19] A. Gupta, A. Goyal, P. Gupta, S. Diwakar, A survey of modern optical character recognition techniques, arXiv preprint arXiv:1412.4183 (2014).

[20] C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, W. Shi, Photo-realistic single image super-resolution using a generative adversarial network, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4681–4690. doi:10.1109/CVPR.2017.19.

[21] W. Zhang, Y. Liu, C. Dong, Y. Qiao, RankSRGAN: Generative adversarial networks with ranker for image super-resolution, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 3096–3105. doi:10.1109/ICCV.2019.00319.

[22] Y. Iwamoto, K. Takeda, Y. Li, A. Shiino, Y.-W. Chen, Unsupervised MRI super-resolution using deep external learning and guided residual dense network with multimodal image priors, IEEE Transactions on Emerging Topics in Computational Intelligence 7 (2) (2023) 426–435. doi:10.1109/TETCI.2022.3215137.

[23] T. Liu, Y. Liu, C. Zhang, Y. Liyin, X. Sui, Q. Chen, Hyperspectral image super-resolution via dual-domain network based on hybrid convolution, arXiv preprint arXiv:2304.04589 (2023).

[24] J. Ma, Z. Liang, L. Zhang, A text attention network for spatial deformation robust scene text image super-resolution, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5901–5910. doi:10.1109/CVPR52688.2022.00582.

[25] Y. Goldberg, A primer on neural network models for natural language processing, Journal of Artificial Intelligence Research 57 (2016) 345–420. doi:10.1613/jair.4992.

[26] G. Salton, C. Buckley, Term-weighting approaches in automatic text retrieval, Information Processing & Management 24 (5) (1988) 513–523. doi:10.1016/0306-4573(88)90021-0.

[27] T. Mikolov, K. Chen, G. Corrado, J. Dean, Eficient estimation of word representations in vector space, arXiv preprint arXiv:1301.3781 (2013).

[28] J. Pennington, R. Socher, C. Manning, GloVe: Global vectors for word representation, in: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1532–1543. doi:10.3115/v1/D14-1162.

[29] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of tricks for eficient text classification, arXiv preprint arXiv:1607.01759 (2016).

[30] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2019, pp. 4171–4186. doi:10.18653/v1/N19-1423.

[31] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, Improving language understanding by generative pre-training, Tech. rep., OpenAI (2018).

[32] C. G. M. Snoek, M. Worring, A. W. M. Smeulders, Early versus late fusion in semantic video analysis, in: Proceedings of the 13th Annual ACM International Conference on Multimedia, 2005, pp. 399–402. doi:10.1145/1101149.1101236.

[33] P. K. Atrey, M. A. Hossain, A. El Saddik, M. S. Kankanhalli, Multimodal fusion for multimedia analysis: A survey, Multimedia Systems 16 (6) (2010) 345–379. doi:10.1007/ s00530-010-0182-0.

[34] S. Poria, E. Cambria, R. Bajpai, A. Hussain, A review of afective computing: From unimodal analysis to multimodal fusion, Information Fusion 37 (2017) 98–125. doi:10.1016/j.inffus. 2017.02.003.

[35] G. Jocher, YOLOv5 by Ultralytics, https://github.com/ultralytics/yolov5 (2021).

[36] Y. Du, C. Li, R. Guo, X. Yin, W. Liu, J. Zhou, Y. Bai, Z. Yu, Y. Yang, Q. Dang, et al., PP-OCR: A practical ultra lightweight OCR system, arXiv preprint arXiv:2009.09941 (2020).

[37] Z. Wang, A. C. Bovik, Mean squared error: Love it or leave it? a new look at signal fidelity measures, IEEE Signal Processing Magazine 26 (1) (2009) 98–117. doi:10.1109/MSP.2008. 930649.

[38] K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale image recognition, arXiv preprint arXiv:1409.1556 (2014).

[39] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations (ICLR), 2021.

[40] M. Lin, Q. Chen, S. Yan, Network in network, in: International Conference on Learning Representations (ICLR), 2014.

[41] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.

[42] T. Baltrušaitis, C. Ahuja, L.-P. Morency, Multimodal machine learning: A survey and taxonomy, IEEE Transactions on Pattern Analysis and Machine Intelligence 41 (2) (2019) 423–443. doi:10.1109/TPAMI.2018.2798607.

[43] G. Brauwers, F. Frasincar, A general survey on attention mechanisms in deep learning, IEEE Transactions on Knowledge and Data Engineering 35 (4) (2023) 3279–3298. doi:10.1109/ TKDE.2021.3126456.

[44] X. He, J. Cheng, Revisiting L1 loss in super-resolution: A probabilistic view and beyond, arXiv preprint arXiv:2201.10084 (2022).

[45] B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee, Enhanced deep residual networks for single image super-resolution, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017, pp. 136–144. doi:10.1109/CVPRW.2017.151.

[46] R. Timofte, E. Agustsson, L. Van Gool, M.-H. Yang, L. Zhang, B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee, et al., NTIRE 2017 challenge on single image super-resolution: Methods and results, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017, pp. 114–125.

[47] Baidu, Baidu document image dataset, https://aistudio.baidu.com/aistudio/ competition/detail/493/0/datasets, accessed: 2024.

[48] M. Bevilacqua, A. Roumy, C. Guillemot, M.-L. Alberi Morel, Low-complexity single-image super-resolution based on nonnegative neighbor embedding, in: Proceedings of the British Machine Vision Conference (BMVC), 2012, pp. 135.1–135.10. doi:10.5244/C.26.135.

[49] R. Zeyde, M. Elad, M. Protter, On single image scale-up using sparse-representations, in: Curves and Surfaces, Vol. 6920 of Lecture Notes in Computer Science, Springer, 2012, pp. 711–730. doi:10.1007/978-3-642-27413-8\_47.

[50] D. Martin, C. Fowlkes, D. Tal, J. Malik, A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics, in: Proceedings of the Eighth IEEE International Conference on Computer Vision (ICCV), Vol. 2, 2001, pp. 416–423. doi:10.1109/ICCV.2001.937655.

[51] J.-B. Huang, A. Singh, N. Ahuja, Single image super-resolution from transformed selfexemplars, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 5197–5206. doi:10.1109/CVPR.2015.7299156.

[52] Y. Matsui, K. Ito, Y. Aramaki, A. Fujimoto, T. Ogawa, T. Yamasaki, K. Aizawa, Sketch-based manga retrieval using Manga109 dataset, Multimedia Tools and Applications 76 (20) (2017) 21811–21838. doi:10.1007/s11042-016-4020-z.

[53] J. Gu, C. Dong, Interpreting super-resolution networks with local attribution maps, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 9199–9208. doi:10.1109/CVPR46437.2021.00908.

[54] Z. Kong, X. Yang, L. He, A comprehensive comparison of multi-dimensional image denoising methods, arXiv preprint arXiv:2011.03462 (2020).