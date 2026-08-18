# TISC: A Text-Driven Image Semantic Communication System for Faithful Reconstruction

Feifan Zhang<sup>∗</sup>, Yuyang Du<sup>∗</sup>, Xiaoyan Liu, Soung Chang Liew Life Fellow, IEEE

Abstract—Generative image semantic communication converts an image into a text description and then performs text-to-image reconstruction at the receiver via diffusion-based generative models. This paradigm has attracted broad attention due to its extremely low bandwidth cost. However, existing methods still face two critical bottlenecks across image-to-text (I2T) semantic extraction at the transmitter and text-to-image (T2I) semantic reconstruction at the receiver: (i) semantic loss and distortion in I2T, where holistic image descriptions may omit fine-grained object attributes and spatial-position information, causing the generated text to deviate from the original image semantics; and (ii) insufficient semantic faithfulness in T2I, where even with the same semantically faithful text description, different initial noise settings may lead diffusion-based reconstruction to produce images with different levels of semantic consistency with the original image. These issues jointly limit the semantic faithfulness of image reconstruction. To address them, we propose TISC, a textdriven image semantic communication framework tailored for faithful reconstruction. TISC incorporates two key designs: (1) Tree-Structured Attribute Semantic Extraction (TSASE), which decomposes semantic extraction into global scene, background, and object-level attribute descriptions, covering spatial position, shape/pose, color, material, and other physical attributes for each detected object; and (2) an Initial Noise Optimization (INO) mechanism, which selects an initial noise seed at the transmitter according to a comprehensive similarity score that jointly considers visual and semantic consistency. Experiments on multiple datasets show that TSASE improves object-position recovery and semantic description faithfulness, while the INO parameter study supports the adopted configuration for noise selection.

Index Terms—Semantic communication, text-driven image communication, multimodal large language model, diffusion model

## I. INTRODUCTION

With the evolution from 1G to 5G, data transmission rates in wireless communication systems have continued to increase [1]. However, cutting-edge physical-layer designs are now approaching the Shannon limit, making further improvements in effective throughput increasingly challenging [2], [3]. To meet the critical bandwidth requirement for future 6G, research in wireless networks is now moving beyond traditional architectures to semantic communication systems [4]–[6].

A semantic communication system captures the essential properties of information that are relevant to specific applications. By generating, processing, and transmitting only the most critical information, it reduces channel resource usage while maintaining effectiveness for the target applications. Semantic communications have been widely investigated as a key enabler for next-generation wireless networks [7]–[9].

Image transmission is a major application area of semantic communication. Classic image semantic communication systems send compressed visual features, such as semantic segmentation maps [10]–[12], latent representations [13]–[19], or hierarchical image features [20], over the communication link and recover the image at the receiver side with a well trained AI model tailored to feature-based image reconstruction. Recent technical advancements in multimodal large language models (MLLMs) open up a new possibility for image semantic communications [21]: the semantic information of an image can be summarized into text descriptions by a transmitterside MLLM for transmission to the receiver. The receiver then uses a generative model to convert the text back to an image. This emerging paradigm, also known as text-driven image semantic communication, introduces significantly reduced bandwidth usage compared to conventional pixel-wise image transmission or feature-based image semantic communication. We further verify this bandwidth advantage in the subsequent experiment. The results show that, at a comparable level of reconstruction similarity, the proposed text-driven semantic transmission scheme requires over 20 times fewer compressed bits than conventional image transmission (see Appendix A for details). This finding further demonstrates the potential of text-driven image semantic communication for low-bandwidth image transmission.

Despite pioneering efforts in this line of research [22]– [25], existing works exhibit notable limitations in semantic faithfulness in both I2T semantic extraction at the transmitter side and T2I semantic reconstruction at the receiver side. Fig. 1 illustrates two existing problems.

Incomplete and distorted I2T semantic representation: The MLLM for I2T semantic extraction may produce image descriptions that omit critical details from the original image or inaccurately capture certain semantic elements [26]. Such issues can arise in the representation of the number of objects, visual attributes such as colors and materials, or spatial attributes such as object positions and relationships [27]. When these text descriptions are used for T2I at the receiver, the resulting image may exhibit inconsistencies in object attributes or errors in spatial layouts.

Insufficient semantic faithfulness in T2I reconstruction: Even if the transmitter extracts and transmits a semantically faithful text description, the receiver may still fail to reconstruct an image that is semantically consistent with the original one. This is because diffusion-based generation starts from a random initial noise, and different initial noises may guide the generation process along different trajectories [28]. Therefore, even with the same text description, different initial noises can still lead to reconstructed images with varying degrees of semantic fidelity [29].

![](images/64111d5c0738d6dd37c735df1e84b18b92b69bcaf268e0b9cbc68d773b83aac9.jpg)  
Fig. 1. Comparative illustrations of conventional methods versus our generative semantic communication framework from two aspects: (a) I2T semantic distortion: conventional methods may suffer from attribute-level semantic distortions such as missing or incorrect color and spatial information, which impair semantic extraction, while our approach mitigates such distortions and improves the reliability of semantic extraction; (b) insufficient semantic faithfulness in T2I reconstruction: conventional methods randomly sample the initial noise, so different noises may lead to reconstruction results with different semantic faithfulness even under the same text description. The proposed method evaluates multiple candidate initial noises at the transmitter and selects the one that produces the most faithful reconstruction within the candidate set.

Overall, we need to answer two outstanding questions.

Question 1: I2T – How can we extract a semantically faithful text description from the original image, thereby reducing information loss and distortion during semantic extraction?

Question 2: T2I – Given a semantically faithful text description, how can we generate a reconstructed image that is semantically consistent with the original image?

This paper puts forth TISC, a text-driven image semantic communication framework addressing these questions through the following technical innovations. First, as illustrated in Fig. 1a, we develop a Tree-Structured Attribute Semantic Extraction (TSASE) module to realize accurate semantic representations. This approach decomposes complex semantic extraction into a hierarchical tree structure, where the entire image is treated as the root node and key objects within the image serve as the first-level child nodes under the root. Each object node is then connected to a set of attribute leaf nodes, such as color, shape, material, and spatial position. Through this hierarchical tree modeling, object-level semantics and finer-grained attribute-level semantics are progressively integrated into the image description. At the receiver, we further adapt the LLM-grounded Diffusion (LMD) method [30] so that it can directly use the structured text description provided by TSASE.

Second, as illustrated in Fig. 1b, we introduce Initial Noise Optimization (INO) to select an initial noise that is more favorable for semantically faithful reconstruction. The transmitter is equipped with an image generation model identical to the one at the receiver side. Before transmitting an image’s text description, the transmitter tests multiple candidate noise seeds and selects the one that yields the highest reconstruction similarity. This selected noise seed, together with the text description, is delivered to the receiver for image reconstruction.

Related work: Existing image semantic communication approaches can be grouped into two lines of research. The first delivers compressed image features, such as semantic segmentation maps [10]–[12], latent representations [13]–[19], or hierarchical semantic features [20], between the transmitter and the receiver. Although these approaches significantly reduce bandwidth consumption compared to conventional pixelwise image transmission, the compressed image features may still incur non-negligible communication overhead.

This paper belongs to the second line of research, which delivers the text modality between the transmitter and the receiver for highly efficient bandwidth utilization. In this image-text-image paradigm, the semantic content of an image is extracted as text information, and the receiver reconstructs the image using generative models. Existing studies in this direction typically employ MLLMs or image captioning models to directly generate holistic text descriptions from input images [22]–[24]. For these studies, the semantic extraction performance mainly depends on the inherent capability of the adopted model; there is no explicit optimization of the semantic extraction process for faithful image description. As a result, semantic distortions can be introduced due to potential inaccuracies in the text descriptions. In addition, the lack of explicit spatial information may result in the receiver placing objects at wrong positions. This paper aims to address these problems.

![](images/218bb0b4d8cd7c9ccb8ea8d1c45ad4761208f13c26d6fa0e34dffbfc7d9f9587.jpg)  
Fig. 2. Overall architecture of the proposed TISC framework. At the transmitter, TSASE extracts the structured semantic description D, while INO selects the initial noise seed $s ^ { * }$ . These two components are introduced in Sections II and III, respectively. The communication network delivers D and $s ^ { * }$ to the receiver, where they are used to guide LLM-grounded Diffusion-based image reconstruction.

Following the second line of research, we emphasize that this paper focuses on compressing an image source into a compact and semantically faithful text representation for transmission, and on reconstructing the image from this representation at the receiver using a generative model. The problem studied in this paper is therefore more closely related to the source coding and decoding process in communication systems: the transmitter determines which reconstructionrelevant information should be preserved and encoded while removing redundant content, and the receiver recovers the target image based on the transmitted information. Accordingly, we position TISC as a text-driven semantic source coding and decoding framework. Under this positioning, we assume that the transmitted text information can be reliably delivered through the underlying communication network, for example, via conventional channel coding, retransmission mechanisms, or TCP-like reliable transport protocols [31].

## II. I2T: ADDRESSING QUESTION 1

To address Question 1, we propose TSASE, which extracts a structured and semantically faithful text description from the original image to reduce information loss and distortion during I2T semantic extraction. As shown in the overall TISC framework in Fig. 2, this extraction process is performed at the transmitter, where the input image is converted into a textbased semantic description. This description is then used as the semantic condition for receiver-side image reconstruction. This section focuses on the design of TSASE. Subsection II-A presents the TSASE process and defines the organization of its structured semantic output. Subsection II-B explains how the receiver uses the TSASE description for image reconstruction.

## A. Tree-Structured Attribute Semantic Extraction

Straightforward approaches [22]–[24] obtain an image description by directly feeding the entire image to MLLMs.

When dealing with complex inputs, such as images with multiple objects, these methods often overlook subtle semantic details, leading to inaccurate image descriptions at the transmitter side. TISC addresses this issue with the TSASE paradigm introduced as follows.

TSASE decomposes a complex image description task into a set of simpler subtasks, with each subtask aiming to clearly describe one object within the image. For each object-specific subtask, its description is further decomposed into several attribute-specific mini-tasks, with each of these mini-tasks focusing on one aspect of the object’s attribute (e.g., spatial position, shape, color, material, etc.). With the completion of all these attribute-specific mini-tasks, we obtain detailed descriptions for an object. In addition to these detailed object descriptions, a global scene of the image and a specific description focusing on the background are also included in the image description to ensure faithful reconstruction at the receiver side.

We now elaborate the TSASE paradigm with Fig. 3. At the beginning of the semantic extraction process, we take the whole image as the “root node” of the tree structure. We give the whole image to an MLLM and ask the model to generate the root node, which includes the image’s global description $D _ { \mathrm { g l o b a l } }$ and the image’s background description $D _ { \mathrm { b g } } .$ . In Fig. 3, for example, $D _ { \mathrm { g l o b a l } }$ highlights the primary content and relationships in the image with description “A corgi dog and an orange tabby cat sit side by side on green grass in $s u n l i g h t ^ { \prime }$ , while $D _ { \mathrm { b g } }$ emphasizes the background of the image with description “An outdoor garden scene bathed in warm, golden sunlight”.

After obtaining the global scene information, TSASE employs YOLOv11 [32] to detect and localize all objects within the image. Let us assume N objects within the given image. The set of rectangular bounding boxes for detected objects are denoted by $\textbf { \em B } = ~ \langle b _ { 1 } , \ldots , b _ { N } \rangle$ where bounding box $\boldsymbol { b _ { i } } ~ = ~ ( x _ { i } ^ { \mathrm { { t l } } } , y _ { i } ^ { \mathrm { { t l } } } , x _ { i } ^ { \mathrm { { b r } } } , y _ { i } ^ { \mathrm { { b r } } } )$ specifies the top-left and bottomright coordinates of object i. The origin (0, 0) of the image coordinate system is defined at the top-left corner of the whole image. To prevent mistaken object detections, such as multiple bounding boxes for a single object, we have a pre-defined confidence threshold for the object detection, which is used to compare with the confidence score that YOLOv11 provided for each detected object, to filter out low-confidence detections that are likely to be problematic.

![](images/f8b7df4a9c7045ea49c1cd68fb8680559f19235a5c300e3883bf37a88d302118.jpg)  
Fig. 3. Example of the tree-structured attribute semantic extraction process.

Based on the remaining bounding boxes, the whole image is cropped into N single-object sub-images, denoted by $\{ \bar { I } _ { i } ^ { \mathrm { o b j } } \} _ { i = 1 } ^ { N }$ which are then used to extract the fine-grained visual attributes of the corresponding objects. These subimages serve as first-level child nodes of the original image. In TSASE, we have a pre-defined list of finer-grained attributes of interest to us, which include the object’s spatial position, shape/pose, color, and material/physical attributes<sup>1</sup>. For subimage $\overline { { I _ { i } ^ { \mathrm { o b j } } } }$ corresponding to the i-th first-level child node, these attributes of that sub-image serve as the second-level leaf nodes. Among these attributes, the spatial-position leaf node is directly given by the bounding box $b _ { i }$ obtained from YOLOv11. For example, the spatial attribute of the first object (i.e., dog in Fig. 3) is $\boldsymbol { b _ { 1 } } ~ = ~ ( 0 , 5 0 , 3 2 7 , 4 6 8 )$ . For each remaining attribute leaf node, we ask the MLLM to describe one attribute of sub-image $I _ { i } ^ { \mathrm { o b j } }$ at one time, helping the MLLM to concentrate on one specific aspect of the sub-image in one interaction. In Fig. 3, the shape/pose description of the first sub-image could be: “The dog is sitting on the ground with its front legs visible and its body upright. Its mouth is slightly open, and its ears are upright and pointed outward.”

After all attribute nodes are obtained, TSASE integrates the spatial-position attribute and the MLLM-generated attribute descriptions of the i-th object to form a semantic representation $d _ { i }$ . Note that $d _ { i }$ is a simple concatenation of attribute descriptions, which can be rather lengthy (see the “long version” object description illustrated in Fig. 4). To reduce information redundancy, we leverage an LLM $( \mathrm { e . g . }$ , GPT-4o [33] in this paper) to further refine the long version into a “short version” object description. For example, for the first sub-image in Fig. 3, its long version object description “[0,50,327,468]: A tricolor dog with a fluffy, medium-length coatfeaturing shades of brown, black, and white. The dog is sitting upright on the ground with its front legs visible. . . ” can be compressed as “[0,50,327,468]: A fluffy, tricolor, medium-sized, white-faced, brown-and-black corgi $d o g '$ . In our experiments, we find that short version descriptions can achieve generation results no worse than longer ones. Hence, this paper uses the short description of object i, which is denoted as $d _ { i } ^ { S }$ , as the object semantic representation.

With the short-version description for each object, we obtain the object-level representation of the whole image as $D _ { \mathrm { o b j } } ~ = ~ [ d _ { 1 } ^ { S } , \ldots , d _ { N } ^ { S } ]$ . Finally, $D _ { \mathrm { g l o b a l } } , D _ { \mathrm { b g } }$ , and $D _ { \mathrm { o b j } }$ are concatenated to form a comprehensive image description $D = \{ D _ { \mathrm { g l o b a l } } , D _ { \mathrm { b g } } , D _ { \mathrm { o b j } } \}$

To further clarify the practical organization of the TSASE output, Fig. 4 provides an example of the structured semantic description D. As illustrated in Fig. 4, the information is organized as follows:

1) the beginning of the image description includes $D _ { \mathrm { g l o b a l } } ,$ which provides an overall summary of the image content.

![](images/e9d99deef3e9243bda05ae0a7e0dc9467a5d103c2a752e79cb765f06918752e4.jpg)  
Fig. 4. An example of the structured semantic description produced by TSASE.

2) the background description $D _ { \mathrm { b g } }$ supplements contextual information of the environment, facilitating more accurate reconstruction of the image background.

3) object-level descriptions $D _ { \mathrm { o b j } }$ are presented together.

After obtaining D, the transmitter treats it as the text-based semantic information to be delivered. In this paper, we assume that the transmitted text information can be reliably delivered through the communication network, e.g., via TCP. As shown by the yellow arrow in Fig. 2, D is transmitted to the receiver and used as the semantic condition for image reconstruction.

## B. TSASE-Guided Image Reconstruction

As shown in Fig. 2, the receiver obtains the TSASE output D through the reliable communication network. The receiverside image reconstruction in TISC is built upon the opensource LLM-grounded Diffusion (LMD) model [30], and is jointly guided by D and the noise seed $s ^ { * }$ . This subsection focuses on how D is adapted to the input of LMD, such that the global scene description, background description, and object-level attribute descriptions can be used for image reconstruction. The selection of $s ^ { * }$ and its role in providing the initial noise for the LMD generation process will be discussed in Section III. We only briefly introduce the LMD functions relevant to this adaptation, without delving into its implementation details or underlying mechanisms (we refer interested readers to Section III in [30] for details).

To clarify how LMD is adapted in our framework, we first briefly review the original LMD generation pipeline. In generation, the image reconstruction pipeline in original LMD can be divided into two major steps: 1) LLM-driven reasoning of semantic information, and 2) diffusion-based image generation with object-level semantic and spatial alignment.

For the first step, LMD employs an LLM to interpret the text description input. The LLM parses object information, background information, and relative spatial relationships among objects from the free-form text. Leveraging its contextual understanding and prior knowledge of the real world, the LLM estimates the spatial coordinates of different objects and generates the description of each object, the background description of the image, and the global scene description of the image to enrich them accordingly.

For the second step, the enriched semantic information, along with a randomly generated noise vector, is given to a diffusion model for guiding the image generation. During this process, LMD first generates the semantic representation of each object based on its description. To maintain spatial alignment, each object’s semantic representation will be placed at a location predefined in its spatial description. The above objectlevel treatments ensure semantic and spatial alignment for all objects within the image. Following object-level treatments, LMD generates the whole image based on the representation of each object, the background and the global descriptions of the image, and the initial noise vector.

We further clarify the difference between our adaptation of the LMD input flow and the original LMD framework. In the original LMD, step (1) is performed by the LLM to generate semantic information from the input text description via model reasoning. In our framework, however, D already contains precise semantic details extracted from the original image, making the LLM reasoning in step 1 unnecessary—we directly use D as the input for step 2 of the original LMD pipeline.

## III. T2I: ADDRESSING QUESTION 2

As discussed in Section I, even when the receiver obtains a semantically faithful text description, diffusion-based reconstruction may still produce images with different levels of semantic faithfulness due to different initial noises. Therefore, Question 2 focuses on how to select an initial noise that is more favorable for reconstructing the semantic content of the original image.

To address this question, we introduce an Initial Noise Optimization (INO) mechanism at the transmitter. INO tests multiple candidate initial noises before transmission and uses the comprehensive similarity score $S _ { \mathrm { c o m p } }$ to evaluate the consistency between each corresponding reconstruction and the original image. Here, $S _ { \mathrm { c o m p } }$ is an image similarity metric defined in this paper, which jointly considers visual similarity and semantic similarity. INO then selects the initial noise with the highest $S _ { \mathrm { c o m p } }$ score to support more semantically faithful image reconstruction at the receiver.

This section introduces INO from two aspects. Section III-A first presents the basic pipeline of INO. Section III-B then defines the proposed similarity evaluation metric $S _ { \mathrm { c o m p } }$ used in INO.

## A. Initial Noise Optimization

At the transmitter side, we deploy an image generation model $G ( \cdot )$ that is identical to the one at the receiver. The aim of doing so is to assess the expected image reconstruction performance under different initial noises before the transmission process. Let us assume that the transmitter tests m noise vectors, denoted by $Z = [ z _ { 1 } , \dots , z _ { m } ]$ , to select the bestperforming initial noise among these candidates. The choice of m controls the search budget of INO and will be determined through the parameter study in Experiment 1 of Section IV-B. For each $z _ { q } \in Z$ , conditioned on the semantic description D produced by TSASE, the reconstructed image is generated as $\hat { I } _ { q } = G ( D , z _ { q } )$

At the transmitter, we evaluate the similarity between the original image I and each candidate reconstruction $\hat { I } _ { q }$ using the proposed comprehensive similarity score $S _ { \mathrm { c o m p } } ( \cdot , \cdot )$ (see Section III-B and Eq. 6 for more details). Among all possible noise vectors, the one resulting in the highest similarity with I is selected, i.e.,

$$
z ^ { * } = \arg \operatorname* { m a x } _ { z _ { q } \in Z } S _ { \mathrm { c o m p } } ( I , \hat { I } _ { q } ) .\tag{1}
$$

After finding the best-performing noise vector $z ^ { * }$ , the next issue is how to convey this selected noise to the receiver efficiently. We note that there is no need to transmit the complete noise vector for communication efficiency—transmitting the random seed for generating the noise is enough. Let us denote the random noise seed that one-to-one maps to $z ^ { * }$ by $s ^ { * }$ and write it as $s ^ { * } = f ( z ^ { * } )$ , where f denotes the mapping function.

Subsequently, as indicated by the blue arrow in Fig. 2, the selected seed $s ^ { * }$ is transmitted to the receiver through the communication network. Here, $s ^ { * }$ is essentially a random seed used to generate the initial noise $z ^ { * }$ , and thus can be transmitted as text-form information with very low communication overhead. In our receiver-side pipeline, we set $z ^ { * }$ as the starting point of the LMD diffusion generation process, and use the semantic description D as the structured semantic input to LMD for image reconstruction, following the method described in Section II-B.

B. Comprehensive Similarity Metric for Candidate Noise Evaluation

In INO, the comprehensive similarity score $S _ { \mathrm { c o m p } } ( I , \hat { I } _ { q } )$ is used to measure the consistency between the original image I and the q-th candidate reconstruction $\hat { I } _ { q } .$ This score indicates whether the corresponding candidate initial noise $z _ { q }$ is favorable for reconstructing the semantic content of the original image. Since TISC aims at semantically faithful image reconstruction, this consistency should be evaluated from both visual and semantic perspectives. Therefore, we define two components, namely the visual similarity $S _ { \mathrm { v i s } }$ and the semantic similarity $S _ { \mathrm { s e m } }$ , and then construct $S _ { \mathrm { c o m p } }$ based on them.

To measure visual consistency, we adopt the Learned Perceptual Image Patch Similarity (LPIPS) distance [34]. LPIPS quantifies the perceptual distance between two images using a pretrained convolutional neural network (CNN), such as AlexNet [35] or VGG [36]. Let us denote the network by $F ( \cdot )$ , and we write the network’s output embedding at layer l as $F _ { l } ( I )$ . Given I and $\hat { I } _ { q } ,$ the LPIPS of the two images is given by:

$$
\mathrm { L P I P S } ( I , \hat { I } _ { q } ) = \sum _ { l } w _ { l } \cdot \left\| F _ { l } ( I ) - F _ { l } ( \hat { I } _ { q } ) \right\| _ { 2 } ^ { 2 } .\tag{2}
$$

where $w _ { l }$ is a predefined weight of layer l within the network, and $\left\| \cdot \right\| _ { 2 } ^ { 2 }$ denotes the normalized Euclidean distance between two vectors. A lower $\mathrm { L P I P S } ( I , \hat { I } _ { q } )$ indicates that the two images are closer in the perceptual feature space. To make the visual term consistent with the subsequent semantic similarity term, where a larger value indicates higher similarity, we convert the LPIPS distance into a visual similarity score:

$$
S _ { \mathrm { v i s } } ( I , \hat { I } _ { q } ) = 1 - \mathrm { L P I P S } ( I , \hat { I } _ { q } ) .\tag{3}
$$

Therefore, a larger $S _ { \mathrm { v i s } }$ indicates higher perceptual similarity between the original image and the candidate reconstruction.

For semantic similarity, we focus on the similarity between the tree-structured semantic descriptions of the original image I and the q-th candidate reconstruction $\hat { I } _ { q } .$ . To this end, we use TSASE to extract structured descriptions from both images, and further compare their text similarities at three levels: the global scene, the background, and object-level semantics. Specifically, for the original image I, TSASE extracts the global scene description $D _ { \mathrm { g l o b a l } }$ , the background description $D _ { \mathrm { b g } } .$ , and object-level semantic descriptions $D _ { \mathbf { o b j } } =$ $[ d _ { 1 } ^ { S } , \ldots , d _ { n _ { 1 } } ^ { S } ]$ , where $d _ { i } ^ { S }$ denotes the short-version structured description of the i-th detected object (see Section II-A for more details). For the corresponding reconstructed image $\hat { I } _ { q } ,$ TSASE similarly extracts the corresponding descriptions $D _ { \mathrm { g l o b a l } } ^ { \prime } , \ D _ { \mathrm { b g } } ^ { \prime } .$ , and ${ \bf \dot { \cal D } _ { o b j } ^ { \prime } } = [ d _ { 1 } ^ { \prime S } , \dots , { \dot { d ^ { \prime } } _ { n _ { 2 } } ^ { S } } ]$ , where $n _ { 1 }$ and n<sub>2</sub> denote the numbers of detected objects in I and $\hat { I } _ { q } ,$ respectively.

To measure the semantic closeness between these text descriptions, we adopt Sentence-BERT (SBERT) [37] as the text similarity model. SBERT encodes each text description into a semantic embedding vector, and the similarity between two descriptions is computed based on the cosine similarity of their embeddings. In this paper, SBERT(·, ·) denotes the normalized SBERT-based similarity score in [0, 1], where a larger value indicates higher semantic similarity.

Based on this definition, the global scene similarity and the background similarity are computed as SBERT $( D _ { \mathrm { g l o b a l } } , D _ { \mathrm { g l o b a l } } ^ { \prime } )$ and SBERT $( D _ { \mathrm { { b g } } } , D _ { \mathrm { { b g } } } ^ { \prime } )$ respectively. For object-level semantic similarity, it is not appropriate to directly compare the object-level descriptions according to the detection order of the corresponding objects, because the detected object sets in the original image and the candidate reconstruction may differ in both size and order. Therefore, we first establish object correspondences between the two images according to their spatial locations. These correspondences determine which object-level descriptions can be compared in a one-to-one manner.

To establish this spatial matching relationship, we compute the Intersection over Union (IoU) using the bounding-box coordinates contained in the object-level descriptions. IoU is a commonly used metric for measuring the spatial overlap between two bounding boxes [38]. Specifically, for the i-th object description $d _ { i } ^ { S }$ from the original image, its spatialposition attribute provides the bounding box $b _ { i } ;$ for the $j -$ th object description from the candidate reconstruction, its spatial-position attribute provides the bounding box $b _ { j } ^ { \prime }$ . Fig. 5 presents an illustrative example of the IoU calculation, where the IoU between $b _ { i }$ and $b _ { j } ^ { \prime }$ is computed as:

$$
\mathrm { I o U } = \frac { \arctan ( b _ { i } \cap b _ { j } ^ { \prime } ) } { \arctan ( b _ { i } \cup b _ { j } ^ { \prime } ) } .\tag{4}
$$

![](images/d9f83e92ceafcdf26970b73bb41c8c7c4e4d753a125b37e88ec00ade163d65e1.jpg)  
(a)

![](images/c3b42ef2544b4f86cfcdec6aa006a3f29a8970ff6ef8c247f928a7e5d32e80fd.jpg)  
(b)

![](images/870b5c86f780ee976995c794a535452c7dcdebfc3e720dd9e7fa23ca578c5c5a.jpg)  
(c)

Fig. 5. Illustration of IoU calculation for spatial matching. (a) Original image, where the yellow box denotes the bounding box $\bar { b _ { i } }$ provided by the spatial-position attribute of the i-th object description $d _ { i } ^ { S } .$ (b) Candidate reconstruction, where the blue box denotes the bounding box $\dot { \mathbf { \delta } } _ { b _ { j } ^ { \prime } } ^ { \prime }$ provided by the spatial-position attribute of the j-th object description $d _ { \ j } ^ { \prime S }$ . (c) Spatial overlap between $\mathbf { \delta } _ { b _ { i } }$ and $b _ { j } ^ { \prime } .$ , which is used to compute their IoU.  
![](images/65e794ea64e0d3412341548f97e9e2f61894c1314378c52c52ed7b40109e62c9.jpg)  
Fig. 6. Illustration of matched, missed, and hallucinated objects based on spatial matching. The left panel shows the original image, and the right panel shows the reconstructed image. Green boxes indicate matched objects that appear in both images with sufficient spatial overlap. The red box indicates a missed object that appears only in the original image, while the orange box indicates a hallucinated object that appears only in the reconstructed image.

Here, $( b _ { i } \cap b _ { i } ^ { \prime } )$ is the overlap area and $( b _ { i } \cup b _ { j } ^ { \prime } )$ is the union area. A larger IoU indicates a higher degree of spatial overlap between the two detected objects. Following the commonly used criterion in object detection evaluation [38], we perform spatial matching between detected objects in the original image and the candidate reconstruction. Two detected objects are regarded as spatially matched if their IoU is larger than 0.5.

Based on this spatial matching result, the objects in the original and reconstructed images are divided into three categories: matched, missed, and hallucinated objects. Fig. 6 shows an illustrative example. Matched objects refer to object pairs that appear in both images and are spatially matched (i.e., the red mug and the blue book in Fig. 6). We use $N _ { \mathrm { m a t } }$ to denote the number of matched object pairs (e.g., $N _ { \mathrm { m a t } } = 2$ in Fig. 6), and $M = \{ i _ { k } , j _ { k } \} _ { k = 1 } ^ { N _ { \mathrm { m a t } } }$ to denote the set of all matched object pairs, where $i _ { k }$ and $j _ { k }$ index the k-th matched objects in the original and reconstructed images, respectively. For the k-th matched object pair $( i _ { k } , j _ { k } )$ in M, we compute its objectlevel semantic similarity as SBERT $( d _ { i _ { k } } ^ { S } , d _ { j _ { k } } ^ { \prime S } )$ . Missed objects refer to objects that appear in the original image but have no spatially matched counterpart in the reconstructed image $( \mathrm { i . e . }$ the plant in Fig. 6), and their number is denoted by $N _ { \mathrm { m i s s } }$ (e.g., $N _ { \mathrm { m i s s } } = 1$ in Fig. 6), and hallucinated objects refer to extra objects generated in the reconstructed image but absent from the original image (i.e., the toy duck in Fig. 6), and their number is denoted by $N _ { \mathrm { h a l } } ~ ( \mathrm { e . g . , } ~ N _ { \mathrm { h a l } } = 1$ in $\operatorname { F i g . } \ 6 )$ . The similarity scores of both missed and hallucinated objects are set to zero.

Based on the above definitions, the final TSASE-based semantic similarity is computed as the average similarity over the global scene, the background, and all object-level semantic descriptions:

$$
\begin{array} { c } { { \displaystyle \mathrm { S B E R T } ( D _ { \mathrm { g l o b a l } } , D _ { \mathrm { g l o b a l } } ^ { \prime } ) + \mathrm { S B E R T } ( D _ { \mathrm { b g } } , D _ { \mathrm { b g } } ^ { \prime } ) } } \\ { { + \displaystyle \sum _ { k = 1 } ^ { N _ { \mathrm { m a t } } } \mathrm { S B E R T } ( d _ { i _ { k } } ^ { S } , d _ { j _ { k } } ^ { \prime S } ) } } \\ { { S _ { \mathrm { s e m } } ( I , \hat { I } _ { q } ) = \displaystyle \frac { { K _ { \mathrm { m a t } } + N _ { \mathrm { m i s s } } + N _ { \mathrm { h a l } } + 2 } \quad \quad } { N _ { \mathrm { m a t } } + N _ { \mathrm { m i s s } } + N _ { \mathrm { h a l } } + 2 } } } \end{array}\tag{5}
$$

Here, the additional term +2 in the denominator corresponds to the global scene description and the background description, while $N _ { \mathrm { m a t } } , \ N _ { \mathrm { m i s s } } .$ , and $N _ { \mathrm { h a l } }$ correspond to the numbers of matched, missed, and hallucinated objects, respectively. Since missed and hallucinated objects are assigned zero similarity scores, they do not appear in the numerator but are included in the denominator, thereby reducing the semantic similarity score. Compared with a single globalcaption similarity metric, this TSASE-based metric evaluates semantic faithfulness from three levels: the overall scene, the background information, and object-level semantic details. It also explicitly reflects the impact of missing objects and hallucinated objects in the reconstructed image.

After obtaining the visual similarity $S _ { \mathrm { v i s } } ( I , \hat { I } _ { q } )$ and the semantic similarity $S _ { \mathrm { s e m } } ( I , \hat { I } _ { q } )$ , we fuse them with a weighted sum to obtain the comprehensive similarity score for candidate noise evaluation:

$$
\begin{array} { r } { S _ { \mathrm { c o m p } } ( I , \hat { I } _ { q } ) = \alpha S _ { \mathrm { v i s } } ( I , \hat { I } _ { q } ) + ( 1 - \alpha ) S _ { \mathrm { s e m } } ( I , \hat { I } _ { q } ) . } \end{array}\tag{6}
$$

Here, $\alpha \in [ 0 , 1 ]$ controls the relative weights of the visual and semantic similarities, and its value is determined through the parameter study (see Experiment 2 in Section IV-B for more details). Since both $\bar { S _ { \mathrm { v i s } } } ( I , \hat { I } _ { q } )$ and $S _ { \mathrm { s e m } } ( I , \hat { I } _ { q } )$ lie in $[ 0 , 1 ] , S _ { \mathrm { c o m p } } ( I , \hat { I } _ { q } )$ also lies in [0, 1]. A larger $S _ { \mathrm { c o m p } }$ indicates higher consistency between the candidate reconstruction and the original image in both visual and semantic aspects.

## IV. EXPERIMENTAL RESULTS

This section evaluates the key designs of the proposed TISC framework through experiments. The experiments are organized into two subsections. Section IV-A conducts an ablation study on the TSASE module to evaluate its effectiveness in semantic extraction. Specifically, we assess the contribution of TSASE from two perspectives: spatial-attribute recovery and the semantic faithfulness of the extracted descriptions. Section IV-B further analyzes the parameter settings of the INO mechanism, focusing on the number of candidate initial noises m and the weighting factor α between visual similarity and semantic similarity in the comprehensive similarity score $S _ { \mathrm { c o m p } }$ . The datasets used in these experiments are summarized as follows.

Tested Datasets: we tested five representative image datasets with complementary characteristics, covering diverse visual scenes and object complexities. Tested dataset includes

![](images/4e0f8ee6030f7f1c42db1088f39dfca8b88859fbf94ed3c024182c14327bf0a1.jpg)  
Fig. 7. Comparison of average IoU between TISC and the w/o TSASE ablation across tested datasets.

COCO [39], Visual Genome V1.2 [40], Flickr30K [41], Open Images V7 [42], and the ILSVRC2012 validation set [43].

## A. TSASE Ablation Study

This subsection evaluates the semantic extraction capability of the TSASE module. To this end, we compare the complete TISC framework with the following ablation setting:

w/o TSASE: The TSASE module is removed. In this setting, the transmitter no longer extracts tree-structured descriptions. Instead, an MLLM is directly used to generate a one-shot holistic description of the entire image. To focus the comparison on the role of TSASE, all other system settings remain unchanged. The receiver still uses the noise seed selected by INO and reconstructs the image through the LMD model based on this alternative text description. Note that this setting is consistent with the holistic-description-based imagetext-image pipeline adopted by the related works [22]–[24] discussed in Section I. Therefore, while this setting is designed as an ablation case for isolating the role of TSASE, it can also be viewed as a representative benchmark setting for these prior text-driven semantic communication systems.

TSASE extracts a structured semantic description from the original image, which includes not only fine-grained visual attributes such as color, shape, and material, but also the spatial-position attributes of objects. Therefore, we evaluate its semantic extraction capability from two perspectives. First, we examine whether the spatial attributes extracted by TSASE help improve the recovery of object positions in the reconstructed image. Second, we evaluate whether TSASE can more accurately extract semantic information beyond spatial position.

We first evaluate the recovery of spatial attributes. Given an original image transmitted and the image recovered at the receiver side under the ablation setup, we evaluate the spatial accuracy of the reconstructed image with IoU (see Eq. (4) for more details).

For each test dataset, we compute the average IoU over all matchable objects to measure the overall recovery of object positions in the reconstructed images. Let us assume a total of n objects in the dataset for calculating the IoU, then the average IoU of a tested dataset can be expressed as

$$
{ \overline { { \mathrm { I o U } } } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathrm { I o U } _ { i } ,\tag{7}
$$

![](images/57a0d40041a0451fa69086e5268418984aedf4e64c3212d829e489145a0a0952.jpg)  
Fig. 8. Average TIFA scores (TIFA) of TISC and w/o TSASE across five tested datasets.

where $\mathrm { I o U } _ { i }$ denotes the IoU of the i-th detected object.

Fig. 7 benchmarks the average IoU of TISC and the w/o TSASE ablation across five tested datasets. The figure shows that TISC consistently achieved higher average IoUs than w/o TSASE across all datasets—with most datasets exceeding 75% (e.g., 78.53% on COCO, 79.31% on ILSVRC), while the ablation case is around 30% in general.

Next, we evaluate the accuracy of TSASE in extracting semantic description. Since TSASE aims not only to provide spatial positions, but also to generate more accurate object attributes, background information, and global scene descriptions, we employed the TIFA metric (Text-to-Image Faithfulness Assessment) [44] to quantify the consistency between the extracted semantic description and the original image. TIFA is an automated evaluation framework based on fine-grained Visual Question Answering (VQA) powered by an external MLLM, designed to measure the alignment between text-toimage generations and the ground-truth text descriptions. In our use of TIFA, only the original image and the transmitterside text description are required, i.e., this evaluation does not involve the receiver-side image reconstruction process. Therefore, this experiment directly compares the semantic extraction accuracy of TSASE with the one-shot holistic description used in w/o TSASE.

We introduce the TIFA evaluation pipeline with the illustration of Fig. 9. TIFA applies the follow evaluation pipeline to both TISC and w/o TSASE:

1) Given a text description D, TIFA parses it into distinct semantic components (e.g., objects, color) for later processing.

2) Observing components obtained in 1), TIFA asks a set of questions $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { n } \}$ according to predefined rules and templates.

3) a corresponding answer set $A ~ = ~ \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { n } \}$ is derived from the description D, where each question requires a binary (yes/no) answer (see the yellow part in Fig. 9).

4) To obtain the ground-truth answers that accurately reflects the semantics of the original image, Q and I are fed into a VQA model (such as BLIP-2 [45]) to produce standard answer set $A ^ { * } = \left\{ a _ { 1 } ^ { * } , a _ { 2 } ^ { * } , \ldots , a _ { n } ^ { * } \right\}$ (see the blue part in Fig. 9).

TIFA measures the alignment between text description D and image I by comparing A against A<sup>∗</sup>. TIFA score is defined as the accuracy over all questions, which is written as:

![](images/8bb9e4177d2eb450c383d631491228b7f5c2dd19b7a5e486f38a6f9cb2cdc988.jpg)  
Fig. 9. TIFA evaluation pipeline.

$$
\mathrm { T I F A } ( { \cal D } , { \cal I } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { I } ( a _ { i } = a _ { i } ^ { * } ) ,\tag{8}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function that outputs 1 if $a _ { i } = a _ { i } ^ { * }$ (otherwise, outputs 0).

For a given dataset T with N images $\{ I _ { i } \} _ { i = 1 } ^ { N }$ . The datasetlevel average TIFA is defined as

$$
\overline { { \mathrm { T I F A } } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { T I F A } ( D _ { i } , I _ { i } ) ,\tag{9}
$$

For TISC, we use the TSASE module to extract a comprehensive text description $D _ { \mathbf { T S A S E } }$ , and we have $\textbf { = }$ $D _ { \mathbf { T S A S E } }$ in the TIFA evaluation pipeline. For w/o TSASE, as introduced in the experimental set, we use GPT-4o [33] to place the TSASE module, and we ask the model to directly generate an overall description of the image. Let us denote the text description generated by GPT-4o as $\pmb { D } _ { 4 0 }$

Fig. 8 shows the average TIFA scores of TISC and w/o TSASE across five datasets, from which we see that TISC consistently achieves higher average TIFA score on all datasets.

## B. INO Parameter Study

This subsection further analyzes two key parameter settings in the INO mechanism. Unlike the TSASE ablation study, the performance of INO mainly depends on the search range of candidate initial noises and the evaluation criterion used to select among candidate reconstructions. Therefore, this subsection is organized into two experiments corresponding to the following two questions:

First, how should the number of candidate initial noises m be determined to obtain a high-quality reconstruction with high probability?

Second, how should the weighting parameter α in the comprehensive similarity score $S _ { \mathrm { c o m p } }$ be set to balance visual similarity and semantic similarity?

## Experiment 1: Selection of the Number of Noise Seeds

We first address the selection of the candidate seed number m. Since different initial noise seeds may lead to different candidate reconstructions, the seed budget m should ensure, with high probability, that the candidate set contains at least one high-quality reconstruction, while avoiding unnecessary candidate image generation. Our goal is to obtain, through rigorous statistical reasoning, a unified seed budget that is independent of any specific image. Otherwise, if the required number of seeds varied with the particular image, an additional method would be needed to identify the proper seed number for each image, introducing a cumbersome image-specific procedure.

To formalize the above seed budget selection problem, we introduce the following notation. Consider the r-th original image I . Suppose the INO module uses m independent initial noise seeds to generate m candidate reconstructions, denoted by $\{ \hat { I } _ { r , q } \} _ { q = 1 } ^ { m }$ . The corresponding comprehensive similarity scores are given by $\{ \bar { S } _ { \mathrm { c o m p } } ( \bar { I } _ { r } , \hat { I } _ { r , 1 } ) , \bar { S } _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , 2 } ) , \dots , S _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , m } ) \}$ . Here, $S _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , q } )$ denotes the comprehensive similarity score between the original image I and the corresponding candidate reconstruction $\hat { I } _ { r , q }$ generated from the q-th seed. We further use $S _ { \mathrm { c o m p } } ^ { ( r ) }$ to denote the random variable of the comprehensive similarity score induced by a randomly sampled seed for the r-th image. Since the INO module selects the reconstruction with the highest similarity score, the quality achieved under seed budget m is determined by $\mathrm { m a x } _ { q = 1 , \dots , m } S _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , q } )$

Under this notation, the objective of seed budget selection is to choose the smallest possible m such that the probability that at least one candidate in the set reaches a high-quality reconstruction threshold is no smaller than a prescribed confidence level $p .$ Formally, we require

$$
P \left( \underset { q = 1 , \ldots , m } { \operatorname* { m a x } } S _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , q } ) \geq t _ { r } ( k ) \right) \geq p .\tag{10}
$$

Here, $t _ { r } ( k )$ denotes the image-dependent quality threshold for the r-th image, where k controls how strict the high-quality criterion is (see Eq. (11) for more details). Although $t _ { r } ( k )$ depends on the distribution parameters of each image, the resulting seed budget m will be shown to depend only on predefined hyperparameter k and $p ,$ rather than on the image index r. This property is important because it provides a uniform seed selection rule across all images. To evaluate the probability in Eq. (10) and solve for the required seed budget, we first need to characterize the distribution of the similarity score $S _ { \mathrm { c o m p } } ^ { ( r ) }$

To empirically characterize the distribution of $S _ { \mathrm { c o m p } } ^ { ( r ) }$ , we randomly select three images from the COCO image dataset and generate 500 candidate reconstructions for each image using 500 different initial noise seeds. For each generated candidate, we compute the comprehensive similarity score according to Eq. (6), while keeping the semantic description, generation model, and inference configuration fixed. We then plot the histogram of the obtained comprehensive similarity scores and fit a Gaussian distribution to each histogram.

As shown in Fig. 10, for each fixed image, the comprehensive similarity scores produced by different initial noise seeds form a stable unimodal distribution. The fitted Gaussian curves are well aligned with the empirical histograms, indicating that the distribution of $S _ { \mathrm { c o m p } } ^ { ( r ) }$ can be reasonably approximated by a Gaussian distribution. Therefore, for the r-th image, we model $S _ { \mathrm { c o m p } } ^ { ( r ) } \sim \mathcal { N } ( \mu _ { r } , \sigma _ { r } ^ { 2 } )$ , where $\mu _ { r }$ and $\sigma _ { r }$ denote the mean and standard deviation of the comprehensive similarity scores under random seed sampling.

Under this Gaussian model, we define a high-quality candidate as a reconstruction whose comprehensive similarity score is significantly higher than the average seed outcome for the same image. Specifically, we set the high-quality threshold as

$$
t _ { r } ( k ) = \mu _ { r } + k \sigma _ { r } ,\tag{11}
$$

where k specifies how many standard deviations above the image-specific mean the high-quality threshold is set. A larger k requires a candidate reconstruction to exceed the average similarity level by more standard deviations, and therefore corresponds to a stricter definition of high-quality reconstruction. In this paper, we use $k = 1$ as the default setting, meaning that a candidate reconstruction is regarded as high-quality if its comprehensive similarity score is at least one standard deviation above the image-specific mean. For a Gaussian distribution, samples above $\mu _ { r } + \sigma _ { r }$ account for approximately the top 15.87% of the distribution.

According to the Gaussian distribution, the probability that a single seed fails to reach this threshold is

$$
P \left( S _ { \mathrm { c o m p } } ^ { ( r ) } < t _ { r } ( k ) \right) = P \left( \frac { S _ { \mathrm { c o m p } } ^ { ( r ) } - \mu _ { r } } { \sigma _ { r } } < k \right) = \Phi ( k ) ,\tag{12}
$$

where $\Phi ( \cdot )$ is the cumulative distribution function of the standard normal distribution. It is important to note that this probability depends only on k, rather than on the image index r. Although different images may have different $\mu _ { r }$ and $\sigma _ { r }$ , these image-dependent parameters are removed by standardization.

Based on the above derivation and the probability requirement in Eq. (10), the required seed budget m varies with the chosen parameter k in $t _ { r } ( k )$ . Therefore, we denote it as $m _ { k }$ in the following derivation.

If $m _ { k }$ independent candidate reconstructions are generated, the probability that none of them reaches the high-quality threshold is

$$
P \left( \operatorname* { m a x } _ { q = 1 , \dots , m _ { k } } S _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , q } ) < t _ { r } ( k ) \right) = \Phi ( k ) ^ { m _ { k } } .\tag{13}
$$

Thus, the probability that at least one candidate reaches the high-quality threshold is

$$
P \left( \underset { q = 1 , \ldots , m _ { k } } { \operatorname* { m a x } } S _ { \mathrm { c o m p } } ( I _ { r } , \hat { I } _ { r , q } ) \geq t _ { r } ( k ) \right) = 1 - \Phi ( k ) ^ { m _ { k } } .\tag{14}
$$

To ensure that this probability is no smaller than the prescribed confidence level $p ,$ we require

$$
\begin{array} { r } { 1 - \Phi ( k ) ^ { m _ { k } } \geq p . } \end{array}\tag{15}
$$

Solving for $m _ { k }$ , we obtain

$$
m _ { k } \geq \left\lceil { \frac { \ln ( 1 - p ) } { \ln ( \Phi ( k ) ) } } \right\rceil .\tag{16}
$$

Here, ⌈·⌉ denotes the ceiling operation, ensuring that the seed budget is the smallest integer satisfying the probability constraint.

In our default setting, we set $k = 1$ and the target confidence level to $p ~ = ~ 0 . 9 0$ , which corresponds to a seed budget of $m _ { k } ~ = ~ 1 4$ under the above formulation. This means that the candidate set is expected to contain at least one highquality reconstruction with 90% probability. Therefore, in the subsequent experiments, we use 14 initial noise seeds by default for candidate reconstruction generation and selection.

Beyond the specific value of $m _ { k } = 1 4$ used in the subsequent experiments, the above procedure can also be regarded as a practical calibration example for configuring INO. Rather than claiming $m _ { k } = 1 4$ as a universal seed budget, our goal is to demonstrate a reproducible procedure for selecting the seedsearch budget under a given deployment setting. In practice, the appropriate seed budget may depend on the target image domain, the adopted generative model, the similarity metric, and the available transmitter-side computational budget. A system designer can follow the same procedure used in this paper: sample candidate seeds under the target deployment setting, characterize the resulting reconstruction-quality distribution, and select $m _ { k }$ according to the desired confidence level and quality threshold.

![](images/0d6e0d3d17c334ea4c964bb843c1325fc74195948b0c595af6806c138fdad42a.jpg)

![](images/2e06cdb21391fe44351cdcf564d965729c7e7aeeb1d85a966f102421dce37cd4.jpg)

![](images/4761b5d6945cca9be21eca8198b016b27149aef685960451d307d9c08bd78705.jpg)  
Fig. 10. Empirical distribution of comprehensive similarity scores induced by random initial noise seeds. The three subfigures correspond to COCO image IDs 99, 93, and 71, respectively. For each image, we fix the semantic description, generation model, and inference configuration, and generate 500 candidate reconstructions using 500 different initial noise seeds. The comprehensive similarity score of each candidate is computed according to Eq. (6). The blue histogram shows the empirical distribution of the 500 comprehensive similarity scores, where the y-axis denotes probability density rather than sample count, computed as the bin count divided by the product of the total number of samples and the bin width, so that the total area of the histogram is normalized to one. The x-axis represents the comprehensive similarity score. The red curve shows the Gaussian distribution fitted using the empirical mean and standard deviation of the comprehensive similarity scores.

Experiment 2: Validation of the Weighting Parameter α After determining the candidate seed number as $m = 1 4$   
in Experiment 1, we further study how to set the weighting   
parameter α between visual similarity and semantic similarity   
in $S _ { \mathrm { c o m p } }$ . Since candidate selection is directly determined by   
$S _ { \mathrm { c o m p } } ,$ , an inappropriate α may bias the selection toward either   
visually similar but semantically incomplete reconstructions or   
semantically similar but visually less faithful reconstructions.

To investigate this effect, we consider five weighting configurations for $S _ { \mathrm { c o m p } }$ . For each original image, all configurations select one reconstruction from the same candidate set, and they differ only in the value of α:

• Configuration $\mathrm { A } \colon \alpha = 1$ , using visual similarity only;

• Configuration B: $\alpha = 0 . 8 ;$

• Configuration ${ \mathrm { C } } \colon \alpha = 0 . 5$ , assigning equal weights to visual and semantic similarities;

• Configuration D: $\alpha = 0 . 2 ;$

• Configuration E: $\alpha = 0 ,$ , using semantic similarity only.

Here, Configurations A and E correspond to visual-only and semantic-only selection, respectively. Configuration C assigns equal weights to the two similarity terms, while Configurations B and D place more emphasis on visual similarity and semantic similarity, respectively. By comparing the reconstructions selected under these configurations, we can analyze the individual and combined effects of visual and semantic similarities on candidate reconstruction selection.

We next describe how different weighting configurations are evaluated on a test dataset. We first introduce the notation used. Given a test dataset $\mathcal { T } = \{ I _ { r } \} _ { r = 1 } ^ { N _ { \mathrm { t e s t } } }$ <sup>t</sup> , for the r-th original image $I _ { r } ,$ , INO first generates a candidate reconstruction set using $m = 1 4$ different random seeds, denoted by $\Omega = \{ \hat { I } _ { r , q } \} _ { q = 1 } ^ { m }$ Let $L = \{ A , B , C , D , E \}$ denote the set of five weighting configurations. For a given configuration $l \in L ,$ , we compute $S _ { \mathrm { c o m p } }$ for each reconstruction in the candidate set using the corresponding value of $\alpha ,$ , and select the reconstruction with the highest score as the output of this configuration:

$$
\hat { I } _ { r } ^ { ( l ) } = \arg \operatorname* { m a x } _ { \hat { I } _ { r , q } \in \Omega } S _ { \mathrm { c o m p } } ^ { ( l ) } ( I _ { r } , \hat { I } _ { r , q } ) ,\tag{17}
$$

where $S _ { \mathrm { c o m p } } ^ { ( l ) }$ denotes the comprehensive similarity score computed under the value of α associated with configuration l. Therefore, for each original image $I _ { r } ,$ , the five weighting configurations produce five configuration outputs, denoted by $\{ \hat { I } _ { r } ^ { ( l ) } \} _ { l \in L }$

To determine which weighting configuration is more suitable, we introduce an external evaluator (i.e., either a human annotator or a human-validated MLLM). The evaluator selects, from the five configuration outputs $\{ \hat { I } _ { r } ^ { ( l ) } \} _ { l \in L }$ , the reconstruction that is most similar to the original image by jointly considering visual appearance and semantic content. Let $\hat { \bar { I } } _ { r } ^ { * }$ denote the reconstruction selected by the external evaluator for the r-th original image. Since different weighting configurations may select the same reconstruction from the same candidate pool Ω, a configuration is considered to have selected the best reconstruction for the r-th sample if its output satisfies $\hat { I } _ { r } ^ { ( l ) } = \hat { I } _ { r } ^ { * }$

For a given weighting configuration l, its best-reconstruction selection rate over the test dataset is defined as

$$
R _ { l } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { r = 1 } ^ { N _ { \mathrm { t e s t } } } \mathbb { I } \left( \hat { I } _ { r } ^ { ( l ) } = \hat { I } _ { r } ^ { * } \right) ,\tag{18}
$$

where I(·) is the indicator function. $R _ { l }$ represents the proportion of test samples for which configuration l successfully selects the reconstruction considered closest to the original image by the external evaluator. Therefore, $R _ { l }$ is used as the dataset-level metric for comparing different weighting configurations in INO. The remaining issue is how to determine a suitable external evaluator for obtaining $\hat { I } _ { r } ^ { * }$

Human evaluation can directly reflect human perceptual preferences and can therefore serve as the external evaluator described above. However, since this paper needs to compare different weighting configurations over the full test sets of multiple datasets, relying entirely on human annotation would incur a high annotation cost. Therefore, we employ GPT-4o as the MLLM-based external evaluator to evaluate different weighting configurations at the dataset scale. To ensure that the MLLM judgments can reasonably reflect human preferences, we first construct a small-scale human validation dataset before applying the MLLM to the full test datasets.

![](images/422083a607ffca21e9f3d0abcd244a620e2b3f259820c7641118011266a0e77d.jpg)  
Fig. 11. MLLM-guided evaluation of different weighting configurations.

Specifically, we randomly sample 20 original images from each dataset, resulting in 100 original images in total. For each original image $I _ { r }$ in the validation set, we obtain five configuration outputs $\{ \hat { I } _ { r } ^ { ( l ) } \} _ { l \in L }$ following the procedure described above. Both human evaluation and MLLM-based evaluation are performed on these five outputs, where the evaluator selects the reconstruction that is overall closest to the original image.

Note that, during this evaluation stage, revealing the weighting configuration, the value of $\alpha ,$ the associated similarity score, or the method name for a given output may introduce subjective bias unrelated to the image content. To avoid this potential bias, we use an anonymized candidate presentation protocol for both human and MLLM-based evaluations. Specifically, the five configuration outputs are presented to the evaluator using only anonymous labels A, B, C, D, and E, without revealing their corresponding weighting configurations, α values, $S _ { \mathrm { c o m p } }$ scores, or method names.

Based on this anonymized presentation protocol, in the human evaluation stage, for each original image $I _ { r } ,$ , a human annotator is asked to select, from $\{ \hat { I } _ { r } ^ { ( \tilde { l } ) } \} _ { l \in L }$ , the reconstruction that is most similar to the original image by considering both semantic content and visual appearance. The human-selected reconstruction is denoted by $\hat { I } _ { r } ^ { * , ( \mathrm { h u m a n } ) }$

For MLLM-based annotation, each original image and its five corresponding reconstructed candidates are jointly provided to the MLLM. The MLLM then selects the candidate that is most similar to the original image. The prompt used for this evaluation is as follows:

You are given one reference image and five reconstructed candidate images. Please select the reconstructed candidate that is most similar to the reference image. When making the selection, consider both visual appearance and semantic content in an overall manner. You must choose exactly one candidate from A, B, C, D, and E. Return only a JSON object in the following format:

$$
\left\{ { \ " } \mathrm { w i n n e r } { \ " } : \quad { \ " } \mathbb { A } ^ { \prime } \right\}
$$

This prompt asks the MLLM to evaluate the reconstructed candidates from both visual and semantic perspectives. To facilitate automatic parsing, we require the MLLM to return only a JSON object containing the field winner. The MLLMselected reconstruction is denoted by $\hat { I } _ { r } ^ { * , ( \mathrm { M L L M } ) }$

Based on the above, the human-MLLM agreement rate on the validation set is defined as

$$
A _ { \mathrm { a g r e e } } = \frac { 1 } { N _ { \mathrm { v a l } } } \sum _ { r = 1 } ^ { N _ { \mathrm { v a l } } } \mathbb { I } \left( \hat { I } _ { r } ^ { * , ( \mathrm { M L L M } ) } = \hat { I } _ { r } ^ { * , ( \mathrm { h u m a n } ) } \right) ,\tag{19}
$$

where $N _ { \mathrm { v a l } } = 1 0 0$ denotes the number of original images in the validation set, and $\mathbb { I } ( \cdot )$ is the indicator function. In our validation experiment, the MLLM achieves a 93% agreement rate with human selections, indicating that its judgments are well aligned with human perceptual preferences in this evaluation setting.

Therefore, we further adopt the MLLM as the external evaluator to assess different weighting configurations on the full test datasets. Based on the MLLM selections, we compute the best-reconstruction selection rate $R _ { l }$ for each weighting configuration l on each tested dataset (see Eq. (18) for more details).

As shown in Fig. 11, Configuration C achieves the highest best-reconstruction selection rate across all datasets. This result indicates that assigning equal weights to visual similarity and semantic similarity provides a better balance between perceptual consistency and semantic faithfulness. Therefore, we set $\alpha = 0 . 5$ in $S _ { \mathrm { c o m p } }$ as the default weighting configuration for INO.

## C. End-to-End Robustness Analysis in Wireless Channels

In the preceding experiments, we mainly evaluated TISC as a text-driven semantic source coding and decoding framework under the reliable-delivery assumption. To further examine its robustness over wireless channels, we extend TISC into an end-to-end wireless communication system. Specifically, the text semantic description generated at the transmitter and the noise seed selected by INO are first converted into a bit stream using UTF-8 encoding. The bit stream is then protected by an LDPC channel code with rate $3 / 8$ , modulated using BPSK, and transmitted over an AWGN channel. At the receiver, the received signal is demodulated and channel-decoded to recover the text semantic information and the noise seed, which are then used for image reconstruction. Under this setting, we consider AWGN channels with different SNRs to evaluate the end-to-end noise robustness of TISC.

To compare TISC with conventional image transmission and existing semantic communication schemes over wireless channels, we introduce the following baselines:

JPEG baseline: JPEG is a commonly used lossy image source coding standard [46]. In this baseline, each image is first compressed into a JPEG bitstream using a JPEG encoder with quality factor $Q = 1 ^ { 2 }$ . The resulting JPEG bitstream is then protected by the same LDPC channel code with rate $3 / 8 ,$ , modulated using BPSK, and transmitted over the same AWGN channel. At the receiver, if a valid image cannot be reconstructed after standard JPEG decoding and image recovery procedures, the sample is treated as a reconstruction failure. For such failed samples, a zero image similarity score is assigned.

![](images/7d447f37effb5db2a1d5b778c8e7b21fe395cf7599da38dab0a881c474546ac1.jpg)  
Fig. 12. End-to-end noise robustness comparison over an AWGN channel. To characterize the channel resource usage of different schemes, we report the average number of channel symbols used per image. For TISC and the JPEG baseline, this number is counted from the output symbols after BPSK modulation. For DeepJSCC, since its end-to-end model implicitly includes source coding, channel coding, and modulation-related operations, we directly count the number of channel symbols produced by its encoder. Under this experimental setting, TISC, the JPEG baseline, and DeepJSCC use approximately 12.6k, 72.2k, and 16.4k channel symbols per image on average, respectively. This indicates that the performance advantage of TISC is not obtained by using more channel resources.

DeepJSCC: DeepJSCC is an end-to-end joint sourcechannel coding framework for image semantic transmission [14]. It uses neural networks to directly map the input image into continuous channel symbols for transmission and reconstructs the image from the received symbols at the receiver. Therefore, the source coding, channel coding, and modulationrelated operations are implicitly modeled within the end-to-end learned system.

The experiment is conducted on 100 images randomly sampled from the COCO dataset. For each method, we compute the average comprehensive similarity score (between the transmitted image and the recovered one) over this dataset under different SNR conditions, where the score is defined in Eq. 6.

As shown in Fig. 12, TISC achieves higher comprehensive similarity scores than both baselines across all SNR values tested. In addition to the superior performance, TISC uses fewer channel symbols than DeepJSCC and substantially fewer channel symbols than the JPEG baseline (see the caption of Fig. 12 for statistic details).

At the extremely low SNR of 0 dB, 32 out of the 100 image transmissions cannot fully recover the transmitted text. Yet, our TISC receiver can still generate a highly related image based on the partly corrupted semantic information in text (with an average similarity score of 0.5149), while DeepJSCC and the standard JPEG baseline perform worse in such conditions. This result indicates that the text semantic representation in TISC provides a certain degree of error tolerance: even when part of the received text is corrupted, the receiver may still exploit the remaining semantic information to generate an image that is semantically related to the original one.

When the SNR increases to 2 dB, the text semantic information of only 4 out of 100 images is not exactly recovered, and TISC achieves an average similarity score of 0.5708, which is already close to its error-free performance of 0.5729. From 4 dB onward, both the text semantic information and the selected noise seed are fully recovered for all test images, and thus the TISC performance remains stable as the SNR further increases. In contrast, the JPEG bitstream is more sensitive to residual bit errors and may fail to decode under low-SNR conditions. DeepJSCC exhibits a smoother performance variation with SNR, but its reconstruction similarity remains lower than that of TISC in this setting.

## V. CONCLUSION

This paper proposed TISC, a text-driven image semantic communication framework for semantically faithful image reconstruction. Two major techniques, namely Tree-Structured Attribute Semantic Extraction (TSASE) and Initial Noise Optimization (INO), have been developed in our TISC framework: TSASE extracts a structured semantic description to reduce information loss and distortion, while INO selects the most favorable initial noise seed at the transmitter to improve the visual and semantic consistency of receiver-side reconstruction. We have conducted massive experiments to validate the effectiveness of the two techniques proposed. Our results on multiple datasets show that TSASE substantially improves both object-position recovery and semantic description faithfulness over the without-TSASE ablation. Specifically, across all datasets, TISC improves object-position alignment by at least 42 percentage points and achieves text-image faithfulness scores above 0.68, compared with only 0.33–0.39 for this ablation. Furthermore, parameter study on INO determines reasonable settings for the number of candidate noises and the weighting between visual and semantic similarities in the comprehensive similarity score, providing practical guidance for configuring INO in the proposed system. These results validate the effectiveness and practical feasibility of the proposed framework for deployment in future wireless networks.

## APPENDIX A

## EXPERIMENTAL EVIDENCE SUPPORTING THE MOTIVATION

This appendix provides a motivating experiment to illustrate the bandwidth-saving potential of text-driven image semantic communication. We compare JPEG-based image transmission with the proposed text-driven semantic transmission scheme, TISC, where JPEG [46] is a classical lossy image compression standard. The comparison is conducted in terms of the number of compressed bits to be transmitted and the comprehensive similarity score between the original and reconstructed images. The comprehensive similarity score jointly measures visual and semantic similarities between the two images (see Eq. 6 for more details).

The experiment is conducted on 100 images sampled from the COCO dataset. For each test image, we apply both JPEG and TISC to obtain the corresponding reconstructed images, and then compute the comprehensive similarity score between each reconstruction and the original image. For JPEG-based transmission, we vary the JPEG quality factor to obtain reconstructions under different compression levels. For each quality factor, we compute the average number of compressed bits and the average comprehensive similarity score over the 100 images, thereby forming a curve between bit cost and reconstruction similarity. For TISC, the image is first compressed into text semantic information to be transmitted, and the required number of bits is computed from the UTF-8 encoding length of this text semantic information. The average bit cost and average comprehensive similarity score are also computed over the same 100 images. To focus on the compression efficiency of different source representations, we only count the bits after source compression/encoding, without including channel coding overhead, and assume reliable information delivery.

![](images/cc47335edf9652a63d78c8a72411ad65b4d898c747f65377bd9a4b18f2c19fbb.jpg)  
Fig. 13. Comparison between JPEG-based image transmission and the proposed text-driven semantic transmission scheme in terms of compressed bits and the comprehensive similarity score. The x-axis reports the average number of compressed bits per image, and the y-axis reports the dataset-level average comprehensive similarity score computed according to Eq. 6.

As shown in Fig. 13, at the reconstruction similarity level achieved by TISC, JPEG requires more than 20 times as many compressed bits to reach a comparable similarity level. This result provides empirical evidence for the motivation of text-driven image semantic communication: by transmitting compact text semantic information instead of image pixels or dense visual features, the communication system can substantially reduce the transmitted bit cost while preserving semantic similarity in the reconstructed image.

## REFERENCES

[1] M. Agiwal, A. Roy, and N. Saxena, “Next generation 5g wireless networks: A comprehensive survey,” IEEE communications surveys & tutorials, vol. 18, no. 3, pp. 1617–1655, 2016.

[2] C. E. Shannon, “A mathematical theory of communication,” The Bell system technical journal, vol. 27, no. 3, pp. 379–423, 1948.

[3] E. C. Strinati and S. Barbarossa, “6g networks: Beyond shannon towards semantic and goal-oriented communications,” Computer Networks, vol. 190, p. 107930, 2021.

[4] Y. Shao, Q. Cao, and D. Gund ¨ uz, “A theory of semantic communication,”¨ IEEE Trans. Mob. Comput., 2024.

[5] D. Gund ¨ uz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener,¨ K. K. Wong, and C.-B. Chae, “Beyond transmitting bits: Context, semantics, and task-oriented communications,” IEEE J. Sele. Areas Commu., vol. 41, no. 1, pp. 5–41, 2022.

[6] X. Luo, H.-H. Chen, and Q. Guo, “Semantic communications: Overview, open issues, and future research directions,” IEEE Wireless communications, vol. 29, no. 1, pp. 210–219, 2022.

[7] Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, “Semantic communications: Principles and challenges,” arXiv preprint arXiv:2201.01389, 2021.

[8] X. Luo, H.-H. Chen, and Q. Guo, “Semantic communications: Overview, open issues, and future research directions,” IEEE Wireless communications, vol. 29, no. 1, pp. 210–219, 2022.

[9] W. Yang, H. Du, Z. Q. Liew, W. Y. B. Lim, Z. Xiong, D. Niyato, X. Chi, X. Shen, and C. Miao, “Semantic communications for future internet: Fundamentals, applications, and challenges,” IEEE Communications Surveys & Tutorials, vol. 25, no. 1, pp. 213–250, 2022.

[10] D. Huang, F. Gao, X. Tao, Q. Du, and J. Lu, “Toward semantic communications: Deep learning-based image semantic coding,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 1, pp. 55– 71, 2023.

[11] J. Wu, C. Wu, Y. Lin, T. Yoshinaga, L. Zhong, X. Chen, and Y. Ji, “Semantic segmentation-based semantic communication system for image transmission,” Digital Communications and Networks, vol. 10, no. 3, pp. 519–527, 2024.

[12] E. Hosonuma, T. Yamazaki, T. Miyoshi, A. Taya, Y. Nishiyama, and K. Sezaki, “Image generative semantic communication with multi-modal similarity estimation for resource-limited networks,” arXiv preprint arXiv:2404.11280, 2024.

[13] G. Cicchetti, E. Grassucci, J. Park, J. Choi, S. Barbarossa, and D. Comminiello, “Language-oriented semantic latent representation for image transmission,” in 2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2024, pp. 1–6.

[14] E. Bourtsoulatze, D. B. Kurka, and D. Gund ¨ uz, “Deep joint source-¨ channel coding for wireless image transmission,” in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 4774–4778.

[15] J. Dai, S. Wang, K. Tan, Z. Si, X. Qin, K. Niu, and P. Zhang, “Nonlinear transform source-channel coding for semantic communications,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 8, pp. 2300– 2316, 2022.

[16] F. Zhang, Y. Du, Y. Xiang, X. Liu, and S. C. Liew, “Sa-oosc: A multimodal llm-distilled semantic communication framework for enhanced coding efficiency with scenario understanding,” in 2026 International Conference on Computing, Networking and Communications (ICNC). IEEE, 2026, pp. 1–7.

[17] H. Wu, Y. Shao, E. Ozfatura, K. Mikolajczyk, and D. Gund¨ uz,¨ “Transformer-aided wireless image transmission with channel feedback,” IEEE Transactions on Wireless Communications, vol. 23, no. 9, pp. 11 904–11 919, 2024.

[18] K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, and P. Zhang, “Swinjscc: Taming swin transformer for deep joint source-channel coding,” IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 1, pp. 90–104, 2025.

[19] D. B. Kurka and D. Gund¨ uz, “Deepjscc-f: Deep joint source-channel¨ coding of images with feedback,” IEEE Journal on Selected Areas in Information Theory, vol. 1, no. 1, pp. 178–193, 2020.

[20] X. Wang, D. Ye, C. Feng, H. H. Yang, X. Chen, and T. Q. Quek, “Trustworthy image semantic communication with genai: Explainablity, controllability, and efficiency,” IEEE Wireless Communications, vol. 32, no. 2, pp. 68–75, 2025.

[21] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,” National Science Review, vol. 11, no. 12, p. nwae403, 2024.

[22] A. Mahgoub and E. Yaacoub, “Semantic communication of images using image generation and image captioning models,” in Web Information Systems Engineering – WISE 2024 PhD Symposium, Demos and Workshops, ser. Lecture Notes in Computer Science. Springer, 2025, vol. 15463, pp. 138–147.

[23] H. Nam, J. Park, J. Choi, M. Bennis, and S.-L. Kim, “Language-oriented communication with semantic coding and knowledge distillation for text-to-image generation,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 13 506–13 510.

[24] S. Ribouh and O. Saleem, “Large language model-based semantic communication system for image transmission,” arXiv preprint arXiv:2501.12988, 2025.

[25] F. Zhang, Y. Du, K. Chen, Y. Shao, and S. C. Liew, “Out-of-distribution in image semantic communication: A solution with multimodal large language models,” IEEE Transactions on Machine Learning in Communications and Networking, pp. 997–1013, 2025.

[26] H. Liu, W. Xue, Y. Chen, D. Chen, X. Zhao, K. Wang, L. Hou, R. Li, and W. Peng, “A survey on hallucination in large vision-language models,” arXiv preprint arXiv:2402.00253, 2024.

[27] Z. Bai, P. Wang, T. Xiao, T. He, Z. Han, Z. Zhang, and M. Z. Shou, “Hallucination of multimodal large language models: A survey,” arXiv preprint arXiv:2404.18930, 2024.

[28] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[29] D. Ahn, J. Kang, S. Lee, J. Min, M. Kim, W. Jang, H. Cho, S. Paul, S. Kim, E. Cha et al., “A noise is worth diffusion guidance,” arXiv preprint arXiv:2412.03895, 2024.

[30] L. Lian, B. Li, A. Yala, and T. Darrell, “Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models,” arXiv preprint arXiv:2305.13655, 2023.

[31] W. Eddy, “Transmission control protocol (tcp),” Internet Engineering Task Force, Tech. Rep., 2022.

[32] Ultralytics, “Yolov11x,” https://github.com/ultralytics/ultralytics, 2024, accessed: 2024-08-27.

[33] OpenAI, “Gpt-4o system card,” https://openai.com/index/ gpt-4o-system-card/, 2024, accessed: 2026-07-27.

[34] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[35] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.

[36] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in 3rd international conference on learning representations (ICLR 2015). Computational and Biological Learning Society, 2015.

[37] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[38] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” International journal of computer vision, vol. 88, no. 2, pp. 303–338, 2010.

[39] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in European conference on computer vision. Springer, 2014, pp. 740–755.

[40] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma et al., “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” International journal of computer vision, vol. 123, no. 1, pp. 32–73, 2017.

[41] B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2641– 2649.

[42] A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov et al., “The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale,” arXiv preprint arXiv:1811.00982, 2018.

[43] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein et al., “Imagenet large scale visual recognition challenge,” International journal of computer vision, vol. 115, no. 3, pp. 211–252, 2015.

[44] Y. Hu, B. Liu, J. Kasai, Y. Wang, M. Ostendorf, R. Krishna, and N. A. Smith, “Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20 406–20 417.

[45] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[46] G. K. Wallace, “The jpeg still picture compression standard,” Communications of the ACM, vol. 34, no. 4, pp. 30–44, 1991.

[47] Pillow Contributors, “Pillow Image File Formats: JPEG,” https:// pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg, accessed: Aug. 14, 2026.