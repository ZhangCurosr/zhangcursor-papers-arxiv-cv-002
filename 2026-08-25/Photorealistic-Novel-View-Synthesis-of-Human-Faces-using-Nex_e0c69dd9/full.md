# Photorealistic Novel View Synthesis of Human Faces using Next-Scale Transformers

Federico Stella<sup>1⋆</sup>, Fei Jiang<sup>2</sup>, Zhongshi Jiang<sup>2</sup>, Zohar Barzelay<sup>2</sup>, Emanuel Garbin<sup>2</sup>, Amin Jourabloo<sup>2</sup>, and Liuhao Ge<sup>2</sup>

<sup>1</sup> École polytechnique fédérale de Lausanne, Switzerland federico.stella@epfl.ch <sup>2</sup> Meta

Abstract. Photorealistic novel view synthesis of people remains challenging at high spatial resolutions and across multiple target cameras, where preserving identity, fine appearance details, and geometric coherence is critical. We build on the next-scale autoregressive paradigm and adapt it for human-centric view synthesis by enabling higher image resolutions, multi-view outputs and stronger cross-view consistency in a single forward pass. We train on a synthetic dataset of human faces spanning diverse identities and apparel. Contrary to difusion models, this paradigm does not need 2D pre-training and, thanks to its nextscale architecture, it benefits from lower-resolution, general-purpose pretrainings, with the full-sized purpose-specific images being used only in the last training stages. This enables our architecture to converge with a smaller amount of purpose-specific training data, allowing us to use a smaller but more realistic training dataset. The resulting model produces sharp and realistic views, with the option to synthesize multiple novel viewpoints simultaneously for improved agreement across views. Empirically, we observe gains in perceptual fidelity and cross-view coherence on human subjects, demonstrating that next-scale autoregression is an efective backbone for scalable, multi-output human view synthesis. We also couple our pipeline with an existing transformer-based model for pixel-aligned 3D gaussian lifting from multi-view facial inputs, resulting in accurate and photorealistic 3D models of human faces.

## 1 Introduction

Synthesizing photorealistic novel views of human faces and bodies remains challenging at high resolution and across wide viewpoint changes, where methods must preserve identity while maintaining geometric, shading, and appearance consistency. Recent approaches tackle complementary subproblems but exhibit trade-ofs: near-frontal pipelines (e.g., Splatter Image [41] and VoluMe [20]) can run in real time and emphasize quality around canonical views but struggle as viewpoint departs from frontal, while large-scale, heavily pre-trained image-to-3D systems (e.g., FaceLift [24]) achieve strong detail yet often produce artificiallooking renderings and rely on extensive synthetic data with varied lighting and expressions. We pursue a diferent path: extending a next-scale autoregressive transformers (VAR [43]) for human-centric view synthesis that operates under data scarcity and does not necessitate of a large 2D image-generation pretraining or extensive synthetic data. Our system builds on the novel-view synthesis architecture introduced in ArchonView [48] and extends it to support higher resolutions (512 × 512) and multiple simultaneous input and output views, converging with minimal purpose-specific realistic training data and achieving higher accuracy and detail in a single forward pass. While transformer-based difusion models can naturally provide multi-view attention over multiple passes, the same is not true for autoregressive models, which operate on a single pass. Our next-scale autoregressive architecture, instead, allows tokens of diferent scales to attend to previous scales of every output view, achieving cross-view attention in a single forward pass, with faster speeds compared to difusion models. This design prioritizes realism, especially on of-frontal views, while enforcing identity preservation and geometric coherence across simultaneously synthesized cameras. In head-to-head comparisons on our validation set and a subset of images curated by prior work, our approach delivers more realistic results than competing methods, which can fail to converge correctly when training with naturally scarce realistic data, highlighting the importance of our architectural choices and training scheme. As pre-trained next-scale autoregressive backbones such as [13, 22] become more available, akin to difusion models, our approach ofers a practical foundation for controllable, consistent multi-camera rendering of people that can capitalize on stronger initializations without sacrificing its core strength: convergence and realism in low-data regimes.

![](images/36734c7c5c1bdd00c3eb89e4466b9de80acbd63c363320d64a4fe56c5df7cd1a.jpg)  
Fig. 1: Face reconstruction from single view: our method achieves more realistic results than competing methods. <sup>†</sup>Variant of our method trained to handle multiple looselyposed inputs, it achieves even better results thanks to the additional inputs.

To summarize our contributions:

– We extend the next-scale autoregressive transformer architecture for multiview human face novel view synthesis, supporting multiple simultaneous inputs and outputs;

– We introduce a training strategy that enables the model to converge at high resolution (512×512) without the need to re-train the model from scratch, thus lowering the amount of needed compute and data;

– We couple our pipeline with an existing transformer-based model for pixelaligned 3D gaussian lifting [11], and we benchmark the results against stateof-the-art models.

Finally, we showcase the flexibility of our method by applying it to human body reconstruction without any architectural changes in the supplementary material, showing a potential direction for future work.

## 2 Related works

Generative models and novel view synthesis. Difusion models have shown impressive results in generating high-quality images [14,26, 36]. Autoregressive approaches have been used for image generation as well [3, 9, 32, 33, 47], but they have been largely outperformed by difusion models in terms of image quality and diversity. Recently, the next-scale transformer architecture has been proposed for high-quality image generation [43], changing the traditional raster-scan order of autoregressive models to a scale-based order, which allows for better modeling and speed, outperforming difusion models [43] and originating a new research direction. It has been developed and applied to multiple tasks [4, 30, 34, 35, 42, 51], including novel view synthesis of objects [48], demonstrating superior performance compared to similarly-trained difusion models such as Zero 1-to-3 [23], Zero 123-XL [6] and EscherNet [18], without the need for large 2D image pretraining. However, the need for large 3D datasets, generally unavailable for human faces and bodies, has limited its application to these domains. We propose a training strategy that maximizes the use of available data and models, unlocking the use of this powerful architecture for human face novel view synthesis. Concurrent works such as [13, 22] have further improved the next-scale transformer architecture, so they represent a promising direction for future work to further improve the results of our method.

Face reconstruction. Face reconstruction has been a long-standing problem in computer vision, with applications in various fields such as virtual reality, gaming, and human-computer interaction. Traditional methods for face reconstruction include 3D morphable models (3DMM) [45], which use a parametric model to represent the shape and appearance of faces. More recently, deep learningbased approaches have been proposed for face reconstruction, such as GANs [12, 15, 16]. These methods have shown promising results, but they often struggle with consistency in non-frontal views. Subsequent works such as PanoHead [1], Rodin [46] and RodinHD [49] greatly improved the consistency of novel views, but still struggle with realism. Methods such as [2,17,38] have shown high quality results, but may require a lengthy or inaccessible capture process [20], sometimes also requiring posed images, which limit their applicability. FaceLift [24] focuses on a simpler capture scenario: it employs a two-stage approach for face reconstruction from a single frontal view, first generating 6 canonical views using a difusion model, and then using a GS-LRM [50] to lift these views to a full 3D representation. However, the use of a difusion model requires large-scale 2D image pre-training as well as suficiently large dataset for face reconstruction which, given the lack of suficiently large 3D datasets of human faces, is synthetically generated. This can limit the realism of the method. In contrast, our method converges using a relatively small dataset of 3D face scans and achieves increased realism, while also allowing the use of multiple conditioning views to further improve the consistency and quality of the generated novel views.

A diferent approach to face reconstruction is proposed by VoluMe [20], which is based on Splatter Image [41], a fast 2D U-Net architecture that generates a volumetric representation of the face from a single image at real-time speed. This is particularly suitable for near-frontal views, as the method predicts the 3D representation directly from the input image, and it finds applications in video conferencing scenarios, but it struggles with detail and realism in non-frontal views. In contrast, our method is suitable for generating high-quality views from a wide range of angles, including non-frontal views.

Finally, a diferent line of works focuses on reconstructing a full body model from one or more images, such as LHM [28], IDOL [53] and PF-LHM [29]. We do not focus on full-body reconstruction in this work, but we qualitatively show that our method can be applied to this domain without any architectural changes, showing the flexibility of the method and a potential direction for future work.

## 3 Method

Similarly to FaceLift [24], we approach the problem of modeling photorealistic human head avatars by decomposing it in two stages. In the first stage, we generate 6 specific “canonical” views of the head from an input conditioning, at angles 0, 45, 90, 180, 270, 315 degrees. In the second stage, we use an existing model to reconstruct the full 3D head from these views. The pipeline does not need the camera poses of the input images, since the only poses needed throughout the process are the fixed canonical ones. Contrary to FaceLift, which works only with one input view, our pipeline also accepts three loosely posed input images (front and side views), which provide more complete information about the subject to be reconstructed. The first stage of our pipeline is based on a novel next-scale transformer architecture named VAR [43], in its novel-view synthesis variant ArchonView [48]. We introduce the basics of this architecture below, and then describe how we adapt it to our task, enabling it to handle multiple input views and generate multiple output views simultaneously. For the second stage, we use an existing GS-LRM named MV2Splat [11].

## 3.1 Image encoding

In VAR transformers, images are encoded using a residual [21] multi-scale [43] variant of the original VQ-VGAN [9] architecture. The VQ-VAE encoder encodes an image $\mathbf { I } \in \mathbb { R } ^ { h \times w \times 3 }$ into a set of discrete latent feature maps $\{ \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } , \dotsc , \mathbf { Z } _ { K } \}$ at K diferent scales, where each $\mathbf { Z } _ { k } \in \mathbb { R } ^ { h _ { k } \times w _ { k } \times C }$ is a 2D grid of discrete tokens obtained by quantizing continuous latent features with a learned codebook, with C the embedding dimension. The scales are arranged such that $\left( h _ { 1 } , w _ { 1 } \right) < \left( h _ { 2 } , w _ { 2 } \right) < \ldots < \left( h _ { K } , w _ { K } \right)$ , allowing the model to capture both finegrained details and global context. The codebook has size V, it is learned during training, and it is shared across all scales. The VQ-VAE decoder reconstructs the image from the multi-scale latent feature maps by progressively upsampling and decoding them back to the original resolution. The size of the final scale $h _ { K } \times w _ { K }$ and a downsampling factor control the size of the output image. The auto-encoder is trained with a compound loss which includes a perceptual and a discriminative loss, and it is frozen after the initial training.

![](images/8d12dcadfd388f6f4d80f8038e1001f38ba8064d0f09591e197816cb45d0a227.jpg)  
Fig. 2: Our architecture: input images are encoded into multi-scale features by the VQ-VAE encoder and used as inputs for the transformer. They are also embedded into a global conditioning along with the camera-pose RT matrix, used both in the AdaLN layers and the SOS tokens. The transformer outputs one scale at a time for every output view simultaneously, up to the final scale. The output features are decoded by the VQ-VAE decoder to produce the final RGB images, which can be lifted to 3D gaussians by an of-the-shelf GS-LRM model. During training, previous scales are teacher-forced, while during inference the model is auto-regressive. The architecture is the same for all our models, with the only diference being the number of input and output views.

![](images/a2c15969d5467717ae59f84a3573ae2c6f17964e2eba11a75447c3a3cf317b67.jpg)  
Fig. 3: Multi-scale VQ-VAE outputs for diferent resolutions and scale configurations. Notice that the small details appear at the last 2-3 scales, regardless of the resolution and the number of scales.

## 3.2 Next-scale transformer for multi-view synthesis

In the VAR architecture the goal is to shift the next-token paradigm of traditional transformers [44] for image generation [9], which linearizes images through raster-scanning and generates them by predicting one token at a time, to a setting where the model generates multiple tokens at once, corresponding to an entire image scale. This is done multiple times, each corresponding to a diferent image scale, in order from lowest to highest detail. In its novel-view synthesis formulation, named ArchonView [48], given an image conditioning x and a target camera transformation $( R , T )$ , the autoregressive likelihood of generating the set of multi-scale latent feature maps $\{ r _ { 1 } , r _ { 2 } , \dots , r _ { K } \}$ is modeled as:

$$
g ( x , R , T ) = p _ { \theta } ( r _ { 1 } , r _ { 2 } , \ldots , r _ { K } | x , R , T ) = \prod _ { k = 1 } ^ { K } p _ { \theta } ( r _ { k } | r _ { < k } , x , R , T )\tag{1}
$$

where $r _ { < k } = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { k - 1 } \}$ are the previously generated scales, and $\theta$ are the learnable parameters of the transformer. The generation process starts from the lowest resolution scale $r _ { 1 }$ and progressively generates higher resolution scales $r _ { 2 } , r _ { 3 } , \ldots , r _ { K }$ , conditioning each scale on the previously generated ones. The final tokens are predicted through a linear classification head as logits over the codebook entries.

We extend this framework to handle multiple input views and output views simultaneusly, as shown in Fig. 2. Given N conditioning images $X = \{ x _ { 1 } , x _ { 2 } \}$ $\dots , x _ { N } \}$ and M target camera transformations with respect to the first image x<sub>1</sub> $( R , T ) \ = \ \{ ( R _ { 1 } , T _ { 1 } ) , ( R _ { 2 } , T _ { 2 } ) , \ldots , ( R _ { M } , T _ { M } ) \}$ , one per output view, the autoregressive likelihood of generating the set of multi-scale latent feature maps {r<sub>1,1</sub>, $r _ { 1 , 2 } , \ldots , r _ { 1 , M } , \ldots , r _ { K , M } \}$ of M output images is modeled as:

$$
\begin{array} { l } { { g ( X , R , T ) = p _ { \theta } ( r _ { 1 , 1 } , r _ { 1 , 2 } , \dots , r _ { 1 , M } , \dots , r _ { K , M } | X , R , T ) \ ~ } } \\ { { \displaystyle ~ = \prod _ { k = 1 } ^ { K } p _ { \theta } ( r _ { k , 1 } , \dots , r _ { k , M } | r _ { < k , 1 } , \dots , r _ { < k , M } , X , R , T ) \ ~ } } \end{array}\tag{2}
$$

Notice that this formulation generates all output views at each scale before proceeding to the next scale, allowing the model to capture correlations between diferent views at multiple levels of detail, with the attention mask adjusted accordingly to allow cross-view attention within each scale, and complete availability of previous scales across all views.

## 3.3 Input conditioning

The input conditioning comprises both a global and a local conditioning. In [48], for a single input image x the global conditioning is represented by a posed CLIP [31] embedding [23]:

$$
\tau ( x , R , T ) = W ( W _ { i } ( \mathrm { C L I P } ( x ) \oplus [ \theta , s i n \phi , c o s \phi , r ] ) + b _ { i } ) + b\tag{3}
$$

where $W _ { i } , b _ { i } , W ,$ b are learnable parameters of two linear layers, ⊕ is the concatenation operator, and $( \theta , \phi , r )$ are the spherical coordinates of the target camera transformation (R, T). The first linear layer embeds the camera transformation into the CLIP features, while the second layer maps the resulting vector to the transformer embedding dimension, which is set as $w = 6 4 d$ , with d the number of transformer blocks. The global conditioning is used both as the SOS token and in the AdaLN [27] layers inside transformer blocks. As in [48] and contrary to [43], we empirically decide to avoid the use of AdaLN before the final classification head. To extend this to multiple output views, we compute a diferent SOS token for each view, using its view-specific $( R _ { m } , T _ { m } )$ while the AdaLN layers share the same global conditioning computed from the first output pose. For multiple input views, the global conditioning is computed only from the first one, which is considered the main view (near-frontal) and is the one used to define the target camera transformations.

The local conditioning, instead, consists of the multi-scale latent feature maps of the input image x $\{ i _ { 1 } , i _ { 2 } , \ldots , i _ { K } \}$ , obtained by encoding them with the frozen VQ-VAE encoder. When dealing with multiple input views $X = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ ， we concatenate their features together, obtaining $\{ i _ { 1 , 1 } , i _ { 2 , 1 } , . . . , i _ { K , 1 } , . . . , i _ { K , N } \}$

## 3.4 Training strategy

Training a VAR architecture from scratch requires a large amount of data and computational resources. In [48], the model is trained on the large-scale Objaverse [7] dataset, which contains around 800k 3D objects from various categories, and it requires several days of training on hundreds of GPUs, at resolution $2 5 6 \times 2 5 6$ pixels. Training such a model specifically for photorealistic human heads is challenging both due to the lack of large-scale datasets and the need to generate higher-resolution images to capture details. Training a model from scratch at resolution 512×512 pixels would require even more data and computational resources, making it impractical. Fine-tuning a VAR model pre-trained at a diferent resolution or for a diferent purpose is also not straighforward, as the available high-quality human head data is not suficient for the adaptation, as shown in Tab. 4. To overcome these challenges, we propose a three-stage training strategy, which allows us to efectively adapt a pre-trained VAR-derived model to our specific task by increasing its resolution and supporting multiple views.

First stage. We adapt the pre-trained ArchonView [48] model, which is trained for novel-view synthesis on objects, to work at a higher resolution of 512×512 pixels. To do this we can either modify the architecture to support additional feature scales, corresponding to increasingily higher resolutions, or we can keep the same number of scales and increase the size of each scale instead. Increasing the number of scales makes intuitive sense, as it could leverage a model that already works at 256 by upscaling it at 512 using a few additional scales, but we argue that this is not optimal. In Fig. 3 we show the scales of the VQ-VAE encoder for a sample image at resolution 256×256 pixels (first row) and at resolution 512×512 pixels using the same number of scales (second row) or using additional scales (third row). We observe that in every case the details appear progressively, but most of the facial features appear in the last two scales. This suggests that adding more scales would require a substantial change in the behavior of the transformer, which has learned to converge with a set number of scales. Therefore, we decide to keep the same number of scales and increase the size of each scale instead. In this first stage we train the model for generalpurpose novel-view synthesis by using a proprietary large scale object dataset (SS3D, see Sec. 4.1), selecting random input and target views. We validate this approach and the scale choice in Tab. 4.

Second stage. In this stage, we fine-tune the model to specialize it for photorealistic human head synthesis from frontal input views. We use the PSGS [11] dataset, with around 2.9K training subjects, and we train the model to generate random turntable views. We specialize two separate models, one that deals with a single frontal input view, and one that deals with two additional side views at approximately ±90 degrees yaw. These additional views are loosely posed: their ground truths are generated with 32 camera and head pose variations per subject, mimicking real-world selfie captures (see Fig. 2). In this stage, we also fine-tune the VQ-VAE weights from VAR [43], which were trained on the Open-Images [19] dataset at a resolution of 256 × 256 pixels. To adapt them to our task, we finetune them on the PSGS [11] dataset of human heads at resolution 512×512. Notice that, at this stage, the models are not constrained to output canonical views, and can output any turntable view.

Third stage. Finally, we further specialize the models to symultaneously predict the 6 canonical views needed for the subsequent 3D reconstruction stage, using the same data as in the second stage. This allows the model to better capture correlations between the diferent views, improving the overall quality and consistency of the generated images. To achieve this, we empirically determined that classifier-free guidance is not beneficial (see Fig. 7, right), so we do not employ it in this training stage. We conjecture that the limited amount of training data, alongside with the model being specialized from a general-purpose one, contribute to this behavior.

## 4 Experiments

## 4.1 Datasets

SS3D. A proprietary large scale object dataset, containing around 2M unique textured objects. They have been rendered at resolution of 512x512 from 24 random cameras sampled on a sphere and using area lights.

PSGS [11]. A proprietary dataset obtained by capturing calibrated multi-view images of real individuals from a dome capture setup, and fitting a high-fidelity gaussian avatar to each of them, which can then be rendered with arbitrary camera extrinsics and intrinsics with head pose variations. The dataset is more realistic but smaller than existing synthetic datasets [20, 24], as it consists of 3.2K subjects, of which 288 are used for validation. We render the subjects from 32 turntable views at a fixed distance. We additionally render 32 variants per subject, each with a randomly perturbed camera pose and head pose, to use as loosely-posed training inputs for our multi-input model.

Ava-256 [25]. A publicly released multi-view human-head dataset comprising 256 subjects, with 80 high-resolution dome camera views per subject.

## 4.2 Metrics

Across our experiments we employ the following metrics, which are standard in the field [20, 24]: Peak Signal to Noise Ratio (PSNR), to measure the overall quality; Structural Similarity Index Measure (SSIM), to measure the perceived similarity; Learned Perceptual Image Patch Similarity (LPIPS) [52], with the VGG [39] backbone, to measure the distance between images in feature space; DreamSim (DS) [10,40], which measures image distance by focusing on mid-level features, fine-tuned on human perceptual judgements; ArcFace [8], measuring human face-centric identity shift as cosine similarity in feature space. Since it requires face detection to be measured, if a face is detected in the predicted image but not in the ground truth or viceversa we define the score as 0 (worst), whereas if a face is not detected in any of them, we define the score as 1 (best).

## 4.3 Baselines

There exist numerous methods that tackle similar problems — see Sec. 2. To the best of our knowledge, the most recent and accurate baselines for 3D face reconstruction without requiring camera poses are FaceLift [24] and VoluMe [20].

FaceLift [24] is divided in two models: the first one is a difusion-based novel view synthesis model that predicts 6 canonical views from a single input image, akin to our multi-output model, and is trained from Stable Difusion V2-1-unCLIP [37]; the second one is a GS-LRM model that lifts the 6 views from the previous model into pixel-aligned 3D gaussians, which can then be rendered from any view. We refer to the full pipeline as FaceLift, and we refer to the first half of the pipeline as FaceLift-NVS. Moreover, since the data used to train the model is not publicly available, we ensure a fair comparison with our model by also re-training FaceLift-NVS with our PSGS data directly from the Stable Difusion checkpoint. We refer to this model as FaceLift-NVS (PSGS).

VoluMe [20] is single-stage pipeline that directly predicts pixel-aligned 3D gaussians from a single input image. Neither the code nor the data used for VoluMe are publicly available at the time of writing, making a direct comparison impossible. However VoluMe is based on Splatter Image [41], which is shown in [20] to produce similar results when trained on the same data. Thus, we train it on our PSGS data and employ it as our baseline, named SI (PSGS). Due to VRAM requirements, it has been trained at resolution 256x256. The model is trained in two stages, we include results from both stages. If it is not specified, it is

assumed to be the final model.

For a fair comparison we use 24 transformer blocks in our models, for a total of approximately 1B parameters, which is comparable to the size of Facelift-NVS. Additional training details are available in the supplementary material.

## 4.4 Experimental results

Table 1: Quantitative evaluation on PSGS [11] validation set for NVS models (6 canonical views). Best in bold, second best in italics.
<table><tr><td></td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>DS ↓</td><td>ArcFace ↑</td></tr><tr><td>SI (PSGS) - I stage e [41]</td><td>20.15</td><td>0.8640 0.8614</td><td>0.2612 0.2351</td><td>0.02606</td><td>0.6761</td></tr><tr><td>SI (PSGS) - II stage [41]</td><td>20.16</td><td></td><td></td><td>0.02297</td><td>0.7403</td></tr><tr><td>FaceLift-NVS [24]</td><td>17.76</td><td>0.7942</td><td>0.2785</td><td>0.04531</td><td>0.6554</td></tr><tr><td>FaceLift-NVS (PŠGS) [24]</td><td>16.75</td><td>0.7900</td><td>0.3206</td><td>0.02262</td><td>0.6680</td></tr><tr><td>Ours</td><td>22.29</td><td>0.8806</td><td>0.1905</td><td>0.02191</td><td>0.6944</td></tr><tr><td>Ours MO</td><td>22.40</td><td>0.8752</td><td>0.1927</td><td>0.01729</td><td>0.7069</td></tr><tr><td>Ours MI-MO</td><td>23.13</td><td>0.8816</td><td>0.1762</td><td>0.01379</td><td>0.7206</td></tr></table>

![](images/d423aa6cf8960c274010db4dcbe640dc69fe14f3914d35bd0bf5d14b2c0269ea.jpg)  
Fig. 4: Qualitative results on PSGS [11] validation set on the 6 canonical views.

We show quantitative and qualitative results for the novel view synthesis stage of the pipeline on the PSGS dataset in Tab. 1 and Fig. 4, using the 6 canonical output views defined by FaceLift [24]. Our approach outperforms the baselines across most metrics, with the exception of ArcFace, where Splatter Image (PSGS) performs sligthly better. Hoerver, while it is able to achieve good results in terms of identity preservation in near-frontal views, it struggles to produce high-quality back and side views, because it is not designed to do so, resulting in blurry images, as visible in Fig. 4. It also struggles to correctly center the head, resulting in slight misalignments in the side views. Our method is designed to produce high-quality results across all views, and to do so with correct alignment for use in downstream 3D reconstruction pipelines. FaceLift-NVS, instead, tends to produce very detailed but artificial-looking results, likely due to its reliance on a synthetic training dataset. When it is trained on our fewer but more realistic data the results become less detailed but more realistic. We also notice that, compared to our method, FaceLift-NVS (PSGS) tends to produce small visual artifacts, hindering its overall quality. We conjecture that this is due to the small amount of training data, which might not be suficient to easily train a large difusion model. Our model, instead, is able to better leverage the training data and correctly converges even with small amounts of data, thanks to the training strategy we propose. Compared to our single-output model, our multi-output model (MO) is able to further improve the metrics thanks to the cross-attention between the diferent views during generation. This is also visible in Fig. 4 (right), where Ours MO is more cross-view consistent in terms of beard and hair style compared to Ours. Moreover, the multi-input multi-output model (MI-MO) is able to further improve the results by leveraging the additional loosely-posed input views, producing very realistic and consistent non-frontal views. It is also worth noting that, while FaceLift requires around 5 seconds to generate the 6 views on our setup, our method is able to generate the same views in around 1.5 seconds. The fastest method remains Splatter Image which, while not designed to render non-frontal views, can run in real time.

We compare the baselines on the same dataset using 32 turntable views in Tab. 2 and Fig. 5. While Splatter Image can naturally render these views, for the other methods we use the pipelines described in Sec. 3. The results are consistent with the ones obtained on the 6 canonical views, with our method outperforming the baselines across the metrics. Notice that, similarly to Splatter Image, our single-input single-output model is not trained to produce only the 6 canonical views, and thus can be used to render any target view of the subject without an additional step (see Fig. 5, row 7). Compared to Splatter Image, this approach produces much more detailed results, including non-frontal views, but with inconsistencies across views, as it is not designed to handle them. When consistency is needed, the full pipeline should be employed.

We also show similar results on the Ava-256 [25] dataset in Tab. 3 and Fig. 6, using the 10 subjects and 11 views defined by FaceLift [24]. These results have been obtained following the experimental description of FaceLift [24] and VoluMe [20]. Due to camera-pose definitions in this dataset, the original formulation of this experiment required manual aligment. Since the experimental code has not been released by neither [24] nor [20], it is impossible to directly compare results, as well as to ensure unbiased alignment. As a consequence, this dataset remains challenging to work with in this specific scenario, and we believe its results to be less reliable compared to previous experiments. However, to partially overcome this limitation we implement an automatic background removal and alignment procedure based on Rembg [5], and apply it to all methods. The final results are similar to our previous benchmarks, with the most notable diference being the improved performance of FaceLift in terms of identity preservation, likely due to its larger training dataset, compared to when it is trained with PSGS data. The results on the other metrics are largely unchanged and, with the same training data, our method fully outperforms FaceLift.

We additionally refer the reader to our supplementary material for qualitative results on full body data.

![](images/54dcf0aa8c00c99a2fa69d3bfd9aaba36403e62ffbf8de28451f26676e61d26b.jpg)  
Fig. 5: Qualitative results on PSGS [11] validation set on the 32 turntable views.

## 4.5 Ablation studies

In order to better understand the impact of the diferent components of our method, we perform an ablation study on the validation set of PSGS [11] with the previously defined 6 canonical views. In Tab. 4 we see that the original resolution of ArchonView [48], 256×256, is not suficient to achieve good results on faces. Moreover, training the model from scratch on face data does not lead to good results, as well as starting from a powerful non-NVS model such as VAR-d36 [43]. Even though such model is already trained on 512x512 images, it produces highquality results but without correctly preserving identity and view consistency, see Fig. 7 (left) row 4. Thus, adapting the existing ArchonView 256x256 model to our target resolution is crucial, and has to be approached by keeping the number of scales the same, as conjectured in Sec. 3.4. In Tab. 4 row 5-6 and Fig. 7 (left) row 1-2 we see that this choice is key to achieve good results. Moreover, while small models (d12) can be easily adapted to the new resolution, larger models (d30) struggle to converge, justifying the need for our first training stage, which allows the model to adapt to the new resolution before being trained on the target data. Finally, the fine-tuned VQ-VAE slightly improves the results.

Table 2: Quantitative evaluation on PSGS [11] validation set of the full image-to-3D pipelines (32 turntable views). Best in bold, second best in italics.
<table><tr><td></td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>DS ↓</td><td>ArcFace ↑</td></tr><tr><td>SI (PSGS) - I stage [41]</td><td>18.46</td><td>0.8548</td><td>0.2755</td><td>0.03257</td><td>0.7108</td></tr><tr><td>SI (PSGS) - II stage [41]</td><td>18.32</td><td>0.8488</td><td>0.2549</td><td>0.03184</td><td>0.7596</td></tr><tr><td>FaceLift [24]</td><td>13.28</td><td>0.7791</td><td>0.3103</td><td>0.1008</td><td>0.7121</td></tr><tr><td>FaceLift (PŠGS) [24]</td><td>15.41</td><td>0.8377</td><td>0.2668</td><td>0.07339</td><td>0.7357</td></tr><tr><td>Ours + [11]</td><td>20.78</td><td>0.8805</td><td>0.2248</td><td>0.02440</td><td>0.7520</td></tr><tr><td>Ours MO + [11]</td><td>20.82</td><td>0.8772</td><td>0.2249</td><td>0.02319</td><td>0.7608</td></tr><tr><td>Ours  $M I { - } M O + [ 1 1 ]$ </td><td>21.91</td><td>0.8865</td><td>0.2110</td><td>0.01188</td><td>0.7737</td></tr></table>

Table 3: Quantitative evaluation on AVA-256, 10 subjects, 11 views, as defined in [24]. Best in bold, second best in italics.
<table><tr><td></td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>DS ↓</td><td>ArcFace ↑</td></tr><tr><td>SI (PSGS) [41]</td><td>13.06</td><td>0.7736</td><td>0.2911</td><td>0.06242</td><td>0.6689</td></tr><tr><td>FaceLift [24]</td><td>14.49</td><td>0.8038</td><td>0.2546</td><td>0.03482</td><td>0.7699</td></tr><tr><td>FaceLift (PŠGS) [24]</td><td>14.51</td><td>0.8146</td><td>0.2489</td><td>0.03584</td><td>0.5837</td></tr><tr><td>Ours + [11]</td><td>15.34</td><td>0.8201</td><td>0.2357</td><td>0.03616</td><td>0.6275</td></tr><tr><td>Ours MO + [11]</td><td>15.24</td><td>0.8197</td><td>0.2448</td><td>0.03385</td><td>0.6405</td></tr><tr><td>Ours MI-MO + [11]</td><td>15.67</td><td>0.8209</td><td>0.2350</td><td>0.04187</td><td>0.6251</td></tr></table>

Table 4: Ablation study of the NVS-stage of our pipeline on PSGS [11] validation set (6 canonical views) for single-input single-output models. K=10 unless otherwise specified, and using around half the training steps of the final model.

<table><tr><td>Depth</td><td>Resolution</td><td>Checkpoint</td><td></td><td>1st stage</td><td>2nd stage</td><td>VQ-VAE</td><td>PSNR ↑</td><td>ArcFace ↑</td></tr><tr><td>d12</td><td>256</td><td>AV</td><td>[48]</td><td></td><td></td><td>default</td><td>12.02</td><td>17.01</td></tr><tr><td>d12</td><td>256</td><td>AV [48]</td><td></td><td></td><td>√</td><td>default</td><td>19.77</td><td>39.20</td></tr><tr><td>d12</td><td>256</td><td>Scratch</td><td></td><td></td><td>√</td><td>default</td><td>19.68</td><td>34.87</td></tr><tr><td>d36</td><td>512</td><td>VAR [43]</td><td></td><td></td><td>√</td><td>default</td><td>16.72</td><td>38.17</td></tr><tr><td>d12</td><td>512 K=12</td><td>AV [48]</td><td></td><td></td><td>√</td><td>default</td><td>15.70</td><td>33.71</td></tr><tr><td>d12</td><td>512</td><td>AV [48]</td><td></td><td></td><td>√</td><td>default</td><td>21.43</td><td>57.75</td></tr><tr><td>d30</td><td>512</td><td>AV [48]</td><td></td><td></td><td>√</td><td>default</td><td>20.34</td><td>54.51</td></tr><tr><td>d30</td><td>512</td><td>AV [48]</td><td></td><td>√</td><td>√</td><td>default</td><td>21.66</td><td>63.38</td></tr><tr><td>d30</td><td>512</td><td>AV [48]</td><td></td><td>√</td><td>√</td><td>fine-tuned</td><td>22.24</td><td>67.72</td></tr></table>

In Fig. 7 (right) we also show the negative efect of inference-time classifierfree guidance during the third training stage. Moreover, compared to the d12 256×256 CFG=0 setting seen in the figure, which achieves a PSNR of 19.99 and an ArcFace of 0.4044, the same model without classifier-free guidance during the third training stage achieves a PSNR of 20.28 and an ArcFace of 0.4296, corroborating the importance of this choice.

![](images/7fb4e6b9f1d438a7b4c4095c674b74c015c6c0e582ad391178cf36dfdb6dcbd9.jpg)  
Fig. 6: Qualitative results on AVA-256 [25] for subject 20210901–0833–LAS440

## 4.6 Limitations

While achieving good results across all the metrics and experiments, our method still has some limitations. In particular, the training data only contains neutral expressions and no accessories, thus the model struggles to correctly generalize to unseen expressions and accessories, such as hats. These limitations are similar to the ones reported by FaceLift [24]. Additionally, we noticed that all the methods tend to sufer from background color bleeding. This is noticeable in Fig. 6, where the black background is not perfectly removed and a portion of the hair is incorrectly tinted in black in the other views. Brighter background colors worsen the issue. More fine-grained background removal techniques would surely help to mitigate this, but it is still a common problem in the field.

## 5 Conclusions

We have presented a novel method for photorealistic view-consistent novel view synthesis of human faces using next-scale transformers, which can be applied to other domains as well, such as full bodies. Our method achieves state-of-the-art results on the PSGS dataset, outperforming previous methods in terms of both quantitative metrics and qualitative visual quality, and is also efective on the Ava-256 dataset, showing that it can generalize well to diferent datasets and scenarios. We believe that future work can focus on addressing limitations by exploring more diverse training data, as well as recent developments in the next-scale VAR framework, and further specializing the architecture to diferent scenarios such as full body reconstruction.

![](images/de942aeb7a98196dee97fde1750fc3c2c7b5f9f446c9ed905460788108a06ca3.jpg)

![](images/1a606c0155c3ca8f6d3325c7227ecba73bc361e6d414694285af1bd5bbac7371.jpg)  
Fig. 7: (Left) Adapting the model to single-output 512×512 resolution. The first three rows use ArchonView [48] (d30) as the initial checkpoint, which was originally trained for NVS of objects at 256×256, while the fourth row uses VAR [43] (d36), which was originally trained for image generation at 512×512. VQ-VAE is not fine-tuned in any of the models. (Right) Impact of Classifier-Free Guidance (CFG) on our d12 multi-output model during inference. Similar efects are noticed on bigger models.

## References

1. An, S., Xu, H., Shi, Y., Song, G., Ogras, U., Luo, L.: Panohead: Geometry-aware 3d full-head synthesis in 360<sup>◦</sup> (2023) 3

2. Cao, C., Simon, T., Kim, J.K., Schwartz, G., Zollhoefer, M., Saito, S., Lombardi, S., Wei, S.E., Belko, D., Yu, S.I., Sheikh, Y., Saragih, J.: Authentic volumetric avatars from a phone scan. ACM Trans. Graph. 41(4) (Jul 2022). https://doi. org/10.1145/3528223.3530143, https://doi.org/10.1145/3528223.3530143 3

3. Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., Sutskever, I.: Generative pretraining from pixels. In: III, H.D., Singh, A. (eds.) Proceedings of the 37th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 119, pp. 1691–1703. PMLR (13–18 Jul 2020), https://proceedings.mlr.press/v119/chen20s.html 3

4. Chen, Y., Lan, Y., Zhou, S., Wang, T., Pan, X.: Sar3d: Autoregressive 3d object generation and understanding via multi-scale 3d vqvae. In: CVPR (2025) 3

5. Daniel Gatis: Rembg, a tool to remove images background, https://github.com/ danielgatis/rembg 12

6. Deitke, M., Liu, R., Wallingford, M., Ngo, H., Michel, O., Kusupati, A., Fan, A., Laforte, C., Voleti, V., Gadre, S.Y., VanderBilt, E., Kembhavi, A., Vondrick, C., Gkioxari, G., Ehsani, K., Schmidt, L., Farhadi, A.: Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663 (2023) 3

7. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. arXiv preprint arXiv:2212.08051 (2022) 7

8. Deng, J., Guo, J., Xue, N., Zafeiriou, S.: Arcface: Additive angular margin loss for deep face recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019) 9

9. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. CVPR (2021) 3, 4, 6

10. Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., Isola, P.: Dreamsim: Learning new dimensions of human visual similarity using synthetic data (2023) 9

11. Garbin, E., Adam, G., Krams, O., Barzelay, Z., Guendelman, E., Schwarz, M., Presutto, M., Vatelmacher, M., Shenkman, Y., Peker, E., Druker, I., Patish, U., Blum, Y., Bluvstein, M., Li, J., Khirodkar, R., Saito, S.: Capture, canonicalize, splat: Zero-shot 3d gaussian avatars from unstructured phone images (2025), https://arxiv.org/abs/2510.14081 2, 4, 8, 10, 12, 13, 14, 20, 21, 22, 24, 25

12. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2. p. 2672–2680. NIPS’14, MIT Press, Cambridge, MA, USA (2014) 3

13. Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., Liu, X.: Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis (2024), https://arxiv.org/abs/2412.04431 2, 3

14. Ho, J., Salimans, T.: Classifier-free difusion guidance. In: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (2021), https: //openreview.net/forum?id=qw8AKxfYbI 3

15. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 4401–4410. Computer Vision Foundation / IEEE (2019). https://doi.org/ 10.1109/CVPR.2019.00453, http://openaccess.thecvf.com/content\_CVPR\_ 2019/html/Karras\_A\_Style-Based\_Generator\_Architecture\_for\_Generative\_ Adversarial\_Networks\_CVPR\_2019\_paper.html 3

16. Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., Aila, T.: Analyzing and improving the image quality of stylegan. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8107–8116 (2020). https: //doi.org/10.1109/CVPR42600.2020.00813 3

17. Kirschstein, T., Romero, J., Sevastopolsky, A., Nießner, M., Saito, S.: Avat3r: Large animatable gaussian reconstruction model for high-fidelity 3d head avatars. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 12089–12100 (October 2025) 3

18. Kong, X., Liu, S., Lyu, X., Taher, M., Qi, X., Davison, A.J.: Eschernet: A generative model for scalable view synthesis. arXiv preprint arXiv:2402.03908 (2024) 3

19. Kuznetsova, A., Rom, H., Alldrin, N.G., Uijlings, J.R.R., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., Ferrari, V.: The open images dataset v4. International Journal of Computer Vision 128, 1956 – 1981 (2018), https://api.semanticscholar.org/CorpusID:53296866 8

20. de La Gorce, M., Hewitt, C., Takacs, T., Gerdisch, R., Hosenie, Z., Meishvili, G., Kowalski, M., Cashman, T.J., Criminisi, A.: VoluMe – authentic 3d video calls from live gaussian splat prediction (2025), https://arxiv.org/abs/2507.21311 1, 3, 4, 9, 11

21. Lee, D., Kim, C., Kim, S., Cho, M., Han, W.S.: Autoregressive image generation using residual quantization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11523–11532 (2022) 4

22. Liu, J., Han, J., Yan, B., Wu, H., Zhu, F., Wang, X., Jiang, Y., Peng, B., Yuan, Z.: Infinitystar: Unified spacetime autoregressive modeling for visual generation (2025), https://arxiv.org/abs/2511.04675 2, 3

23. Liu, R., Wu, R., Hoorick, B.V., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1- to-3: Zero-shot one image to 3d object (2023) 3, 6

24. Lyu, W., Zhou, Y., Yang, M.H., Shu, Z.: Facelift: Learning generalizable single image 3d face reconstruction from synthetic heads. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 12691–12701 (October 2025) 1, 2, 3, 4, 9, 10, 11, 13, 14, 24, 25

25. Martinez, J., Kim, E., Romero, J., Bagautdinov, T., Saito, S., Yu, S.I., Anderson, S., Zollhöfer, M., Wang, T.L., Bai, S., Li, C., Wei, S.E., Joshi, R., Borsos, W., Simon, T., Saragih, J., Theodosis, P., Greene, A., Josyula, A., Maeta, S.M., Jewett, A.I., Venshtain, S., Heilman, C., Chen, Y.T., Fu, S., Elshaer, M.E.A., Du, T., Wu, L., Chen, S.C., Kang, K., Wu, M., Emad, Y., Longay, S., Brewer, A., Shah, H., Booth, J., Koska, T., Haidle, K., Andromalos, M., Hsu, J., Dauer, T., Selednik, P., Godisart, T., Ardisson, S., Cipperly, M., Humberston, B., Farr, L., Hansen, B., Guo, P., Braun, D., Krenn, S., Wen, H., Evans, L., Fadeeva, N., Stewart, M., Schwartz, G., Gupta, D., Moon, G., Guo, K., Dong, Y., Xu, Y., Shiratori, T., Prada, F., Pires, B.R., Peng, B., Bufalini, J., Trimble, A., McPhail, K., Schoeller, M., Sheikh, Y.: Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars. NeurIPS Track on Datasets and Benchmarks (2024) 9, 11, 14, 21, 25

26. Peebles, W., Xie, S.: Scalable difusion models with transformers. arXiv preprint arXiv:2212.09748 (2022) 3

27. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.C.: Film: Visual reasoning with a general conditioning layer. In: AAAI (2018) 7

28. Qiu, L., Gu, X., Li, P., Zuo, Q., Shen, W., Zhang, J., Qiu, K., Yuan, W., Chen, G., Dong, Z., Bo, L.: Lhm: Large animatable human reconstruction model from a single image in seconds. In: arXiv preprint arXiv:2503.10625 (2025) 4

29. Qiu, L., Li, P., Zuo, Q., Gu, X., Dong, Y., Yuan, W., Zhu, S., Han, X., Chen, G., Dong, Z.: Pf-lhm: 3d animatable avatar reconstruction from pose-free articulated human images. ArXiv abs/2506.13766 (2025), https://api.semanticscholar. org/CorpusID:279410430 4

30. Qu, Y., Yuan, K., Hao, J., Zhao, K., Xie, Q., Sun, M., Zhou, C.: Visual autoregressive modeling for image super-resolution. arXiv preprint arXiv:2501.18993 (2025) 3

31. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision (2021), https://arxiv.org/abs/ 2103.00020 6

32. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation (2021), https://arxiv.org/abs/ 2102.12092 3

33. Razavi, A., van den Oord, A., Vinyals, O.: Generating diverse high-fidelity images with vq-vae-2 (2019) 3

34. Ren, S., Yu, Q., He, J., Shen, X., Yuille, A., Chen, L.C.: FlowAR: Scale-wise autoregressive image generation meets flow matching. In: Forty-second International Conference on Machine Learning (2025), https://openreview.net/forum?id= JfLgvNe1tj 3

35. Ren, S., Yu, Y., Ruiz, N., Wang, F., Yuille, A., Xie, C.: M-var: Decoupled scalewise autoregressive modeling for high-quality image generation (2024), https:// arxiv.org/abs/2411.10433 3

36. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models (2021) 3

37. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2021) 9

38. Saito, S., Schwartz, G., Simon, T., Li, J., Nam, G.: Relightable gaussian codec avatars. In: CVPR (2024) 3

39. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: International Conference on Learning Representations (2015) 9

40. Sundaram, S., Fu, S., Muttenthaler, L., Tamir, N.Y., Chai, L., Kornblith, S., Darrell, T., Isola, P.: When does perceptual alignment benefit vision representations? (2024), https://arxiv.org/abs/2410.10817 9

41. Szymanowicz, S., Rupprecht, C., Vedaldi, A.: Splatter image: Ultra-fast singleview 3d reconstruction. In: The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 1, 2, 4, 9, 10, 13, 24, 25

42. Tang, H., Wu, Y., Yang, S., Xie, E., Chen, J., Chen, J., Zhang, Z., Cai, H., Lu, Y., Han, S.: HART: Eficient visual generation with hybrid autoregressive transformer. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=q5sOv4xQe4 3

43. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. NeurIPS (2024) 2, 3, 4, 7, 8, 12, 13, 15

44. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 6000–6010. NIPS’17, Curran Associates Inc., Red Hook, NY, USA (2017) 6

45. Vetter, T., Blanz, V.: Estimating coloured 3d face models from single images: An example based approach. In: Burkhardt, H., Neumann, B. (eds.) Computer Vision — ECCV’98. pp. 499–513. Springer Berlin Heidelberg, Berlin, Heidelberg (1998) 3

46. Wang, T., Zhang, B., Zhang, T., Gu, S., Bao, J., Baltrusaitis, T., Shen, J., Chen, D., Wen, F., Chen, Q., Guo, B.: Rodin: A generative model for sculpting 3d digital avatars using difusion. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4563–4573 (2023). https://doi.org/10.1109/ C 2729 2023 00443 3

47. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., Hutchinson, B., Han, W., Parekh, Z., Li, X., Zhang, H., Baldridge, J., Wu, Y.: Scaling autoregressive models for content-rich text-toimage generation. Transactions on Machine Learning Research (2022), https:// openreview.net/forum?id=AFDcYJKhND, featured Certification 3

48. Yuan, S., Zhao, H.: Next-scale autoregressive models are zero-shot single-image object view synthesizers (2025), https://arxiv.org/abs/2503.13588 2, 3, 4, 6, 7, 12, 13, 15, 21

49. Zhang, B., Cheng, Y., Wang, C., Zhang, T., Yang, J., Tang, Y., Zhao, F., Chen, D., Guo, B.: Rodinhd: High-fidelity 3d avatar generation with difusion models. In: European Conference on Computer Vision. pp. 465–483. Springer (2025) 3

50. Zhang, K., Bi, S., Tan, H., Xiangli, Y., Zhao, N., Sunkavalli, K., Xu, Z.: Gslrm: Large reconstruction model for 3d gaussian splatting. In: Computer Vision - ECCV 2024: 18th European Conference, Milan, Italy, September 29 - October 4, 2024, Proceedings, Part XXII. pp. 1–19. Springer-Verlag, Berlin, Heidelberg (2024). https://doi.org/10.1007/978-3-031-72670-5\_1, https://doi.org/10.1007/ 978-3-031-72670-5\_1 3

51. Zhang, Q., Dai, X., Yang, N., An, X., Feng, Z., Ren, X.: Var-clip: Text-to-image generator with visual auto-regressive modeling (2024) 3

52. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: CVPR (2018) 9

53. Zhuang, Y., Lv, J., Wen, H., Shuai, Q., Zeng, A., Zhu, H., Chen, S., Yang, Y., Cao, X., Liu, W.: Idol: Instant photorealistic 3d human creation from a single image (2024), https://arxiv.org/abs/2412.14963 4

## A.1 Training details

Our model is trained for 50 hours on 64 NVIDIA A100 GPUs in the first stage. Subsequently, the single-input model is fine-tuned (second stage) for 18 hours on the same hardware. We refer to this model as "Ours". It is specialized for multioutput generation (third stage) for 2 hours. We refer to this model as Ours MO. The multi-input model is fine-tuned (second stage) for 14 hours on 128 A100 GPUs, and specialized for multi-output generation (third stage) for 2 hours on the same hardware. We refer to this model as Ours MI-MO. The VQ-VAE is fine-tuned on the target dataset for 30 epochs on 8 A100 GPUs, has a codebook size V = 4096 and uses the following scales for the K = 10 latent feature maps: (1, 2, 3, 4, 6, 9, 13, 18, 24, 32) for 512x512 images, and (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) for 256x256 images. In the experiments with K = 12 latent feature maps, we use the following scales: (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 24, 32).

## A.2 Multiple-output SOS token embedding

We justify the choice of our multi-view SOS tokens by testing three scenarios using a d12 model at 256x256: a separate SOS token per output view (our model); a shared SOS token across all views, obtained by embedding the concatenation of all the target views; a shared SOS token across all views, obtained without any view information (since the views are always fixed and in a known order). These three approaches achieve a PSNR of 19.99, 19.85 and 19.85 respectively, showing little to no impact. We opted for the first approach since it is the most intuitive and straightforward.

## A.3 A diferent scenario: human bodies

Our method is not limited to human faces, and can be applied to other categories of data as well, such as human bodies. We train our single-input pipeline on the PSGS [11] dataset for human body reconstruction, following the same training process described in the main paper for human faces (including VQ-VAE finetuning). We report qualitative results from the validation set in Fig. A.1, showing the canonical views produced by our architecture, and in Fig. A.2, showing the final views rendered after the GS-LRM 3D lifting process. Notice that our single-input single-output model (Ours) is not trained to produce only the 6 canonical views, and thus can be used to render any target view of the subject. In the second row of Fig. A.2 we show turntable views rendered directly by Ours without the lifting step. This demonstrates that our method can be applied to this domain without any architectural changes, showing the flexibility of the method and a potential direction for future work.

![](images/c8ce9f7b895f67a2d301126510fcd3ffb21dc069ba7f63ad077df878257557cf.jpg)  
Fig. A.1: Qualitative results of our method on the PSGS [11] dataset for human body reconstruction. We show the canonical views produced by our method, compared to the GT and the input views.

## A.4 Object NVS training

While we use the weights from ArchonView [48] for the initialization of our models, we adapt them to the higher 512×512 resolution during the first stage of our training process. We show in Fig. A.3 example views of the SS3D dataset we employed.

## A.5 Multi-scale VQ-VAE fine-tuning

While fine-tuning an existing component is not a core contribution of this work, we find that fine-tuning the VQ-VAE on the new resolution and dataset can further improve the results of our method, with very little cost. Complementing the ablation study in the main paper, we show in Fig. A.4 the impact of fine-tuning the VQ-VAE on PSGS [11]. The results are obtained by simply autoencoding the 6 canonical views of the validation set, and show small but clear improvements in the quality and identity preservation of the reconstructed images.

## A.6 Additional qualitatives

We show in Fig. A.5 the canonical views produced by our method for the subject shown in Fig. 5 of the main paper. We also show additional qualitative results on the AVA-256 [25] dataset in Fig. A.6.

<table><tr><td>GT</td><td>↑ ↑ ↑ 1 1J 育 1 1 1 1 1 ↑ 才 ↑ 1 才 1 1才 才</td></tr><tr><td>Ours Ours + [11]</td><td>才 1 1 ↑ 卜 1 f 才方書 1 1 11↑↑ll ↑↑</td></tr><tr><td>Ours MO + [11]</td><td>↑↑ 1A↑FI ↑ 1 1 1 ↑</td></tr><tr><td>GT</td><td>才 1 介 1 1 1 1 1 上 小 1 1 1</td></tr><tr><td>Ours</td><td>育 育 育 1 1 1 育 1 1 1 1 1 1 </td></tr><tr><td>Ours + [11]</td><td>才 育 1 1 1 1 1 1 L 1 1 1 1</td></tr><tr><td>Ours MO + [11]</td><td>育 言 1 1 n1 1 1 1 上 1 1 1 1</td></tr></table>

Fig. A.2: Qualitative results of our method on the PSGS [11] dataset for human body reconstruction. We show the final views rendered after the GS-LRM 3D lifting process, compared to the GT. Since Ours is not trained to produce only canonical views, it can also be used to render turntable views directly, without the lifting step, as shown in the second rows.

![](images/78bd262a44696308338c484dc9401deba321a602a75119bc202ecd61963e3b41.jpg)  
Fig. A.3: Example object views from the SS3D dataset.

![](images/a179b344f47a56172f8ccff5dfed11d19e9048af888cfb53e6bc00232e1b5c8d.jpg)

Fig. A.4: Multi-scale VQ-VAE finetuning results. We show the reconstructions from the base and finetuned VQ-VAEs at both 256 and 512 resolutions, compared to the GT, showing small but noticeable improvements in reconstruction quality after finetuning.

# Inputs GT SI (PSGS) I [41] 1221A1 SI (PSGS) II [41] L2RIAS FaceLift [24] LLALAI FaceLift 1LRLAA (PSGS) [24] Ours 12R128 Ours MO Ours MI-MO

Fig. A.5: Qualitative results on PSGS [11] validation set on the 6 canonical views for the NVS part of each model.

![](images/6be36940d11b67334e234ea5e94c4da3f093255856359f084500db2f1a2aea37.jpg)  
Fig. A.6: Qualitative results on AVA-256 [25] for subject 20220310–1128–ZSC414