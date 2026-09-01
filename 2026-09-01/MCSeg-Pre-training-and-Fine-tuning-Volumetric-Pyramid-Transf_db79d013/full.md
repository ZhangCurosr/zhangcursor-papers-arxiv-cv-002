CardiacSeg SwinUNETR Dice=0.7488 Dice=0.7531

# MCSeg: Pre-training and Fine-tuning Volumetric Pyramid Transformer for Multi-modal Cardiac Image Segmentation

Zhiyu Ye, Hairong Zheng, Senior Member, IEEE, and Tong Zhang, Member, IEEE

Abstract— Automatic cardiac image segmentation is pivotal for diagnosing and treating cardiac diseases. In this work, we introduce MCSeg, a volumetric transformer-based network tailored for multi-modal cardiac segmentation. To overcome the architectural mismatch inherent in existing hybrid networks, we propose a novel Scaling Feature Pyramid (SFP). Unlike conventional skip connections, the SFP effectively bridges the single-scale 3D Vision Transformer (ViT) encoder and the multi-scale CNN decoder by transforming the ViT’s output into a hierarchical feature pyramid, ensuring that global contextual information is effectively leveraged. For the training paradigm, the ViT encoder first undergoes self-supervised pre-training via masked image modeling. Subsequently, the network is fine-tuned on downstream tasks, during which a regional mutual information (RMI) loss is integrated to improve boundary segmentation accuracy. In experiments, MCSeg consistently outperforms eleven SOTA methods on CT dataset ImageCHD, multi-modal dataset MM-WHS, MRI dataset HVSMR-2.0 and MSD Heart, highlighting the effectiveness of our MCSeg for multi-modal cardiac segmentation tasks. Furthermore, MCSeg’s superior performance in few-shot experiment showcases its significant potential in adapting to limited data scenarios. Codes and pre-trained ViT-B weights are open-sourced at https://openi.pcl. ac.cn/OpenMedIA/MCSeg.

Index Terms— Multi-modal cardiac segmentation, volumetric vision transformer, scaling feature pyramid, selfsupervised pre-training.

## I. INTRODUCTION

EDICAL image segmentation, an important application extensively researched, encompassing various organs and diseases [2]–[5]. Automated segmentation of cardiac images is a crucial step for non-invasive assessment of cardiac structures and functions. It offers indispensable support for diagnosis, disease monitoring, treatment planning, and prognosis [6].

Image  
GT  
![](images/d294d37e73fdcabda1a87d78642d4cfb3fc87e13c4cfb39ec510bc56c447e4aa.jpg)  
Fig. 1: Qualitative visualizations of models in few-shot experiment on image with ID “1085” from ImageCHD dataset. Results are presented in axial, coronal and sagittal views.

Cardiac imaging examinations commonly involve multimodal imaging techniques such as echocardiography, computed tomography (CT) and magnetic resonance imaging (MRI). Unlike echocardiography, which is highly operatordependent, CT and MRI examinations can generate comprehensive and consistent 3D visual representations of the cardiac structure. Such capability enables the diagnosis and quantification of cardiac structural abnormalities and facilitates precise surgical planning. Consequently, whole-heart segmentation is routinely conducted on CT and MRI images.

Conventional methods for cardiac image segmentation, such as probability atlases [7], [8] and early machine learning methods [9], have largely been superseded by deep learning, particularly convolutional neural network (CNN) based approaches [10], [11]. While CNNs excel at learning hierarchical features on specific small datasets, their inductive biases often limit domain generalization across diverse datasets. Furthermore, medical image segmentation inherently suffers from a severe scarcity of finely annotated 3D data and a long-tailed distribution of rare diseases. Unlike natural image datasets with millions of samples, medical datasets typically contain only dozens to hundreds of cases, posing a significant obstacle to training robust and universally applicable models.

To tackle these challenges, especially in scenarios with limited annotated data and relatively abundant unlabeled images, self-supervised learning (SSL) — particularly masked image modeling (MIM) — has emerged as a powerful paradigm. Within this context, the Vision Transformer (ViT) [12] has demonstrated exceptional pre-training capabilities and an innate ability to model long-range dependencies. Our objective is to leverage these advantages in developing a network that can be universally applied to diverse 3D cardiac segmentation tasks. Additionally, we aim to enhance the network’s adaptability to images that are rarely or even never encountered.

Building upon our prior work CardiacSeg [13], we introduce MCSeg, a volumetric transformer-based network tailored for multi-modal cardiac segmentation. MCSeg employs a ViT encoder pre-trained via masked autoencoders (MAE) [14] on unannotated cardiac volumes to establish prior anatomical knowledge. Notably, this pre-training stage scales up the data volume by five times compared to our previous work. For downstream fine-tuning, this pre-trained encoder is integrated with our customized Scaling Feature Pyramid (SFP), which explicitly bridges single-scale ViT and multi-scale decoder. Furthermore, we incorporate the regional mutual information (RMI) loss [15] during the fine-tuning process to further enhance boundary segmentation accuracy. As illustrated in the few-shot experiment (Fig. 1), MCSeg significantly outperforms existing baselines, highlighting its exceptional capability in overcoming medical data scarcity.

The main contributions of this paper are outlined as follows:

• We introduce MCSeg, a volumetric transformer-based network tailored for multi-modal cardiac segmentation. The key innovation of this architecture is a novel Scaling Feature Pyramid (SFP), which explicitly bridges singlescale ViT and multi-scale decoder, thereby better exploiting the capabilities of the pre-trained 3D ViT.

• We adopt a two-stage training paradigm that combines self-supervised 3D MAE pre-training with downstream task fine-tuning. during the fine-tuning, we incorporate a regional mutual information (RMI) loss to improve boundary segmentation accuracy.

• We achieve state-of-the-art (SOTA) performance across four diverse cardiac datasets (ImageCHD, HVSMR-2.0, MM-WHS, and MSD). Extensive experiments demonstrate that MCSeg exhibits cross-modality adaptability and few-shot learning potential.

## II. RELATED WORKS

## A. Volumetric Medical Image Segmentation

U-Net [16] and its 3D variants [17], [18] established the typical symmetric encoder-decoder paradigm for medical image segmentation. Recently, volumetric transformers have emerged to model long-range dependencies, broadly categorized into pure transformer and hybrid CNN-transformer architectures. Pure transformers, such as nnFormer [19] and VT-UNet [20], utilize self-attention mechanisms in both encoding and decoding stages but often require substantial computational resources. Conversely, hybrid models integrate transformer encoders with CNN decoders to balance global context modeling with local detail sensitivity. For instance, CoTr [21] connects a deformable transformer to CNN decoder, while UNETR [22] and SwinUNETR [23] bridge ViT and Swin Transformer encoders to CNN decoders via multi-resolution skip connections. Concurrently, modernized CNN baselines like Med-NeXt [24] and foundation models like SAM-Med3D [25] have demonstrated strong baseline performance. While these architectures demonstrate powerful capabilities, applying a dedicated pre-training and fine-tuning paradigm exclusively to cardiac-specific tasks aligns much better with practical clinical requirements. To this end, MCSeg couples a tailored hybrid network design with specialized large-scale cardiac pretraining, constructing prior contextual information for highly accurate and clinically reliable segmentation.

## B. Masked Image modeling

While the massive parameter count of transformers enables them to achieve excellent performance, it also makes them highly dependent on large-scale training data. To alleviate this heavy data dependency, MIM has been widely adopted for self-supervised ViT pre-training. Adapted from natural language processing (NLP) [26], visual approaches like BEiT [27] and SimMIM [28] reconstruct masked image patches to learn semantic representations. Notably, MAE [14] employs an asymmetric encoder-decoder architecture with a high masking ratio, significantly reducing training computation, a paradigm subsequently extended to 3D medical imaging [29]. By leveraging 3D MAE on unannotated cardiac datasets, our MCSeg effectively overcomes the data scarcity challenge, establishing comprehensive anatomical priors that directly benefits downstream fine-tuning.

## C. Feature Pyramid Networks

For pixel-level vision tasks such as object detection and semantic segmentation, preserving original input resolution throughout training is computationally prohibitive and impedes the extraction of high-level semantic information. Feature Pyramid Networks (FPNs) [30] leverage the naturally occurring multi-scale feature maps of CNN backbones by combining a top-down pathway with lateral connections to construct semantically rich feature pyramids. In segmentation frameworks such as Panoptic FPN [31], FPNs further facilitate multi-scale feature integration during decoder up-sampling. Since standard ViTs output only single-scale token sequences, frameworks like ViTDet [32] incorporate feature pyramids to adapt ViT backbones for dense prediction tasks. In volumetric medical segmentation, simple skip-connection schemes struggle to efficiently bridge the scale and semantic gap between the ViT’s single-scale representations and the CNN decoder’s multi-scale feature requirements. To resolve this, MCSeg employs a customized SFP that explicitly transforms the deepest well-encoded ViT features into a multi-scale pyramid, ensuring optimal cross-scale contextual utilization for the decoder.

## III. METHOD

## A. Network Architecture

Our MCSeg comprises three primary components: a volumetric transformer encoder, an SFP, and a CNN decoder (Fig.

![](images/e2f772e580bf8feed6476e1339c8cd6c10c8af1b4c655d81d3edc0e1205c8885.jpg)  
Fig. 2: The schematic overview of MCSeg. During the pre-training stage, the ViT encoder undergoes training using the MAE method. Subsequently, in the fine-tuning stage, the ViT encoder loads the pre-trained weights and freezes its parameters. The training process in this stage exclusively focuses on the SFP and the decoder.

2). Unlike conventional U-shaped networks that rely on hierarchical encoders, MCSeg leverages a powerful, non-hierarchical pre-trained ViT backbone to extract global contextual representations, which are subsequently remapped into a multi-scale pyramid via the SFP to facilitate precise segmentation.

1) Volumetric Embedding and Encoder: Given a 3D input volume $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times D }$ with height H, width $W ,$ , and depth $D ,$ the initial step involves partitioning the volume into N non-overlapping cubic patches $p _ { i } \in \mathbb { R } ^ { P \times P \times P } .$ , where $P = 1 6$ denotes the fixed patch size and $\begin{array} { r } { N = \frac { H \times W \times D } { P ^ { 3 } } } \end{array}$ . Each patch $p _ { i }$ is flattened and projected into an E-dimensional latent space through a linear mapping function $f : p _ { i } \to e _ { i } \in$ $\mathbb { R } ^ { E }$ . This process yields a sequence of patch embeddings $\mathbf { x _ { e } } = [ e _ { 1 } , \bar { e } _ { 2 } , \ldots , e _ { N } ] \in \mathbb { R } ^ { N \times E }$ To preserve spatial information within the 1D sequence, learnable positional embeddings $\mathbf { x _ { p } } ~ \in ~ \mathbb { R } ^ { N \times E }$ are added to the sequence. The combined representation, ${ \bf z _ { 0 } } = { \bf x _ { e } } + { \bf x _ { p } } .$ , serves as the input to the ViT encoder. We adopt the ViT-B configuration consisting of $L =$ 12 successive transformer blocks, where each block models long-range volumetric dependencies through multi-head selfattention. The encoder maintains a constant latent resolution of $\begin{array} { r } { { \frac { H } { P } } \times { \frac { W } { P } } \times { \frac { D } { P } } } \end{array}$ across all layers. This structure is strategically chosen to align with the requirements of large-scale MIM pretraining, ensuring that the global semantic features are robustly encoded before being passed to the decoder.

2) Scaling Feature Pyramid (SFP) and Decoder: The integration of SFP with the CNN decoder enables segmentation generation from the single-scale outputs of the ViT encoder. To enrich the feature representations for progressive up-sampling, the SFP transforms the ViT’s single-scale outputs into multiscale features, allowing the decoder to access both high-level contextual information and scale-appropriate spatial details. The SFP transforms the final outputs $\mathbf { z _ { 1 2 } }$ of the encoder, which have a size of $\begin{array} { r } { \left( \frac { H } { 1 6 } , \frac { W } { 1 6 } , \frac { D } { 1 6 } , \widehat { E } \right) _ { * } } \end{array}$ (i.e., a scale of $\textstyle { \frac { 1 } { 1 6 } } )$ into feature maps at scales ${ \begin{array} { l } { { \frac { 1 } { 4 } } , } \end{array} } { \frac { 1 } { 8 } } , \ { \frac { 1 } { 1 6 } }$ , and $\frac { 1 } { 3 2 }$ using four SFP blocks (see Fig. 2). The decoder consists of four CNN decoder blocks. In the bottom three decoder blocks, feature representations of relatively small resolution are up-sampled using transpose convolution layers. These up-sampled feature maps are then concatenated with the corresponding feature maps from the SFP. The last decoder block is similar to the other decoder blocks, except that it concatenates the upsampled lower-resolution feature maps with the original input image. After these decoder blocks, the segmentation result is obtained through a fully connected layer.

## B. Fine-tuning for Cardiac Segmentation

1) Training Strategy: In the training stage, the pre-trained ViT backbone was adopted as the encoder and frozen during training. In contrast, only the SFP and decoder were trained to construct the segmentation results in this stage.

2) Loss Function: During training, we optimize the network using a combination of soft Dice loss [18], cross-entropy (CE) loss, and regional mutual information (RMI) loss [15]. While Dice and CE losses strictly assess overall class-level and voxel-level accuracy respectively, they lack local spatial restrictions. Since accurate and smooth boundaries are crucial in medical image segmentation [33], we incorporate RMI loss. By maximizing the mutual information between the predicted and ground-truth regions, RMI loss essentially enhances local consistency and improves boundary segmentation accuracy. Therefore, the RMI loss is defined as described in [15]:

$$
L _ { r m i } \left( G , P \right) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { c = 1 } ^ { C } \left( - I ^ { b , c } \left( G , P \right) \right)\tag{1}
$$

where B is the number of regions in the input, which is determined by the down-sampling factor $d _ { s }$ and square region size $r . \ I \left( G , P \right)$ represents the mutual information between G and P, calculated according to the formula in:

$$
I \left( G , P \right) = \sum _ { x \in \mathcal { G } } \sum _ { y \in \mathcal { P } } p \left( x , y \right) \log \frac { p \left( x , y \right) } { p \left( x \right) p \left( y \right) } .\tag{2}
$$

In summary, the goal of the training stage is to minimize the compound loss function as follows:

$$
\begin{array} { r l r } & { } & { L \left( G , P \right) = L _ { d i c e } \left( G , P \right) + \lambda _ { c e } L _ { c e } \left( G , P \right) } \\ & { } & { + \sigma _ { e p _ { r m i } } \lambda _ { r m i } L _ { r m i } \left( G , P \right) , \quad } \end{array}\tag{3}
$$

where $\lambda _ { c e }$ and $\lambda _ { r m i }$ are hyper-parameters. In our experiments, $\lambda _ { c e } = 0 . 5$ and $\lambda _ { r m i } = 0 . 1 . ~ \sigma _ { e p _ { r m i } }$ is a smooth step function designed for regulating the timing of incorporating RMI loss into the compound loss function during training, it is defined as

$$
\sigma _ { e p _ { r m i } } = \frac { 1 } { 1 + e ^ { e p _ { r m i } - e p _ { c u r } } } ,\tag{4}
$$

where $e p _ { r m i }$ is a hyper-parameter, $e p _ { c u r }$ and $e p _ { m a x }$ represent the current and maximum epoch during training, $e p _ { c u r } \in$ $[ 0 , e p _ { m a x } ] \cap \mathbb { Z }$

## IV. EXPERIMENTS

## A. Datasets

1) ImageCAS: ImageCAS [34] is a large-scale coronary artery segmentation dataset consisting of 1000 CTA images. As the complete heart structure is visible in coronary CTA images, we utilize the entire dataset solely for pre-training.

2) ImageCHD: ImageCHD [35] is a comprehensive dataset dedicated to the segmentation of congenital heart diseases (CHDs). It consists of 110 3D CT images, covering 16 types of CHDs. Each image in the dataset is annotated with labels for seven cardiac substructures, including the left ventricle (LV), right ventricle (RV), left atrium (LA), right atrium (RA), myocardium (Myo), aorta (AO), and pulmonary artery (PA). During fine-tuning, the dataset was divided into training, validation, and test sets, consisting of 77, 11, and 22 cases, respectively, based on disease types. Images in the training and validation sets were also utilized for pre-training.

3) HVSMR-2.0: HVSMR-2.0 [36] is a 3D cardiovascular MRI dataset for whole-heart segmentation in CHDs, which consisting of 60 cardiovascular MRI scans with manual segmentation masks of the four cardiac chambers and four great vessels, i.e., LV, RV, LA, RA, AO, PA, superior vena cava (SVC), and inferior vena cava (IVC). The dataset was divided into training, validation, and test sets comprising 42, 6, and 12 samples, respectively, for fine-tuning, and all images were utilized for pre-training except for those in the test set.

4) MM-WHS: MM-WHS [37], [38] is a 3D multi-modal whole-heart segmentation dataset, consisting of 60 CT and 60 MRI images. Each modality containing 20 annotated cases, all of which were utilized in comparative experiments. The remaining 40 unlabeled images in each modality were utilized for pre-training. The cardiac substructures annotated in this dataset are identical to those in the ImageCHD dataset.

5) MSD Task02 Heart: The Task02 of Medical Segmentation Decathlon (MSD) [39] is the left atrium segmentation for MRI images, including 20 labeled training data. This data was employed for comparative experiment.

## B. Implementation Details

Our MCSeg framework was built on PyTorch<sup>1</sup> and MONAI<sup>2</sup>. The model was pre-trained on four 80G A100 GPUs using the open-source implementation available at OpenMed $\mathrm { \mathbf { A } } ^ { 3 4 }$ [40]. All subsequent fine-tuning and ablation studies were conducted on a single 32G V100 GPU. The ViT backbone underwent pre-training for 3,200 epochs with batch size of 12 per GPU, using AdamW optimizer with a learning rate of $1 0 ^ { - 3 }$ . During fine-tuning stage, models were trained for 300 epochs with batch size of 2, utilizing AdamW optimizer (learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 5 } )$ , complemented by a 20-epoch linear warm-up and a cosine annealing learning rate scheduler. As part of the pre-processing steps, all input data was initially resampled to spacing of $1 \times 1 \times 1 \ m m ^ { 3 }$ . For CT images, the HU values were clipped into [500, 2000] (for ImageCHD) or [−300, 700] (for MM-WHS) and normalized to the range [0, 1]. For MRI images, 1% to 99% of the image intensities were clipped and normalized to [0, 1]. Subsequently, the input image was cropped into a bounding box containing only the foreground based on its label. This bounding box was then randomly cropped into four cubes, each of size (128, 128, 128), serving as input samples. Data augmentations included random addition of Gaussian noise with standard deviation of 0.05, random scaling of intensity by factor of 0.1, and intensity shifting with randomly selected offset from [−0.1, 0.1] for the images.

The Dice coefficient [18] is employed to measure the experimental results.

## C. Comparative Experiments

MCSeg was compared with eleven volumetric medical image segmentation networks, including three pure CNNbased methods (UNet++ [41], nnUNet [42], MedNeXt [24]), two pure transformer-based methods (nnFormer [19], VT-UNet [20]), and six hybrid CNN–Transformer methods (CoTr [21], TransUNet [43], UNETR [22], SwinUNETR [23], CardiacSeg [13], SAM-Med3D [25]). In the experiments, VT-UNet, SwinUNETR, CardiacSeg, and SAM-Med3D employed the pre-trained weights and fine-tuning strategies mentioned in their original papers. For UNETR<sup>5</sup> [22] and MCSeg, we used the pre-trained ViT-B weights proposed in this study and kept their parameters frozen during training.

TABLE I: Quantitative comparison of full-data experimental results for different models on (a) ImageCHD and (b) HVSMR-2.0 datasets. († indicates models adopt their original pre-trained weights. Subscripts $V i T - B _ { 2 2 3 }$ and $V i T - B _ { 7 2 3 }$ denote models utilized the ViT-B pre-trained on 223 and 723 CT images, respectively, in this study. Wilcoxon signed-rank test p-values (each model vs. MCSeg): $^ { * } ( p < 0 . 0 5 )$ $^ { * * } ( p < 0 . 0 1 )$ , <sup>∗∗∗</sup>(p < 0.001).)  
(a) Results on ImageCHD dataset.
<table><tr><td>Networks</td><td>Overall</td><td>LV</td><td>RV</td><td>LA</td><td>RA</td><td>Myo</td><td>AO</td><td>PA</td></tr><tr><td colspan="9">Pure CNN-based Methods</td></tr><tr><td>UNet++**</td><td> $0 . 6 2 1 \pm 0 . 0 5 6$ </td><td> $0 . 7 1 5 \pm 0 . 0 9 2$ </td><td> $0 . 4 8 7 \pm 0 . 1 2 4$ </td><td> $0 . 7 9 7 \pm 0 . 0 5 7$ </td><td> $0 . 6 9 8 \pm 0 . 0 5 8$ </td><td> $0 . 4 6 0 \pm 0 . 1 0 4$ </td><td> $0 . 5 6 2 \pm 0 . 0 8 3$ </td><td> $0 . 6 1 9 \pm 0 . 1 1 4$ </td></tr><tr><td>nnUNet*</td><td> $0 . 8 8 2 \pm 0 . 0 5 0$ </td><td> $\mathbf { 0 . 9 2 5 \pm 0 . 0 2 2 }$ </td><td> $\underline { { 0 . 8 8 2 \pm 0 . 0 3 9 } }$ </td><td> $\underline { { 0 . 9 1 6 \pm 0 . 0 2 2 } }$ </td><td> $0 . 8 9 6 \pm 0 . 0 2 5$ </td><td> $0 . 8 1 0 \pm 0 . 2 6 1$ </td><td> $\underline { { 0 . 8 8 7 \pm 0 . 0 6 6 } }$ </td><td> $\underline { { 0 . 8 5 7 \pm 0 . 0 8 9 } }$ </td></tr><tr><td>MedNeXt**</td><td> $0 . 8 5 8 \pm 0 . 0 4 3$ </td><td> $0 . 8 6 6 \pm 0 . 0 8 4$ </td><td> $0 . 8 2 6 \pm 0 . 1 0 7$ </td><td> $0 . 8 9 5 \pm 0 . 0 2 9$ </td><td> $0 . 8 9 1 \pm 0 . 0 3 2$ </td><td> $0 . 8 4 9 \pm 0 . 0 9 1$ </td><td> $0 . 8 6 6 \pm 0 . 0 6 6$ </td><td> $0 . 8 1 5 \pm 0 . 0 8 1$ </td></tr><tr><td colspan="9">Pure Transformer-based Methods</td></tr><tr><td>nnFormer**</td><td> $0 . 8 5 0 \pm 0 . 0 3 3$ </td><td> $0 . 8 9 3 \pm 0 . 0 4 6$ </td><td> $0 . 8 5 1 \pm 0 . 0 7 9$ </td><td> $0 . 8 7 5 \pm 0 . 0 3 2$ </td><td> $0 . 8 9 4 \pm 0 . 0 2 9$ </td><td> $0 . 8 7 2 \pm 0 . 0 4 9$ </td><td> $0 . 8 1 4 \pm 0 . 0 7 9$ </td><td> $0 . 7 5 9 \pm 0 . 0 8 9$ </td></tr><tr><td> $\mathbf { V T - U N e t } ^ { \dagger * * }$ </td><td> $0 . 8 2 0 \pm 0 . 0 6 0$ </td><td> $0 . 8 9 7 \pm 0 . 0 5 0$ </td><td> $0 . 8 3 2 \pm 0 . 0 8 2$ </td><td> $0 . 8 5 4 \pm 0 . 0 3 3$ </td><td> $0 . 8 6 6 \pm 0 . 0 4 7$ </td><td> $0 . 8 0 0 \pm 0 . 2 5 4$ </td><td> $0 . 7 8 7 \pm 0 . 0 8 6$ </td><td> $0 . 7 0 4 \pm 0 . 0 8 1$ </td></tr><tr><td colspan="9">CNN &amp; Transformer Hybrid Methods</td></tr><tr><td>CoTr**</td><td></td><td></td><td></td><td> $0 . 8 8 6 \pm 0 . 0 2 6$ </td><td> $0 . 8 6 4 \pm 0 . 0 4 3$ </td><td>0.849 ± 0.050</td><td> $0 . 8 4 5 \pm 0 . 0 6 8$ </td><td> $0 . 8 0 3 \pm 0 . 0 9 5$ </td></tr><tr><td>TransUNet**</td><td> $0 . 8 5 2 \pm 0 . 0 4 0$ </td><td> $0 . 9 0 2 \pm 0 . 0 3 7$ </td><td> $0 . 8 1 8 \pm 0 . 1 2 9$ </td><td> $0 . 9 1 1 \pm 0 . 0 2 2$ </td><td> $\underline { { 0 . 9 0 7 \pm 0 . 0 2 6 } }$ </td><td> $0 . 8 7 9 \pm 0 . 0 5 3$ </td><td> $0 . 8 8 1 \pm 0 . 0 4 4$ </td><td> $0 . 8 4 0 \pm 0 . 0 6 5$ </td></tr><tr><td></td><td> $\underline { { 0 . 8 8 7 \pm 0 . 0 2 6 } }$ </td><td> $0 . 9 1 6 \pm 0 . 0 4 0$ </td><td> $0 . 8 7 7 \pm 0 . 0 7 6$ </td><td> $0 . 9 0 1 \pm 0 . 0 4 0$ </td><td> $\overline { { 0 . 8 9 6 \pm 0 . 0 3 3 } }$ </td><td> $\mathbf { 0 . 8 9 2 \pm 0 . 0 4 3 }$ </td><td> $0 . 8 7 6 \pm 0 . 0 5 7$ </td><td>0.817 ± 0.116</td></tr><tr><td> $\mathrm { U N E T R } _ { V i T - B _ { 7 2 3 } } * *$  SwinUNETR†**</td><td> $0 . 8 7 9 \pm 0 . 0 3 6$   $0 . 8 7 0 \pm 0 . 0 3 6$ </td><td> $0 . 9 1 0 \pm 0 . 0 4 4$   $0 . 8 9 5 \pm 0 . 0 4 4$ </td><td> $0 . 8 6 7 \pm 0 . 0 6 9$   $0 . 8 4 8 \pm 0 . 0 9 3$ </td><td> $0 . 8 9 8 \pm 0 . 0 2 9$ </td><td> $0 . 8 9 6 \pm 0 . 0 2 9$ </td><td> $0 . 8 7 5 \pm 0 . 0 5 4$ </td><td> $0 . 8 6 2 \pm 0 . 0 8 3$ </td><td> $0 . 8 2 2 \pm 0 . 0 9 9$ </td></tr><tr><td>CardiacSeg†**</td><td> $0 . 8 7 5 \pm 0 . 0 3 1$ </td><td> $0 . 9 1 8 \pm 0 . 0 3 2$ </td><td> $0 . 8 6 1 \pm 0 . 0 5 5$ </td><td> $0 . 8 9 9 \pm 0 . 0 2 9$ </td><td> $0 . 8 9 9 \pm 0 . 0 3 1$ </td><td> $0 . 8 7 8 \pm 0 . 0 5 5$ </td><td> $0 . 8 6 1 \pm 0 . 0 5 4$ </td><td> $0 . 8 1 3 \pm 0 . 0 8 3$ </td></tr><tr><td> $\mathbf { S A M - M e d } 3 \mathbf { D } ^ { \dagger * * }$ </td><td> $0 . 5 4 4 \pm 0 . 1 3 7$ </td><td> $0 . 7 6 3 \pm 0 . 1 1 8$ </td><td> $0 . 7 0 2 \pm 0 . 2 2 0$ </td><td> $0 . 4 8 9 \pm 0 . 2 4 0$ </td><td> $0 . 6 7 1 \pm 0 . 1 0 6$ </td><td> $0 . 4 7 5 \pm 0 . 2 1 9$ </td><td> $0 . 3 1 2 \pm 0 . 1 6 4$ </td><td> $0 . 3 9 6 \pm 0 . 1 6 2$ </td></tr><tr><td> $\mathbf { M C S e g _ { V i T - B _ { 7 2 3 } } }$  (Ours)</td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 2 4 }$ </td><td> $\underline { { 0 . 9 2 3 \pm 0 . 0 3 3 } }$ </td><td> $\mathbf { 0 . 8 8 8 \pm 0 . 0 4 4 }$ </td><td> $\mathbf { 0 . 9 1 8 \pm 0 . 0 2 1 }$ </td><td> $\mathbf { 0 . 9 0 9 \pm 0 . 0 2 6 }$ </td><td> $\underline { { 0 . 8 8 6 \pm 0 . 0 3 8 } }$ </td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 5 5 }$ </td><td> $\mathbf { 0 . 8 5 7 \pm 0 . 0 7 7 }$ </td></tr></table>

(b) Results on HVSMR-2.0 dataset.
<table><tr><td>Networks</td><td>Overall</td><td>LV</td><td>RV</td><td>LA</td><td>RA</td><td>AO</td><td>PA</td><td>SVC</td><td>IVC</td></tr><tr><td colspan="10">Pure CNN-based Methods</td></tr><tr><td>UNet++***</td><td> $0 . 7 8 3 \pm 0 . 0 8 3$ </td><td> $0 . 8 8 5 \pm 0 . 0 8 9$ </td><td> $0 . 7 6 5 \pm 0 . 2 2 2$ </td><td> $0 . 7 8 6 \pm 0 . 1 7 2$ </td><td> $0 . 8 3 8 \pm 0 . 0 5 2$ </td><td> $0 . 8 4 4 \pm 0 . 0 9 0$ </td><td> $0 . 6 9 6 \pm 0 . 1 8 7$ </td><td> $0 . 7 3 8 \pm 0 . 1 1 5$ </td><td> $0 . 7 2 7 \pm 0 . 0 6 6$ </td></tr><tr><td>nnUNet*</td><td> $0 . 8 1 6 \pm 0 . 0 4 3$ </td><td> $0 . 8 7 2 \pm 0 . 0 3 4$ </td><td> $0 . 8 2 7 \pm 0 . 1 2 2$ </td><td> $\underline { { 0 . 8 3 8 \pm 0 . 0 2 3 } }$ </td><td> $0 . 7 7 8 \pm 0 . 2 5 9$ </td><td> $0 . 8 7 0 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 7 9 7 \pm 0 . 0 6 0 }$ </td><td> $0 . 7 7 8 \pm 0 . 0 4 9$ </td><td> $0 . 7 7 1 \pm 0 . 0 4 6$ </td></tr><tr><td>MedNeXt***</td><td> $0 . 8 0 7 \pm 0 . 0 7 9$ </td><td> $0 . 8 8 9 \pm 0 . 0 9 9$ </td><td> $0 . 8 0 3 \pm 0 . 2 2 2$ </td><td> $0 . 8 0 9 \pm 0 . 0 9 9$ </td><td> $\underline { { 0 . 8 5 0 \pm 0 . 0 8 1 } }$ </td><td> $0 . 8 6 0 \pm 0 . 0 9 0$ </td><td> $0 . 7 0 9 \pm 0 . 1 7 0$ </td><td> $0 . 7 5 8 \pm 0 . 1 2 4$ </td><td> $0 . 7 9 4 \pm 0 . 0 9 2$ </td></tr><tr><td colspan="10">Pure Transformer-based Methods</td></tr><tr><td>nnFormer***</td><td> $0 . 7 2 7 \pm 0 . 1 0 2$ </td><td> $0 . 8 1 8 \pm 0 . 1 5 1$ </td><td> $0 . 7 2 5 \pm 0 . 2 3 5$ </td><td> $0 . 7 1 2 \pm 0 . 1 5 4$ </td><td> $0 . 7 8 9 \pm 0 . 1 2 8$ </td><td> $0 . 8 1 8 \pm 0 . 0 8 2$ </td><td> $0 . 6 3 4 \pm 0 . 1 7 2$ </td><td> $0 . 6 4 3 \pm 0 . 1 2 4$ </td><td> $0 . 7 0 1 \pm 0 . 1 5 7$ </td></tr><tr><td> ${ \mathbf { V } } { \mathrm { T - U N e t } } ^ { \dagger * * }$ </td><td> $0 . 8 1 6 \pm 0 . 0 7 3$ </td><td> $0 . 9 1 7 \pm 0 . 0 5 5$ </td><td> $\mathbf { 0 . 8 4 3 \pm 0 . 2 2 1 }$ </td><td> $0 . 8 1 5 \pm 0 . 0 9 0$ </td><td> $0 . 7 9 5 \pm 0 . 2 4 6$ </td><td> $0 . 8 8 2 \pm 0 . 0 6 4$ </td><td> $0 . 7 5 0 \pm 0 . 1 4 4$ </td><td> $0 . 7 3 6 \pm 0 . 1 5 1$ </td><td> $0 . 7 8 9 \pm 0 . 0 4 4$ </td></tr><tr><td colspan="10">CNN &amp; Transformer Hybrid Methods</td></tr><tr><td>CoTr**</td><td> $0 . 8 1 0 \pm 0 . 0 8 9$ </td><td></td><td> $0 . 8 3 2 \pm 0 . 1 9 0$ </td><td> $0 . 8 3 1 \pm 0 . 1 0 4$ </td><td> $0 . 7 8 4 \pm 0 . 2 5 8$ </td><td> $0 . 8 8 3 \pm 0 . 0 8 0$ </td><td> $0 . 7 5 4 \pm 0 . 1 4 8$ </td><td> $0 . 7 0 9 \pm 0 . 2 2 5$ </td><td> $0 . 8 0 0 \pm 0 . 0 6 2$ </td></tr><tr><td>TransUNet*</td><td> $\underline { { 0 . 8 1 8 \pm 0 . 0 6 1 } }$ </td><td> $0 . 8 9 9 \pm 0 . 0 8 7$   $0 . 8 7 8 \pm 0 . 0 7 8$ </td><td> $0 . 8 1 2 \pm 0 . 1 8 7$ </td><td> $0 . 8 2 8 \pm 0 . 0 6 0$ </td><td> $0 . 8 3 7 \pm 0 . 0 9 7$ </td><td> $0 . 8 6 9 \pm 0 . 0 7 2$ </td><td> $0 . 7 5 0 \pm 0 . 1 4 5$ </td><td> $0 . 7 8 6 \pm 0 . 0 8 7$ </td><td> $0 . 7 8 7 \pm 0 . 0 6 4$ </td></tr><tr><td> $\mathrm { U N E T R } _ { V i T - B _ { 2 2 3 } } \mathrm { \mathrm { } } ^ { * }$ </td><td> $\overline { { 0 . 8 0 7 \pm 0 . 0 8 3 } }$ </td><td> $0 . 8 8 7 \pm 0 . 1 0 2$ </td><td> $0 . 8 0 6 \pm 0 . 2 1 8$ </td><td> $\mathbf { 0 . 8 4 1 \pm 0 . 0 7 0 }$ </td><td> $0 . 8 1 0 \pm 0 . 2 0 5$ </td><td> $\underline { { 0 . 8 8 7 \pm 0 . 0 7 2 } }$ </td><td> $0 . 7 0 5 \pm 0 . 1 8 8$ </td><td> $\underline { { 0 . 7 8 3 \pm 0 . 1 4 9 } }$ </td><td> $0 . 7 6 6 \pm 0 . 0 5 8$ </td></tr><tr><td> $\mathrm { S w i n U N E T R } ^ { \dag \ast }$  #</td><td> $0 . 7 9 5 \pm 0 . 0 7 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 9 5$ </td><td> $0 . 7 7 7 \pm 0 . 2 4 3$ </td><td> $0 . 8 2 3 \pm 0 . 0 3 6$ </td><td> $\mathbf { 0 . 8 5 1 \pm 0 . 0 7 1 }$ </td><td> $0 . 8 3 9 \pm 0 . 0 7 1$ </td><td> $0 . 7 3 5 \pm 0 . 1 4 2$ </td><td> $0 . 7 5 7 \pm 0 . 1 2 0$ </td><td> $0 . 7 5 0 \pm 0 . 0 7 1$ </td></tr><tr><td> $\mathrm { C a r d i a c S e g ^ { \dagger * } }$ </td><td> $0 . 8 1 6 \pm 0 . 0 6 5$ </td><td> $0 . 8 7 0 \pm 0 . 1 0 3$ </td><td> $0 . 8 2 0 \pm 0 . 0 9 4$ </td><td> $0 . 8 0 8 \pm 0 . 0 9 7$ </td><td> $0 . 7 9 6 \pm 0 . 2 5 5$ </td><td> $0 . 8 8 1 \pm 0 . 0 6 2$ </td><td> $0 . 7 6 2 \pm 0 . 1 3 2$ </td><td> $\mathbf { 0 . 7 9 3 \pm 0 . 0 7 0 }$ </td><td> $0 . 8 0 5 \pm 0 . 0 6 7$ </td></tr><tr><td> $\mathbf { S A M - M e d } 3 \mathbf { D } ^ { \dagger * * * }$ </td><td> $0 . 5 9 5 \pm 0 . 1 3 1$ </td><td> $0 . 8 4 1 \pm 0 . 0 9 5$ </td><td> $0 . 8 1 8 \pm 0 . 1 1 5$ </td><td> $0 . 5 5 9 \pm 0 . 1 4 1$ </td><td> $0 . 7 7 2 \pm 0 . 0 7 9$ </td><td> $0 . 4 1 7 \pm 0 . 2 1 8$ </td><td> $0 . 5 2 6 \pm 0 . 2 2 4$ </td><td> $0 . 4 6 0 \pm 0 . 1 4 2$ </td><td> $\overline { { 0 . 3 6 8 \pm 0 . 2 1 8 } }$ </td></tr><tr><td> $\mathbf { M C S e g _ { V i T - B _ { 2 2 3 } } }$  (Ours)</td><td> $\mathbf { 0 . 8 3 2 \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 9 2 5 \pm 0 . 0 4 2 }$ </td><td> $\underline { { 0 . 8 4 0 \pm 0 . 1 7 5 } }$ </td><td> $0 . 8 2 5 \pm 0 . 1 2 5$ </td><td> $0 . 8 1 7 \pm 0 . 1 9 9$ </td><td> $\mathbf { 0 . 8 9 4 \pm 0 . 0 5 8 }$ </td><td> $\underline { { 0 . 7 7 5 \pm 0 . 1 4 3 } }$ </td><td> $0 . 7 7 5 \pm 0 . 1 4 6$ </td><td> $\mathbf { 0 . 8 1 0 \pm 0 . 0 5 0 }$ </td></tr></table>

1) Whole-heart Segmentation on ImageCHD Dataset: On the ImageCHD dataset, we evaluate model adaptability under both full-data and few-shot settings.

Few-shot Experiment. The few-shot training set was constructed by randomly selecting eight samples from the original training set, while the test set remained unchanged. Fig. 5 compares four transformer-CNN hybrid networks. Among them, only SwinUNETR employs a Swin Transformer as the encoder, while the others utilize ViT encoders. Leveraging pretrained ViT weights markedly improves few-shot performance: compared to training from scratch, UNETR, CardiacSeg, and MCSeg achieve relative Dice score improvements of 43%, 46%, and 58%, respectively. MCSeg benefits the most from pre-training, achieving the highest Dice scores in both full-data $( 0 . 8 9 6 \pm 0 . 0 2 4 )$ and few-shot $( 0 . 7 7 6 \pm 0 . 0 6 5 )$ settings, with the smallest performance drop between the two. Moreover, MCSeg’s performance under the few-shot setting is the closest to its full-data result. A case study is illustrated in Fig. 1, where MCSeg generates segmentations most closely aligned with the ground truth, whereas other models exhibit varying degrees of missegmentation. These results highlight the efficacy of pre-training in boosting MCSeg’s robustness and enhanced adaptability to limited data scenarios.

Full-data Experiment. Table Ia reports the overall and per-substructure average Dice scores, along with standard deviations, for each model. MCSeg demonstrates statistically superior performance, achieving the highest overall score and leading in five of the seven substructures. Notably, MCSeg maintains robust stability across all regions, particularly excelling on challenging substructures like the RV, AO, and PA, where competing models suffer from degraded accuracy. Qualitatively, Fig. 3 shows that our method generates segmentations most closely aligned with the ground truth across three views and 3D renderings.

2) Whole-heart Segmentation on HVSMR-2.0 Dataset: As shown in Table Ib, MCSeg achieves the highest overall Dice score $( 0 . 8 3 2 \pm 0 . 0 7 1 )$ , outperforming competing methods by 1.4%. Notably, MCSeg yields the best or second best performance on substructures LV, RV, AO, PA, and IVC, with low standard deviations, indicating high accuracy and consistency. Fig. 4 presents a case study, in which we observe that while

Sagittal

Coronal

![](images/0813fb3268af42a499265373a47940d6867eded1bd6233cb5f5cf86ea1f3bb52.jpg)  
Fig. 3: Qualitative comparison for different models on image with $\mathrm { I D \ ^ { \cdots } } 1 0 2 9 ^ { \cdots }$ from ImageCHD dataset.

other models exhibit varying degrees of missegmentation across different views, the segmentation result of MCSeg closely aligns with the ground truth. These results demonstrate that MCSeg shows a greater advantage over other models on MRI images compared to CT images.

3) Whole-heart Segmentation on MM-WHS Dataset: MM-WHS dataset includes both CT and MRI modalities. For each modality, we conducted five-fold cross-validation using UNETR, SwinUNETR, CardiacSeg, and MCSeg, trained both from scratch and with pre-trained weights. The statistic analyses are shown in Fig. 6a&6b. As illustrated, $\mathbf { M C S e g } _ { V i T - B _ { 2 2 3 } }$ achieves the highest Dice scores and the lowest intra-group variance in nearly all folds. Remarkably, despite the distinct image distributions of CT and MRI images, the ViT-B pretrained solely on CT images yields even greater performance improvements on MRI segmentation than on CT itself. Case studies are provided in supplementary materials.

4) Left Atrium Segmentation on MSD Heart Dataset: Although this task is less challenging than whole-heart segmentation, the statistical analysis presented in Fig. 6c clearly demonstrates the advantages of MCSeg and its pre-trained weights, which are even more apparent compared to the MM-WHS experiments. This indicates that MCSeg effectively leverages the capabilities of the ViT backbone obtained during pre-training. These results highlight the robust performance of MCSeg across tasks of varying difficulty. Furthermore, as shown in the case study Fig. 7, a qualitative comparison of segmentation results on two representative images further confirms that MCSeg consistently outperforms other models.

![](images/60099dfccb3f9c3995058ff128888e039e3226d96537e10544746bff01525b19.jpg)  
Fig. 4: Qualitative comparison for different models on image with $\mathrm { I D \ ^ {  } p a t { l } } 1 ^ { \prime \prime }$ from HVSMR-2.0 dataset.

![](images/ec3aadc7dca26ba62e5bdaf630f9b18c9882c67a906e5f4bc7768b2ad6eead98.jpg)  
Fig. 5: Comparison of models training with and without pre-training weights under few-shot and full-data settings on ImageCHD dataset.

5) Computational Efficiency: Table II summarizes the number of network parameters, GPU memory consumption during training, and the average inference time per case for MCSeg and all baseline models. Both GPU memory usage and inference time were measured on ImageCHD with an input size of (128, 128, 128). As shown, MCSeg requires fewer trainable parameters, uses less GPU memory, and achieves the shortest inference time among all compared methods.

![](images/edaa91096cd3e54027f372969a59255881395fa1cd063d3be017ef17c34acb50.jpg)  
(a) Whole-heart segmentation on MM-WHS CT dataset.

![](images/9aac64ea9e1f8c00ed3e3b45c5c354f1de3df29a1b1a93665e726c3cf5ec7554.jpg)  
(b) Whole-heart segmentation on MM-WHS MRI dataset.

![](images/f6ea6c5d2e447622bf5576e84268bd0451036f866a40b9784ecda32339862119.jpg)  
(c) LA segmentation on MSD Heart dataset.  
Fig. 6: Results of 5-fold cross-validation for eight models on various datasets. Models in reds are trained from scratch.

GT  
MCSeg  
CardiacSeg SwinUNETR  
![](images/13dc327c68e404cbb45e0a936337ae4f534deedb5bde77e07165e87f4957b59f.jpg)  
Fig. 7: Qualitative comparisons for four models on cases “la 024” and $" \mathrm { l a } \_ 0 2 9 "$ from MSD Heart dataset.

## D. Ablation Studies

To optimize the performance of MCSeg, we conducted a series of ablation experiments towards configuration of SFP, the effects of data scale and modality used for pre-training, fine-tuning strategy, and the combination of loss function.

1) SFP Design: Table IIIa demonstrates that our proposed SFP effectively bridges the encoder and decoder, outperforming the conventional FPN [30] by 1.5%. Finally, we investigated the feature source and scaling configurations within the SFP (Table IIIb). Constructing the multi-scale pyramid exclusively from the deepest ViT block $\left( z _ { 1 2 } \right)$ alongside a scaling configuration of $\textstyle { \left( { \frac { \hat { 1 } } { 3 2 } } , { \frac { 1 } { 1 6 } } , { \frac { 1 } { 8 } } , { \frac { 1 } { 4 } } \right) }$ yields the optimal performance. This indicates that the deepest representations $\left( z _ { 1 2 } \right)$ contain more comprehensive semantic information, serving as a robust foundation for multi-scale decoding compared to fusing intermediate block outputs.

TABLE II: Comparison of GPU memory usage during training, numbers of parameters and average inference time per case for various networks in ImageCHD experiments.
<table><tr><td>Networks</td><td>GPU Memory Usage (G)</td><td>#(Params) (M) Trainable</td><td>Total</td><td>Inference Time (s)</td></tr><tr><td>SAM-Med3D† [25]</td><td>43.4</td><td>100.5</td><td>100.5</td><td>70.7</td></tr><tr><td>nnFormer [19]</td><td>59.7</td><td>37.6</td><td>37.6</td><td>42.7</td></tr><tr><td>VT-UNet† [20]</td><td>20.1</td><td>11.8</td><td>11.8</td><td>36.8</td></tr><tr><td>nnUNet [42]</td><td>20.2</td><td>31.2</td><td>31.2</td><td>25.4</td></tr><tr><td>TransUNet [43]</td><td>24.6</td><td>41.4</td><td>41.4</td><td>19.3</td></tr><tr><td>MedNeXt [24]</td><td>59.2</td><td>12.0</td><td>12.0</td><td>14.4</td></tr><tr><td>UNet++ [41]</td><td>36.1</td><td>7.0</td><td>7.0</td><td>4.3</td></tr><tr><td>SwinUNETR† [23]</td><td>47.9</td><td>19.6</td><td>62.2</td><td>4.0</td></tr><tr><td>CoTr [21]</td><td>31.0</td><td>41.9</td><td>41.9</td><td>1.9</td></tr><tr><td> $\mathrm { U N E T R } _ { V i T - B _ { 7 2 3 } }$  [22]</td><td>29.5</td><td>2.7</td><td>92.8</td><td>1.1</td></tr><tr><td> $\mathbf { M C S e g } _ { V i T - B _ { 7 2 3 } }$  (ours)</td><td>28.8</td><td>7.5</td><td>97.6</td><td>1.0</td></tr></table>

TABLE III: Ablation studies of (a) pyramid design and (b) SFP settings on ImageCHD dataset.  
(a)
<table><tr><td>Pyramid Design</td><td>Dice</td></tr><tr><td>None (baseline)</td><td> $0 . 8 7 6 \pm 0 . 0 3 7$ </td></tr><tr><td>FPN [30]</td><td> $0 . 8 7 7 \pm 0 . 0 4 2$ </td></tr><tr><td>SFP (ours)</td><td> $\mathbf { 0 . 8 9 2 \pm 0 . 0 3 3 }$ </td></tr></table>

(b)
<table><tr><td>Features</td><td>Pyramid Scale</td><td>Dice</td></tr><tr><td> $( z _ { 3 } , z _ { 6 } , z _ { 9 } , z _ { 1 2 } )$ </td><td> $\textstyle { \left( { \frac { 1 } { 1 6 } } , { \frac { 1 } { 8 } } , { \frac { 1 } { 4 } } , { \frac { 1 } { 2 } } \right) }$ </td><td> $0 . 8 7 6 \pm 0 . 0 3 5$ </td></tr><tr><td>z12</td><td> $\textstyle { \left( { \frac { 1 } { 1 6 } } , { \frac { 1 } { 8 } } , { \frac { 1 } { 4 } } , { \frac { 1 } { 2 } } \right) }$ </td><td> $0 . 8 7 6 \pm 0 . 0 4 5$ </td></tr><tr><td> $( z _ { 3 } , z _ { 6 } , z _ { 9 } , z _ { 1 2 } )$ </td><td> $\textstyle { \left( { \frac { 1 } { 3 2 } } , { \frac { 1 } { 1 6 } } , { \frac { 1 } { 8 } } , { \frac { 1 } { 4 } } \right) }$ </td><td> $0 . 8 8 5 \pm 0 . 0 3 6$ </td></tr><tr><td>z12</td><td> $\textstyle { \left( { \frac { 1 } { 3 2 } } , { \frac { 1 } { 1 6 } } , { \frac { 1 } { 8 } } , { \frac { 1 } { 4 } } \right) }$ </td><td> $\mathbf { 0 . 8 9 2 \pm 0 . 0 3 3 }$ </td></tr></table>

2) Data Scale and Modalities for Pre-training: Table IV illustrates the impact of varying pre-training data scales and modalities on downstream segmentation performance and computational costs. Utilizing pre-trained weights consistently achieves significant performance gains over training from scratch, improving Dice scores by over 2.3% on ImageCHD and ranging from 7.0% to 9.5% on two MRI datasets. Considering both performance gains and computational costs, $\mathrm { V i T - B _ { 2 2 3 } }$ provides the optimal trade-off between multi-modal segmentation performance and pre-training efficiency.

3) Fine-tuning Strategy: Table Va compares three training strategies for MCSeg. Fine-tuning with a frozen ViT encoder achieves the highest Dice scores on both ImageCHD and HVSMR-2.0 datasets, proving to be the most effective strategy for downstream adaptation.

4) Loss Function: Table Vb evaluates various loss combinations. The integration of Dice, CE, and RMI losses achieves the highest Dice score (0.892) and the lowest HD95 (13.4), confirming that RMI loss effectively refines boundary accuracy beyond the standard Dice and CE combination.

TABLE IV: Impact of pre-training data scales and modalities. Downstream segmentation performance is reported alongside pre-training costs (GPU hours and peak memory usage). The first row denotes the baseline model trained from scratch.
<table><tr><td rowspan="2">Model</td><td rowspan="2">CT</td><td rowspan="2">Data CTA</td><td rowspan="2">MRI</td><td colspan="3">Dice</td><td colspan="2">Pre-training Cost</td></tr><tr><td>ImageCHD (CT)</td><td>HVSMR-2.0 (MRI)</td><td>MM-WHS (MRI)</td><td>GPU Hours (h)</td><td>Peak Memory Usage (GB)</td></tr><tr><td>ViT-B</td><td>1</td><td></td><td>一</td><td> $0 . 8 7 2 \pm 0 . 0 5 3$ </td><td> $0 . 7 4 3 \pm 0 . 1 2 7$ </td><td> $0 . 7 5 2 \pm 0 . 0 6 9$ </td><td></td><td></td></tr><tr><td> $\mathrm { V i T - B _ { 3 1 1 } }$ </td><td>223</td><td></td><td>88</td><td> $0 . 8 9 6 \pm 0 . 0 2 5$ </td><td> $\mathbf { 0 . 8 3 8 \pm 0 . 0 6 1 }$ </td><td> $0 . 8 2 1 \pm 0 . 0 4 3$ </td><td>108.38</td><td>69.37</td></tr><tr><td>ViT-B223</td><td>223</td><td></td><td>一</td><td> $0 . 8 9 5 \pm 0 . 0 2 7$ </td><td> $0 . 8 3 2 \pm 0 . 0 7 1$ </td><td> $\mathbf { 0 . 8 2 2 \pm 0 . 0 3 3 }$ </td><td>84.87</td><td>69.37</td></tr><tr><td> $\mathrm { V i T - B } _ { 7 2 3 }$ </td><td>223</td><td>500</td><td></td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 2 4 }$ </td><td> $0 . 7 8 9 \pm 0 . 1 3 0$ </td><td> $0 . 8 1 7 \pm 0 . 0 4 9$ </td><td>252.62</td><td>69.37</td></tr><tr><td> $\mathrm { V i T - B _ { 1 2 2 3 } }$ </td><td>223</td><td>1000</td><td></td><td> $0 . 8 9 5 \pm 0 . 0 2 2$ </td><td> $0 . 8 2 5 \pm 0 . 0 8 1$ </td><td> $0 . 8 0 6 \pm 0 . 0 5 7$ </td><td>423.08</td><td>69.37</td></tr></table>

TABLE V: Comparison of (a) training strategies on multimodal datasets with ${ \mathrm { V i T } } { \mathrm { - } } \mathbf { B } _ { 2 2 3 } .$ , and (b) various loss function combinations on ImageCHD dataset.  
(a)
<table><tr><td rowspan="2">Training Strategy</td><td colspan="2">Dice</td></tr><tr><td>ImageCHD</td><td>HVSMR-2.0</td></tr><tr><td colspan="2">From scratch</td><td> $0 . 7 4 3 \pm 0 . 1 2 7$ </td></tr><tr><td colspan="2">Fine-tuning (ViT unfrozen)</td><td> $0 . 8 3 1 \pm 0 . 0 5 8$ </td></tr><tr><td colspan="2">Fine-tuning (ViT frozen)</td><td> $\mathbf { 0 . 8 3 2 \pm 0 . 0 7 1 }$ </td></tr><tr><td colspan="3">(b)</td></tr><tr><td> $L _ { d i c e }$   $L _ { c e }$ </td><td> $L _ { r m i }$  一</td><td>HD95↓</td></tr><tr><td>√ X</td><td> $0 . 8 5 4 \pm 0 . 0 4 6$ </td><td> $1 9 . 0 \pm 1 0 . 0$ </td></tr><tr><td>√ √</td><td> $0 . 8 8 5 \pm 0 . 0 3 6$ </td><td> $1 4 . 3 \pm 5 . 1 6$ </td></tr><tr><td>√ X √</td><td> $0 . 8 6 2 \pm 0 . 0 5 2$ </td><td> $1 7 . 2 \pm 6 . 1 7$ </td></tr><tr><td>√</td><td>√  $\mathbf { 0 . 8 9 2 \pm 0 . 0 3 3 }$ </td><td> ${ \bf 1 3 . 4 \pm 5 . 3 5 }$ </td></tr></table>

## E. Discussion

Our experimental results demonstrate that MCSeg outperforms SOTA methods on four datasets. Notably, pre-training solely on CT images yields substantial performance gains on both CT and MRI datasets. This cross-modality transferability stems from their shared anatomical structures. Notably, MRI tasks, which often suffer from high variability due to different scanners and protocols, benefit tremendously from the robust anatomical priors learned from CT. Conversely, incorporating vatious modalities like CTA or limited MRI into pre-training provides negligible or inconsistent benefits, largely due to cross-modality domain gaps and insufficient data scales.

Regarding model architecture, we eliminated the pretraining variant by applying our pre-trained ViT weights to UNETR. The consistent superiority of MCSeg over UNETR strictly isolates and validates the efficacy of our SFP module in bridging the encoder-decoder gap. Furthermore, compared to SwinUNETR and CardiacSeg, our approach validates that a domain-specific, large-scale pre-training paradigm is practical for multi-modal success.

While large-scale 3D MAE pre-training yields substantial improvements, we acknowledge its practical limitations regarding computational and memory demands. As detailed in Table IV, pre-training requires hundreds of GPU hours on advanced hardware (e.g., 80G A100s). To ensure broad accessibility and fully support reproducibility for typical users with constrained resources, we have open-sourced our pre-trained ViT weights, the fine-tuned downstream model weights, and the complete source code, allowing researchers and clinicians to deploy the model out-of-the-box.

Furthermore, a conceptual limitation is that our pre-trained weights are explicitly cardiac-specific. In an era drawn to “universal” medical foundation models designed for all-around tasks, a specialized focus might seem restrictive. Yet, from a practical standpoint, we argue that this anatomical specialization is precisely what makes the model reliable and readily deployable in real-world clinical pathways, where targeted precision typically outweighs broad but shallow capabilities.

## V. CONCLUSION

In this study, we propose MCSeg, a volumetric transformerbased framework for multi-modal cardiac image segmentation. By integrating a 3D ViT-B/16 encoder pre-trained via masked autoencoders with a specifically designed SFP, and incorporating RMI loss during fine-tuning, MCSeg effectively captures both global anatomical context and fine-grained boundary details. While large-scale pre-training significantly elevates the performance floor, the SFP and RMI loss further raise the ceiling of segmentation precision.

Extensive evaluations confirm that MCSeg consistently surpasses existing SOTA methods on both CT and MRI datasets. Furthermore, its exceptional performance in few-shot scenarios underscores the model’s generalization capabilities and adaptability when faced with scarce annotated data.

Ultimately, beyond providing an accurate tool for cardiac segmentation, this study establishes a effective clinical pipeline. By coupling organ-specific large-scale pre-training with a tailored downstream architecture, MCSeg offers a robust and transferable paradigm for developing clinical-grade medical image analysis solutions.

## REFERENCES

[1] R. Wang, T. Lei, R. Cui, B. Zhang, H. Meng, and A. K. Nandi, “Medical image segmentation using deep learning: A survey,” IET Image Processing, vol. 16, no. 5, pp. 1243–1267, 2022.

[2] P. Bilic, P. Christ, H. B. Li, E. Vorontsov, A. Ben-Cohen, G. Kaissis, A. Szeskin, C. Jacobs, G. E. H. Mamani, G. Chartrand et al., “The liver tumor segmentation benchmark (lits),” Medical Image Analysis, vol. 84, p. 102680, 2023.

[3] N. Heller, F. Isensee, K. H. Maier-Hein, X. Hou, C. Xie, F. Li, Y. Nan, G. Mu, Z. Lin, M. Han et al., “The state of the art in kidney and kidney tumor segmentation in contrast-enhanced ct imaging: Results of the kits19 challenge,” Medical image analysis, vol. 67, p. 101821, 2021.

[4] G. Wang, W. Li, S. Ourselin, and T. Vercauteren, “Automatic brain tumor segmentation using cascaded anisotropic convolutional neural networks,” in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: Third International Workshop, BrainLes 2017, Held in Conjunction with MICCAI 2017, Quebec City, QC, Canada, September 14, 2017, Revised Selected Papers 3. Springer, 2018, pp. 178–190.

[5] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, Y. Burren, N. Porz, J. Slotboom, R. Wiest et al., “The multimodal brain tumor image segmentation benchmark (brats),” IEEE transactions on medical imaging, vol. 34, no. 10, pp. 1993–2024, 2014.

[6] C. Chen, C. Qin, H. Qiu, G. Tarroni, J. Duan, W. Bai, and D. Rueckert, “Deep learning for cardiac image segmentation: a review,” Frontiers in Cardiovascular Medicine, vol. 7, p. 25, 2020.

[7] X. Zhuang and J. Shen, “Multi-scale patch and multi-modality atlases for whole heart segmentation of mri,” Medical image analysis, vol. 31, pp. 77–87, 2016.

[8] T. K. Ghosh, M. K. Hasan, S. Roy, M. A. Alam, E. Hossain, and M. Ahmad, “Multi-class probabilistic atlas-based whole heart segmentation method in cardiac ct and mri,” IEEE Access, vol. 9, pp. 66 948–66 964, 2021.

[9] F. Xu, L. Lin, Z. Li, Q. Hong, K. Liu, Q. Wu, Q. Li, Y. Zheng, and J. Tian, “Mrdff: A deep forest based framework for ct whole heart segmentation,” Methods, vol. 208, pp. 48–58, 2022.

[10] F. yan Li, W. Li, X. Gao, and B. Xiao, “A novel framework with weighted decision map based on convolutional neural network for cardiac mr segmentation,” IEEE Journal of Biomedical and Health Informatics, vol. 26, no. 5, pp. 2228–2239, 2021.

[11] H. Cui, Y. Wang, Y. Li, D. Xu, L. Jiang, Y. Xia, and Y. Zhang, “An improved combination of faster r-cnn and u-net network for accurate multi-modality whole heart segmentation,” IEEE Journal of Biomedical and Health Informatics, 2023.

[12] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[13] Z. Ye, H. Zheng, and T. Zhang, “Cardiacseg: Customized pre-training volumetric transformer with scaling pyramid for 3d cardiac segmentation,” in International Workshop on Statistical Atlases and Computational Models of the Heart. Springer, 2023, pp. 3–14.

[14] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked au-´ toencoders are scalable vision learners,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 000–16 009.

[15] S. Zhao, Y. Wang, Z. Yang, and D. Cai, “Region mutual information loss for semantic segmentation,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[16] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[17] O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,<sup>¨</sup> “3d u-net: learning dense volumetric segmentation from sparse annotation,” in International conference on medical image computing and computer-assisted intervention. Springer, 2016, pp. 424–432.

[18] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in 2016 fourth international conference on 3D vision (3DV). IEEE, 2016, pp. 565–571.

[19] H.-Y. Zhou, J. Guo, Y. Zhang, X. Han, L. Yu, L. Wang, and Y. Yu, “nnformer: volumetric medical image segmentation via a 3d transformer,” IEEE Transactions on Image Processing, 2023.

[20] H. Peiris, M. Hayat, Z. Chen, G. Egan, and M. Harandi, “A robust volumetric transformer for accurate 3d tumor segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2022, pp. 162–172.

[21] Y. Xie, J. Zhang, C. Shen, and Y. Xia, “Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation,” in International conference on medical image computing and computer-assisted intervention. Springer, 2021, pp. 171–180.

[22] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, “Unetr: Transformers for 3d medical image segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 574–584.

[23] Y. Tang, D. Yang, W. Li, H. R. Roth, B. Landman, D. Xu, V. Nath, and A. Hatamizadeh, “Self-supervised pre-training of swin transformers for 3d medical image analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 20 730–20 740.

[24] S. Roy, G. Koehler, C. Ulrich, M. Baumgartner, J. Petersen, F. Isensee, P. F. Jaeger, and K. H. Maier-Hein, “Mednext: transformer-driven scaling of convnets for medical image segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 405–415.

[25] H. Wang, S. Guo, J. Ye, Z. Deng, J. Cheng, T. Li, J. Chen, Y. Su, Z. Huang, Y. Shen et al., “Sam-med3d: towards general-purpose segmentation models for volumetric medical images,” in European Conference on Computer Vision. Springer, 2024, pp. 51–67.

[26] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[27] H. Bao, L. Dong, S. Piao, and F. Wei, “Beit: Bert pre-training of image transformers,” in International Conference on Learning Representations, 2021.

[28] Z. Xie, Z. Zhang, Y. Cao, Y. Lin, J. Bao, Z. Yao, Q. Dai, and H. Hu, “Simmim: A simple framework for masked image modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 9653–9663.

[29] Z. Chen, D. Agarwal, K. Aggarwal, W. Safta, M. M. Balan, and K. Brown, “Masked image modeling advances 3d medical image analysis,” in Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 1970–1980.

[30] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2117–2125.

[31] A. Kirillov, R. Girshick, K. He, and P. Dollar, “Panoptic feature pyramid´ networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6399–6408.

[32] Y. Li, H. Mao, R. Girshick, and K. He, “Exploring plain vision transformer backbones for object detection,” arXiv preprint arXiv:2203.16527, 2022.

[33] J. R. Clough, N. Byrne, I. Oksuz, V. A. Zimmer, J. A. Schnabel, and A. P. King, “A topological loss function for deep-learning based image segmentation using persistent homology,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 12, pp. 8766–8778, 2020.

[34] A. Zeng, C. Wu, G. Lin, W. Xie, J. Hong, M. Huang, J. Zhuang, S. Bi, D. Pan, N. Ullah et al., “Imagecas: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images,” Computerized Medical Imaging and Graphics, vol. 109, p. 102287, 2023.

[35] X. Xu, T. Wang, J. Zhuang, H. Yuan, M. Huang, J. Cen, Q. Jia, Y. Dong, and Y. Shi, “Imagechd: A 3d computed tomography image dataset for classification of congenital heart disease,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2020, pp. 77–87.

[36] D. F. Pace, H. T. Contreras, J. Romanowicz, S. Ghelani, I. Rahaman, Y. Zhang, P. Gao, M. I. Jubair, T. Yeh, P. Golland et al., “Hvsmr-2.0: A 3d cardiovascular mr dataset for whole-heart segmentation in congenital heart disease,” Scientific Data, vol. 11, no. 1, p. 721, 2024.

[37] X. Zhuang, “Challenges and methodologies of fully automatic whole heart segmentation: a review,” Journal of healthcare engineering, vol. 4, no. 3, pp. 371–407, 2013.

[38] X. Zhuang and J. Shen, “Multi-scale patch and multi-modality atlases for whole heart segmentation of mri,” Medical image analysis, vol. 31, pp. 77–87, 2016.

[39] M. Antonelli, A. Reinke, S. Bakas, K. Farahani, A. Kopp-Schneider, B. A. Landman, G. Litjens, B. Menze, O. Ronneberger, R. M. Summers et al., “The medical segmentation decathlon,” Nature communications, vol. 13, no. 1, p. 4128, 2022.

[40] J.-X. Zhuang, X. Huang, Y. Yang, J. Chen, Y. Yu, W. Gao, G. Li, J. Chen, and T. Zhang, “Openmedia: Open-source medical image analysis toolbox and benchmark under heterogeneous ai computing platforms,” in Pattern Recognition and Computer Vision: 5th Chinese Conference, PRCV 2022, Shenzhen, China, November 4–7, 2022, Proceedings, Part I. Springer, 2022, pp. 356–367.

[41] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, “Unet++: Redesigning skip connections to exploit multiscale features in image segmentation,” IEEE Transactions on Medical Imaging, 2019.

[42] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.

[43] J. Chen, J. Mei, X. Li, Y. Lu, Q. Yu, Q. Wei, X. Luo, Y. Xie, E. Adeli, Y. Wang et al., “Transunet: Rethinking the u-net architecture design for medical image segmentation through the lens of transformers,” Medical Image Analysis, p. 103280, 2024.