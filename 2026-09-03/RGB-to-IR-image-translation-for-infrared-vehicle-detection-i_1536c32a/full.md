# RGB-to-IR image translation for infrared vehicle detection in unseen UAV domains

Thijs A. Eker<sup>a</sup>, Ella P. Fokkinga<sup>a</sup>, Jan Erik van Woerden<sup>a</sup>, Elfi I.S. Hofmeijer<sup>a</sup>, Sebastiaan P. Snel<sup>a</sup>, Klamer Schutte<sup>a</sup>, and Friso G. Heslinga<sup>a</sup>

<sup>a</sup>TNO - Intelligent Imaging, Oude Waalsdorperweg 63, the Hague, the Netherlands

## ABSTRACT

Synthetic training data has become an important resource for developing image-based AI systems in domains where measured data is scarce. This is particularly true for thermal infrared (IR) imagery, where the limited availability and diversity of real-world datasets constrain the training of IR-informed foundation models and subsequent finetuning for domain-specific tasks such as vehicle detection in air-to-ground scenarios. In contrast, large amounts of UAV-recorded RGB imagery are readily available, motivating the use of RGB-to-IR image translation to generate additional infrared training data. However, there is no one-to-one mapping between RGB and infrared appearance, as infrared signatures depend on physical characteristics that are not directly observable in RGB imagery (e.g., historical and engine-related heat signatures). This makes it challenging to learn transferable RGB-to-IR mappings.

In this work, we investigate whether generative RGB-to-IR image translation can improve infrared vehicle detection on previously unseen UAV datasets. RGB-to-IR translators are trained on paired RGB-IR imagery from multiple source datasets and applied to held-out target datasets to generate synthetic infrared training data. For each target dataset, RGB images from the training split are used as input to the RGB-to-IR translation models, while the corresponding infrared evaluation split is used only to evaluate vehicle detection performance. The evaluated methods include supervised GANs, ControlNet-based difusion models, and foundation-mode image editing using LoRA. The resulting synthetic IR imagery is used to train RF-DETR vehicle detectors. Experiments are conducted on five aerial RGB-IR vehicle datasets, with Kust4K and VTUAV serving as unseen target domains. Results show that synthetic IR consistently improves detection performance over simpler RGB and grayscale baselines. The best-performing approach, Stable Difusion 3.5 with ControlNet, improves mAP from 50.8 to 60.1 on Kust4K and from 25.6 to 38.4 on VTUAV, compared to detectors trained only on sourcedomain infrared data. Further gains are observed on VTUAV by increasing output diversity through multiple difusion seeds (+1.1 mAP) and prompt variations (+3.3 mAP). Although a gap remains to the upper bound obtained with real target-domain IR data, the results demonstrate that generative RGB-to-IR translation is an efective strategy for augmenting scarce infrared training data and improving cross-domain aerial vehicle detection.

Keywords: RGB-to-IR translation; ControlNet; Guided difusion; Generative AI; Data augmentation

## 1. INTRODUCTION

Object detectors achieve strong performance across applications such as trafic monitoring, surveillance, autonomous systems, and military reconnaissance. Although recent advances, including foundation models, have reduced the amount of task-specific training data required in some fields,<sup>1</sup> suficient annotated data remains important for specialized domains such as infrared (IR) imagery, which are less well represented in common vision foundation models.<sup>2,</sup> <sup>3</sup> Unlike RGB cameras, infrared sensors capture emitted thermal radiation. This allows objects to remain observable under challenging illumination conditions, including nighttime operation, adverse weather, and camouflage scenarios. These characteristics make infrared imagery particularly valuable for defense and security applications, where robust vehicle detection is required across a wide range of operating conditions.

![](images/73706e39fe4a644e8d05cabd49fa0cf13d16b5fc4f1b5ec50aa606ecd32d2627.jpg)  
Figure 1: Overview of the proposed workflow. Paired RGB–infrared source datasets (DroneVehicle, Caltech, and M3OT) are used to train RGB-to-IR translation models. These models are subsequently applied to RGB imagery from the training splits of target datasets unseen during translation-model training (Kust4K and VTUAV) to generate synthetic infrared training data. The generated infrared imagery is added to the detector training set, and detection performance is evaluated on real infrared imagery from the separate target-domain evaluation splits. Orange indicates training data and blue indicates evaluation data.

Despite its operational relevance, the development of high-performance infrared object detectors remains constrained by the limited availability of annotated infrared datasets. Synthetic data provides one approach to address data scarcity and has shown promising results for training vehicle detectors in RGB imagery.<sup>4,</sup> <sup>5</sup> However, generating physically realistic infrared imagery using simulation is challenging. In contrast, large amounts of real RGB imagery are readily available and already capture representative scene geometry and variability. Consequently, RGB-to-IR image translation ofers a promising approach for leveraging RGB data to increase the amount of infrared training data available for detector development.

However, generating realistic infrared imagery from RGB observations is fundamentally challenging. The relationship between visible appearance and thermal appearance is neither direct nor deterministic. Infrared images exhibit physical characteristics that are largely absent from RGB imagery, including engine heat signatures, thermal reflections, material-dependent heat retention, and temperature variations caused by historical environmental conditions. Many of these factors cannot be inferred reliably from RGB information. Furthermore, most existing RGB-to-IR translation methods rely on paired RGB-infrared training data collected within a specific dataset or operational context. In practice, paired infrared observations are often unavailable for the target scenario where synthetic data is needed, and translators frequently struggle to generalize to diferent sensors, environments, or viewpoints. This creates a domain adaptation challenge: can an RGB-to-IR translation mode trained on a limited number of source datasets produce useful infrared imagery for previously unseen domains?

In this work, we investigate whether generalized RGB-to-IR image translation can be used as a form of data augmentation for infrared vehicle detection. Figure 1 provides an overview of the proposed training and evaluation procedure. Rather than training on a single dataset, we train RGB-to-IR translation models using paired data from multiple RGB-infrared datasets with the goal of learning infrared representations that transfer across domains. The resulting translation models are then used to generate synthetic infrared imagery from RGB images originating from previously unseen datasets. We evaluate whether adding these generated images to the detector training set improves vehicle detection performance in unseen infrared domains. In addition, we compare multiple RGB-to-IR translation approaches, ranging from conventional image-to-image translation methods to recent generative AI models, and contrast them with simpler baselines such as grayscale conversion. Through these experiments, we assess both the feasibility of cross-domain RGB-to-IR translation and its practical value for improving infrared air-to-ground vehicle detection.

Specifically, we (i) construct a multi-dataset RGB-infrared training dataset for air-to-ground vehicle detection, (ii) compare grayscale-based baseline with generative adversarial network (GAN) and difusion-based methods for RGB-to-IR image translation, and (iii) evaluate the value of the generated training data for improving downstream vehicle detection performance on unseen infrared datasets.

## 2. RELATED WORKS

## 2.1 Infrared UAV vehicle detection and domain shift

Vehicle detection in UAV-based imagery is challenging due to small apparent size of objects, varying viewpoints and flight altitudes, complex backgrounds, and changing environmental conditions. Infrared imagery is frequently used in this context, because it is less dependent on visible illumination and can remain informative under lowlight and nighttime conditions. Consequently, several UAV datasets containing infrared imagery and vehicle annotations have been introduced for the development and evaluation of infrared vehicle detection systems. These datasets cover a broad range of acquisition conditions and perception tasks, including vehicle detection, tracking, and semantic segmentation.<sup>6–12</sup>

Collectively, these datasets span urban and natural environments, diferent times of day, flight altitudes, viewpoints, illumination conditions, sensors, and weather conditions. As a result, substantial domain shifts exist both within and between infrared UAV datasets. The Caltech Aerial RGB–thermal dataset highlights temporal and geographical domain shifts,<sup>7</sup> while RTDOD formulates RGB–thermal UAV detection under incrementally changing domains, including changes in weather and illumination.<sup>13</sup> These studies highlight domain shift as a challenge in infrared UAV perception, making robust cross-domain generalization dificult.<sup>14</sup>

## 2.2 RGB-to-IR image translation

Although several thermal UAV datasets have become available, collecting and annotating infrared imagery remains expensive and time-consuming. Because of this, infrared dataset availability is often insuficient for the specific domain in which a detector will ultimately be deployed. RGB-to-IR image translation provides a way to exploit more readily available RGB imagery and annotations for generating synthetic infrared training data. Early approaches mainly relied on GANs, with pix2pixHD providing a representative high-resolution conditional GAN approach.<sup>15</sup> In the aerial domain, AVIID established a dedicated benchmark for visible-to-infrared translation,<sup>16</sup> while DR-AVIT introduced a GAN-based approach specifically targeting diverse and realistic aerial infrared generation.<sup>17</sup> Lee et al. further extended GAN-based translation with an edge-guided multi-domain approach that uses reference infrared images to control thermal appearance.<sup>18</sup>

More recently, difusion-based methods have been applied to RGB-to-IR translation.<sup>19</sup> ControlNet-based approaches adapt pretrained latent difusion models using visible imagery as spatial conditioning.<sup>20,</sup> <sup>21</sup> Other methods incorporate semantic and vision-language guidance to improve infrared translation.<sup>22,</sup> <sup>23</sup> Recent work has also adapted pretrained generative image models such as FLUX for cross-spectral translation using lightweight LoRA fine-tuning.<sup>24</sup>

## 2.3 Controllable and multi-domain infrared synthesis

As discussed in the introduction, RGB-to-IR translation is a one-to-many problem, as thermal appearance is not uniquely determined by the corresponding RGB image. Difusion-based image translation naturally supports this by producing diferent plausible outputs for the same conditioning image when diferent random seeds are used.<sup>25</sup> Such stochastic variation can increase the diversity of synthetic training data, but provides no explicit contro over which thermal appearance is generated. Recent approaches therefore introduce additional conditioning mechanisms to steer this variation toward particular infrared appearances.

Lee et al. encode a reference infrared image as a style representation, allowing the same RGB content to be translated with diferent thermal appearances,<sup>18</sup> while ThermalGen explicitly disentangles thermal style to account for variations between sensors and acquisition conditions.<sup>26</sup> Text conditioning provides another mechanism for controlling the generated infrared domain. F-ViTA can generate long-wave IR (LWIR), mid-wave IR (MWIR), and near IR (NIR) translations from the same visible image,<sup>23</sup> while TherA uses thermal-aware visual-language conditioning to control factors such as time of day, weather, and object thermal state.<sup>27</sup> In this work, we investigate whether simple dataset-specific captions can provide a lightweight way to learn and select dataset-associated thermal appearances from multiple training datasets (Section 3.2: dataset-specific prompt styles).

## 2.4 Synthetic infrared data for downstream perception

The aforementioned methods focus on controlling the generated infrared appearance. However, the ultimate objective is often to improve downstream perception performance. Several works evaluate synthetic infrared imagery through its usefulness for downstream perception, including object detection, semantic segmentation, and tracking.<sup>18,</sup> <sup>28,</sup> <sup>29</sup> Lee et al. trained an object detector using translated infrared images and transferred RGB annotations, obtaining better detection performance than using alternative image translation methods.<sup>18</sup> SSL-RGB2IR similarly demonstrates improvements in object detection and semantic segmentation when models are trained with synthetically generated infrared imagery.<sup>28</sup> ControlNet-based translation has also been used to generate synthetic infrared training data for object detection.<sup>21</sup> In this work, Reinhardt et al. additionally adapt their translation model using infrared imagery from the target domain before generating synthetic training data. More recently, Clouser et al. adapt a flow-matching foundation model with LoRA and demonstrate improved infrared detection using translated RGB imagery,<sup>24</sup> but also use paired target-domain RGB-infrared data to adapt the translator.

Although these studies show that translated infrared imagery can improve downstream perception, some approaches use target-domain infrared data during translation training or adaptation. We instead compare diferent RGB-to-IR translation approaches in unseen UAV domains where RGB imagery is available for synthetic data generation, while target-domain infrared imagery is entirely withheld from translator training.

## 3. METHODS

Figure 1 provides an overview of our proposed workflow. Five paired RGB–infrared UAV datasets were collected and aligned (Section 3.1). Then, RGB-to-IR translation models are trained on source-domain RGB–infrared pairs and applied to RGB imagery from unseen target domains to generate synthetic infrared training data (Section 3.2). Finally, the generated imagery was used to augment the training of an infrared vehicle detector, whose performance was evaluated on real infrared imagery from the target domains (Section 3.3).

## 3.1 Datasets

Five paired RGB-IR UAV datasets were used: DroneVehicle,<sup>6</sup> Caltech Aerial RGB-Thermal,<sup>7</sup> M3OT,<sup>8</sup> Kust4K,<sup>9</sup> and VTUAV.<sup>10</sup> Together, these datasets cover a wide range of sensors, image resolutions, flight heights, viewpoints, and scene types. All datasets were converted to a paired object-detection format containing aligned RGB images, infrared images, and vehicle bounding-box annotations. Since the downstream detection task was formulated as single-class vehicle detection, all retained vehicle categories were mapped to a single vehicle class. For all datasets, preprocessing was performed to obtain spatially aligned RGB-infrared pairs with consistent object-detection annotations. Following alignment, all annotations were manually verified. Missing bounding boxes were added, incorrect labels were removed, and regions that could not be labeled reliably were masked out.

Table 1 summarizes the infrared sensor characteristics and dataset statistics after preprocessing. The role of each dataset in training and evaluation is described in the experimental setup in Section 3.3. Some characteristics were not reported in the original publications and are therefore marked as unknown. All datasets capture LWIR imagery in the 8–14 µm wavelength range. RGB-IR pairs were aligned with MatchAnything.<sup>30</sup> Cross-modal correspondences were used to estimate an afine transformation with RANSAC, after which RGB images and RGB labels were warped into the infrared image coordinate system. Pairs were discarded when the alignment did not meet predefined quality criteria, including a minimum number of cross-modal correspondences and maximum allowed values for rotation, shear, and translation. These thresholds varied slightly between datasets. Rejected pairs most commonly occurred in low-light RGB imagery, where insuficient reliable correspondences could be found. A brief description of each dataset, together with any dataset-specific preprocessing steps, is provided below.

Table 1: Infrared camera characteristics and dataset statistics after preprocessing. Resolution denotes the stored infrared image resolution; the native thermal sensor resolution is 640×512 for all datasets. Aligned pairs denotes the number of RGB-infrared image pairs retained after preprocessing, alignment, and manual verification. The Used for column indicates the role of each dataset in the experimental setup described in Section 3: three source datasets (DroneVehicle, Caltech, M3OT) are used for RGB-to-IR translation and detector training, while two target datasets (Kust4K, VTUAV) are reserved for cross-domain evaluation.
<table><tr><td>Dataset</td><td>Infrared camera</td><td>Flight height</td><td>Resolution</td><td>Aligned pairs</td><td>Used for</td></tr><tr><td>DroneVehicle6</td><td>Not specified</td><td>80, 100, 120 m</td><td> $6 4 0 \times 5 1 2$ </td><td>23,529</td><td>Training; in-domain test</td></tr><tr><td>Caltech Aerial RGB-Thermal7</td><td>FLIR ADK</td><td>Mostly 40 m</td><td> $9 6 0 \times 6 0 0$ </td><td>223</td><td>Training; in-domain test</td></tr><tr><td> $\mathrm { M 3 O T ^ { 8 } }$ </td><td>Mavic 3T microbolometer</td><td>100-120 m</td><td> $6 4 0 \times 5 1 2$ </td><td>952</td><td>Training; in-domain test</td></tr><tr><td> $\mathrm { K u s t 4 K ^ { 9 } }$ </td><td>Uncooled VOx sensor</td><td>Not specified</td><td> $6 4 0 \times 5 1 2$ </td><td>2,279</td><td>Close-domain test</td></tr><tr><td> $\mathrm { V T U A V ^ { 1 0 } }$ </td><td>Zenmuse H20T</td><td>5-20 m</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>1,577</td><td>Far-domain test</td></tr></table>

## 3.1.1 DroneVehicle

The DroneVehicle dataset<sup>6</sup> is the largest dataset used in this study, containing UAV imagery at a resolution of 840 × 712 pixels. Before alignment, both modalities were center-cropped to 640 × 512 pixels to remove a white border surrounding the image content. After alignment, 23,529 paired samples were retained. The dataset contains many vehicles and mostly small bounding boxes, with most objects below 100 × 100 pixels. Example aligned DroneVehicle RGB-infrared pairs are shown in Figure 2.

![](images/42831908f2b3124beb15a1566b23b0cf2df6c8bd5fe1bec4f1171ddde9e4461a.jpg)  
Figure 2: Example aligned RGB-infrared pairs from DroneVehicle after cropping and alignment. Top row shows RGB images, bottom row shows IR images.

## 3.1.2 Caltech

The Caltech Aerial RGB-Thermal dataset<sup>7</sup> contains aerial imagery at a resolution of 960 × 600 pixels. All frames containing annotated vehicles were selected together with an equal number of vehicle-free frames. Vehicle segmentation masks were converted to bounding-box annotations and manually verified after RGB-infrared alignment. This dataset is substantially smaller, with 223 paired samples remaining after matching and manual correction, but provided additional diversity through diferent backgrounds and environmental conditions. Example Caltech RGB-infrared pairs after alignment and bounding-box correction are shown in Figure 3.

![](images/90aadf45f4e8a6dcfc86db6496f2a4b5886336b478b90a3eadc891ef4092df6a.jpg)  
Figure 3: Example aligned RGB-infrared pairs from the Caltech Aerial RGB-Thermal dataset after alignment and manual bounding-box correction.

## 3.1.3 M3OT

M3OT contains paired UAV imagery captured at 640 × 512 pixels. A radial distortion correction was applied to the infrared imagery prior to alignment, and improperly annotated image regions were manually masked. Due to limited variation between consecutive frames, the dataset was subsampled at every tenth frame, resulting in 952 retained pairs. Similar to DroneVehicle, the vehicle bounding boxes are generally small due to the high flight altitude of 100 − 120m. Example M3OT RGB-infrared pairs after subsampling, distortion correction, and manual masking are shown in Figure 4.

![](images/e045f4aa8d3f71cfcc1c651c723437bd309390e2824aec0ed8dc78c971af1b71.jpg)  
Figure 4: Example aligned RGB-infrared pairs from M3OT after subsampling, radial distortion correction, alignment, and manual masking. Masked areas are indicated by the grey boxes.

## 3.1.4 Kust4K

Kust4K contains UAV imagery at 640 × 512 pixels and is visually similar to DroneVehicle and M3OT in terms of viewpoint. Radial distortion correction was applied to the infrared images before alignment. After alignment and filtering, 2,279 pairs were retained. Example Kust4K RGB-infrared pairs after distortion correction and alignment are shown in Figure 5.

## 3.1.5 VTUAV

VTUAV difers more strongly from the source datasets (DroneVehicle, Caltech, and M3OT) than Kust4K. It contains UAV imagery at 1920 × 1080 pixels, collected from a lower flight altitude, resulting in larger apparent object sizes and a diferent viewing geometry. Furthermore, as VTUAV is a tracking dataset, the original annotations follow only a single object per sequence. To make the dataset suitable for object detection, a subset of frames is sampled from each sequence and automatically pre-labeled using GroundingDINO.<sup>1</sup> The annotations are subsequently verified and corrected manually, after which the RGB-infrared pairs and labels are aligned using the same procedure described above. This resulted in 1,028 training pairs and 549 test pairs. Example VTUAV RGB-infrared pairs after sampling, alignment, prelabeling, and manual correction are shown in Figure 6.

![](images/ba876a994dc75fc733c9bf1ca9dcdb387d0842effade9943df2d7a17e9308dc0.jpg)  
Figure 5: Example aligned RGB-infrared pairs from Kust4K after radial distortion correction and alignment.

![](images/7701799fba77460d1c5fa6ac596cef1ed94a2a201d2efb54f07adef7471532d0.jpg)  
Figure 6: Example aligned RGB-infrared pairs from VTUAV after sequence sampling, alignment, GroundingDINO pre-labeling, and manual bounding-box correction.

## 3.2 RGB-to-IR translation methods

Multiple RGB-to-IR translation approaches are evaluated, covering three major families of image-to-image generation methods: ControlNet-conditioned difusion models, supervised GAN-based translation, and instructionbased foundation-model image editing. A simple grayscale conversion is included as a fourth, non-generative baseline. For methods requiring a diferent input size, images are resized with aspect-ratio preservation and padded to the required dimensions. After translation, the padding is removed and the output is resized back to the original image dimensions.

ControlNet-based translation. ControlNet<sup>31</sup> introduces additional spatial conditioning to a pretrained generative model, allowing an RGB image to guide the structure of the generated output. A similar ControlNet-based approach has previously been applied to RGB-to-IR translation by Bolanos et al.<sup>20</sup> In our setup, the RGB image is used as the conditioning input and the target modality is specified using the fixed prompt “Aerial infrared photo.”. We used two pretrained generative backbones in combination with ControlNet: Stable Difusion 3.5 Medium (SD)<sup>32</sup> and FLUX.1-dev (FLUX).<sup>33</sup> For both backbones, only the ControlNet parameters are optimized while the pretrained base model, text encoders, and variational autoencoder remain frozen. The models are trained for 60,000 steps with an efective batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 5 }$

Diferent random seeds can produce diferent infrared translations for the same RGB image. We hypothesize that using multiple independently generated variants can increase the diversity of the detector training data and improve downstream detection performance. For both SD and FLUX, we therefore compare training with one generated variant per RGB image against training with three variants generated using diferent random seeds.

Stable Difusion 3.5 Medium. Stable Difusion 3.5 Medium (SD)<sup>32</sup> is evaluated as the first ControlNet backbone. From this point onward, this method is referred to as SD-ControlNet. At inference, SD uses 20 sampling steps, a conditioning scale of 1.0, and a guidance scale of 7.0.

We additionally investigated dataset-specific prompt styles for SD. During training, each image pair is associated with a prompt of the form “Aerial infrared photo in the style of < dataset >”, where the dataset identifier corresponds to the source dataset from which the pair originates. This provides the model with information about dataset-specific imaging characteristics and allows it to learn dataset-dependent RGB-to-IR mappings. At inference, each target-domain RGB image is translated using the prompt styles of all three source datasets, producing three infrared variants. We hypothesized that this increases appearance diversity and improves detector robustness by exposing the detector to multiple plausible infrared interpretations of the same scene.

FLUX.1-dev. FLUX.1-dev (FLUX)<sup>33</sup> is evaluated as the second ControlNet backbone. From this point onward, this method is referred to as FLUX-ControlNet. Apart from the seed-variation experiment described above, no prompt-style conditioning is applied for this backbone. At inference, FLUX uses 28 sampling steps, a conditioning scale of 1.0, and a guidance scale of 3.5.

pix2pixHD. pix2pixHD<sup>15</sup> is a supervised paired image-to-image translation method. The model is trained to map aligned RGB images directly to their corresponding infrared images using the paired source-domain training data. The model is trained for 100 epochs with a batch size of 4 and a learning rate of 2 × 10<sup>−4</sup>.

FLUX.2 Klein with LoRA image editing. Stein et al., show that a domain shift can also be introduced with an difusion based editing model.<sup>34</sup> Therefore, FLUX.2 [klein] 9B Base<sup>35</sup> was fine-tuned as a representative foundation-model adaptation approach for paired RGB-to-IR image editing using low-rank adaptation (LoRA).<sup>36</sup> From this point onward, this method is referred to as FLUX-LoRA. The model is trained exclusively on paired RGB–IR images from the three source datasets using the fixed instruction “convert this RGB drone image to thermal infrared.” This approach is related to Clouser et al.,<sup>24</sup> who adapt FLUX.1 Kontext with LoRA for few-shot cross-spectral RGB-to-IR translation and subsequently use the generated imagery for infrared object detection. In contrast, our adapter is trained on the full source-domain training set and is evaluated by translating RGB imagery from target domains whose infrared imagery is entirely withheld during translator training.

The configuration uses rank-16 LoRA layers with a LoRA alpha of 16 and bf16 precision. The base model and text encoder remain frozen, and only the LoRA parameters are optimized. Training is configured for 21,000 optimization steps with a batch size of 4, a learning rate of $1 \times 1 0 ^ { - 4 }$ , and AdamW 8-bit optimization using 512- and 768-pixel resolution buckets. At inference time, target-domain RGB training images are provided as input images together with the same instruction. Images are generated at 640 × 512 pixels using 30 inference steps, a guidance scale of 4, and a LoRA scale of 1.0.

## 3.3 Experimental setup

The five datasets serve diferent roles in the experimental setup. DroneVehicle, M3OT, and Caltech are used as source datasets, providing paired RGB-infrared data for training the RGB-to-IR translation models and the baseline infrared object detector. Kust4K and VTUAV are held out during translator training and serve as target datasets. Kust4K is considered a close-domain target dataset because of its similarity in viewpoint and image characteristics to the source datasets, whereas VTUAV represents a far-domain target dataset, featuring imagery captured from substantially lower flight altitudes.

The baseline detector (Section 3.3.1) is trained exclusively on infrared imagery from the source datasets and evaluated on three test scenarios: the in-domain source test sets, the held-out Kust4K test set, and the held-out VTUAV test set. This establishes performance when no target-domain data are used for detector training.

The RGB images from the target-domain datasets Kust4K and VTUAV train split are translated into synthetic infrared imagery and added to the source-domain infrared training data. For comparison, two simpler alternatives are considered that use the same target-domain RGB images directly, either as RGB images or converted to grayscale. The latter serves as a simple infrared proxy that preserves scene geometry without requiring a learned translation model.

In addition, we evaluated whether grayscale and generative translation provide complementary benefits by training detectors on combinations of grayscale and SD-ControlNet-generated imagery. Three combinations are considered: single-seed SD-ControlNet with grayscale augmentation, multi-seed SD-ControlNet with grayscale augmentation, and prompt-style SD-ControlNet with grayscale augmentation. Finally, we performed an upperbound experiment in which the real target-domain infrared imagery is added during training.

## 3.3.1 Object detector

RF-DETR is used as the downstream vehicle detector for all experiments, because it combines a modern transformer-based detector with a strong pretrained visual backbone, provides fast inference, and a straightforward fine-tuning workflow. RF-DETR is a real-time DETR-style transformer detector developed by Roboflow, using DINOv2 as its vision backbone.<sup>37</sup> All experiments are based upon RF-DETR Medium and the detector is trained for a single class: vehicle. Each detector configuration is trained for 20 epochs and uses 5 warm-up epochs. For experiments that increase the amount of training data through multiple generated variants per image, such as the multi-seed, prompt-style, or combined experiments, the number of training and warm-up epochs is reduced proportionally to ensure that the total number of training iterations remains approximately constant. The model is always initialized from pretrained RF-DETR Medium weights and the experiments are repeated using three random seeds. Images are resized to an input resolution of 640×640 when needed, and the batch size is 12. The main learning rate is $1 \times 1 0 ^ { - 4 }$ and the encoder learning rate is $1 . 5 \times 1 0 ^ { - 4 }$ . The default augmentation setup for infrared drone imagery applies horizontal flipping, small brightness and contrast perturbations, and mild afine transformations. Additionally, random image inversion is included during training, so the detector sees both white-hot and black-hot style infrared appearances equally.

The evaluation is performed at the downstream task level using mean Average Precision (mAP) on real infrared test imagery from the target-domain datasets Kust4K and VTUAV. We report the COCO-style mAP<sub>50:95</sub> metric, averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05,<sup>38</sup> and averaged over three detector training seeds.

## 4. RESULTS

Figures 7 and 8 show example translations on the held-out Kust4K and VTUAV target datasets. Overall, all generative methods preserve the geometric structure of the input image well. Vehicles, road layouts, vegetation, and buildings remain spatially consistent with the RGB input. For Kust4K, the diferences between the grayscale baseline and the learned translation methods are particularly visible in the low-light example shown in the second row of Figure 7. Because grayscale conversion directly preserves visible-spectrum intensities, the resulting image remains unrealistically dark and some vehicles become dificult to distinguish. The learned RGB-to-IR models generally introduce stronger thermal contrast, although this varies between methods and examples; in the lowlight Kust4K example, FLUX-LoRA preserves the visibility of the dark vehicles particularly well. The generated infrared imagery exhibits substantial appearance variation across methods. For example, roads surfaces may appear either relatively hot or relatively cold, while vehicles and surrounding vegetation also vary in thermal contrast.

Figure 9 illustrates the efect of varying the difusion seeds for SD-ControlNet. While object geometry and scene layout remain unchanged, the thermal appearance varies across generations. In particular, the seeds difer in the relative temperature of the road and vehicles: some generations produce warmer backgrounds with comparatively cooler vehicles, while others increase the vehicle-to-background contrast. Figure 10 shows the efect of dataset-specific prompt styles. The Caltech style produces weak vehicle contrast. The DroneVehicle style resembles the default SD-ControlNet appearance most closely. In contrast, the M3OT style produces darker backgrounds and brighter foreground objects, reflecting characteristics observed in the M3OT dataset and resulting in an appearance that is qualitatively closer to the real VTUAV infrared imagery.

![](images/bcb6c63295ec78e9b90f0854ce491987a2b6ec9678a0d59ef833473082d4767c.jpg)  
Figure 7: Three RGB-to-IR translation example scenes of the Kust4K training set. From left to right, each row shows an RGB image, the translations generated by the grayscale baseline, SD-ControlNet, FLUX-ControlNet, pix2pixHD, and FLUX-LoRA, and the corresponding real infrared images.

![](images/bc4fddb095df4b88668fcd82f6f0d4bd7af301500f178b7d11cec63ca683256f.jpg)  
Figure 8: Three RGB-to-IR translation example scenes of the VTUAV training set. From left to right, each row shows an RGB image, the translations generated by the grayscale baseline, SD-ControlNet, FLUX-ControlNet, pix2pixHD, and FLUX-LoRA, and the corresponding real infrared image.

## 4.1 Object detection

Tables 2 and 3 summarize the object detection results. Without any target-domain data, performance is substantially lower on VTUAV (25.6 mAP) than on Kust4K (50.8 mAP). Adding the real infrared target-domain training data yields large improvements on both datasets, increasing performance to 67.5 and 52.3 mAP, respectively.

Adding target-domain RGB images improves performance by +6.3 and +4.5 mAP on Kust4K and VTUAV, respectively. Converting these images to grayscale provides a further improvement, resulting in total gains of +8.5 and +8.1 mAP.

Among the learned translation methods, SD-ControlNet performs best overall, improving the source-only baseline by +9.2 mAP on Kust4K and +11.7 mAP on VTUAV with a single generated variant. Using three seeds provides little additional benefit on Kust4K but increases the VTUAV gain to +12.8 mAP. Datasetspecific prompt styles further improve VTUAV to +15.0 mAP, while providing no additional benefit on Kust4K. FLUX-ControlNet is less efective on Kust4K but remains competitive on VTUAV, particularly when using three seeds. pix2pixHD provides smaller gains on both datasets, whereas FLUX-LoRA reduces performance below the source-only baseline. As illustrated in Figure 11, FLUX-LoRA can alter the position of vehicles during translation, causing the transferred RGB annotations to become misaligned.

Seed 1  
Real RGB  
Seed 2  
Seed 3  
Real infrared  
![](images/21277fd556404a6d68abb02dca37f04bb089ce90bb7fcdcdca6e24fa20d71648.jpg)  
Figure 9: Efect of difusion seed variation for SD-ControlNet on the VTUAV training set. From left to right, each row shows an RGB image, SD-ControlNet outputs generated with diferent random seeds, and the rea infrared target.

Real RGB  
Caltech prompt style  
DroneVehicle prompt style  
M3OT prompt style  
Real infrared  
![](images/49df9248f175198b358b45313faa62de55fdbdc42910dae71cb6133284e2ffef.jpg)  
Figure 10: Efect of dataset-specific prompt styles for SD-ControlNet on the VTUAV training set. From left to right, each row shows an RGB image, the SD-ControlNet translations generated using prompts corresponding to the three source datasets (DroneVehicle, Caltech, and M3OT), and the real infrared target.

![](images/dee6ec596087d48792b083f1ec89320fce043f05b51020ec7236550cf0f7f760.jpg)  
(a) RGB input

![](images/3fca8ad99fba25f00b0740fec310fef4d18226c14b9730ca921ec6a5ef1a7927.jpg)  
(b) SD-ControlNet

![](images/337d9471fe437f67b8453effb4ee562dc118d5e5230b47d10e8263626a3c0592.jpg)  
(c) FLUX-LoRA  
Figure 11: Example of spatial inconsistency in the FLUX-LoRA translation. The bounding boxes correspond to the original RGB annotations. SD-ControlNet preserves the vehicle locations, while FLUX-LoRA shifts the vehicles, causing the transferred annotations to become misaligned.

Table 2: Efect of adding Kust4K and VTUAV target-domain training data to the RF-DETR multi-source baseline. Values are mean ± standard deviation over three completed runs, multiplied by 100. Delta values (∆) are reported relative to the multi-source baseline (“None”).
<table><tr><td>Test dataset</td><td colspan="3">Kust4k</td><td colspan="3">VTUAV</td></tr><tr><td>Added training dataset</td><td></td><td>mAP [±std]</td><td>∆mAP</td><td></td><td>mAP [±std]</td><td>∆mAP</td></tr><tr><td>None</td><td>50.8</td><td>[±0.1]</td><td>0.0</td><td>25.6</td><td>[±0.3]</td><td>0.0</td></tr><tr><td>Real RGB</td><td>57.1</td><td>[±1.0]</td><td>+6.3</td><td>30.1</td><td>[±0.3]</td><td>+4.5</td></tr><tr><td>Grayscale</td><td>59.3</td><td>[±0.7]</td><td>+8.5</td><td>33.7</td><td>[±1.4]</td><td>+8.1</td></tr><tr><td>SD-ControlNet</td><td>60.0</td><td>[±0.2]</td><td>+9.2</td><td>37.3</td><td>[±0.8]</td><td>+11.7</td></tr><tr><td>SD-ControlNet, three seeds</td><td>60.1</td><td>[±0.5]</td><td>+9.3</td><td>38.4</td><td>[±0.4]</td><td>+12.8</td></tr><tr><td>SD-ControlNet, prompt styles</td><td>59.3</td><td>[±0.2]</td><td>+8.5</td><td>40.6</td><td>[±0.2]</td><td>+15.0</td></tr><tr><td>FLUX-ControlNet</td><td>54.8</td><td>[±1.8]</td><td>+4.0</td><td>38.6</td><td>[±0.6]</td><td>+13.0</td></tr><tr><td>FLUX-ControlNet, three seeds</td><td>57.1</td><td>[±0.4]</td><td>+6.3</td><td>39.3</td><td>[±0.7]</td><td>+13.7</td></tr><tr><td>pix2pixHD</td><td>56.3</td><td>[±0.2]</td><td>+5.5</td><td>32.0</td><td>[±0.5]</td><td>+6.4</td></tr><tr><td>FLUX-LoRA</td><td>46.3</td><td>[±1.3]</td><td>-4.5</td><td>21.1</td><td>[±2.3]</td><td>-4.5</td></tr><tr><td>Real infrared</td><td></td><td>67.5 [±0.4]</td><td>+16.7</td><td>52.3 [±0.3]</td><td></td><td>+26.7</td></tr></table>

Performance on all in-domain source test sets remains largely stable across the target-domain augmentation experiments. The source-only baseline achieves 63.3±0.1 mAP. Adding real target-domain infrared data improves this performance to 63.8±0.0 mAP, while the RGB, grayscale, and synthetic infrared variants lead to only small decreases, with a maximum reduction of 0.4 mAP.

Table 3 reports the object detection results when grayscale and SD-ControlNet are combined. Combining grayscale with SD-ControlNet improves over grayscale alone in all settings. On Kust4K, the strongest result is obtained with the simplest SD-ControlNet configuration, increasing performance from 59.3 to 62.4 mAP, while additional seeds or prompt styles do not provide further gains. On VTUAV, all combinations perform similarly at around 40 mAP, suggesting that the main benefit comes from combining the two representations rather than from adding more generative variation.

Table 3: Efect of combining grayscale and SD-ControlNet target-domain training data on the RF-DETR multisource baseline. Values are mean ± standard deviation over three completed runs, multiplied by 100. Delta values (∆) are reported relative to the “Grayscale” baseline.
<table><tr><td>Test dataset</td><td colspan="2">Kust4K</td><td colspan="3">VTUAV</td></tr><tr><td>Added training dataset</td><td>mAP [±std]</td><td>∆mAP</td><td>mAP [±std]</td><td></td><td>∆mAP</td></tr><tr><td>Grayscale</td><td>59.3 [±0.7]</td><td>0.0</td><td></td><td>33.7 [±1.4]</td><td>0.0</td></tr><tr><td>Grayscale + SD-ControlNet</td><td>62.4 [±0.3]</td><td>+3.1</td><td></td><td>40.2 [±0.1]</td><td>+6.5</td></tr><tr><td>Grayscale + SD-ControlNet, three seeds</td><td>61.4 [±0.1]</td><td>+2.1</td><td>39.8</td><td>[±0.2]</td><td>+6.1</td></tr><tr><td>Grayscale + SD-ControlNet, prompt styles</td><td>61.2 [±0.5]</td><td>+1.9</td><td></td><td>41.6 [±1.1]</td><td>+7.4</td></tr><tr><td>Real infrared</td><td>67.5 [±0.4]</td><td></td><td>+8.2</td><td>52.3 [±0.3]</td><td>+18.6</td></tr></table>

## 5. DISCUSSION

In this work, we investigated the extent to which synthetic infrared imagery, generated from target-domain RGB images using RGB-to-IR translation, can be used as a form of data augmentation in the absence of enough target-domain infrared training images.

## 5.1 Benefits of target-domain data under domain shift

The source-only baseline already highlights the diferent levels of domain shift between the two target datasets. The detector achieves 50.8 mAP on the close-domain Kust4K, and performance drops to 25.6 mAP on the fardomain VTUAV. This confirms that VTUAV constitutes the more challenging, far-domain, transfer scenario. Adding target-domain RGB images improves performance by +6.3 and +4.5 mAP, respectively, while grayscale conversion increases these gains to +8.5 and +8.1 mAP. This indicates that a substantial part of the benefit comes from exposing the detector to target-domain scene content, including object scales, viewpoints, backgrounds, and annotation distributions.

The strong performance of grayscale conversion suggests that part of the domain gap is unrelated to infrared appearance. By introducing target-domain geometry and scene content without requiring a learned translation model, grayscale imagery already provides most of the genuinely new information available from the target domain. This observation is consistent with the fact that both the detector and the translation models learn their notion of infrared appearance from the same source-domain infrared datasets (Figure 1). A translation method may render a vehicle with a more realistic infrared appearance, but similar infrared vehicle appearances have already been seen in the source-domain infrared training data.

Nevertheless, learned RGB-to-IR translation provides an additional benefit. SD-ControlNet with three seeds improves performance by +9.3 mAP on Kust4K and +12.8 mAP on VTUAV, outperforming both the RGB and grayscale baselines. The improvement over grayscale is small on Kust4K (+0.8 mAP), but substantially larger on VTUAV (+4.7 mAP). This suggests that learned translation, and thus more realistic infrared appearance, becomes increasingly important as the domain shift grows. Although using real target-domain infrared remains substantially better (+16.7 and +26.7 mAP), synthetic target-domain infrared recovers a part of the performance lost under domain shift using only target-domain RGB imagery.

The low performance of pix2pixHD may be related to the inherently one-to-many nature of RGB-to-IR translation. Visible appearance does not uniquely determine infrared appearance: from RGB imagery alone, it is generally unknown which objects or object parts are warm and how strongly they contrast with the background. In our paired setup, pix2pixHD learns a predominantly deterministic mapping from RGB to infrared and therefore tends toward a single source-derived translation for a given visible input.

FLUX-LoRA performs substantially worse than the other translation methods, reducing detection performance to below the source-only baseline. Qualitative inspection suggests that the main problem is spatial inconsistency: the model sometimes changes the position of vehicles instead of only translating their appearance (Figure 11). Since the original RGB bounding boxes are reused for the generated images, these changes introduce incorrect labels, which is especially problematic for the small objects in aerial imagery. This does not necessarily mean that foundation-model image editing is unsuitable for RGB-to-IR translation. Clouser et al.<sup>24</sup> successfully use a LoRA-adapted foundation model for cross-spectral translation and downstream detection. In our setting, however, stronger preservation of the input geometry appears to be necessary.

## 5.2 Role of infrared appearance and generative diversity

Generating multiple translations with diferent random seeds provides only a limited benefit. On Kust4K, increasing the number of SD-ControlNet seeds from one to three has a negligible efect (60.0 versus 60.1 mAP), indicating that the close-domain setting already contains suficient variability. On VTUAV, however, performance slightly improves from 37.3 to 38.4 mAP. This suggests that exposing the detector to multiple plausible infrared appearances might be useful when the domain gap increases, although random seed variation does not provide explicit control over which infrared appearance characteristics are changed.

Variation can be created in a more controlled manner by dataset-specific prompt conditioning, and a larger improvement is obtained in this case. By associating each source dataset with a diferent prompt during training, the model learns several dataset-associated infrared styles that can be selected at inference time. This also gives less represented source styles a more explicit role at inference, rather than allowing the appearance of the much larger DroneVehicle dataset to dominate the translation. Using the prompt styles increases VTUAV performance to 40.6 mAP, suggesting that explicitly varying the learned infrared style is more useful than relying on random seed variation alone. Qualitatively, the M3OT-associated style produces an infrared appearance that resembles VTUAV more closely, despite substantial diferences in viewpoint, object scale, and scene content (Figure 10). This indicates that infrared characteristics learned from one dataset can remain useful even when applied to very diferent RGB scenes. In contrast, the Caltech-associated style often produces weaker vehicle-to-background contrast, which may reflect the limited similarity between the Caltech training imagery and the highway scenes present in VTUAV.

The combination experiments suggest that grayscale and SD-translated infrared are to a certain extent complementary methods. Grayscale images provide the target-domain geometry and scene content without introducing translation artifacts, while SD-ControlNet adds an infrared-like appearance learned from the source datasets. This combination is more efective than either representation alone, particularly on VTUAV. However, increasing the number of translated variants through multiple seeds or prompt styles provides little or inconsistent additional improvement once grayscale is included. This suggests that most of the benefit comes from combining grayscale and translated infrared representations, rather than simply adding more generated samples.

## 5.3 Limitations and future work

A fundamental limitation of RGB-to-IR translation is that thermal appearance cannot be uniquely inferred from RGB imagery. Efects such as engine heat, object temperature, and environmental thermal history are not directly visible, so the translator can only reproduce thermal patterns learned from the source data. This is further limited by the relatively small and imbalanced set of source infrared datasets, which is dominated by DroneVehicle. A larger and more balanced collection could expose the model to a wider range of sensors, day/night conditions, object thermal states, and other infrared appearances. An alternative route is to add a limited amount of in-domain real data to the synthetic data, as was shown to be an efective strategy for vehicle detection in RGB imagery.<sup>39–41</sup>

Future work should therefore focus on representing thermal variation more explicitly. The prompt-style experiment suggests that controlled variation can be more useful than relying on random seed variation alone, and this could be extended beyond dataset-level styles toward specific thermal characteristics such as sensor type, time of day, road temperature, or vehicle operating state. At the same time, grayscale was deliberately kept as a simple baseline in this study. For a fairer comparison with increasingly sophisticated generative methods, future work should also investigate stronger non-generative alternatives, such as semantic-aware grayscale augmentation.<sup>12</sup>

Finally, the evaluation is limited to two unseen UAV target domains. Additional datasets covering diferent sensors, environments, flight configurations, and thermal conditions would be needed to determine how broadly the findings generalize. Real target-domain infrared data still substantially outperforms all synthetic alternatives. RGB-to-IR translation should therefore be viewed as an intermediate strategy: when target-domain RGB imagery and annotations are available but corresponding infrared training data are scarce or unavailable, synthetic translation can recover a substantial part of the lost cross-domain detection performance.

## REFERENCES

[1] Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., and Zhang, L., “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” in [Computer Vision – ECCV 2024], 38–55, Springer Nature Switzerland, Cham (2024).

[2] Medeiros, H. R., Belal, A., Muralidharan, S., Granger, E., and Pedersoli, M., “Visual modality prompt for adapting vision-language object detectors,” in [Proceedings of the IEEE/CVF International Conference on Computer Vision], 2172–2182 (2025).

[3] Moshtaghi, M., Khajavi, S. H., and Pajarinen, J., “RGB-Th-Bench: A dense benchmark for visual-thermal understanding of vision language models,” arXiv preprint arXiv:2503.19654 (2025).

[4] Ruis, F. A., Liezenga, A. M., Heslinga, F. G., Ballan, L., den Hollander, R. J., van Leeuwen, M. C., Masinia, B., Dijk, J., and Huizinga, W., “Improving object detector training on synthetic data by starting with a strong baseline methodology,” in [Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications II], 13035, SPIE Defense + Commercial Sensing (2024).

[5] Eker, T. A., Heslinga, F. G., Ballan, L., den Hollander, R. J., and Schutte, K., “The efect of simulation variety on a deep learning-based military vehicle detector,” in [Artificial Intelligence for Security and Defence Applications], 12742, 183–196, SPIE Sensors + Imaging (2023).

[6] Sun, Y., Cao, B., Zhu, P., and Hu, Q., “Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning,” IEEE Transactions on Circuits and Systems for Video Technology 32(10), 6700–6713 (2022).

[7] Lee, C., Anderson, M., Ranganathan, N., Zuo, X., Do, K., Gkioxari, G., and Chung, S.-J., “Caltech aerial rgb-thermal dataset in the wild,” in [European Conference on Computer Vision], 236–256, Springer (2025).

[8] Nie, Z., Xue, L., Fang, Z., Ren, J., Wei, Y., and Zheng, J., “M3ot: A multi-drone multi-modality dataset for multi-object tracking,” Scientific Data 12(1927) (2025).

[9] Ouyang, J., Wang, Q., Shang, Y., Jin, P., Zhong, H., Zhou, L., and Shen, T., “An rgb-tir dataset from uav platform for robust urban trafic scenes semantic segmentation,” Scientific Data 12(1701) (2025).

[10] Zhang, P., Zhao, J., Wang, D., Lu, H., and Ruan, X., “Visible-thermal uav tracking: A large-scale benchmark and new baseline,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition], (2022).

[11] Suo, J., Wang, T., Zhang, X., Chen, H., Zhou, W., and Shi, W., “HIT-UAV: A high-altitude infrared thermal dataset for unmanned aerial vehicle-based object detection,” Scientific Data 10, 227 (2023).

[12] D, M., Sikdar, A., Gurunath, P., Udupa, S., and Sundaram, S., “Saga: Semantic-aware gray color augmentation for visible-to-thermal domain adaptation across multi-view drone and ground-based vision systems,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops], 4617–4627 (June 2025).

[13] Feng, H., Zhang, L., Zhang, S., Wang, D., Yang, X., and Liu, Z., “RTDOD: A large-scale rgb-thermal domain-incremental object detection dataset for UAVs,” Image and Vision Computing 140, 104856 (2023).

[14] Hofmeijer, E. I., Fokkinga, E. P., Heslinga, F. G., Schutte, K., and Karlholm, J. M., “Domain generalization and synthetic data in object detection: the enabler, the probe, and the gap,” in [Artificial Intelligence for Security and Defence Applications IV], 14290, 26, SPIE Sensors + Imaging (2026).

[15] Wang, T.-C., Liu, M.-Y., Zhu, J.-Y., Tao, A., Kautz, J., and Catanzaro, B., “High-resolution image synthesis and semantic manipulation with conditional gans,” in [Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition], (2018).

[16] Han, Z., Zhang, Z., Zhang, S., Zhang, G., and Mei, S., “Aerial visible-to-infrared image translation: Dataset, evaluation, and baseline,” Journal of Remote Sensing 3, 0096 (2023).

[17] Han, Z., Zhang, S., Su, Y., Chen, X., and Mei, S., “DR-AVIT: Toward diverse and realistic aerial visibleto-infrared image translation,” IEEE Transactions on Geoscience and Remote Sensing 62, 1–13 (2024).

[18] Lee, D.-G., Jeon, M.-H., Cho, Y., and Kim, A., “Edge-guided multi-domain RGB-to-TIR image translation for training vision tasks with challenging labels,” in [2023 IEEE International Conference on Robotics and Automation (ICRA)], 8291–8298 (2023).

[19] Fokkinga, E. P., Eker, T. A., van Woerden, J. E., Witon, J.-M., Stallinga, S. O., Visser, A., Schutte, K., and Heslinga, F. G., “Generative AI methods for synthesis of image data to train AI for automated scene understanding in a military context: a review of opportunities,” in [Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications III], 13459, 9–31, SPIE Defense + Commercia Sensing (2025).

[20] Bolanos, L., Urwin, G., Walsh, R., Clark, R., Hamari, J., and Zardadi, M., “EO2IR ControlNet: Synthetic infrared image generation for automatic target recognition: Experimental results in MIST,” in [Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications III], Manser, K. E., Howell, C. L., Rao, R. M., De Melo, C., and Prussing, K. F., eds., 13459, 134590V, SPIE (2025).

[21] Reinhardt, C. N., Anderson, C., and Schenck, E., “V2IR-CnLDM: A generative visible-to-infrared image translation using ControlNet-guided conditional latent difusion model,” Optical Engineering 64(9), 092206 (2025).

[22] Ran, L., Wang, L., Wang, G., Wang, P., and Zhang, Y., “DifV2IR: Visible-to-infrared difusion model via vision-language understanding,” (2025).

[23] Paranjape, J. N., De Melo, C. M., and Patel, V. M., “F-ViTA: Foundation model guided visible to infrared translation,” in [Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)], 5633–5642 (March 2026).

[24] Clouser, M., Khezeli, K., and Kalantari, J., “Few-shot LoRA adaptation of a flow-matching foundation model for cross-spectral object detection,” in [Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops], 1531–1539 (March 2026).

[25] Saharia, C., Chan, W., Chang, H., Lee, C. A., Ho, J., Salimans, T., Fleet, D. J., and Norouzi, M., “Palette: Image-to-image difusion models,” in [ACM SIGGRAPH 2022 Conference Proceedings], 1–10, Association for Computing Machinery (2022).

[26] Xiao, J., Nayak, R., Zhang, N., Toertei, D., and Loianno, G., “Thermalgen: Style-disentangled flowbased generative models for RGB-to-thermal image translation,” in [The Thirty-ninth Annual Conference on Neural Information Processing Systems], (2025).

[27] Lee, D.-G., Rhee, T. H., Jang, H., Shin, Y.-S., Shin, U., and Kim, A., “TherA: Thermal-aware visuallanguage prompting for controllable RGB-to-thermal infrared translation,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)], 36803–36813 (2026).

[28] Sikdar, A., Saadiyean, Q., Anand, P., and Sundaram, S., “SSL-RGB2IR: Semi-supervised RGB-to-IR imageto-image translation for enhancing visual task training in semantic segmentation and object detection,” in [2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)], 5017–5023 (2024).

[29] Zhang, L., Gonzalez-Garcia, A., van de Weijer, J., Danelljan, M., and Khan, F. S., “Synthetic data generation for end-to-end thermal infrared tracking,” IEEE Transactions on Image Processing 28(4), 1837–1850 (2019).

[30] He, X., Yu, H., Peng, S., Tan, D., Shen, Z., Bao, H., and Zhou, X., “Matchanything: Universal crossmodality image matching with large-scale pre-training,” arXiv preprint arXiv:2501.07556 (2025).

[31] Zhang, L., Rao, A., and Agrawala, M., “Adding conditional control to text-to-image difusion models,” in [Proceedings of the IEEE/CVF International Conference on Computer Vision], 3836–3847 (2023).

[32] Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., and Rombach, R., “Scaling rectified flow transformers for high-resolution image synthesis,” arXiv preprint arXiv:2403.03206 (2024).

[33] Labs, B. F., “FLUX.” https://github.com/black-forest-labs/flux (2024).

[34] Stein, I. D., Eker, T. A., Snel, S. P., Schutte, K., Ambrogioni, L., and Heslinga, F. G., “Generative image editing for camouflaged military vehicle detection in low-data regimes,” in [Artificial Intelligence for Security and Defence Applications IV], 14290, 31, SPIE Sensors + Imaging (2026).

[35] Black Forest Labs, “FLUX.2 [klein] 9b base.” Hugging Face model card (2026).

[36] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W., “LoRA: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685 (2021).

[37] Roboflow, “RF-DETR: Real-time object detection, instance segmentation, and keypoint detection.” GitHub repository (2026). Accessed 20 July 2026.

[38] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L., “Microsoft coco: Common objects in context,” in [European Conference on Computer Vision], 740–755, Springer (2014).

[39] Heslinga, F. G., Eker, T. A., Fokkinga, E. P., van Woerden, J. E., Ruis, F. A., den Hollander, R. J. M., and Schutte, K., “Combining simulated data, foundation models, and few real samples for training object detectors,” in [Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications II], 13035, SPIE Defense + Commercial Sensing (2024).

[40] Heslinga, F. G., Fokkinga, E. P., Eker, T. H., Liezenga, A. M., den Hollander, R. J. M., Oppeneer, V. O., van Heteren, A. M., van Vossen, R., Kuijf, H. J., van de Sande, J. J. M., van der Burg, D. W., Weyland, L. F., Henderson, H. C., Schadd, M. P. D., and Schutte, K., “On the use of simulated data for target recognition and mission planning,” in [Artificial Intelligence for Security and Defence Applications II], 13206, SPIE Sensors + Imaging (2024).

[41] Snel, S. P., Eker, T. A., Fokkinga, E. P., Visser, A., Schutte, K., and Heslinga, F. G., “Data augmentation for vehicle detection with difusion-based object inpainting,” in [Artificial Intelligence for Security and Defence Applications III], 13679, 294–307, SPIE Sensors + Imaging (2025).