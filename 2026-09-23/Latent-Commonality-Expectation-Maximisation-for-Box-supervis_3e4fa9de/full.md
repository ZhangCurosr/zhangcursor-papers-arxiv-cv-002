# Latent Commonality Expectation-Maximisation for Box-supervised Tree Crown Instance Segmentation

Thomas Pitts<sup>∗</sup>, Kunqi Li, Bin Liang

Department of Data Science, University of Technology Sydney

## Abstract

Individual tree crown segmentation from aerial imagery underpins tree-level carbon accounting, biodiversity, and restoration monitoring at landscape scale. However, existing models are predominantly trained on dense canopy forest imagery and degrade in savannah and drylands, where tree crowns are sparse, of variable appearance, and significantly underrepresented in annotated benchmarks. These models also typically depend on costly polygon crown annotations. We introduce LACE (LAtent Commonality Expectation-maximisation), a box-supervised instance segmentation model, evaluated on 0.1 m/px aerial RGB tree crown imagery. LACE uses a frozen DINOv3-web ViT-L/16 encoder, applied at four spatial ofsets and interlaced into a denser feature grid, with a lightweight CenterNet-style detection head trained solely on bounding boxes. We use expectation-maximisation to separate recurring appearance, the “treeness”, within bounding boxes from surroundings. On the OAM-TCD benchmark test set, LACE reaches a mask AP of 0.663 ± 0.001 (3 seeds) trained on 900 box-annotated images and without mask annotations, above the 0.626 scored by Restor Foundation’s released mask-supervised Mask R-CNN, which was trained on the full ∼4.2k image set. On a sparse-canopy holdout set, mask AP rises to 0.691 versus 0.612 for a mask-supervised baseline Detectree2. On NeonTreeEvaluation, using the oficial evaluation code, LACE reaches 0.728 ± 0.003 F1@0.4 (5 seeds) from 23,424 hand-annotated RGB boxes alone, matching the authors’ DeepForest model’s published 0.719, while using under 0.1% of its training annotations i.e. without the LiDAR-derived 30M-crown pretraining set. By leveraging frozen self-supervised features, LACE matches or surpasses fully-supervised specialist baselines from boxes alone, removing the need for polygon annotation in tree crown instance segmentation for sparse-canopy environments where labelled data is scarce.

Keywords: Tree crown instance segmentation, self-supervised vision transformers, remote sensing, carbon accounting, DINOv3, expectation-maximisation, box-supervised instance segmentation

## 1 Introduction

Accurate measurement of individual tree crowns can serve as a proxy for woody biomass estimation via allometries [1, 2] and thereby enable monitoring of carbon sequestration within areas containing trees since carbon content can be derived from woody biomass. Measurement techniques in this domain have traditionally focused on dense canopy areas and forests, but there is an increasing body of literature that highlights the limitations of this approach as distribution, density, cover, and carbon content of trees in sparse-canopy biomes (biomes where trees occur at low canopy cover: savannah, shrubland and arid to semi-arid dryland; Table 6 lists the WWF biomes we treat as such) are not well understood at sub-continental to continental scales [2–4]. Recent studies highlight that the global contribution of sparse-canopy biomes as a carbon sink is underestimated due to the underestimation of tree mass in sparse-canopy areas [2, 5–7]. Moreover, approximately 1/4 of all trees globally are located in grassland, savanna, desert and tundra biomes [8], and 29% of Africa’s tree cover lies outside areas classified as forest [9].

Several conventional methodologies have been applied to the measurement of tree cover and biomass, including field inventories of plots [10], statistical sampling [11] and hybrid solutions pairing plots with Remote Sensing (RS) covariates such as optical, multispectral, and LiDAR sensors [12–14]. Machine learning and deep learning applied to satellite and UAV data have since shifted the field from aggregate cover fractions toward mapping every individual tree: Brandt et al. [5] mapped over 1.8 billion crowns across 1.3 million km<sup>2</sup> of the Sahara and Sahel from 0.5 m imagery; Tucker et al. [2] extended this to carbon estimation over 9.9 billion trees. Reiner et al. [9] found that 29% of Africa’s tree cover lies outside areas previously classified as forest. Mugabowindekwe et al. [15] produced nation-wide tree-level carbon estimates for Rwanda using 0.25 m/px aerial RGB imagery and found that 48.6% of Rwanda’s aboveground carbon stock in trees was located outside forests. They also found that “even very detailed manual forest delineation. . . missed 38.4% of the isolated trees in Rwanda, which account for 25.5% of the national aboveground carbon stocks” (emphasis added). These studies establish that individual crown detection in open, scattered-tree environments is tractable, but depend on (often proprietary) sub-metre imagery and labelled data, motivating more transferable and label-eficient methods.

Modern deep learning approaches focus on instance segmentation on RGB imagery (Table 1 summarises representative methods): DeepForest [16] established a RetinaNet-based baseline evaluated on the NeonTreeEvaluation benchmark [17], while Detectree2 [18] applied Mask R-CNN [19] to tropical crown polygons. Tong & Zhang [20] applied StarDist [21], a U-Net [22] backbone model applying star-convex polygons as proposals during the detection stage rather than the more commonly used axis-aligned bounding boxes, allowing for improved Non-Maximum Suppression (NMS) performance and more accurate detections. This approach was designed as a solution to the challenging problem of individual tree crown instance segmentation in dense forest canopy, where crowns are often touching and overlapping, which can contribute to many valid detections being needlessly dropped due to the coarse tessellation of bounding box rectangles. Applying a diferent approach to a similar dense-canopy problem domain, the authors of the tropical rainforest SelvaMask dataset [23] utilised a DINO [24] (a DETR-family detector, unrelated to the self-supervised DINO backbones used here) based tree crown detection model coupled with a Segment Anything Model (SAM) [25] for prediction mask generation, applied to instance segmentation in dense canopy rainforests. Similar concepts of utilising SAM for RS tasks as a second-stage mask generator by providing it with prompts underpins models such as RSPrompter [26] and BalSAM [27], the latter both with and without the integration of a Digital Surface Model (DSM) depth map alongside RGB data.

We aim to address the tree crown instance segmentation space from two angles underexplored in the literature, both largely stemming from the lack of availability of data. Firstly, existing approaches focus on dense-canopy forests, and overwhelmingly exclude biomes such as savannah and dryland. For example, the authors of Detectree2 deliberately exclude images with ≤ 40% tree canopy cover from their datasets [18]. In the DeepForest paper, Weinstein et al. highlight Onaqui, Utah (ONAQ) as the model’s worst-performing site since it is a “desert scrub site with a diferent vegetation structure from any of the training data” [16]. Secondly, in order to produce output individual tree crown instance polygon masks, existing models typically rely on mask annotations for training [18, 28–32], which need to be produced manually by human annotators.

## 1.1 Lack of sparse-canopy environments

We recognise a number of contributing factors to the exclusion of sparse-canopy biomes:

1. the limitations of spatial resolution, since an accurate model of ITC cover is hard to create with most publicly available satellite RGB datasets at 5–10 m/px. In practical usage, high (< 1 m/px) or ultra-high (≤ 0.1 m/px) resolution is required for ITC [33]. For reference, in the OAM-TCD holdout set used in this work (439 images, 25,705 individual tree crowns), downsampling from the images’ native 0.1 m/px to a Ground Sample Distance (GSD) of 1 m (i.e. the real-world area corresponding to one pixel) results in 25.0% of tree crowns enclosing no complete pixel. At 2 m GSD, 57.0% of crowns enclose no complete pixel, and 22.6% have a bounding box narrower than a single pixel on at least one axis;

2. underrepresentation in high-quality annotated datasets. Conventional global tree cover maps such as Hansen et al. [34] have been shown to systematically underestimate tree cover in drylands [4] and are markedly less accurate outside closed-canopy forests [35], and biomass and carbon products used for drylands [36–38] typically apply a single method to forest and dryland vegetation alike [2]. We speculate that this may also have led to a subtle vicious cycle: under-counting causes woody biomass in sparse-canopy biomes to appear too scarce to justify annotation efort, which in turn stymies further exploration of tree crown detection and instance segmentation in exactly the biomes where cover is most uncertain [6]. By way of illustration, in their study of 9.9 billion individual tree crowns in the Sahara/Sahel/Sudan region, Tucker et al. [2] found that “areas with scattered trees are often represented by zero values”, attributing this to “the fact that previous models are rarely developed, trained and validated with plots of very sparse tree cover”.

In short, the development of RS Computer Vision techniques for Tree Canopy monitoring in sparse-canopy areas has been underexplored, yet semi-arid ecosystems account for roughly half of the trend and 39–47% of the interannual variability in the global land carbon sink [39], and drove the record 2011 sink [40]. Drylands as a whole (hyperarid, arid, semi-arid and dry sub humid zones, using the United Nations Food and Agriculture Organization category definitions) cover ∼41.5% of the Earth’s land surface [6] and are expected to expand from this figure by 10–23% by the year 2100 [41]. Recent datasets such as SelvaBox [3] & SelvaMask [23] have broadened geographic coverage of publicly available annotations, and the OAM-TCD dataset [33] in particular incorporates a variety of biomes across the globe, including urban, tundra, and savannah.

## 1.2 Reliance on manual mask annotations

Besides domain coverage and distribution, tree crown instance segmentation also remains understudied due to the lack of individual tree crown mask annotations in datasets [27]. Since crown polygons are slower and typically more expensive to annotate than boxes, box-supervised instance segmentation [42–44] ofers a practical route to instance-level delineation without poly gon labels. The recent rise of self-supervised backbones such as DINOv2 [45] and DINOv3 [46] has demonstrated that models pretrained on very large sets of diverse web imagery (1.7 billion images in the case of DINOv3-web) are capable of matching or even exceeding Earth-observationspecific foundation models (e.g. DINOv3-sat) on high-resolution RGB RS tasks, with the latter retaining an advantage chiefly in physically grounded regression such as canopy height estimation. Apples-to-apples comparison across this literature remains dificult however, as papers variously report $\mathrm { A P _ { 5 0 } }$ , AP<sub>50:95</sub>, instance- or semantic-F1 at difering IoU thresholds against boxes and/or polygons.

In this work we propose a 3-stage model “LACE” built on a frozen DINOv3-web (ViT-L/16) [46] encoder, with a 3.2M-parameter CenterNet-style [47] detector head, and a novel lightweight expectation-maximisation module that leverages latent commonality of the ground truth bounding box annotations. In other words, we ask the question: “can we leverage the vector space signature of the tree-ness within DINO latent feature vectors, based only on the commonality between rectangular bounding box examples, to isolate the trees/foreground from the background?” Our experiments across a range of biomes including temperate, tropical, xeric, and urban scenes, demonstrate that even with diverse tree species and background context, a latent commonality module is suficient to infer and delineate objects of interest from rough rectangular annotations alone.

LACE achieves a mask AP of $0 . 6 6 3 \pm 0 . 0 0 1$ on the OAM-TCD holdout benchmark and a box F1 of $0 . 7 2 8 \pm 0 . 0 0 3$ on NeonTreeEvaluation (evaluated at IoU = 0.4). The creators of NeonTreeEvaluation propose 0.4 IoU as a suitable threshold for tree crown detection tasks based on comparative analysis of agreement and overlap between diferent human annotations of the same datasets, and we follow their convention on this dataset for direct comparison with their results.

By using only the hand-annotated bounding box annotations (which are sampled from 13 of the total 22 sites present in the test dataset), our model was able to match DeepForest’s published results of 0.719 F1, evaluated at IoU ≥ 0.4. We also did not make use of a “canopy height model” (CHM) (a LiDAR derived height raster at 1 m spatial resolution), or any of the NeonTreeEvaluation dataset’s hyperspectral imagery, since our focus is the underexplored and data-scarce sparse-canopy context. By demonstrating the eficacy of LACE on RGB data alone, we hope our contribution to the remote sensing community will have a broader application to data-scarce contexts.

In summary, the lack of publicly available tree crown-annotated datasets generates a need for box-supervised, performant TCIS models suitable for diverse sparse-canopy contexts. Our contribution is deliberately orthogonal to dense-canopy tropical forest models, which we use as baselines for comparison in the Results section. In broad terms, most existing state-of-the-art models focus on improving the precision of delineation in dense environments. By contrast, our focus is on identifying tree crowns in sparse-canopy landscapes. Nevertheless, LACE is still able to demonstrate SOTA performance in apples-to-apples comparisons, even with less training data and box supervision. By leveraging the rich DINOv3 features which enable rich cross-biome transferability, LACE is, to the best of our knowledge, the strongest performing biome-agnostic model in the TCIS space as evidenced by results on the OAM-TCD benchmark.

In this work, we present:

• a) our model LACE, built on a DINOv3-web (ViT-L/16) backbone [46], with a trained CenterNet-style detector head [47] that identifies crown centres for detection based on the DINO output embeddings. We assess the performance of this approach against existing models on two publicly available datasets: Restor’s OAM-TCD and DeepForest’s NEON annotation dataset.

• b) a novel box-to-mask module designed to extract the commonality in latent space, using an Expectation-Maximisation (EM) algorithm to isolate “treeness” within the bounding box. Unlike other existing implementations such as EM-Adapt [48], which amortises the E-step into a gradient-trained CNN with implicit cross-box sharing, we fit an explicit closed-form commonality mixture over frozen features. We exploit the similarities between signatures of composite embeddings within ground truth bounding boxes, thematically following previous works such as DiscoBox [49] and DDT [50]. In our model however, we align more closely with architectures such as STEGO [51] by utilising a World Model/JEPA [52] style approach, eschewing pixel-space calculations in favour of strictly latent space analysis. This approach is made possible by the capabilities of Meta’s DINOv3 encoder foundation model [46], a Vision-transformer-based model trained on 1.7 Billion images using Self-supervised Learning.

## 2 Related Work

The use of computer vision models to process aerial and satellite RGB imagery for the purposes of tree crown measurements has seen a number of advancements in recent years [32]. Two important benefits of RGB methods above other data modalities such as LiDAR or hyperspectral data-based methods are firstly the widespread availability of aerial RGB imagery, and secondly the high spatial resolution of that data up to and including submeter per pixel precision. Moreover, aerial RGB data can also be collected by UAVs (drones) with relative ease. Consequently, in recent years there has been a focus in the RS community on RGB datasets which can directly leverage Computer Vision deep learning models [53] especially in the use of Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) [54].

Note on terminology We follow Ball et al. [18] in defining Tree Crown Instance Segmentation (TCIS) as the precise delineation of individual tree crowns. Since individual trees are the fundamental unit that aggregates to biomass over a landscape [55] and therefore carbon sequestration, in this work we focus on TCIS. Other related categories include Tree Crown Detection (TCD) models, which aim to detect instances of a given object and delineate them with a rectangular bounding box e.g. YOLO family [56], and Semantic Segmentation (TCSS) models [57, 58] which aim to categorise each pixel into a class. We use the TCIS acronym rather than other (perfectly reasonable) proposed conventions such as Tree Crown Detection and Delineation (TCDD) [29] or Individual Tree Crown Detection and Delineation (ITDCD) [32], primarily to distinguish clearly and unambiguously between the above downstream tasks.

## 2.1 Tree Crown Models

Convolutional Neural Networks Modern CNN-based object detection models are comprised of three parts:

1. a feature extractor network (‘backbone’) which is a neural network that scans with convolutions to pick up on features in the image, with simpler features typically extracted earlier in the network’s layers and more complex composite features requiring greater depth [59];

2. a ‘neck’, which can merge the detections across diferent layers of the backbone and provide access to the whole gamut of feature scale and complexity;

3. the Detector ‘head(s)’, which take the encoded features as inputs and produce the entire CNN’s output, which can be, for example, bounding boxes around detected objects, polygons for a precise segmentation mask, or classification categories. Detector heads are generally either one-stage or two-stage detectors. Two-stage detectors first utilise a module to create many possible bounding boxes which are region proposals, before the second stage module extracts features from the proposals [60]. In this way, the model can accurately detect within a large image by focusing on high-probability smaller regions.

One-stage detectors include YOLO (‘You Only Look Once’) series of models [56, 61], RetinaNet [62] and EficientDet [63]. Within TCD, the dominant two-stage detectors descend from the Region-based CNN (R-CNN) [64]: Fast R-CNN [65] shares convolutional features across proposals, Faster R-CNN [66] replaces external proposals with a learned Region Proposal Network (RPN), and Mask R-CNN [19] extends Faster R-CNN with a parallel per-instance mask head, which is what makes it the standard architecture for TCIS.

## 2.1.1 Star-convex

Star-convex representations ofer a diferent route to separating touching crowns. Tong & Zhang [20] adapt StarDist [21] to tree crowns, replacing axis-aligned boxes with star-convex polygons so that NMS prunes overlapping proposals on crown geometry rather than box overlap, and report gains over Mask R-CNN and other SOTA models on dense, overlapping canopy in RGB imagery. MP-PolarMask [67], although not a TCIS-domain model, extends the star-convex representation with auxiliary polar centres to capture concave shapes, which could perhaps be suitable for non-star-convex crowns such as palm trees. FG-TreeSeg [68] takes a diferent route: it starts from a canopy semantic mask, then applies Cellpose-SAM, a cell-segmentation model that predicts, for every canopy pixel, a flow vector pointing towards its crown centre. By following these vectors, pixels can be grouped by the centre they converge on, so touching crowns are split into instances without any instance annotations.

## 2.1.2 DeepForest

DeepForest [16] is a significant baseline model in Tree Crown Detection (TCD). DeepForest is a RetinaNet on a ResNet50 backbone and Feature Pyramid Network (FPN) trained on over 30 million LiDAR-prompted algorithmically generated crowns from 22 National Ecological Observatory Network (NEON) sites and further fine-tuned using 10,000 hand-annotated crowns.

## 2.1.3 Pretrained Foundation Models

Detectron2 [69] is the object detection and segmentation library on which several of the tree crown baselines in this work are built, including Detectree2 and Restor’s Mask R-CNN; it supplies the Mask R-CNN, RetinaNet and Faster R-CNN reference implementations together with the training, inference and evaluation harness used to fine-tune them.

Meta’s Segment Anything Model (SAM) [25], now in its third iteration [70], is a promptable class-agnostic segmentation foundation model: given a point, box or mask prompt it emits an instance mask, which makes it a natural box-to-mask module for detectors that emit only rectangles, and it is used in that role by several of the baselines compared here.

Meta’s latest version of the Self-Distillation with No Labels (DINO) family [45, 71] is DINOv3 [46]: a SOTA encoder trained with fully Self-Supervised Learning on 1.7 billion images pulled from various online sources. Utilising SSL facilitates training regimes on orders of magnitude more image data since weakly and fully supervised methods typically require high-quality labels and/or metadata for the training images [72–74]. Highly context-agnostic, DINOv3-web is an example of a single frozen SSL backbone that can serve as a “universal visual encoder” [46] capable of generating rich embedding vectors that capture the semantic information from the raw pixel patches in context. Incorporating both a global image-level objective and a local iBOT-style [75] patch-level latent reconstruction objective in the loss function results in a model that excels at encoding global and local features. In this work, we use the DINOv3-web ViT-L distilled model, with a native 16 px patch.

SelvaBox [3] is a UAV-captured RGB dataset of tropical dense-canopy forest in Panama, Brazil, and Ecuador, composed of 83,000 ITC manual bounding box annotations. Its authors publish a DINO (DETR) [24] Swin-L based model trained on NeonTreeEvaluation (see Datasets), OAM-TCD, and another dataset called QuebecTrees, which achieves a best result of box mAP<sub>50:95</sub> of 44. $. 2 9 \pm 0 . 3 3$ on OAM-TCD holdout set, improving upon DeepForest $( 3 9 . 0 0 \pm 0 . 2 1 ) $ and a Faster R-CNN ResNet50 (38.34 ± 0.26) finetuned on the same data. A second related dataset, SelvaMask [23], provides ITC instance segmentation masks and assesses the SelvaBox detection model (also referred to by the authors as ‘SelvaBox’) in combination with both frozen and fine-tuned SAM 3 decoder modules on the OAM-TCD dataset for the TCIS task. The authors compare results on a number of benchmark datasets including OAM-TCD alongside Detectree2 and DeepForest with a similar frozen SAM 3 mask prediction module. SelvaMask’s focus is increased precision of tree crown boundaries in dense canopy, whereas our focus is on sparsecanopy detection accuracy. They note that their model is specialised “to dense, interlocking canopies, slightly reducing its transferability to temperate forests”.

Leveraging SAM as an out-of-the-box instance segmenter for remote sensing has been explored in RSPrompter [26], which overcomes SAM’s reliance on prompts such as points, boxes or masks by training a prompter module that learns the intermediate layer features of the SAM encoder on labelled training images, and then using those embeddings as input to SAM’s mask decoder in order to produce category-labelled instance masks.

Table 1: Representative individual tree crown methods, grouped by the annotation type used for training. Output: boxes or instance masks. Extra net(s): pretrained networks required at inference beyond the method’s own backbone. Biome coverage refers to evaluation.
<table><tr><td></td><td></td><td>GSD</td><td></td><td colspan="4">Biome</td></tr><tr><td>Method</td><td></td><td>Output Input</td><td>Extra net(s)</td><td>(cm) Tr Te Sv</td><td></td><td></td><td></td></tr><tr><td>Trained on crown polygons</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mask R-CNN (OAM-TCD) [33] Mask</td><td></td><td>RGB</td><td></td><td>10</td><td>0</td><td>0</td><td>o</td></tr><tr><td>Detectree2 [18]</td><td>Mask</td><td>RGB</td><td></td><td>10</td><td>•</td><td>X</td><td>X</td></tr><tr><td>Mask R-CNN / DETR [76]</td><td>Mask</td><td>MS+LiD</td><td></td><td>5</td><td>X</td><td>●</td><td>×</td></tr><tr><td>CrownViM [77]</td><td>Mask</td><td>RGB</td><td></td><td>10</td><td>0</td><td>0</td><td>0</td></tr><tr><td>BalSAM [27]</td><td>Mask</td><td>RGB+DSM SAM</td><td></td><td>&lt;5</td><td></td><td></td><td>X</td></tr><tr><td>SelvaMask [23]</td><td>Mask</td><td>RGB</td><td>SAM3</td><td>1-4</td><td></td><td></td><td>0</td></tr><tr><td>Trained on bounding boxes</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepForest [16]</td><td>Box</td><td>RGB</td><td></td><td>10</td><td>o</td><td></td><td></td></tr><tr><td>SelvaBox [3]</td><td>Box</td><td>RGB</td><td></td><td>1-5</td><td></td><td></td><td>O</td></tr><tr><td>LACE (ours)</td><td>Mask</td><td>RGB</td><td></td><td>10</td><td>O</td><td></td><td></td></tr><tr><td>No instance annotations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SAM 2 + LiDAR prompts [78]</td><td>Mask</td><td>RGB+LiD</td><td>SAM2</td><td>10</td><td>X</td><td></td><td>X</td></tr><tr><td>FG-TreeSeg [68]</td><td>Mask</td><td>RGB</td><td>SegFormer, Cellpose-SAM</td><td>≤10</td><td>X</td><td>●</td><td>X</td></tr></table>

Tr, tropical; Te, temperate and boreal; Sv, savannah/dryland. • evaluated and reported separately; ◦ present in an aggregate benchmark (e.g. OAM-TCD), not reported as a stratum; × not evaluated. LiD, LiDAR; MS, multispectral; DSM, digital surface model. <sup>‡</sup>Trained on semantic canopy labels only; instances are recovered without instance annotations.

## 3 Materials and Methods

## 3.1 Datasets

Restor OAM-TCD The OAM-TCD dataset [33], introduced by Veitch-Michaelis et al. (ETH Zurich, Restor and collaborators), uses OpenAerialMap (OAM) RGB imagery and provides 5072 2048 × 2048 px images at 0.1 m/px resolution with associated human-labelled instance masks for over 280k individual and 56k groups of trees (groups are hereafter referred to as “canopy”). In our work, the OAM-TCD dataset has been an invaluable benchmark to assess the comparative performance of our LACE model in an apples-to-apples comparison against other SOTA models, due to the inclusion of not only bounding boxes but also precise polygon masks, as well as its broad range of ecological biomes to capture the diversity and morphology of trees in diferent terrestrial biomes including both urban and natural environments. This multi-region, multi-biome distribution of annotated aerial RGB data allows for a much more diverse training set which aids cross-biome transfer, and also contains savannah & arid dryland landscapes which

are scarce in existing annotated datasets.

Of the 5072 images, 439 are reserved as a holdout benchmark test set, allowing for a global multi-biome comparison between models. The 439-image test set contains 25,705 ITC annotations, and 366 images within the set contain ≥1 crown, i.e., 73 of the images have 0 ITCs. The same test set also contains 6,782 canopy annotations, which are groups of trees in a wide range of real-world sizes. 364 of 439 tiles contain ≥1 canopy region. Annotators were told to use the canopy class when it was not possible to reliably delineate individual trees, including when it was “not obvious whether a tree was an individual or multiple”. 342 images contain at least one of each class and 51 tiles are fully empty. The densest tile holds 422 individual crowns; the median tile holds 29.

NeonTreeEvaluation The U.S. National Science Foundation’s National Ecological Observatory Network (NEON) publicly available aerial imagery dataset is a cornerstone of RS research, comprising 22 separate locations across the continental United States and including a variety of diferent biomes. The DeepForest model released alongside the NeonTreeEvaluation benchmark [17] was pretrained on all 22 sites, using an unsupervised LiDAR-based algorithm [79] to generate millions of moderate-quality annotations, and was then fine-tuned on a further 10,000 manual annotations of RGB imagery from six sites. The quality and scale of the NEON dataset as well as its fundamental role in the development of the DeepForest model was a key reason to build our pipeline to natively support the NEON resolution of 0.1 m/px.

Data preprocessing: We pinned NeonTreeEvaluation @1.8.0 and built the evaluation GT by intersecting annotation XMLs with RGB tiles. This resulted in 194 scored tiles / 6,634 boxes / 22 sites. We downloaded the hand-annotated RGB training tiles, which are spatially disjoint from the evaluation tiles, and cropped 18 kept tiles (13 sites, 23,424 source boxes) into 2,063 non-overlapping 400 px patches (26,800 boxes after clipping to patch bounds), restricted to genuinely-annotated regions only and capping empty negatives at 0.3× positives per tile. Two tiles were dropped due to low annotation coverage (SJER 41.8% / TOOL 0% coverage). The train/val split is patch-level stratified (every 8th patch of each tile, 1,807 train / 256 val), so all sites appear in training; val is used only as the early-stopping signal and the reported metric is the fully disjoint 194-tile benchmark.

Both are RGB-only (we did not use the dataset’s LiDAR/CHM/HSI data at any time), at native 0.1 m/px, and the 400 px patch geometry matches the 400 px evaluation tiles so training and test see the same scale. Features are then extracted by padding each patch to 512 px and encoding it to a 32 × 32 DINOv3 grid per pass (64 × 64 after interlacing), taking layers 21–24 (4096 dimensions, against the single final layer used on OAM-TCD; Section 3.2.1).

## 3.2 LACE

The method converts a frozen self-supervised encoder into instance masks for individual tree crowns using bounding-box supervision only. It has three stages: a dense feature grid obtained by interlacing four ofset passes of a frozen DINOv3-web ViT (Section 3.2.1); an anchor-free detector trained on bounding boxes (Section 3.2.2); and a training-free box-to-mask module that converts each predicted box into a mask by foreground/background discrimination in a whitened feature space (Section 3.2.3). Predictions are then ranked by combining the detector score with the mask module’s own posterior (Section 3.2.5), which adds no parameter and leaves the masks untouched. No crown polygon is read at any stage of encoding, detection or mask fitting. The only mask labels used anywhere in the pipeline are the validation images on which the three inference scalars are selected (Section 3.2.4).

## 3.2.1 Dense features by phase interlacing

A ViT with patch size p emits one embedding per $p \times p$ patch, giving a feature grid p times coarser than the image. For crowns whose diameter is a small multiple of $p$ this is the binding resolution constraint. Bilinear upsampling of that grid increases the sampling rate but introduces no new measurements.

Rather than upsample, we instead evaluate the frozen encoder four times per tile, shifting the input by $( \delta _ { y } , \delta _ { x } ) \in \{ 0 , p / 2 \} ^ { 2 }$ , and interlace the four grids, which we found produced higher performance (Section 4.1). With $p = 1 6$ this yields a real $8 ~ \mathrm { p x }$ lattice. Writing $F ^ { ( \delta _ { y } , \delta _ { x } ) }$ for the grid from one ofset, the assembled feature tensor $\widetilde { F }$ is

$$
\begin{array} { r } { \tilde { F } \Big [ : , 2 i + \frac { \delta _ { y } } { 8 } , 2 j + \frac { \delta _ { x } } { 8 } \Big ] = F ^ { ( \delta _ { y } , \delta _ { x } ) } [ : , i , j ] , \qquad i , j \in \{ 0 , \dots , 1 2 7 \} , } \end{array}\tag{1}
$$

so that cell $( Y , X )$ of $\widetilde { F }$ carries the embedding of the patch centred at pixel $( 8 X + 8 , 8 Y + 8 )$ Each cell is therefore a genuine encoder output at its own location with a virtual 4 px border included, rather than an interpolation of its neighbours, and the union of the four phases’ patch centres is exactly the 8 px lattice.

Encoding uses DINOv3 ViT-L/16 with weights frozen throughout. On OAM-TCD we take final-layer (layer 24) activations (1024 dimensions; on NEON, layers 21–24, 4096 dimensions), giving $\widetilde { F } \in \dot { \mathbb { R } } ^ { 1 0 2 4 \times 2 5 6 \times 2 5 6 }$ for a 2048 px tile. Registration is verified independently: a synthetic blob is localised to within one cell across 200 positions, and interlaced cells agree with bilinearly interpolated ones only at $\cos \approx 0 . 9 5 5$ , confirming that the four passes contribute distinct evidence.

## 3.2.2 Detection

Detection uses an anchor-free centre-based head operating directly on $\widetilde { F }$ at stride 8. $\mathrm { ~ A ~ } 1 \times 1$ stem projects 1024 channels to width 256, followed by five $3 \times 3$ convolutional blocks with group normalisation and ReLU (3.2M parameters in total; 4.0M with the 4096-dimensional NEON input), and two $1 \times 1$ output heads: a single-channel centre heatmap and a four-channel regression map carrying a local sub-cell ofset and log box size. This is based on the CenterNet design [47]. The heatmap bias is initialised to −2.19 so the initial centre probability is $\approx 0 . 1$ , and the size biases are initialised to a 20 px crown at stride 8.

Training minimises a focal loss on the centre heatmap plus smooth- $L _ { 1 }$ losses on ofset and size at annotated centres, the size term weighted by 0.1. Because the interlaced grid is already at the target stride, the head contains no internal upsampling; the half-cell diference between the assembled cell centre 8X + 8 and the target cell centre $8 X + 4$ is a uniform translation absorbed by the ofset head, whose bias is initialised accordingly. Centre targets are encoded as CenterNet Gaussians whose radius follows the standard IoU criterion of CornerNet [80] at an overlap of 0.7, clamped never to fall below one feature cell so that the smallest crowns retain a non-degenerate target. Boxes are decoded by local-maximum selection on the heatmap, retaining candidates scoring above 0.05 up to a maximum of 600 per tile. Average precision is computed over the full ranked list; where precision, recall and F1 are reported at a single operating point, that point is a detector score of 0.4.

## 3.2.3 Training-free box-to-mask conversion

Given a box, the task remains to decide which of its cells belong to the crown. We treat this as foreground/background discrimination in a whitened feature space, with the model fitted once on training tiles using boxes only and then held fixed.

The module has four key components: a whitened feature space; a frozen background mixture; a set of foreground prototypes; and a spatial prior over relative position within the box. The last two are learned by an alternating EM-based fit, while the first two stay fixed.

Feature space LACE’s core idea is to leverage the rich DINOv3-web embeddings as suficiently descriptive representations of each patch to build a high-performing tree-crown instance segmentation model. We begin with a, a given cell’s raw unmodified embedding, and PCA-whiten it: a is centred, projected onto the leading $d = 1 2 8$ principal directions of the training-cell distribution, and rescaled so that each retained component has unit variance. A final $L _ { 2 }$ normalisation then places the result on the unit sphere, giving the unit vector z.

$$
\tilde { z } ~ = ~ { \frac { ( a - \mu ) W } { \lambda } } , \qquad z ~ = ~ { \frac { \tilde { z } } { \| \tilde { z } \| } } ,\tag{2}
$$

Here $a \in \mathbb { R } ^ { 1 0 2 4 }$ is one cell of ${ \tilde { F } } ;$ the cell mean $\mu \in \mathbb { R } ^ { 1 0 2 4 }$ , the whitening matrix $W \in \mathbb { R } ^ { 1 0 2 4 \times d }$ whose columns are the retained principal directions, and the scale vector $\lambda \in \mathbb { R } ^ { d }$ of the corresponding singular values are all estimated once on training cells and then frozen. This whitening/spherising equalises the variance of the retained directions so that no direction dominates by scale alone. The final normalisation then places every cell on the unit sphere, making all subsequent comparisons simple cosine similarities and allowing both foreground and background to be modelled by von Mises–Fisher components [81] with a shared concentration κ.

Background mixture Cells are partitioned into three categories: in-box cells, clear background cells, and a ring of cells immediately outside each box. We exclude cells that fall within GT boxes so as not to accidentally define valid tree cells as background. The ring cells share illumination, phenology and local context (e.g. ground shadow) with the crown and are therefore suitable hard negatives. A background mixture of $K _ { b g }$ components $\{ ( w _ { j } , G _ { j } ) \} _ { j = 1 } ^ { K _ { b g } }$ is fitted to the union of background and ring cells by spherical k-means and frozen, giving the background log-likelihood $\ell _ { b g }$ as follows:

$$
\begin{array} { r } { \ell _ { b g } ( z ) = \log \sum _ { j } w _ { j } \exp \Bigl ( \kappa z ^ { \top } G _ { j } \Bigr ) } \end{array}\tag{3}
$$

Each background centroid $G _ { j }$ is a unit vector on the same sphere as the cells, the mixture weights $w _ { j }$ sum to one, and $z ^ { \top } G _ { j }$ is therefore just a cosine similarity. The log-sum-exp acts as a smooth maximum over the components, so $\ell _ { b g } ( z )$ is a single scalar per cell reading as how well that cell matches its best-matching background component. It is neither a threshold nor an objective, but the evidence term that Equation (4) scores the foreground against.<sup>1</sup>

Foreground mixture Foreground and background take the same von Mises–Fisher form with a shared concentration κ, and difer in one respect only: the background weights $w _ { j }$ are global and frozen, whereas the foreground weights are supplied per cell by the spatial prior and re-estimated at every M-step. Foreground is represented by $K _ { f g }$ unit prototypes $\{ C _ { k } \} _ { k = 1 } ^ { K _ { f g } }$ , the exact counterpart of the background centroids $G _ { j }$ , initialised by spherical k-means over in-box cells restricted to those atypical of the background, i.e. cells which have a value of $\ell _ { b g }$ below the box’s median $\ell _ { b g }$ . This is merely a heuristic to kick-start the fitting. Without this restriction, the initialisation is seeded from in-box background elements such as soil and roads, which are numerous and not the intended crown foreground.

Spatial prior We observe that a crown typically occupies a roughly central and roughly circular region of a reasonably tight GT box. We encode this as a spatial prior $\rho$ over the crown’s relative position within the box, to push the model towards producing reasonable tree crown-like prediction masks. In particular, we aim to increase LACE’s robustness to dense-canopy and lush undergrowth situations, where the box-to-mask module can unintentionally produce masks that are not tree crown-like. The box is divided into an $N \times N$ grid of bins in normalised coordinates $( u , v ) \in [ 0 , 1 ] ^ { 2 }$ . Because those coordinates are a fraction of the box, $\rho$ is by construction invariant to crown size, tile size and ground sample distance. Priors are maintained separately for three crown-size bands, whose edges are the terciles of the training box-size distribution, since at a fixed feature stride the number of cells spanning a crown depends on its size. Taken together, $\rho$ therefore carries one value for each (size band, prototype, u bin, v bin) combination: it is a joint prior over where in the box a cell sits and which prototype is expected there. We write $\hat { \rho } _ { k } = \rho _ { k } / \sum _ { k ^ { \prime } } \rho _ { k ^ { \prime } }$ for the prior at a bin normalised across prototypes, so that $\hat { \rho }$ is a distribution over the $K _ { f g }$ prototypes at that bin while the unnormalised total $\sum _ { k } \rho _ { k }$ remains the probability that the bin is crown at all. Equations (4) and (6) draw on these two quantities separately. $\rho$ is initialised uniform, so all prototypes begin with identical spatial profiles and diferentiate only through Equation (10).

Two constraints are applied to $\rho$ at every M-step. The first pins its mean per-bin foreground mass, summed over prototypes and averaged over the $N ^ { 2 }$ bins, to $\pi / 4 \approx 0 . 7 8 5$ , the area fraction of an ellipse inscribed in its bounding box and hence independent of box aspect ratio. The second caps any single bin at 0.95, so that no relative position is ever treated as certainly foreground. Pinning the mass to a constant derived from geometry rather than fitted is what keeps the prior label-free, and it means the fit decides only where the foreground mass sits, never how much of it there is: without the constraint $\rho$ inflates towards unity, stops discriminating position, and the α term of Equation (6) degenerates into an additive constant.<sup>2</sup>

Fitting Once we have these objects (the whitening $( \mu , W , \lambda )$ , the frozen background mixture $\{ ( w _ { j } , G _ { j } ) \}$ , the foreground prototypes $\{ C _ { k } \}$ , and the spatial prior $\rho )$ , we can run the EM fitting algorithm, which alternates a per-box E-step with a global M-step, set out in full in Algorithm 1. Only $\{ C _ { k } \}$ and $\rho$ are learned here, the rest are estimated once and held fixed.

E-step. For the cells of one box, with $\rho _ { k }$ the spatial prior evaluated at each cell’s bin for prototype k and $\sum _ { k } \rho _ { k }$ the total foreground prior at that bin, the posterior follows in three steps: an appearance log-ratio (4), its within-box recentring (5), and the posterior itself (6).

$$
\begin{array} { r } { A ( z ) = \log \sum _ { k } \hat { \rho } _ { k } \exp \Bigl ( \kappa z ^ { \top } C _ { k } \Bigr ) - \ell _ { b g } ( z ) , } \end{array}\tag{4}
$$

$$
\bar { A } ( z ) = A ( z ) - \mathrm { m e a n } _ { z ^ { \prime } \in \mathrm { b o x } } A ( z ^ { \prime } ) ,\tag{5}
$$

$$
p _ { f g } ( z ) = \sigma \Bigl ( \bar { A } ( z ) + \alpha \mathrm { l o g i t } ( \Sigma _ { k } \rho _ { k } ) \Bigr ) .\tag{6}
$$

Equation (4) is a log-likelihood ratio. Its first term is the foreground log-likelihood, built from the prototypes $C _ { k }$ exactly as Equation (3) is built from the background centroids $G _ { j } { \mathrm { : } }$ ; subtracting $\ell _ { b g }$ gives $A ( z ) = \log ( p ( z \mid$ crown) $) / p ( z \mid$ background). A cell must therefore look crown-like and unlike background to score highly. This is the first of two filters that in-box soil/ground has to pass, and it removes the soil/ground that resembles the background mixture outright.

Equation (5) is the second filter, and the step that makes the model work on frozen ViT features. Problematically, attention over the surrounding tile inflates the absolute foreground score of every cell inside a box, so A is confounded by context and its absolute level is not suficiently informative: e.g. soil surrounded by canopy can still score positively, and therefore survive the first filter. However, recentring isolates A within a box, asking not whether a cell is crown-like but whether it is more crown-like than the rest of its own box. The behaviour that follows is what we wanted to produce: a heterogeneous box containing crown and soil has a wide spread of $\bar { A }$ and is carved appropriately, whereas a box in uniformly closed canopy has a narrow spread and stays closer to filled, which is generally correct.

Equation (6), the posterior, adds the only box-dependent term, and it is worth noting that the spatial prior enters the two equations in diferent roles. In Equation (4) it appears only as the normalised weight $\hat { \rho } _ { k }$ , which sums to one over k and therefore selects which prototype applies at a given relative position while discarding how much total foreground prior there is. That discarded total re-enters on its own in Equation (6), where the prior weight α scales it. The appearance term $\bar { A }$ is thus entirely box-independent, and Equation (6) contains the only point at which box geometry influences the posterior. Equation (6) gives the probability that a cell is crown, but the M-step also needs to know which prototype accounts for it. The E-step therefore returns a second quantity, the responsibilities $r _ { n k }$

$$
r _ { n k } = p _ { f g } ( z _ { n } ) \frac { \rho _ { k } \exp \Bigl ( \kappa z _ { n } ^ { \top } C _ { k } \Bigr ) } { \sum _ { k ^ { \prime } } \rho _ { k ^ { \prime } } \exp \bigl ( \kappa z _ { n } ^ { \top } C _ { k ^ { \prime } } \bigr ) } ,\tag{7}
$$

which distribute each cell’s foreground probability over the $K _ { f g }$ prototypes in proportion to the weighted components of Equation (4), so that $\begin{array} { r } { \sum _ { k } r _ { n k } = p _ { f g } ( z _ { n } ) } \end{array}$ . (Note that Equation (7) is written with $\rho _ { k }$ rather than $\hat { \rho } _ { k }$ because the normalisation appears in both numerator and denominator and therefore cancels.) Equations (4)–(7) are evaluated for all the cells of one box at a time and the results accumulated across all training boxes.

M-step. The prototypes are updated in two steps. The first is the ordinary generative pull: prototype k is drawn towards the responsibility-weighted mean of all cells over all boxes,

$$
P _ { k } = \frac { \sum _ { n } r _ { n k } z _ { n } } { \| \sum _ { n } r _ { n k } z _ { n } \| } ,\tag{8}
$$

which expresses the idea “recurs across boxes”. The second move repels the prototype from the outside-box signature it most resembles. The outside-box cells (clear background and ring) are softmax-assigned to the current prototypes at the same concentration $\kappa ;$ the resulting perprototype negative centroid $N _ { k }$ is the normalised weighted mean of the cells assigned to prototype $k ,$ and the prototype becomes

$$
C _ { k } = \frac { P _ { k } - \beta \bar { N } _ { k } } { \| P _ { k } - \beta \bar { N } _ { k } \| } .\tag{9}
$$

A soil-like prototype’s nearest negatives are themselves soil, so the subtraction tears it away from soil: commonality is defined as agreement-across-boxes minus outside-box signature, not merely whatever fills the box. The contrastive weight $\beta$ controls the strength of that repulsion; $\beta = 0$ recovers the purely generative update of Equation (8), whose cost is reported in Section 4.1, and we use $\beta = 0 . 5$ throughout.

The spatial prior is re-estimated from the same responsibilities. Writing $b ( n )$ for the bin of cell n and $N _ { b }$ for the number of cells falling in bin $b ,$

$$
\tilde { \rho } _ { k } ( b ) = \frac { 1 } { N _ { b } } \sum _ { n : b ( n ) = b } r _ { n k } ,\tag{10}
$$

accumulated separately per size band, after which $\tilde { \rho }$ is rescaled to the $\pi / 4$ mean per-bin mass and capped at 0.95 per bin as described above to give $\rho .$ . Finally, after a burn-in of five iterations, any prototype whose share of the total foreground responsibility, $\begin{array} { r l } { \sum _ { n , b } r _ { n k } \big / \sum _ { n , b , k ^ { \prime } } r _ { n k ^ { \prime } } } & { { } } \end{array}$ , falls below $0 . 2 5 / K _ { f g }$ (a quarter of the uniform share $1 / K _ { f g } )$ is deleted and $K _ { f g }$ decreases accordingly; pruning never reduces $K _ { f g }$ below two.

Fitting is deterministic given the seed. The procedure is not gradient-based, reads no crown polygons, and produces a compact set of parameters $\{ \mu , W , \lambda , C , G , w , \rho \}$

Algorithm 1 Box-only masker fit. Inputs: training tiles, their boxes, and the frozen encoder.   
No crown polygon is read.   
1: Whitening. Estimate $\mu ,$ W, λ on all training cells; map every cell by Equation (2).   
2: Background. Partition cells into in-box, ring and clear background by box geometry; fit   
$\{ ( w _ { j } , G _ { j } ) \}$ on ring ∪ clear by spherical k-means; freeze. Precompute $\ell _ { b g }$ once (Equation (3));   
it never changes.   
3: Initialise C. Spherical k-means over in-box cells with $\ell _ { b g }$ below its median, giving $K _ { f g }$   
prototypes.   
4: Initialise $\rho .$ Uniform, at $\pi / 4 / K _ { f g }$ in every bin of every size band.   
5: for $t = 1$ to $T = 3 0$ do   
6: for each training box do ▷ E-step, per box   
7: Evaluate A, $\bar { A } , p _ { f g } , r _ { n k }$ by Equations $( 4 ) - ( 7 )$   
8: Accumulate $\textstyle \sum _ { n } r _ { n k } z _ { n }$ and the per-bin sums of $r _ { n k }$   
9: end for   
10: $P _ { k } $ Equation $( 8 ) ; ~ C _ { k } \gets$ Equation (9). ▷ M-step, global   
11: $\rho \gets$ Equation (10), then rescale to $\pi / 4$ and cap at 0.95.   
12: if $t > 5$ then delete prototypes whose responsibility share is below $0 . 2 5 / K _ { f g }$ (a quarter   
of uniform), keeping at least two.   
13: end if   
14: end for   
15: return $\{ \mu , W , \lambda , C , G , w , \rho \}$

This loop has two important properties. First, the background mixture is frozen after step 2, so $\ell _ { b g }$ is a constant of the fit rather than a quantity being jointly estimated. Second, the procedure is EM-style rather than EM proper: the within-box recentring of Equation (5), the contrastive term of Equation (9) and the projection of $\rho$ onto the mass and cap constraints each break the correspondence with a joint likelihood, so no objective is monotonically ascended and there is no convergence test: T is fixed at 30 iterations.

## 3.2.4 Inference

For each predicted box, cells within the box are projected by Equation (2), the matching size band is selected, and posterior $p _ { f g }$ is evaluated by Equations (4)–(6). Cells above a mask threshold $\tau = 0 . 2 5$ form the predicted tree crown mask, which is then rasterised at the evaluation resolution. Only cells lying within the box are evaluated, and the rasterised mask is intersected with the box, so a mask is contained in its box by construction rather than by penalty.

Three scalars are exposed at inference, and they are the only quantities in the pipeline selected against mask labels. The prior weight α scales the box-derived spatial term in Equation (6): values $\leq 1$ relax reliance on a predicted box, which is not expected to always be perfectly precise, so that the appearance term dominates. (Because the spatial prior is defined in normalised box coordinates, α is invariant to crown size and GSD. What it tracks is the precision of the boxes it is applied to). The concentration κ is raised from the value of 10 used during fitting to 16 at inference, in Equations (3) and (4) alike; since $\bar { A }$ is zero-mean within each box, this scales the spread of the appearance term and thereby sharpens it. 10 was fixed a priori, and 16 is simply the best of {10, 13, 16, 20} on validation, a range over which mask $\mathrm { A P _ { 5 0 } }$ varies by under 0.003. The robust finding is directional: any $\kappa > 1 0$ outperforms $\kappa = 1 0$ at every α.

The mask threshold $\tau = 0 . 2 5$ is the posterior cut that defines the mask, lowered from 0.5 so that small crowns under-covered by imprecise predicted boxes are recovered. All three are selected on the 108 validation images held out from the 900 training images (the detector trains on the remaining 792) under a rule fixed in advance, and none requires refitting. We therefore describe the method as trained without mask supervision but broadly calibrated for tree crown mask generation on a small held-out set of mask labels, rather than as entirely label-free. We take these to be reasonable ‘tree crown’ hyperparameters and do not attempt to optimise them for use on other datasets such as NeonTreeEvaluation.

## 3.2.5 Confidence from the mask posterior

Now that we have the mask generated, we want to update the confidence score of each tree crown rather than just retain the score provided by the detector head, which is $s = \sigma ( h )$ , the sigmoid of the centre heatmap logit h at the decoded peak of Section 3.2.2. This value correlates to whether a crown centre is present at a pixel within the box, but does not necessarily correctly score the resulting mask (from Section 3.2.3). However, we have already calculated the posterior $p _ { f g }$ of Equation (6) for every cell of a box. To generate a confidence score we can reduce the posterior over the $| B |$ cells of a box B to two terms which we can use as multipliers on the $\sigma ( h )$ score to produce a new score $s ^ { \prime } ( B )$ which reflects the updated confidence based on LACE’s experience of the training boxes it has seen.

$$
\bar { p } ( B ) \ : = \ : \frac { 1 } { | B | } \sum _ { n \in B } p _ { f g } ( z _ { n } ) ,\tag{11}
$$

$$
m ( B ) = \frac { 1 } { | B | } \sum _ { n \in B } | 2 p _ { f g } ( z _ { n } ) - 1 | .\tag{12}
$$

Because the recentring of Equation (5) sets the mean of $\bar { A }$ to zero within every box, both statistics read the shape of the within-box posterior rather than its level. For $\bar { p }$ this does not force the value $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ , since the mean of a sigmoid is not the sigmoid of the mean: writing $\begin{array} { r } { \sigma ( x ) = \frac { 1 } { 2 } + g ( x ) } \end{array}$ with g odd and saturating, $\bar { p }$ is $\frac { 1 } { 2 }$ plus the mean of $g ( { \bar { A } } )$ over the box, up to the near-constant ofset contributed by the spatial term of Equation (6), and that mean vanishes only when $\bar { A }$ is symmetric. $\bar { p }$ therefore records the asymmetry of the evidence. Where a minority of cells sits far above the box mean, the saturation of σ compresses their contribution and the mild majority below the mean dominates, giving $\begin{array} { r } { \bar { p } < \frac { 1 } { 2 } ; } \end{array}$ a box filled by crown apart from a few much darker cells gives $\bar { p } > \frac { 1 } { 2 }$ . Read this way $\bar { p }$ is a soft estimate of the fraction of its box that the crown occupies. The decisiveness m folds each cell to its distance from $\frac { 1 } { 2 }$ before averaging, and so records the spread of the same distribution: $m = 1$ when every cell is confidently inside or outside, $m = 0$ when every cell is a coin flip. Folding before averaging is what makes it a measure of spread rather than of level, since a box split half confidently-crown and half confidently-background gives $m \approx 1$ but $\begin{array} { r } { \bar { p } = \frac { 1 } { 2 } } \end{array}$ . The two are near-orthogonal descriptions of one posterior: its asymmetry and its dispersion.

The three quantities are combined by multiplication,

$$
s ^ { \prime } ( B ) \ = \ s ( B ) \bar { p } ( B ) \ m ( B ) ,\tag{13}
$$

and predictions are ranked by $s ^ { \prime }$ in place of s. Independence is what licenses the product: two estimates of the same event that are independent given the truth combine multiplicatively, and the construction of Section 3.2.3 supplies that independence architecturally. The form is also a conjunction, which is the behaviour we want. In log space Equation (13) is a sum, so a low value on any one factor cannot be rescued by the other two, and a confident detection carrying a collapsed posterior falls just as a well-separated mask on a box the detector disbelieves does.

Equation (13) adds no parameter, no fitting stage and no forward pass. Both $\bar { p }$ and m are two reductions over a vector Equation (6) already returns, so nothing inside the module changes and no quantity is recomputed; the predicted masks, the detection set and the number of predictions are all identical before and after, and only the ordering moves. Because $s ^ { \prime }$ is a product of three quantities in [0, 1] it does not share a scale with s, so the 0.05 decode floor of Section 3.2.2 is applied to s at decode and the resulting set is then held frozen; applying a floor to $s ^ { \prime }$ would select a diferent set of detections and the comparison would no longer be like for like. The efect of Equation (13) is reported in Section 4.1.

Nothing in Equation (13) is estimated from data, so unlike the three scalars of Section 3.2.4 it introduces no fitted quantity and requires no validation mask labels to set. The functional form was chosen on the same 108 held-out validation images, by box average precision over a small set of parameter-free candidates, and the unweighted product was selected as the exponent-free member of a group that could not be separated at that precision. We therefore count Equation (13) as a choice of form rather than as a fourth calibrated scalar.

## 3.2.6 Implementation

Table 2 collects every key configuration used.

The EM fit is CPU-only and completes in minutes. It is performed once, and the resulting masker is shared across all detector seeds, so the seed-to-seed variation reported in Section 4.1 is detector variation.

## 4 Results

Tree Crown Instance Segmentation For TCIS, only OAM-TCD is a suitable end-to-end evaluation dataset since it contains instance masks which are required for test-time evaluation. We restrict our evaluation to OAM-TCD’s Tree class (cat=1). One approach in the literature [3], which we agree with and utilize, is to ignore all canopy annotations (cat=2) altogether. Unsurprisingly, the model’s predictions often fall inside areas marked as canopy, which contain many closely grouped and entirely valid tree crowns, but we do not attribute any positive or negative score to these predictions and they are excluded from the evaluation metrics entirely (SelvaBox deletes all pixels in canopy areas; we instead use COCO’s iscrowd=1 to ignore them). The rationale here is that the annotating of canopy into groups necessarily leads to subjective groupings. For example, should a clump of 10 tree crowns be best delineated by 1 canopy of 10 crowns, or 2 of 5? We claim both are equally valid, and by using subjective groupings, the $\mathrm { A P _ { 5 0 } }$ metric becomes corrupted since IoU > 0.5 is based on agreement of delineations. Every OAM-TCD figure we report is therefore "canopy-neutral", unless otherwise specified.

Tables 3 and 4 summarise our model’s performance across COCO average precision (AP; with a single class, mAP and AP coincide) for OAM-TCD and instance-level detection metrics at IoU 0.4 for NeonTreeEvaluation respectively, following the authors’ conventions. We report evaluations alongside baseline model results for comparison. For simplicity, consistency, and replicability, we use seed values starting at 0 and incremented by 1 (i.e., 0, 1, 2 for 3 seeds). Every baseline in Table 3 was re-run and re-scored by us under a single protocol. Each baseline was run in its own native prediction geometry, so only the measurement is held fixed, and we comment in the table notes on any configuration choice that materially afects a row.

For all OAM-TCD tests, our scorer is standard pycocotools COCOeval (iouType segm then bbox), pooled over images and with a single category ‘tree’. Full details are in Appendix B: the scorer configuration in Table 16 and each row’s prediction geometry in Table 17. We include all 439 images in the GT always, without dropping images containing zero predictions.

Table 3 includes Box2Mask, the only model with identical supervision to LACE (900 images, boxes only). With its strongest backbone it reaches 0.540 mask $\mathrm { A P _ { 5 0 } }$ against LACE’s 0.663 on identical training crops, an identical tiling grid and the same scorer. It should be noted that Box2Mask was developed on COCO objects spanning hundreds of pixels, whereas the median OAM-TCD crown at 0.1 m/px is significantly smaller.

Table 2: Configuration. Encoder and masker-fitting hyperparameters were fixed a priori; the three inference scalars were selected on validation (Section 3.2.4); detector optimisation follows standard practice for anchor-free centre-based heads. The ranking rule contributes no fitted quantity (Section 3.2.5).
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Encoder</td><td>DINOv3 ViT-L/16, frozen, layer 24 only, 1024 dimensions (NEON: layers 21–24, 4096)</td></tr><tr><td>Feature grid</td><td>4 offsets  $( \delta _ { y } , \delta _ { x } ) \in \{ 0 , 8 \} ^ { 2 }$  , interlaced to 8 px;  $2 5 6 \times 2 5 6$  cells per 2048 px tile</td></tr><tr><td>Detector</td><td> $1 \times 1$  stem to width 256;  $5 \times 3 \times 3$  blocks, group norm, ReLU; heatmap and 4-channel regression heads; stride 8</td></tr><tr><td>Target encoding</td><td>CenterNet Gaussians, grid 256, stride 8, radius at IoU overlap 0.7, floor  $\geq 1$  cell</td></tr><tr><td>Detector training</td><td>Adam, lr  $1 0 ^ { - 3 }$  , weight decay  $1 0 ^ { - 4 }$  , cosine schedule, batch 3, max 40 epochs, early stopping on validation box  $\mathrm { A P _ { 5 0 } }$  (minimum 12 epochs, patience 2), best-on-validation checkpoint</td></tr><tr><td>Decoding</td><td>local maxima, score  $> 0 . 0 5 .$  top 600 per tile; operating point 0.4 for single-threshold metrics</td></tr><tr><td>Masker – feature space</td><td>120 training tiles; PCA  $d = 1 2 8$ </td></tr><tr><td>Masker – mixtures</td><td> $K _ { f g } = 1 6$  foreground prototypes;  $K _ { b g } = 1 2$  background components; concentration  $\kappa = 1 0 ;$  contrastive weight  $\beta = 0 . 5$ </td></tr><tr><td>Masker – spatial prior</td><td> $N = 8$  bins per axis; 3 size bands at the training terciles; mean per-bin foreground mass  $\pi / 4 ;$  per-bin cap 0.95</td></tr><tr><td>Masker – EM schedule</td><td>30 iterations, no convergence test; prototype pruning after 5 iterations at responsibility share below  $0 . 2 5 / K _ { f g }$  , a quarter of uniform</td></tr><tr><td>Inference</td><td>prior weight  $\alpha = 0 . 3 ;$  concentration  $\kappa = 1 6$  (fitted at 10); mask threshold  $\tau = 0 . 2 5 ;$  masks rasterised at 512 px per tile</td></tr><tr><td>Ranking</td><td> $s ^ { \prime } = s \bar { p } m$  (Equation (13)); no fitted parameter; decode floor applied to  $s ,$  detection set frozen</td></tr></table>

Table 3: Instance segmentation on the OAM-TCD 439-image holdout. Every row was produced and scored by us under one frozen protocol (using standard COCOeval). All are trained on OAM-TCD, and Imgs is the number of OAM-TCD training images seen. Restor and SelvaBox were not fine-tuned by us; the others were.
<table><tr><td>Method</td><td>Superv.</td><td>Imgs</td><td> $\bf { A P _ { 5 0 } }$ </td><td> $\bf { A P _ { 5 0 : 9 5 } }$ </td><td>Box  $\bf { A P _ { 5 0 } }$ </td></tr><tr><td>Restor Mask R-CNNa</td><td>masks</td><td>4169</td><td>0.626</td><td>0.277</td><td>0.654</td></tr><tr><td>Detectree2b</td><td>masks</td><td>900</td><td>0.597</td><td>0.258</td><td>0.591</td></tr><tr><td>SelvaBox → SAM 3c</td><td>box+SAM</td><td>3024</td><td>0.569</td><td>0.203</td><td>0.730</td></tr><tr><td>Box2Mask Swin-Ld</td><td>box</td><td>900</td><td>0.540</td><td>0.198</td><td>0.582</td></tr><tr><td>Box2Mask  $\mathrm { R } { - } 5 0 ^ { \mathrm { d } }$ </td><td>box</td><td>900</td><td>0.387</td><td>0.130</td><td>0.471</td></tr><tr><td>LACE (ours) (3-seed)</td><td>box</td><td>900</td><td>0.663</td><td>0.279</td><td>0.640</td></tr><tr><td></td><td></td><td></td><td>±0.001</td><td>±0.004</td><td>±0.024</td></tr></table>

Table 4: Individual tree-crown detection on the NeonTreeEvaluation benchmark (RGB-only, IoU 0.4, 194 tiles; precision/recall macro-averaged over images). Metrics at each model’s best- $F _ { 1 }$ operating point; max R is the recall ceiling (score threshold → 0)
<table><tr><td>Method</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>maxR</td></tr><tr><td>DeepForest, publisheda</td><td>0.659</td><td>0.790</td><td>0.719</td><td></td></tr><tr><td>DeepForest, replicated  $( \mathrm { v } 2 . 1 . 0 ) ^ { \mathrm { b } }$ </td><td>0.745</td><td>0.709</td><td>0.726</td><td>0.765</td></tr><tr><td>Ours (5-seed)</td><td> $0 . 7 2 7 \pm 0 . 0 0 9$ </td><td>0.729 ± 0.009</td><td>0.728 ± 0.003</td><td> $0 . 8 4 9 \pm 0 . 0 0 3$ </td></tr></table>

<sup>a</sup>Weinstein et al. (2021, PLoS Comput. Biol.), Table 3 image-annotated RGB operating point; full PR-curve / recall ceiling not reported. <sup>b</sup>deepforest v2.1.0 pretrained release model, evaluated by us on the same 194 tiles (deterministic; no seed variance). All rows scored with the benchmark authors’ deepforest.evaluate\_boxes.

Sparse-canopy holdout To test model eficacy on sparse-canopy biomes such as savannah and dryland, we take a ‘sparse-canopy’ slice of the OAM-TCD data: 236 tiles carrying 14,937 crowns, none of which appears among the 900 training images. We reuse models on this slice without any additional training or fine-tuning; Table 5 reports the result. To generate the slice, we categorise by the OAM-TCD dataset’s own metadata which includes a biome category for each image and select the most appropriate biomes by name, listed in Table 6. The slice is sparse by FAO’s yardstick as well as by name: labelled tree cover on its median tile is 7.4%, below the 10% canopy cover at which FAO counts land as forest [82], and 59.5% of its tiles fall under that line, compared with 31.0% of the tiles drawn from the six forest biomes (labelled cover is a lower bound wherever annotators left gaps). Table 7 isolates the box-to-mask module from detection by prompting it, and two SAM 3 variants, with identical ground-truth boxes on all 439 tiles.

Table 5: Instance segmentation on the sparse-canopy OAM-TCD holdout (236 tiles, 14,937 crowns, no tile shared with the 900 training images). Canopy is scored as iscrowd throughout, as in Table 3. LACE bands are ddof = 1 over detector seeds 0–2, Detectree2 is single seed.
<table><tr><td rowspan="2">Method</td><td colspan="3">Canopy neutral</td><td rowspan="2">Box  $\bf { A P _ { 5 0 } }$ </td></tr><tr><td> $\bf { A P _ { 5 0 } }$ </td><td> $\mathbf { A P _ { 7 5 } }$ </td><td>AP50:95</td></tr><tr><td>Detectree2</td><td>0.6118</td><td>0.2155</td><td>0.2800</td><td>0.5897</td></tr><tr><td rowspan="2">LACE (ours)</td><td>0.6913</td><td>0.2326</td><td>0.3148</td><td>0.6442</td></tr><tr><td>±0.0095</td><td>±0.0107</td><td>±0.0069</td><td>±0.0272</td></tr></table>

Scored with the same frozen COCOeval protocol as Table 3 (pycocotools, masks at $\overline { { { 5 1 2 ^ { 2 } } } } .$ , maxDets 600, canopy as iscrowd). Both methods score higher here than on the full 439 (LACE 0.691 against 0.663; Detectree2 0.612 against 0.597).

## 4.1 Ablation Studies

We ablate the three central contributions: the interlaced feature grid, the training-free box-tomask module, and the ranking rule that multiplies the module’s posterior into the detector score. Unless stated otherwise, ablations of the module hold the detector fixed, so every arm scores the identical set of predicted boxes and any diference is attributable to mask generation alone. Bands are the sample standard deviation (ddof = 1) over independent seeds.

Table 6: Composition of the sparse-canopy slice. Tiles are selected by WWF biome label, excluding all 900 training tiles. One eligible tile of biome 12 was dropped at build time as unreadable, leaving 236. Crown px and canopy px are the mean per-tile fraction of labelled crown and labelled canopy pixels.
<table><tr><td>Code Biome (WWF)</td><td></td><td>Tiles Crown px</td><td>Canopy px</td></tr><tr><td colspan="4">Sparse-canopy slice</td></tr><tr><td>7</td><td>Trop./Subtrop. Grassland, Savanna &amp; Shrubland</td><td>102</td><td>2.3% 8.7%</td></tr><tr><td>12</td><td>Mediterranean Forest, Woodland &amp; Scrub</td><td>56</td><td>4.8% 7.6%</td></tr><tr><td>8</td><td>Temperate Grassland, Savanna &amp; Shrubland</td><td>31</td><td>4.8% 11.5%</td></tr><tr><td>13</td><td>Desert &amp; Xeric Shrubland</td><td>24</td><td>6.2% 15.2%</td></tr><tr><td>10</td><td>Montane Grassland &amp; Shrubland</td><td>16</td><td>1.1% 1.9%</td></tr><tr><td>9</td><td>Flooded Grassland &amp; Savanna</td><td>7</td><td>1.0% 0.8%</td></tr><tr><td colspan="4">Subtotal 236</td></tr><tr><td colspan="4"></td></tr><tr><td>Excluded 1</td><td>Trop./Subtrop. Moist Broadleaf F.</td><td>3.6%</td><td>29.0%</td></tr><tr><td>4</td><td>Temperate Broadleaf &amp; Mixed F.</td><td></td><td>2.7% 23.9%</td></tr><tr><td>5</td><td>Temperate Conifer F.</td><td></td><td>2.5% 25.3%</td></tr><tr><td>2</td><td>Trop./Subtrop. Dry Broadleaf F.</td><td></td><td>3.8% 25.8%</td></tr><tr><td>3</td><td>Trop./Subtrop. Conifer F.</td><td></td><td>1.7% 32.6%</td></tr><tr><td>14</td><td>Mangrove</td><td></td><td>1.6% 5.9%</td></tr><tr><td>6</td><td>Boreal Forest / Taiga</td><td>3.5%</td><td>13.7%</td></tr><tr><td>11</td><td>Tundra</td><td>3.7%</td><td>14.9%</td></tr><tr><td>98</td><td>(non-WWF code)</td><td>1.4%</td><td>7.6%</td></tr><tr><td>-1</td><td>Unmatched</td><td></td><td>1.8% 18.4%</td></tr></table>

Table 7: Box-to-mask modules prompted with ground-truth boxes, OAM-TCD 439-tile holdout, 25,692 crowns (the 13 crowns whose mask covers fewer than 4 pixels at the $5 1 2 ^ { 2 }$ reference raster are excluded).
<table><tr><td>Box-to-mask module</td><td>Mean IoU</td><td>≥ 0.5</td><td> $\ge 0 . 7 5$ </td><td> $\geq \mathbf { 0 . 9 }$ </td></tr><tr><td>LACE commonality EM (ours)</td><td>0.770</td><td>0.979</td><td>0.659</td><td>0.051</td></tr><tr><td>SAM 3, zero-shota</td><td>0.741</td><td>0.956</td><td>0.545</td><td>0.037</td></tr><tr><td>SAM 3, SelvaMask fine-tunedb</td><td>0.623</td><td>0.789</td><td>0.274</td><td>0.011</td></tr></table>

<sup>a</sup> facebook/sam3 image model, box-prompted at the native 2048<sup>2</sup> tile.  
<sup>b</sup> CanopyRS/sam3-multi-selvabox-selvamask-FT, run at the 1777 px tile size i.e. the configuration its authors deploy. All 685 checkpoint tensors were asserted to match before inference.

## 4.1.1 Feature Grid: Interlacing Versus Interpolation

DINOv3 emits one embedding per $1 6 \times 1 6$ patch. Our encoder evaluates the frozen backbone at four spatial ofsets and interlaces the results into a real 8 px grid. The obvious alternative is to simply interpolate the native 16 px grid to 8 px. Table 8 isolates that choice on NeonTreeEvaluation: encoder, detector, targets, tiles and scorer are held identical, and the encode/decode path is byte-identical between arms (TargetConfig(grid=64, $\mathsf { s t r i d e { = } } 8 )$ ), so the comparison is purely feature quality.

Table 8: Interlaced versus interpolated 8 px features on NeonTreeEvaluation (194 tiles, 5 seeds per arm, detection F1 at IoU 0.4). Both arms share the same detector, targets and scorer.
<table><tr><td>Feature grid</td><td>F1@0.4</td><td>max Recall</td><td>NIWO site F1</td></tr><tr><td>Native 16 px, interpolated to 8 px</td><td> $0 . 6 8 9 \pm 0 . 0 1 3$ </td><td> $0 . 7 7 9 \pm 0 . 0 0 9$ </td><td> $0 . 4 2 7 \pm 0 . 0 1 0$ </td></tr><tr><td>Interlaced 4-offset, real 8 px</td><td> $\mathbf { 0 . 7 2 8 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 8 4 9 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 5 8 8 \pm 0 . 0 1 7 }$ </td></tr><tr><td> $\Delta$ </td><td>+0.039</td><td>+0.070</td><td>+0.161</td></tr></table>

Interlacing is not equivalent to upsampling: real and interpolated cells agree only at cos ≈ 0.955. Registration was verified independently (blob test, $\leq 1$ cell over 200 positions). The NIWO gain is roughly 10× the seed standard deviation and holds precision.

The gain is largest at NIWO, the densest and smallest-crown site. We observe here and elsewhere that this is the context where the 16 px grid struggles most, since it under-resolves small individual crowns. Interpolation adds resolution to the sampling lattice but no new evidence, whereas the four ofsets add evidence.

## 4.1.2 Hyperparameter tuning

The module’s hyperparameters were fixed a priori except for the contrastive repulsion strength $\beta$ and the three inference-time scalars of Section 3.2.4: the spatial-prior weight $\alpha ,$ the concentration $\kappa$ (raised from its fit-time value of 10 to 16) and the mask threshold $\tau .$ Table 9 ablates α and κ on OAM-TCD, with $\tau$ held at 0.25 throughout; $\beta$ is ablated in Table 10, which separates it from the E-step recentring our implementation ties it to.

Table 9: Inference scalars on the OAM-TCD 439-tile test split, selected on 108 held-out validation images. Both rows use detector seeds 0–2, so the comparison is paired.
<table><tr><td>Configuration</td><td>Mask  $\mathbf { A P } _ { 5 0 }$ </td><td>Mask  $\mathbf { A P } _ { 5 0 : 9 5 }$ </td></tr><tr><td>α = 1.0,  $\kappa = 1 0$  (3 seeds)</td><td> $0 . 6 2 3 2 \pm 0 . 0 0 2 6$ </td><td> $0 . 2 4 0 4 \pm 0 . 0 0 3 9$ </td></tr><tr><td> $\alpha = 0 . 3 .$   $\kappa = 1 6$  (3 seeds)</td><td> $\mathbf { 0 . 6 3 0 1 \pm 0 . 0 0 5 4 }$ </td><td> $\mathbf { 0 . 2 5 6 8 \pm 0 . 0 0 0 5 }$ </td></tr></table>

α and κ were chosen by a rule pre-registered before any validation cell was read, and the test split was not used for their selection. The gain from the knobs is +0.0069 mask $\mathrm { A P _ { 5 0 } ; }$ an earlier five-seed vanilla band $( 0 . 6 1 5 \pm 0 . 0 1 1 )$ overstated it $\mathrm { a t + 0 . 0 1 5 }$ by including two seeds absent from the knobbed arm. The validation surface is a broad plateau (validation $\mathrm { A P _ { 5 0 } }$ 0.630–0.639 across the $\alpha \times \kappa$ grid; these are validation figures on 108 images and are not comparable to the test bands above, which they resemble only by coincidence), so these values are not uniquely optimal; what is robust is directional: $\alpha < 1$ beats $\alpha = 1$ at every $\kappa ,$ and $\kappa > 1 0$ beats $\kappa = 1 0$ at every $\alpha .$ The values themselves are coarse: $\kappa = 1 6$ is simply the best of {10, 13, 16, 20}, a range over which $\mathrm { A P _ { 5 0 } }$ moves by under 0.003. The mask threshold $\tau = 0 . 2 5$ is the third such scalar; the validation grid spanned $\tau \in \{ 0 . 2 5 , 0 . 3 0 \}$ , and the earlier move from 0.5 to 0.25 predates the validation protocol. Because these scalars are fitted against validation mask labels, we describe them as selected on 108 held-out images rather than as label-free.

Turning to $\beta \colon$ our implementation ties the E-step recentring flag to it, so a $\beta = 0$ arm also disables recentring and the two cannot be read apart from a single pair of runs. Table 10 crosses them on the validation split.

Table 10: M-step repulsion against E-step recentring, $K _ { f g } = 1 6 .$ EM seed 0, detector seed 0, 108 validation tiles. Mean per-crown mask IoU over 7041 crowns for the oracle-box columns and 5428 box-matched true positives for the predicted-box columns.
<table><tr><td>M-step</td><td>E-step</td><td>GT Oracle boxes</td><td>Predicted boxes</td></tr><tr><td>generative  $( \beta = 0 )$ </td><td>absolute</td><td>0.7402</td><td>0.6717</td></tr><tr><td>generative  $( \beta = 0 )$ </td><td>recentred</td><td>0.7895</td><td>0.7188</td></tr><tr><td>contrastive  $( \beta = 0 . 5 )$ </td><td>absolute</td><td>0.7758</td><td>0.7063</td></tr><tr><td>contrastive  $( \beta = 0 . 5 )$ </td><td>recentred</td><td>0.7885</td><td>0.7193</td></tr><tr><td>repulsion alone</td><td></td><td>+0.0356</td><td>+0.0346</td></tr><tr><td>recentring alone</td><td></td><td>+0.0493</td><td>+0.0471</td></tr><tr><td>both</td><td></td><td>+0.0483</td><td>+0.0476</td></tr><tr><td>interaction (both — sum of singles)</td><td></td><td>-0.0366</td><td>-0.0341</td></tr></table>

Efects are relative to the generative/absolute cell. The two mechanisms are substitutes rather than complements: either alone recovers roughly +0.04, and both together recover the same +0.048, giving a large negative interaction in both box regimes. The interaction is the departure from additivity: were the two independent, enabling both would give $+ 0 . 0 8 4 9$ on oracle boxes, whereas it gives $+ 0 . 0 4 8 3$ . Recentring is the slightly larger term throughout, consistent with the ablation on our development set, where removing recentring drives the fraction of mask mass in the box corners from 0.37 to 0.61 against 0.37 to 0.41 for removing repulsion.

## 4.1.3 Confidence from the Mask Posterior

Section 3.2.5 replaces the ranking key s by the product $s ^ { \prime } = s \bar { p } m$ . Because the detection set is frozen and only the ordering moves, every arm below scores an identical set of predictions, and the masks themselves are byte-identical throughout. Table 11 reports the efect on both splits at three detector seeds.

Table 11: Ranking by the mask posterior. Both splits use the same frozen scorer and identical predictions per seed; only the ordering difers. The sparse split shares no tile with the 900 training images, so it is a generalisation test rather than a second read of the same distribution.
<table><tr><td>Split</td><td>Ranking</td><td> $\bf { A P 5 0 }$ </td><td> $\mathbf { A P 7 5 }$ </td><td> $\bf { A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td rowspan="2">OAM-TCD 439</td><td rowspan="2">s (detector score)  $s ^ { \prime } = s \bar { p } m$ </td><td> $0 . 6 3 0 \pm 0 . 0 0 5$ </td><td> $0 . 1 4 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 5 7 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathbf { 0 . 6 6 3 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 6 5 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 2 7 9 \pm 0 . 0 0 4 }$ </td></tr><tr><td rowspan="3">Sparse 236</td><td>Detectree2 (FT)</td><td>0.612</td><td>0.216</td><td>0.280</td></tr><tr><td>s (detector score)</td><td> $0 . 6 5 6 \pm 0 . 0 1 2$ </td><td> $0 . 1 9 7 \pm 0 . 0 0 9$ </td><td> $0 . 2 9 0 \pm 0 . 0 0 7$ </td></tr><tr><td> $s ^ { \prime } = s \bar { p } m$ </td><td> $\mathbf { 0 . 6 9 1 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 2 3 3 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 3 1 5 \pm 0 . 0 0 7 }$ </td></tr></table>

Canopy-neutral mask AP, 3 detector seeds, bands are ddof = 1.  
Both factors contribute (Table 12): dropping either loses roughly a third of the gain (+0.0234 and +0.0222 against +0.0365 together).

Table 12: Dropping each factor of Equation (13), detector seed 0, OAM-TCD 439 (canopy-neutral mask AP) with box $\mathrm { A P _ { 5 0 } }$ on the 108 validation images. Every row ranks the same frozen set of 159,687 predictions.
<table><tr><td>Ranking</td><td> $\bf { A P 5 0 }$ </td><td> $\mathbf { A P 7 5 }$ </td><td> $\bf { A P _ { 5 0 : 9 5 } }$ </td><td>Val box  $\bf { A P 5 0 }$ </td></tr><tr><td>S</td><td>0.6250</td><td>0.1445</td><td>0.2581</td><td>0.4982</td></tr><tr><td>s m (drop p)</td><td>0.6484</td><td>0.1577</td><td>0.2719</td><td>0.5148</td></tr><tr><td> $s \bar { p }$  (drop m)</td><td>0.6472</td><td>0.1638</td><td>0.2737</td><td>0.5094</td></tr><tr><td> $s { \bar { p } } m$ </td><td>0.6615</td><td>0.1728</td><td>0.2822</td><td>0.5177</td></tr></table>

The sparse-split rows of Table 11 suggest Equation (13) is not tuned to the test distribution.

On 236 tiles which do not overlap with the training/validation set at all, the gain is larger rather than smaller (+0.0354 against +0.0325).

## 4.1.4 Dense canopy

Foreground/background commonality within a box rests to some extent on the assumption that the box contains mostly one crown, meaning crowded trees with overlapping crowns could be a challenge.

We call a crown touching when its ground-truth box overlaps that of another crown, which happens to be the case for 39.3% of the 25,705 test crowns. Touching and isolated crowns score almost identically: mean mask IoU $0 . 7 0 9 3 \pm 0 . 0 0 1 5$ against $0 . 7 1 4 8 \pm 0 . 0 0 1 9$ , and they fail at statistically indistinguishable rates $( 0 . 0 2 5 0 \pm 0 . 0 0 3 1$ versus $0 . 0 2 5 2 \pm 0 . 0 0 0 8$ , where failure is a box-matched true positive whose mask IoU falls below 0.5). Touching crowns are also larger (mean box side wh of 70.2 versus 44.1 px), which may partly ofset neighbour interference.

We can examine the tile-level labelled-canopy fraction to give a (coarser) corroboration of the same conclusion (Table 13).

Table 13: Performance by labelled-canopy fraction, 439 OAM-TCD test tiles, 3 seeds. AP is pooled within each bin against that bin’s own ground-truth count. The mask-minus-box gap is the module’s contribution once detection is accounted for.
<table><tr><td>Canopy-class fraction</td><td>Tiles</td><td>Mask  $\mathbf { A P } _ { 5 0 }$ </td><td>Box  $\mathbf { A P } _ { 5 0 }$ </td><td>Mask – Box</td></tr><tr><td>[0, 0.10)</td><td>196</td><td>0.5838</td><td>0.5611</td><td>+0.0227</td></tr><tr><td>[0.10, 0.25)</td><td>95</td><td>0.6900</td><td>0.6655</td><td>+0.0245</td></tr><tr><td>[0.25, 0.50)</td><td>86</td><td>0.7162</td><td>0.6902</td><td>+0.0260</td></tr><tr><td>[0.50, 1.00]</td><td>62</td><td>0.6563</td><td>0.6238</td><td>+0.0325</td></tr></table>

Bins are the fraction of the tile covered by OAM-TCD’s canopy class (groups of trees, $\mathsf { c a t } = 2 )$ , which excludes individually labelled crowns; tile counts sum to the 439 test tiles. Box $\mathrm { A P _ { 5 0 } }$ is masker-invariant and is reported so the module’s contribution can be read independently of detection.

Table 13 shows a higher labelled-canopy fraction does not cause a decline in $\mathrm { A P _ { 5 0 } }$ . Mask $\mathrm { A P _ { 5 0 } }$ is lowest on the most open tiles (0.5838 in the [0, 0.10) bin) and higher wherever more of the tile is labelled canopy, peaking at 0.7162 in the [0.25, 0.50) bin. The mask-minus-box gap, i.e. the masker’s own contribution, stays between +0.023 and +0.033 across the four bins and in fact rises slightly as canopy becomes more dense, if anything.

## 5 Discussion

## 5.1 A note of caution

Taking a step back, the conversation relating to the carbon sequestration of trees globally requires some nuance. In a warming world, it is tempting to reduce the objective to something along the lines of ’tree maximalism’ i.e. more trees is always better. However, as discussed by [4], it is vital to factor in the biomes and environments before drawing unfounded conclusions about

## 5.2 TCIS

On the 439-image OAM-TCD test set, LACE scores the highest $\mathrm { A P _ { 5 0 } }$ of 0.663 and efectively ties with Restor’s Mask R-CNN at $\mathrm { A P _ { 5 0 : 9 5 } }$ despite some significant limitations. Foremost is its handicapped training regime of only boxes rather than masks, and also its limited training set of 900 images rather than 4169. LACE appears competitive with fully supervised models and therefore presents a viable option for TCIS tasks utilising much more time- and cost-eficient box annotations, and has demonstrated eficacy over a wide range of biomes. This is made possible by the quality of the DINOv3-web encoder, which, through its massive pretraining set of 1.7 billion images, not to mention its authors’ numerous innovations, is highly robust to domain transfer. In this way, LACE presents a new approach to TCIS, particularly for sparse-canopy environments where the focus is less on precise delineation and instead on measuring tree crown count, size, and allometries over large, diverse landscapes.

By taking the 8 px interlaced grid as the unit of crown delineation, LACE concedes an inbuilt disadvantage relative to pixel-based maskers, namely that it is fundamentally limited to a coarser resolution. This would be expected to cost most at high IoU thresholds such as $\mathrm { A P _ { 7 5 } , A P _ { 9 0 } }$ , but less at $\mathrm { A P _ { 5 0 } }$ , where boundary precision is less strict. This is borne out in the narrowing gap between LACE’s $\mathrm { A P _ { 5 0 } }$ lead of 0.037 vs its $\mathrm { A P _ { 5 0 : 9 5 } }$ which is within seed noise of Restor’s Mask R-CNN.

## 5.2.1 Domain gap in SelvaMask

The sam3-multi-selvabox-selvamask-FT checkpoint was fitted on tropical UAV imagery at 1.3–3.5 cm/px and is applied here at 10 cm/px. As shown in the results, it nevertheless exhibits best-in-class scores for Box $\mathrm { A P _ { 5 0 } }$ , demonstrating the capabilities of its DINO (DETR) Swin-L based detector and rich tree crown fine-tuning across high-quality, diverse datasets including SelvaBox. SelvaMask is a dataset of highly dense tropical forest canopy, so the fine-tuned masker checkpoint is likely still not optimised for the open landscapes of OAM-TCD.

## 5.2.2 Spatial Prior

Although the spatial prior is maintained separately for three crown-size bands, in practice we observe very little diference between the three fitted priors, contrary to our expectations. We leave the stratification in as a defensive component that does not add any significant complexity, but acknowledge that a single pooled prior would likely perform comparably.

## 5.3 Comment on crown annotations in dense canopy

Helpfully, NeonTreeEvaluation contains many GT annotations within dense canopy areas, annotated individually allowing for precise individual instance segmentation evaluation even in canopy areas. Existing datasets that try to provide instance segmentation labels very often will tend to attempt to draw definitive ‘ground truth’ on dense canopy where there arguably is simply not enough information in the image to make that determination definitively [18, 83]. There are a variety of potential solutions to this challenge, and naturally, there is a need for pragmatism in annotating a close approximation of the actual ground truth in a scenario where it is impossible to be certain. Nevertheless, we observe that it is characteristic of datasets that annotate individual tree crowns in dense canopy scenarios, either with rectangular bounding boxes or closed polygons/masks, to leave a large number of gaps between units. As an approximation and indeed for pixel-wise classification loss, this can be suficient, but introduces some important nuance when it comes to inference-time test set benchmarks.

Firstly, where GT annotations leave non-trivial gaps between annotation boundaries, a model that produces more inclusive, justified boundaries can be penalised. Secondly, pycocotools standard mean average precision $\mathrm { ( m A P _ { 5 0 } ) }$ is based on Intersection over Union (IoU) and, for each ground truth (can be bounding box or polygon), algorithmically looks for predictions that overlap with the ground truth, ordered by confidence score (descending). If one is found that exceeds 0.5, i.e., the area of intersection between prediction and ground truth (GT) is 50% or more of the union of the two, that prediction is counted as a True Positive (TP). In annotated dense canopy, this entails that the resulting benchmark contains some subjectivity, i.e., many predictions that are defensible and possibly even more accurate than the ground truth can be counted as False Positives (FP). This is particularly an issue with RGB imagery that includes trees of varying scales i.e. smaller and larger, as a mismatched scale between prediction and GT leads to much higher FP rates since (m)AP is highly sensitive to scale mismatch for purely geometric reasons: a small prediction’s overlap with a large GT is much less likely to cross the 50% IoU threshold, and equally a large prediction’s overlap with a small GT is much less likely to cross it. If one of the two is less than 50% of the area of the other, it will be counted as a FP regardless of placement. In contexts such as dense canopy where, for example, a single large crown could be one tree or two trees, the metric is sensitive to the arbitrary choice of division. The same is true of Precision, Recall, and F1 metrics derived from instance predictions (not including area-/pixel-based Precision, Recall, or F1 metrics which are calculated over total pixels rather than individual bounding boxes or polygons).

## 6 Conclusion

In this study we present LACE, a domain-agnostic model for instance segmentation in boxsupervised Tree Crown tasks, where annotations are scarce. Our box-to-mask inference method, by leveraging the statistical signature of the DINOv3 latent composite, provides a SOTAcompetitive solution to true Tree Crown Instance Segmentation (TCIS) without the need for time-consuming and/or expensive polygon masks. With this contribution, we hope to support the Remote Sensing community in the previously underexplored area of TCD and TCIS in diverse global biomes, particularly sparse-canopy landscapes. We also believe the designs discussed here could be transferable to other Computer Vision domains where availability of annotated data is also a known challenge, since we did not rely on domain-specific priors to any significant degree, although this is beyond the scope of this work.

The eficacy of each component within LACE has been validated through comprehensive ablation studies.

## Author Contributions

Conceptualisation, methodology, software, experiments and writing – original draft, T.P.; supervision and writing – review and editing, K.L. and B.L. All authors have read and agreed to the published version of the manuscript.

## Funding

This research received no external funding.

## Data Availability

Both datasets used in this study are publicly available: the OAM-TCD dataset [33] and the NeonTreeEvaluation benchmark [17] (v1.8.0). All code for LACE, together with the training configurations, model checkpoints and evaluation scripts needed to reproduce every table in this paper, will be released upon publication.

## Conflicts of Interest

The authors declare no conflicts of interest.

## Abbreviations

AP, average precision; CHM, canopy height model; CNN, convolutional neural network; DINO, self-distillation with no labels; DSM, digital surface model; EM, expectation-maximisation; FAO, Food and Agriculture Organization of the United Nations; FPN, feature pyramid network; GSD, ground sample distance; GT, ground truth; IoU, intersection over union; ITC, individual tree crown; NEON, National Ecological Observatory Network; NMS, non-maximum suppression; OAM, OpenAerialMap; PCA, principal component analysis; RPN, region proposal network; RS, remote sensing; SAM, Segment Anything Model; SOTA, state of the art; SSL, self-supervised learning; TCD, tree crown detection; TCIS, tree crown instance segmentation; TCSS, tree crown semantic segmentation; UAV, unmanned aerial vehicle; ViT, vision transformer; WWF, World Wide Fund for Nature.

## References

[1] Pierre Hiernaux, Hassane Bil-Assanou Issoufou, Christian Igel, Ankit Kariryaa, Moussa Kourouma, Jérôme Chave, Eric Mougin, and Patrice Savadogo. Allometric equations to estimate the dry mass of Sahel woody plants mapped with very-high resolution satellite imagery. For. Ecol. Manage., 529:120653, 2023. doi: 10.1016/j.foreco.2022.120653.

[2] Compton Tucker, Martin Brandt, Pierre Hiernaux, Ankit Kariryaa, Kjeld Rasmussen, Jennifer Small, Christian Igel, Florian Reiner, Katherine Melocik, Jesse Meyer, Scott Sinno, Eric Romero, Erin Glennie, Yasmin Fitts, August Morin, Jorge Pinzon, Devin McClain, Paul Morin, Claire Porter, Shane Loefler, Laurent Kergoat, Bil-Assanou Issoufou, Patrice Savadogo, Jean-Pierre Wigneron, Benjamin Poulter, Philippe Ciais, Robert Kaufmann, Ranga Myneni, Sassan Saatchi, and Rasmus Fensholt. Sub-continental-scale carbon stocks of individual trees in African drylands. Nature, 615(7950):80–86, 2023. doi: 10.1038/ s41586-022-05653-6.

[3] Hugo Baudchon, Arthur Ouaknine, Martin Weiss, Mélisande Teng, Thomas R. Walla, Antoine Caron-Guay, Christopher Pal, and Etienne Laliberté. SelvaBox: A high-resolution dataset for tropical tree crown detection. In Proceedings of the 14th International Conference on Learning Representations (ICLR 2026), 2026. doi: 10.48550/arXiv.2507.00170. arXiv:2507.00170.

[4] Matthew E. Fagan. A lesson unlearned? Underestimating tree cover in drylands biases global restoration maps. Glob. Change Biol., 26(9):4679–4690, 2020. doi: 10.1111/gcb.15187.

[5] Martin Brandt, Compton J. Tucker, Ankit Kariryaa, Kjeld Rasmussen, Christin Abel, Jennifer Small, Jerome Chave, Laura Vang Rasmussen, Pierre Hiernaux, Abdoul Aziz Diouf, Laurent Kergoat, Ole Mertz, Christian Igel, Fabian Gieseke, Johannes Schöning, Sizhuo Li, Katherine Melocik, Jesse Meyer, Scott Sinno, Eric Romero, Erin Glennie, Amandine Montagu, Morgane Dendoncker, and Rasmus Fensholt. An unexpectedly large count of trees in the West African Sahara and Sahel. Nature, 587(7832):78–82, 2020. doi: 10.1038/s41586-020-2824-5.

[6] Jean-François Bastin, Nora Berrahmouni, Alan Grainger, Danae Maniatis, Danilo Mollicone, Rebecca Moore, Chiara Patriarca, Nicolas Picard, Ben Sparrow, Elena Maria Abraham, et al. The extent of forest in dryland biomes. Science, 356(6338):635–638, 2017. doi: 10.1126/science.aam6527.

[7] David L. Skole, Jay H. Samek, Moussa Dieng, and Cheikh Mbow. The contribution of trees outside of forests to landscape carbon and climate change mitigation in West Africa. Forests, 12(12):1652, 2021. doi: 10.3390/f12121652.

[8] T. W. Crowther, H. B. Glick, K. R. Covey, C. Bettigole, D. S. Maynard, S. M. Thomas, J. R. Smith, G. Hintler, M. C. Duguid, G. Amatulli, M.-N. Tuanmu, W. Jetz, C. Salas, C. Stam, D. Piotto, R. Tavani, S. Green, G. Bruce, S. J. Williams, S. K. Wiser, M. O. Huber, G. M. Hengeveld, G.-J. Nabuurs, E. Tikhonova, P. Borchardt, C.-F. Li, L. W. Powrie, M. Fischer, A. Hemp, J. Homeier, P. Cho, A. C. Vibrans, P. M. Umunay, S. L. Piao, C. W. Rowe, M. S. Ashton, P. R. Crane, and M. A. Bradford. Mapping tree density at a global scale. Nature, 525(7568):201–205, 2015. doi: 10.1038/nature14967.

[9] Florian Reiner, Martin Brandt, Xiaoye Tong, David Skole, Ankit Kariryaa, Philippe Ciais, Andrew Davies, Pierre Hiernaux, Jérôme Chave, Maurice Mugabowindekwe, Christian Igel, Stefan Oehmcke, Fabian Gieseke, Sizhuo Li, Siyu Liu, Sassan Saatchi, Peter Boucher, Jenia Singh, Simon Taugourdeau, Morgane Dendoncker, Xiao-Peng Song, Ole Mertz, Compton J. Tucker, and Rasmus Fensholt. More than one quarter of Africa’s tree cover is found outside areas previously classified as forest. Nat. Commun., 14:2258, 2023. doi: 10.1038/ s41467-023-37880-4.

[10] Pierre Hiernaux, Lassine Diarra, Valérie Trichon, Eric Mougin, Nogmana Soumaguel, and Frédéric Baup. Woody plant population dynamics in response to climate changes from 1984 to 2006 in Sahel (Gourma, Mali). J. Hydrol., 375(1–2):103–113, 2009. doi: 10.1016/j.jhydrol.2009.01.043.

[11] Minna Räty, Mikko Kuronen, Mari Myllymäki, Annika Kangas, Kai Mäkisara, and Juha Heikkinen. Comparison of the local pivotal method and systematic sampling for national forest inventories. For. Ecosyst., 7:54, 2020. doi: 10.1186/s40663-020-00266-9.

[12] C. Sudhakar Reddy and K. V. Satish. Assessment of tree density, tree cover, species diversity and biomass in semi-arid human dominated landscape using large area inventory and remote sensing data. Anthr. Sci., 2(3–4):197–211, 2023. doi: 10.1007/s44177-024-00066-8.

[13] Thangavelu Mayamanikandan, Suraj Reddy, Rakesh Fararoda, Kiran Chand Thumaty, Mutyala Soma Satya Praveen, Gopalakrishnan Rajashekar, Chandra Shekar Jha, Iswar Chandra Das, and Jaisankar Gummapu. Quantifying the influence of plot-level uncertainty in above ground biomass up scaling using remote sensing data in central Indian dry deciduous forest. Geocarto Int., 37(12):3489–3503, 2022. doi: 10.1080/10106049.2020.1864029.

[14] Tommaso Jucker, Gregory P. Asner, Michele Dalponte, Philip G. Brodrick, Christopher D. Philipson, Nicholas R. Vaughn, Yit Arn Teh, Craig Brelsford, David F. R. P. Burslem, Nicolas J. Deere, Robert M. Ewers, Jakub Kvasnica, Simon L. Lewis, Yadvinder Malhi, Sol Milne, Reuben Nilus, Marion Pfeifer, Oliver L. Phillips, Lan Qie, Nathan Renneboog, Glen Reynolds, Terhi Riutta, Matthew J. Struebig, Martin Svátek, Edgar C. Turner, and David A. Coomes. Estimating aboveground carbon density and its uncertainty in Borneo’s structurally complex tropical forests using airborne laser scanning. Biogeosciences, 15(12): 3811–3830, 2018. doi: 10.5194/bg-15-3811-2018.

[15] Maurice Mugabowindekwe, Martin Brandt, Jérôme Chave, Florian Reiner, David L. Skole, Ankit Kariryaa, Christian Igel, Pierre Hiernaux, Philippe Ciais, Ole Mertz, Xiaoye Tong, Sizhuo Li, Gaspard Rwanyiziri, Thaulin Dushimiyimana, Alain Ndoli, Valens Uwizeyimana, Jens-Peter Barnekow Lillesø, Fabian Gieseke, Compton J. Tucker, Sassan Saatchi, and Rasmus Fensholt. Nation-wide mapping of tree-level aboveground carbon stocks in Rwanda. Nat. Clim. Change, 13(1):91–97, 2023. doi: 10.1038/s41558-022-01544-w.

[16] Ben G. Weinstein, Sergio Marconi, Mélaine Aubry-Kientz, Gregoire Vincent, Henry Senyondo, and Ethan P. White. DeepForest: A Python package for RGB deep learning tree crown delineation. Methods Ecol. Evol., 11(12):1743–1751, 2020. doi: 10.1111/2041-210X.13472.

[17] Ben G. Weinstein, Sarah J. Graves, Sergio Marconi, Aditya Singh, Alina Zare, Dylan Stewart, Stephanie A. Bohlman, and Ethan P. White. A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network. PLoS Comput. Biol., 17(7): e1009180, 2021. doi: 10.1371/journal.pcbi.1009180.

[18] James G. C. Ball, Sebastian H. M. Hickman, Tobias D. Jackson, Xian Jing Koay, James Hirst, William Jay, Matthew Archer, Mélaine Aubry-Kientz, Grégoire Vincent, and David A. Coomes. Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN. Remote Sens. Ecol. Conserv., 9(5):641–655, 2023. doi: 10.1002/rse2.332.

[19] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. IEEE Trans. Pattern Anal. Mach. Intell., 42(2):386–397, 2020. doi: 10.1109/TPAMI.2018.2844175.

[20] Fei Tong and Yun Zhang. Individual tree crown delineation in high resolution aerial RGB imagery using StarDist-based model. Remote Sens. Environ., 319:114618, 2025. doi: 10.1016/j.rse.2025.114618.

[21] Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell detection with star-convex polygons. In Medical Image Computing and Computer-Assisted Intervention (MICCAI 2018), volume 11071 of Lecture Notes in Computer Science, pages 265–273, 2018. doi: 10.1007/978-3-030-00934-2\_30.

[22] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015), volume 9351 of Lecture Notes in Computer Science, pages 234–241, 2015. doi: 10.1007/978-3-319-24574-4\_28.

[23] Simon-Olivier Duguay, Hugo Baudchon, Etienne Laliberté, Helene Muller-Landau, Gonzalo Rivas-Torres, and Arthur Ouaknine. SelvaMask: Segmenting trees in tropical forests and beyond. arXiv, 2026. doi: 10.48550/arXiv.2602.02426. arXiv:2602.02426.

[24] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. DINO: DETR with improved denoising anchor boxes for end-to-end object detection. In International Conference on Learning Representations (ICLR), 2023. arXiv:2203.03605.

[25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3992–4003, 2023. doi: 10.1109/ICCV51070.2023.00371.

[26] Keyan Chen, Chenyang Liu, Hao Chen, Haotian Zhang, Wenyuan Li, Zhengxia Zou, and Zhenwei Shi. RSPrompter: Learning to prompt for remote sensing instance segmentation based on visual foundation model. IEEE Trans. Geosci. Remote Sens., 62:1–17, 2024. doi: 10.1109/TGRS.2024.3356074.

[27] Mélisande Teng, Arthur Ouaknine, Etienne Laliberté, Yoshua Bengio, David Rolnick, and Hugo Larochelle. Bringing SAM to new heights: Leveraging elevation data for tree crown segmentation from drone imagery. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pages 9481–9513, 2025. doi: 10.52202/085713-0290.

[28] Zhenbang Hao, Lili Lin, Christopher J. Post, Elena A. Mikhailova, Minghui Li, Yan Chen, Kunyong Yu, and Jian Liu. Automated tree-crown and height detection in a young forest plantation using mask region-based convolutional neural network (Mask R-CNN). ISPRS J. Photogramm. Remote Sens., 178:112–123, 2021. doi: 10.1016/j.isprsjprs.2021.06.003.

[29] José R. G. Braga, Vinícius Peripato, Ricardo Dalagnol, Matheus P. Ferreira, Yuliya Tarabalka, Luiz E. O. C. Aragão, Haroldo F. de Campos Velho, Elcio H. Shiguemori, and Fabien H. Wagner. Tree crown delineation algorithm based on a convolutional neural network. Remote Sens., 12(8):1288, 2020. doi: 10.3390/rs12081288.

[30] Nuri Erkin Ocer, Gordana Kaplan, Firat Erdem, Dilek Kucuk Matci, and Ugur Avdan. Tree extraction from multi-scale UAV images using Mask R-CNN with FPN. Remote Sens. Lett., 11(9):847–856, 2020. doi: 10.1080/2150704X.2020.1784491.

[31] Andrew J. Chadwick, Tristan R. H. Goodbody, Nicholas C. Coops, Anne Hervieux, Christopher W. Bater, Lee A. Martens, Barry White, and Dominik Röeser. Automatic delineation and height measurement of regenerating conifer crowns under leaf-of conditions using UAV imagery. Remote Sens., 12(24):4104, 2020. doi: 10.3390/rs12244104.

[32] Haotian Zhao, Justin Morgenroth, Grant Pearse, and Jan Schindler. A systematic review of individual tree crown detection and delineation with convolutional neural networks (CNN). Curr. For. Rep., 9(3):149–170, 2023. doi: 10.1007/s40725-023-00184-3.

[33] Josh Veitch-Michaelis, Andrew Cottam, Daniella Schweizer, Eben N. Broadbent, David Dao, Ce Zhang, Angelica Almeyda Zambrano, and Simeon Max. OAM-TCD: A globally diverse dataset of high-resolution tree cover maps. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track, pages 49749–49767, 2024. doi: 10.52202/079017-1574.

[34] M. C. Hansen, P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland, A. Kommareddy, A. Egorov, L. Chini, C. O. Justice, and J. R. G. Townshend. High-resolution global maps of 21st-century forest cover change. Science, 342(6160):850–853, 2013. doi: 10.1126/science.1244693.

[35] John Brandt and Fred Stolle. A global method to identify trees outside of closed-canopy forests with medium-resolution satellite imagery. Int. J. Remote Sens., 42(5):1713–1737, 2021. doi: 10.1080/01431161.2020.1841324.

[36] A. Baccini, S. J. Goetz, W. S. Walker, N. T. Laporte, M. Sun, D. Sulla-Menashe, J. Hackler, P. S. A. Beck, R. Dubayah, M. A. Friedl, S. Samanta, and R. A. Houghton. Estimated carbon dioxide emissions from tropical deforestation improved by carbon-density maps. Nat. Clim. Change, 2(3):182–185, 2012. doi: 10.1038/nclimate1354.

[37] Valerio Avitabile, Martin Herold, Gerard B. M. Heuvelink, Simon L. Lewis, Oliver L. Phillips, Gregory P. Asner, John Armston, Peter S. Ashton, Lindsay Banin, Nicolas Bayol, et al. An integrated pan-tropical biomass map using multiple reference datasets. Glob. Change Biol., 22(4):1406–1420, 2016. doi: 10.1111/gcb.13139.

[38] Alexandre Bouvet, Stéphane Mermoz, Thuy Le Toan, Ludovic Villard, Renaud Mathieu, Laven Naidoo, and Gregory P. Asner. An above-ground biomass map of African savannahs and woodlands at 25 m resolution derived from ALOS PALSAR. Remote Sens. Environ., 206:156–173, 2018. doi: 10.1016/j.rse.2017.12.030.

[39] Anders Ahlström, Michael R. Raupach, Guy Schurgers, Benjamin Smith, Almut Arneth, Martin Jung, Markus Reichstein, Josep G. Canadell, Pierre Friedlingstein, Atul K. Jain,

Etsushi Kato, Benjamin Poulter, Stephen Sitch, Benjamin D. Stocker, Nicolas Viovy, Ying Ping Wang, Andy Wiltshire, Sönke Zaehle, and Ning Zeng. The dominant role of semi-arid ecosystems in the trend and variability of the land $\mathrm { C O _ { 2 } }$ sink. Science, 348(6237): 895–899, 2015. doi: 10.1126/science.aaa1668.

[40] Benjamin Poulter, David Frank, Philippe Ciais, Ranga B. Myneni, Niels Andela, Jian Bi, Gregoire Broquet, Josep G. Canadell, Frederic Chevallier, Yi Y. Liu, Steven W. Running, Stephen Sitch, and Guido R. van der Werf. Contribution of semi-arid ecosystems to interannual variability of the global carbon cycle. Nature, 509(7502):600–603, 2014. doi: 10.1038/nature13376.

[41] FAO. Trees, forests and land use in drylands: The first global assessment. Technical Report FAO Forestry Paper 184, Food and Agriculture Organization of the United Nations, Rome, Italy, 2019. CA7148EN. https://www.fao.org/3/ca7148en/ca7148en.pdf (accessed on 10 September 2026).

[42] Zhi Tian, Chunhua Shen, Xinlong Wang, and Hao Chen. BoxInst: High-performance instance segmentation with box annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5439–5448, 2021. doi: 10. 1109/CVPR46437.2021.00540.

[43] Wentong Li, Wenyu Liu, Jianke Zhu, Miaomiao Cui, Risheng Yu, Xian-Sheng Hua, and Lei Zhang. Box2Mask: Box-supervised instance segmentation via level-set evolution. IEEE Trans. Pattern Anal. Mach. Intell., 46(7):5157–5173, 2024. doi: 10.1109/TPAMI.2024.3363054.

[44] Tianheng Cheng, Xinggang Wang, Shaoyu Chen, Qian Zhang, and Wenyu Liu. BoxTeacher: Exploring high-quality pseudo labels for weakly supervised instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3145–3154, 2023. doi: 10.1109/CVPR52729.2023.00307.

[45] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024. doi: 10.48550/arXiv.2304.07193. arXiv:2304.07193.

[46] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. DINOv3. arXiv, 2025. doi: 10.48550/arXiv.2508.10104. arXiv:2508.10104.

[47] Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. arXiv, 2019. doi: 10.48550/arXiv.1904.07850. arXiv:1904.07850.

[48] George Papandreou, Liang-Chieh Chen, Kevin P. Murphy, and Alan L. Yuille. Weakly- and semi-supervised learning of a deep convolutional network for semantic image segmentation. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1742–1750, 2015. doi: 10.1109/ICCV.2015.203.

[49] Shiyi Lan, Zhiding Yu, Christopher Choy, Subhashree Radhakrishnan, Guilin Liu, Yuke Zhu, Larry S. Davis, and Anima Anandkumar. DiscoBox: Weakly supervised instance segmentation and semantic correspondence from box supervision. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3386–3396, 2021. doi: 10.1109/ICCV48922.2021.00339.

[50] Xiu-Shen Wei, Chen-Lin Zhang, Yao Li, Chen-Wei Xie, Jianxin Wu, Chunhua Shen, and Zhi-Hua Zhou. Deep descriptor transforming for image co-localization. In Proceedings of

the 26th International Joint Conference on Artificial Intelligence (IJCAI), pages 3048–3054, 2017. doi: 10.24963/ijcai.2017/425.

[51] Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T. Freeman. Unsupervised semantic segmentation by distilling feature correspondences. In Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022. doi: 10.48550/arXiv.2203.08414. arXiv:2203.08414.

[52] Yann LeCun. A path towards autonomous machine intelligence, version 0.9.2. OpenReview, 2022. https://openreview.net/forum?id=BZ5a1r-kVsf.

[53] Xiao Xiang Zhu, Devis Tuia, Lichao Mou, Gui-Song Xia, Liangpei Zhang, Feng Xu, and Friedrich Fraundorfer. Deep learning in remote sensing: A comprehensive review and list of resources. IEEE Geosci. Remote Sens. Mag., 5(4):8–36, 2017. doi: 10.1109/MGRS.2017. 2762307.

[54] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021. doi: 10.48550/arXiv.2010.11929. arXiv:2010.11929.

[55] Hancong Fu, Hengqian Zhao, Jinbao Jiang, Yujiao Zhang, Ge Liu, Wanshan Xiao, Shouhang Du, Wei Guo, and Xuanqi Liu. Automatic detection tree crown and height using Mask R-CNN based on unmanned aerial vehicles images for biomass mapping. For. Ecol. Manage., 555:121712, 2024. doi: 10.1016/j.foreco.2024.121712.

[56] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 779–788, 2016. doi: 10.1109/CVPR.2016.91.

[57] Felix Schiefer, Teja Kattenborn, Annett Frick, Julian Frey, Peter Schall, Barbara Koch, and Sebastian Schmidtlein. Mapping forest tree species in high resolution UAV-based RGB-imagery by means of convolutional neural networks. ISPRS J. Photogramm. Remote Sens., 170:205–215, 2020. doi: 10.1016/j.isprsjprs.2020.10.015.

[58] José Augusto Correa Martins, Keiller Nogueira, Lucas Prado Osco, Felipe David Georges Gomes, Danielle Elis Garcia Furuya, Wesley Nunes Gonçalves, Diego André Sant’Ana, Ana Paula Marques Ramos, Veraldo Liesenberg, Jefersson Alex dos Santos, Paulo Tarso Sanches de Oliveira, and José Marcato Junior. Semantic segmentation of tree-canopy in urban environment with pixel-wise deep learning. Remote Sens., 13(16):3054, 2021. doi: 10.3390/ rs13163054.

[59] Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Computer Vision – ECCV 2014, volume 8689 of Lecture Notes in Computer Science, pages 818–833. Springer, 2014. doi: 10.1007/978-3-319-10590-1\_53.

[60] Petru Soviany and Radu Tudor Ionescu. Optimizing the trade-of between single-stage and two-stage deep object detectors using image dificulty prediction. In Proceedings of the 20th International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC), pages 209–214, 2018. doi: 10.1109/SYNASC.2018.00041.

[61] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. YOLOv4: Optimal speed and accuracy of object detection. arXiv, 2020. doi: 10.48550/arXiv.2004.10934. arXiv:2004.10934.

[62] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2999–3007, 2017. doi: 10.1109/ICCV.2017.324.

[63] Mingxing Tan, Ruoming Pang, and Quoc V. Le. EficientDet: Scalable and eficient object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10778–10787, 2020. doi: 10.1109/CVPR42600.2020.01079.

[64] Ross Girshick, Jef Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 580–587, 2014. doi: 10.1109/CVPR.2014.81.

[65] Ross Girshick. Fast R-CNN. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1440–1448, 2015. doi: 10.1109/ICCV.2015.169.

[66] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. IEEE Trans. Pattern Anal. Mach. Intell., 39(6):1137–1149, 2017. doi: 10.1109/TPAMI.2016.2577031.

[67] Ke-Lei Wang, Pin-Hsuan Chou, Young-Ching Chou, Chia-Jen Liu, Cheng-Kuan Lin, and Yu-Chee Tseng. MP-PolarMask: A faster and finer instance segmentation for concave images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3705–3714, 2024. doi: 10.1109/CVPRW63382.2024.00374.

[68] Pengyu Chen, Fangzheng Lyu, Sicheng Wang, and Cuizhen Wang. FG-TreeSeg: Flow-guided tree crown segmentation without instance annotations. IEEE Geosci. Remote Sens. Lett., 23:2503705, 2026. doi: 10.1109/LGRS.2026.3693969.

[69] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/facebookresearch/detectron2, 2019. Accessed on 4 September 2026.

[70] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. SAM 3: Segment anything with concepts. arXiv, 2025. doi: 10.48550/arXiv.2511.16719. arXiv:2511.16719.

[71] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9630–9640, 2021. doi: 10.1109/ICCV48922.2021.00951.

[72] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), pages 8748–8763, 2021. https://proceedings.mlr.press/v139/radford21a.html.

[73] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In Proceedings of the 40th International Conference on Machine Learning (ICML), pages 7480–7512, 2023. https://proceedings. mlr.press/v202/dehghani23a.html.

[74] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv, 2025. doi: 10.48550/arXiv.2504.13181. arXiv:2504.13181.

[75] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT pre-training with online tokenizer. In Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022. doi: 10.48550/arXiv. 2111.07832. arXiv:2111.07832.

[76] Stefan Dersch, Alfred Schöttl, Peter Krzystek, and Marco Heurich. Towards complete tree crown delineation by instance segmentation with Mask R-CNN and DETR using UAV-based multispectral imagery and lidar data. ISPRS Open J. Photogramm. Remote Sens., 8:100037, 2023. doi: 10.1016/j.ophoto.2023.100037.

[77] Erkang Shi, Ziyang Shi, Fulin Su, Lin Li, Ruifeng Liu, Fangying Wan, and Kai Zhou. CrownViM: Context clustering meets vision Mamba for precise tree crown segmentation in aerial RGB imagery. Remote Sens., 18(6):860, 2026. doi: 10.3390/rs18060860.

[78] Yun Zhu, William Locke, Jingyi Yuan, Yunqian Zhang, Qin Ma, and Lu Liang. Leveraging SAM 2 and LiDAR for automated individual tree crown delineation: A comparative evaluation of prompting methods. Inf. Geogr., 1(2):100025, 2025. doi: 10.1016/j.infgeo.2025. 100025.

[79] Carlos A. Silva, Andrew T. Hudak, Lee A. Vierling, E. Louise Loudermilk, Joseph J. O’Brien, J. Kevin Hiers, Steve B. Jack, Carlos Gonzalez-Benecke, Heezin Lee, Michael J. Falkowski, and Anahita Khosravipour. Imputation of individual longleaf pine (pinus palustris mill.) tree attributes from field and LiDAR data. Can. J. Remote Sens., 42(5):554–573, 2016. doi: 10.1080/07038992.2016.1196582.

[80] Hei Law and Jia Deng. CornerNet: Detecting objects as paired keypoints. In Proceedings of the European Conference on Computer Vision (ECCV), volume 11218 of Lecture Notes in Computer Science, pages 765–781, 2018. doi: 10.1007/978-3-030-01264-9\_45.

[81] Arindam Banerjee, Inderjit S. Dhillon, Joydeep Ghosh, and Suvrit Sra. Clustering on the unit hypersphere using von Mises–Fisher distributions. Journal of Machine Learning Research, 6:1345–1382, 2005.

[82] FAO. Global forest resources assessment 2020: Terms and definitions. Technical Report Forest Resources Assessment Working Paper 188, Food and Agriculture Organization of the United Nations, Rome, Italy, 2018. https://www.fao.org/3/I8661EN/i8661en.pdf (accessed on 17 September 2026).

[83] Janik Steier, Mona Goebel, and Dorota Iwaszczuk. Is your training data really ground truth? a quality assessment of manual annotation for individual tree crown delineation. Remote Sens., 16(15):2786, 2024. doi: 10.3390/rs16152786.

## A Supplementary Analyses

## A.1 Mask quality with detection quality controlled for

## A.2 Foreground mixture collapse

The module was designed as a mixture of $K _ { f g }$ foreground prototypes, the idea being that diferent trees could present as distinct prototypes in latent space (e.g. conifers vs palm trees). However, what we found was that with the inclusion of the contrastive term (that is, a contrastive weight $\beta > 0 )$ , the mixture collapses to a single foreground prototype. Mean pairwise cosine rises smoothly with $\beta$ (Table 15) to 0.994 at the deployed $\beta = 0 . 5$ , an efective rank of 1.66 of 16. Pruning tests responsibility share, which the highly similar prototypes survive because each contributes more or less equally. In practice, pruning is activated at intermediate $\beta$ where some prototypes starve.

Table 14: Mean per-crown mask IoU within strata of matched box IoU, OAM-TCD 439-tile holdout. Computed on the 10,020 crowns that every row matched.
<table><tr><td></td><td>Box IoU</td><td>Mask IoU</td><td colspan="5">Mask IoU within box-IoU stratum</td><td></td></tr><tr><td>Method</td><td></td><td></td><td>[0.5, 0.6)</td><td>[0.6, 0.7)</td><td>[0.7, 0.8)</td><td>[0.8, 0.9)</td><td>[0.9, 1.0]</td><td></td></tr><tr><td>Restor Mask R-CNN</td><td>0.781</td><td>0.766</td><td>0.594</td><td>0.682</td><td>0.749</td><td>0.811</td><td>0.863</td><td></td></tr><tr><td>Detectree2 (fine-tuned)</td><td>0.760</td><td>0.760</td><td>0.616</td><td>0.690</td><td>0.757</td><td>0.818</td><td>0.865</td><td></td></tr><tr><td>SelvaBox  $ \mathrm { S A M \ 3 }$ </td><td>0.798</td><td>0.696</td><td>0.524</td><td>0.601</td><td>0.664</td><td>0.728</td><td>0.800</td><td></td></tr><tr><td>Box2Mask Swin-L</td><td>0.772</td><td>0.694</td><td>0.548</td><td>0.617</td><td>0.683</td><td>0.744</td><td>0.797</td><td></td></tr><tr><td>Box2Mask R-50</td><td>0.742</td><td>0.644</td><td>0.500</td><td>0.574</td><td>0.651</td><td>0.723</td><td>0.775</td><td></td></tr><tr><td>LACE (ours)</td><td>0.735</td><td>0.743</td><td>0.650</td><td>0.713</td><td>0.755</td><td>0.789</td><td></td><td>0.819</td></tr></table>

Stratifying on the detection box. Per-stratum counts difer by row (each row’s boxes distribute diferently over the strata) and range from 303 to 3,999. Restor has the highest mask IoU despite not leading in any stratum (Simpson’s paradox). LACE is seed 0. LACE leads the two lowest strata and trails in the two highest. For highly accurate boxes, pixel-level supervision can convert to a better mask than an 8 px lattice, whereas LACE is more robust to imperfect detections.

Table 15: Foreground prototype geometry and accuracy, E-step recentring held on throughout so that $\beta$ is the only variable. Accuracy is mean per-crown mask IoU on the 108 validation tiles at EM seed 0 and detector seed 0, matching the geometry columns. Efective rank is exp of the entropy of the normalised singular-value spectrum of the prototype matrix i.e. it counts directions.
<table><tr><td rowspan="2">Config.</td><td rowspan="2"> $K _ { f g }$  init</td><td rowspan="2"> $K _ { f g }$  final</td><td rowspan="2">Eff. rank</td><td rowspan="2">Pairwise COS</td><td colspan="2">Mean crown IoU</td></tr><tr><td>Oracle</td><td>Pred.</td></tr><tr><td> $\beta = 0$ </td><td>16</td><td>16</td><td>11.43</td><td>0.234</td><td>0.7895</td><td>0.7188</td></tr><tr><td> $\beta = 0 . 1$ </td><td>16</td><td>15</td><td>9.41</td><td>0.334</td><td>0.7900</td><td>0.7194</td></tr><tr><td> $\beta = 0 . 2 5$ </td><td>16</td><td>13</td><td>6.37</td><td>0.525</td><td>0.7901</td><td>0.7199</td></tr><tr><td> $\beta = 0 . 5$ </td><td>16</td><td>16</td><td>1.66</td><td>0.994</td><td>0.7885</td><td>0.7193</td></tr><tr><td> $\beta = 0$ </td><td>2</td><td>2</td><td>2.00</td><td>0.053</td><td>0.7888</td><td>0.7181</td></tr><tr><td> $\beta = 0 . 5$ </td><td>2</td><td>2</td><td>1.30</td><td>0.988</td><td>0.7891</td><td>0.7196</td></tr></table>

Crown IoU is flat throughout: efective rank falls sevenfold across the $\overline { { K _ { f g } = 1 6 } }$ block while mean crown IoU spans 0.0011 on predicted boxes and 0.0016 on oracle boxes, both within the masker-seed band. At $\beta = 0 . 5$ both $K _ { f g }$ settings collapse, and the two reach the same direction $( \cos ( { \bar { C } } ) = 0 . 9 9 4 2 .$ , exceeding the between-seed agreement within $K _ { f g } = 1 6$ itself, 0.9915–0.9922 over three EM seeds)

We deploy $\beta = 0 . 5$ , but Table 10 shows that $\beta = 0$ with recentring reaches the same accuracy, and the reason is that the two subtractions remove the same thing. Repulsion subtracts the background signature from each prototype once, at the M-step (Equation (9)); recentring subtracts the box mean from each cell’s log-ratio at the E-step (Equation (5)), and that mean is dominated by the background the box shares with its surroundings. Either removal on its own separates crown from background; applying both removes little more, which is the negative interaction in Table 10. On validation, $\beta = 0$ with recentring matches the deployed masker on ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ (0.2680 for both) and sits 0.0014 $\mathrm { A P _ { 5 0 } }$ above it, inside the detector-seed band. What $\beta = 0 . 5$ has instead is identifiability (i.e. it generates an explicit “treeness direction”): refit at three EM seeds and at two values of $K _ { f g }$ it recovers the same direction every time (cos 0.9915 to

0.9942), whereas the $\beta = 0$ mixture spreads the same foreground over sixteen prototypes whose individual identities are not determined by the data. With neither subtraction applied, masks occupy much more of the box corners and mask $\mathrm { A P _ { 5 0 } }$ on the 439-tile holdout drops from 0.6203 to 0.5790 (tested with the fit-time scalars $\alpha = 1 , \kappa = 1 0$ and identical detection set).

We theorise that the variance and diversity of OAM-TCD imagery is a key driver of this collapse: it efectively reduces the diferent tree prototypes, although fairly diverse, to a single common direction in latent space.

These results also demonstrate that masker refitting is much less variable than detector training. With three EM seeds (detector fixed), mask $\mathrm { A P _ { 5 0 } }$ varies by ±0.0004, against the ±0.0054 across detector seeds in Table 9.

## B Scoring Protocol and Baseline Configurations

## B.1 Frozen scoring protocol

Every OAM-TCD figure in this paper is produced by one scorer under one configuration, fixed before the baselines were run and unchanged since. The evaluation engine is unmodified pycocotools COCOeval; what follows is its configuration and the per-row prediction geometry it consumes. Because several settings depart from the COCO defaults, the figures here are internally comparable across rows but not directly comparable to numbers published elsewhere for the same checkpoints. Restor’s own published 0.432 against the 0.626 we measure for their released model is the clearest instance, and is a diference of protocol rather than of model.

## B.1.1 Canopy rule

COCO’s crowd rule scores intersection over detection area, so at a threshold of 0.50 it coincides exactly with the “more than half the prediction lies in canopy” rule used elsewhere in this literature. Both were computed and compared rather than assumed equivalent. Canopy-ignore is genuinely free: an ignored detection is removed from the true-positive and false-positive counts alike, so predicting into unlabelled canopy costs nothing. This matters most for SelvaBox, which never saw canopy during training.

Empty tiles Of the 439 tiles, 73 carry no individual crown and 51 carry no annotation of either class. These cost precision but never recall, since the recall denominator is the pooled 25,705 crowns and an unannotated tile contributes nothing to it. Exposure is small and no row is an outlier: 4.0% of LACE’s detections fall on the 73, against 2.1% for Restor, 4.7% for Detectree2 and 6.6% for Box2Mask.

## B.2 Prediction geometry and per-row deviations

No model is forced out of its own distribution; each runs in its native geometry and only the measurement is held fixed. Where we depart from a released configuration, the departure and its reason are given below. Three classes of knob are aligned across rows because the metric requires it: the confidence threshold, the per-image detection budget, and any cap that truncates the ranked list before the scorer sees it. Average precision computed over a censored ranking is not average precision.

Restor’s proposal budget Mask R-CNN detects only what its region proposal network proposes, and the released configuration caps that at 512, half Detectron2’s FPN default, against tiles holding up to 422 crowns. We report Detectron2’s own FPN default of 1000 (Base-RCNN-FPN.yaml), which the released configuration halves; this also brings the Restor row in line with the Detectree2 row, which inherits that default. The released 512 is reported as a sensitivity in Table 18. The cap starves rather than rescores: the maximum detection score is identical across settings and the additional proposals arrive entirely as low-confidence tail.

Table 16: Scorer configuration. Settings marked † depart from the COCO defaults.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Engine</td><td>pycocotools COCOeval, unmodified; iouType segm then bbox</td></tr><tr><td>Unit</td><td>the 439 whole  $2 0 4 8 ^ { 2 }$  holdout tiles; AP pooled over images, not averaged per tile</td></tr><tr><td>Categories</td><td>one, tree (OAM-TCD cat = 1). With a single category the class mean is degenerate, which is why we write  $\mathrm { A P _ { 5 0 } }$  rather than  $\mathrm { m A P 5 0 }$ </td></tr><tr><td>IoU thresholds</td><td>linspace(0.5, 0.95, 10), the COCO convention. An earlier arange(0.5, 0.96, 0.05) placed 0.60, 0.75 and 0.85 one ulp high and rejected IoUs landing exactly there, worth 0.0008  $\mathrm { A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td>Recall thresholds</td><td>101-point (COCO default)</td></tr><tr><td>Area ranges</td><td>all only. Per-instance areas are persisted, so a size-stratified cut can be made later without re-scoring</td></tr><tr><td>maxDets†</td><td>600, against  $\mathrm { C O C O ^ { \circ } s }$  100. Forced: the densest test tile holds 450 ground-truth instances (421 crowns plus 29 canopy groups), so 100 would censor every row. Where it binds is reported below</td></tr><tr><td>Score floor†</td><td>0.05, applied to every row before scoring. COCO scores the full ranked list; this is protocol harmonisation, and it is the knob at which each method&#x27;s own confidence threshold is aligned</td></tr><tr><td>Mask raster†</td><td> $5 1 2 ^ { 2 }$  for every row. No method predicts at 512: each produces masks at its native resolution and they are reduced identically at the last step</td></tr><tr><td>Canopy†</td><td>OAM-TCD&#x27;s canopy class  $( \mathsf { c a t } = 2 )$  enters as iscrowd = 1 ground truth, so a prediction landing in unlabelled canopy is ignored rather than scored</td></tr><tr><td>Ground truth</td><td>all 439 image ids always present, so a method that predicts nothing on a tile is charged zero recall there rather than given a shrunken denominator</td></tr></table>

Table 17: Per-row prediction geometry. Deviations are from each method’s released configuration.
<table><tr><td>Row</td><td>Geometry</td><td>Deviations from released configuration</td></tr><tr><td>LACE (ours)</td><td>whole  $2 0 4 8 ^ { 2 }$ </td><td>none; decoder floor 0.05 and topk 600 are the protocol values</td></tr><tr><td>Restor Mask R-CNN</td><td>whole  $2 0 4 8 ^ { 2 }$  (their own test configuration)</td><td>SCORE_THRESH_TEST  $0 . 2  0 . 0 5 ;$  DETECTIONS PER IMAGE  $5 1 2  6 0 0 ;$  RPN.PRE/POST_NMS_TOPK_TEST 512 → 1000</td></tr><tr><td>Detectree2</td><td> $1 0 2 4 ^ { 2 }$  at 0.5 overlap,  $3 \times 3$  subtiles, stitched</td><td>INPUT.MIN_SIZE_TEST set to its own MIN_SIZE_TRAIN of 1000 rather than Detectron2&#x27;s inherited COCO default of 800, which would downscale our subtiles at test but not at train; stitched at the protocol floor 0.05 rather than its 0.1; clean_crowns merge (IoU &gt; 0.7 or containment  $> 0 . 8 5 )$ </td></tr><tr><td>SelvaBox → SAM 3</td><td> $1 0 2 4 ^ { 2 }$  at 0.5 overlap, native 0.1 m/px</td><td>their OAM-TCD benchmark configuration, not the 1777 px at 0.045 m/px deployment preset, which would resample this imagery  $2 . 2 2 \times \mathrm { u p } ;$  SAM 3 run at the 1777 px tile size its wrapper defaults to; IoU-NMS 0.7 merge; their edge-band cull not applied</td></tr><tr><td>Box2Mask</td><td>as the Detectree2 row</td><td>same crops, grid and stitcher; the architecture&#x27;s cap of 100 instances per  $1 0 2 4 ^ { 2 }$  subtile is left in place</td></tr></table>

Table 18: Restor’s proposal budget, canopy-neutral mask AP on the 439 tiles. Recall is at IoU $0 . 5 ;$ the second row is the operating point reported in Table 3.
<table><tr><td>RPN topk</td><td>Dets/tile</td><td>Recall</td><td> $\bf { A P 5 0 }$ </td><td> $\bf { A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td>512 (released)</td><td>97.9</td><td>0.6442</td><td>0.5706</td><td>0.2575</td></tr><tr><td>1000 (Detectron2 default, reported)</td><td>128.6</td><td>0.7227</td><td>0.6255</td><td>0.2766</td></tr></table>

SelvaBox Edge trim CanopyRS’s released deployment preset sets edge\_band\_buffer\_ percentage = 0.05, which drops any polygon not wholly inside a subtile shrunk by 5%; we do not apply it, since on whole $2 0 4 8 ^ { 2 }$ tiles it would cull the 4,108 of 25,705 ground-truth crowns (16.0%) touching the outer frame, where no neighbouring subtile can recover them, capping recall at 0.84 before the model predicts anything.

## B.3 Asymmetries

Detection budget The cap of 600 detections per tile is our protocol choice. The maximum number of tree crowns in the test set is 422, leaving comfortably enough room for accurate detections without penalty. Only two models actually emit more than 600 detections when the cap is removed: Detectree2 emits up to 1,511 detections per tile at the protocol floor and

SelvaBox up to 1,600. Re-scored at their own budgets they gain 0.004 (0.5969 → 0.6012) and 0.008 $( 0 . 5 6 8 7  0 . 5 7 6 7 )$ mask $\mathrm { A P _ { 5 0 } }$ respectively, which does not change any ranking order in Table 3 and demonstrates that the residuals are a long tail of low-confidence predictions. Restor and Box2Mask never reach the cap, with at most 377 and 584 detections per tile respectively.

The scoring raster favours coarse maskers Reducing every mask to $5 1 2 ^ { 2 }$ discards boundary error that a pixel-level model would otherwise be credited for avoiding. The ground-truth-box experiment of Table 7 measures the size of this: moving from 512 to the native 2048 raster costs our module 0.016 mean IoU against SAM 3’s 0.009, and our advantage at $\mathrm { I o U } \geq 0 . 9$ falls from 2.4× to 1.4×. The 512 raster is thus the one protocol choice that plausibly flatters an 8 px-lattice masker, which is why we take the native one as the headline for that experiment.

Seeds LACE is reported over three detector seeds for the OAM-TCD results, using seed values of 0, 1, 2; the NeonTreeEvaluation results of Table 4 use five detector seeds, 0–4. Where single-seed results are reported, e.g. in some ablations, we use seed= 0. Restor and SelvaBox are single released checkpoints with no seed variation available. Detectree2 and Box2Mask are trained by us but we did not run multiseed experiments.