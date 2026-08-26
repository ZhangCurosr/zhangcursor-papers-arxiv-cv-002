# Weakly Supervised Seafloor Segmentation for Seagrass Habitat Mapping in Side-Scan Sonar Imagery

Hayat Rajani, Member, IEEE, Nuno Gracias, Member, IEEE, and Rafael Garcia, Member, IEEE

Abstract—Seagrass meadows are crucial blue-carbon habitats, and mapping their extent is a prerequisite for coastal management and carbon inventory. Optical satellite sensors cover large areas but cannot reach deep or turbid water, whereas side-scan sonar (SSS) images the seabed at high resolution and at any depth. Interpreting SSS, however, still relies on dense manual annotation, which is slow and costly. We address this by adapting a weakly supervised semantic segmentation framework to SSS benthic habitat mapping, so that pixel-level maps are learned from image-level labels alone. The framework couples a ViT-based encoder-decoder with a classification branch, extracts class activation maps, and refines them into pseudo-labels with a dense conditional random field that we tune for the noise and weak boundaries of acoustic imagery. It follows an iterative self-training scheme, together with a sampling strategy to cope with the strong class imbalance of the data. We also study the effect of different loss functions on segmentation quality, finding Lovasz-Softmax loss the most effective. On a held-out transect,´ the refined pseudo-labels reached an mIoU of 89.3% against the ground truth, and the segmentation branch, trained without any pixel-level labels, reached 87.6%. Self-supervised pretraining on unlabelled SSS added a further 3% in mean intersection-overunion. Field trials further demonstrate the generalizability of the trained model. These results show that accurate and labelefficient benthic habitat mapping from side-scan sonar is feasible at the scale needed for coast-wide seagrass monitoring.

Index Terms—side-scan sonar, weakly-supervised semantic segmentation, seafloor segmentation, benthic habitat mapping, seagrass mapping, blue carbon footprint

## I. INTRODUCTION

EAGRASS meadows are among the most valuable coastal S ecosystems. They provide habitat and nursery grounds, stabilise sediments, and store large amounts of organic carbon in their soils over long time scales, a service commonly termed blue carbon [1]. Their global extent has declined by about 29% since records began in 1879, driven by coastal development, declining water quality, and climate change [2]. In the Mediterranean, the endemic seagrass Posidonia oceanica forms extensive meadows that hold some of the largest sedimentary carbon stores in the basin, while Cymodocea nodosa frequently occupies shallower, sandy, or disturbed areas alongside it [1]. Reliable mapping of the spatial extent of these meadows is therefore a prerequisite for coastal management and for any inventory of the carbon they hold [3].

Mapping these meadows over large areas requires imaging the seabed at fine spatial resolution. Side-scan sonar (SSS) produces high-resolution acoustic imagery of the seabed and is widely used in underwater applications such as marine archaeology, structural inspection, and environmental monitoring [4].

It has long been applied to map Mediterranean seagrass meadows, including Posidonia oceanica, at depths where optical sensors cannot operate [3]. Optical satellite and aerial sensors map shallow meadows well but cannot reach the deeper limits of these habitats, so acoustic mapping with SSS complements space-borne and airborne Earth observation rather than competing with it [3]. Unlike optical imagery, however, SSS lacks mature automated or semi-automated annotation tools. General-purpose segmenters such as the Segment Anything Model (SAM) [5] and curated underwater image databases such as FathomNet [6] are built for optical data and do not transfer to acoustic imagery. As a result, interpreting SSS maps still relies on the manual annotation of terrain and structural features, which is labour-intensive and costly [7]. Automating seabed classification would make large-area habitat mapping practical, and running it on board autonomous underwater vehicles (AUVs) would allow classification to proceed during the survey and support more informed, real-time decisions.

This work builds on our earlier work [7], in which we developed an encoder-decoder architecture that combines Vision Transformers (ViTs) and convolutional neural networks (CNNs) for semantic segmentation of the seafloor in SSS imagery. The model improved on the previous state-of-theart by a large margin while meeting real-time computational requirements. Despite its strong performance and its generalisation to unseen data, it was trained under noisy supervision. The seafloor categorisations provided by geophysicists are often produced at a coarse level on SSS mosaics, whereas time-critical online processing must operate directly on raw SSS waterfalls. Transferring annotations from mosaics to waterfalls is not straightforward, especially without access to the internal parameters used for mosaicing, and this introduces discrepancies. The dense ground truth used for supervision is therefore not pixel-accurate. This setting can be treated as a form of weak supervision, in which training proceeds under noisy or incomplete labels.

Weakly supervised semantic segmentation (WSSS) aims to produce a pixel-level segmentation while relying only on coarse or inexpensive labels, for example image-level tags that state which classes appear in an image without indicating where they are [8], [9]. This lowers the annotation effort substantially, at the cost of a harder learning problem, and it is well suited to domains such as SSS where dense labels are scarce. This motivates the main objective of the present work, a segmentation pipeline for side-scan sonar aimed at label-efficient benthic habitat mapping at the scale needed for coast-wide seagrass monitoring.

In this work, we do not propose a new model. Instead, we adapt an existing WSSS framework and analyse the effect of label refinement and loss function design. Specifically, we tune the parameters of a dense conditional random field (CRF) [10] and compare segmentation performance under several loss functions, including cross-entropy, focal loss [11], and Lovasz-´ Softmax loss [12]. Our analysis also shows that commonly used evaluation metrics such as the mean intersection-overunion (mIoU) can obscure meaningful differences between models, particularly under class imbalance. The improvements we report are better understood by interpreting the metric together with the qualitative segmentation behaviour.

We frame the study around three questions:

• Can a weakly supervised approach based on image-level labels produce accurate benthic habitat maps from SSS imagery?

• How does it compare with fully supervised methods in terms of accuracy and efficiency?

• What are its limitations for large-area seagrass and seafloor mapping, and how can they be addressed?

• How well does the approach transfer across survey sites and acoustic conditions, as required for large-area habitat monitoring?

## II. RELATED WORK

Optical satellite and aerial sensors are the most common choice for seagrass mapping because they cover large areas at low cost. Traganos et al. mapped seagrass extent across the entire Mediterranean from Sentinel-2 imagery and linked the result to blue carbon accounting [13]. Gimenez-Romero´ et al. trained a convolutional network to map Posidonia oceanica meadows across the Mediterranean, with a focus on generalisation to new regions [14]. Jeon et al. compared deep networks for the semantic segmentation of seagrass habitat from drone imagery [15]. These studies confirm that optical data support coast-wide monitoring, but they share a physical limit. Light is attenuated in the water column, so optical sensors map shallow meadows well but cannot reliably reach the deeper limit of Posidonia oceanica, which extends to about 40 m. Additionally, the maps created by such optical systems tend to have a coarse spatial resolution, as in the 10 m pan-Mediterranean map of [13], which is too coarse to resolve fine meadow boundaries or to separate sparse from dense cover.

Acoustic sensors fill this gap, because sound propagates well in turbid and deep water. Side-scan sonar (SSS) produces high-resolution backscatter imagery of the seabed at decimetre resolution and reaches depths that optical sensors cannot, thereby resolving structure that a 10 m pixel cannot. It has long been used for seagrass cartography. Pasqualini et al. combined aerial photographs for shallow water with SSS for the deeper range to map Posidonia oceanica off Corsica [3], and later used SSS to support the management of Mediterranean littoral ecosystems [16]. Acoustic mapping is therefore complementary to satellite mapping, extending it into the depth and turbidity range and the fine spatial scale where optical systems stop. More recent acoustic mapping of Posidonia oceanica continues in this direction, for example the multi-platform survey of Piazzolla et al., which reconstructed bottom types and seagrass coverage from echosounder data [17], and Hamouda et al. classified benthic habitats, including seagrass beds, from side-scan sonar and sub-bottom profiling in the Egyptian Mediterranean [18]. These acoustic studies, however, rely on manual interpretation or on classical objectbased and echosounder classification, and they are often paired with optical data or other sensors.

On the other hand, broader work on seafloor semantic segmentation and seabed sediment classification in general has shifted to deep learning. Burguera and Bonin-Font trained a fully convolutional network for on-line multi-class segmentation of the seafloor into terrain types from an autonomous underwater vehicle, and released a labelled dataset [19]. Yang et al. designed a multi-channel convolutional network with large kernels to segment SSS images for autonomous navigation [20]. Zhao et al. proposed a dimension-invariant residual network for seabed sediment classification that preserves feature resolution across channels [21]. Rajani et al. introduced a convolutional Vision Transformer for seafloor semantic segmentation and reported state-of-the-art accuracy under real-time constraints [7].

A separate and larger body of recent work addresses target and instance segmentation in SSS rather than semantic segmentation. These studies focus on discrete man-made objects such as mines, submarine pipelines, shipwrecks and other structures instead of benthic classes. Huang et al. and Tang et al. segmented sonar targets with attention-based SOLO and U-Net variants [22], [23], Wang et al. proposed a recurrent pyramid frequency network for instance segmentation of seabed objects [24], and Sethuraman et al. released a benchmark for shipwreck segmentation [25]. In all of these the aim is to locate sparse targets rather than label continuous seabed cover, distinguishing them from benthic habitat mapping.

A limitation shared across both these groups is that they are trained with dense, pixel-level labels, which are slow and expensive to produce for SSS. This is what limits its use for large-area habitat mapping. The method proposed here removes that obstacle by learning to segment benthic habitats from SSS under image-level supervision only, which makes high-resolution, depth-independent seagrass mapping practical at scale and lets acoustic surveys extend basin-scale optical inventories into the areas they cannot see. To our knowledge, our proposed pipeline is the first to apply weakly supervised semantic segmentation to side-scan sonar, and the first to apply weak supervision to benthic habitat mapping in this modality. The closest work is that of Sledge et al., who used imagelevel labels for circular-scan synthetic-aperture sonar [26], a modality that differs from side-scan sonar in acquisition geometry and resolution and that addresses seafloor and object classes rather than habitats.

## III. METHODOLOGY

We build our approach on the iterative self-improved framework, ISIM [27], which learns a pixel-level segmentation from image-level labels alone. We retain the overall structure of the framework and adapt two compotenents to the SSS setting. The dense CRF that refines the CAM-derived pseudolabels is retuned for the low contrast, speckle noise, and weak object boundaries of acoustic imagery, which make the initial CAMs particularly fragmented. The segmentation loss is then reconsidered to address the strong class imbalance of the dataset and the partial supervision provided by incomplete pseudo-labels. The remainder of this section presents the framework and its backbone, the dense CRF refinement, and the segmentation loss, while the training procedure and the handling of class imbalance are described in Section IV-B.

![](images/bb4603532b41bf741238f91ae2f1c23834a8374005b1b66cbc86832184ad2b86.jpg)  
Fig. 1. The overall weakly-supervised segmentation framework. (a) denotes the encoder, (b) denotes the classification sub-network, and (c) denotes the decoder.

## A. Weakly-supervised Segmentation Framework

Figure 1 depicts the overall framework. It couples an encoder-decoder segmentation network with a classification sub-network. The classification branch, formed by the encoder and the classification sub-network, is trained for multi-label classification on the image-level labels. Class activation maps (CAMs) [28] are extracted from this branch and thresholded to give initial pseudo-labels, which a dense conditional random field (dCRF) [10] then refines to sharpen boundaries and reduce label noise. The refined masks supervise the decoder through a pixel-wise segmentation loss. Training proceeds in an iterative fashion. The classification branch is first trained for a number of epochs, after which CAMs are computed for every training image and converted to pseudo-segmentation labels by the dCRF. These pseudo-labels are then used to train the full encoder-decoder as if they were ground truth. The cycle repeats, so that a stronger segmentation branch yields better CAMs, which in turn yield better pseudo-labels. In this way the framework narrows the supervision gap using only image-level labels and the consistency of the image content. We modify this framework by incorporating our ViTbased encoder-decoder architecture developed for SSS seafloor segmentation [7]. The classification sub-network sits on top of the encoder features, with a global average pooling (GAP) layer followed by a 1 × 1 convolution that produces the class scores used both for multi-label classification and for CAM extraction.

## B. Pseudo-label Refinement

CAMs indicate object presence but tend to activate only the most discriminative parts of an object, so the pseudolabels derived from them are often fragmented and poorly aligned with true boundaries. Following the baseline, we use a dense CRF as a post-processing step to turn these CAMs into more coherent masks. The dCRF operates on the raw CAM-derived masks and uses low-level image cues, such as pixel intensity and spatial position, to correct label boundaries by local consistency. Its energy function combines a unary term from the soft CAM outputs, a pairwise bilateral term that encourages nearby pixels of similar appearance and position to share a label, and a spatial smoothness term that penalises isolated label changes. Rather than alter the model, we tune the main dCRF hyperparameters, namely the spatial and colour standard deviations that set the influence radius of the pairwise terms, the weights of the bilateral and spatial kernels, and the number of inference iterations. The effect of this tuning on mask quality is reported in Section V.

## C. Segmentation Loss

In the original framework, the decoder is supervised on the dCRF-refined pseudo-labels with a standard pixel-wise crossentropy (CE) loss. CE works well under full supervision, but it can underperform under weak supervision, where class imbalance and partial supervision from incomplete pseudo-labels bias training towards the majority class. We therefore study two loss functions that are better suited to this setting. Focal loss [11] adds a modulating factor to the cross-entropy that down-weights well-classified pixels and concentrates training on hard examples. Lovasz-Softmax loss [12] optimises the´ mean intersection-over-union directly through a convex surrogate of the Jaccard index, which aligns the objective with the metric used to evaluate segmentation. We integrate each loss into the segmentation branch on its own, keeping all other training settings fixed, and compare their effect in Section V.

## D. Self-supervised Pre-training

Weak supervision reduces but does not remove the need for labelled data, whereas unlabelled SSS imagery is available in far larger quantities. To exploit this, we pre-train the encoder in a self-supervised manner on the unlabelled SSS archive before the weakly-supervised fine-tuning. We adopt EsViT [29], an efficient self-supervised method for vision transformers that builds on the self-distillation framework of DINO [30].

Self-distillation trains a student and a teacher network of identical architecture but different parameters. Several augmented views of an image are generated, comprising global views at high resolution and smaller local views. The student processes all views while the teacher processes only the global ones, and the student is trained to match the teacher output across views, so that the local views are made consistent with the global context. To prevent collapse, the teacher output is centred with a running mean over the mini-batch and sharpened with a temperature, and the teacher parameters are updated as an exponential moving average of the student parameters rather than by gradient descent.

EsViT retains this view-level objective and adds a regionlevel objective. Besides matching whole views, it matches corresponding local regions between the student and teacher, which encourages the representation to capture fine-grained correspondences between image regions that the view-level task alone tends to lose. This property is desirable for SSS, where discriminative cues are often small and textural. After pre-training, the encoder provides a strong initialisation for both the classification and the segmentation branches of the weakly supervised framework.

## IV. EXPERIMENTAL SETUP

## A. Dataset

We use the BenthiCat dataset, a large-scale opti-acoustic dataset for benthic classification and habitat mapping [31], to train and evaluate our framework. The dataset contains a subset comprising approximately one million SSS tiles for self-supervised representation learning, collected along the coast of Catalonia, Spain, which covers a wide variety of benthic habitats. This large volume of unannotated SSS data makes it well suited to self-supervised pre-training with DINO. For evaluation, we use the annotated subset of about 36,000 tiles, each provided with a pixel-wise segmentation mask over 12 benthic habitat classes for supervised fine-tuning. Figure 2 illustrates some examples of the SSS images along with their multi-class labels for weak supervision.

The raw 12-bit SSS waterfall data underwent several preprocessing steps. First, the data were subjected to logarithmic compression and normalisation to the range [0, 1], which stabilised training and preserved low-intensity detail:

$$
I _ { \mathrm { n o r m a l i z e d } } ^ { \prime } = \frac { \ln ( 1 + I _ { \mathrm { r a w } } ) } { \operatorname* { m a x } ( \ln ( 1 + I _ { \mathrm { r a w } } ) ) } .\tag{1}
$$

Next, slant-range correction was applied under the flat-floor assumption to convert the slant range $r _ { s }$ to ground range $r _ { g } \mathrm { : }$

$$
r _ { g } = \sqrt { r _ { s } ^ { 2 } - h ^ { 2 } } ,\tag{2}
$$

where h is the sensor altitude above the seabed; this step removed the nadir compression and the blind zone.

Finally, the data were tiled into overlapping 384×384 pixel patches with a 192-pixel stride in both the along- and acrosstrack directions.

## B. Training

The training set is strongly imbalanced. Of about 38,000 SSS images, the large majority contain a single class, while roughly 9,250 contain two classes and fewer than 550 contain three or more. This imbalance is harmful here because of two connected reasons. First, a CAM is a class-specific weighting of the encoder features, so a sharp CAM requires the classifier to have learned features that separate that class from the others in space. When almost every image is filled by a single class, the multi-label objective can be minimised from global image statistics alone, without ever having to localise one class against a competing one, so the encoder is driven towards image-level recognition rather than spatial class separation. The CAMs it produces are therefore diffuse, and in the few multi-class images the per-pixel argmax across class CAMs collapses onto the dominant class. Second, the minority classes receive few positive gradients, so their class weight vectors stay under-trained and their CAMs remain weak and low in confidence. A fixed threshold then yields sparse or empty pseudo-masks for these classes, or masks that are absorbed by the majority class. Because the encoder is shared with the segmentation branch, these biased pseudolabels are fed back into training and the bias is amplified across self-training iterations. To counter this, we adopt a random sampling strategy. At each epoch we draw single-class images in proportion to the number of multi-class images in the batch, so that the multi-class examples, which are the only ones that force the network to separate co-occurring classes, contribute a comparable share of the gradient rather than being swamped by the single-class majority.

Alongside the sampling strategy, the dense CRF is tuned as a separate step. Because the dCRF acts only as a postprocessing stage on the CAM outputs, its hyperparameters are searched independently of the network weights. We vary the spatial and intensity standard deviations of the pairwise terms, the weights of the bilateral and spatial kernels, and the number of inference iterations, and retain the setting that produces the cleanest refined masks. This configuration is then held fixed while the pseudo-labels are generated for the self-training loop.

All our models were trained on an NVIDIA A100 Tensor Core GPU for 50 epochs with a batch size of 64. We utilized the AdamW optimizer with a weight decay of $1 e ^ { - 2 }$ and learning rate of $6 e ^ { - 5 }$ , decayed using a polynomial learning rate scheduler with a warm-up of 3 epochs. The CAMs generated by the classification branch are updated every 20 epochs before being refined via dCRF. We also adopted standard data augmentation techniques such as random rotation, random resized crop, random horizontal and vertical flip, random variations in contrast and/or sharpening and Gaussian blur. The models were implemented in PyTorch 1.11.0 and Python 3.8.10. The source code with all hyper-parameter configurations are available at https://github.com/CIRS-Girona/w-s3Tseg.

(a)  
(d)  
(b)  
(e)  
(c)  
(f)
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>Ripples</td><td rowspan=1 colspan=1>Rocks</td><td rowspan=1 colspan=1>Maerl</td><td rowspan=1 colspan=1>Mud</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1><img src="images/9d2147a53fa588e3fc6b46785285d997763996a8c6285a788310f9c9e1eb2674.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>(d)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Fig. 2. Examples of patches generated from SSS waterfalls and the corresponding image-level ground truth.

The self-supervised pre-training, on the other hand, was run for 300 epochs with a learning rate of $5 e ^ { - 4 }$ , decayed using a polynomial learning rate scheduler with a warm-up of 10 epochs. All other hyper-parameter configurations and the source code is available at https://github.com/DeeperSense/ deepersense-seafloorscan/tree/main/self supervised/esvit.

## C. Evaluation

The trained models were initially evaluated on the manually annotated test set. We further carried out field trials using the Girona1000 AUV retrofitted with a Klein 3000 SSS. Small transects were recorded along the port of St. Feliu de Guixols, Spain. This dataset served to demonstrate the generalizability of the model not only to previously unseen data but also to a SSS with diverse configurations and characteristics.

All trained models were then evaluated on an NVIDIA Jetson AGX Orin Developer Kit running Jetpack 5.1.1 with Python 3.8.10 and PyTorch 2.0.0+nv23.5. We report model performance in terms of mean Intersection over Union (mIoU) and inference speed in number of images processed per second (FPS).

## V. RESULTS AND DISCUSSION

Figure 3 presents the evolution of pseudo-masks during the course of training. As the results depict, the model can identify even the pixels corresponding to small scale objects while also quite accurately capturing inter-class boundaries. This clearly showcases the significant potential of this approach in generating pixel-level annotations from image-level labels. Figures 4 and 5 illustrate some of these pixel-level predictions generated by the segmentation branch when iteratively trained using the pseudo masks produced by the classification branch.

The mean IOU between the pixel-level predictions and the pseudo segmentation masks on the test transect was 92.94%. This suggests that even with an approach that iteratively refines the targets of the segmentation branch over the course of training, the model can generalize well. To further analyse the results, pixel-level ground truth for this test set was manually created. The mean IOU of the pseudo segmentation masks with the ground truth was about 89.3% whereas that between the pixel-level predictions from the segmentation branch and the ground truth was about 87.6%. However, self-supervised pretraining resulted in a further increase of about 3% in mIOU on top of these results.

Figure 6, on the other hand, depict results from the field tests. Here, a notable discrepancy in classification can be observed along the middle portion of the transect where the port and starboard side images are merged. This discrepancy results from the blind zone and shadows cast near the first bottom return. As a consequence, the fully-supervised model misclassifies these featureless areas as mud and the darker shadowy areas as rocks. The weakly-supervised model, on the other hand, classifies the entire region as mud since it was not explicitly trained to classify shadows as rocks and it therefore associates such dark featureless regions as mud, considering it the closest match.

## VI. CONCLUSION

We presented a weakly supervised approach to benthic habitat mapping, aimed at reducing the annotation cost that has constrained the use of SSS for large-area seagrass monitoring. By adapting an iterative self-improved framework, the method learns pixel-level segmentation from image-level labels alone, using class activation maps that are refined into pseudo-labels by a dense conditional random field retuned for acoustic imagery. A ViT-based encoder-decoder serves as the backbone, a random sampling strategy mitigates the strong class imbalance of the data, and Lovasz-Softmax loss directly optimizes the´ intersection-over-union resulting in better separation of class boundaries.

On a held-out transect, the segmentation branch reached a mean intersection-over-union of 87.6% against the ground truth without any pixel-level supervision, close to the 89.3% of the refined pseudo-labels, and it recovered small-scale objects and inter-class boundaries well. Self-supervised pretraining on unlabelled SSS gave a further improvement of about 3%.

![](images/8274afbe68f215be86c8a396469248d4cfc82ccdf6041a6dd3a4e43531b433e7.jpg)  
Fig. 3. Evolution of pseudo-masks during the course of training.

![](images/d7c082d25644a189603957001cb84097751ba7a5b1bc43e85bce6b9b417474de.jpg)  
Fig. 4. Visualization of predictions on SSS images from the test set.

These results indicate that side-scan sonar can be segmented into benthic habitats accurately and at low labelling cost, which makes it a practical complement to optical mapping in the deep and turbid areas that satellites cannot reach.

## ACKNOWLEDGMENTS

This work was supported by Spanish Government through the projects ”Intelligent Underwater Robot for Blue Carbon Inventorying (IURBI)” under grant CNS2023-144688 and ”Automated Seabed Analysis through Self-Supervised Deep

![](images/5e88788bdd2af7e58bf71d23be06188cb1280c7a80793b1074567faf88dbfbb6.jpg)  
Fig. 5. (Continued) Visualization of predictions on SSS images from the test set.

Learning Sonar Technology (ASSiST)” under grant PID2023- 149413OB-I00.

## REFERENCES

[1] J. W. Fourqurean, C. M. Duarte, H. Kennedy, N. Marba, M. Holmer,\` M. A. Mateo, E. T. Apostolaki, G. A. Kendrick, D. Krause-Jensen, K. J. McGlathery et al., “Seagrass ecosystems as a globally significant carbon stock,” Nature geoscience, vol. 5, no. 7, pp. 505–509, 2012.

[2] M. Waycott, C. M. Duarte, T. J. Carruthers, R. J. Orth, W. C. Dennison, S. Olyarnik, A. Calladine, J. W. Fourqurean, K. L. Heck Jr, A. R. Hughes et al., “Accelerating loss of seagrasses across the globe threatens coastal

![](images/54c6ce7bc1e51887076617569caadfbbc5c4778990b811599c39e100e2d7aaa5.jpg)  
Fig. 6. Comparison of results between fully- and weakly-supervised models during field trials.

ecosystems,” Proceedings of the national academy of sciences, vol. 106, no. 30, pp. 12 377–12 381, 2009.

[3] V. Pasqualini, C. Pergent-Martini, P. Clabaut, and G. Pergent, “Mapping ofposidonia oceanicausing aerial photographs and side scan sonar: Application off the island of corsica (france),” Estuarine, Coastal and Shelf Science, vol. 47, no. 3, pp. 359–367, 1998.

[4] P. Blondel, The Handbook of Sidescan Sonar. Berlin, Heidelberg: Springer, 2009.

[5] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in 2023 IEEE/CVF international conference on computer vision (ICCV). IEEE, 2023, pp. 3992–4003.

[6] K. Katija, E. Orenstein, B. Schlining, L. Lundsten, K. Barnard, G. Sainz, O. Boulais, M. Cromwell, E. Butler, B. Woodward et al., “Fathomnet: A global image database for enabling artificial intelligence in the ocean: K. katija et al.” Scientific reports, vol. 12, no. 1, p. 15914, 2022.

[7] H. Rajani, N. Gracias, and R. Garcia, “A convolutional vision transformer for semantic segmentation of side-scan sonar data,” Ocean Engineering, vol. 286, p. 115647, 2023.

[8] Z. Chen and Q. Sun, “Weakly-supervised semantic segmentation with image-level labels: from traditional models to foundation models,” ACM Computing Surveys, vol. 57, no. 5, pp. 1–29, 2025.

[9] J. Ahn and S. Kwak, “Learning pixel-level semantic affinity with imagelevel supervision for weakly supervised semantic segmentation,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018, pp. 4981–4990.

[10] P. Krahenb¨ uhl and V. Koltun, “Efficient inference in fully connected¨ crfs with gaussian edge potentials,” Advances in neural information processing systems, vol. 24, 2011.

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[12] M. Berman, A. Rannen Triki, and M. B. Blaschko, “The lovasz-softmax´ loss: A tractable surrogate for the optimization of the intersectionover-union measure in neural networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 4413–4421.

[13] D. Traganos, C. B. Lee, A. Blume, D. Poursanidis, H. Ci<sup>ˇ</sup> zmek, J. Deter,ˇ V. Maciˇ c, M. Montefalcone, G. Pergent, C. Pergent-Martini´ et al., “Spatially explicit seagrass extent mapping across the entire mediterranean,” Frontiers in Marine Science, vol. 9, p. 871799, 2022.

[14] A. Gim<sup>\`</sup> enez-Romero, D. Ferchichi, P. Moreno-Spiegelberg, T. Sintes,´ and M. A. Mat´ıas, “A generalizable deep learning framework for largescale mapping of seagrass habitats,” Ecological Indicators, vol. 180, p. 114349, 2025.

[15] E.-i. Jeon, S. Kim, S. Park, J. Kwak, and I. Choi, “Semantic segmentation of seagrass habitat from drone imagery based on deep learning: A comparative study,” Ecological Informatics, vol. 66, p. 101430, 2021.

[16] V. Pasqualini, P. Clabaut, G. Pergent, L. Benyoussef, and C. Pergent-Martini, “Contribution of side scan sonar to the management of mediterranean littoral ecosystems,” International Journal of Remote Sensing, vol. 21, no. 2, pp. 367–378, 2000.

[17] D. Piazzolla, S. Scanu, F. P. Mancuso, M. Bosch-Belmar, S. Bonamano, A. Madonia, E. Scagnoli, M. F. Tantillo, M. Russi, A. Savini et al., “An integrated approach for the benthic habitat mapping based on innovative surveying technologies and ecosystem functioning measurements,” Scientific Reports, vol. 14, no. 1, p. 5888, 2024.

[18] A. Z. Hamouda, A. Fekry, and S. El-Gharabawy, “Acoustic-based classification of marine geophysical data for benthic habitat mapping in the littoral zone of qaitbay citadel of alexandria,” Egyptian Journal of Aquatic Research, vol. 50, no. 1, pp. 8–16, 2024.

[19] A. Burguera and F. Bonin-Font, “On-line multi-class segmentation of side-scan sonar imagery using an autonomous underwater vehicle,” Journal of Marine Science and Engineering, vol. 8, no. 8, p. 557, 2020.

[20] D. Yang, C. Cheng, C. Wang, G. Pan, and F. Zhang, “Side-scan sonar image segmentation based on multi-channel cnn for auv navigation,” Frontiers in Neurorobotics, vol. 16, p. 928206, 2022.

[21] Y. Zhao, K. Zhu, T. Zhao, L. Zheng, and X. Deng, “Seabed sediments classification based on side-scan sonar images using dimension-invariant residual network,” Applied Ocean Research, vol. 130, p. 103429, 2023.

[22] H. Huang, Z. Zuo, B. Sun, P. Wu, and J. Zhang, “Dsa-solo: double split attention solo for side-scan sonar target segmentation,” Applied Sciences, vol. 12, no. 18, p. 9365, 2022.

[23] Y. Tang, L. Wang, H. Li, and S. Bian, “Side-scan sonar underwater target segmentation using the bhp-unet,” EURASIP Journal on Advances in Signal Processing, vol. 2023, no. 1, p. 76, 2023.

[24] Z. Wang, S. Zhang, C. Zhang, and B. Wang, “Rpfnet: Recurrent pyramid frequency feature fusion network for instance segmentation in sidescan sonar images,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2023.

[25] A. V. Sethuraman, A. Sheppard, O. Bagoren, C. Pinnow, J. Anderson, T. C. Havens, and K. A. Skinner, “Machine learning for shipwreck segmentation from side scan sonar imagery: Dataset and benchmark,” The International Journal of Robotics Research, vol. 44, no. 3, pp. 341– 354, 2025.

[26] I. J. Sledge, D. M. Byrne, J. L. King, S. H. Ostertag, D. L. Woods, J. L. Prater, J. L. Kennedy, T. M. Marston, and J. C. Principe, “Weaklysupervised semantic segmentation of circular-scan, synthetic-aperturesonar imagery,” arXiv preprint arXiv:2401.11313, 2024.

[27] C. Bircanoglu and N. Arica, “Isim: Iterative self-improved model for

weakly supervised segmentation,” arXiv preprint arXiv:2211.12455, 2022.

[28] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning deep features for discriminative localization,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition, 2016, pp. 2921– 2929.

[29] C. Li, J. Yang, P. Zhang, M. Gao, B. Xiao, X. Dai, L. Yuan, and J. Gao, “Efficient self-supervised vision transformers for representation learning,” arXiv preprint arXiv:2106.09785, 2021.

[30] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and´ A. Joulin, “Emerging properties in self-supervised vision transformers,” in 2021 IEEE/CVF international conference on computer vision (ICCV). IEEE, 2021, pp. 9630–9640.

[31] H. Rajani, V. Franchi, B. Martinez-Clavel Valles, R. Ramos, R. Garcia, and N. Gracias, “Benthicat: An opti-acoustic dataset for advancing benthic classification and habitat mapping,” Earth System Science Data Discussions, vol. 2026, pp. 1–32, 2026.