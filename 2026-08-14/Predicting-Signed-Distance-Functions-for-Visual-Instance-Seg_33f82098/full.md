# Predicting Signed Distance Functions for Visual Instance Segmentation

Emil Brissman<sup>1,2</sup>, Joakim Johnander<sup>1,3</sup>, Michael Felsberg<sup>1</sup>

<sup>1</sup>Computer Vision Laboratory, Dept. of Electrical Engineering, Linköping University

<sup>2</sup>Saab, Sweden

<sup>3</sup>Zenseact, Sweden

{emil.brissman, joakim.johnander, michael.felsberg}@liu.se

Abstract— Visual instance segmentation is a challenging problem and becomes even more difficult if objects of interest varies unconstrained in shape. Some objects are well described by a rectangle, however, this is hardly always the case. Consider for instance long, slender objects such as ropes. Anchor-based approaches classify predefined bounding boxes as either negative or positive and thus provide a limited set of shapes that can be handled. Defining anchor-boxes that fit well to all possible shapes leads to an infeasible number of prior boxes. We explore a different approach and propose to train a neural network to compute distance maps along different directions. The network is trained at each pixel to predict the distance to the closest object contour in a given direction. By pooling the distance maps we obtain an approximation to the signed distance function (SDF). The SDF may then be thresholded in order to obtain a foreground-background segmentation. We compare this segmentation to foreground segmentations obtained from the state-of-the-art instance segmentation method YOLACT. On the COCO dataset, our segmentation yields a higher performance in terms of foreground intersection over union (IoU). However, while the distance maps contain information on the individual instances, it is not straightforward to map them to the full instance segmentation. We still believe that this idea is a promising research direction for instance segmentation, as it better captures the different shapes found in the real world.

## I. INTRODUCTION

Instance segmentation is a core computer vision problem. Given an image, the aim is to detect and segment all objects in an image. A class label is associated to each pixel in the image, similar to semantic segmentation, except that multiple objects of the same class shall be treated as separate entities. This problem has been extensively studied in recent years but remains challenging. The dominant approaches rely on anchorboxes [1], [6]. Anchorboxes is a predefined set of axis-aligned bounding boxes with one or a few different aspect ratios. The idea is that each object shape and object location in the scene should be approximately described by one or more anchorboxes. These approaches then train a neural network to classify each anchorbox as an object or as background. Usually, most objects are fairly well described by one or more of these anchorboxes. However, several common objects such as ropes or road lanes, are not well described by this approach.

In this work, we explore a new direction with the aim to tackle instance segmentation. The idea is to at each pixel predict the distance to the closest object contour, in a given direction. We use Signed Distance Functions (SDFs), giving us a positive distance if we are within an object and a negative distance if we are outside an object. By thresholding, we immediately know whether we are within an object or not. Furthermore, the object contours are found at the zero-crossings of the SDF. In practice, the SDF will deviate slightly; negative if the contour is too far inside and positive if it is too far outside. This is like a spring model and the global equilibrium determines the location of the contour. By representing detected objects with the SDF, avoiding the predefined set of anchorboxes, we are able to capture objects of arbitrary shapes.

![](images/725f62fbe1db175af014a4f53571bc263a0268f98f184819eb8ffac4945fd0b1.jpg)  
Fig. 1. We show two example segmentations predicted by our method. In both cases, we have long, slender objects that are not well described by anchor-boxes. Our network instead predicts, for each pixel, the signed distance to the closest contour along a set of directions. By taking the sign of these functions, we directly obtain a segmentation of the input image.

Our contributions are

• We propose to represent segmentations with signed distance functions

• We show that standard encoder-decoder neural networks are able to learn to predict these functions

• We quantitatively demonstrate that using the signed distance function output, we are able to produce a foreground-background image segmentations of quality comparable to what is obtained with state-of-the-art instance segmentation algorithms

We are able to efficiently get both the foreground and the contours of objects. An efficient way to separate the different instances from the signed distance function is subject to future work.

![](images/0803aa32cf1790ce6e9a4bc5037f50c8ed8c07ccd128eb8826d54168a2eb97b2.jpg)  
Fig. 2. Overview of our approach. The encoder-decoder network outputs distance maps along the directions of θ. The output y is divided into a positive part $y ^ { + } \in \mathbb { R } ^ { 4 \times H \times W }$ that defines the foreground and a corresponding negative part, that defines the background. We apply min-pooling to the parts separately and subtract the positive part with the negative. After subtraction, z ≈ f(I) and the foreground segmentation is directly obtained by taking the sign of $z > 0 .$

## II. RELATED WORK

A lot of research work has been put down into increasing the accuracy of instance segmentation and a lot of work is still ongoing. In three sections we review related methods (Section II-A) and highlight alternative approaches to the instance segmentation problem (Sections II-B to II-C).

## A. Anchor-boxes

Instance segmentation approaches that rely on anchorboxes still dominate, to a large extent, the baseline of stateof-the art methods. Mask R-CNN [6] is anchor-based and splits the task into two subtasks, first object detection then object segmentation. Subsequent computations restrict this approach for real-time speed. Single-stage instance segmentation, like YOLACT [1], improve this aspect but yield lower accuracy. This work directly predicts pixel position representations that are assembled into final results. For segmentation, Bolya et al. [1] suggest to compute object foreground as a logistic regression problem where regression hyper parameters are inferred at each pixel position. In contrast to our approach we do not have any prior assumption on position or shape for the instance segmentation task.

## B. Spatial embeddings

Subsequent methods attempts to perform instance segmentation without anchor-boxes and a popular branch is proposal-free methods. CenterNet [15] suggest to model each object as its centerpoint. Similarly, but for the instance segmentation task, CenterMask [14] propose to adopt local shape information from the representation of object centerpoints to compute the object segmentations. In Neven et al. [12], segmentations are acquired by clustering pixel embeddings that assigns a pixel to an object. A major drawback is that objects could acquire the same centerpoint, which makes it hard to discriminate between the instances. A recent method [2] considers all the pixels as object queries from which self-attention retain those with coherent activations. The top remaining queries are subsequently assigned as background or foreground, and in the latter case, class and bounding box are predicted.

## C. Signed distance function

TensorMask [3] created a foundation for exploring new directions for research within instance segmentation. Chen et al. [3] motivated to building a strong image understanding from structured 4D tensors to represent complex object shapes. This inspired our work to consider signed distance functions. Such functions are simplistic but able to model complex object shapes. In 3D-graphics the signed distance function show more frequent usage, like in previous work [10], [13], but that only considers a single object.

## III. METHOD

We aim to develop a neural network that learns to approximate a Signed Distance Function (SDF) and then use the SDF for visual instance segmentation. The trained network should at each pixel predict the distance to the closest object contour in a set of different directions. These distances provide the information necessary to segment object instances, and the image is easily segmented into foreground and background by thresholding the SDF. An overview of the approach is shown in Fig. 2.

## A. Distance maps

A signed distance function $f : \mathbb { R } ^ { 2 } $ R is a continuous function that, for a given pixel position $\mathbf { p } \in \mathbb { R } ^ { 2 }$ in an image I, outputs the distance to the closest object contour. The function sign encodes whether p is on the object (positive) or outside of the object (negative). Furthermore, all underlying object contours are implicitly represented by the iso-curves $f \left( \cdot \right) = 0$ . The SDF, f, is defined as

$$
f \left( \mathbf { p } \right) = s \cdot \displaystyle \operatorname* { m i n } _ { d \in \mathcal { D } _ { \theta } ^ { \mathbf { p } } } d \mathbf { \Sigma } ,\tag{1}
$$

![](images/1aa4a7c69b4f6958e1340455a9c084870c37e4e03a462997c29818043159b2fa.jpg)  
Fig. 3. We show two examples by the positive part, $y ^ { + } \in \mathbb { R } ^ { 4 \times H \times W }$ , along the directions {0, 90, 180, 270}. The leftmost image depicts a woman playing tennis and the rightmost image depicts a horse. The bottom row show the original image and the three last distance annotations, {90, 180, 270}. Further details about the generation of distance annotations can be found in Section III-C. Although the ground truth neglects the tennis net that occludes the woman who plays tennis, the distance predictions do not.

where

$$
{ \mathcal { D } } _ { \theta } ^ { \mathbf { p } } = \{ \infty \} \cup \{ d \in \mathbb { R } ^ { + } : { \mathbf { p } } + d \left( { \cos ( \theta ) } \right) \in \Omega \} \ .\tag{2}
$$

Here, d represents distances to contours in the direction θ and $s \in \{ - 1 , 1 \}$ is the sign. If p is within an object we let s be positive, and otherwise negative. The set Ω contains a subset of image pixel positions, each of which belongs to object contours. That is, $\mathcal { D } _ { \theta } ^ { \mathbf { p } } .$ in (2), contains the distances to object contours that are intersected by the lines $\mathbf { p } + d ( \cos ( \theta ) ) ^ { \mathrm { ~ * ~ } } \sin ( \theta ) ) ^ { T }$ . The signed distance value is the minimum of this set, see (1). The set will only contain the value of infinity if there is no intersection for each of the directions. In other words, the network encodes the distance to the closest object contour in a discrete number of directions. Since the number of directions is discrete, we consider the SDF in (1) as an approximation. See Fig. 3 for an example on the output maps.

## B. Our approach

Similar to semantic segmentation, we want to predict a vector that encodes distances to the closest object contours for each pixel. That is, given an input image, the output $\mathbf { y } = g ( \mathcal { T } )$ encodes the geometric structures established by the training dataset. For this purpose we use an encoderdecoder network $g : \mathbb { R } ^ { 3 \times H \times W } { \overset { \bullet } { \to } } \mathbb { R } ^ { 2 \times 4 \times H \times W }$ that predicts distances in four directions of $\theta _ { k } = 9 0 k$ degrees, where $k \in$ $\{ 0 , \ldots , 3 \}$ . The output y is divided into two parts. A negative part $y ^ { - } \in \mathbb { R } ^ { 4 \times H \times W }$ that defines the background, and a corresponding positive part, $y ^ { + }$ , that defines the foreground. To compute an approximate SDF $z _ { u v } \approx f ( \mathbf { p } )$ , we use

$$
z _ { u v } = \operatorname* { m i n } _ { k } y _ { k u v } ^ { + } - \operatorname* { m i n } _ { k } y _ { k u v } ^ { - } \mathrm { ~ , ~ }\tag{3}
$$

where $\mathbf { p } = \left( u \quad v \right) ^ { T }$ . In other words, we take the minimum value along the first dimension of both parts and subtract these for all pixels.

We apply a $L _ { 1 }$ -loss to the distance output of $^ { g , }$ such that

$$
\mathcal { L } _ { \mathrm { d i s t } } = \sum _ { ( u , v ) \in \mathcal { T } } \sum _ { \forall k } | \hat { y } _ { k u v } ^ { + } - y _ { k u v } ^ { + } | + | \hat { y } _ { k u v } ^ { - } - y _ { k u v } ^ { - } |\tag{4}
$$

is minimised. In (4) we denote $y ^ { + }$ as the annotated foreground distance and ${ \hat { y } } ^ { + }$ as the predicted foreground distance. Similarly for the background part of (4).

In ${ \mathcal { L } } _ { \mathrm { d i s t } }$ we utilise the value zero for two purposes, 1) to minimise the distance between the predicted distance and the true distance to an object contour and 2) for invalid distances, that is, when $\mathcal { D } _ { \theta } ^ { \bar { \mathbf { p } } }$ contains the value of infinity. From a pixel position that in a given direction does not intersect any object contours is in fact invalid. To use zero in the latter case maintains a simple loss. For the $L _ { \mathrm { 1 } } \mathrm { - } 1 0 8 8$ this means that the expected value for each invalid location with a predicted positive distance should be suppressed to zero.

We initially observed that the predicted distances were noisy. We therefore regularise the network by adding a binary cross-entropy loss $\mathcal { L } _ { \mathrm { a u x } }$ to z. This loss drives the network to predict correct foreground-background segmentations, consistent with the distance output y. We sum the two sub-losses into the total loss ${ \mathcal { L } } _ { \mathrm { t o t } } = { \mathcal { L } } _ { \mathrm { d i s t } } + { \mathcal { L } } _ { \mathrm { a u x } }$ and backpropagate.

## C. Implementations details

We train the model, including the encoder, with batch size 8 on one GPU using the COCO 2017 benchmark [9]. This dataset is divided into 118K training samples and 5K validation samples. The encoder used ImageNet [4] pretrained weights and the decoder was randomly initialised sampling from a uniform distribution. We train with SGD for 54 epochs starting at an initial learning rate of $2 * 1 0 ^ { - 3 }$ and divide with 10 reaching epochs 19, 40, 48 and 51. Weight decay and momentum was $1 0 ^ { - 4 }$ and 0.9 respectively. For data augmentation we used the same as in SSD [11], also used by YOLACT [1]. Training takes 4 days on a single Nvidia V100 GPU.

1) Encoder-decoder: In our model we use a ResNet50 [7] feature extractor as the encoder and a DFN [9] as the decoder. DFN [9] predicts a high-resolution output by fusing the deep feature maps with successively shallower features. Our distance model is motivated by this aspect. Hence, we hypothesise that consistency are kept for distance predictions to the contours of multi-scale objects. In the pooling layer the predicted distance maps are heuristically merged using two min-pooling operations that are concatenated with the correct sign s, for background and foreground. Finally, this is summed along the first dimension, see (3).

2) Distance annotations: COCO [9] comprises 80 classes of objects. These classes were selected to well represent frequently used words to describe visually identifiable objects. The selection is based on the PASCAL VOC dataset [5], a list of the 1200 most common words to describe objects, where several children between 4 and 8 were asked to name every object encountered in indoor and outdoor environments [9]. The result is a set of 80 distinct categories, most fairly highlevel, e.g. car, human, or dog. In each image, each object is annotated with a segmentation mask. Based on these masks, we compute the index map. In some cases, multiple object masks overlap, leading to some pixels being claimed by multiple objects. Our distance annotations rely on each pixel mapping the distance to the single closest object contour. We therefore, for those pixels, select the single object mask corresponding to the smallest mask of all the overlapping object masks. The rationale is that we then give all objects a chance to appear. This can of course turn out wrong in certain cases. As in the upper left image of Fig. 4, where people in the background, which are occluded by people at the front, will be prioritised incorrectly.

Distance maps, used in the loss which is described in Section III-B, are computed for the directions {0, 90, 180, 270}. We first rotate the index map according to the directions. This enables row-wise computation of the distance to the closest object contour. Finally the distance maps are rotated back to their initial state. The procedure is vectorised to increase overall GPU utilisation, which means faster training.

## IV. RESULTS

We report a result on COCO instance segmentation [9]. Our model as, well as YOLACT [1], is trained on the 118K train2017 images. We test out approach on the 5K val2017 images and compare our foreground segmentation with the foreground segmentation obtained from YOLACT [1]. We use the intersection-over-union averaged over all 5K samples in val2017.

Table I compares the IoU of the foreground segmentation by topk detections and changing thresholds on z. Our approach yields a 4% increase over the state-of-the-art method YOLACT [1]. This result demonstrates that managing of complex shapes can be beneficial. We also observe a negative effect when increasing the number of topk detections as well as for larger and larger thresholds on z. However, the latter is expected, since small objects should disappear as result of the distance modelling. The former is contradictory towards this anchor-based method, since it should be expected that performance would go up when enabling more objects to be detected.

Using our method to predict foreground segmentation is faster compared to YOLACT [1]. We use the same feature extractor as in Bolya et al. [1], but without a computationally expensive Feature Pyramid Network [8]. However, this speed is not fair to compare, since more computations in our approach is necessary to achieve full instance segmentation.

TABLE I  
COMPARING FOREGROUND SEGMENTATION BETWEEN YOLACT [1] AND OUR APPROACH ON COCO V A L2017 [9]. BOTH WITH A RESNET50 [7] BACKBONE.
<table><tr><td>Method</td><td>z</td><td>topk</td><td>mIoU</td></tr><tr><td rowspan="2">YOLACT [1]</td><td></td><td>5</td><td>0.66</td></tr><tr><td></td><td>50</td><td>0.65</td></tr><tr><td></td><td>=</td><td>100</td><td>0.64</td></tr><tr><td rowspan="3">Ours</td><td>0.0</td><td></td><td>0.70</td></tr><tr><td>3.0</td><td></td><td>0.67</td></tr><tr><td>5.0</td><td></td><td>0.63</td></tr><tr><td></td><td>10.0</td><td></td><td>0.53</td></tr></table>

In Fig. 3 we show two qualitative examples of our predicted distances. The rightmost show a simple example of a hoarse where prediction and ground truth are comparable. In the leftmost example a woman is playing tennis. The distance predictions are harder to compare to the ground truth since our model finds that the woman is occluded by a tennis net.

Furthermore, in Fig. 4 we show that our method predicts accurate distance maps. These are used to approximate a signed distance function where the positive set of function values describe the foreground segmentation. We also show cases that our method has difficulty with, such as object reflections, object shadows, low-light scenes, and very thin object parts.

## V. CONCLUSION AND FUTURE WORK

We explored a new direction for the instance segmentation problem, where the predictions made by a neural network has a direct geometric interpretation. That is, we train the network to predict distances to object contours that are used to compute a signed distance function. We analyse this approach and show that it outperforms the state-of-the-art instance segmentation YOLACT at foreground-background segmentation. A possible explanation is that some shapes are not well described by anchor-boxes.

Although the predicted distance maps contain information on the individual instances it remains challenging to map this representation to a full instance segmentation. For this representation, an object can consist of different segmentation blobs that are separated due to e.g. occlusion. To label these object related parts with the same label identification is one of the challenges with instance segmentation, even though the blobs fit well to the object scope. We therefore believe this representation to be a promising direction for instance segmentation methods.

## VI. ACKNOWLEDGEMENT

This work was partially supported by the Wallenberg AI, Autonomous Systems, and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

## REFERENCES

[1] D. Bolya, C. Zhou, F. Xiao, and Y. J. Lee. Yolact: Real-time instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

![](images/1537d4454f4bc3c92a8c5908dd04d75186b241154b29d64334495ebcd1fc61b9.jpg)  
Fig. 4. Our method predicts accurate distance maps that are used to separate the foreground from the background (first row). However, it has difficulty with very thin parts that belong to objects, such as the bicycle frame or the legs of the giraffes (first column). Our method also has difficulty with object reflections and object shadows (second row), as well as with poorly lit scenes (third row).

[2] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.

[3] X. Chen, R. Girshick, K. He, and P. Dollar. Tensormask: A foundation for dense object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

[4] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[5] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88(2), July 2010.

[6] K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask r-cnn. In

Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.

[7] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[8] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[9] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and L. Zitnick. Microsoft coco: Common objects in context. In ECCV, September 2014.

[10] S. Liu, Y. Zhang, S. Peng, B. Shi, M. Pollefeys, and Z. Cui. Dist:

Rendering deep implicit signed distance function with differentiable sphere tracing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[11] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. E. Reed, C.-Y. Fu, and A. Berg. Ssd: Single shot multibox detector. In ECCV, 2016.

[12] D. Neven, B. D. Brabandere, M. Proesmans, and L. V. Gool. Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[13] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[14] Y. Wang, Z. Xu, H. Shen, B. Cheng, and L. Yang. Centermask: single shot instance segmentation with point representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9313–9321, 2020.

[15] X. Zhou, D. Wang, and P. Krähenbühl. Objects as points. arXiv preprint arXiv:1904.07850, 2019.