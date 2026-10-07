# Supermarket Product Detection and Recognition: Utilizing Deep Learning with Rectified Imagery

Mayank Sah<sup>a,∗</sup>, Jimson Mathew<sup>a,∗∗</sup>

<sup>a</sup>Department of Computer Science and Engineering, Indian Institute of Technology, Patna, India

## Abstract

Product Identification has sprung up to become one of the most challenging problems in the automation of the retail industry. With the new industry 5.0 standards, automated inventory management, and catalog creation tasks are vitally important. Objec identification models have emerged as a viable answer with their unprecedented identification and localization accuracy. However, the close-knit rack design of supermarkets generates the problem of angle variation in capturing images. The angle-variant densely0 packed images(a single image contains many objects) become overwhelming for these models alone. In this paper, we try to2 supplement object detection models with traditional Hough transform (HT) and homogeneous estimation concepts. We studyt the efect of rectified images using homography estimation and hough transform and their limitations on the problem of grocery<sup>c</sup> identification. We make a case for creating a new dataset to test the efects of such rectification and produce analytical results on diferent scenarios of angle variation and object densities per image. Extensive experiments on diferent object detection models6 suggest that image rectification of angled images improves the detection accuracy of grocery products in images. The results also highlight the limitation of rectification on the angle of image capture and the object density of the image.

Keywords: Grocery Identification, Deep Neural Network, Hough Transform, Homography, Dense Object Detection, Object Detection<sup>.</sup>

## 1. INTRODUCTION

The transition toward Industry 5.0 has increased the need for   
automated inventory management, product identification, and   
catalog generation in retail environments. Although grocery-1   
product identification is straightforward for humans, its au-8   
tomation remains challenging due to the large number of visu-<sup>0</sup>   
ally similar products, dense shelf arrangements, occlusion, and0 variations in imaging conditions. The scale of modern retail   
further emphasizes the need for reliable and scalable computer-<sup>6</sup>   
vision-based solutions. According to Capital One Shopping<sub>:</sub><sup>2</sup>   
Research [1], global retail sales reached USD 29.3 trillion,<sup>v</sup>   
highlighting the importance of eficient automation across both   
physical and online retail. Advances in computer vision have   
established object detection as an efective approach for auto-<sup>a</sup> mated product identification. Early feature based techniques [2] were followed by deep learning-based detectors, including R-CNN [3], SPPNet [4], YOLO [5, 6, 7, 8], and RetinaNet [9]. Subsequent developments introduced anchor-free detectors, such as FCOS [10], and CenterNet [11], along with adaptive anchor-based approaches [12, 13] and transformer-based architectures such as DETR [14]. These methods have demonstrated strong performance across diverse applications, including crack detection [15] and food detection [16]. However, their efectiveness in grocery-product identification can be limited by the characteristics of real-world supermarket scenes. Grocery

shelves typically contain numerous closely packed and visually similar products, often captured from oblique viewpoints. Such viewpoints introduce perspective distortion, nonuniform object scales, and geometric deformation, making accurate localization and recognition particularly challenging in dense grocery scenes.

Our previous work, RectyNet [17], demonstrated the poten tial of viewpoint correction for dense grocery-product detection using trainable rectification modules. However, incorporating additional trainable components increases model complexity. In this work, we instead investigate rectification as an independent preprocessing stage and evaluate its efect on single-stage and transformer-based object detectors. The captured image is first analyzed to estimate its dominant orientation, followed by homography-based perspective correction. The resulting rectified image is then provided to a pretrained object detector for product localization and recognition. To estimate the dominant orientation without introducing an additional learned module, we employ the Hough Transform (HT), a classical technique for detecting geometric structures in images. By mapping imagespace lines into a parameter space, HT identifies the dominant shelf orientation from structural edges present in grocery-shelf images. Hassanein et al. [18] provide a comprehensive review of Hough-based techniques and their applications in computer vision. The estimated orientation is then used to select an appropriate homography for correcting left- and right-oriented views. We initially explored this approach in our previous work [19]; however, that study was limited in terms of experimental scope and analysis. In this work, we provide a more comprehensive evaluation of image rectification, investigating its efect across diferent viewpoints, object densities, and state-of-theart object detection models.

![](images/cae054a3619b037f1023148150d0ad1af2d9fc8a17db75d7eae19b7866c6cdc5.jpg)  
Figure 1: The illustration depicts the scheme of grocery identification. A user captures an image of the grocery shelf. The image is then passed through an object detection module that identifies the input image’s products. The identified products are then passed through a description database that associates product descriptions with each product.

Several datasets have been developed for grocery-product detection, including SKU110K [20] and the Freiburg Groceries Dataset [21]. However, these datasets do not explicitly focus on the efect of systematic camera viewpoint variation on dense product detection. Therefore, we use a subset of our previously curated Grocer-Help dataset [22], containing frontal, leftoriented, and right-oriented grocery-shelf images, to systematically analyze the efects of viewpoint and perspective rectification.

The main contributions of this work are as follows:

1. A geometry-based rectification framework combining the Linear Hough Transform and homography estimation is developed to estimate dominant orientation and correct perspective distortion without adding trainable modules to the detector.

2. The efect of image rectification is systematically evaluated across single-stage and transformer-based object detectors and diferent camera orientations.

3. The influence of camera viewpoint and object density on rectification efectiveness is analyzed, including the limitations of rectification in dense grocery scenes.

## 2. METHODOLOGY

In this section, we provide a detailed description of the methodology followed. Fig. 2 illustrates the process of image rectification in test images. The methodology deals with two main aspects,

1. Angle identification using Hough transform.

## 2. Homography estimation for Left and right aligned images.

A test image is taken as input for the angle identification, and hough lines are detected in the test image. The angle of this representative line decides the orientation of the image, whether the image is left-aligned or right-aligned. The image’s alignment is decided by the camera’s position that captures the image. For homography estimation, instances of known images are pushed in. Traditional feature-based methods are used in our work for ease of implementation. The homographies are estimated for two scenarios: left-aligned images named $H _ { L }$ and right-aligned images named $H _ { R } .$ Finally, the test image is rectified based on the homography, and hough transform to provide a frontal view of an angled image. We have used a passive in tegration of Hough transform-assisted Homography estimation with the object detection models. The passive integration signifies that the homography estimation is done on two sets of representative images from the dataset before going for the object detection scheme. The two representative sets contain a pair of left-aligned images with its frontal image and a pair of right-aligned images with its frontal image. The homography matrices generated from the representative sets are maintained statically and remain the same for all the experiments. In the existing datasets and our dataset, the number of frontal view images heavily outnumbers the left-aligned and right-aligned images, thus providing skewed training for the object detection model. Thus, training the model to identify angled images correctly becomes more problematic. Using the image rectification module, we correct the angle variation in the angled images and warp the image into a frontal view. As a result, the rectified image is similar to the images used for training and thus improves the performance of the existing object detection algorithms. To summarize the methodology, we capture an image of the grocery shelf from our camera; the captured image is taken through an image rectification module, which provides a warped frontal image in case of an angled image or passes the image as it is when there is no significant angle deviation recorded. The warped images then travel through an object detection model and predict objects.

![](images/3aab15e0808962bc1c78af90764c3403fbeb31fec64d0af536e985e3b8789aa9.jpg)  
Figure 2: The figure shows the process of rectifying angled images. As input, two pairs of images (Right angled+front view, left angled+front view) are taken as input, and appropriate homography is estimated. Hough lines are generated on the test image, which is associated the image’s alignment. Input image with appropriate homography generates a rectified image.

For angle estimation in the test images, linear hough transform was used to generate hough lines using Python CV2 4.7.0 version with a canny edge detector and a constant threshold of 500. The angles of the generated hough lines depicted the alignment of the image. Homography was estimated by first resizing the input image to 640x640, followed by a SIFT feature generator to create key points in the aligned image and its frontal counterpart. The key points are then matched using a flann-based matcher algorithm to match the key points in the representative set. The matched key points are calculated using a RANSAC algorithm to calculate the fundamental matrix and inliers. This fundamental matrix is then used to rectify the images. The rectified image is then taken through an object detection model pipeline. Section 4 discusses the experimental results of the methodology in detail. We further divide this section into three subsections where the first subsection describes the dataset accumulation and annotation process. The second subsection describes the process of generating hough transforms from the dataset images for identifying image orientation, and the Third Subsection deals with applying appropriate rectification operations to generate the rectified image for testing.

Table 1: Comparison of representative grocery datasets with Grocer-Help.
<table><tr><td>Dataset</td><td>DS MS</td><td>GD</td><td>BC FG RW</td></tr><tr><td>Grozi-120</td><td>X X</td><td>X V</td><td>√ √</td></tr><tr><td>Freiburg</td><td>X X</td><td>X √√</td><td>√</td></tr><tr><td>RPC</td><td>√ X</td><td>X √ √</td><td>√</td></tr><tr><td>SKU110K</td><td>√ V V</td><td>X </td><td>X √</td></tr><tr><td>Grocer-Help √</td><td>√</td><td>√ V</td><td>√ √</td></tr></table>

DS: Dense Scenes; MS: Multi-store Collection; GD: Geographic Diversity; BC: Brand-based Classes; FG: Fine-grained Products; RW: Real-world Retail Setting.

## 2.1. Dataset

A subset of the Grocer-Help dataset is used to evaluate the proposed viewpoint estimation and perspective rectification methodology. The selected subset contains substantial variation in viewpoint, camera-to-rack distance, object density, and product appearance, making it suitable for evaluating rectification under realistic grocery-store conditions. The images include frontal, left-oriented, right-oriented, and extreme left/right-oriented views. Images were captured at varying distances from the product racks, resulting in both sparse and densely populated scenes, with dense images containing at least 35 objects. The captured racks contain up to five shelf rows, while extreme-view images typically cover two to four rows.

The subset contains 1946 images, with 1310 images used for training and 636 for testing. Training-time augmentation using contrast, saturation, blur, hue, and flipping increases the training set to 7784 images, of which 5240 images constitute the augmented training set. The evaluation set comprises 2544 images, including 944 frontal, 800 right-oriented, and 800 leftoriented images. The frontal images include both close- and long-range views, providing variation in object scale and density. A total of 176 product classes are considered.

![](images/4a8cd1bb38963eee10aa37cf2cf02fa8fb62fcc2410c67aef289860089e452fe.jpg)  
Figure 3: The image illustrates the procedure for generating Hough Lines and Image Instances of Hough Transform on left-aligned and right-aligned images

To ensure implementation consistency, all images are resized to $6 4 0 \times 6 4 0 \times 3$ . No additional preprocessing is applied. The images are manually annotated with bounding boxes, and class assignments follow the product grouping of the Grocer-Help dataset, where visually similar products may share a class while visually distinct products from the same brand are retained as separate classes.

## 2.2. Using Hough Transform to identify image orientation

The Hough Transform (HT) generally transforms an image from image space to parameter space. This conversion to parameter space leads to the problem of curve detection in the image and the peak detection problem in parameter space. In general terms, if we consider a line in image space, then this line converts to a point in the parameter space. Figure 3 showcases the procedure for calculating angles for diferent angle-aligned images. Our implementation uses a Probabilistic Hough Transform instead of a Simple Hough Transform. The input image is first transformed into a greyscale image, and an accumulation matrix is calculated from the greyscale image. The voting mechanism finds the points where a high number of lines intersect. The accumulation matrix then uses a voting mechanism to highlight the key points in the Euclidean space and transform them into a parameter space. The hough lines are then generated from the parameter space. The parameter space also provides the θ and ρ value for the test image, providing the test image’s polarity.

In our method, we use the Hough transform in conjunction with the edge detection module, namely the Canny Edge Detection module, and thus, deal with straight-line parameterization.

The equation of a line is written as:

$$
y = m x + c\tag{1}
$$

Considering this eqn. (1) in the image space, we define the image in terms of $( x , y )$ whereas in transformed space, we define the image in terms of (m,c). Thus transforming the image parameters. A further parameterization of the equation is done as m and c can be infinite; thus, to curtail this down, we get a parameterized equation as:

$$
y C o s ( \theta ) = x S i n ( \theta ) + \rho\tag{2}
$$

where, $0 \leq \theta \leq \pi .$

Based on this concept of HT, we input the grocery images and generate the hough lines in the input image, as shown in the figure.3.

The lines generated as the HT output decide the image’s orientation.

$$
\begin{array} { l } { { \bf { i f ~ } } \theta < 8 5 \mathrm { ~ t h e n } } \\ { { \cal { I } } m a g e i s ^ { , \prime \prime } L E F T ^ { , \prime \prime } O r i e n t e d } \\ { { \bf { e n d ~ i f } } } \\ { { \bf { i f } } \theta > 9 5 \mathrm { ~ t h e n } } \\ { { \cal { I } } m a g e i s ^ { , \prime \prime } R I G H T ^ { , \prime } O r i e n t e d } \\ { { \bf { e n d ~ i f } } } \end{array}
$$

![](images/aa073878e6f335182609da81d18cfcba5c569d1829d6a4a87141b5a22ddb3314.jpg)  
Figure 4: The figure illustrates the efect of Image rectification on Left and Right Aligned Images using respective aligned Homography

As the deviation in line angle is very low on the shelf images, thus the range of θ is kept very low. Fig. 3 showcases the low deviation of angles on the right-aligned and left-aligned images’ output images. The hyperparameters (Accumulator threshold parameter) in the Hough line generator are maintained such that the shelf display angles take priority over the edges detectable from the product boundaries, as the product boundaries may cause erroneous image angle detection.

## 2.3. Homography Estimation and Image Rectification

In our dataset, we consciously chose to take images in rightaligned and left-aligned scenes. These images capture the same data with diferent viewpoints and thus contain a diferent feature set. As we have also emphasized variation in camera angles, there is variation among the images of the same orientation. This variation causes multiple homography matrices for diferent angles. We only use a single-layer homography here, so we take a representative left and right homography matrix at 40 to 60 degrees in each alignment. We derive two homography matrices for the two viewpoints to analyze the best possible object detection scenario.

1. The first homography matrix takes the input images that are right aligned and creates a homography matrix according to its perspective straight view. The right-aligned homography matrix is denoted as $H _ { R } .$

2. The second homography matrix takes the input images that are left aligned and creates the homography matrix with its perspective straight view. The left-aligned homography matrix is denoted as $H _ { L }$

The rectification of the input test image is done by a vector dot product of the input image with the appropriate homography matrix. The output is a perspective projection of an angled image in frontal view.

Each viewpoint in the image space can be seen as a matrix. Suppose we take our aligned image as a matrix A. We are required to transform this matrix A to another matrix B, which is the frontal view of the image. Thus, this operation is possible using the following equation [23]. If the image is left aligned:

$$
B = A . H _ { L }\tag{3}
$$

If the image is right aligned:

$$
B = A . H _ { R }\tag{4}
$$

where, H is the homography matrix [23] such that,

$$
H = \left| { \begin{array} { c c c } { h _ { 1 1 } } & { h _ { 1 2 } } & { h _ { 1 3 } } \\ { h _ { 2 1 } } & { h _ { 2 2 } } & { h _ { 2 3 } } \\ { h _ { 3 1 } } & { h _ { 3 2 } } & { h _ { 3 3 } } \end{array} } \right|\tag{5}
$$

where,

$$
h _ { 1 1 } ^ { 2 } + h _ { 1 2 } ^ { 2 } + h _ { 1 3 } ^ { 2 } + h _ { 2 1 } ^ { 2 } + h _ { 2 2 } ^ { 2 } + h _ { 2 3 } ^ { 2 } + h _ { 3 1 } ^ { 2 } + h _ { 3 2 } ^ { 2 } + h _ { 3 3 } ^ { 2 } = 1\tag{6}
$$

where, $[ h _ { 1 1 } , h _ { 1 2 } , h _ { 2 1 } , h _ { 2 2 } ]$ represent the afine transformation, $[ h _ { 1 3 } , h _ { 2 3 } ]$ represents the translation transformation and $[ h _ { 3 1 } , h _ { 3 2 } ]$ represents the perspective transformation values. $h _ { 3 3 }$ is normalized to 1 so that the homography matrix has a degree of freedom to be 8. The homography estimation module determines the $H _ { L }$ and $H _ { R }$ values. $\because \mathbf { B } ^ { \prime }$ is the input image, and $\ ' \mathbf { A } '$ is the rectified image, transforming the angled image into a rectified frontal image. The values of $[ h _ { 1 1 } , h _ { 1 2 } , h _ { 2 1 } , h _ { 2 2 } , h _ { 1 3 } , h _ { 2 3 } , h _ { 3 1 } , h _ { 3 2 } ]$ are calculated by matching the key-points of the angled image and its equivalent frontal image by a flann based matcher, and then running RANSAC algorithm on the identified key-points.

A wise procedure was maintained to calculate the homography matrices:

1. Input: Aligned image and Normal Image of some similarity

2. Generate Keypoint Features and Descriptors for both the input images

3. Match Descriptor of the aligned image with normal image and generate point mappings

4. Choose Point maps and compute the homography using the RANSAC algorithm

5. Output: Homography matrix H

Fig. 4 showcases the diference in the product visibility in an angled image and its rectified image. The image rectification module increases the visibility of grocery products. However, image rectification may lead to some loss of information from the non-rectified images. The steeper the viewing angle, the more the loss of information in rectification loss. Even though image rectification leads to information loss, it propels the product’s identification accuracy on a shelf. The rectification results also highlight a constraint that the images should not be very densely packed, as the information loss in such images outweighs the increase in identification accuracy.

## 3. EXPERIMENTAL SCENARIO AND ANALYSIS

This section explains the experimental setup referred to for training and testing images. We also provide a qualitative analysis of training and testing our dataset of state-of-the-art methods, namely YOLOv3[5], YOLOv5[6], YOLOv7[7] and YOLOv8[8] and RetinaNet[9]. Table 3 highlights the results achieved on our dataset on state-of-the-art models.

![](images/ec9b0044cd27ab293a765094b8b3f484bd67ca809d392f50bfbe70fbbe5ef1c7.jpg)  
(a) Frontal View of the Grocery Shelf

![](images/c9fe105d1edfceb2d095fb6f4ed25b620da1af08f9fb1d0f90b00a88fdaa887a.jpg)  
(b) Left Aligned View

![](images/5d184565818eae9db2434a683f40c6e8d7d9abf1668ca6eff2d5dc3cae878c2d.jpg)  
(c) Right Aligned View  
Figure 5: The Figure highlights the camera angle’s efect on the image’s view. Image (a) showcases a frontal view of a grocery rack, (b) Depicts the view of the rack when the image is taken from a left-aligned camera, and (c) Depicts the view of the rack when the image is taken from a right aligned camera.

## 3.1. Experimental scenario

We captured images from a single grocery store to test the efects of image rectification on object identification. The selection of grocery stores was made such that there was variation in product brands in every rack to accommodate more diversity in image features. The camera angles were also maintained to capture a more holistic view of the racks. Fig. 5 showcases the camera angle’s efect on the image’s view.

A steep camera angle would lead to an image requiring more rectification. In our works, we have considered only a singlelayer rectification; thus, the steep view of images leads to higher information loss. The angled images are used for testing, whereas the training images are a mix of frontal and angled images. The experiments were carried out on a GeForce GTX1080Ti GPU with 100 epochs. The training set was maintained at 4800 images with over 60,000 objects. For uniformity of experimentation throughout models, hyper-parameters like Input image size, learning rate, weight decay, and momentum were maintained and kept at 640, 0.01, 0.0005, and 0.937, respectively. The objects per image varied from 2 to 55. For testing, 200 right-aligned images, 200 left-aligned images, and 136 front-view images were maintained, and the camera angles for angled images varied from 15 to 80 degrees at each viewing angle. We have only used one-stage object detection models for experimentation as they are much faster than traditional twostage models. We have also used a transformer model as the parameters, and floating point operations almost matched the one-stage detector methods.

## 3.2. Experimental Analysis

The existing datasets mostly contain images in frontal view; thus, the rectification module cannot be tested on these existing datasets. The results in table 2 showcase the accuracy of the models we use in our analysis of the existing datasets. The same models and their variants are applied to our dataset with the same hyperparameters to analyze the impact of image rectification on product identification.

The experiments carried out with our dataset on state-of-theart single-stage detector models are tabulated in Table 3. The analysis is done in four vertical segments. The first segment elucidates the results on only frontal view images with no angle deviation. This situation showcases the best results over all the models used. The second vertical segment contains all images that are right aligned. The right-aligned images also vary in the image capture angle, and the results show that the identification accuracy drops significantly in such a scenario. The third vertical segment is similar, but the images are all left-aligned with angle variation. This variation also showcases a dip in detection accuracy but is identical to that of right-angled images. All the cases discussed are non-rectified, i.e., our image rectification module has not yet worked in any of the scenarios. Our goal is to convert the angled images into frontal view, so we only take the angled images for rectification. The last vertical segment showcases the results after rectifying the angled images.

Table 3 illustrates that all the models perform the same despite the deviation in accuracy. All models work best with nonrectified frontal images and then showcase a dip when dealing with angled photos. The performance improves for all the models as we rectify the angled pictures. The best mAP 0.5 (mean average precision) in the case of frontal view image showcases the accuracy of identifying the correct grocery product is a decent 86.7, but if we take any pictures with specific alignment due to camera positioning, the MAP 0.5 drops from 86.7 to the range of 56 to 58, which is a significant drop in the accuracy of correct identification. The accuracy in rectified images is somewhat higher than the angled images at 60.8. The numbers are high but do not match the frontal view scenario because some loss in object boundaries is also encountered due to warping. Figure 7 highlights the gain every algorithm showcases when the angle information is truncated from the image.

<table><tr><td>Dataset</td><td>Images</td><td>Classes</td><td>YOLOv3</td><td>YOLOv5m</td><td>YOLOv7</td><td>YOLOv8m</td><td>YOLOv9</td><td>RetinaNet</td></tr><tr><td>GroZi-120[24]</td><td>11870</td><td>120</td><td>24.1</td><td>23.9</td><td>25.61</td><td>25.2</td><td>26.3</td><td>24.1</td></tr><tr><td>Webmarket[25]</td><td>3153</td><td>200</td><td>29.4</td><td>29.2</td><td>31.2</td><td>31.8</td><td>32.1</td><td>31.3</td></tr><tr><td>GroZi-3.2K[26]</td><td>8350</td><td>80</td><td>27.1</td><td>26.1</td><td>27.5</td><td>26.7</td><td>29.1</td><td>26.6</td></tr><tr><td>FreiBurg[21]</td><td>4947</td><td>25</td><td>88.1</td><td>73</td><td>79.35</td><td>92.4</td><td>93.5</td><td>78.85</td></tr><tr><td>SKU110K[20]</td><td>11762</td><td>1</td><td>91.7</td><td>89</td><td>88.49</td><td>90.9</td><td>92.2</td><td>90.0</td></tr></table>

Table 2: mAP of SOTA models on existing Grocery Dataset

![](images/43a9fc97bd3dbd10f6140ba9e4e310bb25591c99689b8de34d1508134c4a0f07.jpg)

Figure 6: The Figure highlights the Precision-Recall curve of best-performing algorithms on our dataset
<table><tr><td rowspan="3">Model</td><td rowspan="3">FLOPS</td><td rowspan="3">Parameters</td><td colspan="6">Non Rectified Images</td><td colspan="2">Rectified Images</td></tr><tr><td colspan="2">Frontal View</td><td colspan="2">Right Angled</td><td colspan="2">Left Angled</td><td rowspan="2">mAP 0.5</td><td rowspan="2">mAP 0.9</td></tr><tr><td></td><td>mAP 0.5</td><td>mAP 0.9 mAP 0.5</td><td>mAP 0.9</td><td>mAP 0.5</td><td>mAP 0.9</td></tr><tr><td>YOLO v3</td><td>283.7 G</td><td>103.83 M</td><td>87.6</td><td>73.8</td><td>52.23</td><td>38.67</td><td>51.1</td><td>39.7</td><td>54.79</td><td>40.12</td></tr><tr><td>YOLO v5n</td><td>8.8 G</td><td>2.86 M</td><td>60.1</td><td>46.2</td><td>49.12</td><td>40.1</td><td>49.4</td><td>41.38</td><td>56.2</td><td>45.1</td></tr><tr><td>YOLO v5s</td><td>24.4 G</td><td>9.19 M</td><td>83.9</td><td>65.5</td><td>52.3</td><td>41.2</td><td>53.1</td><td>41.7</td><td>57.2</td><td>45.2</td></tr><tr><td>YOLO v5m</td><td>64.9 G</td><td>25.17 M</td><td>88.3</td><td>71.4</td><td>54.1</td><td>41.6</td><td>55.1</td><td>45.2</td><td>58.6</td><td>43.9</td></tr><tr><td>YOLO v7</td><td>108.1 G</td><td>38.14 M</td><td>74.34</td><td>62.7</td><td>52.3</td><td>41.4</td><td>50.5</td><td>38.7</td><td>56.2</td><td>47.3</td></tr><tr><td>YOLO v7-tiny</td><td>14.7 G</td><td>6.48 M</td><td>62.1</td><td>50.6</td><td>40.2</td><td>28.1</td><td>41.2</td><td>29.1</td><td>45.2</td><td>36.0</td></tr><tr><td>YOLO v8n</td><td>7.4 G</td><td>2.77 M</td><td>42.2</td><td>33.9</td><td>34.3</td><td>28.1</td><td>34.4</td><td>28.1</td><td>36.1</td><td>28.8</td></tr><tr><td>YOLO v8s</td><td>29.0 G</td><td>11.2 M</td><td>82.0</td><td>65.30</td><td>53.1</td><td>42.12</td><td>53.9</td><td>43.5</td><td>56.4</td><td>47.8</td></tr><tr><td>YOLO v8m</td><td>79.6 G</td><td>25.96 M</td><td>85.77</td><td>70.24</td><td>55.1</td><td>42.45</td><td>56.8</td><td>45.1</td><td>59.5</td><td>49.1</td></tr><tr><td>YOLO v9t</td><td>7.1 G</td><td>1.8M</td><td>37.8</td><td>30.5</td><td>29.7</td><td>22.1</td><td>30.1</td><td>22.4</td><td>33.2</td><td>25.1</td></tr><tr><td>YOLO v9s</td><td>23.1 G</td><td>6.3 M</td><td>64.9</td><td>54.7</td><td>52.8</td><td>44.8</td><td>52.1</td><td>46.2</td><td>57.3</td><td>48.6</td></tr><tr><td>YOLO v9m</td><td>61.8 G</td><td>16.85 M</td><td>83.4</td><td>68.7</td><td>57.8</td><td>45.6</td><td>58.1</td><td>46.2</td><td>60.1</td><td>49.7</td></tr><tr><td>YOLO v10n</td><td>9.3 G</td><td>2.88M</td><td>51.2</td><td>41.4</td><td>40.0</td><td>37.8</td><td>40.2</td><td>37.9</td><td>44.5</td><td>39.2</td></tr><tr><td>YOLO v10s</td><td>25.5 G</td><td>8.2 M</td><td>80.3</td><td>65.7</td><td>54.2</td><td>46.6</td><td>55.0</td><td>47.1</td><td>60.1</td><td>50.0</td></tr><tr><td>YOLO v10m</td><td>65.1 G</td><td>16.69 M</td><td>86.7</td><td>71.4</td><td>57.1</td><td>44.9</td><td>57.8</td><td>46.1</td><td>60.8</td><td>50.7</td></tr><tr><td>RTDETR-l</td><td>108.8 G</td><td>33.17 M</td><td>50.4</td><td>39.8</td><td>40.2</td><td>30.3</td><td>41.0</td><td>30.9</td><td>42.5</td><td>31.1</td></tr><tr><td>RTDETR-x</td><td>233.1 G</td><td>67.66 M</td><td>79.2</td><td>65.1</td><td>52.5</td><td>43.1</td><td>53.2</td><td>43.3</td><td>57.2</td><td>47.1</td></tr><tr><td>Retina-Net</td><td>60 G</td><td>34M</td><td>76.9</td><td></td><td>56.3</td><td></td><td>56.9</td><td></td><td>59.8</td><td></td></tr></table>

Table 3: The table summarises the accuracy of diferent Object detection models on our dataset with a frontal view, angled view, and rectified images with mAP 0.5 and mAP 0.9 as the evaluation metric (Here G stands for Giga and M stands for Million)

![](images/dc9f7ad461b31944abc2b6acd925f1116c054fb1e6162c3f4b469205fd56f89a.jpg)  
Figure 7: The Figure highlights the gain perceived in all the existing algorithms when run with rectified images in comparison to the angled images on Grocer-Help dataset

<table><tr><td rowspan="2">Angle</td><td colspan="2"> $\overline { { \mathbf { O } / \mathbf { I } \leq \mathbf { 1 0 } } }$ </td><td colspan="2"> $\overline { { \mathbf { O } / \mathbf { I } } } \geq \mathbf { 1 1 }$ </td></tr><tr><td>mAP 0.5</td><td> $\mathbf { m A P 0 . 5 \left( R . I \right) }$ </td><td>mAP 0.5</td><td>mAP 0.5 (R.I)</td></tr><tr><td>10 - 30</td><td>41.2</td><td>45.5</td><td>14.3</td><td>13.9</td></tr><tr><td>30 - 60</td><td>74.8</td><td>80.2</td><td>34.2</td><td>36.1</td></tr><tr><td>60 - 80</td><td>83.5</td><td>84.0</td><td>56.2</td><td>57.1</td></tr><tr><td>110 - 130</td><td>83.1</td><td>83.2</td><td>57.1</td><td>56.3</td></tr><tr><td>130 - 150</td><td>71.6</td><td>75.7</td><td>36.1</td><td>36.8</td></tr><tr><td>150 - 170</td><td>40.08</td><td>42.2</td><td>11.01</td><td>11.00</td></tr></table>

Table 4: The table highlights the efect of angle variation in mAP 0.5 on densely packed and sparsely packed images. Here O/I is Objects per Image and R.I is Rectified Image

The results published in table 4 showcase the efect of change in camera positioning on MAP when the O/I (Objects per Image) is maintained less than or equal to 10. In such a scenario, the best results are achieved when the camera position is maintained at an angle of 60 to 130 degrees. However, it is also worth noting that the best possible efect of image rectification is perceived when the camera angles are maintained at an angle of 30 to 60 degrees from either side. The same efect is observed when the O/I is maintained at greater than 10. It is noteworthy that the MAP drops considerably when the O/I increases. So, a densely packed image with a very steep inclination of 10 to 30 degrees shows inferior accuracy in grocery detection. The results of table 4 showcase that the best efect of rectification is achieved when the camera angle is maintained at 30 to 60 degrees. Thus, to study the impact of O/I specifically, we test 60 images when the camera angle is left aligned at 30 to 60 degrees and 58 images when the camera angle is right aligned at 30 to 60 degrees.

<table><tr><td rowspan="2">0/I</td><td colspan="2">30 ≤ right_alignment ≤ 60</td><td colspan="2">30 ≤ left_alignment ≤ 60</td></tr><tr><td>mAP 0.5</td><td>mAP 0.5 (R.I)</td><td>mAP 0.5</td><td>mAP 0.5 (R.I)</td></tr><tr><td>less than 5</td><td>89.8</td><td>89.9</td><td>88.4</td><td>89.0</td></tr><tr><td>5 - 10</td><td>85.3</td><td>86.4</td><td>84.1</td><td>84.9</td></tr><tr><td>10 - 20</td><td>79.2</td><td>83.1</td><td>78.8</td><td>83.1</td></tr><tr><td>20 - 40</td><td>49.1</td><td>50.0</td><td>48.2</td><td>50.0</td></tr><tr><td>more than 40</td><td>18.2</td><td>15.1</td><td>19.1</td><td>18.2</td></tr></table>

Table 5: The table highlights the efect of Objects per Image (O/I) with the viewing angle restricted to 30≤ θ ≤ 60. R Istand f orRecti f iedImage

The impact of O/I is clearly understood from the results achieved in table 5, which signifies that it is tough to identify angled images in a densely packed scenario. The O/I, when less than 5, showcases the highest identification accuracy; however, the efect of rectification in such a scenario is minimal as the objects are visible in the original image. Thus, the O/I range of 10-20 shows a considerable change in improving identification accuracy due to rectification. Therefore, image rectification works best when the O/I is kept at 10-20 and θ is maintained such that $3 0 \leq \theta \leq 6 0$ on either side.

## 4. LIMITATIONS AND FUTURE WORK

Although image rectification improves detection perfor mance for angled images, its efectiveness is subject to several

1. Loss of image content in dense scenes: For densely packed images captured at steep angles, rectification may remove objects near the image boundaries or introduce distorted and noisy regions. In extreme cases, the rectified image may contain only approximately 20% of the original image content. This occurs primarily because a fixed homography produces a large geometric transformation for steep viewpoints. A gradual homography transformation could reduce this loss of information.

2. Dependence on visible rack boundaries: The Hough Transform performs reliably when shelf or rack boundaries are clearly visible. When these structural boundaries are absent, the detected lines may instead correspond to product edges, resulting in an inaccurate orientation estimate. This limitation is particularly evident in closely captured images or racks with minimal visible structural boundaries. Incorporating additional structural cues, such as vacant shelf regions, could improve orientation estimation when rack boundaries are unavailable.

3. Limited correction for extreme viewpoints: At very steep viewing angles, a single global homography may not adequately transform the image into a frontal view, resulting in partial or distorted rectification. The current approach uses fixed homography matrices for left- and rightoriented images irrespective of the magnitude of the viewpoint angle. A viewpoint-dependent or gradually varying homography could provide more efective correction across a wider range of angles.

4. Single-rack constraint: The current rectification framework assumes that the input image contains a single dominant rack orientation. Images containing multiple racks with diferent orientations may lead to ambiguous line detection and incorrect orientation estimation.

These limitations indicate several directions for future research. A deep Hough Transform integrated into the detection pipeline could enable joint estimation of image orientation and object detection while reducing dependence on manually visible rack boundaries. For steep viewpoints, progressive or multistage homography correction could minimize information loss and distortion. In addition, lightweight and quantized detection models can be investigated to develop computationally eficient rectification-based detection systems suitable for practical retail environments.

## 5. CONCLUSION

In this paper, we propose a dataset that contains standard authentic life images of grocery stores and their available products. We also specified some pictures with an angled or skewed view of the grocery products on shelves. We then introduced the idea of using Hough Transform to generate Hough lines in the angled images, providing us with the skewness of the objects in the image. The hough lines generated identify all test images as left or right-aligned. The aligned images are then rectified using the respective view homography matrix. As far as we know, image rectification using Hough transform-assisted Homography is the novelty we have proposed in our work, and no work has been carried out in this domain. The experiments carried out in this paper help establish the positive role image rectification plays in camera variation in grocery identification.

The results generated on single-stage object detection algorithms showcase that image rectification improves product identification if the images contain 5 to 10 objects. With the photos getting dense, the accuracy of prediction drops. In the future, we plan to solve the problem of applying hough transform in densely packed images to make this work more diverse, mimicking real-time conditions.

Moreover, this paper deals with the problem of grocery identification in a real-life scenario where images may be from far away, from near, or from a random angle. Also, it may be a slight blur due to motion. This is quite in contrast to the existing datasets considered for experimental setups. We aim to extend the dataset to encapsulate more images per class and more classes available in the Indian market. In our future endeavour, we plan to implement object-wise regional attention to identify the products on the run. We plan to explore the domain of catalogue creation directly from images, which automates the grocery cataloguing process.

## References

[1] “Brick and mortar (vs ecommerce) statistics: Latest 2024 data.” https:// capitaloneshopping.com/research/ brick-and-mortar-retail-statistics/, Apr. 2023. Accessed: 2024-9-27.

[2] C. Huang and J. Huang, “A fast HOG descriptor using lookup table and integral image,” CoRR, vol. abs/1703.06256, 2017.

[3] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” 2014.

[4] K. He, X. Zhang, S. Ren, and J. Sun, Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, p. 346–361. Springer International Publishing, 2014.

[5] J. Redmon and A. Farhadi, “Yolov3: An incremental improvement,” 2018.

[6] G. Jocher, A. Stoken, J. Borovec, NanoCode012, A. Chaurasia, TaoXie, L. Changyu, A. V, Laughing, tkianai, yxNONG, A. Hogan, lorenzomammana, AlexWang1900, J. Hajek, L. Diaconu, Marc, Y. Kwon, oleg, wanghaoyang0106, Y. Defretin, A. Lohia, ml5ah, B. Milanko, B. Fineran, D. Khromov, D. Yiwei, Doug, Durgesh, and F. Ingham, “ultralytics/yolov5: v5.0 - YOLOv5-P6 1280 models, AWS, Supervise.ly and YouTube integrations,” Apr. 2021.

[7] C. Wang, A. Bochkovskiy, and H. M. Liao, “Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,” CoRR, vol. abs/2207.02696, 2022.

[8] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8,” 2023.

[9] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” 2018.

[10] Z. Tian, C. Shen, H. Chen, and T. He, “Fcos: A simple and strong anchor-free object detector,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 4, pp. 1922–1933, 2020.

[11] K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang, and Q. Tian, “Centernet: Keypoint triplets for object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 6569–6578, 2019.

[12] X. Xie, C. Lang, S. Miao, G. Cheng, K. Li, and J. Han, “Mutual-assistance learning for object detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

[13] H. Liu, Q. Tong, L. Miao, and X. Liu, “Efective anchor adaptation and feature enhancement strategies for tiny object detection in aerial images,” IEEE Access, 2024.

[14] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” 2020.

[15] M. Sah, S. Raj, N. Mohariya, R. Gade, J. Mathew, and B. Paul, “Deepinspect: a deep learning tool for automated structural health monitoring with crack analysis,” International Journal of Pervasive Computing and Communications, pp. 1–24, 2025.

[16] M. Sah, S. Suman, and J. Mathew, “Retrieval augmented generation for smart calorie estimation in complex food scenarios,” Journal of Visual Communication and Image Representation, p. 104632, 2025.

[17] M. Sah and J. Mathew, “Rectynet: Geometry-aware feature rectification for robust object detection,” Pattern Recognition Letters, 2026.

[18] A. S. Hassanein, S. Mohammad, M. Sameer, and M. E. Ragab, “A survey on hough transform, theory, techniques and applications,” arXiv preprint arXiv:1502.02160, 2015.

[19] M. Sah and J. Mathew, “Deep learning based supermarket product detection and recognition with rectified images,” Procedia Computer Science, vol. 260, pp. 1026– 1033, 2025.

[20] E. Goldman, R. Herzig, A. Eisenschtat, O. Ratzon, I. Levi, J. Goldberger, and T. Hassner, “Precise detection in densely packed scenes,” 2019.

[21] P. Jund, N. Abdo, A. Eitel, and W. Burgard, “The freiburg groceries dataset,” 2016.

[22] M. Sah, J. Mathew, and P. Dayananda, “A real-world framework for automated product recognition and catalog generation: dataset, model, and analysis,” Scientific Reports, vol. 16, no. 1, p. 14834, 2026.

[23] Y. Luo, X. Wang, Y. Liao, Q. Fu, C. Shu, Y. Wu, and Y. He, “A review of homography estimation: Advances and challenges,” Electronics, vol. 12, no. 24, 2023.

[24] M. Merler, C. Galleguillos, and S. Belongie, “Recognizing groceries in situ using in vitro training data,” in 2007 IEEE Conference on Computer Vision and Pattern Recognition, pp. 1–8, 2007.

[25] Y. Zhang, L. Wang, R. Hartley, and H. Li, “Where’s the weet-bix?,” in Computer Vision–ACCV 2007: 8th Asian Conference on Computer Vision, Tokyo, Japan, November 18-22, 2007, Proceedings, Part I 8, pp. 800–810, Springer, 2007.

[26] M. George and C. Floerkemeier, “Recognizing products: A per-exemplar multi-label image classification approach,” pp. 440–455, 09 2014.