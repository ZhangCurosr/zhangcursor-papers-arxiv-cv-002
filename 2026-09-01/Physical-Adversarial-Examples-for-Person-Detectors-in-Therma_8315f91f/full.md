# Physical Adversarial Examples for Person Detectors in Thermal Images Based on 3D Modeling

Xiaopei Zhu, Siyuan Huang, Zhanhao Hu, Jianmin Li, Jun Zhu, Fellow, IEEE, and Xiaolin Hu, Senior Member, IEEE

Abstract—Thermal Infrared detection is widely used in autonomous driving, medical AI, etc., but its security has only attracted attention recently. We propose infrared adversarial clothing designed to evade thermal person detectors in real-world scenarios. The design of the adversarial clothing is based on 3D modeling, which makes it easier to simulate multiangle scenes near the real world compared to 2D modeling. We optimized the black patch layout pattern of 3D clothing based on the adversarial example technique and made physical adversarial clothing using the aerogel. The idea is to paste a set of square aerogel patches, which display black squares in thermal images, in the inner side of clothing at specific locations with specific orientations. To enhance realism, we propose a method to build infrared 3D models with real infrared photos and develop texture maps for 3D models to simulate varied infrared characteristics over time and location. In physical attacks, we achieved an attack success rate of 80.11% indoors and 76.85% outdoors against YOLOv9. In contrast, randomly placed patches yielded much lower success rates (26.53% indoors and 23.03% outdoors). The adversarial clothing also showed good transferability to unknown detectors with an ensemble attack method, demonstrating the effectiveness of our approach.

Index Terms—Physical adversarial example, Object detection, 3D modeling, Thermal imaging.

✦

## 1 INTRODUCTION

D <sup>EEP</sup> <sup>learning</sup> <sup>models</sup> <sup>can</sup> <sup>be</sup> <sup>fooled</sup> <sup>by</sup> <sup>carefully</sup> <sup>de-</sup>signed inputs, which are called adversarial examples. signed inputs, which are called adversarial examples. Adversarial examples include digital and physical varieties. The study of physical adversarial examples has great significance for the security of AI systems, because deep learning-based technologies such as face recognition and autonomous driving are now widely used in the real world.

Adversarial examples not only exist in the visible light domain [1], [2], [3], [4], [5] but also in the thermal infrared (infrared for short) domain [6], [7], [8]. In recent years, infrared imaging has been widely used in autonomous driving, medical diagnose, and body temperature measurement. Infrared imaging has unique advantages [6]. First, infrared images contain temperature information. Second, infrared imaging can work even if there is occlusion. Third, infrared imaging does not depend on external light sources and can work at night. Research on physical infrared adversarial examples has important significance for the safety of infrared imaging systems but attracted little attention.

Many studies [1], [2], [3], [4], [6], [7] about physical adversarial examples are based on the 2D patch. However, these studies have limitations. The 2D patch does not cover the entire 3D surface of the object. For example, the adversarial T-shirt [2] can only perform attacks from the front of the human body and does not work from other angles (e.g.,

![](images/42ec8359eea633e05d5d91dfbd94a10e2ae5aaf8da890edcb65efd7a561da4d4.jpg)  
Fig. 1. One example of infrared physical attacks. The adversarial clothing looks like ordinary clothing in the visible light view. The facial areas are blurred for privacy reasons. The person wearing adversarial clothing is not detected by the detectors at multiple angles in infrared imaging, while the other person wearing ordinary clothes is detected (indicated by the bounding boxes).

the side of the body).

Adversarial examples based on 3D modeling help to solve the above problem. With 3D modeling, we can design adversarial patterns from different angles. Compared with the 2D image, the 3D model is closer to real-world objects, which can simulate the physical adversarial attacks more realistically.

In the visible light domain, there have been some physical adversarial studies [9], [10], [23], [48] based on 3D modeling. However, besides the difference in the modality (infrared and visible light) we are concerned with, there are other important differences between these studies and the present study. Specifically, the study in [23] used nondifferentiable rendering and trained a neural network to simulate it. In contrast, we propose to use a differentiable render to make the optimization more accurate. Different from the previous studies [10], [48] where detectors were attacked at a narrow range (e.g., -50 to 50 degrees), we attack the detectors through the full range of 0-360 degrees. Different from the classifier attack [9], we focus on the object detection attack, which is more challenging.

To the best of our knowledge, there is currently a lack of 3D model-based physical adversarial research in the infrared domain. Infrared 3D models are constructed based on the shape and temperature of the object. The pixel values of grayscale images reflect the temperature of the surface of the object. Unlike the RGB 3D model, the pattern of the infrared 3D model is not easily “printed” in the physical world.

Recently, we designed a QR code-like clothing to attack person detectors from multiple angles [8], but that method is based on 2D pattern design and does not use 3D modeling of either a person or clothing. That approach has several limitations. Due to the lack of the 3D model, there is a big gap between the 2D digital simulation and 3D physical implementation. Besides, the thermal insulation materials are pasted outside the clothing, which looks strange. In addition, the manufacturing process of the clothing is quite complex and requires a lot of human power (7 persons over 15 days) and material costs.

In this study, we propose 3D model-based physical infrared adversarial clothing. To simulate the infrared characteristics more realistically, we propose a method to build infrared 3D models with real infrared photos. Using above method, we build 3D digital models with real infrared characteristics for both the human body and clothing to minimize the gap between the digital simulation and physical implementation. We also build different texture maps for 3D clothing model to simulate changeable infrared clothing characteristics in different times and places. Then, we manufacture the adversarial clothes in the physical world. Adversarial clothing looks similar to ordinary clothing in visible light imaging but can be invisible to object detectors in infrared imaging. In addition, the proposed method requires much fewer materials and makes it easier to manufacture clothes than the QR code approach [8]. Figure 1 shows an example. To the best of our knowledge, this study is the first to investigate infrared physical attacks based on 3D modeling.

## 2 RELATED WORKS

## 2.1 2D adversarial attacks

2D adversarial attacks can be divided into digital attacks and physical attacks. Physical adversarial attacks have attracted significant attention in recent years due to their high practical value in real-world applications.

## 2.1.1 2D digital attacks

Adversarial attacks originate from the vulnerability of deep neural networks (DNNs) revealed by Szegedy et al. [12]. Early adversarial attacks focused on the digital domain [11], [13], [14], [16], [17]. The input with carefully designed perturbations causes DNNs to make mistakes in the digital world. 2D digital attacks are usually classified into three types: gradient-based methods [13], [14], [15], [29], [70], [71], optimization-based methods [11], [12], [21], [72], [73], and GAN-based methods [16], [17], [24]. Digital attacks assume that attackers can modify the input of the model, which is difficult to achieve in the real world. However, digital attacks are the basis for physical attacks.

## 2.1.2 2D physical attacks

Physical attacks are designed to attack DNNs in the real world. Most physical attacks focus on the visible light domain, and infrared physical attacks have only attracted attention in recent years.

In the visible-light domain, researchers typically design pixel-level RGB patterns and then print these patterns on paper, stickers, or clothing for physical display. For adversarial papers, Thys et al. [1] printed the adversarial example on a piece of paper, making the detector unable to identify the person holding the paper. Du et al. [42] attached adversarial paper on car roofs and parking lines to attack an aerial imagery object detector.

For adversarial stickers, Eykholt et al. [4] develop stickers with robust physical perturbations (RP2) to mask the stop sign at specific locations, making DNNs misclassify the stop sign as a speed limit sign. Tao et al. [41] designed stickers with perturbations that are nearly imperceptible to the human eye, causing DNNs to misclassify various signs. Wei et al. [74] simultaneously optimized the positions and perturbations for 2D adversarial patch attacks and attacked face and traffic sign recognition models in the black-box setting. Wei et al. [75] proposed adversarial stickers to achieve black-box attacks in the physical world, which is a physically feasible and stealthy attack method.

For adversarial clothes, Xu et al. [2] proposed an adversarial T-shirt and used the thin plate spline (TPS) method to simulate clothing deformation. Wu et al. [20] attacked YOLOv2 using wearable clothes in the physical world, and evaluated the performance of such attacks with different metrics. Hu et al. [3] used GAN to generate adversarial examples, which made them more natural when printed on clothes. We [5] proposed the adversarial texture, which was printed on clothes to fool the visible light person detector at multiple angles.

In the infrared domain, researchers typically perform mathematical modeling of the infrared characteristics of basic components, then design algorithms to optimize the key parameters of these components, and finally conduct physical implementation based on the optimized parameters. Physical implementation methods can be divided into two categories: one based on heating or cooling elements, and the other based on insulating materials. For heating or cooling device-based methods, we [6] proposed an adversarial board based on small bulbs to fool the infrared pedestrian detector YOLOv3, and then we expand this work to attack the infrared and visible detectors simultaneously. After that, we [79] proposed the adversarial clothes based on carbon heaters to deceive infrared pedestrian detectors. Wei et al. [76] applied warming and cooling paste to hide from the infrared person detectors. Hu et al. [80] proposed adversarial infrared blocks to attack infrared detectors in a black-box setting. After that, they [81] proposed the adversarial infrared curves based on Bezier curve optimization to further enhance the efficiency in physical deployment. Tiliwalidi et al. [82] proposed the AdvGrid to hide from the infrared pedestrian detectors using readily available materials and minimal setup.

For thermal insulation material-based methods, we [8] proposed the adversarial “QR codes” texture and made the adversarial clothing to fool the infrared pedestrian detectors at multiple angles. Kim et al. [7] proposed the adversarial board to attack infrared detector using materials with different emissivity. After that, they [83] propose the adversarial clothes based on low-E films to conduct cross-modal attacks. Wei et al. [77] proposed infrared adversarial patches with learnable shapes and locations. Wei et al. [78] proposed unified adversarial patches for cross-modal attacks in the physical world.

## 2.2 3D adversarial attacks

2D adversarial attacks usually exhibit effectiveness only within a narrow angular range (e.g., approximately -50<sup>◦</sup> to 50<sup>◦</sup>), whereas 3D adversarial attacks can cover a full 0–360<sup>◦</sup> viewpoint range. Currently, 3D adversarial attacks mainly focus on the visible and LiDAR domains, while 3D infrared attacks have not been fully explored.

## 2.2.1 3D digital attacks

In the visible-light field, Liu et al. [44] optimized the perturbations by changing conditions such as lighting so that the 3D shoe model was incorrectly classified when only the lighting changed. Xiao et al. [45] optimized the rendering to generate the 3D adversarial car model in the digital world. Wang et al. [48] proposed the 3D adversarial logo to fool the person detector in the digital world. Wang et al. [84] proposed a full-coverage vehicle camouflage for multi-view physical adversarial attack. Suryanto et al. [85] proposed a differentiable transformation network (DTN) to simulate the adversarial car attack realistically in the digital world.

In the LiDAR field, Xiang et al. [43] generated 3D point cloud perturbations by shifting, reshaping, or repositioning the position of the point cloud so that the DNNs misidentified the point cloud of the bottle as various objects. Tu et al. [26] optimized a 3D mesh of a specific shape and then placed the mesh on the roof of a 3D car model, which fooled the 3D LiDAR detector.

## 2.2.2 3D physical attacks

In the visible domain, Athalye et al. [9] 3D-printed an adversarial turtle that was misclassified by a DNN as a rifle. Huang et al. [10] proposed Universal Physical Camouflage (UPC), where they simulated an adversarial attack using a 3D model and then attacked a detector in the physical world. Wang et al. [86] proposed a dual attention suppression (DAS) method to optimize adversarial stickers on 3D vehicle surfaces, and Li et al. [87] introduced a flexible physical camouflage generation method based on a differential optimization approach.

In the LiDAR field, Cao et al. [88] proposed an adversarial sensor attack on LiDAR-based perception in autonomous driving. Subsequently, Jin et al. [89] developed a physical laser attack against LiDAR-based 3D object detection in autonomous driving. Additionally, Cao et al. [90] generated interfering obstacles for LiDAR by 3D-printing adversarial geometric shapes and optimizing their forms to achieve adversarial effects.

## 2.3 Infrared stealth materials

Traditional infrared stealth materials primarily include iron composites [46], zinc composites [47], aluminum composites [30], etc. They have low infrared emissivity due to their good electrical conductivity. Among them, aluminum has the lowest cost and is widely used. However, metal composites are susceptible to surface topography. Therefore, some studies have turned to addressing the limitations of metal composites triggered by semiconductor materials. Shang et al. [31] observed that silica composite aerogels could exhibit good thermal insulation stability at room temperature. Wang et al. [32] proposed bonding the aerogel with polysiloxane to enhance its thermal insulation ability. These studies show that the aerogel has excellent thermal insulation properties.

## 3 METHODS

We first propose a method to build infrared 3D models with real infrared photos. Then, we build the 3D infrared human body model and adversarial clothing model. To make 3D modeling more realistic and adaptable, we simulate infrared clothing characteristics in different times and places. After that, we design the adversarial pattern in 3D clothing and define the optimization loss function. Finally, we introduce the physical implementation of adversarial clothing. Figure 2 shows the pipeline of the proposed method.

## 3.1 Building 3D infrared models with real infrared photos

A naive infrared modeling method is to directly convert 3D RGB models to grayscale models. Although infrared images are also grayscale images, there is a big domain gap between the rendered 3D models and the real infrared images due to the huge difference in imaging mechanisms between RGB cameras and infrared cameras.

To address the above problem, we propose a method to use real infrared images to build the 3D human model and 3D clothing model. For example, Figure 3(a) shows a 3D mesh model of the clothing. Then we use infrared photos captured by infrared cameras to create “skins” for these mesh models. One challenge is how to paste these 2D photos to the surface of 3D mesh models. To address the above challenge, we first unfold the faces of 3D mesh models onto a 2D plane, which is called faces maps, as shown in Figure 3(b). After that, we divide these faces into different regions, such as arms, back, etc. by using the MAYA software. This process facilitates the alignment of 2D real infrared images with the 3D mesh models.

![](images/86e3d5e6246264cf9250ec67bb25ac94a7b0b024495139f00edf830c4af23335.jpg)  
Fig. 2. Pipeline of the proposed method. Top: optimization in the digital world. Bottom: physical test process. The facial areas are blurred for privacy reasons.

Based on the faces map, we crop the infrared images to different parts and attach the cropped images onto the faces map using Photoshop software, and obtain the infrared texture map of 3D clothing model, as shown in Figure 3(c). We remark that the cropped infrared images may not perfectly align with the corresponding parts in the faces map. To address this issue, we handle it in two scenarios. First, if the captured image contains all the textures corresponding to the face map but exhibits slight shape mismatches, we use the distortion function in Photoshop to slightly deform the cropped images, ensuring better alignment with the corresponding face map areas. Second, if the captured image only contains partial textures corresponding to the face map (e.g., due to occlusion during capture), we first stitch images captured from different perspectives using Photoshop to create a complete texture. Then, we apply Photoshop’s distortion function to deform the stitched image, ensuring it aligns with the face map.

Finally, we use the render of Pytorch3D [51] to map the infrared texture map to the surface of the 3D clothing model. This method establishes a correspondence between the 3D mesh models and real infrared images. Figure 3(d) shows the rendered 3D clothes model based on real infrared characteristics.

Since the 3D models we build are based on real-world infrared images, it minimizes the domain gap between the rendered 3D models and the real infrared images. This method helps to simulate physical adversarial attacks in the digital world more realistically than the naive grayscale infrared modeling method. Figure 4 shows the differences of the two kinds of infrared models. It indicates that the 3D infrared clothing model using our real infrared characteristicbased method has more infrared texture information and details than the grayscale infrared model.

For the generalization of the proposed method, our method can be applied to any 3D object because we only need to take infrared photos of these objects and paste them on the surface of 3D objects. Therefore, it has good generalization properties.

![](images/636b75b5d9cc94702cc838b886ad102a93e6d2cf1eabf0203b46f5c7ece70ddf.jpg)  
(c)  
Fig. 3. Construction of 3D clothing model with real infrared characteristics. (a) 3D clothing mesh model. (b) Faces map of 3D clothing model. (c) infrared texture map of 3D clothing model. (d) Rendered 3D infrared clothing model.

![](images/7f53ff7addfa06a604940686af7144b53bf2adddd4c759857792b5d22e32e01e.jpg)  
Fig. 4. Comparison of 3D clothing models using the (a) naive grayscale infrared modeling method and (b) real infrared characteristic modeling method (Ours).

## 3.2 3D infrared human body model

2D images are uniformly represented by pixel arrays, while 3D models can be represented in multiple ways, such as meshes, point clouds, and voxel grids. We use meshes to build the proposed 3D model. Meshes are collections of vertices and faces. Compared with point cloud and voxel representation, the 3D model represented by mesh typically has a smoother surface and richer details. The human 3D mesh we constructed is shown in Figure 2, which consists of 257166 points and 128583 edges. Next, we build a “skin” of the 3D model to get closer to the real-world person. The “skin” is a figure shown in Figure $^ { 2 , }$ which can be mapped to the surface of a 3D object using the render of Pytorch3D [51]. The “skin” is created using the method described in Section 3.1. The render inputs scene information (lights, camera, materials, geometry, and textures) and outputs an image. By changing the position of the camera, we can obtain rendered images of 3D models at multiple angles in different scenes. A rendered image of the infrared human body model is shown in Figure 2.

## 3.3 3D infrared adversarial clothing model

In addition to the 3D human model, we also build a 3D clothing model to simulate people wearing clothes more realistically. The mesh of the clothing (Figure 2) is composed of 28092 points and 14064 edges. The layout of the clothing is the proposed optimization target. We want to find a specific layout pattern that has an adversarial effect when it is rendered to the surface of the clothing. To facilitate physical implementation, we design a pattern consisting of multiple black patches, as shown in Figure 2. In infrared images, the lower pixel value represents the lower temperature; thus, the black area represents the thermal insulation area, and the other areas are without thermal insulation. We introduce the physical implementation of physical adversarial clothing in Section 3.6. The combinations of multiple black patches form different patterns on the surface of 3D clothes. We can change the pattern on the surface of 3D clothes by changing the position and orientation of the black patches. A rendered image of 3D clothing with an optimized pattern is shown in Figure 2 (upper panel).

![](images/91517e4154c5f9569720689cb447adda661e2750847d12df5d388d178e7aadc7.jpg)  
Fig. 5. Different infrared characteristics of clothes at (a) day indoors, (b) day outdoors, (c) night indoors, and (d) night outdoors.

## 3.4 Simulating changeable infrared clothing characteristics in different times and places

We found that the infrared characteristics of clothes also changed with different times and places. To simulate these different characteristics, we took infrared photos of the same clothes at different times (day and night) and different places (indoors and outdoors). After that, we used our proposed infrared 3D model building method (Section 3.1) to build infrared clothing models in different situations, as shown in Figure 5. During the optimization and testing process, we used the above method to randomly change the infrared texture map of 3D clothes to simulate the infrared characteristics of clothes at different times and places.

## 3.5 Optimization of adversarial pattern in 3D infrared adversarial clothing

Figure 2 shows the optimization process of the adversarial pattern in 3D infrared adversarial clothing. We assume that there are N black squares with the same size on the pattern $P _ { \mathrm { t e x } } .$ . The variable z records the coordinates $( x _ { i } , \bar { y } _ { i } ) , i = 1 , . . . , N$ and orientations $\alpha _ { i } , i = 1 , . . . , N$ of the center points of the $N$ patches. For ease of calculation, $x _ { i }$ and $y _ { i }$ are normalized to (0, 1) according to the size of the pattern. The variable z can be represented by the following equation:

$$
z = \left[ { \begin{array} { l l l } { x _ { 1 } } & { y _ { 1 } } & { \alpha _ { 1 } } \\ { x _ { 2 } } & { y _ { 2 } } & { \alpha _ { 2 } } \\ & { \dots } & \\ { x _ { N } } & { y _ { N } } & { \alpha _ { N } } \end{array} } \right] .\tag{1}
$$

![](images/74b7ddb59c439d8a2b548e7e6cc4d63adc58bc71eaa6ae648f35729e9db4d708.jpg)  
Fig. 6. Optimization area mapped to the surface of the 3D clothing model. The dark gray area is the mapping area. The center coordinates of the patches are constrained within the black bounding box areas.

The pattern $P _ { \mathrm { t e x } }$ is generated by the variable $z ,$ and the generating function is G:

$$
P _ { \mathrm { t e x } } = G \left( z \right) .\tag{2}
$$

As mentioned in Section 3.4, we build four different texture maps for 3D infrared clothes in different times and places. During the optimization and testing process, we randomly change the infrared texture map of 3D clothes to simulate the infrared characteristics of clothes at different times and places. Let Change denote this process, so:

$$
P _ { \mathrm { t e x - C } } = \mathrm { C h a n g e } \left( P _ { \mathrm { t e x } } \right) .\tag{3}
$$

The goal of this study is to perform physical attacks; thus, we need to consider the perturbations in the real world. For example, when a person is moving, the positions of the black patches on the clothes may be shifted slightly. Therefore, we added some random noise to the coordinates and orientations of these patches. In addition, when the ambient temperature changes, the pixel value of the black patch in the infrared image may change. Therefore, we set the pixel value of the black patch to vary randomly within a certain range. To simulate the diverse textures of real clothing, we introduce random grayscale variations on the clothing surface during training. The method to simulate disturbance in the physical world is called expectation over transformation (EOT) [9]. The pattern after EOT transformation is $\begin{array} { r l } { P _ { \mathrm { t e x - C E } } \colon } & { { } } \end{array}$

$$
P _ { \mathrm { t e x - C E } } = \mathrm { E O T } \left( P _ { \mathrm { t e x - C } } \right) .\tag{4}
$$

The render in Pytorch 3D maps a 2D pattern to the surface of a 3D clothing mesh and generates a rendered image. Figure 6 shows the mapping relationship between the area on the pattern and the 3D clothing surface in our experiment. It looks like we have cut the 3D clothes into four fabrics and put them on a 2D plane. Specific areas in the pattern (marked by dark gray in Figure 6) will be mapped to the surface of the 3D clothing. The center coordinates of the patches were constrained within the black bounding box area, which ensures that black patches are in the valid mapping area. The scene parameters φ of the renderer R include lights, camera and so on. With the input of 3D clothing mesh

$M _ { \mathrm { c l o t h } }$ , pattern $P _ { \mathrm { t e x - C E } }$ , and scene parameters $\varphi ,$ the render R outputs the rendered image I<sub>cloth</sub>:

$$
I _ { \mathrm { c l o t h } } = R \left( M _ { \mathrm { c l o t h } } , P _ { \mathrm { t e x - C E } } , \varphi \right) .\tag{5}
$$

We combine the 3D person model and 3D clothing model, in other words, we let the 3D person “wear” the 3D clothing. The 3D person model includes 3D person mesh $M _ { p e r s o n }$ and skin image $P _ { s k i n }$ . We use the render to generate the rendered image of a man wearing the clothing:

$$
I _ { \mathrm { m a n } } = R \left( M _ { \mathrm { p e r s o n } } , P _ { \mathrm { s k i n } } , I _ { \mathrm { c l o t h } } \right) .\tag{6}
$$

To simulate the attack effect in different scenes, we paste the rendered images $I _ { \mathrm { m a n } }$ to different background images $I _ { \mathrm { b a c k } }$ and obtain the pasted image $I _ { \mathrm { p a s t e } } \mathrm { : }$

$$
I _ { \mathrm { p a s t e } } = \mathrm { P a s t e } \left( I _ { \mathrm { m a n } } , I _ { \mathrm { b a c k } } \right) .\tag{7}
$$

We input the pasted image $I _ { \mathrm { p a s t e } }$ into the object detection model f parameterized by θ. The output of the detection model includes the bounding box $f _ { \mathrm { p o s } } \mathrm { \hat { ( } } I _ { \mathrm { p a s t e } } , \theta \mathrm { ) }$ , the object score $f _ { \mathrm { o b j } } \left( I _ { \mathrm { p a s t e } } , \theta \right)$ , the class score $f _ { \mathrm { c l s } } \left( I _ { \mathrm { p a s t e } } , \theta \right)$ . The goal is to make the detector unable to detect the person. Then, we try to lower $f _ { \mathrm { o b j } } \left( I _ { \mathrm { p a s t e } } , \theta \right)$ as much as possible.

Considering that there are some other persons in the background images $I _ { \mathrm { b a c k } } ,$ the detector may output multiple predicted bounding boxes $\boldsymbol { B } ~ = ~ \left\{ b _ { j } \right\} _ { j = 1 } ^ { T ^ { * } }$ . To avoid the interference of these background persons, we only keep the bounding box closest to the ground truth of the 3D model $( \mathrm { i . e . , }$ the box that has the max object score when its intersection over union (IOU) with the ground truth (GT) is above a certain threshold ε<sub>IOU</sub>). We define the following loss function $L _ { \mathrm { d e t } }$

$$
L _ { \mathrm { d e t } } = \operatorname* { m a x } _ { j \in \mathcal { T } } f _ { \mathrm { o b j } } ^ { ( j ) } ( I _ { \mathrm { p a s t e } } , \theta ) ,\tag{8}
$$

where:

$$
\mathcal { I } = \{ 1 , 2 , . . . , T \} \cap \{ j | \mathrm { I O U } \left( \mathrm { g t } , b _ { j } \right) > \varepsilon _ { \mathrm { I O U } } \} .\tag{9}
$$

The optimization target is the positions and orientations of black patches on the surface of 3D clothes. We want to optimize a pattern with a special arrangement of black square patches on the clothes that can hide from detectors. These black patches correspond to physical insulation materials. To facilitate the physical implementation, we try to minimize the overlap between different patches. We propose a new loss function, $L _ { \mathrm { d i s t . } }$ , which aims to increase the average distance between every two patches as much as possible within a certain range. $( x _ { i } , y _ { i } )$ and $( x _ { j } , y _ { j } )$ are the coordinates of the center point of every two patches, and −d is the lower bound of the $L _ { \mathrm { d i s t } }$ to prevent the distance between every two patches from being too large:

$$
L _ { \mathrm { d i s t } } = \operatorname* { m a x } \left( - \frac { \displaystyle \sum _ { \forall i \neq j } ^ { N } \sqrt { \left( x _ { i } - x _ { j } \right) ^ { 2 } + \left( y _ { i } - y _ { j } \right) ^ { 2 } } } { N \times \left( N - 1 \right) / 2 } , - d \right) .\tag{10}
$$

The loss function of the detector is:

$$
{ \cal L } = { \cal L } _ { \mathrm { d e t } } + \lambda { \cal L } _ { \mathrm { d i s t } } ,\tag{11}
$$

where λ is a hyperparameter that is determined empirically.

For ensemble attacks, we want to lower the object scores of multiple detectors concurrently. We assume that there are K detectors, and the object score output by each detector is $L _ { \mathrm { d e t } } ^ { ( k ) } , k = 1 , . . , K$ . The loss function of the ensemble attack is:

$$
L _ { \mathrm { e n s e m } } = \sum _ { k = 1 } ^ { K } L _ { \mathrm { d e t } } ^ { ( k ) } + \lambda L _ { \mathrm { d i s t } } .\tag{12}
$$

We use the backpropagation algorithm to update the variable z according to the loss function and update the pattern of the clothes $\dot { P _ { \mathrm { t e x } } }$

## 3.6 Physical implementation

Thermal insulation materials block the thermal radiation of the human body; thus, the thermal-insulated areas appear black in the infrared images. We compared the thermal insulation performance of two common fabrics (cotton and polyester), a common tape (rubber tape), two thermal insulation tapes (polyimide, Teflon), and a new material - aerogel. We found that the aerogel has good thermal insulation properties that are stable over time. We purchased a piece of 1.5m × 1.5m × 6mm aerogel cloth from ZhongPu Company (Hebei, China) and cut it into many 6cm × 6cm × 6mm aerogel patches.

We can choose a piece of physical clothing that is similar to the shape and size of the 3D clothes created in the digital world. According to Figure 6, we draw four rectangles on the clothes. In each rectangle, we set up a Cartesian coordinate system whose origin is at the center of the box, and the X and Y axes are parallel to the sides of the rectangular box. This process is performed on both the digital and physical clothes. Then, the positions and orientations of patches in the digital world are mapped to the physical world. This mapping does not need to be very precise because the optimization of the patch layout pattern in the digital world has considered random perturbations (see EOT transformations in Section 3.3). The aerogel patches are fixed at those positions. The areas outside the rectangular boxes on the clothes are not our optimization areas; thus, we allow some differences between the physical clothes and the 3D model in these outside areas.

In our experiment, we used a sunscreen coat that had a zipper on the front (Figure 2, bottom), which was helpful for placing the aerogel patches inside the coat. The collar of the coat was different from the collar of the 3D model clothes, but as explained above, this difference is allowed. We used tape and thread to fix the aerogel patches on the coat.

Infrared imaging has the penetrating ability. Even if these aerogel patches are on the inner side of clothes, the pattern on the clothes can be displayed under infrared imaging. The physical adversarial clothes look like ordinary clothes in a visible light view.

## 4 EXPERIMENTS

## 4.1 Dataset

We chose the FLIR ADAS 1 3 [34] dataset released by FLIR Corporation. FLIR ADAS 1 3 provides annotated infrared images for training and testing object detection tasks. The dataset contains a total of 10228 infrared images. We used 7160 images as the training set and 3068 images as the test set. These images were taken on the highways and streets of Santa Barbara, USA, from November to May. The infrared camera was a FLIR Tau2 (NETD<60 mK, FPA 640 × 512, 13mmf /1.0). Infrared images were manually annotated and contain four types of objects: people, dogs, cars, and bicycles. We used these real-world infrared images to finetune the infrared object detector and as the background for the 3D models.

## 4.2 Target detectors

We chose the latest YOLOv9 [61] detector as our primary target detector, and finetuned the officially pre-trained YOLOv9 detector on the FLIR ADAS 1 3 dataset. The average precision for person category of the finetuned detector was 0.98 on the training set and 0.95 on the test set. After attacking YOLOv9 in the white-box setting, we evaluated the transferability of our method to other unseen detectors such as DETR [39], DINO [62], DDQ [63], DAB-DETR [67], YOLOv7 [64], YOLOv8 [65], YOLOv11 [66], Mask-RCNN [37], and Cascade RCNN [49] in the black-box setting. These models are provided by the mmdetection library [69].

## 4.3 Evaluation Metrics

We used the attack success rate (ASR) as the evaluation metric, which is commonly used in many researches [2], [4], [5], [6], [8], [16] in this area. In digital attacks, we defined the annotation of the 3D person model (digital attack) or real persons (physical attack) as ground truth (GT), and used the intersection over union (IOU) method to calculate the detection accuracy. Because the background image (FLIR ADAS dataset or real photos) contains other persons, we defined the evaluation metric ASR as the number of undetected 3D human models divided by annotated 3D human models (digital attack) or undetected persons divided by annotated persons (physical attack). When the detector outputs multiple bounding boxes, we selected the box with the IOU with GT greater than 0.5 and the highest object score with the threshold 0.7 as the detection result.

## 4.4 Simulation of physical attacks

We conducted all experiments on 8 2080TI GPUs.

(a)  
(b)  
![](images/4ac2bbc79d0beeec428a4248da368ae1d490cf5d7215a57dd99024c0bb41dda5.jpg)  
(c)  
(d)

![](images/f890dbdce52b392d859cd456e812dbfdedec19609cede3b64d121f26171580bb.jpg)

![](images/b344834f304634e99ab20f2b3b868a594e560e966571902bac9f733eae35b73a.jpg)

![](images/fbee94d7effa8df35cd774a53586344df5892802c924c5332403669fb62487cc.jpg)

![](images/1f40bbbbb69a731fcfe2faa8ef7959fd7a13348c761bdaacd3c3d2f5bdb9fde6.jpg)  
Fig. 7. Different patterns of the texture map and rendered 3D clothing model. (a) Clean pattern. (b) Random position pattern. (c) Adversaria pattern optimized by YOLOv9. (d) Adversarial pattern optimized by ensemble model.

TABLE 1  
Evaluation of digital attacks
<table><tr><td>Pattern of 3D clothing model</td><td>ASR</td></tr><tr><td>Ordinary clothing</td><td>4.65%</td></tr><tr><td>Random pattern</td><td>38.19%</td></tr><tr><td>Optimized pattern</td><td>86.38%</td></tr></table>

## 4.4.1 Attack YOLOv9 in the digital world

As mentioned in Section 3.5, the pattern $P _ { \mathrm { t e x } }$ of 3D clothing contains N black patches. We took N = 36 in our experiments (see Section 4.4.2 for the study of other values). According to the ratio of different areas on the clothes, we mapped 12 (N/3) patches onto the front of the clothes, 12 (N/3) patches onto the back of the clothes, and 6 (N/6) patches onto the left and right sides of the clothes. The center coordinates of the patches were constrained within the bounding box (the area marked by the black bounding box in Figure 6) during optimization and were randomly initialized. This ensured that black patches were always in the valid mapping area (marked by dark gray area in Figure 6). The EOT transformations include random noise ([-0.01,0.01]), random shifting ([-0.5 cm, 0.5 cm]), random rotation ([-0.03 radians, 0.03 radians]), random brightness ([0, 0.1]), and random grayscale variations ([-0.3, 0.3]).

We used the differentiable render in Pytorch3D to map the pattern on the 3D clothing surface. Then we combined the 3D clothing model and 3D person model, that is, we let the 3D person “wear” the 3D clothing. We set the height of the 3D human model to 1.75m and the size of the black patch to be 6cm × 6cm (see Section 4.4.3 for the study of other sizes). In the digital world, the distance between the camera and the human body varied from 2.5m to 5m, and the angle varied randomly between 0-360 degrees. We obtained rendered images at different angles and distances. We randomly selected the background images from FLIR ADAS 1 3 training set, and pasted the rendered images to the center of the background images, so that we could switch various scenes for attack simulation.

We input the images with the background into the YOLOv9 detector. We used the Adam optimizer with momentum. The optimizer used the backpropagation algorithm to update the variable z, which records the positions and orientations of the 36 black patches in the pattern, and then updates the patch layout pattern. The hyperparameter λ in Equation (11) was 0.1 (the sensitivity of λ is analyzed in Section 4.4.4). The initial learning rate was 0.03. If the loss function did not drop after 50 iterations, then the learning rate gradually decreased until 0.001, and then the optimization process ended at that time. We obtained a stable pattern using this method. It took approximately 6 hours to obtain the optimization result on our GPU server for each experiment. Figure 7(c) shows the optimized pattern.

Next, we rendered the optimized pattern on the surface of the 3D clothing model, and tested its attack effect on the test set of FLIR ADAS 1 3. The settings of the testing process were the same as the optimization process. For fair comparison, we used the ordinary clothing pattern (Figure 7(a)) and random postion pattern (Figure 7(b), with 36 randomly placed black patches) for control experiments. These different patterns were rendered on the same 3D model, and the rendered images are input to the YOLOv9 detector. We defined the annotation of the 3D person model as ground truth (GT), and used the intersection over union (IOU) method to calculate the detection accuracy. ASR is used as the attack evluation metric.

Table 1 shows the results. The ASR of the optimized pattern (86.38%) against YOLOv9 was obviously better than that of random patten (38.19%) and clean pattern (4.65%). Figure 8 shows three types of clothes at three different viewing angles.

## 4.4.2 Effect ofthe number ofblack patches

As described in Section 4.4.1, the pattern included 36 black patches. Then, we tried different patterns, which included 6, 12, 24, 36, 48, and 60 black patches, to investigate how the number of black patches in the clothing pattern affects the attack performance of 3D clothing. The optimization and testing process were the same as described in section 4.4.1. The results are shown in Table 2. The results indicated that when the number of patches increased from 6 to 60, the attack performance increased first and then decreased. When N was 36, the attack effect was the best. The decreasing attack effect with many more patches is consistent with our recent study [8] that fully insulated clothing had only weak attack effect.

## 4.4.3 Effect of black patch size

In Section 4.4.1, the size of the black patch was 6cm×6cm in 3D modeling. Then, we studied the attack effect of patterns with patch sizes of 2 cm, 4 cm, 6 cm, 8 cm, 10 cm, and 12 cm. Other settings were the same as in Section 4.4.1. The results are shown in Table 3. The results indicated that with the increase in patch size, the ASR first increased and then decreased. The size of 6cm × 6cm was the best.

![](images/e00260c29720beeca18e5223250350cbd74c60ae53878c4682bc7bff8c3ab21b.jpg)

![](images/9e9bce8519a31523c95a82eb6ae4e1980468c16eee6ec2fc1bb9850e4c9fc61e.jpg)

![](images/d96c8218c69984c91141a3d7df60366b8734c25567ce4514b335cf828ba38780.jpg)

![](images/cf27831dc5972a220e581089df3533ccb2bf5eabb2a31344c7de1d4764d82696.jpg)

![](images/725b903af4c63de7370fc952964113d4fb767fa0abaad5ec015c98e71a9ce2fc.jpg)

![](images/061f0b0e4a150bd0c169ec64ad359190604583e6832fb57d14648da7712267c4.jpg)

![](images/500d8b97c8876d7eb93205e92ef6e2b6e04835ac3895b742b84271923192d9de.jpg)  
(a)

![](images/dd3eb0329c8fa72a5f4598978d3dfa97f786a976965f1111fe1e75ddcd7ae638.jpg)  
(b)

![](images/8a8e2b0f92a0a29067a534459a907c56f9db60ecb46e4f060623504f39a68820.jpg)  
(c)  
Fig. 8. Examples of digital attacks. (a) Ordinary clothing. (b) Random pattern. (c) Optimized pattern.

TABLE 2  
Effect of the number of black patches
<table><tr><td>The number</td><td>6</td><td>12</td><td>24</td><td>36</td><td>48</td><td>60</td></tr><tr><td>ASR</td><td>22.38%</td><td>50.32%</td><td>72.18%</td><td>86.38%</td><td>85.25%</td><td>83.73%</td></tr></table>

TABLE 3

Effect of black patch size
<table><tr><td>Black patch size (cm)|</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td></tr><tr><td>ASR</td><td>51.30%</td><td>70.07%</td><td>86.38%</td><td>85.95%</td><td>81.31%</td><td>78.12%</td></tr></table>

## 4.4.4 Sensitivity analysis of λ in the loss function

The hyperparameter λ in Equation (11) balances $L _ { d e t }$ and $L _ { d i s t }$ . We set λ = 0.1 by default. To analyze the sensitivity of λ, we set λ to be 0, 0.01, 0.05, 0.1, 0.5, and 1. We kept other optimization and testing settings same as described in Section 4.4.1. Table 4 shows the results. The results indicated that when the λ value increased from 0 to 0.1, the ASR increased. But when λ was greater than 0.1, the ASR decreased. $\lambda = 0 .$ 1 was the best.

## 4.4.5 Effect ofdetection threshold

We tested different object score thresholds and calculated the ASRs at different thresholds (from 0.1 to 0.9). We plot the curve of ASR with different thresholds, as shown in Figure 10.

The results show that the ASR will increase as the detection threshold increases. Despite the change in threshold, the ASRs of our adversarial pattern are obviously higher than that of random pattern and clean pattern. However, we found that if the threshold is too high, the recall of clean pedestrian detection will decrease. If the threshold is too low, the false detection rate will increase. Therefore, a typical threshold is 0.75 in the original YOLOv9 paper [61], which is similar to 0.7. Besides, in pedestrian attacks, the commonly used settings are generally 0.6 or 0.7, which is the same in many previous works [2], [5], [6], [8].

## 4.4.6 Evaluation on different datasets

We evaluated the proposed method on the other two infrared pedestrian datasets: KAIST [58] and FLIR ADAS v2 [59]. We used these datasets to finetune the object detector YOLOv9 and as the background for 3D model simulation. The settings of other experiments are consistent with Section 4.4.1. The results are shown in Table 5. Our method achieved good results on different datasets, and the difference in ASR is not significant. It indicates the generalizability of our method across different datasets.

![](images/4356f947ea09ca183a428e9ac49314cae9a808226c5ddb0efd1714aea44e26cf.jpg)

![](images/82b6a3d6e3736a5cbf02132396119ab5ac30173b28a78a48bdac2ae155eafb2c.jpg)

![](images/39717bcc363952370cf0261f62bc8aca14751764301a32bab0bd19dfc7be96f5.jpg)

![](images/7998dc0a09a1b607a339438c035fcc03152b6cba70a6fee0395fd9c97704efab.jpg)

![](images/88cee2e9e0915ea4b4e49b18fc107445ca14eaf29c8234f6df0eb55363d03259.jpg)

(a)  
![](images/750e9b885fdb3deb5fe430e2d6495f2616bee43449bd2c18853d4a2583c37156.jpg)  
(b)

![](images/550297921b3d690639f9516c4504dba392334a5558dba996f931a1333e3191c8.jpg)  
(c)

![](images/b95384659c4a1d330fc319b8c14ddeb5ab3c0c90bfe25fe18d266aa1afdb65eb.jpg)  
(d)

Fig. 9. Physical adversarial clothing. (a) Ordinary clothing. (b) Random pattern clothing. (c) Adversarial clothing optimized with respect to YOLOv9. (d) Adversarial clothing optimized with respect to the ensemble model.  
TABLE 4  
Sensitivity analysis of λ in the loss function
<table><tr><td>The value of λ</td><td>0</td><td>0.01</td><td>0.05</td><td>0.1</td><td>0.5</td><td>1</td></tr><tr><td>ASR</td><td>72.66%</td><td>75.35%</td><td>83.19%</td><td>86.38%</td><td>80.10%</td><td>76.61%</td></tr></table>

TABLE 5

ASRs on different datasets.
<table><tr><td></td><td>Dataset</td><td>FLIR_ADAS_1_3</td><td>KAIST</td><td>FLIR_ADAS_v2</td></tr><tr><td colspan="2">Method</td><td></td><td></td><td></td></tr><tr><td colspan="2">Clean</td><td>4.65% 38.19%</td><td>6.59% 36.71%</td><td>5.31% 38.27%</td></tr><tr><td colspan="2">Random Adversarial</td><td>86.38%</td><td>83.27%</td><td>85.58%</td></tr></table>

![](images/a7d6d80aa1ecc2ce4a1a3668c9937ace6c25ab2d8c93d688276028cb383769b0.jpg)  
Fig. 10. ASRs at different object score thresholds.

## 4.5 Evaluation of attacks in the physical world

In our previous work [8], we tested the thermal insulation properties of five materials, and found that the aerogel material had better thermal insulation properties than other materials and remained stable over time. We therefore used this material in this study.

## 4.5.1 Attack YOLOv9 in the physical world

We manufactured physical adversarial clothing based on the optimized 3D clothing model, and the manufacturing process is described in Section 3.6. For a fair comparison, we used random pattern clothing (the clothing with 36 randomly placed aerogel patches) and ordinary clothing for the control experiments. All these clothes had the same size. Figure 9 shows these physical clothes.

We tested the attack effect of these clothes in the physical world. We invited 8 volunteers. The human-related experiments have been approved by the Ethics Committee of Tsinghua University. The volunteers wore the adversarial clothing, the random pattern clothing, or ordinary clothing. We used the infrared camera FLIR T630sc to capture these volunteers in multiple scenes indoors and outdoors. They could stand, sit or in any posture they want. The angle from the camera to human body varied between 0-360 degrees, and the distance was between 1-12 meters. We input the infrared photos into the YOLOv9 detector with a threshold 0.7. Some examples are shown in Figure 11. The results showed that people wearing adversarial clothing were not detected, while people wearing the random pattern clothing, or ordinary clothing were detected. This indicated the effectiveness of the proposed method. See Supplementary Video for the demo.

![](images/b3dd586e2ec627952fd4cd0a9230ef68447777e6de9f840af66706f52b144a7a.jpg)  
Fig. 11. Examples of physical infrared attacks. A: adversarial clothing. R: random pattern clothing. O: ordinary clothing.

To quantitatively evaluate the attack effect, we recorded 100 videos. Half of the videos were recorded indoors; the other half were recorded outdoors. We fixed the position of the camera, and the distance between the volunteer and the camera was $3 m , 5 m ,$ or $7 m .$ . When we recorded the videos, the volunteers rotated in situ at a constant speed. For a fair comparison, the same volunteers wore ordinary clothes, the random pattern clothing, and the adversarial clothing, respectively at the same position. We sampled these videos at 3 frames per second and obtained 6000 images. The images that include each kind of clothing took $1 / 3$ of the total images. We manually annotated all images and input them into YOLOv9. Table 6 shows the results. The results indicated that, the ASRs of adversarial clothing against YOLOv9 were 80.11% (indoor) and 76.85% (outdoor) in the physical world, while ASRs of the random pattern clothing and ordinary clothing were only 26.53% and 7.12%, respectively indoors, and 23.03% and 8.38%, respectively outdoors in the physical world. The results showed the effectiveness of our physical attacks compared to the control experiments. We also noticed that physical attacks were more difficult in outdoor scenes, possibly because the outdoor environment is more complex. For example, when the weather is hot (e.g. higher than 30 degrees Celsius), the temperature of the aerogel patches is close to that of the human body. Therefore, the pixel values of black patches are close to those of other areas in infrared images of adversarial clothing, and the adversarial pattern is not shown clearly.

TABLE 6  
Evaluation of physical attacks
<table><tr><td>Physical clothing</td><td>ASR (indoor)</td><td>ASR (outdoor)</td></tr><tr><td>Ordinary clothing</td><td>7.12%</td><td>8.38%</td></tr><tr><td>Random pattern clothing</td><td>26.53%</td><td>23.03%</td></tr><tr><td>Adversarial clothing</td><td>80.11%</td><td>76.85%</td></tr></table>

![](images/ebbf6fdf27c680418647177289a5b9ccd16f765b04e9d2e82e615794fcb6177a.jpg)  
(a)

![](images/cc2511756beeae72a49fe14f4a0226c0b2fdfe24a3bbfb20533538e74f5dceaa.jpg)  
(b)  
Fig. 12. Analysis of physical attack effects at (a) different angles and (b) different distances.

![](images/c6d19195c46fe8040582ea8a1d855347eccdc69cf845e0f59536305816e0d4b9.jpg)  
Fig. 13. Examples of different weather conditions: (a) sunny, (b) cloudy, and (c) foggy. A: adversarial clothing. R: random pattern clothing. O: ordinary clothing.

We then analyzed the attack performance of adversarial clothing from different angles. The distance between the volunteers and the camera was fixed at 5 meters. The volunteers rotated in situ at a constant speed. We took 11 sample points between 0-360 degrees and marked key angles on the ground. We selected 100 images at each angle for statistics, and used the ASR as the evaluation method. We input these images into the YOLOv9 detector with the threshold 0.7. Figure 12(a) shows the results. The results indicated that the ASR of our adversarial clothing was higher than 77%in the range of 0-360 degrees. Besides, the ASR of the front (or back) of the adversarial clothing is better than ASR of the side, which may be due to the relatively small adversarial area on the side of the clothing.

Next, we analyzed the attack performance of adversarial clothing at different distances. The volunteers faced the camera (0 degrees) and the distance varied from 3 meters to 10 meters. We took 8 sample points between 3-10 meters and selected 100 images at each sample point. We also used ASR as the evaluation method and input these images into the YOLOv9 with the threshold of 0.7. The results are shown in Figure 12(b). The results indicated that our adversarial clothing had more than 78% ASR in the range of 3-8 meters. When the distance was greater than 8 meters, the attack performance dropped quickly, because the adversarial pattern on the clothes was difficult to attack the detector in a smaller view.

## 4.5.2 Evaluation across various weather and landscape conditions

We evaluated the performance of the adversarial clothing (Figure 9(c)) under more diverse weather and landscape conditions. The weather conditions included sunny, cloudy, and foggy environments, while the landscapes included urban roads, hillsides, and riversides. For each case, we used a FLIR T560 infrared camera to capture 1200 images of volunteers wearing the adversarial clothing, and the attack success rate was computed accordingly. Other experimental settings followed those described in Section 4.5.1. The results are presented in Tables 7 and 8. Several visual examples are shown in Figure 13 and 14.

TABLE 7  
Evaluation across various weather conditions.
<table><tr><td>Weather</td><td>Sunny</td><td>Cloudy</td><td>Foggy</td></tr><tr><td>ASR</td><td>78.08%</td><td>75.83%</td><td>74.58%</td></tr></table>

![](images/96e4e5851ea22ac9bb821af2d2a019f2fd623863e0a6657ffa753968b3838c7e.jpg)  
Fig. 14. Visual examples of different landscapes: (a) city road, (b) hillside and (c)riverside. A: adversarial clothing. R: random pattern clothing. O: ordinary clothing.

The results show that our method achieved an ASR over 74% across various weather and landscape conditions, indicating its effectiveness. This performance can be attributed to the following factors. First, our algorithm pipeline incorporates simulations of multiple physical-world perturbations, such as random noise, temperature variations, diverse clothing texture changes, and random background switching. These augmentations enhance the robustness of our method to environmental variations in real-world scenarios. Second, thermal imaging inherently benefits from its penetration capability, allowing it to capture thermal images clearly even in the presence of environmental disturbances such as fog. As a result, the adversarial patterns we designed remain clear and effective in complex settings.

## 4.5.3 Evaluation of the stability of adversarial clothing over time

We evaluated the stability of the adversarial clothing over time. We recorded 3600 images of volunteers wearing the adversarial clothing (Figure 9(c)) over a total duration of 5 hours. The other experimental settings were consistent with those described in Section 4.5.1. The results are presented in Table 9.

The results indicate that while the attack success rate of our adversarial clothing decreases slightly over time, it still maintains a success rate of 77.83% after 5 hours, demonstrating the long-term stability of our method. This stability can be attributed to findings from our previous work [8], where we tested the thermal insulation properties of five materials, and found that the aerogel material had better thermal insulation properties than other materials and remained stable over time. Consequently, our aerogel-based clothing can consistently display the adversarial patterns, ensuring sustained attack effectiveness.

TABLE 8  
Evaluation across various landscape conditions.
<table><tr><td>Landscapes</td><td>City Road</td><td>Hillside</td><td>Riverside</td></tr><tr><td>ASR</td><td>76.25%</td><td>77.17%</td><td>75.08%</td></tr></table>

TABLE 9  
Evaluation of the stability of adversarial clothing over time.
<table><tr><td>Time</td><td>0h</td><td>1h</td><td>2h</td><td>3h</td><td>4h</td><td>5h</td></tr><tr><td>ASR</td><td>83.17%</td><td>82.50%</td><td>82.00%</td><td>80.33%</td><td>78.67%</td><td>77.83%</td></tr></table>

TABLE 10

Transferability in the digital world. The numbers are ASRs.
<table><tr><td>Test Train  $\sim$ </td><td>DETR</td><td>DINO</td><td>DDQ</td><td>DAB</td><td>YOLOv7</td><td>YOLOv8</td><td>YOLOv11</td><td>Mask</td><td>Cascade</td></tr><tr><td>YOLOv9</td><td>29.75%</td><td>26.27%</td><td>27.51%</td><td>30.83%</td><td>53.96%</td><td>58.36%</td><td>50.28%</td><td>43.36%</td><td>40.06%</td></tr><tr><td>Ensemble_models</td><td>81.03%</td><td>70.49%</td><td>75.20%</td><td>73.42%</td><td>78.13%</td><td>80.28%</td><td>76.01%</td><td>72.57%</td><td>70.35%</td></tr></table>

TABLE 11

Transferability in the physical world. The numbers are ASRs.
<table><tr><td>Test Train</td><td>DETR</td><td>DINO</td><td>DDQ</td><td>DAB</td><td>YOLOv7</td><td>YOLOv8</td><td>YOLOv11</td><td>Mask</td><td>Cascade</td></tr><tr><td>YOLOv9 Ensemble_models</td><td>21.69% 70.05%</td><td>18.55% 57.29%</td><td>19.84% 60.85%</td><td>22.49% 56.25%</td><td>46.29% 65.16%</td><td>50.11% 68.33%</td><td>41.91% 62.55%</td><td>39.22% 68.04%</td><td>31.06% 59.73%</td></tr></table>

## 4.5.4 Evaluation of the robustness against different postures

We evaluated the robustness of our physical attack method against different postures. Volunteers could choose either a standing or sitting posture, and for each, their arm positions were categorized as arms hanging, arms lifted, arms spread, or arms crossed. Some visualization examples are shown in Figure 15. For each case, we recorded six videos of approximately 10 seconds each and then computed the ASR. The results are shown in Table 12 and Table 13.

The results indicate that, first, the ASR for standing poses is higher than for sitting poses, likely because our

![](images/222b65a6831df4ff65d4b8a8b47b128b86ec3da43901c669ed4264765a15d36e.jpg)  
(a)

![](images/9c99f348c7a7762854a94ccfbe0690b0bc545abd5d1c86cf734e3a1e23293cd3.jpg)  
(b)

![](images/4ec8bf5610d13fa726b1852411dd3f4ebb302b66f6b534bbe0a08741e123db22.jpg)  
(c)

![](images/8eb6ec7a29b9795152530f19af09060f722707f3a6e304d4f0f71cb8a66778bb.jpg)  
(d)  
Fig. 15. Visual examples of volunteers with arms (a) hanging, (b) lifted, (c) spread, and (d) crossed in a standing (top row) or sitting (bottom row) status.

3D simulation model represents a standing posture. Second, the highest ASR is observed when the arms are hanging down, which aligns with our 3D model. The ASR decreases slightly (by 2.21-7.84%) when the arms are lifted or spread, but it drops significantly (20.27-21.32%) when the arms are crossed. This is because our adversarial pattern design concentrates most of the patterns on the front or back of the body—regions that remain relatively stable across different poses and greatly influence the attack’s effectiveness. While lifting or spreading the arms does not disturb these core areas, crossing the arms covers the adversarial patterns on the front, leading to a more pronounced decrease in ASR.

## 4.6 Ensemble attacks

We found that the attack transferability of a single model (YOLOv9) was limited. The ASR of optimized pattern (Figure 7(c)) against DETR [39], DINO [62], DDQ [63], DAB-DETR [67], YOLOv7 [64], YOLOv8 [65], YOLOv11 [66], Mask-RCNN [37], and Cascade RCNN [49] was only 29.75%, 26.27%, 27.51%, 30.83% 53.96%, 58.36%, 50.28%, 43.36%, and 40.06%, respectively in the digital world.

To improve the attack transferability, we used the model ensemble method as described in Section 3.5. We obtained a new 3D clothing pattern (Figure 7(d)) by integrating Deformable DETR [68], YOLOv9, and Faster-RCNN [60]. The optimization process was the same as Section 4.4.1. As shown in Table 10, the ASR of ensemble optimized pattern against DETR , DINO, DDQ, DAB-DETR, YOLOv7, YOLOv8, YOLOv11, Mask-RCNN, and Cascade RCNN was 81.03%, 70.49%, 75.20%, 73.42%, 78.13%, 80.28%, 76.01%, 72.57%, and 70.35%, respectively in the digital world. It indicated that the model ensemble effectively improved the attack transferability of the proposed method.

TABLE 12  
Impact of arm posture on the ASR in a standing state.
<table><tr><td>Arm posture</td><td>Hanging</td><td>Lifted</td><td>Spread</td><td>Crossed</td></tr><tr><td>ASR</td><td>81.06%</td><td>76.85%</td><td>73.22%</td><td>60.79%</td></tr></table>

TABLE 13

Impact of arm posture on the ASR in a sitting state.
<table><tr><td>Arm posture</td><td>Hanging</td><td>Lifted</td><td>Spread</td><td>Crossed</td></tr><tr><td>ASR</td><td>77.51%</td><td>75.30%</td><td>71.98%</td><td>56.19%</td></tr></table>

Next, we manufactured physical clothes (Figure 9(d)) based on the 3D clothing model obtained by ensemble model. We tested its attack performance in the physical world, and the results was shown in Table 11. The results showed that the ASRs of ensemble optimized pattern against DETR, DINO, DDQ, DAB-DETR, YOLOv7, YOLOv8, YOLOv11, Mask-RCNN, and Cascade RCNN were obviously higher than that of the adversarial pattern optimized by single model YOLOv9 in the physical world. It indicated that our ensemble method could perform transferable attacks to unseen models in the real world.

The ensemble model-based method generally achieves better attack performance on black-box models compared to single-model optimization, possibly due to the following reasons:

First, the ensemble model-based method can reduce the overfitting of the adversarial pattern to a single model, and the adversarial pattern obtained by the ensemble model can be more robust. Optimizing adversarial patterns for a single model (e.g., Model A) may yield patterns that are highly effective against Model A but fail on other models. In contrast, optimizing adversarial patterns simultaneously for multiple models (e.g., Models A, B, and C) requires the patterns to align with the decision boundaries of various models, making them more versatile in disrupting feature extraction across different models. As a result, the patterns are more likely to succeed against unseen models like Model D.

Second, the ensemble model can capture the diversity of different models, and the adversarial pattern obtained by the ensemble model can attack the common features of different models. Different models vary in architecture, training data, and loss functions. Integrating multiple models into the optimization process produces adversarial patterns that capture a broader range of disruptive modes, increasing resilience to inter-model differences. Such patterns are less reliant on exploiting specific weaknesses of a single model, resulting in higher attack success rates against previously unseen models.

## 4.7 Adversarial defense methods

We tested seven typical adversarial defense methods to defend against the proposed 3D model-based attack in the digital world. These methods include SAC [52], PAD [91], and DIFFender [92], Random Patch Augmentation, Adversarial Training [13], Pixel Mask [53], Bit Squeezing [19], Spatial Smoothing [19], and Total Variation Minimization (TVM) [54].

For SAC, PAD, and DIFFender, we used the official code implementation of the original paper [52], [91], [92]. For

TABLE 14  
Evaluation of defense methods
<table><tr><td>Defense Methods</td><td>ASR decrease</td></tr><tr><td>SAC</td><td>8.67%</td></tr><tr><td>PAD</td><td>9.58%</td></tr><tr><td>DIFFender</td><td>10.27%</td></tr><tr><td>Random Pattern Augmentation</td><td>9.18%</td></tr><tr><td>Adversarial Training</td><td>11.28%</td></tr><tr><td>Pixel Mask</td><td>7.31%</td></tr><tr><td>Bit Squeezing</td><td>3.85%</td></tr><tr><td>Spatial Šmoothing</td><td>4.52%</td></tr><tr><td>TVM</td><td>6.19%</td></tr></table>

Random Patch Augmentation, we incorporated the random black patches as data augmentation during pedestrian object detection training. The ratio of images with random black patches to clean images during the training process was 1:9 (Incorporating more adversarial examples in the training set such as the ratio of 2:8 or 3:7 causes the AP of clean images to drop by more than 10%), which is to balance the robustness and AP of the model. For Adversarial Training, we used data augmentation. We generated adversarial images using the methods described in Section 3. Then we added the adversarial images to the training set to train a more roboust model. The ratio of adversarial images to clean images during the training process was also 1:9. For Pixel Mask, we used one 20×20 mask to erase the adversarial clothing pattern at random positions. For Bit Squeezing, we converted 8-bit images to 6-bit images. For Spatial Smoothing, we used the typical median filtering method, which was implemented in the SciPy [50] Ndimage library. For TVM, we used the TVM module in Adversarial Robustness Toolbox [55] library. The other optimization settings were the same as those described in Section 4.4.1.

We used the adversarial pattern (Figure 7) to test the effect of these defense methods. Results in Table 14 indicated that the effect of these defense methods was limited. The most effective method (adversarial training) decreased the ASR of YOLOv9 by only 11.28% (from 86.38% to 75.10%), while the proposed attack method still achieved the ASR of 75.10%.

## 5 CONCLUSION

This paper presents a new infrared physical attack method based on 3D modeling. We constructed 3D human and clothing models under infrared imaging, and simulated their attack effects in different scenes. To simulate the infrared characteristics more realistically, we propose a method to build infrared 3D models with real infrared photos. We also build different texture maps for 3D clothing model to simulate changeable infrared clothing characteristics in different times and places. We designed a pattern of 3D infrared clothing by optimizing the positions and orientations of black patches, and manufactured a physical clothing based on the thermal-insulated material aerogel. Our clothing looks like ordinary clothing in the visible light view, but can hide from infrared detectors at multiple angles in different scenes in the physical world. We effectively improved the attack transferability to unknown models by using the model ensemble technique.

## 6 LIMITATION

Our method also has several limitations. First, when the adversarial clothes are too far from the camera, it is difficult to attack the infrared detectors in a quite small view. This is reasonable because the adversarial images are blurry when scaled to very small sizes. Second, when the physical adversarial pattern is severely occluded, its attack effect decreases a lot. This is because the occluded adversarial pattern in the physical world deviates greatly from the adversarial pattern optimized in the digital world. We will address these challenges in future work.

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (No. U2341228).

## REFERENCES

[1] S. Thys, W. V. Ranst, and T. Goedeme, “Fooling automated surveil-´ lance cameras: Adversarial patches to attack person detection,” in IEEE Conf. Comput. Vis. Pattern Recog. Workshops, 2019.

[2] K. Xu, G. Zhang, S. Liu, Q. Fan, M. Sun, H. Chen, P. Chen, Y. Wang, and X. Lin, “Adversarial t-shirt! evading person detectors in a physical world,” in Eur. Conf. Comput. Vis., 2020.

[3] Y.-C.-T. Hu, B.-H. Kung, D. S. Tan, J.-C. Chen, K.-L. Hua, and W.-H. Cheng, “Naturalistic physical adversarial patch for object detectors,” in Int. Conf. Comput. Vis., 2021.

[4] K. Eykholt, I. Evtimov, E. Fernandes, B. Li, A. Rahmati, C. Xiao, A. Prakash, T. Kohno, and D. Song, “Robust physical-world attacks on deep learning visual classification,” in IEEE Conf. Comput. Vis. Pattern Recog., 2018.

[5] Z. Hu, S. Huang, X. Zhu, F. Sun, B. Zhang, and X. Hu, “Adversarial texture for fooling person detectors in the physical world,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022.

[6] X. Zhu, X. Li, J. Li, Z. Wang, and X. Hu, “Fooling thermal infrared pedestrian detectors in real world using small bulbs,” in AAAI Conf. Art. Int., 2021.

[7] T. Kim, H. J. Lee, and Y. M. Ro, “Map: Multispectral adversarial patch to attack person detection,” in IEEE Int. Conf. Acous. Spe. Sig. Proc. (ICASSP). IEEE, 2022, pp. 4853–4857.

[8] X. Zhu, Z. Hu, S. Huang, J. Li, and X. Hu, “Infrared invisible clothing: Hiding from infrared detectors at multiple angles in realworld,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022.

[9] A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, “Synthesizing robust adversarial examples,” in Int. Conf. Mach. Learn., 2018.

[10] L. Huang, C. Gao, Y. Zhou, C. Xie, A. L. Yuille, C. Zou, and N. Liu, “Universal physical camouflage attacks on object detectors,” in IEEE Conf. Comput. Vis. Pattern Recog., 2020.

[11] N. Carlini and D. A. Wagner, “Towards evaluating the robustness of neural networks,” in IEEE Symp. Sec. Priv., 2017, pp. 39–57.

[12] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. J. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” in Int. Conf. Learn. Represent., 2014.

[13] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Int. Conf. Learn. Represent., 2015.

[14] A. Kurakin, I. J. Goodfellow, and S. Bengio, “Adversarial machine learning at scale,” in Int. Conf. Learn. Represent., 2017.

[15] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in Int. Conf. Learn. Represent., 2018.

[16] C. Xiao, B. Li, J. Zhu, W. He, M. Liu, and D. Song, “Generating adversarial examples with adversarial networks,” in Int. Joi. Conf. Art. Int., 2018.

[17] A. Liu, X. Liu, J. Fan, Y. Ma, A. Zhang, H. Xie, and D. Tao, “Perceptual-sensitive GAN for generating adversarial patches,” in AAAI Conf. Art. Int., 2019.

[18] M. Kristo, M. Ivasic-Kos, and M. Pobar, “Thermal object detection in difficult weather conditions using YOLO,” in IEEE Access, vol. 8, pp. 125 459–125 476, 2020.

[19] W. Xu, D. Evans, and Y. Qi, “Feature squeezing: Detecting adversarial examples in deep neural networks,” in Net. Dist. Sys. Sec. Symp., 2018.

[20] Z. Wu, S.-N. Lim, L. S. Davis, and T. Goldstein, “Making an invisibility cloak: Real world adversarial attacks on object detectors,” in Europ. Conf. Comp. Vis., 2020.

[21] P. Chen, H. Zhang, Y. Sharma, J. Yi, and C. Hsieh, “ZOO: zeroth order optimization based black-box attacks to deep neural networks without training substitute models,” in ACM Work. Art. Intel. Sec., 2017, pp. 15–26.

[22] G. K. Dziugaite, Z. Ghahramani, and D. M. Roy, “A study of the effect of JPG compression on adversarial images,” in CoRR, vol. abs/1608.00853, 2016.

[23] Y. Zhang, P. H. Foroosh, and B. Gong, “Camou: Learning a vehicle camouflage for physical adversarial attack on object detections in the wild,” in Int. Conf. Learn. Represent., 2019.

[24] D. Deb, J. Zhang, and A. K. Jain, “Advfaces: Adversarial face synthesis,” in IEEE Int. Joint Conf. on Bio., 2020.

[25] R. Duan, X. Mao, A. K. Qin, Y. Chen, S. Ye, Y. He, and Y. Yang, “Adversarial laser beam: Effective physical-world attack to dnns in a blink,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021.

[26] J. Tu, M. Ren, S. Manivasagam, M. Liang, B. Yang, R. Du, F. Cheng, and R. Urtasun, “Physically realizable adversarial examples for lidar object detection,” in IEEE Conf. Comput. Vis. Pattern Recog., 2020.

[27] Y. Cao, C. Xiao, D. Yang, J. Fang, R. Yang, M. Liu, and B. Li, “Adversarial objects against lidar-based autonomous driving systems,” in CoRR, 2019.

[28] Y. Cao, N. Wang, C. Xiao, D. Yang, J. Fang, R. Yang, Q. A. Chen, M. Liu, and B. Li, “Invisible for both camera and lidar: Security of multi-sensor fusion based perception in autonomous driving under physical-world attacks,” in IEEE Symp. Sec. Priv., 2021.

[29] S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “Deepfool: a simple and accurate method to fool deep neural networks,” in IEEE Conf. Comput. Vis. Pattern Recog., 2016.

[30] Q. Fan, L. Zhang, H. Xing, H. Wang, and X. Ji, “Microwave absorption and infrared stealth performance of reduced graphene oxide-wrapped al flake,” in Journ. Mater. Scien.: Mater. Elect., vol. 31, no. 4, 2020.

[31] L. Shang, Y. Lyu, and W. Han, “Microstructure and thermal insulation property of silica composite aerogel,” in Materials, vol. 12, no. 6, 2019.

[32] W. Wang, Z. Tong, R. Li, D. Su, and H. Ji, “Polysiloxane bonded silica aerogel with enhanced thermal insulation and strength,” in Materials, vol. 14, no. 8, 2021.

[33] F. Peng, Y. Jiang, J. Feng, H. Cai, J. Feng, and L. Li, “Thermally insulating, fiber-reinforced alumina–silica aerogel composites with ultra-low shrinkage up to 1500° c,” in Chem. Engin. Journ., vol. 411, 2021.

[34] FLIR, “Free flir thermal dataset for algorithm training,” [EB/OL], https://www.flir.com/oem/adas/adas-dataset-form/ Accessed Nov. 12, 2021.

[35] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: towards real-time object detection with region proposal networks,” in IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, 2016.

[36] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “Ssd: Single shot multibox detector,” in Eur. Conf. Comput. Vis., 2016.

[37] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask r-cnn,” in´ Int. Conf. Comput. Vis., 2017.

[38] J. Pang, K. Chen, J. Shi, H. Feng, W. Ouyang, and D. Lin, “Libra rcnn: Towards balanced learning for object detection,” in IEEE Conf. Comput. Vis. Pattern Recog., 2019.

[39] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in Europ. Conf. on Comp. Vis., 2020.

[40] ultralytics, “Yolov5,” [EB/OL], https://github.com/ultralytics/ yolov5 Accessed Nov. 21, 2021.

[41] T. Bai, J. Luo, and J. Zhao, “Inconspicuous adversarial patches for fooling image recognition systems on mobile devices,” in IEEE Int. Thing. Journ., 2021.

[42] A. Du, B. Chen, T.-J. Chin, Y. W. Law, M. Sasdelli, R. Rajasegaran, and D. Campbell, “Physical adversarial attacks on an aerial imagery object detector,” in IEEE Wint. Conf. Appl. Comp. Vis., 2022.

[43] C. Xiang, C. R. Qi, and B. Li, “Generating 3d adversarial point clouds,” in IEEE Conf. Comput. Vis. Pattern Recog., 2019.

[44] H.-T. D. Liu, M. Tao, C.-L. Li, D. Nowrouzezahrai, and A. Jacobson, “Beyond pixel norm-balls: Parametric adversaries using an analytically differentiable renderer,” in CoRR, 2018.

[45] C. Xiao, D. Yang, B. Li, J. Deng, and M. Liu, “Meshadv: Adversarial meshes for visual recognition,” in IEEE Conf. Comput. Vis. Pattern Recog., 2019.

[46] K.-H. Wu, W.-C. Huang, J.-C. Wang, and W.-C. Hung, “Infrared stealth and microwave absorption properties of reduced graphene oxide functionalized with fe3o4,” in Mater. Scien. Engin.: B, vol. 276, p. 115575, 2022.

[47] Y. Yang, S. Tan, G. Fang, Z. Yang, Y. Li, and C. Yang, “The compatible performance of three-dimensional sio2–zno amorphous photonic crystals in adjustable structural color and low infrared emissivity,” in Optic. Mater., vol. 107, p. 110105, 2020.

[48] Y. Wang, J. Zhou, T. Chen, S. Liu, S. Chang, C. Bajaj, and Z. Wang, “Can 3d adversarial logos cloak humans?” in CoRR, 2020.

[49] Z. Cai and N. Vasconcelos, “Cascade r-cnn: high quality object detection and instance segmentation,” in IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 5, pp. 1483–1498, 2019.

[50] SciPy, “Scipy,” [EB/OL], https://github.com/scipy/scipy Accessed Nov. 21, 2021.

[51] N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson, and G. Gkioxari, “Accelerating 3d deep learning with pytorch3d,” in CoRR, 2020.

[52] J. Liu, A. Levine, C. P. Lau, R. Chellappa, and S. Feizi, “Segment and complete: Defending object detectors against adversarial patch attacks with robust patch detection,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022.

[53] A. Agarwal, M. Vatsa, R. Singh, and N. Ratha, “Cognitive data augmentation for adversarial defense via pixel masking,” in Pat. Recog. Let., vol. 146, pp. 244–251, 2021.

[54] C. Guo, M. Rana, M. Cisse, and L. van der Maaten, “Countering ´ adversarial images using input transformations,” in Int. Conf. Learn. Represent., 2018.

[55] M.-I. Nicolae, M. Sinn, M. N. Tran, B. Buesser, A. Rawat, M. Wistuba, V. Zantedeschi, N. Baracaldo, B. Chen, H. Ludwig, I. Molloy, and B. Edwards, “Adversarial robustness toolbox v1.2.0,” in CoRR, 2018.

[56] L. A. Gatys, A. S. Ecker, and M. Bethge, “Image style transfer using convolutional neural networks,” in IEEE Conf. Comput. Vis. Pattern Recog., 2016, pp. 2414–2423.

[57] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-toimage translation using cycle-consistent adversarial networks,” in IEEE Int. Conf. Comp. Vis., 2017.

[58] Y. Choi, N. Kim, S. Hwang, K. Park, J. S. Yoon, K. An, and I. S. Kweon, “Kaist multi-spectral day/night data set for autonomous and assisted driving,” in IEEE Trans. Intel. Transp. Sys., vol. 19, no. 3, pp. 934–948, 2018.

[59] FLIR, “Flir adas v2 dataset,” [EB/OL], https://www.flir.com/ oem/adas/adas-dataset-form/ Accessed June. 30, 2024.

[60] S. Ren, K. He, R. B. Girshick, and J. Sun, “Faster R-CNN: towards real-time object detection with region proposal networks,” in IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, pp. 1137–1149, 2017.

[61] C.-Y. Wang, I.-H. Yeh, and H.-Y. M. Liao, “Yolov9: Learning what you want to learn using programmable gradient information,” in CoRR, 2024.

[62] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L. M. Ni, and H.- Y. Shum, “Dino: Detr with improved denoising anchor boxes for end-to-end object detection,” in CoRR, 2022.

[63] S. Zhang, X. Wang, J. Wang, J. Pang, C. Lyu, W. Zhang, P. Luo, and K. Chen, “Dense distinct query for end-to-end object detection,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023.

[64] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023.

[65] M. Sohan, T. Sai Ram, R. Reddy, and C. Venkata, “A review on yolov8 and its advancements,” in Int. Conf. Dat. Intel. Cog. Inform., 2024.

[66] R. Khanam and M. Hussain, “Yolov11: An overview of the key architectural enhancements,” arXiv preprint arXiv:2410.17725, 2024.

[67] S. Liu, F. Li, H. Zhang, X. Yang, X. Qi, H. Su, J. Zhu, and L. Zhang, “DAB-DETR: Dynamic anchor boxes are better queries for DETR,” in International Conference on Learning Representations, 2022.

[68] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable detr: Deformable transformers for end-to-end object detection,” in CoRR, 2020.

[69] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy,

and D. Lin, “MMDetection: Open mmlab detection toolbox and benchmark,” in CoRR, 2019.

[70] S. Bai, Y. Li, Y. Zhou, Q. Li, and P. H. Torr, “Adversarial metric attack and defense for person re-identification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 6, pp. 2119– 2126, 2020.

[71] Z. Yuan, J. Zhang, Z. Jiang, L. Li, and S. Shan, “Adaptive perturbation for adversarial attack,” IEEE Trans. Pattern Anal. Mach. Intell., 2024.

[72] F. Yin, Y. Zhang, B. Wu, Y. Feng, J. Zhang, Y. Fan, and Y. Yang, “Generalizable black-box adversarial attack with meta learning, IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 3, pp. 1804–1818, 2023.

[73] Y. Shi, Y. Han, Q. Hu, Y. Yang, and Q. Tian, “Query-efficient blackbox adversarial attack with customized iteration and sampling,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 2, pp. 2226–2245, 2022.

[74] X. Wei, Y. Guo, J. Yu, and B. Zhang, “Simultaneously optimizing perturbations and positions for black-box adversarial patch attacks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 7, pp. 9041–9054, 2022.

[75] X. Wei, Y. Guo, and J. Yu, “Adversarial sticker: A stealthy attack method in the physical world,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 3, pp. 2711–2725, 2022.

[76] H. Wei, Z. Wang, X. Jia, Y. Zheng, H. Tang, S. Satoh, and Z. Wang, “Hotcold block: Fooling thermal infrared detectors with a novel wearable design,” in AAAI Conf. Art. Intel., 2023.

[77] X. Wei, J. Yu, and Y. Huang, “Physically adversarial infrared patches with learnable shapes and locations,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023.

[78] X. Wei, Y. Huang, Y. Sun, and J. Yu, “Unified adversarial patch for visible-infrared cross-modal attacks in the physical world,” IEEE Trans. Pattern Anal. Mach. Intell., 2023.

[79] X. Zhu, Z. Hu, S. Huang, J. Li, X. Hu, and Z. Wang, “Hiding from infrared detectors in real world with adversarial clothes,” Applied Intelligence, vol. 53, no. 23, pp. 29 537–29 555, 2023.

[80] C. Hu, W. Shi, T. Jiang, W. Yao, L. Tian, X. Chen, J. Zhou, and W. Li, “Adversarial infrared blocks: A multi-view black-box attack to thermal infrared detectors in physical world,” Neural Networks, vol. 175, p. 106310, 2024.

[81] C. Hu, W. Shi, W. Yao, T. Jiang, L. Tian, X. Chen, and W. Li, “Adversarial infrared curves: An attack on infrared pedestrian detectors in the physical world,” Neural networks, vol. 178, p. 106459, 2024.

[82] K. Tiliwalidi, C. Hu, G. Lu, M. Jia, and W. Shi, “Advgrid: A multi-view black-box attack on infrared pedestrian detectors in the physical world,” Applied Soft Computing, vol. 174, p. 112981, 2025.

[83] T. Kim, Y. Yu, and Y. M. Ro, “Multispectral invisible coating: Laminated visible-thermal physical attack against multispectral object detectors using transparent low-e films,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 1, 2023, pp. 1151–1159.

[84] D. Wang, T. Jiang, J. Sun, W. Zhou, Z. Gong, X. Zhang, W. Yao, and X. Chen, “Fca: Learning a 3d full-coverage vehicle camouflage for multi-view physical adversarial attack,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 2, 2022, pp. 2414–2422.

[85] N. Suryanto, Y. Kim, H. Kang, H. T. Larasati, Y. Yun, T.-T.-H. Le, H. Yang, S.-Y. Oh, and H. Kim, “Dta: Physical camouflage attacks using differentiable transformation network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 15 305–15 314.

[86] J. Wang, A. Liu, Z. Yin, S. Liu, S. Tang, and X. Liu, “Dual attention suppression attack: Generate adversarial camouflage in physical world,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8565–8574.

[87] Y. Li, W. Tan, T. Wang, X. Liang, and Q. Pan, “Flexible physical camouflage generation based on a differential approach,” arXiv preprint arXiv:2402.13575, 2024.

[88] Y. Cao, C. Xiao, B. Cyr, Y. Zhou, W. Park, S. Rampazzi, Q. A. Chen, K. Fu, and Z. M. Mao, “Adversarial sensor attack on lidar-based perception in autonomous driving,” in Proceedings of the 2019 ACM SIGSAC conference on computer and communications security, 2019, pp. 2267–2281.

[89] Z. Jin, X. Ji, Y. Cheng, B. Yang, C. Yan, and W. Xu, “Pla-lidar: Physical laser attacks against lidar-based 3d object detection in autonomous vehicle,” in 2023 IEEE Symposium on Security and Privacy (SP). IEEE, 2023, pp. 1822–1839.

[90] Y. Cao, N. Wang, C. Xiao, D. Yang, J. Fang, R. Yang, Q. A. Chen, M. Liu, and B. Li, “Invisible for both camera and lidar: Security of multi-sensor fusion based perception in autonomous driving under physical-world attacks,” in 2021 IEEE symposium on security and privacy (SP). IEEE, 2021, pp. 176–194.

[91] L. Jing, R. Wang, W. Ren, X. Dong, and C. Zou, “Pad: Patchagnostic defense against adversarial patch attacks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 472–24 481.

[92] X. Wei, C. Kang, Y. Dong, Z. Wang, S. Ruan, Y. Chen, and H. Su, “Diffender: Diffusion-based adversarial defense against patch attacks,” in Eur. Conf. Comput. Vis., 2024.

![](images/2eb00f2fb55b9f7579d7bf24e6487d0ff868fe2aee7c7c803eb70c859420592a.jpg)

![](images/2db2864fdeebb6d766644959ee50611285061bd7b8f49403001b11926a6787ef.jpg)  
Xiaopei Zhu received his Ph.D. degree in Tsinghua University, Beijing, China, in 2024. He is now a postdoctoral fellow in Tsinghua University, working with Prof. Jun Zhu. His current research interests include computer vision, adversarial example, generative model, 3D modeling, AI4sicence, etc. He has published several papers in CVPR, AAAI, ACM MM, etc.

with prestigious conferences, including ICML, NeurIPS, ICLR, IJCAI and AAAI. He was selected as “AI’s 10 to Watch” by IEEE Intelligent Systems. He is a fellow of AAAI, and an associate editor-in-chief of IEEE Transactions on Pattern Analysis and Machine Intelligence.

![](images/d1ac23cf787cde56b2d39af998e8d5ada3fd7087df98bd64ed4635cc424573bb.jpg)

Siyuan Huang received a B.E. degree in Software Engineering from the Wuhan University of Technology, Wuhan, China, in 2018, and M.S. in Computer Science from the George Washington University, Washington, DC, US, in 2020. He is currently a Ph.D. student at the Department of Electrical and Computing Engineering, Johns Hopkins University, Baltimore, US. His current research interest is computer vision. Previously he was a Graduate Research Assistant at Tsinghua Laboratory of Brain and Intelligence, Ts-

Zhanhao Hu is a postdoc in the Department of Electrical Engineering and Computer Sciences at the University of California, Berkeley. He received his Ph.D. in Computer Science and Technology from Tsinghua University in 2023 and his B.S. in Mathematics and Physics from Tsinghua University in 2017. His current research includes security issues in deep learning, especially in Computer Vision and Large Language Models.

Jun Zhu (Fellow, IEEE) received the BS and PhD degrees from the Department of Computer Science and Technology, Tsinghua University, where he is currently a Bosch AI professor. He was a postdoctoral fellow and adjunct faculty with the Machine Learning Department, Carnegie Mellon University. His research interest is primarily on developing machine learning methods to understand scientific and engineering data arising from various fields. He regularly serves as senior area chairs and area chairs

inghua University, and an Associate Researcher at Alternative Computing Group, National Institute of Standards and Technology.

![](images/0b48e55314f3fe4bf6c04052b213b6b8657f86fde6255b7f2ff9fceaea7bff4b.jpg)

Jianmin Li received the Ph.D. degree in computer applications from the Department of Computer Science and Technology, Tsinghua University, in 2003. He is currently an Associate Professor with the Department of Computer Science and Technology, Tsinghua University. His main research interests include image and video analysis, image and video retrieval, and machine learning. He has published over 80 journal and conference papers.

![](images/f4454d41f13521dfa5b331a2beaadae9ed17f6fac253cf4b9ef642004366f09a.jpg)

![](images/52a4dddcd87ace8c8c94f0fb89646805d1940975d97d88a526aa0c5fed535e23.jpg)

Xiaolin Hu (S’01, M’08, SM’13) received B.E. and M.E. degrees in automotive engineering from the Wuhan University of Technology, Wuhan, China, in 2001 and 2004, respectively, and a Ph.D. degree in automation and computeraided engineering from the Chinese University of Hong Kong, Hong Kong, in 2007. He is currently an Associate Professor at the Department of Computer Science and Technology, Tsinghua University, Beijing, China. His current research interests include deep learning and computational neuroscience. At present, he is an Associate Editor of IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), IEEE Transactions on Image Processing (TIP) and Cognitive Neurodynamics. Previously he was an Associate Editor of the IEEE Transactions on Neural Networks and Learning Systems.