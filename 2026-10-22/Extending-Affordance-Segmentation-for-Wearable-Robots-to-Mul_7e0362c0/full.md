# Extending Affordance Segmentation for Wearable Robots to Multi-Object Images

Simone Lugani, Matteo Montani, Paolo Gastaldo, Rodolfo Zunino, Edoardo Ragusa

DITEN, University of Genoa, Genoa, Italy; contact: simone.lugani@edu.unige.it

Abstract—This paper proposes a refinement to existing visionbased portable pipelines for the recognition of graspable object parts. Such pipelines can effectively support automatic grasp selection and preshaping in upper-limb prosthetic devices. However, realistic scenarios in which multiple objects are present still represent a challenge. The proposed method exploits a minor modification to the conventional hardware setup and a custom training approach to extend the applicability of visionbased pipelines. A laser pointer is introduced in the setup as an effective mechanism for object disambiguation. Accordingly, a suitable retraining of the deep convolutional neural networks supporting recognition of graspable parts is implemented. A mixed reality strategy allows the reuse of available high-quality segmentation masks to generate realistic images procedurally, extending the range of scenarios supported by the final device. Experimental results confirm that the proposed pipeline improves target disambiguation accuracy by up to 9% compared with existing solutions, while preserving the computational cost.

Keywords—Affordance segmentation, Mixed Reality, Efficient CNNs, Embedded Systems, Wearable robots, Grasping

## I. INTRODUCTION

Upper-limb prosthetics aim to restore motor function and improve the quality of life after amputation, but their effective control remains a challenge. Conventional myoelectric control is often sequential and cognitively demanding, especially for multi-degree-of-freedom prostheses, which limits usability and leads to high rejection rates [1]. Shared-autonomy paradigms, where the prosthesis exploits environmental sensing to assist the user during task execution [2], alleviate these issues.

Vision-based approaches support automatic grasp selection and preshaping [3], [4], but existing systems rely on the user’s ability to frame the target object [5]. This becomes problematic in realistic scenarios with multiple objects, leading to ambiguity in target selection.

We address this limitation by introducing a simple mechanism for object disambiguation in vision-based prosthetic control: a laser pointer integrated into the wearable setup. The pointer, activated via EMG signals, provides an explicit indication of the user’s intended target. This information is exploited by a lightweight affordance segmentation (AS) module implemented with Deep Neural Networks (DNNs), which identifies the object of interest and extracts its graspable regions, enabling reliable operation even in multi-object environments [6], [7].

The proposed approach preserves the efficiency required for embedded deployment, while improving the robustness of target disambiguation in semi-autonomous prosthesis control.

Experimental results demonstrate that the pointer enables accurate target selection and enhances system performance in challenging multi-object scenarios.

## II. BACKGROUND

The literature presents a wide range of prototypes and approaches based on teleceptive sensing devices [8], [9], including RGB cameras [10], radars [11], and depth sensors [5]. Data from these sensors are integrated into autonomous control pipelines to enhance user performance [5], [12]. Modern robotic arms leverage hardware accelerators to enable onboard data processing [8], [9], [13], opening new opportunities for semi-autonomous control [14], [15].

Affordance segmentation (AS) describes the possible interactions with different parts of an object and offers valuable inputs for control pipelines [16]–[18]. Several approaches to AS have been proposed in the literature [16], [17], [19]– [21]. However, hardware constraints and the involvement of a user in the control loop of wearable robots call for dedicated solutions. The method introduced in [6] proposed a simplified and portable version of basic AS, designed to benefit from human interaction. Subsequent works further improved this approach by integrating DNN modules [22], [23] and additional sensory inputs [24]. The research in [7] demonstrated that such networks can run in real-time on high-end MCUs while supporting semi-autonomous control pipelines.

The integration of depth sensors [3], [13] into DNN pipelines has been extensively explored [24]–[26], but it increases both computational cost and system complexity, due to modality fusion and calibration requirements [27].

Few works have addressed target-object identification in prosthetic shared-control pipelines. They infer the target by combining vision with user-related cues, such as prosthesis motion, depth-based reaching, or gaze and eye-hand coordination [5], [28], [29]. Although effective, they are not formulated for AS pipelines and often rely on implicit cues, accurate framing, additional sensors, or demanding perception modules. Thus, target disambiguation remains open for portable RGBbased AS pipelines, where the visual stream should support both target selection and graspable-part segmentation.

Hardware considerations distinguish AS implementation strategies into unconstrained, hardware-efficient, and tiny models [30]. These groups loosely map to hardware platforms with different computational capabilities. While most existing research focuses on unconstrained or tiny models, mid-range hardware-efficient solutions remain relatively unexplored. To the best of the authors’ knowledge, the works in [6], [23], [30] represent the only AS architectures specifically designed for embedded accelerators. These approaches assume a clear framing of a single object while, in this work, this limitation is addressed by introducing a direct target selection mechanism based on a laser pointer.

![](images/fb5dd4891704fb64c33ddfd178f1e4c87c77b0384e56f8fa1d40a367c37bf42e.jpg)  
Fig. 1: AS pipeline: the image from a hand-mounted camera is segmented to identify graspable parts; high-level descriptors are then extracted and used for intelligent prosthesis control.

## III. PROPOSAL

## A. Sensing system

In the proposed setup, AS assigns functional labels to specific object regions to identify the parts that support a given action (e.g., grasping). As shown in Figure 1, the AS module processes the RGB frame acquired from the hand-held camera to generate a segmented mask of the graspable parts, e.g., a handle of an object. It enables the system to focus on that part when estimating object properties and extracting high-level descriptors, which in turn leads to a proper configuration of the hand. Thus, the AS provides automatic segmentation into relevant object parts, without explicit user instruction.

The system is designed for a semi-autonomous control pipeline, where a user relies on EMG signals for volitional control of the robotic hand. In typical semi-autonomous procedures, the user points the hand toward the object and activates, with a gesture, a trigger that initiates automatic grasping or preshaping. After the trigger, a camera captures images of the scene, and AS processes the frames to identify the graspable parts. Then, the resulting AS mask can be used to extract geometric cues (Figure 1), such as an oriented bounding box on the graspable region. In some setups, a DNN automatically controls the hand actuators. In other setups, a suitable control logic exploits the output provided by the network for hand preshaping. In both cases, the DNN supporting AS should first identify the target object.

In the proposed approach, when the first trigger is sent, the system activates a laser pointer mounted on the hand and waits for a few seconds to allow the user to tune the framing, i.e., centering the pointer on the target object. Figure 2 shows one possible implementation of the proposed setup, where both the RGB camera and the laser pointer are mounted beneath the hand. The AS module uses the pointer as a reference and segments the object targeted by the user. This setup extends the system to scenarios like those shown in Figure 3, which are currently too challenging for state-of-the-art portable setups [30]. In addition, the pointer makes otherwise ambiguous target objects explicit.

![](images/741b824cca6efe122dbef3fe3a72087fb476fc08e98729c2ea544eb89ab8d4ef.jpg)  
Fig. 2: Proposed framework: a pointer attached near the camera adds a light spot on the image

![](images/ee61a2151b7ebd2f812a291f5895e432fc85353b04ff86519d0475bc3f33e6be.jpg)  
Fig. 3: Examples: IIT real scenes with synthetic laser annotations (I, II), and UMD mixed-reality scenes (III, IV).

## B. Training the AS module with a mixed reality approach

In the proposed pipeline, the AS module should be trained to focus only on the target object, namely the one pointed by the user. Hence, one cannot rely only on standard datasets for AS that do not include pointers on the images. In addition, most of the existing datasets for AS have been generated with cameras mounted on robots leading to different viewpoints. These limitations motivate the need for a mixed-reality approach, where suitable images are augmented to generate a more representative dataset. A real pointer produces a clear light spot surrounded by smooth intensity gradients. Though full realism would require additional precautions, simple synthetic patterns can approximate its appearance.

The most widely used dataset for AS is UMD. It contains a set of everyday objects framed from many angles. Each object is recorded under different orientations against a fixed blue background, and a corresponding segmentation mask is provided for each view. The labeling procedure produced good segmentation masks, making this dataset a standard benchmark; however, the fixed background and illumination limit the generalization capabilities of models trained on it. Using the object segmentation mask, we isolate it from the background with good precision and extract a set of object images from the original dataset. These images are further refined through post-processing based on the fixed background color, smoothing object borders. The segmented objects are then placed onto random realistic backgrounds to generate mixed-reality scenes, as shown in Fig. 3.

Another common dataset for AS is IIT-Aff (IIT). It provides rectangular bounding boxes and semantic segmentation for each object, and includes many multi-object scenes, as in the first and second images in Figure 3.

<table><tr><td rowspan=2 colspan=5>MODELSTesting Setup                                    P_GP</td></tr><tr><td rowspan=1 colspan=1>No_P [30]</td><td rowspan=1 colspan=1>P_OBJ</td><td rowspan=1 colspan=1>P_BG</td><td rowspan=1 colspan=1>P_GP</td></tr><tr><td rowspan=1 colspan=1>No_Pointer</td><td rowspan=1 colspan=1>95.69</td><td rowspan=1 colspan=1>95.79</td><td rowspan=1 colspan=1>95.57</td><td rowspan=1 colspan=1>91.74</td></tr><tr><td rowspan=1 colspan=1>P_Object</td><td rowspan=1 colspan=1>95.58</td><td rowspan=1 colspan=1>95.77</td><td rowspan=1 colspan=1>95.59</td><td rowspan=1 colspan=1>81.30</td></tr><tr><td rowspan=1 colspan=1>P_Background</td><td rowspan=1 colspan=1>95.55</td><td rowspan=1 colspan=1>95.77</td><td rowspan=1 colspan=1>95.59</td><td rowspan=1 colspan=1>85.71</td></tr><tr><td rowspan=1 colspan=1>P_Graspable</td><td rowspan=1 colspan=1>95.61</td><td rowspan=1 colspan=1>95.61</td><td rowspan=1 colspan=1>95.45</td><td rowspan=1 colspan=1>96.18</td></tr></table>

TABLE I: Performance of different models on foreground images for different placements of the pointer.

## IV. EXPERIMENTS

## A. Setup

The AS module uses a lightweight DNN previously embedded in AS pipelines [23]. It requires 638 MFLOPs and 3.6 M parameters, with an inference latency of 80 ms and a power consumption of 6 W on a Jetson Nano. It was trained and tested using a workstation with an NVIDIA GeForce RTX 2080 Ti with Tensorflow. All networks were trained for 100 epochs using learning rate scheduling and early stopping. Testing data were never involved in parameter or hyperparameter tuning. We used weighted cross-entropy with class weights inversely proportional to class frequencies. Geometric and color augmentation was applied throughout training.

## B. Single object images

The first setup, summarized in Table I, evaluates the pointer effect on the generalization performance over UMD foreground images, following [30]. Four models were trained: No P without any pointer; P OBJ, P BG, and P GP with the pointer added to the object, the background near the object, and the object graspable part, respectively. P BG assesses whether the model relies on the pointer even when it does not indicate the object. Each model was tested on the same-domain test set and on the three remaining domains. Thus, as per Table I, the No P model has been evaluated on a test set including only images without a pointer and on the other three test sets. In Table I, columns denote models and rows denote test datasets. Colored cells indicate matching training/test datasets. Each entry reports the average accuracy over the three classes.

In all cases, the models achieve high accuracy. The configuration without pointers scores an average accuracy of 95.69%. When the pointer is placed on the object or in the background, it is ignored by the network: the accuracy remains almost constant for all test sets. When the pointer is placed in the graspable part of the object in all the training images, it plays a role in the inference phase. When tested using the corresponding test set, it leads to a slight improvement in accuracy. In the other cases, the accuracy decreases, which confirms that the network relies on the pointer.

## C. Multiple object images

The second analysis in Table II considers images with multiple objects. The columns identify the training dataset. The rows, divided into groups using a color scheme, represent the testing set. SELF identifies that the network has been tested on the test set corresponding to the training set, i.e. data from the same dataset but that were not in the training set. The green rows remark two test sets originated with UMD foreground images with only one object. The first does not contain a pointer, the second does. The last three rows (yellow) remark three test sets with 3 objects per image. The test set named 3OB contains the pointer on the target object, 3OB-NO-P does not contain the pointer, and 3OB-RND has the pointer placed in a random position, instead of having it on the object.

<table><tr><td></td><td>MR</td><td>MR-IIT</td><td>MR-UMD</td><td>MR-IIT-UMD</td></tr><tr><td>SELF</td><td>92.38</td><td>93.36</td><td>91.98</td><td>94.29</td></tr><tr><td>No_P</td><td>87.34</td><td>86.50</td><td>95.23</td><td>95.52</td></tr><tr><td>P_OBJ</td><td>90.49</td><td>93.44</td><td>95.24</td><td>95.74</td></tr><tr><td>3OB</td><td>90.83</td><td>89.98</td><td>90.72</td><td>89.62</td></tr><tr><td>3OB-NO-P</td><td>83.53</td><td>80.61</td><td>83.47</td><td>80.12</td></tr><tr><td>3OB-RND</td><td>62.20</td><td>53.97</td><td>60.41</td><td>54.21</td></tr></table>

TABLE II: Performance on images with multiple objects

Four models have been trained, one for each column. The first one was trained with a dataset generated with the proposed approach (MR). The second column (MR-IIT) indicates a dataset composed of a mixture of the MR and IIT. Similarly, the third (MR-UMD) joined MR with UMD, and the last (MR-IIT-UMD) used all three datasets. For the case of UMD, only 20% of the dataset was included to maintain the focus of the networks on images with multiple objects. In all the training images, a pointer on the target object was included.

All models perform well on their respective test sets, with average accuracy above 90% for each model. On singleobject foreground images, the pointer improves performance, showing that models can exploit it. The yellow section further highlights its role: accuracy stays high with a pointer, drops by 7–9% without it, and decreases sharply when it is randomly placed, confirming that the network focuses on it.

## D. Pointer shape and color

The last analysis concerns the shape and color of the pointer. The virtual addition of the pointer can be significantly different from a real-world image. To evaluate this aspect, 4 versions of the virtual pointer have been tested after training the networks with the virtual pointer used in the previous setups.

Figure 4 reports the accuracy of different models for the different versions of the pointer. In the plot, Pointers denotes the virtual pointer used in the training, Red, Green, and Blue are virtual pointers with a single color. Finally, Star is a pointer roughly of the same size as the pointers used in the other images but with a specific pattern, i.e., a star. The testing set was always the set with 3 objects per image, and the pointer was placed on the target object.

The results indicate that the model is quite robust against different synthetic variations in the pointers’ color and shape. In all cases, the deterioration in this challenging setup is limited: the color of the training and testing pointer is different, but still, the network can recognize the target object. This robustness to variations in pointer appearance is likely supported by the use of color augmentation during training.

![](images/626e153f990c2c03c921e2afb1752614187476c4d381c568741a002983eafc34.jpg)  
Fig. 4: Effect of pointers’ shape and color

## V. CONCLUSIONS

This work introduces a laser pointer to extend portable AS pipelines for semi-autonomous prostheses to multi-object scenes. Through mixed-reality training, the system learns to identify the target object from the laser position. The results show that the model exploits the pointer to select the target object. The system proved robust to variations in pointer appearance, including changes in color and shape.

The proposal enables target disambiguation in portable vision-based prosthetic systems while preserving a lightweight pipeline, as the required change is limited to the addition of a small laser pointer. Current validation is limited to controlled and mixed-reality scenarios. In real deployments, laser visibility may be affected by ambient illumination, transparent or specular surfaces, and user-related factors. Future work should validate the approach in real operating conditions, as well as the usability and cognitive load associated with pointer-based target selection.

## REFERENCES

[1] S. Salminger, H. Stino, L. H. Pichler, C. Gstoettner, A. Sturma, J. A. Mayer, M. Szivak, and O. C. Aszmann, “Current rates of prosthetic usage in upper-limb amputees–have innovations had an impact on device acceptance?” Disability and Rehabilitation, vol. 44, no. 14, pp. 3708– 3713, 2022.

[2] W. Guo, W. Xu, Y. Zhao, X. Shi, X. Sheng, and X. Zhu, “Toward humanin-the-loop shared control for upper-limb prostheses: A systematic analysis of state-of-the-art technologies,” IEEE Transactions on Medical Robotics and Bionics, vol. 5, no. 3, pp. 563–579, 2023.

[3] N. E. Krausz and L. J. Hargrove, “A survey of teleceptive sensing for wearable assistive robotic devices,” Sensors, vol. 19, no. 23, p. 5238, 2019.

[4] J. Mouchoux, S. Carisi, S. Dosen, D. Farina, A. F. Schilling, and M. Markovic, “Artificial perception and semiautonomous control in myoelectric hand prostheses increases performance and decreases effort,” IEEE Transactions on Robotics, vol. 37, no. 4, pp. 1298–1312, 2021.

[5] M. N. Castro and S. Dosen, “Continuous semi-autonomous prosthesis control using a depth sensor on the hand,” Frontiers in Neurorobotics, vol. 16, 2022.

[6] E. Ragusa, C. Gianoglio, S. Dosen, and P. Gastaldo, “Hardware-aware affordance detection for application in portable embedded systems,” IEEE Access, vol. 9, pp. 123 178–123 193, 2021.

[7] E. Ragusa, S. Dosen, R. Zunino, and P. Gastaldo, “Affordance segmentation using tiny networks for sensing systems in wearable robotic devices,” IEEE Sensors Journal, 2023.

[8] F. Hundhausen, D. Megerle, and T. Asfour, “Resource-aware object classification and segmentation for semi-autonomous grasping with prosthetic hands,” in 2019 IEEE-RAS 19th International Conference on Humanoid Robots (Humanoids). IEEE, 2019, pp. 215–221.

[9] J. Starke, P. Weiner, M. Crell, and T. Asfour, “Semi-autonomous control of prosthetic hands based on multimodal sensing, human grasp demonstration and user intention,” Robotics and Autonomous Systems, vol. 154, p. 104123, 2022.

[10] P. Weiner, J. Starke, S. Rader, F. Hundhausen, and T. Asfour, “Designing prosthetic hands with embodied intelligence: The kit prosthetic hands,” Frontiers in Neurorobotics, vol. 16, 2022.

[11] E. Mastinu, A. Coletti, J. van den Berg, and C. Cipriani, “Explorations of autonomous prosthetic grasping via proximity vision and deep learning,” IEEE Transactions on Medical Robotics and Bionics, 2024.

[12] K. Rana, J. Abou-Chakra, S. Garg, R. Lee, I. Reid, and N. Suenderhauf, “Affordance-centric policy learning: Sample efficient and generalisable robot policy learning using affordance-centric task frames,” arXiv preprint arXiv:2410.12124, 2024.

[13] F. Chiariotti, P. Mamidanna, S. Suman, C. Stefanovi<sup>ˇ</sup> c, D. Farina,´ P. Popovski, and S. Dosen, “The future of bionic limbs: The untappedˇ synergy of signal processing, control, and wireless connectivity,” IEEE Signal Processing Magazine, vol. 41, no. 4, pp. 58–75, 2024.

[14] Z. Tang, L. Zhang, X. Chen, J. Ying, X. Wang, and H. Wang, “Wearable supernumerary robotic limb system using a hybrid control approach based on motor imagery and object detection,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 1298–1309, 2022.

[15] J. W. Sensinger and S. Dosen, “A review of sensory feedback in upperlimb prostheses from the perspective of human motor control,” Frontiers in Neuroscience, vol. 14, 2020.

[16] Z. Jiang, Y. Zhu, M. Svetlik, K. Fang, and Y. Zhu, “Synergies between affordance and geometry: 6-dof grasp detection via implicit representations,” arXiv preprint arXiv:2104.01542, 2021.

[17] Z. O. Khalifa and S. A. A. Shah, “Towards visual affordance learning: A benchmark for affordance segmentation and recognition,” arXiv preprint arXiv:2203.14092, 2022.

[18] W. Wang, H. Zhu, and M. H. Ang Jr, “Sgsin: Simultaneous grasp and suction inference network via attention-based affordance learning,” IEEE Transactions on Industrial Electronics, 2024.

[19] M. Hassanin, S. Khan, and M. Tahtali, “Visual affordance and function understanding: A survey,” ACM Computing Surveys (CSUR), vol. 54, no. 3, pp. 1–35, 2021.

[20] E. Corona, A. Pumarola, G. Alenya, F. Moreno-Noguer, and G. Rogez, “Ganhand: Predicting human grasp affordances in multi-object scenes,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5031–5041.

[21] <sup>˙</sup>I. Ozc¸il and A. B. Koku, “Affordance labeling and exploration: A<sup>¨</sup> manifold-based approach,” arXiv preprint arXiv:2407.15479, 2024.

[22] T. Apicella, A. Cavallaro, R. Berta, P. Gastaldo, F. Bellotti, and E. Ragusa, “An affordance detection pipeline for resource-constrained devices,” in 2021 28th IEEE International Conference on Electronics, Circuits, and Systems (ICECS). IEEE, 2021, pp. 1–6.

[23] S. Lugani, E. Ragusa, R. Zunino, and P. Gastaldo, “Lightweight neural networks for affordance segmentation: Enhancement of the decoder module,” in International Conference on Applications in Electronics Pervading Industry, Environment and Society. Springer, 2023, pp. 437– 443.

[24] E. Ragusa, M. P. Ghezzi, R. Zunino, and P. Gastaldo, “Affordance segmentation using rgb-d sensors for application in portable embedded systems,” in Applications in Electronics Pervading Industry, Environment and Society: APPLEPIES 2022. Springer, 2023, pp. 109–116.

[25] S. Deng, X. Xu, C. Wu, K. Chen, and K. Jia, “3d affordancenet: A benchmark for visual object affordance understanding,” in proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1778–1787.

[26] C. Gao, C. Liu, J. Shu, F. Liu, J. Liu, L. Yang, X. Gao, and D. Meng, “Are dense labels always necessary for 3d object detection from point cloud?” arXiv preprint arXiv:2403.02818, 2024.

[27] W. Zhou, B. Jian, M. Fang, X. Dong, Y. Liu, and Q. Jiang, “Dgpinetkd: Deep guided and progressive integration network with knowledge distillation for rgb-d indoor scene analysis,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[28] B. Zhong, H. Huang, and E. Lobaton, “Reliable vision-based grasping target recognition for upper limb prostheses,” IEEE Transactions on Cybernetics, 2020.

[29] M. Cognolato, M. Atzori, R. Gassert, and H. Muller, “Improving robotic¨ hand prosthesis control with eye tracking and computer vision: A multimodal approach based on the visuomotor behavior of grasping,” Frontiers in artificial intelligence, p. 199, 2022.

[30] E. Ragusa, G. P. Canuti, S. Lugani, R. Zunino, and P. Gastaldo, “Filling the pareto optimal front for affordance segmentation on embedded devices using rgb-d cameras,” IEEE Sensors Journal, 2025.