# Unlocking Motion in Expressions: Temporal Calibration for Referring Video Object Segmentation

Yiwen Jiang School of Computer Science & Technology Soochow University Suzhou, China ywjiang7@stu.suda.edu.cn

Ruixin Zhang School of Computer Science & Technology Soochow University Suzhou, China 20245227041@stu.suda.edu.cn

Zhengtong Zhu School of Computer Science & Technology Soochow University Suzhou, China 20245227002@stu.suda.edu.cn

Jiaqing Fan School of Computer Science & Technology Soochow University Suzhou, China jqfan@suda.edu.cn

## Abstract

Referring Video Object Segmentation (RVOS) aims to segment referred objects at the pixel level in video sequences based on natural language descriptions. Existing methods typically introduce motion information within a unified cross-modal temporal modeling framework, where language cues are used for target localization and segmentation. However, the dependency of expressions on motion semantics is not explicitly modeled, making it dificult to adaptively adjust the use of motion information according to dif ferent semantic requirements. To address these issues, we propose an Expression-driven Motion Calibration (EMC) framework for RVOS that explicitly unlocks and leverages the motion semantics within expressions. The proposed method extracts interpretable motion control signals from expressions via a Motion Signal Pro cessing (MSP) module, and employs a Motion Influence Calibration (MIC) module to adjust the contribution of motion cues during temporal decision making. In addition, a Semantic Temporal Stage Construction (STSC) module is introduced to build expression relevant temporal stages, providing a compact temporal candidate space for motion calibration. Through extensive evaluation on six standard benchmarks, including Ref-YouTubeVOS, Ref-DAVIS17, MeViS (valid/valid<sup>�</sup>), A2D-Sentences, and JHMDB-Sentences, the superiority of our method is validated. We will release the code on https://github.com/Jeven7/EMC.

## CCS Concepts

• Computing methodologies → Video segmentation.

## Keywords

Video Object Segmentation, Temporal Calibration

## ACM Reference Format:

![](images/989c0227d34d0267c8116bedd510e308c35523a9ca23c67a972007eb8e2337bf.jpg)  
Figure 1: Overview of EMC. (a) Existing RVOS methods typically apply motion modeling to entire video sequences, incorporating motion cues into expressions without explicit diferentiation. (b) Our method models expression-related motion importance as motion signals and calibrates motion efects within the local structured semantic temporal space.

Yiwen Jiang, Zhengtong Zhu, Ruixin Zhang, and Jiaqing Fan. 2026. Unlocking Motion in Expressions: Temporal Calibration for Referring Video Object Segmentation. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3767308.3835229

## 1 Introduction

Referring Video Object Segmentation (RVOS) aims to perform pixellevel segmentation of target objects within video sequences based on language descriptions[20, 35, 37]. Unlike semi-supervised video object segmentation, which relies on pixel-level annotations in the first frame[1, 6, 31], or unsupervised video object segmentation, which identifies salient motion regions without explicit object guidance[11, 18, 45], RVOS uses language as the means of target specification, requiring models to jointly possess accurate visual understanding and fine-grained semantic parsing capabilities in complex video scenarios. Meanwhile, recent advances in multimodal learning have promoted the alignment between visual and linguistic information, ranging from vision-language perception and multimodal reasoning to large-scale multimodal foundation models[13, 14, 24]. Owing to its practical value in applications such as video editing, human–computer interaction, and intelligent content analysis, this task has attracted increasing research attention in recent years. However, due to the high diversity and abstract nature of natural language expressions, together with the presence of complex object motions, occlusions, and appearance variations in videos, efectively aligning language semantics with temporal video information remains a core challenge in RVOS.

In recent years, with the introduction of Mevis[9], RVOS methods have gradually shifted their research focus toward motion expressions, attempting to further enhance model segmentation capabilities in complex motion scenarios. Existing RVOS approaches generally treat motion as an inherent temporal property of video data and incorporate it through unified temporal representations that encode object evolution across frames[3, 21, 30, 35, 36]. These methods incorporate language as a global semantic condition for target localization and segmentation. Recent work has begun to focus more directly on motion information itself, either through explicit modeling of motion-related language descriptions or by decoupling static appearance cues from motion cues in modeling[30, 41], to enhance segmentation performance in scenarios involving complex motion expressions. However, most of these methods generally uniformly employ motion cues without distinguishingly adjusting the level of involvement in motion information based on the importance of motion semantics in expressions. This limits the models adaptability to the semantic demands of diferent expressions.

To address these challenges, We propose EMC, an expressiondriven motion calibration framework(Figure 6) that modulates the influence of motion cues in temporal decision making by explicitly modeling the importance of motion semantics. Specifically, we design a Motion Signal Processing (MSP) module that analyzes motion-related semantic components in expressions and explicitly models the importance of motion semantics as an interpretable temporal control prior. MSP decomposes motion importance into two complementary aspects: whether motion cues are necessary under the given expression, and when necessary, how strongly they should influence temporal decision making process. Based on this expression-driven prior, we further introduce a Motion Influence Calibration (MIC) module, which adaptively calibrates the influence of motion cues during temporal decision making process, suppressing redundant motion noise in static expressions while enhancing discriminative motion cues in dynamic ones.

Additionally, when introducing expression-driven motion calibration mechanisms into RVOS, we face the challenge of determining an appropriate temporal influence space within video sequences. Directly adjusting motion influence at the per-frame level is computationally redundant and susceptible to noise frame interference. To address this, we propose the Semantic Temporal Stage Construction (STSC) module. By organizing videos into semantic temporal stages through text-guided temporal spectrum clustering, STSC reduces redundancy and improves eficiency while ensuring temporal coverage. This approach provides compact, expressionrelevant candidate temporal structures for subsequent motion influence calibration. Together with MSP and MIC, STSC enables expression-aware motion calibration by providing a structured temporal space for adaptive motion reasoning. Finally, to comprehensively validate its efectiveness, we evaluated our method on six RVOS benchmarks: Ref-Youtube-VOS[28], MeViS(valid/valid<sup>�</sup>)[9], Ref-DAVIS17[19], A2D-Sentences[15] and JHMDB-Sentences[15]. Experimental results show that our approach achieves state-of-theart performance across multiple metrics on diverse benchmarks. Overall, our contributions are summarized as follows:

• We propose EMC, a novel expression-driven motion calibration framework for RVOS. This framework extracts motion significance from expressions as control signals through a motion signal processing (MSP) module, guiding more accurate motion-aware temporal decisions making process.

• We introduce a Motion Influence Calibration (MIC) module, which adaptively regulates the relative importance and contribution of motion cues during temporal decision making utilizing the motion signals explicitly extracted from MSP.

• We design a Semantic Temporal Stage Construction (STSC) module, which organizes videos into expression-related temporal stages, thereby providing a concise and compact temporal structure for subsequent motion calibration.

• Our method achieves competitive result across six datasets, with consistent improvements over existing methods.

## 2 Related Work

## 2.1 Referring Video Object Segmentation

RVOS aims to segment target objects in videos based on language descriptions[23, 39, 47]. Early RVOS approaches primarily extended image referring segmentation to video scenarios[28]. With the introduction of DETR-based architectures, the query-based end-to-end paradigm became dominant in RVOS research[3, 21, 35], Language were explicitly incorporated as object queries to guide the spatiotemporal decoding process within unified models. This paradigm enables direct cross-modal alignment and simplifies the optimization pipeline. Over the past three years, RVOS research has shifted toward scalability, robustness, and deployment requirements in complex scenarios. A series of works[16, 34] explored online or long video settings, enhancing temporal consistency through mechanisms like query propagation or representation updates. These methods emphasize eficient temporal modeling under streaming or resource-constrained conditions. Concurrently, with the emergence of large scale benchmarks like MeViS[9], RVOS has been systematically studied under more challenging expressions and scenarios, further integrating with foundational models and large scale visual-language systems[8, 20, 27, 33, 38]. Such integration significantly improves generalization ability and semantic understanding across diverse visual contexts, benefiting from multimodal foundation models and their applications in visual understanding and video content creation [14, 44, 46].Unlike highly coupled end-to-end paradigms, Findtrack[7] proposes decoupling object recognition from temporal propagation. This approach demonstrates stronger robustness in referential ambiguity and complex scenarios, highlighting the potential advantages of structured modeling in RVOS.

## 2.2 Motion Expression Video Segmentation

Compared to traditional RVOS tasks, this task primarily segments objects based on their motion descriptions within the video. This shift places greater emphasis on capturing fine-grained temporal dynamics and motion-aware semantics. Early LMPM[10] explored the interaction between linguistic information and temporal visual features, enhancing cross-frame object understanding through language-guided spatio-temporal modeling. However, motion semantics remained unified with appearance information in the model. Subsequently, DsHmp[17] explicitly distinguished static perception from hierarchical motion perception, introducing a motion perception mechanism to better diferentiate objects with similar appearances but distinct motion patterns. LoSh[41], approaching from the linguistic side, noted that long term representations often contain motion or temporal descriptions. It efectively mitigated the model’s over reliance on motion semantics by jointly modeling long term and short term expressions. DMVS[12] further decoupled the understanding of motion expressions from video instance segmentation, modeling static and motion cues separately. This significantly improved segmentation robustness in expression scenarios dominated by motion semantics. Although the above methods incorporate motion expression modeling at diferent lev els, most still employ motion cues in a uniform manner. They fail to diferentiate modeling based on the semantic importance of motion within expressions, potentially resulting in limited adaptability in expression scenarios with varied semantic requirements in practice, highlighting the need for more adaptive modeling strategies.

## 3 Methodology

## 3.1 Overview

The overall architecture of our proposed method is illustrated in Figure 7, primarily consisting of three core modules. Given an input video $\mathcal { V } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ composed of� frames and a referring expression $E ,$ the goal of referring video object segmentation is to predict a sequence of segmentation masks $\{ \hat { Y } _ { t } \} _ { t = 1 } ^ { T }$ that localize the target object specified by in each frame. First, STSC comprehensively models the input video and expression across three levels, constructing expression-related semantic temporal stages. It extracts candidate key frames for subsequent decision making in motion calibration. Simultaneously, MSP leverages LLM to semantically analyze input expressions, assess the significance of motion cues within them, and extract explicit motion control signals, which are then passed to MIC. Subsequently, MIC applies diferentiated processing to static and dynamic expressions based on these control signals. By adjusting the participation level of motion saliency in the keyframe selection process, it calibrates the keyframe decisions for dynamic expressions. Finally, based on the segmentation results of the selected key frames, the model employs a bi-directional propagation strategy to generate a complete video segmentation mask.

## 3.2 Semantic Temporal Stage Construction

STSC models the semantic temporal structure of a video by jointly considering visual consistency, text-guided semantic relevance, and temporal proximity. By constructing a joint similarity matrix and applying spectral clustering, the video is partitioned into a set of semantic temporal stages, within which frames are both visually coherent and consistently aligned with the referring expression. Representative frames with the highest text relevance from each stage are then selected into a keyframe pool for subsequent process.

Text-Guided Consistency Similarity. We introduce a textguided consistency similarity to ensure that frames within the semantic temporal stage exhibit consistent relevance to the referring expression. Specifically, we compute an expression relevance score $s _ { i }$ for each frame based on semantic correspondence with expression and define the text-guided consistency similarity as:

$$
S _ { i j } ^ { \mathrm { t e x t } } = \exp ( - | s _ { i } - s _ { j } | ) ,\tag{1}
$$

where $s _ { i }$ denotes the expression relevance score of frame �. This formulation encourages frames with similar semantic responses to be grouped into the same temporal stage while suppressing semantically inconsistent frames during clustering.

Temporal Proximity Similarity. To encourage temporal continuity within semantic temporal stages, we introduce a temporal proximity similarity $S ^ { \mathrm { t i m e } }$ to model the temporal distance between frames. This similarity provides a local temporal smoothness constraint during stage construction, encouraging temporally adjacent frames to be grouped together while avoiding excessive connections between distant frames. Formally, given the timestamps �<sub>�</sub> and $t _ { j }$ of frames � and $j ,$ the similarity is defined as:

$$
S _ { i j } ^ { \mathrm { t i m e } } = e ^ { - \lambda | t _ { i } - t _ { j } | } ,\tag{2}
$$

where � is a decay factor controlling the impact of temporal distance. This formulation assigns higher similarity to temporally adjacent frames while reducing the influence of distant frame pairs, thereby preserving temporal coherence within semantic temporal stages.

Visual Similarity. We introduce a visual similarity $S ^ { \mathrm { v i s } }$ to measure appearance consistency between frames. Given corresponding visual features $v _ { i }$ and $v _ { j }$ extracted from frames � and �, respectively, the visual similarity is defined as the cosine similarity:

$$
S _ { i j } ^ { \mathrm { v i s } } = \frac { \boldsymbol { v } _ { i } ^ { \top } \boldsymbol { v } _ { j } } { \lVert \boldsymbol { v } _ { i } \rVert \lVert \boldsymbol { v } _ { j } \rVert } ,\tag{3}
$$

which provides a normalized measure of feature alignment and reduces the influence of feature scale variations. This similarity encourages visually consistent frames to be grouped into the same semantic temporal stage, providing a stable appearance constraint for reliable stage construction and improving robustness against transient motion and background noise in complex scenarios.

Temporal Stage Aggregation. Based on the three similarity terms, we jointly aggregate them to construct a joint similarity matrix that captures visual coherence, expression relevance consistency, and temporal proximity between frames.We first normalize the matrices of the three categories to obtain normalized similarity values $\widetilde { S } ^ { \mathrm { t e x t } } , \widetilde { S } ^ { \mathrm { t i m e } }$ , and $\widetilde { S } ^ { \mathrm { v i s } }$ , ensuring comparability among diferent complementary similarity cues during fusion. We then compute the overall similarity between frames � and � as a balanced weighted combination of the normalized similarities:

$$
S _ { i j } = \frac { w _ { t } \widetilde { S } _ { i j } ^ { \mathrm { t e x t } } + w _ { \tau } \widetilde { S } _ { i j } ^ { \mathrm { t i m e } } + w _ { v } \widetilde { S } _ { i j } ^ { \mathrm { v i s } } } { w _ { t } + w _ { \tau } + w _ { v } } ,\tag{4}
$$

where $w _ { t } , w _ { \tau }$ and $w _ { v }$ denote the weights for text-guided, temporal, and visual similarities, respectively. Based on the resulting joint similarity matrix �, we apply spectral clustering to partition the video into $n _ { c l u s t e r }$ semantic temporal stages.

![](images/6c6d8998b052461e52510cbb2da1bf2b33df5c489d10b4ad7caac79a6532b890.jpg)  
Figure 2: The architecture of EMC. First, STSC constructs expression-related temporal phases based on input video and expressions, defining a compact and eficient temporal influence space for subsequent adaptive temporal decision making process. Simultaneously, MSP models the importance of motion semantics within gesture expressions to extract explicit motion control signals. Finally, MIC calibrates the contribution of motion cues during the temporal decision making process. Through this approach, motion information can be utilized selectively and explicitly according to the semantic requirements of diferent expressions, thereby more efectively enhancing the robustness and flexibility of the RVOS system in complex scenarios.

For each temporal stage, representative key frames are selected according to their expression relevance scores and then collected into a keyframe pool K, which serves as the structured temporal candidate frame set for subsequent motion-aware calibration.

## 3.3 Motion Signal Processing

MSP models the importance of motion semantics in the referring expression at the expression level and represents it as a structured semantic signal, which characterizes the extent to which the current expression relies on motion cues for target grounding. Rather than modeling the specific content of motion semantics, MSP focuses primarily on the relative weight of motion cues within the overall expression semantics. Specifically, MSP captures motion importance from two complementary perspectives: it determines whether motion cues are required under the given expression and, if so, quantifies how strongly they should influence the subsequent temporal decision-making process. Unlike approaches that implicitly encode motion-related information into high-dimensional language representations, MSP explicitly formulates the latent motion importance in the expression as a low-dimensional and interpretable semantic prior. This design enables the dependency on motion cues from the language side to be more clearly separated and modeled.

MSP operates purely on the linguistic modality and leverages LLM to perform semantic analysis of the referring expression. Through language-level parsing, the model determines whether the expression involves explicit motion semantics associated with the target object and estimates their relative importance under the given expression. The output of MSP is defined as a structured and interpretable motion control signal, denoted as $\operatorname { M S P } ( E ) = \{ \mathcal { G } , \mathcal { M } \}$ where $\mathcal { G } \in \{ 0 , 1 \}$ indicates whether the expression contains motion semantics, and $M \in [ 0 , 1 ]$ quantifies the importance of motion semantics within the overall expression.

The resulting motion strength serves as an expression-conditioned semantic prior and is used as a gating factor in the subsequent MIC process, indicating the extent to which motion cues should participate in the scoring function. It does not represent the physical magnitude of motion in the video, nor does it directly produce motion saliency; instead, it provides a language-driven quantifica tion of motion importance, thereby modulating the contribution of the motion term during key frame scoring. To ensure semantic consistency, MSP focuses exclusively on motion performed by the referred object itself and ignores non-action factors such as appearance attributes and spatial descriptions; when no motion semantics are present in the expression, M is set to zero.

By explicitly modeling motion semantic importance as a structured low-dimensional signal, MSP efectively establishes a clear interface between language understanding and temporal decision making, providing an expression-driven semantic guidance prior for the subsequent adaptive motion calibration process.

## 3.4 Motion Influence Calibration

MIC module incorporates motion cues into the key frame scoring process as calibration factors under expression conditions, rather than directly involving them in decision making as a unified temporal signal. MIC first calculates a base reliability score for each candidate frame based on $E ,$ which serves as the primary criterion for key frame selection. Subsequently, MIC further calibrates this base score using local motion salience near the candidate frame as an auxiliary cue, with the calibration weight controlled by the motion importance signal output from MSP. MIC outputs key frame scores based on the candidate key frame set constructed by STSC under expression conditions, which are used for subsequent key frame selection and temporal propagation.

Base Reliability Scoring. Upon receiving the motion signal processed by the MPS, the MIC first calculates the base reliability score for candidate frames. The base reliability score for candidate frame � measures whether the frame can independently support target referencing under a given expression without relying on motion cue calibration. This score is composed by the prediction confidence of the target from single frame segmentation results and the semantic alignment between the predicted mask region and the expression. The base reliability score is calculated as follows:

$$
S _ { \mathrm { b a s e } } ( k ) = \beta S _ { \mathrm { s e g } } ( k ) + ( 1 - \beta ) S _ { \mathrm { a l i g n } } ( k ) ,\tag{5}
$$

where $S _ { \mathrm { s e g } } ( k )$ denotes the segmentation confidence of the predicted mask on frame $k ,$ and $S _ { \mathrm { a l i g n } } ( k )$ measures the semantic consistency between the mask region and the referring expression. The weighting factor $\beta$ balances visual reliability and language alignment. The result will be sent to the temporal decision making part.

Motion Calibration. Motion Calibration introduces motion cues as an expression-conditioned correction to the base reliability score. For each candidate frame $k ,$ we first estimate a local motion saliency score $S _ { \mathrm { { m o t i o n } } } ( k )$ to characterize the relative motion strength of the referred object in a short temporal neighborhood around �.Starting from the predicted object mask on frame �, the mask is propagated forward and backward within a local temporal window $[ k - \delta , k + \delta ]$ For each propagated frame �, we compute the centroid of the object mask and measure the displacement between adjacent frames. The average centroid displacement within the window is used to define the motion saliency score:

$$
S _ { \mathrm { m o t i o n } } ( k ) = \frac { 1 } { 2 \delta } \sum _ { t = k - \delta + 1 } ^ { k + \delta } \| \mathbf { c } _ { t } - \mathbf { c } _ { t - 1 } \| ,\tag{6}
$$

where $c _ { t }$ denotes the centroid of the object mask at frame �.

After obtaining local motion saliency scores, MIC modulates the base reliability scores under expression conditions using motion. It employs the motion importance signal M provided by MSP to control the degree to which saliency scores participate in this modification.The final score for candidate frame � is defined as:

$$
S _ { \mathrm { d y n a m i c } } ( k ) = S _ { \mathrm { b a s e } } ( k ) \cdot ( 1 + \alpha \cdot S _ { \mathrm { m o t i o n } } ( k ) \cdot { \mathcal M } ) ,\tag{7}
$$

where � controls the overall strength of motion calibration. This formulation ensures that motion cues act only as a corrective factor rather than a defining element.

Temporal Decision Making. The Temporal decision making phase distinguishes decision processes under diferent expressive contexts through an explicit gating mechanism. We utilize previously extracted motion signals to assign candidate frames to two independent scoring branches. When the expression’s semantics do not require motion cues, the model makes decisions using base score. When the expression contains motion semantics, the motionaware calibration branch is activated to further refine the base score.

The final scoring of candidate frames is formally defined as:

$$
S _ { \mathrm { f i n a l } } ( k ) = \left\{ \begin{array} { l l } { S _ { \mathrm { b a s e } } ( k ) , } & { \mathrm { i f } \mathcal { G } = 0 , } \\ { S _ { \mathrm { d y n a m i c } } ( k ) , } & { \mathrm { i f } \mathcal { G } = 1 , } \end{array} \right.\tag{8}
$$

After obtaining the final score for each candidate frame, the keyframe is selected by maximizing the score of candidates:

$$
k ^ { * } = \arg \operatorname* { m a x } _ { k \in \mathcal { K } } S _ { \mathrm { f i n a l } } ( k ) ,\tag{9}
$$

The selected keyframe $k ^ { * }$ is then treated as the initial anchor frame for video object segmentation. The object mask predicted on the anchor frame is propagated forward and backward in time using a bi-directional propagation strategy, producing temporally consistent segmentation masks for all frames in the video sequence.

## 4 Experiment

## 4.1 Experimental Setup

4.1.1 Datasets and Metrics. We evaluate EMC on six public RVOS benchmarks: Ref-YouTube-VOS[28], Ref-DAVIS17[19], MeViS[9], A2D-Sentences[15] and JHMDB-Sentences[15]. Ref-YouTube-VOS contains 3,978 videos with approximately 15K referring expressions. Ref-DAVIS17 includes 90 videos and around 1.5K expressions, while MeViS consists of about 2K videos annotated with 28K expressions. A2D-Sentences contains 3.7K videos with 6.6K expressions, and JHMDB-Sentences includes 928 videos with 928 expressions. For MeViS, we conduct experiments on both valid set and valid<sup>�</sup> set, where valid<sup>�</sup> is mainly used for ofline evaluation during training. For Ref-YouTube-VOS, Ref-DAVIS17, and MeViS, performance is evaluated using region similarity J, contour accuracy $\mathcal { F } _ { ; }$ , and their average J&F. For A2D-Sentences and JHMDB-Sentences, we employ overall IoU (oIoU) and mean IoU (mIoU) metrics.

4.1.2 Implementaion Details. We adopt the pretrained EVF-SAM[43] with a BEiT[2] pretrained backbone for single frame segmentation and confidence estimation, and use SAM2[26] for bi-directional temporal propagation. Image–text alignment scores are computed using Alpha-CLIP[29]. Visual features used for similarity computation are extracted with CLIP[25]. Motion-related semantic signals are extracted using Meta-Llama-3-8B-Instruct. For hyperparameter settings, we set $\lambda = 0 . 5 , w _ { t } = 1 , w _ { \tau } = 1 , w _ { v } = 1 , \beta = 0 . 5$ and � = 2. The motion calibration strength � and the number of stages $n _ { c l u s t e r }$ are respectively set to 0.2 and 5 on on Ref-YouTube-VOS and A2D-Sentences, 0.4 and 4 on Ref-DAVIS17 and JHMDB-Sentences, and 0.1 and 10 on MeViS, respectively. Our experiments are conducted on a single V100 GPU with 32GB of memory.

## 4.2 Quantitative Results

4.2.1 Results on Ref-Youtube-VOS. As shown in Table 11, EMC achieves the best overall performance on Ref-YouTube-VOS[28]. Specifically, EMC attains a J&F score of 74.7, outperforming the previous state-of-the-art MPG-SAM2[27] by 0.8% and the sencond ReferDINO[20] by 5.4%. Improvements are consistently observed on both $\mathcal { T }$ score and F score, across diferent evaluation settings, indicating that EMC benefits from more accurate mask localization as well as improved boundary quality. The primary improvement is attributed to STSC, where a more structured and compact temporal space enables the model to locate targets with greater accuracy.

Table 1: Performance comparison with previous methods on Ref-YouTube-VOS, Ref-DAVIS17 and MeViS. The highest result in each column will be marked in bold, and the second-highest result will be underlined.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Publication</td><td colspan="3">Ref-YouTube-VOS</td><td colspan="3">Ref-DAVIS17</td><td colspan="3">MeViS</td></tr><tr><td>J&amp;F</td><td>J</td><td>F</td><td>I&amp;F</td><td>J</td><td>F</td><td>T&amp;F</td><td>J</td><td>F</td></tr><tr><td>URVOS [28]</td><td>ECCV 2020</td><td>47.2</td><td>45.3</td><td>49.2</td><td>51.6</td><td>47.3</td><td>56.0</td><td>31.0</td><td>29.8</td><td>32.2</td></tr><tr><td>ReferFormer [35]</td><td>CVPR 2022</td><td>62.9</td><td>61.3</td><td>64.6</td><td>61.1</td><td>58.1</td><td>64.1</td><td>27.8</td><td>25.7</td><td>29.9</td></tr><tr><td>MTTR [3]</td><td>CVPR 2022</td><td>55.3</td><td>54.0</td><td>56.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HTML [16]</td><td>ICCV 2023</td><td>63.4</td><td>61.6</td><td>65.2</td><td>62.1</td><td>59.2</td><td>65.1</td><td></td><td></td><td></td></tr><tr><td>SgMg [22]</td><td>ICCV 2023</td><td>65.7</td><td>63.9</td><td>67.4</td><td>63.3</td><td>60.6</td><td>66.0</td><td></td><td></td><td></td></tr><tr><td>LMPM [9]</td><td>ICCV 2023</td><td>65.7</td><td>63.9</td><td>67.4</td><td>63.3</td><td>60.6</td><td>66.0</td><td>37.2</td><td>34.2</td><td>40.2</td></tr><tr><td>MUTR [39]</td><td>AAAI 2024</td><td>68.4</td><td>66.4</td><td>70.4</td><td>68.0</td><td>64.8</td><td>71.3</td><td></td><td></td><td></td></tr><tr><td>DsHmp [17]</td><td>CVPR 2024</td><td>67.1</td><td>65.0</td><td>69.1</td><td>64.9</td><td>61.7</td><td>68.1</td><td>46.4</td><td>43.0</td><td>49.8</td></tr><tr><td>DMVS [12]</td><td>CVPR 2025</td><td>64.3</td><td>62.4</td><td>66.2</td><td>65.2</td><td>62.2</td><td>68.2</td><td>48.6</td><td>44.2</td><td>52.9</td></tr><tr><td>SSA [23]</td><td>CVPR 2025</td><td>64.3</td><td>62.2</td><td>66.4</td><td>67.3</td><td>64.0</td><td>70.7</td><td>48.9</td><td>44.3</td><td>53.4</td></tr><tr><td>SAMWISE [8]</td><td>CVPR 2025</td><td>69.2</td><td>67.8</td><td>70.6</td><td>70.6</td><td>67.4</td><td>74.5</td><td>49.5</td><td>46.6</td><td>52.4</td></tr><tr><td>ReferDINO [20]</td><td>ICCV 2025</td><td>69.3</td><td>67.0</td><td>71.5</td><td>68.9</td><td>65.1</td><td>72.9</td><td>49.3</td><td>44.7</td><td>53.9</td></tr><tr><td>MPG-SAM 2 [27]</td><td>ICCV 2025</td><td>73.9</td><td>71.7</td><td>76.1</td><td>72.4</td><td>68.8</td><td>76.0</td><td>53.7</td><td>50.7</td><td>56.7</td></tr><tr><td>TQF[42]</td><td>ACM MM 2025</td><td>69.1</td><td>67.7</td><td>70.5</td><td>68.6</td><td>64.3</td><td>73.0</td><td>52.8</td><td>49.4</td><td>56.1</td></tr><tr><td>FlowRVS [32]</td><td>ICLR 2026</td><td>69.6</td><td>67.1</td><td>72.1</td><td>73.3</td><td>68.4</td><td>78.2</td><td>51.1</td><td>47.6</td><td>54.6</td></tr><tr><td>EMC</td><td>Ours</td><td>74.7</td><td>72.6</td><td>76.8</td><td>77.7</td><td>74.5</td><td>80.8</td><td>54.9</td><td>52.2</td><td>57.6</td></tr></table>

Table 2: Performance comparison with previous methods on the valid<sup>�</sup> set of MeViS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Publication</td><td colspan="3">MeViS (validu)</td></tr><tr><td>T&amp;F</td><td>J</td><td>F</td></tr><tr><td>LMPM [9]</td><td>ICCV 2023</td><td>40.2</td><td>36.5</td><td>43.9</td></tr><tr><td>DsHmp [17]</td><td>CVPR 2024</td><td>55.3</td><td>51.0</td><td>60.4</td></tr><tr><td>VISA-7B [38]</td><td>ECCV 2024</td><td>51.2</td><td>48.0</td><td>54.4</td></tr><tr><td>DMVS [12]</td><td>CVPR 2025</td><td>58.3</td><td>52.6</td><td>63.9</td></tr><tr><td>SSA [23]</td><td>CVPR 2025</td><td>57.8</td><td>52.3</td><td>63.1</td></tr><tr><td>EMC</td><td>Ours</td><td>62.5</td><td>59.6</td><td>65.4</td></tr></table>

4.2.2 Results on Ref-DAVIS17. On the Ref-DAVIS17[19] benchmark, EMC achieves a J&F score of 77.7, significantly outperforms existing state-of-the-art methods. Compared to the previously best performing FlowRVS[32], it achieved a 4.4% improvement, and compared to the second best MPG-SAM2, it achieved an 5.3% improvement. Ref-DAVIS17 is characterized by high quality annotations and relatively short video sequences, where fine-grained spatial accuracy and temporal consistency are particularly important. Under this setting, selectively activating motion calibration only when action semantics are present allows EMC to avoid unnecessary motion interference for expressions dominated by appearance or spatial cues, leading to more stable predictions overall.

4.2.3 Results on MeViS. We conduct experiments on both the valid and valid<sup>�</sup> sets of the MeViS dataset. On the MeViS valid<sup>�</sup> split, which is a smaller ofline evaluation set used during training, EMC achieves the best performance with a J&F score of 62.5. Compared with the previous best method DMVS[12], EMC yields a clear improvement of 4.2%. This improvement is reflected in both J and F. On the valid subset of MeViS dataset, EMC further achieves the best results among all comparison methods, with a J&F score of 54.9. Compared to the previous SOTA method MPG-SAM, EMC improved the overall metric by 1.2%; this improvement was reflected in both J and F, with a more pronounced increase in J and a stable gain in F. Furthermore, compared with other beyond the second best approach, EMC maintains a stable performance margin across diferent evaluation settings. ReferDINO achieves a J&F score of 49.3 on MeViS, while SSA[23] reports 48.9, and EMC outperforms them by approximately 5.6% and 6.0%, respectively. These results show that explicitly calibrating motion cues leads to more reliable segmentation under complex motion conditions.

Table 3: Performance comparison with previous methods on A2D-Sentences and JHMDB-Sentences.
<table><tr><td rowspan="2">Method</td><td colspan="2">A2D-Sentences</td><td colspan="2">JHMDB-Sentences</td></tr><tr><td>oIoU</td><td>mIoU</td><td>oIoU</td><td>mIoU</td></tr><tr><td>ReferFormer [35]</td><td>78.6</td><td>70.3</td><td>73</td><td>71.8</td></tr><tr><td>SgMg [22]</td><td>79.9</td><td>72</td><td>73.7</td><td>72.5</td></tr><tr><td>SOC[21]</td><td>80.7</td><td>72.5</td><td>73.6</td><td>72.3</td></tr><tr><td>WaveCL[5]</td><td>79.2</td><td>71.0</td><td>73.0</td><td>71.8</td></tr><tr><td>DAVLS [4]</td><td>80.9</td><td>72.4</td><td>73.6</td><td>72.5</td></tr><tr><td>EMC</td><td>81.7</td><td>72.6</td><td>73.8</td><td>73.0</td></tr></table>

4.2.4 Results on A2D-Sentences and JHMDB-Sentences. On the A2D-Sentences[15] and JHMDB-Sentences[15] benchmarks, EMC also demonstrates competitive performance, as shown in Table 3. On A2D-Sentences, EMC achieves an oIoU of 81.7 and an mIoU of 72.6, outperforming all previous methods. Compared with the strong baseline DAVLS, EMC improves oIoU by 0.8%. It also achieves the highest mIoU among all methods, slightly surpassing SOC indicating consistent improvements in both overall overlap and per-instance segmentation quality. On JHMDB-Sentences, EMC achieves the best performance in terms of mIoU, surpassing all existing methods, while maintaining competitive oIoU performance. Compared with the previous best method SgMg, EMC improves mIoU by 0.4%. Overall, the results on these two datasets further validate the generalization capability of EMC in practice. By explicitly calibrating motion influence according to expression semantics, the proposed method consistently improves segmentation quality across datasets with diferent motion characteristics.

Table 4: Ablation study of main components of EMC.
<table><tr><td rowspan="2">ID</td><td colspan="3">Components</td><td colspan="3">MeViS</td></tr><tr><td>STSC</td><td>MSP</td><td>MIC</td><td>T&amp;F</td><td>J</td><td>F</td></tr><tr><td>1</td><td>x</td><td>x</td><td>x</td><td>52.6</td><td>49.8</td><td>55.3</td></tr><tr><td>2</td><td>√</td><td>X</td><td>X</td><td>53.3</td><td>50.5</td><td>56.1</td></tr><tr><td>3</td><td>x</td><td>√</td><td>√</td><td>53.7</td><td>50.9</td><td>56.5</td></tr><tr><td>4</td><td></td><td></td><td></td><td>54.9</td><td>52.2</td><td>57.6</td></tr></table>

## 4.3 Ablation Study

4.3.1 Component Analysis. As shown in Table 4, we conduct a component analysis on MeViS to examine the contribution of each module in EMC, including STSC, MSP, and MIC. When STSC is disabled, the model will employ to uniform temporal sampling for keyframe selection. The baseline configuration without any components (ID 1) achieves a $\mathcal { T } \& \mathcal { F }$ score of 52.6. Enabling STSC alone (ID 2) improves performance to 53.3 J&F, indicating that replacing uniform sampling with expression-aware temporal stage construction provides a more efective temporal candidate space. When both MSP and MIC are enabled without STSC (ID 3), the performance further increases to 53.7 J&F. This result suggests that explicitly expression-level action modeling and motion calibration can still provide benefits under uniform sampling, but the improvement remains limited. Finally, combining all components (ID 4) yields the best performance, achieving 54.9 J&F, with consistent gains on both J and F. The additional improvement over ID 3 demonstrates that STSC and motion-aware calibration are strongly complemen tary mechanisms, and structured temporal stages are essential for fully exploiting expression-driven motion modeling on MeViS.

4.3.2 Hyperparameter Analysis. We analyze the influence of the motion calibration strength � on MeViS, as shown in Table 5. When � varies from 0.05 to 0.3, the performance remains stable, with J&F fluctuating within a narrow range. The best result is achieved at �=0.1, yielding a J&F score of 54.9. Smaller values of � slightly weaken the efect of motion calibration, while larger values do not provide further improvements and may introduce redundant motion influence. These results indicate that EMC is not highly sensitive to �, excessive or insuficient involvement of motion cues can lead to decreased performance, and a moderate calibration strength is suficient to achieve robust performance.

Table 5: Ablation study of hyperparameter �
<table><tr><td rowspan="2">α</td><td colspan="3">MeViS</td></tr><tr><td>T&amp;F</td><td>J</td><td>F</td></tr><tr><td>0.05</td><td>54.7</td><td>52.0</td><td>57.5</td></tr><tr><td>0.1</td><td>54.9</td><td>52.2</td><td>57.6</td></tr><tr><td>0.2</td><td>54.7</td><td>52.0</td><td>57.4</td></tr><tr><td>0.3</td><td>54.8</td><td>52.1</td><td>57.5</td></tr></table>

![](images/e5db25256677fe8bdec22775bb4cf767031723ab1d42f71c07182007442721ef.jpg)  
Figure 3: Ablation study on number of stages.

Figure 8 reports the efect of the number of semantic temporal stages $n _ { c l u s t e r }$ on Ref-YouTube-VOS, Ref-DAVIS17. On Ref-YouTube-VOS, performance improves as $n _ { c l u s t e r }$ increases from 3 to 5, reaching the best result at $n _ { c l u s t e r } = 5 ,$ and gradually decreases when more stages are introduced. A similar trend is observed on Ref-DAVIS17, where the optimal performance is obtained at $n _ { c l u s t e r } { = } 4$ . The performance initially increases with the number of semantic temporal stages, then declines. These results suggest that an intermediate number of temporal stages provides a good balance between temporal coverage and stage coherence, while excessive partitioning may lead to redundant candidates in stage construction process.

4.3.3 Similarity Components Analysis. Table 6 systematically analyzes the contribution of diferent similarity components in STSC on MeViS, including visual similarity, text-guided consistency, and temporal proximity similarity. When only visual similarity is used (ID 1), the model achieves a $\mathcal { T } \& \mathcal { F }$ score of 54.5, indicating that appearance consistency provides a reasonable baseline. Using only text-guided similarity (ID 2) leads to lower performance, suggesting that expression signals alone are insuficient. Introducing temporal proximity together with visual similarity (ID 3) yields comparable performance, showing that temporal locality helps stabilize similarity estimation. Combining cross-modal cues does not always bring improvements. The combination of visual and text similarities (ID 4) results in a performance drop to 53.7, indicating that cross-modal consistency without temporal constraints may introduce noise. When temporal proximity is incorporated with text similarity (ID 5), performance recovers to 54.5. Finally, integrating all three components (ID 6) achieves the best performance of 54.9, demonstrating that these cues jointly provide complementary benefits for constructing reliable semantic temporal stages.

Table 6: Ablation study of similarity components of STSC.
<table><tr><td rowspan="2">ID</td><td colspan="3">Components</td><td colspan="3">MeViS</td></tr><tr><td>Visual</td><td>Text</td><td>Time</td><td>T&amp;F</td><td>J</td><td>F</td></tr><tr><td>1</td><td>√</td><td>x</td><td>x</td><td>54.5</td><td>51.8</td><td>57.2</td></tr><tr><td>2</td><td>x</td><td>」</td><td>X</td><td>54.4</td><td>51.7</td><td>57.1</td></tr><tr><td>3</td><td>√</td><td>x</td><td>√</td><td>54.5</td><td>51.7</td><td>57.2</td></tr><tr><td>4</td><td>√</td><td>√</td><td>x</td><td>53.7</td><td>50.9</td><td>56.4</td></tr><tr><td>5</td><td>x</td><td>√</td><td>√</td><td>54.5</td><td>51.9</td><td>57.2</td></tr><tr><td>6</td><td></td><td></td><td></td><td>54.9</td><td>52.2</td><td>57.6</td></tr></table>

![](images/4aef41c35431af85614575f30baa18488c5db0de23bdb32a828a85d0bb081abb.jpg)  
Figure 4: Performance under static and dynamic expressions.

4.3.4 Behavior under Static and Dynamic Expressions. We categorize referring expressions into static and dynamic types according to the motion strength inferred by the Motion Signal Processing (MSP) module, and analyze expression-level J&F performance dis tributions under the two semantic conditions. As shown in Figure 9, on Ref-YouTube-VOS, the baseline model exhibits noticeable performance variability, particularly for dynamic expressions, with a significant number of low scoring cases. In contrast, after introducing EMC, the performance distributions for both static and dynamic expressions become more concentrated, accompanied by an elevated lower bound and reduced dispersion, indicating improved stability across expressions. On the more challenging MeViS dataset, where dynamic expressions involve complex motion patterns often with intricate temporal dependencies, the baseline method sufers from severe degradation under dynamic conditions, characterized by a pronounced long tail of low performance samples. EMC mitigates this issue by lifting the lower tail of the distribution and narrowing the gap between static and dynamic expressions. These observations demonstrate that explicitly calibrating motion sensitivity based on expression semantics leads to more robust and consistent segmentation behavior across diferent expression types.

## 4.4 Qualitative Results

Figure 5 presents qualitative results of our method on the Ref-YouTube-VOS and MeViS datasets. As shown in Figure 5(a), our method accurately segments dynamic targets on Ref-YouTube-VOS by leveraging motion cues based on expression semantics, demonstrating the efectiveness of motion influence calibration. Specifically, in scenarios where multiple objects share similar appearance, our model is able to precisely distinguish the referred target by jointly modeling spatial position and motion-aware semantics. Additionally, even under partial occlusion or cluttered backgrounds, the predicted masks remain temporally stable and consistent across frames, highlighting the robustness of our temporal modeling. As illustrated in Figure 5(b), when the target appears only in the middle portion of the video and occupies a relatively small temporal span, our method still successfully localizes and segments the target. For example, in motion-centric expressions such as “man moving to left” or “the individual is moving in the leftward direction”, The model effectively captures fine-grained directional motion cues and focuses on the correct temporal segments. This indicates that STSC can construct expression-related semantic temporal stages, enabling the model to emphasize temporally relevant segments while efectively filtering out irrelevant portions ofthe video. These qualitative results demonstrate that expression-guided motion calibration improves target localization and temporal consistency efectively.

![](images/a0cc50513335b9649660f97cbe933d238c1a66d7b08d7cd36552202fe4aa169c.jpg)  
Figure 5: Qualitative results on the Ref-YouTube-VOS (a) and MeVis (b) datasets. Time steps are directed from left to right.

## 5 Conclusion

In this paper, we propose EMC, a novel expression-driven motion calibration framework. We introduce a motion signal processing module and a motion influence calibration module to perform adaptive forward adjustment of motion cues under diverse expression conditions, with explicit modeling of motion importance based on the expressions. Combined with a semantic temporal stage construction module, this establishes a more precise and efective temporal modeling space for temporal decision making in complex video scenarios. Experimental results on six mainstream datasets, including Ref-YouTube-VOS, Ref-DAVIS17, MeViS(valid/valid<sup>�</sup>), A2D-Sentences, and JHMDB-Sentences, demonstrate the superior performance and generalization ability of our proposed method across diverse datasets. In the future, we aim to further enhance the robustness and stability of the model across more challenging scenarios.

## Acknowledgments

This work is supported in part by the National Natural Science Foundation of China (No. 62176172, 61672364); partially by the National Key Research and Development Program of China (No.2018YFA0701700, No. 2018YFA0701701); partially by Basic Research Program of Jiangsu (No. BK20250789).

## References

[1] Jong-Hyeon Baek, Jiwon Oh, and Yeong Jun Koh. 2025. EVOLVE: Event-Guided Deformable Feature Transfer and Dual-Memory Refinement for Low-Light Video Object Segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 11273–11282.

[2] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. 2022. BEiT: BERT Pre-Training of Image Transformers. In International Conference on Learning Representations. https://openreview.net/forum?id=p-BhZSz59o4

[3] Adam Botach, Evgenii Zheltonozhskii, and Chaim Baskin. 2022. End-to-end referring video object segmentation with multimodal transformers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4985– 4995.

[4] Ying Cao, Yu Wang, Lijuan Sun, Xiaomei Zou, and Yuxiang Ma. 2026. Dynamic alignment of visual and language semantics for referring video object segmenta tion. Information Processing & Management 63, 6 (2026), 104712.

[5] Ran Chen, Taiyi Su, and Hanli Wang. 2025. WaveCL: Wavelet Calibration Learn ing for Referring Video Object Segmentation. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 3856–3864.

[6] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. 2024. Putting the object back into video object segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3151–3161.

[7] Suhwan Cho, Seunghoon Lee, Minhyeok Lee, Jungho Lee, and Sangyoun Lee. 2025. Find First, Track Next: Decoupling Identification and Propagation in Referring Video Object Segmentation. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision (ICCV) Workshops. 3814–3824.

[8] Claudia Cuttano, Gabriele Trivigno, Gabriele Rosi, Carlo Masone, and Giuseppe Averta. 2025. Samwise: Infusing wisdom in sam2 for text-driven video segmenta tion. In Proceedings of the Computer Vision and Pattern Recognition Conference. 3395–3405.

[9] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. 2023. MeViS: A large-scale benchmark for video segmentation with motion expressions. In Proceedings of the IEEE/CVF international conference on computer vision. 2694–2703.

[10] Zihan Ding, Tianrui Hui, Junshi Huang, Xiaoming Wei, Jizhong Han, and Si Liu. 2022. Language-bridged spatial-temporal interaction for referring video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4964–4973.

[11] Jiaqing Fan, Hanwen Qian, Mengjuan Jiang, and Fanzhang Li. 2025. PeriodVOS: Learning Periodic Patterns for Unsupervised Video Object Segmentation via Adaptive Contextual Coupling. In Proceedings of the 33rd ACM International Conference on Multimedia. 4455–4463.

[12] Hao Fang, Runmin Cong, Xiankai Lu, Xiaofei Zhou, Sam Kwong, and Wei Zhang. 2025. Decoupled Motion Expression Video Segmentation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 13821–13831.

[13] Jun Gao, Yongqi Li, Ziqiang Cao, and Wenjie Li. 2025. Interleaved-modal chain-of thought. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 19520–19529.

[14] Jun Gao, Qian Qiao, Tianxiang Wu, Zili Wang, Ziqiang Cao, and Wenjie Li. 2025. Aim: Let any multimodal large language models embrace eficient in-context learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 3077–3085.

[15] Kirill Gavrilyuk, Amir Ghodrati, Zhenyang Li, and Cees GM Snoek. 2018. Actor and action video segmentation from a sentence. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 5958–5966.

[16] Mingfei Han, Yali Wang, Zhihui Li, Lina Yao, Xiaojun Chang, and Yu Qiao. 2023. Html: Hybrid temporal-scale multimodal learning framework for referring video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13414–13423

[17] Shuting He and Henghui Ding. 2024. Decoupling static and hierarchical motion perception for referring video segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13332–13341.

[18] Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo Kanazawa, and Qianqian Wang. 2025. Segment Any Motion in Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3406–3416.

[19] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. 2019. Video Object Segmentation with Referring Expressions. In Computer Vision – ECCV 2018 Workshops. Springer International Publishing, Cham, 7–12.

[20] Tianming Liang, Kun-Yu Lin, Chaolei Tan, Jianguo Zhang, Wei-Shi Zheng, and Jian-Fang Hu. 2025. ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 20009–20019.

[21] Zhuoyan Luo, Yicheng Xiao, Yong Liu, Shuyan Li, Yitong Wang, Yansong Tang, Xiu Li, and Yujiu Yang. 2023. Soc: Semantic-assisted object cluster for referring video object segmentation. Advances in Neural Information Processing Systems 36 (2023), 26425–26437.

[22] Bo Miao, Mohammed Bennamoun, Yongsheng Gao, and Ajmal Mian. 2023. Spectrum-guided multi-granularity referring video object segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 920–930.

[23] Feiyu Pan, Hao Fang, Fangkai Li, Yanyu Xu, Yawei Li, Luca Benini, and Xiankai Lu. 2025. Semantic and sequential alignment for referring video object segmentation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 19067– 19076.

[24] Qian Qiao, Yu Xie, Jun Gao, Tianxiang Wu, Shaoyao Huang, Jiaqing Fan, Ziqiang Cao, Zili Wang, and Yue Zhang. 2024. DNTextSpotter: Arbitrary-shaped scene text spotting via improved denoising training. In Proceedings ofthe 32nd ACM International Conference on Multimedia. 10134–10143.

[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[26] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. 2025. SAM 2: Segment Anything in Images and Videos. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=Ha6RTeWMd0

[27] Fu Rong, Meng Lan, Qian Zhang, and Lefei Zhang. 2025. MPG-SAM 2: Adapting SAM 2 with Mask Priors and Global Context for Referring Video Object Seg mentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV). 23979–23989.

[28] Seonguk Seo, Joon-Young Lee, and Bohyung Han. 2020. URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark. In Computer Vision – ECCV 2020. Springer International Publishing, Cham, 208–223.

[29] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. 2024. Alpha-clip: A clip model focusing on wherever you want. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 13019–13029.

[30] Jiajin Tang, Ge Zheng, and Sibei Yang. 2023. Temporal collection and distri bution for referring video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15466–15476.

[31] Duolin Wang, Guanyu Xing, and Yanli Liu. 2025. FlowTrack: Integrating Adjacent-Frame Motion Tracking and Adaptive Prediction for Robust Semi-Supervised VOS. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 3468–3476.

[32] Zanyi Wang, Dengyang Jiang, Liuzhuozheng Li, Sizhe Dang, Chengzu Li, Harry Yang, Guang Dai, Mengmeng Wang, and Jingdong Wang. 2026. Deforming Videos to Masks: Flow Matching for Referring Video Segmentation. In The Fourteenth International Conference on Learning Representations. https://openreview.net/ forum?id=3KaIcArMAB

[33] Cong Wei, Yujie Zhong, Haoxian Tan, Yingsen Zeng, Yong Liu, Hongfa Wang, and Yujiu Yang. 2025. Instructseg: Unifying instructed visual segmentation with multi-modal large language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 20193–20203.

[34] Dongming Wu, Tiancai Wang, Yuang Zhang, Xiangyu Zhang, and Jianbing Shen. 2023. Onlinerefer: A simple online baseline for referring video object segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 2761–2770.

[35] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. 2022. Language as queries for referring video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4974–4984.

[36] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. 2023. Segment every reference object in spatial and temporal spaces. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2538–2550.

[37] Zishan Xu, Yifu Guo, Yuquan Lu, Fengyu Yang, Junxin Li, and Lihua Cai. 2026. Videoseg-r1: reasoning video object segmentation via reinforcement learning. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 40. 11496–11504.

[38] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. 2024. Visa: Reasoning video object segmentation via large language models. In European Conference on Computer Vision. Springer, 98–115.

[39] Shilin Yan, Renrui Zhang, Ziyu Guo, Wenchao Chen, Wei Zhang, Hongyang Li, Yu Qiao, Hao Dong, Zhongjiang He, and Peng Gao. 2024. Referred by multi-modality:

A unified temporal transformer for video object segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 6449–6457.

[40] Zhao Yang, Jiaqi Wang, Xubing Ye, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. 2024. Language-aware vision transformer for referring segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 47, 7 (2024), 5238–5255.

[41] Linfeng Yuan, Miaojing Shi, Zijie Yue, and Qijun Chen. 2024. Losh: Long-short text joint prediction network for referring video object segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14001– 14010.

[42] Dingwei Zhang, Dong Zhang, and Jinhui Tang. 2025. Mitigating Query Selection Bias in Referring Video Object Segmentation. In Proceedings of the 33rd ACM International Conference on Multimedia. 3952–3961.

[43] Yuxuan Zhang, Tianheng Cheng, Lianghui Zhu, Rui Hu, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. 2024. Evf-sam: Early vision-language fusion for text-prompted segment anything model. arXiv preprint arXiv:2406.20076 (2024).

[44] Yue Zhang, Zhizhou Zhong, Minhao Liu, Zhaokang Chen, Bin Wu, Yubin Zeng, Chao Zhan, Yingjie He, Junxin Huang, and Wenjiang Zhou. 2024. Musetalk: Realtime high-fidelity video dubbing via spatio-temporal sampling. arXiv preprint arXiv:2410.10122 (2024).

[45] Xiangyu Zheng, Songcheng He, Wanyun Li, Xiaoqiang Li, and Wei Zhang. 2025. Shallow Features Matter: Hierarchical Memory with Heterogeneous Interaction for Unsupervised Video Object Segmentation. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 796–805.

[46] Zhizhou Zhong, Yicheng Ji, Zhe Kong, Yiying Liu, Jiarui Wang, Jiasun Feng, Lupeng Liu, Xiangyi Wang, Yanjia Li, Yuqing She, et al. 2025. Anytalker: Scaling multi-person talking video generation with interactivity refinement. arXiv preprint arXiv:2511.23475 (2025).

[47] Zixin Zhu, Xuelu Feng, Dongdong Chen, Junsong Yuan, Chunming Qiao, and Gang Hua. 2024. Exploring pre-trained text-to-video difusion models for referring video object segmentation. In European Conference on Computer Vision. Springer, 452–469.

## A Additional Experimental Analysis

## A.1 Reliability Analysis of MSP

To evaluate the reliability of MSP, we randomly sample referring expressions from Ref-DAVIS17, Ref-YouTube-VOS, and MeViS, and manually annotate whether each expression contains motion semantics. The MSP outputs are then compared with human annotations using standard classification metrics. The quantitative results are summarized in Table 7. MSP achieves consistently high performance across all datasets, with accuracy ranging from 95% to 97%. This demonstrates that MSP can reliably distinguish dynamic expressions from static ones and extract meaningful motion-related signals from natural language. A closer look at the metrics reveals that MSP maintains a good balance between precision and recall. In particular, MSP achieves perfect recall on Ref-YouTube-VOS, indicating that motion-related expressions are rarely missed. On Ref-DAVIS17, the slightly lower precision suggests that MSP may occasionally over-detect motion in ambiguous or weak-motion ex pressions. Nevertheless, the overall performance remains stable across datasets. These results indicate that MSP provides robust and reliable motion signals, which form a solid foundation for subsequent motion calibration and temporal modeling in our framework. Furthermore, the reliability of MSP enables consistent motion cues for downstream modules, enhancing overall framework stability.

## A.2 Error Case Analysis of MSP

To further analyze the limitations of MSP, we examine its prediction errors from both quantitative and qualitative perspectives. We first analyze the distribution of error types across diferent datasets, as summarized in Table 8. The results show that over-detection is the dominant error type across all datasets, while weak and implicit motion errors mainly occur in MeViS, which contains more complex and dynamic expressions. To provide a more intuitive understanding, Table 9 presents representative failure cases covering diverse error patterns. As shown in Table 9, the errors of MSP mainly fall into three categories. First, MSP may fail to detect weak or subtle motion, such as “eating”, where motion exists but is not salient or visually prominent. Second, MSP struggles with implicit motion, where dynamic behavior is implied but not explicitly expressed. Third, MSP may over-detect motion when encountering action-related verbs or ambiguous semantics. These patterns are consistent with the quantitative results, where MSP achieves high recall but relatively lower precision on certain datasets, indicating a tendency toward over-detection. This behavior suggests that MSP prefers to capture potential motion cues rather than risk missing dynamic expressions. Such a design trade-of is beneficial in practice, as missing dynamic cues may lead to more severe errors in subsequent motion-aware temporal decision making and modeling.

## A.3 Distribution of Motion Strength

The distribution of extracted motion strength from MSP across diferent datasets is shown in Figure 6. Ref-YouTube-VOS and Ref-DAVIS17 exhibit a relatively high proportion of expressions assigned to the lowest motion strength level, indicating that a substantial portion of their referring expressions describe static targets. Meanwhile, both datasets retained a significant proportion of expressions associated with moderate to strong action intensity, reflecting a mixed semantic composition in terms of action intensity. In contrast, MeViS demonstrates a markedly diferent distribution pattern, where higher motion strength dominate the overall composition. Specifically, expressions with motion strength values of 0.6 and 0.8 account for the majority, suggesting that dynamic expressions are more prevalent in this dataset.

Table 7: Reliability evaluation of MSP.
<table><tr><td>Dataset</td><td>#Samples Acc (%) Prec (%) Rec (%) F1 (%)</td><td></td><td></td><td></td><td></td></tr><tr><td>Ref-DAVIS17</td><td>100</td><td>96.0</td><td>90.6</td><td>96.7</td><td>93.5</td></tr><tr><td>Ref-YouTube-VOS</td><td>150</td><td>96.7</td><td>92.3</td><td>100.0</td><td>96.0</td></tr><tr><td>MeViS</td><td>200</td><td>95.5</td><td>97.9</td><td>95.9</td><td>96.9</td></tr></table>

Table 8: Error type distribution across datasets.
<table><tr><td>Dataset</td><td>Over- detection</td><td>Weak motion</td><td>Implicit / Ambiguous</td><td>Total</td></tr><tr><td>Ref-DAVIS17</td><td>3</td><td>0</td><td>1</td><td>4</td></tr><tr><td>Ref-YouTube-VOS</td><td>5</td><td>0</td><td>0</td><td>5</td></tr><tr><td>MeViS</td><td>3</td><td>3</td><td>3</td><td>9</td></tr><tr><td>Total</td><td>11</td><td>3</td><td>4</td><td>18</td></tr></table>

![](images/b535f36fc1043addb4e2489ad0941da40767d963620be543c71861bcd5dd8513.jpg)  
Figure 6: Distribution of motion strengths across datasets.

## A.4 Additional Precision Comparisons

As shown in Table 10, we provide additional precision comparisons with previous methods on A2D-Sentences and JHMDB-Sentences under multiple IoU thresholds to evaluate the localization accuracy of diferent methods under varying levels of strictness. On A2D-Sentences, our method achieves the best performance across most thresholds, especially under higher IoU thresholds. EMC surpasses previous methods with clear margins at �@0.7, �@0.8, and �@0.9, demonstrating its superior capability in precise object localization. Notably, the improvement becomes more pronounced as the IoU threshold increases, indicating that EMC is more efective in capturing fine-grained spatial details and producing accurate segmentation boundaries. On JHMDB-Sentences, EMC consistently outperforms existing methods across all thresholds. The performance gains are particularly evident at stricter thresholds, further validating the robustness of our approach in scenarios requiring high localization precision. Overall, these results demonstrate that EMC not only improves overall precision but also shows stronger performance under high-IoU conditions, highlighting its advantage in accurate and robust referring video object segmentation.

Table 9: Representative error cases of MSP.
<table><tr><td>Type</td><td>Expression</td><td colspan="2">MSP PredictionHuman Annotation</td><td>Analysis</td></tr><tr><td>Weak motion</td><td>the rabbit that is eating</td><td>Static</td><td>Dynamic</td><td>subtle motion is overlooked</td></tr><tr><td>Implicit motion</td><td>the last sheep to appear</td><td>Static</td><td>Dynamic</td><td>motion is implied but not explicit</td></tr><tr><td></td><td>Over-detection a guitar being played by second man</td><td>Dynamic</td><td>Static</td><td>verb triggers false motion detection</td></tr><tr><td></td><td>Ambiguous action monkey with a monkey on its back</td><td>Dynamic</td><td>Static</td><td>does not necessarily imply motion</td></tr><tr><td>Semantic ambiguity</td><td>a cow is standing far behind</td><td>Dynamic</td><td>Static</td><td>misinterpreted as motion-related context</td></tr></table>

Table 10: Precision comparison with previous methods on A2D-Sentences and JHMDB-Sentences.
<table><tr><td colspan="5">A2D-Sentences</td></tr><tr><td rowspan="2">Method</td><td colspan="4">Precision</td></tr><tr><td>P@0.5</td><td>P@0.6</td><td>P@0.7 P@0.8</td><td>P@0.9</td></tr><tr><td>Gavrilyuk et al. [15]</td><td>47.5</td><td>34.7</td><td>21.1</td><td>8.0 0.2</td></tr><tr><td>MTTR [3]</td><td>75.4</td><td>71.2</td><td>63.8 48.5</td><td>16.9</td></tr><tr><td>ReferFormer [35]</td><td>83.1</td><td>80.4</td><td>74.1 57.9</td><td>21.2</td></tr><tr><td>SOC [21]</td><td>82.1</td><td>82.7</td><td>76.5 60.7</td><td>25.2</td></tr><tr><td>OnlineRefer [34]</td><td>83.1</td><td>80.2</td><td>73.4 56.8</td><td>21.7</td></tr><tr><td>HTML [16]</td><td>84.0</td><td>81.5</td><td>75.8 59.2</td><td>22.8</td></tr><tr><td>LAVT [40]</td><td>82.8</td><td>79.3</td><td>71.5 54.6</td><td>19.5</td></tr><tr><td>EMC</td><td>84.6</td><td>82.5</td><td>77.5 64.3</td><td>27.4</td></tr></table>

JHMDB-Sentences
<table><tr><td rowspan="2">Method</td><td colspan="5">Precision</td></tr><tr><td>P@0.5</td><td>P@0.6</td><td>P@0.7</td><td>P@0.8</td><td>P@0.9</td></tr><tr><td>Gavrilyuk et al. [15]</td><td>69.9</td><td>46.0</td><td>17.3</td><td>1.4</td><td>0.0</td></tr><tr><td>MTTR [3]</td><td>93.9</td><td>85.2</td><td>61.6</td><td>16.6</td><td>0.1</td></tr><tr><td>ReferFormer [35]</td><td>96.2</td><td>90.2</td><td>70.2</td><td>21.0</td><td>0.3</td></tr><tr><td>SOC [21]</td><td>96.9</td><td>91.4</td><td>71.1</td><td>21.3</td><td>0.1</td></tr><tr><td>OnlineRefer [34]</td><td>96.1</td><td>90.4</td><td>71.0</td><td>21.9</td><td>0.2</td></tr><tr><td>EMC</td><td>97.7</td><td>92.3</td><td>72.3</td><td>23.3</td><td>0.3</td></tr></table>

## A.5 Visualization of Semantic Temporal Stages

To qualitatively analyze the behavior of the Semantic Temporal Stage Construction (STSC) module, we visualize the constructed temporal stages on a representative example from the Ref-YouTube-VOS dataset. As shown in the Figure 7, STSC decomposes the video into multiple semantically coherent temporal stages according to the motion semantics implied in the referring expression. For the expression “a white parachute is opened in the sky and flying downward”, the video is automatically partitioned into five consecutive stages, where frames within each stage exhibit high visual and semantic consistency, while diferent stages correspond to distinct phases of the downward motion.This structured decomposition reflects the gradual evolution of motion dynamics over time. The red bounding boxes indicate the representative key frames selected for each stage, which efectively summarize the dominant visual and motion characteristics of each stage and provide reliable anchors for subsequent motion-aware calibration.

"a white parachute is opened in the sky and flying downward"  
![](images/388ccacb68a63261b8b3eaa6a63123e0ebcd59df584cb6b96992504bed8fc5bf.jpg)  
Figure 7: Visualization of Semantic Temporal Stages.

## A.6 Extra Ablation Study

A.6.1 Semantic Temporal Stages on MeViS. We further conduct an ablation study on MeViS to analyze the efect of the number of semantic temporal stages constructed by STSC. Figure 8 illustrates the performance variation as the number of temporal stages increases. As shown, segmentation performance exhibits a clear upward trend when increasing the number of semantic temporal stages, indicating that finer temporal partitioning enables better alignment with complex motion evolution on MeViS. The performance reaches its peak at a moderate stage number, after which further increasing the number of stages leads to performance saturation and slight degradation. This behavior suggests that while richer temporal decomposition is beneficial, excessively fine partitioning may weaken semantic coherence and introduce redundancy. These results clearly and empirically demonstrate that properly selecting the number of stages is critical for achieving a balance between temporal granularity and semantic consistency on MeViS.

A.6.2 Motion calibration strength on Ref-DAVIS17. As shown in table 11, the relatively slow performance improvement observed when increasing the motion calibration strength is closely related to the characteristics of Ref-DAVIS17, where many videos exhibit static or weakly dynamic motion patterns. As the calibration strength increases from 0.1 to 0.4, the J&F score shows a smooth but limited gain, rising from 77.56 to 77.65, with consistent trends in both $\mathcal { T }$ and F. Specifically, the incremental improvements remain marginal across all intervals, indicating a diminishing return as the calibration strength increases. This indicates that while motion calibration contributes positively, moderate calibration is generally suficient for Ref-DAVIS17 due to its predominantly stable motion content. Moreover, excessive calibration may overemphasize motion cues, yielding limited additional benefit under such scenarios.

![](images/bf612be4a133457e8f893e3ac20d14630ce8d2a93a970979bf85f00006aee2c4.jpg)  
Figure 8: Ablation study on number of stages on MeViS.

Table 11: Ablation study of motion calibration strength on Ref-DAVIS17.
<table><tr><td rowspan="2">α</td><td colspan="3">Ref-DAVIS17</td></tr><tr><td>T&amp;F</td><td>J</td><td>F</td></tr><tr><td>0.1</td><td>77.56</td><td>74.41</td><td>80.71</td></tr><tr><td>0.2</td><td>77.57</td><td>74.42</td><td>80.72</td></tr><tr><td>0.3</td><td>77.63</td><td>74.49</td><td>80.77</td></tr><tr><td>0.4</td><td>77.65</td><td>74.51</td><td>80.79</td></tr></table>

A.6.3 Sensitivity Analysis of�. We further investigate the sensitivity of the hyperparameter � on Ref-YouTube-VOS, Ref-DAVIS17, and MeViS. As shown in Table 12, the performance remains relatively stable when $\beta$ varies from 0.25 to 1.0, demonstrating that EMC is not sensitive to the specific choice of ${ \mathrm { ~ ~ } } \cdot \beta .$ In particular, � = 0.5 consistently achieves the best performance across all three datasets. In contrast, setting � = 0 leads to a noticeable performance drop, indicating the importance of motion calibration in temporal decision making. Based on these results, we set $\beta = 0 . 5$ as the default value in our experiments.

## A.7 Inference Eficiency

We evaluate the inference eficiency of EMC in terms of inference speed, runtime, and memory consumption. As shown in Table 13, MSP and STSC achieve 16.81 and 20.11 FPS, respectively, indicating that both preprocessing stages can be eficiently performed. Since their outputs are independent of the motion calibration process, they can be precomputed and cached before inference. With the outputs of MSP and STSC precomputed, the remaining EMC inference process achieves 7.34 FPS, with a runtime of 6.73 seconds per pair and 10.0 GB of memory usage. These results demonstrate that EMC maintains practical inference eficiency while enabling expression-driven motion calibration for temporal decision making.

Table 12: Sensitivity analysis of $\beta .$
<table><tr><td rowspan="2"> $\beta$ </td><td colspan="3">Ref-YouTube</td><td colspan="3">Ref-DAVIS17</td><td colspan="3">MeViS</td></tr><tr><td> $\mathcal { T } \& \mathcal { F }$ </td><td> $\mathcal { T }$ </td><td> $\mathcal { F }$ </td><td>J&amp;F</td><td> $\mathcal { T }$ </td><td> $\mathcal { F }$ </td><td>J&amp;F</td><td> $\mathcal { T }$ </td><td> $\mathcal { F }$ </td></tr><tr><td>0</td><td>71.8</td><td>69.8</td><td>73.9</td><td>75.5</td><td>71.7</td><td>79.3</td><td>52.2</td><td>49.3</td><td>55.1</td></tr><tr><td>0.25</td><td>74.4</td><td>72.4</td><td>76.3</td><td>77.5</td><td>74.3</td><td>80.7</td><td>54.8</td><td>52.1</td><td>57.5</td></tr><tr><td>0.5</td><td>74.7</td><td>72.6</td><td>76.8</td><td>77.7</td><td>74.5</td><td>80.8</td><td>54.9</td><td>52.2</td><td>57.6</td></tr><tr><td>0.75</td><td>74.2</td><td>72.3</td><td>76.1</td><td>77.1</td><td>73.9</td><td>80.3</td><td>54.5</td><td>51.8</td><td>57.2</td></tr><tr><td>1</td><td>74.1</td><td>72.2</td><td>76.0</td><td>77.1</td><td>74</td><td>80.2</td><td>54.4</td><td>51.7</td><td>57.1</td></tr></table>

Table 13: Inference eficiency of EMC.
<table><tr><td>Setting</td><td>FPS (frames/s)</td><td>Runtime (s/pair)</td><td>Memory (GB)</td></tr><tr><td>MSP</td><td>16.81</td><td>1.57</td><td>15.0</td></tr><tr><td>STSC</td><td>20.11</td><td>1.31</td><td>1.3</td></tr><tr><td>EMC</td><td>7.34</td><td>6.73</td><td>10.0</td></tr></table>

## B Inference Details in MSP

For motion signal extraction, we perform inference using a frozen large language model, Meta-Llama-3-8B-Instruct, without any taskspecific fine-tuning. The model is accessed through a standard text generation interface and executed in half precision to reduce memory consumption during inference. For each referring expression, the prompt described above is provided to the model in a single forward pass, and the generated output is directly parsed to obtain structured motion control signals. To ensure deterministic and stable predictions across diferent runs, we disable stochastic decoding strategies during inference and adopt greedy decoding with a fixed maximum token budget. This design avoids randomness introduced by sampling based decoding and guarantees consistent motion signal extraction for identical inputs. The inference process is lightweight and performed independently for each expression, incurring negligible overhead compared to overall video segmentation pipeline. The case and prompt are shown in Figure 9 and 10.

## C More Visualizations

We provide more visualizations of diverse objects in Figure 11 and Figure 12 to demonstrate the robustness of our model under diverse referring expressions and object interactions. The examples cover a wide range of scenarios, including static objects, single object motion, and complex multi-object interactions in real scenes with similar appearances or overlapping trajectories. As shown, our method is able to correctly localize and segment the referred target under diverse and challenging practical conditions, even when multiple objects of the same category coexist, or when the expression emphasizes subtle motion diferences, role relations, or temporal action changes. These results indicate that the proposed framework can consistently and efectively integrate expression semantics and temporal cues to handle challenging referring video segmentation cases across diverse object types and motion patterns.

Prompt for Motion Signal Processing   
You are an expert in video action understanding.Given a natural-language referring expression,   
your task is to determine whether the expression contains dynamic action, and assign a motion   
strength score.   
motion\_strength ∈ [0.0, 1.0]:   
- 0.0–0.2 : static or almost static actions (standing, looking, lying)   
- 0.2–0.4 : mild actions (slight turning, leaning, shifting)   
- 0.4–0.6 : moderate locomotion (walking, approaching, following)   
- 0.6–0.8 : strong actions (running, chasing, jumping)   
- 0.8–1.0 : highly dynamic or multi-stage actions (turning around to steal, jumping over   
objects, spinning and grabbing, rapid interactions)   
Rules:   
- Only consider actions performed by the target object.   
- Ignore appearance attributes and spatial descriptors.   
- If the expression contains no action, motion\_strength = 0 and is\_dynamic = false.   
Output JSON only:   
{   
"expression": the original expression,   
"is\_dynamic": boolean,   
"dynamic\_expression": "the extracted action phrase (or empty)",   
"motion\_strength": float   
}   
Expression:   
[EXPRESSION]

![](images/1c6f2355a5e3929d5438350174f2c849ba26238c6508aa8f6a67e43d6f40d27f.jpg)  
Figure 9: Prompt and case for Motion Signal Processing.

![](images/2b2bfeb7d2414a55471995eb9728963cea31c5823b29bab45170256458d2e241.jpg)  
Figure 10: Cases for Motion Signal Processing.

![](images/13e63fec2357f7bbb3aee46185beb46953b21a709091012ea092168008434c3f.jpg)

![](images/97aee062c54679200458604ef928f34682a15a4ae17175eb85b01d6885827037.jpg)

![](images/0b65309a9eb01c9f11d6b42694e33e7e2e81f9dd024e8426d7975b724aebd17a.jpg)  
"a bathroom mirror" "the white outlet"

![](images/d0324117aad3798758dd5aa74a9e1bb30d42b929f6a5333d29fe4fc2b954465c.jpg)

![](images/b406f02be0f827bd7ecc57fa4934a2b8a6f581294bf95fbea6839e46316f1f31.jpg)

![](images/3840f2f15a5ed29defc0cb610869f963df68636a53704f2ea761c845741528dc.jpg)

![](images/bbf263b0db0cf9660cc52fa73c48289751d1e6cd47b5f832b5934f45c8da7304.jpg)

![](images/883e82d0ddca515244857850df5331d346813538acad87cfbf14b4a43586c3f3.jpg)

![](images/89360967f36613f80f250810b72c3552928a7b26815da5465f0b42f0183857fd.jpg)

![](images/3af967bbc0d42138be158e48d956753bb2602f45535cb587888d667dbb373936.jpg)  
"an elephant in the right is playing with another one in the left" "an elephant in the left is playing with another"

![](images/a3b7631464e869a74a3b4a1297a184bbc738b23c91733eff7cbdb6fcecb4df5f.jpg)

![](images/4595e4cf68adff64c1d4828481c52f25b136c446be668f482082b77a4288cda7.jpg)

![](images/2993397f8f0138ebcf9ea640fd5ec4251ac44eef9ee9bbe1eab8616e759c9ff7.jpg)  
"a white and red pillow"

![](images/ad3e257950b8fcd8099847c70f418d3e61cdea00c1070347ba601564095156ed.jpg)

![](images/9efb1b03c51674fdd82ce1b8aa36d6316308d5a24ed8299c060070cf9a796735.jpg)  
"a black cat playing with a sock"

![](images/958cb26c574e6bce916e20c368cb1924ff17bfee195b90625b71cb89ae39b4c6.jpg)

![](images/eec6bdb57af17e4f55aabd4774e1c81a82d731cfa0bedcc8d94989ee746589e0.jpg)

![](images/6b6549049a38dee8605071fe388a48fff926f3bff0b76e0d102b8f765299196d.jpg)

![](images/4ae6c4081b0cf423567a4f0e97de7081f7e8a3ad0684214bc01fb66c70f84b18.jpg)

![](images/b8ac859b64c2c4df88b999e4d4bba078d16325e02e00a616cdba71a333e17a0d.jpg)  
"a white parachute is opened in the sky and flying downward" "a hand of a person pushing two skydivers out of a plane"

![](images/fbeca23bfff84aeacea9432e0fe6301850097e97a936db646e4968f3fca7237f.jpg)

![](images/d1e5865ec4bafcca87469be10c8e502c71b0c4c1c0a433a7630a66f9cf086c66.jpg)

![](images/9e50b438fe3e451b8f52be57e14737591ba457eef034bf3b5599ff9d8cf74760.jpg)  
"a green bucket is behind a brown bear on the grass" "the brown bear is standing behind a green container and a large rock"

![](images/73676f1daa4849dbec04c7f427ba55f75e25185c2754b57d9d74b02017fc3b53.jpg)

![](images/d5fcf707c21480ec1b53fae0d2bd7ce20669e3b2bc02f7381897b4a112e09a94.jpg)  
Figure 11: Visualization for multiple text references.

![](images/7cabb4e32c942032cd3e84272f38849fe3486f51a814a3648ff3e99b603fc3e4.jpg)

![](images/f9958c766ec5bac3cc8e9931f0ef4b9661bcc3ce487cd138d3111bc8767631c6.jpg)

![](images/54efe2b7ff3a9c2bc104a225e96873c53a48e599efe5140346ba4c65e662c418.jpg)

![](images/9e24eb45ea18ff7e6dc9bb009416511713a1d794f742ddc29699c1b17b27583f.jpg)

![](images/ae5967928020a0cda1622a32b04a921ebb612bf3198f40a86d1c3b4eba75a072.jpg)  
"the person seated on the steps of the building facing the road" "the second car driving to the left" "the two girls moving towards the left"

![](images/a21544d6e43b84ddcf5ebf1a2db734d2768799fccbe458a76fcd04a26eb6fe29.jpg)

![](images/4df0d175ac845dccc7d4abeeefa7ab202b2334c32d321ab8adc94144a10e3945.jpg)

![](images/7bd78bc773929277bdd0a30eb7728dd77c602807cb690ace560507eb474f4bf3.jpg)

![](images/f435419172830adaac72f879b0cdf139aa8494caeffd09aafea3576cde54698a.jpg)

![](images/06cccdec5a0cd1bc53cb9875067f7172e78fc1c2c2a701c4709f0d3a48aca3e2.jpg)  
"the unmoving turtle in its original spot" "the darker turtle in the fight" "the lighter colored turtle in the fight"

![](images/d72cb052cfc0efe31c8ee6e6686d8ae26222914ffdf423b964d31c0ae6820be0.jpg)

![](images/68e14c08b50a48c46408d4a80d52ac0273d1e67deae197f5bec4ce51ab6c9e35.jpg)

![](images/9524206124440f5fc5a5cc01b267dd42939528a8a0c021123a91102266728b02.jpg)

![](images/c7903e4332944f7328d5e6d04fb87d701e27fd6573b87e021017084aa8eb242f.jpg)

![](images/5a593fb3216d1e335854d55e4235eccb5c713c3f2bb0b4cdaf0d9e8cdf9a8b52.jpg)  
"the cat on the bed, getting attacked by another cat" "the cat sitting down and then redirecting its attention to attack another cat"

![](images/34457d0d71766f93c8d5a40a87d17747286ef387e3e7bb2ffa7cad1f767d7ba9.jpg)

![](images/7ba26084148164d992062beaeb7f882b0ddceb7f0451d8d5f416ed171f50bccc.jpg)

![](images/3404716ee5e7064109eb9a9485f5cc8f4aff07dad5b2515cc3896028f11eb349.jpg)

![](images/b3e90c5ebdd6da4298173fad0b829895eab2f9131a866b9db6721bc369ee2b40.jpg)

![](images/6142a5cff568637e8eb8e12b770b56c79c03903f49fa4638580797e4ea5688a7.jpg)  
"the bear that is being pinned down by the other bear" "the bear that is pinning down the other bear"

![](images/0d6770d13e1d143c7fd70ead337161f2840c62d786197c2cc61143af3b2f78aa.jpg)

![](images/1cf44e567825d95ff4148ecaa935a2ba94c02ca8928f6ffc6e872b052f7c473c.jpg)

![](images/909d8c7111aef0f436364bc5339210d80f9c257c689705dabdf595b10c5cda83.jpg)  
"that horse is munching on food while remaining stationary"

![](images/667a93e62e062730e615f37308562aa9b121a6d7c886402abd34bcda325f81a5.jpg)

![](images/8935aeb50587f93f2008d9cf7acea471a86ddcb930df8e9a7b6af5b590180ccb.jpg)  
"the horse that starts facing right, then turns around to face backward, and finally turns to the left" "black horse"

Figure 12: Visualization for multiple text references.