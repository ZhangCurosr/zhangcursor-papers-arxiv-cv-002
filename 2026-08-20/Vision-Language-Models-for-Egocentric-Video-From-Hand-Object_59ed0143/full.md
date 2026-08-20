# Vision-Language Models for Egocentric Video: From Hand-Object Interaction to Embodied AI

Mohammad Zamani<sup>1</sup> and Fatemeh Ziaeetabar<sup>1,\*</sup>

<sup>1</sup>Department of Computer Science, School of Mathematics, Statistics and Computer Science, College of Science, University of Tehran, Tehran, Iran <sup>\*</sup>Address correspondence to: fziaeetabar@ut.ac.ir

## Abstract

Egocentric video captures activities from the wearer’s perspective, providing an actioncentered view of human attention, object manipulation, and interaction with the surrounding environment. This perspective is increasingly important for assistive technologies, wearable and immersive computing, and embodied intelligent systems, yet it introduces distinctive challenges, including continuous camera motion, occlusion, small active objects, viewpoint-dependent appearance, and long-range temporal dependencies. Vision–language models (VLMs) ofer promising capabilities for addressing these challenges by connecting visual observations with semantic knowledge and natural-language supervision. This survey presents a structured and critical review of VLMs for egocentric video understanding, tracing the progression from conventional recognition architectures to multimodal foundation models. We organize the literature around core tasks, datasets, and developments in hand–object interaction, temporal reasoning, frame and clip selection, multimodal representation learning, prompting, model adaptation, and semantic alignment. Particular attention is devoted to graph-based reasoning as a structured means of modeling spatial, temporal, and semantic relations among hands, objects, actions, and scene context. We further examine applications ranging from action recognition and anticipation to video question answering, assistive systems, and human-to-robot skill transfer. Our synthesis shows that current models remain more reliable at identifying visible objects than interpreting actions, evolving interactions, and user intent, especially in long activities. We identify temporally grounded and relationally explicit reasoning, eficient long-video processing, cross-domain generalization, multimodal integration, privacy, and trustworthy evaluation as key research priorities toward robust and deployable embodied intelligence.

## 1 INTRODUCTION

## 1.1 Motivation

Egocentric video, captured from a first-person viewpoint using wearable cameras, records scenes the way a human experiences them and reveals the wearer’s attention, behavior, and goals, making it a powerful substrate for analyzing human behavior that third-person imagery cannot match [1, 2]. Because the data originate from wearable devices, first-person understanding underpins augmented and virtual reality, human–computer interaction, robotics, and assistive technology, where accurately reading the wearer’s current and future actions enables real-time guidance—from natural AR interaction to warning a user before they touch a dangerous object [1, 2].

Perhaps most significantly, the first-person perspective directly exposes how people interact with nearby objects using their hands, positioning egocentric video as a natural bridge between human demonstration and embodied artificial intelligence [3]. Recent work treats egocentric human video as a scalable source of embodiment data: for example, EgoMimic collects egocentric human videos paired with 3D hand tracking and co-trains a single policy on human and robot data, achieving strong gains on long-horizon manipulation tasks and generalizing to entirely new scenes [4]. This trajectory, from understanding hand–object interaction toward learning embodied behavior, motivates the present survey, which examines how vision-language models can serve as the unifying interface that connects egocentric perception with language-based reasoning and action. As wearable sensing hardware improves and large multimodal models mature, first-person AI is increasingly seen as a key bridge between perception and action [1].

Beyond consolidating this literature, the present survey adopts a perspective that distinguishes it from existing egocentric overviews: we treat explicit relational structure—scene graphs, hand– object interaction graphs, and graph-guided reasoning—as a first-class organizing principle rather than a peripheral technique. A first-person action is defined less by appearance than by a relation unfolding in time—which hand engages which object, and how—and we argue throughout that graph-based, object-centric reasoning is a principled response to the well-documented failure of dense vision-language models to capture that structure. In this sense the survey is not merely a survey of egocentric vision-language models but a graph-aware survey of them, in which hand–object and scene-graph structure serves as the connective tissue linking low-level interaction perception to language-grounded, embodied reasoning.

## 1.2 Why Vision-Language Models for Egocentric Video?

Vision-language models learn a shared image–text representation from large-scale paired data— CLIP, for example, replaces fixed-label supervision with natural-language supervision to enable strong zero-shot transfer [5]. Building on this paradigm, EgoVLP pioneered egocentric videolanguage pretraining with the EgoClip set and the egocentric-aware EgoNCE objective [6], and EgoVLPv2 pushed cross-modal fusion into the backbone [7]. Beyond retrieval and recognition, such models now power first-person assistants like Vinci [8] and are being pushed toward explicit reasoning via reinforcement learning for temporal grounding [9]. Vision-language models thus ofer a unified interface connecting egocentric visual streams with language-based queries, recognition, and reasoning [5, 6].

## 1.3 Challenges in Egocentric Video Understanding

Despite this progress, egocentric video understanding faces distinctive obstacles. Data scale is a fundamental bottleneck: the largest egocentric corpus, Ego4D, ofers only 3,670 hours against the billions of image–text pairs available exocentrically, and downstream benchmarks remain limited in scale and quality [1]. The footage itself is uncurated, long-form, and dominated by continuous camera motion that complicates stable learning [1, 3, 10]. Finally, benchmarks skewed toward kitchen hand–object interaction constrain how broadly the knowledge in vision-language models can be evaluated [11]. Section 2.3 analyzes these perception challenges in depth.

## 1.4 Contributions of This Survey

A distinguishing feature of this survey, relative to prior egocentric overviews, is its graph-aware lens: throughout, we foreground explicit relational structure as the bridge between first-person perception and language-grounded reasoning. The main contributions are summarized as follows:

• We provide a comprehensive survey of vision-language models for egocentric video understanding, structured along a clear trajectory that progresses from hand–object interaction to embodied artificial intelligence.

• We establish the foundations of the field by formalizing egocentric video, its core tasks (action recognition, action anticipation, captioning, retrieval, and hand–object interaction understanding), and the perception challenges unique to the first-person setting, including ego-motion, occlusion, fine-grained manipulation, and long temporal dependencies.

• We review and comparatively analyze the major egocentric video datasets and benchmarks, and we discuss their requirements, biases, and limitations.

• We trace the evolution of video understanding models from classical convolutional and twostream architectures, through transformer-based models, to modern vision-language models, and we examine why standard vision-language models underperform on egocentric video together with the egocentric-specific architectures designed to address this gap.

• We ofer a focused treatment of hand–object interaction understanding, covering hand detection and tracking, object-centric and afordance-based interaction, and fine-grained interaction stages, as a central component of egocentric understanding.

• We survey spatiotemporal reasoning as well as graph-based and object-centric reasoning for egocentric video, including scene graphs, hand–object interaction graphs, graph-guided frame sampling, and prompting and semantic-alignment techniques for vision-language models.

• We discuss applications toward embodied intelligence, spanning assistive systems, human activity understanding, AR/VR and wearable AI, and human–robot collaboration, and we identify open challenges and future research directions to guide subsequent work.

Table 1: Coverage of representative recent surveys versus this one (Section 1.5). ✓ = covered in depth, ∼ = partial/incidental, × = not covered. “Graph/rel.” denotes scene-graph, hand–objectinteraction-graph, or relational structure used as an organizing lens—the axis on which this survey is distinctive.
<table><tr><td>Survey</td><td>Venue/Yr</td><td>Ego</td><td>Data/tasks</td><td>VLMs</td><td>HOI</td><td>Spat.-temp.</td><td>Graph/rel.</td><td>Prompt/align.</td><td>Embodied</td></tr><tr><td>Rodin et al. [2]</td><td>CVIU &#x27;21</td><td>√</td><td>2</td><td>X</td><td>2</td><td>2</td><td>×</td><td>X</td><td>2</td></tr><tr><td>Bandini &amp; Zariffa [12]</td><td>TPAMI &#x27;23</td><td>√</td><td>2</td><td>X</td><td>√</td><td>×</td><td>×</td><td>X</td><td>2</td></tr><tr><td>Plizzari et al. [3]</td><td>IJCV &#x27;24</td><td>√</td><td>√</td><td>2</td><td>2</td><td>2</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Li et al. [1]</td><td>MIR &#x27;26</td><td>√</td><td>√</td><td>2</td><td>√</td><td>2</td><td>×</td><td>×</td><td>2</td></tr><tr><td>Zhang et al. [13]</td><td>TPAMI &#x27;24</td><td>×</td><td>2</td><td>√</td><td>×</td><td>×</td><td>×</td><td>√</td><td>×</td></tr><tr><td>Li et al. [14]</td><td>Neurocomp. &#x27;24</td><td>×</td><td>2</td><td>2</td><td>2</td><td>√</td><td>√</td><td>×</td><td>×</td></tr><tr><td>This survey</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 1.5 Comparison with Existing Surveys

Several recent, well-cited surveys already map parts of this landscape, but each covers a diferent slice of it, and none adopts the graph-aware lens that organizes our review. The closest firstperson surveys are complementary in scope rather than competing: Rodin et al. [2] focus on the anticipation/prediction family, Bandini and Zarifa [12] on hand analysis, Plizzari et al. [3] on a forward-looking “outlook” of applications, and Li et al. [1] on a broad subject/object/environment taxonomy of egocentric vision. Orthogonal to these, Zhang et al. [13] survey vision-language models for general (non-egocentric) vision, and Li et al. [14] survey scene-graph generation without an egocentric or embodied focus. Table 1 contrasts their coverage against ours along the dimensions this survey spans. The pattern it exposes is the gap we fill: the one survey that centers relational/graph structure is not egocentric, the egocentric surveys do not treat that structure as an organizing principle, and none unifies egocentric vision-language modeling, hand–object interaction, graphbased reasoning, and embodied AI within a single trajectory. Ours is, to our knowledge, the first to do so—and the only entry checked on both the egocentric and graph/relational columns together.

## 1.6 Survey Methodology and Paper Selection

To keep this survey both comprehensive and representative of the current state of the art, we followed a structured search and selection procedure for identifying the literature reviewed in this paper. We searched the major computer-vision, machine-learning, and robotics venues—CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, AAAI, ACM MM, ICRA, and CoRL—together with the leading journals in the field (IEEE TPAMI, IJCV, and CVIU), and supplemented these with keyword searches over Google Scholar, IEEE Xplore, the ACM Digital Library, DBLP, and arXiv. The queries combined egocentric-vision terms (“egocentric,” “first-person,” “hand–object interaction,” “Ego4D,” “EPIC-KITCHENS”) with modeling terms (“vision-language model,” “video transformer,” “scene graph,” “action anticipation,” “embodied learning”).

From the resulting pool we applied three inclusion criteria, in decreasing order of priority. Relevance: a work had to bear directly on the survey’s trajectory from hand–object interaction to embodied intelligence, or to supply a foundational method (e.g., CLIP, ViT) on which that trajectory builds. Venue and impact: we preferentially selected papers published at the top-tier venues listed above or already well cited relative to their publication date, using citation count as a signal of community uptake rather than a hard threshold. Recency: we emphasized work from the last five years (2020–2026), which captures the shift to vision-language and embodied models, while retaining the seminal earlier papers that define the field’s foundations. Because egocentric vision is a fast-moving area in which many recent and relevant contributions first appear as preprints, we also included selected arXiv manuscripts that had not yet completed peer review. Such works are explicitly identified as preprints in the reference list and are interpreted cautiously throughout the survey. The literature search covered publications available through the end of July 2026. Works that were redundant with a stronger included reference, or that fell outside the first-person scope, were excluded.

![](images/da2b993e1b2dea51c948d59ae8ee3597bba4dbf63fd565e83e7acdecf6908700.jpg)  
Open Challenges & Future Directions (Sec. XI): better temporal reasoning · interaction-aware understand ing · graph-enhanced VLMs · eficient frame sampling · multimodal egocentric learning · embodied deployment  
Figure 1: Roadmap of this survey. The paper is organized as a single trajectory from hand–object interaction to embodied AI. Foundations and datasets (Sec. 2–3) supply the substrate; the model lineage advances from classical and transformer architectures to egocentric vision-language models (Sec. 4–5); interaction and reasoning form the conceptual core, spanning hand–object interaction, spatiotemporal reasoning, graph-based reasoning, and prompting and semantic alignment (Sec. 6– 9); and these converge on applications toward embodied intelligence (Sec. 10). Open challenges (Sec. 11) cut across every stage, shown by the dashed feedback links.

## 1.7 Organization of the Paper

The remainder of this paper is organized as follows, and is depicted as a visual roadmap in Fig. 1. Section 2 presents the foundations of egocentric video understanding, including its definition, core tasks, and characteristic perception challenges. Section 3 reviews major egocentric video datasets, providing a comparative analysis and discussing their biases and limitations. Section 4 traces the evolution of video understanding models from classical and transformer-based architectures to vision-language models. Section 5 examines vision-language models for egocentric video understanding, analyzing why standard models fail and surveying egocentric architectures. Section 6 focuses on hand–object interaction understanding, while Section 7 covers spatiotemporal reasoning and Section 8 addresses graph-based and object-centric reasoning. Section 9 discusses prompting and semantic alignment in vision-language models. Section 10 presents applications toward embodied intelligence. Section 11 outlines open challenges and future directions, and Section 12 concludes the paper.

## 2 Foundations of Egocentric Video Understanding

## 2.1 What is Egocentric Video?

Egocentric vision, also called first-person vision, captures visual and multimodal data through cameras or sensors worn on the human body, ofering a perspective that simulates human visual experience [1]. The paradigm was articulated early by Kanade and Hebert, who argued that sensing the environment and the subject’s activities from a wearable viewpoint is advantageous precisely because it records the scene from the subject’s own vantage point, in contrast to the traditional thirdperson, or surveillance, perspective [15]. Because the camera is worn, its movements are driven by the wearer’s intentions and attention, and the hands and manipulated objects often occupy the central field of view, although they are frequently afected by self-occlusion and object occlusion [2, 12].

This setting difers markedly from exocentric vision, in which the camera is stable and disjoint from the user. The two viewpoints are complementary: the egocentric perspective captures close-by hand–object interactions and the wearer’s attention, whereas the exocentric perspective captures full body pose and surrounding scene context [16]. The cost of the first-person viewpoint is that the camera moves with the body, causing fast motion and sudden illumination changes that can degrade video quality [12]. Egocentric recordings are also typically long, unscripted, captured during daily activities, and accompanied by multimodal signals such as audio, eye gaze, and inertial data [1, 10].

The modality is recorded with wearable devices mounted on the head or chest, including action cameras such as GoPro and, more recently, purpose-built smart glasses such as Project Aria [12, 17]. Its growth has been driven by large-scale datasets: Ego4D ofers 3,670 hours of daily-life video from 931 wearers across 74 locations [10], the EPIC-KITCHENS series provides densely annotated unscripted kitchen activity [18], and Ego-Exo4D adds simultaneously captured first- and thirdperson video of skilled activities [16]. Together, these resources establish first-person perception as a distinct research frontier [10].

## 2.2 Core Tasks in Egocentric Video Understanding

Egocentric video understanding has crystallized around five canonical tasks, each inherited from third-person video analysis but reshaped by the first-person viewpoint—severe camera motion, hand occlusions, of-center objects, and the tight coupling of perception to the wearer’s intent [3, 15]. Table 2 defines each task alongside its defining first-person dificulty. Nearly all of them are anchored by two benchmarks: EPIC-KITCHENS-100, with 89,977 fine-grained (verb, noun) action segments over 100 hours in 45 kitchens, and Ego4D, with 3,670 hours of daily-life video across 74 locations [10, 18].

Table 2: The five canonical egocentric video-understanding tasks (Section 2.2): what each is and its defining first-person dificulty.
<table><tr><td>Task</td><td>What it is</td><td></td><td>Key difficulty</td><td></td></tr><tr><td>Action recog- nition [18]</td><td>action</td><td>Classify a trimmed clip as a (verb, noun)</td><td>Long-tailed positions; small active jects</td><td>com- blur; ob-</td></tr><tr><td>Action antic- ipation 20] Video</td><td>[19,</td><td>Predict future ac- tion(s) before they occur</td><td>Inherent tainty;</td><td>uncer- many plausible futures</td></tr><tr><td>tioning 23] Video</td><td>cap- [21-</td><td>Generate language narrations of events</td><td>Grounding HOI; coherence</td><td>fine long-video</td></tr><tr><td>trieval 25] HOI under-</td><td>re- [24,</td><td>Cross-modal text↔video match- ing Detect</td><td>Many-to-many relevance; horizon queries Interaction (verb)</td><td>long-</td></tr><tr><td>standing [26, 27]</td><td></td><td>hands/objects, contact, and state change</td><td>vs. mere recognition</td><td>object</td></tr></table>

Methodologically these tasks track the broader shift from convolutional and recurrent models to video transformers and, most recently, egocentric video-language pretraining such as EgoVLP [6], developed in Section 4. Hand–object interaction understanding is the most distinctively egocentric of the five, since the wearer’s hands dominate the frame [12], and we treat it in depth in Section 6. A single theme unites all five: current models name objects well but falter on the interaction and long-horizon structure that define first-person activity—EgoVLMs collapse under single verb substitutions [11], and billion-parameter models still trail humans badly on long-form question answering [28]. That gap motivates the interaction- and graph-centric treatment of the rest of this survey.

## 2.3 Challenges of Egocentric Perception

Egocentric video compounds the dificulties of generic video with challenges intrinsic to the firstperson viewpoint—an unconstrained head-mounted camera framing a narrow, shifting slice of the world dominated by close-range interaction [1, 3, 15]. Four stand out.

## 2.3.1 Ego-motion

The camera is rigidly coupled to the wearer’s head, so head rotations and gaze shifts induce large, unpredictable motion and blur with no analogue in the stabilized footage on which standard backbones are trained, degrading clip features and breaking trackers [1, 3]. Mitigations either suppress it— hand-focused stabilization yields a +17% (and +22% combined) Top-1 gain on Ego-Exo4D keystep recognition [29]—or exploit it as signal, reading head trajectory from IMU and visual–inertial SLAM as a cue to attention and 6-DoF pose [17, 30].

## 2.3.2 Occlusion and Partial Visibility

Acting hands and held objects persistently occlude the most informative region, while the narrow field of view truncates objects and lets them leave and re-enter view as attention shifts [31, 32]. This breaks trackers tuned on framed footage—EgoTracks scores far below conventional benchmarks because objects vanish and must be re-detected over minutes—and motivates occlusion-robust, amodal, high-capacity models such as HaMeR [32, 33].

## 2.3.3 Fine-grained Manipulation

Daily activity is dominated by subtle near-field manipulations—“take” vs. “put,” or one small object among many—reflected in EPIC-KITCHENS-100’s compositional space of 97 verbs and 300 nouns [18]. Because the discriminative evidence lies in brief contact moments, global-context models confuse fine classes and even large EgoVLMs are flipped by single verb or noun swaps [11]; progress leans on contact-rich datasets such as HOI4D, Assembly101, and ARCTIC [34–36].

## 2.3.4 Long Temporal Dependencies

Egocentric recordings run for minutes to hours, with steps that derive meaning only from the whole activity—far beyond the few-second clips standard backbones ingest [3, 10]. In the original EgoSchema evaluation, the tested billion-parameter models scored below 33%, whereas humans reached approximately 76% [28]. This gap has motivated memory-augmented transformers such as MeMViT and MC-ViT, although many systems still rely on appearance shortcuts rather than genuine temporal reasoning [9, 37, 38].

## 3 Egocentric Video Datasets

The progress of vision-language models for first-person understanding is inseparable from the datasets that train and evaluate them. Egocentric corpora difer markedly from the third-person collections that have driven mainstream video understanding, both in the visual phenomena they capture and in the annotation efort they demand [3]. This section first articulates what egocentric understanding requires of a dataset (Section 3.1), then surveys five widely used datasets that span the design space (Section 3.2), compares them along dimensions relevant to vision-language modeling (Section 3.3), and finally examines the biases and limitations that constrain how far models trained on them can generalize (Section 3.4).

## 3.1 Dataset Requirements for Egocentric Understanding

First-person, or first-person-vision (FPV), data imposes requirements that are largely absent from third-person video. Because the camera is worn, it records exactly what is in front of the user, its motion is driven by the wearer’s own activity and attention, and the hands together with the objects they manipulate tend to occupy the center of the field of view [12]. This intimate coupling between sensor and behavior is precisely what makes the first-person perspective valuable: it captures the environment and the subject’s activities from a vantage point that is more informative about intent and interaction than an external, surveillance-style view [15].

Several properties follow from this viewpoint and define what an egocentric dataset must capture. First, the imagery is dominated by fast, unconstrained head and camera motion, severe occlusion, and background clutter, none of which are typical of curated third-person footage [39]. Second, because action from a first-person viewpoint is largely a matter of understanding how the hands engage objects, hand–object interaction is the central signal rather than a peripheral one [12, 39]. Third, everyday behavior unfolds as long, unscripted activity, so datasets must preserve long-term temporal structure rather than isolated, trimmed clips [18]. Fourth, the first-person setting naturally afords rich multimodality—inertial measurement units (IMU), eye gaze, audio, stereo, and threedimensional scene geometry—signals that are dificult to obtain from third-person video and that have been shown to be important for tasks such as anticipating future activity [2]. Finally, alwayson wearable capture raises privacy and consent concerns that responsible datasets must address [3]. Recent surveys organize the resulting research landscape around the understanding of the subject, the objects, the environment, and their interactions [1], and they emphasize that the field is still far from the always-on, personalized, life-enhancing wearable systems that motivate it [3].

## 3.2 Major Egocentric Datasets

We review five datasets that together cover the principal axes of egocentric data design: from constrained three-dimensional hand-object capture to massive in-the-wild collection, and from egoonly recording to simultaneous first- and third-person views.

## 3.2.1 EPIC-KITCHENS

EPIC-KITCHENS-100 [18] is the largest unscripted dataset of daily kitchen activity: 100 hours from 37 participants across 45 kitchens, densely narrated into 89,977 (verb, noun) action segments (97 verbs, 300 nouns; 4,025 action classes). Its suite of recognition, anticipation, retrieval, and domainadaptation challenges made it the standard egocentric testbed, and the VISOR extension [31] adds 272K pixel-level hand/object masks with hand–object relations.

## 3.2.2 Ego4D

Ego4D [10] is the field’s scale anchor: 3,670 hours of daily-life video from 931 wearers across 74 locations in nine countries. Portions of the collection are accompanied by audio, eye gaze, stereo, three-dimensional meshes, and/or synchronized recordings from multiple egocentric cameras. The dataset defines five benchmark suites (Episodic Memory, Hands and Objects, Audio-Visual Diarization, Social, Forecasting). Its dense free-form narrations made it the de facto backbone for egocentric video-language pretraining; the derived EgoSchema benchmark [28] exposes the long-form reasoning gap: in its original evaluation, the tested large models scored below 33%, whereas humans reached approximately 76%.

## 3.2.3 H2O

H2O [39] instead targets fine-grained 3D understanding of two hands manipulating objects: 571,645 synchronized multi-view RGB-D frames (4 subjects, 36 actions, 8 objects) with dense 3D poses of both hands, 6DoF object poses, meshes, and point clouds. It is a key resource for 3D two-hand– object grounding, despite its small subject count.

## 3.2.4 Charades-Ego

Charades-Ego [40] is defined by paired first- and third-person capture of the same scripted activities: 68,536 instances over 68.8 hours, 112 actors, and the 157 Charades activity classes. The paired recordings make it an early testbed for the ego-exo correspondence problem.

## 3.2.5 Ego-Exo4D

Ego-Exo4D [16] couples first- and third-person capture at scale for skilled activity: 1,286 hours from 740 participants across 13 cities, in 5,035 takes spanning physical (sports, dance, music) and procedural (cooking, repair) domains. Each take pairs Project Aria glasses (RGB, gaze, IMU, audio) with four to five calibrated static cameras, and it adds expert-commentary language and benchmarks for keystep recognition, proficiency estimation, ego-exo correspondence, and 3D body/hand pose.

## 3.3 Comparative Dataset Analysis

The five datasets occupy complementary positions in the design space, as summarized in Table 3. They difer by roughly an order of magnitude or more in scale, from the 100 hours of EPIC-KITCHENS-100 to the 3,670 hours of Ego4D, and along a spectrum of naturalism that runs from the constrained laboratory capture of H2O, through the scripted activities of Charades-Ego, to the fully unscripted recording of EPIC-KITCHENS, Ego4D, and Ego-Exo4D. They also difer in viewpoint: EPIC-KITCHENS and Ego4D are ego-only, H2O augments the egocentric view with static cameras, and Charades-Ego and Ego-Exo4D provide paired and simultaneous first- and third-person views, respectively. Annotation richness varies in kind as well as degree: H2O ofers dense threedimensional hand and object pose, Ego-Exo4D adds three-dimensional body and hand pose together with point clouds, whereas EPIC-KITCHENS and Charades-Ego remain largely two-dimensional.

These diferences map directly onto the downstream tasks each dataset can support for visionlanguage models. Language-grounded pretraining and video-text retrieval are best served by EPIC-KITCHENS-100, Ego4D, and Ego-Exo4D, whose free-form narrations provide aligned textual supervision; long-form video question answering is probed by EgoSchema; three-dimensional hand-object grounding relies on H2O and related datasets; and ego-exo transfer and skill or proficiency estimation are uniquely supported by Ego-Exo4D and Charades-Ego. This complementarity has motivated work on hand-object interaction datasets that trade scale for annotation density, such as Assembly101 [35], ARCTIC [36], HOI4D [34], and HOT3D [41], which provide multi-view and three-dimensional supervision in more controlled settings

Table 3: Comparison of major egocentric video datasets along dimensions relevant to vision-language modeling. Ego4D scale refers to the full video collection, while annotations and auxiliary modalities cover subsets; H2O is reported in frames. Figures are version-dependent for Ego4D and Ego-Exo4D (see Section 3.4).
<table><tr><td>Dimension</td><td>EPIC-KITCHENS-100</td><td>Ego4D</td><td>H2O</td><td>Charades-Ego</td><td>Ego-Exo4D</td></tr><tr><td>Scale</td><td>100 h</td><td>3,670 h</td><td>571,645 frames</td><td>68.8 h</td><td>1,286 h</td></tr><tr><td>Participants</td><td>37</td><td>931</td><td>4</td><td>112</td><td>740</td></tr><tr><td>Geographic spread</td><td>4 cities</td><td>9 countries</td><td>3 envs. (lab)</td><td>worldwide homes</td><td>13 cities</td></tr><tr><td>Naturalism</td><td>unscripted</td><td>unscripted</td><td>lab</td><td>scripted</td><td>unscripted</td></tr><tr><td>Viewpoint</td><td>ego</td><td>ego</td><td>ego + static</td><td>paired ego-exo</td><td>simultaneous ego-exo</td></tr><tr><td>Domain</td><td>kitchen</td><td>daily life</td><td>tabletop HOI</td><td>household</td><td>skilled activity</td></tr><tr><td>3D annotation</td><td>2D (masks)</td><td>partial</td><td>rich 3D</td><td>2D</td><td>rich 3D</td></tr><tr><td>Language supervision</td><td>strong</td><td>strong</td><td>weak</td><td>moderate</td><td>strong</td></tr></table>

## 3.4 Dataset Biases and Limitations

Despite their scale, egocentric datasets carry biases that constrain the generalization of models trained on them. The most pervasive is geographic and domain narrowness. EPIC-KITCHENS is confined to kitchen activity recorded in only four Western cities, and even Ego4D, the most geographically diverse, acknowledges that its 74 locations fall well short of global coverage, that camera wearers are concentrated in urban and college-town areas, and that pandemic-era collection skewed its footage toward stay-at-home scenarios [10]. These limitations are a concrete instance of the dataset-bias phenomenon long recognized in computer vision, whereby collections that set out to represent the visual world instead become closed worlds that generalize poorly to one another [42].

A second source of bias lies in the annotations themselves. Free-form narrations are authored by participants or crowd workers and are overwhelmingly English-centric, so the very language signal used to align video and text inherits lexical noise and culturally specific word choices; Ego4D explicitly notes that its narrations, produced by annotators at two sites, are biased toward local word usage [10]. Demographic skew compounds these efects, as does the small number of subjects in densely annotated three-dimensional datasets such as H2O, which limits the diversity of hand morphologies and manipulation styles represented. Finally, always-on first-person capture raises privacy and consent concerns; datasets mitigate these through informed consent and de-identification, or by recording in controlled environments [3].

For vision-language models, the practical consequence is that systems pretrained on Western, kitchen-heavy, English-narrated egocentric data risk degraded performance when transferred to other cultures, languages, viewpoints, and activity domains. Mitigations include explicit cross-domain and domain-adaptation evaluation, balanced multi-domain pretraining mixtures, and transparent reporting of narration provenance. Standardized capture platforms such as Project Aria [17] help unify modalities across newer datasets, but by concentrating data collection on common hardware they may also introduce shared, sensor-induced biases that future work will need to characterize.

![](images/8d6662ba71745a424e03cad329bbed11471c73df13465f4373c4776ce684a314.jpg)

Figure 2: Evolution of video understanding models (Section 4). Three eras, read left to right, each inheriting representational machinery from the preceding era while changing its temporal modeling and supervision strategy. Classical models hand-engineer temporal structure through spatiotempora convolutions, recurrent aggregation, and parallel appearance–motion streams (Section 4.1). Transformer models replace those inductive biases with factorized space-time attention, made trainable at video scale by masked pretraining (Section 4.2). Vision-language models replace the fixed labe space itself with natural-language supervision, a lineage running CLIP → VideoCLIP → Frozenin-Time → EgoVLP. Red boxes mark the gap each era left open and the next set out to close; the horizontal dashed links mark what each era inherited rather than replaced—image pretraining and convolutional backbones carried into the transformer era, and transformer space-time encoders carried into egocentric video-language pretraining.

## 4 Evolution of Video Understanding Models

Modern egocentric video understanding inherits its core machinery from a decade of progress on third-person action recognition. Before transformer-based vision–language models reshaped the field, three families of deep architectures defined the state of the art: pure convolutional networks operating directly on stacks of frames, hybrid CNN–RNN pipelines that delegated temporal modeling to recurrent units, and two-stream networks that processed appearance and motion as parallel pathways. We review these classical models because the egocentric literature, from EPIC-KITCHENS [18] to Ego4D [10], still relies on them as backbones, baselines, and conceptual scafolding for the hand–object reasoning that motivates this survey. Fig. 2 summarizes the trajectory this section traces, from hand-engineered temporal modeling to language-supervised, open-vocabulary representation.

## 4.1 Classical Video Understanding

Three pre-transformer families defined video action recognition. Each solved one bottleneck and left another open; Table 4 summarizes the progression.

## 4.1.1 CNN-Based Models

Convolutional models learn motion directly from pixels. After Karpathy et al. [43] showed that naive frame fusion barely beats single-frame baselines, C3D [44] demonstrated that homogeneous 3×3×3 convolutions yield compact, transferable spatiotemporal descriptors; P3D [45] and R(2+1)D [46] then factorized these costly 3D kernels into cheaper spatial and temporal parts, and I3D [47] inflated 2D ImageNet filters into 3D to inherit mature image pretraining. Later variants chased eficiency and range—the dual-rate pathways of SlowFast [48], the scaled tiny backbone of X3D [49], the zero-cost channel shift of TSM [50], and non-local blocks for long-range context [51]. Left unsolved: fixed-length clips still cannot span the minutes-long, goal-directed structure of egocentric activity.

## 4.1.2 CNN–RNN Architectures

A second family kept a per-frame CNN and delegated time to a recurrent network, modeling variablelength structure that fixed clips could not. LRCN [52] and the pooling/LSTM study of Ng et al. [53] aggregated per-frame features over longer spans; ConvLSTM [54] and VideoLSTM [55] restored spatial locality inside the recurrent cell; and encoder–decoder LSTMs previewed self-supervised pretraining [56]. This family fit first-person video naturally, where hands and gaze must be tracked over time: dedicated first-person cue encoders [57], hand- and object-supervised twin streams [58], and the attention unit LSTA [59] reached the state of the art on EGTEA and EPIC-KITCHENS [18]. Left unsolved: recurrence captured order but not the appearance–motion split that separates visually similar first-person actions.

## 4.1.3 Two-Stream Networks

The third family processed appearance and motion in parallel: an RGB stream and a stacked-opticalflow stream fused late [60], later refined by better fusion placement and residual cross-connections [61, 62], sparse segment sampling in TSN [63], multi-scale temporal relations in TRN [64], and learnable aggregation in ActionVLAD [65]. It transferred well to first-person video, where rapid hand and head motion is informative, culminating in EPIC-Fusion [66], which added audio to set the state of the art on EPIC-KITCHENS. Left unsolved: precomputed optical flow is expensive and brittle under the severe ego-motion of wearable capture.

Taken together, these families established the representational vocabulary—spatiotemporal convolutions, recurrent temporal aggregation, and parallel appearance–motion streams—inherited by the transformer-based and vision–language models examined next. Their common limitation, a fixed label space with no language interface, is exactly what the vision-language models of Section 5 set out to remove.

Table 4: The three classical (pre-transformer) video-understanding families: what each generation contributed and the gap it left for egocentric vision-language understanding (Section 4.1).
<table><tr><td colspan="2">Generation</td><td colspan="2">Representative models</td><td colspan="2">Key innovation</td><td colspan="2">Problem solved Left unsolved for egocen- tric</td></tr><tr><td>CNN CNN</td><td>3D-</td><td>C3D [44], SlowFast TSM [50]</td><td>I3D [47], [48],</td><td>Spatiotemporal convolu- tions; inflate 2D pretrain- ing into 3D</td><td>Learns motion from pix- els with mature image priors</td><td>Fixed short clips miss long- horizon, goal-directed activ- ity</td></tr><tr><td>CNN-RNN</td><td></td><td>LRCN [52], STM [54], LSTA [59]</td><td>ConvL-</td><td>Per-frame CNN + recur- rent temporal aggrega- tion</td><td>Variable-length model- ing; tracks hands/gaze over time</td><td>Weak appearance-motion separation; closed vocabu- lary</td></tr><tr><td>Two-stream</td><td></td><td>Two-Stream TSN [63], Fusion [66]</td><td>[60], EPIC-</td><td>Parallel appearance + motion (and audio) path- ways</td><td>Explicit motion and multimodal cues for fine actions</td><td>Optical flow costly/brittle under ego-motion; no lan- guage grounding</td></tr></table>

## 4.2 Transformer-Based Video Understanding

Attention-based architectures reshaped video understanding in a few years. The lineage is direct: the Vision Transformer (ViT) establishes patch tokenization, TimeSformer and ViViT carry selfattention to the temporal axis through factorized schemes, and VideoMAE makes these data-hungry video transformers trainable via self-supervised masking. We take these four as the pillars and connect each to the egocentric setting.

## 4.2.1 Vision Transformer (ViT)

ViT [67] imports the self-attention primitive of Vaswani et al. [68] into vision as the first puretransformer image classifier: an image becomes a sequence of linearly embedded patches processed by a standard Transformer encoder. Its thesis—that an inductive-bias-free design matches CNNs given enough data—also names its weakness for egocentric corpora, which are far smaller than web scale. Two extensions relax that data demand: DeiT [69] trains on ImageNet-1k alone via heavy augmentation and a distillation token, while Swin [70] adds hierarchical, shifted-window local attention with linear complexity, seeding the later Video Swin.

## 4.2.2 TimeSformer

TimeSformer [71] applies attention directly to frame patches and compares five space-time schemes, finding that divided space-time attention—temporal then spatial within each block—gives the best accuracy/eficiency trade-of by decoupling the quadratic cost of joint attention. The distinction is egocentric-relevant: space-only attention sufices on appearance-dominated Kinetics but the temporal component is essential on motion-centric data such as Something-Something and EPIC-KITCHENS, exactly the hand–object regime.

## 4.2.3 VideoMAE

Masked modeling attacks the data appetite directly. Following image MAE [72], VideoMAE [73] masks video with a tube strategy at an extreme 90–95% ratio—enabled by temporal redundancy— and trains strongly even on a few thousand videos without extra data, which is why its backbones dominate Ego4D and EPIC-KITCHENS leaderboards. The concurrent ST-MAE [74] uses random spacetime masking (the key contrast), and VideoMAE V2 [75] scales the paradigm to a billionparameter video foundation model via dual masking.

## 4.2.4 ViViT

ViViT [76] tokenizes a clip (optionally via tubelet embedding) and proposes four factorizations that trade cost for accuracy; its factorized self-attention variant coincides with TimeSformer’s divided attention, showing the two core video transformers converged on the same principle, while its factorized encoder is often the best compute trade-of. It, too, reports EPIC-KITCHENS results, tying the design to the egocentric theme.

## 4.2.5 Architectural Breadth and Egocentric Applications

Beyond the four pillars, a cluster of variants advanced complementary axes: the multiscale MViT/ MViTv2 family [77, 78], Video Swin’s 3D shifted windows [79], Motionformer’s trajectory attention along motion paths [80], feature-prediction pretraining in MaskFeat [81], the convolution–attention hybrid UniFormer [82], and eficiency-oriented MTV [83], TokenLearner [84], and the memoryaugmented MeMViT [37] for minutes-scale recognition—the last especially suited to long egocentric streams.

These backbones recur throughout egocentric video-language pretraining: EgoVLP [6] and EgoVLPv2 [7] build on TimeSformer-style space-time encoders, LaViLa [23] learns a contrastive embedding on the same backbone from LLM-generated narrations, and VideoMAE(V2) anchors most Ego4D challenge entries—closing the loop between general transformer architectures and egocentric practice.

## 4.3 From Vision Models to Vision-Language Models

The architectures of Section 4.2 learn visual representations through supervised or self-supervised objectives, but they do not provide an explicit natural-language interface. Vision-language models instead align visual encoders with natural language, yielding open-vocabulary representations that can transfer to unseen tasks without task-specific labels. This subsection follows that progression as a genealogy: CLIP establishes image-text contrastive pretraining, VideoCLIP extends contrastive pretraining to video, Frozen-in-Time introduces a space-time encoder that jointly handles images and video, and EgoVLP transfers this design to the egocentric domain.

## 4.3.1 CLIP

CLIP [5] jointly trains an image encoder and a text encoder with a symmetric contrastive objective over 400M web image-text pairs, replacing fixed class labels with natural-language supervision (Fig. 3). After pretraining, language references the learned visual concepts, enabling strong zero-shot transfer (e.g., 76.2% zero-shot ImageNet top-1, matching a supervised ResNet-50)—the open-vocabulary property that fixed-label backbones lack.

![](images/eeffdc0a174677efe04cbc6b69ffa1fd3e8c0ec0d2ceaf039a21166743c29527.jpg)  
Figure 3: The CLIP framework. (Left) Contrastive pretraining jointly aligns an image encoder and a text encoder over batches of image-text pairs; (right) zero-shot prediction constructs a classifier from natural-language prompts. Illustration inspired by [5].

## 4.3.2 VideoCLIP

VideoCLIP [85] carries contrastive pretraining to video with a dual encoder trained on HowTo100M, using two ideas: positives from temporally overlapping (not exactly aligned) video-text clips, and retrieval-mined hard negatives. It reaches strong zero-shot text-video retrieval, QA, and action segmentation, in cases beating supervised methods.

## 4.3.3 Frozen-in-Time

Frozen-in-Time [25] pairs a ViT/TimeSformer-style space-time visual encoder with a text encoder, ingesting images and video uniformly (an image is a single-frame video) under a curriculum that gradually adds temporal context. Trained on the modest WebVid-2M plus CC3M, it still reaches state-of-the-art retrieval—and this space-time encoder is the direct architectural ancestor of EgoVLP.

## 4.3.4 EgoVLP

EgoVLP [6] transfers this design to first-person video along three axes (Fig. 4): EgoClip, a ∼3.8M clip-text pretraining set curated from Ego4D; EgoNCE, a contrastive objective that mines egocentricaware positives (clips sharing actions or nouns) and negatives; and the EgoMCQ alignment benchmark. Built on the Frozen-in-Time / TimeSformer encoder, it transfers strongly across five egocentric tasks and won the EPIC-KITCHENS multi-instance-retrieval and OSCC challenges, establishing the template the rest of this survey builds on.

![](images/94a6725008f1ddf4a8ff059a5a01cfae08b7ca3fd295b16aaa5d1767d9af3736.jpg)  
Figure 4: The EgoVLP framework: the EgoClip pretraining set curated from Ego4D, a spacetime video-language model trained with the EgoNCE objective, and the EgoMCQ development benchmark, with transfer to egocentric downstream tasks. Illustration inspired by [6].

## 4.4 Summary of Model Evolution

The trajectory traced across this section moves from hand-engineered temporal modeling toward language-supervised, open-vocabulary representation. Classical video understanding established three representational vocabularies: spatiotemporal convolutions, from the Sports-1M study [43] through C3D [44] and its factorized successors [45, 46] to the Kinetics-pretrained I3D [47]; recurrent temporal aggregation over per-frame CNN features [52, 53], refined for first-person video by LSTA [59]; and parallel appearance–motion streams [60], extended to the multimodal EPIC-Fusion Temporal Binding Network [66]. These families supplied the backbones and baselines on which egocentric benchmarks such as EPIC-KITCHENS [18] and Ego4D [10] still rely.

The transition to attention replaced these inductive biases with learned space-time interactions. ViT [67] introduced patch tokenization, TimeSformer [71] and ViViT [76] factorized self-attention across space and time, and VideoMAE [73] made data-hungry video transformers trainable through self-supervised masked modeling—the property that made these backbones dominant on Ego4D and EPIC-KITCHENS leaderboards. The final shift, from vision models to vision-language models, supervised visual encoders with natural language: CLIP [5] established image-text contrastive pretraining, VideoCLIP [85] and Frozen-in-Time [25] carried it to video, and EgoVLP [6] transferred the design to the egocentric domain via EgoClip and EgoNCE. Each generation thus inherited representational machinery from the previous one while changing its temporal modeling and supervision strategy, setting the stage for the egocentric vision-language models that the remainder of this survey examines.

## 5 Vision-Language Models for Egocentric Video Understanding

Building on the model evolution reviewed in Section 4, this section examines how vision-language models extend conventional video representations toward the language-grounded understanding required for egocentric and embodied AI.

We first diagnose why standard, predominantly exocentric vision-language models underperform on egocentric video (Section 5.1), then organize the egocentric vision-language landscape into four broad model groups (Section 5.2), and finally synthesize their tradeofs (Section 5.3).

## 5.1 Why Standard VLMs Fail in Egocentric Videos

Despite their success on web-scale third-person benchmarks, general-purpose vision-language models degrade sharply when confronted with first-person video. The failures cluster into four interrelated categories.

## 5.1.1 Weak Interaction Understanding

The most direct evidence comes from Xu et al. [11], whose EgoHOIBench asks a model to pick the correct Ego4D caption from candidates difering by a single verb or noun, scoring a trial correct only when both are right (Fig. 5). Even models pretrained on millions of interaction clips fail: LaViLa scores 74.33% on nouns but only 46.61% on verbs—a ∼28-point gap traced to object-recognition shortcuts. Their EgoNCE++ objective, which mines HOI-aware hard-negative captions, raises the LaViLa++ variant’s verb accuracy by 34.02 percentage points [11]. This verb–noun asymmetry is the empirical core of the survey’s thesis: standard models recognize what objects are present but not how they are manipulated.

## 5.1.2 Poor Temporal Reasoning

EgoSchema [28] crystallized the temporal-reasoning gap: its long-form Ego4D questions carry a median “temporal certificate” of roughly 100 seconds—far longer than those of prior datasets. In the original evaluation, the tested billion-parameter models scored below 33% (chance 20%), whereas humans reached approximately 76%. Xu et al. [9] trace this limitation to objectives that reward answer correctness without temporal consistency, allowing models to exploit frame-level shortcuts. Their order-aware reinforcement-learning method, which contrasts ordered and shufled frames, lifts a 3B backbone from 20.4% to 49.7% on EgoSchema and outperforms substantially larger baselines, indicating that objective design is an important factor beyond model scale.

## 5.1.3 Object Ambiguity

The active object is often small, hand-held, partially occluded, of-center, or truncated, and must be told apart from many distractors. EgoTracks [32] shows single-object trackers scoring far below their conventional-benchmark level because objects repeatedly vanish and must be re-detected, and

![](images/d5f606cc6e2cf1012279eb50d01d3ae2830dc8a4101305f51c0a6ce531cd7144.jpg)  
Figure 5: The EgoHOIBench diagnostic. A model must select the correct caption from candidates that difer by only a single verb or noun, scoring a trial correct only when both are right; egocentric video-language models exhibit a large verb–noun accuracy gap, revealing only superficial hand– object interaction understanding. Illustration inspired by [11].

VISOR [31] reports small objects as a dominant failure mode even after fine-tuning. Active-object work [26, 86] shows that identifying which object is in contact needs explicit hand-contact reasoning generic detectors lack—so for VLMs this surfaces as the noun-shortcut bias: they name salient objects but cannot bind them to the acting hand.

## 5.1.4 Motion-Induced Noise

Continuous ego-motion, blur, and viewpoint instability corrupt the dense spatiotemporal features VLMs rely on. Chavis et al. [29] show a stabilized hand-focused crop improves Ego-Exo4D keystep recognition (+17–22% Top-1), while EgoDistill [30] instead exploits the head-motion signal via IMU to reconstruct video features at ∼200× fewer multiply–adds. Ego-motion is thus at once a nuisance to suppress and a signal to exploit—a duality exocentric VLMs never learn to handle.

## 5.2 Egocentric Vision-Language Architectures

Egocentric vision-language models can be organized into four broad groups distinguished by their primary objective and output modality.

## 5.2.1 Retrieval-Based Methods

Retrieval-based models learn a joint video–text embedding via contrastive learning and are evaluated on multi-instance retrieval and multiple-choice matching. EgoVLP [6] pioneered egocentric pretraining with three components: EgoClip, a 3.8M clip–text pretraining set curated from Ego4D; EgoNCE, a contrastive objective that mines action-aware positives and scene-aware negatives; and EgoMCQ, a 39K-question development benchmark with inter-video and intra-video settings. EgoVLP adapts the dual-encoder Frozen-in-Time architecture [25] with a TimeSformer video backbone. EgoVLPv2 [7] advances this by inserting cross-modal fusion directly into the video and language backbones via gated cross-attention, switchable between dual-encoder and fusion-encoder modes; this fusion in the backbone is more parameter-eficient than stacking fusion-specific layers and achieves consistent improvements across downstream vision-language tasks [7]. HelpingHands [87] augments a dual encoder with an object-aware decoder trained to predict hand and object positions from noisy, sparse hand–object-detector annotations, improving object grounding while requiring only RGB at inference and achieving strong zero-shot transfer on EPIC-KITCHENS-100 multi-instance retrieval and EGTEA [87].

## 5.2.2 Captioning-Based Methods

Captioning-based methods use natural-language generation as both supervision and output. LaViLa [23] repurposes a pretrained large language model as a visually-conditioned narrator that densely pseudo-labels Ego4D clips, together with a rephraser for textual diversity; the contrastively learned embedding gains an absolute +10.1% on EGTEA classification and +5.9% on EPIC-KITCHENS 100 multi-instance retrieval, and a model trained on only half the Ego4D narrations outperforms a baseline trained on the full set [23]. Vid2Seq [22], though not egocentric-specific, established the dense-captioning paradigm relevant to first-person streams: it augments a language model with special time tokens to jointly predict event boundaries and captions in a single sequence, pretrained on the 18M-video YT-Temporal-1B corpus using transcribed-speech boundaries as pseudo-events [22]. These narration-as-supervision approaches directly target the temporal-synchronization and textdiversity weaknesses of scraped-narration pretraining.

## 5.2.3 Action-Centric Methods

Action-centric models optimize directly for the verb–noun action structure central to egocentric tasks. Beyond the recognition and anticipation methods of Section 4, recent vision-language work exploits the verb–noun decomposition explicitly. The EgoNCE++ objective [11] can be viewed as an action-centric pretraining criterion that targets the verb-understanding deficit by mining hard-negative captions that alter the verb. EgoHOD [88] complements this direction by generating fine-grained hand–object dynamics narrations and training a lightweight motion adapter, directly targeting interaction-sensitive representation learning.

## 5.2.4 Multimodal and Embodied Extensions

This group covers multimodal wearable models together with embodied extensions that use egocentric perception for assistance or control. ImageBind [89] is a general multimodal foundation model, rather than an egocentric-specific architecture, but its joint embedding across image, text, audio, depth, thermal, and inertial measurements is directly relevant to multimodal egocentric sensing.

Vinci [8] is a real-time embodied assistant built on an egocentric vision-language model for portable devices. It operates continuously to answer queries about current and historical observations, plan tasks from past interactions, and generate step-by-step visual demonstrations, with audio-based hands-free interaction.

EgoMimic [4] scales imitation learning by treating egocentric human video, captured using Project Aria glasses with 3D hand tracking, and robot data as complementary embodied demonstrations. It co-trains a unified policy and reports that one hour of additional human hand data can provide greater benefit than one hour of additional robot data.

EgoVLA [90] extends egocentric vision-language modeling to embodied control. It trains a visionlanguage-action model on egocentric human videos to predict future wrist and hand motion and retargets these predictions through inverse kinematics to a bimanual humanoid. Pretraining on human video improves both in-domain and out-of-domain generalization on the proposed 12-task Ego Humanoid Manipulation Benchmark, using 100 demonstrations per task [90].

Together, these models represent the transition from multimodal egocentric perception and assistance toward embodied action.

## 5.3 Comparative Analysis of Existing Methods

Table 5 consolidates the models discussed below. These model groups trace a clear trajectory from representation learning toward embodied agency, with distinct tradeofs. Retrieval-based models— EgoVLP, EgoVLPv2, and HelpingHands—are the most eficient and excel at alignment tasks such as EgoMCQ and EPIC-KITCHENS-100 multi-instance retrieval, but their contrastive objectives are precisely what the diagnostic studies expose as brittle: by rewarding coarse video–text matching, they develop the noun-biased, temporally shallow representations that fail EgoHOIBench’s verb substitutions [11] and EgoSchema’s long-horizon reasoning [28]. Captioning-based models, LaViLa and Vid2Seq, address the data-quality side by generating dense, synchronized, and diverse narration, which improves both retrieval and downstream recognition [23]; their generative backbones also enable open-ended question answering that pure dual encoders cannot support. Action-centric methods such as EgoNCE++ and EgoHOD target the verb–noun interaction structure directly. Multimodal models such as ImageBind and Vinci broaden sensing and assistance, whereas embodied extensions such as EgoMimic and EgoVLA learn policy or action outputs for deployment.

Backbones and data scale vary accordingly: retrieval and captioning models converge on TimeSformer or ViT video encoders contrastively trained on Ego4D-derived corpora such as EgoClip’s 3.8M pairs [6], while embodied policy models such as EgoMimic and EgoVLA train on combinations of human video and robot demonstrations and predict action or policy outputs rather than text alone. Eficiency spans orders of magnitude, from EgoDistill’s roughly 200× reduction in multiply– adds [30] to multi-billion-parameter models that still underperform humans on EgoSchema [28]. The persistent gaps identified in Section 5.1—weak interaction understanding, reliance on spatial shortcuts over temporal reasoning, active-object ambiguity, and motion-induced feature corruption— remain only partially addressed: temporal objective redesign closes much of the temporal gap [9] and EgoNCE++ much of the interaction gap [11], but no single model yet unifies robust fine-grained interaction understanding, long-horizon temporal reasoning, and real-time embodied action. Closing that gap is the central open challenge on the path from hand–object interaction to embodied AI.

Table 5: Comparison of representative egocentric vision-language models and relevant multimodal or embodied extensions discussed in Section 5.2. “EK-100 MIR” denotes EPIC-KITCHENS-100 multi-instance retrieval; “cls.” denotes classification.
<table><tr><td>Model</td><td>Category</td><td colspan="2">Backbone data EgoClip</td><td colspan="2">Key idea EgoNCE action-</td><td colspan="2">Reported result</td></tr><tr><td>EgoVLP [6]</td><td>Retrieval</td><td colspan="2">TimeSformer; (3.8M)</td><td colspan="2">aware positives and scene-aware</td><td colspan="2">59.4% nDCG on EK-100 MIR; 90.7%/57.2% on inter-/intra-video EgoMCQ</td></tr><tr><td>EgoVLPv2 [7]</td><td>Retrieval</td><td colspan="2">TimeSformer-B RoBERTa; EgoClip</td><td colspan="2">negatives Gated modal inside the video</td><td colspan="2">Consistent gains on EgoTaskQA, CharadesEgo, and EK-100 MIR</td></tr><tr><td>HelpingHands [87]</td><td>Retrieval</td><td colspan="2">Dual encoder; Ego4D</td><td colspan="2">language backbones Object-aware decoder</td><td colspan="2">Strong zero-shot performance on EK-100 MIR and EGTEA</td></tr><tr><td>LaViLa [23]</td><td>Captioning</td><td colspan="2">TimeSformer; Ego4D nar- rations</td><td colspan="2">manipulated- object positions LLM-based Narrator</td><td colspan="2">+10.1% on EGTEA cls.; +5.9% on EK-100 MIR</td></tr><tr><td>Vid2Seq [22]</td><td></td><td colspan="2">Language model + time</td><td colspan="2">Rephraser erate dense and diverse narrations Time tokens</td><td colspan="2">State-of-the-art dense captioning</td></tr><tr><td>EgoHOD [88]</td><td></td><td colspan="2">tokens; YT-Temporal-1B EgoVideo 十</td><td colspan="2">jointly predict event boundaries and captions</td><td colspan="2">performance; not egocentric- specific +6.3% on EK-100 MIR; +16.3% on</td></tr><tr><td></td><td>Action-centric</td><td>adapter; Ego4D</td><td></td><td colspan="2">HOI-dynamics narrations erated using detector and an</td><td colspan="2">EGTEA in zero-shot evaluation</td></tr><tr><td>EgoVLA [90]</td><td>Embodied exten- sion</td><td>VLA model; human video</td><td>egocentric</td><td colspan="2">LLM Wrist and MANO action representa- tion with inverse- kinematics retar-</td><td colspan="2">Evaluated on the 12-task Ego Hu- manoid Manipulation Benchmark</td></tr><tr><td>ImageBind [89]</td><td>Multimodal foun- dation</td><td>Six modalities; paired data</td><td>image-</td><td colspan="2">geting Jointly aligns im- age, text, audio, depth, thermal,</td><td colspan="2">Emergent cross-modal retrieval across multiple modalities</td></tr><tr><td>Vinci [8]</td><td>Multimodal assis- tant</td><td>Egocentric portable-device</td><td>VLM; set-</td><td colspan="2">and IMU signals Real-time as- sistance with</td><td colspan="2">Real-time hands-free assistant pro- totype</td></tr><tr><td>EgoMimic [4]</td><td>Embodied exten-</td><td>ting Policy model; Aria and</td><td></td><td colspan="2">memory and visual demonstra- tion generation Co-trains human</td><td colspan="2">One hour of human hand data can</td></tr><tr><td></td><td>sion</td><td>robot demonstrations</td><td></td><td colspan="2">and robot data as complementary embodied demon- strations</td><td colspan="2">provide greater benefit than one hour of robot data</td></tr></table>

## 6 Hand–Object Interaction Understanding in Egocentric Videos

The first-person camera frames the world through the wearer’s hands. Where third-person video treats hands as one body part among many, egocentric video places them at the center of nearly every informative moment: the hands are what move, what occlude, and what carry the wearer’s intent into the scene. This makes hand–object interaction (HOI) understanding the empirical core of egocentric perception, and the natural hinge between low-level visual parsing and the languagegrounded, embodied reasoning that the rest of this survey pursues. This section traces that hinge from the bottom up: why hands are foundational rather than peripheral (Section 6.1), how the field detects and reconstructs them (Section 6.2), how it identifies and reasons about the objects they engage (Section 6.3), how the interaction unfolds as a temporal process from reach to release (Section 6.4), and where current understanding still breaks down (Section 6.5).

## 6.1 Why Hands Matter in Egocentric Understanding

In first-person video the hands are not merely visible—they are the privileged signal. Because the camera is worn and its motion is driven by the wearer’s attention, the hands and the objects they manipulate tend to occupy the center of the frame, and reading an action is largely a matter of reading how the hands engage objects [12, 39]. This is why the dominant egocentric benchmarks are organized around manipulation: EPIC-KITCHENS-100 decomposes every action into a (verb, noun) pair grounded in hand activity [18], and Ego4D dedicates an entire benchmark suite, Hands and Objects, to the moments in which hands change the state of the world [10].

The dependence runs deeper than annotation convenience: hand understanding measurably drives downstream performance. AssemblyHands demonstrates this directly, showing that improving the quality of estimated 3D hand poses yields a corresponding improvement in egocentric action recognition [91]. Hands are therefore best treated not as a peripheral cue to be detected once, but as a foundational primitive whose fidelity propagates through the entire perception stack—from recognition and anticipation up to the vision-language and embodied models of Sections 5 and 10. The cost of this centrality is that hands are also the hardest thing in the frame to parse: they are small, fast, self-similar, frequently self-occluded, and routinely occlude the very objects that define the action [12]. The remainder of this section is, in efect, the story of how the field has confronted that dificulty.

## 6.2 Hand Detection and Tracking

Recovering the hands is the entry point to every downstream interaction task, and the field has moved through three clearly separable regimes: sparse 2D keypoints, parametric mesh models, and—most recently—transformer-based reconstructors that target the egocentric failure modes head-on. We follow that progression, treating the MANO hand model as the parametric bridge that connects the keypoint era to the mesh era.

## 6.2.1 Keypoint-Based Methods

Early RGB keypoint detectors beat occlusion with geometry rather than labels: Simon et al. [92] introduced multiview bootstrapping, triangulating dome detections into 3D and reprojecting them to relabel occluded views, yielding an occlusion-robust 21-keypoint detector. This underpins Open-Pose [93] and, later, the mobile-real-time MediaPipe Hands [94]. But sparse 2D keypoints cannot express contact, orientation, or articulation under occlusion—the quantities that matter for manipulation. MANO [95] closed this gap with a low-dimensional parametric hand mesh whose pose space spans a standard grasp taxonomy, becoming the near-universal output representation for mesh-based methods. Accurate 3D supervision then followed from InterHand2.6M [96] and, in the egocentric regime, AssemblyHands [35, 91] (3.0M images, 490K egocentric).

![](images/91bd1bec94d6e09ac5086d093efc79ba89c34fb486c6745678b8354d51937dc2.jpg)  
Figure 6: The 100DOH hand-object detector. For each hand it jointly predicts a bounding box, side (left/right), contact state (none, self, other-person, portable, or non-portable), and a link to the box of the object in contact, recovering the active object among many distractors. Illustration inspired by [26].

## 6.2.2 Transformer-Based Hand Modeling

The transformer replaced hand-crafted mesh regressors with attention over vertices and joints: METRO cast mesh recovery as masked sequence modeling for occlusion robustness [97], Mesh Graphormer added graph convolution for local surface structure [98], and HandOccNet built occlusion handling into the architecture by injecting features into occluded regions before regressing MANO [99].

The current frontier is scale and egocentric realism. HaMeR scales a plain ViT backbone with large in-the-wild data for robustness across viewpoints and occlusions [33], and WiLoR adds real-time localization and tracking from monocular video [100]. The 2024–2025 wave then targets specifically egocentric failure modes: WildHands corrects near-field perspective distortion using field-of-view cues, topping the ARCTIC egocentric split at a fraction of HaMeR’s size [36, 101], while HaWoR [102] and Dyn-HaMR [103] recover world-frame and 4D interacting-hand motion from a moving camera by coupling SLAM with hand priors. The trajectory is clear—from where the hands are, to their shape, to where and how they move in the world—precisely what embodied agents require.

## 6.3 Object-Centric Interaction Understanding

Recovering the hand is only half the problem; understanding the interaction requires identifying which object is being acted upon and how it afords being acted upon. These are the questions of

manipulated-object detection and afordance understanding.

## 6.3.1 Manipulated Object Detection

The defining challenge is that a frame holds many objects but only one or two are active. The foundational “100 Days of Hands” detector jointly predicts, per hand, its box, side, contact state, and a link to the in-contact object [26]; Hands23 enriched this with pixel-level segments and a finer touch/hold/use vocabulary [27], and EPIC-KITCHENS VISOR supplied dense egocentric hand/active-object masks under heavy occlusion [31].

A complementary thread anticipates the active object before contact—central to assistive applications. Furnari et al. [104] forecast the next-active object from trajectories; StillFast unified this into an end-to-end short-term anticipation model (next object, verb, time-to-contact) that topped the Ego4D leaderboard [105]; and ANACTO extends the idea [86]. Helping Hands feeds this back into representation learning, adding an object-aware decoder that grounds hands and objects while staying RGB-only at inference [87].

## 6.3.2 Afordance Understanding

Beyond which object lies how it can be acted upon—afordance—and egocentric video teaches it at scale. Nagarajan et al. [106] learn weakly-supervised interaction hotspots predicting where an object would be manipulated even at rest; EGO-TOPO lifts afordances from objects to a topological map of activity zones [107]; and AGD20K addresses the supervision bottleneck by transferring from abundant exocentric demonstrations to egocentric views, now the standard weakly-supervised benchmark [108].

## 6.4 Fine-grained Interaction Understanding

An interaction is not an instant but a process: the hand approaches, makes contact, closes into a grasp, manipulates the object through a state change, and releases. Understanding this process at fine temporal granularity is what separates genuine interaction understanding from object recognition. There is no single canonical taxonomy of these stages in the literature; we therefore organize them along the most empirically grounded axis available—Ego4D’s state-change timeline, which annotates every interaction with a pre-condition frame, a contact frame, a point-of-no-return (PNR), and a post-condition frame [10]—and overlay the contact-state vocabulary of 100DOH/Hands23 [26, 27] and the grasp taxonomy of Feix et al. [109] as the semantic dimension. The five stages below should be read as this lifecycle, not as a fixed label set.

## 6.4.1 Approach

The interaction begins before contact, as the hand reaches toward its target. This pre-contact phase is the domain of next-active-object anticipation and hand-trajectory forecasting: predicting which object will be engaged and where the hand is heading, from motion and scene context alone [86,

104, 105]. Afordance hotspots provide the spatial prior for this stage, indicating the likely landing point of the reach before the hand arrives [106].

## 6.4.2 Touch

Contact onset is the first observable transition of the interaction, marked explicitly in Ego4D as the contact frame and captured in 100DOH as the shift from a no-contact to an in-contact state [10, 26]. Detecting this transition reliably is non-trivial in the egocentric view, where the hand frequently occludes the contact point itself; pixel-level resources such as VISOR were built precisely to study hand–object boundaries through these contact events [31].

## 6.4.3 Grab

Once contact is established, the hand closes into a grasp, and the manner of grasping carries rich information about both object and intent. The canonical vocabulary is the GRASP taxonomy of Feix et al. [109], which consolidates prior work into 33 distinct static grasp types organized by opposition, virtual-finger assignment, and the power–precision spectrum. Hands23 operationalizes this for in-the-wild images by annotating grasp type alongside contact state [27], and the MANO pose space was itself constructed to span these grasps, which is why modern mesh reconstructors can express them [95].

## 6.4.4 Hold and Manipulation

The manipulation phase—holding and using the object to efect a change—is where the interaction does its work. In Ego4D, this phase may include the point-of-no-return, which marks the frame at which an irreversible state change begins, while state-change classification categorizes the resulting change [10]. Here contact is dynamic rather than static, which dedicated datasets capture explicitly: ARCTIC records dexterous bimanual manipulation of articulated objects with dense, time-varying 3D contact [36]; ContactPose and GRAB provide hand–object and whole-body contact maps paired with pose [110, 111]; and HOI4D, Assembly101, and HOT3D supply temporally segmented, contactrich manipulation sequences in egocentric and multi-view settings [34, 35, 41].

## 6.4.5 Release

The interaction may terminate as the hand disengages. Ego4D’s post-condition frame documents the resulting object state, but it does not explicitly annotate release in every interaction [10]. Detecting release nevertheless closes the temporal loop and enables the system to anticipate the next interaction, making release detection important for parsing long, continuous first-person activities.

## 6.5 Limitations of Current HOI Understanding

Table 6: Comparison of representative hand–object interaction methods and resources, spanning hand recovery, manipulated-object reasoning, and interaction datasets (Section 6).
<table><tr><td>Method / Re- source</td><td>Category</td><td>Output / representa- tion</td><td>Egocentric fo- cus</td><td>Key contribution</td></tr><tr><td>100DOH [26]</td><td>Detection</td><td>Hand box, side, contact state, in-contact object</td><td>Internet-scale</td><td>Joint hand + active-object con- tact detection</td></tr><tr><td>Hands23 [27]</td><td>Detection</td><td>Pixel segments, grasp, contact vocabulary</td><td>In-the-wild</td><td>Richer 2D parsing (touch/hold/use)</td></tr><tr><td>MANO [95]</td><td>Parametric model</td><td>Low-dimensional pose + shape mesh</td><td></td><td>Near-universal hand output rep- resentation</td></tr><tr><td>HandOccNet [99]</td><td>Reconstruction</td><td>MANO mesh</td><td>Occlusion-heavy</td><td>Feature injection into occluded regions</td></tr><tr><td>HaMeR [33]</td><td>Reconstruction</td><td>3D hand mesh</td><td>In-the-wild</td><td>ViT-scale, robust across view- points</td></tr><tr><td>WildHands [101]</td><td>Reconstruction</td><td>3D hand pose</td><td>Everyday ego</td><td>FOV cues; best 3D pose on ARC- TIC ego split</td></tr><tr><td>HaWoR [102]</td><td>Reconstruction</td><td>World-frame hand mo- tion</td><td>Ego video</td><td>SLAM-decoupled recovery with FOV infilling</td></tr><tr><td>StillFast [105]</td><td>Anticipation</td><td>Next active object, verb, time-to-contact</td><td>Ego4D</td><td>Tops short-term anticipation leaderboard</td></tr><tr><td>VISOR [31]</td><td>Dataset</td><td>Dense masks + relations</td><td>hand/object Ego (EPIC)</td><td>Pixel-level under occlusion and state change</td></tr><tr><td>ARCTIC [36]</td><td>Dataset</td><td>Dense time-varying 3D contact</td><td>Ego + exo</td><td>Dexterous bimanual articulated manipulation</td></tr></table>

Table 6 consolidates the representative methods and resources surveyed in this section. Despite this maturity, current HOI understanding remains brittle in ways that matter for the embodied applications ahead. The most consequential limitation is conceptual rather than perceptual: even egocentric video–language models trained on millions of interaction clips show limited fine-grained interaction understanding. EgoHOIBench shows that these models latch onto nouns and scene context while failing systematically on fine-grained verb distinctions, mistaking what object is present for how it is being manipulated [11]. This verb–noun asymmetry is the empirical core of the gap this survey identifies, and it persists precisely because contrastive objectives reward coarse video–text matching over genuine interaction reasoning.

The perceptual limitations are equally concrete and are, tellingly, the same egocentric failure modes that the newest reconstruction methods were built to attack. Occlusion remains pervasive— hands occlude objects and themselves, which is why occlusion-robust architectures are a research thread of their own [99]. Perspective distortion of near-field hands degrades 3D estimation in everyday egocentric images [101], and hands routinely leave the narrow field of view entirely, forcing methods to infill motion they cannot observe [102, 103]. A second limitation is the supervision bottleneck: the highest-quality 3D contact and pose annotations depend on multi-view capture rigs unavailable in the wild, and even automated pipelines such as AssemblyHands’ presuppose such rigs [91], while afordance learning leans on weak supervision and cross-view transfer to escape the same constraint [108]. Finally, models trained on Western, kitchen-heavy footage generalize poorly across cultures, environments, and manipulation styles—a bias that compounds every limitation above.

Taken together, these gaps frame the central problem for the sections that follow: no current system unifies robust fine-grained interaction understanding, long-horizon temporal reasoning over the interaction lifecycle, and the world-frame hand recovery that embodied control demands. Closing that gap is the path from hand–object interaction to embodied AI.

## 7 Spatiotemporal Reasoning in Egocentric Video

The perception stack of Sections 6 and 4 tells us where the hands are, what objects they touch, and when contact begins and ends. None of that, by itself, is understanding. Understanding requires reasoning over the relations between those entities and over their evolution in time: that the knife is in the right hand, that the right hand is cutting the onion that sits on the board, and that this step follows peeling and precedes frying. This section surveys the modeling layer that sits on top of perception and performs exactly this reasoning. We argue throughout that explicit structure—scene graphs, hand–object interaction graphs, environment topologies, and the order- and memory-aware objectives that operate over them—provides a promising structured bridge from the low-level hand– object interaction discussed in Section 6 to the embodied reasoning of Section 10. We treat spatial reasoning (Section 7.1) and temporal reasoning (Section 7.2) in turn, show that their fusion in interaction-aware temporal models is where the field is converging (Section 7.3), and close with hard evidence that current vision-language models still largely fail at genuine temporal reasoning (Section 7.4).

## 7.1 Spatial Reasoning

Spatial reasoning in video has steadily moved away from dense, monolithic feature tensors toward explicit relational structure, in which a frame becomes a set of nodes (hands, objects, regions, places) connected by typed edges (contact, support, proximity, afordance). This shift matters for egocentric video in particular, because the first-person signal is fundamentally relational: an action is defined less by appearance than by which hand engages which object, and how.

## 7.1.1 Object Relations

The decisive idea is the spatio-temporal scene graph. Action Genome [112] reframed an action as a sequence of per-frame scene graphs (attention, spatial, and contact predicates) with dense annotations, enabling strong few-shot recognition. This relational view was absorbed into video transformers by ORViT [113], which injects object-region tokens from the earliest layers, and pushed to its limit by STLT [114], which strips appearance away entirely and reasons over object categories and bounding-box layouts alone—yet still recovers much of the discriminative signal for compositional actions. The common ancestor is Wang and Gupta’s space-time region graph [115], the template egocentric hand–object graphs later specialize.

## 7.1.2 Hand–Object Relations

In the egocentric setting the most informative edges are those between the wearer’s hands and the objects they manipulate. Interaction-reasoning networks make these edges first-class: a transformer interaction unit reasons jointly about each acting hand, its relation to the other hand, and the engaged objects, showing that explicit two-hand–object relational modeling is critical for fine-grained egocentric recognition [116]. Egocentric Object Manipulation Graphs [117] go further and build a graph encoding the contact and anticipated-contact state between each hand and the surrounding objects, then reason over the resulting node-state sequence to anticipate the next action—an explicit acknowledgement that the hands are the driving force of state change in manipulation activity. These methods make concrete the verb-centric structure that, as Section 7.4 will show, appearance-based models systematically miss.

## 7.1.3 Scene Context

Beyond the immediate grasp lies the structure of the surrounding environment and the longer activity. EGO-TOPO [107] parses untrimmed first-person video into a topological graph of activity-centric zones, modeling a video as a sequence of visits to functional places and using that graph for afordance discovery and long-term anticipation. Egocentric Action Scene Graphs (EASG) [118] unify the object- and scene-level views into a single temporally evolving representation: a dynamic graph with a camera-wearer node, verb nodes, and direct, active, and peripheral object nodes, grounded with bounding boxes across the pre-condition, point-of-no-return, and post-condition frames of each interaction (Fig. 7). Built as a manual extension of Ego4D, EASG defines a scene-graph-generation task and yields measurable downstream gains on action anticipation and activity summarization, and has since been extended into a video question-answering benchmark for graph-grounded reasoning [119]. The multi-view Home Action Genome [120], which couples a synchronized ego-view with dense scene-graph and hierarchical activity labels, provides the egocentric-adjacent bridge back to the Action Genome lineage.

## 7.2 Temporal Reasoning

If spatial reasoning organizes what relates to what, temporal reasoning organizes what follows what. Three problems dominate: learning representations sensitive to temporal order, carrying information across the long horizons of first-person video, and localizing the transitions that separate one action from the next.

## 7.2.1 Temporal Ordering

The foundational insight is that temporal order is itself a free supervisory signal. Shufle-and-Learn [121] trains a network to verify whether a triplet of frames is correctly ordered, yielding representations attuned to temporally varying structure without any labels. This idea has re-emerged at the center of modern egocentric post-training: rather than using order verification as pretext, recent work converts it into a reward. Xu et al. [9] contrast a model’s outputs on temporally ordered versus shufled frames to derive globally normalized, verifiable rewards that explicitly favor temporally coherent reasoning, lifting EgoSchema accuracy from a 20.4% base to 49.7% and suppressing the frame-level spatial shortcuts that ordinary correctness-based objectives reward. The lineage from a 2016 pretext task to a 2026 reinforcement-learning objective is the clearest evidence that order-awareness, far from being solved, remains the active frontier.

![](images/9fd51abf59969b017e3e8ce8735ea3298a643d2e518754ad3b6c1ad6425bf057.jpg)  
Figure 7: Egocentric Action Scene Graph (EASG). An interaction is represented as a temporally evolving graph rooted at the camera-wearer node. Verb (action) nodes connect to direct objects, while the manipulated object is linked via active edges to the acting hands. Peripheral objects provide contextual relations. The graph is grounded by bounding boxes on the pre-condition, pointof-no-return (PNR), and post-condition frames, capturing both spatial hand–object relations and their evolution over time. Illustration inspired by [118].

## 7.2.2 Long-Term Dependencies

Because egocentric streams run for minutes to hours, dense attention over their full extent is infeasible, and the field has converged on memory. MeMViT [37] caches a compressed memory of past clips so that attention can reference roughly 30× longer temporal context for only about 4.5% additional compute, processing long video online rather than as one giant input. MC-ViT [38] instead re-purposes a pretrained video transformer by fine-tuning it to attend to non-parametrically consolidated memories of past activations, achieving roughly an order of magnitude of memory compression and setting the state of the art on EgoSchema while outperforming far larger models. EgoSchema [28] itself quantifies why this horizon matters: its temporal-certificate sets have a median length near 100 seconds. In the original evaluation, the tested billion-parameter models scored below 33%, whereas humans reached approximately 76%. Memory, in short, is not an eficiency trick but a prerequisite for reasoning over the procedural structure of real first-person activity.

## 7.2.3 Action Transitions

Reasoning over a long activity requires both temporal grounding and transition localization. Ego4D [10] operationalizes these through two distinct tasks: natural-language query grounds a textual query to a temporal window, whereas the Hands-and-Objects benchmark localizes the point-of-noreturn frame and classifies the object state change across the pre- and post-condition frames. The latter directly represents irreversible action transitions and connects temporal segmentation to the hand–object interaction lifecycle of Section 6.4.

## 7.3 Interaction-Aware Temporal Modeling

The two threads above are most powerful when fused: when interaction structure is used to organize temporal reasoning rather than treated as a separate perception output. This is the conceptual hinge of the section, and the point at which graph-based reasoning most clearly carries the paper from perception toward embodiment. EASG [118] is one instantiation, in that its graphs are defined over the temporal spine of each interaction. AMEGO [122] is another and arguably the cleanest: from a single long egocentric video it constructs a semantic-free active memory of hand–object interaction tracklets and key locations (Fig. 8), so that arbitrary queries about sequencing, concurrency, and temporal grounding can be answered without reprocessing the video, and it introduces the Active Memories Benchmark of over 20K queries on which it substantially outperforms standard video question-answering baselines. At the representation level, modeling fine-grained hand– object dynamics during pretraining—through a hand–object detection pipeline, dynamics narrations, and a dedicated motion adapter—yields large zero-shot gains, including roughly +6.3% on EPIC-KITCHENS-100 multi-instance retrieval and +16.3% on EGTEA classification [88], confirming that interaction structure improves temporal representations and not merely recognition accuracy. Finally, interaction structure can steer computation itself: graph-structured frame selection builds an object-centric scene graph per clip and uses it to choose the few query-relevant keyframes worth attending to in long-video reasoning [123], a direct instance of graph-guided frame sampling. Across all of these, the pattern is identical—HOI structure is not the endpoint of perception but the scafold for temporal reasoning.

## 7.4 Limitations of Existing Temporal Reasoning

Despite this progress, recent evidence indicates that current vision-language models often exhibit limited or shortcut-driven temporal reasoning, relying on spatial and appearance cues that may sufice on existing benchmarks. EgoTempo [124] makes this concrete and uncomfortable: a textonly language model with no video input already reaches 31.3% on EgoSchema, and a model given a single frame reaches about 51% on EgoSchema yet only 9.1% on EgoTempo, whose questions are constructed to be unanswerable without integrating information across time. Moving from singleto multi-frame input improves EgoTempo accuracy by roughly 4.3×, against only about 1.4× on prior datasets—a gap that exposes how little genuine temporal integration those prior datasets ever required, and the best model still trails human accuracy by more than 20 points. The interactionlevel analogue is EgoHOIBench [11], on which egocentric video-language models trained on millions of clips collapse under single verb or noun substitutions, revealing a representation biased toward recognizing objects over understanding manipulation. Xu et al. [9] trace the root cause to objectives that reward answer correctness without rewarding temporal consistency, allowing models to succeed without reasoning over order at all. Two conclusions follow for the rest of this survey. First, the apparent competence of current models is partly an artifact of benchmarks that reward bag-offrames recognition, so robust evaluation must isolate temporal and interaction reasoning explicitly, as EgoTempo and EgoHOIBench begin to do. Second, the structured, interaction-aware approaches of Section 7.3 are not merely one option among many but a principled response to a diagnosed failure—and closing the gap between recognition and genuine spatiotemporal reasoning is precisely what the transition to embodied intelligence in Section 10 demands.

![](images/94b529117b58f2389d53d91e044ed291fe7d415f02855596af9655e0f8c3f92a.jpg)  
Figure 8: Interaction-aware temporal modeling in egocentric video. Hand–object interaction tracklets extracted from long videos are organized into a structured active memory indexed by interaction events and locations. This representation supports eficient temporal reasoning—including sequencing, concurrency, temporal grounding, and retrieval—without reprocessing the full video. Interaction-aware representations and graph-guided frame selection further improve long-horizon reasoning by using hand–object structure as the scafold for temporal understanding. Illustration inspired by [122].

## 8 Graph-Based and Object-Centric Reasoning

Section 5 left us with a verdict: the strongest egocentric vision-language models recognize what is in front of the camera but not how the hands are acting on it. EgoHOIBench [11] pins the failure precisely—a single verb substitution is enough to fool models trained on millions of interaction clips—and the cause is structural, not incidental. A dense grid of patch tokens has no place to store the fact that this hand grasps that knife which cuts this onion; it can only hope the relation survives in the soup of attention weights, and it routinely does not.

Graph-based reasoning is a natural structured response, and the field has reached for it quickly. Make the relation a first-class object: nodes for hands and the objects they touch, typed edges for contact and manipulation, and a model that reasons over the graph rather than over the pixels. Our position in this section is that this response is correct in principle but, for egocentric video specifically, still mostly borrowed. Almost every mature method—the scene-graph generators, the object-centric encoders, the graph-guided frame samplers—was built and validated on framed, thirdperson footage, then transferred to the first-person setting on the assumption that a graph is a graph. It is not. The genuinely egocentric graph literature is small, recent, and disproportionately made of preprints and workshop papers, and where it works it often works by leaning on a large foundation model to paper over the construction problems that ego-motion creates. The structure is right; the egocentric foundations under it are thin. We organize the section to make that gap visible rather than to hide it: why graphs suit first-person video (Section 8.1), what the third-person scene-graph literature ofers and where it breaks on transfer (Section 8.2), the hand–object interaction graph as the one genuinely native egocentric construction (Section 8.3), graph structure as a controller over which frames a model even sees (Section 8.4) and over the representations it learns (Section 8.5), and the open problems that follow (Section 8.6).

## 8.1 Why Graphs for Egocentric Video?

The argument for graphs is not that they are elegant; it is that a first-person action is defined by a relation and only incidentally by an appearance. “Cutting an onion” and “holding an onion” are nearly identical pixel distributions separated entirely by the state of the hand–knife–onion relationship over time. A model that encodes that relationship implicitly will, under pressure, fall back on the easier cue—the onion—which is exactly the noun-shortcut bias the diagnostics of Section 5.1 exposed. An explicit graph may reduce reliance on this shortcut by representing the interaction relation and the active object as distinct structural elements.

Crucially, the egocentric setting is unusually friendly to this idea, and we think the field has under-stated how much. Three properties line up. The entity set is small and stable—a handful of hands and manipulated objects dominate nearly every informative frame—so a graph over them stays compact where a third-person crowd scene would explode. The same objects recur across minuteslong activities, so a structured, queryable graph amortizes across the whole video instead of being rebuilt frame by frame; this is the insight that makes AMEGO’s interaction-tracklet memory [122] practical on hour-long footage where dense processing is hopeless. And the graph coincides with supervision the community already pays for: EPIC-KITCHENS-100 labels every action as a (verb, noun) pair [18], and Ego4D’s Hands and Objects benchmark annotates precisely the object state changes that graph edges are meant to encode [10]. The conceptual machinery is inherited from the spatio-temporal graph—ST-GCN over body joints [125], the space-time region graph of Wang and Gupta [115]—but the fit to first-person data is tighter than it ever was for third-person video. In the egocentric case the graph is close to the native data model, not one abstraction among many. That, and not aesthetic preference, is why the rest of this section treats graphs as the principled bet.

## 8.2 Scene Graphs in Video Understanding

Most of what the field knows about generating scene graphs from video, it learned from third-person data, and the lineage is worth tracing because it is precisely what gets transferred—and what breaks on transfer. The anchor is Action Genome [112], which recast an action as a sequence of per-frame scene graphs with attention, spatial, and contact predicates and supplied dense annotations at the scale of hundreds of thousands of frames; it is the benchmark essentially all video scene-graph generation (VidSGG) still reports on, and that monoculture is itself a problem we return to below.

The methods themselves have improved steadily and genuinely. STTran [126] made dynamic scene-graph generation a transformer problem—spatial encoder for within-frame relations, temporal decoder for cross-frame ones—and set the baseline everyone still compares against. OED [127] did away with the brittle detect-then-classify pipeline, posing the whole task as one end-to-end set prediction over subject–object pair tokens and dropping the external trackers and handcrafted trajectories that earlier systems leaned on; this is a real architectural simplification, not a relabeling. Two moves then push past mere description. SceneSayer’s scene-graph anticipation [128] models how relationships evolve as a continuous latent process with neural ODEs/SDEs and forecasts future graphs—a genuinely forward-looking innovation that the egocentric literature has barely touched. HyperGLM [129] abandons the pairwise-edge assumption altogether for a hypergraph of higherorder interactions, fuses a spatial entity graph with a procedural causal one, and feeds the result into an LLM; its accompanying dataset deliberately spans third-person, egocentric, and drone views, which is the first serious acknowledgement from this community that viewpoint matters. A parallel and overdue correction targets the field’s severe long-tail bias: VISA [130] debiases visually and semantically and posts double-digit relative gains on rare predicates, conceding what the recall@Kon-Action-Genome leaderboard had been hiding—that these models were quietly ignoring most of the relationship vocabulary. A full account of the area is available in survey form [14].

Overall, this literature is strong but remains predominantly third-person, and its transfer to egocentric video is non-trivial. The framed, stable-camera assumption that lets STTran link objects cleanly across frames is exactly what head motion, truncation, and persistent hand occlusion destroy (Section 2.3). One of the few explicitly egocentric video scene-graph-generation pipelines is SAMJAM [131], and it is telling how it works: rather than train a VidSGG model on first-person data—which barely exists at scale—it stitches SAM2’s mask tracking to a Gemini VLM that proposes the graphs zero-shot, then binds predicted objects to tracked masks for temporal consistency on EPIC-KITCHENS. It is a clever proof of concept and we cite it as such, but it is also a tell: the egocentric community is reaching for frozen foundation models because the native data and methods to do this properly are not yet there. That absence, not SAMJAM’s results, is the real state of the art.

## 8.3 Hand–Object Interaction Graphs

If anything in this section is genuinely egocentric rather than borrowed, it is the hand–object interaction (HOI) graph: a scene graph whose privileged nodes are the wearer’s hands and whose load-bearing edges are the contact and manipulation relations binding hands to the objects they act on. This is the structure Section 6 identified as the empirical core of first-person perception, and it is where the egocentric literature has actually contributed rather than imported.

## 8.3.1 Graph Construction

Construction comes down to three decisions—what is a node, what is an edge, how are both obtained—and the egocentric field has converged on a sensible answer. Hands and active objects are nodes; edges carry the contact vocabulary (no contact, self-contact, contact with another person, portable-object contact, and non-portable-object contact) established by the 100DOH detector [26] and Hands23 [27] refined. Two construction philosophies then diverge, and the contrast is instructive. EGO-OMG [117] and EASG [118] build semantically rich graphs: EGO-OMG tracks contact and anticipated-contact state per hand, and EASG defines a full node taxonomy—camera-wearer, verb, and direct/active/peripheral object nodes—grounded with boxes and hand-extended from Ego4D, later turned into a QA benchmark for graph-grounded reasoning [119] (Fig. 7). The strength is expressiveness; the cost, which we should state plainly, is that this richness depends on dense manual annotation that does not exist at scale for egocentric video. AMEGO [122] takes the opposite bet and, in our reading, the more scalable one: a semantic-free graph of interaction tracklets and key locations that needs no relationship labels, scales to hour-long video, and is queried rather than classified (Fig. 8). The tension between annotation-hungry expressiveness and annotation-free scalability is the defining design axis of egocentric HOI graphs, and it is unresolved.

## 8.3.2 Graph Dynamics Over Time

A static graph is a snapshot; the understanding lives in how it changes. The field has a ready-made temporal spine for this—Ego4D’s interaction lifecycle of pre-condition, contact, point-of-no-return, and post-condition frames [10], developed in Section 6.4—and the strongest egocentric methods exploit it. EASG pins its graphs directly to the PRE/PNR/POST frames [118]; EGO-OMG reasons over the resulting node-state sequence to anticipate the next action [117]; AMEGO encodes dynamics as the birth, persistence, and death of tracklets so that sequencing and concurrency queries become operations on lifetimes [122]. But notice what egocentric work does not yet do: it tracks how graphs change, it does not forecast it. The third-person side already models edge evolution as a continuous latent process and predicts future graphs outright [128]. Porting that capability to HOI graphs— forecasting the next contact edge rather than detecting it after the fact—is, in our view, one of the clearest near-term opportunities the egocentric field is currently leaving on the table.

## 8.3.3 Graph-based Interaction Modeling

Constructing the graph is half the job; the model still has to reason over it, and the evidence here directly substantiates the section’s thesis. Interaction-reasoning networks make hand–object edges first-class via a transformer unit that jointly relates each acting hand, the other hand, and the engaged objects, and show that this explicit two-hand–object modeling is what fine-grained egocentric recognition actually needs [116]. ORViT [113] generalizes the lesson—object-region structure belongs inside the backbone, injected from the earliest layers, not appended afterward—and improves EPIC-KITCHENS-100 accordingly. The sharpest evidence is STLT [114], which throws away appearance entirely and reasons over object categories and bounding-box layouts alone, yet still recovers most of the discriminative signal for compositional actions. Read against Section 5.1, this result provides evidence that explicit layout dynamics carry useful information for compositional actions and that dense appearance-based models may underuse such relational structure. This is the concrete form of our claim—dense models ask what is present, graph models are built to ask how things relate, and on egocentric video the second question is the one that matters.

## 8.4 Graph-guided Frame Sampling

Graphs can also decide which frames a model ever processes—and on hour-long egocentric video this is not a side issue but a hard ceiling on quality. Every long-video VLM must discard most of its input, because accuracy degrades and latency grows with the frame budget, so the sampler quietly bounds everything downstream of it. The egocentric case is the worst case for naive sampling, which is exactly why it deserves attention here.

## 8.4.1 Random vs. Interaction-aware Sampling

Uniform sampling—every k-th frame—is a common default but can be poorly matched to the sparse and irregular interaction events found in first-person video. The informative moments (contact onset, point of no return, state change) are sparse and irregularly spaced, so uniform sampling spends its budget as readily on a motion-blurred head turn as on a grasp. The first real correction was to make sampling query-aware: SeViLA [132] turns an image-language model into a localizer that picks language-relevant keyframes and self-supervises without moment labels, and Frame-Voyager [133] goes further by ranking whole frame combinations according to the loss they induce in a downstream video-LLM, capturing inter-frame dependencies that per-frame scoring throws away. The finding the field has now established—and that the egocentric setting makes especially stark—is that on first-person video, which frames the model sees governs accuracy as much as how it processes them, because relevance is so unevenly distributed in time. The honest caveat: SeViLA and Frame-Voyager are general-video methods; the egocentric community inherits them rather than having produced them.

## 8.4.2 Segment-aware Sampling

Operating on graph-structured segments rather than isolated frames is where this turns principled. STEP [134] induces a spatio-temporal scene graph from raw video and uses it to self-generate multi-step reasoning supervision, letting structure decide what a video-LLM learns to attend to. The cleanest egocentric instance is FocusGraph [123]: a trainable scene-caption selector first picks query-relevant clips by reasoning over compact, graph-based textual captions—a structured scene description, not a stack of low-resolution frames—after which a training-free patchwise sparse-flow step retains keyframes within those clips. The payof is concrete: state-of-the-art accuracy on egocentric long-video QA (FindingDory, and the Ego4D-sourced HourVideo) at under one token per frame and a fraction of the latency of agent-based selectors. GraphVideoAgent [135] makes the same bet with a dynamic entity-relation graph memory steering iterative selection, with consistent EgoSchema gains. The recurring pattern is the one from Section 8.3: interaction structure is not the output of perception but the scafold that organizes it—here, the scafold that decides where to look. We note, though, that FocusGraph is a 2026 preprint leaning on a 7B VLM to write its graph captions; the result is promising and the dependence on a heavy frozen model is real.

## 8.5 Graph-enhanced Representation Learning

The most durable use of a graph is not at inference at all but during training—folding relational structure back into the learned features. The egocentric high point here is EgoHOD [88], and it is worth being precise about why it matters: it builds a hand–object detection pipeline, autogenerates fine-grained interaction-dynamics narrations with a language model, and trains a dedicated motion adapter on them, yielding zero-shot gains on the order of +6.3% on EPIC-KITCHENS-100 multi-instance retrieval and +16.3% on EGTEA classification. The significance is not the headline numbers but what they isolate: encoding interaction structure during pretraining improves the representation, not just a downstream classifier, which is the cleanest available evidence that the structure carries information the dense objective was discarding. The more general object-centric machinery points the same direction without quite landing in the egocentric setting: slot-based encoders such as SAVi [136] bind scene content to a few object slots, and ObjectPrompt [137] pulls object-centric features from a VLM without finetuning and retrieves the relevant objects for long-term anticipation on Ego4D and EGTEA Gaze+—complementing ORViT’s in-backbone object tokens [113]. Graph-contrastive self-supervision over frame- or region-level relational graphs [138] ofers a route to structure-aware features without dense labels. The pattern, and the limitation, is by now familiar: with the single exception of EgoHOD, the representation-learning toolkit is general-video stock, validated on egocentric benchmarks rather than designed for the first-person regime—and EgoHOD’s outsized gains are the best indication of how much is being left unclaimed by not designing for it from the start.

## 8.6 Open Challenges in Graph-based Reasoning

The thesis of this section has a corollary: if egocentric graph reasoning is the right idea executed on borrowed and immature foundations, then the open problems are not refinements but the loadbearing work that still has to be done. Four stand out.

Graphs do not survive ego-motion. The detectors and trackers that populate graph nodes were tuned on framed third-person video, and the head motion, blur, and truncation of first-person capture (Section 2.3) corrupt node identity and edge assignment directly. That both SAMJAM [131] and FocusGraph [123] resort to heavy foundation models or optical-flow heuristics just to hold their graphs together across motion is the symptom; robust, lightweight egocentric graph construction is the unmet need.

Length breaks the graph. A graph that adds a node per object instance and an edge per relation per frame is intractable over the minutes-to-hours horizons of real egocentric activity. AMEGO’s semantic-free tracklets [122] and FocusGraph’s compact textual graphs [123] are the only serious answers so far, and neither ofers a general account of how to compress, forget, and re-instantiate graph structure over long horizons.

The annotations are not there. The expressive graphs the field wants depend on dense supervision—Action Genome’s [112], EASG’s [118]—that is expensive in third-person video and barely exists for egocentric. This single fact explains the recent lurch toward zero-shot, foundation-model-driven construction, and whether weak and zero-shot supervision can match annotation-based quality remains unresolved.

The symbol–vector gap is still open. A graph is discrete and symbolic; a VLM reasons in continuous embeddings, and bolting one onto the other is unsolved. HyperGLM’s hypergraph-into-an-LLM [129] and STEP’s graph-guided self-training [134] are the first credible attempts, but no method yet unifies robust egocentric graph construction, long-horizon graph reasoning, and the open-ended language interface of a VLM in one system. Closing that gap is, we argue, the decisive step from hand–object interaction to the embodied intelligence of Section 10—and it is the step the egocentric field has not yet taken.

## 9 Prompting and Semantic Alignment in Vision-Language Models

The transition from task-specific egocentric models to general-purpose vision-language models (VLMs) shifts the central engineering problem. It is no longer “how do we design a network for action recognition,” but “how do we phrase a query, and how do we guarantee that the phrasing and the video actually refer to the same thing.” These two questions—how to prompt, and how to align—are inseparable in the egocentric setting, where the visual signal is cluttered, the camera is constantly moving, and the textual supervision is short, noisy, and frequently ambiguous. This section traces prompting from its hand-crafted origins to interaction-aware designs, and examines why semantic alignment, although extensively studied in third-person vision, remains particularly challenging for first-person video.

## 9.1 Prompt Engineering in Vision-Language Models

Prompting begins with CLIP [5]. Because CLIP synthesizes a classifier directly from natural language, the wording of the class description is not cosmetic—it is the classifier. Radford et al. showed that the template “a photo of a {label}” and, more strikingly, ensembling 80 context prompts improved ImageNet accuracy by several points over a single bare label [5]. This is the uncomfortable lesson that motivated everything after it: a model can be frozen and powerful, yet its usable accuracy hinges on a sentence a human guessed.

Hand-tuning that sentence does not scale, so the field replaced discrete words with learnable vectors. CoOp [139] introduced Context Optimization, modeling a prompt’s context tokens as continuous vectors trained end-to-end while CLIP stays frozen; with only sixteen labeled images per class it beat carefully engineered prompts by roughly fifteen percent on average across eleven datasets [139]. The weakness was equally clear: a single static prompt overfits the classes it was trained on. CoCoOp [140] answered this with a lightweight Meta-Net that generates an instance-conditional token per image, so the prompt adapts to what is actually in front of the camera—a small change that substantially improved generalization to unseen classes and domain shift. The same logic was then pushed into the vision branch by VPT [141], which prepends a handful of learnable tokens to a frozen ViT and tunes under one percent of parameters, and into both branches jointly by MaPLe [142], which couples vision and language prompts so the two modalities are adapted in step rather than in isolation.

Video forced a further generalization, because a prompt now has to address time as well as appearance. ActionCLIP [143] reframed action recognition as a “pre-train, prompt, fine-tune” matching problem using textual label prompts, and Ju et al. [144] showed that optimizing a few continuous prompt vectors is enough to adapt a frozen image-text model across recognition, localization, and retrieval. Vita-CLIP [145] took the idea to its natural conclusion with a unified multimodal scheme: global video-level prompts model the data distribution, local frame-level prompts provide per-frame conditioning, and a summary prompt condenses the clip, with text-side prompts added in parallel—letting a single frozen-backbone model remain competitive in the supervised regime while keeping the zero-shot capability that full fine-tuning destroys. The trajectory across this subsection is consistent and worth stating plainly: prompting moved from words humans choose, to text vectors models learn, to visual and temporal vectors models learn jointly. A broader catalogue of this lineage is given in the recent VLM survey of Zhang et al. [13].

## 9.2 Video–Text Semantic Alignment

Table 7: Egocentric video–text semantic alignment approaches (Section 9.2) and the weakness each targets.
<table><tr><td>Method</td><td>Alignment strat- Target weakness egy</td><td></td><td>/ finding</td><td></td></tr><tr><td>EgoVLP [6]</td><td>EgoNCE ego-aware tives/negatives</td><td>mines Naive posi- ignores</td><td>verb/noun</td><td>InfoNCE shared seman-</td></tr><tr><td></td><td>EgoVLPv2 [7] Cross-modal fusion Late fusion limits in the backbone</td><td></td><td>tics video-text interac-</td><td></td></tr><tr><td>LaViLa [23]</td><td>LLM re-narration Sparse, (Narrator</td><td></td><td>tion + timed Ego4D narra-</td><td>unevenly</td></tr><tr><td></td><td>Rephraser) EgoNCE++ [11HOI-aware negative captions</td><td></td><td>tions hard- Object-over-action (verb) bias</td><td></td></tr><tr><td></td><td>EgoHOD [88] HOI-dynamics nar- Condensed</td><td></td><td>rations + motion tions lack fine detail</td><td>narra-</td></tr><tr><td>Wray al. [24]</td><td>adapter et Semantic evaluation</td><td></td><td>(many- One-caption-one- to-many) relevance video assumption is false</td><td></td></tr></table>

Prompting assumes the text and the video can be compared in a shared space. Building that space is the alignment problem. Table 7 summarizes the main egocentric alignment approaches and the weaknesses they target. In the general domain it is solved by large-scale contrastive pretraining: VideoCLIP [85] contrasts temporally overlapping video-text pairs against hard negatives retrieved by nearest-neighbor search, and Frozen-in-Time [25] trains a joint image-and-video dual encoder end-to-end for retrieval. Both work well precisely because their text—scraped captions and ASR—is loosely but broadly descriptive.

Egocentric video breaks that assumption, and the alignment literature is largely a sequence of responses to how it breaks. EgoVLP [6] made the first systematic attempt, contributing three pieces that are now standard reference points: EgoClip, a 3.8M clip-text pretraining set curated from Ego4D; EgoNCE, a contrastive objective that mines egocentric-aware positives (clips sharing verb or noun semantics) and negatives rather than treating every other clip as negative; and EgoMCQ, a multiple-choice benchmark built specifically to measure video-text alignment. EgoVLPv2 [7] then moved cross-modal fusion out of the fine-tuning head and into the backbone itself, so video and language interact during pretraining instead of only at the end.

The most influential recent direction attacks the text side directly. LaViLa [23] observes that Ego4D narrations are sparse and unevenly distributed in time, and repurposes a pretrained large language model as a visually-conditioned “Narrator” (with a “Rephraser” for diversity) to densely re-narrate long videos before contrastive training. The payof is concrete—absolute gains of 10.1% on EGTEA classification and 5.9% on EPIC-KITCHENS-100 multi-instance retrieval [23]—and it established a template the field still follows: when the supervision is weak, generate better supervision. Underlying all of this is a conceptual point made sharply by Wray et al. [24]: the standard instance-based assumption that one caption matches exactly one video is simply false. Many videos can be equally valid answers to one description and vice versa, so retrieval should be evaluated by semantic relevance rather than instance identity. This reframing matters far beyond evaluation, because the many-to-many structure it describes is the root of the ambiguity discussed next.

## 9.3 Semantic Ambiguity in Egocentric Videos

Egocentric narrations are short, telegraphic, and written by annotators who saw the action live; “#C C puts it down” is typical. This produces three compounding problems: temporal misalignment between narration and the moment it describes, vocabulary inconsistency across annotators, and genuine verb–noun polysemy where the same words cover visually distinct actions. The result is that a model can score well by exploiting the easy, stable part of the signal—the objects—while ignoring the hard, time-dependent part—the interaction.

Three recent diagnostic studies converge on exactly this failure from independent angles, and together they make the case that egocentric alignment is not yet solved. Xu et al. [11], with their EgoHOIBench benchmark, show that state-of-the-art EgoVLMs are easily flipped by substituting a single verb or noun in an interaction description, and attribute the gap to insuficient fine-grained supervision and a marked dificulty in recognizing verbs relative to nouns—a quantified objectover-action bias. Their proposed EgoNCE++, an asymmetric contrastive objective that mines HOI-aware negative captions on the video-to-text side while preserving an object-centric text-tovideo space, improves three EgoVLMs across seven downstream tasks. From the multimodal-LLM direction, EgoTempo [124] demonstrates that on existing egocentric QA benchmarks strong models score “remarkably high” using only the text of the question or a single frame—direct evidence that they lean on language and object priors rather than temporal content—and introduces a benchmark whose questions cannot be answered without integrating the whole video. EgoSchema [28] formalizes the long-form version of the same problem through its notion of a temporal certificate: its questions have a median certificate near 100 seconds. In the original evaluation, the tested billion-parameter models scored below 33%, whereas humans reached approximately 76%. Taken together, these results indicate that current VLM predictions can rely heavily on object identity and language priors rather than on fine-grained interaction and temporal understanding.

## 9.4 Interaction-aware Prompting and Conditioning

If the diagnosis is that models attend to objects and underuse interactions, the remedy is to make interaction structure an explicit conditioning signal. Interaction-aware prompting and conditioning inject hands, contact, and manipulated-object structure into the visual input, textual supervision, or auxiliary training objectives, rather than relying on a generic prompt to recover it.

The clearest instance on the supervision side is Helping Hands [87], which attaches an objectaware decoder that is trained to predict hand positions, manipulated-object positions, and object labels from the paired caption, while remaining RGB-only at inference; this explicit interaction signal improves zero-shot egocentric retrieval and classification. EgoHOD [88] pushes the same idea into the text itself: it couples a hand-object detector with an LLM to generate narrations that explicitly describe fine-grained hand-object dynamics—rather than the heavily condensed original narrations—and trains the EgoVideo model with a lightweight motion adapter on top, reporting zero-shot gains of 6.3% on EK-100 multi-instance retrieval and 16.3% on EGTEA classification [88]. On the visual-conditioning side, POV [146] uses frame-level interactive masking prompts that steer the model toward the hand-object region to transfer from third- to first-person views, and actionconditioned prompting [147] uses an LLM to expand each action into a descriptive, structured text prompt and aligns those knowledge concepts with the video for open-vocabulary recognition. A related line grounds detected objects through a VLM as a reasoning anchor for discovering novel actions [148], and HOI-Ref [149] adapts VLM prompts to refer explicitly to hands and objects in egocentric images.

The common thread is a reversal of the default assumption. Standard prompting asks the model to find the interaction; interaction-aware prompting tells the model where and what the interaction is, and lets it reason from there. Given the evidence in Section 9.3 that the interaction is exactly what generic models miss, this is not a refinement but a necessary correction—and, as the embodied applications in later sections demand faithful modeling of manipulation rather than scene gist, it is likely to become the dominant prompting paradigm for first-person video.

## 10 Applications Toward Embodied Intelligence

The methods surveyed so far—hand–object reasoning, spatial–temporal modeling, graph-based representations, and semantic alignment—are not ends in themselves. Their value is ultimately measured by what they enable an embodied agent to do: assist a person in real time, understand a skilled activity, run on a wearable device, or transfer a manipulation skill to a robot. A striking convergence has emerged across these goals. The same first-person sensing stack—Ego4D [10], Ego-Exo4D [16], and Project Aria smart glasses [17]—now feeds both human-facing assistants and robot-learning pipelines. This shared data substrate lets us view egocentric perception and embodied action as a single pipeline (capture → understanding → assistance or policy) rather than four disconnected application areas. The recurring obstacle that binds them is the gap between what a head-mounted camera observes a human hand doing and what a downstream agent—a robot gripper, or an assis tant inferring intent—can act on. We organize this section around that pipeline and return to the gap in Section 10.5.

## 10.1 Assistive Systems

The most direct application of egocentric VLMs is the real-time assistant: a system that watches the user’s first-person view and ofers timely guidance. Vinci [8] is a representative real-time assistant prototype, built on an egocentric video-language backbone and designed for portable devices. It processes a live stream, supports hands-free audio interaction, maintains a memory module for historical context, and can synthesize step-by-step how-to demonstrations. EgoLife [150] pushes the assistant toward week-scale, always-on use: six participants wore Aria glasses for a week, yielding a 300-hour multimodal dataset and the EgoLifeQA benchmark for long-context queries such as event recall and habit insight. Its EgoButler system couples an omni-modal VLM with retrieval-augmented memory (EgoRAG) to answer questions over ultra-long histories—a concrete blueprint for cognitive and accessibility assistance.

A second, complementary thread treats assistance as error correction rather than narration. HoloAssist [151] captures 166 hours of two-party interaction—a remote instructor watching a performer’s live egocentric feed and intervening verbally—over 350 instructor–performer pairs and 20 manipulation tasks, with annotations for mistakes and intervention type. This distinguishes passive guidance from active coaching, an important axis for AR tutors. The 2024 wave of procedural mistake-detection work sharpened the problem. PREGO [152] formulates online, open-set mistake detection by comparing the recognized current action against an anticipated one, introducing the Assembly101-O and Epic-tent-O benchmarks. EgoPER [153] contributes a richer error taxonomy (omission, addition, modification, slip, correction) grounded in task graphs, while CaptainCook4D [154] provides recipe-execution videos with deliberately induced errors. Together these define what a “mistake” formally is and expose how brittle current models remain on as-it-happens detection—the capability an interactive assistant most needs.

## 10.2 Human Activity Understanding

Assistance presupposes understanding not just what a person does but how well. Ego-Exo4D [16] is the foundational resource, pairing first- and third-person skilled-activity video with expert commentary and benchmarks for keystep recognition, proficiency estimation, and 3D ego-pose; its proficiency task—ranking novice-to-expert and localizing “good” actions and “tips”—is markedly harder than action labels, and submissions remain short of human agreement. Procedural understanding has deeper roots in Assembly101 [35] and AssemblyHands [91], and a recurring finding is that where the model looks matters: a stabilized hand-centric crop yields +22% Top-1 on Ego-Exo4D keystep recognition [29], reinforcing the survey-wide theme that hands are the most informative region.

## 10.3 AR/VR and Wearable AI

Deploying egocentric AI capabilities depends on wearable platforms that combine multimodal sensing with eficient machine-perception tools. Project Aria [17] provides a research platform for capturing multimodal egocentric data and supports associated perception capabilities. Aria Gen 2 [155] extends this platform with richer sensing, on-device computation, and real-time and ofline perception outputs, including SLAM and hand tracking. On the perception side, HOT3D [41] demonstrates that multi-view egocentric capture can improve 3D hand and 6-DoF object tracking and support transfer across diferent headset configurations.

Despite these advances, wearable deployment exposes a substantial reasoning and eficiency gap.

In the original EgoSchema evaluation [28], the tested billion-parameter models scored below 33%, whereas humans achieved approximately 76%. Closing long-horizon reasoning gaps under the limited power, memory, and latency budgets of wearable devices, rather than relying on server-scale computation, therefore remains a central challenge for always-on AR/VR assistants.

## 10.4 Human–Robot Collaboration

The most ambitious use of egocentric video is teaching robots to manipulate by watching people.   
This line is best read as a spectrum defined by how much robot data each method requires.

Representation transfer sits at one end. R3M [156] pretrains a visual representation on Ego4D via time-contrastive and video–language objectives, improving downstream manipulation success by over 20% relative to training from scratch and over 10% relative to CLIP/MoCo features—the seminal evidence that human egocentric video yields transferable manipulation priors.

Generalist vision–language–action (VLA) models occupy the robot-data-heavy middle. RT-2 [157] introduced the VLA formulation, representing actions as text tokens and co-training a VLM on robot trajectories and web data; OpenVLA [158] made the approach open and eficient, with a 7B model trained on 970K demonstrations that surpasses the 55B RT-2-X by 16.5% absolute success across 29 tasks; and π<sub>0</sub> [159] pairs a VLM with a flow-matching action expert for high-frequency dexterous control. These set the baselines against which egocentric-trained policies are measured.

Egocentric-driven policies reduce the robot-data requirement by treating human first-person video as a first-class training source. EgoMimic [4] co-trains a single policy on Aria-captured human demonstrations and robot demonstrations, using a low-cost manipulator that minimizes the kinematic gap to human hands. EgoVLA [90] trains a VLA on egocentric human video in a unified action space (wrist pose plus MANO hand parameters), converting human actions to robot actions through inverse kinematics and retargeting, then fine-tuning on a few robot demonstrations. At the far end, EgoZero [160] learns from Aria glasses with zero robot data, reporting 70% zero-shot success across seven tasks with only twenty minutes of data collection per task via a morphology-agnostic state representation. Where human demonstrations are scarce, DexMimicGen [161] amplifies them, synthesizing 21K bimanual trajectories from 60 source demonstrations. Read together, these works trace a clear trajectory toward learning dexterous manipulation primarily from human video, with robot data as a thin alignment layer.

## 10.5 Open Challenges for Embodied Deployment

Despite rapid progress, several obstacles recur across all four application areas. The foremost is the embodiment gap: a human hand and a robot gripper difer in morphology, degrees of freedom, and contact dynamics, so even perfectly understood human demonstrations do not map cleanly onto robot actions. Closely related is the visual domain gap—diferences in camera height, viewpoint, and field of view between a head-mounted human perspective and a robot’s sensors—which limits naive representation transfer. For real-time assistants and on-device wearables, latency and compute remain hard constraints: the long-form reasoning gap exposed by EgoSchema [28] must be closed within the power budget of glasses, not a datacenter. Skewed and subjective labels, evident in the proficiency setting of Ego-Exo4D [16], complicate both training and evaluation of skill-aware systems. Finally, always-on egocentric capture raises privacy concerns that the field’s own forward-looking analyses flag as a precondition for real-world adoption [3]. Progress on embodied intelligence will depend less on any single benchmark than on jointly addressing this cluster of perception, transfer, eficiency, and trust problems.

## 11 Open Challenges and Future Directions

Every section of this survey has ended at the same wall. Section 5 showed that the strongest egocentric vision-language models recognize what is in the frame but not how the hands act on it; Section 7 showed that they exploit spatial shortcuts in place of genuine temporal reasoning; Section 8 showed that the structural fix—graphs—is still built on borrowed, third-person foundations; and Section 10 showed that even when perception works, the gap to embodied action remains open. These are not six unrelated complaints. They are facets of a single deficit: current models perceive first-person video as a bag of frames and objects, when the signal is fundamentally about relations unfolding in time. This section lays out where the field must go, organized so that each challenge names both the diagnosed failure and the most promising line of attack. We are deliberately specific about what is solved, what is merely patched, and what remains genuinely open. Table 8 summarizes the six challenges below, each pairing a diagnosed failure with its supporting evidence and the most promising direction.

Table 8: Summary of open challenges and future directions (Section 11). Each row pairs a diagnosed failure with its supporting evidence and the most promising line of attack.
<table><tr><td rowspan=1 colspan=3>Challenge          Diagnosed failure         Evidence                        Promising direction</td></tr><tr><td rowspan=5 colspan=3>Temporal reason- Models  exploit   spa-  Text-only 31.3% and single-frame TGPO lifts a 3B model to 49.7%ing (Sec. 11.1)       tial/appearance shortcuts  ~51% on EgoSchema; single- on EgoSchema [9]; EgoVLM ex-rather than order            frame collapses to 9.1% onplores complementary GRPO-EgoTempo [124]                  based policy optimization [162]Interaction   un-  Verb-noun   asymmetry:  LaViLa scores 74.33% on nouns  Generate interaction super-vs. 46.61% on verbs (Ego- vision: EgoNCE++ (+34.02HOIBench) [11]                   percentage points in LaV-iLa++ verb accuracy) andEgoHOD [11, 88]Layout-only STLT still recov-Hypergraph-into-LLM, graph-</td></tr><tr><td rowspan=1 colspan=1>derstanding</td><td rowspan=1 colspan=1>nouns recognized, verbs</td></tr><tr><td rowspan=1 colspan=1>(Sec. 11.2)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Graph-enhanced</td><td rowspan=1 colspan=1>Dense tokens store no ex-</td></tr><tr><td rowspan=1 colspan=1>VLMs (Sec. 11.3)</td><td rowspan=1 colspan=1>plicit relation; symbol-</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>vector gap</td><td></td></tr><tr><td rowspan=1 colspan=1>Efficient    frame</td><td rowspan=1 colspan=1>Uniform sampling wastes</td><td></td></tr><tr><td rowspan=1 colspan=1>sampling</td><td rowspan=1 colspan=1>budget on uninformative</td><td></td></tr><tr><td rowspan=1 colspan=1>(Sec. 11.4)</td><td rowspan=1 colspan=1>frames</td><td rowspan=1 colspan=1>and efficiency [123, 132, 133]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Multimodal learn-</td><td rowspan=1 colspan=1>Video-only models ignore</td><td></td></tr><tr><td rowspan=1 colspan=1>ing (Sec. 11.5)</td><td rowspan=1 colspan=1>IMU, gaze, audio, and</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SLAM</td><td></td></tr><tr><td rowspan=1 colspan=1>Embodied deploy-</td><td></td><td></td></tr></table>

## 11.1 Better Temporal Reasoning

One of the clearest diagnostic results is that a language model with no video input scores 31.3% on EgoSchema, while a model given one frame reaches about 51% [124]. These findings indicate that EgoSchema permits substantial performance through non-temporal shortcuts and that leading models may rely less on temporal integration than their aggregate scores suggest. EgoTempo [124] makes the point unavoidable: the same single-frame model that reaches 51% on EgoSchema col lapses to 9.1% on questions constructed to be unanswerable without integrating the whole video. EgoSchema [28] had already quantified the horizon with its temporal-certificate sets—a median near 100 seconds, far longer than any prior dataset—and the sub-33% accuracy of the models tested in the original EgoSchema evaluation against roughly 76% for humans is the size of the gap that remains.

The promising result is that objective design appears to be an important factor beyond model scale. The clearest evidence is that order-aware reinforcement learning, which contrasts a model’s outputs on temporally ordered versus shufled frames to reward temporally coherent reasoning, lifts a 3-billion-parameter backbone to 49.7% on EgoSchema—outperforming far larger baselines [9]. The lineage here, from Shufle-and-Learn’s 2016 pretext task to a 2026 reinforcement-learning reward (Section 7.2.1), is the clearest signal that order-awareness is the live frontier rather than a solved detail. We expect the next advances to come from verifiable-reward post-training—GRPO-style optimization applied directly to egocentric understanding, as early work without any supervised finetuning already suggests [162]—coupled with the long-context memory mechanisms of Section 7.2.2. The open question is no longer whether models can be made to reason over time, but whether certificate-long horizons can be reached under the compute budget of a wearable rather than a server.

## 11.2 Interaction-aware Video Understanding

If temporal reasoning is the survey’s headline failure, the verb–noun asymmetry is its root cause. EgoHOIBench [11] reduced the problem to a single, reproducible number: substitute one verb in a caption and state-of-the-art egocentric models trained on millions of interaction clips flip their answer, with LaViLa scoring 74.33% on nouns but only 46.61% on verbs. A model that knows the onion but not the cutting has not understood the interaction; it has recognized the scene. This is exactly the deficit that defines first-person video, because an action there is a relation between hand and object, not an appearance (Section 6.1).

The corrective is to make the interaction an explicit training signal rather than something the model is left to infer. The two most convincing demonstrations both work by generating the supervision that scraped narrations lack: the EgoNCE++ objective mines hard-negative captions that alter the verb, raising the LaViLa++ variant’s verb accuracy by 34.02 percentage points [11], while Ego-HOD auto-generates fine-grained hand–object dynamics narrations and trains a dedicated motion adapter on them, yielding zero-shot gains of roughly +6.3% on EPIC-KITCHENS-100 multi-instance retrieval and +16.3% on EGTEA [88]. The principle is now established—when the supervision is weak, generate better supervision—and it is the template interaction-aware prompting (Section 9.4) extends. The open challenge is scale: these gains depend on detector-driven or LLM-generated annotations whose fidelity is itself limited by the egocentric failure modes of Section 6.5. Closing the verb gap at Ego4D scale, without a dense-annotation crutch, is the work that remains.

## 11.3 Graph-enhanced Vision-Language Models

Section 8 argued that graphs are a promising response to the noun-shortcut bias because they represent interactions and active objects as explicit relational elements, potentially reducing reliance on appearance shortcuts. The egocentric setting is particularly suitable for this approach because its entity set is relatively small, stable, and recurrent across minutes-long activities. Supporting evidence comes from models that reason over object categories and bounding-box layouts without appearance features while retaining substantial discriminative information for compositional actions [114]. This suggests that explicit layout dynamics capture relational information that dense models may otherwise underuse.

Three problems stand between this promise and a working system. First, the symbol–vector gap: a graph is discrete and symbolic, a VLM reasons in continuous embeddings, and uniting them is unsolved. The first credible attempts inject a hypergraph of higher-order interactions into an

LLM [129] and use a spatio-temporal scene graph to self-generate multi-step reasoning supervision [134], but neither unifies robust egocentric graph construction with the open-ended language interface of a VLM. Second, graphs do not survive ego-motion: the detectors that populate nodes were tuned on stable third-person footage, and the fact that the cleanest egocentric instances lean on frozen foundation models just to hold their graphs together across head motion [123, 131] is the symptom of a missing native solution. Third, the field tracks graph change but does not yet forecast it—the third-person literature already models edge evolution as a continuous latent process and predicts future graphs outright [128], and porting that capability to hand–object graphs, forecasting the next contact edge rather than detecting it after the fact, is the clearest near-term opportunity the egocentric field is leaving on the table.

## 11.4 Eficient Frame Sampling

On hour-long egocentric video, the sampler is not a preprocessing detail—it is a hard ceiling on quality, because every long-video model must discard most of its input, and uniform sampling spends its budget as readily on a motion-blurred head turn as on a grasp. The informative moments of first-person activity, contact onset and state change, are sparse and irregularly spaced (Section 6.4), which makes the egocentric case the worst case for naive sampling and the most rewarding for a learned alternative. The field has established that on first-person video, which frames the model sees governs accuracy as much as how it processes them.

The trajectory is from query-aware to structure-aware selection. SeViLA turns an image-language model into a keyframe localizer [132], and Frame-Voyager ranks whole frame combinations by the loss they induce downstream [133]; the more recent move is to let interaction structure decide where to look, as in the graph-caption-driven clip selection of FocusGraph, which reaches state-of-the-art egocentric long-video QA at under one token per frame [123]. The recurring pattern is the one from Section 7.3: HOI structure is not the output of perception but the scafold that organizes it—here, the scafold that decides what is worth attending to. The honest caveat is that the strongest selectors are still general-video methods the egocentric community inherits, and the most capable egocentric instance is a 2026 preprint leaning on a heavy frozen VLM [123]. A lightweight, native, interactionaware sampler that runs within a wearable’s budget does not yet exist, and it is a prerequisite for everything downstream of it.

## 11.5 Multimodal Egocentric Learning

A head-mounted device is not a camera; it is a sensor suite. The wearable platforms that now anchor the field—Project Aria [17] and its on-device successor [155]—stream RGB alongside eye gaze, inertial measurements, audio, and SLAM, yet the dominant models still treat egocentric understanding as a video-only problem and leave most of that signal unused. This is a missed opportunity precisely where the visual channel is weakest: inertial signals remain available when ego-motion degrades dense visual features, and the very head motion that blurs a frame is itself an attention signal about where the wearer is looking.

The foundations for fusing these streams are in place. ImageBind binds image, text, audio, depth, thermal, and inertial measurements into one embedding space using only image-paired data, enabling emergent cross-modal retrieval directly applicable to egocentric rigs [89], and EgoDistill turns the head-motion signal into an eficiency lever, reconstructing heavy video features from sparse frames plus lightweight inertial readings at roughly 200× fewer GFLOPs [30]. The frontier is weekscale, always-on, genuinely multimodal assistance: EgoLife wires six participants into Aria glasses for a week and couples an omni-modal model with retrieval-augmented memory to answer longcontext queries over the resulting 300-hour stream [150]. The open problem is no longer whether the modalities can be aligned, but how to fuse them adaptively—trusting the IMU when the camera is blurred, the gaze when the scene is cluttered—under real-time, on-device constraints rather than ofline on a server.

## 11.6 Toward Embodied Intelligence

Every thread in this survey converges on a single ambition: teaching machines to act by watching people act. Section 10 traced the spectrum of how much robot data this requires, from R3M’s Ego4D-pretrained representations that lift manipulation success by over 20% [156], through the robot-data-heavy generalist VLAs where OpenVLA’s 7B model surpasses the 55B RT-2-X by 16.5% across 29 tasks [158] and $\pi _ { 0 }$ adds a flow-matching action expert for dexterous control [159], to the egocentric-driven policies that treat human first-person video as a first-class training source. The endpoint of that spectrum is striking: EgoZero learns from smart glasses with zero robot data, reporting 70% zero-shot success across seven manipulation tasks [160], while EgoMimic [4] and EgoVLA [90] show that egocentric human demonstrations can complement robot data and reduce the amount of robot-specific supervision required.

What stands in the way is the cluster of gaps named in Section 10.5, and they are sobering precisely because they are not perceptual. The embodiment gap—a human hand and a robot gripper difer in morphology, degrees of freedom, and contact dynamics—means even perfectly understood demonstrations do not map cleanly onto actions. The visual domain gap between a head-mounted human view and a robot’s sensors limits naive transfer. And the long-form reasoning gap exposed by EgoSchema [28] must be closed within the power budget of glasses, not a datacenter. Two directions look most likely to move the field. The first is egocentric world models: simulators that predict future first-person observations from action, which turn passive video into a controllable environment for planning—an idea moving quickly from concept to realistic generation [163, 164]. The second is reliability and trust: as always-on capture moves toward real deployment, the privacy concerns that the field’s own forward-looking analyses flag as a precondition for adoption [3] become as load-bearing as accuracy, and emerging benchmarks that probe egocentric hallucination begin to measure whether a model’s confident answer can be believed at all [165]. Progress toward embodied intelligence will be measured less by any single leaderboard than by jointly closing this cluster of perception, transfer, eficiency, and trust problems—the final step on the path from hand–object interaction to embodied AI that has organized this entire survey.

## 12 conclusion

Egocentric video provides a distinctive window into human attention, object manipulation, and purposeful interaction with the surrounding environment. This survey has traced the development of egocentric video understanding from conventional recognition architectures to contemporary vision– language models, covering the field’s principal tasks, datasets, perceptual challenges, methodologica advances, and emerging applications. Particular attention has been given to hand–object interaction, spatiotemporal reasoning, frame and clip selection, multimodal learning, prompting, and semantic alignment. Within this broader landscape, graph-based and object-centric approaches have been examined as promising means of explicitly representing the spatial, temporal, and semantic relations that characterize first-person activities.

Across the reviewed literature, a consistent limitation emerges: current models generally identify visible objects more reliably than they understand actions, interaction stages, and evolving user intent. Their apparent competence can therefore depend heavily on appearance-based shortcuts or isolated frames rather than genuine integration of relations unfolding over time. Vision–language pretraining has substantially expanded the semantic capabilities of egocentric systems, but language supervision and model scale alone are unlikely to resolve these weaknesses. Further progress will require temporally grounded objectives, interaction-aware supervision, eficient and informative frame selection, adaptive multimodal fusion, and representations that preserve explicit relational structure. Graph-based reasoning is especially valuable in this regard, not as a replacement for dense video–language representations, but as a complementary mechanism for organizing hands, objects, actions, and scene context.

Future advances must also be evaluated beyond aggregate benchmark performance. More rigorous assessment of temporal reasoning, compositional generalization, cross-domain transfer, computational eficiency, privacy, and reliability will be essential for real-world deployment. Overall, the field is moving from recognizing the contents of first-person video toward understanding how interactions develop and how such understanding can support assistance and embodied action. Achieving this transition will depend on models that interpret egocentric experience robustly, eficiently, and trustworthily, thereby strengthening the connection between human activity understanding and embodied artificial intelligence.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used OpenAI’s ChatGPT to assist with language editing and improve the clarity and readability of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Funding

The authors received no specific funding for this work.

## Conflicts of Interest

The authors declare that they have no conflicts of interest.

## Data Availability

No new data were created or analyzed in this review article. Data sharing is not applicable to this article.

## References

1. Li X, Qiu H, Wang L, et al. Challenges and trends in egocentric vision: A survey. Mach. Intell. Res. 2026;23:1–33.

2. Rodin I, Furnari A, Mavroeidis D, and Farinella GM. Predicting the future from first person (egocentric) vision: A survey. Comput. Vis. Image Underst. (CVIU) 2021;211:103252.

3. Plizzari C et al. An outlook into the future of egocentric vision. Int. J. Comput. Vis. 2024;132:4880–936.

4. Kareer S, Patel D, Punamiya R, et al. EgoMimic: Scaling imitation learning via egocentric video. In: Proc. IEEE Int. Conf. Robot. Autom. (ICRA). 2025.

5. Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language supervision. In: Proc. Int. Conf. Mach. Learn. (ICML). 2021:8748–63.

6. Lin KQ, Wang AJ, Soldan M, et al. Egocentric video-language pretraining. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 35. 2022:7575–86.

7. Pramanick S, Song Y, Nag S, et al. EgoVLPv2: Egocentric video-language pre-training with fusion in the backbone. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2023:5285–97.

8. Huang Y et al. Vinci: A real-time embodied smart assistant based on egocentric visionlanguage model. 2024. arXiv: 2412.21080.

9. Xu Z, Qin T, Jin B, et al. Incentivizing temporal-awareness in egocentric video understanding models. 2026. arXiv: 2603.27184.

10. Grauman K, Westbury A, Byrne E, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:18995–9012.

11. Xu B, Wang Z, Du Y, Song Z, Zheng S, and Jin Q. Do egocentric video-language models truly understand hand-object interactions? In: Proc. Int. Conf. Learn. Represent. (ICLR). 2025.

12. Bandini A and Zarifa J. Analysis of the hands in egocentric vision: A survey. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 2023;45:6846–66.

13. Zhang J, Huang J, Jin S, and Lu S. Vision-language models for vision tasks: A survey. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 2024;46:5625–44.

14. Li H, Zhu G, Zhang L, et al. Scene graph generation: A comprehensive survey. Neurocomputing 2024;566:127052.

15. Kanade T and Hebert M. First-person vision. Proc. IEEE 2012;100:2442–53.

16. Grauman K et al. Ego-Exo4D: Understanding skilled human activity from first- and thirdperson perspectives. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2024.

17. Engel J et al. Project Aria: A new tool for egocentric multi-modal AI research. 2023. arXiv: 2308.13561.

18. Damen D, Doughty H, Farinella GM, et al. Rescaling egocentric vision: Collection, pipeline and challenges for EPIC-KITCHENS-100. Int. J. Comput. Vis. (IJCV) 2022;130:33–55.

19. Furnari A and Farinella GM. Rolling-unrolling LSTMs for action anticipation from first-person video. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 2021;43:4021–36.

20. Girdhar R and Grauman K. Anticipative video transformer. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:13505–15.

21. Krishna R, Hata K, Ren F, Fei-Fei L, and Niebles JC. Dense-captioning events in videos. In: Proc. IEEE Int. Conf. Comput. Vis. (ICCV). 2017:706–15.

22. Yang A, Nagrani A, Seo PH, et al. Vid2Seq: Large-scale pretraining of a visual language model for dense video captioning. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:10714–26.

23. Zhao Y, Misra I, Kr¨ahenb¨uhl P, and Girdhar R. Learning video representations from large language models. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:6586–97.

24. Wray M, Doughty H, and Damen D. On semantic similarity in video retrieval. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2021:3650–60.

25. Bain M, Nagrani A, Varol G, and Zisserman A. Frozen in time: A joint video and image encoder for end-to-end retrieval. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:1728–38.

26. Shan D, Geng J, Shu M, and Fouhey DF. Understanding human hands in contact at internet scale. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2020:9869–78.

27. Cheng T, Shan D, Hassen AS, Higgins REL, and Fouhey D. Towards a richer 2D understanding of hands at scale. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 36. 2023.

28. Mangalam K, Akshulakov R, and Malik J. EgoSchema: A diagnostic benchmark for very longform video language understanding. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 36. 2023:46212–44.

29. Chavis Z, Guy SJ, and Park HS. Improving keystep recognition in ego-video via dexterous focus. 2025. arXiv: 2506.00827.

30. Tan S, Nagarajan T, and Grauman K. EgoDistill: Egocentric head motion distillation for eficient video understanding. In: Adv. Neural Inf. Process. Syst. (NeurIPS). 2023.

31. Darkhalil A, Shan D, Zhu B, et al. EPIC-KITCHENS VISOR benchmark: VIdeo Segmentations and Object Relations. In: Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track. Vol. 35. 2022.

32. Tang H, Liang KJ, Grauman K, Feiszli M, and Wang W. EgoTracks: A long-term egocentric visual object tracking dataset. In: Adv. Neural Inf. Process. Syst. (NeurIPS) Datasets and Benchmarks Track. 2023.

33. Pavlakos G, Shan D, Radosavovic I, Kanazawa A, Fouhey D, and Malik J. Reconstructing hands in 3D with transformers. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2024:9826–36.

34. Liu Y, Liu Y, Jiang C, et al. HOI4D: A 4D egocentric dataset for category-level human-object interaction. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:21013– 22.

35. Sener F, Chatterjee D, Shelepov D, et al. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:21096–106.

36. Fan Z, Taheri O, Tzionas D, et al. ARCTIC: A dataset for dexterous bimanual hand-object manipulation. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:12943– 54.

37. Wu CY, Li Y, Mangalam K, et al. MeMViT: Memory-augmented multiscale vision transformer for eficient long-term video recognition. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:13587–97.

38. Balaˇzevi´c I, Shi Y, Papalampidi P, Chaabouni R, Koppula S, and H´enaf OJ. Memory consolidation enables long-context video understanding. In: Proc. Int. Conf. Mach. Learn. (ICML). 2024.

39. Kwon T, Tekin B, St¨uhmer J, Bogo F, and Pollefeys M. H2O: Two hands manipulating objects for first person interaction recognition. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:10138–48.

40. Sigurdsson GA, Gupta A, Schmid C, Farhadi A, and Alahari K. Charades-Ego: A large-scale dataset of paired third and first person videos. 2018. arXiv: 1804.09626.

41. Banerjee P, Shkodrani S, Moulon P, et al. HOT3D: Hand and object tracking in 3D from egocentric multi-view videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025.

42. Torralba A and Efros AA. Unbiased look at dataset bias. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2011:1521–8.

43. Karpathy A, Toderici G, Shetty S, Leung T, Sukthankar R, and Fei-Fei L. Large-scale video classification with convolutional neural networks. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2014:1725–32.

44. Tran D, Bourdev L, Fergus R, Torresani L, and Paluri M. Learning spatiotemporal features with 3D convolutional networks. In: Proc. IEEE Int. Conf. Comput. Vis. (ICCV). 2015:4489– 97.

45. Qiu Z, Yao T, and Mei T. Learning spatio-temporal representation with pseudo-3D residual networks. In: Proc. IEEE Int. Conf. Comput. Vis. (ICCV). 2017:5533–41.

46. Tran D, Wang H, Torresani L, Ray J, LeCun Y, and Paluri M. A closer look at spatiotemporal convolutions for action recognition. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2018:6450–9.

47. Carreira J and Zisserman A. Quo vadis, action recognition? A new model and the Kinetics dataset. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2017:6299–308.

48. Feichtenhofer C, Fan H, Malik J, and He K. SlowFast networks for video recognition. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2019:6202–11.

49. Feichtenhofer C. X3D: Expanding architectures for eficient video recognition. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2020:203–13.

50. Lin J, Gan C, and Han S. TSM: Temporal shift module for eficient video understanding. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2019:7083–93.

51. Wang X, Girshick R, Gupta A, and He K. Non-local neural networks. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2018:7794–803.

52. Donahue J, Hendricks LA, Guadarrama S, et al. Long-term recurrent convolutional networks for visual recognition and description. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2015:2625–34.

53. Ng JYH, Hausknecht M, Vijayanarasimhan S, Vinyals O, Monga R, and Toderici G. Beyond short snippets: Deep networks for video classification. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2015:4694–702.

54. Shi X, Chen Z, Wang H, Yeung DY, Wong WK, and Woo WC. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 28. 2015:802–10.

55. Li Z, Gavrilyuk K, Gavves E, Jain M, and Snoek CGM. VideoLSTM convolves, attends and flows for action recognition. Comput. Vis. Image Underst. (CVIU) 2018;166:41–50.

56. Srivastava N, Mansimov E, and Salakhutdinov R. Unsupervised learning of video representations using LSTMs. In: Proc. 32nd Int. Conf. Mach. Learn. (ICML). Vol. 37. 2015:843– 52.

57. Singh S, Arora C, and Jawahar CV. First person action recognition using deep learned descriptors. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2016:2620–8.

58. Ma M, Fan H, and Kitani KM. Going deeper into first-person activity recognition. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2016, pp. 1894–. 1903:1894–903.

59. Sudhakaran S, Escalera S, and Lanz O. LSTA: Long short-term attention for egocentric action recognition. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2019:9954– 63.

60. Simonyan K and Zisserman A. Two-stream convolutional networks for action recognition in videos. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 27. 2014:568–76.

61. Feichtenhofer C, Pinz A, and Zisserman A. Convolutional two-stream network fusion for video action recognition. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2016, pp. 1933–. 1941:1933–41.

62. Feichtenhofer C, Pinz A, and Wildes RP. Spatiotemporal residual networks for video action recognition. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 29. 2016.

63. Wang L, Xiong Y, Wang Z, et al. Temporal segment networks: Towards good practices for deep action recognition. In: Proc. Eur. Conf. Comput. Vis. (ECCV). 2016:20–36.

64. Zhou B, Andonian A, Oliva A, and Torralba A. Temporal relational reasoning in videos. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 11205. LNCS. 2018:831–46.

65. Girdhar R, Ramanan D, Gupta A, Sivic J, and Russell B. ActionVLAD: Learning spatiotemporal aggregation for action classification. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2017:971–80.

66. Kazakos E, Nagrani A, Zisserman A, and Damen D. EPIC-Fusion: Audio-visual temporal binding for egocentric action recognition. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2019:5491–500.

67. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In: Proc. Int. Conf. Learn. Represent. (ICLR). 2021.

68. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 30. 2017:5998–6008.

69. Touvron H, Cord M, Douze M, Massa F, Sablayrolles A, and J´egou H. Training data-eficient image transformers & distillation through attention. In: Proc. Int. Conf. Mach. Learn. (ICML). Vol. 139. 2021:10347–57.

70. Liu Z, Lin Y, Cao Y, et al. Swin Transformer: Hierarchical vision transformer using shifted windows. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:9992–10002.

71. Bertasius G, Wang H, and Torresani L. Is space-time attention all you need for video understanding? In: Proc. Int. Conf. Mach. Learn. (ICML). Vol. 139. 2021:813–24.

72. He K, Chen X, Xie S, Li Y, Doll´ar P, and Girshick R. Masked autoencoders are scalable vision learners. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:16000–9.

73. Tong Z, Song Y, Wang J, and Wang L. VideoMAE: Masked autoencoders are data-eficient learners for self-supervised video pre-training. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 35. 2022:10078–93.

74. Feichtenhofer C, Fan H, Li Y, and He K. Masked autoencoders as spatiotemporal learners. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 35. 2022:35946–58.

75. Wang L, Huang B, Zhao Z, et al. VideoMAE V2: Scaling video masked autoencoders with dual masking. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:14549– 60.

76. Arnab A, Dehghani M, Heigold G, Sun C, Luˇci´c M, and Schmid C. ViViT: A video vision transformer. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:6836–46.

77. Fan H, Xiong B, Mangalam K, et al. Multiscale vision transformers. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:6824–35.

78. Li Y, Wu CY, Fan H, et al. MViTv2: Improved multiscale vision transformers for classification and detection. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:4804–14.

79. Liu Z, Ning J, Cao Y, et al. Video Swin Transformer. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:3202–11.

80. Patrick M, Campbell D, Asano YM, et al. Keeping your eye on the ball: Trajectory attention in video transformers. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 34. 2021:12493–506.

81. Wei C, Fan H, Xie S, Wu CY, Yuille A, and Feichtenhofer C. Masked feature prediction for self-supervised visual pre-training. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:14668–78.

82. Li K, Wang Y, Zhang J, et al. UniFormer: Unifying convolution and self-attention for visual recognition. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 2023;45:12581–600.

83. Yan S, Xiong X, Arnab A, et al. Multiview transformers for video recognition. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:3333–43.

84. Ryoo MS, Piergiovanni A, Arnab A, Dehghani M, and Angelova A. TokenLearner: Adaptive space-time tokenization for videos. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 34. 2021:12786–97.

85. Xu H, Ghosh G, Huang PY, et al. VideoCLIP: Contrastive pre-training for zero-shot video-text understanding. In: Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP). 2021:6787– 800.

86. Thakur S, Beyan C, Morerio P, Murino V, and Bue AD. Anticipating next active objects for egocentric videos. IEEE Access 2024;12:51530–46.

87. Zhang C, Gupta A, and Zisserman A. Helping hands: An object-aware ego-centric video recognition model. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2023:13901–12.

88. Pei B, Huang Y, Xu J, et al. Modeling fine-grained hand-object dynamics for egocentric video representation learning. In: Proc. Int. Conf. Learn. Represent. (ICLR). 2025.

89. Girdhar R, El-Nouby A, Liu Z, et al. ImageBind: One embedding space to bind them all. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:15180–90.

90. Yang R et al. EgoVLA: Learning vision-language-action models from egocentric human videos. 2025. arXiv: 2507.12440.

91. Ohkawa T, He K, Sener F, Hodan T, Tran L, and Keskin C. AssemblyHands: Towards egocentric activity understanding via 3D hand pose estimation. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:12999–3008.

92. Simon T, Joo H, Matthews I, and Sheikh Y. Hand keypoint detection in single images using multiview bootstrapping. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR). 2017:4645–53.

93. Cao Z, Hidalgo G, Simon T, Wei SE, and Sheikh Y. OpenPose: Realtime multi-person 2D pose estimation using part afinity fields. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 2021;43:172–86.

94. Zhang F, Bazarevsky V, Vakunov A, et al. MediaPipe Hands: On-device real-time hand tracking. 2020. arXiv: 2006.10214.

95. Romero J, Tzionas D, and Black MJ. Embodied hands: Modeling and capturing hands and bodies together. ACM Trans. Graph. (Proc. SIGGRAPH Asia) 2017;36:245:1–245:17.

96. Moon G, Yu SI, Wen H, Shiratori T, and Lee KM. InterHand2.6M: A dataset and baseline for 3D interacting hand pose estimation from a single RGB image. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 12365. LNCS. 2020:548–64.

97. Lin K, Wang L, and Liu Z. End-to-end human pose and mesh reconstruction with transformers. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2021, pp. 1954–. 1963:1954–63.

98. Lin K, Wang L, and Liu Z. Mesh Graphormer. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:12939–48.

99. Park J, Oh Y, Moon G, Choi H, and Lee KM. HandOccNet: Occlusion-robust 3D hand mesh estimation network. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:1496–505.

100. Potamias RA, Zhang J, Deng J, and Zafeiriou S. WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025:12242–54.

101. Prakash A, Tu R, Chang M, and Gupta S. 3D hand pose estimation in everyday egocentric images. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 15136. LNCS. 2024:183–202.

102. Zhang J, Deng J, Ma M, and Potamias RA. HaWoR: World-space hand motion reconstruction from egocentric videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025.

103. Yu Z, Zafeiriou S, and Birdal T. Dyn-HaMR: Recovering 4D interacting hand motion from a dynamic camera. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025:27716–26.

104. Furnari A, Battiato S, Grauman K, and Farinella GM. Next-active-object prediction from egocentric videos. J. Vis. Commun. Image Represent. 2017;49:401–11.

105. Ragusa F, Farinella GM, and Furnari A. StillFast: An end-to-end approach for short-term object interaction anticipation. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW). 2023:3635–44.

106. Nagarajan T, Feichtenhofer C, and Grauman K. Grounded human-object interaction hotspots from video. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2019:8688–97.

107. Nagarajan T, Li Y, Feichtenhofer C, and Grauman K. EGO-TOPO: Environment afordances from egocentric video. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2020:163–72.

108. Luo H, Zhai W, Zhang J, Cao Y, and Tao D. Learning afordance grounding from exocentric images. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:2252–61.

109. Feix T, Romero J, Schmiedmayer HB, Dollar AM, and Kragic D. The GRASP taxonomy of human grasp types. IEEE Trans. Human-Mach. Syst. (THMS) 2016;46:66–77.

110. Brahmbhatt S, Tang C, Twigg CD, Kemp CC, and Hays J. ContactPose: A dataset of grasps with object contact and hand pose. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 12358. LNCS. 2020:361–78.

111. Taheri O, Ghorbani N, Black MJ, and Tzionas D. GRAB: A dataset of whole-body human grasping of objects. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 12349. LNCS. 2020:581– 600.

112. Ji J, Krishna R, Fei-Fei L, and Niebles JC. Action Genome: Actions as compositions of spatiotemporal scene graphs. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2020:10236–47.

113. Herzig R, Ben-Avraham E, Mangalam K, et al. Object-Region Video Transformers. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:3148–59.

114. Radevski G, Moens MF, and Tuytelaars T. Revisiting spatio-temporal layouts for compositional action recognition. In: Proc. British Mach. Vis. Conf. (BMVC). 2021.

115. Wang X and Gupta A. Videos as space-time region graphs. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 11209. LNCS. 2018:413–31.

116. Ma J and Damen D. Hand-object interaction reasoning. In: Proc. 18th IEEE Int. Conf. Advanced Video Signal-Based Surveillance (AVSS). 2022:1–8.

117. Dessalene E, Maynord M, Devaraj C, Ferm¨uller C, and Aloimonos Y. Egocentric object manipulation graphs. 2020. arXiv: 2006.03201.

118. Rodin I, Furnari A, Min K, Tripathi S, and Farinella GM. Action scene graphs for longform understanding of egocentric videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2024:18622–32.

119. Rodin I, Wu TY, Min K, et al. EASG-Bench: Video Q&A benchmark with egocentric action scene graphs. 2025. arXiv: 2506.05787.

120. Rai N, Chen H, Ji J, et al. Home Action Genome: Cooperative compositional action understanding. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2021:11184– 93.

121. Misra I, Zitnick CL, and Hebert M. Shufle and learn: Unsupervised learning using temporal order verification. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 9905. LNCS. 2016:527–44.

122. Goletto G, Nagarajan T, Averta G, and Damen D. AMEGO: Active memory from long egocentric videos. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 15071. LNCS. 2024:92– 110.

123. Zemskova T, Andryushenko S, Obrubov I, et al. FocusGraph: Graph-structured frame selection for embodied long video question answering. 2026. arXiv: 2603.04349.

124. Plizzari C, Tonioni A, Xian Y, Kulshrestha A, and Tombari F. Omnia de EgoTempo: Benchmarking temporal understanding of multi-modal LLMs in egocentric videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025:24129–38.

125. Yan S, Xiong Y, and Lin D. Spatial temporal graph convolutional networks for skeleton-based action recognition. In: Proc. 32nd AAAI Conf. Artif. Intell. (AAAI). 2018:7444–52.

126. Cong Y, Liao W, Ackermann H, Rosenhahn B, and Yang MY. Spatial-temporal transformer for dynamic scene graph generation. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:16372–82.

127. Wang G, Li Z, Chen Q, and Liu Y. OED: Towards one-stage end-to-end dynamic scene graph generation. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2024:27938– 47.

128. Peddi R, Singh S, Saurabh, Singla P, and Gogate V. Towards scene graph anticipation. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 15146. LNCS. 2024:159–75.

129. Nguyen TT, Nguyen P, Cothren J, Yilmaz A, and Luu K. HyperGLM: HyperGraph for video scene graph generation and anticipation. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025:29150–60.

130. Li Y, Li Z, Chen H, and Xu L. Unbiased video scene graph generation via visual and semantic dual debiasing. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025.

131. Li J, Cantu FJP, Yu E, Wong A, Cui Y, and Chen Y. SAMJAM: Zero-shot video scene graph generation for egocentric kitchen videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW). 2025:467–73.

132. Yu S, Cho J, Yadav P, and Bansal M. Self-chained image-language model for video localization and question answering. In: Adv. Neural Inf. Process. Syst. (NeurIPS). Vol. 36. 2023.

133. Yu S, Jin C, Wang H, et al. Frame-Voyager: Learning to query frames for video large language models. In: Proc. Int. Conf. Learn. Represent. (ICLR). 2025.

134. Qiu H, Gao M, Qian L, et al. STEP: Enhancing video-LLMs’ compositional reasoning by spatio-temporal graph-guided self-training. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025:3284–94.

135. Chu M, Li Y, and Chua TS. Understanding long videos via LLM-powered entity relation graphs. 2025. arXiv: 2501.15953.

136. Kipf T, Elsayed GF, Mahendran A, et al. Conditional object-centric learning from video. In: Proc. Int. Conf. Learn. Represent. (ICLR). 2022.

137. Zhang C, Fu C, Wang S, et al. Object-centric video representation for long-term action anticipation. In: Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV). 2024:6751–61.

138. Li S, Du Y, Torralba A, Sivic J, and Russell B. Weakly supervised human-object interaction detection in video via contrastive spatiotemporal regions. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2021:1845–55.

139. Zhou K, Yang J, Loy CC, and Liu Z. Learning to prompt for vision-language models. Int. J. Comput. Vis. (IJCV) 2022;130:2337–48.

140. Zhou K, Yang J, Loy CC, and Liu Z. Conditional prompt learning for vision-language models. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2022:16816–25.

141. Jia M, Tang L, Chen BC, et al. Visual prompt tuning. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 13693. LNCS. 2022:709–27.

142. Khattak MU, Rasheed H, Maaz M, Khan S, and Khan FS. MaPLe: Multi-modal prompt learning. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:19113– 22.

143. Wang M, Xing J, and Liu Y. ActionCLIP: A new paradigm for video action recognition. 2021. arXiv: 2109.08472.

144. Ju C, Han T, Zheng K, Zhang Y, and Xie W. Prompting visual-language models for eficient video understanding. In: Proc. Eur. Conf. Comput. Vis. (ECCV). Vol. 13695. LNCS. 2022:105–24.

145. Wasim ST, Naseer M, Khan S, Khan FS, and Shah M. Vita-CLIP: Video and text adaptive CLIP via multimodal prompting. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2023:23034–44.

146. Xu B, Zheng S, and Jin Q. POV: Prompt-oriented view-agnostic learning for egocentric handobject interaction in the multi-view world. In: Proc. ACM Int. Conf. Multimedia (ACM MM). 2023:2807–16.

147. Jia C, Luo M, Chang Z, et al. Generating action-conditioned prompts for open-vocabulary video action recognition. In: Proc. ACM Int. Conf. Multimedia (ACM MM). 2024:4640–9.

148. Kundu S, Trehan S, and Aakur SN. Discovering novel actions from open world egocentric videos with object-grounded visual commonsense reasoning. In: Proc. Eur. Conf. Comput. Vis. (ECCV). LNCS. 2024.

149. Bansal S, Wray C, and Damen D. HOI-Ref: Hand-object interaction referral in egocentric vision. 2024. arXiv: 2404.09933.

150. Yang J, Liu S, Guo H, et al. EgoLife: Towards egocentric life assistant. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2025:28885–900.

151. Wang X, Kwon T, Rad M, et al. HoloAssist: An egocentric human interaction dataset for interactive AI assistants in the real world. In: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). 2023:20270–81.

152. Flaborea A, Melendugno GMD di, Plini L, et al. PREGO: Online mistake detection in PRocedural EGOcentric videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2024:18483–92.

153. Lee SP, Lu Z, Zhang Z, Hoai M, and Elhamifar E. Error detection in egocentric procedural task videos. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). 2024:18655– 66.

154. Peddi R, Saurabh S, Vupputuri A, et al. CaptainCook4D: A dataset for understanding errors in procedural activities. In: Adv. Neural Inf. Process. Syst. (NeurIPS) Datasets and Benchmarks Track. 2024.

155. Kong C et al. Aria Gen 2 Pilot Dataset. 2025. arXiv: 2510.16134.

156. Nair S, Rajeswaran A, Kumar V, Finn C, and Gupta A. R3M: A universal visual representation for robot manipulation. In: Proc. Conf. Robot Learn. (CoRL). Vol. 205. Proc. Mach. Learn. Res. (PMLR). 2022:892–909.

157. Zitkovich B, Yu T, Xu S, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In: Proc. Conf. Robot Learn. (CoRL). Vol. 229. Proc. Mach. Learn. Res. (PMLR). 2023:2165–83.

158. Kim MJ, Pertsch K, Karamcheti S, et al. OpenVLA: An open-source vision-language-action model. In: Proc. Conf. Robot Learn. (CoRL). 2024.

159. Black K, Brown N, Driess D, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. 2024. arXiv: 2410.24164.

160. Liu V, Adeniji A, Zhan H, Bhirangi R, Abbeel P, and Pinto L. EgoZero: Robot learning from smart glasses. 2025. arXiv: 2505.20290.

161. Jiang Z, Xie Y, Lin K, et al. DexMimicGen: Automated data generation for bimanual dexterous manipulation via imitation learning. In: Proc. IEEE Int. Conf. Robot. Autom. (ICRA). 2025.

162. Vinod A, Pandit S, Vavre A, and Liu L. EgoVLM: Policy optimization for egocentric video understanding. 2025. arXiv: 2506.03097.

163. Tu Y, Luo H, Chen X, Bai X, Wang F, and Zhao H. PlayerOne: Egocentric world simulator. In: Adv. Neural Inf. Process. Syst. (NeurIPS). . (Oral.) 2025.

164. Bai Y, Tran D, Bar A, LeCun Y, Darrell T, and Malik J. Whole-body conditioned egocentric video prediction. 2025. arXiv: 2506.21552.

165. Seth A, Tyagi U, Selvakumar R, et al. EGOILLUSION: Benchmarking hallucinations in egocentric video understanding. In: Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP). 2025:28461–80.