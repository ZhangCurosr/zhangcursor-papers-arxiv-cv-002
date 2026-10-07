# M3SunAgent: Monocular 3D Spatial Understanding Agent for Metric Depth Estimation and 3D Visual Grounding

Jinsong Zhang, Kejun Wu, Senior Member, IEEE, Ming Zhu, Renjie Qiao, Chengtao Cai, Senior Member, IEEE, and Zhengguo Li, Fellow, IEEE

Abstract—Monocular metric depth estimation and 3D visual grounding represent the two complementary cornerstones of monocular 3D spatial understanding (M3Sun), from which the fundamental 3D spatial information required by M3Sun can be acquired. However, these complementary tasks are generally conducted by separate frameworks, which pose challenges of inflexible and unaligned spatial information access for embodied intelligence systems. In this paper, we propose a unified agent for monocular 3D spatial understanding (M3SunAgent) that leverages a large language model (LLM) as a task planner for spatial visual programming, which flexibly generate structured programs and coordinate tools. For instance-level metric depth estimation task, M3SunAgent invokes an object detector tool to locate the target, estimates depth at selected points with a depth estimation tool, and aggregates these predictions into an instancelevel depth estimate. We also construct the M3Sun Instance (M3SI) dataset, a benchmark with 2,910 samples for evaluation. For monocular 3D visual grounding task, M3SunAgent uses a vision-language model (VLM) tool to locate the target and output basic spatial attributes, then combines back-projection tool with a dimension-lifting tool to predict its 3D bounding box. Experimental results demonstrate the superior performance of M3SunAgent. Specifically, in evaluations of instance-level monocular metric depth estimation, M3SunAgent achieves the best performance among all compared models, 52.61% predicted instances are distributed below depth error 0.25 $\left( \delta _ { < 0 . 2 5 } \right)$ . In evaluations of monocular 3D visual grounding, M3SunAgent demonstrates overall competitive performance than vision and VLM models, reaching a 3D mean intersection over union (mIoU) of 41.73% and exceeding the state-of-the-art MonoVLM model by 3.62%.

Index Terms—Monocular 3D spatial understanding, metric depth estimation, monocular 3D visual grounding, large language models, vision-language models.

## I. INTRODUCTION

ONOCULAR 3D spatial understanding (M3Sun) aims involving recognizing objects, and estimating their distances, 3D locations, physical extents, and orientations. The precise spatial properties of individual objects provided by M3Sun plays a vital role in embodied intelligence, teleoperation, and autonomous driving. For example, when a robot is instructed to “pick up the pen on the table”, it needs to locate the referred pen in the input image and estimate the distance to guide its motion planning [1]. In autonomous driving, an instruction such as “Avoid the car ahead on the right” requires the system to identify the referred vehicle and estimate its 3D position, dimensions, and heading. Such scenarios require systems to locate objects described in natural language and infer their spatial properties, motivating continued advances in M3Sun.

Instance-level monocular metric depth estimation and monocular 3D visual grounding represent the two cornerstones of M3Sun. These tasks offer complementary 3D spatial information, e.g., instance-level depth and 3D bounding box. The former estimates the metric depth of an instance in physical units, while the latter identifies the referred object and recovers its 3D structure, including its position, dimensions, and orientation [2]. Both types of spatial information are important for understanding objects from a single image. Metric depth places object distance on a physical scale, enabling quantitative distance judgments. Accurate metric depth estimation can also provide additional cues for spatial reasoning. 3D structure characterizes the object’s location and spatial extent. When accurate 3D structures are provided as auxiliary cues, they can improve reasoning about object arrangement and spatial relationships [3]. Both tasks require systems to ground language descriptions in images and infer quantitative spatial properties by combining visual evidence with semantic cues. The common demand to identify objects and produce quantitative spatial estimates motivates studying these two tasks together.

Recent research in M3Sun has explored various approaches to improve the accuracy of estimated spatial properties. Training on diverse scenes and modeling camera parameters have improved metric distance estimation [4]. Beyond scale and camera modeling, planar geometry and scene semantics provide additional cues for monocular depth estimation [5], [6]. These developments provide a basis for estimating the distance to a specified object after its location has been identified in the image. Improving estimates of the target’s 3D location and extent requires better alignment between language descriptions and image geometry. Spatial descriptions can provide cues for predicting the target’s 3D structure [7]. Target localization is also fundamental to this task. Accurately identifying the referred object in the image helps focus geometric estimation on that object and reduce the influence of irrelevant scene elements [8]. Together, these advances establish a strong basis for recovering object-level spatial information from a single RGB image.

Estimating the distance to a referred object requires locating the target and extracting depth measurements from the corresponding instance. However, many monocular depth estimators predict depth across the entire image [4], [9]. Such scene-level predictions do not identify the referred object or directly provide its instance-level depth. In monocular 3D visual grounding, specialized architectures have achieved strong performance [7]. However, models designed for this specific task may lack the flexibility to handle other spatial queries. Advances in large language models (LLMs) and vision-language models (VLMs) have enabled their application to a wider range of visual tasks [10]. These developments have motivated attempts to improve the flexibility of visual models for spatial understanding. With spatial training, VLMs can reason about object positions, orientations, and 3D relationships [11]. However, achieving such performance may require substantial 3D training data, computational resources, and multi-stage training procedures [12]. Moreover, VLMs may still face challenges in producing accurate metric distance estimates due to the difficulty of grounding visual observations into physical scales [13].

Meanwhile, advances in LLM-based agents provide a potential solution to these limitations. In an agent architecture, the LLM serves as a central cognitive planner that performs reasoning and coordinates tool execution. Specialized models and computational tools execute specific operations and pass intermediate results between steps [14]. The agent then integrates their outputs to generate the final response. This division of labor allows the planning LLM to be applied without additional training, reducing the related cost. It also allows the agent to coordinate different expert models within a shared framework and support multiple tasks. Additional models and tools can be incorporated to extend the range of spatial queries that the agent can handle [15]. Furthermore, quantitative outputs from specialized components can provide the LLM with more reliable evidence for generating the final response. These capabilities have contributed to the growing use of agent-based methods for 3D spatial understanding [16]. For these two tasks, a unified agent should retain intermediate measurements of the queried object and combine them to produce the required spatial output.

To address the need for a unified agent in M3Sun, we propose M3SunAgent, a monocular spatial understanding agent that converts a natural language query into an executable program. The LLM generates programs by identify ing the requested task and arranging the required operations in the proper order. A program interpreter then executes these operations through registered visual tools and geometric computations while transferring intermediate results between operations. For instance-level metric depth estimation, the agent localizes the referred object and estimates its depth at selected image points. It aggregates these predictions into an instance-level metric distance estimate. For monocular 3D visual grounding, a VLM predicts the target’s basic spatial attributes. Back-projection and a specialized 2D-to-3D lifting model use these predictions to recover the 3D geometry required to construct its 3D bounding box. Both pipelines share the same planning and execution framework. Our main contributions are summarized as follows:

• We propose M3SunAgent, a monocular spatial understanding agent that flexibly supports instance-level metric depth estimation and 3D visual grounding. M3SunAgent leverages spatial visual programming to understand diverse natural-language queries by a LLM task planner, converting complex queries into executable programs. These programs coordinate distinct models and operations, allowing them to collaboratively execute for spatial understanding.

• For metric depth estimation, we use focal-length normalization to reduce the effects of intrinsics, together with multi-point sampling and outlier filtering to improve robustness. For monocular 3D visual grounding, we use a VLM for target localization and basic spatial attribute prediction, explicit back-projection for camera adaptation, and a specialized lifting model for accurate dimensions and yaw estimation.

• We construct the M3Sun Instance (M3SI) dataset, a benchmark with 2,910 samples for instance-level metric depth estimation. We evaluate M3SunAgent on M3SI for this task and on Mono3DRefer for monocular 3D visual grounding. Across these evaluations, the agent outperforms the spatial VLM baselines and achieves competitive performance to task-specific vision methods.

## II. RELATED WORK

## A. Monocular metric depth estimation

Monocular metric depth estimation aims to recover depth values in physical units from a single RGB image. Its generalization is influenced by camera parameters, scene scale, and imaging conditions. The focus of this task has gradually shifted from improving benchmark performance toward realworld scale recovery, cross-domain generalization, and adaptation to diverse camera models [17]. Methods that consider focal length and depth scale seek to improve absolute depth prediction in unseen scenes [18]. Some methods estimate camera parameters directly from the input image to improve high-resolution depth prediction [19]. Self-supervised methods have also addressed challenging conditions, including day– night illumination changes [20]. These advances improve depth prediction across diverse scenes and imaging conditions. However, these depth maps primarily describe scene-level geometry. They do not directly provide the distance to an object identified by language.

Research on monocular 3D detection has also explored object-level depth estimation. Some methods separate objects from background regions and estimate object depth to assist monocular 3D object detection, while others exploit objectrelated depth information to improve 3D localization [21], [22]. These approaches support 3D localization but are designed for detection rather than instance-level metric depth estimation. At the point level, DepthLM uses visual prompts to estimate metric depth at specified image locations [23]. A single point may nevertheless be insufficient for an extended or partially occluded object, and a target region may contain background surfaces. Thus, obtaining a stable distance for a specific instance requires both reliable depth measurements from the target and strategies to reduce the influence of unreliable samples.

## B. Monocular 3D visual grounding

Recent studies have explored aligning language with visual content represented in different forms [24]. Within this context, visual grounding aims to associate a referring expression with a target object in an image or a 3D scene. In 2D images, visual grounding methods use visual and linguistic information to locate the referred object [25]. Grounding in 3D scenes further incorporates geometric information to identify the referred object. ScanRefer matches language descriptions to objects in RGB-D scans [26]. Pseudo-EV addresses ambiguity in viewpoint-dependent spatial descriptions [27]. However, these 3D approaches use observations that are unavailable when only a single RGB image is available.

Monocular 3D visual grounding aims to infer the referred object’s 3D location and extent from a single image and language descriptions. Mono3DVG introduced this setting using descriptions that include both appearance and geometric cues [2]. Subsequent methods use cross-modal interaction to incorporate geometric features [28] and dimension-decoupled text encoding to represent spatial expressions [7]. More recently, VLM-based approaches have also emerged. MonoVLM adapts a VLM for this task [8]. GR3D integrates 2D and 3D grounding within a spatial VLM [29]. Although these methods achieve strong performance, they are generally designed for specific tasks. As a result, they may lack the flexibility to handle other types of spatial queries.

## C. LLM Agents for Spatial Understanding

LLMs can use contextual information to support reasoning [30]. Agent-based methods further use LLMs to plan operations and coordinate specialized models to complete different tasks. HuggingGPT uses an LLM to plan subtasks and select expert models. It then executes the subtasks with those models and integrates their outputs [15]. Another line of work represents the required operations as executable programs. VisProg generates such programs from language instructions and composes visual operations to obtain a result [16]. For 3D spatial questions, VADAR uses a dynamic API that can be extended with functions for new queries [31]. SpaceTools further investigates spatial tool coordination by training a VLM through interactive exploration and feedback [14]. These studies demonstrate the potential of LLM agents for modular perception and embodied reasoning. However, their application to monocular metric depth estimation and monocular 3D visual grounding remains underexplored. M3SunAgent addresses this gap by using a unified agent framework to coordinate specialized visual models and geometric tools for both tasks. It can be adopted to integrate conventional physicsdriven control methods and emerging data-driven vision language action to address long horizon tasks from the hybrid system point of view [32].

## III. METHOD

## A. Overview of M3SunAgent

M3SunAgent takes a single RGB image and a naturallanguage query as inputs. As shown in Fig. 1, a LLM serves as the task planner, determining whether the query corresponds to instance-level metric depth estimation or monocular 3D visual grounding. Through spatial visual programming, the LLM then generates the corresponding domain-specific language (DSL) program [31]. The program interpreter then executes the specified operations by invoking registered visual tools and geometric computations from the Tool Registry. The resulting spatial estimate is passed to the LLM, which combines it with the original query to formulate the final answer. This shared planning and execution framework supports both M3Sun tasks.

The two tasks produce different forms of spatial information. A distance query requires a metric estimate associated with the referred instance. A 3D grounding query requires a 3D bounding box describing the instance’s position, dimensions, and orientation. In both cases, the DSL program passes target-specific intermediate results between operations, ensuring that subsequent operations remain associated with the same referred object. This keeps subsequent estimates associated with the same object. The following subsections describe the program execution mechanism and the operators used for each task.

## B. Spatial Visual Programming

LLMs and VLMs provide strong instruction understanding and semantic reasoning, but reliable metric depth estimation and monocular 3D visual grounding remain challenging. Specialized vision models provide explicit localization and geometric estimates, but are typically designed for predefined perception tasks. To combine these complementary capabilities, M3SunAgent adopts a spatial visual programming paradigm [16]. The workflows of the two tasks are shown in Algorithms 1 and 2. Given a natural language query, the LLM generates a DSL program composed of task-specific operators. The program interpreter executes these operators by invoking visual tools and geometric computations registered in the Tool Registry for target localization, depth estimation, aggregation, 3D center recovery, geometric attribute estimation, and 3D box construction.

Algorithm 1 Execution for Instance-Level Metric Depth Es  
timation   
Notation: E denotes the in-context DSL examples. P denotes the generated   
DSL program. T denotes the tool registry. r denotes the LLM invocation   
index.   
1: for r ∈ {1, 2} do   
2: if r = 1 then   
3: LLM(QUERY, E) → (TASK, DESCRIPTION, P)   
4: if TASK = INSTANCE DEPTH then   
5: Generated DSL program P:   
6: TARGET = LOCATE(IMAGE, DESCRIPTION)   
7: DEPTHS = DEPTH(IMAGE, TARGET, INTRINSICS)   
8: DISTANCE = AGGREGATE(DEPTHS)   
9: ToolRegistry → T   
10: ProgramInterpreter(P, T) → DISTANCE   
11: end if   
12: else if r = 2 ∧ TASK = INSTANCE DEPTH then   
13: LLM(QUERY, DISTANCE) → ANSWER   
14: return ANSWER   
15: end if   
16: end for

![](images/5ecf27780bf3989dc536531b6c333ec3b8b323ad2cd0f06b9b51b9913a11b1d4.jpg)  
Fig. 1. Overview of M3SunAgent. Given an RGB image and a natural-language query, the LLM selects the task and generates a corresponding DSL program, which is executed by the program interpreter using perception models and geometric operations. For instance-level metric depth estimation, point-wise depth predictions within the referred object are aggregated into an instance-level metric distance. For monocular 3D visual grounding, a VLM locates the target and predicts basic spatial attribute, which is combined with back-projection and a 2D-to-3D lifting model to construct the 3D bounding box. The result is then used to generate the final response.

Algorithm 2 Execution for Monocular 3D Visual Grounding Notation: E denotes the in-context DSL examples. P denotes the generated DSL program. T denotes the tool registry. r denotes the LLM invocation index.

![](images/4e92f10eddb72d77cca122ea0c5e18b3bd44a4880e9d24266b854279f4cc2fd4.jpg)

M3SunAgent uses in-context examples to guide DSL program generation. We construct several example programs based on the predefined task workflows and provide them to the LLM together with the current query. Rather than directly predicting the final spatial result, the LLM extracts the target description and generates a DSL program based on these examples. Each DSL statement assigns the result of an operator call to an output variable. For example, target localization is expressed as TARGET = LOCATE(IMAGE, DESCRIPTION), where LOCATE is the DSL operator, IMAGE and DESCRIPTION are its input variables, and TARGET is its output variable. The output variables can be referenced by subsequent statements, thereby defining the data dependencies within the program. The LLM therefore serves as the task planner rather than directly predicting the requested spatial result.

After generation, the DSL program is parsed and executed by the Program Interpreter [16]. The interpreter processes the statements in execution order and retrieves the tool corresponding to each operator from the Tool Registry. A registered tool may be either a learned model or a deterministic computations. After each tool is invoked, its result is stored in the corresponding output variable and passed to subsequent operations when required. For example, the target localized by LOCATE can be passed to DEPTH to estimate depth values at selected target points. This execution mechanism allows M3SunAgent to connect multiple specialized tools through the intermediate variables defined in the DSL program.

The agent calls the LLM twice for each query. In the first invocation, the LLM uses the query and in-context examples to identify the requested task and generate a corresponding DSL program. The interpreter then executes the program by invoking registered visual tools and geometric operations. Together, these operations produce either the referred object’s metric distance or its 3D bounding box. In the second invocation, the LLM combines the computed result with the original query to formulate the final answer. The LLM thus handles task planning and response synthesis, while specialized tools perform spatial estimation. Defining DSL operators separately from their implementations also allows tools to be added or replaced.

![](images/d770f307131daff4b73a9bfd10f17f21dcc3d2df4df01690b1aee1b128e1d175.jpg)  
Fig. 2. Overview of the instance-level metric depth estimation pipeline in M3SunAgent. LOCATE identifies the object referred to in the input description and returns its 2D bounding box. DEPTH normalizes the focal length and samples nine points within the box. It then uses a depth estimation model to predict the depth of these sampled points. AGGREGATE filters unreliable predictions and computes the median of the remaining depth estimates to obtain the target’s distance.

## C. Instance-Level Metric Depth Estimation Pipeline

For instance-level metric depth estimation, M3SunAgent generates a DSL program with three operators: LOCATE, DEPTH, and AGGREGATE. As summarized in Algorithm 1, the first LLM invocation identifies the requested task and generates the program. The interpreter then executes it using registered tools. These tools localize the referred object, estimate metric depth at selected points within its image region, and aggregate the predictions into an instance-level metric distance. The second LLM invocation uses this distance and the original query to formulate the final answer. Fig. 2 illustrates the detailed execution process of the DSL program. The specific details are given below.

LOCATE The LOCATE operator takes the input image and the target description extracted during task planning. It invokes a registered localization tool built on an open-vocabulary object detector [33]. Given IMAGE and DESCRIPTION, the tool identifies the referred instance and returns its 2D bounding box as TARGET. This step focuses subsequent depth sampling on the target region.

DEPTH Instance-level metric depth estimation requires the point-wise depth predictions to be associated with the referred instance. When executing DEPTH, the interpreter invokes a registered depth estimation model [23]. Given IMAGE, TARGET, and INTRINSICS, the tool applies focal length normalization and estimates metric depth at multiple locations within the target region.

Metric depth prediction is sensitive to variations in camera focal length [34]. To reduce this sensitivity, the input image is resized to normalize its effective focal length to a canonical value. The target bounding box is then used to sample nine points: one center point, four axial points, and four points near the corners. This pattern covers different parts of the target and avoids relying on a single location.

A visual marker is placed at each sampled point, and the resulting marked images are passed to the depth model. The model predicts the metric depth at each location and returns the nine values as DEPTHS. Using a small set of representative points provides spatial coverage of the target while keeping the number of model evaluations limited.

AGGREGATE A 2D bounding box provides only an approximate target region and may include background pixels. Sampled points near object boundaries or occluded regions may therefore produce depths that do not represent the referred instance. The AGGREGATE operator invokes a registered aggregation tool to reduce the influence of these unreliable depth predictions. The tool first removes non-finite values and negative depth predictions. If fewer than four valid values remain, outlier removal is skipped because quartile statistics may be unreliable for such a small sample. Let a and b denote the 25th and 75th percentiles of the valid depth predictions, respectively. The range of retained values is defined as

$$
\left[ a - 1 . 5 ( b - a ) , b + 1 . 5 ( b - a ) \right] .\tag{1}
$$

Values outside this range are treated as outliers and removed.

![](images/f149bea33ba805a943e94d7adfcb9d55657658db32daafbf13dfa818cb49d25e.jpg)  
Fig. 3. Overview of the monocular 3D visual grounding pipeline in M3SunAgent. Given an RGB image and a target description, GROUND predicts the referred object’s 2D bounding box, the image-plane projection of its 3D center, and its depth. BACKPROJECT uses the projected center, depth, and camera intrinsics to recover the 3D center. LIFT estimates the object’s dimensions and yaw angle from the image, 2D box, and camera intrinsics. BUILD combines the recovered attributes to construct the final 3D bounding box.

The median of the remaining values provides a robust estimate of the target distance and is returned as DISTANCE.

## D. Monocular 3D Visual Grounding Pipeline

Monocular 3D visual grounding requires identifying a language-referred object and recovering its 3D bounding box in camera coordinates. For this task, the LLM generates a DSL program comprising four operations. GROUND predicts the target’s 2D bounding box, projected 3D center, and depth. BACKPROJECT uses camera intrinsics to recover its 3D center from these estimates. LIFT estimates the target’s dimensions and orientation. Finally, BUILD combines these attributes into a 3D bounding box. The interpreter executes the operations using registered tools and passes intermediate results between them. Fig. 3 illustrates the detailed execution process. Algorithm 2 summarizes how the program is generated and executed.

GROUND For monocular 3D grounding, the target description may also contain spatial information, such as its relative position, distance, and spatial relationships with other objects. Such information can provide useful cues for estimating the target’s 3D properties. To make better use of these semantic cues during target localization, we do not rely solely on an open-vocabulary detector as in Section III-C. Instead, we employ a VLM [35] that predicts the target’s 2D bounding box while also estimating several basic spatial attributes, including the image-plane projection of its 3D center and its depth. These outputs are returned as B2D, C2D, and DZ, respectively.

The bounding box B2D is represented as $[ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ]$ , and the projected center C2D is represented as [u, v]. The depth DZ denotes the depth of the target center in the camera coordinate system. The projected center is not necessarily the geometric center of the 2D bounding box. Instead, it corresponds to the projection of the target 3D center onto the image plane. According to the pinhole camera model, the target 3D center $\left( X _ { c } , Y _ { c } , Z _ { c } \right)$ and its projected coordinates (u, v) satisfy

$$
u = f _ { x } \frac { X _ { c } } { Z _ { c } } + c _ { x } , \qquad v = f _ { y } \frac { Y _ { c } } { Z _ { c } } + c _ { y } ,\tag{2}
$$

where $f _ { x }$ and $f _ { y }$ are the focal lengths in pixels along the horizontal and vertical directions, respectively, and $\left( c _ { x } , c _ { y } \right)$ denotes the camera principal point.

BACKPROJECT The BACKPROJECT operator invokes a registered geometric tool to recover the target 3D center. Given C2D, DZ, and INTRINSICS, the tool performs deterministic back-projection as

$$
X _ { c } = \frac { u - c _ { x } } { f _ { x } } d _ { Z } , \qquad Y _ { c } = \frac { v - c _ { y } } { f _ { y } } d _ { Z } , \qquad Z _ { c } = d _ { Z } .\tag{3}
$$

The recovered 3D center $\left( X _ { c } , Y _ { c } , Z _ { c } \right)$ is returned as C3D. The explicit back-projection directly incorporates camera intrinsics into the computation of the 3D center. This allows the recovered coordinates to adapt to different camera configurations rather than relying on fixed camera parameters.

LIFT The remaining geometric attributes are estimated by the LIFT operator. It invokes a registered lifting tool built on a specialized 2D-to-3D model [36]. Given IMAGE, B2D, and INTRINSICS, the tool estimates the object dimensions and 3D rotation. The dimensions are represented as $s = ( w , h , l )$ in meters, where w, h, and l denote the width, height, and length, respectively. The rotation is represented by a quaternion $( q _ { w } , q _ { x } , q _ { y } , q _ { z } )$

In our 3D box representation, the target orientation is expressed as a yaw angle around the Y-axis of the camera coordinate system. Under the quaternion convention used in our implementation, the yaw angle is computed as

$$
\theta _ { Y } = \mathrm { a t a n 2 } \left( 2 ( q _ { w } q _ { y } - q _ { x } q _ { z } ) , 1 - 2 ( q _ { y } ^ { 2 } + q _ { z } ^ { 2 } ) \right) .\tag{4}
$$

The estimated dimensions and yaw angle are returned as SIZE and YAW. Using the lifting tool reduces the amount of 3D geometric information that must be predicted directly by the VLM.

BUILD Finally, the BUILD operator combines C3D, SIZE, and YAW through a deterministic geometric operation to construct the target’s 3D bounding box. The constructed box is stored as BOX3D. In the second LLM invocation, the LLM uses this result and the original query to formulate the final answer.

## IV. EXPERIMENTS

## A. Experimental Settings

1) Datasets: M3SI To evaluate our framework on instancelevel metric depth estimation, we construct a new benchmark, named the M3Sun Instance (M3SI) dataset, based on ETH3D [37]. ETH3D provides images and pixel-level depth annotations for 13 diverse real-world scenes. Although it is widely used for monocular depth estimation and stereo matching, its original annotations are not directly applicable to instance-level distance estimation, which aims to measure the metric distance from each object instance to the camera. We adapt ETH3D to this task through a multi-stage annotation pipeline. A LLM is used to identify candidate object categories and generate object descriptions for each image. When multiple objects of the same category appear in an image, attributes such as color, spatial location, and relative size are added to distinguish individual instances. The resulting object descriptions are then used as text prompts for Grounding DINO [33] to obtain the corresponding 2D bounding boxes. Given the detected boxes and the associated depth maps, we sample valid depth values within each detected object region. The sampled values represent axial depth along the camera Z-axis rather than the Euclidean distance from the camera center. We therefore use the camera intrinsics to convert each axial depth value into a Euclidean distance. The median of the sampled distances is used as the ground-truth distance for each object instance. Following this pipeline, we construct M3SI, which contains 2,910 samples for instance-level metric depth estimation. Representative annotated instances are shown in Fig. 4.

Mono3DRefer We evaluate M3SunAgent on the Mono3DRefer dataset [2] to assess its performance on monocular 3D visual grounding. It contains 29,990 training samples, 5,735 validation samples, and 5,415 test samples. We use the training split to fine-tune the VLM responsible for target localization in the 3D visual grounding pipeline. We then evaluate the resulting model on the test split following the evaluation protocol defined in the original Mono3DRefer study. Following the subset definitions of Qu et al. [8], the test set is organized along three dimensions: target uniqueness,

![](images/ba91a1385552bf0867e0a8ef1804e88d37c49d3f737c5ec4df056bbd80b8bef7.jpg)  
① black trashbin · 7.20 m ② wooden picnic bench · 3.60 m

![](images/aaca36c4f58e9d779224669d88e54e668813a0eb6a2887a3b9890bc332415cbd.jpg)  
① black trash bin far right · 13.19 m ② yellow poster · 13.59 m

![](images/b659a3d726d880f0a17b48d940e41832c77a43fc7c6b8e255f9ac7692e31d0d5.jpg)

![](images/1b8a29e7780db9dc233f96abcd6c1f841df50feff6e2dbc7df323a9d4306711f.jpg)  
① black trash bin · 13.19 m ② red fire extinguisher box · 13.25 m ③ red fire extinguisher · 13.21 m

Fig. 4. Examples of annotated instances in M3SI. Blue bounding boxes mark the target objects. The text beneath each image provides their descriptions and ground-truth distances from the camera in meters.

target depth, and localization difficulty. Target uniqueness divides the samples into Unique and Multiple subsets. In the Unique subset, the target is the only instance of its category in the scene, whereas the Multiple subset contains other instances of the same category. Target depth divides the samples into Near (0–15 m), Medium (15–35 m), and Far (> 35 m) subsets. Localization difficulty follows the standard KITTI definitions [38]. Samples are categorized as Easy, Moderate, or Hard according to their degrees of occlusion and truncation.

2) SOTA models for comparison: In the metric depth estimation task, M3SunAgent is compared with two groups of models. The first group consists of general-purpose VLMs, including the closed-source model GPT-5 and the open-source models Pixtral-12B [39], InternVL3-8B [40], InternVL3.5-8B [41], the Qwen2.5-VL series [42], the Qwen3-VL series [35], Molmo2-8B [43], and LLaVA-1.5-7B [44]. The second group contains VLMs trained on task-specific datasets to enhance spatial perception and understanding, including SpaceLLaVA [1] and SpaceQwen [45], as well as DepthVLM [46], which augments a VLM with a depth prediction head for generating dense depth maps. All models are evaluated on the same test dataset.

In monocular 3D visual grounding, we consider both VLMbased methods and computer vision (CV) methods as baselines. The VLM-based methods include MiMo-VL-7B [47], Qwen2.5-VL-72B [42], Gemini-2.5-Flash, GPT-5, GPT-o3, Gemini-2.5-Pro, and MonoVLM [8], which is specifically trained for this task. The CV methods include Mono3DVG [2], Mono3DVG-TGE [28] and some two-stage methods. Their results are taken from the corresponding original papers.

3) Evaluation metrics: We evaluate instance-level metric depth estimation using absolute and relative errors, threshold accuracy, and the distribution of relative errors. Let $\hat { d } _ { i }$ and $d _ { i }$ denote the predicted and ground-truth distances of sample i, respectively. The relative error of a valid prediction is

TABLE I  
COMPARISON OF INSTANCE-LEVEL METRIC DEPTH ESTIMATION RESULTS ON M3SI. THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>Year</td><td>Valid/Total</td><td>MAE↓</td><td>RMSE↓</td><td>MedAE↓</td><td>MRE (%) ↓</td><td>MedRE (%) ↓</td><td> $\delta _ { < 0 . 2 5 } ~ ( \% ) ~ \uparrow$ </td></tr><tr><td colspan="9">General-Purpose VLMs</td></tr><tr><td>GPT-5</td><td>2025</td><td>1970/2910</td><td>3.31</td><td>5.07</td><td>1.80</td><td>39.22</td><td>34.43</td><td>24.33</td></tr><tr><td>Pixtral-12B [39]</td><td>2024</td><td>2810/2910</td><td>4.06</td><td>6.83</td><td>2.28</td><td>58.25</td><td>38.86</td><td>31.27</td></tr><tr><td>InternVL3-8B [40]</td><td>2025</td><td>2910/2910</td><td>4.60</td><td>6.96</td><td>2.84</td><td>51.68</td><td>53.06</td><td>16.36</td></tr><tr><td>InternVL3.5-8B [41]</td><td>2025</td><td>2910/2910</td><td>5.03</td><td>7.76</td><td>2.85</td><td>54.24</td><td>51.60</td><td>20.55</td></tr><tr><td>Qwen2.5-VL-3B [42]</td><td>2025</td><td>2910/2910</td><td>14.79</td><td>82.42</td><td>3.83</td><td>203.50</td><td>63.42</td><td>17.04</td></tr><tr><td>Qwen2.5-VL-7B [42]</td><td>2025</td><td>2900/2910</td><td>5.11</td><td>10.13</td><td>3.74</td><td>90.34</td><td>58.74</td><td>22.89</td></tr><tr><td>Qwen3-VL-4B [35]</td><td>2025</td><td>2910/2910</td><td>5.00</td><td>7.26</td><td>2.88</td><td>49.85</td><td>52.17</td><td>16.39</td></tr><tr><td>Qwen3-VL-8B [35]</td><td>2025</td><td>2910/2910</td><td>4.78</td><td>6.85</td><td>2.91</td><td>51.54</td><td>52.52</td><td>14.91</td></tr><tr><td>Molmo2-8B [43]</td><td>2026</td><td>2910/2910</td><td>4.84</td><td>10.19</td><td>2.41</td><td>75.99</td><td>46.64</td><td>29.28</td></tr><tr><td>LLaVA-1.5-7B [44]</td><td>2023</td><td>2875/2910</td><td>30.89</td><td>144.69</td><td>4.65</td><td>463.15</td><td>88.99</td><td>4.91</td></tr><tr><td colspan="9">Task-Specific Spatial VLMs</td></tr><tr><td>SpaceLLaVA [1]</td><td>2024</td><td>2871/2910</td><td>32.66</td><td>53.98</td><td>6.71</td><td>426.52</td><td>95.91</td><td>2.44</td></tr><tr><td>DepthVLM [46]</td><td>2026</td><td>2910/2910</td><td>5.05</td><td>7.20</td><td>3.02</td><td>55.03</td><td>56.68</td><td>7.01</td></tr><tr><td>SpaceQwen2.5-VL-3B [45]</td><td>2025</td><td>2910/2910</td><td>6.29</td><td>8.49</td><td>4.28</td><td>76.71</td><td>80.78</td><td>4.40</td></tr><tr><td>M3SunAgent (Ours)</td><td>一</td><td>2910/2910</td><td>2.26</td><td>3.76</td><td>1.26</td><td>27.04</td><td>23.28</td><td>52.61</td></tr></table>

$$
r _ { i } = \frac { | \hat { d } _ { i } - d _ { i } | } { d _ { i } } .\tag{5}
$$

Let V be the set of samples with valid numerical predictions, $N _ { v } = | \nu |$ , and N is the total number of test samples. We report mean absolute error (MAE), root mean squared error (RMSE), median absolute error (MedAE), mean relative error (MRE), and median relative error (MedRE) over V. MRE and MedRE are expressed as percentages.

We define a metric, $\delta _ { < 0 . 2 5 }$ , which represents the percentage of samples with a relative error below 25%. It is calculated as follows:

$$
\delta _ { < 0 . 2 5 } = \frac { 1 } { N } \sum _ { i \in \mathcal { V } } \mathbb { I } ( r _ { i } < 0 . 2 5 ) ,\tag{6}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. Using N as the denominator counts invalid predictions as incorrect.

To examine the error distribution, we divide the test outputs into five mutually exclusive categories. The first four contain valid predictions with relative errors in

$$
\begin{array} { l l } { { \mathcal { E } _ { 1 } = [ 0 , 0 . 1 0 ) , } } & { { \mathcal { E } _ { 2 } = [ 0 . 1 0 , 0 . 2 0 ) , } } \\ { { \mathcal { E } _ { 3 } = [ 0 . 2 0 , 0 . 2 5 ) , } } & { { \mathcal { E } _ { 4 } = [ 0 . 2 5 , 0 . 5 0 ) . } } \end{array}\tag{7}
$$

Their proportions are

$$
p _ { k } = \frac { 1 } { N } \sum _ { i \in \mathcal { V } } \mathbb { I } ( r _ { i } \in \mathcal { E } _ { k } ) , \qquad k \in \{ 1 , 2 , 3 , 4 \} .\tag{8}
$$

The fifth category includes valid predictions with relative errors of at least 50% and all invalid predictions:

$$
p _ { 5 } = \frac { 1 } { N } \left[ ( N - N _ { v } ) + \sum _ { i \in \mathcal { V } } \mathbb { I } ( r _ { i } \ge 0 . 5 0 ) \right] .\tag{9}
$$

This distribution provides a detailed view of prediction errors across different accuracy ranges while also accounting for large errors and invalid outputs.

For monocular 3D visual grounding, we measure the volumetric overlap between each predicted 3D bounding box and its ground-truth box. Let IoU<sup>3D</sup><sub>i</sub> denote this overlap for sample i, and let S be the evaluation set with $N s = | S |$ . Mean 3D IoU (mIoU) is reported as a percentage:

$$
\mathrm { m I o U } ( \mathcal { S } ) = \frac { 1 } { N _ { S } } \sum _ { i \in \mathcal { S } } \mathrm { I o U } _ { i } ^ { 3 D } .\tag{10}
$$

A higher mIoU indicates greater overall overlap between the predicted and ground-truth 3D bounding boxes. It provides a continuous measure of 3D localization quality and reflects errors in the predicted object position, dimensions, and orientation.

We also report grounding accuracy at a given IoU threshold τ :

$$
\operatorname { A c c @ } \tau ( \boldsymbol { \mathcal { S } } ) = \frac { 1 } { N s } \sum _ { i \in \boldsymbol { S } } \mathbb { I } \left( \operatorname { I o U } _ { i } ^ { 3 D } \geq \tau \right) .\tag{11}
$$

We report Acc@0.25 and Acc@0.5, which represent the percentages of samples with 3D IoU values of at least 0.25 and 0.5, respectively. While mIoU measures the average overlap quality, Acc@τ measures the proportion of predictions that meet a specified localization threshold. The two metrics therefore provide complementary measures of 3D grounding performance.

B. Comparison with State-of-the-Art Monocular Metric Depth Estimation Methods

1) Quantitative Evaluation: The quantitative results are reported in Table I and Fig. 5. The “Valid” column indicates the number of test samples for which a model produces a valid and parseable numerical prediction. MAE, RMSE, MedAE, MRE, and MedRE are calculated only over these valid predictions. The threshold accuracy metrics are evaluated over the entire dataset, with invalid outputs treated as incorrect predictions. The results show that many VLMs struggle to predict accurate metric depth. Some models produce large numerical errors, while others occasionally return responses that cannot be parsed. M3SunAgent achieves the best performance across all reported error metrics and produces valid predictions for all 2,910 test samples. It also obtains substantially lower errors than GPT-5. Moreover, 52.61% of its predictions have a relative error below 25%, demonstrating both high accuracy and stable output quality.

![](images/e73b2e0095eaefe23a2ed87018ef24d3f9bcca0a1e253d259542b1c01bd07d4e.jpg)

Fig. 5. Distribution of per-instance relative depth errors on M3SI. Each stacked bar shows the percentage of test instances in five mutually exclusive categories: errors below 10%, in [10%, 20%), [20%, 25%), or [25%, 50%), and errors of at least 50% or invalid predictions.  
![](images/9fdf9735dd42758271e5e32caa69539c3056b5f32fa2b326d049a0aced9946ce.jpg)  
Fig. 6. Qualitative comparison of M3SunAgent and general-purpose VLMs on real-world instance-level metric depth estimation. Each column shows the image and query, the ground-truth distance (GT), and the models’ responses. Green boxes mark the referred objects, while red dots indicate the depth-samplin locations used by M3SunAgent.

Task-specific spatial VLMs do not consistently perform well on this task. SpaceLLaVA shows limited accuracy despite being trained on large-scale spatial data. DepthVLM adds a dense depth prediction head to LLM, but this design does not improve instance-level metric depth estimation on our benchmark. In contrast, M3SunAgent combines the target localization capability of an open-vocabulary detector with the metric depth estimation capability of a specialized VLM. This combination enables accurate target localization and reliable depth estimation, resulting in stable instance-level metric depth predictions across most samples.

TABLE II  
COMPARISON OF 3D VISUAL GROUNDING ACCURACY (%) ON THE MONO3DREFER [2] TEST SET. THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Year</td><td colspan="3">Acc@0.25</td><td colspan="3">Acc@0.5</td></tr><tr><td>Unique</td><td>Multiple</td><td>Overall</td><td>Unique</td><td>Multiple</td><td>Overall</td></tr><tr><td colspan="8">CV Methods</td></tr><tr><td>Cube R-CNN + Rand [2]</td><td>2024</td><td>32.76</td><td>13.36</td><td>17.02</td><td>14.61</td><td>7.21</td><td>8.60</td></tr><tr><td>Cube R-CNN + Best [2]</td><td>2024</td><td>35.29</td><td>60.52</td><td>55.77</td><td>16.67</td><td>32.99</td><td>29.92</td></tr><tr><td>ZSGNet + backproj [2]</td><td>2024</td><td>9.02</td><td>16.56</td><td>15.14</td><td>0.29</td><td>2.23</td><td>1.87</td></tr><tr><td>FAOA + backproj [2]</td><td>2024</td><td>11.96</td><td>13.79</td><td>13.44</td><td>2.06</td><td>2.12</td><td>2.11</td></tr><tr><td>ReSC + backproj [2]</td><td>2024</td><td>11.96</td><td>23.69</td><td>21.48</td><td>0.49</td><td>3.94</td><td>3.29</td></tr><tr><td>Mono3DVG [2]</td><td>2024</td><td>57.65</td><td>65.92</td><td>64.36</td><td>33.04</td><td>46.85</td><td>44.25</td></tr><tr><td>Mono3DVG-TGE [28]</td><td>2025</td><td>62.45</td><td>69.83</td><td>68.44</td><td>44.22</td><td>52.83</td><td>51.21</td></tr><tr><td colspan="8">General-Purpose VLMs</td></tr><tr><td>MiMo-VL-7B [47]</td><td>2025</td><td>0.81</td><td>1.18</td><td>1.11</td><td>0.00</td><td>0.06</td><td>0.05</td></tr><tr><td>Qwen2.5-VL-72B [42]</td><td>2025</td><td>0.69</td><td>0.08</td><td>0.20</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Gemini-2.5-Flash</td><td>2025</td><td>0.23</td><td>0.86</td><td>0.74</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GPT-5</td><td>2025</td><td>2.09</td><td>6.89</td><td>5.98</td><td>0.36</td><td>0.21</td><td>0.23</td></tr><tr><td>GPT-03</td><td>2025</td><td>1.71</td><td>3.04</td><td>2.79</td><td>0.19</td><td>0.18</td><td>0.18</td></tr><tr><td>Gemini-2.5-Pro</td><td>2025</td><td>0.81</td><td>2.04</td><td>1.81</td><td>0.00</td><td>0.09</td><td>0.07</td></tr><tr><td colspan="8">Task-Specific VLMs</td></tr><tr><td>MonoVLM-Qwen [8]</td><td>2026</td><td>53.99</td><td>63.77</td><td>61.89</td><td>31.02</td><td>39.08</td><td>38.13</td></tr><tr><td>MonoVLM-MiMo [8]</td><td>2026</td><td>61.59</td><td>71.23</td><td>69.41</td><td>32.68</td><td>45.34</td><td>42.96</td></tr><tr><td>M3SunAgent (Ours)</td><td>一</td><td>67.94</td><td>71.58</td><td>70.90</td><td>36.47</td><td>44.10</td><td>42.66</td></tr></table>

TABLE III

COMPARISON OF 3D VISUAL GROUNDING ACCURACY (%) ON SELECTED SUBSETS OF THE MONO3DREFER TEST SET. THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="2">Near</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Moderate</td></tr><tr><td>Acc@0.25</td><td>Acc@0.5</td><td>Acc@0.25</td><td>Acc@0.5</td><td>Acc@0.25</td><td>Acc@0.5</td><td>Acc@0.25</td><td>Acc@0.5</td></tr><tr><td colspan="9">CV Methods</td></tr><tr><td>Cube R-CNN + Rand [2]</td><td>2024</td><td>17.40</td><td>11.45</td><td>21.12</td><td>11.41</td><td>18.01</td><td>8.15</td><td>17.85</td><td>8.01</td></tr><tr><td>Cube R-CNN + Best [2]</td><td>2024</td><td>67.76</td><td>41.45</td><td>59.66</td><td>33.05</td><td>60.69</td><td>30.35</td><td>60.56</td><td>33.45</td></tr><tr><td>ZSGNet + backproj [2]</td><td>2024</td><td>24.87</td><td>0.59</td><td>21.33</td><td>3.35</td><td>16.74</td><td>3.71</td><td>13.87</td><td>0.63</td></tr><tr><td>FAOA + backproj [2]</td><td>2024</td><td>18.03</td><td>0.53</td><td>17.51</td><td>3.43</td><td>15.64</td><td>3.95</td><td>12.18</td><td>1.34</td></tr><tr><td>ReSC + backproj [2]</td><td>2024</td><td>33.68</td><td>0.59</td><td>27.90</td><td>5.71</td><td>24.03</td><td>6.15</td><td>19.23</td><td>1.97</td></tr><tr><td>Mono3DVG [2]</td><td>2024</td><td>64.74</td><td>53.49</td><td>72.36</td><td>51.80</td><td>75.44</td><td>55.48</td><td>69.23</td><td>48.66</td></tr><tr><td>Mono3DVG-TGE [28]</td><td>2025</td><td>68.02</td><td>56.64</td><td>76.91</td><td>60.30</td><td>78.49</td><td>61.87</td><td>73.66</td><td>56.48</td></tr><tr><td colspan="9">General-Purpose VLMs</td></tr><tr><td>MiMo-VL-7B [47]</td><td>2025</td><td>1.63</td><td>0.10</td><td>0.69</td><td>0.00</td><td>0.73</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Qwen2.5-VL-72B [42]</td><td>2025</td><td>0.18</td><td>0.00</td><td>0.21</td><td>0.00</td><td>0.24</td><td>0.00</td><td>0.25</td><td>0.00</td></tr><tr><td>Gemini-2.5-Flash</td><td>2025</td><td>1.13</td><td>0.00</td><td>0.64</td><td>0.00</td><td>0.40</td><td>0.00</td><td>0.58</td><td>0.00</td></tr><tr><td>GPT-5</td><td>2025</td><td>5.14</td><td>0.07</td><td>4.81</td><td>0.08</td><td>7.12</td><td>0.50</td><td>7.18</td><td>0.38</td></tr><tr><td>GPT-o3</td><td>2025</td><td>3.01</td><td>0.15</td><td>2.98</td><td>0.18</td><td>3.06</td><td>0.13</td><td>3.33</td><td>0.14</td></tr><tr><td>Gemini-2.5-Pro</td><td>2025</td><td>2.72</td><td>0.15</td><td>1.65</td><td>0.12</td><td>1.29</td><td>0.00</td><td>1.24</td><td>0.00</td></tr><tr><td colspan="9">Task-Specific VLMs</td></tr><tr><td>MonoVLM-Qwen [8]</td><td>2026</td><td>74.06</td><td>49.72</td><td>68.53</td><td>40.03</td><td>54.52</td><td>33.08</td><td>55.64</td><td>34.05</td></tr><tr><td>MonoVLM-MiMo [8]</td><td>2026</td><td>78.15</td><td>52.18</td><td>74.14</td><td>45.25</td><td>63.74</td><td>36.89</td><td>64.80</td><td>38.40</td></tr><tr><td>M3SunAgent (Ours)</td><td></td><td>84.74</td><td>64.34</td><td>77.34</td><td>50.73</td><td>71.57</td><td>40.00</td><td>68.94</td><td>43.24</td></tr></table>

The stacked bar chart in Fig. 5 further compares the models under different relative error thresholds. M3SunAgent achieves the highest accuracy at all four thresholds. Under the strict 10% threshold, it reaches 23.71%, while the best competing model, Molmo2-8B, achieves 13.23%. Under the more relaxed 50% threshold, M3SunAgent achieves 85.09%, whereas the second-best model reaches 59.48%. These consistent gains indicate that M3SunAgent provides more accurate and reliable instance-level metric depth estimates across both strict and moderate error tolerances.

TABLE IV  
GROUNDING MIOU (%) ON THE MONO3DREFER TEST SET ACROSS OBJECT OCCURRENCE, DISTANCE, AND DIFFICULTY SUBSETS. THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Year</td><td colspan="9">mIoU (%) ↑</td></tr><tr><td colspan="3">Object Occurrence</td><td colspan="3">Distance</td><td colspan="3">Difficulty</td></tr><tr><td>Unique</td><td>Multiple</td><td>Overall</td><td>Near</td><td>Medium</td><td>Far</td><td>Easy</td><td>Moderate</td><td>Hard</td></tr><tr><td colspan="10">General-Purpose VLMs</td></tr><tr><td>MiMo-VL-7B [47]</td><td>2025</td><td>1.45</td><td>1.76</td><td>1.71</td><td>2.25</td><td>1.45</td><td>0.94</td><td>0.95</td><td>1.47</td><td>2.84</td></tr><tr><td>Qwen2.5-VL-72B [42]</td><td>2025</td><td>1.33</td><td>0.79</td><td>0.89</td><td>0.99</td><td>0.85</td><td>0.74</td><td>1.00</td><td>0.92</td><td>0.72</td></tr><tr><td>Gemini-2.5-Flash</td><td>2025</td><td>1.02</td><td>1.35</td><td>1.29</td><td>1.82</td><td>0.94</td><td>0.66</td><td>1.00</td><td>1.23</td><td>1.72</td></tr><tr><td>GPT-5</td><td>2025</td><td>5.91</td><td>7.91</td><td>7.53</td><td>8.16</td><td>6.96</td><td>6.94</td><td>7.44</td><td>7.42</td><td>7.73</td></tr><tr><td>GPT-03</td><td>2025</td><td>4.83</td><td>6.18</td><td>5.93</td><td>6.36</td><td>5.99</td><td>5.05</td><td>5.81</td><td>6.29</td><td>5.77</td></tr><tr><td>Gemini-2.5-Pro</td><td>2025</td><td>1.58</td><td>2.55</td><td>2.37</td><td>3.25</td><td>1.95</td><td>1.14</td><td>2.03</td><td>2.24</td><td>2.90</td></tr><tr><td colspan="10">Task-Specific VLMs</td></tr><tr><td>MonoVLM-Qwen [8]</td><td>2026</td><td>25.31</td><td>30.02</td><td>29.13</td><td>34.35</td><td>25.97</td><td>22.58</td><td>31.11</td><td>26.35</td><td>28.90</td></tr><tr><td>MonoVLM-MiMo [8]</td><td>2026</td><td>33.34</td><td>39.22</td><td>38.11</td><td>43.46</td><td>34.80</td><td>31.46</td><td>40.02</td><td>35.69</td><td>37.68</td></tr><tr><td>M3SunAgent (Ours)</td><td>一</td><td>38.61</td><td>42.46</td><td>41.73</td><td>54.33</td><td>40.51</td><td>30.53</td><td>46.67</td><td>41.72</td><td>34.83</td></tr></table>

Positioned on the left side of the road between a silver car and a black car, the second car on my left is a red one, standing still and facing towards me. Its height is estimated at around 1.4 meters and it is situated approximately 10 to 20 meters away from me at a northwesterly direction of 30 degrees.

At a distance of around 10 meters from me, there is a silver-colored car parked on the right side of the road. It's the first car in the line and is facing away from me. The car is positioned at a bearing of roughly 20 degrees north-east.

A black car, located approximately 30 degrees northwest of my position and about 10 meters away, is traveling in the left lane and situated closest to me. This first car is oriented with its back facing me as it travels straight ahead.

At a distance of about 10 meters from me, the first car on the right side of the road, about 4.3 meters long and color black, is parked with its back facing towards me, at a direction of approximately 20 degrees north-east from my position.

The black car is parked on the left side of the road, approximately 10 degrees northwest of me. It is within 20 meters of my position, and it's facing towards me. It is the first car on the left and is located in front of the second black car.

![](images/38680d1c6834690b2cf7fa3d5e748b110fc4f6b223327be1ece979abff68962e.jpg)  
Fig. 7. Qualitative comparison on five representative samples from the Mono3DRefer test set. Each column shows a referring description and an input image followed by the 3D bounding boxes predicted by GPT-5, Gemini-2.5-Pro, Gemini-2.5-Flash, Mono3DVG, and M3SunAgent. The boxes are shown as image projections and in 3D views. Blue indicates the ground truth, red indicates baseline predictions, and green indicates M3SunAgent predictions.

2) Qualitative Evaluation: Fig. 6 presents a qualitative comparison between M3SunAgent and general-purpose VLMs on instance-level metric depth estimation. The test images were captured with a mobile phone. The camera intrinsics were estimated from the EXIF metadata of each image. During image capture, a handheld laser rangefinder was used to measure the actual distance from the target object to the camera. This measurement was used as the ground-truth distance. The top row of the figure shows the target localization results produced by M3SunAgent. The green bounding boxes indicate the detected target regions, while the red points denote the sampling locations selected within each box. M3SunAgent can identify the referred target from the natural language query and predict a physical distance close to the laser measurement, even in real-world mobile-phone images. General-purpose VLMs can often recognize the target and understand the query. However, their distance estimates are usually coarse and rely heavily on visual and semantic priors. Some examples show large numerical errors, vague descriptions, or no valid distance value.

C. Comparison with State-of-the-Art Monocular 3D Visual Grounding Methods

1) Quantitative Evaluation: Table II compares M3SunAgent with the baseline methods using Acc@0.25 and Acc@0.5 under the Unique, Multiple, and Overall settings. Table III reports these metrics on the Near and Medium distance subsets and the Easy and Moderate difficulty subsets. Table IV presents the mIoU results across different object occurrence, target distance, and localization difficulty categories. As can be seen, M3SunAgent matches or exceeds the MonoVLM variants on most major metrics. It also remains competitive with task-specific vision methods such as Mono3DVG.

Tables II and IV show that M3SunAgent performs consistently under both the Unique and Multiple settings. On the Unique subset, it achieves an Acc@0.25 of 67.94% and an mIoU of 38.61%. On the Multiple subset, M3SunAgent reaches an Acc@0.25 of 71.58% and an mIoU of 42.46%. Its mIoU is 3.24 percentage points higher than that of MonoVLM-MiMo. Under the stricter Acc@0.5 criterion, M3SunAgent achieves an overall accuracy of 42.66%, close to the 42.96% obtained by MonoVLM-MiMo.

M3SunAgent performs well on nearby and less occluded targets. On the Near and Easy subsets, it achieves Acc@0.25 scores of 84.74% and 77.34%, together with mIoU scores of 54.33% and 46.67%, respectively. On the Medium and Moderate subsets, the corresponding mIoU scores are 40.51% and 41.72%. Its Acc@0.25 scores are 71.57% and 68.94%, while its Acc@0.5 scores are 40.00% and 43.24%, respectively. On the Far and Hard subsets, M3SunAgent obtains mIoU scores of 30.53% and 34.83%. These results rank second among the VLM-based methods and are 0.93 and 2.85 percentage points lower than those of MonoVLM-MiMo, respectively. The decrease from Near to Far and from Easy to Hard indicates that distant targets and severe occlusion or truncation remain challenging for accurate 3D localization. Nevertheless, M3SunAgent remains competitive among the VLM-based methods across all reported distance and difficulty subsets.

2) Qualitative Evaluation: Fig. 7 presents a qualitative comparison between M3SunAgent and several representative baseline methods. MonoVLM are not involved in comparison due to lack of open-source mdoel weight. General-purpose VLMs can infer the approximate target location from the language description. Their estimates of object length, width, and height are also reasonably close to the ground truth. However, their predictions of the yaw angle and 3D center contain large errors. These errors reduce the overlap between the predicted and ground-truth 3D bounding boxes. In some cases, the predicted box is completely displaced from the target. M3SunAgent accurately identifies the referred object from detailed language descriptions and produces 3D bounding boxes with greater overlap. Its estimates of the object dimensions, 3D center, and yaw angle are close to the corresponding ground-truth values.

## V. CONCLUSIONS

This paper presents M3SunAgent, a monocular spatial understanding agent for instance-level metric depth estimation and monocular 3D visual grounding. Given an RGB image and a natural-language query, a LLM generates an executable program that coordinates specialized visual models and geometric operations. For depth estimation, the agent localizes the referred object and aggregates pointwise predictions to obtain an instance-level metric distance. For 3D visual grounding, it combines VLM-based target localization and spatial attribute prediction with camera back-projection and specialized lifting to estimate the object’s 3D bounding box. This shared framework connects language-guided target identification with quantitative spatial estimation across both tasks. We also introduce the M3SI Dataset, which contains 2,910 samples for evaluating instance-level metric depth estimation. Experiments on M3SI show that M3SunAgent achieves the best overall performance across the reported depth metrics. On Mono3DRefer, it achieves overall Acc@0.25 and Acc@0.5 scores of 70.90% and 42.66%, respectively. These results demonstrate the effectiveness of coordinating specialized tools to estimate the spatial properties of language-referred objects from a single RGB image. M3SunAgent may be useful to integrate physicsdriven controller and data-driven vision language action to address long horizon tasks from hybrid system point of view.

## REFERENCES

[1] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia, “Spatialvlm: Endowing vision-language models with spatial reasoning capabilities,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 14 455–14 465.

[2] Y. Zhan, Y. Yuan, and Z. Xiong, “Mono3dvg: 3d visual grounding in monocular images,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 6988–6996.

[3] Q. Wang, J. He, Y. Pan, S. Y. Yeo, X. Yang, and S. Li, “Monosr: Openvocabulary spatial reasoning on monocular images,” in Computer Vision – ECCV 2026. Springer, 2026, pp. 20–37.

[4] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, “Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 10 579–10 596, 2024.

[5] L. Liu, R. Zhu, J. Deng, Z. Song, W. Yang, and T. Zhang, “Plane2Depth: Hierarchical adaptive plane guidance for monocular depth estimation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 2, pp. 1136–1149, 2025.

[6] R. Zhu, C. Wang, Z. Song, L. Liu, J. He, J. Deng, T. Zhang, and Y. Zhang, “Scaledepth: Decomposing metric depth estimation into semantic-aware scale prediction and adaptive relative depth estimation,” IEEE Transactions on Circuits and Systems for Video Technology, 2026.

[7] Y. Li, M. Liu, Z. Li, Y. Bian, X. Wang, E. Zhai, and Y. Wang, “Mono3dvg-ensd: Enhanced spatial-aware and dimension-decoupled text encoding for monocular 3d visual grounding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 8, 2026, pp. 6726–6734.

[8] H. Qu, H. N. Mahjoub, V. Tadiparthi, K. Lee, and T. Chen, “Monovlm: Monocular 3d visual grounding with vision language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 30 986–30 996.

[9] S. Shao, Z. Pei, W. Chen, D. Sun, P. C. Chen, and Z. Li, “Monodiffusion: Self-supervised monocular depth estimation using diffusion model,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 4, pp. 3664–3678, 2024.

[10] K. Wu, F. Li, W. Liu, Q. Liu, and Y. Yang, “Corrupted bitstream semantic understanding by adaptive-modal large language models,” Pattern Recognition, vol. 180, p. 114151, 2026.

[11] W. Ma, L. Ye, C. de Melo, A. L. Yuille, and J. Chen, “Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025.

[12] C. H. Song, V. Blukis, J. Tremblay, S. Tyree, Y. Su, and S. Birchfield, “Robospatial: Teaching spatial understanding to 2d and 3d visionlanguage models for robotics,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 15 768–15 780.

[13] S. D. Bhat and T. Yamasaki, “Consistent yet wrong: Evidence insensitivity in spatial vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2026, pp. 7463–7472.

[14] S. Chen, M. A. Uy, C. H. Song, F. Ladhak, A. Murali, Q. Qu, S. Birchfield, V. Blukis, and J. Tremblay, “Spacetools: Tool-augmented spatial reasoning via double interactive rl,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 37 109–37 120.

[15] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face,” Advances in Neural Information Processing Systems, vol. 36, pp. 38 154–38 180, 2023.

[16] T. Gupta and A. Kembhavi, “Visual programming: Compositional visual reasoning without training,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14 953–14 962.

[17] S. Shao, Z. Pei, W. Chen, P. C. Chen, and Z. Li, “Iebins: Iterative elastic bins for monocular depth estimation and completion,” International Journal of Computer Vision, vol. 133, no. 5, pp. 2463–2486, 2025.

[18] C. Wei, M. Yang, L. He, and N. Zheng, “Fs-depth: Focal-and-scale depth estimation from a single image in unseen indoor scene,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 11, pp. 10 604–10 617, 2024.

[19] A. Bochkovskiy, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. Richter, and V. Koltun, “Depth pro: Sharp monocular metric depth in less than a second,” in International Conference on Learning Representations, 2025. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ hash/bc8b2058fd96978a4146f18298cb2d39-Abstract-Conference.html

[20] S. Hou, M. Fu, R. Wang, Y. Yang, and W. Song, “Self-supervised monocular depth estimation for all-day images based on dual-axis transformer,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 10, pp. 9939–9953, 2024.

[21] X. Wang, W. Yin, T. Kong, Y. Jiang, L. Li, and C. Shen, “Task-aware monocular depth estimation for 3d object detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, 2020, pp. 12 257–12 264.

[22] L. Peng, X. Wu, Z. Yang, H. Liu, and D. Cai, “Did-m3d: Decoupling instance depth for monocular 3d object detection,” in European Conference on Computer Vision. Springer, 2022, pp. 71–88.

[23] C.-F. Yeh, H. Xu, Z. Liu, G. P. Meyer, X. Lei, C. Zhao, S.-W. Li, V. Chandra, Y. Shi et al., “Depthlm: Metric depth from vision language models,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 134 491–134 505.

[24] J. Liang, K. Wu, X. Hu, and T. Liu, “Cibic: Pixel-free foundation model for robust corrupted image bitstream captioning,” Pattern Recognition, p. 114238, 2026.

[25] C. Wang, W. Feng, S. Lyu, G. Cheng, X. Li, B. Liu, and Q. Zhao, “A masked reference token supervision-based iterative visual-language framework for robust visual grounding,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 1, pp. 75–90, 2025.

[26] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object localization in rgb-d scans using natural language,” in European conference on computer vision. Springer, 2020, pp. 202–221.

[27] L. Geng, J. Yin, G. Chen, and Q. Jia, “Pseudo-ev: Enhancing 3d visual grounding with pseudo embodied viewpoint,” IEEE transactions on circuits and systems for video technology, vol. 35, no. 8, pp. 8031–8044, 2025.

[28] Y. Li, M. Liu, Y. Bian, X. Wang, Z. Li, G. Li, and Y. Wang, “Dual enhancement on 3d vision-language perception for monocular 3d visual grounding,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 4552–4561.

[29] A.-C. Cheng, Y. Fu, Y. Ji, L. Zhu, G. Zhan, Z. Zhang, Z. Yang, S. Han, Y. Lu, P. Molchanov, V. N. Murali, J. Kautz, X. Wang, H. Yin, and S. Liu, “Grounded 3d-aware spatial vision-language modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 16 688–16 700.

[30] J. Gao, K.-H. Yap, K. Wu, D. T. Phan, K. Garg, and B. S. Han, “Contextual human object interaction understanding from pre-trained large language model,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 13 436–13 440.

[31] D. Marsili, R. Agrawal, Y. Yue, and G. Gkioxari, “Visual agentic ai for spatial reasoning with a dynamic api,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 19 446–19 455.

[32] Z. Li, Y. Soh, and C. Wen, Switched and Impulsive Systems: Analysis, Design and Applications, ser. Lecture Notes in Control and Information Sciences. Berlin, Heidelberg: Springer-Verlag, 2005, vol. 313.

[33] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.

[34] K. Wu, Y. Yang, M. Yu, and Q. Liu, “Block-wise focal stack image representation for end-to-end applications,” Optics Express, vol. 28, no. 26, pp. 40 024–40 043, 2020.

[35] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[36] W. Huang, J. Zhang, S. Li, J. Duan, Y. Cheng, J. Cho, M. Wallingford, R. Soraki, C. D. Kim, S. Liu et al., “Wilddet3d: Scaling promptable 3d detection in the wild,” 2026.

[37] T. Schops, J. L. Sch ¨ onberger, S. Galliani, T. Sattler, K. Schindler,¨ M. Pollefeys, and A. Geiger, “A multi-view stereo benchmark with highresolution images and multi-camera videos,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[38] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354–3361.

[39] P. Agrawal, S. Antoniak, E. B. Hanna, B. Bout, D. Chaplot, J. Chudnovsky, D. Costa, B. De Monicault, S. Garg, T. Gervet et al., “Pixtral 12b,” arXiv preprint arXiv:2410.07073, 2024.

[40] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[41] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.

[42] S. Bai et al., “Qwen2.5-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2502.13923

[43] C. Clark, J. Zhang, Z. Ma, J. S. Park, R. Tripathi, S. Lee, M. Salehi, J. Ren, C. D. Kim, Y. Yang et al., “Molmo2: Open weights and data for vision-language models with video understanding and grounding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 28 652–28 668.

[44] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892– 34 916, 2023.

[45] M. Jia, Z. Qi, S. Zhang, W. Zhang, X. Yu, J. He, H. Wang, and L. Yi, “Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 35 634–35 670.

[46] H. Yu, X. Qu, Y. Wang, J. Zhu, and L. Ke, “Unlocking dense metric depth estimation in vlms,” arXiv preprint arXiv:2605.15876, 2026.

[47] Core Team et al., “Mimo-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2506.03569