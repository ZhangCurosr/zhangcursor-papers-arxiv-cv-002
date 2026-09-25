# Retrieve-to-Localize: Bridging Large Language Models and LiDAR Geometry for Spatial Grounding

Byounggun Park<sup>1,∗</sup>, Giyong Moon<sup>2,∗</sup>, Jusung Kim<sup>1</sup>, and Soonmin Hwang<sup>2,†</sup>

Abstract— LiDAR provides precise geometric information for spatial perception tasks such as object detection in autonomous driving and outdoor robotics. However, recognizing and localizing individual objects is not sufficient to answer questions that require composing spatial relations and grounding the intended target. Motivated by recent advances in large language models (LLMs) for autonomous driving, we leverage their language priors to interpret complex spatial questions and ground the referred target in LiDAR geometry. To support this spatial grounding capability, we introduce SpatialLiDAR-QA, which combines single- and multi-step relational grounding with complementary spatial understanding tasks. We further propose SpatialLiDAR-LM, which aligns LiDAR point features with an LLM and grounds target coordinates through languageconditioned, position-aware proposal retrieval and local point refinement. This design derives target coordinates directly from local LiDAR geometry rather than through textual language decoding. Experiments demonstrate substantial improvements over representative LiDAR–language models and multi-camera VLMs on precise coordinate prediction tasks. Our dataset and model training code will be publicly released.

## I. INTRODUCTION

Recent advances in multimodal learning have enabled language-based understanding across diverse modalities, including images [1], [2], speech and audio [3], point clouds [4], [5], and radar [6]. These developments have encouraged the use of large language models (LLMs) in autonomous driving, connecting sensor observations with question answering, reasoning, and decision-making [7], [8]. In driving and outdoor robotics environments, such capabilities require understanding not only what objects are present, but also where they are and how they are spatially related.

Within this broader research direction, LiDAR–language models have established capabilities in object detection, captioning, and question answering [9], [10]. Building on these advances, we focus on spatial grounding: questions that couple relational reasoning with precise localization. For example, a question may ask for the object in front of another object that is itself to the left of a specified reference. Answering it requires identifying the reference, following the first relation to an intermediate object, and resolving the second relation to the final target. Predicting the target’s class and coordinates then requires connecting this relational interpretation to the observed geometry. As illustrated in Fig. 1, this motivates tasks that jointly require reference identification, relation composition, and precise coordinate prediction, rather than evaluating object recognition or location prediction in isolation.

![](images/847c953910f27c9a4cf2a1a49969e04f310a78f2570aecb90de59fadb4967eb5.jpg)  
Fig. 1: LiDAR provides precise geometry for object detection, but spatial questions require more than localizing individual objects: relations must be composed to ground the intended target.

Constructing such tasks requires spatial relations that are clearly specified and supervision that connects their interpretation to target locations. Existing datasets provide relational understanding, object detection, and geometry-based spatial evaluation [9], [11], [12]. In particular, some relational QA formulations represent object-to-object directions through six angular sectors referenced to the ego vehicle’s forward axis [11]. These sectors support relational questions but describe broad spatial regions with limited metric specificity. Their boundaries can also make directional labels sensitive to small changes in relative position. When multiple objects satisfy the same directional relation, the relation label alone may not uniquely identify the intended referent. Categorical relational answers alone do not verify whether the target is precisely localized, while object detection alone fails to test whether the target is identified through relational reasoning. Supervision that systematically connects these capabilities for both training and evaluation remains limited. These gaps motivate a dataset that combines clearly defined relations with single- and multi-step relational grounding requiring precise coordinate predictions, together with complementary spatial understanding tasks to address these challenges.

To this end, we introduce SpatialLiDAR-QA, a LiDARnative dataset of 108,811 QA pairs constructed from nuScenes keyframes [13]. We define direct directional relations from relative object-center coordinates, with the specified reference object determining the relational origin and the ego vehicle defining fixed directional axes. The generation process filters out directionally ambiguous cases and composes valid relations into single- and multi-step relational grounding questions that require predicting the final target’s class and LiDAR-frame coordinates. Complementary Localization and Object Identification tasks assess object-to-coordinate and coordinate-to-class correspondence, while maneuver feasibility evaluates the use of surrounding geometry for ego-centric decisions. Together, these tasks provide spatial grounding supervision, but a model must still distinguish the referred instance from similar candidates and recover its coordinates from the supporting LiDAR geometry. Existing LiDAR–language models demonstrate that aligned scene features can support spatial grounding. Nevertheless, decoding location tokens from these representations does not explicitly derive the predicted coordinates from the points supporting the selected target [9].

We therefore propose SpatialLiDAR-LM, an end-to-end architecture that aligns point features with an LLM. Its Point-Retrieved Localization (PRL) module uses the LLM’s contextual language representation to retrieve question-relevant spatial proposals and refine the target coordinate from their supporting local points. Proposal-Key Positional Encoding incorporates proposal locations into retrieval, helping distinguish similar instances according to the spatial conditions expressed in the question. This design connects languageconditioned target selection with explicit spatial geometry, enabling the model to use LiDAR’s spatial information when answering spatial grounding questions.

Our contributions are summarized as follows:

• We introduce SpatialLiDAR-QA, which couples explicitly defined spatial relations with single- and multistep relational grounding and complementary spatial understanding tasks, supporting both training and evaluation of LiDAR–language spatial grounding.

• We propose SpatialLiDAR-LM, which connects contextual language representations to languageconditioned, position-aware proposal retrieval and local point refinement, deriving target coordinates from question-relevant LiDAR geometry.

• We evaluate SpatialLiDAR-LM on SpatialLiDAR-QA and demonstrate substantial improvements over representative LiDAR–language and multi-camera models on precise coordinate prediction.

## II. RELATED WORK

## A. LiDAR-Based MLLMs for Outdoor Scene Understanding

Recent LiDAR-based MLLMs have extended language understanding to outdoor point-cloud scenes. BEV-LLM [14] combines LiDAR and multi-view image features in a unified BEV representation for scene captioning and uses absolute positional encoding to generate view-specific descriptions. LiDAR-LLM [9] supports captioning, question answering, and 3D grounding by aligning detector-pretrained BEV features with an LLM. Its Position-Aware Transformer divides BEV features into six camera-aligned views and injects the corresponding positional embeddings into the features and queries, and its grounding task generates location tokens specifying 3D boxes. B4DL [10] extends LiDAR–language modeling to sequential frames for spatio-temporal tasks, including temporal reasoning and time grounding, using aligned frame-wise features and ego-motion metatokens.

Building on these advances, our model connects relational spatial reasoning with precise coordinate estimation from LiDAR geometry.

## B. Spatial Reasoning Datasets for Autonomous Driving

Existing autonomous-driving QA datasets provide useful supervision for scene understanding, but differ in how spatial relations are represented and evaluated. NuScenes-QA [11] constructs 3D scene graphs with object-to-object relations defined by six angular categories relative to the ego vehicle’s forward axis. Although this supports relational questions, each category describes a broad spatial sector, and its existence, counting, recognition, status, and comparison tasks produce categorical answers rather than precise target coordinates. Its formulation therefore evaluates relational QA without directly requiring precise localization of the referred object. LiDAR-LLM [9] provides captioning and 3D grounding supervision, but its captioning tasks are generated from camera images using 2D MLLMs and GPT-4 filtering, and its tasks do not systematically combine precise localization with compositional object-relative reasoning. NuScenes-SpatialQA [12] uses ground-truth 3D geometry to construct spatial understanding and reasoning questions. However, its object references are derived from cameraimage captions, and it is designed for image-based VLM evaluation rather than providing training supervision for LiDAR–language alignment. To complement these benchmarks, SpatialLiDAR-QA supports both training and evaluation on tasks requiring spatial reasoning, such as singleand multi-step relational grounding with precise coordinate prediction.

## III. SPATIALLIDAR-QA DATASET

As illustrated in Fig. 2, SpatialLiDAR-QA defines five complementary tasks for training and evaluation.

## A. Dataset Details

1. Direct Localization: Given an object category, the model predicts the LiDAR-frame center coordinates of the instance closest to the ego vehicle, testing distance-based instance selection and precise localization.

2. Object Identification: Given a LiDAR-frame coordinate (x, y), the model predicts the object class at that location, testing spatial-to-semantic correspondence.

3. Single-Step Relational Grounding: Given a reference object and a directional relation along the ego-aligned axes, the model predicts the target’s class and LiDAR-frame coordinates. Partially occluded targets are also included, subject to the criteria in Sec. III-B.

4. Multi-Step Relational Grounding: The model composes two directional relations through an intermediate object to predict the final target’s class and LiDAR-frame coordinates, keeping the ego-aligned axes fixed.

5. Maneuver Feasibility: Given the surrounding object configuration, the model judges whether continuing straight or changing lanes to either side is geometrically feasible, testing ego-centric free-space reasoning.

<table><tr><td rowspan=1 colspan=2>Target Object            Reference Object            Intermediate Object</td></tr><tr><td rowspan=1 colspan=1>Localization</td><td rowspan=1 colspan=1>Single/Multi-step Relation</td></tr><tr><td rowspan=1 colspan=1>Q : Identify the [x, y] center coordinates of the car that is closest tothe ego vehicle.A : localization(target=obj(xy=[6.9,-1.6]))</td><td rowspan=1 colspan=1>Q : Starting from the car at [-14.55, -5.90], there is a car to itsback. What are the [x, y] center coordinates of the car directly tothe right of that car ?A ： multi_chain(target=obj(xy=[-10.4,-14.3]))</td></tr><tr><td rowspan=1 colspan=1>Maneuver9</td><td rowspan=1 colspan=1>Object Identification</td></tr><tr><td rowspan=1 colspan=1>Q : Is it geometrically feasible for the ego vehicle to execute a leftlane change maneuver safely right now?A ： maneuver(feasible=false,maneuver=&quot;left_lane_change&quot;)</td><td rowspan=1 colspan=1>Q: Which type of object is located within a 1-meter radius of thecoordinates [-2.95, -27.09]?A : object_at(target_cls=&quot;bicycle&quot;)</td></tr></table>

Fig. 2: Overview of the five spatial grounding tasks in SpatialLiDAR-QA: Direct Localization, Object Identification, Single/Multi-Step Relational Grounding, and Maneuver Feasibility, all formulated from LiDAR observations.

TABLE I: Distribution of QA pairs in SpatialLiDAR-QA.
<table><tr><td>Task Category</td><td>Training</td><td>Validation</td><td>Total</td></tr><tr><td>Localization</td><td>24,586</td><td>5,273</td><td>29,859</td></tr><tr><td>Object Identification</td><td>5,998</td><td>1,303</td><td>7,301</td></tr><tr><td>Single-step Relational</td><td>27,050</td><td>5,826</td><td>32,876</td></tr><tr><td>Multi-step Relational</td><td>8,152</td><td>2,400</td><td>10,552</td></tr><tr><td>Maneuver Feasibility</td><td>23,762</td><td>4,461</td><td>28,223</td></tr><tr><td>Total</td><td>89,548</td><td>19,263</td><td>108,811</td></tr></table>

Dataset Statistics. Table I summarizes the distribution of QA pairs across task categories. Following the official nuScenes splits [13], SpatialLiDAR-QA comprises 89,548 training and 19,263 validation QA pairs, totaling 108,811 pairs across nuScenes keyframes.

## B. Dataset Construction Process

SpatialLiDAR-QA is generated on official nuScenes keyframes [13] using geometry-verified rules across eight object categories (car, truck, bus, trailer, construction vehicle, pedestrian, bicycle, motorcycle). Strict filtering removes ambiguous directional cases, while maneuver-feasibility labels are balanced to avoid skewed answer distributions. Specifically, following the difficulty criteria of the Waymo Open Dataset [15], objects containing fewer than 5 LiDAR points are filtered out to eliminate heavily occluded or unidentifiable instances.

We express ground-truth object centers in a BEV frame centered at the LiDAR sensor, with ego-aligned axes such that +x points right and +y forward. Accordingly, front, back, right, and $l e f t$ correspond to +y, −y, +x, and −x, respectively. Relative displacements are measured from the reference object along these fixed axes rather than its heading, maintaining a consistent directional convention across reference changes in multi-step questions. A directional relation is accepted only when the displacement along the corresponding axis is at least three times the perpendicular displacement magnitude. Thus, diagonal cases remain undefined rather than being assigned a directional label.

For each reference–direction pair, we select an admissible target within 3–20 m, excluding candidates outside this range. The pair is discarded if the distances of the two nearest candidates to the reference differ by less than 2 m. We further reject locally crowded configurations and pairs that fail a geometric-clearance check along the connecting segment. These filters reduce ambiguity in target identification when composing relations across multiple steps. Human verification was conducted across QA pairs to manually validate geometric correctness and question–answer consistency.

## IV. METHODS

We present SpatialLiDAR-LM, an end-to-end architecture that aligns an LLM with a general-purpose 3D pointcloud backbone. SpatialLiDAR-LM preserves point-cloud representations and grounds language in their spatial structure as illustrated in Fig. 3.

## A. Model Architecture

LiDAR Point Feature Encoding. Rather than using proposals produced by a task-specific 3D detector, we use Point-TransformerV3 (PTv3) [16], initialized from a checkpoint trained for LiDAR semantic segmentation on nuScenes [13]. PTv3 converts the full LiDAR scan into point-level features, which are projected into the LLM embedding space by a lightweight two-layer MLP and processed jointly with the question.

![](images/5d6caddcfd37950e80f15c3bfbf18c711407d8431b5681943e429bea11cfb53d.jpg)  
Fig. 3: Overview of SpatialLiDAR-LM. LiDAR point tokens and questions are processed in a shared LLM embedding space. For coordinate prediction, PRL constructs spatial proposals through proposal scoring and xy-NMS, retrieves a questionrelevant proposal using the retrieval query and position-aware proposal keys, and refines its LiDAR-frame coordinate from local point features.

Language Model. We fine-tune the LLM using LoRA [17] to jointly process the projected point tokens and the tokenized question, in which coordinates are discretized into learned bin tokens (0.25 m resolution over ±54 m). Coordinate prediction is handled by the PRL module described in Sec. IV-B.

## B. Point-Retrieved Localization

Several localization-oriented MLLMs formulate coordinate prediction as sequential token generation, representing locations with textual numbers or discrete location tokens [18], [19]. LiDAR-LLM also supports location-tokenbased grounding [9]. Our contribution is not the introduction of a localization token, but how the localization-token representation is connected to the observed geometry. Rather than using this representation solely to decode coordinates, Point-Retrieved Localization (PRL) uses it to retrieve a questionrelevant spatial proposal and derives the final coordinate from the local points supporting that proposal. This provides an explicit geometric readout through proposal retrieval and local point refinement.

Spatial Proposal Selection. Directly searching over all point tokens retains substantial spatial redundancy. PRL therefore assigns each encoded point feature $\mathbf { F } ^ { ( i ) }$ , corresponding to the i-th point token, a query-independent proposal score,

$$
a _ { i } = g _ { \mathrm { p r o p } } ( \mathbf { F } ^ { ( i ) } ) + \eta ~ \mathrm { s g } \Big ( \| \mathbf { F } ^ { ( i ) } \| _ { 2 } \Big ) ,\tag{1}
$$

where $g _ { \mathrm { p r o p } }$ is a learned scalar projection and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. The feature-energy term $( \eta = 0 . 1 )$ provides a non-learned ranking signal before $g _ { \mathrm { p r o p } }$ becomes informative, while detaching it prevents gradients through this prior from directly encouraging larger feature magnitudes. The learned component of $a _ { i }$ is optimized through the grounding objective without separate objectness supervision (Sec. IV-D), and the score does not predict object categories or boxes. PRL then selects up to M spatially separated proposals according to these scores, using non-maximum suppression in the xy plane to avoid repeatedly sampling the same region. Let ${ \bf o } _ { j } \in \mathbb { R } ^ { 2 }$ denote the initial LiDAR-frame location of the j-th proposal and $\mathbf { c } _ { i } \in \mathbb { R } ^ { 2 }$ the coordinate of the i-th point token. We define the local neighborhood around the j-th proposal as $\mathcal { N } ( j ) = \{ i : \| \mathbf { c } _ { i } - \mathbf { o } _ { j } \| _ { 2 } \leq r _ { \mathrm { a g g } } \}$ . The features and coordinates of the point tokens in $\mathcal { N } ( j )$ are then aggregated as

$$
\begin{array} { l } { { \displaystyle w _ { i j } = \mathrm { s o f t m a x } _ { i \in \mathcal { N } ( j ) } ( a _ { i } ) } , } \\ { { \displaystyle ~ { \bf f } _ { j } = \sum _ { i \in \mathcal { N } ( j ) } w _ { i j } { \bf F } ^ { ( i ) } } , \qquad { \bf p } _ { j } = \sum _ { i \in \mathcal { N } ( j ) } w _ { i j } { \bf c } _ { i } , } \end{array}\tag{2}
$$

where $w _ { i j }$ is the normalized weight of the i-th point token within $\mathcal { N } ( j )$ . As defined in Eq. 2, each proposal is represented by a local feature $\mathbf { f } _ { j }$ and its LiDAR-frame center $\mathbf { p } _ { j }$

Language-Conditioned Retrieval and Refinement. Prior MLLMs extend the language vocabulary with task-specific tokens and use their hidden representations to condition non-linguistic decoders, such as mask prediction in LISA and 3D grounding in LLaVA-3D [20], [21]. Following this paradigm, we add a localization token <loc> to the LLM vocabulary and replace each coordinate in the response that requires spatial grounding with this token during training. The language modeling objective therefore teaches the model when a geometric readout is required. Let $\mathbf { h } _ { \mathrm { l o c } }$ denote the contextual hidden state at the generated <loc> position. PRL projects it into a retrieval query $\mathbf { q } = \mathbf { h } _ { \mathrm { l o c } } W _ { q }$ , which carries the question context and the model’s resolved referent. For the j-th proposal, we construct a retrieval key $\mathbf { k } _ { j }$ using its aggregated feature $\mathbf { f } _ { j }$ and positional information, as described in Sec. IV-C.

The query is matched against the proposal keys using scaled dot-product attention to retrieve the relevant scene

region:

$$
\begin{array} { l l } { { \displaystyle s _ { j } = \frac { { \bf q } { \bf k } _ { j } ^ { \top } } { \sqrt { d _ { \mathrm { a t t n } } } } , \qquad \alpha _ { j } = \mathrm { s o f t m a x } ( { \bf s } ) _ { j } , } } \\ { { \displaystyle \tilde { \bf p } = \sum _ { j = 1 } ^ { M } \alpha _ { j } { \bf p } _ { j } , \qquad \ j ^ { * } = \arg \operatorname* { m a x } _ { j } s _ { j } } . } \end{array}\tag{3}
$$

As shown in Eq. 3, the soft proposal distribution yields the coarse coordinate $\tilde { \mathbf { p } } ,$ while $j ^ { * }$ selects the local region used for refinement. PRL then compares the query with the point tokens around the selected center, $\mathcal { N } _ { \mathrm { r e f } } ( j ^ { * } ) ~ =$ $\{ i : \| \mathbf { c } _ { i } - \mathbf { p } _ { j ^ { * } } \| _ { 2 } \leq r _ { \mathrm { a g g } } \}$ , and predicts the final coordinate as

$$
\hat { \mathbf { p } } = \sum _ { i \in \mathcal { N } _ { \mathrm { r e f } } ( j ^ { * } ) } \beta _ { i } \mathbf { c } _ { i } , \qquad \beta _ { i } = \mathrm { s o f t m a x } \left( \frac { \mathbf { q } ( \mathbf { F } ^ { ( i ) } W _ { k } ^ { \prime } ) ^ { \top } } { \sqrt { d _ { \mathrm { a t t n } } } } \right) _ { i } .\tag{4}
$$

Here, $\alpha _ { j }$ and $\beta _ { i }$ denote the proposal-level and local pointlevel attention weights, respectively. The refinement in Eq. 4 grounds the final prediction in the observed geometry by taking an attention-weighted combination of the local point coordinates. When the selected proposal has insufficient point support, PRL retains the coarse estimate p˜. Fig. 4 shows both stages on a validation example.

## C. Proposal-Key Positional Encoding (Key PE)

Spatial Ambiguity. Proposal features describe the local geometry around each candidate, but they may not sufficiently distinguish multiple instances with similar structure. This is common in driving scenes containing several vehicles or pedestrians, where semantic similarity alone can lead the query to retrieve the wrong instance. The issue becomes more pronounced for relational questions, which require the model to distinguish candidates using both their semantic content and their LiDAR-frame position.

Position-Aware Proposal Keys. To expose this spatial information during retrieval, we augment each semantic proposal key with an encoding of its LiDAR-frame center:

$$
\mathbf { k } _ { j } = \mathbf { f } _ { j } W _ { k } + W _ { p } \phi ( \mathbf { p } _ { j } ) .\tag{5}
$$

As shown in Eq. 5, the positional term is added only to the proposal key used for retrieval. For a LiDAR-frame proposal center normalized by the perceptual range R, we use Fourier features [22] over B logarithmically spaced frequency bands:

$$
\phi ( \mathbf { p } _ { j } ) = \left[ \sin \left( \frac { \pi 2 ^ { b } \mathbf { p } _ { j } } { R } \right) , \cos \left( \frac { \pi 2 ^ { b } \mathbf { p } _ { j } } { R } \right) \right] _ { b = 0 } ^ { B - 1 } \in \mathbb { R } ^ { 4 B } .\tag{6}
$$

Equation 6 exposes the proposal location at multiple spatial scales, allowing the query to distinguish geometrically similar candidates by position while retaining their semantic features. The positional projection $W _ { p }$ is initialized to zero, so the model starts from the position-agnostic proposal keys and gradually learns the positional residual.

![](images/4b8f3c244c052b4d0ceff1f1d055e88b0b46be50d67c3454cadd974723f57d6a.jpg)  
Reference Object Ground truth Selected proposal � Refined prediction �ˆ �∗

Fig. 4: Point-Retrieved Localization (PRL) on a validation example. The retrieval query first selects a questionrelevant spatial proposal using proposal-level attention $\alpha _ { j } ,$ after which local attention $\beta _ { i }$ over the point tokens in $\mathcal { N } _ { \mathrm { r e f } }$ refines the final coordinate. The red box denotes the groundtruth bounding box.

## D. Training Objective and Inference

SpatialLiDAR-LM is trained jointly for language generation and coordinate prediction using the Smooth L1 loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { P R L } } \left[ \mathrm { S m o o t h L 1 } ( \hat { \mathbf { p } } , \mathbf { p } _ { \mathrm { g t } } ) + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } \right] .\tag{7}
$$

Following the multi-positive formulation of supervised contrastive learning [23], the instance-aware contrastive term is defined over the proposal retrieval scores from Eq. 3:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \frac { 1 } { | \mathcal { P } ^ { + } | } \sum _ { j \in \mathcal { P } ^ { + } } \log \frac { \exp ( s _ { j } / \tau ) } { \sum _ { m = 1 } ^ { M } \exp ( s _ { m } / \tau ) } ,\tag{8}
$$

where ${ \mathcal { P } } ^ { + }$ contains proposals whose centers fall within the target box enlarged by 0.5 m along each horizontal axis; when the box extent is unavailable, proposals within 2.0 m of $\mathbf { p } _ { \mathrm { g t } }$ are used. The contrastive term is omitted when no proposal satisfies this criterion, and $\tau$ is a fixed temperature. In Eq. 7, $\mathcal { L } _ { \mathrm { L M } }$ is the causal language modeling loss, the Smooth L1 term supervises point-level coordinate refinement, and Eq. 8 directly trains the language-conditioned score $s _ { j }$ to retrieve target-consistent proposals. The query-independent score $a _ { i }$ instead selects the proposal set. Seed selection and NMS are discrete, but the local aggregation in Eq. 2 remains differentiable, allowing $\mathcal { L } _ { \mathrm { c o n } }$ to update $g _ { \mathrm { p r o p } }$ through the resulting proposal features.

At inference, PRL constructs spatial proposals directly from the encoded point tokens and requires neither target annotations nor an external detector.

## V. EXPERIMENTS

## A. Experimental Setup

Implementation Details. The LiDAR encoder is Point-TransformerV3 (PTv3) [16], pretrained on nuScenes, following its original preprocessing setup. As base LLMs, we use Qwen3.5-0.8B [24] and SmolVLM-500M-Instruct [25], with their vision encoders removed. For PRL, we empirically select $M = 9 6$ after evaluating M ranging from 48 to 128. We set $r _ { \mathrm { N M S } } = 2 . 0  { \mathrm { n } }$ and $r _ { \mathrm { a g g } } = 2 . 5 \mathrm { m }$ . Key PE uses $B = 6$ frequency bands with perceptual range $R = 5 4 { \mathrm { m } }$

TABLE II: Quantitative results (accuracy, %, within 0.5, 1.0, and 2.0 m) on Coordinate prediction tasks. Language CE predicts coordinates through language decoding instead of PRL; best results per base LLM are in bold.
<table><tr><td rowspan="2">Base LLM</td><td rowspan="2">Method</td><td rowspan="2">Modality</td><td colspan="3">Localization</td><td colspan="3">Single-Step</td><td colspan="3">Multi-Step</td></tr><tr><td>@0.5m</td><td>@1.0m</td><td>@2.0m</td><td>@0.5m</td><td>@1.0m</td><td>@2.0m</td><td>@0.5m</td><td>@1.0m</td><td>@2.0m</td></tr><tr><td rowspan="7">SmolVLM</td><td>SmolVLM-500M-Instruct</td><td>Multi-Camera</td><td>14.9</td><td>30.5</td><td>52.3</td><td>8.2</td><td>24.5</td><td>51.5</td><td>3.9</td><td>11.7</td><td>26.8</td></tr><tr><td>BEV-LLM [14]</td><td>Multi-Camera</td><td>2.4</td><td>4.1</td><td>6.5</td><td>8.5</td><td>23.5</td><td>49.1</td><td>3.8</td><td>11.2</td><td>25.3</td></tr><tr><td>BEV-LLM [14]</td><td>LiDAR</td><td>2.6</td><td>4.3</td><td>6.5</td><td>8.1</td><td>24.0</td><td>49.2</td><td>3.4</td><td>12.2</td><td>24.1</td></tr><tr><td>LiDAR-LLM [9]</td><td>LiDAR</td><td>0.5</td><td>1.6</td><td>5.3</td><td>3.1</td><td>11.5</td><td>34.7</td><td>0.9</td><td>5.8</td><td>21.8</td></tr><tr><td>SpatialLiDAR-LM (Language CE)</td><td>LiDAR</td><td>6.7</td><td>15.6</td><td>31.9</td><td>3.6</td><td>9.3</td><td>22.4</td><td>2.2</td><td>8.7</td><td>20.0</td></tr><tr><td>SpatialLiDAR-LM (Ours)</td><td>LiDAR</td><td>57.2</td><td>77.0</td><td>85.0</td><td>28.4</td><td>46.3</td><td>59.9</td><td>36.4</td><td>53.3</td><td>63.0</td></tr><tr><td rowspan="7">Qwen3.5-0.8B</td><td></td><td>Multi-Camera</td><td>38.5</td><td>63.1</td><td>79.3</td><td>12.4</td><td>32.5</td><td>58.9</td><td>8.5</td><td>21.5</td><td>42.4</td></tr><tr><td>BEV-LLM [14]</td><td>Multi-Camera</td><td>4.5</td><td>6.8</td><td>12.0</td><td>9.4</td><td>25.7</td><td>51.8</td><td>5.3</td><td>13.4</td><td>27.6</td></tr><tr><td>BEV-LLM [14]</td><td>LiDAR</td><td>2.4</td><td>4.3</td><td>6.4</td><td>10.6</td><td>26.0</td><td>49.3</td><td>3.8</td><td>12.6</td><td>25.6</td></tr><tr><td>LiDAR-LLM [9]</td><td>LiDAR</td><td>0.6</td><td>3.6</td><td>6.7</td><td>5.8</td><td>16.3</td><td>42.6</td><td>1.6</td><td>4.8</td><td>18.4</td></tr><tr><td>SpatialLiDAR-LM (Language CE)</td><td>LiDAR</td><td>10.3</td><td>22.7</td><td>41.0</td><td>6.4</td><td>19.0</td><td>45.5</td><td>5.8</td><td>12.7</td><td>29.5</td></tr><tr><td>SpatialLiDAR-LM (Ours)</td><td>LiDAR</td><td>57.6</td><td>78.2</td><td>85.4</td><td>34.7</td><td>56.2</td><td>71.4</td><td>44.1</td><td>62.9</td><td>74.0</td></tr></table>

The model is optimized end-to-end using AdamW [26] with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , and weight decay 0.01. LoRA $( r = 6 4 , \alpha = 1 2 8$ , dropout 0.05) is applied to all attention projection layers. We use cosine annealing with a warmup ratio of 0.03 and peak learning rates of $2 \times 1 0 ^ { - 4 }$ for the LoRA adapters and point projector and $1 \times 1 0 ^ { - 5 }$ for PTv3. The loss weights are $\lambda _ { \mathrm { { P R L } } } = \lambda _ { \mathrm { { c o n } } } = 0 . 5 .$ with temperature $\tau = 0 . 0 7$ . Training and evaluation use NVIDIA RTX A6000 GPUs (48 GB VRAM each); training runs for 2 epochs on four GPUs with a per-device batch size of 4 (effective batch size 16).

Baselines and Evaluation Metrics. We compare SpatialLiDAR-LM against LiDAR-LLM [9], BEV-LLM [14] with multi-camera and LiDAR inputs, and multi-camera VLM baselines. All baselines are re-implemented with the same base LLMs and trained on SpatialLiDAR-QA with the same prompt templates and LoRA schedule as our model. Following the multi-view setup used in driving VLMs and VLAs [8], [27], [28], the VLM baselines take synchronized surround-view images as input. For Localization and coordinate prediction tasks, we report accuracy within 0.5 m, 1.0 m, and 2.0 m of the ground-truth location. For Class prediction and Maneuver Feasibility, we report exact-match accuracy. Unless otherwise specified, results are reported on the SpatialLiDAR-QA validation split.

## B. Quantitative Results

Coordinate prediction tasks. In Table II, SpatialLiDAR-LM achieves the highest accuracy on Localization and Single/Multi-Step Relational Grounding across all distance thresholds for both base LLMs. With Qwen3.5 at 0.5 m, SpatialLiDAR-LM reaches 57.6% on Localization, compared with 38.5% for the multi-camera VLM, and 44.1% on Multi-Step Relational Grounding, substantially higher than the VLM’s 8.5%. With both SmolVLM and Qwen3.5, the same proposal retrieval and local point refinement design yields consistent gains in localization and relational grounding. This consistency shows that SpatialLiDAR-LM is compatible with different base LLMs.

TABLE III: Ablation of the grounding mechanism and Key PE. Accuracy (%) is reported within 0.5, 1.0, and 2.0 m of the ground-truth location.
<table><tr><td>Base LLM</td><td>Task</td><td>Threshold</td><td>Language CE</td><td>Smooth L1</td><td>PRL</td><td>PRL + Key PE</td></tr><tr><td>SmolVLM</td><td>Localization</td><td>@0.5m</td><td>6.7</td><td>1.0</td><td>54.8</td><td>57.2</td></tr><tr><td></td><td></td><td>@1.0m</td><td>15.6</td><td>4.1</td><td>76.5</td><td>77.0</td></tr><tr><td></td><td></td><td>@2.0m</td><td>31.9</td><td>14.7</td><td>84.6</td><td>85.0</td></tr><tr><td></td><td>Single-Step</td><td>@0.5m</td><td>3.6</td><td>5.9</td><td>14.5</td><td>28.4</td></tr><tr><td></td><td></td><td>@1.0m</td><td>9.3</td><td>16.9</td><td>25.7</td><td>46.3</td></tr><tr><td></td><td></td><td>@2.0m</td><td>22.4</td><td>42.8</td><td>38.7</td><td>59.9</td></tr><tr><td></td><td>Multi-Step</td><td>@0.5m</td><td>2.2</td><td>1.2</td><td>17.9</td><td>36.4</td></tr><tr><td></td><td></td><td>@1.0m</td><td>8.7</td><td>4.9</td><td>27.8</td><td>53.3</td></tr><tr><td></td><td></td><td>@2.0m</td><td>20.0</td><td>20.9</td><td>35.8</td><td>63.0</td></tr><tr><td>Qwen3.5</td><td>Localization</td><td>@0.5m</td><td>10.3</td><td>3.2</td><td>53.4</td><td>57.6</td></tr><tr><td></td><td></td><td>@1.0m</td><td>22.7</td><td>11.5</td><td>74.4</td><td>78.2</td></tr><tr><td></td><td></td><td>@2.0m</td><td>41.0</td><td>29.9</td><td>82.3</td><td>85.4</td></tr><tr><td></td><td>Single-Step</td><td>@0.5m</td><td>6.4</td><td>5.2</td><td>26.2</td><td>34.7</td></tr><tr><td></td><td></td><td>@1.0m</td><td>19.0</td><td>18.7</td><td>44.1</td><td>56.2</td></tr><tr><td></td><td></td><td>@2.0m</td><td>45.5</td><td>46.3</td><td>56.2</td><td>71.4</td></tr><tr><td></td><td>Multi-Step</td><td>@0.5m</td><td>5.8</td><td>1.2</td><td>32.8</td><td>44.1</td></tr><tr><td></td><td></td><td>@1.0m</td><td>12.7</td><td>4.8</td><td>51.1</td><td>62.9</td></tr><tr><td></td><td></td><td>@2.0m</td><td>29.5</td><td>19.6</td><td>60.3</td><td>74.0</td></tr></table>

While other models struggle on relational grounding, SpatialLiDAR-LM achieves substantially higher accuracy on both Single-Step and Multi-Step, suggesting that it composes spatial relations reliably. Its lower accuracy on Single-Step than on Multi-Step is partly due to occlusion: since a Single-Step target lies close to its reference object, it is more often partially occluded and captured by fewer LiDAR points.

Grounding Mechanism Analysis. We further examine the grounding mechanism by comparing alternative coordinate prediction formulations using the same point encoder and base LLM (Table III). Language CE generates coordinates as text using token-level cross-entropy. The Smooth L1 baseline retains the same <loc> token as PRL but directly regresses coordinates from its contextual hidden state using the same Smooth L1 coordinate loss. With Qwen3.5, Localization accuracy at 0.5 m is 10.3% for Language CE and 3.2% for Smooth L1, compared with 53.4% for PRL.

At the same 0.5 m threshold, PRL also increases Localization accuracy from 6.7% to 54.8% with SmolVLM. On relational grounding, the gains over Language CE are 10.9 and 19.8 percentage points for Single-Step and 15.7 and

27.0 points for Multi-Step with SmolVLM and Qwen3.5, respectively. Smooth L1 also remains below PRL on all three tasks with both base LLMs at this threshold. These results highlight the benefit of using the contextual representation to retrieve question-relevant proposals and derive coordinates from their supporting local LiDAR points, rather than directly decoding or regressing coordinates.

Key PE Analysis. Key PE provides only modest gains on Localization but substantially larger gains on relational grounding. At 0.5 m, Localization accuracy improves by 2.4 percentage points with SmolVLM and 4.2 points with Qwen3.5, whereas Single-Step improves by 13.9 and 8.5 points, respectively, and Multi-Step by 18.5 and 11.3 points. This indicates that proposal-level positional information is particularly important for retrieving the correct proposal under relational constraints. Both the gains from PRL and the larger relational gains from Key PE are consistent across the two base LLMs.

TABLE IV: Quantitative results (exact-match accuracy, %) on Class prediction tasks and Maneuver Feasibility. C and L denote multi-camera and LiDAR.
<table><tr><td>Base LLM Method</td><td></td><td>Modality</td><td>Maneuver</td><td>Object Identification</td><td>Single-Step</td><td>Multi-Step</td></tr><tr><td rowspan="5">SmolVLM</td><td>SmolVLM-500M-Instruct</td><td>C</td><td>89.7</td><td>62.2</td><td>64.0</td><td>71.5</td></tr><tr><td>BEV-LLM [14]</td><td>C</td><td>60.6</td><td>24.0</td><td>58.8</td><td>70.4</td></tr><tr><td>BEV-LLM [14]</td><td>L</td><td>60.7</td><td>26.0</td><td>59.0</td><td>70.8</td></tr><tr><td>LiDAR-LLM [9]</td><td>L</td><td>69.0</td><td>16.0</td><td>57.0</td><td>64.6</td></tr><tr><td>SpatialLiDAR-LM (Ours)</td><td>L</td><td>87.2</td><td>38.0</td><td>59.4</td><td>71.8</td></tr><tr><td rowspan="5">Qwen3.5</td><td>Qwen3.5-0.8B</td><td>C</td><td>92.2</td><td>77.1</td><td>75.9</td><td>80.0</td></tr><tr><td>BEV-LLM [14]</td><td>C</td><td>81.9</td><td>29.5</td><td>60.2</td><td>72.7</td></tr><tr><td>BEV-LLM [14]</td><td>L</td><td>60.5</td><td>28.0</td><td>60.1</td><td>72.7</td></tr><tr><td>LiDAR-LLM [9]</td><td>L</td><td>59.4</td><td>27.1</td><td>60.3</td><td>71.6</td></tr><tr><td>SpatialLiDAR-LM (Ours)</td><td>L</td><td>85.5</td><td>73.0</td><td>58.2</td><td>72.4</td></tr></table>

Class prediction tasks and Maneuver Feasibility. Table IV evaluates categorical text outputs: Object Identification, the class predictions of Single- and Multi-Step Relational Grounding, and Maneuver Feasibility. The VLMs generally achieve the highest accuracy on these tasks, consistent with the richer semantic and contextual cues available in images. For example, with Qwen3.5, the multi-camera VLM achieves 92.2% on Maneuver Feasibility, compared with 85.5% for SpatialLiDAR-LM. Nevertheless, SpatialLiDAR-LM achieves higher or competitive performance compared with the other LiDAR-based baselines on most tasks. In particular, on Maneuver Feasibility, it reaches 87.2% with SmolVLM and 85.5% with Qwen3.5, substantially outperforming the LiDAR-based BEV-LLM and LiDAR-LLM baselines. SpatialLiDAR-LM is also competitive with the multi-camera VLM on selected semantic tasks. With SmolVLM, it achieves 71.8% on Multi-Step class prediction, compared with 71.5% for the VLM; with Qwen3.5, it reaches 73.0% on Object Identification, compared with 77.1% for the VLM. These results indicate that the proposed LiDAR–language architecture supports semantic recognition alongside precise coordinate grounding.

## C. Cross-Dataset Evaluation and Efficiency

We compare the zero-shot cross-dataset performance of SpatialLiDAR-LM and the multi-camera VLM on the Argoverse 2 (AV2) [29] validation split and the NAVSIM [30] test split. QA pairs for both datasets are generated using the same pipeline as SpatialLiDAR-QA (Sec. III-B). Both models use Qwen3.5-0.8B and are fine-tuned on the same nuScenesbased SpatialLiDAR-QA training split, without target-dataset fine-tuning. We use the coordinate-accuracy metrics defined in Sec. V-A and map target-dataset object classes to our taxonomy: AV2 covers seven of our eight classes, while NAVSIM provides vehicle, pedestrian, and bicycle labels. The VLM uses surround-view images as input.

TABLE V: Cross-dataset grounding accuracy (%). Both models use Qwen3.5-0.8B and are fine-tuned on nuScenesbased SpatialLiDAR-QA; evaluation on AV2 and NAVSIM is zero-shot. Distance thresholds are in meters.
<table><tr><td rowspan="2">Eval. dataset</td><td rowspan="2">Model</td><td colspan="3">Localization</td><td colspan="3">Single-Step</td><td colspan="3">Multi-Step</td></tr><tr><td>@0.5</td><td>@1.0</td><td>@2.0</td><td>@0.5</td><td>@1.0</td><td>@2.0</td><td>@0.5</td><td>@1.0</td><td>@2.0</td></tr><tr><td rowspan="2">nuScenes (source)</td><td>Ours</td><td>57.6</td><td>78.2</td><td>85.4</td><td>34.7</td><td>56.2</td><td>71.4</td><td>44.1</td><td>62.9</td><td>74.0</td></tr><tr><td>VLM</td><td>38.5</td><td>63.1</td><td>79.3</td><td>12.4</td><td>32.5</td><td>58.9</td><td>8.5</td><td>21.5</td><td>42.4</td></tr><tr><td rowspan="2">AV2 [29]</td><td>Ours</td><td>47.1</td><td>67.8</td><td>76.2</td><td>36.3</td><td>55.4</td><td>65.3</td><td>31.6</td><td>50.4</td><td>57.9</td></tr><tr><td>VLM</td><td>12.1</td><td>28.9</td><td>54.6</td><td>7.6</td><td>20.7</td><td>44.2</td><td>1.7</td><td>6.7</td><td>23.6</td></tr><tr><td rowspan="2">NAVSIM [30]</td><td>Ours</td><td>67.8</td><td>77.3</td><td>79.9</td><td>43.5</td><td>57.4</td><td>63.2</td><td>41.4</td><td>54.4</td><td>59.5</td></tr><tr><td>VLM</td><td>6.3</td><td>18.0</td><td>34.9</td><td>9.6</td><td>25.3</td><td>49.0</td><td>3.3</td><td>10.8</td><td>29.2</td></tr></table>

As shown in Table V, SpatialLiDAR-LM outperforms the multi-camera VLM on all three coordinate prediction tasks at every distance threshold on both target datasets. At 0.5 m, Localization accuracy reaches 47.1% on AV2 and 67.8% on NAVSIM, compared with 12.1% and 6.3% for the VLM, respectively. The consistent advantage across direct localization and relational grounding supports stronger zeroshot cross-dataset generalization of SpatialLiDAR-LM under the same source-dataset fine-tuning setup. These results suggest that language-conditioned proposal retrieval and local geometric readout transfer effectively beyond nuScenes.

We measure the per-query inference latency of SpatialLiDAR-LM over all 19,263 validation queries on an NVIDIA RTX A6000 with batch size 1. With Qwen3.5-0.8B as the language backbone, SpatialLiDAR-LM has a mean latency of 574.4 ms and peak allocated GPU memory of 5.60 GB. With SmolVLM-500M-Instruct as the language backbone, SpatialLiDAR-LM has a mean latency of 490.9 ms and peak allocated GPU memory of 4.76 GB.

## VI. CONCLUSION

In this work, we advance LiDAR–language understanding by connecting language-based spatial reasoning with the metric geometry of outdoor scenes. We introduce SpatialLiDAR-QA to support training and evaluation on tasks that require identifying objects, following or composing spatial relations, and grounding the intended target. SpatialLiDAR-LM addresses these tasks through language-conditioned retrieval and local point-based coordinate prediction, linking the LLM’s interpretation of a question to the observed LiDAR geometry. Experiments and ablation studies support the proposed design for precise localization and relational grounding. Competitive performance on selected semantic tasks and zero-shot cross-dataset results further suggest that this approach can support broader LiDAR–language understanding. Together, SpatialLiDAR-QA and SpatialLiDAR-LM provide a foundation for developing geometrically grounded language models that integrate semantic recognition, spatial reasoning, and precise localization. We will publicly release the dataset and training code to support further research toward language-guided spatial understanding in autonomous driving and outdoor robotics.

Limitations. While our approach effectively aligns language with precise LiDAR coordinates, the current formulation focuses on explicit directional relations and questionrelevant localization within a shared spatial frame. Future work will focus on strengthening semantic understanding and extending spatial grounding to temporal reasoning to support a broader range of tasks in autonomous driving and outdoor robotics.

## REFERENCES

[1] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892–34 916, 2023.

[2] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond,” 2023.

[3] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. Ma, and C. Zhang, “Salmonn: Towards generic hearing abilities for large language models,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 16 607–16 629.

[4] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3dllm: Injecting the 3d world into large language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 20 482–20 494.

[5] R. Xu, X. Wang, T. Wang, Y. Chen, J. Pang, and D. Lin, “Pointllm: Empowering large language models to understand point clouds,” in European Conference on Computer Vision. Springer, 2024, pp. 131– 147.

[6] R. Guan, R. Zhang, N. Ouyang, J. Liu, K. L. Man, X. Cai, M. Xu, J. Smith, E. G. Lim, Y. Yue, et al., “Talk2radar: Bridging natural language with 4d mmwave radar for 3d referring expression comprehension,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 10 884–10 891.

[7] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li, “Drivelm: Driving with graph visual question answering,” in European conference on computer vision. Springer, 2024, pp. 256–274.

[8] X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao, “DriveVLM: The convergence of autonomous driving and large vision-language models,” in 8th Annual Conference on Robot Learning, 2024.

[9] S. Yang, J. Liu, R. Zhang, M. Pan, Z. Guo, X. Li, Z. Chen, P. Gao, H. Li, Y. Guo, et al., “Lidar-llm: Exploring the potential of large language models for 3d lidar understanding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9247–9255.

[10] C. Choi, Y. Shin, G. Han, D.-J. Lee, and J. Kim, “B4dl: A benchmark for 4d lidar llm in spatio-temporal understanding,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 3399–3407.

[11] T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y.-G. Jiang, “Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 5, 2024, pp. 4542–4550.

[12] K. Tian, J. Mao, Y. Zhang, J. Jiang, Y. Zhou, and Z. Tu, “Nuscenesspatialqa: A spatial understanding and reasoning benchmark for visionlanguage models in autonomous driving,” in 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). IEEE, 2025, pp. 4626–4635.

[13] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2020, pp. 11 618–11 628.

[14] F. Brandstatter, E. Sch¨ utz, K. Winter, and F. B. Flohr, “Bev-llm:¨ Leveraging multimodal bev maps for scene captioning in autonomous driving,” in 2025 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2025, pp. 345–350.

[15] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 2443–2451.

[16] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang, T. He, and H. Zhao, “Point transformer v3: Simpler, faster, stronger,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 4840–4851.

[17] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.

[18] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, “Shikra: Unleashing multimodal llm’s referential dialogue magic,” arXiv preprint arXiv:2306.15195, 2023.

[19] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, Q. Ye, and F. Wei, “Grounding multimodal large language models to the world,” in The Twelfth International Conference on Learning Representations, 2024.

[20] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language model,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 9579–9589.

[21] C. Zhu, T. Wang, W. Zhang, J. Pang, and X. Liu, “Llava-3d: A simple yet effective pathway to empowering lmms with 3d capabilities,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 4295–4305.

[22] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 7537–7547.

[23] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” Advances in neural information processing systems, vol. 33, pp. 18 661–18 673, 2020.

[24] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5

[25] A. Marafioti, O. Zohar, M. Farre, M. noyan, E. Bakouch, P. M. C.´ Jimenez, C. Zakka, L. B. allal, A. Lozhkov, N. Tazi, V. Srivastav,´ J. Lochner, H. Larcher, M. Morlon, L. Tunstall, L. V. Werra, and T. Wolf, “SmolVLM: Redefining small and efficient multimodal models,” in Second Conference on Language Modeling, 2025.

[26] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019.

[27] X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li, “Holistic autonomous driving understanding by bird’s-eye-view injected multimodal large models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 13 668– 13 677.

[28] Z. Zhou, T. Cai, S. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma, “Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38, Main Conference. Curran Associates, Inc., 2025, pp. 27 920–27 956.

[29] B. Wilson, W. Qi, T. Agarwal, J. Lambert, J. Singh, S. Khandelwal, B. Pan, R. Kumar, A. Hartnett, J. K. Pontes, D. Ramanan, P. Carr, and J. Hays, “Argoverse 2: Next generation datasets for self-driving perception and forecasting,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021), 2021.

[30] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta, “Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.