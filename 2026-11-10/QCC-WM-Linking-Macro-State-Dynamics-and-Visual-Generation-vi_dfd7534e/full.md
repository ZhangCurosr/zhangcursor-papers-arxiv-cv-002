# QCC-WM: Linking Macro-State Dynamics and Visual Generation via Quotient Causal Construction

Zhizhong Fu<sup>a,1</sup>, Wei Zhou<sup>a,e,g,1</sup>, Zhaoyang Jiang<sup>b</sup>, Yudong Lin<sup>a</sup>, Songlun Li<sup>d</sup>, Zhi Li<sup>e</sup>, Feifan Chen<sup>g</sup>, Gaoming Li<sup>a</sup>, Xiaorong Ding<sup>a</sup>, Yifan Chen<sup>a,∗</sup>

<sup>a</sup>School ofLife Science and Technology, University ofElectronic Science and Technology ofChina, Chengdu, 610054, China <sup>b</sup>School ofHealth and Wellbeing, University ofGlasgow, Glasgow, G12 8TB, United Kingdom

<sup>c</sup>Department ofOrgan Transplantation, Sichuan Provincial People’s Hospital, University ofElectronic Science and Technology ofChina, Chengdu, China <sup>d</sup>School ofInformation and Software Engineering, University ofElectronic Science and Technology ofChina, Chengdu, 611731, China <sup>e</sup>Department ofRadiology, Huzhou Maternity & Child Health Care Hospital, Huzhou, 313000, China

<sup>f</sup>Hepatological Surgery Department, Huzhou Central Hospital, Fifth School ofClinical Medicine ofZhejiang Chinese Medical University, Huzhou, 313000, China <sup>g</sup>Department ofRadiology, Huzhou Maternity & Child Health Care Hospital Afiliated to Huzhou Normal University, Huzhou, 313000, China

## Abstract

A key challenge in action-conditioned world modeling is to construct macro-states from high-dimensional observations that support recursive prediction under actions, while keeping their predicted evolution consistent with visual changes in generated observations. We propose QCC-WM, a visual world model based on Quotient-based Causal Construction. QCC-WM uses action responses to guide macro-variable construction for recursive action-conditioned prediction. A shared reference quotient measurement constrains the consistency between predicted states and observed changes, allowing macro-level predictions to provide structured constraints for future visual generation. Using translational and rotational actions and their predicted responses, theV model aggregates local visual features into macro-nodes with spatial correspondence through causal abstraction, and recursivelyC predicts node content and motion under a shared reference measurement. Macro-level predictions guide future observation gen-<sup>.</sup> eration, while predicted motion transports historical appearance information. Node-level errors between generated observationsc and macro-level targets are then mapped back to the corresponding spatial regions for correction, improving consistency between<sup>[</sup> visual generation and macro-level predictions. On the US-VLA dataset, QCC-WM achieves an ROI-MSE of 0 01861 ± 0 00008 for four-step prediction. Compared with the best baseline at each prediction horizon, it reduces LPIPS by 48 59%, 41 68%, andv 30 23% for four-step prediction and eight- and sixteen-step recursive extrapolation, respectively. Ablation studies and mechanism3 analyses show that motion-guided reference transport mainly improves pixel reconstruction, while node-level error feedback further<sup>5</sup> improves perceptual quality and consistency between generated observations and macro-level predictions.

Keywords: Action-conditioned world models; quotient-based causal construction; macro-state prediction; motion-guided reference transport; spatial error correction; ultrasound video prediction

## 1. Introduction

Causal representation learning (CRL) aims to identify highlevel variables related to the underlying generative mechanisms from high-dimensional observations, providing representations<sup>r</sup> for analyzing action outcomes, intervention efects, and distribution shifts [1]. For world models, this goal further requires learned states to evolve recursively under actions while maintaining their correspondence with observable changes. However, a fundamental challenge in high-dimensional visual settings is that macro-level causal variables suitable for recursive prediction are generally not given and must be constructed from fine-grained observations. Local features can describe visual appearance and changes, but they do not necessarily form macro-states with consistent measurement definitions and predictable transition rules. Building actionconditioned world models therefore requires addressing both how macro-variables are formed and how they represent state evolution under successive actions.

Research on macro-variable construction and causal abstraction ofers theoretical insights into this problem. Chalupka et al. studied how macro-level cause and efect variables can be constructed from the causal behavior of micro-level measurements [2]. Hoel et al. showed that, for specific systems and under the efective information measure, suitable macro-level descriptions can exhibit greater causal efectiveness than microlevel descriptions [3]. Rubenstein et al. further emphasized that causal models at diferent levels should yield consistent predictions under explicitly defined intervention mappings [4]. These studies suggest that micro-to-macro mappings should be designed around the mechanisms and dynamics they preserve. For world models, we therefore consider a dynamics-oriented abstraction principle: macro-variable construction should be aligned with future prediction so that abstract state changes correspond to evolution at the observation level.

Existing visual world models have explored various state representations and prediction methods. DINO-WM learns actionconditioned dynamics in pretrained spatial features [5], while IRASim incorporates action trajectories into a difusion model to generate future observations [6]. For structured modeling, SlotFormer learns recursive dynamics over object-level representations [7], and Slot-JEPA organizes prediction through semantic slots and spatial masks [8]. Motion-I2V uses predicted motion to transport reference features, connecting motion modeling with visual generation [9]. These studies provide important foundations for state prediction, structured representations, and spatially organized generation. Recursive macro-state prediction nevertheless raises several questions: whether nodes have comparable measurement definitions across time steps, whether predicted state changes are constrained by observed evolution, and whether macro-level predictions retain the correspondence needed to map them back to observation space. These questions directly afect the preservation of state meaning and the accumulation of prediction errors during recursive prediction.

To address these issues, we introduce QCC (Quotient-based Causal Construction) and develop the action-conditioned world model QCC-WM. QCC uses action-guided soft quotient mappings to aggregate local visual features into macro-nodes with spatial support. It recursively predicts content and motion states through inter-node dependencies and action-conditioned transitions. Within each prediction window, the model fixes the reference quotient mapping and its alignment, and constrains macro-state prediction through supervision on states and their temporal increments. This encourages consistency with observed evolution under the same quotient measurement.

To translate macro-level predictions into observation generation, the model uses predicted motion to transport reference features and latent variables, construct observable macro-level targets, and correct the initial generation. It then reads back the generated observations through the same quotient measurement, computes node-level errors relative to the macro-level targets, and maps these errors back to local latent space through the node–space correspondence. The reference quotient mapping thus connects node construction, macro-state prediction, and spatial correction, linking macro-level dynamics with future observation generation.

Our main contributions are as follows:

1. An action-guided quotient-based causal construction mechanism, QCC. We construct macro-nodes with spatial support through soft quotient mappings, discover a macro-level causal structure of temporal dependencies among nodes under a shared reference measurement, and use this structure to organize action-conditioned dynamics. This connects macro-node construction and structure discovery with recursive prediction of content and motion states.

2. A macro-level dynamics learning method under a fixed reference quotient measurement. We fix the reference quotient mapping and content alignment within each prediction window to provide consistent measurement definitions for macro-states across time steps. Supervision on states and their temporal increments constrains recursive content and motion prediction, encouraging consistency between macro-level predictions and observed evolution under the same reference measurement.

3. A generation correction interface combining motion transport with readback under the same quotient measurement. We use predicted motion to transport historical appearance information, then map node-level errors obtained through readback to the generation latent space through spatial correspondence, improving consistency between future observations and macro-level predictions.

The remainder of this paper is organized as follows. Section 2 reviews related work on causal representation learning, structured world models, and motion-guided generation. Section 3 presents QCC-WM, including quotient-based causal construction, macro-state prediction, motion-guided reference transport, spatial correction, and staged optimization. Section 4 describes the experimental setup and evaluates model performance and module contributions through comparative experiments, ablation studies, and mechanism analyses. Section 5 concludes the paper and discusses future directions.

## 2. Related Work

2.1. Causal Representation Learning and Macro-Variable Construction

Causal representation learning seeks to identify latent variables related to generative mechanisms from high-dimensional observations, enabling mechanism analysis, intervention prediction, and generalization across distributions [1]. In dynamic visual settings, CITRIS uses temporal observations and known intervention targets to identify latent causal factors under specific assumptions [10]. BISCUIT studies causal variable learning with unknown binary interaction mechanisms and estab lishes corresponding identifiability conditions [11]. These studies show that causal interpretations depend closely on additional supervision and structural assumptions. QCC focuses on constructing macro-nodes for action-conditioned prediction. Its node measurements and dependency structures are learned and evaluated as predictive interfaces within the model; action inputs and interventions within the model alone do not provide such identifiability guarantees.

Macro-variable construction also concerns relationships between diferent levels of description. Chalupka et al. used causal partitions to connect micro-level measurements with macro-level cause and efect variables [2]. Hoel et al. ana lyzed macro-level causal efectiveness in specific systems using efective information [3]. Rubenstein et al. formalized in tervention consistency across levels through exact transformations between structural equation models [4]. In reinforcement learning, DeepMDP learns state representations by predicting rewards and next-state distributions in latent space, and estab lishes connections to bisimulation [12]. These studies provide a basis for abstractions that preserve underlying dynamics. We apply this idea to spatial aggregation within visual observations:

action-guided soft quotient mappings define macro-level measurements, whose future evolution is constrained to match observed changes under the same reference mapping. This spatial quotient construction addresses a diferent modeling object from full environment-state abstraction, and the prediction consistency imposed here does not imply strict intervention consistency.

## 2.2. Action-Conditioned Dynamics and Structured World Models

Latent world models support future prediction by learning transition rules in compact state spaces. PlaNet combines deterministic and stochastic transitions and introduces latent overshooting as a multi-step variational objective [13]. Dreamer optimizes behavior over imagined latent trajectories, using recursive state prediction for control [14]. DINO-WM learns actionconditioned patch dynamics from pretrained spatial features and performs planning through goal-feature prediction [5]. These methods connect latent state prediction with downstream decision-making. QCC further addresses how local observations are organized into macro-level prediction units. A shared reference measurement constrains node states and their temporal increments, providing comparable observation-based targets for recursive prediction in the abstract state space.

Structured world models explicitly organize prediction units within a scene. DINOSAUR learns object-level representations by reconstructing self-supervised pretrained features [15]. SlotFormer learns autoregressive dynamics over slots to predict future object states [7]. Slot-JEPA decomposes dense features into semantic slots and time-varying spatial masks, combining transition memory and transport objectives for prediction and planning [8]. These methods establish foundations for feature grouping, structured recursive prediction, and spatial correspondence. Our specific design incorporates action plans into quotient mapping construction and uses the same reference quotient mapping for macro-state measurement, future-state supervision, and mapping predictions back to spatial locations. QCC nodes are defined as action-conditioned aggregate measurements, whose predictive value must be assessed through appropriate controlled comparisons.

## 2.3. Generative World Models and Motion-Guided Spatial Feedback

Generative world models synthesize future observations from actions or structured states. IRASim models robotic interaction videos using a difusion Transformer with frame-wise action conditioning [6]. To connect structured representations with generation, SlotDifusion uses a slot-conditioned latent difusion decoder and applies the learned representations to object dynamics modeling [16]. SlotDiT generates future slot trajectories through text-conditioned difusion and then decodes them into images [17]. QCC uses macro-level content and motion as states for recursive prediction and converts these predictions into generation conditions in image latent space through explicit quotient mappings. Its focus is the interface between macro-level dynamics and observation generation.

Explicit motion modeling provides another way to connect state changes with visual synthesis. Motion-I2V first predicts pixel trajectories and then propagates reference image features through motion-enhanced temporal attention [9]. FLIP com bines flow generation, flow-conditioned video generation, and value modeling for generative planning in visual space [18]. DeXplicit predicts explicit 3D states and deformations from actions and transports appearance information to guide video generation [19]. These studies provide direct precedents for motion prediction and reference content transport. We use a learned visual quotient space as an intermediate representation, where regional motion and content changes arise from recursive macrostate prediction and are mapped back to local space through the retained quotient mapping. Spatial feedback thus forms a prediction pipeline together with QCC node measurements and recursive dynamics. Its contribution is assessed through generation evaluation and interventions on the model interfaces.

## 3. Methods

## 3.1. Problem Formulation and QCC Overview

Given historical visual observations $X = I _ { - L + 1 : 0 }$ and a known action plan $A = ( a _ { 0 } , \dotsc , a _ { H - 1 } )$ , we learn the conditional distribution $p ( Y \mid X , A )$ of future observations $Y = I _ { 1 : H }$ . The central problem is to construct macro-states that support actionconditioned prediction while preserving their connection to observation space, allowing their predictions to guide future generation.

We propose QCC (Quotient-based Causal Construction), which uses action-guided soft quotient mappings to organize local visual representations into macro-nodes with explicit spatial origins. QCC jointly defines the aggregation sources, reference measurements, and spatial correspondence of nodes, and establishes action-conditioned transitions and inter-node dependencies over the constructed macro-variables. We refer to these state units as macro-level causal nodes, which represent actiondriven evolution and inter-node interactions within the model.

As shown in Fig. 1, the framework consists of four components: quotient-based causal construction, action-conditioned macro-state prediction, motion-guided reference transport, and readback under the same quotient measurement followed by spatial correction:

$$
\begin{array} { r l } & { ( \Pi _ { 0 } ^ { b } , \Gamma _ { 0 } ^ { b } , S _ { 0 } ^ { b } ) = Q _ { \phi } ^ { b } ( X , A ^ { b } ) , } \\ & { \widehat { S } _ { k + 1 } ^ { b } = F _ { \theta } ^ { b } \Big ( \widehat { S } _ { k } ^ { b } , a _ { k } , A ^ { b } , \ell _ { k + 1 } ^ { b } ; G _ { z } ^ { b } , G _ { m } ^ { b } \Big ) , } \\ & { \widetilde { x } _ { 1 : H } \sim p _ { \psi } \big ( x _ { 1 : H } \mid X , A , C _ { \mathrm { Q C C } } \big ) , } \\ & { \widehat { I } _ { k } = D \Big ( x _ { k } ^ { \mathrm { M } } + \alpha _ { \mathrm { E } } \mathcal { R } _ { \Pi _ { 0 } } ( e _ { k } ; \cdot ) \Big ) , } \end{array}\tag{1}
$$

where $b \in \mathcal { B } = \{ \mathrm { T } , \mathrm { R } \}$ denotes the quotient construction branch guided by translational or rotational action components, and $A ^ { b }$ is the corresponding action guidance. Both dynamics models and the generator retain the complete action input. $\Pi _ { 0 } ^ { b }$ and $\Gamma _ { 0 } ^ { b }$ denote the reference soft assignment and reference content alignment, respectively, while $G _ { z } ^ { b } , G _ { m } ^ { b }$ specify the dependency supports for the content and motion channels. $S _ { k } ^ { b } \ =$ $( Z _ { k } ^ { b } , M _ { k } ^ { b } , g _ { k } ^ { z , b } , g _ { k } ^ { m , b } )$ contains node content, node motion memory, and their corresponding shared states; $\ell _ { k + 1 } ^ { b }$ is the node control derived from the action responses predicted by QCC. C<sub>QCC</sub> denotes the generation conditions produced by macro-state prediction, $\widetilde { x _ { k } }$ is the initially generated latent variable, $x _ { k } ^ { \mathrm { { M } } }$ is the latent variable after motion-guided reference correction, $e _ { k }$ is the remaining node-level error obtained through readback under the same quotient measurement, and D is the observation decoder.

## 3.2. Action-Response-Guided Quotient-Based Causal Construction

We use the pretrained visual encoder $E _ { \mathrm { o b s } }$ of I-JEPA [20] to extract observation representations and obtain frame-wise content features through branch-specific mappings: $F _ { k } ^ { b } \ =$ $\Phi ^ { b } ( E _ { \mathrm { o b s } } ( I _ { k } ) )$ . Historical features and their local temporal differences are used to organize nodes and initialize motion, while frame-wise content features measure node states at each time step. Each branch maintains K global prototypes $\{ c _ { j } ^ { b } \} _ { j = 1 } ^ { K } .$ , with prototype indices identifying the nodes. QCC uses the action plan to modulate the afinity between historical local features and prototypes. Combined with spatial distances, these afinities determine soft assignments $\Pi _ { p j } ^ { \hat { b } , \mathrm { p r o t o } }$ within the valid observation region. The assignments are normalized across nodes and refined by iteratively updating node centroids, thereby defining the local spatial support of each node.

Action-response-driven quotient refinement.. To connect node organization with potential action-induced changes, QCC further predicts local action responses from historical descriptors. For future time step k, let $\begin{array} { r } { \bar { a } _ { k } ^ { \bar { b } } = \sum _ { r = 0 } ^ { k - 1 } a _ { r } ^ { \bar { b } } } \end{array}$ denote the cumulative action guidance:

$$
\widehat { r } _ { k , p } ^ { b } = b _ { \phi , k } ^ { b } ( h _ { p } ^ { b } ) + J _ { \phi , k } ^ { b } ( h _ { p } ^ { b } , r _ { p } ) \bar { a } _ { k } ^ { b } ,\tag{2}
$$

where $b _ { \phi , k } ^ { b }$ is the response base term conditioned on history, and $J _ { \phi , k } ^ { b }$ parameterizes the interaction between local mechanisms and actions. The response includes cumulative motion and appearance changes relative to the reference latent variable, connecting feature grouping with the future changes to be represented.

We subtract the spatial mean of the action response term over the valid region, normalize it using training statistics, and concatenate it across time to form the local response descriptor $\chi _ { p } ^ { b } .$ This descriptor compares relative responses across locations under the current action plan. Starting from the prototype grouping, QCC sequentially refines spatial assignments using history-conditioned response encodings and explicit mechanism–action response encodings. Both refinements alternate between response aggregation and assignment updates. For each refinement, let $\bar { \Pi } ^ { b , \bar { \mathrm { p r i o r } } }$ denote the input assignment and $\chi ^ { b }$ the corresponding response descriptor:

$$
\begin{array} { l } { \displaystyle \bar { \chi } _ { j } ^ { b , ( n ) } = \frac { \sum _ { p } \Pi _ { p j } ^ { b , ( n ) } \chi _ { p } ^ { b } } { \sum _ { p } \Pi _ { p j } ^ { b , ( n ) } + \epsilon } , } \\ { \displaystyle \widetilde { \Pi } _ { p j } ^ { b , ( n + 1 ) } \propto m _ { p } \exp \left[ \log \left( \Pi _ { p j } ^ { b , \mathrm { p r i o r } } + \epsilon \right) - \lambda _ { r } \Vert \chi _ { p } ^ { b } - \bar { \chi } _ { j } ^ { b , ( n ) } \Vert _ { 2 } ^ { 2 } \right] , } \\ { \displaystyle \Pi _ { p j } ^ { b , \mathrm { n e w } } = ( 1 - \zeta _ { b } ) \Pi _ { p j } ^ { b , \mathrm { p r i o r } } + \zeta _ { b } \widetilde { \Pi } _ { p j } ^ { b } , } \end{array}\tag{3}
$$

where $\zeta _ { b }$ is a bounded strength learned separately for each refinement layer, and the final assignment is denoted by $\Pi _ { 0 } ^ { b } .$ Node assignments are thus jointly constrained by historical spatial structure and predicted action responses, organizing local evolution information while preserving prototype identities. At inference, action responses are predicted entirely from history and the known action plan; observed future responses are used only for training supervision.

Reference quotient measurement and node initialization.. QCC further uses historical descriptors and predicted action responses to produce a bounded reference content alignment $\Gamma _ { 0 } ^ { b } ,$ which remains fixed within the current prediction window. We define the normalized aggregation weights as

$$
\omega _ { 0 , p j } ^ { b } = \frac { \Pi _ { 0 , p j } ^ { b } } { \sum _ { q } \Pi _ { 0 , q j } ^ { b } + \epsilon } , \mathcal { P } _ { \Pi _ { 0 } ^ { b } , \Gamma _ { 0 } ^ { b } } ( F ) _ { j } = \sum _ { p } \omega _ { 0 , p j } ^ { b } ( \Gamma _ { 0 , p } ^ { b } \odot F _ { p } ) .\tag{4}
$$

The corresponding reference quotient measurement is

$$
\begin{array} { r } { \mathcal { M } _ { 0 } ^ { b } ( I ) = \mathcal { P } _ { \Pi _ { 0 } ^ { b } , \Gamma _ { 0 } ^ { b } } \Big ( \Phi ^ { b } ( E _ { \mathrm { o b s } } ( I ) ) \Big ) , \qquad Z _ { k } ^ { b , \star } = \mathcal { M } _ { 0 } ^ { b } ( I _ { k } ) . } \end{array}\tag{5}
$$

This definition unifies spatial assignment, content alignment, and frame-wise measurement. The initial node content is set to $Z _ { 0 } ^ { b } = Z _ { 0 } ^ { b , \star }$ . Motion memory is initialized by quotient aggregation of historical content changes, estimated historical optical flow, and local mechanism descriptors. Reference centroids are computed from the final assignment $\Pi _ { 0 } ^ { b } .$ Shared content and motion states are obtained through global aggregation over the valid region, providing a reference for overall changes to the local nodes.

## 3.3. Action-Conditioned Macro-State Prediction under a Fixed Quotient Measurement

Node dependencies and step-wise response control.. We construct candidate node dependency supports through temporal dependency analysis of macro-level measurements from training observations, incorporating action and reference conditions [21]. The analysis controls for the complete action plan, reference support, and past measurements of shared states. Fixed directed supports $G _ { z } ^ { b } , G _ { m } ^ { b }$ organize inter-node messages for the content and motion channels. For channel $c \in \{ z , m \}$ incoming edge weights are defined as

$$
B _ { i j } ^ { c , b } = \frac { G _ { c , i j } ^ { b } \mathrm { c l i p } ( \gamma _ { i j } ^ { c , b } , 0 , 1 ) } { 1 + \sum _ { l } G _ { c , l j } ^ { b } \mathrm { c l i p } ( \gamma _ { l j } ^ { c , b } , 0 , 1 ) } ,\tag{6}
$$

where $\gamma _ { i j } ^ { c , b }$ is a learnable strength within the support, and the direction from i to j represents information transfer from node i at the previous time step to node $j$ at the current time step. The directed supports remain fixed during recursive prediction, with content and motion using their respective message weights.

The action responses predicted by QCC also enter the recursive dynamics through step-wise node controls. We take temporal diferences of the cumulative motion responses in Eq. (2) and aggregate them under the reference quotient mapping:

$$
\ell _ { k , j } ^ { b } = \sum _ { p } \omega _ { 0 , p j } ^ { b } \psi _ { \ell } ^ { b } \Big ( \widehat { r } _ { k , p , \mathrm { m o t } } ^ { b } - \widehat { r } _ { k - 1 , p , \mathrm { m o t } } ^ { b } \Big ) , \quad \widehat { r } _ { 0 , p , \mathrm { m o t } } ^ { b } = 0 ,\tag{7}
$$

![](images/8b5449a3f5ee96b6d07082d029fc3f5d0431b46f1fe8711056e9015880e22436.jpg)  
Figure 1: Overview of QCC-WM. QCC first constructs macro-nodes with spatial origins and reference measurements. Action-conditioned dynamics recursively predict content and motion states. Predicted motion transports reference features and latent variables to construct observable macro-level targets and perform motion-guided reference correction, respectively. The model then reads back the current generation under the same quotient measurement and maps the remaining node-level errors to the corresponding spatial locations to obtain future observations. The reference quotient mapping connects node construction, macro-state prediction, and spatial generation.

where $\psi _ { \ell } ^ { b }$ is a bounded response encoding function. This control allows local action responses learned during construction to contribute to node updates at every step, jointly driving macrostate evolution with the complete action input.

Recursive evolution of shared and local states.. To separate overall changes from local diferences, we first update the shared states:

$$
( g _ { k + 1 } ^ { z , b } , g _ { k + 1 } ^ { m , b } ) = F _ { g } ^ { b } ( g _ { k } ^ { z , b } , g _ { k } ^ { m , b } , a _ { k } , A ^ { b } ) ,\tag{8}
$$

and define local residuals as ${ \widetilde { Z } } _ { k , j } ^ { b } ~ = ~ Z _ { k , j } ^ { b } - g _ { k } ^ { z , b }$ and $\widetilde { M } _ { k , j } ^ { b } ~ =$ $M _ { k , j } ^ { b } - g _ { k } ^ { m , b }$ . Content messages combine node content, motion, reference positions, and fixed historical context, while motion messages are extracted from motion residuals:

$$
\begin{array} { r l } & { \mathfrak { m } _ { k , j } ^ { z , b } = \displaystyle \sum _ { i } B _ { i j } ^ { z , b } V _ { z } ^ { b } \bigl ( \widetilde { Z } _ { k , i } ^ { b } , \widetilde { M } _ { k , i } ^ { b } , \mu _ { i } ^ { b } ; X \bigr ) , } \\ & { \mathfrak { m } _ { k , j } ^ { m , b } = \displaystyle \sum _ { i } B _ { i j } ^ { m , b } V _ { m } ^ { b } ( \widetilde { M } _ { k , i } ^ { b } ) , } \end{array}\tag{9}
$$

where $V _ { z } ^ { b } , V _ { m } ^ { b }$ denote the message projections for the respective channels. Inter-node messages use the states at the start of each step, while each node continues to evolve within its residual update blocks.

Node content is updated through a residual mapping, while motion memory is updated through a gated recurrent unit:

$$
\begin{array} { r l } & { \widetilde { Z } _ { k + 1 , j } ^ { b } = \mathcal { F } _ { z } ^ { b } \big ( \widetilde { Z } _ { k , j } ^ { b } , \widetilde { M } _ { k , j } ^ { b } , \Pi _ { k , j } ^ { z , b } , a _ { k } , A ^ { b } , \ell _ { k + 1 , j } ^ { b } , g _ { k } ^ { b } , \mu _ { j } ^ { b } ; X \big ) , } \\ & { Z _ { k + 1 , j } ^ { b } = \widetilde { Z } _ { k + 1 , j } ^ { b } + g _ { k + 1 } ^ { z , b } , } \\ & { \widetilde { M } _ { k + 1 , j } ^ { b } = \mathrm { G R U } ^ { b } \big ( \widetilde { M } _ { k , j } ^ { b } , \Psi _ { m } ^ { b } \big ( \widetilde { Z } _ { k + 1 , j } ^ { b } , \Pi _ { k , j } ^ { m , b } , a _ { k } , \ell _ { k + 1 , j } ^ { b } , \mu _ { j } ^ { b } \big ) \big ) , } \\ & { M _ { k + 1 , j } ^ { b } = \widetilde { M } _ { k + 1 , j } ^ { b } + g _ { k + 1 } ^ { m , b } , } \\ & { \widetilde { u } _ { k + 1 , j } ^ { b } = \rho _ { m } ^ { b } ( \widetilde { M } _ { k + 1 , j } ^ { b } ) + \rho _ { g } ^ { b } ( g _ { k + 1 } ^ { m , b } ) , } \end{array}\tag{10}
$$

where $g _ { k } ^ { b } = [ g _ { k } ^ { z , b } ; g _ { k } ^ { m , b } ] , \widehat { u } _ { k + 1 , j } ^ { b }$ is the node motion response, and $\mathcal { F } _ { z } ^ { b }$ is the parameterized residual update mapping for node content in branch b. This update combines each node’s history, dependency messages, shared changes, and local action responses to recursively predict future sequences from the same initial macro-state without future observations.

Content dynamics are learned under a fixed reference quotient measurement, connecting recursively predicted states with corresponding observed changes:

$$
\mathrm { p r } _ { z } F _ { \theta } ^ { b , ( k ) } ( S _ { 0 } ^ { b } , A ) \approx { \cal M } _ { 0 } ^ { b } ( I _ { k } ) ,\tag{11}
$$

where $\mathrm { p r } _ { z }$ extracts node content. Initial and future observations are measured using the same $\Pi _ { 0 } ^ { b } , \Gamma _ { 0 } ^ { b } \colon$ , providing a common ref-

erence across prediction steps. Supervision is applied to both states and their temporal increments:

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { \mathrm { Q } } = \Big \langle \lambda _ { z } { \bf M } { \bf S } \mathrm { E } ( \widehat { Z } ^ { b } , Z ^ { b , \star } ) + \lambda _ { n } { \bf M } { \bf S } \mathrm { E } \big ( \mathrm { L N } ( \widehat { Z } ^ { b } ) , \mathrm { L N } ( Z ^ { b , \star } ) \big ) } \\ & { } & { + \lambda _ { \Delta } { \bf M } { \bf S } \mathrm { E } \big ( \Delta \widehat { Z } ^ { b } , \Delta Z ^ { b , \star } \big ) \Big \rangle _ { b } , \quad \Delta Z _ { k } = Z _ { k } - Z _ { k - 1 } , } \end{array}\tag{12}
$$

where $\langle \cdot \rangle _ { b }$ denotes averaging over branches, and MSE is averaged over samples, prediction steps, nodes, and channels. This constraint aligns macro-states and their changes under a fixed observation reference, providing state sequences with a common measurement basis for subsequent spatial generation.

## 3.4. Quotient-Mapping-Based Motion-Guided Reference Transport

QCC first uses spatial correspondence to condition the initial generation on macro-states. The model performs frame-level and temporal readouts of node content and motion, and injects node content increments and motion geometry into the generator’s local features through the reference quotient mapping. For example, the spatial field corresponding to node content changes is

$$
\Delta F _ { k } ^ { b } ( p ) = \sum _ { j } \Pi _ { 0 , p j } ^ { b } ( \widehat { Z } _ { k , j } ^ { b } - Z _ { 0 , j } ^ { b } ) .\tag{13}
$$

Together with macro-level motion conditions, this field forms C and guides the generation of the initial latent block ${ \widetilde { x } } _ { 1 : H } ,$ while retaining historical visual information and direct action conditioning.

To further translate macro-level motion into changes in observation space, we use the same reference assignment to recover local displacement fields and accumulate them over prediction steps:

$$
\widehat { \nu } _ { k } ^ { b } ( p ) = \sigma _ { f } \sum _ { j } \Pi _ { 0 , p j } ^ { b } \widehat { u } _ { k , j } ^ { b } , \qquad d _ { k } ^ { b } ( p ) = \sum _ { r = 1 } ^ { k } \widehat { \nu } _ { r } ^ { b } ( p ) ,\tag{14}
$$

where $\sigma _ { f }$ converts normalized motion responses to the displacement scale required for transport. We define the reference transport operator through bilinear backward sampling as $\mathcal { W } _ { d } ( F ) ( p ) = F ( p - d ( p ) )$ ), with displacements interpolated and rescaled to the corresponding feature grid.

Reference information is transported separately in content feature space and generation latent space. For branch $^ { b , }$ the reference content features $F _ { 0 } ^ { b }$ are transported using the predicted displacement and measured through the fixed quotient interface to form an observable macro-level target:

$$
\widehat { C } _ { k } ^ { b } = \mathcal { P } _ { \Pi _ { 0 } ^ { b } , \Gamma _ { 0 } ^ { b } } \big ( \mathcal { W } _ { d _ { k } ^ { b } } ( F _ { 0 } ^ { b } ) \big ) ,\tag{15}
$$

where $\widehat { C } _ { k } ^ { b }$ represents the observable changes induced by predicted motion in the reference content and is used for comparison with generated observations under the same quotient measurement. The recursively predicted latent state $\widehat { Z } _ { k } ^ { b }$ supports macro-state prediction and generation conditioning. The two quantities thus serve latent prediction and observable correction, respectively.

In latent space, we average the displacement fields from the two branches to obtain $\bar { d } _ { k }$ and transport the latent variable $x _ { 0 }$ of the last historical observation:

$$
\bar { d } _ { k } = \frac 1 { | \mathcal B | } \sum _ { b } d _ { k } ^ { b } , \qquad x _ { k } ^ { \mathrm { r e f } } = \mathcal W _ { \bar { d } _ { k } } ( x _ { 0 } ) .\tag{16}
$$

QCC also converts node motion memory into a spatial gate:

$$
g _ { k } ( p ) = \kappa ~ { \mathrm { I n t e r p } } \left[ \frac { 1 } { | \mathcal { B } | } \sum _ { b } \sum _ { j } \Pi _ { 0 , p j } ^ { b } s i g m a \left( h _ { m } ^ { b } ( \widehat { M } _ { k , j } ^ { b } ) \right) \right] ,\tag{17}
$$

where $h _ { m } ^ { b }$ is the node motion gating network, $\sigma$ is the sigmoid function, and κ is a bounded strength. The gate depends on predicted motion memory and is expanded over the reference spatial support of each node. Motion-guided reference correction is defined as

$$
\boldsymbol { x } _ { k } ^ { \mathrm { M } } = \widetilde { \boldsymbol { x } } _ { k } + \alpha _ { \mathrm { M } } g _ { k } \odot ( \boldsymbol { x } _ { k } ^ { \mathrm { r e f } } - \widetilde { \boldsymbol { x } } _ { k } ) ,\tag{18}
$$

where $\widetilde { x _ { k } }$ is the initially generated latent variable, $x _ { k } ^ { \mathrm { r e f } }$ is the reference latent variable transported using predicted motion, $g _ { k }$ contains spatial gating weights, $\alpha _ { \mathbf { M } }$ controls the overall correction strength, and ⊙ denotes element-wise multiplication. The diference between the reference latent variable and the initial generation defines the correction direction. The spatial gate controls the correction magnitude at each location, incorporating transported historical appearance information into the generation.

## 3.5. Readback under the Same Quotient Measurement and Spatial Correction of Node Errors

After motion-guided reference transport, QCC uses the same reference quotient measurement to evaluate the discrepancy between the current generation and the observable macro-level target. It first decodes the motion-corrected latent variable and reads back its macro-level content:

$$
C _ { k } ^ { b , \mathrm { M } } = \mathcal { M } _ { 0 } ^ { b } \big ( D ( x _ { k } ^ { \mathrm { M } } ) \big ) , \qquad e _ { k } ^ { b } = \widehat { C } _ { k } ^ { b } - C _ { k } ^ { b , \mathrm { M } } .\tag{19}
$$

Because target construction and generation readback share $\Pi _ { 0 } ^ { b } , \Gamma _ { 0 } ^ { b } , e _ { k } ^ { b }$ has fixed node definitions and spatial origins. It represents changes that the current generation has not yet captured under the macro-level measurement. Readback follows motionguided reference correction, allowing error correction to operate on generation results that have already been adjusted using reference motion.

Spatial feedback expands errors through the node–location relationships defined by QCC. One path directly projects node errors, while the other modulates error representations using predicted content and content read back from the generation:

$$
\begin{array} { r l } & { E _ { k } ^ { b , \mathrm { b a s e } } ( p ) = \displaystyle \sum _ { j } \Pi _ { 0 , p j } ^ { b } W _ { e } ^ { b } e _ { k , j } ^ { b } , } \\ & { r _ { k , j } ^ { b } = W _ { r } ^ { b } e _ { k , j } ^ { b } \odot \left[ 1 + \xi ^ { b } \big ( [ \widehat { C } _ { k , j } ^ { b } ; C _ { k , j } ^ { b , \mathrm { M } } ] \big ) \right] , } \\ & { E _ { k } ^ { b , \mathrm { c t x } } ( p ) = \displaystyle \sum _ { j } \Pi _ { 0 , p j } ^ { b } r _ { k , j } ^ { b } , } \end{array}\tag{20}
$$

where $W _ { e } ^ { b } , W _ { r } ^ { b }$ are channel projections and $\xi ^ { b }$ is a node content modulation function. The error fields are interpolated to the generation latent grid and combined with local context from the current and reference latent variables to produce spatial corrections:

$$
\begin{array} { r l } & { \delta x _ { k } ^ { \mathrm { E } } = \mathcal { D } _ { \mathrm { b a s e } } \left( \{ E _ { k } ^ { b , \mathrm { b a s e } } \} _ { b } ; x _ { k } ^ { \mathrm { M } } \right) + \mathcal { D } _ { \mathrm { c t x } } \left( \{ E _ { k } ^ { b , \mathrm { c t x } } \} _ { b } ; x _ { k } ^ { \mathrm { M } } , x _ { 0 } \right) , } \\ & { ~ \widehat { x } _ { k } = x _ { k } ^ { \mathrm { M } } + \alpha _ { \mathrm { E } } \delta x _ { k } ^ { \mathrm { E } } , \qquad \widehat { I _ { k } } = D ( \widehat { x } _ { k } ) , } \end{array}\tag{21}
$$

where $\mathcal { D } _ { \mathrm { b a s e } }$ and $\mathcal { D } _ { \mathrm { c t x } }$ include branch fusion, spatial interpolation, and local context modulation, and are collectively denoted by ${ \mathcal { R } } _ { \Pi _ { 0 } }$ . This interface converts macro-level errors into generation adjustments with explicit spatial correspondence.

## 3.6. Learning Objectives and Staged Optimization

The model is trained for quotient construction, macro-level content prediction, conditional generation, and spatial correction. The main objective is summarized as

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { m a i n } } = \lambda _ { \mathrm { g r o u p } } \mathcal { L } _ { \mathrm { g r o u p } } + \lambda _ { \mathrm { Q } } \mathcal { L } _ { \mathrm { Q } } } & { } \\ { + \lambda _ { \mathrm { g e n } } \mathcal { L } _ { \mathrm { g e n } } + \lambda _ { \mathrm { r e t } } \mathcal { L } _ { \mathrm { r e t } } , } \end{array}\tag{22}
$$

where $\mathcal { L } _ { \mathrm { g r o u p } }$ encourages quotient construction to preserve observed response information, $\mathcal { L } _ { \mathrm { Q } }$ supervises macro-level content prediction under a fixed reference measurement, $\mathcal { L } _ { \mathrm { g e n } }$ trains conditional difusion generation, and $\mathcal { L } _ { \mathrm { r e t } }$ trains spatial correction. Training also includes auxiliary supervision on local responses, node and local motion, shared states, and transport alignment, as detailed below. These objectives are applied to the corresponding modules in separate training stages. The spatial correction module is trained separately after the core model is frozen.

Local response prediction is supervised using cumulative optical flow and changes relative to the reference latent variable from training observations. Let $r _ { k } ^ { \star }$ denote the observed response normalized using training statistics. We supervise Eq. (2) with both response prediction error and local response error after removing the spatial mean. Response reconstruction after node aggregation further encourages quotient construction to preserve information relevant to future evolution:

$$
\bar { r } _ { k , j } ^ { b , \star } = \sum _ { p } \omega _ { 0 , p j } ^ { b } r _ { k , p } ^ { \star } , \mathcal { L } _ { \mathrm { g r o u p } } = \Big \langle \left. \Pi _ { 0 } ^ { b } \bar { r } ^ { b , \star } - r ^ { \star } \right. _ { \Omega } ^ { 2 } \Big \rangle _ { b } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } ,\tag{23}
$$

where Ω denotes the valid support, and $\mathcal { L } _ { \mathrm { r e g } }$ regularizes node occupancy and near-duplicate normalized assignment columns. These objectives connect node organization with observed responses during training, while construction itself uses only history and actions.

Content learning in the macro-level dynamics uses the quotient consistency objective in $\operatorname { E q } .$ . (12). Node motion targets are obtained by aggregating observed optical flow $f _ { k }$ under the same reference assignment:

$$
u _ { k , j } ^ { b , \star } = \sum _ { p } \omega _ { 0 , p j } ^ { b } f _ { k , p } , \qquad \widehat { f } _ { k , p } ^ { b } = \sum _ { j } \Pi _ { 0 , p j } ^ { b } \widehat { u } _ { k , j } ^ { b } ,\tag{24}
$$

where predictions and targets use the same motion scale. The corresponding losses compare node motion $\widehat { \boldsymbol { u } } ^ { b }$ with $u ^ { b , \star }$ and reconstructed local fields ${ \widehat { f } } ^ { b }$ with $f ,$ while also supervising shared content increments and shared motion. Construction and response adaptation additionally use local motion errors and alignment errors between transported references and true future latent variables to coordinate response control, macrolevel motion, and reference transport.

Initial observations are generated in a pretrained visual latent space. Let $x _ { k } = E _ { \mathrm { v a e } } ( I _ { k } )$ . We construct difusion targets using the reference residual $r _ { k } ^ { x } = ( x _ { k } - x _ { 0 } ) / \sigma _ { x } \mathrm { : }$

$$
x _ { s , k } = \sqrt { \bar { \alpha } _ { s } } r _ { k } ^ { x } + \sqrt { 1 - \bar { \alpha } _ { s } } \epsilon _ { k } , \qquad \nu _ { s , k } ^ { \star } = \sqrt { \bar { \alpha } _ { s } } \epsilon _ { k } - \sqrt { 1 - \bar { \alpha } _ { s } } r _ { k } ^ { x } .\tag{25}
$$

Conditioned on history, complete actions, and C<sub>QCC</sub>, the generator minimizes a denoising objective weighted by the valid region:

$$
\mathcal { L } _ { \mathrm { g e n } } = \frac { \mathbb { E } \left[ w ( p ) ( \widehat { \nu } _ { \psi , s , k } ( p , c ) - \nu _ { s , k } ^ { \star } ( p , c ) ) ^ { 2 } \right] } { \mathbb { E } [ w ( p ) ] } ,\tag{26}
$$

where $w ( p )$ is the valid observation region weight, and the expectation averages over samples, difusion steps, future steps, positions, and channels c.

The spatial correction module is trained using initial samples generated by the frozen core model. For its training output $I _ { k } ^ { \mathrm { r e t } }$ , we combine image reconstruction, perceptual distance, structural similarity, and generation–prediction and generation– observation alignment under the same quotient measurement:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e t } } = \lambda _ { \mathrm { p i x } } \mathcal { L } _ { \mathrm { p i x } } ( I ^ { \mathrm { r e t } } , I ) + \lambda _ { \mathrm { p e r } } \mathcal { L } _ { \mathrm { p e r } } ( I ^ { \mathrm { r e t } } , I ) + \lambda _ { \mathrm { s t r } } \mathcal { L } _ { \mathrm { s t r } } ( I ^ { \mathrm { r e t } } , I ) } \\ & { \quad \quad \quad \quad + \lambda _ { \mathrm { P } } \Big \langle \mathrm { M S E } \Big ( \mathcal { M } _ { 0 } ^ { b } ( I ^ { \mathrm { r e t } } ) , \widehat { C } ^ { b } \Big ) \Big \rangle _ { b } } \\ & { \quad \quad \quad \quad \quad + \lambda _ { 0 } \Big \langle \mathrm { M S E } \Big ( \mathcal { M } _ { 0 } ^ { b } ( I ^ { \mathrm { r e t } } ) , \mathcal { M } _ { 0 } ^ { b } ( I ) \Big ) \Big \rangle _ { b } , } \end{array}\tag{27}
$$

where $\mathcal { L } _ { \mathrm { p i x } }$ is image MSE over the valid region, $\mathcal { L } _ { \mathrm { p e r } }$ is perceptual distance, and $\mathcal { L } _ { \mathrm { s t r } }$ is structural similarity loss. The two node-level alignment terms encourage generated observations to match the observable macro-level targets and the true future observations under a shared measurement, respectively.

## 4. Results

## 4.1. Ablation Study

To examine the efect of QCC branch configurations on prediction performance, we compare four settings: a direct actionconditioned baseline (Direct), a single-branch QCC model with only the translation-guided branch (T-only), a single-branch QCC model with only the rotation-guided branch (R-only), and the complete QCC-WM with both branches. The T-only and R-only models are trained independently, with six macro-nodes per branch. The complete system combines dual-branch macrostate prediction, motion-guided reference transport, and spatial correction of node-level errors. All configurations use the same evaluation split and sampling settings. Results are reported as the mean and sample standard deviation across three training seeds.

As shown in Table 1, both single-branch QCC variants outperform Direct in the mean values of all five metrics, indicating that action-guided macro-level modeling improves future observation prediction. The two single-branch configurations perform similarly overall: T-only achieves slightly lower ROI-MSE and temporal MSE, while R-only performs slightly better in SSIM and LPIPS. Both achieve a PSNR of approximately 24 98 dB.

Table 1: Prediction performance of diferent QCC configurations. Dual QCC is the complete system with macro-state prediction, motion-guided reference transport, and spatial correction of node-level errors.
<table><tr><td>Setting</td><td>ROI-MSE↓  $( \times 1 0 ^ { - 3 } )$ </td><td>PSNR ↑ (dB)</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Temporal MSE ↓  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>Direct</td><td> $2 0 . 4 6 \pm 0 . 0 9$ </td><td> $2 4 . 9 1 \pm 0 . 0 1$ </td><td>0.7440 ± 0.0003</td><td> $0 . 0 7 9 0 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $6 . 1 7 \pm 0 . 0 4$ </td></tr><tr><td>Single QCC: T-only</td><td> $1 9 . 9 0 _ { \pm 0 . 0 9 }$ </td><td> $2 4 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 7 4 5 0 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $0 . 0 7 8 2 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $6 . 0 5 \pm 0 . 0 4$ </td></tr><tr><td>Single QCC: R-only</td><td> $1 9 . 9 8 \pm 0 . 1 0$ </td><td> $2 4 . 9 8 \pm 0 . 0 4$ </td><td> $0 . 7 4 5 1 \pm 0 . 0 0 1 1$ </td><td> $0 . 0 7 7 9 _ { \pm 0 . 0 0 0 7 }$ </td><td> $6 . 0 6 \pm 0 . 0 5$ </td></tr><tr><td>Dual QCC: T and R</td><td> $\mathbf { 1 8 . 6 1 } _ { \pm 0 . 0 8 }$ </td><td> $2 5 . 1 7 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 7 4 9 2 _ { \pm 0 . 0 0 0 1 } }$ </td><td>0.0731 ± 0.0002</td><td> ${ \bf 5 . 7 6 _ { \pm 0 . 0 2 } }$ </td></tr></table>

Note: Results are reported as the mean ± sample standard deviation across three training seeds. Both MSE metrics are reported in units of $1 0 ^ { - 3 }$ . Bold indicates the best mean in each column.

The complete dual-branch QCC system achieves the best mean values across all five metrics. Compared with Direct, it reduces ROI-MSE, LPIPS, and temporal MSE by 9 05%, 7 51%, and 6 64%, respectively. It also improves all metrics over both single-branch variants, indicating that combining dual-branch macro-state prediction with subsequent spatial correction improves pixel reconstruction, structural similarity, perceptual quality, and prediction of inter-frame changes.

## 4.2. Mechanism Analysis of QCC-WM

We analyze QCC from three perspectives: spatial node organization, motion prediction, and generation correction.

## 4.2.1. Spatial Organization ofMacro-Nodes

QCC aggregates local visual features through action-guided soft quotient mappings while preserving the spatial support of each node in the reference observation. The mapping remains fixed within each prediction window, ensuring that node-state comparison and subsequent spatial feedback use the same spatial correspondence. Figure 2 shows that nodes in both branches cover the valid imaging region and form distinct local aggregation regions, providing explicit spatial origins for macro-states.

We define the efective number of nodes using normalized node masses $p _ { j }$ as $\begin{array} { r } { K _ { \mathrm { e f f } } = \exp ( - \sum _ { j } p _ { j } } \end{array}$ log $p _ { j } )$ . With six nodes per branch, the efective node counts of the T and R branches are $5 . 4 2 \pm 0 . 1 7$ and $5 . 5 0 \pm 0 . 1 9$ , respectively. Their mean minimum node masses are $( 7 . 5 3 \pm 0 . 7 3 ) \%$ and $( 8 . 3 4 \pm 1 . 4 6 ) \%$ and the cosine similarities between assignment columns are $0 . 0 6 8 9 \pm 0 . 0 0 2 5$ and $0 . 0 6 9 6 \pm 0 . 0 0 6 7$ . These results indicate that node capacity is well utilized and that spatial weights have limited overlap across nodes. This spatial organization allows macro-level motion and node-level errors to be mapped back to their corresponding local regions.

## 4.2.2. Macro-Level Motion and Local Displacement

The action-conditioned dynamics predict the motion response of each node and recover local displacement fields through the reference quotient mapping. Motion diferences between macro-nodes can therefore produce diferent displacements across image regions, driving reference information transport. Figure 3(a,b) compares predicted displacements with zero-motion, local constant-velocity, and global constantvelocity extrapolation. Using observed optical flow as the reference, we compute the MSE of two-dimensional cumulative displacement over the valid region and the corresponding error after removing its spatial mean.

QCC achieves a four-step average cumulative displacement MSE of $0 . 5 9 8 0 { \scriptstyle \pm 0 . 0 0 9 1 }$ , lower than the three controls at 0 7803, 0 9451, and 0 7071. After removing the spatial mean of the cumulative error, QCC achieves an MSE of $0 . 3 3 2 1 \pm 0 . 0 0 5 3$ which remains below 0 3527 for both zero-motion and global constant-velocity extrapolation, and 0 5906 for local constantvelocity extrapolation. These results indicate that macrolevel dynamics reduce cumulative displacement errors and further improve the prediction of local motion diferences. Figure 3(c,d) illustrates diferences in the direction and magnitude of node responses, showing how the fixed quotient mapping translates macro-level predictions into spatially varying motion fields.

## 4.2.3. Complementary Efects of Motion Transport and Error Correction

Motion-guided reference transport moves historical appearance information according to predicted displacements and corrects the initial generation through spatial gating. Readback under the same quotient measurement then measures the remaining node-level discrepancies and maps them back to local latent space. Figure 4 examines the two paths using four settings that independently enable or disable each correction. Node-level errors are recomputed from the generated results under each setting.

Enabling motion-guided reference correction alone reduces ROI-MSE from $0 . 0 1 9 8 4 9 \pm 0 . 0 0 0 0 4 2 \mathrm { { t o } } 0 . 0 1 8 6 2 7 \pm 0 . 0 0 0 0 9 6 ,$ indicating that transported historical appearance improves pixel reconstruction. However, Fig. 4(b,c) shows increases in both LPIPS and generation–prediction node error. Readback after motion transport is therefore needed to correct the remaining discrepancies from the macro-level targets. Adding node-level error correction reduces LPIPS from $0 . 0 7 9 0 5 1 \pm 0 . 0 0 0 3 7 5$ to $0 . 0 7 3 0 5 9 \pm 0 . 0 0 0 2 4 3$ , a relative reduction of 7 58% (paired 95% interval: [6 39% 8 93%]). The two paths thus have complementary efects: motion transport provides spatially relevant appearance references, while node-level error correction mainly improves perceptual quality and macro-level alignment. Compared with disabling both corrections, the complete interface reduces ROI-MSE, LPIPS, and temporal MSE by 6 25%, 6 45%, and 4 55%, respectively.

![](images/6393065679e40f47d5984514d297b3d9f5a0b8a518916cf1f778ffad0b370030.jpg)  
Figure 2: Spatial origins and efective occupancy of QCC macro-nodes. Left: reference observations and soft quotient assignments of the T and R branches for three samples. Colors indicate node indices within each branch. Right: efective number of occupied nodes, minimum node mass, and cosine similarity between assignment columns. Diamonds and error bars indicate the mean and sample standard deviation across three training seeds; circles show individual seed results. Dashed lines mark the six predefined nodes and the 1% low-occupancy threshold, respectively.

## 4.2.4. Efects of Readback and Spatial Correspondence

Readback uses the fixed reference measurement $M _ { 0 }$ to compare the observable macro-level target $\widehat { C } _ { k }$ produced by motion transport with the current generation, giving node-level errors a common measurement definition and spatial origin. To evaluate correction, we compute node-level errors between generated observations and both macro-level targets and true future observations:

$$
E _ { \mathrm { G - P } } ( k ) = \mathrm { M S E } \big ( M _ { 0 } ( \widehat { U } _ { k } ) , \widehat { C } _ { k } \big ) ,\tag{28}
$$

$$
E _ { \mathrm { G - O } } ( k ) = \mathrm { M S E } \Big ( M _ { 0 } ( \widehat { U } _ { k } ) , M _ { 0 } ( U _ { k } ) \Big ) ,\tag{29}
$$

where $\widehat { U } _ { k }$ and $U _ { k }$ denote generated and true future observations, respectively, and the metrics are averaged over the T and R branches.

Adding node-level error feedback reduces generation– prediction node MSE from 0 004187 to 0 001604 and generation–observation node MSE from 0 010688 to 0 006570, corresponding to reductions of 61 69% and 38 53% (paired 95% intervals: [53 36% 67 91%] and [30 34% 46 14%], respectively). Figure 5(d,e) shows that both errors decrease at all four prediction steps. Readback correction therefore improves both agreement with macro-level predictions and macro-level correspondence with true future observations.

To examine the efect of error feedback locations, we replace the final feedback mapping with a spatially uniform assignment or shufle its spatial positions, while keeping all other predictions, readback measurements, and correction conditions unchanged. Global feedback preserves the valid support and total node masses. Results for spatial shufling are averaged over five predefined shufles. Compared with these two controls, complete spatial feedback reduces generation–prediction node MSE by 8 28% and 7 79%, and generation–observation node MSE by 4 39% and 4 07%, respectively. ROI-MSE values for global, shufled, and complete feedback are 0 018530, 0 018563, and 0 018609, respectively. Complete feedback has slightly higher pixel error, with its main advantage lying in macro-level alignment. These results indicate that preserving node–location correspondence helps translate macro-level errors into generation adjustments at the appropriate regions.

Compared with using the initial content measurement as the error decoder target, using the future observable target reduces generation–prediction node error by 1 20%. ROI-MSE changes by only 0 02%, with its paired interval including zero. Together, these results suggest that the current correction gains mainly arise from combining motion-guided reference transport with node-level error feedback, while correct spatial correspondence further improves consistency between macro-level predictions and generated observations.

## 4.3. Comparative Experiments

We evaluate action-conditioned ultrasound video prediction through short-window prediction and multi-step extrapolation, with prediction horizons $H \in \{ 4 , 8 , 1 6 \}$ . All methods use model weights trained at H = 4 and recursively generate future observations in blocks of four frames. The H = 8 and H = 16 settings correspond to extrapolation over two and four prediction blocks, respectively. Each model starts from four real historical frames. Each prediction block receives the corresponding four-step sequence of six-dimensional actions, and generated observations are used as history for the next block. The H = 8 results are taken from the first eight steps of the same sixteenstep trajectory to preserve the same noise and prediction prefix.

![](images/44a31360d01a1d155995ceed6f5f019baa2372c71a2125b5e8e018e6e7370075.jpg)

![](images/673f32ddf6c17beffb5fa511b038737fb5fd5736fd663c0f3397e73142b3dbb0.jpg)

![](images/5a6db8949afd646485740e9d47241e345ad12966d1612d1eabec384f97286752.jpg)  
Figure 3: Macro-level motion prediction. (a) Cumulative displacement MSE. (b) MSE after removing the spatial mean of the cumulative error. Optical flow estimated from observations serves as the reference. Curves and shaded regions show the mean and sample standard deviation across three random seeds. (c,d) Cumulative responses of T and R nodes in Case 2: open circles indicate aggregated observed optical flow, and crosses indicate predictions. Successive markers correspond to consecutive prediction steps. Displacements are enlarged sixfold for visualization.

The compared methods include PredRNN-Act [22, 23], PredRNN-V2-Act [23], SimVP [24], IRASim-S/4 [6], DINO-WM [5], MAU [25], and MMVP [26]. The PredRNN models use the action-conditioned extensions released by their authors. SimVP, MAU, and MMVP use implementations from Open-STL [27]. Each baseline retains its original backbone, with separate interfaces adapting action inputs and observation representations. The PredRNN models, SimVP, IRASim, MAU, and MMVP share the same frozen domain-specific VAE. DINO-WM uses its native DINOv2 feature dynamics and oficial RGB decoder. During recursive prediction, QCC feeds back the corrected latent variables and recomputes USFM features, JEPA features, and historical optical flow from generated frames to construct quotient states for the next prediction block. DINO-WM feeds back predicted patch features, while the remaining methods feed back predicted latent variables. All baselines are trained for 30 epochs and evaluated using the raw final-epoch weights.

Based on recorded timestamps, the median temporal spans for H = 4 8 16 are approximately 0 133 s, 0 267 s, and 0 533 s, respectively.

## 4.3.1. Four-Step Prediction

Table 2 reports four-step prediction results. QCC achieves the best mean values across all five metrics. Compared with SimVP, the baseline with the lowest ROI-MSE, QCC reduces ROI-MSE by 2.70% and increases PSNR by 0 862 dB. Compared with MMVP, the strongest baseline in perceptual quality and temporal prediction, QCC reduces LPIPS and temporal MSE by 47.84% and 13.06%, respectively. These results indicate improvements in pixel reconstruction, perceptual quality, and prediction of inter-frame changes within the short prediction window, with the largest relative reduction observed in perceptual distance.

## 4.3.2. Eight-Step Extrapolation

Table 3 reports results after recursively generating two prediction blocks. QCC maintains the best mean values in ROI-MSE, PSNR, LPIPS, and temporal MSE. Compared with the best baseline for each metric, QCC reduces ROI-MSE by 2.05% relative to SimVP, and reduces LPIPS and temporal MSE by 41.68% and 5.65%, respectively, relative to PredRNN-V2. QCC achieves an SSIM of 0.64475, close to MMVP at 0.64526. Reconstruction errors and perceptual distances increase for all methods compared with four-step prediction. QCC nevertheless retains a substantial LPIPS reduction beyond the training window, indicating a perceptual quality advantage in multi-step extrapolation.

## 4.3.3. Sixteen-Step Extrapolation

Table 4 reports results after recursively generating four prediction blocks. QCC retains the best mean values in ROI-MSE,

![](images/66d5da8eae4ff0fe2ca4ecd8f90ce61ec461581d05c52ac38e058ad5f90f2935.jpg)

![](images/89e4ee481f0d7cd6db148f04798c88239267a1f856ab2ed8b9e4d577108ceabe.jpg)

![](images/5d8ea2d1b303bd54e888febc809e99ac826a39684d30ae5740bd5820f912304f.jpg)

![](images/261368b45550d68b4803b528c058fec55ba2ca3398f63f645867dd4c757fd98a.jpg)  
Case 2, fixed reference-based crop, prediction step 4.  
Matched frozen-checkpoint interventions; mean ± s.d. across 3 seeds.

Figure 4: Complementary efects of motion transport and node-level error correction. (a–c) ROI-MSE, LPIPS, and generation–prediction node MSE under four interface settings. The horizontal axis indicates whether motion-guided reference correction is enabled. Gray and teal indicate disabled and enabled node-level error correction, respectively. Error bars show the sample standard deviation across three random seeds. Below: local crops at prediction step 4 in Case 2, showing the reference observation, transported reference, true future observation, initial generation, and generation with motion correction only.  
Table 2: Cumulative prediction results at $H = 4 .$
<table><tr><td>Method</td><td>ROI-MSE↓  $( \times 1 0 ^ { - 3 } )$ </td><td>PSNR ↑ (dB)</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Temporal MSE ↓  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>PredRNN-Act [22]</td><td> $2 4 . 5 1 \pm 0 . 7 6$ </td><td> $2 3 . 2 2 \pm 0 . 1 1$ </td><td> $0 . 7 0 5 3 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $0 . 1 5 3 0 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $6 . 9 2 \pm 0 . 0 9$ </td></tr><tr><td>PredRNN-V2-Act [23]</td><td> $2 3 . 9 9 \pm 0 . 8 1$ </td><td> $2 3 . 3 0 \pm 0 . 1 2$ </td><td> $0 . 7 0 2 5 { \scriptstyle \pm 0 . 0 0 1 7 }$ </td><td> $0 . 1 5 1 5 _ { \pm 0 . 0 0 0 8 }$ </td><td> $6 . 8 5 \pm 0 . 0 9$ </td></tr><tr><td>SimVP+Act [24]</td><td> $1 9 . 7 0 \pm 0 . 2 4$ </td><td> $2 4 . 1 4 \pm 0 . 0 7$ </td><td> $0 . 7 2 5 1 \scriptstyle \pm 0 . 0 0 2 2$ </td><td> $0 . 2 1 1 0 { \scriptstyle \pm 0 . 0 0 2 0 }$ </td><td> $7 . 3 5 \pm 0 . 0 8$ </td></tr><tr><td>IRASim-S/4 [6]</td><td> $4 8 . 6 7 \pm 5 . 1 9$ </td><td> $2 0 . 2 8 \pm 0 . 3 8$ </td><td> $0 . 6 1 5 2 \substack { \pm 0 . 0 4 2 9 }$ </td><td> $0 . 1 6 7 7 { \scriptstyle \pm 0 . 0 3 3 8 }$ </td><td> $2 8 . 4 3 \pm 8 . 2 1$ </td></tr><tr><td>DINO-WM+Dec. [5]</td><td> $3 7 . 8 4 \pm 2 . 3 7$ </td><td> $2 0 . 5 3 \pm 0 . 2 8$ </td><td> $0 . 5 9 7 7 { \scriptstyle \pm 0 . 0 0 2 5 }$ </td><td> $0 . 2 8 7 9 _ { \pm 0 . 0 4 1 7 }$ </td><td> $1 2 . 0 8 \pm 0 . 6 1$ </td></tr><tr><td>MAU+Act [25]</td><td> $2 3 . 5 4 \pm 0 . 2 0$ </td><td> $2 3 . 1 1 \pm 0 . 0 9$ </td><td> $0 . 6 9 6 4 \substack { \pm 0 . 0 0 6 1 }$ </td><td> $0 . 2 0 9 5 _ { \pm 0 . 0 0 3 7 }$ </td><td> $7 . 2 9 \pm 0 . 1 6$ </td></tr><tr><td>MMVP+Act [26]</td><td> $2 2 . 9 8 \pm 0 . 2 5$ </td><td> $2 3 . 9 3 \pm 0 . 0 4$ </td><td> $0 . 7 2 9 4 \substack { \pm 0 . 0 0 2 4 }$ </td><td> $0 . 1 4 2 1 \pm 0 . 0 0 7 0$ </td><td> $6 . 7 3 \pm 0 . 0 4$ </td></tr><tr><td>QCC-WM</td><td> $\mathbf { 1 8 . 6 1 } _ { \pm 0 . 0 8 }$ </td><td> $2 5 . 1 7 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 7 4 9 2 _ { \pm 0 . 0 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 7 3 1 } _ { \pm 0 . 0 0 0 2 }$ </td><td> ${ \bf 5 . 7 6 _ { \pm 0 . 0 2 } }$ </td></tr></table>

Note: All methods use the same 226 test starting points. Results are the mean ± sample standard deviation across three training seeds. Both MSE metrics are reported in units of 10<sup>−3</sup>; bold indicates the best mean. Act denotes an action-conditioning interface, and Dec. denotes the oficial RGB decoder.

Table 3: Cumulative prediction results at $H = 8 .$
<table><tr><td>Method</td><td>ROI-MSE↓  $( \times 1 0 ^ { - 3 } )$ </td><td>PSNR ↑ (dB)</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Temporal MSE ↓  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>PredRNN-Act [22]</td><td> $4 0 . 3 3 \pm 1 . 4 4$ </td><td> $2 1 . 0 4 \pm 0 . 1 5$ </td><td> $0 . 6 2 6 4 \substack { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 1 9 5 8 \scriptstyle \pm 0 . 0 0 2 2$ </td><td> $7 . 5 1 \pm 0 . 0 6$ </td></tr><tr><td>PredRNN-V2-Act [23]</td><td> $3 9 . 9 6 \pm 1 . 7 3 $ </td><td> $2 1 . 0 9 _ { \pm 0 . 1 6 }$ </td><td> $0 . 6 2 2 6 { \scriptstyle \pm 0 . 0 0 3 3 }$ </td><td> $0 . 1 9 1 3 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $7 . 4 3 \pm 0 . 0 6$ </td></tr><tr><td>SimVP+Act [24]</td><td> $3 5 . 8 6 \pm 0 . 5 8$ </td><td> $2 1 . 6 8 \pm 0 . 0 8$ </td><td> $0 . 6 4 1 4 \substack { \pm 0 . 0 0 1 8 }$ </td><td> $0 . 2 5 4 2 \scriptstyle \pm 0 . 0 0 1 9$ </td><td> $8 . 2 7 \pm 0 . 0 9$ </td></tr><tr><td>IRASim-S/4 [6]</td><td> $8 5 . 8 9 \pm 1 7 . 2 0$ </td><td> $1 7 . 9 7 \pm 0 . 6 3$ </td><td> $0 . 5 1 5 1 \pm 0 . 0 3 5 4$ </td><td> $0 . 2 1 4 7 \pm 0 . 0 2 8 8$ </td><td> $2 8 . 4 5 \pm 7 . 4 1$ </td></tr><tr><td>DINO-WM+Dec. [5]</td><td> $5 0 . 6 5 \pm 3 . 5 7$ </td><td> $1 9 . 4 0 \pm 0 . 3 0$ </td><td> $0 . 5 6 0 1 \substack { \pm 0 . 0 0 4 1 }$ </td><td> $0 . 2 9 7 8 \pm 0 . 0 4 0 3$ </td><td> $1 0 . 1 7 \pm 0 . 3 6$ </td></tr><tr><td>MAU+Act [25]</td><td> $4 0 . 1 1 \pm 0 . 8 1$ </td><td> $2 0 . 9 4 \substack { \pm 0 . 0 7 }$ </td><td> $0 . 6 1 9 9 _ { \pm 0 . 0 0 6 0 }$ </td><td> $0 . 2 3 6 3 \substack { \pm 0 . 0 0 4 8 }$ </td><td> $7 . 8 5 \pm 0 . 1 7$ </td></tr><tr><td>MMVP+Act [26]</td><td> $3 8 . 1 0 \pm 0 . 3 2 $ </td><td> $2 1 . 5 7 \pm 0 . 0 8$ </td><td> $\mathbf { 0 . 6 4 5 3 _ { \pm 0 . 0 0 3 2 } }$ </td><td> $0 . 2 0 5 8 \substack { \pm 0 . 0 0 8 7 }$ </td><td> $7 . 5 4 \pm 0 . 0 6$ </td></tr><tr><td>QCC-WM</td><td> $3 5 . 1 2 \scriptstyle \pm 0 . 3 0$ </td><td> $2 2 . 2 6 \pm 0 . 0 5$ </td><td> $0 . 6 4 4 8 \substack { \pm 0 . 0 0 0 7 }$ </td><td> $\mathbf { 0 . 1 1 1 6 } _ { \pm 0 . 0 0 0 3 }$ </td><td> $7 . 0 2 \pm 0 . 0 2$ </td></tr></table>

Note: All methods use the same 226 test starting points. Results are the mean ± sample standard deviation across three training seeds. Both MSE metrics are reported in units of $1 0 ^ { - 3 } ;$ bold indicates the best mean. Act denotes an action-conditioning interface, and Dec. denotes the oficial RGB decoder.

![](images/73f6d02f7687ad4913d80a7582e4ad524d780f401c541594479b39a30e64b4ad.jpg)

b  
![](images/bbc649ffeb06dd85461585a09d75222d557340ef138f90db998a92dff5511132.jpg)

c  
![](images/9a1d2058ba8dcbafa8399af20f8fc96f37eea17cfbc82be442fa1e64274467bc.jpg)

![](images/d5cd845d66b13645bae12fe55417817320be599b3914dbd880ea5b5cf6416975.jpg)

![](images/7d4ce15db750d7546b683bd4b993ffe85cfd82ead40fa7eac8c98797f39c7385.jpg)  
Figure 5: Analysis of readback under the same quotient measurement and spatial feedback. (a–c) Reductions in image error, generation–prediction node error, and generation–observation node error for the complete model relative to four controls: no node-level error correction, global feedback, spatially shufled feedback, and initial content targets. Positive values favor the complete model; horizontal intervals indicate paired 95% intervals. (d,e) Node MSE at each prediction step after motion correction and after adding node-level error correction. Curves and shaded regions show the mean and sample standard deviation across three random seeds. All settings use the same reference quotient measurement.

PSNR, LPIPS, and temporal MSE. Its LPIPS is 0.16209, a 30.23% reduction relative to the best baseline, PredRNN-V2. QCC achieves an ROI-MSE of 0.056069, which is 0.77% lower than MMVP at 0.056504, and a temporal MSE of 0.007315, which is 1.55% lower than PredRNN-V2. As the extrapolation horizon increases, QCC’s relative advantages in pixel and temporal errors decrease, while its advantage in perceptual distance remains substantial. For SSIM, MMVP achieves the best result at 0.56690. QCC achieves 0.55178, below MMVP, SimVP, and PredRNN-Act, indicating that these methods provide higher structural similarity at this horizon.

Across the three prediction horizons, QCC achieves the best mean values for all five metrics at four steps and maintains its lead in ROI-MSE, PSNR, LPIPS, and temporal MSE at eight and sixteen steps. Its LPIPS reductions relative to the best baseline at each horizon are 47.84%, 41.68%, and 30.23%, respectively, representing its most prominent performance advantage across prediction horizons.

As the prediction horizon increases from 4 to 16 steps, QCC’s cumulative ROI-MSE rises from 0.01861 to 0.05606, and LPIPS increases from 0.0731 to 0.1620, approximately 2.92 and 2.19 times their four-step values, respectively. PSNR decreases by 5 145 dB, and SSIM decreases by 0.19010. This trend reflects error accumulation during recursive prediction. Overall, QCC maintains favorable perceptual quality during multi-step extrapolation, while structural similarity at longer horizons remains an area for improvement.

## 4.4. Experiments on an Additional Dataset

To evaluate the transfer performance of QCC-WM on additional ultrasound data, we conduct supplementary experiments on the public Freehand US dataset [28, 29]. The dataset contains ultrasound scans of both forearms from 19 volunteers, with 12 scanning sequences per volunteer and 228 sequences in total. Acquisition covers straight, C-shaped, and S-shaped trajectories, with the probe oriented parallel or perpendicular to the forearm. The B-mode images have an original resolution of 480 × 640 and a sampling rate of approximately 20 Hz. The released data also include optically tracked poses and calibration information. We represent calibrated relative SE(3) transformations between adjacent frames using three-dimensional translation and three-dimensional axis-angle rotation, forming sixdimensional action conditions for predicting subsequent ultrasound images.

The training, validation, and test sets are split by subject and contain 13, 3, and 3 subjects, respectively. The training set provides 1437 four-step prediction windows. The validation set contains 207 windows with at least 16 consecutive future steps, and the test set contains 278 common prediction starting points. Images are resized to 224 × 224 and normalized to [−1 1]. We retain the original temporal sampling and align images with poses using the oficial calibration information. All methods receive four historical frames and the corresponding action sequences and are trained with a supervision horizon of H = 4. During evaluation, the same H = 4 model weights are reused to recursively generate predictions at H = 8 and H = 16 in four-frame blocks, with generated observations fed back between blocks. All three horizons are evaluated at the same 278 test starting points. Their median temporal spans are approximately 0 196, 0 391, and 0 798 s, respectively.

Table 4: Cumulative prediction results at $H = 1 6 .$
<table><tr><td>Method</td><td> $\mathrm { R O I - M S E \downarrow }$   $( \times 1 0 ^ { - 3 } )$ </td><td>PSNR ↑ (dB)</td><td></td><td>LPIPS ↓</td><td>Temporal MSE ↓  $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td></td><td></td><td></td><td>SSIM ↑</td><td></td><td></td></tr><tr><td>PredRNN-Act [22]</td><td> $5 8 . 0 5 \pm 2 . 2 8 $ </td><td> $1 9 . 2 3 \pm 0 . 1 8$ </td><td> $0 . 5 5 5 3 { \scriptstyle \pm 0 . 0 0 2 0 }$ </td><td> $0 . 2 4 0 7 \substack { \pm 0 . 0 0 3 7 }$ </td><td> $7 . 5 1 \pm 0 . 0 4$ </td></tr><tr><td>PredRNN-V2-Act [23]</td><td> $5 8 . 1 7 \pm 2 . 1 6$ </td><td> $1 9 . 2 5 \substack { \pm 0 . 1 6 }$ </td><td> $0 . 5 5 0 6 { \scriptstyle \pm 0 . 0 0 2 9 }$ </td><td> $0 . 2 3 2 3 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td><td> $7 . 4 3 \pm 0 . 0 4$   $8 . 3 4 \pm 0 . 0 4$ </td></tr><tr><td>SimVP+Act [24] IRASim-S/4 [6]</td><td> $5 6 . 7 9 \pm 1 . 3 1$ </td><td> $1 9 . 5 0 \pm 0 . 1 0$ </td><td> $0 . 5 6 3 3 { \scriptstyle \pm 0 . 0 0 2 4 }$   $0 . 4 2 0 0 { \scriptstyle \pm 0 . 0 2 8 5 }$ </td><td> $0 . 2 9 3 0 { \scriptstyle \pm 0 . 0 0 2 5 }$ </td><td> $2 8 . 8 7 \pm 7 . 0 6$ </td></tr><tr><td>DINO-WM+Dec. [5]</td><td> $1 4 7 . 5 0 \pm 5 0 . 7 1 $ </td><td> $1 5 . 7 4 \pm 1 . 1 6$   $1 8 . 1 0 \pm 0 . 3 3$ </td><td> $0 . 5 1 2 1 \pm 0 . 0 1 0 1$ </td><td> $0 . 2 6 7 8 \pm 0 . 0 2 3 3$   $0 . 3 1 3 4 \substack { \pm 0 . 0 3 9 0 }$ </td><td> $8 . 8 2 \pm 0 . 2 1$ </td></tr><tr><td>MAU+Act [25]</td><td> $6 9 . 3 1 \pm 5 . 4 4$   $6 4 . 1 2 \pm 2 . 8 6$ </td><td> $1 8 . 8 7 \pm 0 . 1 5$ </td><td> $0 . 5 4 2 3 \pm 0 . 0 0 8 2$ </td><td> $0 . 2 6 6 8 \substack { \pm 0 . 0 0 6 0 }$ </td><td> $7 . 8 1 \pm 0 . 1 9$ </td></tr><tr><td>MMVP+Act [26]</td><td> $5 6 . 5 0 \pm 1 . 5 3 $ </td><td> $1 9 . 5 3 \pm 0 . 1 3$ </td><td> $\mathbf { 0 . 5 6 6 9 } _ { \pm 0 . 0 0 5 1 }$ </td><td> $0 . 2 7 1 5 _ { \pm 0 . 0 0 5 9 }$ </td><td> $7 . 6 1 \pm 0 . 0 8$ </td></tr><tr><td>QCC-WM</td><td> $\mathbf { 5 6 . 0 7 } \pm 0 . 0 7$ </td><td> $\mathbf { 1 9 . 8 6 } _ { \pm 0 . 0 2 }$ </td><td> $0 . 5 5 1 8 \scriptstyle \pm 0 . 0 0 1 0$ </td><td> $\mathbf { 0 . 1 6 2 1 } _ { \pm 0 . 0 0 0 7 }$ </td><td> $7 . 3 2 _ { \pm 0 . 0 2 }$ </td></tr></table>

Note: All methods use the same 226 test starting points. Results are the mean ± sample standard deviation across three training seeds. Both MSE metrics are reported in units of $1 0 ^ { - 3 } ;$ bold indicates the best mean. Act denotes an action-conditioning interface, and Dec. denotes the oficial RGB decoder.

Table 5 reports the comparative results on Freehand US. QCC-WM achieves the best mean values across all five metrics at all three horizons. At H = 4, QCC-WM reduces ROI-MSE by 33.09% and increases PSNR by 2 101 dB relative to SimVP-Act, the baseline with the lowest pixel error. It reduces LPIPS by 23.57% relative to PredRNN-Act, the baseline with the lowest perceptual distance. These results indicate that QCC-WM improves pixel reconstruction, structural similarity, and perceptual prediction quality in this transfer setting.

As the prediction horizon increases, ROI-MSE generally rises and SSIM decreases across methods, reflecting error accumulation during recursive prediction. At H = 8 and $H = 1 6 ,$ QCC-WM still reduces ROI-MSE by 18.24% and 5.88%, respectively, relative to the best baseline at each horizon. At $H \ = \ 1 6 ,$ , QCC-WM achieves an SSIM of 0 7788, exceeding SimVP-Act at 0 7698, and reduces temporal MSE by 8.17% relative to PredRNN-V2-Act, the best baseline for this metric. Although the relative pixel-error advantage narrows as the extrapolation horizon increases, QCC-WM maintains favorable structural and temporal prediction performance on this additional dataset.

## 5. Conclusion

We presented QCC-WM, an action-conditioned ultrasound video prediction model that constructs macro-states with spatial correspondence through action-guided quotient mappings. By combining recursive state prediction, motion-guided reference transport, and readback correction under the same quotient measurement, QCC-WM connects macro-level dynamics with future observation generation. Comparative experiments on US-VLA and transfer experiments on Freehand US show improved ultrasound video prediction quality and a sustained perceptual advantage during recursive extrapolation beyond the training window. Ablation studies and mechanism analyses further show that motion-guided reference transport mainly improves pixel reconstruction, while node-level error correction further improves perceptual quality and consistency between generated observations and macro-level predictions. Future work will focus on improving structural preservation in longhorizon prediction and evaluating the framework for closedloop ultrasound navigation.

## References

[1] B. Schölkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal, Y. Bengio, Toward causal representation learning, Proceedings of the IEEE 109 (5) (2021) 612–634. doi:10.1109/JPROC.2021.3058954. URL https://doi.org/10.1109/JPROC.2021. 3058954

[2] K. Chalupka, F. Eberhardt, P. Perona, Multi-level causeefect systems, in: Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, Vol. 51 of Proceedings of Machine Learning Research, PMLR, 2016, pp. 361–369. URL https://proceedings.mlr.press/v51/ chalupka16.html

[3] E. P. Hoel, L. Albantakis, G. Tononi, Quantifying causal emergence shows that macro can beat micro, Proceedings of the National Academy of Sciences 110 (49) (2013) 19790–19795. doi:10.1073/pnas.1314922110. URL https://doi.org/10.1073/pnas.1314922110

[4] P. K. Rubenstein, S. Weichwald, S. Bongers, J. M. Mooij, D. Janzing, M. Grosse-Wentrup, B. Schölkopf, Causal consistency of structural equation models, in: Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence, 2017, pp. 151–160. URL https://proceedings.mlr.press/r15/ rubenstein-17a.html

[5] G. Zhou, H. Pan, Y. LeCun, L. Pinto, DINO-WM: World models on pre-trained visual features enable zero-shot

Table 5: Action-conditioned ultrasound video prediction on Freehand US. All methods use the same 278 test starting points. Results are the mean ± sample standard deviation across three random seeds. At H = 8 16, models trained for four-step prediction are reused for longer-horizon extrapolation. ROI-MSE and tempora MSE are reported in units of 10<sup>−3</sup>. Bold indicates the best mean at each prediction horizon.
<table><tr><td></td><td>Horizon Method</td><td>ROI-MSE↓</td><td>PSNR/dB ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Temporal MSE ↓</td></tr><tr><td rowspan="8"> $H = 4$ </td><td>PredRNN-Act [22]</td><td> $4 . 3 8 2 { \pm } 0 . 1 5 0 $ </td><td> $2 9 . 7 7 5 { \scriptstyle \pm 0 . 1 5 9 }$ </td><td> $0 . 8 2 7 8 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $0 . 1 4 4 8 { \pm } 0 . 0 0 1 4$ </td><td> $1 . 5 5 3 { \pm } 0 . 0 1 4$ </td></tr><tr><td>PredRNN-V2-Act [23]</td><td> $4 . 5 2 4 { \pm } 0 . 1 1 8$ </td><td> $2 9 . 6 4 2 { \scriptstyle \pm 0 . 1 0 7 }$ </td><td> $0 . 8 2 8 3 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $0 . 1 4 7 5 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $1 . 5 6 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td></tr><tr><td>SimVP-Act [24]</td><td> $4 . 0 5 6 { \pm } 0 . 0 1 5$ </td><td> $3 0 . 0 4 8 { \pm } 0 . 0 1 5$ </td><td> $0 . 8 4 0 8 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 9 3 3 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $1 . 6 6 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td>IRASim-S/4 [6]</td><td> $2 0 . 2 3 7 { \scriptstyle \pm 9 . 0 0 6 }$ </td><td> $2 4 . 2 8 5 { \scriptstyle \pm 2 . 0 1 0 }$ </td><td> $0 . 6 3 2 5 { \scriptstyle \pm 0 . 0 4 5 7 }$ </td><td> $0 . 2 1 3 0 { \scriptstyle \pm 0 . 0 2 9 2 }$ </td><td> $1 4 . 1 2 4 { \scriptstyle \pm 4 . 6 1 4 }$ </td></tr><tr><td>DINO-WM [5]</td><td> $8 . 1 4 8 { \pm } 0 . 5 0 9$ </td><td> $2 7 . 2 2 0 { \scriptstyle \pm 0 . 2 0 8 }$ </td><td> $0 . 7 5 2 7 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td><td> $0 . 2 5 0 0 { \scriptstyle \pm 0 . 0 0 6 1 }$ </td><td> $2 . 6 3 1 { \pm } 0 . 0 6 9$ </td></tr><tr><td>MAU-Act [25]</td><td> $4 . 2 0 3 { \pm } 0 . 1 8 2$ </td><td> $2 9 . 9 2 2 { \scriptstyle \pm 0 . 1 8 7 }$ </td><td> $0 . 8 3 0 7 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $0 . 2 0 4 8 { \pm } 0 . 0 0 1 0$ </td><td> $1 . 5 8 1 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td>MMVP-Act [26]</td><td> $4 . 1 3 9 { \pm } 0 . 0 9 3$ </td><td> $3 0 . 0 3 9 { \scriptstyle \pm 0 . 0 9 9 }$ </td><td> $0 . 8 4 5 0 { \scriptstyle \pm 0 . 0 0 3 5 }$ </td><td> $0 . 1 4 6 9 { \scriptstyle \pm 0 . 0 1 1 3 }$ </td><td> $1 . 5 2 2 { \scriptstyle \pm 0 . 0 5 0 }$ </td></tr><tr><td>QCC-WM</td><td> $\pm . 7 1 4 { \pm } 0 . 0 1 4$ </td><td> $\mathbf { 3 2 . 1 4 9 { \scriptstyle \pm 0 . 0 2 1 } }$ </td><td> $\mathbf { 0 . 8 6 2 6 { \scriptstyle \pm 0 . 0 0 0 2 } }$ </td><td> $\mathbf { 0 . 1 1 0 7 { \pm } 0 . 0 0 0 9 }$ </td><td> $\mathbf { 1 . 1 3 2 { \pm } 0 . 0 0 3 }$ </td></tr><tr><td rowspan="8"> $H = 8$ </td><td>PredRNN-Act [22]</td><td> $6 . 0 8 0 { \scriptstyle \pm 0 . 0 9 9 }$ </td><td> $2 8 . 5 5 0 { \scriptstyle \pm 0 . 0 9 4 }$ </td><td> $0 . 7 8 9 6 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 1 7 7 7 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $1 . 3 1 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>PredRNN-V2-Act [23]</td><td> $6 . 2 6 3 { \scriptstyle \pm 0 . 2 4 1 }$ </td><td> $2 8 . 4 1 6 { \pm } 0 . 1 5 4$ </td><td> $0 . 7 9 0 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 1 8 1 0 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td><td> $1 . 3 1 5 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>SimVP-Act [24]</td><td> $5 . 3 8 5 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $2 8 . 9 9 9 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 8 1 0 6 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 2 2 0 5 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $1 . 4 0 2 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>IRASim-S/4 [6]</td><td> $3 5 . 1 2 7 { \scriptstyle \pm 1 9 . 7 0 0 }$ </td><td> $2 2 . 3 3 9 { \scriptstyle \pm 2 . 4 8 3 }$ </td><td> $0 . 5 7 0 7 { \scriptstyle \pm 0 . 0 5 7 7 }$ </td><td> $0 . 2 4 8 2 { \scriptstyle \pm 0 . 0 3 2 9 }$ </td><td> $1 3 . 2 9 7 { \scriptstyle \pm 4 . 5 8 1 }$ </td></tr><tr><td>DINO-WM [5]</td><td> $1 0 . 3 5 2 { \scriptstyle \pm 0 . 7 9 2 }$ </td><td> $2 6 . 3 1 2 { \scriptstyle \pm 0 . 2 5 2 }$ </td><td> $0 . 7 3 6 0 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $0 . 2 5 7 1 { \scriptstyle \pm 0 . 0 0 6 3 }$ </td><td> $1 . 8 9 4 { \pm } 0 . 0 3 5$ </td></tr><tr><td>MAU-Act [25]</td><td> $5 . 7 9 7 { \scriptstyle \pm 0 . 3 0 9 }$ </td><td> $2 8 . 7 2 8 { \scriptstyle \pm 0 . 2 3 0 }$ </td><td> $0 . 7 9 7 9 { \scriptstyle \pm 0 . 0 0 2 2 }$ </td><td> $0 . 2 2 3 8 { \pm } 0 . 0 0 1 8$ </td><td> $1 . 3 3 9 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td>MMVP-Act [26]</td><td> $5 . 7 0 2 { \scriptstyle \pm 0 . 1 4 3 }$ </td><td> $2 8 . 8 5 1 { \scriptstyle \pm 0 . 1 0 8 }$ </td><td> $0 . 8 0 9 9 { \scriptstyle \pm 0 . 0 0 3 6 }$ </td><td> $0 . 1 8 8 3 { \scriptstyle \pm 0 . 0 1 2 1 }$ </td><td> $1 . 3 0 3 { \scriptstyle \pm 0 . 0 3 7 }$ </td></tr><tr><td>QCC-WM</td><td> $\mathbf { 4 . 4 0 3 { \pm } 0 . 0 0 5 }$ </td><td> $\mathbf { 3 0 . 4 1 3 { \scriptstyle \pm 0 . 0 1 1 } }$ </td><td> $\mathbf { 0 . 8 2 5 1 { \pm } 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 1 3 9 1 { \pm } 0 . 0 0 1 5 }$ </td><td> $\mathbf { 1 . 1 0 6 { \pm } 0 . 0 0 1 }$ </td></tr><tr><td rowspan="8"> $H = 1 6$ </td><td>PredRNN-Act [22]</td><td> $9 . 4 2 4 { \scriptstyle \pm 0 . 6 3 2 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>PredRNN-V2-Act [23]</td><td> $9 . 5 3 3 { \pm } 0 . 5 6 9$ </td><td> $2 6 . 9 3 4 { \scriptstyle \pm 0 . 1 6 8 }$ </td><td> $0 . 7 4 4 7 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td><td> $0 . 2 1 8 2 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $1 . 1 9 6 { \pm } 0 . 0 0 8$ </td></tr><tr><td>SimVP-Act [24]</td><td> $8 . 2 6 0 { \scriptstyle \pm 0 . 1 0 2 }$ </td><td> $2 6 . 8 4 1 \pm 0 . 2 2 9$ </td><td> $0 . 7 4 4 2 { \scriptstyle \pm 0 . 0 0 1 4 }$ </td><td> $0 . 2 2 0 6 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $1 . 1 9 3 { \pm } 0 . 0 0 3$ </td></tr><tr><td>IRASim-S/4 [6]</td><td> $6 9 . 1 1 3 { \scriptstyle \pm 5 1 . 1 2 6 }$ </td><td> $2 7 . 4 5 8 { \pm } 0 . 0 3 9$ </td><td> $0 . 7 6 9 8 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $0 . 2 5 0 0 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td><td> $1 . 2 7 6 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>DINO-WM [5]</td><td> $1 5 . 3 3 4 \pm 1 . 4 5 9$ </td><td> $2 0 . 0 3 5 { \scriptstyle \pm 3 . 2 0 9 }$ </td><td> $0 . 4 9 1 5 { \scriptstyle \pm 0 . 0 7 8 5 }$ </td><td> $0 . 2 9 9 9 { \scriptstyle \pm 0 . 0 3 8 7 }$ </td><td> $1 3 . 4 4 9 { \scriptstyle \pm 5 . 1 1 9 }$ </td></tr><tr><td>MAU-Act [25]</td><td> $9 . 4 8 0 { \scriptstyle \pm 0 . 6 9 1 }$ </td><td> $2 4 . 8 4 9 { \scriptstyle \pm 0 . 3 1 9 }$ </td><td> $0 . 7 0 8 1 { \scriptstyle \pm 0 . 0 0 2 1 }$   $0 . 7 5 3 2 { \scriptstyle \pm 0 . 0 0 5 4 }$ </td><td> $0 . 2 7 0 9 { \scriptstyle \pm 0 . 0 0 7 0 }$   $0 . 2 4 8 3 { \pm } 0 . 0 0 4 5$ </td><td> $1 . 5 0 8 { \pm } 0 . 0 2 1$ </td></tr><tr><td>MMVP-Act [26]</td><td> $8 . 5 5 0 { \scriptstyle \pm 0 . 2 4 9 }$ </td><td> $2 6 . 9 5 0 { \scriptstyle \pm 0 . 2 9 1 }$   $2 7 . 3 4 8 { \pm } 0 . 1 2 9$ </td><td> $0 . 7 6 7 7 { \scriptstyle \pm 0 . 0 0 4 0 }$ </td><td> $0 . 2 3 4 4 { \scriptstyle \pm 0 . 0 0 9 6 }$ </td><td> $1 . 2 2 3 { \pm } 0 . 0 0 8$   $1 . 1 9 4 { \pm } 0 . 0 2 5$ </td></tr><tr><td>QCC-WM</td><td> ${ \bf 7 . 7 7 5 { \pm } 0 . 1 0 2 }$ </td><td> $\mathbf { 2 8 . 3 4 9 { \scriptstyle \pm 0 . 0 2 4 } }$ </td><td> $\mathbf { 0 . 7 7 8 8 { \pm } 0 . 0 0 0 3 }$ </td><td> $\mathbf { 0 . 1 8 5 2 { \pm } 0 . 0 0 3 1 }$ </td><td> $\mathbf { 1 . 0 9 5 { \pm 0 . 0 0 1 } }$ </td></tr></table>

planning, in: Proceedings of the 42nd International Conference on Machine Learning, Vol. 267 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 79115–79135. URL https://proceedings.mlr.press/v267/ zhou25t.html

[6] F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, T. Kong, IRASim: A fine-grained world model for robot manipulation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 9834–9844. doi:10.1109/ICCV51701.2025.00917. URL https://doi.org/10.1109/ICCV51701.2025. 00917

[7] Z. Wu, N. Dvornik, K. Gref, T. Kipf, A. Garg, Slot-Former: Unsupervised visual dynamics simulation with object-centric models, in: The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= TFbwV6I0VLg

[8] P. Wen, Q. Xu, M. Lei, S. Dai, Y. Liu, Q. Huang, Slot-JEPA: Object-structured prediction interfaces for latent world models, arXiv preprint (2026). arXiv:2610. 28074. URL https://arxiv.org/abs/2610.28074

[9] X. Shi, Z. Huang, F.-Y. Wang, W. Bian, D. Li, Y. Zhang, M. Zhang, K. C. Cheung, S. See, H. Qin, J. Dai, H. Li, Motion-I2V: Consistent and controllable image-to-video generation with explicit motion modeling, in: ACM SIG-GRAPH 2024 Conference Papers, Association for Computing Machinery, 2024, pp. 1–11. doi:10.1145/ 3641519.3657497. URL https://doi.org/10.1145/3641519.3657497

[10] P. Lippe, S. Magliacane, S. Löwe, Y. M. Asano, T. Cohen, S. Gavves, CITRIS: Causal identifiability from temporal intervened sequences, in: Proceedings of the 39th International Conference on Machine Learning, Vol. 162 of Proceedings of Machine Learning Research, PMLR, 2022, pp. 13557–13603. URL https://proceedings.mlr.press/v162/ lippe22a.html

[11] P. Lippe, S. Magliacane, S. Löwe, Y. M. Asano, T. Cohen, E. Gavves, BISCUIT: Causal representation learning from binary interactions, in: Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, Vol. 216 of Proceedings of Machine Learning Research, PMLR, 2023, pp. 1263–1273. URL https://proceedings.mlr.press/v216/ lippe23a.html

[12] C. Gelada, S. Kumar, J. Buckman, O. Nachum, M. G. Bellemare, DeepMDP: Learning continuous latent space models for representation learning, in: Proceedings of the 36th International Conference on Machine Learning,

Vol. 97 of Proceedings of Machine Learning Research, PMLR, 2019, pp. 2170–2179. URL https://proceedings.mlr.press/v97/ gelada19a.html

[13] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, Learning latent dynamics for planning from pixels, in: Proceedings of the 36th International Conference on Machine Learning, Vol. 97 of Proceedings of Machine Learning Research, PMLR, 2019, pp. 2555–2565. URL https://proceedings.mlr.press/v97/ hafner19a.html

[14] D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, Dream to control: Learning behaviors by latent imagination, in: International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= S1lOTC4tDS

[15] M. Seitzer, M. Horn, A. Zadaianchuk, D. Zietlow, T. Xiao, C.-J. Simon-Gabriel, T. He, Z. Zhang, B. Schölkopf, T. Brox, F. Locatello, Bridging the gap to real-world object-centric learning, in: The Eleventh International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2209.14860

[16] Z. Wu, J. Hu, W. Lu, I. Gilitschenski, A. Garg, SlotDifusion: Object-centric generative modeling with difusion models, in: Advances in Neural Information Processing Systems, Vol. 36, 2023. URL https://proceedings.neurips. cc/paper\_files/paper/2023/hash/ 9fa03b16dbd6cabc7601fe98c6ec291e-Abstract-Conferenc html

[17] G. Plepi, S. Behnke, SlotDiT: Object-centric representations for difusion transformers, accepted at BMVC 2026; arXiv preprint (2026). arXiv:2609.17414. URL https://arxiv.org/abs/2609.17414

[18] C. Gao, H. Zhang, Z. Xu, Z. Cai, L. Shao, FLIP: Flowcentric generative planning as general-purpose manipulation world model, arXiv preprint (2024). arXiv:2412. 08261. URL https://arxiv.org/abs/2412.08261

[19] A. Lee, M. Xu, Y. Cao, S. Tian, Z. Liu, DeXplicit: Making physical state explicit in dexterous manipulation world models, arXiv preprint (2026). arXiv:2610.30781. URL https://arxiv.org/abs/2610.30781

[20] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, N. Ballas, Self-supervised learning from images with a joint-embedding predictive architecture, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 15619–15629.

[21] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic, Detecting and quantifying causal associations in large nonlinear time series datasets, Science Advances 5 (11) (2019) eaau4996. doi:10.1126/sciadv. aau4996. URL https://doi.org/10.1126/sciadv.aau4996

[22] Y. Wang, M. Long, J. Wang, Z. Gao, P. S. Yu, PredRNN: Recurrent neural networks for predictive learning using spatiotemporal lstms, in: Advances in Neural Information Processing Systems, Vol. 30, 2017. URL https://proceedings.neurips. cc/paper\_files/paper/2017/hash/ e5f6ad6ce374177eef023bf5d0c018b6-Abstract. html

[23] Y. Wang, H. Wu, J. Zhang, Z. Gao, J. Wang, P. S. Yu, M. Long, PredRNN: A recurrent neural network for spatiotemporal predictive learning, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (2) (2023) 2208–2225. doi:10.1109/TPAMI.2022.3165153. URL https://arxiv.org/abs/2103.09504

[24] Z. Gao, C. Tan, L. Wu, S. Z. Li, SimVP: Simpler yet better video prediction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3170–3180. URL https://openaccess.thecvf.com/content/ CVPR2022/html/Gao\_SimVP\_Simpler\_Yet\_Better\_ Video\_Prediction\_CVPR\_2022\_paper.html

[25] Z. Chang, X. Zhang, S. Wang, S. Ma, Y. Ye, X. Xiang, W. Gao, MAU: A motion-aware unit for video prediction and beyond, in: Advances in Neural Information Processing Systems, Vol. 34, 2021. URL https://papers.nips.cc/paper/2021/hash/ e25cfa90f04351958216f97e3efdabe9-Abstract. html

[26] Y. Zhong, L. Liang, I. Zharkov, U. Neumann, MMVP: Motion-matrix-based video prediction, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4273–4283. URL https://openaccess.thecvf.com/ content/ICCV2023/html/Zhong\_MMVP\_ Motion-Matrix-Based\_Video\_Prediction\_ICCV\_ 2023\_paper.html

[27] C. Tan, S. Li, Z. Gao, W. Guan, Z. Wang, Z. Liu, L. Wu, S. Z. Li, OpenSTL: A comprehensive benchmark of spatio-temporal predictive learning, in: Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023. URL https://github.com/chengtan9907/OpenSTL

[28] Q. Li, Z. Shen, Q. Li, D. C. Barratt, T. Dowrick, M. J. Clarkson, T. Vercauteren, Y. Hu, Freehand ultrasound without external trackers, Zenodo, version v2 (2022). doi:10.5281/zenodo.7740734. URL https://zenodo.org/records/7740734

[29] Q. Li, Z. Shen, Q. Li, D. C. Barratt, T. Dowrick, M. J. Clarkson, T. Vercauteren, Y. Hu, Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames, in: 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), 2023, pp. 1–5. doi:10.1109/ISBI53787. 2023.10230773.