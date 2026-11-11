# TanCRAFT: Context-Reasoned Anticipatory Feedback for Assembly Tasks

Sassan Mokhtar<sup>1</sup>, Lars Doorenbos<sup>1,2</sup>, Marius Bock<sup>1,2</sup>, Sina Mokhtar<sup>3</sup>,

Katharina Beckh<sup>2,4</sup>, Dominik Bach<sup>1</sup>, and Jurgen Gall¨ <sup>1,2</sup>

Abstract— Preventing human error in physical assembly tasks requires assistance systems that can intervene before mistakes are made. This is especially challenging in settings where users are free to choose the construction order and correctness depends on the evolving assembly rather than a fixed next step. We present TanCRAFT, a four-stage real-time predictive assistance system that tracks the scene state, reasons about the current partial assembly, forecasts the near-future motion of manipulated pieces, and intervenes when the predicted placement is incompatible with the target. We instantiate our framework and confirm its effectiveness in a controlled tangram assembly task through offline benchmarks and a within-subject user study. Our findings show that the predictive feedback significantly reduces mistakes relative to unguided and reactive sessions, while yielding the lowest mean perceived workload and shortest mean completion time. Overall, these results demonstrate the value of predictive feedback for error prevention and provide a step toward cost-aware assistance for automation and human-robot workcells.

## I. INTRODUCTION

Human error remains a significant contributor to failures and quality defects in manual assembly, industrial production, and human-robot workcells [1], [2], [3]. While some mistakes are easily reversed, others are costly once completed. The cost therefore depends not only on whether errors are detected, but also on when. For example, a misplaced component or unsafe action may be easy to correct before commitment, but costly, time-consuming, or even unsafe after completion. This motivates the need for assistance systems that go beyond post-hoc error detection and are capable of real-time error forecasting and prevention.

Anticipating errors is challenging because systems must infer human actions, intentions, and potential mistakes from external observations. Moreover, anticipation alone is insufficient for assistance, as it does not address how a system should act to prevent the mistake. In this context, previous predictive-assistance methods anticipate supportive robot behaviors or human assembly actions [4], [5]. TRAFA [6] extends prediction to proactive error feedback but operates in a sequential task with a fixed next correct action. Still, many real scenarios, such as kit or furniture assembly, are far less constrained, letting users choose their own strategy, revise placements, and manipulate pieces without committing them. This freedom makes interaction more natural, but error anticipation substantially harder. To address this challenge, we propose TanCRAFT, a real-time predictive assistance framework for non-sequential assembly comprising four stages: Track, Reason, Forecast, and Intervene.

![](images/a0c6f7d01348ae6a30cd76088b5ca469d1433309a8fc04cddbf275c0db5eb30c.jpg)  
Fig. 1. TanCRAFT overview. (A) The user assembles the target tangram shape while the system observes the workspace. (B) The system reasons about the evolving assembly state and assigns placed pieces to target-shape subgraphs. (C) As the user moves a piece toward an incompatible location, TanCRAFT issues audio-visual feedback to prevent the mistake.

We instantiate the framework in a controlled tangram assembly task, which recently gained popularity in robotics assembly and manipulation tasks because it combines objective correctness with an open construction order, interchangeable pieces, and exploratory manipulation [7], [8], i.e., the same properties that make unconstrained assembly a difficult task. Unlike prior works, where robots assemble the pieces, we use tangram to study predictive assistance to humans, as it provides a natural way to induce human error without the costs associated with a full industrial setup. We validate our system through offline benchmarks, and evaluate the full system in a within-subject user study comparing two variants of TanCRAFT to unguided and reactive (post-hoc) feedback conditions. The user study shows that predictive feedback significantly outperforms both no-feedback and reactive feedback baselines in mistake reduction, while also achieving the lowest mean perceived workload [9], the highest success rate, and the shortest mean completion time.

In short, the main contributions of this work are:

• We introduce TanCRAFT, a real-time framework for predictive assistance in non-sequential assembly tasks.

• We show through a user study that predictive feedback significantly reduces mistakes, while achieving the lowest mean workload and shortest mean completion time.

• We will release the code, trained model weights, and multi-view tangram assembly dataset upon acceptance.

## II. RELATED WORK

## A. Predictive assistance for robots in HRC and assembly

Predictive assistance has been widely studied in humanrobot collaboration (HRC), particularly in assembly and manufacturing, where robots need to anticipate human behavior to adapt their actions. Prior work has modeled multistep activity for robot planning [10], introduced hierarchical planning and task allocation for collaborative assembly [11], and integrated human motion prediction with robot planning [12]. Other approaches combine plan recognition and trajectory prediction to infer human plans and adapt robot actions accordingly [13]. Preference-based methods predict supportive robot behaviors [4] or adapt to human assembly preferences for proactive part delivery [5]. In contrast, our focus is on anticipating human errors during ongoing human manipulation, where exploratory motion makes placement intent ambiguous, and feedback must arrive early enough to redirect the piece.

## B. Feedback, guidance, and the cost of intervention

Feedback and guidance are commonly used in assisted manual work to communicate task instructions, support execution, or flag deviations from the intended procedure [14], [15]. However, online feedback during an ongoing action also competes for the user’s attention. Interruption studies show that unsolicited messages can increase completion time, error rate, annoyance, and anxiety, with a cost that depends on the user’s task state when they are delivered [16], [17]. Warningsystem research further shows that unreliable alerts and false alarms can reduce compliance and reliance on automated warnings [18]. In human-robot collaborations, Mannem et al. [19] show that robot queries can reduce perceived helpfulness even when objective task performance is largely unaffected. LeMasurier et al. [20] anticipate robot failures and provide proactive explanations, which participants rate more favorably than reactive explanations. Together with work on cost-aware suggestion timing [21], these studies motivate selective, cost-aware intervention in non-sequential assembly: feedback should help prevent errors while avoiding unnecessary interruptions and additional workload.

![](images/70f36a4f07e46a88cc068781eb027d05fa2f2b2280c207e90b97083b0f0f2ad4.jpg)  
Fig. 2. Tangram pieces and target shapes used in the study. (A) The seven physical tangram pieces used for assembly. (B) The six target shapes used during data collection and training. In the user study, the rocket was used in the written instructions, the house was used for the warm-up session, and the four animal shapes were used for testing the feedback modes.

## C. Predictive feedback beyond sequential tasks

Predictive feedback for human users in non-sequential assembly remains largely unexplored. TRAFA [6] uses a Track-Forecast-Act loop to warn users before incorrect actions are completed. However, TRAFA operates in a sequential Duplo-stacking task with a fixed target order. In contrast, tangram assembly allows users to choose their placement order and construct multiple partial structures in parallel, making fixed next-step rules insufficient. TanCRAFT addresses this challenge by combining motion forecasting with reasoning about the evolving assembly: it predicts where a piece will be placed and checks whether that subgraph can still accommodate its type. This allows the system to anticipate errors without assuming a fixed placement order.

## III. METHOD

Users recreate a target shape S from the Tangram pieces $\mathcal { P } = \{ p _ { i } \} _ { i = 1 } ^ { N }$ , where each piece has a geometric type $\tau ( p _ { i } ) \in$ T. The pieces consist of two large triangles, one medium triangle, two small triangles, one square, and one parallelogram, with same-type pieces interchangeable (Fig. 2). A completed assembly is correct when the final configuration of all pieces is geometrically consistent with the target shape S. We decompose the target shape into subgraphs $\mathcal { G } ( S ) =$ $\{ g _ { k } \} _ { k = 1 } ^ { K }$ , where each subgraph corresponds to a part of the target shape, such as the head or body of an animal. Each subgraph has a prescribed composition, where $r _ { k } ( \tau )$ denotes the required number of pieces of type τ in subgraph $g _ { k }$

We restrict our notion of a mistake to consequential errors: placing a piece into a subgraph with no remaining capacity of its type. This excludes minor adjustments made during placement, such as when orienting a piece within its intended subgraph. We denote the set of pieces being manipulated at time t by $\mathcal { A } _ { t } \subseteq \mathcal { P }$ . Then, for an active piece $a _ { t } \in \mathcal A _ { t } .$ , let $n _ { k } ^ { t } ( \tau )$ count the other pieces of type τ currently assigned to subgraph $g _ { k }$ . A subgraph is invalid for an active piece $a _ { t }$ when the subgraph has no remaining capacity for that piece’s type, i.e., $n _ { k } ^ { t } ( \tau ( a _ { t } ) ) \geq r _ { k } ( \tau ( a _ { t } ) )$ ). This, in turn, forms the set of invalid subgraph destinations

$$
\begin{array} { r } { \mathcal { T } _ { t } ( a _ { t } ) = \{ g _ { k } \in \mathcal { G } ( S ) : n _ { k } ^ { t } ( \tau ( a _ { t } ) ) \geq r _ { k } ( \tau ( a _ { t } ) ) \} . } \end{array}\tag{1}
$$

This set changes as pieces are placed, removed, or reassigned. Thus, our task is to anticipate, for each active piece and before its placement is complete, whether it is being moved into a subgraph in $\textstyle { \mathcal { T } } _ { t } ( a _ { t } )$ , and to intervene only in that case.

## A. System Overview

Our framework forecasts the mistakes defined above from an overhead RGB stream and converts them into actionable feedback. At time t, the system processes the overhead RGB stream to obtain and maintain a task state

$$
\mathbf { x } _ { t } = \left( \{ \mathbf { z } _ { i } ^ { t } \} _ { i = 1 } ^ { N } , \mathbf { h } _ { t } , \{ \hat { s } _ { i } ^ { t } \} _ { i = 1 } ^ { N } , \mathcal { A } _ { t } , S \right) ,\tag{2}
$$

where $\mathbf { z } _ { i } ^ { t }$ denotes the estimated planar position and orientation of piece $p _ { i }$ , h<sub>t</sub> denotes the estimated hand state, and $\hat { s } _ { i } ^ { t } \in \mathcal G ( S ) \cup \{ \varnothing \}$ is the predicted subgraph assignment of piece $p _ { i } .$ , with ∅ indicating that a piece is not currently assigned to any target subgraph. The system uses this state to anticipate incompatible placements through four stages: Track, Reason, Forecast, and Intervene (Fig. 3).

## B. Track: Perception and Scene-State Estimation

The first stage converts the overhead RGB stream into the scene state by segmenting tangram pieces, assigning persistent instance identities, deriving the piece type from mask geometry, and using color to disambiguate shape duplicates. We use SAM3 [22] to initialize and propagate piece masks through the video stream. From each tracked mask, we extract the image-plane center and in-plane orientation, yielding the planar pose,

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { t } = ( \mathbf { c } _ { i } ^ { t } , \theta _ { i } ^ { t } ) , } \end{array}\tag{3}
$$

where $\mathbf { c } _ { i } ^ { t } \in \mathbb { R } ^ { 2 }$ is the piece center and $\theta _ { i } ^ { t } \in [ 0 , 2 \pi )$ is its orientation. In parallel, we use MediaPipe Hands [23] to estimate 2D hand keypoints, which define the hand pose h .

A key difficulty is hand occlusion: the active piece is often covered when its motion is most informative for forecasting. We therefore track each piece with a constantvelocity Kalman filter [24] over its planar pose, using measurement updates when reliable masks are available and prediction-only updates otherwise. This smoothes trajectories and bridges short gaps in case of occlusions. From these, the active object set $\boldsymbol { A } _ { t }$ is inferred from hand–object contact and recent object motion, denoting a piece as active when it overlaps with the fingertips.

## C. Reason: Unconstrained Assembly Reasoning

The Reason stage estimates the current assembly state by assigning each piece $p _ { i }$ to a subgraph in G(S) or ∅. The assignments of all pieces are denoted by $\hat { \bf s } ^ { t } = $ $( \hat { s } _ { 1 } ^ { t } , \dots , \hat { s } _ { N } ^ { t } )$ ). For the assignment, we use a learned Piece-to-Subgraph Classifier $f _ { \phi }$ , applied independently to each piece. We first group spatially adjacent pieces into clusters, which are connected components of the piece-adjacency graph, with adjacency determined by mask dilation and a centroiddistance threshold. The vector $\mathbf { r } _ { i } ^ { t }$ encodes which detected cluster contains piece $p _ { i } ,$ including a separate category for pieces that belong to no clusters. These geometric groupings do not carry the information about the subgraphs, which is inferred by the classifier $f _ { \phi }$

Given a piece’s planar pose ${ \bf z } _ { i } ^ { t } ,$ type $\tau ( p _ { i } )$ , cluster membership $\mathbf { r } _ { i } ^ { t } ,$ , and the target shape S, the classifier $f _ { \phi }$ predicts the subgraph in which the piece is placed, or the null class otherwise:

$$
\begin{array} { r } { \pmb { \pi } _ { i } ^ { t } = f _ { \phi } \left( \mathbf z _ { i } ^ { t } , \tau ( p _ { i } ) , \mathbf r _ { i } ^ { t } , S \right) , \qquad \pmb { \pi } _ { i } ^ { t } \in \Delta ^ { K + 1 } , } \end{array}\tag{4}
$$

where $\boldsymbol { \pi } _ { i } ^ { t }$ is the probability distribution over the K subgraphs and the null class, and $\Delta ^ { K + 1 }$ denotes the probability simplex. The piece is then assigned to the most likely class,

$$
\hat { s } _ { i } ^ { t } = \arg \operatorname* { m a x } \pi _ { i } ^ { t } .\tag{5}
$$

The resulting assignments $\hat { \mathbf { s } } ^ { t }$ determine the occupancy counts $n _ { k } ^ { t } ( \tau )$ used in Eq. (1). The target shape S is supplied as a one-hot label vector over the target shapes, which lets a single model operate across all shapes.

## D. Forecast: Active-Piece Motion Forecasting

The Forecast stage uses a learned forecaster $f _ { \psi }$ to predict the future 2D center trajectory of active pieces over H time steps, conditioned on their preceding $T$ observed positions, the current scene context, and formed assembly clusters:

$$
\hat { \mathbf { c } } _ { a _ { t } } ^ { t + 1 : t + H } = f _ { \psi } \left( \mathbf { c } _ { a _ { t } } ^ { t - T + 1 : t } , \{ \mathbf { c } _ { i } ^ { t } \} _ { i = 1 } ^ { N } , a _ { t } , \{ \bar { \mathbf { c } } _ { m } ^ { t } \} _ { m = 1 } ^ { M } \right) ,\tag{6}
$$

where $\mathbf { c } _ { a _ { t } } ^ { t - T + 1 : t }$ is the recent position history of the active piece, $\{ \mathbf { c } _ { i } ^ { t } \} _ { i = 1 } ^ { N }$ are the current positions of all pieces in the workspace, $a _ { t }$ is the active piece, and $\{ \bar { \mathbf { c } } _ { m } ^ { t } \} _ { m = 1 } ^ { \bar { M } }$ are the centroids of the M currently formed assembly clusters.

The forecaster predicts the future trajectory as displacements $\Delta \hat { \mathbf { c } } _ { a _ { t } } ^ { t + 1 : t + H }$ . These displacements are accumulated from the current active-piece position to obtain the future locations:

$$
\hat { \mathbf { c } } _ { a _ { t } } ^ { t + \ell } = \mathbf { c } _ { a _ { t } } ^ { t } + \sum _ { j = 1 } ^ { \ell } \Delta \hat { \mathbf { c } } _ { a _ { t } } ^ { t + j } , \qquad \ell = 1 , \ldots , H .\tag{7}
$$

These forecasts are passed to the intervention stage to estimate the active piece’s likely destination before placement is complete.

## E. Intervene: Placement-Intent Estimation and Predictive Intervention

The Intervene stage estimates whether the ongoing manipulation will produce an invalid placement and decides whether this estimate warrants feedback. It draws on the assembly estimate $\hat { \mathbf { s } } ^ { t }$ from the reasoning stage and the forecast trajectory $\hat { \mathbf { c } } _ { a _ { t } } ^ { t + 1 : t + H }$ from the forecasting stage to estimate the intended destination of the active piece before placement is complete, using a learned placement-intent model $f _ { \omega }$ that outputs a distribution over possible subgraph destinations:

$$
\begin{array} { r } { \mathbf q _ { t } = f _ { \omega } \left( \hat { \mathbf c } _ { a _ { t } } ^ { t + 1 : t + H } , \delta _ { t } \right) , \qquad \mathbf q _ { t } \in \Delta ^ { K + 1 } , } \end{array}\tag{8}
$$

where the additional null class represents non-placement or uncertain motion. The proximity vector $\pmb { \delta } _ { t } \in \mathbb { R } ^ { K }$ has entries $\delta _ { t , k } = \exp ( - d _ { t , k } / \sigma )$ , where $d _ { t , k }$ is the image-plane distance from the active piece’s current center to the nearest pixel of subgraph $g _ { k } \mathbf { \dot { s } }$ mask and $\sigma = 8 0$ pixels is a distance scale. For an empty subgraph, $\delta _ { t , k } = 0$

![](images/08f462eae5ed8dcd7c721604c2c1aa3081f8b73be7df6da338a1525505fc6cbd.jpg)  
Fig. 3. Overview of TanCRAFT. Given an overhead RGB stream, the Track stage estimates hand poses, tracks tangram-piece masks, and estimates object poses using a Kalman filter. The Reason stage infers the evolving assembly state with a Piece-to-Subgraph Classifier. The Forecast stage predicts the near-future motion of active pieces from their recent trajectories and scene context. Finally, the Intervene stage estimates active-piece destinations and checks their compatibility with the remaining subgraph capacities to drive predictive audio-visual warnings or continuous feedback.

From these predictions, the intervention policy evaluates whether the most likely destination is compatible with the assembly constraints. Specifically, we use the subgraph index and confidence,

$$
\hat { k } _ { t } = \arg \operatorname* { m a x } _ { k \in \{ 1 , \dots , K \} } q _ { t , k } , \qquad \hat { q } _ { t } = q _ { t , \hat { k } _ { t } } ,\tag{9}
$$

to obtain the frame-level invalid-placement confidence

$$
\rho _ { t } = { \left\{ \begin{array} { l l } { { \hat { q } } _ { t } , } & { g _ { { \hat { k } } _ { t } } \in { \mathcal { T } } _ { t } ( a _ { t } ) , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{10}
$$

The confidence $\rho _ { t }$ corresponds to the predicted probability $\hat { q } _ { t }$ that the active piece $a _ { t }$ will be added to a subgraph $\textstyle { \mathcal { T } } _ { t } ( a _ { t } )$ (1) it does not belong to. The set $\textstyle { \mathcal { T } } _ { t } ( a _ { t } )$ changes as pieces are placed, removed, or reassigned.

Our predictive approach emits a warning according to

$$
u _ { t } ^ { \mathrm { p r e d } } = \left\{ \begin{array} { l l } { \mathrm { w a r n } , } & { \rho _ { t } \geq \gamma , } \\ { \mathrm { s i l e n t } , } & { \rho _ { t } < \gamma , } \end{array} \right.\tag{11}
$$

where $\gamma$ controls the tradeoff between early sensitivity and false alarms. The null class is excluded from destination selection: a warning requires a non-null destination probability of at least $\gamma$

## F. Training and Online Inference

We train the modules offline using data from a separate pilot cohort, described in Sec. IV. The piece-to-subgraph network $f _ { \phi } \ ( 4 )$ is a three-layer multilayer perceptron (MLP), trained with cross-entropy over the K target subgraphs and the null class,

$$
\mathcal { L } _ { \mathrm { s u b g r a p h } } = - \sum _ { k = 0 } ^ { K } y _ { i , k } ^ { t } \log \pi _ { i , k } ^ { t } ,\tag{12}
$$

with $\mathbf { y } _ { i } ^ { t }$ being the ground-truth label of the subgraph the piece $p _ { i }$ belongs to at time t.

The active-piece forecaster $f _ { \psi }$ (6) is implemented as a temporal convolutional network (TCN) [25]. It encodes three inputs in parallel: the active piece’s past trajectory via a twolayer TCN, and the other pieces’ positions and the cluster centroids each via a two-layer MLP. These are concatenated into a single fused representation, which a three-layer temporal convolutional decoder maps to per-step displacements. The forecaster is trained with a position regression loss over a horizon H:

$$
\mathcal { L } _ { \mathrm { f o r e c a s t } } = \sum _ { \ell = 1 } ^ { H } \left\| \hat { \mathbf { c } } _ { a _ { t } } ^ { t + \ell } - \mathbf { c } _ { a _ { t } } ^ { t + \ell } \right\| _ { 2 } ^ { 2 } .\tag{13}
$$

The placement-intent model $f _ { \omega }$ (8) predicts the destination subgraph of an active piece within its horizon. It encodes the forecast trajectory using a single-layer gated recurrent unit (GRU), concatenates the resulting representation with the proximity vector $\delta _ { t } ,$ and classifies the features with a three-layer MLP. The model is trained with class-weighted cross-entropy over the $K + 1$ destination classes:

$$
\mathcal { L } _ { \mathrm { i n t e n t } } = - \sum _ { k = 0 } ^ { K } w _ { k } y _ { t , k } \log q _ { t , k } ,\tag{14}
$$

where $w _ { k }$ are inverse training-class frequencies, normalized to one. During online inference, perception updates piece and hand poses and identifies active pieces, while reasoning assigns non-active pieces to target subgraphs. For each active piece, the forecaster predicts its short-horizon trajectory, the placement-intent model estimates its destination, and the intervention policy combines destination confidence with subgraph-capacity constraints to trigger predictive warning or continuous feedback. Our system operates at ≈9.7 fps, with runtime dominated by SAM3 mask propagation.

## IV. EXPERIMENTS AND RESULTS

We evaluate our framework with offline benchmarks (Secs. IV-B, IV-C) and a user study (Secs. IV-D, IV-E). We implement our framework for the standard tangram set, which has $N = 7$ pieces, and set $K = 2 ,$ , decomposing the target shapes into two subgraphs of three and four pieces (e.g., the head and body of an animal). Still, the formulation is general and extends to more pieces and finer subgraph decompositions.

TABLE I  
PIECE-TO-SUBGRAPH CLASSIFICATION ACCURACY.
<table><tr><td>Method</td><td>Overall accuracy</td></tr><tr><td>Location Prior</td><td>71.0%</td></tr><tr><td>Bayesian Location Update</td><td>73.4%</td></tr><tr><td>Piece-to-Subgraph Classifier</td><td>95.8%</td></tr></table>

## A. Data Collection and Training Splits

For developing the modules and creating the offline benchmarks, we collected 72 pilot assembly sessions from 16 participants across six target shapes, with 12 sessions per shape. Of these, the first 36 sessions were conducted without feedback, providing the data to train the initial system. Then, 36 sessions used this initial system, with 12 each under reactive (warning after incorrect placement), predictive (warning before anticipated incorrect placement), and continuous feedback (a sustained tone whose loudness reflects predicted error confidence). Pilot trials ended upon completion or after five minutes. Participants assembled the target shape displayed on a monitor; mistakes arose naturally. Following this data collection, we trained and evaluated the final modules on appropriate subsets of these recordings covering the different feedback modalities with both correct and incorrect placements.

The subgraph classifier and motion forecaster use a shapestratified partition of 60 development and 12 test sessions, with 85% of annotated frames in the development partition used for training and the remaining 15% for validation. All baselines are trained on the same 60 sessions for developing and 12 for testing. In contrast, the placement-intent estimator uses only the 48 no-feedback and reactive sessions, excluding predictive feedback modes as these alter the intended placement, yielding 30 training, 6 validation, and 12 test sessions. For the full system-level benchmark, we use the 40 no-feedback and reactive sessions from the final user study (Sec. IV-D), again avoiding trajectories altered by predictive or continuous feedback.

## B. Offline Module Benchmarks

Here, we evaluate the three learned modules on the pilot partitions described in Sec. IV-A to validate their design and select configurations for online use.

First, we report the accuracy of the piece-to-subgraph classifier in Tab. I against two baselines: Location Prior, which assigns pieces using training set location priors, and Bayesian Location Update, which treats the location distribution as a prior and updates it using a likelihood score based on how well the piece types within a cluster match the expected composition of each subgraph. The classifier uses 958 training, 168 validation, and 263 test labeled frames. It achieves 95.8% overall accuracy (Tab. I).

Second, Tab. II evaluates active-piece trajectory forecasting using T = 9 observed positions to predict H = 9 future positions at approximately 9 Hz, corresponding to roughly one second of history and prediction. The forecasting benchmark uses 26,208 training, 4,718 validation, and 7,513 test trajectory windows. We compare linear extrapolation with input ablations of the same TCN architecture. Additional inputs comprise Context, the other pieces’ positions; Cluster, assembly-cluster centroids; and Hand, MediaPipe hand keypoints encoded by a spatial-temporal graph convolutional network (ST-GCN) [26]. All learned variants outperform linear extrapolation. Combining piece history, context, and cluster centroids achieves the lowest errors and is selected for deployment.

TABLE II  
ACTIVE-PIECE TRAJECTORY FORECASTING BENCHMARK.
<table><tr><td>Method</td><td>Hand</td><td>Context</td><td>Cluster</td><td>ADE</td><td>FDE↓</td><td>Miss↓</td></tr><tr><td>Linear Extrap.</td><td></td><td></td><td></td><td>82.0</td><td>129.9</td><td>69.6</td></tr><tr><td>Proposed</td><td></td><td></td><td></td><td>32.6</td><td>44.6</td><td>25.2</td></tr><tr><td>Proposed</td><td>√</td><td></td><td></td><td>33.2</td><td>45.7</td><td>24.2</td></tr><tr><td>Proposed</td><td></td><td>√</td><td></td><td>31.7</td><td>44.8</td><td>25.2</td></tr><tr><td>Proposed</td><td></td><td></td><td>√</td><td>34.2</td><td>53.4</td><td>32.5</td></tr><tr><td>Proposed</td><td>√</td><td>√</td><td></td><td>33.1</td><td>45.2</td><td>24.8</td></tr><tr><td>Proposed</td><td>√</td><td></td><td>√</td><td>33.7</td><td>44.9</td><td>24.6</td></tr><tr><td>Proposed</td><td></td><td>√</td><td>√</td><td>31.6</td><td>43.0</td><td>22.9</td></tr><tr><td>Proposed</td><td>√</td><td>√</td><td>√</td><td>33.3</td><td>45.0</td><td>25.5</td></tr></table>

All variants use pose history; checkmarks indicate additional inputs. Errors are reported in pixels. ADE is the mean L2 error, FDE is the L2 error at the final frame, and Miss is the percentage of trajectories with FDE > 50.

TABLE III  
PLACEMENT-INTENT BENCHMARK.
<table><tr><td>Method</td><td>F1</td><td>Prec.</td><td>Recall</td></tr><tr><td>Rule-Based Intent Baseline</td><td>0.50</td><td>0.49</td><td>0.51</td></tr><tr><td>Distance Features</td><td>0.58</td><td>0.50</td><td>0.69</td></tr><tr><td>Forecast Features</td><td>0.63</td><td>0.52</td><td>0.79</td></tr><tr><td>Forecast + Distance</td><td>0.67</td><td>0.58</td><td>0.81</td></tr></table>

Third, Tab. III evaluates destination prediction using 14,626 training, 3,092 validation, and 5,478 test samples, with test performance scored over 641 interaction events. We compare to a rule-based baseline that predicts a destination when the forecast trajectory enters a subgraph mask, along with ablated versions of our learned module. For evaluation, each interaction event is assigned the first non-null destination whose probability reaches $\gamma .$ . If the threshold is not reached, we use the most likely class at the last frame the piece was still active. Predictions that retain the piece’s previous subgraph are treated as null. For our module, we use γ=0.55. We evaluate the impact of $\gamma$ in Sec. IV-C. Overall, combining both forecast and distance features achieves the highest $F _ { 1 }$ (0.67), supporting the configuration used in deployment.

## C. System-Level Mistake Anticipation

After validating the design of the individual modules, we now turn to evaluating the complete reasoning, forecasting, and intervention pipeline against other end-to-end learned alternatives. Only for evaluation, we use the 40 no-feedback and reactive sessions from the final user study as described in Sec. IV-A, and assess performance during events where a piece enters a subgraph to which it was not previously assigned. An event is incorrect when the resulting subgraph exceeds its prescribed capacity for that piece type (i.e., a mistake). Applying this rule to manual annotations yielded 498 events, of which 96 incorrect and 402 correct.

TABLE IV  
SYSTEM-LEVEL MISTAKE ANTICIPATION BENCHMARK
<table><tr><td rowspan="2">Method</td><td colspan="3">Seen shapes</td><td rowspan="2">Unseen shapes</td></tr><tr><td>Prec. Recall</td><td></td><td> $\overline { { F _ { 1 } } }$ </td></tr><tr><td>GRU [27]</td><td>0.26</td><td>0.61</td><td>0.36</td><td> $\overline { { F _ { 1 } } }$  0.33</td></tr><tr><td>Relational Transformer [28]</td><td>0.24</td><td>0.48</td><td>0.32</td><td>0.36</td></tr><tr><td>Frozen MOMENT [29]</td><td>0.24</td><td>0.57</td><td>0.34</td><td>0.37</td></tr><tr><td>Frozen V-JEPA 2† [30]</td><td>0.28</td><td>0.58</td><td>0.38</td><td>0.26</td></tr><tr><td>TRAFA (adapted) [6]</td><td>0.25</td><td>0.79</td><td>0.38</td><td>0.36</td></tr><tr><td>TanCRAFT</td><td>0.45</td><td>0.77</td><td>0.57</td><td>0.51</td></tr></table>

<sup>†</sup>Additional RGB input. Unseen shapes: leave-one-shape-out evaluation.

We compare against several learned end-to-end baselines: a GRU [27], a relational Transformer [28], and frozen MOMENT [29] and V-JEPA 2 [30] encoders with trainable prediction heads. We also adopt TRAFA [6] by replacing its next-block rule with a capacity check over fixed, priorderived subgraph regions. All models receive the same tracking stream, with V-JEPA 2 additionally receiving RGB clips. An anticipated mistake is evaluated as correct if it precedes the event by at least 0.44 s. Tab. IV shows that TanCRAFT achieves the highest precision and $F _ { 1 }$ , while TRAFA achieves slightly higher recall but substantially lower precision, indicating more false warnings.

We also report the threshold sensitivity in Fig. 4. Increasing the threshold trades recall for precision over the plotted range, with $\gamma { = } 0 . 5 5$ achieving the highest $F _ { 1 }$

Generalization to unseen shapes. The model used in our user study was trained on all shapes used. Here, we additionally assess system generalization to unseen shapes via a leaveone-shape-out protocol. For each fold, we retrain the models without the held-out shape’s pilot sessions, exclude that shape from validation, and evaluate on its no-feedback and reactive study sessions. Architectures and training hyperparameters remain fixed. For the held-out shape, the classifier’s one-hot shape indicator is unseen during training, so subgraph assignment relies primarily on shape-independent features (piece pose, type, and cluster membership). The held-out shape’s prescribed subgraph composition is provided for assembly reasoning and capacity checking. We pool prediction counts across folds and compare against an all-shape control evaluated on the same study sessions. As shown in Tab. IV, TanCRAFT retains the highest pooled unseen-shape $F _ { 1 }$ at 0.51, compared with 0.57 for its retrained all-shape control. These results support generalization to unseen target shapes within the tangram domain given their prescribed subgraph compositions.

## D. User study design

We now turn to validating TanCRAFT via a user study, comparing two TanCRAFT variants against two baselines:

• NO FEEDBACK does not assist the user.

![](images/70c5809160ad57187110f2c497b34328f7e1ca66237aaf008a6ccd27da42a8a9.jpg)  
Fig. 4. System-level precision, recall, and $F _ { 1 }$ across thresholds. The dashed line marks the operating point selected before the user study.

• REACTIVE FEEDBACK provides feedback after an incorrect placement is complete.

• PREDICTIVE FEEDBACK warns the user when the predicted invalid-placement confidence exceeds $\gamma = 0 . 5 5$

• CONTINUOUS FEEDBACK maps invalid-placement confidence to the loudness of a sustained tone, saturating at a confidence of $\gamma = 0 . 6 5$

Both PREDICTIVE and CONTINUOUS FEEDBACK use the full system, differing only in how the feedback is conveyed.

The user study comprises 80 sessions from 20 new participants recruited from university campus, disjoint from the pilot cohort, with 20 sessions per feedback condition. As a result, no user study sessions were used for model training or tuning. None were affiliated with the research presented in this paper, and all provided informed consent. Participants also wore a head-mounted camera that recorded an egocentric view for dataset release only; this stream was not used by the system. A monitor displayed the target silhouette and a color-coded reference panel indicating the semantic subgraph associated with each piece, without revealing the final piece placements as shown in Fig. 1.

After written instructions and one warm-up round on the house shape, each participant completed four main rounds on fox, swan, camel, and penguin. Each round ended upon completion and was capped at four minutes. Participants completed the NASA-TLX questionnaire [9] to assess their perceived workload immediately after each main round. Feedback order was counterbalanced with $\textbf { a 4 } \times 4$ Williams Latin square replicated five times, and mode-target pairing was balanced so that each (mode, shape) pair occurred five times. The warm-up mode was assigned by blocked randomization and differed from the first main-round mode. To reduce memorization and order effects, the initial piececolor arrangement and the color order in the reference panel were re-randomized at the start of every main round.

We evaluated the feedback conditions by their task success rate, number of mistakes, completion time, number of feedback events, and perceived workload. Our analysis focuses on the three primary policies: NO FEEDBACK, REACTIVE FEEDBACK, and PREDICTIVE FEEDBACK. CONTINUOUS FEEDBACK is included as an additional comparison to test whether continuously providing feedback improves or burdens the interaction. Because the metrics are counts (mistakes), binary (success), continuous (TLX), and censored (completion time, capped at 240 s), we analyze each with an appropriate mixed-effects model with modes as a fixed effect and participants as a random effect: Poisson for mistakes, binomial for success, linear for TLX, and a Cox proportionalhazards model treating capped trials as right-censored for completion time. We report planned contrasts via estimated marginal means with Bonferroni correction, using one-sided tests for our directional hypotheses.

TABLE V  
USER STUDY RESULTS ACROSS FEEDBACK MODES.
<table><tr><td>Mode</td><td># Mistakes↓</td><td>Overall NASA-TLX ↓</td><td>Success Rate ↑</td><td>Time (s) ↓</td><td># Feedback</td></tr><tr><td>NO FEEDBACK</td><td> $\overline { { 4 . 9 0 \pm 5 . 4 } }$ </td><td> $3 . 6 3 \pm 1 . 1 8$ </td><td>0.45</td><td> $\overline { { 1 9 3 . 8 \pm 7 0 . 9 } }$ </td><td> $\overline { { 0 . 0 0 \pm 0 . 0 0 } }$ </td></tr><tr><td>REACTIVE FEEDBACK</td><td> $1 . 9 5 \pm 2 . 4$ </td><td> $3 . 2 0 \pm 1 . 1 2$ </td><td>0.50</td><td> $1 7 2 . 9 \pm 7 8 . 4$ </td><td> $1 . 2 0 \pm 1 . 2$ </td></tr><tr><td>PREDICTIVE FEEDBACK</td><td> $0 . 9 0 \pm 1 . 4$ </td><td> $\mathbf { 3 . 0 5 \pm 1 . 0 9 }$ </td><td>0.70</td><td> ${ \bf 1 4 8 . 5 \pm 7 7 . 5 }$ </td><td> $2 . 7 5 \pm 1 . 8$ </td></tr><tr><td>CONTINUOUS FEEDBACK</td><td> ${ \bf 0 . 7 5 \pm 1 . 0 }$ </td><td> $3 . 3 2 \pm 1 . 0 9$ </td><td>0.70</td><td> $1 7 0 . 1 \pm 6 4 . 7$ </td><td> $1 . 8 5 \pm 1 . 7$ </td></tr></table>

Values are mean ± standard deviation except success rate, which is reported as a proportion. Overall NASA-TLX is the unweighted mean of the six NASA-TLX subscales, with perceived success scores reversed so that lower values indicate lower workload. One effort rating was missing, so Overal NASA-TLX is computed over $N = 1 9 ;$ all other metrics use $N = 2 0$

![](images/23adf90994f47e8fc88af917357cb023d37357fecc5f06359810a90b45970c52.jpg)  
Fig. 5. Within-subject comparison of feedback policies. Each panel shows participant-level outcomes for all four modes evaluated. Lines connect results for the same participants. Asterisks mark significant planned model-based contrasts (p < .05).

## E. User Study Results

We show the results of our user study in Tab. V, while Fig. 5 visualizes the participant-level distributions. Participants made $0 . 9 0 \pm 1 . 4$ mistakes with PREDICTIVE FEEDBACK, compared with $4 . 9 0 \pm 5 . 4$ mistakes with NO FEEDBACK and 1.95 ± 2.4 mistakes with REACTIVE FEED-BACK, resulting in a significant reduction relative to both NO FEEDBACK $( p ~ < ~ . 0 0 1 )$ and REACTIVE FEEDBACK $( p ~ = ~ . 0 3 6 )$ . Perceived workload followed a similar pattern. PREDICTIVE FEEDBACK produced the lowest workload score (3.05±1.09), compared to NO FEEDBACK (3.63±1.18) and REACTIVE FEEDBACK (3.20 ± 1.12), with the reduction relative to NO FEEDBACK significant $( p \ = \ . 0 1 5 )$ . While PREDICTIVE FEEDBACK does produce more warning events than REACTIVE FEEDBACK $( 2 . 7 5 \pm 1 . 8 \ \mathrm { v s . 1 . 2 0 \pm 1 . 2 ) }$ , the absolute number remained low.

The other performance metrics show the same favorable direction. Success rate increased from 0.45 with NO FEED-BACK and 0.50 with REACTIVE FEEDBACK to 0.70 with

PREDICTIVE FEEDBACK $( p \ = \ . 0 2 9 \ \mathrm { \ v s } .$ NO FEEDBACK). Completion time also decreased: PREDICTIVE FEEDBACK produced the shortest mean completion time $( 1 4 8 . 5 \pm 7 7 . 5 $ s), and the Cox model confirmed faster completion than NO FEEDBACK $( p = . 0 0 2 )$ . CONTINUOUS FEEDBACK achieved the lowest mean mistake count and matched PREDICTIVE FEEDBACK’s success rate, but had higher mean workload and longer mean completion time.

## V. DISCUSSION

A central challenge in non-sequential assembly is distinguishing valid exploration from an impending mistake without prescribing the user’s next action. TanCRAFT addresses this by evaluating predicted placements against the evolving assembly’s remaining capacities, allowing different construction orders while selectively warning about incompatible outcomes. The system-level benchmark shows stronger mistake anticipation than the evaluated baselines, including on unseen tangram shapes. The user study extends prior findings on proactive assistance to error warnings in non-sequential assembly, showing that selective predictive feedback reduces mistakes relative to no feedback and reactive warnings.

Moreover, CONTINUOUS FEEDBACK offers an alternative to binary predictive warnings by conveying invalidplacement confidence through a graded tone. In our study, it achieved similarly low mistake counts and the same success rate as predictive feedback, suggesting that graded feedback can also support error prevention. However, continuous feedback was associated with higher mean workload and longer mean completion time. Adaptive gain, confidence smoothing, or user-specific thresholds may help reduce this burden.

Although tangram is a controlled domain, it captures key challenges of non-sequential assembly: objective correctness, flexible construction order, and exploratory manipulation. Moreover, the leave-one-shape-out evaluation supports generalization to unseen tangram targets when their subgraph compositions are provided. However, this offline evaluation does not establish generalization to new piece inventories, workspaces, or assembly tasks. Extending the framework to more complex assemblies and robot-assisted manipulation remains future work.

Beyond system evaluation, we will release our multi-view tangram assembly dataset comprising 152 sessions and eight hours of activity, recorded synchronously from overhead and egocentric cameras (≈16 hours of video in total). The dataset includes tracked objects and hands, subgraph assignments, placements, mistakes, and feedback metadata. By capturing free-form manipulation and user responses to feedback, it can support future research on action forecasting, mistake anticipation, assistance timing, and reasoning tasks.

## VI. CONCLUSION

We presented TanCRAFT, a real-time predictive assistance framework for non-sequential assembly that tracks the scene, reasons about the partial assembly, forecasts active-piece motion, and selectively intervenes before incompatible placements. System-level comparisons show improved mistake anticipation over the evaluated learned baselines, including generalization to unseen tangram shapes. In a within-subject study with 20 participants, our system significantly reduced the number of mistakes, while achieving the lowest mean perceived workload and shortest mean completion time. More broadly, TanCRAFT provides a step toward cost-aware assistance in automation and human-robot workcells, where warnings should be both reliable and timely.

## REFERENCES

[1] A. Saptari, J. X. Leau, and N. A. Mohamad, “The effect of time pressure, working position, component bin position and gender on human error in manual assembly line,” in 2015 International Conference on Industrial Engineering and Operations Management (IEOM). IEEE, 2015, pp. 1–6.

[2] Y. Torres, S. Nadeau, and K. Landau, “Classification and quantification of human error in manufacturing: A case study in complex manual assembly,” Applied Sciences, vol. 11, no. 2, p. 749, 2021.

[3] D. Antonelli and D. Stadnicka, “Predicting and preventing mistakes in human-robot collaborative assembly,” IFAC-PapersOnLine, vol. 52, no. 13, pp. 743–748, 2019.

[4] E. C. Grigore, A. Roncone, et al., “Preference-based assistance prediction for human-robot collaboration tasks,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 4441–4448.

[5] H. Nemlekar, N. Dhanaraj, et al., “Transfer learning of human preferences for proactive robot assistance in assembly tasks,” in Proceedings of the 2023 ACM/IEEE International Conference on Human-Robot Interaction, 2023, pp. 575–583.

[6] S. Mokhtar, L. Doorenbos, et al., “Trafa: Anticipating user actions to reduce errors in procedural tasks with predictive feedback,” arxiv:2605.24526, 2026.

[7] C. Zhao, C. Jiang, et al., “Master rules from chaos: Learning to reason, plan, and interact from chaos for tangram assembly,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 11 503–11 509.

[8] S. Lee, D. Ekpo, et al., “Imagine, verify, execute: Memory-guided agentic exploration with vision-language models,” arXiv:2505.07815, 2025.

[9] S. G. Hart and L. E. Staveland, “Development of nasa-tlx (task load index): Results of empirical and theoretical research,” in Advances in psychology. Elsevier, 1988, vol. 52, pp. 139–183.

[10] K. P. Hawkins, N. Vo, et al., “Probabilistic human action prediction and wait-sensitive planning for responsive human-robot collaboration,” in 2013 13th IEEE-RAS International Conference on Humanoid Robots (Humanoids). IEEE, 2013, pp. 499–506.

[11] L. Johannsmeier and S. Haddadin, “A hierarchical human-robot interaction-planning framework for task allocation in collaborative industrial assembly processes,” IEEE Robotics and Automation Letters, vol. 2, no. 1, pp. 41–48, 2016.

[12] V. V. Unhelkar, P. A. Lasota, et al., “Human-aware robotic assistant for collaborative assembly: Integrating human motion prediction with planning in time,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2394–2401, 2018.

[13] Y. Cheng, L. Sun, et al., “Towards efficient human-robot collaboration with robust plan recognition and trajectory prediction,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2602–2609, 2020.

[14] L. Eversberg and J. Lambrecht, “Evaluating digital work instructions with augmented reality versus paper-based documents for manual, object-specific repair tasks in a case study with experienced workers,” The International Journal of Advanced Manufacturing Technology, vol. 127, no. 3, pp. 1859–1871, 2023.

[15] E. Marino, L. Barbieri, et al., “Assessing user performance in augmented reality assembly guidance for industry 4.0 operators,” Computers in Industry, vol. 157, p. 104085, 2024.

[16] B. P. Bailey and J. A. Konstan, “On the need for attention-aware systems: Measuring effects of interruption on task performance, error rate, and affective state,” Computers in human behavior, vol. 22, no. 4, pp. 685–708, 2006.

[17] P. D. Adamczyk and B. P. Bailey, “If not now, when? the effects of interruption at different moments within task execution,” in Proceedings of the SIGCHI conference on Human factors in computing systems, 2004, pp. 271–278.

[18] S. R. Dixon, C. D. Wickens, and J. S. McCarley, “On the independence of compliance and reliance: Are automation false alarms worse than misses?” Human factors, vol. 49, no. 4, pp. 564–572, 2007.

[19] S. Mannem, W. Macke, et al., “Exploring the cost of interruptions in human-robot teaming,” in 2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids). IEEE, 2023.

[20] G. LeMasurier, A. Gautam, et al., “Reactive or proactive? how robots should explain failures,” in Proceedings of the 2024 ACM/IEEE international conference on human-robot interaction, 2024.

[21] D. Yu, R. Desai, et al., “Optimizing the timing of intelligent suggestion in virtual reality,” in Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology, 2022, pp. 1–20.

[22] N. Carion, L. Gustafson, et al., “Sam 3: Segment anything with concepts,” arXiv:2511.16719, 2025.

[23] C. Lugaresi, J. Tang, et al., “Mediapipe: A framework for building perception pipelines,” arXiv:1906.08172, 2019.

[24] R. E. Kalman, “A new approach to linear filtering and prediction problems,” 1960.

[25] Y. A. Farha and J. Gall, “Ms-tcn: Multi-stage temporal convolutional network for action segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019.

[26] S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[27] K. Cho, B. Van Merrienboer, ¨ et al., “Learning phrase representations using rnn encoder–decoder for statistical machine translation,” in Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 2014, pp. 1724–1734.

[28] J. Lee, Y. Lee, et al., “Set transformer: A framework for attentionbased permutation-invariant neural networks,” in International conference on machine learning. PMLR, 2019, pp. 3744–3753.

[29] M. Goswami, K. Szafer, et al., “Moment: A family of open time-series foundation models,” arXiv preprint arXiv:2402.03885, 2024.

[30] M. Assran, A. Bardes, et al., “V-jepa 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.