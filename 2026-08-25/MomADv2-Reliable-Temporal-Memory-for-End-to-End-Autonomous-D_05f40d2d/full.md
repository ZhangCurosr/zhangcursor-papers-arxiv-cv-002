# MomADv2: Reliable Temporal Memory for End-to-End Autonomous Driving

Ziying Song<sup>1</sup>, Shengkai Zhang<sup>2</sup>, Lin Liu<sup>2</sup>, Peiliang Wu<sup>1</sup>, Lei Yang<sup>3∗</sup>,

Dongyang Xu<sup>4</sup>, Bin Sun<sup>5</sup>, Li Wang<sup>6∗</sup>, Shaoqing Xu<sup>7</sup>, Caiyan Jia<sup>2</sup>, Yadan Luo<sup>8</sup>

<sup>1</sup>School of Artificial Intelligence (School of Software), Yanshan University

<sup>2</sup>Beijing Jiaotong University

<sup>3</sup>Nanyang Technological University

<sup>4</sup>Tsinghua University

<sup>5</sup>China Automotive Technology and Research Center Co., Ltd. <sup>6</sup>School of Mechanical Engineering, Beijing Institute of Technology <sup>7</sup>University of Macau <sup>8</sup>The University of Queensland

## Abstract

Long-horizon planning is critical for safe autonomous driving in complex scenarios. Existing methods improve planning continuity with temporal memory, but such memory may become invalid and mislead decisions when the driving command changes. Thus, selectively leveraging useful history while suppressing command-inconsistent memory remains a key challenge. To address this issue, we propose MomADv2, a reliable state-space memory framework for long-horizon endto-end autonomous driving. At its core, MomADv2 introduces a Selective State-Space Planning Memory Query Module, which filters historical planning queries based on temporal continuity and command consistency, selects planning modes relevant to the current command, and models the evolution of planning intentions through a selective state-space mechanism. To further alleviate local trajectory deviations and error accumulation in long-horizon planning, we design a Flow-Matching Trajectory Residual Refiner. It learns a continuous residual correction field from the refined planning output to the expert trajectory, enabling fine-grained trajectory refinement while preserving the stability of anchor-based planning. Extensive experiments on closed-loop NAVSIM and Bench2Drive, as well as open-loop nuScenes, demonstrate that MomADv2 improves long-horizon planning consistency and reduces the average collision rate by 15.6% over MomAD under 6-second planning.

## Introduction

End-to-end autonomous driving (E2E-AD) has recently attracted increasing attention due to its ability to integrate perception, prediction, and planning into a unified learnable framework (Hu et al. 2023; Jiang et al. 2023). Instead of separately optimizing each driving component, recent methods directly learn ego planning from sensor observations, surrounding agents, map elements, and high-level navigation commands (Jia et al. 2023, 2025). With the development of query-based perception, motion prediction, sparse scene representation, world modeling, and anchor-based planning, end-to-end driving systems have achieved promising performance in complex urban scenarios (Sun et al. 2024; Song et al. 2025; Xing et al. 2025).

Long-horizon planning provides autonomous driving systems with a stronger ability to anticipate future behaviors, enabling them to proactively perceive potential risks in complex trafic interactions and generate smoother, more stable, and safer ego trajectories. A fundamental requirement of long-horizon planning is temporal memory (Hu et al. 2023; Zhang et al. 2025a; Jiang et al. 2023; Song et al. 2025; Su et al. 2026). Long-horizon planning requires temporally consistent planning states across consecutive frames. MomAD (Song et al. 2025) shows that preserving planning momentum through historical planning queries can reduce mode instability and trajectory jitter. Therefore, efective temporal memory is essential for stable ego trajectory generation.

Existing temporal E2E-AD methods exploit history through spatial-temporal modeling, structured planning queries, task memory, or state-space representations (Zhang et al. 2025a; Song et al. 2025; Jia et al. 2025; Su et al. 2026). While temporal memory improves planning continuity, it is not always reliable. Under dynamic scene changes, temporal discontinuities, or command shifts, historical information may become inconsistent with the current intention and mislead planning. Therefore, long-horizon planning should not blindly reuse history, but selectively retain reliable memory while suppressing invalid or command-inconsistent interference.

In this work, we propose MomADv2, a reliable statespace memory framework for long-horizon E2E-AD that selectively retains reliable history, rather than fully trusting historical information as MomAD does. Instead of blindly reusing historical planning states, MomADv2 introduces a Selective State-Space Planning Memory Query Module (SSM-Q) to filter historical planning queries based on temporal continuity and command consistency, select commandrelevant planning modes, and model the temporal evolution of planning intentions. To further mitigate local deviations and error accumulation, we design a Flow-Matching Trajectory Residual Refiner (FM-Ref), which learns a continuous residual correction field from the refined planning output to the expert trajectory, enabling fine-grained trajectory refinement while preserving the stability of anchor-based planning. Extensive experiments on NAVSIM, Bench2Drive, and nuScenes show that MomADv2 improves long-horizon planning consistency and trajectory accuracy, reducing the average 6-second collision rate by 15.6% over MomAD (Song et al. 2025). The main contributions of this work are summarized as follows:

![](images/41552701351fb1f724daa0d7186327ee4c95b275947dddfd2eed5eaa36755264.jpg)

![](images/908019d987a1a6e43c88761719cf795d2ba1b16b407b3e1883c1102265415338.jpg)  
Figure 1: Motivation of MomADv2. (a) Long-horizon planning requires consistent intentions across planning cycles. (b) Existing methods, represented by MomAD (Song et al. 2025; Jia et al. 2025; Zhang et al. 2025a), indiscriminately reuse history, introducing new interference. (c) MomADv2 selectively preserves reliable states and refines local trajectory deviations via flow matching. (d) This reliable memory inheritance enables more accurate and safer long-horizon planning.

• We propose MomADv2, a reliable state-space memory framework for long-horizon E2E-AD, which selectively leverages useful historical information while suppressing invalid memory interference.

• We design two core modules: the Selective State-Space Planning Memory Query Module for reliable historical query filtering and intention modeling, and the Flow-Matching Trajectory Residual Refiner for fine-grained trajectory correction.

• Extensive experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that MomADv2 efectively improves long-horizon planning consistency and trajectory accuracy.

## Related Work

## End-to-End Autonomous Driving

Existing studies have advanced E2E-AD through unified representations, eficient decoding, generative planning, VLA models, and world models. Representative methods include planning-oriented unified frameworks (Hu et al. 2023; Jiang et al. 2023, 2026), eficient planning decoders (Jia et al. 2023, 2025; Sun et al. 2024; Li et al. 2024), generative trajectory modeling (Liao et al. 2025; Xing et al. 2025; Liu et al. 2025b; Song et al. 2026b), reliable planning optimization (Shang et al. 2025; Zheng et al. 2026), large-model-based driving cognition (Zhou et al. 2025), and world-model-based longhorizon reasoning (Song et al. 2026a). However, existing Temporal E2E-AD methods overlook temporal memory reliability. When scenes or commands change, reused history may conflict with the current intention and cause negative interference. Thus, long-horizon planning should selectively retain reliable memory while suppressing invalid history.

## State Space Models for Autonomous Driving

State space models (SSMs) ofer an eficient sequence modeling paradigm for capturing long-range dependencies through latent dynamical systems. Representative models such as Mamba (Gu and Dao 2024), and Mamba-2 (Dao and Gu 2024) improve long-sequence modeling via structured dynamics, selective state transitions, and hardware-aware linear-complexity computation. Beyond language modeling, SSMs have been extended to visual representation learning (Zhu et al. 2024; Liu et al. 2024), occupancy prediction (Li et al. 2025a), and E2E-AD (Yuan et al. 2024; Su et al. 2026; Wang, Jiang, and Xu 2025; Lu et al. 2025). Existing Mamba-based E2E-AD methods mainly focus on efficient temporal modeling, while the reliability of historical memory in long-horizon planning remains underexplored.

## Methodology

## Overview of MomADv2

As shown in Figure 2, MomADv2 inputs multi-view camera images and first extracts sparse scene representations for surrounding agents and map elements. An anchor-based motion and planning decoder then produces multi-modal motion predictions and initial ego planning candidates. Built upon the anchor-based planner, MomADv2 introduces two modules: a Selective State-Space Memory Query Module that filters and enhances reliable historical planning queries, and a Flow-Matching Residual Refiner that corrects planned trajectories with query-conditioned residual fields.

![](images/1b4ca5835d8c1f2b9727634dff99694c1a10d26fe0b79d308abf1c890ea23943.jpg)  
Figure 2: Overview of MomADv2. Multi-view images are encoded into map and detection representations to generate initial ego planning queries. A Selective State-Space Planning Memory Query module retains valid historical queries based on scene, token, command, and bufer consistency. A Flow-Matching Trajectory Residual Refiner then predicts a conditional residual velocity field toward the expert trajectory, producing smooth, accurate, and temporally consistent long-horizon planning.

![](images/970f4ae71e6026f636e12c9bbe53b9860ba53197a4c07cca49aa88dfecaa1fe7.jpg)  
Figure 3: Flow Matching Supervision learns a queryconditioned residual velocity field that transforms the SSMenhanced planner trajectory toward the expert trajectory. During inference, Gated Residual Refinement integrates this field with a few Euler steps and uses a bounded residual gate to improve trajectory accuracy while preserving anchorbased planning stability.

## Selective State-Space Planning Memory Query

As shown in Figure 2, the proposed Selective State-Space Planning Memory Query Module (SSM-Q) enhances the current planning query by selectively retrieving reliable historical planning states. Instead of directly propagating historical predictions, we maintain a token-keyed memory bank that stores raw planning queries and baseline commandspecific candidate trajectories before temporal enhancement:

$$
\begin{array} { r } { \mathcal { M } _ { t } = \left\{ \chi _ { t - k } \mapsto \left( \mathbf { Q } _ { t - k } ^ { \mathrm { r a w } } , \mathbf { Y } _ { t - k } ^ { \mathrm { b a s e } , \mathrm { c m d } } , c _ { t - k } \right) \right\} _ { k = 1 } ^ { K } . } \end{array}\tag{1}
$$

Here, $\left( c _ { t - k } , \chi _ { t - k } \right)$ serves as the memory key, while $( \mathbf { Q } _ { t - k } ^ { \mathrm { r a w } } , \mathbf { Y } _ { t - k } ^ { \mathrm { b a s e , c m d } } )$ forms the associated value. Specifically, $c _ { t - k }$ denotes the high-level driving command, $\mathrm { e . g . }$ $\{ \mathtt { l e f t }$ , straight, right}, and $\chi _ { t - k }$ is the temporalcontinuity token. For each command, the planner maintains $M _ { c }$ candidate trajectories, each paired with a corresponding raw planning query. For example, when the current command is left, only the $M _ { c }$ left-turn candidates and their paired queries from a temporally continuous historical frame are eligible for retrieval, while candidates associated with straight or right are excluded. Storing pre-enhancement queries and baseline trajectories prevents SSM-corrected outputs from being recursively written back into memory, thereby avoiding self-amplified temporal drift. Reliable Historical Query Retrieval. Historical planning states are used only when they are command-consistent, temporally continuous, and available in memory:

$$
m _ { t , k } = \mathbb { I } \left[ c _ { t } = c _ { t - k } , ~ \chi _ { t } ^ { ( - k ) } = \chi _ { t - k } , ~ \mathbf { Q } _ { t - k } ^ { \mathrm { r a w } } \neq \mathbf { 0 } \right] ,\tag{2}
$$

where $m _ { t , k }$ denotes the validity indicator for the historical state at time $t - k$ . Since planning candidates are organized according to high-level driving commands, historical retrieval is restricted to the candidate subset associated with the current command. However, the ordering of candidate trajectories may vary across frames due to changes in scene context and planning ambiguity. Therefore, for each current command-specific planning candidate $j ,$ we identify the most trajectory-consistent historical candidate under the same command:

$$
h _ { t , k , j } ^ { * } = \arg \operatorname* { m i n } _ { h \in \{ 1 , \dots , M _ { c } \} } D _ { \mathrm { s h i f t } } \left( \mathbf { Y } _ { t - k , h } ^ { \mathrm { b a s e , c m d } } , \mathbf { Y } _ { t , j } ^ { \mathrm { b a s e , c m d } } \right)\tag{3}
$$

The corresponding historical query is retrieved as

$$
\widehat { \mathbf { Q } } _ { t , k , j } ^ { \mathrm { h i s t } } = \mathbf { Q } _ { t - k } ^ { \mathrm { r a w } } \left[ \mathrm { i d x } \left( c _ { t } , h _ { t , k , j } ^ { * } \right) \right] ,\tag{4}
$$

where $\operatorname { i d x } ( c , h ) = c M _ { c } + h$ maps the command-specific candidate index to the global candidate index. Here, $D _ { \mathrm { s h i f t } } ( \cdot , \cdot )$ denotes a temporally shifted trajectory distance that compensates for frame-to-frame trajectory ofsets. This trajectorylevel alignment enables each current planning candidate to retrieve the most relevant historical query, rather than relying on a fixed candidate index across time.

Reliability-aware Temporal Modeling. The aligned historical queries are aggregated according to validity, trajectory similarity, and temporal decay:

$$
\bar { \mathbf { Q } } _ { t , j } ^ { \mathrm { h i s t } } = \sum _ { k = 1 } ^ { K } w _ { t , k , j } \widehat { \mathbf { Q } } _ { t , k , j } ^ { \mathrm { h i s t } } .\tag{5}
$$

The aggregation weight is computed as

$$
w _ { t , k , j } = \frac { \phi _ { t , k , j } } { \sum _ { k ^ { \prime } = 1 } ^ { K } \phi _ { t , k ^ { \prime } , j } + \epsilon } ,\tag{6}
$$

where

$$
\phi _ { t , k , j } = m _ { t , k } \exp \left( - \frac { d _ { t , k , j } } { \alpha } \right) \exp \left( - \frac { k } { \beta } \right) .\tag{7}
$$

Here, $d _ { t , k , j }$ is the aligned trajectory distance, while α and $\beta$ control the sensitivity to trajectory similarity and temporal decay, respectively.

We further compute a reliability gate to prevent unreliable historical states from perturbing the current planning query:

$$
\boldsymbol { r } _ { t , j } = \boldsymbol { r } _ { t } ^ { \mathrm { c n t } } \cdot \boldsymbol { r } _ { t , j } ^ { \mathrm { d i s t } } ,\tag{8}
$$

where

$$
r _ { t } ^ { \mathrm { c n t } } = \operatorname* { m i n } \left( \frac { \sum _ { k = 1 } ^ { K } m _ { t , k } } { K _ { 0 } } , 1 \right) ,\tag{9}
$$

and

$$
r _ { t , j } ^ { \mathrm { d i s t } } = \sigma \left( \frac { d _ { 0 } - \operatorname* { m i n } _ { k } { d _ { t , k , j } } } { \gamma _ { d } } \right) .\tag{10}
$$

Here, $K _ { 0 }$ denotes the minimum number of valid historical states required for reliable temporal modeling, $d _ { 0 }$ is the distance threshold, and $\gamma _ { d }$ controls the sharpness ofthe distancebased reliability gate. Thus, historical memory is activated only when suficient valid and trajectory-consistent historical information is available.

For each command-specific planning candidate, the aligned historical queries and the current query are concatenated into a temporal sequence:

$$
\begin{array} { r } { \mathbf { Z } _ { t , j } ^ { 0 } = \operatorname { C o n c a t } \left( \widehat { \mathbf { Q } } _ { t , K , j } ^ { \operatorname { h i s t } } , \ldots , \widehat { \mathbf { Q } } _ { t , 1 , j } ^ { \operatorname { h i s t } } , \mathbf { Q } _ { t , j } \right) . } \end{array}\tag{11}
$$

The sequence is processed by a selective state-space query encoder composed of layer normalization, causal depthwise convolution, input-dependent selective scan, SiLU gating, and residual connection:

$$
\mathbf { H } _ { t , j } ^ { \mathrm { c u r } } = \mathrm { S S M E n c o d e r } \left( \mathbf { Z } _ { t , j } ^ { 0 } \right) [ - 1 ] .\tag{12}
$$

The encoder models the temporal transition from historical planning intentions to the current planning intention while preserving candidate-wise planning semantics.

Reliability-gated Residual Injection. The SSM-induced residual direction is computed as

$$
\begin{array} { r } { \Delta \mathbf { Q } _ { t , j } = \operatorname { t a n h } \left( \mathbf { H } _ { t , j } ^ { \operatorname { c u r } } - \mathbf { Q } _ { t , j } \right) . } \end{array}\tag{13}
$$

To avoid excessive perturbation to the original planner, we apply a reliability-gated and norm-constrained residual update:

$$
\mathbf { U } _ { t , j } = \mathrm { C l i p } \left( g _ { t , j } \Delta \mathbf { Q } _ { t , j } , \mathbf { \nabla } \rho \left\| \mathbf { Q } _ { t , j } \right\| _ { 2 } \right) ,\tag{14}
$$

where $g _ { t , j }$ combines the valid-history indicator, the global residual gate, the reliability gate, the candidate-wise gate, and the residual scale. The coeficient $\rho$ limits the maximum magnitude of query perturbation. The enhanced planning query is obtained by

$$
\begin{array} { r } { \widetilde { \mathbf { Q } } _ { t , j } = \mathbf { Q } _ { t , j } + \mathbf { U } _ { t , j } . } \end{array}\tag{15}
$$

Finally, the enhanced planning query is fed into the planning refinement head:

$$
\{ \boldsymbol { \pi } _ { t } ^ { \mathrm { s s m } } , \mathbf { Y } _ { t } ^ { \mathrm { s s m } } , \mathbf { s } _ { t } ^ { \mathrm { s s m } } \} = f _ { \mathrm { r e f } } \left( \widetilde { \mathbf { Q } } _ { t } \right) ,\tag{16}
$$

where $\pi _ { t } ^ { \mathrm { s s m } } , \mathbf { Y } _ { t } ^ { \mathrm { s s m } }$ , and $\mathbf { s } _ { t } ^ { \mathrm { s s m } }$ denote the planning classifica tion, trajectory regression, and planning status, respectively.

## Flow-Matching Trajectory Residual Refiner

We propose a Flow-Matching Trajectory Residual Refiner (FM-Ref) to improve ego trajectory accuracy and smoothness, as shown in Figure 3. Instead of generating trajectories from random noise as difusion methods do, our refiner starts from planner-refined candidates and learns a continuous residual correction field toward the expert trajectory. This residual refinement preserves the stability of the anchorbased planner while enhancing local trajectory quality.

Conditional Flow Field. Let ${ \bf Y } _ { t } ^ { \mathrm { r e f } } \in \breve { \mathbb { R } } ^ { B \times \mathbf { \bar { 1 } } \times \mathbf { \bar { N } } \times T \times 2 }$ denote the planner-refined candidate trajectories in displacement form, where $B$ is the batch size, N is the number of planning candidate trajectories maintained by the planning head, and $T$ is the number of future timesteps. The trajectory ${ \bf Y } _ { t } ^ { \mathrm { r e f } }$ is produced by the SSM-enhanced planning branch. Since the planning head predicts trajectory ofsets, we first convert the displacement-form trajectories into the absolute trajectory space by cumulative summation:

$$
\overline { { \mathbf { Y } } } _ { t } ^ { \mathrm { r e f } } = \mathrm { C u m s u m } \left( \mathbf { Y } _ { t } ^ { \mathrm { r e f } } \right) ,\tag{17}
$$

where Cumsum(·) is performed along the temporal dimension. The normalized initial trajectory state is then defined as

$$
\mathbf { x } _ { 0 } = \frac { \overline { { \mathbf { Y } } } _ { t } ^ { \mathrm { r e f } } } { s } ,\tag{18}
$$

where s is a trajectory scale factor. During the optimization of the flow branch, the planner-refined trajectory and the enhanced planning query are treated with stop-gradient to prevent the auxiliary flow objective from destabilizing the planner.

Given the enhanced planning query $\widetilde { \mathbf { Q } } _ { t }$ as the conditional planning context, an intermediate trajectory state $\mathbf { x } _ { \tau }$ as the variable to be refined, and a continuous flow time $\tau \in [ 0 , 1 ]$ we introduce a Conditional Velocity Field to predict a timedependent residual velocity field. Specifically, the estimator first embeds the enhanced planning query, the trajectory state, and the flow time:

$$
\mathbf { z } _ { \tau } = \mathrm { C o n c a t } \left( \phi _ { q } \left( \widetilde { \mathbf { Q } } _ { t } \right) , \phi _ { x } \left( \mathbf { x } _ { \tau } \right) , \phi _ { \tau } \left( \tau \right) \right) ,\tag{19}
$$

where $\phi _ { q } ( \cdot ) , \phi _ { x } ( \cdot )$ , and $\phi _ { \tau } ( \cdot )$ denote the query projection, trajectory encoder, and time embedding function, respectively. The fused representation is processed by an MLP with LayerNorm and a Transformer encoder:

$$
\mathbf { h } _ { \tau } = \mathrm { T r a n s E n c } \left( \mathrm { L N } \left( \mathrm { M L P } \left( \mathbf { z } _ { \tau } \right) \right) \right) .\tag{20}
$$

Finally, an MLP-based velocity head predicts the residual velocity field:

$$
\begin{array} { r } { \mathbf { v } _ { \theta } = f _ { \theta } \left( \widetilde { \mathbf { Q } } _ { t } , \mathbf { x } _ { \tau } , \tau \right) = \mathrm { M L P } _ { \mathrm { v e l } } \left( \mathbf { h } _ { \tau } \right) . } \end{array}\tag{21}
$$

This estimator conditions the residual correction field on both the planning intention encoded by $\dot { \mathbf { Q } } _ { t }$ and the geometric state of the current trajectory.

Flow Matching Supervision. During training, the expert ego trajectory $\mathbf { Y } _ { t } ^ { \star }$ is also converted into the absolute trajectory space and normalized:

$$
\overline { { \mathbf { Y } } } _ { t } ^ { \star } = \mathrm { C u m s u m } \left( \mathbf { Y } _ { t } ^ { \star } \right) ,\tag{22}
$$

$$
\mathbf { x } _ { 1 } = \frac { \overline { { \mathbf { Y } } } _ { t } ^ { \star } } { s } .\tag{23}
$$

Here, $\mathbf { Y } _ { t } ^ { \star }$ denotes the expert trajectory in displacement form. In supervised training, the loss is computed on the planning candidate selected by the planning sampler, while the candidate index is omitted for notational simplicity.

We sample $\tau \sim \mathcal { U } ( 0 , 1 )$ and construct an intermediate trajectory state along the linear path from the planner-refined trajectory to the expert trajectory:

$$
\mathbf { x } _ { \tau } = ( 1 - \tau ) \mathbf { x } _ { 0 } + \tau \mathbf { x } _ { 1 } .\tag{24}
$$

The target velocity is defined as the residual displacement between the two endpoints in the normalized absolute trajectory space:

$$
\mathbf { v } ^ { \star } = \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } .\tag{25}
$$

The flow-matching objective supervises the predicted residual velocity field:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } = \left\| \int _ { \boldsymbol { \theta } } \left( \widetilde { \mathbf { Q } } _ { t } , \mathbf { x } _ { \tau } , \tau \right) - \mathbf { v } ^ { \star } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{26}
$$

Through this objective, the velocity estimator learns a queryconditioned residual transport field that moves the plannerrefined trajectory toward the expert trajectory in the normalized absolute trajectory space.

Gated Residual Inference. During inference, no expert trajectory is available. Therefore, we start from the normalized planner-refined trajectory $\mathbf { x } _ { \mathrm { 0 } }$ and progressively update it by Euler integration:

$$
{ \bf x } _ { k + 1 } = { \bf x } _ { k } + \frac { 1 } { S } f _ { \theta } \left( \widetilde { \bf Q } _ { t } , { \bf x } _ { k } , \tau _ { k } \right) ,\tag{27}
$$

where $\tau _ { k } \ = \ ( k + 0 . 5 ) / S$ , and S denotes the number of integration steps. In our implementation, we set $S = 2$ for eficient inference.

To preserve the stability of the anchor-based planner and avoid overly aggressive corrections, we introduce a learnable bounded residual gate:

$$
\eta = \eta _ { \mathrm { m i n } } + \eta _ { \mathrm { m a x } } \sigma \left( g _ { \mathrm { f l o w } } \right) ,\tag{28}
$$

where $\eta _ { \mathrm { m i n } }$ provides a small base correction ratio and η<sub>max</sub> bounds the maximum residual strength. The final flowrefined trajectory in the absolute trajectory space is obtained by applying the gated residual correction to the initial planner-refined trajectory:

$$
\overline { { \mathbf { Y } } } _ { t } ^ { \mathrm { f l o w } } = s \left[ \mathbf { x } _ { 0 } + \eta \left( \mathbf { x } _ { S } - \mathbf { x } _ { 0 } \right) \right] .\tag{29}
$$

When required by the planning decoder, the absolute-form trajectory is converted back into the displacement form:

$$
\mathbf { Y } _ { t } ^ { \mathrm { f l o w } } = \mathrm { D i f f } \left( \overline { { \mathbf { Y } } } _ { t } ^ { \mathrm { f l o w } } \right) ,\tag{30}
$$

where $\mathrm { D i f f } \left( { \cdot } \right)$ denotes the inverse operation of cumulative summation along the temporal dimension. Therefore, the proposed module performs gated residual refinement rather than directly replacing the planner output, which maintains the stability of anchor-based planning while improving local trajectory accuracy and smoothness.

Training Objectives. During training, the planner branch is supervised by the standard planning objective and the temporal refinement losses introduced by the SSM-enhanced query module. In addition to the flow-matching objective, the flowrefined trajectory is further optimized with a regression loss:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \ell _ { \mathrm { r e g } } \left( \mathbf { Y } _ { t } ^ { \mathrm { f l o w } } , \mathbf { Y } _ { t } ^ { \star } \right) .\tag{31}
$$

In practice, the flow output can be supervised in both displacement and cumulative trajectory spaces:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \lambda _ { \Delta } \ell _ { \mathrm { r e g } } \left( \mathbf { Y } _ { t } ^ { \mathrm { f l o w } } , \mathbf { Y } _ { t } ^ { \star } \right) + \lambda _ { \mathrm { c u m } } \ell _ { \mathrm { r e g } } \left( \overline { { \mathbf { Y } } } _ { t } ^ { \mathrm { f l o w } } , \overline { { \mathbf { Y } } } _ { t } ^ { \star } \right) ,\tag{32}
$$

where $\lambda _ { \Delta }$ and $\lambda _ { \mathrm { c u m } }$ balance displacement-form and cumulative-form supervision.

The overall objective for the planning refinement stage is formulated as

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { p l a n } } + \mathcal { L } _ { \mathrm { S S M } } + \lambda _ { \mathrm { f l o w } } \mathcal { L } _ { \mathrm { f l o w } } + \lambda _ { \mathrm { F M } } \mathcal { L } _ { \mathrm { F M } } ,\tag{33}
$$

where ${ \mathcal { L } } _ { \mathrm { p l a n } }$ denotes the standard planning loss, $\mathcal { L } _ { \mathrm { S S M } }$ denotes the temporal trajectory refinement losses associated with the SSM-enhanced planning branch, and $\lambda _ { \mathrm { H o w } }$ and $\lambda _ { \mathrm { F M } }$ are balancing weights for the flow-refined regression loss and the flow-matching objective, respectively. During inference, $\mathbf { Y } _ { t } ^ { \mathrm { { f i o w } } }$ is used as the final ego planning trajectory.

## Experiments

## Datasets and Metrics

Open-Loop. Our MomADv2 is evaluated on nuScenes dataset. The nuScenes (Caesar et al. 2020) (NuS) dataset consists of 1,000 driving scenes, each providing six synchronized camera images and LiDAR point clouds with a $3 6 0 ^ { \circ }$ field of view. In our experiments, we only use multi-view image data as model inputs. For nuScenes, we report the standard $L _ { 2 }$ displacement error and Collision Rate to jointly evaluate long-horizon planning accuracy and safety.

Closed-Loop. We evaluate MomADv2 on Bench2Drive and NAVSIM. Bench2Drive (Jia et al. 2024) is a closed-loop evaluation benchmark built upon CARLA Leaderboard 2.0 for E2E-AD. It provides an oficial training set, from which we use the base split (1,000 clips) to ensure fair comparison with prior methods, and an evaluation set consisting of 220 predefined routes. Following the oficial protocol, we report Driving Score (DS) and Success Rate (SR). NAVSIMv1/2 (Dauner et al. 2024; Cao et al. 2025) is a planning benchmark derived from OpenScene, featuring multiview camera and LiDAR data for 360<sup>◦</sup> perception, with annotations at 2 Hz including HD maps and object bounding boxes. It adopts non-reactive simulation with closed-loop evaluation for comprehensive planning assessment.

Table 1: Planning results for 6-second long-horizon planning on the nuScenes validation set.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="7">L2 (m) ↓</td><td colspan="7">Col. Rate (%) ↓</td></tr><tr><td></td><td></td><td>3s</td><td>4s</td><td>5s</td><td>6s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>4s</td><td>5s</td><td>6s</td><td>Avg.</td></tr><tr><td>UniAD (Hu et al. 2023)</td><td>CVPR&#x27;23</td><td>0.47</td><td>0.91</td><td>1.35</td><td>1.91</td><td>2.47</td><td>3.07</td><td>1.70</td><td>0.25</td><td>0.36</td><td>0.61</td><td>0.99</td><td>1.64</td><td>2.51</td><td>1.06</td></tr><tr><td>SparseDrive (Sun et al. 2024)</td><td>CVPR&#x27;23</td><td>0.43</td><td>0.87</td><td>1.23</td><td>1.75</td><td>2.32</td><td>2.95</td><td>1.59</td><td>0.19</td><td>0.31</td><td>0.56</td><td>0.87</td><td>1.54</td><td>2.33</td><td>0.97</td></tr><tr><td>MomAD (Song et al. 2025)</td><td>CVPR&#x27;25</td><td>0.41</td><td>0.85</td><td>1.13</td><td>1.67</td><td>1.98</td><td>2.45</td><td>1.42</td><td>0.17</td><td>0.30</td><td>0.54</td><td>0.83</td><td>1.43</td><td>2.13</td><td>0.90</td></tr><tr><td>LAW (Li et al. 2025c)</td><td>ICLR&#x27;25</td><td>0.40</td><td>0.87</td><td>1.16</td><td>1.71</td><td>2.03</td><td>2.61</td><td>1.46</td><td>0.19</td><td>0.33</td><td>0.57</td><td>0.86</td><td>1.51</td><td>2.31</td><td>0.96</td></tr><tr><td>Epona (Zhang et al. 2025b)</td><td>ICCV&#x27;25</td><td>0.39</td><td>0.91</td><td>1.17</td><td>1.73</td><td>2.02</td><td></td><td>1.50</td><td>0.14</td><td>0.18</td><td>0.45</td><td>0.74</td><td>1.48</td><td>2.23</td><td>0.87</td></tr><tr><td>DIVER (Song et al. 2026b)</td><td>TPAMI&#x27;26</td><td>0.38</td><td>0.75</td><td>1.10</td><td>1.53</td><td>1.98</td><td>2.15</td><td>1.37</td><td>0.13</td><td>0.31</td><td>0.44</td><td>0.80</td><td>1.41</td><td>2.11</td><td>0.87</td></tr><tr><td>GuideFlow (Liu et al. 2025b)</td><td>CVPR&#x27;26</td><td>0.42</td><td>0.83</td><td>1.21</td><td>1.73</td><td>2.05</td><td>2.63</td><td>1.48</td><td>0.12</td><td>0.22</td><td>0.42</td><td>0.79</td><td>1.44</td><td>2.15</td><td>0.86</td></tr><tr><td>MomADv2(Ours)</td><td></td><td>0.28</td><td>0.54</td><td>0.89</td><td>1.32</td><td>1.82</td><td>2.40</td><td>1.21</td><td>0.00</td><td>0.12</td><td>0.33</td><td>0.72</td><td>1.33</td><td>2.03</td><td>0.76</td></tr></table>

Table 2: Open-Loop and Closed-Loop results on Bench2Drive (V0.0.3) under base training set. \* denotes expert feature distillation. ‘DS’ denotes Driving Score, ‘SR’ denotes Success Rate, ‘Efi’ denotes Eficiency, and ‘Comf denotes Comfortness.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td>Open-loop</td><td colspan="4">Closed-loop</td></tr><tr><td>Avg. L2</td><td>DS ↑</td><td>SR (%) ↑</td><td>Effi</td><td>Comf ↑</td></tr><tr><td>DriveAdapter* (Jia et al. 2023)</td><td>ICCV&#x27;23</td><td>1.01</td><td>64.22</td><td>33.08</td><td>70.22</td><td>16.01</td></tr><tr><td>DriveDPÓ (Shang et al. 2025)</td><td>NeurIPS&#x27;25</td><td></td><td>62.02</td><td>30.62</td><td></td><td></td></tr><tr><td>Raw2Drive (Yang et al. 2025)</td><td>NeurIPS&#x27;25</td><td></td><td>71.36</td><td>50.24</td><td></td><td></td></tr><tr><td>DriveTrans* (Jia et al. 2025)</td><td>ICLR&#x27;25</td><td>0.62</td><td>63.46</td><td>35.01</td><td>100.64</td><td>20.78</td></tr><tr><td>WoTE* (Li et al. 2025e)</td><td>ICCV&#x27;25</td><td></td><td>61.71</td><td>31.36</td><td></td><td></td></tr><tr><td>DIVER* (Song et al. 2026b)</td><td>TPAMI&#x27;26</td><td>1.11</td><td>68.90</td><td>36.75</td><td>72.34</td><td>22.34</td></tr><tr><td>MomADv2 (Ours) 水</td><td></td><td>0.77</td><td>78.82</td><td>46.50</td><td>83.41</td><td>31.23</td></tr><tr><td>UniAD-Base (Hu et al. 2023)</td><td>CVPR&#x27;23</td><td>0.73</td><td>45.81</td><td>16.36</td><td>129.21</td><td>43.58</td></tr><tr><td>VAD (Jiang et al. 2023)</td><td>ICCV&#x27;23</td><td>0.91</td><td>42.35</td><td>15.00</td><td>157.94</td><td>46.01</td></tr><tr><td>GenAD (Zheng et al. 2024b)</td><td>ECCV&#x27;24</td><td></td><td>44.81</td><td>15.90</td><td></td><td></td></tr><tr><td>MomAD (Song et al. 2025)</td><td>CVPR&#x27;25</td><td>0.82</td><td>47.91</td><td>18.11</td><td>174.91</td><td>51.20</td></tr><tr><td>DIVER (Song et al. 2026b)</td><td>TPAMI&#x27;26</td><td>1.13</td><td>47.95</td><td>19.47</td><td>164.66</td><td>51.28</td></tr><tr><td>MomADv2 (Ours)</td><td></td><td>0.76</td><td>52.32</td><td>24.24</td><td>169.09</td><td>55.11</td></tr></table>

Table 3: Comparison on planning-oriented NAVSIMv1 navtest split with Closed-Loop metrics.
<table><tr><td>Method</td><td>Venue</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>Comf.↑</td><td>EP↑</td><td>PDMS↑</td></tr><tr><td>TransFuser (Chitta et al. 2023)</td><td>TPAMI&#x27;22</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>VADv2 (Jiang et al. 2026)</td><td>ICLR&#x27;26</td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>GoalFlow (Xing et al. 2025)</td><td>CVPR&#x27;25</td><td>98.3</td><td>93.8</td><td>94.3</td><td>100</td><td>79.8</td><td>85.7</td></tr><tr><td>Hydra-MDP (Li et al. 2024)</td><td>Arxiv&#x27;24</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>FUMP (Liu et al. 2025a)</td><td>Arxiv&#x27;25</td><td>98.1</td><td>96.2</td><td>94.2</td><td>100</td><td>82.0</td><td>87.8</td></tr><tr><td>DiffusionDrive (Liao et al. 2025)</td><td>CVPR&#x27;25</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>DIVER (Song et al. 2026b)</td><td>TPAMI&#x27;26</td><td>98.5</td><td>96.5</td><td>94.9</td><td>100</td><td>82.6</td><td>88.3</td></tr><tr><td>DriveSuprim (Yao et al. 2026)</td><td>AAAI&#x27;26</td><td>97.8</td><td>97.3</td><td>93.6</td><td>100</td><td>86.7</td><td>89.9</td></tr><tr><td>MomAv2 (Òurs)</td><td></td><td>99.0</td><td>97.1</td><td>95.5</td><td>100</td><td>83.2</td><td>89.9</td></tr></table>

## Implementation Details

We compare MomADv2 with representative end-to-end autonomous driving baselines, including MomAD (Song et al. 2025) and TransFuser (Chitta et al. 2023). MomAD is used as the baseline on Bench2Drive, nuScenes, and Adv-nuSc, while TransFuser is adopted on NAVSIM following the official Navtrain split. For nuScenes and Adv-nuSc, we use a ResNet-50 (He et al. 2016) backbone with an input resolution of 256 × 704, a 55 m detection range, a 60 × 30 m mapping range, and 6 motion trajectory modes. For Bench2Drive, we use a ResNet-50 backbone with 6 decoder layers, 640 × 352 input resolution, 900 agent queries, 100 map queries, and 480 planning queries. For NAVSIM, we follow TransFuser with the same perception modules and ResNet-34 backbone for fair comparison. All experiments are conducted on 8 NVIDIA RTX 4090 GPUs. The batch size is 48 for nuScenes and Bench2Drive and 64 for NAVSIM. Models are trained for 20 epochs on nuScenes, 2 epochs on Bench2Drive, and

Table 4: Comparison with SOTA methods on the NAVSIMv2 navhard split (Cao et al. 2025).
<table><tr><td>Method</td><td>Stage NC↑ DAC↑ DDC↑ TL↑ EP↑ TTC↑LK↑ HC↑ EC↑ EPDMS↑</td></tr><tr><td>TransFuser</td><td>Stage1 96.2 79.5 99.1 99.5 84.1 95.1 94.2 97.5 79.1 23.1 Stage2 77.7 70.2 84.2 98.085.1 75.6 45.4 95.7 75.9</td></tr><tr><td></td></tr><tr><td>Stage1 96.0 79.7 97.4 99.581.3 93.1 90.8 96.8 73.8 24.2</td></tr><tr><td>DiffusionDrive 98.7 85.1 Stage2 82.1 72.2 88.5 78.8 49.2 89.3 71.2 Stage1 96.6 80.5 96.3 99.3 82.3 94.9 91.5 97.7 67.8</td></tr></table>

Table 5: Comparison with SOTA methods on the NAVSIMv2 navtest split (Cao et al. 2025).
<table><tr><td>Method</td><td>NC↑DAC↑DDC↑TL↑EP↑TTC↑LK↑HC↑EC↑EPDMS↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TransFuser (Chitta et al. 2023)</td><td></td><td>96.9 89.9</td><td></td><td></td><td></td><td>97.8 99.7 87.1 95.4 92.7 98.3 87.2</td><td>76.7</td></tr><tr><td>DiffusionDrive (Liao et al. 2025)</td><td></td><td>98.2 95.9</td><td></td><td></td><td></td><td>99.4 99.887.5 97.3 96.8 98.3 87.7</td><td>84.5</td></tr><tr><td>Hydra-MDP++ (Li et al. 2025b)</td><td></td><td>97.2 97.5</td><td></td><td></td><td></td><td>99.4 99.6 83.1 96.5 94.4 98.2 70.9</td><td>81.4</td></tr><tr><td>DriveSuprim (Yao et al. 2026)</td><td></td><td>97.5 96.5</td><td></td><td></td><td></td><td>99.4 99.688.4 96.6 95.5 98.3 77.0</td><td>83.1</td></tr><tr><td>DiffusionDriveV2 (Zou et al. 2025) 97.7 96.6</td><td></td><td></td><td></td><td></td><td></td><td>99.299.888.997.2 96.0 97.8 91.0</td><td>87.5</td></tr><tr><td>DriveWorld-VLA (Liu et al. 2026)</td><td></td><td>98.6 99.1</td><td></td><td></td><td></td><td>99.699.887.4 97.9 97.0 97.8 78.6</td><td>86.8</td></tr><tr><td>DriveVLA-W0 (Li et al. 2025d)</td><td></td><td>98.5 99.1</td><td></td><td></td><td></td><td>98.0 99.786.4 98.1 93.2 97.9 58.9</td><td>86.1</td></tr><tr><td>Recogdrive (Li et al. 2025f)</td><td></td><td>98.3 95.2</td><td></td><td></td><td></td><td>98.3 99.887.1 97.5 96.6 99.5 86.5</td><td>83.6</td></tr><tr><td>MomADv2 (Ours)</td><td></td><td>98.3 97.4</td><td></td><td></td><td></td><td>99.7 99.3 89.6 97.8 95.5 98.6 91.7</td><td>87.9</td></tr></table>

100 epochs on NAVSIM, taking approximately 10.3, 49.1, and 3.0 hours, respectively.

## Main Results

Long-Horizon Planning on nuScenes (Open-Loop). Table 1 reports open-loop 6-second planning results on the nuScenes validation set. Existing E2E-AD methods sufer from increasing errors and collision rates as the horizon extends. In contrast, MomADv2 achieves the best performance across all timestamps, with the lowest average L error of 1.21 m and collision rate of 0.76%. Compared with the strongest prior method, it improves average L<sub>2</sub> error by 11.7% and collision rate by 11.6%, demonstrating the efectiveness of selective state-space memory and flow-matching residual refinement.

Bench2Drive (Closed-Loop). Table 2 reports Bench2Drive results. MomADv2 achieves strong closed-loop performance, reaching 78.82 DS and the best Comfortness of31.23. It also improves MomAD from 47.91 to 52.32 DS and from 18.11% to 24.24% SR, demonstrating more reliable driving via efective temporal memory modeling.

NAVSIMv1 navtest (Closed-Loop). Table 3 reports closedloop results on NAVSIMv1 navtest. MomADv2 achieves strong performance with 99.0 NC, 97.1 DAC, and 95.5 TTC.

Table 6: Ablation study of the two core modules on the NAVSIMv1 navtest split. SSM-Q denotes the Selective State-Space Planning Memory Query Module, and FM-Ref denotes the Flow-Matching Trajectory Residual Refiner.
<table><tr><td>SSM-Q</td><td>FM-Ref</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>Comf.↑</td><td>EP↑</td><td>PDMS↑</td></tr><tr><td></td><td></td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>√</td><td></td><td>98.2</td><td>94.5</td><td>94.0</td><td>100</td><td>82.1</td><td>86.9</td></tr><tr><td>√</td><td>√</td><td>99.0</td><td>97.1</td><td>95.5</td><td>100</td><td>83.2</td><td>89.9</td></tr></table>

Table 7: Ablation study of reliable temporal memory on the nuScenes validation set. All MomADv2 variants keep the FM-Ref and only difer in the historical memory strategy.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Memory Strategy</td><td colspan="2">Col. Rate (%)↓</td></tr><tr><td>1s</td><td>2s 3s 4s 5s 6s Avg.</td></tr><tr><td>MomAD</td><td>Historical query reuse</td><td>0.170.300.540.831.432.130.90</td></tr><tr><td>MomADv2</td><td>No historical memory</td><td>0.160.290.520.861.482.180.92</td></tr><tr><td></td><td>MomADv2 Naive historical query fusion</td><td>0.180.340.600.941.622.351.01</td></tr><tr><td></td><td>MomADv2 Command-aware memory</td><td>0.12 0.250.470.821.462.19 0.89</td></tr><tr><td></td><td></td><td>MomADv2 + Temporal-continuity filtering 0.08 0.21 0.410.78 1.39 2.12 0.83</td></tr><tr><td></td><td></td><td>MomADv2 + Trajectory-level alignment 0.04 0.160.370.75 1.35 2.07 0.79</td></tr><tr><td>MomADv2</td><td>Full selective memory</td><td>0.00 0.12 0.33 0.72 1.33 2.03 0.76</td></tr></table>

It also obtains 83.2 EP and 89.9 PDMS, showing competitive safety, progress, and planning stability.

NAVSIMv2 navhard (Closed-Loop). Table 4 reports results on the challenging NAVSIMv2 navhard split. MomADv2 achieves the best overall EPDMS of 39.5, clearly outperforming TransFuser, DifusionDrive, and GuideFlow. It maintains strong Stage-2 performance, especially on DAC and TTC, showing better robustness under long-horizon closed-loop interactions.

NAVSIMv2 navtest (Closed-Loop). Table 5 reports the results on the NAVSIMv2 navtest split. MomADv2 achieves the best EPDMS of 87.9 and ranks first on DDC, EP, and EC. These results demonstrate that reliable temporal memory improves long-horizon planning consistency and closed-loop decision quality.

## Ablation Studies

Roles ofDiferent Modules in MomADv2. Table 6 validates the efectiveness of SSM-Q and FM-Ref. SSM-Q improves PDMS from 84.0 to 86.9, showing the benefit of reliable temporal memory. With FM-Ref, PDMS further increases to 89.9, demonstrating that residual trajectory refinement provides complementary gains.

Ablation on Reliable Temporal Memory. Table 7 compares historical memory strategies. The full selective strategy achieves the lowest collision rate of 0.76%, outperforming naive fusion and partial filtering variants. This confirms that reliable memory selection is essential for safe long-horizon planning.

Ablation on Historical Memory Length. Table 8 studies the efect of memory length K. K = 4 achieves the best PDMS of 89.9, while longer histories gradually degrade performance. This suggests that moderate and reliable memory is more beneficial than excessively long historical context.

Table 8: Ablation study of historical memory length K on the NAVSIMv1 navtest split. K denotes the number of historical planning states used in the SSM-Q.
<table><tr><td>K</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>Comf.↑</td><td>EP↑</td><td>PDMS↑</td></tr><tr><td>1</td><td>98.4</td><td>95.4</td><td>94.7</td><td>100</td><td>82.4</td><td>88.2</td></tr><tr><td>2</td><td>98.7</td><td>96.3</td><td>95.0</td><td>100</td><td>82.8</td><td>89.1</td></tr><tr><td>4</td><td>99.0</td><td>97.1</td><td>95.5</td><td>100</td><td>83.2</td><td>89.9</td></tr><tr><td>6</td><td>98.5</td><td>95.7</td><td>94.6</td><td>100</td><td>82.3</td><td>88.5</td></tr><tr><td>8</td><td>98.1</td><td>95.0</td><td>94.1</td><td>100</td><td>81.9</td><td>87.7</td></tr></table>

![](images/bb0441965304e15757afe50ba7c0a870dcb4738ba070c7cb20343ae4d78c873c.jpg)  
Figure 4: Visualization of MomADv2 on nuScenes dataset.

## Visualization

Figure 4 visualizes consecutive planning results of MomAD and MambaFlow. Compared with MomAD, MomADv2 generates more temporally consistent trajectories across t−1, t, and t+1, with predictions closer to the ground truth. Additional visualizations are provided in the appendix.

## Conclusion

We propose MomADv2, a reliable state-space memory framework that selectively preserves useful history while suppressing stale or command-inconsistent memory. Specifically, we introduce the Selective State-Space Planning Memory Query Module, which filters historical planning queries according to temporal continuity and command consistency, selects command-relevant planning modes, and models the evolution of planning intentions through a selective statespace mechanism. We further propose the Flow-Matching Trajectory Residual Refiner, which learns a continuous residual correction field to mitigate local trajectory deviations and accumulated planning errors while preserving the stability of anchor-based planning. Extensive experiments on NAVSIM, Bench2Drive, and nuScenes demonstrate that MomADv2 consistently improves long-horizon planning accuracy, temporal consistency, and driving safety.

Limitation and Future Work. MomADv2 relies on predefined driving commands for memory selection and does not explicitly model uncertainty in future scene evolution. Future work will investigate command-free intention inference, uncertainty-aware memory selection, and worldmodel-based future reasoning.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant No. 52502496, the Beijing Natural Science Foundation under Grant No. L2609087, and the Natural Science Foundation of Chongqing, China under Grant No. CSTB2025NSCQ-GPX0413.

## References

Caesar, H.; Bankiti, V.; Lang, A. H.; Vora, S.; Liong, V. E.; Xu, Q.; Krishnan, A.; Pan, Y.; Baldan, G.; and Beijbom, O. 2020. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11621–11631.

Cao, W.; Hallgarten, M.; Li, T.; Dauner, D.; Gu, X.; Wang, C.; Miron, Y.; Aiello, M.; Li, H.; Gilitschenski, I.; et al. 2025. Pseudo-simulation for autonomous driving. In Conference on Robot Learning.

Chitta, K.; Prakash, A.; Jaeger, B.; Yu, Z.; Renz, K.; and Geiger, A. 2023. TransFuser: Imitation With Transformer-Based Sensor Fusion for Autonomous Driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11): 12878–12895.

Dao, T.; and Gu, A. 2024. Transformers are SSMs: Generalized Models and Eficient Algorithms Through Structured State Space Duality. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, 10041–10071. PMLR.

Dauner, D.; Hallgarten, M.; Li, T.; Weng, X.; Huang, Z.; Yang, Z.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; et al. 2024. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37: 28706–28719.

Gu, A.; and Dao, T. 2024. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In First Conference on Language Modeling.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 770– 778.

Hu, Y.; Yang, J.; Chen, L.; Li, K.; Sima, C.; Zhu, X.; Chai, S.; Du, S.; Lin, T.; Wang, W.; Lu, L.; Jia, X.; Liu, Q.; Dai, J.; Qiao, Y.; and Li, H. 2023. Planning-Oriented Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17853– 17862.

Jia, X.; Gao, Y.; Chen, L.; Yan, J.; Liu, P. L.; and Li, H. 2023. Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7953–7963.

Jia, X.; Yang, Z.; Li, Q.; Zhang, Z.; and Yan, J. 2024. Bench2drive: Towards multi-ability benchmarking ofclosedloop end-to-end autonomous driving. arXiv preprint arXiv:2406.03877.

Jia, X.; You, J.; Zhang, Z.; and Yan, J. 2025. Drivetransformer: Unified transformer for scalable end-to-end autonomous driving. In International Conference on Learning Representations.

Jiang, B.; Chen, S.; Gao, H.; Liao, B.; Zhang, Q.; Liu, W.; and Wang, X. 2026. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. In International Conference on Learning Representations.

Jiang, B.; Chen, S.; Xu, Q.; Liao, B.; Chen, J.; Zhou, H.; Zhang, Q.; Liu, W.; Huang, C.; and Wang, X. 2023. Vad: Vectorized scene representation for eficient autonomous driving. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 8340–8350.

Li, H.; Hou, Y.; Xing, X.; Ma, Y.; Sun, X.; and Zhang, Y. 2025a. Occmamba: Semantic occupancy prediction with state space models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11949– 11959.

Li, K.; Li, Z.; Lan, S.; Xie, Y.; Zhang, Z.; Liu, J.; Wu, Z.; Yu, Z.; and Alvarez, J. M. 2025b. Hydra-mdp++: Advancing endto-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820.

Li, P.; and Cui, D. 2025. Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving. In International Conference on Learning Representations (ICLR).

Li, Y.; Fan, L.; He, J.; Wang, Y.; Chen, Y.; Zhang, Z.; and Tan, T. 2025c. Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations.

Li, Y.; Shang, S.; Liu, W.; Zhan, B.; Wang, H.; Wang, Y.; Chen, Y.; Wang, X.; An, Y.; Tang, C.; et al. 2025d. DriveVLA-W0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796.

Li, Y.; Wang, Y.; Liu, Y.; He, J.; Fan, L.; and Zhang, Z. 2025e. End-to-end driving with online trajectory evaluation via bev world model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 27137–27146.

Li, Y.; Xiong, K.; Guo, X.; Li, F.; Yan, S.; Xu, G.; Zhou, L.; Chen, L.; Sun, H.; Wang, B.; et al. 2025f. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052.

Li, Z.; Li, K.; Wang, S.; Lan, S.; Yu, Z.; Ji, Y.; Li, Z.; Zhu, Z.; Kautz, J.; Wu, Z.; et al. 2024. Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation. arXiv preprint arXiv:2406.06978.

Liao, B.; Chen, S.; Yin, H.; Jiang, B.; Wang, C.; Yan, S.; Zhang, X.; Li, X.; Zhang, Y.; Zhang, Q.; et al. 2025. DifusionDrive: Truncated Difusion Model for End-to-End Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Highlight.

Liu, L.; Jia, C.; Song, Z.; Pan, H.; Liao, B.; Sun, W.; Zhang, Y.; Yang, L.; and Luo, Y. 2025a. Fully Unified Motion Planning for End-to-End Autonomous Driving. arXiv preprint arXiv:2504.12667.

Liu, L.; Jia, C.; Yu, G.; Song, Z.; Li, J.; Jia, F.; Wu, P.; Hao, X.; and Luo, Y. 2025b. GuideFlow: Constraint-Guided Flow Matching for Planning in End-to-End Autonomous Driving. arXiv preprint arXiv:2511.18729.

Liu, L.; Song, Z.; Jia, C.; Ye, H.; Hao, X.; Chen, L.; et al. 2026. DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving. arXiv preprint arXiv:2602.06521.

Liu, Y.; Tian, Y.; Zhao, Y.; Yu, H.; Xie, L.; Wang, Y.; Ye, Q.; and Liu, Y. 2024. VMamba: Visual State Space Model. In Advances in Neural Information Processing Systems, volume 37.

Lu, S.; Liu, R.; Yang, D.; and He, L. 2025. M<sup>3</sup>S-HEV: Mamba-Enhanced Deep Reinforcement Learning for Endto-End Autonomous Driving with BEV-Perception. arXiv preprint arXiv:2508.06074.

Shang, S.; Chen, Y.; Wang, Y.; Li, Y.; and Zhang, Z. 2025. Drivedpo: Policy learning via safety dpo for end-to-end autonomous driving. In Advances in Neural Information Processing Systems.

Song, Z.; Jia, C.; Liu, L.; Pan, H.; Zhang, Y.; Wang, J.; Zhang, X.; Xu, S.; Yang, L.; and Luo, Y. 2025. Don’t Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, 22432–22441.

Song, Z.; Jia, C.; Liu, L.; Yang, L.; Zhang, S.; Jia, F.; Zhao, F.; Wu, P.; Xu, S.; Lv, C.; et al. 2026a. GraphWorld: Long-Horizon Planning with World Models for End-to-End Autonomous Driving. arXiv preprint arXiv:2606.16274.

Song, Z.; Liu, L.; Pan, H.; Liao, B.; Guo, M.; Yang, L.; Zhang, Y.; Xu, S.; Jia, C.; and Luo, Y. 2026b. Diver: Reinforced difusion breaks imitation bottlenecks in end-to-end autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Su, H.; Wu, W.; Song, F.; Zhang, J.; Yang, Z.; and Yan, J. 2026. DriveMamba: Task-Centric Scalable State Space Model for Eficient End-to-End Autonomous Driving. In International Conference on Learning Representations. Accepted to ICLR 2026.

Sun, B.; Zhang, B.; Lu, J.; Feng, X.; Shang, J.; Cao, R.; Zheng, M.; Wang, C.; Yang, S.; Cao, Y.; et al. 2026. FocalAD: Local Motion Planning for End-to-End Autonomous Driving. Automotive Innovation.

Sun, W.; Lin, X.; Shi, Y.; Zhang, C.; Wu, H.; and Zheng, S. 2024. SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation. arXiv preprint arXiv:2405.19620.

Wang, J.; Jiang, C.; and Xu, H. 2025. GMF-Drive: Gated Mamba Fusion with Spatial-Aware BEV Representation for End-to-End Autonomous Driving. arXiv preprint arXiv:2508.06113.

Xing, Z.; Zhang, X.; Hu, Y.; Jiang, B.; He, T.; Zhang, Q.; Long, X.; and Yin, W. 2025. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-toend autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1602–1611.

Yang, Z.; Jia, X.; Li, Q.; Yang, X.; Yao, M.; and Yan, J. 2025. Raw2Drive: Reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). In Advances in Neural Information Processing Systems.

Yao, W.; Li, Z.; Lan, S.; Wang, Z.; Sun, X.; Alvarez, J. M.; and Wu, Z. 2026. Drivesuprim: Towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 11910–11918.

Yuan, C.; Zhang, Z.; Sun, J.; Sun, S.; Huang, Z.; Lee, C. D. W.; Li, D.; Han, Y.; Wong, A.; Tee, K. P.; et al. 2024. Drama: An eficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601.

Zhang, B.; Song, N.; Jin, X.; and Zhang, L. 2025a. Bridging past and future: End-to-end autonomous driving with historical prediction and planning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 6854–6863.

Zhang, K.; Tang, Z.; Hu, X.; Pan, X.; Guo, X.; Liu, Y.; Huang, J.; Yuan, L.; Zhang, Q.; Long, X.-X.; et al. 2025b. Epona: Autoregressive difusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 27220–27230.

Zheng, W.; Chen, W.; Huang, Y.; Zhang, B.; Duan, Y.; and Lu, J. 2024a. Occworld: Learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, 55–72. Springer.

Zheng, W.; Song, R.; Guo, X.; and Chen, L. 2024b. Genad: Generative end-to-end autonomous driving. arXiv preprint arXiv:2402.11502.

Zheng, Z.; Chen, S.; Yin, H.; Zhang, X.; Zou, J.; Wang, X.; Zhang, Q.; and Zhang, L. 2026. ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3729– 3739.

Zhou, Z.; Cai, T.; Zhao, S.; Zhang, Y.; Huang, Z.; Zhou, B.; and Ma, J. 2025. AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 27920–27956. Curran Associates, Inc.

Zhu, L.; Liao, B.; Zhang, Q.; Wang, X.; Liu, W.; and Wang, X. 2024. Vision Mamba: Eficient Visual Representation Learning with Bidirectional State Space Model. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, 62429–62442. PMLR.

Zou, J.; Chen, S.; Liao, B.; Zheng, Z.; Song, Y.; Zhang, L.; Zhang, Q.; Liu, W.; and Wang, X. 2025. Difusion-DriveV2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745.

## A Appendix

## A.1 Broader Impacts

MomADv2 has the potential to improve the safety and reliability of autonomous driving by enhancing long-horizon planning consistency and suppressing unreliable temporal memory. Its selective state-space memory mechanism preserves useful historical context, while the flow-matching trajectory refiner improves trajectory accuracy and smoothness. These capabilities may support safer interactions, more stable decision-making, and greater robustness in complex traffic environments. More broadly, the proposed reliable temporal modeling paradigm may also benefit other sequential decision-making systems in robotics and embodied intelligence.

## A.2 Contributions

Our contributions are summarized as follows.

MomADv2 Framework. We propose MomADv2, a reliable temporal memory framework for end-to-end autonomous driving. Unlike prior methods that directly reuse historical planning states, MomADv2 selectively preserves useful temporal information while suppressing stale or commandinconsistent memory. This design improves long-horizon planning consistency, trajectory stability, and robustness under dynamic driving conditions.

SSM-Q and FM-Ref. We introduce the Selective State-Space Planning Memory Query Module (SSM-Q), which retrieves historical planning queries based on command consistency, temporal continuity, and trajectory-level alignment, and models the evolution of planning intentions through a selective state-space mechanism. We further propose the Flow-Matching Trajectory Residual Refiner (FM-Ref), which learns a conditional residual velocity field to progressively correct local trajectory deviations while preserving the stability of anchor-based planning.

## A.3 Datasets

NAVSIM. Following the oficial NAVSIM protocol, we report results on three settings: NAVSIM-v1 navtest, NAVSIMv2 navtest, and NAVSIM-v2 navhard. The navtest split evaluates general planning ability on standard real-world driving scenes, while navhard focuses on more challenging and safety-critical long-tail scenarios. NAVSIM-v1 uses the Predictive Driver Model Score (PDMS) to assess key aspects of driving behavior, including safety, feasibility, comfort, and progress. NAVSIM-v2 adopts the Extended Predictive Driver Model Score (EPDMS), which further incorporates rule- and comfort-related metrics such as driving-direction compliance, trafic-light compliance, lane keeping, and extended comfort. For NAVSIM-v2 navhard, we follow the oficial two-stage protocol: Stage 1 evaluates the planner on the original observation, and Stage 2 re-evaluates it with synthesized future observations around the Stage-1 endpoint. All NAVSIM scores are reported as percentages, and higher values indicate better planning quality.

Bench2Drive. We further evaluate closed-loop driving performance on Bench2Drive (Jia et al. 2024), a CARLA-based benchmark designed for fine-grained multi-ability evaluation. Its evaluation set contains 220 short routes covering 44 interactive scenarios across diverse towns and weather conditions. Unlike open-loop log-replay evaluation, the executed actions afect subsequent vehicle states and sensor observations, enabling direct assessment of planning performance under closed-loop interactions. Following the oficial protocol, we report Driving Score (DS), Success Rate (SR), Eficiency (Efi.), Comfortness (Comf.), and five advanced driving abilities: Merging, Overtaking, Emergency Brake, Give Way, and Trafic Sign. For open-loop evaluation on the logged Bench2Drive data, we additionally report the average L2 displacement error and trajectory diversity.

nuScenes. We additionally conduct open-loop planning experiments on the nuScenes validation set (Caesar et al. 2020). nuScenes contains 1,000 real-world driving scenes collected using a full surround-view sensor suite, with each scene lasting approximately 20 seconds. Under the open-loop protocol, the planner predicts future ego trajectories from logged observations without afecting subsequent scene evolution. We evaluate short-horizon planning at 1, 2, and 3 seconds and long-horizon planning at 4, 5, and 6 seconds. The reported metrics include L2 displacement error, collision rate, and, for multi-mode planners, trajectory diversity. Together, NAVSIM, Bench2Drive, and nuScenes provide complementary evaluations of pseudo-closed-loop planning quality, closed-loop interaction ability, open-loop accuracy, safety, multimodality, and long-horizon robustness.

## A.4 Evaluation Metrics

PDM and PDMS. Following NAVSIM (Dauner et al. 2024), the Predictive Driver Model (PDM) refers to the rule-based planner used to generate and score trajectory proposals. Given an observation $o _ { t }$ and a set of candidate trajectories $\mathcal { T } = \{ \tau _ { i } \} _ { i = 1 } ^ { N }$ , PDM selects the trajectory with the highest PDM score:

$$
\tau ^ { \star } = \arg \operatorname* { m a x } _ { \tau _ { i } \in \mathcal { T } } \mathrm { S c o r e } _ { \mathrm { P D M } } ( \tau _ { i } ) ,\tag{34}
$$

where Score<sub>PDM</sub> evaluates each candidate by simulating it and aggregating safety, feasibility, progress, and comfort terms. NAVSIM adopts this PDM-style scoring function as the benchmark metric, namely the Predictive Driver Model Score (PDMS).

For NAVSIM-v1, evaluation first unrolls the predicted trajectory in a non-reactive simulator and computes normalized subscores in [0, 1]. These subscores are divided into hard penalties and soft objectives. The hard penalties include noat-fault collision (NC) and drivable-area compliance (DAC), while the soft objectives include ego progress (EP), time-tocollision (TTC), and comfort (C). Let $\mathcal { M } _ { \mathrm { h a r d } } ^ { \mathrm { ~ - ~ } } = \{ \mathrm { N C } , \mathrm { D A C } \}$ and $\mathcal { M } _ { \mathrm { { s o f t } } } = \{ \mathrm { E P } , \mathrm { T T C } , \mathrm { C } \}$ . The final score is computed

$$
\begin{array} { r } { \mathrm { P D M S } = \underbrace { \prod _ { m \in \mathcal { M } _ { \mathrm { h a r d } } } s _ { m } } _ { \mathrm { h a r d p e n a l i t i e s } } } \\ { \times \underbrace { \sum _ { m \in \mathcal { M } _ { \mathrm { s o f t } } } \alpha _ { m } s _ { m } } _ { \mathrm { w i g h e d s f i e d s } } . } \end{array}\tag{35}
$$

Here, $s _ { m }$ denotes the normalized subscore. NAVSIM uses $\alpha _ { \mathrm { { E P } } } = 5 , \alpha _ { \mathrm { { T T C } } } = 5 ,$ , and $\alpha _ { \mathrm { { C } } } = 2 .$ . The multiplicative penalty term makes safety-critical violations dominate the final score: if the ego trajectory causes an at-fault collision, s<sub>NC</sub> becomes zero; if the trajectory leaves the drivable area, s<sub>DAC</sub> becomes zero. Collisions with static objects are assigned a softer penalty in NAVSIM, while non-at-fault collisions under the non-reactive setting are ignored.

The soft terms measure driving quality when the trajectory is admissible. The ego-progress subscore s<sub>EP</sub> is computed as the ratio between the ego progress along the route centerline and a safe upper-bound progress estimated by the privileged PDM-Closed planner, clipped to [0, 1]. The time-to-collision subscore $s _ { \mathrm { T T C } }$ is initialized to one and is set to zero if, at any simulation step within the 4-second horizon, the projected ego motion violates the predefined TTC safety threshold with respect to surrounding vehicles. The comfort subscore s<sub>C</sub> evaluates whether the trajectory satisfies acceleration and jerk thresholds. Therefore, PDMS rewards trajectories that simultaneously remain collision-free, stay on the road, make suficient route progress, preserve safety margins, and maintain smooth motion.

EPDMS. For NAVSIM-v2, the benchmark extends PDMS to the Extended Predictive Driver Model Score (EPDMS) by adding more fine-grained rule-compliance and comfort terms. In addition to NC, DAC, EP, and TTC, EPDMS includes driving-direction compliance (DDC), trafic-light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC). Following the NAVSIM-v2 protocol (Cao et al. 2025), the single-stage extended score can be written as

$$
\begin{array} { r } { \mathrm { E P D M S } = \underbrace { \displaystyle \prod _ { m \in \mathcal { M } _ { \mathrm { p e n } } } f _ { m } ( \tau _ { \mathrm { a g e n t } } , \tau _ { \mathrm { h u m a n } } ) } _ { \mathrm { p e n a l t y t e r m s } } } \\ { \times \underbrace { \sum _ { m \in \mathcal { M } _ { \mathrm { a v g } } } \beta _ { m } f _ { m } ( \tau _ { \mathrm { a g e n t } } , \tau _ { \mathrm { h u m a n } } ) } _ { \displaystyle \sum _ { m \in \mathcal { M } _ { \mathrm { a v g } } } \beta _ { m } } . } \end{array}\tag{36}
$$

Here,

$$
\begin{array} { r } { \mathcal { M } _ { \mathrm { p e n } } = \{ \mathrm { N C } , \mathrm { D A C } , \mathrm { D D C } , \mathrm { T L C } \} , } \\ { \mathcal { M } _ { \mathrm { a v g } } = \{ \mathrm { E P } , \mathrm { T T C } , \mathrm { L K } , \mathrm { H C } , \mathrm { E C } \} . } \end{array}\tag{37}
$$

The default weights are $\beta _ { \mathrm { E P } } = 5 , \beta _ { \mathrm { T T C } } = 5 .$ , and $\beta _ { \mathrm { L K } } =$ $\beta _ { \mathrm { H C } } = \beta _ { \mathrm { E C } } = \bar { 2 }$

The function $f _ { m } ( \tau _ { \mathrm { a g e n t } } , \tau _ { \mathrm { h u m a n } } )$ denotes the humanfiltered subscore for metric m. This filtering mechanism ignores a rule violation when the same violation is also committed by the human trajectory in the corresponding scene, reducing false penalties caused by annotation noise or contextually necessary maneuvers. DDC evaluates whether the ego vehicle follows the legal driving direction, TLC checks compliance with trafic lights, LK evaluates lane-keeping behavior, and HC and EC measure trajectory smoothness under the extended NAVSIM-v2 protocol.

NAVSIM-v2 Navhard. For NAVSIM-v2 navhard, evaluation follows a two-stage pseudo-simulation protocol. Stage 1 scores the planner from the original real observation, producing $s _ { 1 }$ . Stage 2 evaluates the planner on a set of pre-generated synthetic observations around plausible future ego states, producing $\{ s _ { 2 } ^ { ( i ) } \} _ { i = 1 } ^ { K }$ . These Stage-2 scores are aggregated by a Gaussian-weighted average according to the distance between each synthetic start point $x _ { i }$ and the Stage-1 endpoint xˆ:

$$
\begin{array} { l } { \displaystyle { s _ { 2 } = \sum _ { i = 1 } ^ { K } \hat { w } _ { i } s _ { 2 } ^ { ( i ) } , } } \\ { \displaystyle { \hat { w } _ { i } = \frac { w _ { i } } { \sum _ { j = 1 } ^ { K } w _ { j } } , } } \\ { \displaystyle { w _ { i } = \exp \left( - \frac { \| x _ { i } - \hat { x } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) . } } \end{array}\tag{38}
$$

The final navhard score multiplies the original-observation score and the aggregated synthetic-observation score:

$$
\mathrm { E P D M S _ { n a v h a r d } } = s _ { 1 } s _ { 2 } .\tag{39}
$$

This two-stage aggregation evaluates both immediate planning quality and robustness to future observation shifts, making navhard stricter than the single-stage NAVSIM-v1 PDMS or NAVSIM-v2 navtest evaluation.

Bench2Drive Metrics. Bench2Drive (Jia et al. 2024) evaluates end-to-end driving under closed-loop interactions. Success Rate (SR) measures the proportion of routes completed within the allotted time without trafic violations. Driving Score (DS) jointly considers route completion and infraction penalties:

$$
\begin{array} { l } { \displaystyle \mathrm { S R } = \frac { N _ { \mathrm { s u c c } } } { N _ { \mathrm { r o u t e } } } , } \\ { \displaystyle \mathrm { D S } = \frac { 1 } { N _ { \mathrm { r o u t e } } } \sum _ { r = 1 } ^ { N _ { \mathrm { r o u t e } } } { \mathrm { R C } _ { r } \prod _ { q = 1 } ^ { Q _ { r } } { p _ { r , q } } } . } \end{array}\tag{40}
$$

Here, $N _ { \mathrm { s u c c } }$ and $N _ { \mathrm { r o u t e } }$ denote the numbers of successfully completed and total routes, respectively. RC<sub>r</sub> is the routecompletion percentage of route $r , Q _ { r }$ is the number of infractions occurring on that route, and $p _ { r , q }$ is the penalty associated with its q-th infraction. Consequently, DS gives partial credit for route progress while penalizing collisions, trafic-rule violations, and other unsafe behaviors.

Eficiency measures the average ratio between the egovehicle speed and the average speed of nearby vehicles at regularly spaced route checkpoints. It may exceed 100% when the ego vehicle travels faster than the surrounding trafic. Comfortness measures the proportion of trajectory segments satisfying predefined bounds on longitudinal and lateral acceleration, yaw rate, yaw acceleration, and jerk.

Bench2Drive further groups its interactive scenarios into five advanced abilities: Merging, Overtaking, Emergency Brake, Give Way, and Trafic Sign. The corresponding ability-wise scores provide a more fine-grained assessment than the overall DS and SR, while their arithmetic mean is reported as the overall Multi-Ability score. Higher values indicate better performance for all closed-loop Bench2Drive metrics. For open-loop Bench2Drive evaluation, we additionally report the average L2 displacement error and average trajectory diversity, using the same definitions as the nuScenes metrics below.

nuScenes Metrics. For open-loop planning on nuScenes (Caesar et al. 2020), we evaluate the selected ego trajectory using L2 displacement error and collision rate. Given N evaluation samples, the displacement error at future timestamp t is

$$
\mathrm { L } 2 ^ { ( t ) } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left. \hat { \mathbf { p } } _ { n } ^ { ( t ) } - \mathbf { p } _ { n , \mathrm { g t } } ^ { ( t ) } \right. _ { 2 } ,\tag{41}
$$

where $\hat { \mathbf { p } } _ { n } ^ { ( t ) }$ and $\mathbf { p } _ { n , \mathrm { g t } } ^ { ( t ) }$ denote the predicted and ground-truth ego positions of sample n at timestamp t, respectively. The average L2 error is obtained by averaging $\mathrm { L 2 } ^ { \left( t \right) }$ over the reported future timestamps. Lower values indicate more accurate imitation of the logged human trajectory.

Following SparseDrive (Sun et al. 2024), collision rate measures the percentage of evaluation samples in which the predicted oriented ego footprint overlaps with any surrounding actor at the evaluated horizon. The ego heading is estimated from consecutive trajectory points instead of being assumed constant. This oriented-box implementation reduces false collision detections caused by coarse occupancy rasterization or fixed-heading approximations. A lower collision rate indicates safer planning.

For a multi-mode planner producing M candidate trajectories, we additionally adopt the time-conditioned trajectory Diversity Metric $\mathrm { D i v } ^ { ( t ) }$ (Song et al. 2026b). Let $\mathbf { p } _ { t } ^ { ( i ) }$ denote the waypoint of the i-th trajectory mode at timestamp t. The metric is computed as

$$
\begin{array} { r l } & { \displaystyle { d _ { i j } ^ { ( t ) } = \left\| { \bf p } _ { t } ^ { ( i ) } - { \bf p } _ { t } ^ { ( j ) } \right\| _ { 2 } , } } \\ & { \displaystyle { D _ { \mathrm { r a w } } ^ { ( t ) } = \frac { 2 } { M ( M - 1 ) } \sum _ { i = 1 } ^ { M - 1 } \sum _ { j = i + 1 } ^ { M } d _ { i j } ^ { ( t ) } , } } \\ & { \quad \quad \quad \quad \quad Z _ { t } = \displaystyle { \epsilon + \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left\| { \bf p } _ { t } ^ { ( i ) } \right\| _ { 2 } } , } \\ & { \quad \quad \quad \quad \mathrm { D i v } ^ { ( t ) } = \operatorname* { m i n } \left( 1 , \frac { D _ { \mathrm { r a w } } ^ { ( t ) } } { Z _ { t } } \right) . } \end{array}\tag{42}
$$

Here, $D _ { \mathrm { r a w } } ^ { ( t ) }$ is the mean pairwise distance between trajectory modes, $Z _ { t }$ normalizes this distance according to the trajectory scale, and ϵ is a small constant preventing division by zero. The resulting metric is bounded within [0, 1]. A higher value indicates that the predicted modes are more diverse and better separated in trajectory space, whereas a value close to zero indicates mode collapse. The average diversity is computed by averaging $\mathrm { D i v } ^ { ( t ) }$ over the reported future timestamps. This metric is applicable only to multi-mode planning methods.

## A.5 More Details of MomADv2

Temporal-Link Construction and Memory Reset. The temporal-continuity token used by SSM-Q is a frame-level identifier rather than a learnable representation. Let prev $\left( \chi _ { t } \right)$ denote the identifier of the immediate predecessor of frame t according to the sequential metadata. We define

$$
\chi _ { t } ^ { ( - k ) } = \mathrm { p r e v } ^ { k } ( \chi _ { t } ) ,\tag{43}
$$

where prev<sup>k</sup> applies the predecessor operation k times. Therefore, a historical entry at time $t - k$ is temporally valid only when $\chi _ { t } ^ { ( - k ) } = \chi _ { t - k }$ . The identifier is used only for memory indexing and hard validity masking; it is not embedded or provided as an input feature to the SSM encoder.

The memory bank is reset at the beginning of each sequence and whenever the current frame belongs to a new scene, has no valid predecessor, or violates the expected frame interval. During the first K frames of a sequence, only the available valid entries are used. If no valid historical entry exists, the temporal residual is set to zero and the model falls back to the single-frame planning branch.

Memory Contents and Causal Update. At frame $t ,$ the baseline planner produces the scene-conditioned planning queries $\bar { \mathbf { Q } } _ { t } ^ { \mathrm { { r a w } } }$ and command-specific candidate trajectories $\bar { \mathbf { Y } } _ { t } ^ { \mathrm { b a s e , c m d } }$ before temporal enhancement. Here, “raw” means pre-SSM rather than a fixed planning-anchor embedding. Each memory entry is represented as

$$
\begin{array} { r } { \mathbf { \mathcal { E } } _ { t } = \left( \chi _ { t } , c _ { t } , \mathbf { Q } _ { t } ^ { \mathrm { r a w } } , \mathbf { Y } _ { t } ^ { \mathrm { b a s e , c m d } } \right) . } \end{array}\tag{44}
$$

The implementation indexes each frame-level entry by $\chi _ { t }$ and stores $c _ { t }$ as command metadata; retrieval therefore treats $\left( \chi _ { t } , c _ { t } \right)$ as a compound logical key. Candidates associated with commands diferent from $c _ { t }$ are never eligible for retrieval.

We use a first-in–first-out memory containing at most K historical entries. Following the memory-length ablation in the main paper, we set $K = 4$ unless otherwise specified. At each frame, the model first reads $\mathcal { M } _ { t - 1 }$ , retrieves and aggregates valid historical states, produces the final planning result, and only then appends $\mathcal { E } _ { t }$ to the memory bank. Entries older than K frames are discarded. Only the preenhancement query and baseline trajectory are stored; neither the SSM-enhanced query nor the flow-refined trajectory is written back. This read-before-write strategy prevents the current frame from retrieving itself and avoids recursive amplification of temporal corrections.

Temporally Shifted Trajectory Distance. The trajectorylevel alignment in SSM-Q compares historical and current planning candidates over the same absolute future times. This temporal alignment is necessary because a trajectory predicted at time $t - k$ starts k frames earlier than a trajectory predicted at time t. Let $\Delta _ { \mathrm { f r a m e } }$ denote the interval between consecutive planning frames and $\Delta _ { \mathrm { t r a j } }$ denote the interval between trajectory waypoints. The number of elapsed trajectory steps is

$$
s _ { k } = \mathrm { r o u n d } \left( \frac { k \Delta _ { \mathrm { f r a m e } } } { \Delta _ { \mathrm { t r a j } } } \right) .\tag{45}
$$

For a historical candidate $\mathbf { Y } _ { t - k , h }$ , we first discard its first $s _ { k }$ waypoints, which correspond to time steps that have already elapsed. The remaining points are transformed from the historical ego coordinate system to the current ego coordinate system using the relative ego pose $\mathcal { T } _ { t  t - k }$ . The shifted trajectory distance is then defined over the overlapping horizon. Specifically, the aligned historical waypoint is

$$
\widetilde { \mathbf { y } } _ { t - k , h } ^ { \ell } = { \mathcal { T } } _ { t  t - k } ( \mathbf { y } _ { t - k , h } ^ { \ell + s _ { k } } ) ,\tag{46}
$$

and the shifted distance is

$$
D _ { \mathrm { s h i f t } } = \frac { \displaystyle \sum _ { \ell = 1 } ^ { T - s _ { k } } \left\| \widetilde { \mathbf { y } } _ { t - k , h } ^ { \ell } - \mathbf { y } _ { t , j } ^ { \ell } \right\| _ { 2 } } { T - s _ { k } } .\tag{47}
$$

When the temporal ofset does not correspond to an integer number of trajectory steps, the historical trajectory is linearly interpolated at the timestamps of the current trajectory before computing the distance. Thus, the second predicted waypoint from the previous frame is compared with the first predicted waypoint from the current frame when one trajectory step has elapsed. This avoids treating the same driving intention as inconsistent solely because the two predictions have diferent temporal origins.

For each current command-specific candidate $j ,$ SSM-Q selects the historical candidate with the minimum shifted distance. The resulting distance is also used by the similarity weight and reliability gate:

$$
d _ { t , k , j } = D _ { \mathrm { s h i f t } } \left( \mathbf { Y } _ { t - k , h ^ { * } } , \mathbf { Y } _ { t , j } \right) .\tag{48}
$$

If $s _ { k } \geq T ,$ , the two trajectories have no overlapping future horizon and the corresponding historical entry is marked invalid. This operation is used only for candidate association and reliability estimation: it does not shift the planning query itself or overwrite the stored trajectory.

Handling Inaccurate Baseline Candidates. The baseline trajectory is used as an association cue rather than directly propagated as the final output. A historical candidate with a large aligned distance receives a small similarity weight through e $\mathrm { { \varepsilon } } \mathrm { p } ( - d _ { t , k , j } / \alpha )$ and a small distancereliability value through $r _ { t , j } ^ { \mathrm { d i s t } }$ . Invalid, temporally discontinuous, or command-inconsistent entries receive zero weight. If all retrieved entries are rejected, the reliability-gated residual vanishes and the current query remains unchanged. Consequently, an unreliable baseline match can be suppressed instead of being unconditionally injected into the current planning state.

Causal Online Inference. MomADv2 performs causal stateful inference without test-time optimization. For every input frame, the perception backbone is evaluated once. The mode then executes the baseline planning head, historical-memory retrieval, trajectory-level alignment, SSM-based query enhancement, and the planning refinement head in sequence.

The resulting trajectory is further processed by the flow refiner using $S = 2$ Euler evaluations. Finally, the current preenhancement query and baseline trajectory are appended to the memory bank for the next frame. All model parameters remain fixed during this process; only the explicit memory state changes. The proposed inference procedure is therefore online stateful inference rather than test-time training or test-time adaptation.

## A.6 Implementation Details

Training Protocol on nuScenes. For the nuScenes experiments, we adopt MomAD as the baseline and train MomADv2 using a two-stage optimization protocol.

Stage I: Baseline Pretraining. We first follow the original MomAD training procedure to obtain a fully trained end-toend autonomous driving model comprising sparse perception, motion prediction, and ego planning modules. Based on this pretrained model, most components of the original Topological Trajectory Matching (TTM) and Momentum Planning Interactor (MPI) modules are replaced with the proposed Selective State-Space Planning Memory Query Module (SSM-Q) and Flow-Matching Trajectory Residual Refiner (FM-Ref).

Stage II: Module Adaptation. MomADv2 is initialized from the pretrained MomAD checkpoint. During this stage, the image backbone, sparse perception modules, motion prediction branch, and most parameters of the original planning head are frozen. We optimize only the newly introduced temporal memory and trajectory refinement components, including the selective state-space encoder, the SSM-related reliability and residual gates, the trajectory residual prediction head, and the flow-matching refinement head. When enabled, the planning regression branch in the final refinement layer is also jointly fine-tuned to improve the integration of the proposed modules with the original anchor-based planner. All Batch Normalization layers are maintained in evaluation mode to preserve the pretrained feature statistics.

Training Configuration. The second stage is trained for 10 epochs with a learning rate of $3 \times 1 0 ^ { - 6 }$ . Following the baseline configuration, we employ ResNet-50 as the backbone with an input resolution of $2 5 6 \times 7 0 4$ . The detection range is defined as a circular region with a radius of 55 m, while the online mapping range is set to 60 × 30 m along the longitudinal and lateral directions, respectively. The motion prediction branch maintains six trajectory modes. All experiments are conducted on 8 NVIDIA RTX 4090 GPUs.

Optimization Rationale. The first stage establishes a strong and stable perception–motion–planning baseline, whereas the second stage focuses on adapting SSM-Q and FM-Ref without disrupting the pretrained representation and planning capabilities. This decoupled optimization strategy mitigates interference caused by large-scale parameter updates, stabilizes the integration of reliable temporal memory and residual trajectory refinement, and improves training eficiency under a limited additional optimization budget.

## A.7 More Planning Results

Multi-Ability Evaluation on Bench2Drive. As shown in Table 9, we further evaluate the driving ability of MomADv2 under diverse interactive scenarios. With expert feature distillation, MomADv2 achieves the best performance on Merging and Trafic Sign, and obtains competitive mean performance of 49.02%, outperforming DriveAdapter and DIVER by 16.5% and 16.5%, respectively. Without expert feature distillation, MomADv2 consistently ranks first across all abilities, improving the mean score from 22.67% to 25.84% compared with DIVER. These results demonstrate that MomADv2 improves complex driving behaviors such as merging, overtaking, emergency braking, and trafic-rule compliance.

Table 9: Multi-Ability results on Bench2Drive (V0.0.3) under base training set. ‘mmt’ refers multi-mode trajectory variant of VAD and <sup>†</sup> denotes the re-implementation. \* denotes expert feature distillation.
<table><tr><td rowspan="2">Method</td><td colspan="6">Multi-Ability(%) ↑</td></tr><tr><td>Merging ↑</td><td>Overtaking ↑</td><td>EmergencyBrake ↑</td><td>Give Way ↑</td><td>Traffic Sign ↑</td><td>Mean ↑</td></tr><tr><td>DriveAdapter* (Jia et al. 2023)</td><td>28.82</td><td>26.38</td><td>48.76</td><td>50.00</td><td>56.43</td><td>42.08</td></tr><tr><td>DriveDPÓ (Shang et al. 2025)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Raw2Drive (Yang et al. 2025)</td><td>43.35</td><td>51.11</td><td>60.00</td><td>50.00</td><td>62.26</td><td>53.34</td></tr><tr><td>DriveTrans* (Jia et al. 2025)</td><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>38.60</td></tr><tr><td>WoTE* (Li et al. 2025e)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DIVER (Song et al. 2026b)</td><td>35.08</td><td>25.09</td><td>41.09</td><td>50.00</td><td>59.21</td><td>42.09</td></tr><tr><td>MomADv2 (Ours)</td><td>43.71</td><td>34.39</td><td>50.54</td><td>50.00</td><td>69.82</td><td>49.02</td></tr><tr><td></td><td></td><td></td><td>21.67</td><td>10.00</td><td></td><td></td></tr><tr><td>UniAD-Base (Hu et al. 2023) VAD (Jiang et al. 2023)</td><td>14.10 8.11</td><td>17.78 24.44</td><td>18.64</td><td>20.00</td><td>14.21 19.15</td><td>15.55 18.07</td></tr><tr><td>GenAD (Zheng et al. 2024b)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MomAD (Song et al. 2025)</td><td>13.21</td><td>21.02</td><td>18.01</td><td>20.00</td><td>21.07</td><td>18.66</td></tr><tr><td>DIVER (Song et al. 2026b)</td><td>13.83</td><td>29.09</td><td>25.51</td><td>20.00</td><td>24.93</td><td>22.67</td></tr><tr><td>MomADv2 (Ours)</td><td>18.79</td><td>32.86</td><td>27.73</td><td>20.00</td><td>27.96</td><td>25.84</td></tr></table>

Table 10: Comparison of trajectory prediction consistency on the nuScenes validation set. TPC denotes Trajectory Prediction Consistency, where lower values indicate better temporal stability.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="3">TPC (m)↓</td></tr><tr><td>4s</td><td>5s</td><td>6s</td></tr><tr><td>UniAD (Hu et al. 2023)</td><td>CVPR&#x27;23</td><td>1.49</td><td>1.81</td><td>2.41</td></tr><tr><td>VAD (Jiang et al. 2023)</td><td>ICCV&#x27;23</td><td>1.55</td><td>1.73</td><td>2.17</td></tr><tr><td>SparseDrive (Sun et al. 2024)</td><td>ICRA&#x27;25</td><td>1.33</td><td>1.66</td><td>1.99</td></tr><tr><td>MomAD (Song et al. 2025)</td><td>CVPR&#x27;25</td><td>1.19</td><td>1.45</td><td>1.61</td></tr><tr><td>MomADv2 (Ours)</td><td></td><td>1.04</td><td>1.32</td><td>1.61</td></tr></table>

Table 11: Planning results for 3-second short-horizon planning on the nuScenes validation set.
<table><tr><td rowspan="2">Method</td><td colspan="4">L2 2 (m) ↓</td><td colspan="4">Col. Rate (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>UniAD (Hu et al. 2023)</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.10</td><td>0.15</td><td>0.61</td><td>0.29</td></tr><tr><td>VAD (Jiang et al. 2023)</td><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.11</td><td>0.24</td><td>0.42</td><td>0.26</td></tr><tr><td>DiffusionDrive (Liao et al. 2025)</td><td>0.29</td><td>0.58</td><td>0.96</td><td>0.61</td><td>0.02</td><td>0.05</td><td>0.22</td><td>0.09</td></tr><tr><td>DIVER (Song et al. 2026b)</td><td></td><td></td><td></td><td></td><td>0.01</td><td>0.05</td><td>0.15</td><td>0.07</td></tr><tr><td>FocalAD (Sun et al. 2026)</td><td>0.27</td><td>0.57</td><td>0.96</td><td>0.60</td><td>0.00</td><td>0.04</td><td>0.24</td><td>0.09</td></tr><tr><td>GuideFlow (Liu et al. 2025b)</td><td></td><td></td><td></td><td></td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td></tr><tr><td>SparseDrive (Sun et al. 2024)</td><td>0.30</td><td>0.58</td><td>0.96</td><td>0.61</td><td>0.01</td><td>0.05</td><td>0.23</td><td>0.10</td></tr><tr><td>LAW (Li et al. 2025c)</td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><td>GenAD (Zheng et al. 2024b)</td><td>0.28</td><td>0.49</td><td>0.78</td><td>0.52</td><td>0.08</td><td>0.14</td><td>0.34</td><td>0.19</td></tr><tr><td>Drive-OccWorld (Zheng et al. 2024a)</td><td>0.25</td><td>0.44</td><td>0.72</td><td>0.47</td><td>0.03</td><td>0.08</td><td>0.22</td><td>0.11</td></tr><tr><td>SSR (Li and Cui 2025)</td><td>0.19</td><td>0.36</td><td>0.62</td><td>0.39</td><td>0.10</td><td>0.10</td><td>0.24</td><td>0.15</td></tr><tr><td>Mom AD (Song et al. 2025)</td><td>0.31</td><td>0.57</td><td>0.91</td><td>0.60</td><td>0.01</td><td>0.05</td><td>0.22</td><td>0.09</td></tr><tr><td>Graph World (Song et al. 2026a)</td><td>0.29</td><td>0.55</td><td>0.89</td><td>0.57</td><td>0.00</td><td>0.04</td><td>0.20</td><td>0.08</td></tr><tr><td>MomADv2(Ours)</td><td>0.26</td><td>0.52</td><td>0.85</td><td>0.54</td><td>0.00</td><td>0.03</td><td>0.18</td><td>0.07</td></tr></table>

Trajectory Prediction Consistency in nuScenes. As shown in Table 10, we further evaluate temporal planning consistency on the nuScenes validation set using the Trajectory Prediction Consistency (TPC) metric introduced by MomAD.

Table 12: Ablation study of the two core modules on the nuScenes validation set under open-loop evaluation.
<table><tr><td></td><td></td><td colspan="3">L2 (m)↓</td><td colspan="3">Col. Rate (%)↓</td></tr><tr><td>SSM-Q</td><td>FM-Ref</td><td>4s</td><td>5s</td><td>6s</td><td>4s</td><td>5s</td><td>6s</td></tr><tr><td></td><td></td><td>1.67</td><td>1.98</td><td>2.45</td><td>0.87</td><td>1.54</td><td>2.33</td></tr><tr><td>√</td><td></td><td>1.33</td><td>1.84</td><td>2.43</td><td>0.80</td><td>1.44</td><td>2.12</td></tr><tr><td>√</td><td></td><td>1.32</td><td>1.82</td><td>2.40</td><td>0.72</td><td>1.33</td><td>2.03</td></tr></table>

Table 13: Ablation study of temporal memory modeling methods. Avg. L2 and Avg. Col. are evaluated on nuScenes, and EPDMS is evaluated on NAVSIMv2 navtest.
<table><tr><td>Temporal Modeling</td><td>Avg. L2↓</td><td>Avg. Col.↓</td><td>EPDMS↑</td><td>FPS↑</td></tr><tr><td>Concat + MLP</td><td>1.52</td><td>1.26</td><td>84.2</td><td>51.0</td></tr><tr><td>GRU</td><td>1.41</td><td>0.77</td><td>86.4</td><td>41.9</td></tr><tr><td>LSTM</td><td>1.34</td><td>0.80</td><td>86.7</td><td>39.6</td></tr><tr><td>Transformer</td><td>1.31</td><td>0.82</td><td>87.2</td><td>33.0</td></tr><tr><td>Vanilla SSM</td><td>1.25</td><td>0.81</td><td>87.0</td><td>43.6</td></tr><tr><td>Selective SSM / Mamba</td><td>1.21</td><td>0.76</td><td>87.9</td><td>43.0</td></tr></table>

Table 14: Ablation on solver choice on NAVSIMv2 navtest.
<table><tr><td>Solver</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>Comf.↑</td><td>EP↑</td><td>PDMS↑</td><td>FPS↑</td></tr><tr><td>Euler</td><td>99.0</td><td>97.1</td><td>95.5</td><td>100</td><td>83.2</td><td>89.9</td><td>43</td></tr><tr><td>Heun</td><td>98.5</td><td>96.7</td><td>95.2</td><td>100</td><td>82.9</td><td>89.0</td><td>38</td></tr></table>

MomADv2 achieves the lowest TPC errors of 1.04 m and 1.32 m at the 4- and 5-second horizons, corresponding to relative improvements of 12.6% and 9.0% over MomAD, respectively. At the more challenging 6-second horizon, MomADv2 obtains a TPC error of 1.61 m, matching the best-performing baseline. Overall, MomADv2 reduces the average TPC error by 6.6% compared with MomAD, demonstrating that selective temporal memory and flow-matching-based trajectory refinement improve temporal stability and mitigate planning drift over extended horizons.

3-second short-horizon planning on nuScenes. Table 11 reports the short-horizon planning results on the nuScenes validation set. MomADv2 achieves an average collision rate of 0.07%, matching the best performance of DIVER and GuideFlow, while maintaining zero collisions at the 1-second horizon. Compared with MomAD, MomADv2 reduces the average collision rate from 0.09% to 0.07% and the 3-second collision rate from 0.22% to 0.18%. It also improves over GraphWorld, reducing its average collision rate from 0.08% to 0.07%. In terms of trajectory accuracy, MomADv2 obtains an average $L _ { 2 }$ error of 0.54 m, improving upon MomAD and GraphWorld by 10.0% and 5.3%, respectively. Although SSR achieves the lowest displacement error, its average collision rate is notably higher than that of MomADv2. These results demonstrate that MomADv2 maintains competitive short-horizon trajectory accuracy while providing a favorable balance between planning accuracy and safety.

![](images/206571f037b04007d2110a7c15e8a68d1406be3b381ecdd60346b2937d498a70.jpg)  
Figure 5: Qualitative visualization of MomADv2 on NAVSIM across diverse driving scenarios. Compared with GuideFlow, MomADv2 generates smoother and more temporally consistent trajectories, better adheres to lane geometry and drivable-area constraints, and enables safer interactions with surrounding trafic participants.

## A.8 More Ablation Results

Roles of Diferent Modules in MomADv2. As shown in Table 12, we further evaluate the contributions of SSM-Q and FM-Ref on the nuScenes validation set under open-loop evaluation. Introducing SSM-Q consistently reduces both trajectory error and collision rate across all planning horizons, demonstrating the efectiveness of reliable temporal memory for long-horizon planning. Further incorporating FM-Ref yields additional improvements, achieving the lowest $L _ { 2 }$ errors of 1.32, 1.82, and 2.40 m, together with collision rates of 0.72%, 1.33%, and 2.03% at 4, 5, and 6 seconds, respectively. These results confirm that selective temporal memory and flow-matching residual refinement provide complementary benefits for accurate and safe long-horizon planning.

Efectiveness of Selective State-Space Memory Modeling. Table 13 compares diferent temporal memory modeling methods. Concat + MLP is eficient but lacks temporal modeling ability, while GRU/LSTM and Transformer improve planning performance at the cost of higher latency.

Vanilla SSM achieves a better eficiency-performance tradeof, and Selective SSM/Mamba further obtains the best planning accuracy and collision performance. This shows that selective state-space modeling is more suitable for reliable long-horizon planning memory.

Efect of Solver Choice. Table 14 compares diferent solvers in the Flow-Matching Trajectory Residual Refiner. Euler achieves better planning performance than Heun, improving PDMS from 89.0 to 89.9 while also increasing FPS from 38 to 43. This shows that the simple Euler solver is suficient for residual trajectory refinement and provides a better trade-of between planning quality and inference eficiency.

## A.9 More Qualitative Study of Planning Results

To further examine the planning behavior of MomADv2, we conduct qualitative studies on both NAVSIM and nuScenes. We select representative scenarios involving intersections, curved roads, dense trafic, and long-horizon maneuvering. Specifically, we provide two types of visualization: Visualization on NAVSIM, and Long-Horzion Planning Visualization on nuScenes.

Visualization on NAVSIM. We qualitatively compare MomADv2 with GuideFlow across diverse driving scenarios. As illustrated in Fig. 5, MomADv2 generates smoother and more temporally consistent trajectories, while better conforming to lane geometry and drivable-area constraints. In complex intersections and curved-road scenarios, MomADv2 also maintains safer interactions with surrounding trafic participants and avoids overly aggressive trajectory deviations. These results demonstrate that reliable temporal memory improves planning continuity and closed-loop decision sta-

![](images/13105e24527dec0c36fd3f33a564b613bc76590c7167c60db0ff7c2dde7d7dcf.jpg)  
Figure 6: Qualitative visualization of 6-second long-horizon planning by MomADv2 on nuScenes.

## bility.

Long-Horzion Planning Visualization on nuScenes. We further qualitatively examine the 6-second planning performance of MomADv2 on the nuScenes validation set. As illustrated in Fig. 6, MomADv2 generates smooth and temporally coherent trajectories across diverse driving scenarios, including turning maneuvers and dense trafic. By selectively retaining reliable historical context and suppressing stale or command-inconsistent memory, MomADv2 maintains stable planning intentions over extended horizons. The flowmatching trajectory residual refiner further mitigates local deviations and accumulated errors, leading to more accurate and consistent long-horizon planning.