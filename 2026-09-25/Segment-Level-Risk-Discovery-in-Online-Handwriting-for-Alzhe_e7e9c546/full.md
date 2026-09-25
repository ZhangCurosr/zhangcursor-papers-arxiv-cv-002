# Segment-Level Risk Discovery in Online Handwriting for Alzheimer’s Disease Detection

Changqing Gong, Huafeng Qin and Mounˆım A. El-Yacoubi

Abstract—Online handwriting provides a non-invasive and lowcost behavioral biomarker for Alzheimer’s disease (AD) detection, as it reflects both cognitive planning and fine motor control. Existing handwriting-based AD detection methods usually rely on global trajectory features or whole-sample representations, which can be strongly affected by individual writing style, task-specific variation, and acquisition noise. In this paper, we propose NormPaST-Risk, a healthy-normative Paper-Air selective trajectory state-space risk network for interpretable AD detection from online handwriting. Instead of treating the entire trajectory as a single holistic representation, our method reformulates AD handwriting detection as local disease-relevant segment discovery. Specifically, a multi-scale temporal encoder captures stroke dynamics at different temporal resolutions, while a selective Paper-Air state-space encoder models long-range handwriting progression and distinguishes on-paper motor execution from inair planning and transition behaviors. To explicitly characterize abnormal deviations, a healthy normative branch learns normal handwriting dynamics from healthy controls, and a task-aware multi-expert segment-risk module estimates segment-level AD risk calibrated by hidden-state changes and normative deviations. A weakly supervised segment-level objective further enables highrisk segment discovery without manual segment annotations. Experiments on the DARWIN benchmark demonstrate that the proposed framework achieves superior AD/HC classification performance compared with existing methods. Moreover, the discovered high-risk segments can be projected back to the original handwriting trajectory, providing interpretable evidence associated with AD-related handwriting variations.

Index Terms—Alzheimer’s disease, handwriting analysis, digital biomarkers, machine learning, computer-aided diagnosis.

## I. INTRODUCTION

Alzheimer’s disease (AD), the most common cause of dementia, is a progressive neurodegenerative disorder characterized by cognitive decline affecting memory, reasoning, and daily functioning [1]. Early identification is increasingly important for timely clinical intervention and disease management [2, 3]. However, established biomarker modalities, such as amyloid positron emission tomography (PET) and cerebrospinal fluid (CSF) assays, remain constrained in largescale screening by their cost, limited accessibility, and/or invasiveness [4, 5]. Neuropsychological assessments, such as the Mini-Mental State Examination (MMSE) and Montreal Cognitive Assessment (MoCA), are more accessible, but their performance can be influenced by cultural and educational factors, while repeated assessments may also be affected by practice effects and rater dependence [6, 7]. Therefore, there remains a strong need for accessible, objective, and low-cost approaches to support large-scale AD screening.

To this end, a variety of behavioral signals, including eye movements, speech, gait, and motor activity, have been investigated as potential digital markers of cognitive decline and dementia [8, 9]. Among these behavioral modalities, handwriting is particularly promising because it requires the coordinated engagement of cognitive, visuospatial, linguistic, and fine motor processes [10, 11]. With digital tablets, online handwriting tasks can capture fine-grained spatiotemporal dynamics, including pen trajectories, pressure, velocity, pauses, and in-air movements, providing an unobtrusive window into cognitive-motor function [12, 13]. Previous studies have reported alterations in handwriting kinematics and dynamics in individuals with AD and mild cognitive impairment, including changes in movement timing, fluency, pressure, and in-air behavior [11, 12, 13, 14].

Handwriting-based AD detection has traditionally relied on manually engineered kinematic, spatial, and pressure-related features combined with conventional machine learning classifiers [15, 16, 17]. With the development of deep learning, recent studies have increasingly explored learned handwriting representations, including sequential representation learning [12], convolutional neural networks [18], deep transfer learning [19], and multimodal Transformer architectures integrating 1D dynamic signals with 2D handwriting images [20]. Beyond task-level classification, recent research has also investigated more fine-grained handwriting characteristics, including the effects of word semantics and phonology on handwriting dynamics [14] and individual stroke-level representations for AD prediction [21]. These developments are consistent with broader evidence that AD-related changes can manifest across motor, visuospatial, and linguistic aspects of handwriting [11].

Building on these advances toward finer-grained handwriting analysis, an important question is whether disease-related handwriting evidence can be more effectively captured at the local trajectory level. Such evidence may not be uniformly distributed throughout an entire handwriting sequence, while global or task-level representations can be influenced by nondisease-related variations, including individual writing style, habitual pen usage, task-specific execution strategies, and acquisition noise. These factors may dilute or obscure subtle local abnormalities. Preserving local trajectory structure may therefore provide a more direct way to identify diseaserelevant patterns while reducing the influence of irrelevant global variations. Similar observations have also been reported in recent time-series studies, where modeling informative local patches or discriminative subsequences has shown advantages over relying exclusively on global representations [22, 23].

Learning such localized disease handwriting evidence, however, is challenging because only a single label is available for each complete handwriting trajectory, while no segmentlevel annotations indicate which local regions contain diseaserelated abnormalities. We therefore adopt weakly supervised segment-level risk learning, in which local segment risks are learned indirectly from trajectory-level labels through top-k aggregation. This encourages the model to focus on a subset of informative handwriting regions that are more relevant to disease discrimination. To address these challenges, we propose NormPaST-Risk, a segment-level trajectory modeling framework for interpretable handwriting-based AD detection. The model first partitions online handwriting trajectories into state-consistent local segments that preserve Paper/Air boundaries, and a multi-scale temporal encoder captures short- and mid-range kinematic dynamics. A Selective Paper-Air statespace encoder further models long-range dependencies while explicitly incorporating pen-contact states to distinguish onpaper motor execution from in-air planning and transition behavior. A healthy normative branch learns normal handwriting dynamics from healthy controls, while a task-aware multiexpert segment-risk module estimates local disease risks and identifies diagnostically informative segments. Together, these components integrate local abnormality discovery with longrange trajectory modeling for interpretable AD detection. The main contributions of this work are summarized as follows:

We introduce segment-level disease-risk modeling into online handwriting-based AD classification. Instead of treating all trajectory regions as equally informative, the proposed framework explicitly estimates segment-level disease relevance and uses the learned risks to adaptively weight local representations while retaining the complete trajectory context.

We propose NormPaST-Risk, a novel trajectory model that integrates multi-scale temporal encoding, Paper-Air state modulation, selective state-space modeling, a healthy normative branch, and task-aware multi-expert segment-risk learning. The framework captures both local and long-range handwriting dynamics while enabling interpretable localization of disease-relevant segments.

• Extensive experiments demonstrate the effectiveness of NormPaST-Risk. The proposed method consistently outperforms representative machine learning and sequencemodeling baselines, while ablation and segment-deletion studies further validate the contributions of its key components and the discriminative value of the discovered high-risk handwriting segments.

The remainder of this paper is organized as follows: Section 2 reviews related work, Section 3 presents NormPaST-Risk, Section 4 reports the experiments, and Section 5 concludes the paper.

## II. RELATED WORK

Dynamic handwriting analysis has been widely investigated as a non-invasive means of characterizing cognitive and motor alterations associated with neurodegenerative disorders [24, 11]. Early handwriting-based AD detection mainly relied on manually engineered kinematic and spatial descriptors, such as writing duration, in-air and on-paper time, velocity, acceleration, pressure, jerk, and trajectory statistics, followed by conventional machine learning classifiers [15, 16]. The release of the DARWIN dataset further enabled systematic evaluation across multiple handwriting and drawing tasks [16]. Although these handcrafted representations are interpretable, they typically summarize an entire task and may therefore obscure subtle local abnormalities.

More recent studies have shifted toward learned representations of handwriting dynamics. El-Yacoubi et al. introduced sequential representation learning to capture temporal handwriting behavior beyond global kinematic statistics [12]. Cilia et al. encoded online handwriting dynamics into synthetic images for deep transfer learning [19], while Dao et al. directly modeled handwriting sequences using a 1D convolutional neural network [25]. Erdogmus and Kabakus further explored convolutional representations derived from handwriting signals [18]. More recently, attention-based and multimodal architectures have been investigated: Kang et al. employed self-attention across multiple handwriting tasks [26], whereas Gong et al. integrated 1D dynamic signals with 2D handwriting images using a hybrid Transformer [20]. Despite increasingly powerful representation learning, these methods primarily focus on task- or sample-level prediction.

Fine-grained handwriting dynamics have consequently received increasing attention. Online handwriting contains both on-paper and in-air movements, whose temporal and kinematic characteristics can provide complementary information about handwriting behavior [11, 13]. Cilia et al. further investigated how linguistic factors such as word semantics and phonology affect handwriting dynamics in AD [14]. More importantly, Nardone et al. analyzed individual handwriting strokes rather than compressing each task into a single aggregated feature vector, demonstrating the benefit of preserving fine-grained movement information [21]. Explainability studies have also attempted to identify discriminative temporal patterns in online handwriting; for example, Sweidan et al. interpreted CNN predictions and revealed localized movement characteristics associated with AD [27]. Nevertheless, existing fine-grained approaches mainly rely on predefined stroke units or post-hoc interpretation rather than directly learning disease relevance for local trajectory segments.

Online handwriting is also inherently sequential, requiring the modeling of both short-term motor variations and long-range writing dependencies. State-space models (SSMs) provide an efficient framework for long-sequence modeling. Structured and diagonal SSMs enable efficient modeling of long-range dependencies [28, 29, 30], while selective SSMs introduce input-dependent state propagation to adapt sequence dynamics to the current input [31]. However, SSM-based modeling remains largely unexplored for handwriting-based

AD detection. In particular, existing handwriting approaches typically summarize Paper/Air behavior through temporal or kinematic descriptors rather than explicitly using pen-contact state to modulate long-range sequence dynamics.

Overall, handwriting-based AD detection has progressed from handcrafted task-level descriptors to deep sequential, attention-based, multimodal, and stroke-level representations. However, three limitations remain. First, most methods perform prediction using global task representations or predefined stroke units rather than explicitly learning disease-relevant local segments from trajectory-level supervision. Second, Paper/Air information is generally represented through derived features rather than explicitly modeled as contextual state governing trajectory dynamics. Third, localized disease-risk discovery, long-range sequence modeling, and deviations from healthy handwriting dynamics have not yet been jointly modeled within a unified framework. These limitations motivate NormPaST-Risk, which constructs state-consistent Paper-Air segments, models long-range trajectory dynamics using a selective state-space encoder, learns healthy normative dynamics, and discovers high-risk local handwriting segments under weak supervision.

## III. PROPOSED APPROACH

Handwriting-based Alzheimer’s disease (AD) detection is challenging because disease-related abnormalities may be sparse, localized, and task-dependent. A complete online handwriting trajectory contains substantial variations arising from individual writing style, task-specific execution, and acquisition noise, whereas discriminative AD-related evidence may appear only in short local segments, such as hesitation, unstable stroke execution, abnormal in-air movement, or irregular Paper-Air transitions. Therefore, instead of representing a handwriting sample solely by a global trajectory descriptor, we propose NormPaST-Risk, a healthy-normative Paper-Air selective trajectory state-space risk network for interpretable AD detection from online handwriting.

The proposed framework is motivated by four considerations. First, online handwriting is a long temporal sequence with multi-scale dynamics, requiring both short-range stroke dynamics and long-range writing progression to be modeled. Second, on-paper and in-air movements convey different behavioral information: Paper trajectories primarily reflect executed motor behavior, whereas Air trajectories may encode planning, hesitation, and cognitive-motor transitions. Third, local AD-related abnormalities can be more explicitly characterized by measuring deviations from healthy handwriting dynamics. Fourth, AD-related manifestations may vary across handwriting tasks and local trajectory segments, motivating task-aware multi-expert risk modeling to capture heterogeneous abnormal patterns and localize disease-relevant handwriting evidence. As shown in Fig.1, NormPaST-Risk consists of six components: (1) a multi-scale encoder, (2) a temporal segment aggregator, (3) a Selective Paper-Air SSM, (4) a healthy normative branch, (5) a task-aware multi-expert segment-risk module, and (6) a classification head with weakly supervised segment-risk learning.

## A. Input Representation

Given an online handwriting sample, we denote the raw trajectory as

$$
\mathbf { X } = \{ \mathbf { x } _ { t } \} _ { t = 1 } ^ { T _ { r } } , \quad \mathbf { x } _ { t } \in \mathbb { R } ^ { F } ,\tag{1}
$$

where $T _ { r }$ is the raw sequence length and $F$ is the feature dimension. We define the raw valid mask, paper mask, and air mask as m, m<sup>p</sup>, $\mathbf { m } ^ { a } \in \{ 0 , 1 \} ^ { T _ { r } }$ , where $m _ { t } = 1$ indicates a valid raw point, $m _ { t } ^ { p } = 1$ indicates a pen-down point, and $m _ { t } ^ { a } = 1$ indicates a pen-up point. For every valid point, the paper and air states are mutually exclusive.

## B. Multi-Scale Trajectory Encoder

AD-related handwriting abnormalities may occur at different temporal scales. For example, short receptive fields can capture local stroke instability, while longer receptive fields can capture hesitation and movement organization. Therefore, we first encode the raw trajectory with a multi-scale temporal encoder.

Each raw point is projected into a latent space and augmented with positional encoding:

$$
\begin{array} { r } { \mathbf { h } _ { t } ^ { 0 } = \phi _ { \mathrm { i n } } ( \mathbf { x } _ { t } ) + \mathbf { p } _ { t } , } \end{array}\tag{2}
$$

where $\phi _ { \mathrm { i n } } ( \cdot )$ is a linear projection and $\mathbf { p } _ { t }$ is the positional encoding. Then, Q temporal convolutional streams with different kernel sizes and dilation rates are applied:

$$
{ \bf H } ^ { q } = { \mathcal E } _ { q } ( { \bf H } ^ { 0 } ) , \quad q = 1 , \ldots , Q ,\tag{3}
$$

where $\mathbf { H } ^ { q } = \{ \mathbf { h } _ { t } ^ { q } \} _ { t = 1 } ^ { T _ { r } }$ and $\mathcal { E } _ { q } ( \cdot )$ denotes the q-th temporal stream. Each stream preserves the original sequence length.

To adaptively fuse different scales at each raw point, we compute scale weights:

$$
a _ { t , q } = \frac { \exp { ( s ( \mathbf { h } _ { t } ^ { q } ) ) } } { \sum _ { l = 1 } ^ { Q } \exp { ( s ( \mathbf { h } _ { t } ^ { l } ) ) } } ,\tag{4}
$$

where $s ( \cdot )$ is a learnable scoring function. The fused pointlevel representation is

$$
\bar { \mathbf { h } } _ { t } = \sum _ { q = 1 } ^ { Q } a _ { t , q } \mathbf { h } _ { t } ^ { q } .\tag{5}
$$

This operation produces a point-wise sequence

$$
\bar { \mathbf { H } } = \{ \bar { \mathbf { h } } _ { t } \} _ { t = 1 } ^ { T _ { r } } , \quad \bar { \mathbf { h } } _ { t } \in \mathbb { R } ^ { D } .\tag{6}
$$

## C. Temporal Segment Aggregation

Directly feeding all raw points into the sequence encoder can be inefficient and sensitive to point-level noise. However, naive fixed-stride downsampling may merge pen-down and pen-up states into the same token, which weakens the meaning of paper-air modeling. To address this issue, we introduce a state-consistent temporal segment aggregator.

Let $\gamma$ be the maximum number of raw points represented by one segment token. The aggregator partitions the valid raw trajectory into segment intervals:

$$
\begin{array} { r } { \mathcal { T } = \{ I _ { j } \} _ { j = 1 } ^ { T _ { f } } , \quad I _ { j } = [ s _ { j } , e _ { j } ) , } \end{array}\tag{7}
$$

![](images/3498901c4adf9908848e9796cc74c1e31d5a3fd576306be892361df398244aad.jpg)  
Fig. 1. Overall architecture of NormPaST-Risk. The framework consists of online handwriting representation, multi-scale trajectory encoding, state-aware segmentation and Paper-Air selective state-space modeling, normative-calibrated segment-risk discovery, and AD prediction.

where $T _ { f }$ is the number of segment tokens. Each interval satisfies

$$
1 \leq \ell _ { j } = e _ { j } - s _ { j } \leq \gamma .\tag{8}
$$

A Paper/Air state transition always terminates the current interval. Therefore, all raw points inside the same interval share the same pen state.

The raw trajectory is aggregated using mean pooling:

$$
\bar { \mathbf { x } } _ { j } = \frac { 1 } { \ell _ { j } } \sum _ { t = s _ { j } } ^ { e _ { j } - 1 } \mathbf { x } _ { t } .\tag{9}
$$

The fused point-level features are aggregated using the same segment layout:

$$
\mu _ { j } = \frac { 1 } { \ell _ { j } } \sum _ { t = s _ { j } } ^ { e _ { j } - 1 } \bar { \mathbf { h } } _ { t } .\tag{10}
$$

The same intervals are also used to aggregate scale weights and to construct segment-level valid, paper, and air masks. Thus, every downstream token is aligned with a true raw-point interval and belongs to exactly one Paper/Air state.

## D. Length-Aware Segment Token Fusion

Mean pooling provides a stable gradient path for all raw points inside a segment, but it can dilute salient local responses. Therefore, we additionally compute a channel-wise max-pooled token:

$$
\nu _ { j } = \operatorname* { m a x } _ { t \in I _ { j } } \bar { \mathbf { h } } _ { t } ,\tag{11}
$$

where the maximum is taken independently for each channel.

The mean token and max token are combined by a learnable channel-wise convex mixture:

$$
\mathbf { f } _ { j } = \pmb { \mu } _ { j } + \mathbf { w } \odot \left( \pmb { \nu } _ { j } - \pmb { \mu } _ { j } \right) , \quad \mathbf { w } = \sigma ( \pmb { \theta } ) ,\tag{12}
$$

where $\pmb { \theta } \in \mathbb { R } ^ { D }$ is a learnable parameter, $\mathbf { w } \in ( 0 , 1 ) ^ { D }$ , and ⊙ denotes element-wise multiplication. This design keeps the stable mean-pooling path while giving salient responses an additional length-independent path.

Because segments can have different true lengths, especially when a segment is terminated early by a Paper/Air transition, we encode the duration of each segment using two descriptors:

$$
\ell _ { j } ^ { \mathrm { l i n } } = \frac { \ell _ { j } } { \gamma } , \qquad \ell _ { j } ^ { \mathrm { l o g } } = \frac { \log ( 1 + \ell _ { j } ) } { \log ( 1 + \gamma ) } .\tag{13}
$$

The final segment token is

$$
\mathbf { z } _ { j } = \mathbf { f } _ { j } + \phi _ { \ell } ( [ \ell _ { j } ^ { \mathrm { l i n } } , \ell _ { j } ^ { \mathrm { l o g } } ] ) ,\tag{14}
$$

where $\phi _ { \ell } ( \cdot )$ is a learnable duration projection. The resulting segment-token sequence is

$$
\mathbf { Z } = \left\{ \mathbf { z } _ { j } \right\} _ { j = 1 } ^ { T _ { f } } , \quad \mathbf { z } _ { j } \in \mathbb { R } ^ { D } .\tag{15}
$$

## E. Selective Paper-Air State-Space Encoder

Dynamic handwriting inherently couples motor execution with higher-level cognitive processes such as planning and coordination, while on-paper and in-air movements exhibit distinct behavioral characteristics and have shown different sensitivity to cognitive impairment [24, 32]. This motivates us to treat the observed Paper/Air state as contextual information rather than as an ordinary kinematic feature. Inspired by feature-wise conditional modulation [33], we use layer-specific Paper-Air modulation to adapt intermediate trajectory representations to the observed pen state at different abstraction levels. The resulting state-aware sequence is subsequently modeled by selective state-space dynamics. Structured and diagonal SSMs have demonstrated strong capability for longrange sequence modeling [28, 29, 30], while selective SSMs further enable input-dependent state propagation through dynamically parameterized state-space components [31]. These observations motivate a hierarchical encoder in which $\mathrm { P a } -$ per/Air state modulates the representation space and selective SSMs adapt the temporal state evolution.

![](images/78e7bd3b4a76cd6003d30561bd47a70b9b32fc402ff730d5eb28a51b18c1de80.jpg)  
Fig. 2. Illustration of the selective diagonal SSM.

The segment-token sequence still contains long-range temporal dependencies. To model them efficiently, we use a Selective Paper-Air state-space encoder composed of L stacked selective SSM blocks. At each layer l, the current token representation is adaptively modulated according to its observed Paper/Air state:

$$
\tilde { \mathbf { h } } _ { j } ^ { l } = \left( \mathbf { h } _ { j } ^ { l - 1 } + \mathbf { e } _ { q _ { j } } ^ { l } \right) \odot \left( 0 . 7 5 + 0 . 5 \mathbf { g } _ { j } ^ { l } \right) ,\tag{16}
$$

where ${ \bf h } _ { i } ^ { ( 0 ) } = { \bf z } _ { j } , q _ { j }$ denotes the observed Paper/Air state, $\mathbf { e } _ { q _ { i } } ^ { l } \in \mathbb { R } ^ { \beta }$ is its layer-specific learnable latent embedding, and $\mathbf { g } _ { j } ^ { l ^ { \prime } } \in ( 0 , 1 ) ^ { D }$ is a layer-specific Paper-Air modulation gate predicted from the concatenation of $\mathbf { \bar { h } } _ { j } ^ { l - 1 }$ and ${ \mathbf e } _ { { q } _ { j } } ^ { l }$ . Although the observed state $q _ { j }$ remains unchanged across layers, its latent representation and modulation effect are learned independently at different representation levels.

Inside the l-th selective diagonal SSM block, a gated input token is computed as

$$
\mathbf { v } _ { j } ^ { l } = \mathbf { o } _ { j } ^ { l } \odot \sigma ( \mathbf { g } _ { j } ^ { s , l } ) , \quad [ \mathbf { o } _ { j } ^ { l } ; \mathbf { g } _ { j } ^ { s , l } ] = \phi _ { \mathrm { i n } } ^ { s , l } \left( \mathrm { L N } ( \tilde { \mathbf { h } } _ { j } ^ { l } ) \right) .\tag{17}
$$

where $\phi _ { \mathrm { i n } } ^ { s , l }$ denotes a learnable linear input projection in the l-th selective SSM layer.

The current token also generates channel-wise selection factors:

$$
[ \mathbf { b } _ { j } ^ { l } ; \mathbf { c } _ { j } ^ { l } ; \delta _ { j } ^ { l } ] = \phi _ { \mathrm { s e l } } ^ { l } \left( \mathrm { L N } ( \tilde { \mathbf { h } } _ { j } ^ { l } ) \right) ,\tag{18}
$$

where $\mathbf { b } _ { j } ^ { l } , \mathbf { c } _ { j } ^ { l } , \delta _ { j } ^ { l } \in \mathbb { R } ^ { D } , \phi _ { \mathrm { s e l } } ^ { l }$ denotes a learnable linear projection in the l-th selective SSM layer. The selection factors are transformed as

$$
{ \bf b } _ { j } ^ { + , l } = 2 \sigma ( { \bf b } _ { j } ^ { l } ) , \quad { \bf c } _ { j } ^ { + , l } = 2 \sigma ( { \bf c } _ { j } ^ { l } ) ,\tag{19}
$$

$$
\pmb { \Delta } _ { j } ^ { l } = \mathrm { S o f t p l u s } \left( \pmb { \eta } _ { \Delta } ^ { l } + 0 . 5 \operatorname { t a n h } ( \pmb { \delta } _ { j } ^ { l } ) \right) ,\tag{20}
$$

where $\pmb { \eta } _ { \Delta } ^ { l } \in \mathbb { R } ^ { D }$ is a learnable channel-wise base parameter for the SSM step size. The diagonal transition parameter is constrained to be stable:

$$
\mathbf { A } ^ { l } = - \exp ( \mathbf { A } _ { \mathrm { l o g } } ^ { l } ) .\tag{21}
$$

For channel $d ,$ the SSM state is $\mathbf { h } _ { j , d } ^ { \mathrm { s s m } , l } \in \mathbb { R } ^ { N _ { s } }$ , where $N _ { s }$ is the state dimension. The selective diagonal recurrence is

$$
\mathbf { h } _ { j , d } ^ { \mathrm { s s m } , l } = \exp \left( \mathbf { A } _ { d } ^ { l } \Delta _ { j , d } ^ { l } \right) \odot \mathbf { h } _ { j - 1 , d } ^ { \mathrm { s s m } , l } + \left( b _ { j , d } ^ { + , l } \mathbf { B } _ { d } ^ { l } \right) v _ { j , d } ^ { l } ,\tag{22}
$$

$$
y _ { j , d } ^ { l } = \left( c _ { j , d } ^ { + , l } \mathbf { C } _ { d } ^ { l } \right) ^ { \top } \mathbf { h } _ { j , d } ^ { \mathrm { s s m } , l } + D _ { d } ^ { l } v _ { j , d } ^ { l } ,\tag{23}
$$

where $\mathbf { A } _ { d } ^ { l } , \mathbf { B } _ { d } ^ { l } , \mathbf { C } _ { d } ^ { l } \in \mathbb { R } ^ { N _ { s } }$ are channel-wise SSM parameters, and $D _ { d } ^ { l }$ is a learnable direct feedthrough parameter.

Finally, the channel-wise SSM responses are aggregated and projected to obtain the SSM output $\mathbf { y } _ { j } ^ { l } \in \mathbb { R } ^ { D }$ . The resulting output is then combined with the input of the current layer through a residual connection followed by layer normalization. The operations of the l-th Selective Paper-Air diagonal SSM layer can be summarized as

$$
\mathbf { h } _ { j } ^ { l } = \mathrm { L N } \left( \mathbf { h } _ { j } ^ { l - 1 } + \mathcal { D } _ { \mathrm { S S M } } ^ { l } \left( \mathcal { M } _ { \mathrm { P A } } ^ { l } \left( \mathbf { h } _ { j } ^ { l - 1 } , q _ { j } \right) \right) \right) .\tag{24}
$$

Here, $\mathcal { M } _ { \mathrm { P A } } ^ { l }$ represents the Paper-Air modulation, while $\mathcal { D } _ { \mathrm { S S M } } ^ { l }$ represents the selective diagona SSM transformation, including state propagation, input writing, state readout, and output projection.

After the final layer, we obtain the context-aware state sequence

$$
\mathbf { S } = \{ \mathbf { s } _ { j } \} _ { j = 1 } ^ { T _ { f } } , \qquad \mathbf { s } _ { j } = \mathbf { h } _ { j } ^ { L } \in \mathbb { R } ^ { D } .\tag{25}
$$

To capture abrupt hidden-state changes, we compute a normalized state-change score:

$$
\chi _ { j } = { \frac { \left\| \mathbf { s } _ { j } - \mathbf { s } _ { j - 1 } \right\| _ { 2 } } { \operatorname* { m a x } \left( \operatorname* { m a x } _ { k } \left\| \mathbf { s } _ { k } - \mathbf { s } _ { k - 1 } \right\| _ { 2 } , \epsilon \right) } } , \qquad j \geq 2 ,\tag{26}
$$

This score is used only as a calibration factor for learned segment risk, rather than as an independent disease indicator.

## F. Healthy Normative Branch

To explicitly characterize healthy handwriting patterns, we introduce a healthy normative branch, as shown in Fig.3 (a). Given the contextual state sequence S, the branch estimates the subsequent aggregated trajectory token as

$$
\hat { \bar { \mathbf { x } } } _ { j + 1 } = f _ { \mathrm { n o r m } } ( \mathbf { s } _ { j } ) , \quad j = 1 , \ldots , T _ { f } - 1 .\tag{27}
$$

Here, ${ \bf s } _ { j }$ is the context-aware state representation of the j-th segment, and the estimation target is the subsequent aggregated trajectory token $\bar { \mathbf { x } } _ { j + 1 }$ . To ensure that $f _ { \mathrm { n o r m } }$ captures healthy handwriting dynamics, its prediction objective is optimized exclusively using HC samples during training.

For valid consecutive segments, the normative deviation is computed as

$$
d _ { j + 1 } = \frac { 1 } { F } \left. \hat { \bar { \mathbf { x } } } _ { j + 1 } - \bar { \mathbf { x } } _ { j + 1 } \right. _ { 2 } ^ { 2 } ,\tag{28}
$$

where $F$ denotes the dimensionality of the trajectory feature vector. To align the deviation sequence with the segment-token sequence, we set $d _ { 1 } = 0$

The segment-wise deviation is normalized within each sample as

$$
\tilde { d } _ { j } = \frac { d _ { j } } { \operatorname* { m a x } \left( \operatorname* { m a x } d _ { k } , \epsilon \right) } ,\tag{29}
$$

where $\epsilon > 0$ ensures numerical stability.

The sample-level normative score is computed from the unnormalized deviation:

$$
d _ { \mathrm { s c o r e } } = \frac { \sum _ { j = 1 } ^ { T _ { f } - 1 } m _ { j } ^ { \mathrm { s t e p } } d _ { j + 1 } } { \sum _ { j = 1 } ^ { T _ { f } - 1 } m _ { j } ^ { \mathrm { s t e p } } } ,\tag{30}
$$

where $m _ { j } ^ { \mathrm { s t e p } } = 1$ means that both tokens $j$ and $j + 1$ are valid.

The normative prediction objective is optimized only using healthy control samples. Therefore, the normative predictor learns segment-level trajectory regularities characteristic of healthy handwriting without directly fitting AD samples. During inference, deviations from these healthy regularities provide auxiliary abnormality cues for segment-risk calibration and subject-level classification.

## G. Task-Aware Multi-Expert Segment-Risk Modeling

Different handwriting tasks may emphasize different cognitive and motor functions. Therefore, we use a task-aware multi-expert module to estimate token-level AD risk.

As shown in Fig.3 (b), each expert produces a segment-risk logit:

$$
\begin{array} { r } { r _ { j } ^ { e } = f _ { e } ( \mathbf { s } _ { j } ) , \quad e = 1 , \ldots , E , } \end{array}\tag{31}
$$

where E is the number of experts. A global state vector is obtained by masked mean pooling:

$$
\mathbf { s } _ { g } = \frac { \sum _ { j = 1 } ^ { T _ { f } } \bar { m } _ { j } \mathbf { s } _ { j } } { \sum _ { j = 1 } ^ { T _ { f } } \bar { m } _ { j } } ,\tag{32}
$$

where $\bar { m } _ { j }$ is the segment-level valid mask. The router input is the concatenation of the global state and the task embedding:

$$
\mathbf { z } _ { r } = [ \mathbf { s } _ { g } ; \mathbf { e } _ { \tau } ] ,\tag{33}
$$

![](images/e2f5995426b62457ec11a20ae5d6ff7c096659bfa72afc2de2b73b92f261d01e.jpg)

![](images/bfe46ad0bb8125c9643889c81c9cdf8d93c35404c544345249e60975d14df48c.jpg)  
Fig. 3. Illustration of (a) the Healthy Normative Branch and (b) the Task-Aware Multi-Expert Segment Risk module.

where τ denotes the task identity. The router produces expert logits:

$$
\rho = \frac { R ( { \bf z } _ { r } ) } { \tau _ { r } } ,\tag{34}
$$

where $\tau _ { r }$ is the router temperature. The router logits are converted into normalized expert weights using a softmax function:

$$
\pi _ { e } = \frac { \exp ( \rho _ { e } ) } { \sum _ { l = 1 } ^ { E } \exp ( \rho _ { l } ) } , \quad e = 1 , \ldots , E .\tag{35}
$$

The raw segment-risk logit is

$$
r _ { j } = \sum _ { e = 1 } ^ { E } \pi _ { e } r _ { j } ^ { e } .\tag{36}
$$

The final segment-risk logit is calibrated by hidden-state change and healthy-normative deviation:

$$
\tilde { r } _ { j } = r _ { j } \left( 1 + \lambda _ { c } \chi _ { j } + \lambda _ { n } \tilde { d } _ { j } \right) ,\tag{37}
$$

where $\lambda _ { c }$ and $\lambda _ { n }$ control the contributions of state-change and normative-deviation calibration, respectively. The normative deviation is detached in this calibration path, so that AD labels do not directly optimize the healthy-only normative predictor through the segment-risk objective.

The segment-level AD probability is

$$
p _ { j } = \sigma ( \tilde { r } _ { j } ) .\tag{38}
$$

The temporal risk weights are computed by masked softmax:

$$
\alpha _ { j } = \frac { \bar { m } _ { j } \exp ( \tilde { r } _ { j } ) } { \sum _ { k = 1 } ^ { T _ { f } } \bar { m } _ { k } \exp ( \tilde { r } _ { k } ) } .\tag{39}
$$

The handwriting-level risk representation is obtained by riskweighted pooling:

$$
\mathbf { s } _ { \mathrm { r i s k } } = \sum _ { j = 1 } ^ { T _ { f } } \alpha _ { j } \mathbf { s } _ { j } .\tag{40}
$$

## H. Final Classification

To further incorporate the overall deviation from healthy handwriting patterns, the sample-level normative score is projected into the same feature space and added to $\bf { s } _ { \mathrm { { r i s k } } } .$

$$
\mathbf { s } _ { \mathrm { r i s k } } ^ { \prime } = \mathbf { s } _ { \mathrm { r i s k } } + \phi _ { n } \left( \log ( 1 + d _ { \mathrm { s c o r e } } ) \right) ,\tag{41}
$$

where $\phi _ { n }$ is a learnable projection. The normative score is detached before this operation to prevent the classification objective from directly optimizing the healthy normative predictor.

Finally, the AD/HC prediction is

$$
\hat { \mathbf { y } } = \operatorname { S o f t m a x } \left( \mathbf { W } _ { c } \mathbf { s } _ { \mathrm { r i s k } } ^ { \prime } + \mathbf { b } _ { c } \right) .\tag{42}
$$

## I. Training Objective

The model is trained with a compact objective:

$$
\mathcal { L } = \mathcal { L } _ { C E } + \lambda _ { s e g } \mathcal { L } _ { s e g } + \lambda _ { n o r m } \mathcal { L } _ { n o r m } .\tag{43}
$$

where $\lambda _ { s e g } ~ = ~ 0 . 1$ and $\lambda _ { n o r m } ~ = ~ 0 . 2$ . These weights keep the auxiliary segment-level and normative objectives effective without overwhelming the primary classification loss, thereby maintaining a balanced and stable optimization.

The sample-level classification loss is the standard crossentropy loss:

$$
\mathcal { L } _ { C E } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \hat { y } _ { i , y _ { i } } .\tag{44}
$$

Since no manual segment-level annotations are available, we use weakly supervised segment-level multiple-instance learning. For sample $i ,$ let $\begin{array} { r } { L _ { i } = \sum _ { j = 1 } ^ { T _ { f } } \bar { m } _ { i , j } } \end{array}$ be the number of valid segment tokens. The number of selected top-risk tokens is

$$
K _ { i } = \operatorname* { m a x } \left( 1 , \left\lceil \eta L _ { i } \right\rceil \right) ,\tag{45}
$$

where η is the top-k ratio. The sample-level logit is

$$
\bar { r } _ { i } = \frac { 1 } { K _ { i } } \sum _ { j \in \mathrm { T o p K } _ { i } } \tilde { r } _ { i , j } .\tag{46}
$$

The segment MIL loss is

$$
\mathcal { L } _ { s e g } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ y _ { i } \log \sigma ( \bar { r } _ { i } ) + ( 1 - y _ { i } ) \log \left( 1 - \sigma ( \bar { r } _ { i } ) \right) \right] .\tag{47}
$$

This loss encourages AD samples to contain at least some high-risk segments, while suppressing strong AD-like segment responses in HC samples.

The healthy normative loss is applied only to healthy control samples:

$$
\mathcal { L } _ { n o r m } = \frac { \sum _ { i \in { \mathcal { B } _ { H C } } } \sum _ { j = 1 } ^ { T _ { f } - 1 } m _ { i , j } ^ { \mathrm { s t e p } } \frac { 1 } { F } \left. \hat { \bar { \mathbf { x } } } _ { i , j + 1 } - \bar { \mathbf { x } } _ { i , j + 1 } \right. _ { 2 } ^ { 2 } } { \sum _ { i \in { \mathcal { B } _ { H C } } } \sum _ { j = 1 } ^ { T _ { f } - 1 } m _ { i , j } ^ { \mathrm { s t e p } } } ,\tag{48}
$$

where $B _ { H C }$ is the set of healthy control samples in the batch. If no healthy-control sample is present in a mini-batch, $\mathcal { L } _ { n o r m }$ is set to zero. AD samples do not contribute to this loss, so the normative predictor learns healthy segment-level trajectory regularities rather than disease-specific patterns.

## IV. EXPERIMENTS

## A. Dataset

We used the DARWIN-RAW dataset [16, 19], a public online handwriting benchmark for Alzheimer’s disease (AD) detection. The dataset contains recordings from 174 participants, including 89 AD patients and 85 healthy controls (HC). Each participant completed 25 handwriting tasks designed for early AD detection, covering memory and dictation (M), copying (C), and graphic (G) tasks. The handwriting data were collected using a Wacom Bamboo tablet at a sampling rate of 200 Hz. In our experiments, each participant-task recording is treated as one trajectory sample with a subjectlevel AD/HC label. From the raw signals, we derive six dynamic kinematic features, including velocity, acceleration, jerk, curvature, pressure variation, and angular velocity, which are used as the input sequence of our model.

## B. Experimental Setup

We evaluated the proposed method using five-fold crossvalidation on the DARWIN-RAW dataset. In each fold, all samples from the same participant were assigned to the same split to avoid subject information leakage. Each participanttask recording was treated as one trajectory sample, and all 25 handwriting tasks were used for joint training and evaluation.

For a fair comparison, all baseline models were evaluated under the same subject-level splits, input features, and evaluation protocol as the proposed method. We compared our method with classical machine learning and deep sequence modeling baselines, Random Forest (RF)[34], CNN-1D(AD)[25], BiLSTM[35], Reservoir Computing(AD)[36], Transformer[37], Mamba[31], HSDA-MS(AD)[20]. During training, we used the AdamW optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of $3 \times 1 0 ^ { - 4 }$ . The batch size was set to 8, and the maximum number of training epochs was set to 100. A cosine annealing learning rate scheduler was adopted, with the minimum learning rate set to 1% of the initial learning rate. Early stopping was applied when the validation performance did not improve for 10 consecutive epochs. The validation set is randomly split from the training data, accounting for 10% of the full dataset. All experiments were implemented in PyTorch and conducted on NVIDIA™ GPUs.

We use balanced accuracy as the primary metric to account for potential class imbalance between AD and HC participants. We further report Precision-AD, F1-AD, sensitivity, and specificity for a comprehensive evaluation. We additionally report the area under the receiver operating characteristic curve (AUC) to evaluate the model’s discriminative ability.

The main architectural configurations of the proposed NormPaST-Risk model are summarized in Table I.

TABLE I  
ARCHITECTURAL SETTINGS OF THE PROPOSED NORMPAST-RISK MODEL.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Input Dimension</td><td>6</td></tr><tr><td>Hidden Dimension</td><td>96</td></tr><tr><td>Multi-scale Kernel Sizes</td><td>[3, 7]</td></tr><tr><td>Segment Token Stride</td><td>128 points</td></tr><tr><td>Number of SSM Layers</td><td>4</td></tr><tr><td>SSM State Dimension</td><td>16</td></tr><tr><td>Segment-risk Experts</td><td>8</td></tr><tr><td>Expert Hidden Dimension</td><td>48</td></tr><tr><td>Router Hidden Dimension</td><td>96</td></tr><tr><td>State-change Weight  $( \lambda _ { c } )$ </td><td>0.5</td></tr><tr><td>Normative Risk Weight  $( \lambda _ { n } )$ </td><td>0.3</td></tr><tr><td>Dropout Rate</td><td>0.3</td></tr></table>

## C. Comparison with Representative Baselines

To evaluate the effectiveness of the proposed NormPaST-Risk, we compare it with a diverse set of representative methods, including a classical machine learning model (Random Forest), recurrent and convolutional neural networks (BiLSTM and CNN-1D), reservoir computing, a Transformerbased model, the selective state-space model Mamba, and the handwriting-oriented HSDA-MS. All methods are evaluated under the same experimental protocol, and the results are reported as mean ± standard deviation (%) across five folds. The best result for each metric is highlighted in bold, while the second-best result is underlined.

As shown in Table II, NormPaST-Risk achieves the best performance on five of the six evaluation metrics, including an AUC of 95.84%, a balanced accuracy of 92.55%, a Precision-AD of 92.43%, an F1-AD score of 92.74%, and a specificity of 91.77%. Compared with the second-best result for each metric, the proposed method improves balanced accuracy, Precision-AD, F1-AD, and specificity by 3.50, 6.30, 2.31, and 9.42 percentage points, respectively, while also achieving a slightly higher AUC.

It is worth noting that the highest sensitivity does not necessarily correspond to the best overall classification performance. Mamba achieves the highest sensitivity of 97.71%, while its specificity is comparatively lower at 79.93%. Similarly, Transformer obtains a sensitivity of 95.49% with a specificity of 82.35%, suggesting a trade-off between detecting AD participants and correctly identifying healthy controls. This pattern is also reflected in their Precision-AD values of 84.59% and 85.39%, respectively, which are not correspondingly high despite their high sensitivity. HSDA-MS shows a similar pattern, achieving 95.75% sensitivity and 82.35% specificity, indicating that the performance between the two classes remains somewhat unbalanced.

In contrast, NormPaST-Risk achieves 93.33% sensitivity together with 91.77% specificity, resulting in the highest balanced accuracy of 92.55%. This more balanced sensitivity– specificity trade-off suggests that the proposed method can effectively identify AD participants while maintaining reliable recognition of healthy controls. Such balanced discrimination is desirable for AD detection, as excessive false-positive predictions may limit the practical reliability of a classification system. These results suggest that relying mainly on a global sequence representation may capture discriminative but less disease-specific variations, whereas identifying localized abnormal handwriting segments can reduce the influence of irrelevant trajectory portions and provide more robust ADrelated handwriting evidence.

The performance differences among the sequence models further demonstrate the effectiveness of the proposed architecture. Although BiLSTM, CNN-1D, Reservoir, Transformer, and Mamba are capable of modeling temporal handwriting signals to different extents, their F1-AD scores remain between 77.78% and 90.43%, which are lower than the 92.74% achieved by NormPaST-Risk. Moreover, NormPaST-Risk achieves the highest Precision-AD of 92.43%, indicating that the proposed model not only identifies AD participants more effectively but also produces fewer false-positive AD predictions. These results suggest that generic temporal modeling alone is insufficient to fully characterize the AD-related abnormalities embedded in online handwriting trajectories.

A key advantage of NormPaST-Risk is that it incorporates handwriting-specific structural priors rather than treating the trajectory as a homogeneous sequence. First, the multi-scale temporal encoder captures handwriting dynamics at different temporal resolutions, enabling the model to characterize both short-term kinematic variations and longer motion patterns. This provides richer local representations than conventional CNN-1D or generic sequence models relying on a single temporal modeling mechanism.

Second, NormPaST-Risk explicitly distinguishes on-paper and in-air behaviors through Paper–Air state-aware modeling. These two states characterize different aspects of the handwriting process: on-paper trajectories primarily reflect writing execution, whereas in-air movements are associated with spatial transitions and movement preparation. By preserving these heterogeneous states rather than modeling them as an undifferentiated sequence, the proposed framework maintains state-specific dynamics and captures variations across writing execution and transitional behaviors. This provides a more handwriting-specific representation than generic sequence architectures such as BiLSTM, Transformer, and Mamba.

Third, the selective state-space encoder complements local multi-scale representations by capturing long-range dependencies across the handwriting sequence. Rather than relying exclusively on local convolution or recurrent accumulation, it enables interactions between temporally distant segments and characterizes changes in handwriting dynamics over longer temporal ranges. This is particularly useful when informative abnormalities occur intermittently rather than continuously throughout a trajectory.

In addition, the Healthy Normative Branch provides an explicit reference for normal handwriting dynamics. Rather than relying solely on direct AD–HC discrimination, it models regular trajectory patterns from HC samples and quantifies deviations from the learned healthy dynamics. This normative deviation provides complementary information to the discriminative representation and encourages the model to focus on handwriting patterns that deviate from normal progression rather than relying only on subject or task variation.

TABLE II  
COMPARISON WITH REPRESENTATIVE BASELINE METHODS FOR ONLINE HANDWRITING-BASED AD DETECTION. RESULTS ARE REPORTED AS MEAN ± STANDARD DEVIATION (%) OVER FIVE FOLDS. THE BEST RESULT IS SHOWN IN BOLD AND THE SECOND-BEST RESULT IS UNDERLINED.
<table><tr><td>Model</td><td>BAcc</td><td>Precision-AD</td><td>F1-AD</td><td>Sensitivity</td><td>Specificity</td><td>AUC</td></tr><tr><td>CNN-1D</td><td> $7 4 . 8 0 \pm 8 . 2 8$ </td><td> $7 1 . 2 1 \pm 5 . 9 2$ </td><td> $7 7 . 7 8 \pm 9 . 2 2$ </td><td> $8 6 . 6 7 \pm 1 6 . 0 1$ </td><td> $6 2 . 9 4 \pm 9 . 6 7$ </td><td> $8 8 . 3 9 \pm 9 . 9 4$ </td></tr><tr><td>BiLSTM</td><td> $7 6 . 9 0 \pm 5 . 4 7$ </td><td> $7 7 . 8 0 \pm 8 . 6 5$ </td><td> $7 8 . 0 5 \pm 4 . 0 8$ </td><td> $7 9 . 6 7 \pm 9 . 6 6$ </td><td> $7 4 . 1 2 \pm 1 6 . 4 3$ </td><td> $8 4 . 8 1 \pm 4 . 0 8$ </td></tr><tr><td>RF</td><td> $7 7 . 8 8 \pm 1 0 . 7 3$ </td><td> $7 8 . 2 4 \pm 1 4 . 4 5$ </td><td> $7 9 . 7 1 \pm 1 0 . 0 0$ </td><td> $8 3 . 9 9 \pm 1 4 . 8 0$ </td><td> $7 1 . 7 6 \pm 2 2 . 5 5$ </td><td> $8 5 . 5 0 \pm 7 . 1 9$ </td></tr><tr><td>Reservoir</td><td> $8 1 . 1 6 \pm 5 . 0 0$ </td><td> $7 6 . 5 6 \pm 3 . 5 4$ </td><td> $8 3 . 5 7 \pm 4 . 9 9$ </td><td> $9 2 . 0 9 \pm 7 . 5 3$ </td><td> $7 0 . 2 2 \pm 4 . 2 4$ </td><td> $9 0 . 2 6 \pm 4 . 0 7$ </td></tr><tr><td>Transformer</td><td> $8 8 . 9 2 \pm 6 . 5 9$ </td><td> $8 5 . 3 9 \pm 8 . 2 5$ </td><td> $8 9 . 9 6 \pm 6 . 2 6$ </td><td> $9 5 . 4 9 \pm 7 . 2 6$ </td><td> $\underline { { 8 2 . 3 5 } } \pm 1 0 . 1 9$ </td><td> $9 2 . 7 3 \pm 5 . 7 2$ </td></tr><tr><td>Mamba</td><td> $8 8 . 8 2 \pm 6 . 6 1$ </td><td> $8 4 . 5 9 \pm 9 . 3 1$ </td><td> $9 0 . 4 3 \pm 5 . 3 1$ </td><td> ${ \bf 9 7 . 7 1 } \pm 3 . 1 3$ </td><td> $7 9 . 9 3 \pm 1 3 . 4 5$ </td><td> $9 3 . 6 2 \pm 4 . 2 1$ </td></tr><tr><td>HSDA-MS</td><td> $\underline { { 8 9 . 0 5 \pm 6 . 0 8 } }$ </td><td> $\underline { { 8 6 . 1 3 } } \pm 9 . 7 2$ </td><td> $9 0 . 3 6 \pm 4 . 5 6$ </td><td> $9 5 . 7 5 \pm 2 . 8 4$ </td><td> $\underline { { 8 2 . 3 5 } } \pm 1 4 . 4 1$ </td><td> $9 3 . 8 8 \pm 4 . 1 5$ </td></tr><tr><td>Ours</td><td> ${ \bf 9 2 . 5 5 \pm 4 . 3 4 }$ </td><td> $9 2 . 4 3 \pm 5 . 7 2$ </td><td> ${ \bf 9 2 . 7 4 \pm 4 . 2 3 }$ </td><td> $9 3 . 3 3 \pm 6 . 0 9$ </td><td> ${ \bf 9 1 . 7 7 \pm 6 . 7 1 }$ </td><td> ${ \bf 9 5 . 8 4 } \pm \bf { 2 . 9 0 }$ </td></tr></table>

More importantly, NormPaST-Risk does not rely solely on a global trajectory representation for classification. The multiexpert segment-risk module estimates AD-related predictive evidence at the segment level and emphasizes locally informative regions. By incorporating normative deviation into segment-risk calibration, the model prioritizes discriminative segments while accounting for their deviation from learned healthy dynamics, thereby reducing the influence of less informative trajectory portions.

Overall, NormPaST-Risk combines multi-scale dynamics, Paper-Air state modeling, healthy normative deviation, and segment-level risk discovery to capture handwriting-specific AD evidence more effectively than generic temporal models, resulting in more discriminative and balanced participant-level predictions.

## D. Ablation Study

To investigate the contribution of each component in NormPaST-Risk, we conduct a series of ablation experiments under the same five-fold evaluation protocol. For each variant, only the corresponding component is removed or replaced, while the remaining network architecture and training settings are kept unchanged. The ablation results in Table III demonstrate that all major components contribute to the overall performance of NormPaST-Risk. Removing the segment-risk module causes the largest degradation in BAcc and F1-AD, decreasing them from 92.55% and 92.74% to 88.56% and 88.46%, respectively. Sensitivity also drops from 93.33% to 87.71%. These results confirm that explicitly identifying and aggregating disease-relevant local segments is more effective than treating all trajectory regions equally.

TABLE III  
ABLATION STUDY OF THE PROPOSED NORMPAST-RISK MODEL.
<table><tr><td>Variant</td><td>BAcc</td><td>F1-AD</td><td>Sens.</td><td>Spec.</td></tr><tr><td>w/o Segment-Risk</td><td>88.56</td><td>88.46</td><td>87.71</td><td>89.41</td></tr><tr><td>w/o Paper-Air Modulation</td><td>89.67</td><td>89.59</td><td>88.76</td><td>90.59</td></tr><tr><td>w/o Selective SSM</td><td>89.74</td><td>89.33</td><td>85.36</td><td>94.12</td></tr><tr><td>w/o Multi-scale Encoder</td><td>90.23</td><td>90.46</td><td>91.04</td><td>89.41</td></tr><tr><td>w/o Healthy Normative Branch</td><td>90.95</td><td>90.10</td><td>89.93</td><td>91.97</td></tr><tr><td>Full NormPaST-Risk</td><td>92.55</td><td>92.74</td><td>93.33</td><td>91.77</td></tr></table>

![](images/b8cc986f22e082459b4965ee756f9de68d3682ab32c40caa3d9a5ec0eb6af9ec.jpg)  
(a)

![](images/13850a6928d07adc3095d3528916b8fb1d4d7b291111f5ba9e965a212c0358b2.jpg)  
(b)  
Fig. 4. Performance sensitivity: (a) the number of experts and (b) the top-k ratio η.

Removing Paper-Air modulation reduces BAcc to 89.67% and sensitivity to 88.76%, indicating that explicitly distinguishing on-paper motor execution from in-air planning behavior provides useful state-dependent information for AD recognition. Similarly, removing the Selective SSM decreases BAcc to 89.74% and produces the largest reduction in sensitivity, from 93.33% to 85.36%, despite achieving a higher specificity of 94.12%. This suggests that long-range state-space modeling is particularly important for identifying AD participants, while its removal biases the model toward more conservative HC predictions.

The multi-scale encoder also provides consistent improvements across all evaluation metrics. Without multi-scale temporal modeling, BAcc and F1-AD decrease to 90.23% and 90.46%, respectively, demonstrating the benefit of capturing handwriting dynamics at different temporal scales. Finally, removing the Healthy Normative Branch reduces BAcc from 92.55% to 90.95% and sensitivity from 93.33% to 89.93%. Although this degradation is smaller than that caused by removing the other major components, it confirms that deviations from healthy handwriting dynamics provide complementary information for calibrating segment-level risk.

To further examine the robustness of NormPaST-Risk to key design choices in the segment-risk module, we conduct parameter sensitivity analyses on the number of risk experts and the segment top-k ratio. The former determines the capacity of the multi-expert module to model heterogeneous abnormal handwriting patterns, while the latter controls the proportion of high-risk segments selected for weakly supervised segmentrisk learning.

TABLE IV  
PARTICIPANT-LEVEL SEGMENT-RISK STATISTICS FOR HC AND AD GROUPS.
<table><tr><td>Aggregation</td><td>HC</td><td>AD</td><td></td><td>∆Risk Cohen&#x27;s d</td></tr><tr><td>Mean</td><td> $0 . 3 7 7 \pm 0 . 0 1 4$ </td><td> $0 . 4 3 5 \pm 0 . 0 3 6$ </td><td>0.058</td><td>2.07</td></tr><tr><td>Max</td><td> $0 . 4 6 6 \pm 0 . 0 2 4$ </td><td> $0 . 5 6 1 \pm 0 . 0 5 5$ </td><td>0.096</td><td>2.24</td></tr><tr><td>Top 50%</td><td> $0 . 4 1 1 \pm 0 . 0 1 7$ </td><td> $0 . 4 8 5 \pm 0 . 0 4 6$ </td><td>0.074</td><td>2.12</td></tr></table>

TABLE V

PERFORMANCE UNDER SEGMENT-LEVEL DELETION. ∆ DENOTES THE PERFORMANCE DECREASE RELATIVE TO THE ORIGINAL PREDICTION.
<table><tr><td>Strategy</td><td>Ratio</td><td>BAcc (%)</td><td>∆BAcc</td><td>AUC (%)</td><td>∆AUC</td></tr><tr><td>Original</td><td>一</td><td>92.55</td><td>一</td><td>95.84</td><td>一</td></tr><tr><td rowspan="3">Low-risk Random Top-risk</td><td>0.1</td><td>91.34</td><td>1.21</td><td>95.77</td><td>0.07</td></tr><tr><td>0.1</td><td>91.01</td><td>1.54</td><td>95.75</td><td>0.09</td></tr><tr><td>0.1</td><td>85.20</td><td>7.35</td><td>93.88</td><td>1.96</td></tr><tr><td rowspan="3">Low-risk Random Top-risk</td><td>0.2</td><td>90.75</td><td>1.80</td><td>95.78</td><td>0.06</td></tr><tr><td>0.2</td><td>90.67</td><td>1.88</td><td>95.65</td><td>0.19</td></tr><tr><td>0.2</td><td>83.01</td><td>9.54</td><td>92.36</td><td>3.48</td></tr><tr><td>Low-risk</td><td>0.3</td><td>90.36</td><td>2.19</td><td>95.51</td><td>0.33</td></tr><tr><td rowspan="3">Random Top-risk</td><td>0.3</td><td>90.37</td><td>2.18</td><td>95.35</td><td>0.49</td></tr><tr><td>0.3</td><td>83.01</td><td>9.54</td><td>91.98</td><td>3.86</td></tr><tr><td>0.4</td><td>90.03</td><td>2.52</td><td></td><td></td></tr><tr><td>Low-risk Random</td><td>0.4</td><td>89.66</td><td>2.89</td><td>95.45 95.34</td><td>0.39 0.50</td></tr><tr><td>Top-risk</td><td>0.4</td><td>82.45</td><td>10.10</td><td>91.30</td><td>4.54</td></tr><tr><td>Low-risk</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>0.5 0.5</td><td>89.44 89.37</td><td>3.11 3.18</td><td>95.39</td><td>0.45</td></tr><tr><td>Top-risk</td><td></td><td></td><td></td><td>95.03</td><td>0.81</td></tr><tr><td></td><td>0.5</td><td>79.71</td><td>12.84</td><td>89.62</td><td>6.22</td></tr></table>

Increasing the number of experts generally improves the performance of the segment-risk module. Compared with a single expert, using multiple experts provides clear gains in balanced accuracy, F1-AD, and AUC, suggesting that different experts can capture complementary AD-related handwriting patterns. As shown in Fig.4 (a), the 8-expert configuration achieves the best overall performance. In contrast, the improvement from four to eight experts is relatively moderate, indicating that increasing model capacity beyond a certain point yields diminishing returns. These results support the use of multiple risk experts for modeling heterogeneous ADrelated handwriting abnormalities.

The segment top-k ratio controls the proportion of highrisk segments selected for weakly supervised segment-level learning. As shown in Fig.4 (b), as the ratio increases from 0.1 to 0.5, the performance consistently improves, indicating that AD-related abnormalities are distributed across multiple informative handwriting segments rather than being restricted to only a few isolated regions. The performance slightly decreases at 0.7, suggesting that including more lower-ranked segments may dilute discriminative AD-related evidence. A moderate top-k ratio therefore helps retain sufficient abnormal segments while suppressing less informative regions that may contain variations associated with individual writing style, task-specific differences, or acquisition noise. This selective aggregation enables the model to focus more consistently on disease-relevant local patterns rather than irrelevant trajectory variability.

## E. Interpretability and Faithfulness Analysis

To examine whether the learned segment-risk scores differentiate HC and AD participants, we compare participant-level calibrated risks using mean, maximum, and top-50% aggregation, as shown in Table IV. AD participants consistently exhibit higher risk scores than HC participants across all three aggregation strategies. The absolute differences between the two groups are 0.058, 0.096, and 0.074 for mean, maximum, and top-50% aggregation, respectively.

Although these absolute differences are numerically modest on the [0, 1] risk scale, the corresponding Cohen’s d values are all greater than 2.0, indicating large standardized differences between the HC and AD groups. Cohen’s d was used to quantify the effect size, with values of 0.2, 0.5, and 0.8 conventionally interpreted as small, medium, and large effects, respectively [38]. In particular, maximum and top-50% aggregation yield larger group differences than mean aggregation, suggesting that emphasizing high-risk segments enhances the distinction between HC and AD participants. These findings support the segment-level risk modeling strategy and suggest that AD-related predictive evidence is more strongly concentrated in selected high-risk segments rather than being uniformly distributed throughout the trajectory. To further assess the contribution of these high-risk segments to the final prediction, we subsequently perform a segment-level deletion analysis.

To evaluate the faithfulness of the learned segment-risk scores, we perform a post-hoc deletion experiment without model retraining. At inference time, valid segments are ranked by their risk scores, and the highest-risk, lowest-risk, or randomly selected segments are masked from segment-risk pooling at ratios $\rho _ { \mathrm { d e l } } \in 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5$ . Random deletion is repeated and averaged, while all model parameters and evaluation settings are kept fixed. Results are reported as fivefold participant-level averages.

As shown in Table V, removing high-risk segments consistently causes substantially larger performance degradation than removing random or low-risk segments. When only 10% of the highest-risk segments are removed, BAcc and AUC decrease by 7.35 and 1.96 percentage points, respectively. In comparison, removing the same proportion of low-risk segments results in only a 1.21 point decrease in BAcc and almost no change in AUC. As the deletion ratio increases, the overall degradation becomes more pronounced. At a deletion ratio of 0.5, removing high-risk segments leads to decreases of 12.84 percentage points in BAcc and 6.22 percentage points in AUC, whereas low-risk deletion results in only a 3.11 point decrease in BAcc and negligible AUC degradation.

These results indicate that the learned segment-risk scores effectively rank handwriting segments according to their contribution to AD discrimination. The substantially larger degradation caused by removing high-risk segments suggests that the localized high-risk regions are closely associated with the discriminative handwriting evidence used by the model, thereby supporting the faithfulness of the proposed segmentlevel risk localization.

To qualitatively assess model interpretability, we visualize one representative HC participant and one representative AD participant on a subset of handwriting tasks, while both participants completed all 25 tasks. Similar or visually redundant tasks are omitted for clarity. Each task is presented with three views: the original trajectory (solid lines: on-paper; dashed lines: in-air), the highest-risk segment selected by the model, and the full trajectory risk heatmap.

As shown in Fig. 5, clear differences can be observed between the two participants across multiple tasks. For the HC participant, most trajectories exhibit relatively low risk, with only sparse local high-risk regions. In contrast, the AD participant shows more frequent and pronounced high-risk regions across different task types. Importantly, these regions are localized rather than uniformly distributed over the entire trajectory, indicating that the model relies on specific handwriting portions rather than assigning globally elevated risk to the whole sample.

![](images/6036fe22803a3f77ab4e56a0808f607f0829411864ddfd717f1af3e4a2b843cb.jpg)  
Fig. 5. Qualitative visualization of segment-level risk for one representative HC participant and one representative AD participant across selected handwriting tasks. Similar tasks are omitted for clarity. For each task, three subplots are shown: the original trajectory (solid lines indicate on-paper writing and dashed lines indicate in-air movement), the highest-risk segment selected by the model, and the trajectory-level risk heatmap, where warmer colors indicate higher predicted AD-related risk.

Despite substantial differences in trajectory shape and task content, localized high-risk regions repeatedly appear in the AD participant, whereas the HC participant generally maintains lower risk across the selected tasks. This suggests that the learned segment-risk representation can identify localized disease-related evidence across heterogeneous handwriting tasks, rather than being restricted to a single trajectory pattern.

High-risk regions are often located around portions with relatively complex local dynamics, such as curved strokes, direction changes, stroke transitions, and Paper-Air switching. However, these patterns should be interpreted as modelderived associations rather than direct evidence of specific cognitive or motor impairments. The localized risk distribution further suggests that the model identifies disease-related evidence from selected trajectory segments rather than from uniformly abnormal behavior throughout the sample.

The selected highest-risk segments generally coincide with locally elevated regions in the full risk maps. More importantly, this qualitative observation is consistent with the segment-deletion experiment, where removing high-risk segments results in a substantially larger performance decrease than removing low-risk or randomly selected segments. This provides additional evidence that the identified regions are relevant to the model’s AD predictions.

Overall, the visualization suggests that NormPaST-Risk can localize high-risk handwriting segments that contribute to participant-level AD prediction. The contrast between HC and AD risk patterns supports the interpretability of the proposed segment-level risk discovery mechanism. These visualizations should nevertheless be regarded as qualitative model evidence rather than direct clinical biomarkers.

## V. CONCLUSION

In this work, we proposed NormPaST-Risk, a trajectorybased framework for online handwriting-based AD detection. The model integrates multi-scale temporal encoding, Paper-Air state-aware modeling, selective state-space modeling, healthy normative learning, and task-aware multi-expert segment-risk estimation to capture both global handwriting dynamics and localized AD-related abnormalities. In addition to participantlevel prediction, the learned segment risks enable high-risk segments to be projected back onto the original handwriting trajectory, providing interpretable handwriting evidence for AD detection.

Experiments demonstrate that NormPaST-Risk achieves strong and balanced classification performance, reaching 92.55% BAcc and 95.84% AUC at the participant level. Ablation studies verify the contributions of the main components, while segment-deletion experiments show that removing highrisk segments causes substantially larger performance degradation than removing low-risk or randomly selected segments, supporting the reliability of the discovered local evidence. Overall, these results suggest that explicitly modeling Paper-Air dynamics and localized abnormal handwriting patterns provides an effective alternative to relying solely on global trajectory representations.

Several limitations remain. First, the current segment construction uses a fixed length of 128 trajectory points. Although this provides a consistent temporal unit for segment-level modeling, the resulting boundaries do not necessarily align with semantic handwriting units such as complete strokes, letters, or words. Consequently, a meaningful writing unit may be split across multiple segments, potentially weakening semantic continuity and limiting the interpretability of the discovered high-risk regions. Future work could explore adaptive or stroke-aware segmentation to better preserve meaningful handwriting structures. Second, the current evaluation is mainly based on a single online handwriting dataset, and the robustness of the proposed framework across different acquisition devices, handwriting tasks, and populations still requires further validation. Third, the segment-risk module is learned under weak supervision without manual segmentlevel annotations, meaning that the identified high-risk regions should be interpreted as model-derived evidence rather than direct clinical biomarkers. Finally, the healthy normative branch relies on HC samples to learn normal handwriting dynamics, which may be sensitive to the diversity and representativeness of the healthy reference population. Future work will therefore focus on adaptive segmentation, cross-dataset validation, more robust normative modeling, and clinical verification of the discovered high-risk handwriting patterns.

## REFERENCES

[1] D. S. Knopman, H. Amieva, R. C. Petersen, G. Chetelat,´ D. M. Holtzman, B. T. Hyman, R. A. Nixon, and D. T. Jones, “Alzheimer disease,” Nature reviews Disease primers, vol. 7, no. 1, p. 33, 2021.

[2] J. M. Long and D. M. Holtzman, “Alzheimer disease: an update on pathobiology and treatment strategies,” Cell, vol. 179, no. 2, pp. 312–339, 2019.

[3] C. R. Jack Jr, J. S. Andrews, T. G. Beach, T. Buracchio, B. Dunn, A. Graf, O. Hansson, C. Ho, W. Jagust, E. McDade et al., “Revised criteria for diagnosis and staging of alzheimer’s disease: Alzheimer’s association workgroup,” Alzheimer’s & Dementia, vol. 20, no. 8, pp. 5143–5169, 2024.

[4] H. Hampel, S. E. O’Bryant, J. L. Molinuevo, H. Zetterberg, C. L. Masters, S. Lista, S. J. Kiddle, R. Batrla, and K. Blennow, “Blood-based biomarkers for alzheimer disease: mapping the road to the clinic,” Nature Reviews Neurology, vol. 14, no. 11, pp. 639–652, 2018.

[5] T. K. Karikari, N. J. Ashton, G. Brinkmalm, W. S. Brum, A. L. Benedet, L. Montoliu-Gaya, J. Lantero-Rodriguez, T. A. Pascoal, M. Suarez-Calvet, P. Rosa-Neto et al., “Blood phospho-tau in alzheimer disease: analysis, interpretation, and clinical utility,” Nature Reviews Neurology, vol. 18, no. 7, pp. 400–418, 2022.

[6] K. K. Tsoi, J. Y. Chan, H. W. Hirai, S. Y. Wong, and T. C. Kwok, “Cognitive tests to detect dementia: a systematic review and meta-analysis,” JAMA internal medicine, vol. 175, no. 9, pp. 1450–1458, 2015.

[7] L. C. Kourtis, O. B. Regele, J. M. Wright, and G. B. Jones, “Digital biomarkers for alzheimer’s disease: the mobile/wearable devices opportunity,” NPJ digital medicine, vol. 2, no. 1, p. 9, 2019.

[8] W. Qi, X. Zhu, B. Wang, Y. Shi, C. Dong, S. Shen, J. Li, K. Zhang, Y. He, M. Zhao et al., “Alzheimer’s disease digital biomarkers multidimensional landscape and ai model scoping review,” npj Digital Medicine, vol. 8, no. 1, p. 366, 2025.

[9] G. Cepukaityt<sup>ˇ</sup> e, C. Newton, and D. Chan, “Early detec-˙ tion of diseases causing dementia using digital navigation and gait measures: A systematic review of evidence,” Alzheimer’s & Dementia, vol. 20, no. 4, pp. 3054–3073, 2024.

[10] J. H. Yan, S. Rountree, P. Massman, R. S. Doody, and H. Li, “Alzheimer’s disease and mild cognitive impairment deteriorate fine movement control,” Journal of Psychiatric Research, vol. 42, no. 14, pp. 1203–1212, 2008.

[11] C. P. Fernandes, G. Montalvo, M. Caligiuri, M. Pertsinakis, and J. Guimaraes, “Handwriting changes in alzheimer’s disease: a systematic review,” Journal of Alzheimer’s Disease, vol. 96, no. 1, pp. 1–11, 2023.

[12] M. A. El-Yacoubi, S. Garcia-Salicetti, C. Kahindo, A.-S. Rigaud, and V. Cristancho-Lacroix, “From aging to early-

stage alzheimer’s: uncovering handwriting multimodal behaviors by semi-supervised learning and sequential representation learning,” Pattern Recognition, vol. 86, pp. 112–133, 2019.

[13] J. Kawa, A. Bednorz, P. Stepien, J. Derejczyk, and M. Bugdol, “Spatial and dynamical handwriting analysis in mild cognitive impairment,” Computers in Biology and Medicine, vol. 82, pp. 21–28, 2017.

[14] N. D. Cilia, C. De Stefano, F. Fontanella, and S. M. Siniscalchi, “How word semantics and phonology affect handwriting of alzheimer’s patients: a machine learning based analysis,” Computers in Biology and Medicine, vol. 169, p. 107891, 2024.

[15] P. Ghaderyan, A. Abbasi, and S. Saber, “A new algorithm for kinematic analysis of handwriting data; towards a reliable handwriting-based tool for early detection of alzheimer’s disease,” Expert Systems with Applications, vol. 114, pp. 428–440, 2018.

[16] N. D. Cilia, G. De Gregorio, C. De Stefano, F. Fontanella, A. Marcelli, and A. Parziale, “Diagnosing alzheimer’s disease from on-line handwriting: A novel dataset and performance benchmarking,” Engineering Applications of Artificial Intelligence, vol. 111, p. 104822, 2022.

[17] J. Garre-Olmo, M. Faundez-Zanuy, K. L´ opez-de Ipi´ na,˜ L. Calvo-Perxas, and O. Turr´ o-Garriga, “Kinematic and´ pressure features of handwriting and drawing: preliminary results between patients with mild cognitive impairment, alzheimer disease and healthy controls,” Current Alzheimer Research, vol. 14, no. 9, pp. 960–968, 2017.

[18] P. Erdogmus and A. T. Kabakus, “The promise of convolutional neural networks for the early diagnosis of the alzheimer’s disease,” Engineering Applications of Artificial Intelligence, vol. 123, p. 106254, 2023.

[19] N. D. Cilia, T. D’Alessandro, C. De Stefano, F. Fontanella, and M. Molinara, “From online handwriting to synthetic images for alzheimer’s disease detection using a deep transfer learning approach,” IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 12, pp. 4243–4254, 2021.

[20] C. Gong, H. Qin, and M. A. El-Yacoubi, “Hybrid transformer for early alzheimer’s detection: Integration of handwriting-based 2d images and 1d signal features,” IEEE Journal of Biomedical and Health Informatics, 2025.

[21] E. Nardone, C. De Stefano, N. D. Cilia, and F. Fontanella, “Handwriting strokes as biomarkers for alzheimer’s disease prediction: a novel machine learning approach,” Computers in Biology and Medicine, vol. 190, p. 110039, 2025.

[22] J. Park and S. Kang, “Paano: patch-based representation learning for time-series anomaly detection,” arXiv preprint arXiv:2602.01359, 2026.

[23] Z. Liu, Y. Wang, B. Li, J. Zheng, E. Eldele, M. Wu, and Q. Ma, “A unified shape-aware foundation model for time series classification,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026, pp. 23 972– 23 980.

[24] D. Impedovo and G. Pirlo, “Dynamic handwriting anal-

ysis for the assessment of neurodegenerative diseases: a pattern recognition perspective,” IEEE reviews in biomedical engineering, vol. 12, pp. 209–220, 2018.

[25] Q. Dao, M. A. El-Yacoubi, and A.-S. Rigaud, “Detection of alzheimer disease on online handwriting using 1d convolutional neural network,” IEEE Access, vol. 11, pp. 2148–2155, 2022.

[26] L. Kang, X. Zhang, J. Guan, K. Huang, and R. Wu, “Early alzheimer’s disease diagnosis via handwriting with self-attention mechanisms,” Journal of Alzheimer’s Disease, vol. 102, no. 1, pp. 173–180, 2024.

[27] J. Sweidan, M. A. El-Yacoubi, and A.-S. Rigaud, “Explainability of cnn-based alzheimer’s disease detection from online handwriting,” Scientific Reports, vol. 14, no. 1, p. 22108, 2024.

[28] A. Gu, K. Goel, and C. Re, “Efficiently modeling long´ sequences with structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.

[29] A. Gupta, A. Gu, and J. Berant, “Diagonal state spaces are as effective as structured state spaces,” Advances in neural information processing systems, vol. 35, pp. 22 982–22 994, 2022.

[30] A. Gu, K. Goel, A. Gupta, and C. Re, “On the parame-´ terization and initialization of diagonal state space models,” Advances in neural information processing systems, vol. 35, pp. 35 971–35 983, 2022.

[31] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[32] S. Muller, O. Preische, P. Heymann, U. Elbing, and¨ C. Laske, “Increased diagnostic accuracy of digital vs. conventional clock drawing test for discrimination of patients in the early course of alzheimer’s disease from cognitively healthy individuals,” Frontiers in aging neuroscience, vol. 9, p. 101, 2017.

[33] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[34] L. Breiman, “Random forests,” Machine learning, vol. 45, no. 1, pp. 5–32, 2001.

[35] M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.

[36] N. Mwamsojo, F. Lehmann, M. A. El-Yacoubi, K. Merghem, Y. Frignac, B.-E. Benkelfat, and A.-S. Rigaud, “Reservoir computing for early stage alzheimer’s disease detection,” IEEE Access, vol. 10, pp. 59 821– 59 831, 2022.

[37] A. Vaswani, “Attention is all you need,” arXiv preprint arXiv:1706.03762, 2017.

[38] J. Cohen, Statistical power analysis for the behavioral sciences. routledge, 2013.