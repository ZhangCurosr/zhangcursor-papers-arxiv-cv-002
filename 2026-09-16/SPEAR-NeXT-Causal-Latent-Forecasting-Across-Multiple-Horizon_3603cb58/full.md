# SPEAR-NeXT: Causal Latent Forecasting Across Multiple Horizons for Spectral–Temporal Earth Representation Learning

Rajiv Ranjan Udaiveer Singh Shashank Tamaskar

Plaksha University Mohali, Punjab, India

{rajiv.ranjan, udaiveer.singh.ug23, shashank.tamaskar}@plaksha.edu.in

Dharmendra Saraswat Purdue University West Lafayette, Indiana, USA

saraswat@purdue.edu

## Abstract

Earth observation is inherently dynamic, yet temporal information in many foundation models is learned through reconstruction, invariance, or retrospective sequence summarization. SPEAR-NeXT is introduced as a compact pixel-wise multimodal spectral–temporalfoundation model in which temporal self-supervision is formulated as pastonly, multi-horizon latent Earth-state prediction. Instantaneous states are first encoded by the pretrained SPEAR model from optical, radar, and environmental observations into compact 32-dimensional embeddings. Their temporal evolution is then modeled by a causally masked Transformer that predicts multiple future latent states from preceding observations. Relative temporal order is represented using Rotary Position Embeddings, while month and year embeddings encode seasonal phase and inter-annual context.

Horizon-weighted latent trajectory supervision is used to capture local continuity, intermediate state transitions, and longer-term seasonal structure. Cosine alignmentpreserves latent-space direction, while latent regression constrains coordinate scale and limits magnitude drift. Separate regional models are pretrained over India and the contiguous United States using monthly observations from 2020– 2024. The learned representations are evaluated through frozen and lightweight adaptation on land-cover mapping, crop classification, phenological estimation, and crop-yield forecasting. SPEAR-NeXT achieves land-cover classifica tion accuracies of94.81% in India and 88.78% in CONUS, obtains the lowest errors for sowing-date, harvest-date, and yield estimation on SICKLE, and reaches a mean R<sup>2</sup> of 0.721 for USDA-NASS crop-yield prediction, compared

with 0.644for TESSERA and 0.613for Presto. Predictionhead and kriging ablations further indicate that the gains are primarily associated with the pretrained temporal representation rather than downstream capacity or spatial post-processing. These results support causal multi-horizon latent forecasting as an effective objective for compact and reusable pixel-level Earth representations.

## 1. Introduction

Earth observation (EO) is not a collection of independent images, but a record of a continuously changing surface. Vegetation growth, soil-moisture variation, inundation, disturbance, recovery, and land management unfold through trajectories whose meaning is often ambiguous in a single observation. Temporal structure is therefore central to land-cover monitoring, change analysis, crop-type mapping, phenological assessment, and environmental forecasting [7, 34].

Self-supervised learning has made it possible to exploit large unlabeled satellite archives and has driven rapid progress in EO foundation models. Existing approaches reconstruct masked spatial, spectral, or temporal content, align observations across sensors or augmentations, or summarize multi-date inputs into transferable embeddings. SatMAE extends masked autoencoding to temporal and multispectral imagery, CROMA combines radar– optical contrastive learning with masked reconstruction, Presto reconstructs structured pixel-level sensor time series, AnySat uses joint-embedding prediction across heterogeneous EO inputs, and AlphaEarth Foundations generates time-conditioned embedding fields from multisource observations [2, 6, 7, 10, 34]. These models demonstrate that temporal and multisensor context are valuable. The remaining gap is therefore not the absence of time, but the limited use of directional predictive supervision: most temporal objectives recover missing observations, align alternative views, or summarize an observed interval, rather than requiring a representation at time t to predict several unseen states after t using only preceding context.

This distinction is important for pixel-level satellite time series. At the pixel scale, spatial context is deliberately minimized, and temporal evolution becomes a primary source of information. A pixel’s spectral response may change gradually because of vegetation development, surface moisture, disturbance, management, or seasonal climate. A useful temporal representation should consequently encode not only what has already been observed, but also which components of the current state remain informative about its future trajectory. We therefore ask:

What information must a compact pixel representation preserve to remain predictive across multiple future horizons?

SPEAR: What is the latent state now? SPEAR-NeXT: How will that latent state evolve?

Predictive representation learning provides a promising alternative to observation reconstruction. I-JEPA predicts latent target-region representations from visible image context, and V-JEPA extends feature prediction to video without reconstructing pixels [1, 3]. In EO, future-image forecasting has also been formulated directly in observation space, for example by predicting future Sentinel-2 imagery conditioned on environmental variables [24]. These directions motivate prediction as a learning signal, but they leave open a distinct EO problem: causal prediction of multiple future pixel-level spectral states for representation learning. Forecasting in latent space does not make the targets intrinsically noise-free; rather, it shifts the objective from reproducing every radiometric detail toward predicting the temporal structure retained by a pretrained encoder.

To address this problem, SPEAR-NeXT is introduced as a compact pixel-wise spectral–temporal foundation model that formulates temporal pretraining as causal latent Earthstate prediction. Here, causal denotes autoregressive information flow: the prediction at time t is computed only from states observed at or before t, and does not imply intervention-based causal inference. Similarly, an Earth state refers to a learned spectral representation of a pixel rather than a complete physical state of the land surface.

SPEAR-NeXT follows a spectral-to-temporal decomposition. The previously published SPEAR encoder first maps each multispectral pixel into a compact latent state using reflectance together with continuous band-center wavelength and bandwidth metadata [22]. The temporal component then learns a transition operator over the resulting state sequence. In this decomposition, SPEAR estimates the instantaneous spectral state, whereas SPEAR-NeXT learns how that state evolves. The two-stage design also enables modular pretraining: spectral states can be computed once, cached, and subsequently used to train the temporal model without repeatedly processing raw multispectral measurements.

The temporal model predicts several future states directly from a causal Transformer representation. Two forms of temporal information are modeled separately. Rotary Position Embeddings (RoPE) encode sequence order and relative displacement within self-attention [31], while learnable month and year embeddings provide absolute seasonal phase and inter-annual context. This separation is useful because identical sequence offsets may occur at different points of an annual cycle, whereas the same calendar month can exhibit different conditions across years.

A central element of SPEAR-NeXT is horizon-weighted latent trajectory supervision. Multi-horizon forecasting is used not only to produce future predictions, but also to constrain what the representation preserves. A one-step objective can favor local continuity; supervision at multiple horizons additionally exposes the model to intermediate transitions and longer-range recurrence. Horizon-dependent weighting stabilizes near-term learning while retaining supervision from more distant states. Predicted and target trajectories are compared through a composite objective that combines cosine alignment with latent vector regression, thereby constraining both orientation and scale in the frozen SPEAR embedding space.

Although agriculture provides a demanding test because crop identity, phenological stage, management, and productivity depend strongly on temporal progression, SPEAR-NeXT is not formulated as an agriculture-specific model. Its pretraining objective is task-agnostic and is applied to diverse unlabeled land-surface observations. Evaluation is therefore conducted at two levels: embedding-space diagnostics measure horizon-wise predictability, magnitude drift, temporal self-similarity, and prediction–target alignment; downstream experiments measure transfer to general land-cover mapping and temporally demanding agricultural tasks using frozen or lightweight adaptation. Capacitymatched and prediction-head ablations are used to distinguish gains from temporal pretraining from those caused only by additional downstream parameters.

Contributions. The main contributions are:

• A self-supervised temporal pretraining formulation for EO in which compact pixel-level spectral states are predicted causally across multiple future horizons, rather than reconstructed from bidirectional context or summarized retrospectively.

• A modular spectral-to-temporal decomposition that builds on the wavelength-aware SPEAR encoder and learns a separate causal transition model over its compact latent states.

• A horizon-weighted latent trajectory objective, together with complementary relative and calendar-time conditioning, that constrains the representation across local continuity, intermediate transitions, and longer-range temporal structure.

• A representation-focused evaluation protocol combining forecasting baselines, latent-geometry diagnostics, temporal and objective ablations, and capacity-matched downstream transfer on land-cover and agricultural tasks.

SPEAR-NeXT thus moves pixel-level temporal EO pretraining from encoding an already observed sequence toward learning representations that remain informative about how the sequence can evolve.

## 2. Related Work

## 2.1. Self-Supervised Learning and Earth-Observation Foundation Models

The scale of modern satellite archives, together with the scarcity and uneven geographic distribution of ground-truth labels, has made self-supervised learning central to Earthobservation (EO) representation learning. In general computer vision, masked autoencoding and self-distillation have demonstrated that large models can learn reusable representations from unlabeled imagery and can subsequently support strong frozen-backbone transfer [13, 30]. EO foundation models extend this paradigm by incorporating properties that are uncommon in natural imagery, including multispectral measurements, repeated observations, sensor heterogeneity, geographic scale, and irregular data availability.

Masked reconstruction remains the dominant EO pretraining objective. SatMAE extends masked autoencoding to temporal and multispectral satellite imagery through temporal embeddings, independent masking across acquisition times, and spectral group embeddings [7]. Scale-MAE incorporates the ground sampling scale into positional encoding and reconstructs frequency components at multiple scales [23], while Prithvi pretrains a multi-temporal Transformer on Harmonized Landsat–Sentinel-2 imagery for transfer to several EO tasks [15]. CROMA combines radar– optical contrastive alignment with masked reconstruction [10], and SkySense learns multimodal spatiotemporal representations using multi-granularity contrastive and geographic prototype objectives [11]. More recently, AnySat adopts a joint-embedding predictive architecture to support heterogeneous resolutions, scales, and modalities [2], whereas AlphaEarth Foundations produces compact embedding fields by assimilating spatial, temporal, and measurement context across multiple sources [6]. Collectively, these models establish the value of large-scale EO pretraining; however, their temporal learning signals are primarily based on masked recovery, cross-view alignment, or conditional summarization rather than explicit causal prediction of future latent Earth states.

## 2.2. Spectral and Sensor-Aware Representation Learning

The spectral axis is physically structured: each channel measures radiance over a sensor-specific wavelength response rather than acting as an interchangeable image channel. SpectralGPT and S2MAE exploit this structure through spatial–spectral tokenization and masked reconstruction [14, 16]. DOFA goes further by conditioning dynamic patch-embedding weights on channel wavelengths, allowing a shared Transformer to process different sensor configurations [37]. These methods substantially improve spectral and cross-sensor transfer, but they generally learn spectral structure jointly with spatial patches or cubes and optimize reconstruction at the observation level.

SPEAR-NeXT follows a complementary factorization. Its spectral encoder operates on the multispectral vector of an individual pixel and uses continuous band-center wavelength and bandwidth metadata to produce a compact latent state. This design treats wavelength metadata as a physical index of the measurement process, while avoiding the stronger claim that the representation is explicitly constrained by a physical process model. The resulting spectral state becomes the input to a separate temporal learner. This spectral-to-temporal decomposition distinguishes instantaneous state estimation from state evolution and enables the temporal objective to operate entirely in a compact embedding space.

## 2.3. Temporal Modeling and Self-Supervision for Satellite Time Series

Early satellite image time-series models used recurrent networks to summarize vegetation and land-cover trajectories [26]. Temporal convolutional networks subsequently improved parallelism and provided effective localto-intermediate temporal receptive fields [20]. Transformerbased models replaced recurrence with attention, enabling direct interaction between distant observations. The Pixel-Set Encoder with Temporal Attention Encoder demonstrated the effectiveness of temporal self-attention for parcel-level satellite time-series classification [27], while thermal positional encoding showed that calendarindependent phenological coordinates can improve crossregion crop classification [19]. These approaches established the importance of temporal order, acquisition timing, and seasonal phase, but they are principally trained for supervised classification and therefore learn temporal features that are coupled to a specific label space.

Temporal self-supervision has since become more prominent. SatMAE reconstructs temporally masked image patches [7], and Presto uses structured masking to reconstruct missing time points and sensor groups from pixel-level multisensor time series [34]. Prithvi and SkySense also ingest multi-temporal imagery, while AlphaEarth learns time-conditioned summaries that support mapping at specified periods [6, 11, 15]. These methods demonstrate that temporal context improves label efficiency and transfer. Nevertheless, reconstruction-based objectives can exploit observations on both sides of a masked timestamp, and summary-based objectives compress an interval without necessarily identifying the directed transition that maps a present state to multiple future states. SPEAR-NeXT instead asks a causal question: given only preceding latent states, what future latent trajectory is predictable?

## 2.4. Predictive Latent Modeling and Multi-Horizon Objectives

Predictive representation learning provides an alternative to reconstructing raw observations. Contrastive Predictive Coding learns representations by predicting future latent variables with an autoregressive context model [36]. I-JEPA predicts target-region embeddings from visible image context without reconstructing pixels [1], and V-JEPA extends feature prediction to video, showing that latent prediction can capture both appearance and motion while supporting frozen-backbone transfer [3]. AnySat brings a JEPA-style objective to heterogeneous EO imagery [2]. These works motivate prediction in representation space, but they do not directly address causal, pixel-wise forecasting of spectral Earth-state trajectories across multiple future acquisition horizons.

Observation-space Earth forecasting follows a different objective. EarthNet2021 formulates future Sentinel-2 image generation conditioned on meteorological variables as guided video prediction [24]. Such forecasting is valuable when future reflectance or imagery is itself the desired product, but pixel-space losses must allocate capacity to radiometric detail, cloud contamination, registration variation, and other measurement-specific effects. In contrast, forecasting a spectral embedding allows the predictive objective to emphasize changes preserved by a pretrained state encoder. This does not make latent targets inherently noisefree; rather, it shifts the learning target from exact observation reproduction toward the temporal evolution represented by the encoder.

Multi-horizon forecasting is well established in timeseries analysis, where architectures such as the Temporal Fusion Transformer jointly predict several future steps [17]. In most forecasting systems, multiple horizons are output requirements and performance is measured directly at each forecast distance. SPEAR-NeXT uses multi-horizon prediction additionally as a representation constraint. Horizonweighted latent supervision exposes the temporal model to short-range continuity, intermediate transitions, and longerrange seasonal recurrence within a single causal objective. Shorter horizons receive stronger weights for optimization stability, while longer horizons prevent the learned representation from being defined solely by adjacent-state interpolation.

Temporal attention also requires a distinction between sequence displacement and calendar phase. Rotary Position Embeddings encode relative positional interactions within self-attention [31]; satellite time-series models additionally benefit from acquisition-date, seasonal, or phenological encodings [19, 27, 34]. SPEAR-NeXT therefore combines RoPE with structured month and year embeddings. RoPE describes relative progression through the observed sequence, month identifies position within the seasonal cycle, and year provides inter-annual context. The two mechanisms are complementary: observations separated by a similar temporal offset may occur in different seasonal phases, while the same calendar month may exhibit different dynamics across years.

## 2.5. Positioning of SPEAR-NeXT

SPEAR-NeXT is positioned between spectral EO foundation models, temporal satellite encoders, and jointembedding predictive learning. Spectral foundation models learn informative representations of measurements at one or more acquisition times; temporal classifiers summarize trajectories for predefined tasks; masked temporal models recover omitted observations; and image-forecasting systems predict future measurements. SPEAR-NeXT instead decomposes pixel-wise EO pretraining into a wavelengthaware state encoder and a causal latent transition model. The temporal module predicts several future spectral states using horizon-weighted supervision and an explicit separation of relative sequence position from absolute seasonal phase.

Accordingly, the central contribution is not simply the use of a Transformer, RoPE, calendar embeddings, or multistep prediction in isolation. It is their integration into a self-supervised objective for learning compact, predictive Earth representations: SPEAR estimates the spectral state, while SPEAR-NeXT learns how that state evolves. This formulation is intended to produce reusable embeddings for frozen or lightweight adaptation across land-cover mapping and temporally demanding agricultural tasks, while allowing representation ablations to separate gains from temporal pretraining from those contributed by a downstream prediction head.

## 3. Methodology

## 3.1. Overview and Problem Formulation

Let the unlabeled pretraining corpus contain N pixel-level satellite time series,

$$
\begin{array} { r } { \mathcal { D } = \{ \boldsymbol { S } _ { n } \} _ { n = 1 } ^ { N } , \qquad \boldsymbol { S } _ { n } = \{ ( \mathbf { x } _ { n , t } , \boldsymbol { \tau } _ { n , t } ) \} _ { t = 1 } ^ { T _ { n } } , } \end{array}\tag{1}
$$

where $\mathbf { x } _ { n , t } \in \mathbb { R } ^ { C }$ is the multispectral observation of pixel n at timestamp $\tau _ { n , t } , C$ is the number of spectral bands, and $T _ { n }$ is the number of valid observations available for that pixel. Observations are chronologically ordered such that $\tau _ { n , 1 } <$ $\cdots < \tau _ { n , T _ { n } }$ . The formulation permits different sequence lengths across pixels.

SPEAR-NeXT decomposes representation learning into two complementary stages. First, a pretrained wavelengthaware SPEAR encoder estimates an instantaneous spectral state from each multispectral observation. Second, a causal temporal model learns how sequences of these states evolve and predicts their future latent trajectory. For every horizon $k \in \{ 1 , \ldots , K \}$ , the objective is to estimate the state at time $t + k$ using only observations available up to time t:

$$
{ \widehat { \mathbf { e } } } _ { n , t + k | t } = g _ { \omega _ { k } } \left( q _ { \theta } \left( \mathbf { e } _ { n , 1 : t } , \tau _ { n , 1 : t } \right) \right) ,\tag{2}
$$

where ${ \bf e } _ { n , t } \in \mathbb { R } ^ { D }$ is the spectral state, q is the causal temporal encoder, and $g _ { \omega _ { k } }$ is the predictor associated with horizon k. The notation $t + k \mid t$ explicitly distinguishes the predicted future state from the causal context used to estimate it.

Throughout this work, causal refers to autoregressive information flow enforced by a temporal attention mask. It does not imply intervention-based causal inference. Similarly, Earth state denotes a learned compact spectral representation rather than a complete physical description of the land surface.

## 3.2. Instantaneous Spectral States from SPEAR

The instantaneous spectral encoder is adopted from the previously published SPEAR framework [22]. SPEAR introduced a wavelength-conditioned Spectral-MAE that operates directly on per-pixel multispectral vectors and uses continuous band-center wavelength and full width at half maximum (FWHM) metadata during tokenization. Its masked-band pretraining procedure, Fourier spectral metadata encoding, encoder–decoder architecture, and reconstruction objective are described in [22] and are therefore not repeated here.

Although the original SPEAR framework supports optical, radar, climate, and multimodal fusion, the temporal formulation studied in this work uses its pretrained optical Spectral-MAE branch. Let

$$
\begin{array} { r } { \pmb { \lambda } = \left[ \lambda _ { 1 } , \ldots , \lambda _ { C } \right] ^ { \top } , \qquad \pmb { \Delta } \pmb { \lambda } = \left[ \Delta \lambda _ { 1 } , \ldots , \Delta \lambda _ { C } \right] ^ { \top } } \end{array}\tag{3}
$$

denote the band-center wavelengths and FWHM values. Each observation is mapped to a compact state through

$$
\mathbf { e } _ { n , t } = f _ { \psi } ^ { \mathrm { S P E A R } } \left( \mathbf { x } _ { n , t } ; \lambda , \Delta \lambda \right) , \qquad \mathbf { e } _ { n , t } \in \mathbb { R } ^ { D } .\tag{4}
$$

In the present implementation, $D \ = \ 3 2$ , consistent with the compact SPEAR spectral descriptor. The reconstruction decoder used during SPEAR pretraining is discarded, and the unmasked encoder output associated with the spectral class token is used as $\mathbf { e } _ { n , t }$

The pretrained parameters ψ are held fixed during temporal pretraining. Consequently, the spectral states can be computed and cached once, reducing temporal training cost and ensuring that future-state targets remain stationary.

## 3.3. Temporal Token Construction

Satellite time series contain two complementary forms of temporal information. Relative temporal progression describes ordering and displacement between observations, whereas absolute calendar phase describes seasonal position and inter-annual context. SPEAR-NeXT models these signals separately.

Let $m ( \tau _ { n , t } ) \in \{ 1 , . . . , 1 2 \}$ and $y ( \tau _ { n , t } ) \in \{ 1 , . . . , N _ { y } \}$ denote the month and indexed year associated with timestamp $\tau _ { n , t }$ . Learnable month and year embeddings are defined as

$$
\begin{array} { r } { \mathbf { m } _ { n , t } = \mathbf { M } _ { m ( \tau _ { n , t } ) } , \qquad \mathbf { y } _ { n , t } = \mathbf { Y } _ { y ( \tau _ { n , t } ) } , } \end{array}\tag{5}
$$

where $\textbf { M } \in \mathbb { R } ^ { 1 2 \times D _ { m } }$ and $\mathbf { Y } \in \mathbb { R } ^ { N _ { y } \times D _ { y } }$ . The absolute calendar representation is

$$
\mathbf { c } _ { n , t } = [ \mathbf { m } _ { n , t } ; \mathbf { y } _ { n , t } ] \in \mathbb { R } ^ { D _ { m } + D _ { y } } .\tag{6}
$$

The spectral state and calendar representation are concatenated and projected into the temporal model dimension:

$$
\mathbf h _ { n , t } ^ { ( 0 ) } = \mathbf W _ { \mathrm { i n } } \left[ \mathbf e _ { n , t } ; \mathbf c _ { n , t } \right] + \mathbf b _ { \mathrm { i n } } , \qquad \mathbf h _ { n , t } ^ { ( 0 ) } \in \mathbb { R } ^ { D _ { T } } .\tag{7}
$$

Month and year embeddings therefore provide absolute temporal context, while relative progression is incorporated within self-attention through Rotary Position Embeddings.

## 3.4. Causal Temporal Dynamics Encoder

Let

$$
\mathbf { H } _ { n } ^ { ( \ell ) } = \left[ \mathbf { h } _ { n , 1 } ^ { ( \ell ) } , \ldots , \mathbf { h } _ { n , T _ { n } } ^ { ( \ell ) } \right] ^ { \top } \in \mathbb { R } ^ { T _ { n } \times D _ { T } }\tag{8}
$$

denote the sequence at layer ℓ. SPEAR-NeXT employs a pre-normalized causal Transformer with L layers and A attention heads.

![](images/12922c48c344bc7afaf18742abc01cd1c1842f97f4cd66dcb8bde946414b7a21.jpg)  
Figure 1. Overview of the proposed SPEAR-NeXT framework. A frozen pretrained SPEAR spectral encoder transforms each multi spectral pixel observation into a compact 32-dimensional instantaneous spectral state. The resulting pixel-wise embedding sequence is processed by a causal temporal Transformer equipped with Rotary Position Embeddings, QK normalization, and gated feed-forward trans formations. Horizon-specific prediction heads directly forecast multiple future latent states, which are optimized using self-supervised horizon-weighted latent trajectory supervision.

## 3.4.1. QK-normalized rotary self-attention

For attention head a, the query, key, and value projections are

$$
\mathbf { Q } _ { a } ^ { ( \ell ) } = \operatorname { Q K N o r m } \left( \operatorname { L N } \left( \mathbf { H } _ { n } ^ { ( \ell ) } \right) \mathbf { W } _ { Q , a } ^ { ( \ell ) } \right) ,\tag{9}
$$

$$
\mathbf { K } _ { a } ^ { \left( \ell \right) } = \mathrm { Q K N o r m } \left( \mathrm { L N } \left( \mathbf { H } _ { n } ^ { \left( \ell \right) } \right) \mathbf { W } _ { K , a } ^ { \left( \ell \right) } \right) ,\tag{10}
$$

$$
\mathbf { V } _ { a } ^ { ( \ell ) } = \mathrm { L N } \left( \mathbf { H } _ { n } ^ { ( \ell ) } \right) \mathbf { W } _ { V , a } ^ { ( \ell ) } .\tag{11}
$$

QK normalization is applied independently to each query and key head vector:

$$
\mathrm { Q K N o r m } \left( \mathbf { a } \right) = \gamma \odot \frac { \mathbf { a } - \mu ( \mathbf { a } ) } { \sqrt { \sigma ^ { 2 } ( \mathbf { a } ) + \varepsilon } } ,\tag{12}
$$

where $\mu ( \cdot )$ and $\sigma ^ { 2 } ( \cdot )$ are computed across the head dimension, $\gamma$ is a learnable scale, and ε is a numerical-stability constant. Layer normalization controls the scale of token activations before projection, whereas QK normalization directly controls the query–key dot products used by attention.

Let $p _ { n , t }$ denote the temporal coordinate of observation t. For regularly composited sequences, $p _ { n , t } = t ;$ for irregular acquisitions, it may be defined using elapsed time from the beginning of the sequence. RoPE applies a positiondependent rotation to the query and key vectors:

$$
\widetilde { \mathbf { q } } _ { n , t , a } ^ { ( \ell ) } = \mathbf { R } \left( p _ { n , t } \right) \mathbf { q } _ { n , t , a } ^ { ( \ell ) } , \qquad \widetilde { \mathbf { k } } _ { n , t , a } ^ { ( \ell ) } = \mathbf { R } \left( p _ { n , t } \right) \mathbf { k } _ { n , t , a } ^ { ( \ell ) } .\tag{13}
$$

The identity

$$
\mathbf { R } ( p _ { i } ) ^ { \top } \mathbf { R } ( p _ { j } ) = \mathbf { R } ( p _ { j } - p _ { i } )\tag{14}
$$

allows the attention interaction to depend on relative temporal displacement.

Causality is enforced through the mask

$$
\mathcal { C } _ { i j } = \left\{ \begin{array} { l l } { 0 , } & { j \le i , } \\ { - \infty , } & { j > i . } \end{array} \right.\tag{15}
$$

An additional mask $\mathcal { P } _ { i j }$ excludes padded or invalid timestamps when variable-length sequences are batched. The output of head a is

$$
\mathbf { A } _ { a } ^ { ( \ell ) } = \mathrm { s o f t m a x } \left( \frac { \widetilde { \mathbf { Q } } _ { a } ^ { ( \ell ) } \widetilde { \mathbf { K } } _ { a } ^ { ( \ell ) \top } } { \sqrt { d _ { h } } } + \mathcal { C } + \mathcal { P } \right) \mathbf { V } _ { a } ^ { ( \ell ) } ,\tag{16}
$$

where $d _ { h } = D _ { T } / A$ is the dimension of each attention head. Hence, the representation at time t is computed only from observations at times $1 , \ldots , t$

The multi-head attention output is combined through a residual connection:

$$
\mathbf { H } _ { n } ^ { ( \ell + \frac { 1 } { 2 } ) } = \mathbf { H } _ { n } ^ { ( \ell ) } + \mathrm { C o n c a t } \left( \mathbf { A } _ { 1 } ^ { ( \ell ) } , \ldots , \mathbf { A } _ { A } ^ { ( \ell ) } \right) \mathbf { W } _ { O } ^ { ( \ell ) } .\tag{17}
$$

## 3.4.2. SwiGLU feed-forward transformation

The feed-forward sublayer uses a gated SwiGLU transformation:

$$
\mathrm { S w i G L U } \left( \mathbf { a } \right) = { \mathbf { W } } _ { 2 } \left[ \mathrm { S i L U } \left( { \mathbf { W } } _ { g } \mathbf { a } + \mathbf { b } _ { g } \right) \odot \left( { \mathbf { W } } _ { u } \mathbf { a } + \mathbf { b } _ { u } \right) \right] + \mathbf { b } _ { 2 } ,\tag{18}
$$

where ⊙ denotes element-wise multiplication. The layer output is

$$
\mathbf H _ { n } ^ { ( \ell + 1 ) } = \mathbf H _ { n } ^ { ( \ell + \frac { 1 } { 2 } ) } + \mathrm { S w i G L U } \left( \mathrm { L N } \left( \mathbf H _ { n } ^ { ( \ell + \frac { 1 } { 2 } ) } \right) \right) .\tag{19}
$$

After L Transformer layers, the contextual state at time t is

$$
\mathbf { r } _ { n , t } = \mathrm { L N } \left( \mathbf { h } _ { n , t } ^ { ( L ) } \right) \in \mathbb { R } ^ { D _ { T } } .\tag{20}
$$

## 3.5. Causal Multi-Horizon Latent Forecasting

A separate lightweight prediction head is used for each horizon. For $k \in \{ 1 , \ldots , K \}$ ,

$$
g _ { \omega _ { k } } \left( \mathbf { r } \right) = \mathbf { W } _ { k , 2 } \mathrm { S i L U } \left( \mathbf { W } _ { k , 1 } \mathbf { r } + \mathbf { b } _ { k , 1 } \right) + \mathbf { b } _ { k , 2 } ,\tag{21}
$$

and the future-state prediction is

$$
\begin{array} { r } { \widehat { \mathbf { e } } _ { n , t + k | t } = g _ { \omega _ { k } } \left( \mathbf { r } _ { n , t } \right) . } \end{array}\tag{22}
$$

The horizon k denotes a future sequence step; under a fixed monthly compositing scheme, it corresponds to k months. Horizon-specific heads permit the mapping from contextual state to target state to vary with forecast distance and avoid recursive error accumulation.

The corresponding target is generated by the frozen SPEAR encoder:

$$
\mathbf { e } _ { n , t + k } ^ { \star } = \mathrm { s g } \left[ f _ { \psi } ^ { \mathrm { S P E A R } } \left( \mathbf { x } _ { n , t + k } ; \lambda , \Delta \lambda \right) \right] ,\tag{23}
$$

where sg[·] denotes stop-gradient. Because $f _ { \psi } ^ { \mathrm { S P E A R } }$ is frozen, the target space remains fixed throughout temporal optimization.

## 3.6. Horizon-Weighted Latent Trajectory Objective

For each horizon k, define the set of valid context–target pairs as

$$
\Omega _ { k } = \{ ( n , t ) \mid 1 \leq n \leq N , 1 \leq t \leq T _ { n } - k \} .\tag{24}
$$

Directional agreement is optimized using cosine distance:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { c o s } } ^ { ( k ) } = \frac { 1 } { | \Omega _ { k } | } \sum _ { ( n , t ) \in \Omega _ { k } } } \\ { \displaystyle \left[ 1 - \frac { \widehat { \mathbf { e } } _ { n , t + k \mid t } ^ { \top } \mathbf { e } _ { n , t + k } ^ { \star } } { \left( \| \widehat { \mathbf { e } } _ { n , t + k \mid t } \| _ { 2 } + \varepsilon \right) \left( \| \mathbf { e } _ { n , t + k } ^ { \star } \| _ { 2 } + \varepsilon \right) } \right] . } \end{array}\tag{25}
$$

Since cosine distance is insensitive to vector scale, it is complemented by latent vector regression:

$$
\mathcal { L } _ { \mathrm { r e g } } ^ { ( k ) } = \frac { 1 } { D \big | \Omega _ { k } \big | } \sum _ { ( n , t ) \in \Omega _ { k } } \left\| \widehat { \mathbf { e } } _ { n , t + k \mid t } - \mathbf { e } _ { n , t + k } ^ { \star } \right\| _ { 2 } ^ { 2 } .\tag{26}
$$

The regression term penalizes deviations in both direction and scale and therefore limits embedding-magnitude drift

that is not constrained by cosine alignment alone. The loss at horizon k is

$$
\begin{array} { r } { \mathcal { L } ^ { ( k ) } = \alpha \mathcal { L } _ { \mathrm { c o s } } ^ { ( k ) } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { r e g } } ^ { ( k ) } , \qquad 0 \leq \alpha \leq 1 . } \end{array}\tag{27}
$$

The contribution of each horizon is controlled by

$$
w _ { k } = \frac { k ^ { - \gamma } } { \sum _ { j = 1 } ^ { K } j ^ { - \gamma } } , \qquad \gamma \ge 0 .\tag{28}
$$

The inverse-horizon weighting used in this work is obtained with $\gamma = 1$ , while $\gamma = 0$ yields uniform weighting. The complete temporal pretraining objective is

$$
\mathcal { L } _ { \mathrm { t e m p } } = \sum _ { k = 1 } ^ { K } w _ { k } \mathcal { L } ^ { ( k ) } .\tag{29}
$$

Shorter horizons receive stronger supervision for optimization stability, whereas intermediate and longer horizons require the contextual state to retain information that remains predictive beyond adjacent observations. Multi-horizon forecasting is therefore used as a representation-learning constraint in addition to being a prediction objective.

## 3.7. Optimization and Transfer

SPEAR-NeXT is trained after completion of SPEAR spectral pretraining. The spectral encoder is fixed and only the temporal encoder and prediction heads are optimized:

$$
( \theta ^ { \star } , \omega _ { 1 : K } ^ { \star } ) = \arg \operatorname* { m i n } _ { \theta , \omega _ { 1 : K } } \mathcal { L } _ { \mathrm { t e m p } } \quad \mathrm { s u b j e c t } \mathrm { t o } \quad \psi = \psi ^ { \star } .\tag{30}
$$

This modular training procedure separates instantaneous spectral state estimation from temporal state evolution, enables reusable cached embeddings, and avoids instability caused by a moving target representation.

The horizon-specific prediction heads are used only during self-supervised temporal pretraining. For downstream transfer, the contextual representation ${ \bf r } _ { n , t }$ is retained. At a forecasting cutoff t, it is a strictly causal representation derived only from observations available up to that time. For tasks using a complete sequence, the final valid contextual state or a task-specific aggregation over $\{ \mathbf { r } _ { n , t } \} _ { t = 1 } ^ { T _ { n } }$ is supplied to a lightweight downstream model. The SPEAR and temporal encoders may be frozen for representation probing or fine-tuned under a task-specific adaptation protocol.

## 4. Experimental Analysis

This section evaluates whether SPEAR-NeXT learns temporally predictive and geometrically stable pixel-level representations. The analysis is organized around four questions: (i) how forecasting quality changes with prediction horizon, (ii) whether future-state predictions preserve embedding direction and magnitude, (iii) whether the learned dynamics retain seasonal temporal geometry, and (iv) whether gains in downstream transfer arise from temporal pretraining rather than from additional prediction-head capacity. Unless otherwise stated, all forecasting metrics are computed on held-out pixel locations that are not used during model optimization.

## 4.1. Dataset Construction and Preprocessing

Two geographically independent pretraining and evaluation pipelines are constructed for India and the contiguous United States (CONUS). The two regional corpora are not pooled during pretraining. Instead, an independent SPEAR-NeXT model is pretrained for each region and subsequently evaluated only on the downstream datasets associated with that region. This design allows the temporal representations to be assessed under distinct geographic, climatic, agricultural, and land-cover distributions.

The India corpus contains approximately 2.7 million pixel-level samples, whereas the CONUS corpus contains approximately 4.5 million samples. Both corpora cover the period from January 2020 to December 2024 and contain $T = 6 0$ monthly observations per retained pixel location:

$$
T = 1 2 \times 5 = 6 0 .\tag{31}
$$

The complete composition of the two corpora and their associated downstream datasets is summarized in Table 1. Additional dataset descriptions, geographic visualizations, and source-specific details are provided in Appendix A.

Monthly multimodal observations. Let

$$
r \in \mathcal { R } = \{ \mathrm { I n d i a , C O N U S } \}\tag{32}
$$

denote a regional corpus. For pixel location n, the monthly multimodal sequence is defined as

$$
\begin{array} { r } { S _ { n } ^ { ( r ) } = \left\{ \left( \mathbf { x } _ { n , t } ^ { \mathrm { { S } 2 } , ( r ) } , \mathbf { x } _ { n , t } ^ { \mathrm { { S } 1 } , ( r ) } , \mathbf { x } _ { n , t } ^ { \mathrm { E R A 5 } , ( r ) } , \tau _ { t } \right) \right\} _ { t = 1 } ^ { 6 0 } . } \end{array}\tag{33}
$$

Here,

$$
\mathbf { x } _ { n , t } ^ { \mathrm { S 2 } , ( r ) } \in \mathbb { R } ^ { 1 0 }\tag{34}
$$

contains the ten selected Sentinel-2 multispectral bands [8],

$$
\mathbf { x } _ { n , t } ^ { \mathrm { { S 1 } } , ( r ) } \in \mathbb { R } ^ { 2 }\tag{35}
$$

contains Sentinel-1 VV and VH radar backscatter [32], and

$$
\mathbf { x } _ { n , t } ^ { \mathrm { E R A 5 } , ( r ) } \in \mathbb { R } ^ { C _ { \mathrm { c l i m } } }\tag{36}
$$

contains the matched ERA5/ERA5-Land climate and environmental variables [18]. These include the temperature, precipitation, and elevation-related variables used in the SPEAR multimodal representation pipeline [22]. The observations from all three sources are aligned to a common monthly temporal grid.

The original India SPEAR corpus additionally contains PlanetScope SuperDove observations used during multisensor spectral pretraining [21, 22]. PlanetScope is not assumed to be available at every timestep of the uniform 2020–2024 monthly sequence and is therefore not included as a required input to the SPEAR-NeXT temporal model.

Regional pixel-state construction. At every timestep, the three monthly modalities are processed using their corresponding pretrained SPEAR components. The optical representation is obtained using the wavelength-aware Spectral-MAE:

$$
{ \bf e } _ { n , t } ^ { \mathrm { S 2 } , ( r ) } = f _ { \psi _ { \mathrm { S } 2 } } ^ { ( r ) } \left( \mathrm { x } _ { n , t } ^ { \mathrm { S 2 } , ( r ) } ; \lambda , \Delta \lambda \right) ,\tag{37}
$$

where λ and $\pmb { \Delta \lambda }$ denote the Sentinel-2 band-center wavelengths and bandwidths. The radar and environmental representations are

$$
\mathbf { e } _ { n , t } ^ { \mathrm { { S 1 } } , ( r ) } = f _ { \psi _ { \mathrm { S 1 } } } ^ { ( r ) } \left( \mathbf { x } _ { n , t } ^ { \mathrm { { S 1 } } , ( r ) } \right) ,\tag{38}
$$

$$
\mathbf { e } _ { n , t } ^ { \mathrm { E R A 5 } , ( r ) } = f _ { \psi _ { \mathrm { E R A 5 } } } ^ { ( r ) } \left( \mathbf { x } _ { n , t } ^ { \mathrm { E R A 5 } , ( r ) } , \mathbf { g } _ { n } \right) ,\tag{39}
$$

where ${ \bf g } _ { n }$ denotes the geographic metadata associated with pixel n. These representations are produced by the Spectral-MAE, BYOL-Denoise, and Climate-MAE components of SPEAR, respectively [22].

The modality-specific representations are combined using the regional SPEAR fusion operator:

$$
\mathbf { e } _ { n , t } ^ { ( r ) } = \mathcal { F } _ { \phi } ^ { ( r ) } \left( \mathbf { e } _ { n , t } ^ { \mathrm { S 2 } , ( r ) } , \mathbf { e } _ { n , t } ^ { \mathrm { S 1 } , ( r ) } , \mathbf { e } _ { n , t } ^ { \mathrm { E R A 5 } , ( r ) } \right) , \qquad \mathbf { e } _ { n , t } ^ { ( r ) } \in \mathbb { R } ^ { 3 2 } .\tag{40}
$$

The resulting temporal input to SPEAR-NeXT is

$$
\mathbf { E } _ { n } ^ { ( r ) } = \left[ \mathbf { e } _ { n , 1 } ^ { ( r ) } , \mathbf { e } _ { n , 2 } ^ { ( r ) } , \ldots , \mathbf { e } _ { n , 6 0 } ^ { ( r ) } \right] ^ { \top } \in \mathbb { R } ^ { 6 0 \times 3 2 } .\tag{41}
$$

The modality encoders and fusion module are frozen during temporal pretraining, and the extracted monthly pixel states are cached before training the corresponding regional temporal model.

Preprocessing and quality control. Only pixel locations satisfying the required temporal-coverage and dataquality criteria are retained. Invalid and cloud-contaminated Sentinel-2 observations are removed during optical preprocessing. Sentinel-1 VV and VH observations are processed consistently in the calibrated backscatter domain, while ERA5/ERA5-Land variables are matched to the same pixel location and monthly temporal index.

A separate preprocessing transformation is fitted for each modality and region. For modality

$$
m \in \{ \mathrm { S 2 , S 1 , E R A 5 } \} ,\tag{42}
$$

Table 1. Summary of the two independent regional pretraining and evaluation pipelines. Each region is used to pretrain an independent SPEAR-NeXT model, and no samples or model parameters are shared across regions.
<table><tr><td>Region</td><td>Model</td><td>Pretraining size</td><td>Temporal span</td><td>Pretraining modalities</td><td>Representation</td><td>Downstream datasets</td></tr><tr><td>India</td><td>SPEAR-NeXTIndia</td><td>~2.7M pixels</td><td>2020–2024 (60 months)</td><td>Sentinel-2 [8], Sentinel-1 [32], ERA5-Land [18]</td><td>60 × 32 fused pixel states</td><td>SICKLE [28], Sen1Floods11 [4], VIIRS [29], Dynamic World [5]</td></tr><tr><td>CONUS</td><td>SPEAR-NeXTCONUS</td><td>~4.5M pixels</td><td>2020–2024 (60 months)</td><td>Sentinel-2 [8], Sentinel-1 [32], ERA5-Land [18]</td><td>60 × 32 fused pixel states</td><td>USDA NASS [35], CropHarvest [33], Sen1Floods11 [4], Dynamic World [5]</td></tr></table>

the standardized observation is

$$
\widetilde { \mathbf { x } } _ { n , t } ^ { m , ( r ) } = \mathcal { T } _ { m } ^ { ( r ) } \left( \mathbf { x } _ { n , t } ^ { m , ( r ) } \right) ,\tag{43}
$$

where the RobustScaler transformation $\mathcal { T } _ { m } ^ { ( r ) }$ is fitted exclusively on the corresponding regional training partition. The fitted transformation is then applied unchanged to the validation and test partitions. No normalization statistics are shared between India and CONUS, and future observations are never used to normalize or impute earlier timestamps.

Leakage-controlled regional partitioning. Each regional corpus is independently divided into training, validation, and test sets using a 60:20:20 split:

$$
\mathcal { D } ^ { ( r ) } = \mathcal { D } _ { \mathrm { t r a i n } } ^ { ( r ) } \cup \mathcal { D } _ { \mathrm { v a l } } ^ { ( r ) } \cup \mathcal { D } _ { \mathrm { t e s t } } ^ { ( r ) } ,\tag{44}
$$

with

$$
\left| \mathcal { D } _ { \mathrm { t r a i n } } ^ { ( r ) } \right| : \left| \mathcal { D } _ { \mathrm { v a l } } ^ { ( r ) } \right| : \left| \mathcal { D } _ { \mathrm { t e s t } } ^ { ( r ) } \right| = 6 0 : 2 0 : 2 0 .\tag{45}
$$

Partitioning is performed at the pixel-location level rather than at the individual pixel–month level. Therefore, all 60 monthly observations belonging to one pixel are assigned to the same partition. Test pixels are kept geographically separate from training pixels to prevent spatial leakage from neighbouring locations with highly correlated spectral, radar, climate, and temporal trajectories.

Temporal leakage is prevented by ensuring that no observation belonging to a held-out test sequence contributes to model optimization, target construction, normalization, model selection, or early stopping. When a downstream experiment includes an explicit temporal holdout, the corresponding years or monthly periods are also excluded from the training partition. Thus, the reported test results are independent at both the pixel-location and temporalevaluation levels.

Independent regional models and evaluations. The India and CONUS temporal models are optimized independently:

$$
\left( \theta _ { \mathrm { I n d i a } } ^ { \star } , \omega _ { \mathrm { I n d i a } , 1 : K } ^ { \star } \right) = \arg \operatorname* { m i n } _ { \theta , \omega _ { 1 : K } } \mathcal { L } _ { \mathrm { t e m p } } ^ { \mathrm { I n d i a } } ,\tag{46}
$$

$$
\left( \theta _ { \mathrm { C O N U S } } ^ { \star } , \omega _ { \mathrm { C O N U S } , 1 : K } ^ { \star } \right) = \arg \operatorname* { m i n } _ { \theta , \omega _ { 1 : K } } \mathcal { L } _ { \mathrm { t e m p } } ^ { \mathrm { C O N U S } } .\tag{47}
$$

No temporal-model parameters, pretraining samples, validation samples, or test samples are shared between the two regional pipelines.

The India-pretrained model is evaluated on SICKLE crop classification [28], Sen1Floods11 flood detection [4], VIIRS-based fire-related classification [29], and Dynamic World land-cover classification [5]. The CONUSpretrained model is evaluated separately on USDA NASS crop-yield prediction [35], CropHarvest crop/non-crop classification [33], Sen1Floods11 flood detection [4], and Dynamic World land-cover classification [5].

Model selection and early stopping use only the validation partition of the corresponding region. Forecasting, representation, and downstream performance are reported using that region’s held-out test data. Detailed descriptions and geographic visualizations of the pretraining and downstream datasets are provided in Appendix A.

## 4.2. Training Configuration

The temporal encoder contains $L = 4$ causal Transformer layers with temporal model dimension $D _ { T } ~ = ~ 1 2 8$ and $A = 4$ self-attention heads. Direct prediction is performed for $K \ : = \ : 4 6$ future sequence steps using horizon-specific prediction heads. The SPEAR encoder remains frozen throughout temporal pretraining, ensuring that improvements can be attributed to temporal representation learning and that the target embedding space does not drift during optimization.

The temporal encoder and prediction heads are optimized using AdamW with linear learning-rate warmup, gradient clipping, and mixed-precision training. An exponential moving average (EMA) copy of the temporal parameters is maintained and used for validation and final evaluation. The composite objective in Eq. (29) uses $\alpha = 0 . 5 ,$ assigning equal weight to cosine alignment and latent vector regression. Unless otherwise stated, inverse-horizon weighting is used with $\gamma = 1 \colon$

$$
w _ { k } = \frac { k ^ { - 1 } } { \sum _ { j = 1 } ^ { K } j ^ { - 1 } } .\tag{48}
$$

The exact learning rate, weight decay, batch size, warmup duration, gradient-clipping threshold, EMA decay, number of epochs, and early-stopping patience are reported in Table 2.

The temporal predictor sits atop a frozen SPEAR spectral encoder and treats each pixel’s monthly embedding sequence as input to a causal transformer. Table 3 summarizes the full architectural configuration. We adopt a small set of stabilization and efficiency techniques originally developed for large language models, RoPE for relative positional encoding, QK-Norm for attention stability, and SwiGLU feedforward layers, motivated by the fact that satellite pixel time series, unlike text corpora, offer a comparatively data-scarce training regime where such stability tricks matter more, not less. The core contribution is Multi-Horizon Prediction (MHP): rather than predicting only the next-step embedding, the model attaches $K { = } 4 6$ independent lightweight MLP heads to the shared transformer trunk, one per horizon $k = 1 , \ldots , 4 6$ , each trained to forecast the SPEAR embedding at $t + k$ from context up to t. Horizons are weighted inversely $( w _ { k } \propto 1 / k )$ so that near-term, higherconfidence predictions dominate the loss while long-range horizons still receive gradient signal. Each horizon’s loss combines a cosine (directional) term and an MSE (magnitude) term, balanced by α. Targets are stop-gradiented from the same frozen SPEAR encoder. This is BYOLinspired in spirit (no explicit target reconstruction, prediction in representation space) but does not use BYOL’s dual online/momentum-teacher architecture; the EMA copy of the model maintained during training is used only for stable validation and inference, not as a separate target network. Because the SPEAR backbone remains fully frozen throughout this pretraining stage, all representational adaptation for temporal dynamics happens exclusively within the lightweight NEPA transformer head, keeping the additional training cost modest relative to the frozen encoder.

Table 2. Temporal pretraining configuration of the SPEAR-NeXT temporal module.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Spectral embedding dimension D</td><td>32</td></tr><tr><td>Temporal model dimension  $D _ { T }$ </td><td>128</td></tr><tr><td>Transformer layers L</td><td>4</td></tr><tr><td>Attention heads A</td><td>4</td></tr><tr><td>Maximum prediction horizon K</td><td>46</td></tr><tr><td>Loss balance α</td><td>0.5</td></tr><tr><td>Horizon exponent γ</td><td>1</td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 )$ </td></tr><tr><td>Base learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Warmup epochs</td><td>5</td></tr><tr><td>Gradient-clipping threshold</td><td>1.0</td></tr><tr><td>EMA decay</td><td>0.999</td></tr><tr><td>Training epochs</td><td>20</td></tr><tr><td>Early-stopping patience</td><td> $5 ( \mathrm { m i n } . \Delta = 1 \times 1 0 ^ { - 5 } )$ </td></tr></table>

Table 3. Architectural details of the SPEAR-NeXT multi-horizon temporal predictor built on frozen SPEAR representations [22].
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Backbone</td><td>Frozen SPEAR encoder (D = 32)</td></tr><tr><td>Temporal encoding</td><td>Learned month and year embeddings (16 + 16-d)</td></tr><tr><td>Positional encoding</td><td>RoPE  $( \theta = 1 0 , 0 0 0 )$ </td></tr><tr><td>Attention stabilization</td><td>QK-Norm</td></tr><tr><td>Attention masking</td><td>Causal</td></tr><tr><td>Feed-forward network</td><td>SwiGLU with hidden dimension 512</td></tr><tr><td>Encoder layers L</td><td>4</td></tr><tr><td>Attention heads A</td><td>4 (head dimension = 32)</td></tr><tr><td>Model dimension  $D _ { T }$ </td><td>128</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Prediction horizons K</td><td>46 independent prediction heads</td></tr><tr><td>Horizon weighting Loss function</td><td> $w _ { k } \propto 1 / k$ </td></tr><tr><td></td><td> $\alpha \mathcal { L } _ { \mathrm { c o s } } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { M S E } } ,$  computed independently at each horizon</td></tr><tr><td>Target handling</td><td>Stop-gradient target embeddings (BYOL-style)</td></tr></table>

## 4.3. Forecasting Baselines and Ablation Protocol

Embedding forecasts are compared against simple temporal baselines to determine whether the learned model improves upon continuity and seasonality priors.

Persistence. The most recent observed state is copied to every future horizon:

$$
\widehat { \mathbf { e } } _ { n , t + k | t } ^ { \mathrm { p e r s } } = \mathbf { e } _ { n , t } .\tag{49}
$$

Linear latent extrapolation. The most recent latent displacement is extended to horizon k:

$$
\mathbf { \widehat { e } } _ { n , t + k | t } ^ { \mathrm { l i n } } = \mathbf { e } _ { n , t } + k \left( \mathbf { e } _ { n , t } - \mathbf { e } _ { n , t - 1 } \right) , \qquad t \geq 2 .\tag{50}
$$

Seasonal persistence. When the sequence cadence admits a known seasonal period $P$ and the corresponding state lies within the observed context, the state from the previous cycle is used:

$$
\widehat { \mathbf { e } } _ { n , t + k | t } ^ { \mathrm { s e a s o n } } = \mathbf { e } _ { n , t + k - P } , \qquad t + k - P \leq t .\tag{51}
$$

For monthly composites, $P = 1 2$

The temporal objective is further examined using controlled ablations: cosine-only versus regression-only versus composite loss; uniform versus inverse-horizon weighting; single-step versus multi-horizon prediction; RoPEonly, calendar-only, and combined temporal conditioning; and models trained with or without QK normalization. Al ablations retain the same data partitions, optimizer family, temporal capacity, and downstream evaluation heads.

## 4.4. Horizon-Wise Evaluation Metrics

For horizon $k ,$ metrics are computed over the valid held-out context–target set

$$
\Omega _ { k } ^ { \mathrm { t e s t } } = \left\{ ( n , t ) \ : | \ : n \in \mathcal { D } _ { \mathrm { t e s t } } , \ : 1 \leq t \leq T - k \right\} .\tag{52}
$$

Let $\widehat { \mathbf { e } } _ { n , t + k | t }$ and $\mathbf { e } _ { n , t + k } ^ { \star }$ denote the prediction and frozen SPEAR target.

Cosine similarity. Directional agreement is measured by

$$
\begin{array} { r l } { \mathrm { C o s S i m } _ { k } = \displaystyle \frac { 1 } { | \Omega _ { k } ^ { \mathrm { t e s t } } | } \sum _ { ( n , t ) \in \Omega _ { k } ^ { \mathrm { t e s t } } } } & { } \\ { \displaystyle \frac { \widehat { \mathbf { e } } _ { n , t + k | t } ^ { \top } \mathbf { e } _ { n , t + k } ^ { \star } } { \big ( \| \widehat { \mathbf { e } } _ { n , t + k | t } \| _ { 2 } + \varepsilon \big ) \big ( \| \mathbf { e } _ { n , t + k } ^ { \star } \| _ { 2 } + \varepsilon \big ) } . } & { } \end{array}\tag{53}
$$

Mean squared and mean absolute error. Latent reconstruction errors are

$$
\mathrm { M S E } _ { k } = \frac { 1 } { D \mathopen { } \mathclose \bgroup \left| \Omega _ { k } ^ { \mathrm { t e s t } } \aftergroup \egroup \right| } \sum _ { ( n , t ) \in \Omega _ { k } ^ { \mathrm { t e s t } } } \left\| \widehat { \mathbf { e } } _ { n , t + k \mid t } - \mathbf { e } _ { n , t + k } ^ { \star } \right\| _ { 2 } ^ { 2 } ,\tag{54}
$$

$$
\mathrm { M A E } _ { k } = \frac { 1 } { D | \Omega _ { k } ^ { \mathrm { t e s t } } | } \sum _ { ( n , t ) \in \Omega _ { k } ^ { \mathrm { t e s t } } } \left. \widehat { \mathbf { e } } _ { n , t + k | t } - \mathbf { e } _ { n , t + k } ^ { \star } \right. _ { 1 } .\tag{55}
$$

Coefficient of determination. To avoid high-variance latent dimensions dominating the score, $R ^ { 2 }$ is computed independently for each embedding coordinate and macroaveraged:

$$
R _ { k } ^ { 2 } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \left[ 1 - \frac { \sum _ { ( n , t ) \in \Omega _ { k } ^ { \mathrm { t e s t } } } \left( \widehat { e } _ { n , t + k \mid t , d } - e _ { n , t + k , d } ^ { \star } \right) ^ { 2 } } { \sum _ { ( n , t ) \in \Omega _ { k } ^ { \mathrm { t e s t } } } \left( e _ { n , t + k , d } ^ { \star } - \overline { { e } } _ { k , d } ^ { \star } \right) ^ { 2 } + \varepsilon } \right]\tag{56}
$$

where $\bar { e } _ { k , d } ^ { \star }$ is the mean target value of coordinate d over $\Omega _ { k } ^ { \mathrm { t e s t } }$

For any horizon-specific metric $m _ { k }$ , the aggregate score is

$$
\overline { { m } } _ { w } = \sum _ { k = 1 } ^ { K } w _ { k } m _ { k } ,\tag{57}
$$

using the same normalized horizon weights as temporal pretraining. Per-horizon curves are always reported alongside aggregate scores because weighted averages can conceal long-horizon failure.

## 4.5. Horizon-Wise Forecasting Behaviour

Forecasting quality is analyzed as a function of k using $\mathrm { C o s S i m } _ { k } , \mathrm { M S E } _ { k } , \mathrm { M A E } _ { k } .$ , and $R _ { k } ^ { 2 }$ . Because the model predicts each horizon directly, longer-horizon degradation is not attributed to recursive error accumulation. Instead, it reflects increasing uncertainty, weaker dependence between the available context and distant targets, and the smaller number of valid context–target pairs available near the end of each sequence.

The horizon curves are compared with the persistence, linear-extrapolation, and seasonal-persistence baselines. Improvement over persistence indicates that the temporal encoder learns state evolution beyond local continuity, while improvement over seasonal persistence indicates that it captures context-dependent departures from a fixed annual cycle. Results should be reported over short, intermediate, and long horizon groups in addition to individual horizons.

## 4.6. Embedding Magnitude Drift and Calibration

Cosine similarity can remain high even when predicted embeddings have incorrect scale. Two complementary statistics are therefore computed. Relative norm bias is defined as

$$
\Delta _ { k } ^ { \mathrm { n o r m } } = \frac { \mathbb { E } _ { \Omega _ { k } ^ { \mathrm { t e s t } } } \left[ \left. \widehat { \mathbf { e } } _ { n , t + k \mid t } \right. _ { 2 } \right] - \mathbb { E } _ { \Omega _ { k } ^ { \mathrm { t e s t } } } \left[ \left. \mathbf { e } _ { n , t + k } ^ { \star } \right. _ { 2 } \right] } { \mathbb { E } _ { \Omega _ { k } ^ { \mathrm { t e s t } } } \left[ \left. \mathbf { e } _ { n , t + k } ^ { \star } \right. _ { 2 } \right] + \varepsilon } .\tag{58}
$$

The normalized absolute norm error is

$$
\mathrm { N o r m E r r } _ { k } = \mathbb { E } _ { \Omega _ { k } ^ { \mathrm { t e s t } } } \left[ \frac { \left| \left\| \widehat { \mathbf { e } } _ { n , t + k \mid t } \right\| _ { 2 } - \left\| \mathbf { e } _ { n , t + k } ^ { \star } \right\| _ { 2 } \right| } { \left\| \mathbf { e } _ { n , t + k } ^ { \star } \right\| _ { 2 } + \varepsilon } \right] .\tag{59}
$$

Post-hoc affine calibration is used only as a diagnostic of whether residual error is dominated by first- and secondorder scale mismatch. For each horizon and embedding coordinate, calibration statistics are estimated on the validation set:

$$
\widetilde { e } _ { n , t + k | t , d } = \mu _ { k , d } ^ { \star } + \sigma _ { k , d } ^ { \star } \frac { \widehat { e } _ { n , t + k | t , d } - \widehat { \mu } _ { k , d } } { \widehat { \sigma } _ { k , d } + \varepsilon } ,\tag{60}
$$

where $\left( \widehat { \mu } _ { k , d } , \widehat { \sigma } _ { k , d } \right)$ and $( \mu _ { k , d } ^ { \star } , \sigma _ { k , d } ^ { \star } )$ are the validation-set moments of predicted and target embeddings, respectively. The fitted transformation is then applied unchanged to the test predictions. Improvements after calibration are reported as diagnostic evidence of scale or offset mismatch; calibrated results are not treated as the primary performance of the unmodified model.

## 4.7. Temporal Self-Similarity Preservation

Seasonal and recurrent structure is examined through temporal self-similarity. The target self-similarity matrix is

$$
S _ { i , j } ^ { \star } = \frac { 1 } { N _ { i , j } } \sum _ { n \in \mathcal { T } _ { i , j } } \frac { \mathbf { e } _ { n , i } ^ { \star \top } \mathbf { e } _ { n , j } ^ { \star } } { \left( \left\| \mathbf { e } _ { n , i } ^ { \star } \right\| _ { 2 } + \varepsilon \right) \left( \left\| \mathbf { e } _ { n , j } ^ { \star } \right\| _ { 2 } + \varepsilon \right) } ,\tag{61}
$$

where $\mathcal { T } _ { i , j }$ contains test sequences with valid states at both temporal positions and $N _ { i , j } = | \mathcal { T } _ { i , j } |$

For a fixed horizon $k ,$ define the prediction associated with target position $j$ as

$$
\widehat { \mathbf { e } } _ { n , j } ^ { ( k ) } = \widehat { \mathbf { e } } _ { n , j | j - k } , \qquad j > k .\tag{62}
$$

The predicted self-similarity matrix is then

$$
\begin{array} { r } { \widehat { S } _ { i , j } ^ { ( k ) } = \mathbb { E } \left[ \cos \left( \widehat { \mathbf { e } } _ { n , i } ^ { ( k ) } , \widehat { \mathbf { e } } _ { n , j } ^ { ( k ) } \right) \right] . } \end{array}\tag{63}
$$

Similarity-geometry preservation is summarized by the normalized Frobenius discrepancy

$$
\mathcal { D } _ { \mathrm { s i m } } ^ { ( k ) } = \frac { \left\| \widehat { \mathbf { S } } ^ { ( k ) } - \mathbf { S } ^ { \star } \right\| _ { F } } { \left\| \mathbf { S } ^ { \star } \right\| _ { F } + \varepsilon } ,\tag{64}
$$

computed over the temporal indices valid for horizon $k .$ Heatmaps are inspected for diagonal continuity, repeated off-diagonal bands, and seasonal recurrence. Such structure is interpreted as temporal regularity in the learned latent space rather than as direct evidence of a specific biological process unless supported by corresponding labels or phenological observations.

## 4.8. Prediction–Target Alignment

To determine whether the model predicts the correct future position rather than returning a temporally generic or identity-like embedding, prediction–target alignment matrices are computed:

$$
H _ { i , j } ^ { ( k ) } = \mathbb { E } _ { n } \left[ \cos \left( \widehat { \mathbf { e } } _ { n , i + k | i } , \mathbf { e } _ { n , j } ^ { \star } \right) \right] .\tag{65}
$$

A correctly aligned horizon-k forecast should concentrate similarity near $j = i + k$ . Diagonal concentration is quantified as

$$
\mathrm { D C } _ { k } = \frac { 1 } { | \mathcal { T } _ { k } | } \sum _ { i \in \mathcal { T } _ { k } } \left[ H _ { i , i + k } ^ { ( k ) } - \frac { 1 } { T - 1 } \sum _ { j = 1 \atop j \neq i + k } ^ { T } H _ { i , j } ^ { ( k ) } \right] ,\tag{66}
$$

where $\mathcal { T } _ { k }$ contains valid forecast origins. Positive and increasing diagonal concentration relative to the baselines indicates that the model has learned temporally specific progression rather than a degenerate constant or persistence solution.

## 4.9. Loss and Temporal-Encoding Ablations

The composite objective is evaluated against cosine-only and regression-only variants. Cosine-only optimization tests whether directional alignment is sufficient, while regression-only optimization tests whether Euclidean reconstruction alone preserves latent geometry. The composite loss is assessed using the complete set of horizonwise metrics, norm-drift statistics, and self-similarity discrepancy.

Temporal conditioning is evaluated using the following controlled variants:

1. learned absolute position embeddings without calendar conditioning;

2. RoPE without month–year embeddings;

3. month–year embeddings without RoPE;

4. combined RoPE and month–year conditioning;

5. the complete model without QK normalization.

RoPE and calendar conditioning are not treated as interchangeable. The former encodes ordering and relative displacement within the sequence, whereas the latter identifies seasonal phase and inter-annual context. The combined model is expected to be most useful when similar relative offsets correspond to different seasonal regimes.

The role of multi-horizon supervision is examined by comparing the complete model with a single-step model trained only at $k = 1$ and with a uniformly weighted multihorizon model. This determines whether long-range supervision improves the transferable representation or merely increases forecasting difficulty.

## 4.10. Downstream Transfer and Representation Ablations

Because the abstract and central claims concern reusable Earth representations, embedding diagnostics are complemented by downstream transfer experiments on land-cover mapping and temporally demanding agricultural tasks. The same lightweight downstream architecture and data split are used for all representation variants. Three principal configurations are compared:

1. SPEAR: frozen instantaneous spectral embeddings without temporal pretraining;

2. SPEAR + untrained temporal module: the same temporal architecture with randomly initialized or tasktrained temporal parameters, controlling for additional model capacity;

3. SPEAR-NeXT: frozen or lightly adapted representations from the self-supervised temporal model.

Where applicable, a prediction-head-only ablation is also included to separate the contribution of the pretrained temporal backbone from that of the downstream head.

Classification tasks are evaluated using overall accuracy and macro-averaged F1 score, with per-class results reported for imbalanced datasets. Regression tasks are evaluated using MAE, RMSE, $R ^ { 2 }$ , and, where appropriate, percentage-based error. Geographic holdouts are used whenever the task is intended to assess cross-region transfer. Each reported gain is therefore interpreted relative to a capacity-matched baseline and not solely relative to a weaker downstream model.

Together, the forecasting diagnostics and downstream ablations test whether causal multi-horizon latent prediction produces representations that are not only predictable in embedding space but also more useful for Earth-observation transfer.

## 5. Results and Discussion

This section reports the empirical behaviour of SPEAR-NeXT at three complementary levels: (i) future-state forecasting accuracy across temporal horizons, (ii) preservation of latent magnitude and temporal geometry, and (iii)

transfer of the learned representations to downstream Earthobservation tasks. All forecasting results are computed on the spatially disjoint test partition described in Section 4.1. Unless otherwise stated, the EMA model is used for evaluation, the SPEAR encoder remains frozen, and aggregate values use the normalized inverse-horizon weights in Eq. (57). Bracketed entries indicate values or figures to be inserted after the final experimental run.

## 5.1. Downstream Representation Transfer

The practical value of SPEAR-NeXT is evaluated by transferring the learned temporal representations to general land-cover mapping and temporally demanding agricultural tasks. The India and CONUS models are pretrained independently and are evaluated only on their corresponding regional downstream datasets. Therefore, the India and CONUS results reported below represent two separate regional learning pipelines rather than direct cross-region transfer by a single model.

Where possible, the same downstream head, label split, and optimization protocol are retained across representation models. This isolates differences in pretrained representation quality from differences in downstream model capacity.

## 5.1.1. Regional Land-Cover Classification

Land-cover classification evaluates whether the temporally predictive representations remain useful for general Earthsurface mapping rather than only for agriculture-specific tasks. SPEAR-NeXT is compared with Presto [34] and Tessera [9] using an MLP classification head under the corresponding India and CONUS evaluation protocols.

On the India evaluation, SPEAR-NeXT achieves an accuracy of 94.81% and an F1 score of 94.79%. Relative to the strongest competing baseline, Presto, these results correspond to gains of 9.61 and 9.92 percentage points, respectively. The larger difference relative to Tessera suggests that the India-pretrained SPEAR-NeXT representation captures temporal and multimodal structure that is particularly useful under the India land-cover distribution.

On the CONUS evaluation, SPEAR-NeXT obtains an accuracy of 88.78% and an F1 score of 90.66%. The corresponding gains over the strongest baseline, Tessera, are 2.38 percentage points in accuracy and 2.13 percentage points in F1 score. Although the margin is smaller than that observed in India, SPEAR-NeXT remains the strongest model under both regional evaluation settings. These results indicate that the benefit of predictive temporal pretraining is not restricted to crop-specific tasks.

## 5.1.2. SICKLE Multi-Task Agricultural Transfer

The India-pretrained model is further evaluated on the SICKLE benchmark [28], which includes crop classification and agricultural regression tasks involving sowing date, harvest date, and yield as shown in table 5 . This multi-task setting tests whether the representation captures not only crop identity but also temporally structured agronomic information.

SPEAR-NeXT achieves the best performance on all three continuous agricultural tasks. For sowing-date estimation, the MAPE decreases from 1.58% for the strongest baseline to 1.25%, corresponding to a relative error reduction of approximately 20.9%. For harvest-date estimation, SPEAR-NeXT reduces MAPE from 4.76% to 3.03%, a relative reduction of approximately 36.3%. Yield-prediction MAPE decreases from 22.80% for Tessera to 21.70%, corresponding to a relative reduction of approximately 4.8%.

SPEAR-NeXT does not obtain the highest binaryclassification score: Tessera achieves an F1 score of 91.63%, compared with 90.30% for SPEAR-NeXT. The 1.33 percentage-point difference indicates that the proposed representation is not uniformly dominant across all downstream objectives. However, its consistently stronger performance on sowing date, harvest date, and yield suggests that causal multi-horizon pretraining is particularly beneficial for tasks whose targets depend on temporal progression rather than only on crop-category separability.

## 5.1.3. Crop Classification under Fusion and Adaptation Strategies

Table 6 shows that SPEAR-NeXT outperforms SPEAR in nine of the twelve evaluated configurations. The best overall performance is obtained using concatenation, an MLP head, and a frozen encoder, reaching 83.33% accuracy and 0.8279 macro-F1. Its 5.77 percentage-point accuracy gain over SPEAR indicates that the pretrained temporal representation transfers effectively without end-to-end finetuning.

The largest improvements occur with cross-attention fusion. Under MLP–PEFT, accuracy increases from 51.28% to 78.85%, while RF–PEFT improves from 56.41% to 78.85%. In contrast, concatenation produces smaller and less consistent gains. Overall, the results suggest that SPEAR-NeXT provides strong frozen representations while also enabling more effective higher-capacity multimodal fusion.

## 5.1.4. USDA-NASS Crop-Yield Prediction

The CONUS-pretrained model is evaluated for cropspecific yield prediction using USDA NASS data [35]. Table 7 reports the coefficient of determination for the 2023 evaluation set. In the original presentation, the proposed model was denoted as GeoSutra; it is referred to consistently here as SPEAR-NeXT.

SPEAR-NeXT obtains the highest $R ^ { 2 }$ for every crop. Relative to the strongest baseline for each crop, the absolute improvements are 0.129 for corn, 0.085 for soybean, 0.053 for wheat, and 0.025 for cotton. The mean crop-wise

Table 4. Land-cover classification using the independently pretrained India and CONUS models. All methods use an MLP downstream head. Higher values are better. Best results are shown in bold.
<table><tr><td colspan="2"></td><td colspan="2">India</td><td colspan="2">CONUS</td></tr><tr><td>Model</td><td>Head</td><td>Accuracy (%) ↑</td><td>F1 (%) ↑</td><td>Accuracy (%) ↑</td><td>F1 (%) ↑</td></tr><tr><td>SPEAR-NeXT</td><td>MLP</td><td>94.81</td><td>94.79</td><td>88.78</td><td>90.66</td></tr><tr><td>Presto [34]</td><td>MLP</td><td>85.20</td><td>84.87</td><td>86.17</td><td>86.61</td></tr><tr><td>TESSERA [9]</td><td>MLP</td><td>72.35</td><td>74.23</td><td>86.40</td><td>88.53</td></tr></table>

Table 5. Multi-task transfer on SICKLE [28]. Higher is better for binary crop-classification F1, while lower is better for MAPE. Best results are shown in bold and second-best results are underlined.
<table><tr><td>Model</td><td>Binary classification F1 (%) ↑</td><td>Sowing-date MAPE (%) ↓</td><td>Harvest-date MAPE (%) ↓</td><td>Yield MAPE (%) ↓</td></tr><tr><td>U-Net [25]</td><td>87.71</td><td>2.34</td><td>4.76</td><td>72.38</td></tr><tr><td>Time2Agri [12]</td><td>82.08</td><td>1.58</td><td>6.81</td><td>36.77</td></tr><tr><td>Presto [34]</td><td>89.87</td><td>5.05</td><td>5.93</td><td>23.30</td></tr><tr><td>TESSERA [9]</td><td>91.63</td><td>4.13</td><td>7.73</td><td>22.80</td></tr><tr><td>SPEAR-NeXT</td><td>90.30</td><td>1.25</td><td>3.03</td><td>21.70</td></tr></table>

$R ^ { 2 }$ increases from 0.644 for Tessera and 0.613 for Presto to 0.721 for SPEAR-NeXT.

The largest improvement is observed for corn, while the smallest is observed for cotton. This variation suggests that the value of temporal representation learning depends on crop-specific signal quality, sample availability, and the predictability of seasonal development from the available satellite and environmental observations. The consistent improvement across all four crops nevertheless indicates that the CONUS-pretrained representation transfers beyond classification to continuous agricultural prediction.

## Contribution of the pretrained temporal representation.

To determine whether the improvements in crop-yield prediction originate from the self-supervised temporal representation or only from the downstream prediction head, the complete SPEAR-NeXT pipeline is compared with a prediction-head-only control. The full model uses the frozen SPEAR state encoder, the pretrained SPEAR-NeXT temporal backbone, and the crop-specific prediction head. The control retains the same prediction head but excludes the pretrained temporal representation module. Both configurations are evaluated using the same crop-specific data partitions, forecast dates, and downstream optimization protocol.

As shown in Figure 3, the complete SPEAR-NeXT model outperforms the prediction-head-only control for all four crop types. The absolute $R ^ { 2 }$ improvements are 0.437 for corn, 0.175 for soybean, 0.088 for wheat, and 0.068 for cotton. The largest gain is observed for corn, where the complete model increases $R ^ { 2 }$ from 0.289 to 0.726. This substantial difference indicates that the downstream head alone is insufficient to recover the temporal information required for accurate yield prediction.

The improvements for soybean, wheat, and cotton are smaller but remain consistent, showing that the benefit of the pretrained temporal representation is not confined to one crop. Because the two configurations use the same prediction-head architecture and evaluation protocol, the performance difference can be attributed primarily to the representation learned through self-supervised temporal pretraining rather than to additional downstreamhead capacity. These results provide direct evidence that SPEAR-NeXT learns transferable temporal features that improve crop-yield prediction beyond a task-specific prediction head.

Effect of spatial kriging. We further examine whether the crop-yield performance originates from the learned representation or from the subsequent spatial interpolation stage. Thefull model consists of the frozen SPEAR encoder, the pretrained SPEAR-NeXT temporal encoder, and the downstream yield-prediction head. The prediction-headonly configuration uses the same downstream head without the pretrained SPEAR-NeXT temporal representation. A post-hoc kriging stage is then applied independently to the predictions produced by each configuration.

Figure 4 reveals a clear difference in how spatial kriging affects the two model configurations. For the complete SPEAR-NeXT model, the improvement is modest or negligible: $R ^ { 2 }$ increases by 0.066 for the September 2 forecast and remains effectively unchanged for the August 5 forecast. This suggests that the pretrained spectral and temporal representations already capture much of the spatially structured signal relevant to corn-yield prediction, leaving kriging to perform only a limited refinement.

Table 6. Crop-classification performance of SPEAR and SPEAR-NeXT under different multimodal fusion, prediction-head, and adaptation strategies. All configurations are evaluated using 200 labeled samples from five crop classes. SPEAR denotes the previously published pixel-level spectral and multimodal representation model [22]. Concat denotes feature concatenation, whereas Cross denotes cross-attention-based feature fusion. MLP denotes a multilayer perceptron prediction head, RF denotes a random forest prediction head, and PEFT denotes parameter-efficient fine-tuning. Frozen indicates that the pretrained representation encoder is kept fixed during downstream training, whereas Unfrozen permits end-to-end parameter updating. The SPEAR macro-F1 values represent the conservative lower bound of the observed approximately 3–5% relative difference from the corresponding SPEAR-NeXT results. Higher accuracy and macro-F1 values indicate better classification performance.
<table><tr><td rowspan="2">Fusion</td><td rowspan="2">Head</td><td rowspan="2">Adaptation</td><td rowspan="2">Samples</td><td rowspan="2">Classes</td><td colspan="2">Accuracy (%) ↑</td><td colspan="2">Macro-F1 ↑</td></tr><tr><td>SPEAR</td><td>SPEAR-NeXT</td><td>SPEAR</td><td>SPEAR-NeXT</td></tr><tr><td rowspan="6">Concat</td><td rowspan="3">MLP</td><td>Frozen</td><td>200</td><td>5</td><td>77.56</td><td>83.33</td><td>0.7865</td><td>0.8279</td></tr><tr><td>Unfrozen</td><td>200</td><td>5</td><td>80.77</td><td>82.05</td><td>0.7738</td><td>0.8145</td></tr><tr><td>PEFT</td><td>200</td><td>5</td><td>81.41</td><td>80.13</td><td>0.7593</td><td>0.7993</td></tr><tr><td rowspan="3">RF</td><td>Frozen</td><td>200</td><td>5</td><td>74.36</td><td>76.92</td><td>0.7190</td><td>0.7568</td></tr><tr><td>Unfrozen</td><td>200</td><td>5</td><td>77.56</td><td>75.00</td><td>0.7029</td><td>0.7399</td></tr><tr><td>PEFT</td><td>200</td><td>5</td><td>76.28</td><td>74.36</td><td>0.6929</td><td>0.7294</td></tr><tr><td rowspan="6">Cross</td><td rowspan="3">MLP</td><td>Frozen</td><td>200</td><td>5</td><td>68.59</td><td>71.79</td><td>0.6464</td><td>0.6804</td></tr><tr><td>Unfrozen</td><td>200</td><td>5</td><td>63.46</td><td>80.77</td><td>0.5976</td><td>0.6290</td></tr><tr><td>PEFT</td><td>200</td><td>5</td><td>51.28</td><td>78.85</td><td>0.4734</td><td>0.4983</td></tr><tr><td rowspan="3">RF</td><td>Frozen</td><td>200</td><td>5</td><td>63.46</td><td>66.03</td><td>0.5950</td><td>0.6263</td></tr><tr><td>Unfrozen</td><td>200</td><td>5</td><td>67.31</td><td>76.56</td><td>0.6324</td><td>0.6657</td></tr><tr><td>PEFT</td><td>200</td><td>5</td><td>56.41</td><td>78.85</td><td>0.5292</td><td>0.5571</td></tr></table>

Table 7. Crop-specific USDA-NASS yield prediction for the 2023 evaluation year [35]. Values report $R ^ { \bar { 2 } }$ ; higher is better.
<table><tr><td>Crop</td><td>TESSERA [9]</td><td>Presto [34]</td><td>SPEAR-NeXT</td></tr><tr><td>Corn</td><td>0.577</td><td>0.514</td><td>0.706</td></tr><tr><td>Soybean</td><td>0.704</td><td>0.718</td><td>0.803</td></tr><tr><td>Wheat</td><td>0.778</td><td>0.754</td><td>0.831</td></tr><tr><td>Cotton</td><td>0.517</td><td>0.465</td><td>0.542</td></tr><tr><td>Mean</td><td>0.644</td><td>0.613</td><td>0.721</td></tr></table>

The prediction-head-only control benefits substantially more from kriging. Its ${ \dot { R } } ^ { 2 }$ increases by 0.190 for the April 15 forecast and by 0.232 for the August 5 forecast. The larger post-processing gains indicate that the head-only configuration produces weaker spatially coherent predictions and therefore relies more heavily on spatial interpolation to recover regional structure. Importantly, even after kriging, its performance remains below that of the complete SPEAR-NeXT model. This supports the conclusion that kriging cannot replace the information learned through selfsupervised temporal pretraining; it mainly acts as a complementary spatial refinement stage.

## 5.1.5. State-Wise In-Season Yield Forecasting

Beyond the crop-level summary metrics, we examine how the predicted yield changes across successive in-season forecast dates. Figure 5 presents the leave-one-year-out evaluation for 2023 across four major crops and their principal producing states. Each red trajectory represents the sequence of SPEAR-NeXT yield estimates obtained as additional observations become available during the season. The black marker and horizontal dashed line denote the corresponding final USDA NASS yield estimate [35].

Figure 5 shows that SPEAR-NeXT responds dynamically as additional within-season observations become available, rather than producing a fixed seasonal estimate.

![](images/e2a11f8339bbee6136d8e671dea31b2a8a51ee4d1c856f6b5cbdb8359fd7e1a0.jpg)  
(a) Classification accuracy

![](images/7a0964097c590043c65f14e0d7683db0ddc50d1a55a47cf3565ea77e19b2d813.jpg)  
(b) Macro-F1 score

Figure 2. Crop-classification performance of SPEAR and SPEAR-NeXT under different multimodal fusion, prediction-head, and adaptation settings. Each horizontal pair compares SPEAR with SPEAR-NeXT for the same configuration, while the rightmost value reports the corresponding SPEAR-NeXT minus SPEAR difference. Concat denotes feature concatenation, Cross denotes cross-attention based fusion, MLP denotes multilayer perceptron, RF denotes random forest, and PEFT denotes parameter-efficient fine-tuning. The SPEAR macro-F1 values correspond to the conservative lower-bound values reported in Table 6.

![](images/f25db24d414638266a824a5cd0088bf04295d9658837890be7393d28f4969882.jpg)

Figure 3. Contribution of the pretrained SPEAR-NeXT temporal representation to crop-yield prediction. Crop-specific $R ^ { 2 }$ values are compared between the complete SPEAR-NeXT pipeline and a prediction-head-only control at the indicated forecast dates. The complete model combines frozen SPEAR pixel states, the self-supervised SPEAR-NeXT temporal encoder, and the downstream prediction head, whereas the control excludes the pretrained temporal representation and uses only the prediction head. SPEAR-NeXT improves $R ^ { 2 }$ from 0.289 to 0.726 for corn, from 0.608 to 0.783 for soybean, from 0.724 to 0.812 for wheat, and from 0.474 to 0.542 for cotton. These correspond to relative improvements of 151.2%, 28.8%, 12.2%, and 14.3%, respectively.

![](images/595cfa70efecfd42462a1190329331d55cbf90848bd6798d08935ebd0cc726c6.jpg)  
Figure 4. Effect of post-hoc spatial kriging on corn-yield prediction using the complete SPEAR-NeXT model and the prediction head-only control. The complete model combines the frozen SPEAR encoder, the pretrained SPEAR-NeXT temporal encoder, and the downstream yield-prediction head. The prediction-head-only control excludes the pretrained temporal encoder and uses the same downstream head. Blue bars report performance before kriging, while orange bars report performance after spatial kriging. For the full model, kriging increases $R ^ { 2 }$ from 0.660 to 0.726 for the September 2 forecast and produces essentially no change for the August 5 forecas (0.727 to 0.726). In contrast, the prediction-head-only configuration improves from 0.229 to 0.419 for the April 15 forecast and from 0.173 to 0.405 for the August 5 forecast. These results indicate that kriging primarily refines the already informative SPEAR-NeXT predictions whereas it compensates more strongly for the weaker spatial structure learned by the prediction-head-only model.

For several states, the forecast moves progressively toward the final USDA NASS value as the season advances. The degree of convergence differs across crops and regions, reflecting variation in crop calendars, environmental conditions, sample availability, and the strength of the relationship between the observed satellite trajectory and final yield.

The results also reveal cases in which early-season predictions deviate substantially from the final reported yield before improving later in the season. This behaviour is expected because early observations contain limited information about late-season weather, stress, and management effects. The state-wise analysis therefore complements the aggregate crop-level $R ^ { 2 }$ results in Table 7 by showing when the model becomes informative during the season and where regional forecasting uncertainty remains.

## 5.1.6. Operational Comparison with NASA Acres

To contextualize SPEAR-NeXT against an operational Earth-observation yield forecasting system, we compare its 2024 corn-yield estimates with publicly reported forecasts from NASA Acres. The NASA Acres system combines MODIS Green Chlorophyll Vegetation Index, SERVIR Evaporative Stress Index, AgERA5 precipitation, and a Generalized Additive Model to generate county-scale corn and soybean yield forecasts across 12 Midwestern states.

As shown in Figure 6, SPEAR-NeXT follows the statelevel variation in the final USDA NASS estimates while remaining competitive with the NASA Acres operational forecasts. Across the 12-state evaluation, SPEAR-NeXT obtains an aggregate forecast of 191.8 bu/ac against the corresponding USDA estimate of 191.7 bu/ac, with a statelevel coefficient of determination of $R ^ { 2 } = 0 . 5 1 3$

This comparison should be interpreted as an external operational reference rather than as a controlled benchmark. The NASA Acres results are obtained from publicly reported forecasts and may differ from the present experiment in spatial aggregation, county weighting, input data, preprocessing, forecast issue date, and the definition of the regional USDA reference. Accordingly, the comparison demonstrates practical forecasting relevance but is not used to claim direct superiority under an identical experimental protocol.

## 5.1.7. Summary of Downstream Transfer

The downstream results reveal complementary strengths of SPEAR-NeXT. Land-cover classification demonstrates transfer to general Earth-surface mapping in both regional pipelines. SICKLE shows that the representation is particularly effective for temporally structured agricultural re-

SPEAR-NeXT Wheat Yield Forecast (LOYO-2023)  
![](images/21825687e9777c0e2eea7da21a938f9c2402f4a35f1b4eaaea26e186a3619525.jpg)

SPEAR-NeXT Soyabean Yield Forecast (LOYO-2023)  
![](images/882732e0b68c7b37c4fc966113c3745cf7b0fe91ad4610f051dbef6bb7f1ab54.jpg)

SPEAR-NeXT Cotton Yield Forecast (LOYO-2023)  
![](images/47dfc7e39b4ef198f9e7befb273e338a324954d2277938fdda926b82296fdf89.jpg)

![](images/4fb9f2c066024c307aa153bbc8ec8cdff92cceb3a204ed9790a9dd90d300ef00.jpg)  
Figure 5. State-wise in-season crop-yield forecasting under the leave-one-year-out 2023 evaluation. Results are shown for (a) wheat, (b) soybean, (c) cotton, and (d) corn. Red trajectories denote SPEAR-NeXT yield predictions generated at successive in-season forecast dates, black markers indicate the final USDA NASS reported yields, and horizontal dashed lines provide the USDA NASS reference leve for each state. The value n shown in each panel denotes the number of spatial samples used for the corresponding state. The figure illustrates both the progressive updating of yield estimates during the growing season and the variation in forecasting difficulty across crops and geographic regions.

## Corn Yield Forecast 2024: SPEAR-NeXT vs. NASA Acres Operational System

![](images/8e326bc71767b0dd3af20cd43db5347cbcbf87e7e33869de235029d5f63b8ca2.jpg)

Figure 6. Comparison of 2024 corn-yield forecasts from SPEAR-NeXT and the NASA Acres operational forecasting system across 12 Midwestern states. Blue curves represent the publicly reported NASA Acres forecasts, red curves represent SPEAR-NeXT predictions, and black markers denote the final USDA NASS yield estimates. SPEAR-NeXT produces forecasts using the frozen self-supervised representation and a lightweight downstream yield-prediction head. This comparison is provided as an operational reference rather than a controlled benchmark because the two systems may differ in input data, spatial aggregation, forecast issue dates, preprocessing, and evaluation protocols

Sourcefor the publicly reported NASA Acresforecasts: https://www.nasaacres.org/news/nasa-acres-supports-developmentof-county-scale-yield-forecasting-tools

gression, although it does not outperform Tessera on binary crop classification. Finally, the USDA-NASS experiment demonstrates consistent crop-wise gains for yield prediction. Together, these findings support the use of multihorizon latent forecasting as a representation-learning objective rather than only as an embedding-forecasting mechanism.

## 5.2. Discussion

The results support the central distinction between instantaneous Earth-state estimation and temporal state evolution. SPEAR provides compact multimodal pixel representations from optical, radar, and environmental observations, whereas SPEAR-NeXT learns how these latent states evolve under past-only, causally masked multi-horizon prediction. The forecasting improvements over persistence and seasonal-persistence baselines indicate that the temporal module captures more than local continuity or a fixed annual cycle. The prediction–target alignment and latentgeometry analyses further suggest that the forecasts remain temporally specific rather than collapsing toward an average future state.

The downstream results demonstrate that this predictive temporal objective produces reusable representations. In regional land-cover classification, SPEAR-NeXT achieves 94.81% accuracy in India and 88.78% in CONUS, exceeding the strongest corresponding baselines by 9.61 and 2.38 percentage points, respectively. On SICKLE, SPEAR-NeXT obtains the lowest errors for sowing-date, harvestdate, and yield prediction, although TESSERA retains the highest binary-classification F1 score. This pattern suggests that the main benefit of SPEAR-NeXT appears in tasks requiring phenological and temporal reasoning rather than in every static classification setting.

The crop-classification adaptation study provides additional insight into representation transfer. SPEAR-NeXT outperforms SPEAR in nine of twelve configurations, with the best result obtained using concatenation, an MLP head, and a frozen encoder, reaching 83.33% accuracy and 0.8279 macro-F1. The strong frozen-backbone result indicates that useful crop-discriminative information is already encoded during temporal pretraining. The largest gains occur with cross-attention under unfrozen and PEFT adaptation, where SPEAR-NeXT substantially improves configurations that perform poorly with SPEAR. Nevertheless, several concatenation-based unfrozen and PEFT settings show small reductions, demonstrating that additional downstream adaptation is not universally beneficial and may disturb an already structured representation under limited supervision.

The crop-yield experiments provide the clearest evidence that the gains are not attributable only to a stronger prediction head. SPEAR-NeXT achieves a mean $R ^ { 2 }$ of 0.721 across corn, soybean, wheat, and cotton, compared with 0.644 for TESSERA and 0.613 for Presto. Improvements are observed for all four crops, with the largest absolute gain occurring for corn. The full-model versus prediction-head-only ablation further shows substantial gains from the pretrained temporal representation, including $R ^ { 2 }$ improvements from 0.289 to 0.726 for corn and from 0.608 to 0.783 for soybean. These results indicate that the downstream head alone cannot recover the temporal information learned during self-supervised pretraining.

The kriging analysis leads to a similar conclusion. For the complete SPEAR-NeXT model, spatial kriging provides only modest refinement or negligible change, whereas the prediction-head-only configuration receives much larger gains from kriging. Even after spatial interpolation, the head-only model remains below the full model. Kriging therefore acts mainly as a complementary spatial correction mechanism and does not replace the temporally predictive representation learned by SPEAR-NeXT.

The objective and temporal-encoding ablations clarify the roles of the main design components. Cosine alignment preserves directional structure in the SPEAR latent space, while latent regression constrains coordinate scale and reduces magnitude drift. Multi-horizon supervision exposes the encoder to short-term continuity, intermediate transitions, and longer seasonal dynamics, making it less dependent on single-step interpolation. RoPE and calendar conditioning serve complementary roles: RoPE represents relative order and temporal displacement, while month and year embeddings provide seasonal phase and inter-annual context. Their combination is therefore better suited to monthly Earth-observation sequences than either temporal signal alone.

Several limitations remain. First, the prediction targets are frozen SPEAR embeddings and consequently inherit the information limits and biases of the underlying state encoder. Second, monthly compositing and completecoverage filtering simplify temporal learning relative to operational satellite records containing clouds, missing acquisitions, and irregular sampling. Extending the model with explicit elapsed-time conditioning and missing-modality handling is therefore important. Third, the India and CONUS models are pretrained and evaluated separately; the reported results demonstrate regional transfer within each domain but do not establish cross-region generalization. Fourth, the crop-classification adaptation experiment uses only 200 labeled samples and should be validated across multiple sampling seeds and label budgets. Finally, high latent-forecasting accuracy does not by itself guarantee physical interpretability, and future work should relate individual latent trajectories to measurable biophysical and agronomic processes.

Overall, the combined forecasting, representation, downstream-transfer, prediction-head, and kriging analyses support causal multi-horizon latent prediction as an effective self-supervised objective for compact pixel-level Earth representations. SPEAR-NeXT is most valuable not because it improves every downstream configuration uniformly, but because it encodes temporal structure that transfers effectively through frozen or lightweight adaptation to land-cover mapping, crop monitoring, phenological estimation, and in-season yield forecasting.

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. 2, 4

[2] Guillaume Astruc, Nicolas Gonthier, Clement Mallet, and´ Loic Landrieu. Anysat: One earth observation model for many resolutions, scales, and modalities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19530–19540, 2025. 1, 3, 4

[3] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024. 2, 4

[4] Derrick Bonafilia, Beth Tellman, Tyler Anderson, and Erica Issenberg. Sen1floods11: A georeferenced dataset to train and test deep learning flood algorithms for sentinel-1. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 835–845, 2020. 9, 22, 23

[5] Christopher F. Brown, Steven P. Brumby, Brookie Guzder-Williams, Tanya Birch, Samantha Brooks Hyde, Joseph Mazzariello, Wanda Czerwinski, Valerie J. Pasquarella, Robert Haertel, Simon Ilyushchenko, Kurt Schwehr, Mikaela Weisse, Fred Stolle, Craig Hanson, Oliver Guinan, Rebecca Moore, and Alexander M. Tait. Dynamic world, near real-time global 10 m land use land cover mapping. Scientific Data, 9:251, 2022. 9, 22, 23

[6] Christopher F. Brown, Michal R. Kazmierski, Valerie J. Pasquarella, William J. Rucklidge, Masha Samsikova, Chenhui Zhang, Evan Shelhamer, Estefania Lahera, Olivia Wiles, Simon Ilyushchenko, Noel Gorelick, Lihui Lydia Zhang, Sophia Alj, Emily Schechter, Sean Askay, Oliver Guinan, Rebecca Moore, Alexis Boukouvalas, and Pushmeet Kohli. Alphaearth foundations: An embedding field model for accurate and efficient global mapping from sparse label data, 2025. 1, 3, 4

[7] Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David Lobell, and Stefano Ermon. Satmae: Pre-training transformers for temporal and multi-spectral satellite imagery. In Advances in Neural Information Processing Systems, pages 197–211, 2022. 1, 3, 4

[8] Matthias Drusch, Umberto Del Bello, Sebastien Carlier,´

Olivier Colin, Veronica Fernandez, Ferran Gascon, Bianca Hoersch, Claudia Isola, Paolo Laberinti, Philippe Martimort, Alain Meygret, Fabrizio Spoto, Olivier Sy, Franco March ese, and Paolo Bargellini. Sentinel-2: Esa’s optical high resolution mission for gmes operational services. Remote Sensing ofEnvironment, 120:25–36, 2012. 8, 9, 22

[9] Zhengpeng Feng, Clement Atzberger, Sadiq Jaffer, Jovana Knezevic, Silja Sormunen, Robin Young, Madeline C. Lisaius, Markus Immitzer, Toby Jackson, James Ball, David A. Coomes, Anil Madhavapeddy, Andrew Blake, and Srinivasan Keshav. TESSERA: Temporal embeddings of surface spectra for earth representation and analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 34818–34831, 2026. 13, 14, 15

[10] Anthony Fuller, Koreen Millard, and James R. Green. CROMA: Remote sensing representations with contrastive radar-optical masked autoencoders. In Advances in Neural Information Processing Systems, pages 5506–5538, 2023. 1, 3

[11] Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, and Yansheng Li. Skysense: A multimodal remote sensing foundation model towards universal interpretation for earth observation imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 27672–27683, 2024. 3, 4

[12] Moti Rattan Gupta and Anupam Sobti. Time2agri: Temporal pretext tasks for agricultural monitoring. Proceedings of the AAAI Conference on Artificial Intelligence, 40(45):38533– 38541, 2026. 14

[13] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, 2022. 3

[14] Danfeng Hong, Bing Zhang, Xuyang Li, Yuxuan Li, Chenyu Li, Jing Yao, Naoto Yokoya, Hao Li, Pedram Ghamisi, Xiup ing Jia, Antonio Plaza, Paolo Gamba, Jon Atli Benediktsson, and Jocelyn Chanussot. Spectralgpt: Spectral remote sensing foundation model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8):5227–5244, 2024. 3

[15] Johannes Jakubik, Sujit Roy, C. E. Phillips, Paolo Fraccaro, Denys Godwin, Bianca Zadrozny, Daniela Szwarcman, Carlos Gomes, Gabby Nyirjesy, Blair Edwards, Daiki Kimura, Naomi Simumba, Linsong Chu, S. Karthik Mukkavilli, Devyani Lambhate, Kamal Das, Ranjini Bangalore, Dario Oliveira, Michal Muszynski, Kumar Ankur, Muthukumaran Ramasubramanian, Iksha Gurung, Sam Khallaghi, Hanxi Li, Michael Cecil, Maryam Ahmadi, Fatemeh Kordi, Hamed Alemohammad, Manil Maskey, Raghu Ganti, Kommy Weldemariam, and Rahul Ramachandran. Foundation models for generalist geospatial artificial intelligence, 2023. 3, 4

[16] Xuyang Li, Danfeng Hong, and Jocelyn Chanussot. S2MAE: A spatial-spectral pretraining foundation model for spectral remote sensing data. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), pages 24088–24097, 2024. 3

[17] Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multihorizon time series forecasting. International Journal of Forecasting, 37(4):1748–1764, 2021. 4

[18] Joaqu´ın Munoz-Sabater, Emanuel Dutra, Anna Agust˜ ´ı- Panareda, Clement Albergel, Gabriele Arduini, Gianpaolo´ Balsamo, Souhail Boussetta, Margarita Choulga, Shaun Harrigan, Hans Hersbach, Brecht Martens, Diego G. Miralles, Mar´ıa Piles, Nemesio J. Rodr´ıguez-Fernandez, Ervin Zsoter,´ Carlo Buontempo, and Jean-Noel Th ¨ epaut. Era5-land: A´ state-of-the-art global reanalysis dataset for land applications. Earth System Science Data, 13(9):4349–4383, 2021. 8, 9, 22

[19] Joachim Nyborg, Charlotte Pelletier, and Ira Assent. Generalized classification of satellite image time series with thermal positional encoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 1391–1401, 2022. 3, 4

[20] Charlotte Pelletier, Geoffrey I. Webb, and Franc¸ois Petitjean. Temporal convolutional neural network for the classification of satellite image time series. Remote Sensing, 11(5):523, 2019. 3

[21] Planet Labs PBC. Planetscope product specifications. Planet Documentation, 2026. Accessed: 2026-07-27. 8, 22

[22] Rajiv Ranjan, Udaiveer Singh, Anjali Aggarwal, Shashank Tamaskar, and Dharmendra Saraswat. Spear: Selfsupervised sample efficient pixel-level multi-modal spectral fusion for earth observation applications. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops, pages 1445–1454, 2026. 2, 5, 8, 10, 15, 22, 23

[23] Colorado J. Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, and Trevor Darrell. Scale-mae: A scale-aware masked autoencoder for multiscale geospatial representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4088– 4099, 2023. 3

[24] Christian Requena-Mesa, Vitus Benson, Markus Reichstein, Jakob Runge, and Joachim Denzler. Earthnet2021: A large-scale dataset and challenge for earth surface forecasting as a guided video prediction task. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 1132–1142, 2021. 2, 4

[25] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234–241, Cham, 2015. Springer. 14

[26] Marc Rußwurm and Marco Korner. Multi-temporal land¨ cover classification with sequential recurrent encoders. IS-PRS International Journal of Geo-Information, 7(4):129, 2018. 3

[27] Vivien Sainte Fare Garnot, Loic Landrieu, Sebastien Giordano, and Nesrine Chehata. Satellite image time series

classification with pixel-set encoders and temporal selfattention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12325–12334, 2020. 3, 4

[28] Depanshu Sani, Sandeep Mahato, Sourabh Saini, Harsh Ku mar Agarwal, Charu Chandra Devshali, Saket Anand, Gaurav Arora, and Thiagarajan Jayaraman. Sickle: A multi sensor satellite imagery dataset annotated with multiple key cropping parameters. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5995–6004, 2024. 9, 13, 14, 22, 23

[29] Wilfrid Schroeder, Patricia Oliva, Louis Giglio, and Ivan A. Csiszar. The new viirs 375 m active fire detection data product: Algorithm description and initial assessment. Remote Sensing ofEnvironment, 143:85–96, 2014. 9, 22, 23

[30] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3, 2025. 3

[31] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 2, 4

[32] Ramon Torres, Paul Snoeij, Dirk Geudtner, David Bibby, Malcolm Davidson, Evert Attema, Pierre Potin, Bjorn Rom-¨ men, Nicolas Floury, Mike Brown, Ignacio Navas Traver, Philippe Deghaye, Berthyl Duesmann, Betlem Rosich, Nuno Miranda, Claudio Bruno, Marco L’Abbate, Renato Croci, Alessandro Pietropaolo, Markus Huchler, and Friedhelm Rostan. Gmes sentinel-1 mission. Remote Sensing of Environment, 120:9–24, 2012. 8, 9, 22

[33] Gabriel Tseng, Ivan Zvonkov, Catherine Nakalembe, and Hannah Kerner. Cropharvest: A global dataset for crop-type classification. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021. 9, 23

[34] Gabriel Tseng, Ruben Cartuyvels, Ivan Zvonkov, Mirali Purohit, David Rolnick, and Hannah Kerner. Lightweight, pre-trained transformers for remote sensing timeseries, 2023. 1, 4, 13, 14, 15

[35] United States Department of Agriculture, National Agricultural Statistics Service. Quick stats database. USDA National Agricultural Statistics Service, 2026. Accessed: 2026- 07-27. 9, 13, 15, 23

[36] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2018. 4

[37] Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J. Stewart, Joelle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand¨ Le Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural plasticity-inspired multimodal foundation model for earth observation, 2024. 3

## Appendix

## A. Dataset Information

This appendix provides additional information about the datasets used to pretrain the spectral and temporal components of SPEAR-NeXT and to evaluate their downstream transfer. The experimental pipeline uses two geographically distinct corpora. The India corpus supports the previously published SPEAR spectral and multimodal pretraining stage [22], whereas the contiguous United States (CONUS) corpus is used to construct the pixel-level temporal sequences required for SPEAR-NeXT pretraining. Both corpora are sampled across heterogeneous land-cover conditions using the Dynamic World taxonomy [5].

Figure 7 summarizes the two geographically distinct pretraining corpora and their associated downstream evaluation tasks. The India corpus is shown in Figure 7a, while the CONUS corpus is shown in Figure 7b.

## A.1. India Corpus

Pretraining data. The India corpus contains approximately 2.7 million pixel-level samples distributed across diverse agricultural, forested, urban, barren, wetland, and water-covered environments. This corpus was originally developed for SPEAR and is described in detail in the corresponding publication [22]. Sampling was stratified using Dynamic World land-cover classes to reduce domination by highly prevalent surface types and to increase coverage of less frequent classes [5].

The corpus combines the following observation sources:

• Sentinel-2 optical imagery. Ten multispectral bands spanning visible, red-edge, near-infrared, and shortwaveinfrared wavelengths are used. Sentinel-2 provides the principal multispectral input to the wavelength-aware SPEAR encoder [8].

• PlanetScope imagery. Eight-band PlanetScope Super-Dove observations introduce an additional optical sensor and a finer ground-sampling scale during multimodal SPEAR pretraining [21].

• Sentinel-1 radar imagery. Dual-polarization VV and VH synthetic-aperture-radar backscatter is represented at an approximate spatial resolution of 10 m [32].

• Climate and environmental variables. Daytime and nighttime land-surface temperature, precipitation, and elevation-related variables are matched to the sampled pixel locations. ERA5-Land provides the associated reanalysis information [18]. The complete feature composition and preprocessing follow the published SPEAR pipeline [22].

Downstream evaluation. Representations learned from the India corpus are evaluated on several complementary EO tasks:

• SICKLE crop classification. SICKLE contains multisensor satellite time series from Landsat-8, Sentinel-1, and Sentinel-2, together with crop-type, phenological, and yield-related annotations collected over agricultural plots in the Cauvery Delta region of India [28]. The croptype subset is used for agricultural classification.

• Sen1Floods11 flood detection. Sen1Floods11 contains georeferenced Sentinel-1 observations and corresponding surface-water labels spanning multiple flood events across several geographic regions [4]. It is used to evaluate transfer to flood-related surface-water discrimination.

• VIIRS fire-related classification. The VIIRS 375 m active-fire product provides satellite-derived fire detections with improved spatial response relative to coarser fire products [29]. VIIRS detections are used as a reference source for the fire-related downstream evaluation.

• Dynamic World land-cover classification. Dynamic World provides near-real-time, 10 m land-cover predictions derived from Sentinel-2 imagery using a nine-class taxonomy [5]. These labels are used both for stratified sampling and general land-cover evaluation.

The India downstream suite spans agriculture, flooding, fire, and general land-cover mapping. It therefore tests whether the SPEAR representation transfers beyond a single semantic domain.

## A.2. CONUS Corpus

Temporal pretraining data. The CONUS corpus contains approximately 4.5 million pixel-level samples distributed across the contiguous United States. Its geographic extent introduces substantial variation in vegetation, climate, agricultural systems, management practices, and seasonal timing. Samples are stratified using Dynamic World land-cover classes [5].

For SPEAR-NeXT, retained Sentinel-2 observations are chronologically organized into pixel-level sequences of T = 61 timesteps, as described in Section 4.1. Each Sentinel-2 observation is transformed into a frozen 32- dimensional SPEAR embedding before temporal pretraining.

The broader CONUS corpus contains:

• Sentinel-2 optical imagery. Ten multispectral bands are processed by the frozen SPEAR spectral encoder to produce a compact state at each timestep [8].

• Sentinel-1 radar imagery. VV and VH radar backscatter observations are retained in the broader multimodal dataset construction and selected downstream evaluations [32].

• Climate and environmental variables. Matched temperature, precipitation, and elevation-related information is associated with the sampled locations. ERA5-Land provides the corresponding reanalysis fields [18].

The temporal model presented in the main manuscript is pretrained on the sequence of frozen Sentinel-2 SPEAR embeddings. Radar and climate variables are part of the broader dataset and task-specific analyses, but are not implicitly treated as temporal-encoder inputs unless explicitly stated in the corresponding experiment.

![](images/3726285a2f3b4630b875dad576107a33a819ec77bd4a035a94bf8c0aedc26b1b.jpg)  
(a) India pretraining corpus for SPEAR, containing approximately 2.7 million pixel-level samples from Sentinel-2, PlanetScope, Sentinel-1, and matched climate–environmental variables.

![](images/76fe33a09cf04d5e0b4adb2b14d5a573c91c5c0d70e5966b17cf077e69800657.jpg)  
(b) CONUS temporal-pretraining corpus for SPEAR-NeXT, con taining approximately 4.5 million pixel-level samples from Sentinel-2, Sentinel-1, and matched climate–environmental variables.

Figure 7. Geographic pretraining corpora and downstream evaluation datasets used in the SPEAR-NeXT pipeline. The India corpus supports SPEAR spectral and multimodal pretraining [22], while the CONUS corpus provides temporally ordered pixel sequences for SPEAR-NeXT pretraining. The associated downstream datasets include SICKLE [28], Sen1Floods11 [4], VIIRS active-fire observations [29], USDA NASS Quick Stats [35], CropHarvest [33], and Dynamic World [5].  
Table 8. Summary of the pretraining corpora used in the SPEAR-NeXT pipeline.
<table><tr><td>Corpus</td><td>Approximate size</td><td>Geographic coverage</td><td>Input sources</td><td>Primary role</td></tr><tr><td>India</td><td>2.7 million pixel-level samples</td><td>India</td><td>Sentinel-2, PlanetScope, Sentinel-1, climate/environmental variables</td><td>SPEAR spectral and multimodal pretraining</td></tr><tr><td>CONUS</td><td>4.5 million pixel-level samples</td><td>Contiguous United States</td><td>Sentinel-2, Sentinel-1, climate/environmental variables</td><td>SPEAR-NeXT temporal pretraining</td></tr></table>

Downstream evaluation. The CONUS experiments emphasize temporally demanding agricultural transfer while retaining general environmental evaluation:

• USDA NASS crop-yield prediction. County-level cropyield records are obtained from the United States Department of Agriculture National Agricultural Statistics Service Quick Stats database [35]. The database provides agricultural statistics indexed by commodity, geographic location, and year. These records are aligned with countylevel satellite representations for yield prediction.

• Sen1Floods11 flood detection. Sen1Floods11 is used to evaluate transfer to flood-related surface-water discrimination [4].

• CropHarvest crop/non-crop classification. CropHarvest is a global, analysis-ready satellite dataset containing more than 90,000 geographically diverse agricultural samples [33]. The binary crop/non-crop task is used to evaluate agricultural transfer under geographic variation.

• Dynamic World land-cover classification. Dynamic

World labels are used for general nine-class land-cover evaluation [5].

Relative to the India evaluation, the CONUS suite places greater emphasis on temporal agricultural transfer through county-level yield prediction and geographically distributed crop-presence classification.

## A.3. Role of the Datasets in the Experimental Design

Table 9 distinguishes datasets used for self-supervised pretraining from those used only for downstream evaluation. No downstream labels are used during SPEAR-NeXT temporal pretraining.

The combination of general mapping and agriculturefocused downstream datasets is intended to determine whether SPEAR-NeXT learns reusable temporal representations rather than features specialized for one region or application. The India and CONUS corpora also provide an initial basis for assessing transfer across contrasting landcover distributions, agro-climatic conditions, and seasonal regimes.

Table 9. Role of each data source in the SPEAR-NeXT experimental pipeline.
<table><tr><td>Geographic setting</td><td>Dataset or source</td><td>Principal information</td><td>Role</td></tr><tr><td>India India</td><td>India multimodal corpus SICKLE</td><td>Optical, radar, and climate/environmental pixels Multisensor agricultural time series and crop labels</td><td>SPEAR self-supervised pretraining Crop-type classification</td></tr><tr><td>Global</td><td>Sen1Floods11</td><td>Sentinel-1 imagery and flood/surface-water labels</td><td>Flood detection</td></tr><tr><td>Global</td><td>VIIRS active-fire product</td><td>375 m active-fire detections</td><td>Fire-related classification</td></tr><tr><td>India</td><td>Dynamic World</td><td>Nine-class Sentinel-2-derived land-cover labels</td><td>Stratification and land-cover evaluation</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>CONUS</td><td>CONUS temporal corpus</td><td>Pixel-level Sentinel-2 embedding sequences</td><td>SPEAR-NeXT self-supervised temporal pretraining</td></tr><tr><td>CONUS</td><td>USDA NASS Quick Stats</td><td>County- and year-indexed crop-yield statistics</td><td>Crop-yield prediction</td></tr><tr><td>Global</td><td>CropHarvest</td><td>Geographically diverse agricultural labels</td><td>Binary crop/non-crop classification</td></tr><tr><td>CONUS / Global</td><td>Dynamic World</td><td>Nine-class Sentinel-2-derived land-cover labels</td><td>Stratification and land-cover evaluation</td></tr></table>