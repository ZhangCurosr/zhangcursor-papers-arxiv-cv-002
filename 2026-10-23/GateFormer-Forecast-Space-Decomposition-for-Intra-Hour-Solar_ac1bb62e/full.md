# GateFormer: Forecast-Space Decomposition for Intra-Hour Solar Irradiance Prediction from Sky Images

David Hamlyn <sup>a,</sup>\*, Sunny Chaudhary <sup>a</sup>, Jonathon Hare <sup>a</sup>, Tasmiat Rahman <sup>a</sup>

<sup>a</sup> School of Electronics and Computer Science, University of Southampton, University Road, Southampton, SO17 1BJ, United Kingdom

## Abstract

Sub-hourly dispatch, balancing, and storage control increasingly depend on forecasts that anticipate rapid variability in solar generation. However, accurate sky-image-based irradiance forecasting requires a combination of visual cloud dynamics with recent irradiance state while adapting to different short-term irradiance-change regimes, a challenge for many existing approaches. To address this, we introduce GateFormer, a GHI-conditioned dual-decoder Vision Transformer (ViT) for intra-hour global horizontal irradiance (GHI) forecasting through forecast-space decomposition. GateFormer embeds observed GHI into the visual token stream, allowing interaction with sky-image patch features through self-attention. From this shared representation, a Trajectory Head predicts future GHI sequence, while a Delta Head predicts finalhorizon change from the latest observed GHI. The two final-horizon forecasts are adaptively weighted in prediction space, with magnitude- and direction-supervision targeting the remaining error. On the NREL-TSI benchmark, GateFormer attains 125.01 W/m² RMSE at +30 min, improving over a state-of-the-art transformer baseline by 9.6% under paired-timestamp evaluation. Across NREL-TSI, Chilbolton, and SIRTA, GateFormer achieves the lowest final-horizon RMSE among the evaluated baselines, with forecast skill of 22.97–28.86% relative to smart persistence. Component ablations show adaptive weighting outperforms both individual decoders and fixed averaging, distinguishing GateFormer from conventional forecast combination, while the supervised correction further reduces RMSE by 1.68 W/m² and MAE by 4.86 W/m². Ramp-regime diagnostics show larger gains during operationally important positive ramps, reducing RMSE by 26.27 W/m². These results identify forecast-space decomposition as an alternative forecasting formulation, with potential relevance to other multimodal short-horizon forecasting tasks and deployment using only sky images and measured GHI.

Keywords: Computer vision, Multimodal learning, Solar irradiance forecasting, Solar ramps, Vision transformer

## 1. Introduction

As photovoltaic (PV) penetration increases, power-system operation becomes more sensitive to rapid changes in solar generation. This sensitivity is becoming more operationally important as electricity systems move toward higher-granularity operation, including 5–15 min dispatch, sub-hourly balancing, and increasingly dynamic intraday markets [1],[2],[3],[4]. Sub-hourly irradiance forecasts are therefore important for grid balancing, storage scheduling, ramp-rate control, and short-term energy management [1],[2],[3],[4]. At these horizons, forecast errors often arise from local cloud advection, transient solar obstruction, and cloudedge enhancement, which can rapidly change ground-level irradiance. Conventional numerical weather prediction (NWP) often falls short at this resolution due to spatial and temporal scales below those of typical operational weather-model outputs [5]. Ground-based sky images provide direct observations of the local cloud field, including cloud structure, motion cues, solar occlusion, and cloud-edge patterns, making them a valuable input for intra-hour global horizontal irradiance (GHI) forecasting [4],[6],[7],[8]. Accurate intra-hour forecasting therefore requires visual cloud information to be interpreted together with the current irradiance state.

Recent deep learning (DL) approaches have improved intra-hour irradiance forecasting by learning predictive representations directly from sky-image sequences [9], [10]. These advances include future visualstate prediction, transformer-based sequence modelling, and multimodal fusion of sky imagery with irradiance history and clear-sky information [9],[10],[11],[12]. However, strong performance in this setting still depends on how the model represents short-term irradiance change, particularly when cloud motion produces rapid departures from persistence-like behaviour. This motivates a forecasting formulation that does not rely on a single output pathway but instead forms distinct predictions of future irradiance state and learns how to effectively combine them.

To address this limitation, we introduce GateFormer, a dual-decoder Vision Transformer (ViT) that implements adaptive gating in the prediction space for intra-hour irradiance forecasting. Rather than using a single decoder to map encoded sky-image information directly to the final forecast, GateFormer first encodes the joint visual and irradiance state by embedding observed GHI into the visual token stream, allowing irradiance-state information to interact with sky-image patch features through self-attention. From this shared representation, a Trajectory decoder predicts GHI at each future step up to and including the final horizon, meaning the furthest point in time forecast by the model, while a Delta decoder predicts the change from the latest observed GHI directly to that same final horizon. The forecast-space decomposition means forming alternative forecasts with different prediction targets from a shared representation and exposing their values and disagreement to the gate before the final prediction is formed. The two final-horizon predictions allow a shared architecture to model complementary views of the prediction space which are then combined using a learned adaptive blend to form a Base forecast, and a residual head predicts the remaining post-blend correction using magnitude and direction supervision. Unlike conventional ensembles that combine independently trained models, or Mixture-of-experts (MoE) architectures that route inputs among alternative expert pathways, GateFormer forms both candidate forecasts from one shared representation using distinct prediction targets and applies gating only after explicit irradiance predictions have been produced. This structure allows the model to represent both gradual trajectory evolution and direct final-horizon displacement, adaptively blend them in prediction space and improve post-blend error.

Contributions and motivations. Reliable intra-hour irradiance forecasting is critical for ramp-rate control, storage scheduling, and grid stability, yet multimodal methods for forming and combining complementary predictions remain comparatively underexplored. Building on state-of-the-art ViTs, we introduce GateFormer, a novel forecast-space decomposition architecture. (1) A dual-decoder framework predicts absolute multi-horizon GHI trajectories and direct final-horizon GHI displacement from a shared GHIconditioned visual representation, allowing the model to capture both stable trajectory behaviour and larger ramp displacements. (2) A prediction-space adaptive blending that learns a per sample combination of the two forecasts, rather than fusing latent features, benefiting from the decoders’ complementary representations. (3) A separately supervised post-blend residual that predicts the magnitude and direction of the error remaining after adaptive gating. Experiments across three sky-imaging datasets and component-level ablations demonstrate the learned blend improves over both individual decoders and fixed averaging, while residual correction provides a further overall reduction in error especially in positive ramps.

## 2. Related Work

Short-term solar forecasting has increasingly used ground-based sky images as they provide local information on cloud position, motion, and solar obstruction that is unavailable from scalar irradiance histories alone [4]. Classical methods convert these observations into short-term forecasts through cloud segmentation, optical flow, and tracking [6],[7]. Although physically interpretable, their performance depends on preprocessing quality and assumptions concerning cloud motion, height, and transmittance, motivating DL approaches that learn predictive visual representations directly from imagery [4], [5], [6], [7].

Recent studies have applied explainable machine learning to GHI estimation [13], wavelet-assisted deep learning to GHI modelling [14], and feature-engineered machine learning to rainfall prediction [15]. Solar system modelling has also been explored through Arduino-based PV water-pumping simulations [16].

Self-attention enables long-range relationships to be modelled across input sequences [17], while ViTs extend this mechanism to spatial relationships between image patches [18]. Image-only ViTs have shown that sky-image representations can support accurate irradiance estimation [19]. The Solar Irradiance Anticipative Transformer (SIAT) extends this capability to temporal forecasting from sky-image sequences [10]. Paletta et al. jointly predict future visual states and irradiance to anticipate cloud-induced perturbations [9]. However, visual information alone does not explicitly encode current irradiance state or its recent evolution. Piantadosi et al. demonstrate the value of non-image inputs for Transformer-based PV forecasting [20]. Liu et al. further demonstrate this by fusing sky-image features with historical and clear-sky GHI through cross-modal attention, reporting an nRMSE of 4.28% at the 10-min horizon on NREL SRRL [11]. Zhang et al. compare early, late, and gated image–numerical fusion, demonstrating trade-offs between overall error and ramp-event performance. Their Transformer-based early fusion improves ramp-event balanced accuracy by 9.43% and 3.91% at the 2- and 6-min horizons, respectively [12]. These approaches improve representation and modality interaction but still optimise a single forecast mapping rather than forming complementary predictions with different forecasting objectives.

Architectures have begun to separate modelling roles through specialised pathways, including sequential and parallel frameworks to extract distinct visual and irradiance information [21]. Specialisation also appears in the handling of heterogeneous inputs, with Chen et al. combining spatiotemporal satellite and NWP inputs within an interpretable Temporal Fusion Transformer for intra-day irradiance forecasting [22]. At the processing-stage level, neural feature extraction has been separated from probabilistic prediction [23]. More explicit branching appears in PV-MM-Diffusion, which couples image and PV pathways to jointly generate future sky states and probabilistic PV trajectories [24], while Xiong et al. generate multiple future sky scenarios, forecast power for each, and aggregate the resulting distributions [25]. These studies demonstrate the value of architectural specialisation, but their components separate modalities, processing stages, or deterministic and probabilistic dynamics; they do not generate alternative irradiance forecasts from the same latent representation and combine them in prediction space. This distinction is technically important because retaining both candidates as internal forecasts exposes their values and disagreement to the gate before the final prediction is formed, unlike latent fusion that produces one output or conventional combination applied to completed forecasts from separate systems.

These specialised pathways can be combined adaptively, with dual gates improving interactions among spatial and temporal features [26]. Forecasts from wavelet-decomposed PV subseries have also been weighted dynamically across multiple horizons [27]. MoE models formalise input-dependent weighting of specialised predictors [28], while forecast-combination theory establishes that combining imperfect forecasts outperform individual predictors [29]. Nouri et al. apply this principle to solar forecasting blending separately generated ASI, satellite, and persistence forecasts, reducing RMSE and MAE by 13.6% and 17.0%, respectively, relative to standalone satellite forecast [30]. This demonstrates the value of forecast-level complementarity, although candidate forecasts originate from separate forecasting systems.

Collectively, these studies establish the value of multimodal context, specialised pathways, gating, and forecast-level combination. What remains unresolved is whether a shared visual–irradiance representation can support multiple forecast objectives, allow their final-horizon predictions to be weighted conditionally on the input, and explicitly correct the error remaining after combination. GateFormer addresses this gap by coupling a shared GHI-conditioned visual backbone with a Trajectory Head for multi-horizon evolution and a Delta Head for final-horizon displacement, followed by adaptive forecast-space blending and supervised residual correction. Unlike approaches requiring additional measured meteorological variables or satellite products, GateFormer uses sky-image sequences and observed GHI, supplemented only by deterministically derived temporal and clear-sky context.

## 3. Method

We introduce GateFormer by first outlining the overall architecture in 3.1, then describing the dual forecasting heads in 3.2, the adaptive blend-residual mechanism in 3.3, the loss functions in 3.4, and the datasets and evaluation protocol in the following sections.

## 3.1. Overall Architecture

Figure 1 shows the GateFormer pipeline, $\mathcal { Z } _ { \mathrm { i m g } }$ denotes the encoded sky-image sequence representation, $\mathcal { Z } _ { \mathrm { p h y s } }$ denotes the temporal and clear-sky context, w is the adaptive blend weight, and $\widehat { \Gamma } _ { \mathrm { t + H } }$ is the predicted post-blend residual correction. Let t denote the latest observed time, $h \in \{ 1 , \ldots , H \}$ the forecast step, and H the final forecast step. In this study, H=3. Each input sample consists of four past sky images, their corresponding GHI anchor values, and temporal/clear-sky features. A DeiT-III encoder maps each image into patch tokens [31]. For each input frame the observed GHI value is projected into the transformer embedding space and inserted as a learned GHI token, allowing irradiance state information to interact with image patch representations through self-attention.

Frame-level time encodings are added to the image-token representations, while clear-sky and recent-GHI-change features condition the decoders and adaptive gate/residual module. The resulting patch-level memory is pooled into frame-level representations, and temporal-difference tokens are formed from consecutive frame summaries and the latest-minus-first frame summary before being passed to two lightweight decoders: a multi-step Trajectory decoder and a Delta displacement decoder. The decoder memory contains both per-frame visual state information and short-term visual-change information across the four-image sequence. Their final-horizon predictions are adaptively blended, and a residual head predicts the remaining post-blend correction.

![](images/3e3c55176fe844bf92f333a2dfa84b1e040039b65b6ddd4c63b665c32b1cfe58.jpg)  
Figure 1 – Overview of the GateFormer architecture. A GHI-token-conditioned ViT encodes the sky-image sequence and irradiance history. A Trajectory Head and Delta Head produce complementary forecasts, which are combined through an adaptive blend and refined by a residual correction.

Table 1 – Architecture-summary table
<table><tr><td>Component</td><td>Input</td><td>Configuration</td><td>Output</td><td>Param (M⁺)</td></tr><tr><td>Shared Encoder</td><td>Four 224 by 224 images corresponding GHI</td><td>DeiT-III Medium: 16 by 16 patches, d=512, 12 Blocks, 8 Heads, MLP=2048</td><td>M ∈ RB×8×512</td><td>38,347,804</td></tr><tr><td>Trajectory Head</td><td>Decoder memory M, learned query qT</td><td>Two-layer Transformer decoder: d=512, 8 Heads, MLP=2048, dropout 0.1</td><td> $[ \widehat { \mathbf { y } } _ { \mathrm { t + 1 } } ^ { \mathrm { T } } , \widehat { \mathbf { y } } _ { \mathrm { t + 2 } } ^ { \mathrm { T } } , \widehat { \mathbf { y } } _ { \mathrm { t + H } } ^ { \mathrm { T } } ]$ </td><td>8,410,627</td></tr><tr><td>Delta Head</td><td>Decoder memory M, learned query qD</td><td>Two-layer Transformer decoder (dim=Trajectory Head)</td><td> $\widehat { \mathbf { y } } _ { t + H } ^ { \Delta }$ </td><td>8,409,601</td></tr><tr><td>Multimodal Context</td><td>18 numeric features, three visual summaries</td><td>Numeric 18 → 256, visual 1536→256, fused 512→256; LayerNorm, SiLU, dropout 0.05</td><td> $c \in \mathbb { R } ^ { 8 \times 2 5 6 }$ </td><td>533,796</td></tr><tr><td>Adaptive Gate</td><td>Context vector c</td><td>Linear 256→1 followed by sigmoid</td><td>ω ∈ [0,1]</td><td>257</td></tr><tr><td>Residual Head</td><td>Context vector c</td><td>Separate Linear 256→1 unbounded</td><td> $\widehat { \Gamma } _ { t + H }$ </td><td>257</td></tr><tr><td>GateFormer</td><td>All components above</td><td>Shared encoder, two forecast heads, adaptive gate and residual correction</td><td> $\widehat { \boldsymbol { y } } _ { t + H }$ </td><td>55,702,342</td></tr></table>

For input frame k, the encoder output is pooled into a frame-level representation $z _ { k } .$ , and the four frame representations are combined with consecutive and full-window temporal differences to form the shared decoder memory M:

$$
z _ { k } = \mathrm { P o o l } \big [ E _ { \theta } \big ( [ P _ { k } ; e _ { G } ( g _ { k } ) ] + e _ { t } ( k ) \big ) \big ]\tag{1}
$$

$$
\mathsf { M } = [ z _ { 1 } , z _ { 2 } , z _ { 3 } , z _ { 4 } , z _ { 2 } - z _ { 1 } , z _ { 3 } - z _ { 2 } , z _ { 4 } - z _ { 3 } , z _ { 4 } - z _ { 1 } ]\tag{2}
$$

Here, $P _ { k }$ denotes the patch tokens, $e _ { G } ( g _ { k } )$ the GHI token, $e _ { t } ( k )$ the temporal encoding, $E _ { \theta }$ the encoder, and Pool token averaging. The resulting memory M contains four frame-state representations, three consecutive-change representations, and one full-window change representation. This shared memory is supplied to both forecasting heads.

## 3.2. Dual Heads

The shared visual-temporal encoder feeds two complementary forecasting heads. The first is the Trajectory Head, which predicts the full future irradiance sequence across all forecast steps. This head is designed to capture the temporal evolution of GHI over the forecast window, including gradual trends and horizon-dependent changes in irradiance. Here, gradual trends refer to progressive variation in GHI across successive forecast steps, while horizon-dependent changes mean that each lead time has its own predicted GHI value. The second is the Delta Head, which focuses directly on the final forecast step. Rather than predicting the entire future path, it predicts the final horizon change relative to the latest observed GHI. Here, the final-horizon change is the signed difference between GHI at the final forecast step and the latest observed GHI, and this predicted difference is added to the latest observation to produce the Delta forecast. This gives the model an alternative final-horizon decoder that directly models displacement from the latest irradiance state, rather than deriving the final forecast from the multi-horizon Trajectory output. Together, these two heads provide complementary views of the forecasting problem. The adaptive blend is therefore the central decision mechanism: it determines, for each sample, whether the final forecast will rely more on the trajectoryderived forecast or the direct displacement forecast.

$$
\mathsf { u } ^ { T } = \mathsf { D e c } _ { T } ( q _ { T } , \mathsf { M } )\tag{3}
$$

$$
\mathsf { u } ^ { a } = \mathrm { D e c } _ { \varDelta } ( q _ { \varDelta } , \mathsf { M } )\tag{4}
$$

Output projections map $\mathrm { u ^ { T } }$ to the multi-horizon Trajectory forecast and $\mathtt { u } ^ { \Delta }$ to the predicted final-horizon change $\hat { y } _ { \mathrm { t + H } } ^ { \Delta }$ . Let $\Delta \mathrm { y } _ { \mathrm { t + H } } = \mathrm { y } _ { \mathrm { t + H } } - \mathrm { y } _ { \mathrm { t } }$ denote the measured final horizon change and $\Delta \hat { \tt y } _ { \mathrm { t + H } }$ its	prediction. The corresponding absolute GHI forecast is $\hat { \mathbf { y } } _ { \mathrm { t + H } } ^ { \Delta } = \mathbf { y } _ { \mathrm { t } } + \Delta \hat { \mathbf { y } } _ { \mathrm { t + H } }$

## 3.3. Soft Gating Mechanism

In GateFormer, the gating module consists of an adaptive prediction-space blend followed by a residual correction stage. The blend estimates the relative weighting of the Trajectory and Delta forecasts, while the residual head predicts the remaining post-blend error. The Trajectory Head and Delta Head produce two complementary predictions for forecast horizon H. Let $\widehat { \mathbf { y } } _ { \mathrm { t + H } } ^ { \mathrm { T } }$ denote the Trajectory Head prediction and $\hat { y } _ { \mathrm { t + H } } ^ { \Delta }$ denote the Delta Head prediction. The adaptive blend weight is predicted from a context vector containing the encoded sky-image sequence representation, recent GHI-state information, the Trajectory Head forecast, the Delta-derived final-horizon forecast, and the disagreement between the two decoder outputs. Here, $\mathcal { Z } _ { \mathrm { i m g } }$ contains the three visual summaries, while $\mathcal { Z } _ { \mathrm { p h y s } }$ contains the numerical, clear-sky, and decoder-derived forecast features. These are fused as:

$$
c = M L P _ { f u s e } ( [ \mathcal { Z } _ { \mathrm { i m g } } , \mathcal { Z } _ { p h y s } ] )\tag{5}
$$

The gate therefore does not operate directly on raw images or raw irradiance values; instead, it uses the learned visual-temporal representation together with the two-candidate final-horizon predictions to determine their relative contribution to the Base forecast. Rather than selecting one decoder output with a hard decision, GateFormer learns a sample-specific blend weight $\mathfrak { o } \in ( 0 , 1 )$ which forms the Base forecast:

$$
b _ { t + H } = \omega \hat { y } _ { t + H } ^ { T } + \left( 1 - \omega \right) \hat { y } _ { t + H } ^ { A } , \qquad \omega \in [ 0 , 1 ]\tag{6}
$$

The blend weight � determines the relative contribution of the trajectory-derived and delta-derived predictions for each sample. A value close to 1 places greater weight on the Trajectory Head, while a value close to 0 places greater weight on the Delta Head. This allows the model to combine the two decoder outputs in prediction space after both forecasts have already been formed. After the base forecast is produced, a residual head predicts the remaining post-blend error. Let $\widehat { \Gamma } _ { \mathrm { t + H } }$ denote the predicted residual correction. The final corrected forecast is then

$$
\hat { y } _ { t + H } = b _ { t + H } + \hat { r } _ { t + H }\tag{7}
$$

This design separates forecast formation into two stages. The adaptive blend first combines the two decoder predictions into the Base forecast, and the residual then corrects the remaining error after blending. The blend is not directly supervised to select the locally better decoder. Instead, the blend weight (w) and residual correction are learned through the final prediction objective and the residual-supervision terms described below.

## 3.4. Loss Functions

Training follows two sequential stages with separate objectives. First, the shared encoder, Trajectory Head, and Delta Head are trained using the decoder-pretraining objective. These components are then frozen, and the adaptive blend-residual module is trained using the gate objective. The Trajectory loss and the change- and event-weighting terms are defined as follows:

$$
\mathcal { L } _ { \mathrm { T r a j } } = \sqrt { \mathbb { E } _ { h } \left[ \left( y _ { t + h } - \hat { y } ^ { T } _ { t + h } \right) ^ { 2 } \right] + \varepsilon }\tag{8}
$$

$$
\mathcal { W } ^ { c h g } = 1 + \lambda _ { c h g } \mathrm { c l i p } \left( \frac { | \varDelta y _ { t + H } | } { s _ { c h g } } , 0 , c _ { c h g } \right)\tag{9}
$$

$$
\mathcal { W } ^ { e v e n t } = ( | \varDelta y _ { t + H } | \geq \tau _ { e v e n t } ) \mathrm { c l i p } \left( \frac { | \varDelta y _ { t + H } | } { s _ { e v e n t } } , 0 , c _ { e v e n t } \right)\tag{10}
$$

Before evaluating the losses, $\mathcal { W } ^ { \mathrm { c h g } }$ and �<sup>9:9;<</sup> are divided by their respective mini-batch means. Their normalised forms are denoted by $\widetilde { \mathcal { W } } ^ { \mathrm { c h g } }$ and $\widetilde { \mathcal { W } } ^ { \mathrm { e v e n t } }$ . Here, $\ell _ { \mathrm { e v e n t } }$ <sub><</sub> denotes the positive underprediction error for large upward changes and the positive overprediction error for large downward changes. If a mini batch contains no qualifying event, $\mathcal { L } _ { \mathrm { e v e n t } }$ is set to zero.

$$
\mathcal { L } _ { \Delta , b a s e } = \sqrt { \mathbb { E } \left[ \widetilde { \mathcal { W } } ^ { c h g } ( \varDelta y _ { t + H } - \Delta \hat { \mathbf { y } } _ { \mathrm { t + H } } ) ^ { 2 } \right] + \varepsilon }\tag{11}
$$

$$
\mathcal { L } _ { e v e n t } = \sqrt { \mathbb { E } \big [ \widetilde { \mathcal { W } } ^ { e v e n t } \ell _ { e v e n t } ^ { 2 } \big ] + \varepsilon }\tag{12}
$$

$$
\mathcal { L } _ { \Delta } = ~ \mathcal { L } _ { \Delta , b a s e } + \lambda _ { e v e n t } \mathcal { L } _ { e v e n t }\tag{13}
$$

$$
\mathcal { L } _ { H e a d s } = \mathcal { L } _ { \mathrm { T r a j } } { + } \mathcal { L } _ { \Delta }\tag{14}
$$

After decoder pretraining, the encoder and both forecasting heads are frozen. After the Base forecast $\mathsf { b } _ { \mathrm { t + H } }$ is formed, the true post-blend residual is defined as $\mathbf { r _ { t + \mathrm { H } } } = \left( \mathbf { y _ { t + \mathrm { H } } } - \mathbf { b _ { t + \mathrm { H } } } \right)$ . The blend-residual module is then trained using a final prediction loss, a residual-error loss, and a residual-direction loss:

$$
\mathcal { L } _ { \mathrm { f i t } } = \sqrt { \mathbb { E } [ ( y _ { t + H } - \hat { y } _ { t + H } ) ^ { 2 } ] + \varepsilon }\tag{15}
$$

$$
\mathcal { L } _ { \mathrm { r e s } } = \sqrt { \mathbb { E } [ ( r _ { t + H } - \hat { r } _ { t + H } ) ^ { 2 } ] + \varepsilon }\tag{16}
$$

$$
\mathcal { L } _ { \mathrm { d i r } } = \mathbb { E } [ \mathrm { s o f t p l u s } ( - \mathrm { s g n } ( r _ { t + H } ) \hat { r } _ { t + H } ) ]\tag{17}
$$

$$
{ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { f i t } } + { \mathcal { L } } _ { \mathrm { r e s } } + { \mathcal { L } } _ { \mathrm { d i r } }\tag{18}
$$

The fit loss trains the final corrected forecast, the residual loss encourages the residual head to predict the remaining post-blend error using a stop-gradient residual target, and the sign loss penalizes corrections that move in the wrong direction. The total objective therefore trains GateFormer to form an adaptive Base forecast and then apply a directionally consistent residual correction. The equations above define the core objective used to train the adaptive blend and residual correction. Stage-specific weighting, event weighting, and decoder-specific loss terms used during optimization are described in the Implementation section.

## 4. Dataset

We evaluate GateFormer on three sky-imaging datasets with co-located GHI measurements: National Renewable Energy Laboratory Total Sky Imager (NREL-TSI), Chilbolton, and SIRTA. NREL-TSI provides hemispheric total-sky imagery from Golden, Colorado, USA (39.742 °N, 105.18 °W) at 10-minute cadence [32]. Chilbolton provides all-sky imagery from the CFARR facility in Hampshire, UK (51.145 °N, 1.437 °W) at 5-minute cadence [33], [34], [35]. SIRTA provides sky imagery and irradiance measurements from Palaiseau, France (48.713 °N, 2.208 °E) at 2-minute cadence [36]. All streams are time-aligned to paired image-radiometry timestamps, filtered for missing frames, and evaluated in GHI units of W/m², using data from 2017 to 2019. Sky images are resized to 224 by 224 pixels and standardised via ImageNet preprocessing [37]. In this study we adopt a deliberately minimalist augmentation scheme: aside from ImageNet normalisation, no rotation, colour perturbations, blurring, or similar transforms are applied. This keeps architectural comparison transparent. Investigating richer, sky-specific augmentations remains valuable future work. Sequences are constructed at the site’s native cadence using four past frames and three future horizons, providing intermediate trajectory supervision while retaining a common final forecast step for the Delta Head and adaptive blend. After timestamp alignment, daylight filtering, split assignment, and sequence construction, the final datasets contained 68,740 NREL-TSI sequences, 158,461 Chilbolton sequences, and 365,811 SIRTA sequences. For GateFormer, the train/validation/test sample counts were 46,229/11,323/11,188 (NREL +30 min), 107,198/26,327/24,936 (Chilbolton +15 min), and 248,110/59,638/58,063 (SIRTA +6 min). We use a ±10s timestamp tolerance for pairing images with radiometry and drop sequences with any missing frame. Unless stated otherwise, metrics are reported on the last horizon (e.g., +30 min at NREL-TSI). This split was done into a training set and an evaluation dataset as well as a separate testing dataset. Days 5–9 of each month were used as the fixed test set, days 15–19 as the validation set, and the remaining eligible days as the training set. This intra-month partition follows the SIAT evaluation protocol [10], enabling direct comparison while ensuring that each subset spans all months and seasons. The split is applied before sequence construction; each split is generated independently, and no training sequences share images or GHI target timestamps with validation or test sets. A daylight evaluation filter retains sequences only when future target GHI values exceed 10 W/m² and days with maximum GHI below 50 W/m² are excluded.

## 5. Implementation and Evaluation Metrics

We report MAE, RMSE, and FS vs a reference forecaster using paired-timestamp evaluation. We use $\mathrm { F S } _ { \mathrm { \tiny { s m a r t } } }$ (clear-sky scaled persistence). For a reference forecaster $\hat { \bf y } _ { \mathrm { r e f } } .$ , FS follows the RMSE-based meteorological definition relative to a reference forecaster [38],[39]. Clear-sky irradiance $G ^ { c s }$ is computed via

pvlib [40] using the simplified Solis model [41], [42]. Simplified Solis is used only to construct the common clear-sky-scaled persistence reference and does not affect the GateFormer predictions themselves.

$$
\mathrm { F S } _ { \mathrm { r e f } } ( \% ) = 1 0 0 \left( 1 - \frac { \mathrm { R M S E } ( \widehat { \mathrm { y } } , \mathrm { y } ) } { \mathrm { R M S E } ( \widehat { \mathrm { y } } _ { \mathrm { r e f } } , \mathrm { y } ) } \right)\tag{19}
$$

$$
\hat { y } _ { \mathrm { s m a r t } } ( t + \varDelta ) = y ( t ) \frac { G ^ { \mathrm { c s } } ( t + \varDelta ) } { \operatorname* { m a x } ( G ^ { \mathrm { c s } } ( t ) , \varepsilon ) }\tag{20}
$$

The backbone model employs a ViT pretrained on ImageNet [43]. Images are split into $1 6 \times 1 6$ patches, then passed through multi-head self-attention layers. We use the DeiT-III Medium backbone with two forecast decoders and a compact gating MLP [31]. Optimisation employs AdamW with a learning rate of $2 \times 1 0 ^ { - 4 }$ weight decay $2 . 5 \times \bar { 1 } 0 ^ { - 4 }$ batch size 8 [44]. Training follows a two-stage schedule. First, the shared encoder, Trajectory decoder, and Delta decoder are pretrained for 4 epochs with the backbone frozen except for the final 8 transformer blocks. The Trajectory decoder is supervised on the full future GHI sequence using RMSE loss in absolute irradiance space. Absolute GHI is retained as the prediction target to preserve a direct mapping from measured irradiance to future measured irradiance, consistent with the benchmarks used. Clear-sky information is provided as contextual input. In this study $\varepsilon = 1 0 ^ { - 8 }$ . To emphasise larger irradiance-change events, the Delta loss uses change-magnitude weighting of 2, scale 200 W/m², and cap 3. An additional eventasymmetric term $( \lambda _ { e v e n t } )$ with weight 0.15 penalises under-prediction during large upward changes and overprediction during large downward changes, using a 150 W/m² event threshold, scale 250 W/m², and cap 4. These values were fixed training hyperparameters and may be further optimised for each site. Second, the encoder and both heads are frozen, and the adaptive blend-residual module is trained for 6 epochs using the blend-residual objective. Training is performed in CUDA automatic mixed precision on a single NVIDIA RTX 3090 GPU with 24 GB memory. All site splits, configurations, and random seeds are to be released in the public repository for exact reproducibility upon acceptance.

Table 2 – Core Training Hyper Parameter Table
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>DeiT-III Medium</td></tr><tr><td>Parameters</td><td>55.70 M</td></tr><tr><td>Batch size</td><td>8</td></tr><tr><td>Optimiser</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Head pretraining</td><td>4 epochs</td></tr><tr><td>Blend-residual training</td><td>6 epochs</td></tr></table>

Hyperparameters were selected through preliminary experiments using validation RMSE on NREL-TSI. The resulting GateFormer architecture, optimisation settings and loss coefficients were then fixed across NREL-TSI, Chilbolton and SIRTA without site-specific tuning. The sensitivity analyses evaluate the resilience of the principal architectural and loss choices around these selected values.

## 6. Results and Discussion

We first evaluate GateFormer at three geographically distinct sites as described in the Dataset section. The same preprocessing pipeline, model configuration, and training protocol are used for each site, allowing performance to be compared across different camera optics, atmospheric conditions, and cloud morphology without introducing site-specific methodological changes. We reproduce the SIAT baseline using the official implementation provided by the authors. DeiT and ResNet-50 use the latest observed image, while ConvLSTM uses the last four. All evaluations are performed at each site’s native sampling cadence, Step 1 and Step 3 represent +10 and +30 min for NREL-TSI, +5 and +15 min for Chilbolton, and +2 and +6 min for SIRTA.

Table 3 – Performance comparison across sites under paired-timestamp evaluation. MAE/RMSE reported in W/m²; FS reported in %.
<table><tr><td colspan="2">Future Timesteps</td><td colspan="3">Step 1</td><td colspan="3">Step 3</td></tr><tr><td>Dataset</td><td>Model</td><td>MAE</td><td>RMSE</td><td>FS</td><td>MAE</td><td>RMSE</td><td>FS</td></tr><tr><td>NREL-TSI</td><td>Reference</td><td>57.82</td><td>128.83</td><td></td><td>81.33</td><td>162.29</td><td></td></tr><tr><td>NREL-TSI</td><td>DeiT</td><td>78.98</td><td>119.85</td><td>6.97</td><td>89.33</td><td>140.17</td><td>13.63</td></tr><tr><td>NREL-TSI</td><td>ConvLSTM</td><td>97.94</td><td>143.54</td><td>-11.42</td><td>126.62</td><td>175.88</td><td>-8.37</td></tr><tr><td>NREL-TSI</td><td>ResNet-50</td><td>73.39</td><td>114.07</td><td>11.46</td><td>82.38</td><td>134.29</td><td>17.25</td></tr><tr><td>NREL-TSI</td><td>SIAT</td><td>65.26</td><td>113.03</td><td>12.26</td><td>79.76</td><td>138.35</td><td>14.75</td></tr><tr><td>NREL-TSI</td><td>GateFormer</td><td>51.97</td><td>103.98</td><td>19.29</td><td>70.44</td><td>125.01</td><td>22.97</td></tr><tr><td>Chilbolton</td><td>Reference</td><td>55.93</td><td>126.41</td><td></td><td>76.14</td><td>150.69</td><td></td></tr><tr><td>Chilbolton</td><td>SIAT</td><td>70.57</td><td>107.54</td><td>14.93</td><td>72.66</td><td>114.60</td><td>23.95</td></tr><tr><td>Chilbolton</td><td>GateFormer</td><td>46.89</td><td>98.67</td><td>21.94</td><td>59.63</td><td>111.98</td><td>25.69</td></tr><tr><td>SIRTA</td><td>Reference</td><td>38.13</td><td>91.22</td><td></td><td>59.21</td><td>122.47</td><td></td></tr><tr><td>SIRTA</td><td>SIAT</td><td>41.19</td><td>72.86</td><td>20.13</td><td>50.05</td><td>88.20</td><td>27.98</td></tr><tr><td>SIRTA</td><td>GateFormer</td><td>28.98</td><td>71.70</td><td>21.40</td><td>40.92</td><td>87.13</td><td>28.86</td></tr></table>

Table 3 compares GateFormer with standard vision and sequence baselines. On NREL-TSI, GateFormer achieves the lowest RMSE at both forecast steps and improves the Step 3 RMSE from 138.35 W/m² for SIAT to 125.01 W/m². Final-horizon improvements are observed at each site, where GateFormer reduces SIAT RMSE by up to 9.64%. Across all three sites, GateFormer achieves the highest final-horizon FS, ranging from 22.97% to 28.86%, indicating that the proposed dual-decoder blend-residual structure improves performance across multiple sky-imaging datasets.

## 6.1. Seasonal Performance

To determine GateFormer’s effectiveness across seasonal regimes, we split the NREL-TSI +30 min test predictions by meteorological season. To evaluate the framework under different irradiance ranges, sun-path conditions, and cloud morphologies, and to determine whether the residual correction provides an additional benefit beyond the adaptive Base forecast across the seasons as seen in Table 4.

Table 4 – Performance by season (Step = 3). RMSE in W/m²; ${ \underline { { \mathrm { F S } } } } _ { \mathrm { r e s } }$ is forecasting skill rel. to Base blend (higher is better).
<table><tr><td>Season</td><td>Delta Head</td><td>Trajectory Head</td><td>Base RMSE</td><td>Final RMSE</td><td>∆RMSE</td><td>∆MAE</td><td>∆FSres(%)</td></tr><tr><td>S1 DJF (Dec – Feb)</td><td>82.34</td><td>72.19</td><td>70.21</td><td>68.47</td><td>-1.73</td><td>-1.84</td><td>+2.47</td></tr><tr><td>S2 MAM (Mar – May)</td><td>135.19</td><td>126.97</td><td>125.44</td><td>124.61</td><td>-0.83</td><td>-2.48</td><td>+0.66</td></tr><tr><td>S3 JJA (Jun – Aug)</td><td>181.95</td><td>167.25</td><td>163.64</td><td>161.42</td><td>-2.22</td><td>-7.64</td><td>+1.35</td></tr><tr><td>S4 SON (Sep – Nov)</td><td>112.43</td><td>107.58</td><td>105.15</td><td>103.14</td><td>-2.01</td><td>-6.34</td><td>+1.91</td></tr></table>

Despite the absolute RMSE changes by season, the FS of both the Base Forecast and Final Forecast remains similar across each season, the range of FS for Base and GateFormer changes 2–3%. Demonstrating GateFormer performs reliably across the year, indicating resilience under sun-path and cloud-regime shifts. The residual gives a small but consistent extra lift on top of an already strong Base Forecast. The largest absolute seasonal RMSE occurs in JJA at 161.42 W/m², where the residual also provides the largest RMSE and MAE gains, improving RMSE by −2.22 W/m² and MAE by −7.64 W/m². The smallest RMSE gain occurs in MAM at −0.83 W/m², but the residual contribution remains positive in every season. Across seasons, the same component ordering is preserved: the Trajectory Head improves over persistence, the adaptive blend improves over the individual heads, and the residual final prediction provides the best seasonal RMSE. This indicates that the proposed decomposition remains useful under different seasonal irradiance ranges and cloud regimes.

## 6.2. Ablations and Analysis

Table 5–6 reports the component-level ablation on NREL-TSI at the final forecast step. This is done to isolate each part of GateFormer and its contribution to the final prediction. The “No GHI Token” variant removes token-level irradiance conditioning from the transformer encoder. The No GHI Token variant removes GHI only from the transformer token sequence. Historical GHI-derived information remains available through the decoder and gating context, so the ablation isolates token-level irradiance conditioning rather than removing all GHI information. “No Residual (Base)” keeps the adaptive blend but removes the final residual. “GatePersist” replaces the learned Delta Head with a persistence-based branch to evaluate whether the second pathway needs to be learned. “Equal-Average” replaces the learned blend with a fixed average of the Trajectory and Delta predictions. The individual Trajectory Head and Delta Head rows report the performance of each decoder alone, while the DeiT row provides the backbone-only baseline. The GHI-only variant receives the same four historical GHI observations and corresponding temporal/clear-sky context as GateFormer but receives no sky images. ΔRMSE and ΔMAE are reported relative to the full GateFormer framework, positive values indicate worse performance than the full model.

Table 5 – Ablation study on NREL-TSI (+30 min). ΔRMSE and ΔMAE are reported as error increases rel. to the GateFormer.
<table><tr><td>Variant</td><td>RMSE</td><td>MAE</td><td>ΔRMSE</td><td>∆MAE</td></tr><tr><td>GateFormer (Final)</td><td>125.01</td><td>70.44</td><td></td><td></td></tr><tr><td>No GHI Token</td><td>125.44</td><td>71.09</td><td>+0.43</td><td>+0.65</td></tr><tr><td>No Residual (Base)</td><td>126.69</td><td>75.30</td><td>+1.68</td><td>+4.86</td></tr><tr><td>GatePersist</td><td>128.00</td><td>72.76</td><td>+2.99</td><td>+2.32</td></tr><tr><td>Equal-Average</td><td>129.62</td><td>77.42</td><td>+4.61</td><td>+6.98</td></tr><tr><td>Trajectory Head</td><td>129.23</td><td>79.12</td><td>+4.22</td><td>+8.68</td></tr><tr><td>Delta Head</td><td>139.36</td><td>84.81</td><td>+14.35</td><td>+14.37</td></tr><tr><td>Plain DeiT</td><td>140.17</td><td>89.33</td><td>+15.16</td><td>+18.89</td></tr><tr><td>GHI-only</td><td>143.37</td><td>86.62</td><td>+18.36</td><td>+16.18</td></tr></table>

The ablation confirms the role of each component in the proposed architecture. The Trajectory Head is the stronger individual decoder overall, reaching 129.23 W/m² RMSE compared with 139.36 W/m² for the Delta Head. However, the Delta Head is not redundant: it outperforms the Trajectory Head in the most extreme absolute-change regime, reducing RMSE from 319.48 to 294.74 W/m² for | ΔGHI | >250 W/m², and especially during large downward ramps, where RMSE decreases from 296.10 to 210.10 W/m² for drops exceeding 250 W/m². Whereas a fixed equal-average blend of the two heads reaches 129.62 W/m² RMSE, which does not improve over the Trajectory Head. In contrast, the learned adaptive blend reduces RMSE to 126.69 W/m² and MAE to 75.30 W/m², improving over fixed averaging by 2.93 W/m² RMSE and 2.12 W/m² MAE. This shows that the two decoder outputs are complementary, but that their contribution must be weighted adaptively rather than combined with a fixed average.

The GatePersist variant helps isolate the value of the learned Delta Head. In this variant, the Delta Head is replaced with a hard-coded anchor-persistence branch, the adaptive blend therefore receives less learned visual-irradiance information from the second pathway. As a result, the blend and residual stages provide a limited additional benefit: the Trajectory branch within this ablation achieves 128.23 W/m² RMSE, while the residual-corrected output reaches 128.00 W/m². This suggests that the learned Delta Head does more than provide another forecast; it supplies complementary image-informed structure that makes the adaptive blend and the residual correction more effective, even if that learned branch may also carry some forecast error.

Res Dir (%) is the percentage of samples for which the learned residual correction moves the Base forecast towards the observed final-horizon GHI by increasing an underprediction or decreasing an overprediction. It does not describe the direction of the irradiance ramp itself.

Table 6 – Overall ablation and extreme-ramp analysis on NREL-TSI at +30 min. RMSE, MAE and ΔRMSE are reported in W/m².
<table><tr><td>Variant/output</td><td>RMSE</td><td>MAE</td><td>|∆|≥250</td><td>Drop 250</td><td>Res Dir (%)</td><td> $\Delta \mathbf { R } \mathbf { M } \mathbf { S } \mathbf { E } _ { \mathbf { R i s e 2 5 0 } }$ </td><td> $\Delta \mathbf { R M S E _ { D r o p 2 5 0 } }$ </td></tr><tr><td>GateFormer</td><td>125.01</td><td>70.44</td><td>303.39</td><td>289.78</td><td>61.39</td><td>-26.27</td><td>+15.46</td></tr><tr><td>No GHI Token</td><td>125.44</td><td>71.09</td><td>303.83</td><td>282.54</td><td>68.68</td><td>-24.19</td><td>+26.33</td></tr><tr><td>No Residual (Base)</td><td>126.69</td><td>75.30</td><td>308.03</td><td>274.33</td><td></td><td></td><td></td></tr><tr><td>GatePersist</td><td>128.00</td><td>72.76</td><td>314.00</td><td>290.56</td><td>58.29</td><td>+8.12</td><td>-8.74</td></tr><tr><td>Equal Average</td><td>129.62</td><td>77.42</td><td>301.31</td><td>246.88</td><td></td><td></td><td></td></tr><tr><td>Trajectory Head</td><td>129.23</td><td>79.12</td><td>319.48</td><td>296.10</td><td></td><td></td><td></td></tr><tr><td>Delta Head</td><td>139.36</td><td>84.81</td><td>294.74</td><td>210.10</td><td></td><td></td><td></td></tr></table>

The GHI-token ablation suggests that residual direction accuracy is not the only determinant of correction quality. Although removing the GHI token increases overall push-direction accuracy from 61.39% to 68.68%, it reduces the positive-ramp residual gain for rises above 250 W/m² from 26.27 to 24.19 W/m² and worsens negative-ramp degradation for drops below 250 W/m² from 15.46 to 26.33 W/m². Despite the residual sign accuracy being less favourable, this indicates that token-level GHI conditioning provides useful irradiancestate information for estimating the magnitude of upward irradiance changes.

Having established the contribution of the principal components, Tables 7–9 examine sensitivity to the ramp-oriented loss configuration, input-history length, and Delta-head target design.

Table 7 – Ramp-oriented loss sensitivity on NREL-TSI.
<table><tr><td>Parameter changed</td><td>Setting</td><td>Overall RMSE</td><td>MAE</td><td>Rise ≥ 250</td><td>Drop ≤ 250</td></tr><tr><td>Paper configuration</td><td></td><td>125.011</td><td>70.439</td><td>319.798</td><td>289.784</td></tr><tr><td rowspan="3">Change Coefficient  $( \lambda _ { c h g } )$ </td><td>0</td><td>126.488</td><td>70.581</td><td>326.591</td><td>295.763</td></tr><tr><td>1</td><td>126.293</td><td>71.313</td><td>331.781</td><td>282.193</td></tr><tr><td>4</td><td>128.152</td><td>71.896</td><td>324.477</td><td>288.723</td></tr><tr><td rowspan="3">Event Coefficient  $( \lambda _ { e v e n t } )$ </td><td>0</td><td>127.007</td><td>72.453</td><td>342.006</td><td>287.789</td></tr><tr><td>0.075</td><td>128.065</td><td>73.100</td><td>324.492</td><td>276.407</td></tr><tr><td>0.30</td><td>127.572</td><td>71.804</td><td>328.995</td><td>282.023</td></tr><tr><td rowspan="2">Event Threshold  $( \pmb { \tau _ { e v e n t } } )$ </td><td>75</td><td>126.874</td><td>71.191</td><td>325.832</td><td>289.729</td></tr><tr><td>300</td><td>125.938</td><td>70.425</td><td>337.610</td><td>280.582</td></tr><tr><td rowspan="2">Event Scale  $( s _ { e v e n t } )$ </td><td>125</td><td>126.722</td><td>72.135</td><td>329.607</td><td>289.951</td></tr><tr><td>500</td><td>126.176</td><td>70.321</td><td>334.962</td><td>289.270</td></tr><tr><td rowspan="2">Event Cap  $( c _ { e v e n t } )$ </td><td>2</td><td>125.978</td><td>72.031</td><td>327.127</td><td>268.513</td></tr><tr><td>8</td><td>126.176</td><td>70.321</td><td>334.962</td><td>289.270</td></tr><tr><td rowspan="2">Change Scale  $( s _ { c h g } )$ </td><td>100</td><td>127.993</td><td>72.458</td><td>315.911</td><td>304.525</td></tr><tr><td>400</td><td>126.022</td><td>70.721</td><td>332.173</td><td>272.203</td></tr><tr><td rowspan="2">Change Cap  $( c _ { c h g } )$ </td><td>1.5</td><td>126.136</td><td>71.470</td><td>340.070</td><td>282.183</td></tr><tr><td>6</td><td>128.007</td><td>72.881</td><td>324.684</td><td>294.415</td></tr></table>

The paper configuration provided the best overall RMSE among the tested settings. Removing changemagnitude weighting degraded overall, positive-ramp, and negative-ramp performance. Removing the asymmetric event term also degraded overall and positive-ramp performance but slightly improved negativeramp RMSE. The observed behaviour therefore indicates a trade-off between aggregate accuracy and direction-specific extreme-ramp performance rather than a uniform degradation caused by the asymmetric loss.

Table 8 – Input-history sensitivity (NREL-TSI +30). Changes are rel. to four-frame configuration, evaluated on the same samples.
<table><tr><td>Historical frames</td><td>Look-back duration</td><td>∆RMSE</td><td>∆MAE</td></tr><tr><td>1</td><td>0 min</td><td>+3.39</td><td>+1.54</td></tr><tr><td>2</td><td>10 min</td><td>+3.07</td><td>+2.03</td></tr><tr><td>3</td><td>20 min</td><td>+1.49</td><td>+0.94</td></tr><tr><td>4</td><td>30 min</td><td>0.00</td><td>0.00</td></tr><tr><td>5</td><td>40 min</td><td>+1.47</td><td>+1.01</td></tr><tr><td>6</td><td>50 min</td><td>+2.34</td><td>+2.02</td></tr></table>

Table 9 – Delta-horizon sensitivity on NREL-TSI at +30 min. Changes are relative to the final-horizon-only Delta configuration.
<table><tr><td>Delta configuration</td><td>Delta Targets</td><td>Delta to Gate</td><td>ΔRMSE</td><td>∆MAE</td></tr><tr><td>GateFormer</td><td>∆yt+H</td><td> $\widehat { \mathbf { y } } _ { t + H } ^ { \Delta }$ </td><td>0.00</td><td>0.00</td></tr><tr><td>Multi-horizon Delta</td><td> $[ \varDelta y _ { t + 1 } , \varDelta y _ { t + 2 } , \varDelta y _ { t + H } ]$ </td><td> $\widehat { \mathsf { y } } _ { t + H } ^ { \Delta }$ </td><td>+4.04</td><td>+2.01</td></tr><tr><td>Multi-horizon Delta (Outputs to Gate)</td><td> $[ \varDelta y _ { t + 1 } , \varDelta y _ { t + 2 } , \varDelta y _ { t + H } ]$ </td><td> $\left[ \widehat { \mathbf { y } } _ { t + 1 } ^ { \Delta } , \widehat { \mathbf { y } } _ { t + 2 } ^ { \Delta } , \widehat { \mathbf { y } } _ { t + H } ^ { \Delta } \right]$ </td><td>+4.16</td><td>+2.09</td></tr></table>

Four historical frames achieved the lowest error among the tested NREL-TSI configurations and were therefore retained across all datasets. Predicting Delta values at additional horizons increased final-horizon error in this sensitivity analysis, supporting the Delta Head’s role as a final-horizon specialist while the Trajectory Head provides the intermediate forecasts.

To determine whether the improvement arises simply from model capacity or conventional forecast combination, Table 10 compares GateFormer with parameter-matched (PM) single-head models and combinations of independently trained Trajectory-PM and Delta-PM forecasts. The Trajectory-PM models predict either the complete trajectory or only the final horizon, while Delta-PM predicts the final-horizon change. The combination controls use equal averaging, a fixed validation-selected weight, ordinary least squares (OLS) stacking, or a sample-dependent post-hoc gate.

Table 10 – Forecast-space controls on NREL-TSI (+30 min).
<table><tr><td>Variant</td><td>Control</td><td>Param (M⁺)</td><td>RMSE</td><td>MAE</td></tr><tr><td>Trajectory-PM</td><td>Single Head</td><td>55.692</td><td>130.00</td><td>78.15</td></tr><tr><td>Trajectory-PM (H=3 only)</td><td>Single Head</td><td>55.691</td><td>129.82</td><td>77.70</td></tr><tr><td>Delta-PM</td><td>Single Head</td><td>55.691</td><td>134.93</td><td>81.03</td></tr><tr><td>Equal Average</td><td>Two Model Combination</td><td>111.383</td><td>127.37</td><td>73.39</td></tr><tr><td>Validation Weighted Blend</td><td>Two Model Combination</td><td>111.383</td><td>127.08</td><td>73.38</td></tr><tr><td>OLS Stacking</td><td>Two Model Combination</td><td>111.383</td><td>126.94</td><td>73.84</td></tr><tr><td>Post-hoc Dynamic Gate</td><td>Two Model Combination</td><td>111.383</td><td>127.20</td><td>73.48</td></tr><tr><td>GateFormer (Base)</td><td>Single Model Dual-head</td><td>55.702</td><td>126.69</td><td>75.30</td></tr><tr><td>GateFormer Final</td><td>Base + Residual</td><td>55.702</td><td>125.01</td><td>70.44</td></tr></table>

All parameter-matched single-head controls produce higher error than GateFormer Base, while the complete GateFormer also outperforms the independently trained post-hoc combinations. Relative to the stronger final-only single-head control, GateFormer Base reduces RMSE by 3.13 W/m² and GateFormer Final reduces it by 4.81 W/m². GateFormer Final also reduces RMSE by 1.93 W/m² compared with the strongest conventional combination.

## 6.3. Ramp-Regime Diagnostics

We next evaluate GateFormer across different final-horizon irradiance-change regimes on NREL-TSI. The regimes are defined using the measured change between the latest observed GHI and the final +30 min target. This separates low-change periods from positive and negative ramp events, allowing us to determine whether the residual head improves the adaptive Base forecast uniformly or only under specific irradiancechange conditions. Figure 2 illustrates representative stable, positive-ramp, and negative-ramp sequences, where positive- and negative-ramp days contain the corresponding ramp type but are not mutually exclusive. While Table 11 provides the corresponding regime-wise quantitative assessment.

![](images/c18ce2768452dfe748c66c9821764dff451c5567297d5ac5d36b78427a8c33c4.jpg)  
Figure 2 – Representative NREL-TSI +30 min test-set examples across distinct irradiance regimes. Continuous predictions are shown for GHI truth, smart persistence, the Trajectory Head, the Base forecast, and the final GateFormer forecast, together with the predicted residual correction.

Table 11 – Residual diagnostics. ΔRMSE and ΔMAE are calculated as Final minus Base (negative values indicate improvement)
<table><tr><td>Regime</td><td>n</td><td>Base(RMSE)</td><td>Final(RMSE)</td><td>∆RMSE</td><td>Base(MAE)</td><td>Final(MAE)</td><td>∆MAE</td><td>Res Dir %</td></tr><tr><td>All</td><td>11188</td><td>126.69</td><td>125.01</td><td>-1.68</td><td>75.30</td><td>70.44</td><td>-4.86</td><td>61.39</td></tr><tr><td>|∆|&lt;75</td><td>5294</td><td>66.68</td><td>67.37</td><td>+0.68</td><td>40.69</td><td>38.53</td><td>-2.15</td><td>58.41</td></tr><tr><td>Rise 150</td><td>960</td><td>274.76</td><td>253.43</td><td>-21.33</td><td>224.43</td><td>199.69</td><td>-24.74</td><td>85.31</td></tr><tr><td>Drop 150</td><td>1097</td><td>228.56</td><td>242.69</td><td>+14.12</td><td>184.55</td><td>195.00</td><td>+10.45</td><td>37.92</td></tr><tr><td>|∆|&gt;150</td><td>2057</td><td>251.18</td><td>247.76</td><td>-3.42</td><td>203.16</td><td>197.19</td><td>-5.97</td><td>60.04</td></tr><tr><td>Rise 250</td><td>497</td><td>346.07</td><td>319.80</td><td>-26.27</td><td>300.39</td><td>268.22</td><td>-32.17</td><td>86.12</td></tr><tr><td>Drop 250</td><td>630</td><td>274.33</td><td>289.78</td><td>+15.46</td><td>236.29</td><td>248.11</td><td>+11.82</td><td>37.62</td></tr><tr><td>|∆|&gt;250</td><td>1127</td><td>308.03</td><td>303.39</td><td>-4.65</td><td>264.56</td><td>256.98</td><td>-7.58</td><td>59.01</td></tr></table>

Although the proposed framework demonstrates strong performance and overall forecasting accuracy, the residual correction is not equally effective across all regimes: it improves positive ramp events but can degrade negative ramp cases. The residual correction reduces RMSE substantially during positive ramps, with gains of 21.33 W/m² for rises above 150 W/m² and 26.27 W/m² for rises above 250 W/m². However, it degrades negative ramps, where residual direction accuracy is not as strong as its positive ramp counterpart. This identifies negative-ramp correction as a valuable direction for future research.

From a practical perspective, GateFormer adds training cost through its dual decoders and second-stage optimisation but retains a single shared ViT backbone rather than requiring multiple independently trained models. Inference therefore requires one backbone pass followed by comparatively lightweight gating and residual operations. This additional complexity is balanced by consistent accuracy gains across sites and improved positive-ramp forecasting, supporting applications in sub-hourly balancing, storage scheduling, and ramp-rate management using only sky images and measured GHI. Training was completed on a single NVIDIA RTX 3090, although systematic timing comparisons and deployment on lower-resource hardware were not evaluated.

## 7. Conclusion

Operational solar forecasting requires vision models that can convert sky-image sequences into accurate irradiance forecasts. GateFormer addresses this by embedding observed GHI directly into the transformer token stream, producing two complementary forecast decoders, and adaptively blending them before applying a residual correction. On NREL-TSI at +30 min, GateFormer achieved 125.01 W/m² RMSE and 22.97% FS, improving over the SIAT transformer baseline by 9.6% under paired-timestamp evaluation. Across NREL-TSI, Chilbolton, and SIRTA, GateFormer achieved the lowest final-horizon RMSE among the evaluated baselines, with FS ranging from 22.97% to 28.86% relative to smart persistence. Component ablations show that equal averaging does not improve over the stronger Trajectory Head, whereas the learned blend improves over both individual decoders and residual correction provides a further 1.68 W/m² RMSE and 4.86 W/m² MAE reduction after blending.

These results demonstrate that short-term irradiance forecasting benefits from decomposing the prediction into complementary stages: visual-irradiance encoding, trajectory prediction, final-horizon change, adaptive blending, and residual correction. Although negative-ramp correction remains an area for improvement, rampregime diagnostics show residual correction improves performance overall and reduces RMSE by up to 26.27 W/m² during positive-ramp events. These gains are relevant to short-term PV balancing, storage scheduling, and ramp-rate management under rapidly changing cloud conditions. Demonstrating that beyond the design of input pathways, feature fusion, and separate system forecast combination, substantial performance gains may also be realised by explicitly structuring how complementary predictions are formed and combined within a shared architecture. By separating representation learning from forecast formation, GateFormer provides a foundation with clear potential for deployment across diverse PV sites using only sky images and measured GHI.

## CRediT Authorship Contribution Statement

David Hamlyn: Writing – original draft, Visualisation, Software, Investigation, Conceptualisation. Sunny Chaudhary: Writing – review & editing, Software, Supervision, Conceptualisation. Jonathon Hare: Writing – review & editing, Supervision, Conceptualisation. Tasmiat Rahman: Writing – review & editing, Supervision, Conceptualisation.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability Statement

The underlying data and model code are available upon request.

## Acknowledgements

D. Hamlyn acknowledges the support of the Engineering and Physical Sciences Research Council (EPSRC) Doctoral Training Partnership (DTP) Funding grant EP/W524621/1. T. Rahman acknowledges support from EPSRC grant EP/X033333/1. We would like to thank the Chilbolton observatory for providing the data used in this publication.

## References

[1] D. Yang et al., “Verification of deterministic solar forecasts,” Solar Energy, vol. 210, pp. 20–37, Nov. 2020, doi: 10.1016/J.SOLENER.2020.04.019.

[2] R. A. Rajagukguk, R. A. A. Ramadhan, and H. J. Lee, “A Review on Deep Learning Models for Forecasting Time Series Data of Solar Irradiance and Photovoltaic Power,” Energies 2020, Vol. 13, Page 6623, vol. 13, no. 24, p. 6623, Dec. 2020, doi: 10.3390/EN13246623.

[3] A. K. Barnes, J. C. Balda, and J. K. Hayes, “Modelling PV Clouding Effects Using a Semi-Markov Process with Application to Energy Storage,” IFAC Proceedings Volumes, vol. 47, no. 3, pp. 9444– 9449, Jan. 2014, doi: 10.3182/20140824-6-ZA-1003.00825.

[4] F. Lin, Y. Zhang, and J. Wang, “Recent advances in intra-hour solar forecasting: A review of groundbased sky image methods,” Int. J. Forecast., vol. 39, no. 1, pp. 244–265, Jan. 2023, doi: 10.1016/J.IJFORECAST.2021.11.002.

[5] Q. Paletta et al., “Advances in solar forecasting: Computer vision with deep learning,” Advances in Applied Energy, vol. 11, p. 100150, Sep. 2023, doi: 10.1016/J.ADAPEN.2023.100150.

[6] C. W. Chow et al., “Intra-hour forecasting with a total sky imager at the UC San Diego solar energy testbed,” Solar Energy, vol. 85, no. 11, pp. 2881–2893, Nov. 2011, doi: 10.1016/J.SOLENER.2011.08.025.

[7] S. Quesada-Ruiz, Y. Chu, J. Tovar-Pescador, H. T. C. Pedro, and C. F. M. Coimbra, “Cloud-tracking methodology for intra-hour DNI forecasting,” Solar Energy, vol. 102, pp. 267–275, Apr. 2014, doi: 10.1016/J.SOLENER.2014.01.030.

[8] R. Marquez and C. F. M. Coimbra, “Intra-hour DNI forecasting based on cloud tracking image analysis,” Solar Energy, vol. 91, pp. 327–336, May 2013, doi: 10.1016/J.SOLENER.2012.09.018.

[9] Q. Paletta, A. Hu, G. Arbod, and J. Lasenby, “ECLIPSE : Envisioning CLoud Induced Perturbations in Solar Energy,” Appl. Energy, vol. 326, Nov. 2022, doi: 10.1016/j.apenergy.2022.119924.

[10] T. M. Mercier, T. Rahman, and A. Sabet, “Solar Irradiance Anticipative Transformer,” IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, vol. 2023-June, pp. 2065–2074, May 2023, doi: 10.1109/CVPRW59228.2023.00200.

[11] J. Liu, H. Zang, L. Cheng, T. Ding, Z. Wei, and G. Sun, “A Transformer-based multimodal-learning framework using sky images for ultra-short-term solar irradiance forecasting,” Appl. Energy, vol. 342, p. 121160, Jul. 2023, doi: 10.1016/J.APENERGY.2023.121160.

[12] L. Zhang, R. Wilson, M. Sumner, and Y. Wu, “Advanced multimodal fusion method for very shortterm solar irradiance forecasting using sky images and meteorological data: A gate and transformer mechanism approach,” Renew. Energy, vol. 216, p. 118952, Nov. 2023, doi: 10.1016/J.RENENE.2023.118952.

[13] C. Ganvir, D. Dinesh, R. Gupta, S. K. Jha, and P. K. Raghuvanshi, “Prediction of Global Horizontal Irradiance based on eXplainable Artificial Intelligence,” 2024 International Conference on Intelligent and Innovative Technologies in Computing, Electrical and Electronics (IITCEE), pp. 1–4, Jan. 2024, doi: 10.1109/IITCEE59897.2024.10467440.

[14] R. Gupta, S. K. Jha, P. Jha, K. Chaprana, and S. K. Singh, “Enhancing the accuracy of global horizontal irradiance estimation model using convolutional neural network coupled with wavelet transform,” The European Physical Journal Plus, vol. 139, no. 10, p. 924, Oct. 2024, doi: 10.1140/epjp/s13360-024- 05730-x.

[15] R. Gupta, A. K. Yadav, S. K. Jha, and P. K. Pathak, “Comparative analysis of advanced machine learning classifiers based on feature engineering framework for weather prediction,” Scientia Iranica, 2024, doi: 10.24200/sci.2024.61305.7242.

[16] K. Bharti, S. K. Singh, S. K. Jha, and R. Gupta, “Modelling and Simulation of Solar Water Pump Using Arduino Uno in Proteus,” 2023 International Conference on Intelligent and Innovative Technologies in Computing, Electrical and Electronics (IITCEE), pp. 774–781, Jan. 2023, doi: 10.1109/IITCEE57236.2023.10091075.

[17] A. Vaswani et al., “Attention Is All You Need,” Adv. Neural Inf. Process. Syst., vol. 2017-December, pp. 5999–6009, Jun. 2017, Accessed: Mar. 26, 2025. [Online]. Available: https://arxiv.org/abs/1706.03762v7

[18] A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021 - 9th International Conference on Learning Representations, Oct. 2020, Accessed: Mar. 26, 2025. [Online]. Available: https://arxiv.org/abs/2010.11929v2

[19] T. M. Mercier, A. Sabet, and T. Rahman, “Vision transformer models to measure solar irradiance using sky images in temperate climates,” Appl. Energy, vol. 362, p. 122967, May 2024, doi: 10.1016/J.APENERGY.2024.122967.

[20] G. Piantadosi, S. Dutto, A. Galli, S. De Vito, C. Sansone, and G. Di Francia, “Photovoltaic power forecasting: A Transformer based framework,” Energy and AI, vol. 18, p. 100444, Dec. 2024, doi: 10.1016/J.EGYAI.2024.100444.

[21] D. Hamlyn, S. Chaudhary, and T. Rahman, “Vision transformers for estimating irradiance using data scarce sky images,” Energy and AI, vol. 21, p. 100560, Sep. 2025, doi: 10.1016/J.EGYAI.2025.100560.

[22] S. Chen, T. Jing, M. Li, H. H. Lee, M. C. Lam, and S. Bu, “Interpretable transformer based intra-day solar forecasting with spatiotemporal satellite and numerical weather prediction inputs,” Energy and AI, vol. 23, p. 100667, Jan. 2026, doi: 10.1016/J.EGYAI.2025.100667.

[23] Z. Song, F. Xiao, Z. Chen, and H. Madsen, “Probabilistic ultra-short-term solar photovoltaic power forecasting using natural gradient boosting with attention-enhanced neural networks,” Energy and AI, vol. 20, p. 100496, May 2025, doi: 10.1016/J.EGYAI.2025.100496.

[24] J. Huang, B. Shao, Y. Ke, Y. Gao, and S. Mazzoni, “PV-MM-diffusion: An end-to-end multi-modal diffusion model for ultra-short-term probabilistic photovoltaic forecasting,” Appl. Energy, vol. 413, p. 127816, Jun. 2026, doi: 10.1016/J.APENERGY.2026.127816.

[25] B. Xiong et al., “Multimodal ultra-short-term probabilistic solar power forecasting with generative AI and Transformer,” Advances in Applied Energy, vol. 20, p. 100250, Dec. 2025, doi: 10.1016/J.ADAPEN.2025.100250.

[26] X. Liao, M. S. Wong, and R. Zhu, “Dual-gate Temporal Fusion Transformer for estimating large-scale land surface solar irradiation,” Renewable and Sustainable Energy Reviews, vol. 214, p. 115510, May 2025, doi: 10.1016/J.RSER.2025.115510.

[27] A. Sardarabadi, A. Heydarian Ardakani, S. Matrone, E. Ogliari, and E. Shirazi, “Multi-temporal PV power prediction using long short-term memory and wavelet packet decomposition,” Energy and AI, vol. 21, p. 100540, Sep. 2025, doi: 10.1016/J.EGYAI.2025.100540.

[28] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive Mixtures of Local Experts,” Neural Comput., vol. 3, no. 1, pp. 79–87, Mar. 1991, doi: 10.1162/NECO.1991.3.1.79.

[29] J. M. Bates and C. W. J. Granger, “The Combination of Forecasts,” Journal of the Operational Research Society, vol. 20, no. 4, pp. 451–468, Dec. 1969, doi: 10.1057/JORS.1969.103.

[30] B. Nouri, J. Lezaca, Y. Fabel, A. Hammer, N. Blum, and S. Wilbert, “Enhancing Intra-Hour Solar Irradiance Forecasting for Solar Applications: A Blended Model of Satellite, Sky Imager, and Persistence,” Solar RRL, vol. 9, no. 24, p. e202500486, Dec. 2025, doi: 10.1002/SOLR.202500486.

[31] H. Touvron, M. Cord, and H. Jégou, “DeiT III: Revenge of the ViT,” in ECCV 2022: 17th European Conference, Tel Aviv, Israel: Springer Science and Business Media Deutschland GmbH, 2022, pp. 516–533. doi: 10.1007/978-3-031-20053-3\_30.

[32] “NREL Solar Radiation Research Laboratory (SRRL): Baseline Measurement System (BMS), Golden, Colorado | NREL Data Catalog.” Accessed: Oct. 03, 2025. [Online]. Available: https://data.nrel.gov/submissions/7

[33] “Dataset Record: Chilbolton Facility for Atmospheric and Radio Research (CFARR) Meteorological Sensor Data, Chilbolton Site.” Accessed: Mar. 24, 2025. [Online]. Available: https://catalogue.ceda.ac.uk/uuid/45b25a7c531563f4422afcaeea0f07a7/

[34] “Dataset Record: Chilbolton Facility for Atmospheric and Radio Research (CFARR) Direct Visible Radiometer Data.” Accessed: Mar. 24, 2025. [Online]. Available: https://catalogue.ceda.ac.uk/uuid/504312b7a9f4e905181c8484f8a3c405/

[35] “Dataset Record: Chilbolton Facility for Atmospheric and Radio Research (CFARR): cloud camera 2 imagery from Chilbolton, Hampshire (2016-present).” Accessed: Mar. 24, 2025. [Online]. Available: https://catalogue.ceda.ac.uk/uuid/f55f5649110b4b98b3d5177d8ff2eac9/

[36] M. Haeffelin et al., “SIRTA, a ground-based atmospheric observatory for cloud and aerosol research,” Ann. Geophys., vol. 23, no. 2, pp. 253–275, Feb. 2005, doi: 10.5194/ANGEO-23-253-2005.

[37] J. Deng, W. Dong, R. Socher, L. J. Li, K. Li, and L. Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database,” 2009 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2009, pp. 248–255, 2009, doi: 10.1109/CVPR.2009.5206848.

[38] A. H. Murphy and E. S. Epstein, “Skill scores and correlation coefficients in model verification,” Mon. Weather Rev., vol. 117, no. 3, pp. 572–582, Mar. 1989, doi: 10.1175/1520- 0493(1989)117<0572:SSACCI>2.0.CO;2.

[39] M. P. Mittermaier, “The potential impact of using persistence as a reference forecast on perceived forecast skill,” Weather Forecast., vol. 23, no. 5, pp. 1022–1031, 2008, doi: 10.1175/2008WAF2007037.1.

[40] W. F. Holmgren, C. W. Hansen, and M. A. Mikofski, “pvlib python: a python package for modeling solar energy systems,” J. Open Source Softw., vol. 3, no. 29, p. 884, Sep. 2018, doi: 10.21105/JOSS.00884.

[41] P. Ineichen, “Validation of models that estimate the clear sky global and beam solar irradiance,” Solar Energy, vol. 132, pp. 332–344, Jul. 2016, doi: 10.1016/J.SOLENER.2016.03.017.

[42] P. Ineichen, “A broadband simplified version of the Solis clear sky model,” Solar Energy, vol. 82, no. 8, pp. 758–762, Aug. 2008, doi: 10.1016/J.SOLENER.2008.02.009.

[43] R. Wightman et al., “rwightman/pytorch-image-models: v0.8.10dev0 Release”, doi: 10.5281/ZENODO.7618837.

[44] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” 7th International Conference on Learning Representations, ICLR 2019, Nov. 2017, Accessed: Mar. 24, 2025. [Online]. Available: https://arxiv.org/abs/1711.05101v3