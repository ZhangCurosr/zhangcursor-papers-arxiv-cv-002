# GlassFormer: Learning Real-time Glass Segmentation using Radar-Depth Fusion

Suhani Grover, Astik Srivastava, Viswas Dinesh, Avinash Sharma, and Madhava Krishna

Abstract— Transparent surfaces are ubiquitous in built environments, yet they remain a persistent failure case for robotic perception. RGB cameras perceive the background behind glass rather than the surface itself, while depth sensors such as LiDAR, time-of-flight, and RGB-D often return invalid or background measurements in transparent regions. As a result, systems that rely solely on optical sensing may misinterpret glass walls, doors, or mirrors as free space, compromising safe and reliable navigation.

Existing glass segmentation approaches address this by learning visual cues such as reflections, boundaries, and semantic context from RGB images. While effective under favourable lighting and viewing conditions, these cues degrade in lowlight environments, under glare, or when glass surfaces are featureless or partially occluded. In this work, we propose a multimodal framework that fuses millimetre-wave radar with RGB-D sensing for real-time transparent surface segmentation. Radar reflects strongly off glass surfaces, providing a geometric cue that remains reliable precisely where vision and depth fail. We exploit this cross-modal inconsistency to generate a radarguided spatial prior, which is integrated into a lightweight transformer-based segmentation network, GlassFormer, via cross-modal attention.

To evaluate our approach, we collect a synchronized RGB-D-radar dataset spanning diverse glass types, including doors, windows, and mirrors, across a wide range of lighting conditions from daylight to near-dark environments. We report results on a mixed-condition test split covering all scene types and a dedicated low-light split designed to stress vision-only methods. GlassFormer achieves 0.88 mIoU on the mixed split, and 0.59 mIoU on the low light split, demonstrating substantial robustness gains over vision-only baselines while maintaining real-time performance on resource-constrained platforms.

## I. INTRODUCTION

Transparent surfaces such as glass doors, partitions, and windows remain a fundamental failure case for robotic perception systems. Unlike most obstacles, glass is simultaneously invisible to conventional depth sensors and visually ambiguous to RGB-based methods, making it difficult to detect reliably across real-world operating conditions.

This challenge stems from the two complementary properties of glass. First, glass lacks intrinsic colour or texture; its visual appearance is determined almost entirely by the scene around and behind it, making it ambiguous to RGBbased methods, particularly under low lighting or featureless backgrounds. Second, glass simultaneously reflects, refracts, and transmits incident electromagnetic energy. As a result, RGB-D cameras, time-of-flight sensors, and LiDAR frequently misinterpret glass as free space or return invalid depth at transparent interfaces, which is a drawback that is intrinsic to the sensing modality and persists irrespective of ambient illumination. RGB-only methods that rely on visual cues such as reflections and boundaries degrade specifically under low-light, glare or featureless conditions as these cues become unreliable when insufficient texture or lighting is available. While ultrasonic sensors can detect transparent obstacles at short range, their sparse and limited spatial measurements make dense segmentation challenging. Prior work has explored a range of visual cues to address this, including boundaries, reflections, polarization and semantic context [1]–[6]. While these methods perform well under favourable conditions, their core assumptions - visible boundaries, identifiable reflections and adequate illumination break down in scenarios relevant to practical deployment, such as low-light corridors, high-glare atriums, and featureless glass panes. Large-scale datasets such as ClearPose [7] and TRansPose [8] have significantly advanced transparent object perception. However, they primarily target objectlevel manipulation or multispectral perception rather than dense scene-level glass segmentation for mobile robotics. This reflects a fundamental limitation of optical sensing in the presence of transparent surfaces. We observe that the millimeter-wave (mmWave) radar occupies a complementary position in this sensing landscape. At 60 GHz, radar wavelengths are insensitive to ambient lighting and crucially, are not refracted through glass, addressing illuminationdependent failure mode of RGB cues and the illuminationindependent failure mode of depth sensing simultaneously. While radar’s interaction differs physically for transmissive glass versus specular mirrors, our evaluation and qualitative results (Fig. 4) indicate the method performs effectively on both surface types. Despite widespread adoption of radar in robotics for proximity sensing and automotive perception, its use as a spatial prior for vision-based segmentation has not yet been explored. While radar alone cannot produce pixeldense segmentation masks, even a coarse range estimate can narrow down the search space considerably for a downstream vision model.

![](images/af86f675ca242fcb83cae8f367c97c4210365757cd64c2bedc54a583f8e5cd3d.jpg)  
Fig. 1. RGB-D and LiDAR fail on glass, returning background depth or invalid returns through the surface, while mmWave radar reflects strongly off it. This cross-modal disagreement yields a radar-guided prior (pink) over likely transparent regions, which GlassFormer fuses with RGB to segment glass reliably, even in low light.

In this work, we introduce GlassFormer, a radar-guided transparent surface segmentation framework designed for real-time deployment on resource-constrained robotics platforms. A mmWave pulsed coherent radar estimates the range of dominant scene reflectors, which is projected into the image plane and fused with RGB-D depth inconsistency cues to generate a coarse spatial prior localizing transparent surface candidates. The prior is then injected into a SegFormer-B2 backbone using a cross-modal RadarAttention module, which modulates deep semantic features with range-conditioned spatial context.

Our main contributions are as follows:

• A radar-guided mask generation pipeline that fuses millimeter-wave range measurements with RGB-D depth inconsistencies to produce coarse but reliable spatial priors for transparent surface localization in the image plane.

• GlassFormer, which injects this prior into a SegFormer-B2 backbone via cross-modal RadarAttention, achieving substantial gains over state-of-the-art methods under low-light and high-glare conditions where vision-only approaches degrade.

• Synchronized RGB-D-radar dataset for transparent surface segmentation comprising annotated frames across diverse scenes, including low-light and high-glare conditions underrepresented in existing benchmarks.

• An open-source ROS2 driver enabling real-time integration of pulsed coherent radar within robotic perception pipelines.

## II. RELATED WORK

Reliable perception of transparent surfaces is critical for safe robotic operation. This section summarizes prior work on transparent surface detection, including vision-based segmentation methods and multimodal sensing approaches that combine complementary sensors.

## A. Vision-Based Transparent Surface Segmentation

Early work on transparent surface segmentation works were done using only RGB images. [9] were the first to introduce a dedicated glass detection network, using largefield contextual feature integration to capture global cues.

Although these methods give good results, RGB cannot provide good cues for glass objects. Subsequent methods incorporated glass-specific physical priors to address this. [2] exploit reflections as a refinement signal, while [1] focus on boundary learning via a refined differential module and edge-aware graph convolution. GlassSemNet [3] takes a different approach, reasoning over semantic co-occurrence patterns to provide contextual priors through a dual-backbone architecture combining SegFormer with a semantic ResNet branch. Progressive Glass Segmentation [10] further improves segmentation through coarse-to-fine refinement, while PanoGlassNet [11] exploits panoramic RGB and intensity images for multimodal glass detection in large-scale environments.

TransLab [12] encodes boundary information directly into the transformer pipeline, and Trans4Trans [13] targets realworld navigation assistance for visually impaired users. These models still assume good lighting and weak specular reflection.

## B. Multimodal Perception for Transparent Surfaces

The limitations of optical sensing have motivated a range of multimodal approaches. RGB-D cameras are the most common pairing: several works fuse depth with RGB features via cross-modal attention [14]–[16] or weighted feature fusion [17] to exploit the inconsistency between depth failures and visual context. However, these methods rely on depth, which fails on transparent surfaces.

To overcome this, researchers have turned to sensors whose physical interaction with glass differs fundamentally from optical devices. LiDAR-based approaches exploit intensity variance [18], [19] and multi-echo processing [20], [21] but are expensive and typically limited to 2D. Ultrasonic sensors can confirm transparent surfaces at short ranges and have been fused with RGB-D cameras for object reconstruction [22], but they have a very small spatial resolution and are extremely sensitive to noise. Polarization cameras capture the rotation of light waves off specular surfaces and have been combined with both deep learning [4] and LiDAR [23] for glass detection, but require expensive and specialised hardware that make them impractical for robotic applications. Thermal imaging provides illumination-invariant cues and has been fused with RGB for segmentation [24].

Most closely related to our work is the radar-RGB-D fusion framework of FuseGrasp [25], which uses radar sensing to assist robotic grasping of transparent objects. However, FuseGrasp relies on higher-dimensional radar processing and is designed for manipulation rather than real-time scene understanding. In contrast, our approach uses a 1D radar signal to derive a region prior that can be fused with RGB observations for transparent surface segmentation, enabling real-time operation on low-compute platforms. [26] designed a ToF-ultrasonic fusion system, which demonstrates realtime transparent obstacle mapping on a low-SWaP aerial platform using only CPU inference, but this functions for a very limited angular range. These works establish that nonoptical range sensors provide complementary and reliable cues where traditional vision-based methods fail.

## III. PRELIMINARIES

## A. Pulsed Coherent Radar and Signal Model

We use a 60.5 GHz millimeter-wave pulsed coherent radar to detect transparent surfaces that are often invisible to optical depth sensors. The radar estimates target range by transmitting short electromagnetic pulses and measuring their round-trip propagation delay $t _ { \mathrm { d e l a y } }$ as:

$$
d = \frac { v t _ { \mathrm { d e l a y } } } { 2 } , \qquad v = \frac { c _ { 0 } } { \sqrt { \varepsilon _ { r } } } ,\tag{1}
$$

here, v is the wave speed in the medium, $c _ { 0 }$ is the speed of light in vacuum, and $\varepsilon _ { r }$ is the relative permittivity of the medium. The factor of 2 accounts for the round-trip travel of the pulse.

For each frame, the radar outputs complex in-phase and quadrature (IQ) measurements as:

$$
x [ s , d ] = I [ s , d ] + j Q [ s , d ] ,\tag{2}
$$

where s indexes temporal sweeps and d indexes discrete range bins along the radar line of sight. Each signal encodes a magnitude that indicates reflector strength and phase information that captures sub-wavelength displacement. In our work, we use the magnitude profile across range bins to localize dominant reflecting surfaces. Although glass is optically transparent, it exhibits a relative permittivity of $\varepsilon _ { r } ~ \approx ~ 6 { - } 8$ at millimeter-wave frequencies. The resulting impedance discontinuity at the air-glass interface produces a measurable radar return even when visible and infrared light is transmitted or specularly reflected. Consequently, planar glass surfaces generate consistent amplitude peaks in the radar range profile, whereas structured-light and stereo depth sensors often give invalid or inconsistent measurements at glass boundaries.

## B. Sensor Geometry and Configuration

The radar outputs a one-dimensional range profile per sweep, consisting of discrete range bins corresponding to the reflected amplitude and phase from a specific radial distance. The spacing between bins is determined by the configured step length. In our configuration, 200 range bins are sampled, defining an effective sensing window of approximately 0.30 m to 13.0 m. To improve measurement stability, hardware averaging per sample (HWAAS = 64) is applied, meaning each range bin value is computed from 64 internally averaged pulse measurements to increase signalto-noise ratio. Additionally, three sweeps are aggregated per frame (sweeps per frame = 3), providing short temporal averaging while preserving responsiveness for real-time perception.

The radar integrates an Antenna-in-Package (AiP) with an approximate half-power beamwidth (HPBW) of $4 0 ^ { \circ }$ The beamwidth defines a conical sensing volume centered along the radar boresight within which reflected signals are received with at least half of the peak radiated power. This determines the spatial region projected into the image plane for radar-guided mask generation (Section IV-A).

TABLE I  
RADAR CONFIGURATION FOR DATA ACQUISITION
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Start point</td><td>120</td></tr><tr><td>Number of range points</td><td>200</td></tr><tr><td>Step length</td><td>24</td></tr><tr><td>Sweeps per frame</td><td>3</td></tr><tr><td>HWAAS</td><td>64</td></tr><tr><td>Receiver gain</td><td>13</td></tr><tr><td>Pulse Repetition Frequency (PRF)</td><td>8.7 MHz</td></tr></table>

## IV. METHODOLOGY

## A. Radar-Guided Mask Generation

1) Range Peak Extraction: For each synchronized frame, the radar produces a one-dimensional magnitude profile across discrete range bins (as shown in Fig. 2). The dominant reflector is estimated by selecting the bin with maximum intensity:

$$
r ^ { * } = \arg \operatorname* { m a x } _ { r _ { i } } A ( r _ { i } )\tag{3}
$$

The corresponding radial distance $d _ { r }$ represents the most prominent surface detected along the radar boresight. This peak-selection approach assumes a dominant single-interface reflection; for thicker glass or multipath conditions, distinct front and back surface returns may occur, in which case the selected peak may not correspond exactly to the near airglass interface.

2) Field-of-View Projection: Since the radar provides no angular resolution, spatial localization is performed by projecting the conical radar field of view (FoV) into the RGB image plane. Given the camera focal length $f _ { x }$ and principal point $( c _ { x } , c _ { y } )$ from intrinsic calibration, the radar FoV θ is converted to a pixel width:

$$
w _ { \mathrm { F o V } } = \theta f _ { x }\tag{4}
$$

This defines a region of interest (ROI) centered at $\left( c _ { x } , c _ { y } \right)$ within the RGB image (Fig. 2). Although the radar beam has a conical radiation pattern, we approximate its projection as a rectangular ROI in the image plane for computational simplicity and alignment with pixel grid discretization. All subsequent depth-based reasoning is restricted to this projected FoV. Unlike approaches that assume a frontal robot pose relative to the glass surface, our radar-guided prior is not restricted to perpendicular viewing as the projection remains valid across the full calibrated field-of-view shared between the radar and RGB-D sensor.

3) Depth Consistency Filtering: Within the projected FoV, a pixel $( x , y )$ is labeled as a transparent candidate if the depth sensor does not corroborate the surface detected by the radar:

$$
M ( x , y ) = \mathbf { 1 } \big [ D ( x , y ) = i n v a l i d \lor D ( x , y ) - d _ { r } > \tau \big ] ,\tag{5}
$$

where $D ( x , y )$ denotes the depth measurement from the RGB-D sensor at pixel $( x , y ) , \ d _ { r }$ denotes the dominant radar range obtained via peak extraction from the onedimensional range profile, and $\tau$ is a positive tolerance margin. The first condition captures pixels with invalid or missing depth returns. The second identifies pixels whose measured depth lies sufficiently beyond the radar-confirmed surface, indicating a likely transparent interface between the sensors and background geometry. The tolerance τ serves as a practical cross-modal gating parameter to accommodate residual calibration error, surface thickness, and multi-path effects. An example is shown in Fig. 2. The resulting binary mask is refined using morphological opening and closing with a 5×5 kernel to suppress isolated artifacts and fill small discontinuities, yielding a per-frame transparent-surface prior for the segmentation network described in Section IV-B.

![](images/98df57b5f9e1e27ab8b7e66b44c96107e7c8479d954275c1fd5f20bbc630c71d.jpg)  
Fig. 2. Radar-guided mask generation pipeline. The radar produces a one-dimensional range magnitude profile from which the dominant reflection peak is extracted to estimate the radial distance dr of the nearest surface. This distance corresponds to the most prominent reflector along the radar boresight. Th radar field-of-view (FoV) is then projected into the RGB image plane using camera intrinsics, defining a region of interest centered at the principal point. Within this projected FoV, the radar-derived range is compared against per-pixel depth measurements from the RGB-D sensor. Pixels with invalid depth values or depths significantly larger than the radar-confirmed surface are labeled as transparent candidates, since the depth sensor observes background geometry through the surface while the radar detects the physical interface. The resulting binary mask serves as a radar-derived transparency prior that highlights regions where optical sensing is unreliable.

## B. GlassFormer Architecture

Our model architecture, GlassFormer, builds on SegFormer-B2 [27] with lightweight RadarAttention (RA) modules at the deeper encoder stages (Fig. 3).

1) SegFormer Backbone: SegFormer consists of a hierarchical Mix Transformer (MiT) encoder paired with a lightweight MLP decoder. The encoder processes an input RGB image through four stages of overlapping patch merging and efficient self-attention, while producing multiscale feature maps at resolutions $\textstyle { \frac { H } { 4 } } , { \frac { H } { 8 } } , { \frac { H } { 1 6 } } , { \frac { H } { 3 2 } }$ with channel dimensions {64, 128, 320, 512} for the B2 variant.

The MiT encoder has a Mix-FFN, which embeds a $3 { \times } 3$ depth-wise convolution inside each feed-forward block to give positional information implicitly. This removes the need for fixed positional encodings that degrade when test resolution differs from training resolution. Efficient self-attention further reduces the sequence length of keys and values by a reduction ratio across stages, cutting computational complexity from $\mathcal { O } ( N ^ { 2 } )$ to $\mathcal { O } ( N ^ { 2 } / R )$ and enabling operation at high spatial resolutions.

The MLP decoder unifies the channel dimension across all four stages. Then, the features are upsampled to $\textstyle { \frac { H } { 4 } }$ and concatenated together, after which an MLP layer is adopted to fuse these concatenated features. Finally, another MLP layer takes the fused features to predict a segmentation mask with $\begin{array} { r } { \frac { H } { 4 } \times \frac { W } { 4 } \times N _ { \mathrm { c l s } } } \end{array}$ . This decoder design takes advantage of the large effective receptive field of the transformer encoder, which eliminates the need for heavy context modules (e.g. ASPP [28]) required by CNN-based decoders.

2) RadarAttention Module: The last two MiT stages operate at resolutions $\begin{array} { r } { \mathrm { ~ ( ~ } \frac { H } { 1 6 } \mathrm { ~ \times ~ } \frac { W } { 1 6 } } \end{array}$ and $\frac { H } { 3 2 } \times \frac { W } { 3 2 } )$ , where features are semantically rich but spatially compressed. We insert a RadarAttention (RA) module after each of these stages to incorporate the radar-derived transparency prior at the feature level. Stages 1 and 2 are left unmodified since the cost of computing the attention maps at those resolution outweighs any advantage that the radar might give.

Each RA module takes a backbone feature map and the radar mask, downsamples to $( h , w )$ via bilinear interpolation, and passes through a two-layer convolutional encoder to produce a dense embedding at the same spatial resolution as the feature map. Backbone features from RGB serve as key and value, while radar embeddings provide queries. A learnable spatial gating λ is also used to control how much influence attention maps have on the RGB features. This allows the model to learn to reject false positives that radar might introduce due to noise in the depth estimates and

![](images/97678d8f9e5369ca2c82e235df9498f05876bb6196793f750fb6a6203afb8d71.jpg)  
Fig. 3. GlassFormer architecture. A SegFormer-B2 encoder extracts a 4-scale RGB feature pyramid (H/4 to H/32). At the two lowest-resolution stages, a 2-layer CNN encodes the radar mask into queries, while RGB features serve as keys and values in a RadarAttention (RA) module; a learnable spatial gate λ controls how strongly the radar prior modulates each RGB feature, letting the model down-weight radar noise. The two RA-modulated stages and the two unmodified high-resolution stages are concatenated and decoded by an MLP followed by two 1 × 1 convolutions to produce the segmentation mask.

TABLE II  
COMPARISON WITH STATE-OF-THE-ART GLASS SEGMENTATION METHODS IN WELL-LIT CONDITIONS
<table><tr><td>Method</td><td>mIoU ↑</td><td>MAE ↓</td><td>F-measure ↑</td><td>BER↓</td></tr><tr><td>GDNet [9]</td><td>0.5485</td><td>0.2894</td><td>0.6951</td><td>0.2618</td></tr><tr><td>GlassSemNet [3]</td><td>0.6336</td><td>0.1945</td><td>0.7769</td><td></td></tr><tr><td>SegFormer [27]</td><td>0.8799</td><td>0.0938</td><td>0.9361</td><td>0.0517</td></tr><tr><td>Radar Overlay</td><td>0.4687</td><td>0.3474</td><td>0.6382</td><td>0.3348</td></tr><tr><td>Ours (GlassFormer)</td><td>0.8818</td><td>0.0835</td><td>0.9372</td><td>0.0517</td></tr></table>

TABLE III

COMPARISON WITH STATE-OF-THE-ART GLASS SEGMENTATION METHODS IN LOW LIGHT CONDITIONS
<table><tr><td>Method</td><td>mIoU ↑</td><td>MAE↓</td><td>F-measure ↑</td><td>BER↓</td></tr><tr><td>GDNet</td><td>0.4824</td><td>0.4031</td><td>0.682</td><td>0.3981</td></tr><tr><td>GlassSemNet</td><td>0.4623</td><td>0.3620</td><td>0.7010</td><td></td></tr><tr><td>SegFormer</td><td>0.4912</td><td>0.3588</td><td>0.6588</td><td>0.2819</td></tr><tr><td>Radar Overlay</td><td>0.5407</td><td>0.3148</td><td>0.7019</td><td>0.3097</td></tr><tr><td>Ours (GlassFormer)</td><td>0.5904</td><td>0.2895</td><td>0.7424</td><td>0.2406</td></tr></table>

resolution limitation.

$$
Q = W _ { Q } E _ { M } , \quad K = W _ { K } F , \quad V = W _ { V } F ,\tag{6}
$$

$$
F ^ { \prime } = \mathrm { S o f t m a x } \biggl ( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } \biggr ) V \times \lambda + F ,\tag{7}
$$

where $d _ { k }$ is the key dimension, $E _ { M }$ is radar embedding and $F$ is the RGB embedding.

3) Loss Function: To supervise the segmentation output, we employ a combination of binary cross-entropy (BCE) loss and Lovasz hinge loss. The BCE loss´ $\ell _ { \mathrm { b c e } }$ operates at the pixel level and provides stable gradients during training by penalizing misclassified pixels. The Lovasz formulation´ directly optimizes the intersection-over-union objective by sorting prediction errors and computing the gradient of the Lovasz extension of the Jaccard loss. This property´ makes it particularly effective for segmentation tasks with class imbalance or thin structures, such as transparent glass surfaces.

The final training objective combines the two losses with equal weight:

$$
\begin{array} { r } { \mathcal { L } = \ell _ { \mathrm { b c e } } + \ell _ { \mathrm { l o v a s z } } . } \end{array}\tag{8}
$$

This combination encourages both accurate pixel-wise classification and improved region-level overlap, leading to more stable training and better alignment with the evaluation metrics used in our experiments.

# 1474 PA ATNU

RGB

Ground Truth

Radar Overlay

GDNet [9]

GlassSemNet [3]

SegFormer [27]

Ours (GlassFormer)

Fig. 4. Qualitative comparison across scenes (top to bottom: window, occluded glass wall night outdoor, glass door, elevator glass in low light, night outdoor, mirror) against baselines.

## V. EXPERIMENTS AND RESULTS

## A. Implementation Details

All models were implemented in PyTorch and trained on an NVIDIA RTX-4060 laptop GPU with 8 GB VRAM and 16 GB RAM. Inference runs at the speed of 70.6 fps, ensuring real-time performance. The model can also be inferred at 13.2 fps on a computer with Ryzen 9 6900hs CPU with 16 GB RAM.

## B. Dataset

The Acconeer XM125 pulsed coherent radar is aligned with the Intel RealSense D455 using a custom 3D printed mount (shown in Fig. 1). The CAD model of the mount is used to obtain the static transform between the radar and RGB-D camera sensor axes. We tested our proposed method on a self-collected dataset of 1800 synchronised RGB-D radar frames across diverse indoor and outdoor scenes. The dataset includes common cases of glass doors, windows, and full-length glass walls as well as mirrors, tinted glass, and partially occluded glass (as shown in Fig. 4). Though the mirror instances are less numerous as compared to transparent glass cases, given their relative scarcity in the environments sampled.

The data sequences are captured under a range of lighting conditions, distances, and angles. Cross-domain evaluation on existing glass segmentation benchmarks (e.g., Trans10K,

GDD) was not performed, as these datasets do not include synchronized radar data required by our proposed method.

![](images/e15ef0c5ec63d6a512b2a9b2fbac93ffb7d6e2bb5bf86d05c2a2049a47325016.jpg)  
Fig. 5. GlassFormer across viewing angles and distances. Segmentation stays reliable across the radar–RGB-D shared FoV, not just frontal viewing

## C. Effect of Radar Prior

We first evaluate the standalone radar-derived binary mask (Radar Overlay) against the learned baselines. As shown in Tables II and III, the radar-only mask underperforms visionbased methods in well-lit conditions (0.469 mIoU) but is competitive with, and in low light exceeds, several visiononly baselines (0.541 vs. 0.462–0.491 mIoU for GDNet, GlassSemNet, and SegFormer). This reflects complementary failure modes, that is, the radar-derived prior depends on depth-radar geometric consistency rather than visual appearance, so it remains stable regardless of illumination, but is susceptible to radar multipath/interference and to RGB-D depth artifacts under glare or strong reflections. Vision-only methods perform best when lighting and texture are favorable but degrade sharply once these cues disappear, motivating fusion rather than reliance on either modality alone.

## D. Effect of RadarAttention

Incorporating the RadarAttention (RA) yields a modest gain over the fine-tuned SegFormer backbone in well-lit conditions (0.882 vs. 0.880 mIoU), where RGB cues are already near-saturated, but a substantially larger gain in low light (0.590 vs. 0.491 mIoU, BER improves from 0.282 to 0.241). This supports our hypothesis that the radarguided prior contributes most precisely where RGB features are weakest, letting RadarAttention selectively redirect the model toward candidate transparent regions when visual evidence alone is insufficient, rather than uniformly biasing predictions irrespective of condition.

## VI. DISCUSSION AND CONCLUSION

Transparent surfaces present a fundamental challenge for robotic perception due to the limitations of optical sensing. While transformer-based methods demonstrate strong robustness even under challenging lighting conditions, they remain vulnerable to structural ambiguity, glare, and depth sensing failures.

In this work, we demonstrate that millimeter-wave radar provides a complementary geometric signal that can be integrated as a spatial prior within a transformer architecture. Although radar alone is insufficient for dense segmentation, its fusion via RadarAttention consistently improves region overlap and boundary refinement without degrading classification stability. Fig. 4 (last row) shows a representative mirror scene, where GlassFormer is able to segment the reflective surface despite the smaller number of mirror examples in the training set. In particular, the proposed approach improves performance in low-light scenes. Importantly, the proposed approach maintains real-time performance and requires only lightweight modifications to an existing segmentation backbone, making it suitable for deployment on resource-constrained robotic platforms.

However, the use of a single 1D radar introduces inherent spatial sparsity and can produce noisy reflections, which limits precise localization. Similarly, depth-guided masking strategies often misinterpret transparent surfaces as free space due to background depth returns, leading to systematic segmentation errors. By incorporating radar as a geometric prior within the transformer attention mechanism, our method mitigates these limitations by guiding the model toward regions where optical cues alone are unreliable while preserving the strong contextual reasoning of the vision backbone. Finally, while the radar itself supports ranging up to 13 m, the effective operating range of the current sensor setup is limited to approximately 6 m by the depth range of the Intel RealSense D455 used for depth-consistency filtering. This is a sensor-integration limitation rather than a limitation of the radar-guided approach; pairing the radar with a longerrange depth sensor would extend the effective segmentation range which we note as a direction for extending safenavigation range in future deployments.

Future work will extend the proposed framework by exploiting the complex in-phase and quadrature (IQ) radar signal, including phase information rather than only ampli tude and range, which may improve sub-wavelength surface discrimination and reduce ambiguity in the depthconsistency prior;enabling richer geometric reasoning and more accurate transparent surface segmentation. We also plan to explore uncertainty-aware fusion, dynamic radar confidence weighting, and broader evaluation under extreme illumination degradation and adverse weather conditions.

## VII. ACKNOWLEDGEMENT

We acknowledge IHub-Data for supporting this work via project M2-029.

## REFERENCES

[1] H. He, X. Li, G. Cheng, J. Shi, Y. Tong, G. Meng, V. Prinet, and L. Weng, “Enhanced boundary learning for glass-like object segmentation,” 2021. [Online]. Available: https://arxiv.org/abs/2103. 15734

[2] J. Lin, Z. He, and R. W. Lau, “Rich context aggregation with reflection prior for glass surface detection,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 13 410– 13 419.

[3] J. Lin, Y. H. Yeung, and R. W. H. Lau, “Exploiting semantic relations for glass surface detection,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https://openreview.net/ forum?id=WrIrYMCZgbb

[4] H. Mei, B. Dong, W. Dong, J. Yang, S.-H. Baek, F. Heide, P. Peers, X. Wei, and X. Yang, “Glass segmentation using intensity and spectral polarization cues,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12 612–12 621.

[5] Z. Xu and Q. Chen, “Glass segmentation with multi scales and primary prediction guiding,” 2024. [Online]. Available: https: //arxiv.org/abs/2402.08571

[6] G. Chen, J. Yuan, H. Chen, H. Li, and J. Chen, “Enhancing glass segmentation accuracy via boundary-context guided physics-aware modeling,” IEEE Access, vol. 13, pp. 143 210–143 222, 2025.

[7] X. Chen, H. Zhang, Z. Yu, A. Opipari, and O. C. Jenkins, “Clearpose: Large-scale transparent object dataset and benchmark,” 2022. [Online]. Available: https://arxiv.org/abs/2203.03890

[8] J. Kim, M.-H. Jeon, S. Jung, W. Yang, M. Jung, J. Shin, and A. Kim, “Transpose: Large-scale multispectral dataset for transparent object,” 2023. [Online]. Available: https://arxiv.org/abs/2307.05016

[9] H. Mei, X. Yang, Y. Wang, Y. Liu, S. He, Q. Zhang, X. Wei, and R. W. Lau, “Don’t hit me! glass detection in real-world scenes,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6 2020.

[10] L. Yu, H. Mei, W. Dong, Z. Wei, L. Zhu, Y. Wang, and X. Yang, “Progressive glass segmentation,” IEEE Transactions on Image Processing, vol. 31, p. 2920–2933, 2022. [Online]. Available: http://dx.doi.org/10.1109/TIP.2022.3162709

[11] Q. Chang, H. Liao, X. Meng, S. Xu, and Y. Cui, “Panoglassnet: Glass detection with panoramic rgb and intensity images,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1–15, 2024.

[12] E. Xie, W. Wang, W. Wang, P. Sun, H. Xu, D. Liang, and P. Luo, “Segmenting transparent object in the wild with transformer,” 2021. [Online]. Available: https://arxiv.org/abs/2101.08461

[13] J. Zhang, K. Yang, A. Constantinescu, K. Peng, K. Muller, and¨ R. Stiefelhagen, “Trans4trans: Efficient transformer for transparent object segmentation to help visually impaired people navigate in the real world,” 2021. [Online]. Available: https://arxiv.org/abs/2107. 03172

[14] Y. Wan, Q. Zhao, J. Xu, H. Wang, and L. Fang, “Dagnet: Depth-aware glass-like objects segmentation via cross-modal attention,” Journal of Visual Communication and Image Representation, vol. 100, p. 104121, 2024. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S1047320324000762

[15] J. Lin, Y.-H. Yeung, S. Ye, and R. W. H. Lau, “Leveraging rgb-d data with cross-modal context mining for glass surface detection,” 2024. [Online]. Available: https://arxiv.org/abs/2206.11250

[16] A. Costanzino, P. Z. Ramirez, M. Poggi, F. Tosi, S. Mattoccia, and L. D. Stefano, “Learning depth estimation for transparent and mirror surfaces,” 2023. [Online]. Available: https://arxiv.org/abs/2307.15052

[17] H. Lin, Z. Zhu, T. Wang, A. Ioannou, and Y. Huang, “Glass surface segmentation with an rgb-d camera via weighted feature fusion for service robots,” 2025. [Online]. Available: https: //arxiv.org/abs/2508.01639

[18] J. Chae, H. Seo, S. Lee, Y. Park, H.-S. Park, and K.-J. Park, “Pinmap: A cost-efficient algorithm for glass detection and mapping using low-cost 2-d lidar,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–14, 2025.

[19] H. Tibebu, J. Roche, V. De Silva, and A. Kondoz, “Lidar-based glass detection for improved occupancy grid mapping,” Sensors, vol. 21, no. 7, 2021. [Online]. Available: https://www.mdpi.com/1424-8220/ 21/7/2263

[20] S.-W. Yang and C.-C. Wang, “On solving mirror reflection in lidar sensing,” IEEE/ASME Transactions on Mechatronics, vol. 16, no. 2, pp. 255–265, 2011.

[21] R. Koch, S. May, P. Murmann, and A. Nuchter, “Identification ¨ of transparent and specular reflective material in laser scans to discriminate affected measurements for faultless robotic slam,” Robotics and Autonomous Systems, vol. 87, pp. 296–312, 2017.

[Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0921889015302736

[22] M. Ye, Y. Zhang, R. Yang, and D. Manocha, “3d reconstruction in the presence of glasses by acoustic and stereo fusion,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 4885–4893.

[23] E. Yamaguchi, H. Higuchi, A. Yamashita, and H. Asama, “Glass detection using polarization camera and lrf for slam in environment with glass,” in 2020 21st International Conference on Research and Education in Mechatronics (REM), 2020, pp. 1–6.

[24] D. Huo, J. Wang, Y. Qian, and Y.-H. Yang, “Glass segmentation with rgb-thermal image pairs,” IEEE Transactions on Image Processing, vol. 32, pp. 1911–1926, 2023.

[25] H. Deng, T. Xue, and H. Chen, “Fusegrasp: Radar-camera fusion for robotic grasping of transparent objects,” 2025. [Online]. Available: https://arxiv.org/abs/2502.20037

[26] M. Hopkins, V. Murali, V. Kumar, and C. J. Taylor, “Real-time glass detection and reprojection using sensor fusion onboard aerial robots,” 2025. [Online]. Available: https://arxiv.org/abs/2510.06518

[27] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” 2021. [Online]. Available: https: //arxiv.org/abs/2105.15203

[28] M. Yang, K. Yu, C. Zhang, Z. Li, and K. Yang, “Denseaspp for semantic segmentation in street scenes,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.