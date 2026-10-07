# LiDAR Resolution Recovery via Foundation-Model-Guided Diffusion

Samed Dogan, Nico Leuze, Alfred Sch˘ ottl¨

Dept. of Electrical Engineering and Information Technology

Munich University of Applied Sciences, 80335 Munich, Germany

Email: samed.dogan@hm.edu

Abstract—High-beam-count LiDAR sensors are costly, yet many perception pipelines require dense angular sampling. Using a pretrained Stable Diffusion model as the backbone, we fine-tune a LiDAR-conditioned depth model with pseudo-depth targets from a 2D foundation model. During training, the LiDAR conditioning is randomly decimated at different beam budgets. We then investigate how much of a LiDAR scan can be recovered from heavily decimated input and characterize performance across the input beam budget. We evaluate against physically held-out real beams on nuScenes and report recovery separately from fit accuracy. Our model yields its largest advantage in very sparse regimes, achieving a δ<sub>1.25</sub> accuracy of 66.8% from 4-beam input where scattered interpolation reaches only 45.1%. A class-stratified error breakdown further reveals that planar surfaces recover first while objects introducing depth discontinuities degrade earliest. Together, these results quantify the recovery/resolution trade-off for foundation-model-guided LiDAR enhancement.

Index Terms—LiDAR, depth completion, diffusion models, sensor resolution

## I. INTRODUCTION

Accurate 3D geometric perception is fundamental to autonomous navigation. Consequently, LiDAR sensing systems have gained widespread adoption in automotive and robotic applications [1]–[3]. However, high-resolution LiDAR sensors carry a steep cost in terms of energy and data rate consumption, despite being the superior choice for 3D scene understanding. Low beam-count sensors offer a realistic trade-<sup>.</sup> off between overall system cost and performance. A recurring practical question is therefore whether a low-beam sensor can be used to approximate the angular fidelity of a costly one. Several recent works address LiDAR generation using diffusion-based frameworks [4]–[8]. These models operate on spherical range images, parameterized by the LiDAR’s elevation and azimuth angles. Even though this formulation is a native representation of sensor geometry, it is confined to the sensor’s own measurement manifold. In contrast, models operating in camera-space can inject dense priors from 2D foundation models, including geometry, semantics, and object boundaries learned at web scale, into the recovery process [9]. Diffusion-based LiDAR upsampling is most often cast as a masked completion or super-resolution task (Sec. II-B), typically relying on high-resolution LiDAR supervision. Building upon the architecture introduced in [9], our approach operates without high-resolution LiDAR supervision and recovers physically held-out beams as dense depth maps in the camera view from sparse LiDAR conditioning under 2D foundation model supervision. Concretely, our contributions are as follows:

• We report a degradation curve of recovery accuracy as a function of input beam budget, evaluated against held-out real beams.

• We provide a class-stratified failure analysis, inspecting where recovery breaks down first under very sparse inputs.

• We inspect the influence of scene prompt description on reconstruction quality, when a large portion of information is missing.

The study is scoped to camera-view depth recovery against real held-out measurements, characterizing the additional properties of image-space diffusion relative to the native spherical domain representation.

## II. RELATED WORK

## A. Generative LiDAR Models

Motivated by the success of diffusion and score-based generative models [10]–[12], a growing body of work models LiDAR scans directly in their native spherical range representation [4]–[8], extending image-domain latent diffusion [13] to the range image parameterized by sensor azimuth and elevation. These models draw their prior entirely from the sensor’s own measurement manifold, which preserves range statistics but ties the model to a single sensor geometry. Our setting instead conditions an image diffusion model on the projected scans and draws its prior from outside the LiDAR distribution. We therefore view our approach as complementary, trading native 360° coverage for access to dense 2D priors.

## B. LiDAR Completion and Super-Resolution

Several range-domain diffusion models address sparse scan completion, but by different mechanisms. R2DM [5] performs completion at sampling time via RePaint-style inpainting [14]. RangeLDM [8] trains a conditional latent diffusion model for upsampling, mapping a sparse range image to a dense one under dense-LiDAR supervision. A separate line of work fuses sparse projected depth maps with registered camera images to regress dense metric depth [15], [16], typically supervised by accumulated LiDAR. Lastly, [17] demonstrates the robustness of diffusion-based dense depth estimation under sparsity within the 2D image domain. Unlike these approaches, we require no dense-LiDAR supervision, train under a continuum of beam budgets, and measure per-beam recovery on physically held-out points in camera-view depth. This enables evaluation of camera-space depth recovery under physically decimated LiDAR conditioning rather than dense-LiDAR supervised upsampling.

## III. METHOD

## A. Background

The camera-space projection and latent construction follow the architecture introduced in [9]. Let $\mathbf { \mathcal { P } } ~ = ~ \{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { N }$ be a LiDAR scan in the sensor frame, where each $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ has a corresponding intensity $r _ { i } .$ Given calibrated camera intrinsics K and LiDAR-to-camera extrinsics $\mathbf { T } _ { C L }$ , the scan is rendered into a conditioning latent $\mathbf { z } _ { c }$ using the projection operator $\Pi \in \mathbb { R } ^ { 2 }$

$$
\mathbf { z } _ { c } = \Pi ( \mathcal { P } ; \mathbf { T } _ { C L } , \mathbf { K } ) \in \mathbb { R } ^ { 2 \times H \times W } \mathrm { ~ . ~ }\tag{1}
$$

Its two channels contain depth and intensity, with occlusion consistency enforced by z-buffering. The conditioning is constructed for all V camera views. Π is applied after the beam decimation described in Sec. III-B, ensuring that the conditioning only contains projections of retained beams. The training objective uses the canonical ϵ-parameterized denoising objective for Latent Diffusion Models (LDMs) [13]. The dense depth target is encoded by a frozen Variational Autoencoder (VAE) [18] into $\mathbf { z } _ { 0 }$ and corrupted with Gaussian noise to obtain $\mathbf { z } _ { t }$ at timestep t. A U-Net [19] $\epsilon _ { \theta }$ is trained to invert this process, conditioned on the max-pooled projection $\mathbf { z } _ { c }$ concatenated with $\mathbf { z } _ { t }$ along the channel dimension:

$$
\mathcal { L } _ { L D M } = \mathbb { E } _ { \mathbf { z } _ { 0 } , \epsilon , t } \Big [ w ( t ) \lVert \epsilon - \epsilon _ { \theta } \left( \left[ \mathbf { z } _ { t } , \mathbf { z } _ { c } \right] , t , c \right) \rVert _ { 2 } ^ { 2 } \Big ] ,\tag{2}
$$

where [·, ·] denotes channel concatenation, c is the text conditioning and w(t) is the SNR-weighted schedule [20].

## B. Randomized Beam Decimation

We treat the native 32-beam nuScenes [21] scan as the dense reference and condition the model on the decimated view. Decimation is performed on the physical LiDAR beams using their native ring indices $\rho _ { i } \in \{ 0 , \ldots , 3 1 \}$ . We refer to the number of retained beams as the beam budget b.

a) Budget Sampling: The beam budget is sampled per scan as $b \sim \mathcal { U } ( \{ b _ { \operatorname* { m i n } } , \dotsc , \dotsc , 3 2 \} )$ ) with $b _ { \operatorname* { m i n } } = 4 .$ , rather than using fixed sparsity levels. Training across the full budget range reduces overfitting to a fixed set of elevation angles and yields a single model evaluable at every budget. The degradation curve in Sec. IV-B therefore reports direct measurements at each budget, rather than interpolation between separately trained models.

b) Ring Selection: We mimic a lower-beam-count sensor by selecting LiDAR rings at a uniform stride over the available ring indices determined by the budget b. This yields two subsets of $\mathcal { P } \colon$ the retained set $\mathcal { P } _ { R } \subseteq \mathcal { P }$ that forms the conditioning given Eq. 1, and the complement $\mathcal { P } _ { H } = \mathcal { P } \backslash \mathcal { P } _ { R } .$ which is withheld for evaluation. The dense supervision target is independent of the beam budget. Consequently, the model is tasked to reconstruct the same dense depth from varying amounts of observed information. Furthermore, the stride phase is randomly sampled during training to prevent the model from memorizing a fixed subset of ring positions. During evaluation, we keep the stride phase fixed to zero.

## C. Evaluation Protocol

Depth recovery is evaluated on the held-out point set $\mathcal { P } _ { H }$ of the nuScenes [21] validation split. For each unseen scene, the procedure in Sec. III-B is applied at a fixed beam budget, yielding the disjoint retained and held-out sets. This enforces separation at two levels: unseen scenes prevent scenelevel memorization, while held-out beams prevent evaluation from reducing to a mere reproduction of the conditioning measurements. Dense per-view relative depth is then aligned to metric scale and shift by least-squares using only the retained points $\mathcal { P } _ { R }$ . The aligned metric depth is scored against the heldout returns at their projected pixel locations. We report the Absolute Mean Relative Error (AbsRel), Root Mean Square Error (RMSE) and $\delta _ { 1 . 2 5 }$ metrics widely adopted in dense depth estimation literature [17], [22]–[24]. As a baseline, the retained projection is densified using nearest-neighbor scattered interpolation over valid retained pixels. This isolates the contribution of the learned prior from geometry recoverable by interpolation alone. We skip the affine transformation for the baselines given their native metric scale.

## IV. EXPERIMENTS

## A. Setup

Training is performed on the nuScenes [21] training split under randomized beam decimation (Sec. III-B) and evaluated on the validation split. The U-Net is initialized with Stable Diffusion 1.5 weights. Dense depth pseudo-labels are generated with Depth Anything 3 [24] using the depth preprocessing procedure described in [9]. All recovery metrics are computed on held-out beams, with per-view scale and shift fit on retained beams only (Sec. III-C). We trace the degradation curve (Fig. 2 left) by evaluating every integer budget $b \in \{ 4 , \dots , 3 2 \}$ . We train for 100 epochs on mixed precision using AdamW with a learning rate of $5 \times 1 0 ^ { - 4 } .$ , weight decay of 0.01 and a minimum learning rate of $5 \times 1 0 ^ { - 5 }$ on a cosine schedule with 1000 warmup steps. We use a batch size of 32 with 6 views per batch on 8 NVIDIA A40 GPUs, resulting in an effective batch size of 1536.

## B. Degradation Curve

TABLE I reports representative budgets, while the full curve appears in Fig. 2 (left). The proposed model outperforms interpolation at every budget, with the largest AbsRel improvement in the sparsest regime, where geometry cannot be recovered by interpolation alone (0.692 vs. 0.260 at b= 4). The gap narrows as beam density increases and interpolation becomes more effective (0.151 vs. 0.271 at $b = 1 6 )$ . Recovery error decreases monotonically as more beams are retained and plateaus at $b = 1 6 \AA$ . The full-density fit metric (b= 32) provides a reference for reconstruction without beam withholding, yielding AbsRel

![](images/2e4abf2c6aa440f83a9031936e617de957fb7fba2e455398bdeb63673ae613b7.jpg)  
Fig. 1. Partial view of the model output in point cloud space across 3 fused views (gray) given 8 input beams (red) compared to ground truth (yellow).

TABLE I  
HELD-OUT BEAM RECOVERY VS. BEAM BUDGET. “NEAREST” APPLIES NEAREST VALID-PIXEL SCATTERED INTERPOLATION. AT b=32 NO BEAMS ARE HELD OUT. THE FIT IS REPORTED AS A FULL-DENSITY REFERENCE.
<table><tr><td>Budget</td><td>Method</td><td>AbsRel↓</td><td>RMSE↓</td><td> $\delta _ { 1 . 2 5 } \uparrow$ </td></tr><tr><td rowspan="3">b=4</td><td>Nearest</td><td>0.692</td><td>12.62</td><td>0.451</td></tr><tr><td>Ours (no scene)</td><td>0.260</td><td>8.67</td><td>0.668</td></tr><tr><td>Ours (scene)</td><td>0.270</td><td>8.85</td><td>0.653</td></tr><tr><td rowspan="3">b=8</td><td>Nearest</td><td>0.311</td><td>8.14</td><td>0.670</td></tr><tr><td>Ours (no scene)</td><td>0.169</td><td>7.34</td><td>0.787</td></tr><tr><td>Ours (scene)</td><td>0.174</td><td>7.53</td><td>0.779</td></tr><tr><td rowspan="3">b=16</td><td>Nearest</td><td>0.271</td><td>7.66</td><td>0.768</td></tr><tr><td>Ours (no scene)</td><td>0.151</td><td>6.85</td><td>0.808</td></tr><tr><td>Ours (scene)</td><td>0.155</td><td>7.00</td><td>0.801</td></tr><tr><td rowspan="2">b=32 (fit)</td><td>Ours (no scene)</td><td>0.147</td><td>7.03</td><td>0.806</td></tr><tr><td>Ours (scene)</td><td>0.151</td><td>7.16</td><td>0.799</td></tr></table>

0.147. A qualitative example of geometric reconstruction with b=8 is given in Fig. 1.

## C. Class-Stratified Recovery

Recovery error is stratified by semantic class and tracked across beam budgets (Fig. 2 right). Recovery is strongly classdependent and primarily driven by scene geometry rather than object category. Large planar surfaces (driveable surface, sidewalk, terrain) are reconstructed most accurately, consistent with the stronger geometric constraints provided by their locally smooth structure. Error increases for classes with dominant vertical extent and sharp depth boundaries (manmade, vegetation, barrier) and is highest for large vehicles (truck, trailer, bus, and construction vehicle). One possible explanation is that thin or irregular objects are more difficult to recover because sparse observations provide limited geometric constraints, leaving multiple depth configurations compatible with the observations. This interpretation is consistent with the observed budget dependence: large-vehicle error decreases rapidly as additional beams are retained, whereas planar surfaces show only marginal improvement. A partial vertical profile may be more compatible with multiple vehicle geometries, leaving unconstrained dimensions ambiguous until additional beams provide stronger geometric cues. Nevertheless, a residual error gap persists across all budgets, suggesting that depth discontinuities remain difficult to recover consistently across semantic classes. Thin and dynamic objects (pedestrian, bicycle, motorcycle) provide too few held-out beams per budget for reliable analysis. We therefore refrain from detailed interpretation but include them for completeness (dashed line). A more thorough analysis of output variance under severe decimation is left for future work.

![](images/59697f62a3545add28a323be76ff699d948ff770b55c48dc050e1702c1d2688f.jpg)

![](images/29fb465350f3662965d5d4f98fa5a9d356d1cdc28f00ab62b78f532d3de82590.jpg)  
Fig. 2. Recovery against input beam budget on nuScenes [21] validation split. Left (overall recovery): the proposed model outperforms nearest-neighbor interpolation at every budget, with performance largely plateauing beyond 16 beams. Right (recovery per semantic group): large-vehicle error (orange) falls steeply as additional beams constrain output geometry. Planar ground (green) shows marginal improvement.

## D. Effect of Scene Prompt

We compare recovery with and without scene-description conditioning to assess whether textual context aids reconstruction under severe beam sparsity. Scene conditioning provides no benefit for geometric recovery and marginally degrades it at every budget. This indicates that depth reconstruction depends primarily on the LiDAR conditioning rather than semantic scene context. Accordingly, the no-scene configuration is used as the primary setting for the reported results.

## V. CONCLUSION

This work investigated camera-space LiDAR depth recovery under physically motivated beam decimation. We trained a single model across the full range of beam budgets and evaluated recovery exclusively on physically held-out LiDAR beams. The model outperforms nearest-neighbor interpolation at every budget, particularly in the sparsest regime $( \delta _ { 1 . 2 5 }$ of 66.8% versus 45.1% at $b = 4 )$ , while recovery largely plateaus beyond b= 16, approaching the performance obtained without decimation. Scene-description conditioning yielded no measurable benefit, suggesting that recovery is driven primarily by the LiDAR conditioning. The class-stratified analysis further shows that recovery quality depends strongly on the geometric structure. These results characterize the practical limits of camera-space depth recovery as LiDAR sampling density decreases and provide a basis for evaluating learned LiDAR enhancement under sensor sparsity.

## ACKNOWLEDGMENTS

The research leading to these results is funded by the German Federal Ministry for Economic Affairs and Energy within the project “NXT GEN AI METHODS – Generative Methoden fur Perzeption, Pr¨ adiktion und Planung” (grant no.¨ 19A23014M).

[1] S. Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, “Pv-rcnn: Point-voxel feature set abstraction for 3d object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 10 529–10 538.

[2] T. Yin, X. Zhou, and P. Krahenbuhl, “Center-based 3d object detection and tracking,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 11 784–11 793.

[3] F. B. Malavazi, R. Guyonneau, J.-B. Fasquel, S. Lagrange, and F. Mercier, “Lidar-only based navigation algorithm for an autonomous agricultural robot,” Computers and Electronics in Agriculture, vol. 154, pp. 71–79, 2018. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0168169918302679

[4] V. Zyrianov, H. Che, Z. Liu, and S. Wang, “Lidardm: Generative lidar simulation in a generated world,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 6055–6062.

[5] K. Nakashima and R. Kurazume, “Lidar data synthesis with denoising diffusion probabilistic models,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 14 724–14 731.

[6] V. Zyrianov, X. Zhu, and S. Wang, “Learning to generate realistic lidar point clouds,” in European Conference on Computer Vision. Springer, 2022, pp. 17–35.

[7] H. Ran, V. Guizilini, and Y. Wang, “Towards realistic scene generation with lidar diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 738– 14 748.

[8] Q. Hu, Z. Zhang, and W. Hu, “Rangeldm: Fast realistic lidar point cloud generation,” in European Conference on Computer Vision. Springer, 2024, pp. 115–135.

[9] S. Dogan, N. Leuze, and A. Sch˘ ottl, “Geometry Without Coordi-¨ nates: LiDAR Diffusion as a 3D Feature Bridge,” arXiv e-prints, p. arXiv:2609.10322, Sep. 2026.

[10] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[11] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International conference on machine learning. pmlr, 2015, pp. 2256– 2265.

[12] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations, 2021.

[13] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[14] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 461–11 471.

[15] F. Ma, G. V. Cavalheiro, and S. Karaman, “Self-supervised sparseto-dense: Self-supervised depth completion from lidar and monocular camera,” in 2019 international conference on robotics and automation (ICRA). IEEE, 2019, pp. 3288–3295.

[16] W. Van Gansbeke, D. Neven, B. De Brabandere, and L. Van Gool, “Sparse and noisy lidar completion with rgb guidance and uncertainty,” in 2019 16th International Conference on Machine Vision Applications (MVA), 2019, pp. 1–6.

[17] M. Gui, J. Schusterbauer, U. Prestel, P. Ma, D. Kotovenko, O. Grebenkova, S. A. Baumann, V. T. Hu, and B. Ommer, “Depthfm: Fast generative monocular depth estimation with flow matching,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 3203–3211.

[18] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” in 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

[19] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[20] T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo, “Efficient diffusion training via min-snr weighting strategy,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 7441–7451.

[21] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in CVPR, 2020.

[22] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9492–9502.

[23] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 3, pp. 1623–1637, 2020.

[24] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” arXiv preprint arXiv:2511.10647, 2025.