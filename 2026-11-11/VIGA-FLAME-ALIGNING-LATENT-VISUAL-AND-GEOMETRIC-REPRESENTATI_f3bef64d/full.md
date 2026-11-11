# VIGA-FLAME: ALIGNING LATENT VISUAL AND GEOMETRIC REPRESENTATIONS FOR ROBUST 3D HEAD EXPRESSION ESTIMATION

Wen Hao He, Jiacheng Wu, Jie Chen<sup>†</sup>

Department of Computer Science, Hong Kong Baptist University, Hong Kong SAR

## ABSTRACT

Estimating 3D head expression from a single image requires mapping visual appearance to underlying facial geometry. We propose ViGA-FLAME, a two-stage framework that aligns semantic visual features with a structured geometric latent space. First, a Parametric–Geometric Variational Autoencoder (PG-VAE) learns a unified representation of canonical head surfaces and FLAME parameters, supporting both detailed surface reconstruction and interpretable parameter recovery. Supervision on geometry and parameters, together with consistency between the two decoding branches, connects surface detail with parametric structure. Second, a trainable Visual-Geometric Adapter (ViGA) maps features from a pretrained semantic visual encoder into this latent space. Asymmetric attention enables ViGA to align semantic image features from the visual encoder with the structured geometric information represented in the PG-VAE latent space, without altering the original visual stream. By grounding image evidence in this latent prior, ViGA-FLAME enables robust 3D head expression estimation across severe facial deformations and large viewing angles, efficiently improving robustness over competing feed-forward methods. More visualizations can be found via the project page.

Index Terms— Visual–geometric alignment; parametric head modeling; 3D expression estimation; multimodal representation learning

## 1. INTRODUCTION

Recovering 3D head geometry and expression from a single image is challenging because facial appearance entangles identity, expression, pose, and viewpoint, while selfocclusion conceals surface deformations. FLAME [1] provides a compact, animatable representation of these factors, yet reliably inferring its parameters remains difficult under extreme expressions and profile views. Predicting FLAME parameters from images requires visual features that capture appearance cues and a geometric prior for 3D interpretation.

Existing FLAME estimation methods rely on either optimization or feed-forward prediction. Optimization-based approaches minimize landmark[2, 3], photometric [4], and appearance objectives [5] for each input image. Although flexible, these methods incur substantial test-time cost and can be sensitive to initialization, occlusion, and rendering mismatch. Feed-forward methods directly regress parameters or geometry: DECA [6] predicts FLAME parameters and geometric-detail codes, SMIRK [7] and TEASER [8] leverage appearance-aware supervision for expression recovery, and SHeaP [9] learns a 3D morphable-model predictor through reanimation-based self-supervision with 3DMMrigged 2D Gaussian rendering. However, these methods typically regress FLAME parameters directly from image features, without first organizing dense geometry and parametric factors in a shared latent representation.

We propose ViGA-FLAME, a feed-forward framework that maps images into a learned parametric-geometric latent space for 3D head reconstruction. First, PG-VAE learns a unified parametric-geometric latent space from canonical FLAME surfaces, jointly supporting dense geometric reconstruction with parametric decoding. Second, ViGA augments a frozen image encoder (Sapiens2 [10]) with a read-only side-branch adapter, whose semantic and geometric tokens extract visual features without modifying the backbone and project them into the PG-VAE latent space. By aligning visual features with this geometric prior, ViGA-FLAME transfers image context to structured prediction, enabling surface reconstruction and FLAME parameter recovery in a single forward pass. Our contributions are threefold: (1) PG-VAE, which unifies dense geometry and parametric FLAME representations; (2) ViGA, a read-only adapter that aligns visual features with the geometric latent space; and (3) extensive experiments demonstrating improved accuracy and robustness across shape, expression, pose, and viewpoint variations.

## 2. PROPOSED METHOD

As shown in Fig. 1, our framework first learns a unified parametric-geometric prior from canonical FLAME surfaces using PG-VAE. ViGA then predicts the corresponding latent representation from a single image by aligning visual features with the learned latent space, enabling decoding into either FLAME parameters or dense surface vertices.

![](images/c8d0bb9ece28449f1fa26042112ff55b4756532735dccdba3157759128297814.jpg)  
Fig. 1. Method overview. We first introduce PG-VAE (Sec. 2.1) to learn a unified geometric representation from canonical FLAME surfaces. We then align its frozen representation with image features using the ViGA-FLAME, described in Sec. 2.2.

## 2.1. PG-VAE: Unified Geometry and FLAME Representation

Parametric head models [1] provide interpretable control but constrain geometric expressivity, whereas non-parametric surfaces preserve detail without semantic factorization. PG-VAE aims to combine both by learning a unified latent representation with geometric tokens for dense surface reconstruction and semantic tokens for FLAME parameter recovery.

Model Architecture. PG-VAE learns a joint parametricgeometric tokenized representation from a canonical, rootcentered FLAME mesh $\mathcal { M } = \left( V , E \right)$ , where $V ~ \in ~ \mathbb { R } ^ { N \times 3 }$ contains N = 5023 vertex coordinates and E denotes the fixed FLAME face connectivity. Fourier features and linear projections encode the vertex coordinates V into point features $P ~ \in ~ \mathbb { R } ^ { N \times d _ { e } }$ , where $d _ { e }$ is the encoder channel size. The encoder comprises four parameter-conditioned semantic queries and $N _ { g }$ learnable geometric queries. Specifically, the FLAME parameters are denoted by $\{ \beta , r , \psi , J \}$ , where $\beta$ and ψ represent shape and expression coefficients, r contains the global orientation together with the rotations of the key FLAME joints, and J denotes the corresponding joint positions. Four independent MLPs project these factors into shape, rotation, expression, and position queries, respectively:

$$
\begin{array} { r } { Q ^ { e n c } = [ q _ { \mathrm { s h a p } } , q _ { \mathrm { r o t } } , q _ { \mathrm { e x p } } , q _ { \mathrm { p o s } } , q _ { 1 } ^ { \mathrm { g e o } } , \dots , q _ { N _ { g } } ^ { \mathrm { g e o } } ] ^ { \top } . } \end{array}\tag{1}
$$

The queries aggregate point features through cross-attention and interact through self-attention, yielding a variational token latent Z. PG-VAE uses geometric and parametric decoders. The geometric decoder $D _ { \mathrm { g e o } }$ performs explicit fixed-topology mesh decoding. It uses N learnable vertexindexed queries $Q ^ { \mathrm { d e c } } ~ = ~ [ q _ { 1 } ^ { \mathrm { d e c } } , \dots , q _ { N } ^ { \mathrm { d e c } } ] ^ { \intercal }$ , with one query associated with each fixed FLAME vertex index. These queries cross-attend to the decoded latent tokens and regress the corresponding canonical vertex coordinates, yielding $\hat { V } _ { \mathrm { g e o } } = \hat { D _ { \mathrm { g e o } } } ( Z , \bar { Q } ^ { \mathrm { d e c } } ) \in \mathbb { R } ^ { N \times 3 }$ . This explicit ordered-vertex decoding preserves fixed FLAME vertex correspondence while coupling dense geometric reconstruction with interpretable parametric factors.

Training Objectives. Let $V _ { \mathrm { g t } }$ denote the canonical target head surface, $\hat { V } _ { \mathrm { g e o } }$ the output of the geometric decoder, and $\{ \hat { \beta } , \hat { r } , \hat { \psi } , \hat { J } \}$ the predictions of the FLAME-semantic branch. The predicted FLAME parameters are further decoded into a surface, $\hat { V } _ { \mathrm { p a r } } = \mathcal { D } _ { \mathrm { F L A M E } } ( \hat { \beta } , \hat { r } , \hat { \psi } )$ . We optimize PG-VAE using complementary losses that supervise both decoding branches and explicitly align their geometric outputs.

First, the geometric branch is supervised by a vertex-wise Huber loss $\mathcal { L } _ { \mathrm { g e c } }$ between $\hat { V } _ { \mathrm { g e o } }$ and $V _ { \mathrm { g t } }$ . Since all meshes share the FLAME topology, this loss directly compares corresponding vertex coordinates and encourages the geometric tokens to preserve fine-grained surface deformations. Second, ${ \mathcal { L } } _ { \mathrm { { f l a m e } } }$ supervises the FLAME predictions: mean-squared error for shape/expression coefficients, $S O ( 3 )$ geodesic distance for global orientation and joint rotations, and Huber loss for joint positions. This term encourages the semantic tokens to retain explicit control over identity, expression, and articulation. Third, parameter-space supervision alone is insufficient because equal errors in different FLAME coefficients can induce substantially different surface deformations. We therefore decode the predicted parameters into $\hat { V } _ { \mathrm { p a r } }$ and impose a vertex-wise Huber loss $\mathcal { L } _ { \mathrm { p a r - g t } }$ against $V _ { \mathrm { g t } }$ . This decodedsurface loss grounds parametric estimation in canonical 3D geometry rather than solely in coefficient space. Fourth, to couple the two branches, we apply a consistency loss ${ \mathcal { L } } _ { \mathrm { p a r - g e o } }$ between $\hat { V } _ { \mathrm { p a r } }$ and $\hat { V } _ { \mathrm { g e o } } .$ . Finally, the KL divergence ${ \mathcal { L } } _ { \mathrm { K L } }$ regularizes the variational posterior toward a unit Gaussian prior. To summarize, the overall objective of PG-VAE is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P G } } = \mathcal { L } _ { \mathrm { g e o } } + \lambda _ { f } \mathcal { L } _ { \mathrm { f l a m e } } + \lambda _ { g } \mathcal { L } _ { \mathrm { p a r - g t } } + \lambda _ { c } \mathcal { L } _ { \mathrm { p a r - g e o } } + \beta \mathcal { L } _ { \mathrm { K L } } , } \\ { ( 2 ) } \end{array}
$$

where $\lambda _ { f } , \lambda _ { g } , \lambda _ { c }$ and $\beta$ are hyperparameters. We set $\lambda _ { f } =$ $0 . 1 , \lambda _ { g } = 1 , \lambda _ { c } = 0 . 1$ , and adapt β within $[ 1 0 ^ { - 6 } , 3 \times 1 0 ^ { - 3 } ]$

(c)

(h)

(f)

(g)

## 2.2. ViGA: Visual-to-Geometric Adapter

We design a Visual-to-Geometric Adapter (ViGA) to predict the corresponding PG-VAE latent vectors from a single image. ViGA is attached to a pretrained visual Transformer, using Sapiens2 [10] as the backbone. With the backbone fully frozen, ViGA reads intermediate token features through a side branch without modifying the original parameters, token states, or outputs. This asymmetric design preserves the pretrained semantic representation while enabling ViGA tokens to aggregate contextual visual cues for latent alignment.

Model Architecture. ViGA maintains learnable token embeddings as model input, comprising four FLAMEsemantic and $N _ { g }$ geometric tokens. Let $X ^ { l }$ denote the native token sequence of the frozen visual encoder at layer l. ViGA introduces an extra token sequence $Y ^ { l } ~ = ~ [ F ^ { l } ; G ^ { l } ]$ , where $F ^ { l }$ denotes the hidden features of the four FLAME-semantic tokens and $G ^ { l }$ denotes those of the geometric tokens. At each visual-encoder layer, the frozen backbone follows its original forward pass, while ViGA updates only its geometric stream:

$$
X ^ { l + 1 } = \mathcal { B } ^ { l } ( X ^ { l } ) , \qquad Y ^ { l + 1 } = \mathrm { C A } ^ { l } \big ( Y ^ { l } , [ X ^ { l } ; Y ^ { l } ] \big ) ,\tag{3}
$$

where $B ^ { l }$ denotes the frozen backbone self-attention block and $\mathrm { C A } ^ { l }$ denotes cross-attention whose second argument provides both keys and values. Thus, the geometric tokens can read contextual visual features and interact with one another, but the native visual tokens never attend to or are updated by the geometric tokens. This asymmetric design preserves the original visual representation while adding a dedicated pathway for geometric reasoning. At the final layer $L _ { d } ,$ a projection head maps the ViGA token features $\lceil F ^ { L _ { d } } ; G ^ { L _ { d } } \rceil$ ] to the PG-VAE latent space, yielding $\hat { Z } = W _ { z } [ \boldsymbol { F } ^ { \tilde { L _ { d } } } ; { \boldsymbol { G } } ^ { L _ { d } } ]$

Training Objective. The supervision target is the posterior mean produced by the frozen PG-VAE encoder from the canonical target surface, $Z ^ { \star } ~ = ~ \mathrm { s g } [ \mu _ { \mathrm { P G - V A E } } ( V _ { \mathrm { g t } } ) ]$ . We use a group-balanced alignment objective, $\mathcal { L } _ { \mathrm { a l i g n } } = \textstyle { \frac { 1 } { 2 } } \big ( \| \hat { Z } _ { \mathrm { s e m } } -$ $Z _ { \mathrm { s e m } } ^ { \star } \| _ { 2 } ^ { 2 } + \| \hat { Z } _ { \mathrm { g e o } } - Z _ { \mathrm { g e o } } ^ { \star } \| _ { 2 } ^ { 2 } )$ , preventing the more numerous geometric tokens from dominating the four semantic tokens. During training, only the geometric queries, cross-attention blocks, and latent projection head are optimized. At inference, the frozen PG-VAE decoder maps $\bar { \hat { Z } }$ to an ordered geometric reconstruction or interpretable FLAME parameters.

## 2.3. Training Data Synthesis

We generate training pairs by sampling FLAME parameters and rendering corresponding images with Nvdiffrast [12], which provides exact correspondence among images, canonical surfaces, and FLAME parameters. We randomize identity, expression, local pose, texture [13], hair [14], background, and illumination to improve appearance diversity. PG-VAE is trained on FLAME surfaces sampled online, whereas ViGA is trained on 100K rendered images at $7 6 8 \times 1 0 2 4$ resolution.

![](images/8f991e06605380bfd5946e5b0d850ac99c45217354bf88071710f7584bcda092.jpg)  
(a)  
(b)  
(e)  
(d)  
Fig. 2. Qualitative face reconstruction comparisons. Each row shows one example: (a) Input image; (b) frontal-view display; (c) Scan/GT mesh; (d) DECA [6]; (e) SMIRK [7]; (f) TEASER [8]; (g) SHeaP [9]; (h) Our ViGA-FLAME.

## 3. EXPERIMENTS

## 3.1. Implementation Details

We initialize PG-VAE attention layers from Hunyuan3D 2.0 [15] and optimize newly introduced and transferred parameters using AdamW with learning rates of $1 0 ^ { - 4 }$ and $1 0 ^ { - 5 }$ respectively, 500 warm-up steps, a weight decay of 0.01, and a global batch size of 1024. ViGA is trained on 768 × 1024 images using AdamW with a learning rate of $1 0 ^ { - 4 }$ , weight decay 0.05, and a global batch size of 256. Both stages use bfloat16 mixed-precision training on 4 NVIDIA H200 GPUs.

## 3.2. Single-Image FLAME Parameter Estimation

We evaluate whether the learned representation generalizes beyond the FLAME2023 geometry used to train our model. We construct a controlled synthetic benchmark with random FLAME parameters. For each identity, we render five fixed camera views corresponding to frontal $( 0 ^ { \circ } )$ , oblique $( \pm 4 5 ^ { \circ } )$ , and profile $( \pm 9 0 ^ { \circ } )$ observations. We synthesise head variations by sampling FLAME shape, expression, and pose parameters under five regimes: all typical, shape extreme, expression extreme, pose extreme, and all extreme. Each regime contains 50 instances. All factors remain within their typical ranges in all typical, while only the specified factor is extreme in each single-factor regime and all factors are extreme in all extreme. For shape and expression, typical PCA coefficients lie within 2.5σ; extreme shape/expression regimes set at least one leading 20 coefficient between 2.5σ and 3.5σ, while extreme pose regimes use rotations near the physical limits of the neck, jaw, and eyes. Real-world evaluations use RenderMe360 [11]. We compare the reconstruction and parametric outputs of ViGA-FLAME with FLAME-based regressors: DECA [6], SMIRK [7], TEASER [8], and SHeaP [9]. We evaluate only a common face-only surface region. Each predicted surface is aligned to the FLAME ground-truth face mesh using rigid ICP with rotation and translation only. We report the averaged symmetric Chamfer-L1 distance (CD-L1, mm) and the F-score at a 5 mm threshold (F@5).

Table 1. Face-only reconstruction on synthetic FLAME subsets and the RenderMe360 [11] dataset. Results are averaged over all five camera views after rigid ICP alignment. CD-L1 is reported in millimeters. Best results are in bold. ∆ denotes ViGA-FLAME Param. minus the strongest baseline for each metric; negative CD-L1 and positive F@5 indicate improvements.
<table><tr><td>Method</td><td colspan="2">All typical</td><td colspan="2">Shape extreme</td><td colspan="2">Expression extreme</td><td colspan="2">Pose extreme</td><td colspan="2">All extreme</td><td colspan="2">RenderMe360 [11]</td></tr><tr><td></td><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td></tr><tr><td>DECA [6]</td><td>3.181</td><td>0.864</td><td>3.468</td><td>0.822</td><td>3.285</td><td>0.849</td><td>6.460</td><td>0.650</td><td>6.642</td><td>0.608</td><td>3.972</td><td>0.747</td></tr><tr><td>SMIRK [7] TEASER [8]</td><td>3.267</td><td>0.848</td><td>3.489</td><td>0.819</td><td>3.341</td><td>0.835</td><td>6.016</td><td>0.653</td><td>6.260 5.817</td><td>0.618</td><td>4.184 4.492</td><td>0.723</td></tr><tr><td>SHeaP [9]</td><td>3.443 4.269</td><td>0.821 0.823</td><td>3.662 4.988</td><td>0.792 0.769</td><td>3.517 4.786</td><td>0.812</td><td>5.620 8.023</td><td>0.642 0.613</td><td>8.113</td><td>0.615 0.585</td><td>3.754</td><td>0.693 0.780</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.788</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ViGA-FLAME Recon.</td><td>2.693</td><td>0.937</td><td>2.774</td><td>0.923</td><td>2.727</td><td>0.930</td><td>3.581</td><td>0.835</td><td>3.768</td><td>0.798</td><td>4.430</td><td>0.696</td></tr><tr><td>ViGA-FLAME Param.</td><td>2.556</td><td>0.949 +0.085</td><td>2.710</td><td>0.929</td><td>2.659</td><td>0.938</td><td>3.212</td><td>0.868</td><td>3.445</td><td>0.833</td><td>4.074</td><td>0.731</td></tr><tr><td>∆ vs. best baseline</td><td>-0.625</td><td></td><td>-0.758</td><td>+0.107</td><td>-0.626</td><td>+0.089</td><td>-2.408</td><td>+0.215</td><td>-2.372</td><td>+0.215</td><td>+0.320</td><td>-0.049</td></tr></table>

Table 2. Face-only reconstruction across camera viewpoints. Results are averaged across all five synthetic FLAME parameter subsets after rigid ICP alignment. CD-L1 is reported in millimeters. Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Frontal (0°)</td><td colspan="2">Oblique (±45°)</td><td colspan="2">Profile (±90°)</td></tr><tr><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td><td>CD-L1↓</td><td>F@5↑</td></tr><tr><td>DECA [6]</td><td>5.263</td><td>0.756</td><td>4.387</td><td>0.775</td><td>4.499</td><td>0.744</td></tr><tr><td>SMIRK [7]</td><td>4.394</td><td>0.779</td><td>4.373</td><td>0.774</td><td>4.616</td><td>0.723</td></tr><tr><td>TEASER [8]</td><td>4.116</td><td>0.772</td><td>4.407</td><td>0.745</td><td>4.564</td><td>0.710</td></tr><tr><td>SHeaP [9]</td><td>5.642</td><td>0.736</td><td>5.408</td><td>0.743</td><td>6.860</td><td>0.678</td></tr><tr><td>ViGA-FLAME Recon.</td><td>3.019</td><td>0.902</td><td>3.011</td><td>0.896</td><td>3.251</td><td>0.864</td></tr><tr><td>ViGA-FLAME Param.</td><td>2.723</td><td>0.930</td><td>2.820</td><td>0.915</td><td>3.109</td><td>0.879</td></tr></table>

As shown in Table 1, both ViGA-FLAME outputs outperform all baselines across the synthetic regimes, with the parametric branch achieving the strongest results under the most challenging all extreme setting, highlighting robustness when shape, expression, and pose vary jointly. Table 2 further shows that the parametric branch achieves the best reconstruction accuracy at frontal, oblique, and profile viewpoints. On the real-world RenderMe360 dataset, our parametric branch surpasses SMIRK and TEASER, though trailing DECA and SHeaP. This transfer is notable as ViGA-FLAME is trained exclusively on synthetic renders, whereas baselines leverage real-image supervision or self-supervision. The residual gap reflects limited synthetic appearance diversity (e.g., heavy makeup, age-related skin textures), pointing to data augmentation rather than a flaw in our latent-alignment formulation. Moreover, optimization-based methods rely on 2D landmark fitting, whose localization becomes unreliable under oblique and profile views due to severe self-occlusion. We therefore restrict quantitative comparisons to feed-forward methods operating under a matched single-image inference protocol.

## 3.3. Ablation Studies

PG-VAE. We evaluate five variants on 8k randomly sampled FLAME parameters. We report mean corresponding-vertex error distance in millimeters for the topology-aware reconstruction branch (Recon.), the parametric FLAME branch (Param), and the cross-branch discrepancy (Recon.↔Param.). Lower values indicate more accurate geometry or stronger agreement between the two decoding pathways.

Table 3. Ablation studies of PG-VAE variations.
<table><tr><td>Method</td><td>Recon. ↓</td><td>Param. ↓</td><td>Recon.↔Param. ↓</td></tr><tr><td>Base</td><td>1.0281</td><td>0.9472</td><td>1.2707</td></tr><tr><td> $+ \lambda _ { g } \mathcal { L } _ { \mathrm { p a r - g t } }$ </td><td>0.9942</td><td>0.8152</td><td>1.1726</td></tr><tr><td> $+ + \lambda _ { c } \mathcal { L } _ { \mathrm { p a r - g e o } }$ </td><td>1.0347</td><td>0.7643</td><td>1.2344</td></tr><tr><td> $+ + + + \operatorname { C o n d } _ { q u e r y }$ </td><td>0.8647</td><td>0.4975</td><td>0.9713</td></tr><tr><td> $+ + + \thinspace \mathbf { C o n d } _ { a d d }$ </td><td>0.9831</td><td>0.5830</td><td>1.1605</td></tr></table>

Table 3 shows that parametric-surface supervision $\mathcal { L } _ { \mathrm { p a r - g } }$ improves all metrics over the baseline, confirming that direct supervision in canonical vertex space complements coefficient-level FLAME regression. Adding the crossbranch consistency term, $\mathcal { L } _ { \mathrm { p a r - g e o } } ,$ further improves parametric reconstruction, despite slightly degrades the geometric reconstruction and measured branch discrepancy. This trade-off suggests that consistency gradients can perturb the geometric branch before the parametric branch converges.

We compare unconditional learnable semantic embeddings with parameter-conditioned semantic tokens. Replacing semantic query embeddings with factor-specific MLP projections of FLAME factors achieves the best performance on all metrics, outperforming additive fusion. Thus, factorspecific conditioning provides a stronger inductive bias for organizing semantic token slots and aligning parametric decoding with dense geometric reconstruction.

## 4. CONCLUSION

We present ViGA-FLAME, a feed-forward framework aligning a frozen image encoder with PG-VAE’s joint parametricgeometric latent space, enabling dense surface reconstruction and FLAME estimation from a single image. Experiments demonstrate improved robustness under controlled variations in shape, expression, pose, and viewpoint, while showing transfer to real-world head estimation.

## 5. ACKNOWLEDGEMENTS

This research was supported by the Theme-based Research Scheme, Research Grants Council of Hong Kong (T45- 205/21-N), and the Guangdong and Hong Kong Universities “1+1+1” Joint Research Collaboration Scheme (2025- A0505000003).

## 6. REFERENCES

[1] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero, “Learning a model of facial shape and expression from 4d scans.,” ACM Transactions on Graphics, vol. 36, no. 6, pp. 194–1, 2017.

[2] Adrian Bulat and Georgios Tzimiropoulos, “How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks),” in 2017 IEEE international conference on computer vision (ICCV). IEEE, 2017, pp. 1021–1030.

[3] Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays, Fan Zhang, Chuo-Ling Chang, Ming Guang Yong, Juhyun Lee, et al., “Mediapipe: A framework for building perception pipelines,” arXiv preprint arXiv:1906.08172, 2019.

[4] Wojciech Zielonka, Timo Bolkart, and Justus Thies, “Towards metrical reconstruction of human faces,” in European conference on computer vision. Springer, 2022, pp. 250–269.

[5] Shenhan Qian, “Vhap: Versatile head alignment with adaptive appearance priors,” sep 2024.

[6] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart, “Learning an animatable detailed 3d face model from in-the-wild images,” ACM Transactions on Graphics (ToG), vol. 40, no. 4, pp. 1–13, 2021.

[7] George Retsinas, Panagiotis P. Filntisis, Radek Danecek, Victoria F. Abrevaya, Anastasios Roussos, Timo Bolkart, and Petros Maragos, “3d facial expressions through analysis-by-neural-synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2490–2501.

[8] Yunfei Liu, Lei Zhu, Lijian Lin, Ye Zhu, Ailing Zhang, and Yu Li, “Teaser: Token enhanced spatial modeling for expressions reconstruction,” in International Conference on Learning Representations, 2025, vol. 2025, pp. 75290–75309.

[9] Liam Schoneveld, Zhe Chen, Davide Davoli, Jiapeng Tang, Saimon Terazawa, Ko Nishino, and Matthias Nießner, “Sheap: Self-supervised head geometry predictor learned via 2d gaussians,” in 2025 IEEE/CVF

International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 14162–14172.

[10] Rawal Khirodkar, He Wen, Julieta Martinez, Yuan Dong, Zhaoen Su, and Shunsuke Saito, “Sapiens2,” in International Conference on Learning Representations, 2026, vol. 2026, pp. 58484–58507.

[11] Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, et al., “Renderme-360: A large digital asset library and benchmarks towards high-fidelity head avatars,” Advances in Neural Information Processing Systems, vol. 36, pp. 7993–8005, 2023.

[12] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila, “Modular primitives for high-performance differentiable rendering,” ACM Transactions on Graphics (ToG), vol. 39, no. 6, pp. 1–14, 2020.

[13] Haoran Bai, Di Kang, Haoxian Zhang, Jinshan Pan, and Linchao Bao, “Ffhq-uv: Normalized facial uv-texture dataset for 3d face reconstruction,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 362–371.

[14] Radu Alexandru Rosu, Keyu Wu, Yao Feng, Youyi Zheng, and Michael J Black, “Difflocks: Generating 3d hair from a single image using diffusion models,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 10847– 10857.

[15] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al., “Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation,” arXiv preprint arXiv:2501.12202, 2025.