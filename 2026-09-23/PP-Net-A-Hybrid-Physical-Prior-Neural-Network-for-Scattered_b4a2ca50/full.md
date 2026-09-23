# PP-Net: A Hybrid Physical-Prior Neural Network for Scattered Light Removal in Biomedical Images on Embedded Devices

Yongfei Guo, Tingjin Chu, Mengzhuo Liu, Hongwei Lou, and Yuanhao Gong

Abstract—Scattered light is common in biomedical images, yet its removal remains challenging. The difficulty arises from three aspects: first, aligned scattered-light-free biomedical ground truth is often unavailable; second, scattering is coupled with weak illumination and sensor-induced noise; and third, many learning-based restoration models are computationally expensive for embedded devices in Internet of Medical Things (IoMT) scenarios. To address these issues, this paper proposes PP-Net, a hybrid physical-prior neural network for biomedical scattered light removal. The proposed method consists of three components: DFN-Net suppresses sensor-induced noise, ASAP estimates the scattering map and recovers a physics-based prior map, and GF-Net refines the prior map by fusing it with the denoised observation. To reduce the dependence on paired biomedical ground truth, a progressive synthetic training and cross-domain transfer strategy is developed. Experiments show that the physical-prior branch improves the peak signal-to-noise ratio (PSNR) by up to 1.26 dB on paired synthetic benchmarks. Under joint noise-and-scattering degradation, PP-Net improves PSNR by more than 10.8 dB and the structural similarity index measure (SSIM) by more than 0.62 compared with representative baseline methods. On real W2S biomedical images, the proposed method reduces the average Natural Image Quality Evaluator (NIQE) score by 43.3%. Edge deployment with RKNN conversion and INT8 quantization achieves an average inference latency of approximately 200 ms per 512 × 512 image over 360 test images. These results demonstrate that PP-Net provides an effective and deployable solution for microscopic imaging, endoscopic inspection, and edge-assisted biomedical analysis in IoMT scenarios.

Index Terms—Internet of Medical Things (IoMT), biomedical image restoration, scattered-light removal, physical prior, hybrid neural network, edge deployment, embedded devices.

## I. INTRODUCTION

Scattered light is a common phenomenon in optical imaging. In natural environments, light scattering caused by atmospheric particles, aerosols, water droplets, and other turbid media often produces haze-like degradation, reducing image contrast and obscuring scene details. Similar scattering effects also exist in biomedical imaging, where photons interact with biological tissues, cellular structures, and heterogeneous media during acquisition. Unlike natural scene imaging, however, biomedical imaging is often performed under weak, localized, and safety-constrained illumination, making the captured images more vulnerable to scattering corruption, low signal-tonoise ratios, and sensor-induced noise.

In Internet of Medical Things (IoMT) scenarios, biomedical imaging devices such as endoscopes and microscopes serve as front-end sensing nodes for clinical observation, remote diagnosis, and downstream visual analysis [1]–[4]. Degraded images may blur tissue boundaries, reduce structural visibility, and obscure diagnostically relevant details. Meanwhile, practical IoMT systems require not only accurate restoration but also reduced dependence on paired biomedical ground truth, efficient inference, and feasible deployment on resourceconstrained edge devices. These requirements motivate a restoration method that integrates physical interpretability, deep feature refinement, and edge-oriented efficiency.

## A. From Natural Scattering to Biomedical Imaging

In natural image processing, scattered light is commonly studied in the form of haze or atmospheric scattering. Classical dehazing methods usually rely on an image formation model that relates the observed hazy image to scene radiance, transmission, and atmospheric light. Based on this model, a variety of physical priors have been developed to estimate the scattering component and recover the latent clear image. Among them, the dark channel prior (DCP) has become a representative approach because of its simplicity, interpretability, and strong empirical performance in natural scenes [5].

Although natural image dehazing and biomedical scatteredlight removal share similar physical intuition, the two problems are not identical. In natural scenes, paired hazy and hazefree images can often be synthesized or approximated using outdoor image formation models. In biomedical imaging, however, scattering is usually coupled with tissue morphology, weak illumination, sensor noise, and device-dependent acquisition conditions. More importantly, strictly aligned scatteringfree biomedical ground truth is generally unavailable in real clinical settings. Therefore, methods designed for natural dehazing cannot be directly transferred to biomedical scatteredlight removal without considering the specific degradation characteristics of biomedical images.

## B. Physical Priors for Scattered-Light Removal

Physical-prior-based methods are attractive for scatteredlight removal because they provide interpretable intermediate estimates and do not necessarily require large-scale paired training data. In natural image dehazing, DCP-based methods and their variants estimate scattering-related statistics from local image neighborhoods [5]–[7]. Similar ideas have also inspired biomedical scattered-light removal, where prior-based estimation can provide physically meaningful guidance when clean targets are unavailable [4], [8], [9].

However, conventional priors remain limited in biomedical scenarios. First, fixed-window estimation may cross tissue boundaries and introduce boundary leakage or edge-shifting artifacts. Second, dark-channel statistics can be corrupted by high-frequency sensor noise and heterogeneous tissue textures. Third, biomedical images often contain fine structural details and irregular local transitions, making a single fixed neighborhood insufficient for reliable scattering estimation. These limitations suggest that the prior estimation mechanism should be adapted to local biomedical structures rather than relying on a fixed support window. This motivates the adaptive scatteringprior estimation strategy adopted in this work.

## C. Deep Restoration Under Biomedical Constraints

Deep learning has significantly advanced image restoration, including denoising, deblurring, dehazing, low-light enhancement, and general image reconstruction [10]–[12]. Compared with handcrafted priors, deep models provide stronger nonlinear representation capability and can recover complex local textures from degraded observations. For scattered-light removal, learning-based refinement is particularly useful because a physical prior alone may not fully restore subtle tissue details or suppress residual artifacts.

Nevertheless, fully supervised deep restoration is difficult to apply directly to biomedical scattered-light removal. Most supervised restoration networks require paired degraded and clean images, whereas scattering-free biomedical ground truth is rarely available in real acquisition settings. In addition, models trained on synthetic data may suffer from domain gaps when transferred to real biomedical images. Recent weakly supervised, unsupervised, and cross-domain restoration strategies have attempted to reduce this dependence on paired data [13]– [15]. However, without a stable physical constraint, crossdomain restoration may generate structurally inconsistent or diagnostically unreliable details. These observations motivate a hybrid design that combines physical-prior guidance with deep restoration.

## D. Biomedical Restoration on Embedded Devices

Beyond restoration quality, practical deployment is an important requirement in IoMT-oriented biomedical imaging. Edge-side processing can reduce data transmission, preserve privacy, and support local visual enhancement in bandwidth-constrained or delay-sensitive scenarios. However, high-performance deep restoration networks are often computationally intensive, which limits their deployment on resourceconstrained edge devices. Therefore, biomedical scatteredlight removal methods for IoMT applications should be designed with both restoration accuracy and computational efficiency in mind.

Existing lightweight restoration models improve inference efficiency through compact backbones, efficient convolutions, and deployment-oriented optimization [16], [17]. However, many of them are designed for general natural-image restoration and do not explicitly address biomedical scattering, sensor noise, limited paired supervision, and physical interpretability at the same time. This creates a need for an edge-oriented biomedical restoration framework that is both physically meaningful and computationally practical.

## E. Motivations and Contributions

The above analysis shows that biomedical scattered-light removal in IoMT scenarios requires a framework that can jointly address scattering corruption, sensor-induced noise, limited paired supervision, and edge-side deployment constraints. Physical priors provide interpretability but are sensitive to noise and local structural variations. Deep restoration models provide strong representation capability but usually require paired clean targets and may be difficult to deploy on resource-constrained devices. These challenges motivate the proposed PP-Net, a hybrid physical-prior neural network for biomedical scattered-light removal.

PP-Net follows a progressive network-prior-network design. A Denoising Front-End Network (DFN-Net), implemented using FFDNet [18], first suppresses sensor-induced noise. An Advanced Scattering Adaptive Prior (ASAP) module then estimates the scattering map and recovers a physics-based prior map in a physically interpretable manner. Finally, a Guided Fusion Network (GF-Net) refines the prior map by fusing it with the denoised observation. To reduce the dependence on paired biomedical ground truth, we further develop a progressive synthetic training and cross-domain transfer strategy.

The contributions of this work are summarized as follows:

• We formulate a noise-aware biomedical scattered-light degradation model that decomposes image degradation into tissue scattering and sensor-induced noise.

• We propose PP-Net, a hybrid physical-prior neural network that integrates DFN-Net, ASAP, and GF-Net for biomedical scattered-light removal.

• We develop a progressive synthetic training and crossdomain transfer strategy to reduce the dependence on paired biomedical ground-truth data.

• We implement the proposed method on a resourceconstrained edge device and verify its feasibility for edgeside biomedical image enhancement in IoMT scenarios.

## II. HYBRID PHYSICAL-PRIOR NETWORK FRAMEWORK

This section presents the proposed hybrid pipeline for biomedical scattered-light removal. The framework integrates adaptive physics-based prior estimation, lightweight deep refinement, and progressive synthetic training. Given a degraded biomedical image, the pipeline sequentially estimates a denoised observation, a scattering map, a physics-based prior map, and the final restored image. As shown in Fig. 1, the framework consists of three stages: 1) initial noise suppression using DFN-Net, 2) scattering-map estimation and prior-map recovery using ASAP, and 3) high-fidelity prior-map refinement using GF-Net.

For clarity, the branch composed of ASAP and GF-Net is denoted as PP-Net , representing the physical-prior-guided restoration branch. The complete restoration method with DFN-Net, ASAP, and GF-Net is denoted as PP-Net.

![](images/ad94141a0b613c006e47da7f62404283cc7ac4e9dd8beb53af8968a80f2bd20e.jpg)  
Fig. 1: Overview of the proposed hybrid pipeline for biomedical scattered-light removal. (a) System-level architecture for IoMT edge/server deployment. (b) Three-stage algorithmic data flow. (c) DFN-Net for edge-friendly initial noise suppression. (d) ASAP for scattering-map estimation and structure-preserving prior-map recovery. (e) GF-Net for guided fusion and high-fidelity detail recovery.

## A. System-Level Architecture and Pipeline Overview

To satisfy the low-latency and privacy-preserving requirements of IoMT applications, the proposed framework is designed for flexible deployment on either local edge devices or remote servers. As illustrated in Fig. 1(a), raw biomedical images acquired by devices such as endoscopes and microscopes are first fed into the restoration pipeline. Depending on the application scenario, the pipeline can run on a local RK3588 edge node for edge-side enhancement or on a remote server for collaborative medical analysis.

At the algorithmic level, the proposed framework adopts a cascaded network-prior-network architecture, as shown in Fig. 1(b). Let $F ( x , y )$ denote the degraded biomedical input. The overall restoration process is summarized as follows:

$$
\Big [ F ( x , y ) \Big ] \neg \Big [ T ( x , y ) \Big ] \neg \Big [ S ( x , y ) \Big ] \neg \Big [ U _ { \mathrm { m a p } } ( x , y ) \Big ] \neg \Big [ \hat { U } ( x , y ) \Big ]
$$

Here, $T ( x , y )$ denotes the denoised intermediate image, $S ( x , y )$ denotes the scattering map estimated by ASAP, $U _ { \mathrm { m a p } } ( x , y )$ denotes the physics-based prior map recovered from $T ( x , y )$ and $S ( x , y )$ , and $\hat { U } ( x , y )$ denotes the final restored image generated by GF-Net. Under this formulation, the clean synthetic setting uses $\mathrm { P P - N e t _ { P } }$ for physical-priorguided restoration, whereas the noisy synthetic and biomedical transfer settings use the complete PP-Net with DFN-Net.

The first stage suppresses sensor-induced noise, the second stage estimates the scattering map and recovers a physicsbased prior map, and the third stage refines this prior map using GF-Net. This progressive design improves interpretability and reduces the learning burden imposed on the final refinement stage.

## B. Biomedical Scattered Light Model

To guide the restoration process, we formulate a taskoriented degradation model for biomedical light scattering. In natural image dehazing, the classical atmospheric scattering model is commonly written as

$$
I ( x , y ) = J ( x , y ) t ( x , y ) + A \left( 1 - t ( x , y ) \right) ,\tag{1}
$$

where $I ( x , y )$ is the observed degraded image, $J ( x , y )$ is the latent clear image, $t ( x , y )$ is the transmission map, and A denotes the global atmospheric light.

Although biomedical scattering shares certain similarities with atmospheric haze, its imaging conditions are substantially different. In many biomedical imaging systems, illumination is provided by localized active light sources rather than global ambient illumination. Under this assumption, the scattering process can be reformulated using a scattering map $S ( x , y ) =$ $1 - t ( x , y )$ as

$$
I ( x , y ) = J ( x , y ) \left( 1 - S ( x , y ) \right) + S ( x , y ) .\tag{2}
$$

To model sensor noise in low-light biomedical acquisition, we adopt the following task-oriented degradation model:

$$
F ( x , y ) = \alpha U ( x , y ) \left( 1 - S ( x , y ) \right) + S ( x , y ) + N ( x , y ) ,\tag{3}
$$

where $F ( x , y ) \in [ 0 , 1 ]$ denotes the observed degraded biomedical image, $U ( x , y )$ denotes the latent clean image, $S ( x , y )$

denotes the scattering map, $\alpha > 0$ is a scaling coefficient used for illumination and attenuation compensation, and $N ( x , y )$ denotes sensor noise. For analytical simplicity, $N ( x , y )$ is represented as a generic additive noise term. In the subsequent noisy synthetic training stage, mixed Poisson–Gaussian noise is adopted to approximate practical noise characteristics in biomedical acquisition.

This formulation explicitly decomposes biomedical degradation into additive sensor noise and multiplicative scattering corruption, thereby motivating the progressive restoration strategy adopted in this work.

## C. Initial Noise Suppression via DFN-Net

According to (3), the degraded biomedical image contains both scattering corruption and sensor-induced noise. Since sensor noise may distort local statistics and reduce the reliability of subsequent prior estimation, a dedicated denoising stage is introduced before scattering estimation.

To meet edge-side efficiency requirements, we employ DFN-Net, implemented using FFDNet [18], as a lightweight feed-forward denoising front end. As shown in Fig. 1(c), DFN-Net adopts a fully convolutional structure. The network first applies a $3 \times 3$ convolution followed by a PReLU activation to extract shallow features. The resulting feature maps are then processed by a sequence of dilated convolutional blocks with dilation factors $D \in \{ 1 , 2 , 4 , 8 \}$

The dilated design enlarges the receptive field with limited computational overhead. In addition, a residual learning strategy is adopted to encourage the network to focus on the noise component. After feature reconstruction using a final $3 \times 3$ convolution and PReLU activation, the denoised image $T ( x , y )$ is obtained and passed to the subsequent prior estimation stage.

## D. Advanced Scattering Adaptive Prior via ASAP

After the denoising stage, the intermediate image $T ( x , y )$ can be approximated as

$$
\begin{array} { c } { { T ( x , y ) \approx F ( x , y ) - N ( x , y ) } } \\ { { = \alpha U ( x , y ) \left( 1 - S ( x , y ) \right) + S ( x , y ) , } } \end{array}\tag{4}
$$

which provides a more reliable basis for scattering estimation.

Classical dark channel prior methods usually adopt a fixed local window to estimate transmission-related statistics [5]. However, biomedical images often contain complex structures, fine tissue boundaries, and heterogeneous textures. Under such conditions, fixed-window estimation may introduce boundary leakage and structural artifacts. To address this issue, we propose ASAP, which is derived from a multi-scale adaptive dark-channel estimation mechanism, as illustrated in Fig. 1(d).

Instead of using a single fixed patch, ASAP evaluates multiple candidate windows with different scales and spatial morphologies for each target pixel. Specifically, the method considers multiple window sizes and nine spatial configurations, including full, half, and quarter window patterns. By adaptively selecting the most suitable local neighborhood, ASAP improves the robustness of scattering estimation while preserving local structural boundaries.

Our Prior =  
![](images/0ade0e09c6770c8288c6b5ba91304fcc0695a6d4d815e6bf219114cdedc11a64.jpg)  
(a) original

![](images/66b30b3420662e0ea968d9ce02634e6823825932f95514819a193ce7902c92c7.jpg)  
(b) WinSize=10

![](images/4ec6e2d2eeb531b532cb657d7b2c8360172764a2a7878be6f347a334fd12e6fa.jpg)

![](images/944d08eb53ee32767509bbd3e770d6c84da983c167ae8d1d27b57f561e7e7f8c.jpg)  
(c) WinSize=20

![](images/1aca52dd7b574848c10900d222f4c05457e327714dc8594e8fcecc8e7527d0d4.jpg)  
(d) WinSize=30

![](images/9787afeb8e490bf264b089be6184987d48a74d1597bcecbd45cb00df931eacf0.jpg)  
(e) WinSize=10  
(f) WinSize=20

![](images/20096fb9e3505993e0013e9498776e398dcee71e845bf1986b6e2e247bb929b0.jpg)  
(g) WinSize=30

Fig. 2: Visual comparison of scattering-estimation artifacts. The traditional fixed-window DCP introduces severe edgeshifting artifacts, whereas the proposed ASAP achieves more accurate edge preservation.  
![](images/7ea8fbf967f2df133ffd4dabe2ebc8bfc91dfb6ff7acebe98b530b3494b62015.jpg)  
Fig. 3: Illustration of the nine window morphologies evaluated in the proposed dual-adaptive mechanism. Full, half, and quarter spatial configurations are considered to reduce boundary leakage during scattering estimation.

Based on the optimal local window $W _ { \mathrm { o p t } } ( x , y )$ , the coarse scattering map $S _ { 0 } ( x , y )$ is estimated as

$$
S _ { 0 } ( x , y ) = \operatorname* { m i n } _ { q \in W _ { \mathrm { o p t } } ( x , y ) } \left( \operatorname* { m i n } _ { c \in \{ r , g , b \} } T ^ { c } ( q ) \right) ,\tag{5}
$$

where $T ^ { c } ( q )$ denotes the value of the c-th channel of the denoised image $T ( x , y )$ at pixel location q.

To further refine the scattering map while preserving structural discontinuities, we employ total-variation-guided refinement. The corresponding objective is formulated as

$$
\mathcal { E } ( S ) = \frac { 1 } { 2 } \left\| S - S _ { 0 } \right\| _ { 2 } ^ { 2 } + \lambda \left\| \nabla S \right\| _ { 1 } ,\tag{6}
$$

where ∇ is the gradient operator and λ controls the regularization strength. By minimizing this objective, the refined scattering map $S ( x , y )$ is obtained.

Once $T ( x , y )$ and $S ( x , y )$ are available, the physics-based prior map is recovered as

$$
U _ { \mathrm { m a p } } ( x , y ) = \frac { 1 } { \alpha } \frac { T ( x , y ) - S ( x , y ) } { 1 - S ( x , y ) } .\tag{7}
$$

To keep the recovered prior map within a physically meaningful intensity range and avoid numerical instability, the scaling coefficient α is adaptively determined as

$$
\alpha = \operatorname* { m a x } _ { x , y } \left( \frac { T ( x , y ) - S ( x , y ) } { 1 - S ( x , y ) } \right) .\tag{8}
$$

The resulting prior map removes the dominant scattering component and provides a physically interpretable intermediate representation. GF-Net then refines this prior map by fusing it with the denoised observation.

## E. Guided Fusion via GF-Net

Although the physics-based prior map attenuates the dominant scattering effects and recovers the global structure, it may still be insufficient for restoring subtle local details in complex biomedical tissues. To further improve restoration fidelity, we introduce a deep refinement stage based on GF-Net, as shown in Fig. 1(e).

GF-Net jointly exploits the denoised observation $T ( x , y )$ and the prior map $U _ { \mathrm { m a p } } ( x , y )$ . The two inputs are concatenated along the channel dimension and projected into a highdimensional feature space through a $3 \times 3$ convolution. In this manner, the network can exploit both scattering-suppressed structural guidance and retained local texture cues.

To improve computational efficiency, GF-Net is built upon the nonlinear activation-free block (NAFB). Two operations are particularly important. First, the SimpleGate operation introduces nonlinearity through element-wise interactions between split feature channels. Second, simplified channel attention (SCA) recalibrates channel-wise feature importance using global average pooling and a lightweight 1 × 1 convolution.

At the network level, GF-Net adopts a four-level encoder– decoder architecture with skip connections. The hierarchical design enlarges the receptive field and facilitates multi-scale fusion, while the skip connections help preserve spatial information. In addition, a global residual connection is introduced so that the network learns a residual correction with respect to the prior map and the denoised observation.

Accordingly, the final restored image is expressed as

$$
\hat { U } ( x , y ) = U _ { \mathrm { m a p } } ( x , y ) + F _ { \mathrm { G F } } \big ( T ( x , y ) , U _ { \mathrm { m a p } } ( x , y ) \big ) ,\tag{9}
$$

where $F _ { \mathrm { G F } } ( \cdot )$ represents the nonlinear restoration function learned by GF-Net.

F. Progressive Synthetic Training and Cross-Domain Transfer Strategy

A major challenge in biomedical image restoration is the lack of paired degraded/clean data in real clinical environments. Instead of directly performing supervised training on biomedical images, we adopt a progressive synthetic training strategy and then transfer the trained model to real biomedical images for inference. This design is motivated by two considerations: paired scattering-free biomedical ground truth is generally unavailable, and the denoising front end is required only when scattering and sensor noise coexist.

To match the structure of the proposed framework, the training process is divided into two stages. In the first stage, $\mathrm { P P - N e t _ { P } }$ is trained for physical-prior-guided restoration under scattering-dominated degradation without explicit sensor noise. In the second stage, mixed synthetic noise is introduced to train the complete PP-Net for joint scattered-light removal and noise suppression. After this progressive two-stage training, the trained PP-Net is directly applied to real biomedical images for cross-domain inference and evaluation.

1) Stage I: Synthetic Training of $\ P P - N e t _ { \mathrm { P } } .$ In the first stage, the model is trained on the ITS subset of RESIDE [19] to learn prior-guided restoration from paired synthetic data. Since this stage considers scattering-dominated degradation without explicit sensor noise, only ASAP and GF-Net are used. ASAP estimates the scattering map and recovers a physics-based prior map from the degraded input, and GF-Net refines the prior map to recover the restored image. The resulting physicalprior branch is denoted as PP-Net<sub>P</sub>.

Let $F _ { \mathrm { s y n } }$ denote the synthetic hazy input and $U _ { \mathrm { s y n } }$ denote the corresponding clean target image. The network is optimized using the Charbonnier loss

$$
\mathcal { L } _ { \mathrm { s u p } } = \sqrt { \left\| \hat { U } _ { \mathrm { s y n } } - U _ { \mathrm { s y n } } \right\| _ { 2 } ^ { 2 } + \epsilon ^ { 2 } } ,\tag{10}
$$

where $\hat { U } _ { \mathrm { s y n } }$ is the restored output and ϵ is a small constant for numerical stability. This stage establishes physical-priorguided restoration capability from paired synthetic data.

2) Stage II: Synthetic Training of PP-Net Under Joint Noise-and-Scattering Degradation: Although the first stage provides a useful initialization, practical biomedical image acquisition often involves both scattering degradation and sensor-induced noise. To simulate this condition, we add mixed Poisson–Gaussian noise to the ITS images and train the complete PP-Net pipeline. In this stage, DFN-Net suppresses sensor-induced noise, ASAP estimates the scattering map and recovers a physics-based prior map, and GF-Net refines the prior map by fusing it with the denoised observation. The mixed noise model approximates photon fluctuations and sensor noise commonly observed in biomedical imaging.

The noisy-stage training is optimized using the following supervised objective:

$$
\mathcal { L } _ { \mathrm { s u p } } ^ { \mathrm { n o i s e } } = \sqrt { \left\| \hat { U } _ { \mathrm { s y n } } ^ { \mathrm { n o i s e } } - U _ { \mathrm { s y n } } \right\| _ { 2 } ^ { 2 } + \epsilon ^ { 2 } } ,\tag{11}
$$

where $\hat { U } _ { \mathrm { s y n } } ^ { \mathrm { n o i s e } }$ denotes the restored output under noisy degradation. This stage enables PP-Net to jointly address scattering removal and noise suppression.

3) Cross-Domain Transfer to Real Biomedical Images: After the two-stage training process, the resulting PP-Net is directly transferred to real biomedical images for cross-domain inference and evaluation. Since paired biomedical ground truth is unavailable during training, this step is treated as crossdomain transfer rather than supervised adaptation.

In summary, the proposed training strategy is consistent with the progressive structure of the restoration framework. The first stage establishes prior-guided restoration capability through PP-Net , while the second stage introduces DFN-Net to form the complete PP-Net and improve robustness under joint noise-and-scattering degradation. The final trained PP-Net is then transferred to real biomedical images for inference and evaluation.

## III. EXPERIMENTS

This section evaluates the proposed framework from four aspects. First, we validate the effectiveness of the proposed ASAP module on multiple dehazing benchmarks. Second, we compare the proposed hybrid restoration framework with representative learning-based dehazing methods on paired synthetic datasets. Third, we investigate the robustness of the complete PP-Net under joint noise-and-scattering degradation. Finally, after progressive training on synthetic hazy and noisyhazy data, we directly transfer the trained model to real biomedical images for qualitative analysis and no-reference image quality assessment.

## A. Experimental Setup

Experiments are conducted on both synthetic dehazing benchmarks and real biomedical images. The evaluation is designed to examine the effectiveness of the proposed adaptive prior, the restoration capability of the hybrid framework, its robustness under noisy degradation, and its transferability to real biomedical imaging scenarios.

a) Prior evaluation on multi-domain benchmarks.: To validate the effectiveness of ASAP, we compare it with representative physics-based dehazing methods on five datasets: SOTS, HSTS, I-HAZE, O-HAZE, and D-HAZY. These datasets cover synthetic and real haze conditions, as well as indoor and outdoor scenes, thereby providing a comprehensive evaluation of prior-based scattering estimation.

b) Hybrid pipeline evaluation on paired synthetic datasets.: To evaluate the effectiveness of the proposed physical-prior-guided restoration framework, we conduct experiments on the RESIDE-ITS and RESIDE-6K datasets. These two paired datasets are used to assess whether PP-Net can outperform representative learning-based dehazing models under paired synthetic supervision.

c) Robustness evaluation under noisy degradation.: To simulate the noisy biomedical imaging scenario considered in this work, we construct noisy synthetic data by injecting mixed Poisson–Gaussian noise into hazy images. The Poisson scaling factor and Gaussian noise level are set to $\lambda _ { p } = 3 0$ and $\sigma _ { g } = 0 . 0 5$ , respectively. This setting is designed to assess the robustness of PP-Net when scattering degradation and sensor noise coexist.

d) Cross-domain evaluation on real biomedical images.: To examine cross-domain transferability, we directly apply the trained model to multiple categories of real biomedical images. Since paired scattering-free biomedical ground truth is unavailable and no target-domain fine-tuning is performed, the biomedical experiments are used mainly for qualitative analysis and no-reference image quality assessment.

For synthetic datasets with paired references, PSNR and SSIM are adopted as full-reference evaluation metrics. For real biomedical images, NIQE and BRISQUE are used as noreference quality metrics, where lower values indicate better perceptual quality.

1) Implementation Details: All experiments are conducted on a workstation equipped with four NVIDIA RTX 6000 Ada GPUs and an Intel Core i9-14900K CPU. The proposed model is implemented in PyTorch. Unless otherwise specified, the input patch size is set to $2 5 6 \times 2 5 6$ , the batch size is set to 32, and the Adam optimizer is used for training. The initial learning rate is set to $1 \times 1 0 ^ { - 4 }$ and is updated using the

TABLE I: Quantitative comparison between the proposed ASAP and representative physics-based dehazing methods on five benchmark datasets. Higher PSNR and SSIM indicate better restoration quality. Red, green, and blue denote the best, second best, and third-best results, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">SOTS [19]</td><td colspan="2">HSTS [19]</td><td colspan="2">I-HAZE [38]</td><td colspan="2">O-HAZE [39]</td><td colspan="2">D-HAZY [40]</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td>(ICCV&#x27;09) FVR [20]</td><td>0.7388</td><td>13.2622</td><td>0.7002</td><td>10.7931</td><td>0.4823</td><td>10.0210</td><td>0.2543</td><td>13.6041</td><td>0.8051</td><td>14.3512</td></tr><tr><td>(TPAMI’ 11) DCP [5]</td><td>0.8028</td><td>14.6035</td><td>0.7942</td><td>13.7737</td><td>0.5361</td><td>10.2095</td><td>0.6250</td><td>15.0083</td><td>0.8312</td><td>15.0923</td></tr><tr><td>(TIP&#x27;15) CAP [21]</td><td>0.8129</td><td>18.5728</td><td>0.8391</td><td>19.7211</td><td>0.6602</td><td>13.6274</td><td>0.3247</td><td>15.3071</td><td>0.7264</td><td>13.6844</td></tr><tr><td>(TIP’18) MR [22]</td><td>0.8421</td><td>17.4497</td><td>0.8267</td><td>15.8208</td><td>0.6160</td><td>10.9294</td><td>0.3374</td><td>15.7883</td><td>0.7951</td><td>14.3503</td></tr><tr><td>(TIP’18) CEP [23]</td><td>0.7200</td><td>13.9373</td><td>0.7209</td><td>13.7334</td><td>0.5275</td><td>10.5288</td><td>0.4791</td><td>13.3644</td><td>0.7590</td><td>14.3906</td></tr><tr><td>(TCSVT’20) CC [24]</td><td>0.8703</td><td>18.3954</td><td>0.8523</td><td>17.2227</td><td>0.5116</td><td>9.3274</td><td>0.3636</td><td>15.3048</td><td>0.7870</td><td>15.3652</td></tr><tr><td>(TIP’20) NLBF [25]</td><td>0.5933</td><td>13.9673</td><td>0.6024</td><td>13.1347</td><td>0.4181</td><td>10.5647</td><td>0.2572</td><td>13.7174</td><td>0.7053</td><td>13.9485</td></tr><tr><td>(TIP’23) SLP [26]</td><td>0.8758</td><td>19.8575</td><td>0.8533</td><td>19.2735</td><td>0.6763</td><td>13.2467</td><td>0.6369</td><td>16.1066</td><td>0.8293</td><td>14.3466</td></tr><tr><td>(TPAMI&#x27;23) ROP+ [27]</td><td>0.5924</td><td>11.2562</td><td>0.6016</td><td>13.0116</td><td>0.5086</td><td>15.1343</td><td>0.4225</td><td>13.6476</td><td>0.5160</td><td>11.9830</td></tr><tr><td>(TIP’25) ALSP [28]</td><td>0.7932</td><td>16.8118</td><td>0.8075</td><td>17.2046</td><td>0.5700</td><td>12.2069</td><td>0.3464</td><td>12.9536</td><td>0.4871</td><td>10.4283</td></tr><tr><td>(TMM’26) GLP [7]</td><td>0.8408</td><td>18.5458</td><td>0.7864</td><td>18.7565</td><td>0.5705</td><td>15.3048</td><td>0.3934</td><td>17.1895</td><td>0.7869</td><td>14.4929</td></tr><tr><td>(TIP’26) IHDCP [6]</td><td>0.8941</td><td>20.8982</td><td>0.9122</td><td>21.5930</td><td>0.7619</td><td>16.6717</td><td>0.3945</td><td>15.4133</td><td>0.7292</td><td>13.4291</td></tr><tr><td>(Ours) ASAP</td><td>0.9188</td><td>23.2113</td><td>0.9165</td><td>22.0553</td><td>0.6996</td><td>16.7522</td><td>0.5882</td><td>17.5845</td><td>0.8505</td><td>16.1583</td></tr></table>

CosineAnnealingLR schedule. The total number of training epochs is set to 500.

2) Baseline Methods: To provide comprehensive comparisons, we consider two categories of baseline methods. For evaluating ASAP, we compare it with representative physicsbased dehazing methods, including FVR [20], DCP [5], CAP [21], MR [22], CEP [23], CC [24], NLBF [25], SLP [26], ROP+ [27], ALSP [28], GLP [7], and IHDCP [6]. For evaluating the proposed hybrid framework, we further compare it with representative learning-based dehazing methods, including MSCNN [29], AOD-Net [11], GFN [30], MSBDN [31], PFDN [32], FFA-Net [33], TBN [34], CDVA [35], IDB [36], and MPMF-Net [37]. For the noisy degradation setting, the publicly available models of the compared learning-based methods are directly tested on the same noisy synthetic inputs without additional retraining, unless otherwise specified. For the W2S biomedical evaluation, we additionally compare with DCP [5], HDCP [8], QDCP [9], and Dark [4] as representative prior-based or biomedical image enhancement baselines.

3) Training and Transfer Setting: The experimental protocol is consistent with the progressive synthetic training strategy described in Section II-F. Specifically, PP-Net is first trained on paired synthetic hazy data to learn physical-prior-guided restoration without explicit sensor noise. Then, the complete PP-Net is trained on noisy synthetic data to handle joint noiseand-scattering degradation. After two-stage training, PP-Net is directly applied to real biomedical images for cross-domain inference and evaluation.

## B. Evaluation of the Proposed ASAP

We first evaluate ASAP on five benchmark datasets, including SOTS, HSTS, I-HAZE, O-HAZE, and D-HAZY, by comparing it with representative physics-based dehazing methods.

Table I reports the quantitative comparison. The proposed ASAP achieves the best overall performance on most datasets, particularly on SOTS, HSTS, O-HAZE, and D-HAZY. On I-HAZE, ASAP remains competitive and achieves the highest PSNR. These results indicate that the proposed adaptive prior provides more reliable scattering estimation than conventional handcrafted priors across diverse haze conditions.

Fig. 4 presents representative visual comparisons. Compared with existing prior-based methods, ASAP better preserves structural boundaries, suppresses halo-like artifacts, and restores more natural contrast. These qualitative observations are consistent with the quantitative results in Table I.

## C. Performance of PP-Net<sub>P</sub> on RESIDE-ITS and RESIDE-6K

We next evaluate $\mathrm { P P - N e t _ { P } }$ on RESIDE-ITS and RESIDE-6K to verify whether ASAP and GF-Net can effectively cooperate under paired synthetic supervision.

To further examine the optimization process, Fig. 5 shows the training and validation loss curves of PP-Net on RESIDE-ITS and RESIDE-6K over 500 epochs. The losses decrease rapidly in the early stage and then converge smoothly. The validation curves follow the training curves with a similar decreasing trend, indicating stable optimization and generalization during training.

As shown in Table II, PP-Net achieves the best PSNR and SSIM on both datasets. This result demonstrates that the proposed physical-prior-guided refinement strategy effectively combines scattering-map estimation, prior-map recovery, and learning-based detail reconstruction on paired synthetic data.

Fig. 6 shows representative qualitative examples. Compared with the competing dehazing networks, PP-Net produces clearer structures, more faithful textures, and fewer residual haze artifacts, especially in dense haze regions.

## D. Robustness of PP-Net Under Noisy Degradation

To evaluate robustness under joint haze-and-noise degradation, we further test PP-Net on noisy ITS and noisy RESIDE-6K. This setting corresponds to the second-stage training process, where mixed Poisson–Gaussian noise is introduced to simulate practical biomedical acquisition noise.

![](images/1533db9ef8f1fb6c0c806f676e6deb89a1655d3549200ea33254ab5de72270f3.jpg)  
Fig. 4: Visual comparison of different prior-based dehazing methods on the SOTS, HSTS, I-HAZE, O-HAZE, and D-HAZY datasets. The proposed ASAP preserves clearer boundaries, restores more natural contrast, and suppresses halo-like artifacts across diverse scene types.

![](images/3a8411d4b6291cbfaa09ed115512bcdcb4fa500cee2a915c19df30836b06ad76.jpg)

![](images/9ee87ebea77736d11b9673b46d608b4f15e5c297d55e6ccb8e9c5a489f228ee2.jpg)  
Fig. 5: Training and validation loss curves of PP-Net<sub>P</sub> on RESIDE-ITS and RESIDE-6K under paired synthetic supervision. Both datasets are trained for 500 epochs using the Charbonnier loss.

Table III shows that PP-Net significantly outperforms all compared methods on both datasets. The result verifies the importance of introducing DFN-Net before scattering-map estimation, prior-map recovery, and GF-Net refinement. Frontend noise suppression provides more reliable inputs for ASAPbased scattering-map estimation and GF-Net refinement, improving PP-Net stability when noise and scattering coexist.

Fig. 7 provides representative qualitative comparisons. PP-Net simultaneously removes haze and suppresses noise, producing cleaner structures and more stable visual restoration than the competing methods.

TABLE II: Quantitative comparison between PP-Net and representative learning-based dehazing methods on ITS and RESIDE-6K under the haze-only setting. Higher PSNR/SSIM indicates better quality. Red, green, and blue indicate the best, second-best, and third-best results.
<table><tr><td colspan="4">Noise-Free Haze Case</td></tr><tr><td rowspan="2">Method</td><td>ITS [19]</td><td></td><td>RESIDE-6K [19]</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑ PSNR↑</td></tr><tr><td>(ECCV’16) MSCNN [29]</td><td>0.8342</td><td>19.8443</td><td>0.8262 22.8021</td></tr><tr><td>(ICCV’17) AOD-Net [11]</td><td>0.8164</td><td>20.5132</td><td>0.8554 20.2754</td></tr><tr><td>(CVPR’18) GFN [30]</td><td>0.8802</td><td>22.3023</td><td>0.9053 23.5245</td></tr><tr><td>(CVPR’20) MSBDN [31]</td><td>0.9852</td><td>33.6725</td><td>0.9661 28.5632</td></tr><tr><td>(ECCV’20) PFDN [32]</td><td>0.9761</td><td>32.6802</td><td>0.9621 28.1546</td></tr><tr><td>(AAAI&#x27;21) FFA-Net [33]</td><td>0.9765</td><td>36.3913</td><td>0.9731 29.9623</td></tr><tr><td>(TETCI&#x27;24) TBN [34]</td><td>0.8510</td><td>18.3860</td><td>0.8614 19.0325</td></tr><tr><td>(TIM’25) CDVA [35]</td><td>0.7333</td><td>16.0261</td><td>0.7805 15.7067</td></tr><tr><td>(TITS’25) IDB [36]</td><td>0.6176</td><td>18.8099</td><td>0.6090 20.9200</td></tr><tr><td>(AAAI&#x27;25) MPMF-Net [37]</td><td>0.8421</td><td>20.2619</td><td>0.8352 23.0786</td></tr><tr><td>(Ours) PP-NetP</td><td>0.9901</td><td>37.6547</td><td>0.9792 30.2354</td></tr></table>

## E. No-Reference Evaluation on Real Biomedical Images

Finally, we evaluate PP-Net on the W2S biomedical dataset under different averaging/noise levels, including avg1, avg4, avg16, and avg400. These settings represent different acquisition conditions and are used to examine the robustness of the proposed framework on real biomedical images. Since paired scattering-free ground truth is unavailable, PP-Net is compared with the degraded inputs and representative priorbased baselines using no-reference quality metrics.

![](images/b90824b3cca362961209f68c13ef83efb407d7548695429c497c9bac6361033f.jpg)  
Fig. 6: Visual comparison of different dehazing networks on the ITS and RESIDE-6K datasets. The first and second rows correspond to ITS and RESIDE-6K, respectively. PP-Net<sub>P</sub> produces clearer structures, more faithful textures, and fewer residual haze artifacts.

![](images/68c82fd9a41699255b461865773f673875bffd4adf66e1d79a2030277b740a2a.jpg)  
Fig. 7: Visual comparison under joint haze-and-noise degradation on the ITS and RESIDE-6K datasets. The first and second rows correspond to ITS and RESIDE-6K, respectively. The proposed method effectively suppresses haze and noise while preserving clearer structures and more faithful details.

TABLE III: Quantitative comparison between PP-Net and representative learning-based dehazing methods on ITS and RESIDE-6K under the haze-and-noise setting. Higher PSNR/SSIM indicates better quality. Red, green, and blue indicate the best, second-best, and third-best results.
<table><tr><td colspan="4">Noisy Haze Case</td></tr><tr><td rowspan="2">Method</td><td>ITS [19]</td><td></td><td>RESIDE-6K [19]</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑ PSNR↑</td></tr><tr><td>(ECCV’16) MSCNN [29]</td><td>0.1023</td><td>11.0782</td><td>0.0728 12.1120</td></tr><tr><td>(ICCV’17) AOD-Net [11]</td><td>0.0638</td><td>12.6801</td><td>0.1199 12.4083</td></tr><tr><td>(CVPR’18) GFN [30]</td><td>0.0892</td><td>10.2018</td><td>0.0924 11.0352</td></tr><tr><td>(CVPR’20) MSBDN [31]</td><td>0.1246</td><td>11.3412</td><td>0.1021 11.4012</td></tr><tr><td>(ECCV’20) PFDN [32]</td><td>0.1103</td><td>12.0119</td><td>0.1240 10.2147</td></tr><tr><td>(AAAI&#x27;21) FFA-Net [33]</td><td>0.0423</td><td>10.5311</td><td>0.0928 10.1249</td></tr><tr><td>(TETCI’24) TBN [34]</td><td>0.1089</td><td>14.0227</td><td>0.1994 14.4773</td></tr><tr><td>(TIM’25) CDVA [35]</td><td>0.1023</td><td>11.3123</td><td>0.1732 13.6241</td></tr><tr><td>(TITS’25) IDB [36]</td><td>0.1325</td><td>13.5427</td><td>0.1324 11.4512</td></tr><tr><td>(AAAI&#x27;25) MPMF-Net [37]</td><td>0.0963</td><td>12.3733</td><td>0.1533 11.9649</td></tr><tr><td>(Ours) PP-Net</td><td>0.8093</td><td>24.9645</td><td>0.8236 25.3423</td></tr></table>

NIQE and BRISQUE are adopted as no-reference metrics, where lower values indicate better perceptual quality. As reported in Table IV, PP-Net achieves the best NIQE scores across all four averaging levels, indicating improved naturalness and perceptual quality according to the NIQE criterion. The BRISQUE results are less consistent across different averaging levels, suggesting that different no-reference metrics may emphasize different image statistics. Therefore, the noreference evaluation on real biomedical images is interpreted together with qualitative visual evidence.

Fig. 8 presents representative W2S visual results under avg1, avg4, avg16, and avg400. Compared with the degraded inputs and selected baselines, PP-Net improves structural visibility and preserves finer biomedical details, especially in the highlighted local regions.

## F. Ablation Study

We further conduct ablation experiments to verify the main design choices of the proposed framework. The ablation study includes pipeline-level ablation, prior-design ablation, and training-strategy ablation.

1) Pipeline-Level Ablation: Since the proposed framework consists of DFN-Net, ASAP, and GF-Net, we conduct modulelevel ablation experiments to analyze their individual effects and joint contribution.

TABLE IV: No-reference evaluation on the W2S dataset under different averaging/noise levels. Lower NIQE and BRISQUE values indicate better perceptual quality. Red, green, and blue denote the best, second-best, and third-best results, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="7">(ECCVW&#x27;20) Widefield2SIM [41]</td></tr><tr><td>avg1</td><td></td><td>avg4</td><td></td><td>avg16</td><td></td><td>avg400</td></tr><tr><td></td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓ BRISQUE↓</td></tr><tr><td>INPUT</td><td>15.4081</td><td>41.4164</td><td>13.8807</td><td>37.4247</td><td>11.1614</td><td>31.7543 6.9195</td><td>32.8347</td></tr><tr><td>(TPAMI’11) DCP</td><td>15.3282</td><td>41.3114</td><td>13.7268</td><td>37.1064</td><td>10.9883</td><td>31.1809 6.8975</td><td>33.3567</td></tr><tr><td>(ISBI&#x27;23) HDCP</td><td>12.3006</td><td>41.4668</td><td>13.7673</td><td>37.7673</td><td>11.0959</td><td>31.8450 6.9037</td><td>32.2912</td></tr><tr><td>(IEEE Access&#x27;25) QDCP</td><td>15.3019</td><td>41.4695</td><td>13.7730</td><td>37.5326</td><td>11.0971</td><td>31.8459 6.9012</td><td>32.2662</td></tr><tr><td>(Nature Methods&#x27;25) Dark</td><td>17.8489</td><td>43.2703</td><td>19.9137</td><td>43.5090</td><td>17.1220</td><td>43.4596 8.2914</td><td>35.3332</td></tr><tr><td>(Ours) PP-Net</td><td>5.5820</td><td>40.8959</td><td>6.1972</td><td>44.9863</td><td>6.4928</td><td>46.4840 6.6210</td><td>46.5043</td></tr></table>

![](images/fdec75d9dac48712d9e9f2d0d6386375f6715973567278883d97541a517f87a4.jpg)  
Fig. 8: Qualitative comparison on the W2S dataset under different averaging levels. The values shown in each image denote NIQE/BRISQUE scores, where lower values indicate better perceptual quality. PP-Net improves structural visibility and preserves finer biomedical details across different averaging levels.

We first evaluate the clean synthetic setting on the ITS dataset, where $\mathrm { P P - N e t _ { P } }$ is used without DFN-Net. As shown in Table V, both ASAP and GF-Net contribute to restoration performance, while their combination achieves the best result. This verifies the effectiveness of coupling adaptive prior estimation with GF-Net refinement.

TABLE V: Module-level ablation of PP-Net on the clean ITS dataset, where higher PSNR and SSIM indicate better restoration quality.
<table><tr><td>DFN-Net</td><td>ASAP</td><td>GF-Net</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>x</td><td>x</td><td>√</td><td>16.6585</td><td>0.8107</td></tr><tr><td>x</td><td>√</td><td>x</td><td>19.5059</td><td>0.8415</td></tr><tr><td>x</td><td>√</td><td>√</td><td>37.6547</td><td>0.9901</td></tr></table>

TABLE VI: Module-level ablation of PP-Net on the noisy ITS dataset, where higher PSNR and SSIM indicate better restoration quality.
<table><tr><td>DFN-Net</td><td>ASAP</td><td>GF-Net</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>x</td><td>x</td><td>√</td><td>12.2658</td><td>0.1346</td></tr><tr><td>x</td><td>√</td><td>√</td><td>12.3694</td><td>0.1324</td></tr><tr><td>√</td><td>x</td><td>√</td><td>13.9053</td><td>0.6573</td></tr><tr><td>√</td><td>√</td><td>x</td><td>18.7490</td><td>0.7258</td></tr><tr><td>√</td><td>√</td><td>√</td><td>24.9645</td><td>0.8093</td></tr></table>

We then evaluate the noisy synthetic setting on the noisy ITS dataset, where the complete PP-Net is used. Table VI shows that the complete framework achieves the best performance. In particular, the comparison confirms that DFN-Net is important for stabilizing restoration under noisy degradation, while ASAP and GF-Net provide complementary benefits in prior-map recovery and detail refinement.

2) Prior-Design Ablation: We next replace ASAP with alternative prior formulations while keeping the GF-Net refinement backbone unchanged.

TABLE VII: Ablation study on prior design. The refinement backbone is fixed as GF-Net, and only the prior estimation strategy is changed. Higher PSNR and SSIM indicate better restoration performance.
<table><tr><td>Prior Variant</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td> $\overline { { \mathrm { D C P } + \mathrm { G F } - \mathrm { N e t } } }$ </td><td>0.8234</td><td>17.1612</td></tr><tr><td> $\mathrm { H D C P + G F - N e t }$ </td><td>0.9742</td><td>36.6547</td></tr><tr><td> $\mathrm { Q D C P + G F  – N e t }$ </td><td>0.9832</td><td>33.8157</td></tr><tr><td> $\mathrm { A S A P + G F  – N e t }$ </td><td>0.9901</td><td>37.6547</td></tr></table>

Table VII shows that the proposed adaptive prior provides the most effective guidance among the compared prior variants. This result shows that the performance gain comes from both GF-Net refinement and improved ASAP-based scatteringmap estimation, confirming the complementary roles of physical prior estimation and neural refinement.

3) Training-Strategy Ablation: Finally, we evaluate the proposed progressive synthetic training strategy.

As shown in Table VIII, progressive two-stage training achieves the best overall trade-off between noisy synthetic restoration performance and biomedical transferability. This result validates the effectiveness of progressively bridging synthetic haze removal, noisy degradation handling, and real biomedical image inference.

TABLE VIII: Ablation study on training strategy. Higher PSNR and SSIM indicate better performance on noisy synthetic data, while lower NIQE and BRISQUE indicate better perceptual quality on real biomedical images.
<table><tr><td rowspan="2">Training Strategy</td><td>Noisy ITS</td><td>W2S avg1</td><td></td></tr><tr><td>PSNR↑ SSIM↑</td><td>NIQE↓</td><td>BRISQUE↓</td></tr><tr><td>Clean pretraining</td><td>12.3694 0.1324</td><td>14.4642</td><td>41.9512</td></tr><tr><td>Noisy-only training</td><td>22.4543 0.7821</td><td>9.5643</td><td>41.0512</td></tr><tr><td>Progressive training (ours)</td><td>24.9645 0.8093</td><td>5.5820</td><td>40.8959</td></tr></table>

Overall, the experiments demonstrate the effectiveness of ASAP, the strong restoration performance of $\mathrm { P P - N e t _ { P } }$ and PP-Net, and the necessity of the major design choices. In the next section, we further investigate the practical deployment feasibility of PP-Net on on Embedded Devices.

## IV. PP-NET ON EMBEDDED DEVICES

To further evaluate the practical feasibility of PP-Net on embedded devices, we deploy the trained model on an RK3588- based platform. Rather than treating RK3588 as the only target hardware, we use it as a representative embedded device to examine whether the proposed network can be converted, quantized, and executed efficiently for edge-side biomedical image enhancement.

## A. Model Conversion and Quantization Workflow

The deployment follows a staged model-conversion workflow. The trained PP-Net model is first evaluated in PyTorch with FP32 precision, and the corresponding output is used as the desktop-side reference. The model is then exported to ONNX format and further converted into RKNN format for edge-side inference. The overall deployment route is summarized as follows:

![](images/2d237a263199ee382ca7f50f08e5e7679fc41692cbbbc03cc80fac318a576956.jpg)

After model conversion, INT8 post-training quantization is adopted for edge acceleration. Since biomedical image restoration is sensitive to intensity shifts, contrast distortion, and local structural artifacts, representative W2S images from different averaging levels are used for quantization calibration. This domain-specific calibration helps cover diverse noise levels and intensity distributions, thereby reducing quantizationinduced degradation when the model is applied to real biomedical inputs.

Before final deployment, the converted RKNN model is checked against the original PyTorch implementation to ensure visual consistency. This step is important for biomedical image enhancement because small structural distortions introduced during model conversion or quantization may affect the visibility of local tissue details.

## B. Deployment Configuration and Edge Inference Results

The deployment configuration is summarized in Table IX. The PP-Net model is executed on the RK3588 platform with INT8 quantization enabled. At an input resolution of $5 1 2 \times 5 1 2$ the average single-image inference latency is approximately 200 ms over 360 test images, supporting efficient edge-side biomedical enhancement on the RK3588 platform.

TABLE IX: Deployment configuration and performance of PP-Net on the RK3588 platform.
<table><tr><td>Item</td><td>Configuration</td></tr><tr><td>Deployment model</td><td>PP-Net</td></tr><tr><td>Training framework</td><td>PyTorch / Python</td></tr><tr><td>Deployment framework</td><td>RKNN</td></tr><tr><td>Hardware platform</td><td>TOP-EET iTOP-RK3588</td></tr><tr><td>Processor</td><td>RK3588</td></tr><tr><td>Input resolution</td><td>512 × 512</td></tr><tr><td>Inference mode</td><td>Single-image inference</td></tr><tr><td>Average inference latency per image</td><td>~200 ms over 360 images</td></tr><tr><td>Quantization</td><td>INT8-enabled</td></tr></table>

To further assess deployment reliability, we compare the desktop-side PyTorch output with the RK3588-side output. As shown in Fig. 9, the RK3588 results remain visually close to the PyTorch results under different W2S averaging levels, preserving the main structural details and local tissue textures. This comparison suggests that RKNN conversion and INT8 quantization do not introduce obvious deployment-induced visual artifacts.

## C. Structure Visibility Analysis

In addition to the visual deployment comparison shown in Fig. 9, we further quantify the structural visibility of the RK3588 outputs using a Gaussian-smoothed Tenengrad score. Since W2S biomedical images often contain sensorinduced noise, each image is first lightly smoothed using a $5 \times 5$ Gaussian filter with $\sigma = 0 . 9 5$ , and the Sobel gradient magnitude is then computed. The resulting Tenengrad score is used as a no-reference indicator of local structural sharpness.

Let I denote the input image and $G _ { \sigma }$ denote the $5 \times 5$ Gaussian filter with $\sigma ~ = ~ 0 . 9 5$ . The smoothed image is computed as

$$
I _ { \sigma } = G _ { \sigma } * I ,\tag{12}
$$

where ∗ denotes convolution. The horizontal and vertical Sobel gradients are then obtained by

$$
G _ { x } = S _ { x } * I _ { \sigma } , \qquad G _ { y } = S _ { y } * I _ { \sigma } ,\tag{13}
$$

where $S _ { x }$ and $S _ { y }$ are the Sobel operators along the horizontal and vertical directions, respectively. The Gaussian-smoothed Tenengrad score is defined as

$$
T ( I ) = \frac { 1 } { | \Omega | } \sum _ { ( x , y ) \in \Omega } \left( G _ { x } ^ { 2 } ( x , y ) + G _ { y } ^ { 2 } ( x , y ) \right) ,\tag{14}
$$

where Ω denotes the image domain. A higher T(I) indicates stronger local structural sharpness after Gaussian smoothing.

As shown in Table X, RK3588 PP-Net obtains higher Gaussian-smoothed Tenengrad scores than the degraded inputs at different W2S averaging levels. This result quantitatively

![](images/025fc10eac4781f0a137d606a82711a4188082768736b090fa41df30947135d7.jpg)  
Fig. 9: Representative RK3588 deployment results under different W2S averaging levels. The RK3588 outputs remain visually close to the PyTorch outputs, indicating that RKNN conversion and INT8 quantization preserve the main structural details during edge-side inference.

TABLE X: Structure visibility analysis on W2S images using Gaussian-smoothed Tenengrad scores. Higher values indicate stronger structural sharpness, and all scores are multiplied by 10<sup>4</sup> for readability.
<table><tr><td>W2S Level</td><td>Input</td><td>RK3588 PP-Net</td><td>Improvement</td></tr><tr><td>avg1</td><td>184.30</td><td>191.19</td><td>3.74%</td></tr><tr><td>avg16</td><td>186.73</td><td>208.66</td><td>11.74%</td></tr><tr><td>avg400</td><td>80.48</td><td>97.25</td><td>20.84%</td></tr></table>

supports the visual observation in Fig. 9, indicating that edge-side PP-Net inference improves structural visibility while maintaining practical deployment efficiency.

## V. CONCLUSION

This paper presented PP-Net, an edge-oriented hybrid physical-prior neural network for biomedical scattered-light removal. PP-Net integrates DFN-Net, ASAP, and GF-Net into a progressive network-prior-network pipeline, combining noise suppression, scattering-map estimation, prior-map recovery, and lightweight GF-Net refinement. To reduce the dependence on paired biomedical ground truth, we further developed a progressive synthetic training and cross-domain transfer strategy for real biomedical image inference.

Experiments on multiple benchmark datasets demonstrated the effectiveness of ASAP, the strong restoration performance of $\mathrm { P P - N e t _ { P } }$ , and the robustness of PP-Net under joint noiseand-scattering degradation. Direct transfer to real biomedical images also showed promising visual enhancement and competitive no-reference image quality results. Furthermore, RK3588 deployment with RKNN conversion and INT8 quantization achieved an average inference latency of approximately 200 ms per image over 360 test images, confirming efficient edge-side biomedical enhancement.

Future work will focus on unsupervised biomedical domain adaptation, task-oriented clinical validation, and hardwareaware acceleration for real-time biomedical video enhancement on embedded devices.

## ACKNOWLEDGMENTS

This work was supported in part by National Natural Science Foundation of China under Grant 12471502, Science and Technology Development Plan Project of Jilin Province, China under Grant 20260204053YY, and CAS Hundred Talents Program.

## REFERENCES

[1] A. Ghubaish, T. Salman, M. Zolanvari, D. Unal, A. Al-Ali, and R. Jain, “Recent advances in the internet-of-medical-things (iomt) systems security,” IEEE Internet of Things Journal, vol. 8, no. 11, pp. 8707–8718, 2021.

[2] C. Huang, J. Wang, S. Wang, and Y. Zhang, “Internet of medical things: A systematic review,” Neurocomputing, vol. 557, p. 126719, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0925231223008421

[3] X. Ma, H. Wang, X. Ren, and Y. Ma, “A hybrid attention-based fuzzy pooling network model for locating polyp positions in gastroscopic image in internet of medical things,” IEEE Internet of Things Journal, vol. 12, no. 22, pp. 45 985–45 994, 2025.

[4] R. Cao, Y. Li, Y. Zhou, M. Li, F. Lin, W. Wang, G. Zhang, G. Wang, B. Jin, W. Ren, Y. Sun, Z. Zhao, W. Zhang, J. Sun, Y. Hou, X. Xu, J. Hu, W. Shi, S. Fu, Q. Liang, Y. Lu, C. Li, Y. Zhao, Y. Li, D. Kuang, J. Wu, P. Fei, J. Qu, and P. Xi, “Dark-based optical sectioning assists background removal in fluorescence microscopy,” Nature Methods, vol. 22, no. 6, pp. 1299–1310, 2025. [Online]. Available: https://doi.org/10.1038/s41592-025-02667-6

[5] K. He, J. Sun, and X. Tang, “Single image haze removal using dark channel prior,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 1956–1963.

[6] Y. Liu, T. Li, C. Tan, W. Ren, C. Ancuti, and W. Lin, “Ihdcp: Single image dehazing using inverted haze density correction prior,” IEEE Transactions on Image Processing, vol. 35, pp. 1448–1461, 2026.

[7] L. He, Z. Yi, P. Li, S. Wang, C. Chen, and M. Lu, “Efficient single image dehazing based on gradient line prior,” IEEE Transactions on Multimedia, pp. 1–14, 2026.

[8] Y. Gong, W. Huang, and W. Wu, “Removing scattered light in biomedical images,” in 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), 2023, pp. 1–5.

[9] Y. Gong, “Removing scattered light in biomedical images via total variation guided filter,” IEEE Access, vol. 13, pp. 114 495–114 505, 2025.

[10] Y. Song, Z. He, H. Qian, and X. Du, “Vision transformers for single image dehazing,” IEEE Transactions on Image Processing, vol. 32, pp. 1927–1941, 2023.

[11] B. Li, X. Peng, Z. Wang, J. Xu, and D. Feng, “Aod-net: All-in-one dehazing network,” in 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 4780–4788.

[12] Q. Qin, L. Shui, Y. Zhang, S. Song, and J. Jiang, “Mcrfs-net: single image dehazing based on multi-scale contrastive regularization and frequency selection,” Scientific Reports, vol. 15, no. 1, p. 25501, Jul 2025. [Online]. Available: https://doi.org/10.1038/s41598-025-08690-z

[13] J. Zhang and D. Tao, “Famed-net: A fast and accurate multi-scale endto-end dehazing network,” IEEE Transactions on Image Processing, vol. 29, pp. 72–84, 2020.

[14] Y. Gong, M. Xu, Y. Li, and M. Magno, “Removing scattered light in biomedical images via an unsupervised deep neural network,” in 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology, 2023, pp. 65–66.

[15] Y. Gong, Q. Liu, and W. Lin, “Dsnet: Removing scattered light in biomedical images using a dual stream neural network,” in 2025 4th Asia Conference on Algorithms, Computing and Machine Learning (CACML), 2025, pp. 1–5.

[16] Z. Hajduk, “Reconfigurable fpga implementation of neural networks,” Neurocomputing, vol. 308, pp. 227–234, 2018. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231218305393

[17] M. Qian, Y. Wang, S. Liu, Z. Xu, Z. Ji, M. Chen, H. Wu, and Z. Zhang, “Real time wire rope detection method based on rockchip rk3588,” Scientific Reports, vol. 15, no. 1, p. 30625, Aug 2025. [Online]. Available: https://doi.org/10.1038/s41598-025-16043-z

[18] K. Zhang, W. Zuo, and L. Zhang, “Ffdnet: Toward a fast and flexible solution for cnn-based image denoising,” IEEE Transactions on Image Processing, vol. 27, no. 9, pp. 4608–4622, 2018.

[19] B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang, “Benchmarking single-image dehazing and beyond,” IEEE Transactions on Image Processing, vol. 28, no. 1, pp. 492–505, 2019.

[20] J.-P. Tarel and N. Hautiere, “Fast visibility restoration from a single\` color or gray level image,” in 2009 IEEE 12th International Conference on Computer Vision, 2009, pp. 2201–2208.

[21] Q. Zhu, J. Mai, and L. Shao, “A fast single image haze removal algorithm using color attenuation prior,” IEEE Transactions on Image Processing, vol. 24, no. 11, pp. 3522–3533, 2015.

[22] S. Salazar-Colores, E. Cabal-Yepez, J. M. Ramos-Arreguin, G. Botella, L. M. Ledesma-Carrillo, and S. Ledesma, “A fast image dehazing algorithm using morphological reconstruction,” IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2357–2366, 2019.

[23] T. M. Bui and W. Kim, “Single image dehazing using color ellipsoid prior,” IEEE Transactions on Image Processing, vol. 27, no. 2, pp. 999– 1009, Feb 2018.

[24] S. Kanti Dhara, M. Roy, D. Sen, and P. Kumar Biswas, “Color cast dependent image dehazing via adaptive airlight refinement and nonlinear color balancing,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 5, pp. 2076–2081, 2021.

[25] S. C. Raikwar and S. Tapaswi, “Lower bound on transmission using nonlinear bounding function in single image dehazing,” IEEE Transactions on Image Processing, vol. 29, pp. 4832–4847, 2020.

[26] P. Ling, H. Chen, X. Tan, Y. Jin, and E. Chen, “Single image dehazing using saturation line prior,” IEEE Transactions on Image Processing, vol. 32, pp. 3238–3253, 2023.

[27] J. Liu, R. W. Liu, J. Sun, and T. Zeng, “Rank-one prior: Real-time scene recovery,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 8845–8860, 2023.

[28] L. He, Z. Yi, J. Liu, C. Chen, M. Lu, and Z. Chen, “Alsp+: Fast scene recovery via ambient light similarity prior,” IEEE Transactions on Image Processing, vol. 34, pp. 4470–4484, 2025.

[29] W. Ren, S. Liu, H. Zhang, J. Pan, X. Cao, and M.-H. Yang, “Single image dehazing via multi-scale convolutional neural networks,” in European Conference on Computer Vision, 2016.

[30] W. Ren, L. Ma, J. Zhang, J. Pan, X. Cao, W. Liu, and M.-H. Yang, “Gated fusion network for single image dehazing,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 3253–3261.

[31] D. Hang, P. Jinshan, H. Zhe, L. Xiang, Z. Xinyi, W. Fei, and Y. Ming-Hsuan, “Multi-scale boosted dehazing network with dense feature fusion,” in CVPR, 2020.

[32] J. Dong, Jiangxinand Pan, “Physics-based feature dehazing networks,” in Computer Vision – ECCV 2020. Cham: Springer International Publishing, 2020, pp. 188–204.

[33] X. Qin, Z. Wang, Y. Bai, X. Xie, and H. Jia, “Ffa-net: Feature fusion attention network for single image dehazing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 07, 2020, pp. 11 908–11 915.

[34] X. Wang, X. Chen, W. Ren, Z. Han, H. Fan, Y. Tang, and L. Liu, “Compensation atmospheric scattering model and two-branch network for single image dehazing,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 8, no. 4, pp. 2880–2896, 2024.

[35] Y. Shi, Z. Weng, Y. Lin, C. Shi, X. Guo, X. Yang, and L. Lin, “Scaling up single image dehazing algorithm by cross-data vision alignment

for richer representation learning and beyond,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–9, 2025.

[36] Z. Li, W. Kuang, B. Bhanu, Y. Deng, Y. Chen, and K. Xu, “Lowvisibility scene enhancement by isomorphic dual-branch framework with attention learning,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 5, pp. 7127–7141, 2025.

[37] Y. Wen, T. Gao, J. Zhang, Z. Li, and T. Chen, “Multi-axis prompt and multi-dimension fusion network for all-in-one weather-degraded image restoration,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 8, 2025, pp. 8323–8331.

[38] C. Ancuti, C. O. Ancuti, and R. Timofte, “I-haze: A dehazing benchmark with real hazy and haze-free indoor images,” in Advanced Concepts for Intelligent Vision Systems. Cham: Springer International Publishing, 2018, pp. 620–631.

[39] C. O. Ancuti, C. Ancuti, R. Timofte, and C. De Vleeschouwer, “Ohaze: A dehazing benchmark with real hazy and haze-free outdoor images,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2018, pp. 867–8678.

[40] C. Ancuti, C. O. Ancuti, and C. De Vleeschouwer, “D-hazy: A dataset to evaluate quantitatively dehazing algorithms,” in 2016 IEEE International Conference on Image Processing (ICIP), 2016, pp. 2226–2230.

[41] R. Zhou, M. El Helou, D. Sage, T. Laroche, A. Seitz, and S. Susstrunk,¨ “W2S: Microscopy data with joint denoising and super-resolution for widefield to SIM mapping,” in ECCVW, 2020.