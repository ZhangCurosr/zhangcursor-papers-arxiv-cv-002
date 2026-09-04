# Preprocessing Failure and Adversarial Detection in Depthwise-Separable Edge Vision Systems

Jannatul Masruk Mukta<sup>1</sup>, Rifa Sanjida<sup>1</sup>, Adrita Rahman Tory<sup>1</sup>, Md. Saifur Rahman1, and Khondokar Fida Hasan2⋆

Bangladesh University of Business and Technology (BUBT), Mirpur-2, Dhaka-1216, Bangladesh

University of New South Wales (UNSW), ACT 2601, Australia fida.hasan@unsw.edu.au

Abstract. Preprocessing-based defenses are the standard first-line response to adversarial attacks on edge vision systems, requiring no retraining, no architectural changes, and widely recommended as modelagnostic mitigations. Yet the foundational evaluations of these defenses were conducted on residual or Inception-class architectures, not on the depthwise-separable CNNs that dominate edge deployments. This untested assumption leaves a gap in the security evaluation literature. This paper closes that gap by evaluating six preprocessing defenses against adversarial perturbations across both architecture families. Across all perturbation levels and defenses tested, the two depthwise-separable architectures show consistently poor recovery while the residual architecture shows partial recovery; ablation results are consistent with an architectural rather than parametric explanation, though only three architectures and one attack family are evaluated. Crucially, this failure is not merely a negative result. The same output divergence that disqualifies preprocessing as a recovery mechanism reveals a detection opportunity: preprocessing consistently disrupts clean predictions while leaving adversarial predictions largely unchanged, an asymmetry that is directly measurable without retraining or architectural modification. We further show that standard image quality metrics are unreliable proxies for defense effectiveness, a methodological gap in current evaluation practice. A practitioner decision framework is provided for adversarially resilient edge vision deployment.

Keywords: Adversarial Robustness · FGSM Attack · Edge Vision Security · Preprocessing Defenses · Depthwise-Separable CNN · IoT Security · Adversarial Detection

## 1 Introduction

MobileNetV2 [1] and EfficientNetB0 [2] are the architectures of choice for edge vision inference, covering applications such as surveillance, biometric access control, embedded medical screening, and IoT nodes. Both offer accuracy-tocost trade-offs that enable on-device inference on resource-constrained hardware [3], yet both remain fundamentally susceptible to adversarial examples: imperceptible perturbations that induce high-confidence misclassification, formalized through FGSM by Goodfellow et al. [4].

Preprocessing defenses, including Gaussian blurring, JPEG compression, bitdepth reduction, median filtering, and spatial resizing, are the standard first-line response. They require no architectural modification or retraining and are widely recommended as model-agnostic mitigations [5,6,7,8,9]. The foundational evaluations by Guo et al. [5] and the Feature Squeezing framework of Xu et al. [6] established this paradigm’s legitimacy. Those studies, however, evaluated defenses on residual or Inception-class architectures. Whether their findings generalize to depthwise-separable CNNs has never been tested, which is a meaningful gap, because the architectures that carry the assumption are not the architectures being deployed.

This paper closes that gap. Across all six defenses and four perturbation magnitudes, defense success stays below operationally useful levels on both depthwise-separable architectures, while the residual architecture shows partial recovery. Because FGSM is the weakest adversary in the threat landscape, failure here is a conservative lower bound, meaning any stronger attacker yields strictly worse outcomes. The result implies that edge vision systems relying on preprocessing as their primary defense are effectively unprotected.

This failure, however, enables a complementary finding. The output divergence that disqualifies preprocessing as a recovery mechanism constitutes a consistent detection signal. Preprocessing disrupts model outputs on clean inputs substantially while leaving adversarial outputs largely unchanged, an asymmetric response that is directly measurable and exploitable without retraining. We propose redeploying preprocessing from recovery to detection mode, grounded in the empirical evidence from this study and building on the Feature Squeezing detection principle [6].

The contributions are: (1) creating a systematic evaluation of preprocessing defense failure on MobileNetV2 and EfficientNetB0, with architectural causation confirmed via parameter ablation and cross-architecture transferability; (2) statistical disproof that PSNR and SSIM can proxy for defense effectiveness; (3) characterization of the output divergence asymmetry as a practical detection signal on depthwise-separable architectures; and (4) a practitioner decision framework with concrete, cited recommendations for adversarially resilient edge vision deployment.

The remainder of this paper is organized as follows. Section 2 reviews related work on adversarial attacks, preprocessing defenses, and architecture-dependent robustness. Section 3 formalizes the threat model and evaluation metrics. Section 4 describes the experimental setup, architectures, dataset, and defense configurations. Section 5 presents the experimental results across all architectures and defense methods. Section 6 interprets the findings and discusses the detection proposal and its security implications. Section 7 outlines limitations and future work, and Section 8 concludes the paper.

## 2 Related Work

Table 1 summarizes the most directly relevant prior work and the gap this study fills.

Table 1. Summary of related work. No prior study evaluates preprocessing defenses on depthwise-separable architectures with joint quality evaluation or detection analysis.
<table><tr><td>Reference</td><td>Attack</td><td>Architecture</td><td>Defense</td><td>Quality</td><td>Edge Detect</td><td></td></tr><tr><td>Goodfellow et al. [4]</td><td>FGSM</td><td>Various</td><td>None</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Guo et al. [5]</td><td>FGSM, DeepFool</td><td>Inception-v3</td><td>Preprocessing (5)</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Xu et al. [6]</td><td>FGSM, JSMA,</td><td>ResNet, DenseNet</td><td>Feature Squeezing</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Madry et al. [10]</td><td>C&amp;W PGD</td><td>ResNet</td><td>Adv. Training</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Dziugaite et al. [11]</td><td>FGSM</td><td>AlexNet</td><td>JPEG</td><td>Partial</td><td>No</td><td>No</td></tr><tr><td>Meng &amp; Chen [12]</td><td>FGSM, JSMA</td><td>ResNet</td><td>Reconstruction</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Croce &amp; Hein [13]</td><td>AutoAttack</td><td>ResNet, WideResNet</td><td>Various</td><td>No</td><td>No</td><td>No</td></tr><tr><td>This work</td><td>FGSM</td><td>ResNet50, MobV2, EffBO</td><td>6 preprocessing</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

Adversarial attacks. Szegedy et al. [14] established that imperceptible perturbations cause high-confidence misclassification. FGSM [4] remains the standard first-line evaluation; PGD [10], C&W [15], and AutoAttack [13] provide progressively stronger benchmarks via RobustBench [16]. The exclusive use of FGSM here yields upper-bound DSR estimates; categorical failure under the weakest attacker is the most conservative possible security statement.

Preprocessing defenses. Guo et al. [5] demonstrated moderate robustness improvements on Inception-v3; Xu et al. [6] extended the paradigm to detection via Feature Squeezing on ResNet and DenseNet; Dziugaite et al. [11] evaluated JPEG on AlexNet. The implicit assumption across all of this work, that efficacy is architecture-agnostic, has never been tested on depthwise-separable CNNs.

Architecture-dependent robustness. Tsipras et al. [17] established a formal accuracy-robustness trade-off; Shao et al. [18] showed Vision Transformers substantially outperform CNNs under both white-box and black-box attack. Bhojanapalli et al. [19] extended such comparisons across model families. None of this work characterizes the specific behavior of preprocessing defenses on the depthwise-separable family.

Adversarial detection. Xu et al. [6] formalized detection via output inconsistency between raw and preprocessed inputs. Meng and Chen [12] proposed MagNet, using reconstruction error from autoencoder networks. Both frameworks were evaluated exclusively on residual architectures; the present study provides the first evidence that detection signals are stronger on depthwiseseparable CNNs.

## 3 Problem Formulation

## 3.1 Notation and Threat Model

Let $x \in \mathbf { R } ^ { \mathrm { H } \times \mathrm { W } \times \mathrm { C } } \left( H = W = 2 2 4 , C = 3 \right)$ be a clean image normalized to [−1, 1] with label y. The FGSM adversarial example is:

$$
\chi _ { \mathrm { a d v } } = x + \varepsilon \cdot \mathrm { s i g n } ( \nabla _ { \mathrm { x } } \mathrm { L } ( \theta , x , y ) )\tag{1}
$$

A preprocessing defense dϕ is applied prior to inference. The threat model is white-box, untargeted, and non-adaptive: the attacker has full model access but does not know the defense. DSR values are therefore upper bounds; an adaptive adversary [15] or iterative attacker [10] yields strictly lower defense success.

## 3.2 Metrics

Defense Success Rate (DSR, primary security metric):

$$
\mathrm { D S R } = \frac { 1 } { N } \sum _ { \mathrm { i = 1 } } ^ { \mathbb { N } } \mathbf { 1 } [ f _ { \mathrm { \theta } } ( d _ { \mathrm { \phi } } ( x _ { \mathrm { a d v , i } } ) ) = y _ { \mathrm { i } } ]\tag{2}
$$

PSNR [20] and SSIM [21] measure pixel-level and structural fidelity respectively; values above 40 dB PSNR are perceptually lossless. The Composite Score integrates security and fidelity:

$$
\mathrm { C S } = 0 . 5 \cdot \mathrm { D S R } + 0 . 3 \cdot \operatorname* { m i n } \ \frac { \mathrm { P S N R } } { 4 0 } , 1 ^ { \circ } + 0 . 2 \cdot \operatorname* { m a x } ( \mathrm { S S I M } , 0 )\tag{3}
$$

DSR weight 0.5 reflects security primacy; rankings are stable across DSR weights 0.4 and 0.6.

Detection divergence δ formalizes the detection signal from Section 5.5:

$$
\delta = \left. p _ { \Theta } ( x _ { \mathrm { a d v } } ) - p _ { \Theta } ( d _ { \Phi } ( x _ { \mathrm { a d v } } ) ) \right. _ { 1 }\tag{4}
$$

where $p _ { \Theta } ( \cdot )$ is the softmax probability vector. An input is flagged as adversarial when $\delta > \tau ,$ , with τ calibrated on clean validation data. This formalizes the Feature Squeezing detection principle [6]; Section 5.5 establishes empirically that δ is large and consistent for adversarial inputs on depthwise-separable architectures.

![](images/68e787d85d3b0ce7e61f5fde90857d87c25b43c84445c3ce5a77321bacbf0fe2.jpg)  
Fig. 1. Overview of the experimental methodology pipeline.

## 4.1 Architectures and Dataset

Three architectures span the residual-to-depthwise-separable design spectrum. ResNet50 [22] (25.6M parameters) uses residual skip connections for multiscale feature reuse. MobileNetV2 [1] (3.4M parameters) uses inverted residual blocks with depthwise-separable convolutions, decoupling per-channel spatial filtering from cross-channel projection. EfficientNetB0 [2] (5.3M parameters) adds compound scaling and squeeze-and-excitation blocks to the same depthwiseseparable primitive, and despite a higher parameter count than MobileNetV2, achieves 100% FGSM ASR at ε=0.001, the highest vulnerability of the three. All architectures use ImageNet-1K pretrained weights without fine-tuning, evaluated via TensorFlow/Keras 2.19.0 on CPU-based Google Colab. The dataset comprises 500 Tiny ImageNet [23] images (224 × 224, normalized to [−1, 1]) restricted to those correctly classified by all three models under clean conditions. Full specifications are in Table 2. One methodological note: EfficientNetB0’s internal input scaling pipeline dominates PSNR/SSIM regardless of preprocessing applied; pixel differences between dϕ(xadv) and xadv are confirmed non-zero for all defenses (the transforms operate correctly), but the reported constant PSNR of 4.56 dB reflects internal normalization rather than perturbation magnitude. EfficientNetB0 values are therefore excluded from comparative quality analysis.

## 4.2 Defense Configurations

Six model-agnostic preprocessing defenses follow Guo et al. [5]: Gaussian blur (spatial smoothing), JPEG compression (DCT quantization [11]), median filtering (rank-order statistics), bit-depth reduction (intensity quantization [6]), spatial resizing (downscale-upscale), and an ensemble of Gaussian blur followed by JPEG. No defense-specific retraining was performed. Configurations are summarized in Table 3.

Table 2. Experimental configuration and system specifications.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Architectures</td><td>ResNet50 (25.6M), EfficientNetB0 (5.3M), MobileNetV2 (3.4M)</td></tr><tr><td>Pre-training</td><td>ImageNet-1K (weights = imagenet)</td></tr><tr><td>Input resolution</td><td>224 × 224 × 3, normalized to [−1, 1]</td></tr><tr><td>Dataset</td><td>Tiny ImageNet, 500 images (all three models correct on clean)</td></tr><tr><td>Attack</td><td>FGŠM, untargeted, white-box, non-adaptive</td></tr><tr><td>Perturbation levels (ε)</td><td>0.001, 0.01, 0.05, 0.1</td></tr><tr><td>Defenses evaluated</td><td>6 preprocessing methods (Table 3)</td></tr><tr><td>Framework</td><td>TensorFlow / Keras 2.19.0, Python 3.12.12</td></tr><tr><td>Hardware</td><td>Google Colab, CPU inference</td></tr></table>

Table 3. Defense method configurations.
<table><tr><td>Defense</td><td>Parameters</td><td>Mechanism</td></tr><tr><td>Gaussian Blur</td><td>σ = 0.5, 1.0, 2.0</td><td>High-frequency attenuation</td></tr><tr><td>JPEG Compression</td><td>Quality Q = 50, 75, 90</td><td>DCT coefficient quantization</td></tr><tr><td>Median Filter</td><td>3×3, 5×5</td><td>Rank-order spatial statistics</td></tr><tr><td>Bit-Depth Reduction</td><td>3-7 bits</td><td>Intensity quantization</td></tr><tr><td>Resize Defense</td><td>0.5×, 0.75× then upscale</td><td>Spatial quantization noise</td></tr><tr><td>Ensemble</td><td>σ=1.0 then Q=75</td><td>Sequential complementary transforms</td></tr></table>

## 5 Results

## 5.1 Baseline Attack Efficacy

FGSM reliably compromises all three architectures, establishing the precondition for defense evaluation (Figure 2, Table 4). ResNet50 ASR rises from 94.20% at ε=0.001 to 99.00% at ε=0.1, while PSNR falls from 51.16 dB to 20.19 dB. MobileNetV2 reaches 90.8% ASR at ε=0.001 and 98.8% by ε=0.01. EfficientNetB0 achieves 100% ASR at ε=0.001, the highest baseline vulnerability of the three, sustaining near-complete attack success throughout.

## 5.2 Consistent Defense Failure on Depthwise-Separable Architectures Under FGSM

All six preprocessing defenses fail on both depthwise-separable architectures across all configurations and perturbation magnitudes, the central security finding of this study (Figure 3, Table 5). EfficientNetB0 achieves a maximum DSR of

(B) Image Quality vs Perturbation Strength  
Table 4. FGSM attack efficacy (n=500) with 95% binomial confidence intervals. EfficientNetB0 PSNR/SSIM are constant due to internal scaling (excluded from quality analysis).
<table><tr><td>ε</td><td colspan="3">ASR (%) [95% CI]</td><td colspan="2">PSNR (dB)</td><td colspan="2">SSIM</td></tr><tr><td></td><td>ResNet50</td><td>MobileNetV2 EfficientNetB</td><td></td><td>Res50</td><td>MobV2</td><td>Res50</td><td>MobV2</td></tr><tr><td>0.001</td><td>94.20 [91.9, 96.0]</td><td>90.80 [88.0, 93.1]</td><td>100.00 [99.3, 100]</td><td>51.16</td><td>51.16</td><td>0.9984</td><td>0.9985</td></tr><tr><td>0.01</td><td>97.60 [95.9, 98.7]</td><td>98.80 [97.5, 99.5]</td><td>100.00</td><td>40.04</td><td>44.18</td><td>0.9673</td><td>0.9878</td></tr><tr><td>0.05</td><td>98.80 [97.5, 99.5]</td><td>97.60 [95.9, 98.7]</td><td>[99.3, 100] 99.80</td><td>26.28</td><td>31.91</td><td>0.6251</td><td>0.8336</td></tr><tr><td>0.1</td><td>99.00 [97.8, 99.6]</td><td>98.00 [96.4, 98.9]</td><td>[99.0, 100] 99.40 [98.4, 99.8]</td><td>20.19</td><td>26.28</td><td>0.3561</td><td>0.6266</td></tr></table>

![](images/1e001c8ef0d1f2e405b4d3d01fda522b59d9481c0222ff755d37814577d7771c.jpg)

![](images/b8f78d5e71102f1bd3e9e719e2260c5e89fed09aca1175ceaf7efa22debd1cfb.jpg)  
Fig. 2. (A) FGSM Attack Success Rate vs. perturbation magnitude. EfficientNetB0 achieves 100% ASR at ε=0.001. (B) PSNR and SSIM vs. ε for ResNet50 and MobileNetV2. EfficientNetB0 PSNR is constant (≈4.56 dB, internal scaling artifact) and is excluded.

0.6% (Gaussian Blur σ=1.0 and JPEG Q=50 at ε=0.001); the majority of configurations record exactly 0.0% at all perturbation levels. This failure is epsilonindependent: DSR does not vary with ε across any defense or parameter setting. That EfficientNetB0 fails more completely than MobileNetV2 despite having 5.3 M vs. 3.4M parameters confirms that architectural design family, not pa-rameter count, determines preprocessing effectiveness. MobileNetV2 shows par-tial recovery only at ε=0.001 (Ensemble: 25.99% DSR; Bit-Depth 5: 25.55%), collapsing below 9% at all higher perturbation magnitudes. ResNet50 achieves 52.23% DSR (Bit-Depth 4, ε=0.001). The non-overlapping 95% confidence inter-vals across this three-tier hierarchy confirm statistically significant architectural differentiation.

![](images/6f72971b5632c03b08d284dabbdf53c1173be8db67d62fd706f795c808b33af1.jpg)  
Fig. 3. DSR (%) heatmap across all architectures, defenses, and perturbation magnitudes (n=500). ResNet50 (left) shows partial effectiveness at low ε. MobileNetV2 (center) collapses at $\varepsilon \ge 0 . 0 1$ . EfficientNetB0 (right) is near-zero at all $\varepsilon ,$ an epsilonindependent categorical failure.

Table 5. Defense success rates at ε=0.001 with 95% binomial confidence intervals.
<table><tr><td>Defense</td><td>ResNet50 DSR [cǐ]</td><td></td><td>PSNR</td><td>SSIM</td><td>MobV2 DSR [CI]</td><td>EffBo DSR [cI]</td><td></td></tr><tr><td>Bit-Depth 4</td><td></td><td>52.23 [47.7, 56.7]</td><td>34.36</td><td>0.9093</td><td>16.30 [13.2, 19.8]</td><td>0.20 [0.0, 1.1]</td><td></td></tr><tr><td>Ensemble (GB+JPEG)</td><td>31.21 [27.1,35.6]</td><td></td><td>36.63</td><td>0.9677</td><td>25.99 [22.3, 30.0]</td><td>0.00 [0.0, 0.7]</td><td></td></tr><tr><td>Gaussian Blur σ=2.0</td><td>29.09 [25.1, 33.4]</td><td></td><td>34.06</td><td>0.9462</td><td>17.62 [14.4, 21.2]</td><td>0.40 [0.1, 1.4]</td><td></td></tr><tr><td>Median 5×5</td><td></td><td>22.08 [18.5, 26.0]</td><td>33.66</td><td>0.9356</td><td>13.22 [10.4, 16.5]</td><td></td><td>0.00 [0.0, 0.7]</td></tr><tr><td>JPEG Q=50</td><td></td><td>22.08 [18.5, 26.0]</td><td>38.26</td><td>0.9723</td><td>18.28 [15.0, 21.9]</td><td></td><td>0.60 [0.2, 1.7]</td></tr><tr><td>Resize 0.5×</td><td></td><td>15.71 [12.6, 19.2]</td><td>37.39</td><td>0.9756</td><td>5.51 [3.6, 8.1]</td><td></td><td>0.40 [0.1, 1.4]</td></tr></table>

## 5.3 Architectural Versus Parametric Failure

Varying Gaussian blur σ from 0.1 to 5.0, spanning under-smoothing through over-smoothing, leaves MobileNetV2 DSR flat below 9% throughout, while PSNR and SSIM remain stable, confirming the defense transform operates correctly. An identical sweep on ResNet50 yields a non-trivial DSR profile peaking near σ=2.0. The contrast between these two responses under identical conditions constitutes direct evidence that failure is architectural, not parametric.

## 5.4 Quality-Security Trade-off

Bit-Depth 4 reduction achieves the highest DSR (52.23%) and Composite Score (CS=0.70) for ResNet50, making it the Pareto-dominant defense (Table $^ { 6 , }$ Figure 4). Critically, higher PSNR does not predict higher DSR: Spearman $\rho =$ $- 0 . 3 1 \left( p = 0 . 2 8 \right)$ for ResNet50 and $\rho = 0 . 0 9 \left( p = 0 . 7 6 \right)$ for MobileNetV2, neither statistically significant. For EfficientNetB0, PSNR is constant across all configurations, rendering ρ structurally undefined, itself a finding confirming that image fidelity is informationally independent of security outcome for this architecture. Practitioners who evaluate preprocessing defenses using PSNR or SSIM alone will receive a misleading assessment of actual defense effectiveness.

![](images/4caad7c33b92784da5747f7742193a6df4e72576fbfe65007a3a88b54b053a14.jpg)

![](images/a5f8f4d51456d3d4ab96f2a2d4557b610e24889e8e488b039b3e5d569055a923.jpg)  
Fig. 4. Quality-security scatter for ResNet50 (n=500, ε=0.001). Bubble size represents Composite Score. The non-monotonic DSR-PSNR relationship (Spearman $\pmb { \rho } = - 0 . 3 1$ $p = 0 . 2 8 )$ confirms image fidelity cannot proxy for security.

Table 6. Quality-security Pareto frontier for ResNet50 at ε=0.001.
<table><tr><td>Defense</td><td>Best Param.</td><td>DSR (%)</td><td>PSNR (dB)</td><td>SSIM</td><td>CS</td></tr><tr><td>Bit-Depth Reduction</td><td>4 bits</td><td>52.23</td><td>34.36</td><td>0.9093</td><td>0.70</td></tr><tr><td>Ensemble (GB+JPEG)</td><td>σ=1.0, Q=75</td><td>31.21</td><td>36.63</td><td>0.9677</td><td>0.62</td></tr><tr><td>Gaussian Blur</td><td>σ=2.0</td><td>29.09</td><td>34.06</td><td>0.9462</td><td>0.59</td></tr><tr><td>JPEG Compression</td><td>Q=50</td><td>22.08</td><td>38.26</td><td>0.9723</td><td>0.59</td></tr><tr><td>Resize Defense</td><td>0.5×</td><td>15.71</td><td>37.39</td><td>0.9756</td><td>0.55</td></tr><tr><td>Median Filter</td><td>5×5</td><td>22.08</td><td>33.66</td><td>0.9356</td><td>0.55</td></tr></table>

## 5.5 Output Divergence and Transferability

Transferability. Cross-architecture transfer achieved 100% ASR in both directions (MobileNetV2 ↔ ResNet50) across all ε, ruling out gradient-magnitude or gradient-structure explanations for the observed failure.

Output divergence asymmetry. The detection signal is quantified directly in Table 7. For EfficientNetB0, preprocessing disrupts clean accuracy by an average of 24.1 percentage points across all 14 defense configurations, while DSR (adversarial prediction change) averages below 0.3%. The gap between these two quantities is the detection opportunity: preprocessing moves clean outputs substantially but fails to move adversarial outputs. For Median Filter 5×5, clean accuracy falls by 44.8 pp while DSR is exactly 0.0%; for Gaussian Blur σ=2.0, 36.8 pp clean disruption against 0.4% DSR. Figure 5 visualizes this asymmetry across all configurations.

Table 7. Output divergence asymmetry for EfficientNetB0 (n=500, ε=0.001). Asymmetry Gap = Clean Acc. Drop minus DSR; this gap is the detection signal.
<table><tr><td>Defense</td><td>Clean Acc. After Defense</td><td>Clean Drop (pp)</td><td></td><td>DSR (%) Asymmetry Gap (pp)</td></tr><tr><td></td><td>(%)</td><td>8.2</td><td>0.2</td><td>8.0</td></tr><tr><td>Gaussian Blur σ=0.5 Gaussian Blur σ=1.0</td><td>91.8 74.0</td><td>26.0</td><td>0.6</td><td>25.4</td></tr><tr><td>Gaussian Blur σ=2.0</td><td>63.2</td><td>36.8</td><td>0.4</td><td>36.4</td></tr><tr><td>JPEG Q=50</td><td>80.4</td><td>19.6</td><td>0.6</td><td>19.0</td></tr><tr><td>JPEG Q=75</td><td>84.2</td><td>15.8</td><td>0.4</td><td>15.4</td></tr><tr><td>JPEG Q=90</td><td>91.0</td><td>9.0</td><td>0.0</td><td>9.0</td></tr><tr><td>Median 3×3</td><td>79.4</td><td>20.6</td><td>0.4</td><td>20.2</td></tr><tr><td>Median 5×5</td><td>55.2</td><td>44.8</td><td>0.0</td><td>44.8</td></tr><tr><td>Bit-Depth 4</td><td>57.0 75.6</td><td>43.0</td><td>0.2</td><td>42.8</td></tr><tr><td>Bit-Depth 5</td><td>89.2</td><td>24.4 10.8</td><td>0.0</td><td>24.4</td></tr><tr><td>Bit-Depth 6</td><td>73.4</td><td>26.6</td><td>0.0 0.4</td><td>10.8</td></tr><tr><td>Resize 0.5× Resize 0.75×</td><td>83.0</td><td>17.0</td><td>0.2</td><td>26.2</td></tr><tr><td>Ensemble</td><td>78.8</td><td>21.2</td><td>0.0</td><td>16.8</td></tr><tr><td></td><td></td><td></td><td></td><td>21.2</td></tr></table>

pp = percentage points. Mean Asymmetry Gap = 24.1 pp.

![](images/f16f56f3a1ed6f0beee176eb43822384bbb56d331ad09ff898797bd8b2d3cb49.jpg)  
Fig. 5. Output divergence asymmetry for EfficientNetB0 (n=500, ε=0.001). Dark bars: clean accuracy drop per defense. Light bars: DSR. The consistently large gap confirms that preprocessing generates a strong, measurable detection signal on depthwiseseparable architectures.

## 5.6 Computational Overhead

Preprocessing adds negligible overhead: bit-depth reduction executes in 0.02 ms per image, Gaussian blur in 0.15 ms, and JPEG compression in 0.74 ms, against inference latencies of 844 ms (ResNet50), 390 ms (MobileNetV2), and ≈404 ms (EfficientNetB0). The barrier to deploying preprocessing is not cost but security efficacy. Critically, the detection proposal in Section 6 adds only a single preprocessing pass and one additional inference to the pipeline, well within edge latency budgets.

## 6 Discussion

## 6.1 Architectural Origins of Defense Failure

The depthwise-separable factorization, per-channel spatial filtering followed by cross-channel projection, processes each feature channel independently before mixing. This likely produces FGSM gradient sign patterns that are more spatially concentrated and channel-consistent than those from standard convolutions, aligning decision boundaries more tightly with the perturbation direction. Preprocessing that attenuates perturbation energy without reversing that direction leaves residual adversarial signal sufficient to sustain misclassification. The epsilon-independent failure of EfficientNetB0 extends this argument: the squeeze-and-excitation recalibration mechanism appears to produce even tighter adversarial gradient alignment than MobileNetV2’s inverted residuals, explaining why a 5.3M parameter model fails more completely than a 3.4M parameter model. ResNet50’s residual pathways, by contrast, distribute adversarial signal across multiple resolution scales; clean coarse-scale information flows through skip connections, allowing preprocessing-disrupted fine-scale perturbations to drop below the decision boundary threshold. Gradient geometry analysis via Grad-CAM [24] across all three architectures is identified as the priority future direction for formal verification of this hypothesis.

## 6.2 Redeploying Preprocessing as a Detection Mechanism

The recovery-failure-as-detection-signal insight is structurally precise. Recovery requires output divergence δ to be large and in the direction of the correct class. Detection requires only that δ be large. The first condition fails on depthwiseseparable CNNs; the second holds consistently, as Table 7 establishes, with a mean asymmetry gap of 24.1 pp and individual configurations reaching 44.8 pp. For ResNet50, the asymmetry is weaker because preprocessing sometimes shifts the prediction toward the correct class, making δ directionally informative and noisier as a detection signal. Counterintuitively, the architectures most resistant to recovery are most amenable to detection.

The practical proposal builds on Feature Squeezing [6]: run inference twice, once on the raw input, once on the preprocessed input. If $\delta = \parallel p _ { \Theta } ( x ) - p _ { \Theta } ( d _ { \Phi } ( x ) )$ ∥1 exceeds threshold τ (calibrated on clean validation data), reject the input as adversarial. No retraining, no architectural change, and negligible overhead (0.15– 0.74 ms per preprocessing pass). Two limitations apply: an adaptive adversary aware of the detection criterion can minimize δ to evade it; and threshold calibration requires clean validation data representative of deployment conditions. Both are identified as priorities for future empirical validation with full ROC analysis.

The present results also provide the first evidence that the Feature Squeezing detection signal is stronger on depthwise-separable architectures than on the residual ones on which it was originally proposed and evaluated [6]. This inversion of the conventional robustness hierarchy, harder to defend, easier to detect, is a novel and practically consequential finding.

## 6.3 Security Implications

Edge vision systems with depthwise-separable backbones that rely on preprocessingonly defenses should be considered effectively unprotected against even a basic FGSM attacker. The non-adaptive evaluation means DSR values are upper bounds; a stronger adversary makes the situation strictly worse. The broader methodological failure is that defense evaluation practices dominant in the adversarial robustness literature, benchmarking on ResNet and Inception-class architectures, are systematically unrepresentative of edge deployment reality, and their implicit generalization assumption is empirically false. Three concrete remedies exist: replace the depthwise-separable backbone with a residual architecture; incorporate adversarial training [10] at training time for architectureindependent hardening; or deploy the preprocessing-as-detection approach described above. Table 8 provides a practitioner decision guide.

Table 8. Practitioner Defense Decision Guide for Edge Vision Security.
<table><tr><td>Category</td><td>Depthwise-Sep. CNN</td><td>Residual CNN</td><td>Unknown Arch</td></tr><tr><td>Architecture</td><td>MobileNetV2, EffNetB0</td><td>ResNet50, ResNet101</td><td>Unknown</td></tr><tr><td>Preprocessing Def.</td><td>Ineffective</td><td>Partially effective</td><td>Evaluate first</td></tr><tr><td>Recommended Alt.</td><td>(DSR &lt;1%) (1) Adv. training (2) Feat. squeezing</td><td>(~52% max DSR) (1) Bit-depth (4-bit) Char. DSR first (2) Ensemble</td><td></td></tr><tr><td>Risk Level</td><td>(3) Replace arch High</td><td>(3) Monitor ε Moderate</td><td>Unknown</td></tr></table>

Note: DSR values are upper bounds (non-adaptive FGSM only). Stronger attacks  
(e.g., PGD, AutoAttack) would likely yield lower DSR.

## 7 Limitations and Future Work

Exclusive use of FGSM yields upper-bound DSR estimates, strengthening rather than weakening the security conclusion: categorical failure under the weakest attacker implies failure under all stronger adversaries. The most important extension is evaluation under adaptive attacks [15,25] and the AutoAttack benchmark [13,16]. The three-architecture comparison does not generalize across the full CNN landscape; the depthwise-separable failure hypothesis requires extension to MobileNetV3 [3], ShuffleNetV2, and lightweight Vision Transformers such as MobileViT [26]. EfficientNetB0 cross-architecture transferability evaluation was not conducted. The detection proposal requires full empirical validation: threshold calibration, ROC analysis, and evaluation against adversaries aware of the detection mechanism. Extension to certified defenses (randomized smoothing [27]) and diffusion-based purification [28] would situate preprocessing within the broader defense landscape. Grad-CAM [24] gradient geometry analysis across all three architectures would formally test the mechanistic hypothesis in Section 6.

## 8 Conclusion

Under non-adaptive FGSM, preprocessing defenses fail to provide meaningful recovery on depthwise-separable CNNs, with the best configuration reaching only 0.6% DSR on EfficientNetB0. MobileNetV2 collapses below 9% DSR at operational magnitudes. ResNet50 recovers up to 52.23%. This three-tier hierarchy is statistically confirmed by non-overlapping 95% confidence intervals; its architectural causation is confirmed by parameter ablation; and gradient-magnitude explanations are ruled out by 100% bidirectional FGSM transferability. PSNR and SSIM are statistically shown to be unreliable security proxies.

The failure, however, enables detection. Preprocessing disrupts clean model outputs on EfficientNetB0 by an average of 24.1 percentage points while leaving adversarial outputs unchanged, an asymmetric, measurable signal requiring no retraining and adding under 1 ms of overhead per image. Edge vision systems with depthwise-separable backbones should treat preprocessing-only defenses as ineffective for recovery and redeploy them as detection mechanisms, supplemented where necessary by adversarial training [10] or detection-layer augmentation following the Feature Squeezing principle [6].

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. MobileNetV2: Inverted residuals and linear bottlenecks. In Proc. CVPR, pages 4510–4520, Salt Lake City, UT, USA, 2018.

2. Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Proc. ICML, pages 6105–6114, Long Beach, CA, USA, 2019.

3. Andrew Howard et al. Searching for MobileNetV3. In Proc. ICCV, pages 1314– 1324, Seoul, Korea, 2019.

4. Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In Proc. ICLR, San Diego, CA, USA, 2015.

5. Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens van der Maaten. Countering adversarial images using input transformations. In Proc. ICLR, Vancouver, Canada, 2018.

6. Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. In Proc. NDSS, San Diego, CA, USA, 2018.

7. Adrita Rahman Tory, Khondokar Fida Hasan, Md Saifur Rahman, Nickolaos Koroniotis, and Mohammad Ali Moni. Mind the gap: Missing cyber threat coverage in nids datasets for the energy sector. In International Conference on Big Data, IoT and Machine Learning, pages 434–447. Springer, 2025.

8. Khondokar Fida Hasan, Yanming Feng, and Yu-Chu Tian. Exploring the potential and feasibility of time synchronization using gnss receivers in vehicleto-vehicle communications. In Proceedings of the 49th Annual Precise Time and Time Interval Systems and Applications Meeting, pages 80–90, 2018.

9. Khondokar Fida Hasan and Md Morshedul Islam. Evolution of the 4 th generation mobile communication network: Lte-advanced. International Journal of Computer Technology and Applications, 2(4), 2011.

10. Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In Proc. ICLR, Vancouver, Canada, 2018.

11. Gintare Karolina Dziugaite, Zoubin Ghahramani, and Daniel M. Roy. A study of the effect of JPG compression on adversarial images, 2016. arXiv:1608.00853.

12. Dongyu Meng and Hao Chen. MagNet: A two-pronged defense against adversarial examples. In Proc. ACM CCS, pages 135–147, Dallas, TX, USA, 2017.

13. Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In Proc. ICML, pages 2206– 2216, Vienna, Austria, 2020.

14. Christian Szegedy et al. Intriguing properties of neural networks. In Proc. ICLR, Banff, Canada, 2014.

15. Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In Proc. IEEE S&P, pages 39–57, San Jose, CA, USA, 2017.

16. Francesco Croce et al. RobustBench: A standardized adversarial robustness benchmark. In Proc. NeurIPS Datasets and Benchmarks, 2021.

17. Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, and Aleksander Madry. Robustness may be at odds with accuracy. In Proc. ICLR, New Orleans, LA, USA, 2019.

18. Rui Shao, Zuxuan Shi, Jinfeng Yi, Pin-Yu Chen, and Cho-Jui Hsieh. On the adversarial robustness of vision transformers, 2021. arXiv:2103.15670.

19. Srinadh Bhojanapalli et al. Understanding robustness of transformers for image classification. In Proc. ICCV, pages 10231–10241, Montreal, Canada, 2021.

20. Alain Hore and Djemel Ziou. Image quality metrics: PSNR vs. SSIM. In Proc. ICPR, pages 2366–2369, Istanbul, Turkey, 2010.

21. Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Trans. Image Process., 13(4):600–612, 2004.

22. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. CVPR, pages 770–778, Las Vegas, NV, USA, 2016.

23. Ya Le and Xuan Yang. Tiny ImageNet visual recognition challenge. Technical report, Stanford University, 2015. CS231N Course Report.

24. Ramprasaath R. Selvaraju et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proc. ICCV, pages 618–626, Venice, Italy, 2017.

25. Florian Tramèr, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to adversarial example defenses. In Proc. NeurIPS, Vancouver, Canada, 2020.

26. Sachin Mehta and Mohammad Rastegari. MobileViT: Light-weight, generalpurpose, and mobile-friendly vision transformer. In Proc. ICLR, 2022.

27. Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In Proc. ICML, pages 1310–1320, Long Beach, CA, USA, 2019.

28. Y. Nie et al. Diffusion models for adversarial purification. In Proc. ICML, Baltimore, MD, USA, 2022.