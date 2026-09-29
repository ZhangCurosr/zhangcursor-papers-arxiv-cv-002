# GLF-Q: GLOBAL-LOCAL FEATURE-BASED QUANTI-ZATION FOR VISION TRANSFORMERS

Peilin Sun<sup>1,2</sup> Guang Liang<sup>1,2,3</sup> Jin Tong<sup>1,2</sup> Jianxin Wu<sup>1,2∗</sup>

<sup>1</sup>State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing 210023, China

<sup>2</sup>School of Artificial Intelligence, Nanjing University, Nanjing 210023, China

<sup>3</sup>Zhongguancun Academy, Beijing 100094, China

{sunpl, liangg, tongj}@lamda.nju.edu.cn, wujx2001@nju.edu.cn

## ABSTRACT

Post-training quantization (PTQ) efficiently compresses Vision Transformers (ViTs) without retraining, yet suffers severe accuracy degradation at low bitwidths. Existing optimization-based PTQ methods guide block reconstruction via either soft logits or second-order Hessian proxies. Logit supervision is prone to overfitting on limited calibration data, while Hessian approximations incur structural truncation errors. To address these limitations, we propose GLF-Q, a novel PTQ framework guided by Global-Local Feature alignment. GLF-Q propagates quantized block outputs through downstream full-precision layers to align penultimate-layer representations under local output regularization, providing downstream feature supervision without explicitly approximating the Hessian or using a Taylor expansion. Furthermore, offline Hadamard transformations are introduced with zero runtime overhead to disperse activation outliers across channels, effectively contracting dynamic ranges and reducing quantization errors. Meanwhile, optimizing this loss via a Straight-Through Estimator (STE) achieves rapid convergence, bypassing continuous relaxation rounding formulations such as AdaRound. Extensive experiments across representative ViT architectures demonstrate that GLF-Q with standard uniform quantizers substantially outperforms state-of-the-art methods under 3-bit quantization on image classification. In addition, GLF-Q exhibits strong out-of-domain calibration robustness and achieves speedups under 8-bit GPU deployment.

## 1 INTRODUCTION

Vision Transformers (ViTs) (Dosovitskiy et al., 2020) have achieved remarkable success across diverse visual recognition tasks, such as image classification (Touvron et al., 2021), as well as object detection and instance segmentation (He et al., 2017; Cai & Vasconcelos, 2018; Liu et al., 2021), owing to their powerful capability in modeling long-range dependencies. However, their massive model sizes, intensive self-attention computations, and heavy memory bandwidth demands severely limit practical deployment on resource-constrained edge devices. Among various model compression techniques, network quantization has attracted widespread attention, as it converts floating point weights and activations into low-bit representations, thereby significantly reducing memory footprints and substantially accelerating hardware inference simultaneously.

Existing quantization approaches can be broadly divided into Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ). Although QAT can effectively recover model accuracy through retraining, it requires full labeled data and expensive retraining, severely restricting its practical deployment. In contrast, PTQ requires only a small amount of unlabeled data for calibration, making it a practical and efficient approach.

Depending on whether optimization is required, existing PTQ methods primarily fall into two categories: calibration-only methods (Yuan et al., 2022; Li et al., 2023; Ding et al., 2022) and optimization-based methods (Nagel et al., 2020; Li et al., 2021; Liu et al., 2023; Wu et al., 2025a;b;

Hwang et al., 2026). Calibration-only methods adjust quantization parameters solely during the calibration phase. However, in low-bit settings, quantization errors increase substantially, leading to severe accuracy degradation. To overcome this limitation, optimization-based methods perform block-wise reconstruction to compensate for quantization loss, emerging as the prevailing paradigm for ViT PTQ.

In block-wise reconstruction, the choice of supervision signal directly affects the accuracy of the quantized model. Existing reconstruction approaches can be broadly classified into two categories based on their supervisory signals. The first category comprises logit-guided methods (Liu et al., 2023), which seek global supervision by minimizing the prediction difference on output soft logits at the task head. However, fitting output logits on a small calibration set can lead to overfitting and limit generalization, as shown in Appendix D.

Hessian-guided reconstruction (Li et al., 2021; Wu et al., 2025a;b; Hwang et al., 2026) uses secondorder loss approximations to weight reconstruction errors according to their estimated impact on the task loss. Since computing the full Hessian is expensive, these methods employ tractable Hessian approximations, such as diagonal or low-rank representations. The accuracy of the estimated loss change depends on both the Hessian approximation and the second-order Taylor approximation. Under low-bit quantization, larger perturbations may make higher-order effects more relevant and reduce the accuracy of the second-order approximation.

These limitations motivate evaluating feature differences directly through the full-precision downstream network, without explicitly constructing a Hessian approximation or truncating a Taylor expansion. We combine penultimate-layer feature alignment with local reconstruction regularization to improve generalization on limited calibration data.

Motivated by these insights, we propose GLF-Q, an efficient post-training quantization framework for Vision Transformers based on Global-Local Feature alignment. GLF-Q introduces a Global-Local Feature (GLF) loss, which combines penultimate feature alignment with local reconstruction regularization. Furthermore, to address the critical challenge of extreme activation outliers and inter-channel scale disparities in ViTs, we systematically introduce offline Hadamard transformations (Ashkboos et al., 2024b) into the ViT PTQ pipeline for the first time. This cheap operation evenly distributes energy concentrated in outlier channels across all dimensions, contracting activation dynamic ranges to effectively suppress quantization errors with zero runtime overhead. In addition, regarding the reconstruction optimization mechanism, GLF-Q avoids continuous rounding relaxation methods such as AdaRound (Nagel et al., 2020). Instead, it uses a basic Straight-Through Estimator (STE) (Bengio et al., 2013) to achieve fast and stable convergence, thereby improving reconstruction efficiency.

Our main contributions are summarized as follows:

• We propose the Global-Local Feature (GLF) loss, which combines global feature alignment with local reconstruction regularization to mitigate calibration overfitting and avoid explicit Hessian approximations.

• To mitigate activation outliers, we introduce offline Hadamard transformations to spread large activation values across channels, reducing quantization errors. For reconstruction optimization, we use a simple STE instead of continuous rounding relaxation to achieve fast convergence.

• We demonstrate practical hardware efficiency and strong out-of-domain generalizability. GLF-Q achieves higher accuracy and lower inference latency under practical 8-bit GPU deployment. It also maintains strong accuracy under out-of-domain calibration.

## 2 RELATED WORK

Model quantization (Gholami et al., 2022) converts floating-point weights and activations into lowbit representations to reduce memory usage and accelerate inference. Existing quantization methods can be classified into Quantization-Aware Training (QAT), which requires retraining, and Post-Training Quantization (PTQ), which does not. Although QAT methods (Esser et al., 2019; Li et al., 2022; Liang et al., 2026) achieve competitive accuracy, their reliance on full labeled data and expensive retraining severely limit practical deployment.

In contrast, PTQ calibrates models using only a small set of unlabeled data without retraining, making it an efficient and popular approach. Depending on whether optimization is involved, PTQ methods broadly fall into calibration-only methods (Yuan et al., 2022; Ding et al., 2022; Li et al., 2023; Wu et al., 2024; Zhong et al., 2024; Moon et al., 2024; Fu et al., 2025; Jiang et al., 2026) and optimization-based methods (Nagel et al., 2020; Li et al., 2021; Wei et al., 2022; Liu et al., 2023; Zhong et al., 2023; Yang et al., 2024; Ma et al., 2024; Wu et al., 2025b;a; Hwang et al., 2026).

Calibration-only methods determine quantization parameters directly during a forward calibration pass without backpropagation. RepQ-ViT (Li et al., 2023) uses scale reparameterization to convert complex activation quantizers into hardware-friendly forms for efficient inference. AdaLog (Wu et al., 2024) addresses heavy-tailed activations via dynamic log-base tuning paired with a progressive search strategy. UQ-ViT (Jiang et al., 2026) introduces DeMax and NormQuant to accommodate extreme activation distributions with uniform quantizers. Despite these efforts, calibration-only methods suffer substantial accuracy loss in low-bit settings, while those employing specialized nonuniform formats remain difficult to deploy on general hardware.

To recover accuracy, optimization-based methods perform block-wise reconstruction to reduce quantization loss. BRECQ (Li et al., 2021) optimizes weight rounding using second-order error analysis across residual blocks. QDrop (Wei et al., 2022) randomly drops activation quantization during reconstruction to enhance model flatness and noise tolerance. PD-Quant (Liu et al., 2023) minimizes prediction differences on soft classification logits for global guidance. DopQ-ViT (Yang et al., 2024) handles activation outliers during reconstruction via LayerNorm input reparameterization and specialized non-uniform quantizers. APHQ-ViT (Wu et al., 2025b) couples an average perturbation Hessian loss with MLP reconstruction. FIMA-Q (Wu et al., 2025a) constructs a composite Hessian loss by modeling both diagonal and low-rank cross-channel dependencies. LS-ViT (Hwang et al., 2026) formulates Hessian proxy estimation as a least-squares problem over calibration gradients, achieving rapid reconstruction with single-pass backpropagation. Despite these advances, existing reconstruction methods still face limitations. Logit-guided methods can overfit limited calibration data, while Hessian-guided methods rely on approximations that may become less accurate under low-bit quantization.

## 3 METHOD

## 3.1 PRELIMINARIES AND MOTIVATION

Logit-Guided Block Reconstruction. In block-wise reconstruction, parameter optimization relies heavily on the supervisory signal. To incorporate global task guidance, logit-guided methods, represented by PD-Quant (Liu et al., 2023), minimize the prediction difference on output soft logits at the final task head. However, fitting output logits on a small calibration set can lead to overfitting and limit generalization.

Hessian-Guided Block Reconstruction. Let θ denote the parameters of the pretrained model and ∆θ be the quantization perturbation. The change in task loss can be approximated via a second-order Taylor expansion:

$$
\mathbb { E } \left[ \mathcal { L } ( \boldsymbol { \theta } + \Delta \boldsymbol { \theta } ) \right] - \mathbb { E } \left[ \mathcal { L } ( \boldsymbol { \theta } ) \right] \approx \mathbb { E } \left[ \Delta \boldsymbol { \theta } ^ { \top } \boldsymbol { g } ^ { ( \boldsymbol { \theta } ) } + \frac { 1 } { 2 } \Delta \boldsymbol { \theta } ^ { \top } \boldsymbol { H } ^ { ( \boldsymbol { \theta } ) } \Delta \boldsymbol { \theta } \right] ,\tag{1}
$$

where $\boldsymbol { g } ^ { ( \theta ) } = \nabla _ { \theta } \mathcal { L }$ is the gradient and $H ^ { ( \theta ) } = \nabla _ { \theta } ^ { 2 } \mathcal { L }$ is the Hessian matrix. Since a well-converged model satisfies $g ^ { ( \theta ) } \approx 0$ , existing reconstruction-based PTQ methods map parameter perturbations to the module output perturbation $\Delta y$ via the chain rule:

$$
\mathcal { L } _ { \mathrm { H } } ( \Delta y ) = \frac { 1 } { 2 } \Delta y ^ { \top } H ^ { ( y ) } \Delta y ,\tag{2}
$$

where $H ^ { ( y ) } = \nabla _ { u } ^ { 2 } \mathcal { L }$ is the Hessian matrix with respect to the module output $y .$ Compared with vanilla MSE, ${ \mathcal { L } } _ { \mathrm { H } }$ weights quantization errors according to directional task sensitivity.

Because computing the dense matrix $H ^ { ( y ) }$ is intractable, existing methods commonly adopt diagonal, low-rank, or Fisher matrix approximations (Li et al., 2021; Wu et al., 2025a;b; Hwang et al.,

(a) Global-local feature reconstruction  
![](images/5393e6c6d50adc387f875faa827f90d77e4949871be2af28ab98acecf0aff991.jpg)

(b) Offline Hadamard Transformation  
![](images/9da20f335b75eae7ead40564c1620bf4e7d496b8786be985fc2efd1492ec5be9.jpg)  
Figure 1: Overview of the proposed GLF-Q framework. (a) Global-local feature reconstruction. For block $l ,$ the quantized block output $\hat { y } _ { l }$ is propagated through the frozen downstream full-precision network $F _ { l }$ to match the precomputed full-precision reference feature $f ,$ under local reconstruction regularization. Gradients are backpropagated directly to update the quantized block via STE. (b) Offline Hadamard transformation. Orthogonal transformations are fused offline into adjacent linear weights to suppress activation outliers and balance channel distributions with zero runtime overhead.

2026). However, the accuracy of these proxies depends on the second-order Taylor approximation and the structural assumptions used to estimate curvature. Under low-bit quantization, larger perturbations may reduce the accuracy of the local quadratic approximation, motivating direct measurement of downstream feature differences.

## 3.2 GLOBAL-LOCAL FEATURE LOSS

To address the two limitations discussed above, we introduce the Global-Local Feature (GLF) loss, illustrated in Figure 1(a). Specifically, the GLF loss directly leverages global penultimate-layer feature representations (Wang et al., 2021) to evaluate downstream quantization loss under local regularization.

When reconstructing block l, preceding quantized blocks are frozen, with their cached activations serving as input $\hat { x } _ { l }$ . We cache the global feature $f \in \mathbb { R } ^ { C }$ and local block output $y _ { l } \in \mathbb { R } ^ { N \times C }$ on $\mathcal { D } _ { \mathrm { c a l } }$ using the full-precision teacher after offline transformations. During reconstruction, the dequantized block output $\hat { y } _ { l } \in \mathbb { R } ^ { N \times C }$ is dynamically computed from $\hat { x } _ { l }$ and fed into the frozen downstream fullprecision suffix $F _ { l } \colon$

$$
\hat { f } = F _ { l } ( \hat { y } _ { l } ) ,\tag{3}
$$

where $\hat { f } \in \mathbb { R } ^ { C }$ denotes the global penultimate representation immediately preceding the task head. For image classification, $f$ is the final class token representation for ViT and DeiT, and the globally averaged token representation for Swin. The global feature loss is formulated as

$$
\mathcal { L } _ { \mathrm { g l o b a l } } ^ { ( l ) } = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { c a l } } } \left[ \frac { 1 } { C } \left\| \hat { \boldsymbol f } - \boldsymbol f \right\| _ { 2 } ^ { 2 } \right] .\tag{4}
$$

This loss measures the feature differences caused by quantization after propagation through the full precision downstream network. Concurrently, we introduce the local feature loss as a regularizer:

$$
\mathcal { L } _ { \mathrm { l o c a l } } ^ { ( l ) } = \mathbb { E } _ { { x } \sim \mathcal { D } _ { \mathrm { c a l } } } \left[ \frac { 1 } { N C } \left. \hat { y } _ { l } - y _ { l } \right. _ { F } ^ { 2 } \right] .\tag{5}
$$

The GLF loss for block l is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G L F } } ^ { ( l ) } = \mathcal { L } _ { \mathrm { g l o b a l } } ^ { ( l ) } + \lambda \mathcal { L } _ { \mathrm { l o c a l } } ^ { ( l ) } , } \end{array}\tag{6}
$$

where $\lambda > 0$ is the regularization coefficient controlling the strength of local regularization. In practice, each loss term is normalized by its value on the first reconstruction minibatch, with the normalization factors fixed throughout optimization.

During optimization of block $l ,$ the downstream suffix $F _ { l }$ remains frozen, yet gradients backpropagate through $F _ { l }$ via STE to update this block’s parameters $\theta _ { l }$ and quantization scale $s _ { l } .$ . Once reconstructed, block l is frozen in its quantized state, and the pipeline advances sequentially to the next block.

## 3.3 GLF-Q FRAMEWORK

As shown in Figure 1 and Algorithm 1 in Appendix C, GLF-Q follows a block-wise quantization pipeline. First, we apply offline Hadamard transformations to mitigate activation outliers with zero runtime overhead. We then perform MLP reconstruction by replacing GELU with strictly nonnegative ReLU to expand the effective dynamic range. Finally, we carry out block reconstruction guided by the GLF loss.

Offline Hadamard Transformation. Directly quantizing Vision Transformers remains challenged by channel scale disparities and activation outliers (Darcet et al., 2024; Sun & Wu, 2026), which widen the dynamic range and increase quantization errors. To disperse activation outliers across channels without additional inference overhead, we apply an offline Hadamard transformation (Ashkboos et al., 2024b) prior to reconstruction, as depicted in Figure 1(b). Specifically, let $M \stackrel { } { = } I - \frac { 1 } { C } \mathbf { 1 } \mathbf { 1 } ^ { \top } \in \mathbb { R } ^ { C \times C }$ denote the mean-subtraction projection matrix, where I is the identity matrix and $\mathbf { \bar { \textbf { 1 } } } \in \mathbb { R } ^ { C }$ is the all-ones vector, such that the centered activation satisfies $x _ { \mathrm { c } } ~ = ~ x M$ Following SliceGPT (Ashkboos et al., 2024a), both the mean centering and affine operations of LayerNorm are absorbed into adjacent linear layers:

$$
\mathrm { L N } ( x ) W + b = \mathrm { R M S N o r m } ( x _ { \mathrm { c } } ) \widetilde { W } + \widetilde { b } ,\tag{7}
$$

where the preceding layer absorbs mean-centering:

$$
\widetilde W _ { \mathrm { p r e v } } = W _ { \mathrm { p r e v } } M , \qquad \widetilde b _ { \mathrm { p r e v } } = b _ { \mathrm { p r e v } } M ,\tag{8}
$$

and the subsequent layer absorbs the scaling $\gamma$ and bias $\beta \colon$

$$
\widetilde { W } = \mathrm { D i a g } ( \gamma ) W , \qquad \widetilde { b } = \beta W + b .\tag{9}
$$

This transformation converts LayerNorm into standard RMSNorm. We then apply a randomized Hadamard transform $R _ { 1 }$ to rotate the residual stream:

$$
x W = ( x R _ { 1 } ) ( R _ { 1 } ^ { \top } W ) ,\tag{10}
$$

where $R _ { 1 } R _ { 1 } ^ { \top } = I$ . Both $R _ { 1 }$ and $R _ { 1 } ^ { \top }$ are fused into weights offline before quantization, distributing activation outliers across channels. For multi-head self-attention, head-wise orthogonal rotations are applied to QK and VO projections to eliminate outliers without cross-head interference. Full network equivalence, coordinate alignment, and generalized Kronecker constructions for non- $- 2 ^ { k }$ dimensions are detailed in Appendix B.

MLP Reconstruction. For nonlinear activations, the asymmetric negative tail of GELU wastes the dynamic range of low-bit quantizers and causes quantization loss. Following APHQ-ViT (Wu et al., 2025b), before block reconstruction, we conduct an MLP reconstruction step. We replace GELU with ReLU in the MLP module to avoid allocating quantization levels to negative values, and then reconstruct the MLP to reduce the changes caused by activation replacement.

Table 1: Top-1 accuracy comparison across representative Vision Transformers on ImageNet. “\*” denotes results reimplemented using the FIMA-Q framework, as QDrop and PD-Quant were originally designed for CNNs. “Opt.” indicates whether a PTQ method is optimization-based (✓) or calibration-only (×). “Spec.” indicates reliance on a specialized non-uniform quantizer $( \checkmark )$ versus a standard uniform quantizer (×). The best results within each bit-width setting are highlighted in boldface.
<table><tr><td>Method</td><td>Opt.</td><td>Spec.</td><td>W/A</td><td>ViT-S</td><td>ViT-B</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td>Full-Prec</td><td>-</td><td>-</td><td>32/32</td><td>81.39</td><td>84.54</td><td>72.21</td><td>79.85</td><td>81.80</td><td>83.23</td><td>85.27</td></tr><tr><td>PTQ4ViT (Yuan et al., 2022)</td><td>X</td><td>√</td><td>3/3</td><td>0.10</td><td>0.10</td><td>3.50</td><td>0.10</td><td>31.06</td><td>28.69</td><td>20.13</td></tr><tr><td>RepQ-ViT (Li et al., 2023)</td><td>×</td><td>√</td><td>3/3</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td></tr><tr><td>AdaLog (Wu et al., 2024)</td><td>X</td><td>√</td><td>3/3</td><td>13.88</td><td>37.91</td><td>31.56</td><td>24.47</td><td>57.47</td><td>64.41</td><td>69.75</td></tr><tr><td>I&amp;S-ViT (Żhong et al., 2023)</td><td>√</td><td>√</td><td>3/3</td><td>45.16</td><td>63.77</td><td>41.52</td><td>55.78</td><td>73.30</td><td>74.20</td><td>69.30</td></tr><tr><td>DopQ-ViT (Yang et al., 2024)</td><td>r</td><td>√</td><td>3/3</td><td>54.72</td><td>65.76</td><td>44.71</td><td>59.26</td><td>74.91</td><td>74.77</td><td>69.63</td></tr><tr><td>QDrop* (Wei et al., 2022)</td><td>√</td><td>X</td><td>3/3</td><td>41.05</td><td>74.75</td><td>46.88</td><td>50.95</td><td>72.97</td><td>74.67</td><td>76.57</td></tr><tr><td>PD-Quant* (Liu et al., 2023)</td><td>√</td><td>X</td><td>3/3</td><td>40.52</td><td>75.72</td><td>53.23</td><td>60.72</td><td>74.54</td><td>74.59</td><td>76.71</td></tr><tr><td>APHQ-ViT (Wu et al., 2025b)</td><td>√</td><td>X</td><td>3/3</td><td>63.17</td><td>76.31</td><td>55.42</td><td>68.76</td><td>76.31</td><td>76.10</td><td>78.14</td></tr><tr><td>FIMA-Q (Wu et al., 2025a)</td><td>√</td><td>X</td><td>3/3</td><td>64.09</td><td>77.63</td><td>55.55</td><td>69.13</td><td>76.54</td><td>77.26</td><td>78.82</td></tr><tr><td>LS-ViT (Hwang et al., 2026)</td><td>√</td><td>X</td><td>3/3</td><td>64.10</td><td>77.65</td><td>55.72</td><td>69.41</td><td>76.57</td><td>77.39</td><td>79.40</td></tr><tr><td>GLF-Q (Ours)</td><td>√</td><td>X</td><td>3/3</td><td>67.40</td><td>78.18</td><td>56.48</td><td>71.55</td><td>77.20</td><td>78.10</td><td>80.45</td></tr><tr><td>PTQ4ViT (Yuan et al., 2022)</td><td>X</td><td>√</td><td>4/4</td><td>42.57</td><td>30.69</td><td>36.96</td><td>34.08</td><td>64.39</td><td>76.09</td><td>74.02</td></tr><tr><td>APQ-ViT (Ding et al., 2022)</td><td>X</td><td>√</td><td>4/4</td><td>47.95</td><td>41.41</td><td>47.94</td><td>43.55</td><td>67.48</td><td>77.15</td><td>76.48</td></tr><tr><td>RepQ-ViT (Li et al., 2023)</td><td>X</td><td>√</td><td>4/4</td><td>65.05</td><td>68.48</td><td>57.43</td><td>69.03</td><td>75.61</td><td>79.45</td><td>78.32</td></tr><tr><td>ERQ (Zhong et al., 2024)</td><td>X</td><td>√</td><td>4/4</td><td>68.91</td><td>76.63</td><td>60.29</td><td>72.56</td><td>78.23</td><td>80.74</td><td>82.44</td></tr><tr><td>IGQ-ViT (Moon et al., 2024)</td><td>X</td><td>√</td><td>4/4</td><td>73.61</td><td>79.32</td><td>62.45</td><td>74.66</td><td>79.23</td><td>80.98</td><td>83.14</td></tr><tr><td>AdaLog (Wu et al., 2024)</td><td>X</td><td>√</td><td>4/4</td><td>72.75</td><td>79.68</td><td>63.52</td><td>72.06</td><td>78.03</td><td>80.77</td><td>82.47</td></tr><tr><td>UQ-ViT (Jiang et al., 2026)</td><td>X</td><td>X</td><td>4/4</td><td>68.34</td><td>72.07</td><td>59.71</td><td>72.13</td><td>76.59</td><td>79.93</td><td>81.96</td></tr><tr><td>I&amp;S-ViT (Zhong et al., 2023)</td><td>√</td><td>√</td><td>4/4</td><td>74.87</td><td>80.07</td><td>65.21</td><td>75.81</td><td>79.97</td><td>81.17</td><td>82.60</td></tr><tr><td>DopQ-ViT (Yang et al., 2024)</td><td>√</td><td>√</td><td>4/4</td><td>75.69</td><td>80.95</td><td>65.54</td><td>75.84</td><td>80.13</td><td>81.71</td><td>83.34</td></tr><tr><td>QDrop* (Wei et al., 2022)</td><td>√</td><td>X</td><td>4/4</td><td>71.84</td><td>82.63</td><td>65.27</td><td>72.64</td><td>79.96</td><td>81.21</td><td>82.99</td></tr><tr><td>PD-Quant* (Liu et al., 2023)</td><td>√</td><td>X</td><td>4/4</td><td>72.98</td><td>82.72</td><td>66.23</td><td>74.98</td><td>79.90</td><td>81.28</td><td>82.92</td></tr><tr><td>OASQ (Ma et al., 2024)</td><td>√</td><td>X</td><td>4/4</td><td>72.88</td><td>76.59</td><td>66.31</td><td>76.00</td><td>78.83</td><td>81.02</td><td>82.46</td></tr><tr><td>APHQ-ViT (Wu et al., 2025b)</td><td>√</td><td>X</td><td>4/4</td><td>76.07</td><td>82.41</td><td>66.66</td><td>76.40</td><td>80.21</td><td>81.81</td><td>83.42</td></tr><tr><td>FIMA-Q (Wu et al., 2025a)</td><td>√</td><td>X</td><td>4/4</td><td>76.68</td><td>83.04</td><td>66.84</td><td>76.87</td><td>80.33</td><td>81.82</td><td>83.60</td></tr><tr><td>LS-ViT (Hwang et al., 2026)</td><td>√</td><td>X</td><td>4/4</td><td>76.67</td><td>83.08</td><td>67.05</td><td>76.89</td><td>80.41</td><td>81.82</td><td>83.62</td></tr><tr><td>GLF-Q (Ours)</td><td>r</td><td>×</td><td>4/4</td><td>77.82</td><td>83.22</td><td>67.37</td><td>77.34</td><td>80.53</td><td>82.01</td><td>83.80</td></tr></table>

Efficient Block Reconstruction via STE. Unlike conventional reconstruction methods that introduce continuous relaxation variables and require tens of thousands of iterations, such as AdaRound (Nagel et al., 2020), we directly optimize full-precision weights $\theta _ { l }$ and quantization scales $s _ { l }$ using STE (Bengio et al., 2013) under the GLF loss in Eq. (6). This direct formulation converges within only 3000 iterations, significantly accelerating reconstruction while achieving superior low-bit accuracy.

## 4 EXPERIMENTS

We evaluate GLF-Q on image classification, object detection, and instance segmentation. We further conduct ablation studies, assess out-of-domain calibration robustness, and compare inference latency and training time. Additional experiments are provided in the appendix.

## 4.1 EXPERIMENTAL SETUP

Datasets and Models. For image classification, we evaluate on ImageNet (Deng et al., 2009) across ViT (Dosovitskiy et al., 2020), DeiT (Touvron et al., 2021), and Swin (Liu et al., 2021). For object detection and instance segmentation, we conduct experiments on COCO (Lin et al., 2014) using Mask R-CNN (He et al., 2017) and Cascade Mask R-CNN (Cai & Vasconcelos, 2018) with Swin backbones. All pretrained full-precision Vision Transformers are obtained from the timm library<sup>1</sup>, while pretrained detection and segmentation models are sourced from MMDetection (Chen et al., 2019).

Implementation Details. We apply standard channel-wise uniform quantization to weights and layer-wise uniform quantization to activations, including Softmax outputs, as detailed in Appendix A. Following reconstruction-based PTQ methods (Li et al., 2021; Wei et al., 2022; Wu et al., 2025b;a), we randomly sample 1024 unlabeled images from ImageNet for image classification and 256 unlabeled images from COCO for object detection and instance segmentation as the calibration sets. We set the batch size, learning rate for activation quantization, learning rate for tuning weight, and reconstruction iterations as 32, 2e-4, 2e-5, and 3000, respectively. The regularization coefficient λ in Eq. (6) is set to 2 across all experiments. For detection and segmentation, the global term averages the normalized bbox and mask RoI feature losses. For Swin, where hierarchical patch merging hinders exact RMSNorm conversion, we retain LayerNorm without residual Hadamard, pairing head-wise QK/VO Hadamard rotations with LayerNorm input reparameterization (Yang et al., 2024). More implementation details are provided in Appendix G.

## 4.2 EXPERIMENTS ON IMAGE CLASSIFICATION

We first evaluate the performance of our method on the classification task on ImageNet in terms of Top-1 accuracy, compared to the state-of-the-art PTQ approaches. We report results across various representative Transformer architectures, including ViT, DeiT, and Swin, under 4-bit and 3-bit settings.

As displayed in Table 1, GLF-Q achieves strong performance across diverse architectures, with particularly significant gains under 3-bit quantization. Concretely, for 4-bit quantization, while some existing methods suffer noticeable accuracy degradation, our method maintains robust performance across all architectures. Under the challenging 3-bit quantization, the performance of competing methods degrades severely, with methods such as PTQ4ViT (Yuan et al., 2022) and RepQ-ViT (Li et al., 2023) suffering severe accuracy degradation. In comparison, our proposed GLF-Q maintains robust accuracy, outperforming the second-best approach by 3.30% on ViT-S.

It is worth noting that many compared approaches such as PTQ4ViT (Yuan et al., 2022), RepQ ViT (Li et al., 2023), AdaLog (Wu et al., 2024), and DopQ-ViT (Yang et al., 2024) attempt to boost performance by designing specific quantizers, which however are generally difficult to implement on hardware in practice. In contrast, our method relies strictly on a standard uniform quantizer while achieving superior accuracy, facilitating hardware-friendly deployment on general-purpose platforms.

## 4.3 EXPERIMENTS ON OBJECT DETECTION AND INSTANCE SEGMENTATION

We evaluate 4-bit quantization on COCO (Lin et al., 2014) using Mask R-CNN (He et al., 2017) and Cascade Mask R-CNN (Cai & Vasconcelos, 2018) with Swin backbones. As shown in Table 2, GLF-Q achieves competitive performance across object detection and instance segmentation, obtaining the best results on several model configurations while using standard uniform quantization. These results demonstrate the effectiveness of GLF-Q on dense prediction tasks.

## 4.4 ABLATION STUDIES

On different quantization losses. We compare five reconstruction losses within the same 3-bit framework using STE for 3,000 iterations: local feature loss (LF), DPLR-FIM from FIMA-Q (Wu et al., 2025a), Logits + LF from PD-Quant (Liu et al., 2023), global feature loss (GF), and GLF in Eq. (6). We additionally evaluate FIMA-Q (Wu et al., 2025a) with MLP reconstruction and offline Hadamard transformations using AdaRound for 20,000 iterations, reported separately in Table 3. As shown in Table 3, GLF achieves the highest mean accuracy of 72.77%, exceeding DPLR-FIM, Logits + LF, and GF by 1.38%, 1.30%, and 0.98%, respectively. With MR and Hadamard, FIMA-Q achieves 71.59% mean accuracy using AdaRound for 20k iterations, 1.17% below GLF-Q using STE for 3k iterations.

On main components in GLF-Q. To evaluate the contribution of main components in our framework, we conduct an ablation study on MLP Reconstruction and Offline Hadamard Transformation under 3-bit quantization on ImageNet. As shown in Table 4, the full combination achieves the highest accuracy across all evaluated architectures. Adding MR to GLF improves accuracy on five of the seven architectures, with a slight decrease on DeiT-T and no change on DeiT-B. Adding Hadamard transformations improves accuracy across all seven architectures.

Table 2: Object detection and instance segmentation performance on COCO with Mask R-CNN and Cascade Mask R-CNN under 4-bit quantization.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Opt.</td><td rowspan="2">Spec.</td><td rowspan="2">W/A</td><td colspan="4">Mask R-CNN</td><td colspan="4">Cascade Mask R-CNN</td></tr><tr><td>Swin-T</td><td></td><td></td><td>Swin-S</td><td></td><td>Swin-T</td><td>Swin-S</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $\mathsf { A P } ^ { b }$ </td><td> $\mathbf { A P } ^ { m }$ </td><td> $\mathsf { A P } ^ { b }$ </td><td> $\mathbf { A P } ^ { m }$ </td><td> $\mathsf { A P } ^ { b }$ </td><td> $\mathbf { A P } ^ { m }$ </td><td> $\mathbf { A P } ^ { b }$ </td><td> $\mathbf { A P } ^ { m }$ </td></tr><tr><td>Full-Precision</td><td>-</td><td>-</td><td>32/32</td><td>46.0</td><td>41.6</td><td>48.5</td><td>43.3</td><td>50.4</td><td>43.7</td><td>51.9</td><td>45.0</td></tr><tr><td>PTQ4ViT (Yuan et al., 2022)</td><td>X</td><td>√</td><td>4/4</td><td>6.9</td><td>7.0</td><td>26.7</td><td>26.6</td><td>14.7</td><td>13.5</td><td>0.5</td><td>0.5</td></tr><tr><td>APQ-ViT (Ding et al., 2022)</td><td>X</td><td>√ √</td><td>4/4</td><td>23.7 36.1</td><td>22.6 36.0</td><td>44.7 44.2</td><td>40.1</td><td>27.2</td><td>24.4 41.1</td><td>47.7 49.3</td><td>41.1</td></tr><tr><td>RepQ-ViT (Li et al., 2023)</td><td>X</td><td>√</td><td>4/4</td><td></td><td>36.6</td><td>43.4</td><td>40.2 40.7</td><td>47.0</td><td>42.1</td><td></td><td>43.1</td></tr><tr><td>ERQ (Zhong et al., 2024)</td><td>X √</td><td>√</td><td>4/4 4/4</td><td>36.8 37.5</td><td>36.6</td><td>43.4</td><td>40.3</td><td>47.9 48.2</td><td>42.0</td><td>50.0 50.3</td><td>43.6</td></tr><tr><td>I&amp;S-ViT (Zhong et al., 2023)</td><td>√</td><td>√</td><td>4/4</td><td>37.5</td><td>36.5</td><td>43.5</td><td>40.4</td><td>48.2</td><td>42.1</td><td>50.3</td><td>43.6 43.7</td></tr><tr><td>DopQ-ViT (Yang et al., 2024)</td><td>√</td><td>X</td><td>4/4</td><td>36.2</td><td>35.4</td><td>41.6</td><td>39.2</td><td>47.0</td><td>41.3</td><td>49.0</td><td>42.5</td></tr><tr><td>QDrop* (Wei et al., 2022)</td><td>√</td><td>X</td><td>4/4</td><td>38.9</td><td>38.1</td><td>44.1</td><td>41.0</td><td>48.9</td><td>42.7</td><td>50.3</td><td>43.7</td></tr><tr><td>APHQ-ViT (Wu et al., 2025b)</td><td>√</td><td>X</td><td>4/4</td><td>38.7</td><td>37.8</td><td>44.2</td><td>41.1</td><td>48.7</td><td>42.5</td><td>50.4</td><td>43.7</td></tr><tr><td>FIMA-Q (Wu et al., 2025a) GLF-Q (Ours)</td><td>√</td><td>X</td><td>4/4</td><td>39.6</td><td>38.2</td><td>44.3</td><td>41.1</td><td>48.8</td><td>42.6</td><td>50.6</td><td>44.0</td></tr></table>

Table 3: Top-1 accuracy comparison (%) with different quantization losses and an additional baseline on ImageNet under 3-bit quantization. †: FIMA-Q + MR + Hadamard with AdaRound, 20k iterations.
<table><tr><td>Setting</td><td>ViT-S</td><td>ViT-B</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td>LF</td><td>60.48</td><td>77.06</td><td>53.16</td><td>70.16</td><td>76.76</td><td>76.66</td><td>78.46</td></tr><tr><td>DPLR-FIM (Wu et al., 2025a)</td><td>63.83</td><td>77.15</td><td>54.94</td><td>70.67</td><td>76.60</td><td>77.30</td><td>79.19</td></tr><tr><td>Logits + LF (Liu et al., 2023)</td><td>65.16</td><td>76.89</td><td>55.26</td><td>70.71</td><td>76.87</td><td>76.80</td><td>78.58</td></tr><tr><td>GF</td><td>66.09</td><td>77.17</td><td>54.06</td><td>70.41</td><td>76.69</td><td>77.75</td><td>80.30</td></tr><tr><td>GLF (Ours)</td><td>67.40</td><td>78.18</td><td>56.48</td><td>71.55</td><td>77.20</td><td>78.10</td><td>80.45</td></tr><tr><td> $\mathrm { F I M A { \mathrm { - } } Q + M R + H a d . } ^ { \dagger }$ </td><td>66.88</td><td>76.13</td><td>55.29</td><td>69.81</td><td>76.24</td><td>77.37</td><td>79.42</td></tr></table>

Table 4: Top-1 accuracy comparison (%) with different framework components on ImageNet under 3-bit quantization. “GLF” denotes Global-Local Feature loss, “MR” denotes MLP Reconstruction, and “Hadamard” denotes Offline Hadamard Transformation.
<table><tr><td>Method</td><td>ViT-S</td><td>ViT-B</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td>GLF only</td><td>60.37</td><td>72.50</td><td>55.28</td><td>68.75</td><td>75.79</td><td>76.92</td><td>79.31</td></tr><tr><td>GLF + MR</td><td>63.53</td><td>73.60</td><td>55.24</td><td>69.00</td><td>75.79</td><td>77.96</td><td>80.09</td></tr><tr><td>GLF + Hadamard</td><td>65.04</td><td>77.16</td><td>55.69</td><td>71.30</td><td>76.91</td><td>77.63</td><td>79.81</td></tr><tr><td>GLF-Q</td><td>67.40</td><td>78.18</td><td>56.48</td><td>71.55</td><td>77.20</td><td>78.10</td><td>80.45</td></tr></table>

## 4.5 EXPERIMENTS ON OUT-OF-DOMAIN CALIBRATION

Table 5 reports ImageNet accuracy after calibration on 1024 unlabeled SUN397 images (Xiao et al., 2010). GLF-Q outperforms FIMA-Q (Wu et al., 2025a) on all three models at both bit-widths, with gains of up to 31.97% on W3A3 DeiT-S. GLF also outperforms DPLR-FIM within our framework at both bit-widths.

Table 5: ImageNet top-1 accuracy (%) with SUN397 calibration. †: DPLR-FIM in our framework.
<table><tr><td rowspan="2">Method</td><td colspan="3">3-bit</td><td colspan="3">4-bit</td></tr><tr><td>ViT-S</td><td>DeiT-S</td><td>Swin-S</td><td>ViT-S</td><td>DeiT-S</td><td>Swin-S</td></tr><tr><td>FIMA-Q (Wu et al., 2025a)</td><td>41.86</td><td>36.03</td><td>69.95</td><td>73.37</td><td>75.89</td><td>80.72</td></tr><tr><td>DPLR-FIM†</td><td>47.80</td><td>65.58</td><td>70.77</td><td>73.88</td><td>76.02</td><td>80.51</td></tr><tr><td>GLF-Q (Ours)</td><td>58.72</td><td>68.00</td><td>74.71</td><td>75.34</td><td>76.46</td><td>81.07</td></tr></table>

Table 6: Comparison of inference latency and Top-1 accuracy under 8-bit quantization. Latency (ms) is measured on a single NVIDIA H800 GPU with a batch size of 64 using NVIDIA TensorRT (NVIDIA Corporation, 2024), and Top-1 accuracy is evaluated on ImageNet.
<table><tr><td rowspan="2">Method</td><td colspan="3">Latency</td><td rowspan="2"></td><td colspan="3">Top-1 Accuracy</td></tr><tr><td>ViT-S</td><td>DeiT-S</td><td>Swin-S</td><td>ViT-S</td><td>DeiT-S</td><td>Swin-S</td></tr><tr><td>FIMA-Q (Wu et al., 2025a)</td><td>2.95</td><td>2.95</td><td>6.83</td><td></td><td>79.19</td><td>78.72</td><td>83.06</td></tr><tr><td>GLF-Q (Ours)</td><td>2.50</td><td>2.49</td><td>5.88</td><td></td><td>80.35</td><td>79.07</td><td>83.08</td></tr></table>

Table 7: Comparison of training time (minutes) on ImageNet under 3-bit quantization.
<table><tr><td>Method</td><td>ViT-S</td><td>ViT-B</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td>FIMA-Q (Wu et al., 2025a)</td><td>48</td><td>87</td><td>44</td><td>49</td><td>88</td><td>115</td><td>135</td></tr><tr><td>LS-ViT (Hwang et al., 2026)</td><td>40</td><td>56</td><td>41</td><td>40</td><td>56</td><td>101</td><td>104</td></tr><tr><td>GLF-Q (Ours)</td><td>21</td><td>34</td><td>19</td><td>20</td><td>34</td><td>57</td><td>63</td></tr></table>

## 4.6 ANALYSIS OF INFERENCE EFFICIENCY

Since quantization below 8 bits requires specialized hardware (Li et al., 2021; Zhong et al., 2024), we evaluate real-world deployment under 8-bit quantization. As shown in Table 6, GLF-Q reduces inference latency by 15% on average compared to FIMA-Q (Wu et al., 2025a), while improving accuracy across all evaluated models. For ViT and DeiT models, RMSNorm removes the meancentering operation in LayerNorm, while ReLU simplifies activation computation compared with GELU. Swin models retain LayerNorm but also benefit from the simpler ReLU activation. These computational simplifications can contribute to the observed latency reductions. Deployment and timing details are provided in Appendix G.

## 4.7 ANALYSIS OF TRAINING EFFICIENCY

To evaluate practical training efficiency, we compare training time against FIMA-Q (Wu et al., 2025a) and LS-ViT (Hwang et al., 2026) under 3-bit quantization on ImageNet. All training times are measured on a single NVIDIA H800 GPU. For all methods, the reported time includes calibration and block reconstruction. For GLF-Q, it also includes MLP reconstruction. As shown in Table 7, GLF-Q is approximately 2.0 to 2.6 times faster than FIMA-Q and 1.6 to 2.2 times faster than LS-ViT across the seven architectures. GLF-Q converges within 3,000 iterations using simple STE, whereas FIMA-Q and LS-ViT use AdaRound with 20,000 reconstruction iterations.

## 5 CONCLUSION

In this paper, we proposed GLF-Q, an efficient post-training quantization framework for Vision Transformers. To address calibration overfitting in logit-guided reconstruction and the approximation limitations of Hessian-based objectives, we introduce the Global-Local Feature loss, which combines penultimate-layer feature alignment through the full-precision downstream network with local reconstruction regularization. Furthermore, offline Hadamard transformations disperse activation outliers to suppress clipping errors with zero runtime overhead, while basic STE optimization achieves rapid convergence. Extensive experiments demonstrate that GLF-Q with standard uniform quantizers substantially outperforms state-of-the-art methods under 3-bit quantization on image classification. GLF-Q also exhibits strong out-of-domain calibration robustness and delivers speedups under 8-bit GPU deployment.

## ACKNOWLEDGMENTS AND DISCLOSURE OF FUNDING

This work was partly supported by the National Natural Science Foundation of China under Grant 62276123, by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118), and by the Zhongguancun Academy Project No. XTS0066.

JW identified problems and proposed conjectures in optimization-based PTQ and guided PS in conducting the experiments. PS, with the help of JW, designed and implemented GLF-Q. GL and JT provided valuable assistance and contributed to discussions on method design and experiments. JW and PS wrote the paper.

## REFERENCES

Saleh Ashkboos, Maximilian Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. Slicegpt: Compress large language models by deleting rows and columns. In International Conference on Learning Representations, volume 2024, pp. 11682–11701, 2024a.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37:100213–100240, 2024b.

Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients´ through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 6154–6162. Ieee, 2018.

Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. In International Conference on Learning Representations, 2024.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Yifu Ding, Haotong Qin, Qinghua Yan, Zhenhua Chai, Junjie Liu, Xiaolin Wei, and Xianglong Liu. Towards accurate post-training quantization for vision transformer. In Proceedings of the 30th ACM international conference on multimedia, pp. 5380–5388, 2022.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. arXiv preprint arXiv:1902.08153, 2019.

Minghao Fu, Hao Yu, Jie Shao, Junjie Zhou, Ke Zhu, and Jianxin Wu. Quantization without tears. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4462– 4472. IEEE, 2025.

Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. In Low-power computer vision, pp. 291–326. Chapman and Hall/CRC, 2022.

Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask r-cnn. In´ Proceedings ofthe IEEE international conference on computer vision, pp. 2961–2969, 2017.

Hyunha Hwang, Xuan Truong Nguyen, and Hyuk-Jae Lee. Ls-vit: Least-squares hessian based block reconstruction for low-bit post-training quantization of vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 33588–33597, 2026.

Tao Jiang, Yucheng Jiang, Xiwen Yao, Gong Cheng, and Junwei Han. Uq-vit: Harmonizing extreme activations with hardware-friendly uniform quantization in vision transformers. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 22354–22362, 2026.

Yanjing Li, Sheng Xu, Baochang Zhang, Xianbin Cao, Peng Gao, and Guodong Guo. Q-vit: Accurate and fully quantized low-bit vision transformer. Advances in neural information processing systems, 35:34451–34463, 2022.

Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426, 2021.

Zhikai Li, Junrui Xiao, Lianwei Yang, and Qingyi Gu. Repq-vit: Scale reparameterization for posttraining quantization of vision transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17181–17190. IEEE, 2023.

Guang Liang, Xinyao Liu, and Jianxin Wu. Gplq: A general, practical, and lightning qat method for vision transformers. Advances in Neural Information Processing Systems, 38:7033–7057, 2026.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In´ European conference on computer vision, pp. 740–755. Springer, 2014.

Jiawei Liu, Lin Niu, Zhihang Yuan, Dawei Yang, Xinggang Wang, and Wenyu Liu. Pd-quant: Posttraining quantization based on prediction difference metric. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24427–24437. IEEE, 2023.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF international conference on computer vision (ICCV), pp. 9992–10002. Ieee, 2021.

Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. Outlier-aware slicing for post-training quantization in vision transformer. In Forty-first International Conference on Machine Learning, 2024.

Jaehyeon Moon, Dohyung Kim, Junyong Cheon, and Bumsub Ham. Instance-aware group quantization for vision transformers. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16132–16141. IEEE, 2024.

Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In International conference on machine learning, pp. 7197–7206. PMLR, 2020.

NVIDIA Corporation. NVIDIA TensorRT, 2024. https://developer.nvidia.com/ tensorrt.

Peilin Sun and Jianxin Wu. Nonlinear bipolar compensation: Handling outliers in post-training quantization. arXiv preprint arXiv:2605.16423, 2026.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J´ egou. Training data-efficient image transformers & distillation through attention. In´ International conference on machine learning, pp. 10347–10357. PMLR, 2021.

Guo-Hua Wang, Yifan Ge, and Jianxin Wu. Distilling knowledge by mimicking features. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):8183–8195, 2021.

Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu, and Fengwei Yu. Qdrop: Randomly dropping quantization for extremely low-bit post-training quantization. arXiv preprint arXiv:2203.05740, 2022.

Zhuguanyu Wu, Jiaxin Chen, Hanwen Zhong, Di Huang, and Yunhong Wang. Adalog: Post-training quantization for vision transformers with adaptive logarithm quantizer. In European Conference on Computer Vision, pp. 411–427. Springer, 2024.

Zhuguanyu Wu, Shihe Wang, Jiayi Zhang, Jiaxin Chen, and Yunhong Wang. Fima-q: Posttraining quantization for vision transformers by fisher information matrix approximation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14891–14900. IEEE, 2025a.

Zhuguanyu Wu, Jiayi Zhang, Jiaxin Chen, Jinyang Guo, Di Huang, and Yunhong Wang. Aphq-vit: Post-training quantization with average perturbation hessian based reconstruction for vision transformers. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9686–9695. IEEE, 2025b.

Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pp. 3485–3492. IEEE, 2010.

Lianwei Yang, Haisong Gong, Haokun Lin, Yichen Wu, Caifeng Shan, Zhenan Sun, and Qingyi Gu. Dopq-vit: towards distribution-friendly and outlier-aware post-training quantization for vision transformers. arXiv preprint arXiv:2408.03291, 2024.

Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, and Guangyu Sun. Ptq4vit: Post-training quantization for vision transformers with twin uniform quantization. In European conference on computer vision, pp. 191–207. Springer, 2022.

Yunshan Zhong, Jiawei Hu, Mengzhao Chen, Rongrong Ji, et al. I&s-vit: An inclusive & stable method for pushing the limit of post-training vits quantization. arXiv preprint arXiv:2311.10126, 2023.

Yunshan Zhong, Jiawei Hu, You Huang, Yuxin Zhang, and Rongrong Ji. Erq: Error reduction for post-training quantization of vision transformers. In Forty-first International Conference on Machine Learning, 2024.

## APPENDIX

## A UNIFORM QUANTIZER FORMULATION

For efficient hardware deployment, we employ the standard uniform quantizer. Given bit-width b, uniform quantization maps a continuous tensor x to its low-bit integer representation x<sup>q</sup>:

$$
x ^ { \mathfrak { q } } = \mathrm { c l i p } \left( \left\lfloor { \frac { x } { s } } \right\rceil + z , 0 , 2 ^ { b } - 1 \right) ,\tag{11}
$$

where $\lfloor \cdot \rceil$ denotes the round-to-nearest operator, $\mathrm { c l i p } ( \cdot , 0 , 2 ^ { b } - 1 )$ restricts values to $[ 0 , 2 ^ { b } - 1 ]$ $s \in \mathbb { R } ^ { + }$ is the quantization scale, and $z \in \mathbb { Z }$ is the zero-point offset:

$$
s = { \frac { \operatorname* { m a x } ( x ) - \operatorname* { m i n } ( x ) } { 2 ^ { b } - 1 } } , \quad z = \mathrm { c l i p } \left( \left\lfloor - { \frac { \operatorname* { m i n } ( x ) } { s } } \right\rceil , 0 , 2 ^ { b } - 1 \right) .\tag{12}
$$

The dequantized tensor xˆ approximates the original tensor x via:

$$
\hat { x } = s \times \left( x ^ { \mathrm { q } } - z \right) .\tag{13}
$$

## B COMPREHENSIVE DETAILS AND EQUIVALENCE PROOFS FOR OFFLINE HADAMARD TRANSFORMATIONS

This section describes the conversion from LayerNorm to RMSNorm, offline Hadamard transformations and their equivalence, and Hadamard construction for dimensions that are not powers of two. We use the row-vector convention throughout.

## B.1 LAYERNORM-TO-RMSNORM CONVERSION

Let C denote the channel dimension. Define the mean-centering matrix and centered activation as:

$$
M = I - \frac { 1 } { C } { \bf 1 1 } ^ { \top } \in \mathbb { R } ^ { C \times C } , \qquad x _ { \mathrm { c } } = x M ,\tag{14}
$$

where $\mathbf { 1 } \in \mathbb { R } ^ { C }$ is the all-ones vector. LayerNorm can be expressed using affine-free RMSNorm as:

$$
\mathrm { L N } ( x ) = \mathrm { R M S N o r m } ( x M ) \mathrm { D i a g } ( \gamma ) + \beta ,\tag{15}
$$

where $\gamma , \beta \in \mathbb { R } ^ { C }$ are the affine scaling and bias. RMSNorm retains the original ϵ, while the affine parameters are folded into the subsequent linear layer:

$$
\widetilde { W } = { \mathrm { D i a g } } ( \gamma ) W , \qquad \widetilde { b } = \beta W + b .\tag{16}
$$

For a standard pre-LN residual sublayer $x ^ { + } = x + F ( \mathrm { L N } ( x ) )$ , post-centering yields $x _ { \mathrm { c } } ^ { + } = x ^ { + } M =$ $( x + F ( \mathrm { L N } ( x ) { \dot { ) } } ) M = x _ { \mathrm { c } } + F ( \mathrm { L N } ( { \dot { x } } ) ) M$ . To maintain $x _ { \mathrm { c } , l } = x _ { l } M$ across layers, we fold centering into the patch embedding and the output projections of attention and MLP branches:

$$
\begin{array} { r } { E _ { \mathrm { p a t c h } }  E _ { \mathrm { p a t c h } } M , \quad b _ { E }  b _ { E } M , } \\ { W _ { \mathrm { o u t } }  W _ { \mathrm { o u t } } M , \quad b _ { \mathrm { o u t } }  b _ { \mathrm { o u t } } M . } \end{array}
$$

The class token and positional embeddings are also centered as $t  t M$ and $p \gets p M$ . These updates preserve the features after the final normalization.

## B.2 RESIDUAL HADAMARD TRANSFORMATIONS

Following the computational invariance established in SliceGPT (Ashkboos et al., 2024a), we apply a shared rotation to the residual stream. We select an orthogonal transformation matrix $\boldsymbol { R _ { 1 } } \in \mathbb { R } ^ { \boldsymbol { \hat { C } } \times \boldsymbol { \bar { C } } }$ satisfying $R _ { 1 } ^ { \top } R _ { 1 } = R _ { 1 } R _ { 1 } ^ { \top } = I$ and define the rotated residual activation as:

$$
x _ { \mathrm { c } } ^ { \mathrm { r o t } } = x _ { \mathrm { c } } R _ { 1 } .\tag{17}
$$

Affine-free RMSNorm satisfies orthogonal equivariance (Ashkboos et al., 2024b):

$$
\mathrm { R M S N o r m } ( x _ { \mathrm { c } } R _ { 1 } ) = \frac { x _ { \mathrm { c } } R _ { 1 } } { \sqrt { \frac { 1 } { C } \| x _ { \mathrm { c } } R _ { 1 } \| _ { 2 } ^ { 2 } + \epsilon } } = \frac { x _ { \mathrm { c } } R _ { 1 } } { \sqrt { \frac { 1 } { C } \| x _ { \mathrm { c } } \| _ { 2 } ^ { 2 } + \epsilon } } = \mathrm { R M S N o r m } ( x _ { \mathrm { c } } ) R _ { 1 } ,\tag{18}
$$

Table 8: Parameter transformation rules for the centered and affine-folded Vision Transformer.
<table><tr><td>Network Position</td><td>Parameter Transformation Rule</td></tr><tr><td>Input Patch Embedding</td><td> $E _ { \mathrm { p a t c h } }  E _ { \mathrm { p a t c h } } R _ { 1 } , \quad b _ { E }  b _ { E } R _ { 1 }$ </td></tr><tr><td>Class Token &amp; Positional Embeddings</td><td> $t  t R _ { 1 } , \quad p  p R _ { 1 }$ </td></tr><tr><td>QKV Projection &amp; MLP First Layer</td><td> $W _ { \mathrm { i n } } \to R _ { 1 } ^ { \top } W _ { \mathrm { i n } } , \quad b _ { \mathrm { i n } }$  (unchanged)</td></tr><tr><td>Attention Output Projection &amp; MLP Second Layer</td><td> $W _ { \mathrm { o u t } } \to W _ { \mathrm { o u t } } R _ { 1 } , \quad b _ { \mathrm { o u t } } \to b _ { \mathrm { o u t } } R _ { 1 }$ </td></tr><tr><td>Final Classification Head</td><td> $W _ { \mathrm { h e a d } } \to R _ { 1 } ^ { \top } W _ { \mathrm { h e a d } } , \quad b _ { \mathrm { h e a d } } ( \mathrm { u n c h a n g e d } )$ </td></tr></table>

because orthogonal transformations preserve Euclidean norms. Table 8 summarizes the corresponding offline parameter transformations.

Input and output projections absorb $R _ { 1 } ^ { \top }$ and $R _ { 1 }$ , respectively, so both residual branches share the same rotation. RMSNorm equivariance allows this rotation to propagate through the network, while the classification head absorbs its inverse, preserving the logits.

## B.3 HEAD-WISE QK AND VO TRANSFORMATIONS

Head-wise QK Rotation. Let $Q _ { h } , K _ { h } \in \mathbb { R } ^ { N \times C _ { h } }$ denote the query and key states of attention head h, where $C _ { h } = C / N _ { h }$ . For an orthogonal rotation $R _ { 2 , h } \in \mathbb { R } ^ { C _ { h } ^ { \mathbf { i } } \times C _ { h } }$ , we define $Q _ { h } ^ { \prime } = Q _ { h } R _ { 2 , h }$ and $K _ { h } ^ { \prime } = K _ { h } R _ { 2 , h }$ , with the rotation folded into the query and key projections offline. Then

$$
\begin{array} { r } { Q _ { h } ^ { \prime } { K _ { h } ^ { \prime } } ^ { \top } = ( Q _ { h } R _ { 2 , h } ) ( K _ { h } R _ { 2 , h } ) ^ { \top } = Q _ { h } R _ { 2 , h } R _ { 2 , h } ^ { \top } K _ { h } ^ { \top } = Q _ { h } K _ { h } ^ { \top } . } \end{array}\tag{19}
$$

Because the inner products entering Softmax remain unchanged, the resulting attention probabilities are also unchanged.

Head-wise VO Rotation. Let $P _ { h } \in { \mathbb R } ^ { N \times N }$ denote the attention probability matrix, $V _ { h } \in \mathbb { R } ^ { N \times C _ { h } }$ the value states, and $W _ { O , h } \in \mathbb { R } ^ { C _ { h } \times C }$ the row slice of the output projection corresponding to head h. Under the row-vector convention, this head contributes $( P _ { h } \dot { V } _ { h } ) \dot { W } _ { O , h }$ to the attention output. For an orthogonal rotation $R _ { 3 , h } \in \mathbb { R } ^ { C _ { h } \times C _ { h } }$ , we define $V _ { h } ^ { \prime } = V _ { h } R _ { 3 , h }$ and fold the inverse rotation into the output projection as $W _ { O , h } ^ { \prime } = R _ { 3 , h } ^ { \top } W _ { O , h }$ . Then

$$
( P _ { h } V _ { h } ^ { \prime } ) W _ { O , h } ^ { \prime } = ( P _ { h } V _ { h } R _ { 3 , h } ) ( R _ { 3 , h } ^ { \top } W _ { O , h } ) = P _ { h } V _ { h } W _ { O , h } .\tag{20}
$$

Summing these contributions over all heads and retaining the output projection bias preserves the multi-head self-attention output.

## B.4 RANDOMIZED HADAMARD CONSTRUCTION

Following QuaRot (Ashkboos et al., 2024b), we construct normalized randomized Hadamard matrices for the dimensions used in our models. For $d = 2 ^ { k }$ , we start with $H _ { 1 } = [ 1 ]$ and recursively construct

$$
H _ { 2 n } = { \left[ \begin{array} { l l } { H _ { n } } & { H _ { n } } \\ { H _ { n } } & { - H _ { n } } \end{array} \right] }
$$

until the dimension reaches d. We then define the normalized randomized Hadamard matrix as

$$
R _ { d } = { \frac { D H _ { d } } { \sqrt { d } } } ,\tag{21}
$$

where D is a diagonal matrix with independent entries uniformly sampled from $\{ - 1 , + 1 \}$

For $d = 1 2 \cdot 2 ^ { k }$ , we combine a fixed Hadamard matrix $H _ { 1 2 }$ with $H _ { 2 ^ { k } }$ using the Kronecker product:

$$
R _ { d } = \frac { D \left( H _ { 1 2 } ^ { \top } \otimes H _ { 2 k } \right) } { \sqrt { d } } .\tag{22}
$$

This construction supports hidden dimensions such as 192, 384, and 768 without padding or truncation. Since $H _ { m } ^ { \top } H _ { m } = m I _ { m }$ and $D ^ { \top } D = I _ { d }$ , both constructions satisfy $R _ { d } ^ { \top } R _ { d } = I _ { d }$

## C OVERALL PIPELINE OF GLF-Q

The overall pipeline of GLF-Q is summarized in Algorithm 1.

Algorithm 1 Overall Pipeline of GLF-Q   
Require: Pretrained model M; unlabeled calibration data $\mathcal { D } _ { \mathrm { c a l } } ;$ bit-widths $( b _ { w } , b _ { a } )$ ; reconstruction   
iterations $T = 3 0 0 0 ;$ regularization coefficient $\lambda = 2 .$   
Ensure: Quantized model ${ \widehat { \mathcal { M } } } .$   
1: Initialize teacher $\mathcal { M } _ { \mathrm { f p } }$ and student $\mathcal { M } _ { \mathrm { s } }$ from $\mathcal { M } .$   
2: Apply identical offline transformations to both models following Section 3.3 and the   
architecture-specific settings in Section 4.1.   
3: Freeze $\mathcal { M } _ { \mathrm { f p } } ;$ replace student MLP GELU with ReLU and reconstruct against $\mathcal { M } _ { \mathrm { f p } }$   
4: Insert and calibrate student uniform quantizers; apply the Swin reparameterization when appli   
cable.   
5: for each reconstruction block l in forward order do   
6: Cache student inputs $\hat { x } _ { l }$ from the quantized prefix and teacher targets $( y _ { l } , f )$ on $\mathcal { D } _ { \mathrm { c a l } }$   
7: Let $F _ { l }$ be the frozen teacher suffix; optimize only the current student block.   
8: for $t = 1 , \dots , T$ do   
9: Sample matching batches of $( \hat { x } _ { l } , y _ { l } , f )$   
10: Compute the current quantized block output $\hat { y } _ { l }$ from $\hat { x } _ { l }$ and obtain $\hat { f } = F _ { l } ( \hat { y } _ { l } )$   
11: Evaluate $\mathcal { L } _ { \mathrm { g l o b a l } } ^ { ( l ) }$ and $\bar { \mathcal { L } } _ { \mathrm { l o c a l } } ^ { ( l ) }$ on the minibatch using Eqs. (4) and (5).   
12: $\mathbf { i f } t = 1$ then   
13: Cache the detached loss values, each clamped below at $1 0 ^ { - 1 2 }$ , as fixed normalization   
factors.   
14: end if   
15: Normalize both loss terms by their cached factors and combine them using Eq. (6).   
16: Backpropagate through $F _ { l }$ and the quantizers using STE; update $\theta _ { l }$ and $s _ { l }$ with Adam.   
17: end for   
18: end for

## D EFFECT OF LOSS FUNCTIONS ON GENERALIZATION

To evaluate generalization under limited calibration data, Table 9 compares the top-1 accuracy of logit and feature supervision on the calibration set and the ImageNet validation set. Logits + LF combines logit supervision with local feature regularization from PD-Quant (Liu et al., 2023), while GLF denotes the proposed Global-Local Feature loss. Both variants follow the 3-bit setting in Table 3 and differ only in the loss function.

Table 9: Top-1 accuracy (%) on the calibration set and ImageNet validation set under 3-bit quantization.
<table><tr><td rowspan="2">Model</td><td colspan="2">Logits + LF</td><td colspan="2">GLF</td></tr><tr><td>Calibration</td><td>Validation</td><td>Calibration</td><td>Validation</td></tr><tr><td>ViT-S</td><td>86.91</td><td>65.16</td><td>87.70</td><td>67.40</td></tr><tr><td>ViT-B</td><td>91.11</td><td>76.89</td><td>91.11</td><td>78.18</td></tr><tr><td>DeiT-T</td><td>77.05</td><td>55.26</td><td>76.66</td><td>56.48</td></tr><tr><td>DeiT-S</td><td>90.04</td><td>70.71</td><td>89.65</td><td>71.55</td></tr><tr><td>DeiT-B</td><td>96.00</td><td>76.87</td><td>96.00</td><td>77.20</td></tr><tr><td>Swin-S</td><td>87.40</td><td>76.80</td><td>87.70</td><td>78.10</td></tr><tr><td>Swin-B</td><td>92.38</td><td>78.58</td><td>92.48</td><td>80.45</td></tr><tr><td>Mean</td><td>88.70</td><td>71.47</td><td>88.76</td><td>72.77</td></tr></table>

With nearly identical mean calibration accuracy (88.76% for GLF and 88.70% for Logits + LF), GLF yields higher validation accuracy across all seven architectures. On average, GLF improves validation accuracy by 1.30% and narrows the performance gap between the calibration and validation sets from 17.23% to 15.99%. These results suggest better generalization with GLF under limited calibration data.

## E STABILITY ACROSS RANDOM SEEDS

Table 10 reports multi-seed ImageNet top-1 accuracy for GLF-Q across seven architectures under 3-bit and 4-bit quantization.

Table 10: Multi-seed top-1 accuracy (%) of GLF-Q on ImageNet under 3-bit and 4-bit quantization, reported as mean ± standard deviation over three random seeds.
<table><tr><td>W/A</td><td>ViT-S</td><td>ViT-B</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td>3/3</td><td> $6 7 . 3 4 \pm 0 . 1 2$ </td><td> $7 8 . 2 1 \pm 0 . 0 4$ </td><td> $5 6 . 1 7 \pm 0 . 2 7$ </td><td> $7 1 . 6 7 \pm 0 . 1 0$ </td><td> $7 7 . 2 2 \pm 0 . 1 0$ </td><td> $7 8 . 0 7 \pm 0 . 0 6$ </td><td> $8 0 . 4 4 \pm 0 . 0 3$ </td></tr><tr><td>4/4</td><td> $7 7 . 6 9 \pm 0 . 1 5$ </td><td> $8 3 . 1 2 \pm 0 . 1 0$ </td><td> $6 7 . 2 2 \pm 0 . 1 6$ </td><td> $7 7 . 4 0 \pm 0 . 0 6$ </td><td> $8 0 . 5 1 \pm 0 . 0 4$ </td><td> $8 2 . 0 5 \pm 0 . 0 8$ </td><td> $8 3 . 8 1 \pm 0 . 0 1$ </td></tr></table>

## F ADDITIONAL ABLATION STUDIES

Sensitivity to the regularization coefficient. We evaluate GLF-Q with $\lambda \in \{ 1 , 2 , 4 \}$ under 3-bit quantization on ImageNet. As shown in Table 11, mean top-1 accuracy across seven architectures ranges from 72.77% to 72.81%. These results suggest limited sensitivity to λ within the evaluated range.

Table 11: Top-1 accuracy (%) of GLF-Q with different regularization coefficients under 3-bit quantization on ImageNet.
<table><tr><td>λ</td><td>ViT-S</td><td>ViT-B</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-S</td><td>Swin-B</td><td>Mean</td></tr><tr><td>1</td><td>67.53</td><td>78.29</td><td>56.31</td><td>71.47</td><td>77.33</td><td>78.22</td><td>80.49</td><td>72.81</td></tr><tr><td>2</td><td>67.40</td><td>78.18</td><td>56.48</td><td>71.55</td><td>77.20</td><td>78.10</td><td>80.45</td><td>72.77</td></tr><tr><td>4</td><td>67.48</td><td>78.18</td><td>56.61</td><td>71.59</td><td>77.22</td><td>78.14</td><td>80.35</td><td>72.80</td></tr></table>

## G IMPLEMENTATION DETAILS

## G.1 MLP RECONSTRUCTION

MLP reconstruction is performed block-wise in floating-point precision before quantization to compensate for the error introduced by replacing GELU with ReLU. The teacher retains GELU, while the student uses ReLU. Both use identical coordinate transformations. Each student MLP is optimized using cached teacher inputs x and outputs y with the objective

$$
\mathcal { L } _ { \mathrm { M R } } = \frac { 1 } { N C } \left. f _ { \mathrm { R e L U } } ( x ) - y \right. _ { F } ^ { 2 } + \frac { 2 } { N C } \left. f _ { \mathrm { R e L U , c l i p } } ( x ) - y \right. _ { F } ^ { 2 } ,\tag{23}
$$

where N and C denote the number of output tokens and channels, respectively. The function $f _ { \mathrm { R e L U , c l i p } }$ applies upper clipping to the student’s FC2 input, with a fixed threshold for each block set to the 99th percentile of positive activations at the corresponding teacher FC2 input over the calibration data.

Each MLP is optimized for 20,000 iterations using Adam without weight decay. The learning rate follows an independent cosine schedule for each block, decaying from $4 \times \mathrm { i 0 ^ { - 5 } }$ to zero. Only the weights and biases of the current student MLP’s FC1 and FC2 layers are updated. All other parameters remain frozen.

## G.2 OBJECT DETECTION AND SEGMENTATION

To extend GLF-Q to object detection and instance segmentation, we use intermediate bbox and mask RoI features before the prediction layers as global supervision targets. The bbox and mask branches use up to 1000 and 128 student RPN proposals per image, respectively, preserving their original order. Teacher and student paths use identical RoI coordinates for each loss. The proposals and teacher reference features are cached once and reused throughout reconstruction.

Backbone reconstruction combines local, bbox RoI, and mask RoI feature MSE losses with weights of 2, 0.5, and 0.5, respectively. Each loss term is normalized by its value at the first reconstruction iteration, with the normalization factor held fixed for that block. FPN reconstruction uses only the normalized local MSE with a weight of 2, while the RPN and RoI heads undergo calibration-only quantization.

## G.3 TENSORRT DEPLOYMENT AND LATENCY MEASUREMENT

To support TensorRT’s standard INT8 path, asymmetric quantization representations that are not directly compatible are exported in floating-point form and requantized using a common scheme. We use symmetric absmax quantization per output channel for weights and per tensor for activations.

All latencies are measured on the same NVIDIA H800 80GB GPU using TensorRT 10.9, with an input resolution of 224 × 224 and a batch size of 64. After 50 warm-up runs, we time 200 inference runs using CUDA events and report the mean batch latency. Timing covers only GPU engine execution, excluding data loading, preprocessing, and data transfers. Engines are built with INT8 enabled and FP32/TF32 fallback allowed.