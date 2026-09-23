# LATENT DATASET DISTILLATION FOR HUMAN MOTION PREDICTION

Ge Tian Guang Li<sup>∗</sup> Takahiro Ogawa Miki Haseyama

Hokkaido University {tian, guang, ogawa, mhaseyama}@lmd.ist.hokudai.ac.jp

## ABSTRACT

Dataset distillation (DD) compresses a large training set into a compact synthetic set while preserving downstream training utility. Although DD has been widely studied for images and recently extended to time-series forecasting, its application to human motion prediction remains largely unexplored. Human motion is highdimensional and structurally coupled, and gradient matching (GM) in the original motion space optimizes many correlated variables without a prior on pose plausibility or temporal dynamics, which frequently yields implausible and unstable synthetic motions. To address this limitation, we propose a latent DD framework that regularizes distillation with a learned motion prior. Motions are first compressed by a residual-quantized variational autoencoder (RVQ-VAE), and distillation then updates only a learnable latent bank through the frozen quantizer and decoder. The pretrained decoder restricts synthetic motions to its output space, while residual quantization progressively refines the latent approximation across multiple codebooks and alleviates the representational bottleneck of single-stage vector quantization. Experiments on Human3.6M, CMU, and 3DPW with two prediction backbones show that the proposed framework outperforms direct GM in 27 of 30 evaluated settings and random subsets in every setting, and produces visibly more plausible synthetic motions in qualitative comparisons.

Index Terms— Dataset Distillation, Human Motion Prediction, Gradient Matching, RVQ-VAE

## 1. INTRODUCTION

Dataset distillation (DD) replaces a large training set with a compact synthetic set that preserves downstream training utility [1, 2]. Gradient matching (GM) synthesizes such data by aligning the parameter gradients induced by real and synthetic samples [3], and later studies improve condensation through differentiable augmentation [4], distribution matching [5], trajectory matching [6], synthetic-data parameterization [7], and representative matching [8]. The resulting synthetic sets have enabled applications such as privacy-preserving medical data sharing [9]. All of these methods were developed for images, where a synthetic sample is a freely optimized pixel array with a fixed layout.

DD has recently been extended to temporal data through frequency and trajectory matching [10], harmonic matching for forecasting [11], and spatio-temporal condensation [12]. Human motion, however, differs fundamentally from generic time series. Each frame is a pose whose parameters are coupled by the kinematic structure of the body, and motion predictors model the temporal dependencies among these parameters using recurrent [13], graph-based [14, 15], attention-based [16, 17], and MLP [18] architectures. Training such predictors is routinely repeated across architectures and horizons, so a compact synthetic set that retains training utility is of practical interest.

Despite this potential, applying GM directly to motion data is problematic. Direct GM treats every value of a synthetic sequence as an independent variable and updates it using gradient signals from the predictor alone. For articulated motion, this means searching over many strongly correlated rotation parameters without a prior on which joint configurations and temporal transitions are plausible. Distilled motions then exhibit implausible joint configurations and abrupt transitions, training becomes unstable, and the resulting sets can be inferior even to random subsets.

A natural solution is to regularize distillation with a learned motion prior rather than optimizing freely in the original space. Latentspace DD reduces the optimization burden of synthetic data for images [19], and discrete representations learned by VQ-VAE [20] are effective for motion modeling [21]. A single codebook, however, imposes a restrictive bottleneck that limits how well a synthetic set adapts to the distillation objective, whereas residual vector quantization (RVQ) represents a latent vector as a sum of codewords drawn progressively from multiple codebooks and relieves this bottleneck while retaining a structured motion prior [22]. Building on this observation, we pretrain an RVQ-VAE motion autoencoder and distill in its pre-quantization latent space, updating only a learnable latent bank by GM through the frozen quantizer and decoder, and evaluate the framework on three benchmarks with two prediction backbones.

Our contributions are summarized as follows.

• We study dataset distillation for human motion prediction and show that direct GM in the original motion space, lacking a learned motion prior, yields implausible synthetic motions.

• We propose a latent distillation framework built on a pretrained RVQ-VAE, in which only a learnable latent bank is optimized by GM through the frozen residual quantizer.

• We show empirically that, under a shared objective, distilling in the residual-quantized latent space outperforms direct GM as well as continuous and single-stage latent variants.

## 2. METHODOLOGY

## 2.1. Preliminaries

Gradient matching. GM constructs synthetic data by matching the parameter gradients of real and synthetic samples. A sample is divided into an observed sequence $\dot { P } \in \mathbb { R } ^ { T _ { \mathrm { i n } } \times D }$ and a future sequence $F \in \mathbb { R } ^ { T _ { \mathrm { o u t } } \times D }$ , where D is the dataset-specific feature dimension. Given a backbone $f _ { \theta }$ and prediction objective ℓ, the gradients induced by real and synthetic samples are $g _ { r } = \nabla _ { \theta } \ell ( f _ { \theta } ( P _ { r } ) , F _ { r } )$ and $g _ { s } = \nabla _ { \theta } \ell ( f _ { \theta } ( P _ { s } ) , F _ { s } )$ . We minimize the normalized discrepancy between the two,

![](images/7456ac4e47f0123d54cfbd6443a7129ff8cd42ab638ff5f107e1b9f19f2165c2.jpg)  
Fig. 1. An illustration of the proposed latent dataset distillation framework. Stage I learns a compact motion representation with an RVQ-VAE. In Stage II, real segments are encoded to initialize a learnable latent bank H, which is the only quantity optimized while the RVQ-VAE remains frozen.

$$
\mathcal { L } _ { \mathrm { G M } } = \frac { \sum _ { p } \left. \boldsymbol { g } _ { s } ^ { ( p ) } - \mathrm { s g } ( \boldsymbol { g } _ { r } ^ { ( p ) } ) \right. _ { 2 } ^ { 2 } } { \sum _ { p ^ { \prime } } \left. \mathrm { s g } ( \boldsymbol { g } _ { r } ^ { ( p ^ { \prime } ) } ) \right. _ { 2 } ^ { 2 } + \epsilon } ,\tag{1}
$$

where $p$ and $p ^ { \prime }$ index parameter tensors and $\operatorname { s g } ( \cdot )$ denotes stopgradient. The real gradients are fixed targets, whereas the synthetic branch retains the computation graph so that the synthetic representation can be optimized. Unlike the original GM formulation, which interleaves synthetic-data updates with network training, the backbone is held fixed as a shared gradient probe.

Motion data format. Each motion sequence is represented as $X = [ x _ { 1 } , \dots , x _ { T } ]$ ] with $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { D }$ , where $x _ { t }$ is the pose at time $t , \mathbf A$ 35-frame window is divided into $T _ { \mathrm { i n } } = 2 5$ observed and $T _ { \mathrm { o u t } } = 1 0$ future frames, and each synthetic trajectory contains 50 frames so that several prediction windows can be extracted from it.

## 2.2. Residual-Quantized Motion Representation

Direct GM optimizes motion features without an explicit structural prior. To regularize the optimization with a learned motion prior, we first train an autoencoder that combines a stochastic VAE encoder with residual vector quantization.

Stochastic latent encoder. Given a motion segment $X \in$ $\mathbb { R } ^ { T \times D }$ , the encoder produces the mean and log-variance of a latent distribution, $( \mu , \log \bar { \sigma } ^ { 2 } ) = E _ { \phi } ( X )$ , from which $z = \mu + \sigma \odot \varepsilon$ is sampled with $\varepsilon \sim \mathcal { N } ( 0 , I )$ . Temporal convolutions reduce the temporal resolution and map each latent step to a 64-D vector.

Residual vector quantization. Instead of quantizing z with a single codebook, RVQ progressively represents the remaining residual. Starting from $r ^ { ( 0 ) } = \bar { z } ,$ , the k-th codebook performs

$$
\boldsymbol { q } ^ { ( k ) } = \arg \operatorname* { m i n } _ { e \in \mathcal { C } ^ { ( k ) } } \left\| \boldsymbol { r } ^ { ( k - 1 ) } - e \right\| _ { 2 } ^ { 2 } , \qquad \boldsymbol { r } ^ { ( k ) } = \boldsymbol { r } ^ { ( k - 1 ) } - \mathrm { s g } ( \boldsymbol { q } ^ { ( k ) } ) ,\tag{2}
$$

and the final quantized representation is $\begin{array} { r } { Q ( z ) = \sum _ { k = 1 } ^ { K } q ^ { ( k ) } } \end{array}$ . Each stage refines the approximation left by the preceding one, so the representation capacity grows with K instead of being fixed by a single codebook. Quantizers of this form were introduced for neural audio coding [23] and later used for motion synthesis [22]. We exploit the same property to obtain a distillation space compact enough to regularize GM yet expressive enough for the motions it represents.

Pretraining objective. Since the nearest-codeword selection is not differentiable, a straight-through estimator passes gradients from the decoder to the continuous latent representation during training. The RVQ-VAE is optimized using $\begin{array} { r } { \mathcal { L } _ { \mathrm { R V Q } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { v } \mathcal { L } _ { \mathrm { v e l } } ^ { \mathrm { r e c } } + } \end{array}$ L<sub>codebook</sub> + L<sub>commit</sub> + $\lambda _ { \mathrm { K L } } ( t ) \mathcal { L } _ { \mathrm { K L } }$ , where $\mathcal { L } _ { \mathrm { r e c } }$ reconstructs the motion features, $\mathcal { L } _ { \mathrm { v e l } } ^ { \mathrm { r e c } }$ reconstructs their first-order differences with weight $\lambda _ { v }$ (not to be confused with the distillation term $\mathcal { L } _ { \mathrm { v e l } }$ in Eq. (4)), L<sub>codebook</sub> and $\mathcal { L } _ { \mathrm { c o m m i t } }$ are the codebook and commitment losses of VQ-VAE [20] summed over the K stages, and $\lambda _ { \mathrm { K L } } ( t )$ is an annealed KL weight. The KL term pulls the pre-quantization latent toward a standard Gaussian, bounding the scale of the space in which distillation later operates. After pretraining, the encoder, codebooks, and decoder are frozen.

## 2.3. Latent Dataset Distillation

The framework that performs distillation over the pretrained latent space is illustrated in Fig. 1.

Latent-bank initialization. For a budget of M synthetic trajectories, we first select real 50-frame segments $\{ X _ { i } ^ { ( 0 ) } \} _ { i = 1 } ^ { M }$ and initialize a latent bank $\mathcal { H } = \{ h _ { i } \} _ { i = 1 } ^ { M }$ using the encoder mean branch, $h _ { i } ^ { ( 0 ) } \ = \ E _ { \mu } ( X _ { i } ^ { ( 0 ) } )$ . A trajectory ${ \widetilde { X } } _ { i } ~ = ~ D _ { \psi } ( Q ( h _ { i } ) )$ is then decoded from each slot with the frozen RVQ and decoder, so distillation searches over H rather than over all motion values.

Gradient flow through the frozen quantizer. Because nearestcodeword selection is piecewise constant in $h _ { i } ,$ its gradient is zero almost everywhere and gives no optimization signal. During distillation, we therefore apply the same straight-through estimator to the frozen quantizer,

$$
Q _ { \mathrm { S T } } ( h ) = h + \mathrm { s g } ( Q ( h ) - h ) ,\tag{3}
$$

so that the forward pass decodes $Q ( h )$ while the gradient of the distillation objective reaches h directly.

Motion-space regularization. At each iteration, a batch of 50-frame real segments is sampled from the full training pool, and synthetic segments are decoded from randomly selected latent slots, so the sequence or action used for initialization does not determine the real–synthetic pairing. For each pair, 35-frame windows $W _ { r } ~ = ~ [ P _ { r } , \bar { F _ { r } } ]$ and $\hat { W _ { s } } ~ = ~ \ [ P _ { s } , F _ { s } ]$ are extracted with the same random temporal offset and passed to the shared backbone to compute L<sub>GM</sub>. We further regularize the decoded motion using $\mathcal { L } _ { \mathrm { p o s e } } \ = \ \mathrm { M S E } ( W _ { s } , W _ { r } )$ and $\bar { \mathcal { L } } _ { \mathrm { v e l } } ~ = ~ \mathrm { M S E } ( \Delta W _ { s } , \Delta W _ { r } )$ where $\Delta W _ { t } ~ = ~ W _ { t + 1 } - W _ { t }$ These regularizers do not assume semantic correspondence between paired windows. Under random pairing, their expectation splits into a term minimized at the mean real window and a term independent of $W _ { s } ,$ so they act as a meanshrinkage regularizer that anchors the decoded motion to the bulk of the real distribution. Note that $\mathcal { L } _ { \mathrm { v e l } }$ operates on differences of rotation parameters rather than physical angular velocities.

Table 1. Comparisons at the 1× synthetic-data budget on Human3.6M, CMU, and 3DPW. Results are MPJPE (mm) at frames #2/#4/#8/#10/#14, given as mean±std over five runs, and Mean5 is the average of the five horizon-wise means. Best compressed-set results are in bold, and Full is trained under the same fixed budget and serves as a reference.
<table><tr><td>Dataset</td><td>Method</td><td>#2</td><td>#4</td><td>#8</td><td>#10</td><td>#14</td><td>Mean5</td></tr><tr><td colspan="8">(a) Simple</td></tr><tr><td rowspan="5">H36M</td><td>Full</td><td>7.48±0.37</td><td>15.14±0.67</td><td>31.72±1.08</td><td>40.72±1.19</td><td>58.98±1.86</td><td>30.81</td></tr><tr><td>Random</td><td>17.72±1.22</td><td>34.88±3.17</td><td>68.34±4.71</td><td>82.98±6.00</td><td>107.94±7.66</td><td>62.37</td></tr><tr><td>GM</td><td>45.84±10.42</td><td>56.12±16.20</td><td>92.16±18.85</td><td>114.08±12.39</td><td>119.52±14.32</td><td>85.54</td></tr><tr><td>Ours</td><td>13.78±0.60</td><td>26.50±0.89</td><td>47.38±1.27</td><td>57.04±1.81</td><td>73.94±1.13</td><td>43.73</td></tr><tr><td>Full</td><td>11.82±1.27</td><td>23.09±2.26</td><td>47.62±3.08</td><td>60.56±2.80</td><td>87.90±2.93</td><td>46.20</td></tr><tr><td rowspan="4">CMU</td><td>Random</td><td>30.54±2.63</td><td>58.65±6.83</td><td>111.07±12.96</td><td>131.82±14.65</td><td>176.85±13.34</td><td>101.79</td></tr><tr><td>GM</td><td>32.36±5.48</td><td>51.58±7.41</td><td>88.38±11.10</td><td>108.54±15.44</td><td>133.25±12.93</td><td>82.82</td></tr><tr><td>Ours</td><td>21.41±0.99</td><td>41.47±1.31</td><td>77.03±2.49</td><td>92.56±3.55</td><td>118.73±5.00</td><td>70.24</td></tr><tr><td>Full</td><td>18.83±2.77</td><td>34.94±3.28</td><td>67.34±7.44</td><td>79.32±8.14</td><td>100.85±13.81</td><td>60.26</td></tr><tr><td rowspan="4">3DPW</td><td>Random</td><td>38.18±11.20</td><td>68.57±19.25</td><td>112.72±30.57</td><td>130.32±31.00</td><td>155.73±35.67</td><td>101.10</td></tr><tr><td>GM</td><td>31.83±5.27</td><td>49.60±5.13</td><td>82.22±11.43</td><td>96.23±10.24</td><td>116.63±12.05</td><td>75.30</td></tr><tr><td>Ours</td><td>22.82±0.96</td><td>42.68±2.65</td><td>74.68±3.29</td><td>86.56±3.71</td><td>104.85±4.14</td><td>66.32</td></tr><tr><td></td><td></td><td></td><td>(b) DLinear</td><td></td><td></td><td></td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="4">H36M</td><td>Full</td><td>14.46±2.53</td><td>30.16±3.45</td><td>64.54±8.30</td><td>81.52±10.22</td><td>112.02±14.38</td><td>60.54</td></tr><tr><td>Random</td><td>17.92±7.18</td><td>34.70±8.10</td><td>66.84±8.25</td><td>78.42±10.22</td><td>107.58±13.53</td><td>61.09</td></tr><tr><td>GM</td><td>19.88±4.81</td><td>35.24±4.88</td><td>60.28±3.45</td><td>70.20±2.61</td><td>85.04±3.15</td><td>54.13</td></tr><tr><td>Ours</td><td>11.44±1.05</td><td>24.28±2.10</td><td>47.62±4.95</td><td>57.96±5.57</td><td>80.16±9.81</td><td>44.29</td></tr><tr><td rowspan="4">CMU</td><td>Full</td><td>19.33±0.36</td><td>36.89±0.61</td><td>71.49±0.75</td><td>86.64±0.66</td><td>112.29±1.12</td><td>65.33</td></tr><tr><td>Random</td><td>16.93±1.67</td><td>35.93±2.86</td><td>75.14±4.74</td><td>92.80±5.25</td><td>123.27±6.81</td><td>68.81</td></tr><tr><td>GM</td><td>20.76±2.67</td><td>39.79±3.52</td><td>74.44±2.74</td><td>89.37±2.06</td><td>111.79±1.09</td><td>67.23</td></tr><tr><td>Ours</td><td>12.85±1.35</td><td>28.96±3.07</td><td>67.33±7.36</td><td>86.11±9.60</td><td>118.18±13.53</td><td>62.69</td></tr><tr><td rowspan="4">3DPW</td><td>Full</td><td>19.05±0.29</td><td>37.51±0.62</td><td>69.79±0.58</td><td>81.86±0.74</td><td>99.63±1.05</td><td>61.57</td></tr><tr><td>Random</td><td>19.69±3.56</td><td>38.87±7.79</td><td>74.84±12.01</td><td>88.53±13.36</td><td>109.45±18.92</td><td>66.28</td></tr><tr><td>GM</td><td>18.83±1.74</td><td>37.83±3.69</td><td>70.85±4.21</td><td>83.63±4.26</td><td>102.82±5.60</td><td>62.79</td></tr><tr><td>Ours</td><td>17.96±0.81</td><td>36.05±1.52</td><td>70.65±3.38</td><td>84.70±3.91</td><td>107.12±5.31</td><td>63.30</td></tr></table>

![](images/4a9db4098a1a34b4d825be033a05d2e1f50e94679ee0f1f7db6a69ba2acccdc4.jpg)  
Fig. 2. Synthetic training trajectories (not predictions) initialized from two Human3.6M sequences; the left column shows the initialization, and frame indices #0–#30 index the 50-frame trajectory rather than prediction horizons. The distilled motions need not retain the initializing action. In the examples shown, direct GM produces implausible joint configurations and abrupt transitions, whereas motions decoded from the latent bank stay within the decoder output space.

Distillation objective. The complete objective is

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \lambda _ { \mathrm { G M } } \mathcal { L } _ { \mathrm { G M } } + \lambda _ { \mathrm { v e l } } \mathcal { L } _ { \mathrm { v e l } } + \lambda _ { \mathrm { p o s e } } \mathcal { L } _ { \mathrm { p o s e } } .\tag{4}
$$

Only H is updated, while the RVQ-VAE remains fixed and the backbone provides gradient-matching signals. After optimization, all latent slots are decoded once into 50-frame sequences, yielding a standard synthetic dataset that trains predictors without the RVQ-VAE.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Datasets and Evaluation Metrics. We evaluate the framework on Human3.6M [24], the CMU Graphics Lab Motion Capture Database [25], and 3DPW [26]. Human3.6M is represented by 99-D exponential maps, CMU by 70-D normalized features, and 3DPW by 72-D SMPL axis-angle features. Prediction quality is measured by mean per-joint position error (MPJPE, mm) on 3D joint positions recovered from each representation, where a lower value is better. Models observe 25 frames and predict the next 10; at test time the predicted block is appended to the observation window, and the predictor is reapplied until 25 future frames are produced, so horizons beyond #10 come from rollout. We report frames #2/#4/#8/#10/#14, corresponding to 80/160/320/400/560 ms at 25 fps. Mean5 averages these horizons, and Mean8 adds #18/#22/#25 (720/880/1000 ms).

Baselines and Budgets. We compare Full-data training, Random subset selection, direct GM, and the proposed method. Direct

![](images/aa08d5a39ce2cd9c0cdce852d10565421712b0f608a87ddb8e8a3f7931ccfc15.jpg)

![](images/dc5334e3d81793df1e6805e6324820cc43937216364b7e7e917c9d66b2ab60be.jpg)  
Fig. 3. Ablation studies. (a) Mean5 MPJPE on Human3.6M with DLinear as the budget grows, where Full is budget-independent and trained for the same 4000 updates. (b) Mean5 and Mean8 MPJPE on CMU with Simple at the 2× budget. GM+Reg is direct GM trained with the objective of Eq. (4), $K = 1$ is a single-stage VQ variant, and error bars are standard deviations over five runs.

GM optimizes the same initialized segments in the feature space using only the GM objective, with the same budget, backbone, iterations, and learning rate as Ours; the ablation also reports direct GM trained with the objective of Eq. (4), denoted GM+Reg. The backbone used as the gradient probe is randomly initialized and kept fixed within each run. Adapting temporal DD methods [10–12] to articulated motion is left for future work. The budget is the number M of synthetic trajectories, each of which yields 16 overlapping, hence non-independent, 35-frame training windows. The $1 \times / 2 \times / 4 \times$ budgets correspond to 5%/10%/20% of the real training sequences on Human3.6M, 10%/20%/40% on 3DPW, and 1/2/4 trajectories per action class on CMU, with identical M for Random, GM, and Ours.

Implementation Details. We use a lightweight backbone based on SiMLPe [18], denoted Simple, and DLinear [27]. The RVQ-VAE uses a 64-D latent space, six residual stages, and 128 entries per codebook, pretrained for 4000 iterations with a learning rate of 2 × $1 0 ^ { - 4 }$ . We set ${ \lambda } _ { v } = 1$ and warm up λ<sub>KL</sub> $( t ) = 1 0 ^ { - 4 }$ min(1, t/1000) over the first 1000 iterations. The latent bank is distilled for 2500 iterations with Adam, learning rate $1 0 ^ { - 2 } .$ , and batch size 16, with $\lambda _ { \mathrm { G M } } ~ = ~ 1 0$ and $\lambda _ { \mathrm { p o s e } } ~ = ~ \lambda _ { \mathrm { v e l } } ~ = ~ 1 . 5$ . All downstream models, including Full, use the same setting of 4000 iterations, learning rate $3 \times 1 0 ^ { - 4 }$ , and weight decay $1 0 ^ { - 4 }$ , so Full is a reference under this protocol rather than a converged upper bound. We report results averaged over five random seeds.

## 3.2. Evaluation Results

Comparisons at the Fixed Budget. Table 1 compares the proposed method with Full, Random, and direct GM at the 1× budget. Ours achieves the lowest Mean5 MPJPE in five of the six dataset– backbone combinations, cutting the error of direct GM by 48.9% on Human3.6M with Simple and by 6.8% to 18.2% on four pairs, while 3DPW with DLinear is 0.8% worse. Over the 30 horizon-level comparisons, Ours leads direct GM in 27 and Random in all 30. Direct GM is unstable on Human3.6M with Simple, where its error exceeds that of Random with a large variance. Its distilled motions contain implausible joint configurations (Fig. 2). Ours falls below Full on several DLinear settings, where Full is trained for the same 4000 updates and has not converged.

Ablation on the Data Budget. Fig. 3(a) varies the syntheticdata budget on Human3.6M with DLinear. Ours achieves 44.29, 43.82, and 41.72 Mean5 at the three budgets, beating direct GM and Random at every budget. Random and direct GM instead degrade at 4×, where the fixed downstream budget gives each trajectory fewer passes over training.

Cross-Backbone Transfer. Table 2 asks whether a set distilled with one backbone stays useful for another, and Ours yields lower transfer error than direct GM in all four directions. The gain is clearest for Simple→DLinear, where the transferred data stays close to the in-backbone result and on CMU outperforms it. The distilled sets thus retain cross-backbone utility in absolute accuracy, while the relative degradation from in-backbone to transferred training is larger for Ours than for direct GM in two of the four directions.

Table 2. Cross-backbone transfer at the 2× budget. Each cell is the Mean5 MPJPE (mm) of the target backbone trained on the set distilled with the source backbone (S: Simple, D: DLinear). Parentheses give the target backbone trained on a set distilled with the target backbone itself.
<table><tr><td>Transfer</td><td>GM</td><td>Ours</td></tr><tr><td>CMU S→D</td><td>70.40±3.48 (68.37)</td><td>58.45±2.61 (63.32)</td></tr><tr><td>CMU D→S</td><td>100.34±7.41 (80.21)</td><td>90.98±4.35 (67.51)</td></tr><tr><td>H36M S→D</td><td>66.12±14.37 (57.22)</td><td>47.30±4.05 (43.82)</td></tr><tr><td>H36M D→S</td><td>102.16±14.05 (70.97)</td><td>78.41±9.18 (43.89)</td></tr></table>

Ablation on the Latent Representation. Fig. 3(b) compares latent representations at the 2× budget. Adding the regularizers of Eq. (4) to direct GM lowers Mean5 from 80.2 to 74.9, and all latent variants use this same objective, so the remaining differences reflect the representation alone. Relative to GM+Reg, the continuous VAE latent space lowers Mean5 to 70.8, a single-stage VQ variant is worse than the continuous space at 76.2, and increasing the residual depth recovers the loss, with $K = 6$ reaching 67.5 Mean5 and 97.1 Mean8, 9.9% and 6.0% below GM+Reg and with the smallest spread among all variants. A larger K also enlarges the codebooks, so the gain in residual depth includes the added capacity.

Qualitative Comparison. Fig. 2 shows synthetic training trajectories, not predictions. Since pairing ignores the initializing sequence, the distilled motions need not retain the initializing action. In the examples shown, direct GM produces implausible joint configurations and abrupt transitions, whereas motions decoded from the latent bank stay within the decoder output space.

## 4. CONCLUSION

We have presented a latent dataset distillation framework for human motion prediction. Rather than optimizing motion features directly, it distills in the pre-quantization latent space of a pretrained RVQ-VAE and updates only a latent bank through the frozen quantizer and decoder, so gradient matching is regularized by a learned motion prior. On Human3.6M, CMU, and 3DPW with two backbones, the distilled sets outperform direct gradient matching and random subsets under a fixed downstream budget, and residual quantization improves over single-stage quantization among latent variants sharing the same objective. The distilled sets also remain useful when transferred to a backbone other than the one used for distillation.

## 5. REFERENCES

[1] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A. Efros, “Dataset distillation,” arXiv preprint arXiv:1811.10959, 2018.

[2] Guang Li, Bo Zhao, and Tongzhou Wang, “Awesome dataset distillation,” https://github.com/Guang000/ Awesome-Dataset-Distillation, 2022.

[3] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen, “Dataset condensation with gradient matching,” in International Conference on Learning Representations, 2021.

[4] Bo Zhao and Hakan Bilen, “Dataset condensation with differentiable siamese augmentation,” in Proceedings of the 38th International Conference on Machine Learning, 2021, pp. 12674–12685.

[5] Bo Zhao and Hakan Bilen, “Dataset condensation with distribution matching,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 6514–6523.

[6] George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu, “Dataset distillation by matching training trajectories,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10718–10727.

[7] Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong, Jung-Woo Ha, and Hyun Oh Song, “Dataset condensation via efficient synthetic-data parameterization,” in Proceedings ofthe 39th International Conference on Machine Learning, 2022, pp. 11102–11118.

[8] Yanqing Liu, Jianyang Gu, Kai Wang, Zheng Zhu, Wei Jiang, and Yang You, “DREAM: Efficient dataset distillation by representative matching,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17314– 17324.

[9] Guang Li, Ren Togo, Takahiro Ogawa, and Miki Haseyama, “Compressed gastric image generation based on soft-label dataset distillation for medical data sharing,” Computer Methods and Programs in Biomedicine, vol. 227, pp. 107189, 2022.

[10] Hao Miao, Ziqiao Liu, Yan Zhao, Chenjuan Guo, Bin Yang, Kai Zheng, and Christian S. Jensen, “Less is more: Efficient time series dataset condensation via two-fold modal matching,” Proceedings ofthe VLDB Endowment, vol. 18, no. 2, pp. 226– 238, 2024.

[11] Seungha Hong, Sanghwan Jang, Wonbin Kweon, Suyeon Kim, Gyuseok Lee, and Hwanjo Yu, “Harmonic dataset distillation for time series forecasting,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 26, pp. 21770– 21778, 2026.

[12] Taehyung Kwon, Yeonje Choi, Yeongho Kim, and Kijung Shin, “Effective dataset distillation for spatio-temporal forecasting with bi-dimensional compression,” in 2026 IEEE 42nd International Conference on Data Engineering (ICDE), 2026, pp. 1–14.

[13] Julieta Martinez, Michael J. Black, and Javier Romero, “On human motion prediction using recurrent neural networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[14] Wei Mao, Miaomiao Liu, Mathieu Salzmann, and Hongdong Li, “Learning trajectory dependencies for human motion prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9489–9497.

[15] Maosen Li, Siheng Chen, Yangheng Zhao, Ya Zhang, Yanfeng Wang, and Qi Tian, “Dynamic multiscale graph neural networks for 3D skeleton based human motion prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[16] Wei Mao, Miaomiao Liu, and Mathieu Salzmann, “History repeats itself: Human motion prediction via motion attention,” in Proceedings of the European Conference on Computer Vision, 2020, pp. 474–489.

[17] Emre Aksan, Manuel Kaufmann, Peng Cao, and Otmar Hilliges, “A spatio-temporal transformer for 3D human motion prediction,” in 2021 International Conference on 3D Vision, 2021, pp. 565–574.

[18] Wen Guo, Yuming Du, Xi Shen, Vincent Lepetit, Xavier Alameda-Pineda, and Francesc Moreno-Noguer, “Back to MLP: A simple baseline for human motion prediction,” in Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 2023, pp. 4809–4819.

[19] Yuxuan Duan, Jianfu Zhang, and Liqing Zhang, “Dataset distillation in latent space,” arXiv preprint arXiv:2311.15547, 2023.

[20] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu, “Neural discrete representation learning,” in Advances in Neural Information Processing Systems, 2017, vol. 30.

[21] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan, “Generating human motion from textual descriptions with discrete representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14730–14740.

[22] Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng, “MoMask: Generative masked modeling of 3D human motions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1900– 1910.

[23] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi, “SoundStream: An end-toend neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2022.

[24] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu, “Human3.6M: Large scale datasets and predictive methods for 3D human sensing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 7, pp. 1325–1339, 2014.

[25] Carnegie Mellon University, “CMU graphics lab motion capture database,” https://mocap.cs.cmu.edu/.

[26] Timo von Marcard, Roberto Henschel, Michael J. Black, Bodo Rosenhahn, and Gerard Pons-Moll, “Recovering accurate 3D human pose in the wild using IMUs and a moving camera,” in Proceedings of the European Conference on Computer Vision, 2018, pp. 601–617.

[27] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu, “Are transformers effective for time series forecasting?,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 9, pp. 11121–11128, 2023.