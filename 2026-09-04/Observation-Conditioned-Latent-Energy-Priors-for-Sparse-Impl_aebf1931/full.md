# Observation-Conditioned Latent Energy Priors for Sparse Implicit Neural Shape Completion

Paul Büschl<sup>1,2,6</sup> , Ezequiel de la Rosa<sup>1</sup> , Julia Wolleb<sup>1,3</sup> , Julian McGinnis<sup>4,5</sup> , César Nombela-Arrieta<sup>2</sup> , and Bjoern Menze<sup>1,6</sup>

<sup>1</sup> Department of Quantitative Biomedicine, University of Zurich, Zurich, Switzerland paul.bueschl@uzh.ch

<sup>2</sup> Department of Medical Oncology and Hematology, University Hospital Zurich, Zurich, Switzerland

3 Department of Computer Science, ETH Zurich, Zurich, Switzerland

Department of Computer Science, Technical University of Munich, Munich, Germany

Munich Center for Machine Learning (MCML), Munich, Germany <sup>6</sup> ETH AI Center, ETH Zurich, Zurich, Switzerland

Abstract. Implicit neural representations (INRs) can model continuous 3D shapes with a shared coordinate decoder and per-instance latent codes. At test time, autodecoder-style models commonly freeze the decoder and optimize a new latent code from sparse of-grid SDF samples. When these samples underconstrain inference, the latent can drift toward regions that fit the observations but decode implausible unobserved geometry. We propose a post-hoc observation-conditioned latent energy prior for frozen INR decoders. The energy scores standardized latents conditioned on a permutation-invariant encoding of the sparse observation set and is used as a residual expert alongside an L2 latent prior selected on validation data. We evaluate on a controlled cellnucleus SDF dataset and a public MedShapeNet-derived SDF completion dataset. The proposed L2 objective augmented with conditional energy improves consistently over a validation-selected L2 baseline in the sparsest cell-nucleus regimes and, on MedShapeNet, outperforms both L2 and a six-component GMM latent-density prior across all reported readouts. A shufled-context ablation is consistently weaker than matched context, supporting an observation-specific contribution. These results suggest that lightweight conditional energies can make pretrained INR decoders more observation-aware without retraining.

Keywords: Implicit neural representations · Test-time optimization · Energy-based priors · Shape completion · Signed distance functions

## 1 Introduction

Implicit neural representations (INRs) model signals as continuous functions of spatial coordinates [18,16,21], making them well suited to of-grid medical images, volumes, and shapes [7,12,6,3]. Two paradigms are commonly distinguished: INRs optimized per instance, as e.g., in MRI reconstruction [15,8,11] and instance-wise image registration [20], and latent-conditioned INRs, which learn population-level priors [21,6] and thereby enable applications such as atlas construction [4]. In the latent-conditioned setting [14,5,10], a decoder shared across the population receives a coordinate x and an instance-specific latent code z and predicts a coordinate-wise quantity such as intensity, signed distance, or a segmentation logit [17]

This decoder-plus-latent formulation underlies autodecoder shape models such as DeepSDF, which represent shapes as continuous signed-distance fields and support completion from partial or noisy observations [14]. Related medical and biological work has used INRs for sparse anatomical shape reconstruction, implicit segmentation, image fitting, or cell-shape modeling [1,17,11,19].

In autodecoder-style INRs, training learns a shared decoder together with latent codes for the training instances. For a held-out instance, the decoder is kept fixed and inference optimizes a new latent code so that the decoded field matches the available observations. This test-time optimization is attractive for sparse shape completion: a small unordered set of of-grid SDF samples can be lifted to a full continuous field through the learned decoder. However, sparse inference is underconstrained. Many latent codes may explain the measured samples while decoding diferent unobserved geometry, so optimization can reduce the observation loss while moving toward a latent region that yields an implausible completion. We refer to this failure mode as latent drift.

The standard lightweight stabilization is an L2 penalty on the optimized latent code, which corresponds to a zero-mean isotropic Gaussian latent prior [1]. This prior is useful because it controls latent scale and gives stable tail behavior, but it is global and observation-agnostic: the same radial preference is applied regardless of which sparse points were observed. Stronger global latent-density priors, such as covariance-aware penalties, Gaussian mixtures, or learned latent energy models, can capture anisotropy, multimodality, or higher-order structure in the train-latent distribution [13]. Yet these priors still score latent plausibility independently of the actual sparse observation set. Sparse shape completion is a conditional problem: among many plausible latent codes, inference should prefer latents that are plausible for the particular observed coordinate-value pairs.

This leaves a gap for lightweight post-hoc priors that can be attached to frozen latent-conditioned INRs while conditioning their latent preference on sparse of-grid measurements. We therefore study whether a post-hoc conditional energy, trained from frozen-decoder latents and sparse observation sets generated by subsampling coordinate/SDF pairs, can act as an observation-aware residual prior during test-time latent optimization. The energy is trained with an Energy-Matching-inspired objective and receives a permutation-invariant encoding of the sparse observation set, following the principle that functions on unordered sets should be invariant to input order [2,22]. At inference, the decoder remains fixed and only the latent code is optimized; the conditional energy is used alongside a tuned L2 prior rather than as a replacement for it.

![](images/587b065cf0b3f9899f5eb0196f0bca57d5887be47f01ecced16f7a5d1bc49ff5.jpg)  
Fig. 1. Observation-conditioned latent energy prior workflow. Block 1: a latentconditioned INR/autodecoder provides a frozen decoder and training latents. Block 2: sparse observation sets are generated by subsampling coordinate/SDF pairs from training shapes and paired with their latents to train a conditional scalar energy. Block 3: for a new sparse observation set, only the latent is optimized while the decoder and energy remain fixed. The L2 prior weight is selected on validation data.

We evaluate the approach in sparse SDF shape-completion settings, including a controlled cell-nucleus testbed and a public heterogeneous MedShapeNet mesh/SDF track [9]. Our contributions are:

1. We identify latent drift as a failure mode of sparse test-time optimization in latent-conditioned INRs.

2. We introduce a post-hoc observation-conditioned energy prior that acts as a residual expert alongside a tuned L2 latent prior for frozen-decoder inference.

3. We evaluate the method against lightweight global priors and context ablations in controlled and public SDF completion settings.

Code and configuration files are available at https://github.com/pbueschl/ latent-energy-priors\_clean.git.

## 2 Method

Figure 1 summarizes the proposed workflow. As shown in block 1, we start from a pretrained latent-conditioned INR decoder and its training latents. As shown in block 2, we train a post-hoc conditional energy on paired training latents and sparse observation sets generated by subsampling coordinate/SDF pairs from the same training shapes. As shown in block 3, test-time inference keeps the decoder and energy fixed and optimizes only the latent code using the observation loss, an L2 base prior, and the conditional energy.

## 2.1 Frozen Decoder and Sparse Latent Inference

Let $f _ { \theta }$ be a pretrained latent-conditioned decoder. In our SDF setting, $f _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { z } )$ predicts the signed distance at normalized of-grid coordinate x for an instance latent code z. For a new instance, we observe a sparse set $\mathcal { O } ^ { \star } = \{ ( \mathbf { x } _ { j } , s _ { j } ) \} _ { j = 1 } ^ { m }$ where $\mathbf { x } _ { j }$ is a coordinate and $s _ { j }$ is the observed signed-distance value at that coordinate. With frozen decoder parameters, standard autodecoder inference optimizes only the latent:

$$
\operatorname* { m i n } _ { \mathbf { z } } \ \mathcal { L } _ { \mathrm { o b s } } ( \mathbf { z } ; f _ { \boldsymbol { \theta } } , \boldsymbol { \mathcal { O } } ^ { \star } ) , \qquad \mathcal { L } _ { \mathrm { o b s } } ( \mathbf { z } ; f _ { \boldsymbol { \theta } } , \boldsymbol { \mathcal { O } } ^ { \star } ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \ell _ { \mathrm { S D F } } ( f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { j } , \mathbf { z } ) , s _ { j } ) \ .\tag{1}
$$

Latent drift occurs when this optimization is underconstrained: the optimized latent fits the observed samples but decodes implausible unobserved geometry.

## 2.2 Observation-Conditioned Product-of-Experts Inference

After training the base decoder, we extract the training latents $\mathcal { Z } _ { \mathrm { t r a i n } } = \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ and compute per-dimension statistics $\mu _ { \mathrm { t r a i n } }$ and $\sigma _ { \mathrm { t r a i n } }$ . We standardize any latent as $\tilde { \mathbf { z } } ( \mathbf { z } ) \mathbf { \Psi } = \mathbf { \Psi } ( \mathbf { z } - \mu _ { \mathrm { t r a i n } } ) / ( \sigma _ { \mathrm { t r a i n } } + \epsilon )$ . The L2 base prior is a standard squared latent penalty $R _ { 2 } ( \mathbf { z } )$ . In practice, constant normalizations of this penal $\mathrm { \Delta [ y , }$ such as division by the latent dimension, can be absorbed into the validation-selected weight $\lambda _ { 2 }$

To condition the prior on sparse observations, we encode the unordered set O with a DeepSets-style context encoder,

$$
\mathbf { c } _ { \psi } ( \mathcal { O } ) = \rho _ { \psi } \left( \operatorname { p o o l } _ { ( \mathbf { x } , s ) \in \mathcal { O } } h _ { \psi } ( [ \mathbf { x } , s ] ) \right) ,\tag{2}
$$

where $h _ { \psi }$ embeds each coordinate and SDF value, the pooling operation is invariant to the order of observations, and $\rho _ { \psi }$ maps the pooled vector to a context vector. The conditional energy $E _ { \phi } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d _ { c } }  \mathbb { R }$ is a scalar network that scores a standardized latent in the context of this observation embedding, where d is the latent dimensionality and $d _ { c }$ is the context dimension.

The main test-time objective is

$$
\operatorname* { m i n } _ { \mathbf { z } } \ \mathcal { L } _ { \mathrm { o b s } } ( \mathbf { z } ; f _ { \boldsymbol { \theta } } , \mathcal { O } ^ { \star } ) + \lambda _ { 2 } R _ { 2 } ( \mathbf { z } ) + \lambda _ { C } E _ { \boldsymbol { \phi } } ( \widetilde { \mathbf { z } } ( \mathbf { z } ) , \mathbf { c } _ { \boldsymbol { \psi } } ( \mathcal { O } ^ { \star } ) ) .\tag{3}
$$

The L2 expert controls latent scale, while the conditional energy acts as an observation-aware residual expert. The decoder parameters $\theta$ are frozen; only the latent code z is optimized during inference.

## 2.3 Training the Conditional Energy

The conditional energy and context encoder are trained after the decoder has been fitted. For each training shape i, we pair its training latent $\mathbf { z } _ { i }$ with sparse observation sets ${ \mathcal { O } } _ { i }$ generated by subsampling coordinate/SDF pairs from the same training shape. Each minibatch samples the number of observed SDF samples m from the dataset-specific training values and then subsamples m coordinate/SDF pairs. The conditional energy is trained only from train-split latents and train-split observations.

We adapt Energy Matching to the conditional latent space. Let $\mathbf { c } _ { i } = \mathbf { c } _ { \psi } ( \mathcal { O } _ { i } )$ denote the observation context and $\tilde { \bf z } _ { i } = \tilde { \bf z } ( { \bf z } _ { i } )$ the standardized training latent. We sample Gaussian noise $\tilde { z } _ { 0 } \sim \mathcal { N } ( 0 , I )$ and $t \sim \mathcal { U } ( 0 , 1 )$ , form the interpolant $\tilde { \mathbf { z } } _ { t } = ( 1 - t ) \tilde { \mathbf { z } } _ { 0 } + t \tilde { \mathbf { z } } _ { i }$ , and regress the energy gradient so that the negative gradient points toward the matched training latent:

$$
\mathcal { L } _ { \mathrm { C E M } } = \mathbb { E } \left[ \frac { 1 } { d } \left. \nabla _ { \tilde { z } _ { t } } E _ { \phi } ( \tilde { z } _ { t } , c _ { i } ) + ( \tilde { z } _ { i } - \tilde { z } _ { 0 } ) \right. _ { 2 } ^ { 2 } \right] .\tag{4}
$$

We add ranking losses to make the matched latent-context pair lower energy than mismatched pairs. For $k \neq i$ and margin $m _ { \mathrm { r a n k } }$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r a n k } } = \mathbb { E } _ { i , k } \left[ \mathrm { s o f t p l u s } \left( E _ { \phi } ( \tilde { z } _ { i } , c _ { i } ) + m _ { \mathrm { r a n k } } - E _ { \phi } ( \tilde { z } _ { k } , c _ { i } ) \right) \right] } \\ & { \quad \quad \quad + \mathbb { E } _ { i , k } \left[ \mathrm { s o f t p l u s } \left( E _ { \phi } ( \tilde { z } _ { i } , c _ { i } ) + m _ { \mathrm { r a n k } } - E _ { \phi } ( \tilde { z } _ { i } , c _ { k } ) \right) \right] . } \end{array}\tag{5}
$$

The first term compares the correct latent with a mismatched latent under the same context; the second compares the correct context with a mismatched context for the same latent. Together, they discourage the energy from ignoring either the latent or the observation context. Finally, we include of-manifold negatives $\tilde { z } ^ { - }$ , initialized from broad latent noise or train-latent perturbations and refined by short Langevin chains. These negatives are used only during energy training. Let

$$
\bar { E } _ { \mathrm { p o s } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } E _ { \phi } ( \tilde { z } _ { i } , c _ { i } )\tag{6}
$$

denote the average matched-pair energy in the current minibatch. The negative calibration loss is

$$
\mathcal { L } _ { \mathrm { n e g } } = \mathbb { E } \left[ \mathrm { s o f t p l u s } \left( \bar { E } _ { \mathrm { p o s } } + m _ { \mathrm { n e g } } - E _ { \phi } ( \tilde { z } ^ { - } , c _ { i } ) \right) \right] .\tag{7}
$$

Thus, valid of-manifold negatives are encouraged to have higher energy than the average matched train pair in the current minibatch.

The full training loss is

$$
\mathcal { L } _ { \mathrm { e n e r g y } } = \alpha _ { \mathrm { C E M } } \mathcal { L } _ { \mathrm { C E M } } + \alpha _ { \mathrm { r a n k } } \mathcal { L } _ { \mathrm { r a n k } } + \alpha _ { \mathrm { n e g } } \mathcal { L } _ { \mathrm { n e g } } + \alpha _ { \mathrm { s c a l e } } \mathbb { E } _ { ( \tilde { z } , c ) \in \mathcal { B } _ { \mathrm { e n e r g y } } } \left[ E _ { \phi } ( \tilde { z } , c ) ^ { 2 } \right] .\tag{8}
$$

Here, $B _ { \mathrm { e n e r g y } }$ denotes the set of energy evaluations used in the current training step, including matched pairs, mismatched latent-context pairs, and negative samples. The interpolated latents and Langevin-refined negatives are trainingonly samples; test-time inference uses the learned energy as a diferentiable prior.

## 3 Experiments

## 3.1 Datasets, Sparse Observations, and Metrics

We evaluate sparse SDF completion in a controlled cell-nucleus setting and a public MedShapeNet-derived SDF setting. The number of observed of-grid SDF samples is denoted by m; all remaining samples are held out for SDF-error evaluation. The nucleus split contains 2,924/699/492 train/validation/lockedtest nuclei; each training nucleus contributes one latent code. Each nucleus has 4,096 of-grid SDF samples, and sparse inference uses $m \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$

MedShapeNet is a public collection of anatomical 3D meshes. We use bladder, brain, heart, liver, skull MRI, and vertebrae with a balanced 960/120/120 train/validation/test split. Meshes are normalized and converted to clipped ofgrid SDF samples for autodecoder training and sparse inference with $m \in$ {16, 32, 64, 128, 256}.

The primary metric is held-out SDF error on unobserved of-grid samples. Secondary grid readouts include Grid SDF MAE and occupancy Dice/IoU after thresholding the SDF at zero; the nucleus setting also reports surface Dice, which measures boundary agreement within the fixed distance tolerance of 0.1 in normalized coordinate units.

## 3.2 Baselines, Ablations, and Inference Protocol

All methods use the same frozen decoder, matched sparse observations, latent initialization, optimization budget, and evaluation points within each setting. We distinguish baselines from ablations. The baselines are: No prior, which minimizes only ${ \mathcal { L } } _ { \mathrm { o b s } } ;$ Tuned L2, which adds the validation-selected squared latent penalty; and GMM, a global latent-density baseline that fits a six-component Gaussian mixture to standardized training latents and uses − log p<sub>GMM</sub>(˜z) as a prior $( \lambda _ { \mathrm { G M M } } = 1 0 ^ { - 5 } )$ . Our main method is L2 + Conditional Energy. The ablations are Conditional Energy only, which removes the L2 expert, and shufled context, which evaluates the conditional energy with a mismatched observation context. For nuclei, latent optimization runs for 1,000 Adam steps with learning rate $1 0 ^ { - 2 }$ and latent dimension $d = 6 4$ . Grid-based metrics use a $3 2 ^ { 3 }$ grid over normalized coordinates $[ - 1 . 5 , 1 . 5 ] ^ { 3 }$ . For MedShapeNet, locked-test inference runs for 2,000 Adam steps. Prior weights are selected by validation grid search using the setting-specific selection metric, and the locked test is evaluated once after freezing the selected weights and hyperparameters. For nuclei, final-step surface Dice selects $\lambda _ { 2 } = 0 . 3 , \lambda _ { C } = 3 { \times } 1 0 ^ { - 6 }$ , and $\lambda _ { 2 } + \lambda _ { C } = 0 . 3 + 1 0 ^ { - 5 }$ . For MedShapeNet, validation Eval L1 selects $\lambda _ { 2 } = 1 0 ^ { - 3 } , \lambda _ { C } = 1 0 ^ { - 5 }$ , and $\lambda _ { 2 } + \lambda _ { C } = 1 0 ^ { - 3 } + 3 { \times } 1 0 ^ { - 5 }$ We report paired bootstrap confidence intervals; for nuclei, we additionally use paired Wilcoxon tests with FDR correction.

Table 1. Nucleus locked-test performance. CondE: Conditional Energy; Ours: L2 plus CondE.
<table><tr><td colspan="5">Grid SDF MAE↓</td><td colspan="4">Surface Dice ↑</td></tr><tr><td></td><td>Obs. No prior</td><td>CondE</td><td>Tuned L2</td><td>Ours</td><td>No prior</td><td>CondE</td><td>Tuned L2</td><td>Ours</td></tr><tr><td>16</td><td>0.1026</td><td>0.0968</td><td>0.0851</td><td>0.0838</td><td>0.7818</td><td>0.8143</td><td>0.8590</td><td>0.8650</td></tr><tr><td>32</td><td>0.0756</td><td>0.0724</td><td>0.0685</td><td>0.0664</td><td>0.8804</td><td>0.8986</td><td>0.9138</td><td>0.9222</td></tr><tr><td>64</td><td>0.0606</td><td>0.0581</td><td>0.0558</td><td>0.0551</td><td>0.9336</td><td>0.9412</td><td>0.9521</td><td>0.9545</td></tr><tr><td>128</td><td>0.0501</td><td>0.0496</td><td>0.0485</td><td>0.0485</td><td>0.9667</td><td>0.9677</td><td>0.9725</td><td>0.9728</td></tr></table>

Table 2. MedShapeNet locked-test performance. GMM K6: six-component global latent mixture; Ours: L2+CondE.
<table><tr><td>Obs.</td><td>Method</td><td>Eval L1 ↓</td><td>Grid SDF MAE ↓</td><td>Dice ↑</td><td>IoU ↑</td></tr><tr><td>16</td><td>Tuned L2</td><td>0.0252</td><td>0.1156</td><td>0.5102</td><td>0.3666</td></tr><tr><td></td><td>GMM K6</td><td>0.0246</td><td>0.1125</td><td>0.5268</td><td>0.3855</td></tr><tr><td></td><td>Ours</td><td>0.0245</td><td>0.1079</td><td>0.5500</td><td>0.4056</td></tr><tr><td>32</td><td>Tuned L2</td><td>0.0212</td><td>0.1046</td><td>0.5854</td><td>0.4425</td></tr><tr><td></td><td>GMM K6</td><td>0.0207</td><td>0.0998</td><td>0.6019</td><td>0.4587</td></tr><tr><td></td><td>Ours</td><td>0.0200</td><td>0.0919</td><td>0.6271</td><td>0.4863</td></tr><tr><td>64</td><td>Tuned L2</td><td>0.0172</td><td>0.0917</td><td>0.6668</td><td>0.5304</td></tr><tr><td></td><td>GMM K6</td><td>0.0165</td><td>0.0867</td><td>0.6847</td><td>0.5509</td></tr><tr><td></td><td>Ours</td><td>0.0163</td><td>0.0814</td><td>0.6908</td><td>0.5571</td></tr><tr><td>128</td><td>Tuned L2</td><td>0.0129</td><td>0.0754</td><td>0.7495</td><td>0.6253</td></tr><tr><td></td><td>GMM K6</td><td>0.0129</td><td>0.0759</td><td>0.7505</td><td>0.6276</td></tr><tr><td></td><td>Ours</td><td>0.0125</td><td>0.0697</td><td>0.7632</td><td>0.6424</td></tr><tr><td>256</td><td>Tuned L2</td><td>0.0096</td><td>0.0624</td><td>0.8156</td><td>0.7089</td></tr><tr><td></td><td>GMM K6</td><td>0.0094</td><td>0.0614</td><td>0.8208</td><td>0.7156</td></tr><tr><td></td><td>Ours</td><td>0.0093</td><td>0.0571</td><td>0.8237</td><td>0.7199</td></tr></table>

## 4 Results

## 4.1 Locked-Test Performance

On the locked nucleus test set, L2 + Conditional Energy gives the best or tied-best performance across sparse observation settings (Table 1). Conditional-Energy-only is reported as an ablation: it tests whether the learned energy can replace the L2 expert. It improves over No prior in sparse settings but remains below Tuned L2, supporting the product-of-experts design in which the conditional energy complements rather than replaces the L2 prior.

On MedShapeNet, the global GMM prior often improves over Tuned L2, indicating that modeling train-latent density is useful. However, L2 + Conditional Energy gives the best performance for all reported MedShapeNet readouts and numbers of observed samples (Table 2). This suggests that observation conditioning adds information beyond a global multimodal latent prior.

![](images/40e81c69aa517f9ea681b393f98d29ff36cb779bba724e53ab8560f3c905cc41.jpg)  
Fig. 2. Paired locked-test gains of L2 + Conditional Energy over Tuned L2 and, for MedShapeNet, over GMM K6. Positive values denote error reduction or score increase. Error bars show 95% paired-bootstrap confidence intervals; open markers cross zero.

## 4.2 Paired Gains over Global Priors

Figure 2 reports paired locked-test gains. Positive values denote error reduction for L1/MAE metrics and score increase for Dice/IoU metrics. For nuclei, L2 + Conditional Energy improves most clearly over Tuned L2 at 16 and 32 observed samples, and gains shrink as observations become denser. FDR-corrected paired Wilcoxon tests show significant improvements over Tuned L2 for all four nucleus metrics at 16, 32, and 64 samples, with largest corrected p-value $\phantom { + } 3 8 \times 1 0 ^ { - 2 } ;$ at 128 samples, only surface Dice remains significant $( p = 8 . 0 { \times } 1 0 ^ { - 4 } )$ . On Med-ShapeNet, L2 + Conditional Energy improves over Tuned L2 across 16–256 samples and improves over GMM K6 most clearly for Grid SDF MAE and overlap-based readouts.

## 4.3 Context Ablation and Qualitative Examples

The shufled-context ablation tests whether the learned energy uses the matched observation set or only acts as an additional global prior. On MedShapeNet, shuffled context remains better than Tuned L2 on Grid SDF MAE, indicating a general latent-prior component. However, matched context improves over shufled context at every observed-sample count, reducing Grid SDF MAE by 0.0026– 0.0072 relative to shufled context. This supports an observation-specific contribution of the DeepSets context. Figure 3 shows representative MedShapeNet reconstructions. Compared with Tuned L2 and GMM K6, L2 + Conditional Energy more consistently preserves the category-compatible shape structure under the same sparse observations.

Table 3. MedShapeNet shufled-context ablation on Grid SDF MAE. Lower is better. ∆ values denote reductions relative to Tuned L2.
<table><tr><td>Obs.</td><td>Tuned L2</td><td>Ours</td><td>Ours shuf.</td><td>∆ matched</td><td>∆ shuf.</td></tr><tr><td>16</td><td>0.1156</td><td>0.1079</td><td>0.1124</td><td>0.0077</td><td>0.0032</td></tr><tr><td>32</td><td>0.1046</td><td>0.0919</td><td>0.0991</td><td>0.0127</td><td>0.0055</td></tr><tr><td>64</td><td>0.0917</td><td>0.0814</td><td>0.0868</td><td>0.0103</td><td>0.0049</td></tr><tr><td>128</td><td>0.0754</td><td>0.0697</td><td>0.0723</td><td>0.0057</td><td>0.0031</td></tr><tr><td>256</td><td>0.0624</td><td>0.0571</td><td>0.0600</td><td>0.0053</td><td>0.0023</td></tr></table>

![](images/75d59356b61fc403c0d8fbda3aab6b704e0c6fbf2940c4758a230c36b73ea779.jpg)  
Fig. 3. Qualitative MedShapeNet reconstructions from sparse observations. Rows show ground truth, Tuned L2, GMM K6, and L2 + Conditional Energy. Columns show anatomical categories.

## 5 Discussion & Conclusion

The main takeaway is that observation conditioning is most useful when sparse test-time inference is underconstrained. Tuned L2 provides stable global shrinkage, while the conditional Energy acts as an observation conditioned complement. The shufled context preserves a general latent-prior efect, while the matched context improves on it at all tested observation counts, which supports the additional observation-specific contribution interpretation. The GMM baseline indicates that modeling global train-latent density is useful on heterogeneous MedShapeNet.

The latent space-comparison in Fig. 4 provides a descriptive view of the two latent distributions. Looking at the per-dimension standardized latents, Med-ShapeNet shows stronger category structure and larger nearest-neighbor distance than the nucleus dataset. This is consistent with the results that show that a single isotropic L2 prior is too coarse, especially for the latent space of the more heterogeneous MedShapeNet dataset. Together with the improvement of L2+CondE over GMM K6 on the reported MedShapeNet readouts, this suggests that the conditional energy adds information beyond a global multimodal latent prior. Several limitations remain. We evaluate one controlled cell-nucleus SDF setting and one public MedShapeNet-derived SDF setting. Broader context randomization, out-of-distribution tests, and comparisons to additional structured priors beyond GMM, such as PCA-Gaussian, normalizing-flow, or difusion priors, remain important future work. Adaptive prior weights or uncertainty estimates may also help detect cases where sparse observations conflict with the learned prior. These results suggest that lightweight post-hoc priors can make pretrained INR decoders more observation-aware without retraining. Future work will study stronger conditional energy architectures, realistic observation patterns like partial surfaces or samples and broader medical and non-medical INR tasks, including topology-rich reconstruction problems such as vessel trees.

![](images/6db5d6081acfdd593d2d0056be212688e0827b9b3e6f9fc31edae6c0ffa52dc7.jpg)

![](images/a882c80864ba78e664c88fff43ce3d131a2f47d75b60b1cf1a3d62fbdf92f8ec.jpg)

![](images/ff711280968c25203bed5c1c9ff7fe18de2c5633a8b97dfca29d22021ca62382.jpg)  
Fig. 4. Standardized latent-space comparison. MedShapeNet latents show stronger category structure and larger nearest-neighbor distances than nucleus latents. This is consistent with, but does not establish, the hypothesis that a conditional energy is useful when one isotropic L2 prior is too coarse.

Acknowledgments. This work has been supported by the Helmut Horten Foundation.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Amiranashvili, T., Lüdke, D., Li, H.B., Menze, B., Zachow, S.: Learning shape reconstruction from sparse measurements with neural implicit functions. In: Proceedings of The 5th International Conference on Medical Imaging with Deep Learning. pp. 22–34 (2022)

2. Balcerak, M., Amiranashvili, T., Shit, S., Terpin, A., Kaltenbach, S., Koumoutsakos, P., Menze, B.: Energy matching: Unifying flow matching and energy-based models for generative modeling. arXiv preprint arXiv:2504.10612 (2025)

3. Coifier, G., Béthune, L.: 1-lipschitz neural distance fields. In: Computer Graphics Forum. vol. 43, p. e15128. Wiley Online Library (2024)

4. Dannecker, M., Kyriakopoulou, V., Cordero-Grande, L., Price, A.N., Hajnal, J.V., Rueckert, D.: Cina: Conditional implicit neural atlas for spatio-temporal representation of fetal brains. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 181–191. Springer (2024)

5. Dupont, E., Kim, H., Eslami, S., Rezende, D., Rosenbaum, D.: From data to functa: Your data point is a function and you can treat it like one. arXiv preprint arXiv:2201.12204 (2022)

6. Friedrich, P., Bieder, F., McGinnis, J., Wolleb, J., Rueckert, D., Cattin, P.C.: Medfuncta: A unified framework for learning eficient medical neural fields. In: Medical Imaging with Deep Learning (2026), https://openreview.net/forum? id=URCOfkQOwP

7. Gropp, A., Yariv, L., Haim, N., Atzmon, M., Lipman, Y.: Implicit geometric regularization for learning shapes. arXiv preprint arXiv:2002.10099 (2020)

8. Huang, W., Li, H.B., Pan, J., Cruz, G., Rueckert, D., Hammernik, K.: Neural implicit k-space for binning-free non-cartesian cardiac mr imaging. In: International Conference on Information Processing in Medical Imaging. pp. 548–560. Springer (2023)

9. Li, J., Zhou, Z., Yang, J., Pepe, A., Gsaxner, C., et al.: Medshapenet: A large-scale dataset of 3D medical shapes for computer vision. arXiv preprint arXiv:2308.16139 (2023)

10. Liu, H.T.D., Williams, F., Jacobson, A., Fidler, S., Litany, O.: Learning smooth neural functions via lipschitz regularization. In: ACM SIGGRAPH 2022 conference proceedings. pp. 1–13 (2022)

11. McGinnis, J., Shit, S., Li, H.B., Sideri-Lampretsa, V., Graf, R., Dannecker, M., Pan, J., Stolt-Ansó, N., Mühlau, M., Kirschke, J.S., Rueckert, D., Wiestler, B.: Single-subject multi-contrast MRI super-resolution via implicit neural representations (2023), preprint

12. Molaei, A., Aminimehr, A., Tavakoli, A., Kazerouni, A., Azad, B., Azad, R., Merhof, D.: Implicit neural representation in medical imaging: A comparative survey. In: 2023 IEEE/CVF International Conference on Computer Vision Workshops (IC-CVW). pp. 2373–2383. IEEE (2023)

13. Pang, B., Han, T., Nijkamp, E., Zhu, S.C., Wu, Y.N.: Learning latent space energybased prior model. In: Advances in Neural Information Processing Systems. vol. 33, pp. 21994–22008 (2020)

14. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: DeepSDF: Learning continuous signed distance functions for shape representation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019)

15. Shen, L., Pauly, J., Xing, L.: Nerp: implicit neural representation learning with prior embedding for sparsely sampled image reconstruction. IEEE transactions on neural networks and learning systems 35(1), 770–782 (2022)

16. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., Wetzstein, G.: Implicit neural representations with periodic activation functions. Advances in neural information processing systems 33, 7462–7473 (2020)

17. Stolt-Ansó, N., McGinnis, J., Pan, J., Hammernik, K., Rueckert, D.: Nisf: Neural implicit segmentation functions. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 734–744. Springer (2023)

18. Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., Ng, R.: Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems 33, 7537–7547 (2020)

19. Wiesner, D., Suk, J., Dummer, S., Svoboda, D., Wolterink, J.M.: Implicit neural representations for generative modeling of living cell shapes. arXiv preprint arXiv:2207.06283 (2022)

20. Wolterink, J.M., Zwienenberg, J.C., Brune, C.: Implicit neural representations for deformable image registration. In: International Conference on medical imaging with deep learning. pp. 1349–1359. PMLR (2022)

21. Xie, Y., Takikawa, T., Saito, S., Litany, O., Yan, S., Khan, N., Tombari, F., Tompkin, J., Sitzmann, V., Sridhar, S.: Neural fields in visual computing and beyond. In: Computer graphics forum. vol. 41, pp. 641–676. Wiley Online Library (2022)

22. Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R.R., Smola, A.J.: Deep sets. Advances in neural information processing systems 30 (2017)