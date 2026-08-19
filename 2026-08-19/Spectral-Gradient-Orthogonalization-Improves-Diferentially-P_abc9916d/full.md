# Spectral Gradient Orthogonalization Improves Diferentially Private Training at Scale

Sabari Shanmugam

Nick Barnes

Kerry Taylor

School of Computing, Australian National University, Canberra, Australia {sabari.shanmugam, nick.barnes, kerry.taylor}@anu.edu.au

## Abstract

Diferentially private training adds isotropic Gaussian noise to clipped gradients, corrupting every singular direction equally. In vision models, where spatial correlation concentrates gradient energy into a low-rank subspace, most of this noise falls in directions that carry little signal. Spectral gradient orthogonalization via polar decomposition is introduced as a postprocessing step that recovers directional signal from the noisy gradient’s low-rank structure at zero additional privacy cost. A phase transition governs the utility of this approach: orthogonalization improves accuracy only when the per-direction spectral signal-to-noise ratio (SNR) sufices for singular vector recovery; in low-SNR regimes, the directional bias of the gradient is replaced by a nearly random orthogonal update, and the transformation is harmful. The recovery threshold is determined by the spectral gap of the gradient and is surpassed at large batch sizes. Empirically, the benefit scales with model capacity: spectral orthogonalization achieves a +20.9% improvement over DP-SGD on WRN-28-10 (B = 4096) and +14.9% on ResNet-18, while reducing inter-run variance by a factor of two to three. In the fine-tuning regime, spectral orthogonalization matches the stability of DP-Adam while maintaining a first-order memory footprint. Combining spectral with temporal denoising yields 50.3% on CIFAR-10 (ε = 4), the highest accuracy in any tested configuration. These gains are specific to moderate-to-high-SNR regimes such as large-batch training of higher-capacity models. Small-batch or low-SNR settings are better served by DP-SGD or temporal denoising.

Keywords: Diferential privacy · Spectral methods · Gradient orthogonalization · Private training · Polar decomposition

## 1 Introduction

Computer vision models trained on sensitive data, such as medical images or visual biometrics (facial recognition, iris scans), must protect individual privacy while maintaining utility [19]. Diferential privacy (DP) [9] provides a formal guarantee: the model’s behavior changes negligibly whether or not any single example is included in the training set, preventing the memorization or recall of individual training samples. The standard mechanism, diferentially private stochastic gradient descent (DP-SGD) [1], enforces this by clipping each per-sample gradient to a fixed norm and adding calibrated Gaussian noise before aggregation. The noise ensures that the contribution of any one example is masked, but it degrades the optimization signal, reducing model accuracy. This utility cost is especially severe when training from scratch, where every parameter must be learned under noise. For vision architectures with large convolutional and attention layers, the gradient of each weight matrix is independently perturbed by isotropic noise, regardless of its structure.

(a) Clean gradient  
![](images/da9299cb0816f0e7d0927065994bc48346dc33aff1c4f4ee06126d4b6a741ba7.jpg)

(b) After DP noise  
![](images/07d71aecca27e32b7a5be6dcc127e8a35f09a48c1e781f4f4d1b1848da7b1813.jpg)

(c) Direction recovery  
![](images/563f4e7b8e151b4574df69033e939c7d6aef4bc272be46eef0bdef91927c212f.jpg)

(d) CIFAR-10, =4  
![](images/c0b4bb4c66bd271d31feafa834ad84c3022f508ee3b590c45f0ae310fe15e6a4.jpg)  
Figure 1: Spectral gradient orthogonalization overview. (a) Gradient singular values from a representative convolutional layer (epoch 1, CIFAR-10): most energy concentrates in a small number of directions (efective rank 6). (b) After DP noise addition $( \varepsilon = 4 )$ , isotropic perturbation raises the efective rank to 483: the signal becomes indistinguishable from the noise-dominated singular values (Marchenko–Pastur bulk), making the noisy gradient nearly directionless. (c) Direction recovery: cosine similarity between the leading singular vector of the noisy and clean gradients, measured via Monte Carlo simulation (100 trials per batch size) using the measured WRN-16-4 spectrum with noise calibrated to table 1. High similarity indicates that the gradient’s directional signal persists through the noise, providing the geometric condition for orthogonalization to succeed. Recovery transitions sharply from 0.90 at B = 256 to $\ge 0 . 9 9$ at $B \geq 1 0 2 4$ . (d) Test accuracy on CIFAR-10 $( \varepsilon = 4 )$ follows the same pattern: DP-SGD stagnates while spectral methods scale with batch size past the recovery threshold.

Neural network training proceeds by computing gradients of the loss with respect to each weight matrix. Recent work in optimization has shown that these gradients are approximately low-rank [22, 2], a property exploited by optimizers such as Shampoo [14] and Muon [18] for faster convergence. In vision models this structure is especially pronounced: the efective rank of a convolutional gradient is typically an order of magnitude smaller than the matrix dimension, a property we characterize empirically (section 4). The noise added by DP-SGD, however, is spectrally indiscriminate: isotropic Gaussian perturbation corrupts every singular direction equally, regardless of how much signal that direction carries. The result is a spectral mismatch (fig. 1b): the few directions that carry signal receive the prescribed noise, while the many directions that carry little signal become pure noise, yet contribute equally to the aggregate update.

While spectral orthogonalization is established in non-private optimization [18], its utility under diferential privacy is fundamentally regime-dependent. Replacing the noisy gradient with its nearest orthogonal matrix $U V ^ { \top }$ via polar decomposition (computed eficiently through Newton-Schulz iteration) is shown to transition from harmful to substantially beneficial at a predictable SNR threshold, providing the first principled criterion for when spectral structure can be recovered from the noise bulk in private training. By the post-processing theorem [10], this operation incurs zero additional privacy cost. Orthogonalization preserves the gradient’s singular vectors but sets all singular values to one, suppressing the inflated magnitudes of noise-dominated directions. This is beneficial when the true signal directions are recoverable from the noisy gradient, but harmful otherwise: at low SNR, orthogonalization normalizes the noise bulk to unit norm, producing a nearly random update that is strictly worse than the noisy-but-directionally-biased gradient used by DP-SGD.

This leads to a theoretical prediction: orthogonalization should improve utility when the gradient signal-to-noise ratio (SNR) is high enough for the true singular vectors to be recoverable from the noisy gradient, and be neutral or harmful otherwise. This prediction is verified empirically, revealing a phase transition governed by the SNR. At large batch sizes, the noise per element shrinks inversely with B, pushing the SNR past the recovery threshold. The crossover batch size depends on task complexity: it occurs at $B \approx 1 0 2 4$ on CIFAR-10, shifts to $B \approx 4 0 9 6$ to 8192 on CIFAR-100 where the gradient signal is ∼9× weaker, and is verified across three datasets in section 5.2.

This phenomenon is characterized through a systematic empirical study spanning batch sizes $B \in \{ 2 5 6 , 5 1 2 , \ldots , 8 1 9 2 \}$ and privacy budgets $\varepsilon \in \{ 2 , 3 , 4 , 6 , 8 \}$ . The contributions of this work are three-fold. First, a recovery condition (Proposition 1) linking batch size, noise multiplier, and spectral gap is derived and verified across CIFAR-10, CIFAR-100, and SVHN, revealing that spectra orthogonalization transitions from harmful to substantially beneficial as the gradient SNR crosses a predictable threshold. Second, spectral gradient orthogonalization via approximate polar decomposition (Newton-Schulz iteration) is introduced as a deterministic post-processing step that preserves the $( \varepsilon , \delta ) – \mathrm { D P }$ guarantee at zero additional privacy cost, achieving +20.9% over DP-SGD on WRN-28-10 $( B = 4 0 9 6 , \varepsilon = 4 )$ and reducing inter-run variance by a factor of two to three. Third, the complementarity of spectral and temporal denoising is demonstrated: combining orthogonalization with low-pass filtering (DOPPLER) yields 50.3% on CIFAR-10 (ε = 4), the highest accuracy in any tested configuration.

## 2 Related Work

Diferentially private optimization. DP-SGD [1] provides $( \varepsilon , \delta ) – \mathrm { D P }$ guarantees via per-sample gradient clipping and Gaussian noise addition. Subsequent work has improved the utility-privacy trade-of through better accounting [26, 4], larger batch training [7], and architectural choices [28]. DP-Adam [20] applies adaptive learning rates to privatized gradients but has shown limited benefit at moderate privacy budgets in our experiments, consistent with findings that adaptive methods require careful tuning under DP [6].

Noise reduction via spectral and temporal filtering. A growing line of work reduces the impact of DP noise through spectral or temporal post-processing. Spectral-DP [12] adds calibrated noise in the frequency domain rather than the spatial domain and applies spectral filtering, achieving the same $( \varepsilon , \delta ) – \mathrm { D P }$ guarantee with lower efective perturbation. DOPPLER [41] treats the privatized gradient sequence as a noisy signal and applies a low-pass filter to suppress high-frequency noise components while preserving low-frequency gradient information, yielding three to ten percent accuracy improvements across datasets and models. DiSK [40] extends this direction with a simplified Kalman filter that adapts its gain over training, achieving best-reported from-scratch results on CIFAR-100 and ImageNet-1k using augmentation multiplicity and architecture-specific tuning. Other approaches include frequency-domain gradient computation (GReDP [34]) and joint spectral-temporal filtering (FFTKF [38]).

These methods modify the privacy mechanism or its inputs. Spectral-DP and GReDP shape where noise enters by operating in the frequency domain, while DOPPLER and DiSK filter the privatized gradient across the training trajectory. Anisotropic Gaussian mechanisms [17] take a related route by reshaping the noise covariance at injection, which requires estimating that covariance from public data or a separate privacy budget and modifying the accounting. Spectral orthogonalization difers on the axis that is central to this work. It leaves the standard isotropic mechanism untouched and applies only deterministic post-processing to the already-privatized gradient, recovering directional signal from its low-rank matrix structure at zero additional privacy cost, with no covariance estimation and no change to the accounting. Because it targets matrix structure rather than component-wise magnitudes, it is complementary to the frequency-domain and temporal methods above (section 4.3).

Parameter-eficient private fine-tuning. Parameter-eficient methods (e.g., DP-BiTFiT [5], low-rank reparametrization [37], and LoRA [16, 23]) reduce trainable parameters to lower perparameter noise. In this regime, all optimizers achieve similar accuracy, consistent with the finding that low-rank constraints remove the spectral mismatch that orthogonalization addresses in fullparameter training.

Gradient structure and low-rank methods. Neural network gradients exhibit low-rank structure [22, 2]: most gradient energy concentrates in a small subspace whose dimension correlates with task complexity. GaLore [42] projects gradients into a low-rank subspace for memory eficiency. Our work difers in applying spectral processing as a post-processing step on the already-privatized gradient, preserving the DP guarantee by the post-processing theorem [10].

Spectral methods in optimization. Shampoo [14], SOAP [33], and Muon [18] exploit spectral structure for preconditioning or orthogonalization in non-private training. Their interaction with DP noise injection has not been studied; spectral gradient orthogonalization is investigated here in the DP setting, identifying the batch-size regime in which it provides a benefit.

## 3 Method

## 3.1 Preliminaries

Diferential Privacy. A randomized mechanism M satisfies $( \varepsilon , \delta )$ -diferential privacy [9] if for all adjacent datasets $D , D ^ { \prime }$ difering in a single record and all measurable sets S: $\operatorname* { P r } [ \mathcal { M } ( D ) \in S ] \leq$ $e ^ { \varepsilon } \operatorname* { P r } [ \mathcal { M } ( D ^ { \prime } ) \in S ] + \delta$

DP-SGD. Each per-sample gradient $g _ { i }$ is clipped to bound its Frobenius norm: $\bar { g } _ { i } ~ = ~ g _ { i }$ $\operatorname* { m i n } ( 1 , C / \| g _ { i } \| _ { F } )$ , where $C$ is the clipping threshold. The clipped gradients are averaged and perturbed:

$$
\tilde { G } = \frac { 1 } { B } \left( \sum _ { i = 1 } ^ { B } \bar { g } _ { i } + \mathcal { N } ( 0 , \sigma ^ { 2 } C ^ { 2 } \mathbf { I } ) \right) ,\tag{1}
$$

where $\sigma$ is the noise multiplier calibrated via Rényi DP accounting [26].

Post-Processing Theorem. Any function of the output of a diferentially private mechanism preserves the same $( \varepsilon , \delta )$ guarantee [10]. This is the key property enabling our approach: all spectral processing is applied after noise addition.

## 3.2 Spectral Gradient Orthogonalization

For a 2D parameter tensor, the method proceeds as:

Step 0: DP-SGD Gradient. Per-sample gradients are clipped and noised exactly as in standard DP-SGD (eq. (1)), producing the privatized gradient $\tilde { G } \in \mathbb { R } ^ { m \times n }$ . All subsequent steps operate on $\tilde { G }$ and are therefore post-processing.

Step 1: Momentum Accumulation. Following the Muon formulation [18], EMA-style Nesterov momentum is applied to the privatized gradient $\tilde { G } \mathrm { : }$ $M _ { t } \ : = \ : \beta M _ { t - 1 } + ( 1 - \beta ) \tilde { G } _ { t } , \ : \hat { M } _ { t }$ = $( 1 - \beta ) \tilde { G } _ { t } + \beta M _ { t }$ , where $\beta$ is the momentum coeficient and $\hat { M } _ { t }$ is the Nesterov lookahead. The $( 1 - \beta )$ scaling produces a weighted average rather than an accumulation, following the original Muon implementation.

Step 2: Orthogonalization via Newton-Schulz. The momentum matrix is orthogonalized using five iterations of a quintic Newton-Schulz scheme [18]:

$$
X _ { 0 } = \hat { M } _ { t } / \| \hat { M } _ { t } \| _ { F } , \quad X _ { k + 1 } = a X _ { k } + b X _ { k } X _ { k } ^ { \top } X _ { k } + c X _ { k } ( X _ { k } X _ { k } ^ { \top } ) ^ { 2 } ,\tag{2}
$$

where $( a , b , c ) = ( 3 . 4 4 4 5 , - 4 . 7 7 5 0 , 2 . 0 3 1 5 )$ are coeficients optimized for convergence speed. Frobeniusnorm initialization ensures all singular values of $X _ { 0 }$ are at most one (since $\| X _ { 0 } \| _ { 2 } \le \| X _ { 0 } \| _ { F } = 1 )$ which is suficient for Newton-Schulz convergence. The output $X _ { K } \approx U V ^ { \top }$ , the closest orthogonal matrix to $\hat { M } _ { t }$ . The iterations require only matrix multiplications. The resulting wall-clock and memory overhead is modest and is measured per configuration in supp material.

Step 3: Dimension Scaling. A correction factor $\hat { G } = \sqrt { \operatorname* { m a x } ( 1 , m / n ) } \cdot X _ { K }$ accounts for rectangular gradients.<sup>0</sup> The parameter update is $W _ { t + 1 } = W _ { t } - \eta \hat { G }$ . For 1D parameters (biases, layer norms), standard DP-SGD with momentum is used.

## 3.3 Magnitude-Preserving Variant and Privacy

Pure orthogonalization discards all singular value information. At higher SNR the leading singular value carries a meaningful magnitude signal, so a variant rescales the orthogonal update:

$$
\hat { G } _ { \mathrm { s c a l e d } } = \sigma _ { 1 } ( M _ { t } ) \cdot \sqrt { \operatorname* { m a x } \biggl ( 1 , \frac { m } { n } \biggr ) } \cdot X _ { K } ,\tag{3}
$$

where $M _ { t } = \beta M _ { t - 1 } + \tilde { G } _ { t }$ is a heavy-ball momentum bufer (without Nesterov lookahead), $X _ { K }$ is the Newton-Schulz output applied to $M _ { t }$ , and $\sigma _ { 1 } ( M _ { t } )$ is its spectral norm. Since $\sigma _ { 1 } ( M _ { t } )$ is computed from the noisy bufer, it overestimates the clean magnitude by up to $\| N \| _ { 2 } ,$ a bias that diminishes as the SNR increases. Both variants apply only deterministic post-processing to the Gaussian mechanism output $( \mathrm { { e q . } \ ( 1 ) ) ; }$ by the post-processing theorem, the privacy guarantee is identical to DP-SGD with no additional cost.

Table 1: Measured gradient SNR $( \| G \| _ { F } / \| N \| _ { F } )$ for WRN-16-4 at $\varepsilon = 4$ , averaged over the first epoch. σ is the noise multiplier (identical for both datasets at $N = 5 0 \mathrm { K } )$ . Signal and noise norms are shown for CIFAR-10. As training progresses, gradient structure typically becomes more lowrank, which lowers the recovery threshold.
<table><tr><td>B</td><td> $\sigma$ </td><td> $\| G \| _ { F }$  (signal)</td><td> $\| N \| _ { F }$  (noise)</td><td>SNR (C-10)</td><td>SNR (C-100)</td></tr><tr><td>256</td><td>0.88</td><td>0.33</td><td>5.71</td><td>0.06</td><td>0.05</td></tr><tr><td>512</td><td>1.08</td><td>0.34</td><td>3.50</td><td>0.10</td><td>0.07</td></tr><tr><td>1024</td><td>1.38</td><td>0.57</td><td>2.23</td><td>0.25</td><td>0.09</td></tr><tr><td>2048</td><td>1.83</td><td>1.00</td><td>1.48</td><td>0.68</td><td>0.12</td></tr><tr><td>4096</td><td>2.49</td><td>1.76</td><td>1.01</td><td>1.75</td><td>0.20</td></tr></table>

In practice, the magnitude-preserving variant is recommended only at moderate-to-high SNR, where the spectral-norm estimate $\sigma _ { 1 } ( M _ { t } )$ is reliable. At low SNR magnitude bias dominates, and pure orthogonalization is preferable. It is also not recommended on SVHN where it shows unstable bimodal convergence.

## 4 Why Orthogonalization Helps at Large Batch Sizes

## 4.1 Signal-to-Noise Ratio in DP Gradients

The noisy gradient (1) decomposes as ${ \tilde { G } } = G + N$ , where $\begin{array} { r } { G = \frac { 1 } { B } \sum _ { i } \bar { g } _ { i } } \end{array}$ is the clipped average and $\begin{array} { r } { N \sim \mathcal { N } ( 0 , \frac { \sigma ^ { 2 } C ^ { 2 } } { B ^ { 2 } } \mathbf { I } ) } \end{array}$ is the privacy noise. The noise N is i.i.d. Gaussian by construction of the Gaussian mechanism [1, 9] and is independent of the data; clipping afects the signal G but not the noise distribution. The per-element noise standard deviation scales as $\sigma C / B \mathrm { : }$ larger batches reduce the noise magnitude relative to the signal.

The signal G is approximately low-rank, with efective rank [29] erank $\begin{array} { r } { ( G ) = ( \sum _ { i } \sigma _ { i } ( G ) ) ^ { 2 } / \sum _ { i } \sigma _ { i } ( G ) ^ { 2 } \ll } \end{array}$ min $( m , n )$ . After adding DP noise, the efective rank increases dramatically (e.g., from ∼ 11 to $\mathord { \sim } 1 9 7$ for WRN-16-4 at $\varepsilon = 4 )$ , as noise fills all singular directions.

table 1 and fig. 2 report measured SNR values across batch sizes for WRN-16-4 at $\varepsilon = 4$ . On CIFAR-10, the SNR increases from 0.06 at $B = 2 5 6$ to 1.75 at $B = 4 0 9 6$ , crossing 1.0 between $B = 2 0 4 8$ and $B = 4 0 9 6$ . On CIFAR-100, the SNR is substantially lower at every batch size (0.20 vs 1.75 at B = 4096; 0.12 vs 0.68 at $B = 2 0 4 8 )$ , because the gradient signal norm decreases with batch size on this harder task rather than growing as on CIFAR-10. The Frobenius SNR is a conservative proxy for the spectral recovery condition: since $\lVert N \rVert _ { F } / \lVert N \rVert _ { 2 } \approx \sqrt { \operatorname* { m i n } ( m , n ) }$ , a Frobenius SNR of 1 implies $\sigma _ { 1 } ( G ) > \| N \| _ { 2 }$ by a factor of $\sqrt { \operatorname* { m i n } ( m , n ) }$ , so the actual recovery threshold is reached at smaller batch sizes than the SNR = 1 line suggests in fig. 2. In particular, orthogonalization can recover the leading singular direction even when the Frobenius SNR is below 1, provided the gradient signal is concentrated in a spiked leading direction whose magnitude exceeds the noise operator norm $\| N \| _ { 2 }$

## 4.2 Spectral Recovery Threshold

The spectral-gap and SNR measurements used in this section (table 1) are obtained from a separate post-hoc analysis run. They are not released and are not used during training, hyperparameter selection, or model release. Deployed models are accounted for under standard DP-SGD, with no contribution from these statistics. The recovery threshold below is therefore an explanatory framework, not a private estimator for selecting batch size or enabling orthogonalization.

![](images/6f379c986e7da6a5eaa6a2963e3a29c132333e9c1b05719f37974202d2d7b4c9.jpg)

![](images/ab1a2ed180e6d4fc66463daca33906a698b77b860cabd28c46bc788156f4b9bf.jpg)  
Figure 2: Gradient SNR vs. batch size at $\varepsilon = 4 .$ . (a) CIFAR-10 SNR with signal and noise Frobenius norms; the horizontal dashed line marks $\mathrm { S N R } = 1$ , above which orthogonalization recovers useful gradient structure. (b) CIFAR-10 vs. CIFAR-100: the SNR gap widens with batch size, reaching ${ \sim } 9 \times$ at $B = 4 0 9 6$

Orthogonalization replaces $\tilde { G }$ with $U V ^ { \top }$ , projecting onto the Stiefel manifold [11]. This transformation preserves directions but normalizes magnitudes. The efect depends on the SNR:

Low SNR (small batch, small $\varepsilon )$ : The noise dominates. $U V ^ { \top }$ reflects the singular structure of the noise rather than the signal. Orthogonalization amplifies noisy directions to unit norm, providing no benefit.

High SNR (large batch, large ε): The signal’s top singular directions persist through the noise. By the Wedin sin θ theorem [35], the angle θ between the clean and noisy leading singular vectors satisfies sin $\theta \leq \| N \| _ { 2 } / ( \sigma _ { 1 } ( G ) { - } \bar { \sigma _ { 2 } } ( G ) )$ . When the spectral gap exceeds the noise magnitude, orthogonalization approximately recovers the signal directions.

The observed threshold behavior is conceptually analogous to the phase transitions studied in spiked matrix models [3], where signal recovery occurs only when the dominant singular value exceeds the noise operator norm. The Wedin sin θ bound provides a rigorous recovery condition without requiring the distributional assumptions of the spiked model.

Proposition 1 (Recovery condition). Let $\tilde { G } = G + N \in \mathbb { R } ^ { m \times n }$ with m $\leq n$ , where G has singular values $\sigma _ { 1 } ( G ) \geq \sigma _ { 2 } ( G ) \geq 0$ and N has i.i.d. entries with variance $\sigma ^ { 2 } C ^ { 2 } / B ^ { 2 }$ $B y$ the Wedin sin θ bound $[ 3 5 ]$ , the angle θ between the leading left singular vectors of $\tilde { G }$ and G satisfies:

$$
\sin \theta \leq \frac { \| N \| _ { 2 } } { \sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G ) } .\tag{4}
$$

Since $\| N \| _ { 2 } \leq \sigma C ( { \sqrt { m } } + { \sqrt { n } } ) / B$ with high probability $[ 3 1 ] ,$ , orthogonalization recovers the signal subspace when:

$$
B > \frac { \sigma C ( \sqrt { m } + \sqrt { n } ) } { \sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G ) } .\tag{5}
$$

The recovery condition is derived by applying the Wedin sin θ bound [35, 32] to the scaling laws of the Gaussian mechanism; a step-by-step derivation linking each bound to its source theorem is provided in the supplementary material, Section 1. The spectral gap $\sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G )$ is measured at epoch 1, the hardest regime for recovery since gradient structure becomes more low-rank as training progresses [22]; the threshold is therefore conservative. Equation (5) yields a minimum batch size that depends on the noise multiplier $\sigma ,$ the clipping threshold $C ,$ the layer dimensions $m , n ,$ , and the spectral gap of the clean gradient. Datasets with weaker gradient signal (i.e. smaller spectral gap) require proportionally larger batches, consistent with the ${ \sim } 9 \times \ \mathrm { S N R }$ gap between CIFAR-10 and CIFAR-100 (table 1) and the corresponding shift in the crossover batch size observed in section 5.2. Proposition 1 provides a necessary condition for leading singular vector recovery; full subspace recovery requires analogous gaps at each rank, which is verified empirically via the efective rank measurements in table 1.

Numerical validation. Plugging measured spectral gaps from ResNet-18 [15] (epoch 1, CIFAR-10) into eq. (5) with the noise multipliers from table 1, most layers satisfy the recovery condition already at $B = 2 5 6 .$ , and all layers do by B = 512 (per-layer thresholds in table 11). The networkwide bottlenecks are the deepest convolutions $( 5 1 2 \times 4 6 0 8$ , spectral gap $\Delta \sigma \approx 0 . 2 7 , \sqrt { m } + \sqrt { n } \approx 9 0 )$ and layer3.1.conv1 $( 2 5 6 \times 2 3 0 4 , \Delta \sigma \approx 0 . 1 4 )$ , which transition between $B = 2 5 6$ and $B = 5 1 2$ . This aligns with the empirical crossover for ResNet-18 in fig. 6(b), where DP-Muon already leads DP-SGD by $+ 5 . 8 \%$ at $B = 2 5 6$ (when most layers have recovered) and the gap widens steadily with batch size. For the wider WRN-16-4, increased layer dimensionality raises ${ \sqrt { m } } + { \sqrt { n } }$ and pushes the predicted $B ^ { * }$ higher, consistent with the observed crossover at $B \approx 2 0 4 8$

This aligns with the observed phase transition (table 1): at $B \leq 5 1 2 \ \mathrm { ( S N R < 0 . 1 ) }$ orthogonalization amplifies noise; at $B = 4 0 9 6 ~ \mathrm { ( S N R = 1 . 7 5 ) }$ the signal dominates and improvements are consistent. DP-SGD stagnates because it uses the noisy gradient as-is, gaining little from improved directional structure; orthogonalization converts that structure into a cleaner update, explaining why spectral methods scale faster with both B and $\varepsilon \ ( \mathrm { { f i g } . \ 4 ) }$ .

## 4.3 Comparison with Temporal Filtering

Recent methods such as DOPPLER [41] and DiSK [40] reduce DP noise by applying temporal filters to the sequence of privatized gradients, also as post-processing steps incurring no additional privacy cost. The two classes of methods scale diferently with batch size: temporal filtering provides roughly constant noise reduction determined by the filter coeficient $\beta _ { i }$ , largely independent of batch size. Spectral orthogonalization, by contrast, depends on the per-step SNR and grows more efective with batch size as the spectral gap exceeds the noise operator norm (section 4). At small batch sizes temporal filtering dominates; at large batch sizes spectral orthogonalization catches up and eventually dominates. Since the two target diferent noise structures (temporal averaging vs. spatial structure), they are complementary and can be combined (section 5.4).

## 5 Experiments

## 5.1 Experimental Setup

Architecture and Datasets. Wide ResNet (WRN-16-4) [39] is used for all from-scratch experiments. Three datasets are evaluated: CIFAR-10 and CIFAR-100 [21] (50K training samples each) and SVHN [27] (73K training samples). Per-sample Frobenius norm clipping with $C = 1 . 0$ and $\delta = 1 0 ^ { - 5 }$ is used throughout. The noise multiplier σ is calibrated via Rényi DP accounting [26, 36] to achieve the target ε. Training epochs are held fixed across batch sizes; larger batches therefore take fewer gradient steps per epoch.

![](images/243dfc7cafa40392d515105fddf538fa02225066376f907513523f3648a5d757.jpg)  
Figure 3: Test accuracy vs. batch size at $\varepsilon = 4$ across three datasets. (a) CIFAR-10: DP-SGD is flat while spectral methods scale upward (+9.4% at $B = 4 0 9 6 )$ . (b) CIFAR-100: the crossover shifts to larger batch sizes, consistent with the lower SNR. (c) SVHN: DP-SGD degrades with batch size while DP-Muon improves (+14.9% at $B = 4 0 9 6 )$ . Vertical dotted lines mark the approximate $\mathrm { S N R } \approx 1$ threshold. Mean over three seeds.

Methods. Eight methods are compared, all sharing the same per-sample clipping and noise addition: (i) DP-SGD (momentum $\beta = 0 . 9 , \ \mathrm { L R } = 0 . 3 )$ ; (ii) DP-Adam [20] $( \beta _ { 1 } = 0 . 9 , \ \beta _ { 2 } = 0 . 9 9 9$ $\mathrm { L R } { = } 0 . 0 0 1 )$ ; (iii) DP-Muon $( U V ^ { \top }$ orthogonalization, Nesterov β = 0.95, LR = 0.02); (iv) DP-Muon-S $( \sigma _ { 1 } \cdot U V ^ { \top }$ , eq. (3), heavy-ball $\beta = 0 . 9 , \ \mathrm { L R } = 0 . 3 )$ ; (v) DiSK-SGD [40] (Kalman filter, $\kappa { = } 1 . 5 .$ , on DP-SGD); (vi) DiSK-Muon (Kalman filter on DP-Muon); (vii) DOPPLER-SGD [41] (EMA low-pass on DP-SGD); (viii) DOPPLER-Muon (EMA low-pass on DP-Muon). All methods use cosine learning rate decay with linear warmup. Results report best test accuracy (%) across epochs, averaged over three seeds (42, 43, 44) with standard deviation. DP-Muon uses $\beta = 0 . 9 5$ following the original Muon implementation [18]; the orthogonalization step, not the momentum coeficient, drives the gain, as DP-Muon-S uses identical $\beta = 0 . 9$ heavy-ball momentum to DP-SGD yet achieves comparable improvements.

## 5.2 Batch Size Sweep

Batch size is varied at a fixed privacy budget of $\varepsilon = 4$ . tables 2 and 3 report results on CIFAR-10/100; fig. 3 visualizes the trends across all three datasets.

CIFAR-10: At $B = 2 5 6$ , all methods perform within 1.3% of each other, consistent with the low-SNR regime (SNR = 0.06, table 1). DP-SGD remains flat across batch sizes (37.0 to 39.7%), while spectral methods scale upward: DP-Muon reaches 48.7% at $B = 4 0 9 6 \ ( + 9 . 4 \% )$ . DP-Muon-S peaks at $B = 2 0 4 8 \ ( 4 7 . 8 \% )$ and declines at $B = 4 0 9 6$ , suggesting magnitude preservation is less beneficial at high SNR. DP-Adam shows no batch-size scaling [6].

CIFAR-100: The gradient SNR is ${ \sim } 9 \times$ lower than on CIFAR-10 (table 1), so Proposition 1 predicts a proportionally larger batch is needed. At $B \leq 1 0 2 4$ , DP-Muon trails DP-SGD by 3 to 5%: in this low-SNR regime, orthogonalization replaces a noisy but directionally biased gradient with a nearly random orthogonal update, which is strictly worse (as formalized by the recovery condition of Proposition 1). Direction recovery (fig. 1c) exceeds 0.99 only at B = 4096 (vs. B = 1024 on CIFAR-10), yet the accuracy crossover lags because orthogonalization normalizes noise-only directions (the Marchenko–Pastur bulk [25]) to unit norm alongside recovered signal directions. With tuned learning rates, DP-SGD peaks at B = 2048 (29.3%) and declines to 21.6% at B = 8192, while DP-Muon increases steadily to 25.8%, overtaking by +4.2% at the largest batch size.

Table 2: Test accuracy (%) on CIFAR-10 at $\varepsilon = 4 .$ Mean ± std over three seeds. Best per column in bold.
<table><tr><td>Method</td><td>B=256</td><td> $B { = } 5 1 2$ </td><td>B=1024</td><td> $B { = } 2 0 4 8$ </td><td> $B { = } 4 0 9 6$ </td></tr><tr><td>DP-SGD</td><td> $3 9 . 4 \pm 1 . 2$ </td><td> $3 9 . 4 \pm 0 . 7$ </td><td> $3 9 . 7 \pm 2 . 6$ </td><td> $3 7 . 0 \pm 2 . 6$ </td><td> $3 9 . 3 \pm 0 . 3$ </td></tr><tr><td>DP-Adam</td><td> ${ \bf 3 9 . 9 \pm 2 . 9 }$ </td><td> $4 1 . 0 \pm 1 . 1$ </td><td> $4 1 . 8 \pm 1 . 8$ </td><td> $4 2 . 0 \pm 0 . 5$ </td><td> $3 8 . 8 \pm 1 . 7$ </td></tr><tr><td>DP-Muon</td><td> $3 9 . 0 \pm 0 . 5$ </td><td> $4 0 . 5 \pm 1 . 7$ </td><td> $4 3 . 3 \pm 1 . 6$ </td><td> $4 7 . 3 \pm 1 . 8$ </td><td> ${ \bf 4 8 . 7 \pm 0 . 6 }$ </td></tr><tr><td>DP-Muon-S</td><td> $3 8 . 7 \pm 1 . 0$ </td><td> ${ \bf 4 4 . 0 \pm 1 . 9 }$ </td><td> ${ \bf 4 4 . 5 \pm 2 . 9 }$ </td><td> ${ \bf 4 7 . 8 \pm 1 . 7 }$ </td><td> $4 6 . 3 \pm { 0 . 8 }$ </td></tr></table>

Table 3: Test accuracy (%) on CIFAR-100 at $\varepsilon = 4$ with tuned learning rates (DP-SGD $\eta = 1 . 0 ,$ DP-Adam $\eta { = } 0 . 0 1$ , DP-Muon η =0.1, DP-Muon-S η =0.5). Mean ± std over three seeds. Best per column in bold.
<table><tr><td>Method</td><td> $B { = } 5 1 2$ </td><td> $B { = } 1 0 2 4$ </td><td>B=2048</td><td> $B { = } 4 0 9 6$ </td><td>B=8192</td></tr><tr><td>DP-SGD</td><td> $2 3 . 5 \pm 1 . 0$ </td><td> $2 6 . 6 \pm 2 . 6$ </td><td> ${ \bf 2 9 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 2 6 . 8 \pm 0 . 6 }$ </td><td> $2 1 . 6 \pm 0 . 9$ </td></tr><tr><td>DP-Adam</td><td> ${ \bf 2 5 . 5 \pm 1 . 3 }$ </td><td> ${ \bf 2 7 . 4 \pm 1 . 8 }$ </td><td> $2 8 . 9 \pm 0 . 7$ </td><td> $2 6 . 0 \pm 0 . 7$ </td><td> $1 8 . 6 \pm 1 . 2$ </td></tr><tr><td>DP-Muon</td><td> $1 9 . 4 \pm 1 . 7$ </td><td> $2 2 . 0 \pm 1 . 0$ </td><td> $2 5 . 8 \pm 0 . 7$ </td><td> $2 5 . 9 \pm 0 . 5$ </td><td> ${ \bf 2 5 . 8 \pm 0 . 6 }$ </td></tr><tr><td>DP-Muon-S</td><td> $2 0 . 9 \pm 1 . 2$ </td><td> $2 4 . 8 \pm 0 . 3$ </td><td> $2 6 . 5 \pm 0 . 7$ </td><td> $2 6 . 3 \pm 0 . 9$ </td><td> $2 2 . 6 \pm 1 . 0$ </td></tr></table>

SVHN: DP-SGD accuracy decreases monotonically (fig. 3c) from 38.6% at B = 256 to 20.3% at B = 4096. DP-Muon increases by 14.9% from 27.8% to 35.2% at the largest batch size. We omit DP-Muon-S due to bimodal convergence.

Cross-Dataset Summary: The crossover batch size shifts with SNR: B ≈ 1024 on CIFAR-10, B ≈ 4096 to 8192 on CIFAR-100, and B ≈ 256 on SVHN, consistent with Proposition 1. DiSK [40] reports 42.0% on CIFAR-100 (ε = 8) using augmentation multiplicity and WRN-28-10; isolating the interaction with spectral processing is deferred to future work.

## 5.3 Privacy Budget Sweep

Figure 4 shows accuracy at B = 2048 across privacy budgets $\varepsilon \in \{ 2 , 3 , 4 , 6 , 8 \}$ . DP-Muon accuracy scales steadily with the privacy budget (from 41.2% at ε = 2 to 50.8% at ε = 8 on CIFAR-10), while DP-SGD remains largely flat (36.7 to 40.0%). On CIFAR-100, DP-Muon-S outperforms DP-SGD at $\varepsilon \geq 3$ , with the advantage growing to +3.0% at ε = 8. On SVHN, DP-SGD is flat at ∼20% regardless of $\varepsilon ,$ while DP-Muon improves from 28.6% to 35.3%.

## 5.4 Comparison with Temporal Denoising

Recent temporal denoising methods, DiSK [40] (Kalman filtering) and DOPPLER [41] (low-pass filtering), reduce DP noise by filtering the sequence of privatized gradients across training steps. As discussed in section 4.3, temporal and spectral filtering target diferent noise reduction mechanisms and are in principle complementary. Table 4 tests this hypothesis on CIFAR-10 at ε = 4; fig. 5 visualizes the crossover. At $B \leq 5 1 2$ , temporal filtering dominates: DiSK-SGD reaches 49.0% at $B \ = \ 2 5 6 .$ , outperforming all spectral methods by 9%. At $B \ \geq \ 2 0 4 8$ , the pattern reverses: DOPPLER-Muon achieves 50.3% at $B = 4 0 9 6$ , the highest accuracy in any configuration. DiSK-SGD declines from 49.0% to 39.9% as fewer steps reduce the filter’s averaging window; DiSK-Muon outperforms DiSK-SGD above $B = 5 1 2$ , confirming that spectral processing improves temporal filtering once SNR is suficient.

![](images/ba5d8c7bbc05077f1c7de05acb82ce51630f0ad2e5b1ac5783e1c346d524e748.jpg)  
Figure 4: Test accuracy vs. ε at $B = 2 0 4 8$ across three datasets. DP-SGD stagnates on CIFAR-10 and SVHN (shaded bands) while spectral methods scale with the privacy budget. DP-Muon-S is strongest on CIFAR-100 at all ε. Mean over three seeds.

![](images/cd384fb6aa0d17405d2485ae974bd23442ff8c5edb3ea26516b6c91042a876a2.jpg)  
Figure 5: Temporal vs. spectral denoising on CIFAR-10 at $\varepsilon = 4 .$ . Temporal methods (DiSK-SGD, dashed brown) dominate at small batch sizes but decline as fewer gradient steps reduce the filter’s averaging window. Spectral methods (DP-Muon, solid green) improve with batch size as per-step SNR increases. DOPPLER-Muon (dotted pink) combines both mechanisms and reaches the highest accuracy (50.3%) at $B = 4 0 9 6$ . Error bars show ±1 std over three seeds.

Table 4: Temporal vs. spectral denoising on CIFAR-10 (ε = 4). DiSK-SGD applies the Kalman filter [40] (κ = 1.5) to DP-SGD. DOPPLER-Muon combines low-pass filtering [41] with spectral orthogonalization. Mean ± std over three seeds.
<table><tr><td>Denoising</td><td>Method</td><td>B=256</td><td>B=512</td><td>B=1024</td><td> $B { = } 2 0 4 8$ </td><td> $B { = } 4 0 9 6$ </td></tr><tr><td>None</td><td>DP-SGD</td><td> $3 9 . 4 \pm 1 . 2$ </td><td> $3 9 . 4 \pm 0 . 7$ </td><td> $3 9 . 7 \pm 2 . 6$ </td><td> $3 7 . 0 \pm 2 . 6$ </td><td> $3 9 . 3 \pm 0 . 3$ </td></tr><tr><td>Temporal</td><td>DiSK-SGD</td><td> ${ \bf 4 9 . 0 \pm 2 . 8 }$ </td><td> ${ \bf 4 7 . 8 \pm 1 . 0 }$ </td><td> $4 6 . 7 \pm 1 . 3$ </td><td> $4 3 . 5 \pm 2 . 0$ </td><td> $3 9 . 9 \pm 0 . 6$ </td></tr><tr><td>Temporal</td><td>DOPPLER-SGD</td><td> $3 7 . 7 \pm 2 . 0$ </td><td> $3 9 . 3 \pm 0 . 9$ </td><td> $3 9 . 3 \pm 2 . 0$ </td><td> $3 4 . 2 \pm { 1 . 8 }$ </td><td> $3 2 . 3 \pm 1 . 1$ </td></tr><tr><td>Spectral</td><td>DP-Muon</td><td> $3 9 . 0 \pm 0 . 5$ </td><td> $4 0 . 5 \pm 1 . 7$ </td><td> $4 3 . 3 \pm 1 . 6$ </td><td> $4 7 . 3 \pm 1 . 8$ </td><td> $4 8 . 7 \pm 0 . 6$ </td></tr><tr><td>Both</td><td>DiSK-Muon</td><td> $4 1 . 7 \pm 2 . 7$ </td><td> $4 7 . 0 \pm 1 . 0$ </td><td> ${ \bf 4 8 . 8 \pm 1 . 0 }$ </td><td> $4 7 . 6 \pm 0 . 8$ </td><td> $4 5 . 8 \pm 1 . 2$ </td></tr><tr><td>Both</td><td>DOPPLER-Muon</td><td> $4 0 . 1 \pm 1 . 6$ </td><td> $4 2 . 6 \pm 0 . 9$ </td><td> $4 4 . 9 \pm 2 . 0$ </td><td> ${ \bf 4 9 . 0 \pm 0 . 5 }$ </td><td> ${ \bf 5 0 . 3 \pm 1 . 0 }$ </td></tr></table>

Table 5: Two-point settings, best per column in bold. (Left) Full fine-tuning of ViT-Small on CIFAR-100 (ε=4), mean ± std over three seeds. (Right) From-scratch NF-ResNet-50 on ImageNet-1k (B=32,768, 120 epochs, K=4, three seeds).
<table><tr><td>Method</td><td>LR</td><td>B=2048</td><td>B=4096</td></tr><tr><td>DP-SGD</td><td>0.01</td><td> $8 6 . 9 \pm 1 . 1$ </td><td> $7 9 . 5 \pm 3 . 4$ </td></tr><tr><td>DP-Adam</td><td>0.0003</td><td> ${ \bf 8 9 . 4 \pm 0 . 3 }$ </td><td> ${ \bf 8 9 . 7 \pm 0 . 1 }$ </td></tr><tr><td>DP-Muon</td><td>0.002</td><td> $8 9 . 1 \pm 0 . 2 $ </td><td> $8 9 . 3 \pm 0 . 3$ </td></tr></table>

<table><tr><td>Method</td><td>ε=4</td><td>ε=8</td></tr><tr><td>DP-SGD</td><td>18.47</td><td>24.60</td></tr><tr><td>DP-AdamW [24]</td><td>10.37</td><td>18.64</td></tr><tr><td>DP-Muon</td><td>20.97</td><td>27.32</td></tr></table>

## 5.5 Fine-Tuning

In the parameter-eficient regime (LoRA, rank 8), all methods achieve 85.6 to 86.8%, a spread of 1.2%, confirming that the spectral benefit is specific to full-parameter training. To test whether the benefit extends to full fine-tuning, all parameters of ViT-Small [8] (pretrained on ImageNet-21k [30]) are unfrozen and fine-tuned on CIFAR-100 at $\varepsilon = 4$

DP-Muon matches DP-Adam within 0.4% at both batch sizes, while DP-SGD collapses from 86.9% to 79.5% at B = 4096. DP-Muon achieves this parity as a stateless post-processing step with no second-moment bufer, whereas DP-Adam doubles optimizer memory. That DP-SGD collapses while both remain stable indicates gradient processing is essential for fine-tuning at scale.

## 5.6 Architecture Diversity

To verify that the spectral benefit generalizes beyond WRN-16-4, the batch sweep is repeated on ResNet-18 and WRN-28-10 (CIFAR-10, ε = 4).

On ResNet-18, DP-Muon leads from $B = 2 5 6 \ ( + 5 . 8 \% )$ , consistent with eq. (5): narrower layers yield smaller ${ \sqrt { m } } + { \sqrt { n } } ,$ crossing the recovery threshold earlier. The gap grows to +14.9% at B = 4096. On WRN-28-10, the standard architecture in the DP optimization literature, the gap is even larger: DP-Muon reaches 56.2% at B = 2048 (+17.5%) and 54.9% at B = 4096 (+20.9% over DP-SGD). DP-SGD accuracy on WRN-28-10 monotonically decreases from 44.1% at B = 512 to 34.0% at B = 4096, as the spectral mismatch grows with parameter count. DP-Muon improves through B = 2048, confirming that orthogonalization is most efective when this mismatch is most severe.

![](images/1d6c1c8215dae1f8262caf46ef601baf0af2a51ab1b64877defc397b3bf036f1.jpg)

![](images/5805b72faff83061a8dfbe7edfe164467ec42f0eb24aba104a5727d05895d95e.jpg)  
Figure 6: Batch-size sweeps as smooth curves. (a) Tiny-ImageNet (WRN-16-4, ε=4). (b) Architecture sweep on CIFAR-10 (ε=4); ResNet-18 (solid), WRN-28-10 (dashed). Markers are measured points; curves are monotone cubic (PCHIP) interpolation.

## 5.7 Beyond CIFAR Resolution

Figure 6(a) repeats the batch sweep on Tiny-ImageNet (64 × 64, 200 classes, 100K images, WRN-$1 6 \ – 4 , \varepsilon = 4 )$ . DP-SGD degrades from 21.1% to 17.3% while spectral methods improve (DP-Muon-S: 21.3% at $B = 4 0 9 6 , \ + 4 . 0 \% )$ , with crossover between B = 1024 and 2048, consistent with Proposition 1. Table 5 extends the from-scratch evaluation to full ImageNet-1k scale (224 × 224, 1000 classes) with NF-ResNet-50, under an identical protocol for all optimizers. DP-Muon improves over DP-SGD by +2.50% at ε = 4 and +2.72% at ε = 8, with seven- and three-fold lower acrossseed variance. The spectral-recovery benefit therefore holds at standard scale, in the standalone setting without temporal denoising.

## 6 Conclusion

An SNR-governed phase transition determines the utility of spectral gradient processing in diferentially private training. The recovery condition derived from the Wedin bound (Proposition 1) provides a predictive threshold for singular vector recovery from isotropic noise, validated quantitatively against per-layer spectral measurements. Below this threshold, orthogonalization is harmful. Above it, the benefit is consistent and grows with batch size, privacy budget, and model capacity. Empirical results across four architectures (WRN-16-4, ResNet-18, WRN-28-10, ViT-Small) and three resolutions confirm that while DP-SGD stagnates or degrades at scale, spectral orthogonalization enables continued scaling, achieving +20.9% over DP-SGD on WRN-28-10 (B = 4096) and +14.9% on ResNet-18. Complementarity with temporal filtering yields 50.3% on CIFAR-10 (ε = 4), the highest accuracy in any tested configuration.

Limitations and future work. DP-Muon-S exhibits bimodal convergence on SVHN due to noise-dominated $\sigma _ { 1 }$ fluctuations (section 3.3). Layer-wise selective orthogonalization and optimal singular value shrinkage [13] are natural extensions.

## References

[1] Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., Zhang, L.: Deep learning with diferential privacy. In: ACMConf.Comput.Commun.Secur. pp. 308–318 (2016). https://doi.org/10.1145/2976749.2978318

[2] Aghajanyan, A., Zettlemoyer, L., Gupta, S.: Intrinsic dimensionality explains the efectiveness of language model fine-tuning. In: Assoc.Comput.Linguist. pp. 7319–7328 (2021)

[3] Baik, J., Arous, G.B., Péché, S.: Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. Annals of Probability 33(5), 1643–1697 (2005). https://doi. org/10.1214/009117905000000233

[4] Balle, B., Barthe, G., Gaboardi, M.: Privacy profiles and amplification by subsampling. J. Privacy and Confidentiality 10(1) (2020). https://doi.org/10.29012/jpc.726

[5] Bu, Z., Wang, Y.X., Zha, S., Karypis, G.: Diferentially private bias-term fine-tuning of foundation models. arXiv preprint arXiv:2210.00036 (2022)

[6] Bu, Z., Wang, Y.X., Zha, S., Karypis, G.: Diferentially private optimization on large model at small cost. In: Int.Conf.Mach.Learn. vol. 202, pp. 3192–3218 (2023)

[7] De, S., Berrada, L., Hayes, J., Smith, S.L., Balle, B.: Unlocking high-accuracy diferentially private image classification through scale. In: Adv.NeuralInform.Process.Syst. (2022)

[8] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: Int.Conf.Learn.Represent. (2021)

[9] Dwork, C., McSherry, F., Nissim, K., Smith, A.: Calibrating noise to sensitivity in private data analysis. In: TheoryofCryptographyConf. pp. 265–284 (2006). https://doi.org/10. 1007/11681878\_14

[10] Dwork, C., Roth, A.: The algorithmic foundations of diferential privacy. Found.TrendsTheor.Comput.Sci. 9(3–4), 211–407 (2014). https://doi.org/10.1561/ 0400000042

[11] Edelman, A., Arias, T.A., Smith, S.T.: The geometry of algorithms with orthogonality constraints. SIAM Journal on Matrix Analysis and Applications 20(2), 303–353 (1998). https://doi.org/10.1137/S0895479895290954

[12] Feng, C., Xu, N., Wen, W., Venkitasubramaniam, P., Ding, C.: Spectral-DP: Diferentially private deep learning through spectral perturbation and filtering. In: IEEE Symp. Security and Privacy. pp. 1944–1960 (2023). https://doi.org/10.1109/SP46215.2023.10179457

<sub>[13] Gavish, M., Donoho, D.L.: The optimal hard threshold for singular values is 4/</sub>√<sub>3.</sub> IEEETrans.Inform.Theory 60(8), 5040–5053 (2014). https://doi.org/10.1109/TIT.2014. 2323359

[14] Gupta, V., Koren, T., Singer, Y.: Shampoo: Preconditioned stochastic tensor optimization. In: Int.Conf.Mach.Learn. pp. 1842–1850 (2018)

[15] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEEConf.Comput.Vis.PatternRecog. pp. 770–778 (2016). https://doi.org/10.1109/CVPR. 2016.90

[16] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: Int.Conf.Learn.Represent. (2022)

[17] Ji, T., Li, P.: Less is more: Revisiting the Gaussian mechanism for diferential privacy. In: USENIX Security Symp. (2024)

[18] Jordan, K.: Muon: An optimizer for hidden layers in neural networks (2024), blog post, https://kellerjordan.github.io/posts/muon/, accessed 2026-06-26

[19] Kaissis, G.A., Makowski, M.R., Rückert, D., Braren, R.F.: Secure, privacy-preserving and federated machine learning in medical imaging. Nature Machine Intelligence 2(6), 305–311 (2020). https://doi.org/10.1038/s42256-020-0186-1

[20] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Int.Conf.Learn.Represent. (2015)

[21] Krizhevsky, A.: Learning multiple layers of features from tiny images. Tech. rep., University of Toronto (2009)

[22] Li, C., Farkhoor, H., Liu, R., Yosinski, J.: Measuring the intrinsic dimension of objective landscapes. In: Int.Conf.Learn.Represent. (2018)

[23] Li, X., Tramèr, F., Liang, P., Hashimoto, T.: Large language models can be strong diferentially private learners. In: Int.Conf.Learn.Represent. (2022)

[24] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: Int.Conf.Learn.Represent. (2019)

[25] Marchenko, V.A., Pastur, L.A.: Distribution of eigenvalues for some sets of random matrices. Matematicheskii Sbornik 114(4), 507–536 (1967)

[26] Mironov, I.: Rényi diferential privacy. In: IEEEComput.Secur.Found. pp. 263–275 (2017). https://doi.org/10.1109/CSF.2017.11

[27] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., Ng, A.Y.: Reading digits in natural images with unsupervised feature learning. In: NeurIPS Workshop on Deep Learning and Unsupervised Feature Learning (2011)

[28] Ponomareva, N., Hazimeh, H., Kurakin, A., Xu, Z., Denison, C., McMahan, H.B., Vassilvitskii, S., Chien, S., Thakurta, A.G.: How to DP-fy ML: A practical guide to machine learning with diferential privacy. J. Artificial Intelligence Res. 77, 1113–1201 (2023). https://doi.org/10. 1613/jair.1.14649

[29] Roy, O., Vetterli, M.: The efective rank: A measure of efective dimensionality. In: Eur.SignalProcess.Conf. pp. 606–610 (2007)

[30] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: ImageNet large scale visual recognition challenge. Int.J.Comput.Vis. 115(3), 211–252 (2015). https://doi.org/10.1007/ s11263-015-0816-y

[31] Vershynin, R.: Introduction to the non-asymptotic analysis of random matrices. Cambridge University Press (2012). https://doi.org/10.1017/CBO9780511794308.006

[32] Vu, V.Q., Lei, J.: Minimax sparse principal subspace estimation in high dimensions. The Annals of Statistics 41(6), 2905–2947 (2013), earlier version: arXiv:1211.0373, 2012

[33] Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson, L., Kakade, S.: SOAP: Improving and stabilizing Shampoo using Adam. arXiv preprint arXiv:2409.11321 (2024)

[34] Wang, H., Jiang, T., Guo, Y., Jia, X., Cai, C., Ding, C.: GReDP: A more robust approach for diferential privacy training with gradient-preserving noise reduction. arXiv preprint arXiv:2409.11663 (2024)

[35] Wedin, P.Å.: Perturbation bounds in connection with singular value decomposition. BIT Numerical Mathematics 12(1), 99–111 (1972). https://doi.org/10.1007/BF01932678

[36] Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., Mironov, I.: Opacus: User-friendly diferential privacy library in PyTorch. arXiv preprint arXiv:2109.12298 (2021)

[37] Yu, D., Zhang, H., Chen, W., Yin, J., Liu, T.Y.: Large scale private learning via low-rank reparametrization. In: Int.Conf.Mach.Learn. pp. 12208–12218 (2021)

[38] Yun, J.: FFTKF: Fast Fourier transform-based spectral and temporal gradient filtering for diferentially private optimization. arXiv preprint arXiv:2505.04468 (2025)

[39] Zagoruyko, S., Komodakis, N.: Wide residual networks. In: Brit.Mach.Vis.Conf. (2016)

[40] Zhang, X., Bu, Z., Balle, B., Hong, M., Razaviyayn, M., Mirrokni, V.: DiSK: Diferentially private optimizer with simplified Kalman filter for noise reduction. In: Int.Conf.Learn.Represent. (2025)

[41] Zhang, X., Bu, Z., Hong, M., Razaviyayn, M.: DOPPLER: Diferentially private optimizers with low-pass filter for privacy noise reduction. In: Adv.NeuralInform.Process.Syst. (2024)

[42] Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., Tian, Y.: GaLore: Memory-eficient LLM training by gradient low-rank projection. arXiv preprint arXiv:2403.03507 (2024)

## Appendix: Supplementary Material

## A Derivation of Proposition 1

Proposition 1 in the main text derives a minimum batch size for leading singular vector recovery. The derivation combines two standard results: the Wedin sin θ perturbation bound and a highprobability operator norm bound for Gaussian random matrices. The three steps are made explicit below.

Step 1: Wedin sin θ bound. Let $\tilde { G } = G + N$ where $G \in \mathbb { R } ^ { m \times n }$ is the clean (noiseless) gradient matrix and N is the additive noise. Let G have compact SVD $G = U \Sigma V ^ { \top }$ with singular values $\sigma _ { 1 } ( G ) \geq \sigma _ { 2 } ( G ) \geq \cdot \cdot \cdot \geq 0$ , and let $\tilde { G } = \tilde { U } \tilde { \Sigma } \tilde { V } ^ { \top }$ be the SVD of the noisy observation. The Wedin sin θ theorem [35] states the following.

Theorem (Wedin, 1972). Let $G , \tilde { G } \in \mathbb { R } ^ { m \times n }$ with ${ \tilde { G } } = G + N$ . If $\sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G ) > \| N \| _ { 2 }$ , then the angle θ between the leading ${ \it l e f t }$ singular vectors satisfies

$$
\sin \theta ( \tilde { u } _ { 1 } , u _ { 1 } ) \leq \frac { \| N \| _ { 2 } } { \sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G ) } .\tag{6}
$$

In the notation of [35], $\sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G )$ is the absolute spectral gap between the leading singular subspace and the remainder of the spectrum. When this gap exceeds the perturbation norm $\| N \| _ { 2 } ,$ the leading singular vector of the noisy matrix $\tilde { G }$ remains close to that of $G ;$ when the gap is smaller than the perturbation, no such guarantee holds and the recovered direction may be arbitrary.

Step 2: Gaussian operator norm bound. In the DP-SGD mechanism, the noise matrix $\boldsymbol { N } \in \mathbb { R } ^ { m \times n }$ has i.i.d. entries drawn from $\mathcal { N } ( 0 , \sigma ^ { 2 } C ^ { 2 } / B ^ { 2 } )$ , where $\sigma$ is the noise multiplier (set by the privacy accountant for a given $\varepsilon , \delta )$ , C is the per-sample clipping threshold, and B is the batch size. The operator norm of such a matrix is controlled by the following concentration result.

Theorem (Vershynin [31], Theorem 5.39). Let N be an $m \times n$ random matrix $( m \leq n )$ whose entries are i.i.d. $\textstyle { \mathcal { N } } ( 0 , \sigma _ { n } ^ { 2 } )$ . Then for any $t \geq 0$ ，

$$
\| N \| _ { 2 } \leq \sigma _ { n } ( { \sqrt { m } } + { \sqrt { n } } + t )\tag{7}
$$

with probability at least $1 - 2 \exp ( - t ^ { 2 } / 2 )$

Since N has i.i.d. entries with variance $\sigma ^ { 2 } C ^ { 2 } / B ^ { 2 }$ , the operator norm scales linearly: $\| N \| _ { 2 } ~ =$ $( \sigma C / B ) \| Z \| _ { 2 }$ where $Z$ has unit-variance entries. Applying the theorem to $Z$ with $\sigma _ { n } = \sigma C / B$ and taking $t = 0$ (median bound) gives:

$$
\| N \| _ { 2 } \leq \frac { \sigma C ( \sqrt { m } + \sqrt { n } ) } { B } .\tag{8}
$$

Step 3: Substitution and rearrangement. Substituting (8) into the Wedin condition $\| N \| _ { 2 } <$ $\sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G )$ :

$$
\frac { \sigma C ( \sqrt { m } + \sqrt { n } ) } { B } < \sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G ) .\tag{9}
$$

Rearranging for $B { : }$

$$
B > \frac { \sigma C ( \sqrt { m } + \sqrt { n } ) } { \sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G ) } .\tag{10}
$$

Table 6: Learning rate sweep on CIFAR-100 at B = 2048, ε = 4. Entries with ± are mean over three seeds; † denotes a single-seed run from a controlled sweep on the same hardware.
<table><tr><td>Method</td><td>LR</td><td>Accuracy</td></tr><tr><td>DP-SGD</td><td>0.3</td><td> $2 5 . 2 \pm 0 . 9$ </td></tr><tr><td>DP-Muon</td><td>0.02</td><td> $2 1 . 5 \pm 0 . 5$ </td></tr><tr><td>DP-Muon</td><td>0.1</td><td> $2 3 . 2 ^ { \dagger }$ </td></tr><tr><td>DP-Muon</td><td>0.3</td><td> $1 6 . 8 \pm 0 . 7$ </td></tr><tr><td>DP-Muon-S</td><td>0.3</td><td> ${ \bf 2 5 . 6 \pm 0 . 6 }$ </td></tr></table>

Table 7: Test accuracy (%) on CIFAR-10 at $B = 2 0 4 8$ across privacy budgets. Mean ± std over three seeds. Best per row in bold.
<table><tr><td>ε</td><td>DP-SGD</td><td>DP-Adam</td><td>DP-Muon</td><td> $\mathrm { D P \mathrm { - M u o n { - } S } }$ </td></tr><tr><td>2</td><td> $3 6 . 7 \pm 0 . 4$ </td><td> $4 0 . 6 \pm 2 . 6$ </td><td> $4 1 . 2 \pm 1 . 7$ </td><td> ${ \bf 4 2 . 6 \pm 1 . 6 }$ </td></tr><tr><td>3</td><td> $3 7 . 0 \pm 1 . 3$ </td><td> $4 0 . 7 \pm 1 . 3$ </td><td> $4 5 . 7 \pm 0 . 5$ </td><td> ${ \bf 4 8 . 0 \pm 3 . 4 }$ </td></tr><tr><td>4</td><td> $3 7 . 0 \pm 2 . 6$ </td><td> $4 2 . 0 \pm 0 . 5$ </td><td> $4 7 . 3 \pm 1 . 8$ </td><td> ${ \bf 4 7 . 8 \pm 1 . 7 }$ </td></tr><tr><td>6</td><td> $4 0 . 0 \pm 2 . 0$ </td><td> $4 1 . 8 \pm 0 . 7$ </td><td> ${ \bf 5 0 . 0 \pm 1 . 1 }$ </td><td> $4 8 . 5 \pm 0 . 9$ </td></tr><tr><td>8</td><td> $3 9 . 1 \pm 1 . 9$ </td><td> $4 3 . 6 \pm 1 . 0$ </td><td> ${ \bf 5 0 . 8 \pm 2 . 9 }$ </td><td> $4 9 . 6 \pm 1 . 3$ </td></tr></table>

This is the batch size threshold of Proposition 1. It depends on four quantities: the noise multiplier σ (set by the privacy budget ε), the clipping threshold C, the layer dimensions $m , n ,$ and the spectral gap $\sigma _ { 1 } ( G ) - \sigma _ { 2 } ( G )$ of the clean gradient. Datasets with weaker gradient signal $( i . e .$ , smaller spectral gap) require proportionally larger batch sizes, consistent with the ∼9× SNR gap between CIFAR-10 and CIFAR-100 reported in the main text.

This is a necessary condition for recovery of the leading singular vector only. Recovery of the full rank-r signal subspace requires analogous gap conditions at each singular value: $\sigma _ { k } ( G ) ~ \textrm { - }$ $\sigma _ { k + 1 } ( G ) > \| N \| _ { 2 }$ for $k = 1 , \ldots , r$ . In practice, gradient singular values decay rapidly (efective rank ≪ min(m, n)), so the leading gap is the binding constraint.

## B Learning Rate Ablation

Table 6 reports a learning rate sweep for all methods on CIFAR-100 at $B = 2 0 4 8 , \varepsilon = 4$ . DP-Muon without scaling is sensitive to LR (best at 0.02), while DP-Muon-S matches the standard DP-SGD learning rate of 0.3.

## C Per-ε Accuracy Tables

Tables 7–9 report test accuracy across privacy budgets $\varepsilon \in \{ 2 , 3 , 4 , 6 , 8 \}$ at fixed batch size $B = 2 0 4 8$ for all three datasets. On CIFAR-10, spectral methods lead at every ε. On CIFAR-100, DP-SGD leads at $\varepsilon = 2$ (below the recovery threshold) but DP-Muon-S overtakes from $\varepsilon \geq 3$ , with the gap widening to +3.0% at $\varepsilon = 8$

Table 8: Test accuracy (%) on CIFAR-100 at B = 2048 across privacy budgets. Mean ± std over three seeds. ∆ denotes DP-Muon-S minus DP-SGD.
<table><tr><td>ω</td><td>DP-SGD</td><td> $\mathrm { D P - A d a m }$ </td><td>DP-Muon</td><td> $\mathrm { D P \mathrm { - M u o n { - } S } }$ </td><td>∆</td></tr><tr><td>2</td><td> ${ \bf 2 1 . 3 \pm 0 . 1 }$ </td><td> $1 4 . 3 \pm { 0 . 8 }$ </td><td> $1 6 . 7 \pm 0 . 2$ </td><td> $2 0 . 7 \pm 1 . 0$ </td><td>-0.6</td></tr><tr><td>3</td><td> $2 3 . 5 \pm 0 . 5$ </td><td> $1 5 . 7 \pm 0 . 4$ </td><td> $1 9 . 6 \pm 0 . 3$ </td><td> ${ \bf 2 4 . 0 \pm 0 . 3 }$ </td><td>+0.5</td></tr><tr><td>4</td><td> $2 5 . 4 \pm 0 . 6$ </td><td> $1 6 . 6 \pm 0 . 5$ </td><td> $2 1 . 6 \pm 0 . 3$ </td><td> ${ \bf 2 6 . 4 \pm 0 . 3 }$ </td><td> $+ 1 . 0$ </td></tr><tr><td>6</td><td> $2 7 . 5 \pm 1 . 0$ </td><td> $1 8 . 7 \pm 0 . 2$ </td><td> $2 4 . 7 \pm 1 . 2$ </td><td> ${ \bf 2 8 . 7 \pm 1 . 0 }$ </td><td> $+ 1 . 2$ </td></tr><tr><td>8</td><td> $2 8 . 1 \pm 0 . 6$ </td><td> $2 0 . 0 \pm 0 . 1$ </td><td> $2 6 . 2 \pm 1 . 6$ </td><td> ${ \bf 3 1 . 1 \pm 0 . 6 }$ </td><td>+3.0</td></tr></table>

Table 9: Test accuracy (%) on SVHN at B = 2048 across privacy budgets. Mean ± std over three seeds.
<table><tr><td>ω</td><td>DP-SGD</td><td> $\mathrm { D P - A d a m }$ </td><td>DP-Muon</td></tr><tr><td>2</td><td> $2 0 . 9 \pm 1 . 5$ </td><td> $2 4 . 3 \pm 3 . 2$ </td><td> ${ \bf 2 8 . 6 \pm 0 . 4 }$ </td></tr><tr><td>3</td><td> $2 0 . 2 \pm 0 . 4$ </td><td> $2 3 . 2 \pm 3 . 0$ </td><td> ${ \bf 2 9 . 7 \pm 1 . 6 }$ </td></tr><tr><td>4</td><td> $2 0 . 1 \pm 0 . 5$ </td><td> $2 2 . 5 \pm 3 . 4$ </td><td> ${ \bf 3 0 . 0 \pm 2 . 4 }$ </td></tr><tr><td>6</td><td> $2 0 . 1 \pm 0 . 5$ </td><td> $2 2 . 5 \pm 4 . 4$ </td><td> ${ \bf 3 2 . 8 \pm 2 . 4 }$ </td></tr><tr><td>8</td><td> $2 0 . 2 \pm 0 . 5$ </td><td> $2 2 . 9 \pm 4 . 0$ </td><td> ${ \bf 3 5 . 3 \pm 2 . 9 }$ </td></tr></table>

Table 10: Test accuracy (%) on SVHN at $\varepsilon = 4 .$ $\mathrm { M e a n } \pm \mathrm { s t d }$ over three seeds.
<table><tr><td>Method</td><td>B=256</td><td>B=512</td><td>B=1024</td><td>B=2048</td><td>B=4096</td></tr><tr><td>DP-SGD</td><td> ${ \bf 3 8 . 6 \pm 1 2 . 4 }$ </td><td> $2 6 . 5 \pm 3 . 0$ </td><td> $2 1 . 3 \pm 1 . 8$ </td><td> $2 0 . 1 \pm 0 . 5$ </td><td> $2 0 . 3 \pm 0 . 4$ </td></tr><tr><td>DP-Adam</td><td> $2 9 . 0 \pm 3 . 4$ </td><td> $2 7 . 1 \pm 1 . 4$ </td><td> $2 4 . 4 \pm 0 . 7$ </td><td> $2 2 . 5 \pm 3 . 4$ </td><td> $2 1 . 8 \pm 1 . 9$ </td></tr><tr><td>DP-Muon</td><td> $2 7 . 8 \pm 0 . 2$ </td><td> ${ \bf 2 7 . 5 \pm 0 . 5 }$ </td><td> ${ \bf 2 8 . 6 \pm 0 . 8 }$ </td><td> ${ \bf 3 0 . 0 \pm 2 . 4 }$ </td><td> ${ \bf 3 5 . 2 \pm 3 . 1 }$ </td></tr></table>

## D SVHN Batch Size Sweep

Table 10 extends the batch size sweep to SVHN (ε = 4). DP-SGD achieves its best result at B = 256 but with extreme variance (±12.4%), reflecting bimodal convergence. DP-Muon improves monotonically from 27.8% to 35.2% as batch size increases, consistent with the spectral recovery prediction.

## E Per-Layer Recovery Threshold Validation

Table 11 reports the predicted recovery threshold $B ^ { * }$ from Proposition 1 (eq. (5)) for each 2D weight layer of ResNet-18, using spectral gaps measured at epoch 1 on CIFAR-10. The noise multiplier σ is computed self-consistently for each batch size via the PRV accountant $( \varepsilon = 4 , \delta = 1 0 ^ { - 5 }$ , 50 epochs). At B = 256, three layers (layer3.1.conv1, layer4.1.conv1, layer4.1.conv2) have $B ^ { * } > 2 5 6$ yielding an 86% recovery rate. By B = 512, all 21 layers satisfy the recovery condition. The binding layers are the deepest convolutions (512 × 4608), where large ${ \sqrt { m } } + { \sqrt { n } } \approx 9 0$ raises $B ^ { * }$ despite moderate spectral gaps. This is consistent with the empirical crossover for ResNet-18 in fig. 6(b), where DP-Muon already leads at B = 256 (when 86% of layers have recovered) and the gap widens steadily.

Table 11: Per-layer recovery threshold $B ^ { * }$ for ResNet-18 on CIFAR-10 at ε = 4. ✓: $B \geq B ^ { * }$ (leading singular vector recoverable). Spectral gaps measured at epoch 1.
<table><tr><td>Layer</td><td> $( m , n )$ </td><td> $\Delta \sigma$ </td><td>B=256</td><td>B=512</td><td>B=1024</td><td>B=2048</td><td>B=4096</td><td>B=8192</td></tr><tr><td>conv1</td><td>(64, 147)</td><td>0.418</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer1.0.c1</td><td>(64, 576)</td><td>0.156</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer1.0.c2</td><td>(64, 576)</td><td>0.216</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer1.1.c1</td><td>(64, 576)</td><td>0.236</td><td>√</td><td>√</td><td>V</td><td>√</td><td>v&gt;</td><td>V</td></tr><tr><td>layer1.1.c2</td><td>(64, 576)</td><td>0.247</td><td>V</td><td>V</td><td>V</td><td>√</td><td></td><td>r</td></tr><tr><td>layer2.0.c1</td><td>(128, 576)</td><td>0.318</td><td>√</td><td>V</td><td>V</td><td>√</td><td>&gt;&gt;</td><td>V</td></tr><tr><td>layer2.0.c2</td><td>(128, 1152)</td><td>0.204</td><td>√</td><td>V</td><td>V</td><td>√</td><td></td><td>V</td></tr><tr><td>layer2.0.ds</td><td>(128, 64)</td><td>0.122</td><td>V</td><td>V</td><td>V</td><td>V</td><td></td><td>√</td></tr><tr><td>layer2.1.c1</td><td>(128, 1152)</td><td>0.240</td><td>V</td><td>v&gt;</td><td>v&gt;</td><td>V</td><td></td><td></td></tr><tr><td>layer2.1.c2</td><td>(128, 1152)</td><td>0.182</td><td>V</td><td></td><td></td><td>V</td><td></td><td>v√</td></tr><tr><td>layer3.0.c1</td><td>(256, 1152)</td><td>0.324</td><td>√</td><td>√</td><td>V</td><td>√</td><td>v&gt;</td><td>V</td></tr><tr><td>layer3.0.c2</td><td>(256, 2304)</td><td>0.474</td><td>V</td><td>&gt;&gt;</td><td>v&gt;</td><td>v√</td><td></td><td>v√</td></tr><tr><td>layer3.0.ds</td><td>(256, 128)</td><td>0.241</td><td>V</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>layer3.1.c1</td><td>(256, 2304)</td><td>0.138</td><td></td><td>√</td><td>V</td><td>V</td><td></td><td>V</td></tr><tr><td>layer3.1.c2</td><td>(256, 2304)</td><td>0.595</td><td>V</td><td>V</td><td>V</td><td>√</td><td>vv&gt;</td><td></td></tr><tr><td>layer4.0.c1</td><td>(512, 2304)</td><td>0.421</td><td>V</td><td>V</td><td>V</td><td>V</td><td></td><td></td></tr><tr><td>layer4.0.c2</td><td>(512, 4608)</td><td>0.443</td><td>√</td><td>V</td><td>V</td><td>V</td><td>vv√</td><td></td></tr><tr><td>layer4.0.ds</td><td>(512, 256)</td><td>0.568</td><td>V</td><td>V</td><td>V</td><td>V</td><td></td><td></td></tr><tr><td>layer4.1.c1</td><td>(512, 4608)</td><td>0.272</td><td></td><td>√</td><td>V</td><td>V</td><td></td><td></td></tr><tr><td>layer4.1.c2</td><td>(512, 4608)</td><td>0.274</td><td></td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td></tr><tr><td>fc</td><td>(10, 512)</td><td>1.671</td><td>√</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td></tr><tr><td colspan="9">Recovery rate 86% 100% 100%</td></tr></table>

## F ImageNet-1k Gradient Spectrum Analysis

To evaluate whether the spectral structure required by Proposition 1 persists at ImageNet-1k resolution (224 × 224, 1000 classes), the gradient spectrum of a randomly initialized ResNet-18 (GroupNorm) is measured on a single batch of ImageNet-1k training data. Table 12 reports the predicted recovery threshold $B ^ { * }$ for representative layers, using noise multipliers calibrated for ε = 8, δ = 1/N<sup>1.1</sup> (N = 1,281,167), and 20 epochs.

At B = 256, 67% of layers (14/21) satisfy the recovery condition. Recovery increases to 81% at B = 512, 95% at B = 2048, and 100% at $B = 8 1 9 2$ . The binding layer is layer1.0.conv2 (64 × 576, $\Delta \sigma = 0 . 0 5 6 , B ^ { * } \approx 5 8 0 0 )$ , where the small spectral gap relative to layer dimensions requires very large batches. Compared to CIFAR-10 (section E), ImageNet gradients exhibit smaller spectral gaps in early layers, consistent with the higher task complexity (1000 vs. 10 classes). The predicted crossover at B ≈ 2048 to 8192 is consistent with the scaling patterns observed on CIFAR-100 (200 classes) in the main text.

## G Optimization Budget vs. Spectral Recovery

A potential confound in the batch size sweep is that larger batches reduce the total number of gradient steps (at fixed epochs), so DP-SGD’s degradation could reflect fewer optimization steps rather than spectral mismatch. Table 13 disentangles these efects by reporting both the step count and the gradient SNR alongside accuracy for DP-SGD and DP-Muon.

DP-SGD accuracy shows weak positive correlation with step count (Pearson r = +0.35) and no correlation with SNR $( r = - 0 . 1 5 )$ , consistent with a step-limited optimizer that cannot exploit improved per-step signal quality. DP-Muon accuracy shows strong positive correlation with SNR $( r = + 0 . 8 7 )$ and strong negative correlation with step count $( r = - 0 . 9 0 )$ , confirming that its gains come from spectral recovery rather than optimization budget. This is the expected signature of the phase transition: DP-Muon’s accuracy tracks the signal quality per step, not the number of steps.

Table 12: Per-layer recovery threshold $B ^ { * }$ for ResNet-18 on ImageNet-1k at $\varepsilon = 8 .$ . Representative layers shown. ✓: $B \geq B ^ { * }$
<table><tr><td>Layer</td><td> $( m , n )$ </td><td> $\Delta \sigma$ </td><td>B=256</td><td> $B { = } 5 1 2$ </td><td>B=1024</td><td>B=2048</td><td>B=4096</td><td>B=8192</td></tr><tr><td>conv1</td><td>(64, 147)</td><td>0.592</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer1.0.c1</td><td>(64, 576)</td><td>0.094</td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer1.0.c2</td><td>(64, 576)</td><td>0.056</td><td></td><td></td><td>一</td><td>一</td><td>一</td><td>√</td></tr><tr><td>layer2.0.c1</td><td>(128, 576)</td><td>0.318</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer2.0.c2</td><td>(128, 1152)</td><td>0.204</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer3.0.c2</td><td>(256, 2304)</td><td>0.474</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer4.0.c2</td><td>(512, 4608)</td><td>0.443</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>layer4.1.c1</td><td>(512, 4608)</td><td>0.272</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>fc</td><td>(10, 512)</td><td>1.671</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Recovery rate (all 21 layers)</td><td></td><td></td><td>67%</td><td>81%</td><td>81%</td><td>95%</td><td>95%</td><td>100%</td></tr></table>

Table 13: DP-SGD accuracy tracks step count while DP-Muon tracks gradient SNR. CIFAR-10, WRN-16-4, ε = 4. Pearson r in last row.
<table><tr><td>B</td><td>Steps</td><td>SNR</td><td>DP-SGD</td><td>DP-Muon</td></tr><tr><td>256</td><td>9750</td><td>0.06</td><td>39.4</td><td>39.0</td></tr><tr><td>512</td><td>4850</td><td>0.10</td><td>39.4</td><td>40.5</td></tr><tr><td>1024</td><td>2400</td><td>0.25</td><td>39.7</td><td>43.3</td></tr><tr><td>2048</td><td>1200</td><td>0.68</td><td>37.0</td><td>47.2</td></tr><tr><td>4096</td><td>600</td><td>1.75</td><td>39.3</td><td>48.7</td></tr><tr><td colspan="3">Pearson r</td><td> $r _ { \mathrm { s t e p s } } { = } + 0 . 3 5$ </td><td> $r _ { \mathrm { S N R } } { = } + 0 . 8 7$ </td></tr></table>

## H WRN-28-10 Convergence Curves

Figure 7 shows test accuracy across training for DP-SGD and DP-Muon on WRN-28-10 at $B =$ 4096, ε = 4. The advantage of spectral orthogonalization emerges by epoch 10 and widens consistently, confirming that the +20.9% improvement (Table 5 in the main text) reflects a sustained convergence advantage rather than a late-epoch fluctuation. DP-Muon reaches $5 4 . 9 \pm 0 . 8 \%$ with low inter-seed variance, while DP-SGD plateaus at $3 4 . 0 \pm 2 . 0 \%$

## I Variance Reduction Across Datasets

Figure 8 visualizes the standard deviation of test accuracy across three seeds as a function of batch size for all three datasets. On SVHN at B = 256, DP-SGD exhibits a standard deviation of 10.1%, indicating bimodal convergence: some seeds converge to ∼50% while others remain near 20%. DP-Muon’s standard deviation on the same configuration is 0.2%, a 50× reduction. Across all datasets, spectral methods maintain lower variance at large batch sizes, consistent with the hypothesis that orthogonalization stabilizes the optimization trajectory by projecting onto a consistent subspace regardless of noise realization.

![](images/847a1ecfc2838649406729d838fea93c4662b591f5d07a70261bad78e581929a.jpg)  
Figure 7: Test accuracy vs. epoch for DP-SGD and DP-Muon on WRN-28-10 (CIFAR-10, B = 4096, $\varepsilon = 4 )$ . Mean ± std over three seeds. The gap emerges by epoch 10 and widens steadily.

![](images/f5bf951d4b01172dd472d24dd95d9c49c9e5d8e9d3b996a06115ca853bde8d86.jpg)  
Figure 8: Standard deviation of test accuracy (over three seeds) vs. batch size across three datasets $( \varepsilon = 4 )$ . Spectral methods (DP-Muon, DP-Muon-S) maintain consistently lower variance, especially at extreme batch sizes where DP-SGD variance is highest.

## J Newton-Schulz Iteration Convergence

The approximate polar decomposition used by DP-Muon is computed via five iterations of the quintic Newton-Schulz recurrence (Algorithm 1 in the main text). Table 14 reports the cosine similarity cos $( X _ { k } , U V ^ { \top } )$ between the k-th iterate and the exact polar factor, computed on gradient matrices from ResNet-18 (CIFAR-10, epoch 1).

For layers with moderate aspect ratios (conv1: $6 4 \times 1 4 7$ , fc: $1 0 \times 5 1 2 )$ , convergence exceeds 0.93 by $k = 5$ . The deepest layer (layer4.0.conv2: $5 1 2 \times 4 6 0 8 )$ converges more slowly (0.67 at $k = 5 )$ due to its extreme aspect ratio and noise-dominated spectrum, but this layer is precisely where the recovery condition predicts limited benefit. In all cases, additional iterations beyond k = 5 yield diminishing returns, justifying the choice of five iterations as a practical default.

Table 14: Newton-Schulz convergence on ResNet-18 gradient matrices (CIFAR-10, epoch 1). cos $( X _ { k } , U V ^ { \top } )$ : cosine similarity between the k-th iterate and the exact polar factor.
<table><tr><td> $k$ </td><td>conv1  $( 6 4 \times 1 4 7 )$ </td><td> $\mathrm { l a y e r 2 . 0 . c 1 }$   $( 1 2 8 \times 5 7 6 )$ </td><td> $\mathrm { l a y e r 4 . 0 . c 2 }$   $( 5 1 2 \times 4 6 0 8 )$ </td><td> $\operatorname { f c }$   $( 1 0 \times 5 1 2 )$ </td></tr><tr><td>0</td><td>0.550</td><td>0.631</td><td>0.123</td><td>0.460</td></tr><tr><td>1</td><td>0.653</td><td>0.731</td><td>0.295</td><td>0.791</td></tr><tr><td>2</td><td>0.806</td><td>0.866</td><td>0.367</td><td>0.910</td></tr><tr><td>3</td><td>0.939</td><td>0.964</td><td>0.484</td><td>0.933</td></tr><tr><td>4</td><td>0.983</td><td>0.982</td><td>0.585</td><td>0.938</td></tr><tr><td>5</td><td>0.985</td><td>0.986</td><td>0.667</td><td>0.934</td></tr></table>

## K Hyperparameters and Privacy Parameters

Table 15 lists all method hyperparameters. Table 16 reports the noise multiplier σ and total training steps for each (dataset, batch size) combination, computed via the PRV accountant (Opacus) at $\varepsilon = 4 , \delta = 1 0 ^ { - 5 }$ , 50 epochs. All experiments use automatic per-layer Frobenius norm clipping $( C =$ 1.0), cosine learning rate decay with 5% linear warmup, and RandomHorizontalFlip + RandomCrop (padding 4) augmentation.

Table 15: Method hyperparameters. NS = Newton-Schulz iterations.
<table><tr><td>Method</td><td>Optimizer</td><td>LR</td><td> $\beta$ </td><td>Nesterov</td><td>Post-processing</td><td>NS</td></tr><tr><td>DP-SGD</td><td>SGD</td><td>0.3</td><td>0.9</td><td></td><td>None</td><td></td></tr><tr><td>DP-Adam</td><td>Adam</td><td>0.001</td><td></td><td></td><td>None</td><td></td></tr><tr><td>DP-Muon</td><td>SGD</td><td>0.02</td><td>0.95</td><td>√</td><td> $U V ^ { \top }$ </td><td>5</td></tr><tr><td>DP-Muon-S</td><td>SGD</td><td>0.3</td><td>0.9</td><td></td><td> $\sigma _ { 1 } U V ^ { \top }$ </td><td>5</td></tr><tr><td>DiSK-SGD</td><td>SGD</td><td>0.01</td><td></td><td></td><td>Kalman (κ=1.5)</td><td></td></tr><tr><td>DOPPLER-SGD</td><td>SGD</td><td>3.0</td><td></td><td></td><td> $\mathrm { E M A } \ ( \beta = 0 . 9 )$ </td><td></td></tr><tr><td>DOPPLER-Muon</td><td>SGD</td><td>0.02</td><td></td><td></td><td> $\mathrm { E M A } + U V ^ { \top }$ </td><td>5</td></tr></table>

Table 16: Noise multiplier σ and total training steps per batch size at ε = 4, $\delta = 1 0 ^ { - 5 }$ , 50 epochs. Computed via PRV accountant.
<table><tr><td rowspan="2">B</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">SVHN</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td>σ</td><td>Steps</td><td>σ</td><td>Steps</td><td>σ</td><td>Steps</td><td>σ</td><td>Steps</td></tr><tr><td>256</td><td>0.85</td><td>9750</td><td>0.85</td><td>9750</td><td>0.77</td><td>14300</td><td>0.72</td><td>19500</td></tr><tr><td>512</td><td>1.04</td><td>4850</td><td>1.04</td><td>4850</td><td>0.92</td><td>7150</td><td>0.85</td><td>9750</td></tr><tr><td>1024</td><td>1.32</td><td>2400</td><td>1.32</td><td>2400</td><td>1.15</td><td>3550</td><td>1.04</td><td>4850</td></tr><tr><td>2048</td><td>1.74</td><td>1200</td><td>1.74</td><td>1200</td><td>1.48</td><td>1750</td><td>1.32</td><td>2400</td></tr><tr><td>4096</td><td>2.35</td><td>600</td><td>2.35</td><td>600</td><td>1.99</td><td>850</td><td>1.74</td><td>1200</td></tr><tr><td>8192</td><td>3.25</td><td>300</td><td>3.25</td><td>300</td><td>2.71</td><td>400</td><td>2.35</td><td>600</td></tr></table>

## L Extended Tiny-ImageNet Results

The main paper reports Tiny-ImageNet results (64 × 64, 200 classes, 100K images, WRN-16-4, $\varepsilon = 4 )$ for batch sizes $B \in \{ 1 0 2 4 , 2 0 4 8 , 4 0 9 6 \}$ . Figure 9 and table 17 extend this to $B = 8 1 9 2$ with three-seed averages, where the contrast between spectral and non-spectral methods is most pronounced. DP-SGD degrades from 21.1% at B = 1024 to 12.0% at B = 8192, approaching random performance (5% for 200 classes). DP-Adam follows the same collapse (16.1% → 10.6%). Spectral methods remain stable: DP-Muon declines only 0.3% (19.2% → 18.9%) and DP-Muon-S declines 1.1% $( 2 0 . 0 \%  1 8 . 9 \% )$ , yielding a gap of +6.9% over DP-SGD at B = 8192. The low inter-seed variance of spectral methods $( \leq 0 . 3 \% )$ contrasts with DP-SGD’s increasing instability at large batches (±0.7% at B = 8192).

![](images/cbbfb8a1a5455dd7ee8469f5df0b1262a0b1d6ebdb516dbb7663db17b32345e4.jpg)  
Figure 9: Batch size sweep on Tiny-ImageNet (ε = 4, WRN-16-4). Mean $\pm$ std over three seeds. DP-SGD and DP-Adam collapse at large B while spectral methods remain stable.

Table 17: Test accuracy (%) on Tiny-ImageNet at ε = 4 (WRN-16-4). Mean ± std over three seeds. Best per column in bold.
<table><tr><td>Method</td><td> $B { = } 1 0 2 4$ </td><td> $B { = } 2 0 4 8$ </td><td> $B { = } 4 0 9 6$ </td><td>B=8192</td></tr><tr><td>DP-SGD</td><td> ${ \bf 2 1 . 1 \pm 0 . 1 }$ </td><td> $2 0 . 7 \pm 0 . 3$ </td><td> $1 7 . 6 \pm 0 . 4$ </td><td> $1 2 . 0 \pm 0 . 7$ </td></tr><tr><td>DP-Adam</td><td> $1 6 . 1 \pm 0 . 5$ </td><td> $1 5 . 4 \pm 0 . 4$ </td><td> $1 3 . 3 \pm 0 . 3$ </td><td> $1 0 . 6 \pm 0 . 4$ </td></tr><tr><td>DP-Muon</td><td> $1 9 . 2 \pm 0 . 2$ </td><td> $2 0 . 0 \pm 0 . 2$ </td><td> $2 0 . 4 \pm 0 . 2$ </td><td> $1 8 . 9 \pm 0 . 1$ </td></tr><tr><td>DP-Muon-S</td><td> $2 0 . 0 \pm 0 . 1$ </td><td> ${ \bf 2 0 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 2 0 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 1 8 . 9 \pm 0 . 2 }$ </td></tr></table>

## M ImageNet-1k From-Scratch Classification

Beyond the gradient-spectrum analysis of section F, the spectral mechanism is evaluated on ful from-scratch training of NF-ResNet-50 on ImageNet-1k $( 2 2 4 \times 2 2 4$ , 1000 classes), the standard large-scale setting. All three optimizers use an identical protocol (batch size 32,768, 120 epochs, augmentation multiplicity K = 4 applied identically), so the comparison isolates the optimizer. Table 18 reports top-1 accuracy at $\varepsilon \in \{ 4 , 8 \}$ averaged over three seeds.

DP-Muon improves over DP-SGD by $+ 2 . 5 0 \%$ at $\varepsilon = 4$ and $+ 2 . 7 2 \%$ at $\varepsilon = 8$ . Across-seed variance is 7× and 3× lower respectively, indicating more reliable training and not only higher accuracy. DP-AdamW trails both, consistent with the weak batch-size scaling of adaptive DP optimizers. This is the standalone spectral mechanism at full scale, with no temporal-denoising components.

Table 18: Top-1 accuracy (%) for from-scratch NF-ResNet-50 on ImageNet-1k. $B = 3 2 , 7 6 8$ , 120 epochs, K = 4, three seeds. Best per column in bold.
<table><tr><td>Method</td><td> $\varepsilon = 4$ </td><td> $\varepsilon = 8$ </td></tr><tr><td>DP-SGD</td><td> $1 8 . 4 7 \pm 0 . 4 1$ </td><td> $2 4 . 6 0 \pm 1 . 1 8$ </td></tr><tr><td>DP-AdamW</td><td> $1 0 . 3 7 \pm 0 . 2 1$ </td><td> $1 8 . 6 4 \pm 0 . 1 0$ </td></tr><tr><td>DP-Muon</td><td> ${ \bf 2 0 . 9 7 \pm 0 . 1 3 }$ </td><td> ${ \bf 2 7 . 3 2 \pm 0 . 1 6 }$ </td></tr></table>

## N Compute and Memory Overhead

The wall-clock and memory cost of the Newton-Schulz orthogonalization is measured on WRN-16-4 (2.7M parameters) in fp32 on a single NVIDIA A100 (40 GB, PCIe). Table 19 reports milliseconds per logical optimization step (mean ± std over three runs) for DP-SGD and DP-Muon.

The five Newton-Schulz iterations add a roughly constant ∼6.5 ms per step, independent of batch size and augmentation multiplicity, because orthogonalization acts on each parameter matrix rather than on each sample. Per-sample backpropagation scales with B·K, so the relative overhead shrinks as the workload grows. It falls from +2.92% at B = 2048, K = 1 to +0.77% at $B = 4 0 9 6 , K = 4$ Peak GPU memory is within 1% across optimizers at each configuration. DP-Muon maintains a single first-order momentum bufer (1.0× model size), versus DP-Adam’s two (2.0×, first and second moments).

Table 19: Wall-clock overhead of Newton-Schulz orthogonalization. WRN-16-4, fp32, single A100. Milliseconds per logical step, mean ± std over three runs. $\Delta$ is the relative overhead of DP-Muon over DP-SGD.
<table><tr><td>Configuration</td><td>DP-SGD</td><td>DP-Muon</td><td> $\Delta$ </td></tr><tr><td>B = 2048, K = 1</td><td> $4 8 8 . 1 \pm 1 . 1 $ </td><td> $5 0 2 . 4 \pm 0 . 8$ </td><td> $+ 2 . 9 2 \%$ </td></tr><tr><td> $B = 4 0 9 6 .$   $K = 4$ </td><td> $3 8 9 4 . 6 \pm 1 . 0$ </td><td> $3 9 2 4 . 4 \pm 1 1 . 1$ </td><td>+0.77%</td></tr></table>