# LoopREG: Representation Supervision Enables Test-Time Recurrent Depth in Diffusion Transformers

Zipeng Wang<sup>1,2</sup>   
d250232003@stu.cqupt.edu.cn   
Yi Jiang 1,\*   
jiangyi@cqupt.edu.cn   
Xiang Liu<sup>2</sup>   
liux750@chinaunicom.cn   
Anping Xiong<sup>1</sup>   
xiongap@cqupt.edu.cn   
Yanbing Xiao<sup>1</sup>   
d240201045@stu.cqupt.edu.cn   
Huan Hu<sup>2</sup>   
huh30@chinaunicom.cn   
Xiaoya Zhu<sup>2</sup>   
zhuxy172@chinaunicom.cn   
Zezhou Chen<sup>2</sup>   
chenzz51@chinaunicom.cn   
Kai Wang<sup>2</sup>   
wangk115@chinaunicom.cn Zhaoxiang Liu<sup>2</sup> Shiguo Lian<sup>2,\*</sup>   
liuzx178@chinaunicom.cn liansg@chinaunicom.cn Huanlin Gao<sup>2</sup> Zhengrui Huang ghl@nwafu.edu.cn d250201016@stu.cqupt.edu.cn   
<sup>1</sup>Chongqing University of Posts and Telecommunications <sup>3</sup>Dongbei University of Finance and Economics Ye LV<sup>3,\*</sup> Fang Zhao<sup>2</sup>   
lvy30@chinaunicom.cn zhaof50@chinaunicom.cn Bo Shen<sup>2</sup> Yuren You<sup>2</sup>   
shenb15@chinaunicom.cn youyr@chinaunicom.cn   
<sup>2</sup>China United Network Communications Group Co., Ltd.   
<sup>4</sup>University of Electronic Science and Technology of China

\* Corresponding authors (in order): Yi Jiang, Ye LV, Shiguo Lian.

## ABSTRACT

Reusing Transformer blocks offers a way to increase diffusion-model computation at test time, but naive recurrence can degrade generation beyond its training depth. We introduce LoopREG, a representation-supervised recurrent diffusion Transformer that repeatedly applies a shared block at the state supervised by a pretrained visual encoder. Alignment is applied after the final recurrent visit during training; inference extends the same learned operator to additional visits. On ImageNet 256 × 256, LoopREG reduces SiT-B/2 FID from the local REG baseline’s 15.08 to 4.91 at 400K training steps, a 67.41% reduction. Within a checkpoint trained with two visits, one extra inference visit improves FID from 14.71 to 5.34 with unchanged weights. The benefit persists on SiT-L/2 and SiT-XL/2. Controlled interventions show that useful recurrence follows the location of representation supervision, while REPA alignment produces a smaller extrapolation benefit. These results make representation-supervised recurrence a practical route to improving diffusion generation through additional inference depth.

SiT-B/2 · ImageNet 256 × 256 · 400K training steps  
![](images/eaa459ba96ed3e30b84a3d9157c8403d079f82b5ff8c1588bf050a4508e4f414.jpg)  
Figure 1: LoopREG reduces REG’s FID by 67.41% on SiT-B/2. At 400K training steps, horizontal references show published REPA (24.40; Wu et al., 2025) and local REG (15.08). LoopREG $( K _ { \mathrm { t r a i n } } = 3 )$ reaches FID 4.91 at $K _ { \mathrm { i n f e r } } = 5$ (star). Figure 3 gives full sweeps. Local evaluations: EMA, 50K samples, 250 denoiser calls, CFG=1.

(a) Representation-supervised recurrence  
![](images/b67a7d87deb0623c51a4edffe128c5df2e098ff36839e51014dce9c6a57f945e.jpg)  
Figure 2: LoopREG: representation-supervised shared recurrence. SiT-B/2 is shown. (a) One clean image supplies both the VAE latent, noised after encoding, and frozen DINOv2 targets. Orange denotes CLS/semantic tokens; blue denotes image/patch tokens. Alignment reads the final recurrent state through a projector. (b) Shared weights are reused beyond the training horizon, illustrated by $K _ { \mathrm { t r a i n } } = 2$ and $\bar { K } _ { \mathrm { i n f e r } } = 3$ . Sampling starts from noise.

## 1 INTRODUCTION

Diffusion Transformers such as DiT and SiT devote a fixed stack of blocks to each denoiser evaluation (Peebles & Xie, 2023; Ma et al., 2024). Increasing the number of sampling steps adds computation across diffusion time. Shared recurrence offers a complementary possibility: spend more computation refining the hidden state within each evaluation, while reusing the same parameters. For this to improve generation, a learned block must remain useful when applied beyond its training horizon.

What makes recurrent computation reusable beyond its training horizon? Our experiments identify representation supervision as an enabling ingredient. Naively repeating a block in plain SiT yields at most a marginal matched-depth improvement and degrades sharply with an extra inference visit. The same recurrent construction becomes useful under REPA alignment and substantially more effective under REG’s representation entanglement (Yu et al., 2025; Wu et al., 2025). This contrast connects the value of extra computation to how the recurrent state is supervised.

We introduce LoopREG, a recurrent diffusion Transformer built on REG. LoopREG reuses one Transformer block at the representation-supervised state, applies alignment after the final trainingtime visit, and extends that block’s trajectory at inference (Figure 2). It retains REG’s joint image and semantic-token denoising objective. Inference adds visits with unchanged weights and a fixed diffusion sampling schedule.

The resulting generation gains are substantial. On ImageNet 256 × 256 at 400K training steps, LoopREG-B/2 reaches FID 4.91, compared with 15.08 for the local REG baseline: a 67.41% reduction (Figure 1). Within a LoopREG checkpoint trained with two visits, a third visit reduces FID from 14.71 to 5.34. Useful extra depth also persists on SiT-L/2 and SiT-XL/2. On XL/2, extra inference depth reaches the matched-depth 400K quality at 220K training steps.

The key design choice is where to recur. With supervision fixed at block four, extra recurrence helps at the supervised state and fails at the other tested sites. Moving supervision and recurrence together restores the benefit at other depths. A compact analysis relates this placement to direct supervision of the recurrent endpoint and interprets the finite useful inference horizon.

Our main contributions are:

1. LoopREG. We introduce representation-supervised shared recurrence for diffusion Transformers, enabling additional inference depth with the same trained parameters.

2. Substantial generation gains. LoopREG lowers SiT-B/2 FID from 15.08 to 4.91, with useful recurrent-depth extrapolation across B/2, L/2, and XL/2. Training-horizon sweeps characterize the resulting quality–depth curves.

3. A supervision-based design principle. Plain SiT/REPA/REG controls and site/depth interventions show that representation supervision and its location determine whether extra recurrent computation is useful in the tested settings.

## 2 RELATED WORK

Diffusion and representation supervision. DiT replaces the denoising backbone with a Transformer, and SiT studies flow and diffusion within a stochastic-interpolant framework (Peebles & Xie, 2023; Ma et al., 2024). REPA aligns projected intermediate states with clean pretrained visual representations (Yu et al., 2025). REG jointly denoises image latents and a semantic token while retaining representation alignment (Wu et al., 2025). SRA instead uses the model’s own EMA representations for alignment (Jiang et al., 2026). We study a further capability of representation supervision: enabling a recurrent operator to improve generation beyond its training horizon. The supervision-site interventions identify where this capability emerges.

Looped Transformers and latent computation. Universal Transformers combine shared depth with adaptive computation (Dehghani et al., 2019). Theoretical studies show that looped Transformers can implement programs (Giannou et al., 2023), learn iterative data-fitting procedures (Yang et al., 2024), and realize multistep gradient descent in controlled in-context-learning settings (Gatmiry et al., 2024). Recurrent-depth language models explore scaling latent computation beyond a fixed forward pass (Geiping et al., 2025). We investigate when repeated computation improves a diffusion denoiser, using representation supervision as the central intervention.

Elastic depth and visual generation. LoopFormer conditions recurrent updates on time and step size and trains shortcut-consistent trajectories (Jeddi et al., 2026); PonderNet learns a stopping distribution (Banino et al., 2021). ELT explores looped visual generation with intra-loop selfdistillation and reports gains beyond training depth (Goyal et al., 2026). We instead investigate external representation supervision and its location, connecting recurrent-depth extrapolation to supervised-state dynamics. Our RandomLoop control samples terminal horizons from finite uniform distributions; heavy-tailed random horizons are also used in recurrent-depth language-model training (Geiping et al., 2025).

Dynamics and inference-time reuse. Deep equilibrium models learn implicitly defined fixed points (Bai et al., 2019). Recurrent-dynamics work distinguishes regimes in which extra iterations help, preserve, or degrade predictions (Viakhirev et al., 2026); training-free looping uses damped updates to mitigate failures of naive reapplication (Chen et al., 2026). LoopREG trains the shared operator explicitly and reuses its learned update at inference. Internal Guidance combines intermediate and final denoising predictions to guide sampling (Zhou et al., 2026). LoopREG varies the recurrent depth of a representation-supervised block within each denoiser call. Its generation benefit depends on the supervision location and the inference horizon.

## 3 LOOPREG

LoopREG adds shared recurrence to a representation-supervised diffusion Transformer. Its defining choice is to repeat the block whose output receives representation supervision, so that the supervised state is also the state refined by extra inference computation.

## 3.1 REPRESENTATION-SUPERVISED SHARED RECURRENCE

Image and representation inputs. Let $x _ { 0 }$ be the VAE latent of a clean training image, y its class label, and $z = ( z _ { \mathrm { c l s } } , z _ { \mathrm { p a t c h } } )$ the frozen DINOv2 features of the same image. Following REG, we

form an image state and a semantic-token state at a common time $t \sim \mathcal { U } [ 0 , 1 ] $

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon , \qquad c _ { t } = ( 1 - t ) z _ { \mathrm { c l s } } + t \epsilon _ { c } ,\tag{1}
$$

where ϵ and $\epsilon _ { c }$ are independent Gaussian noise. The Transformer processes image tokens and the semantic token together, conditioned on t and $y ,$ and predicts both velocities. Figure 2 shows how the clean image supplies the latent and the alignment targets.

Reusing the supervised block. Write the blocks before the recurrent site as $H ,$ the shared block as $F _ { \theta }$ , and the remaining blocks and output projection as $G _ { \phi }$ . At a fixed sampling state $\boldsymbol { u } _ { t } = \left( \boldsymbol { x } _ { t } , \boldsymbol { c } _ { t } \right)$ ,

$$
h _ { 0 } = H ( u _ { t } ; t , y ) , \quad h _ { k + 1 } = F _ { \theta } ( h _ { k } ; t , y ) , \quad ( \hat { v } _ { K } ^ { x } , \hat { v } _ { K } ^ { c } ) = G _ { \phi } ( h _ { K } ; t , y ) .\tag{2}
$$

The depth K counts all applications of the shared block, including its ordinary first visit. Thus $K = 1$ executes it once and $K = 3$ executes the same module three times. Each visit receives the preceding visit’s state with the same time and class conditioning. LoopREG uses no recurrent-index embedding or per-visit parameters.

The supervision depth d counts Transformer blocks in the original backbone. We co-locate the recurrent block at $\ell = d$ and read the representation after its final recurrent visit. A nonlinear projector $P$ maps that endpoint to the clean teacher features. Alignment is evaluated once per forward pass, so the training objective supervises the endpoint of the learned trajectory. The main B/2 model uses $d = \ell = 4$ , and L/2 and $\mathrm { { X L } } \bar { / 2 }$ use $d = \ell = \bar { 8 }$ . Block labels here are one-based.

## 3.2 TRAINING AND INFERENCE

Training objective. Let M count teacher tokens, including the CLS token, and let bars denote unit normalization along the feature dimension. LoopREG retains REG’s negative-cosine alignment and joint velocity losses:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { r e p } } ( h _ { K } , z ) = - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \langle \overline { { P ( h _ { K } ) _ { i } } } , \bar { z } _ { i } \rangle , } \\ { \displaystyle \mathcal { L } = \mathbb { E } [ \mathrm { M S E } ( \hat { v } _ { K } ^ { x } , \epsilon - x _ { 0 } ) + 0 . 0 3 \mathrm { M S E } ( \hat { v } _ { K } ^ { c } , \epsilon _ { c } - z _ { \mathrm { c l s } } ) + 0 . 5 \mathcal { L } _ { \mathrm { r e p } } ( h _ { K } , z ) ] . } \end{array}\tag{3}
$$

(4)

MSE averages over the coordinates of each prediction; the expectation averages over training examples, noise, and time. Fixed-depth training uses $K = K _ { \mathrm { t r a i n } }$ throughout. Since every visit shares $F _ { \theta }$ , gradients update one operator over the states induced by its recurrent trajectory. RandomLoop instead samples a terminal horizon per optimization step, shared across distributed ranks, to train multiple usable endpoints.

Extending depth at inference. Generation initializes both x and c from noise. Each denoiser call evaluates Equation (2) at a chosen $K = K _ { \mathrm { i n f e r } }$ , and the sampler updates both states using the predicted velocities. The frozen visual teacher and clean representations are used only during training. Changing $K _ { \mathrm { i n f e r } }$ changes the number of shared-block applications within a call, while preserving all learned parameters and the diffusion time grid. For example, a model trained with $K _ { \mathrm { t r a i n } } = 2$ can be evaluated with $K _ { \mathrm { i n f e r } } = 3$ to add one recurrent refinement at every sampling step.

Computation. For a backbone with D blocks and a recurrent core of q blocks, the number of block executions per denoiser call is

$$
B ( K ) = D + ( K - 1 ) q .\tag{5}
$$

The main model uses $q = 1$ , so one extra visit adds one block execution. For B/2, the trained $K _ { \mathrm { t r a i n } } = 2$ call executes 13 blocks and $K _ { \mathrm { i n f e r } } = 3$ executes 14. Shared recurrence adds computation without adding copies of the block’s weights. We report block executions alongside generation metrics; measured runtime also depends on the sampler and implementation.

Controlled variants. Plain LoopSiT uses the same recurrence with image-velocity regression alone. LoopREPA adds patch-token alignment with coefficient 0.5 and has no REG semantic-token path or semantic loss. Independent-copy controls replace the shared operator with separately trained blocks. These variants distinguish the effects of representation supervision, REG’s joint representation objective, and parameter reuse.

## 3.3 INTERPRETING SUPERVISED RECURRENCE

Co-location directly supervises the recurrent endpoint. If the loop is before supervision, let $T _ { \ell \to d }$ map its endpoint to the supervised layer. The alignment gradient is

$$
\nabla _ { h _ { K } } \mathcal { L } _ { \mathrm { r e p } } = J _ { T } ( h _ { K } ) ^ { \top } \nabla _ { T ( h _ { K } ) } \ell _ { \mathrm { r e p } } ( P ( T ( h _ { K } ) ) , z ) .\tag{6}
$$

At co-location, $T = I { \mathrm { : } }$ the representation objective reads the recurrent endpoint directly. Earlier loops receive that gradient through intervening blocks, while a later block with distinct parameters lies outside the alignment branch. This structure provides a reason to co-locate recurrence and supervision. Section 4.3 tests the design by moving the two sites independently and jointly.

A shared operator learns from its own recurrent trajectory. Supervising the endpoint also trains the transitions that reach it. Let $g _ { k } = \nabla _ { h _ { k } } \mathcal { L }$ and let θ contain the shared block’s parameters. Backpropagation through $K = K _ { \mathrm { t r a i n } }$ visits gives

$$
g _ { k } = J _ { h } F _ { \theta } ( h _ { k } ) ^ { \top } g _ { k + 1 } , \qquad \nabla _ { \theta } \mathcal { L } = \sum _ { k = 0 } ^ { K - 1 } J _ { \theta } F _ { \theta } ( h _ { k } ) ^ { \top } g _ { k + 1 } .\tag{7}
$$

Time and class conditioning are held fixed within this recurrence. The same parameters receive updates at each visited state, including states produced by their own earlier applications. Co-location makes representation alignment part of the endpoint signal $g _ { K }$ . These two choices work together: alignment shapes the state at which recurrence ends, while sharing trains one transition over the path leading there. An unshared stack assigns these transitions to different parameter sets. This explains the design rationale for a reusable refinement operator, with its useful test-time horizon determined by the measured trajectory.

Useful refinement can end before equilibrium. Additional visits trace a longer hidden-state trajectory. Even a stable trajectory can pass through a task-preferred state before its fixed point. For the recurrence $h _ { k + 1 } = h _ { \infty } \dot { + } \lambda ( \dot { h _ { k } } - \dot { h _ { \infty } } ) , 0 < \lambda \dot { < } 1$ , and a preferred state $h _ { g } = h _ { \infty } + r ( \dot { h } _ { 0 } - h _ { \infty } )$ $0 < r < 1 ,$

$$
\| h _ { k } - h _ { g } \| ^ { 2 } = \| h _ { 0 } - h _ { \infty } \| ^ { 2 } ( \lambda ^ { k } - r ) ^ { 2 } .\tag{8}
$$

The error first decreases and then increases as the trajectory passes $h _ { g }$ . This finite-horizon interpretation is consistent with LoopREG’s observed FID minima beyond training and its later degradation. Appendix B gives the assumptions, formal statements, and derivations. We select inference depths from the measured generation curves.

## 4 EXPERIMENTS

Protocol. We evaluate class-conditional ImageNet at $2 5 6 \times 2 5 6$ , using checkpoints trained for 400K optimization steps with global batch 256. Main sweeps use EMA weights, 50K generated samples, 250 denoiser calls with an Euler–Maruyama SDE sampler, and CFG=1. We use the local ImageNet reference and OpenAI Inception evaluator. Appendix A gives the full sampling settings and provenance.

## 4.1 GENERATION GAINS FROM TEST-TIME DEPTH

LoopREG delivers large gains over REG and its trained endpoint. For LoopREG-B/2 trained at $K _ { \mathrm { t r a i n } } = 2$ , FID improves from 14.71 to 5.34 and remains below the matched value at $K _ { \mathrm { i n f e r } } = 4$ The relative reduction at $K _ { \mathrm { i n f e r } } = 3$ is 63.67%. Increasing the training horizon shifts the minimum: $K _ { \mathrm { t r a i n } } = 3$ reaches FID 4.91 at $K _ { \mathrm { i n f e r } } = 5$ , while $K _ { \mathrm { t r a i n } } \stackrel { = } { = } 4$ reaches 5.19 at $K _ { \mathrm { i n f e r } } = 6$ . Relative to the local REG baseline at 15.08, the best result reduces FID by 67.41%. Figure 3 shows how all three recurrent horizons cross below the baseline before reaching their finite optima. Within-checkpoint sweeps use unchanged weights. Across training horizons, equal optimization steps involve different training compute, as quantified by Equation (5).

<table><tr><td>Model</td><td>Steps</td><td> $K _ { \mathrm { t r a i n } }$ </td><td> $K _ { \mathrm { i n f e r } }$ </td><td>FID↓</td><td>sFID ↓</td><td>IS↑</td><td>P↑</td><td>R↑</td></tr><tr><td>SiT-B/2 (local)</td><td>400K</td><td>1</td><td>1</td><td>35.25</td><td>6.65</td><td>42.14</td><td>0.530</td><td>0.650</td></tr><tr><td>REPA-B/2*</td><td>400K</td><td>一</td><td>1</td><td>24.40</td><td>6.40</td><td>59.90</td><td>0.590</td><td>0.650</td></tr><tr><td>REG-B/2 (local)</td><td>400K</td><td>1</td><td>1</td><td>15.08</td><td>6.73</td><td>95.53</td><td>0.649</td><td>0.640</td></tr><tr><td>LoopSiT-B/2</td><td>400K</td><td>2</td><td>2</td><td>34.45</td><td>6.43</td><td>43.01</td><td>0.533</td><td>0.639</td></tr><tr><td>LoopSiT-B/2</td><td>400K</td><td>2</td><td>3</td><td>46.41</td><td>10.01</td><td>34.28</td><td>0.464</td><td>0.618</td></tr><tr><td>LoopREPA-B/2</td><td>400K</td><td>2</td><td>2</td><td>21.93</td><td>6.37</td><td>66.58</td><td>0.601</td><td>0.651</td></tr><tr><td>LoopREPA-B/2</td><td>400K</td><td>2</td><td>3</td><td>16.74</td><td>6.68</td><td>84.67</td><td>0.629</td><td>0.615</td></tr><tr><td>LoopREG-B/2</td><td>400K</td><td>2</td><td>2</td><td>14.71</td><td>6.76</td><td>97.13</td><td>0.650</td><td>0.631</td></tr><tr><td>LoopREG-B/2</td><td>400K</td><td>2</td><td>3</td><td>5.34</td><td>5.80</td><td>185.98</td><td>0.796</td><td>0.508</td></tr><tr><td>LoopREG-B/2</td><td>400K</td><td>3</td><td>3</td><td>13.75</td><td>6.69</td><td>100.82</td><td>0.658</td><td>0.634</td></tr><tr><td>LoopREG-B/2</td><td>400K</td><td>3</td><td>5</td><td>4.91</td><td>5.40</td><td>217.61</td><td>0.851</td><td>0.452</td></tr><tr><td rowspan="4">SiT-L/2  $\mathrm { { R E P A - L } } / 2 ^ { \ast }$   $\mathrm { { R E G - L } } / 2 ^ { \ast }$ </td><td>400K</td><td>一</td><td>一</td><td>18.80</td><td>5.29</td><td>72.00</td><td>0.640</td><td>0.640</td></tr><tr><td>400K</td><td>一</td><td></td><td>10.00</td><td>5.20</td><td>109.20</td><td>0.690</td><td>0.650</td></tr><tr><td>400K</td><td>一</td><td></td><td>4.60</td><td>5.21</td><td>167.60</td><td>0.750</td><td>0.630</td></tr><tr><td>LoopREG-L/2 400K</td><td>2</td><td>2</td><td>4.23</td><td>5.09</td><td>170.36</td><td>0.752</td><td>0.642</td></tr><tr><td rowspan="2">LoopREG-L/2 SiT  ${ \bf \cdot } { \bf \cal X } { \bf L } / 2 ^ { * }$ </td><td>400K</td><td>2</td><td>3</td><td>3.21</td><td>4.44</td><td>262.66</td><td>0.859</td><td>0.520</td></tr><tr><td>7M</td><td>一</td><td></td><td>8.30</td><td>6.32</td><td>131.70</td><td>0.680</td><td>0.670</td></tr><tr><td> $\mathrm { { R E P A } } { \cdot } \mathrm { { X L } } { \cdot } 2 ^ { \ast }$ </td><td>4M</td><td>一</td><td></td><td>5.90</td><td>5.73</td><td>157.80</td><td>0.700</td><td>0.690</td></tr><tr><td> $\mathbf { R E G - X L } / 2 ^ { * }$ </td><td>400K</td><td>一</td><td></td><td>3.40</td><td>4.87</td><td>184.10</td><td>0.760</td><td>0.640</td></tr><tr><td>LoopREG-XL/2</td><td>400K</td><td>2</td><td>2</td><td>3.38</td><td>4.85</td><td>184.71</td><td>0.758</td><td>0.635</td></tr><tr><td>LoopREG-XL/2</td><td>400K</td><td>2</td><td>3</td><td>2.84</td><td>4.64</td><td>254.70</td><td>0.839</td><td>0.549</td></tr></table>

Table 1: Generation quality across model scales. Horizontal rules separate backbones; spacing separates method families. Shaded rows are baselines; an asterisk denotes published results from Wu et al. (2025), Table 9 (SiT/REPA attributed there to Ma et al. (2024); Yu et al. (2025)). Training steps are shown explicitly; the published REPA-XL/2 reference uses 4M steps. All other rows are local evaluations. Bold marks the best displayed metric within each multi-row checkpoint group; best depths are selected within the reported sweeps. P/R: precision/recall. Appendix A gives provenance, including the L/2 server summary.  
![](images/de5eb96bddc0700f14e99ed29783fe42ea6de862d678c93d6c0683636deca353.jpg)  
Figure 3: Training horizon shifts the useful inference horizon. Dotted lines mark training horizons; stars and labels mark the best observed FID. The dashed horizontal line shows the local REG baseline. All three training horizons produce large improvements over this baseline, with optima moving to later recurrent depths. The logarithmic vertical axis preserves the severe under-roll failures.

Representation supervision unlocks recurrent extrapolation. Plain LoopSiT at matched depth slightly improves on the local SiT baseline (FID 35.25 to 34.45), but an additional visit worsens FID to 46.41. LoopREPA improves by 23.64% from $K _ { \mathrm { i n f e r } } = 2 \mathrm { t o } K _ { \mathrm { i n f e r } } = 3$ , despite having no REG semantic token, then degrades at $\dot { K } _ { \mathrm { i n f e r } } = 4$ . Alignment alone therefore enables useful extrapolation in this control, while REG substantially increases the gain and useful recurrent horizon.

![](images/556b0199408cf24823bbc16313f60ab562104914ad24beaaba418530a15f5978.jpg)  
Figure 4: Extra recurrent computation improves the optimization-step–FID curve. Left: fixed $K _ { \mathrm { i n f e r } } = 2 , 3 , 4$ curves from 50K steps along one XL training run, with published REG reference points. Right: 400K results; $K _ { \mathrm { i n f e r } } = 2 , 3$ share one checkpoint, while REG is a published reference. Lines connect evaluated checkpoints; 210K/220K include $K _ { \mathrm { i n f e r } } = 3$ only.

## 4.2 SCALING TO LARGER BACKBONES

The extrapolation benefit persists as the backbone grows. At 400K steps, LoopREG-L/2 improves from FID 4.23 at $K _ { \mathrm { i n f e r } } = 2 \mathrm { t o } 3 . 2 1$ at $K _ { \mathrm { i n f e r } } = 3$ . LoopREG-XL/2 uses 28 blocks, supervision at depth eight, and shared block eight. It improves from 3.3823 to 2.8376, a 16.10% reduction, with sFID also improving. Table 1 places these results beside the published SiT and REG baselines at each scale. Together with the B/2 results, these experiments establish useful recurrent extrapolation across three model scales.

One extra block application yields a large quality return. Table 2 isolates the added recurrent depth within each checkpoint. Moving from $K _ { \mathrm { i n f e r } } = 2$ to 3 adds one shared-block execution per denoiser call across all three backbones, with unchanged weights and the same 250-call sampler. The relative increase in block executions becomes smaller for deeper backbones, while each model retains a generation benefit. This gives LoopREG a direct way to spend additional computation inside the denoiser, complementary to increasing the number of sampling steps. Block counts measure executed Transformer depth; their runtime cost also depends on the backbone and implementation.
<table><tr><td>Backbone</td><td>Blocks/call</td><td>Added blocks (%)</td><td> $\mathrm { F I D } ( K _ { \mathrm { i n f e r } } \colon 2  3 )$ </td><td>FID reduction (%)</td></tr><tr><td>SiT-B/2</td><td>13→14</td><td>7.69</td><td>14.71→5.34</td><td>63.67</td></tr><tr><td>SiT-L/2</td><td>25→26</td><td>4.00</td><td>4.23→3.21</td><td>24.13</td></tr><tr><td>SiT-XL/2</td><td>29→30</td><td>3.45</td><td>3.38→2.84</td><td>16.10</td></tr></table>

Table 2: Return from one extra shared-block application. All models train with $K _ { \mathrm { t r a i n } } = 2$ for 400K steps. Added block executions are relative to matched $K _ { \mathrm { i n f e r } } = 2 ; \mathrm { F I D }$ reductions compare the same checkpoint at $K _ { \mathrm { i n f e r } } = 2$ and 3.

Reaching lower FID with fewer optimization steps. Along the same XL training trajectory, 220K at $K _ { \mathrm { i n f e r } } = 3$ reaches FID 3.3675, compared with 3.3823 at 400K and $K _ { \mathrm { i n f e r } } = \bar { 2 } \colon 4 5 \%$ fewer optimization steps at a slightly lower FID. Against published REG-XL thresholds (Wu et al., 2025), the earliest evaluated crossings use 40K/60K/100K/220K steps versus 50K/100K/200K/400K (Figure 4). These are optimization-step comparisons with additional inference recurrence; measured runtime savings require a separate assessment. Appendix F reports the full checkpoint grid and metric comparisons.

![](images/ff3d05c90773108eb4eba9bafb0a74e575eb02b91ccc5dd04af68a21f56a146d.jpg)

![](images/76d98967c81befa545590eb51c8091d0c2cad9b9d74d61d2f250e3ccbfe41bff.jpg)  
Figure 5: Useful recurrence follows representation supervision in the tested interventions. Left: fixed supervision at depth four, varied loop block. Right: supervision and recurrence moved together. Both panels reuse the d4/B4 anchor. Each model trains at $\bar { K } _ { \mathrm { t r a i n } } = 2$

Fidelity and coverage. The 400K XL $K _ { \mathrm { i n f e r } } = 2$ to 3 improvement accompanies higher precision and lower recall (0.6347 to 0.5490). For the 220K $K _ { \mathrm { i n f e r } } = 3$ versus 400K ${ \bar { K } } _ { \mathrm { i n f e r } } = { \bar { 2 } }$ comparison, sFID improves from 4.8470 to 4.5372, while recall falls to 0.5180. At 60K–100K, $K _ { \mathrm { i n f e r } } = 3$ improves FID but slightly worsens sFID. Precision and recall in Table 1 make this fidelity–coverage change explicit. Training with random horizons offers an alternative when retaining coverage is a priority (Section 4.4).

## 4.3 WHERE SHOULD LOOPREG RECUR?

Useful extrapolation depends on the supervised state. We first hold REG supervision at $d = 4$ and move the recurrent block. At matched $K _ { \mathrm { i n f e r } } = 2$ , all four tested sites have similar FID, but the unseen $K _ { \mathrm { i n f e r } } = 3$ response separates them sharply (Figure 5 and Table 4). B4 improves, while B2, B6, and B8 degrade. Thus matched-depth quality alone does not predict useful extrapolation.

We then move supervision with the loop. Co-located d2/B2, d6/B6, and d8/B8 each improve at $K _ { \mathrm { i n f e r } } = 3$ , with a weaker effect at depth eight. The benefit moves with supervision, supporting the co-located design in Section 3.3. The two groups share the same d4/B4 anchor.

Representation geometry tracks useful recurrent refinement. We probe the recurrent states of the two LoopREG-B/2 checkpoints on 1,024 forward-noised ImageNet images at five diffusion times. Patch CKA compares mean hidden patch features with mean DINOv2 patch features, using a fixed 256-image subset. Spatial correlation compares pairwise patch-token similarities, using 64 sampled positions across all 1,024 images. These complementary measurements describe semantic organization and spatial structure along the recurrent trajectory.

Figure 6 displays every tested depth together with generation FID. For $K _ { \mathrm { t r a i n } } = 2 ,$ patch CKA and spatial correlation both peak at $\bar { K _ { \mathrm { i n f e r } } } = 3$ , where FID is lowest. Training at $K _ { \mathrm { t r a i n } } = 3$ moves these geometry peaks to $K _ { \mathrm { i n f e r } } = 5$ , again matching the best generation depth. The extended trajectory therefore contains states with stronger teacher-referenced geometry than the state at which training ends. This correspondence connects the generation gains to refinement of the representation selected by LoopREG’s supervision.

The trained projection and extended trajectory play different roles. Native patch-projector cosine reaches its maximum at the trained depth in both checkpoints. The projector thus favors the supervised endpoint, while the best generation states occur farther along the trajectory. This distinction fits the finite-horizon interpretation in Section 3.3: a block can continue to improve taskrelevant geometry after reaching the endpoint used by its training objective. Appendix G provides all five models and individual diffusion times, including the plain-recurrence control. Paired output-depth interventions in Appendix C further identify a contribution from REG’s semantic-state updates.

Measured values at each recurrent depth  
(a) LoopREG-B/2, $K _ { \mathrm { t r a i n } } = 2$  
(b) LoopREG-B/2, $K _ { \mathrm { t r a i n } } = 3$
<table><tr><td rowspan=6 colspan=2>FID ↓9Patch CKA ↑Spatial corr. ↑Projector cos. ↑.</td><td rowspan=1 colspan=2>0.6</td><td rowspan=1 colspan=1>14.7</td><td rowspan=1 colspan=1>5.3</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>9.7</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>48.6</td><td rowspan=1 colspan=1>88.7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>118</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>13.8</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1>4.9</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>6.8</td><td rowspan=1 colspan=1>9.5</td></tr><tr><td rowspan=1 colspan=1>A↑</td><td rowspan=1 colspan=2>.633</td><td rowspan=1 colspan=1>.768.</td><td rowspan=1 colspan=1>792</td><td rowspan=1 colspan=1>.780</td><td rowspan=1 colspan=1>.754</td><td rowspan=1 colspan=1>.720</td><td rowspan=1 colspan=1>.683</td><td rowspan=1 colspan=1>.645</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.572</td><td rowspan=1 colspan=1>.685</td><td rowspan=1 colspan=1>.772</td><td rowspan=1 colspan=1>.797</td><td rowspan=1 colspan=1>.803.</td><td rowspan=1 colspan=1>798</td><td rowspan=1 colspan=1>.788</td><td rowspan=1 colspan=1>.774</td></tr><tr><td rowspan=1 colspan=2>.579</td><td rowspan=1 colspan=1>.631</td><td rowspan=1 colspan=1>.635</td><td rowspan=1 colspan=1>.627</td><td rowspan=1 colspan=1>.615</td><td rowspan=1 colspan=1>.599</td><td rowspan=1 colspan=1>.583</td><td rowspan=1 colspan=1>.566</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.522</td><td rowspan=1 colspan=1>.578</td><td rowspan=1 colspan=1>.619</td><td rowspan=1 colspan=1>.633</td><td rowspan=1 colspan=1>.637</td><td rowspan=1 colspan=1>.635</td><td rowspan=1 colspan=1>.631</td><td rowspan=1 colspan=1>.624</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>612</td><td rowspan=1 colspan=1>.678</td><td rowspan=1 colspan=1>.664</td><td rowspan=1 colspan=1>.636</td><td rowspan=1 colspan=1>.600</td><td rowspan=1 colspan=1>.560</td><td rowspan=1 colspan=1>.520</td><td rowspan=1 colspan=1>.481</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.582</td><td rowspan=1 colspan=1>.653</td><td rowspan=1 colspan=1>.681</td><td rowspan=1 colspan=1>.675</td><td rowspan=1 colspan=1>.659</td><td rowspan=1 colspan=1>.640</td><td rowspan=1 colspan=1>.618</td><td rowspan=1 colspan=1>.595</td></tr><tr><td rowspan=2 colspan=3>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Infe</td><td rowspan=1 colspan=2>rence dep</td><td rowspan=1 colspan=1>th Kin</td><td rowspan=1 colspan=1>fer</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Figure 6: Useful extra depth improves representation geometry. Cells show measured values; darker shading indicates a better rank within each row, and outlines mark that row’s optimum. Blue depth labels mark $K _ { \mathrm { t r a i n } } .$ Patch CKA, spatial correlation, and FID prefer the same extra depth for each LoopREG checkpoint. Native projector cosine instead peaks at the trained depth. Representation measurements use the final $_ { \mathrm { N = 1 } , 0 2 4 }$ probe (CKA subset 256), averaged over five diffusion times; FID uses the separate 50K generation evaluation.

## 4.4 CHOOSING A DEPTH POLICY

Fixed training horizons give LoopREG its strongest peak FID. They also concentrate generation around a narrower depth range: at its best B/2 depth, the $K _ { \mathrm { t r a i n } } = 3$ model has precision 0.851 and recall 0.452, versus the REG baseline’s 0.649/0.640. RandomLoop broadens the useful range and preserves more coverage. Training with $K \in \{ 1 , 2 , 3 \}$ yields a best observed FID of 11.68 with recall 0.640. This offers a choice between peak fidelity and flexibility in the number of inference visits.

A timestep-dependent policy adds another operating point. Applying $K _ { \mathrm { i n f e r } } = 3$ for $t \geq 0 . 6$ and $K _ { \mathrm { i n f e r } } = \mathrm { \bar { 2 } }$ for $t < 0 . 6$ reaches FID 7.25, between the two fixed-depth results, with recall 0.566. It reduces total block executions by 4.17% relative to fixed $K _ { \mathrm { i n f e r } } = 3 .$ Appendix D reports the complete random-depth curves, compute accounting, and unshared controls.

## 5 CONCLUSION

LoopREG makes additional test-time depth useful by repeatedly applying a shared Transformer block at a representation-supervised state. It reduces SiT-B/2 FID from 15.08 to 4.91 at 400K training steps and extends recurrent-depth gains to L/2 and XL/2. The supervision-site interventions identify a transferable design choice within these models: useful recurrence follows the supervised representation. These results establish a practical way to improve diffusion generation by extending learned recurrent computation.

Scope and future directions. The experiments cover class-conditional ImageNet, SiT backbones, and DINOv2 supervision. Useful depth has a finite optimum, and improved FID often accompanies lower recall. Extending the approach to other generation settings and selecting depths without a generation sweep are natural next steps.

## REPRODUCIBILITY STATEMENT

Appendix A specifies training, sampling, evaluation, and pretrained assets. Appendices H and G report final-checkpoint generation sweeps and representation trajectories. The explanatory derivations, paired output-depth interventions, and depth-selection diagnostics appear in Appendices B, C, and E. All reported numerical results are drawn from recorded experimental evaluations.

## AI ASSISTANCE DISCLOSURE

Generative AI tools assisted with manuscript drafting and revision, literature retrieval, code auditing, result interpretation, development of the dynamical-systems framework and proofs, experiment planning, architecture-figure generation and editing, and implementation of data-processing, visualization, and evaluation scripts. Numerical results in this manuscript are drawn from the project’s recorded experimental evidence.

## REFERENCES

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Advances in Neural Information Processing Systems, 2019. URL https://arxiv.org/abs/1909.01377.

Andrea Banino, Jan Balaguer, and Charles Blundell. Pondernet: Learning to ponder, 2021. URL https://arxiv.org/abs/2107.05407.

Lizhang Chen, Jonathan Li, Chen Liang, Ni Lao, and Qiang Liu. Training-free looped transformers, 2026. URL https://arxiv.org/abs/2605.23872.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019. URL https: //arxiv.org/abs/1807.03819.

Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar. Can looped transformers learn to implement multi-step gradient descent for in-context learning? In International Conference on Machine Learning, 2024. URL https://proceedings.mlr. press/v235/gatmiry24b.html.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2502.05171.

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In International Conference on Machine Learning, 2023. URL https://proceedings.mlr.press/v202/ giannou23a.html.

Sahil Goyal, Swayam Agrawal, Gautham Govind Anil, Prateek Jain, Sujoy Paul, and Aditya Kusupati. Elt: Elastic looped transformers for visual generation, 2026. URL https://arxiv.org/ abs/2604.09168.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. Loopformer: Elastic-depth looped transformers for latent reasoning via shortcut modulation. In International Conference on Learning Representa tions, 2026. URL https://arxiv.org/abs/2602.11451.

Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li, Lei Zhang, Haoyu Wang, Wei Wei, Guang Dai, Yanning Zhang, and Jingdong Wang. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2505.02831.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, 2019. URL https://arxiv.org/abs/1905.00414.

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, 2024. URL https://arxiv. org/abs/2401.08740.

Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. URL https://arxiv.org/abs/2304.07193.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, 2023. URL https://arxiv.org/abs/2212. 09748.

Ivan Viakhirev, Kirill Borodin, Amirah Almutairi, Serguei Barannikov, Maxim Abramov, and Grach Mkrtchian. Think shallow, solve deep: Controlling recurrent dynamics for reliable test-time depth, 2026. URL https://arxiv.org/abs/2608.18222.

Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, Ming-Ming Cheng, and Xiang Li. Representation entangle ment for generation: Training diffusion transformers is much easier than you think. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2507. 01467.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2311.12424.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025. URL https: //arxiv.org/abs/2410.06940.

Xingyu Zhou, Qifan Li, Xiaobin Hu, Hai Chen, and Shuhang Gu. Guiding a diffusion transformer with the internal dynamics of itself. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. URL https://arxiv.org/abs/2512.24176.

## A EXPERIMENTAL SETUP AND EVIDENCE PROVENANCE

We study class-conditional ImageNet generation at 256 × 256. Generation metrics and representation probes are retained in separate tables with their checkpoint, recurrent-depth, and evaluation settings. All numerical tables and inline results are generated from these records. The final 1,024-image DINO probe supplies the representation analysis; output-depth interventions and smaller-sample depth-selection diagnostics use their own evaluation protocols.

Training. SiT-B/2 has 12 blocks of width 768 with 12 heads; SiT-L/2 has 24 blocks of width 1024 with 16 heads. Image latents have four channels and spatial size $3 2 \times 3 2 .$ , with latent patch size two. The semantic teacher is DINOv2 ViT-B/14 (Oquab et al., 2023). Training uses global batch 256, AdamW with learning rate $1 0 ^ { - 4 }$ , betas (0.9, 0.999), epsilon $1 0 ^ { - 8 }$ , no weight decay, gradient clipping at one, class dropout 0.1, FP16 mixed precision, and TF32. The default launcher uses eight GPUs and checkpoints every 10K steps. RandomLoop’s horizon draw is a deterministic function of the experiment seed and optimization step and is synchronized across ranks; equal configured weights implement a uniform distribution over the specified choices. Horizon sampling has no learned halting mechanism. Nominally equal step counts do not match training FLOPs across horizons or supervision methods.

Sampling. The sampler constructs 250 time points from 1 to 0.04 and appends zero. It applies 249 stochastic Euler–Maruyama updates and a final deterministic drift update, totaling 250 network calls at CFG=1. REG jointly evolves image and semantic-token states. Network inputs/weights are FP32 with TF32 enabled; velocities and state updates are promoted to FP64. The VAE decoder is evaluated separately. Generated images are clamped, multiplied by 255, and converted to uint8 by truncation. The evaluator reads a local ImageNet reference and the local OpenAI TensorFlow Inception graph. FID, sFID, IS, precision, and recall are imported as reported, without recomputing them from unavailable image batches.

The configured global seed is zero and the rank seed is seed × world\_size + rank. Raw artifacts show different world sizes or per-GPU batches across runs; therefore these are not matched-noise paired comparisons. The source’s resume branch skips the sampler for completed batches and does not advance the random-number stream through their internal SDE noise draws. This prevents a blanket claim of bit-identical resumed and uninterrupted generation. The available artifacts do not establish which final cells, if any, were affected. The paired output-depth experiments below instead use fresh full runs and record their complete random-number traces.

Late evidence and precision. The final N=1,024 mechanism bundle and unshared-x3 K3/K6 metrics supersede earlier development results or missing-data notes. The representation analysis in Appendix G uses this final bundle. The separate velocity probe remains N=256 because no larger velocity probe is supplied. The L/2 main-result curve is supplied as the experiment owner’s final server-summary Markdown, at four decimal places for FID; it is parsed automatically, but its original metric/configuration JSON is not yet part of this evidence set. The verified launcher and snapshot checkpoint-status metadata are corroborating implementation/status evidence, not a replacement for those metric artifacts.

Baseline and selection scope. Local REG-B/2 is the comparison baseline; proximity to the published REG result is a sanity check, not proof of evaluator identity. No independently evaluated local non-loop REPA baseline or local non-loop L/2 baseline is inserted. Best inference depths are descriptive minima over observed sweeps. Appendix H presents full inference-depth sweeps at the final core-model checkpoints; the selected configurations include all evaluated inference depths, including unfavorable outcomes. No standard errors or confidence intervals are synthesized from a single seed.

Independent output-depth evaluations. Appendix C adds paired 50K evaluations at generation seeds 2 and 3 for both train-two and train-three B/2 checkpoints. GPU/batch configuration is fixed within each checkpoint study. Fresh full runs reproduce the matched and one-step-extrapolated endpoints and evaluate mixed output depths. The mixed policies require two network evaluations per state update. They are distinct records from the historical depth sweeps and do not resolve missing provenance for other historical checkpoints.

Pretrained assets. ImageNet provides the training images and class labels. Image latents use the pretrained VAE, and alignment targets use frozen DINOv2 ViT-B/14 features. The visual teacher is used during training and representation analysis; generation requires the trained diffusion model and VAE decoder.

## B DYNAMICAL INTERPRETATION AND DERIVATIONS

LoopREG has two interacting iteration scales: recurrent applications evolve the hidden state within a denoiser call, while diffusion steps evolve the image and semantic states across calls. We analyze where supervision enters the first trajectory, why its useful horizon can be finite, and how its outputs affect the second trajectory. The position interventions appear in Section 4.3; the output-path interventions appear in Appendix C.

## B.1 SUPERVISION LOCALITY

Proposition 1 (Gradient at the recurrent endpoint). Let d be the supervision depth and ℓ the loop site. For $\ell \leq d ,$ , write $T = T _ { \ell \to d } f o r$ the intervening transformation and $\mathcal { L } _ { \mathrm { r e p } } = \dot { \ell _ { \mathrm { r e p } } } ( P ( T ( h _ { K } ) ) , \dot { z } )$ . At differentiable points,

$$
\nabla _ { h _ { K } } \mathcal { L } _ { \mathrm { r e p } } = J _ { T } ( h _ { K } ) ^ { \top } J _ { P } ( T ( h _ { K } ) ) ^ { \top } \nabla _ { 1 } \ell _ { \mathrm { r e p } } ( P ( T ( h _ { K } ) ) , z ) .\tag{9}
$$

At $\ell = d , T = I .$ $H \ell > d$ and the loop parameters are distinct from upstream parameters, this alignment loss has zero derivative with respect to the loop parameters.

Co-location makes the projector read the recurrent endpoint directly. Earlier recurrence receives alignment gradients through the intervening layers; later recurrence is trained by downstream denoising losses but lies outside the alignment branch. This gradient structure motivates the hypothesis that co-location favors useful extrapolation. Moving supervision and recurrence separately and jointly tests that hypothesis, rather than selecting a block from matched-depth quality alone.

## B.2 FINITE USEFUL HORIZONS

Proposition 2 (Task-optimal transient state). Consider $h _ { k + 1 } = h _ { \infty } + \lambda ( h _ { k } - h _ { \infty } )$ , with $0 < \lambda < 1$ and $h _ { 0 } \neq h _ { \infty } .$ . For a task-preferred state $h _ { g } = h _ { \infty } + r ( h _ { 0 } - h _ { \infty } ) , 0 < r < 1$ , the increments shrink geometrically, while $Q ( k ) = \| h _ { k } - h _ { g } \| ^ { 2 }$ has continuous minimum at

$$
k _ { \mathrm { c o n t } } ^ { \star } = \frac { \log r } { \log \lambda } .\tag{10}
$$

An integer minimizer is one of the two bracketing integers, selected by comparing Q. Beyond the crossing, Q increases toward a positive limit.

Stability and task quality can therefore prefer different stopping times. The proposition characterizes a stable scalar recurrence; the learned Transformer’s finite FID optimum is an empirical observation. It motivates studying the whole recurrent trajectory rather than treating smaller state increments as a stopping criterion.

For a family satisfying $\lambda _ { K _ { \mathrm { t r a i n } } } ^ { K _ { \mathrm { t r a i n } } } = c$ and a shared task-preferred progress $r _ { g }$ , with $0 < c < 1$ and $0 < r _ { g } < 1$ , substitution gives

$$
k _ { \mathrm { c o n t } } ^ { \star } = K _ { \mathrm { t r a i n } } \frac { \log r _ { g } } { \log c } .\tag{11}
$$

The continuous optimum lies beyond training exactly when $r _ { g } < c ;$ Appendix B derives the stricter condition for an extra integer application to help. Under the shared-progress assumptions, a longer training horizon distributes latent progress over smaller updates and moves the optimum to a later depth. The observed $2  3 , 3  5 ,$ and $4  6$ horizons are consistent with this direction. The constants specify the assumed family; selecting real-model depths requires estimating their task dependence. The derivations and a schematic follow below; Appendix E evaluates candidate depth-selection models.

## B.3 SEMANTIC FEEDBACK ACROSS SAMPLING STEPS

For REG, the recurrent computation returns image and semantic velocities $( \hat { v } _ { K } ^ { x } , \hat { v } _ { K } ^ { c } )$ . Under the linear interpolant, CFG=1, and squared diffusion amplitude $g ( t ) ^ { 2 } = 2 t$ , the implemented reverse-time drifts at $t > 0$ are

$$
b _ { K } ^ { x } = x + ( 2 - t ) \hat { v } _ { K } ^ { x } ( x , c , t , y ) , \qquad b _ { K } ^ { c } = c + ( 2 - t ) \hat { v } _ { K } ^ { c } ( x , c , t , y ) .\tag{12}
$$

An Euler–Maruyama step updates each state with its corresponding drift and additive noise; the time increment $\Delta t _ { j }$ is negative.

Proposition 3 (Delayed image effect of a semantic update). Start two updates from the same $( x _ { j } , c _ { j } )$ with identical noise increments. Hold the image velocityfixed and change the semantic velocity by $\delta v _ { j } ^ { c }$ . Then

$$
\delta x _ { j + 1 } = 0 , \qquad \delta c _ { j + 1 } = \Delta t _ { j } ( 2 - t _ { j } ) \delta v _ { j } ^ { c } .\tag{13}
$$

Ifthe next image update uses the samefunction $b ^ { x }$ in both trajectories and the same noise,

$$
\delta x _ { j + 2 } = \Delta t _ { j + 1 } \big [ b ^ { x } ( x _ { j + 1 } , c _ { j + 1 } + \delta c _ { j + 1 } , t _ { j + 1 } , y ) - b ^ { x } ( x _ { j + 1 } , c _ { j + 1 } , t _ { j + 1 } , y ) \big ] .\tag{14}
$$

Changing only the final semantic output leaves the final image unchanged.

Thus an extra-depth semantic update can affect images through subsequent evaluations of the image vector field. This identifies a pathway to intervene on: evaluate two depths at the same current joint state, and choose the image and semantic outputs separately. Our directional hypothesis is that, when the image output already uses extra depth, the extra-depth semantic output further improves generation. The structure determines the intervention; the sign of its FID effect is tested across two training horizons in Appendix C.

## B.4 DERIVATIONS

Proof of Proposition 1. For $s ~ = ~ T ( h _ { K } )$ and $a \ = \ P ( s )$ , the total differential is $d { \mathcal { L } } _ { \mathrm { r e p } } =$ $\nabla _ { 1 } \ell _ { \mathrm { r e p } } ( a , z ) ^ { \top } J _ { P } ( s ) J _ { T } ( h _ { K } ) d h _ { K }$ Transposing gives Equation (9). At co-location $J _ { T } = I .$ . If alignment is evaluated before the recurrent block, its computational graph has no dependency on the distinct parameters of that later block, so that partial derivative is zero. This claim presumes those parameters are not also tied to upstream modules; the audited local-block implementation satisfies this condition. Other downstream losses may still provide nonzero gradients.

For a shared core, set $J _ { j } = \partial F _ { \theta } ( h _ { j } ) / \partial h _ { j }$ and $A _ { j } = \partial F _ { \theta } ( h _ { j } ) / \partial \theta$ . Holding the prefix state fixed with respect to the core parameters, define adjoints $g _ { K } = \nabla _ { h _ { K } } \mathcal { L }$ and $g _ { j } = J _ { j } ^ { \top } g _ { j + 1 }$ . Then

$$
\nabla _ { \theta } \mathcal { L } = \sum _ { j = 0 } ^ { K - 1 } A _ { j } ^ { \top } g _ { j + 1 } .\tag{15}
$$

This is the ordinary backpropagation-through-time identity. The shared operator aggregates gradients over states reached by its own trajectory. Contributions can reinforce or cancel; an unshared stack instead updates separate parameter sets.

Proof of Proposition 2. Induction gives $h _ { k } = h _ { \infty } + \lambda ^ { k } ( h _ { 0 } - h _ { \infty } )$ . Consequently,

$$
\| h _ { k + 1 } - h _ { k } \| = ( 1 - \lambda ) \lambda ^ { k } \| h _ { 0 } - h _ { \infty } \| ,\tag{16}
$$

$$
Q ( k ) = \| h _ { 0 } - h _ { \infty } \| ^ { 2 } ( \lambda ^ { k } - r ) ^ { 2 } .\tag{17}
$$

For continuous $k \geq 0 , \lambda ^ { k }$ decreases strictly from one to zero and crosses r once, at $\log ( r ) / \log ( \lambda )$ Before the crossing, the error strictly decreases; afterwards it strictly increases. A discrete minimizer must be one of the bracketing integers. For $m = \lfloor k _ { \mathrm { c o n t } } ^ { \star } \rfloor$ , choose m if $r \geq ( \lambda ^ { m } + \lambda ^ { m + 1 } ) / 2$ , and $m + 1$ otherwise, with a tie at equality. Thus rounding $k _ { \mathrm { c o n t } } ^ { \star }$ to the nearest integer is not generally correct because the trajectory is nonlinear in depth. The limiting error is $r ^ { 2 } \| h _ { 0 } - h _ { \infty } \| ^ { 2 } > 0$

The same derivation applies to a vector trajectory restricted to a single stable eigenmode when the preferred state lies on that trajectory. General multidimensional dynamics need not be unimodal, normal, or contractive. Neither shrinking observed increments nor a U-shaped FID curve establishes a spectral-radius bound for a Transformer.

Latent discretization. Taking logarithms of $\lambda _ { K _ { \mathrm { t r a i n } } } ^ { K _ { \mathrm { t r a i n } } } = c$ yields log $\lambda _ { K _ { \mathrm { t r a i n } } } = \log c / K _ { \mathrm { t r a i n } }$ . Substitution into the continuous optimum gives Equation (11). For $r _ { g } < c ,$ both logarithms are negative and log $r _ { g } / \log c > 1$ , giving a continuous optimum beyond the training horizon. The through-origin descriptive fit to the observed horizon pairs has coefficient 1.5517. This summarizes the observed horizon shift under the latent-progress model; estimating its variability requires additional training configurations.

When does one extra application improve the task error? Write $C = \| h _ { 0 } - h _ { \infty } \| ^ { 2 } > 0$ and $\lambda = \lambda _ { K _ { \mathrm { t r a i n } } } , \mathsf { s o } \lambda ^ { K _ { \mathrm { t r a i n } } } = \bar { c }$ . Subtracting the two task errors gives

$$
Q ( K _ { \mathrm { t r a i n } } + 1 ) - Q ( K _ { \mathrm { t r a i n } } ) = C c ( \lambda - 1 ) \big [ c ( 1 + \lambda ) - 2 r _ { g } \big ] .\tag{18}
$$

Since $c > 0$ and $\lambda < 1$ , one extra integer application improves $Q$ if and only if

$$
r _ { g } < \frac { c ( 1 + \lambda ) } { 2 } .\tag{19}
$$

Equality gives a tie. This is stricter than $r _ { g } < c ,$ which only places the continuous optimum beyond training. For example, $\lambda = 1 / 2 , K _ { \mathrm { t r a i n } } = 2$ and $r _ { g } = 0 . 2 4$ place the continuous optimum after two, yet the discrete optimum remains two. The criterion identifies the missing task-dependent quantity: the desired progress $r _ { g }$ must be estimated independently of the observed best depth before this model can select an inference horizon.

Proof of Proposition 3. For a state u $\in \{ x , c \}$ , the linear interpolant gives

$$
\mathrm { s c o r e } _ { K } ^ { u } = { \frac { - ( 1 - t ) { \hat { v } } _ { K } ^ { u } - u } { t } } , \qquad b _ { K } ^ { u } = { \hat { v } } _ { K } ^ { u } - { \textstyle { \frac { 1 } { 2 } } } ( 2 t ) \mathrm { s c o r e } _ { K } ^ { u } = u + ( 2 - t ) { \hat { v } } _ { K } ^ { u } .\tag{20}
$$

The sampler evaluates this expression at positive times, including $t = 0 .$ 04 for its final update to zero. Its stochastic steps are

$$
x _ { j + 1 } = x _ { j } + \Delta t _ { j } b ^ { x } ( x _ { j } , c _ { j } , t _ { j } , y ) + \sqrt { 2 t _ { j } | \Delta t _ { j } | } \xi _ { j } ^ { x } ,\tag{21}
$$

$$
c _ { j + 1 } = c _ { j } + \Delta t _ { j } b ^ { c } ( x _ { j } , c _ { j } , t _ { j } , y ) + \sqrt { 2 t _ { j } | \Delta t _ { j } | } \xi _ { j } ^ { c } .\tag{22}
$$

Take two steps from the same current state with the same two noise tensors. Their image drifts are identical, giving $\delta x _ { i + 1 } = 0$ . Subtracting the semantic updates cancels the noise and the current state, giving Equation (13). At the next step, subtracting the image updates again cancels the common image state and noise; the difference of the image drift evaluations gives Equation (14). The final sampler call returns only the image update, so its semantic output cannot affect the returned image.

For a differentiable next-step image drift, a semantic intervention of strength α yields the local sensitivity

$$
\left. \frac { d x _ { j + 2 } } { d \alpha } \right| _ { \alpha = 0 } = \Delta t _ { j + 1 } \frac { \partial b ^ { x } } { \partial c } \Delta t _ { j } ( 2 - t _ { j } ) \delta v _ { j } ^ { c } .\tag{23}
$$

The Jacobian is evaluated along the unperturbed trajectory. This expression identifies the feedback channel; the FID direction is an empirical property tested by the complete sampling intervention.

Teacher-forced error and generated trajectories. The velocity probe measures risk on forwardnoised data, whereas the sampler visits model-induced joint image/semantic states. The terminal image depends on the directions of successive drift changes and on subsequent sensitivity to both states. Consequently, the observed reduction in FID can accompany an increase in teacher-forced NMSE. The projector is a nonlinear MLP; its local Jacobian and the downstream network can be sensitive to different hidden-state directions.

## C PAIRED OUTPUT-DEPTH INTERVENTIONS

Experimental question and replication. We use shared Block4 SiT-B/2 REG checkpoints trained for 400K steps at $K _ { \mathrm { t r a i n } } = m \in \{ 2 , 3 \}$ . Mixed policies evaluate depths m and $m + 1$ at the same current $( x , c , t , y )$ of that condition’s trajectory; equal-depth policies evaluate the network once. We select image velocity from depth $K _ { x }$ and semantic velocity from depth $K _ { c } .$ Conditions A/B/C/D use $( m , m \bar { ) } , ( m + \bar { 1 , } m ) , ( m , \bar { m } + 1 ) , ( m + 1 , m + 1 )$ throughout sampling. Each selected output updates its corresponding state, so later inputs depend on both policies.

![](images/08d97c8421bb2bf6573a4f0758cb946231daf7f749c3978f5db27d2e64632ecc.jpg)

![](images/c88b842e7bb3d1350b2664ed8c5499aad5ebcb4794162677c6aedfd870bc3d2d.jpg)  
Figure 7: Transient optima and horizon-dependent latent progress. A contractive trajectory can pass through a task-preferred state before reaching its fixed point. Under a fixed total-progress assumption, a larger training horizon corresponds to smaller recurrent steps and a later optimal inference horizon.

For train-two, the four conditions and primary directional comparison $\mathrm { F I D } ( B ) > \mathrm { F I D } ( D )$ were specified before a seed-1 screening run with 5K images per condition. Seeds 2 and 3 then evaluated all four conditions at 50K each with the checkpoint, depths, and comparison fixed. Train-three was designated as a cross-horizon replication with depths three and four before its evaluation. Its two 50K seeds use the same factorial design. All 16 final evaluations are reported; screening samples are excluded from the table and means. A/D reproduce each checkpoint’s native matched and onestep-extrapolated policies. This intervention tests a fixed extra step, rather than searching for the best depth of either checkpoint.

Sampling and common random numbers. All conditions use EMA, CFG=1, the original 250- update Euler–Maruyama time grid, FP32 network evaluation with TF32, FP64 state arithmetic, and the same VAE, quantization, ImageNet reference, and Inception evaluator. Train-two uses six H100 GPUs and train-three uses four, each with batch 64. Rank seeds are $s W + r ,$ where s is the generation seed, W the world size, and r the rank. Within each checkpoint and seed, initial image and semantic noise, class labels, and every SDE noise increment are shared across all four conditions. Each condition is a fresh complete sampling run. Different world sizes make cross-horizon comparisons unpaired, with partially overlapping rank seeds. The two checkpoints represent different training horizons, not repeated training seeds.

Outputs, trajectories, and compute. For a mixed policy, the two network calls use identical current inputs before either state is advanced. Thus B→D keeps the image-update function at depth $m + 1$ but changes future semantic inputs to that function. A→C similarly isolates the semantic-output policy at image depth $m .$ Changing image depth under a fixed semantic policy gives A→B and C→D. The mixed policies require 500 network calls because both depths are evaluated separately; the native joint endpoints require 250. These interventions identify output-path effects rather than provide a compute-matched alternative sampler.

Implementation checks. Routed A/D match the native sampler’s image latents at their respective depths with identical inputs and random states. Mixed outputs match direct model outputs evaluated at those same inputs. Replacing only the final semantic output leaves the final image unchanged, a Proposition 3 requires. Model forwards consume no random numbers, and the recorded native SDE draws preserve pairing.

FID, interaction, and coverage. Both checkpoints and both seeds give $\mathbf { D { < } B { < } C { < } A }$ in FID. For train-two, B→D improves FID by 29.1% and 28.4%; train-three gives 29.6% and 29.0%. Joint refinement improves sFID, IS, and precision while lowering recall in all four groups. The FID-scale interaction $J _ { 1 1 } - J _ { 1 0 } - J _ { 0 1 } + J _ { 0 0 }$ is positive in each group: the joint FID reduction is smaller than the sum of isolated reductions. This nonlinear metric contrast does not assign additive percentages of the gain to separate mechanisms. Means average two separate 50K metrics within each checkpoint; they are not pooled 100K FIDs or estimates of training-seed uncertainty.

<table><tr><td>Condition</td><td>Kx</td><td>Kc</td><td>FID↓</td><td>sFID ↓</td><td>IS↑ Prec. ↑</td><td>Rec. ↑</td><td>Calls</td></tr><tr><td> $K _ { \mathrm { t r a i n } } = 2 ,$  generation seed 2; 50,000 images per condition</td><td colspan="5"></td><td></td><td></td></tr><tr><td>A</td><td>2</td><td>2</td><td>14.709</td><td>6.702</td><td>97.933</td><td>0.64730.6326</td><td></td><td>250</td></tr><tr><td>B</td><td>3</td><td>2</td><td>7.533</td><td>6.236</td><td>142.720</td><td>0.7455</td><td>50.5599</td><td>500</td></tr><tr><td>C</td><td>2</td><td>3</td><td>9.362</td><td>6.097</td><td>132.561</td><td></td><td>0.7021 0.5974</td><td>500</td></tr><tr><td>D</td><td>3</td><td>3</td><td>5.339</td><td>5.783</td><td>185.375</td><td>0.7934</td><td>0.5143</td><td>250</td></tr><tr><td> $K _ { \mathrm { t r a i n } } = 2 ,$ </td><td colspan="6">generation seed 3; 50,000 images per condition</td><td></td><td></td></tr><tr><td>A</td><td>2</td><td>2</td><td>14.567</td><td>6.632</td><td>98.236</td><td></td><td>0.6482 0.6381</td><td>250</td></tr><tr><td>B</td><td>3</td><td>2</td><td>7.398</td><td>6.207</td><td>144.263</td><td>0.7472</td><td>20.5660</td><td>500</td></tr><tr><td>C</td><td>2</td><td>3</td><td>9.175</td><td>6.067</td><td>133.129</td><td></td><td>0.7047 0.5971</td><td>500</td></tr><tr><td>D</td><td>3</td><td>3</td><td>5.296</td><td>5.758</td><td>187.057</td><td>0.7970</td><td>0.5130</td><td>250</td></tr><tr><td colspan="7"> $K _ { \mathrm { t r a i n } } = 3 ,$  generation seed 2; 50,000 images per condition</td><td></td><td></td></tr><tr><td>A</td><td>3</td><td>3</td><td>13.924</td><td>6.552</td><td></td><td>100.5340.65550.6365</td><td></td><td>250</td></tr><tr><td>B</td><td>4</td><td>3</td><td>7.438</td><td>5.480</td><td>139.488</td><td>30.7519</td><td>90.5747</td><td>500</td></tr><tr><td>C</td><td>3</td><td>4</td><td>9.279</td><td>6.155</td><td>132.510</td><td>0.7111</td><td>0.6005</td><td>500</td></tr><tr><td>D</td><td>4</td><td>4</td><td>5.235</td><td>5.295</td><td>176.790</td><td>0.8011</td><td>0.5296</td><td>250</td></tr><tr><td colspan="7"> $K _ { \mathrm { t r a i n } } = 3 ,$  generation seed 3; 50,000 images per condition</td><td></td><td></td></tr><tr><td>A</td><td>3</td><td>3</td><td>13.848</td><td>6.612</td><td>100.628</td><td>0.65830.6339</td><td></td><td>250</td></tr><tr><td>B</td><td>4</td><td>3</td><td>7.519</td><td>5.540</td><td>139.223</td><td>0.7535</td><td>0.5700</td><td>500</td></tr><tr><td>C</td><td>3</td><td>4</td><td>9.303</td><td>6.193</td><td>130.924</td><td>0.7167</td><td>0.5902</td><td>500</td></tr><tr><td>D</td><td>4</td><td>4</td><td>5.339</td><td>5.351</td><td>175.520</td><td>0.8038</td><td>0.5294</td><td>250</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: All paired 50K output-depth evaluations. Bold marks the best metric within each checkpoint and generation seed. Calls counts denoiser evaluations per image: A/D use 250 and B/C use 500, all with 250 state updates.
<table><tr><td>Sup. depth Loop block FID</td><td></td><td> $K _ { \mathrm { i n f e r } } = 2$  FID</td><td> $K _ { \mathrm { i n f e r } } = 3$ </td><td> $\mathrm { F I D } K _ { \mathrm { i n f e r } } = 4$ </td><td>Gain (%) R at</td><td> $K _ { \mathrm { i n f e r } } = 3$ </td></tr><tr><td colspan="7">Fixed supervision at depth four</td></tr><tr><td>4</td><td>2</td><td>13.92</td><td>32.56</td><td>58.85</td><td>-133.93</td><td>0.676</td></tr><tr><td>4</td><td>4</td><td>14.71</td><td>5.34</td><td>6.20</td><td>63.67</td><td>0.508</td></tr><tr><td>4</td><td>6</td><td>14.53</td><td>31.11</td><td>69.58</td><td>-114.13</td><td>0.611</td></tr><tr><td>4</td><td>8</td><td>14.18</td><td>29.97</td><td>59.55</td><td>-111.33</td><td>0.618</td></tr><tr><td colspan="7">Supervision and recurrence moved together</td></tr><tr><td>2</td><td>2</td><td>15.61</td><td>7.49</td><td>10.25</td><td>51.99</td><td>0.544</td></tr><tr><td>4</td><td>4</td><td>14.71</td><td>5.34</td><td>6.20</td><td>63.67</td><td>0.508</td></tr><tr><td>6</td><td>6</td><td>14.81</td><td>5.98</td><td>7.78</td><td>59.61</td><td>0.528</td></tr><tr><td>8</td><td>8</td><td>16.54</td><td>12.88</td><td>28.32</td><td>22.16</td><td>0.541</td></tr></table>

Table 4: Site/depth interventions. Bold marks the lowest FID within each row. Gain is the relative FID reduction from $K _ { \mathrm { i n f e r } } = 2$ to $K _ { \mathrm { i n f e r } } = 3 $ negative values indicate degradation. The last column reports recall at the extrapolated depth. The d4/B4 anchor is repeated across groups. At d8/B8, FID improves while sFID worsens.

## D SITE INTERVENTIONS, DEPTH POLICIES, AND CONTROLS

## D.1 SITE AND SUPERVISION DEPTH

Table 4 supplements Figure 5 with both inference depths and recall for every tested site/depth pair.

## D.2 TRAINING AND INFERENCE DEPTH POLICIES

Training over different horizons controls the range of useful inference depths. LoopREG-x3 at $K _ { \mathrm { i n f e r } } = 5$ has precision 0.851 and recall 0.452, compared with the REG baseline’s 0.649 and 0.640. Lower FID therefore coexists with reduced measured coverage. RandomLoop broadens the region of useful depths and retains recall closer to the baseline, at a worse peak FID. For example, Random {1, 2, 3} attains its best observed FID of 11.68 at $K _ { \mathrm { i n f e r } } = 5$ with recall 0.640. Fixed depth favors peak FID; random depth favors a broader useful range. Some random-policy curves continue improving at the evaluated boundary.

A timestep-dependent depth schedule gives another operating point. Using $K _ { \mathrm { i n f e r } } = 2$ for $t < 0 . 6$ and $K _ { \mathrm { i n f e r } } = 3$ for $t \geq 0 . 6$ yields FID 7.25 and recall 0.566, between fixed $K _ { \mathrm { i n f e r } } = 2$ and fixed $K _ { \mathrm { i n f e r } } = 3$ . Its mean recurrent depth is 2.416; relative to fixed $K _ { \mathrm { i n f e r } } = 3$ , recurrent-core executions decrease by 19.47%, and total block executions by only 4.17%. Neither quantity is a measured wall-clock or full-model FLOP saving.

![](images/ea83f89a06cff1e03c896fd7892110411547591c575d95ba4b65c608ada08362.jpg)

![](images/9c1c607a8586c60d844a6e2541e36ccd14cd0ff3d4b6eeffbda6cd9a4c4d770c.jpg)  
Figure 8: Fixed and random training horizons. The full observed depth range exposes under-roll failures, finite optima, and recall differences. The dotted recall line is the local REG baseline. Random {3, 4, 5} and {1, 2, 3, 4, 5} are still improving in FID at $K _ { \mathrm { i n f e r } } = 8 ,$ the evaluated boundary.

## D.3 SHARED AND PERIODIC UNSHARED OPERATORS

For an unshared pair, $H = F _ { b } \circ F _ { a }$ and $K _ { \mathrm { i n f e r } } = 2 m$ evaluates $H ^ { m }$ ; for a triplet, $H = F _ { c } \circ F _ { b } \circ F _ { a }$ and $K _ { \mathrm { i n f e r } } = 3 m$ . Inference cycles through the trained copies without adding weights. The matched pair attains FID 14.15, slightly better than shared train-two’s 14.71. It improves to 12.27 on its second cycle before deteriorating sharply. The triplet worsens from 13.32 to 40.91 on its first extra cycle. Thus sharing does not explain a matched-depth advantage; its main benefit here is greater useful extrapolation.

<table><tr><td> $K _ { \mathrm { t r a i n } }$  / Policy</td><td> $K _ { \mathrm { i n f e r } }$ </td><td>Blocks</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>P↑</td><td>R↑</td></tr><tr><td>2 / fixed</td><td>1</td><td>12</td><td>151.9461</td><td>50.3472</td><td>6.8845</td><td>0.1480</td><td>0.1539</td></tr><tr><td>2 / fixed</td><td>2</td><td>13</td><td>14.1520</td><td>6.6645</td><td>99.3560</td><td>0.6541</td><td>0.6383</td></tr><tr><td>2 / periodic</td><td>2</td><td>13</td><td>14.1520</td><td>6.6645</td><td>99.3560</td><td>0.6541</td><td>0.6383</td></tr><tr><td>2 / periodic</td><td>4</td><td>15</td><td>12.2748</td><td>8.2114</td><td>137.2766</td><td>0.6315</td><td>0.5338</td></tr><tr><td>2 / periodic</td><td>6</td><td>17</td><td>61.1025</td><td>19.6300</td><td>39.9223</td><td>0.2887</td><td>0.4964</td></tr><tr><td>2 / periodic</td><td>8</td><td>19</td><td>147.5555</td><td>52.2028</td><td>10.3376</td><td>0.0813</td><td>0.2113</td></tr><tr><td>3 / periodic</td><td>3</td><td>14</td><td>13.3211</td><td>6.5114</td><td>103.6684</td><td>0.6622</td><td>0.6294</td></tr><tr><td>3 / periodic</td><td>6</td><td>17</td><td>40.9129</td><td>23.2540</td><td>53.8996</td><td>0.4237</td><td>0.5638</td></tr></table>

Table 5: Unshared controls at 400K. Every evaluated depth is included, grouped by training horizon and inference policy. The fixed and periodic pair share their matched-depth anchor. Bold marks metric optima within each group.

## D.4 TEACHER-FORCED VELOCITY ERROR

<table><tr><td>Model</td><td> $K _ { \mathrm { t r a i n } }$ </td><td>NMSE  $K _ { \mathrm { i n f e r } }$ </td><td>FID  $K _ { \mathrm { i n f e r } }$ </td><td>NMSE train</td><td>NMSE FID-best</td></tr><tr><td>LoopSiT-x2</td><td>2</td><td>2</td><td>2</td><td>0.46074</td><td></td></tr><tr><td>LoopREG-x2</td><td>2</td><td>2</td><td>3</td><td>0.45706</td><td>0.45898</td></tr><tr><td>LoopREG-x3</td><td>3</td><td>3</td><td>5</td><td>0.45684</td><td>0.46031</td></tr></table>

Table 6: The separate N=256 teacher-forced velocity probe, averaged over five noise times. The FID-best horizon differs from the NMSE-best horizon for both LoopREG checkpoints. $^ { 6 6 } - ^ { 5 9 }$ means that the two horizons coincide.

For LoopREG, velocity NMSE is lowest at the training horizon, whereas generation FID is lowest at a larger depth. The extra-depth benefit therefore cannot be reduced to improvement in this measured one-step teacher-forced error.

## E DEPTH-SELECTION DIAGNOSTICS

The main method selects inference depth from measured sweeps. We additionally tested two candidate forecasting rules on the existing train-two, train-three, and train-four checkpoints. Their historical endpoint behavior was known during development; predictions were frozen before new evaluation samples were generated. These diagnostics use 5K or 4,096 images per metric, separately from the main 50K results.

Early training curves. Exponential fits $E _ { q } ( s ) = E _ { q , \infty } + A _ { q } e ^ { - \gamma _ { q } s }$ model FID’s mean and covariance contributions from observations at 50K, 100K, and 150K steps. The fitted rate must be positive and the limit nonnegative; invalid fits abstain. The grid contains three training horizons and six inference depths, $K _ { \mathrm { i n f e r } } \mathsf { \bar { \in } } \left\{ 2 , \ldots , 7 \right\}$ . Calibration uses seed 1701; 200K and 400K evaluations use seed 2901 with paired streams and 5K samples per cell. All three rules select the lowest-FID
<table><tr><td>Forecast</td><td>MAE, 200K</td><td>MAE, 400K</td><td>Depth hits, 400K</td><td>Selected  $( K _ { \mathrm { t r a i n } } , K _ { \mathrm { i n f e r } } )$ </td></tr><tr><td>Component exponential</td><td>0.638</td><td>3.262</td><td>1/3</td><td>(3,5)</td></tr><tr><td>Total-FID exponential</td><td>0.652</td><td>3.299</td><td>1/3</td><td>(3,5)</td></tr><tr><td>Latest observation (150K)</td><td>2.237</td><td>5.258</td><td>1/3</td><td>(3,5)</td></tr></table>

Table 7: Early-curve forecasts on 5K-sample FID. MAE averages 18 candidates. Depth hits count correct inference-depth selections across three trained models.

pair $( K _ { \mathrm { t r a i n } } , K _ { \mathrm { i n f e r } } ) = ( 3 , 5 )$ , but their per-model selections 4, 5, 7 match only one of the observed optima 3, 5, 6. Exponential fitting improves numerical forecasts without improving depth selection over the 150K ranking. At 400K, 16 of 18 measured FIDs fall below their fitted limits. This rule requires early trained checkpoints and does not select training depth before training.

Local sampling responses. The second rule interpolates both image and semantic velocities as $\hat { v } _ { \alpha } = ( 1 - \alpha ) \hat { v } _ { k } + \alpha \hat { v } _ { k + 1 }$ Complete sampling at $\alpha \in \{ 0 , 1 / 1 6 , \hat { 1 } / 8 , 1 / 4 \}$ produces terminal Inception features after decoding and quantization. Linear and quadratic fits extrapolate these features to α = 1, using 4,096 paired samples at calibration seed 5101. Predictions are fixed before independent-seed endpoint evaluation at seed 5201. The eight edges are $2  3 , 3  .$ 4 for train-two, $3  4 , 4  5 , 5  6$ for train-three, and $4  5 , 5  6 , 6  7$ for train-four. Endpoint evaluations again favor depths 3, 5, 6. All three nonzero extrapolators predict deterioration on every edge, missing the five beneficial edges and selecting the trained depths. Their mean selection loss is 8.744 FID. The $K _ { \mathrm { i n f e r } } = K _ { \mathrm { t r a i n } } + 1$ rule performs better; ranking observed calibration anchors recovers all three optima using its multiple-depth measurements. The tested local feature approximations therefore do not supply a predictive stopping rule.

Earlier representation diagnostics. A development score combining gradient compatibility, geometry gain, and perturbation response selects co-located depth two, whereas depth four yields the best measured FID. All four reported semantic-direction perturbation ratios exceed one. A median geometry-peak rule predicts horizons 3, 4, 6, compared with observed 3, 5, 6. A separate frozen REG-baseline score selects its already supervised block four while its mean extra-step geometry gains remain negative at every site. These results distinguish supervision placement and recurrent training from prediction of the optimal depth.

<table><tr><td>Method</td><td>Direction hits</td><td>Depth hits</td><td>Change MAE</td><td>Mean selection loss</td></tr><tr><td>Quadratic feature extrapolation</td><td>3/8</td><td>0/3</td><td>8550.500</td><td>8.744</td></tr><tr><td>Quadratic, half step</td><td>3/8</td><td>0/3</td><td>82100.946</td><td>8.744</td></tr><tr><td>Linear feature extrapolation</td><td>3/8</td><td>0/3</td><td>242.900</td><td>8.744</td></tr><tr><td>Zero change</td><td>0/8</td><td>0/3</td><td>3.560</td><td>8.744</td></tr><tr><td> $K _ { \mathrm { t r a i n } } + 1$ </td><td></td><td>1/3</td><td></td><td>0.405</td></tr><tr><td>Observed calibration anchors</td><td></td><td>3/3</td><td></td><td>0.000</td></tr></table>

Table 8: Local feature extrapolation on 4,096-sample FID. Direction hits compare signs over eight edges. Selection loss is measured FID at the selected depth minus the best measured FID; depths are chosen by accumulated predicted changes, with smaller-depth tie-breaking.

## F XL TRAINING MILESTONES

Table 9 summarizes the 50K, 100K, 200K, and 400K checkpoints, together with the 220K checkpoint used in the main-text comparison with published REG. Each row identifies the evaluated inferencedepth set; a dash marks a checkpoint evaluated at only one depth. Gain is relative to the same checkpoint at $K _ { \mathrm { i n f e r } } = 2 ,$ , and recall is reported at its best observed FID. Published REG results are
<table><tr><td>Steps Tested  $K _ { \mathrm { i n f e r } }$ </td><td>FID at 2</td><td>Best  $K _ { \mathrm { i n f e r } }$ </td><td>Best FID</td><td>Gain (%)</td><td>Recall</td></tr><tr><td>50K 2,3,4,5</td><td>25.1962</td><td>4</td><td>11.8154</td><td>53.11</td><td>0.3407</td></tr><tr><td>100K 2,3,4,5</td><td>8.3360</td><td>3</td><td>4.7552</td><td>42.96</td><td>0.4682</td></tr><tr><td>200K 2,3,4,5</td><td>4.6383</td><td>3</td><td>3.4942</td><td>24.67</td><td>0.5010</td></tr><tr><td>220K 3</td><td></td><td>一</td><td>3.3675</td><td></td><td>0.5180</td></tr><tr><td>400K 1,2,3,4,5</td><td>3.3823</td><td>3</td><td>2.8376</td><td>16.10</td><td>0.5490</td></tr></table>

Table 9: Representative XL training checkpoints. Best depths are minima over the listed sets. The 220K checkpoint supplies a single-depth observation. All metrics at the final 400K checkpoint appear in Appendix H.

reference values from Table 9 of Wu et al. (2025); the comparison uses no local non-loop REG-XL run. Checkpoint crossings reported in the main text refer to the earliest evaluated observations in the full record, rather than exact crossing times.

## G DINO REPRESENTATION TRAJECTORIES

The final probe evaluates 1,024 forward-noised real images at $t \in \{ . 1 , . 3 , . 5 , . 7 , . 9 \}$ . Unbiased linear CKA (Kornblith et al., 2019) compares mean hidden patch features with either mean DINO patch features or DINO CLS features. The launcher caps each CKA calculation at a deterministic 256- image subset; this cap is specified by the launcher rather than the saved per-model metadata. “CLS $\mathrm { C K } \bar { \mathsf { A } } ^ { , , }$ names the teacher target, not the model’s semantic-token state. Spatial correlations compare off-diagonal normalized patch Gram entries, while native cosine compares normalized projected features with DINO targets. Both use all 1,024 images and 64 sampled patch positions.

![](images/21e5d582f6272714e761ace42bbbbf8cca98a939bb859069ee89d1141c736d58.jpg)  
Figure 9: Representation geometry and native alignment along recurrent depth. FID, patch-mean CKA/spatial correlation, and native projector cosine use aligned depth axes. Geometry is averaged over five noise times from the final 1,024-image probe (CKA subset: 256). These are teacher-forced hidden trajectories; FID comes from separate full-generation evaluations. Vertical scales differ across panels.

For LoopREG trained at $K _ { \mathrm { t r a i n } } = 2 .$ , patch-mean CKA and spatial correlation peak at $K _ { \mathrm { i n f e r } } = 3$ matching the FID minimum; native projector cosine peaks at $\mathrm { \bar { \it K } _ { \mathrm { i n f e r } } = 2 \ }$ . For $K _ { \mathrm { t r a i n } } = 3 .$ , all three geometry metrics peak at $K _ { \mathrm { i n f e r } } = 5 \bar { }$ , while native cosine peaks at $K _ { \mathrm { i n f e r } } = 3$ . The correspondence depends on the metric and model: teacher-CLS CKA for train-two peaks at four, and LoopSiT’s geometry similarity increases even as FID deteriorates. Relative hidden-state increments also do not establish global contraction. The full probe covers five models, eight recurrent depths, and five diffusion times, including baseline hidden-state probes. Those baseline probes are not generation results at additional depths.

## H FINAL-CHECKPOINT GENERATION SWEEPS

The following tables give every evaluated inference depth at the 400K checkpoint for the core SiT/REPA/REG comparisons and both larger scales. This selection is by model and checkpoint, not by metric value. Site interventions, random-depth policies, and unshared controls are summarized in Appendix D.

“Blocks” counts Transformer-block executions, not exact FLOPs. Bold marks each metric’s optimum within a model and checkpoint; FID and sFID are minimized, and IS, precision, and recall are maximized. Displayed values are rounded; canonical records retain source precision.

## H.1 SIT-B/2: SUPERVISION AND TRAINING HORIZON

<table><tr><td> $K _ { \mathrm { i n f e r } }$ </td><td>Blocks</td><td>FID↓</td><td>sFID↓</td><td>IS ↑</td><td>P↑</td><td>R↑</td></tr><tr><td colspan="7">LoopSiT-B/2, B4,  $K _ { \mathrm { t r a i n } } = 2$ </td></tr><tr><td>1</td><td>12</td><td>52.0813</td><td>11.3411</td><td>28.9425</td><td>0.3676</td><td>0.6608</td></tr><tr><td>2</td><td>13</td><td>34.4524</td><td>6.4313</td><td>43.0118</td><td>0.5334</td><td>0.6387</td></tr><tr><td>3</td><td>14</td><td>46.4096</td><td>10.0098</td><td>34.2806</td><td>0.4637</td><td>0.6182</td></tr><tr><td>4</td><td>15</td><td>78.8033</td><td>19.1662</td><td>20.7103</td><td>0.3153</td><td>0.5667</td></tr><tr><td>5</td><td>16</td><td>118.9361</td><td>31.6541</td><td>12.4613</td><td>0.1959</td><td>0.4317</td></tr><tr><td>6</td><td>17</td><td>149.0437</td><td>42.9569</td><td>8.6622</td><td>0.1284</td><td>0.2810</td></tr><tr><td>7</td><td>18</td><td>168.4506</td><td>53.1676</td><td>6.6430</td><td>0.0944</td><td>0.1592</td></tr><tr><td>8</td><td>19</td><td>181.3879</td><td>63.7429</td><td>5.4062</td><td>0.0754</td><td>0.0928</td></tr><tr><td colspan="7">LoopREPA-B/2, d=4, B4, Ktrain = 2</td></tr><tr><td>1</td><td>12</td><td>88.8098</td><td>20.6534</td><td>14.7986</td><td>0.2660</td><td>0.6509</td></tr><tr><td>2</td><td>13</td><td>21.9274</td><td>6.3655</td><td>66.5780</td><td>0.6013</td><td>0.6515</td></tr><tr><td>3</td><td>14</td><td>16.7427</td><td>6.6802</td><td>84.6709</td><td>0.6285</td><td>0.6147</td></tr><tr><td>4</td><td>15</td><td>28.3614</td><td>9.0526</td><td>61.8826</td><td>0.4946</td><td>0.6118</td></tr><tr><td>5</td><td>16</td><td>59.4920</td><td>14.8863</td><td>30.6378</td><td>0.2966</td><td>0.5953</td></tr><tr><td>6</td><td>17</td><td>101.3782</td><td>27.0815</td><td>15.0184</td><td>0.1545</td><td>0.4585</td></tr><tr><td>7</td><td>18</td><td>134.6486</td><td>45.3181</td><td>9.2247</td><td>0.0925</td><td>0.2664</td></tr><tr><td>8</td><td>19</td><td>158.8770</td><td>64.4720</td><td>6.6918</td><td>0.0646</td><td>0.1125</td></tr><tr><td colspan="7">LoopREG-B/2, d=4, B4,  $K _ { \mathrm { t r a i n } } = 2$ </td></tr><tr><td>1</td><td>12</td><td>90.5972</td><td>16.9459</td><td>14.6801</td><td>0.2443</td><td>0.6385</td></tr><tr><td>2</td><td>13</td><td>14.7094</td><td>6.7629</td><td>97.1320</td><td>0.6496</td><td>0.6309</td></tr><tr><td>3</td><td>14</td><td>5.3434</td><td>5.8021</td><td>185.9796</td><td>0.7958</td><td>0.5077</td></tr><tr><td>4</td><td>15</td><td>6.2050</td><td>6.6180</td><td>207.7940</td><td>0.8078</td><td>0.4337</td></tr><tr><td>5</td><td>16</td><td>9.6552</td><td>9.4030</td><td>174.6141</td><td>0.7287</td><td>0.4142</td></tr><tr><td>6</td><td>17</td><td>20.9611</td><td>15.3694</td><td>108.8510</td><td>0.5695</td><td>0.4171</td></tr><tr><td>7</td><td>18</td><td>48.6352</td><td>25.9636</td><td>50.0054</td><td>0.3735</td><td>0.4216</td></tr><tr><td>8</td><td>19</td><td>88.6875</td><td>40.5806</td><td>22.4223</td><td>0.2294</td><td>0.3521</td></tr><tr><td colspan="7">LoopREG-B/2, d=4, B4,  $K _ { \mathrm { t r a i n } } = 3$ </td></tr><tr><td>1</td><td>12</td><td>117.7844</td><td>33.5290</td><td>10.0884</td><td>0.1792</td><td>0.4871</td></tr><tr><td>2</td><td>13</td><td>66.7103</td><td>16.5472</td><td>23.7143</td><td>0.3315</td><td>0.7063 0.6337</td></tr><tr><td>3</td><td>14</td><td>13.7500</td><td>6.6929</td><td>100.8156</td><td>0.6578</td><td></td></tr><tr><td>4</td><td>15</td><td>5.2002</td><td>5.4001</td><td>178.3913</td><td>0.8023</td><td>0.5243 0.4515</td></tr><tr><td>5</td><td>16</td><td>4.9136</td><td>5.4032</td><td>217.6060 224.6076</td><td>0.8506</td><td></td></tr><tr><td>6</td><td>17</td><td>5.7158</td><td>5.9812</td><td></td><td>0.8513</td><td>0.4107</td></tr><tr><td>7 8</td><td>18</td><td>6.8081</td><td>7.1051</td><td>209.2107</td><td>0.8195</td><td>0.3972</td></tr><tr><td></td><td>19</td><td>9.5347</td><td>8.9836</td><td>173.4586</td><td>0.7508</td><td>0.3977</td></tr><tr><td colspan="7">LoopREG-B/2, d=4, B4, Ktrain = 4</td></tr><tr><td>1</td><td>12</td><td>121.6970</td><td>35.4991</td><td>9.3515</td><td>0.1827</td><td>0.4151</td></tr><tr><td>2</td><td>13</td><td>93.8612</td><td>23.3516</td><td>14.4801 39.1530</td><td>0.2425 0.4310</td><td>0.6437 0.7128</td></tr><tr><td>3</td><td>14</td><td>44.1952</td><td>11.7995</td><td></td><td></td><td></td></tr><tr><td>4</td><td>15</td><td>13.2435</td><td>6.5524</td><td>102.6734</td><td>0.6605</td><td>0.6335</td></tr><tr><td>5</td><td>16</td><td>5.8749</td><td>5.3505</td><td>161.3391</td><td>0.7849</td><td>0.5518</td></tr><tr><td>6</td><td>17</td><td>5.1874</td><td>5.2995</td><td>193.9240</td><td>0.8394</td><td>0.4845</td></tr><tr><td>7</td><td>18</td><td>5.8128</td><td>5.6755</td><td>206.5647</td><td>0.8531</td><td>0.4439</td></tr><tr><td>8</td><td>19</td><td>6.6785</td><td>6.2577</td><td>204.0657</td><td>0.8496</td><td>0.4178</td></tr></table>

## H.2 SIT-L/2 AND SIT-XL/2

<table><tr><td> $K _ { \mathrm { i n f e r } }$ </td><td>Blocks</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>P↑</td><td>R↑</td></tr><tr><td colspan="7">LoopREG-L/2, d=8, B8,  $K _ { \mathrm { t r a i n } } = 2$ </td></tr><tr><td>1</td><td>24</td><td>59.4139</td><td>12.6041</td><td>26.0546</td><td>0.3214</td><td>0.7664</td></tr><tr><td>2</td><td>25</td><td>4.2286</td><td>5.0924</td><td>170.3622</td><td>0.7516</td><td>0.6420</td></tr><tr><td>3</td><td>26</td><td>3.2082</td><td>4.4353</td><td>262.6577</td><td>0.8593</td><td>0.5201</td></tr><tr><td>4</td><td>27</td><td>5.4971</td><td>4.5898</td><td>291.8665</td><td>0.8827</td><td>0.4438</td></tr><tr><td>5</td><td>28</td><td>7.0317</td><td>4.9216</td><td>267.8294</td><td>0.8450</td><td>0.4206</td></tr><tr><td colspan="7">LoopREG-XL/2, d=8, B8,  $K _ { \mathrm { t r a i n } } = 2$ </td></tr><tr><td>1</td><td>28</td><td>52.4630</td><td>10.6201</td><td>31.5308</td><td>0.3699</td><td>0.7668</td></tr><tr><td>2</td><td>29</td><td>3.3823</td><td>4.8470</td><td>184.7094</td><td>0.7584</td><td>0.6347</td></tr><tr><td>3</td><td>30</td><td>2.8376</td><td>4.6353</td><td>254.6991</td><td>0.8386</td><td>0.5490</td></tr><tr><td>4</td><td>31</td><td>3.7734</td><td>5.1102</td><td>239.4408</td><td>0.8179</td><td>0.5240</td></tr><tr><td>5</td><td>32</td><td>7.8614</td><td>7.0541</td><td>164.5297</td><td>0.7154</td><td>0.5457</td></tr></table>