# FFAI: Flow Matching with Feedback Training and Adaptive Initialization for CT Reconstruction

Jigang Duan<sup>†</sup>, Heran Wang<sup>†</sup>, Genwei Ma, and Xing Zhao<sup>∗</sup>

Abstract—Although low-dose, sparse-view, and limited-angle CT involve distinct degradation mechanisms, all three pose reconstruction challenges due to limited measurement information and can therefore be addressed within a unified framework. In iterative image restoration with flow matching, prediction errors may cause inference states to deviate from the prescribed training path, while a fixed initialization may be unsuitable for inputs with varying degradation severity. We propose a unified conditional flow matching method that conditions on the initial reconstruction and combines feedback-aware training with degradation-adaptive initialization. During training, a teacher network simulates inference updates to generate shifted states, from which a student network learns to recover high-quality reference images, improving its ability to handle prediction errors. During inference, the starting time is selected based on the noise level or extent of missing projection data, and the noise added to the initial state is adjusted accordingly to adapt the starting point to the input. Experiments on public benchmark datasets show that the proposed method achieves better quantitative results than the compared methods across all three reconstruction tasks, effectively suppressing noise and artifacts while preserving anatomical structures and details.

Index Terms—Computed tomography, conditional flow matching, degradation-adaptive initialization, feedback-aware training, image reconstruction.

## I. INTRODUCTION

important role in disease screening and clinical diagnosis. Maintaining image quality while reducing radiation dose remains a central goal in medical CT [1]. Radiation dose is typically reduced by decreasing either the incident photon count per view or the number of projection views. The former yields low-dose CT, whereas the latter can be achieved by increasing the angular sampling interval or reducing the scan angular range, corresponding to sparse-view and limitedangle CT, respectively [2]. Despite their distinct degradation mechanisms, all three conditions provide limited measurement information, weakening the constraints that measured projections impose on the reconstructed image. We collectively refer to these conditions as degraded acquisition and investigate a unified reconstruction method.

A. Trends in Models and Algorithms

Model-based iterative reconstruction (MBIR) combines data fidelity and image priors to constrain reconstruction. For lowdose CT, reconstruction can use a likelihood derived from measurement statistics or a penalized weighted least-squares formulation [3], [4]. For sparse-view and limited-angle CT, the projection operator is defined by the acquired angles. Priors such as total variation, prior-image constraints, and learned sparsifying transforms further compensate for insufficient measurements [5]–[8]. These methods explicitly combine measurements with prior information, but their performance remains dependent on the expressiveness of the prior and the choice of regularization parameters. Preserving details while removing noise and artifacts remains challenging.

Deep learning methods learn richer image features from data. Direct mapping methods recover high-quality images in the image domain, projection domain, or both [9]–[11]. Noise2Noise can also learn restoration mappings from independent noisy observations of the same signal, reducing reliance on clean targets [12]. However, without explicit measurement constraints during inference, conditioning on a degraded input alone does not guarantee consistency with the measured projections. Model-driven networks integrate data-fidelity updates with learned modules, allowing measurements to guide multiple stages of restoration [13]–[16]. To accommodate different acquisition conditions, ProCT uses view prompts for sparse-view and limited-angle reconstruction, while PromptCT incorporates sampling prompts into an unrolled network to handle multiple sparse-view configurations [17], [18]. TAMP addresses all three degradation conditions through large-scale simulation-based pretraining, dualdomain supervision, and parameter-efficient fine-tuning [2]. These studies provide a foundation for unified reconstruction under different acquisition conditions, but adaptation across conditions and error control during iterative restoration remain distinct challenges.

Generative models learn the distribution of high-quality images to provide priors for structural recovery from insufficient measurements. Generative adversarial networks have been used to enhance detail and texture in low-dose CT [19], [20], while diffusion, score-based generative, and consistency models have been applied to low-dose, sparse-view, and limited-angle reconstruction [21]–[26]. However, an image consistent with the training distribution may not accurately represent the imaged object, and a mismatch between the prior and the object can produce spurious structures [27]. In multistep generation, each prediction also affects subsequent inputs, allowing prediction errors to propagate [28]. Applying generative priors to reconstruction therefore requires attention to the network’s ability to handle the states encountered during inference.

Previous studies have strengthened measurement guidance by jointly optimizing generative priors and data-fidelity terms, incorporating measurement constraints into generative updates, or integrating iterative reconstruction with generative sampling [29]–[33]. FORCE combines a PFGM++ prior with taskspecific measurement correction for low-dose CT, sparseview CT, and metal artifact correction [34]. These methods demonstrate that generative priors can accommodate different observation models. However, measurement correction also changes the state passed to the network at the next step. A network trained only on prescribed noisy or interpolated states may not fully accommodate these changes.

To address state changes during iterative restoration, CoreDiff re-degrades network predictions for training, enabling the restoration network to handle subsequent inputs derived from its own predictions [28]. CDDM trains an additional discrepancy mitigation model within a cascaded diffusion framework to handle states after data-consistency correction [35]. These studies indicate that, in addition to incorporating measurement constraints at inference, training should cover the states produced by the actual updates. Flow matching (FM) learns a velocity field connecting source and target distributions, providing a convenient formulation for this joint design [36]. Along a linear conditional path, the current state and predicted velocity directly yield an estimate of the target endpoint. FLOWER uses this formulation to solve inverse problems by combining endpoint prediction, measurement correction, and state advancement [37]. Here, we further investigate state shifts caused by sequential predictions and the suitability of a fixed initialization for inputs with varying degradation severity. Feedback-aware training and degradation-guided initialization connect the requirements of inference with the existing FM training path.

## B. Motivation

Although the three CT tasks involve different degradation mechanisms, their initial reconstructions all retain some structural information about the imaged object. Based on this shared property, we develop FFAI, a unified conditional FM method that uses the initial reconstruction and acquisition conditions to guide restoration. As shown in Fig. 1(A), the three tasks share the same training objective and inference rules, with parameters learned separately for each task. Feedbackbased error correction during training mitigates the effects of prediction errors, while degradation adaptation during inference selects an appropriate starting point for each input.

Training should equip the network to correct prediction errors. Standard FM training uses states sampled from a prescribed path, whereas each inference input depends on the preceding prediction. Prediction errors may shift subsequent inputs away from the training-state distribution, making them difficult for the network to handle and allowing errors to propagate, as illustrated on the left of Fig. 1(B). To address this issue, a teacher network simulates inference updates, and the resulting shifted states serve as additional training inputs. Using high-quality reference images as targets, the restoration network learns to continue restoration from these states, as shown on the right of Fig. 1(B). Training on both prescribed path states and teacher-generated feedback states teaches the network the restoration process as well as how to handle deviations encountered along the way.

![](images/8a77997de0d02425c706cf222a0b88e4ee5e12cdde87870cf9c4c3101a2b46a4.jpg)  
Fig. 1. Motivation and key mechanisms of FFAI. (A) The three CT tasks share a unified training procedure, with parameters learned separately for each task. (B) Feedback-based error correction: shifted states generated by teachersimulated inference are incorporated into training, with high-quality reference images supervising restoration to improve the network’s ability to handle prediction errors. (C) Degradation adaptation: the starting time and initial state are jointly determined by the input degradation severity to accommodate different restoration needs.

Inference should adapt its starting point to the input degradation severity. Initial reconstructions with different degradation levels retain different amounts of information. A fixed starting configuration may therefore be unsuitable for some inputs and lead to substantial variation in restoration quality, as shown on the left of Fig. 1(C). We quantify degradation severity using the noise level or sampling conditions and use it to select a starting time on the existing FM training path. The initial reconstruction is then used to construct an inference input that approximately matches the training state at that time. As illustrated on the right of Fig. 1(C), this degradationadaptive initialization selects a corresponding starting time and image state for each input, adapting restoration to its degradation severity.

## C. Our Contributions

The main contributions are as follows:

• We develop FFAI, a unified conditional FM method for low-dose, sparse-view, and limited-angle CT, in which the initial reconstruction and acquisition conditions jointly guide restoration.

• We design feedback-aware training guided by an exponential moving average (EMA) teacher, enabling the network to recover target images from shifted states produced by sequential predictions.

<table><tr><td rowspan="2">LACT</td><td rowspan="2">2  $c _ { \mathrm { L A } }$ </td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Feedback-Aware FM Training</td><td rowspan="2"></td><td rowspan="2">LACT Model</td><td rowspan="2"></td><td rowspan="2"> $\pmb { v } _ { \theta _ { \mathrm { L A } } }$ </td></tr><tr><td></td><td>Feedback States</td><td></td></tr><tr><td>SVCT</td><td> $\pmb { c } _ { \mathrm { S V } }$ </td><td></td><td></td><td>M00-00  $\boldsymbol { v } _ { \boldsymbol { \theta } _ { \omega } }$ </td><td>EMA Update</td><td>10m-</td><td></td><td>SVCT Model</td><td></td><td> $\pmb { v } _ { \theta _ { \mathrm { S V } } }$ </td></tr><tr><td>LDCT</td><td> ${ \pmb { c } } _ { \mathrm { L D } }$ </td><td></td><td></td><td>Student Network</td><td>ω ∈ {LA,SV,LD}</td><td> $\boldsymbol { v } _ { \widetilde { \theta } _ { \omega } }$  Teacher Network</td><td></td><td>LDCT Model</td><td></td><td> $\boldsymbol { v } _ { \boldsymbol { \theta } _ { \mathrm { L D } } }$ </td></tr></table>

Fig. 2. Unified training framework for the three tasks. Each task uses the initial reconstruction as a condition and shares the same flow matching and feedback training mechanisms, while network parameters are learned independently. After training, the student model for each task is retained for inference.

• We introduce a quantitative, degradation-guided initialization that jointly determines the starting time and initial state on the existing FM training path to accommodate reconstruction at different degradation levels.

## II. METHODOLOGY

## A. Conditional Flow Matching for CT Reconstruction

FFAI progressively restores initial CT reconstructions obtained under degraded acquisition conditions using a conditional velocity field. As shown in Fig. 2, low-dose, sparseview, and limited-angle CT share the same training objective and inference rules, with parameters learned separately for each task. The model first learns the restoration velocity field through standard FM and then learns to handle prediction errors using feedback states. During inference, the starting point is determined by the input degradation severity, and the image is progressively restored along the learned velocity field.

Let $\omega \in \{ \mathrm { L D } , \mathrm { S V } , \mathrm { L A } \}$ denote the task, $\scriptstyle { \mathbf { 3 } } \omega$ the projection measurements, and $\scriptstyle { \pmb x } ^ { \star }$ the paired high-quality reference image. An initial reconstruction is obtained using a direct reconstruction operator $B _ { \omega } ,$ and a Gaussian perturbation is added to construct the source endpoint for training:

$$
\begin{array} { l } { { { \pmb { c } } _ { \omega } = B _ { \omega } ( { \pmb y } _ { \omega } ) , } } \\ { { { \pmb z } } _ { \omega } = { \pmb { c } } _ { \omega } + \gamma _ { \omega } { \pmb { \epsilon } } , \qquad { \pmb { \epsilon } } \sim { \mathcal { N } } ( { \bf 0 } , { \pmb I } ) , } \end{array}\tag{1}
$$

where $\gamma _ { \omega } \geq 0$ controls the perturbation amplitude at the source endpoint. All images are normalized consistently. The linear path connecting the reference image and the source endpoint is

$$
\pmb { x } _ { t } = ( 1 - t ) \pmb { x } ^ { \star } + t \pmb { z } _ { \omega } , \qquad t \in [ 0 , 1 ] .\tag{2}
$$

The path corresponds to the source endpoint at t = 1 and the reference image at $t = 0 ;$ reconstruction therefore proceeds backward in time.

We describe a single task below and abbreviate its parameters $\theta _ { \omega }$ as θ. The target velocity along the path is ${ \pmb u } _ { \omega } = z _ { \omega } - { \pmb x } ^ { \star }$ . The velocity network takes the current state $\mathbf { \nabla } _ { \mathbf { x } _ { t } } .$ , time t, and initial reconstruction $\scriptstyle { c _ { \omega } }$ as inputs and is trained by minimizing

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \Big [ \| \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t \mid \pmb { c } _ { \omega } ) - \pmb { u } _ { \omega } \| _ { 2 } ^ { 2 } \Big ] .\tag{3}
$$

The expectation is taken over training samples, random perturbations, and sampled times. The image condition $\scriptstyle { c _ { \omega } }$ remains fixed throughout restoration, providing object-specific information for velocity prediction at each step.

## B. Feedback-Aware FM Training

Standard FM trains on states sampled from a prescribed path, whereas inference states are generated by successive network predictions. To accommodate state shifts caused by prediction errors, FFAI uses two-stage training. First, an initial network is trained by minimizing only $\mathcal { L } _ { \mathrm { F M } }$ . This network then serves as the student, and an EMA teacher is initialized with the same parameters. Feedback training is subsequently introduced while retaining standard FM supervision. Fig. 3 shows the two training branches in the second stage.

To match the time interval of the feedback update to that used during inference, we define the inference grid as ${ \boldsymbol { s } } =$ $\{ s _ { j } = j / ( \mathsf { \bar { N } } - 1 ) \} _ { j = 0 } ^ { N - 1 }$ , where N is the number of nodes. During training, t is randomly sampled from a denser time grid, and the two adjacent inference nodes bracketing t are selected, with $t _ { L }$ denoting the larger time and $t _ { R }$ the smaller time, such that $0 < t _ { R } < t < t _ { L } \leq 1$ . The states ${ \mathbf { } } _ { { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } \mathbf { } { \mathbf { } } ^ { \mathbf { } } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf } \mathbf { \mathbf \mathbf } \mathbf { \mathbf } \mathbf \mathbf { \mathbf } \mathbf \mathbf \mathbf  $ and $\mathbf { \mathcal { x } } _ { t _ { L } }$ are constructed using (2) with the same training sample and random perturbation. The standard branch evaluates (3) at $t _ { R } .$ In the feedback branch, the teacher performs one Euler update from $\mathbf { \mathcal { x } } _ { t _ { L } }$

$$
\widetilde { \pmb { x } } _ { t _ { R } } = \mathrm { s g } \big [ \pmb { x } _ { t _ { L } } + ( t _ { R } - t _ { L } ) ~ \pmb { v } _ { \widetilde { \theta } } ( \pmb { x } _ { t _ { L } } , t _ { L } \mid \pmb { c } _ { \omega } ) \big ] ,\tag{4}
$$

where $\bar { \theta }$ denotes the teacher parameters and $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operation. The deviation of this state from the prescribed path satisfies

$$
\begin{array} { r } { \widetilde { \pmb { x } } _ { t _ { R } } - \pmb { x } _ { t _ { R } } = \left( t _ { R } - t _ { L } \right) \left[ \pmb { v } _ { \bar { \theta } } ( \pmb { x } _ { t _ { L } } , t _ { L } \mid \pmb { c } _ { \omega } ) - \pmb { u } _ { \omega } \right] . } \end{array}\tag{5}
$$

Thus, the feedback state in Fig. 3 reflects the teacher’s actual prediction error, exposing the student during training to shifted inputs generated by a single inference step.

The original path velocity label no longer applies to a shifted state. Using the endpoint relation of the linear path, we convert the student’s predicted velocity into a target image estimate and supervise it directly with the reference image:

$$
\begin{array} { r l } & { \widehat { \pmb { x } } _ { \mathrm { F B } } = \widetilde { \pmb { x } } _ { t _ { R } } - t _ { R } { \pmb { v } } _ { \theta } ( \widetilde { \pmb { x } } _ { t _ { R } } , t _ { R } \mid { \pmb { c } } _ { \omega } ) , } \\ & { \mathcal { L } _ { \mathrm { F B } } = \mathbb { E } \left[ \left\| \widehat { \pmb { x } } _ { \mathrm { F B } } - { \pmb { x } } ^ { \star } \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{6}
$$

The teacher only generates the inputs; the supervision target remains $\scriptstyle { \pmb x } ^ { \star }$ . Gradients are stopped at the feedback input, so this loss updates only the student parameters.

The joint objective for the second stage is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { F B } } \mathcal { L } _ { \mathrm { F B } } , } \end{array}\tag{7}
$$

![](images/2add1ee43a6680f1b4f605ce5a50993df1ba055b7dd8318b853e1e6eb7563144.jpg)  
Fig. 3. Feedback-aware FM training. The standard branch learns the target velocity at the prescribed path state $\mathbf { \mathscr { x } } _ { t _ { R } } .$ . In the feedback branch, the EMA teacher performs a single update from ${ \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } \qquad { \mathbf { } } _ { { \mathbf { } } } _ { { \mathbf { } \mathbf { } } } _ { { \mathbf } }$ to generate a stop-gradient feedback state $\widetilde { \pmb { x } } _ { t _ { R } } .$ and the student’s endpoint prediction is supervised by the reference image. $t _ { L }$ and $t _ { R }$ are the nearest inference nodes on either side of the randomly sampled time t.

```latex
Algorithm 1 Two-Stage Feedback-Aware FM Training
Require: Task $\omega ,$ paired training data, inference grid S,
training iterations $T _ { \mathrm { p r e } }$ and $T _ { \mathrm { f b } }$
Ensure: Task model parameters $\theta _ { \omega }$
1: Initialize student parameters θ
2: Stage 1: Standard FM Pretraining
3: for $n = 1$ to $T _ { \mathrm { p r e } }$ do
4: Sample a training pair, perturbation ϵ, and time t
5: Construct $\begin{array} { r } { { \cal c } _ { \omega } , z _ { \omega } , } \end{array}$ and $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ using (1)–(2)
6: Update θ by minimizing $\mathcal { L } _ { \mathrm { F M } }$
7: end for
8: Stage 2: Feedback-Aware Training
9: Initialize the teacher $\bar { \theta }  \theta$
10: for $n = 1$ to $T _ { \mathrm { f b } }$ do
11: Sample a training pair, perturbation $\epsilon ,$ and time t
12: Select the nearest inference nodes on either side of t,
satisfying $0 < t _ { R } < t < t _ { L } \leq 1$
13: Construct $\mathbf { \mathcal { x } } _ { t _ { L } }$ and ${ \mathbf { } } _ { { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } \mathbf { } { \mathbf { } } ^ { \mathbf { } } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf } \mathbf { \mathbf \mathbf } \mathbf { \mathbf } \mathbf \mathbf { \mathbf } \mathbf \mathbf \mathbf  $ on the same path
14: Compute $\mathcal { L } _ { \mathrm { F M } }$ for the student at $\scriptstyle { \pmb { x } } _ { t _ { R } }$
15: Generate the feedback state using the teacher via (4)
16: Compute $\mathcal { L } _ { \mathrm { F B } }$ for the student using (6)
17: Update θ by minimizing (7)
18: Update <sup>¯</sup>θ using (8)
19: end for
20: $\theta _ { \omega } \gets \theta$
```

where $\lambda _ { \mathrm { F B } } ~ > ~ 0$ balances path velocity supervision and feedback endpoint supervision. After each student update, the teacher parameters are updated as

$$
\bar { \theta }  \alpha \bar { \theta } + ( 1 - \alpha ) \theta , \qquad \alpha \in [ 0 , 1 ) .\tag{8}
$$

After training, the student parameters $\theta _ { \omega }$ are retained as the task model. Algorithm 1 summarizes the two-stage training procedure.

## C. Degradation-Adaptive Inference

The information retained in the initial reconstruction varies with degradation severity. FFAI adjusts the starting point accordingly: milder degradation corresponds to a smaller starting time, and more severe degradation to a larger starting time. As shown in Fig. 4, the degradation descriptor $\mathbf { \Delta } \mathbf { q } _ { \omega }$ determines the restoration interval through the time mapping $\tau _ { \omega } = \mathcal { T } _ { \omega } ( q _ { \omega } )$ and the selected time is then used to construct the initial image.

The initial image is constructed to match the noise scale of the training path. From (2), the state at time $\tau _ { \omega }$ is $\begin{array} { r l } {  { \big ( 1 - \tau _ { \omega } \big ) { \pmb x } ^ { \star } + } } \end{array}$ $\tau _ { \omega } c _ { \omega } + \tau _ { \omega } \gamma _ { \omega } \epsilon$ . Since the reference image is unavailable during inference, we approximate $\scriptstyle { \pmb x } ^ { \star }$ with $\mathbf { } _ { c _ { \omega } , }$ , yielding

$$
\begin{array} { r } { \pmb { x } _ { \mathrm { i n i t } } = \mathcal { T } _ { \omega } ( \pmb { c } _ { \omega } , \pmb { q } _ { \omega } , \tau _ { \omega } , \pmb { \epsilon } ) } \\ { = \pmb { c } _ { \omega } + \tau _ { \omega } \gamma _ { \omega } \pmb { \epsilon } , \qquad \ \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } ) . } \end{array}\tag{9}
$$

The standard deviation of the added noise is thus scaled with the starting time as $\tau _ { \omega } \gamma _ { \omega } .$ , where $\gamma _ { \omega }$ is unchanged from training. For the same noise realization, the difference between this initialization and the corresponding training state is $( 1 - \tau _ { \omega } ) ( \pmb { c } _ { \omega } - \pmb { x } ^ { \star } )$ . This construction preserves the scale of the added noise at the selected time, while the state discrepancy is determined by the initial reconstruction error.

The three tasks use the following time selection rules.

1) Low-Dose CT: Low-dose degradation primarily manifests as increased noise. We therefore estimate the noise level from the initial reconstruction as $\pmb q _ { \mathrm { L D } } = \widehat { \sigma } = \mathcal { E } ( \pmb { c } _ { \mathrm { L D } } )$ . Specifically, the image is divided into local $2 \times 2$ pixel blocks, and differences between sorted pixel values within each block are computed. These differences are calibrated to the Gaussian noise scale and aggregated across blocks to estimate the overall noise standard deviation. The starting time is set to

$$
\tau _ { \mathrm { L D } } = \mathrm { c l i p } \left( \frac { \widehat { \sigma } } { \sigma _ { \mathrm { r e f } } } , 0 , 1 \right) ,\tag{10}
$$

where $\sigma _ { \mathrm { r e f } } > 0$ is a reference noise level defined under the same image normalization, and clip(·, 0, 1) restricts the value to [0, 1]. A higher estimated noise level yields a longer restoration interval and adjusts the initial perturbation accordingly through (9).

2) Sparse-View CT: Sparse-view degradation arises from angular undersampling over the full scan range. Let Θ denote the reference angle set for a full scan and $\Omega \subset \Theta$ the acquired angle set. The missing-view fraction $\mathbf { \Delta } q _ { \mathrm { S V } } = \eta$ determines the starting time:

$$
\tau _ { \mathrm { S V } } = \eta = 1 - \frac { | \Omega | } { | \Theta | } .\tag{11}
$$

![](images/a519bd62a75d6382c05314e63e329007d6e0ffe09a1b865883bfcf1849db9051.jpg)  
Fig. 4. Degradation-adaptive inference. The starting time $\tau _ { \omega }$ is selected from the degradation descriptor $\mathbf { } q _ { \omega } ,$ and the corresponding initial state is constructed. Conditioned on the fixed image ${ \bf c } _ { \omega } ,$ the task-specific student network performs Euler updates along a decreasing time grid to obtain the final reconstruction.

TABLE I  
QUANTITATIVE RESULTS FOR QUARTER-DOSE CT RECONSTRUCTION (MEAN ± STANDARD DEVIATION). SSIM IS REPORTED AS A PERCENTAGE. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST MEANS, RESPECTIVELY.
<table><tr><td>Metric</td><td>FBP</td><td></td><td>Noise2Noise MTD-GAN</td><td>FoundDiff</td><td>TAMP</td><td>FORCE</td><td>FFAI</td></tr><tr><td>PSNR (dB) ↑</td><td> $4 0 . 9 9 \pm 0 . 0 8$ </td><td> $4 1 . 7 9 \pm 0 . 2 3$ </td><td> $4 6 . 0 3 \pm 0 . 0 8$ </td><td> $4 5 . 5 1 \pm 0 . 0 6$ </td><td> $4 2 . 4 5 \pm 1 . 1 3$ </td><td> $4 0 . 8 1 \pm 0 . 3 3$ </td><td> ${ \bf 4 8 . 3 0 \pm 0 . 0 5 }$ </td></tr><tr><td>SSIM (%) ↑</td><td> $9 3 . 7 0 \pm 0 . 1 4$ </td><td> $9 5 . 5 0 \pm 0 . 1 0$ </td><td> $9 7 . 8 1 \pm 0 . 0 5$ </td><td> $9 7 . 8 7 \pm 0 . 5 3 $ </td><td> $9 6 . 4 3 \pm 1 . 3 9$ </td><td> $9 6 . 0 4 \pm 0 . 4 7$ </td><td> ${ \bf 9 7 . 9 6 \pm 0 . 3 9 }$ </td></tr></table>

For a fixed scan range and sampling scheme, fewer views yield a larger starting time; with more views available, restoration starts closer to the target endpoint.

3) Limited-Angle CT: Limited-angle degradation arises from a missing contiguous angular range. The acquisition interval is specified by its starting and ending angles, $\phi _ { \mathrm { s } }$ and $\phi _ { \mathrm { e } } ,$ with the descriptor

$$
\pmb { q } _ { \mathrm { L A } } = \left[ \sin \phi _ { \mathrm { s } } , \cos \phi _ { \mathrm { s } } , \sin \phi _ { \mathrm { e } } , \cos \phi _ { \mathrm { e } } \right] ^ { \mathsf { T } } .\tag{12}
$$

The actual angular coverage $\Delta \phi$ is computed according to the known scan direction and normalized by the reference fullscan range $\Delta \phi _ { \mathrm { r e f } }$ under the same imaging geometry:

$$
\tau _ { \mathrm { L A } } = 1 - \frac { \Delta \phi } { \Delta \phi _ { \mathrm { r e f } } } .\tag{13}
$$

This rule controls the restoration interval through the missingangle fraction: smaller angular coverage yields a larger starting time.

For all three tasks, the selected time is mapped to the nearest positive node in S. The discretized time, still denoted by $\tau _ { \omega } ,$ is used for initialization in (9). We then take the decreasing nodes from $\tau _ { \omega }$ to 0 on the predefined grid, denoted by $\tau _ { \omega } =$ $t _ { 0 } > t _ { 1 } > \cdots > t _ { K } = 0 ,$ , set $\mathbf { \Delta } \mathbf { x } _ { t _ { 0 } } = \mathbf { \Delta } \mathbf { x } _ { \mathrm { { i n i t } } }$ , and perform Euler updates using the task-specific student network:

$$
\pmb { x } _ { t _ { k + 1 } } = \pmb { x } _ { t _ { k } } + \left( t _ { k + 1 } - t _ { k } \right) \pmb { v } _ { \theta _ { \omega } } ( \pmb { x } _ { t _ { k } } , t _ { k } \mid \pmb { c } _ { \omega } ) ,\tag{14}
$$

where $k = 0 , \ldots , K - 1$ . This update uses the same time step and discretization as feedback training. The random perturbation is sampled only once, at initialization, and the final reconstruction is $\widehat { \mathbf { x } } _ { \omega } = \mathbf { x } _ { t _ { K } }$ . Algorithm 2 summarizes the inference procedure.

## III. EXPERIMENTS AND RESULTS

We evaluated FFAI on the Mayo 2016 low-dose CT dataset for low-dose, limited-angle, and sparse-view CT reconstruction. All two-dimensional slices had a matrix size of $5 1 2 \times 5 1 2$ pixels. The low-dose experiments used the official paired quarter-dose (QD25) and full-dose (FD) images as inputs and references, respectively. The training set comprised 1923 slice pairs from 8 patients (L067, L096, L109, L143, L192, L286, L291, and L310), and the test set included all 211 slice pairs from patient L506. For limited-angle CT, 1156 slices from 5 patients (L067, L096, L109, L143, and L192) were used for training. The test set comprised 210 slices from patient L286, of which 50 were randomly selected for evaluation. Two angular ranges, $6 0 ^ { \circ }$ and $9 0 ^ { \circ } .$ , were considered by retaining contiguous projections within [0<sup>◦</sup>, 60<sup>◦</sup>) and $[ 0 ^ { \circ } , 9 0 ^ { \circ } )$ , respectively. The sparse-view experiments used the same 5 training patients as the limited-angle experiments and 525 slices from patient L286 for evaluation. Two sparse-view settings were generated by uniformly selecting 40 or 80 projection views over the full $3 6 0 ^ { \circ }$ range. The corresponding full-dose CT images served as references in both the limited-angle and sparse-view experiments.

Algorithm 2 Degradation-Adaptive Inference   
Require: Task $\omega ,$ test projections $\scriptstyle { \mathbf { 1 } } _ { u _ { \omega } }$ and acquisition infor  
mation, student parameters $\theta _ { \omega } ,$ inference grid S   
Ensure: Reconstructed image $\widehat { \pmb { x } } _ { \omega }$   
1: Compute the initial reconstruction $\mathbf { c } _ { \omega } = B _ { \omega } ( \pmb { y } _ { \omega } )$   
2: Obtain the task-specific degradation descriptor $\mathbf { \Delta } \mathbf { q } _ { \omega }$   
3: Select $\tau _ { \omega }$ using the task-specific time rule and map it to   
the nearest positive inference node   
4: Sample $\epsilon \sim \mathcal { N } ( 0 , I )$   
5: Construct $\pmb { x } _ { \mathrm { i n i t } }$ using (9)   
6: Select decreasing nodes $\tau _ { \omega } = t _ { 0 } > \cdot \cdot \cdot > t _ { K } = 0$ and set   
$\mathbf { \Delta } \mathbf { x } _ { t _ { 0 } } = \mathbf { x } _ { \mathrm { i n i t } }$   
7: for $k = 0$ to $K - 1$ do   
8: Update to $\boldsymbol { x } _ { t _ { k + 1 } }$ using (14)   
9: end for   
10: $\widehat { \pmb { x } } _ { \omega } \gets \pmb { x } _ { t _ { K } }$

![](images/ebe61cf5f979a333bfde2eac5182d98619755324e6ff035f98ff6d0f33d09494.jpg)  
Fig. 5. Quarter-dose CT reconstructions obtained with different methods. Representative pelvic, thoracic, and abdominal slices are shown from top to bottom In each group, the upper row shows CT images, and the lower row shows difference maps relative to the full-dose reference. Reference denotes the full-dose reference image, LDCT denotes the quarter-dose input image, and Proposed denotes FFAI. The display window is [−260, 340] HU, and the color scale for the difference maps spans [−100, 100].

## A. Low-Dose CT Study

The low-dose experiments used the official quarter-dose images from the Mayo 2016 dataset, with the corresponding full-dose images as references. FFAI was compared with Noise2Noise [12], MTD-GAN [38], FoundDiff [26], TAMP [2], and FORCE [34], with FBP included as a baseline. Table I reports the mean and standard deviation of PSNR and SSIM for each method. FFAI achieved a mean PSNR of 48.30 dB and a mean SSIM of 97.96%, the highest values in this comparison. FFAI exceeded MTD-GAN, which ranked second in PSNR, by 2.27 dB and FoundDiff, which ranked second in SSIM, by 0.09 percentage points.

Fig. 5 shows representative pelvic, thoracic, and abdominal slices and their difference maps relative to the fulldose references. Noticeable granular residuals remained after Noise2Noise processing. Tissue contours were visible to varying degrees in the difference maps of FoundDiff, TAMP, and FORCE, indicating residual structure-related differences from the reference images. MTD-GAN and FFAI produced smaller differences overall, with FFAI more closely matching the references in the displayed regions. These results indicate that FFAI effectively suppresses noise while preserving anatomical structures under the evaluated low-dose setting.

## B. Limited-Angle CT Study

The limited-angle experiments considered two contiguous projection ranges of 60<sup>◦</sup> and 90<sup>◦</sup>, using the corresponding full-dose CT images as references. FFAI was compared with IRON [16], ProCT [17], TLIR [33], TAMP [2], and PWD [39], with FBP included as a baseline. Table II summarizes the quantitative results for both angular ranges. FFAI achieved mean PSNR and SSIM values of 38.94 dB and 94.28% at 60<sup>◦</sup>, and 40.92 dB and 95.97% at 90<sup>◦</sup>, respectively, attaining the highest means for both metrics in both settings. Compared with the second-best method, ProCT, FFAI improved PSNR by 3.96 dB and 2.09 dB and SSIM by 1.72 and 0.30 percentage points, respectively.

FBP  
TABLE II  
QUANTITATIVE RESULTS FOR 60<sup>◦</sup> AND 90<sup>◦</sup> LIMITED-ANGLE CT RECONSTRUCTION (MEAN ± STANDARD DEVIATION). SSIM IS REPORTED AS A PERCENTAGE. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST MEANS, RESPECTIVELY, FOR EACH ANGULAR RANGE.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $6 0 ^ { \circ }$ </td><td colspan="2"> $9 0 ^ { \circ }$ </td></tr><tr><td>PSNR (dB) ↑</td><td>SSIM (%) ↑</td><td>PSNR (dB) ↑</td><td>SSIM (%) ↑</td></tr><tr><td>FBP</td><td> $1 7 . 2 8 \pm 1 . 9 7$ </td><td> $5 3 . 5 5 \pm 3 . 0 6$ </td><td> $1 9 . 6 2 \pm 1 . 8 9$ </td><td> $6 4 . 0 9 \pm 4 . 0 5$ </td></tr><tr><td>IRON</td><td> $3 1 . 5 7 \pm 1 . 7 0$ </td><td> $9 0 . 3 6 \pm 3 . 2 6$ </td><td> $3 5 . 6 1 \pm 3 . 6 5$ </td><td> $9 4 . 6 0 \pm 2 . 6 6$ </td></tr><tr><td>ProCT</td><td> ${ \underline { { 3 4 . 9 8 \pm 1 . 5 9 } } }$ </td><td> $9 2 . 5 6 \pm 2 . 0 0$ </td><td> $3 8 . 8 3 \pm 1 . 9 0 $ </td><td> $9 5 . 6 7 \pm 1 . 5 2 $ </td></tr><tr><td>TLIR</td><td> $2 0 . 3 3 \pm 1 . 1 4$ </td><td> $5 4 . 5 7 \pm 1 2 . 0 4$ </td><td> $2 5 . 2 3 \pm 3 . 0 2$ </td><td> $4 6 . 0 9 \pm 3 . 5 9$ </td></tr><tr><td>TAMP</td><td> $2 0 . 7 9 \pm 1 . 7 5$ </td><td> $6 0 . 8 7 \pm 3 . 3 3$ </td><td> $2 4 . 5 7 \pm 3 . 0 1$ </td><td> $7 7 . 6 2 \pm 2 . 5 3 $ </td></tr><tr><td>PWD</td><td> $2 8 . 8 7 \pm 2 . 3 9$ </td><td> $7 4 . 0 9 \pm 4 . 8 9$ </td><td> $3 1 . 5 5 \pm 2 . 7 7$ </td><td> $8 2 . 8 4 \pm 8 . 2 6$ </td></tr><tr><td>FFAI</td><td> ${ \bf 3 8 . 9 4 \pm 1 . 7 9 }$ </td><td> ${ \pm } \mathbf { \ } 2 \mathbf { 8 } \pm 2 . \mathbf { 0 } 7$ </td><td> $\mathbf { 4 0 . 9 2 \pm 1 . 8 9 }$ </td><td> ${ \bf 9 5 . 9 7 \pm 1 . 5 7 }$ </td></tr></table>

ProCT  
TLIR  
TAMP  
PWD  
Proposed  
![](images/4515c9c85ff38832f3f0c22fae1edad1476e7d6020dff78e43405a86cfc8decf.jpg)  
Fig. 6. Reconstructions obtained with different methods under the 60<sup>◦</sup> limited-angle setting. The upper and lower groups show representative thoracic and abdominal slices, respectively. In each group, the upper row shows CT images, and the lower row shows difference maps relative to the reference. Reference denotes the full-dose reference image, and Proposed denotes FFAI. The display window is [−260, 340] HU, and the color scale for the difference maps spans [−200, 200].

## C. Sparse-View CT Study

Figs. 6 and 7 show representative thoracic and abdominal reconstructions and their difference maps. FBP images exhibited pronounced directional streaks and intensity distortions. Strong artifacts remained in some regions of the TAMP and PWD reconstructions, and the TLIR difference maps also showed pronounced structure-related residuals. IRON and ProCT improved the recovery of tissue contours, but distinct tissue boundaries remained visible in their difference maps. FFAI better preserved tissue morphology in the displayed slices and produced weaker structure-related residuals. Together with the quantitative results, these examples indicate that FFAI provides a favorable balance between artifact suppression and structural recovery under both limited-angle settings.

The sparse-view experiments used 40 and 80 projection views uniformly sampled over the full 360<sup>◦</sup> range, with the corresponding full-dose CT images as references. FFAI was compared with LipCT [18], MetaInv [40], DPS [32], TAMP [2], and FORCE [34], with FBP included as a baseline. Table III summarizes the quantitative results for both view settings. FFAI achieved mean PSNR and SSIM values of 39.08 dB and 90.55% with 40 views, and 41.67 dB and 94.68% with 80 views, respectively, attaining the highest means for both metrics. Compared with the second-best method, FORCE, FFAI improved PSNR by 1.53 dB and 1.81 dB and SSIM by 1.79 and 1.91 percentage points, respectively.

Reference

FBP  
IRON  
ProCT  
TLIR  
TAMP  
PWD  
Proposed  
![](images/9c1f3c096a37339f4377e0a5df7ed50a2fa3f5bb3016bd18e4df028d0b0cb3ff.jpg)  
Fig. 7. Reconstructions obtained with different methods under the 90<sup>◦</sup> limited-angle setting. The upper and lower groups show representative thoracic and abdominal slices, respectively. In each group, the upper row shows CT images, and the lower row shows difference maps relative to the reference. Reference denotes the full-dose reference image, and Proposed denotes FFAI. The display window is [−260, 340] HU, and the color scale for the difference maps spans [−200, 200].

TABLE III  
QUANTITATIVE RESULTS FOR SPARSE-VIEW CT RECONSTRUCTION WITH 40 AND 80 VIEWS (MEAN ± STANDARD DEVIATION). SSIM IS REPORTED AS A PERCENTAGE. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST MEANS, RESPECTIVELY, FOR EACH VIEW SETTING.
<table><tr><td rowspan="2">Method</td><td colspan="2">40 views</td><td colspan="2">80 views</td></tr><tr><td>PSNR (dB) ↑</td><td>SSIM (%) ↑</td><td>PSNR (dB) ↑</td><td>SSIM (%) ↑</td></tr><tr><td>FBP</td><td> $2 6 . 6 5 \pm 1 . 0 7$ </td><td> $4 8 . 5 2 \pm 2 . 8 2$ </td><td> $3 2 . 2 5 \pm 0 . 7 9$ </td><td> $7 3 . 0 7 \pm 1 . 6 2$ </td></tr><tr><td>LipCT</td><td> $3 7 . 2 9 \pm 0 . 5 6$ </td><td> $8 7 . 2 7 \pm 0 . 9 7$ </td><td> $3 9 . 2 5 \pm 0 . 2 1$ </td><td> $9 0 . 7 2 \pm 0 . 8 6$ </td></tr><tr><td>MetaInv</td><td> $3 5 . 5 6 \pm 1 . 7 6$ </td><td> $8 4 . 5 0 \pm 2 . 3 4$ </td><td> $3 7 . 0 8 \pm 1 . 1 4$ </td><td> $8 8 . 0 8 \pm 0 . 9 7$ </td></tr><tr><td>DPS</td><td> $3 2 . 3 3 \pm 0 . 4 0$ </td><td> $7 4 . 7 5 \pm 1 . 1 4$ </td><td> $3 2 . 4 2 \pm 0 . 4 3$ </td><td> $7 5 . 0 4 \pm 0 . 9 6$ </td></tr><tr><td>TAMP</td><td> $3 0 . 2 5 \pm 0 . 6 5$ </td><td> $8 0 . 5 5 \pm 1 . 0 7$ </td><td> $3 6 . 0 0 \pm 0 . 4 9$ </td><td> $8 9 . 1 1 \pm 0 . 9 5$ </td></tr><tr><td>FORCE</td><td> $\underline { { 3 7 . 5 5 \pm 0 . 6 4 } }$ </td><td> $8 8 . 7 6 \pm 0 . 9 4$ </td><td> $\underline { { 3 9 . 8 6 \pm 0 . 3 5 } }$ </td><td> $9 2 . 7 7 \pm 0 . 7 8$ </td></tr><tr><td>FFAI</td><td> ${ \bf 3 9 . 0 8 \pm 0 . 3 1 }$ </td><td> ${ \bf 9 0 . 5 5 \pm 1 . 1 8 }$ </td><td> ${ \bf 4 1 . 6 7 \pm 0 . 3 6 }$ </td><td> ${ \pm } 4 . 6 8 \pm 1 . 0 6$ </td></tr></table>

Figs. 8 and 9 show representative thoracic and abdominal reconstructions and their difference maps. Pronounced streak artifacts were present in the FBP images. LipCT and TAMP suppressed many streaks, but some tissue boundaries and internal details remained blurred. Local structural distortions or granular residuals were still visible in the MetaInv and DPS reconstructions. FORCE recovered more tissue details, although tissue contours remained visible in its difference maps. FFAI better preserved tissue boundaries and internal structures in the displayed slices and produced weaker structure-related residuals in the difference maps. Together with the quantitative results, these examples indicate that FFAI provides a favorable balance between streak suppression and structural recovery under both sparse-view settings.

## D. Ablation Study

To assess the contributions of error correction and adaptive initialization, we evaluated two ablation variants: removing feedback-aware training (W/o FAT) and replacing the adaptive starting point with a fixed one (W/o AS). FAT denotes the feedback-aware mechanism used during training, whereas AS selects the inference starting point based on degradation severity. Experiments were conducted under quarter-dose, 60<sup>◦</sup> limited-angle, and 40-view sparse-view settings. Table IV presents the quantitative results.

Removing FAT reduced the mean PSNR by 6.86, 9.97, and 4.21 dB for the low-dose, limited-angle, and sparse-view tasks, respectively, indicating that feedback-aware training improves reconstruction quality across all three tasks. Replacing the adaptive starting point with a fixed one reduced the mean PSNR by 17.59, 6.25, and 3.66 dB, respectively, showing that degradation-adaptive initialization outperforms the fixed starting configuration used here. The full model achieved the highest mean PSNR and SSIM in all three tasks.

![](images/837a7e66d10a32739550d525a36637bd9539da49e0ea7414350b4b83a1f1307d.jpg)  
Fig. 9. CT reconstructions obtained with different methods under the 80-view sparse-sampling setting. The upper and lower groups show representative thoracic and abdominal slices, respectively. In each group, the upper row shows CT images, and the lower row shows difference maps relative to the reference. Reference denotes the full-dose reference image, and Proposed denotes FFAI. The display window is [−260, 340] HU, and the color scale for the difference maps spans [−200, 200].

Fig. 10 shows the reconstructions and difference maps for each variant. Without FAT, appreciable noise or structurerelated residuals remained. With a fixed starting point, tissue contours remained visible in the difference maps. In comparison, the full model produced smaller differences overall and better preserved tissue boundaries and internal structures in the displayed slices.

TABLE IV  
ABLATION RESULTS FOR QUARTER-DOSE, 60<sup>◦</sup> LIMITED-ANGLE, AND 40-VIEW SPARSE-VIEW CT RECONSTRUCTION (MEAN ± STANDARD DEVIATION). FAT AND AS DENOTE FEEDBACK-AWARE TRAINING AND AN ADAPTIVE STARTING POINT, RESPECTIVELY. ✓ AND × INDICATE WHETHER EACH COMPONENT IS ENABLED OR REMOVED; REMOVING AS CORRESPONDS TO USING A FIXED STARTING POINT. SSIM IS REPORTED AS A PERCENTAGE, AND BOLDFACE INDICATES THE BEST MEAN VALUES.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">FAT</td><td rowspan="2">AS</td><td colspan="2">LDCT (25%)</td><td colspan="2">LACT  $( 6 0 ^ { \circ } )$ </td><td colspan="2">SVCT (40 views)</td></tr><tr><td></td><td>PSNR (dB) ↑ SSIM (%) ↑</td><td>PSNR (dB) ↑S</td><td>SSIM (%) ↑</td><td>PSNR (dB) ↑ SSIM (%) ↑</td><td></td></tr><tr><td>W/o FAT</td><td>X</td><td>√</td><td> $4 1 . 4 4 \pm 1 . 6 0$ </td><td> $9 3 . 9 1 \pm 2 . 1 7$ </td><td> $2 8 . 9 7 \pm 1 . 8 0 $ </td><td> $7 4 . 3 5 \pm 2 . 9 8$ </td><td> $3 4 . 8 7 \pm 0 . 6 8$ </td><td> $7 9 . 8 8 \pm 1 . 2 0 $ </td></tr><tr><td>W/o AS</td><td>√</td><td>X</td><td> $3 0 . 7 1 \pm 1 . 7 7$ </td><td> $8 1 . 8 5 \pm 2 . 2 0$ </td><td> $3 2 . 6 9 \pm 0 . 9 4$ </td><td> $8 2 . 4 4 \pm 1 . 3 9$ </td><td> $3 5 . 4 2 \pm 0 . 6 3$ </td><td> $8 1 . 1 2 \pm { 1 . 0 1 }$ </td></tr><tr><td>FFAI</td><td>√</td><td>√</td><td> ${ \bf 4 8 . 3 0 \pm 0 . 0 5 }$ </td><td> ${ \bf 9 7 . 9 6 \pm 0 . 3 9 }$ </td><td> ${ \bf 3 8 . 9 4 \pm 1 . 7 9 }$ </td><td> ${ \pm } \mathbf { \ } 2 \mathbf { 8 } \pm 2 . \mathbf { 0 } 7$ </td><td> ${ \bf 3 9 . 0 8 \pm 0 . 3 1 }$ </td><td> ${ \bf 9 0 . 5 5 \pm 1 . 1 8 }$ </td></tr></table>

![](images/a99f0ee88ab6c26608e92aed207b9c6659203b0aa6c6ef4c26f42636c2f1f47b.jpg)  
Fig. 10. CT reconstructions obtained with different ablation variants. From top to bottom, the three groups correspond to quarter-dose, 60<sup>◦</sup> limited-angle, and 40-view sparse-view experiments. In each group, the upper row shows the reconstructions and the lower row shows their differences from the reference images. W/o FAT and W/o AS denote the variants without feedback-aware training and with a fixed starting point, respectively. Proposed denotes the full FFAI model. The color scale for the difference maps is [−200, 200].

## E. Generalizability Study

To assess cross-dataset generalizability, we evaluated the proposed method on the CT-ORG [41] and LDCT-and-Projection-data [42] datasets provided by TCIA. The sparseview and limited-angle experiments used CT-ORG, which includes abdominal and whole-body scans acquired under varying conditions, including contrast-enhanced and noncontrast examinations. We selected one slice from each of 140 CT scans to form a shared reference image set for sparseview reconstruction with 80 projection views and limitedangle reconstruction over a 90<sup>◦</sup> angular range. The lowdose experiments used 20 abdominal cases from LDCT-and-Projection-data. Five image pairs were selected per case by stratified sampling along the slice axis, yielding 100 pairs in total. The inputs were the dataset-provided reconstructions simulated at 25% of routine dose by projection-domain noise insertion, and the references were the corresponding routinedose images. All images were standardized to 512×512 pixels, and reconstruction quality was evaluated using PSNR and SSIM.

As shown in Fig. 11, the proposed method achieved the highest PSNR and SSIM in all three tasks. For low-dose reconstruction, it achieved 51.89 dB and 99.31%, exceeding MTD-GAN, the best-performing comparison method, by 1.39 dB and 0.26 percentage points, respectively. For sparse-view reconstruction, it achieved 43.73 dB and 97.10%, exceeding FORCE by 1.21 dB and 0.63 percentage points, respectively.

The advantage was more pronounced for limited-angle reconstruction. The proposed method achieved a PSNR of 35.48 dB and an SSIM of 91.12%. Its PSNR exceeded that of the second-best method, TLIR, by 7.41 dB, while its SSIM exceeded that of the second-best method, TAMP, by 28.25 percentage points. These results show that the proposed method retained its reconstruction advantage on the additional test datasets, supporting its generalizability across the three CT reconstruction tasks.

## IV. DISCUSSION AND CONCLUSION

We proposed FFAI, a conditional flow matching method for low-dose, sparse-view, and limited-angle CT reconstruction. Conditioned on an initial degraded reconstruction, FFAI incorporates teacher-generated feedback states during training to learn to recover high-quality images from states shifted by prediction errors. During inference, it selects the starting time based on input degradation severity and constructs the corresponding initial state. By combining feedback-aware training with degradation-adaptive initialization, FFAI provides a common modeling and reconstruction procedure for all three tasks.

![](images/7275fd5c56f3532445205fe2cb1812a9e805fddb9dcb70a3f1a30469cb4b5c19.jpg)  
Fig. 11. Quantitative comparison of cross-dataset generalizability. Low-dose CT (LDCT) uses 25% of routine dose, sparse-view CT (SVCT) uses 80 projection views, and limited-angle CT (LACT) uses a $9 0 ^ { \circ }$ angular range. Circles and squares indicate PSNR (dB) and SSIM (%), respectively. Proposed denotes the proposed method.

Experiments on public benchmark datasets showed that FFAI outperformed the comparison methods quantitatively across reconstruction tasks while providing good visual quality. It effectively suppressed noise and artifacts while preserving tissue structures and details, demonstrating its ability to handle different types of degradation.

Despite these results, FFAI has several limitations. It currently requires task-specific settings for some inference parameters and a separately trained model for each task. The present study also focuses primarily on two-dimensional reconstruction, and applicability to a broader range of imaging conditions requires further validation. Future work may address these limitations in the following directions:

1) Automated inference configuration. Acquisition information and image features could guide the joint selection of the starting time, noise scale, and number of inference steps. This may reduce manual task-specific configuration and enable more flexible tradeoffs between reconstruction quality and computational cost.

2) Cross-task parameter sharing. A common representation of degradation conditions could be incorporated into the framework to support joint training and parameter sharing across tasks. Exploiting commonalities among reconstruction tasks may reduce the cost of training and maintaining separate models and facilitate adaptation to combined degradations.

3) Extension to three-dimensional reconstruction. Extending the method to three-dimensional reconstruction could exploit structural information from adjacent slices to improve spatial continuity in volumetric images. Efficient network architectures and inference strategies could also help control the GPU memory requirements and computational costs of volumetric reconstruction.

4) Adaptation to different imaging conditions. Data from more scanners, acquisition protocols, and anatomical regions could support further investigation of adaptability to different imaging conditions. Multicenter validation, reader studies, and downstream analysis tasks could provide a more comprehensive assessment of reconstruction quality and practical utility.

## REFERENCES

[1] R. Smith-Bindman, P. W. Chu, H. A. Firdaus, C. Stewart, M. Malekhedayat, S. Alber, W. E. Bolch, M. Mahendra, A. B. de Gonzalez, and´ D. L. Miglioretti, “Projected lifetime cancer risks from current computed tomography imaging,” JAMA Intern. Med., vol. 185, no. 6, pp. 710–719, 2025.

[2] R. Ge, Y. Liu, Z. Wu, S. Yang, Y. Gao, C. You, G. Wang, S. Li, Y. He, and Y. Chen, “Imaging foundation model for universal enhancement of non-ideal measurement CT,” Nature Communications, vol. 17, p. 7155, 2026. [Online]. Available: https://doi.org/10.1038/s41467-026-73943-y

[3] I. A. Elbakri and J. A. Fessler, “Statistical image reconstruction for polyenergetic x-ray computed tomography,” IEEE Trans. Med. Imaging, vol. 21, no. 2, pp. 89–99, 2002.

[4] J. Wang, T. Li, H. Lu, and Z. Liang, “Penalized weighted least-squares approach to sinogram noise reduction and image reconstruction for lowdose x-ray computed tomography,” IEEE Trans. Med. Imaging, vol. 25, no. 10, pp. 1272–1283, 2006.

[5] E. Y. Sidky and X. Pan, “Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization,” Phys. Med. Biol., vol. 53, no. 17, pp. 4777–4807, 2008.

[6] G.-H. Chen, J. Tang, and S. Leng, “Prior image constrained compressed sensing (PICCS): A method to accurately reconstruct dynamic CT images from highly undersampled projection data sets,” Med. Phys., vol. 35, no. 2, pp. 660–663, 2008.

[7] X. Yang, Y. Long, and S. Ravishankar, “Multilayer residual sparsifying transform (MARS) model for low-dose CT image reconstruction,” Med. Phys., vol. 48, no. 10, pp. 6388–6400, 2021.

[8] L. Chen, X. Yang, Z. Huang, Y. Long, and S. Ravishankar, “Multi-layer clustering-based residual sparsifying transform for low-dose CT image reconstruction,” Med. Phys., vol. 50, no. 10, pp. 6096–6117, 2023.

[9] J. Shen, M. Luo, H. Liu, P. Liao, H. Chen, and Y. Zhang, “MLF-IOSC: Multi-level fusion network with independent operation search cell for low-dose CT denoising,” IEEE Trans. Med. Imaging, vol. 42, no. 4, pp. 1145–1158, 2023.

[10] L. Yang, Z. Li, R. Ge, J. Zhao, H. Si, and D. Zhang, “Low-dose CT denoising via sinogram inner-structure transformer,” IEEE Trans. Med. Imaging, vol. 42, no. 4, pp. 910–921, 2023.

[11] R. Li, Q. Li, H. Wang, S. Li, J. Zhao, Y. Qiang, and L. Wang, “DDPTransformer: Dual-domain with parallel transformer network for sparse view CT image reconstruction,” IEEE Trans. Comput. Imaging, vol. 8, pp. 1101–1116, 2022.

[12] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, “Noise2Noise: Learning image restoration without clean data,” in Proc. Int. Conf. Mach. Learn., ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 2965–2974. [Online]. Available: https://proceedings.mlr.press/v80/lehtinen18a.html

[13] S. Ye, Z. Li, M. T. McCann, Y. Long, and S. Ravishankar, “Unified supervised–unsupervised (SUPER) learning for x-ray CT image reconstruction,” IEEE Trans. Med. Imaging, vol. 40, no. 11, pp. 2986–3001, 2021.

[14] W. Wu, D. Hu, C. Niu, H. Yu, V. Vardhanabhuti, and G. Wang, “DRONE: Dual-domain residual-based optimization network for sparse-view CT reconstruction,” IEEE Trans. Med. Imaging, vol. 40, no. 11, pp. 3002– 3014, 2021.

[15] D. Hu, Y. Zhang, J. Liu, S. Luo, and Y. Chen, “DIOR: Deep iterative optimization-based residual-learning for limited-angle CT reconstruction,” IEEE Trans. Med. Imaging, vol. 41, no. 7, pp. 1778–1790, 2022.

[16] J. Pan, H. Yu, Z. Gao, S. Wang, H. Zhang, and W. Wu, “Iterative residual optimization network for limited-angle tomographic reconstruction,” IEEE Trans. Image Process., vol. 33, pp. 910–925, 2024.

[17] C. Ma, Z. Li, J. He, J. Zhang, Y. Zhang, and H. Shan, “Universal pretraining for generalizable incomplete-view CT reconstruction,” Pattern Recognit., vol. 178, p. 113513, 2026.

[18] B. Shi, K. Jiang, Q. Lian, X. Yu, and H. Fu, “Prompting lipschitzconstrained network for multiple-in-one sparse-view CT reconstruction,” IEEE Trans. Med. Imaging, vol. 45, no. 4, pp. 1446–1460, 2026.

[19] Y. Zhang, D. Hu, Q. Zhao, G. Quan, J. Liu, Q. Liu, Y. Zhang, G. Coatrieux, Y. Chen, and H. Yu, “CLEAR: Comprehensive learning enabled adversarial reconstruction for subtle structure enhanced lowdose CT imaging,” IEEE Trans. Med. Imaging, vol. 40, no. 11, pp. 3089–3101, 2021.

[20] Y. Fu, S. Dong, M. Niu, L. Xue, H. Guo, Y. Huang, Y. Xu, T. Yu, K. Shi, Q. Yang, Y. Shi, H. Zhang, M. Tian, and C. Zhuo, “AIGAN: Attentionencoding integrated generative adversarial network for the reconstruction of low-dose CT and low-dose PET images,” Med. Image Anal., vol. 86, p. 102787, 2023.

[21] F. Liao, Y. Tang, Q. Du, J. Wang, M. Li, and J. Zheng, “Domain progressive low-dose CT imaging using iterative partial diffusion model,” IEEE Trans. Med. Imaging, vol. 44, no. 9, pp. 3610–3628, 2025.

[22] D. Hein, G. Stevens, A. Wang, and G. Wang, “PFCM: Poisson flow consistency models for low-dose CT image denoising,” IEEE Trans. Med. Imaging, vol. 44, no. 7, pp. 2989–3001, 2025.

[23] K. Xu, S. Lu, B. Huang, W. Wu, and Q. Liu, “Stage-by-stage wavelet optimization refinement diffusion model for sparse-view CT reconstruction,” IEEE Trans. Med. Imaging, vol. 43, no. 10, pp. 3412–3424, 2024.

[24] L. Yang, J. Huang, G. Yang, and D. Zhang, “CT-SDM: A sampling diffusion model for sparse-view CT reconstruction across various sampling rates,” IEEE Trans. Med. Imaging, vol. 44, no. 6, pp. 2581–2593, 2025.

[25] J. Zhang, H. Mao, X. Wang, Y. Guo, and W. Wu, “Wavelet-inspired multi-channel score-based model for limited-angle CT reconstruction,” IEEE Trans. Med. Imaging, vol. 43, no. 10, pp. 3436–3448, 2024.

[26] Z. Chen, Q. Gao, Z. Li, J. Zhang, Y. Zhang, J. Zhao, and H. Shan, “FoundDiff: Foundational diffusion model for generalizable low-dose CT denoising,” IEEE Trans. Med. Imaging, vol. 45, no. 8, pp. 4366– 4379, 2026.

[27] S. Bhadra, V. A. Kelkar, F. J. Brooks, and M. A. Anastasio, “On hallucinations in tomographic image reconstruction,” IEEE Trans. Med. Imaging, vol. 40, no. 11, pp. 3249–3260, 2021.

[28] Q. Gao, Z. Li, J. Zhang, Y. Zhang, and H. Shan, “CoreDiff: Contextual error-modulated generalized diffusion model for low-dose CT denoising and generalization,” IEEE Trans. Med. Imaging, vol. 43, no. 2, pp. 745– 759, 2024.

[29] L. He, W. Du, P. Liao, F. Fan, H. Chen, H. Yang, and Y. Zhang, “Solving zero-shot sparse-view CT reconstruction with variational score solver,” IEEE Trans. Med. Imaging, vol. 44, no. 9, pp. 3586–3599, 2025.

[30] W. Wu, J. Pan, Y. Wang, S. Wang, and J. Zhang, “Multi-channel optimization generative model for stable ultra-sparse-view CT reconstruction,” IEEE Trans. Med. Imaging, vol. 43, no. 10, pp. 3461–3475, 2024.

[31] S. Han, Y. Xu, D. Wang, B. Morovati, L. Zhou, J. S. Maltz, G. Wang, and H. Yu, “Physics-informed score-based diffusion model for limitedangle reconstruction of cardiac computed tomography,” IEEE Trans. Med. Imaging, vol. 44, no. 9, pp. 3629–3640, 2025.

[32] H. Chung, J. Kim, M. T. McCann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” in Proc. Int. Conf. Learn. Represent., 2023. [Online]. Available: https://openreview.net/forum?id=OnD9zGAGT0k

[33] Q. Li, T. Wang, R. Li, Y. Qiang, B. Zhang, J. Sun, J. Zhao, and W. Wu, “TLIR: Two-layer iterative refinement model for limited-angle CT reconstruction,” Biomed. Signal Process. Control, vol. 100, p. 107058, 2025.

[34] W. Xia, C. Niu, and G. Wang, “Tomographic foundation model— FORCE: Flow-oriented reconstruction conditioning engine,” IEEE Trans. Med. Imaging, vol. 45, no. 6, pp. 3137–3149, 2026.

[35] H. Chen, Z. Hao, L. Guo, and L. Xiao, “Mitigating data consistency induced discrepancy in cascaded diffusion models for sparse-view CT reconstruction,” IEEE Trans. Med. Imaging, vol. 44, no. 7, pp. 3012– 3024, 2025.

[36] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Represent., 2023.

[37] M. Pourya, B. E. Rawas, and M. Unser, “FLOWER: A flow-matching solver for inverse problems,” in Proc. Int. Conf. Learn. Represent., 2026.

[38] S. Kyung, J. Won, S. Pak, S. Kim, S. Lee, K. Park, G.-S. Hong, and N. Kim, “Generative adversarial network with robust discriminator through multi-task learning for low-dose CT denoising,” IEEE Trans. Med. Imaging, vol. 44, no. 1, pp. 499–518, 2025.

[39] Y. Liu, Y. Wen, Z. Zhou, J. Ma, L. Wang, Y. Yao, L. Shi, and Q. Liu, “PWD: Prior-aware wavelet diffusion for efficient dental limited-angle

CT reconstruction,” IEEE Trans. Med. Imaging, vol. 45, no. 6, pp. 2994– 3006, 2026.

[40] H. Zhang, B. Liu, H. Yu, and B. Dong, “MetaInv-Net: Meta inversion network for sparse view CT image reconstruction,” IEEE Trans. Med. Imaging, vol. 40, no. 2, pp. 621–634, 2021.

[41] B. Rister, D. Yi, K. Shivakumar, T. Nobashi, and D. L. Rubin, “CT-ORG, a new dataset for multiple organ segmentation in computed tomography,” Scientific Data, vol. 7, no. 1, p. 381, 2020.

[42] T. R. Moen, B. Chen, D. R. Holmes, III, X. Duan, Z. Yu, L. Yu, S. Leng, J. G. Fletcher, and C. H. McCollough, “Low-dose CT image and projection dataset,” Medical Physics, vol. 48, no. 2, pp. 902–911, 2021.