# From Generic to Dedicated: A Novel Optimizer for Online Continual Learning

Yongyi Wu University of Houston Houston, USA yongyiwu.cn@gmail.com

Zheng Wang University of Houston Houston, USA zwang214@cougarnet.uh.edu

Sen Lin<sup>†</sup> University of Houston Houston, USA slin50@central.uh.edu

## Abstract

Online Continual Learning (OCL) requires models to learn from non-stationary data streams under a strict single-pass constraint, making them highly susceptible to catastrophic forgetting. While existing studies have explored various strategies from data, architecture, and optimization perspectives, the optimizer often directly uses generic approaches such as SGD and AdamW. In this work, we reveal a new link between optimization dynamics and OCL needs by recasting Polyak-Ruppert averaging as the engine of “plasticity” and Primal averaging as the anchor of “stability”. This inspires our novel optimizer, SPIN (Stability-Plasticity INterpolation), which explicitly decouples plasticity and stability by interpolating between a fast-moving “plasticity” sequence and an adaptive “stability” sequence. We theoretically demonstrate that SPIN implicitly employs an inverse Hessian approximation, providing crucial Tikhonov regularization to damp noisy, singlesample Hessian estimates and mitigate forgetting. More importantly, SPIN can be seamlessly integrated into existing OCL methods by taking resulted gradients as input and replacing standard optimizers. Extensive experiments show that using SPIN instead of standard optimizers yields consistent and significant performance gains across multiple OCL approaches, including the challenging rehearsal-free setting and under strict memory constraints.

## 1 Introduction

Continual Learning (CL) aims to enable AI systems to continuously acquire new knowledge while retaining previously learned skills, a capability essential for many real-world applications where data arrives sequentially [62, 76, 59, 9]. However, the common assumption in offline CL, that models are able to revisit task data multiple times during training, rarely holds. This has motivated the rise of Online Continual Learning (OCL), a more realistic paradigm where each sample is observed only once [21, 58, 41]. Nevertheless, OCL inevitably exacerbates key challenges like catastrophic forgetting [16, 27] and the plasticity-stability dilemma [43] under strict single-pass constraints.

To address this, existing OCL research has largely followed the classical offline CL paradigm, focusing on data, loss functions, and architectural designs, such as replay, regularization, and distillation [50, 7, 24, 74, 71]. In most of these approaches, the lower-level optimizer is treated as a fixed, secondary component, directly using generic strategies such as vanilla SGD [51] or AdamW [26, 39]. In both SGD and AdamW, the gradient is evaluated at the current parameter given the newly arrived sample. Consequently, the optimization direction is dominated by the instantaneous data distribution, even when momentum is used. This behavior can lead to accelerated forgetting under non-stationary data streams. From the perspective of online optimization theory, however, the optimizer itself plays a decisive role in determining convergence, stability, and generalization [23]. Many works have explored the role of optimization in OCL, proposing methods that modify gradient updates via projection, reweighting, proximal constraints, or curvature-aware preconditioning [38, 8, 69, 73]. These approaches operate primarily at the level of gradient manipulation, adapting the update direction while still relying on standard optimizers. Therefore, a critical question arises: “Instead ofrelying on generic optimizers, can we design a novel optimizer specifically tailored to OCL that effectively mitigates theforgetting problem andfurther pushes the limits ofexisting OCL approaches?”

To answer this question, we revisit the fundamentals of stochastic optimization and establish a direct connection between its averaging mechanisms and the stability–plasticity dilemma in OCL. Specifically, we reinterpret Polyak–Ruppert averaging [48, 52] as the engine of plasticity, since it evaluates gradients at the current, fast-evolving iterate and thus quickly adapts to new data. In contrast, Primal averaging [44, 10, 13] serves as the anchor of stability, as it computes gradients at the averaged, slowly updating point, thereby retaining previously acquired knowledge. We empirically examine these two averaging schemes, as shown in Figure 1. Polyak–Ruppert averaging achieves higher learning accuracy but suffers from greater forgetting, reflecting its strong plasticity yet limited stability. Primal averag ing, on the other hand, exhibits the opposite behavior, showing lower plasticity while maintaining superior stability.

![](images/eca2d4ea67f82bdcd97f0fcb0bc935945deea2ff418fd716ba50eaa78f466fb8.jpg)

![](images/0a8b7a7e1febbbcc164b2e3066beec5a3db1ab1326b902e762c3fd2c3b9e8566.jpg)  
Figure 1: The comparison of plasticity (Learning Accuracy) and stability (Average Forgetting) of Polyak-Ruppert averaging and Primal averaging on split CIFAR-100. Learning Accuracy for the j-th task is the accuracy evaluated on the test set of task j immediately after the network has been trained sequentially from task 1 up to task j.

Driven by these observations, we propose a novel optimizer, SPIN (Stability-Plasticity INterpolation), which explicitly manages this trade-off by operating on two coupled iterate sequences. SPIN maintains a plasticity model sequence for rapid adaptation and a stability model sequence for consolidated knowledge. Its core mechanism links these sequences in two ways. First, gradients are evaluated at an interpolated point between the two models, enabling a unified view of multiple averaging strategies. Second, the stability sequence is updated by smoothly integrating information from the plastic sequence, preventing corruption under the high-variance, non-stationary OCL stream. We find that this formulation not only blends the benefits of different averaging approaches, but also connects conceptually to several regularization-based and meta-continual learning methods. Our main contributions can be summarized as follows:

(1) We propose a novel optimizer dedicated for OCL, namely SPIN, which explicitly decouples plasticity and stability at the model level. Crucially, SPIN can be directly integrated into existing OCL methods by simply replacing their lower-level optimizers. In other words, SPIN operates similarly to SGD by taking only the gradients as input, regardless ofhow those gradients are modified by the upstream OCL methods.

(2) We establish a unified optimization framework for OCL, providing new theoretical insights into the plasticity-stability dilemma. Within this framework, we theoretically justify the design of SPIN and its effectiveness in mitigating catastrophic forgetting.

(3) Extensive experiments validate the effectiveness and generality of SPIN, showing consistent, significant performance gains across diverse OCL approaches, including the challenging rehearsalfree setting and under strict memory constraints.

## 2 Related Work

Continual Learning. CL aims to enable models to learn from a sequence of tasks without forgetting previous knowledge [47, 76, 35]. Conventionally, CL strategies primarily fall into three categories: Memory-based methods revisit past knowledge, including replay-based methods that retrain on stored old data [50, 6, 65, 14] and orthogonal projection-based methods that project gradients to prevent interference [69, 34, 8]; Regularization-based methods, which add penalties to the loss function to prevent significant changes to weights deemed important for old tasks [60, 27, 77]; and Parameter Isolation methods, which use dedicated network parameters for different tasks, either by expanding the architecture or by isolating pathways within a fixed network [70, 61, 33, 72, 32]. CLS-ER combines short- and long-term semantic memories with experience replay to support adaptation and knowledge retention [2]. Parameter averaging offers another approach: CoMA combines model parameters from previous and current tasks to balance stability and plasticity, while CoFiMA further weights parameters using Fisher information [42]. While optimization-oriented methods exist in offline CL (e.g., [66, 34, 54]), their fundamental reliance on accessing past data makes them incompatible with the single-pass constraints of OCL.

Online Continual Learning. OCL requiress models to learn from a strict single-pass data stream [23]. Replay-based strategies dominate this space [57], ranging from simple buffer combinations (ER [50], DER [6]) to advanced objective modifications addressing representation drift (ER-ACE, OCM, GSA, OnPro [7, 17, 67, 18]). Recent methods further boost adaptability via multi-level supervision (MOSE [71]) or advanced state-space and collaborative modeling (CCL-DC [63], S6MOD [37]). CBA addresses task-recency bias under distribution shifts through a continual bias adaptor [64]. A-GEM constrains the current gradient to avoid increasing the loss on past data by projecting it onto the feasible region defined by reference gradients from memory [8]. LPR applies a layer-wise proximal preconditioning that stabilizes optimization by limiting abrupt changes to past representations while learning from both new and replay data [73]. In contrast, our work focuses on the lower-level optimizer design that takes upper-level gradients as input, and provides a complementary and orthogonal perspective that can be applied on top ofdiverse OCL methods.

Averaging-Based Optimization. Classical optimization methods use coupled iterate sequences and averaging to accelerate convergence or reduce variance, as exemplified by Nesterov’s accelerated methods and Polyak–Ruppert averaging [45, 46, 48, 52]. Further analyses establish connections between iterate averaging, momentum, and convergence guarantees [56, 11]. In deep learning, stochastic weight averaging (SWA) improves generalization by averaging weights along the training trajectory [25], while Lookahead periodically interpolates and synchronizes fast and slow weights [75]. Schedule-Free optimization combines interpolated gradient evaluation with cumulative iterate averaging to avoid a prespecified training horizon [12], with nonconvex convergence guarantees established for its SGD variant [1]. Generalized Primal Averaging (GPA) uses independent coefficients for gradient evaluation and iterate averaging to smooth DiLoCo updates and improve LLM training efficiency [13]. SPIN shares this coupled interpolation-and-averaging structure. Our focus is on how the gradient evaluation point and averaging dynamics jointly influence the stability–plasticity tradeoff in non-stationary OCL, where each update must accommodate incoming data while preserving knowledge of previously encountered tasks.

## 3 Preliminaries

Online Continual Learning Setting. Consider the standard OCL setting, where a model $f _ { \theta }$ with parameters θ learns from a non-stationary data stream of $T$ tasks [21, 58, 41]. The dataset for the t-th task is denoted as $\mathcal { D } _ { t } = \{ ( v _ { i } , y _ { i } ) \}$ with input-label pairs, where $\mathcal { C } _ { t }$ is the set of classes for the t-th task. Let $\textstyle { \mathcal { D } } = \bigcup _ { t = 1 } ^ { T } { \mathcal { D } } _ { t }$ . Following the common protocol that $\mathcal { C } _ { t } \cap \mathcal { C } _ { t ^ { \prime } } = \emptyset$ for $t \neq t ^ { \prime } ,$ , the model must infer the correct class from the entire set of seen classes $\textstyle { \mathcal { Y } } = \bigcup _ { i = 1 } ^ { T } { \mathcal { C } } _ { i }$ at test time [57]. The model processes this data stream sequentially online, where data from each task is provided as a stream of mini-batches. We denote a single mini-batch from the current stream as $\dot { B _ { \mathrm { s t r e a m } } } = \{ ( v _ { i } , y _ { i } ) \} _ { i = 1 } ^ { B } ,$ with batch size B. Crucially, OCL imposes a strict single-pass constraint: the model observes each training sample $( v , y )$ exactly once and cannot re-access past task data (unless stored in a buffer). This constraint, combined with data distribution shift, leads to catastrophic forgetting [16], where the model’s performance on previous tasks degrades significantly as it learns new ones. To mitigate forgetting, many OCL methods employ experience replay [50]. A small memory buffer M with a fixed capacity M is maintained to store a subset of past samples. At each training step, the model trains on the union of the current stream batch and a replay batch drawn from $\mathcal { M }$ . The goal is to learn parameters $\theta _ { T }$ at the end of the training stream that minimizes the empirical risk across all encountered tasks: min<sub>θ</sub> $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { ( v , y ) \sim \mathcal { D } _ { t } } \left[ \mathcal { L } ( f ( v ; \theta _ { T } ) , y ) \right] } \end{array}$ ], where $\mathcal { L }$ is the loss function.

Averaging-Based Methods in Optimization. Nesterov’s accelerated method is an optimization algorithm designed to achieve faster convergence rates for L-smooth convex functions [45, 46]. One common form, known as AC-SA [29], utilizes an interpolation between two sequences of iterates:

$$
x _ { t } = ( 1 - c _ { t + 1 } ) \theta _ { t } + c _ { t + 1 } \phi _ { t } ,\tag{1}
$$

$$
\phi _ { t + 1 } = \phi _ { t } - \frac { t + 1 } { 2 L } \nabla \mathcal { L } ( x _ { t } ) ,\tag{2}
$$

$$
\theta _ { t + 1 } = ( 1 - c _ { t + 1 } ) \theta _ { t } + c _ { t + 1 } \phi _ { t + 1 } .\tag{3}
$$

Here, $\phi _ { t }$ is the base sequence, $\theta _ { t }$ is the averaged sequence, and $x _ { t }$ is the point at which the gradient is evaluated. The coefficient $c _ { t + 1 }$ is typically set as $c _ { t + 1 } = 2 / ( t + 2 )$ for non-strongly convex functions.

Gradient-Space Interpolation in OCL. Many replay-based OCL methods can be written as a gradient-space interpolation between the current task and past tasks [69]. Let ${ \boldsymbol g } _ { t } ^ { \mathrm { c u r } }$ denote the gradient on the current sample at step $t ,$ and $\{ g _ { t } ^ { ( i ) } \} _ { i < t }$ the gradients estimated from the replay buffer for previous tasks. A generic form is

$$
\begin{array} { r } { u _ { t } = g _ { t } ^ { \mathrm { c u r } } + \sum _ { i < t } \lambda _ { i } g _ { t } ^ { ( i ) } , } \end{array}\tag{4}
$$

where $u _ { t }$ is the update direction fed to lower-level optimizers and $\lambda _ { i }$ are task-dependent weights.

Different choices of $\lambda _ { i }$ recover many existing OCL algorithms: experience replay methods such as ER and DER use uniform averaging over buffered gradients [50, 6, 69]; gradient-alignment methods such as GEM, A-GEM, and MER implicitly realize Eq. 4 via projection-based or bi-level updates [38, 8, 49]; Pareto-style methods (e.g. , POCL) optimize λ to balance previous tasks within a multi-objective formulation [69]. In all these cases, the gradients are evaluated at the current parameter $\theta _ { t }$ , and only their linear combination in gradient space is modified.

Which Averaging Mechanism Suits OCL. Under the viewpoint of Polyak–Ruppert averaging and Primal averaging, the two possible ways to form an averaged direction are: (i) averaging the gradients at $\phi _ { t }$ and $\theta _ { t } ,$ or (ii) averaging the iterates and then taking a single gradient at the interpolated point. Gradient-space averaging’s formulation can be generalized as a weighted average of gradients:

$$
g _ { t } ^ { \mathrm { g r a d } } = ( 1 - \beta ) \nabla \mathcal { L } ( \phi _ { t } ) + \beta \nabla \mathcal { L } ( \theta _ { t } ) .\tag{5}
$$

Compared with directly averaging gradients from different iterates, parameter-space averaging evaluates the gradient at a single interpolated point between two coupled sequences:

$$
g _ { t } ^ { \mathrm { p a r a m } } = \nabla \mathcal { L } ( ( 1 - \beta ) \phi _ { t } + \beta \theta _ { t } ) .\tag{6}
$$

This formulation induces a coherent optimization direction along the shared parameter manifold while implicitly incorporating local curvature information through the interpolated evaluation point. In contrast, gradient-space averaging mixes gradients computed at different locations in parameter space, which may lead to inconsistent update directions under non-stationary data streams. Motivated by these observations, we adopt parameter-space interpolation as the core design principle of SPIN. In Section 4.2, we further show that this formulation admits a unified optimization interpretation and naturally connects to curvature-aware continual learning methods. Empirical comparisons between gradient-space and parameter-space averaging are presented in Section 5.3.

## 4 Method

In this section, we first present the general form of our SPIN optimizer in Section 4.1, and then provide a theoretical perspective that connects our optimizer with existing methods in Section 4.2. Finally, we establish the theoretical guarantees and empirical validation of our optimizer in Section 4.3.

## 4.1 The SPIN Optimizer

We propose our SPIN optimizer as follows:

$$
x _ { t } = ( 1 - \beta ) \phi _ { t } + \beta \theta _ { t }\tag{7}
$$

$$
\phi _ { t + 1 } = O p t ( \phi _ { t } , \nabla \mathcal { L } ( x _ { t } ) , \gamma )\tag{8}
$$

$$
\theta _ { t + 1 } = \theta _ { t } + \alpha ( \phi _ { t + 1 } - \theta _ { t } )\tag{9}
$$

where $O p t ( \cdot )$ denotes the update function of a standard base optimizer $( \mathrm { e . g . }$ , SGD or AdamW) with learning rate γ. Here, ϕ is the fast-moving plasticity sequence, $x _ { t }$ is the interpolated point where the stochastic gradient $\nabla \mathcal { L } ( \boldsymbol { x } _ { t } )$ is evaluated, and $\theta _ { t }$ is a slow-moving stability sequence that evolves jointly. The coefficients $\alpha , \dot { \beta } \in [ 0 , 1 ]$ control the synchronization strength and the interpolation ratio between the two sequences, respectively. Note that $\mathcal { L }$ is non-stationary due to the changing data distributions [57, 41] and depends on the upstream algorithm design. Detailed pseudo-code of the SPIN algorithm is provided in Appendix I.

SPIN adopts a synchronous dual-sequence coupling mechanism: the plasticity sequence $\phi _ { t }$ rapidly adapts to incoming data, while the stability sequence $\theta _ { t }$ incrementally consolidates these updates into a robust representation. This design ensures that both sequences evolve jointly at every iteration. The plasticity sequence $\phi _ { t }$ captures fast adaptation to new data, whereas the stability sequence $\theta _ { t }$ gradually consolidates these changes into a consistent and robust representation. In the OCL setting, where the data distribution drifts over time, such tightly coupled dynamics allow $\theta _ { t }$ to promptly absorb relevant new information while preventing abrupt forgetting of past knowledge. Moreover, SPIN evaluates the gradient at the interpolated point $x _ { t }$ . This intermediate evaluation allows the optimizer to blend the short-term plastic response of $\phi _ { t }$ with the long-term stability encoded in $\theta _ { t }$ effectively guiding the update toward a compromise direction between adaptation and retention.

## 4.2 A Unified View

To understand the internal dynamics of the SPIN optimizer, we first analyze its update rules from first principles. The optimizer maintains two sequences, $\phi _ { t }$ and $\theta _ { t }$ , updated as per Eq. 9. Gradients $g _ { t } = \nabla \mathcal { L } ( x _ { t } )$ are computed at $x _ { t } = \theta _ { t } + ( 1 - \beta ) d _ { t }$ , where ${ d _ { t } } = { \phi _ { t } } - { \theta _ { t } }$ is the gap vector. First, assuming the loss function $\mathcal { L } _ { t } ( \boldsymbol { \theta } )$ is sufficiently smooth, we employ the first-order Taylor expansion of the gradient $g _ { t } = \nabla \mathcal { L } ( x _ { t } )$ around the iterate $\theta _ { t } .$ , which gives:

$$
{ \boldsymbol { g } } _ { t } \approx \nabla { \boldsymbol { \mathcal { L } } } _ { t } ( { \boldsymbol { \theta } } _ { t } ) + \nabla ^ { 2 } { \boldsymbol { \mathcal { L } } } _ { t } ( { \boldsymbol { \theta } } _ { t } ) ( x _ { t } - { \boldsymbol { \theta } } _ { t } ) = \nabla { \boldsymbol { \mathcal { L } } } _ { t } ( { \boldsymbol { \theta } } _ { t } ) + ( 1 - \beta ) H _ { t } d _ { t }\tag{10}
$$

where $H _ { t } : = \nabla ^ { 2 } \mathcal { L } _ { t } ( \theta _ { t } )$ is the Hessian at $\theta _ { t }$ . Second, invoking a steady-state assumption for the gap vector $d _ { t } \left( \mathrm { i . e . , } d _ { t + 1 } \approx d _ { t } \right)$ ), allows us to express $d _ { t }$ as a function of the gradient $\nabla { \mathcal { L } } _ { t } ( \theta _ { t } )$ and Hessian $H _ { t }$ [3]. This analysis leads to the following result:

Proposition 1 Suppose the gradient $g _ { t } = \nabla \mathcal { L } ( x _ { t } )$ can be approximated by its first-order Taylor expansion around $\theta _ { t }$ as in Eq. 10, and assume the gap vector ${ d _ { t } } = { \phi _ { t } } - { \theta _ { t } }$ reaches a quasi-steady state $( d _ { t + 1 } \approx d _ { t } ) .$ . Then the effective updatefor the iterate $\theta _ { t }$ simplifies to:

$$
\theta _ { t + 1 } - \theta _ { t } \approx - \gamma [ I + \rho _ { t } H _ { t } ] ^ { - 1 } \nabla \mathcal { L } _ { t } ( \theta _ { t } ) ,\tag{11}
$$

where $\rho _ { t } = \gamma ( 1 - \alpha ) ( 1 - \beta ) / \alpha \geq 0 .$ . (Full derivation in Appendix A.1.)

Recent work [68] unifies regularization-based methods and Meta-Continual Learning by showing that they both approximate a unified iterative update of the form:

$$
\theta  \theta - \alpha \tilde { H } ^ { - 1 } \nabla \mathcal { L } ( \theta ) ,\tag{12}
$$

where $\tilde { H } ^ { - 1 }$ represents a specific (explicit or implicit) inverse Hessian approximation $( \mathrm { e . g . }$

Table 1: A Unified View of CL methods as Hessian Approximation.
<table><tr><td>Method Family</td><td>Hessian Approximation</td></tr><tr><td>Regularization (EWC) [27]</td><td> $\begin{array} { r } { \approx ( \sum _ { i = 1 } ^ { j - 1 } H _ { i } ) ^ { - 1 } } \end{array}$ </td></tr><tr><td>Meta-CL (La-MAML) [19]</td><td> $\approx ( H _ { M } ^ { j } ) ^ { - 1 }$ </td></tr><tr><td>SPIN</td><td> $\approx \gamma ( \dot { I } + \rho _ { t } H _ { t } ) ^ { - 1 }$ </td></tr></table>

$H _ { r e g } ^ { - 1 }$ from past-task penalties or $H _ { M } ^ { - 1 }$ from bi-level meta-objectives). Upon comparing our derived effective update for SPIN (Eq. 11) with this unified form (Eq. 12), a clear structural equivalence emerges. The SPIN optimizer, despite its distinct formulation, implicitly arrives at the same general structure by effectively using $\tilde { H } ^ { - 1 } \approx \gamma ( I + \rho _ { t } H _ { t } ) ^ { - 1 }$ as its inverse Hessian approximation. This finding allows us to surprisingly place SPIN optimization within the same conceptual framework as regularization and meta-CL methods, as summarized in Table 1. This derived update rule (Eq. 11) provides a new lens for understanding SPIN’s underlying mechanism, revealing a deep and previously unnoted connection to a unified theoretical framework that links seemingly disparate CL approaches.

## 4.3 Theoretical Analysis and Empirical Validation

Theoretical Mechanisms. Our analysis in Section 4.2 reveals that SPIN implicitly utilizes an effective inverse Hessian approximation of the form $\tilde { H } ^ { - 1 } \approx \gamma ( I + \rho _ { t } H _ { t } ) ^ { - 1 }$ . A primary difficulty in OCL arises from high-variance data streams [68], where the local Hessian $H _ { t }$ estimated from individual samples is extremely noisy and ill-conditioned. Conventional CL methods relying on purely data-dependent Hessian approximations risk catastrophic forgetting because they over-trust unreliable low-curvature estimates. In contrast, SPIN’s formulation adds a “floor” (akin to Tikhonov regularization [5, 4]). This effectively damps extreme curvature ratios while preserving relative geometry. It acts as a curvature-aware method in high-curvature directions but gracefully reverts to a stable update in uncertain, low-curvature directions.

![](images/e432aec133384c39cc8884a57b1312e6a497310fe0d2e83b5de362aa2e438c2a.jpg)

![](images/f5b2f952cdb4a4505dc437da15848cdca1cd41660337491261a845321644217f.jpg)  
(a)

![](images/8f76380c47b156d59858785c019bd466befe3ac9cd45bc567c9f607688f8fc49.jpg)  
(b)  
Figure 2: Empirical validation of SPIN. (a) Verification of the quasi-steady state assumption. (b) Stability of the update L2 norm compared to a baseline optimizer.

Empirical Validation. We empirically validate these core mechanisms. (For detailed experimental setup, see the appendix.) First, our derivation relies on the quasi-steady state (QSS) assumption $( d _ { t + 1 } \approx d _ { t } )$ . Figure 2a validates this assumption, showing that $\frac { \| d _ { t + 1 } \| } { \| d _ { t } \| } \approx 1$ throughout training, indicating stable magnitude. Meanwhile, $\cos ( d _ { t + 1 } , d _ { t } )$ remains close to 1, implying strong directional consistency. Notably, both properties remain stable even around task boundaries, indicating that $d _ { t }$ evolves smoothly despite distribution shifts, which justifies the QSS assumption. Second, we validate our theory by comparing the L2 norm of the optimization updates, as this behavior is the direct, measurable symptom of the underlying scaling factor stability our theory addresses. As Figure 2b illustrates, the magnitude of SPIN’s updates (orange line) is stable and “not violently fluctuating,” in stark contrast to the naive optimizer’s volatile updates (blue line).

Besides, we have the following theorem to further characterize the performance of SPIN.

Theorem 1 (Regret Bound) Assume the loss $\scriptstyle { \mathcal { L } } _ { t }$ are convex, G-Lipschitz and η-smooth over a closed convex domain $\mathcal { \bar { X } } \subset \mathbb { R } ^ { d }$ with diameter D, and that the gradients are unbiased with bounded variance $\mathbb { E } [ \| g _ { t } - \nabla \mathcal { L } _ { t } ( x _ { t } ) \| ^ { 2 } ] \le \sigma ^ { 2 }$ and bounded magnitude $\| g _ { t } \| \leq G$ . Let $\begin{array} { r } { \theta ^ { \star } = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \sum _ { t = 1 } ^ { T } \mathcal { L } _ { t } ( \theta ) } \end{array}$ and define the static regret $\begin{array} { r } { R _ { T } = \sum _ { t = 1 } ^ { T } \left( \mathcal { L } _ { t } ( \theta _ { t } ) - \mathcal { L } _ { t } ( \theta ^ { \star } ) \right) } \end{array}$ . Let $\alpha \in ( 0 , 1 )$ and step size $\gamma ^ { \star } =$ ${ \frac { D } { G { \sqrt { T } } } } { \sqrt { \frac { \alpha } { 2 - \alpha } } } .$ . Then with probability at least $1 - \delta f o r a n y \delta \in ( 0 , 1 )$ , we have

$$
\begin{array} { r } { R _ { T } \leq D G ( \sqrt { \frac { ( 2 - \alpha ) T } { \alpha } } + \frac { 2 } { 3 } \ln \frac { 1 } { \delta } ) + D \sigma \sqrt { 2 T \ln \frac { 1 } { \delta } } . } \end{array}\tag{13}
$$

Theorem 1 shows that, under mild assumptions, our proposed optimizer achieves a nearly-optimal $\tilde { \mathcal { O } } ( { \sqrt { T } } )$ regret bound with high probability in the online optimization setting. This result theoretically substantiates the efficiency and stability of the proposed optimization framework. Detailed proof is provided in Appendix A.2.

## 5 Experiments

## 5.1 Experimental Setups

Datasets. We evaluate our method on four datasets, i.e. , CIFAR-100 (100 classes) [28], Tiny-ImageNet (200 classes) [31], CLEAR100 (100 classes) [36], and Imagenet-1K (1000 classes) [53]. More details are provided in Appendix D.1.

Baselines. To show the effectiveness of our proposed optimizer, we integrate it into 11 representative OCL methods, which span various categories, include ER [50], DER++ [6], COPE [30], SCR [40],

Table 2: Comparison of final Average Accuracy (Acc. %) using different optimizers on Split CIFAR-100 (M=2k) and Split Tiny-ImageNet (M=5k). Our SPIN is benchmarked against 4 standard optimizers across 11 SOTA OCL methods. The best results are highlighted using bold font.
<table><tr><td colspan="2"></td><td colspan="6">Split CIFAR-100 (10 tasks, M=2k)</td><td colspan="6">Split Tiny-ImageNet (20 tasks, M=5k)</td></tr><tr><td>Baseline</td><td></td><td>SGD</td><td>SPIN-SGD</td><td>RMSprop</td><td>Adam</td><td>AdamW</td><td>SPIN-AdamW</td><td>SGD</td><td>SPIN-SGD</td><td>RMSprop</td><td>Adam</td><td>AdamW</td><td>SPIN-AdamW</td></tr><tr><td>ER</td><td>(2019)</td><td>27.94</td><td>33.31</td><td>26.38</td><td>31.02</td><td>33.07</td><td>35.92</td><td>21.68</td><td>25.24</td><td>19.25</td><td>23.38</td><td>23.90</td><td>25.94</td></tr><tr><td>DER++</td><td>(2020)</td><td>26.10</td><td>35.90</td><td>27.17</td><td>30.84</td><td>31.11</td><td>36.06</td><td>18.95</td><td>27.27</td><td>23.37</td><td>26.34</td><td>26.69</td><td>29.79</td></tr><tr><td>COPE</td><td>(2021)</td><td>30.81</td><td>34.18</td><td>31.12</td><td>32.90</td><td>33.55</td><td>35.63</td><td>18.31</td><td>20.43</td><td>18.01</td><td>19.80</td><td>20.77</td><td>21.66</td></tr><tr><td>SCR</td><td>(2021)</td><td>29.82</td><td>31.53</td><td>28.07</td><td>30.52</td><td>31.32</td><td>32.86</td><td>17.26</td><td>18.77</td><td>17.11</td><td>18.30</td><td>18.74</td><td>20.39</td></tr><tr><td>ER-ACE(2022)</td><td></td><td>28.54</td><td>36.14</td><td>27.65</td><td>32.43</td><td>34.77</td><td>37.02</td><td>24.95</td><td>28.11</td><td>23.75</td><td>28.10</td><td>28.52</td><td>30.42</td></tr><tr><td>OCM</td><td>(2022)</td><td>33.21</td><td>36.06</td><td>33.69</td><td>36.78</td><td>37.45</td><td>40.15</td><td>22.19</td><td>24.91</td><td>23.47</td><td>24.83</td><td>25.68</td><td>27.29</td></tr><tr><td>OnPro</td><td>(2023)</td><td>32.44</td><td>34.35</td><td>29.56</td><td>33.56</td><td>34.25</td><td>36.61</td><td>21.38</td><td>23.42</td><td>20.21</td><td>21.99</td><td>22.26</td><td>24.94</td></tr><tr><td>GSA</td><td>(2023)</td><td>33.25</td><td>36.79</td><td>30.78</td><td>35.90</td><td>36.53</td><td>37.73</td><td>20.80</td><td>22.44</td><td>20.84</td><td>22.16</td><td>25.67</td><td>26.63</td></tr><tr><td>MOSE</td><td>(2024)</td><td>44.18</td><td>47.18</td><td>43.46</td><td>46.21</td><td>46.95</td><td>48.33</td><td>28.21</td><td>29.91</td><td>28.02</td><td>29.57</td><td>29.87</td><td>30.63</td></tr><tr><td>CCL-DC(2024)</td><td></td><td>38.80</td><td>41.74</td><td>42.17</td><td>42.97</td><td>43.99</td><td>45.68</td><td>27.01</td><td>28.91</td><td>27.84</td><td>28.10</td><td>28.35</td><td>29.32</td></tr><tr><td>S6MOD (2025)</td><td></td><td>34.12</td><td>34.81</td><td>33.08</td><td>35.19</td><td>35.55</td><td>37.37</td><td>22.02</td><td>24.00</td><td>20.94</td><td>23.43</td><td>24.01</td><td>25.17</td></tr><tr><td>Average</td><td></td><td>32.66</td><td>36.54</td><td>32.10</td><td>35.30</td><td>36.23</td><td>38.49</td><td>21.25</td><td>23.95</td><td>21.01</td><td>24.09</td><td>24.95</td><td>26.56</td></tr></table>

Table 3: Comparison of Averaged Anytime Accuracy (AAA %) using different optimizers on Split CIFAR-100 (M=2k) and Split Tiny-ImageNet (M=5k). Our SPIN is benchmarked against 4 standard optimizers across 11 SOTA OCL methods. The best results are highlighted using bold font.
<table><tr><td colspan="2"></td><td colspan="6">Split CIFAR-100 (10 tasks, M=2k)</td><td colspan="6">Split Tiny-ImageNet (20 tasks, M=5k)</td></tr><tr><td>Baseline</td><td></td><td>SGD</td><td>SPIN-SGD</td><td>RMSprop</td><td>Adam</td><td>AdamW</td><td>SPIN-AdamW</td><td>SGD</td><td>SPIN-SGD</td><td>RMSprop</td><td>Adam</td><td>AdamW</td><td>SPIN-AdamW</td></tr><tr><td>ER</td><td>(2019)</td><td>36.86</td><td>46.18</td><td>34.54</td><td>40.68</td><td>41.50</td><td>48.08</td><td>28.36</td><td>34.38</td><td>28.14</td><td>32.47</td><td>32.81</td><td>37.33</td></tr><tr><td>DER++</td><td>(2020)</td><td>35.16</td><td>45.49</td><td>33.68</td><td>37.99</td><td>38.86</td><td>46.26</td><td>21.20</td><td>35.39</td><td>30.50</td><td>36.49</td><td>36.87</td><td>40.08</td></tr><tr><td>COPE</td><td>(2021)</td><td>37.76</td><td>44.62</td><td>34.09</td><td>39.48</td><td>42.01</td><td>47.38</td><td>26.49</td><td>26.80</td><td>24.44</td><td>28.54</td><td>27.99</td><td>31.01</td></tr><tr><td>SCR</td><td>(2021)</td><td>42.55</td><td>45.70</td><td>37.74</td><td>41.24</td><td>42.44</td><td>48.20</td><td>24.03</td><td>26.01</td><td>23.15</td><td>26.19</td><td>26.24</td><td>28.30</td></tr><tr><td>ER-ACE(2022)</td><td></td><td>38.38</td><td>45.91</td><td>38.48</td><td>41.26</td><td>42.05</td><td>46.61</td><td>32.20</td><td>37.25</td><td>31.56</td><td>36.55</td><td>37.54</td><td>40.03</td></tr><tr><td>OCM</td><td>(2022)</td><td>40.17</td><td>41.83</td><td>39.98</td><td>42.26</td><td>43.39</td><td>47.06</td><td>27.28</td><td>30.73</td><td>26.44</td><td>30.08</td><td>31.50</td><td>34.21</td></tr><tr><td>OnPro</td><td>(2023)</td><td>38.09</td><td>40.29</td><td>37.20</td><td>41.21</td><td>41.44</td><td>43.97</td><td>22.44</td><td>29.86</td><td>24.87</td><td>29.41</td><td>28.24</td><td>32.48</td></tr><tr><td>GSA</td><td>(2023)</td><td>44.91</td><td>50.43</td><td>39.44</td><td>44.40</td><td>45.40</td><td>51.37</td><td>27.45</td><td>31.16</td><td>27.08</td><td>31.87</td><td>33.08</td><td>35.93</td></tr><tr><td>MOSE</td><td>(2024)</td><td>56.53</td><td>58.55</td><td>53.49</td><td>58.94</td><td>58.76</td><td>61.79</td><td>33.65</td><td>36.89</td><td>33.06</td><td>36.48</td><td>37.01</td><td>40.74</td></tr><tr><td>CCL-DC(2024)</td><td></td><td>47.80</td><td>51.29</td><td>49.53</td><td>55.30</td><td>55.61</td><td>58.58</td><td>34.21</td><td>36.52</td><td>33.79</td><td>35.13</td><td>35.84</td><td>37.48</td></tr><tr><td>S6MOD (2025)</td><td></td><td>42.21</td><td>45.02</td><td>41.35</td><td>44.17</td><td>44.67</td><td>47.96</td><td>28.08</td><td>29.93</td><td>29.24</td><td>30.21</td><td>31.00</td><td>32.85</td></tr><tr><td>Average</td><td></td><td>41.86</td><td>46.85</td><td>39.96</td><td>44.27</td><td>45.10</td><td>49.75</td><td>27.76</td><td>32.27</td><td>28.54</td><td>32.13</td><td>32.55</td><td>34.59</td></tr></table>

ER-ACE [7], OCM [17], OnPro [67], GSA [18], MOSE [71], CCL-DC [63], S6MOD [37]. To rigorously isolate and evaluate the contribution of our optimization strategy, we compare the performance of these OCL methods under our SPIN optimizer and that under four widely-used generic optimizers: SGD [51], Adam [26], AdamW [39] and RMSProp [51].

Implementation details. For all experiments, we employ a ResNet-18 [22] backbone, by following the common architecture in the broad OCL literature [63, 37, 67]. The streaming input batch size is set to 10 and the memory buffer batch size to 64, consistent with other methods [63, 37]. All main results use fixed hyperparameters α = β = 0.05. For each OCL method, we use the same learning rate and weight decay for AdamW and SPIN-AdamW, and likewise for SGD and SPIN-SGD. All other settings follow the corresponding baseline implementations. Results are averaged over 5 runs except for ImageNet-1K, which is evaluated with a single run due to computational constraints. For evaluation, we measure the model performance using Final Average Accuracy (Acc) and Averaged Anytime Accuracy (AAA). Details are provided in Appendix D.3.

## 5.2 Main Results

Performance on Standard Benchmarks. As shown in Table 2 and Table 3, SPIN consistently outperforms generic optimizers across 11 OCL baselines on Split CIFAR-100 and Tiny-ImageNet. Notably, SPIN-AdamW raises MOSE’s final accuracy (ACC) from 46.95% to 48.33% and GSA’s anytime accuracy (AAA) from 45.40% to 51.37% on CIFAR-100. This demonstrates its strong ability to improve peak performance and effectively mitigate forgetting.

Scalability to Million-Scale Data and Robustness to Natural Distribution Shifts. On the largescale ImageNet-1K dataset, replacing SGD with SPIN yields substantial gains (Table 4). For instance, it improves DER++’s ACC from 14.19% to 17.51% and AAA from 18.50% to 22.02%, confirming its effectiveness and scalability on complex data streams. On CLEAR100, which exhibits real world natural shifts devoid of clear task boundaries, SPIN consistently surpasses both standard SGD and

Table 4: Comparison of final Average Accuracy (Acc.) and Averaged Anytime Accuracy (AAA) on Imagenet-1K.
<table><tr><td></td><td colspan="2"> $\operatorname { A c c . } ( { \% } )$ </td><td colspan="2">AAA(%)</td></tr><tr><td>Baseline</td><td>SGD</td><td>SPIN</td><td>SGD</td><td>SPIN</td></tr><tr><td>ER</td><td>13.21</td><td>15.37</td><td>19.76</td><td>21.58</td></tr><tr><td>DER++</td><td>14.19</td><td>17.51</td><td>18.50</td><td>22.02</td></tr></table>

Table 5: Comparison on CLEAR100 dataset (Acc %).
<table><tr><td>一</td><td colspan="3">M=200</td><td colspan="3">M=5000</td></tr><tr><td>Baseline</td><td>SGD</td><td>SGD-M</td><td>SPIN</td><td>SGD</td><td>SGD-M</td><td>SPIN</td></tr><tr><td>ER</td><td>32.63</td><td>36.01</td><td>37.99</td><td>41.38</td><td>51.87</td><td>53.02</td></tr><tr><td>DER++</td><td>33.54</td><td>36.87</td><td>37.96</td><td>41.20</td><td>50.49</td><td>52.21</td></tr><tr><td>GSA</td><td>37.76</td><td>40.62</td><td>42.38</td><td>43.86</td><td>46.20</td><td>47.88</td></tr></table>

SGD-M, i.e., SGD based on Exponential Moving Average (EMA), across varying memory sizes (Table 5), proving its stability-plasticity coupling is superior to naive parameter averaging.

## 5.3 Ablation Studies and Analysis

To further understand the behavior of SPIN, we conduct a series of ablation studies and analyses to answer the following key questions:

## Q1: Does SPIN effectively balance stability and plasticity?

To better understand the learning dynamics, Figure 3 compares the task-wise Learning Accuracy (reflecting plasticity) and Relative Forgetting (reflecting stability) between SPIN and AdamW. SPIN consistently achieves higher learning accuracy while maintaining lower relative forgetting throughout the task stream. These results suggest that SPIN enables effective adaptation to new tasks without excessively overwriting previously acquired

![](images/baf084e6eab978e3451132c514e766ebd71ad498855ed58735d830702f794228.jpg)

![](images/bc631b302403d6846f5cc309a08f13842089be95aa98edfae504a326d31dc2c1.jpg)  
Figure 3: Task-wise Learning Accuracy and Relative Forgetting of the ER method $( M = 2 0 0 0 )$ , comparing the performance of AdamW and SPIN.

knowledge. This behavior is consistent with the core design of SPIN, where the interpolation between plastic and stable iterate sequences promotes a better balance between adaptation and retention during continual optimization. We further compare Relative Forgetting(RF) across 11 OCL methods on Split CIFAR-100 (Table 6). SPIN lowers average RF from 31.94% to 29.99% with SGD and from 30.83% to 28.63% with AdamW. Combined with the higher final accuracies in Table 2, these results indicate a better overall accuracy–forgetting trade-off.

Table 6: Relative Forgetting(RF) (% ↓) on Split CIFAR-100 (M = 2k). Best in each optimizer pair is bolded.
<table><tr><td>Method</td><td>SGD</td><td>SPIN-SGD</td><td>AdamW</td><td>SPIN-AdamW</td></tr><tr><td>ER</td><td>39.14</td><td>36.70</td><td>36.15</td><td>33.09</td></tr><tr><td>DER++</td><td>53.30</td><td>42.41</td><td>45.43</td><td>36.49</td></tr><tr><td>COPE</td><td>13.64</td><td>12.50</td><td>12.07</td><td>10.91</td></tr><tr><td>SCR</td><td>50.47</td><td>51.33</td><td>46.98</td><td>50.19</td></tr><tr><td>ER-ACE</td><td>17.24</td><td>19.27</td><td>17.20</td><td>16.72</td></tr><tr><td>OCM</td><td>14.01</td><td>13.78</td><td>13.51</td><td>12.98</td></tr><tr><td>OnPro</td><td>16.40</td><td>15.24</td><td>21.88</td><td>17.46</td></tr><tr><td>GSA</td><td>36.53</td><td>39.45</td><td>35.76</td><td>34.76</td></tr><tr><td>MOSE</td><td>33.47</td><td>31.35</td><td>37.20</td><td>36.11</td></tr><tr><td>CCL-DC</td><td>37.56</td><td>33.78</td><td>37.62</td><td>32.05</td></tr><tr><td>S6MOD</td><td>39.62</td><td>34.11</td><td>35.35</td><td>34.20</td></tr><tr><td>Average</td><td>31.94</td><td>29.99</td><td>30.83</td><td>28.63</td></tr></table>

Table 7: Component ablation of SPIN-AdamW with GSA on Split CIFAR-100 $( M = 1 \mathbf { k } )$ , comparing boundary settings of α and β. ACC and AAA (%) are reported. Best results are bolded.
<table><tr><td>Method</td><td>Acc.(%) ↑</td><td>AAA(%) ↑</td></tr><tr><td>AdamW</td><td>27.84</td><td>39.53</td></tr><tr><td> $\mathrm { S P I N } ( \alpha { = } 0 )$ </td><td>1.04</td><td>2.93</td></tr><tr><td> $\mathrm { S P I N } ( \alpha { = } 1 )$ </td><td>27.52</td><td>37.70</td></tr><tr><td> $\mathrm { S P I N } ( \beta { = } 1 )$ </td><td>22.30</td><td>30.63</td></tr><tr><td> $\operatorname { S P I N } ( \beta = 0 )$ </td><td>31.34</td><td>40.28</td></tr><tr><td>SPIN</td><td>32.20</td><td>42.39</td></tr></table>

Q2: What are the key components behind SPIN’s effectiveness? We first analyze several boundary cases of SPIN in Table 7. When $\alpha = 0 .$ , the optimization collapses since the model parameters remain nearly unchanged throughout training. When $\alpha = 1$ , SPIN degenerates into standard AdamW. Similarly, when $\beta = \bar { 1 }$ , gradients are evaluated entirely on the stability sequence, resulting in overly conservative updates that hinder adaptation to new tasks. When $\beta = 0 ,$ SPIN reduces to an EMAbased optimizer (i.e. , AdamW+EMA), similar to prior temporal ensemble approaches [58]. Although EMA improves optimization stability, SPIN consistently achieves superior performance across all

Table 8: Comparison of Averaged Anytime Accuracy (AAA %) and Average Accuracy (ACC %) using different Rehearsal-Free methods on Split CIFAR-10 and Split CIFAR-100. The best and second best results are highlighted using bold and underline fonts.  
![](images/6d20fcdaf7758a517c75868b1dd8f29047b4fb45ffe647854beda3beac719e4e.jpg)

![](images/d9c664e46c5d9c0980c03d60e9cc88cb72ebefeac54c0c424760696fa8f4c65f.jpg)

<table><tr><td rowspan="2">Baseline</td><td colspan="2">Split CIFAR-10</td><td colspan="2">Split CIFAR-100</td></tr><tr><td>AAA</td><td>Acc</td><td>AAA</td><td>Acc</td></tr><tr><td>on-EWC</td><td>40.27</td><td>18.46</td><td>13.35</td><td>4.99</td></tr><tr><td>SI</td><td>39.84</td><td>17.49</td><td>12.32</td><td>5.20</td></tr><tr><td>SPIN-SGD</td><td>40.52</td><td>19.09</td><td>14.63</td><td>5.91</td></tr><tr><td>SPIN-AdamW</td><td>41.16</td><td>19.18</td><td>15.60</td><td>5.58</td></tr></table>

![](images/9c86dd374e3bf87ce53c7acf3a4c3c3d890e5d9676b528a27aae0d5991e009e3.jpg)

![](images/7ecf16c0c9f69b27611272c12c6b9eec91b27085815bbba8ba933bdf6d9241b9.jpg)  
Figure 4: Impact of Replay Buffer Size

Table 9: Runtime per training step (ms/step) and peak memory (MB) of different optimizers with GSA on split CIFAR-100.
<table><tr><td>Method</td><td>Runtime</td><td>Peak Memory</td></tr><tr><td>SGD</td><td>1.04(ms/step)</td><td>155.86(MB)</td></tr><tr><td>SGD+EMA</td><td>1.08(ms/step)</td><td>240.75(MB)</td></tr><tr><td>AdamW</td><td>1.68(ms/step)</td><td>267.58(MB)</td></tr><tr><td>SPIN-SGD</td><td>1.13(ms/step)</td><td>240.75(MB)</td></tr></table>

Table 10: Accuracy under approximately matched total memory with GSA on CIFAR-100. SGD uses a 30k replay buffer to offset SPIN’s additional optimizer states with a 15k buffer. Best results are bolded.

<table><tr><td rowspan="2">Buffer Size</td><td colspan="2">SGD</td><td>SPIN</td></tr><tr><td>15k</td><td>30k</td><td>15k</td></tr><tr><td>Acc (%)</td><td>41.61</td><td>46.93</td><td>48.46</td></tr><tr><td>AAA (%)</td><td>47.41</td><td>51.97</td><td>55.67</td></tr></table>

settings. Additional comparisons with SGD+EMA baselines are provided in Appendix F. These results suggest that the gains of SPIN cannot be solely attributed to temporal averaging, and instead highlight the importance of gradient-location interpolation between the plastic and stable sequences.

Q3: Is SPIN robust under different memory settings? An ideal CL method should be robust across different memory sizes in real world application. To investigate whether SPIN is effective with different memory constraints, we conducted experiments on CIFAR-100 using various memory sizes. As seen from Figure 4, SPIN consistently improves performance across all replay buffer sizes, including strict memory-constrained settings, suggesting that SPIN is effective even with different buffer sizes.

Q4: Can SPIN maintain stability without replay memory? To further investigate whether SPIN can induce intrinsic stability beyond replay mechanisms, we evaluate SPIN under the rehearsal-free setting. We compare SPIN against representative rehearsal-free continual learning methods, including On-EWC [24] and SI [74], which explicitly regularize parameter updates using importance-aware constraints. As shown in Table 8, both SPIN-SGD and SPIN-AdamW consistently outperform these rehearsal-free baselines across all metrics. These results suggest that SPIN is able to preserve past knowledge through its optimization dynamics alone, which is consistent with our theoretical analysis that SPIN implicitly induces Hessian-aware stabilization during continual optimization.

Q5: Are SPIN’s gains simply due to increased optimizer memory? As shown in Table 9, SPIN-SGD introduces only modest runtime and memory overhead compared to vanilla SGD, while remaining substantially more efficient than AdamW. To rule out the possibility that SPIN’s gains arise merely from additional optimizer states, we evaluate SPIN under a strict equal-memory budget. On CIFAR-100, the extra optimizer state of SPIN-SGD is equivalent to approximately 15k replay samples. We therefore compare SPIN with a 15k replay buffer against SGD with a 30k replay buffer, ensuring comparable total memory usage. As shown in Table 10, SPIN (15k) still achieves higher ACC and AAA than the memory-matched SGD (30k). These results suggest that SPIN’s improvements primarily arise from its optimization dynamics and stability–plasticity balancing mechanism.

Q6: Which averaging mechanism is more suitable for OCL? We compare parameter-space interpolation with gradient-space averaging under the same optimization framework. As shown in

![](images/6c22393abd82b46d8e0a53b6fe13c749997f6e257787ef8ab9d0d0c6149b79e8.jpg)  
(a) Task-wise AAA under different averaging mechanisms.

![](images/7c5f420279ff9feba4fcb5293636812bb606963c325bc27277940d28f502d8f6.jpg)  
(b) Training time comparison.  
Figure 5: Comparison between parameter-space interpolation and gradient-space averaging in OCL.

Figure 6a, parameter-space interpolation consistently achieves higher task-wise AAA, suggesting that evaluating gradients at a shared interpolated point yields a more coherent optimization trajectory under non-stationary data streams. In contrast, directly averaging gradients computed at different parameter locations introduces conflicting update directions. We further compare their computational efficiency in Figure 6b. Since gradient-space averaging requires two forward-backward passes, its training time is nearly twice that of parameter-space interpolation. In contrast, SPIN only requires a single gradient evaluation while achieving better continual learning performance.

## 6 Conclusion and Limitations

In this paper, we investigate the optimizer design for OCL and introduce a novel and dedicated optimizer SPIN, which explicitly balances the stability and plasticity tradeoff. Through its interpolationbased update mechanism, SPIN achieves smooth and robust optimization under non-stationary data streams, effectively mitigating catastrophic forgetting. Extensive experiments across diverse OCL frameworks demonstrate that SPIN consistently improves both accuracy and stability compared to widely used generic optimizers, regardless of the presence or size of replay buffers and even under strict memory constraints. These findings highlight that optimizer design plays a critical role in achieving a stable learning–forgetting balance.

Limitations: Our theoretical analysis relies on smoothness assumptions and the quasi-steadystate approximation, which may not fully characterize the optimization dynamics of highly overparameterized large-scale neural networks. In addition, although maintaining dual iterate sequences brings substantial continual learning improvements, it also introduces additional optimizer states compared to standard optimizers. Exploring more parameter-efficient interpolation mechanisms that maintain the trade-off between stability and plasticity, thereby making them more effective on edge devices, remains an important direction for future research. Another promising direction is to integrate SPIN with continual learning methods built on large-scale pretrained models, such as CLIP and pretrained ViTs, and investigate its role in balancing adaptation and retention during prompt tuning and low-rank adaptation.

## Acknowledgments and Disclosure of Funding

Acknowledgments. We thank the anonymous reviewers and the area chair for their constructive feedback, which helped improve this work.

Funding. The authors received no external funding or third-party support for this work.

Competing interests. The authors declare no competing interests.

## References

[1] Kwangjun Ahn, Gagik Magakyan, and Ashok Cutkosky. General framework for onlineto-nonconvex conversion: Schedule-free SGD is also effective for nonconvex optimization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 772–795. PMLR, 2025. URL https: //proceedings.mlr.press/v267/ahn25a.html.

[2] Elahe Arani, Fahad Sarfraz, and Bahram Zonooz. Learning fast, learning slow: A general continual learning method based on complementary learning system. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= uxxFrDwrE7Y.

[3] Michal Arbel and Julien Mairal. Amortized implicit differentiation for stochastic bilevel optimization. ArXiv, abs/2111.14580, 2021. URL https://api.semanticscholar.org/ CorpusID:244714983.

[4] Hedy Attouch, Aïcha Balhag, Zaki Chbani, and Hassan Riahi. Accelerated gradient methods combining tikhonov regularization with geometric damping driven by the hessian. Applied Mathematics & Optimization, 88(2):29, 2023.

[5] Radu Ioan Bo¸t, Ernö Robert Csetnek, and Szilárd Csaba László. Tikhonov regularization of a second order dynamical system with hessian driven damping. Mathematical Programming, 189 (1):151–186, 2021.

[6] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. ArXiv, abs/2004.07211, 2020. URL https://api.semanticscholar.org/CorpusID:215768806.

[7] Lucas Caccia, Rahaf Aljundi, Nader Asadi, Tinne Tuytelaars, Joelle Pineau, and Eugene Belilovsky. New insights on reducing abrupt representation change in online continual learning. ArXiv, abs/2203.03798, 2021. URL https://api.semanticscholar.org/CorpusID: 247315339.

[8] Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. ArXiv, abs/1812.00420, 2018. URL https://api. semanticscholar.org/CorpusID:54443381.

[9] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7): 3366–3385, 2022. doi: 10.1109/TPAMI.2021.3057446.

[10] Aaron Defazio. Momentum via primal averaging: Theoretical insights and learning rate schedules for non-convex optimization, 2020. URL https://api.semanticscholar.org/ CorpusID:235266276.

[11] Aaron Defazio and Robert Mansel Gower. The power of factorial powers: New parameter settings for (stochastic) optimization. In Asian Conference on Machine Learning, 2020. URL https://api.semanticscholar.org/CorpusID:224815911.

[12] Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and Ashok Cutkosky. The road less scheduled, 2024.

[13] Aaron Defazio, Konstantin Mishchenko, Parameswaran Raman, Hao-Jun Michael Shi, and Lin Xiao. Smoothing diloco with primal averaging for faster training of llms. arXiv preprint arXiv:2512.17131, 2025.

[14] Junze Deng, Qinhang Wu, Peizhong Ju, Sen Lin, Yingbin Liang, and Ness B. Shroff. Unlocking the power of rehearsal in continual learning: A theoretical perspective. ArXiv, abs/2506.00205, 2025. URL https://api.semanticscholar.org/CorpusID:279074801.

[15] David A Freedman. On tail probabilities for martingales. the Annals of Probability, pages 100–118, 1975.

[16] Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999.

[17] Yiduo Guo, Bing Liu, and Dongyan Zhao. Online continual learning through mutual information maximization. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 8109–8126, 17–23 Jul 2022.

[18] Yiduo Guo, Bing Liu, and Dongyan Zhao. Dealing with cross-task class discrimination in online continual learning. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11878–11887, 2023. URL https://api.semanticscholar.org/CorpusID: 258865951.

[19] Gunshi Gupta, Karmesh Yadav, and Liam Paull. La-maml: Look-ahead meta learning for continual learning. arXiv preprint arXiv:2007.13904, 2020.

[20] Elad Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization, 2(3-4):157–325, 2016.

[21] Jiangpeng He, Runyu Mao, Zeman Shao, and Fengqing Zhu. Incremental learning in online scenario. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13926–13935, 2020.

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[23] Steven CH Hoi, Doyen Sahoo, Jing Lu, and Peilin Zhao. Online learning: A comprehensive survey. Neurocomputing, 459:249–289, 2021.

[24] Ferenc Huszár. Note on the quadratic penalties in elastic weight consolidation. Proceedings of the National Academy of Sciences, 115:E2496 – E2497, 2017. URL https: //api.semanticscholar.org/CorpusID:3415694.

[25] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In Proceedings ofthe Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, 2018. URL https: //arxiv.org/abs/1803.05407.

[26] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL https://api.semanticscholar.org/CorpusID:6628106.

[27] James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114: 3521 – 3526, 2016. URL https://api.semanticscholar.org/CorpusID:4704285.

[28] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. URL https: //api.semanticscholar.org/CorpusID:18268744.

[29] Guanghui Lan. An optimal method for stochastic composite optimization. Mathematical Programming, 133(1):365–397, 2012.

[30] Matthias De Lange and Tinne Tuytelaars. Continual prototype evolution: Learning online from non-stationary data streams. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8230–8239, 2020. URL https://api.semanticscholar.org/CorpusID: 221447946.

[31] Yann Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.

[32] Hongbo Li, Sen Lin, Lingjie Duan, Yingbin Liang, and Ness B Shroff. Theory on mixture-ofexperts in continual learning. arXiv preprint arXiv:2406.16437, 2024.

[33] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23638–23647, 2024.

[34] Sen Lin, Li Yang, Deliang Fan, and Junshan Zhang. Trgp: Trust region gradient projection for continual learning. In The Tenth International Conference on Learning Representations, 2022.

[35] Sen Lin, Peizhong Ju, Yingbin Liang, and Ness Shroff. Theory on forgetting and generalization of continual learning. In International Conference on Machine Learning, pages 21078–21100. PMLR, 2023.

[36] Zhiqiu Lin, Jia Shi, Deepak Pathak, and Deva Ramanan. The clear benchmark: Continual learning on real-world imagery. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

[37] Sihao Liu, Yibo Yang, Xiaojie Li, David A Clifton, and Bernard Ghanem. Enhancing online continual learning with plug-and-play state space model and class-conditional mixture of discretization. arXiv preprint arXiv:2412.18177, 2024.

[38] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In Neural Information Processing Systems, 2017. URL https://api.semanticscholar. org/CorpusID:37308416.

[39] Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. ArXiv, abs/1711.05101, 2017. URL https://api.semanticscholar.org/CorpusID:3312944.

[40] Zheda Mai, Ruiwen Li, Hyunwoo J. Kim, and Scott Sanner. Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3584–3594, 2021. URL https://api.semanticscholar.org/CorpusID: 232352581.

[41] Zheda Mai, Ruiwen Li, Jihwan Jeong, David Quispe, Hyunwoo Kim, and Scott Sanner. Online continual learning in image classification: An empirical survey. Neurocomputing, 469:28–51, 2022. ISSN 0925-2312. doi: https://doi.org/10.1016/j.neucom.2021.10.021. URL https: //www.sciencedirect.com/science/article/pii/S0925231221014995.

[42] Imad Eddine Marouf, Subhankar Roy, Enzo Tartaglione, and Stéphane Lathuilière. Weighted ensemble models are strong continual learners. In European Conference on Computer Vision (ECCV), 2024. URL https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/html/ 9002\_ECCV\_2024\_paper.php.

[43] Martial Mermillod, Aurélia Bugaiska, and Patrick Bonin. The stability-plasticity dilemma: Investigating the continuum from catastrophic forgetting to age-limited learning effects, 2013.

[44] Yu Nesterov and Vladimir Shikhman. Quasi-monotone subgradient methods for nonsmooth convex minimization. Journal of Optimization Theory and Applications, 165(3):917–940, 2015.

[45] Yurii Nesterov. A method for solving the convex programming problem with convergence rate o (1/k2). In Dokl akad nauk Sssr, volume 269, page 543, 1983.

[46] Yurii Nesterov et al. Lectures on convex optimization, volume 137. Springer, 2018.

[47] Oleksiy Ostapenko, Mihai Marian Puscas, Tassilo Klein, Patrick Jähnichen, and Moin Nabi. Learning to remember: A synaptic plasticity driven framework for continual learning. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11313– 11321, 2019. URL https://api.semanticscholar.org/CorpusID:102353035.

[48] Boris T Polyak. New stochastic approximation type procedures. Automat. i Telemekh, 7(98-107): 2, 1990.

[49] Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn without forgetting by maximizing transfer and minimizing interference. In In International Conference on Learning Representations (ICLR), 2019.

[50] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Greg Wayne. Experience replay for continual learning. In Neural Information Processing Systems, 2018. URL https://api.semanticscholar.org/CorpusID:53860287.

[51] Sebastian Ruder. An overview of gradient descent optimization algorithms. ArXiv, abs/1609.04747, 2016. URL https://api.semanticscholar.org/CorpusID:17485266.

[52] David Ruppert. Efficient estimations from a slowly convergent robbins-monro process. 1988. URL https://api.semanticscholar.org/CorpusID:108279905.

[53] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. CoRR, abs/1409.0575, 2014. URL http://arxiv.org/abs/1409.0575.

[54] Gobinda Saha, Isha Garg, and Kaushik Roy. Gradient projection memory for continual learning. arXiv preprint arXiv:2103.09762, 2021.

[55] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends® in Machine Learning, 4(2):107–194, 2025.

[56] Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. ArXiv, abs/1212.1824, 2012. URL https://api.semanticscholar.org/CorpusID:1217704.

[57] Albin Soutif-Cormerais, Antonio Carta, Andrea Cossu, Julio Hurtado, Vincenzo Lomonaco, Joost Van de Weijer, and Hamed Hemati. A comprehensive empirical evaluation on online continual learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3518–3528, 2023.

[58] Albin Soutif-Cormerais, Antonio Carta, and Joost Van de Weijer. Improving online continual learning performance and stability with temporal ensembles. In Conference on Lifelong Learning Agents, pages 828–845. PMLR, 2023.

[59] Gido M. van de Ven and Andreas Savas Tolias. Three scenarios for continual learning. ArXiv, abs/1904.07734, 2019. URL https://api.semanticscholar.org/CorpusID: 119309522.

[60] Liyuan Wang, Mingtian Zhang, Zhongfan Jia, Qian Li, Chenglong Bao, Kaisheng Ma, Jun Zhu, and Yi Zhong. Afec: Active forgetting of negative transfer in continual learning. arXiv preprint arXiv:2110.12187, 2021.

[61] Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, and Jun Zhu. Hierarchical decomposition of prompt-based continual learning: Rethinking obscured sub-optimality. Advances in Neural Information Processing Systems, 2023.

[62] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. IEEE TPAMI, 46(8):5362–5383, 2024. doi: 10.1109/ TPAMI.2024.3367329.

[63] Maorong Wang, Nicolas Michel, Ling Xiao, and Toshihiko Yamasaki. Improving plasticity in online continual learning via collaborative learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23460–23469, June 2024.

[64] Quanziang Wang, Renzhen Wang, Yichen Wu, Xixi Jia, and Deyu Meng. Cba: Improving online continual learning via continual bias adaptor. In ICCV, 2023.

[65] Quanziang Wang, Renzhen Wang, Yuexiang Li, Dong Wei, Hong Wang, Kai Ma, Yefeng Zheng, and Deyu Meng. Relational experience replay: Continual learning by adaptively tuning task-wise relationship. IEEE TMM, 26:9683–9698, 2024. doi: 10.1109/TMM.2024.3397048.

[66] Shipeng Wang, Xiaorong Li, Jian Sun, and Zongben Xu. Training networks in null space of feature covariance for continual learning. In CVPR, pages 184–193, 2021.

[67] Yujie Wei, Jiaxin Ye, Zhizhong Huang, Junping Zhang, and Hongming Shan. Online prototype learning for online continual learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18764–18774, 2023.

[68] Yichen Wu, Long-Kai Huang, Renzhen Wang, Deyu Meng, and Ying Wei. Meta continual learning revisited: Implicitly enhancing online hessian approximation via variance reduction. In ICLR, 2024.

[69] Yichen Wu, Hong Wang, Peilin Zhao, Yefeng Zheng, Ying Wei, and Long-Kai Huang. Mitigating catastrophic forgetting in online continual learning by modeling previous task interrelations via pareto optimization. In ICML, 2024.

[70] Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma, and Ying Wei. Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning. arXiv preprint arXiv:2501.13198, 2025.

[71] Hongwei Yan, Liyuan Wang, Kaisheng Ma, and Yi Zhong. Orchestrate latent expertise: Advancing online continual learning with multi-level supervision and reverse self-distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23670–23680, 2024.

[72] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3013–3022, 2021. URL https://api.semanticscholar.org/ CorpusID:232427923.

[73] Jason Yoo, Yunpeng Liu, Frank Wood, and Geoff Pleiss. Layerwise proximal replay: A proximal point method for online continual learning. arXiv preprint arXiv:2402.09542, 2024.

[74] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pages 3987–3995. Pmlr, 2017.

[75] Michael R Zhang, James Lucas, Geoffrey Hinton, and Jimmy Ba. Lookahead optimizer: k steps forward, 1 step back. arXiv preprint arXiv:1907.08610, 2019.

[76] Da-Wei Zhou, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Classincremental learning: A survey. IEEE TPAMI, 46(12):9851–9873, 2024. doi: 10.1109/TPAMI. 2024.3429383.

[77] Minghao Zhou, Quanziang Wang, Jun Shu, Qian Zhao, and Deyu Meng. Diagnosing batch normalization in class incremental learning. arXiv preprint arXiv:2202.08025, 2022.

## Appendix

A Proofs 17   
A.1 Proof of Proposition 1 (Implicit Hessian) . 17   
A.2 Proof of Theorem 1 (Regret Bound) 18   
B An Intuitive Understanding of Two Averaging 19   
C pipeline 19   
D Implementation Details 20   
D.1 Dataset Details 20   
D.2 Evaluation Metrics 20   
D.3 Training Details . 21   
E Performance on Different Architecture 22   
F Analysis of Hyperparameter Robustness And Comparisons with SGD+EMA Baselines. 23   
G Relation to Multi-Sequence Optimizers 23   
H Detailed Experimental Setup for Sec. 4.3 24   
H.1 Setup for verifying the quasi-steady-state assumption . 24   
H.2 Setup for comparing update norms with Newton’s method 25   
I Algorithm Pseudocode 26

## A Proofs

## A.1 Proof of Proposition 1 (Implicit Hessian)

We analyze the local dynamics of the SPIN algorithm to verify the effective update rule presented in the main text (Proposition 1). Let $d _ { t } : = \phi _ { t } - \theta _ { t }$ denote the displacement between the fast (plastic) and slow (stable) iterates. Using the interpolation $x _ { t } = \theta _ { t } + ( 1 - \beta ) d _ { t }$ , we rewrite the system dynamics in terms of the variables $( \theta _ { t } , d _ { t } )$

$$
\begin{array} { r } { \theta _ { t + 1 } - \theta _ { t } = \alpha ( d _ { t } - \gamma g _ { t } ) , \quad \quad } \\ { d _ { t + 1 } = ( 1 - \alpha ) ( d _ { t } - \gamma g _ { t } ) , \quad \quad } \end{array}\tag{14}
$$

where $g _ { t } = \nabla \mathcal { L } _ { t } ( x _ { t } )$

Local Linearization. To expose the curvature dependence, we perform a first-order Taylor expansion of the gradient around the slow iterate $\theta _ { t }$ . Let $H _ { t } = \nabla ^ { 2 } \dot { \mathcal { L } } _ { t } ( \theta _ { t } )$ ) denote the local Hessian. Neglecting higher-order terms, we approximate the gradient at the interpolated point $x _ { t }$ as:

$$
g _ { t } = \nabla \mathcal { L } _ { t } ( \theta _ { t } + ( 1 - \beta ) d _ { t } ) \approx \nabla \mathcal { L } _ { t } ( \theta _ { t } ) + ( 1 - \beta ) H _ { t } d _ { t } .\tag{15}
$$

Substituting this linear approximation into the evolution equation for $d _ { t }$ in Eq. 14 yields:

$$
d _ { t + 1 } \approx ( 1 - \alpha ) \left[ d _ { t } - \gamma \left( \nabla \mathcal { L } _ { t } ( \theta _ { t } ) + ( 1 - \beta ) H _ { t } d _ { t } \right) \right] .\tag{16}
$$

Quasi-Steady State Analysis. Since SPIN is designed as a two-time-scale system where the fast variate $\phi _ { t }$ (and consequently $d _ { t } )$ evolves more rapidly than $\theta _ { t } ,$ we apply a quasi-steady-state approximation [3]. We solve for the equilibrium displacement $d _ { t } ^ { * }$ by imposing the stationarity condition $d _ { t + 1 } \approx d _ { t }$

$$
d _ { t } ^ { * } = ( 1 - \alpha ) d _ { t } ^ { * } - ( 1 - \alpha ) \gamma \nabla \mathcal { L } _ { t } ( \theta _ { t } ) - ( 1 - \alpha ) \gamma ( 1 - \beta ) H _ { t } d _ { t } ^ { * } .\tag{17}
$$

Rearranging terms to isolate $d _ { t } ^ { * }$ , we obtain:

$$
\left[ \alpha I + ( 1 - \alpha ) \gamma ( 1 - \beta ) H _ { t } \right] d _ { t } ^ { \ast } = - ( 1 - \alpha ) \gamma \nabla \mathcal { L } _ { t } ( \theta _ { t } ) .\tag{18}
$$

Solving for $d _ { t } ^ { * }$ gives the explicit form:

$$
d _ { t } ^ { * } = - ( 1 - \alpha ) \gamma \left[ \alpha I + ( 1 - \alpha ) \gamma ( 1 - \beta ) H _ { t } \right] ^ { - 1 } \nabla \mathcal { L } _ { t } ( \theta _ { t } ) .\tag{19}
$$

Effective Update Rule. We now substitute the equilibrium displacement $d _ { t } ^ { * }$ back into the update rule for the slow iterate $\theta _ { t }$ in Eq. 14. Note that the update term is $\overline { { \alpha ( d _ { t } - \gamma g _ { t } ) } } ^ { + }$ . From the stationarity condition $d _ { t } \approx ( 1 - \alpha ) ( d _ { t } - \gamma g _ { t } )$ , we have the identity $\begin{array} { r } { d _ { t } - \gamma g _ { t } \overset { . } { \approx } \frac { 1 } { 1 - \alpha } \overset { . } { d } _ { t } } \end{array}$ . Therefore, the update becomes:

$$
\theta _ { t + 1 } - \theta _ { t } \approx \frac { \alpha } { 1 - \alpha } d _ { t } ^ { * } = - \alpha \gamma \left[ \alpha I + ( 1 - \alpha ) \gamma ( 1 - \beta ) H _ { t } \right] ^ { - 1 } \nabla \mathcal { L } _ { t } ( \theta _ { t } ) .\tag{20}
$$

To align with the form presented in the main text, we factorize α from the inverse term. We define the effective regularization coefficient $\rho _ { t }$ as:

$$
\rho _ { t } = \frac { \gamma ( 1 - \alpha ) ( 1 - \beta ) } { \alpha } .\tag{21}
$$

Using this definition, we can rewrite the inverse term as:

$$
[ \alpha I + ( 1 - \alpha ) \gamma ( 1 - \beta ) H _ { t } ] ^ { - 1 } = \frac { 1 } { \alpha } \left[ I + \frac { \gamma ( 1 - \alpha ) ( 1 - \beta ) } { \alpha } H _ { t } \right] ^ { - 1 } = \frac { 1 } { \alpha } [ I + \rho _ { t } H _ { t } ] ^ { - 1 } .\tag{22}
$$

Substituting this back into the update equation, the α terms cancel out, yielding the final effective update:

$$
\theta _ { t + 1 } \approx \theta _ { t } - \gamma \left[ I + \rho _ { t } H _ { t } \right] ^ { - 1 } \nabla \mathcal { L } _ { t } ( \theta _ { t } ) .\tag{23}
$$

Connection to Preconditioning. For sufficiently small step sizes, applying the Neumann series approximation $( I + A ) ^ { - 1 } \approx I - A$ yields:

$$
\begin{array} { r } { \theta _ { t + 1 } \approx \theta _ { t } - \gamma \left( I - \rho _ { t } H _ { t } \right) \nabla \mathcal { L } _ { t } ( \theta _ { t } ) . } \end{array}\tag{24}
$$

This confirms that SPIN performs a gradient descent step preconditioned by the implicit inverse Hessian approximation $\tilde { H } ^ { - 1 } \approx \gamma ( I + \rho _ { t } H _ { t } ) ^ { - 1 }$

## A.2 Proof of Theorem 1 (Regret Bound)

Theorem 1. Let $\mathcal { X } \subset \mathbb { R } ^ { d }$ be a closed convex set with diameter D. Under Assumptions $( i ) – ( i \nu )$ , if we choose $\begin{array} { r } { \gamma = \frac { D } { G \sqrt T } \sqrt { \frac { \alpha } { 2 - \alpha } } } \end{array}$ and initialize $\theta _ { 1 } = \phi _ { 1 }$ , the regret R<sub>T</sub> satisfies the following bound with probability at least $1 \stackrel { } { - } \delta :$

$$
R _ { T } \leq \tilde { \mathcal { O } } ( \sqrt { T } ) .\tag{25}
$$

We consider online convex optimization [55, 20] with the update rule defined as:

$$
x _ { t } = ( 1 - \beta ) \phi _ { t } + \beta \theta _ { t } ,\tag{26}
$$

$$
\begin{array} { r } { \phi _ { t + 1 } = \Pi _ { \mathcal { X } } \big ( \phi _ { t } - \gamma g _ { t } \big ) , \qquad g _ { t } : = \nabla \mathcal { L } _ { t } ( x _ { t } ) , } \end{array}\tag{27}
$$

$$
\theta _ { t + 1 } = \theta _ { t } + \alpha ( \phi _ { t + 1 } - \theta _ { t } ) ,\tag{28}
$$

where $\alpha \in ( 0 , 1 )$ is a constant. We let $\mathcal { F } _ { t }$ denote the filtration generated by the random variables up to round t.

For the analysis, we adopt the following standard assumptions:

(i) Domain: $\mathcal { X }$ is a closed convex set with diameter bounded by $\begin{array} { r } { D , \mathrm { i . e . , s u p } _ { x , y \in \mathcal { X } } \left\| x - y \right\| \leq D } \end{array}$

(ii) Loss Function: Each loss function $\mathcal { L } _ { t }$ is convex and G-Lipschitz continuous on $\mathcal { X }$

(iii) Stochastic Gradient: The stochastic gradients $g _ { t }$ are unbiased, $\mathbb { E } [ g _ { t } \mid \mathcal { F } _ { t } ] = \nabla \mathcal { L } _ { t } ( x _ { t } )$ , and bounded such that $\| g _ { t } \| \leq G$ almost surely.

(iv) Variance: The conditional variance of the stochastic gradients is bounded by $\sigma ^ { 2 }$ , i.e., $\mathbb { E } [ \Vert g _ { t } - \nabla \mathcal { L } _ { t } ( x _ { t } ) \Vert ^ { 2 } \mid \mathcal { F } _ { t } ] \leq \sigma ^ { 2 }$

We aim to bound the static regret $\begin{array} { r } { R _ { T } : = \sum _ { t = 1 } ^ { T } ( \mathcal { L } _ { t } ( \theta _ { t } ) - \mathcal { L } _ { t } ( \theta ^ { \star } ) ) } \end{array}$

1. Regret Decomposition. By the convexity of $\mathcal { L } _ { t }$ , we have $\mathcal { L } _ { t } ( x _ { t } ) - \mathcal { L } _ { t } ( \theta ^ { \star } ) \leq \langle \nabla \mathcal { L } _ { t } ( x _ { t } ) , x _ { t } -$ $\theta ^ { \star } \big \rangle$ [20]. We decompose the instantaneous regret as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { t } ( \theta _ { t } ) - \mathcal { L } _ { t } ( \theta ^ { \star } ) = \big ( \mathcal { L } _ { t } ( \theta _ { t } ) - \mathcal { L } _ { t } ( x _ { t } ) \big ) + \big ( \mathcal { L } _ { t } ( x _ { t } ) - \mathcal { L } _ { t } ( \theta ^ { \star } ) \big ) } \\ & { \qquad \leq G \| \theta _ { t } - x _ { t } \| + \langle \nabla \mathcal { L } _ { t } ( x _ { t } ) , x _ { t } - \theta ^ { \star } \rangle . } \end{array}\tag{29}
$$

Using the definition of $x _ { t }$ in Eq. 26, we have $\lVert \theta _ { t } - x _ { t } \rVert = ( 1 - \beta ) \lVert \theta _ { t } - \phi _ { t } \rVert$ . Next, we introduce the martingale difference sequence $\xi _ { t } : = \langle \nabla { \mathcal L } _ { t } ( x _ { t } ) - g _ { t } , x _ { t } - \theta ^ { \star } \rangle$ . We rewrite the inner product term using the stochastic gradient $g _ { t } \mathrm { . }$

$$
\langle \nabla { \mathcal { L } } _ { t } ( x _ { t } ) , x _ { t } - \theta ^ { \star } \rangle = \langle g _ { t } , x _ { t } - \theta ^ { \star } \rangle + \xi _ { t } = \langle g _ { t } , \phi _ { t } - \theta ^ { \star } \rangle + \langle g _ { t } , x _ { t } - \phi _ { t } \rangle + \xi _ { t } .
$$

Noting that $x _ { t } - \phi _ { t } = \beta ( \theta _ { t } - \phi _ { t } )$ and $\| g _ { t } \| \leq G$ (Assumption iii), we bound $\langle g _ { t } , x _ { t } - \phi _ { t } \rangle \leq$ $G \beta \| \bar { \theta _ { t } } - \phi _ { t } \|$ . Combining these terms yields:

$$
\begin{array} { r } { \mathcal { L } _ { t } ( \boldsymbol { \theta } _ { t } ) - \mathcal { L } _ { t } ( \boldsymbol { \theta } ^ { \star } ) \leq \langle g _ { t } , \phi _ { t } - \boldsymbol { \theta } ^ { \star } \rangle + G \lVert \theta _ { t } - \phi _ { t } \rVert + \xi _ { t } . } \end{array}\tag{30}
$$

2. Bounding the Gradient Descent Term. The update for $\phi _ { t }$ corresponds to Online Gradient Descent (OGD) [20]. By the standard analysis of OGD and the non-expansiveness of the projection operator $\Pi _ { \mathcal { X } }$ , we have:

$$
\langle g _ { t } , \phi _ { t } - \theta ^ { \star } \rangle \leq \frac { \| \phi _ { t } - \theta ^ { \star } \| ^ { 2 } - \| \phi _ { t + 1 } - \theta ^ { \star } \| ^ { 2 } } { 2 \gamma } + \frac { \gamma } { 2 } \| g _ { t } \| ^ { 2 } .\tag{31}
$$

3. Bounding the Drift Term. Let $\Delta _ { t } = \theta _ { t } - \phi _ { t }$ . From the update rule $\theta _ { t + 1 } = ( 1 - \alpha ) \theta _ { t } + \alpha \phi _ { t + 1 }$ the recurrence for the difference is:

$$
\Delta _ { t + 1 } = ( 1 - \alpha ) \theta _ { t } + \alpha \phi _ { t + 1 } - \phi _ { t + 1 } = ( 1 - \alpha ) ( \theta _ { t } - \phi _ { t + 1 } ) .\tag{32}
$$

Taking norms and applying the triangle inequality with $\lVert \phi _ { t } - \phi _ { t + 1 } \rVert \leq \gamma G \colon$

$$
\| \Delta _ { t + 1 } \| \leq ( 1 - \alpha ) \| \Delta _ { t } \| + ( 1 - \alpha ) \gamma G .\tag{33}
$$

Summing this geometric series over $t = 1 , \dots , T$ yields:

$$
\sum _ { t = 1 } ^ { T } \| \theta _ { t } - \phi _ { t } \| \leq \frac { \| \theta _ { 1 } - \phi _ { 1 } \| } { \alpha } + T \frac { 1 - \alpha } { \alpha } \gamma G .\tag{34}
$$

![](images/dbb9870e95d7bdef6eefaec99514529d46b4d4f0999fd5684a1bf97ab7c28954.jpg)  
(a) Polyak-Ruppert averaging.

![](images/717f3e3591ea169793fc26befdab27ac48a884690051b574156e183caeb4d77e.jpg)  
(b) Primal averaging.  
Figure 6: Trajectories of Polyak-Ruppert Averaging and Primal Averaging. Polyak-Ruppert Averaging computes the gradient at the current, fast-moving model iterate $\theta _ { t } \colon g _ { t } = \nabla \mathcal { L } ( \theta _ { t } )$ ). The final model is the average of this noisy sequence: $\begin{array} { r } { \bar { \theta } _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \theta _ { t } } \end{array}$ . Primal Averaging computes the gradient at the averaged iterate over the current trajectory: $g _ { t } = \nabla \mathcal { L } ( \bar { \theta } _ { t } )$ , where $\begin{array} { r } { \bar { \theta } _ { t } = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \theta _ { i } } \end{array}$ . This stable gradient is then used to guide the update.

4. Concentration Analysis. Since $\| x _ { t } - \theta ^ { \star } \| \leq D$ (Assumption i) and gradients are bounded by $G ,$ , the martingale difference is bounded as $| \xi _ { t } | \leq 2 G D$ . The conditional variance is bounded by $\mathbb { E } [ \xi _ { t } ^ { 2 } \mid \mathcal { F } _ { t } ] \le \breve { D } ^ { 2 } \sigma ^ { 2 }$ (Assumption iv). Applying Freedman’s inequality [15], with probability at least $1 - \delta \colon$

$$
\sum _ { t = 1 } ^ { T } \xi _ { t } \leq D \sigma \sqrt { 2 T \ln \frac { 1 } { \delta } } + \frac { 2 G D } { 3 } \ln \frac { 1 } { \delta } .\tag{35}
$$

5. Final Bound. Summing Eq. 30 over $t = 1$ to $T$ and substituting the bounds derived above:

$$
R _ { T } \leq \frac { G \| \theta _ { 1 } - \phi _ { 1 } \| } { \alpha } + \left[ \frac { D ^ { 2 } } { 2 \gamma } + \frac { \gamma T G ^ { 2 } } { 2 } + \gamma T G ^ { 2 } \frac { 1 - \alpha } { \alpha } \right] + \sum _ { t = 1 } ^ { T } \xi _ { t } .\tag{36}
$$

Substituting the optimal step size $\begin{array} { r } { \gamma ^ { \star } = \frac { D } { G \sqrt T } \sqrt { \frac { \alpha } { 2 - \alpha } } } \end{array}$ , the term in brackets simplifies to $D G \sqrt { T } \sqrt { \frac { 2 - \alpha } { \alpha } }$ Thus, with probability $\geq 1 - \delta \colon$

$$
\begin{array} { r l } { { R _ { T } } \displaystyle \leq \frac { G \| \theta _ { 1 } - \phi _ { 1 } \| } { \alpha } + D G \sqrt { T } \sqrt { \frac { 2 - \alpha } { \alpha } } } & { { } + D \sigma \sqrt { 2 T \ln \frac { 1 } { \delta } } + \frac { 2 G D } { 3 } \ln \frac { 1 } { \delta } . } \end{array}\tag{37}
$$

Assuming the initialization $\theta _ { 1 } = \phi _ { 1 }$ , the first term vanishes, and we obtain the final asymptotic bound:

$$
R _ { T } \leq \tilde { \mathcal { O } } ( \sqrt { T } ) .\tag{38}
$$

Conclusion. Theorem 1 establishes that the proposed algorithm achieves sublinear regret matching the optimal $\tilde { \mathcal { O } } ( \sqrt { T } )$ lower bound for stochastic OCO. The constant $\sqrt { ( 2 - \alpha ) / \alpha }$ highlights the tradeoff between the memory retention of the slow weights (parameterized by α) and the convergence speed.

## B An Intuitive Understanding of Two Averaging

We present these two averaging schemes, as shown in Figure 6.

## C pipeline

We present the pipeline of SPIN in Figure 7.

![](images/d0962d174c3ce1575759e6d882053ba3b7fbf8bd3168bad083881d449123aec4.jpg)  
Figure 7: The pipeline of SPIN. The gradient ∇L at the model interpolation is obtained from the upstream design of OCL algorithms.

## D Implementation Details

## D.1 Dataset Details

In this work, we experiment with five benchmark datasets in online continual learning setting: CIFAR10 [28], CIFAR100 [28], Tiny-Imagenet [31], CLEAR100 [36], and Imagenet1K [53], which are detailed as follows:

Seq-CIFAR10 [28]. Seq-CIFAR10 is a widely used benchmark in Online Continual Learning (OCL), constructed by splitting the CIFAR-10 dataset into sequential tasks. It contains 10 classes and 60,000 RGB images of size 32 × 32, and is commonly used to evaluate catastrophic forgetting under class-incremental learning settings.

Seq-CIFAR100 [28]. Seq-CIFAR100 is a continual learning benchmark derived from CIFAR-100, consisting of 100 fine-grained categories and 60,000 images. Due to its larger number of classes and more complex task splits, it provides a more challenging setting for evaluating the stability and generalization ability of OCL methods.

Tiny-ImageNet [31]. Tiny-ImageNet is a reduced version of ImageNet containing 200 classes with 64 × 64 resolution images. Its higher visual diversity and semantic complexity make it a popular benchmark for testing representation learning and forgetting resistance in online continual learning scenarios.

CLEAR100 [36]. CLEAR100 is specifically designed for online continual learning and emphasizes realistic streaming data settings. It includes distribution shifts, sequential data streams, and limited replay constraints, making it suitable for evaluating adaptability and robustness in dynamic learning environments.

ImageNet-1K [53]. ImageNet-1K is one of the most influential large-scale image classification datasets, containing 1,000 categories and over one million high-resolution images. In OCL research, it is commonly used to build large-scale class-incremental or streaming benchmarks for evaluating long-term memory retention and continual adaptation.

## D.2 Evaluation Metrics

We adopt several metrics commonly used in online continual learning. Let $a _ { i , j }$ denote the accuracy of the model after learning task i on the test data of task $j$

Average Accuracy (Acc). Acc measures the final performance after completing all tasks:

$$
\operatorname { A c c } = \frac { 1 } { T } \sum _ { j = 1 } ^ { T } a _ { T , j } .\tag{39}
$$

Averaged Anytime Accuracy (AAA). AAA evaluates the performance throughout the learning process:

$$
\mathrm { A A A } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \left( \frac { 1 } { i } \sum _ { j = 1 } ^ { i } a _ { i , j } \right) .\tag{40}
$$

Learning Accuracy (LA). LA quantifies plasticity as the accuracy on each task immediately after it is learned:

$$
l _ { j } = a _ { j , j } ,\tag{41}
$$

and the overall learning accuracy is

$$
\mathrm { L A } = \frac { 1 } { T } \sum _ { j = 1 } ^ { T } l _ { j } .\tag{42}
$$

Average Forgetting (AF). AF measures the degradation on old tasks:

$$
\mathrm { A F } = \frac { 1 } { T - 1 } \sum _ { j = 1 } ^ { T - 1 } \left( \operatorname* { m a x } _ { k \le T } a _ { k , j } - a _ { T , j } \right) .\tag{43}
$$

Relative Forgetting (RF). Relative Forgetting measures the proportion of performance degradation on previous tasks. For the j-th task after learning task k, it is defined as

$$
f _ { j } ^ { k } = \operatorname* { m a x } _ { i \in \{ 1 , \dots , k \} } \left( 1 - \frac { a _ { j } ^ { k } } { a _ { j } ^ { i } } \right) , \quad j \leq k .\tag{44}
$$

The overall relative forgetting averaged across all tasks is

$$
\mathrm { R F } = \frac { 1 } { T } \sum _ { j = 1 } ^ { T } f _ { j } ^ { T } .\tag{45}
$$

## D.3 Training Details

For all baselines except MOSE [71] and SCR [40], we follow the standard experimental configuration used in CCL-DC [63]. All methods adopt a randomly initialized ResNet-18 backbone without pretraining. The model operates in a strict one-pass online continual learning setting, where each training sample is observed exactly once. The streaming batch size is fixed to 10, and at every iteration a mini-batch of 64 samples is drawn from the replay buffer. The memory buffer is maintained using reservoir sampling, and identical data augmentation is applied across all baselines to ensure fairness.

MOSE and SCR follow the experimental setup of MOSE. Both methods use a nearest-class-mean (NCM) classifier for final prediction. The same online protocol is applied, including a streaming batch size of 10, a replay batch size of 64, and reservoir-based memory management. All experiments were conducted using one NVIDIA RTX 4090 GPU.

Learning rates for the 11 OCL methods are summarized in Table 11. Note that in most of our experiments, SPIN can replace the existing optimizer without any additional adjustments.

Data Augmentation. Data augmentation has been shown to effectively improve the performance of online continual learning (OCL) methods [6, 63]. Since different methods may respond differently to augmentation strength, we consider two commonly used augmentation strategies to ensure a fair and comprehensive comparison.

1. Partial Augmentation Strategy. This strategy adopts relatively weak data augmentation, including random cropping with probability $p = 0 . 5$ and random horizontal flipping with probability $p = 0 . 5$

2. Full Augmentation Strategy. The full augmentation strategy extends the partial augmentation setting by additionally incorporating color jitter and random grayscale transformations. Specifically, the color jitter parameters are set to (0.4, 0.4, 0.4, 0.1) with probability $p = 0 . 8 .$ while random grayscale is applied with probability $p = 0 . 2$

Table 11: Learning rates for each OCL method and optimizer.
<table><tr><td>Method</td><td>SGD</td><td>SPIN-SGD</td><td>AdamW</td><td>SPIN-AdamW</td></tr><tr><td>ER</td><td>0.01</td><td>0.01</td><td>0.0005</td><td>0.0005</td></tr><tr><td>DER++</td><td>0.03</td><td>0.03</td><td>0.0005</td><td>0.0005</td></tr><tr><td>COPE</td><td>0.03</td><td>0.03</td><td>0.0005</td><td>0.0005</td></tr><tr><td>SCR</td><td>0.05</td><td>0.05</td><td>0.005</td><td>0.005</td></tr><tr><td>ER-ACE</td><td>0.01</td><td>0.01</td><td>0.0005</td><td>0.0005</td></tr><tr><td>OCM</td><td>0.05</td><td>0.05</td><td>0.001</td><td>0.001</td></tr><tr><td>OnPro</td><td>0.03</td><td>0.03</td><td>0.0005</td><td>0.0005</td></tr><tr><td>GSA</td><td>0.03</td><td>0.03</td><td>0.0005</td><td>0.0005</td></tr><tr><td>MOSE</td><td>0.05</td><td>0.05</td><td>0.001</td><td>0.001</td></tr><tr><td>CCL-DC</td><td>0.03</td><td>0.03</td><td>0.0005</td><td>0.0005</td></tr><tr><td>S6MOD</td><td>0.03</td><td>0.03</td><td>0.0005</td><td>0.0005</td></tr></table>

Table 12: Hyperparameter for part of baseline methods.
<table><tr><td>Method</td><td>HP</td><td>Values</td></tr><tr><td rowspan="6">ER</td><td>optimizer</td><td>[SGD, AdamW, SPIN, RMSProp, Adam]</td></tr><tr><td>lr</td><td>[0.1, 0.3, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]</td></tr><tr><td>weight decay</td><td>[1e-4]</td></tr><tr><td>momentum (for SGD)</td><td>[0.9]</td></tr><tr><td>aug. strat.</td><td>[partial, full]</td></tr><tr><td>optimizer</td><td></td></tr><tr><td rowspan="6">DER++</td><td>lr</td><td>[SGD, AdamW, SPIN, RMSProp, Adam] [0.1, 0.3, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]</td></tr><tr><td>weight decay</td><td>[0, 1e-4]</td></tr><tr><td>momentum (for SGD)</td><td>[0, 0.9]</td></tr><tr><td>aug. strat.</td><td>[partial, full]</td></tr><tr><td>alpha</td><td></td></tr><tr><td>beta</td><td>[0.2] [1.0]</td></tr><tr><td rowspan="6">GSA</td><td>optimizer</td><td>[SGD, AdamW, SPIN, RMSProp, Adam]</td></tr><tr><td>lr</td><td>[0.1, 0.3, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]</td></tr><tr><td>weight decay</td><td>[1e-4]</td></tr><tr><td>momentum (for SGD)</td><td>[0.9]</td></tr><tr><td>aug. strat.</td><td></td></tr><tr><td></td><td>[partial, full]</td></tr></table>

For fair comparison, the models trained with SPIN also adopt the same augmentation strategy in the baseline loss branch. Note that the overall augmentation setting is kept consistent with CCL-DC [63]. We present hyperparameter searches for some of the methods in Table 12.

## E Performance on Different Architecture

To further evaluate the robustness of our method across different model architectures, we additionally conduct experiments on a Transformer-based backbone. Specifically, we compare the optimization performance of AdamW and our SPIN under the Vision Transformer (ViT) architecture using the GSA framework. In these experiments, the memory buffer size is set to 2000 and the learning rate is fixed at 0.0001. The results demonstrate that our method consistently improves performance across different optimization landscapes and network structures, highlighting its generalization capability in online continual learning settings. Note that ResNet-18 is the standard backbone in OCL, while ViT remains largely unexplored.

Table 13: Optimization performance on the ViT architecture.
<table><tr><td>Metric</td><td>AdamW</td><td>SPIN</td></tr><tr><td>Acc (%)</td><td>25.89</td><td>26.72</td></tr><tr><td>AAA (%)</td><td>33.65</td><td>34.98</td></tr></table>

![](images/97444dab08fddeb05110b835e6add6c9ee13b85a9d2ad30f305bb5dd153ed943.jpg)  
Figure 8: Analysis of hyperparameter robustness and comparisons with different baselines.

## F Analysis of Hyperparameter Robustness And Comparisons with SGD+EMA Baselines.

We evaluate SPIN’s sensitivity to α and β in Figure 8. Even randomly selected $( \alpha , \beta )$ pairs consistently outperform standard SGD. Specifically, fixing one parameter to an empirical default (e.g. , $\alpha = 0 . 0 5$ or $\beta = 0 . 0 5 )$ while tuning the other within [0.01, 0.99] yields peak performance, allowing SPIN to consistently surpass the tuned EMA baseline. This superiority arises because EMA effectively operates as a subset of SPIN. Finally, ScheduleFree performs poorly in this context, as its equal-weighted scheduling implicitly assumes a stationary distribution, which mismatches the strictly non-stationary nature of OCL.

## G Relation to Multi-Sequence Optimizers

We discuss how multi-sequence optimizers choose gradient evaluation points and average iterates. Throughout this section, z<sub>t</sub> denotes the base iterate, $x _ { t }$ the averaged iterate, and $y _ { t }$ the gradient evaluation point. The formulas below use gradient-descent updates without weight decay to isolate the averaging mechanisms.

Lookahead. Lookahead [75] performs k base-optimizer steps on fast weights, interpolates the slow weights toward the resulting fast weights, and resets the fast weights to the updated slow weights. With SGD as the base optimizer and $z _ { t , 0 } = x _ { t }$ , these updates are

$$
z _ { t , i + 1 } = z _ { t , i } - \gamma _ { t , i } g _ { t , i } ( z _ { t , i } ) , \quad i = 0 , \ldots , k - 1 ,\tag{46}
$$

$$
x _ { t + 1 } = ( 1 - \lambda ) x _ { t } + \lambda z _ { t , k } ,\tag{47}
$$

$$
z _ { t + 1 , 0 } = x _ { t + 1 } .\tag{48}
$$

Here, t indexes synchronization rounds, i indexes inner steps, $g _ { t , i }$ is the stochastic gradient at that step, and $\lambda \in ( 0 , 1 ]$ is the slow-weight interpolation coefficient. Its averaging is periodic; the methods below instead couple their sequences at every step.

AC-SA form. For an L-smooth convex objective $f ,$ the accelerated formulation discussed in the main text can be written as [45, 29, 12]

$$
y _ { t } = ( 1 - c _ { t + 1 } ) x _ { t } + c _ { t + 1 } z _ { t } ,\tag{49}
$$

$$
\boldsymbol { z } _ { t + 1 } = \boldsymbol { z } _ { t } - \frac { t + 1 } { 2 L } \nabla f ( \boldsymbol { y } _ { t } ) ,\tag{50}
$$

$$
x _ { t + 1 } = ( 1 - c _ { t + 1 } ) x _ { t } + c _ { t + 1 } z _ { t + 1 } ,\tag{51}
$$

where $c _ { t + 1 } = 2 / ( t + 2 )$ . The same time-dependent coefficient controls both interpolations, together with a growing base step size. This coupling is motivated by accelerated convergence for a fixed objective.

Schedule-Free optimization. The basic uniform-averaging version of Schedule-Free SGD takes the form [12]

$$
y _ { t } = ( 1 - \beta ) z _ { t } + \beta x _ { t } ,\tag{52}
$$

$$
z _ { t + 1 } = z _ { t } - \gamma _ { t } g _ { t } ( y _ { t } ) ,\tag{53}
$$

$$
x _ { t + 1 } = { \frac { t } { t + 1 } } x _ { t } + { \frac { 1 } { t + 1 } } z _ { t + 1 } ,\tag{54}
$$

where $g _ { t } ( y _ { t } )$ is a stochastic gradient. Unlike the AC-SA form above, the gradient interpolation coefficient is fixed while the averaging coefficient decreases. The framework also supports weighted averages, and nonconvex guarantees have been established for Schedule-Free SGD [1]. In OCL, a decreasing averaging coefficient may reduce responsiveness to newly arriving data; its effect on retention and adaptation requires empirical assessment.

Generalized Primal Averaging (GPA). GPA [13] uses two independent interpolation coefficients:

$$
y _ { t } = \mu _ { y } x _ { t } + ( 1 - \mu _ { y } ) z _ { t } ,\tag{55}
$$

$$
z _ { t + 1 } = z _ { t } - \gamma _ { t } g _ { t } ( y _ { t } ) ,\tag{56}
$$

$$
x _ { t + 1 } = \mu _ { x } x _ { t } + ( 1 - \mu _ { x } ) z _ { t + 1 } .\tag{57}
$$

Here, $\mu _ { y }$ controls the gradient evaluation point, whereas $\mu _ { x }$ controls the persistence of the averaged sequence. Expanding the last recursion gives

$$
x _ { t + 1 } = \mu _ { x } ^ { t + 1 } x _ { 0 } + \left( 1 - \mu _ { x } \right) \sum _ { j = 0 } ^ { t } \mu _ { x } ^ { t - j } z _ { j + 1 } .\tag{58}
$$

Thus, GPA assigns exponentially decaying weights to past base iterates, with a constant weight on each new iterate. GPA was developed to smooth the periodic updates of single-worker DiLoCo and improve training efficiency. Its fixed averaging coefficient does not automatically inherit the schedule-free property of cumulative averaging.

Perspective for OCL. Our focus is the behavior of these dynamics under non-stationary, singlepass learning: how gradient interpolation and averaging balance acquisition of new knowledge with retention of previous tasks. Motivated by the stability–plasticity dilemma, SPIN interprets the fast sequence as adaptation and the slow sequence as progressive consolidation. The curvature interpretation in Appendix 4.2 and the accuracy–forgetting analysis in Appendix 5.3 examine these questions in the OCL setting.

## H Detailed Experimental Setup for Sec. 4.3

## H.1 Setup for verifying the quasi-steady-state assumption

To validate the quasi-steady-state (QSS) assumption used in our derivation, we measure

$$
r _ { t } = \frac { \| d _ { t + 1 } \| } { \| d _ { t } \| } ,\tag{59}
$$

where ${ d _ { t } } = { \phi _ { t } } - { \theta _ { t } }$ is the gap between the plasticity sequence and the stability sequence.

We use the ER method as the base OCL framework and keep the setup fully consistent with the main experiments. The experiment is conducted on Split CIFAR-100, with 10 tasks and 10 classes per task. The backbone is ResNet-18. The stream batch size is 10, the replay batch size is $^ { 6 4 , }$ and the replay buffer size is 2000. We use SPIN-AdamW with $\alpha = 0 . 0 5$ and $\beta = 0 . 0 5$ . All remaining hyperparameters are the same as those used in the corresponding main experiments.

During training, we record $r _ { t }$ at every optimization step. The resulting curve is reported in Figure 2a of the main paper.

## H.2 Setup for comparing update norms with Newton’s method

To further validate the curvature-related interpretation of SPIN, we compare the $\ell _ { 2 }$ norm of its update with that of a Newton-type method.

Because computing Newton updates on CIFAR-100 with a ResNet-18 backbone is computationally too expensive due to the high dimensionality of the model parameters and the cost of Hessian computation, we instead perform this experiment on MNIST. This choice allows efficient computation of second-order information while preserving the purpose of the experiment, namely, comparing update norm behavior.

We train a small multilayer perceptron (MLP) classifier on MNIST using the standard cross-entropy loss. The network consists of two fully connected hidden layers with 256 units each and ReLU activations. The batch size is set to 128, and the model is trained for 20 epochs.

We compare two optimization methods:

• SPIN: our proposed optimizer built on top of SGD.

• Newton: a Newton-type method that computes updates using the gradient and the Hessian of the current mini-batch loss.

For SPIN, we use a base learning rate of 0.01, with hyperparameters $\alpha = 0 . 0 5$ and $\beta = 0 . 0 5$ , which are consistent with the settings used in the main experiments.

At each training step $t ,$ we compute the parameter update vector $\Delta _ { t }$ produced by each method and record its $\ell _ { 2 }$ norm $\| \Delta _ { t } \| _ { 2 }$ . For $\bar { \mathrm { S P I N } } , \Delta _ { t } ^ { - }$ denotes the actual update applied by the optimizer to the model parameters. For the Newton method, $\Delta _ { t }$ corresponds to the second-order update computed from the gradient and Hessian.

Both methods use the same data order and training objective to ensure a fair comparison. Since the purpose of this experiment is theoretical validation rather than performance benchmarking, we focus on the evolution of the update norm during training instead of the final classification accuracy. The resulting comparison is shown in fig. 2b of the main paper.

## I Algorithm Pseudocode

Algorithm 1: SPIN-SGD for OCL   
Input: Task stream $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T } ;$ replay buffer capacity $\mathcal { M } ;$ interpolation coefficient $\beta ;$ stability   
merging coefficient α; learning rate $\gamma ;$ weight decay $\bar { \lambda . }$   
Output: Final stability parameters $\theta _ { S }$   
Initialize network parameters $\theta _ { 1 } ;$ set $\phi _ { 1 }  \theta _ { 1 }$   
Initialize replay buffer ${ \mathcal { M } } \gets \emptyset .$   
Initialize global step counter $s \gets 1$   
for t = 1 to $T$ do   
for each incoming mini-batch $B ^ { s t r e a m } \subset { \mathcal { D } } _ { t }$ do   
Sample replay batch $B ^ { \mathcal { M } } \sim \mathcal { M }$ if M is not empty.   
Construct training batch $B _ { s } \gets B ^ { \mathrm { s t r e a m } } \cup B ^ { M }$   
Compute interpolated parameters $x _ { s } \gets ( 1 - \beta ) \phi _ { s } + \beta \theta _ { s }$   
Compute loss $\bar { \mathcal { L } } _ { s }$ on $B _ { s }$ using the upstream OCL method with parameters $x _ { s } .$   
Compute stochastic gradient $g _ { s } \gets \nabla { \mathcal { L } } _ { s } ( x _ { s } )$   
$/ /$ Plasticity update   
$\phi _ { s + 1 }  \phi _ { s } - \gamma ( g _ { s } + \lambda x _ { s } )$   
$/ /$ Stability update   
$\theta _ { s + 1 }  \theta _ { s } + \alpha ( \phi _ { s + 1 } - \theta _ { s } )$   
Update replay buffer M with $B ^ { \mathrm { s t r e a m } }$   
$s \gets s + 1$   
end   
end   
return $\theta _ { S }$