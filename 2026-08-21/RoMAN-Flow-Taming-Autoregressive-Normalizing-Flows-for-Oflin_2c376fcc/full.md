# RoMAN-Flow: Taming Autoregressive Normalizing Flows for Ofline Reinforcement Learning in Robotic Manipulation

Shaoxuan Wang<sup>1∗</sup>, Guangting Zheng<sup>1∗</sup>, Rui Huang<sup>1</sup>, Zhipeng Tang<sup>1</sup>, Sha Zhang<sup>2</sup>, Jiajun Deng<sup>1</sup>, Yanyong Zhang<sup>1†</sup>

<sup>1</sup>University of Science and Technology of China, <sup>2</sup>The Chinese University of Hong Kong {wangshaoxuan, zgt, huangrui2002, tangzhipeng}@mail.ustc.edu.cn, zhangsha2048@gmail.com, {dengjj, yanyongz}@ustc.edu.cn

## Abstract

Ofline reinforcement learning improves robotic policies using previously collected data without further environment interaction. Yet prevalent difusion- and flow-matching robot policies lack tractable likelihoods, limiting their use in likelihoodbased ofline RL post-training. AR-NFs ofer both expressive action modeling and exact likelihood evaluation, but their sequential sampling incurs substantial sampling overhead during policy optimization and deployment. We present RoMAN-Flow (Robotic Manipulation with Autoregressive Normalizing Flows), an ofline reinforcement learning framework that makes AR-NF policies practical for robotic manipulation by addressing this sampling bottleneck in both stages. During policy optimization, RoMAN-Flow employs a sampling-free, advantage-weighted likelihood objective that assigns higher likelihood to high-advantage actions from the ofline dataset without sampling from the autoregressive policy. For eficient deployment, it distills the optimized autoregressive policy into a one-step action generator, enabling low-latency action prediction. Experiments across multiple simulated manipulation benchmarks and real-world robotic platforms demonstrate that RoMAN-Flow achieves competitive policy performance while substantially reducing inference latency. Code is available at https://github.com/konnyaku28/RoMAN-Flow.

## 1 Introduction

Vision-language-action (VLA) models have shown strong potential for robotic manipulation, yet pretrained policies often require further adaptation to downstream tasks. Ofline reinforcement learning (RL) provides a practical approach to robotic policy post-training by improving policies from previously collected reward-labeled data without additional environment interaction.

Likelihood-based ofline policy optimization has recently shown strong efectiveness in large language model (Achiam et al. 2023; Touvron et al. 2023) post-training, where autoregressive models provide tractable exact log-likelihoods for direct optimization on fixed datasets (Rafailov et al. 2023; Richemond et al. 2024). In robotic manipulation, however, this paradigm remains comparatively underexplored because the dominant difusion (Chi et al. 2025) and flowmatching (Lipman et al. 2022) policies do not readily expose low-cost exact conditional action likelihoods. Autoregressive policies (Kim et al. 2024; Zitkovich et al. 2023) provide an alternative with tractable likelihoods, but they typically convert continuous robot actions into discrete token sequences. Such discretization introduces quantization error and may compromise fine-grained control by obscuring the continuous geometry and temporal coordination of robotic actions. Recent transformer-based autoregressive normalizing flows (AR-NFs) (Patacchiola et al. 2024; Kingma et al. 2016; Papamakarios, Pavlakou, and Murray 2017) have demonstrated strong generative modeling capability in image generation (Zhai et al. 2025; Gu et al. 2026; Zheng et al. 2025; Zhao et al. 2025). By combining autoregressive dependency modeling with invertible transformations, AR-NFs substantially improve the expressiveness of conventional normalizing flows (Kobyzev, Prince, and Brubaker 2020; Dinh, Sohl-Dickstein, and Bengio 2017; Dinh, Krueger, and Bengio 2014; Kingma and Dhariwal 2018; Kolesnikov, Pinto, and Tschannen 2024) while retaining tractable exact likelihood evaluation. These advances motivate us to investigate AR-NFs as a policy representation for likelihood-based ofline robotic reinforcement learning.

Despite these attractive properties, the autoregressive inverse of AR-NFs makes action generation sequential, introducing substantial sampling overhead during both policy optimization and deployment. In this work, we present RoMAN-Flow (Robotic Manipulation with Autoregressive Normalizing Flows), an ofline robotic reinforcement learning framework that addresses these two eficiency bottlenecks from the policy-optimization and inference perspectives.

During ofline RL post-training, RoMAN-Flow builds upon Implicit Q-Learning (IQL) (Kostrikov, Nair, and Levine 2021), whose actor update performs advantage-weighted maximum likelihood over actions contained in the ofline dataset. Because this update does not require actions sampled from the current policy, the AR-NF only applies its parallel forward transformation to evaluate the exact likelihoods of dataset action chunks, avoiding costly autoregressive reverse sampling during reinforcement learning. RoMAN-Flow therefore directly increases the likelihood of high-value ofline actions while retaining the expressive continuous action distribution modeled by the AR-NF policy.

To accelerate inference, we adapt one-step distillation (Zheng et al. 2025; Lu et al. 2025) to compress the trained RoMAN-Flow policy into a one-step actor. By matching the post-trained teacher’s intermediate flow states and final action chunks, the one-step student learns to approximate its inverse transformation while generating all actions in parallel. The distilled policy largely preserves the teacher’s task performance and substantially reduces inference latency by eliminating the sequential autoregressive reverse process.

We evaluate RoMAN-Flow on LIBERO (Liu et al. 2023), MetaWorld (Yu et al. 2020), RoboMimic (Mandlekar et al. 2021), and real-world robotic manipulation tasks. Across these settings, RoMAN-Flow achieves competitive policy performance while reducing policy inference latency by nearly an order of magnitude. These results demonstrate that modern autoregressive normalizing flows provide a practical alternative to difusion and flow-matching policies (Chi et al. 2025; Black et al. 2024; Physical Intelligence et al. 2025) for ofline robotic policy post-training.

Our main contributions are summarized as follows:

• An ofline reinforcement learning framework for robot policy learning, namely RoMAN-Flow, that enables expressive continuous-action modeling while retaining tractable, exact likelihood evaluation through autoregressive normalizing flows (AR-NFs).

• A sampling-free NF-IQL procedure that performs advantage-weighted likelihood optimization exclusively on ofline dataset actions, avoiding costly autoregressive policy sampling during post traing.

• A one-step policy distillation to eliminate autoregressive inference overhead while preserving policy performance, and validate the framework across multiple simulated benchmarks and real-world robotic tasks.

## 2 Related Work

## 2.1 Generative Policies for Robot Learning

Generative policies have become a dominant approach to robotic manipulation because they can represent complex and multimodal action distributions. Difusion Policy introduced conditional difusion models for visuomotor control (Chi et al. 2025), followed by extensions to 3D observations and vision-language-action modeling (Ze et al. 2024; Wen et al. 2025). Flow matching provides an alternative generative formulation based on conditional vector fields (Lipman et al. 2022) and has been adopted by large-scale VLA models such as $\pi _ { 0 }$ (Black et al. 2024) and $\pi _ { 0 . 5 }$ (Physical Intelligence et al. 2025). However, difusion and flow-matching policies do not expose low-cost exact action likelihoods: difusion models rely on denoising objectives, while continuous flows require density tracking along the generation ODE (Ho, Jain, and Abbeel 2020; Song et al. 2020; Chen et al. 2018; Grathwohl et al. 2018). Their ofline RL post-training therefore typically requires surrogate or specialized optimization procedures.

## 2.2 Ofline RL and Normalizing-Flow Policies

Ofline RL improves policies using fixed reward-labeled datasets while controlling distribution shift. Representative methods include behavior-regularized optimization (Fujimoto and Gu 2021) and advantage-weighted policy extraction (Peng et al. 2019; Nair et al. 2020; Kostrikov, Nair, and Levine 2021). Likelihood-based ofline policy optimization has also shown strong efectiveness in large language model post-training, where autoregressive models permit direct evaluation of policy likelihoods (Rafailov et al. 2023; Richemond et al. 2024). For difusion policies, Difusion-QL introduces value guidance during denoising (Wang, Hunt, and Zhou 2022), EDP adapts difusion training to established ofline RL objectives (Kang et al. 2023), and IDQL performs critic-guided extraction from a difusion behavior model (Hansen-Estruch et al. 2023). These methods avoid direct exact-likelihood optimization through denoising surrogates, value guidance, or sampling-based extraction.

Normalizing flows (Kobyzev, Prince, and Brubaker 2020; Papamakarios et al. 2021; Mathieu and Nickel 2020) instead provide exact density evaluation through invertible transformations (Rezende and Mohamed 2015; Dinh, Sohl-Dickstein, and Bengio 2017), making them naturally compatible with likelihood-based policy optimization. Although conventional normalizing flows have historically been limited in expressiveness, recent autoregressive normalizing flows, including TarFlow (Zhai et al. 2025), STARFlow (Gu et al. 2026), and SimFlow (Zhao et al. 2025), demonstrate substantially improved generative modeling while retaining exact likelihoods. Their autoregressive inversion, however, leads to slow generation; FARMER (Zheng et al. 2025) and BiFlow (Lu et al. 2025) address a similar limitation through a learned one-step inverse. Building on these advances, we study AR-NFs as robotic policies, post-train them through likelihood-based ofline RL, and distill the optimized policy for eficient deployment.

## 3 Method

We develop a likelihood-based ofline robotic reinforcement learning framework, named RoMAN-Flow, based on autoregressive normalizing flow (AR-NF) policies. This section starts with the problem setup and the construction of the conditional AR-NF policy. Then, we elaborate on NF-IQL post-training, which directly increases the exact likelihood of high-value actions from the ofline dataset. After that, we detail the one-step policy distillation for eficient deployment. An overview of our framework is shown in Figure 1.

## 3.1 Problem Setup

We consider imitation learning followed by ofline reinforcement learning post-training. Given a fixed dataset of rewardlabeled trajectories, we organize each trajectory into overlapping length-H chunk transitions

$$
\begin{array} { r } { x _ { n , t } = \left( c _ { n , t } , \mathbf { a } _ { n , t } , \mathbf { r } _ { n , t } , c _ { n , t + H } \right) , \mathcal { D } _ { H } = \left\{ x _ { n , t } \right\} _ { n , t } , } \end{array}\tag{1}
$$

where $\mathbf { a } _ { n , t } = ( a _ { n , t } , \ldots , a _ { n , t + H - 1 } )$ is a continuous action chunk and $\mathbf { r } _ { n , \imath }$ <sub>t</sub> contains the rewards. Imitation learning models the demonstrated action distribution, whereas ofline RL uses the reward annotations to favor higher-value action chunks without additional environment interaction.

![](images/b6b4c4bd932212190869baeb97abeeca68a5c161fd7d726bfe5560cca5304bfe.jpg)  
Figure 1: Overview ofRoMAN-Flow. RoMAN-Flow adopts an invertible AR-NF architecture that maps continuous action chunks to latent variables conditioned on multimodal inputs. For imitation learning, it maximizes the exact conditional likelihood of demonstrated action chunks under a Gaussian latent prior. For action generation, it samples latent variables from the Gaussian prior and recovers action chunks through the autoregressive inverse transformation. For likelihood-based reinforcement learning, NF-IQL directly increases the exact likelihoods of high-advantage ofline action chunks without autoregressive policy sampling.

## 3.2 Autoregressive Normalizing Flow Policy

Given a chunk-level sample $\left( { { c } _ { t } } , { { \bf { a } } _ { t } } \right)$ , RoMAN-Flow first encodes the visual observations, language instruction, and optional proprioceptive states into multimodal context tokens $\mathbf { C } _ { t }$ using a pretrained encoder. Conditioned on context tokens $\mathbf { C } _ { t }$ , RoMAN-Flow then maps an observed action chunk a to a Gaussian latent $\mathbf { z } _ { t }$ through an invertible AR-NF $F _ { \theta }$ The forward transformation provides exact conditional likelihoods for imitation learning and NF-IQL, whereas action generation applies the autoregressive inverse transformation $\mathbf { \bar { \boldsymbol { F } } } _ { \theta } ^ { - 1 }$ to a latent sampled from the Gaussian prior.

Multimodal context encoder. To fully exploit visual, linguistic, and proprioceptive conditions for action generation, we employ a pretrained vision-language model as the multimodal encoder. Given the policy condition $c _ { t } .$ , the encoder produces context tokens $\mathbf { C } _ { t }$ . The context tokens are provided to every AR-NF flow block to condition action generation.

Autoregressive normalizing flow. We adopt SimFlow (Zhao et al. 2025) as the backbone of our conditional AR-NF model and adapt it to model continuous robotic action chunks. The AR-NF transforms an action chunk $\mathbf { a } _ { t }$ into a latent $\mathbf { z } _ { t }$ through L conditional invertible flow blocks ${ \cal F } _ { \theta } = f _ { L } \circ f _ { L - 1 } \circ \cdot \cdot \cdot \circ f _ { 1 } .$

$$
\mathbf { h } _ { t } ^ { ( 0 ) } = \mathbf { a } _ { t } , \mathbf { h } _ { t } ^ { ( l ) } = f _ { l } \left( \mathbf { h } _ { t } ^ { ( l - 1 ) } ; \mathbf { C } _ { t } \right) , \mathbf { z } _ { t } = \mathbf { h } _ { t } ^ { ( L ) } ,\tag{2}
$$

where $\mathbf { h } _ { + } ^ { ( l ) }$ denotes the output of the l-th AR-NF block.

Each block employs a Transformer with a prefix-causal attention mask. At action position $j ,$ the Transformer predicts

an afine shift $\mu _ { t , j } ^ { ( l ) }$ and log-scale $\mathbf { s } _ { t , j } ^ { ( l ) }$ using the multimodal context and preceding action positions:

$$
\left( \pmb { \mu } _ { t , j } ^ { ( l ) } , \mathbf { s } _ { t , j } ^ { ( l ) } \right) = { \cal T } _ { l } \left( \mathbf { C } _ { t } , \mathbf { h } _ { t , < j } ^ { ( l - 1 ) } \right) .\tag{3}
$$

The corresponding forward transformation is

$$
\mathbf { h } _ { t , j } ^ { ( l ) } = \left( \mathbf { h } _ { t , j } ^ { ( l - 1 ) } - { \pmb { \mu } } _ { t , j } ^ { ( l ) } \right) \odot \exp \left( - \mathbf { s } _ { t , j } ^ { ( l ) } \right) .\tag{4}
$$

Because the afine parameters at position $j$ depend only on $\mathbf { C } _ { t }$ and preceding positions, each flow block has a triangular Jacobian. The exact conditional log-likelihood is therefore

$$
\log \pi _ { \boldsymbol \theta } \left( \mathbf { a } _ { t } \mid c _ { t } \right) = \log p _ { 0 } \left( \mathbf { z } _ { t } \right) - \sum _ { l = 1 } ^ { L } \sum _ { j = 1 } ^ { H } \mathbf { s } _ { t , j } ^ { ( l ) } ,\tag{5}
$$

where $p _ { 0 } ( \mathbf { z } ) = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is the standard Gaussian density.

Imitation learning and action generation. We initialize RoMAN-Flow by minimizing the negative log-likelihood of demonstrated action chunks:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I L } } = - \mathbb { E } _ { ( c _ { t } , \mathbf { a } _ { t } ) \sim \mathcal { D } _ { H } } \left[ \log \pi _ { \theta } \left( \mathbf { a } _ { t } \mid c _ { t } \right) \right] . } \end{array}\tag{6}
$$

Because the complete action chunk is observed during training, the afine parameters for all positions can be evaluated in parallel through a single masked Transformer forward pass. This enables eficient exact-likelihood optimization during both imitation learning and NF-IQL post-training.

At inference time, the policy samples $\mathbf z _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and generates an action chunk through

$$
\mathbf { a } _ { t } = F _ { \theta } ^ { - 1 } \left( \mathbf { z } _ { t } ; \mathbf { C } _ { t } \right) , \qquad F _ { \theta } ^ { - 1 } = f _ { 1 } ^ { - 1 } \circ f _ { 2 } ^ { - 1 } \circ \cdot \cdot \cdot \circ f _ { L } ^ { - 1 } .\tag{7}
$$

Within each inverse block, the afine parameters depend on previously recovered action positions, requiring the action chunk to be reconstructed sequentially. This forward–inverse asymmetry provides parallel exact-likelihood evaluation during training but introduces substantial latency at inference.

## 3.3 NF-IQL Post-Training

To improve the behavior-cloned RoMAN-Flow policy beyond imitation learning, we develop NF-IQL, an ofline reinforcement learning framework built upon IQL (Kostrikov, Nair, and Levine 2021) and tailored to AR-NF policies. Conventional actor-critic methods, such as TD3+BC (Fujimoto and Gu 2021), require evaluating the critic on actions generated by the current policy during actor optimization. For an AR-NF policy, generating these actions invokes the sequential inverse transformation and therefore incurs substantial training overhead. NF-IQL instead performs policy improvement exclusively on action chunks contained in the ofline dataset, avoiding current-policy action sampling.

NF-IQL consists of two coupled components. First, it learns chunk-level action values and a state-value baseline from the ofline data. Second, it converts their diference into an advantage weight and uses this weight to optimize the exact conditional likelihood of each dataset action chunk. The complete training procedure is summarized in the Appendix.

Chunk-level Q and state-value learning. We jointly learn an ensemble of M Chunk Transformer critics $\{ Q _ { \omega _ { m } } \} _ { m = 1 } ^ { M }$ and a state-value network $V _ { \phi } ,$ where $\omega _ { m }$ and $\phi$ denote their trainable parameters. For stable value estimation, we maintain a target copy $\{ Q _ { \bar { \omega } _ { m } } \} _ { m = 1 } ^ { M }$ of the online critic ensemble, where $\bar { \omega } _ { m }$ is updated from $\omega _ { m }$ through Polyak averaging and receives no gradient updates. Given an ofline action chunk, each critic $Q _ { \omega _ { m } }$ outputs Q-value estimates for all action prefixes, while $V _ { \phi }$ maps the policy condition $c _ { t }$ to a scalar state value $V _ { \phi } ( c _ { t } )$ . The state value serves both as the bootstrap target for critic learning and as the state-dependent baseline for subsequent policy improvement.

Following CO-RFT (Huang et al. 2025), the m-th critic produces prefix-level Q-value estimates in a single causally masked forward pass: $\{ Q _ { \omega _ { m } } ^ { ( j ) } ( c _ { t } , a _ { t : t + j } ) \} _ { j = 0 } ^ { H - 1 }$ , m = $1 , \ldots , { \dot { M } }$ , where $j$ indexes the action prefix and $a _ { t : t + j } =$ $( a _ { t } , \ldots , a _ { t + j } )$ . Each action token attends to the policy condition and preceding actions while being prevented from accessing future actions.

For the j-th action prefix, we construct a temporaldiference target using the discounted rewards accumulated within the prefix and the bootstrapped value of the subsequent state:

$$
y _ { t , j } = \sum _ { i = 0 } ^ { j } \gamma ^ { i } r _ { t + i } + \gamma ^ { j + 1 } \operatorname { s g } \left[ V _ { \phi } \left( c _ { t + j + 1 } \right) \right] ,\tag{8}
$$

where $\gamma \in ( 0 , 1 ]$ denotes the discount factor and $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operation. The online critics are trained by regressing all prefix-level Q-values toward their correspond-

ing temporal-diference targets:

$$
\mathcal { L } _ { Q } = \mathbb { E } _ { \mathcal { D } _ { H } } \left[ \frac { 1 } { M H } \sum _ { m = 1 } ^ { M } \sum _ { j = 0 } ^ { H - 1 } \left( Q _ { \omega _ { m } } ^ { ( j ) } \left( c _ { t } , a _ { t : t + j } \right) - y _ { t , j } \right) ^ { 2 } \right] .\tag{9}
$$

To obtain a scalar value for the complete action chunk, we average the target-critic estimates over both action prefixes and ensemble members:

$$
\bar { Q } _ { \bar { \omega } } \left( c _ { t } , \mathbf { a } _ { t } \right) = \frac { 1 } { M H } \sum _ { m = 1 } ^ { M } \sum _ { j = 0 } ^ { H - 1 } Q _ { \bar { \omega } _ { m } } ^ { ( j ) } \left( c _ { t } , a _ { t : t + j } \right) ,\tag{10}
$$

where $\bar { \omega } _ { m }$ denotes the parameters of the m-th target critic.

The state-value network is optimized through expectile regression over the aggregated target-critic estimates:

$$
\mathcal { L } _ { V } = \mathbb { E } _ { ( c _ { t } , \mathbf { a } _ { t } ) \sim \mathcal { D } _ { H } } \left[ \rho _ { \tau } \left( \mathrm { s g } \left[ \bar { Q } _ { \bar { \omega } } \left( c _ { t } , \mathbf { a } _ { t } \right) \right] - V _ { \phi } ( c _ { t } ) \right) \right] _ { \mathbf { \sigma } _ { \lambda } }\tag{11}
$$

where $\rho _ { \tau } ( u ) = | \tau - \mathbb { I } [ u < 0 ] | u ^ { 2 }$ and $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ is the expectile parameter. For $\tau > 0 . 5$ , this objective yields an upper-expectile state-value baseline.

Advantage-weighted likelihood optimization. Using the aggregated target-critic estimate and the state-value baseline, we define the advantage of each ofline action chunk and its corresponding likelihood weight as

$$
A _ { t } = \bar { Q } _ { \bar { \omega } } \left( c _ { t } , \mathbf { a } _ { t } \right) - V _ { \phi } ( c _ { t } ) , \qquad w _ { t } = \exp \left( \beta A _ { t } \right) ,\tag{12}
$$

where $\beta > 0$ controls the strength of advantage weighting. Action chunks whose estimated values exceed the state-value baseline receive larger weights, whereas lower-value action chunks contribute less to policy optimization.

We post-train the RoMAN-Flow actor by minimizing the advantage-weighted negative log-likelihood:

$$
\mathcal { L } _ { \pi } = - \mathbb { E } _ { ( c _ { t } , \mathbf { a } _ { t } ) \sim \mathcal { D } _ { H } } \left[ \mathrm { s g } \left[ w _ { t } \right] \log \pi _ { \theta } \left( \mathbf { a } _ { t } \mid c _ { t } \right) \right] .\tag{13}
$$

The target-critic estimate, state value, advantage, and likelihood weight are treated as fixed during the actor update. NF-IQL therefore shifts the policy toward higher-value behaviors without requiring current-policy action generation.

Training schedules, benchmark-specific reward relabeling, and target-network updates are detailed in the Appendix.

## 3.4 One-Step Policy Distillation

Although NF-IQL avoids autoregressive action sampling during ofline post-training, deploying the resulting AR-NF policy still requires sequential inverse transformation. To accelerate policy execution, inspired by FARMER (Zheng et al. 2025) and BiFlow (Lu et al. 2025), we freeze the NF-IQLpost-trained RoMAN-Flow policy as a teacher and distill its autoregressive inverse mapping into a bidirectional student that generates an entire action chunk in one forward pass.

Data-induced trajectory distillation. As shown in Figure 2, we instantiate the student $g _ { \psi }$ as a bidirectional Transformer that processes all action positions in parallel. Given an ofline action chunk $\mathbf { a } _ { t }$ , we first apply a small Gaussian perturbation, $\widetilde { \mathbf { a } } _ { t } = \mathbf { a } _ { t } + \mathbf { \epsilon } \epsilon _ { t }$ with $\mathbf { \epsilon } \epsilon _ { t } \sim \mathrm { \mathcal { N } } ( \mathbf { \bar { 0 } } , \sigma ^ { 2 } \mathbf { I } )$ . The frozen teacher maps $\widetilde { \mathbf { a } } _ { t }$ to a latent code $\mathbf { z } _ { t }$ and records the intermediate flow states $\{ \mathbf { h } _ { t } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ . Conditioned on the same multimodal context $\mathbf { C } _ { t }$ , the student predicts the complete action chunk and its intermediate states:

![](images/e430d7850bb65911c8e94b27151a6c6040ff6cd3668ffe2332c79ba646463aaf.jpg)  
Figure 2: One-step policy distillation. The student model aligns with the teacher’s intermediate reverse states and reconstructs the final action chunk in a single forward pass.

$$
\left( \widehat { \mathbf { u } } _ { t } ^ { \left( 1 \right) } , \ldots , \widehat { \mathbf { u } } _ { t } ^ { \left( L \right) } , \widehat { \mathbf { a } } _ { t } \right) = g _ { \psi } \left( \mathbf { z } _ { t } ; \mathbf { C } _ { t } \right) .\tag{14}
$$

We train the student through intermediate-state alignment and action-chunk reconstruction:

$$
\mathcal { L } _ { \mathrm { d a t a } } = \frac { \lambda _ { s } } { L } \sum _ { r = 1 } ^ { L } \left. \widehat { \mathbf { u } } _ { t } ^ { ( r ) } - \mathbf { h } _ { t } ^ { ( L - r ) } \right. _ { 2 } ^ { 2 } + \lambda _ { a } \left. \widehat { \mathbf { a } } _ { t } - \mathbf { a } _ { t } \right. _ { 2 } ^ { 2 } ,\tag{15}
$$

where $\lambda _ { s }$ and $\lambda _ { a }$ balance intermediate-state alignment and action reconstruction, respectively.

Prior-sampled trajectory distillation. The data-induced branch covers only latent codes obtained from ofline action chunks. To expose the student to a broader portion of the posttrained teacher distribution, we additionally introduce priorsampled distillation. For each policy condition, we sample latent $\mathbf { z } _ { t } ^ { p }$ and decode it through the frozen NF-IQL teacher:

$$
\mathbf { z } _ { t } ^ { p } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) , \left( \mathbf { a } _ { t } ^ { p } , \{ \mathbf { h } _ { t } ^ { p , ( l ) } \} _ { l = 0 } ^ { L - 1 } \right) = F _ { \theta ^ { \star } } ^ { - 1 } \left( \mathbf { z } _ { t } ^ { p } ; \mathbf { C } _ { t } \right) .\tag{16}
$$

where $\theta ^ { \star }$ denotes the frozen teacher parameters. Applying the same intermediate-state alignment and action-reconstruction objective to these teacher-generated trajectories defines the prior-sampling loss ${ \mathcal { L } } _ { \mathrm { { p r i o 1 } } }$

The complete distillation objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i s t i l l } } = \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r i o r } } , } \end{array}\tag{17}
$$

where $\lambda _ { p }$ controls the contribution of the prior-sampling loss.

## 4 Experiments

## 4.1 Experimental Setup

Benchmarks. We evaluate RoMAN-Flow on MetaWorld-MT50 (Yu et al. 2020), LIBERO (Liu et al. 2023),

![](images/570ab9dc354eea2900ab0dc90bdf4389211112d1cf1e91f2c9e03ac04e6fb701.jpg)  
Figure 3: Qualitative results ofRoMAN-Flow on MetaWorld, LIBERO-Long, RoboMimic, and real-world tasks. Each row shows selected frames from a rollout, while the bottomright inset presents our real-robot platform with a seven-DoF Franka arm and a twelve-DoF XHand dexterous hand.

RoboMimic MH (Mandlekar et al. 2021), and a real-robot platform. MetaWorld-MT50 contains 50 manipulation tasks and evaluates large-scale multitask learning; we report the average success rate across all four dificulty levels. LIBERO consists of four ten-task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, which evaluate generalization across spatial configurations, object identities, language-specified goals, and long-horizon behaviors, respectively. On RoboMimic MH, we conduct a controlled comparison with SERNF (Yang et al. 2026), which combines a conventional RealNVP policy with TD3+BC.

Real-world evaluation. We evaluate RoMAN-Flow on the Franka–XHand platform shown in Figure 3, consisting of a seven-DoF Franka arm and a twelve-DoF XHand dexterous hand. We consider four tasks: Pick Beaker, Pick Cylinder, Place Beaker, and Put Beaker on Balance. At evaluation time, objects are initialized outside the spatial distribution covered by the ofline demonstrations. This setting assesses spatial out-of-distribution generalization and robustness to real-world visual variation, sensing noise, and control errors.

## 4.2 Main Results

We compare three stages of RoMAN-Flow: imitation learning, denoted as RoMAN-Flow (IL); NF-IQL post-training, denoted as RoMAN-Flow (NF-IQL); and the distilled policy, denoted as RoMAN-Flow (One-Step). All controlled comparisons use the same ofline data and evaluation settings. Training and evaluation settings are detailed in the Appendix.

<table><tr><td>Method</td><td>#Params</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Very Hard</td><td>Average</td></tr><tr><td>Diffusion Policy (Chi et al. 2025)</td><td>157M</td><td>23.1</td><td>10.7</td><td>1.9</td><td>6.1</td><td>10.5</td></tr><tr><td>TinyVLA (Wen et al. 2025)</td><td>1.3B</td><td>77.6</td><td>21.5</td><td>11.4</td><td>15.8</td><td>31.6</td></tr><tr><td>SmolVLA (Shukor et al. 2025)</td><td>2.25B</td><td>87.1</td><td>51.8</td><td>70.0</td><td>64.0</td><td>68.2</td></tr><tr><td>π0 (Black et al. 2024)</td><td>3.3B</td><td>77.9</td><td>51.8</td><td>53.3</td><td>20.0</td><td>50.8</td></tr><tr><td>π0 + Flow-SDE (Chen et al. 2025)</td><td>3.3B</td><td>92.1</td><td>74.6</td><td>61.7</td><td>84.0</td><td>78.1</td></tr><tr><td>π0.5 (Physical Intelligence et al. 2025)</td><td>2.3B</td><td>68.2</td><td>37.3</td><td>41.7</td><td>28.0</td><td>43.8</td></tr><tr><td>π0.5 + Flow-SDE (Chen et al. 2025)</td><td>2.3B</td><td>86.4</td><td>55.5</td><td>75.0</td><td>66.0</td><td>70.7</td></tr><tr><td>RoMAN-Flow (IL)</td><td>1.45B</td><td>80.4</td><td>63.6</td><td>75.0</td><td>72.0</td><td>72.8</td></tr><tr><td>RoMAN-Flow (NF-IQL)</td><td>1.45B</td><td>80.0</td><td>74.5</td><td>90.0</td><td>80.0</td><td>81.1</td></tr><tr><td>RoMAN-Flow (One-Step)</td><td>0.56B</td><td>81.4</td><td>72.7</td><td>80.0</td><td>80.0</td><td>78.5</td></tr></table>

Table 1: Success rates (%) on MetaWorld-MT50. Avg. is the unweighted mean over the four dificulty groups. Bold and underlined numerical entries denote the best and second-best results, respectively.
<table><tr><td>Method</td><td>#Params</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Average</td></tr><tr><td>Diffusion Policy (Chi et al. 2025)</td><td>157M</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td></tr><tr><td>OpenVLA (Kim et al. 2024)</td><td>7.5B</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>NORA-Long (Hung et al. 2025)</td><td>3B</td><td>92.2</td><td>95.4</td><td>89.4</td><td>74.6</td><td>87.9</td></tr><tr><td>SmolVLA (Šhukor et al. 2025)</td><td>2.25B</td><td>93.0</td><td>94.0</td><td>91.0</td><td>77.0</td><td>88.8</td></tr><tr><td>GR00T-N1 (Bjorck et al. 2025)</td><td>2.2B</td><td>94.4</td><td>97.6</td><td>93.0</td><td>90.6</td><td>93.9</td></tr><tr><td>π0 (Black et al. 2024)</td><td>3.3B</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>UniVLA (Full) (Bu et al. 2025)</td><td>~7.5B</td><td>96.5</td><td>96.8</td><td>95.6</td><td>92.0</td><td>95.2</td></tr><tr><td>RoMAN-Flow (IL)</td><td>1.45B</td><td>95.0</td><td>99.2</td><td>94.2</td><td>85.6</td><td>93.5</td></tr><tr><td>RoMAN-Flow (NF-IQL)</td><td>1.45B</td><td>95.0</td><td>99.4</td><td>94.6</td><td>92.2</td><td>95.3</td></tr><tr><td>RoMAN-Flow (One-Step)</td><td>0.56B</td><td>94.4</td><td>95.6</td><td>91.8</td><td>93.0</td><td>93.7</td></tr></table>

Table 2: Success rates (%) on the four LIBERO task suites. Average is the unweighted mean across suites. Bold and underlined numerical entries denote the best and second-best results, respectively.

Qualitative results. Figure 3 presents representative successful rollouts from MetaWorld, LIBERO-Long, RoboMimic, and our real-world setup. The sequences show coherent multi-stage behaviors, including approaching and manipulating objects, transporting them across the workspace, and completing precise target placement. In particular, RoMAN-Flow successfully executes the real-world task of placing a beaker onto a balance. Additional qualitative examples are provided in the Appendix.

Large-scale multitask evaluation on MetaWorld. Table 1 reports success rates on MetaWorld-MT50 using the four dificulty groups and evaluation protocol defined by π (Chen et al. 2025). Compared with RoMAN-Flow (IL), NF-IQL increases the unweighted mean across dificulty groups from 72.8% to 81.1%, with gains concentrated in the Medium, Hard, and Very Hard groups. RoMAN-Flow (One-Step) achieves 78.5%, retaining most of the NF-IQL teacher’s multitask performance while replacing sequential AR-NF inversion with one-step inference. Under the aligned protocol, RoMAN-Flow (NF-IQL) also outperforms π<sub>0</sub> + Flow-SDE by 3.0 percentage points. This comparison is particularly notable because π<sub>0</sub> + Flow-SDE uses online environment interaction, whereas NF-IQL operates exclusively on fixed ofline data.

Multi-dimensional task generalization on LIBERO. Table 2 reports results on the four oficial LIBERO suites.

NF-IQL increases the mean success rate from 93.5% to 95.3%, primarily through a 6.6-percentage-point improvement on LIBERO-Long, while preserving the strong IL performance on Spatial, Object, and Goal. RoMAN-Flow (NF-IQL) achieves the highest average success rate among the compared methods while using fewer inference-time parameters than the listed large-scale VLA baselines. The one-step student attains an average success rate of 93.7% and achieves the highest result on LIBERO-Long, indicating that the distilled policy remains efective on long-horizon manipulation.

Comparison with SERNF on RoboMimic. Table 4 compares RoMAN-Flow with SERNF (Yang et al. 2026) under the training and evaluation protocol used by SERNF. At the imitation-learning stage, RoMAN-Flow matches or exceeds SERNF on all three tasks, supporting the efectiveness of AR-NFs for continuous-action policy modeling. After ofline RL post-training, RoMAN-Flow (NF-IQL) achieves success rates of 100%, 96%, and 85% on Lift, Can, and Square, respectively, compared with 91%, 96%, and 68% for SERNF (TD3+BC). The one-step student retains strong performance, achieving 100%, 95%, and 80% on the three tasks.

Real-world manipulation. Table 3 reports success rates on four real-robot manipulation tasks. NF-IQL increases the average success rate from 57.3% to 81.5%, corresponding to an absolute improvement of 24.2 percentage points. The largest gains occur on Pick Beaker, Pick Cylinder, and Put

<table><tr><td>Method</td><td>Pick_beaker</td><td>Pick_cylinder</td><td>Place_beaker</td><td>Balance</td><td>Average</td></tr><tr><td>Diffusion Policy (Chi et al. 2025)</td><td>6</td><td>0</td><td>0</td><td>0</td><td>1.5</td></tr><tr><td>(Black et al. 2024)  $\pi _ { 0 }$ </td><td>80</td><td>3</td><td>93</td><td>73</td><td>62.3</td></tr><tr><td> $\pi _ { 0 . 5 }$  (Physical Intelligence et al. 2025)</td><td>93</td><td>10</td><td>97</td><td>87</td><td>71.8</td></tr><tr><td>RoMAN-Flow (IL)</td><td>36</td><td>20</td><td>100</td><td>73</td><td>57.3</td></tr><tr><td>RoMAN-Flow (NF-IQL)</td><td>100</td><td>33</td><td>100</td><td>93</td><td>81.5</td></tr><tr><td>RoMAN-Flow (One-Step)</td><td>78</td><td>20</td><td>80</td><td>100</td><td>69.5</td></tr></table>

Table 3: Success rates (%) on four real-robot manipulation tasks. Average denotes the unweighted mean across the four tasks.

<table><tr><td>Method</td><td>Lift</td><td> $\mathbf { C a n }$ </td><td>Square</td></tr><tr><td>SERNF (IL)</td><td> $\overline { { 7 9 \pm 0 } }$ </td><td> $\overline { { 9 6 \pm 0 } }$ </td><td> $\overline { { 6 1 \pm 4 } }$ </td></tr><tr><td>SERNF (TD3+BC)</td><td> $9 1 \pm 4$ </td><td> $9 6 \pm 2$ </td><td> $6 8 \pm 5$ </td></tr><tr><td>RoMAN-Flow (IL)</td><td> $\mathbf { 1 0 0 \pm 0 }$ </td><td> $\mathbf { \overline { { 9 7 \pm 2 } } }$ </td><td> $\overline { { 8 0 \pm 3 } }$ </td></tr><tr><td>RoMAN-Flow (NF-IQL)</td><td> ${ \bf 1 0 0 \pm 0 }$ </td><td> $9 6 \pm 3$ </td><td> ${ \bf 8 5 \pm 3 }$ </td></tr><tr><td>RoMAN-Flow (One-Step)</td><td> ${ \bf 1 0 0 \pm 0 }$ </td><td> $9 5 \pm 2$ </td><td> $8 0 \pm 3$ </td></tr></table>

Table 4: Success rates (%) on RoboMimic MH. Results are reported as the mean and standard deviation over four training seeds, with 100 evaluation rollouts per seed.

Beaker on Balance, where success rates improve by 64, 13, and 20 percentage points, respectively. These results show that ofline RL post-training can efectively use reward information to refine grasping and precise placement behaviors under real-world visual and control variability. RoMAN-Flow (NF-IQL) further outperforms these baselines across all four tasks, achieving an average success rate of 81.5%, a 9.7-percentage-point improvement over the stronger π<sub>0.5</sub> baseline.

## 4.3 One-Step Distillation and Eficient Inference

RoMAN-Flow (NF-IQL) incurs high inference latency because its autoregressive inverse transformation generates action chunks sequentially. We therefore distill it into a bidirectional one-step student. Across the simulation benchmarks, the student achieves performance comparable to its NF-IQL teacher, slightly outperforming it on LIBERO-Long and retaining approximately 97% of its performance on MetaWorld-MT50 and RoboMimic MH. As shown in Figure 4, the student achieves a LIBERO-Long success rate of 93.0%, compared with 92.2% for the teacher, while reducing action-chunk generation latency from approximately 697 ms to 81.5 ms, corresponding to an 8.55× speedup.

## 4.4 Ablation Study

Efect of AR-NF actor capacity. We study the scaling behavior of the AR-NF actor on RoboMimic Square-MH. Holding the visual encoder, ofline dataset, and training and evaluation protocol fixed, we vary only the width, number of attention heads, and depth of the AR-NF actor. As shown in Table 5, the S, B, and L configurations span 33.8M to 466.2M parameters but achieve nearly identical success rates of 77%– 78%. This suggests that increasing capacity within the smallto-medium scale regime does not yet translate into improved modeling of fine-grained action structure. Once scaled to

![](images/31d33ef96d0ac789244d657d88e6ac98206ff99d6eccfb553ba48381414a6fe6.jpg)

Figure 4: Speed–performance trade-of on LIBERO-Long with action horizon H = 16. Each point reports the average success rate and action-chunk generation latency over the ten LIBERO-Long tasks, with marker area proportional to the number of inference-time policy parameters.
<table><tr><td>Size</td><td>Dim.</td><td>Heads</td><td>Layer Depth</td><td>#Params</td><td>SR</td></tr><tr><td>S</td><td>384</td><td>6</td><td>[2, 2, 2, 2, 2, 2]</td><td>33.8M</td><td>78.0</td></tr><tr><td>B</td><td>768</td><td>12</td><td>[2, 2, 2, 2, 2, 2]</td><td>134.8M</td><td>77.0</td></tr><tr><td>L</td><td>1024</td><td>16</td><td>[2, 2, 2, 2, 2, 14]</td><td>466.2M</td><td>78.0</td></tr><tr><td>XL</td><td>1152</td><td>16</td><td>[2, 2, 2, 2, 2, 18]</td><td>685.5M</td><td>85.0</td></tr></table>

Table 5: The capacity ablation of AR-NF actor on RoboMimic Square-MH. SR denotes success rate (%).

685.5M parameters, the XL configuration reaches an 85% success rate. These results indicate that suficiently large actor capacity can better capture the complex, precisionsensitive action distributions required by Square-MH.

## 5 Conclusion

We introduced RoMAN-Flow, a likelihood-based ofline reinforcement learning framework tailored for autoregressive normalizing flow policies in robot manipulation. RoMAN-Flow addresses the two main challenges of AR-NFs: NF-IQL avoids costly autoregressive sampling during post-training, while one-step distillation removes sequential inversion at deployment. Experiments on simulation and real-robot benchmarks demonstrate efective policy improvement and an 8.55× inference speedup with limited performance loss. Our work establishes AR-NFs as a practical policy family for likelihood-based ofline robotic reinforcement learning.

## Acknowledgments

We would like to extend our deepest appreciation to Qinyu Zhao for insightful discussions.

## References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Bjorck, J.; Castañeda, F.; Cherniadev, N.; Da, X.; Ding, R.; Fan, L.; Fang, Y.; Fox, D.; Hu, F.; Huang, S.; et al. 2025. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734.

Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; et al. 2024. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164.

Bu, Q.; Yang, Y.; Cai, J.; Gao, S.; Ren, G.; Yao, M.; Luo, P.; and Li, H. 2025. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111.

Chen, K.; Liu, Z.; Zhang, T.; Guo, Z.; Xu, S.; Lin, H.; Zang, H.; Zhang, Q.; Yu, Z.; Fan, G.; et al. 2025. πRL: Online rl fine-tuning for flow-based vision-language-action models. arXiv preprint arXiv:2510.25889.

Chen, R. T.; Rubanova, Y.; Bettencourt, J.; and Duvenaud, D. 2018. Neural ordinary diferential equations. arXiv preprint arXiv:1806.07366.

Chi, C.; Xu, Z.; Feng, S.; Cousineau, E.; Du, Y.; Burchfiel, B.; Tedrake, R.; and Song, S. 2025. Difusion policy: Visuomotor policy learning via action difusion. The International Journal ofRobotics Research, 44(10-11): 1684–1704.

Dinh, L.; Krueger, D.; and Bengio, Y. 2014. Nice: Nonlinear independent components estimation. arXiv preprint arXiv:1410.8516.

Dinh, L.; Sohl-Dickstein, J.; and Bengio, S. 2017. Density estimation using Real NVP. In International Conference on Learning Representations.

Fujimoto, S.; and Gu, S. S. 2021. A minimalist approach to ofline reinforcement learning. Advances in neural information processing systems, 34: 20132–20145.

Geng, S.; Pacchiano, A.; Kolobov, A.; and Cheng, C.-A. 2023. Improving ofline rl by blending heuristics. arXiv preprint arXiv:2306.00321.

Grathwohl, W.; Chen, R. T.; Bettencourt, J.; Sutskever, I.; and Duvenaud, D. 2018. Ffjord: Free-form continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367.

Gu, J.; Chen, T.; Berthelot, D.; Zheng, H.; Wang, Y.; Zhang, R.; Dinh, L.; Bautista, M. A.; Susskind, J.; and Zhai, S. 2026. Starflow: Scaling latent normalizing flows for highresolution image synthesis. Advances in Neural Information Processing Systems, 38: 120986–121022.

Hansen-Estruch, P.; Kostrikov, I.; Janner, M.; Kuba, J. G.; and Levine, S. 2023. Idql: Implicit q-learning as an actor-critic method with difusion policies. arXiv preprint arXiv:2304.10573.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising difusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Huang, D.; Fang, Z.; Zhang, T.; Li, Y.; Zhao, L.; and Xia, C. 2025. Co-rft: Eficient fine-tuning of vision-languageaction models through chunked ofline reinforcement learning. arXiv preprint arXiv:2508.02219.

Hung, C.-Y.; Sun, Q.; Hong, P.; Zadeh, A.; Li, C.; Tan, U.; Majumder, N.; Poria, S.; et al. 2025. Nora: A small opensourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854.

Kang, B.; Ma, X.; Du, C.; Pang, T.; and Yan, S. 2023. Eficient difusion policies for ofline reinforcement learning. Advances in Neural Information Processing Systems, 36: 67195–67212.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, P.; et al. 2024. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246.

Kingma, D. P.; and Dhariwal, P. 2018. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31.

Kingma, D. P.; Salimans, T.; Jozefowicz, R.; Chen, X.; Sutskever, I.; and Welling, M. 2016. Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems, 29.

Kobyzev, I.; Prince, S. J.; and Brubaker, M. A. 2020. Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence, 43(11): 3964–3979.

Kolesnikov, A.; Pinto, A. S.; and Tschannen, M. 2024. Jet: A modern transformer-based normalizing flow. arXiv preprint arXiv:2412.15129.

Kostrikov, I.; Nair, A.; and Levine, S. 2021. Ofline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169.

Lipman, Y.; Chen, R. T.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2022. Flow matching for generative modeling. In The eleventh international conference on learning representations.

Liu, B.; Zhu, Y.; Gao, C.; Feng, Y.; Liu, Q.; Zhu, Y.; and Stone, P. 2023. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791.

Lu, Y.; Sun, Q.; Wang, X.; Jiang, Z.; Zhao, H.; and He, K. 2025. Bidirectional normalizing flow: From data to noise and back. arXiv preprint arXiv:2512.10953.

Mandlekar, A.; Xu, D.; Wong, J.; Nasiriany, S.; Wang, C.; Kulkarni, R.; Fei-Fei, L.; Savarese, S.; Zhu, Y.; and Martín-Martín, R. 2021. What matters in learning from ofline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298.

Mathieu, E.; and Nickel, M. 2020. Riemannian continuous normalizing flows. Advances in neural information processing systems, 33: 2503–2515.

Nair, A.; Gupta, A.; Dalal, M.; and Levine, S. 2020. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359.

Papamakarios, G.; Nalisnick, E.; Rezende, D. J.; Mohamed, S.; and Lakshminarayanan, B. 2021. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57): 1–64.

Papamakarios, G.; Pavlakou, T.; and Murray, I. 2017. Masked autoregressive flow for density estimation. Advances in neural information processing systems, 30.

Patacchiola, M.; Shysheya, A.; Hofmann, K.; and Turner, R. E. 2024. Transformer neural autoregressive flows. arXiv preprint arXiv:2401.01855.

Peng, X. B.; Kumar, A.; Zhang, G.; and Levine, S. 2019. Advantage-weighted regression: Simple and scalable of-policy reinforcement learning. arXiv preprint arXiv:1910.00177.

Physical Intelligence; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; et al. 2025. π<sub>0.5</sub>: A Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054.

Rafailov, R.; Sharma, A.; Mitchell, E.; Ermon, S.; Manning, C. D.; and Finn, C. 2023. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290.

Rezende, D.; and Mohamed, S. 2015. Variational inference with normalizing flows. In International conference on machine learning, 1530–1538. PMLR.

Richemond, P. H.; Tang, Y.; Guo, D.; Calandriello, D.; Azar, M. G.; Rafailov, R.; Pires, B. A.; Tarassov, E.; Spangher, L.; Ellsworth, W.; et al. 2024. Ofline regularised reinforcement learning for large language models alignment. arXiv preprint arXiv:2405.19107.

Shukor, M.; Aubakirova, D.; Capuano, F.; Kooijmans, P.; Palma, S.; Zouitine, A.; Aractingi, M.; Pascal, C.; Russi, M.; Marafioti, A.; et al. 2025. Smolvla: A vision-language-action model for afordable and eficient robotics. arXiv preprint arXiv:2506.01844.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971.

Wang, Z.; Hunt, J. J.; and Zhou, M. 2022. Difusion policies as an expressive policy class for ofline reinforcement learning. arXiv preprint arXiv:2208.06193.

Wen, J.; Zhu, Y.; Li, J.; Zhu, M.; Tang, Z.; Wu, K.; Xu, Z.; Liu, N.; Cheng, R.; Shen, C.; et al. 2025. Tinyvla: Towards fast, data-eficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters.

Yang, C.; Tarasov, D.; Liconti, D.; Guntz, R.; Zheng, H.; and Katzschmann, R. K. 2026. SERNF: Sample-Eficient Real-World Dexterous Policy Fine-Tuning via Action-Chunked Critics and Normalizing Flows. arXiv preprint arXiv:2602.09580.

Yu, T.; Quillen, D.; He, Z.; Julian, R.; Hausman, K.; Finn, C.; and Levine, S. 2020. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, 1094–1100. PMLR.

Ze, Y.; Zhang, G.; Zhang, K.; Hu, C.; Wang, M.; and Xu, H. 2024. 3d difusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954.

Zhai, S.; Zhang, R.; Nakkiran, P.; Berthelot, D.; Gu, J.; Zheng, H.; Chen, T.; Bautista, M. A.; Jaitly, N.; and Susskind, J. M. 2025. Normalizing flows are capable generative models. In Forty-second International Conference on Machine Learning.

Zhao, Q.; Zheng, G.; Yang, T.; Zhu, R.; Leng, X.; Gould, S.; and Zheng, L. 2025. SimFlow: Simplified and End-to-End Training of Latent Normalizing Flows. arXiv preprint arXiv:2512.04084.

Zheng, G.; Zhao, Q.; Yang, T.; Xiao, F.; Lin, Z.; Wu, J.; Deng, J.; Zhang, Y.; and Zhu, R. 2025. Farmer: Flow autoregressive transformer over pixels. arXiv preprint arXiv:2510.23588.

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; et al. 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, 2165– 2183. PMLR.

## 6 Implementation Details

## 6.1 Training Pipeline

The training pipeline is summarized in Algorithm 1. We first train the AR-NF policy on ofline action chunks through maximum-likelihood imitation learning. NF-IQL post-training then proceeds in two phases: we first freeze the actor and warm up the prefix-level critics and state-value function, and subsequently update the critics, value function, and actor using advantage-weighted likelihood optimization over ofline action chunks. Finally, we freeze the post-trained AR-NF policy as the teacher and train a one-step student to approximate its inverse mapping through intermediatestate alignment and action-chunk reconstruction, optionally augmented with teacher trajectories generated from priorsampled latents. The pipeline outputs the post-trained AR-NF policy and the distilled one-step student.

## 6.2 Reward Relabeling

To alleviate the credit-assignment dificulty caused by sparse rewards, we adopt trajectory-level reward relabeling inspired by HUBL (Geng et al. 2023). For trajectory n, we define the terminal-aware discount as $\gamma _ { n , t } = \dot { \gamma } ( 1 - \dot { d } _ { n , t } )$ , where $d _ { n , t }$ denotes the terminal indicator, and compute the Monte Carlo return as

$$
G _ { n , t } = r _ { n , t } + \gamma _ { n , t } G _ { n , t + 1 } .\tag{18}
$$

Let $T _ { n }$ denote the number of transitions in trajectory n. We define its trajectory score as the mean Monte Carlo return,

$$
s _ { n } = { \frac { 1 } { T _ { n } } } \sum _ { t = 0 } ^ { T _ { n } - 1 } G _ { n , t } ,\tag{19}
$$

and determine a trajectory-level mixing coeficient according to its empirical rank in the ofline dataset:

$$
\lambda _ { n } = \alpha \frac { 1 } { N } \sum _ { m = 1 } ^ { N } \mathbb { I } [ s _ { m } \leq s _ { n } ] ,\tag{20}
$$

where α controls the relabeling strength. We then define the relabeled reward and bootstrap discount as

$$
\widetilde { r } _ { n , t } = r _ { n , t } + \gamma _ { n , t } \lambda _ { n } G _ { n , t + 1 } , \qquad \widetilde { \gamma } _ { n , t } = \gamma _ { n , t } \big ( 1 - \lambda _ { n } \big ) .\tag{21}
$$

This formulation adaptively interpolates between the observed Monte Carlo return and value-function bootstrapping, assigning greater reliance on observed future returns to higher-return trajectories. For action chunks, we recursively accumulate the relabeled rewards and discounts to construct the TD target for each action prefix.

## 7 Experimental Details

## 7.1 Model and Training Configurations

RoMAN-Flow is trained in three stages: imitation learning, NF-IQL post-training, and one-step distillation. The complete optimization procedure is summarized in Algorithm 1; here, we describe the benchmark-specific model and training configurations.

Algorithm 1: RoMAN-Flow Training and Distillation   
Input: Ofline dataset D   
Output: Post-trained policy $\pi _ { \theta ^ { \star } }$ and one-step student $g _ { \psi ^ { \star } }$   
1: Initialize $\pi _ { \boldsymbol { \theta } } , \{ Q _ { \omega _ { k } } , Q _ { \bar { \omega } _ { k } } \} _ { k = 1 } ^ { M } ,$ and $V _ { \phi } .$   
2: Stage I: Imitation learning   
3: for $n = 1$ to $N _ { \mathrm { I L } }$ do   
4: Sample $\left( c _ { t } , \mathbf { a } _ { t } \right) \sim \mathcal { D } .$   
5: Update θ with $\mathbf { \dot { \mathcal { L } } } _ { \mathrm { I L } } = - \mathbb { E } [ \log \pi _ { \theta } ( \mathbf { a } _ { t } \mid c _ { t } ) ]$   
6: end for   
7: Stage II: NF-IQL post-training   
8: Freeze π<sub>θ</sub> during value warm-up.   
9: for $n = 1 , \ldots , N _ { \mathrm { w a r m } }$ do   
10: Sample $\boldsymbol { B } \sim \mathcal { D }$ and construct prefix targets $y _ { t , j } .$   
11: Update the critics and value function using ${ \check { \mathcal { L } } } _ { Q }$ and   
$\bar { \mathcal { L } _ { V } }$   
12: Polyak-update the target critics.   
13: end for   
14: Unfreeze $\pi _ { \theta } .$   
15: for $n = 1$ to $N _ { \mathrm { I Q L } }$ do   
16: Update the critics and value function.   
17: Compute $A _ { t } = \bar { Q } _ { \bar { \omega } } ( c _ { t } , \mathbf { a } _ { t } ) - V _ { \phi } ( c _ { t } ) .$   
18: Update θ using $\mathcal { L } _ { \pi } = - \mathbb { E } [ e ^ { \beta A _ { t } }$ log $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t } \mid c _ { t } ) \big ]$   
19: end for   
20: Stage III: One-step distillation   
21: Freeze π ⋆ and initialize $g _ { \psi }$   
22: for $n = 1$ to $N _ { \mathrm { a l i g n } }$ do   
23: Sample $\left( c _ { t } , \mathbf { a } _ { t } \right) \sim \mathcal { D } .$   
24: Run the teacher forward on $\mathbf { a } _ { t }$ to obtain $\mathbf { z } _ { t }$ and   
$\{ \mathbf { h } _ { t } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$   
25: Run $g _ { \psi } ( \mathbf { z } _ { t } ; c _ { t } )$ to obtain $\{ \widehat { \mathbf { h } } _ { t } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ and $\widehat { \mathbf { a } } _ { t }$   
26: Sample $\mathbf { z } _ { t } ^ { p } \sim p _ { 0 }$   
27: Run the teacher inverse under $c _ { t }$ to obtain $\{ \mathbf { u } _ { t , p } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$   
and $\mathbf { a } _ { t } ^ { p } .$   
28: Run $g _ { \psi } ( \mathbf { z } _ { t } ^ { p } ; c _ { t } )$ to obtain $\{ \widehat { \mathbf { u } } _ { t , p } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ and $\widehat { \mathbf { a } } _ { t } ^ { p } .$   
29: Compute ${ \mathcal { L } } _ { \mathrm { d a t a } }$ using $\{ \widehat { \mathbf { h } } _ { t } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ and $\{ \mathbf { h } _ { t } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 } , \widehat { \mathbf { a } } _ { t }$   
and $\mathbf { a } _ { t } .$   
30: Compute ${ \mathcal { L } } _ { \mathrm { p r i o r } }$ using $\{ \widehat { \mathbf { u } } _ { t , p } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ and $\{ \mathbf { u } _ { t , p } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 } , \widehat { \mathbf { a } } _ { t } ^ { p }$   
and $\mathbf { a } _ { t } ^ { p } .$   
31: Update ψ using $\begin{array} { r } { \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r i o r } } . } \end{array}$   
32: end for   
33: return $\pi _ { \theta ^ { \star } }$ and $g _ { \psi ^ { \star } }$

Backbones and observations. For LIBERO, MetaWorld-MT50, and the real-robot experiments, we use SmolVLM-500M-Instruct as the multimodal encoder for visual observations, language instructions, and proprioceptive states. The suite-specific LIBERO policies are initialized from a shared checkpoint pretrained on LIBERO-90 for 100k steps. For RoboMimic MH, we follow the perception setup of SERNF (Yang et al. 2026) for a controlled comparison, replacing the VLM with two separate ImageNet-pretrained ResNet-18 encoders for the third-person and wrist-camera RGB observations.

Model sizes. The SmolVLM-based AR-NF policy used for imitation learning and NF-IQL contains approximately 1.45B inference-time parameters, while the corresponding one-step student contains approximately 0.56B parameters. The ResNet-based AR-NF policy and one-step student used on RoboMimic contain approximately 0.95B and 78.2M parameters, respectively. Table 6 summarizes the model architectures and training hyperparameters.

## 7.2 Evaluation Protocols

All policies are evaluated from fixed checkpoints without parameter updates. We use episode-level success rate as the primary metric and execute each predicted action chunk completely before replanning.

LIBERO. We follow the oficial LIBERO protocol (Liu et al. 2023) in terms of task suites, success criteria, and initial states. We evaluate LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, each containing ten tasks. Each task is evaluated over 50 episodes, resulting in 500 episodes per suite. All policies predict and execute 16- step action chunks.

MetaWorld-MT50. Following the evaluation protocol of π<sub>RL</sub> (Chen et al. 2025), we evaluate each of the 50 tasks over ten episodes using four parallel environments and a maximum episode length of 200 steps. All policies predict and execute 5-step action chunks. We report the unweighted mean success rate across the Easy, Medium, Hard, and Very Hard groups.

RoboMimic MH. We follow the training and evaluation protocol of SERNF (Yang et al. 2026) on Lift, Can, and Square. For each offour training seeds, we evaluate the policy on the same 100 fixed initial states using episode seeds 0– 99, yielding 400 episodes per task and training stage. Each evaluation uses a single environment, a maximum episode length of 500 steps, and 10-step action chunks.

Real-robot evaluation. For real-robot evaluation, we use a dedicated GPU inference server and a separate robotcontrol workstation. Target objects are initialized within predefined evaluation regions that include locations outside the initial-position distribution represented in the ofline demonstrations. The resulting protocol therefore covers both in-distribution and spatial out-of-distribution initializations. Each policy predicts 16-step action chunks.

## 8 Benchmarks and Qualitative Examples 8.1 Metaworld

MetaWorld-MT50 is a public multitask ofline manipulation dataset released by LeRobot and built upon the MetaWorld simulation benchmark (Yu et al. 2020). It covers 50 tabletop manipulation tasks performed by a Sawyer robot, spanning button pressing, articulated- object interaction, object grasping and transport, precise assembly, and long-horizon placement. The dataset contains 2,500 expert trajectories and 204,806 transitions, corresponding to 50 trajectories per task. Each sample provides a single-view RGB observation, a 4- dimensional robot state, a 4-dimensional continuous Cartesian action, and a natural-language task description. Following the task partition adopted by π<sub>RL</sub> (Chen et al. 2025), the 50 tasks are divided into Easy, Medium, Hard, and Very Hard groups containing 28, 11, 6, and 5 tasks, respectively. Figure 5 visualizes representative rollouts from the four dificulty groups: Door Open, Bin Picking, Assembly, and Shelf Place, illustrating articulated-object interaction, cross-bin pick-and-place, precise assembly, and long-horizon elevated placement.

## 8.2 LIBERO

LIBERO (Liu et al. 2023) is a public language-conditioned manipulation benchmark designed to study knowledge transfer in multitask and lifelong robot learning. The complete benchmark contains 130 tabletop manipulation tasks performed by a Franka Panda robot. Following the common evaluation convention in recent VLA studies, we consider four ten-task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, where LIBERO-Long corresponds to LIBERO-10 in the original benchmark. Spatial, Object, and Goal isolate knowledge transfer across spatial layouts, object identities, and language-specified goals, respectively, whereas Long contains longer-horizon tasks involving multiple consecutive manipulation stages. Each task provides 50 human-teleoperated demonstrations, resulting in 2,000 trajectories across the four evaluation suites. RoMAN-Flow uses two RGB views, an 8-dimensional proprioceptive state, and a language instruction. The suite-specific policies are initialized from a shared checkpoint pretrained on the 90 tasks and 4,500 demonstrations of LIBERO-90. Figure 6 presents representative RoMAN-Flow rollouts across the LIBERO suites.

## 8.3 Robomimic

RoboMimic (Mandlekar et al. 2021) is a public benchmark for learning robot manipulation policies from ofline demonstrations. Its Multi-Human (MH) datasets contain behaviorally diverse demonstrations collected from six human operators with diferent levels of proficiency. We consider three tasks: Lift, which requires grasping and lifting a cube; Can, which requires transferring a can into a target bin; and Square, which requires precisely placing a square nut onto its corresponding peg. Each task provides 300 demonstrations, resulting in 900 trajectories and 174,614 transitions across the three tasks. To enable a controlled comparison with SERNF (Yang et al. 2026), RoMAN-Flow does not use a VLM or proprioceptive inputs in these experiments. Instead, it observes two 84 × 84 RGB views from a third-person camera and a wristmounted camera and predicts 7-dimensional continuous control commands. The three tasks impose progressively greater manipulation demands, with Square requiring a longer sequence of precise grasping, alignment, and insertion behaviors. Figure 7 presents representative RoMAN-Flow rollouts on Lift, Can, and Square.

<table><tr><td></td><td colspan="4">RoMAN-Flow (IL / NF-IQL)</td><td>RoMAN-Flow (One-Step)</td></tr><tr><td>Setting</td><td>LIBERO</td><td>MT50</td><td>RoboMimic</td><td>Real Robot</td><td>Default</td></tr><tr><td colspan="6">Model configuration</td></tr><tr><td>Hidden dimension</td><td>1152</td><td>1152</td><td>1152</td><td>1152</td><td>512</td></tr><tr><td>Attention heads</td><td>16</td><td>16</td><td>16</td><td>16</td><td>8</td></tr><tr><td>Flow / ViT blocks</td><td>6</td><td>6</td><td>6</td><td>6</td><td>7</td></tr><tr><td>Inference parameters</td><td>1.45B</td><td>1.45B</td><td>0.95B</td><td>1.45B</td><td>0.56B / 78.2M</td></tr><tr><td colspan="6">Training configuration</td></tr><tr><td>Training steps</td><td>20k / 50k</td><td>100k / 60k</td><td>30k / 50k</td><td>50k / 30k</td><td>20k</td></tr><tr><td>Global batch size</td><td>64 / 64</td><td>128 / 128</td><td>144 / 144</td><td> $3 2 / 3 2$ </td><td>64</td></tr><tr><td>Actor LR</td><td> $5 { \times } 1 0 ^ { - 5 } / 2 { \times } 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 4 } / 5 { \times } 1 0 ^ { - 6 }$ </td><td> $1 { \times } 1 0 ^ { - 4 } / 2 { \times } 1 0 ^ { - 4 }$ </td><td> $5 { \times } 1 0 ^ { - 5 } / 1 { \times } 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Student LR</td><td></td><td></td><td></td><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Critic LR</td><td> $- / 1 \times 1 0 ^ { - 4 }$ </td><td> $- / 5 { \times } 1 0 ^ { - 5 }$ </td><td> $- / 2 { \times } 1 0 ^ { - 4 }$ </td><td> $- / 5 { \times } 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Expectile τ</td><td>-/0.80</td><td>-/0.75</td><td>-/0.75</td><td>-/0.75</td><td></td></tr><tr><td>Advantage temperature β</td><td>-/10</td><td>-/45</td><td>-/10</td><td>-/3</td><td></td></tr><tr><td>Advantage clip</td><td>-/100</td><td>-/5</td><td>-/20</td><td>-/20</td><td></td></tr><tr><td>HUBL coefficient</td><td>-/0.20</td><td>-10</td><td>-/0.15</td><td>-/0</td><td></td></tr><tr><td>Critic warm-up</td><td>-/1k</td><td>-/5k</td><td>-/5k</td><td>-/1k</td><td></td></tr><tr><td>Discount γ</td><td>-/0.995</td><td>-/0.995</td><td>-/0.997</td><td>-/0.997</td><td></td></tr><tr><td>Target update rate</td><td>-/0.005</td><td>-/0.005</td><td>-/0.05</td><td>-/0.005</td><td></td></tr><tr><td>Prior-loss weight</td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>Prior-sample fraction</td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>Prior-sampling temperature</td><td></td><td></td><td></td><td></td><td>0.5</td></tr><tr><td>EMA decay</td><td></td><td></td><td></td><td></td><td>0.9999</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr></table>

Table 6: Model configurations and training hyperparameters. Paired entries under RoMAN-Flow (IL / NF-IQL) report the IL and NF-IQL settings, respectively, while single entries are shared by both stages. The One-Step column reports the default distillation configuration, with its two parameter counts corresponding to the SmolVLM- and ResNet-based students. A dash denotes an inapplicable setting.

![](images/d2a19839c2c5a034675150753a381639f72afc2dbabede8c63544c944ef7060d.jpg)  
Shelf place

Figure 5: Qualitative results of RoMAN-Flow on representative MetaWorld-MT50 manipulation tasks.

Spatial

![](images/4786de86fbf2c95bdef64b30f7a5889316db8da4263483f7b7417fa50cc47520.jpg)

![](images/54e5dbae734b3139e7fff3b101b09791e559449ca0c8960f4d1d74e5aa2709cc.jpg)

![](images/0975981ab77b2553554238e7e055ae8b2e05d1606f8ea83ea1cf24153d845788.jpg)

![](images/5e9377c63c1e37ace4d34be2a297bdd245a451e9a1922f445efff4db23c8617c.jpg)

![](images/c6a92ddf340abd6a04d7f81e0c6de7ed82cb93cd23cd2bdf036007bafb439fd4.jpg)

![](images/186a4ba6f52c1e0731525a94509c7ef3307279bd2eb047f1d3a5407c394a584b.jpg)

![](images/68c5899ca2ad7632665462b3d375256210488cc89431e7ec83c2cb8a4b2bd4dd.jpg)

![](images/4a1831913bf98edbdc7b6ba8bb3b506b024f939fc18767046b66d6872f6488f3.jpg)

![](images/3d73a73c308c8c46e9dbc6747c4d5b24a7868b5bad0dffc0327fe927c7cdfbf2.jpg)  
Pick up the black bowl from table center and place it on the plate

Object

![](images/c3948f696843eeaf31c58d6483762f5dffb01a12808bd29239d138c822f28100.jpg)

![](images/6ab2f280f376e905a71c26bdc0b061bc28f0cc8e045733dc8aafdcf1819e2c08.jpg)

![](images/17af76f604aabe7a74a3b9a43cea4e6590632805db55c7727e19e8a6853a9cf8.jpg)

## Pick up the butter and place it in the basket

Goal

![](images/1d5d93b023344bb64d3b56bb2e544ff9971e24cb6d5905447f7d9379154cc3e5.jpg)

![](images/b83e57d9fde968d78e8d1b8c37964d5b2d601ad4be6f494e1dd85c3de476be32.jpg)  
Open the middle drawer of the cabinet

![](images/c6e1e2ac85c90248426975b2f37ec0bb806651bd813ef576389770d1960c1f66.jpg)

Long

![](images/0bd06839ec61f7f67104024a25f6a5bcbfcf585dc91174eee8c2f5edd10a781b.jpg)

![](images/fd0d708aec9e87c0f9bacfbfcef0ef613b53985aef1b069b6291b078743574ec.jpg)  
Put the white mug on the left plate and put the yellow and white mug on the right plate

Figure 6: Qualitative rollouts of RoMAN-Flow on representative tasks from the LIBERO benchmark.  
![](images/b6de7f236d47545cef562b01b77db5f2912f22b175df22567d59e51f127d46d8.jpg)

![](images/5c85cc5d8df32ad9ac5cb5a5093999ead564a984b65476da4cf3aa157cbb1f7d.jpg)  
Pick up the square nut and place it onto the square peg

Figure 7: Qualitative rollouts of RoMAN-Flow on the Lift, Can, and Square tasks from RoboMimic MH.