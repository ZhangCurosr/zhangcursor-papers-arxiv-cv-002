# Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization

Andy Zeyi Liu<sup>1,\*</sup> Haoran Sun<sup>1,\*</sup> Lucas Baker<sup>2</sup>

Randall Balestriero<sup>3</sup> John Sous<sup>1,†</sup>

<sup>1</sup>Yale University <sup>2</sup>Jump Trading <sup>3</sup>Brown University Equal contribution <sup>†</sup>Correspondence: john.sous@yale.edu

## Abstract

Joint-Embedding Predictive Architecture (JEPA) world models learn a compact latent representation of the world that supports prediction and planning, but their capability to learn physics and generate physically realistic dynamics remains hitherto untested. In this work, we introduce SemiGroup-JEPA (SG-JEPA), which extends the LeWorldModel framework by supplying the parameter governing the physics to the temporal model via action-conditioning and jointly training an encoder and predictor through an autoregressive latent rollout. To evaluate the model’s ability to generalize out of distribution, we design dynamical tasks under diferent gravitational fields that, despite obeying the same physical law, exhibit qualitatively diferent dynamics, ranging from floating motion in weak gravitational fields to rapid bouncing in strong ones. In contrast to DINO-WM, SG-JEPA reduces open-loop prediction error by up to 2× on two-dimensional datasets, and increases control success rate up to 2.5× for three-dimensiona robotic datasets, for which we train independent difusion policies. To explain this advantage, we develop a linear feature model that separates local law-conditioned error from its recursive amplification under rollout. Guided by this model, we find that back-propagating the multi-step rollout loss into the representation trains the encoder to keep the features that the predictor can carry forward, and that those are the features the dynamics depend on, so most of the gain comes from the encoder learning better features rather than from the predictor learning better dynamics. See project page at sg-jepa.github.io.

## 1 Introduction

The long-standing goal of world models is to understand dynamics of the environment in which the model agent is deployed so that the agent can predict subsequent states and consequences of its actions. Joint-Embedding Predictive Architecture (JEPA) emerged as a promising approach, whereby rather than reconstructing future pixels, it learns latent representations by jointly training an encoder and predictor [LeCun, 2022, Sobal et al., 2025, Zhou et al., 2025]. Recent proposals such as DINO-WM [Zhou et al., 2025] and LeWorldModel (LeWM) [Maes et al., 2026] have demonstrated that these latent predictions support downstream planning and control under a fixed frame-to-frame transition law. We push this further by asking the question:

![](images/f63779d3516ea18122bd647ba8b8f0c635858550cc262019b5088645318484ba.jpg)  
Figure 1: SG-JEPA training and control. (a) The encoder and gravity-conditioned predictor are jointly trained with a discounted K-step autoregressive rollout loss and SIGReg. Targets use the same trainable encoder. (b) A gravity-conditioned difusion policy is trained on demonstrations from successful episodes with the encoder frozen. At inference, it generates $a _ { t : t + A - 1 }$ , executes the first E actions, and replans from new observations.

If the answer is yes, the consequences are significant. A world model that genuinely learns dynamics would enable an agent to adapt in new environments, e.g. a model trained on Earth dynamics generalizing to Martian dynamics.

In this work, we take gravitational dynamics as the primary testbed for these questions, and introduce Semigroup-JEPA (SG-JEPA), a gravity-conditioned extension of LeWM with recursive physical prediction (Fig. 1). SG-JEPA supplies gravity alongside action and jointly trains its encoder and predictor through a discounted autoregressive rollout loss, while SIGReg [Balestriero and LeCun, 2025] regularizes the latent space. On various MuJoCo datasets covering two-dimensional (2D) rigid body freefall, three-dimensional (3D) projectile motion and robot-arm control, we train SG-JEPA on a narrow range of gravity values and evaluate on a wider grid that extends far beyond it. We study two aspects of generalization, namely transfer to OOD gravity values and accuracy under long-horizon rollout. Our results show that the dynamics SG-JEPA learns within a narrow gravity range remain accurate over long horizons at OOD gravity values, demonstrating zero-shot physics generalization.

Across these gravitational environments we find that how well the encoder preserves dynamical information determines the ability to generalize out of distribution. To interpret this behavior, we mechanistically isolate the encoder from the predictor and measure how performance depends on the rollout horizon and on the distance of the gravity parameter from the training range. Starting from a trained checkpoint, we replace either the encoder or the predictor with a freshly initialized one and retrain jointly. Re-training the encoder leaves performance unchanged, whereas re-training the predictor degrades it. The gain is therefore associated with what the GRU predictor learns, and this advantage widens as gravity moves further from the training range and as the rollout horizon grows.

Related work. JEPA models introduce a new paradigm of learning by predicting target representations rather than reconstructing pixels, and have been extended from images to video [LeCun, 2022, Assran et al., 2023, Bardes et al., 2024, Assran et al., 2025]. DINO-WM [Zhou et al., 2025] applies this idea with frozen DINOv2 features [Oquab et al., 2024], while LeWM [Maes et al., 2026] jointly trains its encoder and predictor with SIGReg [Balestriero and LeCun, 2025]. Recent work probes physics generalization under held-out configurations and asks which physical parameters predictive latents retain [Xue et al., 2026, Tan et al., 2026]. Our setting difers in isolating gravity as a physical quantity governing the dynamics, which lets us systematically test the learned dynamics outside the training range. A separate line of work addresses rollout error by training on the model’s own predictions or optimizing beyond one step [Bengio et al., 2015, Venkatraman et al., 2015, Lambert et al., 2021]. In this work, we address these two problems jointly and introduce SG-JEPA, which back-propagates a recursive latent rollout loss through both the encoder and predictor, enabling, as we show below, OOD generalization over long-horizon dynamics. The name reflects that, for action-free trajectories at a fixed gravity value, the repeated updates form a discrete semigroup. Related operator-based analyses of how local representation and predictor errors propagate appear in [Williams et al., 2015, Lusch et al., 2018]. Appendix A provides more discussion.

## 2 Problem Setting

In this section, we introduce Semigroup-JEPA (SG-JEPA). We first describe the architecture and training loss. We then introduce the customized, gravity-controlled MuJoCo environments and the baseline models we compare against. Finally we explain evaluation protocols for prediction of physical quantities and latent control.

## 2.1 Model architecture

Semigroup-JEPA. In Fig. 1 we sketch the architecture of SG-JEPA. It keeps the two components of LeWM, an encoder and a predictor. At time step t, the model has observation $o _ { t }$ and control $u _ { t }$ . The observation $o _ { t }$ is then encoded using a Vision Transformer [Dosovitskiy et al., 2021] into low-dimensional latents $z _ { t }$ . Unless stated otherwise, we use the ViT-Tiny configuration, with 12 Transformer layers, 3 attention heads, and hidden width 192. We use $8 \times 8$ patches for $1 2 8 \times 1 2 8$ resolution and $1 6 \times 1 6$ for 256 × 256 resolution. The encoder is trained from random initialization jointly with the predictor. We use the final layer [CLS] token $h _ { t } \in \mathbb { R } ^ { 1 9 2 }$ as a global summary of the observed frame $o _ { t }$ and map it to $z _ { t } \in \mathbb { R } ^ { 2 5 6 }$ using a one-hidden-layer projector. The latents $z _ { t }$ are then fed into the predictor. Gravity enters the model through the action. Each episode has a signed gravity scalar parameter $g _ { \mathrm { ; } }$ , which we treat as an extra coordinate of the action, and maintain constant across frames,

$$
\begin{array} { r } { \tilde { a } _ { t } = [ u _ { t } ; g ] , \qquad c _ { t } = q _ { \psi } ( \tilde { a } _ { t } ) , } \end{array}\tag{1}
$$

where $q _ { \psi }$ is the action encoder, a temporal convolution followed by a SiLU MLP, which yields $c _ { t } \in \mathbb { R } ^ { 2 5 6 }$ . The input $g$ is z-scored using statistics from the training split. In the case where there are no external actions, e.g. free-fall, we still input a single scalar g in the action conditioning.

We consider three choices of predictor, namely a Transformer, a GRU, and a state-space model (SSM). Unless otherwise noted, SG-JEPA refers to either the GRU or the SSM variant. The Transformer predictor has 6 layers and 16 attention heads, in line with the Original LeWM, with actions fed into every block through Adaptive Layer Normalization (AdaLN) [Peebles and Xie, 2023]. The GRU predictor defaults to 3 residual layers of width 512. In each layer, the hidden state is concatenated with the projected action embedding, passed through an MLP, and then processed by a single-layer GRU. The SSM predictor uses the same depth, width, and action-concatenation scheme, replacing each GRU with a Mamba/S6-style selective state-space block [Gu and Dao, 2024]. The predictor takes as input a history of length H and predicts the next latent

$$
\hat { z } _ { t + 1 } = p _ { \theta } ( z _ { t - H + 1 : t } , c _ { t - H + 1 : t } ) .\tag{2}
$$

Exact architectures and gravity-fusion ablations are given in Appendix C.

![](images/6b61ae1c3ab6721428d9f55263da287482e2280ba1072017d04d1263e375e445.jpg)  
(a) 2D shapes

![](images/32ff3b47ff70468379c389996432d2e2ab4682d222925ef1e89d37ce06bda29a.jpg)  
(b) Approach Ball

![](images/a92b00a313fd35ea1b6b04219724ef142377aa1c5884a64d1d911bbf149ca4f0.jpg)  
(c) Arm Catcher Ball

![](images/a26a63e4ccf42ff8956a80f28d11b50d9fb502c3c0c14cfc0c4d466e45287f9a.jpg)  
(d) Arm Paddle Ball

![](images/99d440a07fd6076ef1b6e26bb8a203594cfb5f395739c0fae4b5eeb6b8f8f9e8.jpg)  
(e) Franka Paddle-to-Basket  
Figure 2: Evaluation environments. (a) Planar shapes: each object is given an impulse at $t = 0$ and then fall and collide freely after inside the box. (b) Approach Ball: projectile flight from the far end of the scene. (c) Arm Catcher Ball: a robotic arm catches a ball with a catcher. (d) Arm Paddle Ball: the same arm sustaining bounces with a tilting paddle. (e) Franka Paddle-to-Basket: a Panda arm strikes an incoming ball into a basket. Orange arrows sketch the motion.

Training objective. For fixed g and an action-free trajectory, iterating the shared history update $F _ { \theta , g }$ gives $S _ { \theta , g } ( k ) = F _ { \theta , g } ^ { \circ k }$ , with $S _ { \theta , g } ( 0 ) = I$ and $S _ { \theta , g } ( k + \ell ) = S _ { \theta , g } ( \ell ) \circ S _ { \theta , g } ( k )$ . These iterates form a discrete semigroup on the latent history. Without controls, consecutive update blocks compose into this semigroup. With controls, they still compose, but each block is modified by its action (Appendix H.6).

SG-JEPA combines a K-step autoregressive rollout loss with SIGReg. Unlike LeWM’s one-step teacher-forced objective, in which the predictor is always given true encoded latents as input rather than its own predictions, it starts from H encoded frames and recursively predicts K future latents, inserting each prediction into the next length-H history window. With $z _ { s } ^ { \mathrm { r o l l } } = z _ { s }$ for $s \leq t$ and $z _ { s } ^ { \mathrm { r o l l } } = \hat { z } _ { s }$ for $s > t ,$ the rollout is $\hat { z } _ { t + k } = p _ { \theta } \big ( z _ { t + k - H : t + k - 1 } ^ { \mathrm { r o l l } } , c _ { t + k - H : t + k - 1 } \big )$ , with $k = 1 , \ldots , K$ . We use the normalized rollout loss

$$
\mathcal { L } _ { \mathrm { r o l l } } = \sum _ { k = 1 } ^ { K } w _ { k } \left\| \hat { z } _ { t + k } - z _ { t + k } \right\| _ { 2 } ^ { 2 } , \quad \mathrm { w h e r e } \quad w _ { k } = \frac { \gamma ^ { k - 1 } } { \sum _ { j = 1 } ^ { K } \gamma ^ { j - 1 } } .
$$

By construction, a composition law emerges from repeatedly applying the shared predictor. For $\gamma < 1$ , later predictions receive less weight, so accumulated errors do not dominate the objective. The full regularized objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r o l l } } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { S I G R e g } } . } \end{array}\tag{3}
$$

SIGReg acts only on encoded latents and encourages their random one-dimensional projections to match a standard Gaussian. Target latents come from the same trainable encoder, without stop-gradient.

Training setup. For training, we use a hybrid Muon/AdamW optimizer. Muon [Jordan et al., 2024] applies an orthogonalized update to matrix-valued parameters, so all 2D tensors are optimized with Muon at a learning rate of $1 \times 1 0 ^ { - 4 }$ and all other tensors with AdamW at $5 \times 1 0 ^ { - 5 }$ . In Appendix D.1.2 we perform a controlled comparison against pure AdamW [Loshchilov and Hutter, 2019], finding that Muon lowers the validation loss more rapidly and uses input data more eficiently. We train all models for 20 epochs. Unless stated otherwise, we use a history of $H = 2 0 , K = 5$ . The history-length ablation in Appendix D.2.1 demonstrates that this choice yields the best downstream performance and eficiency tradeof. Appendix D.3 shows that $\gamma = 0 . 9 5$ produces the strongest

overall position and velocity performance. We study the efect of changing $\lambda _ { \mathrm { s i g } }$ in Appendix D.1.3.   
Appendix C provides the complete training configuration.

## 2.2 Datasets

We generate eight datasets in MuJoCo [Todorov et al., 2012], covering passive rigid-body motion (freefall), projectile motion, and robot-arm control. We use six of them as environments for the experiments in the main text and the other two (2D pentagon and house) in the ablation studies in Appendix E.2. Fig. 2 gives an overview and illustrates the ball dynamics. For training, g is sampled from a narrow Gaussian, $\mathcal { N } ( 4 , 0 . 5 ^ { 2 } )$ for the 2D planar shapes and Arm Catcher Ball, and $\mathcal { N } ( 9 . 8 , 2 . 0 ^ { 2 } )$ for the remaining 3D datasets. Held-out test sets use a wider grid of $g$ values, including OOD ones. Five datasets test prediction: four planar shapes receive one initial impulse and then move freely inside a box, while Approach Ball contains 3D projectile motion. Three datasets test control: Arm Catcher Ball, Arm Paddle Ball, and Franka Paddle-to-Basket. Appendix B gives the exact data splits, action schemas and other dataset details.

Baselines. We compare the performance of SG-JEPA against the following baselines. We first compare against the Original LeWM, the vanilla model of [Maes et al., 2026], which uses a one-step latent prediction loss, SIGReg on the encoded latents, and a causal Transformer predictor. We also compare against DINO-WM [Zhou et al., 2025], for which we keep the pretrained DINOv2 encoder frozen and train an action-conditioned Transformer predictor, with action conditioning performed in the same way as in LeWM. We label the GRU- and SSM-predictor variants of SG-JEPA SG-JEPA (GRU) and SG-JEPA (SSM), respectively.

## 2.3 Downstream evaluation

Prediction. For prediction tasks, we freeze the trained world model and fit an MLP probe $r _ { \eta }$ that, given a short latent window, predicts the physical state $s _ { t }$ (position, velocity, and rotation in 2D; position and velocity in 3D) at time t. The probe is trained only on training episodes (one episode is a video sample). On held-out episodes we apply it to both encoded ground-truth frames and open-loop autoregressive rollouts at forecast horizons h up to $\mathfrak { h } _ { \operatorname* { m a x } } = 4 4$ frames. Let $\ell _ { r }$ denote the probe-window length. We report the excess physical-state error

$$
\frac { 1 } { \mathfrak { h } } \sum _ { k = 1 } ^ { \mathfrak { h } } \Big ( \mathrm { N M S E } \Big ( r _ { \eta } ( z _ { t + k - \ell _ { r } + 1 : t + k } ^ { \mathrm { r o l l } } ) , s _ { t + k } \Big ) - \mathrm { N M S E } \big ( r _ { \eta } ( z _ { t + k - \ell _ { r } + 1 : t + k } ) , s _ { t + k } \big ) \Big ) ,\tag{4}
$$

where NMSE denotes the normalized mean-squared error, with state coordinates normalized using statistics from the probe’s training set. Subtracting the probe error on encoded ground-truth latents isolates the error introduced by the rollout.

Control. For each control task and source model, we freeze the visual encoder and train a separate gravity-conditioned Difusion Policy [Chi et al., 2023] on its features. SG-JEPA and LeWM provide projected CLS latents, while DINO-WM provides mean-pooled frozen DINOv2 patch features. Fig. 1(b) shows the resulting control loop. A two-layer causal GRU compresses the most recent H feature vectors into $\bar { h } _ { t } = \mathrm { L a y e r N o r m } ( \mathrm { G R U } ( z _ { t - H + 1 : t } ) _ { \mathrm { f i n a l } } )$ , where LayerNorm denotes layer normalization, and the policy is conditioned on $\xi = [ \bar { h } _ { t } ; g _ { \mathrm { p o l } } ]$ , with $g _ { \mathrm { p o l } }$ the z-scored gravity. The noise predictor $\epsilon _ { \omega }$ is a conditional 1D U-Net. At difusion step τ , a control block $\mathbf { u } = u _ { t : t + A - 1 }$ of A steps from the training trajectories is corrupted as $\mathbf { u } ^ { ( \tau ) } = \sqrt { \bar { \alpha } _ { \tau } } \mathbf { u } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \epsilon ,$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$

![](images/bede1a4ee4c2a38208839784e19e3a3c37ef3f617cf3a76287dd411df9993a75.jpg)  
Figure 3: Long-horizon prediction at 44 rollout steps for 2D datasets, evaluated with MLP probes. Bars show position and velocity $L _ { 2 }$ error and cumulative-rotation mean-absolute error (MAE), normalized by the DINO-WM error for each shape and metric. The absolute DINO-WM errors are shown below each group. Results are averaged over held-out episodes for each shape across 25 gravity values from −2 to 10, with training gravity sampled from $\mathcal { N } ( 4 , 0 . 5 ^ { 2 } )$ . Lower is better.

and $\bar { \alpha } _ { \tau }$ is the cumulative coeficient of the cosine noise schedule. The U-Net is trained to predict the added noise from $\mathbf { u } ^ { ( \tau ) }$ , τ , and the conditioning vector ξ by minimizing

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P } } = \mathbb { E } _ { \tau , \epsilon } \bigg [ \Big \| \epsilon _ { \omega } ( \mathbf u ^ { ( \tau ) } , \tau \mid \xi ) - \epsilon \Big \| _ { 2 } ^ { 2 } \bigg ] . } \end{array}\tag{5}
$$

We train the GRU and U-Net while keeping the visual encoder frozen. All policies use $H = 2 0$ $A = 1 6 ,$ a 256-dimensional GRU, and a 100-step cosine difusion schedule. At inference, the policy predicts A controls, executes the first $E ,$ and replans from the new observation. We use $E = 8$ for Arm Catcher Ball and Franka Paddle-to-Basket and E = 4 for Arm Paddle Ball. Note that this execution interval is distinct from the world-model training rollout length K.

Our tasks difer from static-goal benchmarks such as Two Rooms and Push-T in that the ball moves, so the policy must infer its future motion from the observation history and gravity. Using a goal-conditioned Cross-Entropy Method planner in the same way would require a future target frame, which would give away the ball’s future position. Our policies therefore do not use one.

## 3 Empirical results

In this section, we compare SG-JEPA against LeWM and DINO-WM across our 2D and 3D datasets. SG-JEPA improves long-horizon prediction and achieves higher average control success rates than the Original LeWM and DINO-WM, particularly at OOD gravity values.

2D datasets. In Fig. 3 we present long-horizon physical-state prediction results. Each episode contains 64 frames and our models take a history of H = 20, so at test time we roll out for 44 steps and probe the position, velocity, and rotation. Note that rotation is computed by integrating the probed angular velocity over the rollout, and its error is measured in number of turns. Complete per-gravity and error-by-horizon results are detailed in Appendix E. On the square object, we find that both variants of SG-JEPA have a clear advantage: relative to DINO-WM, SG-JEPA (GRU) reduces the three error types by 31–48%, with SG-JEPA (SSM) close behind, while Original

![](images/8b421ed8e3354ee786263627a5a740ece3cd04b72703b5245bc0a9dedfbb467e.jpg)  
Figure 4: 3D prediction and control tasks across held-out gravity values. Open loop(a)–(b): Approach Ball position $L _ { 2 }$ error on held-out episodes with a trained probe. Closed loop(c)–(f): Evaluation results for Difusion Policies trained on frozen world-model features, aggregated by task and resolved by gravity (higher is better). Results are averaged over five seeds. Arm Catcher Ball and Franka Paddle-to-Basket use A = 16, E = 8; Arm Paddle Ball uses A = 16, E = 4. Yellow shading marks the training-gravity distribution, $\mathcal { N } ( 4 , 0 . 5 ^ { 2 } )$ for Arm Catcher Ball and $\mathcal { N } ( 9 . 8 , 2 . 0 ^ { 2 } )$ for the other two.

LeWM remains near the DINO-WM baseline. The advantage is most evident over longer rollouts (Appendix Table 8). An SG-JEPA variant leads every square metric from 5 through 44 rollout-steps. On triangles, although DINO-WM wins by a small margin for short-horizon position and velocity, SG-JEPA achieves the lowest 44-rollout-step errors on both. This behavior is consistent with rollout training reducing the accumulation of errors when predictions are fed back into the model. The triangle dataset remains a challenging case: DINO-WM has slightly lower cumulative-rotation error, which we attribute mostly to asymmetric contacts amplifying small errors after collisions. As for OOD generalization, SG-JEPA has the lowest error across most of the held-out gravity grid on the square dataset, with significant advantages in velocity and rotation for large gravity parameter values (Appendix Fig. 18). Beyond gravity, we also test generalization to an unseen shape. We train SG-JEPA on triangles and squares and evaluate frozen-model rollouts on a “house” dataset, formed by attaching the right triangle to the square along the triangle’s longest side, using a probe trained on house states. The frozen model still yields a useful representation of the house state, and its 44-rollout-step position and velocity predictions transfer reasonably well, particularly with a larger ViT encoder. A detailed breakdown is given in Appendix E.2.

## 3D datasets.

Prediction. On the Approach Ball dataset, we find that the GRU and SSM SG-JEPA variants reduce mean position error by approximately 34% relative to DINO-WM and 50% relative to Original LeWM (Appendix Table 10). From Fig. 4(a), DINO-WM performs comparably well at short horizons but drifts more quickly during rollout. Both SG-JEPA variants remain more accurate over longer rollouts. The GRU and SSM variants perform similarly and alternate as the best method across horizons, suggesting that the improvement does not depend on a single temporal architecture. From Fig. 4(b) we see that this pattern holds across gravity values. SG-JEPA is best at 22 of the 25 test values, with the exceptions being for very small gravity values, where the Original LeWM performs best. In Appendix F.1 we present more results and observe similarly that SG-JEPA retains its advantage for velocity generalization.

Control. For control tasks we evaluate each frozen encoder by training a task-specific Difusion Policy on its latents. The policy runs in a receding-horizon loop using real observations. It uses the latest H = 20 history frames to generate A actions, executes the first E, and then observes the scene again before generating the next action sequence. We use A = 16, E = 8 for Arm Catcher Ball and Franka Paddle-to-Basket, and $A = 1 6 , E = 4$ for Arm Paddle Ball. Control is thus open-loop within each action sequence but closed-loop over the episode. We use task-specific physical events to define success criteria. Arm Catcher Ball requires the ball to enter the central area of the catcher and be latched before the episode ends. Franka Paddle-to-Basket requires a valid paddle hit on the ball, followed by the ball landing in the basket. Arm Paddle Ball requires the robot arm to manipulate the paddle so that the ball never hits the floor throughout the rollout. Moreover, we require it to complete at least one paddle bounce whose apex reaches 0.28 m. This is to avoid the degenerate case where the ball stops bouncing and stays on the paddle.

In Fig. 4(c) we show the aggregate results for the three tasks and (d)–(f) show their per-gravity breakdown. Each episode is evaluated with 5 policy rollouts, and we plot the mean and standard deviation. To keep the main figure readable, we present only the SG-JEPA (GRU) vs. DINO-WM comparison, as DINO-WM provides a stronger baseline for these tasks. Full results including Original LeWM can be found in the relevant appendices. Below we summarize our main observations.

• The largest and most consistent gain occurs for Arm Catcher Ball, where SG-JEPA (GRU) raises capture success from 9.5% to 23.3%. The gain peaks around the training-set values and extends from g = 0 to $g = 9$ . A success peak is observed at $g = 0$ likely because for this gravitational field the ball is not bouncing so the trajectory is relatively easy to predict. On the other hand, for strong gravitational fields the ball bounces frequently, making it dificult to accurately predict the dynamics.

• The Franka Paddle-to-Basket results are less uniform across gravity values. The advantage of SG-JEPA (GRU) appears after impact. DINO-WM achieves a slightly higher paddle-hit success rate (both achieve above 95% hit rates; Appendix Fig. 22) but SG-JEPA (GRU) converts more contacts into basket entries, raising average success from 27.4% to 30.5%. Thus the main dificulty lies in producing the post-contact position and velocity that send the ball into the basket. At low g the struck ball can rise for too long and miss the basket, whereas at high g a stronger and more precise impulse is needed to reach it.

• For Arm Paddle Ball, SG-JEPA (GRU) raises success from 17.7% to 23.8%. SG-JEPA (GRU) both avoids floor contact more often and converts more retained episodes into qualifying bounces. At $g = 0$ , both policies enable the arm to retain the ball but not to complete a bounce because for this value of g an upward-moving ball never falls back down. For high g, the flight time $2 v _ { z } / g$ and apex height $v _ { z } ^ { 2 } / ( 2 g )$ both shrink, leaving less time to reposition the paddle and making the 0.28 m apex threshold harder to reach.

Overall, gravity exposes diferent bottlenecks in each task: interception timing, post-impact targeting, or repeated contact regulation. Together, they strengthen our claim that SG-JEPA’s representations better preserve the dynamics and generalize better OOD than existing baselines. Complete 3D dataset results are in Appendices F.2, F.3, and F.4.

Takeaway. We propose Semigroup-JEPA (SG-JEPA), a gravity-conditioned latent world model that jointly trains the encoder and predictor over recursive latent rollouts, demonstrating performance gains over Original LeWM and DINO-WM. Relative to DINO-WM, for example, it reduces position prediction error by 34% on Approach Ball and improves closed-loop capture success from 9.5% to 23.3% on Arm Catcher Ball across a range of in-distribution and OOD gravity values.

## 4 Where Does the Long-Horizon Advantage Come From?

SG-JEPA’s long-horizon OOD advantage can arise at two stages. A diference may already be present in the one-step prediction (which we refer to as the local transition in the theoretical model below) at an unseen gravity value, before any prediction is fed back, and recursive rollout can then amplify or shrink that diference. In this section we use a linear feature model to separate these stages, which tells us what to measure to locate the advantage.

Theoretical model. Let $\phi _ { t } ~ \in ~ \mathbb { R } ^ { d _ { \phi } }$ be a feature state with dynamics $\phi _ { t + 1 } = T ( g ) \phi _ { t } + \xi _ { t + 1 }$ where $T ( g )$ is the one-step transition at gravitational parameter g and $\mathbb { E } [ \xi _ { t + 1 } \ | \ \phi _ { t } , g ] = 0$ . Let $z _ { t } = W \phi _ { t } \in \mathbb { R } ^ { d _ { z } }$ be the learned latent. We assume $d _ { z } \leq d _ { \phi }$ and $W \in \mathbb { R } ^ { d _ { z } \times \bar { d } _ { \phi } }$ is row-orthonormal, with $W W ^ { \top } = I _ { d _ { z } }$ , and define $\begin{array} { r } { A _ { W } ( g ) = W T ( g ) W ^ { \top } } \end{array}$ and $C _ { W } ( g ) = W T ( g ) - A _ { W } ( g ) W$ . When a teacher-forced predictor $\widehat { A } ( g )$ starts from the true latent, its conditional-mean defect is

$$
\delta _ { t } ( g ) = C _ { W } ( g ) \phi _ { t } + [ A _ { W } ( g ) - \widehat { A } ( g ) ] z _ { t } .\tag{6}
$$

The closure term $C _ { W } ( { \boldsymbol { g } } ) \phi _ { t }$ is the part of the next latent that still depends on features discarded by W. If $C _ { W } ( g ) = 0$ , the representation is predictively closed at gravity $g \colon$ the current latent contains all information needed for the conditional mean of the next latent. The realized teacher-forced error is $z _ { t + 1 } - \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } = \delta _ { t } ( g ) + W \xi _ { t + 1 }$ , where $\widehat { z } _ { t + 1 } ^ { \mathrm { T F } } = \widehat { A } ( g ) z _ { t }$ . Composition at a fixed gravity and transfer across gravity values are separate questions. Suppose $\begin{array} { r } { T ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) T _ { k } } \end{array}$ and the projected and learned operators share this finite law basis. Let $\psi ( g ) = ( \psi _ { 1 } ( g ) , \ldots , \psi _ { m } ( g ) ) ^ { \top } , M _ { \psi } = \mathbb { E } _ { g \sim P _ { \mathrm { t r } } } [ \psi ( g ) \psi ( g ) ^ { \top } ] \succ 0$ and $\mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) = \psi ( g _ { \star } ) ^ { \top } M _ { \psi } ^ { - 1 } \psi ( g _ { \star } )$ , which measures how well the test gravity value $g _ { \star }$ is covered by the training distribution. Appendix H.3 proves that

$$
\begin{array} { r } { \| \delta _ { t } ( g _ { \star } ) \| _ { 2 } \leq \sqrt { { \mathcal L } _ { \mathrm { l a w } } ( g _ { \star } ) } \left( B _ { z } \epsilon _ { \mathrm { o p } } + B _ { \phi } \epsilon _ { \mathrm { c l } } \right) , } \end{array}\tag{7}
$$

where $B _ { z }$ and $B _ { \phi }$ bound the latent and feature states, while $\boldsymbol { \epsilon } _ { \mathrm { o p } }$ and $\epsilon _ { \mathrm { c l } }$ are the training-average predictor and closure errors. For free flight, gravity enters the transition afinely, so $\psi ( g ) = ( 1 , g ) ^ { \intercal }$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) = 1 + \frac { ( g _ { \star } - \mu _ { \mathrm { t r } } ) ^ { 2 } } { \sigma _ { \mathrm { t r } } ^ { 2 } } } \end{array}$ . The bound therefore grows with the distance of $g _ { \star }$ from the training gravity mean. Its main implication is that gravity conditioning alone does not guarantee transfer: unseen-gravity accuracy also depends on law coverage, predictor fit, and representation closure.

From local to long-horizon error. At fixed $g , S _ { g } ( \mathfrak { h } ) = T ( g ) ^ { \mathfrak { h } }$ and $\widehat { S } _ { g } ( \mathfrak { h } ) = \widehat { A } ( g ) ^ { \mathfrak { h } }$ form the true and learned discrete evolution semigroups. Starting from the same latent, the free-rollout error $e _ { \mathfrak { h } } = z _ { \mathfrak { h } } - { \widehat { z } } _ { \mathfrak { h } }$ satisfies

$$
e _ { \mathfrak { h } } = \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \widehat { A } ( g ) ^ { \mathfrak { h } - 1 - j } \left[ \delta _ { j } ( g ) + W \xi _ { j + 1 } \right] .\tag{8}
$$

An error introduced at step $j$ is therefore propagated through the ${ \mathfrak { h } } - 1 - j$ learned updates that follow it. Importantly, one-step errors of the same size can have very diferent consequences after rollout, depending on how the learned updates transform them. A one-step objective is blind to this diference, whereas a multi-step objective penalizes it directly. Appendix H.2 formalizes this through a semigroup-intertwining defect theorem, and Appendix H.7 treats dynamical regime changes at events such as collisions.

(a) Frozen encoder-predictor crossover  
![](images/67c47f314278aec884115ea16b078af0d92b7a0e1e3c3fea78ba9021844d2f68.jpg)

(b) Teacher-forced local error by gravity  
![](images/1de3af2daae3debb294aa852efcc7fcae79841a1bed883f549d0c3c28e479b7b.jpg)

(c) Local versus free-rollout source gap  
![](images/1e4bb719ff6a484caf2bc1f3d99bb98f214fdbbc0bf904334363bffb3da301b4.jpg)  
Figure 5: (a) Frozen-source predictor crossover on held-out 2D square episodes. Color denotes source; line style denotes fresh predictor. (b) Teacher-forced physical MSE from true H = 20 context, with three predictor seeds and pointwise 95% bootstrap intervals. (c) DINO-minus-Dyn teacher-forced and free-rollout gaps over 2,800 paired OOD episodes, with 95% simultaneous intervals. Positive values favor SG-JEPA.

The theoretical model suggests two empirical tests. First, if SG-JEPA’s advantage lies in the encoder’s representation, it should survive replacing the original predictor with a freshly trained one, and it should already be visible under teacher forcing, before any predicted latent is fed back. Second, if feeding predictions back matters, the performance gap between encoders should difer between teacher forcing and free rollout, although the sign of that diference and how it varies with horizon are empirical questions.

Empirical observations. Fig. 5 applies these two tests to the 2D square dataset. Panel (a) freezes the encoders learned jointly with the GRU and Transformer predictors, then fits a new GRU and a new Transformer predictor to each encoder. Panel (b) freezes the SG-JEPA (GRU) and DINO-WM encoders, fits a fresh GRU predictor of matched size to each, and measures teacher-forced mean-squared error (MSE) on collision-free transitions. Panel (c) compares this teacher-forced gap with the gap obtained when predictions are fed back recursively. The main takeaways are as follows.

• The advantage lies in the GRU-trained representation. Panel 5(a) discards the original predictors and fits the same fresh GRU and Transformer architectures to each frozen encoder. With either predictor, the GRU-trained encoder lowers mean rollout error by about 12% relative to the Transformer-trained encoder: 1.376 versus 1.555 with a fresh GRU, and 1.269 versus 1.453 with a fresh Transformer. That the ranking of the two encoders does not depend on which fresh predictor is fitted shows that the gain follows the representation learned during GRU training, rather than the original GRU predictor.

• SG-JEPA is better before recursive feedback. Panel 5(b) tests each local transition before predicted latents are reused. For each frozen encoder, we fit a fresh GRU and restore the true H = 20 context before every collision-free prediction. SG-JEPA has the lower local-error point estimate at all 25 test gravity values. On the far-OOD gravity values $g \leq 2 \ \mathrm { o r } \ g \geq 6 .$ , it reduces local error by about 32% relative to DINO-WM. Applying the same physical-state probe to the true and predicted next latents removes the probe error common to both models. The advantage therefore arises before rollout error accumulates.

• Recursive rollout widens the advantage. In Panel 5(c), positive values favor SG-JEPA.

From horizon 2 onward, the free-rollout gap is several times larger than the teacher-forced gap, peaking near 0.38 around horizon 20, while the local gap remains below 0.06. This is consistent with Eq. 8, which says that once predictions are fed back, a small one-step diference between the encoders grows into a much larger gap. That gap shrinks again at long horizons, so the amplification is not monotone.

Ablations. Ablations are presented in the appendices. Appendix D.4 tests whether the predictor actually uses gravity through an evaluation-time counterfactual, in which we hold the physical trajectories fixed and feed the predictor a gravity value that difers from the true one. For both GRU and Transformer predictors, the correct gravity minimizes rollout error, while larger mismatches generally increase it. Appendix G studies adaptation from sparse post-training data. We post-train on disjoint support episodes at $\mathcal { G } _ { \mathrm { s u p } } = \{ 0 , 2 , 6 , 8 \}$ and evaluate on 13 unseen interpolation gravity values. We find that post-training reduces SG-JEPA (GRU) error by 13.6% on the square and 20.8% on the triangle. Across all eight model-shape settings, it improves on the pretrained model and outperforms post-training on the target gravity alone, averaging a 14.1% reduction versus 6.9%.

Takeaway. SG-JEPA’s OOD advantage lies in the representation the encoder learns during GRU training. We know this because the advantage survives replacing the predictor and is already present before any feedback, where it lowers far-OOD one-step error by 32%. The linear feature model explains why such a small one-step edge matters. Gravity conditioning alone does not guarantee transfer, and once predictions are fed back, a small one-step diference between encoders grows into a much larger long-horizon gap, which is what we see under recursive rollout.

## 5 Conclusion

In this work, we introduce Semigroup-JEPA (SG-JEPA), which combines gravity conditioning with joint encoder–predictor training through recursive latent rollouts. To test whether a world model learns physical dynamics rather than a single transition law, we build eight MuJoCo datasets in which gravity is sampled from a narrow band at training time and from a much wider grid at test time. On 2D and 3D prediction and robot control, SG-JEPA outperforms Original LeWM and DINO-WM both at longer rollout horizons and at OOD gravity values, with the largest gains at long horizons and far from the training gravity. The same frozen representation transfers to closed-loop control, where a Difusion Policy trained on SG-JEPA features raises success rates on all three manipulation tasks. To explain these gains, we develop a linear feature model that separates local law-conditioned error from its amplification under rollout and bounds the one-step error at an unseen gravity value by how well the training gravities cover it. Guided by this model, we train fresh predictors on frozen checkpoints and find that SG-JEPA’s advantage lies in the encoder’s representation, and that recursive rollout amplifies this local advantage over the horizon. Ablations further show that the predictor actually uses the supplied gravity, and that sparse post-training at a few new gravity values improves accuracy at the unseen values between them.

Limitations. Several important directions remain for future work. First, our experiments vary a single scalar, gravity. It would be of interest to study a setting with vector-valued physical variables, or even to have the model infer dynamical parameters directly from observation. Second, transfer across object shapes is uneven. Training on 2D triangle and square datasets transfers much of the house’s translational dynamics but not its rotation, and more complex shapes such as the pentagon remain challenging for the current setup, which may require a generalist model trained on diverse data. Finally, our theory uses a linear feature model, whereas the neural predictor is nonlinear and history dependent, and contacts can perturb transition branches. Understanding the behavior of JEPA-type models calls for new theoretical models that incorporate nonlinearities as well as action-perturbing efects.

## References

Yann LeCun. A path towards autonomous machine intelligence. OpenReview, 2022. Version 0.9.2.

Uladzislau Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free ofline data: A case for planning with latent dynamics models. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-1465.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 79115–79135. PMLR, 2025.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorld-Model: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026.

Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. doi: 10.1109/CVPR52729.2023.01499.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024.

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

Haotian Xue, Yipu Chen, Liqian Ma, Zelin Zhao, Lama Moukheiber, Yuchen Zhu, and Yongxin Chen. ACWM-Phys: Investigating generalized physical interaction in action-conditioned video world models. arXiv preprint arXiv:2605.08567, 2026.

Kaizhen Tan, Xin Xu, Siru Tao, Hanzhe Hong, Yang Feng, and Heqing Du. What can latent world models know? physical parameter identifiability in multimodal predictive representations. arXiv preprint arXiv:2607.27017, 2026.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems, volume 28, 2015.

Arun Venkatraman, Martial Hebert, and J. Andrew Bagnell. Improving multi-step prediction of learned time series models. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pages 3024–3030, 2015. doi: 10.1609/aaai.v29i1.9590.

Nathan O. Lambert, Albert Wilcox, Howard Zhang, Kristofer S. J. Pister, and Roberto Calandra. Learning accurate long-term dynamics for model-based reinforcement learning. In 2021 60th IEEE Conference on Decision and Control, pages 2880–2887, 2021. doi: 10.1109/CDC45484.2021.9683134.

Matthew O. Williams, Ioannis G. Kevrekidis, and Clarence W. Rowley. A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition. Journal of Nonlinear Science, 25(6):1307–1346, 2015. doi: 10.1007/s00332-015-9258-5.

Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Deep learning for universal linear embeddings of nonlinear dynamics. Nature Communications, 9:4950, 2018. doi: 10.1038/s41467-018-07210-0.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. doi: 10.1109/ICCV51070.2023.00387.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. Online, 2024. Blog post.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033, 2012. doi: 10.1109/IROS.2012.6386109.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Difusion policy: Visuomotor policy learning via action difusion. In Proceedings of Robotics: Science and Systems, 2023.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems, volume 31, 2018.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2555–2565. PMLR, 2019.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, 2025. doi: 10.1038/s41586-025-08744-2.

Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, 2024.

Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. IntPhys 2019: A benchmark for visual intuitive physics understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5016–5025, 2022. doi: 10.1109/TPAMI.2021.3083839.

Daniel Bear, Elias Wang, Damian Mrowca, Felix Binder, Hsiao-Yu Tung, R. T. Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, Fei-Fei Li, Nancy Kanwisher, Joshua Tenenbaum, Daniel Yamins, and Judith Fan. Physion: Evaluating physical prediction from vision in humans and machines. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 948–958, 2026. doi: 10.1109/WACV61042.2026.00099.

Shuai Wang, Yaxin Feng, Xuekun Jiang, Shihan Tian, Ningyu Yan, Xing Shen, Chaoyang Lyu, Hui Wang, Yunsong Zhou, Hanqing Wang, Jiangmiao Pang, Yang Xiang, Xing Gao, Chunhua Shen, and Weinan Zhang. GAUGE: A measurement-grounded benchmark for physical fidelity in simulation engines and video world models. arXiv preprint arXiv:2608.05948, 2026.

Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun. Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831, 2025.

Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaf, Rex Ying, Jure Leskovec, and Peter W. Battaglia. Learning to simulate complex physics with graph networks. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 8459–8468. PMLR, 2020.

Yunzhu Li, Toru Lin, Kexin Yi, Daniel Bear, Daniel Yamins, Jiajun Wu, Joshua Tenenbaum, and Antonio Torralba. Visual grounding of learned physical models. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5927–5936. PMLR, 2020.

Taylor W. Killian, Samuel Daulton, George Konidaris, and Finale Doshi-Velez. Robust and eficient transfer learning with hidden parameter markov decision processes. In Advances in Neural Information Processing Systems, volume 30, pages 6250–6261, 2017.

Kimin Lee, Younggyo Seo, Seunghyun Lee, Honglak Lee, and Jinwoo Shin. Context-aware dynamics model for generalization in model-based reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5757–5766. PMLR, 2020.

Matthieu Kirchmeyer, Yuan Yin, Jeremie Dona, Nicolas Baskiotis, Alain Rakotomamonjy, and Patrick Gallinari. Generalizing to new physical systems via context-informed dynamics model. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 11283–11301. PMLR, 2022.

Junjie Wang, Qichao Zhang, Yao Mu, Dong Li, Dongbin Zhao, Yuzheng Zhuang, Ping Luo, Bin Wang, and Jianye Hao. Prototypical context-aware dynamics for generalization in visual control with model-based reinforcement learning. IEEE Transactions on Industrial Informatics, 20(9): 10717–10727, 2024. doi: 10.1109/TII.2024.3396525.

Wenhao Yu, Jie Tan, C. Karen Liu, and Greg Turk. Preparing for the unknown: Learning a universal policy with online system identification. In Proceedings of Robotics: Science and Systems, 2017. doi: 10.15607/RSS.2017.XIII.048.

Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. RMA: Rapid motor adaptation for legged robots. In Proceedings of Robotics: Science and Systems, 2021. doi: 10.15607/RSS.2021. XVII.011.

Philip J. Ball, Cong Lu, Jack Parker-Holder, and Stephen Roberts. Augmented world models facilitate zero-shot dynamics generalization from a single ofline environment. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 619–629. PMLR, 2021.

JB Lanier, Kyungmin Kim, Armin Karamzade, Yifei Liu, Ankita Sinha, Kat He, Davide Corsi, and Roy Fox. Adapting world models with latent-state dynamics residuals. In Proceedings of the 8th Annual Learning for Dynamics and Control Conference, volume 331 of Proceedings of Machine Learning Research, pages 117–144. PMLR, 2026.

Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 23–30, 2017. doi: 10.1109/IROS.2017.8202133.

Erik Talvitie. Model regularization for stable sample rollouts. In Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence, pages 780–789, 2014.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, pages 1724–1734. Association for Computational Linguistics, 2014. doi: 10.3115/v1/D14-1179.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Nathan Lambert, Brandon Amos, Omry Yadan, and Roberto Calandra. Objective mismatch in model-based reinforcement learning. In Proceedings of the 2nd Conference on Learning for Dynamics and Control, volume 120 of Proceedings of Machine Learning Research, pages 761–770. PMLR, 2020.

Christopher Grimm, André Barreto, Satinder Singh, and David Silver. The value equivalence principle for model-based reinforcement learning. In Advances in Neural Information Processing Systems, volume 33, 2020.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588:604–609, 2020. doi: 10.1038/s41586-020-03051-4.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3M: A universal visual representation for robot manipulation. In Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 892–909, 2022.

Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, and Franziska Meier. Where are we in the search for an artificial visual cortex for embodied intelligence? In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/075280-0031.

W. P. M. H. Heemels, Bart De Schutter, and Alberto Bemporad. Equivalence of hybrid dynamical models. Automatica, 37(7):1085–1091, 2001. doi: 10.1016/S0005-1098(01)00059-0.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025.

## Contents

A Additional related work 17   
B Datasets 18   
C Model and training details 20   
C.1 World Model 20   
C.2 Downstream model 22   
D Ablation studies 23   
D.1 Optimizer and hyperparameter choices 23   
D.2 History length and rollout horizon ablations 27   
D.3 Rollout discount 29   
D.4 Counterfactual gravity conditioning 29   
D.5 Where does GRU advantage come from . 29   
D.6 Performance on conventional control tasks 33   
E 2D datasets full results 34   
E.1 Gravity-conditioned rollouts 34   
E.2 Shape generalization 35   
F 3D datasets full results 37   
F.1 Approach-Ball results . 37   
F.2 Arm-Catcher-Ball control results 38   
F.3 Franka paddle-to-basket results 39   
F.4 Arm-Paddle-Ball results 40   
G Sparse-Gravity Post-Training Ablation 41   
H Theory of prediction under changing physical laws 46   
H.1 Setup: the model 46   
H.2 Local prediction error under a fixed gravity 47   
H.3 How gravity coverage controls test error 49   
H.4 Local errors under repeated composition 53   
H.5 Why multi-step composition can change the learned representation 56   
H.6 Relating the theory to the history-based predictor . 59   
H.7 Regime changes and contact . 62   
H.8 Supplementary calculation: SIGReg and its limits . 64

## A Additional related work

This appendix expands Section 1.

Latent world models and physical prediction. Joint-embedding predictive architectures learn to predict features rather than pixels [LeCun, 2022, Assran et al., 2023, Bardes et al., 2024, Assran et al., 2025]. DINO-WM fits dynamics over a frozen DINOv2 encoder [Zhou et al., 2025, Oquab et al., 2024], while LeWM trains its encoder and predictor jointly with the SIGReg regularizer from LeJEPA [Balestriero and LeCun, 2025, Maes et al., 2026]. Such latent models can also plan from reward-free ofline data [Sobal et al., 2025]; reconstruction-based world models retain a pixel or observation decoder [Ha and Schmidhuber, 2018, Hafner et al., 2019, 2025, Hansen et al., 2024]. Physical benchmarks provide a complementary test. IntPhys and Physion evaluate physical events or human judgments [Riochet et al., 2022, Bear et al., 2021], whereas Physics-IQ and GAUGE measure trajectories and physical quantities [Motamed et al., 2026, Wang et al., 2026]. Other work studies intuitive physics in self-supervised video representations [Garrido et al., 2025], structured particle simulators [Sanchez-Gonzalez et al., 2020], visual inference of physical properties [Li et al., 2020], and prediction under held-out physical configurations [Xue et al., 2026, Tan et al., 2026].

Generalization across physical systems. Hidden-parameter MDPs represent related environments with a low-dimensional dynamics variable [Killian et al., 2017]. CaDM and CoDA infer this context from trajectories [Lee et al., 2020, Kirchmeyer et al., 2022], and ProtoCAD extends context-conditioned dynamics to high-dimensional visual observations [Wang et al., 2024]. Universal policies, rapid motor adaptation, and augmented world models also infer or adapt to changing dynamics [Yu et al., 2017, Kumar et al., 2021, Ball et al., 2021]; residual latent dynamics provide a related approach to sim-to-real adaptation [Lanier et al., 2026]. Domain randomization instead trains across parameter variations without exposing the parameter to the policy [Tobin et al., 2017]. SG-JEPA removes system identification from the experiment: the scene and control interface remain fixed, gravity is given to the model, and only that scalar varies. The evaluation therefore isolates interpolation and extrapolation along a known physical coordinate.

Rollout training and downstream use. A one-step model accumulates error when its predictions become future inputs [Bengio et al., 2015]. Existing remedies feed predictions back during training [Talvitie, 2014], correct rollout states with training data [Venkatraman et al., 2015], supervise several latent horizons [Hafner et al., 2019], or optimize long-horizon accuracy directly [Lambert et al., 2021]. SG-JEPA backpropagates its rollout loss through both predictor and encoder, and the frozen-encoder crossover separates their contributions across recurrent, state-space, and attention predictors [Cho et al., 2014, Vaswani et al., 2017, Gu and Dao, 2024]. Because model loss need not track control performance [Lambert et al., 2020], related work has used value-equivalent objectives [Grimm et al., 2020, Schrittwieser et al., 2020] and frozen representations for imitation learning [Nair et al., 2022, Majumdar et al., 2023, Chi et al., 2023]. We likewise report physical prediction and downstream policy performance, while keeping the data and policy training fixed within each controlled comparison. The accompanying analysis uses lifted linear dynamics and a piecewise-afine contact model [Williams et al., 2015, Lusch et al., 2018, Heemels et al., 2001]; these are surrogate analyses, not claims that the learned predictors are linear.

## B Datasets

We evaluate eight datasets generated with MuJoCo [Todorov et al., 2012]: four planar rigid-body prediction tasks, one 3D projectile prediction task, and three 3D robotic control tasks. Table 1 summarizes their observation, action, and state records.

Planar rigid-body prediction. The four 2D datasets contain one red rigid body, a right triangle, square, pentagon, or "house" (concatenation of the right triangle and the square along the leg), moving and colliding inside a planar box. We randomize its initial pose and apply a single planar impulse. The eight-dimensional state is $[ x , z , v _ { x } , v _ { z } , \vartheta , \omega , x _ { \mathrm { a n c h o r } } , z _ { \mathrm { a n c h o r } } ]$ . The physics record also stores the object mass and size, shape identifier, and box dimensions.

3D prediction and control. Approach Ball isolates projectile prediction: a ball is launched with randomized initial velocity and travels freely without actions. Arm Catcher Ball and Arm Paddle Ball use a Unitree Z1 arm. The catcher task moves a catcher to intercept the bouncing ball, while the paddle task controls position and tilt to sustain repeated bounces. Their per-frame records include the commanded and realized catcher or paddle state, contact flags, arm joint positions and velocities, and task success. Franka Paddle-to-Basket instead uses a Franka arm with a blue paddle attached to strike an incoming ball toward a basket. Success requires a valid blade contact followed by the ball entering the basket. In addition to the ball, paddle, contact, physics, and arm records, this dataset stores basket geometry, task events, a strike diagnostic, and a failure label. The scripted outcome cycle contains 70% successes and 5% from each of six informative failure modes.

Table 1: Dataset summary. Each episode consists of 64 frames at 16 Hz. State records contain the task-relevant object and robot states used for supervision and evaluation.
<table><tr><td>Dataset</td><td>Role</td><td>RGB resolution</td><td>Action record</td><td>Principal state record</td></tr><tr><td>Right Triangle</td><td>Prediction</td><td>1282</td><td> $( J _ { x } , J _ { z } , g )$ </td><td>Rigid body (8)</td></tr><tr><td>Square</td><td>Prediction</td><td>1282</td><td> $( J _ { x } , J _ { z } , g )$ </td><td>Rigid body (8)</td></tr><tr><td>Pentagon</td><td>Prediction</td><td>1282</td><td> $( J _ { x } , J _ { z } , g )$ </td><td>Rigid body (8)</td></tr><tr><td>House</td><td>Prediction</td><td>1282</td><td> $( J _ { x } , J _ { z } , g )$ </td><td>Rigid body (8)</td></tr><tr><td>Approach Ball</td><td>Prediction</td><td>2562</td><td>g</td><td>Ball (16)</td></tr><tr><td>Arm Catcher Ball</td><td>Control</td><td>2562</td><td> $( g , \Delta x , \Delta y , \Delta z )$ </td><td>Ball (16), catcher (9), Z1 arm (18)</td></tr><tr><td>Arm Paddle Ball</td><td>Control</td><td> $2 5 6 ^ { 2 }$ </td><td> $( g , \Delta x , \Delta y , \Delta z , \phi , \theta )$ </td><td>Ball (16), paddle (9), Z1 arm (18)</td></tr><tr><td>Franka Paddle-to-Basket</td><td>Control</td><td>2562</td><td> $( g , \Delta x , \Delta y , \Delta z , \phi , \theta )$ </td><td>Ball (16), paddle (9), Panda arm (21), basket (6)</td></tr></table>

Gravity sampling and input encoding. The gravity g is constant within each episode. For the 2D planar tasks, training gravity follows $g = \operatorname* { m a x } ( \mathcal { N } ( 4 , 0 . 5 ^ { 2 } ) , 0 . 1 )$ with 8000 episodes each, and the held-out test split uses the 25-point grid $\{ - 2 , - 1 . 5 , \ldots , 1 0 \}$ with 200 episodes per value. For 3D datasets, Approach Ball, Arm Paddle Ball and Franka Paddle-to-Basket again samples 8000 episodes from $g = \operatorname* { m a x } ( N ( 9 . 8 , 2 ^ { 2 } ) , 0 )$ for training; and 25-value held-out test set $\{ 0 , 1 , \ldots , 2 0 , 0 . 6 2 , 1 . 6 3 , 3 . 7 2 , 8 . 8 7 \}$ , with the last 4 values being the gravity values in Pluto, Moon, Mars and Venus respectively. Arm Catcher Ball uses $g = \operatorname* { m a x } ( \mathcal { N } ( 4 , 0 . 5 ^ { 2 } ) , 0 . 1 )$ and the 23-value grid $\{ - 1 , - 0 . 5 , \hdots , 1 0 \}$

Gravity is stored in physical units in the physics record and as an action coordinate. The planar action is $( J _ { x } , J _ { z } , g )$ , where the two impulse channels are nonzero only at t = 0. Approach Ball receives only g. The arm tasks append Cartesian catcher or paddle commands from a scripted expert, and Arm Paddle Ball and Franka Paddle-to-Basket append two paddle-orientation commands. Before a learned component receives gravity, we z-score it as $( g - \mu _ { g , \mathrm { t r a i n } } ) / \sigma _ { g , \mathrm { t r a i n } }$ . We compute these statistics only on the corresponding training split and keep them fixed for validation, test, and post-training evaluation. Evaluation data never enter the normalization statistics.

Format and storage. Each episode contains 64 frames at 16 Hz over 4 seconds. The 2D videos are $1 2 8 \times 1 2 8$ , and the 3D videos are $2 5 6 \times 2 5 6$ . Action $a _ { t }$ is aligned with the transition from frame t to frame t + 1. The Lance datasets contain one row per frame, including episode and step indices, split identifier, JPEG-encoded RGB, simulator state, aligned action, reward, a compact physics vector, raw gravity, and source-episode index. Controlled tasks add JSON episode metadata and the task-specific records above. Headline 2D plots use fixed named subsets of the larger test split, while some 3D probe diagnostics use separate evaluation manifests; each result states its cohort size.

Figure 6 gives representative trajectories from all eight datasets, grouped by their prediction or control role. Figure 7 compares ground-truth frames with decoded open-loop predictions for one 2D task and two 3D tasks under both in-distribution and out-of-distribution gravity. For the decoded rollouts, we attach a separate online CNN decoder to each world model. The decoder maps the 256-dimensional latent to RGB from an $8 \times 8$ feature map, with 256 base channels, a 32-channel minimum, and one residual block. We train it on detached features with RGB MSE, using Muon at $5 \times 1 0 ^ { - 4 }$ for eligible matrix parameters and AdamW at $1 0 ^ { - 4 }$ for the remaining parameters. The reconstruction loss therefore does not update the world model. The decoder is used only for visualization; quantitative prediction results use frozen physical-state probes.

## C Model and training details

## C.1 World Model

Section 2 gives the model architecture and training objective; in this section, we elaborate on the implementation details needed to reproduce the reported runs. The ViT-Tiny CLS token is projected from 192 to 256 dimensions. The GRU and SSM predictors both use three residual layers of width 512, an MLP width of 2048, dropout 0.1, and action concatenation at every layer. As with the other action channels, gravity g is z-scored before entering the action encoder.

During rollout training, each predicted latent is fed back into the predictor. For the next step, the latent and action histories shift forward together, retaining only their most recent H entries. Every final SG-JEPA model uses $H = 2 0$ , a five-step training rollout $K = 5 ,$ , discount $\gamma = 0 . 9 5$ , and trainable target latents without stop-gradient since SIGReg already regulates the latent space and avoids collapse. SIGReg is evaluated on encoded latents only, using 1024 random projections and 17 knots. The coeficient is selected separately for each task family, as summarized in Table 2. All models are trained for 20 epochs with batchsize 64. Ablation studies for these hyperparameters are presented in the next sections.

Table 2: SG-JEPA configurations used in the main textheadline experiments. All reported models use the fixed epoch-20 checkpoint.
<table><tr><td>Task family</td><td>Image / patch</td><td>Predictor</td><td> $\lambda _ { \mathrm { s i g } }$ </td></tr><tr><td>Planar rigid bodies</td><td> $1 2 8 ^ { 2 } / 8$ </td><td>GRU, SSM</td><td>0.72</td></tr><tr><td>Approach Ball</td><td> $2 5 6 ^ { 2 } / 1 6$ </td><td>GRU, SSM</td><td>0.18</td></tr><tr><td>Arm Catcher Ball</td><td> $2 5 6 ^ { 2 } / 1 6$ </td><td>GRU</td><td>0.18</td></tr><tr><td>Arm Paddle Ball</td><td> $2 5 6 ^ { 2 } / 1 6$ </td><td>GRU</td><td>0.72</td></tr><tr><td>Franka Paddle-to-Basket</td><td> $2 5 6 ^ { 2 } / 1 6$ </td><td>GRU</td><td>0.09</td></tr></table>

All rows use the same hybrid optimizer. Every trainable two-dimensional parameter tensor is optimized by Muon at $1 0 ^ { - 4 } ;$ the remaining parameters use AdamW at $5 \times 1 0 ^ { - 5 }$ with weight decay $1 0 ^ { - 3 }$ . Muon uses Nesterov momentum 0.95, five Newton–Schulz steps, and the adjustment with scale 0.4 Liu et al. [2025].

![](images/f7fdbda0ea5a6e03399209666dc441c9482e2997d4cd1d4c929ba7f1c7fd7f69.jpg)  
Figure 6: Representative evolution of one episode from each dataset. We show frames t ∈ {0, 13, 25, 38, 50, 63}. Rows above the divider are passive prediction environments; rows below it require arm control and are used for control.

![](images/5552060d56819e857ded199792d85e23610e283d94db8871f25b20bc5034a4c7.jpg)  
Figure 7: Decoded open-loop rollouts under in-distribution (ID) and out-of-distribution (OOD) gravity. For each dataset, the upper row shows ground-truth and the lower row decodes the autoregressively predicted latents. The model receives 20 context frames and then rolls out through $t = 6 3$

Baseline configurations. Original LeWM retains the same ViT-Tiny encoder, 256-dimensional projector, action preprocessing, and H = 20 context, but uses a causal Transformer with six layers, width 256, 16 attention heads, MLP width 2048, and dropout 0.1. It is trained with the teacher-forced one-step latent MSE and SIGReg coeficient 0.09. These runs use the same hybrid Muon–AdamW recipe above and fixed epoch-20 checkpoints.

DINO-WM freezes a pretrained DINOv2 ViT-S/14 encoder and extracts a grid of 64 normalized patch tokens, each 384-dimensional, from $1 1 2 \times 1 1 2$ inputs. Its action-conditioned, frame-causal Transformer uses $H = 2 0$ , six layers, 16 attention heads, MLP width 2048, dropout 0.1, and a 10-dimensional action embedding. The predictor is trained by teacher forcing with one-step MSE to the next frozen patch-token grid. The reported DINO-WM predictors are trained for 20 epochs with AdamW at $5 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 2 }$ , batch size 32.

## C.2 Downstream model

State probes. For each frozen source representation, we fit a separate MLP probe. The probe layer-normalizes the concatenated latent window, followed by two GELU layers of widths 512 and 256 with dropout 0.05 and a linear readout. We z-score each target coordinate using statistics from the probe-training split and optimize the probe with AdamW at $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 5 }$ , and batch size 256 for at most 50 epochs, retaining the checkpoint with the lowest validation NMSE. Probes are trained only on encoded ground-truth sequences and are then frozen before being applied to predicted rollouts.

Difusion policies. We train a separate policy for every frozen encoder and control task. In addition to the latent-history GRU described in Section 2.3, the noise network is a conditional 1D U-Net with channel widths 256, 512, 1024, kernel size 5, eight GroupNorm groups, and a 256- dimensional difusion-step embedding. Each policy is trained for 300k updates with batch size 256 using AdamW at $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 6 }$ . The learning rate warms up linearly for 500 updates and then decays with a cosine schedule. The visual encoder remains frozen throughout. In particular, the DINO-WM policy mean-pools frozen patch tokens, and neither it nor the SG-JEPA policies query a world-model predictor online.

## D Ablation studies

## D.1 Optimizer and hyperparameter choices

## D.1.1 Learning rate sweep

We run learning rate sweep on the 2D $1 2 8 \times 1 2 8$ square dataset using a Transformer predictor. The 8,000 source episodes are partitioned by episode into 7,200 training and 800 validation episodes; panels (d)–(f) use extra held-out test split. We vary the AdamW learning rate over $\lbrace 1 0 ^ { - 5 } , 3 \times$ $1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } \}$ , with weight decay $1 0 ^ { - 3 }$ , batch size 64, $H = 2 0 , K = 5$ $\gamma = 0 . 9 5$ , and $\lambda _ { \mathrm { S I G } } = 0 . 7 2$ . Each run trains for at most 20 epochs. Early stopping monitors the validation objective with patience three, and we retain the checkpoint with the lowest validation objective. We then fit temporal-window-4 linear and MLP probes to each checkpoint.

![](images/173a6c31aac15aec931188557a0dfdff488bdd93797542ff505c9fdb6060f196.jpg)  
Figure 8: AdamW learning-rate sweep on 2D square dataset

Results The lowest validation objective occurs at $5 \times 1 0 ^ { - 4 }$ , but its poor probe readout shows that the objective alone is not a reliable measure of representation quality. The $5 \times 1 0 ^ { - 5 }$ setting nearly matches the best MLP-probe validation NMSE and gives the best linear-probe NMSE; the held-out test results in Figure 8 show the same trend. We therefore use $5 \times 1 0 ^ { - 5 }$ for the AdamW parameters.

## D.1.2 Muon versus AdamW

![](images/43602a5411d55c8178a901bfaebc37cab2d558516d7f70cda7320237020b5282.jpg)  
(d) GRU validation objective

![](images/27bfcd1cf81fe78f7b38898a2dfcce5bcef6bfcef871970a5cd5fbb28a2b29fb.jpg)

![](images/8dcdda81c99d8a160e48f749a22a72fad9778ff68c70f189a37927c022b2f810.jpg)  
(e) SSM validation objective  
(f) Transformer validation objective

![](images/02d75c854282284f6c86879c9a6c5fd1542973eae2b7dd9d2da4825e19e2360d.jpg)

![](images/75f939746493a48cea6c98dc4d12f3edbddbed88920ed46ce3c6137a6ead9021.jpg)

![](images/da13c490a2dcf42c69d3284f22e42f2d29ffd5e28f70a6d039317a33c2c35fab.jpg)  
Figure 9: AdamW versus hybrid Muon/AdamW Panels (a-c) show training objectives and panels (d-f) show validation objectives for the GRU, SSM, and Transformer predictors. Lower is better.

In Fig. 9, we compare AdamW with the hybrid Muon/AdamW on the $1 2 8 \times 1 2 8$ square dataset. For each of the GRU, SSM, and Transformer predictors, both conditions use a 256-dimensional latent, $H = 2 0 , K = 5 , \gamma = 0 . 9 5 , \lambda _ { \mathrm { S I G } } = 0 . 7 2$ , batch size 64. Training is capped at 20 epochs, with early stopping after three epochs without validation improvement. AdamW uses $5 \times 1 0 ^ { - 5 }$ for all parameters. The hybrid recipe assigns every two-dimensional parameter tensor to Muon at $1 0 ^ { - 4 }$ and the remaining parameters to AdamW at $5 \times 1 0 ^ { - 5 }$ . Muon uses Nesterov momentum 0.95, five Newton-Schulz iterations, and per-matrix RMS-matching multiplier $0 . 4 \sqrt { \operatorname* { m a x } ( A , B ) }$

Across all three predictors, Muon lowers the training objective faster and reaches a lower best validation objective: 4.990 versus 5.864 for the GRU, 5.403 versus 6.922 for the SSM, and 5.278 versus 6.384 for the Transformer. These are relative reductions of 14.9%, 21.9%, and 17.3%, respectively. The selected checkpoints also occur earlier, at epochs 11, 6, and 7 instead of 12, 8, and 13.

## D.1.3 SIGReg coeficient sweep

In this case we vary $\lambda _ { \mathrm { S I G } } \in \{ 0 . 0 9 , 0 . 1 8 , 0 . 3 6 , 0 . 7 2 , 1 . 4 4 \}$ on 2D square dataset. Rollout metrics and decoded examples use a 5,000-episode selection cohort spanning 25 gravity values from −2 to 10; efective rank uses a 2,000-episode well-spread subset. We select SIGReg coeficient based on the downstream rollout performance.

We note that raw SIGReg loss measures agreement with a standard Gaussian under random one-dimensional projections, not dimensional collapse. We therefore also report the efective rank of the final-context latent covariance. For $z _ { i } \in \mathbb { R } ^ { 2 5 6 }$ and $N = 2 0 0 0$ ，

![](images/1d8ba1982999e78f0260a1c16c7b9dd2eb8315f9b43c355998888fc3fbbfcf2b.jpg)

(b) Effective rank evolution  
![](images/bf8744849d64f8a0e6fa046efbeb6a0b6d2c194a3e4178ae6ff6931913d8979a.jpg)

![](images/7368d7f5435da1eb44c0667092f3b98c302d5ade422692d668cf3ee9475375a7.jpg)

(d) Predicted, 5  
![](images/cee3b63e57619253c4d1fce4f3a92b3b27b5b74b8d5ac613327a827b5b4d3af6.jpg)

(e) Predicted, 20  
(f) Predicted, h44  
![](images/2cb0bfb44eacce5986edb21438962c5290e1d8262cfb1b9e2be0f389f01b2fcf.jpg)

![](images/3a417d72d64648ed7e1ef47543b9c2dd13001cfff2803bd1c32ab13c778077e5.jpg)  
Figure 10: SIGReg coeficient sweep. (a) Raw training SIGReg loss; (b) Efective rank on the same 2,000 episodes at every checkpoint; (c) Linear-probe validation NMSE; and (d–f) MLP-probe NMSE after 5, 20, and 44 autoregressive steps across $g \in [ - 2 , 1 0 ]$ . Panels (c–f) use the same epoch checkpoints. The yellow shade in the background indicates training distribution. Lower is better except in (b).

$$
{ \bar { z } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } z _ { i } , \qquad C = { \frac { 1 } { N - 1 } } \sum _ { i = 1 } ^ { N } ( z _ { i } - { \bar { z } } ) ( z _ { i } - { \bar { z } } ) ^ { \top } .\tag{9}
$$

Let $\lambda _ { 1 } \geq \cdots \geq \lambda _ { D } \geq 0$ denote the eigenvalues of C. We summarize the spectrum by its efective rank,

$$
p _ { j } = \frac { \lambda _ { j } } { \sum _ { k } \lambda _ { k } } , \qquad r _ { \mathrm { e f f } } ( C ) = \exp \left( - \sum _ { j : p _ { j } > 0 } p _ { j } \log p _ { j } \right) .\tag{10}
$$

Fig. 10 separates optimization behavior from representation quality. Although 0.72 and 1.44 reach similarly low raw SIGReg losses in panel (a), panels (b) and (c) favor 0.72: the efective rank is 138.0, compared with 131.0 for 1.44, and it gives the lowest linear-probe validation NMSE. Panels (d) and (e) show that 0.72 also gives the best short- and medium-horizon readout over most of the gravity range. Early rank remains a useful screening signal, with epoch-5 correlations of $\rho _ { s } = - 0 . 9 0$ and −1.00 against later linear- and MLP-probe NMSE, respectively.

Fig. 11 shows the same pattern. The representations obtained with $\lambda _ { \mathrm { S I G } } = 0 . 0 9 , 0 . 1 8 , 0 . 3 6$ decode the square as a rounded or blurred object and accumulate visible pose and position errors. At $\lambda _ { \mathrm { S I G } } = 0 . 7 2$ , the higher-rank representation most clearly preserves corners, orientation, and trajectory over the rollout. Increasing the coeficient to 1.44 retains relatively sharp geometry but introduces larger long-horizon trajectory errors. Finally we select $\lambda _ { \mathrm { S I G } } = 0 . 7 2$ for our 2D experiments because it gives the best balance of efective rank, frozen-probe accuracy, and rollout quality.

![](images/98754db5253ee54e3c01b2f759d9fd1ef37e4643c7658fa8617923ddc89b1b73.jpg)  
Figure 11: Qualitative comparison of the SIGReg coeficient. Matched rollout predictions for one representative square episode at g = 3. The top row shows ground-truth frames, and the remaining rows show decoded predicted latents for each value of λ<sub>SIG</sub>. Columns show the autoregressive rollout horizons 5, 10, 20, 32 and 44. Each prediction row is rendered using the online decoder trained jointly with that model.

## D.2 History length and rollout horizon ablations

To choose a common temporal configuration without running a separate sweep for every dataset, we perform detailed ablations on one planar task (2D Square) and one 3D task $( \mathrm { A p p r o a c h ~ B a l l } )$ . The 2D study varies history length and predictor depth, whereas the 3D study also varies the SIGReg coeficient (2D we’ve done in the section above). Both tasks favor a 20-frame history, and Approach Ball favors a five-step training rollout. We therefore use $H = 2 0$ and $K = 5$ across all datasets for a consistent experimental setup. The detailed results are as follows.

## D.2.1 2D square: history length and predictor depth

On the 2D square dataset, we cross $H \in \{ 5 , 1 0 , 2 0 , 3 0 \}$ with predictor depth $L \in \{ 2 , 3 , 4 \} \ ( H = 4 0$ causes OOM error on one B200). All runs train for at most 10 epochs. Fig. 12 shows prediction results from trained MLP probes. We see that $H = 2 0$ gives the best long-horizon accuracy across depths. At this history length, $L = 4$ is numerically best, with position, orientation, and signed-turn errors of 0.745, 13.91◦, and 0.153 turns. The $L = 3$ model is within 2% to 4% on all three metrics, while reducing the predictor from 19.30M to 14.57M parameters. We therefore use $H = 2 0 , L = 3$ as a balance between performance and training cost.

The real-latent controls explain why we do not use $H = 3 0 .$ . Their orientation error rises from at most $0 . 6 6 ^ { \circ }$ for $H \leq 2 0$ to about $1 1 . 1 ^ { \circ }$ , and rotation error rises from at most 0.052 to 0.446 turns. Thus, the degradation is present in the learned representation before autoregressive error accumulates, even though the $H = 3 0$ runs attain lower training objectives.

![](images/58026345e06fea96e5815218e7092005423d96e85bc916f8733e871cb3e826de.jpg)

![](images/ce94ae709580023b0c6de6ea3576399d8860bf97d2b7620c7a507aea93bc9c09.jpg)

![](images/458e62c985f98fcc5a1ff1a65abe6c6efd8f0b661f7a086f48b8fdeec3feaf24.jpg)  
Figure 12: History-length and predictor-depth ablation on 2D quare dataset at a common 32-step rollout horizon. Solid lines use predicted rollout latents; dashed lines use real encoder latents and isolate representation and probe error. Color and marker shape denote predictor depth. Lower is better.

## D.2.2 Approach Ball: history and training rollout horizon

In this case we jointly vary $( H , K ) \in \{ ( 2 0 , 5 ) , ( 2 0 , 1 0 ) , ( 3 0 , 1 0 ) , ( 3 0 , 2 0 ) \}$ and $\lambda _ { \mathrm { S I G } } \in \{ 0 . 1 8 , 0 . 3 6 , 0 . 7 2 \}$ For each model, an MLP probe predicts the ball state $( x , y , z , v _ { x } , v _ { y } , v _ { z } )$ . Within each $( H , K )$ pair, we select $\lambda _ { \mathrm { S I G } }$ by probe-validation NMSE. Table 3 and Fig. 13 show that $( H , K , \lambda _ { \mathrm { S I G } } ) = ( 2 0 , 5 , 0 . 1 8 )$ gives the lowest probe-validation NMSE, direct-probe test NMSE, and 20-step rollout NMSE. Its 32-step rollout NMSE is 0.185, within 1.3% of the best value of 0.182 from (20, 10, 0.18), while using

61.3 rather than 73.5 GiB of peak host RAM. The best $H = 3 0$ variants are both less accurate and more expensive. We therefore use the task-specific setting $H = 2 0 , K = 5$ and $\lambda _ { \mathrm { S I G } } = 0 . 1 8$ for Approach Ball. The sweep uses one seed, and the resource values describe training-time host memory.
<table><tr><td colspan="3">Training configuration</td><td colspan="3">Normalized MSE ↓</td><td colspan="3">Training resource</td></tr><tr><td>H</td><td>K</td><td>λSIG</td><td>Probe val.</td><td>Probe test</td><td></td><td>Rollout@32</td><td>Decoder frames per batch</td><td>Peak host RAM (GiB)</td></tr><tr><td>20</td><td>5</td><td>0.18</td><td>0.078</td><td>0.110</td><td>0.185</td><td></td><td>800</td><td>61.3</td></tr><tr><td>20</td><td>5</td><td>0.36</td><td>0.118</td><td>0.148</td><td></td><td>0.232</td><td>800</td><td>69.4</td></tr><tr><td>20</td><td>5</td><td>0.72</td><td>0.192</td><td>0.225</td><td>0.363</td><td></td><td>800</td><td>67.2</td></tr><tr><td>20</td><td>10</td><td>0.18</td><td>0.080</td><td>0.114</td><td>0.182</td><td></td><td>960</td><td>73.5</td></tr><tr><td>20</td><td>10</td><td>0.36</td><td>0.092</td><td>0.125</td><td></td><td>0.198</td><td>960</td><td>80.4</td></tr><tr><td>20</td><td>10</td><td>0.72</td><td>0.110</td><td>0.143</td><td></td><td>0.237</td><td>960</td><td>66.4</td></tr><tr><td>30</td><td>10</td><td>0.18</td><td>0.137</td><td>0.166</td><td></td><td>0.347</td><td>1280</td><td>81.9</td></tr><tr><td>30</td><td>10</td><td>0.36</td><td>0.135</td><td>0.167</td><td></td><td>0.236</td><td>1280</td><td>75.5</td></tr><tr><td>30</td><td>10</td><td>0.72</td><td>0.152</td><td>0.185</td><td></td><td>0.263</td><td>1280</td><td>87.6</td></tr><tr><td>30</td><td>20</td><td>0.18</td><td>0.316</td><td></td><td>0.325</td><td>0.392</td><td>1600</td><td>111.7</td></tr><tr><td>30</td><td>20</td><td>0.36</td><td>0.321</td><td></td><td>0.330</td><td>0.401</td><td>1600</td><td>113.8</td></tr><tr><td>30</td><td>20</td><td>0.72</td><td>0.150</td><td></td><td>0.182</td><td>0.260</td><td>1600</td><td>116.3</td></tr></table>

Table 3: Approach Ball temporal ablation.

![](images/1cb680bad63d4114d89e66337278d30d7c701e40e6e69a41a4889261a736fafe.jpg)

![](images/f9e9460b2da9166e442ad77eb37ce1805f35f2fb97ec191c0e49d3ac1efb4f2d.jpg)  
Figure 13: Approach Ball accuracy and resource tradeofs after selecting λ<sub>SIG</sub> by probe-validation NMSE within each (H, K) pair. (a) Rollout NMSE at horizons 20 and 32; (b) peak host RAM and decoder frames per microbatch; and (c) the SIGReg sweep for $H = 2 0 , K = 5$ . The highlighted configuration is selected. Lower is better.

## D.3 Rollout discount

We take Arm Catcher Ball dataset and sweep $\gamma \in \{ 0 . 9 0 , 0 . 9 5 , 0 . 9 8 , 1 . 0 0 , 1 . 0 5 \}$ . All runs are trained to 20 epochs. We freeze the epoch-20 representation and evaluate an MLP state probe on $\mathfrak { h } _ { 4 4 }$ predicted latents. Table 4 shows that $\gamma = 0 . 9 5$ gives the lowest joint NMSE and velocity error at horizon 44. Its position error is 0.122, close to the best value of 0.118 at $\gamma = 1 . 0 0$ . We therefore use $\gamma = 0 . 9 5$ as the default across datasets.

Table 4: Arm Catcher Ball rollout-discount ablation at horizon 44.
<table><tr><td> $\gamma$ </td><td>Joint NMSE ↓ Position</td><td> $L _ { 2 } \downarrow$ </td><td>Velocity  $L _ { 2 } \downarrow$ </td></tr><tr><td>0.90</td><td>0.753</td><td>0.134</td><td>0.398</td></tr><tr><td>0.95</td><td>0.600</td><td>0.122</td><td>0.346</td></tr><tr><td>0.98</td><td>0.683</td><td>0.138</td><td>0.351</td></tr><tr><td>1.00</td><td>0.610</td><td>0.118</td><td>0.354</td></tr><tr><td>1.05</td><td>0.730</td><td>0.145</td><td>0.377</td></tr></table>

## D.4 Counterfactual gravity conditioning

In this section, we want to test whether the model actually uses the input gravity. We conduct experiments by supplying wrong gravity values in the action, while holding other coordinates fixed. We again train and evaluate on the 2D square dataset. For each $g _ { \mathrm { t r u e } } \in \{ 2 , 4 , 6 \}$ , we use the same 200 episodes under $g _ { \mathrm { i n } } \in \{ 0 , 2 , 4 , 6 , 8 \}$ . The full-state probe MSE is normalized by the training-set variance of each target coordinate. Let $( { \mathfrak { h } } _ { 1 } , \dots , { \mathfrak { h } } _ { 7 } ) = ( 1 , 3 , 5 , 9 , 2 0 , 3 2 , 4 4 )$ denote the evaluated forecast steps. We summarize rollout error using the horizon-normalized area under the NMSE curve, computed by the trapezoidal rule:

$$
\mathrm { A U C } ( g _ { \mathrm { i n } } ) = \frac { 1 } { \mathfrak { h } _ { 7 } - \mathfrak { h } _ { 1 } } \sum _ { i = 1 } ^ { 6 } \frac { \mathfrak { h } _ { i + 1 } - \mathfrak { h } _ { i } } { 2 } \left[ \mathrm { N M S E } _ { \mathfrak { h } _ { i } } ( g _ { \mathrm { i n } } ) + \mathrm { N M S E } _ { \mathfrak { h } _ { i + 1 } } ( g _ { \mathrm { i n } } ) \right] .
$$

We report the paired diference $\Delta \mathrm { A U C } = \mathrm { A U C } ( g _ { \mathrm { i n } } ) - \mathrm { A U C } ( g _ { \mathrm { t r u e } } ) ;$ ; positive values indicate that supplying the mismatched gravity increases rollout error.

Results From the figure we see that for both predictors, the error is minimized when $g _ { \mathrm { i n } } = g _ { \mathrm { t r u e } } ,$ while mismatched inputs increase the rollout error. Larger gravity mismatches generally produce larger increases, although the response is not symmetric around the true value. At $g _ { \mathrm { t r u e } } = 4$ , correct conditioning also gives the lowest GRU position error at every reported horizon. These interventions show that the predictor does use the supplied gravity when forecasting dynamics.

## D.5 Where does GRU advantage come from

In this section we investigate where the jointly trained GRU advantage comes from: the recurrent predictor itself, the representation it shapes, or a weakness in Transformer conditioning? To this end we conduct three controlled studies. Matching Transformer action conditioning and initialization narrows, but does not close the gap. Under encoder–predictor crossover, the advantage follows the source representation. Reducing history has little source-specific efect over the first five forecasts, but the GRU-trained representation is less sensitive to short context once recursive errors accumulate over a 44-step rollout. Together, these results identify the representation learned via GRU predictor during joint training as the main source of the gain. Detailed results are as follows.

![](images/4a076a122edc0b07170558af8bc7e7a9c25c2383fecfcf4e2bfb3b0c7d85e31d.jpg)

![](images/f293d2cb77ec8c86576fa69f96ee6a514d163c2ea7445c6f545f247c9c17354d.jpg)

![](images/541c0debc81445e9792a642c8878305a5bbda4fa3ccbe8f68fa01f88d98c1216.jpg)  
Figure 14: Counterfactual gravity conditioning Only the gravity supplied through the action channel is changed. Panels (a) and (b) report the increase in normalized full-state probe NMSE AUC for the GRU and Transformer predictors. Panel (c) reports the mean GRU position $L _ { 2 }$ error in the x-z plane across rollout horizons for $g _ { \mathrm { t r u e } } = 4$

All comparisons use the same seed and the same 5,000 held-out square episodes, with 200 episodes at each of 25 gravity values from −2 to 10. We first average across target dimensions, then across horizons within each episode, and finally across episodes. Subtracting the error obtained from the real encoder latent controls for source-specific probe readability. These intervals measure evaluation-episode variability; all world-model and predictor training uses one seed.

## D.5.1 Can Transformer conditioning close the joint-training gap?

The first experiment asks whether the joint-training gap comes from how actions are supplied to the Transformer or from the initialization of its conditioning layers. For GRU we concatenate each latent with its action embedding at the recurrent input. For the Transformer, we test three alternatives: concatenating each latent state with its action once before the first block, reintroducing the action in the attention and MLP paths of every block, or using the action to modulate every block through AdaLN. For AdaLN, we also initialize the conditioning gate to 0, 0.1, or 1, giving five Transformer variants in total. All variants use H = 20, K = 5, γ = 0.95, SIGReg coeficient 0.72, batch size 64, 20 epochs. More details are in Table 5

Results Fig. 15 shows the results. Initializing the AdaLN gate to 0.1 lowers the Transformer endpoint from 0.458 to 0.428 and closes 38.5% of the gap to the GRU. It remains worse than GRU performance of 0.380, however, with a paired diference of +0.048A gate of <sup>˙</sup> 1.0 degrades the rollout, while input and per-block action concatenation do not improve the aggregate endpoint over AdaLN-0. The best Transformer is worse than the GRU at 22 of 25 gravity values, and per-block concatenation is worse at 25 of 25. Some variants improve individual physical quantities, so the conclusion applies to aggregate rollout error: neither direct action injection nor gate initialization closes the jointly trained GRU gap.

Table 5: Predictor architectures in the action conditioning study.
<table><tr><td>Method</td><td>Action conditioning</td><td>Initialization Depth Width MLP width Parameters</td><td></td><td></td><td></td><td></td></tr><tr><td>GRU</td><td>Action concatenated at recurrent input</td><td>residual 0.1</td><td>3</td><td>512</td><td></td><td>14.57M</td></tr><tr><td>AdaLN-0</td><td>AdaLN in every block</td><td>gate 0</td><td>6</td><td>256</td><td>2048</td><td>14.98M</td></tr><tr><td>AdaLN-0.1</td><td>AdaLN in every block</td><td>gate 0.1</td><td>6</td><td>256</td><td>2048</td><td>14.98M</td></tr><tr><td>AdaLN-1</td><td>AdaLN in every block</td><td>gate 1</td><td>6</td><td>256</td><td>2048</td><td>14.98M</td></tr><tr><td>Input concat.</td><td>One pre-stack action fusion</td><td>residual 0.1</td><td>6</td><td>256</td><td>2776</td><td>14.98M</td></tr><tr><td></td><td>Per-block concat. Action fusion in attention and MLP paths residual 0.1</td><td></td><td>6</td><td>256</td><td>2304</td><td>14.97M</td></tr></table>

(a) Mean error over forecast steps 1 to 44

![](images/cb4e5c630269c004b3fc6268fb8d9c26ada353b48b2e39e4c961e71d845934a4.jpg)  
(b) Error growth during rollout

![](images/fcd0de3b50e7e4729525be4f00484f3de1ad1345405bbe6afe81f0b49d003529.jpg)  
Figure 15: Performance comparison on held-out test set. (a) Mean excess normalized target MSE over forecast steps 1–44; error bars are 95% episode-bootstrap intervals. (b) Excess error by autoregressive forecast step. All models use the same episode IDs and gravity assignments. Lower is better.

## D.5.2 Encoder–predictor crossover: full results

For the second experiment we freeze the encoder from the jointly trained GRU and Transformer with AdaLN, then fit a fresh GRU and Transformer predictor to each latent source. The training setups and hyperparameters are the same except for the predictor. In Section 4 we report the overall crossover result and the main interpretation. Here we provide the full metric breakdown in Table 6. The representation learned jointly with the GRU gives lower overall physical-state error. The new Transformer modestly reduces error on both sources, by -0.107 on the GRU-trained representation and -0.103 on the Transformer-trained representation, yet the GRU-trained representation remains better. This result shows that encoder trained with the GRU predictor produces latents that preserve dynamics-relevant information that either predictor can use. The lower Euclidean latent MSE of the Transformer-trained representation does not translate into better physical-state prediction, especially for velocity. This contrast confirms that preserving dynamics-relevant state is more informative than raw latent distance.

Table 6: Numerical results for the frozen-representation crossover. The primary endpoint and latent MSE are excess errors averaged over horizons 1 to 44; physical errors use predicted latents at horizon 44.
<table><tr><td>Frozen source</td><td>Fresh predictor Primary ↓ Latent MSE ↓ Position ↓ Velocity ↓ Orientation</td><td></td><td></td><td></td><td></td><td> ${ \binom { \circ } { \phantom { + } } } \downarrow$ </td><td>ω↓</td></tr><tr><td>GRU-trained</td><td>GRU</td><td>1.376</td><td>2.247</td><td>3.004</td><td>3.803</td><td>22.88</td><td>2.116</td></tr><tr><td>GRU-trained</td><td>Transformer</td><td>1.269</td><td>2.511</td><td>3.036</td><td>3.769</td><td>21.77</td><td>2.139</td></tr><tr><td>Transformer-trained GRU</td><td></td><td>1.555</td><td>1.638</td><td>2.828</td><td>5.343</td><td>23.15</td><td>2.282</td></tr><tr><td>Transformer-trained</td><td>Transformer</td><td>1.453</td><td>1.732</td><td>2.854</td><td>4.849</td><td>22.57</td><td>2.401</td></tr></table>

## D.5.3 History dependence of the frozen representations

The crossover shows that the advantage follows better representation. We next ask whether the GRU-trained representation is less dependent on long context. For each frozen source, we fit new GRU and Transformer predictors with $m \in \{ 2 , 4 , 8 , 2 0 \}$ context frames, where m is the number of most recent latents available to the predictor. Each model predicts the same five target frames and is evaluated through 44 autoregressive steps on held-out episodes. The primary comparison shortens the context from m = 20 (H20) to $m = 4$ (H4). Following Equation 4, we measure how much the excess physical-state NMSE changes when history is shortened relative to H20. A positive source diference means that the Transformer-trained representation loses more from shorter context. Intervals use 10,000 paired episode bootstrap resamples.

![](images/7d79f9483ac8938d963922651c9f299ae253d156328e0f8976fa945c7a892670.jpg)

![](images/6e0ce8750de7eaeefab4bf966eec14e5773523d612e1e8487e45f1acdff9fec4.jpg)  
Figure 16: History dependence of the frozen representations. (a) Change in excess physical-state NMSE at each forecast step when context is shortened from H20 to H4, averaged over newly trained GRU and Transformer predictors. Positive values mean that shorter context increases error. The gray region marks the first five forecasts; bands show 95% confidence intervals over paired episode diferences. (b) Transformer-source minus GRU-source history penalty over the first five forecasts and the full 44-step rollout. Positive values mean that the Transformer-trained representation depends more on long context.

Results Fig. 16 shows the results. (a) plots the change in error at each forecast step when the predictor receives four rather than 20 context frames. Negative values favor H4; positive values mean that shortening the context is harmful. We see that both representations benefit from H4 during the first two forecasts. From the third forecast onward, however, the Transformer-trained source remains above zero, while the GRU-trained source stays closer to zero and becomes consistently negative after h = 22. The GRU-trained representation therefore retains its rollout accuracy with substantially less context. Panel (b) summarizes this divergence. The Transformer-source minus GRU-source history penalty is only 0.015 over the first five forecasts, but grows to 0.089 over the full 44-step rollout. The gap is therefore small at the start of prediction and becomes pronounced only after predicted latents are fed back recursively. This long-rollout robustness provides a concrete explanation for the stronger dynamics performance of the GRU-trained representation.

## D.6 Performance on conventional control tasks

![](images/00ada19f6c987244cacba61f94dd7fbfce2f36558abe6ef00dce81571a47c2eb.jpg)

![](images/13981c8fd7491b3407e6b84f84642ed7aa9ef077756c0b4c2139b2224e758edc.jpg)  
Figure 17: Representative illustration of the datasets (a) OGBench-Cube Single and (b) DMC Reacher.

The dynamics experiments above favor the GRU predictor over Transformer. We next test whether this choice degrades performance on conventional control tasks. We compare Transformer, GRU, and SSM predictors on OGBench-Cube Single and DMC Reacher (Figure 17), changing only the predictor architecture. All models use the same ViT-tiny encoder, H10/K5 objective, five-step action blocks and 20-epoch training schedule. We evaluate the same latent-space CEM planner on 50 fixed held-out cases per model.

Table 7: Predictor comparison on conventional control tasks. Success is measured over 50 fixed held-out cases, with episode-bootstrap 95% confidence intervals. Latent metrics use all 127,000 held-out H10/K5 clips.
<table><tr><td>Dataset</td><td>Predictor</td><td></td><td>Success ↑ K5 latent MSE ↓</td><td>Mean K1–K5 MSE ↓</td><td>Planning latency (s) ↓</td></tr><tr><td>OGBench-Cube</td><td>Transformer</td><td>68% (54–80)</td><td>0.00590</td><td>0.00445</td><td>0.195</td></tr><tr><td></td><td>GRU</td><td>70% (56–82)</td><td>0.00561</td><td>0.00430</td><td>0.111</td></tr><tr><td></td><td>SSM</td><td>68% (54–80)</td><td>0.00497</td><td>0.00378</td><td>0.279</td></tr><tr><td>DMC Reacher</td><td>Transformer</td><td>100% (100–100)</td><td>0.00226</td><td>0.00182</td><td>0.568</td></tr><tr><td></td><td>GRU</td><td>98% (94–100)</td><td>0.00323</td><td>0.00289</td><td>0.316</td></tr><tr><td></td><td>SSM</td><td>98% (94–100)</td><td>0.00159</td><td>0.00132</td><td>1.115</td></tr></table>

We observe that SG-JEPA (GRU/SSM) predictors preserves downstream control success on both tasks(Table 7). It also has the lowest planning latency and the highest efective rank on both datasets. Thus, the conventional benchmarks show no material loss in control success from using a GRU/SSM predictor. Together with its advantage on the dynamics benchmarks, this result supports our use of the GRU as the default predictor.

## E 2D datasets full results

## E.1 Gravity-conditioned rollouts

In Section 3 we present overall comparison on 2D triangle and square datasets. In this section we elaborate on the detailed breakdown, first by per-held-out-test-gravity evaluation metrics, as in Fig. 18; then results by rollout horizon, as in Table 8.

![](images/ee16d549f4c6a2d0fc73f395d41178ec8ea82bed45cfab92b267efca2cdd1b1c.jpg)  
Figure 18: Per-gravity evaluation at $\mathfrak { h } _ { 4 4 }$ . Rows show right-triangle and square episodes; columns report planar position $L _ { 2 }$ error, planar velocity $L _ { 2 }$ error and cumulative-rotation MAE. The fading yellow background follows the relative training-gravity density $g \sim N ( 4 , 0 . 5 ^ { 2 } )$ .

Results SG-JEPA gives the strongest overall performance, with its clearest advantage appearing as rollout errors accumulate. $\mathrm { O n }$ the square, the GRU and SSM variants lead every metric at every reported step. SSM is slightly stronger at the shortest steps, while GRU is more accurate over longer recursive rollouts. Figure 18 shows that these gains extend across most of the held-out gravity grid, rather than being confined to the center of the training distribution. The few reversals occur mainly for velocity at the highest gravity values.

The triangle is more competitive at short and intermediate steps, where the baselines lead several translation entries. Their advantage does not persist: at $\mathfrak { h } _ { 4 4 }$ , SG-JEPA (GRU) has the lowest position error and SG-JEPA (SSM) has the lowest velocity error. DINO-WM retains the lowest cumulative-rotation error on the triangle, the only metric not led by SG-JEPA at the longest rollout. Across the two shapes, SG-JEPA therefore leads five of the six $\mathfrak { h } _ { 4 4 }$ shape-metric comparisons, supporting its advantage for long-horizon dynamics prediction.

<table><tr><td>Metric</td><td>Method</td><td> ${ \mathfrak { h } } _ { 5 }$ </td><td> $\mathfrak { h } _ { 1 6 }$ </td><td> ${ \mathfrak { h } } _ { 3 2 }$ </td><td> $\mathfrak { h } _ { 4 4 }$ </td></tr><tr><td colspan="6">Right triangle</td></tr><tr><td>Position  $L _ { 2 } ~ ( \mathrm { m } )$ </td><td>Original LeWM</td><td>1.105</td><td>1.943</td><td>2.325</td><td>2.625</td></tr><tr><td rowspan="6">Velocity  $L _ { 2 } ~ ( \mathrm { m / s } )$ </td><td>SG-JEPA (GRU)</td><td>0.968</td><td>2.167</td><td>2.486</td><td>2.529</td></tr><tr><td>SG-JEPA (SSM)</td><td>1.251</td><td>2.455</td><td>2.628</td><td>2.665</td></tr><tr><td>DINO-WM</td><td>0.532</td><td>1.510</td><td>2.329</td><td>2.764</td></tr><tr><td>Original LeWM</td><td>2.324</td><td>1.598</td><td>2.499</td><td>3.121</td></tr><tr><td>SG-JEPA (GRU)</td><td>2.258</td><td>1.920</td><td>2.837</td><td>3.001</td></tr><tr><td>SG-JEPA (SSM)</td><td>2.343</td><td>1.667</td><td>2.592</td><td>2.765</td></tr><tr><td rowspan="5">Cumulative rotation (turns)</td><td>DINO-WM</td><td>1.897</td><td>2.420</td><td>3.275</td><td>3.574</td></tr><tr><td>Original LeWM</td><td>0.065</td><td>0.269</td><td>0.640</td><td>0.946</td></tr><tr><td>SG-JEPA (GRU)</td><td>0.061</td><td>0.268</td><td>0.617</td><td>0.904</td></tr><tr><td>SG-JEPA (SSM)</td><td>0.059</td><td>0.264</td><td>0.618</td><td>0.911</td></tr><tr><td>DINO-WM</td><td>0.062</td><td>0.255</td><td>0.609</td><td>0.875</td></tr><tr><td colspan="6">Square Position  $L _ { 2 } ~ ( \mathrm { m } )$ </td></tr><tr><td></td><td>Original LeWM</td><td>0.930</td><td>1.934</td><td>2.042</td><td>2.074</td></tr><tr><td rowspan="5">Velocity  $L _ { 2 } ~ \mathrm { ( m / s ) }$ </td><td>SG-JEPA (GRU)</td><td>0.374</td><td>0.658</td><td>0.862</td><td>1.034</td></tr><tr><td>SG-JEPA (SSM)</td><td>0.395</td><td>0.950</td><td>1.094</td><td>1.148</td></tr><tr><td>DINO-WM</td><td>0.454</td><td>1.117</td><td>1.570</td><td>2.006</td></tr><tr><td>Original LeWM</td><td>1.791</td><td>1.545</td><td>2.735</td><td>2.631</td></tr><tr><td>SG-JEPA (GRU)</td><td>1.158</td><td>0.896</td><td>1.455</td><td>1.915</td></tr><tr><td rowspan="5">Cumulative rotation (turns)</td><td>SG-JEPA (SSM) DINO-WM</td><td>1.051</td><td>1.007</td><td>1.661</td><td>2.007</td></tr><tr><td></td><td>1.793</td><td>1.921</td><td>3.011</td><td>3.082</td></tr><tr><td>Original LeWM</td><td>0.029</td><td>0.104</td><td>0.215</td><td>0.321</td></tr><tr><td>SG-JEPA (GRU)</td><td>0.018</td><td>0.066</td><td>0.143</td><td>0.236 0.249</td></tr><tr><td>SG-JEPA (SSM) DINO-WM</td><td>0.015 0.027</td><td>0.061 0.104</td><td>0.146 0.223</td><td>0.341</td></tr></table>

Table 8: Rollout results. Forecast steps h<sub>5</sub>, h<sub>16</sub>, h<sub>32</sub>, and $\mathfrak { h } _ { 4 4 }$ correspond to 0.3125, 1.0, 2.0, and 2.75 seconds. bold and underlining mark the best and second-best value within each shape, metric and step.

## E.2 Shape generalization

In this separate ablation we study the model’s ability of shape generalization. To do this we conduct the following experiment: We train SG-JEPA on both 2D right triangle and square datasets. And we ask whether it can generalize dynamics from these component shapes to the composite "house" shape (A "house" joins a right triangle and a square into one rigid body), which is unseen during training. We perform two tests, one trained with triangle + square (T+S), the other with triangle + square + pentagon $( T { + } S { + } P ) $ . We then compare the performance against SG-JEPA trained with only "house" dataset on downstream probe evaluation. Moreover we trained with both Tiny ViT $( D = 1 9 2 )$ and Small ViT $( D = 3 8 4 )$ encoders for these cases to rule out the possibility of model capacity limit when dealing with multiple datasets. All six world models follow the planar SG-JEPA

recipe in Appendix Section C.

Table 9: House rollout error at $\mathfrak { h } _ { 4 4 }$ , averaged over the seven gravity values in the band $g \in [ 2 . 5 , 5 . 5 ]$
<table><tr><td></td><td>Encoder World-model training data Position</td><td> $L _ { 2 } ~ ( \mathrm { m } )$ </td><td>Velocity  $L _ { 2 } ~ \mathrm { ( m / s ) }$ </td><td>Cumulative rotation (turns)</td></tr><tr><td>Tiny</td><td> $T { \ + } S$ </td><td>2.681</td><td>2.846</td><td>0.821</td></tr><tr><td>Tiny</td><td> $T { + } S { + } P$ </td><td>2.733</td><td>2.792</td><td>0.832</td></tr><tr><td>Tiny</td><td>House-trained</td><td>1.723</td><td>1.444</td><td>0.193</td></tr><tr><td>Small</td><td> $T { \ + } S$ </td><td>2.489</td><td>2.725</td><td>0.787</td></tr><tr><td>Small</td><td> $T { + } S { + } P$ </td><td>2.556</td><td>2.762</td><td>0.833</td></tr><tr><td>Small</td><td>House-trained</td><td>1.781</td><td>1.607</td><td>0.234</td></tr></table>

Results In Table 9 and Fig. 19 we present the results. Overall, SG-JEPA transfers a substantial part of the house’s translational dynamics from its component shapes. Position transfers most consistently, velocity is less stable, and rotation remains the hardest quantity to predict. Direct house training is still better near the training distribution. As shown in Table 9, within $g \in [ 2 . 5 , 5 . 5 ]$ Small $T { \ + } S$ has 1.40× the position error and 1.70× the velocity error of the house-trained reference. Adding pentagons brings no consistent improvement, so the observed transfer already comes from the triangle and square components rather than from greater shape diversity.

The main highlight is the OOD gravity result in Figure 19. Across much of both gravity tails, the $T { \ + } S$ model closely follows the house-trained reference on position despite never seeing a house during world-model training. With the Small encoder, it matches or outperforms the house-trained model at several extreme gravity values.

The true-latent control helps localize the remaining gap. On Small $T { \ + } S$ , probing encoded ground-truth house frames gives error ratios of 0.90 for position, 1.23 for velocity, and 1.12 for cumulative rotation relative to the house-trained reference. After autoregressive rollout, the ratios rise to 1.40, 1.70, and 3.36. Thus most of the rotation gap develops during recursive prediction, where small angular-velocity errors accumulate and the new contact geometry matters.

![](images/10503ddf6ebb0d163dde4e385f913911cee9baefa3a4ddabd2196cf64c712140.jpg)  
Figure 19: Per-gravity house rollout error at $\mathfrak { h } _ { 4 4 }$ . Rows show the Tiny (D = 192) and Small (D = 384) encoders; columns show position, velocity, and cumulative-rotation error.

## F 3D datasets full results

## F.1 Approach-Ball results

In main text, Fig. 4(a)(b) we report Approach-Ball position error over forecast steps and held-out gravity values. Here we show the complete results with velocity error and a compact summary of both physical quantities. Note that for this dataset we do not have ball rotation or friction, so the only physical quantities are position and velocity.

Table 10: Approach-Ball physical-state prediction error on 1,600 held-out episodes. The mean is taken over $\mathfrak { h } _ { 1 } - \mathfrak { h } _ { 4 4 }$ , and $\mathfrak { h } _ { 4 4 }$ is 2.75 s at 16 fps. Errors are reported in physical units. Lower is better; bold marks the best result.
<table><tr><td rowspan="2">Method</td><td colspan="2">Position  $L _ { 2 }$  (m)</td><td colspan="2">Velocity  $L _ { 2 } ~ ( \mathrm { m } \mathrm { s } ^ { - 1 } )$ </td></tr><tr><td>Mean</td><td>h44</td><td>Mean</td><td>h44</td></tr><tr><td>Original LeWM</td><td>0.0986</td><td>0.1229</td><td>1.1361</td><td>1.1400</td></tr><tr><td>SG-JEPA (GRU)</td><td>0.0491</td><td>0.0764</td><td>0.5391</td><td>0.7829</td></tr><tr><td>SG-JEPA (SSM)</td><td>0.0495</td><td>0.0785</td><td>0.5664</td><td>0.8861</td></tr><tr><td>DINO-WM</td><td>0.0705</td><td>0.1152</td><td>0.5672</td><td>0.7411</td></tr></table>

Results Position gives the clearest result. The full state readout reinforces the finding in the main text: SG-JEPA (GRU) and SG-JEPA (SSM) have the two lowest position errors, both on average and at the final step, and reduce mean position error by roughly 30% relative to DINO-WM.

Velocity does not show the same uniform advantage however. GRU has a slightly lower rolloutaverage error than DINO-WM, while SSM and DINO-WM are nearly tied. DINO-WM is more accurate at $\mathfrak { h } _ { 4 4 }$ . As shown in Figure 20(a), GRU or SSM is lower over much of the intermediate rollout, but DINO-WM closes the gap and becomes stronger near the end.

The gravity breakdown in Figure 20(b) also separates the extrapolation regimes. GRU is strongest across most of the central and moderate-gravity range, DINO-WM at the high-gravity tail, and Original LeWM at the few lowest values. Taken together, these results show a clear SG-JEPA advantage for position. For velocity, SG-JEPA is competitive with, but not uniformly better than, DINO-WM. Each world model uses one training seed, so these episode averages do not measure variation across training runs.

$$
\mathrm  ~ \left. ~ - \theta ~ o r i g i n a l ~ L e W M ~ \begin{array} { l } { { \mathrm { ~ --- ~ } } } \end{array} ~ { \sf S G - J E P A ~ ( G R U ) } \quad \begin{array} { l } { { \mathrm { ~ --- ~ } } } \end{array} \right. ~ { \sf S G - J E P A ~ ( S S M ) } \quad \mathrm { ~ --- ~ } ~ { \sf D I N O - M M }
$$

![](images/0ef327ba8b43edadb82f10d62542756bf7971d0efa12bece9da9b543f674cdfd.jpg)

![](images/4b086aa94f82a12dc74a1e774cce4c7a5f9a4625f6c01d14c8d6c52677e78ed6.jpg)  
Figure 20: Approach-Ball velocity prediction. The evaluation and plotting setup matches the position results in Figure 4(a,b). (a) Mean ball-velocity $L _ { 2 }$ error at each recursive forecast step. (b) Mean error over $\mathfrak { h } _ { 1 } - \mathfrak { h } _ { 4 4 }$ at each of the 25 held-out gravity values. The fading yellow background shows the training distribution $g \sim \mathcal { N } ( 9 . 8 , 2 . 0 ^ { 2 } )$ ).

## F.2 Arm-Catcher-Ball control results

In main text Figure $4 ( \mathrm { c } , \mathrm { d } )$ we present the success rate comparison for Arm Catcher Ball for SG-JEPA (GRU) and DINO-WM. In this section we report additional results for SG-JEPA (SSM) and Original LeWM (Transformer predictor) for completeness. Difusion policies are trained for each checkpoint respectively. At inference, the policy conditions on a 20-frame latent history, predicts 16 Cartesian actions, executes eight, and replans. Sampling uses the complete 100-step cosine DDPM. Success requires the ball to enter the catcher and be latched before the episode ends. We evaluate the same test set at 23 gravity values under five paired simulator-rollout seeds.

Results In Fig. 21 we observe that SG-JEPA more than doubles the overall capture rate, improving on DINO-WM by 13.8%. The gain is consistent across all five paired rollouts, ranging from 12.9% to 14.3%. It is also not confined to the center of the training distribution. Fig. 4(d) shows a positive mean advantage at every gravity, with a significant gap from $g = 0$ through $g = 9$ . At the training gravity, SG-JEPA reaches 42.3% compared with 16.0% for DINO-WM. Most remaining failures occur at the extreme gravity values or run to the full horizon without a capture. We also compare Original

![](images/e82d29f07c5c75c741ad228ebe86beb1b07a12c66246e27daea5b94735aa70c4.jpg)

![](images/a546ee0acec147ef42339aedfa21e2ae700e95b023e6120269b3448a7942480b.jpg)  
Figure 21: Arm-Catcher-Ball predictor-family comparison. (a) Overall capture rate. Results are averaged over 5 rollout executions. (b) Comparison of four methods by gravity. The fading yellow background shows the training distribution $g \sim \mathcal { N } ( 4 , 0 . 5 ^ { 2 } )$

LeWM and both SG-JEPA predictors. Original LeWM uses its one-step Transformer objective with $\lambda _ { \mathrm { S I G } } = 0 . 0 9$ , while the GRU and SSM variants use the five-step SG-JEPA objective with $\lambda _ { \mathrm { S I G } } = 0 . 1 8$ Both SG-JEPA variants remain ahead of the baselines overall.

Overall, Arm Catcher Ball is a phase-sensitive interception task: successful control requires the policy to preserve the ball’s velocity and contact phase, not merely its current position. As gravity increases, shorter bounce intervals amplify small timing errors and can shift the predicted trajectory to the wrong contact phase. SG-JEPA’s consistent advantage across a broad gravity range therefore indicates that its representation retains more control-relevant dynamical information under changing physics. Performance nevertheless degrades under negative and extreme gravity values, where the motion departs qualitatively from the training regime.

## F.3 Franka paddle-to-basket results

Table 11: Four-method comparison. Relaxed success records valid post-strike basket contact, while strict entry requires the ball to enter the basket. “Strict | blade” conditions strict entry on valid paddle-blade contact. All values are percentages.
<table><tr><td>Representation</td><td>Strict entry</td><td>Relaxed success Blade contact Strict | blade</td><td></td><td></td></tr><tr><td>SG-JEPA (GRU)</td><td>30.36</td><td>64.72</td><td>93.70</td><td>32.40</td></tr><tr><td>Original LeWM</td><td>28.80</td><td>62.00</td><td>90.46</td><td>31.84</td></tr><tr><td>SG-JEPA (SSM)</td><td>30.00</td><td>61.22</td><td>93.54</td><td>32.07</td></tr><tr><td>DINO-WM</td><td>27.60</td><td>67.30</td><td>95.86</td><td>28.79</td></tr></table>

In main-text, Figu $^ { 4 ( \mathrm { c } , \mathrm { e } ) }$ compares SG-JEPA (GRU) and DINO-WM using five paired simulatorrollout seeds. Here we add results for Original LeWM and SG-JEPA (SSM) in Table 11 for

completeness.

Results Both SG-JEPA pipelines give the highest strict-entry rates in the four-method comparison. DINO-WM hits the ball slightly more often and has the highest relaxed-success rate, but converts fewer hits into strict entries. The advantage of SG-JEPA therefore appears after contact, where the policy must produce the appropriate outgoing position and velocity rather than merely intercept the ball. GRU and SSM are efectively tied in this fixed rollout.

In Fig. 22 we show the per-gravity breakdown for paddle hit and final basket entry success rates respectively. Paddle-hit rates remain high beyond the two weakest gravity values, while strict entry varies substantially, reinforcing that post-impact targeting is the harder stage. The entry ordering also changes sharply outside the center of the training distribution. The two largest reversals persist under all five rollout seeds: at $g = 1 3 . 9 \ \mathrm { m / s ^ { 2 } }$ , DINO-WM averages 61.2% strict entry and SG-JEPA (GRU) 1.8%; at $g = 1 6 . 0 \ \mathrm { m / s ^ { 2 } }$ , SG-JEPA averages 58.6% and DINO-WM 0.8%. The five runs use the same episode IDs, action scaling, and fixed checkpoint hashes, with the rollout seed as the only change. These are therefore reproducible task-specific reversals, although two gravity values are not enough to support a broader mechanistic conclusion.

![](images/e18fc80b241762c701b6f534fc472393fa7f3641612005deb23b9321723a1e8a.jpg)  
Figure 22: Gravity-conditioned Franka Basket outcomes in the single-seed four-method comparison. (a) Strict basket entry and (b) valid paddle-blade contact. Each point contains 200 paired episodes. The fading yellow background follows the training-gravity density $g \sim \mathcal { N } ( 9 . 8 , 2 . 0 ^ { 2 } )$ . All control and evaluation settings are matched across methods.

The high paddle-hit rates show that interception is not the main bottleneck. Success instead depends on controlling the ball’s post-impact state: small errors in contact timing, paddle orientation, or incoming velocity can change the outgoing impulse enough to miss the basket. Gravity further changes the required trajectory: weak gravity prolongs flight and can produce overshoot, whereas strong gravity demands a larger and more precisely directed impulse. SG-JEPA’s higher entry rate therefore suggests that its representation better preserves the velocity and contact-phase information needed for gravity-dependent impact control.

## F.4 Arm-Paddle-Ball results

Arm Paddle Ball requires repeated contact control: the policy must keep the ball of the floor and produce at least one qualified bounce. Main-text Fig. 4(c,f) compares the final policy configurations. Here we evaluate the model’s capability of understanding the ball dynamics by training probes and evaluated on held-out test set.

![](images/03e53fd277984a8cf682c1b172b485fbd82cda662b05ef950684023ad954b1ac.jpg)

![](images/c9078603264d4ea6528308e3a324aeac1e892dc8f5e79f4eeb2521ef1ef6e6a4.jpg)  
Figure 23: Open-loop Arm-Paddle-Ball prediction. Mean Euclidean error for (a) ball position and (b) ball velocity over 5,000 held-out episodes. SG-JEPA has lower position error from ${ \mathfrak { h } } _ { 9 }$ onward and lower velocity error at every forecast step.

Prediction results. SG-JEPA has lower position error at 36 of 44 forecast steps and lower velocity error at all 44. At $\mathfrak { h } _ { 4 4 }$ , position error falls from 0.119 to 0.114 m and velocity error from 0.635 to $0 . 4 6 6 \mathrm { m / s } .$ . The larger velocity gap is relevant to sustained bouncing, where small phase errors carry into the next contact. This probe comparison is evidence for more accurate open-loop dynamics.

## G Sparse-Gravity Post-Training Ablation

In this ablation we study whether post-training on a subset of the test set with out-of-distribution gravity values improves long-horizon dynamics prediction between an beyond these values. We evaluate 2D right triangle and square datasets and four methods (Original LeWM, SG-JEPA (GRU), SG-JEPA (SSM) and DINO-WM).

The target post-training set contains 3,200 episodes: 800 episodes at each of the four support gravity values $\mathcal { G } _ { \mathrm { s u p } } = \{ 0 , 2 , 6 , 8 \}$ . The target-only arm uses these 3,200 episodes. The mixed arm adds 5,000 episodes sampled from the original source-training distribution, for 8,200 episodes in total. So for each test value it’s roughly 85 : 15% split. These data are excluded from later evaluation to avoid leakage. Each model is warm-started from its corresponding pretrained source checkpoint and optimized for 20 epochs with a fresh optimizer. We then report rollout errors at horizon 44. The held-out evaluation partition used here contains 2,500 episodes, with 100 episodes at every gravity in $\{ - 2 , - 1 . 5 , \hdots , 9 . 5 , 1 0 \}$ . For condition $c ,$ metric $m ,$ and gravity $^ { g , }$ let $E _ { c , m } ( g )$ be the mean error and let $E _ { 0 , m } ( g )$ denote the corresponding pretrained error. The aggregate improvement is

$$
\Delta _ { c } = 1 0 0 \left[ 1 - \frac { 1 } { 3 | \mathcal { G } _ { \mathrm { i n t } } | } \sum _ { m = 1 } ^ { 3 } \sum _ { g \in \mathcal { G } _ { \mathrm { i n t } } } \frac { E _ { c , m } ( g ) } { E _ { 0 , m } ( g ) } \right] ,\tag{11}
$$

where $\mathcal { G } _ { \mathrm { i n t } } = \{ 0 , 0 . 5 , \ldots , 8 \} \setminus \mathcal { G } _ { \mathrm { s u p } }$ contains the 13 held-out interpolation gravity values. Thus, a positive value indicates a reduction in error relative to the same model’s pretrained baseline. We also check whether post-training preserves performance near the source distribution. A model passes this source-range retention test if, for each metric, its mean error over $g \in \{ 2 . 5 , 3 , . . . , 5 . 5 \}$ is no more than 5% above its pretrained error.

Results Table 12 and Figures 24–25 support four main observations.

• Mixed post-training reduces aggregate error relative to the pretrained model in all eight shape and model combinations. It also outperforms target-only post-training in every combination. The mean reduction increases from 6.85% to 14.05%, a gain of 7.21 percentage points.

• SG-JEPA (GRU) improves by 20.84% on the triangle and 13.60% on the square. The largest mixed improvement for each shape is 20.84% for triangle SG-JEPA (GRU) and 20.81% for square Original LeWM.

• The per-gravity curves show that adaptation does more than improve the four observed support gravity values. Both post-training conditions reduce error at many of the unobserved gravity values between them, and the largest separations from the pretrained curves generally occur toward the low- and high-gravity ends. The efect is therefore consistent with interpolation across the adapted gravity range.

• The gains are not uniform across all metrics. Target-only DINO-WM slightly worsens triangle rotation by 0.22%, while Mixed DINO-WM worsens square velocity by 4.13%. Around the central source range, where the pretrained curves are already lower, the changes are smaller and more method dependent. Four of the eight Mixed configurations pass the source-range retention test; none of the Target-only configurations pass it.

The consistent advantage of Mixed over Target-only suggests that a modest amount of data at a few target gravity values can improve prediction at the unobserved gravity values between them. Source replay also gives stronger aggregate performance than using the target trajectories alone. In settings where the governing dynamics change, this ofers an alternative to training a new world model from scratch: collect a small target set and post-train with source replay.

Figures 24 and 25 provide the complete per-gravity breakdown.

<table><tr><td>Shape</td><td>Predictor</td><td>Arm</td><td>Overall</td><td>Position</td><td>Velocity</td><td>Rotation</td><td>Source retained</td></tr><tr><td>Triangle</td><td>Transformer</td><td>Target-only</td><td>4.22</td><td>5.30</td><td>5.65</td><td>1.70</td><td>no</td></tr><tr><td></td><td>Transformer</td><td>Mixed</td><td>13.80</td><td>17.29</td><td>13.64</td><td>10.47</td><td>no</td></tr><tr><td></td><td>GRU</td><td>Target-only</td><td>17.05</td><td>17.04</td><td>20.83</td><td>13.29</td><td>no</td></tr><tr><td></td><td>GRU</td><td>Mixed</td><td>20.84</td><td>19.69</td><td>21.63</td><td>21.20</td><td>yes</td></tr><tr><td></td><td>SSM</td><td>Target-only</td><td>6.61</td><td>2.73</td><td>11.29</td><td>5.80</td><td>no</td></tr><tr><td></td><td>SSM</td><td>Mixed</td><td>14.53</td><td>12.93</td><td>15.93</td><td>14.74</td><td>yes</td></tr><tr><td></td><td>DINO-WM</td><td>Target-only</td><td>1.79</td><td>2.92</td><td>2.66</td><td>-0.22</td><td>no</td></tr><tr><td></td><td>DINO-WM</td><td>Mixed</td><td>6.12</td><td>5.38</td><td>7.10</td><td>5.89</td><td>yes</td></tr><tr><td>Square</td><td>Transformer</td><td>Target-only</td><td>10.41</td><td>11.46</td><td>8.21</td><td>11.58</td><td>no</td></tr><tr><td></td><td>Transformer</td><td>Mixed</td><td>20.81</td><td>25.43</td><td>18.48</td><td>18.51</td><td>no</td></tr><tr><td></td><td>GRU</td><td>Target-only</td><td>5.39</td><td>7.99</td><td>6.61</td><td>1.56</td><td>no</td></tr><tr><td></td><td>GRU</td><td>Mixed</td><td>13.60</td><td>9.09</td><td>9.60</td><td>22.12</td><td>yes</td></tr><tr><td></td><td>SSM</td><td>Target-only</td><td>3.71</td><td>6.96</td><td>2.51</td><td>1.66</td><td>no</td></tr><tr><td></td><td>SSM</td><td>Mixed</td><td>12.15</td><td>17.84</td><td>10.95</td><td>7.66</td><td>no</td></tr><tr><td></td><td>DINO-WM</td><td>Target-only</td><td>5.59</td><td>15.85</td><td>0.05</td><td>0.86</td><td>no</td></tr><tr><td></td><td>DINO-WM</td><td>Mixed</td><td>10.55</td><td>21.43</td><td>-4.13</td><td>14.35</td><td>no</td></tr><tr><td colspan="2">Target-only mean</td><td></td><td>6.85</td><td>8.78</td><td>7.22</td><td>4.53</td><td></td></tr><tr><td colspan="2">Mixed mean</td><td></td><td></td><td>14.05 16.14</td><td>11.65</td><td>14.37</td><td></td></tr></table>

Table 12: Improvement $( \% )$ on the 13 held-out interpolation gravity values, relative to each predictor’s pretrained baseline. “Overall” is $\Delta _ { c }$ from Eq. (11); “Source retained” reports the source-range retention test defined above.

![](images/a9025f691a74b8f0d74a0f416cf71a035e68d2954d7acb070d36a7ab9755736d.jpg)  
Figure 24: Right-triangle horizon-44 errors. Rows correspond to the four predictors and columns to position, velocity, and cumulative rotation. Curves show the pretrained baseline, target-only post-training, and mixed post-training over all 25 evaluation gravity values.

![](images/e15cc66e8fcc9f2bc0a1c9c755512917b37abbaf1d88fe307b47e53ca63dd5f7.jpg)  
Figure 25: Square horizon-44 errors at epoch 20, using the same layout and conventions as Fig. 24.

## H Theory of prediction under changing physical laws

This appendix develops the theory used in Section 4. The analysis separates three questions. First, what makes a latent state locally predictive when gravity changes? Second, how does a local error enter a recursive rollout? Third, why can a multi-step objective prefer a diferent representation from a one-step objective, e.g. one-step favours DINO-WM but after rollout SG-JEPA has advantage? We answer these questions with a linear feature model. The model is simple enough to analyze exactly, but it is not an identification of the neural encoder or the GRU with linear operators.

We first summarize the main results.

• The one-step latent error separates into predictor error on learned features, information discarded by the representation that still afects the next state, and a conditionally zero-mean residual (Definition H.1 and Proposition H.2).

• If the true and learned dynamics share the same dependence on gravity, average training error bounds the systematic one-step test error, scaled by gravity coverage (Theorem H.4). For afine free flight, the factor is $1 + ( g _ { \star } - \mu _ { \mathrm { t r } } ) ^ { 2 } / \sigma _ { \mathrm { t r } } ^ { 2 }$

• Rollout recursion shows how local error and transition residual are transformed by later predicted dynamics. The corresponding physical-state bound also retains readout error and the transition residual (Theorems H.10 and H.12).

• A multi-step loss applies a diferent quadratic geometry to one-step transition error. Two representations can therefore exchange rank when the training horizon changes (Theorem H.15).

## H.1 Setup: the model

The model mirrors the training objective and fixes the notation used in the proofs.

Implementation-aligned objective. Let $e _ { \vartheta }$ be the shared trainable encoder and projector, and denote the latent vector $\boldsymbol { z } _ { t } = e _ { \vartheta } ( o _ { t } ) \in \mathbb { R } ^ { d _ { \boldsymbol { z } } }$ for observation $O t$ . Let $c _ { t } = q ( u _ { t } , g )$ be the actionconditioning of the action $u _ { t }$ and gravity g. The predictor receives a context of $H \geq 1$ latent states. Starting from a true context, it is applied recursively for $K \geq 1$ training steps:

$$
\widehat { z } _ { t + k } = f \Bigl ( z _ { t + k - H : t + k - 1 } ^ { \mathrm { r o l l } } , c _ { t + k - H : t + k - 1 } \Bigr ) , \qquad k = 1 , \ldots , K ,\tag{12}
$$

$$
z _ { t + j } ^ { \mathrm { r o l l } } = \left\{ \begin{array} { l l } { z _ { t + j } , } & { j \leq 0 , } \\ { \widehat { z } _ { t + j } , } & { j \geq 1 . } \end{array} \right.\tag{13}
$$

For normalized weights

$$
w _ { k } = \frac { \gamma ^ { k - 1 } } { \sum _ { j = 1 } ^ { K } \gamma ^ { j - 1 } } , \qquad \gamma > 0 , \qquad w _ { k } \geq 0 , \qquad \sum _ { k = 1 } ^ { K } w _ { k } = 1 ,\tag{14}
$$

the latent rollout loss is

$$
\mathcal { L } _ { K } ( \boldsymbol { \vartheta } , f ) = \sum _ { k = 1 } ^ { K } w _ { k } \mathbb { E } \left[ \Vert \widehat { \boldsymbol { z } } _ { t + k } - \boldsymbol { z } _ { t + k } \Vert _ { 2 } ^ { 2 } \right] .\tag{15}
$$

The expectation is over the training clips, including initial states, gravity values, action sequences, and stochastic residuals when present. The predictor helps determine which representations receive low training loss. We express this dependence through the profiled representation risk

$$
\mathcal { I } _ { K , H , \mathcal { F } } ( \boldsymbol { \vartheta } ) = \operatorname* { i n f } _ { f \in \mathcal { F } _ { H } } \mathcal { L } _ { K } ( \boldsymbol { \vartheta } , f ) + \lambda _ { \mathrm { S I G } } \mathcal { R } _ { \mathrm { S I G } } ( \boldsymbol { \vartheta } ) .\tag{16}
$$

Here $\mathcal { F } _ { H }$ is the predictor family for an H-step context. Unless marked as empirical, $\mathcal { I } _ { K , H , \mathcal { F } }$ denotes a population risk; $\mathcal { R } _ { \mathrm { S I G } }$ is the expected implemented batch statistic, with the expectation taken over data batches and random directions. The idealized population characteristic-function calculation is defined separately in Section H.8.

For the core argument, let scalar gravity $g \in \mathbb R$ be fixed within an episode. We assume that the feature state $\phi _ { t } \in \mathbb { R } ^ { d _ { \phi } }$ follows the linear transition

$$
\phi _ { t + 1 } = T ( g ) \phi _ { t } + \xi _ { t + 1 } \qquad \mathbb { E } [ \xi _ { t + 1 } \mid \phi _ { t } , g ] = 0 .\tag{17}
$$

When actions are present explicitly, the conditioning set in this equation also includes the known action input. The matrix $T ( g ) \in \mathbb { R } ^ { d _ { \phi } \times d _ { \phi } }$ gives the transition under gravity ${ \mathit { g } } ,$ and $\xi _ { t + 1 }$ is the remaining conditionally zero-mean residual. Systematic approximation error does not belong in $\xi _ { t + 1 } ;$ it must enter a separate deterministic term, such as the approximate-basis remainder introduced below. Section H.6 gives the exact finite-history form with actions. We model the learned representation as

$$
\begin{array} { r } { z _ { t } = W \phi _ { t } , \qquad W \in \mathbb { R } ^ { d _ { z } \times d _ { \phi } } , \qquad d _ { z } \leq d _ { \phi } , \qquad W W ^ { \top } = I _ { d _ { z } } . } \end{array}\tag{18}
$$

Row orthonormality fixes the latent coordinate scale inside the model. The learned one-step predictor is represented by $\widehat { A } ( g ) \in \mathbb { R } ^ { d _ { z } \times d _ { z } }$ . For the physical-state bounds, we assume that the physical quantity of interest is linear in the chosen feature coordinates:

$$
s _ { t } = C _ { \mathrm { p h y s } } \phi _ { t } ,\tag{19}
$$

where $C _ { \mathrm { p h y s } } \in \mathbb { R } ^ { d _ { s } \times d _ { \phi } }$ . Let $r : \mathbb { R } ^ { d _ { z } }  \mathbb { R } ^ { d _ { s } }$ be a frozen probe. When the probe is linear, we write $r ( z ) = R z$ with $R \in \mathbb { R } ^ { d _ { s } \times d _ { z } }$ . The experiments use learned probes, some of which read a short latent history. For a history probe, the input to r is the corresponding stacked latent window. The exact selector for a linear history readout is given in Section H.6. The probe identities below hold for any frozen probe with a compatible input, while the matrix bounds use the linear specialization. $s _ { t }$ is expressed in the fixed physical coordinates in which error is evaluated. We use H only for context length, K only for the training rollout length, and h for a generic evaluation horizon. Vector norms are Euclidean. Matrix norms are Frobenius norms unless a subscript 2 marks the spectral norm.

Fixed-gravity composition. For a fixed gravity and sampling interval, define

$$
S _ { g } ( 0 ) = I _ { d _ { \phi } } , \qquad S _ { g } ( \mathfrak { h } ) = T ( g ) ^ { \mathfrak { h } } , \widehat { S } _ { g } ( 0 ) \qquad = I _ { d _ { z } } , \widehat { S } _ { g } ( \mathfrak { h } ) \qquad = \widehat { A } ( g ) ^ { \mathfrak { h } } .\tag{20}
$$

For $k , \ell \in  { \mathbb { N } } _ { 0 }$

$$
S _ { g } ( k + \ell ) = S _ { g } ( \ell ) S _ { g } ( k ) , \qquad \widehat { S } _ { g } ( k + \ell ) = \widehat { S } _ { g } ( \ell ) \widehat { S } _ { g } ( k ) .\tag{21}
$$

Each gravity therefore indexes a pair of discrete evolution semigroups; gravity is an index, not the composition variable. In the residual-free model these operators describe the realized evolution. With conditionally zero-mean residuals, the true powers describe conditional-mean propagation, while realized trajectories retain the residual terms derived below. The rollout loss trains the accuracy of the first K learned iterates rather than the algebraic composition identity itself.

## H.2 Local prediction error under a fixed gravity

A latent state can fail locally for two reasons. It may discard information needed for the next transition, or the predictor may evolve the retained information incorrectly. The operators below

separate these terms. Define the orthogonal projector onto the row space of W and the two induced operators

$$
P _ { W } = W ^ { \top } W , \quad A _ { W } ( g ) = W T ( g ) W ^ { \top } , \quad C _ { W } ( g ) = W T ( g ) ( I - P _ { W } ) .\tag{22}
$$

Since $P _ { W } \phi _ { t } = W ^ { \top } z _ { t }$

$$
W T ( g ) \phi _ { t } = W T ( g ) P _ { W } \phi _ { t } + W T ( g ) ( I - P _ { W } ) \phi _ { t }\tag{23}
$$

$$
= A _ { W } ( g ) z _ { t } + C _ { W } ( g ) \phi _ { t } .\tag{24}
$$

Equivalently, $C _ { W } ( g ) = W T ( g ) - A _ { W } ( g ) W$ . Encoding Eq. 17 therefore gives

$$
z _ { t + 1 } = A _ { W } ( g ) z _ { t } + C _ { W } ( g ) \phi _ { t } + W \xi _ { t + 1 } .\tag{25}
$$

Definition H.1 (Local law-conditioned error). Let $E _ { A } ( g ) = A _ { W } ( g ) - \widehat { A } ( g )$ . The state-dependent conditional-mean error injected when the predictor starts from the true latent state is

$$
\delta _ { t } ( g ) = E _ { A } ( g ) z _ { t } + C _ { W } ( g ) \phi _ { t } .\tag{26}
$$

The term $E _ { A } ( g ) z _ { t }$ is predictor error on the state. The term $C _ { W } ( { \boldsymbol { g } } ) \phi _ { t }$ is the closure error realized at state $\phi _ { t } \mathbf { : }$ discarded features still afect the next latent. The operator $C _ { W } ( g )$ can be nonzero even on trajectories for which this realized vector happens to vanish.

We call the representation predictively closed at gravity g when $C _ { W } ( g ) = 0$ . In that case, the conditional mean of the next latent depends only on the current latent. Residual variation can still make individual transitions unpredictable.

In the residual-free linear model, predictive closure together with an exact latent operator, $\widehat { A } ( g ) = A _ { W } ( g )$ , gives $W T ( g ) = \widehat { A } ( g ) W$ . Hence

$$
W S _ { g } ( { \mathfrak { h } } ) = \widehat { S } _ { g } ( { \mathfrak { h } } ) W \qquad \mathrm { f o r ~ e v e r y ~ } { \mathfrak { h } } \geq 0 .\tag{27}
$$

With transition residuals, this intertwining identity applies to the conditional-mean evolution;   
individual trajectories still contain the propagated $W \xi _ { t + 1 }$ terms.

Proposition H.2 (What teacher forcing measures). Let $\widehat { z } _ { t + 1 } ^ { \mathrm { T F } } = \widehat { A } ( g ) z _ { t }$ . Then

$$
z _ { t + 1 } - \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } = \delta _ { t } ( g ) + W \xi _ { t + 1 } ,\tag{28}
$$

$$
R z _ { t + 1 } - R \hat { z } _ { t + 1 } ^ { \mathrm { T F } } = R \delta _ { t } ( g ) + R W \xi _ { t + 1 } ,\tag{29}
$$

$$
s _ { t + 1 } - R \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } = ( C _ { \mathrm { p h y s } } - R W ) \phi _ { t + 1 } + R \delta _ { t } ( g ) + R W \xi _ { t + 1 } .\tag{30}
$$

Proof. Subtract $\widehat { A } ( g ) z _ { t }$ from Eq. 25. This gives Eq. 28; applying R gives Eq. 29. For the physical identity, add and subtract $R W \phi _ { t + 1 } = R z _ { t + 1 }$

$$
s _ { t + 1 } - R \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } = C _ { \mathrm { p h y s } } \phi _ { t + 1 } - R W \phi _ { t + 1 } + R ( z _ { t + 1 } - \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } ) ,\tag{31}
$$

and substitute Eq. 28.

For any frozen, possibly nonlinear probe r, the decoded-latent comparison is $r ( z _ { t + 1 } ) - r ( \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } )$ The readout residual $s _ { t + 1 } - r \big ( z _ { t + 1 } \big )$ cancels because the same probe is applied to both latents. If r is $L _ { r } – \mathrm { L i p s c h i t z }$ on the line segment joining the two latents, then

$$
\begin{array} { r } { \big \| r ( z _ { t + 1 } ) - r ( \widehat { z } _ { t + 1 } ^ { \mathrm { T F } } ) \big \| _ { 2 } \leq L _ { r } \| \delta _ { t } ( g ) + W \xi _ { t + 1 } \| _ { 2 } . } \end{array}\tag{32}
$$

The teacher-forced diagnostic in Figure 5 uses this same-probe comparison, so the true-state probe floor cancels for both linear and nonlinear probes. The remaining error still combines the two terms in $\delta _ { t } ( g )$ with the transition residual. Under the conditional-zero-mean assumption, the expected cross term also vanishes after any fixed linear probe:

$$
\mathbb { E } \Big [ \delta _ { t } ( g ) ^ { \top } R ^ { \top } R W \xi _ { t + 1 } \mid g \Big ] = 0 .\tag{33}
$$

This follows from the tower property because $\delta _ { t } ( g )$ is measurable with respect to $( \phi _ { t } , g )$ and $\mathbb { E } [ \xi _ { t + 1 } \mid \phi _ { t } , g ] = 0$ . There is no corresponding cancellation guarantee for a nonlinear probe, and the result says nothing about the readout-residual cross term in Eq. 76. The diagnostic cannot, by itself, assign the diference to $E _ { A } ( g ) z _ { t }$ or $C _ { W } ( { \boldsymbol { g } } ) \phi _ { t }$ separately.

## H.3 How gravity coverage controls test error

The semigroup law composes evolution over h at fixed g. Generalization across gravity is a separate question: it depends on how the one-step operator varies with g and which law directions training covers. Low error on the training gravity values alone gives no bound at an unseen gravity, so we state the required shared structure explicitly.

Shared law basis Suppose a column vector $\psi ( g ) = ( \psi _ { 1 } ( g ) , \ldots , \psi _ { m } ( g ) ) ^ { \intercal }$ describes how gravity enters the transition, with finite second moments under the training distribution, and

$$
T ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) T _ { k } , ~ A _ { W } ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) A _ { k } ,\tag{34}
$$

$$
C _ { W } ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) C _ { k } , ~ \widehat { A } ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) \widehat { A } _ { k } .\tag{35}
$$

The first expansion implies the next two with $A _ { k } = W T _ { k } W ^ { \top }$ and $C _ { k } = W T _ { k } ( I - P _ { W } )$ . The expansion for $\widehat { A } ( g )$ is an additional structural assumption on the learned predictor. Equation equation 35 is therefore an exact-model assumption for the result below, not a consequence of providing g as an input. The neural predictor is not constrained to satisfy it exactly.

Let $P _ { \mathrm { t r } }$ be the training distribution over gravity and define

$$
M _ { \psi } = \mathbb { E } _ { g \sim P _ { \mathrm { t r } } } [ \psi ( g ) \psi ( g ) ^ { \top } ] .\tag{36}
$$

We assume $M _ { \psi } \succ 0$ , so the training distribution covers every direction in the chosen basis. The law-coverage factor at a test gravity $g _ { \star }$ is

$$
\mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) = \psi ( g _ { \star } ) ^ { \top } M _ { \psi } ^ { - 1 } \psi ( g _ { \star } ) .\tag{37}
$$

A large value means that the test law lies in a weakly covered direction of the training design.

Lemma H.3 (Transfer for a matrix-valued law family). Let $\begin{array} { r } { E ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) E _ { k } } \end{array}$ be any matrixvalued residual and set $\epsilon _ { E } ^ { 2 } = \mathbb { E } _ { g \sim P _ { \mathrm { t r } } } \left[ \Vert E ( g ) \Vert _ { F } ^ { 2 } \right] . \ I f M _ { \psi } \succ 0$ , then

$$
\| E ( g _ { \star } ) \| _ { F } ^ { 2 } \le \mathcal L _ { \mathrm { l a w } } ( g _ { \star } ) \epsilon _ { E } ^ { 2 } .\tag{38}
$$

Proof. Let E be the matrix whose kth column is vec $\left( E _ { k } \right)$ . Then vec $( E ( g ) ) = \mathsf E \psi ( g )$ and

$$
\epsilon _ { E } ^ { 2 } = \mathbb { E } \Big [ \psi ( g ) ^ { \top } \mathsf { E } ^ { \top } \mathsf { E } \psi ( g ) \Big ]\tag{39}
$$

$$
\mathbf { \Psi } = \mathrm { t r } ( \mathsf { E } M _ { \psi } \mathsf { E } ^ { \top } ) = \| \mathsf { E } M _ { \psi } ^ { 1 / 2 } \| _ { F } ^ { 2 } .\tag{40}
$$

At $g _ { \star }$

$$
\| E ( g _ { \star } ) \| _ { F } = \left\| \mathsf E M _ { \psi } ^ { 1 / 2 } M _ { \psi } ^ { - 1 / 2 } \psi ( g _ { \star } ) \right\| _ { 2 }\tag{41}
$$

$$
\leq \left. \mathsf E M _ { \psi } ^ { 1 / 2 } \right. _ { 2 } \left. M _ { \psi } ^ { - 1 / 2 } \psi ( g _ { \star } ) \right. _ { 2 }\tag{42}
$$

$$
\leq \| \mathsf E M _ { \psi } ^ { 1 / 2 } \| _ { F } \sqrt { \mathcal L _ { \mathrm { l a w } } ( g _ { \star } ) } .\tag{43}
$$

Substitute Eq. 40 and square both sides.

Define the average training-law errors

$$
\epsilon _ { \mathrm { c l } } ^ { 2 } = \mathbb { E } _ { g \sim P _ { \mathrm { t r } } } \left[ \lVert C _ { W } ( g ) \rVert _ { F } ^ { 2 } \right] ,\tag{44}
$$

$$
\epsilon _ { \mathrm { o p } } ^ { 2 } = \mathbb { E } _ { g \sim P _ { \mathrm { t r } } } \left[ \lVert A _ { W } ( g ) - \widehat { A } ( g ) \rVert _ { F } ^ { 2 } \right] .\tag{45}
$$

The first measures how far the representation is from predictive closure. The second measures how well the predictor fits the transition retained by the representation.

Theorem H.4 (Local error at an unseen gravity). Under the shared-basis assumptions in Eq. 35, suppose $\| z _ { t } \| _ { 2 } \leq B _ { z }$ and $\| \phi _ { t } \| _ { 2 } \leq B _ { \phi }$ . Then

$$
\| \delta _ { t } ( g _ { \star } ) \| _ { 2 } \leq \sqrt { { \mathcal L } _ { \mathrm { l a w } } ( g _ { \star } ) } \left( B _ { z } \epsilon _ { \mathrm { o p } } + B _ { \phi } \epsilon _ { \mathrm { c l } } \right) .\tag{46}
$$

Proof. Apply Lemma H.3 first to $C _ { W } ( g )$ and then to $E _ { A } ( g ) = A _ { W } ( g ) - \widehat { A } ( g )$ . This gives

$$
\| C _ { W } ( g _ { \star } ) \| _ { F } \le \sqrt { \mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) } \epsilon _ { \mathrm { c l } } ,\tag{47}
$$

$$
\lVert E _ { A } ( g _ { \star } ) \rVert _ { F } \leq \sqrt { \mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) } \epsilon _ { \mathrm { o p } } .\tag{48}
$$

Using Definition H.1, the triangle inequality, and $\lVert M x \rVert _ { 2 } \leq \lVert M \rVert _ { F } \lVert x \rVert _ { 2 }$ yields

$$
\| \delta _ { t } ( g _ { \star } ) \| _ { 2 } \leq \| E _ { A } ( g _ { \star } ) \| _ { F } \| z _ { t } \| _ { 2 } + \| C _ { W } ( g _ { \star } ) \| _ { F } \| \phi _ { t } \| _ { 2 } ,\tag{49}
$$

which gives the stated bound.

The theorem is a suficient transfer result. As a concrete example, for collision-free ballistic motion, gravity enters the transition afinely. This gives a concrete two-dimensional law basis.

Corollary H.5 (Coverage factor for afine gravity). Let $\psi ( g ) = ( 1 , g ) ^ { \intercal }$ . If the training gravity has mean $\mu _ { \mathrm { t r } }$ and variance $\sigma _ { \mathrm { t r } } ^ { 2 } > 0$ , then

$$
\mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) = 1 + \frac { ( g _ { \star } - \mu _ { \mathrm { t r } } ) ^ { 2 } } { \sigma _ { \mathrm { t r } } ^ { 2 } } .\tag{50}
$$

Proof. For this basis,

$$
M _ { \psi } = \left[ \begin{array} { c c } { { } } & { { } } \\ { { 1 } } & { { \mu _ { \mathrm { t r } } } } \\ { { } } & { { } } \\ { { \mu _ { \mathrm { t r } } } } & { { \mu _ { \mathrm { t r } } ^ { 2 } + \sigma _ { \mathrm { t r } } ^ { 2 } } } \end{array} \right] .\tag{51}
$$

Its determinant is $\sigma _ { \mathrm { t r } } ^ { 2 } .$ so

$$
M _ { \psi } ^ { - 1 } = \frac { 1 } { \sigma _ { \mathrm { t r } } ^ { 2 } } \left[ \begin{array} { l l } { \mu _ { \mathrm { t r } } ^ { 2 } + \sigma _ { \mathrm { t r } } ^ { 2 } } & { - \mu _ { \mathrm { t r } } } \\ { - \mu _ { \mathrm { t r } } } & { 1 } \end{array} \right] .\tag{52}
$$

Direct multiplication gives

$$
\left[ 1 \quad g _ { \star } \right] M _ { \psi } ^ { - 1 } \left[ \begin{array} { l } { 1 } \\ { \rule { 0 ex } { 5 ex } } \\ { g _ { \star } } \end{array} \right] = \frac { \mu _ { \mathrm { t r } } ^ { 2 } + \sigma _ { \mathrm { t r } } ^ { 2 } - 2 \mu _ { \mathrm { t r } } g _ { \star } + g _ { \star } ^ { 2 } } { \sigma _ { \mathrm { t r } } ^ { 2 } }\tag{53}
$$

$$
= 1 + \frac { ( g _ { \star } - \mu _ { \mathrm { t r } } ) ^ { 2 } } { \sigma _ { \mathrm { t r } } ^ { 2 } } .\tag{54}
$$

The moments in Eq. 50 are the mean and variance of the actual training distribution. The planar experiments draw $g = \operatorname* { m a x } \{ G , 0 . 1 \}$ with $G \sim \mathcal { N } ( 4 , 0 . 5 ^ { 2 } )$ , so the exact factor uses the moments of this clipped distribution.

Proposition H.6 (Free-flight dynamics remain afine in gravity). For time step $\Delta \ : > \ : 0$ and homogeneous state $\phi _ { t } = ( p _ { t } , v _ { t } , 1 ) ^ { \top }$ , let

$$
T _ { g } = \left[ \begin{array} { c c c } { { 1 } } & { { \Delta } } & { { - \frac { 1 } { 2 } g \Delta ^ { 2 } } } \\ { { } } & { { } } & { { } } \\ { { 0 } } & { { 1 } } & { { - g \Delta } } \\ { { } } & { { } } & { { } } \\ { { 0 } } & { { 0 } } & { { 1 } } \end{array} \right] .\tag{55}
$$

For every integer ${ \mathfrak { h } } \geq 1$

$$
T _ { g } ^ { \mathfrak { h } } = \left[ \begin{array} { c c c } { 1 } & { \mathfrak { h } \Delta } & { - \frac { 1 } { 2 } g ( \mathfrak { h } \Delta ) ^ { 2 } } \\ & & \\ { 0 } & { 1 } & { - g \mathfrak { h } \Delta } \\ & & & { } \\ { 0 } & { 0 } & { 1 } \end{array} \right] .\tag{56}
$$

Thus the transition remains afine in g at every free-flight horizon.

Proof. The formula holds at ${ \mathfrak { h } } = 1$ . Assume it holds at h. Multiplying $T _ { g } ^ { \mathfrak { h } }$ by $T _ { g }$ gives the next velocity coeficient − ${ \cdot } g ( \mathfrak { h } + 1 ) \Delta$ . The homogeneous contribution to position is

$$
\begin{array} { r } { - \frac 1 2 g ( { \mathfrak { h } } \Delta ) ^ { 2 } - { \mathfrak { h } } g \Delta ^ { 2 } - \frac 1 2 g \Delta ^ { 2 } = - \frac 1 2 g ( { \mathfrak { h } } + 1 ) ^ { 2 } \Delta ^ { 2 } . } \end{array}\tag{57}
$$

The remaining entries also match Eq. 56 with h replaced by $\mathfrak { h } + 1$ . Induction completes the proof.

This proposition applies only while the trajectory remains in free flight. It is a statement about the displayed physical-coordinate feature state; an arbitrary learned feature map need not inherit the same afine dependence. In more than one spatial dimension, the same calculation applies to the gravity-aligned position and velocity components, while tangential components have no gravity forcing in this idealized free-flight model. Contact can change the transition branch, as discussed in Section H.7.

## H.3.1 When the coverage result does not apply

The coverage theorem needs both a covered basis and a predictor that uses that basis.

Proposition H.7 (Missing law directions prevent a uniform bound). Suppose $M _ { \psi }$ is singular and there is an $a \in \ker ( M _ { \psi } )$ such that $a ^ { \top } \psi ( g _ { \star } ) \neq 0$ . Then a residual family can have zero average training error and nonzero error at $g _ { \star }$

Proof. Because $\boldsymbol { a } ^ { \top } M _ { \psi } \boldsymbol { a } = \mathbb { E } [ ( \boldsymbol { a } ^ { \top } \boldsymbol { \psi } ( \boldsymbol { g } ) ) ^ { 2 } ] = 0$ , we have $a ^ { \top } \psi ( g ) = 0$ almost surely under $P _ { \mathrm { t r } }$ . Choose any nonzero matrix $E _ { 0 }$ and set $E ( g ) = ( a ^ { \top } \psi ( g ) ) E _ { 0 }$ . This residual vanishes almost surely on the training distribution, but it is nonzero at $g _ { \star }$ □

A singular $M _ { \psi }$ is not automatically fatal. If $\psi ( g _ { \star } )$ lies in the covered range of $M _ { \psi }$ , the same type of statement can use the inverse restricted to that range. The failure above occurs when the test law has a component in a direction that training never observes.

Corollary H.8 (Transfer from a finite gravity design). Let $g _ { 1 } , \ldots , g _ { n }$ be observed gravity values and let $\Psi \in \mathbb { R } ^ { n \times m }$ have rows $\psi ( g _ { i } ) ^ { \intercal }$ . For a residual $\begin{array} { r } { E ( g ) = \sum _ { k } \psi _ { k } ( g ) E _ { k } } \end{array}$ , suppose Ψ has full column rank. Then

$$
\| E ( g _ { \star } ) \| _ { F } ^ { 2 } \leq \psi ( g _ { \star } ) ^ { \top } ( \Psi ^ { \top } \Psi ) ^ { - 1 } \psi ( g _ { \star } ) \sum _ { i = 1 } ^ { n } \| E ( g _ { i } ) \| _ { F } ^ { 2 } .\tag{58}
$$

Proof. Let $Y \in \mathbb { R } ^ { n \times q }$ have row vec $( E ( g _ { i } ) ) ^ { \top }$ , where $q$ is the number of entries in each residual matrix. If $\ b { D } \in \mathbb { R } ^ { m \times q }$ stacks the vectorized coeficient matrices, then $Y = \Psi D$ . Full column rank gives $D = ( \Psi ^ { \top } \Psi ) ^ { - 1 } \Psi ^ { \top } Y$ . Hence

$$
\operatorname { v e c } ( E ( g _ { \star } ) ) ^ { \top } = \psi ( g _ { \star } ) ^ { \top } ( \Psi ^ { \top } \Psi ) ^ { - 1 } \Psi ^ { \top } Y .\tag{59}
$$

Cauchy-Schwarz bounds its squared norm by the squared norm of the row vector times $\| Y \| _ { F } ^ { 2 }$ . The row-vector norm is $\psi ( g _ { \star } ) ^ { \top } ( \Psi ^ { \top } \Psi ) ^ { - 1 } \psi ( g _ { \star } )$ , which proves the claim. □

Theorem H.9 (Conditioning alone does not guarantee extrapolation). Let $g _ { 1 } , \ldots , g _ { n }$ be distinct observed gravity values and let $g _ { \star }$ be diferent from all of them. For any analytic matrix-valued function $\widehat { A } _ { 0 } ( g )$ and any matrix $Q _ { i }$ , there is an analytic matrix-valued function $\widehat { A } _ { 1 } ( g )$ such that

$$
\widehat { A } _ { 1 } ( g _ { i } ) = \widehat { A } _ { 0 } ( g _ { i } ) \quad f o r e v e r y i , \qquad \widehat { A } _ { 1 } ( g _ { \star } ) = \widehat { A } _ { 0 } ( g _ { \star } ) + Q .\tag{60}
$$

Proof. Define

$$
q ( g ) = { \frac { \prod _ { i = 1 } ^ { n } ( g - g _ { i } ) } { \prod _ { i = 1 } ^ { n } ( g _ { \star } - g _ { i } ) } } .\tag{61}
$$

The denominator is nonzero, $q ( g _ { i } ) = 0$ , and $q ( g _ { \star } ) = 1$ . The function $\widehat { A } _ { 1 } ( g ) = \widehat { A } _ { 0 } ( g ) + q ( g ) Q$ is analytic and has the required values. □

Even an analytic conditional model can agree at every observed gravity and difer arbitrarily at an unseen one. Supplying gravity is therefore not, by itself, an extrapolation guarantee.

If a residual has the approximate form

$$
E ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) E _ { k } + \rho _ { E } ( g ) ,\tag{62}
$$

write

$$
E _ { \mathrm { b a s i s } } ( g ) = \sum _ { k = 1 } ^ { m } \psi _ { k } ( g ) E _ { k } , \qquad \epsilon _ { E , \mathrm { b a s i s } } ^ { 2 } = \mathbb { E } _ { g \sim P _ { \mathrm { t r } } } \| E _ { \mathrm { b a s i s } } ( g ) \| _ { F } ^ { 2 } .\tag{63}
$$

Lemma H.3 applies to $E _ { \mathrm { b a s i s } }$ and gives

$$
\| E ( g _ { \star } ) \| _ { F } \leq \sqrt { \mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) } \epsilon _ { E , \mathrm { b a s i s } } + \| \rho _ { E } ( g _ { \star } ) \| _ { F } .\tag{64}
$$

Here $\epsilon _ { E , \mathrm { b a s i s } }$ is not automatically bounded by the observed training residual: the basis component and $\rho _ { E }$ may cancel at the training gravity values. For $E _ { A }$ , call the remainder $\rho _ { \mathrm { o p } } ;$ for $C _ { W }$ , call it $\rho _ { \mathrm { c l } }$ Applying the formula to both residuals adds $B _ { z } \| \rho _ { \mathrm { o p } } ( g _ { \star } ) \| _ { F } + B _ { \phi } \| \rho _ { \mathrm { c l } } ( g _ { \star } ) \| _ { F }$ to the right-hand side of Eq. 46. Training error cannot control a remainder that appears only in a region with little or no training mass.

## H.4 Local errors under repeated composition

Teacher forcing measures the local error before predictions are reused. Free rollout feeds each predicted latent back into the predictor, so later learned iterates act on every earlier defect. The following recursion makes this dependence exact.

At a fixed gravity $^ { g , }$ let the true and predicted latent transitions be

$$
z _ { j + 1 } = A _ { W } ( g ) z _ { j } + C _ { W } ( g ) \phi _ { j } + W \xi _ { j + 1 } ,\tag{65}
$$

$$
{ \widehat { z } } _ { j + 1 } = { \widehat { A } } ( g ) { \widehat { z } } _ { j } , \qquad { \widehat { z } } _ { 0 } = z _ { 0 } .\tag{66}
$$

Define the rollout error $e _ { j } = z _ { j } - { \widehat { z } } _ { j }$

Theorem H.10 (Exact recursive latent error). For every $j \geq 0$

$$
e _ { j + 1 } = { \widehat { A } } ( g ) e _ { j } + \delta _ { j } ( g ) + W \xi _ { j + 1 } .\tag{67}
$$

Consequently, for every ${ \mathfrak { h } } \geq 1$

$$
e _ { \mathfrak { h } } = \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \widehat { A } ( g ) ^ { \mathfrak { h } - 1 - j } \left[ \delta _ { j } ( g ) + W \xi _ { j + 1 } \right] .\tag{68}
$$

Proof. Subtract the predicted transition from the true transition:

$$
e _ { j + 1 } = A _ { W } ( g ) z _ { j } + C _ { W } ( g ) \phi _ { j } + W \xi _ { j + 1 } - \widehat { A } ( g ) \widehat { z } _ { j } .\tag{69}
$$

Add and subtract $\widehat { A } ( g ) z _ { j }$

$$
e _ { j + 1 } = \widehat { A } ( g ) ( z _ { j } - \widehat { z } _ { j } ) + [ A _ { W } ( g ) - \widehat { A } ( g ) ] z _ { j } + C _ { W } ( g ) \phi _ { j } + W \xi _ { j + 1 }\tag{70}
$$

$$
= \widehat { A } ( g ) e _ { j } + \delta _ { j } ( g ) + W \xi _ { j + 1 } ,\tag{71}
$$

which proves Eq. 67. Since $e _ { 0 } = 0$ , the first two steps are

$$
e _ { 1 } = \delta _ { 0 } + W \xi _ { 1 } ,\tag{72}
$$

$$
e _ { 2 } = \widehat { A } ( \delta _ { 0 } + W \xi _ { 1 } ) + \delta _ { 1 } + W \xi _ { 2 } .\tag{73}
$$

Repeating the substitution gives Eq. 68. The same conclusion also follows by induction on h.

Since $\widehat { A } ( g ) ^ { q } = \widehat { S } _ { g } ( q )$ , each one-step error is transformed by the remaining learned semigroup iterate. The terms are vectors, so they may reinforce or cancel one another. The identity does not imply that error, or the diference between two models, must grow monotonically with the horizon.

## H.4.1 Physical readout and the central bound

Latent error and physical error are not the same quantity. The readout can hide some latent errors, and the representation can omit physical information even when its latent rollout is accurate.

Proposition H.11 (Exact physical readout decomposition). Let $s _ { \mathfrak { h } } = C _ { \mathrm { p h y s } } \phi _ { \mathfrak { h } }$ and let r be any frozen probe. Define the true-latent readout residual

$$
b _ { r } ( \phi _ { \mathfrak { h } } ) = s _ { \mathfrak { h } } - r ( z _ { \mathfrak { h } } ) .\tag{74}
$$

Then

$$
s _ { \mathfrak { h } } - r ( { \widehat z } _ { \mathfrak { h } } ) = b _ { r } ( \phi _ { \mathfrak { h } } ) + r ( z _ { \mathfrak { h } } ) - r ( { \widehat z } _ { \mathfrak { h } } ) .\tag{75}
$$

The corresponding mean-squared error is

$$
\begin{array} { r } { \mathbb { E } \| s _ { \mathfrak { h } } - r ( \widehat { z } _ { \mathfrak { h } } ) \| _ { 2 } ^ { 2 } = \mathbb { E } \| b _ { r } ( \phi _ { \mathfrak { h } } ) \| _ { 2 } ^ { 2 } + \mathbb { E } \| r ( z _ { \mathfrak { h } } ) - r ( \widehat { z } _ { \mathfrak { h } } ) \| _ { 2 } ^ { 2 } } \\ { + 2 \mathbb { E } \Big [ b _ { r } ( \phi _ { \mathfrak { h } } ) ^ { \top } \{ r ( z _ { \mathfrak { h } } ) - r ( \widehat { z } _ { \mathfrak { h } } ) \} \Big ] . } \end{array}\tag{76}
$$

For the linear probe $r ( z ) = R z$

$$
b _ { r } ( \phi _ { \mathfrak { h } } ) = ( C _ { \mathrm { p h y s } } - R W ) \phi _ { \mathfrak { h } } , \qquad r ( z _ { \mathfrak { h } } ) - r ( \widehat z _ { \mathfrak { h } } ) = R e _ { \mathfrak { h } } .\tag{77}
$$

Proof. Add and subtract $r ( z _ { \mathfrak h } )$

$$
s _ { \mathfrak { h } } - r ( \widehat z _ { \mathfrak { h } } ) = s _ { \mathfrak { h } } - r ( z _ { \mathfrak { h } } ) + r ( z _ { \mathfrak { h } } ) - r ( \widehat z _ { \mathfrak { h } } ) .\tag{78}
$$

Expanding the squared Euclidean norm and taking expectations gives Eq. 76. Substituting $r ( z ) = R z$ gives Eq. 77. □

For the remainder of this subsection, specialize to $r ( z ) = R z$ and write $\widehat { s } _ { \mathfrak { h } } = R \widehat { z } _ { \mathfrak { h } }$ . For ${ \mathfrak { h } } \geq 1$ ， define the physical rollout gain

$$
\Gamma _ { \mathfrak { h } } ( g ; R , { \widehat { A } } ) = \sum _ { q = 0 } ^ { \mathfrak { h } - 1 } \lVert R { \widehat { A } } ( g ) ^ { q } \rVert _ { 2 } .\tag{79}
$$

This quantity bounds how strongly the learned dynamics and readout can propagate a sequence of local latent errors.

Theorem H.12 (Pathwise physical error under an unseen gravity). Assume the shared law basis in $E q .$ 35. Suppose $\| z _ { j } \| _ { 2 } \le B _ { z }$ and $\| \phi _ { j } \| _ { 2 } \le B _ { \phi }$ along the test trajectory at $g _ { \star }$ . Then

$$
\begin{array} { r l } & { \| s _ { \mathfrak h } - \widehat s _ { \mathfrak h } \| _ { 2 } \le \| ( C _ { \mathrm { p h y s } } - R W ) \phi _ { \mathfrak h } \| _ { 2 } } \\ & { \qquad + \Gamma _ { \mathfrak h } ( g _ { \star } ; R , \widehat A ) \sqrt { { \mathscr L } _ { \mathrm { l a w } } ( g _ { \star } ) } \left( B _ { z } \epsilon _ { \mathrm { o p } } + B _ { \phi } \epsilon _ { \mathrm { c l } } \right) } \\ & { \qquad + \displaystyle \sum _ { j = 0 } ^ { \mathfrak h - 1 } \left\| R \widehat A ( g _ { \star } ) ^ { \mathfrak h - 1 - j } W \xi _ { j + 1 } \right\| _ { 2 } . } \end{array}\tag{80}
$$

Proof. Proposition H.11 and Theorem H.10 give

$$
s _ { \mathfrak { h } } - { \widehat { s } } _ { \mathfrak { h } } = ( C _ { \mathrm { p h y s } } - R W ) \phi _ { \mathfrak { h } }\tag{81}
$$

$$
+ \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } R \widehat { A } ( g _ { \star } ) ^ { \mathfrak { h } - 1 - j } \delta _ { j } ( g _ { \star } )\tag{82}
$$

$$
+ \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } R \widehat { A } ( g _ { \star } ) ^ { \mathfrak { h } - 1 - j } W \xi _ { j + 1 } .\tag{83}
$$

Apply the triangle inequality. For the conditional-mean defect terms, Theorem H.4 gives the uniform bound

$$
\| \delta _ { j } ( g _ { \star } ) \| _ { 2 } \leq \sqrt { { \mathcal L } _ { \mathrm { l a w } } ( g _ { \star } ) } \left( B _ { z } \epsilon _ { \mathrm { o p } } + B _ { \phi } \epsilon _ { \mathrm { c l } } \right) .\tag{84}
$$

Therefore

$$
\sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \left. R \widehat { A } ( g _ { \star } ) ^ { \mathfrak { h } - 1 - j } \delta _ { j } ( g _ { \star } ) \right. _ { 2 }\tag{85}
$$

$$
\leq \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \lVert R \widehat { A } ( g _ { \star } ) ^ { \mathfrak { h } - 1 - j } \rVert _ { 2 } \lVert \delta _ { j } ( g _ { \star } ) \rVert _ { 2 }\tag{86}
$$

$$
\leq \Gamma _ { \mathfrak { h } } ( g _ { \star } ; R , \widehat { A } ) \sqrt { \mathcal { L } _ { \mathrm { l a w } } ( g _ { \star } ) } \left( B _ { z } \epsilon _ { \mathrm { o p } } + B _ { \phi } \epsilon _ { \mathrm { c l } } \right) .\tag{87}
$$

Keeping the transition-residual terms explicit proves Eq. 80.

The bound separates the four quantities that afect physical rollout quality: the readout residual, the local unseen-law error, its recursive propagation, and the transition residuals. It is an upper bound, not an additive model of the observed error. The factor $\Gamma _ { \mathfrak { h } }$ is a worst-case bound and does not describe cancellation between propagated residuals. Because the readout residual, trajectory bounds, rollout gain, and realized transition residuals remain in the statement, this is not a stand-alone generalization guarantee based only on the law-coverage factor.

## H.4.2 Cross-step second moments

The orientation and cross-step dependence of the one-step errors also matter. Let

$$
\varepsilon _ { j } = \delta _ { j } ( g ) + W \xi _ { j + 1 } , \qquad S _ { i j } ( g ) = \mathbb { E } [ \varepsilon _ { i } \varepsilon _ { j } ^ { \top } \mid g ] .\tag{88}
$$

From Eq. 68, with $\widehat { A } = \widehat { A } ( g )$

$$
\mathbb { E } [ \Vert R e _ { \mathfrak { h } } \Vert _ { 2 } ^ { 2 } \ | \ g ] = \sum _ { i = 0 } ^ { \mathfrak { h } - 1 } \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \mathrm { t r } \Big ( R \widehat { A } ^ { \mathfrak { h } - 1 - i } S _ { i j } ( g ) ( \widehat { A } ^ { \mathfrak { h } - 1 - j } ) ^ { \top } R ^ { \top } \Big ) .\tag{89}
$$

To derive this identity, write $\begin{array} { r } { R e _ { \mathfrak { h } } = \sum _ { i } R \widehat { A } ^ { \mathfrak { h } - 1 - j } \varepsilon _ { j } } \end{array}$ , expand the outer product, and take the trace. No zero-mean assumption is needed because $S _ { i j } ( g )$ is a second moment rather than a covariance. Its of-diagonal entries record cross-step dependence. A reset comparison changes these second moments as well as the propagation path, so it does not isolate a single spectral gain.

The same idea does not depend on linearity. If the learned nonlinear transition $\widehat { F } _ { g }$ is $L _ { g ^ { - } }$ Lipschitz on a set containing the true and predicted trajectories, and $d _ { j } = \| \widehat { F } _ { g } ( x _ { j } ) - F _ { g } ( x _ { j } ) \| _ { 2 }$ , then

$$
\| \widehat { x } _ { \mathfrak { h } } - x _ { \mathfrak { h } } \| _ { 2 } \leq \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } L _ { g } ^ { \mathfrak { h } - 1 - j } d _ { j } .\tag{90}
$$

Indeed, one step gives $\| \widehat { x } _ { j + 1 } - x _ { j + 1 } \| _ { 2 } \leq L _ { g } \| \widehat { x } _ { j } - x _ { j } \| _ { 2 } + d _ { j }$ , and iteration from a common initial state proves the claim. This nonlinear bound supplies intuition, but the exact linear recursion is the result used above.

## H.5 Why multi-step composition can change the learned representation

The rollout horizon changes the geometry used to compare representations, even in a stable linear system. This section isolates the conditional-mean, action-free system and omits residual variation. The quadratic costs below therefore measure transition bias rather than irreducible noise. The finite-history version with actions appears in Section H.6.

Fix one smooth regime, suppress g, and let $\widehat { S } _ { A } ( \mathfrak { h } ) = A ^ { \mathfrak { h } }$ for a candidate latent operator A. Define the finite-horizon semigroup-intertwining defect

$$
D _ { \mathfrak { h } } ( W ; A ) = W S _ { g } ( \mathfrak { h } ) - \widehat { S } _ { A } ( \mathfrak { h } ) W = W T ^ { \mathfrak { h } } - A ^ { \mathfrak { h } } W , \qquad D _ { 1 } = W T - A W .\tag{91}
$$

When $A = \widehat { A } ( g ) , D _ { 1 } \phi _ { t } = \delta _ { t } ( g )$ . When $A = A _ { W } ( g ) , D _ { 1 } = C _ { W } ( g )$ is the representation closure residual. A general fitted A contains both closure and predictor error.

Theorem H.13 (Finite-horizon intertwining-defect geometry). For every ${ \mathfrak { h } } \geq 1$

$$
D _ { \mathfrak { h } } = \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } A ^ { \mathfrak { h } - 1 - j } D _ { 1 } T ^ { j } .\tag{92}
$$

Let

$$
M _ { \mathfrak { h } } ( A , T ) = \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } ( T ^ { j } ) ^ { \top } \otimes A ^ { \mathfrak { h } - 1 - j } .\tag{93}
$$

For the feature second-moment matrix $\Sigma _ { \phi } = \mathbb { E } [ \phi \phi ^ { \top } ] \succeq 0$ and nonnegative weights $w _ { 1 } , \ldots , w _ { K }$

$$
\sum _ { k = 1 } ^ { K } w _ { k } \operatorname { t r } ( D _ { k } \Sigma _ { \phi } D _ { k } ^ { \top } ) = \operatorname { v e c } ( D _ { 1 } ) ^ { \top } G _ { K } \operatorname { v e c } ( D _ { 1 } ) ,\tag{94}
$$

where

$$
G _ { K } = \sum _ { k = 1 } ^ { K } w _ { k } M _ { k } ^ { \top } ( \Sigma _ { \phi } \otimes I _ { d _ { z } } ) M _ { k } \succeq 0 .\tag{95}
$$

If $w _ { 1 } > 0$ and $\Sigma _ { \phi } \succ 0$ , then $G _ { K } \succ 0$

Proof. For ${ \mathfrak { h } } = 1$ , Eq. 92 is the definition of $D _ { 1 }$ . Suppose the identity holds at h. Then

$$
D _ { \mathfrak { h } + 1 } = W T ^ { \mathfrak { h } + 1 } - A ^ { \mathfrak { h } + 1 } W\tag{96}
$$

$$
= ( W T ^ { \flat } - A ^ { \flat } W ) T + A ^ { \flat } ( W T - A W )\tag{97}
$$

$$
= D _ { \mathfrak { h } } T + A ^ { \mathfrak { h } } D _ { 1 }\tag{98}
$$

$$
= \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } A ^ { \mathfrak { h } - 1 - j } D _ { 1 } T ^ { j + 1 } + A ^ { \mathfrak { h } } D _ { 1 }\tag{99}
$$

$$
= \sum _ { j = 0 } ^ { \mathfrak { h } } A ^ { \mathfrak { h } - j } D _ { 1 } T ^ { j } .\tag{100}
$$

Thus the telescoping identity holds by induction.

Using vec $( L X R ) = ( R ^ { \top } \otimes L )$ vec(X) in each term of Eq. 92 gives

$$
\mathrm { v e c } ( D _ { \mathfrak { h } } ) = M _ { \mathfrak { h } } \mathrm { v e c } ( D _ { 1 } ) .\tag{101}
$$

Also,

$$
\begin{array} { r } { \mathrm { t r } ( D _ { \mathfrak { h } } \Sigma _ { \phi } D _ { \mathfrak { h } } ^ { \top } ) = \mathrm { v e c } ( D _ { \mathfrak { h } } ) ^ { \top } ( \Sigma _ { \phi } \otimes I _ { d _ { z } } ) \mathrm { v e c } ( D _ { \mathfrak { h } } ) . } \end{array}\tag{102}
$$

Substitution and summation give Eq.s 94 and 95. Every summand in $G _ { K }$ is positive semidefinite. $\mathrm { A t } \ \mathfrak { h } = 1 , M _ { 1 } = I$ , so the first summand is $w _ { 1 } \big ( \Sigma _ { \phi } \otimes I _ { d _ { z } } \big )$ , which is positive definite when $w _ { 1 } > 0$ and $\Sigma _ { \phi } \succ 0$ □

The telescoping identity shows how a one-step failure of intertwining enters later compositions. If $D _ { 1 } = 0$ , then $D _ { \mathfrak { h } } = 0$ at every horizon. If $D _ { 1 } \neq 0$ , the true and learned transition geometry determines whether the propagated terms reinforce or cancel. For fixed A, T, and $\Sigma _ { \phi } , G _ { K }$ weights these one-step defect directions by their later consequences. Two defects with the same one-step size can therefore have diferent multi-step costs. If fitting a new representation also changes $A _ { i }$ one common $G _ { K }$ need not rank all representations.

Theorem H.14 (When training and evaluation weight defects diferently). Let $G _ { \mathrm { t r } } \succ 0$ and $G _ { \mathrm { e v } } \succeq 0$ be the defect-weighting matrices for a training horizon profile and an evaluation profile on the same defect space. Set

$$
Q = G _ { \mathrm { t r } } ^ { - 1 / 2 } G _ { \mathrm { e v } } G _ { \mathrm { t r } } ^ { - 1 / 2 } .\tag{103}
$$

Then:

(a) $I f G _ { \mathrm { e v } } = c G _ { \mathrm { t r } }$ for some $c \geq 0$ , every defect has evaluation cost c times its training cost.

(b) If Q is not a scalar multiple of the identity, there are two defects with equal training cost and diferent evaluation cost.

(c) Let S be a nonzero subspace of the whitened coordinate y, let $P _ { S }$ be its orthogonal projector, and define the compressed operator $Q _ { S } = P _ { S } Q | _ { S } . \ I f Q _ { S } \succ 0$ , the largest ratio between evaluation costs of equal-training-cost defects whose whitened coordinates lie in S is $\lambda _ { \operatorname* { m a x } } ( Q _ { S } ) / \lambda _ { \operatorname* { m i n } } ( Q _ { S } )$

Proof. For a vectorized defect d, let $y = G _ { \mathrm { t r } } ^ { 1 / 2 } d .$ . Its training cost is

$$
d ^ { \top } G _ { \mathrm { t r } } d = y ^ { \top } y ,\tag{104}
$$

and its evaluation cost is

$$
d ^ { \top } G _ { \mathrm { e v } } d = y ^ { \top } Q y .\tag{105}
$$

If $G _ { \mathrm { e v } } = c G _ { \mathrm { t r } } .$ , then $Q = c I$ , proving part (a). If Q is not scalar, choose unit eigenvectors $y _ { 1 } , y _ { 2 }$ with distinct eigenvalues. They have the same training cost but diferent evaluation costs, proving part (b). For $y \in S$ , the evaluation cost is $y ^ { \top } Q _ { S } y$ . The Rayleigh quotient of $Q _ { S }$ ranges between its smallest and largest eigenvalues. Choosing the corresponding unit eigenvectors in $S$ attains the ratio in part (c). If a subspace is specified in the original defect coordinate $d ,$ its image under $G _ { \mathrm { t r } } ^ { 1 / 2 }$ is the corresponding subspace S in whitened coordinates. □

## H.5.1 A stable ranking reversal

Explicit representations can make one-step and two-step profiling prefer opposite choices. The construction below is strictly stable, so the reversal does not rely on an unstable transition.

Theorem H.15 (One-step preference can reverse under rollout). Let $\phi _ { t } \in \mathbb { R } ^ { 4 }$ have identity second moment and

$$
T = \mathrm { d i a g } \left( \frac { 2 } { 5 } , \frac { 4 } { 5 } , \frac { 3 } { 2 0 } , \frac { 3 } { 5 } \right) .\tag{106}
$$

Restrict the rank-one representation to

$$
W _ { P } = \frac { 1 } { \sqrt { 2 } } ( 1 , 1 , 0 , 0 ) , \qquad W _ { F } = \frac { 1 } { \sqrt { 2 } } ( 0 , 0 , 1 , 1 ) ,\tag{107}
$$

and let the latent predictor be the scalar $A = a$ . Then:

(a) the profiled one-step loss prefers $W _ { P }$

(b) using each representation’s one-step-optimal predictor, the horizon-two loss prefers $W _ { F _ { . } }$ ;

(c) for the normalized objective

$$
\frac { \| D _ { 1 } \| _ { F } ^ { 2 } + \gamma \| D _ { 2 } \| _ { F } ^ { 2 } } { 1 + \gamma } ,\tag{108}
$$

full profiling prefers $W _ { F }$ whenever the suficient condition $\gamma > 3 6 0 0 / 9 0 0 7 \approx 0 . 3 9 9 6 9$ , including $\gamma = 0 . 9 5$

Proof. For a representation that averages two eigenmodes λ and $\mu ,$

$$
\| D _ { \mathfrak { h } } ( W ; a ) \| _ { F } ^ { 2 } = \frac { 1 } { 2 } \left[ ( \lambda ^ { \mathfrak { h } } - a ^ { \mathfrak { h } } ) ^ { 2 } + ( \mu ^ { \mathfrak { h } } - a ^ { \mathfrak { h } } ) ^ { 2 } \right] .\tag{109}
$$

At one step, diferentiating with respect to a gives the minimizer $a = ( \lambda + \mu ) / 2$ and minimum $( \lambda - \mu ) ^ { 2 } / 4$ . Therefore

$$
\operatorname* { m i n } _ { a } \| D _ { 1 } ( W _ { P } ; a ) \| _ { F } ^ { 2 } = \frac { ( \frac { 4 } { 5 } - \frac { 2 } { 5 } ) ^ { 2 } } { 4 } = \frac { 1 } { 2 5 } ,\tag{110}
$$

$$
\operatorname* { m i n } _ { a } \lVert D _ { 1 } ( W _ { F } ; a ) \rVert _ { F } ^ { 2 } = \frac { ( \frac { 3 } { 5 } - \frac { 3 } { 2 0 } ) ^ { 2 } } { 4 } = \frac { 8 1 } { 1 6 0 0 } .\tag{111}
$$

Since $1 / 2 5 < 8 1 / 1 6 0 0$ , one-step profiling prefers $W _ { P }$

The one-step-optimal predictors are $a _ { P } = 3 / 5$ and $a _ { F } = 3 / 8$ . At horizon two, Equation equation 109 gives

$$
| | D _ { 2 } ( W _ { P } ; a _ { P } ) | | _ { F } ^ { 2 } = \frac { 1 } { 2 } \left[ \left( \frac { 4 } { 2 5 } - \frac { 9 } { 2 5 } \right) ^ { 2 } + \left( \frac { 1 6 } { 2 5 } - \frac { 9 } { 2 5 } \right) ^ { 2 } \right] = \frac { 3 7 } { 6 2 5 } ,\tag{112}
$$

$$
\| D _ { 2 } ( W _ { F } ; a _ { F } ) \| _ { F } ^ { 2 } = \frac { 1 } { 2 } \left[ \left( \frac { 9 } { 4 0 0 } - \frac { 9 } { 6 4 } \right) ^ { 2 } + \left( \frac { 9 } { 2 5 } - \frac { 9 } { 6 4 } \right) ^ { 2 } \right]\tag{113}
$$

$$
= { \frac { 1 } { 2 } } \left[ \left( - { \frac { 1 8 9 } { 1 6 0 0 } } \right) ^ { 2 } + \left( { \frac { 3 5 1 } { 1 6 0 0 } } \right) ^ { 2 } \right] = { \frac { 7 9 4 6 1 } { 2 5 6 0 0 0 0 } } .\tag{114}
$$

Here 79461 $/ 2 5 6 0 0 0 0 < 3 7 / 6 2 5$ , which proves the horizon-two reversal.

For the joint two-step objective, allow the horizon-one and horizon-two scalar predictions for $W _ { P }$ to vary independently. This relaxation can only lower its profiled cost, and it gives the lower bound

$$
\operatorname* { i n f } _ { a } \frac { \| D _ { 1 } ( W _ { P } ; a ) \| _ { F } ^ { 2 } + \gamma \| D _ { 2 } ( W _ { P } ; a ) \| _ { F } ^ { 2 } } { 1 + \gamma } \ge \frac { \frac { 1 } { 2 5 } + \gamma \frac { 3 6 } { 6 2 5 } } { 1 + \gamma } .\tag{115}
$$

For $W _ { F }$ , evaluate the admissible predictor $a = 2 / 5$ . Its one-step and two-step costs are

$$
\| D _ { 1 } ( W _ { F } ; 2 / 5 ) \| _ { F } ^ { 2 } = { \frac { 4 1 } { 8 0 0 } } ,\tag{116}
$$

$$
\| D _ { 2 } ( W _ { F } ; 2 / 5 ) \| _ { F } ^ { 2 } = \frac { 3 7 7 } { 1 2 8 0 0 } .\tag{117}
$$

Hence its profiled cost is at most

$$
{ \frac { { \frac { 4 1 } { 8 0 0 } } + \gamma { \frac { 3 7 7 } { 1 2 8 0 0 } } } { 1 + \gamma } } .\tag{118}
$$

The upper bound in Eq. 118 is smaller than the lower bound in $\operatorname { E q }$ . 115 when

$$
\gamma \left( { \frac { 3 6 } { 6 2 5 } } - { \frac { 3 7 7 } { 1 2 8 0 0 } } \right) > { \frac { 4 1 } { 8 0 0 } } - { \frac { 1 } { 2 5 } } ,\tag{119}
$$

$$
{ \gamma } \frac { 9 0 0 7 } { 3 2 0 0 0 0 } > \frac { 3 6 0 0 } { 3 2 0 0 0 0 } ,\tag{120}
$$

$$
\gamma > \frac { 3 6 0 0 } { 9 0 0 7 } .\tag{121}
$$

In that range, even the displayed upper bound for $W _ { F }$ is below a lower bound for $W _ { P }$ , so full profiling must prefer $W _ { F }$ □

This construction proves that the objective can change representation preference as the rollout horizon changes. It does not prove that a longer rollout always helps, that a particular predictor architecture causes the change, or that this four-mode system is the mechanism learned by SG-JEPA.

## H.6 Relating the theory to the history-based predictor

The preceding sections used a memoryless predictor to keep the mechanism visible. SG-JEPA instead reads a finite latent history together with actions and gravity. A companion state turns this interface into a one-step system on an augmented space, so the same algebra applies without treating the GRU as a literal matrix.

For context length $H ,$ , define the feature-history state and its encoded form

$$
\begin{array} { r } { \boldsymbol { X } _ { t } ^ { ( H ) } = \left[ \begin{array} { c } { \phi _ { t - H + 1 } } \\ { \vdots } \\ { \phi _ { t } } \end{array} \right] \in \mathbb { R } ^ { H d _ { \phi } } , \qquad \mathbb { W } = I _ { H } \otimes \boldsymbol { W } , \qquad \boldsymbol { Z } _ { t } ^ { ( H ) } = \mathbb { W } \boldsymbol { X } _ { t } ^ { ( H ) } . } \end{array}\tag{122}
$$

Let $a _ { t } \in \mathbb { R } ^ { d _ { a } }$ contain every known exogenous action input needed for the transition from t to $t + 1$ If the interface uses an action window, $a _ { t }$ denotes that fixed-dimensional stacked input. The true and predicted rollouts receive the same action sequence. Gravity remains fixed during the rollout and explicit through the law-indexed operators. Closed-loop action diferences would introduce additional terms and are outside this prediction analysis.

For the nonlinear predictor, let $\widehat { \Phi } _ { \theta , g } ^ { ( k ) } \left( Z _ { t } ^ { ( H ) } ; a _ { t : t + k - 1 } \right)$ denote the predicted companion state obtained after the consecutive action block. Recursive application gives the control-sequence composition law

$$
\begin{array} { r l } & { \widehat { \Phi } _ { \theta , g } ^ { ( k + \ell ) } \left( Z _ { t } ^ { ( H ) } ; a _ { t : t + k + \ell - 1 } \right) } \\ & { \quad = \widehat { \Phi } _ { \theta , g } ^ { ( \ell ) } \left( \widehat { \Phi } _ { \theta , g } ^ { ( k ) } \left( Z _ { t } ^ { ( H ) } ; a _ { t : t + k - 1 } \right) ; a _ { t + k : t + k + \ell - 1 } \right) . } \end{array}\tag{123}
$$

The predictor parameters are shared, but each numerical transition is indexed by its action input. In the action-free case, the companion update generates an ordinary discrete semigroup. With known actions, it obeys the controlled composition law above rather than an autonomous semigroup on an individual latent state.

Consider the linear companion systems

$$
X _ { t + 1 } ^ { ( H ) } =  { \mathbb { T } } ( g ) X _ { t } ^ { ( H ) } +  { \mathbb { B } } ( g ) a _ { t } +  { \mathbb { J } } _ { \xi } \xi _ { t + 1 } ,\tag{124}
$$

$$
\widehat { Z } _ { t + 1 } ^ { ( H ) } = \mathbb { A } ( g ) \widehat { Z } _ { t } ^ { ( H ) } + \mathbb { D } ( g ) a _ { t } .\tag{125}
$$

The first $H - 1$ block rows of $\mathbb { T } ( g )$ and $\mathbb { A } ( g )$ shift the history. The final block row performs the true or learned update. The matrix $\mathbb { J } _ { \xi } \in \mathbb { R } ^ { H d _ { \phi } \times d _ { \phi } }$ inserts the new residual in the last feature block. The remaining dimensions are

$$
\mathbb { T } ( g ) \in \mathbb { R } ^ { H d _ { \phi } \times H d _ { \phi } } ,
$$

$$
\mathbb { B } ( g ) \in \mathbb { R } ^ { H d _ { \phi } \times d _ { a } } ,\tag{126}
$$

$$
\mathbb { A } ( g ) \in \mathbb { R } ^ { H d _ { z } \times H d _ { z } } ,
$$

$$
\mathbb { D } ( g ) \in \mathbb { R } ^ { H d _ { z } \times d _ { a } } .\tag{127}
$$

The additive maps $\mathbb { B } ( g )$ and $\mathbb { D } ( g )$ are assumptions of this model. Nonlinear interactions among state, action, and gravity in the implemented predictor fall outside this additive model. For $1 \leq L \leq H$ , let

$$
\begin{array} { r l } { P _ { H , L } = \bigg [ 0 _ { L d _ { z } \times ( H - L ) d _ { z } } } & { { } I _ { L d _ { z } } \bigg ] \in \mathbb { R } ^ { L d _ { z } \times H d _ { z } } } \end{array}\tag{128}
$$

select the newest L latent blocks, and write $P _ { H } = P _ { H , 1 }$ . At the start of rollout, both systems use the same true context:

$$
\widehat { Z } _ { t } ^ { ( H ) } = \mathbb { W } X _ { t } ^ { ( H ) } .\tag{129}
$$

For ${ \mathfrak { h } } \geq 1$ and $0 \leq j < \mathfrak { h }$ , define

$$
\mathfrak { D } _ { \mathfrak { h } } ( g ) = P _ { H } \left[ \mathbb { W } \mathbb { T } ( g ) ^ { \mathfrak { h } } - \mathbb { A } ( g ) ^ { \mathfrak { h } } \mathbb { W } \right] ,\tag{130}
$$

$$
\mathfrak { F } _ { \mathfrak { h } , j } ( g ) = P _ { H } \left[ \mathbb { W } \mathbb { T } ( g ) ^ { \mathfrak { h } - 1 - j } \mathbb { B } ( g ) - \mathbb { A } ( g ) ^ { \mathfrak { h } - 1 - j } \mathbb { D } ( g ) \right] ,\tag{131}
$$

$$
\begin{array} { r } { \mathfrak { N } _ { \mathfrak { h } , j } ( g ) = P _ { H } \mathbb { W } \mathbb { T } ( g ) ^ { \mathfrak { h } - 1 - j } \mathbb { J } _ { \xi } . } \end{array}\tag{132}
$$

Theorem H.16 (Exact finite-history rollout error). Under Eq.s 124 through 129, the newest latent error at horizon h is

$$
z _ { t + \mathfrak { h } } - \widehat { z } _ { t + \mathfrak { h } } = \mathfrak { D } _ { \mathfrak { h } } ( g ) X _ { t } ^ { ( H ) } + \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \mathfrak { F } _ { \mathfrak { h } , j } ( g ) a _ { t + j } + \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \mathfrak { N } _ { \mathfrak { h } , j } ( g ) \xi _ { t + j + 1 } .\tag{133}
$$

If $\cdot V _ { t , \mathfrak { h } }$ stacks $X _ { t } ^ { ( H ) }$ , the h future actions, and the h transition residuals, and ${ \mathfrak { M } } _ { \mathfrak { h } } ( g )$ is the corresponding block row in Equation equation 133, then

$$
\begin{array} { r } { \mathbb { E } \big [ \| z _ { t + \mathfrak { h } } - \widehat { z } _ { t + \mathfrak { h } } \| _ { 2 } ^ { 2 } \mid g \big ] = \mathrm { t r } \Big ( \mathfrak { M } _ { \mathfrak { h } } ( g ) \mathbb { E } [ V _ { t , \mathfrak { h } } V _ { t , \mathfrak { h } } ^ { \top } \mid g ] \mathfrak { M } _ { \mathfrak { h } } ( g ) ^ { \top } \Big ) . } \end{array}\tag{134}
$$

No centering or independence assumption is required.

Proof. Repeated substitution in the true companion system gives

$$
X _ { t + \mathfrak { h } } ^ { ( H ) } = \mathbb { T } ( g ) ^ { \mathfrak { h } } X _ { t } ^ { ( H ) }\tag{135}
$$

$$
+ \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \mathbb { T } ( g ) ^ { \mathfrak { h } - 1 - j } \left[ \mathbb { B } ( g ) a _ { t + j } + \mathbb { J } _ { \xi } \xi _ { t + j + 1 } \right] .\tag{136}
$$

The exact initial context and recursive predictor give

$$
\widehat { Z } _ { t + \mathfrak { h } } ^ { ( H ) } = \mathbb { A } ( g ) ^ { \mathfrak { h } } \mathbb { W } X _ { t } ^ { ( H ) }\tag{137}
$$

$$
+ \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } \mathbb { A } ( g ) ^ { \mathfrak { h } - 1 - j } \mathbb { D } ( g ) a _ { t + j } .\tag{138}
$$

The true newest latent is $P _ { H } \mathbb { W } X _ { t + \mathfrak { h } } ^ { ( H ) }$ , and the predicted newest latent is $P _ { H } \widehat { Z } _ { t + \mathfrak { h } } ^ { ( H ) }$ . Apply these selectors to Eq.s 136 and 138, subtract, and group the context, action, and transition-residual terms. Their coeficients are exactly ${ \mathfrak { D } } _ { \mathfrak { h } } , { \mathfrak { F } } _ { \mathfrak { h } , j }$ , and $\mathfrak { N } _ { \mathfrak { h } , j }$

For the second statement, write the error as ${ \mathfrak { M } } _ { \mathfrak { h } } ( g ) V _ { t , \mathfrak { h } }$ . Then

$$
\mathbb { E } \Vert \mathfrak { M } _ { \mathfrak { h } } V _ { t , \mathfrak { h } } \Vert _ { 2 } ^ { 2 } = \mathbb { E } \operatorname { t r } \left( \mathfrak { M } _ { \mathfrak { h } } V _ { t , \mathfrak { h } } V _ { t , \mathfrak { h } } ^ { \top } \mathfrak { M } _ { \mathfrak { h } } ^ { \top } \right)\tag{139}
$$

$$
= \mathrm { t r } \Big ( \mathfrak { M } _ { \mathfrak { h } } \mathbb { E } [ V _ { t , { \mathfrak { h } } } V _ { t , { \mathfrak { h } } } ^ { \top } ] \mathfrak { M } _ { \mathfrak { h } } ^ { \top } \Big ) .\tag{140}
$$

Conditioning throughout on g proves Eq. 134.

Theorem H.16 is stated for the newest latent to match the rollout loss. Replacing $P _ { H }$ P by $P _ { H , L }$ in Eq.s 130–132 gives the same exact identity for the stacked window $z _ { t + \mathfrak { h } - L + 1 : t + \mathfrak { h } } - \widehat { z } _ { t + \mathfrak { h } - L + 1 : t + \mathfrak { h } }$ . A linear probe on that window is then represented by a matrix multiplying $P _ { H , L }$

Corollary H.17 (Memoryless results on the companion state). Suppose the augmented true and learned operators share the law basis used in Section H.3. Then the local-defect, law-transfer, and recursive-error results apply on the companion state after the substitutions

$$
W \mapsto \mathbb { W } , \qquad T ( g ) \mapsto \mathbb { T } ( g ) , \qquad { \widehat { A } } ( g ) \mapsto \mathbb { A } ( g ) ,\tag{141}
$$

followed $b y$ the selector $P _ { H }$ . For the horizon-geometry result, define

$$
\begin{array} { r l } & { \mathbb { D } _ { \mathfrak { h } } = \mathbb { W } \mathbb { T } ^ { \mathfrak { h } } - \mathbb { A } ^ { \mathfrak { h } } \mathbb { W } , } \\ & { \mathbb { M } _ { \mathfrak { h } } = \displaystyle \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } ( \mathbb { T } ^ { j } ) ^ { \top } \otimes \mathbb { A } ^ { \mathfrak { h } - 1 - j } , \qquad \Sigma _ { X } = \mathbb { E } [ X _ { t } ^ { ( H ) } X _ { t } ^ { ( H ) \top } ] . } \end{array}\tag{142}
$$

The quadratic form for newest-block loss uses

$$
G _ { K } ^ { ( P ) } = \sum _ { k = 1 } ^ { K } w _ { k } \mathbb { M } _ { k } ^ { \top } \left( \Sigma _ { X } \otimes P _ { H } ^ { \top } P _ { H } \right) \mathbb { M } _ { k } \succeq 0 .\tag{143}
$$

For a linear L-block history probe, replace $P _ { H }$ by $P _ { H , L }$ in this quadratic form and compose the result with the probe matrix. This matrix need not be positive definite on every augmented defect because $P _ { H }$ ignores the older output blocks. If $w _ { 1 } > 0$ and $\Sigma _ { X } \succ 0$ , it is positive definite after restriction to the admissible companion-defect space

$$
S _ { \mathrm { a d m } } = \left\{ \mathrm { v e c } \left( \left[ 0 _ { \left( H - 1 \right) d _ { z } \times H d _ { \phi } } \right] \right) : D \in \mathbb { R } ^ { d _ { z } \times H d _ { \phi } } \right\} .\tag{144}
$$

The zero rows are the matched history shifts. If the true and learned action maps $d i f f e r ,$ the action residuals $\mathfrak { F } _ { \mathfrak { h } , j } ( g ) a _ { t + j }$ remain as the explicit extra terms in Eq. 133. The shared basis must include an intercept to represent the gravity-independent shift rows of the companion matrices.

Proof. Eq.s 124 and 125 have the same algebraic form as the memoryless systems on their augmented spaces. Applying the earlier results therefore produces augmented latent errors. Multiplication by $P _ { H }$ selects the output that the model evaluates. The action terms vanish only when their true and learned augmented maps agree; otherwise the exact diference is $\mathfrak { F } _ { \mathfrak { h } , j } ( g ) a _ { t + j }$ . For horizon geometry, vec $( { \mathbb D } _ { \mathfrak { h } } ) = { \mathbb M } _ { \mathfrak { h } } \operatorname { v e c } ( { \mathbb D } _ { 1 } )$ . Selecting the newest block changes the output metric from the identity to $P _ { H } ^ { \top } P _ { H }$ , which gives Eq. 143. Its ${ \mathfrak { h } } = 1$ term is strictly positive for every nonzero defect in $\boldsymbol { S } _ { \mathrm { a d m } }$ when $w _ { 1 } > 0$ and $\Sigma _ { X } \succ 0 \colon P _ { H }$ retains its only potentially nonzero block row, and $\Sigma _ { X }$ is positive definite. □

The companion construction is an exact bridge for the history-state model. It does not assert that the nonlinear GRU is globally linear. It states what the model must contain to represent history, known actions, gravity, and recursive feedback without dropping terms.

## H.7 Regime changes and contact

A time-homogeneous hybrid map can still generate a discrete semigroup on the full state. Contact changes the active local branch, not the underlying update rule, so the branch matrices may vary along a trajectory. The smooth law-transfer analysis applies within one regime, while the telescoping identity below applies along a fixed, shared branch sequence. It does not cover errors that change the event time or send the prediction onto a diferent branch. The later hinge result concerns one continuous piecewise-afine boundary relative to an afine feature dictionary.

Let the true transitions along a path be $T _ { 0 } , \ldots , T _ { \mathfrak { h } - 1 }$ and the latent transitions be $A _ { 0 } , \ldots , A _ { \mathfrak { h } - 1 }$ Define the branch-local transition defects

$$
\Delta _ { j } ^ { \mathrm { b r } } = W T _ { j } - A _ { j } W .\tag{145}
$$

Lemma H.18 (Telescoping identity across changing regimes). With an empty product equal to the identity,

$$
\begin{array} { r l } {  { W T _ { \mathfrak { h } - 1 } \cdot \cdot \cdot T _ { 0 } - A _ { \mathfrak { h } - 1 } \cdot \cdot \cdot A _ { 0 } W } } \\ { \quad } & { = \displaystyle \sum _ { j = 0 } ^ { \mathfrak { h } - 1 } A _ { \mathfrak { h } - 1 } \cdot \cdot \cdot A _ { j + 1 } \Delta _ { j } ^ { \mathrm { { b r } } } T _ { j - 1 } \cdot \cdot \cdot T _ { 0 } . } \end{array}\tag{146}
$$

Proof. For each $j = 0 , \ldots , \mathfrak { h }$ , define

$$
Q _ { j } = A _ { \mathfrak { h } - 1 } \cdot \cdot \cdot A _ { j } W T _ { j - 1 } \cdot \cdot \cdot T _ { 0 } ,\tag{147}
$$

where $Q _ { \mathfrak { h } } = W T _ { \mathfrak { h } - 1 } \cdot \cdot \cdot T _ { 0 }$ and $Q _ { 0 } = A _ { \mathfrak { h } - 1 } \cdot \cdot \cdot A _ { 0 } W$ . Consecutive terms satisfy

$$
Q _ { j + 1 } - Q _ { j } = A _ { \mathfrak { h } - 1 } \cdot \cdot \cdot A _ { j + 1 } ( W T _ { j } - A _ { j } W ) T _ { j - 1 } \cdot \cdot \cdot T _ { 0 }\tag{148}
$$

$$
= A _ { \mathfrak { h } - 1 } \cdot \cdot \cdot A _ { j + 1 } \Delta _ { j } ^ { \mathrm { b r } } T _ { j - 1 } \cdot \cdot \cdot T _ { 0 } .\tag{149}
$$

Summing $Q _ { j + 1 } - Q _ { j }$ from $j = 0$ to h − 1 cancels all intermediate $Q _ { j }$ and yields Eq. 146.

The identity has the same interpretation as the fixed-regime telescope, but the operators before and after each local defect now depend on the active branch.

Theorem H.19 (A continuous piecewise-afine event requires a hinge). Let an open connected set $U \subset \mathbb { R } ^ { d }$ intersect both sides of the hyperplane $a ^ { \top } x = b$ , with $a \neq 0$ . Suppose a continuous map $F : U  \mathbb { R } ^ { m }$ is afine on each side:

$$
F ( x ) = { \left\{ \begin{array} { l l } { A - x + c _ { - } , } & { a ^ { \top } x \leq b , } \\ { A _ { + } x + c _ { + } , } & { a ^ { \top } x \geq b . } \end{array} \right. }\tag{150}
$$

Then there is a vector $u \in \mathbb { R } ^ { m }$ such that

$$
F ( x ) = A _ { - } x + c _ { - } + u ( a ^ { \top } x - b ) + ,\tag{151}
$$

where $( r ) _ { + } = \operatorname* { m a x } \{ r , 0 \}$ $I f u \ne 0$ , the hinge $( a ^ { \top } x - b ) _ { + }$ is not in the afine feature dictionary on $U ,$ so afine features alone cannot represent F. Appending this hinge is suficient. In addition, any feature dictionary that contains the afine functions and admits an exact linear readout of F must contain the hinge in its linear span.

Proof. Choose a point $x _ { 0 } \in U$ on the boundary $a ^ { \top } x _ { 0 } = b$ . Such a point exists because $U$ is connected and intersects both open half-spaces. Let $v \in$ ker $( a ^ { \top } )$ . For suficiently small τ , both $x _ { 0 } + \tau v$ and $x _ { 0 } - \tau v$ remain on the boundary and in U. Continuity of the two afine formulas on the boundary gives

$$
( A _ { + } - A _ { - } ) x + ( c _ { + } - c _ { - } ) = 0\tag{152}
$$

for every boundary point x in a neighborhood of $x _ { 0 }$ . Subtract this identity at $x _ { 0 } + \tau v$ and x<sub>0</sub> to obtain

$$
( A _ { + } - A _ { - } ) v = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } v \in \ker ( a ^ { \top } ) .\tag{153}
$$

Each row of $A _ { + } - A _ { - }$ therefore annihilates the $( d - 1 )$ -dimensional space $\ker ( a ^ { \top } )$ . Its row space lies in the one-dimensional span of $a ^ { \top }$ , so some $u \in \mathbb { R } ^ { m }$ satisfies

$$
A _ { + } - A _ { - } = u a ^ { \top } .\tag{154}
$$

Evaluate the boundary identity at $x _ { 0 }$ . Since $a ^ { \top } x _ { 0 } = b$

$$
0 = u a ^ { \top } x _ { 0 } + c _ { + } - c _ { - } = u b + c _ { + } - c _ { - } ,\tag{155}
$$

and hence $c _ { + } - c _ { - } = - u b$ . On the upper half-space,

$$
\begin{array} { r } { A _ { + } x + c _ { + } = A _ { - } x + c _ { - } + u ( a ^ { \top } x - b ) , } \end{array}\tag{156}
$$

while the added term is zero on the lower half-space. This proves Eq. 151.

If $u \ne 0$ and the hinge were afine on U, it would vanish as an afine function on the nonempty open set $U \cap \{ a ^ { \top } x < b \}$ . Its coeficients would then be zero, forcing it to vanish throughout U. This contradicts its nonzero values on $U \cap \{ a ^ { \top } x > b \}$ . Thus an afine dictionary cannot represent the hinge. Adding the displayed hinge is suficient by Eq. 151. Finally, because u $\neq 0$ , choose $q \in \mathbb { R } ^ { m }$ with $q ^ { \top } u = 1$ . Eq. 151 then gives

$$
( a ^ { \top } x - b ) _ { + } = q ^ { \top } F ( x ) - q ^ { \top } ( A _ { - } x + c _ { - } ) .\tag{157}
$$

If a dictionary contains afine functions and linearly represents $F _ { ; }$ the right-hand side lies in its linear span. The dictionary must therefore span the hinge, which completes the proof. □

Corollary H.20 (Ballistic contact creates a gravity hinge). Suppose the free-flight height at a fixed time is $y _ { \mathrm { f r e e } } ( g ) = a - b g$ with $b > 0$ , and a floor clips the height to

$$
y ( g ) = \operatorname* { m a x } \{ a - b g , 0 \} .\tag{158}
$$

Let $g _ { c } = a / b$ . On any interval that crosses $g _ { c }$

$$
y ( g ) = b ( g _ { c } - g ) _ { + }\tag{159}
$$

is not afine in $g .$

Proof. For $g < g _ { c } ,$ , the height is $a - b g = b ( g _ { c } - g )$ and has derivative $- b$ . For $g > g _ { c } .$ , the height is zero and has derivative zero. The derivative changes at $g _ { c } ,$ so no single afine function agrees on an interval crossing that point. The displayed hinge agrees with both branches. □

The result covers one continuous piecewise-afine boundary. It does not cover discontinuous, frictional, restitution-dependent, or multi-contact dynamics, nor does it imply a literal neural hinge, a larger latent for every collision, or better contact modeling by SG-JEPA. It afects latent closure only if the retained quantity responds to the slope change u.

## H.8 Supplementary calculation: SIGReg and its limits

SIGReg enters Equation equation 16, but it is separate from the composition and law-coverage arguments. Near Gaussian identity covariance, its population statistic penalizes scale mismatch and unequal directional variance, while a finite batch introduces bias. This local calculation neither proves noncollapse nor attributes the dynamics diference to SIGReg.

For a batch of $B \geq 1$ latent samples $z _ { 1 } , \ldots , z _ { B } \in \mathbb { R } ^ { d _ { z } }$ , a random direction $a \sim \mathrm { U n i f } ( \mathbb { S } ^ { d _ { z } - 1 } )$ , knots $t _ { \ell } ,$ and weights $\omega _ { \ell } \geq 0$ , consider

$$
\mathcal { R } _ { \mathrm { S I G } } = B \sum _ { \ell } \omega _ { \ell } \left| \frac { 1 } { B } \sum _ { i = 1 } ^ { B } e ^ { \mathrm { i } t _ { \ell } a ^ { \top } z _ { i } } - e ^ { - t _ { \ell } ^ { 2 } / 2 } \right| ^ { 2 } .\tag{160}
$$

Here $\mathrm { i } ^ { 2 } = - 1$ . The batch average is the empirical characteristic function, a Fourier summary of the distribution after projection onto a. For a zero-mean Gaussian with covariance $C ,$ define the direction-averaged population functional

$$
\mathcal { R } _ { \mathrm { S I G } } ^ { \mathrm { p o p } } ( C ) = \mathbb { E } _ { a } \left[ B \sum _ { \ell } \omega _ { \ell } \left| \mathbb { E } _ { z \sim \mathcal { N } ( 0 , C ) } e ^ { \mathrm { i } t _ { \ell } a ^ { \top } z } - e ^ { - t _ { \ell } ^ { 2 } / 2 } \right| ^ { 2 } \right] .\tag{161}
$$

Proposition H.21 (Local covariance penalty induced by SIGReg). Assume some nonzero knot has positive weight. $I f C = I + \Delta _ { C } \succ 0 , \Delta _ { C } = \Delta _ { C } ^ { \top }$ , and $\| \Delta _ { C } \| _ { F }$ is small, then the functional in $E q$ . 161 satisfies

$$
\mathcal { R } _ { \mathrm { S I G } } ^ { \mathrm { p o p } } ( I + \Delta _ { C } ) = \kappa _ { B , d _ { z } } \left[ 2 \| \Delta _ { C } \| _ { F } ^ { 2 } + ( \operatorname { t r } \Delta _ { C } ) ^ { 2 } \right] + O ( \| \Delta _ { C } \| _ { F } ^ { 3 } ) ,\tag{162}
$$

$$
\kappa _ { B , d _ { z } } = \frac { B } { d _ { z } ( d _ { z } + 2 ) } \sum _ { \ell } \omega _ { \ell } \frac { t _ { \ell } ^ { 4 } } { 4 } e ^ { - t _ { \ell } ^ { 2 } } .\tag{163}
$$

For a finite batch of independent samples and a fixed direction with $\boldsymbol { v } = \boldsymbol { a } ^ { \intercal } C \boldsymbol { a }$ , the exact expectation at one knot is

$$
B \left( e ^ { - t ^ { 2 } v / 2 } - e ^ { - t ^ { 2 } / 2 } \right) ^ { 2 } + 1 - e ^ { - t ^ { 2 } v } .\tag{164}
$$

Consequently, after averaging over both the independent batch and the random direction, for a constant $C _ { 0 }$ that does not depend on $\Delta _ { C }$

$$
\begin{array} { r l } & { \mathbb { E } \mathcal { R } _ { \mathrm { S I G } } ^ { \mathrm { e m p } } ( I + \Delta _ { C } ) = C _ { 0 } + \beta _ { 1 , d _ { z } } \mathrm { t r } \Delta _ { C } } \\ & { \quad \quad \quad \quad + \kappa _ { B , d _ { z } } ^ { \mathrm { e m p } } \left[ 2 \| \Delta _ { C } \| _ { F } ^ { 2 } + ( \mathrm { t r } \Delta _ { C } ) ^ { 2 } \right] + O ( \| \Delta _ { C } \| _ { F } ^ { 3 } ) , } \end{array}\tag{165}
$$

$$
\beta _ { 1 , d _ { z } } = \frac { 1 } { d _ { z } } \sum _ { \ell } \omega _ { \ell } t _ { \ell } ^ { 2 } e ^ { - t _ { \ell } ^ { 2 } } ,
$$

$$
\kappa _ { B , d _ { z } } ^ { \mathrm { e m p } } = \frac { B - 2 } { d _ { z } ( d _ { z } + 2 ) } \sum _ { \ell } \omega _ { \ell } \frac { t _ { \ell } ^ { 4 } } { 4 } e ^ { - t _ { \ell } ^ { 2 } } .
$$

Proof. For fixed $^ { a , }$ the scalar $a ^ { \top } z$ is Gaussian with variance $\boldsymbol { v } = \boldsymbol { a } ^ { \intercal } \boldsymbol { C } \boldsymbol { a }$ . Its characteristic function at t is $e ^ { - t ^ { 2 } v / 2 }$ . Write $v = 1 + \eta$ , where $\eta = a ^ { \top } \Delta _ { C } a$ . A Taylor expansion around $\eta = 0$ gives

$$
e ^ { - t ^ { 2 } ( 1 + \eta ) / 2 } - e ^ { - t ^ { 2 } / 2 } = e ^ { - t ^ { 2 } / 2 } \left( e ^ { - t ^ { 2 } \eta / 2 } - 1 \right)\tag{166}
$$

$$
= - \frac { t ^ { 2 } } { 2 } e ^ { - t ^ { 2 } / 2 } \eta + O ( \eta ^ { 2 } ) ,\tag{167}
$$

$$
\left( e ^ { - t ^ { 2 } ( 1 + \eta ) / 2 } - e ^ { - t ^ { 2 } / 2 } \right) ^ { 2 } = \frac { t ^ { 4 } } { 4 } e ^ { - t ^ { 2 } } \eta ^ { 2 } + O ( | \eta | ^ { 3 } ) .\tag{168}
$$

For a uniform direction on the unit sphere,

$$
\mathbb { E } _ { a } [ \eta ] = \frac { \mathrm { t r } \Delta _ { C } } { d _ { z } } ,\tag{169}
$$

$$
\mathbb { E } _ { a } [ \eta ^ { 2 } ] = \frac { 2 \| \Delta _ { C } \| _ { F } ^ { 2 } + ( \operatorname { t r } \Delta _ { C } ) ^ { 2 } } { d _ { z } ( d _ { z } + 2 ) } .\tag{170}
$$

Multiplying by $B \omega _ { \ell }$ , summing over knots, and using the second identity proves $\operatorname { E q }$ . 162.

For the finite batch, let $Y _ { i } = e ^ { \mathrm { i } t a ^ { \top } z _ { i } }$ . Since $| Y _ { i } | = 1$ 2

$$
\mathbb { E } \left| \frac { 1 } { B } \sum _ { i = 1 } ^ { B } Y _ { i } - e ^ { - t ^ { 2 } / 2 } \right| ^ { 2 } = \left| e ^ { - t ^ { 2 } v / 2 } - e ^ { - t ^ { 2 } / 2 } \right| ^ { 2 }\tag{171}
$$

$$
+ \frac { 1 } { B } \left( 1 - e ^ { - t ^ { 2 } v } \right) .\tag{172}
$$

Multiplication by B gives Eq. 164. Finally,

$$
1 - e ^ { - t ^ { 2 } ( 1 + \eta ) } = 1 - e ^ { - t ^ { 2 } } + t ^ { 2 } e ^ { - t ^ { 2 } } \eta - \frac { t ^ { 4 } } 2 e ^ { - t ^ { 2 } } \eta ^ { 2 } + O ( | \eta | ^ { 3 } ) .\tag{173}
$$

Combining this with the population mismatch, then applying both identities in Eq. 170, gives the stated values of $\beta _ { 1 , d _ { z } }$ and $\kappa _ { B , d _ { z } } ^ { \mathrm { e m p } }$ □

Because at least one nonzero knot has positive weight, $\kappa _ { B , d _ { z } } > 0$ and the population functional has a strict local quadratic minimum at identity covariance. The expected finite-batch statistic also has a linear trace term, so identity covariance need not minimize it. When $B > 2 , \kappa _ { B , d _ { z } } ^ { \mathrm { e m p } } > 0$ as well, but the linear term remains. The expansion supports SIGReg’s local scale and isotropy role. It does not analyze collapse at $C = 0$ , connect efective rank to prediction, or explain the representation diference.