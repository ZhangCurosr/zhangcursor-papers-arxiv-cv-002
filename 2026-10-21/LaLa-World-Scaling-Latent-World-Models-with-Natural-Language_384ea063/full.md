# LaLa-World: Scaling Latent World Models with Natural Language Actions

Junha Song1\* Geonmo Gu¹ Jeongeun Park2 Heejeong Nam³ Lucas Maes4 Byeongho Heo1 Dongyoon Han1 Sangdoo Yun¹†

1NAVER AI Lab 2Georgia Institute of Technology   
3Brown University 4Mila & Université de Montréal

## Abstract

This paper shows that JEPA-style latent world models (LWMs) can learn across diverse domains and data sources by adopting the language-conditioned interface of video generation models. We establish this approach as an effective and efficient basis for planning with imagination and downstream policy fine-tuning. To this end, we introduce LaLa-World, a Languageconditioned Latent World model that learns from simulation and robot trajectories together with broader image and video data. Two designs make this work. First, we describe both numerical actions and visual changes in language, enabling joint training of a single future predictor across heterogeneous datasets. Second, we build the predictor on a language model, allowing it to interpret language conditions while predicting future latent states. The resulting predictor achieves higher average MPC planning performance than existing LWMs trained separately for each domain or task. Adding image and video data further improves average planning performance and robustness to action phrasing. In a matched comparison with an RAEv2-based video generation model baseline, LaLa-World reaches comparable planning performance with 6× less training cost and 13.9× faster planning. Beyond planning, the pretrained LaLa-World serves as an initialization for LaLa-Policy, which jointly predicts actions and future visual representations. LaLa-Policy performs competitively with Cosmos Policy on LIBERO and RoboCasa, with up to 5.5× faster inference.

Website

github.io/lala-world

LaLa-World Code LaLa-World#LaLa-World

LaLa-Policy Code LaLa-World#LaLa-Policy

## Contents

1 Introduction 3   
2 Background and Motivation 5   
3 LaLa-World. 556   
3.1 Language-Conditioned Transition Data   
3.2 Model Architecture and Objective   
4 MPC Planning 7   
4.1 Results and Analysis 88   
4.2 Latent vs. Generative World Models   
5 LaLa-Policy: From World Model To Action Predictor 10   
6 Conclusion 11   
Acknowledgments 12   
References 12   
A Related Work 17   
A.1 Latent World Models 17   
A.2 Video Generation Models 17   
A.3 Planning and Policy Learning with World Models 18   
A.4 Action Prediction Models 18   
A.5 Visual Representation Pretraining 18   
B Future Directions 19   
C Experiments and Analyses 19   
C.1 ImageNet Training and Evaluation of LaLa-World 19   
C.2 Communicable World Modeling 20   
C.3 LaLa-World Architecture Ablations 21   
C.4 Predictor Initialization and Joint Training 23   
C.5 LaLa-World Vision Encoder Ablations 23   
C.6 Additional Results on Planning with Qualitative Language Actions 24   
C.7 Additional Results on Action Phrasing Robustness 25   
C.8 Additional Results on Latent vs. Generative World Models 25   
C.9 Additional Qualitative Planning Results . 25   
C.9.1 Planning with Numerical Language Actions. 25   
C.9.2 Planning with Qualitative Language Actions 26   
C.10 Qualitative Policy Results 26   
D Implementation and Experimental Details 26   
D.1 LaLa-World Training Data Details 26   
D.2 LaLa-World Architecture Details 27   
D.3 LaLa-World Training Details 28   
D.4 MPC Planning and Evaluation Details 28   
D.4.1 Benchmark Settings and Evaluation Metrics 28   
D.4.2 Action Aggregation and Execution 29   
D.4.3 Qualitative Language Action Conversion 29   
D.5 RAEv2 Baseline Details . 30   
D.6 LaLa-Policy Architecture Details 31   
D.7 Policy Model Training Data . 31   
D.8 LaLa-Policy Fine-tuning Details 32   
D.9 Policy Model Evaluation Details 32

![](images/53fbfa6e0b21f507196204ec46a4a05dc441514bdeaeb53dbbb45698cce2a5b1.jpg)

![](images/84e5f3edaec42a69f0b8e173ac523ee582911395da459f4d464f4b118da16ba9.jpg)

![](images/b59347932b61efd44969ce47774cb406cfcca9336356a443c1958eab35246dae.jpg)  
Figure 1: Overview of LaLa-World. (a) Language provides a shared interface for numerical actions and visual change, enabling joint training on control and video data. (b) A single predictor built on a language model uses encoded observations and language conditions to predict future visual representations. Fine-tuning the pretrained LaLa-World yields LaLa-Policy, which predicts actions as well as future visual representations. (c) The jointly trained predictor achieves higher average MPC planning performance than both separately trained LWM baselines and the generative world model baseline, RAEv2. (d) LaLa-Policy performs competitively with our adapted Cosmos Policy baseline on RoboCasa, while reducing average inference time per trajectory by a factor of 5.5×.

## 1 Introduction

A central capability of an intelligent agent is to anticipate how the world will change in response to its actions. Before pushing a cup across a table, we know where it will end up and whether it might fall. World models capture this capability by predicting future states under candidate actions, providing a basis for planning or decision-making (Sutton, 1991; Ha & Schmidhuber, 2018; LeCun, 2022; Hafner et al., 2020). A promising approach to making this predictive capability computationally efficient is to model how visual representations change under actions, without learning to reconstruct images at the pixel level. This motivates our focus on JEPA-style latent world models (LWMs), such as V-JEPA 2-AC (Assran et al., 2025) and DINO-WM (Zhou et al., 2025).¹

This efficiency of LWMs is particularly useful for planning with imagination in the latent space, which requires repeatedly evaluating candidate action sequences, by rolling out multiple candidate futures at low cost (Hansen et al., 2022). However, despite this advantage, the action-conditioned predictors in such LWMs have largely been trained on simulation or robotics datasets, where numerical actions have domain-specific dimensions and physical meanings. For example, [2, 4] may encode horizontal and vertical displacement in one domain and [4] force in another, complicating the learning of shared dynamics. Furthermore, image and video data, which provide a rich source of observations of the world, often lack these numerical action annotations, making them difficult to use for training action-conditioned future predictors. Therefore, scaling LWMs across diverse control actions and broader image and video data requires a shared interface that accommodates heterogeneous data.

We suggest that language can serve as the shared interface. We introduce LaLa-World, a languageconditioned LWM that learns from diverse control datasets and broader image and video data with a single future predictor. Language conditioning already supports learning from diverse visual data in video generation (Wan et al., 2025; Agarwal et al., 2025). We use language as a shared interface for numerical actions and visual changes and can thereby scale LWM training with a single predictor across diverse contexts and actions.

As illustrated in Figure 1, LaLa-World has two main designs. First, we convert numerical actions in simulation or robotics datasets into natural language, allowing them to be expressed in the same form, such as move 2 left and 4 up and push with force 2 downward. We combine these control trajectories with image and video transitions described in language to help the predictor relate descriptions to visual changes across diverse scenes. Second, we equip the future predictor with language understanding. Built on a language model, it interprets language conditions while predicting future latent states. These designs allow one model to learn from different action spaces and data sources through a shared prediction objective. Figure 2 shows that a single LaLa-World predictor serves multiple domains and predicts future states under qualitative actions such as move slightly left,

![](images/4821dbe5db8c53e678196dd6d325c710d571747cf842e79a8f5149c3e00bb1a1.jpg)  
Figure 2: MPC planning with language actions. LaLa-World imagines future representations under candidate actions with MPC and selects the most promising action toward the goal. Because all actions share one language interface, a single LaLa-World predictor can plan across multiple domains, despite differences in action dimensionality and physical meaning. The predictor also supports language-based actions without numerical values (bottom two rows), as detailed in Section D.4.3. The visualized observations result from executing the selected actions in the simulator.

Our experiments show that our jointly trained predictor achieves higher average MPC planning performance than existing LWM baselines trained separately for each domain. We also show that adding image and video data to training further improves planning performance and robustness to action phrasing. We compare LaLa-World with an RAEv2-based generative world model (Singh et al., 2026), whose predictor uses a diffusion Transformer (DiT). With the same language conditions training data, and visual encoder, LaLa-World outperforms this RAEv2 baseline in planning with 6× less training cost and 13.9× faster inference.

Beyond planning, we investigate whether the pretrained LWM can serve as a useful initialization for policy fine-tuning. Cosmos Policy (Kim et al., 2026) demonstrates this use of a pretrained video model by adapting Cosmos-Predict2 (Ali et al., 2025) to predict robot actions together with future frames. Similarly, we fine-tune LaLa-World into LaLa-Policy for joint prediction of actions and future visual representations. On LIBERO and RoboCasa, LaLa-Policy performs competitively with Cosmos Policy while reducing average inference time per trajectory by a factor of up to 5.5. We hope these results draw greater attention to latent world models, showing that even without visually compelling outputs, they can offer practical advantages over generative world models in planning and downstream policy learning.

We summarize our contributions as follows:

• We introduce LaLa-World, a latent world model that uses language as a shared interface for joint training on control trajectories and broader image and video data.

• We show that a single LaLa-World predictor learns dynamics across domains and outperforms existing LWMs in MPC planning.

• We compare LaLa-World with an RAEv2-based generative world model under matched settings, demonstrating higher planning performance and greater training and inference efficiency.

• We fine-tune the pretrained LaLa-World into LaLa-Policy, which performs competitively with existing policy models while achieving faster inference than Cosmos Policy.

![](images/c6180d9be43ccb9ad71ad2d3c3221a4ad2d20dd3055ef1572ee6ceae10fbaa4f.jpg)  
Figure 3: Examples of language-conditioned transition data. Each example consists of past observations, a language condition, and the next observation. Language conditions describe either numerical control actions from simulation and robotics or visual changes in images and videos. This shared interface enables training a single LWM across different action spaces and data sources.

## 2 Background and Motivation

World models aim to predict how an environment evolves under a given condition (Ha & Schmidhuber, 2018; LeCun, 2022; Wan et al., 2025; Agarwal et al., 2025). One approach to making future prediction computationally efficient is to predict changes in latent representations.Latent World Models(LWMs), including V-JEPA 2-AC (Assran et al., 2025) and DINO-WM (Zhou et al., 2025), follow this approach. These models commonly use numerical actions in the following action-conditioned formulation:

$$
\begin{array} { r } { e _ { t } = E _ { \theta } ( x _ { t } ) , \qquad \hat { e } _ { t + 1 } = P _ { \phi } \left( e _ { t - w + 1 : t } , c _ { t - w + 1 : t } \right) , \quad c _ { t } \in \mathbb { R } ^ { k } , } \end{array}\tag{1}
$$

where $x _ { t }$ and $c _ { t }$ denote the observation and k-dimensional numerical action condition at time $t ,$ respectively, $E _ { \theta }$ encodes observations into latent representations, and $P _ { \phi }$ predicts the future latent $\hat { e } _ { t + 1 }$ from a context window of length w. The action-conditioned predictors in these models are commonly trained on simulation or robotics trajectories and evaluated through planning in a domain-specific manner (Sobal et al., 2025; Maes et al., 2026a;b)

Video generation models(i.e., generative world models) are trained to synthesize images or videos conditioned on language (Brooks et al., 2024; Wan et al., 2025; Agarwal et al., 2025). This language interface allows them to learn visual changes across domains and data sources. Through this training, these models acquire world knowledge that extends their capabilities beyond video generation. Their pretrained weights, for instance, serve as a prior for action prediction: Cosmos Policy fine-tunes Cosmos-Predict2 into a robot policy (Kim et al., 2026; Ali et al., 2025).

ForLWMs, however, learning jointly from heterogeneous control datasets and broader visual data remains underexplored, as does their use in downstream policy learning. We therefore explore an approach that retains the efficiency of LWMs while adopting the flexible language interface of generative world models. The resulting model supports evaluation against existing LWMs in planning and against generative world models in both planning and policy fine-tuning.

## 3 LaLa-World

We introduce LaLa-World, a Language-conditioned Latent World model whose predictor takes a language condition z instead of the numerical action condition c in Equation (1). Two components are needed: transition data that describes the change from xt to $x _ { t + 1 }$ in language, and a predictor that understands it. Section 3.1 covers the first, describing how we construct such language-conditioned data, and Section 3.2 covers the second with the architecture and training objective

## 3.1 Language-Conditioned Transition Data

Language unifies the action spaces of multiple domains. Simulation and robotics datasets used to train LWMs record actions as numerical vectors. We convert these vectors into language conditions using fixed templates, as illustrated in Figure 3, retaining each domain's numerical values with their direction and magnitude. Language thus expresses actions across domains in a common form, allowing their control data to be pooled to train a single predictor. We apply this conversion to seven simulation and robotics datasets described in Section 4. The actions in these domains may differ in physical meaning and dimensionality. This raises the key question of whether a single future predictor can learn dynamics across these domains. Details of the dataset composition and conversion templates are provided in Section D.1.

Language enables action-conditioned training on broader image and video data. We also train the same future predictor on image and video transitions without numerical action annotations, using descriptions of visual change as conditions. Such data matter because the converted control datasets describe actions with fixed templates, which exposes the predictor to only a narrow range of language. In addition, descriptions of visual change provide language conditions for these transitions, extending training beyond the control trajectories. We therefore supplement them with captioned videos from OpenVid (Nan et al., 2025) and image-editing pairs with instructions from OmniEdit (Wei et al., 2025) and ShareGPT (Chen et al., 2025b). In these data, the caption describing the change between frames or the editing instruction serves directly as the language condition, so no numerical action annotation is needed. The predictor thereby sees a wider range of language and visual change.

## 3.2 Model Architecture and Objective

As shown in Figure 1(b), LaLa-World follows the encoder-predictor framework of existing LWMs, with simple adaptations for language conditioning. Below, we describe the observation encoder (i.e., vision encoder), transition predictor (i.e., future predictor), and training objective.

Encoder. LWMs use an observation encoder to map each RGB frame $x _ { t }$ to a latent representation $e _ { t }$ (Assran et al., 2025; Terver et al., 2025). For this role, vision transformers pretrained with self-supervised objectives are commonly used, including DINO encoders (Oquab et al., 2024; Siméoni et al., 2025) and V-JEPA 2 encoder (Assran et al., 2025). These encoders are pretrained on large collections of unlabeled images and videos, providing visual representations for diverse environments (Zhou et al., 2025). In V-JEPA 2-AC, the encoder remains frozen during world model training, which prevents encoder collapse and preserves its pretrained visual representations. Following JEPA-WMs (Terver et al., 2025), we use DINOv2-S/14 or DINOv3-L/16 as the frozen observation encoder. More details and ablations on vision encoders are provided in Section C.5.

Predictor. The predictor $P _ { \phi }$ takes the visual context $e _ { t - w + 1 : t }$ and the language condition z as input and predicts the next latent representation $\hat { e } _ { t + 1 }$ . Existing world models typically receive conditions in two ways: LWMs incorporate numerical actions through mechanisms such as AdaLN (Terver et al., 2025), feature conditioning (Zhou et al., 2025), and sequence conditioning (Nam et al., 2026), all of which offer no way to read text. Video generation models support language conditioning through a separate text encoder whose outputs condition the predictor (Wan et al., 2025), which adds parameters and text-encoding cost.

LaLa-World addresses both limitations by using a pretrained language model as the predictor itself: it reads the language condition as text and predicts future latent representations in the same network. Specifically, we use Qwen3.5-0.8B (Qwen Team, 2026) as our default predictor backbone. The language condition z is tokenized using the model's tokenizer and mapped to token embeddings, which are passed to the predictor together with the visual latents. To map the visual latents into the same embedding space, we use a two-layer MLP and add 3D positional embeddings indexed by time and 2D spatial location (Wang et al., 2024). The output $\hat { e } _ { t + 1 }$ is then read out from learned query tokens (Pan et al., 2025). We train the predictor by minimizing the squared error between the predicted and target future representations: $\bar { \mathcal { L } } = \| \hat { e } _ { t + 1 } - \bar { e } _ { t + 1 } \| ^ { 2 }$

Notably, we use proprioceptive state neither as an input nor as a prediction target. While proprioception can benefit LWMs tailored to a specific domain, it is defined in a domain-specific manner, which limits its applicability to a domain-agnostic world model. We further examine the predictor's backbone and architectural choices through ablations in Section C.3 and D.2, respectively, including BERT (Devlin et al., 2019) and Gemma (Gemma Team, 2025) as alternative backbones.

<table><tr><td>Method</td><td>Joint-training</td><td>Wall</td><td>PointMaze</td><td>Push-T</td><td>MetaWorld-R</td><td>MetaWorld-RW</td><td>Average</td></tr><tr><td>DreamerV3 (2025a)</td><td>x</td><td>39.6 (5.0)</td><td>43.8 (5.1)</td><td>32.3 (4.8)</td><td>25.0 (6.2)</td><td>10.4 (4.4)</td><td>30.2 (2.3)</td></tr><tr><td>TD-MPC2 (2024)</td><td>x</td><td>28.1 (4.6)</td><td>8.3 (2.8)</td><td> $1 6 . 7 \ : ( 3 . 8 )$ </td><td>12.5 (4.8)</td><td>22.9 (6.1)</td><td>17.7 (2.0)</td></tr><tr><td>IRIS (2023)</td><td>x</td><td> $3 0 . 2 ( 4 . 7 ) $ </td><td> $4 1 . 7 ( 5 . 0 )$ </td><td> $3 1 . 2 ( 4 . 7 ) $ </td><td>20.8 (5.9)</td><td>6.2 (3.5)</td><td>26.0 (2.2)</td></tr><tr><td>LeWorldModel (2026a)</td><td>x</td><td> $2 4 . 0 ( 4 . 4 )$ </td><td> $2 4 . 0 \ : ( 4 . 4 )$ </td><td>59.4 (5.0)</td><td> $1 6 . 7 ( 5 . 4 )$ </td><td>16.7 (5.4)</td><td>28.1 (2.2)</td></tr><tr><td>DINO-WM (2025)</td><td>x</td><td>90.6 (3.0)</td><td>59.4 (5.0)</td><td>65.6 (4.8)</td><td>52.1 (7.2)</td><td>43.8 (7.2)</td><td>62.3 (2.5)</td></tr><tr><td>JEPA-WMs (2025)</td><td>x</td><td> $7 9 . 2 ( 4 . 1 ) $ </td><td> ${ \bf 6 0 . 4 } \left( { \bf 5 . 0 } \right)$ </td><td>68.8 (4.7)</td><td>66.7 (6.8)</td><td>47.9 (7.2)</td><td>64.6 (2.6)</td></tr><tr><td>LaLa-World</td><td>√</td><td>95.8 (2.0)</td><td> $5 6 . 2 ( 5 . 1 )$ </td><td>81.2 (4.0)</td><td>75.0 (6.2)</td><td>52.1 (7.2)</td><td>72.1 (2.3)</td></tr><tr><td>Method</td><td>Joint-training</td><td>Rope</td><td>Granular</td><td>RoboCasa-R</td><td>RoboCasa-P</td><td>DROID</td><td>Average</td></tr><tr><td>DINO-WM (2025)</td><td>x</td><td>54.2 (10.2)</td><td>50.0 (14.4)</td><td>18.8 (6.9)</td><td>21.9 (7.3)</td><td>39.4 (3.8)</td><td>36.8 (4.1)</td></tr><tr><td>JEPA-WMs (2025)</td><td>x</td><td>62.5 (9.9)</td><td>50.0 (14.4)</td><td>37.5 (8.6)</td><td>31.2 (8.2)</td><td>43.0 (3.8)</td><td>44.8 (4.3)</td></tr><tr><td>V-JEPA 2-AC (2025)</td><td>x</td><td></td><td></td><td> $1 5 . 6 ( 6 . 4 )$ </td><td> $3 4 . 4 ( 8 . 4 )$ </td><td>42.9 (2.5)</td><td></td></tr><tr><td>LaLa-World</td><td>√</td><td> ${ \bf 6 6 . 7 ( 9 . 6 ) }$ </td><td> ${ \bf 5 8 . 3 ( 1 4 . 2 ) }$ </td><td> $\mathbf { 4 0 . 6 \ : ( 8 . 7 ) }$ </td><td> $\mathbf { 4 6 . 9 ( 8 . 8 ) }$ </td><td>40.2 (5.2)</td><td>50.5 (4.4)</td></tr></table>

Table 1: MPC planning results. We compare LaLa-World with existing world-model baselines across simulation and robot domains. LaLa-World achieves the highest average score in both tables. We report success rates (%), except for the DROID action score (Terver et al., 2025), with standard errors in parentheses. RoboCasa is evaluated zero-shot; -R, -RW, and -P denote Reach Reach-Wall, and Place, respectively. Joint-training denotes a single predictor trained across domains. Additionally, scores in Figure 1(c) are averaged over the benchmarks used to compare LaLa-World with the generative world model baseline (Section 4.2).

## 4 MPC Planning

Suppose we want to move a cup from our hand onto a table while avoiding nearby objects. We imagine the outcomes of candidate action sequences and select the one that brings the cup closest to the desired position (i.e., planning with imagination). In robot planning, the world model provides these imagined outcomes: given an initial observation $x _ { \mathrm { i n i t } }$ and a goal observation $x _ { \mathrm { g o a l } } .$ , a planner selects the action sequence whose imagined outcome is closest to the goal. Following prior LWM studies (Zhou et al., 2025; Terver et al., 2025), we evaluate planning with model predictive control (MPC), using the Cross-Entropy Method (CEM) to optimize action sequences (Rubinstein, 1997).

Evaluation Setup. At each planning round, we sample candidate action sequences and, for each sequence, apply Equation (1) autoregressively for H steps and score it by the squared distance between the final predicted latent state and the encoded goal:

$$
\mathcal { C } = \left. \hat { e } _ { t + H } - e _ { \mathrm { g o a l } } \right. ^ { 2 } , \quad e _ { \mathrm { g o a l } } = E _ { \theta } ( x _ { \mathrm { g o a l } } ) .\tag{2}
$$

CEM repeatedly samples sequences and updates the sampling distribution using those with the lowest costs. For LaLa-World, we verbalize candidate actions using the training templates. In simulation, we execute the corresponding numerical actions from an initial segment of the selected sequence. For domains requiring re-planning, we obtain a new observation and repeat this process. Benchmark-specific settings and evaluation criteria are provided in Section D.4.

We use seven control datasets from DINO-WM and JEPA-WMs (Terver et al., 2025): six simulation datasets–Wall (Zhou et al., 2025), PointMaze (Fu et al., 2020), Push-T (Chi et al., 2023), MetaWorld (Yu et al., 2020), Rope and Granular (Zhang et al., 2024)-and the real-robot dataset DROID (Khazatsky et al., 2024). We also evaluate on RoboCasa (Nasiriany et al., 2024), a domain excluded from training, to assess zero-shot transfer. In addition to existing LWM baselines including DINO-WM, we also train the world models of IRIS (Micheli et al., 2023), DreamerV3 (Hafner et al., 2025a), and TD-MPC2 (Hansen et al., 2024) on the same control datasets without reward or task information, which is the same setup for these baselines in DINO-WM. For a fair comparison, we train each baseline using its public implementation on the same data and evaluate all under the same planning settings, including horizon H and none uses proprioceptive states.

Decoder. To visualize and interpret predicted futures, we build a 12-layer transformer decoder D that maps predicted latents to RGB images: $\hat { x } _ { t + 1 } = D ( \hat { e } _ { t + 1 } )$ . We train the decoder with a combination of $L _ { 1 }$ reconstruction and LPIPS perceptual losses (Zhang et al., 2018), following the decoder training recipe of RAEv2 (Singh et al., 2026). It should be noted that the decoder is not used during world model training (Section 3.2).

## 4.1 Results and Analysis

A jointly trained LaLa-World predictor performs well across diverse domains. Table 1 reports MPC planning results across simulation and robot domains. While existing LWMs are typically trained within individual domains, LaLa-World trains its predictor jointly across all domains with different action spaces and physical dynamics. Despite these differences, LaLa-World achieves the highest average score in both benchmark groups in Table 1. On the five simulation tasks in the upper table, LaLa-World reaches an average success rate of 72.1%, exceeding JEPA-WMs by 7.5 percentage points. The largest gain over the best baseline in this group is on Push-T, where success increases from 68.8% to 81.2%. LaLa-World also achieves the highest zero-shot success rates on both RoboCasa tasks, reaching 40.6% on Reach and 46.9% on Place.

Training on image and video data improves performance and robustness to phrasing. We evaluate the effect of training LaLa-World on image and video transitions and find two benefits. The results are shown in Figure 4. First, adding image and video transitions improves planning performance. Specifically, the average score increases from 59.3 with control trajectories alone to 61.6 with all data. Second, LaLa-World becomes more robust when the same actions are expressed in different words. To evaluate this, we paraphrase the action templates at evaluation, for example, replacing move the gripper with shift the robot hand and move the agent with push the dot. LaLa-World remains robust when trained on all data,

![](images/34465f55121c204b158ebc522aacc5f42ca8a1e20bfb4d0c2723ef111c5b4bd2.jpg)  
Fig. 4: Benefits of image /video data inclusion.

including image and video transitions, whereas paraphrasing lowers the average score by 6.8 points with control trajectories alone. Further details and domain-level results are provided in Section C.7.

LaLa-World can predict the future from language without numbers. The language actions in our control trajectories retain numerical magnitudes. We next ask whether LaLa-World can predict outcomes from qualitative descriptions such as move slightly down. Using the action ranges observed in the training data, we define a conversion function that replaces each numerical magnitude with one of three qualitative levels: slightly, no modifier, or far. CEM samples numerical action sequences as before and passes them through this function to obtain the language conditions for LaLa-World. The bottom two rows of Figure 2 show successful planning examples in Wall and RoboCasa under these conditions. These examples indicate that the predictor can use qualitative descriptions to guide planning even when exact action magnitudes are unspecified. We provide more details on the evaluation setup in Section D.4.3 and additional results in Section C.6.

## 4.2 Latent vs. Generative World Models

With language as the action interface, LaLa-World can be compared with generative world models trained on the same kinds of data. LWMs learn to match future visual latents, whereas generative world models are trained for pixel-level generation. We ask how these approaches compare in planning performance and computational cost. To the best of our knowledge, this is the first comparison of latent and generative world models in planning under matched conditions, including language conditions and training data, as discussed in Section A.2.

Setup for a matched comparison. We use RAEv2 (Singh et al., 2026) as our generative world model baseline because it uses DINOv3-L/16 (Siméoni et al., 2025) rather than a VAE as its vision encoder. LaLa-World uses the same encoder in this comparison. Both models use language as the action interface and receive the same language conditions. We train both models on the same control trajectories and image and video transitions used to train LaLa-World. The full-depth predictors are comparable in size: LaLa-World's predictor and RAEv2's DiT (Peebles & Xie, 2023) each contain approximately 0.8B parameters. We initialize RAEv2 from its publicly released pretrained DiT weights and follow its training implementation, including flow matching (Lipman et al., 2023) and REPA (Yu et al., 2025). During planning, we use one or five denoising steps per prediction and compute the cost in Equation (2) between the resulting future representations and the encoded goal. Further details of RAEv2 are provided in Section D.5.

We compare the two world models in terms of training cost and planning time. For training cost, we track planning performance while training the full-depth predictors on eight H100 GPUs with a

![](images/1e5d4a557f43c0f3d9c4fdae4ae166896c718df3ce938286f882d8f58402e6ed.jpg)

![](images/5b846f14ba22efe15ae246256f2898f98d19ac15dbd4239c89f4e5574ec9fbaf.jpg)  
Figure 5: Latent vs. generative world models for MPC planning under matched setups. We match the vision encoder, language conditions, and training data to compare the two world models. (a) We track planning performance over training time for full-depth predictors under a 12-hour training budget. (b) We compare planning performance and average planning time per trajectory across four predictor depths, with all models trained on the same number of transition data. These results show that LaLa-World achieves higher planning performance than RAEv2 with lower training cost and planning time.

![](images/6459835dd9ed2c28074ba0f52954926146010228ea835ad85cf47dea667913e0.jpg)  
Figure 6: Visualization of predicted future states during MPC planning on DROID. We visualize decoded predictions from LaLa-World and RAEv2, using one or five denoising steps for RAEv2. Each example shows the initial and goal observations, the predicted rollout under the selected language actions, and planning time. Decoded images from LaLa-World may have lower visual quality because it is not trained for pixel reconstruction. Both models predict plausible future states while LaLa-World requires less planning time.

12-hour budget. For planning time, we vary the number of predictor layers to obtain four model sizes for each approach, train each variant on 16M transitions in total, and measure average planning time per trajectory on a single H100 GPU. Both experiments follow the planning setup in Section 4.

LaLa-World outperforms the generative baseline with lower training and planning costs. As shown in Figure 5(a), LaLa-World reaches five-step RAEv2's 12-hour performance in two hours and later achieves a planning performance of 59.4%, exceeding RAEv2's best performance of 48.0%. For RAEv2, using five denoising steps per prediction improves performance but increases planning time. In Figure 5(b), full-depth LaLa-World outperforms every RAEv2 variant and plans 2.9× and 13.9× faster than full-depth RAEv2 with one and five denoising steps, respectively. 2 Together, these results show that a world model trained to match future visual latents can support effective and efficient planning without targeting the reconstruction of visual details. Domain-level results are provided in Section C.8. Figure 6 shows imagined DROID rollouts under the selected language actions: RAEv2 generates plausible future observations even with one denoising step, while decoded LaLa-World predictions also depict plausible future states.

<table><tr><td>Method</td><td>Pretrained weights</td><td>Spatial Object</td><td></td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>Dita (2025)</td><td> $\overline { { { \bf D } { \bf I } { \bf N } { \bf O } { \bf v } 2 ^ { ' } { } ^ { 2 4 } + { \bf C } { \bf L } { \bf I } { \bf P } ^ { ' } { } ^ { 2 1 } } }$ </td><td>97.4</td><td>94.8</td><td>93.2</td><td>83.6</td><td>92.3</td></tr><tr><td>π0 (2025)</td><td> $\mathrm { P a l i G e m m a } ^ { \mathrm { ' 2 4 } }$ </td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>UniVLA (2025)</td><td> $\mathrm { P r i s m a t i c } ^ { \mathrm { ' } 2 4 }$ </td><td>96.5</td><td>96.8</td><td>95.6</td><td>92.0</td><td>95.2</td></tr><tr><td>π0.5 (2025)</td><td> $\mathrm { P a l i G e m m a } ^ { \mathrm { ' 2 4 } }$ </td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>OpenVLA-OFT (2025)</td><td> $\mathrm { P r i s m a t i c } ^ { \mathrm { ' } 2 4 }$ </td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td>CogVLA (2025b)</td><td> $\mathrm { P r i s m a t i c } ^ { \mathrm { ' } 2 4 }$ </td><td>98.6</td><td>98.8</td><td>96.6</td><td>95.4</td><td>97.4</td></tr><tr><td>Cosmos Policy (2026)</td><td>Random init. 25</td><td>94.7</td><td>98.9</td><td>96.3</td><td>88.6</td><td>94.6</td></tr><tr><td>Cosmos Policy (2026)</td><td>Cosmos-Predict2</td><td>97.6</td><td>99.8</td><td>97.9</td><td>94.0</td><td>97.3</td></tr><tr><td>LaLa-Policy</td><td>Random init.</td><td>94.4</td><td>99.6</td><td>95.2</td><td>87.2</td><td>94.1</td></tr><tr><td>LaLa-Policy</td><td> $\mathrm { Q w e n } 3 . 5 ^ { ' 2 6 }$ </td><td>96.8</td><td>99.8</td><td>96.0</td><td>90.2</td><td>95.7</td></tr><tr><td>LaLa-Policy</td><td> $_ \mathrm { L a L a - W o r l d }$ </td><td></td><td></td><td>98.6 100.0 97.2</td><td>95.0</td><td>97.7</td></tr></table>

Table 2: LIBERO policy results. We compare LaLa-Policy with existing policies on LIBERO. Pretrained weights indicates the source model for policy finetuning, while Random init. denotes random initialization. This table reports success rates (SR, %).
<table><tr><td>Method</td><td>Pretrained weights</td><td>Demos</td><td>SR (%)</td></tr><tr><td>π0 (2025)</td><td>PaliGemma 24</td><td>300</td><td>62.5</td></tr><tr><td>GR00T-N1.5 (2025)</td><td>Eagle  $2 . 5 ^ { ' 2 5 \mathrm { a } }$ </td><td>300</td><td>64.1</td></tr><tr><td>Video Policy (2025)</td><td>Stable Video Diff.&#x27;23a</td><td>300</td><td>66.0</td></tr><tr><td>LaLa-Policy</td><td>Random init.</td><td>300</td><td>65.1</td></tr><tr><td>LaLa-Policy</td><td>LaLa-World</td><td>300</td><td>66.4</td></tr><tr><td>UVA (2025a)</td><td> $\overline { { { \bf M A R } ^ { ' 2 4 } } }$ </td><td>50</td><td>50.0</td></tr><tr><td>π0.5 (2025)</td><td> $\mathrm { P a l i G e m m a } ^ { \mathrm { ' 2 4 } }$ </td><td>50</td><td>54.5</td></tr><tr><td>MolmoAct2 (2026)</td><td> $\mathrm { M o l m o } 2 – \mathrm { E R } ^ { ' 2 6 }$ </td><td>50</td><td>54.3</td></tr><tr><td>Cosmos Policy (2026)</td><td>Random init.</td><td>50</td><td>53.8</td></tr><tr><td>Cosmos Policy (2026)</td><td> $\mathrm { C o s m o s } { \mathrm { - P r e d i c t } } 2 ^ { ' 2 5 }$ </td><td>50</td><td>56.0</td></tr><tr><td>LaLa-Policy</td><td>Random init.</td><td>50</td><td>55.5</td></tr><tr><td>LaLa-Policy</td><td> $\mathrm { Q w e n } 3 . 5 ^ { ' 2 6 }$ </td><td>50</td><td>56.8</td></tr><tr><td>LaLa-Policy</td><td> $_ \mathrm { L a L a - W o r l d }$ </td><td>50</td><td>57.3</td></tr></table>

Table 3: RoboCasa policy results. We evaluate policy models on RoboCasa. Demos denotes the number of training trajectories.

## 5 LaLa-Policy: From World Model To Action Predictor

We now ask whether this pretrained LaLa-World provides a useful initialization for policy finetuning. Cosmos Policy (Kim et al., 2026) takes this approach by fine-tuning the pretrained video generation model Cosmos-Predict2 (Ali et al., 2025) to jointly predict actions and future frames.

LaLa-Policy. Following Cosmos Policy's fine-tuning procedure, we fine-tune LaLa-World into LaLa-Policy using policy data to predict an action chunk together with a future latent representation. We add learned action queries to the predictor $P _ { \phi }$ alongside its queries for future prediction. Given an observation $x _ { t }$ and a task instruction $\ell ,$ the outputs at the action queries provide a sequence of conditioning features $h _ { t : t + N - 1 }$ for an action generator $A _ { \psi }$ built on a diffusion transformer. We use the action-head architecture of Isaac-GR00T (Bjorck et al., 2025), which conditions action generation on vision-language features. Starting from Gaussian noise, $A _ { \psi }$ generates the action chunk through iterative denoising:

$$
\begin{array} { r l } & { \left( \hat { e } _ { t + N } , h _ { t : t + N - 1 } \right) = P _ { \phi } \left( E _ { \theta } ( x _ { t } ) , \ell \right) , } \\ & { ~ \hat { a } _ { t : t + N - 1 } = A _ { \psi } \left( \epsilon ; h _ { t : t + N - 1 } \right) , \quad \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{3}
$$

Here, N is the number of actions in the chunk, and the future prediction targets $E _ { \theta } ( x _ { t + N } )$ . Modeling an action distribution allows the policy to represent multiple action sequences for reaching the goal.

Policy fine-tuning setup. We compare LaLa-Policy with policies that predict actions alone (Intelligence et al., 2025; Li et al., 2025b) and models that jointly predict actions and future observations (Li et al., 2025a; Liang et al., 2025). For a fair comparison with the joint prediction baselines, we fine-tune Cosmos Policy (Kim et al., 2026) to predict only actions and future observations, without auxiliary future proprioception or value prediction. LaLa-Policy follows the same prediction setup in latent space. We fine-tune on LIBERO (Liu et al., 2023) and RoboCasa (Nasiriany et al., 2024), using demonstration trajectories composed of transitions $( x _ { t } , a _ { t } , x _ { t + 1 } )$ and paired with task instructions. For RoboCasa, we consider two training sets: 50 human demonstrations or 300 demonstrations generated with MimicGen per task. For LaLa-Policy, we keep the observation encoder frozen and jointly train the pretrained predictor and newly added policy modules. The action generator is trained from scratch. Dataset and training details are provided in Sections D.7 and D.8.

Evaluation Setup. The policy models are evaluated in LIBERO and RoboCasa simulators. At each step, we execute 16 predicted actions on LIBERO and the first 16 of the 32 predicted actions on RoboCasa. We report success rates, counting a trial as successful only if it meets the simulator's task-specific success criterion within the allowed number of actions. More details are in Section D.9

LaLa-Policy achieves competitive and efficient action prediction via latent world modeling. Tables 2 and 3 show that LaLa-World can be fine-tuned into an effective policy, performing competitively with Cosmos Policy on both benchmarks. Notably, this performance comes with up to 5.5× faster inference, measured by average inference time per trajectory (Table 4). Together with our planning results, these findings suggest that effective world models and action predictors can be learned without a pixel reconstruction objective. World modeling pretraining also benefits policy fine-tuning. Initializing the predictor from LaLa-World outperforms both random and Qwen3.5 initialization on both LIBERO and RoboCasa. This agrees with the finding of Cosmos Policy (Kim et al., 2026) that Cosmos-Predict2, pretrained on diverse videos, provides a useful prior for policy learning, and extends this observation to latent world models.

<table><tr><td colspan="2">LIBERO RoboCasa</td></tr><tr><td>Cosmos Policy</td><td>4.45s 11.71s</td></tr><tr><td>LaLa-Policy</td><td>0.86s 2.12s</td></tr><tr><td></td><td>(×5.2) (×5.5)</td></tr></table>

Tab. 4: Avg. inference time per trajectory and speedup.

![](images/3954ec673f9ed399ef5b327503c8681a8436d1b4b71818cf7de22004e0f141c7.jpg)  
Figure 7: Predicted future observations on LIBERO and RoboCasa. Each example shows the initial observation and task instruction, followed by simulator observations during policy execution (upper row) and decoded future predictions (lower row). LaLa-Policy predicts $( x _ { t } , \ell ) \to$ $\left( \boldsymbol { \hat { e } } _ { t + N } , \boldsymbol { \hat { a } } _ { t : t + N - 1 } \right)$ , and the simulator executes $\hat { a } _ { t : t + N - 1 }$ to yield $x _ { t + N } ;$ see Equation 3. Since LaLa-Policy is not trained for pixel reconstruction, decoded images may contain visual inaccuracies.

LaLa-Policy can anticipate future latent states resulting from its predicted actions. Given the current observation and a task instruction, LaLa-Policy jointly predicts an action chunk and a future latent representation. During policy execution in LIBERO and RoboCasa, we decode the predicted representations into images and visualize them alongside simulator observations in Figure 7. The decoded futures capture both robot motion and changes in object position during lifting and transport, showing that latent prediction can represent the dynamics of manipulation.

## 6 Conclusion

We presented LaLa-World, a latent world model that learns from heterogeneous control trajectories and broader image and video data through a shared language interface. This enables comparisons with existing LWMs in MPC planning and generative models in planning and policy fine-tuning. Despite sharing one predictor across domains, LaLa-World outperforms existing LWMs in average MPC planning performance. Under matched settings, including training data, it also surpasses the generative baseline in planning performance and efficiency. Following Cosmos Policy's use of world models pretrained on diverse data for policy learning, we fine-tuned LaLa-World into LaLa-Policy. It outperforms evaluated VLA baselines and achieves substantially faster inference than Cosmos Policy. These results demonstrate the value of latent world models for planning and policy learning without training to reconstruct visual details. We hope this work draws greater attention to latent world models, their scalability, and downstream uses.

For broader practical impact, future work could explore scaling training data and model size, extending prediction horizons, and evaluating policies on real robots, as discussed in Section B.

## Acknowledgments

We gratefully acknowledge NAVER AI Lab for providing the resources and support that facilitated this research

## References

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, 2023.

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In CVPR, 2025.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. TMLR, 2024.

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. In RSS, 2025.

Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, et al. Nvidia nemotron 3: Efficient and open intelligence. arXiv preprint arXiv:2512.20856, 2025.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023b.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Technical Report, 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. In RSS, 2025.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Max Ehrlich, et al. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. In NeurIPS, 2025a.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025b.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In RSS, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.

Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In NeurIPS, 2023.

Frederik Ebert, Chelsea Finn, Sudeep Dasari, Annie Xie, Alex Lee, and Sergey Levine. Visual foresight: Model-based deep reinforcement learning for vision-based robotic control. arXiv preprint arXiv: 1812.00568, 2018.

Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai Shirui Chen, Yi Ru Wang, et al. Molmoact2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In ICRA, 2017.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

Quentin Garrido, Mahmoud Assran, Nicolas Ballas, Adrien Bardes, Laurent Najman, and Yann LeCun. Learning and leveraging world models in visual representation learning. arXiv preprint arXiv:2403.00504, 2024.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In NeurIPS, 2018.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020.

Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering Atari with discrete world models. In International Conference on Learning Representations, 2021.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 2025a.

Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025b.

Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. In ICML, 2022.

Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, 2024.

Nicklas Hansen, Hao Su, and Xiaolong Wang. Learning massively multitask world models for continuous control. arXiv preprint arXiv:2511.19584, 2025.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022.

Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, Hengjun Pu, Ronglei Tong, Chengyang Zhao, Xizhou Zhu, Yu Qiao, Jifeng Dai, et al. Dita: Scaling diffusion transformer for generalist vision-language-action policy. In ICCV, 2025.

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025.

Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In ICML, 2024.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In RSS, 2025.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. In International Conference on Learning Representations, 2026.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B Tenenbaum. Learning to act from actionless videos through dense correspondences. In International Conference on Learning Representations, 2024.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. In ICML, 2024.

Yann LeCun. A path towards autonomous machine intelligence. OpenReview, 2022.

Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. In RSS, 2025a.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NeurIPS, 2024.

Wei Li, Renshan Zhang, Rui Shao, Jie He, and Liqiang Nie. Cogvla: Cognition-aligned vision-language-action model via instruction-driven routing & sparsification. In NeurIPS, 2025b.

Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, and Carl Vondrick. Video generators are robot policies. arXiv preprint arXiv:2508.00795, 2025.

Jessy Lin, Yuqing Du, Olivia Watkins, Danijar Hafner, Pieter Abbeel, Dan Klein, and Anca Dragan. Learning to model the world with language. In ICML, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. In NeurIPS, 2023.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026a.

Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, and Randall Balestriero. stable-worldmodel-v1: Reproducible world modeling research and evaluation. arXiv preprint arXiv:2602.08968, 2026b.

Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. In CoRL, 2023.

Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample-efficient world models. In International Conference on Learning Representations, 2023.

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mahmoud Assran, Koustuv Sinha, Michael Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-jepa 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026.

Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, and Randall Balestriero. Causal-jepa: Learning world models through object-level latent masking. In ICML, 2026.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. In International Conference on Learning Representations, 2025.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In RSS, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. TMLR, 2024.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https : //qwen. ai/ blog?id=qwen3.5.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Reuven Y Rubinstein. Optimization of computer simulation models with rare events. European Journal of Operational Research, 1997.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 2020.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang, Eli Shechtman, and Saining Xie. Improved baselines with representation autoencoders. arXiv preprint arXiv:2605.18324, 2026.

Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. arXiv preprint arXiv:2502.14819, 2025.

Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM SIGART Bulletin, 2(4):160–163, 1991.

Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, and Yann LeCun. What drives success in physical planning with joint-embedding predictive world models? arXiv preprint arXiv:2512.24497, 2025.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. NeurIPS, 2024.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. In International Conference on Learning Representations, 2023.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren. AdaJEPA: An adaptive latent world model arXiv preprint arXiv:2606.32026, 2026.

Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. In NeurIPS, 2015.

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In International Conference on Learning Representations, 2025.

Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Pack Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In International Conference on Learning Representations, 2024.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In CoRL, 2020.

Kaifeng Zhang, Baoyu Li, Kris Hauser, and Yunzhu Li. Adaptigraph: Material-adaptive graph-based neural dynamics for robotic manipulation. In RSS, 2024.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. In International Conference on Learning Representations, 2026.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. In ICML, 2025.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022.

Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. In RSS, 2025.

## A Related Work

We summarize in Figure A how LaLa-World and LaLa-Policy compare with representative world models and policy models  
![](images/6e77cc84f5df811ba717b834a68d9df3aa422cffbcf380e9b1329236ce2ce781.jpg)  
Figure A: Comparison with representative world models and policy models. (a) LaLa-World uses natural language conditions, enabling training on control trajectories together with image and video transitions. The RAEv2-based WM is our generative world model baseline under matched settings in Section 4.2. (b) Fine-tuned from LaLa-World, LaLa-Policy jointly predicts actions and future latents, whereas VLAs predict actions alone and Cosmos Policy predicts actions and the future with visual details.

## A.1 Latent World Models

Latent world models predict future states in a learned representation space rather than in pixels. Recent work follows the joint-embedding predictive architecture (Assran et al., 2023; 2025). Existing studies focus on improving latent prediction for planning (Assran et al., 2025) or stabilizing training (Maes et al., 2026a). In the former direction, PLDM (Sobal et al., 2025) demonstrates reward-free planning with learned latent dynamics, while DINO-WM (Zhou et al., 2025) shows the effectiveness of using a frozen DINOv2 vision encoder. V-JEPA 2-AC (Assran et al., 2025) extends latent planning to real-robot manipulation, and JEPA-WMs (Terver et al., 2025) study design choices including multistep rollouts. An orthogonal direction addresses training stability: LeWorldModel (Maes et al., 2026a) trains the vision encoder and predictor end to end, using SIGReg to prevent representation collapse. AdaJEPA (Wang et al., 2026) further improves robustness by updating the predictor at test time. These methods typically train action-conditioned predictors within a single domain, limiting their development into General World Models. Although Newt (Hansen et al., 2025) learns across multiple domains, its training remains focused on control and game environments. Joint training of action-conditioned predictors on control trajectories and broader image and video transitions remains underexplored. We therefore use language to unify actions across control datasets and co-train them with diverse image and video transitions. Our use of language as the action interface differs from Dynalang (Lin et al., 2024), which incorporates language as observations or task descriptions.

## A.2 Video Generation Models

Video generation models (i.e., generative world models) predict future observations conditioned on language (Ho et al., 2022; Blattmann et al., 2023b; Villegas et al., 2023; Kondratyuk et al., 2024). Language conditioning lets them learn from web-scale video (Brooks et al., 2024; Wan et al., 2025), and the resulting models serve as interactive simulators for games (Bruce et al., 2024), driving (Hu et al., 2023), and robotics (Yang et al., 2024; Agarwal et al., 2025; Ali et al., 2025). While video generation models can produce visually compelling content, using them for MPC planning is costly because evaluating candidate action sequences requires repeatedly synthesizing the future with visual details. V-JEPA 2-AC and V-JEPA 2.1 (Assran et al., 2025; Mur-Labadia et al., 2026) report faster planning than video generation baselines (Agarwal et al., 2025; Bar et al., 2025), but these comparisons differ in action interface, encoder, predictor size, and training data. RAE (Zheng et al., 2026; Singh et al., 2026) replaces the VAE encoder (Kingma & Welling, 2013) with a vision encoder pretrained with self-supervision, allowing a generative world model to share its encoder with an LWM. Using RAEv2 as a generative world model baseline, we compare it with LaLa-World under matched settings, including language conditioning.

## A.3 Planning and Policy Learning with World Models

Planning with learned dynamics models has a long history (Sutton, 1991; Ha & Schmidhuber, 2018; Watter et al., 2015; Finn & Levine, 2017; Ebert et al., 2018). PlaNet (Hafner et al., 2019) plans with the Cross-Entropy Method (CEM) in a latent space trained with reconstruction and reward prediction. Dreamer (Hafner et al., 2020; 2021; 2025a;b) and IRIS (Micheli et al., 2023) instead train actor-critic policies on imagined rollouts. MuZero (Schrittwieser et al., 2020) and TD-MPC (Hansen et al., 2022; 2024) learn latent dynamics with reward and value prediction to support tree search and model predictive control, respectively. More recent work defines a world model more narrowly as a predictor of future states (Sobal et al., 2025; Assran et al., 2025; Maes et al., 2026a). Planning with them is performed at test time by sampling-based optimization toward a goal image (i.e., MPC planning). Our experiments follow this reward-free and goal-conditioned protocol.

## A.4 Action Prediction Models

Vision-language-action (VLA) models such as RT-2 (Brohan et al., 2023), OpenVLA (Kim et al. 2024), $\pi _ { 0 . 5 }$ (Intelligence et al., 2025), UniVLA (Bu et al., 2025), and CogVLA (Li et al., 2025b) fine-tune vision-language models on large-scale robot demonstrations to map an observation and an instruction to actions. These models predict actions alone. When humans want to move a cup, however, they can imagine the action, how the cup will move, and where it will end up. Video-based policies such as UniPi (Du et al., 2023) and AVDC (Ko et al., 2024) first generate videos of task execution and then translate them into control actions. Unified video-action models (Li et al., 2025a; Zhu et al., 2025) instead predict actions and future frames jointly. Building on pretrained video models, Cosmos Policy (Kim et al., 2026) fine-tunes Cosmos-Predict2 (Ali et al., 2025) to jointly predict actions, future states, and values. LaLa-Policy follows this recipe but starts from a latent world model, predicting future representations rather than reconstructing visual details.

## A.5 Visual Representation Pretraining

Self-supervised visual encoders provide the representations on which latent world models learn dynamics (Zhou et al., 2025; Assran et al., 2025). Contrastive and self-distillation methods (He et al. 2020; Chen et al., 2020; Caron et al., 2021; Oquab et al., 2024; Siméoni et al., 2025) learn invariances across augmented views, masked image modeling (He et al., 2022; Zhou et al., 2022) predicts masked patches, and JEPAs (Assran et al., 2023; Bardes et al., 2024) predict the representations of masked regions instead of their pixels. These encoders are conventionally evaluated by probing or fine-tuning on recognition tasks such as classification, segmentation, and depth estimation. V-JEPA 2 (Assran et al., 2025) introduced planning as a further evaluation: the pretrained encoder is frozen, an actionconditioned predictor is trained on top of it from robot trajectories, and the pair is used for MPC planning toward image goals. We follow the same evaluation protocol as V-JEPA 2, DINO-WM, and JEPA-WMs (Terver et al., 2025).

## B Future Directions

LaLa-World shows that a shared language interface enables joint LWM training on heterogeneous control trajectories and broader image and video data, with benefits for planning and policy finetuning. These results suggest several directions for extending the current approach.

1. Scaling data and model capacity. With a 0.8B predictor, adding image and video data improves planning performance and robustness to action phrasing (Figure 4). However, our training data remain smaller in scale than those of video generation models such as Wan (Wan et al., 2025), and we have not established whether these gains continue at larger scales. Future work should increase dataset size and predictor capacity separately and together, measuring their effects on planning performance and downstream policy learning.

2. Learning visual representations end to end. We freeze the vision encoder to prevent representation collapse, but this also prevents its representations from adapting to future prediction. Better world models may require visual representations that capture information useful for anticipating future states. Jointly training the vision encoder and predictor while preventing collapse is therefore an important direction for future work.

3. Extending prediction beyond one step. Video generation models produce sequences of future frames, whereas our predictor is trained with supervision only on the next latent state. Whether latent world models can learn to generate similarly long sequences through autoregressive prediction remains an open question. Addressing this would require training on autoregressive rollouts with losses over multiple future states, rather than only the next state. We leave this investigation to future work because of the additional training cost, with particular interest in how prediction quality changes as the rollout length increases.

4. Extending prediction horizons. Following the evaluation protocols of JEPA-WMs and DINO-WM (Terver et al., 2025; Zhou et al., 2025), our MPC experiments use relatively short planning horizons of at most six steps (Table H). Evaluating longer horizons would help assess whether the model can anticipate distant consequences and support planning in real-world tasks that require such foresight. The multi-step supervision discussed above could support this extension.

5. Evaluating on real robots. Our experiments on LIBERO and RoboCasa show that initializing LaLa-Policy from LaLa-World outperforms existing VLAs, demonstrating the value of a pretrained latent world model for policy fine-tuning. While these simulation experiments establish this benefit, extending the evaluation to real robots is an important next step.

6. Best-of-N planning with LaLa-Policy. LaLa-Policy achieves competitive action prediction without value supervision. Cosmos Policy (Kim et al., 2026) further learns from policy rollouts to predict future states and values for candidate action chunks, enabling best-of-N planning. We consider extending LaLa-Policy to such planning a valuable avenue for future work.

We hope these directions broaden the use of LWMs while retaining their efficiency in planning and policy learning.

## C Experiments and Analyses

## C.1 ImageNet Training and Evaluation of LaLa-World

Inspired by Image World Models (IWM) (Garrido et al., 2024), we use ImageNet (Deng et al., 2009) transformations to examine whether LaLa-World can predict visual changes specified in language.

Training. We construct transitions from a source image, a transformed target view, and a language description of the transformation (Figure B, left). Crops, shifts of the view, rotations, and horizontal flips become language actions, allowing static images to enter the same training format as control data. For example, a crop followed by a horizontal flip is paired with a sentence describing both operations. The predictor receives the source representation and this description, and learns to match the encoded target view. We train the predictor jointly on these pairs and other transition data.

Evaluation. For each of 200 ImageNet validation images, we construct a bank of 256 transformed views of that image, including photometric distractors with changes in color or brightness. Given the source and a geometric language action, the predictor must retrieve the corresponding target by the mean cosine similarity of aligned patch representations. We report mean reciprocal rank (MRR), averaging the reciprocal of the target's rank over queries. A target ranked first contributes 1, second 1/2, and third 1/3, so higher MRR indicates more accurate retrieval. Because all candidates show the same underlying image, retrieval requires distinguishing the requested transformation.

![](images/e1d7ab8a7ae985493eada79d338b8e7750e3e7dfd9ef3396b72fc4664bb49fd3.jpg)  
Figure B: ImageNet training pairs and transformation retrieval. Left: a source image and a transformed view form a training pair, with the transformation described in language. Right: the predicted representation is used to rank 256 transformed views of the same validation image; a subset is illustrated. The outlined view is the target specified by the language action.

<table><tr><td>Method</td><td>MRR↑</td></tr><tr><td>IWM (default)†</td><td>0.79</td></tr><tr><td>IWM (strong jitter)†</td><td>0.85</td></tr><tr><td colspan="2">LaLa-World</td></tr><tr><td>Source latent</td><td>0.024</td></tr><tr><td>Predicted latents, empty language</td><td>0.026</td></tr><tr><td>Predicted latents, language action</td><td>0.856</td></tr></table>

Table A: ImageNet transformation retrieval. We report mean reciprocal rank (MRR) over 256 transformed views of each source image. The LaLa-World rows compare retrieval using the source latent with predicted latents conditioned on empty language or a language action. †Published IWM results (Garrido et al., 2024) use different training and transformation settings.

Controls. The source latent is the encoded input image, used directly for retrieval. The predicted latents are obtained from LaLa-World with either empty language (" ") or the language action.

Results. Table A shows that LaLa-World reaches 0.856 MRR with language actions, while using the source latent directly or predicting with empty language gives 0.024 and 0.026, respectively. This gap indicates that the predictor uses language to model the specified visual change, extending the action interface to transformations of static images.

## C.2 Communicable World Modeling

CEM-based MPC requires thousands of imagined action sequences, which limits its practicality even when each world-model rollout is inexpensive. A natural direction is to let a capable LLM act as the decision-maker, using the world model to evaluate its proposed actions. Since LaLa-World reads actions as language, it provides an interface for this interaction. We therefore ask whether an LLM can replace CEM with fewer imaginations, and whether this reduced search budget still yields successful plans.

Planning procedure. We call this setting communicable world modeling, in which an LLM actor and LaLa-World exchange language actions and their evaluation costs (Figure C). The LLM receives the initial and goal observations and proposes N candidate action sequences using the numerical language of the training templates, such as move 0.6 left and 1.7 down. LaLa-World imagines each sequence and returns the cost in Equation (2); a lower cost indicates a predicted outcome closer to the goal. The LLM uses these costs to revise its proposals. After K rounds, we select the lowest-cost sequence for execution and replan when required by the benchmark. During this search, the costs are the only feedback: the LLM sees neither the imagined states nor privileged simulator states.

![](images/a8b79011d838af043cb6f28b36df36bc81d6da482d3a0468be510558f998eff4.jpg)  
Figure C: Communicable world modeling. An LLM takes the place of CEM and talks to LaLa-World in language. It reads the initial and goal observations, writes N candidate plans as language actions (top message), and receives only the goal-distance cost of each plan (bottom message), which is the same cost that CEM minimizes. After K rounds, the lowest-cost plan is executed. The LLM needs two orders of magnitude fewer world-model rollouts than CEM.

<table><tr><td>Planner</td><td>Wall</td><td>PointMaze</td><td>RoboCasa-R</td><td>DROID</td></tr><tr><td>CEM</td><td>95.8 (9,000)</td><td>56.2 (9,000)</td><td>40.6 (54,000)</td><td>40.2 (4,500)</td></tr><tr><td>Claude Opus 5</td><td>100.0 (30×3=90)</td><td>31.2 (30×3=90)</td><td>53.1 (10×2×4=80)</td><td>36.2 (20×2=40)</td></tr><tr><td>GPT-6 Astra</td><td>91.3 (8×1=8)</td><td>26.1 (8×3=24)</td><td>17.4 (8×1×12=96)</td><td>44.7 (16×3=48)</td></tr></table>

Table B: MPC planning with an LLM actor in place of CEM. We report success rates (%), except for the DROID action score, with the number of world-model rollouts per trajectory in parentheses. For the LLM actors, the rollout count is written as $N \times K .$ , e.g., 30 × 3 = 90, where N = 30 is the number of candidate action sequences proposed per round and K = 3 is the number of rounds; for RoboCasa-R, the third factor is the number of replanning steps per trajectory. The LLM actors use 8–96 rollouts per trajectory, compared with 4,500–54,000 for CEM.

Evaluation. We use Claude Opus 5 and GPT-6 Astra as actors and compare their results with the CEM baseline from Table 1 in Table B. We count one imagined candidate sequence as one rollout and report the total per trajectory, including replanning. Success rates measure task completion in simulation, while DROID uses the score defined in Equation (5).

Results. Table B shows that the LLM actors use 8–96 rollouts per trajectory, compared with 4,500–54,000 for CEM. Claude Opus 5 reaches 100.0% success on Wall with 90 rollouts and 53.1% on RoboCasa-R with 80 rollouts. Figure D shows successful simulation episodes. These results demonstrate that an LLM can plan through the language interface of LaLa-World with a small imagination budget, while leaving room to improve performance on tasks requiring precise control

## C.3 LaLa-World Architecture Ablations

We examine how training data and predictor design affect the use of language conditions, beyond what validation loss alone reveals. A predictor may fit transitions using visual regularities or numerical patterns without responding to the meaning of action words.

Language sensitivity. For a held-out transition, we measure prediction accuracy relative to retaining the current state:

$$
S ( z ) = 1 - \frac { \| \hat { e } _ { t + 1 } ( z ) - e _ { t + 1 } \| _ { 2 } } { \| e _ { t } - e _ { t + 1 } \| _ { 2 } } ,\tag{4}
$$

![](images/986a921f71846d507a2cbe1811b9f79b61e8da29507a40b4b13f9b8121aa76ff.jpg)  
Figure D: Qualitative Results of Communicable World Modeling with an LLM Actor. Each row shows the initial and goal observations and the simulator observations after executing the lowest-cost sequence written by the LLM, with its language action below each frame. Below the goal observation, we list the cost of the best sequence at each round, which decreases as the LLM revises its plans from the returned costs.

where $\hat { e } _ { t + 1 } ( z )$ is the prediction conditioned on z. The score is 1 for an exact prediction, 0 for returning $\textstyle e _ { t } ,$ and negative when the prediction is farther from the target than $e _ { t }$

We keep the target fixed and vary the condition: numerical language, qualitative language without numbers, reversed direction words in numerical language, and swapped magnitude words in qualitative language (Figure E(a)). Qualitative descriptions preserve direction but coarsen magnitude, so some loss of accuracy is expected. By contrast, following reversed direction words should move the prediction away from the original target and can make the score negative. Magnitude swaps preserve direction but change the amount of motion, so we expect a smaller drop. Panel (b) illustrates this expected pattern.

Training data. With control data alone, the score drops when numerical magnitudes are replaced by qualitative descriptions (Figure E(c)). Adding image and video transitions narrows this gap and makes the direction-swap score more negative. These results suggest that broader transition data help the predictor interpret qualitative magnitudes and respond to direction words.

Language backbone. We train BERT-0.3B (Devlin et al., 2019), Gemma-0.3B (Gemma Team, 2025), Nemotron-1B (Blakeman et al., 2025), and Qwen3.5-0.8B (Qwen Team, 2026) on the same data. Qwen3.5 most closely follows the expected sensitivity pattern, while Gemma shows a similar pattern with lower scores under the original actions (Figure E(d)). Nemotron reaches a validation loss close to Qwen3.5 (Figure F(a)), yet its score barely changes when direction words are reversed. Thus, low prediction loss alone does not establish that a predictor follows action semantics. Pretrained Qwen3.5 also reaches a lower loss earlier in training than its randomly initialized counterpart, although their final losses are similar.

Positional encoding and readout. We compare the positional encoding and readout designs shown in Figure G. We measure language reliance as $\Delta S = S ( z ) - S ( \emptyset )$ , the score drop when the condition is replaced by empty text. A larger gap indicates that language contributes more to prediction accuracy. The 3D input positional embeddings index time and spatial location, preserving the structure of the visual tokens. In Figure F(b), 3D input positional embeddings with learned query tokens yield the highest reliance among the tested designs. Reading predictions from the last frame's state tokens instead of learned queries reduces reliance under both 3D embeddings and

<table><tr><td>(a) z Alternatives</td><td>Language with numbers (Reference): &quot;Move right by 0.42 and up by 1.3&quot;</td><td>Pure language:</td><td>Direction swap:</td><td>Magnitude swap:</td></tr><tr><td></td><td></td><td>&quot;Move slightly right and far up&#x27;</td><td>&quot;Move left by 0.42 and down by 1.3&quot;</td><td>&quot;Move far right and slightly up&#x27;</td></tr></table>

![](images/e7ef939515e74410ea5eb0712aad45c9d65e1fa21b01e4ec828ed5da3308f7bc.jpg)

![](images/f3f9879f65bf893d45cbd7b43e74fa9fba1e1280b3be1d88f09d1dfde9b53b9f.jpg)

![](images/bfb2abe2ba853e8d2a56a592f8a33bc43741da46a09778feef7304813bf6cadd.jpg)

Figure E: Language sensitivity across training data and predictor backbones. (a) Language conditions used to evaluate the same target transition; red words mark semantic swaps. (b) Expected score pattern for a predictor that follows language. (c) Effect of adding image and video transitions to control data. (d) Comparison of language model backbones  
![](images/ab71f46589e21db6d27010754cc63894184b92cd31bef861dda75a9f048c5999.jpg)

![](images/4fc9a765ae80c89795183e2104d449d0e980c6691c0b1168371bb63fcef70056.jpg)  
Figure F: Predictor backbone, initialization, and architecture ablations. (a) Validation loss for language model backbones, including pretrained and randomly initialized Qwen3.5. (b) Language reliance $S ( z ) - S ( \emptyset )$ for positional encoding and readout variants. 3D input positional embeddings with learned queries yield the highest reliance.

MRoPE (Wang et al., 2024). These results support our default predictor, which combines Qwen3.5 with 3D input positional embeddings and learned queries.

## C.4 Predictor Initialization and Joint Training

We examine how predictor initialization and training scope affect MPC planning (Table C). All variants use the same predictor architecture and planning settings (Section 4). The jointly trained predictors use the full data mixture, including control trajectories and image and video transitions. For comparison, we train separate predictors from pretrained weights on each domain's control transitions, using the DROID-trained predictor for zero-shot evaluation on RoboCasa.

With pretrained initialization, joint training achieves an average score of 60.2, compared with 54.5 for single-domain training. Random initialization lowers the jointly trained predictor's average to 55.6. The two initializations reach similar final validation losses (Figure F(a)), suggesting that prediction loss alone does not fully capture planning performance.

## C.5 LaLa-World Vision Encoder Ablations

Setup. JEPA-WMs (Terver et al., 2025) shows that the preferred vision encoder for planning varies across domains. We examine this dependence in LaLa-World by comparing frozen DINOv2- S/14 (Oquab et al., 2024), DINOv3-L/16 (Siméoni et al., 2025), SigLIP2-So400m/14 (Tschannen et al., 2025), and DINOv2-S/14 + DINOv3-L/16. For each encoder configuration, we train a separate predictor on all training data, including control trajectories and image and video transitions. For DINOv2-S/14 + DINOv3-L/16, one predictor receives both representations and predicts both as future targets. We also report JEPA-WMs with each DINO encoder. Table D presents results on all ten benchmarks and their average score. The comparison in Table 1 allows the vision encoder to vary across benchmarks, as does JEPA-WMs (Terver et al., 2025); we include those reported scores as reference rows. DINO-WM in Table 1 likewise uses the better-performing vision encoder for each benchmark, while V-JEPA 2-AC retains the V-JEPA 2 encoder used in the original method.

![](images/d2a0346b48644f3be456fa80ef29bfbcca779d0b498bd1bc7db66b3fb4abb693.jpg)

Figure G: Predictor positional encoding and readout. (a) Projected state patches, language tokens and copies of one shared learned query vector receive additive 3D positional embeddings. The prediction head maps query outputs to future patch representations. (b) Native 1D uses sequence indices for visual and text tokens, handled by the backbone's positional mechanism. MRoPE uses 3D coordinates to rotate attention queries and keys inside the backbone. Inplace readout uses state-token hidden states.
<table><tr><td>Initialization</td><td>Training scope</td><td>Wall</td><td>PointMaze</td><td>Push-T</td><td>DROID</td><td>RoboCasa-R</td><td>RoboCasa-P</td><td>Average</td></tr><tr><td>Pretrained LLM</td><td>Single-domain</td><td>97.9 (1.5)</td><td>66.7 (4.8)</td><td>70.8 (4.6)</td><td>38.3 (4.5)</td><td>21.9(7.3)</td><td>31.2 (8.2)</td><td>54.5 (2.3)</td></tr><tr><td>Random init.</td><td>Joint</td><td>97.9 (1.5)</td><td>50.0 (5.1)</td><td>79.2 (4.1)</td><td>31.6 (3.8)</td><td>31.2 (8.2)</td><td>43.8 (8.8)</td><td>55.6 (2.4)</td></tr><tr><td>Pretrained</td><td>Joint</td><td>95.8 (2.0)</td><td>56.2 (5.1)</td><td>81.2 (4.0)</td><td>40.2 (5.2)</td><td>40.6 (8.7)</td><td>46.9 (8.8)</td><td>60.2 (2.5)</td></tr></table>

Table C: LaLa-World Predictor initialization and joint training. We compare pretrained and random initialization, and single-domain and joint training. We report success rates (%), except for the DROID action score, with standard errors in parentheses. Average is the unweighted mean of the six benchmark scores.

Results. Our encoder ablations suggest that DINOv2-S/14 and DINOv3-L/16 provide complementary information for planning, motivating further work on combining encoders (Tong et al., 2024) and learning representations tailored to latent world modeling. The preferred DINO encoder varies across benchmarks for both methods. For LaLa-World, DINOv2-S/14 performs better on Push-T and Granular, while DINOv3-L/16 performs better on PointMaze and both MetaWorld tasks. Using DINOv2-S/14 + DINOv3-L/16 with one predictor increases the average to 63.0, although improvements are not uniform across benchmarks.

## C.6 Additional Results on Planning with Qualitative Language Actions

We compare numerical and qualitative language actions to evaluate whether LaLa-World can plan from descriptions that omit exact action magnitudes. The action conversion and planning procedure are detailed in Section D.4.3.

Table E shows that qualitative language actions support planning in several domains. LaLa-World achieves success rates of 83.3% on Wall, 70.8% on MetaWorld-R, and 60.4% on MetaWorld-RW. Success rates decrease on some tasks, as expected when many distinct numerical magnitudes are mapped to only three qualitative levels. Even with this reduced information, it is still surprising that LaLa-World can use descriptions of direction and relative magnitude to guide planning without receiving exact numerical action values.

<table><tr><td>Vision encoder</td><td></td><td></td><td></td><td>Wall PointMaze Push-T MetaW.-R MetaW.-RW Rope Granular RoboC.-R RoboC.-P DROID : Average</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="14">JEPA-WMs</td></tr><tr><td>DINOv2-S/14</td><td>79.2</td><td>60.4</td><td>68.8</td><td>20.8</td><td>4.2</td><td></td><td>54.2 50.0</td><td></td><td>37.5</td><td>31.2</td><td>40.1</td><td>44.6</td></tr><tr><td>DINOv3-L/16</td><td>76.0</td><td>56.2</td><td>47.9</td><td>66.7</td><td></td><td>47.9</td><td>62.5</td><td>41.7</td><td>21.9</td><td>28.1</td><td>43.0</td><td>49.2</td></tr><tr><td>Table 1 (reference)</td><td>79.2</td><td>60.4</td><td>68.8</td><td>66.7</td><td>47.9</td><td></td><td>62.5 50.0</td><td></td><td>37.5</td><td>31.2</td><td>43.0</td><td>54.7</td></tr><tr><td>LaLa-World</td><td colspan="10"></td></tr><tr><td>DINOv2-S/14</td><td></td><td>45.8</td><td>81.2</td><td>8.3</td><td>4.2</td><td>58.3</td><td>58.3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DINOv3-L/16</td><td>95.8 94.8</td><td>56.2</td><td>52.1</td><td>75.0</td><td></td><td>52.1</td><td>66.7 50.0</td><td></td><td>40.6</td><td>43.8</td><td>40.2</td><td>47.7 53.7</td></tr><tr><td>SigLIP2-So400m/1460.4</td><td></td><td>37.5</td><td>66.7</td><td>33.3</td><td></td><td>8.3</td><td>45.8</td><td>41.7</td><td>12.5 9.4</td><td>46.9</td><td>30.3 35.5</td><td>37.3</td></tr><tr><td>DINOv2 + DINOv3</td><td>96.9</td><td>59.4</td><td>80.2</td><td>77.1</td><td></td><td>54.2</td><td>66.7</td><td>66.7</td><td>43.8</td><td>34.4 43.8</td><td>41.5</td><td>63.0</td></tr><tr><td>Table 1 (reference)</td><td>95.8</td><td>56.2</td><td>81.2</td><td>75.0</td><td></td><td>52.1</td><td>66.7</td><td>58.3</td><td>40.6</td><td>46.9</td><td>40.2</td><td>61.3</td></tr></table>

Table D: Vision encoder ablations. We compare frozen vision encoders for JEPA-WMs and LaLa-World. DINOv2 + DINOv3 uses both DINOv2-S/14 and DINOv3-L/16. We report success rates (%), except for the DROID action score. Average is the unweighted mean of the ten benchmark scores. Bold denotes the highest score among encoder configurations within each method; reference rows reproduce the main planning results and are excluded from this comparison.
<table><tr><td>Language action</td><td>Wall</td><td>PointMaze</td><td>Push-T</td><td>MetaWorld-R</td><td>MetaWorld-RW</td><td>RoboCasa-R</td><td>RoboCasa-P</td></tr><tr><td>Qualitative</td><td>83.3 (3.8)</td><td>47.9 (5.1)</td><td>4.2 (2.0)</td><td>70.8 (6.6)</td><td>60.4 (7.1)</td><td>28.1 (7.9)</td><td>25.0 (7.7)</td></tr><tr><td>Numerical</td><td>95.8 (2.0)</td><td>56.2 (5.1)</td><td>81.2 (4.0)</td><td>75.0 (6.2)</td><td>52.1 (7.2)</td><td>40.6 (8.7)</td><td>46.9 (8.8)</td></tr></table>

Table E: MPC planning with numerical and qualitative language actions. Qualitative language actions replace numerical magnitudes with slightly, no modifier, or far. Lower success rates are expected when exact magnitudes are omitted, yet it is still interesting that LaLa-World can plan successfully in several domains from qualitative language actions. Only the language conditions supplied to the predictor change; CEM samples numerical actions, and the simulator executes the selected actions in their original numerical form.

## C.7 Additional Results on Action Phrasing Robustness

We provide the domain-level results underlying Figure 4 and the corresponding action template paraphrases. We compare predictors trained on control trajectories alone and on all data, including image and video transitions. At evaluation, we apply the substitutions in Table F (top) to the language conditions supplied to the predictor, preserving the numerical action values and directions. For each predictor, we keep the model weights and planning settings unchanged. Table F shows that paraphrasing lowers scores in all five domains when training uses control trajectories alone, reducing the average from 59.3 to 52.5. These results suggest that image and video transitions reduce the predictor's dependence on the wording of the action templates.

## C.8 Additional Results on Latent vs. Generative World Models

Figure H breaks down the planning results in Figure 5 by domain. We use the full-depth LLM predictor for LaLa-World and the full-depth DiT for both RAEv2 variants. LaLa-World outperforms both RAEv2 variants on six of the seven benchmarks and matches five-step RAEv2 on MetaWorld-R. Its average score is 59.4, compared with 40.4 and 48.0 for one- and five-step RAEv2, respectively. These results follow the trend in Section 4.2, showing that the planning advantage of LaLa-World extends across domains.

## C.9 Additional Qualitative Planning Results

## C.9.1 Planning with Numerical Language Actions

We provide additional visualizations of LaLa-World with numerical language actions on Wall PointMaze, Push-T, MetaWorld, DROID, and RoboCasa Reach and Place in Figure M, Figure N, Figure O, Figure P, Figure Q, Figure R, and Figure S. The figures show initial and goal observations

<table><tr><td>Phrasing</td><td>Wall</td><td>PointMaze</td><td>MetaWorld-R</td><td>RoboCasa-R / DROID</td></tr><tr><td>Original</td><td>move the agent</td><td>push the ball with force</td><td>move the gripper</td><td>move gripper, rotate roll</td></tr><tr><td>Rephrased</td><td>push the dot</td><td>push the sphere</td><td>shift the robot hand</td><td>move robot hand, turn roll</td></tr></table>

<table><tr><td>Method</td><td>Wall</td><td>PointMaze</td><td>MetaWorld-R</td><td>RoboCasa-R</td><td>DROID</td><td>Average</td></tr><tr><td>Control data only</td><td>96.9 (1.8)</td><td>64.6 (4.9)</td><td>68.8 (6.7)</td><td>25.0 (7.7)</td><td>41.5 (4.6)</td><td>59.3 (2.5)</td></tr><tr><td>+ Paraphrasing</td><td>87.5 (3.4)</td><td>56.2 (5.1)</td><td>60.4 (7.1)</td><td>18.8 (6.9)</td><td>39.8 (5.0)</td><td>52.5 (2.5)</td></tr><tr><td>All data</td><td>95.8 (2.0)</td><td>56.2 (5.1)</td><td>75.0 (6.2)</td><td>40.6 (8.7)</td><td>40.2 (5.2)</td><td>61.6 (2.6)</td></tr><tr><td>+ Paraphrasing</td><td>92.7 (2.7)</td><td>57.3 (5.0)</td><td>72.9 (6.4)</td><td>43.8 (8.8)</td><td>38.3 (5.3)</td><td>61.0 (2.7)</td></tr></table>

Table F: Robustness to action phrasing. We report planning results with original and paraphrased action templates (bottom) and the substitutions used at evaluation (top). All data combines control trajectories with image and video transitions. We report success rates (%), except for the DROID action score, with standard errors in parentheses.

Latent vs. Generative World Models on MPC Planning  
![](images/fe94f0c64f8bd0c6685b3fbcd9fca227f36e33dce867b107907c7c011cfa5927.jpg)  
Figure H: Domain-level comparison of latent and generative world models for MPC planning. We compare LaLa-World with RAEv2 using one and five denoising steps under the matched setup in Section 4.2. We report success rates (%) for simulation tasks and the action score for DROID.

alongside decoded future predictions, with the language actions below the frames; Wall also includes PCA visualizations of the predicted latents.

## C.9.2 Planning with Qualitative Language Actions

We provide additional planning examples with qualitative language actions on Wall, PointMaze, Push-T, MetaWorld, and RoboCasa Reach and Place in Figure T, Figure U, Figure V, Figure W, Figure X, and Figure Y. Each example shows simulator observations during MPC execution, with qualitative action descriptions below the frames; action conversion and execution follow Section D.4.3.

## C.10 Qualitative Policy Results

We provide additional LaLa-Policy rollouts on LIBERO Spatial (Figure Z, Figure AA), Object (Figure AB, Figure AC), Goal (Figure AD, Figure AE), and Long (Figure AF, Figure AG). RoboCasa examples cover coffee tasks (Figure AH, Figure AI), opening doors and drawers (Figure AJ, Figure AK), object transfer (Figure AL, Figure AM), and stove control (Figure AN). For each camera view, simulator observations (upper row) are paired with decoded future predictions (lower row) during policy execution.

## D Implementation and Experimental Details

## D.1 LaLa-World Training Data Details

Table G reports the training data size for each dataset. We combine these datasets and sample in proportion to their sizes. We reserve approximately 2% of episodes or editing pairs for validation and keep sequences from the same source video in one split.

<table><tr><td>Dataset Training data size</td></tr><tr><td>Wall 86,296</td></tr><tr><td>PointMaze 186,675</td></tr><tr><td>Push-T 2,199,488</td></tr><tr><td>MetaWorld 1,162,310</td></tr><tr><td>Rope 18,525</td></tr><tr><td>Granular 18,582</td></tr><tr><td>DROID 1,500,000</td></tr><tr><td>OpenVid-1M 1,023,363</td></tr><tr><td>OmniEdit 295,353</td></tr><tr><td>ShareGPT-4o-Image 45,613</td></tr></table>

Table G: Data used to train LaLa-World. We count target transitions for control and video data and editing pairs for OmniEdit and ShareGPT-4o-Image. Counts exclude validation data and count each control transition once across its two action description formats.

Simulation data. We use the six simulation datasets from DINO-WM and JEPA-WMs (Zhou et al., 2025; Terver et al., 2025) listed in Section 4. For Wall, PointMaze, Push-T, and MetaWorld, each transition spans five simulator steps; for Rope and Granular, it spans one push. We enumerate overlapping transitions within each episode and pair each with both numerical and qualitative action descriptions. Action aggregation is described in Section D.4.2.

DROID. We use the LeRobot release of DROID (Khazatsky et al., 2024), retaining one external camera view and every fourth frame of the 15-fps recordings.3 We derive action descriptions from changes in end-effector position, orientation, and gripper state between retained frames. Our DROID subset contains 1.5 million target transitions, approximately 20% of the original dataset's transitions. This keeps DROID from dominating the mixture under a fixed training budget. The same subset is used for both action description formats.

Image and video data. From OpenVid-1M (Nan et al., 2025), we select clips by duration, motion, and aesthetic scores and extract ten-frame sequences at 0.5-second intervals. Qwen3-VL-30B-A3B-Instruct describes the visible change between each adjacent frame pair to provide its language condition. We also use source-edited image pairs and their instructions from OmniEdit (Wei et al., 2025) and ShareGPT-4o-Image (Chen et al., 2025b).

Further DROID-focused post-training, as in V-JEPA 2-AC (Assran et al., 2025), may improve robot planning; we leave this to future work.

## D.2 LaLa-World Architecture Details

Each example contains observed context frames, language conditions for the corresponding transitions, and one target frame. In multi-frame training, we sample the context length uniformly from one to three frames, subject to the available episode history; image pairs provide one context frame. A two-layer MLP projects each frame's patch representations into the language model's embedding space. As illustrated in Figure G(a), we append one query token per target patch after the context and language tokens. The queries are copies of a shared learned vector, with their target locations specified by 3D positional embeddings. These embeddings encode a temporal index and the row and column of each spatial patch. Context patches share a temporal index within each frame while queries receive the target frame's temporal index and spatial coordinates. We concatenate sinusoidal encodings for the three axes, multiply them by a learnable scale, and add them to the input embeddings while retaining the backbone's native positional encoding. The appended queries receive the preceding visual and language context through the LLM backbone. A two-layer MLP prediction head maps their output hidden states back to the vision encoder's feature dimension, producing the future patch grid in one forward pass. Both projection MLPs use GELU between their linear layers. The target frame is used only to provide supervision for these predictions.

<table><tr><td>Benchmark</td><td>Evaluation trajectories</td><td>Trajectory length</td><td>Planning horizon H</td><td>Executed steps per plan</td><td>Substeps per step</td><td>CEM settings</td><td>Evaluation criterion</td></tr><tr><td>Wall</td><td>96</td><td>6</td><td>6</td><td>6</td><td>5</td><td> $3 0 \times 3 0 0 \times 1 0$ </td><td>Position error &lt; 4.5</td></tr><tr><td>PointMaze</td><td>96</td><td>6</td><td>6</td><td>6</td><td>5</td><td> $3 0 \times 3 0 0 \times 1 0$ </td><td>Position error  $< 0 . 5$ </td></tr><tr><td>Push-T</td><td>96</td><td>6</td><td>6</td><td>6</td><td>5</td><td> $3 0 \times 3 0 0 \times 1 0$ </td><td>Position error &lt; 20 and angle error &lt; 20°</td></tr><tr><td>MetaWorld-R</td><td>48</td><td>20</td><td>6</td><td>3</td><td>5</td><td> $1 5 \times 3 0 0 \times 1 0$ </td><td>Success at any step</td></tr><tr><td>MetaWorld-RW</td><td>48</td><td>20</td><td>6</td><td>3</td><td>5</td><td> $1 5 \times 3 0 0 \times 1 0$ </td><td>Success at any step</td></tr><tr><td>Rope</td><td>24</td><td>5</td><td>5</td><td>1</td><td>1</td><td> $3 0 \times 3 0 0 \times 3 0$ </td><td>Chamfer distance &lt; 0.675</td></tr><tr><td>Granular</td><td>12</td><td>5</td><td>5</td><td>1</td><td>1</td><td> $3 0 \times 3 0 0 \times 3 0$ </td><td>Chamfer distance &lt; 0.323</td></tr><tr><td>DROID</td><td>64</td><td>3</td><td>3</td><td>一</td><td>1</td><td> $1 5 \times 3 0 0 \times 1 0$ </td><td>Action Score</td></tr><tr><td>RoboCasa-R</td><td>32</td><td>12</td><td>3</td><td>1</td><td>5</td><td> $1 5 \times 3 0 0 \times 1 0$ </td><td>Subtask success at final step</td></tr><tr><td>RoboCasa-P</td><td>32</td><td>12</td><td>3</td><td>1</td><td>5</td><td> $1 5 \times 3 0 0 \times 1 0$ </td><td>Subtask success at final step</td></tr></table>

Table H: MPC planning settings and evaluation criteria. One step corresponds to one worldmodel transition, with the listed number of action updates (substeps). CEM settings are iterations × candidate sequences per iteration × retained sequences. For benchmarks requiring replanning, we execute the listed number of steps, observe the environment, and re-optimize the action sequence with CEM. Position and angle errors are measured at the final step, using $L _ { 2 }$ distance for position. DROID is evaluated offline without simulator execution.

## D.3 LaLa-World Training Details

We keep the observation encoder frozen and jointly train the language-model backbone, visual projection layers, and learned query tokens. The same encoder provides the context and target representations. We optimize the prediction objective in Section 3.2. We use AdamW with an effective global batch size of $5 1 2 , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 8 )$ , and weight decay of 0.05. The learning rate increases linearly from $1 \times 1 0 ^ { - 5 } \mathrm { t o } 2 \times 1 0 ^ { - 4 }$ over the first 1,000 optimizer steps, followed by a cosine decay schedule with a minimum learning rate of $1 \times 1 0 ^ { - 6 }$ . We clip the gradient norm at 1.0 and use gradient accumulation to maintain the global batch size. We train for 20 epochs using bfloat16 mixed precision on eight NVIDIA H100 80GB GPUs. We define each epoch as approximately one million training examples sampled from the training pool.

## D.4 MPC Planning and Evaluation Details

## D.4.1 Benchmark Settings and Evaluation Metrics

We follow the planning and evaluation protocols of JEPA-WMs (Terver et al., 2025) and DINO-WM (Zhou et al., 2025), with our own success criteria for Rope and Granular. As in Section 4, we evaluate planning with model predictive control (MPC), using the cross-entropy method (CEM) to optimize action sequences. Table H summarizes the MPC settings, CEM parameters, and evaluation criteria for each benchmark. One planning step corresponds to one transition predicted by the world model. In simulation, each action update within a step is called a substep. The planning horizon H specifies how many steps the model predicts ahead during each candidate rollout, while trajectory length gives the total step budget

In simulation, we execute an initial segment of the sequence optimized by CEM, whose length is specified by executed steps. For benchmarks requiring replanning, we follow the MPC loop: after executing this segment, we obtain a new observation and re-optimize the action sequence with CEM. Figure I illustrates these settings. Wall executes all six planned steps in one round, whereas MetaWorld Reach executes three of six and Rope and Granular execute one of five before replanning.

For DROID, we evaluate planned actions offline without simulator execution. We compute the endpoint position error $\mathrm { e r r o r } _ { \mathrm { x y z } }$ in meters from the cumulative planned displacement and report the Action Score:

$$
\mathrm { A c t i o n ~ S c o r e } = \operatorname* { m a x } \left( 0 , 8 0 0 \left( 0 . 1 - \mathrm { e r r o r } _ { \mathrm { x y z } } \right) \right) .\tag{5}
$$

For zero-shot planning on RoboCasa (Nasiriany et al., 2024), we reuse DROID's action sampling statistics because the two domains share the same action definitions and dimensionality.

![](images/c7976b53332186b8e62f712b5f1ba70ebe7333a74c476ba124c260a7b6b1b2d0.jpg)

Figure I: MPC planning horizons and action execution. One step corresponds to one world-model transition, with the listed number of action updates (substeps). For benchmarks requiring replanning, we obtain a new observation after executing the prefix and optimize a new sequence with CEM.  
![](images/c98e5b2f24d465d7959bd482a56c0ae1e6fec37b7b45625706d8d5304a5969b3.jpg)  
Figure J: Substep aggregation for language conditioning and action execution. (a) Summing across substeps reduces the action dimension per step from 10 to 2 for language conditioning. (b) Within MPC, CEM optimizes these sums, which are distributed equally across five substeps for execution. This discards within-step variation. Yet, it is notable that LaLa-World achieves accurate prediction and high planning success rates.

## D.4.2 Action Aggregation and Execution

The baselines represent each planning step by concatenating the actions across its substeps. For example, five substeps with two action components, x and y, yield a 10-dimensional vector. Verbalizing every component would produce a long language condition. We therefore sum each component across substeps when constructing the training data and verbalize only these sums. In this example, the language condition describes two values, $\textstyle { \bar { x } } = \sum _ { i = 1 } ^ { 5 } x _ { i }$ and $\textstyle { \bar { y } } = \sum _ { i = 1 } ^ { 5 } y _ { i }$ , as shown in Figure J(a).

Within MPC, CEM optimizes action sequences in this aggregated action space, with its initial sampling statistics computed from the aggregated training data. Each sampled action is verbalized using the training templates to condition the predictor. To execute a selected action, we divide each optimized sum by the number of substeps and repeat the resulting action at every substep. For the five-substep example, we apply $( { \bar { x } } / 5 , { \bar { y } } / 5 )$ five times, as illustrated in Figure J(b). This preserves the component sums but discards variation within a step, which can lead to differences between predicted and executed transitions. It is notable that LaLa-World achieves accurate future prediction and high planning success rates despite this loss of information.

## D.4.3 Qualitative Language Action Conversion

Figure K illustrates the conversion rule and the planning procedure.

Action conversion. Numerical language actions preserve each action component's direction and magnitude. To construct qualitative descriptions, we define a conversion rule for each component and direction using the action range observed in the corresponding training data. We divide the interval from zero to the maximum observed magnitude into three equal-width bins, represented by slightly, no modifier, and far, respectively. The direction remains unchanged. For example, when the maximum leftward magnitude is 6, move 0.4 left becomes move slightly left, and move 4.5 left becomes move far left.

![](images/f24190f59de986716f5a5f1c6f6cbecd4a9d0a1167cb59cfba3d2a083fa777e7.jpg)  
Figure K: MPC planning with qualitative language actions. (a) Each magnitude maps to one of three equal-width bins, and the direction is kept. (b) LaLa-World evaluates CEM samples through their qualitative descriptions, while the simulator executes the selected actions with their numerical values. Thus, LaLa-World plans without exact action magnitudes.

MPC planning procedure. At each MPC planning round, CEM samples numerical action sequences as in the main experiments. Before each sequence is evaluated by LaLa-World, the conversion rule maps its actions to qualitative descriptions. These descriptions condition the latent rollout, which CEM scores using the same goal-distance cost in Equation (2). The simulator executes the selected actions using their original numerical values, and re-planning follows the same MPC procedure. We keep the trained predictor and all other planning settings unchanged.

## D.5 RAEv2 Baseline Details

Model and initialization. We initialize the generative world model baseline's DiT from the released RAEv2 checkpoint pretrained for image generation on ImageNet.4 Its frozen DINOv3-L/16 encoder aggregates features from seven layers to form the visual representation $e _ { t } = E _ { \theta } ( x _ { t } )$ . A frozen Qwen3-0.6B encodes the language condition z into text embeddings T(z).

Training. For each transition $\left( x _ { t } , z , x _ { t + 1 } \right)$ , we encode the future observation as $e _ { t + 1 } = E _ { \theta } ( x _ { t + 1 } )$ and interpolate it with Gaussian noise for flow matching (Singh et al., 2026):

$$
e _ { t + 1 } ^ { ( \tau ) } = ( 1 - \tau ) e _ { t + 1 } + \tau \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) , \quad \tau \in [ 0 , 1 ] .\tag{6}
$$

Here, t indexes environment time and $\tau$ denotes the noise level. The DiT receives $e _ { t + 1 } ^ { ( \tau ) } , \tau$ , the current observation representation $e _ { t } .$ , and the text embeddings T(z). We train it with flow matching and a representation alignment (REPA) loss (Yu et al., 2025) that aligns intermediate DiT features with DINOv2-B/14 features of the future observation.

MPC planning. We use the same MPC procedure as LaLa-World, with CEM optimizing candidate action sequences. Starting from Gaussian noise, the DiT produces a future representation $\hat { e } _ { t + 1 }$ through denoising, conditioned on the current observation and language action. We evaluate RAEv2 with one and five denoising steps per prediction. CEM scores candidate action sequences using the mean squared error between the final predicted representation $\hat { e } _ { t + H }$ and the encoded goal $E _ { \theta } ( x _ { \mathrm { g o a l } } )$ following Equation (2). The goal is encoded with the same DINOv3-L/16 feature aggregation as the training targets. Pixel decoding is not used to compute the planning cost.

![](images/34c984490f836d7bf0ebfaa769e85240fadcbb03693dd5737dd11efeb06b047d.jpg)  
Figure L: LaLa-Policy architecture. The predictor, fine-tuned from LaLa-World, receives encoded observations, a task instruction, and learned action and future queries. It predicts future visual latents and provides conditioning features for an action generator, which produces an action chunk from Gaussian noise. Proprio and intermediate projections are omitted for clarity.

Visualization. For Figure 6, RAEv2's predicted representations are mapped to pixels with its released DINOv3-L-K7 decoder,⁵ while LaLa-World uses the decoder described in Section 4.

## D.6 LaLa-Policy Architecture Details

As illustrated in Figure L, LaLa-Policy extends the predictor of LaLa-World with proprioceptive input and action queries. Following previous VLAs that use SigLIP-based visual encoders (Black et al., 2025; Kim et al., 2025; Li et al., 2025b), we use SigLIP2-So400M/14 at 224 × 224 resolution and the 24-layer Qwen3.5-0.8B as the predictor backbone. Each camera view is encoded independently, and a shared two-layer MLP projects its patch representations into the backbone's embedding space. We add learned view embeddings and map the current proprioceptive state to one token through a separate MLP.

The input sequence contains visual tokens, the proprioceptive token, task instruction tokens, action queries, and future queries, in this order. We use N = 16 action queries for LIBERO and N = 32 for RoboCasa, with one future query per spatial patch in each camera view. A two-layer MLP maps the future-query outputs to the encoder's feature dimension to predict the representations at t + N. The action generator follows Isaac-GR00T (Bjorck et al., 2025), using 16 DiT blocks of width 1,536 with alternating cross-attention and self-attention. It generates the action chunk from Gaussian noise conditioned on the predictor's full sequence of output hidden states.

## D.7 Policy Model Training Data

We describe the datasets used to fine-tune LaLa-Policy in Section 5. The episode counts below refer to the data before the training/validation split.

LIBERO. We use the Spatial, Object, Goal, and Long suites of LIBERO (Liu et al., 2023), comprising 40 language-conditioned manipulation tasks. The original data contain 50 human demonstrations per task, for a total of 2,000 episodes. We adopt the replayed datasets released by Cosmos Policy (Kim et al., 2026).6 Its preprocessing removes no-op transitions and filters unsuccessful replays from the policy demonstration set. The resulting set contains 1,700 successful episodes: 430, 456, 438, and 376 from Spatial, Object, Goal, and Long, respectively. We use these episodes for policy training and the full set of 2,000 replays, including failures, for actionconditioned future prediction.

RoboCasa. We use 24 kitchen manipulation tasks from RoboCasa (Nasiriany et al., 2024) under two data settings. For the setting with approximately 50 human demonstrations per task, we use the Cosmos Policy release.7 It provides 1,199 successful episodes for policy training and 1,291 replay episodes, including 92 failures, for action-conditioned future prediction. For the 300-demonstration setting, we use RoboCasa's released data generated with MimicGen (Mandlekar et al., 2023).

## D.8 LaLa-Policy Fine-tuning Details

We initialize the predictor from pretrained LaLa-World and jointly train it with the newly added policy modules, while keeping the observation encoder frozen. The action generator is trained from scratch. We minimize a flow-matching velocity loss on policy examples and a mean squared error loss on future representations for both types of examples. Both loss coefficients are set to 1.

We use AdamW with an effective global batch size of 256, $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 8 )$ , and weight decay of 0.05. The learning rate increases linearly from $1 \times 1 0 ^ { - 6 }$ to $1 \times 1 0 ^ { - 4 }$ over the first 1,000 optimizer steps, then follows a cosine decay to $1 \times \mathrm { 1 0 ^ { - 6 } }$ . We clip the gradient norm at 1.0 and use bfloat16 mixed precision on eight NVIDIA H100 80GB GPUs, with gradient accumulation to maintain the global batch size. We train for 8 epochs on LIBERO, 12 epochs on RoboCasa with approximately 50 human demonstrations per task, and 3 epochs on RoboCasa with 300 MimicGen demonstrations per task.

## D.9 Policy Model Evaluation Details

We evaluate LaLa-Policy through direct interaction with the LIBERO (Liu et al., 2023) and Robo-Casa (Nasiriany et al., 2024) simulators, following the evaluation procedure of Cosmos Policy (Kim et al., 2026). The policy receives the task instruction, current proprioceptive state, and camera observations. LIBERO provides an external camera view and a wrist view, while RoboCasa provides left and right external views and a wrist view.

After 10 initial no-op steps, the policy predicts an action chunk. On LIBERO, the chunk contains 16 actions, all of which are executed before the next query. On RoboCasa, the chunk contains 32 actions, of which we execute the first 16. Each subsequent query uses the latest simulator observations. The predicted future representations are not used for action selection during evaluation.

We run 50 episodes for each of the 40 tasks across LIBERO's Spatial, Object, Goal, and Long suites. For RoboCasa, we run 50 episodes for each of 24 tasks, distributed equally across five test scenes with held-out object instances from split B. The step budgets follow the Cosmos Policy evaluation code: 220, 280, 300, and 520 for the four LIBERO suites, and 300–1,000 depending on the RoboCasa task. An episode ends upon task success or exhaustion of its step budget. We report success rates averaged across suites for LIBERO and across tasks for RoboCasa.

![](images/408e221ed2905210cf9746cbe70b35a7b14d1c5841d41d9c9d1db56f41255e9e.jpg)

![](images/7ed6bbf3c0b4eab1a27e093886e6eca2cde5d80d43d5df1ac594be16437a91dc.jpg)

![](images/441dd4c2235dbbab968831b19f25b64c91f860c4b3f00f6674665788e650fe23.jpg)

![](images/9d0422324eba1e3462884482e29128fe03669b54ceb528fd6f1d394d4b327742.jpg)  
Figure M: Numerical language actions on Wall. Initial and goal observations bracket decoded future predictions (top) and PCA visualizations of predicted latents (bottom), with the selected language actions below each frame.

![](images/f238445409ddb8541689779f0d0d292d70424e391f33a751a60527670ad0b05d.jpg)  
Figure N: Numerical language actions on PointMaze. Each row shows initial and goal observations and decoded future predictions under the selected actions, which specify pushing forces and directions.

![](images/8264087bbed8145e5e0d8cefe4799ac1f2ccdad7ce554b93780599149f6b3ee3.jpg)  
Figure O: Numerical language actions on Push-T. Initial and goal observations bracket decoded future predictions under the selected pusher movements, with numerical displacements described below each frame.

![](images/75c5e346145a073e98255ad670253ecd57dfab7f4f1146ad4baca070cfb3d498.jpg)  
Figure P: Numerical language actions on MetaWorld. Decoded predictions are shown before and after replanning from the simulator observation $x _ { t + 3 \Delta }$ , alongside initial and goal observations.

![](images/569aefc256e310622bc3f195cb2d4cec4ed58201ad09a4874b9978fad4c05398.jpg)  
Figure Q: Numerical language actions on DROID. Each row shows an initial observation, three decoded future predictions, and the goal observation, with language actions below.

![](images/74999c357d62cccf9f08d5a94d96732223c92c32361f513f4a976756da1695f8.jpg)  
Figure R: Numerical language actions on RoboCasa Reach. Decoded future predictions under selected gripper actions are shown between initial and goal observations. RoboCasa is excluded from world-model training.

![](images/65afac8bfe8f67c20b61c2d9e27a60d6471059d98642ac1eef161ce7e4915df4.jpg)  
Figure S: Numerical language actions on RoboCasa Place. Initial and goal observations bracket decoded future predictions during object placement, with the selected gripper actions below each frame.

![](images/4663eed0b3c11eea02acfbe159b6e04ae654bbab688bc0dc5cff13683cf54ba7.jpg)  
Figure T: Qualitative language actions on Wall. Simulator observations show the agent passing through the opening toward the goal. Actions describe directions and relative movement magnitudes.

![](images/a5efbc38db66635e33eccdc51c198d7a85fef77a6df6cf3e160391ea479dd165.jpg)  
Figure U: Qualitative language actions on PointMaze. Simulator observations show the ball moving through the maze under MPC, with pushing directions and qualitative force descriptions below the frames.

![](images/2f54a3e306737fc901856c1546f5c73e23d96b2bf87e15599935229b4fd5e22e.jpg)  
Figure V: Qualitative language actions on Push-T. Initial and goal observations bracket simulator states during MPC, with the pusher's direction and relative displacement described below each frame.

![](images/ff38552bc3a07a53b9a941683f2757103aa6651ea86be4b53a9f6b377d4abad6.jpg)  
Figure W: Qualitative language actions on MetaWorld. Simulator observations show reaching trajectories planned using descriptions of gripper motion and grip adjustment without exact numerical magnitudes.

![](images/3d27d82812d731ce82f816c5c83cb1a93b6c817af0547cfe016e659a1880a62f.jpg)

Figure X: Qualitative language actions on RoboCasa Reach. Simulator observations show zeroshot MPC execution toward the goal, conditioned on qualitative descriptions of gripper translation, rotation, and opening or closing.  
![](images/c4b3cf42cab2bb001ca69d227a3b387127f70e6c0da0dbe638f2d04af487b8d3.jpg)  
Figure Y: Qualitative language actions on RoboCasa Place. Simulator observations show an object-placement trajectory under MPC, with qualitative gripper actions and the goal observation for reference.

LaLa-Policy : “pick up the black bowl on the ramekin and place it on the plate" task05· ep03· 90 steps  
![](images/790c3debd49fef2f5d78eb58c484755359d8b58f52560e2d1af820e2d11cb564.jpg)  
Figure Z: LIBERO Spatial: bowl on a ramekin. LaLa-Policy moves the bowl from the ramekin onto the plate. Primary and wrist views compare simulator observations (upper rows) with decoded future predictions (lower rows).

LaLa-Policy . “pick up the black bowl next to the cookie box and place it on the plate" task06 · ep00 · 105 steps  
![](images/8cf9d7795bfdb1198d1dde0b260e1dfae9966c7a11d90c262222f3c979c5943a.jpg)  
Figure AA: LIBERO Spatial: bowl next to a cookie box. LaLa-Policy moves the specified bowl onto the plate. Simulator observations and decoded future predictions are paired for the primary and wrist views.

LaLa-Policy  “pick up the alphabet soup and place it in the basket" task00 · ep00 · 146 steps  
![](images/fe943b0e020c35138d5c9780d71a0b631601f27d76988a291458ccf92eb1b336.jpg)  
Figure AB: LIBERO Object: alphabet soup. LaLa-Policy picks up the alphabet soup and places it in the basket. Each camera view pairs observed states with decoded future predictions.

LaLa-Policy . “pick up the salad dressing and place it in the basket" task02 · ep00 · 114 steps  
![](images/5b38eea033515621bf3a7320dc4ce7ecac472a167801e7ad8c852cfeb3294869.jpg)  
Figure AC: LIBERO Object: salad dressing. LaLa-Policy transfers the salad dressing to the basket. Primary and wrist views show simulator observations above the corresponding decoded future predictions.

LaLa-Policy : “put the bowl on top of the cabinet" task04· ep01· 86 steps  
![](images/256665dfe54a6ed800b86be1d544847bfdc5afa52fb75884982ff34f600a35d4.jpg)  
Figure AD: LIBERO Goal: bowl on the cabinet. LaLa-Policy places the bowl on top of the cabinet. Primary and wrist views compare observed states with decoded future predictions throughout execution.

LaLa-Policy  “open the top drawer and put the bowl inside" task03 · ep02 · 172 steps  
![](images/64039a2890c3f425a6851edda43b13d6af6fe1b1e7bc9978393bc4810f4a1acf.jpg)  
Figure AE: LIBERO Goal: bowl inside the drawer. LaLa-Policy opens the top drawer and places the bowl inside. Both camera views pair simulator observations with decoded future predictions.

LaLa-Policy . “put both the cream cheese box and the butter in the basket" task01 · ep02 · 239 steps  
![](images/9d16cbab0854648c803cfe403904dfeef61cad32d84b1948d50bfde2d269afe5.jpg)  
Figure AF: LIBERO Long: placing two objects in a basket. LaLa-Policy transfers the cream cheese box and butter to the basket. Observations and decoded future predictions are shown for both camera views.

LaLa-Policy  “turn on the stove and put the moka pot on it" task02 · ep01 · 252 steps  
![](images/3d44fa5664b9fa5747d2be98ce92c74787c12596da22dd4cf84c30b7dd4a047b.jpg)  
Figure AG: LIBERO Long: stove and moka pot. LaLa-Policy turns on the stove and places the moka pot on it. Primary and wrist views compare simulator observations with decoded future predictions.

LaLa-Policy . “press the button on the coffee machine to serve coffee"  
![](images/9866ce1be23c13dd8d7f88cef72f10c49abd350dbf7c97df181c13bcf80a2de4.jpg)  
Figure AH: RoboCasa: pressing the coffee-machine button. Simulator observations (upper rows) and decoded future predictions (lower rows) show LaLa-Policy pressing the button, from two external views and the wrist view.

LaLa-Policy . “pick the mug from under the coffee machine dispenser and place it on the counter" CoffeeServeMug· ep20 · 294 steps  
![](images/e9bd78e5bff2947edaeb5ecc571c1c4c1ab49551a8395a6d183bd4cec71d9c67.jpg)  
Figure AI: RoboCasa: serving a mug. LaLa-Policy moves the mug from beneath the coffeemachine dispenser to the counter. Each of the three views compares observed states with decoded future predictions.

LaLa-Policy  “open the microwave door" OpenSingleDoor · ep00 · 226 steps  
![](images/57819a947e4b07b5b46f3e76d076c8b6f516e33366a209beb03cca77657e04ad.jpg)  
Figure AJ: RoboCasa: opening a microwave. Simulator observations and decoded future predictions show LaLa-Policy opening the microwave door, from the primary, secondary, and wrist views.

LaLa-Policy . “open the left drawer" OpenDrawer · ep10 · 232 steps  
![](images/16b5e91b72ffcd1e54c7da2c89313c8c347eefba98acbef96a12947c8de6be2e.jpg)  
Figure AK: RoboCasa: opening a drawer. LaLa-Policy opens the left drawer. The three camera views pair simulator observations with decoded future predictions during reaching and drawer opening.

LaLa-Policy  “pick the bar from the counter and place it in the cabinet" PnPCounterToCab · ep30 · 233 steps  
![](images/5844b4f155e57fe15b1863cbd21657f7d5624b564a44e38e724f80e44d1f4159.jpg)  
Figure AL: RoboCasa: counter-to-cabinet transfer. LaLa-Policy picks up the bar from the counter and places it in the cabinet. Each camera view compares observed states with decoded future predictions.

LaLa-Policy . “pick the cheese from the pan and place it on the plate" PnPStoveToCounter · ep00 · 238 steps  
![](images/0b2e02681d277c8db89f696e44ce5cd7c5d44af38039bbfe5fc84e39d3c710dc.jpg)  
Figure AM: RoboCasa: pan-to-plate transfer. LaLa-Policy moves the cheese from the pan onto the plate. Observations and decoded future predictions are shown for two external views and the wrist view.

LaLa-Policy . "turn on the front left burner of the stove" TurnOnStove·ep30· 127 steps  
![](images/13df13db89a5b6c1739bbdc52c69aa9ac3980fbdc5530020ebdd106d4df5e7cb.jpg)  
Figure AN: RoboCasa: turning on a stove burner. Simulator observations and decoded future predictions show LaLa-Policy turning on the front-left burner, from the primary, secondary, and wrist views.