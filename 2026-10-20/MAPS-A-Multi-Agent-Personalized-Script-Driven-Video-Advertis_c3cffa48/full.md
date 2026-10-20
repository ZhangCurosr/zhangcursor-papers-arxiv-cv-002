# MAPS: A Multi-Agent Personalized Script-Driven Video Advertising System Enhanced by Reinforcement Learning

Zhongyu Wang<sup>1,2,\*</sup> Chen Zuo<sup>2,\*</sup> Qingxuan Sun<sup>2,\*</sup> Jiayang Qiao<sup>2</sup> Qiuhao Qian<sup>2,†</sup> Songyan Cui<sup>2</sup> Guocheng Chen<sup>2</sup>

<sup>1</sup>Beihang University <sup>2</sup>Kuaishou Technology

## Abstract

With the rapid growth ofonline video consumption, video advertising has become a dominant component of digital advertising. Most personalized video advertising systems follow a retrieval-based paradigm, which matches ad content with user preferences in a feature space and recommends the most similar candidates. However, since the candidate creatives are pre-produced, this paradigm limits the granularity of personalization and leads to performance disparities across diverse user groups. To address this limitation, we propose a multi-agent personalized script-driven video advertising system (MAPS). We design four collaborative agents thatform a closed-loop pipeline ofuser profile modeling, script generation, video synthesis, andfeedbackdriven iterative refinement. To train the Scripter Agent, we develop a reinforcement learning framework equipped with a multi-dimensional rewardfunction, which steers the agent toward producing scripts that simultaneously align with user content preferences and target product information. Extensive offline experiments on representative datasets demonstrate that our approach achieves encouraging performance in generating and integrating personalized creatives. Online A/B experiments in a specific scenario in the Kuaishou advertising system show that MAPS improves CTR by 2.87%, CVR by 2.74%, and ARPU by 9.85% over the generative baseline.

## 1. Introduction

“Once you crack the script, everything elsefollows.”

— Ridley Scott

As the financial backbone of the modern internet ecosystem, digital advertising serves as the primary revenue engine for social media and content platforms [16, 21, 29].

![](images/e57baa93ac7aab4c54d7d072d370e0bd7d9313fb012becb802f3bf61273f2e13.jpg)  
Figure 1. Comparison between our MAPS and existing digital advertising recommendation systems.

Compared to text-based and static image advertisements, video enables richer narrative expression and an enhanced sense of immersion, fostering deeper emotional resonance and stronger user interaction [36, 48, 59]. Consequently, video advertising has emerged as one of the fastest-growing segments within the digital advertising landscape. Over the past several decades, advertising optimization research has been primarily concentrated on targeting, allocation, and bidding mechanisms [19, 22, 52]. As these techniques approach a performance plateau, the next frontier of efficacy improvement has shifted toward the creative itself. Creative optimization grounded in user characteristics and contextual information is increasingly recognized as the pivotal determinant of advertising effectiveness [45, 49, 53, 55].

Nevertheless, video creatives are costly artifacts produced by specialized teams, characterized by limited output and extended production cycles, which collectively constrain both the quantity and diversity of deployable creatives [11, 25]. Under these constraints, the industry has converged on a retrieval-based paradigm: advertisers typically commission professional personnel to pre-produce a limited pool of candidate video creatives or modular assets targeting specific products; at serving time, the most suitable variant is retrieved to match user preferences and contextual information [26, 37, 56]. Despite its practical utility, this paradigm suffers from two fundamental limitations. First, static inventory constraints: the static and finite nature of the creative repository renders it inadequate to accommodate rapidly evolving user preferences, thereby compromising user resonance and overall advertising effectiveness. Second, feedback latency: user feedback signals can only inform subsequent production cycles and offer no mechanism for real-time optimization of creatives currently in deployment, introducing a significant latency between performance signals and creative production.

To address these limitations, we propose MAPS, a multiagent personalized script-driven video advertising system. MAPS consists of four collaborative agents. The Profile Agent constructs structured user profiles from histori cal interaction data and contextual information, including the video currently being consumed by the user, and continuously refines these profiles as new behavioral observations accumulate. The Scripter Agent generates personalized advertisement video scripts conditioned on user preferences, guided by target product information such as core selling points and available creative assets. The Director Agent synthesizes personalized candidate creatives from the generated scripts and target product information, ensuring that the outputs simultaneously align with individual user preferences and faithfully convey target product information. The Reflector Agent attributes online user interaction signals to profile-level and script-level causes and converts them into minimal revisions routed back to the Profile and Scripter Agents for iterative refinement. To train the Scripter Agent, we construct a reinforcement learning framework based on the Group Relative Policy Optimization (GRPO) [32] algorithm and design a multi-dimensional reward function that ensures structural output regularity while steering the generated scripts to align advertisement narratives with target product information and individual user preferences. Offline experiments on two representative tasks demonstrate that MAPS captures user preferences with high fidelity and generates high-quality, personalized, and native video creatives. Online A/B experiments in a specific scenario in the Kuaishou advertising system further demonstrate that MAPS delivers significant improvements in click-through rate (CTR), conversion rate (CVR), and average revenue per user (ARPU) over the generative baseline at industrial scale.

Our main contributions are summarized as follows:

• We propose MAPS, a multi-agent personalized scriptdriven video advertising system that fundamentally transforms personalized video advertising from static retrieval over a discrete creative inventory into dynamic generative optimization across a continuous creative space.

• We establish a closed-loop creative evolution mechanism through post-deployment online user feedback, enabling advertisement creatives to be iteratively refined across successive impressions in response to real-time user in-

teractions.

• We develop a reinforcement learning framework for the Scripter Agent, incorporating a multi-dimensional reward function that jointly steers the generated scripts to align with individual user preferences and faithfully reflect target product information.

• We demonstrate the effectiveness of MAPS through extensive offline experiments, and further validate its practical value at industrial scale through online A/B experiments in a specific scenario in the Kuaishou advertising system.

## 2. Related Work

## 2.1. Video Advertising

The evolution of video advertising can be characterized along two critical dimensions: personalization and nativeness [1, 24, 30]. Traditional advertising formats, such as billboards and broadcast television commercials, occupy the lower end of both dimensions, relying on static creatives and mass-reach distribution. Digital advertising, encompassing programmatic display, search, and retargeting, primarily advances personalization through audience targeting, yet typically achieves only limited integration between creatives and contextual content. Native advertising formats, including product placement, branded content marketing, and influencer marketing, attain a high degree of nativeness through narrative expression and seamless contextual integration, but remain constrained in their capacity for individualized personalization. Scaling creative-level personalization has proven particularly challenging due to prohibitive production costs and labor-intensive creative workflows [2, 10, 28]. Although dynamic creative optimization partially alleviates this tension, the substantial overhead of video production continues to impede the simultaneous realization of high personalization and high nativeness. To address this challenge, we propose MAPS, a multi-agent system that transforms personalized video advertising from static retrieval over a discrete creative inventory into dynamic generative optimization across a continuous creative space, simultaneously achieving high personalization and high nativeness within a unified framework.

## 2.2. Multi-Agent Systems in Personalized Video Advertising

Recent advances in generative artificial intelligence have made large-scale synthesis of advertisement creatives increasingly viable [5, 12, 14, 47, 54]. Prior work has leveraged diffusion models and large language models (LLMs) to generate personalized advertisement images and banner creatives, and has incorporated interaction-oriented objectives, such as click-through rate, to guide the generation process toward utility-driven optimization [6, 17, 20, 23,

![](images/1ae3c082ed3a4e76289c04536a0afff5746807110704e81bca2fd52ace84210c.jpg)  
Figure 2. Overview of the MAPS framework.

35, 38, 40, 43]. Multi-agent systems built upon multimodal large language models (MLLMs) have demonstrated promise in personalized advertising workflows, benefiting from the collaboration among specialized components [18, 33, 42, 44, 46]. However, existing works remain predominantly centered on static image creatives, while generative approaches to personalized video advertising have received less attention. To bridge this gap, MAPS introduces a script-driven, closed-loop multi-agent framework that positions the script as the central interface between planning and generation, and trains the Scripter Agent via reinforcement learning to jointly optimize structural regularity, user preference conformity, and product alignment.

## 3. Method

## 3.1. Architecture Overview

As illustrated in Figure 2, MAPS consists of four collaborative agents: a Profile Agent that extracts user preferences, a Scripter Agent that generates structured scripts, a Director Agent that produces personalized advertisement videos from the generated scripts, and a Reflector Agent that attributes online user feedback to profile-level and script-level causes and routes iterative refinements back to the Profile and Scripter Agents.

## 3.2. Profile Agent

The Profile Agent serves as the perceptual and modeling foundation of the MAPS framework. Its primary responsibility is to construct structured user profiles from heterogeneous historical interaction data and real-time contextual signals, and to continuously refine these profiles as new behavioral observations accumulate, thereby providing highquality user preference priors for the Scripter Agent’s subsequent personalized script generation.

## 3.2.1. Input.

The inputs to the Profile Agent encompass two categories of information. The first category comprises user historical interaction data, including viewing history, play duration, likes, comments, collections, shares, and follows. These multi-dimensional behavioral signals collectively characterize the user’s historical content consumption patterns. The second category consists of real-time contextual information, specifically the video content currently being consumed by the user, which serves as an immediate contextual anchor for preference inference. Both categories of information are jointly fed into the Profile Agent, which employs an MLLM as its backbone to drive the profile construction process.

## 3.2.2. User Profile Construction.

Based on the above inputs, the Profile Agent performs deep inference over user content consumption preferences and produces a structured user profile across four dimensions:

• Topical interests: the content domains and themes preferred by the user, such as gaming, travel, music, and fashion;

• Preferred presentation styles: the content expression styles favored by the user, such as humorous, informative, emotionally resonant, or narratively immersive;

• Implicit needs: latent needs revealed through the user’s content consumption behaviors, such as quality-of-life improvement, self-expression, and entertainment;

• Decision-making tendencies: behavioral preferences exhibited in consumption decisions, such as pricesensitivity, quality-orientation, or trend-following.

The resulting multi-dimensional profile is serialized into structured text and passed to the Scripter Agent as the user preference conditioning for script generation.

## 3.2.3. Dynamic Refinement.

The user profile is not a static artifact but evolves as behavioral observations continue to accumulate. Following each advertisement impression, the Reflector Agent attributes online user feedback to profile-level causes and routes the resulting revisions to the Profile Agent, which incorporates them into its update procedure to drive iterative refinement of the user profile. This dynamic update mechanism enables the system to responsively capture shifts in user preferences, thereby supporting closed-loop creative evolution across successive impressions over extended temporal horizons and facilitating progressive narrative development.

![](images/d6d7314c0979b0064e69dfb115e6b092feb2750878f86b6db72d642fd27fa40c.jpg)  
Figure 3. Overview of our reinforcement learning framework.

## 3.3. Scripter Agent

The Scripter Agent is the core creative generation module of the MAPS framework, responsible for transforming target product information into a structurally complete, personalized advertisement video script conditioned on individual user characteristics. It receives two streams of input: the structured user profile output by the Profile Agent, which encapsulates the user’s interest preferences, aesthetic inclinations, and consumption motivations; and the detailed attribute information of the target product, including product appearance, core selling points, and brand recognition elements. Conditioned on these two input streams, the Scripter Agent produces a fine-grained personalized advertisement script that fully specifies the script content, voiceover, onscreen text, visual descriptions, and pacing for each scene, and passes the resulting script to the Director Agent for subsequent video production. In addition, the Scripter Agent receives script-level revisions routed from the Reflector Agent and incorporates them to iteratively refine the script.

Although general-purpose MLLMs possess a certain baseline capability for script generation, crafting scripts for personalized short-form advertisement videos demands more specialized competencies: naturally weaving product information into narrative contexts aligned with user preferences, adapting tone and expressive style to individual user characteristics, and constructing emotionally resonant call-to-action (CTA) sequences that match user decisionmaking tendencies. To acquire these competencies, we adopt a two-stage training paradigm: a supervised finetuning (SFT) stage to establish the foundational capability for structured script generation, followed by a reinforcement learning (RL) stage to align model outputs with personalized quality objectives.

## 3.3.1. Stage One: Supervised Fine-Tuning for Structural Capability.

The SFT stage aims to teach the model the fundamental syntax and structural conventions of advertisement scripts within the MAPS framework. We formulate the task as a sequence-to-sequence generation problem: the input x consists of the structured user profile and target product attribute information, while the output y is the structured text of the target advertisement script. We fine-tune a base language model on a corpus D of human-authored personalized advertisement scripts, with the training objective of maximizing the conditional log-likelihood of the reference scripts:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ \sum _ { t = 1 } ^ { | y | } \log \pi _ { \theta } ( y _ { t } \mid y _ { < t } , x ) \right]\tag{1}
$$

The SFT cold start stage equips the model with the foundational ability to generate structurally complete scripts that are consistent with the user profile and product attribute information, providing a solid basis for the subsequent personalization alignment stage.

## 3.3.2. Stage Two: Reinforcement Learning for Personalization Alignment

While SFT cold start guarantees structural completeness and content correctness of the generated scripts, it falls short of capturing the fine-grained alignment quality required for personalized short-form advertisement creation, namely the tight correspondence between script content and the core information of the target product, as well as the deep alignment between narrative style and individual user preferences. To address this, we introduce a reinforcement learning framework using the GRPO algorithm as illustrated in Figure 3.

The RL stage employs a hybrid reward function $R _ { \mathrm { t o t a l } }$ composed of three complementary components:

• Rule-Based Structural Reward $( R _ { \mathbf { s t r u c t } } ) { : }$ This component provides an objective signal for the technical correctness of the generated script, evaluating and aggregating normalized scores across four automated checks: Format Compliance (whether the script conforms to the required structural format), Script Completeness (whether all required script components are fully present and accounted for), Scene and Character Consistency (whether scene arrangement and character settings remain coherent throughout), and Physical Rationality (whether character positions and camera geometry are plausible within realworld constraints).

• Personalization Reward $( R _ { \mathbf { p e r s } } ) { : }$ This component measures the degree to which the generated script resonates with the user profile, assessed via an LLM-as-a-Judge paradigm across two core dimensions: presentation style alignment and content preference alignment. For presentation style alignment, the LLM judge evaluates the script against the user’s inferred aesthetic preferences derived from their behavioral history, considering four aspects: tone (whether it matches the user’s preferred warm, cold, or neutral style), lighting treatment (whether it suits the user’s preference for soft, harsh, or natural lighting), textural style (whether it fits the user’s aesthetic inclination), and artistic style (whether it aligns with the user’s taste for minimalistic, modern, vintage, or other aesthetics). For content preference alignment, the LLM judge assesses subject matter relevance (whether the script theme matches the user’s interest domains), scene semantics (whether scene selection fits the user’s preferences), information angle (whether the product messaging focuses on aspects the user cares about, such as performance, aesthetics, or value), and motivational appeal (whether the ad’s appeal resonates with the user’s driving motivations such as quality, efficiency, or lifestyle). The LLM judge outputs a score on a scale of 0 to 10 for each dimension, which are normalized and aggregated to form $R _ { \mathrm { p e r s } }$

• Product Alignment Reward $( R _ { \mathbf { p r o d } } ) \mathrm { : }$ : This component measures the consistency between the product depicted in the generated script and the actual attributes of the target product, also assessed via an LLM-as-a-Judge paradigm. The LLM judge is provided with detailed product information alongside the generated script and evaluates three dimensions: product identity accuracy (whether the script correctly portrays the product’s appearance, color, model, packaging, and accessories), brand and recognition elements (whether the script accurately represents the product’s logo, brand visual language, and design style), and selling points and factual consistency (whether the product claims conveyed in the script accurately reflect the real product features without exaggeration or misrepresentation). The LLM judge outputs a score on a scale of 0 to 10, which is normalized to serve as the $R _ { \mathrm { p r o d } }$ signal.

The final reward function integrates all three components through a weighted combination:

$$
R _ { \mathrm { t o t a l } } ( y ) = \alpha \cdot R _ { \mathrm { s t r u c t } } ( y ) + \beta \cdot R _ { \mathrm { p e r s } } ( y ) + \gamma \cdot R _ { \mathrm { p r o d } } ( y )\tag{2}
$$

where $\alpha = 0 . 3 , \beta = 0 . 4 .$ , and $\gamma = 0 . 3$ . During optimization, for each input x, K candidate scripts $\{ y ^ { ( k ) } \} _ { k = 1 } ^ { K }$ are sampled from the current policy $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ , and their rewards $R _ { k } =$ $R _ { \mathrm { t o t a l } } ( y ^ { ( k ) } )$ are computed. A normalized advantage is then calculated within the group:

$$
A _ { k } = \frac { R _ { k } - \bar { R } } { \sigma _ { R } + \varepsilon } , \qquad \bar { R } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R _ { k }\tag{3}
$$

The policy is updated by maximizing the advantageweighted log-likelihood, constrained by a KL-divergence penalty to prevent large deviations from the SFT initialization:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { G R P O } } = \mathbb { E } _ { { \boldsymbol x } \sim \mathcal { D } } \Bigg [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } A _ { k } \cdot \log \pi _ { \boldsymbol \theta } \Big ( { \boldsymbol y } ^ { ( k ) } \mid { \boldsymbol x } \Big ) \Bigg ] } } \\ & { ~ - \beta _ { \mathrm { K L } } \cdot \mathbb { E } _ { { \boldsymbol x } \sim \mathcal { D } } [ \mathrm { K L } ( \pi _ { \boldsymbol \theta } ( \cdot \mid { \boldsymbol x } ) \parallel \pi _ { \mathrm { S F T } } ( \cdot \mid { \boldsymbol x } ) ) ] } \end{array}\tag{4}
$$

Through the above two-stage training paradigm, the capability of the Scripter Agent advances from generating structurally well-formed advertisement scripts to producing compelling personalized advertisement creatives that align tightly with the core information of the target product and match deeply with individual user preferences.

## 3.4. Director Agent

The Director Agent is the creative realization module of the MAPS framework, responsible for transforming the personalized advertisement script produced by the Scripter Agent into a deliverable video creative. Its inputs come from two sources: the structured advertisement script output by the Scripter Agent, which fully specifies the script content, voiceover, visual descriptions, and pacing for each scene; and the available visual assets of the target product, including product images and brand logos, which serve as visual grounding constraints to ensure that the generated content faithfully reflects the product’s visual characteristics and brand identity.

Upon receiving the above inputs, the Director Agent integrates the script content with the product visual assets to construct a complete multimodal input for the downstream video generation model. Specifically, scene descriptions, camera motion instructions, and voiceover text from the script are converted into textual prompts, while product images are simultaneously introduced as visual reference constraints, forming a multimodal conditioning input that guides the video generation model to synthesize personalized advertisement video content aligned with user preferences while maintaining product visual consistency.

<table><tr><td rowspan="2">Method</td><td colspan="2">Personalization Presentation</td><td rowspan="2">Product Alignment</td><td rowspan="2">Integration Quality</td><td rowspan="2">Diversity</td><td colspan="4">Video Quality</td></tr><tr><td>Content</td><td></td><td>MS</td><td>DD</td><td>AQ</td><td>IQ</td></tr><tr><td colspan="10">PCG-Bench</td></tr><tr><td>GenericAds (Closed)</td><td>1.78</td><td>1.55</td><td>8.62</td><td></td><td>7.15</td><td>0.9737</td><td>0.9470</td><td>0.5427</td><td>0.6564</td></tr><tr><td>NextAds (Closed)</td><td>7.17</td><td>5.21</td><td>8.68</td><td></td><td>6.69</td><td>0.9473</td><td>0.9520</td><td>0.5301</td><td>0.6287</td></tr><tr><td>NextAds (Open) MAPS</td><td>6.00</td><td>3.39</td><td>3.53</td><td>一</td><td>5.52</td><td>0.9482</td><td>0.9526</td><td>0.4025</td><td>0.6364</td></tr><tr><td></td><td>7.32</td><td>5.35</td><td>8.89</td><td>一</td><td>7.30</td><td>0.9862</td><td>0.9871</td><td>0.5603</td><td>0.6690</td></tr><tr><td colspan="10"></td></tr><tr><td>GenericAds (Closed)</td><td>3.92</td><td>2.24</td><td>9.46</td><td>3.98</td><td>9.51</td><td>0.9856</td><td>0.9452</td><td>0.4933</td><td>0.6549</td></tr><tr><td>NextAds (Closed)</td><td>8.45</td><td>8.37</td><td>7.83</td><td>8.71</td><td>2.76</td><td>0.9681</td><td>0.9358</td><td>0.5377</td><td>0.6384</td></tr><tr><td>NextAds (Open)</td><td>5.44</td><td>4.24</td><td>6.44</td><td>6.37</td><td>7.43</td><td>0.9533</td><td>0.9439</td><td>0.4440</td><td>0.5873</td></tr><tr><td>MAPS</td><td>8.96</td><td>8.73</td><td>9.78</td><td>8.99</td><td>9.87</td><td>0.9886</td><td>0.9611</td><td>0.5504</td><td>0.6682</td></tr></table>

Table 1. Performance comparison across PCG-Bench and PCI-Bench.

## 3.5. Reflector Agent

The Reflector Agent serves as the feedback-driven refinement module of the MAPS framework, closing the loop between deployed creatives and upstream generation. It operates on online user interaction signals collected after each advertisement impression, such as play duration, likes, comments, and purchases. It attributes these signals to their underlying causes, converts the attribution results into minimal revision directives, and routes them back to the Profile Agent and the Scripter Agent for iterative refinement.

## 3.5.1. Feedback Attribution.

Given the online interaction signals associated with a deployed creative, the Reflector Agent performs attribution to localize the source of suboptimal performance, distinguishing between two complementary categories of causes:

• Profile-level causes: inaccuracies in user modeling, such as misjudged topical interests, mismatched presentation styles, or incorrectly inferred decision-making tendencies, which indicate that the user profile fails to faithfully capture the user’s actual preferences.

• Script-level causes: deficiencies in the generated script, such as a weak narrative hook, misaligned tone or pacing, unclear CTA, or suboptimal presentation of product selling points, which indicate that the script itself does not effectively engage the target user.

## 3.5.2. Minimal Revision and Routing.

Based on the attribution results, the Reflector Agent produces minimal revision directives that can be executed by upstream agents at low latency and bounded cost, avoiding full regeneration of the creative. Profile-level directives are routed back to the Profile Agent to update the structured user profile, whereas script-level directives are routed back to the Scripter Agent to iteratively refine the advertisement script. Through this targeted feedback loop, MAPS continuously adapts its creatives to evolving user preferences across successive impressions, enabling closed-loop creative evolution without re-running the entire generation pipeline from scratch.

## 4. Experiments

## 4.1. Datasets and Implementation Details

## 4.1.1. Datasets

We conduct offline experiments on two representative benchmarks, PCG-Bench and PCI-Bench [50], corresponding to the Personalized Creative Generation (PCG) and Personalized Creative Integration (PCI) tasks, respectively. The PCG task aims to synthesize standalone personalized video ad creatives, while the PCI task focuses on seamlessly embedding advertising messages into a host video currently being watched by the user in a non-intrusive manner. Both benchmarks include 50 users, each matched with 5 target products to form 250 user-product pairs, sharing a product library of 18 products. PCG-Bench is built upon the Qilin recommendation dataset [4], while PCI-Bench is constructed from the MicroLens dataset [27], with the last interacted video per user designated as the host video for creative integration.

## 4.1.2. Implementation Details

MAPS comprises four collaborative agents: the Profile Agent, built on Qwen3-VL-32B-Instruct [3], extracts multidimensional user preference profiles from historical interaction records; the Scripter Agent, with Qwen3-VL-8B-Instruct [3] as its backbone, converts user profiles and product information into structured storyboard scripts; the Director Agent employs Kling-Omni [41] as the video generation model, taking the structured script as a conditiona input to produce the final personalized ad video; and the Reflector Agent, built on Qwen3-VL-32B-Instruct [3], attributes online user feedback to profile-level and script-level causes and routes minimal revisions back to the Profile and Scripter Agents. The Scripter Agent is trained in two stages: the SFT cold start stage formulates the task as sequenceto-sequence generation to maximize the conditional loglikelihood of ground-truth scripts in a structured format, fine-tuning via LlamaFactory [57] with LoRA [13] of rank 32, using AdamW with a learning rate of 1.0e-5, a batch size of 4, and a maximum sequence length of 8192 tokens for 20 epochs; the RL stage, implemented with the rLLM framework [39], applies the GRPO algorithm to jointly align generated scripts with user preferences and product information, with a KL divergence coefficient of 0.04, AdamW optimizer at a learning rate of 1.0e-6, a batch size of 4, for 5000 training steps. All experiments are conducted on 8 NVIDIA H100 GPUs.

![](images/8610babbdf8e32642a131310ef352aa6ec146667663386dec43e7d18674fbf8d.jpg)  
Figure 4. Demos of the PCG and PCI tasks.

## 4.2. Offline Evaluation

## 4.2.1. Evaluation Metrics

We design an evaluation suite covering five dimensions: video quality, personalization, product alignment, integration quality, and diversity. For video quality, we adopt Motion Smoothness (MS), Dynamic Degree (DD), Aesthetic Quality (AQ), and Imaging Quality (IQ) from VBench [15]. For the remaining dimensions, we employ GPT-5 [34] as a judge, scoring each criterion on a 0–10 scale with predefined rules. Specifically, personalization assesses the alignment between the generated creative and user historical preferences across presentation style and content preference; product alignment evaluates product identity accuracy, brand elements, and factual consistency; diversity measures variation among creatives generated for the same user across visual style and narrative structure; integration quality, applicable only to the PCI task, evaluates contextual flow and user experience disruption after inserting the creative into the host video. Detailed scoring prompts are provided in Sec. 6 of the supplementary material.

## 4.2.2. Comparison with State-of-the-Art Methods

As shown in Table 1, MAPS achieves the best performance across all metrics on both benchmarks. On PCG-Bench, MAPS not only advances personalization and product alignment but also improves diversity from 6.69 to 7.30, and ranks first on all four video quality metrics. On the more challenging PCI-Bench, while NextAds (Closed) improves personalization at the cost of product alignment (dropping from 9.46 to 7.83), MAPS mitigates this trade-off through structured profile extraction and script generation, simultaneously achieving the highest diversity score and the best integration quality. These results validate that multi-agent collaboration enables global optimization across competing objectives, mitigating the trade-offs commonly observed in existing baselines.

<table><tr><td></td><td colspan="3">Agent Components</td><td colspan="2">Training</td><td rowspan="2">Pers.</td><td rowspan="2">Prod. Align.</td></tr><tr><td>Profile</td><td>Scripter</td><td>Director</td><td>Reflector</td><td>SFT RL</td><td></td></tr><tr><td></td><td></td><td>√</td><td></td><td></td><td></td><td>3.10</td><td>7.83</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td></td><td></td><td>4.35</td><td>8.46</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td>6.92</td><td>8.67</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>7.60</td><td>8.74</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>7.81</td><td>9.12</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>8.85</td><td>9.78</td></tr></table>

Table 2. Ablation study on PCI-Bench.

## 4.2.3. Ablation Study

To validate the contribution of each component in MAPS, we conduct ablation studies on PCI-Bench, as shown in Table 2. Starting from a baseline with only the Director Agent, the incremental introduction of the Scripter Agent, Profile Agent, and Reflector Agent drives successive improvements in personalization, confirming their respective roles in enhancing informational consistency, enabling deep personalization, and refining creatives through feedbackdriven attribution. Building upon the multi-agent collaboration, the SFT training stage establishes the structural competence of the Scripter Agent, laying a solid foundation for subsequent alignment. Finally, adding the RL stage, the full MAPS system achieves the best performance on both dimensions, improving personalization by 1.25 points and product alignment by 1.04 points compared to the variant without SFT and RL. This demonstrates that joint optimization via the multi-dimensional reward function drives the Scripter Agent to achieve synergistic alignment with both user preferences and product information, rather than trading one off against the other.

## 4.3. Online Evaluation

We further validate MAPS through online A/B testing in a specific scenario in the Kuaishou advertising system. We compare MAPS against two production baselines that have been successively deployed in the advertising pipeline. Deep Learning Recommendation Model (DLRM) [7, 8, 58, 60] is a retrieval-based recommendation model that matches users with creatives selected from a fixed content pool. Generative Recommendation Model (GRM) [9, 51] extends retrieval-based recommendation by modeling user interest as discrete semantic IDs (SIDs) [31] through autoregressive generation, but uses the predicted SIDs as retrieval indices over a fixed content pool rather than generating new creatives.

<table><tr><td>Method</td><td>CTR</td><td>CVR</td><td>ARPU</td></tr><tr><td>DLRM → MAPS</td><td>+4.88%</td><td>+4.01%</td><td>+12.89%</td></tr><tr><td>GRM → MAPS</td><td>+2.87%</td><td>+2.74%</td><td>+9.85%</td></tr></table>

Table 3. Online A/B evaluation results in a specific scenario in the Kuaishou advertising system.

We report three business-critical metrics: click-through rate (CTR), conversion rate (CVR), and average revenue per user (ARPU), where ARPU is defined as the total revenue divided by the number of users over the same period. As shown in Table 3, MAPS improves CTR by 4.88%, CVR by 4.01%, and ARPU by 12.89% over DLRM. We attribute this gain to the shift from retrieval to generation. DLRM can only match users with creatives drawn from a fixed content pool, and its static and limited inventory cannot keep pace with the evolving preferences of individual users. In contrast, MAPS generates personalized creatives on demand that fit each user and product more closely. MAPS further improves CTR by 2.87%, CVR by 2.74%, and ARPU by 9.85% over GRM. Although GRM already leverages generative models to predict semantic IDs, it still retrieves creatives from a limited content pool. In contrast, MAPS produces native creatives on demand without being bounded by a pre-built inventory. The large ARPU improvement indicates that generating personalized creatives tailored to each user extends beyond better matching and unlocks commercial value that retrieval-based systems cannot reach with a limited creative pool, confirming the practical effectiveness of MAPS at industrial scale.

## 5. Conclusion

In this paper, we introduce MAPS, a multi-agent personalized script-driven video advertising system for advertisement creative generation. By coordinating the Profile, Scripter, Director, and Reflector Agents, MAPS transforms personalized video advertising from static retrieval over a limited creative inventory into dynamic generation within a continuous creative space. We apply reinforcement learning to script generation, enabling the production of structured scripts that integrate product information into narratives aligned with user preferences. Furthermore, MAPS supports continuous creative evolution across successive impressions by integrating post-deployment online user feedback into a closed-loop refinement mechanism. Extensive offline experiments on PCG-Bench and PCI-Bench demonstrate that MAPS outperforms existing methods across all evaluation metrics, and online A/B testing in a specific scenario in the Kuaishou advertising system further validates its effectiveness at industrial scale. Future work will explore finer-grained online feedback modeling and long-horizon creative evolution in larger-scale scenarios.

## References

[1] Gagan Aggarwal, Ashwinkumar Badanidiyuru, Santiago R Balseiro, Kshipra Bhawalkar, Yuan Deng, Zhe Feng, Gagan Goel, Christopher Liaw, Haihao Lu, Mohammad Mahdian, et al. Auto-bidding and auctions in online advertising: A survey. ACM SIGecom Exchanges, 22(1):159–183, 2024. 2

[2] Gagan Aggarwal, Yifan Wang, and Mingfei Zhao. Online advertising with spatial interactions. In Proceedings of the ACM Web Conference 2026, pages 213–224, 2026. 2

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6, 7

[4] Jia Chen, Qian Dong, Haitao Li, Xiaohui He, Yan Gao, Shaosheng Cao, Yi Wu, Ping Yang, Chen Xu, Yao Hu, et al. Qilin: A multimodal information retrieval dataset with applevel user sessions. In Proceedings ofthe 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3670–3680, 2025. 6

[5] Yiyun Chen and Weikai Yang. Refadgen: High-fidelity advertising image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3138–3146, 2026. 2

[6] Yuan Chen, Mengge Xue, Cao Te, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang, et al. R<sup>3</sup>: Advertisement compliance Rectification via group-Relative experience extractor and curriculum Reinforcement. In The 64th Annual Meeting of the Association for Computational Linguistics–Industry Track, 2026. 2

[7] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In Proceedings ofthe 1st workshop on deep learning for recommender systems, pages 7–10, 2016. 8

[8] Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pages 191–198, 2016. 8

[9] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965, 2025. 8

[10] Asmaa El Fraihi, Ines Abdelaziz, and Oana Goga. Is contextual advertising safe? analyzing systemic risks with ads on youtube. In Proceedings of the ACM Web Conference 2026, pages 1739–1750, 2026. 2

[11] Yi Feng, Chuanyi Li, and Vincent Ng. Understanding advertisements. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 28576–28585, 2025. 1

[12] Zhaoyi Joey Hou, Adriana Kovashka, and Xiang Lorraine Li. Leveraging large models to evaluate novel content: A case study on advertisement creativity. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 21169–21188, 2025. 2

[13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen,

et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022. 7

[14] Yinqiu Huang, Hao Ma, Wenshuai Chen, Zongwei Wang, Shuli Wang, Yongqiang Zhang, Xue Wei, Yinhua Zhu, Haitao Wang, and Xingxing Wang. Generative bid shading in real-time bidding advertising. arXiv preprint arXiv:2508.06550, 2025. 2

[15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7

[16] Kaiyuan Ji, Yixuan Gao, Lu Sun, Yushuo Zheng, Zijian Chen, Jianbo Zhang, Xiangyang Zhu, Yuan Tian, Zicheng Zhang, and Guangtao Zhai. A3: Towards advertising aesthetic assessment. In Proceedings of the IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 9478–9490, 2026. 1

[17] Kyungho Kim, Yeonje Choi, Gyurim Hwang, Sejin Chung, Hongseok Lee, Myeong Ho Song, Yeongho Kim, Sunwoo Kim, Jongha Lee, Juyeon Kim, et al. Pre-deployment adver tisement ranking under data scarcity via context-aware criteria generation with vlms. In The 64th Annual Meeting of the Association for Computational Linguistics–Industry Track, 2026. 2

[18] Kai Kruger, Johanna Binnewitt, Kathrin Ehmann, Stefan¨ Winnige, and Alan Akbik. Improving online job advertisement analysis via compositional entity extraction. In Pro ceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 27035–27053, 2025. 3

[19] Ningyuan Li, Zhilin Zhang, Tianyan Long, Yuyao Liu, Rongquan Bai, Yurong Chen, Xiaotie Deng, Pengjie Wang, Chuan Yu, Jian Xu, et al. Beyond advertising: Mechanism design for platform-wide marketing service “quanzhantui”. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 1447– 1457, 2025. 1

[20] Wenwei Li, Ming Xu, Tianle Xia, Lingxiang Hu, Yiding Sun, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, and Jie Jiang. Towards faithful industrial rag: A reinforced coadaptation framework for advertising qa. arXiv preprint arXiv:2602.22584, 2026. 2

[21] Yang Li, Yuchao Ma, and Qi Qi. Optimal auction design in the joint advertising. arXiv preprint arXiv:2507.07418, 2025. 1

[22] Christopher Liaw and Wennan Zhu. Risk-averse and op timistic advertiser incentive compatibility in auto-bidding. arXiv preprint arXiv:2508.16823, 2025. 1

[23] Tongtong Liu, Zhaohui Wang, Meiyue Qin, Zenghui Lu, Xudong Chen, Yuekui Yang, and Peng Shu. Real-time ad retrieval via llm-generative commercial intention for sponsored search advertising. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 28936–28948, 2025. 2

[24] Xinwei Long, Kai Tian, Peng Xu, Guoli Jia, Jingxuan Li, Sa Yang, Yihua Shao, Kaiyan Zhang, Che Jiang, Hao Xu,

et al. Adsqa: Towards advertisement video understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23396–23407, 2025. 2

[25] Yuchao Ma, Ruohan Qian, Bingzhe Wang, Qi Qi, Wenqiang Liu, Qian Tang, Zhao Shen, Wei Zhong, Bo Shen, Yixin Su, et al. Genauction: A generative auction for online advertising. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12372–12380, 2025. 1

[26] Gaku Morio, Harri Rowlands, Dominik Stammbach, Christopher D Manning, and Peter Henderson. A multimodal benchmark for framing of oil & gas advertising and potential greenwashing detection. Advances in Neural Information Processing Systems, 38, 2026. 1

[27] Yongxin Ni, Yu Cheng, Xiangyan Liu, Junchen Fu, Youhua Li, Xiangnan He, Yongfeng Zhang, and Fajie Yuan. A content-driven micro-video recommendation dataset at scale. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pages 6486– 6491, 2025. 6

[28] Samuel Pelissier, Naif Mehanna, Sterenn Roux, Quentin´ Perez, Walter Rudametkin, Johann Bourcier, and Pierre Laperdrix. Users pay twice: The hidden energy cost of web advertising. In Proceedings of the ACM Web Conference 2026, pages 1629–1639, 2026. 2

[29] Yunshan Peng, Ji Wu, Wentao Bai, Yunke Bai, Jinan Pang, Wenzheng Shu, Yanxiang Zeng, Xialong Liu, and Peng Jiang. R&f-inventory: A large-scale dataset for monotonic inventory estimation in reach and frequency advertising. arXiv preprint arXiv:2604.16821, 2026. 1

[30] Dongjun Qian, Kai Su, Yiming Tan, Qishuai Diao, Xian Wu, Chang Liu, Bingyue Peng, and Zehuan Yuan. Vc-llm: Automated advertisement video creation from raw footage using multi-modal llms. arXiv preprint arXiv:2504.05673, 2025. 2

[31] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems, 36:10299–10315, 2023. 8

[32] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2

[33] Yunxiao Shi, Wujiang Xu, Zhang Zeqi, Xing Zi, Qiang Wu, and Min Xu. Personax: A recommendation agent-oriented user modeling framework for long behavior sequence. In Findings of the Association for Computational Linguistics: ACL 2025, pages 5764–5787, 2025. 3

[34] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 7, 1

[35] Mingxuan Song, Yusen Huo, Bohan Zhou, Shenglin Yin, Zhen Xiao, Jieyi Long, Zhilin Zhang, and Chuan Yu. Dara: Few-shot budget allocation in online advertising via incontext decision making with rl-finetuned llms. In Proceed-

ings of the ACM Web Conference 2026, pages 40–50, 2026. 3

[36] Yinghao Song, Xiangji Zeng, Shuai Cui, Lu Sun, Zhaowei Liu, Yuan Yuan, Yulu Wang, Hai Zhou, and Zhaohan Gong. Addriftbench: A benchmark for detecting data drift and la bel drift in short video advertising. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 10305–10321, 2025. 1

[37] Ermis Soumalias, Michael Curry, and Sven Seuken. Truth ful aggregation of llms with an application to online advertising. Advances in Neural Information Processing Systems, 38:145262–145308, 2026. 1

[38] Nguyen Thanh Tam, Khanh Quoc Tran, Dat Thanh Pham, Truong Phu Le, Nguyen Hoang Gia Han, and Binh T Nguyen. Vietcheckmed: Explainable regulatory compliance checking for medical advertisements on vietnamese social media. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1069–1077, 2026. 3

[39] Sijun Tan, Michael Luo, Colin Cai, Tarun Venkat, Kyle Montgomery, Aaron Hao, Tianhao Wu, Arnav Balyan, Manan Roongta, Chenguang Wang, et al. rllm: A framework for post-training language agents, 2025. 7

[40] Cao Te, Mengge Xue, Yuan Chen, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang, et al. Ssr-a: Spatial-and semantic-aware instructions and curriculum reinforcement for advertisement compliant rectification. In The 64th Annual Meeting of the Association for Computational Linguistics–Industry Track, 2026. 3

[41] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025. 6

[42] Heng Wang, Yotaro Shimose, and Shingo Takamatsu. Banneragency: Advertising banner design with multimodal llm agents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 4304–4329, 2025. 3

[43] Lu Wang, Chiming Duan, Pu Zhao, Fangkai Yang, Yong Shi, Xuefeng Luo, Bingjing Xu, Weiwei Deng, Qingwei Lin, and Dongmei Zhang. Icl-bandit: Relevance labeling in advertisement recommendation systems via llm. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 23439–23449, 2025. 3

[44] Zhongyu Wang. Gramrag: Orchestrating multi-agent multi step reasoning via graph memory with reinforcement learn ing. arXiv preprint arXiv:2609.14066, 2026. 3

[45] Zhongyu Wang. Mdocrag-rl: Empowering multi-modal document rag via complex visual reasoning with reinforcement learning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 8641–8651, 2026. 1

[46] Zhongyu Wang and Pengbo Liu. Mars-rl: Enhancing multiagent rag systems for multi-modal documents via strategic reasoning with reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9674–9683, 2026. 3

[47] Xunzhi Xiang, Yabo Chen, Guiyu Zhang, Zhongyu Wang, Zhe Gao, Quanming Xiang, Gonghu Shang, Junqi Liu,

Haibin Huang, Yang Gao, et al. Macro-from-micro planning for high-quality and parallelized autoregressive long video generation. arXiv preprint arXiv:2508.03334, 2025. 2

[48] Binxiao Xu, Junyu Feng, Xiaopeng Lin, Haodong Li, Zhiyuan Feng, Bohan Zeng, Shaolin Lu, Ming Lu, Qi She, and Wentao Zhang. Ad-mir: Bridging the gap from perception to persuasion in advertising video understanding via structured reasoning. arXiv preprint arXiv:2602.07625, 2026. 1

[49] Yexing Xu, Wei Feng, Shen Zhang, Haohan Wang, Yuxin Qin, Yaoyu Li, Ao Ma, Yuhao Luo, Lu Wang, Xudong Ren, Haoran Wang, Run Ling, Zheng Zhang, Jingjing Lv, Junjie Shen, Ching Law, Longguang Wang, and Yulan Guo. Design your ad: Personalized advertising image and text generation with unified autoregressive models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 472–483, 2026. 1

[50] Yiyan Xu, Ruoxuan Xia, Wuqiang Zheng, Fengbin Zhu, Wenjie Wang, and Fuli Feng. Nextads: Towards nextgeneration personalized video advertising. arXiv preprint arXiv:2603.02137, 2026. 6

[51] Ben Xue, Dan Liu, Lixiang Wang, Mingjie Sun, Peng Wang, Pengfei Zhang, Shaoyun Shi, Tianyu Xu, Yunhao Sha, Zhiqiang Liu, et al. Generative recommendation for largescale advertising. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 8346–8357, 2026. 8

[52] Xinxin Yang, Yangyang Tang, Yikun Zhou, Yaolei Liu, Yun Li, and Bo Yang. Ahbid: An adaptable hierarchical bidding framework for cross-channel advertising. In Proceedings of the ACM Web Conference 2026, pages 181–191, 2026. 1

[53] Guangyi Zhang, Ilie Sarpe, and Aristides Gionis. Efficient and practical approximation algorithms for advertising in content feeds. In Proceedings of the ACM on Web Conference 2025, pages 3193–3203, 2025. 1

[54] Guiyu Zhang, Yabo Chen, Xunzhi Xiang, Junchao Huang, Zhongyu Wang, and Li Jiang. Symphomotion: Joint control of camera motion and object dynamics for coherent video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11127– 11137, 2026. 2

[55] Zhaoqi Zhang, Jiaming Deng, Miao Xie, Linyou Cai, Qianlong Xie, Xingxing Wang, Siqiang Luo, and Gao Cong. Beyond single slot: Joint optimization for multi-slot guaranteed display advertising. arXiv preprint arXiv:2605.21556, 2026. 1

[56] Jingyi Zheng, Tianyi Hu, Yule Liu, Zhen Sun, Zongmin Zhang, Zifan Peng, Wenhan Dong, and Xinlei He. Chasm: Unveiling covert advertisements on chinese social media. Advances in Neural Information Processing Systems, 38, 2026. 1

[57] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410, 2024. 7

[58] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun

Gai. Deep interest network for click-through rate prediction. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1059– 1068, 2018. 8

[59] Milton Zhou, Sizhong Qin, Yongzhi Li, Quan Chen, and Peng Jiang. Autocut: End-to-end advertisement video editing based on multimodal discretization and controllable gen eration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 37777–37787, 2026. 1

[60] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1079–1088, 2018. 8

# MAPS: A Multi-Agent Personalized Script-Driven Video Advertising System Enhanced by Reinforcement Learning

Supplementary Material

## 6. Evaluation Prompts

To evaluate the generated creatives, we employ GPT-5 [34] as the LLM judge and design dedicated scoring prompts, as detailed below.

## Judge Prompt for Personalization (Presentation)

## You are an expert in evaluating the presentation style of advertisement creatives.

Given a target product and the covers of the user’s historically interacted videos, assess how well the generated advertisement aligns with the user’s aesthetic preferences in visual presentation. Consider the following aspects:

• Tone: Does the overall tone match the user’s preference (warm, cold, or neutral)?

• Lighting: Is the lighting treatment suited to the user’s preference (soft, harsh, or natural)?

• Texture: Does the textural style (smooth, rough, etc.) fit the user’s aesthetic inclination?

• Artistic style: Does the artistic style (minimalistic, modern, vintage, etc.) align with the user’s taste?

## Scoring Criteria:

• Perfect match (10): The presentation style aligns perfectly with the user’s preferences.

• Good match (7–9): The presentation style aligns well, with only minor differences.

• Moderate match (4–6): The presentation style partially matches the user’s preferences.

• Poor match (1–3): The presentation style does not align well with the user’s preferences.

• No match (0): The presentation style completely fails to align with the user’s preferences.

## Output JSON:

```json
{
"score": number,
"reason": "brief analysis"
}
```

## Judge Prompt for Personalization (Content)

## You are an expert in evaluating how well an advertisement creative matches the user’s content interests.

Given a target product and the covers of the user’s historically interacted videos, assess how well the generated advertisement aligns with the user’s content preferences. Consider the following aspects:

• Subject matter: Does the subject matter of the advertisement align with the user’s interest domains?

• Scene semantics: Does the scene selection (location, setting) fit what the user would enjoy?

• Information angle: Does the advertisement focus on the aspects the user cares about (performance, aesthetics, or value)?

• Motivational appeal: Does the advertisement appeal to motivations that resonate with the user (quality, efficiency, or lifestyle)?

## Scoring Criteria:

• Perfect match (10): The content aligns perfectly with the user’s preferences.

• Good match (7–9): The content aligns well, with only minor differences.

• Moderate match (4–6): The content partially aligns with the user’s preferences.

• Poor match (1–3): The content does not align well with the user’s preferences.

• No match (0): The content is completely irrelevant to the user.

## Output JSON:

"score": number,

"reason": "brief analysis"

## Judge Prompt for Product Alignment

You are an expert in evaluating the consistency between an advertisement creative and the target product.

Given the target product information and the generated advertisement, assess how faithfully the advertisement represents the product. Consider the following aspects:

• Product identity accuracy: Does the advertisement correctly portray the product’s appearance, color, model, packaging, and accessories?

• Brand and recognition elements: Does the advertisement accurately present the product’s logo, brand visual language, and design style?

• Selling points and factual consistency: Do the product claims conveyed in the advertisement reflect the real product features without exaggeration or misrepresentation?

## Scoring Criteria:

• Perfect match (10): The product in the advertisement matches the actual product in every respect.

• Good match (7–9): The product is mostly consistent, with only minor details altered.

• Moderate match (4–6): The product is partially consistent, but with some significant differences.

• Poor match (1–3): The product in the advertisement is mostly inconsistent with the actual product.

• No match (0): The product identity is completely inconsistent with the actual product.

## Output JSON:

"score": number,   
"reason": "brief analysis"   
}

## Judge Prompt for Integration Quality

You are an expert in evaluating how naturally an advertisement is integrated into the surrounding video content.

Given a video in which the advertisement is embedded, assess how seamless or disruptive the advertisement feels to the viewing experience. Consider the following aspects:

• Contextual flow: Is the transition between the organic content and the advertisement smooth, or is it a sudden, jarring break?

• Thematic consistency: Does the advertisement share a similar tone, visual style, or topic with the surrounding content?

• User experience disruption: Is the advertisement placed at a logical breaking point, rather than interrupting a key moment?

• Narrative integration: Is the advertisement woven into the storyline, rather than appearing as an external overlay or hard cut?

## Scoring Criteria:

• Seamless (10): The advertisement is perfectly integrated and barely perceived as a disruption.

• Natural (7–9): The advertisement is well placed and shares thematic elements with the video, causing minimal friction.

• Noticeable but acceptable (4–6): The advertisement is clearly a break from the content, but occurs at a logical pause or maintains a similar aesthetic.

• Intrusive (1–3): The advertisement feels forced and significantly breaks the immersion.

• Highly disruptive (0): The advertisement completely ruins the viewing experience through poor timing or aggressive interruption.

## Output JSON:

```json
{
"score": number,
"reason": "brief analysis"
}
```

## Judge Prompt for Diversity

You are an expert in evaluating the diversity of advertisement creatives.

Given a set of advertisements generated for the same user across different products, assess how distinct these advertisements are from one another. Consider the following aspects:

• Visual style: Is there excessive similarity in tone, lighting, texture, and artistic style across the different product advertisements?

• Narrative form: Is the narrative structure (tutorial, vlog, plot twist, etc.) and tone (professional, humorous, etc.) overly similar across products?

• Subject matter and scene semantics: Is the subject matter or scene selection overly similar across the advertisements targeting the same user?

## Scoring Criteria:

• Fully diverse (10): The advertisements are completely distinct with no noticeable overlap.

• Mostly diverse (7–9): The advertisements are mostly distinct, with only minor similarities.

• Moderately diverse (4–6): The advertisements are somewhat similar but still distinct in key areas.

• Slightly diverse (1–3): The advertisements are largely similar, with only slight variations.

• Not diverse (0): The advertisements are identical in all aspects.

Output JSON:   
{   
"score": number,   
"reason": "brief analysis"   
}

## 7. Qualitative Demos

We present qualitative demos of the ad videos generated by MAPS in Figures 5–11, each targeting a different user content preference and product.

![](images/f2bdbab88d53a5684519c2467698e5206cbdae52ad3a68082cf3c74216e813cb.jpg)

Figure 5. Demo of generated ad video. Content preference: Music festival. Target Product: iPhone 17.  
![](images/00eb251ac351570e5a7050cf075fdda5e65bc2fa0fb2dbb97617d9cc96e1fccc.jpg)

Figure 6. Demo of generated ad video. Content preference: Relationship topics. Target Product: L’Oreal Paris Revitalift.´  
![](images/8207028904a643dd8e788755591818b03d3ad2387bb9e00f05a65a6abf0a68bd.jpg)

Figure 7. Demo of generated ad video. Content preference: Travel. Target Product: Airbnb.  
![](images/2b7aff81fefea53ffa6af9f161496a833eae460cf101d919be761bb3199a9aa6.jpg)

Figure 8. Demo of generated ad video. Content preference: Uniform outfit. Target Product: Walmart retail.  
![](images/93eb20ebbd92a0ed89597e2f0f618719acaeb4b1329667090e0d37855406c5e0.jpg)

Figure 9. Demo of generated ad video. Content preference: Daily life. Target Product: KFC chicken nuggets.  
![](images/11571498e946e9b75e2762bf47762d0163e3ac5e7e913ef228b950422843a3bb.jpg)

Figure 10. Demo of generated ad video. Content preference: Delicious food. Target Product: Coca-Cola.  
![](images/2bc5c75d87dbea61f178080d01db3ac1b7e17fcba2aa457ef297746dfe3e0e6f.jpg)  
Figure 11. Demo of generated ad video. Content preference: Examination. Target Product: Herman Miller chair.