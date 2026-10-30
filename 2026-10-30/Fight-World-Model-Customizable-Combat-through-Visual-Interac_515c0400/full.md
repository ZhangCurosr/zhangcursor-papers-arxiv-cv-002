# Fight World Model: Customizable Combat through Visual Interaction Proxies

Jiahua Li<sup>1</sup> Zixu John<sup>2</sup> Tom Zhong<sup>2</sup> Fuping Wu<sup>3</sup> Tianhao Xu<sup>2</sup>

Ruicheng Zhang<sup>4</sup> Jun Zhou<sup>4</sup> Zunnan Xu<sup>5,\*</sup>

<sup>1</sup>University of Oxford, UK <sup>2</sup>Independent Researcher <sup>3</sup>Imperial College London, UK <sup>4</sup>Tsinghua University, China <sup>5</sup>Nanyang Technological University, Singapore

![](images/48f803f97ce16c347857f592756565daab5b4c0e2c766b5eb748cca04a6ed9ca.jpg)  
Figure 1: Customizable Combat with FWM. Character and arena customization is shown on the left, and controllable skills on the right. The central pipeline connects rule execution to video rendering through an interaction proxy.

## Abstract

A playable, programmable game world must maintain persistent state, enforce executable rules, and support extensible gameplay mechanics. Existing video world models can generate plausible combat, yet struggle to keep hit reactions and health updates consistent with game rules. We introduce Fight World Model (FWM), which separates rulegoverned combat execution from video rendering. An executable engine maintains state and resolves interactions. A deterministic compiler encodes these outcomes and attempted actions in a spatially and temporally aligned interaction proxy. The proxy aligns action and skill windows, brief hit reactions, and persistent health by fighter identity and timing. These cues can be controlled separately to render interactions under modified rules. Identity-bound references preserve character appearance across side switches. We train FWM on FightWorld Corpus, spanning 52 skill effects, 187 character-appearance variants, and 15 arenas. Without retraining the renderer, FWM supports damage scaling, limited-hit immunity, lifesteal, and multi-stage skill mechanisms beyond the native game. It also allows character and arena customization. Experiments demonstrate the effectiveness ofFWM, which outperforms existing game world models in action fidelity, hit reactions, health-state

consistency, and character appearance preservation. Our project is available atfight-world-model.github.io.

## 1 Introduction

Interactive world models aim to simulate how environments evolve under user actions [2, 10, 27, 46, 47]. Recent video world models learn from gameplay recordings to generate playable sequences from controller or text inputs [7, 15, 30, 33, 39, 43]. Customizing such worlds requires changing both gameplay rules and visual appearances. In multi-fighter combat, the same attack may cause different health loss under different rules. A protected fighter may react to a hit without losing health, while lifesteal may restore the attacker’s health. When only character or arena appearance changes, the specified interactions should remain the same. As illustrated in Fig. 1, a customizable system must allow interaction outcomes to follow edited rules while preserving those interactions across appearance changes.

These requirements are difficult to meet when state transitions are inferred within the video generator [5, 7, 15, 33, 37, 39]. The model learns how actions relate to reactions and state changes from recorded gameplay. StatePlay further predicts game-state variables, such as health, alongside video frames [21], but predicting state does not itself provide an editable rule for updating it. Multi-entity action binding, identity-aware instructions, and NPC planning extend control over characters [8, 28, 37, 38, 51]. They do not by themselves determine how interaction outcomes should change under a new rule configuration. When rules change, a generator may retain familiar action–consequence associations from training and deviate from the externally specified outcomes.

Code-driven world models offer a way to execute worldstate transitions outside the video generator and render the resulting spatial proxies [6, 16, 42, 44]. Correct state execution alone, however, does not ensure that the generated interaction follows those results. The renderer must identify the acting and affected fighters, depict each event at the intended time, and preserve the resulting state across subsequent exchanges. Spatial layouts alone do not specify action windows, hit reactions, or persistent health changes. Text and state tokens can carry these quantities, but the renderer must map them to the appropriate body regions, heads-up display (HUD) elements, and video frames. Appearance customization adds a related difficulty: actions and consequences must remain associated with the intended fighters as their appearances change or they exchange positions. The interface must therefore convey attempted actions, events, and persistent state with consistent identity and timing.

We introduce Fight World Model (FWM), a customizable world-model framework that connects executable combat rules to video generation through a visual interaction proxy. As shown in Fig. 2, an executable engine resolves interactions and maintains state. A deterministic compiler rasterizes fighter states, action and skill windows, and transient hit-reaction cues onto a shared canvas and timeline. Reactions and health updates are encoded separately, allowing a hit to trigger a visible response without necessarily reducing health. Rule edits change the executed records and the resulting proxy, while the proxy format and renderer remain unchanged. Character and arena references specify appearance, with identity binding associating character references with the intended fighters across side switches. This design lets the same trained renderer depict interactions under different rules and visual appearances.

We instantiate and evaluate FWM in a fixed-view twofighter combat setting. FightWorld Corpus pairs gameplay videos with interaction proxies across 52 skill effects, 187 character-appearance variants, and 15 arenas. FWM outperforms adapted game world models in action fidelity, hit reactions, persistent health accuracy, and character appearance preservation. Under matched executed records, reference binding, and training budgets, the interaction proxy also achieves higher joint fidelity than spatial proxies augmented with text or state tokens. Without retraining the renderer, FWM supports damage scaling, limited-hit immunity, lifesteal, and multi-stage skill mechanisms beyond native Street Fighter II/III. Appearance and side-switch evaluations test whether specified interactions and reference-toentity associations are preserved under visual changes. We make the following contributions:

• We present FWM, a customizable combat world-model system that supports editable interaction rules and character or arena appearances without retraining the renderer.

• We design a visual interaction proxy that separately encodes action windows, hit reactions, and persistent health with consistent identity and timing. Identity binding connects appearance references to the fighters participating in these interactions.

• We construct FightWorld Corpus and evaluate the system through comparisons with existing methods, controlled encoding studies, rule interventions, and appearance customization.

## 2 Related Work

Game World Models. Action-conditioned video world models predict future frames from user inputs [1, 2, 33]. Recent systems extend this approach toward real-time streaming, longer horizons, and broader environments [5, 7, 9, 11, 30, 39, 43, 48]. Beyond navigation and player-centric interaction, multi-entity control introduces the need to associate each input with the intended character. Per-player keyboard streams can be paired with character references [15], while language conditions, persistent subject tokens, and synchronized inputs support multiple subjects or players [28, 31, 51]. Compact state representations also support NPC planning [8]. ReactiveGWM v1 separates player keyboard control from NPC strategy conditioning [37], and ReactiveGWM v2 uses identity handles to direct action and rule text to either fighter [38]. StatePlay additionally predicts health and skill meters alongside video through a dedicated state branch [21]. These methods improve action specification, entity control, and state prediction, but interaction outcomes remain largely governed by learned video dynamics. We focus on rendered reactions and persistent state under externally edited rules.

World State and Visual Proxies. Spatial and trajectory conditioning provides explicit control over object placement and motion in video generation [25, 35, 36, 45]. Recent world models separate state execution or maintenance from video rendering [3, 6, 16, 42, 44]. MASS maintains a learned authoritative state shared across players [3], while Magpie translates white-box engine frames into generated video [44]. Oneira represents geometry and coarse interaction states through colored 3D boxes [42]. Code World

![](images/f8331302dc3c9e8b1cdb056409e2cde4b758efd80e2bdcbae122048a2dc49b80.jpg)  
Figure 2: Overview of FWM. The Proxy Engine executes gameplay rules, and the resulting state trajectories, action windows, and events are compiled into a visual interaction proxy. The proxy and appearance references are encoded with a shared VAE and concatenated with video tokens. Matching spatiotemporal rotary positions align the proxy with the video, while shared identity embeddings bind character references to the corresponding regions in the first video latent and proxy sequence. The DiT processes the combined sequence, and only video tokens are decoded to generate the continuation.

Model compiles depth, semantic-ID, and instance maps from executable code [6], and Programmable World Model projects 3D boxes into identity, semantic-category, and motion-direction maps [16]. These methods establish explicit interfaces between world state and visual generation. Our focus is on the interaction information needed when edited rules change the relationship between attempted actions, hit reactions, and persistent health.

Reference-Guided Video Generation. Reference-guided video generators condition diffusion models on reference images to preserve subject appearance across frames [13, 17, 23, 40, 49]. Multi-subject methods use spatial masks, semantic alignment from vision-language models, or binding slots to distinguish appearances across entities [20, 41, 50]. Character animation and motion-transfer methods drive reference subjects through explicit pose sequences [14] or in-context spatiotemporal tokens aligned with the output video [26, 41]. In our setting, appearance references must remain associated with the fighters performing each action and receiving each consequence, including through side switches. We study this association jointly with rule-controlled interaction outcomes.

## 3 Method

FWM supports rule and appearance customization by coupling executable state updates with conditional video generation. As shown in Fig. 2, an engine executes the active rules, and a deterministic compiler converts the resulting states and events into a visual interaction proxy. Rule edits change the executed interaction and proxy, while appearance edits retain the proxy and change the references. Both use the same trained renderer.

## 3.1 Rule Execution and Customization

Let $S _ { k }$ be the executable state at engine update k, $u _ { k }$ the available action inputs, and R the rule configuration. The engine computes

$$
( S _ { k + 1 } , e _ { k + 1 } ) = F _ { R } ( S _ { k } , u _ { k } ) ,\tag{1}
$$

where $e _ { k + 1 }$ contains the events produced together with the updated state $S _ { k + 1 }$ . The engine computes entity motion, resolves hit events, and applies the configured state updates. $S _ { k }$ includes persistent entity attributes and the counters or timers needed to execute the rules. The configuration R can be edited in code [6, 16].

We reserve t for output video frames. Engine states and event records are aligned to this frame timeline before compilation. Let $\bar { S } _ { t }$ denote the aligned state and $s _ { t } = P ( \bar { S } _ { t } )$ its visible projection. From the aligned event history, we form active records $E _ { t }$ that specify entity identities, onsets, and durations for action windows and reaction events. An event onset is paired with the state update it produces. The compiler then generates each proxy frame as $\pi _ { t } = D ( s _ { t } , E _ { t } )$

A rule edit replaces R with R<sup>′</sup> and reruns the engine from the same initial state and submitted action sequence. Entity trajectories, hit times, and rule-dependent state are recomputed under R<sup>′</sup>. Multi-stage skill mechanisms are expressed through rules that schedule displacement, impacts, and status windows. The resulting records are compiled and rendered without changing the renderer parameters. For damage scaling, limited-hit immunity, and lifesteal, the appearance references and text conditions also remain fixed. Changes in executed outcomes reach the renderer only through the recomputed proxy.

An appearance edit changes the character or arena references while keeping the executed records and proxy fixed. Character references retain their target entity identities. Rule and appearance edits can also be combined by rendering recomputed records with new appearance conditions. For successive action intervals, the engine carries its state forward, while visual context supports continuity between generated segments.

## 3.2 Interaction Proxy

The proxy separates attempted actions, transient reactions, and persistent state because editable rules can change their relationship. These signals share entity identities and frame indices, allowing their values and durations to vary without changing the conditioning format. We describe the concrete combat encoding used in our experiments below.

For this instantiation, the visible state at video frame t is

$$
s _ { t } = \{ ( i , b _ { t } ^ { i } , f _ { t } ^ { i } , q _ { t } ^ { i } , a _ { t } ^ { i } , h _ { t } ^ { i } ) \} _ { i = 1 } ^ { 2 } ,\tag{2}
$$

where i is a persistent fighter identity, and $b , f , q , a ,$ and h denote its bounding box, facing direction, stance, coarse action label, and health. Each fighter’s box and HUD health bar share an identity color. Facing and stance marks encode orientation and posture, and a separate band encodes the coarse action label, including an unknown value for unannotated actions.

Action and Skill Windows. For the fighter with logged inputs, six attack classes combine punches and kicks with light, medium, and heavy strengths. Horizontal stripes encode punches and vertical stripes encode kicks, with coverage indicating strength. The marks remain active during the command window and supplement the coarse labels available for both fighters. They encode attempted actions, with hit outcomes specified separately. At inference, the engine can assign command windows to either fighter through its persistent identity.

A magenta block in front of the caster follows its position and facing direction to mark a skill’s casting window. Text describes the effect’s type and appearance, so the corpus’s 52 visual skill effects share the same casting marker.

Reactions and Persistent State. Reaction onset is marked by briefly inverting the affected fighter’s box fill and healthbar color. Bar length independently encodes health. This supports a protected hit with a reaction cue but no health decrease, or lifesteal with an additional update to the attacker’s health. Identity colors associate each reaction and state update with the affected fighter across side switches.

The same compiler processes video-derived annotations during training and engine-generated records at inference. Training reaction onsets are inferred from observed health decreases, while the engine specifies reaction events and health under the edited rules. The shared format therefore permits rule-dependent cue combinations beyond those implied by the training annotations. Rule-intervention experiments test whether the trained renderer realizes these combinations. Broad command stripes and four-frame inversions are designed to retain cues through spatial and temporal compression by the video variational autoencoder (VAE) [18, 19]. Detailed motion, textures, and effect appearance are supplied by the renderer.

## 3.3 Video Rendering and Identity Binding

We condition a pretrained video diffusion transformer on the proxy and appearance inputs. Video, reference, and proxy inputs share a VAE and patch embedding. Their tokens are concatenated in video–reference–proxy order, with learned channel-wise gates on the reference and proxy streams. The initial video context, proxy, and references remain clean, while future video latents are denoised. Text descriptions are encoded by the text encoder and supplied through cross-attention. Only video tokens are decoded.

Let $x ^ { \mathrm { i n i t } } = x _ { 0 }$ be the initial context frame and T the number of future video frames. The renderer samples

$$
\hat { x } _ { 1 : T } \sim p _ { \theta } \big ( \cdot \mid x ^ { \mathrm { i n i t } } , \pi _ { 0 : T } , r , c ) ,\tag{3}
$$

where r contains the available character or arena references and the target identity associated with each character reference. The text condition c describes appearance and skilleffect type. The proxy supplies executed interaction states and event schedules.

Proxy tokens share spatiotemporal rotary positions with the corresponding video tokens [32]. Reference tokens use positions outside the video grid, distinguishing appearance conditions from video-aligned interaction cues.

As illustrated in Fig. 2, shared identity embeddings bind character references to persistent entity identities. For each training sample, we assign the fighters distinct slots sampled without replacement from a learned codebook. Each fighter’s slot embedding is added to its corresponding tokens in the first video latent and throughout the proxy sequence, weighted by soft bounding-box coverage. The same embedding is added to all tokens of that fighter’s reference. Random slot assignment discourages a fixed mapping between codebook entries and appearances. At inference, the target identity attached to a character reference selects its slot embedding. Arena references do not use an entity slot. A training sample for character appearance editing contains a source initial image, a reference crop, and an edited target video. The source image establishes the scene and entity locations, while the reference specifies the target appearance in the continuation.

## 3.4 Training

We optimize the diffusion transformer, conditioning gates, and identity codebook with flow matching [22, 24]. The objective in Eq. (5) supervises only future video latents, while the VAE and text encoder remain frozen. We apply condition dropout [12] to proxy, reference, and text inputs. For reference-conditioned samples, appearance-description dropout encourages reliance on the reference image.

## 3.5 FightWorld Corpus

We construct FightWorld Corpus by pairing Street Fighter II [4] gameplay videos with interaction proxies and optional appearance references. A frozen video editor augments native recordings with edits to skill effects, character appearances, and arenas, yielding 112,618 training samples. As summarized in Fig. 3, the corpus spans 52 skill effects, 187 character-appearance variants, and 15 arenas.

Character and arena edits vary visual appearance while retaining the source interaction. We reuse source state annotations only after checking consistency in fighter trajectories, event timing, and health. Each accepted video is paired with a character crop or an arena reference, as illustrated for scene edits in Fig. 4.

Skill edits change the visual effect associated with a casting event and may also alter the visible interaction. We therefore extract state and event annotations from both native and skill-edited videos, retaining only clips that pass quality screening.

Native and edited videos share the same training format, with interaction proxies, text descriptions, and optional character or arena references as conditions. We use these samples to train a shared renderer across skill effects and appearances.

![](images/972fcf770f113e52d5fd08619dad64555ba7a22eb275b711d1ae6eaeac7cb91a.jpg)  
Figure 3: FightWorld Corpus. Native gameplay is expanded with skill effects, character-appearance edits, and scene edits, yielding 112,618 training samples.

Table 1: Effect of Identity Binding. Variants share reference inputs and training budgets, differing only in identity binding. Adherence is a percentage. Values are ranked within each test subset.
<table><tr><td>Test Subset</td><td>Binding</td><td>Adherence ↑</td><td>DINO Similarity ↑</td></tr><tr><td>Non-Switch</td><td>Off</td><td>78.43</td><td>0.47</td></tr><tr><td>Non-Switch</td><td>On</td><td>99.16</td><td>0.61</td></tr><tr><td>Side-Switch</td><td>Off</td><td>53.68</td><td>0.35</td></tr><tr><td>Side-Switch</td><td>On</td><td>98.27</td><td>0.60</td></tr></table>

## 4 Experiments

We evaluate appearance control, interaction fidelity, and rule customization. We compare FWM with adapted game world models and conduct controlled studies to examine the contributions of interaction encoding and identity binding.

## 4.1 Evaluation Setup

Implementation. We evaluate FWM on two-fighter interactions in the fixed 2D view of Street Fighter II [4]. Our renderer uses Wan2.2-TI2V-5B [34] as its backbone and is trained on FightWorld Corpus.

Evaluation Protocol. For recorded interactions, held-out video annotations specify the target interactions. For rule interventions, we fix the initial state and submitted actions, then rerun the engine under each rule configuration to compute trajectories, hit events, and state updates. FWM’s reference images and text condition c remain unchanged for damage scaling, limited-hit immunity, and lifesteal. Appearance evaluations change the references while retaining the executed interaction. Each controlled encoding comparison supplies identical interaction records to all variants.

![](images/40579ed33037a5bcbfa1630cfe24c114de89887d17282372303fedf41f049503.jpg)  
Figure 4: Scene Editing in FightWorld Corpus. Each scene pairs a clean reference with an edited training frame. References exclude fighters and HUD elements and may differ in background details.

Table 2: Interaction Fidelity across Control Representations. We compare variants with different conditioning signals to assess the effects of interaction information and its encoding. Rates are percentages, and Health MAE is in percentage points of maximum health. Across tables, bold and underlined values mark the best and second-best results.
<table><tr><td></td><td>Action ↑</td><td></td><td></td><td></td></tr><tr><td>Conditioning</td><td>Move / Attack / Skill</td><td>Reaction ↑</td><td>Health MAE↓</td><td>J↑</td></tr><tr><td>Actions Only</td><td>70.27 / 56.22 / 58.64</td><td>74.43</td><td>3.42</td><td>34.18</td></tr><tr><td>Actions + Rule Text Spatial Proxy Only</td><td>71.71 / 68.40 / 57.35 99.19 / 92.27 / 46.58</td><td>80.16 78.28</td><td>3.07 2.31</td><td>46.37 33.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial Proxy + State Tokens</td><td>94.48 / 91.23 / 82.42</td><td>87.17</td><td>1.23</td><td>63.28</td></tr><tr><td>Spatial Proxy + Text</td><td>96.72 / 93.63 / 92.43</td><td>92.24</td><td>2.78</td><td>76.35</td></tr><tr><td>FWM Interaction Proxy</td><td>99.78 / 99.24 / 99.06</td><td>99.79</td><td>0.15</td><td>98.48</td></tr></table>

Action and Reaction Metrics. Move Accuracy (MoveAcc) measures agreement between fighter motion and directional commands. Attack Accuracy (AttackAcc) is the fraction of detected punch or kick events supported by the corresponding input commands within their temporal windows. Skill Accuracy (SkillAcc) requires the correct skill class and an onset within the command-relative evaluation window. Reaction measures the fraction of expected hit responses visibly performed by the designated fighter. Qwen judges these responses from body motion in nearby frames, with the HUD excluded.

Health and Persistence. Health MAE measures the mean absolute error between generated and target HUD health, normalized by maximum health and reported in percentage points. The joint health check requires correct updates for both fighters and persistence of the resulting health values

![](images/7272835a81bee214ed0635b1ef6d99a7dc1221f18dc7b56c420f5ce60ad33f37.jpg)  
Figure 5: Character and Scene Reference Conditioning. Interaction rollouts across four character appearances (under default and side switched stances) and two arena references.

between events.

Joint Interaction Fidelity. Joint interaction fidelity (J) treats an attempt as successful only if all applicable action, entity, timing, reaction, and health checks pass. We average attempt-level results within each episode and give episodes equal weight. For rule interventions, $J _ { \mathrm { r u l e } }$ requires the applicable reaction and health checks to pass together.

Appearance and Entity Assignment. Adherence uses a VLM to judge whether the requested appearance is applied to the designated character or arena. The judge receives the target reference and a source-frame identity guide. DINO measures cosine similarity between reference and generated features for characters and arenas separately. Both metrics exclude the fixed initial context.

Baselines. We compare FWM with H3-WORLD [5], ReactiveGWM v1 [37], and ReactiveGWM v2 [38]. We adapt all three methods to FightWorld Corpus and its 52 skill effects. For skill control, we add a dedicated trigger to v1 and v2 and specify activation in H3-WORLD’s interval action text. The +Ref variants use our reference-conditioning branch for appearance editing.

![](images/4000266e8bacd795830262735d0646445583cc16539310db5a09639af7228624.jpg)  
Figure 6: Complete-Information Encodings on the Same Interaction. Held-out rollout under matched executed records (Table 2). Green brackets mark HUD state errors: ↓! unexpected health loss, ↓× missing health loss, delayed update, ↓̸= incomplete update, and ↑! unexpected recovery.

Table 3: Comparison with Adapted Game World Models. Baselines are adapted to FightWorld Corpus. +Ref denotes the addition of our reference-conditioning branch. Rates are percentages, and Health MAE is in percentage points of maximum health. DINO measures cosine similarity. A dash indicates an inapplicable metric.
<table><tr><td>Method</td><td>Action ↑ Move / Attack / Skill</td><td>Reaction ↑</td><td>Health MAE↓</td><td>Adherence ↑ Character / Scene</td><td>Character DINO ↑</td><td>Scene DINO ↑</td></tr><tr><td colspan="7">Without Reference-Image Conditioning</td></tr><tr><td>H3-WORLD</td><td>43.85 / 62.61 / 47.58</td><td>48.25</td><td>8.75</td><td></td><td></td><td></td></tr><tr><td>ReactiveGWM v1</td><td>70.15 / 78.98 / 48.55</td><td>35.56</td><td>4.59</td><td></td><td></td><td></td></tr><tr><td>ReactiveGWM v2</td><td>50.24 / 59.70 / 16.67</td><td>49.21</td><td>4.65</td><td></td><td></td><td></td></tr><tr><td colspan="7">With Reference-Image Conditioning</td></tr><tr><td>H3-WORLD + Ref.</td><td>36.91 / 62.70 / 42.03</td><td>42.22</td><td>7.57</td><td>88.10 / 100.00</td><td>0.47</td><td>0.68</td></tr><tr><td>ReactiveGWM v1 + Ref.</td><td>63.16 / 85.53 / 53.86</td><td>33.97</td><td>4.79</td><td>89.29 / 100.00</td><td>0.50</td><td>0.64</td></tr><tr><td>ReactiveGWM v2 + Ref.</td><td>72.22 / 74.10 / 34.78</td><td>57.14</td><td>4.04</td><td>91.67 / 100.00</td><td>0.41</td><td>0.65</td></tr><tr><td>FWM (Ours)</td><td>99.78 / 99.24 / 99.06</td><td>99.79</td><td>0.15</td><td>99.12 / 100.00</td><td>0.60</td><td>0.72</td></tr></table>

## 4.2 Character and Scene Customization

As illustrated in Fig. 5, FWM renders the same prescribed interaction with different character and arena references. We retain the executed records and proxy while changing appearance, testing whether the specified actions and consequences are preserved.

Identity Association across Side Switches. We further test whether character references remain associated with their intended fighters across side switches. This requires preserving identity independently of the fighters’ screen positions. The variants in Table 1 share reference inputs and training budgets, differing only in whether identity binding is enabled. Binding improves character adherence and DINO similarity, with an adherence gain of approximately 45 percentage points on side-switch clips.

![](images/e8db27ea4947758e63ce5df0f527baefce6d1271c9d3b209b6b06746ecb69f1a.jpg)

Figure 7: Dual-Character Actions and Responses. Top rails show visible action summaries for both fighters. Per-frame badges indicate realized ( ) or unrealized ( ) fighter actions and responses; ? denotes strategy-level NPC text in ReactiveGWM v1; HUD state icons follow Fig. 6.  
Table 4: Contributions of Proxy Cues. Each ablated variant omits one cue family and is retrained under the same training setting. Rates are percentages, and Health MAE is in percentage points of maximum health.
<table><tr><td></td><td>Action ↑</td><td></td><td>Health MAE↓</td><td></td></tr><tr><td>Variant FWM</td><td>Move / Attack / Skill</td><td>Reaction ↑</td><td></td><td>J↑ 98.48</td></tr><tr><td></td><td>99.78 / 99.24 / 99.06</td><td>99.79</td><td>0.15</td><td></td></tr><tr><td>w/o Command Marks w/o Damage-Event Cues</td><td>99.07 / 86.42 / 87.87 99.16 / 98.92 / 98.91</td><td>97.93 80.42</td><td>0.28 0.27</td><td>85.26 78.38</td></tr><tr><td>w/o Health Bars</td><td>99.84 / 99.08 / 99.12</td><td>98.31</td><td>2.18</td><td>57.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Skill Event Trigger</td><td>99.21 / 99.03 / 79.28</td><td>98.07</td><td>0.26</td><td>78.81</td></tr></table>

## 4.3 Control Representation Comparisons

From Commands to Executed Outcomes. Adding rule text to action commands improves joint fidelity, but the generated interactions remain inconsistent. Spatial Proxy Only accurately controls movement while showing low joint fidelity in Table 2. These conditions leave the renderer to infer hit events and health updates. The results motivate supplying executed outcomes alongside attempted actions.

![](images/18e5b131a1e0d3a8c32955691be91c7c9a9d644d78d93e91fa7696852233b818.jpg)  
Figure 8: Skill Timing and Event Consequences. Comparison on a projectile skill commanded at f = 20; HUD state icons follow Fig. 6.

Holding Executed Information Fixed. To isolate the effect of encoding, we compare Spatial Proxy + Text, Spatial Proxy + State Tokens, and FWM under identical executed interaction records. The records specify fighter identities, action classes, affected fighters, event onsets and durations, and frame-wise health trajectories. All variants share geometry, initial context, reference binding, backbone initialization, and training budgets.

Effect of Interaction Encoding. With identical executed information, FWM improves joint fidelity by approximately 22 percentage points over Spatial Proxy + Text, the stronger symbolic baseline in Table 2. State Tokens have lower average health error than Text but poorer joint fidelity, showing that average health accuracy does not fully capture interaction correctness.

Text misassigns damage to the attacker or misses the recipient’s health update in Fig. 6. State Tokens track health more closely on average but lag at impact onset and show local limb blur during contact. Both baselines supply events and health as symbolic records that must be associated with body regions, HUD bars, and video frames. FWM places these cues in the corresponding proxy regions and frames.

## 4.4 Comparison with Existing Methods

Action Control and State Consistency. ReactiveGWM v1 [37] achieves higher player movement and attack precision than H3-WORLD [5] in Table 3. It uses keyboard inputs for the player and coarser strategy text for the NPC. ReactiveGWM v2 [38] directs action instructions to either fighter, but still infers hit outcomes and health updates within video generation. The baselines miss actions and reactions in Fig. 7, and v2 shows spurious health recovery.

Multi-Phase Skills. The baselines struggle to synchronize skill effects with their consequences. In Fig. 8, ReactiveGWM v1 and v2 deduct health before projectile impact, while H3-WORLD shows insufficient health loss after contact. ReactiveGWM v2 also renders a malformed projectile effect.

Appearance and Interaction Fidelity. Adding appearance references improves v2’s movement and skill accuracy by approximately 22 and 18 percentage points, respectively.

![](images/e9aa7992d22f1a4f4f754927e9c09b7106a1f5fd3a83885deb33c1b67ed98e8e.jpg)  
Figure 9: Skill-Level Rule Mechanisms Beyond Native Combat. Five combat mechanisms absent from native Street Fighter II/III, spanning lingering ground damage, per-projectile hits, staggered area hits, action locks, and spatial confinement (boundary locked).

The +Ref baselines achieve high character adherence in Table 3, yet their hit reactions and health updates remain inconsistent. Garment blur and identity drift during rapid contact and side switches further indicate that overall adherence does not guarantee stable visual identity throughout an interaction.

Executed Outcomes as Visual Conditions. FWM improves action accuracy, hit reactions, health consistency, and character adherence over the adapted baselines in Table 3. Its Reaction score exceeds the best +Ref result by approximately 43 percentage points. The rollouts in Figs. 7 and 8 follow the prescribed action and reaction timing while preserving the resulting health across subsequent frames.

## 4.5 Rule Interventions

Rule and Skill Customization. We test whether generated interactions follow edited rules and newly composed skill mechanisms without retraining the renderer. For attacklevel interventions, the initial state, submitted actions, appearance references, and FWM text condition c remain fixed. The engine recomputes motion, hit timing, and state updates, and the changed outcomes reach FWM only through the proxy. FWM maintains highjoint fidelity across the revised rules in Table 5, while symbolic encodings degrade particularly under immunity and lifesteal.

Damage Magnitude. Damage scaling changes the health loss associated with a valid hit. FWM tracks the resulting health changes under both reduced and amplified damage, with low health error and high joint fidelity.

Limited-Hit Immunity. Limited-hit immunity blocks health loss for a specified number of hits while retaining reaction events. FWM renders the corresponding body responses with unchanged health during protection, then shows the prescribed health loss once protection is exhausted.

Lifesteal. Under lifesteal, a hit reduces the recipient’s health and restores the attacker’s health. FWM renders these coupled updates for the correct fighters and preserves the resulting health values after the hit.

Table 5: Interaction Fidelity under Rule Changes. Within each rule setting, encodings share executed records and are ranked separately. Default rules use original damage without immunity or lifesteal. Paired values denote reduced / amplified damage. Health MAE is in percentage points of maximum health, and $J _ { \mathrm { r u l e } }$ is a percentage.
<table><tr><td>Encoding</td><td>Health MAE↓</td><td> $J _ { \mathrm { r u l e } } \uparrow$ </td></tr><tr><td>Default Rules Spatial Proxy + State Tokens</td><td>1.87</td><td>89.26</td></tr><tr><td>Spatial Proxy + Text FWM</td><td>2.42</td><td>76.38</td></tr><tr><td>Damage Magnitude</td><td>0.16</td><td>100.00</td></tr><tr><td>Spatial Proxy + State Tokens</td><td>1.63 / 2.48</td><td>74.62 / 65.37</td></tr><tr><td>Spatial Proxy + Text</td><td>3.17 / 4.86</td><td>56.28 / 43.71</td></tr><tr><td>FWM</td><td>0.18 / 0.22</td><td>97.76 / 98.84</td></tr><tr><td>Limited-Hit Immunity</td><td></td><td></td></tr><tr><td>Spatial Proxy + State Tokens</td><td>4.28</td><td>49.36</td></tr><tr><td></td><td>8.73</td><td></td></tr><tr><td>Spatial Proxy + Text FWM</td><td>0.21</td><td>24.18</td></tr><tr><td></td><td></td><td>99.92</td></tr><tr><td>Lifesteal</td><td></td><td></td></tr><tr><td>Spatial Proxy + State Tokens</td><td>5.67</td><td>37.42</td></tr><tr><td>Spatial Proxy + Text</td><td>10.46</td><td>16.83</td></tr><tr><td>FWM</td><td>0.19</td><td>99.87</td></tr></table>

Multi-Stage Skill Mechanisms. We extend the rule interventions to skills that combine displacement, repeated impacts, and state constraints. In Fig. 9, Earthshatter couples knockback with staged health loss and lingering ground damage. Other mechanisms cover per-projectile hits, staggered area impacts, action locks, and spatial confinement. FWM renders these combinations through the shared proxy without additional skill-specific conditioning channels or renderer retraining.

## 4.6 Proxy Ablations

Attempted Actions and Their Consequences. We retrain each variant with one cue family removed, keeping the training setting unchanged. In Table 4, removing command marks lowers attack precision while leaving Reaction largely unchanged. Removing the skill trigger reduces skill accuracy by approximately 20 percentage points. These results support encoding attempted actions separately from their consequences.

Events and Persistent State. Removing damage-event cues lowers the Reaction score while health error remains low. Removing health bars largely preserves reactions but degrades health accuracy and joint fidelity. These complementary effects support separate conditioning for transient reactions and persistent state. More details, comparisons and visualizations are provided in the appendix.

## 5 Conclusion

We presented Fight World Model (FWM), which connects executable gameplay rules to video generation through a visual interaction proxy. Experiments in a combat setting demonstrate improved interaction and appearance fidelity over adapted game world models. FWM supports edited rules and multi-stage skill mechanisms without retraining the renderer. Character and arena customization preserves the prescribed interactions, with identity binding maintaining character associations across side switches.

## References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in Atari. In NeurIPS, 2024.

[2] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024.

[3] Ziqi Cai, Siqi Yang, Yimu Wang, Zixian Gao, Yunheng Liu, Shuchen Weng, Erwin Wu, Kaipeng Zhang, and Boxin Shi. MASS: Multiplayer world models with authoritative shared state, 2026.

[4] Capcom. Street fighter II. Video game, 1991.

[5] Danze Chen, Zeqing Wang, Ziyue Lin, Xingyi Yang, and Yeying Jin. H3-World: Turning language understanding into world control, 2026.

[6] Yiwen Chen, Guosheng Lin, and Chi Zhang. Code world model: Coding agent as world brain, 2026. Project page: https://buaacyw.github.io/cwm/.

[7] Decart and Etched. Oasis: A universe in a transformer. ht tps://oasis-model.github.io/, 2024.

[8] Zhiyang Deng, Boran Zhang, Danze Chen, and Yeying Jin. WorldMind: Decoupled game world model for state-aware NPC behavior, 2026.

[9] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, et al. Infinite worlds with versatile inter actions, 2026. LingBot-World 2.0 / LingBot-World-Infinity. [10] David Ha and Jürgen Schmidhuber. World models, 2018.

[11] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yi fan Zhang, et al. Matrix-game 2.0: An open-source, real time, and streaming interactive world model, 2025.

[12] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022.

[13] Fa-Ting Hong, Zunnan Xu, Zixiang Zhou, Jun Zhou, Xiu Li, Qin Lin, Qinglin Lu, and Dan Xu. Audio-visual controlled video diffusion with masked selective state spaces modeling for natural talking head generation. In 2025 IEEE/CVF In ternational Conference on Computer Vision (ICCV), pages 12549–12558. IEEE, 2025.

[14] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In CVPR, 2024.

[15] Li Hu, Guangyuan Wang, Peng Zhang, and Bang Zhang. WanToFight: Real-time generative game engine for multiplayer combat interaction, 2026.

[16] Zheng-Hui Huang, Guixu Lin, Jiacheng Lin, Yi-Chuan Huang, Ruihan Yu, Muyao Niu, Siqi Yang, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang, and Zhixiang Wang. Programmable world model, 2026. v2, 13 Oct 2026. https: //alaya-lab.github.io/pwm/.

[17] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. In ICCV, 2025.

[18] Diederik P. Kingma and Max Welling. Auto-encoding vari ational Bayes. In ICLR, 2014.

[19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[20] Guannan Li, Jiaji Chen, Jingyuan Liao, Yu Geng, and Baolan Qiu. MSR: Multiple subject reference for video generation, 2026.

[21] Zijun Lin, Zeqing Wang, Cheston Tan, Bihan Wen, and Yeying Jin. StatePlay: State-aware game world models for mechanics-consistent generation, 2026.

[22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

[23] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via crossmodal alignment, 2025.

[24] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.

[25] Ge Ya Luo, Zhi Hao Luo, Anthony Gosselin, Alexia Jolicoeur-Martineau, and Christopher Pal. Ctrl-V: Higher fidelity video generation with bounding-box controlled object motion, 2024.

[26] Mingshuang Luo, Shuang Liang, Zhengkun Rong, Yuxuan Luo, Tianshu Hu, Ruibing Hou, Hong Chang, Yong Li, Yuan Zhang, and Mingyuan Gao. DreamActor-M2: Universal character image animation via spatiotemporal in-context learning, 2026.

[27] Jack Parker-Holder and Shlomi Fruchter. Genie 3: A new frontier for world models. Google DeepMind blog, https: //deepmind.google/discover/blog/genie-3 -a-new-frontier-for-world-models/, 2025.

[28] Alexander Pondaven, Ziyi Wu, Igor Gilitschenski, Philip Torr, Sergey Tulyakov, Fabio Pizzati, and Aliaksandr Siarohin. ActionParty: Multi-subject action binding in generative video games. In ECCV, 2026.

[29] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, et al. SAM 2: Segment anything in images and videos, 2024.

[30] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, et al. Advancing open-source world models, 2026.

[31] Georgy Savva, Oscar Michel, Daohan Lu, Suppakit Waiwitlikhit, Timothy Meehan, Dhairya Mishra, Srivats Poddar, Jack Lu, and Saining Xie. Solaris: Building a multiplayer video world model in Minecraft, 2026.

[32] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568: 127063, 2024.

[33] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In ICLR, 2025.

[34] Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models, 2025.

[35] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guo qiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis, 2024.

[36] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A unified and flexible motion controller for video generation. In SIGGRAPH, 2024.

[37] Zeqing Wang, Danze Chen, Zhaohu Xing, Zizhao Tong, Yin han Zhang, Xingyi Yang, and Yeying Jin. ReactiveGWM: Steering NPC in reactive game world models, 2026.

[38] Zeqing Wang, Danze Chen, Zhaohu Xing, Zizhao Tong, Yinhan Zhang, Xingyi Yang, and Yeying Jin. ReactiveGWM: Flexible control and NPC reactivity in game world models, 2026. Code: https://github.com/INV-WZQ/Rea ctiveGWM.

[39] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory, 2026. Skywork AI. Weights: https://huggingface.co/Sky work/Matrix-Game-3.0.

[40] Zunnan Xu, Zhentao Yu, Zixiang Zhou, Jun Zhou, Xiaoyu Jin, Fa-Ting Hong, Xiaozhong Ji, Junwei Zhu, Chengfei Cai, Shiyu Tang, et al. HunyuanPortrait: Implicit condition control for enhanced portrait animation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15909–15919. IEEE, 2025.

[41] Wenhao Yan, Fengjia Guo, Zhuoyi Yang, and Jie Tang. SCAIL-2: Unifying controlled character animation with end to-end in-context conditioning, 2026.

[42] Xindi Yang, Baolu Li, Liam Lee, Zhenfei Yin, Songxin Zhang, Zhuoyang Song, Xu Jia, Jianfei Cai, et al. Oneira: From open-ended generation to open-world interaction in video world models, 2026.

[43] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. GameFactory: Creating new games with gen erative interactive videos. In ICCV, 2025.

[44] Xiaoyu Zhan, Xinyu Wang, Xiaohong Zhang, Huanjie Zhu, Tengjiao Sun, Pengcheng Fang, Jiaxing Yu, Yanwen Guo, and Dongjie Fu. Magpie: Real-time world renderer for in teractive games, 2026.

[45] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.

[46] Ruicheng Zhang, Mingyang Zhang, Jun Zhou, Xiaofan Liu, Zunnan Xu, Zhizhou Zhong, Puxin Yan, Haocheng Luo, and Xiu Li. MIND-V: Hierarchical world model for longhorizon robotic manipulation with RL-based physical align ment. arXiv preprint arXiv:2512.06628, 2025.

[47] Ruicheng Zhang, Guangyu Chen, Zunnan Xu, Zihao Liu, Zhizhou Zhong, Mingyang Zhang, Jun Zhou, and Xiu Li. RoboStereo: Dual-tower 4D embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639, 2026.

[48] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, and Yahui Zhou. Matrix-game: Interactive world foundation model, 2025.

[49] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-ID: Towards universal identitypreserving video synthesis, 2025.

[50] Zixiang Zhou, Zhentao Yu, Yifeng Ma, Hongmei Wang, Wenqing Yu, Cong Wang, Zilin Yang, Rui Chen, Jiarong Ou, Yezhou Liu, Yuan Zhou, and Qinglin Lu. Aura: Consistent multi-subject video generation via VLM-grounded semantic alignment, 2026.

[51] Shangwen Zhu, Qianyu Peng, Zhao Pu, Zhilei Shu, Xiangrui Ke, Zhaohu Xing, Zizhao Tong, Zeqing Wang, et al. Incantation: Natural language as the action interface for multi-entity video world models, 2026.

## A Implementation Details

## A.1 Engine and Timeline

The combat engine exposes damage magnitude, the number of protected hits, and the lifesteal rate as editable parameters. Its state retains protection counts and active skill windows across action intervals.

Engine updates indexed by k are mapped to output video frames t before proxy compilation. State samples and event windows use the same frame mapping, so an event onset is paired with its resulting state update. Frame index zero denotes the initial context, and $\hat { x } _ { 1 : T }$ denotes the continuation. The engine carries executable state between action intervals, while visual context conditions the next generated segment.

## A.2 Proxy Encoding Parameters

For light, medium, and heavy attacks, stripes cover the top one third, top two thirds, or full box interior below the coarse action band. They remain active over the command window. The skill-trigger block follows the caster for 30 output video frames, and reaction color inversions last four frames. Both durations are defined at the output video rate and resampled together with the state and event records.

## A.3 Renderer and Binding Parameters

We initialize the renderer from the Wan2.2-TI2V-5B checkpoint [34]. For $F$ video latent frames, character references use temporal coordinate $\tau _ { \mathrm { r e f } } = F + 2$ , and arena references use $\tau _ { \mathrm { r e f } } = F + 3 .$ . Their spatial coordinates lie outside the video grid. Video and proxy inputs use matched temporal indices and the same resize and center-crop transform.

The identity codebook contains six learned slots. Each embedding is normalized to unit root-mean-square magnitude and scaled by 0.25 at all three injection sites. For the first video latent and proxy sequence, soft boundingbox coverage weights the corresponding entity’s embedding, with weights normalized where boxes overlap. The target entity’s embedding is added to all character-reference tokens without a spatial box mask. Persistent entity IDs in the proxy identify the reference targets. Arena references receive no identity embedding.

## A.4 Training Objective and Configuration

Let z denote clean video latents and $\epsilon \sim \mathcal { N } ( 0 , I )$ . We construct noisy latents with flow matching [22, 24]:

$$
z _ { \tau } = ( 1 - \tau ) z + \tau \epsilon , \qquad \tau = \frac { 5 v } { 1 + 4 v } , \quad v \sim \mathcal { U } ( 0 , 1 ) .\tag{4}
$$

Here τ is flow time, distinct from engine index k and video-frame index t. The first latent is replaced with the clean source-image encoding at noise level zero. Future video tokens receive noise level $\tau .$ . With conditions ${ \mathcal { C } } =$ $( x ^ { \mathrm { i n i t } } , \pi , r , c )$ , the loss is

$$
\mathcal { L } = \mathbb { E } \left[ \frac { 1 } { \vert \Omega _ { + } \vert } \sum _ { j \in \Omega _ { + } } \| v _ { \theta } ( z _ { \tau } , \tau ; \mathcal { C } ) _ { j } - ( \epsilon - z ) _ { j } \| _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

where $\Omega _ { + }$ contains future video latent elements. Initialcontext, proxy, and reference tokens are excluded from the loss. At inference, future video latents start from noise.

We use AdamW with a constant learning rate of $8 \times 1 0 ^ { - 5 }$ weight decay of 0.01, and no learning-rate warmup. Training uses bfloat16 autocast with fp32 master parameters and an effective batch size of 128 on 128 H200 GPUs. The budget is 35 epochs, corresponding to 30,765 optimizer steps.

Each clip contains 89 frames at $9 2 8 \times 7 0 4$ resolution. Dropout probabilities for proxy, reference, text, and appearance descriptions are 0.1, 0.1, 0.1, and 0.2, respectively. The controlled encoding variants in Table 2 share initialization, training data, and optimization budget. Only the representation and its required input layers differ.

![](images/95105d2ebfae8f0eeabf71f057c004091a8d16cbb6296ec3bb060ec9cf9aedc6.jpg)

![](images/8f281b72be4a023721e06a32be2997cebddb11c23c638bc66613450509cbf876.jpg)

![](images/5b2918e6838726f454ce8c856ef4954f395865795b80ece2f4bb61c5ceb8a7fc.jpg)

c Transient Event, Persistent Consequence  
![](images/8a4dcbf7dea2dd77d46fe7378404d0a1414e46dde89afcdce1e304566138b134.jpg)

![](images/6e3c12853ee36446c79d1497b6b8fcb63d0ad96e1d0ed90d8a7f31d452c58c13.jpg)

![](images/e6c59a6ec5fed0d0e92659c0c656a61c12d717713a11e5d3e9b01ca5521a59dd.jpg)

![](images/4d32b1ef7da7be328a125fb37aeca7c4f1c39b508e17e02ac9c1c5662d52cee4.jpg)  
Figure 10: Interaction-Proxy Vocabulary. Transient cues and persistent state share fighter identities.

## B Data Construction Details

## B.1 Video Editing and Quality Control

A frozen MiniMax-H3 Ref2VA editor produces appearance and skill-effect variants using edit-specific prompts. Screening checks the requested content, temporal coherence, and interaction consistency.

Appearance-only variants are checked against the source video for fighter trajectories, event timing, and health traces. Only edits that pass these checks reuse the source state annotations. Skill-effect edits are annotated from the edited videos.

## B.2 State and Event Annotation

Gemma-4-31B-it localizes fighters at regularly spaced keyframes to initialize SAM 2 video propagation [29]. Semantic descriptions help distinguish fighters from similar background characters. At each keyframe, the tracker checks the two identity assignments against appearance and trajectory continuity, retaining or swapping them to correct drift and bridge short gaps. Bounding boxes specify spatial state. Facing direction follows relative fighter positions and retains its previous value while the boxes overlap.

Health traces are estimated from HUD bar lengths and cleaned within each round. Nearby decreases are merged into damage events, each recording the affected fighter and event time. Gemma also supplies coarse action and stance labels at selected frames, including frames near damage events. Unannotated action frames use the unknown label. Logged inputs provide fine attack commands and scheduled skill triggers for the controlled fighter. The compiler converts these annotations into interaction proxies.

## C Evaluation Protocols

## C.1 Controlled Interactions

Recorded-interaction tests use held-out video annotations, with source episodes separated between training and testing. Rule interventions rerun the engine from the same initial state and submitted actions under each configuration, recomputing motion trajectories, hit events, and state outcomes. These records define the target interactions for the compared encodings. In FWM’s damage scaling, limitedhit immunity, and lifesteal tests, reference images and text condition c remain fixed. Changed outcomes reach FWM only through the proxy.

Appearance interventions retain the executed records and proxy while changing the references. Evaluation cases include no-hit attempts, repeated attacks, overlap, and side switches.

Complete-Information Encodings. The encodings use matched event-free intervals, sampled values, and temporal indices, with shared resolution and sampling settings. Each uses the input layers required by its representation, and text inputs are checked for truncation.

## C.2 Action Recognition and Timing

Movement. MoveAcc compares tracked fighter motion with directional commands over valid command windows and averages across the four movement keys. Jump positives require an actual airborne jump. Grounded hurt recovery is excluded from jump positives.

Attack and Skill Recognition. We train ResNet-18-based temporal convolutional networks (TCNs) for attack and skill recognition. Attack detections are grouped into temporally contiguous events. AttackAcc measures the fraction of detected punch or kick events supported by the corresponding input command within its temporal window.

The skill classifier uses a 256-channel temporal trunk with dilation rates 1, 2, 4, and 8. It predicts onset and class, with a no-skill outcome in the onset prediction. Class logits are pooled over the predicted effect interval. SkillAcc requires the correct class and onset within the commandrelative window. The same timing rule applies across methods and accommodates skill startup, release, and recovery.

An attempted attack may hit or miss, so AttackAcc and Reaction assess command-grounded action execution and received-hit consequences, respectively.

## C.3 Reaction and Health

Visible Reactions. Reaction measures the fraction of expected hit responses visibly performed by the designated fighter within the evaluation window. Qwen judges body motion in surrounding gameplay frames, using a sourceframe identity guide and excluding the HUD.

Brief reactions can resemble ordinary motion across appearances and effects, so we judge them using local temporal context around expected events. Computing reaction precision would additionally require locating every visible received-hit onset throughout the generated video, a dense detection task that is difficult for current VLMs.

Health and Persistence. Let V contain readable fighterframe observations and H denote maximum health. Health MAE is the normalized mean absolute error, expressed in percentage points:

$$
\mathrm { M A E } _ { h } = \frac { 1 0 0 } { | \mathcal V | H } \sum _ { ( i , t ) \in \mathcal V } | \hat { h } _ { t } ^ { i } - h _ { t } ^ { i } | .\tag{6}
$$

Observed and target health values are obtained from HUD bar lengths. The joint health check uses a one-pixel tolerance and requires correct updates for both fighters, with the resulting values preserved between events. This includes unchanged health under immunity and the attacker’s restored health under lifesteal.

## C.4 Joint Interaction Fidelity

An attempt succeeds only when all applicable action, entity, timing, reaction, and health checks pass. The score J averages attempts within each episode and weights episodes equally:

$$
q _ { \ell e } = \prod _ { k \in { \mathcal { K } } _ { \ell e } } c _ { \ell e , k } , \qquad J = { \frac { 1 } { L } } \sum _ { \ell = 1 } ^ { L } { { \frac { 1 } { N _ { \ell } } } \sum _ { e = 1 } ^ { N _ { \ell } } { q _ { \ell e } } } .\tag{7}
$$

Here L counts episodes, $N _ { \ell }$ counts attempts in episode ℓ, and $c _ { \ell e , k }$ are binary checks indexed by $\kappa _ { \ell e }$ . The applicable set follows the registered interaction. Its checks cover command-grounded actions and timing, the identities of acting and affected fighters, visible body responses, and correct health updates with persistence. For rule interventions, $J _ { \mathrm { r u l e } }$ combines the required reaction and health checks. Both rates are expressed as percentages in the results.

## C.5 Appearance and Entity Assignment

Adherence assesses appearance-edit success for the designated character or arena. A VLM judges sampled continuation frames using the target reference and a source-frame identity guide. Frame judgments are combined into a cliplevel decision.

DINO measures reference-to-generation feature cosine similarity separately for characters and arenas. Proxy boxes localize fighter crops, while fighter and HUD regions are suppressed for arena scoring. The fixed initial context is excluded from both appearance metrics.

## D Reference-Augmented Baselines

We augment H3-WORLD [5], ReactiveGWM v1 [37], and ReactiveGWM v2 [38] with our reference-conditioning branch. Variants are evaluated on the same clips and frame indices as FWM, using explicit appearance references. The comparisons in Figs. 11 to 14 show character drift or garment blur during rapid contact and side switches, together with missed ( ↓ × ), premature ( ↓! ), or incomplete ( ↓̸= ) health updates. FWM preserves appearance and the prescribed health updates more consistently in these examples.

![](images/a00cb7cf6fed1c90fe105a42cd98965f0ec14ee14fd8dd3984b3daeecb80448b.jpg)

Figure 11: Reference-Augmented Comparison (Saitama, Player, No Side Switch). Matched frames corresponding to row 1 of Fig. 5; HUD state icons follow Fig. 6.  
![](images/c7febb3aeb9fe75202d849b50838ea49a640b564d5adb8ece6dbe997cb02b362.jpg)  
Figure 12: Reference-Augmented Comparison (Naruto, Player, Side Switch). Matched frames corresponding to row 2 of Fig. 5; HUD state icons follow Fig. 6.

![](images/fa81558a97aecbd5f72871f740867254a91e4e3722324f31e66f01e88ba63aa2.jpg)

Figure 13: Reference-Augmented Comparison (Lava Humanoid, NPC, No Side Switch). Matched frames corresponding to row 3 of Fig. 5; HUD state icons follow Fig. 6.  
![](images/1822738ea23d695a62ba93457b7e68c98c63fe2453b6bc2629dfd9c6ea0205bc.jpg)  
Figure 14: Reference-Augmented Comparison (Superman, NPC, Side Switch). Matched frames corresponding to row 4 of Fig. 5; HUD state icons follow Fig. 6.