# Edited Endpoints as Partial Goals for Image-to-Video Generation

Sihyun Kim<sup>1</sup>, Sanghyun Jo<sup>1,2,†</sup>, Kyungsu Kim<sup>1,†</sup>

<sup>1</sup>Seoul National University <sup>2</sup>OGQ

<sup>†</sup>Corresponding authors: shjo.april@gmail.com, kyskim@snu.ac.kr

Edited endpoint images make the target of an image-to-video transition explicit and can improve transition completion, but also preserve context in the source image’s layout. Full-frame endpoint conditioning turns this preserved context into requirements on the video’s final state, potentially inducing temporal artifacts. We measure discontinuity with Hard-Cut (HC), late background reversal with Background Motion Reversal (BMR), and motion suppression with Dynamic Degree (DD). These artifacts appear across five video models spanning training-free and learned endpoint conditioning, in model-specific forms. We study Partial, a selective conditioning scheme that constrains only an edit-relevant region estimated from image-editor attention and leaves the remaining terminal content to the video model. Across three backbones on TC-Bench-I2V, Partial reduces temporal artifacts relative to full-frame conditioning, with no significant difference in Transition Completion Ratio (TCR); on Wan2.2-Turbo, uniformly weakening conditioning does not reproduce these reductions. A donor-context intervention keeps full-frame conditioning and replaces only the context outside the edit-relevant region with the final frame of a Partial video. It lowers HC and BMR on HunyuanVideo-1.5 and lowers BMR while restoring DD on Wan2.2-Turbo, with no significant difference in TCR, so constraint extent alone does not account for these artifacts. Together, these findings suggest that edited endpoints should serve as partial future goals, although edit relevance does not guarantee future validity. Code and data will be released at the link below.

Project: wannagetacat.github.io/Edited-Endpoints-I2V Github: github.com/wannagetacat/Edited-Endpoints-I2V Lab: aibl.snu.ac.kr

![](images/77aeda3e98abaef9927f78d34d68caadbee5457782b38ce414cfa7a8a426a334.jpg)

## 1 Introduction

Given a source image, “change the bus to purple” specifies a target appearance but not a trajectory. A video may miss or undo the change before its last frame [10]. Editing the source image makes the target explicit, as edited keyframes already do in video editing and interpolation [19, 21, 23, 26, 30].

An edited endpoint mixes two kinds of information: the purple bus specifies the target appearance, while the unchanged road, buildings, and sign preserve the editor’s input in the source image’s layout. When an edited frame shares its source frame’s time, as in edit-then-propagate video editing, this alignment makes it a reliable anchor. As an endpoint, the same context is imposed at a later time: full-frame terminal conditioning requires the road, buildings, and sign to return to their source-time positions at the video’s end. We call this promotion of preserved context to spatial future requirements spatial over-specification.

Consistent with this concern, full-frame endpoint conditioning can improve completion while suppressing motion or increasing discontinuity and background motion reversal, under both training-free projection and learned endpoint conditioning (Tables 1 and C11). We study whether terminal content affects temporal artifacts beyond constraint extent.

Using established masked conditioning [1, 25], we test content at fixed full-frame support and vary support area, placement, and global mixing weight. Partial constrains the terminal frame to the edited endpoint only within fixed edit-relevant support estimated from image-editor attention, retaining the video’s prediction elsewhere (Fig. 1).

Full-frame endpoint conditioning and Partial differ in both the imposed terminal content and the constraint extent. To test content at fixed extent, we run a donor-context intervention: inside the edit-relevant support M<sub>0</sub>, we keep the original edited endpoint; outside $M _ { 0 } ,$ , we substitute the final frame of a paired Partial rollout, the donor, and constrain the entire terminal frame with the same full-frame projection.

(b) Terminal conditioning  
![](images/bd39db7568c38028a26e6200394448a945db745a3c08bc13f0b82f7caa051df7.jpg)  
Figure 1. Edited endpoints preserve context that need not specify the video’s future. Full-frame endpoint conditioning constrains both requested appearance and preserved context. Partial constrains only estimated edit-relevant support $M _ { 0 }$ from image-editor attention (blue outline); washed-out positions retain the video prediction. Source frame: DAVIS bus [32], via FiVE-Bench [20].

Across 103 TC-Bench-I2V prompts with five paired seeds per condition, we report three findings:

1. Terminal content matters beyond constraint extent. With full-frame support unchanged, the donor-context intervention lowers Hard-Cut (HC) and Background Motion Reversal (BMR) scores on HunyuanVideo-1.5, and lowers BMR while restoring Dynamic Degree (DD) on Wan2.2-Turbo (Fig. 3).

2. Support area and placement yield distinct completion–temporal trade-offs. On Wan2.2-Turbo, edit-relevant placement favors completion at matched area; shrinking support lowers HC, whereas enlarging it raises BMR. Tested global softening does not reproduce Partial’s artifact reductions.

3. Partial reduces temporal artifacts at comparable completion. Across three backbones, Partial mitigates temporal artifacts relative to full-frame endpoint conditioning at comparable Transition Completion Ratio (TCR), and improves completion over Endpoint-free on Wan2.2-Turbo and SANA-Video. Restricting gradient-based Frame Guidance [14] to the same support likewise keeps its completion gain and removes its motion suppression. Partial’s residual failures show that edit relevance need not imply future-valid position or visibility.

## 2 Related Work

Frame guidance and masked conditioning. Frame Guidance optimizes video latents using selected decoded frames, including edited-frame inputs and binary-masked $L _ { 2 }$ guidance [14]. Our projection follows RePaint’s mixing of generated and noise-matched known pixels and Blended Latent Diffusion’s mixing of generated and source-image latents [1, 25]. We use this established mixing principle to analyze terminal constraints, rather than introduce a new guidance operator. Projection lets us vary terminal content, constraint extent, and placement within one fixed sampler. With its frame loss restricted to the same edit-relevant support, Frame Guidance also keeps its completion gain and recovers the motion that full-frame guidance suppresses (Section 5.6).

Edit localization and regional control. Prompt-to-Prompt connects prompt words with image layout through cross-attention [12]. DiffEdit derives masks from contrasts between prompt-conditioned noise predictions, whereas InstDiffEdit derives attention masks during editing [7, 47]. MasaCtrl and MAG-Edit use masks for attention control and local latent optimization, respectively [4, 27]. EditCtrl combines local masked and global temporal context for video editing [22]. For video generation from a single image, we use image-editor attention to estimate edit-relevant support on the terminal frame.

Edited keyframes, target states, and inbetweening. Prior work uses edited or specified frames as appearance anchors to propagate or as states to connect, typically imposing the whole frame at its time index. We instead impose an edited frame at a different time from its source, with no trajectory relating the two, and ask which parts of its spatial content remain valid future requirements. Edit-then-propagate video editing edits the first frame with an image editor and carries it along a source video that supplies motion [19, 23, 30]. TC-Bench’s SDXL+SEINE baseline and MaskINT edit keyframes and interpolate between them [6, 10, 26, 33]. Inbetweening methods connect specified start and end states through fine-tuned or training-free bidirectional generation [9, 37, 42, 44]. Wan2.1-FLF2V, Many-for-Many, and

OmniWeaving instead learn first–last-frame or flexible frame conditioning [21, 31, 36]. VACE, Many-for-Many, and VideoCanvas support spatially masked video conditioning [3, 15, 21], offering learned approaches to partial goals.

Temporal artifacts of conditioning. Conditional image leakage links limited I2V motion to source-image over-reliance at large diffusion timesteps [46]. InfinityStory’s supplementary ablations report restricted natural motion from fixing each shot’s keyframe and rushed transitions toward prescribed positions [8]. DCARL addresses keyframe duplication and boundary discontinuity with noisy conditioning and boundary consistency [29]. TIC-FT diagnoses abrupt content and noise-level transitions under direct concatenation and uses graded-noise buffer frames [18]. Building on these observations, our donor-context intervention tests whether replacing editor-preserved terminal context changes temporal artifacts under unchanged full-frame support and projection (Section 5.2).

## 3 Problem Formulation and Method

## 3.1 Problem Setting

Given a source image $I _ { 0 }$ and a transition prompt, an image editor produces an endpoint $I _ { E }$ from an edit prompt that embeds the transition prompt and asks for the frame after the transition completes (Appendix $\mathrm { A } )$ . A frozen video model generates $V = \left( I _ { 0 } , \ldots \ldots , I _ { T } \right)$ from the source and the same transition prompt under three conditions: Endpoint-free, full-frame endpoint conditioning (Full), and partial endpoint conditioning (Partial). Within each controlled comparison, conditions share the source and prompt; endpoint-conditioned conditions share $I _ { E }$ except in Section 5.2.

## 3.2 Edited Endpoints and Partial Goals

An idealized decomposition makes the mismatch explicit:

$$
{ \cal I } _ { E } \approx M ^ { * } \odot { \cal I } _ { T } ^ { \mathrm { e d i t } } + \left( 1 - M ^ { * } \right) \odot { \cal I } _ { 0 } ,\tag{1}
$$

where $M ^ { * }$ is an idealized edit-relevance mask and $I _ { T } ^ { \mathrm { e d i t } }$ denotes the requested target appearance. This decomposition represents an editor that changes requested content while preserving context in the source layout. The second term adds no appearance beyond $I _ { 0 } ,$ , which already conditions the video; as a terminal requirement, it only fixes where source content must be at time T. Only the first term carries information that $I _ { 0 }$ lacks. Full treats both terms as future requirements. Yet preserved context may change position or visibility as the scene evolves, so forcing it back can impose constraints beyond the edit prompt: spatial over-specification. Our FLUX.2 endpoints follow this pattern: outside the estimated support $M _ { 0 }$ (Section 3.3), they differ from $I _ { 0 }$ by a mean absolute RGB difference of 0.056, versus 0.232 inside (Appendix E.5).

Edit relevance and future validity. Edit relevance associates content with the requested change; future validity concerns its appropriateness as a future constraint. Partial uses the former to select support, without estimating the latter. Neither $M ^ { * }$ nor its estimate below is a future-validity mask.

## 3.3 Edit-Relevant Support Estimation

We estimate binary support $M _ { 0 }$ from the editor’s text–image attention to object and edit phrases preselected from each prompt and fixed across conditions. We normalize and average their attention maps, then apply a two-component Gaussian mixture with an Ising spatial prior (GMM–Ising; Appendix A; Fig. A1).

The object and edit phrases connect the requested change to spatial regions through the editor’s attention, so $M _ { 0 }$ approximates $M ^ { * }$ in Eq. (1) without a separate segmentation model.

## 3.4 Full, Soft, and Partial Conditioning

Partial imposes the endpoint only on $M _ { 0 }$ , keeping the content that $I _ { 0 }$ lacks, and retains the video prediction elsewhere because future context is unknown.

Weighted terminal projection. Let $z _ { E } = E ( I _ { E } )$ encode the endpoint, u index spatial latent positions, and $\hat { z } _ { T } ^ { ( k ) }$ denote the clean prediction of the terminal latent frame at sampling step k. All conditions share one projection with a spatial weight w:

$$
\hat { z } _ { T } ^ { ( k ) } ( u ) \gets w ( u ) z _ { E } ( u ) + [ 1 - w ( u ) ] \hat { z } _ { T } ^ { ( k ) } ( u ) ,\tag{2}
$$

with $w \equiv 1$ for Full, $w \equiv \alpha \in \{ 0 . 7 5 , 0 . 5 , 0 . 2 5 \}$ for Soft, and $w = M _ { 0 }$ for Partial, after transformation to each latent grid. Soft weakens every terminal position uniformly; α is a mixing weight, not a calibrated measure of overall constraint strength. Support stays fixed in terminal spatial coordinates throughout sampling and does not track motion.

Backbone implementations. Wan2.2 applies the same w to the terminal model input, with token timesteps scaled by 1 − w. SANA Partial converts the projected clean prediction back to its solver’s flow, while SANA Full clamps the terminal state. Hunyuan uses a global timestep, so it projects toward the endpoint noised to the current level after each scheduler update. Appendix A specifies encoding, mask transforms, and noise handling. All controlled comparisons are paired within a backbone, so these implementation differences do not enter them. Learned endpoint-conditioning models are additional observational baselines, not implementations of this projection.

## 4 Main Experiments

4.1 Experimental Setup and Metrics. We evaluate the 103 of TC-Bench-I2V’s 120 prompts whose source videos remain available, with five paired seeds per condition (515 videos) [10], using FLUX.2 endpoints; prompt IDs follow TC-Bench. The primary backbones are Wan2.2-Turbo (Wan2.2), SANA-Video 2B [SANA; 5], and HunyuanVideo-1.5 [Hunyuan; 39]. Wan2.2-Turbo is quanhaol’s community-distilled Wan2.2-TI2V-5B-Turbo checkpoint.<sup>1</sup> Appendix A gives sampler configurations.

Image-editing prior and comparison logic. FLUX.2-klein-base-9B edits the source image into a plausible final frame after the requested transition has completed (Appendix A.2). We generate one endpoint per prompt and reuse it for Full and Partial. Endpoint-free receives the source image and transition prompt; Full additionally constrains the entire terminal frame to the edited endpoint, whereas Partial constrains only $M _ { 0 }$ . Thus Endpoint-free versus Full tests the completion benefit and temporal artifacts of adding an explicit image-editing goal, while Full versus Partial tests selective use of that same goal. Partial requires no video-model fine-tuning.

Related baselines. Table 1 also summarizes selected comparisons from Tables G1 and C11. Within the Wan2.2 family, we also include the non-distilled Wan2.2 Base and Wan2.2 with Qwen-Image-Edit endpoints [36, 40]. OmniWeaving, built on HunyuanVideo-1.5, and MfM provide learned endpoint-conditioning baselines [21, 31]. The three controlled blocks compare conditions within each backbone; the additional methods use their native configurations, with edit-type breakdowns and timing in Table G1.

Completion and temporal diagnostics. We report Transition Completion Ratio (TCR) and TC-Score with a Qwen2.5- VL-32B-Instruct judge [2]. TCR conjoins transition and object-consistency assertions, excluding “other objects” as in TC-Bench [10]. TC-Score excludes transition-object consistency from its VLM term and adds adjacent-frame CLIP consistency [34], which also varies with motion, so we use TCR as the primary completion measure.

Temporal diagnostics use a common 8 FPS grid. Dynamic Degree (DD) detects substantial movement using VBench’s RAFT-based decision [13, 35]. Discontinuities and camera drift can raise flow scores [11, 28], so DD does not assess motion appropriateness. Hard-Cut (HC) measures the largest drop of the adjacent-frame similarity $s _ { t }$ below its median, where $\phi$ is the CLIP image encoder and $I _ { t }$ the t-th frame on this grid:

$$
{ \mathrm { H C } } = { \mathrm { m e d i a n } } _ { t } s _ { t } - \operatorname* { m i n } _ { t } s _ { t } , \qquad s _ { t } = \cos { \bigl ( } \phi ( I _ { t } ) , \phi ( I _ { t + 1 } ) { \bigr ) } .\tag{3}
$$

BMR measures the fraction of late along-axis motion mass opposing an early reference direction in the region outside the original estimated support $M _ { 0 } ,$ , which approximates the background and stays fixed across conditions, including shifted/random conditioning supports. Let $F _ { t } ( x )$ denote WAFT flow [38] and L the number of flow pairs; the median is taken componentwise. The magnitude-weighted vector $b _ { t }$ , reference axis $\hat { b } ,$ and reversal score are

$$
\begin{array} { r l } & { b _ { t } = \frac { \sum _ { \boldsymbol { x } \notin { \cal M } _ { 0 } } \Vert \boldsymbol { F } _ { t } ( \boldsymbol { x } ) \Vert F _ { t } ( \boldsymbol { x } ) } { \sum _ { \boldsymbol { x } \notin { \cal M } _ { 0 } } \Vert \boldsymbol { F } _ { t } ( \boldsymbol { x } ) \Vert } , \qquad \hat { b } = \frac { b _ { \mathrm { r e f } } } { \Vert b _ { \mathrm { r e f } } \Vert } , \quad b _ { \mathrm { r e f } } = \mathrm { m e d i a n } _ { t < L / 2 } b _ { t } , } \\ & { \mathrm { B M R } = \frac { \sum _ { t \ge L / 2 } \mathrm { m a x } ( 0 , - \langle b _ { t } , \hat { b } \rangle ) } { \sum _ { t \ge L / 2 } | \langle b _ { t } , \hat { b } \rangle | } . } \end{array}\tag{4}
$$

BMR lies in [0, 1] and depends on the estimated mask and flow. Section 5.7 gives track-based checks.

Table 1. Image-editing goals: completion and temporal trade-offs. TC-Bench-I2V, 103 prompts × five paired seeds. Endpoint-free uses the source image and prompt only. Full and Partial share the same FLUX.2-klein-base-9B edited endpoint; their constraints cover the entire terminal frame and $M _ { 0 } ,$ respectively. Related baselines use the endpoint prior named in each row.
<table><tr><td rowspan="2">Condition / endpoint prior</td><td colspan="3">Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td></td><td>TCR ↑ ∆TCR (pp) ↑ TC-Score ↑</td><td></td><td>HC↓</td><td>DD↑</td><td>BMR↓</td></tr><tr><td>Wan2.2-Turbo</td><td colspan="5"></td></tr><tr><td>Endpoint-free</td><td>33.79</td><td></td><td>0.7841</td><td>4.57</td><td>79.4</td><td>0.360</td></tr><tr><td>Full (FLUX.2)</td><td>41.94</td><td>+8.16*</td><td>0.8352</td><td>6.47</td><td>66.8</td><td>0.599</td></tr><tr><td>Partial (FLUX.2)</td><td>42.91</td><td>+9.13*</td><td>0.8187</td><td>5.76</td><td>76.7</td><td>0.387</td></tr><tr><td>∆ (Partial − Full)</td><td>+0.97</td><td></td><td>-0.0164*</td><td>-0.72*</td><td>+9.90*</td><td>-0.213*</td></tr><tr><td colspan="5"></td><td></td><td></td></tr><tr><td>SANA-Video Endpoint-free</td><td>23.50</td><td></td><td>0.7530</td><td>8.07</td><td></td><td>0.122</td></tr><tr><td>Full (FLUX.2)</td><td>35.73</td><td>+12.23*</td><td>0.7944</td><td>9.77</td><td>42.7 32.8</td><td>0.420</td></tr><tr><td>Partial (FLUX.2)</td><td>35.34</td><td>+11.84*</td><td>0.7935</td><td>8.80</td><td>42.3</td><td>0.135</td></tr><tr><td>∆ (Partial − Full)</td><td>-0.39</td><td></td><td>-0.0008</td><td>-0.97</td><td>+9.51*</td><td>-0.285*</td></tr><tr><td colspan="5">HunyuanVideo-1.5</td><td></td><td></td></tr><tr><td>Endpoint-free</td><td>40.97</td><td></td><td>0.8718</td><td>3.47</td><td>51.5</td><td>0.305</td></tr><tr><td>Full (FLUX.2)</td><td>46.60</td><td>+5.63</td><td>0.8588</td><td>12.96</td><td>52.6</td><td>0.585</td></tr><tr><td>Partial (FLUX.2)</td><td>44.47</td><td>+3.50</td><td>0.8804</td><td>5.69</td><td>51.5</td><td>0.333</td></tr><tr><td>∆ (Partial — Full)</td><td>-2.14</td><td></td><td>+0.0216*</td><td>-7.27*</td><td>-1.17</td><td>-0.252*</td></tr><tr><td colspan="5">Related I2V and endpoint baselines (Tables G1 and C11)</td><td></td><td></td></tr><tr><td>Wan2.2 Base (FLUX.2)</td><td>45.24</td><td>+14.17*</td><td>0.8475</td><td>7.18</td><td>57.9</td><td>0.497</td></tr><tr><td>Wan2.2-Turbo (Qwen-Image-Edit)</td><td>35.15</td><td>+1.36</td><td>0.8001</td><td>7.11</td><td>64.1</td><td>0.617</td></tr><tr><td>OmniWeaving (FLUX.2)</td><td>43.30</td><td>+3.88</td><td>0.8647</td><td>5.65</td><td>36.9</td><td>0.487</td></tr><tr><td>MfM (FLUX.2)</td><td>43.88</td><td>+11.26*</td><td>0.8869</td><td>4.13</td><td>28.2</td><td>0.387</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

∆TCR is the paired difference from Endpoint-free; ∆ (Partial − Full) rows give paired Partial − Full differences in each column, computed from unrounded per-video values. <sup>∗</sup>Significant; intervals in Tables C1 and C8; Partial − Full TC-Score intervals are [−0.028, −0.005], [−0.017, +0.015], and [+0.012, +0.031] for Wan2.2, SANA, and Hunyuan. Bold and underline mark the best and second-best value within each backbone block. TCR/DD are percentages (differences in pp); HC is multiplied by 100. HC, BMR, and DD are temporal diagnostics: HC and BMR measure discontinuity and background reversal directly, while DD measures movement, so a drop relative to Endpoint-free indicates motion suppression.

Uncertainty. We report 95% confidence intervals (CIs) from 100,000 paired prompt-bootstrap draws over 103 prompts (Appendix C). A paired difference is significant when its pointwise CI excludes zero; otherwise we call the conditions comparable, which does not imply equivalence.

4.2 Endpoint-Free vs. Full Conditioning. For Endpoint-free→Full (Table 1), TCR rises significantly by 8.16 percentage points (pp) on Wan2.2 and 12.23 pp on SANA. DD decreases by 12.62 and 9.90 pp, respectively, while HC and BMR increase on both. Full supplies useful target information while inducing temporal artifacts.

Hunyuan’s principal artifacts are discontinuity and background reversal: HC rises from 3.47 to 12.96 and BMR from 0.305 to 0.585, while DD is 51.5% for Endpoint-free and 52.6% for Full. Thus, Full’s temporal artifacts differ across settings. These observations are consistent with spatial over-specification: outside $M _ { 0 }$ , the first–last-frame difference is 0.13–0.23 for Endpoint-free videos but 0.06–0.08 under Full, which returns preserved context near its source-time layout (Appendix E.5).

Beyond these backbones, FLUX.2 Full conditioning significantly raises TCR on Wan2.2 Base (31.07% to 45.24%, +14.17 pp) and MfM (32.62% to 43.88%, +11.26 pp), whereas the gains of OmniWeaving (39.42% to 43.30%, +3.88 pp) and of Wan2.2 with Qwen-Image-Edit endpoints (33.79% to 35.15%, +1.36 pp) are not significant (Table C8). An edited endpoint can aid completion across conditioning approaches, but the magnitude and certainty of the gain vary across models and editors. Full’s temporal artifacts also persist under learned endpoint conditioning: Full sharply lowers DD by 20.19 pp for MfM and 17.67 pp for OmniWeaving, and raises OmniWeaving’s BMR (Table C11).

4.3 Partial vs. Endpoint-Free and Full. Relative to Full, Partial mitigates temporal artifacts at comparable TCR. Across Wan2.2, SANA, and Hunyuan, TCR stays within 2.14 pp of Full. HC, DD, and BMR each move toward their Endpoint-free levels, and every significant temporal difference favors Partial (Fig. 2; Tables 1 and C1). The BMR reduction holds within each edit type (Table C10). Outside $M _ { 0 } .$ , Partial leaves the first–last-frame difference of preserved context within 0.02 of Endpoint-free (Appendix E.5).

(a) Partial − Full: temporal artifacts decrease  
![](images/8461b142d630c2548cae21cace61095ef4b0cc4384f1d7769294b6983f67fb73.jpg)

(b) Partial − Endpoint-free: completion rises  
![](images/ef7d45ee864fab22bd798ce02db315555a59b3c84b1d8915856334474cf74682.jpg)  
Figure 2. Partial reduces temporal artifacts relative to Full. Paired differences with 95% CIs over 103 prompts and five seeds; open markers are not significant. Tables 1 and C1 give levels and intervals.

The mitigation is not a return to Endpoint-free. Partial keeps a significant TCR gain over Endpoint-free on Wan2.2 and SANA (+9.13 and +11.84 pp), and on Wan2.2, global softening at comparable TCR does not reproduce these reductions (Section 5.3). Some temporal artifacts remain relative to Endpoint-free (Table 1). Edit-type breakdowns are in Table C9.

## 5 Controlled Analysis

Section 5.1 localizes Hunyuan’s Full discontinuity at the terminal boundary, and Section 5.2 uses a donor-context intervention to test whether terminal content, not only constraint extent, drives Full’s temporal artifacts. We run the intervention on Hunyuan and on Wan2.2. Wan2.2’s Full artifacts are mainly motion suppression and background reversal, so the Wan2.2 run tests whether the donor effect extends beyond a boundary discontinuity. On Hunyuan, where the discontinuity arises at this boundary, cross-seed donors test how much of the effect depends on matching the recipient trajectory. We use Wan2.2 for global softening, support area, and placement controls because its shared conditioning path applies the same weight $( w = 1 , \alpha , \mathrm { o r } M _ { 0 } )$ to model input and clean-prediction projection, allowing these factors to vary within one mechanism. Section 5.6 then applies the same support to gradient-based guidance on another backbone.

5.1 Localization of Terminal Discontinuity. Unlike Wan2.2 and SANA, Hunyuan Full sharply increases HC without reducing DD (Section 4.2), and the discontinuity is concentrated at the terminal boundary: the adjacent-frame pair entering Hunyuan’s terminal latent slot is the minimum-similarity pair in 95.9% of Full videos, versus 1.7% for Endpoint-free (Appendix E.1). Full therefore preserves ordinary motion while introducing a localized discontinuity where the endpoint is imposed. If this discontinuity depends on what the endpoint imposes, not only on where, replacing terminal content at fixed support should reduce it.

5.2 Donor Context under Fixed Full Support. Full and Partial differ in both imposed content and constraint extent, so we vary terminal content alone. Original Full (O) conditions on the endpoint latent $z _ { E }$ . A paired Partial rollout (P) with the same source, prompt, and seed supplies its final RGB frame $I _ { T } ^ { P }$ . Donor-context Full (D) combines original content inside estimated support with donor context outside:

$$
z _ { D } = M _ { 0 } \odot z _ { E } + ( 1 - M _ { 0 } ) \odot E ( I _ { T } ^ { P } ) ,\tag{5}
$$

where E is the backbone’s terminal-frame encoding, also used for $z _ { E } \ ( \mathrm { A p p e n d i x \ D } )$ . Here $M _ { 0 }$ selects the source of terminal content, not projection support. Within each backbone, O and D constrain all terminal spatial positions with

## (a) Only the context outside $M _ { 0 }$ changes

Prompt: “A yellow spoon turns green when it is soaked into a glass of ice water.”

![](images/b8340645813e2083b0c5c23b5524badcc8884ef0485a3ceafae6b8fa16451c0c.jpg)

![](images/1bd8b1d5f7b63ae87c6c39e893c7c514193bc7781349c178d01170fb9d86f6db.jpg)

![](images/1bf3a6d0adb394895582ee486c9e9fc297a412e88c14e65e28b3cb797c8809e5.jpg)  
✓ both use the same Full projection and recipient seed; a cross-seed donor comes from another seed of the same prompt

## (b) Donor-context Full − Original Full at fixed Full support

Partial − Original Full

![](images/db8b1f3225c938f66bdcbd23a7e959af4532b6ee44a33ae9812588853b0f6832.jpg)

![](images/ff46b7ed697ccc9d0335914c473bf87c1df9025915f22474c32811c876a4aa16.jpg)

![](images/d950338fd83f7f93593d1e3845a3a2f4f54d61a478c794b7e4d069d66455aeb5.jpg)

![](images/505c43f3bb5fc7bc04e7dd8bd7bcd3620ea464ca1901c5f4d895ec1bad78a7bf.jpg)

## (c) Example: the Full hard cut disappears

## Original Full

## Donor-context Full

![](images/274bd526bd7f8af65b4ec38e58f9a202db3c39363d5b754119ca2d3d802aa9aa.jpg)

![](images/99df2943c6f1b32a3509c695c1ec7a240e98bc1f32f63c52af18622b6dcb4a46.jpg)  
Figure 3. Terminal content, not only constraint extent, affects temporal artifacts. (a) The donor-context endpoint keeps the edited endpoint inside $M _ { 0 }$ (blue outline) and uses the final frame of a paired Partial rollout outside it (orange tint). Original and Donor-context Full share the Full projection and the recipient seed; a cross-seed donor comes from another seed of the same prompt. (b) Donor-context minus Original Full with paired 95% CIs over 103 prompts and five seeds; open markers are not significant; gray ticks mark Partial minus Original Full. (c) HunyuanVideo-1.5, prompt 28, seed 22004: native frames and consecutive-frame CLIP similarity; the Original Full discontinuity between frames 114 and 117 is absent under Donor-context Full.

the same Full support, projection, sampler, source, prompt, and paired noises. The donor requires an extra rollout; it is not ground-truth future context and need not fit the new Full rollout.

Terminal content affects temporal artifacts at fixed extent. On Hunyuan, D significantly lowers HC from 12.96 to 8.10 and BMR from 0.585 to 0.358 (paired differences −4.86 and −0.227; relative reductions of 37.5% and 38.8%), while TCR and DD remain comparable. On Wan2.2, D significantly lowers BMR by 0.238 and raises DD by 9.90 pp, while TCR and HC remain comparable; TC-Score decreases slightly (Fig. 3b; Appendix D.5). Replacing terminal context at fixed Full support thus affects more than Hunyuan’s boundary discontinuity.

Cross-seed donors. To test whether compatibility with the same seed alone explains the donor effect, we take each Hunyuan donor from another seed of the same prompt under an assignment fixed before generation, keeping the recipient seed and its noises (Appendix D.2). Cross-seed donors still significantly lower HC by 1.70 and BMR by 0.105, with TCR and DD comparable, but these reductions are significantly smaller than with same-seed donors (Fig. 3b). The improvement therefore does not depend entirely on same-seed pairing.

Interpretation. Constraint extent alone does not account for the observed temporal artifacts; terminal content also matters. Donor position/visibility, generated-image properties, and trajectory compatibility change jointly. The larger effect with same-seed donors is consistent with a contribution from compatibility with the recipient trajectory. A donor from another seed lacks this compatibility and can even conflict with the recipient’s camera and background motion, so the cross-seed effect is a conservative estimate of the role of terminal content. The separate contributions of these three factors remain unidentified. Appendix D gives secondary donor-relative and native-latent contrasts. We next examine which original endpoint constraints to retain without a donor rollout.

Table 2. Selected Wan2.2-Turbo controls: paired comparisons with main Partial (TCR 42.91%). A is original support area. All conditions and intervals: Tables C2–C5.
<table><tr><td></td><td>Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td>Variant</td><td>∆TCR↑</td><td>ΔHC↓</td><td>∆DD↑</td><td>△BMR↓</td></tr><tr><td>Global softening: Partial – Soft</td><td></td><td></td><td></td><td></td></tr><tr><td>Soft  $( \alpha = 0 . 7 5 )$ </td><td>+0.58</td><td>-1.49*</td><td>+7.38*</td><td>-0.224*</td></tr><tr><td>Spatial controls: Variant — Partial</td><td></td><td></td><td></td><td></td></tr><tr><td>0.5A</td><td>-3.50</td><td>-0.53*</td><td>+0.39</td><td> $- 0 . 0 1 7 ^ { * }$ </td></tr><tr><td>1.5A</td><td>-0.97</td><td>+0.00</td><td>+0.19</td><td> $+ 0 . 0 3 6 ^ { * }$ </td></tr><tr><td>Shifted  $M _ { 0 }$ </td><td>-6.60*</td><td>-0.44*</td><td>+0.58</td><td> $+ 0 . 0 4 1 ^ { * }$ </td></tr><tr><td>Random A #1</td><td> $- 5 . 2 4 ^ { * }$ </td><td>-0.04</td><td>-1.36</td><td> $+ 0 . 0 2 8 ^ { * }$ </td></tr></table>

5.3 Global Softening vs. Spatial Selection. Global softening tests whether weakening every terminal constraint can reproduce Partial’s reductions in temporal artifacts; spatial controls separately vary how many and which positions are constrained. All comparisons are paired with the main Partial, and all conditions retain the original edited endpoint.

At $\alpha = 0 . 7 5$ , TCR is comparable (+0.58 pp), but Partial yields significantly higher DD and lower HC and BMR (Table 2). The same directions hold at $\alpha = 0 . 5$ and 0.25 (Tables C2 and C3). The tested global softening therefore does not reproduce these reductions.

5.4 Support Area Analysis. For original support area $\begin{array} { r } { A = \sum _ { u } M _ { 0 } ( u ) } \end{array}$ , we test 0.5A, 0.75A, 1.25A, and 1.5A. Distance-based erosion retains interior cells, while expansion adds nearby exterior cells. This varies constraint extent around the same edit-relevant support, with the endpoint and projection fixed.

Reducing area to 0.5A or 0.75A lowers HC, and 0.5A also lowers BMR, while TCR remains comparable. Enlarging to 1.25A or 1.5A increases BMR, consistent with constraining more preserved context; 1.25A also lowers TCR (Tables 2, C4, and C5). No tested area is best on every metric.

5.5 Support Placement Analysis. To test whether edit-relevant placement matters beyond area, we relocate $M _ { 0 }$ away from its original position while preserving its cell count, and compare three random supports targeting area A. Construction details are in Appendix A.4.

At matched area, shifted support lowers TCR by 6.60 pp relative to edit-relevant placement, and all three random supports also lower it significantly (Tables 2 and C4). Shifted support lowers HC while increasing BMR, and DD remains comparable for shifted and random supports. Completion-favorable placement thus need not minimize temporal diagnostics.

Overlap with $M _ { 0 }$ drives completion. Random and shifted supports, placed without reference to $M _ { 0 } .$ , still cover part of it by chance (9% on average for random supports), and the completion they retain grows with this coverage (Appendix C.3). When we control the overlap directly, a support that misses $M _ { 0 }$ lowers TCR by 12.62 pp, to below Endpoint-free, while one covering 25–75% of $M _ { 0 }$ retains about half of the endpoint’s completion gain. Random supports of four shapes all fall below Partial (Fig. C1).

5.6 Gradient-Based Guidance. Partial imposes the endpoint by projection, so we test whether restricting the terminal constraint to edit-relevant support also helps under a different mechanism. Frame Guidance steers sampling with a frame loss backpropagated through the video decoder [14]. We run Frame Guidance’s official keyframe configuration on its primary backbone, CogVideoX-5B-I2V [43], with the same prompts, seeds, FLUX.2 endpoints, and $M _ { 0 } ;$ ; images and $M _ { 0 }$ are center-cropped to 3:2. We compare Endpoint-free, guidance on the whole last frame (FG-Full), and the same loss weighted by $M _ { 0 } \ ( \mathrm { F G - } M _ { 0 } ;$ Table 3 and Appendix C.5).

Full-frame guidance reproduces the trade-off of full-frame projection: it raises TCR by 9.71 pp but lowers DD by 14.17 pp. Restricting the loss to $M _ { 0 }$ keeps the completion gain (+0.39 pp relative to FG-Full) and raises DD by 12.23 pp, back to the Endpoint-free level, so the benefit of edit-relevant support does not depend on projection. FG-Full barely raises BMR on this backbone, leaving little reversal to remove, and $\mathrm { F G } \mathrm { - } M _ { 0 }$ raises HC by 1.50 through scene changes early in the video, not through jumps toward the endpoint (Appendix C.5). Frame Guidance takes about 57 minutes per video on one A6000, against 7 minutes for Endpoint-free; projection adds no backward pass.

Table 3. Edit-relevant support under gradient-based guidance. Frame Guidance on CogVideoX-5B-I2V, 103 prompts × five paired seeds. FG-Full guides the whole last frame; FG-M weights the same frame loss by $M _ { 0 } .$ <sup>∗</sup>Significant; intervals in Table C12.
<table><tr><td rowspan="2">Condition</td><td rowspan="2">Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td>TCR↑ HC↓</td><td>DD↑</td><td>BMR↓</td></tr><tr><td>Endpoint-free</td><td>27.18</td><td>5.44</td><td>44.7</td><td>0.388</td></tr><tr><td>FG-Full</td><td>36.89</td><td>5.40</td><td>30.5</td><td>0.434</td></tr><tr><td> $\mathrm { F G } \mathrm { - } M _ { 0 }$ </td><td>37.28</td><td>6.90</td><td>42.7</td><td>0.427</td></tr><tr><td>FG-Full – Endpoint-free</td><td> $+ 9 . 7 1 ^ { * }$ </td><td>-0.04</td><td> $- 1 4 . 1 7 ^ { * }$ </td><td>+0.045</td></tr><tr><td> $\mathrm { F G } { \cdot } M _ { 0 } \mathrm { - F G } { \cdot } \mathrm { F u l l }$ </td><td>+0.39</td><td> $+ 1 . 5 0 ^ { * }$  </td><td> $+ 1 2 . 2 3 ^ { \ast }$ </td><td>-0.006</td></tr></table>

5.7 Diagnostic Cross-Checks. Discontinuity. Against blinded Qwen3.6-35B-A3B judgments, HC has AUROC 0.777 (95% CI [0.70, 0.84]) on 1,619 evaluable videos from nine generation conditions, excluding Hunyuan.<sup>2</sup> The judge sees all native frames with method identity hidden and identifies abrupt changes lacking a continuous transition. On the Wan2.2 validation subset, judged event incidence is 0.0% for Endpoint-free, 8.3% for Full, and 5.0% for Partial, supporting the direction of the metric comparison. Protocol and sensitivity checks are reported in Appendix E.

Reversal and return. On Wan2.2, CoTracker3-based Track Reversal Ratio (TRR) and Back-Tracking Ratio (BTR) cross-check BMR [17]. TRR measures late motion opposing an early direction along tracked trajectories; BTR measures return from the largest achieved displacement to the last visible position. On the common 398/515 prompt–seed pairs meeting tracking and motion criteria under all three conditions, all three diagnostics follow Full > Partial > Endpointfree (Table E3). On these pairs, BMR’s pooled Spearman correlations with TRR and BTR are 0.64 and 0.65 (descriptive; paired observations are dependent). Main BMR uses all 515 samples per condition (Appendix E). Whole-frame and inside- $. M _ { 0 }$ BMR, and BMR without near-static clips, give the same Partial–Full and donor-context conclusions (Appendix E.4).

## 6 Discussion and Limitations

Benefits and limits of partial goals. In the evaluated settings, imposing only edit-relevant support reduces discontinuity and background reversal and alleviates motion suppression relative to Full at comparable TCR, which global softening does not reproduce. Partial also has specific limits: it does not raise TCR over Full, retains higher HC than Endpoint-free, raises BMR as support grows, and depends on edit-relevant placement (Sections 4.3–5.5).

Future validity. Edit relevance is a useful support heuristic, not a future-validity estimator. In Fig. 4, an endpointpreserved sign reappears under Partial after being absent in earlier frames, illustrating how retained position and visibility constraints can become inappropriate.

Generalization and evaluation scope. We primarily use FLUX.2 endpoints across three backbones, as FLUX.2 scored highest on TCR and TC-Score among tested editors (Table A1; Fig. A2). Donor-context tests cover Hunyuan and Wan2.2; cross-seed donors only Hunyuan, and spatial controls only Wan2.2. Partial remains untested with learned endpoint conditioners, but its support transfers to gradient-based Frame Guidance on a fourth backbone, where it keeps the completion gain and removes motion suppression (Section 5.6). Diagnostics measure specific temporal artifact rather than overall video quality.

## 7 Conclusion

Edited endpoints preserve unedited context in the source image’s layout, and full-frame conditioning imposes that context as a future requirement, a spatial over-specification. Measured with HC, BMR, and DD, the resulting temporal artifacts appear across five video models spanning training-free and learned endpoint conditioning, in forms that differ by model. Replacing only the preserved context under unchanged Full support reduces these artifacts on two backbones,

$$
+ M _ { 0 }
$$

(b) Partial: the sign is absent, then reappears at frame 114  
![](images/eb3707a9c4d8ae444009774814589fcbbb6dcceeed52636c472711ea848c706f.jpg)  
preserved sign

frame 108  
![](images/908f2753e6d8e5834922fa25d5807614f6ee7cf16d07ce5ae394ca50c9611eae.jpg)  
sign absent

frame 112  
![](images/70edb2a3df7076413fc0ff7da54c0d2f076ab38229f6fdbdf1075e6a0cfc21ac.jpg)  
sign absent

frame 114  
![](images/88a7d072b319f8a807ee5950dfff0e1dcf08b165244806320299d0be3b562f78.jpg)  
sign reappears

Figure 4. Estimated edit-relevant support does not guarantee a future-valid terminal constraint. Wan2.2-Turbo, purple-bus example of Fig. 1 (case 90001, outside TC-Bench), seed 22001. (a) Saved M outline (blue) and token-pooled weights (tint). (b) Partial: the boxed sign is absent at native frames 108 and 112, then reappears at 114. Identical regions are enlarged below.

and on Hunyuan, donors from other seeds retain part of this reduction, so constraint extent alone does not account for them. Imposing only edit-relevant support accordingly mitigates them at comparable TCR, which global softening does not reproduce, while support area and placement trade completion against temporal diagnostics. Residual failures expose the gap between edit relevance and future validity: beyond how strongly to condition on an edited endpoint, the design question is which parts of its content should constrain the future.

## Acknowledgements

This work was partly supported by the KHIDI grant funded by the Korean government (MOHW) [No.RS-2025- 02307233, No.RS-2026-25613012], the NRF or IITP grants funded by the Korean government (MSIT) [No.05-26-04- 0094, No.RS-2026-25472075, No.RS-2025-02305581, No.RS-2025-25442338, and No.RS-2021-II211343], the ITIP grant funded by the Korean government (MOTIR) [No.RS-2026-25549946], the Research grant from SNU, and the Strategic Hub grant for International Research Collaboration of SNU. Kyungsu Kim is affiliated with the School of Transdisciplinary Innovations, Department of Biomedical Science, Interdisciplinary Program in AI (IPAI) & Graduate School of AI, Medical Research Center, and AI Institute at SNU.

## References

[1] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended Latent Diffusion. ACM Transactions on Graphics, 42 (4):149:1–149:11, 2023. doi: 10.1145/3592450. URL https://doi.org/10.1145/3592450.

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, et al. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025. URL https://arxiv.org/abs/2502.13923.

[3] Minghong Cai, Qiulin Wang, Zongli Ye, Wenze Liu, Quande Liu, Weicai Ye, Xintao Wang, Pengfei Wan, Kun Gai, and Xiangyu Yue. VideoCanvas: Unified Video Completion from Arbitrary Spatiotemporal Patches via In-Context Conditioning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026. URL https://arxiv.org/abs/2510.08555.

[4] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22560–22570, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Cao\_MasaCtrl\_Tuning-Free\_Mutual\_Self-Attention\_Control\_for\_Consistent\_Image\_Synthesis\_and\_ ICCV\_2023\_paper.html.

[5] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, and Enze Xie. SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer. In International Conference on Learning Representations, pages 38973–38997, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 41b93c59da0d0f835907fd661d419db2-Paper-Conference.pdf.

[6] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction. In International Conference on Learning Representations, pages 52208–52222, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ e54e6eef11f87a874bf1e4551fc6d04e-Paper-Conference.pdf.

[7] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. DiffEdit: Diffusion-based semantic image editing with mask guidance. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=3lge0p5o-M-.

[8] Mohamed Elmoghany, Liangbing Zhao, Xiaoqian Shen, Subhojyoti Mukherjee, Yang Zhou, Gang Wu, Viet Dac Lai, Seunghyun Yoon, Ryan Rossi, Abdullah Rashwan, Puneet Mathur, Varun Manjunatha, Daksh Dangi, Chien Nguyen, Nedim Lipka, Trung Bui, Krishna Kumar Singh, Ruiyi Zhang, Xiaolei Huang, Jaemin Cho, Yu Wang, Namyong Park, Zhengzhong Tu, Hongjie Chen, Hoda Eldardiry, Nesreen Ahmed, Thien Nguyen, Dinesh Manocha, Mohamed Elhoseiny, and Franck Dernoncourt. InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions. arXiv preprint arXiv:2603.03646, 2026. URL https://arxiv.org/abs/2603.03646.

[9] Haiwen Feng, Zheng Ding, Zhihao Xia, Simon Niklaus, Victoria Abrevaya, Michael J. Black, and Xuaner Zhang. Explorative Inbetweening of Time and Space. In European Conference on Computer Vision, pages 378–395, 2024. doi: 10.1007/978-3-031-73229-4\_22. URL https://doi.org/10.1007/978-3-031-73229-4\_22.

[10] Weixi Feng, Jiachen Li, Michael Saxon, Tsu-Jui Fu, Wenhu Chen, and William Yang Wang. TC-Bench: Benchmarking Temporal Compositionality in Conditional Video Generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4638–4662, 2025. doi: 10.18653/v1/2025.findings-acl.241. URL https://aclanthology.org/2025.findings-acl.241/.

[11] Xingtong Ge, Yi Zhang, Yushi Huang, Dailan He, Xiahong Wang, Bingqi Ma, Guanglu Song, Yu Liu, and Jun Zhang. SALT: Self-Consistent Distribution Matching with Cache-Aware Training for Few-Step Video Generation. In European Conference on Computer Vision, 2026. URL https://eccv.ecva.net/virtual/2026/ poster/3473.

[12] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-Prompt Image Editing with Cross-Attention Control. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=\_CDixzkzeyb.

[13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818, 2024. URL https: //openaccess.thecvf.com/content/CVPR2024/html/Huang\_VBench\_Comprehensive\_ Benchmark\_Suite\_for\_Video\_Generative\_Models\_CVPR\_2024\_paper.html.

[14] Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Jaehong Yoon, Soo Ye Kim, Zhe Lin, and Sung Ju Hwang. Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models. In International Conference on Learning Representations, pages 60977–61003, 2026. URL https://proceedings. iclr.cc/paper\_files/paper/2026/file/636f1f78f7096caf126afaa27deb7b53-Paper-Conference.pdf.

[15] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-One Video Creation and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17191–17202, 2025. URL https://arxiv.org/abs/2503.07598.

[16] Zhangqi Jiang, Zheng Sun, Xianfang Zeng, Yufeng Yang, Xuanyang Zhang, Yongliang Wu, Wei Cheng, Gang Yu, Xu Yang, and Bihan Wen. GEditBench v2: A Human-Aligned Benchmark for General Image Editing. arXiv preprint arXiv:2603.28547, 2026. URL https://arxiv.org/abs/2603.28547.

[17] Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6013–6022, 2025. URL https:// openaccess.thecvf.com/content/ICCV2025/html/Karaev\_CoTracker3\_Simpler\_and\_ Better\_Point\_Tracking\_by\_Pseudo-Labelling\_Real\_Videos\_ICCV\_2025\_paper.html.

[18] Kinam Kim, Junha Hyung, and Jaegul Choo. Temporal In-Context Fine-Tuning with Temporal Reasoning for Versatile Control of Video Diffusion Models. In Advances in Neural Information Processing Systems, volume 38, pages 74491–74518, 2025. doi: 10.52202/085713-2501. URL https://proceedings. neurips.cc/paper\_files/paper/2025/file/6c03ebd7344bcc0c12c4ba8aa8487c00- Paper-Conference.pdf.

[19] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks. Transactions on Machine Learning Research, 2024. URL https: //openreview.net/forum?id=RFrJCkw2oa.

[20] Minghan Li, Chenxi Xie, Yichen Wu, Lei Zhang, and Mengyu Wang. FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16672–16681, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Li\_FiVE-Bench\_A\_Finegrained\_Video\_Editing\_Benchmark\_for\_Evaluating\_Emerging\_Diffusion\_ICCV\_ 2025\_paper.html.

[21] Ruibin Li, Tao Yang, Yangming Shi, Weiguo Feng, Shilei Wen, Bingyue Peng, and Lei Zhang. Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks. In International Conference on Learning Representations, pages 59233– 59269, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 60f9118a849e8e9a0c67e2a36ad80ebf-Paper-Conference.pdf.

[22] Yehonathan Litman, Shikun Liu, Dario Seyb, Nicholas Milef, Yang Zhou, Carl Marshall, Shubham Tulsiani, and Caleb Leak. EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8965–8975, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/ html/Litman\_EditCtrl\_Disentangled\_Local\_and\_Global\_Control\_for\_Real-Time\_Generative\_Video\_CVPR\_2026\_paper.html.

[23] Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, and Jiaya Jia. Generative Video Propagation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17712–17722, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Liu\_Generative\_ Video\_Propagation\_CVPR\_2025\_paper.html.

[24] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. Machine Intelligence Research, 22(4):730–751, 2025. doi: 10.1007/s11633-025-1562-4

[25] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. RePaint: Inpainting Using Denoising Diffusion Probabilistic Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11461–11471, 2022. URL https: //openaccess.thecvf.com/content/CVPR2022/html/Lugmayr\_RePaint\_Inpainting\_ Using\_Denoising\_Diffusion\_Probabilistic\_Models\_CVPR\_2022\_paper.html.

[26] Haoyu Ma, Shahin Mahdizadehaghdam, Bichen Wu, Zhipeng Fan, Yuchao Gu, Wenliang Zhao, Lior Shapira, and Xiaohui Xie. MaskINT: Video Editing via Interpolative Non-autoregressive

Masked Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7403–7412, 2024. URL https://openaccess.thecvf.com/ content/CVPR2024/html/Ma\_MaskINT\_Video\_Editing\_via\_Interpolative\_Nonautoregressive\_Masked\_Transformers\_CVPR\_2024\_paper.html.

[27] Qi Mao, Lan Chen, Yuchao Gu, Zhen Fang, and Mike Zheng Shou. MAG-Edit: Localized Image Editing in Complex Scenarios via Mask-Based Attention-Adjusted Guidance. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 6842–6850, 2024. doi: 10.1145/3664647.3680830. URL https://doi. org/10.1145/3664647.3680830.

[28] Matiur Rahman Minar, Seunghun Oh, GangHyeon Jeong, and Unsang Park. Steady-Forcing: Balancing Spatial Persistence and Motion Continuity in Long-Horizon Nature Video Diffusion. arXiv preprint arXiv:2606.14732, 2026. URL https://arxiv.org/abs/2606.14732.

[29] Junyi Ouyang, Wenbin Teng, Gonglin Chen, Yajie Zhao, and Haiwei Chen. DCARL: A Divide-and-Conquer Framework for Autoregressive Long-Trajectory Video Generation. In European Conference on Computer Vision, 2026. URL https://eccv.ecva.net/virtual/2026/poster/4282.

[30] Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2VEdit: First-Frame-Guided Video Editing via Image-to-Video Diffusion Models. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. URL https://doi.org/10.1145/3680528.3687656.

[31] Kaihang Pan, Qi Tian, Jianwei Zhang, Weijie Kong, Jiangfeng Xiong, Yanxin Long, et al. OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning. arXiv preprint arXiv:2603.24458, 2026. URL https://arxiv.org/abs/2603.24458.

[32] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 724– 732, 2016. URL https://openaccess.thecvf.com/content\_cvpr\_2016/html/Perazzi\_A\_ Benchmark\_Dataset\_CVPR\_2016\_paper.html.

[33] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In International Conference on Learning Representations, pages 1862–1874, 2024. URL https://proceedings. iclr.cc/paper\_files/paper/2024/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf.

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pages 8748–8763, 2021. URL https://proceedings.mlr.press/ v139/radford21a.html.

[35] Zachary Teed and Jia Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In European Conference on Computer Vision, 2020. URL https://www.ecva.net/papers/eccv\_2020/papers\_ ECCV/html/3526\_ECCV\_2020\_paper.php.

[36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, et al. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/ abs/2503.20314.

[37] Xiaojuan Wang, Boyang Zhou, Brian Curless, Ira Kemelmacher-Shlizerman, Aleksander Holynski, and Steve Seitz. Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation. In International Conference on Learning Representations, pages 30594–30609, 2025. URL https://proceedings. iclr.cc/paper\_files/paper/2025/file/4bbdef62653d8088717640e7660a1ebb-Paper-Conference.pdf.

[38] Yihan Wang and Jia Deng. WAFT: Warping-Alone Field Transforms for Optical Flow. In International Conference on Learning Representations, pages 157389–157402, 2026. URL https://proceedings. iclr.cc/paper\_files/paper/2026/file/ff2853ed92a6e6983ecccd9c922a42d2-Paper-Conference.pdf.

[39] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, et al. HunyuanVideo 1.5 Technical Report. arXiv preprint arXiv:2511.18870, 2025. URL https://arxiv.org/abs/2511.18870.

[40] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, et al. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324, 2025. URL https://arxiv.org/abs/2508.02324.

[41] Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, and Huan Ling. ChronoEdit: Towards Temporal Reasoning for In-Context Image Editing and World Simulation. In International Conference on Learning Representations, pages 9886–9899, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/file/10b99813d5992673da90a499b6bcd5f0-Paper-Conference.pdf.

[42] Serin Yang, Taesung Kwon, and Jong Chul Ye. ViBiDSampler: Enhancing Video Interpolation Using Bidirectional Diffusion Sampler. In International Conference on Learning Representations, pages 3441–3459, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 098a3d992f0d2128bab619b3c654e1c3-Paper-Conference.pdf.

[43] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In International Conference on Learning Representations, 2025.

[44] Zuhao Yang, Jiahui Zhang, Yingchen Yu, Shijian Lu, and Song Bai. Versatile Transition Generation with Image-to-Video Diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16981–16990, 2025. URL https://openaccess.thecvf.com/ content/ICCV2025/html/Yang\_Versatile\_Transition\_Generation\_with\_Image-to-Video\_Diffusion\_ICCV\_2025\_paper.html.

[45] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, 2018. URL https://openaccess.thecvf.com/content\_ cvpr\_2018/html/Zhang\_The\_Unreasonable\_Effectiveness\_CVPR\_2018\_paper.html.

[46] Min Zhao, Hongzhou Zhu, Chendong Xiang, Kaiwen Zheng, Chongxuan Li, and Jun Zhu. Identifying and Solving Conditional Image Leakage in Image-to-Video Diffusion Model. In Advances in Neural Information Processing Systems, volume 37, pages 30300–30326, 2024. doi: 10.52202/ 079017-0954. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 35cb54b887e7aafe74829677cce6c5c6-Paper-Conference.pdf.

[47] Siyu Zou, Jiji Tang, Yiyi Zhou, Jing He, Chaoyi Zhao, Rongsheng Zhang, Zhipeng Hu, and Xiaoshuai Sun. Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 7864–7872, 2024. doi: 10.1609/aaai.v38i7.28622. URL https://ojs.aaai.org/index.php/AAAI/article/view/28622.

## Appendix

## Contents

A Implementation Details   
A.1 Endpoint and Support Construction 16   
A.2 Choice of Endpoint Editor 16   
A.3 Video Configurations and Projection 18   
A.4 Support-Size and Placement Controls 19   
B Metric Definitions and Evaluation Populations. 19   
B.1 Transition Completion and TC-Score 20   
B.2 Dynamic Degree and Hard-Cu 20   
B.3 Background Motion Reversal 20   
B.4 Evaluation Populations 20   
C Paired Uncertainty and Additional Results. 21   
C.1 Resampling Protocol 21   
C.2 Global-Softening and Spatial-Control Results. 21   
C.3 Placement Beyond Chance Covera 23   
C.4 Additional Endpoint Comparison 25   
C.5 Frame Guidance with Edit-Relevant Support 26   
D Donor-Context Intervention and Score Decomposition 27   
D.1 Construction, Pairing, and Scope 27   
D.2 Cross-Seed Donors on Hunyuan 28   
D.3 Secondary Donor and Representation Contrasts . 28   
D.4 Descriptive Closure of the Full–Partial Gap 28   
D.5 TC-Score Decomposition and Its Relation to HC 28   
E Diagnostics and Cross-Checks 30   
E.1 Hunyuan Discontinuity Localization 30   
E.2 Blinded Model-Judge Agreement for HC 30   
E.3 Track Reversal Ratio and Back-Tracking Ratio 30   
E.4 BMR Measurement Region and Static Backgrounds. 31   
E.5 Source-Time Layout of Preserved Context 32   
F Residual Failures and Exploratory Support Release 33   
G Broader I2V Completion Comparisons 34

## A Implementation Details

## A.1 Endpoint and Support Construction

Endpoint generation. We generate one endpoint per prompt with FLUX.2-klein-base-9B, seed 42, 50 editing steps, bfloat16 inference, and default guidance. We center-crop the source to the target aspect ratio and resize it with Lanczos to $1 2 8 0 \times 7 0 4$ . We use the following final-frame prompt:

Use the input image as the starting frame of a video. The following sentence describes a transition that happens over the course of the video: “{transition}”

Generate a plausible final frame of that video: the moment right after the transition described above has fully completed.

The frame should look like a later moment from the same continuous shot.

Preserve scene continuity, identity, camera framing, and all persistent background/content, changing only what the transition describes.

Attention aggregation. Each prompt has one object phrase and one edit phrase, fixed across conditions; they were annotated with the assistance of Claude Opus 5 and manually verified. In each joint-attention layer, the softmax runs over all text, endpoint-image, and reference-image tokens. For a phrase token, we keep the post-softmax entries between that token and the endpoint-image tokens, discard the reference-image entries, and average the text-to-image and image-to-text directions. These maps are averaged over heads, the 32 transformer blocks, and the 50 editing steps of the conditional pass, and then over each phrase’s tokens, giving $a _ { o }$ and $a _ { e }$ . We combine the normalized maps as

$$
a ( u ) = \frac { 1 } { 2 } \left[ \mathcal { N } ( a _ { o } ) ( u ) + \mathcal { N } ( a _ { e } ) ( u ) \right] , \qquad \mathcal { N } ( a ) ( u ) = \frac { a ( u ) - \operatorname* { m i n } _ { v } a ( v ) } { \operatorname* { m a x } ( \operatorname* { m a x } _ { v } a ( v ) - \operatorname* { m i n } _ { v } a ( v ) , 1 0 ^ { - 1 2 } ) } .\tag{A1}
$$

We estimate support on the $4 4 \times 8 0$ latent grid (height × width) using a two-component Gaussian mixture with shared variance:

$$
p ( a ( u ) ) = \sum _ { j \in \{ 0 , 1 \} } \pi _ { j } \mathcal { N } ( a ( u ) ; \mu _ { j } , \sigma ^ { 2 } ) , \qquad \mu _ { 0 } < \mu _ { 1 } .\tag{A2}
$$

We initialize the means at 25% and 75% of the observed value range, with equal mixture weights and empirical variance. Variance floors are $1 0 ^ { - 6 }$ at initialization and $1 0 ^ { - 8 }$ during updates. Expectation–maximization runs for up to 300 iterations, stopping when both means satisfy $| \mu _ { j } ^ { \mathrm { n e w } } - \mu _ { j } | \overset { \cdot } { \leq } 1 \bar { 0 } ^ { - 1 2 } + 1 0 ^ { - 5 } | \mu _ { j } |$ . The high-mean component represents edit-relevant evidence. We encourage spatial coherence with

$$
\mathcal { E } ( M ) = \sum _ { u } \left[ \frac { ( a ( u ) - \mu _ { M ( u ) } ) ^ { 2 } } { 2 \sigma ^ { 2 } } - \log \pi _ { M ( u ) } \right] + \beta \sum _ { ( u , v ) \in \mathcal { E } _ { 4 } } \mathbf { 1 } [ M ( u ) \neq M ( v ) ] .\tag{A3}
$$

Here ${ \mathcal { E } } _ { 4 }$ is the four-neighbor grid and $\beta = 2 \colon$ in each update, each of a cell’s four neighbors with a different label adds $\beta$ to that label’s cost. We initialize labels by unary cost and apply up to 12 synchronous conditional-mode updates, replicating border labels as padding, and stop when labels no longer change. Figure A1 shows the resulting supports.

## A.2 Choice of Endpoint Editor

Partial needs only an edited endpoint and a support estimated from the editor’s text–image attention (Appendix A), so the primary editor must be open-weight with accessible attention. We use FLUX.2-klein-base-9B for its editing quality, completion, and layout preservation (Table A1).

Editing quality. On GEditBench v2, FLUX.2 [klein] 9B ranks first among open-weight editors with an Elo of 1,039, on par with Qwen-Image-Edit-2511 at 1,038 [16]. That evaluation uses the 4-step distilled checkpoint; we use its undistilled base, whose 50-step sampler provides the attention we aggregate.

Completion and temporal diagnostics. Under Full conditioning on Wan2.2, FLUX.2 endpoints give the highest TCR of the three editors with the fewest parameters. Paired TCR differences are comparable, +1.75 [−3.69, +7.38] pp versus

Image-editor attention → estimated support $M _ { 0 }$  
![](images/b36b005307718edf17de6b9321ae35bb16c98b0c3e1fc45d9231f36576e0d61c.jpg)  
Figure A1. Estimated edit-relevant support. Source images, edited endpoints, and GMM–Ising masks for TC-Bench prompts 0, 5, and 82. Blue marks support; percentages give constrained area before token pooling.

ChronoEdit [41] and +6.80 [−0.39, +13.98] pp versus Qwen-Image-Edit [40], but only FLUX.2 endpoints significantly raise TCR over Endpoint-free (Table C8). TC-Score is significantly higher than with Qwen-Image-Edit endpoints, +0.035 [+0.018, +0.052], and comparable with ChronoEdit endpoints, +0.014 [−0.001, +0.029]. Full’s temporal artifacts are not specific to FLUX.2: relative to Endpoint-free, all three editors significantly raise HC. Qwen-Image-Edit endpoints also change BMR by +0.256 [+0.207, +0.307] and DD by −15.34 [−20.19, −10.68] pp, and ChronoEdit endpoints by +0.198 [+0.148, +0.249] and −11.46 [−16.89, −6.02] pp. HC, DD, and BMR are comparable between FLUX.2 and Qwen-Image-Edit endpoints.

Layout preservation. FLUX.2 endpoints differ from the source least outside $M _ { 0 }$ and most inside it. The outside difference is lower than for Qwen-Image-Edit, −0.022 [−0.035, −0.008], and ChronoEdit, −0.038 [−0.056, −0.023]. Because $M _ { 0 }$ is estimated from FLUX.2 attention, we also measure Qwen-Image-Edit endpoints outside the support estimated from their own attention. They still differ more from the source than FLUX.2 endpoints do (0.080 versus 0.056; FLUX.2 minus Qwen-Image-Edit, −0.024 [−0.038, −0.010]). Table A1 reports 0.077 for Qwen-Image-Edit because it measures outside the FLUX.2 support. FLUX.2 endpoints thus most closely follow Eq. (1), making Full’s temporal artifacts less likely to stem from unrequested changes outside the edit.

Qualitative examples. In Figure A2, FLUX.2 and Qwen-Image-Edit recolor the chameleon while keeping its pose and texture, whereas ChronoEdit also changes its appearance. For the sunset, FLUX.2 and ChronoEdit lower the sun while keeping the skyline, whereas Qwen-Image-Edit returns a four-panel collage. For the meteor, FLUX.2 extends the streak toward the lower right, Qwen-Image-Edit leaves the source nearly unchanged, and ChronoEdit renders a spray of streaks near the upper left.

Table A1. Endpoint editors under Full conditioning on Wan2.2-Turbo (103 prompts × five paired seeds). Out and In give the mean absolute RGB difference between the endpoint and the source image outside and inside the FLUX.2 support $M _ { 0 } .$ . ChronoEdit uses the original transition prompt and the last frame of a five-frame generation (Appendix G).
<table><tr><td colspan="3"></td><td colspan="3">Full on Wan2.2-Turbo</td><td colspan="3">Endpoint vs. source</td></tr><tr><td>Editor</td><td>Params</td><td>TCR↑</td><td>TC-Score ↑</td><td>HC↓</td><td>DD↑</td><td>BMR↓</td><td>Out</td><td>In</td></tr><tr><td>FLUX.2-klein-base-9B</td><td>9B</td><td>41.94</td><td>0.8352</td><td>6.47</td><td>66.8</td><td>0.599</td><td>0.056</td><td>0.232</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>20B</td><td>35.15</td><td>0.8001</td><td>7.11</td><td>64.1</td><td>0.617</td><td>0.077</td><td>0.161</td></tr><tr><td>ChronoEdit-14B</td><td>14B</td><td>40.19</td><td>0.8214</td><td>6.13</td><td>68.0</td><td>0.558</td><td>0.094</td><td>0.192</td></tr><tr><td>Endpoint-free</td><td>一</td><td>33.79</td><td>0.7841</td><td>4.57</td><td>79.4</td><td>0.360</td><td></td><td></td></tr></table>

Edited endpoints from three image editors  
source image  
FLUX.2-klein-base-9B  
Qwen-Image-Edit-2511  
ChronoEdit-14B

Prompt 0: pink chameleon turns green  
![](images/fa76c83a5be45aa9fe42ad78e6881dc7fad965fd1c30202888c23cd234a2f9e9.jpg)

![](images/9e31d0eac8bfbdba8f0ebd760f5689398abbf4e3ab07e7cc08d9516d56971813.jpg)

![](images/3571af48f6b41f0cd2dc3f89df55b224c3f768de61cf980f81e198bcb2a7cd86.jpg)

![](images/44d47e9fdb8420bf6afa70aaf153096b4be7ef8d59565bee976ee9d4d2f245ae.jpg)

Prompt 103: sunny afternoon turns to sunset  
![](images/6718ee6a52b4c47c6a968898e4095b8efd67c37f1656fe6bfc0596b510327627.jpg)

![](images/cede08b7f519a6f07e25bc7974fef05c2b39ab986c58548c3fe094289525f36e.jpg)

![](images/e6562d9321d2506a035c482c4bd161a8aa44f11afc63e57b8fd30ddcbbdaa382.jpg)

![](images/21706c2cfebc2de9a2fa12d2e3e6e9bd23733880476ee45d1f5447ae19068b12.jpg)

Prompt 105: meteor streaks from upper left to lower right  
![](images/d008e7ef9d939732937f7f92acc4220839641da985547f657f11c03ea76d7cd9.jpg)

![](images/ad58406252f772a33ff3856960b83b111a2eb071fe9bcb576f32899a788949ff.jpg)

![](images/ee8a8a9901c1eb4f0de84502c15b272df37fcbdde9eee53891b077973049c8a5.jpg)

![](images/fc4108abe008b03ba127826f205c7458cc3aad1d2d7de6dd0e1a2195154774ea.jpg)  
Figure A2. Endpoints from three editors. Source images and edited endpoints for TC-Bench prompts 0, 103, and 105. FLUX.2 and Qwen-Image-Edit use the final-frame prompt in Appendix A; ChronoEdit uses the original transition prompt to match its formulation of editing as video generation.

Other editors. Support estimation uses only the endpoint and the attention to the annotated object and edit phrases. Open-weight diffusion and flow editors expose this attention in joint or cross-attention layers, so we expect the procedure to extend to them without changing the video model. Adapting our attention recorder to Qwen-Image-Edit’s token layout, with the same GMM–Ising rule, yields supports for all 103 prompts, covering 6.3% of the frame on average versus 8.9% for FLUX.2. We did not evaluate Partial with these supports; editors without attention access, such as closed APIs, would need supports from another source.

## A.3 Video Configurations and Projection

We use seeds 22001–22005 and share each edited endpoint between Full and Partial. Wan2.2 uses the distilled TI2V-5B checkpoint; SANA uses the 2B 720p checkpoint with guidance scale 6.0.

Wan2.2 average-pools $M _ { 0 }$ in $2 \times 2$ blocks from the 44 $\times 8 0$ latent grid to a $2 2 \times 4 0$ token grid and repeats the weights over the corresponding latent cells. SANA area-pools to its $2 2 \times 4 0$ latent/token grid. Pooled weights can be fractional; they are broadcast over latent channels, with token timesteps scaled by $1 - w .$ . At each step, the terminal state x enters

Table A2. Primary generation configurations. Spatial dimensions are width by height. HunyuanVideo-1.5 selects an aspect-ratiocompatible resolution using its 480p configuration.
<table><tr><td>Backbone</td><td>Steps</td><td>Frames</td><td>FPS</td><td>Resolution</td><td>Terminal condition</td></tr><tr><td>Wan2.2-Turbo</td><td>4</td><td>121</td><td>24</td><td> $1 2 8 0 \times 7 0 4$ </td><td>Input/clean projection</td></tr><tr><td>SANA-Video 2B</td><td>50</td><td>81</td><td>16</td><td> $1 2 8 0 \times 7 0 4$ </td><td>See wrappers below</td></tr><tr><td>HunyuanVideo-1.5</td><td>8</td><td>121</td><td>24</td><td>480p</td><td>Noise-matched projection</td></tr></table>

the Wan2.2 model as $x _ { \mathrm { i n } } = w \odot z _ { E } + ( 1 - w ) \odot x ,$ , and the clean prediction is projected with Eq. (2) before the sampling update; both blends use the same $w ,$ including $w = \alpha$ for Soft.

SANA Partial keeps the solver state x, which starts from the initial noise like every other latent slot, separate from the blended model input $y .$ For the terminal slot at noise level σ and timestep t,

$$
y = w \odot z _ { E } + ( 1 - w ) \odot x , \qquad \hat { z } = w \odot z _ { E } + ( 1 - w ) \odot ( y - \sigma v ) , \qquad \tilde { v } = ( x - \hat { z } ) / \sigma ,\tag{A4}
$$

where $v$ is the model’s flow for input $y$ at token timesteps $t ( 1 - w )$ . The corrected flow v˜ replaces v wherever $w > 0 ,$ and the native DPM-Solver++ update advances x [24]; the first latent slot stays clean and outside the solver. SANA Full instead clamps the terminal state at initialization and after each scheduler step, with zero terminal token timesteps. With $M _ { 0 } \equiv 1$ , the Partial update reduces to this clamp: from identical initial noise, the two implementations’ final latents differ by at most $1 . 2 \overset { \cdot } { \times } 1 0 ^ { - 7 }$ in two verification videos.

For Hunyuan, we resize and center-crop the source and mask to the selected 480p aspect ratio, then pool and threshold the mask at 0.5. We repeat the endpoint over five frames, encode it with the causal video ${ \mathrm { V A E } } ,$ and select the second temporal latent slot, which encodes four copies of the endpoint just as the terminal slot encodes four frames. Endpoint noise $\epsilon _ { E }$ is sampled once from a seed derived from the video seed and held fixed. At noise level $\sigma _ { k }$ , masked projection targets

$$
z _ { E } ^ { ( k ) } = ( 1 - \sigma _ { k } ) z _ { E } + \sigma _ { k } \epsilon _ { E } ,\tag{A5}
$$

at initialization and after each scheduler update, ending with clean projection at zero noise.

Spatial controls are paired with the main Partial. Additional completion comparisons use learned endpoint conditioning in MfM and OmniWeaving, and 50 sampling steps for Wan2.2 Base.

## A.4 Support-Size and Placement Controls

For $\begin{array} { r } { A = \sum _ { u } M _ { 0 } ( u ) } \end{array}$ , size controls target rounded areas 0.5A, 0.75A, 1.25A, and 1.5A. Euclidean distance transforms rank cells: erosion retains the deepest interior cells, while expansion adds the nearest exterior cells. Seeded random values break ties.

Shifted support preserves area while changing location. On the $4 4 \times 8 0$ latent grid, we consider translations that keep the mask inside the frame and use even row and column offsets, which keep $2 \times 2$ token blocks aligned; we sample uniformly among those with IoU below 0.1, or take the lowest-overlap candidate. When no such translation exists, we shift the mask by half the frame width, with a vertical shift of zero or half the frame height, clip it at the border, and regrow it to its original cell count with the distance-based expansion of the size controls, keeping the candidate with the lowest IoU; this occurs for 12 of 103 prompts.

Random support selects $\operatorname { r o u n d } ( A / 4 )$ cells on the $2 2 \times 4 0$ token grid and expands each to a $2 \times 2$ latent block. Area therefore matches $M _ { 0 }$ to within two latent cells. We evaluate three independent random supports per prompt. These supports remain fixed during sampling; random release in Appendix F instead shrinks support over denoising steps.

## B Metric Definitions and Evaluation Populations

Temporal sampling and preprocessing. For N native frames at FPS f, TC-Score uses unique indices $\mathrm { r o u n d } ( k ( N -$ $1 ) / 1 5 ) , k \ = \ 0 , \ldots , 1 5$ . HC samples round $. ( k f / 8 )$ for $k = 0 , \ldots , \lfloor 8 ( N - 1 ) / f \rfloor$ ; DD and BMR take every max(1, round(f/8))-th frame. Every evaluated video has $f \in \{ 8 , 1 6 , 2 4 \}$ , so all three temporal diagnostics select the same frames at 8 FPS. CLIP uses its standard 224-pixel image preprocessing; flow uses decoded resolution. Track diagnostics use native 24 FPS frames resized to $6 4 0 \times 3 5 2$

## B.1 Transition Completion and TC-Score

We evaluate 50 attribute-binding, 38 object-binding, and 15 background-shift prompts using the benchmark assertions and Qwen2.5-VL-32B-Instruct. TCR requires Yes for every assertion except those about “other objects.”

TC-Score combines a VLM assertion average, excluding transition-object consistency, with mean adjacent-frame CLIP ViT-B/32 similarity on the 16-frame grid. We map CLIP similarity from [0.9, 0.98] to [0, 1] and clip it. The VLM and CLIP terms receive weights 2/3 and 1/3, respectively. Adjacent-frame CLIP similarity falls with larger motion and discontinuities, so the CLIP term mixes completion with properties that DD and HC measure, and HC uses the same similarity (Appendix D.5). We therefore treat TCR as the primary completion measure and report TC-Score for comparison with TC-Bench.

## B.2 Dynamic Degree and Hard-Cut

DD follows VBench. For each adjacent pair, m(t) is the mean of the largest 5% of RAFT flow magnitudes. A video is dynamic if $m ( t ) > 6 \operatorname* { m i n } ( H , W ) / 2 5 6$ for at least round(4N/16) pairs, where N is the sampled frame count. The main clips have 41 frames at 8 FPS and require 10 qualifying pairs. DD measures substantial movement, including camera motion.

HC is the difference between the median and minimum adjacent-frame CLIP cosine similarities (Eq. (3)), an absolute rather than proportional drop. We report 100×HC, including its confidence intervals. The minimum-similarity pair t<sub>HC</sub> identifies the event location.

## B.3 Background Motion Reversal

BMR uses WAFT flow at 8 FPS outside the FLUX.2-derived support. We keep this measurement mask fixed across all conditions, resize it with nearest-neighbor interpolation, and take a deterministic strided subset of at most 80,000 pixels. Equation (4) weights flow by magnitude to prevent static pixels from dominating the direction estimate.

The reference axis is the normalized componentwise median flow over the first half of the video; reversal is measured over the second half. The ratio is undefined for a zero reference or zero late along-axis travel. All main and control conditions have 515 finite values, without DD filtering. We cross-check BMR with track diagnostics in Appendix E.3 and repeat the main comparisons over other measurement regions and without near-static clips in Appendix E.4.

## B.4 Evaluation Populations

Table B1 summarizes the cohorts used in each analysis.

Table B1. Appendix populations and references. Counts refer to paired prompt–seed positions unless stated otherwise.
<table><tr><td>Analysis</td><td>Population and reference</td></tr><tr><td>Main backbone comparisons</td><td>103 prompts  $\times 5$  seeds per condition; Table 1.</td></tr><tr><td>Wan2.2-Turbo spatial controls</td><td>103 prompts × 5 seeds per support, paired with the main Wan2.2-Turbo Partial (TCR 42.91%).</td></tr><tr><td>HunyuanVideo-1.5 donor-context intervention</td><td>Original Full reuses the main cohort. Donor P regenerates main Partial with identical settings; TCR 44.66% versus 44.47% reflects run-to-run variation.</td></tr><tr><td>HunyuanVideo-1.5 cross-seed donors</td><td>O, same-seed D, and P are reused; 515 new  $D _ { \mathrm { c r o s s } }$  videos use a donor-seed assignment fixed before generation.</td></tr><tr><td>Wan2.2-Turbo donor-context intervention</td><td>O, D, N, and P are new 515-video runs sharing one noise stream per prompt–seed. P reproduces main Partial&#x27;s HC, DD, and BMR exactly; TCR 42.72% versus 42.91% reflects re-judging variation. O regenerates Full with this noise stream and stays within sampling variation of main Full on all four metrics (TCR 43.30% versus 41.94%, HC 6.49 versus 6.47, DD 65.44% versus 66.80%, BMR 0.600 versus 0.599; no paired difference is significant).</td></tr><tr><td>HC model-judge agreement</td><td>90 prompts × 2 samples  $\times 9$  conditions; 1,619 evaluable videos. HunyuanVideo-1.5 is excluded.</td></tr><tr><td>Track cross-check</td><td>Common 398 pairs measurable in all three Wan2.2-Turbo conditions; main BMR uses all 515.</td></tr><tr><td>controls</td><td>Wan2.2-Turbo overlap and shape 103 prompts × 5 seeds per support, paired with the main Wan2.2-Turbo Partial;</td></tr><tr><td>Frame Guidance (CogVideoX-5B-I2V)</td><td>generated with the control runner, which reproduces main Partial videos bit for bit. 103 prompts × 5 seeds per condition; Endpoint-free, FG-Full, and FG-M0 are compared only with one another.</td></tr></table>

## C Paired Uncertainty and Additional Results

## C.1 Resampling Protocol

For a metric $y ,$ let $y _ { i s c }$ denote its value for prompt i, seed s, and condition c. The paired difference between target a and reference b is

$$
\widehat { \Delta } _ { a - b } = { \frac { 1 } { 1 0 3 } } \sum _ { i = 1 } ^ { 1 0 3 } d _ { i } , \qquad d _ { i } = { \frac { 1 } { 5 } } \sum _ { s = 1 } ^ { 5 } ( y _ { i s a } - y _ { i s b } ) .\tag{C1}
$$

We resample 103 prompts with replacement, retaining each prompt’s five seeds and paired conditions. We use 100,000 draws with NumPy PCG64 seed 20260919 and take the 2.5th/97.5th percentiles. TCR and temporal metrics share the same draws, which also give subgroup estimates: with $c _ { i } ^ { ( b ) }$ the number of times prompt i appears in draw $b ,$ the estimate for a prompt subgroup g, such as an edit type, is $\textstyle \sum _ { i \in g } { \dot { c } } _ { i } ^ { ( b ) } d _ { i } / \sum _ { i \in g } c _ { i } ^ { ( b ) }$ . TCR/DD differences are in pp, HC uses its displayed scale, and BMR its original scale.

Intervals are pointwise, unadjusted, and conditional on these five seeds. A CI spanning zero does not establish equivalence or non-inferiority. Because TCR differences take discrete values, a percentile bound can fall exactly at zero (e.g., [0.00, +11.46] in Table C1); such an interval counts as including zero. We compute differences and ratios from unrounded per-video values.

## C.2 Global-Softening and Spatial-Control Results

Tables C1, C3, and C5 report direct paired differences for the main, softening, and spatial comparisons.

Table C1. Paired mean differences. Each bootstrap draw resamples prompts and carries all five seeds and conditions together. The direction is stated in every row.
<table><tr><td rowspan="2">Difference</td><td rowspan="2">Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td>∆HC↓</td><td>∆DD↑</td><td>∆BMR↓</td></tr><tr><td>Wan2.2-Turbo Full-Endpoint-free</td><td>∆TCR↑ +8.16 [+1.75, +14.56]</td><td>+1.90 [+1.27,+2.59]</td><td>-12.62</td><td>+0.239 [−17.48, -8.16] [+0.189, +0.289]</td></tr><tr><td>Wan2.2-Turbo Partial–Full</td><td>+0.97 [−3.50, +5.44]</td><td>-0.72 [-1.29,-0.17]</td><td>+9.90</td><td>-0.213 [+5.63,+14.56] [-0.262,-0.165]</td></tr><tr><td>Wan2.2-Turbo Partial-Endpoint-free</td><td>+9.13 [+3.11, +15.34]</td><td>+1.18 [+0.72, +1.69]</td><td>-2.72 [−5.24, -0.39]</td><td>+0.027 [+0.004, +0.049]</td></tr><tr><td>SANA-Video Full-Endpoint-free</td><td>+12.23 [+5.44, +19.22]</td><td>+1.70 [+0.67,+2.70]</td><td>-9.90</td><td>+0.298</td></tr><tr><td>SANA-Video Partial–Full</td><td>-0.39</td><td>-0.97</td><td>+9.51</td><td>[−15.53, -4.27] [+0.254, +0.343] -0.285</td></tr><tr><td>SANA-Video Partial-Endpoint-free</td><td>[−6.41,+5.44] +11.84</td><td>[-1.93,+0.01] +0.73</td><td>-0.39</td><td>[+4.08,+15.15] [-0.325,-0.246] +0.013</td></tr><tr><td>HunyuanVideo-1.5</td><td>[+6.99, +16.89] +5.63</td><td>[+0.17, +1.33] +9.50</td><td>[−4.85, +4.08] +1.17</td><td>[-0.012, +0.038] +0.279</td></tr><tr><td>Full-Endpoint-free HunyuanVideo-1.5</td><td>[0.00, +11.46] -2.14</td><td>[+8.53, +10.51] -7.27</td><td>[-1.55, +3.88] -1.17</td><td>[+0.233, +0.326] -0.252</td></tr><tr><td>Partial–Full HunyuanVideo-1.5 Partial-Endpoint-free</td><td>[−7.57,+3.11] +3.50</td><td>[-8.16,-6.40] +2.23</td><td>[-3.50,+1.17] +0.00</td><td>[-0.296,-0.208] +0.028</td></tr></table>

Table C2. Global softening versus Partial on Wan2.2-Turbo. Partial has higher DD and lower HC/BMR than each tested soft setting; all direct paired temporal intervals exclude zero (Table C3).
<table><tr><td rowspan="2">Condition</td><td colspan="2">Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td>TCR↑</td><td>TC-Score ↑</td><td>HC↓</td><td>DD↑</td><td>BMR↓</td></tr><tr><td>Full (α = 1)</td><td>41.94</td><td>0.8352</td><td>6.47</td><td>66.8</td><td>0.599</td></tr><tr><td>Soft (α = 0.75)</td><td>42.33</td><td>0.8280</td><td>7.25</td><td>69.3</td><td>0.611</td></tr><tr><td>Soft (α = 0.5)</td><td>39.03</td><td>0.8155</td><td>7.62</td><td>69.9</td><td>0.630</td></tr><tr><td>Soft (α = 0.25)</td><td>39.22</td><td>0.7997</td><td>8.48</td><td>72.2</td><td>0.614</td></tr><tr><td>Partial (M0)</td><td>42.91</td><td>0.8187</td><td>5.76</td><td>76.7</td><td>0.387</td></tr></table>

Table C3. Direct Partial minus Soft paired differences.
<table><tr><td rowspan="3">Soft α</td><td colspan="2">Transition completion</td><td colspan="6">Temporal diagnostics</td></tr><tr><td colspan="2">∆TCR↑</td><td colspan="2">∆HC↓</td><td colspan="2">∆DD↑</td><td colspan="2">∆BMR↓</td></tr><tr><td>0.75</td><td></td><td>+0.58 [−4.66, +5.63]</td><td></td><td></td><td></td><td>-1.49 [−2.23, −0.84] +7.38 [+3.88, +11.07]</td><td></td><td>-0.224 [−0.272, -0.177]</td></tr><tr><td>0.5</td><td></td><td></td><td></td><td></td><td></td><td>+3.88 [−1.75, +9.51] −1.86 [−2.67, −1.16] +6.80 [+2.91, +10.87]</td><td></td><td>-0.243 [−0.292, -0.195]</td></tr><tr><td>0.25</td><td></td><td>+3.69 [−1.75, +9.13] -2.72 [−3.60, -1.95]</td><td></td><td></td><td></td><td>] +4.47 [+0.78, +8.35]</td><td></td><td>−0.227 [−0.273, -0.182]</td></tr></table>

Table C4. Wan2.2-Turbo support-size and placement controls. Differences are paired with the main Partial.
<table><tr><td></td><td colspan="2">Transition completion</td><td colspan="2">Temporal diagnostics</td></tr><tr><td>Support</td><td>TCR ↑</td><td>∆TCR↑ [95% CI]</td><td>HC↓ DD↑</td><td>BMR↓</td></tr><tr><td>Partial  $( M _ { 0 } )$ </td><td>42.91 一</td><td></td><td>5.76</td><td>76.7 0.387</td></tr><tr><td>0.5A</td><td>39.42 -3.50</td><td>[-8.16, +0.97]</td><td>5.23</td><td>77.1 0.370</td></tr><tr><td>0.75A</td><td>42.72 -0.19</td><td>[−4.08, +3.69]</td><td>5.42</td><td>77.1 0.374</td></tr><tr><td>1.25A</td><td>38.45 -4.47</td><td>[-8.54, -0.39]</td><td>5.86</td><td>76.3 0.405</td></tr><tr><td>1.5A</td><td>41.94 -0.97</td><td>[−4.85, +2.91]</td><td>5.76</td><td>76.9 0.422</td></tr><tr><td>Shifted  $M _ { 0 }$ </td><td>36.31</td><td>-6.60 [−11.84, -1.55]</td><td>5.31</td><td>77.3 0.428</td></tr><tr><td>Random  $A \# 1$ </td><td>37.67</td><td>−5.24 [-9.90, −0.78]</td><td>5.72</td><td>75.3 0.414</td></tr><tr><td>Random A #2</td><td>35.73</td><td>−7.18 [−12.62, −1.75]</td><td>5.80</td><td>75.0 0.418</td></tr><tr><td>Random A #3</td><td>36.70</td><td>-6.21 [−11.26, -1.17]</td><td>5.79</td><td>75.3 0.404</td></tr></table>

Table C5. Support-control differences relative to the main Partial in Table C4. Near-zero bounds retain enough digits to preserve their sign.
<table><tr><td></td><td colspan="6">Temporal diagnostics</td></tr><tr><td>Support</td><td colspan="2">∆HC↓</td><td colspan="2">∆DD↑</td><td colspan="2">∆BMR↓</td></tr><tr><td>0.5A</td><td></td><td>-0.53 [−0.83, -0.25]</td><td></td><td>+0.39 [−1.36, +2.14]</td><td></td><td>-0.017 [-0.033, -0.001]</td></tr><tr><td>0.75A</td><td></td><td>-0.34 [−0.68, -0.02]</td><td></td><td>+0.39 [−1.17, +1.94]</td><td></td><td>-0.013 [−0.026, +0.001]</td></tr><tr><td>1.25A</td><td></td><td>+0.10 [−0.15, +0.36]</td><td></td><td>-0.39 [−2.33, +1.36]</td><td></td><td>+0.019 [+0.001, +0.036]</td></tr><tr><td>1.5A</td><td></td><td>+0.00 [−0.28, +0.29]</td><td></td><td>+0.19 [−1.36, +1.94]</td><td></td><td>+0.036 [+0.017, +0.056]</td></tr><tr><td>Shifted  $M _ { 0 }$ </td><td>-0.44</td><td> $[ - 0 . 8 6 , - 0 . 0 3 ]$ </td><td></td><td> $+ 0 . 5 8 \ [ - 1 . 9 4 , + 3 . 1 1 ]$ </td><td></td><td>+0.041 [+0.018, +0.066]</td></tr><tr><td>Random A #1</td><td>-0.04</td><td> $[ - 0 . 4 1 , + 0 . 3 3 ]$ </td><td></td><td>-1.36 [-3.50, +0.78]</td><td></td><td>+0.028 [+0.002, +0.054]</td></tr><tr><td>Random</td><td></td><td> $A \# 2 \mathrm { \ t } + 0 . 0 5 \left[ - 0 . 3 5 , + 0 . 4 5 \right]$ </td><td>-1.75</td><td> $\left[ - 3 . 8 8 , + 0 . 3 9 \right]$ </td><td>+0.031</td><td> $\left[ + 0 . 0 0 8 , + 0 . 0 5 6 \right]$ </td></tr><tr><td>Random A #3 +0.03 [−0.39, +0.45]</td><td></td><td></td><td></td><td>-1.36 [-3.69, +0.97]</td><td></td><td>+0.017 [-0.010, +0.045]</td></tr></table>

Table C6. Matched-area supports by the share of $M _ { 0 }$ they cover by chance, |S ∩ $M _ { 0 } | / | M _ { 0 } |$ . Random draws (three per prompt, 309 prompt–draw pairs) are split into quartiles of coverage; shifted supports into zero, partial, and higher overlap. ∆ is relative to Partial; kept is the share of the endpoint’s completion gain retained, $( \mathrm { T C R } _ { S } - \mathrm { T C R } _ { \mathrm { E F } } ) / ( \mathrm { T C R } _ { \mathrm { P a r t i a l } } - \mathrm { T C R } _ { \mathrm { E F } } ) .$
<table><tr><td>Support, coverage Pairs (prompts) Mean coverage</td><td></td><td></td><td> $M _ { 0 }$  area</td><td>∆TCR ↑ [95% CI]</td><td></td><td>Kept</td></tr><tr><td colspan="7">Random  $A \ : ( 2 \times 2 \ : b l o c k s )$ </td></tr><tr><td> $0 . 0 { - } 2 . 2 \%$ </td><td>77 (46)</td><td>0.5%</td><td>2.6%</td><td></td><td> $- 9 . 0 9 ^ { * } \ [ - 1 6 . 6 2 , - 1 . 4 5 ]$ </td><td>-0.59</td></tr><tr><td>2.2-6.4%</td><td>77 (52)</td><td>4.2%</td><td>4.5%</td><td></td><td> $- 8 . 3 1 ^ { * } \left[ - 1 5 . 7 0 , - 2 . 3 1 \right]$ </td><td>-0.07</td></tr><tr><td> $6 . 4 \substack { - 1 1 . 9 \% }$ </td><td>77 (51)</td><td>8.9%</td><td>7.3%</td><td>-5.19</td><td> $[ - 1 3 . 0 0 , + 2 . 4 7 ]$ </td><td>+0.44</td></tr><tr><td> $1 1 . 9 \mathrm { - } 5 4 . 0 \%$ </td><td>78 (34)</td><td>22.4%</td><td>20.9%</td><td></td><td>-2.31 [−10.86, +7.13]</td><td>+0.83</td></tr><tr><td> $S h i f t e d M _ { 0 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0%</td><td>-(41)</td><td></td><td></td><td></td><td>4.1% −12.68* [-22.50, -3.75]</td><td>-0.04</td></tr><tr><td>0-8%</td><td>-(27)</td><td></td><td>8.1%</td><td></td><td>-1.48 [-9.47, +6.67]</td><td>+0.33</td></tr><tr><td>&gt; 8%</td><td>- (35)</td><td></td><td>15.1%</td><td></td><td>-3.43 [-11.25, +5.00]</td><td>+0.68</td></tr></table>

## C.3 Placement Beyond Chance Coverage

Chance coverage. Random and shifted supports are placed without reference to $M _ { 0 } ,$ , yet they cover part of it by chance: random supports cover 9.0% of $M _ { 0 }$ on average (median 6%), and shifted supports, despite IoU below 0.1, overlap $M _ { 0 }$ in 62 of 103 prompts (8.0% on average). When the draws are grouped by this coverage (Table C6), supports that cover little of $M _ { 0 }$ lose the endpoint’s completion gain, shifted supports with zero coverage fall to the Endpoint-free level, and supports that cover more of $M _ { 0 }$ retain more of the gain. Coverage also varies with $M _ { 0 } { } ^ { \circ } { \mathbf { s } }$ size across prompts, so we control it directly.

Table C7. Wan2.2-Turbo supports with exactly $| M _ { 0 } |$ cells: controlled overlap with $M _ { 0 }$ and support shape. Differences are paired with the main Partial (103 prompts × five seeds); kept is defined as in Table C6. HC is multiplied by 100.
<table><tr><td>Support</td><td>Coverage</td><td>TCR↑</td><td colspan="2">∆TCR ↑ [95% CI]</td><td></td><td>Kept ∆HC ↓</td><td>∆DD↑</td><td>∆BMR↓</td></tr><tr><td>Partial  $( M _ { 0 } )$ </td><td>100%</td><td>42.91</td><td colspan="2"></td><td></td><td></td><td></td><td></td></tr><tr><td>Overlap 75%</td><td>75%</td><td>38.25</td><td>-4.66*</td><td>[−8.54, -0.78]</td><td>+0.49</td><td>-0.23</td><td>+0.39</td><td>+0.013</td></tr><tr><td>Overlap 50%</td><td>50%</td><td>38.83</td><td>-4.08</td><td>[-8.74, +0.58]</td><td>+0.55</td><td>-0.44*</td><td>-1.17</td><td>+0.015</td></tr><tr><td>Overlap 25%</td><td>25%</td><td>37.86</td><td>-5.05</td><td>[−10.49, +0.19]</td><td>+0.45</td><td>-0.65*</td><td>-0.97</td><td>+0.021</td></tr><tr><td>Overlap 0%</td><td>0%</td><td>30.29</td><td>-12.62*</td><td>[−18.06, -7.38]</td><td>-0.38</td><td>-0.81*</td><td>-0.39</td><td>+0.025*</td></tr><tr><td>Random  $2 \times 2$  (main, 3 draws)</td><td>9.0%</td><td>36.70</td><td>-6.21*</td><td>[−10.87, -1.68]</td><td>+0.32</td><td>+0.01</td><td>-1.49</td><td> $+ 0 . 0 2 5 ^ { \ast }$ </td></tr><tr><td>Random 4 × 4 blocks</td><td>9.2%</td><td>38.64</td><td>-4.27</td><td>[-9.32, +0.58]</td><td>+0.53</td><td>-0.51*</td><td>-0.97</td><td>+0.023*</td></tr><tr><td>Random  $8 \times 8$  blocks</td><td>9.2%</td><td>30.49</td><td>-12.43*</td><td>[-17.48, -7.57]</td><td>-0.36</td><td>-0.77*</td><td>-1.36</td><td> $+ 0 . 0 3 3 ^ { \ast }$ </td></tr><tr><td>One blob, random place</td><td>10.3%</td><td>35.53</td><td>-7.38*</td><td>[−12.82, -2.14]</td><td>+0.19</td><td>-0.90*</td><td>-0.19</td><td> $+ 0 . 0 2 5 ^ { \ast }$ </td></tr><tr><td>Shifted  $M _ { 0 }$  (main)</td><td>8.0%</td><td>36.31</td><td>-6.60*</td><td>[−11.84, -1.55]</td><td>+0.28</td><td>-0.44*</td><td>+0.58</td><td> $+ 0 . 0 4 1 ^ { * }$ </td></tr><tr><td>Endpoint-free</td><td></td><td>33.79</td><td colspan="2"></td><td></td><td>一</td><td>一</td><td></td></tr></table>

![](images/3f69a932581abb044bc9e7a155208d18689e8c3806793e17143033e71d90962f.jpg)

![](images/6863d9aafe9c06512eed05e6a662f66be2c350021e588c7a180dd6e42b607de1.jpg)  
Figure C1. Completion requires covering the edit-relevant support. Wan2.2-Turbo, paired ∆TCR relative to Partial with 95% CIs over 103 prompts and five seeds; open markers are not significant. (a) Supports with exactly $\lvert M _ { 0 } \rvert$ cells covering 0–75% of $M _ { 0 } ,$ , the rest placed outside it; 100% is Partial. (b) Supports placed without reference to $M _ { 0 } ,$ which cover 8–10% of it by chance. The dashed line marks Endpoint-free.

Controlled supports. Every set has exactly $\lvert M _ { 0 } \rvert$ cells on the $4 4 \times 8 0$ latent grid. An overlap set covers a fraction $o \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ of $M _ { 0 }$ with one contiguous sub-blob grown inside $M _ { 0 }$ from a random $M _ { 0 }$ cell and places the remaining cells as one contiguous blob outside $M _ { 0 } ; o = 1$ is Partial. Three shape sets disregard $M _ { 0 } { \mathrm { : } }$ : one contiguous blob grown from a random cell, and scattered 4 × 4 or 8 × 8 blocks on the 4- and 8-cell lattices, versus $2 \times 2$ blocks for the main random supports. Seeds are derived from the prompt and set. All sets use the main Wan2.2 sampler through the control runner, which reproduces the main Partial videos bit for bit, and are paired with the main Partial.

Results. A support that misses $M _ { 0 }$ entirely lowers TCR by 12.62 pp, below Endpoint-free, matching the zero-coverage shifted supports (−12.68 pp). Covering 25–75% of $M _ { 0 }$ retains 45–55% of the endpoint’s completion gain, and only the full region recovers the rest: even 75% coverage stays significantly below Partial (−4.66 pp). All four random shapes lower TCR, three significantly, and all raise BMR significantly (Table C7; Fig. C1). HC rises steadily with coverage, from 4.95 at zero overlap to 5.76 for Partial, consistent with HC arising where endpoint content is imposed; DD stays comparable throughout.

Static versus moving edit targets. Placement matters most when the edited object stays in place. We split prompts at the median optical flow inside $M _ { 0 } ,$ with the split fixed before analysis. Among the 51 prompts below the median, shifted and random supports lower TCR by 10.20 and 9.67 pp (both significant), against 3.08 and 2.82 pp among the

Table C8. Full–Endpoint-free completion differences. All rows use FLUX.2 endpoints except the ChronoEdit and Qwen-Image-Edit rows.
<table><tr><td colspan="4">Transition completion</td></tr><tr><td colspan="4">Endpoint-free</td></tr><tr><td>Backbone</td><td>TCR↑</td><td>Full TCR ↑</td><td>Full–Endpoint-free [95% CI]</td></tr><tr><td>Wan2.2-Turbo</td><td>33.79</td><td>41.94</td><td>+8.16 [+1.75, +14.56]</td></tr><tr><td>Wan2.2-Turbo (ChronoEdit)</td><td>33.79</td><td>40.19</td><td>+6.41 [−0.19, +13.01]</td></tr><tr><td>Wan2.2-Turbo (Qwen-Image-Edit)</td><td>33.79</td><td>35.15</td><td>+1.36 [−5.63, +8.54]</td></tr><tr><td>SANA-Video</td><td>23.50</td><td>35.73</td><td>+12.23 [+5.44, +19.22]</td></tr><tr><td>HunyuanVideo-1.5</td><td>40.97</td><td>46.60</td><td>+5.63 [0.00, +11.46]</td></tr><tr><td>MfM</td><td>32.62</td><td>43.88</td><td>+11.26 [+3.88, +18.64]</td></tr><tr><td>OmniWeaving</td><td>39.42</td><td>43.30</td><td>+3.88 [−2.52, +10.49]</td></tr><tr><td>Wan2.2 Base</td><td>31.07</td><td>45.24</td><td>+14.17 [+8.16, +20.39]</td></tr></table>

Table C9. Wan2.2-Turbo completion by edit type: TCR means and paired differences. Partial uses the main fixed-support cohort.
<table><tr><td rowspan="2">Edit type</td><td colspan="6">Transition completion</td></tr><tr><td>Endpoint- free</td><td></td><td>Full Partial</td><td>Full- Endpoint-free</td><td>Partial- Endpoint-free</td><td>Partial-Full</td></tr><tr><td>Attribute binding</td><td></td><td>31.6048.40</td><td>47.60</td><td>+16.80</td><td>+16.00 [+7.27, +26.79] [+5.56, +26.67] [−6.55, +4.58]</td><td>-0.80</td></tr><tr><td>Background shifts</td><td></td><td>38.67 46.67</td><td>54.67</td><td>+8.00</td><td>+16.00 [−5.33, +23.33] [+3.08, +30.91] [0.00, +20.00]</td><td>+8.00</td></tr><tr><td>Object binding</td><td></td><td>34.74 31.58</td><td>32.11</td><td>-3.16</td><td>-2.63 [−13.02, +6.45] [-9.00, +3.89] [-8.12, +9.41]</td><td>+0.53</td></tr><tr><td>Overall</td><td></td><td>33.7941.94</td><td>42.91</td><td>+8.16</td><td>+9.13 [+1.75, +14.56] [+3.11, +15.34] [−3.50, +5.44]</td><td>+0.97</td></tr></table>

other 52. By contrast, Partial’s BMR reduction relative to Full is the same in both groups (−0.214 and −0.211). Across edit types, the endpoint’s completion gain comes from attribute binding and background shifts (+16.0 pp each), where random and shifted supports retain only 50–58% of it.

## C.4 Additional Endpoint Comparisons

Additional endpoint-free versus Full comparisons use MfM, OmniWeaving, and Wan2.2 Base [21, 31, 36].

Full–Endpoint-free TCR gains are significant for Wan2.2, SANA, MfM, and Wan2.2 Base, and comparable for Wan2.2 with ChronoEdit or Qwen-Image-Edit endpoints, Hunyuan, and OmniWeaving (Table C8).

In Wan2.2’s edit-type results, Partial–Endpoint-free intervals are positive for Attribute binding and Background shifts, while the interval for Object binding spans zero. All subgroup Partial–Full TCR intervals include zero (Table C9), while Partial’s BMR reduction relative to Full is significant in every edit type on all three backbones (Table C10).

Temporal artifacts across conditioning mechanisms. Table C11 extends the Full–Endpoint-free comparison to temporal diagnostics. Wan2.2 (with FLUX.2, Qwen-Image-Edit, or ChronoEdit endpoints), SANA, Hunyuan, and Wan2.2 Base impose the edited endpoint through training-free terminal projection, whereas MfM and OmniWeaving learn endpoint conditioning during training. Every Full setting incurs at least one significant temporal artifact, but its form differs. Training-free projection raises HC and BMR in all six settings. Learned conditioning leaves HC comparable but lowers DD by 20.19 pp for MfM and 17.67 pp for OmniWeaving; OmniWeaving also raises BMR by 0.198. Learned conditioning therefore does not remove the temporal artifacts of full-frame endpoints; it shifts them toward motion suppression. Comparisons are paired within each model, whose frame counts and resolutions differ.

Table C10. Partial–Full differences by edit type on all three backbones (50 attribute-binding, 38 object-binding, and 15 background shift prompts; five seeds each). Subgroup intervals follow Appendix C.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Edit type</td><td colspan="2">Transition completion</td><td colspan="6">Temporal diagnostics</td></tr><tr><td colspan="2">∆TCR↑</td><td colspan="2">∆HC↓</td><td colspan="2">∆DD↑</td><td colspan="2">∆BMR↓</td></tr><tr><td>Wan2.2-Turbo</td><td>Attribute binding</td><td>-0.80</td><td>[−6.55, +4.58]</td><td>-0.68</td><td>[−1.57, +0.16]</td><td></td><td>+9.20* [+3.50, +15.56]</td><td>-0.201*</td><td>[−0.256, −-0.147]</td></tr><tr><td></td><td>Object binding</td><td>+0.53</td><td>[−8.12, +9.41]</td><td>-0.50</td><td>[−1.30, +0.24]</td><td>+10.53* [+2.94, +19.38]</td><td></td><td>-0.210*</td><td>[−0.308, −0.115]</td></tr><tr><td></td><td>Background shifts</td><td>+8.00</td><td>[0.00, +20.00]</td><td>-1.38</td><td>[−3.18, +0.22]</td><td>+10.67 [0.00, +23.64]</td><td>-0.258*</td><td></td><td>[−0.403, −0.121]</td></tr><tr><td>SANA-Video</td><td>Attribute binding</td><td>-6.80</td><td>[−16.25, +2.46]</td><td>-0.07</td><td>[−1.39, +1.18]</td><td>+12.00*</td><td>[+5.42, +18.80]</td><td>-0.309*</td><td>[−0.369, −0.249]</td></tr><tr><td></td><td>Object binding</td><td>+4.21</td><td>[−5.29, +13.55]</td><td>-2.38*</td><td>-3.78, −1.00]</td><td>+4.74 [−5.50, +15.56]</td><td>-0.293*</td><td></td><td>[−0.356, −0.231]</td></tr><tr><td></td><td>Background shifts</td><td>+9.33</td><td>[0.00, +20.00]</td><td>-0.37</td><td>-3.59, +3.66]</td><td>+13.33 [−3.64, +30.00]</td><td></td><td>-0.184*</td><td>[−0.260,−0.107]</td></tr><tr><td>HunyuanVideo-1.5</td><td>Attribute binding</td><td>-4.00</td><td>[-11.30, +3.11]</td><td>-6.82*</td><td>[−8.29, −5.39]</td><td>0.00</td><td>[−3.64, +3.75]</td><td>-0.316*</td><td>[−0.371, -0.259]</td></tr><tr><td></td><td>Object binding</td><td></td><td>0.00 [−10.67, +10.45]</td><td>-7.67*</td><td>-8.98, −6.44]</td><td>-2.11</td><td>[−5.00, +0.62]</td><td>-0.153*</td><td>-0.206,−0.100]</td></tr><tr><td></td><td>Background shifts</td><td>-1.33</td><td>[−10.00, +7.69]</td><td>-7.75*</td><td>-9.79, −5.93]</td><td>-2.67</td><td>[−10.67, +4.44]</td><td>-0.289*</td><td>[−0.465, −0.123]</td></tr></table>

Table C11. Full–Endpoint-free temporal differences by endpoint-conditioning mechanism, with Endpoint-free levels (EF). Paired prompt bootstrap as in Table C1; comparisons are paired within each model. Rows without an editor use FLUX.2 endpoints.
<table><tr><td></td><td colspan="3">HC↓</td><td colspan="3">DD↑(%)</td><td colspan="3">BMR↓</td></tr><tr><td>Model</td><td>EF</td><td>∆ [95% CI]</td><td></td><td>EF</td><td>∆ [95% CI]</td><td></td><td>EF</td><td>∆ [95% CI]</td><td></td></tr><tr><td>Training-free terminal projection</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wan2.2-Turbo</td><td>4.57</td><td></td><td>+1.90 [+1.27, +2.59]</td><td>79.4</td><td></td><td>-12.62 [−17.48, -8.16]</td><td>0.360</td><td></td><td>+0.239 [+0.189, +0.289]</td></tr><tr><td>SANA-Video</td><td>8.07</td><td></td><td>+1.70 [+0.67, +2.70]</td><td>42.7</td><td></td><td>-9.90 [−15.53, -4.27]</td><td>0.122</td><td>+0.298</td><td>[+0.254, +0.343]</td></tr><tr><td>HunyuanVideo-1.5</td><td>3.47</td><td></td><td>+9.50 [+8.53, +10.51]</td><td>51.5</td><td></td><td>+1.17 [−1.55, +3.88]</td><td>0.305</td><td>+0.279</td><td>[+0.233, +0.326]</td></tr><tr><td>Wan2.2 Base</td><td>5.71</td><td></td><td>+1.47 [+0.76, +2.24]</td><td>60.0</td><td></td><td>-2.14 [−6.99, +2.72]</td><td>0.415</td><td>+0.082</td><td>[+0.041,+0.123]</td></tr><tr><td>Wan2.2-Turbo (Qwen-Image-Edit)</td><td>4.57</td><td></td><td>+2.54 [+1.80, +3.35]</td><td>79.4</td><td></td><td>-15.34 [−20.19, -10.68</td><td>0.360</td><td>+0.256</td><td>[+0.207, +0.307]</td></tr><tr><td>Wan2.2-Turbo (ChronoEdit)</td><td>4.57</td><td></td><td>+1.55 [+0.93, +2.18]</td><td>79.4</td><td></td><td>-11.46 [−16.89, −6.02]</td><td>0.360</td><td></td><td>+0.198 [+0.148, +0.249]</td></tr><tr><td>Learned endpoint conditioning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MfM</td><td>3.61</td><td></td><td>+0.52 [−0.02, +1.12]</td><td>48.3</td><td></td><td>-20.19 [-27.57, -13.20]</td><td></td><td></td><td>0.368 +0.019 [−0.025, +0.063]</td></tr><tr><td>OmniWeaving</td><td>6.06</td><td></td><td>−0.41 [−0.90, +0.08]</td><td>54.6</td><td></td><td>-17.67 [−24.27, -11.07]</td><td>0.289</td><td></td><td>+0.198 [+0.149, +0.245]</td></tr></table>

## C.5 Frame Guidance with Edit-Relevant Support

Frame Guidance steers a video model toward target frames by backpropagating a frame loss through the VAE decoder during sampling [14]. We run its official keyframe configuration on its primary backbone, CogVideoX-5B-I2V [43]: 49 frames at 480 × 720 and 8 FPS, 50 steps, and CFG 6, with guidance on the last frame only. Inputs are the same 103 prompts and five seeds (22001–22005); source images and FLUX.2 endpoints are center-cropped to 3:2, and $M _ { 0 }$ cropped accordingly. FG-Full guides the whole last frame toward the endpoint, $\mathrm { F G } \mathrm { - } M _ { 0 }$ weights the same loss by the cropped $M _ { 0 }$ through the official mask\_video option, and Endpoint-free runs the same sampler without guidance. BMR uses the cropped $M _ { 0 }$

Table C12 gives the paired intervals behind Section 5.6. Relative to Endpoint-free, $\mathrm { F G } \mathrm { - } M _ { 0 }$ keeps a TCR gain of +10.10 pp with comparable DD (−1.94 pp). FG-Full’s BMR increase is small (+0.045, [−0.001, +0.090]), leaving no reversal for the restriction to remove. Of the 108 FG-M videos with HC above 10, 39 have their minimum-similarity pair in the first second, against 15 of 75 for FG-Full; these early events are changes to unrelated scenes rather than jumps toward the endpoint.

These three conditions were judged and measured on A100 GPUs, with the same judge model and revision served through a vLLM build for CUDA 12.9 and the same metric code. On the 515 Wan2.2 Full videos, this setup reproduces the original verdicts for 96.7% of videos, with a TCR difference of −0.58 pp [−2.14, +0.78], within the variation of re-judging on the original machine (95.7%; −0.78 pp [−2.52, +0.97]). The three Frame Guidance conditions are compared only with one another.

Table C12. Frame Guidance on CogVideoX-5B-I2V (103 prompts × five seeds). FG-M<sub>0</sub> weights the same frame loss by the cropped $M _ { 0 }$ . Minutes per video on one A6000. HC is multiplied by 100.
<table><tr><td rowspan="2">Condition</td><td>Transition completion</td><td colspan="3">Temporal diagnostics</td><td rowspan="2">Min/video</td></tr><tr><td>TCR↑</td><td>HC↓</td><td>DD↑</td><td>BMR↓</td></tr><tr><td>Endpoint-free</td><td>27.18</td><td>5.44</td><td>44.7</td><td>0.388</td><td>7</td></tr><tr><td>FG-Full (whole last frame)</td><td>36.89</td><td>5.40</td><td>30.5</td><td>0.434</td><td>57</td></tr><tr><td>FG-M0 (last frame,  $M _ { 0 }$  only)</td><td>37.28</td><td>6.90</td><td>42.7</td><td>0.427</td><td>57</td></tr><tr><td>FG-Full — Endpoint-free</td><td>+9.71*</td><td>-0.04</td><td>-14.17*</td><td>+0.045</td><td></td></tr><tr><td rowspan="2"> $\mathrm { F G } { \cdot } M _ { 0 } \mathrm { - F G } { \cdot } \mathrm { F u l l }$ </td><td>[+3.69, +15.92]</td><td></td><td></td><td>[−0.65, +0.59] [-20.19, -8.16] [−0.001, +0.090]</td><td></td></tr><tr><td>+0.39</td><td>+1.50*</td><td>+12.23*</td><td>-0.006</td><td></td></tr><tr><td rowspan="2"></td><td>[-3.69, +4.47]</td><td></td><td></td><td>[+0.89, +2.14] [+7.18, +17.48] [−0.043, +0.032]</td><td></td></tr><tr><td>+10.10*</td><td>+1.46*</td><td>-1.94</td><td>+0.039</td><td></td></tr><tr><td>FG-M0 - Endpoint-free</td><td> $[ + 4 . 4 7 , + 1 5 . 9 2 ]$ </td><td></td><td>[+0.66, +2.30] [−6.99, +3.11] [−0.001, +0.079]</td><td></td><td></td></tr></table>

## D Donor-Context Intervention and Score Decomposition

## D.1 Construction, Pairing, and Scope

The conditions separate terminal content from projection support:

Table D1. Donor-context construction. $\mathrm { O } / \mathrm { D } / D _ { \mathrm { c r o s s } } / \mathrm { N }$ all retain Full support. Content inside $M _ { 0 }$ always comes from the original edited endpoint.
<table><tr><td>Condition</td><td>Inside  $M _ { 0 }$ </td><td>Outside  $M _ { 0 }$ </td><td>Support</td></tr><tr><td>O: Original Full</td><td>Original latent</td><td>Original latent</td><td>Full</td></tr><tr><td>D: Donor-context Full</td><td>Original latent</td><td>Re-encoded donor final RGB</td><td>Full</td></tr><tr><td> $D _ { \mathrm { c r o s s } } \colon$  Cross-seed donor</td><td>Original latent</td><td>Re-encoded final RGB of another seed&#x27;s donor</td><td>Full</td></tr><tr><td>N: Native-latent Full</td><td>Original latent</td><td>Native donor tail latent</td><td>Full</td></tr><tr><td>P: Partial donor</td><td>Original latent</td><td>Video prediction retained</td><td> $M _ { 0 }$ </td></tr></table>

Generation and pairing. On Hunyuan, we use 480p, eight steps, and 121 frames at 24 FPS, with the native scheduler and noise-matched projection. Each of the 103 prompts uses seeds 22001–22005. Within each pair, source, prompt, edited endpoint, transformed $M _ { 0 } ,$ and initial/endpoint noise are fixed.

P is a Partial rollout. We take its final RGB frame before compression and its native terminal latent. We encode the original endpoint and donor RGB through the same five-frame repetition procedure and select the second temporal slot. D uses Eq. (5); N uses the native donor slot instead. Both retain the original endpoint inside $M _ { 0 }$

O reuses the main Full videos; P, D, and N each contain 515 new videos. The donor P cohort is separate from main Partial (Table B1).

On Wan2.2, all four conditions are new 515-video runs with the main four-step sampler, 121 frames at 24 FPS, and one CUDA generator per prompt–seed that drives the initial noise and per-step re-noising of every condition. P is Partial with the main settings; we save its final RGB frame before compression and its native terminal latent. The endpoint and donor RGB use the same tail-slot encoding. $M _ { 0 }$ enters as the pooled weight map of Wan2.2 Partial $( 2 \times 2$ average pooling to the $2 2 \times 4 0$ token grid, repeated to the $4 4 \times 8 0$ latent grid), so it selects content, while O, D, and N use Full support (w = 1).

Both backbones use the same BMR measurement mask for all conditions and all 515 pairs, without DD filtering.   
Confidence intervals follow Appendix C.

Interpretation. O→D changes non-edit terminal content under fixed Full support. The donor is generated, not groundtruth future context; position/visibility, generated-image properties, and trajectory compatibility are not separately

Table D2. Donor-context condition means. Within each backbone, $\mathrm { O } / \mathrm { D } / D _ { \mathrm { c r o s s } } / \mathrm { N }$ use Full support; P is the Partial donor cohort. All conditions share the BMR measurement mask
<table><tr><td rowspan="2">Condition</td><td colspan="2">Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td>TCR↑</td><td>TC-Score ↑</td><td>HC↓</td><td>DD↑</td><td>BMR↓</td></tr><tr><td colspan="6">HunyuanVideo-1.5</td></tr><tr><td>O — Original Full</td><td>46.60</td><td>0.859</td><td>12.96</td><td>52.62</td><td>0.585</td></tr><tr><td>D — Donor-context Full</td><td>47.57</td><td>0.873</td><td>8.10</td><td>52.82</td><td>0.358</td></tr><tr><td> $D _ { \mathrm { c r o s s } }$  — Cross-seed donor</td><td>46.80</td><td>0.869</td><td>11.27</td><td>52.04</td><td>0.480</td></tr><tr><td>N — Native-latent Full</td><td>45.63</td><td>0.879</td><td>7.16</td><td>52.23</td><td>0.354</td></tr><tr><td>P — Partial</td><td>44.66</td><td>0.882</td><td>5.69</td><td>51.46</td><td>0.333</td></tr><tr><td colspan="6">Wan2.2-Turbo</td></tr><tr><td>O — Original Full</td><td>43.30</td><td>0.834</td><td>6.49</td><td>65.44</td><td>0.600</td></tr><tr><td>D — Donor-context Full</td><td>43.11</td><td>0.821</td><td>6.12</td><td>75.34</td><td>0.362</td></tr><tr><td>N — Native-latent Full</td><td>42.72</td><td>0.820</td><td>6.26</td><td>74.95</td><td>0.382</td></tr><tr><td>P — Partial</td><td>42.72</td><td>0.819</td><td>5.76</td><td>76.70</td><td>0.387</td></tr></table>

manipulated. D→N tests terminal representation. Partial itself ignores endpoint content outside $M _ { 0 } ,$ , so replacing that content does not change its projection.

## D.2 Cross-Seed Donors on Hunyuan

For each prompt, a cyclic shift $k \in \{ 1 , 2 , 3 , 4 \}$ assigns donor seed index $( j + k )$ mod 5 to recipient seed index j. Shifts are balanced across prompts (26/26/26/25) and were fixed before generation. The recipient keeps its source, prompt, seed, initial and endpoint noises, and transformed $M _ { 0 } ;$ only the donor frame used outside $M _ { 0 }$ changes. A pilot reproduced the existing Original Full videos and, with the donor seed set to the recipient seed, the same-seed D videos exactly. Cross-seed donors retain 35% ([22, 45]) of the same-seed HC reduction and 46% ([33, 58]) of the BMR reduction (Tables D2 and D3).

## D.3 Secondary Donor and Representation Contrasts

On Hunyuan, donor-context Full has $+ 2 . 9 1 \ \mathrm { [ 0 . 0 0 , + 5 . 8 3 ] } \ \mathrm { p p }$ TCR relative to P, with higher HC and BMR; the completion interval includes zero. Replacing the re-encoded donor with its native slot further reduces HC $( - 0 . 9 4 \ : [ - 1 . 1 7 , - 0 . 7 1 ] )$ , while the BMR interval spans zero (Table D3). On Wan2.2, the native terminal latent leaves TCR, HC, and DD comparable to the re-encoded donor and raises BMR $( + 0 . 0 2 0 \left[ + 0 . 0 0 1 , + 0 . 0 4 0 \right] )$ ). Relative to P, D has comparable TCR and DD, slightly lower BMR, and slightly higher HC (Table D3).

## D.4 Descriptive Closure of the Full–Partial Gap

For an artifact metric m, the two summaries use different denominators:

$$
\mathrm { R e l a t i v e ~ r e d u c t i o n } = \frac { \bar { m } _ { O } - \bar { m } _ { D } } { \bar { m } _ { O } } , \qquad R _ { m } = \mathrm { G a p ~ c l o s u r e } = \frac { \bar { m } _ { O } - \bar { m } _ { D } } { \bar { m } _ { O } - \bar { m } _ { P } } .\tag{D1}
$$

On Hunyuan, O→D reduces HC by 37.5% and BMR by 38.8% relative to O. Using donor P as reference, it closes 66.83% and 90.16% of the Full–Partial gaps. Residual D–P differences are +2.41 [+2.07, +2.75] for HC and +0.025 [+0.004, +0.046] for BMR. On Wan2.2, with the same-run P as reference, D closes 112% ([100, 125]) of the BMR gap, reaching P’s BMR; the HC closure is uncertain $( 5 1 \% , [ - 5 8 , 9 3 ] )$ because the Full–Partial HC gap is small. Gap closure summarizes these mean differences; it is not a causal mediation estimate.

## D.5 TC-Score Decomposition and Its Relation to HC

For $\mathrm { T C - S c o r e } = { \textstyle { \frac { 2 } { 3 } } } V + { \textstyle { \frac { 1 } { 3 } } } C .$ , we bootstrap the VLM term V and mapped CLIP term C over the same paired prompts. Their weighted O→D contributions are +0.00298 and +0.01150, totaling +0.01448 (Table D4). The increase is mainly in CLIP consistency; the VLM interval includes zero. On Wan2.2, the weighted contributions are −0.00258 and −0.01031, totaling −0.01289: the decrease is mainly in CLIP consistency, the VLM interval includes zero, and it

Table D3. Paired donor-context differences. D–O and $D _ { \mathrm { c r o s s } } – 0$ are the primary contrasts; the others are secondary. Appendix C specifies the uncertainty interpretation.
<table><tr><td rowspan="2"></td><td colspan="2">Transition completion</td><td colspan="3">Temporal diagnostics</td></tr><tr><td>Difference ∆TCR↑</td><td>∆TC-Score ↑</td><td>∆HC↓</td><td>∆DD↑</td><td>∆BMR↓</td></tr><tr><td colspan="6">HunyuanVideo-1.5</td></tr><tr><td>D-0</td><td>+0.97</td><td>+0.014 [-4.66, +6.60] [+0.003, +0.026] [−5.75,-3.99] [−1.55, +1.94] [−0.267, −0.186]</td><td>-4.86</td><td>+0.19</td><td>-0.227</td></tr><tr><td> $D _ { \mathrm { c r o s s } } – 0$ </td><td>+0.19 [−5.63, +6.02] [−0.000, +0.021] [−2.47, -0.94] [-2.14, +0.78] [−0.141, -0.068]</td><td>+0.010</td><td>-1.70</td><td>-0.58</td><td>-0.105</td></tr><tr><td> $D _ { \mathrm { c r o s s } } – \mathbf { D }$ </td><td>-0.78</td><td>-0.004 [-3.88, +2.33] [−0.009, +0.001] [+2.65, +3.71] [-2.14, +0.58] [+0.091, +0.154]</td><td>+3.16</td><td>-0.78</td><td>+0.122</td></tr><tr><td>N-O</td><td>-0.97 [−6.60, +4.66] [+0.010, +0.030] [−6.68, -4.94] [-1.94, +1.17] [−0.272, -0.189]</td><td>+0.020</td><td>-5.80</td><td>-0.39</td><td>-0.231</td></tr><tr><td>N-D</td><td>-1.94</td><td>+0.005</td><td>-0.94</td><td>-0.58</td><td>-0.004</td></tr><tr><td>O-P</td><td>+1.94</td><td>-0.023</td><td>+7.27</td><td>+1.17</td><td>[−5.44, +1.36] [−0.000, +0.011] [−1.17, -0.71] [-1.55, +0.39] [−0.018, +0.009] +0.251</td></tr><tr><td>D-P</td><td>+2.91 [0.00, +5.83]</td><td>-0.008</td><td>+2.41</td><td>+1.36 [+0.004, +0.046]</td><td>[-3.50,+7.38] [−0.034, -0.012] [+6.39, +8.17] [-1.17, +3.50] [+0.208,+0.295] +0.025</td></tr><tr><td colspan="6">[−0.014, -0.003] [+2.07, +2.75] [−0.97, +3.50] Wan2.2-Turbo</td></tr><tr><td>D-0</td><td>-0.19 -0.58</td><td>-0.013 [−5.05, +4.27] [−0.024, −0.002] [−0.95, +0.18] [+5.63, +14.56] [−0.283,−0.195] -0.013</td><td>-0.37</td><td>+9.90</td><td>-0.238</td></tr><tr><td>N-O</td><td>-0.39</td><td>[−5.44, +4.08] [−0.026, −0.001] [−0.74, +0.24] [+4.85, +14.37] [−0.263, −0.174] -0.000</td><td>-0.24 +0.14</td><td>+9.51</td><td>-0.218</td></tr><tr><td>N-D</td><td>+0.58</td><td>[-2.91,+2.14] [−0.007, +0.006] [−0.14, +0.41] [-2.14,+1.36] [+0.001,+0.040] +0.015</td><td>+0.74</td><td>-0.39 -11.26</td><td>+0.020 +0.213</td></tr><tr><td>O-P</td><td>+0.39</td><td>+0.002</td><td>+0.36</td><td></td><td>[−4.47, +5.83] [+0.003, +0.027] [+0.22, +1.29] [−16.31, -6.60] [+0.170, +0.258]</td></tr><tr><td>D-P</td><td></td><td>[-3.11, +4.08] [−0.005, +0.009] [+0.05, +0.69] [-3.88, +1.17] [−0.049, -0.001]</td><td></td><td>-1.36</td><td>-0.025</td></tr></table>

Table D4. O→D decomposition of TC-Score on each backbone. Weighted VLM and CLIP terms sum to TC-Score.
<table><tr><td>Component</td><td>O mean</td><td>D mean</td><td>∆ [95% CI]</td><td></td></tr><tr><td colspan="5">HunyuanVideo-1.5</td></tr><tr><td>VLM</td><td>0.86545</td><td>0.86991</td><td></td><td>+0.00447 [−0.01200, +0.02020]</td></tr><tr><td>Mapped CLIP</td><td>0.84545</td><td>0.87995</td><td></td><td>+0.03451 [+0.02604, +0.04341]</td></tr><tr><td>2-31-3 VLM</td><td>0.57696</td><td>0.57994</td><td></td><td>+0.00298 [-0.00800, +0.01347]</td></tr><tr><td>mapped CLIP</td><td>0.28182</td><td>0.29332</td><td></td><td>+0.01150 [+0.00868, +0.01447]</td></tr><tr><td>TC-Score ↑</td><td>0.85878</td><td>0.87326</td><td></td><td>+0.01448 [+0.00293, +0.02575]</td></tr><tr><td colspan="5">Wan2.2-Turbo</td></tr><tr><td>VLM</td><td>0.84855</td><td>0.84468</td><td></td><td>-0.00387 [-0.01737, +0.00941]</td></tr><tr><td>Mapped CLIP</td><td>0.80368</td><td>0.77274</td><td></td><td>-0.03093 [-0.04778, -0.01484]</td></tr><tr><td>2-31-3 VLM</td><td>0.56570</td><td>0.56312</td><td></td><td>-0.00258 [-0.01158, +0.00627]</td></tr><tr><td>mapped CLIP</td><td>0.26789</td><td>0.25758</td><td></td><td>-0.01031 [-0.01593, -0.00495]</td></tr><tr><td>TC-Score ↑</td><td>0.83359</td><td>0.82070</td><td></td><td>-0.01289 [-0.02386, -0.00233]</td></tr></table>

accompanies a 9.90 pp DD increase. TC-Score and HC both use adjacent-frame CLIP similarity, so their agreement is not independent evidence of completion.

Table E1. HunyuanVideo-1.5 Dynamic Degree after removing HC-neighbor pairs or the three largest-flow pairs. The original motion threshold and count requirement remain fixed. HC-neighbor removal drops two pairs at a sequence edge and three otherwise.
<table><tr><td>Condition</td><td>Original DD ↑ Remove</td><td> $t _ { \mathrm { H C } } \pm 1$ </td><td>Remove top-3 flow</td></tr><tr><td>Endpoint-free</td><td>51.5</td><td>48.3</td><td>46.0</td></tr><tr><td>Full</td><td>52.6</td><td>50.5</td><td>47.0</td></tr><tr><td>Partial</td><td>51.5</td><td>49.9</td><td>46.6</td></tr></table>

Table E2. HunyuanVideo-1.5 HC localization and RAFT magnitude. $t _ { \mathrm { H C } }$ is each video’s minimum-similarity pair. The last two columns average RAFT top-5% magnitudes at that pair and at the remaining pairs; they do not use a fixed pair 38. The second column reports the fraction with $t _ { \mathrm { H C } } = 3 8 .$
<table><tr><td>Condition</td><td> $t _ { \mathrm { H C } } = 3 8$ </td><td> $m ( t _ { \mathrm { H C } } )$ </td><td>Other pairs</td></tr><tr><td>Endpoint-free</td><td>1.7%</td><td>19.5 px</td><td>11.3 px</td></tr><tr><td>Full</td><td>95.9%</td><td>88.9 px</td><td>11.4 px</td></tr><tr><td>Partial</td><td>51.8%</td><td>18.3 px</td><td>11.4 px</td></tr></table>

## E Diagnostics and Cross-Checks

## E.1 Hunyuan Discontinuity Localization

Let t<sub>HC</sub> be each video’s minimum-similarity adjacent-frame pair on the 8-FPS grid. We remove either t<sub>HC</sub> and its neighbors or the three largest-flow pairs while keeping DD’s threshold and qualifying-pair count fixed (Table E1). Both removals reduce DD similarly across Endpoint-free, Full, and Partial, so Full’s maintained DD is not explained by a few extreme motion spikes. Pair 38 connects native frames 114 and 117, the onset of Hunyuan’s terminal latent slot, and is t<sub>HC</sub> in 95.9% of Full videos, versus 1.7% for Endpoint-free and 51.8% for Partial. At each video’s own t<sub>HC</sub>, mean RAFT top-5% magnitude is 88.9 px for Full, compared with 19.5 and 18.3 px for Endpoint-free and Partial; all remaining pairs average 11.3–11.4 px (Table E2).

## E.2 Blinded Model-Judge Agreement for HC

Protocol. We use Qwen3.6-35B-A3B with temperature 0, top-p 1, thinking disabled, and a 1,536-token response limit. We evaluate 90 prompts outside the 13-prompt development subset, seeds 22001 and 22002, and nine conditions: Wan2.2 Endpoint-free, Full, Partial, adaptive release, and random release; MfM Full; OmniWeaving Full; and SANA Full and Partial. Hunyuan is excluded. The judge sees all native frames (81–121), resized on a 32-pixel grid within a 768 × 432 per-frame pixel budget. Method names, prompts, endpoints, masks, and metric values are hidden; no DD filter is used.

We ask whether any interval changes abruptly without a continuous transition. The judge provides visible evidence and event intervals before a PRESENT/ABSENT/UNCERTAIN label. Continuous fast motion and smooth fades are excluded, and first/last-frame differences alone are insufficient.

Results. Of 1,620 inputs, 1,619 receive valid judgments, 68 PRESENT and 1,551 ABSENT; one request returned no answer. HC achieves AUROC 0.777 (95% CI [0.70, 0.84]) and AUPRC 0.157, against 4.2% event prevalence. We use stepwise average precision and 1,000 prompt-cluster bootstrap draws (seed 20260917) for AUROC uncertainty. Wan2.2 event rates are 0.0% for Endpoint-free, 8.3% for Full, and 5.0% for Partial.

On 18 prompt–seed pairs across all nine conditions, native-frame judgments agree with judgments using 8 FPS input in 158/162 cases (97.5%) and with those from a repeat evaluation in 159/162 (98.1%).

## E.3 Track Reversal Ratio and Back-Tracking Ratio

Track extraction. CoTracker3 tracks points at the native 24 FPS on videos resized to 640 × 352 [17]. We query a 24 × 24 grid at frames 0, 30, and 60, restricted to the region outside $M _ { 0 }$ dilated by two latent cells.

Table E3. Flow and track diagnostics on the common intersection of 398/515 Wan2.2-Turbo samples applicable under all three conditions. Every mean uses the same samples.
<table><tr><td>Condition</td><td>Common applicable</td><td>BMR↓</td><td>TRR↓</td><td>BTR↓</td></tr><tr><td>Endpoint-free</td><td>398 / 515</td><td>0.386</td><td>0.385</td><td>0.332</td></tr><tr><td>Full</td><td>398 / 515</td><td>0.633</td><td>0.598</td><td>0.600</td></tr><tr><td>Partial</td><td>398 /515</td><td>0.415</td><td>0.410</td><td>0.359</td></tr></table>

Reference-axis construction. Track i is queried at native frame $t _ { i } \in \{ 0 , 3 0 , 6 0 \}$ of $T = 1 2 1$ and has position $p _ { i , t }$ at frame t. Its early reference velocity $v _ { i }$ is the median of valid adjacent displacements during the first third of its remaining duration $T - t _ { i } ;$ at least six valid steps are required. Tracks with $t _ { i } < \lfloor T / 2 \rfloor = 6 0$ , that is, queried at frames 0 and 30, and $\| v _ { i } \| \geq 0 . 3$ pixels per frame provide the global reference

$$
g = { \frac { \sum _ { i } \| v _ { i } \| v _ { i } } { \sum _ { i } \| v _ { i } \| } } , \qquad u _ { g } = g / \| g \| .\tag{E1}
$$

At least ten such tracks and $\| g \| \ge 0 . 2$ pixels per frame are required.

Track applicability. Each subsequently retained track must have at least four valid late steps and an along-axis excursion of at least six pixels; at least ten retained tracks are required.

Trajectory diagnostics. Writing $q _ { i , t } = \langle p _ { i , t } - p _ { i , t _ { i } } , u _ { g } \rangle$ ⟩ for projected displacement and $\Delta q _ { i , t } = q _ { i , t + 1 } - q _ { i , i }$ <sub>t</sub> for a step between adjacent visible frames, TRR measures the fraction of along-axis step mass that is negative in the last third of the video, $t \geq \lfloor 2 T / 3 \rfloor = 8 0 !$

$$
\mathrm { T R R } = \frac { \sum _ { i , t \in \mathcal { L } _ { i } } \operatorname* { m a x } ( 0 , - \Delta q _ { i , t } ) } { \sum _ { i , t \in \mathcal { L } _ { i } } \left| \Delta q _ { i , t } \right| } .\tag{E2}
$$

Here $\mathcal { L } _ { i }$ includes only valid late adjacent steps. BTR measures return from the largest achieved displacement to the last visible track position $q _ { i , \mathrm { e n d } }$ , which need not be the video’s last frame. Write $q _ { i , \operatorname* { m a x } } = \operatorname* { m a x } _ { t } q _ { i , t }$ and $q _ { i , \operatorname* { m i n } } = \operatorname* { m i n } _ { t } q _ { i , \cdot }$ t over visible positions. The per-track ratio and its excursion-weighted aggregation are

$$
e _ { i } = q _ { i , \mathrm { m a x } } - q _ { i , \mathrm { m i n } } , \qquad r _ { i } ^ { \mathrm { r e t u r n } } = \frac { q _ { i , \mathrm { m a x } } - q _ { i , \mathrm { e n d } } } { e _ { i } } ,\tag{E3}
$$

$$
\mathrm { B T R } = \frac { \sum _ { i } e _ { i } r _ { i } ^ { \mathrm { r e t u r n } } } { \sum _ { i } e _ { i } } .\tag{E4}
$$

We add $1 0 ^ { - 6 }$ to normalization denominators for numerical stability. TRR and BTR use point trajectories and a last-third late window, complementing BMR’s flow-based last-half measurement.

Common-population comparison. Track diagnostics apply to 500/515 Endpoint-free, 413/515 Full, and 492/515 Partial videos. We compare all three metrics on their common 398 prompt–seed pairs across 98 prompts (Table E3). Partial–Full differences are −0.218, −0.188, and −0.241 in BMR, TRR, and BTR. Pooled Spearman correlations between BMR and TRR/BTR are 0.64/0.65 across the 1,194 paired, condition-specific observations. Main BMR results use all 515 videos per condition.

## E.4 BMR Measurement Region and Static Backgrounds

Main BMR is measured outside $M _ { 0 } .$ , where Partial leaves the terminal frame unconstrained. Measuring the same quantity over the whole frame, or only inside $M _ { 0 }$ where Full and Partial impose the same edited content, gives the same conclusions: Partial lowers BMR relative to Full on all three backbones, and donor context lowers it relative to Original Full on both tested backbones, in every region (Table E4). Because the reference axis is estimated from early background motion, we also exclude prompt–seed pairs whose early reference motion $\left\| b _ { \mathrm { r e f } } \right\|$ is below one pixel per 8-FPS frame pair in either condition (25–39% of pairs). Partial–Full then remains −0.215 [−0.269, −0.162] on

Table E4. BMR differences by measurement region (103 prompts $\times$ five seeds). Outside $M _ { 0 }$ is the main BMR; the whole-frame and inside- $M _ { 0 }$ variants apply Eq. (4), including its reference axis, to all pixels or to pixels inside $M _ { 0 }$
<table><tr><td>Comparison</td><td colspan="2">Outside  $M _ { 0 }$  (main)</td><td colspan="2">Whole frame</td><td colspan="2">Inside  $M _ { 0 }$ </td></tr><tr><td>Wan2.2-Turbo, Partial – Full</td><td> $- 0 . 2 1 3 ^ { * }$ </td><td>[−0.262, -0.165]</td><td></td><td>-0.202* [-0.251, -0.154]</td><td>-0.141*</td><td> $[ - 0 . 1 8 7 , - 0 . 0 9 7 ]$ </td></tr><tr><td>SANA-Video, Partial — Full</td><td> $- 0 . 2 8 5 ^ { \ast }$ </td><td>[−0.325, -0.246]</td><td>-0.289*</td><td>[-0.329,-0.250]</td><td>-0.281*</td><td> $[ - 0 . 3 2 4 , - 0 . 2 3 8 ]$ </td></tr><tr><td>HunyuanVideo-1.5, Partial – Full</td><td> $- 0 . 2 5 2 ^ { \ast }$ </td><td>[-0.296, -0.208</td><td>-0.248*</td><td>[−0.292, -0.205]</td><td> $- 0 . 2 2 3 ^ { \ast }$ </td><td> $[ - 0 . 2 7 2 , - 0 . 1 7 6 ]$ </td></tr><tr><td>Wan2.2-Turbo, D – O</td><td> $- 0 . 2 3 8 ^ { \ast }$ </td><td>[−0.283, -0.195]</td><td>-0.230*</td><td>[-0.275, -0.186]</td><td> $- 0 . 1 4 3 ^ { * }$ </td><td> $\left[ - 0 . 1 8 8 , - 0 . 0 9 9 \right]$ </td></tr><tr><td>HunyuanVideo-1.5, D – O</td><td> $- 0 . 2 2 7 ^ { \ast }$ </td><td>[−0.267, -0.186]</td><td>-0.218*</td><td>[-0.259,-0.178]</td><td></td><td>-0.183* [−0.225, −0.142]</td></tr></table>

Table E5. Outside- $. M _ { 0 }$ first–last-frame RGB difference of preserved context: mean absolute RGB difference between each video’s first and last frames. For reference, FLUX.2 endpoints differ from the source image by 0.056 [0.045, 0.068] outside $M _ { 0 }$ and 0.232 [0.206, 0.258] inside. Intervals use the paired prompt bootstrap. Wan2.2-Turbo Partial videos were regenerated with identical settings for this measurement.
<table><tr><td rowspan="2">Model</td><td colspan="3">First-last RGB difference</td><td colspan="4">Difference from Endpoint-free</td></tr><tr><td>Endpoint-free</td><td>Full</td><td>Partial</td><td></td><td>Full</td><td></td><td>Partial</td></tr><tr><td>Wan2.2-Turbo</td><td>0.177</td><td>0.076</td><td>0.164</td><td>-0.102</td><td>[-0.116, -0.087]</td><td></td><td>-0.014 [−0.019, -0.009]</td></tr><tr><td>SANA-Video</td><td>0.233</td><td>0.058</td><td>0.216</td><td>-0.174</td><td>[-0.195, −0.155]</td><td></td><td>-0.017 [-0.024, -0.010]</td></tr><tr><td>HunyuanVideo-1.5</td><td>0.127</td><td>0.082</td><td>0.126</td><td>-0.045</td><td>[−0.059, -0.031]</td><td></td><td>-0.001 [-0.002, +0.000]</td></tr></table>

Wan2.2, −0.278 [−0.331, −0.226] on SANA, and −0.298 [−0.355, −0.243] on Hunyuan, and donor context minus Original Full remains −0.251 [−0.303, −0.201] on Wan2.2 and −0.232 [−0.287, −0.176] on Hunyuan.

## E.5 Source-Time Layout of Preserved Context

We measure the first–last-frame RGB difference of preserved context outside the estimated support. Images and frames are downsampled to width 640, and $M _ { 0 }$ is resized using nearest-neighbor interpolation, as for BMR. The primary distance is the mean absolute RGB difference in [0, 1] outside $M _ { 0 } ;$ LPIPS-Alex is secondary [45]. For videos, we compare each video’s first and last frames at its own resolution, avoiding resampling differences between pipelines. In 93% of prompts, FLUX.2 endpoints differ from the source image less outside $M _ { 0 }$ than inside. Outside $M _ { 0 } ,$ , in 91–99% of prompts, Endpoint-free videos change more from their first to their last frame than the endpoint differs from the source. Under Full, the last frame stays close to the first frame outside $M _ { 0 }$ , and Partial leaves the difference within 0.02 of Endpoint-free (Table E5). LPIPS gives the same ordering.

Before computing these quantities, we also specified a dose–response test. Across prompts, we correlate the Endpoint free first–last-frame difference with Full’s HC and BMR increases over Endpoint-free and with Partial’s reductions relative to Full (Spearman correlation with a prompt bootstrap). For Full−Endpoint-free, the difference and the outcomes come from disjoint seeds. Only the two Hunyuan HC tests show a dose–response relation $\left( \rho = + 0 . 2 8 \left[ + 0 . 0 8 , + 0 . 4 7 \right] \right.$ and $- 0 . 3 5 \ [ - 0 . 5 0 , - 0 . 1 7 ] \rangle$ ; the other ten intervals include zero. We therefore do not claim that Full’s artifacts scale with scene motion.

## F Residual Failures and Exploratory Support Release

(a) The edited endpoint preserves an occluding bus-stop sign

![](images/fee86afb9c1d881697855f5ec34e55730d85d7e1dde610f8c671024b1ef01bd2.jpg)  
Figure F1. A residual failure of fixed support. (a) The edited endpoint preserves an occluding bus-stop sign; blue shows $M _ { 0 } ,$ and the red box defines the crop. (b) Identical image-coordinate crops at all eight original frame indices. Red borders mark the sign’s reappearance in Partial and random release; attention-ranked and adaptive release avoid it in this example. The release variants are exploratory. All rows use Wan2.2-Turbo, purple-bus case 90001 and seed 22001, as in Fig. 4; frame intervals are unequal.

Fixed support cannot follow moving content. Figure F1 shows an endpoint-preserved sign reappearing under Partial after disappearing earlier in the video.

Release rules. We explore shrinking support over Wan2.2’s denoising timesteps 1000, 960, 889, 727. Adaptive release starts from the pooled, fractional $M _ { 0 }$ weights w on the 22 × 40 token grid. At step k, we compute the channelwise Euclidean distance between the endpoint latent and the clean prediction of every latent slot except the first and last, average across these slots, and pool spatially by $2 \times 2$ to obtain $G _ { k }$ . With $\tau _ { k }$ the mean of $G _ { k }$ over all tokens, we set $w _ { k + 1 } = w _ { k } \mathbf { 1 } [ G _ { k } < \tau _ { k } ]$ after each of the first three steps; released tokens never re-enter.

Random and attention-ranked release rank the tokens inside $M _ { 0 }$ randomly or by pooled FLUX.2 attention. At each step, they keep tokens in rank order, with their pooled weights, while the cumulative weight stays within the adaptive run’s total for that sample and step; tokens outside $M _ { 0 }$ are never constrained. The same w controls model-input blending, token timesteps, and clean-prediction projection.

Completion–discontinuity trade-off. On the 515-video Wan2.2 cohort, adaptive release yields TCR 38.45%, TC-Score 0.811, HC 4.89, and DD 76.3%, compared with fixed support’s 42.91%, 0.819, 5.76, and 76.7%. Attention-ranked and adaptive release avoid the illustrated sign reappearance, while random release does not. The aggregate adaptive result trades lower HC for lower TCR; this example does not establish a general advantage over fixed support.

## G Broader I2V Completion Comparisons

Table G1. Automatic evaluation results of I2V models on TC-Bench-I2V. FPS reports video-stage throughput.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Params</td><td colspan="8">TC-Bench-I2V ↑</td><td rowspan="2">FPS ↑</td></tr><tr><td colspan="2">Attribute</td><td colspan="2">Object</td><td colspan="2">Background</td><td colspan="2">Overall</td></tr><tr><td></td><td></td><td>TCR</td><td>TC-Score</td><td>TCR</td><td>TC-Score</td><td>TCR</td><td>TC-Score</td><td>TCR</td><td>TC-Score</td><td></td></tr><tr><td>Image-to-Video Generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wan2.2-Turbo Release&#x27;25.08</td><td>5B</td><td>31.60</td><td>0.7647</td><td>34.74</td><td>0.7842</td><td>38.67</td><td>0.8486</td><td>33.79</td><td>0.7841</td><td>2.270</td></tr><tr><td>Wan2.2 Base [36] Release&#x27;25.07</td><td>5B</td><td>32.00</td><td>0.7935</td><td>30.00</td><td>0.7794</td><td>30.67</td><td>0.8442</td><td>31.07</td><td>0.7957</td><td>0.182</td></tr><tr><td>SANA-Video [5] ICLR&#x27;26</td><td>2B</td><td>23.20</td><td>0.7729</td><td>22.11</td><td>0.7245</td><td>28.00</td><td>0.7590</td><td>23.50</td><td>0.7530</td><td>1.031</td></tr><tr><td>HunyuanVideo-1.5 [39] arXiv&#x27;25.11 (480p, 8-step)</td><td>8.3B</td><td>52.80</td><td>0.9076</td><td>25.26</td><td>0.8312</td><td>41.33</td><td>0.8556</td><td>40.97</td><td>0.8718</td><td>0.615</td></tr><tr><td>MfM [21] ICLR&#x27;26</td><td>8.1B</td><td>39.60</td><td>0.8449</td><td>24.74</td><td>0.7887</td><td>29.33</td><td>0.8365</td><td>32.62</td><td>0.8230</td><td>0.034</td></tr><tr><td>OmniWeaving [31] arXiv&#x27;26.03</td><td>8.3B</td><td>42.80</td><td>0.8653</td><td>35.79</td><td>0.8034</td><td>37.33</td><td>0.8305</td><td>39.42</td><td>0.8374</td><td>0.027</td></tr><tr><td>Image-to-Video Generation w/ Image Editing (Full)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FLUX.2 + Wan2.2-Turbo</td><td>14B</td><td>48.40</td><td>0.8466</td><td>31.58</td><td>0.8081</td><td>46.67</td><td>0.8655</td><td>41.94</td><td>0.8352</td><td>2.270</td></tr><tr><td>ChronoEdit + Wan2.2-Turbo</td><td>19B</td><td>40.80</td><td>0.8239</td><td>33.68</td><td>0.7891</td><td>54.67</td><td>0.8948</td><td>40.19</td><td>0.8214</td><td>2.266</td></tr><tr><td>Qwen-Image-Edit + Wan2.2-Turbo</td><td>25B</td><td>41.20</td><td>0.8096</td><td>30.53</td><td>0.7736</td><td>26.67</td><td>0.8357</td><td>35.15</td><td>0.8001</td><td>2.266</td></tr><tr><td>FLUX.2 + Wan2.2 Base</td><td>14B</td><td>54.80</td><td>0.8700</td><td>31.58</td><td>0.8034</td><td>48.00</td><td>0.8838</td><td>45.24</td><td>0.8475</td><td>0.186</td></tr><tr><td>FLUX.2 + SANA-Video</td><td>11B</td><td>44.40</td><td>0.8451</td><td>25.79</td><td>0.7200</td><td>32.00</td><td>0.8136</td><td>35.73</td><td>0.7944</td><td>1.029</td></tr><tr><td>FLUX.2 + HunyuanVideo-1.5</td><td>17.3B</td><td>58.40</td><td>0.9018</td><td>31.05</td><td>0.8069</td><td>46.67</td><td>0.8470</td><td>46.60</td><td>0.8588</td><td>0.565</td></tr><tr><td>FLUX.2 + MfM</td><td>17.1B</td><td>58.00</td><td>0.9344</td><td>25.79</td><td>0.8198</td><td>42.67</td><td>0.8983</td><td>43.88</td><td>0.8869</td><td>0.034</td></tr><tr><td>FLUX.2 + OmniWeaving</td><td>17.3B</td><td>51.60</td><td>0.8977</td><td>32.63</td><td>0.8200</td><td>42.67</td><td>0.8683</td><td>43.30</td><td>0.8647</td><td>0.029</td></tr></table>

We additionally compare ChronoEdit and Qwen-Image-Edit endpoints [40, 41] on the same 103 prompts and five seeds. Image-editing rows use Full conditioning. Tables C9 and C8 provide paired completion differences; Table 1 evaluates Partial.

Endpoint sources and parameter counts. The Params column reports generative-backbone parameter counts in billions, including the editor but excluding separate text/vision encoders and VAEs. FLUX.2 uses klein-base-9B and the final-frame prompt in Appendix A. ChronoEdit uses its 14B checkpoint, which formulates editing as video generation with the input and edited images as the first and last frames. We therefore supply the original transition prompt directly, without the final-frame template, and use the last of five generated frames as the endpoint. Thus both editor and prompt formulation change. Qwen-Image-Edit uses the 20B 2511 release<sup>3</sup> with the FLUX.2 final-frame prompt, so only the editor changes.

Native configurations and timing. MfM uses 97 frames at 1280 × 704 and OmniWeaving 81 at 720p. FPS is computed as the number of decoded frames divided by the median video-stage time, excluding endpoint generation. MfM includes batch-amortized startup; other methods use per-video timers.