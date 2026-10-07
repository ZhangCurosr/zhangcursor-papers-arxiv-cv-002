# KNOWING WHEN TO TRUST A PRIOR: RELIABILITY-GATED CUE FUSION FOR VIDEO GAZE PREDICTION

Lichen Zhu, Yueqian Lin, Yiheng Wang, Hai “Helen” Li, Yiran Chen

Duke University, Durham, NC, USA

## ABSTRACT

Video gaze prediction is led by gaze-trained models, yet gazefree priors carry signal those models have not absorbed, if one knows when to trust them. We propose FocusGate, a gated ensemble of gaze-free priors whose members may abstain. A per-frame gate reads three shape statistics of a defocus map and selects the frames on which the estimator is above chance on average, so rejected frames reduce to the base exactly, while midrank normalisation lets an all-zero prior abstain at zero parameters. Gated fusion is significantly positive on film, sports and web video, whereas unconditional fusion is harmful on sports and null on web. Added to four supervised predictors, the NTIRE 2026 champion among them, FocusGate improves all sixteen model–domain cells in shuffled AUC, fifteen significantly, one domain pre-registered and scored once, while adding only 1% to the champion’s latency. Alone, it surpasses TASED-Net and UNISAL in shuffled AUC on film with a 16-frame causal mean.

Index Terms— visual attention, saliency, cue fusion, reliability, selective prediction

## 1. INTRODUCTION

Filmmakers keep the narratively relevant region in focus, and eye-tracking studies confirm that blur contrast steers fixations [1, 2]. Modern video saliency is dominated by trained networks [3, 4, 5, 6, 7, 8], yet the explicit priors used to compose and interpret such systems have hardly changed: a centre Gaussian, faces [9], motion [10]. Focusness has served salient-object detection in static images [11], and blur identification has been injected into saliency models before [12], but always unconditionally. Reliability-aware fusion has precedents [13, 14], but whether such priors still add anything beside gaze-trained models, and on which frames, has not been asked.

A defocus map is a region cue whose reliability varies from frame to frame: it marks the subject in a shallow depthof-field shot but only texture noise in a deep-focus one. We find that, fused unconditionally, the map of a scratch-trained estimator lowers sAUC on every base we tested (Sec. 4.2). In this work we therefore treat trust as a per-frame decision. A reliability gate reads three shape statistics of the map and selects the frames on which the estimator is, on average, above chance. This removes the harm and, with a stronger estimator, yields a significant gain in every domain tested, while on rejected frames the prediction equals its base by construction, i.e., selective prediction applied to cue fusion.

Across seven cue-estimator pairs under one pipeline, we find that gating helps only where an estimator has an identifiable regime in which it collapses to chance, and not merely where it is weak. Two predictions follow, and both are confirmed. A stronger estimator of the same cue [15, 16, 17] shrinks the regime and turns the removed harm into a significant gain, and domain shift re-opens it, so that on sports the ungated gain flips to a significant harm while the gate stays positive (Sec. 4.3). We call the resulting system FocusGate, a gated ensemble of gaze-free priors (Fig. 1) in which each prior may abstain, either through its own output or through the gate.

Our contributions are as follows.

• FocusGate, a gated ensemble of gaze-free priors that improves the shuffled AUC of four supervised models, the NTIRE 2026 champion among them, in all sixteen model–domain cells (fifteen significantly, one domain pre-registered and scored once) while adding 1% to the champion’s latency, and that alone surpasses 2019– 2020 supervised models in shuffled AUC on film with a 16-frame causal mean (Table 1).

• A per-frame reliability gate for a defocus cue that equals its base on rejected frames, converts the scratch estimator’s harm on film to noise (−0.94<sup>†</sup> to +0.04) and flips frozen sports transfer from −1.40<sup>†</sup> to +0.68<sup>∗</sup> (Secs. 2.1, 4.2, 4.3), and midrank normalisation that keeps an abstaining prior inert at zero parameters (Sec. 4.1).

## 2. METHOD

FocusGate is sketched in Fig. 1. Given a frame, gaze-free priors are rank-summed into a base map B (Sec. 2.2). A defocus estimator produces an in-focus probability map D, and a gate $c _ { t }$ decides from three shape statistics of D whether D enters the sum (Sec. 2.1).

![](images/1211e78f379a24ee0932c32c98d69acd6f409903c48f7d51ef1fd4b1411476f5.jpg)  
Fig. 1. FocusGate. (a) Eight gaze-free priors (fitted weights $w _ { k } )$ are rank-summed into the base B. The defocus map D enters only through the gate $c _ { t } ,$ , which thresholds three shape statistics of D and never sees the fixations. (b) The gate opens on a bimodal $D$ and closes on a unimodal one, where the prediction reduces to B exactly.

Two estimators of one cue. Both predict a per-pixel infocus probability $P$ and are trained on CUHK blur detection [24] with degradations matched to DivX-era footage, motion blur counting as out-of-focus [25]. The scratch estimator is a plain U-Net (2.76M parameters) with pixel AUROC 0.926/0.921 on clean/degraded held-out images. The recipe estimator adds the three ingredients of modern defocus-blur detectors: an ImageNet-pretrained encoder (MobileNetV3- L), train-time depth distillation [19, 15, 16], and a highfrequency input channel [17], reaching AUROC 0.958/0.953 (15 ms, Table 1).

## 2.1. Shallow-DoF reliability gate

From P we compute spread $( p _ { 9 0 } - p _ { 1 0 } )$ , separation of the two modes (mean of P>0.5 minus mean of $P { \leq } 0 . 5 )$ and the sharparea fraction. A frame is accepted $\scriptstyle ( c _ { t } = 1 )$ when spread $\geq$ $0 . 5 5 , \mathrm { s e p } \geq 0 . 4 5$ and $3 \% \leq \mathrm { s h a r p } \leq 7 0 \%$ . The rule reads only the map’s shape, and Sec. 4.3 shows that the gain it selects lives in the map’s spatial content. Thresholds were fixed once on CUHK, before any gaze data was used, and never re-tuned.

## 2.2. Gated rank-fusion

Let r(·) denote per-frame rank normalisation to [0, 1], tied values sharing their midrank, which makes heterogeneous cues scale-free [26]. Cues are ranked individually and the weighted sum is never ranked again, so for a base B and

defocus map $D ,$

$$
\begin{array} { r } { S = B + c _ { t } w _ { d } r ( D ) , \qquad B = \sum _ { k } w _ { k } r ( c _ { k } ) ^ { \gamma _ { k } } . } \end{array}\tag{1}
$$

The reference base sums a centre Gaussian, a face prior [18] and spectral residual (SR) saliency [27]. When ${ { c } _ { t } } \mathrm { { = } } 0 .$ , Eq. (1) is $B ,$ so off-gate behaviour (frames with ${ c _ { t } } \mathrm { { = } } 0 .$ , the rest ongate) is identical by construction. The full ensemble runs Eq. (1) over the eight priors of Fig. 1, where PQFT follows [28] and the ranked DINO attention is blurred with $\sigma { = } 3 . 5$ and re-ranked. The weights $w _ { k }$ are those shown in Fig. 1, $w _ { d } = 0 . 2 5 , \gamma _ { k } = 1 2$ for face, 3 for DINO and 1 elsewhere, and SR and Depth-Anything-V2 (DAv2) [29] were swept to zero, thirteen scalars fitted by coordinate ascent on train sAUC. A supervised model M enters as $r ( M ) + w ^ { * } r ( S )$ with trainfrozen $w ^ { * } = 0 . 4 , 0 . 3 , 0 . 1 5$ and 0.2 for TASED-Net, UNISAL, ViNet-S and ViSAGE. The temporal variant replaces the DINO, flow and defocus maps by the rank of their causal mean over eleven lags spanning 16 frames, refits the scalars and re-admits DAv2 (+63 ms), while the gate still reads the current frame’s D.

## 3. EXPERIMENTAL SETUP

Data. We use Hollywood-2 with the Actions-in-the-Eye recordings [30], 6.66M fixations rebuilt from the raw 500 Hz streams, on the standard 823/884 split, scoring the last frame of up to two clips per video. Transfer uses UCF-Sports [30] and DHF1K [31], every parameter frozen before scoring. DIEM [10] is a confirmatory fourth domain, acquired only after every parameter was frozen, pre-registered (the

Table 1. sAUC (%) of every predictor under one protocol on held-out video (Sec. 3). ms/frame per component on one A100 (batch 1, one forward pass per predicted frame, two for ViSAGE’s expert pair). A “+ FocusGate” row adds 111 ms to its model. Bold with <sup>∗</sup>: paired 95% cluster CI vs. the model’s own row excludes zero and survives a Holm correction over the sixteen cells. <sup>‡</sup>: confirmatory domain, pre-registered and scored once (the ViNet-S cell was added afterwards and is exploratory). These cells measure the ensemble, not the gate, whose pre-specified contrast is null (Sec. 4.2).
<table><tr><td>predictor</td><td>year</td><td>ms/frame</td><td>Film (H2)</td><td>Sports (UCF)</td><td>Web (DHF1K)</td><td>DIEM</td></tr><tr><td>centre prior (chance calibration)</td><td></td><td>&lt;0.1</td><td>49.9</td><td>51.4</td><td>50.6</td><td></td></tr><tr><td>flow residual, camera-comp.</td><td></td><td>19.8</td><td>63.7</td><td>69.9</td><td>60.2</td><td></td></tr><tr><td>face detector [18]</td><td>2023</td><td>12.1</td><td>65.3</td><td>52.9</td><td>53.9</td><td></td></tr><tr><td>depth, MiDaS [19]</td><td>2022</td><td>11.1</td><td>61.5</td><td>63.2</td><td>58.9</td><td></td></tr><tr><td>DINOv2 attention [20, 21]</td><td>2024</td><td>51.5</td><td>71.0</td><td>74.6</td><td>66.9</td><td></td></tr><tr><td>defocus, recipe estimator</td><td>ours</td><td>15.4</td><td>57.6</td><td>50.7</td><td>54.8</td><td></td></tr><tr><td>FocusGate, uniform weights</td><td>ours</td><td>174</td><td>73.8</td><td>77.0</td><td>67.9</td><td>72.2</td></tr><tr><td>FocusGate, single frame</td><td>ours</td><td>111</td><td>77.8</td><td>79.3</td><td>71.3</td><td>75.5</td></tr><tr><td>FocusGate, +causal context</td><td>ours</td><td>174</td><td>80.0</td><td>78.3</td><td>72.7</td><td>76.8</td></tr><tr><td>TASED-Net [3] (21M)</td><td>2019</td><td>50</td><td>78.4</td><td>77.0</td><td>72.6</td><td>74.0</td></tr><tr><td rowspan="2">+ FocusGate UNISAL [4] (3.7M)</td><td rowspan="2">2020</td><td rowspan="2">172</td><td>79.9*</td><td>79.8*</td><td>73.6*</td><td>76.6*</td></tr><tr><td>77.8</td><td>80.5</td><td>69.7</td><td>73.9</td></tr><tr><td>+ FocusGate</td><td></td><td></td><td>79.0*</td><td>81.3*</td><td>71.2*</td><td>75.4*</td></tr><tr><td rowspan="2">ViNet-S [22] (9.5M) + FocusGate</td><td rowspan="2">2025</td><td rowspan="2">69</td><td>80.7</td><td>81.2</td><td>73.5</td><td>77.3</td></tr><tr><td>81.4*</td><td>81.8*</td><td>74.0*</td><td>78.2*</td></tr><tr><td>ViSAGE [23] (6.9B)</td><td>2026</td><td>9,916</td><td>81.3</td><td>80.5</td><td>75.8</td><td>78.4</td></tr><tr><td>+ FocusGate</td><td></td><td></td><td>81.9*</td><td>81.7*</td><td>75.9</td><td>79.4*</td></tr></table>

Hollywood-2 protocol, in a timestamped file of the released repository) and scored once (Table 1, <sup>‡</sup> column).

Leakage-free subsets. TASED-Net, UNISAL and ViNet-S saw these datasets’ training splits, so every row is scored on held-out video only, while ViSAGE, trained on NTIRE 2026 mouse-tracking data, runs zero-shot. The four baselines span 2019–2026 and 3.7M–6.9B parameters. The held-out sets are the Hollywood-2 test split (1,715 frames, 867 videos after 17 undecodable target frames), the UCF-Sports testing directory (993 frames, 47 videos, every third frame up to 40 per video), DHF1K 601–700 (1,175 frames, 100 videos, twelve evenly spaced frames per video minus 25 without a fixation at scoring resolution), and DIEM (168 frames, 84). TASED-Net and ViNet-S consume a 32-frame clip, UNISAL a recurrent state warmed over up to 96 frames. Our TASED-Net reproduction on the DHF1K validation split gives sAUC 71.4 and AUC-J 91.0 (published test-set values 71.2 and 89.5, full-frame protocol), and under that same all-frame protocol (60,273 frames) frozen single-frame FocusGate scores 72.9, +1.5 [+0.2, +2.7] over TASED-Net. Code and every table’s script will be released.

Metrics. We report shuffled AUC (sAUC, tables in %, deltas in points) [26] with negatives drawn from the fixation distribution of other videos, so a predictor exploiting centre bias alone scores chance (Table 1, row 1). Density scores such as NSS track a calibration step rather than the cues (Sec. 4.3), so sAUC is the one metric reported. Scores are per-video means averaged over videos, marginalised with their intervals over twenty negative-sampling seeds, and every interval is a

10,000-resample percentile bootstrap clustered on videos. <sup>∗</sup>/<sup>†</sup> mark 95% CIs excluding zero (gain/harm). The pre-specified primary contrast is gated vs. base for the scratch estimator on the reference base (null, Sec. 4.2), and the paired gatedminus-ungated delta is the gate’s direct measure. Later arms were each specified, train-tuned and scored once, with unadjusted intervals.

## 4. RESULTS

## 4.1. Comparison with prior work

Table 1 puts every predictor on one protocol and on held-out video only, against TASED-Net [3], UNISAL [4], ViNet-S [22] and the NTIRE 2026 champion ViSAGE [23]. Adding the rank-normalised ensemble to each supervised prediction, with one train-frozen weight per model, improves sAUC in all sixteen model–domain cells, and fifteen gains survive the Holm correction. ViSAGE + FocusGate reaches 81.9 on film and 79.4 on DIEM, the best numbers in the table, and all four DIEM cells are significant. We read this as complementarity rather than strength, since the ensemble alone trails ViNet-S and ViSAGE on every domain yet still lifts both. The exception, ViSAGE on web, is within noise (+0.03 [−0.24, +0.34]).

Gaze enters FocusGate only through its thirteen fusion scalars. Even so, with a 16-frame causal mean the ensemble surpasses TASED-Net and UNISAL in sAUC on film and on the untouched DIEM domain, and its single-frame form scores 79.3 on sports against ViSAGE’s 80.5 (−1.2 [−2.6, +0.2]). It runs at 111 ms per frame, against 69 ms for ViNet-S and 9,916 ms for ViSAGE’s two forward passes of a 6.9B backbone with two decoders, so adding it to the champion costs 1% in latency. Inside the ensemble the defocus prior adds little once the other seven priors are present, and the gate acts as a downside bound, since ungated harm grows with its weight $( - 0 . 2 2 ^ { \dagger }$ at w=2 on sports) while the gated term never leaves noise $( - 0 . 0 5 [ - 0 . 1 4 , + 0 . 0 5 ] )$ ).

Table 2. The gate on seven cue-estimator pairs (H2 test split, reference base centre+face+SR, w=0.5). Shaded: the defocus cue that FocusGate gates, with the recipe estimator also transferred frozen. Unshaded: the same rule on other cues (motion: spread $\geq 0 . 1 .$ , moving $\mathrm { f r a c t i o n } \leq 0 . 3 5 ,$ train-selected). rate: accept rate. on/off: accepted/rejected frames.
<table><tr><td rowspan="2">cue, estimator</td><td rowspan="2"></td><td colspan="2">cue sAUC (%)</td><td rowspan="2"></td><td colspan="2"> $\Delta { \mathrm { \ v s . } }$  base</td></tr><tr><td>rate alone</td><td>on off</td><td>ungated</td><td>gated</td></tr><tr><td>defocus, classical</td><td>0%</td><td>45.6</td><td></td><td>45.6</td><td> $- 1 . 8 8 ^ { \dagger }$ </td><td> $+ 0 . 0 0$ </td></tr><tr><td>defocus, scratch</td><td>38%</td><td>52.2</td><td>58.5</td><td>48.0</td><td> $- 0 . 9 4 ^ { \dagger }$ </td><td>+0.04</td></tr><tr><td>defocus, recipe</td><td>40%</td><td>57.6</td><td>62.9</td><td>53.9</td><td> $+ 0 . 3 6 ^ { * }$ </td><td> $+ 0 . 6 0 ^ { * }$ </td></tr><tr><td>frozen, sports</td><td>38%</td><td>50.7</td><td>55.7</td><td>46.9</td><td>–1.40†</td><td>+0.68*</td></tr><tr><td>frozen, web</td><td>55%</td><td>54.8</td><td>57.5</td><td>51.4</td><td>-0.08</td><td>+0.57*</td></tr><tr><td>sharpness, Laplacian</td><td>17%</td><td>60.1</td><td>57.7</td><td>60.4</td><td>-0.67†</td><td>-0.18†</td></tr><tr><td>motion, frame diff.</td><td>56%</td><td>62.1</td><td>59.1</td><td></td><td> $6 5 . 7 ~ + 0 . 6 8 ^ { \ast }$ </td><td> $+ 0 . 0 3$ </td></tr><tr><td>depth, MiDaS</td><td>100%</td><td>61.5</td><td>61.</td><td></td><td> $5 \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \{ \cdot  ^ { } $ </td><td> $+ 1 . 9 5 ^ { * }$ </td></tr><tr><td>PQFT</td><td>0.3%</td><td>66.8</td><td></td><td></td><td>66.9 +0.54*</td><td> $+ 0 . 0 0$ </td></tr></table>

![](images/5100777fe08a9a0f0a054e109e0d7dae1b39e9ca5b83c86b5206fec3acc2b00a.jpg)  
Fig. 2. The collapse regime. Standalone sAUC of each cue on the 1,715 test frames by quintile of the spread of its own map (shaded: 95% CI). Defocus maps sit at chance when unimodal and rise when bimodal, while sharpness and motion never approach chance.

Abstention. One prior signals failure without any gate. The face detector returns nothing on 33%/74%/69% of film/sports/web frames, and its prior is then a constant map, i.e., an abstention. Tie-splitting rank normalisation destroys it by fabricating a full-contrast pattern at the cue’s weight, whereas shared-midrank ranking keeps a constant map constant, so the cue self-abstains. This costs zero parameters and is worth +0.6 points at frozen weights, five times the explicit face gate it replaces $( + 0 . 1 2 \ [ + 0 . 0 4 , + 0 . 2 1 ] )$ .

## 4.2. When does the gate help?

With the scratch estimator, unconditional fusion harms every base composition we tested and the gate removes the harm on all of them. On the reference base (centre+face+SR, 69.0) the pair is −0.94<sup>†</sup> ungated against $+ 0 . 0 4 \left[ - 0 . 1 2 , + 0 . 2 1 \right]$ gated (Table 2), i.e., a significant loss converted to a null, with a paired gated-minus-ungated contrast o $\mathrm { ~ f + 0 . 9 9 ~ [ + 0 . 7 4 , + 1 . 2 4 ] }$

Table 2 then applies the same rule, motion excepted (see caption), to seven cue-estimator pairs, and gating does not reward cue strength. A Laplacian sharpness map outpredicts the scratch defocus map standalone (60.1 vs. 52.2) yet harms under either policy, and for motion and PQFT, both stronger still, the gate only discards signal. What separates the rows is the off-gate column, and Fig. 2 shows why. Binned by the spread of their own map, the defocus estimators fall to chance where the map is unimodal and rise above 60 where it is bimodal, whereas sharpness and motion never approach chance, so the rule has nothing to select. We conclude that the rule is a defocus-reliability test and the other rows its negative control, where it is inert or harmful.

A better estimator shrinks the collapse regime but does not remove the need for the gate (Table 2, row 3). Its off-gate $\mathrm { s A U C }$ rises from 48.0 to 53.9, ungated fusion becomes safe $( + 0 . 3 6 ^ { * } )$ and gating still adds $( + 0 . 6 0 ^ { * } )$ , within noise on film $( + 0 . 2 3 \ : [ - 0 . 0 2 , + 0 . 4 9 ] )$ but decisive under transfer.

## 4.3. Frozen transfer and controls

The two frozen rows of Table 2 transfer the recipe estimator and the reference base to sports and web video with every parameter frozen. Domain shift re-opens the collapse regime and the gate finds it. On sports, where the recipe map alone is barely above chance (50.7), ungated fusion flips to a significant harm $( - 1 . 4 0 ^ { \dagger } )$ while gated fusion stays significantly positive $( + 0 . 6 8 ^ { * } )$ , and on web the ungated term is null $_ { ( - 0 . 0 8 ) }$ while the gated one is again positive $( + 0 . 5 7 ^ { * } )$ , so the thresholds transfer without re-tuning.

Controls. Five checks on the H2 test split rule out alternatives. A random gate at the same accept rate trails the real gate by 0.35 to 0.56 points over twenty draws, so the gain is not an artefact of down-weighting the cue. Permuting the map’s pixels on accepted frames turns the gain into harm $( + 1 . 5 4 ^ { * } \mathrm { \ v s . \ } - 1 . 2 7 ^ { \dagger } )$ , so it lives in the $\mathrm { { \ m a p ^ { \circ } s } }$ spatial content rather than in the frame selection. A soft gate performs identically to the hard one $( - 0 . 0 1 \ [ - 0 . 0 5 , + 0 . 0 4 ] )$ , and the gain holds in both extreme motion quartiles $( + 0 . 3 3 ^ { * } / + 0 . 8 9 ^ { * } )$ and across fifteen recalibrations of the map $\mathrm { ( + 0 . 5 2 \ t o \ + 0 . 6 4 }$ , all significant). Density calibration lifts film NSS from 1.81 to 2.69 at unchanged AUC-Judd, so the priors carry ordering, not density.

## 5. CONCLUSION

We proposed FocusGate, a gated ensemble of gaze-free priors in which each prior may abstain. Its gate reads three shape statistics of the defocus map and selects the frames on which the estimator is above chance on average. This removes the harm of unconditional fusion and, with a stronger estimator, yields a significant gain in every domain tested. Added to four supervised predictors, the NTIRE 2026 champion among them, the ensemble improves every model’s sAUC in every domain, a pre-registered one included. The gate is specific to defocus, the only cue tested with a collapse regime, and future work will look for reliability statistics of other cues.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This retrospective study uses only public eye-tracking data, namely Hollywood-2 with the Actions-in-the-Eye recordings and UCF-Sports [30], DHF1K [31] and DIEM [10] (CC BY-NC-SA 3.0). No new human or animal data was collected and no ethical approval was required.

## 7. REFERENCES

[1] Tim J. Smith, “The attentional theory of cinematic continuity,” Projections, vol. 6, no. 1, pp. 1–27, 2012.

[2] Tingting Zhang, Ling Xia, Xiaofeng Liu, and Xiaoli Wu, “Eye movements during change detection: the role of depth of field,” Cognitive Computation and Systems, vol. 1, no. 2, pp. 55–59, 2019.

[3] Kyle Min and Jason J. Corso, “TASED-Net: temporallyaggregating spatial encoder-decoder network for video saliency detection,” in Proc. IEEE ICCV, 2019, pp. 2394– 2403.

[4] Richard Droste et al., “Unified image and video saliency modeling,” in Proc. ECCV, 2020, pp. 419–435.

[5] Morteza Moradi, Mohammad Moradi, Francesco Rundo, Concetto Spampinato, Ali Borji, and Simone Palazzo, “SalFoM: dynamic saliency prediction with video foundation models,” in Proc. ICPR, 2024, pp. 33–48.

[6] Junwen Xiong, Peng Zhang, Tao You, Chuanyue Li, Wei Huang, and Yufei Zha, “DiffSal: joint audio and video learning for diffusion saliency prediction,” in Proc. IEEE CVPR, 2024.

[7] Andrey Moskalenko, Alexey Bryncev, Dmitry Vatolin, et al., “AIM 2024 challenge on video saliency prediction: methods and results,” in Proc. ECCV Workshops, 2024.

[8] Andrey Moskalenko, Alexey Bryncev, Ivan Kosmynin, et al., “NTIRE 2026 challenge on video saliency prediction: methods and results,” in Proc. IEEE/CVF CVPR Workshops, 2026.

[9] Melissa L.-H. Võ et al., “Do the eyes really have it? Dynamic allocation of attention when viewing moving faces,” Journal of Vision, vol. 12, no. 13, pp. 3–3, 2012.

[10] Parag K. Mital et al., “Clustering of gaze during dynamic scene viewing is predicted by motion,” Cognitive Computation, vol. 3, no. 1, pp. 5–24, 2011.

[11] Shaojie Zhuo and Terence Sim, “Defocus map estimation from a single image,” Pattern Recognition, vol. 44, no. 9, pp. 1852– 1858, 2011.

[12] Yoann Baveye, Fabrice Urban, and Christel Chamaret, “Image and video saliency models improvement by blur identification,” in Proc. ICCVG, 2012.

[13] Tariq Alshawi, Zhiling Long, and Ghassan AlRegib, “Unsupervised uncertainty estimation using spatiotemporal cues in video saliency detection,” IEEE Trans. Image Process., vol. 27, no. 6, pp. 2818–2827, 2018.

[14] Runmin Cong, Jianjun Lei, Changqing Zhang, Qingming Huang, Xiaochun Cao, and Chunping Hou, “Saliency detection for stereoscopic images based on depth confidence analysis and multiple cues fusion,” IEEE Signal Process. Lett., vol. 23, no. 6, pp. 819–823, 2016.

[15] Xiaodong Cun and Chi-Man Pun, “Defocus blur detection via depth distillation,” in Proc. ECCV, 2020, pp. 747–763.

[16] Yuxin Jin, Ming Qian, Jincheng Xiong, Nan Xue, and Gui-Song Xia, “Depth and DOF cues make a better defocus blur detector,” in Proc. IEEE ICME, 2023.

[17] Weihuang Liu, Xi Shen, Chi-Man Pun, and Xiaodong Cun, “Explicit visual prompting for low-level structure segmentations,” in Proc. IEEE CVPR, 2023.

[18] Wei Wu, Hanyang Peng, and Shiqi Yu, “YuNet: a tiny millisecond-level face detector,” Machine Intelligence Research, vol. 20, no. 5, pp. 656–665, 2023.

[19] René Ranftl et al., “Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer,” IEEE Trans. PAMI, vol. 44, no. 3, 2022.

[20] Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al., “DINOv2: Learning robust visual features without supervision,” Trans. Mach. Learn. Res., 2024.

[21] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski, “Vision transformers need registers,” in Proc. ICLR, 2024.

[22] Rohit Girmaji, Siddharth Jain, Bhav Beri, Sarthak Bansal, and Vineet Gandhi, “Minimalistic video saliency prediction via efficient decoder & spatio temporal action cues,” in Proc. IEEE ICASSP, 2025, pp. 1–5.

[23] Kun Wang, Yupeng Hu, Zhiran Li, Hao Liu, Qianlong Xiang, and Liqiang Nie, “ViSAGE @ NTIRE 2026 challenge on video saliency prediction,” in Proc. IEEE/CVF CVPR Workshops, 2026.

[24] Jianping Shi et al., “Discriminative blur detection features,” in Proc. IEEE CVPR, 2014, pp. 2965–2972.

[25] Beomseok Kim et al., “Defocus and motion blur detection with deep contextual features,” Computer Graphics Forum, vol. 37, no. 7, pp. 277–288, 2018.

[26] Zoya Bylinskii, Tilke Judd, Aude Oliva, Antonio Torralba, and Frédo Durand, “What do different evaluation metrics tell us about saliency models?,” IEEE Trans. PAMI, vol. 41, no. 3, pp. 740–757, 2019.

[27] Xiaodi Hou and Liqing Zhang, “Saliency detection: a spectral residual approach,” in Proc. IEEE CVPR, 2007, pp. 1–8.

[28] Chenlei Guo, Qi Ma, and Liming Zhang, “Spatio-temporal saliency detection using phase spectrum of quaternion Fourier transform,” in Proc. IEEE CVPR, 2008.

[29] Lihe Yang et al., “Depth Anything V2,” in Proc. NeurIPS, 2024.

[30] Stefan Mathe and Cristian Sminchisescu, “Actions in the eye: dynamic gaze datasets and learnt saliency models for visual recognition,” IEEE Trans. PAMI, vol. 37, no. 7, pp. 1408– 1424, 2015.

[31] Wenguan Wang et al., “Revisiting video saliency: a large-scale benchmark and a new model,” in Proc. IEEE CVPR, 2018, pp. 4894–4903.

## Supplementary Material

Every value below is produced by src/gen\_supplement.py from the result files that also feed Tables 1 and 2, under the protocol of Sec. 3 (per-video means, twenty negative-sampling seeds, 10,000-resample percentile bootstraps clustered on videos). Nothing here was tuned; the supplement reports intervals, controls and negative results that the four-page limit left out. A. All-frame protocol on DHF1K 601–700

Sec. 3 quotes one number from this run. Both predictors were scored on every annotated frame of the 100 validation videos (60,273 frames), with the DHF1K benchmark’s shuffled-AUC mechanics (negatives from the other 99 videos’ fixations) and AUC-Judd at the 640×360 ground-truth resolution. The gate accepted 54.9% of all frames (55% under the sparse protocol of Table 2).
<table><tr><td>predictor</td><td>sAUC (%)</td><td>AUC-J (%)</td><td>paired ∆ sAUC vs. TASED-Net</td></tr><tr><td>TASED-Net (sliding 32-frame window)</td><td>71.4</td><td>91.0</td><td></td></tr><tr><td>FocusGate, single frame, frozen</td><td>72.9</td><td>84.0</td><td>+1.5 [+0.2, +2.7]* (n = 100 videos)</td></tr></table>

FocusGate ranks fixations better than TASED-Net under this protocol while scoring lower on AUC-Judd, which rewards density and centre mass. The same ordering-versus-density split appears in Sec. E.

## B. The sixteen “+ FocusGate” cells of Table 1 with intervals

Paired per-video deltas of model + FocusGate over the model alone, 95% cluster-bootstrap intervals, the two-sided bootstrap p (floored at 1/B) and the Holm threshold it is compared against. Fifteen of sixteen cells survive the correction; the ViNet-S DIEM cell was added after the DIEM pre-registration.
<table><tr><td>domain</td><td>model</td><td>∆ sAUC [95% CI]</td><td>p</td><td>Holm</td></tr><tr><td>Film (H2)</td><td>TASED-Net</td><td>+1.59 [+1.27, +1.92]*</td><td>0.0001</td><td>yes (≤ 0.0031)</td></tr><tr><td>Film (H2)</td><td>UNISAL</td><td>+1.26 [+1.04, +1.49]*</td><td>0.0001</td><td>yes (≤ 0.0036)</td></tr><tr><td>Film (H2)</td><td>ViNet-S</td><td>+0.64[+0.48, +0.79]*</td><td>0.0001</td><td>yes (≤ 0.0038)</td></tr><tr><td>Film (H2)</td><td>ViSAGE</td><td>+0.62 [+0.42, +0.83]*</td><td>0.0001</td><td>yes (≤ 0.0033)</td></tr><tr><td>Sports (UCF)</td><td>TASED-Net</td><td>+2.77[+2.21, +3.33]*</td><td>0.0001</td><td>yes (≤ 0.0042)</td></tr><tr><td>Sports (UCF)</td><td>UNISAL</td><td>+0.84[+0.47, +1.23]*</td><td>0.0001</td><td>yes (≤ 0.0050)</td></tr><tr><td>Sports (UCF)</td><td>ViNet-S</td><td>+0.65 [+0.45, +0.85]*</td><td>0.0001</td><td>yes (≤ 0.0056)</td></tr><tr><td>Sports (UCF)</td><td>ViSAGE</td><td>+1.18 [+0.81, +1.57]*</td><td>0.0001</td><td>yes (≤ 0.0045)</td></tr><tr><td>Web (DHF1K)</td><td>TASED-Net</td><td>+0.98 [+0.49, +1.50]*</td><td>0.0002</td><td>yes (≤ 0.0125)</td></tr><tr><td>Web (DHF1K)</td><td>UNISAL</td><td>+1.49 [+1.06, +1.96]*</td><td>0.0001</td><td>yes (≤ 0.0063)</td></tr><tr><td>Web (DHF1K)</td><td>ViNet-S</td><td> $+ 0 . 4 8 \ : [ + 0 . 2 6 , + 0 . 7 2 ] ^ { * }$ </td><td>0.0001</td><td>yes (≤ 0.0071)</td></tr><tr><td>Web (DHF1K)</td><td>ViSAGE</td><td>+0.03 [-0.24, +0.34]</td><td>0.8356</td><td>no (≤ 0.0500)</td></tr><tr><td>DIEM</td><td>TASED-Net</td><td>+2.61 [+1.45, +3.92]*</td><td>0.0001</td><td>yes (≤ 0.0083)</td></tr><tr><td>DIEM</td><td>UNISAL</td><td> $+ 1 . 5 5 [ + 0 . 9 7 , + 2 . 2 1 ] ^ { * }$ </td><td>0.0001</td><td>yes (≤ 0.0100)</td></tr><tr><td>DIEM</td><td>ViNet-S</td><td> $+ 0 . 9 2 \ [ + 0 . 3 9 , + 1 . 4 7 ] ^ { * }$ </td><td>0.0008</td><td>yes (≤ 0.0250)</td></tr><tr><td>DIEM</td><td>ViSAGE</td><td> $+ 0 . 9 2 \ [ + 0 . 4 0 , + 1 . 4 9 ] ^ { * }$ </td><td>0.0004</td><td>yes (≤ 0.0167)</td></tr></table>

## C. Table 2 with intervals

The same rows as Table 2, with the 95% intervals of the two deltas. Reference base centre+face+SR, fusion weight w=0.5, H2 test split; the two frozen rows transfer the recipe estimator with every parameter fixed.
<table><tr><td>cue, estimator</td><td>rate</td><td>ungated ∆ [95% CI]</td><td>gated ∆ [95% CI]</td></tr><tr><td>defocus, classical</td><td>0%</td><td> $- 1 . 8 8 [ - 2 . 1 6 , - 1 . 6 1 ] ^ { \dagger }$ </td><td> $+ 0 . 0 0 \left[ + 0 . 0 0 , + 0 . 0 0 \right]$ </td></tr><tr><td>defocus, scratch</td><td>38%</td><td>−0.94 [−1.25, −0.64]†</td><td> $+ 0 . 0 4 \left[ - 0 . 1 2 , + 0 . 2 1 \right]$ </td></tr><tr><td>defocus, recipe</td><td>40%</td><td> $+ 0 . 3 6 \left[ + 0 . 0 5 , + 0 . 6 7 \right] ^ { \ast }$ </td><td> $+ 0 . 6 0 \left[ + 0 . 4 4 , + 0 . 7 6 \right] ^ { * }$ </td></tr><tr><td>frozen, sports</td><td>38%</td><td>-1.40 [−2.70, −0.13]†</td><td> $+ 0 . 6 8 \left[ + 0 . 0 7 , + 1 . 3 6 \right] ^ { * }$ </td></tr><tr><td>frozen, web</td><td>55%</td><td>-0.08 [-0.75, +0.60]</td><td>+0.57[+0.18, +1.01]*</td></tr><tr><td>sharpness, Laplacian</td><td>17%</td><td>−0.67 [−0.86, −0.48]†</td><td>-0.18 [−0.26, −0.11]†</td></tr><tr><td>motion, frame diff.</td><td>56%</td><td>+0.68 [+0.45, +0.91]*</td><td>+0.03 [-0.13, +0.20]</td></tr><tr><td>depth, MiDaS</td><td>100%</td><td>+1.95 [+1.69, +2.23]*</td><td>+1.95 [+1.69, +2.23]*</td></tr><tr><td>PQFT</td><td>0.3%</td><td> $+ 0 . 5 4 \ : [ + 0 . 3 4 , + 0 . 7 4 ] ^ { * }$ </td><td> $+ 0 . 0 0 \left[ - 0 . 0 0 , + 0 . 0 1 \right]$ </td></tr></table>

## D. Gate controls (H2 test split, recipe estimator unless stated)

<table><tr><td>control</td><td>value</td></tr><tr><td>paired gated — ungated, recipe</td><td>+0.23 [−0.02, +0.49]</td></tr><tr><td>paired gated — ungated, scratch</td><td>+0.99[+0.74, +1.24]*</td></tr><tr><td>random gate at the same accept rate (39.7%), 20 draws, gated ∆ vs. base</td><td>+0.04 to +0.25 (real gate +0.60)</td></tr><tr><td>real gate — random gate, mean / min over draws</td><td>+0.45 /+0.35</td></tr><tr><td>accepted frames (n = 465 videos): real map vs. base</td><td>+1.54[+1.12, +1.96]*</td></tr><tr><td>accepted frames: pixel-permuted map vs. base</td><td>-1.27[-1.59, -0.96]†</td></tr><tr><td>accepted frames: real — permuted soft gate (logistic) – hard gate</td><td>+2.81[+2.33, +3.31]*</td></tr><tr><td>gated ∆, lowest / highest motion quartile</td><td>-0.01 [-0.05, +0.04]</td></tr><tr><td>accept rate by motion quartile (low to high)</td><td>+0.33 [+0.09, +0.58]* / +0.89 [+0.41, +1.37]*</td></tr><tr><td>gated ∆ over 15 affine recalibrations aP + b of the map</td><td>30% / 36% / 42% / 52%</td></tr><tr><td>gate decision agreement, adjacent frames / 16 frames apart</td><td>+0.52 to +0.64, all lower bounds &gt; 0</td></tr><tr><td></td><td>92.8% / 82.3%</td></tr></table>

The accept rate rises with motion, yet the gated gain is significant in both extreme quartiles, so the gate is not a motion detector in disguise. The recalibrations rescale and shift P before the three statistics are computed, with the thresholds left untouched.

## E. Density-sensitive metrics on the film test split

sAUC is the paper’s metric because a centre-bias predictor scores chance on it. The table shows what the other common metrics say for the same predictions: FocusGate’s raw rank-sum map carries ordering but little density, a single train-fitted monotone calibration of its values lifts NSS, CC and SIM without touching AUC-Judd (a ranking metric), and the supervised models remain far ahead on density.

<table><tr><td>predictor</td><td>NSS</td><td>CC</td><td>SIM</td><td>AUC-J (%)</td></tr><tr><td>FocusGate, raw rank sum</td><td>1.81</td><td>0.292</td><td>0.108</td><td>87.4</td></tr><tr><td>FocusGate, density-calibrated</td><td>2.69</td><td>0.409</td><td>0.234</td><td>87.4</td></tr><tr><td>TASED-Net</td><td>3.59</td><td>0.540</td><td>0.368</td><td>93.2</td></tr><tr><td>UNISAL</td><td>3.51</td><td>0.523</td><td>0.358</td><td>92.5</td></tr><tr><td>ViNet-S</td><td>3.87</td><td>0.590</td><td>0.390</td><td>94.2</td></tr></table>

## F. Variants that did not help (fitted on train, frozen sets scored once)

Each variant was refitted on Hollywood-2 train with the Sec. 2.2 procedure and, where it passed on train, scored once on the frozen sets against the released ensemble. None was adopted.

<table><tr><td>variant</td><td>train sAUC</td><td>film ∆</td><td>sports ∆</td><td>web ∆</td></tr><tr><td>released ensemble (DINOv2-registers base)</td><td>78.40</td><td></td><td></td><td></td></tr><tr><td>DINOv2-registers large as DINO cue</td><td>77.73</td><td>-0.50 [−0.73, −0.28]†</td><td>-0.61 [−1.14, −0.03]†</td><td>-0.52 [−1.03, −0.01]†</td></tr><tr><td>flip test-time averaging of DINO cue</td><td>78.43</td><td>+0.01 [−0.02, +0.04]</td><td>-0.06 [-0.13, -0.00]</td><td>-0.02 [−0.07, +0.03]</td></tr></table>

Also rejected on train alone: per-cue causal temporal pooling for every cue (gain 0 over the released ensemble), uncertaintyweighted fusion with per-frame entropy (the fitted weight of the uncertainty term is 0), and a DINO-internal failure detector (the best internal statistic predicts the frames on which dropping the DINO term would help with ROC-AUC 0.56 on train, below the 0.6 needed to fit a rule). V-JEPA 2 attention and feature maps were texture-like and were not fitted as a cue.

## G. Fitted scalars

Coordinate ascent on train sAUC over the grids listed in Sec. H; weights multiply the rank-normalised cue after its exponent;   
the centre Gaussian has weight 1 and is the reference. Cues with weight 0 were swept and rejected.

<table><tr><td>cue</td><td>single frame</td><td>+causal context</td></tr><tr><td>face</td><td>8</td><td>8</td></tr><tr><td>spectral residual</td><td>0</td><td>0</td></tr><tr><td>PQFT</td><td>4</td><td>3</td></tr><tr><td>frame-diff motion</td><td>0.5</td><td>0</td></tr><tr><td>flow residual</td><td>4</td><td>5</td></tr><tr><td>MiDaS depth</td><td>5</td><td>3</td></tr><tr><td>Depth-Anything-V2</td><td>0</td><td>0.25</td></tr><tr><td>DINOv2-registers attention</td><td>8</td><td>8</td></tr><tr><td>DINO local deviation</td><td>3</td><td>2</td></tr><tr><td>gated defocus (wd)</td><td>0.25</td><td>1</td></tr><tr><td>face exponent / DINO exponent</td><td>12/ 3</td><td>12/4</td></tr><tr><td>DINO blur σ (rank-map pixels)</td><td>3.5</td><td>3</td></tr><tr><td>causal lags (frames)</td><td></td><td>0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16</td></tr><tr><td>train sAUC</td><td>78.40</td><td>80.41</td></tr></table>

Supervised fusion weight $w ^ { * }$ (one per model, train-frozen): TASED-Net 0.4, UNISAL 0.3, ViNet-S 0.15, ViSAGE 0.2.

## H. Implementation details

Settings as they stand in the released scripts (file names in parentheses). Where no fixed random seed was used, the table says so.

<table><tr><td>item</td><td>setting</td></tr><tr><td colspan="2">Defocus estimators (train_dbd.py, train_dbd_v2.py)</td></tr><tr><td>data and split</td><td>CUHK blur-detection images with ground truth (defocus and motion-blur images, any blur counts as not in focus); file stems sorted, shuffled once with a fixed seed (42), the first fifth held out for validation, the rest for training; the same split for both estimators. No image from any gaze</td></tr><tr><td>input degradation (training only, per image)</td><td>dataset. 320 ×320 RGB in [0, 1]; target 1—mask, white = blurred. downscale by a factor in U(0.25, 0.9) and upscale back (p=0.7); JPEG at quality U(25, 90)  $( p { = } 0 . 7 ) ;$  contrast  $U ( 0 . 7 , 1 . 3 )$  and brightness U(−20, 20) (p=0.5); gamma U(0.7, 1.4) (p=0.3);</td></tr><tr><td>scratch estimator</td><td>Gaussian noise with  $\sigma \sim U ( 2 , 8 ) ( p { = } 0 . 3 )$  . Validation images are undegraded; the “degraded' AUROC applies the same recipe with a fixed seed. U-Net, 24/48/96/192/192 channels, four max-pool levels, 2.76M parameters; loss BCE + 0.5 Dice; AdamW, lr  $3 \times 1 0 ^ { - 4 }$  , weight decay  $\mathrm { \dot { 1 } 0 ^ { - 4 } }$  , batch 32, 40 epochs, cosine schedule;</td></tr><tr><td>recipe estimator</td><td>checkpoint with the best validation AUROC. MobileNetV3-Large encoder (ImageNet weights), first convolution inflated to four input channels with the new channel zero-initialised; fourth channel =  $3 { \times } 3$  Laplacian magnitude of the grey</td></tr><tr><td>seeds</td><td>image, max-normalised per image; auxiliary head regresses MiDaS-small depth precomputed on CUHK (train time only, L1 weight 0.3); loss boundary-weighted BCE + 0.5 Dice + 0.3 depth L1; AdamW, lr  $\cdot 5 \times 1 0 ^ { - 5 }$  (encoder) and  $3 \times 1 0 ^ { - 4 }$  (rest), batch 24, 40 epochs, cosine; 4.64M parameters. no fixed training seed (single run each); the validation loader and the degraded-validation pass use</td></tr><tr><td>Gate (focusgate/dbd.py)</td><td>seed 0.</td></tr><tr><td>statistics thresholds</td><td>on the 96×56 in-focus probability map P: spread = p90 - p10; sep 二 mean(P&gt;0.5)-mean  $( P { \le } 0 . 5 )$  ; sharp fraction = share of  $P { > } 0 . 5 ,$  spread ≥ 0.55, sep ≥ 0.45, 0.03 ≤ sharp ≤ 0.70; constants set by hand on CUHK val-</td></tr><tr><td></td><td>idation maps before any gaze data was used, never changed afterwards; the gated gain is flat over spread  $\in \ \{ 0 . 4 5 , \ldots , 0 . 6 5 \} \times \ \mathrm { s e p } \in \ \{ 0 . 3 5 , \ldots , 0 . 5 5 \}$  with the sharp bounds fixed  $( \mathtt { o r a l \_ h a r d e n i n g . p y } ) .$ </td></tr><tr><td colspan="2">Cues (diem_ext ract_recs . py is the reference implementation of the full per-frame recipe) every frame resized to 384×224; every cue map and fixation mask at 96×56 (masks by nearest-</td></tr><tr><td>frame geometry</td><td>neighbour downsampling of the fixation image).</td></tr><tr><td>centre</td><td>anisotropic Gaussian,  $\sigma _ { x } = 0 . 2 8 W , \sigma _ { y } = 0 . 2 4 H ,$  hard-coded, not tuned on any gaze split (focusgate/priors.py).</td></tr><tr><td>face</td><td>YuNet via OpenCV FaceDetectorYN, long side scaled to 320 px, score threshold 0.6, NMS 0.3; each box filled with its score on the 96 ×56 grid, 5×5 Gaussian blur; midrank; all-zero map when nothing is detected.</td></tr><tr><td>spectral residual</td><td>Hou and Zhang at 192 px working width (width selected on the train split), 3×3 box filter on the log spectrum.</td></tr><tr><td>PQFT motion</td><td>quaternion phase spectrum at 128×64 with frame t—3 as the motion channel. grey frames blurred 5×5; max of the absolute differences  $| g _ { t } - g _ { t - 1 } | , | g _ { t - 1 } - g _ { t - 2 } | ;$  max- normalised.</td></tr><tr><td>flow residual</td><td>Farneback flow (0.5, 3, 21, 3, 5, 1.2) between t—2 and t at 384×216; median flow vector sub- tracted (camera motion); residual magnitude, Gaussian σ=5 (mot res_ext ract . py).</td></tr><tr><td>depth</td><td>MiDaS-small (torch hub, small transform), bicubic to 96×56; Depth-Anything-V2-Small (swept to weight 0).</td></tr><tr><td>DINO</td><td>facebook/dinov2-with-registers-base, input 448×784, last-layerCLS-to-patch at- tention averaged over the 12 heads, resized to 96×56, min-max normalised, rank; the rank map is blurred with σ=3.5 pixels and re-ranked before the exponent (dino_next gen.py). Lo- cal deviation (DEV): L2 norm of each last-layer patch token minus the frame-mean token, rank (explore_dinoagg_frozen_extract.py).</td></tr><tr><td>defocus map D</td><td>recipe estimator at 320×320, output resized to 96×56 (v2_finalize. py).</td></tr><tr><td colspan="2">Fusionfit (stack_v8_fit.py, stack_v9_fit.py, sota_plus_final.py) objective and data</td></tr><tr><td></td><td>mean per-video sAUC on the Hollywood-2 train split only (823 videos, last frame of up to two clips each), twenty negative seeds.</td></tr><tr><td>search</td><td>coordinate ascent, one scalar at a time, repeated until no scalar moves: weights ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8}; exponents ∈ {1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24}; DINO blur σ ∈ {0, 2, 2.5, 3, 3.5, 4, 5, 6, 8}; DEV weight ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6}.</td></tr><tr><td>supervised fusion temporal variant</td><td> $r ( M ) { + } w ^ { * } r ( S )$  with  $w ^ { * } \in \{ 0 . 0 5 , 0 . 1 , 0 . 1 5 , 0 . 2 , 0 . 3 , 0 . 4 \}$  chosen on train per model, then frozen. DINO, flow-residual and defocus maps replaced by the rank of their causal mean over lags</td></tr><tr><td colspan="2">{0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16}; scalars refitted with the same grids.</td></tr><tr><td>sAUC</td><td>Scoring (face_gate.py, focusgate/metrics.py) negatives drawn from the pooled fixation counts of all other videos in the set (the fixation distri-</td></tr><tr><td>intervals</td><td>bution of other videos), twenty seeds, per-video mean, then the mean over videos. paired per-video deltas, 10,000 percentile-bootstrap resamples over videos, seed 0; Holm correc-</td></tr><tr><td>timing</td><td>tion over the sixteen Table-1 cells. one A100, batch 1, one forward pass per predicted frame, fp32 for the cues and the two 2019–</td></tr><tr><td></td><td>2020 models, bf16 autocast for ViSAGE.</td></tr></table>

## I. Abstention under the two rank normalisations

![](images/24d000567b27caaecce7a5b90f44dd8ac9b31a842cef38bfbd9c66c6cf820d6e.jpg)  
Fig. S1. The face prior on two frames. Top: a detected face. Bottom: the detector returns nothing, so the raw prior is all zeros Splitting ties fabricates a full-contrast pattern (standard deviation 0.29 across the map) that enters the sum at the cue’s weight; sharing ties (midrank) gives the constant 0.5 and the fused term stays inert.