# Multimodal Concert Video Editing: System Design and Exploratory Evaluation

Lennart Schiweck, Adrian Kempf, Michelle Retig, Moritz Feucht, Tim Mangliers and Benjamin Himpel

Department of Informatics, Reutlingen University 72762 Reutlingen, Germany Corresponding author: Tim Mangliers Tim.Mangliers@Reutlingen-University.de

## Abstract

Editing multi-camera concerts requires coordinated decisions about musical relevance, visual quality and cut timing. We present a modular framework that combines instrument activity from multitrack audio, visible instruments and image sharpness with beat-based timing and probabilistic camera selection. An interpretable scoring model links these components to the choice of views over complete shot intervals. Component tests and an exploratory expert study on a four-camera concert yield mixed assessments across the editing variants. The findings motivate extending local audiovisual relevance with contextual and sequence-level criteria. The architecture provides a basis for testing these extensions and studying how component behaviour afects the quality of complete edits.

Keywords: automatic video editing; concert recordings; multimodal analysis; camera selection;   
interpretable scoring.

## 1 Introduction

Multi-camera concert editing requires deciding when to cut and which view to show. Editors coordinate musical activity, image quality, framing and the progression of a sequence. A clear view of an active instrument can be useful, but so can a reaction or an interaction elsewhere in the ensemble. Automating this workflow therefore requires more than ranking isolated images.

Automatic camera management has been studied in lecture environments [9], concert mashups [15] and social-camera footage [1]. Concert-directing systems combine auditory and visual instrument information [17]; learned approaches address next-view selection and cut prediction [6, 8]. Temporal and candidate-view context are also modelled in multi-camera editing [2, 12], while theatre systems combine camera selection and shot-size control [16].

We present a modular framework that separates audio analysis, visual analysis, camera scoring and temporal selection. Named audio stems provide instrument activity and roles; visual detections describe instrument visibility, and image-gradient statistics characterise sharpness. An explicit utility combines these signals, aggregates them over a prospective shot interval and controls a probabilistic camera selector. Beat and tempo estimates determine candidate cut boundaries independently of view selection.

The contributions are threefold: a transparent multimodal architecture for concert editing; an explicit mapping from role and quality cues to interval-level camera choices; and a component and expert evaluation that motivates targeted architectural extensions. The study compares a human reference, random camera selection and two score-driven variants. Its mixed results inform further work on shot suitability, contextual reactions and sequence structure without treating component accuracy as a substitute for complete-edit quality.

## 2 Related work

Wang et al. [17] provide a close comparison to the explicit-score design: their system already combines audible–visible instrument matching, blur and shot-scale information. Table 1 compares the audio inputs, scoring, timing and selection mechanisms.

Table 1: Concert-directing comparison with Wang et al. (2023). The table compares system designs rather than performance on a common dataset.
<table><tr><td>Axis</td><td>Wang et al. (2023)</td><td>Our system</td></tr><tr><td>Audio</td><td>Instrument classification from mixed audio.</td><td>Named stems; relative energy for primary/secondary roles.</td></tr><tr><td>ing</td><td>Instrument match- Audio-visual instrument agreement contributes when permitted by shot scale.</td><td>Explicit audible-visible role bonuses and missing/inaudible-label penalties.</td></tr><tr><td>Image criteria</td><td>Blur and shot-scale information.</td><td>Fixed gradient blur thresholds and box-area roles; no framing optimisation.</td></tr><tr><td></td><td>Temporal handling Downbeat information and an estimated switching rate.</td><td>Every-fourth-beat grid and tempo-based target duration; conservative full-interval scoring.</td></tr><tr><td>Cut positions</td><td>Downbeat-based timing.</td><td>Next grid point after a tempo-dependent target duration.</td></tr><tr><td>Camera selection</td><td>Highest eligible score under shot-scale transition rules.</td><td>Proportional sampling with no immediate repeat and uniform fallback. Four full edits, three raters, random and</td></tr><tr><td>Evaluation</td><td>Automatic/manual clips rated by 16 participants.</td><td>annotation-driven diagnostics, original comments.</td></tr></table>

Shrestha et al. [15] combine perceptual image quality, content diversity and cut-point suitability in concert mashups. Their 40-participant evaluation already compares automatic, manual and random edits, so our baseline follows an established strategy. Daniyal and Cavallaro [3] model finite-horizon camera scheduling with content quality and switching costs; our no-repeat sampler does not optimise a sequence-level objective. Stoll et al. [16] combine rule-based camera selection and cropping for four-perspective theatre recording, including audiovisual speaker assignment. Their comparison of four-perspective and single-perspective edited versions holds cut times, shot lengths and shot sizes fixed. Our system selects existing views without generating tighter framing.

Learned concert-editing approaches are also direct comparators. Jiménez et al. [8] learn visual next-shot selection from professionally edited concert videos. Their human study assesses static image choices. Learned next-view selection is therefore an established approach. Gonzálbez-Biosca et al. [6] separate cut-time prediction from view selection: audio and temporal features inform the former, and learned visual representations inform the latter. Their classical-concert evaluation uses pseudo-labelled candidates derived from existing edits and reports cut-prediction and retrieval measures. These are diferent outcomes from perceived quality of a complete realised edit.

Rao et al. [12] introduce the TVMCE benchmark with synchronised multi-camera material including concerts and model both past frames and candidate views. Cabacas-Maso et al. [2] further distinguish history and candidate encoding through a dual-transformer architecture with cross-attention. Such approaches explicitly represent context beyond our no-repeat sampler, but their reference-decision agreement measures do not by themselves establish perceptual quality of our concert edits.

Adjacent editorial approaches make broader objectives explicit. Galvane et al. [4] optimise shot sequences in 3D animation using action, continuity and rhythm costs, with subjective comparisons of degraded cost variants. EditIQ [5] combines dialogue interpretation, saliency and speaker cues with cinematic constraints for virtual views from wide-angle recordings; its user study includes emotions and reactions. EmotionCast [18] additionally studies afective cues in stage-program directing. These works motivate testing editorial context and sequence-level criteria, while their diferent materials and designs prevent transferring their performance claims to this pilot. The concert evaluation motivates testing these ideas for instrumental performance and ensemble interaction.

The emphasis of our framework is an explicit, modular connection between instrument roles, quality cues, shot timing and camera choice. Component measurements and complete-edit assessment provide complementary views of this design.

## 3 System overview and material

The framework takes four synchronised camera streams and concert audio as input. Audio processing has two complementary branches: named instrument stems supply activity and audible roles, while their mono mix supplies beat and tempo estimates. The visual branch uses a pretrained D-FINE object detector with retained instrument categories from Objects365 [11, 14] and imagegradient statistics for sharpness. These signals feed the camera utility, while the timing branch defines intervals over which views are evaluated. Figure 1 gives the complete processing flow.

![](images/42bc716d1cca17cf15242f575c0440f5058d81290d459a9ff8ee63122184a19c.jpg)  
Figure 1: Modular concert-editing architecture. Named stems supply activity analysis and are mixed into a mono waveform for timing. Visual roles and sharpness combine with audible roles in a camera utility; shot boundaries define the aggregation intervals. Camera selection produces the edit. Reference annotations form an alternative label input for the annotation-driven evaluation variant.

The evaluation uses a recording of the Uli Gutscher Quartet performing at JAZZclub IN DER MITTE e.V. in Reutlingen on 11 January 2025. The camera arrangement comprises an audience-oriented view, two close-up views and a wide view (Table 2). Component measurements cover the first 480 s; the expert study uses four complete edits of approximately 7 min 48 s. The annotated audio vocabulary comprises Bass, Drums, Guitar, Piano, Speech and Trombone. Labels describe audible and visible roles, blur and, for the annotation-driven variant, additional pan and clapping cues.

Table 2: Camera mapping and blur reference supports. n is the number of one-second evaluation samples used in the blur evaluation. Supports are ordered as not blurred / partially blurred / fully blurred.
<table><tr><td>Camera</td><td>View</td><td>n</td><td>Class supports</td><td></td></tr><tr><td>CAM1</td><td>Audience-oriented view</td><td>481</td><td></td><td> $7 / 4 7 4 / 0$ </td></tr><tr><td>CAM2</td><td>Close-up view</td><td>480</td><td></td><td>54 / 386 / 40</td></tr><tr><td>CAM3</td><td>Close-up view</td><td>481</td><td></td><td>4 / 453 / 24</td></tr><tr><td>CAM4</td><td>Wide view</td><td>481</td><td></td><td> $4 8 1 / \mathrm { ~ 0 ~ / ~ 0 ~ }$ </td></tr></table>

The prediction-driven variant combines estimated audio roles, visible roles and blur. The annotation-driven variant replaces these inputs with reference labels and also includes pan/clapping cues. It therefore tests annotation-driven camera scoring, rather than a detector-only substitution. Detailed algorithms and parameters are given in Appendix A.

## 4 Camera scoring and selection

The system translates audible and visible instrument roles into per-second camera utilities, combines them with image quality and aggregates them over each candidate shot. The selection module samples one camera per interval while preventing immediate camera repetition.

## 4.1 Local utility

At integer second t, audible primary/secondary labels are matched against visible primary/secondary labels in each camera. Let $R _ { c } ( t )$ sum these matches, with bonuses 4, 1, 1 and 0.5 for primary–primary, primary–secondary, secondary–primary and secondary–secondary matches. The penalty is

$$
P _ { c } ( t ) = 0 . 9 | A _ { p } ( t ) \setminus V _ { c } ( t ) | + 0 . 5 | A _ { s } ( t ) \setminus V _ { c } ( t ) | + 0 . 7 5 | V _ { c } ( t ) \setminus A ( t ) | ,\tag{1}
$$

where $A _ { p } , A _ { s }$ are audible-role sets and $A , V _ { c }$ are the audible and camera-visible unions. The score is

$$
S _ { c } ( t ) = \mathrm { r o u n d } _ { 2 } \big ( [ 4 + R _ { c } ( t ) - P _ { c } ( t ) ] Q _ { c } ( t ) \big ) .\tag{2}
$$

The quality multiplier $Q _ { c }$ is zero for a fully blurred or recognised pan label and otherwise one;   
partial blur is not penalised. Pan cues are used by the annotation-driven variant.

## 4.2 Timing, interval utility and selection

The timing module places boundaries on every fourth tracked beat and samples target durations of 10–15, 6–10 or 3–6 s for slow, middle or fast tempo classes. This assumes four beats per bar and does not establish true downbeats or musical phrasing.

For a prospective interval, scores are aggregated conservatively. If $S _ { c , ( 1 ) } \leq \cdots \leq S _ { c , ( n ) }$ are its integer-bin scores,

$$
q _ { c } = \operatorname* { m a x } \left( 0 , \frac { 1 } { m } \sum _ { j = 1 } ^ { m } S _ { c , ( j ) } \right) , \qquad m = \operatorname* { m a x } ( 1 , \lceil 0 . 1 0 n \rceil ) .\tag{3}
$$

For at most ten bins this is the minimum clipped at zero. Given previous camera i, proportional sampling selects $j \neq i$ with

$$
p _ { j } = \frac { q _ { j } } { \sum _ { k \neq i } q _ { k } } , \qquad p _ { i } = 0 ,\tag{4}
$$

when the denominator is positive; otherwise the fallback is uniform over the other three cameras. A zero score is therefore not a hard veto. This is a no-repeat constraint rather than a sequence

model of editorial grammar. Full-interval scoring and centred audio analysis use future context;   
the present implementation operates ofline.

## 5 Component performance

Audio-role estimation reaches support-weighted F1 scores of 0.654 for primary and 0.601 for secondary roles. Performance varies across instruments, and blur classification is strongly cameradependent. These diferences afect how the corresponding cues can guide camera choice. Appendix B reports the per-class audio and blur results, sampling definitions, role timelines and camera-score plots.

## 6 Expert evaluation

## 6.1 Study design

Four edits of the same recording are compared: V1 is a student-produced human reference under expert supervision; V2 uses random camera selection with beat-based timing; V3 uses predicted camera scores; and V4 uses annotation-driven scores. Three experts from diferent media fields assessed the complete versions without being told the editing method.

The questionnaire requests an overall publication/broadcast-readiness grade and detail ratings for cut timing, transitions, picture quality/sharpness and coherence/aesthetic impression, together with free-text comments. Overall grades use 1 = very good and 5 = poor. Conflicting direction anchors make the detail scales ambiguous, so only the overall grades are interpreted directionally; the detail responses are reported without reversal in Appendix E. Appendix D describes the questionnaire dimensions.

## 6.2 Cut timing and shot durations

The four edits use the same audio. V1 contains 35 internal camera changes; V2–V4 each contain 65. Corresponding automatic cut boundaries difer by at most one frame: V3 and V4 coincide, while V2 is one frame earlier. Thus the automatic variants have efectively the same pacing, allowing their view choices to be compared at similar cut timing.

The clip comprises an opening announcement, music and final applause. We use approximate windows of 0–80 s, 80–460 s and 460–467.7 s. Table 3 reports the duration of displayed segments over the full clip and the music window. Shots crossing a window boundary are clipped for the summary, without counting an additional camera change. Appendix C gives the definitions and additional phase summaries.

V1 begins with an uninterrupted 116.72-s shot, giving longer and more variable durations than the automatic edits. In the music window its mean is 10.86 s, compared with 6.79 s for V2–V4. The automatic versions also make nine cuts during the announcement, whereas V1 makes none. These statistics describe pacing; musical and editorial fit are assessed separately in the expert feedback.

## 6.3 Overall grades

The assessments are mixed across experts and variants. V1 receives consistent overall grades, while V2 and V4 show greater disagreement. All three experts grade V3 less favourably than V2. V4 improves on V3 for two experts and ties it for the third, but does not receive a better grade than V2. In this recording, score-driven selection therefore provides no overall advantage over the random baseline. The individual grades remain important: equal grades do not necessarily express equal preference, as E3 explicitly prefers V2 despite giving V2 and V4 the same grade.

Table 3: Shot-duration summaries from the complete rated videos and the approximately 380-s music window (80–460 s). Durations are in seconds. Cuts are internal camera changes in the window; n is the number of positive displayed segments, including clipped boundary segments. SD is the sample SD, not uncertainty across concerts. V2–V4 are listed separately, not pooled.
<table><tr><td>Variant</td><td>Window</td><td>Cuts</td><td>n</td><td>Mean</td><td>Median</td><td>SD</td></tr><tr><td>V1</td><td>Full</td><td>35</td><td>36</td><td>12.99</td><td>6.94</td><td>19.70</td></tr><tr><td>V2</td><td>Full</td><td>65</td><td>66</td><td>7.09</td><td>6.72</td><td>2.27</td></tr><tr><td>V3</td><td>Full</td><td>65</td><td>66</td><td>7.09</td><td>6.72</td><td>2.27</td></tr><tr><td>V4</td><td>Full</td><td>65</td><td>66</td><td>7.09</td><td>6.72</td><td>2.27</td></tr><tr><td>V1</td><td>Music</td><td>34</td><td>35</td><td>10.86</td><td>7.14</td><td>9.68</td></tr><tr><td>V2</td><td>Music</td><td>55</td><td>56</td><td>6.79</td><td>6.67</td><td>2.18</td></tr><tr><td>V3</td><td>Music</td><td>55</td><td>56</td><td>6.79</td><td>6.67</td><td>2.18</td></tr><tr><td>V4</td><td>Music</td><td>55</td><td>56</td><td>6.79</td><td>6.67</td><td>2.18</td></tr></table>

Table 4: Complete overall grades, 1 = very good and 5 = poor. Mean, median and sample SD summarise three ordinal responses descriptively; they do not estimate uncertainty across concerts.
<table><tr><td>Variant</td><td>E1</td><td>E2</td><td>E3</td><td>Mean</td><td>Median</td><td>SD</td></tr><tr><td>V1: student reference</td><td>3</td><td>3</td><td>3</td><td>3.00</td><td>3</td><td>0.00</td></tr><tr><td>V2: random</td><td>4</td><td>3</td><td>1</td><td>2.67</td><td>3</td><td>1.53</td></tr><tr><td>V3: predicted scores</td><td>5</td><td>4</td><td>3</td><td>4.00</td><td>4</td><td>1.00</td></tr><tr><td>V4: annotated scores</td><td>4</td><td>4</td><td>1</td><td>3.00</td><td>4</td><td>1.73</td></tr></table>

## 6.4 Written feedback

The free-text responses are grouped descriptively by editorial theme. Table 5 retains positive, critical and difering assessments rather than reducing them to a single consensus. The analysis concerns the experts’ expressed judgements; it does not assign an internal technical cause to each reported shot. Appendix F describes the grouping procedure.

The feedback highlights a distinction between technically usable images and editorially useful moments. It also values reactions and band interaction alongside the currently prominent instrument. Criticism of incidental audience views coexists with praise for motivated reactions, so the comments do not support a general audience-view prohibition.

The complete edits and an illustrative side-by-side excerpt accompany the study.<sup>1</sup>

## 7 Discussion

## 7.1 Relating system components to editorial assessment

The evaluation relates component-label agreement, temporal characteristics and expert assessment of the complete sequence. The automatic variants share efectively the same measured boundaries, while their overall grades and reported view choices difer. This narrows the interpretation towards camera selection, cue content and sequence behaviour; it does not identify a single causal component. V4’s improvement on V3 for two raters also cannot isolate recognition accuracy, because annotation source, available cues and potentially random draws change together.

Table 5: Themes in the written expert feedback. Locators identify expert, video and comment field.
<table><tr><td>Concern</td><td>Observation and counterpoint</td><td>Example locators</td></tr><tr><td>Musical action and view choice</td><td>E1 describes V3&#x27;s cut positions as matching V2 but its chosen views as worse, often missing the musician carrying the theme.</td><td>E1/V3: cut comment, remarks</td></tr><tr><td>Appropriate moments</td><td>Nose blowing and people standing or leaving attract criticism in several variants; this is not exclusive to V3.</td><td>E1/V2: cut comment; E1/V4: cut comment; E3/V3: impression comment</td></tr><tr><td>Framing and view diversity</td><td>E2 repeatedly requests better framing and close-ups; E3 criticises the detail-poor wide view. These comments do not prove that better raw alternatives were unavailable.</td><td>E2/V1–V4: remarks; E3/V1: sharpness comment</td></tr><tr><td>Interaction and reactions</td><td>E3 praises reactions, laughter and musical responses in V2. Such views can be valuable without showing only the leading instrument.</td><td>E3/V2: impression comment, remarks</td></tr><tr><td>Different pacing criteria</td><td>E1/E2 find V2&#x27;s pattern schematic or unmotivated, while E3 finds it coherent and dynamic.</td><td>E1/V2: cut, transition comments; E2/V2: cut comment; E3/V2: cut</td></tr><tr><td>Contextual audience value</td><td>E1/E2 criticise V3&#x27;s audience choices; E3 also values some audience reactions while rejecting unmotivated views.</td><td>comment E1/V3: remarks; E2/V3: cut comment; E3/V3: cut and impression comments</td></tr></table>

The feedback specifies what a useful view means to these raters: coverage of musical action, acceptable framing and moments, and motivated reactions or ensemble interaction. These criteria span diferent architectural interfaces. Table 6 links the reported concerns to operations that can influence selection and to changes that can be tested. The feedback motivates each link; it does not show that the named operation caused a particular shot.

## 7.2 Tracing component outputs into camera choices

An important example is the diference between measurement error and operational influence. CAM1’s partial/not-blurred disagreement produces poor classification metrics, but both states have $Q _ { c } = 1$ . Correcting only that disagreement therefore leaves its score unchanged. It cannot, by itself, explain audience-view selection. By contrast, audible-role changes enter both bonuses and penalties, and full-blur or recognised pan labels can zero a per-second score. Even then, the interval aggregate, previous camera and fallback determine whether a score change reaches a displayed view. Appendix G gives conditional score examples and controlled comparisons for studying these efects.

Table 6: Component-to-evaluation links and proposed architectural tests. Operational consequences follow from the scoring and selection rules; feedback locators identify reported outcomes. Proposed improvements have not been evaluated.
<table><tr><td>Component and score pathway</td><td>Expert assessment</td><td>Architectural change and isolated test</td></tr><tr><td>Audio roles: smoothed relative energy changes match bonuses and missing-label penalties.</td><td>E1/V3 remarks report missing musical action; primary-role F1 is 0.654.</td><td>Activity and musical relevance could be separated; smoothing or relevance input could vary independently, with score/choice changes logged.</td></tr><tr><td>Visible roles: box area supplies prominence, without framing, action suitability or reaction context.</td><td>E2/V1-V4 remarks criticise framing; E1/V4 cut comment reports an unsuitable moment; E3/V2 remarks praise reactions.</td><td>Shot eligibility could be distinguished from musical/contextual relevance, with each tested using independently annotated inputs before introducing a new detector.</td></tr><tr><td>Quality map: partial blur and not blurred both have multiplier one.</td><td>CAM1 has 474 partial-blur reference samples but is predicted not blurred; E1/V2 and E3/V3 sharpness</td><td>Correcting those labels alone has no score effect. A calibrated quality map could be tested separately from label accuracy.</td></tr><tr><td>Interval/policy: short intervals use their minimum score clipped at zero; zero utility is not a veto under fallback.</td><td>comments mention focus. E1/V3 cut comment reports poorer views at V2-like cut times; the internal reason is unobserved.</td><td>Eligibility, aggregation and sampling could be varied one at a time, with complete decision logs.</td></tr><tr><td>Timing: a beat/tempo heuristic also covers speech; automatic pacing is matched.</td><td>Nine announcement cuts; E2/V2 and V4 cut comments criticise speech timing; E3/V2 cut comment praises rhythm.</td><td>A speech/music/applause state could support timing tests separately from view selection.</td></tr><tr><td>Cue interface: V4 adds pan/clapping labels alongside reference roles.</td><td>V4 improves on V3 for two raters but does not outgrade V2.</td><td>Label sources could be compared under cue parity, with optional cues then tested as a separate factor.</td></tr></table>

The objective also omits some distinctions the comments make. A large, sharp view of a relevant instrument need not show a suitable action, and a valuable reaction need not depict the leading instrument. Adding a larger instrument bonus would not resolve that representational issue. At the same time, neither audience presence nor a non-leading musician is inherently unsuitable: E3’s positive reaction feedback argues for contextual assessment rather than a blanket penalty. Source coverage must be inspected separately, since selection cannot create an unavailable close-up.

## 7.3 Proposed controlled comparisons

A useful next version would expose separate interfaces for cue availability and reliability, candidate eligibility, musical/contextual relevance, and the sequence policy. Timing would receive an explicit speech/performance/applause state. This factorisation makes it possible to ask whether a recognised property is accurate, whether it changes the utility as intended, and whether the resulting views improve the relevant editorial judgement. It also makes disagreement informative: reaction value and pace preference can be reported per rater rather than hidden in one mean.

Comparison 1: separating suitability and relevance. At common decision opportunities and with unchanged remaining inputs, an initial comparison could add independently annotated eligibility information to the existing selector; a separate comparison could test contextual reaction value. Outcomes would include unsuitable-view exposure, musical-action coverage and paired sequence preference, with any changes in realised cuts caused by eligibility exceptions reported explicitly. This would first test the editorial criterion before introducing prediction error from a new detector. Action-aware and reaction-sensitive objectives have precedents in other settings [4, 5, 18]; their value for this concert remains to be measured.

Comparison 2: adding sequence context. A controlled comparison would hold local scores and timing fixed, retain the same no-repeat and fallback rules, and compare selection with and without one prespecified history/transition term, such as recently shown performers or view roles. An inspectable rule would avoid initially requiring a larger learned model. Temporalcontext systems [2, 12] and learned concert selection [6, 8] ofer later comparison models. Several prespecified realisations and blinded sequence judgements are needed; reference-decision agreement alone is insuficient.

Comparison 3: making quality and fallback operationally explicit. Quality could first be calibrated on development material with supported classes per camera, followed by independent tests of the quality-to-utility map and fallback policy. This comparison would distinguish a hard eligibility constraint from a soft preference. If only the current view is usable, holding it may be preferable to a mandatory switch; if no view is eligible, the interval could be flagged or an explicitly documented exception invoked. Such holds change realised cuts and dwell times, so common decision opportunities do not imply matched realised pacing. The pilot motivates these tests but establishes no improvement from them.

Phase-specific timing is a complementary extension. Speech/content boundaries should be assessed during announcements, and beat/downbeat or phrase cues during music. Its efect requires a separate timing comparison. The approximately 80-s announcement and the diferent rhythm judgements show why a full-clip grade and a music-window duration summary answer diferent questions. A consistent cue vocabulary is also necessary before changing predicted to reference labels; pan/clapping extensions must remain separate factors.

## 7.4 Study scope

The evaluation covers one concert, one camera arrangement, one realisation per automatic method and three experts. It therefore characterises this application rather than establishing a general ranking of editing strategies. Further concerts, repeated random realisations and a broader expert or audience study are needed to assess transfer across material and preferences.

## 8 Conclusion

The framework combines audio and visual analysis, explicit camera scoring, beat-based timing and probabilistic selection in a modular editing system. With efectively matched automatic timing, the prediction-driven edit receives a less favourable overall grade than the available random edit from all three raters. The annotation-driven edit improves on predictions for two raters without receiving a better grade than random selection. The result motivates further development of the selection objective and policy.

The component results and expert feedback support a research agenda that extends beyond detector accuracy. The findings motivate testing musical relevance, suitable moments, framing and motivated reactions as distinct factors; quality labels need a defined operational efect, and eligibility, aggregation and sequence context should be independently testable. The component-toevaluation mapping connects the scoring rules to concrete directions for controlled comparison. A follow-up can therefore investigate how changes propagate from labels to scores, selected views and complete-edit judgement, while retaining both favourable and critical observations.

## Declarations

Funding. This work was supported in part by the German Federal Ministry of Education and Research through the DATIpilot programme.

Acknowledgements. The authors thank Anja Hartmann and Boris Terpinc for their expertise in video editing, Jerg Bengel for supervising the reference edit, and Valmire Nivokazi, Sara Kemmler, Nina Floeter and Ali Senpinar for contributing to it. They thank Thorsten Klaes and Heiko Raiser for hardware and livestream support, and acknowledge the preparatory student work of Areej Jumaa, Celine Schmitt, Fadi Altobaji, Lennart Daumüller, Min Xu and Nina Kretschmer. The Uli Gutscher Quartet granted permission to use the concert recording.

AI assistance. AI tools assisted with manuscript editing, organisation, LaTeX and figure preparation, numerical calculations and descriptive grouping of written feedback. The authors are responsible for the scientific content and final submitted version.

## References

[1] Ido Arev, Hyun Soo Park, Yaser Sheikh, Jessica Hodgins, and Ariel Shamir. Automatic editing of footage from multiple social cameras. ACM Trans. Graph., 33(4), July 2014. ISSN 0730-0301. doi: 10.1145/2601097.2601198. URL https://doi.org/10.1145/2601097.2601198.

[2] Josep Cabacas-Maso, Carles Ventura, and Ismael Benito-Altamirano. A dual-transformer architecture with cross-attention for multi-camera view recommendation. In Computer Vision – ECCV 2026, volume 17059 of Lecture Notes in Computer Science, pages 460–475, Cham, 2026. Springer. doi: 10.1007/978-3-032-37086-0\_25. URL https://doi.org/10.1007/978-3-032-37086-0\_25.

[3] Fahad Daniyal and Andrea Cavallaro. Multi-camera scheduling for video production. In 2011 Conference for Visual Media Production, pages 11–20, 2011. doi: 10.1109/CVMP.2011.8.

[4] Quentin Galvane, Rémi Ronfard, Christophe Lino, and Marc Christie. Continuity editing for 3D animation. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pages 753–761. AAAI Press, 2015. doi: 10.1609/aaai.v29i1.9288.

[5] Rohit Girmaji, Bhav Beri, Ramanathan Subramanian, and Vineet Gandhi. EditIQ: Automated cinematic editing of static wide-angle videos via dialogue interpretation and saliency cues. In Proceedings of the 30th International Conference on Intelligent User Interfaces. Association for Computing Machinery, 2025. doi: 10.1145/3708359.3712113. URL https://arxiv.org/abs/2502.02172.

[6] Daniel Gonzálbez-Biosca, Josep Cabacas-Maso, Carles Ventura, and Ismael Benito-Altamirano. When and how to cut classical concerts? a multimodal automated video editing approach. In Proceedings of the 3rd International Workshop on Multimedia Content Generation and Evaluation: New Methods and Practice, McGE ’25, pages 98–106. Association for Computing Machinery, 2025. doi: 10.1145/3746278.3759387. URL https://doi.org/10.1145/3746278.3759387.

[7] Rafael C. Gonzalez and Richard E. Woods. Digital Image Processing. Pearson, 4 edition, 2018.

[8] Albert Jiménez, Lluís Gómez, and Joan Llobera. Automated video edition for synchronized mobile recordings of concerts. In Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, volume 4: VISAPP, pages 941–948. SCITEPRESS, 2022. doi: 10.5220/0010847600003124. URL https://doi.org/10.5220/0010847600003124.

[9] Qiong Liu, Yong Rui, Anoop Gupta, and J. J. Cadiz. Automating camera management for lecture room environments. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, CHI ’01, pages 442–449, New York, NY, USA, 2001. Association for Computing Machinery. ISBN 1581133278. doi: 10.1145/365024.365310. URL https://doi.org/10.1145/365024.365310.

[10] Brian McFee, Colin Rafel, Dawen Liang, Daniel P. W. Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python. In Proceedings of the 14th Python in Science Conference (SciPy 2015), pages 18–24, 2015.

[11] Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, and Feng Wu. D-FINE: Redefine regression task in DETRs as fine-grained distribution refinement. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 6cf58a87e3097e7d1f9be3e8693a93de-Paper-Conference.pdf.

[12] Anyi Rao, Xuekun Jiang, Sichen Wang, Yuwei Guo, Zihao Liu, Bo Dai, Long Pang, Xiaoyu Wu, Dahua Lin, and Libiao Jin. Temporal and contextual transformer for multi-camera editing of TV shows, 2022. URL https://arxiv.org/abs/2210.08737. arXiv preprint.

[13] R. Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal of Risk, 2(3): 21–41, 2000.

[14] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8429–8438, 2019. doi: 10.1109/ICCV.2019.00852.

[15] Prarthana Shrestha, Peter H.N. de With, Hans Weda, Mauro Barbieri, and Emile H.L. Aarts. Automatic mashup generation from multiple-camera concert recordings. In Proceedings of the 18th ACM International Conference on Multimedia, MM ’10, pages 541–550, New York, NY, USA, 2010. Association for Computing Machinery. ISBN 9781605589336. doi: 10.1145/1873951.1874023. URL https://doi.org/10.1145/1873951.1874023.

[16] Eckhard Stoll, Stephan Breide, Steve Göring, and Alexander Raake. Automatic camera selection, shot size, and video editing in theater multi-camera recordings. IEEE Access, 11:96673–96692, 2023. doi: 10.1109/ACCESS. 2023.3311256.

[17] Caihong Wang, Chuhan Qiu, Wanxin Xu, Weijie Zhu, Hanlu Luo, Lei Chen, Ziyi Yu, Lin Fu, and Xuanjun Chen. Intelligent directing system for music concert scene based on visual and auditory information. In Proceedings of the 2023 ACM International Conference on Interactive Media Experiences Workshops, IMXw ’23, pages 95–102. ACM, June 2023. doi: 10.1145/3604321.3604336.

[18] Xinyi Zhang, Xinran Ba, Feng Hu, and Jin Yuan. EmotionCast: An emotion-driven intelligent broadcasting system for dynamic camera switching. Sensors, 24(16):5401, 2024. doi: 10.3390/s24165401. URL https: //doi.org/10.3390/s24165401.

## A Implementation details

The following definitions specify the audio, visual and selection modules. The camera mapping and class supports are given in Table 2.

## A.1 Beat grid and target shot duration

The named audio stems are mixed to form the timing waveform, which is loaded as mono at its existing sampling rate using librosa [10]. If the WAV file is absent, the cutter extracts a 48 kHz stereo PCM waveform from CAM4 and then loads it as mono. The beat tracker returns times $B = \{ b _ { 0 } , b _ { 1 } , . . . \}$ . We construct an approximate bar grid

$$
\mathcal { D } = \{ b _ { 0 } , b _ { 4 } , b _ { 8 } , . . . \} .\tag{A.1}
$$

This assumes four beats per bar and uses the first detected beat as the phase reference. It does not estimate metre or identify true downbeats. The cutter contains no additional median filter on the beat sequence, and no beat- or bar-annotation accuracy is reported.

Local tempo is estimated over 30 s windows with a 10 s hop. At shot start t, the implementation uses the first overlapping tempo window in chronological order, with a middle-tempo fallback when no window covers t. Its tempo classes and target dwell-time ranges are

$$
\begin{array} { r } { \Delta \sim \mathcal { U } ( a _ { f } , b _ { f } ) , \qquad ( a _ { f } , b _ { f } ) = \left\{ \begin{array} { l l } { ( 1 0 , 1 5 ) \mathrm { ~ s } , \quad f < 9 0 \mathrm { ~ B P M } , } \\ { ( 6 , 1 0 ) \mathrm { ~ s } , \quad } & { 9 0 \leq f < 1 3 0 \mathrm { ~ B P M } , } \\ { ( 3 , 6 ) \mathrm { ~ s } , \quad } & { f \geq 1 3 0 \mathrm { ~ B P M } . } \end{array} \right. } \end{array}\tag{A.2}
$$

The next boundary is the first grid point at or after $t + \Delta$

$$
t ^ { \prime } = \operatorname* { m i n } \{ d \in { \mathcal { D } } : d \geq t + \Delta \} ,\tag{A.3}
$$

or the clip end if no such point remains. The clip end is limited by the audio duration and the largest score timestamp. Equation A.2 bounds the sampled target, not the realised shot duration, which can be longer after grid alignment. Beat alignment therefore provides a timing convention, rather than a demonstrated model of phrasing or dramaturgy.

## A.2 Activity and role estimation from audio stems

Separate stems permit direct energy analysis without identifying instruments from a polyphonic mixture. Stem identities are supplied by the track names. Each stem is loaded at its native sampling rate $f _ { s }$ and analysed with a centred 3 s RMS frame and a 1.5 s hop. For stem i and frame k,

$$
r _ { i } [ k ] = \sqrt { \frac { 1 } { N } \sum _ { n = 0 } ^ { N - 1 } x _ { i } [ k H + n - \lfloor N / 2 \rfloor ] ^ { 2 } } , \quad N = \mathrm { r o u n d } ( 3 f _ { s } ) , \quad H = \mathrm { r o u n d } ( 1 . 5 f _ { s } ) ,\tag{A.4}
$$

with zero padding at signal boundaries. A centred nine-frame moving average produces $\bar { r } _ { i } [ k ]$ . Its efective temporal support is approximately $3 + 8 \times 1 . 5 = 1 5 \mathrm { { s } }$ . This smoothing can obscure short entrances and contributes future context, so the role estimator is also an ofline procedure.

Normalisation uses shared bounds over all smoothed stem energies, ${ \cal L } = \mathrm { p e r c } _ { 1 } ( \{ \bar { r } _ { i } [ k ] \} _ { i , k } )$ and $U = \mathrm { p e r c } _ { 9 9 } ( \{ \bar { r } _ { i } [ k ] \} _ { i , k } )$ , to define the activity:

$$
v _ { i } [ k ] = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { \bar { r } _ { i } [ k ] - L } { U - L } \right) .\tag{A.5}
$$

If the bounds are invalid or $U \leq L$ , the implementation uses $L = 0 , U = 1$ . These are global bounds, not separate percentile scales for individual instruments. Relative stem levels therefore remain dependent on recording gain and microphone bleed.

With $v _ { \operatorname* { m a x } } [ k ] = \operatorname* { m a x } _ { i } v _ { i } [ k ]$ , active stems satisfy both an absolute and a relative threshold:

$$
\begin{array} { r } { A [ k ] = \{ i : v _ { i } [ k ] \geq 0 . 0 7 5 \mathrm { ~ a n d ~ } v _ { i } [ k ] \geq 0 . 3 5 v _ { \operatorname* { m a x } } [ k ] \} . } \end{array}\tag{A.6}
$$

For a nonempty active set, normalised energy shares are

$$
\alpha _ { i } [ k ] = \frac { v _ { i } [ k ] ^ { 2 } } { \sum _ { j \in A [ k ] } v _ { j } [ k ] ^ { 2 } } .\tag{A.7}
$$

A stem is primary if $\alpha _ { i } [ k ] \geq 0 . 9 2 \operatorname* { m a x } _ { j \in A [ k ] } \alpha _ { j } [ k ]$ ; other active stems are secondary. Multiple primary stems are possible. Consecutive equal roles are merged into intervals on the 1.5 s grid. The export maps announcement, trombone, guitar, bass-pickup and piano stems to their semantic labels, and maps both bass-drum and snare stems to Drums. Roles are assigned before this label mapping, so the exported primary and secondary sets can overlap for a grouped instrument. Labels are cropped to 480 s. These operational roles describe relative signal energy and are not equivalent to perceptual soloist identification.

## A.3 Visible instruments and image sharpness

Visual instrument detection uses the pretrained D-FINE-X Objects365 detector [11, 14], with confidence threshold 0.55. A detected instrument is visually primary when its box covers more than 20% of the frame area. The retained classes are Drums, Guitar, Piano, Cello and Trombone; Cello is mapped to Bass as an application-specific proxy. The detector is used without fine-tuning.

A detected bounding box $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ in a frame of width W and height $H _ { v }$ is visually primary if

$$
\frac { ( x _ { 2 } - x _ { 1 } ) ( y _ { 2 } - y _ { 1 } ) } { W H _ { v } } > 0 . 2 0 ;\tag{A.8}
$$

otherwise it is secondary. This is a size proxy; it does not evaluate occlusion, face visibility, shot aesthetics or whether the instrument is being played. The detector is not fine-tuned on the concert, and the visual timeline in Figure B.1 is an example rather than a quantitative detection-accuracy test.

For blur estimation, frames are sampled approximately every 0.2 s using a frame step of max(1, round(0.2 fps)). Grayscale frames at their input resolution are smoothed with a $5 \times 5$ Gaussian kernel (OpenCV’s automatic sigma setting). Sobel derivatives with kernel size 3 produce $g _ { x }$ and $g _ { y } \ [ 7 ]$ . The sharpness statistic is the population standard deviation over pixels,

$$
B _ { c } ( t ) = \mathrm { s t d } _ { x , y } \bigg ( \sqrt { g _ { x } ( x , y , t ) ^ { 2 } + g _ { y } ( x , y , t ) ^ { 2 } } \bigg ) .\tag{A.9}
$$

Frames are fully blurred for $B _ { c } < 2 0$ , partially blurred for $2 0 \leq B _ { c } < 2 5$ , and otherwise not blurred. Consecutive identical labels are merged into half-open intervals. Because the statistic is not scale-normalised, the thresholds depend on image resolution, texture, exposure and viewpoint. They should not be interpreted as universal focus thresholds.

## A.4 Camera utility from role matches and penalties

Let $A _ { p } ( t )$ and $A _ { s } ( t )$ be the global primary and secondary audible label sets, and let $V _ { c , p } ( t )$ and $V _ { c , s } ( t )$ be the corresponding visible sets for camera c. Scores are evaluated at integer seconds. An interval label is present when $s \leq t < e$ . Identical audio intervals repeated across camera tasks are deduplicated. The scoring parser combines Speech and Singing into the token Speech/Singing; clapping labels, when present, are treated as primary event labels.

For audible role $r \in \{ p , s \}$ and label $^ { a , }$ define the match contribution

$$
M _ { c , r } ( a , t ) = \left\{ \begin{array} { l l } { w _ { r , p } , } & { a \in V _ { c , p } ( t ) , } \\ { w _ { r , s } , } & { a \in V _ { c , s } ( t ) \setminus V _ { c , p } ( t ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{A.10}
$$

where visible primary membership takes precedence if both visible roles occur. The match bonus is $\begin{array} { r } { R _ { c } ( t ) = \sum _ { r \in \{ p , s \} } \sum _ { a \in A _ { r } ( t ) } M _ { c , r } ( a , t ) } \end{array}$ . With $V _ { c } = V _ { c , p } \cup V _ { c , s }$ and $A = A _ { p } \cup A _ { s }$ , the positive penalty magnitude is

$$
P _ { c } ( t ) = 0 . 9 | A _ { p } ( t ) \setminus V _ { c } ( t ) | + 0 . 5 | A _ { s } ( t ) \setminus V _ { c } ( t ) | + 0 . 7 5 | V _ { c } ( t ) \setminus A ( t ) | .\tag{A.11}
$$

The per-second score is

$$
S _ { c } ( t ) = \mathrm { r o u n d } _ { 2 } ( [ 4 + R _ { c } ( t ) - P _ { c } ( t ) ] Q _ { c } ( t ) ) ,\tag{A.12}
$$

with the role weights and quality multipliers in Table A.1. The ofset does not guarantee nonnegative scores; nonnegativity is enforced at interval aggregation. Partially blurred footage receives no additional score reduction in this configuration. A fully blurred label or a recognised lowercase pan label sets the score to zero. The prediction-driven variant uses blur and instrument-role estimates; the annotation-driven variant also supplies pan and clapping labels.

Table A.1: System parameters for audio analysis, visual cues, camera scoring and selection.
<table><tr><td>Setting</td><td>Symbol / condition</td><td>Value</td></tr><tr><td>Visual detector</td><td>D-FINE / Objects365</td><td>X</td></tr><tr><td>Detection confidence</td><td>retained predictions</td><td>0.55</td></tr><tr><td>Primary visual role</td><td>fraction of frame area</td><td>&gt; 0.20</td></tr><tr><td>Audio RMS frame / hop</td><td>seconds</td><td>3.0 / 1.5</td></tr><tr><td>Audio smoothing</td><td>centred RMS frames</td><td>9</td></tr><tr><td>Activity normalisation</td><td>shared percentiles</td><td>1 / 99</td></tr><tr><td>Absolute / relative activity</td><td>thresholds</td><td>0.075 / 0.35</td></tr><tr><td>Primary audio role</td><td>relative energy share</td><td>≥ 0.92</td></tr><tr><td>Sharpness sampling</td><td>seconds</td><td>0.2</td></tr><tr><td>Sharpness filters</td><td>Gaussian / Sobel kernels</td><td> $5 \times 5 ~ / ~ 3$ </td></tr><tr><td>Blur classes</td><td>full / partial thresholds</td><td>&lt; 20 / &lt; 25</td></tr><tr><td>Tempo estimation</td><td>window / hop (s)</td><td>30 / 10</td></tr><tr><td>Tempo classes</td><td>BPM boundaries</td><td>90 / 130</td></tr><tr><td>Target dwell times</td><td>slow / middle / fast (s)</td><td>10-15 / 6-10 / 3-6</td></tr><tr><td>Timing grid</td><td>tracked-beat step</td><td>4</td></tr><tr><td>Primary audio, primary visible</td><td> $w _ { p , p }$ </td><td>4.0</td></tr><tr><td>Primary audio, secondary visible</td><td>Wp,s</td><td>1.0</td></tr><tr><td>Secondary audio, primary visible</td><td> $w _ { s , p }$ </td><td>1.0</td></tr><tr><td>Secondary audio, secondary visible</td><td> $w _ { s , s }$ </td><td>0.5</td></tr><tr><td>Missing primary audible label</td><td>penalty per label</td><td>0.9</td></tr><tr><td>Missing secondary audible label</td><td>penalty per label</td><td>0.5</td></tr><tr><td>Visible but inaudible label</td><td>penalty per label</td><td>0.75</td></tr><tr><td>Score offset</td><td>before quality multiplication</td><td>4.0</td></tr><tr><td>Quality multipliers</td><td>fully / partly blurred / pan</td><td> $0 ~ / \mathrm { ~ 1 ~ / ~ 0 ~ }$ </td></tr><tr><td>Lower-tail fraction</td><td>β</td><td>0.10</td></tr><tr><td>Transition weights</td><td>diagonal / off-diagonal</td><td>0 / 0.30</td></tr><tr><td>Selection and repeat constraint</td><td>categorical / no repeat</td><td>enabled</td></tr><tr><td>Rank conversion</td><td>use_rank_weights</td><td>disabled</td></tr><tr><td>Score time offset</td><td>seconds</td><td>0.0</td></tr><tr><td>Initial previous-camera state</td><td>before first selection</td><td>CAM1</td></tr></table>

## A.5 Lower-tail interval aggregation and camera sampling

For candidate interval $[ t , t ^ { \prime } )$ , the code uses the integer-second bins from $\lfloor t \rfloor$ through $\boldsymbol { \lceil t ^ { \prime } \rceil } - 1$ . This includes the bin containing a fractional start time. Let n be their number and let $S _ { c , ( 1 ) } \leq \cdots \leq S _ { c , ( n ) }$ denote the sorted scores. The interval utility is the clipped mean of the lowest m samples,

$$
q _ { c } ( [ t , t ^ { \prime } ) ) = \mathrm { m a x } \left( 0 , \frac { 1 } { m } \sum _ { j = 1 } ^ { m } S _ { c , ( j ) } \right) , \qquad m = \mathrm { m a x } ( 1 , \lceil \beta n \rceil ) , \quad \beta = 0 . 1 0 .\tag{A.13}
$$

Missing score bins are zero. This empirical lower-tail average is related to lower-tail conditional value-at-risk [13]; the ceiling convention is the implemented finite-sample rule, rather than an exact fractional-tail estimator. For $n \leq 1 0 .$ , it reduces to the minimum score clipped at zero. This makes brief failures influential, which can reject an otherwise useful shot; the aggregation is a conservative choice, not a proven editorial optimum.

Define $\textstyle w _ { c } = q _ { c } / \sum _ { j } q _ { j }$ when the sum is positive and $w _ { c } = 1 / 4$ otherwise. Given previous camera $i ,$ the implementation forms

$$
\widetilde { p } _ { j } = T _ { i j } w _ { j } , \qquad p _ { j } = \frac { \widetilde { p } _ { j } } { \sum _ { k } \widetilde { p } _ { k } } , \qquad T _ { i j } = \left\{ { 0 , \qquad i = j , } \atop { 0 . 3 0 , } \right. \qquad i \neq j .\tag{A.14}
$$

When all remaining weights vanish, the fallback is uniform over the three cameras other than i. The next camera is drawn from this categorical distribution for the entire interval, and becomes the previous camera for the next decision. The initial previous-camera state is CAM1, so CAM1 is excluded from the first draw as well.

The of-diagonal value 0.30 is an unnormalised weight: the final normalisation makes it equivalent to $1 / 3$ . With positive eligible utilities, Equation A.14 simplifies to $\begin{array} { r } { p _ { j } = q _ { j } / \sum _ { k \neq i } q _ { k } } \end{array}$ for $j \neq i$ Thus the transition mechanism is a first-order no-repeat constraint, not a learned model of editing grammar. Scheduling methods can use richer transition models [3]; no such benefit is established here. Moreover, a zero utility is not an unconditional shot veto because the all-zero fallback may still choose an unsuitable view.

Speech and singing share a combined event label. The timing module applies the same policy throughout the clip; a separate announcement policy is a direction for further development.

## B Component evaluation

## B.1 Sampling and metric definitions

Audio labels are evaluated on integer bins $\{ \lfloor s \rfloor , \ldots , \lfloor e \rfloor - 1 \}$ for each interval $[ s , e )$ over timestamps 0–480. This yields 481 samples, including a terminal empty bin. Repeated audio labels across camera views form global audible-role sets.

Primary and secondary roles are evaluated as separate multilabel tasks over the six instruments. Not active is the complement of the union of these roles, per instrument. Clapping is outside this six-instrument evaluation. For each label, precision P, recall R and F1 are

$$
P = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } } , \qquad R = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } , \qquad F 1 = { \frac { 2 \mathrm { T P } } { 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } .\tag{B.1}
$$

Support $N _ { \ell }$ is the number of positive reference samples. Weighted summaries are $\textstyle \sum _ { \ell } N _ { \ell } m _ { \ell } / \sum _ { \ell } N _ { \ell }$ for metric $m .$ They are not micro-averages, and weighted F1 is not calculated from weighted precision and recall. Multilabel supports can exceed the number of time samples. Undefined metrics are set to zero in the weighted calculation; classes absent from the reference are shown with dashes in the table.

For blur, each camera is evaluated at integer instants $s \leq t < e$ , with fully blurred labels taking precedence over partially blurred labels. Unlabelled instants are treated as not blurred. The horizon is the maximum interval end in either export, giving the camera-specific sample counts in Table 2. The metrics describe this recording; adjacent temporal samples are dependent.

## B.2 Audio roles

Table B.1: Audio-role classification against reference labels. Each group reports reference support $N ,$ precision, recall and F1. The 481 time samples form a multilabel task. Dashes indicate absent reference classes $( N = 0 )$ ; the underlying evaluator assigns zero to undefined scores.
<table><tr><td></td><td colspan="4">Not active</td><td colspan="4">Primary</td><td colspan="4">Secondary</td></tr><tr><td>Instrument</td><td>N</td><td>P</td><td>R</td><td>F1</td><td>N</td><td>P</td><td>R</td><td>F1</td><td>N</td><td></td><td>P R</td><td>F1</td></tr><tr><td>Bass</td><td>189</td><td>.639</td><td>1.000</td><td>.779</td><td>0</td><td></td><td></td><td></td><td>292</td><td>1.000</td><td>.599</td><td>.749</td></tr><tr><td>Drums</td><td>189</td><td>.508</td><td>1.000</td><td>.674</td><td></td><td>95.667</td><td>.021</td><td>.041</td><td>197</td><td>.541</td><td>.299.386</td><td></td></tr><tr><td>Guitar</td><td>154</td><td>.726</td><td>1.000</td><td>.842</td><td></td><td>95.528</td><td>.905</td><td>.667</td><td>232</td><td>.953</td><td>.435.598</td><td></td></tr><tr><td>Piano</td><td>481</td><td>1.000</td><td>.981</td><td>.991</td><td>0</td><td></td><td></td><td></td><td>0</td><td></td><td></td><td></td></tr><tr><td>Speech</td><td>413</td><td>.998</td><td>.983</td><td>.990</td><td>68</td><td>.905</td><td>.985</td><td>.944</td><td>0</td><td></td><td></td><td></td></tr><tr><td>Trombone</td><td>206</td><td>.787</td><td>.879</td><td>.830</td><td>275</td><td>.954</td><td>.673</td><td>.789</td><td>0</td><td></td><td></td><td></td></tr><tr><td>Weighted</td><td>1632</td><td>.848</td><td>.975.895</td><td></td><td>533</td><td>.820</td><td>.638</td><td>.654</td><td>721</td><td>.859</td><td>.465.601</td><td></td></tr></table>

Table B.1 shows uneven role detection. Speech primary labels have F1 0.944, while primary Drums have recall 0.021 and F1 0.041 despite 95 positive reference samples. Guitar primary recall is 0.905 but precision is 0.528; Trombone primary precision is 0.954 with recall 0.673. The zero reference support for primary Bass and for all active Piano roles prevents claims about detecting those roles. Inactive-role and active-role performance therefore need separate interpretation. Figure B.1 illustrates the fragmentation and timing diferences that remain after smoothing.

## B.3 Blur classification

Table B.2: Blur-classification precision, recall and F1. Reference supports and camera-specific n are given in Table 2. Dashes denote absent reference classes. Weighted summaries use class supports, including the poor results on CAM1.
<table><tr><td></td><td colspan="3">CAM1</td><td colspan="3">CAM2</td><td colspan="3">CAM3</td><td colspan="3">CAM4</td></tr><tr><td>Class</td><td>P</td><td></td><td>R1 F1</td><td></td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>Not blurred</td><td>.01</td><td>1.00</td><td>.03</td><td>.16</td><td>.98</td><td>.28</td><td>.01</td><td>1.00</td><td>.02</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Partially blurred .00</td><td></td><td>.00</td><td>.00</td><td>.90</td><td>.25</td><td>.39</td><td>.94</td><td>.30</td><td>.46</td><td></td><td></td><td></td></tr><tr><td>Fully blurred</td><td></td><td></td><td></td><td>.38</td><td>.45</td><td>.41</td><td>.44</td><td>.29</td><td>.35</td><td></td><td></td><td></td></tr><tr><td>Weighted</td><td>.00</td><td></td><td>.01.00</td><td>.77</td><td>.35</td><td>.38</td><td>.91</td><td>.31</td><td>.45</td><td>1.00</td><td>1.001.00</td><td></td></tr></table>

The partially blurred class on CAM2 and CAM3 has high precision but low recall; fully blurred detections are weaker. CAM1 is predicted as not blurred throughout although 474 of its 481 reference samples are partially blurred, producing near-zero weighted F1. CAM4 contains only not-blurred reference samples, so performance on blurred frames is not assessed for that view. Fixed, unnormalised image-gradient thresholds are particularly vulnerable to camera-dependent appearance. Moreover, the scoring configuration does not penalise partial blur, limiting the impact of this classification even when it is correct.

Figure B.2 compares the resulting camera-score profiles from predicted and reference labels.

![](images/aef2f162bdb62c22f1556b1cae5b740931addbd8482a37f7d85bee4bad01e828.jpg)

![](images/9759af622935ade6a3570fe267397c55d59c3579522420818722a0a7e09a6386.jpg)  
Figure B.1: Audio-role timelines (top, global labels) and visible instrument roles (bottom, CAM3). Coloured rows are predictions; hatched rows are reference annotations. Saturated colours denote primary roles and pale colours secondary roles. The original axis label “Zeit [s]” means time in seconds. The plots illustrate this case and do not establish visual detector accuracy.

![](images/c1facce0aa880e3ea81caeb1ea851ace54261bf6a8ac645e37e420c7bf9c20d7.jpg)

![](images/78b44f4a696418df9e62173b720a6a0c007f99cad064fdb7528c1112789a50d0.jpg)  
Figure B.2: Per-second camera scores from predicted labels (top) and reference annotations (bottom). Background shading indicates the instantaneous highest-scoring camera, while the selector samples cameras over complete intervals. The annotation-driven scores additionally use pan/clapping cues.

## C Additional temporal documentation

For a window $W = [ a , b )$ and successive video boundaries $c _ { j } , c _ { j + 1 }$ , the positive displayed segments are

$$
\ell _ { j } ( W ) = \operatorname* { m a x } \{ 0 , \operatorname* { m i n } ( c _ { j + 1 } , b ) - \operatorname* { m a x } ( c _ { j } , a ) \} .\tag{C.1}
$$

The sample SD is $s = \sqrt { \textstyle \sum _ { j } ( \ell _ { j } - \bar { \ell } ) ^ { 2 } / ( n - 1 ) }$ , undefined for $n = 1$ . Boundary clipping is distinct from a new camera cut.

Table C.1 supplements the music-specific analysis with the announcement and applause. Because these windows are approximate and shots crossing their boundaries are clipped, the applause statistics in particular describe short window fragments rather than independent complete shots. Sample SD is undefined for $n = 1$ , rather than set to zero. The boundary sensitivity rows use only complete shots contained within the music window: 33 for V1 and 54 for each automatic variant. The contrast between longer, more variable manual shots and shorter automatic shots remains descriptively visible; this is not a statistical test of musical appropriateness.

Table C.1: Additional shot-duration summaries (seconds). Announcement and applause rows include clipped boundary segments; “Music: complete” excludes both clipped music-boundary segments. SD is the sample standard deviation.
<table><tr><td>Variant</td><td>Window / inclusion</td><td>n</td><td>Mean</td><td>Median</td><td>SD</td></tr><tr><td>V1</td><td>Announcement</td><td>1</td><td>80.00</td><td>80.00</td><td></td></tr><tr><td>V2</td><td>Announcement</td><td>10</td><td>8.00</td><td>8.86</td><td>3.20</td></tr><tr><td>V3</td><td>Announcement</td><td>10</td><td>8.00</td><td>8.86</td><td>3.21</td></tr><tr><td>V4</td><td>Announcement</td><td>10</td><td>8.00</td><td>8.86</td><td>3.21</td></tr><tr><td>V1</td><td>Applause</td><td>2</td><td>3.85</td><td>3.85</td><td>4.09</td></tr><tr><td>V2</td><td>Applause</td><td>2</td><td>3.85</td><td>3.85</td><td>3.99</td></tr><tr><td>V3</td><td>Applause</td><td>2</td><td>3.85</td><td>3.85</td><td>3.95</td></tr><tr><td>V4</td><td>Applause</td><td>2</td><td>3.85</td><td>3.85</td><td>3.95</td></tr><tr><td>V1</td><td>Music: complete</td><td>33</td><td>10.02</td><td>6.31</td><td>8.82</td></tr><tr><td>V2</td><td>Music: complete</td><td>54</td><td>6.72</td><td>6.66</td><td>2.18</td></tr><tr><td>V3</td><td>Music: complete</td><td>54</td><td>6.72</td><td>6.66</td><td>2.18</td></tr><tr><td>V4</td><td>Music: complete</td><td>54</td><td>6.72</td><td>6.66</td><td>2.18</td></tr></table>

The four edits contain 230 internal camera changes in total. Cut times are defined by the first frame of the new camera view.

## D Questionnaire dimensions

Each video is rated on five dimensions, with four item-specific comment fields and general remarks.   
Table D.1 summarises the dimensions.

Table D.1: Dimensions assessed in the expert evaluation.
<table><tr><td>Field</td><td>Assessed dimension</td></tr><tr><td>Ovr</td><td>Overall quality and readiness for publication or broadcast.</td></tr><tr><td>Ed</td><td>Appropriateness of cut timing to the musical and content progression</td></tr><tr><td>Tr</td><td>Coherence and smoothness of image transitions; disruptive changes.</td></tr><tr><td>Sh</td><td>Overall picture quality and sharpness of the selected views.</td></tr><tr><td>Im</td><td>Coherence and aesthetic quality of the overall impression.</td></tr></table>

## E Individual expert responses

E1–E3 denote the three experts. The table contains all numeric responses; dashes indicate missing answers.

Table E.1: All recorded numbers, E1 / E2 / E3. A dash means no numeric answer.
<table><tr><td>Variant</td><td colspan="3">Ed</td><td colspan="5">Sh</td></tr><tr><td>V1</td><td> $3 ~ / ~ 3 ~ / ~ 3$ </td><td> $3 ~ / ~ 2 ~ / ~ 2$ </td><td></td><td> $3 ~ / ~ 5 ~ / ~ 2$ </td><td> $2 / 1 / 1$ </td><td></td><td></td><td> $2 ~ / ~ 3 ~ / ~ 3$ </td></tr><tr><td>V2</td><td> $4 ~ / ~ 3 ~ / ~ 1$ </td><td></td><td> $4 ~ / ~ 4 ~ / ~ 2$ </td><td> $3 \dot { / } 3 \dot { / } 1$ </td><td> $2 ~ / \mathrm { ~ 1 ~ / ~ 3 ~ }$ </td><td></td><td> $3 ~ / ~ 3 ~ / ~ 2$ </td><td></td></tr><tr><td>V3</td><td> $5 ~ / ~ 4 ~ / ~ 3$ </td><td></td><td> $5 / 4 / -$ </td><td> $/ 5 / -$ </td><td> $- / 1 / -$ </td><td></td><td> $- / 4 / -$ </td><td></td></tr><tr><td>V4</td><td> $4 ~ / ~ 4 ~ / ~ 1$ </td><td></td><td> $/ \mathrm { ~ 4 ~ } / \mathrm { ~ 1 ~ }$ </td><td> $/ \textrm { 3 / 1 }$ </td><td></td><td> $2 ~ / ~ 1 ~ / ~ 2$ </td><td> $/ \mathrm { ~ 4 ~ } / \mathrm { ~ 1 ~ }$ </td><td></td></tr></table>

Table E.2: Number of explicit numeric answers. Missing numeric answers remain missing even where a comment refers to another video.
<table><tr><td>Variant</td><td>Ovr</td><td>Ed</td><td>Tr</td><td>Sh</td><td>Im</td></tr><tr><td>V1</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>V2</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>V3</td><td>3</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>V4</td><td>3</td><td>2</td><td>2</td><td>3</td><td>2</td></tr></table>

## F Descriptive analysis of written feedback

Whole comment fields form the units of the descriptive thematic reading. A field can cover more than one theme. Empty fields and references to other videos are distinguished from substantive statements; a cross-reference does not generate another independent observation. The grouping retains positive and negative statements, explicit preferences and disagreements between experts. Themes describe expressed judgements rather than the prevalence of a problem or its technical cause. AI assistance in grouping the comments is disclosed in the main text.

## G Score sensitivity and camera-selection efects

An example illustrates how a change in a visibility label can afect camera scoring and selection. All other labels and the quality state remain fixed. For an audible label that is primary only, changing its visibility from absent in both visible-role sets to visually primary adds the primary–primary bonus of 4 and removes the missing-primary penalty of 0.9. The per-second score therefore increases by 4.9 when $Q _ { c } = 1 ;$ when $Q _ { c } = 0$ , both scores are zero.

Lower-tail aggregation and clipping at zero can leave the interval utility unchanged. The no-repeat rule also excludes the previous camera regardless of its score. For an eligible camera, normalisation depends on the utilities of the other eligible views: if it is the only view with positive utility, its probability remains one despite an increase in utility. Even when probabilities change, categorical sampling can return the same camera. These conditional efects distinguish label changes, score changes and camera choices; none alone establishes improved editorial quality.