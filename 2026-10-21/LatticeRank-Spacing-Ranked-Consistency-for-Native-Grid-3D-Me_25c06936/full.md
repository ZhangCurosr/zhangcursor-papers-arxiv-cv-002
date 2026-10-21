# LatticeRank: Spacing-Ranked Consistency for Native-Grid 3D Medical Image Segmentation

Jingjia Yan<sup>1</sup>, Yue Zhuo<sup>1</sup>, and Zhonghua Xie<sup>1⋆</sup>

Huizhou University, Huizhou 516007, Guangdong, China 1946084073@qq.com, 2937029343@qq.com, eexie@qq.com

Abstract. Variable voxel spacing changes more than image resolution: it alters the discrete sampling lattice, tensor geometry, and anatomical evidence available to a 3D segmentation network. While conventional pipelines restore images to a canonical grid, direct-grid deployment requires the model itself to tolerate these changes. We observe that candidate grids are not exchangeable views—a coarse grid removes more structural information than a mildly resampled one—yet standard consistency ob jectives assign them equal influence. We therefore propose LatticeRank, a training-only framework that aligns candidate posteriors in physical space and allocates consistency supervision by voxel-spacing order: among endpoint-matched candidates, the auxiliary loss concentrates on the coars est requested grid, while inference retains a single network forward with no candidate branch or ensemble. On MSD Task04, LatticeRank improves coarse-grid Extra Dice over mean consistency by 0.03436 across five seeds while meeting the prespecified reference-grid noninferiority criterion. A controlled ladder shows that grid exposure provides the largest recovery, posterior consistency adds a second gain, and spacing-ranked allocation yields the strongest direct-grid result; requested spacing thus provides a simple, model-independent ordering signal for direct-grid consistency learning.

Keywords: Medical image segmentation · voxel spacing · consistency learning · native-grid inference.

## 1 Introduction

Voxel spacing determines the lattice on which anatomy becomes data: increasing through-plane spacing reduces slice count, strengthens partial-volume mixing, and changes the feature sequence presented to a 3D network, so thin structures may disappear before segmentation begins. Existing systems counter this shift through spacing normalization, synthetic resolution diversity, or resolution-aware models [21,5,19,7,22].

When preprocessing or architecture cannot be changed, some deployments must segment the available tensor, a requirement now shared by speech [28], multimodal [44,25], and consumer systems [40,26] that likewise run on degraded inputs. In this direct-grid path, the model inherits both the altered lattice and the information loss of coarse sampling. Interpolation blur is not the whole problem: tensor depth changes, pooling stages operate on a diferent discrete geometry, and the evidence surviving at 5 mm is not the evidence surviving at 2 mm [1,39]. Training on multiple grids exposes the network to this shift but leaves a consequential choice unresolved: how should supervision be divided among candidate spacings sampled in the same update?

![](images/7a4d71a6f5b9e22ff9143af16913a4ac4a6087634856a0838dfbc8c631469a9a.jpg)  
Fig. 1. LatticeRank training and deployment. Endpoint-matched candidate grids share the segmentation network, and their posteriors are aligned to R1 before consistency is measured against a detached reference posterior. Requested spacing orders the candidates from fine to coarse; the coarsest candidate supplies the auxiliary loss. Candidate branches and alignment operations are removed at inference, leaving one direct-grid forward.

The candidates are not exchangeable augmentations: mean consistency gives mildly and severely undersampled views equal influence, and loss-based selection ties allocation to a moving model state. Voxel spacing provides a dificulty order before the model makes a prediction.

LatticeRank turns that order into supervision: endpoint-matched candidates share one network during training, their posteriors return to the reference lattice, and disagreement is measured against a detached reference prediction, with only the candidate of largest requested spacing supplying the auxiliary consistency loss. The training structure then disappears—inference is one direct-grid forward, with no candidate branch or ensemble. On Task04, the controlled ladder progresses from grid exposure to aligned consistency and finally to spacing-ranked allocation; the last step improves Extra Dice over mean consistency by 0.03436 across five seeds while meeting the prespecified reference-grid noninferiority criterion.

Our contributions are threefold: we formulate native-grid segmentation across voxel spacings as an ordered-view learning problem rather than exchangeable augmentations; we introduce a physically aligned, spacing-ranked consistency objective that concentrates supervision on the coarsest candidate without changing inference; and a controlled training ladder separates grid exposure, posterior consistency, and loss allocation, quantifying the accuracy–eficiency boundary against restoration.

## 2 Related Work

## 2.1 Resolution harmonization and spacing-adaptive segmentation

Work on voxel spacing can be read by where it intervenes. At the input, nnU-Net estimates a target spacing from dataset statistics [21], SynthSeg trains across synthetic contrasts and resolutions [5], and Consispace examines prediction stability under spacing changes [41]. Representation- and architecture-level methods move the intervention inside the model: Splatting maps variable-resolution inputs into a shared space [7], FastSurferVINN normalizes resolution internally [19], Hyper-Space generates spacing-conditioned weights [22], and CROWn addresses aliasing and phase mismatch in 3D downsampling and fusion [20]. These approaches establish spacing as a first-class design variable; LatticeRank uses it to allocate consistency loss during training, with the interpolation operator and backbone held fixed.

## 2.2 Consistency learning across aligned views

Consistency learning ofers a diferent route: preserve predictions while the input, teacher, task, or decoder changes. Mean Teacher introduced weight-averaged teacher targets [38]; medical segmentation work extended the idea through transformation consistency, uncertainty-aware self-ensembling, and teacher–student perturbations [6,42,30]. Dual-task, cross-decoder, and cross-network objectives obtain complementary predictions in other ways [34,35,9], and recent work connects global and local representations across resolutions [32]. The recurring question is how to construct a reliable target; here the targets are already defined after physical alignment, and the unresolved decision is how much optimization weight each ordered disagreement should receive.

## 2.3 Ordered, hard, and worst-view learning

Once several candidate losses coexist, the problem becomes one of selective optimization. Curriculum and self-paced learning schedule examples by dificulty [4,23]; online hard-example mining and focal loss increase the influence of dificult samples or predictions [37,31]; MaxUp backpropagates through the highest-loss augmentation among sampled views [17]; and AdvChain and ASC learn challenging medical-image transformations or augmentation severities [8,2]. LatticeRank uses a geometric ordering of views: requested spacing determines the selection before prediction, and physical alignment makes the posterior disagreements comparable. Unlike loss-based hard-view selection, this rule is independent of the current model state.

## 2.4 Degradation-specific enhancement and medical reconstruction

Degradation-specific networks accept the input as given: shadow removal via maskguided routing [43] or multi-stage benchmarks [33], underwater color fidelity [13], document contamination [15], and cinematic filter emulation from controlled datasets [24]; high-frequency synthesis under adverse acquisition [11] and compact compression of high-dimensional outputs [27]. Medical deployment raises the stakes: robotic surgical guidance enhances degraded camera feeds [29], functional ultrasound is reconstructed auto-regressively [12], generative models construct brain networks from difusion imaging [14], and image synthesis is giving way to digital-twin medical computing [10]. These systems pay of when the target acquisition is known in advance; LatticeRank instead varies one acquisition parameter continuously.

## 3 Method

## 3.1 Ordered native grids

Let $x ^ { ( 1 ) }$ and $y ^ { ( 1 ) }$ denote an image and label on the reference lattice R1 with nominal through-plane spacing 1 mm. For a requested spacing $s ,$ an endpointmatched grid $g ( s )$ preserves the physical upper and lower slice endpoints while changing the integer depth. Images are resampled directly to $g ( s )$ ; only the through-plane spacing varies, while in-plane spacing remains 1 mm. The same network $f _ { \theta }$ processes R1 and all candidate tensors, exposing the model to the discrete geometries of direct-grid inference without a spacing-specific backbone.

The candidate set $\{ g ( s _ { k } ) \} _ { k = 1 } ^ { K }$ is ordered by requested spacing: larger $s _ { k }$ yields fewer through-plane samples and, under endpoint matching, a coarser representation of the same physical extent. This order is a geometric severity prior, independent of labels, confidence, prediction error, gradients, or model state.

This order is deliberately defined on the requested spacing rather than an observed loss: endpoint quantization can make two requests realize the same integer depth, but the requested value remains the frozen variable that generated the view, supplying a deterministic allocation rule. The order encodes the expected information loss of coarser sampling, while posterior disagreement captures the case-specific response.

## 3.2 Physically aligned posterior consistency

The R1 branch produces logits $z _ { 0 } = f _ { \theta } ( x ^ { ( 1 ) } )$ and a detached foreground posterior

$$
q = \mathrm { s t o p g r a d } [ \mathrm { s o f t m a x } ( z _ { 0 } ) ] .\tag{1}
$$

For candidate $k ,$ the network prediction is mapped back to R1 with the known endpoint-matched transform:

$$
p _ { k } = \mathrm { A l i g n } _ { g ( s _ { k } ) \to \mathrm { R 1 } } \Big [ \mathrm { s o f t m a x } ( f _ { \theta } ( x ^ { g ( s _ { k } ) } ) ) \Big ] .\tag{2}
$$

Alignment is essential: disagreement must compare corresponding physical locations, not raw tensor indices. We define the candidate disagreement by foreground soft Dice,

$$
e _ { k } = 1 - \mathrm { S o f t D i c e } ( q , p _ { k } ) .\tag{3}
$$

The uniform consistency comparator averages all candidate disagreements,

$$
L _ { \mathrm { a u x } } ^ { C } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } e _ { k } .\tag{4}
$$

## 3.3 Spacing-ranked allocation

Uniform consistency treats candidate grids as exchangeable. LatticeRank instead selects the maximal requested spacing,

$$
I _ { s } = \arg \operatorname* { m a x } _ { k } s _ { k } , \qquad L _ { \mathrm { a u x } } ^ { S } = \frac { 1 } { \left| I _ { s } \right| } \sum _ { k \in I _ { s } } e _ { k } ,\tag{5}
$$

where exact requested-spacing ties are averaged. In the main method, $K = 2$ and $s _ { k }$ is drawn from a frozen log-uniform schedule on [1, 3] mm. Selecting the maximum shifts the distribution of views contributing auxiliary gradient toward coarser grids. Unlike a worst-loss rule, selection cannot switch when the network temporarily mispredicts a mild view; the allocation remains attached to the known grid geometry.

The supervised R1 objective and total training loss are

$$
L _ { \mathrm { n a t i v e } } = 0 . 5 L _ { \mathrm { D i c e } } ( z _ { 0 } , y ^ { ( 1 ) } ) + 0 . 5 L _ { \mathrm { C E } } ( z _ { 0 } , y ^ { ( 1 ) } ) ,\tag{6}
$$

$$
L = L _ { \mathrm { n a t i v e } } + \lambda ( t ) L _ { \mathrm { a u x } } , \lambda ( t ) = 0 . 1 \operatorname* { m i n } ( 1 , t / 5 0 0 ) ,\tag{7}
$$

where t is the update index. As illustrated in Fig. 1, the reference and candidate branches exist only during training; inference receives one tensor on its deployment grid, executes $f _ { \theta }$ once, and maps the posterior to R1 only when a common-grid output is required.

## 4 Experiments

## 4.1 Dataset and implementation

Experiments use MSD Task04 Hippocampus MRI with anterior and posterior hippocampal labels [3]; the disjoint groups contain 80 training, 20 validation, 20 development, and 40 held-out cases. The main backbone is a compact Static Mamba implementation with 99,411 parameters, used as implementation context rather than the contribution [18]; a same-dataset development analysis replaces only the backbone with a plain 3D U-Net [16] (Section 5.4).

All main runs use full volumes, batch size one, AdamW with learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 5 }$ , bfloat16, no additional augmentation, and 2,500 updates; validation occurs every 250 updates, and the checkpoint with the best R1 foreground Dice is retained. The full ladder uses seeds $1 7 / 2 9 / 4 1$ to separate the contributions of exposure, consistency, and allocation; the primary S-versus-C comparison adds seeds 53/67 for five training runs.

## 4.2 Grid construction and optimization controls

Candidate requests are drawn from frozen per-seed log-uniform schedules, and each candidate image is resampled directly from the raw volume to its endpointmatched lattice rather than generated recursively from R1. Candidate probabilities return to R1 by the known afine transform with align\_corners disabled; the corresponding label is used only by DirectLabel, while C, D, and S compare aligned candidate posteriors with the detached R1 posterior. This common construction prevents interpolation chains or mismatched physical coordinates from becoming hidden method diferences.

C, D, and S all construct and forward two candidate tensors per update; C and D backpropagate through both disagreements, whereas S backpropagates through only the spacing-selected one. The backbone contains LayerNorm and GroupNorm but no BatchNorm, dropout, running-statistic bufer, or other stateupdating layer, so the unselected S forward is unnecessary for the objective or persistent state; it is retained only to match the two candidate forwards in C and D. Random single-candidate allocation is gradient-matched to S but not forward-matched, hence a selector control rather than a compute-equivalent baseline.

## 4.3 Controlled training ladder

The ladder separates the source of direct-grid improvement. Single trains only on R1 and is forwarded directly at every evaluation spacing; Restore resamples each coarse tensor back to R1 before model input; DirectLabel forwards candidate tensors directly and supervises their R1-mapped predictions with mapped labels; C replaces candidate labels with mean aligned-posterior consistency; D weights disagreements with detached error-softmax weights; and S, the proposed LatticeRank rule, assigns consistency to the candidate with maximal requested spacing. Random single-candidate allocation, soft spacing weighting, realized-spacing selection, candidate-count changes, and training-range changes are reported in Sections 5.3 and 5.4.

## 4.4 Evaluation and statistics

Evaluation uses requested spacings [1, 1.5, 2, 2.5, 3, 4, 5] mm; because training samples candidates only from 1–3 mm, we designate 4 and 5 mm as the out-of range coarse-grid stress regime. Their mean mapped Dice was fixed as Extra Dice before the five-seed S-versus-C comparison, and the complete spacing curves are reported to avoid selecting favorable endpoints. $D _ { \mathrm { m a p } } ( s )$ compares a candidate prediction mapped to R1 with the R1 label; Worst-grid Dice is the minimum over 1.5–5 mm; Grid-AUC integrates prediction agreement $D _ { \mathrm { p r e d } }$ over 1.5–5 mm and is not an accuracy metric.

Paired contrasts use a crossed bootstrap with 10,000 draws, independently resampling cases and training seeds before averaging their crossed product; the primary comparison is S minus C Extra Dice on the 40 held-out cases and five seeds. R1 Dice is monitored with a −0.005 noninferiority margin. The deployment benchmark measures warm-cache operator latency from the same precomputed coarse tensor to the same R1 hard prediction, excluding image I/O and raw-tocoarse generation.

![](images/874d18867b7bda8cd05889a8b5172bc6425b193ff670ebd18c6d951f5ac92f49.jpg)

![](images/ae5b3c2f2e9ac635265da51325f2bacfa9ad883ebf405b05721a3fc318c97720.jpg)  
Fig. 2. Controlled ladder on 40 held-out Task04 cases and seeds $1 7 / 2 9 / 4 1$ . Candidate exposure supplies the largest direct-grid recovery, aligned posterior consistency adds a second gain, and LatticeRank (S) achieves the strongest coarse-grid result among direct methods. Restoration remains the accuracy-oriented path.

## 5 Results

## 5.1 Grid exposure and consistency recover direct-grid segmentation

The complete ladder reveals where performance is gained (Fig. 2 and Table 1). A model trained only on R1 nearly collapses at the 4/5-mm stress grids, with Extra Dice 0.02102. Direct candidate exposure with mapped-label supervision raises this value to 0.25414, a gain of 0.23312 [0.19043, 0.27650]. Replacing mapped labels with aligned posterior consistency produces a second step: C reaches 0.33715, improving on DirectLabel by 0.08301 [0.06975, 0.09473], and D and S further increase Extra Dice to 0.35523 and 0.37855.

## 5.2 Spacing ranking improves over uniform consistency

Across five seeds, LatticeRank improves Extra Dice over mean consistency by 0.03436, with a crossed 95% confidence interval of [0.01013, 0.05824] (Fig. 3). Four of the five seed-level efects are positive; seed 41 is retained in the negative direction. The R1 contrast is 0.00012 [−0.00247, 0.00255], its lower bound above the −0.005 guardrail, so the reference-grid result meets the prespecified noninferiority criterion. The S-minus-D Extra Dice contrast is 0.00443 [−0.02495, 0.03387]; this interval includes zero, and the five-seed evidence establishes the gain over uniform consistency.

Table 1. Held-out ladder. Extra is mean mapped Dice at $4 / 5$ mm; Worst is the minimum mapped Dice over 1.5–5 mm. Grid-AUC measures agreement with the R1 prediction. For Extra, Worst, and Grid-AUC, best values are bold and second-best values are underlined.
<table><tr><td>Method</td><td>R1</td><td>Extra</td><td>Worst</td><td>Grid-AUC</td><td>Train h</td></tr><tr><td>Single</td><td>0.88169</td><td>0.02102</td><td>0.00885</td><td>0.21639</td><td>0.63</td></tr><tr><td>Restore</td><td>0.88305</td><td>0.70226</td><td>0.63056</td><td>0.86056</td><td>2.80</td></tr><tr><td>DirectLabel</td><td>0.88348</td><td>0.25414</td><td>0.11299</td><td>0.60313</td><td>2.69</td></tr><tr><td>C</td><td>0.88365</td><td>0.33715</td><td>0.17547</td><td>0.65277</td><td>1.85</td></tr><tr><td>D</td><td>0.88417</td><td>0.35523</td><td>0.18959</td><td>0.66329</td><td>1.84</td></tr><tr><td>S (LatticeRank)</td><td>0.88344</td><td>0.37855</td><td>0.20926</td><td>0.68032</td><td>1.60</td></tr></table>

A Five-seed contrasts  
![](images/e4e85ef6e1979fe18f654312b2dd1afc43aa3a4834931304458c7746b35d1261.jpg)

![](images/4ac3807d3deaed2b86f395245e1679a5b80e6b994e0fcc2e019aaf84327ba443.jpg)  
Fig. 3. Five-seed S-versus-C evidence. LatticeRank improves Extra Dice over uniform consistency, while the R1 diference remains above the prespecified noninferiority margin. Seed-level efects show the variation across training runs, including the negative efect for seed 41.

## 5.3 What the ranking rule changes

The secondary controls compare spacing selection with random allocation and alternative spacing weights: a random single-candidate rule reaches Extra Dice 0.32289, giving S minus Random +0.05566 [0.04155, 0.07290]: selecting only one disagreement is insuficient by itself. Soft spacing weighting reaches 0.36262, and S improves on this formulation by 0.01593 [0.01178, 0.01918]. Requestedand realized-spacing selection choose the same strict winner at every recorded update, difering only when endpoint quantization creates an exact realizedspacing tie; the realized-spacing rule reaches 0.37533, only 0.00322 below S. These comparisons support coarse-grid emphasis among the tested rules.

The qualitative cases preserve the same message without claiming per-case dominance (Fig. 4): at 4 mm and in one 5-mm median-bracketing case, S produces the strongest direct-grid contour among $\mathrm { C / D / S }$ , while the final row shows the lowest 5-mm $S - \operatorname* { m a x } ( C , D )$ case, where S is below both comparators; coarse sampling leaves substantial information loss even with consistency training.

GT  
C  
D  
LatticeRank (S)  
![](images/d293051d1cdc1bfe6a5597ded948644437e35f14ec3a97d7d1059571573668cd.jpg)  
Fig. 4. Unmodified seed-17 qualitative cases. Rows show a 4-mm case, a 5-mm medianbracketing case, and the lowest 5-mm $S - \operatorname* { m a x } ( C , D )$ case. Columns show input, ground truth, restoration, C, D, and LatticeRank (S). Foreground Dice (restore $/ \mathrm { C } / \mathrm { D } / \mathrm { S } )$ is 0.788/0.479/0.510/0.564, 0.511/0.160/0.174/0.210, and 0.520/0.178/0.192/0.175 by row. Blue and orange denote ground-truth and prediction contours.

## 5.4 Candidate-set sensitivity and backbone transfer

The frozen main configuration uses two candidates sampled from 1–3 mm; postselection checks vary one aspect of this choice at a time (Fig. 5 and Table 2). Increasing the candidate count to three raises Extra Dice from 0.37855 to 0.42039, while a fourth gives 0.41774, indicating no monotonic benefit beyond three candidates in this small panel; these variants also increase update time and peak memory and do not replace the frozen $K = 2$ setting. Narrowing the training range to 1–2 mm reduces Extra Dice to 0.06705; the severe loss is concentrated outside the shortened range, indicating that the selector cannot compensate for missing coarse-grid exposure.

A separate Plain 3D U-Net development block asks whether the ordering signal depends on the Static Mamba implementation. With the backbone changed and the $\mathrm { C / D / S }$ protocol otherwise retained, S reaches Extra Dice 0.47420, compared with 0.41656 for C and 0.46165 for D; the S − C contrast is +0.05764 [0.01911, 0.10027], with all three seed-level directions positive. This 20-case development check supports transfer of the ordering efect to a second backbone, although its R1 lower confidence bound slightly crosses the main −0.005 guardrail.

![](images/c52bfc12b7a6a5f6fe803ebd75f912a3c344bb332ce22ba755bf9d1da6914cf2.jpg)  
Fig. 5. Candidate-set sensitivity and same-dataset backbone comparison. Left: postselection analyses of candidate count and training range on the three-seed held-out panel. Right: C/D/S with a Plain 3D U-Net on the development set. These supporting analyses do not redefine the main configuration.

Table 2. Candidate-set sensitivity and same-dataset backbone transfer. The upper block is post-selection held-out analysis; the lower block is development-only. Extra is mean mapped Dice at 4/5 mm.
<table><tr><td>Setting</td><td>Rule</td><td>R1</td><td>Extra</td><td>Worst</td><td>Grid-AUC</td></tr><tr><td>K = 2, 1–3 mm</td><td>S (frozen)</td><td>0.88344</td><td>0.37855</td><td>0.20926</td><td>0.68032</td></tr><tr><td>K = 3, 1–3 mm</td><td>S</td><td>0.88573</td><td>0.42039</td><td>0.25008</td><td>0.70160</td></tr><tr><td>K = 4, 1–3 mm</td><td>S</td><td>0.88361</td><td>0.41774</td><td>0.25150</td><td>0.70080</td></tr><tr><td>K = 2, 1−2 mm</td><td>S</td><td>0.88576</td><td>0.06705</td><td>0.02936</td><td>0.44836</td></tr><tr><td>Plain 3D U-Net</td><td>C</td><td>0.89631</td><td>0.41656</td><td>0.22196</td><td>0.72072</td></tr><tr><td>Plain 3D U-Net</td><td>D</td><td>0.89268</td><td>0.46165</td><td>0.27688</td><td>0.73891</td></tr><tr><td>Plain 3D U-Net</td><td>S</td><td>0.89455</td><td>0.47420</td><td>0.28218</td><td>0.75036</td></tr></table>

## 5.5 Accuracy–eficiency boundary

LatticeRank targets pipelines that require inference on the available grid, improving direct-grid accuracy without inference branches. Restoration provides Extra Dice 0.70226, compared with 0.37855 for LatticeRank. From a precomputed coarse tensor to a common R1 hard prediction, its latency ratio is 0.850 [0.843, 0.857], roughly 15% lower warm-cache latency (median 33.476 ms versus 40.270 ms for restoration), with peak allocated memory efectively unchanged.

![](images/8d5cf7c0926d63870b16ed67a8a8777367e900dda08d39a858d8ca44146d1de2.jpg)

![](images/a6529b31bfcebf4479328157e9a80058b77af8287af7f7ceb8a29bd807d84d13.jpg)  
Fig. 6. Common-endpoint operator benchmark. Direct forwarding reduces warm-cache latency relative to restoration, but the accuracy gap grows on coarse grids and no peak-memory advantage is observed.

As spacing becomes coarser, direct forwarding becomes faster but loses more accuracy (Fig. 6); restoration remains preferable when accuracy takes priority.

## 6 Discussion

Spacing supplies a stable prior over expected information loss. Mean consistency ignores that asymmetry and lets mild views dilute the coarse-grid signal; error- or uncertainty-based selectors can focus learning, but their notion of dificulty moves with the model. Requested spacing is available before optimization, and physical alignment keeps the selected disagreement anatomically meaningful, placing LatticeRank between resolution-aware segmentation and worst-view learning without altering the inference network. Compositional concept extraction [36] ofers one auditing route for what the network uses; the spacing ordering remains an inspectable geometric prior.

The evaluation uses controlled through-plane resampling of Task04 MRI, with $4 / 5$ mm as out-of-range stress tests; it measures robustness under this construction, while clinical reliability at these spacings requires task-specific validation. Independent acquisition shifts, external datasets, and matched comparisons with spacing-adaptive systems remain directions for evaluation.

## 7 Conclusion

LatticeRank treats voxel-spacing candidates as ordered views rather than exchangeable augmentations. By aligning their posteriors in physical space and assigning consistency to the coarsest requested grid, it improves coarse-grid segmentation over uniform consistency while meeting the prespecified referencegrid noninferiority criterion and retaining single-forward inference. Restoration remains the stronger accuracy path, but LatticeRank moves direct-grid inference to a better point on the accuracy–latency curve; requested spacing thus provides a simple, model-independent training signal for direct-grid segmentation under the tested through-plane spacing changes.

Acknowledgments. This work was supported by the National Natural Science Foundation of China (Grant No. 62001184), the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2025A1515110445), the Projects of Educational Commission of Guangdong Province (Grant Nos. 2025ZDZX1038, 2025KTSCX49, and 2023ZDZX1025), the Huizhou University Independent Innovation Capability Enhancement Plan Project (Grant No. HZU202535), and the College Students’ Innovative Entrepreneurial Training Plan Program (Grant Nos. 202410577008 and 202610577034).

## References

1. Aganj, I., Yeo, B.T.T., Sabuncu, M.R., Fischl, B.: On removing interpolation and resampling artifacts in rigid image registration. TIP 22(2), 816–827 (2013)

2. Alamdar, A., Jafarinia, H., Noori, M., Rohban, M.H.: ASC: Learning augmentation severity-consistent representations improves generalization via augmentation search. In: WACV. pp. 1926–1936 (2026)

3. Antonelli, M., Reinke, A., Bakas, S., et al.: The medical segmentation decathlon. Nature Communications 13, 4128 (2022)

4. Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning. In: ICML. pp. 41–48 (2009)

5. Billot, B., Greve, D.N., Puonti, O., Thielscher, A., Van Leemput, K., Fischl, B., Dalca, A.V., Iglesias, J.E.: SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. Medical Image Analysis 83, 102789 (2023)

6. Bortsova, G., Dubost, F., Hogeweg, L., Katramados, I., de Bruijne, M.: Semisupervised medical image segmentation via learning consistency under transformations. In: MICCAI. vol. 11769, pp. 810–818 (2019)

7. Brudfors, M., Balbastre, Y., Ashburner, J., Rees, G., Nachev, P., Ourselin, S., Cardoso, M.J.: Fitting segmentation networks on varying image resolutions using splatting. In: Medical Image Understanding and Analysis. vol. 13413, pp. 271–282 (2022)

8. Chen, C., Qin, C., Ouyang, C., Li, Z., Wang, S., Qiu, H., Chen, L., Tarroni, G., Bai, W., Rueckert, D.: Enhancing MR image segmentation with realistic adversarial data augmentation. Medical Image Analysis 82, 102597 (2022)

9. Chen, X., Yuan, Y., Zeng, G., Wang, J.: Semi-supervised semantic segmentation with cross pseudo supervision. In: CVPR. pp. 2613–2622 (2021)

10. Chen, X., Jing, C., Li, Z., Shen, Y., Huang, G., Liu, Y., Lei, B., Pun, C.M., Wang, S.: From image synthesis to digital twins: A paradigm shift in ai-driven medical image computing. IEEE Transactions on Consumer Electronics 72(3), 6132–6144 (2026)

11. Chen, X., Li, A., Dong, Y., Mao, Y., Liang, Z.: Multivariate feature-guided infrared image synthesis with high-frequency fidelity. IEEE Transactions on Consumer Electronics pp. 1–1 (2026)

12. Chen, X., Li, Z., Shen, Y., Mahmud, M., Pham, H., Ng, M.K.P., Pun, C.M., Wang, S.: High-fidelity functional ultrasound reconstruction via a visual auto-regressive framework. IEEE Journal of Biomedical and Health Informatics pp. 1–16 (2025)

13. Chen, X., Li, Z., Zhang, X., Lei, Y., Dong, Y., Zhang, X., Pun, C.M.: Aquaclarity: Vision transformer enhances underwater image via nonlinear feature fusion and color fidelity improvement. IEEE Transactions on Consumer Electronics 72(1), 1122–1129 (2026)

14. Chen, X., Shen, Y., Mahmud, M., Kwok-Po Ng, M., Tsang, K.F., Wang, S., Pun, C.M., Wang, S.: Connectomedifuser: Generative ai enables brain network construction from difusion tensor imaging. IEEE Transactions on Consumer Electronics 71(3), 7835–7847 (2025)

15. Chen, X., Zhou, Z., Li, Z., Zhang, X., Dong, Y., Tsang, K.F.: Docunfold: Leveraging unfolding network and a real-world large-scale dataset for handwriting contamination removal in documents. IEEE Transactions on Consumer Electronics 72(1), 1138–1145 (2026)

16. Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3d U-Net: Learning dense volumetric segmentation from sparse annotation. In: MICCAI. vol. 9901, pp. 424–432 (2016)

17. Gong, C., Ren, T., Ye, M., Liu, Q.: MaxUp: Lightweight adversarial training with data augmentation improves neural network training. In: CVPR. pp. 2474–2483 (2021)

18. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. In: COLM (2024)

19. Henschel, L., Kuegler, D., Reuter, M.: FastSurferVINN: Building resolutionindependence into deep learning segmentation methods—a solution for HighRes brain MRI. NeuroImage 251, 118933 (2022)

20. Huang, X., Ye, S., Huang, Z., Tang, W., Zhou, H., Zheng, Z., Liu, J., Zhang, X.: CROWn: A unified framework for anti-aliased downsampling and phase-calibrated fusion in 3d medical segmentation. In: CVPR. pp. 8514–8524 (2026)

21. Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203–211 (2021)

22. Joutard, S., Pietsch, M., Prevost, R.: HyperSpace: Hypernetworks for spacingadaptive image segmentation. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. vol. 15009, pp. 339–349 (2024)

23. Kumar, M.P., Packer, B., Koller, D.: Self-paced learning for latent variable models. In: NeurIPS. vol. 23, pp. 1189–1197 (2010)

24. Lei, Y., Li, Z., Pun, C.M., Wu, W., Yang, J., Chen, X.: Promist-5k: A comprehensive dataset for digital emulation of cinematic pro-mist filter efects. In: ICASSP. pp. 9362–9366 (2026)

25. Li, A., Li, J., Lin, P., Xian, J., Xu, H., Dong, Y., Chen, X.: A redundancy-aware multimodal emotion recognition network via cross-modal native sparse attention. IEEE Transactions on Consumer Electronics pp. 1–1 (2026)

26. Li, A., Lin, P., Li, J., Xian, J., Lv, H., Dong, Y., Chen, X.: Naco: Realizing humanlike naturalistic psychological counseling with cora-mcts planning and progressivesimpo alignment. IEEE Transactions on Consumer Electronics pp. 1–1 (2026)

27. Li, A., Lv, H., Xian, J., Li, J., Lin, P., Xu, J., Dong, Y., Chen, X.: Latent4d: Extreme compression of 4d gaussian splatting via variational autoencoders. IEEE Transactions on Consumer Electronics pp. 1–1 (2026)

28. Li, A., Lv, H., Xu, H., Li, S., Qin, S., Li, Z., Pun, C.M., Chen, X.: Atrie: Adaptive tuning for robust inference and emotion in persona-driven speech synthesis. In: ICMR. pp. 2561–2570 (2026)

29. Li, Q., Li, W., Xia, H., Zhou, J., Pun, C.M., Chen, X.: Rga-net: A vision enhancement framework for robotic surgical systems using reciprocal attention mechanisms. In: ICRA. pp. 16648–16655 (2026)

30. Li, X., Yu, L., Chen, H., Fu, C.W., Xing, L., Heng, P.A.: Transformation-consistent self-ensembling model for semisupervised medical image segmentation. TNNLS 32(2), 523–534 (2021)

31. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollar, P.: Focal loss for dense object detection. In: ICCV. pp. 2999–3007 (2017)

32. Liu, P., Zheng, G.: CR-GLoCo: Cross-resolution learning via global-local context consistency for semi-supervised 3d medical segmentation. Computerized Medical Imaging and Graphics 129, 102742 (2026)

33. Luo, T., Dong, Y., Bai, J., Huang, J., Xia, J., Wu, W., Chen, X.: Beyond shadows: A large-scale benchmark and multi-stage framework for high-fidelity facial shadow removal. In: ICASSP. pp. 11602–11606 (2026)

34. Luo, X., Chen, J., Song, T., Wang, G.: Semi-supervised medical image segmentation through dual-task consistency. In: AAAI. vol. 35, pp. 8801–8809 (2021)

35. Ouali, Y., Hudelot, C., Tami, M.: Semi-supervised semantic segmentation with cross-consistency training. In: CVPR. pp. 12674–12684 (2020)

36. Shi, H., Tao, H., Huang, G., Jiang, J., Chen, X., Pun, C., Wang, S., Pan, P.: Intrinsic concept extraction based on compositional interpretability. In: CVPR. pp. 38969–38978 (2026)

37. Shrivastava, A., Gupta, A., Girshick, R.: Training region-based object detectors with online hard example mining. In: CVPR. pp. 761–769 (2016)

38. Tarvainen, A., Valpola, H.: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In: NeurIPS. vol. 30, pp. 1195–1204 (2017)

39. Van Leemput, K., Maes, F., Vandermeulen, D., Suetens, P.: A unifying framework for partial volume segmentation of brain MR images. TMI 22(1), 105–119 (2003)

40. Xu, X., Pan, Y., Zhang, X., Chen, X., Tsang, K.F.: Manifold-guided geometric enhancement for reliable multimodal interaction in consumer electronics. IEEE Transactions on Consumer Electronics 72(1), 1156–1163 (2026)

41. You, X., Yang, R., Zhang, M., Zhang, H., Li, H., Yu, Y., Yang, J., Navab, N., Gu, Y.: Towards voxel spacing consistency for medical image segmentation (2026)

42. Yu, L., Wang, S., Li, X., Fu, C.W., Heng, P.A.: Uncertainty-aware self-ensembling model for semi-supervised 3d left atrium segmentation. In: MICCAI. vol. 11765, pp. 605–613 (2019)

43. Zeng, Y., Chen, X., Li, Z., Zhang, X., Wang, M., Dong, Y., Zhang, X., Pun, C.M., Tsang, K.F.: Dynroutenet: Lightweight facial shadow removal with mask-guided dynamic routing. IEEE Transactions on Consumer Electronics 72(1), 1193–1200 (2026)

44. Zhang, X., Zhu, Y., Zhao, Q., Yuan, X., Cao, J., Chen, X.: Tokenpenalty: Alleviating attention sinks and positional decay in lvlms. In: ACL Findings. pp. 12894–12906 (2026)