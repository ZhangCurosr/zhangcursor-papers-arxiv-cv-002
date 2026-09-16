# Lesion-centered 3D mapping of colonoscopy procedures: validation of a hierarchical ensemble pipeline on public benchmark videos

Hyunjun Kima,d, Hyeonwoo Nab,d, Jaewoo Leec,d,\*

aSchool of Computing, KAIST, Daejeon, Republic of Korea

bDivision of Mechanical and Space Engineering, Faculty of Engineering, Hokkaido University, Sapporo, Japan

cCHA University School of Medicine, Seongnam, Republic of Korea dClinical Imaging Research Institute, Seoul, Republic of Korea

## Abstract

Background and Objective: Colonoscopy recording practice preserves text reports and still photographs, while the spatial information already present in the recorded video — where the scope traveled, where a lesion was observed, and whether the same lesion was seen again — is discarded when the procedure ends. This study determines whether a lesion-centered spatial record can be assembled and validated without full-colon 3D reconstruction. Methods: A four-layer hierarchical pipeline was assembled (1) a global topological map, (2) lesion-level spatio-temporal tracks, (3) ondemand local 3D reconstruction, and (4) persistent lesion identity across repeated observations — and ran end to end on four public videos (two C3VDv2 sequences with ground-truth depth and two full REAL-Colon procedures; 40,245 frames). All components are published, individually validated methods; the contribution is their lesion-centered assembly, linking rules, and evaluation. Results: Revisits, impossible under forward-only mapping by construction, were detected by entry-map Bayesian localization: 5,614 and 4,043 revisit events (56 and 68 distinct nodes) in the two full procedures. Lesion-identity merging at the adopted threshold 0.5 maintained ground-truth purity 1.0 while auto-merging 20 of 231 candidate pairs. The

endoscopy-specific geometry engine outperformed a general-purpose foundation model on all metrics (overall absolute relative error (AbsRel) 0.2276 vs. 0.3523). Conclusions: The results are partial but establish a concrete near-term path: revisit detection, lesion identity, and local 3D each returned quantitative, reproducible output without waiting for complete geometric reconstruction; validating the record on clinical data is the next step.

Keywords: Colonoscopy, Imaging informatics, Topological mapping, Lesion tracking, 3D reconstruction, Revisit detection

## 1. Introduction

Colorectal cancer remains among the most incident cancers worldwide [1], and colonoscopy is the central examination for screening, surveillance, and endoscopic resection [2]. During a procedure, the endoscopist traverses the colonic segments during insertion and withdrawal, detects lesions, and observes, resects, and records them. What persists after the procedure is a narrative findings description and a small number of still photographs — the documentation form codified in colonoscopy quality-indicator guidance [2]. The recorded video itself carries where the scope passed, where and from which viewing angle a lesion was observed, and how many times the same lesion reappeared — but this spatial information is discarded at the end of the examination.

The gap is a recording-culture problem before it is a technology problem. Pathology results persist as standardized reports, whereas lesion location depends on narrative descriptions such as "distal transverse colon," and inaccurate preoperative localization of colonic neoplasia is a documented source of surgical management error [3]. Locating a previously treated lesion at surveillance colonoscopy relies on the earlier video and the physician's memory, which makes between-examination comparison and longitudinal lesion tracking difficult [2]. A fraction of lesions is additionally missed even within a single examination [4], so the evidentiary record that video could provide is lost precisely where completeness matters.

The technical landscape is nonetheless ready to answer this question: public collections now provide complete calibrated procedures [5], real multicenter recordings [6], and phantom sequences with dense ground truth [7]. Recent work in endoscopic 3D reconstruction and place recognition has advanced rapidly, but each line of research pursued a different goal, and no study has asked what these components deliver when assembled around lesions. This study addresses that question: under realistic conditions where tracking breaks frequently [8] and tissue deforms [9], can a 3D map that preserves (1) which segments were traversed, (2) where lesions were observed, and (3) whether the same lesion is being seen again be produced without reconstructing the whole colon as a single precise 3D model?

Several research threads lead to this position. Rigid-motion visual simultaneous localization and mapping (SLAM) fails in real procedures, where peristalsis, haustral-fold occlusion, washing fluid, specular reflections, and fast camera motion break feature tracking [8]. Benchmark studies on exvivo and synthetic data report the same failure modes [10], and deformable tissue motivated dedicated non-rigid SLAM formulations [9, 11]. CudaSIFT-SLAM [8] countered this with GPU SIFT features and a multi-map backend that restarts a submap at every tracking loss and merges submaps over common views; it is one of the few systems to process a full real procedure in real time, yet its authors report that close observation covered only about 38% of the procedure — evidence both of how difficult complete millimeter-scale mapping is and of the need for a higher structure that manages submap fragmentation. ColonSLAM [12] changed the objective from complete geometry to a graph of places, combining global descriptors from a place-recognition network with order priors (the endoscope does not travel backward in the colon) to organize an entire procedure as one topological graph; its verification stage uses LightGlue matching [13]. ColonMapper [14] extended place recognition across examinations with deeply learned global descriptors and a Bayesian filter, and its two-phase protocol — build the map during insertion, localize withdrawal frames against that map — provides the conceptual bridge between within-procedure revisits and across-examination alignment.

On geometry, monocular depth estimation has progressed from singleframe networks to temporally consistent streaming models [15] and to densification approaches that scale-align dense depth to sparse SLAM submaps [16]; integrated online foundation models that output pointmaps, depth, and camera parameters [17], together with general-purpose geometry transformers such as VGGT [18], now offer alternatives to hand-assembled modules. Related pipelines such as PERSEUS [19] and Semantic-SuPer [20] demonstrate semantic 3D reconstruction in endoscopic scenarios. On lesion identity, SALI established the view that lesions are spatio-temporal objects rather than per-frame detections, validated on the large-scale SUN-SEG video polyp segmentation benchmark [21, 22], and promptable segmentation models such as SAM 2 propagate an operator click or box through the entire video [23]. What remains beyond tracking is identity: when propagation loses the target the masklet ends, so a re-observed lesion starts a new track [23], and deciding whether two tracks show the same lesion becomes a separate problem [21].

This study contributes the following. First, it organizes these separately developed components — endoscopic SLAM, topological mapping, depth estimation, video lesion tracking — from a lesion-centered mapping perspective, and states explicitly what each solved and what each left open. Second it proposes a four-layer hierarchical ensemble (global topological map, lesion tracks, selective local 3D, persistent lesion ID) with defined inter-layer linking rules, and runs the entire pipeline end to end on four public videos, reporting what holds and what does not. Third, it identifies the conditions for clinical adoption and the remaining problems, framing the algorithmic contributions planned as follow-up work. The paper reports the implementation and validation of this ensemble on public benchmarks; it does not claim new base algorithms.

## 2. Materials and Methods

## 2.1. Lesion-centered hierarchical ensemble pipeline

The proposed framework takes a recorded procedure video as input and processes it in four layers (Fig. 1). The first layer is the global topological map: topological mapping in the ColonMapper family [14] organizes the procedure route into a node graph, and a Bayesian localization phase assigns withdrawal-phase frames to the insertion-phase map to detect revisits. The second layer is lesion tracking: SAM 2 [23] is seeded from an operator box (or dataset frame boxes) and propagates masklets, producing lesion-level spatiotemporal tracks with representative frames. The third layer is selective local 3D: point clouds are computed with Endo3R [17] only over intervals where tracks exist, not over the whole procedure. The fourth layer is persistent lesion identity: track pairs are scored by combining appearance embeddings of representative frames (EndoFM [24]) with node co-assignment and temporal separation, and each pair is classified into auto-merge, review queue, or reject.

The boundary of the contribution is stated explicitly. Each layer's components are public implementations of existing research; the contribution is the layered structure with its registry contracts and the inter-layer linking rules — node context of tracks, track geometry as the trigger for selective reconstruction, and the pairwise identity score.

![](images/1af4ec771933b64a823cb4823a52359ce0dd54368e5049b25d42729e85ae5bf6.jpg)  
Figure 1: Lesion-centered hierarchical ensemble pipeline. Four layers — global topological map, lesion tracks, selective local 3D, and persistent lesion identity — with the inter-layer linking rules (node context, track geometry, candidate track pairs). Each layer carries one representative output: a re-localization pair (withdrawal frame and matched entry-map node, REAL-Colon), the first/representative/last frames of one lesion track with segmentation contours, a local 3D point cloud, and a ground-truth-verified pair of fragmented tracks of the same lesion, compared by pairwise scoring with the lesion contoured in both views (REAL-Colon); the output card sketches the record fields with placeholder bars.

## 2.2. Data

Four public videos with complementary characteristics were selected (Table 1). The two C3VDv2 sequences [7, 25] are realistic phantom recordings accompanied by ground-truth depth and serve as the quantitative reference for local 3D (tier t1); the cecum sequence provides the depth reference point and the transverse sequence adds a deforming, non-rigid condition. The two REAL-Colon videos [6] are full real procedures and serve as integrated cases for topology, revisit detection, and identity (tier t2); one of them contains a long re-observation gap. The total expansion is 40,245 frames.

Lesion prompts were operator boxes for t1 and REAL-Colon frame boxes in PASCAL VOC XML format for t2. Ground-truth (GT) lesion identifiers and ground-truth depth were used for evaluation only and never entered any pipeline decision — the leakage prohibition is maintained from experiment design through reporting. The pipeline ran in seven stages (frame expansion, topology, lesion tracking, geometry, identity, reporting, quantitative evaluation) and completed in about 4.3 hours on a single GPU.

Table 1: Experiment data — four public videos
<table><tr><td>Slot</td><td>Dataset, video</td><td>Frames (expanded)</td><td>Role</td></tr><tr><td>c3vdv2-cecum</td><td>C3VDv2 c1 cecum t1 v2</td><td>423 (stride 1)</td><td>t1: GT-depth reference</td></tr><tr><td>c3vdv2-transverse</td><td>C3VDv2 c2_transverse1_t4_v3 382 (stride 1)</td><td></td><td>t1: non-rigid condition</td></tr><tr><td>realcolon-1</td><td>REAL-Colon 001-004</td><td>45,663 → 22,832 (stride 2)</td><td>t2: full real procedure</td></tr><tr><td>realcolon-2</td><td>REAL-Colon 002-008</td><td>33,216 → 16,608 (stride 2) t2: re-observation gap</td><td></td></tr></table>

## 2.3. Evaluation protocols

Revisit detection. Following the two-phase protocol of ColonMapper [14], the map is cut at the end of insertion (entry map; cut at the median registry index) and deployed frames are localized against it. The final configuration uses the authors' default gate of 0.5 [14] (entry\_tp050); a laxer gate of 0.33 and a full-map variant are reported alongside for comparison. A revisit event is a frame-to-node assignment whose temporal gap exceeds 150 frames.

Lesion identity. Candidate track pairs from the integrated run are scored, and an auto-merge threshold grid is evaluated against ground-truth lesion identity: the number of auto-merged pairs, review-queue size, rejects, groundtruth purity of merged clusters, and same-lesion recall. The default threshold 0.75 and the adopted threshold 0.5 are reported side by side, with the reject threshold fixed at 0.35.

Local 3D. On the two t1 sequences, 32 frames per video (linearly spaced) are evaluated with per-frame median scale alignment against ground-truth depth over valid ground-truth pixels, reporting absolute relative error (AbsRel), root-mean-square error in millimeters (RMSE), and the fraction of pixels with relative error below 0.25 (δ < 1.25). The endoscopy-specific online engine Endo3R [17] is compared with the general-purpose foundation model VGGT [18].

Qualitative track quality. All tracks are classified by a rule-based proxy (blur from a sharpness percentile, washing from a specular-highlight percentile, drift from mask-area discontinuities, and local-3D feasibility from point-cloud scores and extent), with the threshold constants recorded in the released classification artifacts for reproducibility.

Table 2: Revisit detection — mapping alone vs. entry-map localization (entry\_tp050, final run)
<table><tr><td>Video</td><td>Method</td><td>Localized</td><td>Events (&gt;150)</td><td>Nodes</td><td>Max gap</td></tr><tr><td>realcolon-1</td><td>mapping only</td><td></td><td>0</td><td>0</td><td></td></tr><tr><td>realcolon-1</td><td>entry_tp050</td><td>0.8011</td><td>5,614</td><td>56</td><td>22,203</td></tr><tr><td>realcolon-2</td><td>mapping only</td><td></td><td>0</td><td>0</td><td></td></tr><tr><td>realcolon-2</td><td>entry_tp050</td><td>0.7794</td><td>4,043</td><td>68</td><td>15,630</td></tr><tr><td>c3vdv2-cecum (control)</td><td>entry_tp050</td><td>1.0</td><td>56</td><td>1</td><td>261</td></tr><tr><td>c3vdv2-transverse (control)</td><td>entry_tp050</td><td>1.0</td><td>34</td><td>1</td><td>218</td></tr></table>

Gaps are in registry-frame units; a revisit requires a gap > 150. The t1 controls have small graphs (3–5 nodes) and only a single revisit node, separating clearly from the t2 procedures.

## 3. Results

## 3.1. Revisit detection

A structural fact comes first. Topological mapping grows only forward and compares each incoming frame only with the current proto-node, so revisits cannot arise in the mapping output by construction [14, 12]. This was confirmed empirically: in both full procedures (176 and 200 nodes), the mapping phase alone produced zero revisit nodes. Revisit detection is the responsibility of the localization phase (Fig. 3).

Table 2 and Figs. 2–4 summarize the outcome. The largest gaps (15,630 and 22,203 frames) correspond to end-of-withdrawal frames assigned to the earliest insertion nodes, which agrees with the endoscope re-passing the rectum (Fig. 2). The t1 controls were insensitive to the gate (identical results at gates 0.33 and 0.5), whereas on t2 the laxer gate 0.33 raised the localized fraction to 0.94-0.95 at the cost of possible over-assignment; the reported configuration therefore adopts the authors' default gate 0.5 [14], with all variants disclosed. Entry-map cutting produced 2.1-2.7× more events than full-map localization, confirming the advantage of the entry-map protocol (Fig. 4).

## 3.2. Lesion identity

The identity target is 231 candidate pairs (106 same-lesion, 125 differentlesion by ground truth). Score medians are 0.4612 for same-lesion and 0.3895 for different-lesion pairs — separation exists but is narrow (Fig. 6).

![](images/3c648071b4a488d9b4a84207cbe9152e911ee1a4f32866555920b42595021807.jpg)

![](images/2b8f3f8fb35dff3f538d519a5c67b0ed9638e0c3f61e2b6d8dc4c2a2df4e18fc.jpg)  
Deployed frame index (stride 2)  
Figure 2: Frame-to-entry-map node assignment over the full procedure ((a) REAL-Colon 001-004, (b) REAL-Colon 002-008); the dashed line marks the entry cut at the end of insertion. Vermillion points are revisit-flagged assignments whose temporal gap exceeds 150 frames — insertion-phase re-localization jumps as well as withdrawal re-passages. The assignment of the late-withdrawal tail to early insertion nodes is physically consistent with re-passage through the rectum.

The threshold analysis (Table 3, Fig. 5) shows that 0.5 is the largest threshold that keeps purity at 1.0; below it, mixing appears at 0.45 and collapses at 0.40. The 20 auto-merged pairs at the adopted threshold contain no ground-truth mixing, so proposals contributed only to shrinking the review queue. The node component of the score is zero throughout — under forward-only mapping, node co-assignment between tracks from different times cannot occur — and this absence is the structural origin of the recall ceiling of 0.19. The distributions (Fig. 6) show broad overlap with only the appearance component contributing to separation. Under the default threshold, ground-truth lesions fragment into $6 / 7 / 1 0 / 1 / 6$ clusters: repeated observations of the same lesion remain separate, a conservative behavior documented as a limitation.

## 3.3. Local 3D

On the t1 quantitative comparison (Table 4, Fig. 7), the endoscopyspecific online engine (Endo3R [17]) outperformed the general-purpose foundation model (VGGT [18]) on both videos and on all summary metrics; on the transverse sequence the $\delta < 1 . 2 5$ values are effectively tied (0.535 vs. 0.5324).

![](images/f7a69709ab2620893fda29a075bf4272722d062906f3f79210a754a71ecf5069.jpg)  
Figure 3: Forward-only node growth of the mapping phase. The node count increases monotonically with procedure progress and never returns — the structural reason why mapping alone yields zero revisits.

The weakness on the transverse sequence is structural: a mid-sequence abrupt scale drop and a sharp AbsRel increase over the last five frames were observed, indicating the limit of online monocular depth under strong nonrigid deformation, the difficulty that motivates non-rigid formulations [9, 11].

## 3.4. Qualitative behavior

Rule-based quality classification of the 33 tracks yielded one clean track and 32 flagged tracks (mask instability 31, blur risk 13) (Fig. 8). Mask instability dominates on REAL-Colon while the C3VDv2 cecum track is clean — the difficulty of real procedures is reflected directly. Figures 9–12 show lesion-track examples on both tiers, re-observation frame pairs that ground the revisit detections of Section 3.1, and a local 3D point cloud produced by the selective reconstruction layer.

![](images/285d048c0516d04b876c117d4ce058c69bc1f13ceb2c334077a9c69c4ffb872e.jpg)  
Figure 4: Revisit-detection variant comparison: entry-map vs. full-map localization at gate thresholds 0.33 and 0.50 (a revisit event has a temporal gap > 150 frames). The grey band marks the adopted configuration (entry map, gate 0.50). Cutting the map at the end of insertion yields substantially more revisit events than the full map (5,614 vs. 2,684 and 4,043 vs. 1,473), supporting the two-phase protocol of ColonMapper [14].

## 3.5. What the validation establishes

First, layer separation creates a structure in which each question becomes answerable: zero revisits under mapping alone is not a failure but a structural fact, and attaching the localization phase surfaced thousands of revisit events in the same videos. Second, weak inter-layer linking limits identity: the zero node component and the resulting recall ceiling expose the missing connection between forward-only mapping and the identity score. Third, the difficulty is uneven across layers — local 3D is already usable on phantom data, whereas mask instability and non-rigid deformation in real procedures remain open challenges for both the track and geometry layers [23, 11].

## 4. Discussion

## 4.1. Clinical relevance of lesion-centered maps

A lesion-centered map offers three values: during look-back exploration it provides the observation positions and viewpoints of the previous examination as spatial context; during surveillance it creates continuity of lesion-level records [3]; and for documentation standardization it adds a spatial axis to procedure records [2]. Missed lesions remain a measured concern in practice [4], and a record that preserves which segments were traversed and in what order is a prerequisite for auditing completeness. In this validation, both revisit detection and auto-merging operated at the level of candidate proposal, which matches the confirmation-loop structure in which the final decision rests with the physician.

Table 3: Lesion identity — default threshold 0.75 vs. adopted 0.5 (231 candidate pairs, reject fixed at 0.35)
<table><tr><td>Auto-merge</td><td>Merged</td><td>Queue</td><td>Rejected</td><td>GT-mixed</td><td>Purity</td><td>Same-GT recall</td></tr><tr><td>0.75 (default)</td><td>1</td><td>214</td><td>16</td><td>0</td><td>1.0</td><td>0.0094</td></tr><tr><td>0.5 (adopted)</td><td>20</td><td>195</td><td>16</td><td>0</td><td>1.0</td><td>0.1887</td></tr><tr><td>0.45</td><td>61</td><td>154</td><td>16</td><td>1</td><td>0.9836</td><td>0.566</td></tr><tr><td>0.40</td><td>141</td><td>74</td><td>16</td><td>2</td><td>0.617</td><td>0.8208</td></tr></table>

The adopted threshold 0.5 is the largest value that keeps purity 1.0 (first mixing at 0.45, collapse at 0.40). Under the default threshold, cluster counts per ground-truth lesion are $6 / 7 / 1 0 / 1 / 6 - \mathrm { a }$ fragmented, conservative behavior.

Table 4: Local 3D quantitative results — Endo3R vs. VGGT (32 frames per video, perframe median scale alignment)
<table><tr><td>Engine</td><td>Video</td><td>AbsRel</td><td>RMSE (mm)</td><td> $\delta < 1 . 2 5$ </td></tr><tr><td>Endo3R</td><td>c3vdv2-cecum</td><td>0.1376</td><td>21.66</td><td>0.8208</td></tr><tr><td>Endo3R</td><td>c3vdv2-transverse</td><td>0.3175</td><td>38.33</td><td>0.535</td></tr><tr><td>Endo3R</td><td>overall (mean of medians)</td><td>0.2276</td><td>29.99</td><td>0.6779</td></tr><tr><td>VGGT</td><td>c3vdv2-cecum</td><td>0.3408</td><td>50.58</td><td>0.3804</td></tr><tr><td>VGGT</td><td>c3vdv2-transverse</td><td>0.3638</td><td>44.96</td><td>0.5324</td></tr><tr><td>VGGT</td><td>overall (mean of medians)</td><td>0.3523</td><td>47.77</td><td>0.4564</td></tr></table>

The transverse $\overline { { \delta < 1 . 2 5 } }$ values are effectively tied; Endo3R leads on all remaining metrics.

## 4.2. The gap between public data and clinical data

The quality difference between the phantom tier and the real-procedure tier reflects that public benchmarks do not fully represent clinical imaging difficulty [7, 6, 5]: mask instability in 31 of 33 tracks means SAM 2 [23] propagation breaks far more often in real procedures, which directly affects the volume and fragmentation of identity candidates. Clinical adoption assessment requires validation on real procedure data, together with data acquisition and de-identification procedures. This gap also has an evaluation-methodology dimension: auditing work on AI benchmarks shows that reliable assessment requires checking the task, environment, ground truth, and evaluation components themselves, not only model outputs [26]; the leakage prohibition, threshold-variant disclosure, and run-level reproducibility of this study follow the same principle.

![](images/3b476104e3f6a54dfe7891270c6b014b7783efa6540b20ea7c6416ae7e4306f7.jpg)  
Figure 5: Auto-merge threshold grid — number of auto-merged pairs (bars) and groundtruth precision and recall (lines). The adopted threshold 0.5 is the largest value that maintains precision 1.0.

## 4.3. Limitations and ethical position

The scale of monocular depth is intrinsically uncertain — the motivation for scale-consistent endoscopic reconstruction [17] — and the local 3D of this framework must be read as relative structure. Non-rigid deformation, a standing difficulty for monocular endoscopy [11], degrades end-of-sequence accuracy, as the transverse case showed. Above all, every automated decision is a proposal, and the physician remains the final authority for confirming lesion records. The scale of this validation (four videos) is directional; no generalization claim is made.

![](images/14865b25419dc8d71e59e20645b930d490b7c64231e67e0fd8da0d91b20a6f6e.jpg)

![](images/acc0fa4a6905cf9331d450d6913162c94014838a9b82eb9f6359e5799edf7636.jpg)  
Figure 6: Raw merge-score distributions over 231 candidate pairs — final score (a) and appearance component (b). Green: same ground-truth lesion (106 pairs); gray: different lesions (125). Dotted lines are medians; vermillion dashed lines are the reject 0.35 and auto-merge 0.50 thresholds. The distributions overlap broadly, and only the appearance component contributes to separation.

## 4.4. Future work

Four directions follow directly. (1) Feed localization-phase re-assignments back into the node component of the identity score to relax the recall ceiling. (2) Compare non-rigid formulations and dynamic scene representations (e.g., NR-SLAM [11], ColonSplat [27]) as alternatives for the local 3D layer. (3) Extend persistent lesion identity across examinations via cross-procedure alignment. (4) Replace operator prompts with semi-automatic detection to reduce operator burden. These directions are the algorithmic contributions planned as follow-up work.

## 5. Conclusion

To address the loss of spatial information in colonoscopy recording practice, this study proposed a lesion-centered hierarchical ensemble as an alternative to complete 3D reconstruction and ran it end to end on four public videos. Revisit detection answered with 5,614 and 4,043 events where mapping alone yields zero; lesion identity auto-merged 20 pairs at a threshold that keeps purity 1.0; and local 3D favored the endoscopy-specific engine on every metric. The results are partial but show that preserving the spatial context already present in procedure video, centered on lesions, is a genuinely open path. Strengthening the inter-layer links and validating on clinical data remain the tasks ahead.

![](images/d717e766daf92667828e006ea340efd87a76c4154715ecc4e912a6f68cba01b6.jpg)

![](images/5611b1088dbcab52b4e0b916ddc4cba9a3dcbde3892684ec8d82049d2e2bbc65.jpg)

![](images/f6adefbfc29150ca0a0b062567a73730aab2487944d47328f09f2afcd36013c5.jpg)  
Figure 7: Local 3D quantitative comparison on the t1 videos (Endo3R vs. VGGT): (a) AbsRel, (b) RMSE (mm), and (c) δ < 1.25 over 32 frames per video with per-frame median scale alignment. Overall denotes the mean of the per-video medians.

## CRediT authorship contribution statement

Hyunjun Kim: Conceptualization, Methodology, Software, Investigation, Data curation, Writing – original draft. Hyeonwoo Na: Conceptualization, Investigation, Data curation, Writing – review & editing. Jaewoo Lee: Conceptualization, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Ethics statement

This study used only publicly available, de-identified benchmark datasets (C3VDv2 and REAL-Colon). No participants were recruited and no new human data were collected by the authors; no ethics committee approval

![](images/180ae657dc82675928d488f86779f3059108e123e8a42c5097d693fce351f4b0.jpg)  
Figure 8: Track quality flag distribution (rule-based proxy classification).

was therefore sought because no human participants or personal data were involved.

## Data availability

The datasets analyzed are publicly available from their original publishers (C3VDv2 [25]; REAL-Colon [6]). Other research artifacts are available from the corresponding author on reasonable request.

## Code availability

The complete pipeline code, adapters, experiment configurations, and test suite are openly available at https://github.com/hyunjun1121/endovision-pipeline (Zenodo DOI https://doi.org/10.5281/zenodo.22136766).

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author(s) used ZCode, an AI coding assistant powered by the GLM large language model (Z.ai), in order

(a) First frame (t=384)  
![](images/5d8dc7ede84a2e38d845e09179de4bc7b36f707cade3241ae4c558acbf586bfc.jpg)

(b) Representative frame (t=389)  
![](images/c9175449dd916af992bf32e33492d3465d120ed8fe1ae1664a7e66afc6631755.jpg)

(c) Last frame (t=422)  
![](images/8eebb928e8678253a787e4c5c0f580811cec625ade24edf8f6c4a14de8d327a1.jpg)  
Figure 9: Lesion track examples on C3VDv2 (t1) — SAM 2 [23] mask outlines in red (two disconnected regions on the representative frame).

to assist with drafting and language editing of the manuscript text. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.

## Appendix A. Reproducibility note

The complete run is reproducible from a single configuration file, with the run snapshot and validation report stored alongside the artifacts (base-layer comparison run kci-4v-s03-001; final run kci-4v-s04-001). All quantitative figures (Figs. 2–8) are generated automatically from repository result JSON files, and the qualitative assets are extracted from the same run artifacts. Dataset licenses: C3VDv2, CC BY 4.0 [25]; REAL-Colon, CC BY-NC-SA 4.0 [6]. The complete pipeline code, adapters, configurations, and test suite are openly available (see the Code availability statement).

## References

[1] Bray F, Laversanne M, Sung H, Ferlay J, Siegel RL, et al: Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin 74:229-263, 2024

[2] Rex DK, Anderson JC, Butterly LF, Day LW, Dominitz JA, et al: Quality indicators for colonoscopy. Gastrointest Endosc 100:352-381, 2024

[3] Fernandez LM, Ibrahim RNM, Mizrahi I, DaSilva G, Wexner SD: How accurate is preoperative colonoscopic localization of colonic neoplasia? Surg Endosc 33:1174-1179, 2019

GT lesion 001-004\_1 . longest fragment: track\_0004 · 6 fragments total

![](images/aff5765d135553adbf39287b175f2190c81d4bfbef629d5537f5dee37076937c.jpg)

![](images/cea20bffb14a091764a02c237cd1bdb5e977e4754b7b8170ea97c88895bbbe02.jpg)

![](images/737e957bf489c67582d4fda59fced5dabea09832b7772187ff01c786f41c161f.jpg)

![](images/16d468cdbedace45212869b7a7d0b0743f270e2989cd6ce0d436c045f1670c11.jpg)

![](images/9f28b7df88fca8bd4de295e6e166e2baa0be8c142d474dab29d0f6013160e6f3.jpg)

![](images/46b792ab50625eb053b7b1ee689ade9284ec5fe855a576f7612dcf03cec172f1.jpg)  
Figure 10: Lesion track examples on REAL-Colon (t2) — the longest track per groundtruth lesion, shown together with fragmentation.

[4] Zhao S, Wang S, Pan P, Xia T, Chang X, et al: Magnitude, risk factors, and factors associated with adenoma miss rate of tandem colonoscopy: a systematic review and meta-analysis. Gastroenterology 156:1661-1674, 2019

[5] Azagra P, Sostres C, Ferrández Á, Riazuelo L, Tomasini C, et al: EndoMapper dataset of complete calibrated endoscopy procedures. Sci Data 10:671, 2023

[6] Biffi C, Antonelli G, Bernhofer S, Hassan C, Hirata D, et al: REAL-Colon: a dataset for developing real-world AI applications in colonoscopy. Sci Data 11:539, 2024

[7] Bobrow TL, Golhar M, Vijayan R, Akshintala VS, Garcia JR, Durr NJ: Colonoscopy 3D video dataset with paired depth from 2D-3D registration. Med Image Anal 90:102956, 2023

[8] Elvira R, Tardós JD, Montiel JMM: CudaSIFT-SLAM: multiple-map visual SLAM for full procedure mapping in real human endoscopy. arXiv preprint arXiv:2405.16932, 2024

![](images/77a1a9c99e9b61381f3928b09989eabd4c113488d62f4c10340bf399601f68d9.jpg)  
Figure 11: Re-observation frame pairs — (a) and (b) are algorithmically re-localized matched pairs, each pairing a withdrawal-phase query frame (left) with the insertion-phase node keyframe to which it was re-localized (right). p is the Bayesian node-assignment posterior probability under the adopted entry-map gate of 0.5. No photometric modification was applied to the frames. The figure is a qualitative example of localization output, not a claim of anatomical same-place ground truth.

[9] Lamarca J, Parashar S, Bartoli A, Montiel JMM: DefSLAM: tracking and mapping of deforming scenes from monocular sequences. IEEE Trans Robot 37:291-303, 2021

[10] Ozyoruk KB, Gokceler GI, Bobrow TL, Coskun G, Incetan K, et al: EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos. Med Image Anal 71:102058, 2021

[11] Gómez Rodríguez JJ, Montiel JMM, Tardós JD: NR-SLAM: nonrigid monocular SLAM. IEEE Trans Robot 40:4252-4264, 2024

![](images/4fd09627306c1a11a70370b6d6841b8f616f2cc3c4872bdecdf822b22e3836b1.jpg)  
Figure 12: Selective local 3D reconstruction over a lesion-track interval in the C3VDv2 cecum sequence using Endo3R [17]. Points are colored by the depth-axis coordinate in relative units. The reconstruction is shown as relative geometry rather than metric scale.

[12] Morlana J, Tardós JD, Montiel JMM: Topological SLAM in colonoscopies leveraging deep features and topological priors. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2024, Lecture Notes in Computer Science, vol 15011. Cham: Springer; 2024, pp. 733-743

[13] Lindenberger P, Sarlin PE, Pollefeys M: LightGlue: local feature matching at light speed. In: IEEE/CVF International Conference on Computer Vision (ICCV); 2023, pp. 17627-17638

[14] Morlana J, Tardós JD, Montiel JMM: ColonMapper: topological mapping and localization for colonoscopy. In: IEEE International Conference on Robotics and Automation (ICRA); 2024, pp. 6329-6336

[15] Li H, Lu D, Wang J, Webster RJ, Oguz I: EndoStreamDepth: temporally consistent monocular depth estimation for endoscopic video streams. In: Medical Imaging with Deep Learning (MIDL), Proceedings of Machine Learning Research 315:1697-1721, 2026

[16] Anadón X, Rodríguez-Puigvert J, Montiel JMM: 3D densification for multi-map monocular VSLAM in endoscopy. arXiv preprint arXiv:2503.14346, 2025

[17] Guo J, Dong W, Huang T, Ding H, Wang Z, et al: Endo3R: unified online reconstruction from dynamic monocular endoscopic video. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2025; 2025, pp. 170-180

[18] Wang J, Chen M, Karaev N, Vedaldi A, Rupprecht C, Novotný D: VGGT: visual geometry grounded transformer. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); 2025, pp. 5294-5306

[19] Acar A, Li F, Stern SS, Al-Zogbi L, Li H, et al: Perseus: perception with semantic endoscopic understanding and SLAM. Int J Comput Assist Radiol Surg, 2026. doi:10.1007/s11548-026-03717-w

[20] Lin S, Miao AJ, Lu J, Yu S, Chiu ZY, Richter F, Yip MC: Semantic-SuPer: a semantic-aware surgical perception framework for endoscopic tissue identification, reconstruction, and tracking. In: IEEE International Conference on Robotics and Automation (ICRA); 2023, pp. 4739- 4746

[21] Hu Q, Yi Z, Zhou Y, Peng F, Liu M, Li Q, et al: SALI: short-term alignment and long-term interaction network for colonoscopy video polyp segmentation. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2024; 2024, pp. 531-541

[22] Ji GP, Xiao G, Chou YC, Fan DP, Zhao K, et al: Video polyp segmentation: a deep learning perspective. Int J Autom Comput 19:531-549, 2022

[23] Ravi N, Gabeur V, Hu YT, Hu R, Ryali C, et al: SAM 2: segment anything in images and videos. In: International Conference on Learning Representations (ICLR); 2025

[24] Wang Z, Liu C, Zhang S, Dou Q: Foundation model for endoscopy video analysis via large-scale self-supervised pre-train. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2023; 2023, pp. 101-111

[25] Golhar MV, Galeano Fretes LS, Ayers L, Akshintala VS, Bobrow TL, Durr NJ: C3VDv2 — colonoscopy 3D video dataset with enhanced realism. arXiv preprint arXiv:2506.24074, 2025

[26] Suh H, Ji B, Lee S, Khare R, Khan B, et al: AgentSuite: toward more reliable agent evaluation with a component-based benchmark auditing pipeline. OpenReview preprint 2Exmr1eIKZ (ICML 2026 submission), 2026

[27] Smolak-Dyżewska W, Kaleta J, Dall'Alba D, Spurek P: ColonSplat: reconstruction of peristaltic motion in colonoscopy with dynamic Gaussian splatting. arXiv preprint arXiv:2603.06860, 2026