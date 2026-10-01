# From Wrecks to Wisdom: Recovering Crash Mechanics from Real-World Multi-View Photos

Ondřej Valach<sup>1[0009−0000−7629−0516]</sup> <sup>⋆</sup>, Václav Diviš<sup>1[0000−0001−9935−7824]</sup>, and Ivan Gruber<sup>1[0000−0003−2333−433X]</sup>

University of West Bohemia, Faculty of Applied Sciences, Department of Cybernetics and New Technologies for the Information Society valacho@fav.zcu.cz, vincie@kky.zcu.cz, grubiv@ntis.zcu.cz

Abstract. Estimating accident mechanics from real-world crashes is important for vehicle-safety analysis, injury modeling, and crash-severity prediction. It can also support operational workflows such as insurance claim triage. In standard crash records, key metadata such as impact configuration, principal direction of force, and change in velocity (∆V) may be missing, delayed, or corrupted. By contrast, post-crash photographs are widely available and contain rich visual evidence of deformation. This raises the following question: how much crash mechanics information can be recovered directly from vehicle photos when structured signals are absent? To study this, we formulate crash understanding as supervised prediction from per-case multi-view photo sets. Targets include six Collision Deformation Classification (CDC) descriptors and the longitudinal/lateral components of reconstructed ∆V. Each photo is encoded by a shared visual backbone, and the resulting view-level features are fused into a single case-level representation from which targetspecific heads predict crash descriptors. Using 15.2k training cases from the Crash Investigation Sampling System (≈1.5M photos before filtering), 1.15k validation cases, and 1.15k test cases, we define an evaluation protocol for vision-based crash descriptor estimation from incomplete multi-view evidence. Post-crash imagery alone provides usable signal for several non-trivial crash-mechanics descriptors, while weakly observable and long-tailed targets remain challenging. Within the compared training regimes, the selected joint-training recipe improves several contextdependent targets under this protocol, reducing mean absolute angular error for principal direction of force from 20.1<sup>◦</sup> to 14.05<sup>◦</sup> and lowering longitudinal ∆V MAE from 8.04 to 7.45 km/h. Our work provides a reference point for future multimodal fusion with structured crash metadata.

Keywords: crash severity prediction · crash mechanics · post-crash imagery · multi-view learning · computer vision · deep learning · trafic safety · Collision Deformation Classification · ∆V estimation · multitarget prediction

![](images/fb41d8a8541e327b74c387002c3fcbd4da62b4f093535c3f303a975a36411955.jpg)  
Fig. 1. Comparison of the selected joint-training and single-task reference pipelines. Performance is shown as relative change (%) with respect to the corresponding selected single-task result for each target.

## 1 Introduction

Trafic accidents remain a major public health and safety challenge, causing high rates of fatalities and injuries. Improving the analysis and prediction of crash outcomes is crucial for enhancing road safety, reducing casualties, and supporting more reliable crash reconstruction and interpretation. Active and passive safety systems, such as adaptive airbags, electronic stability control, and automated collision-avoidance systems, have improved substantially over the last decades. However, assessing how these systems perform in real-world crashes still requires accurate reconstruction of the underlying collision mechanics, information that is often absent or incomplete in standard crash records. Moreover, the parameters that describe these mechanics are often not directly measurable in real-world crashes, and many lower severity collisions go unreported (e.g., because they do not require a vehicle tow) [10].

Beyond accident mechanics analysis itself, these crash descriptors are important inputs to downstream risk-modeling and triage pipelines. Unfortunately, they are typically available only after Event Data Recorder (EDR) extraction or crash reconstruction. By contrast, photographs are often among the earliest and most consistently available information channels. This motivates our study of image-based estimation from post-crash photo sets. In this work, we investigate how much of the underlying kinematic information can be recovered from images when structured signals are missing or delayed, and whether such estimates can serve as inputs to injury and severity models that would otherwise rely on structured metadata.

Furthermore, we define crash understanding as multi-target prediction from post-crash photo sets. We predict deformation descriptors derived from the Collision Deformation Classification (CDC) code [16] (Table 1) together with directional ∆V components. These targets capture complementary aspects of collision geometry and dynamics and are commonly used as informative inputs to downstream injury and severity models. To study this setting, we use a unified multi-view architecture built on a SwinV2 backbone [9] with case-level fusion. We compare separate single-task training with a joint-training pipeline under a common input representation and evaluation protocol. As previewed in Figure 1, the selected joint-training configuration improves several of the most contextdependent targets.

In summary, this paper makes three contributions: (i) it introduces a CISSbased evaluation protocol for crash descriptor estimation from real-world, naturally incomplete multi-view post-crash photo sets; (ii) it defines a preprocessing and case-construction pipeline for mapping heterogeneous crash photographs into fixed canonical view slots with masked supervision for missing labels; and (iii) it reports single-task and joint-training reference results for CDC-derived crash descriptors and reconstructed ∆V components. Our goal is not to propose a new vision backbone, but to establish a controlled and reproducible reference point for image-based crash-mechanics estimation and future multimodal fusion with structured crash metadata.

## 2 Related Work

Most prior crash-severity and injury prediction pipelines operate on structured crash records rather than images. Across this literature, kinematic variables such as change in velocity (∆V), principal direction of force, and peak acceleration are consistently treated as informative predictors alongside vehicle, occupant, roadway, and environmental factors [7,1,15,13,11,5]. In particular, prior studies identify ∆V and related crash-mechanics descriptors as important signals for injury estimation and severity modeling [7,1]. However, these quantities are usually assumed to be available from event data recorders, reconstruction pipelines, or investigator-coded records, rather than inferred directly from visual evidence.

A smaller line of work addresses the missing-variable problem itself. Statistical reconstruction methods are used to correct reporting bias and infer unobserved crash quantities from structured records [10]. More broadly, most severity-prediction research still learns from tabular crash databases, police reports, and coded driver/vehicle/roadway variables, using both classical statistical models and modern machine-learning methods [13,15,11,5]. In these settings, crash-mechanics variables serve as inputs when available, not as prediction targets. Our focus is diferent: we study whether such descriptors can be recovered directly from post-crash photographs when structured signals are absent, delayed, or incomplete.

An even more limited amount of work attempts to infer crash characteristics directly from post-crash images. Existing image-based methods typically focus on a narrow output space such as impact location or ∆V, curated single-image settings, or supplement images with synthetic data or vehicle metadata [18,6]. The closest studies are Silver et al. [18], who combine real and synthetic crash images, and Hasija et al. [6], who use a curated frontal-only subset together with vehicle metadata. In contrast, we study real-world crash cases represented by naturally incomplete multi-view photo sets and predict multiple CDC-derived descriptors together with reconstructed ∆V components. Our goal is not to claim direct comparability with these settings, but to provide a reference point for image-based crash understanding at real-world scale.

## 3 Dataset

Datasets suitable for this task are limited. The problem requires crash records with post-crash photographs together with supervisory targets that describe collision mechanics. Nonetheless, we identified three repositories that provide this combination at scale: NHTSA’s CISS/NASS-CDS in the United States, Germany’s GIDAS, and India’s RASSI [12,3,14].

We use NHTSA’s Crash Investigation Sampling System (CISS) dataset, because it is openly available, large, standardized, and richly documented with post-crash imagery. Although GIDAS and RASSI also pair crash records with photographs, we excluded GIDAS due to access constraints and did not use RASSI because its road and trafic environment difers substantially from our target setting. Each retained CISS case includes structured forms and typically dozens of scene and vehicle photographs. From the available records, we construct a CISS-based evaluation setup of 17.5k crash cases (≈1.5M photos before filtering).

For each case, the supervisory targets consist of an eight-character CDC code (investigator-coded) together with reconstructed WinSMASH ∆V components. WinSMASH is a crash reconstruction software, which works with scene evidence and vehicle damage information [17]. From the CDC code, we derive six targets: principal direction of force, deformation plane, longitudinal/lateral and vertical/lateral crush zones, damage distribution, and deformation extent. These targets and their evaluation metrics are summarized in Table 1.

We train on $\varDelta V _ { \mathrm { l o n g } }$ and $\varDelta V _ { \mathrm { l a t } }$ as regression targets, while noting that these labels are reconstruction-derived estimates rather than direct measurements.

The input to the model is an ordered multi-view photo set with up to nine canonical view slots: front, rear, right, left, front-right, front-left, rear-right, rearleft, and top. Cases are split at the crash-case level into disjoint train, validation, and test partitions to prevent leakage, with $N _ { \mathrm { t r a i n } } = 1 5 2 0 0 , N _ { \mathrm { v a l } } = 1 1 5 0$ , and $N _ { \mathrm { t e s t } } = 1 1 5 0$ . Because the target distributions are strongly long-tailed, reflecting the real-world crash statistics, we report both accuracy and tail-sensitive metrics such as macro-F1. The full label distributions are shown in the Electronic Supplementary Material (ESM), Figure 3.

![](images/23f6c609692bae59f81a1df9b7cadf01d8f9a8702a65a11130dba10a9c9b3c66.jpg)  
Fig. 2. Representative crash cases in the fixed multi-view input format. Each case is shown as an ordered subset of available viewpoints together with its available crash descriptor annotations (CDC-derived targets and ∆V when available). The descriptor denoted as ∆V represents the resulting compound vector and is included only for reader interpretation.

Figure 2 illustrates representative cases in the fixed multi-view input format used throughout training and evaluation. Real-world crash photo sets also present several practical dificulties, including uninformative close-ups, missing viewpoints, and incomplete target annotations. We summarize these issues in Section 4.1, full preprocessing details are provided in the ESM, Section 2.

## 4 Methods

Our goal is to predict crash descriptors from post-crash photo sets, so the core of the model is a shared visual encoder followed by case-level multi-view fusion. We use SwinV2 as the backbone for all post-crash views for three reasons. First, it provides hierarchical representations, preserving local cues in early stages while aggregating broader context, which is useful when the evidence is subtle, spatially localized, or low-contrast against the vehicle body (e.g., cracks, panel gaps, broken lamps). Second, its window-based self-attention captures longer-range interactions while remaining computationally feasible at the image resolution used in this study compared to global-attention vision transformers [2]. Third, SwinV2 supports interpolation of pretrained positional information, allowing fine-tuning at a diferent input size while still benefiting from pretrained weights [9].

Each crash case is represented by up to nine post-crash views, with the image order assigned using the dataset’s viewpoint labels. When a canonical viewpoint is unavailable, the corresponding slot remains empty. This fixed ordering is important because it preserves viewpoint identity across cases and allows the fusion module to associate each feature vector with a consistent semantic position. We use metadata-based slots in the main experiments to keep crash descriptor prediction separate from possible viewpoint estimation errors. ESM Table 2 reports an automatic orientation-based slot assignment variant, which matches the metadata-based slots at the reported precision when averaged over three runs.

After the SwinV2 backbone, the architecture performs multi-view fusion, as shown in Figure 3. Each available view is processed independently by a shared SwinV2-base image encoder, producing one feature vector per view. These perview features are treated as a token sequence and fused using a lightweight transformer [19]: we prepend a learned fusion token and add learned positional embeddings so that the fusion module can exploit the consistent view ordering. After two transformer layers, the output is used as a single global representation of the crash case.

On top of the fused embedding, we study two decoder configurations. In the single-task setting, the fused representation is passed to a single multilayer perceptron (MLP) prediction head that outputs one target at a time. In the multi-task setting, the backbone and fusion module remain shared, but the fused representation is first mapped through a shared pre-head into a common hidden space and then passed to multiple task-specific heads. Thus, the two settings share the same case representation, image backbone, and fusion module, while difering in output heads and in training strategy.

![](images/f0476bfa86d24941b5b318184cb13fc617bd286fcd13733c67a5ca373e66ae6d.jpg)  
Fig. 3. Multi-view crash descriptor architecture: a shared SwinV2 encodes nine ordered views, a lightweight fusion transformer aggregates them into a case embedding, and a decoder predicts CDC descriptors and ∆V.

## 4.1 Preprocessing and Data Challenges

Real-world crash photo sets contain irrelevant close-ups, missing viewpoints, incomplete target annotations, and strongly imbalanced labels. We address these issues through vehicle and wheel filtering, fixed 9-slot view construction, viewdrop augmentation, masked losses for missing labels, and imbalance-aware sampling. Full preprocessing, curation, and missing-label details are provided in the ESM, Section 2.

Task Observability and Label Dificulty The targets difer substantially in visual observability. Impact plane and principal direction of force are often supported by the visible damage layout, whereas crush-zone descriptors and deformation extent may depend on subtle, occluded, or missing views. ∆V is the least directly observable target because it is a reconstructed kinematic quantity inferred only indirectly from post-crash appearance. Since labels are case-level and not localized to image regions, the model must identify useful evidence across heterogeneous multi-view inputs, which motivates joint training across related descriptors.

## 4.2 Method Setup

Preliminary ablations over backbones and view-fusion strategies favored independently encoded SwinV2 view tokens fused with a lightweight transformer. Details are reported in the ESM, Section 1. With this case representation fixed, the main study compares two practical training regimes: single-task models, where one descriptor is learned per model instance, and a joint-training model, where the shared trunk is trained with multiple task-specific heads.

## 4.3 Descriptor Targets and Task Formulations

Direction of Force The CDC principal direction of force (DoF) is modeled as a 12-class circular prediction problem. Since neighboring clock positions represent small angular deviations and opposite positions large ones, we replace one-hot labels with Gaussian-smoothed target distributions over circular distance and minimize KL divergence to the model log-probabilities. At evaluation, the predicted class is converted to an angle and scored by mean absolute angular error under minimal circular distance. We also tested sin(θ), cos(θ) regression, but circular classification performed better in preliminary experiments; see the ESM, Section 1.

CDC categorical and zone descriptors For non-circular CDC descriptors, impact plane, damage distribution, longitudinal/lateral zone, and vertical/lateral zone, we use cross-entropy loss with the same imbalance-aware sampling scheme. We report accuracy and macro-F1, the latter reflecting performance on underrepresented classes. Deformation extent (01–09) is ordinal, so we train it as 9-way classification with label smoothing and report exact accuracy, Acc@±1 for predictions within one neighboring grade, and MAE in bin units to capture near-miss errors.

∆V regression For $\varDelta V .$ , we jointly regress the longitudinal and lateral components, $\varDelta V _ { \mathrm { l o n g } }$ and $\varDelta V _ { \mathrm { l a t } }$ , using SmoothL1 (Huber) loss [4]. Because $\varDelta V$ is missing for 38% of cases, training and evaluation use only the remaining 62% with valid labels. We report MAE and RMSE per component; the compound $\varDelta V$ magnitude can be obtained as the Euclidean norm of the two predicted components.

## 4.4 Single-task

We use single-task training as a reference setting in which one descriptor is learned per model instance. All single-task models share the same case representation, image backbone, and fusion module, while the final prediction head is task-specific. This setup provides per-target reference results under a common input pipeline. Class imbalance is handled through task-specific sampling, and samples without the required target are skipped for that task. Full sampling and optimization details are provided in the ESM, Section 3.

## 4.5 Multi-task

Several labels describe overlapping aspects of the same physical event, so joint training is a natural practical baseline: a single shared representation can support all descriptor heads and may encourage the model to encode global crashconfiguration cues. We therefore compare independent single-task reference models with a joint-training configuration in which the SwinV2 backbone and fusion module are shared across targets. This comparison is intended to evaluate two practical training regimes under the same multi-view input protocol, not to isolate multi-task supervision as the only changing factor.

In the joint-training configuration, the fused case embedding is first passed through a shared pre-head MLP and then routed to task-specific heads, as shown in Figure 3. Missing labels are handled through task-specific loss masking, so each sample contributes only to the heads with valid annotations. Class imbalance is handled with a shared task-aware sampling policy, and loss balancing is stabilized with GradNorm and Dynamic Weight Averaging during training.

Because this configuration difers from the single-task reference not only in joint supervision but also in initialization, loss balancing, sampling, and optimization schedule, the comparison should be interpreted as a comparison between selected single-task and joint-training setups under a common input representation, not as a perfectly matched ablation of multi-task supervision alone. Full training details and hyperparameters are provided in the ESM, Section 3 and Table 3.

## 5 Experiments and Results

This section reports the main experimental results. All models were trained on a single GPU (NVIDIA A40/L40S, depending on the run). The target definitions are summarized in Table 1.

To contextualize class imbalance, Table 1 reports the share of the most frequent class among valid labels (Top1%). For categorical targets, this corresponds to the accuracy of an always-majority rule under the matched train/test label distributions. We therefore interpret accuracy together with macro-F1 where applicable and treat small diferences between selected training regimes cautiously.

Table 1. Target summary with missing-label rates (Miss%) and share of the most frequent class among valid labels (Top1%). Circ-Smooth-Cls: circular classification with Gaussian-smoothed targets and KL divergence [8]. Smooth-Cls: classification with label smoothing. Acc: accuracy. AngErr: mean absolute angular error (degrees). MAE: mean absolute error. Acc@±1: accuracy within one neighboring extent bin of the groundtruth label.
<table><tr><td>Target</td><td>Out</td><td>Classes</td><td>Miss%</td><td></td><td>Top1% Metrics</td></tr><tr><td>Impact plane</td><td>Classification</td><td>4</td><td>0.0</td><td>64.9</td><td>Acc</td></tr><tr><td>DoFclock</td><td>Circ-Smooth-Cls</td><td>12</td><td>2.9</td><td>46.1</td><td>AngErr</td></tr><tr><td>Long./lat. zone</td><td>Classification</td><td>9</td><td>0.0</td><td>38.3</td><td>Acc</td></tr><tr><td>Vert./lat. zone</td><td>Classification</td><td>7</td><td>0.0</td><td>89.8</td><td>Acc</td></tr><tr><td>Damage distribution</td><td>Classification</td><td>8</td><td>0.0</td><td>75.4</td><td>Acc</td></tr><tr><td>Deformation extent</td><td>Smooth-Cls</td><td>9</td><td>14.7</td><td>43.6</td><td>Acc@±1</td></tr><tr><td> $\Delta V _ { \mathrm { l o n g } } ~ / ~ \Delta V _ { \mathrm { l a t } }$ </td><td>Regression</td><td>-</td><td>38.1</td><td></td><td>MAE</td></tr></table>

Unless stated otherwise, test results are reported at validation-selected checkpoints: best per head for single-task training and best joint objective for joint training. Because the reported single-task and joint-training models use diferent selected training configurations, summarized in the ESM, Table 3, the comparison should be read as an empirical comparison between two practical training regimes rather than as an isolated ablation of multi-task supervision. Accordingly, gains from the joint model reflect the selected joint-training pipeline as a whole, including shared supervision, initialization, sampling, and loss balancing.

## 5.1 Single-task

We first report single-task baselines, where each descriptor is learned by an independent model using the same SwinV2 encoder and fusion transformer (Section 4). Test performance is reported at validation-selected checkpoints, and Table 2 summarizes the results.

The single-task baselines show a clear split between visually well-supported descriptors and targets limited by weak evidence or label imbalance. Impact plane and DoF are learned well, indicating that the model can recover coarse impact geometry from post-crash views. The longitudinal/lateral zone also shows usable signal, suggesting that the model often localizes the main crush region along the vehicle length. In contrast, vertical/lateral zone and damage distribution are more strongly afected by long-tailed labels, with frequent classes dominating performance. Deformation extent is dificult to predict exactly, but near-miss agreement is substantially higher under Acc@±1, consistent with its graded severity interpretation. For ∆V, the single-task model reaches MAE of 8.04 km/h for the longitudinal component and 5.31 km/h for the lateral component on the labeled subset, showing that visual evidence contains a measurable, though incomplete, signal for reconstructed kinematic quantities.

## 5.2 Multi-task

We next report the joint model, which predicts all crash descriptors jointly using a shared SwinV2 encoder and fusion transformer with task-specific heads (Section 4.5). The same evaluation protocol is used as for the single-task baselines, with the checkpoint selected by the validation joint objective. Table 2 compares both regimes.

Table 2. Comparison between selected single-task reference models and the selected joint-training recipe. ∆ denotes the absolute change from the selected single-task model to the selected joint-training model, and Rel. ∆ denotes the corresponding relative change with respect to the single-task result. For AngErr and MAE, lower is better. For Acc/F1, higher is better. The comparison should be interpreted as an empirical comparison between selected training regimes, not as an isolated ablation of multi-task supervision.
<table><tr><td>Descriptor</td><td>Metric</td><td>Single-task Multi-task</td><td></td><td>Δ (ST→MT)</td><td>Rel. ∆</td></tr><tr><td rowspan="2">Impact plane</td><td> $\operatorname { A c c }$ </td><td>0.924</td><td>0.922</td><td>-0.002</td><td>-0.2%</td></tr><tr><td> $F 1 _ { \mathrm { m a c r o } }$ </td><td>0.714</td><td>0.720</td><td>+0.006</td><td>+0.8%</td></tr><tr><td rowspan="2"> $\mathrm { D o F } _ { \mathrm { c l o c k } }$ </td><td>AngErr (°)</td><td>20.10</td><td>14.05</td><td>-6.05</td><td>-30.1%</td></tr><tr><td>Acc@±1</td><td>0.859</td><td>0.877</td><td>+0.018</td><td>+2.1%</td></tr><tr><td rowspan="2"> $\mathrm { Z o n e _ { L o n g . / l a t . } }$ </td><td>Acc</td><td>0.582</td><td>0.602</td><td>+0.020</td><td>+3.4%</td></tr><tr><td> $F 1 _ { \mathrm { m a c r o } }$ </td><td>0.577</td><td>0.532</td><td>-0.045</td><td>-7.8%</td></tr><tr><td rowspan="2"> $\mathrm { Z o n e _ { V e r t . / l a t . } }$ </td><td>Acc</td><td>0.901</td><td>0.902</td><td>+0.001</td><td>+0.1%</td></tr><tr><td> $F 1 _ { \mathrm { m a c r o } }$ </td><td>0.147</td><td>0.197</td><td>+0.050</td><td>+34.0%</td></tr><tr><td rowspan="2">Damage distribution</td><td>Acc</td><td>0.804</td><td>0.790</td><td>-0.014</td><td>-1.7%</td></tr><tr><td>F1macro</td><td>0.290</td><td>0.293</td><td>+0.003</td><td>+1.0%</td></tr><tr><td rowspan="3">Deformation extent</td><td>Acc</td><td>0.519</td><td>0.543</td><td>+0.024</td><td>+4.6%</td></tr><tr><td>Acc@±1</td><td>0.823</td><td>0.859</td><td>+0.036</td><td>+4.4%</td></tr><tr><td> $\mathrm { M A E _ { b i n s } }$ </td><td>0.939</td><td>0.818</td><td>-0.121</td><td>-12.9%</td></tr><tr><td> $\varDelta V _ { \mathrm { l o n g } }$ </td><td>MAE</td><td>8.04</td><td>7.45</td><td>-0.59</td><td>-7.3%</td></tr><tr><td> $\varDelta V _ { \mathrm { l a t } }$ </td><td>MAE</td><td>5.31</td><td>5.09</td><td>-0.22</td><td>-4.1%</td></tr></table>

The selected joint-training pipeline improves several context-dependent targets, with the largest gain on DoF angular error $( 2 0 . 1 ^ { \circ }  1 4 . 0 5 ^ { \circ } )$ and consistent improvements for deformation extent and both ∆V components. These results suggest that shared training can be useful in this setting, but the evidence should be interpreted at the pipeline level rather than as a clean causal estimate of multitask supervision alone. The joint model difers from the single-task references in initialization, sampling, loss balancing, and optimization schedule, all of which may contribute to the observed changes.

The gains are also not uniform. Impact-plane accuracy changes only marginally, long./lat. zone macro-F1 decreases, and damage-distribution accuracy drops modestly. These failures are informative: they indicate that a single shared sampler and loss-balancing scheme may under-serve targets with diferent long-tail structure compared with task-specific single-task training. Thus, the joint model is best viewed as a compact reference pipeline with favorable aggregate behavior, not as uniformly superior across all descriptors. Detailed cross-paper reference points are provided in the ESM, Section 4.

## 6 Conclusion

We presented a CISS-based evaluation protocol for crash descriptor estimation from real-world, naturally incomplete multi-view post-crash photo sets. Using a unified SwinV2 and transformer-fusion architecture, we showed that post-crash imagery contains usable signal for recovering several CDC-derived deformation descriptors together with reconstructed longitudinal and lateral ∆V components. Under the selected training configurations used in this study, the integrated jointtraining pipeline improved several targets, most notably reducing DoF angular error by 30.1% (from 20.1<sup>◦</sup> to 14.05<sup>◦</sup>), while also improving deformation extent and both ∆V components. These results should be interpreted as practical reference performance for image-based crash-mechanics estimation, not as a replacement for reconstruction or EDR-based analysis. At the same time, some descriptors remained limited by weak observability and strong class imbalance, highlighting that image-based crash understanding is feasible but still incomplete in this real-world setting. Overall, these results support post-crash imagery as a practical complementary source of crash-mechanics information and provide a reference point for future multimodal models that combine images with structured crash metadata.

A natural next step is multimodal crash understanding that combines photo evidence with structured crash metadata commonly used in severity modeling, including vehicle, occupant, roadway, and environmental factors. Such structured fields could improve robustness, especially for weakly observable descriptors and for ∆V , where labels are noisy and incomplete. Architecturally, this suggests a multimodal design that (i) encodes images into a fused case embedding, (ii) embeds structured fields via a lightweight MLP or transformer, and (iii) fuses both modalities jointly. Another important direction is the construction of curated subsets with improved class balance, especially for rare crash descriptors that are dificult to assess reliably in the current long-tailed setting.

## References

1. Dean, M.E., Gabauer, D.J., Riexinger, L.E., Gabler, H.C.: Comparison of vehiclebased crash severity metrics for predicting occupant injury in real-world oblique crashes. Transportation research record 2677(2), 505–518 (2023)

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

3. German In-Depth Accident Study (GIDAS) Consortium: German In-Depth Accident Study (GIDAS) [dataset] (2023), version 2023-1; accessed 2025-06-15

4. Girshick, R.: Fast R-CNN. In: Proceedings of the IEEE international conference on computer vision. pp. 1440–1448 (2015)

5. Gu, C., Xu, J., Li, S., Gao, C., Ma, Y.: Injury risk assessment and interpretation for roadway crashes based on pre-crash indicators and machine learning methods. Applied Sciences 13(12), 6983 (2023)

6. Hasija, V., Takhounts, E.G., Craig, M.J.: A hybrid deep learning model for predicting delta-v from real world crash images. In: Proceedings of the 2025 IRCOBI Conference (2025), iRC-25-17

7. Kraft, M., Kullgren, A., Malm, S., Ydenius, A.: Influence of crash severity on various whiplash injury symptoms: A study based on real-life rear-end crashes with recorded crash pulses. In: Proc. 19th Int. Techn. Conf. on ESV, Paper (2005)

8. Kullback, S., Leibler, R.A.: On information and suficiency. The annals of mathematical statistics 22(1), 79–86 (1951)

9. Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., et al.: Swin Transformer V2: Scaling up capacity and resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12009–12019 (2022)

10. Morando, A.: Method for recovering data on unreported low-severity crashes. arXiv preprint arXiv:2503.04529 (2025)

11. Mostafa, A.M., Aldughayfiq, B., Tarek, M., Alaerjan, A.S., Allahem, H., Elbashir, M.K., Ezz, M., Hamouda, E.: AI-based prediction of trafic crash severity for improving road safety and transportation eficiency. Scientific Reports 15(1), 27468 (2025)

12. National Highway Trafic Safety Administration: Overview of the 2019 crash investigation sampling system. Tech. Rep. DOT HS 813 060, U.S. Department of Transportation (2020)

13. Niyogisubizo, J., Murwanashyaka, E., Nziyumva, E.: A comparative study on machine learning-based approaches for improving trafic accident severity prediction. Int. J. Eng. Res. Technol.(Ahmedabad) 10(10) (2021)

14. RASSI Consortium: Road Accident Sampling System – India (RASSI) [dataset] (2025), accessed 2025-06-15

15. Rifat, M.A.K., Kabir, A., Huq, A.: An explainable machine learning approach to trafic accident fatality prediction. Procedia Computer Science 246, 1905–1914 (2024)

16. SAE International: Collision deformation classification. SAE Standard J224\_202205, SAE International (2022). https://doi.org/10.4271/J224\_202205

17. Sharma, D., Stern, S., Brophy, J., Choi, E.: An overview of NHTSA’s crash reconstruction software WinSMASH. In: Proceedings of the 20th International Technical Conference on Enhanced Safety of Vehicles (2007)

18. Silver, D., Manek, H., Kay, M., Travis, P.: Estimating automobile crash characteristics from images using deep learning. In: The International FLAIRS Conference Proceedings. vol. 35 (2022)

19. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

# From Wrecks to Wisdom: Recovering Crash Mechanics from Real-World Multi-View Photos Supplementary Material

Ondřej Valach<sup>1[0009−0000−7629−0516]</sup> <sup>⋆</sup>, Václav Diviš<sup>1[0000−0001−9935−7824]</sup>, and Ivan Gruber<sup>1[0000−0003−2333−433X]</sup>

University of West Bohemia, Faculty of Applied Sciences, Department of Cybernetics and New Technologies for the Information Society valacho@fav.zcu.cz, vincie@kky.zcu.cz, grubiv@ntis.zcu.cz

## 1 Preliminary Experiments on Direction of Force (CDC Clock)

We ran preliminary experiments on the CDC 12-bin direction-of-force (DoF, clock) descriptor because it provides one of the clearest visual signals in postcrash photo sets and therefore served as a practical proxy for selecting the backbone and multi-view aggregation design. These experiments are intended as model selection evidence for the input and fusion pipeline.

We compared several backbones and multi-view aggregation strategies, and evaluated three practical ways of converting a crash case into a photo-set input: (i) sampling a random subset of K images (with replication or permutation), (ii) concatenating all images along the channel dimension, and (iii) mapping photos into fixed ordered view slots with padding for missing viewpoints.

Across these experiments, tested CNN-style backbones (EficientNet-B4 and InternImage-T) did not make sustained progress: training curves oscillated and validation angular error plateaued early. A modest improvement was observed when InternImage-T consumed all views jointly via channel concatenation, but performance still saturated quickly.

Transformer backbones adapted better to the task. With SwinV2-base, mean pooling provided a small gain, concatenation of per-view tokens improved further, and a dedicated token-fusion transformer operating on fixed ordered views produced the most stable learning behavior and the lowest validation error (about 20<sup>◦</sup> AngErr). This motivated the final design used in the main paper.

## 1.1 Per-Setup Description (P1-P6)

Table 1 summarizes the configurations and we describe each setup referenced by its P-code.

Corresponding Author

Table 1. DoF ablations. Ef-B4: EficientNet-B4. SwinV2-B: SwinV2-base. Intern-T: InternImage-T. Rand K: random subset of available views. Rep: replication if fewer than K views. Perm: random permutation. Fixed-9: fixed ordered 9-slot input. Pad: zero-padding for missing views and edge padding to preserve image aspect ratio. AngErr: mean absolute angular error (degrees) under circular distance.
<table><tr><td>ID</td><td>Backbone</td><td>Views</td><td>Fuse</td><td>Target/Loss</td><td>AngErr (°) ↓</td></tr><tr><td>P1</td><td>Eff-B4</td><td>Rand K (rep)</td><td>Concat.</td><td>sin / cos MSE</td><td>44.1°</td></tr><tr><td>P2</td><td>Intern-T</td><td>Rand K (rep)</td><td>Concat.</td><td>Circ-smooth KL</td><td>44.6°</td></tr><tr><td>P3</td><td>Intern-T</td><td>Chan. conc.</td><td></td><td>Circ-smooth KL</td><td>37.4°</td></tr><tr><td>P4</td><td>SwinV2-B</td><td>Rand K (perm)</td><td></td><td>Mean pool sin / cos MSE</td><td>32.9°</td></tr><tr><td>P5</td><td>SwinV2-B</td><td>Rand K (rep)</td><td>Concat.</td><td>sin / cos MSE</td><td>27.3°</td></tr><tr><td>P6</td><td>SwinV2-B</td><td>Fixed-9 (pad)</td><td></td><td>Transf.fuse Circ-smooth KL</td><td>21.8°</td></tr></table>

P1: EficientNet-B4 + random multi-view replication + concatenation + circular regression P1 uses an EficientNet-B4 encoder shared across views. Each crash case is represented by a random subset of K available images; when fewer than K views are present, images are replicated to keep a fixed view count (Rand K, rep). View features are fused by simple concatenation (Concat.), and the DoF target is trained via circular regression by predicting (sin θ, cos θ) with an MSE loss. In practice, this configuration did not learn reliably: training was unstable and validation error oscillated around a plateau.

P2: InternImage-T + random multi-view replication + concatenation + circular smoothed classification P2 evaluates InternImage-T with the same randomized view construction (Rand K, rep). Per-view features are concatenated (Concat.) and the DoF head is trained as 12-way circular classification using Gaussiansmoothed targets over circular distance with KL divergence (Circ-smooth KL). Similar to P1, this per-view processing + concat setup showed limited learning progress and quickly saturated.

P3: InternImage-T + image per channel concatenation (single forward pass) + circular smoothed classification P3 changes only the input representation: instead of encoding views independently, images are stacked along the channel dimension and fed into InternImage-T in a single pass, so no explicit fusion module is needed. This produced a measurable but modest improvement over P2. However, validation error still plateaued relatively early, suggesting that the model benefited from joint low-level processing but remained limited in multiview reasoning.

P4: SwinV2-base + random view permutation + mean pooling + circular regression P4 uses a SwinV2-base backbone, which processes each view as a token sequence. Views are sampled as a random subset with permutation (Rand K, perm), and fused by mean pooling over embeddings. The head predicts (sin θ, cos θ) with an MSE loss. This configuration learned more consistently than CNN-style baselines, but improvements remained modest.

P5: SwinV2-base + random multi-view replication + concatenation + circular regression P5 keeps the SwinV2-base backbone but uses random-K sampling with replication when fewer than K views are available, and fuses the resulting view features by concatenation rather than mean pooling. This configuration achieved lower angular error than P4, suggesting that preserving per-view identity in the fused representation was beneficial, although the comparison also reflects the change in view construction.

P6: SwinV2-base + fixed ordered 9-view input + transformer fusion + circular smoothed classification P6 aligns with the final design direction: images are mapped into a fixed ordered 9-slot representation (Fixed-9), and missing canonical viewpoints are filled by padding. Per-view tokens are fused by a dedicated transformer-based fusion module (Transf.fuse), and the DoF head uses circularly smoothed classification with KL divergence (Circ-smooth KL). This setup produced the most stable learning behavior and reached the lowest validation error in these experiments (21.8<sup>◦</sup> AngErr), motivating its use in the main model. Note that these numbers reflect results from the preliminary ablation study training runs, therefore the 21.8<sup>◦</sup> is not equal to our best single-task DoF result.

These ablations represent a compact subset of the exploratory experiments conducted during model development and are included because they capture the main design decisions that shaped the final model. Additional runs were performed, but they did not provide suficiently distinct insights to justify separate inclusion here.

## 2 Dataset Curation, Preprocessing, and Evaluation Details

This section provides supplementary implementation details for preprocessing and dataset curation, example filtering outcomes, case construction, missinglabel handling, view-drop augmentation, and checkpoint selection. These details support the main experimental protocol but are not essential for understanding the core architecture or main results.

## 2.1 Crash Case Acquisition and Curation Pipeline

Figure 1 summarizes the acquisition and curation pipeline used to construct the benchmark. We first collected publicly available crash investigation cases from the NHTSA CISS database, including case metadata, case images, and associated documents.

We then applied a data-cleaning stage to remove unusable cases, including entries with empty metadata, repeated or multiply crashed cases, and cases missing essential images. After this initial cleanup, we performed metadata-based image filtering to retain crash-relevant exterior vehicle views and discard visually uninformative categories.

![](images/474ceaf8979229f47c16fa5172af7a675d761997935fe34041de674aee591844.jpg)  
Fig. 1. Overview of the crash case acquisition and curation pipeline. Starting from publicly available NHTSA CISS cases, we applied data cleaning, metadata-based image filtering, YOLOv8s vehicle filtering, and a dedicated RetinaNet-based wheel-detector filtering stage to obtain the final curated crash-image sets used in this study.

The remaining candidate images were passed through a YOLOv8s vehiclefiltering stage, where we kept only images containing a suficiently large detected vehicle. Finally, a dedicated RetinaNet-based wheel detector removed wheeldominant close-up images that provide limited information about global crash deformation context. The resulting curated image sets were used for the downstream case construction and experiments reported in this work.

## 2.2 Preprocessing and Data Challenges

Given a case with post-crash images and annotations, we address four practical issues before training: irrelevant close-ups, missing viewpoints, missing target values, and strong class imbalance.

CISS vehicle photos sometimes include close-up images that carry little information about case-level deformation. A frequent example is a wheel close-up that is technically associated with a vehicle viewpoint but contributes little to global crash reasoning. To reduce this noise, we train a dedicated wheel detector model based on the RetinaNet [4] architecture on the CAWDEC [7] dataset and apply it as a preprocessing filter to remove images dominated by wheels. Representative removed examples are shown in Figure 2.

Most cases are missing one or more canonical viewpoints, so the corresponding slots remain empty. During training, we additionally apply random view-drop augmentation to improve robustness to incomplete evidence. At inference, the model receives only the available post-crash images.

Some cases also lack target values, most commonly ∆V. In single-task training, such samples are skipped for the relevant head. In multi-task training, losses are masked per head so that a sample contributes only to the targets with valid annotations, preserving partially annotated cases. Finally, because the CDCderived targets are strongly imbalanced, we handle long-tail structure during sampling and report macro-F1 in addition to accuracy.

## 2.3 Wheel-Detector Filtering Details

To reduce noise from close-up images that do not capture the global deformation pattern, we trained a dedicated wheel detector on the CAWDEC dataset [7]. The detector was used only as a preprocessing filter: images dominated by wheels were removed before case-level view construction. This was necessary because such close-ups are not identifiable from the available CISS metadata alone, even though they carry little information about crash-level deformation.

In the preprocessing pipeline, the detector was applied to candidate vehicle images prior to slot assignment. Images identified as wheel-dominated were excluded from further processing, while ordinary side views that still contained broader crash evidence were retained. Overall, this step reduced visually uninformative close-ups while preserving images relevant to case-level deformation reasoning.

![](images/f68326e32639757dcbd4f893e61ce8b2b4e1117e1f680c4c59e66f4e5347d013.jpg)  
Fig. 2. Representative examples removed by the wheel-filtering step. The discarded images are dominated by close-up wheel views and contain limited information about global crash deformation.

## 2.4 9-Slot Construction

Each crash case was mapped into a fixed 9-slot representation corresponding to the canonical viewpoints used throughout the study: front, rear, right, left, frontright, front-left, rear-right, rear-left, and top. When a viewpoint was unavailable, the corresponding slot was left empty and filled by padding. This representation preserves viewpoint identity across cases and allows the fusion module to associate tokens with consistent semantic positions.

If multiple images were available for the same canonical slot, the preprocessing pipeline selected one representative image at random. The same ordered slot representation was used during both training and evaluation. To make this step reproducible, the random choice is fixed by the preprocessing seed before model training, so validation and test cases use deterministic slot assignments. Since the slot assignment uses viewpoint metadata, this setup should be understood as image-based prediction with view organization supported by metadata, not as fully unconstrained image-set learning. For datasets without viewpoint metadata, we also evaluate automatic slot assignment using a vehicle orientation estimator trained from the Car Full View Dataset [1]. Table 2 compares this automatic variant with the metadata-based slot assignment used in the main benchmark.

Table 2. Efect of view-slot construction on DoF prediction. Metadata slots use CISS viewpoint labels and are used in the main experiments to isolate crash-descriptor learning from viewpoint-estimation noise. Auto-orientation slots use an image-based vehicleorientation estimator trained from the Car Full View Dataset [1] to assign views to the fixed 9-slot representation. Results are averaged over three runs using the same DoF model and evaluation protocol.
<table><tr><td>Slot construction</td><td>Input organization</td><td>AngErr (°) ↓</td></tr><tr><td>Random / unordered</td><td>Random available views</td><td>25.18</td></tr><tr><td>Metadata slots</td><td>Fixed 9-slot, CISS metadata</td><td>20.10</td></tr><tr><td>Auto-orientation slots</td><td>Fixed 9-slot, predicted orientation</td><td>20.10</td></tr></table>

Both fixed-slot variants substantially reduce DoF angular error compared with unordered random view construction. Averaged over three runs, the autoorientation result matches the metadata-slot result at the reported precision, suggesting that the gain comes mainly from organizing views into consistent canonical slots. For DoF prediction, this indicates that predicted vehicle orientation can replace viewpoint metadata for this part of the pipeline.

## 2.5 View-Drop Augmentation and Missing Labels

During training, we randomly dropped between 0 and 6 available views per case to improve robustness to incomplete evidence. This augmentation exposes the model to varying levels of view availability during training. At inference, no views are synthetically dropped and the model receives only the available post-crash images.

Missing target labels were handled diferently in the two regimes. In singletask training, samples without the required target were skipped for that task. In multi-task training, losses were masked per head so that a sample contributed only to the targets with valid annotations.

## 2.6 Target Coverage and Label Imbalance

Figure 3 visualizes the empirical distributions of all targets used in this study, including CDC-derived categorical descriptors and reconstructed WinSMASH ∆V components. The categorical targets exhibit pronounced long-tail behavior, with frontal and centrally located configurations occurring much more frequently than rarer side, corner, and uncommon deformation patterns.

![](images/270944361e2796075fb240bc1678a77475c38689c5d2f25a0020a6f7863d0278.jpg)  
Fig. 3. Label distributions in our CISS-based dataset. We show the class frequencies of all CDC-derived targets together with the distributions of $\varDelta V _ { \mathrm { l o n g } }$ and $\varDelta V _ { \mathrm { l a t } }$ (WinS-MASH reconstruction). The compound $\varDelta V$ vector is included for reader interpretation. The categorical targets show strong long-tail behavior with dominant frontal/central configurations.

Qualitative failure cases generally fall into four categories: missing or occluded views of the primary damaged region, visually subtle deformation despite non-negligible reconstructed ∆V, rare CDC classes with few training examples, and cases where close-up or partial vehicle images dominate the available photo set. These patterns are consistent with the quantitative results: coarse impact geometry is often visually recoverable, whereas fine-grained crush zones, damage distribution, deformation extent, and ∆V can depend on evidence that is missing, weakly visible, or only indirectly encoded in post-crash appearance.

## 3 Training and Optimization Details

This section provides the training and sampling details used for the single-task and multi-task configurations reported in the main paper. These details are important for reproducibility, but they are not central to the main methodological narrative.

## 3.1 Single-Task Training Details

In the single-task setting, one descriptor is learned per model instance. All singletask models use the same case representation, image backbone, and fusion module. Only the final head and task-specific training configuration difer. This setup provides per-target reference results under a common input pipeline, while allowing label-specific sampling choices for strongly imbalanced targets.

Most descriptor targets are categorical, so we formulate them as supervised classification and address long-tailed labels during sampling. To mitigate class imbalance, we use inverse-frequency sampling together with a stratified window scheme aligned with gradient accumulation. Specifically, samples are weighted by inverse label frequency, optionally raised to a power. Rare labels are defined by a high-quantile threshold, usually 0.90, on these weights, and each accumulation window, i.e., the set of micro-batches forming one optimizer update, is constrained to contain a minimum number of rare samples. The exact sampling setup is tuned separately for each label.

A practical advantage of the single-task setting is that the sampling policy can be tailored to each descriptor, allowing batches to approach a more balanced label distribution without additional dataset curation. In contrast, a single batch in multi-task training must serve several diferently imbalanced targets at once, making long-tail mitigation more dificult.

Across all tasks, we used AdamW [6], a Trapezoidal learning-rate scheduler [10], and an input resolution of 640 × 640 pixels.

## 3.2 Multi-Task Training Details

In the joint-training configuration, the fused case embedding is first passed through a shared pre-head MLP and then routed to task-specific heads. The shared trunk is initialized from the single-task $\mathrm { D o F _ { c l o c k } }$ pretrained model checkpoint. Newly added task-specific layers are initialized randomly, and missing labels are handled through task-specific loss masking.

Because this configuration difers from the single-task reference not only in joint supervision but also in initialization, loss balancing, sampling, and optimization schedule, the comparison should be interpreted as a comparison between selected single-task and joint-training setups under a common input representation, not as a perfectly matched ablation of multi-task supervision alone.

A major dificulty in the joint-training setting is that all heads must share a single sampling policy, even though the categorical targets are imbalanced in diferent ways. A standard inverse-frequency sampler is therefore not suficient, because rarity is task-dependent: the same sample may be rare for one target and common for another. We therefore use task-aware weighted sampling. For each sample, we compute inverse-frequency weights for the categorical targets with available labels and take their maximum as the sampling weight. This increases the sampling probability of cases that are rare for at least one task. Sampling is then performed with replacement according to these weights.

Table 3. Representative training configurations for the reported single-task and multitask results. Both regimes use the same case representation, backbone, and fusion module, but difer in initialization, optimization, and loss balancing. Single-task hyperparameters were tuned per target, so that column is a compact summary rather than one identical configuration for all descriptors. Quant.-strat. rare/update: quantilestratified update-level sampling with at least one rare sample per efective optimizer update. Task-aware WRS: WeightedRandomSampler using the maximum available inverse-frequency weight across categorical targets. Init. strategy: transfer initializes from a related single-task checkpoint, e.g., impact plane for vertical/lateral finetuning; clock-pretrain + random initializes the shared trunk from DoF<sub>clock</sub> pretraining and the remaining components randomly.
<table><tr><td></td><td>Single-task</td><td>Multi-task</td></tr><tr><td>Epochs</td><td>50</td><td>50</td></tr><tr><td>Learning rate</td><td>1×10−5</td><td> $2 / 6 / 3 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR split</td><td></td><td>backbone / fusion+pre-head / task heads</td></tr><tr><td>Warmup / anneal</td><td>3/7</td><td>5  / 20</td></tr><tr><td>Batch accum.</td><td>4/-</td><td>1 /4</td></tr><tr><td>Effective batch</td><td>4</td><td>4</td></tr><tr><td>Views / mask</td><td>9 / k ∈ {0, . . . , 6}</td><td> $9 ~ / ~ k \in \{ 0 , \ldots , 6 \}$ </td></tr><tr><td>Sampler</td><td>quant.-strat. rare/update</td><td>task-aware WRS</td></tr><tr><td>Loss</td><td>CE / KL / SmoothL1</td><td> $\mathrm { C E } + \mathrm { K L } ( \sigma { = } 0 . 8 5 ) + \mathrm { S m o o t h L 1 }$ </td></tr><tr><td>DWA</td><td></td><td>yes</td></tr><tr><td>GradNorm</td><td></td><td>from epoch 6</td></tr><tr><td>Init. strategy</td><td>random / transfer</td><td>clock-pretrain + random</td></tr></table>

While this does not guarantee balanced batches for every target, it improves exposure to underrepresented cases while maintaining a single shared training stream for all heads. From epoch 6 onward, we additionally enable GradNorm [2] and Dynamic Weight Averaging (DWA) [5] to further stabilize gradient balance across task heads. These mechanisms are part of the reported joint-training recipe and should not be separated from the empirical comparison to the singletask references.

## 4 Reference Points to Prior Image-Based Work

Most crash-severity and injury pipelines rely on structured metadata (EDR, reconstruction outputs, or curated case forms). In contrast, we study an imagebased setting at real-world scale, with naturally incomplete multi-view photo sets. The closest prior image-based crash studies difer substantially in target definitions, crash-mode coverage, curation, and input modality. In particular, Silver et al. [9] combine curated real-world photographs with synthetic images generated in the Rigs of Rods simulator [8], while Hasija et al. [3] use curated realworld cases together with vehicle metadata from NHTSA sources. We therefore report their numbers as reference points rather than as direct baselines.

Impact plane prediction. Table 4 reports comparisons for impact plane prediction, comparing prior single-image impact location with our CDC impact plane estimation from multi-view photo sets.

Table 4. Impact plane image prediction. Silver et al. [9] evaluate in a curated single-image setting where the background is masked to retain only the vehicle, inputs are limited to front or rear views, and the label space is analogously reduced to front/back location. In contrast, our benchmark uses real-world crash cases represented as multi-view photo sets with natural viewpoint missingness and 4-plane coverage (front/right/left/back).
<table><tr><td>Work</td><td>Input</td><td>Target</td><td>Acc</td></tr><tr><td>Silver et al. [9]</td><td>1 img</td><td>LOC (f/b)</td><td>0.920</td></tr><tr><td>Ours (ST)</td><td>≤ 9 images</td><td>CDC (f/r/1/b)</td><td>0.924</td></tr><tr><td>Ours (MT)</td><td>≤ 9 images</td><td>CDC (f/r/1/b)</td><td>0.922</td></tr></table>

These numbers are included only to contextualize task dificulty. Because the label spaces, curation protocols, and inputs difer, they should not be interpreted as evidence of performance parity.

∆V estimation. Table 5 reports reference point results for image-based ∆V prediction. Because prior work difers in data curation, crash-mode coverage, available metadata, and evaluation metrics (MAE/RMSE), these comparisons should not be interpreted as direct baselines. Instead, the table is intended to contextualize our real-world image-based setting and to show that post-crash imagery provides a usable signal that could support downstream injury, crash severity, or metadata prediction pipelines.

Taken together, these reference points help position our study within the body of prior image-based crash work. Because the target definitions, curation protocols, crash-mode coverage, and available inputs difer substantially across works, we use these comparisons only for context rather than as direct baselines. Within that constraint, our results provide a practical reference point for crash descriptor estimation from real-world, incomplete multi-view photo sets and motivate multimodal fusion as a natural next step, rather than establishing performance parity with prior curated or metadata-assisted settings.

## References

1. Catruna, A., Betiu, P., Tertes, E., Ghita, V., Radoi, E., Mocanu, I., Dascalu, M.: Car full view dataset: Fine-grained predictions of car orientation from images. Electronics 12(24), 4947 (2023)

Table 5. ∆V estimation results. Scope codes: R1 (Silver et al. [9]) uses a manually curated set (combined from simulation and real world) of small passenger vehicles for front/rear collisions (< 96 km/h), images are grayscale and manually cropped to the vehicle, reporting MAE for $\varDelta V _ { \mathrm { l o n g } } .$ R2 (Hasija et al. [3]) uses a curated frontalonly subset and incorporates vehicle metadata (weight, body type, stifness); we report their main RMSE for $\varDelta V _ { \mathrm { l o n g } }$ on the full test set (6.38 km/h). They additionally report 7.50 km/h on the EDR-verified subset. R3 (ours) evaluates on real-world CISS multi-view photo sets with natural missingness and image-based inputs, we predict component-wise $( \varDelta V _ { \mathrm { l o n g } } , \varDelta V _ { \mathrm { l a t } } )$ and report MAE and RMSE, with missing labels handled by masking and a 4-way impact plane.
<table><tr><td>Work</td><td>Input</td><td>Scope</td><td>Metric</td><td>Target</td><td>Score [km/h]</td></tr><tr><td>Silver et al. [9]</td><td>1 image</td><td>R1</td><td>MAE</td><td> $\varDelta V _ { \mathrm { l o n g } }$ </td><td>4.19</td></tr><tr><td>Hasija et al. [3]</td><td>1 image, meta</td><td>R2</td><td>RMSE</td><td> $\varDelta V _ { \mathrm { l o n g } }$ </td><td>6.38</td></tr><tr><td>Ours (ST)</td><td> $\leq 9$  images</td><td>R3</td><td>MAE</td><td> $( \varDelta V _ { \mathrm { l o n g } } , \varDelta V _ { \mathrm { l a t } } )$ </td><td> $8 . 0 4 \ / \ 5 . 3 1$ </td></tr><tr><td rowspan="2">Ours (MT)</td><td rowspan="2"></td><td rowspan="2">R3</td><td>RMSE</td><td> $( \varDelta V _ { \mathrm { l o n g } } , \varDelta V _ { \mathrm { l a t } } )$ </td><td>11.96 / 8.20</td></tr><tr><td>MAE</td><td> $( \varDelta V _ { \mathrm { l o n g } } , \varDelta V _ { \mathrm { l a t } } )$ </td><td>7.45 / 5.09</td></tr><tr><td></td><td>≤ 9 images</td><td></td><td>RMSE</td><td> $( \Delta V _ { \mathrm { l o n g } } , \Delta V _ { \mathrm { l a t } } )$ </td><td>11.26 / 7.25</td></tr></table>

2. Chen, Z., Badrinarayanan, V., Lee, C.Y., Rabinovich, A.: GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: International conference on machine learning. pp. 794–803. PMLR (2018)

3. Hasija, V., Takhounts, E.G., Craig, M.J.: A hybrid deep learning model for predicting delta-v from real world crash images. In: Proceedings of the 2025 IRCOBI Conference (2025), iRC-25-17

4. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 2980–2988 (2017)

5. Liu, S., Johns, E., Davison, A.J.: End-to-end multi-task learning with attention. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1871–1880 (2019)

6. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

7. Novozámský, A.: Cawdec – car wheels detection & classification. Kaggle dataset, license: CC BY-SA 4.0. Accessed: 2025-06-08

8. Ohlidal, P.: Rigs of rods soft-body physics simulator (2020), accessed: 2020-08-12

9. Silver, D., Manek, H., Kay, M., Travis, P.: Estimating automobile crash characteristics from images using deep learning. In: The International FLAIRS Conference Proceedings. vol. 35 (2022)

10. Xing, C., Arpit, D., Tsirigotis, C., Bengio, Y.: A walk with sgd. arXiv preprint arXiv:1802.08770 (2018)