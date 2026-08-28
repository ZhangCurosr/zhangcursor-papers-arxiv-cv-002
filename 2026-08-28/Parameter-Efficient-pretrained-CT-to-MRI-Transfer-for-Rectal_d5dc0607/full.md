# Parameter-Efficient pretrained-CT-to-MRI Transfer for Rectal Cancer Segmentation: Performance-Calibration Trade-offs

Aneesh Rangnekar[0000–0002–0079–9495], Jorge Tapias Gomez, Joseph O Deasy, and Harini Veeraraghavan

Memorial Sloan Kettering Cancer Center, New York, USA Corresponding author: veerarah@mskcc.org

Abstract. Accurate rectal cancer segmentation from magnetic resonance imaging (MRI) is essential for adaptive radiotherapy and tumor response assessment, but deployment also requires computational efficiency and informative, calibrated uncertainty estimates. We therefore introduce SWIFT, a SWin pretrained model with parameter-eFficient and Tumor-aware fine-tuning for rectal cancer segmentation. A Swin V2 encoder pretrained on 10,444 public 3D CT volumes using a DINOv2-style objective was adapted to T2-weighted MRI through four cumulative configurations: full fine-tuning (SWIFT), decoder compression (SWIFTe), low-rank adaptation (SWIFTe-LoRA), and a four-member LoRA-decoder ensemble (SWIFTe-LDE4). Geometric accuracy, tumor detection, radiomic agreement, and probability calibration were evaluated on a heldout 247-case test set from a single-institution cohort acquired using 1.5 or 3 Tesla GE scanners. Compared with SWIFT, SWIFTe reduced total parameters by 70.1% (from 72.8M to 21.8M) and increased tumor detection rate from 89.9% to 93.9%, while achieving a slightly lower median surface DSC (0.61 versus 0.62) and improved radiomic agreement. In a separate SWIFTe ablation, removing tumor-aware augmentation reduced detection from 93.9% to 89.9% but increased surface DSC from 0.61 to 0.64, demonstrating a detection-boundary-agreement trade-off. SWIFTe-LoRA used 14.6% of SWIFTe's trainable parameters while retaining similar segmentation performance. SWIFTe-LDE4 achieved the lowest calibration errors among the four configurations after temperature scaling (expected calibration error, 0.217; Brier score, 0.222), although the absolute expected calibration error indicates residual miscalibration. Similar efficiency-calibration patterns were observed using the public VoCo checkpoint, supporting robustness across pretrained initializations rather than external clinical generalizability. Code and selected model checkpoints are available in our GitHub repository.

Keywords: Rectal cancer segmentation · Efficient decoder · LoRA . Uncertainty · Performance-calibration trade-off

## 1 Introduction

MR-guided adaptive radiotherapy (MRgART) enables daily treatment adaptation to deliver ablative tumor doses while limiting exposure to adjacent radiosensitive gastrointestinal organs [6]. Daily rectal tumor contouring is a major bottleneck in MRgART [9]. Self-supervised training of high-capacity deep learning (DL) models with diverse imaging datasets has demonstrated adaptability to multiple medical imaging tasks [23,11,4,26] and robustness to some imaging variations [10]. However, models can fail in clinical scenarios not captured during training, requiring manual contouring [13]. Uncertainty metrics generated from voxel-wise probabilistic predictions can provide information about segmentation confidence [22,20] and potentially guide the manual correction process [25,15]. However, poorly calibrated models may fail confidently in unfamiliar testing regimes [16,12,21]. Furthermore, large models require significant computational resources for adaptation, inference, and uncertainty estimation, creating a computational bottleneck in a complex adaptive radiotherapy (aRT) workflow.

Published rectal-tumor MRI studies reported DSC values of 0.68–0.70 on multiparametric MRI [24], while patient-specific adaptation on daily MR-Linac images reduced median HD95 from 22.0 mm to 4.8 mm [13]. Differences in cohorts, protocols, and metrics preclude direct ranking against our results.

DL models deployed in time-sensitive aRT settings must often satisfy three requirements: (i) acceptable geometric accuracy, (ii) practical computational costs, and (iii) calibrated uncertainty estimates to guide when user intervention is needed. Prior works have individually examined methods to improve geometric accuracy [13], reliability of uncertainties [22,21], and strategies to address computational bottlenecks, including streamlined decoders [19], low-rank adaptation (LoRA) for efficient tuning [8], and probabilistic inference with shared heads and LoRA [3,1,17]. However, none have studied the interplay of parameter efficiency, performance, and calibration for reliably accurate tumor segmentation considering clinical deployment. Our key contribution is therefore a benchmarking framework that holistically evaluates performance-calibration trade-offs with parameter efficiency in pretrained models.

We benchmarked a cumulative four-stage curriculum by first creating a model called Swin pretrained model with parameter eFficient and Tumor-aware finetuning (SWIFT) applied to rectal cancer segmentation from MRI. SWIFT uses Swin transformer V2 to extract computationally efficient attention, integrating local and global anatomical context while supporting stable training for improved transferability [5]. It retains the standard Swin UNETR V2 decoder [5] and is fine-tuned end-to-end, providing a full-adaptation reference. Besides imaging differences, rectal cancers exhibit a highly variable appearance on T2-weighted MRI due to tumor morphology, histologic composition, lymph node invasion, and extracellular mucin, confounding accurate segmentation [7,2,14,27]. To address this, we developed a tumor-aware intensity augmentation that selectively varies the intensity statistics within tumors to capture intermediate, dark, and bright tumors. All model variants were fine-tuned with tumor-aware augmentation. Second, we evaluated the impact of reducing decoder capacity and model complexity by replacing the original decoder with an efficient EffiDec3D decoder [19] within SWIFT (or SWIFTe). Third, we analyzed the impact of freezing the pretrained encoders to reduce the parameters for fine-tuning using LoRA [8] within SWIFTe (or SWIFTe-LoRA). Fourth, we evaluated whether extending SWIFTe-LoRA with member-specific adapter-decoder pairs and shared pretrained encoder (or SWIFTe-LDE4) enhanced uncertainty calibration [17].

## 2 Methodology

Data preprocessing: This retrospective study involving patients with rectal cancers imaged before treatment was performed following institutional review board approval and waiver of consent. A total of 416 T2-weighted oblique axial MRIs acquired on 1.5 or 3 Tesla GE scanners with phased-array coils, 2 to 4 mm slice thickness, and field of view \~180 to 220 mm were analyzed using patientspecific splits for training (n = 136), validation (n = 33), and testing (n = 247). Images were resampled to 1 mm3 isotropic spacing, reoriented to RAS, and clipped to [0, 800]. Network inputs were min-max scaled after clipping, whereas radiomics used the clipped intensity scale to preserve the engineered-feature convention. Manual tumor contours served as the reference standard.

Full fine-tuning baseline (SWIFT): SWIFT is based on a CT-to-MRI crossmodality transfer pipeline: a 3D Swin UNETR V2 [5] network is first pre-trained in a self-supervised manner on 10,444 heterogeneous public CT volumes (Fig. 1) via a DINOv2-style [18] self-distillation objective, then fine-tuned for segmenting rectal cancers on MRI. The model consists of a hierarchical Swin V2 encoder and a Swin UNETR decoder with multi-scale skip connections, using a feature size of 48 and a two-channel background/tumor output. In SWIFT, the full encoderdecoder network is fine-tuned end-to-end, serving as the baseline architecture for benchmarking remaining configurations as described below.

Efficient decoder: SWIFTe modifies SWIFT by replacing the Swin UNETR decoder with EffiDec3D [19], while retaining the CT pretrained Swin V2 encoder and downstream training/inference protocol, allowing us to isolate the effect of decoder compression. EffiDec3D uses a fixed channel width across stages (48 decoder channels) and a resolution factor of 2. Omitting the finest full-resolution reconstruction stage, it reconstructs logits at reduced spatial resolution before trilinear upsampling to the input grid.

SWIFTe-LoRA and SWIFTe-LDE4: We next tested encoder-efficient adaptation with SWIFTe-LoRA and probability calibration with SWIFTe-LDE4. SWIFTe-LoRA used parameter-efficient adaptation [8] of SWIFTe by inserting LoRA adapters into the Swin window-attention query, key, value, and outputprojection layers after loading the CT-pretrained encoder. The frozen pretrained encoder was adapted using trainable rank-4 LoRA modules, while jointly training the EffiDec3D decoder. SWIFTe-LDE4 extended this setting to four memberspecific LoRA-adapter and decoder pairs [17] with a shared frozen pretrained encoder. We used four ensemble members as a pragmatic compromise between ensemble diversity and computational cost, since each member adds an adapterdecoder branch and forward pass. Foreground probabilities were averaged for segmentation, while inter-member variance and disagreement quantified uncertainty.

Implementation details: All configurations were optimized with a combined soft Dice and cross-entropy loss using AdamW, a learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ , linear warmup, and cosine annealing. During training only, tumoraware intensity augmentation used the manual tumor mask to simulate rectaltumor appearance variation by locally transforming foreground tumor voxels as $I ^ { \prime } ( x ) = \alpha I ( x ) + \beta _ { 3 }$ where $\alpha \sim \mathcal { U } ( 0 . 5 , 1 . 5 )$ and $\beta \sim \mathcal { U } ( - 0 . 2 , 0 . 2 )$ . Surrounding anatomy was left unchanged, and the transform was applied after spatial augmentation with probability 0.3. LoRA models used the same loss, optimizer family, crop size, augmentation, and inference protocol, with optimization restricted to the trainable adapter and decoder parameters described above. Tumor-aware augmentation was held constant across the four primary configurations and was not part of the parameter-reduction strategy.

The models used 3D crops of $9 6 \times 9 6 \times 6 4$ , instance normalization, and a twoclass foreground/background output. Binary masks were generated using a foreground probability threshold of $\tau = 0 . 5$ . Inference used 3D sliding windows with 50% overlap and test-time augmentation. Automated postprocessing selected the largest connected component as tumor corresponding to the primary gross tumor volume. Inference operated on full volumes without manual-contour information. Detection included all test cases, while model-specific misses were penalized in the primary sDSC analysis as described below.

Case-level uncertainty: For all models, case-level uncertainty was summarized as mean predictive entropy within the predicted tumor. Empty predictions for missed cases were represented by averaging the uncertainty map over the full evaluation volume. For SWIFTe-LDE4, member variance of tumor probability and member disagreement were additionally summarized as ensemble-diversity measures. Retrospective calibration evaluation used a tumor-focused region of interest comprising voxels with tumor probability $\geq 0 . 0 5$ or manual tumor label. Representative entropy maps are shown in Fig. 2.

Evaluation metrics: Segmentation performance across 247 test cases was evaluated using detection rate (prediction-reference intersection-over-union $>$ 0.1) and surface DSC (sDSC) at a 2 mm tolerance. HD95 was not reported because it is undefined for empty predictions, and detected-case-only reporting would produce different model-specific cohorts that could favor models with more misses. We therefore report detection over all cases and assign zero sDSC to model-specific misses within the shared evaluable cohort. Primary sDSC summaries analyzed a shared-detectable cohort of 236 cases, setting model-specific misses to zero and excluding 11 cases missed by all evaluated models, while Lin's concordance correlation coefficient evaluated agreement across IBSI-aligned radiomic feature families. Paired inferential comparisons applied Cochran's Q and exact McNemar tests for detection alongside Friedman and Wilcoxon signed-rank tests for sDSC, with p-values adjusted using the Holm procedure. Reliability diagrams, expected calibration error (ECE), and Brier score were reported following validation-set temperature scaling.

![](images/375f6032807e9ee084c487a475f951a02369291c71b7872b4427604fc0539219.jpg)

![](images/05fa53c447cbf2651d9321c57926a65dc59e794bef7398d5072157abe1ee03ec.jpg)

![](images/b5618943d9c831cbc0bde8254a1a841f69762769be33cd1373ee1a606b587712.jpg)

![](images/9be0ce8a2bcf0fcccbaf81f22624475a7cf0ac0715f5cd4736182739988eb611.jpg)

![](images/7d706f277b4b31e283a89aa47456c61081a5b527ec194bea88911894b88aabf4.jpg)  
Fig. 1: Pretraining context and transfer results. Left: CT10K pretraining subset used for SWIFT encoder pretraining and its scale relative to VoCo. Right: detection, sDSC, ECE, and Brier score across full fine-tuning, efficient decoder (EffiDec), LoRA adaptation, and LDE4 for SWIFT and VoCo.

## 3 Results

SWIFTe achieved higher detection with fewer parameters: SWIFTe reduced total parameters by 70.1%, from 72.8M for SWIFT to 21.8M, while detecting 232 of 247 tumors (93.9%) and retaining a shared-detectable median sDSC of 0.61 (Fig. 1). Although SWIFT reached a slightly higher median sDSC of 0.62, its detection rate was 89.9% compared with 93.9% for SWIFTe; detection rates differed across configurations (Cochran's Q, Holm-adjusted p = 0.039).

SWIFTe-LDE4 improved calibration while retaining similar sDSC: SWIFTe-LDE4 achieved the lowest calibration errors among the SWIFT configurations, reducing ECE from 0.245 for SWIFTe-LoRA to 0.217 and Brier score from 0.244 to 0.222 (Fig. 1). Median sDSC remained similar between SWIFTe-LDE4 and SWIFTe-LoRA (0.60 versus 0.59), while detection was 87.9% and 89.1%, respectively. Probability-derived entropy was available for all configurations, while SWIFTe-LDE4 additionally provided member-variance and disagreement measures for potential review prioritization. Although lowest among the evaluated configurations, an ECE of 0.217 indicates residual absolute miscalibration.

![](images/e76f9f3a2c42faaaeb404ebf1780827e9c517c96ad89c9c5583425a467862463.jpg)  
Fig.2: Representative predictive-entropy maps across six test cases. The top row shows the image with manual and automated contours; remaining rows show entropy maps for SWIFT, SWIFTe, SWIFTe-LoRA, and SWIFTe-LDE4. Yellow indicates higher entropy, and panel numbers denote sDSC.

One scalar temperature per configuration was fitted on the 33-case validation set and applied unchanged to the test set (Fig. 3). Because temperature scaling preserved the 0.5 decision boundary, thresholded contours and segmentation metrics were unchanged. Low-confidence calibration regions corresponded qualitatively to tumor voxels assigned low foreground probability in challenging cases typically consisting of tumors with heterogeneous T2 signal and diffuse boundaries (Fig. 4).

Several radiomic feature families showed strong agreement: Radiomic agreement varied across feature families and SWIFT configurations (Fig. 5). Across 247 matched cases and 288 numeric features, median manual-versusautomated CCC was 0.62 for SWIFT, 0.68 for SWIFTe, and 0.69 for both SWIFTe-LoRA and SWIFTe-LDE4. The fraction of features with $\mathrm { C C C } \geq 0 . 7 5$ was 30%, 46%, 39%, and 38%, respectively. Shape features were the least reproducible family across all configurations (median CCC 0.51–0.54), consistent with modest median sDSC of 0.6. SWIFTe considerably enhanced reproducibility of first-order histogram and intra-tumor texture heterogeneity measures, including the GLRLM, GLSZM, and GLDM features. The LoRA-adapted configuration also improved reproducibility of intra-tumor texture heterogeneity feature families, including GLCM and GLSZM, compared with full fine-tuning.

Manual delineation Auto-segmentation Manual tumor voxels with p < 0.10  
![](images/f188139e6037ab9394eeb868362e683183dd87a62a4d5fc2b05c119a059935c1.jpg)  
(a) SWIFT.

![](images/c7ea2a1f51aba3763537aba96ce940f95dfcfd2fced6d625d11df9d1871623a4.jpg)  
(b) SWIFTe.

![](images/e7e6137ddbc316e08595a26674a8825c0f708a7f26b93d9605ba9779c79ef660.jpg)  
(c) SWIFTe-LoRA.

![](images/c63ceef4882c20b5b05907780c9e434a9de2265e26c79143ec43c24609551121.jpg)  
(d) SWIFTe-LDE4.

Fig. 3: Reliability diagrams show foreground probability calibration for each SWIFT configuration. Gray bars show empirical foreground frequency, orange shading shows the calibration gap, and the dashed line denotes ideal calibration. SWIFTe-LDE4 achieved the lowest ECE and Brier score, illustrating that calibration and segmentation overlap are distinct objectives.  
![](images/7c9b5917384b02ac674deeb8ec99b71e5621fa68374c1c66676a5795bc80b0dc.jpg)

![](images/b8918b08bc99007913c27fef90aaf27894b817a7aa8472b38b7edd0d347a80b0.jpg)

![](images/a1cfcb59bdca198d55a78617a24fcc08d42d4b3e49ac8661ac63dc7edc682f46.jpg)

![](images/097ac47fff82ff1aea11afcf7140d996dad57acb640f90f96072f7d6d9f64998.jpg)  
Fig. 4: Low-confidence tumor examples from calibration analysis. Pink regions indicate manual tumor voxels assigned foreground probability p < 0.10, with manual contours in green and automated contours in orange.

LoRA retained similar sDSC with fewer trainable parameters: SWIFTe-LoRA reduced the trainable burden to 3.2M parameters, corresponding to 14.6% of SWIFTe, while retaining a similar median sDSC (0.59 versus 0.61). Detection rate was lower at 89.1% for SWIFTe-LoRA compared to SWIFTe (93.9%). SWIFTe-LDE4 used 12.7M trainable parameters across four member-specific adapter-decoder pairs and similarly retained median sDSC at 0.60, while providing improved probability calibration.

The efficiency-calibration pattern was reproducible with VoCo: We replicated the decoder and adaptation sequence using VoCo-pretrained weights to assess sensitivity to initialization with a model pretrained using a substantially larger dataset (Fig. 1). Because SWIFT and VoCo differ in pretraining scale and objective, this was not a direct comparison of pretraining methods, but a test of whether the efficiency-segmentation-calibration trends generalized across initializations. The VoCo progression showed the same overall pattern: efficient decoding reduced model size while preserving segmentation performance, LoRA reduced the trainable burden while retaining similar overlap, and LDE4 improved probability calibration while maintaining broadly similar sDSC.

![](images/97e2098009fb9f99f351576ea43f85aa74cb14d7e3e685b1d5272e2f97a71c80.jpg)  
(a) Median CCC by IBSI feature family.

![](images/07e933cf20ed7fe14c8b41b1701a7a1e37a055100aeec32e97e5ff4a3a813b65.jpg)  
(b) Fraction of features with $\mathrm { C C C } \geq 0 . 7 5 .$  
Fig. 5: Radiomic-feature agreement across the SWIFT progression. Panel (a) shows median manual-versus-automated CCC by IBSI feature family; panel (b) shows the fraction of features with $\mathrm { C C C } \geq 0 . 7 5$

Ablation study: Removing tumor-aware intensity augmentation reduced detection for SWIFTe (93.9% to 89.9%) while increasing sDSC from 0.61 to 0.64. SWIFT showed the same detection trend. The transform was retained to prioritize detection over boundary agreement.

For ensemble sizing, we evaluated SWIFTe-LDE variants with 2, 4, and 8 members. Four members provided a pragmatic balance between performance and computational cost, as larger ensembles offered minimal performance gains for added compute.

Temperature scaling reduced Brier scores across all configurations and lowered ECE for SWIFT, SWIFTe, and SWIFTe-LoRA without altering thresholded segmentations. Although SWIFTe-LDE4 ECE rose slightly from 0.207 to 0.217, it remained the lowest among all evaluated configurations.

## 4 Discussion and conclusions

This study benchmarked a cumulative parameter-efficient cross-modality adaptation approach using a CT-pretrained Swin transformer for rectal cancer segmentation from MRI. We found that an efficient decoder that reduced total parameters by 70.1% maintained segmentation performance while enhancing detection performance. LoRA adapter configurations achieved detection performance comparable to full fine-tuning, while the SWIFTe-LDE4 configuration improved calibration relative to the other SWIFT configurations. Importantly,

VoCo showed the same behavior with parameter-efficient adaptation even when pretrained using substantially more examples. These results support efficient decoding and LoRA as parameter-efficient adaptation strategies, with LDE4 providing a relative calibration improvement without substantially degrading segmentation accuracy.

The downstream analyses further showed why overlap alone was insufficient to characterize model behavior. Radiomic agreement varied by feature family: texture features, particularly those reflecting intra-tumor heterogeneity, were more reproducible for SWIFTe and the LoRA-adapted configurations, whereas shape features remained less reproducible across all configurations, consistent with the modest geometric agreement against manual contours. The models also showed systematic contour-volume differences, indicating that similar median sDSC can still correspond to different boundary behavior. Finally, higher entropy and, for SWIFTe-LDE4, higher ensemble disagreement generally corresponded to lower-confidence cases, supporting their use as qualitative review-prioritization signals rather than validated spatial error-localization maps.

This study evaluated one tumor type using a single-institution cohort acquired exclusively on GE scanners; robustness across institutions, vendors, and imaging protocols remains unverified. Comparisons were primarily internal to the SWIFT family, and published results provide context rather than head-to-head benchmarks. The uncertainty signals were descriptive and were not quantitatively validated against manual correction effort or dosimetric impact. Multiinstitutional validation and prospective evaluation of uncertainty-guided review are needed before clinical deployment.

Within the present retrospective cohort, parameter-efficient CT-pretrained transformer adaptations preserved segmentation performance, while SWIFTe-LDE4 improved calibration relative to the other evaluated configurations. Further validation of efficient uncertainty-aware segmentation workflows for adaptive radiotherapy is warranted.

Acknowledgments. This study was supported by the Simons Foundation and the Breast Cancer Research Foundation (through grant MATH-23-001), and the NIH ROBIN cooperative group (grant U54CA274291). This work utilized resources from the High-Performance Computing Group at Memorial Sloan Kettering Cancer Center.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Abboud, Z., Lombaert, H., Kadoury, S.: Sparse bayesian networks: efficient uncertainty quantification in medical image analysis. In: MICCAI. pp. 675–684. Springer (2024)

2. Battersby, N.J., How, P., Moran, B., Stelzner, S., West, N.P., Branagan, G., et al.: Prospective validation of a low rectal cancer magnetic resonance imaging staging system and development of a local recurrence risk stratification model: the mercury ii study. Annals of surgery 263(4), 751–760 (2016)

3. Fuchs, M., Gonzalez, C., Mukhopadhyay, A.: Practical uncertainty quantification for brain tumor segmentation. In: Medical Imaging with Deep Learning (2021)

4. Goyal, P., Caron, M., Lefaudeux, B., Xu, M., Wang, P., Pai, V.o.: Self-supervised pretraining of visual features in the wild. arXiv preprint arXiv:2103.01988 (2021)

5. He, Y., Nath, V., Yang, D., Tang, Y., Myronenko, A., Xu, D.: SwinUNETR-v2: Stronger swin transformers with stagewise convolutions for 3d medical image segmentation. In: MICCAI. pp. 416–426. Springer (2023)

6. Henke, L., Kashani, R., Yang, D., Zhao, T., Green, O., Olsen, L., et al.: Simulated online adaptive magnetic resonance-guided stereotactic body radiation therapy for the treatment of oligometastatic disease of the abdomen and central thorax: Characterization of potential advantages. International Journal of Radiation Oncology, Biology, Physics 96(5), 1078–1086 (2016)

7. Horvat, N., Carlos Tavares Rocha, C., Clemente Oliveira, B., Petkovska, I., Gollub M.J.: MRI of rectal cancer: Tumor staging, imaging techniques, and management. RadioGraphics 39(2), 367–387 (2019)

8. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., et al.: LoRA: Lowrank adaptation of large language models. In: ICLR (2022)

9. Intven, M.P.W., de Mol van Otterloo, S.R., Mook, S., Doornaert, P.A.H., de Grootvan Breugel, E.N., Sikkes, G.G., et al.: Online adaptive MR-guided radiotherapy for rectal cancer: Feasibility of the workflow on a 1.5 T MR-linac: Clinical implementation and initial experience. Radiotherapy and Oncology 154, 172–178 (2021)

10. Jiang, J., Rangnekar, A., Veeraraghavan, H.: Self-supervised learning improves robustness of deep learning lung tumor segmentation models to ct imaging differences. Medical Physics (Dec 2024)

11. Jiang, J., Tyagi, N., Tringale, K., Crane, C., Veeraraghavan, H.: Self-supervised 3d anatomy segmentation using self-distilled masked image transformer (SMIT). In: MICCAI. Springer (2022)

12. Karimi, D., Gholipour, A.: Improving calibration and out-of-distribution detection in deep models for medical image segmentation. IEEE TAI (2022)

13. Kensen, C.M., Simões, R., Betgen, A., Wiersema, L., Lambregts, D.M.J., Peters, F.P., et al.: Incorporating patient-specific information for the development of rectal tumor auto-segmentation models for online adaptive magnetic resonance imageguided radiotherapy. phiRO 32 (2024)

14. Lin, Y.C., Lin, G., Pandey, S., Yeh, C.H., Wang, J.J., Lin, C.Y., et al.: Fully automated segmentation and radiomics feature extraction of hypopharyngeal cancer on mri using deep learning. European Radiology 33(9), 6548–6556 (2023)

15. Maruccio, F.C., Simões, R., van Aalst, J.E., Brouwer, C.L., Sonke, J.J., van Ooijen P., Janssen, T.M.: Leveraging network uncertainty to identify regions in rectal cancer clinical target volume auto-segmentations likely requiring manual edits. phiRO (2025)

16. Mehrtash, A., Wells, W.M., Tempany, C.M., Abolmaesumi, P., Kapur, T.: Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. IEEE Transactions on Medical Imaging 39(12), 3868–3878 (2020)

17. Mühlematter, D.J., Halbheer, M., Becker, A., Narnhofer, D., Aasen, H., Schindler, K., Turkoglu, M.O.: LoRA-ensemble: Efficient uncertainty modelling for selfattention networks. TMLR (2026)

18. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., et al.: DINOv2: Learning robust visual features without supervision. TMLR (2024)

19. Rahman, M.M., Marculescu, R.: Effidec3d: an optimized decoder for highperformance and efficient 3d medical image segmentation. In: CVPR (2025)

20. Rangnekar, A., Nadkarni, N., Jiang, J., Veeraraghavan, H.: Quantifying uncertainty in lung cancer segmentation with foundation models applied to mixed-domain datasets. In: SPIE Medical Imaging 2025: Image Processing (2025)

21. Rangnekar, A., Veeraraghavan, H.: Tumor-anchored deep feature random forests for out-of-distribution detection in lung cancer segmentation. TMLR (2026)

22. Ren, J., Teuwen, J., Nijkamp, J., Rasmussen, M., Gouw, Z., Eriksen, J.G., et al.: Enhancing the reliability of deep learning-based head and neck tumour segmentation using uncertainty estimation with multi-modal images. PMB (2024)

23. Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., et al.: Self-supervised pre-training of swin transformers for 3d medical image analysis. In: CVPR (2022)

24. Trebeschi, S., van Griethuysen, J.J., Lambregts, D.M., Lahaye, M.J., Parmar, C., Bakers, F.C., Peters, N.H., Beets-Tan, R.G., Aerts, H.J.: Deep learning for fullyautomated localization and segmentation of rectal cancer on multiparametric mr. Scientific reports 7(1), 5301 (2017)

25. Wei, Z., Ren, J., Eriksen, J.G., Jensen, K., Mortensen, H.R., Korreman, S.S., Nijkamp, J.: An interactive deep-learning workflow for head and neck gross tumour volume segmentation. phiRO (2025)

26. Wu, L., Zhuang, J., Chen, H.: Large-scale 3d medical image pre-training with geometric context priors. IEEE TPAMI (2025)

27. Yang, S.X., Yu, J., Wang, M.: 21-gene recurrence score and survival outcomes in the phase iii multicenter tailorx clinical trial. Journal of the National Comprehensive Cancer Network 22(6), 376–381 (2024)