# Pretrained, Curriculum-Tuned, and Ensembled: A Tracer-Aware Interactive Segmentation Pipeline for AutoPET V

Xinglong Liang<sup>1,2†[0009−0001−3813−6726]</sup>, Chunyao Lu<sup>1,2†[0000−0002−6911−4036]</sup>, Tianyu Zhang<sup>1,2[0000−0001−9891−6874]</sup>, Jiaju Huang<sup>5[0009−0002−8979−0031]</sup>, Tao Tan<sup>5[0000−0002−4014−3458]</sup>, Yunchao Yin<sup>3[0000−0002−5790−484X]</sup>, and Lishan Cai<sup>1,3,4⋆[0000−0002−3282−291X]</sup>

<sup>1</sup> Department of Radiology, The Netherlands Cancer Institute, Plesmanlaan 121, 1066 CX, Amsterdam, the Netherlands

2 Department of Radiology and Nuclear Medicine, Radboud University Medical Centre, Geert Grooteplein 10, 6525 GA, Nijmegen, The Netherlands

3 Department of Radiation Oncology, Amsterdam University Medical Center, De Boelelaan 1117, 1081 HV Amsterdam, the Netherlands

4 GROW School for Oncology and Developmental Biology, Maastricht University Medical Centre, P. Debyelaan 25, 6229 AZ, Maastricht, The Netherlands

Faculty of Applied Sciences, Macao Polytechnic University, 999078, Macao Special Administrative Region of China

Abstract. Interactive lesion segmentation in whole-body PET/CT requires a model to provide a strong initial prediction while also responding eficiently to sparse corrective scribbles during inference. This setting is particularly challenging because tracer distributions, physiological uptake patterns, lesion appearance, and acquisition characteristics difer substantially between FDG and PSMA studies. We present TRIAGE, Tracer-aware Refinement via Interactive Anatomy-Guided sEgmentation. The core segmentation backbone is a 3D STU-Net initialized through masked autoencoding pre-training with an asynchronous masking strategy, aiming to learn transferable anatomical and cross-modal representations from PET/CT volumes before task-specific fine-tuning. In parallel, we train an auxiliary organ segmentation model whose predictions provide explicit anatomical context and help distinguish physiological uptake from malignant lesions. A dedicated tracer classifier first routes each study to an FDG- or PSMA-specific branch. Within each branch, a first-stage segmentation model consumes CT, PET, and organ context to generate an initial lesion mask. The initial prediction is then combined with cumulative foreground/background scribbles and refined by a second interactive segmentation network. The FDG and PSMA branches share the same overall processing pipeline but are trained independently to account for tracer-specific appearance and error modes. We additionally employ curriculum-style training and model ensembling

These authors contributed equally. ⋆ Corresponding author: l.cai@amsterdamumc.nl

to improve robustness across interaction steps and heterogeneous cohorts. Experiments are conducted using the oficial AutoPET V data and ten-fold split; quantitative results, ablations, and final test-set performance are left as placeholders to be completed after the challenge evaluation. Our framework is designed to jointly optimize strong automatic initialization and rapid correction under sparse human interaction. Code: https://github.com/Liiiii2101/AUTOPET2026-MEDAI.

Keywords: PET/CT · interactive segmentation · AutoPET · STU-Net · masked autoencoder · scribble guidance · tracer-aware learning

## 1 Introduction

Positron Emission Tomography and Computed Tomography (PET-CT) is widely used in oncology by combining functional information from PET with anatomical information from CT [5, 2]. Accurate lesion segmentation from whole-body PET-CT is important for quantitative tumor burden assessment and treatment response evaluation. However, automated segmentation remains challenging because lesions can be small, spatially scattered, and highly heterogeneous in appearance. In addition, physiological tracer uptake in normal organs can resemble pathological uptake and lead to false-positive predictions [1, 4].

The AutoPET V challenge focuses on interactive lesion segmentation in whole-body PET-CT. Instead of producing only a single automatic prediction, the segmentation model is expected to generate an initial lesion mask and subsequently refine it according to sparse user-provided scribbles. This setting introduces additional requirements beyond conventional automatic segmentation: the model should provide a reliable initial prediction, efectively interpret positive and negative corrections, and progressively improve the segmentation with each interaction. Moreover, the challenge includes both FDG and PSMA PET-CT scans, whose physiological uptake distributions and lesion appearances can difer considerably. Therefore, robust segmentation requires both multimodal PET-CT information and tracer-specific modeling.

Recent medical image segmentation frameworks such as nnU-Net [8] and STU-Net [7] have demonstrated strong performance on volumetric segmentation tasks. PET-CT-specific studies have further shown the benefit of combining metabolic PET information with anatomical CT context [10, 3, 9]. Nevertheless, directly training a single segmentation model from limited annotated PET-CT data may not fully exploit the complementary information between PET and CT, while physiological uptake can still introduce substantial false positives. These observations motivate the use of pre-training, anatomical priors, and tracerspecific models for the AutoPET V setting.

In this work, we present a tracer-aware interactive segmentation pipeline for AutoPET. Our framework is based on STU-Net-Small and consists of a pretraining stage followed by a two-stage interactive segmentation pipeline. We first perform masked autoencoder[6] pre-training using PET and CT with independently sampled spatial masks. For downstream segmentation, FDG and PSMA scans are processed by separately trained models following the same pipeline. The first-stage network takes PET and CT as input and generates an initial lesion segmentation. The second-stage network further incorporates anatomical organ masks and interaction information to refine the prediction after scribble corrections. In addition, curriculum-based training and model ensembling are employed to improve robustness during interactive inference.

Our approach is designed specifically for the AutoPET V task by combining multimodal pre-training, tracer-aware specialization, anatomical guidance, and interaction-driven refinement within a unified segmentation framework. Our contributions are summarized as follows:

We combine MAE-pretrained STU-Net features with organ-level anatomical context to strengthen the initial lesion segmentation and suppress physiological false positives.

– We formulate interactive refinement as a second-stage segmentation problem conditioned on the image channels, initial prediction, and cumulative scribble guidance.

– We use curriculum-style training over interaction dificulty and ensemble multiple folds to improve robustness across centers, tracers, and correction steps.

## 2 Methods

Our framework consists of a tracer-aware pre-training and interactive segmentation pipeline. The overall workflow is illustrated in Fig. 1. We first pre-train a STU-Net-Small backbone using a masked autoencoder strategy on paired PET and CT volumes. To better exploit the complementary information between the two modalities, PET and CT are masked independently using diferent spatial mask locations, with a masking ratio of 0.5 for each modality.

For downstream segmentation, FDG and PSMA cases are processed by separately trained models with the same network architecture and training pipeline. A tracer classification network is used to determine whether an input case belongs to the FDG or PSMA domain and route it to the corresponding segmentation model.

The segmentation pipeline contains two stages. The first-stage model takes PET, CT, and anatomical organ masks as input and generates an initial lesion prediction. The second-stage model further refines this prediction by incorporating PET, CT, anatomical organ masks, the initial prediction, and user-provided scribble information. Curriculum-based training is adopted for the refinement stage, and multiple trained models are ensembled during inference to improve robustness.

## 2.1 Data

We used only the oficial AutoPET V training dataset and did not incorporate any external imaging data. The dataset contains 1,014 FDG PET/CT studies from 900 patients and 597 PSMA PET/CT studies from 378 patients. The FDG cohort was acquired at the University Hospital Tübingen and includes patients with malignant melanoma, lymphoma, or lung cancer, as well as negative control cases. The PSMA cohort was acquired at LMU University Hospital Munich and contains prostate cancer studies both with and without PSMA-avid tumor lesions.

![](images/55db4e98f45d4a6d615fc3581715a1657bf63b5d49e7300d7255e52570ef698e.jpg)  
Fig. 1. Overview of the proposed TRIAGE framework. A STU-Net backbone is first pre-trained on paired PET/CT volumes using masked autoencoding. For downstream segmentation, a tracer classifier routes each case to an FDG- or PSMA-specific pipeline. A seperate organ model provides anatomical context, and the first-stage model combines PET, CT, and organ predictions to generate the initial lesion prediction. The second-stage model then integrates the image channels, organ predictions, initial mask, and cumulative foreground/background scribbles to iteratively refine the segmentation.

Each study consists of a whole-body CT volume, a corresponding PET volume in SUV, and a binary manual segmentation of tracer-avid tumor lesions. PET and CT were acquired in the same imaging session and were provided in a spatially aligned form. The released training data follow the nnU-Net NIfTI format, with CT and PET represented as separate image channels. Tumor lesions were manually annotated by experienced readers based on joint PET/CT assessment and clinical information.

We followed the nnU-Net framwork for model development. FDG and PSMA cases were treated as two separate tracer domains, and independent models were trained for the two cohorts using the same overall pipeline.

## 2.2 Data pre-processing

We used the default nnU-Net pre-processing and planning pipeline. For selfsupervised pre-training, PET/CT volumes from both tracers were sampled using fixed 128 × 128 × 128 voxel patches. For supervised segmentation, tracer-specific nnU-Net configurations were used. FDG images were resampled to a target spacing of (3.0, 2.03, 2.03) mm and trained with 128×128×128 patches. PSMA images were resampled to (4.07, 3.27, 4.07) mm and trained with 112×192×112 patches.

All values were automatically determined by the default nnU-Net planning procedure.

## 2.3 Data post-processing

To suppress false positives arising from physiological tracer uptake, we applied a tracer-specific SUV thresholding step to the predicted segmentation. For each voxel predicted as lesion, the corresponding PET SUV value was checked against a fixed threshold, with voxels below the threshold reassigned to background; the threshold was set to 1.5 g/mL for FDG and $1 . 0 ~ \mathrm { g / m L }$ for PSMA to reflect the difering physiological uptake characteristics of the two tracers. The thresholded prediction was then binarized to produce the final lesion mask.

## 2.4 Training and test parameters

Using the pretrained encoder, segmentation training proceeded in three stages. First, the encoder was frozen and the decoder was warmed up alone for 50 epochs with a small learning rate of $1 \times 1 0 ^ { - 5 }$ . Next, the encoder was unfrozen and the entire network was warmed up for a further 50 epochs. Finally, the full model was trained with a learning rate of $1 \times 1 0 ^ { - 4 }$ for a total of 1000 epochs following the default nnU-Net training strategy [8], optimizing the sum of the Dice and cross-entropy losses. More details can be found in Table 4.

## 2.5 Github repository

Link to Github repository: https://github.com/Liiiii2101/AUTOPET2026-MEDAI

## 3 Results

We evaluated the proposed framework using ten-fold cross-validation separately on the FDG and PSMA cohorts. Tables 1 and 2 summarize the validation Dice scores obtained by the first-stage automatic segmentation model and the secondstage refinement model, respectively.

For the first-stage model, the mean validation Dice was $0 . 5 8 8 9 \pm 0 . 0 5 3 7$ for FDG and $0 . 5 8 3 3 { \pm } 0 . 0 2 9 3$ for PSMA. Performance varied across folds, with FDG Dice scores ranging from 0.5093 to 0.6551 and PSMA scores ranging from 0.5343 to 0.6234. These results reflect the dificulty of generating a reliable initial wholebody lesion segmentation from PET/CT alone, particularly in the presence of heterogeneous lesion appearance and physiological tracer uptake.

The second-stage model consistently achieved higher average Dice scores than the first-stage model. The mean Dice increased to $0 . 6 1 7 5 \pm 0 . 0 4 3 3$ for FDG and 0.6405±0.0270 for PSMA, corresponding to absolute improvements of 0.0286 and 0.0572, respectively. The improvement was particularly pronounced for PSMA, suggesting that the additional anatomical context, initial prediction, and interaction guidance used by the refinement stage provide complementary information beyond the original PET/CT channels.

Table 1. Stage-1 validation Dice scores for FDG and PSMA across 10 folds.
<table><tr><td>Fold</td><td>FDG Dice</td><td>PSMA Dice</td></tr><tr><td>0</td><td>0.5534</td><td>0.5990</td></tr><tr><td>1</td><td>0.5750</td><td>0.5343</td></tr><tr><td>2</td><td>0.6551</td><td>0.5949</td></tr><tr><td>3</td><td>0.5124</td><td>0.5712</td></tr><tr><td>4</td><td>0.6464</td><td>0.5579</td></tr><tr><td>5</td><td>0.6365</td><td>0.6074</td></tr><tr><td>6</td><td>0.6221</td><td>0.5966</td></tr><tr><td>7</td><td>0.5642</td><td>0.6234</td></tr><tr><td>8</td><td>0.5093</td><td>0.6025</td></tr><tr><td>9</td><td>0.6145</td><td>0.5459</td></tr><tr><td>Mean</td><td>0.5889</td><td>0.5833</td></tr><tr><td>Mean ± Std.</td><td>0.5889 ± 0.0537 0.5833 ± 0.0293</td><td></td></tr></table>

Table 2. Stage-2 validation Dice scores for FDG and PSMA across 10 folds. ‘\_’ indicates that training was incomplete; parameters from the intermediate checkpoint were used.
<table><tr><td>Fold</td><td>FDG Dice</td><td>PSMA Dice</td></tr><tr><td>0</td><td>0.5604</td><td>0.6717</td></tr><tr><td>1</td><td>0.6278</td><td>0.6134</td></tr><tr><td>2</td><td>0.6621</td><td>0.6383</td></tr><tr><td>3</td><td>0.5565</td><td>0.5937</td></tr><tr><td>4</td><td></td><td>0.6153</td></tr><tr><td>5</td><td>0.6393</td><td>0.6853</td></tr><tr><td>6</td><td>0.6766</td><td>0.6609</td></tr><tr><td>7</td><td>0.6270</td><td>0.6340</td></tr><tr><td>8</td><td>0.5630</td><td>0.6357</td></tr><tr><td>9</td><td>0.6446</td><td>0.6562</td></tr><tr><td>Mean</td><td>0.6175</td><td>0.6405</td></tr><tr><td>Mean ± Std.</td><td colspan="2">0.6175 ± 0.0433 0.6405 ± 0.0270</td></tr></table>

Across the evaluated folds, the second-stage PSMA model achieved a maximum Dice of 0.6853, while the FDG model reached a maximum Dice of 0.6766. For FDG, fold 4 training was incomplete and was therefore excluded from the reported mean and standard deviation; the remaining nine folds were used for the aggregate statistics.

Overall, these cross-validation results demonstrate that the second-stage refinement strategy improves segmentation accuracy over the initial automatic prediction for both tracer domains. The larger improvement observed for PSMA further supports the use of tracer-specific models rather than a single shared model across both PET tracers.

Table 3. Ablation study of network architecture and pre-training on one FDG validation fold from an earlier five-fold experimental setting. Organ context was not used in these experiments. All models were trained and evaluated using the same data split and validation protocol.
<table><tr><td>Method</td><td>Parameters (M)</td><td>FLOPs</td><td>Dice</td></tr><tr><td>nnU-Net</td><td>30.79</td><td>526.29</td><td>0.3990</td></tr><tr><td>STU-Net</td><td>14.55</td><td>138.70</td><td>0.3890</td></tr><tr><td>STU-Net + Pre-training</td><td>14.55</td><td>138.70</td><td>0.4889</td></tr></table>

## 4 Discussion

The cross-validation experiments show that the proposed two-stage strategy provides a clear benefit over the initial automatic segmentation for both FDG and PSMA PET/CT. While the first-stage model is responsible for generating a strong initial lesion mask without user interaction, the second-stage model has access to additional information, including anatomical organ predictions, the initial segmentation, and foreground/background scribbles. The improvement from a mean Dice of 0.5889 to 0.6175 for FDG and from 0.5833 to 0.6405 for PSMA indicates that these additional cues are useful for correcting errors that cannot be resolved from PET and CT alone.

The larger Stage-2 improvement observed for PSMA suggests that anatomical and interaction guidance may be particularly beneficial for this tracer. PSMA PET has tracer-specific physiological uptake patterns that difer substantially from FDG, motivating our decision to train independent FDG and PSMA models rather than forcing both domains into a single segmentation network. Tracerspecific training allows each model to specialize in its corresponding uptake distribution, lesion appearance, and characteristic false-positive patterns while retaining the same overall architecture and training strategy.

The ablation experiment further demonstrates the efectiveness of pre-training and the eficiency of the adopted STU-Net backbone. This experiment was conducted on a single FDG validation fold from an earlier five-fold setting without organ context, and is therefore intended only to compare the relative efects of network architecture and pre-training under the same experimental conditions. STU-Net trained from scratch achieved a mean validation Dice of 0.3890, which was comparable to the nnU-Net baseline of 0.3990, while requiring substantially fewer parameters and FLOPs (14.55M parameters and 138.70 GFLOPs vs. 30.79M and 526.29 GFLOPs). After initializing STU-Net with the proposed pre-trained weights, the validation Dice increased to 0.4889, corresponding to an absolute gain of 0.0999 over the same architecture trained from scratch. Importantly, this improvement was achieved without increasing the model size or inference complexity.

The current results demonstrate a consistent advantage of the refinement stage across both tracer domains and support the overall design of combining tracer-specific specialization, anatomical context, and interaction-aware refinement for whole-body PET/CT lesion segmentation.

## 5 Conclusion

In this article, we presented a tracer-aware two-stage framework for interactive whole-body lesion segmentation in FDG and PSMA PET/CT. The approach combines masked-autoencoder pre-training, tracer-specific segmentation models, anatomical organ guidance, and scribble-conditioned refinement within a unified pipeline. Ten-fold cross-validation showed that the second-stage refinement model improved the mean Dice from 0.5889 to 0.6175 for FDG and from 0.5833 to 0.6405 for PSMA compared with the initial segmentation stage. These results demonstrate the benefit of incorporating anatomical context and interaction information for refining PET/CT lesion predictions. Future work will evaluate the framework on the oficial AutoPET V test set and further investigate the contributions of individual components through systematic ablation studies.

Acknowledgments. The authors would like to acknowledge the Research High-Performance Computing (RHPC) facility of the Netherlands Cancer Institute (NKI).

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Barrington, S.F., O’doherty, M.J.: Limitations of pet for imaging lymphoma. European Journal of Nuclear Medicine and Molecular Imaging 30, S117–S127 (2003)

2. Endo, K., Oriuchi, N., Higuchi, T., Iida, Y., Hanaoka, H., Miyakubo, M., Ishikita, T., Koyama, K.: Pet and pet/ct using 18 f-fdg in the diagnosis and management of cancer patients. International journal of clinical oncology 11, 286–296 (2006)

3. Fu, X., Bi, L., Kumar, A., Fulham, M., Kim, J.: Multimodal spatial attention module for targeting multimodal pet-ct lung tumor segmentation. IEEE Journal of Biomedical and Health Informatics 25(9), 3507–3516 (2021)

4. Gontier, E., Fourme, E., Wartski, M., Blondet, C., Bonardel, G., Le Stanc, E., Mantzarides, M., Foehrenbach, H., Pecking, A.P., Alberini, J.L.: High and typical 18 f-fdg bowel uptake in patients treated with metformin. European journal of nuclear medicine and molecular imaging 35, 95–99 (2008)

5. Groheux, D., Espié, M., Giacchetti, S., Hindié, E.: Performance of fdg pet/ct in the clinical management of breast cancer. Radiology 266(2), 388–405 (2013)

6. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022)

7. Huang, Z., Wang, H., Deng, Z., Ye, J., Su, Y., Sun, H., He, J., Gu, Y., Gu, L., Zhang, S., et al.: Stu-net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training. arXiv preprint arXiv:2304.06716 (2023)

8. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

9. Liang, X., Huang, J., Zhang, T., Han, L., Wang, X., Gao, Y., Lu, C., Sun, Y., Teuwen, J., Tan, T., et al.: Leveraging modality-guided pre-training for dualprompt-driven multi-cancer pet-ct segmentation. Medical Image Analysis p. 104182 (2026)

10. Zhao, X., Li, L., Lu, W., Tan, S.: Tumor co-segmentation in pet/ct using multimodality fully convolutional neural network. Physics in Medicine & Biology 64(1), 015011 (2019)

Table 4. Algorithm details
<table><tr><td>Team name</td><td>algorithm name</td><td>data pre-processing processing</td><td>data post-</td><td>training data augmentation</td></tr><tr><td>MEDAI</td><td>TRIAGE Tracer-aware Refinement via Interactive Anatomy- Guided sEgmentation</td><td>nnUNet Preprocessing</td><td>SUV thresholding (1.5 Brightness, g/mL FDG, 1.0 Random g/mL PSMA)</td><td>Random Gamma, Random Rotation</td></tr><tr><td>test time</td><td>ensembling</td><td>standardized</td><td>network</td><td>loss</td></tr><tr><td>augmentation</td><td>Tracer-based routing (Based on the</td><td>framework? nnUNet v2 (3D)</td><td>architecture STUNet-small (3D)</td><td>DSC + CE</td></tr><tr><td></td><td>pre-trained 10-fold FDG and 10-fold PSMA models.)</td><td></td><td></td><td></td></tr><tr><td>training data</td><td>data/model dimensionality pre-trained and size 1014 FDG + 597 For FDG, the 3D Own developed</td><td>use of models</td><td>GPU hardware for training 1x Nvidia a6000</td><td></td></tr><tr><td>PSMA PET-CT volume size is of autoPET</td><td>128 × 128 × 128, while for PSMA, it is 112 × 192 × 112.</td><td></td><td></td><td></td></tr></table>