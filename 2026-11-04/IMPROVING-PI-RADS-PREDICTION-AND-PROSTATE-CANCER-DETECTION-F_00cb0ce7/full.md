csPCa

PI-RADS

# IMPROVING PI-RADS PREDICTION AND PROSTATE CANCER DETECTION FROM MULTIPARAMETRIC MRI WITH ANATOMICAL GUIDANCE

Seungbin Park<sup>⋆</sup>, Thibault Heintz<sup>⋆</sup>, Rachit Saluja<sup>†⋆</sup>, Francesca Khani<sup>⋆</sup>, Timothy McClure<sup>⋆</sup>, Daniel Margolis<sup>⋆</sup>, Mert R. Sabuncu<sup>⋆†</sup>, Heejong Kim<sup>⋆‡</sup>

<sup>⋆</sup> Weill Cornell Medicine, New York, NY, USA <sup>†</sup> Cornell Tech, New York, NY, USA

## ABSTRACT

Accurate prediction of Prostate Imaging Reporting and Data System (PI-RADS) scores from multiparametric MRI (mpMRI) remains challenging, more so than clinically significant prostate cancer (csPCa) detection. Given the importance of lesion location in PI-RADS scoring, we evaluated whether incorporating prostate zonal masks improves PI-RADS prediction and csPCa detection across multiple architectures (3D ResNet-18, ResNet-50, and ViT) using a large-scale internal dataset. Anatomical guidance generally improved performance across architectures, with the best results achieved by ResNet-18 with 2-zone guidance for PI-RADS prediction (QWK = 0.511) and ResNet-50 with 5-zone guidance for csPCa detection (AUC = 0.809). Fine-tuned results on external datasets further demonstrated the benefits of anatomical guidance. Furthermore, models with anatomical guidance improved the utility of model-predicted PI-RADS scores for PI-RADS-based csPCa detection (positive defined as PI-RADS≥3). Combining model-predicted and expert-assigned PI-RADS scores significantly increased the AUC from 0.711 to 0.756 $( p \ < \ 0 . 0 5 )$ , suggesting that AI-derived PI-RADS scores may provide complementary information to expert assessment for csPCa detection. Overall, our findings highlight the value of incorporating prostate zonal anatomy into deep learning-based prostate MRI assessment.

Index Terms— Prostate MRI, PI-RADS, Clinically significant prostate cancer, Deep learning, Anatomical guidance

## 1. INTRODUCTION

Multiparametric magnetic resonance imaging (mpMRI) is widely used to identify suspicious prostate lesions and guide biopsy decisions. The Prostate Imaging Reporting and Data System (PI-RADS) [1] standardizes mpMRI interpretation by assigning scores from 1 to 5, with a score of 5 indicating clinically significant prostate cancer (csPCa) is highly likely. In clinical practice, PI-RADS≥3 is a common threshold for biopsy referral, and subsequent histopathological evaluation provides the definitive csPCa diagnosis [2].

![](images/2dc193bb2763b6774d0368ea0e456f2fd735f81a1d7984124a54a9400ea613ca.jpg)

![](images/2a7cd9a778f493e1f8f5afabf351c5017b38e4ab7ced85c781b34c12ff00a297.jpg)

![](images/d669b79a9e98627876666629469ee9b78fdec8e3b69f5383aa0327c429f67e1e.jpg)

![](images/3e91324a93237c43c2ef05bdca934d88be8c3049db9ac6e146f1a851aaddd125.jpg)

![](images/2fb01cf8d75550293511c83e402925c85d4002d61b3aed502614a9ce578a8626.jpg)

![](images/de22f6ffa9f15e3ef9a2e601c57088c467cb75ed7f0f0844401a9c5301b2fcd3.jpg)  
Fig. 1. Distribution of PI-RADS and csPCa labels across the internal, UCSF, and PMUB datasets.

Accurate PI-RADS assessment is essential for biopsy decisions. Compared with csPCa detection, automated PI-RADS prediction from mpMRI has received relatively limited attention. Existing deep learning (DL) approaches have demonstrated modest predictive performance [3, 4, 5]. Challenges stem from substantial inter-reader variability [6], overlapping imaging characteristics of benign and malignant lesions, and imbalanced PI-RADS score distributions [7]. Publicly available prostate MRI datasets are relatively small and have highly imbalanced PI-RADS score distributions, as they primarily include patients who underwent biopsy [7] (Fig. 1). To overcome the limitations of existing datasets, we trained and evaluated DL models on a large-scale in-house cohort of 13, 473 mpMRI examinations with patient-level PI-RADS scores.

Prostate zonal anatomy determines how a lesion is assessed in PI-RADS. Diffusion-weighted imaging (DWI) is the dominant sequence for peripheral zone (PZ) lesions, whereas T2-weighted (T2W) imaging is dominant in the transition zone (TZ)[1]. Radiologists interpret imaging findings in the context of zonal anatomy, yet most DL models learn directly from mpMRI without such guidance. Previous studies have shown that incorporating anatomical guidance can improve prostate cancer detection and lesion characterization [9, 10, 11]. However, the impact of anatomical guidance on PI-RADS prediction remains underexplored.

T2W  
DWI  
Table 1. Summary of dataset sizes across training, validation, and test sets. Adjusted csPCa labels treat PI-RADS 1 and 2 cases as negative. Pseudo PI-RADS is radiologistassigned suspicion scores which approximately correspond to PI-RADS v2 [8].
<table><tr><td>Dataset</td><td>Task</td><td>Train</td><td>Val</td><td>Test</td><td>Total</td></tr><tr><td>Internal</td><td>PI-RADS csPCa (adjusted)</td><td>10,852 6,491</td><td>688 437</td><td>1,933 1,257</td><td>13,473 8,185</td></tr><tr><td>UCSF</td><td>PI-RADS csPCa</td><td>678 680</td><td>98 98</td><td>195 195</td><td>971 973</td></tr><tr><td>PMUB</td><td>PI-RADS (pseudo) csPCa</td><td>159 167</td><td>95 100</td><td>382 401</td><td>636 668</td></tr></table>

In this work, we incorporate prostate zone masks as anatomical guidance into DL models and systematically evaluate its effect on PI-RADS and csPCa prediction. We evaluate 3D Vision Transformer (ViT), ResNet-18, and ResNet-50 architectures with and without 2-zone or 5-zone anatomical guidance using internal dataset and two external datasets. We further assessed the clinical utility of model-predicted PI-RADS scores by comparing model-predicted and expertassigned PI-RADS scores for PI-RADS-based csPCa detection (positive defined as PI-RADS≥3). Our results demonstrate that anatomical guidance consistently improves model performance with generalizability across datasets. We publicly release the training code (https://github.com/seungbin20 1803/prostate-anatomy-guidance-ai) and model weights (https: //huggingface.co/spark-25/prostate-anatomy-guidance-ai).

## 2. METHODS AND EXPERIMENTS

## 2.1. Datasets

We used an internal dataset and two external datasets: UCSF-ProstateMR (UCSF) [12] and Prostate-MRI-US-Biopsy (PM UB) [8]. Datasets include mpMRI sequences comprising T2W, DWI, and apparent diffusion coefficient (ADC) maps, along with patient-level PI-RADS scores and csPCa labels. For csPCa prediction, analyses were restricted to cases with biopsy-confirmed outcomes. Dataset sizes and split distributions are summarized in Table 1. All images across datasets were resampled to $1 \times 1 \times 3 ~ \mathrm { m m ^ { 3 } }$ and cropped to a volume of $1 1 2 \times 1 1 2 \times 2 4$ voxels centered on the prostate using the prostate masks provided in the datasets. For ProFound (see 2.4), images were resampled to $0 . 5 \times 0 . 5 \times 1 ~ \mathrm { m m ^ { 3 } }$ and cropped to $2 2 4 \times 2 2 4 \times 6 4$ voxels, to match the pretrained encoder.

ADC  
![](images/ea01285886fec9348c8e774f00f83933ffe21e801341d022bc1a4a7ec8cb3927.jpg)  
Fig. 2. Representative T2W, DWI, and ADC images with corresponding five-zone prostate segmentation masks. PZ: peripheral zone; CZ: central zone; TZ: transition zone; AFS: anterior fibromuscular stroma.

## 2.2. Zonal Segmentation

We constructed zonal masks of PZ, central zone (CZ), TZ, anterior fibromuscular stroma (AFS), and urethra from T2W using a publicly available pretrained 3D nnU-Net [13]. The 5- zone guidance includes all five anatomical regions as separate masks, whereas the 2-zone configuration combined the CZ, TZ, AFS, and urethra into a single TZ mask while preserving the PZ mask. Fig. 2 shows T2W, DWI, and ADC images, along with prostate five-zone segmentation masks.

## 2.3. Model Development

We trained 3D ViT [14], ResNet-18, and ResNet-50 [15] models with and without prostate zonal masks. T2W, DWI, and ADC volumes were concatenated along the channel dimension, and zonal masks were appended as additional channels when anatomical guidance was used.

PI-RADS prediction was formulated as a five-class ordinal classification task using a CORAL head and loss [16] to account for the inherent ordering of PI-RADS scores. For csPCa detection, models were trained as binary classifiers using cross-entropy loss. All models were optimized with AdamW using learning rates of $ { \mathbf { \dot { 2 } } } \times  { 1 0 ^ { - 4 } }$ for models and learning rate $1 \times 1 0 ^ { - 3 }$ for the classification layers, weight decay of $1 \times 1 0 ^ { - 5 }$ , and cosine learning rate schedule. Batch size was 4 with 16 gradient accumulation steps, yielding an effective batch size of 64.

Models were trained for up to 200 epochs with early stopping if the validation metric failed to improve by at least 0.005 for 15 consecutive epochs. Model selection was based on quadratic weighted kappa (QWK) for PI-RADS prediction and area under the receiver operating characteristic curve (AUC) for csPCa detection.

Table 2. Performance for PI-RADS prediction and csPCa detection on the internal dataset. Values are reported as point estimates with 95% confidence intervals. Underlined values indicate the best-performing model. Statistical significance was assessed against the corresponding model without anatomical guidance (gray) using paired tests $( ^ { * } p < 0 . 0 5 )$
<table><tr><td rowspan="2"></td><td>PI-RADS (Ordinal multi-label; 1,2, 3, 4,5)</td><td>csPCa (Binary-label;</td></tr><tr><td>QWK</td><td>Positive/Negative) AUC</td></tr><tr><td>ProFound-Conv</td><td>0.181 (0.141, 0.217)</td><td>0.710 (0.680, 0.741)</td></tr><tr><td>3D ViT</td><td>0.182 (0.140, 0.225)</td><td>0.636 (0.604, 0.669)</td></tr><tr><td>3D ViT+2zones</td><td>0.215 (0.172, 0.257)</td><td>0.678 (0.648, 0.708)*</td></tr><tr><td>3D ViT+5zones</td><td>0.232 (0.190, 0.272)*</td><td>0.689 (0.659, 0.719)*</td></tr><tr><td>3D ResNet-18</td><td>0.471 (0.434, 0.507)</td><td>0.789 (0.761, 0.816)</td></tr><tr><td>3D ResNet-18+2zones</td><td>0.511 (0.474, 0.544)*</td><td>0.797 (0.770, 0.823)</td></tr><tr><td>3D ResNet-18+5zones</td><td>0.488 (0.449, 0.524)</td><td>0.787 (0.759, 0.817)</td></tr><tr><td>3D ResNet-50</td><td>0.453 (0.416, 0.489)</td><td>0.781 (0.752, 0.809)</td></tr><tr><td>3D ResNet-50+2zones</td><td>0.498 (0.460, 0.532)*</td><td>0.804 (0.777, 0.831)*</td></tr><tr><td>3D ResNet-50+5zones</td><td>0.482 (0.446, 0.519)*</td><td>0.809 (0.782, 0.834)*</td></tr></table>

## 2.4. Experimental Design

We first evaluated the effect of 2-zone and 5-zone guidance across ViT, ResNet-18, and ResNet-50 architectures for both PI-RADS prediction and csPCa detection, on the internal dataset. To benchmark our approach against a state-of-the-art method, we compared our approach with ProFound [5], a foundation model for prostate MRI. Publicly available pretrained encoder weights were used, and only the classification layers were fine-tuned.

To assess generalizability, models pretrained on the internal data were subsequently fine-tuned and evaluated on two external datasets (UCSF and PMUB). PI-RADS 1 and 2 cases were excluded because of their small sample sizes. During fine-tuning, pretrained models and randomly initialized classification layers to account for the restricted PI-RADS score range in the external datasets were optimized. Fine-tuning was performed for up to 50 epochs using learning rates of $2 \times 1 0 ^ { - 5 }$ and $1 \times 1 0 ^ { - 3 }$ , respectively, with a weight decay of $1 \times 1 0 ^ { - 2 }$ . The learning rates were linearly warmed up over the first two epochs, followed by cosine decay. Early stopping was applied with a patience of 15 epochs and a minimum validation improvement of 0.01, after an initial 10-epoch training period.

We also evaluated the clinical utility of model-predicted PI-RADS scores for PI-RADS-based csPCa detection, defining $\mathrm { P I } { \cdot } \mathrm { R A D S } \geq 3$ as a positive. Expert-assigned PI-RADS scores, model-predicted PI-RADS scores, and their ensemble were compared for PI-RADS-based csPCa detection. For ensembling, expert-assigned PI-RADS scores were linearly rescaled to [0, 1] using (PI-RADS - 1)/4 and averaged with the model-predicted probability of $\mathrm { P I } { \cdot } \mathrm { R A D S } \geq 3$ to obtain a combined score.

Table 3. Performance on external datasets of models trained from scratch or fine-tuned from models pretrained on the internal dataset. Values are point estimates with 95% confidence intervals. Statistical significance was tested against the corresponding model without anatomical guidance (gray) using paired tests $( ^ { * } p < 0 . 0 5 )$ .
<table><tr><td rowspan="2"></td><td>PI-RADS (Ordinal multi-label; 3,4,5)</td><td>csPCa (Binary-label; Positive/Negative)</td></tr><tr><td>QWK</td><td>AUC</td></tr><tr><td rowspan="7">UCSF</td><td>3D ResNet-18, scratch 0.071 (-0.045, 0.184)</td><td>0.503 (0.416, 0.580)</td></tr><tr><td>3D ResNet-18 0.152 (0.020, 0.280)</td><td>0.633 (0.552, 0.706)</td></tr><tr><td>3D ResNet-18+2zones 0.167 (0.032, 0.291)</td><td>0.668 (0.586, 0.746)</td></tr><tr><td>3D ResNet-18+5zones</td><td>0.125 (-0.007, 0.253) 0.624 (0.541, 0.697)</td></tr><tr><td>3D ResNet-50, scratch 0.053 (-0.064, 0.161)</td><td>0.590 (0.509, 0.671)</td></tr><tr><td>3D ResNet-50 -0.005 (-0.133, 0.140)</td><td>0.604 (0.522, 0.679)</td></tr><tr><td>3D ResNet-50+2zones 0.134 (0.017, 0.252)</td><td>0.611 (0.533, 0.689)</td></tr><tr><td>3D ResNet-50+5zones</td><td>0.231 (0.116, 0.336)*</td><td>0.605 (0.520, 0.681)</td></tr><tr><td rowspan="8">PMUB</td><td>3D ResNet-18, scratch</td><td>0.065 (-0.020, 0.160)</td><td>0.519 (0.462, 0.576)</td></tr><tr><td>3D ResNet-18</td><td>0.296 (0.204, 0.391)</td><td>0.659 (0.609, 0.711)</td></tr><tr><td>3D ResNet-18+2zones</td><td>0.368 (0.279, 0.454)</td><td>0.668 (0.616, 0.721)</td></tr><tr><td>3D ResNet-18+5zones</td><td>0.251 (0.154, 0.340)</td><td>0.683 (0.631, 0.733)</td></tr><tr><td>3D ResNet-50, scratch</td><td>0.022 (-0.072, 0.127)</td><td>0.585 (0.526, 0.642)</td></tr><tr><td>3D ResNet-50</td><td>0.282 (0.191, 0.374)</td><td>0.594 (0.541, 0.648)</td></tr><tr><td>3D ResNet-50+2zones</td><td>0.356 (0.259, 0.448)</td><td>0.586 (0.533, 0.641)</td></tr><tr><td>3D ResNet-50+5zones</td><td>0.365 (0.280, 0.451)</td><td>0.640 (0.586, 0.692)</td></tr></table>

## 3. RESULTS

## 3.1. Internal Evaluation

Table 2 shows that incorporating anatomical guidance generally improved both PI-RADS prediction and csPCa detection across architectures. For ViT, 5-zone guidance significantly improved both PI-RADS prediction (QWK: 0.182 to 0.232) and csPCa detection (AUC: 0.636 to 0.689; both $p < 0 . 0 5 )$ Similarly, 2-zone guidance significantly improved PI-RADS prediction for both ResNet-18 (QWK: 0.471 to 0.511) and ResNet-50 (QWK: 0.453 to 0.498; $p \ < \ 0 . 0 5 )$ . For csPCa detection, ResNet-50 benefited from both 2-zone guidance (AUC: 0.781 to 0.804; $p < 0 . 0 5 )$ ) and 5-zone guidance (AUC: 0.809; $p ~ < ~ 0 . 0 5 )$ . Overall, ResNet-18 with 2-zone guidance achieved the highest PI-RADS prediction performance (QWK = 0.511), whereas ResNet-50 with 5-zone guidance achieved the highest csPCa detection performance (AUC = 0.809).

Error analysis suggested that model performance was driven primarily by the reference labels. For ResNet-18 +2zones, PI-RADS 1 and 4 cases were rarely predicted correctly, and csPCa-positive cases were substantially more challenging than negative ones (39% vs. 95%). Neither patient age, reporting radiologist, nor examination type (initial versus follow-up) was significantly associated with performance. Notably, correct PI-RADS prediction strongly correlated with accurate csPCa detection (88% vs. 65%; odds ratio = 4.04, $p < 0 . 0 0 1 $ ).

Table 4. Comparison of PI-RADS-based csPCa detection (PI-RADS≥3 considered positive) performance using modelpredicted, expert-assigned, and combined PI-RADS scores, along with direct csPCa detection by the model shown in the last row. Values are point estimates with 95% confidence intervals. Statistical significance was assessed against expert scores (gray) using two-sided DeLong tests $( ^ { * } p < 0 . 0 5 )$ .
<table><tr><td>ProFound-Conv</td><td>AUC</td></tr><tr><td>ViT ViT+2zones ViT+5zones ResNet-18</td><td>0.597 (0.547, 0.641)* 0.588 (0.539, 0.634)* 0.639 (0.593, 0.679)* 0.653 (0.607, 0.694)* 0.678 (0.637, 0.717)</td></tr><tr><td>ResNet-18+2zones ResNet-18+5zones ResNet-50 ResNet-50+2zones ResNet-50+5zones</td><td>0.710 (0.668, 0.750) 0.727 (0.688, 0.764) 0.704 (0.661, 0.742) 0.729 (0.689, 0.771) 0.708 (0.668, 0.749)</td></tr><tr><td>Expert Expert + ResNet-18 Expert + ResNet-18+2zones</td><td>0.711 (0.669, 0.750) 0.735 (0.694, 0.774)* 0.747 (0.705, 0.785)*</td></tr><tr><td>Expert + ResNet-18+5zones Expert + ResNet-50</td><td>0.756 (0.716, 0.793)*</td></tr><tr><td>Expert + ResNet-50+2zones Expert + ResNet-50+5zones ResNet-50+5zones, csPCa</td><td>0.745 (0.703, 0.782)* 0.756 (0.714, 0.795)* 0.749 (0.708, 0.786)* 0.809 (0.782, 0.834)</td></tr></table>

## 3.2. External evaluation

Table 3 summarizes performance on the UCSF and PMUB datasets. Fine-tuning models pretrained on the internal cohort consistently outperformed training from scratch, indicating that representations learned from the large internal dataset provided a useful initialization for external adaptation. Anatomical guidance continued to provide benefits across both datasets. On UCSF, 5-zone guidance significantly improved ResNet-50 PI-RADS prediction (QWK: -0.005 to 0.231; $\begin{array} { r } { \begin{array} { r l r } { p } & { { } < } & { 0 . 0 5 ) } \end{array} } \end{array}$ , while ResNet-18+2-zones achieved the highest csPCa AUC (0.668). On PMUB, ResNet-18+2- zones achieved the highest PI-RADS QWK (0.368), whereas ResNet-18+5-zones achieved the highest csPCa AUC (0.683). These results suggest that the benefits of zonal information generalize beyond a single training cohort.

## 3.3. Clinical utility of model-predicted PI-RADS scores

We compared csPCa detection based on expert-assigned PI-RADS scores, model-predicted PI-RADS scores, and their combination (Table 4). While ProFound-Conv and ViT underperformed expert PI-RADS assessment, anatomy-guided ResNet models produced PI-RADS predictions that were at least comparable to expert scoring for csPCa discrimination. The best model-predicted PI-RADS score (ResNet-50+2- zones) achieved an AUC of 0.729, exceeding that of expertassigned PI-RADS scores (0.711). Combining expert and model-predicted PI-RADS scores yielded the strongest performance $( \mathrm { A U C } = 0 . 7 5 6 , p < 0 . 0 5 )$ , indicating that the two sources of information provide complementary value.

## 4. CONCLUSION

We evaluated the impact of anatomical guidance on DL models for PI-RADS prediction and csPCa detection. Across architectures trained using an internal dataset and fine-tuned using external datasets, incorporating prostate zonal masks generally improved performance. Anatomical guidance improved the performance of model-predicted PI-RADS scores and csPCa detection, and enhanced PI-RADS-based detection of csPCa. Combining expert-assigned and model-predicted PI-RADS scores improved PI-RADS-based csPCa detection, suggesting that AI-derived PI-RADS scores may provide complementary information for csPCa detection. These findings highlight the value of incorporating anatomical context into automated prostate MRI assessment.

This study has several limitations. First, segmentation errors in anatomical guidance may propagate to downstream tasks. Second, although external validation was performed on two independent datasets, both are limited to only PI-RADS 3–5 cases, limiting evaluation across the full PI-RADS spectrum. Future work should investigate whether the benefits of anatomical guidance generalize across more datasets and downstream clinical tasks.

## 5. COMPLIANCE WITH ETHICAL STANDARDS

This retrospective study was approved by the Institutional Review Board (IRB), with a waiver of informed consent, and was conducted in compliance with the Health Insurance Portability and Accountability Act (HIPAA).

## 6. ACKNOWLEDGMENTS

This work was partly supported by the National Institutes of Health grants K25CA283145, R01AG053949, R37CA282407, R01AG064027, and R01AG070988. Mert R. Sabuncu serves as a consultant or advisor to Cleerly Inc. and Medara Health, and holds equity interests. Daniel Margolis has served as a clinical advisor to Stratagen Bio and Ravada Health and holds unexercised stock options. Claude Code was used for partial code implementation.

## 7. REFERENCES

[1] Baris Turkbey, Andrew B. Rosenkrantz, Masoom A. Haider, Anwar R. Padhani, Geert Villeirs, Katarzyna J. Macura, Clare M. Tempany, Peter L. Choyke, Francois Cornud, Daniel J. Margolis, Harriet C. Thoeny, Sadhna

Verma, Jelle Barentsz, and Jeffrey C. Weinreb, “Prostate Imaging Reporting and Data System Version 2.1: 2019 Update of Prostate Imaging Reporting and Data System Version 2,” European Urology, vol. 76, no. 3, pp. 340– 351, Sept. 2019.

[2] Anwar R. Padhani, Jelle Barentsz, Geert Villeirs, Andrew B. Rosenkrantz, Daniel J. Margolis, Baris Turkbey, Harriet C. Thoeny, Franc¸ois Cornud, Masoom A. Haider, Katarzyna J. Macura, Clare M. Tempany, Sadhna Verma, and Jeffrey C. Weinreb, “PI-RADS Steering Committee: The PI-RADS Multiparametric MRI and MRI-directed Biopsy Pathway,” Radiology, vol. 292, no. 2, pp. 464–474, Aug. 2019.

[3] Thomas Sanford, Stephanie A. Harmon, Evrim B. Turkbey, Deepak Kesani, Sena Tuncer, Manuel Madariaga, Chris Yang, Jonathan Sackett, Sherif Mehralivand, Pingkun Yan, Sheng Xu, Bradford J. Wood, Maria J. Merino, Peter A. Pinto, Peter L. Choyke, and Baris Turkbey, “Deep-learning-based artificial intelligence for PI-RADS classification to assist multiparametric prostate MRI interpretation: A development study,” Journal of Magnetic Resonance Imaging, vol. 52, no. 5, pp. 1499–1507, Nov. 2020.

[4] Sherif Mehralivand, Dong Yang, Stephanie A. Harmon, Daguang Xu, Ziyue Xu, Holger Roth, Samira Masoudi, Thomas H. Sanford, Deepak Kesani, Nathan S. Lay, Maria J. Merino, Bradford J. Wood, Peter A. Pinto, Peter L. Choyke, and Baris Turkbey, “A Cascaded Deep Learning–Based Artificial Intelligence Algorithm for Automated Lesion Detection and Classification on Biparametric Prostate Magnetic Resonance Imaging,” Academic Radiology, vol. 29, no. 8, pp. 1159–1168, Aug. 2022.

[5] Yipei Wang, Yinsong Xu, Weixi Yi, Shaheer Ullah Saeed, Natasha Thorley, Alexander Ng, Yukun Zhou, Wen Yan, Dean Barratt, Shonit Punwani, Veeru Kasivisvanathan, Mark Emberton, Daniel C. Alexander, and Yipeng Hu, “ProFound: A moderate-sized vision foundation model for multi-task prostate imaging,” 2026.

[6] Andrew B. Rosenkrantz, Luke A. Ginocchio, Daniel Cornfeld, Adam T. Froemming, Rajan T. Gupta, Baris Turkbey, Antonio C. Westphalen, James S. Babb, and Daniel J. Margolis, “Interobserver Reproducibility of the PI-RADS Version 2 Lexicon: A Multicenter Study of Six Experienced Prostate Radiologists,” Radiology, vol. 280, no. 3, pp. 793–804, Sept. 2016.

[7] Eduardo Pooch, Georgios Agrotis, Ivo G. Schoots, Mark Emberton, Veeru Kasivisvanathan, and Yipeng Hu, “Critical appraisal of artificial intelligence studies for prostate cancer detection on MRI,” European Radiology, Oct. 2026.

[8] Shyam Natarajan, Alan Priester, Daniel Margolis, Jiaoti Huang, and Leonard Marks, “Prostate MRI and Ultrasound With Pathology and Coordinates of Tracked Biopsy (Prostate-MRI-US-Biopsy),” 2020.

[9] Haoxin Zheng, Alex Ling Yu Hung, Qi Miao, Weinan Song, Fabien Scalzo, Steven S. Raman, Kai Zhao, and Kyunghyun Sung, “AtPCa-Net: anatomical-aware prostate cancer detection network on multi-parametric MRI,” Scientific Reports, vol. 14, no. 1, pp. 5740, Mar. 2024.

[10] Ahmet Karagoz, Deniz Alis, Mustafa Ege Seker, Gokberk Zeybel, Mert Yergin, Ilkay Oksuz, and Ercan Karaarslan, “Anatomically guided self-adapting deep neural network for clinically significant prostate cancer detection on bi-parametric MRI: a multi-center study,” Insights into Imaging, vol. 14, no. 1, pp. 110, June 2023.

[11] Matin Hosseinzadeh, Anindo Saha, Patrick Brand, Ivan Roan Slootweg, Maarten De Rooij, and Henkjan Huisman, “Deep learning–assisted prostate cancer detection on bi-parametric MRI: minimum training data size requirements and effect of prior knowledge,” European Radiology, vol. 32, no. 4, pp. 2224–2234, Apr. 2022.

[12] Abhejit Rajagopal, Antonio C. Westphalen, Nathan Velarde, Jeffry P. Simko, Hao Nguyen, Thomas A. Hope, Peder E. Z. Larson, and Kirti Magudia, “Mixed Supervision of Histopathology Improves Prostate Cancer Classification From MRI,” IEEE Transactions on Medical Imaging, vol. 43, no. 7, pp. 2610–2622, July 2024.

[13] William Holmlund, Attila Simko, Karin S´ oderkvist,¨ Peter Pal´ asti, Szilvia T´ otin, Kamilla Kalm´ ar, Zs´ ofia´ Domoki, Zsuzsanna Fejes, Zsigmond Tamas Kincses,´ Patrik Brynolfsson, and Tufve Nyholm, “ProstateZones – Segmentations of the prostatic zones and urethra for the PROSTATEx dataset,” Scientific Data, vol. 11, no. 1, pp. 1097, Oct. 2024.

[14] Alexey Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[16] Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka, “Rank consistent ordinal regression for neural networks with application to age estimation,” Pattern Recognition Letters, vol. 140, pp. 325–331, 2020.