# International Multi-Centre Validation of Unsupervised Domain Adaptation for Precise Discrimination between Normal and Abnormal Retinal Development

## Author Information

Zhanhan Tu MBChB, MM, PhD,<sup>1</sup> Nikhil Reddy MSc,<sup>2</sup> Helen J Kuht PhD,<sup>1</sup> Ha-Jun Yoon PhD, <sup>3</sup> Gail

DE Maconachie PhD ,<sup>4</sup> Ravi Purohit BMedSci, PhD,<sup>1</sup> Prateek Pani MSc,<sup>2</sup> Line Kessel, MD, PhD,

<sup>5,6</sup>Jinu Han MD, <sup>7</sup> Jing Jin MD, <sup>8</sup> Shui-Hua Wang PhD,<sup>9</sup> Karthik Bharadwaj MSc, <sup>11</sup> Viral Sheth PhD,<sup>1</sup>

Michael Hisaund BMedSci,<sup>1</sup> Rebecca J McLean MSc, PhD,<sup>1</sup> George Garratt MBChB,<sup>1</sup> Chetan

Kantibhai Patel FRCOphth,<sup>12</sup> Kanmin Xue FRCOphth, PhD,<sup>13</sup> Callum Hunt MSc,<sup>1</sup> Zisen Zhou PhD,<sup>1</sup>

Ian De Silva FRCOphth, <sup>14</sup> Seema Teli BMedSci,<sup>1</sup> Sonal Shah BSc, MCOptom,<sup>1</sup> Vanessa Rodwell

MBChB, MSc,<sup>1</sup> Aarti Patel MBChB, FRCOphth,<sup>1</sup> Irene Gottlob MD, FRCOphth,<sup>1</sup> Yu-Dong Zhang

PhD,<sup>9,10</sup> Zedong Zheng PhD,<sup>9</sup> Girish Varma PhD,<sup>2\*</sup> Mervyn G Thomas PhD, FRCOphth<sup>1\*</sup>

## Affiliations

<sup>1</sup>The University of Leicester Ulverscroft Eye Unit, School of Psychology and Vision Sciences,

University of Leicester, RKCSB, PO Box 65, Leicester, LE2 7LX, UK

<sup>2</sup>International Institute of Information Technology, Gachibowli, Hyderabad - 500 032, India

<sup>3</sup> inAmind Laboratory, Department of Neuroscience, Psychology and Behaviour, University of

Leicester, George Davies Centre, Lancaster Road, Leicester, LE1 7RH, UK.

<sup>4</sup> Academic Unit of Ophthalmology and Orthoptics, University of Sheffield, Sheffield, UK

<sup>5</sup> Department of Ophthalmology, Copenhagen University Hospital, Rigshospitalet-Glostrup,

Copenhagen, Denmark

<sup>6</sup>The Department of Clinical Medicine, University of Copenhagen, Denmark

<sup>7</sup> Institute of Vision Research, Department of Ophthalmology, Gangnam Severance Hospital, Yonsei

University College of Medicine, Seoul, Korea; Foveal Development Investigators Group.

<sup>8</sup>Sidney Kimmel Medical College of Thomas Jefferson University, Philadelphia, Pennsylvania;   
Nemours Children's Health, Wilmington, Delaware.

<sup>9</sup>School of Computing and Mathematical Sciences, University of Leicester, LE1 7RH, UK.

<sup>10</sup> School of Computer Science and Engineering, Southeast University, Nanjing, Jiangsu 211102, China

<sup>11</sup>Université Grenoble Alpes, CNRS, Grenoble INP, Institut Néel, 38000, Grenoble, France.

<sup>12</sup> Oxford Eye Hospital, Oxford University Hospitals NHS Foundation Trust, Oxford, UK; Ophthalmology Department, Great Ormond Street Hospital for Children NHS Foundation Trust, London, UK.

<sup>13</sup>Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, UK; Oxford University Hospitals NHS Foundation Trust, Oxford, UK; Great Ormond Street Hospital for Children NHS Foundation Trust, London, UK.

<sup>14</sup> Department of Ophthalmology, University Hospitals of Leicester, Infirmary Square, LE1 5WW, Leicester, UK.

\*Corresponding authors: Tel: +44 (0)116 252 5879, Fax: +44 (0)116 223 1996, email:

gv53@le.ac.uk and mt350@le.ac.uk

## Abstract

Background: Arrested retinal development during childhood is a major cause of childhood visual impairment and blindness. Diagnosis is achieved by paediatric ophthalmologists using optical coherence tomography (OCT) together with standarised grading of foveal development. OCT based grading informs diagnosis, targeted genetic testing and also prognostication. We aimed to develop an artificial intelligence (AI) based system to accurately grade foveal development, addressing the challenge of variability across different studies, centres, and the diverse OCT devices and acquisition protocols they employ. Specifically, we used Unsupervised Domain Adaptation (UDA), to create a

device and protocol-agnostic classification model distinguishing normal and abnormal retinal development. To enable discovery science approaches, we also tested this within the UK Biobank dataset.

Methods: This multicentre, retrospective cohort study used retinal OCTs across five centres in four countries (UK, Denmark, USA and South Korea) and the UK Biobank (UKB). This included diverse data obtained from four different OCT machines representing both table-mounted and handheld devices. The dataset was split into training (80%) and testing (20%) sets. We used UDA to generalise a classification model differentiating normal and abnormal retinal development in a device- and protocol-agnostic manner training a model on labelled images from a source device and unlabelled images from another target device, ensuring that the model adapts the target distribution to closely match the source distribution.

Foveal scans were acquired from four different manufacturers: (I) TM-OCT1, a table-mounted OCT (3037 scans); (II) HH-OCT1, a handheld OCT (2041 scans); (III) TM-OCT2, (2519 scans); and (IV) TM-OCT3 (242 scans). The foveal B-scans for (I) and (II) were extracted, annotated, and segmented from a dataset of >20,000 OCT scans, whereas TM-OCT2 scans (UK Biobank) were extracted from a database consisting of >85,771 volumetric images. Convolutional neural networks (CNN) backbone ResNet50 was assessed with each possible pair used as source (for supervised training) and target (for unsupervised training). The model’s diagnostic accuracy was compared against clinician graders (7 graders with a range of experience: 1 to 10 years) using performance metrics including (i) sensitivity, (ii) specificity, and (iii) accuracy.

Findings: The cross-domain binary classification model achieved diagnostic accuracy of 87.75– 96.06%, sensitivity of 88.68–98.38%, and specificity of 78.00–98.02% across validation sets. The Domain Adversarial Neural Network (DANN), trained on TM-OCT1 (labelled) and HH-OCT1 (unlabelled) images, demonstrated generalisability on the HH-OCT1 test set, with 96.06% accuracy, 98.38% sensitivity, and 89.11% specificity. The model achieved an Area Under Curve (AUC) of 0.994 (95% CI: 0.981-0.999), surpassing clinician-annotated benchmarks and exceeding state-of-the-art discriminative performance in cross-domain settings. Particularly, the DANN maintained robust

accuracy despite domain shifts between imaging systems (TM-OCT1 to HH-OCT1), demonstrating adaptability to heterogeneous clinical environments. Its high sensitivity across stages suggests clinical utility in early detection, where conventional methods underperform.

Interpretation: We have demonstrated the use of a device-agnostic AI-based system in adult and paediatric OCT interpretation for developmental retinal disorders without extra-labelled data. The diagnostic performance of the AI-based system is comparable to that of a senior clinician (\~10 years of experience). The proposed system has the potential to reduce inter-examiner variability and support clinical decision-making. Classifying normal and abnormal retinal development from OCT scans has both diagnostic and prognostic implications. We also apply this system within the UK Biobank dataset to demonstrate feasibility of deploying UDA at a large scale. This has potential for obtaining novel imaging-derived phenotypes from existing datasets for performing association studies and supporting discovery science approaches across domains. Our AI-based system provides the foundation for a realtime, point-of-use, diagnostic tool for retinal developmental disorders.

Funding: This study was supported by the Medical Research Council (MRC), London, UK (grant numbers MR/J004189/1, MRC/N004566/1 and MC\_PC\_17171); the Ulverscroft Foundation (supporting MGT and ZT); Fight for Sight (grant numbers 5009/5010 and 24NN181); the Medical Research Foundation and Wellcome Trust (grant numbers MRF-JF-EH-23-136 and 222530/Z/21/Z; supporting ZT); the National Institutes of Health (grant number DRF-2014-07-136; supporting AP); and the National Institute for Health and Care Research (grant number CL-2017-11-003; supporting MGT).

## Research in Context

Evidence Before This Study

We searched PubMed, Scopus, Science Direct, and IEEE Explore from database inception to February 18th, 2024, for studies in English that applied machine and deep learning on optical coherence

tomography (OCT) imaging for the identification of abnormal retinal development. We combined the search terms artificial intelligence (“AI” or “deep learning” or “machine learning”), OCT (“OCT” or “optical coherence tomography”), grading (“grading” or “classification” or “classify”), and foveal hypoplasia (“foveal hypoplasia” or “abnormal foveal development”). Studies in languages other than English that included an English abstract were also considered.

We found 1 study using machine learning from OCT imaging in foveal hypoplasia for visual acuity prediction. In our study, we set out to develop and validate a deep-learning-based approach for the automatic validation of foveal hypoplasia and its grading from OCT imaging. Prior to this study, the interpretation of OCT scans for grading arrested retinal development relied on manual assessment by clinicians, introducing subjectivity and variability. Moreover, using it across different datasets including large biobanks has not been previously performed. Existing evidence highlighted the need for a standardised, objective, and efficient method to assess retinal conditions in children. Previous attempts at using artificial intelligence (AI) in OCT interpretation often required extensive labelled datasets, limiting their applicability and scalability.

## Added Value of This Study

We developed a device-agnostic AI system using unsupervised domain adaptation (UDA) to grade arrested retinal development across heterogeneous OCT datasets, including the UK Biobank, without requiring device-specific labelled data. Our model achieved diagnostic accuracy comparable to senior clinicians (approximately 10 years experience) and demonstrated generalisability across OCT platforms (e.g., TM-OCT1, HH-OCT1). This system addresses domain shift, a key barrier to realworld AI deployment. It enables scalable, standardised OCT interpretation across diverse clinical and research settings.

## Implications of All Available Evidence

The proposed AI system achieves diagnostic accuracy comparable to experienced clinicians, performs consistently across different OCT devices, and gives the same output for the same scan, thereby avoiding the inter-examiner variability inherent in human interpretation. Together, these findings show that the system is accurate, generalisable and reproducible, which makes it a promising frontline screening tool for retinal developmental disorders. It can also be applied to large biobank datasets to support genome-wide and phenome-wide association studies. These results advance device-agnostic OCT interpretation and provide a foundation for integrating AI into routine clinical assessment. The next step is to establish how well these findings generalise across clinical practice, including different healthcare settings, patient populations and device types, and to evaluate the effectiveness of implementing the system within clinical care.

## Introduction

Arrested foveal development, or foveal hypoplasia, is a major cause of visual impairment and blindness in children. Retinal disease accounts for between 25% and 47% of severe paediatric visual impairment, and foveal hypoplasia is a common structural correlate across these conditions. Its importance has been underlined by international efforts such as the Foveal Development Investigators Group, which spans 21 centres across 12 countries <sup>1</sup>. Optical coherence tomography (OCT) is central to its assessment and allows the fovea to be imaged non-invasively, and the resulting scans inform diagnosis, direct targeted genetic testing and support prognostication <sup>1</sup>. Foveal development begins in mid-gestation and continues until around 13 years of age <sup>2-4</sup> . It proceeds through characteristic stages, including displacement of the inner retinal layers, migration of cone photoreceptors towards the developing fovea, and specialisation of the foveolar cones <sup>5-7</sup> . Disruption at any stage results in foveal hypoplasia, which is seen in albinism, prematurity, achromatopsia and in disorders arising from variants in PAX6, AHR and SLC38A8 <sup>8-15</sup> .

The Leicester grading system classifies foveal hypoplasia by the stage at which development is arrested <sup>6</sup> . The grade carries both diagnostic and prognostic value, and in preverbal infants it predicts future visual acuity <sup>16</sup>. Accurate grading depends on specialised expertise in paediatric OCT interpretation, which is scarce and unevenly distributed.

Deep learning offers a route to automating OCT interpretation and extending this expertise. Its use in this setting is limited by the diverse set of OCT instruments and protocols, which vary across centres and between table-mounted and handheld devices. This variation produces domain shift, in which the statistical distribution of images differs between the data a model is trained on and the data it later encounters. A model trained on one device therefore tends to perform poorly on another. Correcting this by retraining requires new labelled data for each device. For rare developmental disorders, where scans and expert annotation are both limited, this requirement is difficult to meet and constrains the use of automated grading in practice.

Automated grading that could be trusted across instruments, without relabelling for each new device, would remove this barrier. Unsupervised domain adaptation (UDA) offers such an approach. It learns from labelled images on a source device and unlabelled images on a target device, and aligns their feature distributions so that a classifier transfers between instruments without any labelled data from the target. Here we develop and validate a device-agnostic model that classifies normal against abnormal retinal development from OCT. We test whether UDA maintains performance across different manufacturers, both table-mounted and handheld, and across cohorts differing in age and origin. We then compare the model against expert clinician graders. Such a system would provide a screening tool for retinal developmental disorders and a means of deriving standardised structural phenotypes at scale for genetic and epidemiological discovery.

## Methods

## Study design

This was a retrospective, multicentre study of retinal OCT images. Data were contributed by five centres in four countries, comprising University Hospitals of Leicester NHS Trust (UK), Oxford University Hospitals NHS Foundation Trust (UK), Nemours Children's Health (USA), Copenhagen University Hospital, Rigshospitalet (Denmark) and Yonsei University College of Medicine (South Korea), together with the UK Biobank. All contributing sites are part of the Foveal Development Investigators Group <sup>1</sup> . Images were acquired between January 2009 and January 2023 as part of routine care or research. The study addressed device-agnostic binary classification of normal against abnormal retinal development. Binary classification was chosen because its anticipated value lies in screening and in deriving imaging phenotypes at scale for discovery science within large datasets. Reporting followed the TRIPOD statement <sup>17</sup>.

## OCT datasets and devices

Four datasets were used, spanning table-mounted and handheld instruments from four manufacturers (Table 1, Supplementary Table 1). TM-OCT1 was acquired on a Copernicus HR (table-mounted) at Leicester. HH-OCT1 was acquired on a Leica Envisu (handheld) at Leicester. TM-OCT2 was acquired on a Topcon 3D OCT 1000 Mk2 and drawn from the UK Biobank and Denmark. The fourth dataset was acquired on a Heidelberg Spectralis at three centres in the UK, USA and South Korea. TM-OCT1, HH-OCT1 and TM-OCT2 were used for model development. The Heidelberg dataset was reserved in its entirety as an external inference set and contributed to no stage of training or adaptation. Scan protocols for each device are given in Supplementary Table 1.

## Participants

We included patients with retinal developmental disorders and healthy controls with no evidence of abnormal retinal development. All diagnoses known to be associated with foveal hypoplasia were eligible. Given the rarity of these conditions, cases were not subcategorised by genotype, since retinal structure rather than genetic subgroup is the primary determinant of prognosis<sup>3,6</sup>. Longitudinal data were retained because retinal architecture changes over time in achromatopsia<sup>9,</sup> <sup>12,</sup> <sup>18</sup> and in other forms of foveal hypoplasia<sup>19</sup>. Scans were excluded where the fovea was not clearly visible due to poor quality. Participants below one year of age were excluded, because the outer retinal structures remain immature and foveal specialisation is incomplete at that age, precluding reliable grading <sup>2,</sup> <sup>4</sup>. Demographic characteristics and case mix for each dataset are summarised in Table 1.

## Image preprocessing and data partitioning

For the three development datasets, images were split 80% to 20% into training and test sets at the level of the individual. All longitudinal timepoints and all parafoveal scans from a given participant were confined to a single split, so that no individual contributed to both training and test data. Parafoveal and perifoveal scans can resemble different grades of foveal hypoplasia, so for TM-OCT1 and HH-OCT1 we increased diversity by including parafoveal scans that mimic arrested foveal development, an approach we refer to as parafoveal augmentation. TM-OCT2 and the Heidelberg set did not include parafoveal augmentation. Images were preprocessed using standard methods. During training on TM-OCT1, HH-OCT1 and TM-OCT2 we applied random cropping to 224 by 224 pixels, flipping on the vertical axis, resizing to 256 by 256 pixels and normalisation using ImageNet mean and standard deviation. The Heidelberg set was used in full for testing and received no augmentation. An overview of the study design is shown in Figure 1.

## Fig. 1: The Overview of Study Design

This figure summarises the design of a device-agnostic AI model trained to classify normal and abnormal retinal development using optical coherence tomography (OCT) scans. For example, labelled data from the TM-OCT1 system and unlabelled data from the HH-OCT1 system were integrated via unsupervised domain adaptation (UDA), enabling the model to harmonise differences between imaging devices. Performance was validated internally (TM-OCT1) and externally (HH-OCT1), with accuracy, sensitivity, and specificity compared to expert clinician assessments.

![](images/13069f2a0a3e75206bdd99028dbdebb74ad66ed4736f262454ee482d59f43e00.jpg)

Table 1 Demographic and clinical characteristics of participants stratified by diagnosis.

Data are presented as median (IQR) or n (%).
<table><tr><td rowspan=1 colspan=8>Control         Albinism                                                    Isolated FH                         Total (n=707)Characteristic                                                            IIN (n=42)      INS (n=47)                          Other (n=14)(n=477)          (n=108)                                                        (n=19)</td></tr><tr><td rowspan=1 colspan=1>Age, years, median (IQR)</td><td rowspan=1 colspan=1>31.0 (20.0-45.0)</td><td rowspan=1 colspan=1>24.0 (17.0-42.2)</td><td rowspan=1 colspan=1>20.0 (11.2-42.0)</td><td rowspan=1 colspan=1>20.0 (7.0-35.5)</td><td rowspan=1 colspan=1>41.0 (29.5-48.0)</td><td rowspan=1 colspan=1>34.5 (11.0-47.5)</td><td rowspan=1 colspan=1>30.0 (19.0-44.0)</td></tr><tr><td rowspan=1 colspan=1>Male, n (%)</td><td rowspan=1 colspan=1>211 (44.2)</td><td rowspan=1 colspan=1>64 (59.3)</td><td rowspan=1 colspan=1>22 (52.4)</td><td rowspan=1 colspan=1>28 (59.6)</td><td rowspan=1 colspan=1>10 (52.6)</td><td rowspan=1 colspan=1>6 (42.9)</td><td rowspan=1 colspan=1>341 (48.2)</td></tr><tr><td rowspan=1 colspan=1>Female, n (%)</td><td rowspan=1 colspan=1>258 (54.1)</td><td rowspan=1 colspan=1>44(40.7)</td><td rowspan=1 colspan=1>20 (47.6)</td><td rowspan=1 colspan=1>19 (40.4)</td><td rowspan=1 colspan=1>8 (42.1)</td><td rowspan=1 colspan=1>8 (57.1)</td><td rowspan=1 colspan=1>357 (50.5)</td></tr><tr><td rowspan=1 colspan=1>Unknown sex, n (%)</td><td rowspan=1 colspan=1>8 (1.7)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>1 (5.3)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>9 (1.3)</td></tr><tr><td rowspan=1 colspan=1>White, n (%)</td><td rowspan=1 colspan=1>361 (75.7)</td><td rowspan=1 colspan=1>93 (86.1)</td><td rowspan=1 colspan=1>38 (90.5)</td><td rowspan=1 colspan=1>41 (87.2)</td><td rowspan=1 colspan=1>17 (89.5)</td><td rowspan=1 colspan=1>14 (100.0)</td><td rowspan=1 colspan=1>565 (79.9)</td></tr><tr><td rowspan=1 colspan=1>Asian or Asian British, n(%)</td><td rowspan=1 colspan=1>106 (22.2)</td><td rowspan=1 colspan=1>11 (10.2)</td><td rowspan=1 colspan=1>1 (2.4)</td><td rowspan=1 colspan=1>4 (8.5)</td><td rowspan=1 colspan=1>1 (5.3)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>123 (17.4)</td></tr><tr><td rowspan=1 colspan=1>Black, African, Caribbeanor Black British, n (%)</td><td rowspan=1 colspan=1>2 (0.4)</td><td rowspan=1 colspan=1>4 (3.7)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>5 (0.7)</td></tr><tr><td rowspan=1 colspan=1>Other ethnicity, n (%)</td><td rowspan=1 colspan=1>3 (0.6)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>1 (7.1)</td><td rowspan=1 colspan=1>4 (0.6)</td></tr><tr><td rowspan=1 colspan=1>Undisclosed ethnicity, n(%)</td><td rowspan=1 colspan=1>5 (1.0)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>3 (7.1)</td><td rowspan=1 colspan=1>2 (4.3)</td><td rowspan=1 colspan=1>1 (5.3)</td><td rowspan=1 colspan=1>0 (0.0)</td><td rowspan=1 colspan=1>10 (1.4)</td></tr></table>

## Phenotype and grading

Foveal development was graded using the Leicester grading system for foveal hypoplasia (Figure 2) <sup>6</sup> . We used the clinical grade recorded at the point of care, with independent validation of each grade by expert investigators (HJK, ZT). Disagreements were resolved by a third senior grader. Where no grade was recorded in the case notes, a consensus grade was assigned by three investigators (GG, ZT, HJK). For binary classification, grades 1 to 4 and atypical foveal hypoplasia were combined into a single abnormal class, classified against normal foveal development.

## Fig. 2: Leicester grading system of foveal hypoplasia

(A) and (C) Stages of foveal development and corresponding grades of foveal hypoplasia (B) A diagrammatic representation of the foveal hypoplasia grading scheme,4 with the corresponding real OCT examples representing the respective foveal hypoplasia grades. Abbreviations: RNFL = retinal nerve fibre layer; GCL = ganglion cell layer; IPL = inner plexiform layer; INL = inner nuclear layer; OPL = outer plexiform layer; ONL = outer nuclear layer; ELM; external limiting membrane; IS = inner segment; OS = outer segment; RPE = retinal pigment epithelium. Adapted and reproduced with permission from Thomas et al.

(2011)  
![](images/a1d97fd7fa65d4b162b5bf32962e33818ffeae8461da47d6ba5b5964b93be98f.jpg)

(D)  
![](images/ad4c499d690bd4774653726c96578542310db02ed5e98f70cd7d299755e86a6a.jpg)  
(F)

(E)  
![](images/dfdc0c52a6f89039bc4dee1be2523ed4a971d25c7cff9537d815eb1e3b7fa7d1.jpg)  
The developmental process has not occurred

![](images/e648df2ddcdf392dfb0bfced4ba0fc3481bae89dccae210fdade21b9b6f55e06.jpg)

## Model development and domain adaptation

We first trained a baseline classifier without adaptation. This used a ResNet50 backbone trained on a labelled source device and applied directly to an unlabelled target device, to quantify the effect of device shift.

We then applied unsupervised domain adaptation using Domain Adversarial Neural Networks (DANN) <sup>20</sup>. A DANN learns from labelled source images and unlabelled target images, and aligns their feature distributions so that the learned features are informative for classification but uninformative about the device of origin. We developed models pairwise across TM-OCT1, HH-OCT1 and TM-OCT2, using each dataset in turn as source and another as target, giving six models in total.

The architecture combined a ResNet50 feature extractor with a classifier and a domain discriminator (Figure 3). Features were extracted by ResNet50 followed by global average pooling, yielding a tensor of size 2048. The classifier was a single linear layer mapping 2048 features to one output. The domain discriminator was a three-layer network with shapes 2048 to 1024, 1024 to 1024 and 1024 to one. A gradient reversal layer connected the feature extractor to the discriminator. During backpropagation it reversed the sign of the domain gradient, which drove the feature extractor toward device-invariant representations. Training minimised a combined loss ${ \mathrm { L } } = { \mathrm { L c } } - \lambda { \mathrm { L d } } ,$ where Lc is the classification loss computed on labelled source data, Ld is the domain classification loss computed on source and target data, and λ weights the two terms. Models were trained on the ALICE High Performance Computing facility. The three best-performing models were then applied to the Heidelberg inference set for external validation.

Fig. 3: Domain Adversarial Neural Network (DANN) Training Architecture

Domain Adversarial Neural Network (DANN) Training Architecture utilised for OCT scan classification. The feature extractor network (F) captures discriminative features from OCT images, which are fed into two distinct pathways: the classifier network (C) and the discriminator network (D). The classifier (C), shown in red, uses labelled source data to categorize scans as Abnormal or Normal based on the class loss (Ly). In parallel, the discriminator (D), indicated in yellow, determines the domain origin (source or target) of the scans using unlabelled target data, informed by the domain loss (Ld). The Gradient Reversal Layer (GRL) connects F and D, facilitating the minimisation of domain discrepancies by reversing the gradient sign during backpropagation. The red and blue arrows represent the supervised and unsupervised training phases, respectively.

![](images/efd9881fad342e55e2e5d0051f0164f95cccb2912b5affa4b740ff1afbbd0cb4.jpg)

## Comparison with clinician graders

We compared model performance against 7 expert graders with 1 to 10 years of grading experience, using the same test images on which model accuracy was determined. Graders were blinded to all other clinical information and to one another's assessments. Each grader classified every scan as normal or abnormal.

## Statistical analysis

The primary endpoint was classification accuracy across datasets. We assessed each model using sensitivity, specificity, area under the receiver operating characteristic curve (AUC) and F1 score, and visualised performance using ROC curves. Confidence intervals were obtained by bootstrap resampling with 1,000 replicates for each source-to-target pairing, computing the false-positive and true-positive rates across thresholds to derive 95% confidence intervals. For the clinician comparison we summarised each grader's accuracy, sensitivity and specificity, quantified agreement among graders using Fleiss kappa overall and by experience level, and compared the best model against individual graders using McNemar's test on paired classifications. Tests were two-sided and considered significant at a threshold of 0.05. Analyses were performed in Python version 3.9.

## Results

## Cohort and study design

We assembled a retrospective, multicentre OCT dataset from five centres across four countries (UK, Denmark, USA and South Korea) and the UK Biobank, imaged between January 2009 and January 2023 on four instruments representing both table-mounted and handheld devices (Table 1). Development used three datasets, TM-OCT1 (Copernicus HR, table-mounted; Leicester, UK), HH-OCT1 (Leica Envisu, handheld; Leicester, UK) and TM-OCT2 (Topcon 3D OCT 1000 Mk2; UK Biobank and Denmark). A fourth dataset, acquired on a Heidelberg Spectralis at three centres (UK, USA and South Korea), was held out in its entirety as an external inference set. For the three development datasets, data were split 80% to 20% for training and testing at the level of the individual, with every longitudinal timepoint and every parafoveal B-scan from a participant confined to a single split.

The cohorts differed by design in age, device and case mix, providing a deliberately heterogeneous testbed for device-agnostic classification (Table 1). Foveal abnormality was graded following the Leicester system6, with grades 1 to 4 and atypical foveal hypoplasia combined into a single abnormal class and classified against normal foveal development.TM-OCT1 comprised 620 individuals (mean age 33.7 years, 63.0% abnormal) and HH-OCT1 a younger paediatric cohort of 174 individuals (mean age 12.2 years, 58.6% abnormal), both from Leicester and both including longitudinal data and parafoveal augmentation. TM-OCT2 drew on an older population (mean age 55.6 years, 71.4% abnormal), and the Heidelberg inference set comprised 242 scans from an ethnically diverse population (77.3% abnormal). Neither TM-OCT2 nor the Heidelberg set included longitudinal data or parafoveal augmentation.

## Unsupervised domain adaptation overcomes device shift

Models were built and compared across the three development datasets, TM-OCT1, HH-OCT1 and TM-OCT2. Each was used in turn as the labelled source and another as the target, giving six sourceto-target pairings. We first established a baseline with a ResNet50 classifier trained on one device and applied directly to another, without adaptation. This baseline performed poorly. It over-predicted the abnormal class, with a false-positive rate of 0.95 and an F1 score of 0.86 on the target device. A classifier accurate on its own instrument therefore became unreliable once the acquisition hardware changed. To address this, we applied unsupervised domain adaptation (UDA). UDA aligns the feature distributions of a labelled source device and an unlabelled target device through a domain-adversarial objective. This allows a classifier to transfer between instruments without any labelled data from the target and without additional annotation.

Adaptation closed much of the gap. The false-positive rate fell from 0.95 to 0.11 and the F1 score rose from 0.86 to 0.97. The strongest pairing was a DANN trained on labelled TM-OCT1 and unlabelled HH-OCT1 images. On the held-out HH-OCT1 test set it reached 96.06% accuracy, 98.38% sensitivity and 89.11% specificity, having seen no HH-OCT1 label during training. Across all three pairings, accuracy ranged from 87.75% to 96.06%, sensitivity from 88.68% to 98.38% and specificity from 78.00% to 98.02% (Table 2, Supplementary Figure 1, Supplementary Table 2). Performance varied with the source device. Table-mounted TM-OCT1 generalised well, reaching 96.06% on HH-OCT1 and 92.64% on TM-OCT2. The handheld HH-OCT1 as source was weaker, at 87.74% and 88.86%, consistent with noisier handheld images providing a less transferable source distribution. The choice of source also shaped the error profile. One model adapted to TM-OCT2 showed a lower falsepositive rate and a higher false-negative rate than the other (Figure 4, other model performance in Supplementary Figure 2). This suggests the approach can be tuned toward sensitivity or specificity according to clinical need. Domain adaptation therefore allowed a model that had failed on unseen hardware to perform consistently across instruments without further labelling.

Table 2: Classification metrics for Unsupervised Domain Adaptation on source-target pairs TPR: True Positive rate, FPR: False Positive Rate; $\mathrm { S e n s i t i v i t y = T P / ( T P + F N ) } ;$ Specificity =; $\mathrm { T N } / ( \mathrm { F P + T N } )$ ; Precision = TP/(TP+FP); False Positive ${ \mathrm { R a t e } } = { \mathrm { F P } } / ( { \mathrm { F P } } + { \mathrm { T N } } ) ;$ False Discovery Rate $\mathrm { ( F D R ) } = \mathrm { F P } / ( \mathrm { F P } + \mathrm { T P } )$ ; False Negative Rate $\mathrm { ( F N R ) } = \mathrm { F N } / ( \mathrm { F N + T P } )$ ; F $\mathrm { s c o r e } = 2 ^ { \ast } \mathrm { T P } / ( 2 ^ { \ast } \mathrm { T P } \mathrm { + F P } \mathrm { + F N } )$
<table><tr><td rowspan="2">DANN</td><td rowspan="2">SOURCE TO TARGET</td><td rowspan="2">Accuracy</td><td rowspan="2">Sensitivity TP/(TP+FN)</td><td rowspan="2">Specificity TN/(FP+TN)</td><td rowspan="2">Precision TP/(TP+FP)</td><td rowspan="2">False positive Rate (FPR) FP/(FP+TN)</td><td rowspan="2">False Discovery Rate (FDR) FP/(FP+TP)</td><td rowspan="2">False Negative Rate (FNR) FN/(FN+TP)</td><td rowspan="2">F1 score 2*TP/(2*TP+FP+FN)</td></tr><tr><td></td></tr><tr><td>DANN1</td><td>TM-OCT1 - HH-OCT1</td><td>0.95</td><td>0.96</td><td>0.93</td><td>0.97</td><td>0.07</td><td>0.03</td><td>0.04</td><td>0.97</td></tr><tr><td>DANN2</td><td>TM-OCT1 - TM-OCT2</td><td>0.92</td><td>0.92</td><td>0.93</td><td>0.97</td><td>0.08</td><td>0.03</td><td>0.08</td><td>0.94</td></tr><tr><td>DANN3</td><td>HH-OCT1 - TM-OCT1</td><td>0.87</td><td>0.89</td><td>0.85</td><td>0.91</td><td>0.15</td><td>0.09</td><td>0.12</td><td>0.89</td></tr><tr><td>DANN4</td><td>HH-OCT1 - TM-OCT2</td><td>0.88</td><td>0.92</td><td>0.78</td><td>0.91</td><td>0.22</td><td>0.09</td><td>0.08</td><td>0.92</td></tr><tr><td>DANN5</td><td>TM-OCT2 – TM-OCT1</td><td>0.93</td><td>0.95</td><td>0.89</td><td>0.94</td><td>0.11</td><td>0.06</td><td>0.05</td><td>0.94</td></tr><tr><td>DANN6</td><td>TM-OCT2 – HH-OCT1</td><td>0.97</td><td>0.97</td><td>0.94</td><td>0.98</td><td>0.06</td><td>0.02</td><td>0.03</td><td>0.98</td></tr></table>

## Fig. 4: DANN2 vs DANN4

The figure illustrates the performance of Domain-Adversarial Neural Network (DANN) models (DANN2 and DANN4) in reducing diagnostic errors. Both models were test data set collected from TM-OCT2. DANN2 demonstrates statistically significant lower false positive (FP) and higher false negative (FN) rates compared to DANN4, as indicated by the reduced height of corresponding bars. Error bars represent variability across multiple validation trials.

![](images/d315f44f1f52bcd3f8fb37d461998b55484c6d9a0577221daf293120a65c65ab.jpg)

![](images/1085b6ccc32b4eeaee8e2c9d28ee36cebdfa1f9abd3d5b9ddc2ab00385fe4d91.jpg)

![](images/49cf94e74cfcef519dcaebef3aaef68e969505544a4aa935fe1f77896cce95bb.jpg)

![](images/ea10255a43be510dd4382442f0ddb6d371d263ddb516a321d23f86c185d3e4a6.jpg)  
Generalisation to an unseen device and new populations  
Having shown that adaptation worked on each target device, we next asked whether it transferred to a device the models had never seen. For this we used a held-out inference set that played no part in training or adaptation. It comprised 242 Heidelberg Spectralis scans (187 abnormal, 55 normal) from a manufacturer not used in development, contributed by three centres in the UK, USA and South Korea. The best model for each target device (DANN1, DANN2 and DANN5) performed strongly on its own target test set, with F1 scores of 0.97, 0.94 and 0.94 and areas under the curve of 0.994 (95%

CI 0.981 to 0.999), 0.973 and 0.973 (Figure 5A). We then applied all six models to the Heidelberg set (Figure 5B, Supplementary Table 3). DANN1 to DANN4 maintained discrimination on this unseen device and in new patient populations, with AUCs between range 0.954 to 0.976. DANN5 and DANN6 performed worse, with AUCs of 0.871 and 0.793.

Fig. 5: Comparative ROC Curves for Domain Adaptation between various Imaging Devices The ROC labelled (A1) TM-OCT1(source-domain) to HH-OCT1(target-domain) was the best performing model when adapting to HH-OCT1. Similarly, (A2) and (A3) were best models adapting to TM-OCT2 and TM-OCT1(as targets). Fig (B), plot shows ROC curve for the 6 models.

![](images/fb7e78086181d85c2a34decc356ce1ce22fb8e99c95f30f5f5c2ede60727c7aa.jpg)

![](images/d7057150d7e12e3a5242581e6192ef20f391d2fa25f5df084151b5d441a2ce7c.jpg)

![](images/2e2ace8b4f2b00918a1a0a917fe1784f091e06319b341b40a5477e275f97e946.jpg)

B  
![](images/63bc8648a98b595986394ba72c778aa04a011e09f0568a8d0561a2abfdf4c9f0.jpg)  
Comparison with expert clinician graders

Having shown that the model generalised across devices, we next compared its performance with that of seven expert clinician graders to determine whether it reached the standard of routine clinical grading. Accuracy, sensitivity and specificity were calculated for each grader on the same test set,

alongside their years of clinical experience. Inter-grader agreement was then assessed both overall and according to experience level to quantify variability between clinicians. The model was subsequently compared with individual graders and with the senior and junior grader groups, with paired statistical testing used to evaluate differences in performance. Overall, the model performed at the level of the most experienced graders and showed more consistent classification than was observed across clinicians with differing levels of experience (Figure 6, Supplementary Figure 3).

Fig. 6: Performance of the AI-system demonstrated with receiver operating characteristic (ROC) curve for abnormal versus normal classification

ROC of the best model showing the confidence interval, along with the annotations of clinicians overlayed on top of the confidence interval band.

ROC: Receiver Operating Characteristics, FPR: False Positive rate, TPR: True Positive Rate

ROC Curve for TM-OCT1 to HH-OCT1  
![](images/85cc28b9331e10af9d476180af853923580dbb490cb5e1e98f804eb2db628f74.jpg)

## Discussion

We developed a device-agnostic model that classifies normal against abnormal retinal development from OCT, using unsupervised domain adaptation to transfer between instruments without any

labelled data from the target device. The model performed at the level of experienced clinician graders, and this performance held on an entirely unseen manufacturer and in new patient populations. A single adapted model therefore classified arrested retinal development reliably across tablemounted and handheld devices and across cohorts differing in age and origin.

A conventional classifier trained on one device failed once the acquisition hardware changed, and correcting this by retraining would require new labelled data for every new device. That requirement is difficult to meet for rare developmental disorders, where both scans and expert annotation are limited. By aligning the feature distributions of the source and target devices, UDA removed this dependence and allowed one model to generalise without further labelling. The result is a system that is accurate, generalisable and reproducible together, rather than accurate on a single instrument alone. This combination is what makes deployment across centres, and extraction of phenotypes from existing datasets, practical.

This property also clarifies what the model standardises. Clinicians apply consistent grading criteria, and the variability the model removes does not arise from disagreement over those criteria. It arises from differences in imaging hardware and acquisition protocols across centres, which alter the appearance of the same underlying anatomy. Our own work has shown that grading agreement varies with experience and imaging conditions <sup>21</sup>, and it is this hardware-driven and experience-driven variability that a device-agnostic model absorbs. The model returns a standardised structural readout that is stable across the instruments on which foveal development is imaged.

Prior work in this specific area is limited. Machine learning applied to foveal hypoplasia from OCT has been reported only for visual acuity prediction, and a recent deep learning study graded foveal hypoplasia from colour fundus images with lower binary accuracy and no cross-device design <sup>22</sup>. Broader efforts in medical image domain adaptation have aligned feature distributions across scanners and protocols in other modalities <sup>23-29</sup>, but none has addressed device-agnostic classification of retinal developmental disorders, nor validated transfer between table-mounted and handheld OCT. Our study extends domain adaptation to this setting and tests it on a held-out manufacturer, which is a stronger form of external validation than within-dataset testing.

A device-agnostic classifier could also serve as a screening tool where paediatric OCT expertise is scarce, flagging abnormal retinal development for expert review and supporting more equitable access to diagnosis. Because it transfers to handheld devices, it is applicable to the imaging of infants, in whom handheld OCT is often the only feasible option. Beyond individual diagnosis, the same model can generate standardised, device-independent structural phenotypes at scale. Applied across large imaging datasets, this offers a route to imaging-derived phenotypes for genome-wide and phenomewide association studies, and provides a way to study retinal development in populations that are otherwise limited by inconsistent grading across instruments.

We acknowledge several limitations of this study. First, the model classifies individual 2D B-scans rather than volumetric OCT, so the spatial context of the full volume is not used, and extending domain adaptation to 3D volumes remains an open and non-trivial problem 22. Secondly, the model does not grade severity, and it does not distinguish a flat central macula caused by arrested foveal development from one caused by loss of the inner retina, for example following prenatal haemorrhage, nor does it identify the underlying gene or aetiology. These are boundaries of scope, but they define what the current system can and cannot support clinically. Thirdly, the reference standard was expert grading, so the model was validated against human labels rather than an independent biological measure. Fourthly, case mix and age differed substantially across cohorts, and the proportion of abnormal scans ranged from 58 to 77 percent, so accuracy should be interpreted alongside sensitivity and specificity, and residual confounding by age or sex cannot be excluded. Fifthly, the design was retrospective, ungradable scans were excluded, and there was no prospective or point-of-care evaluation, so performance on unselected, lower-quality scans acquired in routine practice remains to be established. Sixthly, handheld images were a weaker source domain, indicating that transfer is sensitive to the quality of the source data. Finally, the prognostic value of grading is bounded. The grade predicts future visual acuity in preverbal infants 16, but it does not indicate whether a condition will remain stationary or progress, as can occur with central atrophy in achromatopsia, so the prognostic use of a device-agnostic classifier should be framed as supporting acuity-related prognostication rather than predicting disease trajectory.

In summary, unsupervised domain adaptation enabled accurate, generalisable and reproducible classification of arrested retinal development across OCT devices, without any additional labelling. In doing so it offers both a practical screening tool for retinal developmental disorders and a foundation for standardised retinal phenotyping at scale, linking the clinic to population-level genetic and epidemiological discovery.

## Conclusion

Our study addresses a pivotal challenge in medical AI: enabling robust, device-agnostic diagnostic systems without the need for costly relabelling across heterogeneous imaging platforms. By pioneering the application of UDA to OCT, we developed an AI framework capable of accurately grading arrested retinal development across manufacturers, achieving diagnostic performance comparable to that of a senior clinician with over a decade of experience. Using a ResNet50 backbone, the model achieved cross-domain binary classification accuracies of 87.75–96.06%, with sensitivities of 88.68–98.38% and specificities of 78.00–98.02%. These results underscore UDA’s potential to mitigate inter-device variability while maintaining clinical-grade reliability.

Significantly, our work represents the first preliminary validation for a domain-adaptive AI system in OCT interpretation, offering a scalable solution to avoid inter-examiner discrepancies and streamline clinical workflows. The ability of the framework to generalise across imaging devices without additional labelled data is particularly important as OCT becomes increasingly integrated into routine clinical practice and may facilitate the deployment of AI-assisted diagnosis across diverse imaging environments.

The implications extend beyond ophthalmology. By demonstrating capacity of UDA to reduce diagnostic errors (e.g., false positives) and enhance model generalisability, our findings advocate for broader adoption of domain adaptation techniques in medical imaging. This approach holds particular promise for resource-limited settings, where device heterogeneity and sparse expert annotations hinder equitable access to precision diagnostics. Future efforts should focus on validating these results in multi-centre trials and integrating 3D/4D imaging pipelines to further refine spatial-temporal

feature extraction.

## References

1. Kuht, H. J. et al. Genotypic and Phenotypic Spectrum of Foveal Hypoplasia: A Multicenter Study. Ophthalmology 129, 708–718 (2022).

2. Lee, H. et al. In Vivo Foveal Development Using Optical Coherence Tomography. Invest. Ophthalmol. Vis. Sci. 56, 4537–4545 (2015).

3. Thomas, M. G., Papageorgiou, E., Kuht, H. J. & Gottlob, I. Normal and abnormal foveal development. Br. J. Ophthalmol. 106, 593–599 (2022).

4. Vajzovic, L. et al. Maturation of the Human Fovea: Correlation of Spectral-Domain Optical Coherence Tomography Findings With Histology. American Journal of Ophthalmology 154, 779 (2013).

5. Hendrickson, A. E. & Yuodelis, C. The Morphological Development of the Human Fovea. Ophthalmology 91, 603–612 (1984).

6. Thomas, M. G. et al. Structural grading of foveal hypoplasia using spectral-domain optical coherence tomography a predictor of visual acuity? Ophthalmology 118, 1653–1660 (2011).

7. Yuodelis, C. & Hendrickson, A. A qualitative and quantitative analysis of the human fovea during development. Vision Res. 26, 847–855 (1986).

8. Mcallister, J. T., Dubis, A. M. & Tait, D. M. Arrested development: high-resolution imaging of foveal morphology in albinism. Vision Res 50, 810–817 (2010).

9. Thomas, M. G., Kumar, A., Kohl, S., Proudlock, F. A. & Gottlob, I. High-resolution in vivo imaging in achromatopsia. Ophthalmology 118, 882–887 (2011).

10. Hammer, D. X., Iftimia, N. V. & Ferguson, R. D. Foveal fine structure in retinopathy of prematurity: an adaptive optics Fourier domain optical coherence tomography study. Invest

Ophthalmol Vis Sci 49, 2061–2070 (2008).

11. Poulter, J. A., Al-Araimi, M. & Conte, I. Recessive mutations in SLC38A8 cause foveal hypoplasia and optic nerve misrouting without albinism. Am J Hum Genet 93, 1143–1150 (2013).

12. Thomas, M. G., Mclean, R. J., Kohl, S., Sheth, V. & Gottlob, I. Early signs of longitudinal progressive cone photoreceptor degeneration in achromatopsia. Br J Ophthalmol 96, 1232–1236 (2012).

13. Hingorani, M., Williamson, K. A., Moore, A. T. & Van Heyningen, V. Detailed ophthalmologic evaluation of 43 individuals with PAX6 mutations. Invest Ophthalmol Vis Sci 50, 2581–2590 (2009).

14. Mayer, A. K., Mahajnah, M. & Thomas, M. G. Homozygous stop mutation in AHR causes autosomal recessive foveal hypoplasia and infantile nystagmus. Brain 142, 1528–1534 (2019).

15. Kuht, H. J., Han, J. & Maconachie, G. SLC38A8 mutations result in arrested retinal development with loss of cone photoreceptor specialization. Hum Mol Genet 29, 2989–3002 (2020).

16. Rufai, S. R. et al. Can Structural Grading of Foveal Hypoplasia Predict Future Vision in Infantile Nystagmus?: A Longitudinal Study. Ophthalmology 127, 492–500 (2020).

17. Collins, G. S., Reitsma, J. B., Altman, D. G. & Moons, K. G. M. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. BMJ 350, g7594 (2025).

18. Triantafylla, M., Papageorgiou, E. & Thomas, M. G. Longitudinal Evaluation of Changes in Retinal Architecture Using Optical Coherence Tomography in Achromatopsia. Invest Ophthalmol Vis Sci 63 (2022).

19. Lee, H., Purohit, R. & Sheth, V. Retinal Development in Infants and Young Children With Albinism: Evidence for Plasticity in Early Childhood. Am J Ophthalmol 245, 202–211 (2023).

20. Ganin, Y. et al. Domain-adversarial training of neural networks. Journal of machine learning research 17, 1–35 (2016).

21. Shenoy, R. et al. Foveal hypoplasia grading with optical coherence tomography: agreement and challenges across experience levels. Diagnostics 15, 763 (2025).

22. Shen, Y., Sheng, B. & Fang, R. Domain-invariant interpretable fundus image quality assessment. Med Image Anal 61 (2020).

23. Stacke, K., Eilertsen, G., Unger, J. & Lundstrom, C. Measuring Domain Shift for Deep Learning in Histopathology. IEEE J Biomed Health Inform 25, 325–336 (2021).

24. Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. & Herrera, F. A unifying view on dataset shift in classification. Pattern Recognit 45, 521–530 (2012).

25. Guan, H. & Liu, M. Domain Adaptation for Medical Image Analysis: A Survey. IEEE Trans Biomed Eng 69, 1173–1185 (2022).

26. Cheplygina, V., De Bruijne, M. & Pluim, J. Not-so-supervised: A survey of semi-supervised, multi-instance, and transfer learning in medical image analysis. Med Image Anal 54, 280–296 (2019).

27. Ghafoorian, M., Mehrtash, A. & Kapur, T. Transfer Learning for Domain Adaptation in MRI: Application in Brain Lesion Segmentation. (2017).

28. Raghu, M., Zhang, C., Kleinberg, J., Bengio, S. & Brain, G. Transfusion: Understanding Transfer Learning for Medical Imaging. (2017).

29. Kamnitsas, K., Baumgartner, C. & Ledig, C. Unsupervised domain adaptation in brain lesion segmentation with adversarial networks. (2016).

## Acknowledgements

We thank Basu Dawar, Swati Parida, Helena Lee, Teo Bica and Abdullah Aamir for their input with this study. Our sincerest thanks to John Kelly and Leica Microsystems who provided software for navigating and extracting OCT scans. This research used the ALICE High Performance Computing Facility at the University of Leicester. KX would like acknowledge funding from the Wellcome Trust (216593/Z/19/Z), NIHR Oxford Biomedical Research Council (NIHR203311) and Medical Research Council (MR/Z504725/1).

## Ethics approval

The study received ethical approval from the East Midlands, Leicester Central Research Ethics Committee (study title, Application of Artificial Intelligence in Ophthalmology; REC reference 20/EM/0040; protocol 0733) and Health Research Authority approval (IRAS ID 261121). UK Biobank recruitment and data acquisition were approved by the North-West Research Ethics Committee (reference 06/MRE08/75), and this project was approved by the UK Biobank (application ID 85881). The study followed the Declaration of Helsinki.

## Conflict of interest statement

All authors report no competing interests. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript, or in the decision to publish the results.

## Supplementary Table 1 OCT acquisition protocols

Acronyms: Table-mounted OCT1 (TM-OCT1)= Copernicus OCT; Handheld OCT (HH-OCT1) = Leica Envisu; Table-mounted OCT2 (TM-OCT2) = Topcon OCT; Table-mounted OCT3 (TM-OCT3) = Heidelberg OCT.

<table><tr><td rowspan=1 colspan=1>OCT</td><td rowspan=1 colspan=1>A-scans</td><td rowspan=1 colspan=3>ScanB-scans                                                    NotesDimensions</td></tr><tr><td rowspan=3 colspan=1>TM-OCT1</td><td rowspan=1 colspan=1>743</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=1>7mm x 7mm</td><td rowspan=1 colspan=1>3 trails per eye</td></tr><tr><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>10mm x 10mm</td><td rowspan=1 colspan=1>2 protocols</td></tr><tr><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>10mm x 5mm</td><td rowspan=1 colspan=1>2 protocols</td></tr><tr><td rowspan=1 colspan=1>HH-OCT1</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>12mm x 8mm</td><td rowspan=1 colspan=1>at least 2 trails per eye</td></tr><tr><td rowspan=1 colspan=1>TM-OCT2</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>6mm x 6mm(centre of fovea)</td><td rowspan=1 colspan=1>Topcon 3D OCT 1000 Mk2, same protocols</td></tr><tr><td rowspan=1 colspan=1>TM-OCT3</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>30-degree scanangle</td><td rowspan=1 colspan=1>High resolution mode, combination of multiplesingle and raster scans centred on the fovea(mixture)</td></tr></table>

## Supplementary Figure 1 ROC Curves for Pairwise Comparisons of DANN Models

ROC curves comparing pairs of models adapted to the same target device from different source devices. (A) Target HH-OCT1: DANN1 (source TM-OCT1) versus DANN6 (source TM-OCT2). (B) Target TM-OCT2: DANN2 (source TM-OCT1) versus DANN4 (source HH-OCT1). (C) Target TM-OCT1: DANN3 (source HH-OCT1) versus DANN5 (source TM-OCT2).

A  
![](images/1778519d14349c695aa4605075302e557074d867b6e34c57450f49d49b22afa9.jpg)

B  
![](images/4a6845a1c9488a91499d614c5c6b0b467b3fb033b7e52d232a4f36754fa2d6cd.jpg)

C  
![](images/61bcdbaa70888e3c146826054475ff43b44969f02344eed21d034f3d089569b3.jpg)

![](images/29bf2206bed171af2faa729e74cc2a89cde43a6ee9be626bc1a02ea96f458311.jpg)

Supplementary Table 2 Statistical Results of DANN Pairwise Comparisons
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>P-value</td><td rowspan=1 colspan=1>Z-value</td><td rowspan=1 colspan=1>95% Confidence Interval</td></tr><tr><td rowspan=1 colspan=1>DANN 1vs 6</td><td rowspan=1 colspan=1>0.2764</td><td rowspan=1 colspan=1>1.088</td><td rowspan=1 colspan=1>-0.00589 to 0.0206</td></tr><tr><td rowspan=1 colspan=1>DANN 2vs 4</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>3.930</td><td rowspan=1 colspan=1>0.0227 to 0.0678</td></tr><tr><td rowspan=1 colspan=1>DANN 3vs 5</td><td rowspan=1 colspan=1>0.0008</td><td rowspan=1 colspan=1>3.368</td><td rowspan=1 colspan=1>0.0108 to 0.0409</td></tr></table>

Supplementary Figure 2 Classification outcomes of DANN1 to DANN6 on their target test sets

## DANN1

![](images/9143250fce17edf7f46b2c82673f0478fc5fb082b3862ace0987975f93a25200.jpg)

![](images/461ef9fe759ee35b65b5896f9cd6b84f7299ce13aa2de5e25c5c13745a7bca98.jpg)

DANN2  
![](images/a1dbf83e5ecd9ca4d1091955bbcac1efac22008f1fff0d72efea53d1e3d5c15a.jpg)

DANN3  
![](images/7628f80c6692194095ea113363f8106bc16e8c5f5598b6132b6096f3df6a81a8.jpg)

![](images/cc6b2593f0aa5a8e37c607bbd3fb1d906613e6840d72fe5ac7c7c3f1594af498.jpg)

![](images/cf67f4aa7e72edbd821721d56c274509b03c9c7978b885dfe1701c1b5da13f11.jpg)

DANN4  
![](images/dd4b4eef4222c8f4bdb7eb6fd319f26a8f79566749793672c24e379854ec51c0.jpg)

DANN5  
![](images/147b15eac97c1595e9f6ff851ff8109c0f98801f695d0065f87e8a7701f9409b.jpg)

![](images/4cfc35b5dd5b42be0e5081c6851ae966450c363b4bb78d3104b9c02beb46dfbe.jpg)

![](images/ee650520e2346ea3a5be0924101a56c77c06c5e4b601b2e8b052a2b48a57a539.jpg)

DANN6  
![](images/0fb5f092f8260519f95fc5ce0d267f405abe12a22b0dabccc9dbb0ac626fe368.jpg)

Supplementary Table 3 Pairwise Comparison of AUC Statistics Across DANN Models in External Validation
<table><tr><td rowspan=1 colspan=3>Pairwise comparison AUC           p value</td><td rowspan=1 colspan=2>Z value95% CI</td></tr><tr><td rowspan=1 colspan=1>DANN1 vs DANN2</td><td rowspan=1 colspan=1>0.966 / 0.976</td><td rowspan=1 colspan=1>0.5239</td><td rowspan=1 colspan=1>0.637</td><td rowspan=1 colspan=1>-0.0211 to 0.0414</td></tr><tr><td rowspan=1 colspan=1>DANN1 vs DANN3</td><td rowspan=1 colspan=1>0.966 / 0.972</td><td rowspan=1 colspan=1>0.6757</td><td rowspan=1 colspan=1>0.418</td><td rowspan=1 colspan=1>-0.0219 to 0.0337</td></tr><tr><td rowspan=1 colspan=1>DANN1 vs DANN4</td><td rowspan=1 colspan=1>0.966 / 0.954</td><td rowspan=1 colspan=1>0.5371</td><td rowspan=1 colspan=1>0.617</td><td rowspan=1 colspan=1>-0.0273 to 0.0524</td></tr><tr><td rowspan=1 colspan=1>DANN1 vs DANN5</td><td rowspan=1 colspan=1>0.966 / 0.871</td><td rowspan=1 colspan=1>0.0012</td><td rowspan=1 colspan=1>3.233</td><td rowspan=1 colspan=1>0.0376 to 0.153</td></tr><tr><td rowspan=1 colspan=1>DANN1 vs DANN6</td><td rowspan=1 colspan=1>0.966 / 0.793</td><td rowspan=1 colspan=1>&lt;0.0001</td><td rowspan=1 colspan=1>6.141</td><td rowspan=1 colspan=1>0.118 to 0.229</td></tr><tr><td rowspan=1 colspan=1>DANN2 vs DANN3</td><td rowspan=1 colspan=1>0.976 / 0.972</td><td rowspan=1 colspan=1>0.7827</td><td rowspan=1 colspan=1>0.276</td><td rowspan=1 colspan=1>-0.0258 to 0.0343</td></tr><tr><td rowspan=1 colspan=1>DANN2 vs DANN4</td><td rowspan=1 colspan=1>0.976 / 0.954</td><td rowspan=1 colspan=1>0.2824</td><td rowspan=1 colspan=1>1.075</td><td rowspan=1 colspan=1>-0.0187 to 0.0641</td></tr><tr><td rowspan=1 colspan=1>DANN2 vs DANN5</td><td rowspan=1 colspan=1>0.976 / 0.871</td><td rowspan=1 colspan=1>&lt;0.0001</td><td rowspan=1 colspan=1>4.187</td><td rowspan=1 colspan=1>0.0562 to 0.155</td></tr><tr><td rowspan=1 colspan=1>DANN2 vs DANN6</td><td rowspan=1 colspan=1>0.976 / 0.793</td><td rowspan=1 colspan=1>&lt;0.0001</td><td rowspan=1 colspan=1>5.651</td><td rowspan=1 colspan=1>0.120 to 0.247</td></tr><tr><td rowspan=1 colspan=1>DANN3 vs DANN4</td><td rowspan=1 colspan=1>0.972 / 0.954</td><td rowspan=1 colspan=1>0.3806</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>-0.0228 to 0.0598</td></tr><tr><td rowspan=1 colspan=1>DANN3 vs DANN5</td><td rowspan=1 colspan=1>0.972 / 0.871</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>3.599</td><td rowspan=1 colspan=1>0.0462 to 0.157</td></tr><tr><td rowspan=1 colspan=1>DANN3 vs DANN6</td><td rowspan=1 colspan=1>0.972 / 0.793</td><td rowspan=1 colspan=1>&lt;0.0001</td><td rowspan=1 colspan=1>5.709</td><td rowspan=1 colspan=1>0.118 to 0.241</td></tr><tr><td rowspan=1 colspan=1>DANN4 vs DANN5</td><td rowspan=1 colspan=1>0.954 / 0.871</td><td rowspan=1 colspan=1>0.0125</td><td rowspan=1 colspan=1>2.497</td><td rowspan=1 colspan=1>0.0178 to 0.148</td></tr><tr><td rowspan=1 colspan=1>DANN4 vs DANN6</td><td rowspan=1 colspan=1>0.954 / 0.793</td><td rowspan=1 colspan=1>&lt;0.0001</td><td rowspan=1 colspan=1>5.351</td><td rowspan=1 colspan=1>0.102 to 0.220</td></tr><tr><td rowspan=1 colspan=1>DANN5 vs DANN6</td><td rowspan=1 colspan=1>0.871 / 0.793</td><td rowspan=1 colspan=1>0.0436</td><td rowspan=1 colspan=1>2.018</td><td rowspan=1 colspan=1>0.00225 to 0.154</td></tr></table>

## Supplementary Figure 3 ROC Curves for Senior and Junior Graders

The senior grader had an AUC of 0.999 (SE = 0.000365; 95% CI: 0.988–1.000), and the junior grader had an AUC of 0.933 (SE = 0.0146; 95% CI: 0.901–0.957); the difference between AUCs was 0.0665 (SE = 0.0145; 95% CI: 0.0381–0.0949; z = 4.587; p < 0.0001).

![](images/bc486ca4661c814e8cf51c93035f3eed16e4a7e83e4f0802a2bcf112771890f8.jpg)