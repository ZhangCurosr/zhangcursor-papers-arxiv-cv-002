# The BRIDGE dataset: paired brightfield wholeslide images of prostate core-needle biopsies in parafin, after deparafinization and H&E staining

Julia A. Mielcarz<sup>1\*</sup>, Justyna Durślewicz<sup>2,3</sup>, Marek Zdrenka<sup>2</sup>, Ahmad Abbadi<sup>1</sup>, Martin Eklund<sup>1</sup>, Mateusz Matusiak<sup>2,3</sup>, Łukasz Szylberg<sup>2,4</sup>, Andrea Camilloni<sup>1</sup>

<sup>1\*</sup>Department of Medical Epidemiology and Biostatistics, Karolinska Institutet, Stockholm, Sweden. <sup>2</sup>Department of Tumor Pathology and Pathomorphology, Prof. Franciszek Łukaszczyk Oncology Centre, Bydgoszcz, Poland. <sup>3</sup>Center for Translational Pathology, Faculty of Medicine, Bydgoszcz University of Science and Technology, Bydgoszcz, Poland. <sup>4</sup>Department of Obstetrics, Gynaecology and Oncology, Ludwik Rydygier Collegium Medicum in Bydgoszcz, Nicolaus Copernicus University in Toruń, Bydgoszcz, Poland.

\*Corresponding author(s). E-mail(s): julia.mielcarz.jm@gmail.com

## Abstract

Computational approaches are increasingly being explored to streamline histopathology workflows, including label-free methods that aim to reduce reliance on chemical staining. Among these, deep learning-based label-free virtual staining uses images of unstained tissue to computationally generate stained images. Despite growing interest, model development and validation remain constrained by the scarcity of paired, clinically annotated whole-slide imaging datasets. We present BRIDGE, a dataset of prostate core-needle biopsies collected at the Prof. Franciszek Łukaszczyk Oncology Centre in Bydgoszcz, Poland. The dataset comprises 869 glass slides from 169 patients, each digitized sequentially after sectioning, deparafinization, and hematoxylin and eosin staining, yielding in total 2607 whole-slide images. The dataset supports the development and external validation of label-free virtual staining, virtual unstaining, imageregistration and segmentation methods. Each slide is accompanied by a Gleason

score, enabling downstream analyses of cancer classification and grading concordance. The stained images can be used independently for color normalization and cross-scanner robustness studies. All images are de-identified, distributed in their native scanner format and will be available through the BioImage Archive.

Keywords: Stain-free histopathology, Virtual staining, Whole-slide imaging, Prostate cancer, Computational pathology, Digital pathology

## Background

Histopathological diagnosis traditionally relies on tissue sectioning and histochemical staining, most commonly with hematoxylin and eosin (H&E), to reveal tissue architecture and cellular morphology [1]. However, staining adds time, reagents and laboratory processing, motivating the development of stain-free approaches that aim to simplify the conventional workflow.

Deep learning-based label-free virtual staining computationally generates images resembling conventional histochemical stains from unstained tissue images. These approaches use generative models to learn a mapping between label-free tissue images and histologically stained counterparts [2]. Because the tissue is imaged before chemical staining, the same section remains available for subsequent histological processing, special stains, molecular assays or sequencing-based analyses. Stain-free approaches may therefore reduce reagent consumption and processing time while preserving limited tissue material for complementary analyses.

Several imaging modalities have been explored for label-free virtual staining, including tissue autofluorescence microscopy [3], quantitative phase imaging [4], fluorescence-lifetime imaging [5] or photoacoustic microscopy [6]. However, these techniques typically require specialized imaging equipment that is not routinely available in pathology laboratories, potentially limiting their integration into existing clinical workflows. In contrast, brightfield imaging can leverage conventional whole-slide scanners already widely used in digital pathology.

Label-free virtual staining from unstained brightfield images has been investigated in several tissues, including skin biopsies [7], prostate core biopsies [8], brain, lymph nodes and liver tissue [9]. Despite promising demonstrations, studies have generally relied on relatively small cohorts collected at a single institution and under a single acquisition protocol, with limited independent external validation.

Dataset size and diversity are particularly important for generative models, which must capture both the transformation between unstained and stained tissue and variation in morphology and image acquisition. Insuficient representation of these factors can lead to overfitting, unstable training and poor generalization. Scans of the same unstained and stained tissue sections are especially valuable for supervised approaches because, after registration, they provide corresponding tissue structures without the anatomical diferences introduced by adjacent sections.

Robust generalization is particularly important for label-free virtual staining because it is intended as a replacement-level system, in which the generated image substitutes for the physically stained slide and is subsequently interpreted by a pathologist or analyzed by downstream AI systems for classification, grading, stratification or other clinically relevant tasks [10]. Hallucinations introduced during image generation may propagate to subsequent analyses and clinical decisions. Replacement-level systems thus require substantially stronger validation across institutions, scanners and tissue-processing protocols, and ultimately regulatory-grade evidence. Current labelfree virtual staining studies often do not provide this level of evidence, with validation largely restricted to data from a single institution and acquisition setting. Meeting these requirements is challenging in practice because paired brightfield datasets of human unstained and stained tissue are rarely publicly available, necessitating new collections. To our knowledge, the only openly accessible human tissue dataset is E-Staining DermaRepo, comprising 87 paired unstained and H&E-stained skin-biopsy slides from 22 patients [11].

To address this gap, we present the BRIDGE dataset, a clinically annotated dataset of human prostate core-needle biopsies collected at the Prof. Franciszek Łukaszczyk Oncology Centre in Bydgoszcz, Poland. The dataset comprises 869 physical glass slides from 169 patients, with each section digitized sequentially at three stages of histological processing: immediately after sectioning with parafin retained, after deparafinization but before chemical staining, and after conventional H&E staining. The complete collection therefore contains 2607 whole-slide images (WSIs). This design permits direct comparison of the tissue before and after parafin removal and staining while avoiding the biological mismatch inherent to images acquired from adjacent sections.

The dataset can be used both to develop label-free virtual staining and virtualunstaining models and as an external dataset for evaluating models trained on diferent patient populations, scanners or tissue-preparation protocols. The unstained WSIs can additionally support the development and evaluation of segmentation models that identify the tissue area directly from unstained brightfield images, an important preprocessing step for downstream stain-free computational pathology pipelines. The availability of Gleason scores further enables assessment of the clinical applicability of stain-free approaches, including evaluation of computational grading performance and concordance with reported pathological grades. The H&E images can also be used independently of the unstained images for studies of prostate cancer classification, color normalization and the robustness of computational pathology models across scanners.

By releasing paired WSIs of the same prostate tissue sections at three histological processing stages together with diagnostic annotations, the BRIDGE dataset aims to facilitate the development and independent evaluation of robust stain-free computational pathology methods.

## Methods

## Ethical considerations

The patient sample collection was approved by the Bioethical Committee of Politechnika Bydgoska in Bydgoszcz, Poland (permit WM.0025.13.2026KB). Given the retrospective nature of the study and the use of de-identified prostate specimens, the ethics committee did not require informed consent. Access to and processing of the data at Karolinska Institutet were approved by the Swedish Ethical Review Authority (permits 2019-05220, 2025-07557-02). The samples were de-identified before transfer to Karolinska Institutet.

## Study design

This study involves retrospective sample collection of prostate core-needle biopsies, laboratory slide preparation, whole-slide scanning at three stages of tissue preparation and pathologist annotation on the stained image, as summarized in Figure 1.

![](images/1befdf18f202bfe245765d0aae40c909cbfd235ac1d900862b2e7df6fbefc911.jpg)  
Fig. 1 Overview of the sample collection pipeline.

## Laboratory slide processing and scanning

Formalin-fixed tissue specimens were processed and embedded in parafin. In some cases, Safranin O dye was added during parafin embedding to improve the tissue visibility and facilitate sectioning. The amount and concentration of the dye were not standardized, which resulted in variation of the dye intensity and the raw tissue appearance (Figure 2). The resulting formalin-fixed, parafin-embedded (FFPE) tissue blocks were sectioned at a thickness of 3.5 µm and mounted onto glass slides. After sectioning and initial drying, the slides were scanned prior to deparafinization, generating WSIs. These images represented raw tissue sections prior to any chemical processing.

![](images/684596c26ebf0ff7b544188d6f3c086b9f577a19773b107c0e1c205cc48d72f7.jpg)  
Fig. 2 Variation in the appearance of the raw tissue sections due to Safranin O application during parafin embedding. a) no dye, b) faint pink tint, c) pink-brownish tint, d) intense dye color.

Following the initial scan, the slides were incubated in a drying oven at 60 °C for 30 minutes and subsequently deparafinized in two consecutive xylene baths for 5 minutes each. Afterwards, the slides were allowed to air-dry for approximately 3 minutes to remove residual xylene and were rescanned to generate deparafinized WSIs.

The slides were subsequently subjected to routine automated H&E staining using a Tissue-Tek Prisma<sup>®</sup> Plus automated slide stainer and Tissue-Tek<sup>®</sup> H&E Staining Kit (Sakura Finetek). The staining procedure was performed according to the validated Sakura H&E staining protocol and comprised rehydration, hematoxylin staining, differentiation, bluing, eosin counterstaining, dehydration, clearing, and coverslipping. Following H&E staining and coverslipping, the slides were scanned for a third time to obtain the corresponding H&E-stained WSIs.

All slide scans at the three processing stages were generated using the VENTANA DP 600 Slide Scanner (Roche Diagnostics; serial no. DP6000236) at 40× magnification with a resolution of 0.25 µm/pixel and stored in a TIFF-based format. An example of scans resulting from one glass slide is presented in Figure 3.

The dataset includes rare modalities such as raw unstained tissue brightfield images, for which very little public data is available and thus for which there is less established intuition about image quality. Assessment of the scan quality, such as identification of the out-of-focus regions, is more challenging for these images due to low contrast and more unconventional tissue appearance compared to established H&E-stained images. Moreover, the quality of unstained tissue imaging may be sensitive to the scanner vendor, scanner model, and acquisition settings, as these systems can be optimized for conventionally stained histology slides. Each scan was therefore assigned a quality label through visual inspection. The labels represent a subjective, descriptive assessment and are reported both to facilitate comparison of image quality across datasets and scanners and to raise awareness of potential scanner-specific limitations when imaging unstained tissue. We hope that reporting these observations will contribute to a broader discussion within the stain-free imaging community and help establish more systematic approaches to evaluating image quality for this emerging modality. Further details on the quality and dye labels are provided in the Data records section.

![](images/59623929da2107068da910d311de64ed33ba25c1c87b49875ecbf871fa88bc50.jpg)  
Fig. 3 Example of a glass slide scanned at three stages of tissue processing: raw, deparafinized, H&E-stained. Whole-slide thumbnails and regions at 10x magnification are shown for each stage.

## Reference standard protocol

The digitized slides were reviewed by M.Z., a certified uropathologist from Prof. Franciszek Łukaszczyk Oncology Centre in Bydgoszcz with 5 years of diagnostic experience. Each H&E-stained whole-slide image was assessed using Roche VENTANA uPath Digital Pathology software and assigned a slide-level Gleason score. Slides without evidence of malignancy were labeled as benign (0+0). The Gleason scores were then mapped to the corresponding ISUP Grade Groups. No additional pixel-level annotations were provided. A total of 869 H&E-stained WSIs were evaluated, with the final labels associated with the respective de-identified slide IDs.

Table 1 Patient and slide characteristics of the BRIDGE dataset.
<table><tr><td rowspan=1 colspan=2>Number of patients</td><td rowspan=1 colspan=2>Number of slides</td></tr><tr><td rowspan=1 colspan=2>n =169</td><td rowspan=1 colspan=2>n = 869</td></tr><tr><td rowspan=1 colspan=2>Age, years</td><td rowspan=1 colspan=2>ISUP grade</td></tr><tr><td rowspan=1 colspan=1>≤ 49 yrs</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Benign (0+0)</td><td rowspan=1 colspan=1>502</td></tr><tr><td rowspan=1 colspan=1>50–54 yrs</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>ISUP 1 (3 + 3)</td><td rowspan=1 colspan=1>102</td></tr><tr><td rowspan=1 colspan=1>55–59 yrs</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>ISUP 2 (3 + 4)</td><td rowspan=1 colspan=1>78</td></tr><tr><td rowspan=1 colspan=1>60–64 yrs</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>ISUP 3 (4 + 3)</td><td rowspan=1 colspan=1>84</td></tr><tr><td rowspan=1 colspan=1>65–69 yrs</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>ISUP4 (4 + 4, 3 + 5, 5 + 3)</td><td rowspan=1 colspan=1>58</td></tr><tr><td rowspan=1 colspan=1>≥ 70 yrs</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>ISUP5 (4+5,5+4,5+ 5)</td><td rowspan=1 colspan=1>35</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Neuroendocrine cancer</td><td rowspan=1 colspan=1>10</td></tr></table>

## Data records

The deposited data are organized within a single root directory, “BRIDGE”, containing three subfolders corresponding to the consecutive tissue-processing stages: component\_A\_raw, component\_B\_deparafinized, and component\_C\_HEstained. These folders contain WSIs acquired before deparafinization, after deparafinization but before staining, and after H&E staining, respectively. In total, the dataset contains 869 glass slides from 169 patients.

A common non-identifying slide ID is retained across the three processing stages, allowing the corresponding images to be matched directly. File names follow the format <slide\_id>\_<stage>.<ext>, where the stage sufix indicates the tissue-processing state: \_1 for raw, non-deparafinized tissue, \_2 for deparafinized unstained tissue, and \_3 for H&E-stained tissue.

WSI-level metadata is provided in BRIDGE\_metadata.csv. The table includes patient identifier (Patient\_ID), patient age at biopsy (Age), glass slide identifier (Slide\_ID), WSI identifier (WSI\_ID representing slide identifier followed by the processing stage sufix), processing stage identification (Processing\_stage), slide-level Gleason score (GS), corresponding ISUP Grade Group (ISUP) and binary tumor label (Tumor). The table also reports the acquisition quality of each scan (Scan\_quality). For raw tissue scans, the dye intensity (Dye\_intensity) is additionally provided.

Dye intensity was recorded on a four-level scale: 0 (no dye applied), 1 (faint pink tint), 2 (pink-brownish tint) and 3 (intense dye color) as illustrated in Figure 2. Scan quality was labeled as high (high level of detail), medium (minor out-of-focus areas or artifacts), low (unclear quality, seemingly largely out-of-focus) or unacceptable (fully out-of-focus).

## Data usage

The WSIs can be viewed using digital pathology software. Typical preprocessing steps for computational analysis may include tissue segmentation, image tiling and, where appropriate, stain normalization.

## Data availability

The complete collection will be made publicly available through the BioImage Archive upon publication, and the accession number will be provided in the final version of the manuscript. A summary of the dataset and its documentation are available at https://github.com/JAMielcarz/BRIDGE-dataset. The data will be distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. All data are de-identified and contain no direct personal identifiers. Images and metadata can be linked to each other through the anonymized patient and slide identifiers provided with the dataset.

## Code availability

No custom code was used to create or process the dataset. Slide scanning was performed using the scanner software provided by the manufacturer.

## Author contributions

J.A.M. conceptualized the study, curated and processed the data, and drafted the manuscript.

J.D. oversaw the data collection including sample preparation and scanning.

M.M. contributed to laboratory processing of the samples.

M.Z. provided slide-level annotations.

A.A. and M.E. managed the ethical approvals and data access at Karolinska Institutet.

Ł.S. and A.C. supervised the work.

All authors read and approved the manuscript.

## Competing interests

The authors declare no competing non-financial interests. The following competing financial interests are declared: A.C. is a shareholder of Clinsight AB and has received honoraria from Astellas Pharma AB. M.E. is a shareholder of Clinsight AB and A3P Biomedical AB, and holds patents in A3P Biomedical AB.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

## References

[1] Chan, J. K. C. The wonderful colors of the hematoxylin-eosin stain in diagnostic surgical pathology. Int. J. Surg. Pathol. 22, 12–32 (2014).

[2] Latonen, L., Koivukoski, S., Khan, U. & Ruusuvuori, P. Virtual staining for histology by deep learning. Trends Biotechnol. 42, 1177–1191 (2024).

[3] Rivenson, Y. et al. Virtual histological staining of unlabelled tissueautofluorescence images via deep learning. Nat. Biomed. Eng. 3, 466–477 (2019).

[4] Rivenson, Y. et al. PhaseStain: the digital staining of label-free quantitative phase microscopy images using deep learning. Light Sci. Appl. 8, 23 (2019).

[5] Wang, Q. et al. Deep learning-based virtual H&E staining from label-free autofluorescence lifetime images. npj Imaging 2, 17 (2024).

[6] Yoon, C. et al. Deep learning-based virtual staining, segmentation, and classification in label-free photoacoustic histology of human specimens. Light Sci. Appl. 13, 226 (2024).

[7] Hussain, M. A. et al. VISGAB: virtual staining-driven GAN benchmarking for optimizing skin tissue histology. Sci. Rep. 15, 42430 (2025).

[8] Rana, A. et al. Use of deep learning to develop and analyze computational hematoxylin and eosin staining of prostate core biopsy images for tumor diagnosis. JAMA Netw. Open 3, e205111 (2020).

[9] Misra, S. et al. Deep learning based label-free virtual staining and classification of human tissues using digital slide scanner. Med. Image Anal. 108, 103865 (2026).

[10] Misra, S. et al. Virtual histological staining: toward standardization and clinical translation. Biomed. Eng. Lett. 16, 855–882 (2026).

[11] Asaf, M. Z. et al. E-Staining DermaRepo: H&E whole slide image staining dataset. Data Brief 57, 110997 (2024).