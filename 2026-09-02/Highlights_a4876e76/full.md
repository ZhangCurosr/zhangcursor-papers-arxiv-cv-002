## Highlights

## Benchmarking Vision-Language Models for Automated Pathology Diagnosis and Report Generation

Yumi Lee, Harim Oh, Hyoryung Kim, Minji Kim, Eunsu Kim, Hyeseong Lee, Junya Fukuoka, Andrey Bychkov, Jijgee Munkhdelger, Rajiv Kumar Kaushal, Ayushi Sahay, Rajni Yadav, Bharathi Prabakaran, Sulen Sarioglu, Serdar Balcı, Ilknur Turkmen, Yuri Tolkach, Christian Harder, Julian Westerdorf, Reinhard Buettner, Audun Ljone Henriksen, Sepp De Raedt, Byung Hyun Lee, Sungjin Lim, Joohoon Lee, Gwanghyun Kim, Se Young Chun, Suryakant Singh, Saarthak Kapse, Prateek Prasanna, Kyung A Kim, Yousun Kang, Sehwan Yoo, Sungman Hong, Shubham Innani, Michael Feldman, Spyridon Bakas, Ujjwal Baid, Prasad Dutande, Suhas Gajare, Bhakti Baheti, Serkan Sökmen, Ece Tuğba Cebeci, Ahmet Halıcı, Musa Balcı, Kardelen Peçenek, Srividhya Sainath, Kyongseok Jang, Messi H.J. Lee, Noorul Wahab, Bodong Du, Jiaming Zhang, Qixiang Zhang, Jang-Hwan Choi, Sangjeong Ahn

• Introduced the first large-scale, clinically curated Pan-Asia WSI–report dataset for vision–language learning in computational pathology.

• Established the REG 2025 benchmark through a MICCAI challenge for systematic evaluation of pathology report generation models.

• Demonstrated strong cross-regional generalization of models trained on Pan-Asia data to independent European cohorts.

• Provided a comprehensive comparative analysis of preprocessing strategies and multimodal model architectures across 11 submissions.

• Revealed key limitations and characteristics of pathology report generation, including structured reasoning requirements and quantitative instability.

# Benchmarking Vision-Language Models for Automated Pathology Diagnosis and Report Generation<sup>⋆</sup>

Yumi Lee<sup>a,1</sup>, Harim Oh<sup>b,1</sup>, Hyoryung Kim<sup>a</sup>, Minji Kim<sup>a</sup>, Eunsu Kim<sup>c</sup>, Hyeseong Lee<sup>c</sup>, Junya Fukuoka<sup>e,f</sup>, Andrey Bychkov<sup>e</sup>, Jijgee Munkhdelger<sup>e</sup>, Rajiv Kumar Kaushal<sup>g</sup>, Ayushi Sahay<sup>g</sup>, Rajni Yadav<sup>h</sup>, Bharathi Prabakaran<sup>h</sup>, Sulen Sarioglu<sup>d</sup>, Serdar Balcı<sup>d</sup>, Ilknur Turkmen<sup>d</sup>, Yuri Tolkach<sup>i</sup>, Christian Harder<sup>i</sup>, Julian Westerdorf<sup>i</sup>, Reinhard Buettner<sup>i</sup>, Audun Ljone Henriksen<sup>j</sup>, Sepp De Raedt<sup>j</sup>, Byung Hyun Lee<sup>k</sup>, Sungjin Lim<sup>k</sup>, Joohoon Lee<sup>k</sup>, Gwanghyun Kim<sup>k</sup>, Se Young Chun<sup>k</sup>, Suryakant Singh<sup>l</sup>, Saarthak Kapse<sup>l</sup>, Prateek Prasanna<sup>l</sup>, Kyung A Kim<sup>m</sup>, Yousun Kang<sup>n</sup>, Sehwan Yoo<sup>o</sup>, Sungman Hong<sup>p</sup>, Shubham Innani<sup>q</sup>, Michael Feldman<sup>q</sup>, Spyridon Bakas<sup>q</sup>, Ujjwal Baid<sup>r</sup>, Prasad Dutande<sup>s</sup>, Suhas Gajare<sup>s</sup>, Bhakti Baheti<sup>r</sup>, Serkan Sökmen<sup>t</sup>, Ece Tuğba Cebeci<sup>t</sup>, Ahmet Halıcı<sup>t</sup>, Musa Balcı<sup>t</sup>, Kardelen Peçenek<sup>t</sup>, Srividhya Sainath<sup>u</sup>, Kyongseok Jang<sup>v</sup>, Messi H.J. Lee<sup>v</sup>, Noorul Wahab<sup>w</sup>, Bodong Du<sup>x</sup>, Jiaming Zhang<sup>y</sup>, Qixiang Zhang<sup>x</sup>, Jang-Hwan Choi<sup>a,∗</sup> and Sangjeong Ahn<sup>b,∗</sup>

<sup>a</sup>Department ofArtificial Intelligence, Ewha Womans University, 52 Ewhayeodae-gil, Seodaemun-gu, Seoul, 03760, Republic ofKorea   
<sup>b</sup>Department ofPathology, Korea University Anam Hospital, 73 Goryeodae-ro, Seongbuk-gu, Seoul, 02841, Republic ofKorea   
<sup>c</sup>Department ofBiomedical Informatics, Korea University College ofMedicine, 73, Goryeodae-ro, Seongbuk-gu, Seoul, 02841, Republic ofKorea   
<sup>d</sup>Department ofPathology, Memorial Health Group, Burhaniye, Nagehan Sokağı No:4/A D:1, Üsküdar, 34676, İstanbul, Türkiye   
<sup>e</sup>Department ofPathology, Kameda Medical Center, 929 Higashicho, Kamogawa, 296-0041, Chiba, Japan   
<sup>f</sup>Department ofPathology Informatics, Nagasaki University Graduate School ofBiomedical Sciences, 1-7-1 Sakamoto, Nagasaki, 852-8523, Japan   
<sup>g</sup>Department ofPathology, Tata Memorial Hospital, Homi Bhabha National Institute (HBNI), Mumbai, 400012, India   
<sup>h</sup>Department ofPathology, All India Institute OfMedical Sciences Delhi, Sri Aurobindo Marg, Ansari Nagar East, New Delhi, 110029, Delhi, India   
<sup>i</sup>Institute ofPathology, University Hospital Cologne, Medical Faculty, University ofCologne, Kerpener Str. 62, Köln, 50937, Germany   
<sup>j</sup>Institutefor Cancer Genetics and Informatics, Ullernchausseen 70, Oslo, 0372, Norway   
<sup>k</sup>Seoul National University, 1 Gwanak-ro, Gwanak-gu, Seoul, 08826, Republic ofKorea   
<sup>l</sup>Stony Brook University, 100 Nicolls Rd, Stony Brook, NY 11794, New York, United States   
<sup>m</sup>Department ofPathology, Yonsei University College ofMedicine, Seoul, Republic ofKorea   
<sup>n</sup>Tokyo Polytechnic University, 2 Chome-9-5 Honcho, Nakano City, Tokyo 164-8678, Japan   
<sup>o</sup>Nanyang Technological University, 50 Nanyang Ave, 639798, Singapore   
<sup>p</sup>Graduate School ofSoftware and Artificial Intelligence Convergence, Korea University, 145 Anam-ro, Seongbuk-gu, Seoul, 02841, Republic ofKorea   
<sup>q</sup>Division ofComputational Pathology, Department ofPathology and Laboratory Medicine, Indiana University School ofMedicine, 410 W 10th   
St, Indianapolis, IN, 46202, Indiana, United States   
<sup>r</sup>Emory University, 201 Dowman Dr, Atlanta, GA 30322, Georgia, United States   
<sup>s</sup>Shri Guru Gobind Singhji Institute ofEngineering and Technology, Guru Tegh Bahadurji Marg, Vishnupuri, Nanded, 431606, Maharashtra, India   
<sup>t</sup>Viseur AI, Söğütözü, 9 Eylül Cad No: 4 İç Kapı No:1, Çankaya, 06510, Ankara, Türkiye   
<sup>u</sup>EKFZ TU Dresden (KatherLab), Fetscherstraße 74, Dresden, Postfach 151, 01307, Germany   
<sup>v</sup>MTS Company R&D Team, 434 Samsung-ro, Gangnam-gu, Seoul, 06178, Republic ofKorea   
<sup>w</sup>NW-TIA, University ofWarwick, Coventry, CV4 7AL, United Kingdom   
<sup>x</sup>The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong   
<sup>y</sup>Harbin Institute ofTechnology, HIT Campus ofUniversity Town ofShenzhen, Shenzhen, 518055, China

## A R T I C L E I N F O

Keywords:   
Computational Pathology (CPath)   
Pan-Asia   
Pathology Report Generation   
Vision-Language Model (VLM)

## A BS T RA C T

The rapid advancement of vision-language models (VLMs) has accelerated progress in computational pathology; however, whole-slide image (WSI)-based pathology report generation remains limited by the scarcity of large-scale WSI–report datasets and the complexity of mapping spatially distributed visual patterns to structured clinical text. To address this, we introduce a clinically curated Pan-Asia WSI–report dataset of approximately 10,500 pairs from five institutions and establish the REG 2025 benchmark through a MICCAI challenge for systematic evaluation of multimodal models. We analyze submitted methods spanning pretrained VLMs, multiple-instance learning frameworks, hierarchical expert models, retrieval-augmented generation, and cross-modal Transformers. Rather than indicating that VLM use alone was suficient for superior performance, the results suggest that top-performing methods benefited from structured report representations, hierarchical diagnostic decomposition, and efective multimodal grounding. We identify key limitations, including instability in quantitative attribute estimation (e.g., numeric hallucination) and a tendency toward diagnostic overspecification, with some errors resembling known diagnostic pitfalls in routine pathology. These findings establish REG 2025 as a benchmark for evaluating WSI-based structured report generation and vision-language understanding in computational pathology, providing insights for the design of clinically grounded multimodal pathology models.

## 1. Introduction

Pathology report generation from whole-slide images (WSIs) remains a fundamentally challenging problem in computational pathology, despite recent advances in artificial intelligence [59, 23, 44]. Unlike conventional vision–language tasks, this problem requires translating gigapixel-scale images, characterized by spatially distributed and weakly localized visual patterns, into clinically meaningful textual reports that summarize global tissue characteristics and diagnostic conclusions [40, 68, 67]. Such reports are inherently abstract, often reflecting integrated clinical reasoning rather than direct visual descriptions [48, 60]. This introduces a substantial spatial–semantic gap between image and text modalities, further compounded by weak and indirect alignment, where slide-level supervision provides no explicit correspondence between local regions and textual components [82, 20, 22]. In addition, the extreme scale and sparsity of WSIs, where diagnostically relevant regions occupy only a small fraction of the image, make reliable feature aggregation particularly dificult [4, 68]. The problem is further complicated by the one-to-many nature of report generation, as multiple clinically valid descriptions may exist for a single case, and by the variability in reporting styles across pathologists and institutions [28, 5, 51, 53]. Together, these characteristics distinguish pathology report generation from conventional multimodal learning problems and highlight the need for specialized modeling strategies and standardized data resources.

To address these challenges, a wide range of artificial intelligence–based approaches have been developed in computational pathology following the digitization of WSIs [59, 23, 44]. Early methods primarily relied on convolutional neural network (CNN)–based patch-level analysis [19, 24, 84, 39, 61, 8], which were later extended to multiple-instance learning (MIL) frameworks [44, 33, 26, 55, 11, 47, 7, 27] to enable slide-level inference without requiring exhaustive annotations. More recently, foundation models trained on large-scale datasets have been introduced to learn generalpurpose representations that can be transferred across downstream tasks [71, 12, 43, 75, 66, 20]. Building upon these advances, recent studies have further explored vision–language models (VLMs) for pathology report generation, aiming to bridge visual and textual modalities in a unified framework [37, 13, 10, 42, 62, 79]. Despite these eforts, existing approaches remain limited in efectively addressing the intrinsic challenges of pathology report generation, particularly in terms of weak alignment, abstract semantic reasoning, and the lack of standardized paired datasets.

This discrepancy becomes more evident when compared to the rapid progress observed in radiology report generation [69, 57], where large-scale paired image–report datasets and more explicit global image–text alignment have enabled substantial advances. In contrast, advancements in pathology report generation remain comparatively limited [22, 25]. These limitations are primarily driven by the intrinsic characteristics of WSIs. Unlike radiology images, WSIs are gigapixel in scale and require patch-based processing, which disrupts global spatial coherence [4]. Moreover, pathology reports summarize complex spatial patterns that are not directly localized, leading to weak image–text alignment [82, 20, 22].

In addition, pathology reports are predominantly semistructured free text [56, 54], and in practice, standardized reporting protocols such as CAP synoptic guidelines are not consistently adopted across institutions [16, 18]. This issue is particularly pronounced in settings with limited digital pathology infrastructure or incomplete adoption of synoptic reporting systems [58, 65], where diagnostic findings are often documented in narrative form without consistently structured diagnostic fields or explicit final diagnoses. Even when formal guidelines exist, reporting practices frequently vary across pathologists, institutions, and countries in terms of terminology, report structure, and diagnostic granularity [64]. For example, breast pathology reporting practices vary considerably across institutions, with some centers adopting CAP synoptic templates [15], whereas others continue to rely on institution-specific narrative reporting styles. Such heterogeneity introduces significant supervision noise between gigapixel WSIs and text, thereby limiting reliable multimodal learning, weakening image–text alignment, and complicating quantitative evaluation [34, 63].

Publicly available WSI benchmarks, including CAME-LYON [39], PANDA [6], and SurGen [49], have substantially advanced computational pathology research in tasks such as classification, segmentation, and survival prediction. However, pathology image–text benchmarks for WSI-based report generation remain extremely limited, with representative examples including TCGA [73], HISTAI [50], and HANCOCK [21]. In TCGA, pathology reports are typically provided as manually written PDF documents, requiring extensive preprocessing and parsing to extract usable text [29, 9, 30], while the reports themselves remain highly heterogeneous in structure, terminology, and linguistic style across institutions, weakening the supervision signal between WSIs and text. HISTAI, although designed for large-scale pretraining, primarily focuses on representation learning rather than standardized report generation, and its reports remain largely unstructured despite text cleaning procedures [41, 32]. HANCOCK [21] is further constrained by its focus on a specific cancer type, limiting both data diversity and crossorgan generalization. Collectively, these limitations hinder reliable quantitative evaluation and clinically grounded multimodal learning [35, 36], highlighting the lack of largescale, clinically standardized, and multi-institutional WSI– report benchmarks for pathology report generation.

![](images/d67dde7df5acd7c7a2ca221ab473f5a738a01822b69e1253a20f7f1f7a05c852.jpg)  
Figure 1: REG 2025 challenge workflow. (a) Multi-institutional WSI–report pairs were collected using three scanner platforms. (b) The data were curated through organ- and diagnosis-level categorization and anonymization. (c) The curated dataset was distributed to participants, and the generated reports were collected and evaluated as part of the challenge.

To address these limitations, we introduce a high-quality, clinically curated WSI–report dataset constructed in accordance with the College of American Pathologists (CAP)<sup>1</sup> guidelines, which define the core elements of complete pathology reporting and provide a synoptic framework for reducing inter-institutional variability [58]. The proposed dataset comprises approximately 10,500 paired WSI–report samples collected from five institutions across South Korea, India, Japan, Turkey, and Germany, representing the first large-scale Pan-Asia–centric multi-institutional dataset for pathology report generation, to the best of our knowledge.

Building upon this dataset, we further establish a largescale benchmark through the REport Generation in Pathology using Pan-Asia Giga-pixel WSIs 2025 (REG 2025) Challenge at the Medical Image Computing and Computer Assisted Intervention<sup>2</sup> conference, enabling systematic evaluation of multimodal models under realistic clinical settings. A total of 389 participants from 40 countries took part in the challenge, among which 24 teams successfully submitted their final models. The submitted approaches span diverse paradigms, including pretrained VLM-based methods [37, 13, 10, 42, 62, 79], pathology-specific MIL frameworks [44, 26, 55, 11], and cross-modal Transformer architectures [71, 20], reflecting the current landscape of computational pathology.

Beyond dataset construction and benchmarking, we provide a comprehensive empirical analysis of the REG 2025 Challenge, ofering insights into the fundamental characteristics of pathology report generation. Specifically, our analysis reveals that:

1. Models leveraging pretrained VLM backbones with efective multimodal fusion consistently achieve superior performance across report generation quality metrics.

2. Models trained on Pan-Asia data exhibit strong generalization to independent European cohorts, suggesting the potential to mitigate Western-centric bias in existing pathology datasets.

3. Clinical precision, linguistic coherence, and completeness of pathological reasoning are strongly correlated with overall model performance.

4. Conventional approaches remain limited in capturing the structured nature of pathology reports, highlighting the importance of structured learning for pathology report generation.

These findings establish REG 2025 as the first benchmark for jointly evaluating report generation and visionlanguage understanding in computational pathology, while providing practical insights for the development of future multimodal pathology models.

Table 1  
Representative diagnostic categories across seven organs. The pathology dataset included in the challenge were curated and organized according to the following diagnostic categories.
<table><tr><td rowspan=1 colspan=1>Organ</td><td rowspan=1 colspan=1>Diagnostic Categories</td></tr><tr><td rowspan=1 colspan=1>Stomach</td><td rowspan=1 colspan=1>Adenocarcinoma; Tubular adenoma (low/high-grade dysplasia); Extranodal marginal zone B celllymphoma (MALT type); Chronic (active) gastritis; Others</td></tr><tr><td rowspan=1 colspan=1>Prostate</td><td rowspan=1 colspan=1>Acinar adenocarcinoma; Normal; Others</td></tr><tr><td rowspan=1 colspan=1>Lung</td><td rowspan=1 colspan=1>Adenocarcinoma; Squamous cell carcinoma; Small cell carcinoma; Chronic granulomatous inflammation;No evidence of malignancy or granuloma; Others</td></tr><tr><td rowspan=1 colspan=1>Colorectum</td><td rowspan=1 colspan=1>Adenocarcinoma; Tubular adenoma (low/high-grade dysplasia); Hyperplastic polyp; Serrated lesion;Chronic nonspecific inflammation; Others</td></tr><tr><td rowspan=1 colspan=1>Cervix</td><td rowspan=1 colspan=1>Low-grade squamous intraepithelial lesion (LSIL); High-grade squamous intraepithelial lesion (HSIL);Invasive squamous cell carcinoma; Endocervical adenocarcinoma in situ (AIS), HPV-associated; Chronicnonspecific cervicitis; Others</td></tr><tr><td rowspan=1 colspan=1>Bladder</td><td rowspan=1 colspan=1>Invasive urothelial carcinoma; Non-invasive papillary urothelial carcinoma; Urothelial carcinoma in situ;No tumor present</td></tr><tr><td rowspan=1 colspan=1>Breast</td><td rowspan=1 colspan=1>Invasive ductal carcinoma; Invasive lobular carcinoma; Ductal carcinoma in situ (DCIS); Fibroepithelialtumor; Papillary neoplasm; Others</td></tr></table>

Table 2

Number of cases by organ across training, Phase 1, and Phase 2 datasets. Each dataset was stratified to maintain a consistent distribution across organs and institutions.
<table><tr><td rowspan="2">Organs</td><td colspan="3">Number of cases by organ</td></tr><tr><td>Train</td><td>Phase 1</td><td>Phase 2</td></tr><tr><td>Breast</td><td>1916</td><td>112</td><td>112</td></tr><tr><td>Cervix</td><td>621</td><td>37</td><td>37</td></tr><tr><td>Colorectum</td><td>944</td><td>328</td><td>298</td></tr><tr><td>Lung</td><td>898</td><td>50</td><td>50</td></tr><tr><td>Prostate</td><td>1770</td><td>331</td><td>361</td></tr><tr><td>Stomach</td><td>1477</td><td>87</td><td>87</td></tr><tr><td>Bladder</td><td>868</td><td>55</td><td>55</td></tr><tr><td>Total</td><td>8494</td><td>1000</td><td>1000</td></tr></table>

## 2. Materials and Methods

## 2.1. Dataset Description and Preprocessing Pipeline

WSIs and their corresponding pathology reports were collected from five institutions: Korea University Anam Hospital (Korea), Kameda Medical Center (Japan), All India Institute of Medical Sciences (India), Memorial Healthcare Group Istanbul (Türkiye) and University Hospital Cologne (Germany) (Fig. 1-(a)). WSIs were digitized using institutionspecific scanners: Aperio AT2 at Korea University Medical Center and Memorial Healthcare Group; Philips IntelliSite Pathology Solution at Kameda Medical Center; and Hamamatsu NanoZoomer at All India Institute of Medical Sciences and University Hospital Cologne.

The dataset spans seven organs-breast, bladder, cervix, colorectum, lung, prostate, and stomach- and each WSI was paired with its corresponding pathology report (Fig. 1- (b)). As the original reports difered substantially across institutions in structure, terminology, level of descriptive detail, and language, we generated standardized ground-truth pathology reports using a unified template with consistent structure and terminology (Fig. S1-(a)). Initial screening was performed by four pathology trainees and subsequently reviewed by three board-certified pathologists. Each curated ground-truth report included the organ, procedure, histologic type and, when applicable, histologic grade, while excluding information not directly observable from pathological images, such as anatomical location or diagnoses based on clinical context.

Histologic terminology was standardized according to the 5th edition of the World Health Organization (WHO) Classification of Tumours; for example, the term "invasive ductal carcinoma" was redesignated as "invasive carcinoma of no special type". For malignant tumor cases, selected organ-specific elements were additionally included in the pathology report with reference to the corresponding CAP Cancer Protocol. These included parameters such as the Nottingham histologic score for breast carcinoma and Gleason grading for prostate needle biopsy specimens. The CAP Cancer Protocol versions used in this study were invasive carcinoma of breast v1.2.0.0, ductal carcinoma in situ of breast v1.0.1.0, urinary bladder v4.2.0.0 and prostate gland needle biopsy v1.1.0.0 (Fig. S1-(c)). For stomach biopsy specimens, the terminology for low-grade dysplasia difered across institutions and was unified as “tubular adenoma with low-grade dysplasia” through consensus meetings.

Inclusion and exclusion criteria were applied in two stages: technical WSI quality screening followed by expert review of diagnostic certainty. First, cases were excluded on technical grounds if the slide contained insuficient tissue for pathologic assessment, if the lesional focus was too limited for reliable interpretation, or if there was out-offocus scanning, poor staining, excessive cautery artifact, or other technical deficiencies. Second, cases that were technically acceptable but diagnostically equivocal were excluded after expert review. This latter step was intended to reduce label noise arising from unavoidable interobserver variability in borderline lesions (e.g. stomach biopsy between well-diferentiated adenocarcinoma vs. high-grade dysplasia). Together, these steps ensured that the final dataset comprised cases that were both technically adequate and diagnostically reliable, resulting in a curated cohort 15.4% smaller than the initial collection.

All curated datasets were anonymized and the pyramid level corresponding to 20× magnification was extracted and exported as TIFF files. During anonymization, associated images (“label”, “macro”, and “thumbnail”) were removed from Aperio-scanned files (Fig. S1-(b)). For WSIs containing multiple specimens on a single image, each image was manually cropped into separate image regions, ensuring that each cropped slide image corresponded to its respective pathology diagnosis. Anonymized filenames followed the format PIT\_{institution ID}\_{slide number}\_{split number}.tiff, where the split number was used for manually cropped images containing multiple specimens.

In total, the final dataset consists of 10,494 WSI–report pairs, representing a broad spectrum of diagnostic categories across seven organs, including malignant entities (e.g., adenocarcinoma, squamous cell carcinoma), premalignant lesions (e.g., tubular adenoma with low-grade dysplasia), benign conditions, and non-neoplastic tissues (Table 1). This composition reflects the diagnostic spectrum encountered in routine pathology practice and provides a clinically relevant setting for model development.

This study was approved by the Institutional Review Boards (IRB) of Korea University Medical Center (No. K2024-2640-001) and Kameda Medical Center (No. 22- 094), the Ethics Committee of Memorial Healthcare Group Istanbul (decision no. 28.03.2025/001), and the Ethics Committee of the University of Cologne (ethics votes 25-1057 and 20-1583). Data transfer and external use were additionally reviewed and approved by the KUMC Data Review Board (No. D2024-074-CR). For datasets provided by the All India Institute of Medical Sciences, separate IRB approval at the contributing institution was not required because the data were fully anonymized and contained no patient-identifiable information. The REG2025 dataset was released under the CC BY-NC-SA 4.0 license for non-commercial research and benchmarking purposes.

## 2.2. Challenge Design and Workflow

The challenge was hosted on the Grand Challenge<sup>3</sup> platform (Fig. 1-(c)). Participants were provided with approximately 8,500 preprocessed WSI–report paired samples as the training dataset, and the competition was conducted in two stages: Test Phase 1 and Test Phase 2 (Table 2). The dataset was partitioned at the patient level to prevent data leakage. The training data were distributed through the remote server, and participants were given two months after the dataset release to develop and train their models. In each test phase, participants received 1,000 WSI-only samples, for which they were required to generate corresponding pathology reports within a three-week submission period. The evaluation procedure for the generated reports is described in Section 2.3.

Each team was allowed two submissions per phase, and all submitted reports were evaluated within a Docker-based environment to ensure consistency and reproducibility. The leaderboard displayed only the weighted Ranking Score– aggregated from the four evaluation metrics–to prevent participants from fine-tuning their models toward specific metrics. The leaderboard was updated in real time via GitHub<sup>4</sup>, allowing participants to monitor their relative performance throughout the competition.

The primary distinction between Test Phase 1 and Test Phase 2 was the inclusion of European data. Phase 1 consisted of 1,000 Pan-Asia samples collected from the same four hospitals as the training set, whereas Phase 2 included 500 Pan-Asia samples and an additional 500 European samples obtained from a German medical center. This design aimed to evaluate whether the models developed by participants could maintain robust performance across diferent ethnic and institutional domains. The final ranking score was computed as a weighted sum of the two phases:

$$
{ \mathrm { F i n a l ~ S c o r e } } = ( 0 . 2 { \times } { \mathrm { P h a s e ~ 1 ~ S c o r e } } ) + ( 0 . 8 { \times } { \mathrm { P h a s e ~ 2 ~ S c o r e } } )
$$

## 2.3. Evaluation Metrics

Traditional natural language processing (NLP) metrics were not designed to evaluate clinically structured pathology reports, where subtle diferences in wording can alter diagnostic meaning. To address this limitation, we developed a clinically grounded composite evaluation framework in the REG 2024 Challenge<sup>5</sup>, which was conducted as a domestic pilot study with limited scale and served as a preliminary validation of the proposed framework. Rather than relying on a single metric, the evaluation employed a composite score that integrates multiple complementary evaluation metrics designed to reflect the structured and diagnostically relevant properties of pathology reports.

The ranking score was defined as follows:

$$
\begin{array} { r } { \mathrm { R a n k i n g ~ s c o r e } = 0 . 1 5 \times \mathrm { ( R O U G E + B L E U ) } + \phantom { - } } \\ { 0 . 4 \times \mathrm { K E Y ~ s c o r e } + 0 . 3 \times \mathrm { E M B ~ s c o r e } } \end{array}
$$

where the KEY score represents the Jaccard similarity between sets of clinically relevant keywords extracted from reports, and the EMB score measures the cosine similarity between sentence embeddings derived from a biomedical domain-specialized language model [2]. The detailed implementation is available at https://github.com/hrb0/reg/.

As the participating models were trained on curated pathology reports and aimed to reproduce established diagnostic expressions, we evaluated surface-level textual overlap between generated and ground truth reports using ROUGE-L [38] and BLEU-4 [52], two conventional wordlevel similarity metrics. However, simple lexical overlap alone cannot adequately capture clinical equivalence in pathology reporting. For example, semantically equivalent statements such as “core-needle biopsy” and “sono-guided core biopsy” convey identical diagnostic conclusions despite diferences in wording.

Overall performance comparison of participating teams in the REG 2025 Challenge. Phase-specific scores represent the weighted sum of ROUGE, BLEU, KEY, and EMB metrics. Teams are ranked according to the final score (— denotes no submission for the corresponding phase.).
<table><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>Team</td><td rowspan=1 colspan=2>|Phase 1 Phase 2</td><td rowspan=1 colspan=1>Final</td><td rowspan=1 colspan=1>||Rank</td><td rowspan=1 colspan=1>Team</td><td rowspan=1 colspan=2>Phase 1 Phase 2</td><td rowspan=1 colspan=1>Final</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>ICGI</td><td rowspan=1 colspan=2>0.80980.8472</td><td rowspan=1 colspan=1>0.8397</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>TokenStreamers           |</td><td rowspan=1 colspan=2>0.72480.5773</td><td rowspan=1 colspan=1>0.6068</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>ICL_PathReport</td><td rowspan=1 colspan=2>0.73520.8415</td><td rowspan=1 colspan=1>0.8202</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>NUST-SEECS</td><td rowspan=1 colspan=1>0.5144</td><td rowspan=1 colspan=1>0.6121</td><td rowspan=1 colspan=1>0.5926</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>IMAGINE Lab</td><td rowspan=1 colspan=2>0.62580.8494</td><td rowspan=1 colspan=1>0.8047</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>Clinical_Omics_Institute</td><td rowspan=1 colspan=1>0.5531</td><td rowspan=1 colspan=1>0.5984</td><td rowspan=1 colspan=1>0.5894</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>ADCT</td><td rowspan=1 colspan=2>0.77040.8040</td><td rowspan=1 colspan=1>0.7973</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>JX_REG</td><td rowspan=1 colspan=1>0.5073</td><td rowspan=1 colspan=1>0.5309</td><td rowspan=1 colspan=1>0.5262</td></tr><tr><td rowspan=4 colspan=1>5678</td><td rowspan=1 colspan=1>IUCompPath</td><td rowspan=1 colspan=1>0.7906</td><td rowspan=1 colspan=1>0.7899</td><td rowspan=1 colspan=1>0.7900</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>DSG</td><td rowspan=1 colspan=2>0.43850.4286</td><td rowspan=1 colspan=1>0.4306</td></tr><tr><td rowspan=1 colspan=1>TrustPath</td><td rowspan=1 colspan=1>0.6565</td><td rowspan=1 colspan=1>0.8127</td><td rowspan=1 colspan=1>0.7815</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>sk</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.5277</td><td rowspan=1 colspan=1>0.4221</td></tr><tr><td rowspan=1 colspan=1>TeamTiger@REG2025</td><td rowspan=1 colspan=1>0.7777</td><td rowspan=1 colspan=1>0.7763</td><td rowspan=1 colspan=1>0.7766</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>virasoft</td><td rowspan=1 colspan=1>0.1691</td><td rowspan=1 colspan=1>0.4590</td><td rowspan=1 colspan=1>0.4010</td></tr><tr><td rowspan=1 colspan=1>MedInsight-ViseurAI</td><td rowspan=1 colspan=1>0.5326</td><td rowspan=1 colspan=1>0.8093</td><td rowspan=1 colspan=1>0.7539</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>Shubham47</td><td rowspan=1 colspan=1>0.4402</td><td rowspan=1 colspan=1>0.3849</td><td rowspan=1 colspan=1>0.3960</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>PathoMozhi</td><td rowspan=1 colspan=1>0.4611</td><td rowspan=1 colspan=1>0.7960</td><td rowspan=1 colspan=1>0.7290</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>DiceMed_labs</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.3577</td><td rowspan=1 colspan=1>0.2861</td></tr><tr><td rowspan=3 colspan=1>101112</td><td rowspan=1 colspan=1>MTS_REG_2025</td><td rowspan=1 colspan=2>0.60460.7584</td><td rowspan=1 colspan=1>0.7276</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>UTWL</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.3110</td><td rowspan=1 colspan=1>0.2488</td></tr><tr><td rowspan=1 colspan=1>NW-TIA</td><td rowspan=1 colspan=2>0.8282</td><td rowspan=1 colspan=1>0.6626</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>May_REG</td><td rowspan=1 colspan=2>0.5245</td><td rowspan=1 colspan=1>0.1049</td></tr><tr><td rowspan=1 colspan=1>REG_Path</td><td rowspan=1 colspan=2>0.69070.6510</td><td rowspan=1 colspan=1>0.6589</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>IMI REG</td><td rowspan=1 colspan=2>0.5027</td><td rowspan=1 colspan=1>0.1005</td></tr></table>

Moreover, keyword-based matching alone may fail to distinguish clinically critical negations, such as “the specimen includes muscle proper” versus “the specimen does not include muscle proper”. To address these limitations, the evaluation framework integrates both keyword-level assessment of essential diagnostic elements and a semantic similarity component that captures the overall meaning of the report. In the REG 2024 Challenge, this ranking score demonstrated a high level of concordance with expert-based mean opinion scores. Rankings derived from the composite score aligned most closely with pathologists’ subjective assessments of overall report quality and diagnostic adequacy, supporting its use as a practical and clinically meaningful surrogate metric for large-scale benchmarking of pathology report generation models.

## 3. Results

The challenge attracted a total of 389 participants from 40 countries. In Phase 1, submissions were received from 20 teams, comprising 8 individual and 12 group teams, while Phase 2 included 22 teams, consisting of 9 individual and 13 group teams.

## 3.1. Quantitative Evaluation Results

The ranking scores of all teams for each phase are summarized in Table 3. In Phase 1, the highest score was achieved by the ICGI team with a score of 0.8098, while the overall average reached 0.5915. In Phase 2, the IMAGINE Lab team obtained the highest score of 0.8494, with an average score of 0.6523, indicating a clear performance improvement compared to Phase 1. These results suggest that models trained on the Pan-Asia dataset maintained strong performance when evaluated on European cohorts, indicating the potential for cross-regional generalization.

![](images/400720748c967a83faf22eefbcb5075ed4af356a48027293953bb39b89852002.jpg)  
Figure 2: Score distribution of participating teams across Phase 1, Phase 2, and the German subset of Phase 2. Similar score distribution patterns are observed for ROUGE, BLEU, KEY, and EMB metrics in both phases. (R: ROUGE, B: BLEU, K: KEY score, E: EMB score)

Based on Final Score, the top three teams were ICGI, ICL\_PathReport, and IMAGINE Lab, all achieving final scores above 0.8000, underscoring their outstanding performance.

Furthermore, we compared the results according to four evaluation metrics – ROUGE, BLEU, KEY, and EMB scores – as presented in Table S1, S2. When the teams were ranked in descending order of their overall Ranking Score, the KEY and EMB metrics exhibited similar trends in both Phase 1, Phase 2, and the German subset of Phase 2 (Fig. 2).

Interestingly, while distribution patterns were largely consistent across Phase 1, Phase 2, and the German subset, the German subset showed higher mean performance with lower variance than the Pan-Asia subset. In contrast, the Pan-Asia dataset exhibited lower overall performance and higher inter-team variability, suggesting broader institutional and demographic heterogeneity. Although relative team rankings were largely preserved across evaluation subsets, these findings indicate that the Pan-Asia dataset may provide a more demanding setting for evaluating model robustness under diverse clinical conditions. Importantly, this interpretation reflects diferences in dataset composition and variability rather than inherent diferences in population-level dificulty.

![](images/177954ae0325efd0b04a61fa46b839b373b5105c7c6e75c3341db9952d221fad.jpg)  
(a) Performance on General Case

![](images/c8cff245be379825e98ee2768394f58af3ffd0df621ebd5b8e2a347e5e7d3c2c.jpg)  
(c) Organ-wise Performance Comparison

![](images/485fb87fe250a0d7bb0033a4c1ee21d6d1d444c18a891b3c8f7092ecd8478dbe.jpg)

![](images/28a845f71e7799e10a104f4338e00ebeef542c85c9613a10080f973d58dc54a5.jpg)  
(d) Institution-wise Performance Comparison  
Figure 3: Category-wise performance analysis of participating models. (a) Organ-wise performance comparison for general cases . (b) Performance comparison on 11 rare diagnostic cases. (c) Radar plot illustrating organ-wise performance across all cases. (d) Score distribution across contributing institutions.

Moreover, in Phase 2, we observed a sharp decline in BLEU and KEY scores among teams whose Ranking Scores fell below 0.7000. Notable cases include:

• In Phase 1, the IMAGINE Lab team showed an unusually low BLEU score compared to its ROUGE performance.

• In Phase 1, the virasoft team presented abnormally low EMB scores despite typically high values for most participants.

• In Phase 2, the ICL\_PathReport and PathoMozhi teams achieved similar ROUGE scores but displayed substantial diferences in KEY scores.

• In Phase 2, the MedInsight-ViseurAI, ADCT, and IU-CompPath teams showed comparable BLEU scores but varied significantly in ROUGE values.

These case studies are further discussed in Section 3.2, where we provide a detailed comparative analysis. Based on these observations, we confirm that the Ranking Score employed in this challenge appropriately balanced penalization and reward, efectively reflecting the overall quality of the generated pathology reports.

## 3.2. Qualitative Analysis of Generated Reports

We performed a qualitative assessment to examine how well the generated reports captured the diagnostic content and contextual features of the corresponding WSIs and ground-truth pathology reports. The representative WSIs with generated pathology reports are shown in Supplementary Tables (Table S5 – S12).

## 3.2.1. Diagnostic Concordance: Attribute Fidelity and Robustness

1) Comprehensive Diagnostic Accuracy with Attributelevel Fidelity The models generated pathology reports that showed a high level of diagnostic concordance with the ground-truth (GT) while preserving clinically relevant attribute-level pathologic parameters (Fig.3-(a)). In breast core biopsy case, the ICGI team correctly generated the main diagnosis as “Ductal carcinoma in situ (DCIS)” and

Table 4  
Quantitative attribute instability in generated pathology reports. The red annotation highlights small tumor area in a prostate biopsy whole-slide image, corresponding to a ground-truth tumor volume of 5%. Although several teams correctly identified the prostate origin, Gleason’s score and grade group, estimation of tumor burden was highly unstable and frequently inflated, with generated tumor-volume values ranging from 10% to 90%. ( =organ, =procedure, =correct, =incorrect)
<table><tr><td rowspan=1 colspan=4>Whole Slide Image</td></tr><tr><td rowspan=1 colspan=4><img src="images/826a58b289734bab5e5f4f7a253737fe9ce74ce6bd12084a7e887cbd2dd04b5b.jpg"/></td></tr><tr><td rowspan=1 colspan=2>Ground Truth</td><td rowspan=1 colspan=1>ADCT (0.4056)</td><td rowspan=1 colspan=1>ICGI (0.7316)</td></tr><tr><td rowspan=1 colspan=2>Prostate,biopsy;Acinar adenocarcinoma,Gleason&#x27;s score 8 (4+4),grade group 4,tumor volume: 5%</td><td rowspan=1 colspan=1>Prostate, biopsy;No tumor present</td><td rowspan=1 colspan=1>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score6 (3+3),grade group1,tumor volume:80%</td></tr><tr><td rowspan=1 colspan=1>ICL_PathReport (0.9751)</td><td rowspan=1 colspan=1>IMAGINE_Lab (0.7317)</td><td rowspan=1 colspan=1>IUCompPath (0.4056)</td><td rowspan=1 colspan=1>MedInsight-ViseurAI (0.9749)</td></tr><tr><td rowspan=1 colspan=1>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score 8 (4+4),grade group 4,tumor volume:15%</td><td rowspan=1 colspan=1>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score10 (5+5),grade group5,tumor volume:70%</td><td rowspan=1 colspan=1>Prostate, biopsy;No tumor present</td><td rowspan=1 colspan=1>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score 8 (4+4),grade group 4,tumor volume:50%</td></tr><tr><td rowspan=1 colspan=1>REG_Path (0.4056)</td><td rowspan=1 colspan=1>TeamTiger@REG2025 (0.9750)</td><td rowspan=1 colspan=1>PathoMozhi (0.3866)</td><td rowspan=1 colspan=1>NW-TIA (0.7316)</td></tr><tr><td rowspan=1 colspan=2>Prostate, biopsy;                 Prostate, biopsy;No tumor present                 Acinar adenocarcinoma,Gleason&#x27;s score 8 (4+4),grade group 4,tumor volume:90%</td><td rowspan=1 colspan=1>Prostate, biopsy;No tumor present</td><td rowspan=1 colspan=1>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score10 (5+5),grade group 5,tumor volume:10%</td></tr></table>

appropriately specified the DCIS type, nuclear grade, and necrosis status in alignment with the GT report (Table S5).

Similarly, in prostate biopsies, the ICGI team’s generated report- “Acinar adenocarcinoma, Gleason score 7 (3+4), grade group 2 (Gleason pattern 4: 10%), tumor volume: 90%” - accurately captured the histologic pattern, Gleason grading in a structured format consistent with routine pathology reporting. Although tumor volume was overestimated (90% vs 50% in reference), the hierarchy and organization of the diagnostic elements closely mirrored real-world pathology reporting practice (Table S6). Notably, similar limitations in quantitative attribute estimation were also observed across other participating teams, which will be further discussed in Section 3.2.2.

2) Diagnostic Robustness Across Rare and Minor Pathologic Categories The models also exhibited notable diagnostic robustness in rare or minor diagnostic categories represented by fewer than 10 cases in the full dataset (Fig.3-(b)). Accurate generation was observed for entities such as malignant lymphoma, metaplastic carcinoma of the breast (ICGI team), and fundic gland polyp of the stomach (MTS\_REG team) (Table S7, S8, and S9).

Although gastric malignant melanoma was misclassified in most models as chronic gastritis, the MTS\_REG team correctly generated “malignant melanoma,” suggesting preserved discriminatory capacity even in diagnostically uncommon settings.

Table 5  
Quantitative comparison of top-performing teams and benchmark models across evaluation metrics. Comparison of the firstplace teams from Phase 1 and Phase 2 of the REG 2025 Challenge with existing publicly available benchmark models.
<table><tr><td>Model</td><td>Phase 1</td><td>ROUGE</td><td>BLEU</td><td>KEY</td><td>EMB</td><td>Phase 2</td><td>ROUGE</td><td>BLEU</td><td>KEY</td><td>EMB</td></tr><tr><td>ICGI</td><td>0.8098</td><td>0.8188</td><td>0.6372</td><td>0.7514</td><td>0.9696</td><td>0.8472</td><td>0.8564</td><td>0.6852</td><td>0.8080</td><td>0.9759</td></tr><tr><td>IMAGINE Lab</td><td>0.6258</td><td>0.7485</td><td>0.0895</td><td>0.5620</td><td>0.9177</td><td>0.8494</td><td>0.8572</td><td>0.6928</td><td>0.8094</td><td>0.9770</td></tr><tr><td>MI-Gen</td><td>0.6928</td><td>0.6870</td><td>0.4480</td><td>0.6003</td><td>0.9414</td><td>0.7055</td><td>0.6981</td><td>0.4562</td><td>0.6223</td><td>0.9447</td></tr><tr><td>HistGen</td><td>0.4576</td><td>0.5429</td><td>0.4182</td><td>0.3555</td><td>0.5708</td><td>0.4647</td><td>0.5500</td><td>0.4233</td><td>0.3664</td><td>0.5739</td></tr></table>

3) High-fidelity Histomorphology-driven Phenotype Recognition In selected challenging cases, the generated reports appeared to be primarily anchored to histomorphologic pattern recognition (i.e., what the tumor looks like on H&E). Gastric malignant lymphoma was correctly generated by most models; For PIT\_01\_08124\_01, four models misclassified the case as “adenocarcinoma, poorly diferentiated”. For PIT\_01\_09100\_01, two models misclassified the case as “small cell carcinoma” (Table S10). This error is consistent with the well-recognized histomorphologic overlap encountered in limited gastric biopsies and represents a common diagnostic pitfall in routine practice, particularly among less-experienced pathologists.

A particularly illustrative example was the uterine cervix case with “Adenocarcinoma, favor colorectal primary”. Most models generated “Adenocarcinoma, moderately dif ferentiated” of colon/rectum (Table S11). Although the anatomic site in the generated text was incorrect, the output captured a colorectal-type gland-forming phenotype consistent with the suspected primary, underscoring the model’s ability to encode and reproduce diagnostically relevant morphologic features.

## 3.2.2. Diagnostic Discordance: Quantitative Miscalibration and Overcalling

1) Quantitative Attribute Instability and Miscalibration Despite strong pattern recognition, the models demonstrated instability in quantitative attribute estimation, with frequent inflation of numeric values consistent with numeric hallucination. Tumor proportion was particularly prone to overestimation. In the representative prostate biopsy case in Table 4, ICL\_PathReport, TeamTiger@REG2025, and MedInsight-ViseurAI correctly identified acinar adenocarcinoma, Gleason score, and grade group and achieved high REG scores (0.9749-0.9751), but all overestimated tumor volume (80% vs. 5% in GT report). Models that preserved the correct diagnosis of acinar adenocarcinoma but misclassified the Gleason score and grade group—IMAGINE\_Lab, ICGI, and NW-TIA—received intermediate scores (0.73x). By contrast, models that failed to detect carcinoma and instead reported “No tumor present”—ADCT, IUCompPath, REG\_Path, and PathoMozhi—received substantially lower scores (0.3866–0.4056). Together, these findings show that the REG score tracked the hierarchical severity of diagnostic errors.

In breast biopsies, grading-related discordance was observed in mitotic activity and tubule formation scores. For tubule formation, scores were assigned as follows: <10% for score 3, 10–75% for score 2, >75% for score 1 (Table S12). These findings suggest limited capacity to translate visual estimation onto rule-based grading frameworks required for routine reporting.

2) Over-specification and Failure to Preserve Uncertainty A recurring error was diagnostic commitment in settings where diagnostic retention of a broader category would have been more appropriate. Rather than preserving diagnostically meaningful uncertainty, the model frequently collapsed diagnoses into definite labels. Fibroepithelial lesions were frequently over-specified asfibroadenoma or phyllodes tumor. Similarly, diagnoses such as “non-small cell carcinoma,favor adenosquamous carcinoma” was collapsed into definitive adenocarcinoma or squamous cell carcinoma.

In breast biopsies, atypical ductal hyperplasia (ADH) was commonly overcalled as DCIS, indicating dificulty preserving the conservative diagnostic threshold that pathologists apply at the ADH–DCIS boundary.

## 3.3. Benchmark Comparison with Existing Approaches

We further compared the top-performing challenge submissions with previously reported pathology report generation methods and general-purpose vision–language models [9, 22]. As shown in Table 5, publicly available benchmark models exhibited average performance drops of approximately 16.9% in Phase 1 and 30.4% in Phase 2 relative to the leading challenge submissions. Notably, many baseline approaches primarily formulate report generation as a direct sequence generation task from slide-level visual features, without explicitly modeling the structured and hierarchical nature of pathology reports.

In contrast, the highest-performing challenge methods consistently incorporated mechanisms for structured clinical representation, such as pathology-aware report organization, multimodal grounding, or intermediate diagnostic reasoning, enabling more efective preservation of clinically relevant information (Fig. 3-(c), (d)). These findings suggest that pathology report generation extends beyond conventional image captioning and instead requires structured multimodal reasoning capable of maintaining both hierarchical diagnostic relationships and clinically grounded concept consistency.

Category-wise summary of approaches and components. Overview of methods used by participating teams in the REG 2025 Challenge.
<table><tr><td rowspan=1 colspan=1>Team</td><td rowspan=1 colspan=1>Report Processing</td><td rowspan=1 colspan=1>WSI Preprocessing</td><td rowspan=1 colspan=1>Model Architecture / Methods</td><td rowspan=1 colspan=1>Foundation Models</td></tr><tr><td rowspan=1 colspan=1>ICGI</td><td rowspan=1 colspan=1>Category-wise separationand typo correction</td><td rowspan=1 colspan=1>Trident</td><td rowspan=1 colspan=1>ABMIL-based tile integration + n-gram-constrained Transformer decoder</td><td rowspan=1 colspan=1>H-Optimus-1</td></tr><tr><td rowspan=1 colspan=1>ICLPathReport</td><td rowspan=1 colspan=1>Redefined as a hierarchicalannotation tree</td><td rowspan=1 colspan=1>Trident</td><td rowspan=1 colspan=1>PRISM slide encoder aggregation + Tree-of-Experts (ToE)-based hierarchical report gen-eration</td><td rowspan=1 colspan=1>Virchow</td></tr><tr><td rowspan=1 colspan=1>IMAGINELab</td><td rowspan=1 colspan=1>GPT-generated   conceptprompts for organ anddiagnosis categories</td><td rowspan=1 colspan=1>Trident</td><td rowspan=1 colspan=1>MLP-based multimodal fusion + GPT-guidedcross-attention Transformer decoder</td><td rowspan=1 colspan=1>CONCH, TITAN, GPT</td></tr><tr><td rowspan=1 colspan=1>ADCT</td><td rowspan=1 colspan=1>Codebook generation foreach cancer type</td><td rowspan=1 colspan=1>Organ-specific patch sizing</td><td rowspan=1 colspan=1>ABMIL-based slide embedding + Flan-T5-based structured report generation</td><td rowspan=1 colspan=1>CONCH,  GigaPath,UNI,       Virchow,HistoEncoder</td></tr><tr><td rowspan=1 colspan=1>IUCompPath</td><td rowspan=1 colspan=1>HistGen-style tokenization</td><td rowspan=1 colspan=1>CLAM</td><td rowspan=1 colspan=1>LGH-based hierarchical encoding + Cross-modal context Transformer decoder</td><td rowspan=1 colspan=1>UNI, H-Optimus-1</td></tr><tr><td rowspan=1 colspan=1>TeamTiger@REG2025</td><td rowspan=1 colspan=1>WsiCaption-style tokeniza-tion</td><td rowspan=1 colspan=1>Divided   into   non-overlapping patches</td><td rowspan=1 colspan=1>MI-Gen Transformer architecture</td><td rowspan=1 colspan=1>UNI</td></tr><tr><td rowspan=1 colspan=1>Medlnsight-ViseurAI</td><td rowspan=1 colspan=1>Sentence-BERT    basedsemantic similarity post-processing</td><td rowspan=1 colspan=1>HSV-based patch selection</td><td rowspan=1 colspan=1>ViT-based visual encoder + BioGPT-basedreport generation decoder</td><td rowspan=1 colspan=1>UNI, BioGPT</td></tr><tr><td rowspan=1 colspan=1>PathoMozhi</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1>Canny-filtered patch selec-tion + STAMPv2 prepro-cessing</td><td rowspan=1 colspan=1>Gated cross-attention Flamingo framework +BioGPT for pathology report generation</td><td rowspan=1 colspan=1>CONCH, BioGPT</td></tr><tr><td rowspan=1 colspan=1>MTS REG2025</td><td rowspan=1 colspan=1>Report normalization andstructured template recon-struction</td><td rowspan=1 colspan=1>Prov-Gigapath preprocess-ing pipeline</td><td rowspan=1 colspan=1>RAG-based pipeline + Qwen-3 8B report gen-erator</td><td rowspan=1 colspan=1>Prov-Gigapath, Qwen-3 8B</td></tr><tr><td rowspan=1 colspan=1>NW-TIA</td><td rowspan=1 colspan=1>Auxiliary label generation</td><td rowspan=1 colspan=1>Trident</td><td rowspan=1 colspan=1>Embedding-driven BioBART framework withMulti-task auxiliary heads</td><td rowspan=1 colspan=1>CONCH, TITAN, Bio-BART</td></tr><tr><td rowspan=1 colspan=1>REG_Path</td><td rowspan=1 colspan=1>Q-A pair generation</td><td rowspan=1 colspan=1>Threshold-adaptive prepro-cessing</td><td rowspan=1 colspan=1>Two-stage fine-tuned SlideChat $( { \mathsf { V } } { \mathsf { Q A } } + { \mathsf { r } } { \mathsf { e } } -$ port generation)</td><td rowspan=1 colspan=1>CONCH, SlideChat</td></tr></table>

## 4. Overview of Submitted Algorithms

For the final evaluation, we requested code and algorithm descriptions from the top 13 teams that achieved a final score of 0.6000 or higher, and conducted a detailed analysis of the 11 teams that responded. A summary of their algorithmic frameworks is presented in Table 6. The submitted methods commonly followed three main stages – Report Processing, WSI Preprocessing, and Model Architecture – with most teams demonstrating active utilization of foundation models.

In terms of text preprocessing, two key strategies were consistently observed: normalization and hierarchical organization of the pathology reports. For WSI preprocessing (Fig. 4-(a), many teams reused verified open frameworks such as Trident [80], CLAM [44], and Prov-Gigapath [75], reflecting a trend toward the reuse of standardized pipelines. Regarding model architecture, the majority of methods adopted ABMIL- or Transformer-based structures [26, 55, 11, 37], where approaches emphasizing contextual grounding rather than simple feature aggregation tended to achieve higher performance.

Accordingly, this paper provides a detailed comparative analysis of the submitted methods, categorized into two main perspectives: Preprocessing Strategies and Model Architectures.

## 4.1. Preprocessing Strategies

Most teams employed standardized MIL-based frameworks such as Trident [80] or CLAM [44], or adopted similar mechanisms for WSI feature extraction. Several teams additionally incorporated strategies to account for organ-specific variations or to select visually salient tiles for more informative representations. However, the overall WSI preprocessing pipelines were largely consistent across participants, showing no significant performance diferences attributable to preprocessing strategies alone.

In contrast, the report processing stage exhibited notable diferences across participants. In particular, the top three teams–ICGI, ICL\_PathReport, and IMAGINE Lab–which consistently achieved high performance in both Phase 1 and Phase 2, employed strategies that partitioned report information into distinct categories:

• ICGI: This team explicitly separated text sections using tags such as <organ> and <diagnosis>, aligning each section with the corresponding tile features extracted from the Trident slide encoder. They further reduced redundant expressions by utilizing section-wise n-gram frequency analysis.

![](images/4ce6f77db74c9760ca38b69f9a76b281fea5e14068675625d34ea3f04933d9b9.jpg)  
Figure 4: Overview of major computational paradigms for pathology report generation from WSIs. (a) Patch-level feature extraction using CNN or transformer encoders (b) Slide-level feature aggregation for whole-slide representation (c) Organ- or task-specific expert modules (d) Multi-modal fusion frameworks with vision–language models (VLMs) (e) Retrieval-augmented generation leveraging similar historical cases (f) VQA-based diagnostic reasoning for structured report generation.

• ICL\_PathReport: Extending beyond simple section division, this team established hierarchical relationships among report categories using a hierarchical annotation tree. Each sentence was labeled by category, and the relationships between labels were defined as tree nodes, enabling the Tree-of-Experts (ToE) model to generate text sequentially from the root to preserve category-wise coherence.

• IMAGINE Lab: This team automatically generated concept prompts for each report category using a GPT-based model. The resulting prompts were used as guidance signals to enhance fluency and contextual coherence during text generation.

These three teams, all of which leveraged categoryspecific report structuring, achieved substantially higher text quality than other participants, reaching average scores of ROUGE 0.8572 and BLEU 0.6853 in Phase 2.

In addition, the NW-TIA team, which achieved the next highest performance following the top three teams in Phase 2, generated auxiliary clinical labels using a GPT-based text parser. These labels represented clinical categories not explicitly mentioned in the original reports and were used to guide the model in learning pathology-grounded reasoning regions. Similar to the top three teams, these four teams collectively treated pathology reports not as plain text sequences but as structured semantic maps, resulting in a markedly superior average KEY score of 0.7990 compared with other participants. Furthermore, the ADCT team constructed cancer-type-specific codebooks to enable sectionspecific report generation, while the MTS\_REG\_2025 team employed a template-based reconstruction approach for section-wise report synthesis.

In contrast, the remaining teams that tokenized the report data without category-wise structuring exhibited a noticeable decline in both BLEU and KEY scores. When comparing performance across diferent report processing strategies, teams employing category-aware methods achieved an average improvement of +0.09 in BLEU and +0.13 in KEY between Phase 1 and Phase 2, whereas tokenizationbased approaches showed only marginal gains (ΔBLEU ≈ +0.01, ΔKEY ≈ +0.02). This performance gap further widened in Phase 2, with diferences of approximately 0.14 in BLEU and 0.18 in KEY between the two groups.

These findings suggest that explicit report structuring leads to better cross-domain generalization and more stable preservation of key clinical terms under distributional shifts, underscoring the importance of developing structured report representations for robust pathology report generation.

## 4.2. Model Architectures

## 4.2.1. ABMIL-Based Architectures

The ICGI and ADCT teams employed Attention-based Multiple Instance Learning (ABMIL) [26] to generate slidelevel embeddings, which were then passed to Transformeror LLM-based text decoders in a conventional image-captioning pipeline for pathology report generation (Fig. 4-(b)). AB-MIL efectively summarizes a WSI into a global representation by aggregating tile-level features, enabling downstream text decoders to infer pathological context in a more stable and coherent manner.

The ICGI team incorporated an n-gram constrained Transformer decoder to suppress hallucinations commonly observed in LLM-based report generation. This architectural design demonstrated correct predictions in several rare (long-tail) cases (Fig. 3-(b)). In addition, the generated reports occasionally reflected subtle histomorphologic features; for instance, the model explicitly mentioned microcalcifications measuring approximately 30 µm, indicating its ability to generate reports with fine-grained pathological detail.

The ADCT team introduced a cancer-type-specific codebook to reinforce the prior knowledge required for Flan-T5–based structured generation [17], and further integrated cancer-specific knowledge distillation, resulting in more efficient and domain-aware text synthesis. These models also exhibited strong organ anchoring characteristics. In typical cases, this property contributed to producing stable and internally consistent diagnostic outputs. However, in the breast lymphoma case, ADCT generated “breast metaplastic carcinoma” instead of lymphoma. In this instance, organ anchoring itself was preserved, but the model struggled to accurately capture the rare morphologic pattern.

## 4.2.2. Tree-of-Experts-Based Architectures

The ICL\_PathReport team proposed a Tree-of-Experts (ToE) framework that departs from the conventional reliance on a single slide-level embedding (Fig. 4-(c)). Instead, their method explicitly models the inherent hierarchical structure of pathology reports – organ → diagnosis → subtype – through hierarchical feature aggregation and a category-specialized expert hierarchy. In this architecture, each node in the tree corresponds to a category-specific sub-expert, enabling the model to structurally disentangle diverse pathological knowledge encoded in the report. This design allows the model to separately learn organ-level semantics, diagnosis-level decision boundaries, and subtypelevel fine-grained attributes, providing a clearer hierarchical partitioning of the feature space compared with standard MIL-based global embeddings. As a result, the approach ofers inherent advantages in ensuring slot-level information disentanglement, making it better aligned with the structured nature of pathology reporting. A detailed illustration of the tree architecture is provided in Appendix A.2.

This architecture aligns well with the inherent hierarchical structure of pathology reporting, particularly for breast and prostate specimens. It is especially efective at capturing fine-grained diagnostic attributes (e.g., grade/score subcomponents, and tumor extent) In the breast lymphoma case, the model generated “prostate lymphoma,” even though prostate lymphoma cases were relatively few in the dataset. This pattern suggests that the model first committed to an incorrect organ-level routing decision (prostate) and subsequently searched for the most morphologically compatible diagnosis within that constrained branch. This behavior underscores a trade-of: hierarchical decomposition enhances logical clarity but reduces flexibility when upstream routing errors occur.

## 4.2.3. Cross-Modality Transformer-Based Architectures

The IMAGINE Lab, IUCompPath, and TeamTiger@REG2025 teams adopted multi-modal Transformer architectures that introduce cross-attention modules between image tokens extracted from the WSI encoder and textual tokens from the report(Fig. 4-(d)). This design explicitly models the interactions between visual and textual modalities, allowing the model to learn fine-grained alignment between specific regions in the WSI and corresponding categories or expressions in the generated report. Compared with approaches that simply feed a global WSI embedding into a text decoder, these cross-modality frameworks ofer substantially greater flexibility in capturing region-text correspondences.

The IMAGINE Lab team generated organ- and diagnosislevel concept prompts using a GPT-based model (as detailed in Table S3.) and used them as additional guidance signals during multi-modal fusion, thereby strengthening semantic alignment between visual and textual representations.

The IUCompPath team introduced a Cross-modal Context Module (CMC) [22] that contextually integrates image and text tokens, jointly supporting hierarchical encoding and cross-modal reasoning.

Meanwhile, the TeamTiger@REG2025 team preserved patch-level spatial information through a Position-Aware module, combining these spatially enriched image tokens with text tokens via cross-attention to enhance spatially grounded reasoning throughout the report generation process.

The outputs of these models were largely driven by histomorphologic pattern. In some cases, the generated pathology reports remained pathologically plausible despite incorrect organ-of-origin assignment. Language quality varied widely across these architectures, reflecting diferences in the underlying language model and in how strictly report generation was controlled.

## 4.2.4. Vision-Language Model-Based Architectures

The MedInsight-ViseurAI, PathoMozhi, and NW-TIA teams adopted Vision-Language Model (VLM)-based architectures, in which WSI features extracted from powerful vision encoders were provided as input to biomedical foundation models such as BioGPT [45] or BioBART [78] for pathology report generation (Fig. 4-(d)). Because BioGPT and BioBART models are pretrained extensively on biomedical terminology and clinical narrative structures, they are particularly well suited for medical domain text generation tasks, including pathology reporting.

The MedInsight-ViseurAI and PathoMozhi teams employed BioGPT decoders to enhance the generation of domain-specific terminology in WSI-based medical captioning.

The PathoMozhi team further incorporated a Flamingostyle gated cross-attention mechanism [1] to refine visionlanguage alignment.

Meanwhile, the NW-TIA team used a BioBART encoderdecoder architecture augmented with multi-task auxiliary heads to improve training stability and preserve the semantic richness of WSI features throughout the generation process.

## 4.2.5. Retrieval-Augmented Generation (RAG)-Based Architectures

The MTS\_REG\_2025 team proposed a Retrieval Augmented Generation (RAG) framework in which WSI-derived features were used to retrieve semantically similar pathology reports, and both the retrieved documents and the input context were provided to the Qwen3-8B [77] model to generate the final report (Fig. 4-(e)). RAG-based approaches mitigate hallucinations by incorporating explicit external knowledge, ofering improved semantic grounding–particularly for rare subtypes or cases involving complex morphological patterns.

To further enhance reliability, the team employed an Adaptive Retrieval strategy that dynamically adjusts the number of retrieved documents based on the similarity gap. This mechanism prevents both redundant retrieval (which introduces noise) and insuficient retrieval (which leads to information loss), thereby contributing directly to improved report consistency and semantic alignment.

These approaches achieved noticeably higher accuracy on rare (long-tail) cases, but showed reduced robustness on majority (high-prevalence) cases.

## 4.2.6. Visual Question Answering (VQA)-Based Architectures

The REG\_Path team proposed a two-stage framework that departs from conventional end-to-end captioning pipelines (Fig. 4-(f)). Instead of directly generating the entire report, their method first performs slot-level querying of the WSI using a SlideChat [13]-based VQA model, and then composes the final report by assembling the predicted answers into a predefined report template. By prompting the VQA model to explicitly answer key pathological slots–such as organ, histologic type, and invasiveness–this approach ensures clearer concept grounding, reduces hallucinations, and facilitates the generation of clinically meaningful expressions.

Furthermore, the REG\_Path team adopted a two-stage training strategy, consisting of VQA pretraining followed by report fine-tuning. This progressive incorporation of slotlevel supervision signals enhanced both the accuracy and consistency of the final reports.

## 5. Discussion

The quantitative evaluation results also clearly revealed the limitations of conventional natural language generation metrics for pathology report generation. In this challenge, the overall ranking score was constructed by combining ROUGE, BLEU, KEY, and EMB metrics, with the KEY and EMB metrics demonstrating particularly strong consistency with the final ranking. In contrast, ROUGE and BLEU exhibited anomalously high or low values for certain teams, indicating that these metrics do not always align with clinical correctness. For example, some models achieved high ROUGE scores despite failing to generate critical diagnostic terms accurately, while others obtained relatively low BLEU scores but correctly preserved clinically important diagnostic attributes, such as Gleason score, nuclear grade, and necrosis status, as demonstrated in the qualitative cases described in Section 3.2.1. These findings suggest that lexical overlap-based metrics alone may not adequately capture the clinical accuracy required for pathology report generation.

This discrepancy between metric performance and clinical correctness arises because pathology reports, unlike general captioning tasks, require structured diagnostic reasoning and precise representation of critical concepts rather than superficial sentence similarity. Key diagnostic attributes– such as Gleason score, tumor proportion, and histologic subtype–play a decisive role in determining the clinical interpretation of the report. However, conventional metrics do not suficiently evaluate concept-level correctness. While the KEY score used in this study partially mitigates this limitation, more precise evaluation will require the development of pathology-specific metrics, such as ontologyaware evaluation, slot-level accuracy metrics, or clinically grounded concept consistency measures.

Furthermore, both qualitative and quantitative analyses revealed recurring issues of numeric hallucination and diagnostic misclassification, reflecting fundamental limitations of current VLM-based report generation frameworks. In particular, for structured diagnostic attributes such as tumor proportion, grading score, and subtype classification, models tend to generate text based on semantic plausibility rather than grounded evidence. This limitation stems from the probabilistic next-token prediction nature of language models. These findings indicate that multimodal alignment alone is insuficient to address this issue and highlight the need for architectures that incorporate explicit diagnostic reasoning processes.

In this context, reasoning-augmented generation frameworks represent a promising direction for pathology report generation. Approaches such as slot-based intermediate prediction [74, 76, 14], chain-of-thought style structured generation [72, 70, 83, 81], and VQA-based diagnostic decomposition [3, 31] enable models to perform complex diagnostic reasoning in a stepwise and interpretable manner. Notably, in this challenge, approaches incorporating VQAbased frameworks demonstrated reduced hallucination and improved slot-level concept grounding. This suggests that structured, reasoning-based generation frameworks are more suitable for pathology report generation than conventional end-to-end captioning approaches.

In addition, Retrieval-Augmented Generation (RAG) and hierarchical expert-based architectures were found to contribute to reducing hallucination and improving semantic grounding. By leveraging external evidence or structured expert knowledge rather than relying solely on internal model representations, these approaches enhance diagnostic reliability. This highlights that reasoning-aware multimodal architectures and external knowledge integration should be considered key design components in future pathology foundation models.

Overall, the results of this challenge demonstrate that pathology report generation is not merely an image-to-text translation task, but rather a high-dimensional multimodal reasoning problem involving structured diagnostic reasoning, quantitative attribute estimation, and uncertainty-aware decision-making. Accordingly, future research should focus on (1) developing clinically grounded evaluation metrics, (2) designing reasoning-integrated generation frameworks, and (3) incorporating structured pathology knowledge to improve reliability and clinical applicability.

## 6. Conclusion

In this study, through the REG 2025 Challenge, we constructed and publicly released the first large-scale Pan-Asia multi-institutional pathology dataset comprising approximately 10,500 WSI–report pairs. Based on this dataset, we systematically analyzed and compared the models submitted by participating teams for pathology report generation.

Analysis across diverse participant approaches demonstrated that Vision–Language Model (VLM) based frameworks can generate clinically coherent pathology reports across a wide range of organs and pathological categories. Furthermore, the models maintained stable performance on an independent European cohort, highlighting their crossdomain generalization capability. In particular, approaches that leveraged structured representations of pathology reports and multimodal grounding strategies achieved superior performance, underscoring the importance of structured multimodal modeling in pathology report generation.

Importantly, the REG Challenge frames pathology report generation not merely as an image captioning task, but as a complex multimodal diagnostic understanding problem, and provides a comprehensive benchmark for its systematic evaluation. Through comparative analysis of diverse modeling strategies, we identify multimodal alignment, structured report representation, and clinically meaningful feature integration as key factors influencing report generation performance. As the first large-scale benchmark enabling systematic evaluation of pathology report generation and multimodal diagnostic reasoning, the REG Challenge provides an essential foundation for future foundation-level computational pathology research.

## 7. Acknowledgement

This work was partly supported by the National Research Foundation of Korea (NRF) grants funded by the Korea government (MSIT) (RS-2026-25475860, RS-2025-00520578, RS-2025-02215813, and RS-2025-02634603); by the NRF grant funded by the Korea government (MSIT and MOE) (RS-2025-16063688); by the Digital-Bio AI+X Global Innovative Talent Nurturing Project through the NRF funded by the Korea government (MSIT) (RS-2024-00441029); by a Korea University Grant (K2612911); and by funding from Interreg Meuse-Rhine (NL-BE-DE) EU / EFRE North Rhine-Westphalia (Project DigiPathConnect) and the Wilhelm Sander Foundation (Munich, Germany) (Project 2022.040.1); and by the AI Computing Infrastructure Enhancement (GPU Rental Support) User Support Program of the Ministry of Science and ICT (MSIT), Republic of Korea (No. RQT-25-090048).

## CRediT authorship contribution statement

Yumi Lee: Data curation, Formal analysis, Investigation, Project administration, Writing - original draft, Writing - review & editing. Harim Oh: Resources, Data curation, Investigation, Writing - original draft. Hyoryung Kim: Data curation, Writing - review. Minji Kim: Data curation, Writing - review. Eunsu Kim: Data curation. Hyeseong Lee: Data curation. Junya Fukuoka: Resources. Andrey Bychkov: Resources. Jijgee Munkhdelger: Resources. Rajiv Kumar Kaushal: Resources. Ayushi Sahay: Resources. Rajni Yadav: Resources. Bharathi Prabakaran: Resources. Sulen Sarioglu: Resources. Serdar Balcı: Resources. Ilknur Turkmen: Resources. Yuri Tolkach: Resources. Christian Harder: Resources. Julian Westerdorf: Resources. Reinhard Buettner: Resources. Audun Ljone Henriksen: Investigation. Sepp De Raedt: Investigation. Byung Hyun Lee: Investigation. Sungjin Lim: Investigation. Joohoon Lee: Investigation. Gwanghyun Kim: Investigation. Se Young Chun: Investigation. Suryakant Singh: Investigation. Saarthak Kapse: Investigation. Prateek Prasanna: Investigation. Kyung A Kim: Investigation. Yousun Kang: Investigation. Sehwan Yoo: Investigation. Sungman Hong: Investigation. Shubham Innani: Investigation. Michael Feldman: Investigation. Spyridon Bakas: Investigation. Ujjwal Baid: Investigation. Prasad Dutande: Investigation. Suhas Gajare: Investigation. Bhakti Baheti: Investigation. Serkan Sökmen: Investigation. Ece Tuğba Cebeci: Investigation. Ahmet Halıcı: Investigation. Musa Balcı: Investigation. Kardelen Peçenek: Investigation. Srividhya Sainath: Investigation. Kyongseok Jang: Investigation. Messi H.J. Lee: Investigation. Noorul Wahab: Investigation. Bodong Du: Investigation. Jiaming Zhang: Investigation. Qixiang Zhang: Investigation. Jang-Hwan Choi: Conceptualization, Supervision, Writing - review & editing. Sangjeong Ahn: Conceptualization, Supervision, Writing - review & editing.

## References

[1] Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al., 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, 23716–23736.

[2] Ankit Pal, M.S., 2024. Openbiollms: Advancing open-source large language models for healthcare and life sciences. https:// huggingface.co/aaditya/OpenBioLLM-Llama3-70B.

[3] Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D., 2015. Vqa: Visual question answering, in: Proceedings of the IEEE international conference on computer vision, pp. 2425– 2433.

[4] Bilal, M., Jewsbury, R., Wang, R., AlGhamdi, H.M., Asif, A., Eastwood, M., Rajpoot, N., 2023. An aggregation of aggregation methods in computational pathology. Medical image analysis 88, 102885.

[5] Bracamonte, E., Gibson, B.A., Klein, R., Krupinski, E.A., Weinstein, R.S., 2016. Communicating uncertainty in surgical pathology reports: a survey of staf physicians and residents at an academic medical center. Academic Pathology 3, 2374289516659079.

[6] Bulten, W., Kartasalo, K., Chen, P.H.C., Ström, P., Pinckaers, H., Nagpal, K., Cai, Y., Steiner, D.F., Van Boven, H., Vink, R., et al., 2022. Artificial intelligence for diagnosis and gleason grading of prostate cancer: the panda challenge. Nature medicine 28, 154–163.

[7] Cai, L., Huang, S., Zhang, Y., Lu, J., Zhang, Y., 2025. Attrimil: Revisiting attention-based multiple instance learning for whole-slide pathological image classification from a perspective of instance attributes. Medical Image Analysis 103, 103631.

[8] Chang, H., Zhou, Y., Borowsky, A., Barner, K., Spellman, P., Parvin, B., 2015. Stacked predictive sparse decomposition for classification of histology sections. International journal of computer vision 113, 3–18.

[9] Chen, P., Li, H., Zhu, C., Zheng, S., Shui, Z., Yang, L., 2024a. Wsicaption: Multiple instance generation of pathology reports for gigapixel whole-slide images, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 546– 556.

[10] Chen, P., Zhu, C., Zheng, S., Li, H., Yang, L., 2024b. Wsi-vqa: Interpreting whole slide images by generative visual question answering, in: European Conference on Computer Vision, Springer. pp. 401–417.

[11] Chen, R.J., Chen, C., Li, Y., Chen, T.Y., Trister, A.D., Krishnan, R.G., Mahmood, F., 2022. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16144–16155.

[12] Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Song, A.H., Chen, B., Zhang, A., Shao, D., Shaban, M., et al., 2024c. Towards a general-purpose foundation model for computational pathology. Nature medicine 30, 850–862.

[13] Chen, Y., Wang, G., Ji, Y., Li, Y., Ye, J., Li, T., Hu, M., Yu, R., Qiao, Y., He, J., 2025. Slidechat: A large vision-language assistant for whole-slide pathology image understanding, in: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5134– 5143.

[14] Chi, D., Kim, H., Oh, Y., Kim, Y., Lee, D., Jo, D., Kim, J., Baek, J., Ahn, S., Kim, S., 2025. Slot-mllm: Object-centric visual tokenization for multimodal llm. arXiv preprint arXiv:2505.17726 .

[15] Cho, S.Y., Park, S.Y., Bae, Y.K., Kim, J.Y., Kim, E.K., Kim, W.G., Kwon, Y., Lee, A., Lee, H.J., Lee, J.S., et al., 2021. Standardized pathology report for breast cancer. Journal of Pathology and Translational Medicine 55, 1–15.

[16] Christinat, Y., Hamelin, B., Alborelli, I., Angelino, P., Barbié, V., Bisig, B., Dawson, H., Frattini, M., Grob, T., Jochum, W., et al., 2024. Reporting of somatic variants in clinical cancer care: recommendations of the swiss society of molecular pathology. Virchows Archiv 485, 1033–1039.

[17] Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al., 2024. Scaling instruction-finetuned language models. Journal of Machine Learning

Research 25, 1–53.

[18] College of American Pathologists, 2018. Definition of synoptic reporting (version 4.0). URL: https://documents.cap.org/documents/ synoptic\_reporting\_definition\_examples\_v4.0.pdf.

[19] Cruz-Roa, A., Basavanhally, A., González, F., Gilmore, H., Feldman, M., Ganesan, S., Shih, N., Tomaszewski, J., Madabhushi, A., 2014. Automatic detection of invasive ductal carcinoma in whole slide images with convolutional neural networks, in: Medical imaging 2014: Digital pathology, SPIE. p. 904103.

[20] Ding, T., Wagner, S.J., Song, A.H., Chen, R.J., Lu, M.Y., Zhang, A., Vaidya, A.J., Jaume, G., Shaban, M., Kim, A., et al., 2025. A multimodal whole-slide foundation model for pathology. Nature medicine , 1–13.

[21] Dörrich, M., Balk, M., Heusinger, T., Beyer, S., Mirbagheri, H., Fischer, D.J., Kanso, H., Matek, C., Hartmann, A., Iro, H., et al., 2025. A multimodal dataset for precision oncology in head and neck cancer. Nature Communications 16, 7163.

[22] Guo, Z., Ma, J., Xu, Y., Wang, Y., Wang, L., Chen, H., 2024. Histgen: Histopathology report generation via local-global feature encoding and cross-modal context interaction, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 189–199.

[23] Hosseini, M.S., Bejnordi, B.E., Trinh, V.Q.H., Chan, L., Hasan, D., Li, X., Yang, S., Kim, T., Zhang, H., Wu, T., et al., 2024. Computational pathology: a survey review and the way forward. Journal of Pathology Informatics 15, 100357.

[24] Hou, L., Samaras, D., Kurc, T.M., Gao, Y., Davis, J.E., Saltz, J.H., 2016. Patch-based convolutional neural network for whole slide tissue image classification, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2424–2433.

[25] Hu, D., Jiang, Z., Shi, J., Xie, F., Wu, K., Tang, K., Cao, M., Huai, J., Zheng, Y., 2025. Pathology report generation from whole slide images with knowledge retrieval and multi-level regional feature selection. Computer Methods and Programs in Biomedicine 263, 108677.

[26] Ilse, M., Tomczak, J., Welling, M., 2018. Attention-based deep multiple instance learning, in: International conference on machine learning, PMLR. pp. 2127–2136.

[27] Jang, W., Lee, J., Park, K.H., Kim, A., Lee, S.H., Ahn, S., 2024. Molecular classification of breast cancer using weakly supervised learning. Cancer Research and Treatment: Oficial Journal of Korean Cancer Association 57, 116–125.

[28] Kalra, S., Li, L., Tizhoosh, H.R., 2019. Automatic classification of pathology reports using tf-idf features. arXiv preprint arXiv:1903.07406 .

[29] Kefeli, J., Tatonetti, N., 2024. Tcga-reports: A machine-readable pathology report resource for benchmarking text-based ai models. Patterns 5.

[30] Kemper, B., Matsuzaki, T., Matsuoka, Y., Tsuruoka, Y., Kitano, H., Ananiadou, S., Tsujii, J., 2010. Pathtext: a text mining integrator for biological pathway visualizations. Bioinformatics 26, i374–i381.

[31] Khan, Z., BG, V.K., Schulter, S., Chandraker, M., Fu, Y., 2023. Exploring question decomposition for zero-shot vqa. Advances in Neural Information Processing Systems 36, 56615–56627.

[32] Lam, H., Nguyen, F., Wang, X., Stock, A., Lenskaya, V., Kooshesh, M., Li, P., Qazi, M., Wang, S., Dehghan, M., et al., 2022. An accessible, eficient, and accurate natural language processing method for extracting diagnostic data from pathology reports. Journal of Pathology Informatics 13, 100154.

[33] Li, B., Li, Y., Eliceiri, K.W., 2021. Dual-stream multiple instance learning network for whole slide image classification with selfsupervised contrastive learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14318– 14328.

[34] Li, H., Chen, Y., Chen, Y., Yu, R., Yang, W., Wang, L., Ding, B., Han, Y., 2024. Generalizable whole slide image classification with finegrained visual-semantic interaction, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11398– 11407.

[35] Li, Z., Wu, X., Du, H., Liu, F., Nghiem, H., Shi, G., 2025a. A survey of state of the art large vision language models: Alignment, benchmark, evaluations and challenges. arXiv preprint arXiv:2501.02189 .

[36] Li, Z., Wu, X., Du, H., Liu, F., Nghiem, H., Shi, G., 2025b. A survey of state of the art large vision language models: Benchmark evaluations and challenges, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 1587–1606.

[37] Liang, Y., Lyu, X., Chen, W., Ding, M., Zhang, J., He, X., Wu, S., Xing, X., Yang, S., Wang, X., et al., 2025. Wsi-llava: A multimodal large language model for whole slide image, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22718– 22727.

[38] Lin, C.Y., 2004. Rouge: A package for automatic evaluation of summaries, in: Text summarization branches out, pp. 74–81.

[39] Litjens, G., Bandi, P., Ehteshami Bejnordi, B., Geessink, O., Balkenhol, M., Bult, P., Halilovic, A., Hermsen, M., Van de Loo, R., Vogels, R., et al., 2018. 1399 h&e-stained sentinel lymph node sections of breast cancer patients: the camelyon dataset. GigaScience 7, giy065.

[40] Liu, F., Jiang, S., Cai, L., Wang, Z., Zhang, Y., 2026a. Pathflip: Fine-grained language-image pretraining for versatile computational pathology, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 7132–7140.

[41] Liu, L., Blake, V., Barman, M., Gallego, B., Churches, T., Kennedy, G., Ooi, S.Y., Delaney, G.P., Jorm, L., 2026b. Using natural language processing to extract information from clinical text in electronic medical records for populating clinical registries: a systematic review. Journal of the American Medical Informatics Association 33, 484– 499.

[42] Liu, S.W., Fan, H.Y., Chu, W.T., Yang, F.E., Wang, Y.C.F., 2025. Histopathology image report generation by vision language model with multimodal in-context learning. arXiv preprint arXiv:2506.17645 .

[43] Lu, M.Y., Chen, B., Williamson, D.F., Chen, R.J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L.P., Gerber, G., et al., 2024. A visuallanguage foundation model for computational pathology. Nature medicine 30, 863–874.

[44] Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F., 2021. Data-eficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering 5, 555–570.

[45] Luo, R., Sun, L., Xia, Y., Qin, T., Zhang, S., Poon, H., Liu, T.Y., 2022. Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics 23, bbac409.

[46] Maier-Hein, L., Reinke, A., Kozubek, M., Martel, A.L., Arbel, T., Eisenmann, M., et al., 2020. Bias: Transparent reporting of biomedical image analysis challenges. Medical Image Analysis 66, 101796.

[47] Mao, J., Xu, J., Tang, X., Liu, Y., Zhao, H., Tian, G., Yang, J., 2025. Camil: channel attention-based multiple instance learning for whole slide image classification. Bioinformatics 41, btaf024.

[48] Mossanen, M., True, L.D., Wright, J.L., Vakar-Lopez, F., Lavallee, D., Gore, J.L., 2014. Surgical pathology and the patient: a systematic review evaluating the primary audience of pathology reports. Human Pathology 45, 2192–2201.

[49] Myles, C., Um, I.H., Marshall, C., Harris-Birtill, D., Harrison, D.J., 2025. Surgen: 1020 h&e-stained whole-slide images with survival and genetic markers. GigaScience 14, giaf086.

[50] Nechaev, D., Pchelnikov, A., Ivanova, E., 2025. Histai: an opensource, large-scale whole slide image dataset for computational pathology. arXiv preprint arXiv:2505.12120 .

[51] Ng, G.T.E., Phang, S.C., Yu, K.S., Tiwari, L., Khurram, S.A., Sloan, P., Kujan, O., 2025. Understanding interobserver variability of pathologists to improve oral epithelial dysplasia grading. Oral Diseases 31, 838–845.

[52] Papineni, K., Roukos, S., Ward, T., Zhu, W.J., 2002. Bleu: a method for automatic evaluation of machine translation, in: Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318.

[53] Robert, M.E., Rüschof, J., Jasani, B., Graham, R.P., Badve, S.S., Rodriguez-Justo, M., Kodach, L.L., Srivastava, A., Wang, H.L., Tang, L.H., et al., 2023. High interobserver variability among pathologists using combined positive score to evaluate pd-l1 expression in gastric, gastroesophageal junction, and esophageal adenocarcinoma. Modern Pathology 36, 100154.

[54] Santos, T., Tariq, A., Gichoya, J.W., Trivedi, H., Banerjee, I., 2022. Automatic classification of cancer pathology reports: a systematic review. Journal of Pathology Informatics 13, 100003.

[55] Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., Ji, X., et al., 2021. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. Advances in neural information processing systems 34, 2136–2147.

[56] Simpson, R.W., Berman, M.A., Foulis, P.R., Divaris, D.X., Birdsong, G.G., Mirza, J., Moldwin, R., Spencer, S., Srigley, J.R., Fitzgibbons, P.L., 2015. Cancer biomarkers: the role of structured data reporting. Archives of Pathology & Laboratory Medicine 139, 587–593.

[57] Sloan, P., Clatworthy, P., Simpson, E., Mirmehdi, M., 2024. Automated radiology report generation: A review of recent advances. IEEE Reviews in Biomedical Engineering 18, 368–387.

[58] Sluijter, C.E., van Lonkhuijzen, L.R., van Slooten, H.J., Nagtegaal, I.D., Overbeek, L.I., 2016. The efects of implementing synoptic pathology reporting in cancer diagnosis: a systematic review. Virchows Archiv 468, 639–649.

[59] Song, A.H., Jaume, G., Williamson, D.F., Lu, M.Y., Vaidya, A., Miller, T.R., Mahmood, F., 2023. Artificial intelligence for digital and computational pathology. Nature Reviews Bioengineering 1, 930– 949.

[60] Steimetz, E., Mostafidi, E., Castagna, C., Gupta, R., Frasso, R., 2024. Forgotten clientele: a systematic review of patient-centered pathology reports. Plos one 19, e0301116.

[61] Steiner, D.F., MacDonald, R., Liu, Y., Truszkowski, P., Hipp, J.D., Gammage, C., Thng, F., Peng, L., Stumpe, M.C., 2018. Impact ofdeep learning assistance on the histopathologic review of lymph nodes for metastatic breast cancer. The American journal of surgical pathology 42, 1636–1646.

[62] Tan, J.W., Kim, S., Kim, E., Lee, S.H., Ahn, S., Jeong, W.K., 2024. Clinical-grade multi-organ pathology report generation for multiscale whole slide images via a semantically guided medical text foundation model, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 25–35.

[63] Tran, M., Schmidle, P., Guo, R.R., Wagner, S.J., Koch, V., Lupperger, V., Novotny, B., Murphree, D.H., Hardway, H.D., D’Amato, M., et al., 2025. Generating dermatopathology reports from gigapixel whole slide images with histogpt. Nature communications 16, 4886.

[64] Tripathi, A., Waqas, A., Venkatesan, K., Ullah, E., Khan, A., Khalil, F., Chen, W.S., Ozturk, Z.G., Saeed-Vafa, D., Bui, M.M., et al., 2025. Using consensus-based reasoning and large language models to extract structured data from surgical pathology reports. Laboratory Investigation , 104272.

[65] Valverde L, D., Reznichek, R.C., Torres S, M., 2022. Establishing synoptic cancer pathology reporting in low-and middle-income countries: a nicaraguan experience. JCO Global Oncology 8, e1900343.

[66] Vorontsov, E., Bozkurt, A., Casson, A., Shaikovski, G., Zelechowski, M., Severson, K., Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., et al., 2024. A foundation model for clinical-grade computational pathology and rare cancers detection. Nature medicine 30, 2924– 2935.

[67] Wahab, N., Rajpoot, N., 2025. Mpath: Multimodal pathology report generation from whole slide images. arXiv preprint arXiv:2512.11906 .

[68] Wang, R., Ba, W., Zhou, Y., Li, Y., Liu, B., Wang, B., Wang, Y., Yang, Z., Zhang, K., Yan, R., et al., 2026. Qcagent: An agentic framework for quality-controllable pathology report generation from whole slide image. arXiv preprint arXiv:2603.01647 .

[69] Wang, X., Figueredo, G., Li, R., Zhang, W.E., Chen, W., Chen, X., 2025. A survey of deep-learning-based radiology report generation using multimodal inputs. Medical Image Analysis 103, 103627.

[70] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D., 2022a. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 .

[71] Wang, X., Yang, S., Zhang, J., Wang, M., Zhang, J., Yang, W., Huang, J., Han, X., 2022b. Transformer-based unsupervised contrastive learning for histopathological image classification. Medical image analysis 81, 102559.

[72] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al., 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837.

[73] Weinstein, J.N., Collisson, E.A., Mills, G.B., Shaw, K.R., Ozenberger, B.A., Ellrott, K., Shmulevich, I., Sander, C., Stuart, J.M., 2013. The cancer genome atlas pan-cancer analysis project. Nature genetics 45, 1113-1120.

[74] Wu, Z., Hu, J., Lu, W., Gilitschenski, I., Garg, A., 2023. Slotdifusion: Object-centric generative modeling with difusion models. Advances in Neural Information Processing Systems 36, 50932–50958.

[75] Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., González, J., Gu, Y., et al., 2024a. A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181–188.

[76] Xu, J., Lan, C., Xie, W., Chen, X., Lu, Y., 2024b. Slot-vlm: Objectevent slots for video-language modeling. Advances in Neural Information Processing Systems 37, 632–659.

[77] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al., 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 .

[78] Yuan, H., Yuan, Z., Gan, R., Zhang, J., Xie, Y., Yu, S., 2022. Biobart: Pretraining and evaluation of a biomedical generative language model. arXiv preprint arXiv:2204.03905 .

[79] Yuan, R., Zhang, Z., Wang, A., Hu, L., Hua, X., Peng, Y., Luo, J., Yang, G., 2026. Hipath: Hierarchical vision-language alignment for structured pathology report prediction. arXiv preprint arXiv:2603.19957 .

[80] Zhang, A., Jaume, G., Vaidya, A., Ding, T., Mahmood, F., 2025. Accelerating data processing and benchmarking of ai models for pathology. arXiv preprint arXiv:2502.06750 .

[81] Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., Smola, A., 2023. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923 .

[82] Zhao, W., Guo, Z., Fan, Y., Jiang, Y., Yeung, M.C., Yu, L., 2024. Aligning knowledge concepts to whole slide images for precise histopathology image analysis. npj Digital Medicine 7, 383.

[83] Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al., 2022. Least-tomost prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625 .

[84] Zhou, Y., Chang, H., Barner, K., Spellman, P., Parvin, B., 2014. Classification of histology sections via multispectral convolutional sparse coding, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3081–3088.

Table S1  
REG 2025 Challenge (Phase 1): Quantitative performance comparison. Evaluation of participating teams across report generation quality metrics.
<table><tr><td>Rank</td><td>Team</td><td>II Phase 1</td><td>ROUGE</td><td>BLEU</td><td>KEY score</td><td>EMB score</td></tr><tr><td>1</td><td>ICGI</td><td>0.8098</td><td>0.8188</td><td>0.6372</td><td>0.7514</td><td>0.9696</td></tr><tr><td>5</td><td>IUCompPath</td><td>0.7906</td><td>0.7954</td><td>0.6173</td><td>0.7262</td><td>0.9607</td></tr><tr><td>7</td><td>TeamTiger@REG2025</td><td>0.7777</td><td>0.7805</td><td>0.5784</td><td>0.7116</td><td>0.9642</td></tr><tr><td>4</td><td>ADCT</td><td>0.7704</td><td>0.7753</td><td>0.5570</td><td>0.7097</td><td>0.9554</td></tr><tr><td>2</td><td>ICL_PathReport</td><td>0.7352</td><td>0.7291</td><td>0.3644</td><td>0.7049</td><td>0.9640</td></tr><tr><td>13</td><td>TokenStreamers</td><td>0.7248</td><td>0.7211</td><td>0.4860</td><td>0.6468</td><td>0.9501</td></tr><tr><td>12</td><td>REG Path</td><td>0.6907</td><td>0.6626</td><td>0.4458</td><td>0.6114</td><td>0.9331</td></tr><tr><td>6</td><td>TrustPath</td><td>0.6565</td><td>0.6415</td><td>0.4884</td><td>0.5842</td><td>0.8445</td></tr><tr><td>3</td><td>IMAGINE Lab</td><td>0.6258</td><td>0.7485</td><td>0.0895</td><td>0.5620</td><td>0.9177</td></tr><tr><td>10</td><td>MTS_REG_2025</td><td>0.6046</td><td>0.5749</td><td>0.3497</td><td>0.4802</td><td>0.9128</td></tr><tr><td>15</td><td>Clinical Omics Institute</td><td>0.5531</td><td>0.5109</td><td>0.2535</td><td>0.4275</td><td>0.8916</td></tr><tr><td>8</td><td>MedInsight-ViseurAI</td><td>0.5326</td><td>0.4754</td><td>0.2254</td><td>0.4092</td><td>0.8792</td></tr><tr><td>23</td><td>May_REG</td><td>0.5245</td><td>0.4679</td><td>0.2126</td><td>0.3924</td><td>0.8851</td></tr><tr><td>14</td><td>NUST-SEECS</td><td>0.5144</td><td>0.4810</td><td>0.2394</td><td>0.3555</td><td>0.8806</td></tr><tr><td>16</td><td>JX REG</td><td>0.5073</td><td>0.4470</td><td>0.1929</td><td>0.3676</td><td>0.8810</td></tr><tr><td>24</td><td>IMI_REG</td><td>0.5027</td><td>0.5591</td><td>0.0377</td><td>0.3489</td><td>0.9122</td></tr><tr><td>9</td><td>PathoMozhi</td><td>0.4611</td><td>0.4877</td><td>0.1155</td><td>0.2868</td><td>0.8530</td></tr><tr><td>20</td><td>Shubham47</td><td>0.4402</td><td>0.3881</td><td>0.1453</td><td>0.2715</td><td>0.8387</td></tr><tr><td>17</td><td>DSG</td><td>0.4385</td><td>0.3617</td><td>0.1600</td><td>0.3108</td><td>0.7862</td></tr><tr><td>19</td><td>virasoft</td><td>0.1691</td><td>0.0660</td><td>0.0257</td><td>0.0453</td><td>0.4573</td></tr></table>

Table S2

REG 2025 Challenge (Phase 2): Quantitative performance comparison. Evaluation of participating teams across report generation quality metrics.
<table><tr><td>Rank</td><td>Team</td><td>ⅡI Phase 2</td><td>ROUGE</td><td>BLEU</td><td>KEY score</td><td>EMB score</td></tr><tr><td>3</td><td>IMAGINE Lab</td><td>0.8494</td><td>0.8572</td><td>0.6928</td><td>0.8094</td><td>0.9770</td></tr><tr><td>1</td><td>ICGI</td><td>0.8472</td><td>0.8564</td><td>0.6852</td><td>0.8080</td><td>0.9759</td></tr><tr><td>2</td><td>ICL_PathReport</td><td>0.8415</td><td>0.8578</td><td>0.6780</td><td>0.7937</td><td>0.9788</td></tr><tr><td>11</td><td>NW-TIA</td><td>0.8282</td><td>0.8362</td><td>0.6467</td><td>0.7850</td><td>0.9726</td></tr><tr><td>6</td><td>TrustPath</td><td>0.8127</td><td>0.8258</td><td>0.6340</td><td>0.7597</td><td>0.9661</td></tr><tr><td>8</td><td>MedInsight-ViseurAl</td><td>0.8093</td><td>0.8159</td><td>0.6148</td><td>0.7604</td><td>0.9683</td></tr><tr><td>4</td><td>ADCT</td><td>0.8040</td><td>0.8037</td><td>0.6154</td><td>0.7507</td><td>0.9697</td></tr><tr><td>9</td><td>PathoMozhi</td><td>0.7960</td><td>0.8574</td><td>0.6821</td><td>0.6792</td><td>0.9781</td></tr><tr><td>5</td><td>IUCompPath</td><td>0.7899</td><td>0.7887</td><td>0.6141</td><td>0.7297</td><td>0.9585</td></tr><tr><td>7</td><td>TeamTiger@REG2025</td><td>0.7763</td><td>0.7790</td><td>0.5773</td><td>0.7119</td><td>0.9604</td></tr><tr><td>10</td><td>MTS_REG_2025</td><td>0.7584</td><td>0.7607</td><td>0.5436</td><td>0.6923</td><td>0.9528</td></tr><tr><td>12</td><td>REG Path</td><td>0.6510</td><td>0.6223</td><td>0.3959</td><td>0.5572</td><td>0.9179</td></tr><tr><td>14</td><td>NUST-SEECS</td><td>0.6121</td><td>0.6439</td><td>0.4328</td><td>0.4403</td><td>0.9151</td></tr><tr><td>15</td><td>Clinical_Omics_Institute</td><td>0.5984</td><td>0.5583</td><td>0.3300</td><td>0.4921</td><td>0.8944</td></tr><tr><td>13</td><td>TokenStreamers</td><td>0.5773</td><td>0.5277</td><td>0.2863</td><td>0.4718</td><td>0.8882</td></tr><tr><td>16</td><td>JX_REG</td><td>0.5309</td><td>0.4709</td><td>0.2133</td><td>0.4089</td><td>0.8825</td></tr><tr><td>18</td><td>sk</td><td>0.5277</td><td>0.5371</td><td>0.1519</td><td>0.4015</td><td>0.8790</td></tr><tr><td>19</td><td>virasoft</td><td>0.4590</td><td>0.3906</td><td>0.1887</td><td>0.3039</td><td>0.8350</td></tr><tr><td>17</td><td>DSG</td><td>0.4286</td><td>0.3486</td><td>0.1412</td><td>0.3045</td><td>0.7780</td></tr><tr><td>20</td><td>Shubham47</td><td>0.3849</td><td>0.3283</td><td>0.1137</td><td>0.1692</td><td>0.8364</td></tr><tr><td>21</td><td>DiceMed_labs</td><td>0.3577</td><td>0.2731</td><td>0.0000</td><td>0.2003</td><td>0.7887</td></tr><tr><td>22</td><td>UTWL</td><td>0.3110</td><td>0.1772</td><td>0.0204</td><td>0.0894</td><td>0.8185</td></tr></table>

## A. Contributions and Methods of Each Team

## A.1. ICGI (Fig. S2-(a))

Contributions The ICGI team’s primary contributions include the utilization of the H-optimus-1 foundation model to extract tile-level embeddings and the multiple-instance slide level embedding to predict the anatomical site and the procedure that has been taken. Also, they implemented a decoder for predicting slide descriptions, coupled with a post-processing mechanism to refine the output text.

Method The pipeline begins by extracting tile-level features using H-optimus-1, following color correction and tissue detection. They utilize an eight parallel attention heads within the ABMIL backbone, which concatenate weighted sums into a slide-level representation. This embedding is fed into three task-specific heads: two linear classifiers for site and procedure, and a Transformer decoder for text generation. During inference, an n-gram mask is derived from the training corpus to filter out unobserved token sequences.

## A.2. ICL\_PathReport (Fig. S2-(b))

Contributions The ICL\_PathReport team proposes a Tree-of-Experts (ToE) framework that hierarchically predicts report components, including organs, procedures, and histological types, through multiple lightweight MLP classifiers. To address the data scarcity challenge in deep diagnostic levels, they introduce a Noisy Feature Mixup strategy, which enhances model generalization and prediction calibration.

Method The system employs the Trident framework for patch extraction with Virchow for patch-level features, which are then aggregated into slide-level embedding using the PRISM Perceiver encoder. The ToE architecture employs a hierarchy of specialized MLP experts designed to predict detailed diagnostic attributes. During training, the team applies noise perturbation and weighted Mixup to the slide features to augment the limited dataset available for rare or complex diagnoses. The final structured report is synthesized by navigating the ToE’s forward path, ensuring that the generated output strictly adheres to the inherent hierarchy of clinical pathology reports.

An example illustrating the hierarchical tree structure of the pathology annotation format is shown below.

```jsonl
{
"id": "PIT_01_00002_01.tiff",
"organ": "Breast",
"procedure": "core-needle biopsy",
"sub_feature": [
{
"principal": "Invasive carcinoma of no special type",
"scores": "",
"sub_feature": [
{
"principal": "grade",
"scores": "I",
"sub_feature": []
},
{
"principal": "Tubule formation",
"scores": "2",
"sub_feature": []
},
{
"principal": "Nuclear grade",
"scores": "2",
"sub_feature": []
},
{
"principal": "Mitoses",
```

"scores": "1",   
"sub\_feature": []   
}   
]   
}   
]   
}

## A.3. IMAGINE Lab (Fig. S2-(c))

Contributions The IMAGINE Lab team proposes a framework that emphasizes concept-level interpretability. Their primary contribution is the integration of multiple pathology foundation models (CONCH, TITAN, and GECKO) to capture multi-scale domain information, ranging from patch-level morphology to slide-level concepts. Additionally, they implement a vision-concept contrastive learning strategy to align visual features with predefined clinical concepts, ensuring the model’s reasoning is grounded in expert-defined pathology.

Method The architecture is inspired by WsiCaption, employing Transformer-based encoder-decoder model with a CNN-based position-aware module to extract information. The pipeline begins by extracting embeddings from frozen CONCH, TITAN, and GECKO encoders, which are then projected into a common multimodal space via MLP adapters and mixers. They utilize GECKO-derived concept priors, where cosine similarity between visual and textual embeddings is used to pretrain a Multiple Instance Learning (MIL) aggregator. During inference, the team utilizes multiple iterations under perturbations and collates the results using maximum agreement logic to ensure the stability and clinical accuracy of the final report.

## A.4. ADCT (Fig. S3-(d))

Contributions The ADCT team introduces a framework for structured pathology report generation, emphasizing data quality and organ-specific diagnostic standardization. Their primary contribution is the development of comprehensive pathology codebooks that map visual features to structured clinical labels, such as tumor grade and invasion depth. They also implement an edge-based filtering to ensure the high quality of training patches.

Method The proposed method utilizes an attentionbased Multiple Instance Learning (ABMIL) framework for slide-level representation. For all organs except prostate, features are extracted using the CONCH encoder at 20× magnification, while a domain-specific HistoEncoder is employed at 10× for prostate-specific tasks. The architecture incorporates two distinct reporting strategies: a structured MIL-codebook matching approach for eficient outputs and a full fine-tuning of the Flan-T5 model to translate predicted labels into coherent reports.

## A.5. IUCompPath (Fig. S3-(e))

Contributions The IUCompPath team adpots the Hist-Gen architecture, which employs a Local-Global Hierarchical (LGH) encoder and a Cross-modal Context (CMC) module, and integrating the pre-trained UNI2 and H-Optimus-1 foundation model to extract slide-level features.

Method The pipeline starts with feature extraction via UNI2 and H-Optimus-1. The features are processed through the LGH encoder, which uses local and global Transformers to integrate fine-grained patch details with broad slide-level context. The CMC module facilitates interaction between visual and textual embeddings to ensure multimodal alignment. The final report is produced by a 3-layer Transformer decoder using beam search, based on the features from LGH and CMC.

## A.6. Team Tiger@REG2025 (Fig. S3-(f))

Contributions TeamTiger proposes a framework based on the WsiCaption architecture, where they replaced the feature extractor with the UNI2 foundation model. They also demonstrated the feasibility of using general-purpose pathology foundation models for generating reports on diverse Pan-Asia datasets.

Method The pipeline begins by extracting non-overlapping patches from WSIs and generating patch-level embeddings using the pretrained UNI2 model. These embeddings are processed through a Transformer encoder equipped with hierarchical Position-Aware Modules (PAMs) to capture spatial context. The final report is produced by a Transformer decoder.

## A.7. MedInsight - ViseurAI (Fig. S3-(g))

Contributions The MedInsight-ViseurAI team proposed a transformer-based pipeline for pathology report generation. They utilize the UNI encoder for WSI feature extraction and implement a post-processing stage using Sentence-BERT. This stage refines predictions based on their cosine similarity to ground-truth reports to enhance accuracy.

Method The selected patches using a fine-tuned UNI encoder are projected into memory vectors and fed into a 6- layer Transformer decoder aligned with a BioGPT language model. During the final inference stage, Sentence-BERT compares the generated predictions against training data reports to refine the results based on semantic similarity.

## A.8. PathoMozhi (Fig. S3-(h))

Contributions The PathoMozhi team introduces a Flaming style vision-language model tailored for pathology report generation. Their primary contribution is the adaptation of a gated cross-attention mechanism that enables a frozen language model to efectively ingest contextual visual features from WSIs. They also implement an multiple instance learning (MIL) strategy that utilizes a bag of 128 sampled patch embeddings per slide.

Method The framework utilizes a frozen BioGPT backbone, interfaced with a vision encoder through learnable gated cross-attention layers after each of the 48 transformer blocks. For visual input, patch-level features are first extracted using a CONCH encoder. However, instead of processing the entire WSI, the model randomly samples a bag of 128 instances for each slide during training. These sampled embeddings are then used as visual context tokens, where the gated cross attention mechanism modulates the flow of histological information into the language model.

## A.9. MTS\_REG\_2025 (Fig. S4-(j))

Contributions The MTS REG 2025 team introduces a Retrieval-Augmented Generation (RAG) approach to produce standardized pathology reports from WSIs. Their primary contributions include an adaptive retrieval mechanism that dynamically adjusts the number of reference cases based on similarity distances and a semantic similarity-based training strategy for the projection layer, which is trained via contrastive learning.

Method The system utilizes a pretrained Prov-GigaPath encoder to extract slide-level features, which are then transformed into a normalized embedding space via a trained projection head. It uses a “dynamic k-retrieval” process, where retrieval stops if the similarity gap between consecutive cases exceeds a specific threshold, preventing the inclusion of irrelevant data. These retrieved reports are fed into a QEN-3 8B reasoning model, which synthesizes the final report using a sequential majority-voting strategy to select organs, procedures, and histologic types. This approach ensures that the output strictly. adheres to established diagnostic conventions through a structured ensemble of the most similar historical cases.

## A.10. NW-TIA (Fig. S3-(i))

Contributions The NW-TIA team proposes a multitask, embedding-driven pipeline that leverages WSI-level embeddings for eficient pathology report generation without the need for direct pixel-level supervision. A key contribution is the introduction of auxiliary classification heads for organ type, sample type, and findings, which regularize the training process and enhance the clinical grounding of the generated reports.

Method The framework utilizes a frozen BioBART backbone conditioned on slide-level embeddings extracted via CONCH and Titan encoder. This team utilizes a multitask learning strategy; while the decoder generates text, specialized auxiliary MLP heads simultaneously predict the organ, specimen type, and 75 distinct findings from o-the shared WSI representation. These auxiliary tasks are weighted and backpropagated to force the visual projector to capture essential diagnostic features. During inference, the auxiliary heads are removed, leaving a streamlined generator informed by a clinically-enriched embedding space.

## A.11. REG\_Path (Fig. S4-(k))

Contributions The REG Path team proposes a thresholdadaptive preprocessing mechanism that dynamically adjusts segmentation thresholds based on individual slide statistical distributions to ensure more accurate feature extraction. They also introduce a two-stage fine-tuning framework to optimize the model for domain-specific grounding through VQA and clinical coherence in full report generation.

Method Built upon the SlideChat backbone, the model utilizes a pretrained CONCH encoder to extract patch-level features from WSIs using adatpively tuned segmenation thresholds. The training follows a two-stage pipeline: first, a domain alignment phase using 4.2K WSI-VQA pairs to establish visual-textual grounding, followed by an instruction tuning phase on 176K WSI-caption pairs for end-to-end report generation.

## B. Challenge Organization Details

In accordance with the BIAS (Biomedical Image Analysis Challenges) statement [46] reporting guidelines, we provide additional details regarding the challenge organization that were not fully described in the main manuscript.

## B.1. Challenge Timeline

The challenge was conducted according to the following timeline:

• Training data release: May 20, 2025

• Registration deadline: July 4, 2025

• Debug Phase opens: July 4, 2025

• Debug Phase deadline: July 10, 2025

• Test Phase 1 opens and Test Phase 1 data release: July 11, 2025

• Test Phase 1 deadline: August 1, 2025

• Test Phase 2 opens and Test Phase 2 data release: August 2, 2025

• Test Phase 2 deadline: August 23, 2025

• Announcement of winners: September 9, 2025

## B.2. Participation Policy

Multiple registrations from the same participant or institution using diferent accounts were not permitted. This policy was introduced to prevent participants from circumventing the submission limit of two submissions per phase. To ensure fairness and avoid potential conflicts of interest, members afiliated with the organizing institutions did not participate in the challenge.

## B.3. Pre-submission Evaluation

To balance fairness and accessibility, each participating team was allowed a maximum of two submissions per phase. Instead of providing unrestricted leaderboard access, the oficial evaluation code was publicly released, allowing participants to locally evaluate their methods prior to submission. The evaluation code is available at https://github.com/ hrb0/reg/tree/main/metric. Additionally, submissions that did not follow the prescribed JSON format specifications (e.g., missing case IDs or mismatched key names) were not evaluated.

## B.4. Award Policy

Awards and prize funding were provided to the top five teams achieving the highest performance in the challenge. To verify reproducibility, the selected teams were required to submit their source code for independent validation by the organizers. The submitted source code was used internally for reproducibility verification purposes only and was not publicly released. Final awards were granted only to teams that successfully passed the reproducibility verification process. Prize details were provided on the official challenge website (https://reg2025.grand-challenge. org/prizes/). The top five teams were additionally invited to present their methods during the Satellite Day 2 session at MICCAI 2025.

## B.5. Data Usage and License Policy

The REG2025 dataset is released under CC BY-NC-SA 4.0. The anonymized WSI–report dataset may be used, reproduced, and shared for non-commercial research, education, method development, benchmarking, and publication with appropriate attribution. Redistributed or adapted materials must follow the same license. Commercial use, re-identification attempts, and clinical use are prohibited, and data use remains subject to Grand Challenge terms and applicable ethics approvals. Dataset is available at https: //reg2025.grand-challenge.org/reg2025/.

## C. Expertise of Annotators

The annotation and review process involved pathologists with diverse levels of clinical expertise, including three pathologists with more than 20 years of experience, three with 10–20 years of experience, two with 5–10 years of experience, one with less than 5 years of experience, and four non-board-certified pathology trainees.

Table S4  
Table S3  
IMAGINE Lab: Representative histopathological concepts for qualitative analysis. Organ-specific morphological patterns curated by the IMAGINE Lab team to interpret model predictions across breast, lung, and prostate tissues.
<table><tr><td rowspan=1 colspan=1>Organ</td><td rowspan=1 colspan=1>Concepts</td></tr><tr><td rowspan=1 colspan=1>Breast</td><td rowspan=1 colspan=1>“Cribriform pattern&quot;: Tumor glands forming rounded or sieve-like punched-out spaces (cribriform architecture)&quot;Papillary structures&quot;: Epithelial fronds or projections supported by fibrovascular cores (papillary architecture)</td></tr><tr><td rowspan=1 colspan=1>Lung</td><td rowspan=1 colspan=1>&quot;Lepidic growth&quot;: Tumor cells proliferating along intact alveolar walls (lepidic-predominant pattern)“Keratin pearls&quot;: Concentric laminated keratinous material within nests of squamous carcinoma cells</td></tr><tr><td rowspan=1 colspan=1>Prostate</td><td rowspan=1 colspan=1>&quot;Cribriform glands”: Tumor glands forming large rounded sheets with multiple perforated or punched-out lumina&quot;Poorly formed glands&quot;: Small, irregular glandular spaces with scant luminal formation (Gleason pattern 4)</td></tr></table>

Figure S1: Overview of the dataset construction workflow. (a) Institution-specific free-text pathology reports are curated into structured and standardized templates with unified terminology. (b) Multi-institutional WSI acquisition and preprocessing pipeline, including de-identification, quality control, expert review, and exclusion of diagnostically equivocal cases. (c) Final dataset preparation, including multi-specimen slide separation and diagnosis-balanced splitting, with each WSI paired to a curated pathology report.  
![](images/a40d88175c2a78cdeaa7279ae2543d9bcf6b9cbc8cb172650622d875abe49ab7.jpg)

REG2025 challenge organizers. The challenge consisted of the organizing committee and data providing teams. Detailed information is available on the challenge website (https://reg2025.grand-challenge.org/organizers/).
<table><tr><td>Organizer</td><td>Affiliation</td><td>Country</td><td>Organizer</td><td>Affiliation</td><td>Country</td></tr><tr><td>Yumi Lee</td><td>Ewha Womans University</td><td>Korea</td><td>Sulen Sarioglu</td><td>Memorial Healthcare Group</td><td>Turkey</td></tr><tr><td>Harim Oh</td><td>Korea University Anam Hospital</td><td>Korea</td><td>Serdar Balci</td><td>Memorial Healthcare Group</td><td>Turkey</td></tr><tr><td>Junya Fukuoka</td><td>Kameda Medical Center</td><td>Japan</td><td>ilknur Türkmen</td><td>Memorial Healthcare Group</td><td>Turkey</td></tr><tr><td>Andrey Bychkov</td><td>Kameda Medical Center</td><td>Japan</td><td>Yuri Tolkach</td><td>University Hospital Cologne</td><td>Germany</td></tr><tr><td>Jijgee Munkhdelger</td><td>Kameda Medical Center</td><td>Japan</td><td>Christian Harder</td><td>University Hospital Cologne</td><td>Germany</td></tr><tr><td>Rajiv Kumar Kaushal</td><td>Tata Memorial Hospital</td><td>India</td><td>Julian Westerdorf</td><td>University Hospital Cologne</td><td>Germany</td></tr><tr><td>Ayushi Sahay</td><td>Tata Memorial Hospital</td><td>India</td><td>Jang-Hwan Choi</td><td>Ewha Womans University</td><td>Korea</td></tr><tr><td>Rajni Yadav</td><td>AlIMS Delhi</td><td>India</td><td>Sangjeong Ahn</td><td>Korea University Anam Hospital</td><td>Korea</td></tr></table>

Figure S2: Overview of model architectures from teams ICGI, ICL\_PathReport, and IMAGINELab. ICGI utilizes an ABMIL backbone with H-Optimus-1 for multi-task classification and report generation, while ICL\_PathReport features a "Tree-of-Experts" decoder for hierarchical diagnosis. IMAGINELab integrates CONCH and Titan encoders with MLP adapters and a concept-guided Transformer decoder.  
![](images/aa0630421decfaf1335b0cea53cf3cf33e6f83161b493c32379131df5f624bdb.jpg)  
(a) ICGI

![](images/c253ee173b984d9201cc0d8a927cfbe1597c253a4724b2befe50f35fc1efa3de.jpg)  
(b) ICL\_PathReport

![](images/fff4749ef92ca52c3af0c37e833989fe38c1f0650bc2c952a1a11f28554af3fe.jpg)  
(c) IMAGINELab

Figure S3: Overview of the model architectures from teams ADCT, IUCompPath, TeamTiger@REG2025, MedInsight-ViseurAI, PathoMozhi, and NW-TIA. ADCT utilizes a MIL-based processing pipeline with a FLAN-T5 encoder-decoder, while IUCompPath employs a local-global hierarchical encoder with a cross-modal context module. TeamTiger@REG2025 incorporates a positionaware module within a Transformer encoder-decoder, and MedInsight-ViseurAI implements patch selection via a memory vector integrated with BioGPT. The PathoMozhi architecture features 48 interleaved gated cross-attention layers for continuous visua feature injection into a BioGPT-Large backbone, and NW-TIA utilizes a prompt-based approach with auxiliary diagnostic heads for organ and finding prediction through BioBART.

![](images/5560f0c2607b0e1216dc0a08a6ced08f890807a41d638a08bf97040a9e9d2a1b.jpg)  
(d) ADCT

![](images/8175c6a2f23fe2dce508d11131fb16a488b28e69c343b86f6315c396f0720670.jpg)  
(e) IUCompPath

![](images/4a606a000a226ee2ba428142223f35829737ec807054c1d188d289cdfaa93cd3.jpg)  
(f) TeamTiger@REG2025

![](images/b0122f821f9d88ae79da23b0b6e0d7cdf7e107b29510a12a4afc00817128e66b.jpg)  
(g) MedInsight-ViseurAI

![](images/5be55c843ca1323d0ffce81278c0fd9a1e7e943d9c5b3499121eec9c16c35b8e.jpg)  
(h) PathoMozhi

![](images/51d51a04fe8b39a773a8f3d413542aa7e9519927117fb0d79a0ce6b24a7c22c4.jpg)  
(i) NW-TIA

Figure S4: Overview of architectures from teams MTS\_REG\_2025 and REG\_Path. MTS\_REG\_2025 employs a contrastive learning retrieval system with Prov-GigaPath and a dynamic �-retrieval mechanism for report synthesis via QWEN-3 8B. REG\_Path utilizes a two-stage pipeline for domain alignment and instruction tuning, leveraging the slideChat framework with CONCH encoders for generative report production.  
![](images/9cdd07e3fa172787b249088c924267f2b1bfdf5c3e0d649ecd96ae30d4cd78d5.jpg)  
(k) REG\_Path

Table S5  
Attribute-level fidelity in a breast core biopsy. Among the participating teams, ICGI showed full concordance with the ground truth by correctly generating the principal diagnosis of ductal carcinoma in situ (DCIS) and the key pathologic attributes, including type, nuclear grade, necrosis status, and microcalcification. ( =organ, =procedure, =correct, =incorrect)
<table><tr><td colspan="4">Whole Slide I</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Breast, sono-guided core biopsy; 1. Ductal carcinoma in situ - Type: Solid</td><td></td><td>ADCT (0.7115) Breast, core-needle biopsy; 1. Ductal carcinoma in situ - Type: Cribriform</td><td>ICGI (1.000) Breast, sono-guided core biopsy; 1. Ductal carcinoma in situ</td></tr><tr><td>- Nuclear grade: Intermediate - Necrosis: Absent 2. Microcalcification</td><td></td><td>- Nuclear grade: Intermediate - Necrosis: Present (Focal)</td><td>- Type: Solid - Nuclear grade: Intermediate - Necrosis: Absent</td></tr><tr><td>ICL_PathReport (0.5778)</td><td>IMAGINE_Lab (0.5493) breast, biopsy;</td><td>2. Microcalcification IUCompPath (0.2639) Lung, biopsy;</td><td>2. Microcalcification MTS_REG_2025 (0.2566) Lung, biopsy;</td></tr><tr><td>Breast, core-needle biopsy; Ductal carcinoma in situ - Type: Cribriform, Micropapillary - Nuclear grade: Low</td><td>ductal carcinoma in situ - type: cribriform cribriform high - necrosis:</td><td>Non-small cell carcinoma, favor adenocarcinoma</td><td>Adenocarcinoma</td></tr><tr><td>- Necrosis: Present MedInsight-ViseurAI (0.6456) Breast, sono-guided core biopsy;</td><td>present (comedo-type) REG_Path (0.4237) Breast, sono-guided core biopsy;</td><td>TeamTiger@REG2025 (0.2566) Lung, biopsy;</td><td>PathoMozhi (0.5248) Structured Summary:</td></tr><tr><td>Ductal carcinoma in situ - Type: Solid - Nuclear grade: High - Necrosis: Present (Comedo-type)</td><td>Papillary neoplasm</td><td>Adenocarcinoma</td><td>Organ - Breast, Diagnosis - Ductal carcinoma in situ - Type: Solid and cribriform - Nuclear grade: Intermediate - Necrosis: Present (Comedo-type)</td></tr></table>

Table S6

Representative whole-slide image and generated reports for a prostate core biopsy. (A) The red annotation marks the tumorinvolved portion of the biopsied core, corresponding to an estimated tumor volume of 50% of the total core length. (B) Blue indicates Gleason pattern 3 and orange indicates Gleason pattern 4. Pattern 3 is dominant and pattern 4 accounts for 30%, resulting in acinar adenocarcinoma, Gleason score 7 (3+4), grade group 2. Across teams, the generated reports were generally concordant for Gleason pattern/score, whereas tumor volume was frequently misestimated (often overcalled as 80–95%). ( =correct, =incorrect)

<table><tr><td rowspan=1 colspan=9>Whole Slide Image                                                   福A<img src="images/95869fc4610de03286bff7371ad78cd36e6a48a8948829667d48be103b26b404.jpg"/></td></tr><tr><td rowspan=1 colspan=5>Ground Truth</td><td rowspan=1 colspan=3>ADCT (0.9329)</td><td rowspan=1 colspan=1>ICGI (0.9814)</td></tr><tr><td rowspan=9 colspan=5>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score 7 (3+4),grade group 2 (Gleason pattern 4: 30%),tumor volume: 50%</td><td rowspan=2 colspan=3>Prostate, biopsy;</td><td rowspan=9 colspan=1>Prostate, biopsy;Acinar adenocarcinoma,Gleason&#x27;s score 7 (3+4),grade group 2(Gleason pattern 4: 30%),tumor volume:90%</td></tr><tr><td rowspan=2 colspan=2>Acinar adenocarcinoma,</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Gleason&#x27;s score 7(3+3),</td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>grade group 2</td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=2>(Gleason pattern 4: 30%),</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>tumor volume:90%</td></tr><tr><td rowspan=1 colspan=1>ICL_PathReport (0.6175)</td><td rowspan=1 colspan=4>IMAGINE_Lab (0.7130)</td><td rowspan=1 colspan=3>IUCompPath (0.7560)</td><td rowspan=1 colspan=1>MTS_REG_2025 (0.2817)</td></tr><tr><td rowspan=3 colspan=1>Prostate, biopsy;Acinar adenocarcinomaGleason&#x27;s score 7,grade group 2,tumor volume:90%</td><td rowspan=1 colspan=4>prostate, biopsy;</td><td rowspan=1 colspan=3>Prostate, biopsy;</td><td rowspan=1 colspan=1>Breast, sono-guided core biopsy;</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=3>acinar adenocarcinoma,gleason&#x27;s score 7(4+3),grade group3(Gleason pattern 4:80%),tumor volume:90%</td><td rowspan=2 colspan=3>Acinar adenocarcinoma,Gleason&#x27;s score 7(4+3),grade group3(Gleason pattern 4:60%),tumor volume:90%</td><td rowspan=2 colspan=1>1. Invasive carcninoma ofno special type, grade II(Tubule formation: 3,Nuclear grade: 2, Mitoses: 1)2. Microcalcification</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>MedInsight-ViseurAI (0.2877)</td><td rowspan=1 colspan=4>REG_Path (0.7559)</td><td rowspan=1 colspan=3>TeamTiger@REG2025 (0.9429)</td><td rowspan=1 colspan=1>PathoMozhi (0.6891)</td></tr><tr><td rowspan=3 colspan=1>Lung,biopsy;Chronic granulomatousinflammation with necrosis</td><td rowspan=3 colspan=4>Prostate, biopsy;Gleason&#x27;s score 7(4+3),grade group 3(Gleason pattern 4: 70%),tumor volume:90%</td><td rowspan=3 colspan=4>Prostate, biopsy;                  Structured Summary:Acinar adenocarcinoma,           Organ -Prostate,Gleason&#x27;s score 7 (3+4),           Diagnosis -grade group 2                 Acinar adenocarcinoma,(Gleason pattern 4: 10%),       Gleason&#x27;s score 7 (3+4),tumor volume:90%             grade group 2(Gleason pattern 4: 30%),tumor volume:95%</td></tr><tr><td rowspan=1 colspan=1>Acinar adenocarcinon</td></tr><tr><td rowspan=1 colspan=2>Gleason's score 7(4+</td></tr></table>

Table S7  
Representative case of correct identification of colonic malignant lymphoma across teams. The surface shows normal colonic mucosa, while lymphoma cells infiltrate the underlying tissue. The red annotation delineates the area involved by lymphoma. Most team-generated reports were concordant for both organ (colon) and diagnosis (malignant lymphoma), with occasional organ-labe mismatch (e.g., stomach) despite correct identification of lymphoma. ( =correct, =incorrect)
<table><tr><td rowspan=1 colspan=1>Ground Truth</td><td rowspan=1 colspan=2>ADCT (1.000)</td><td rowspan=1 colspan=2>ICGI (1.000)</td><td rowspan=1 colspan=2>ICL_PathReport (1.000)</td></tr><tr><td rowspan=2 colspan=1>Colon, colonoscopic biopsy;Malignant lymphoma</td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=1>Malignant lymphoma</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>IMAGINE_Lab (1.000)</td><td rowspan=1 colspan=2>IUCompPath (1.000)</td><td rowspan=1 colspan=2>MTS_REG _2025 (0.450)</td><td rowspan=1 colspan=2>MedInsight-ViseurAI (1.000)</td></tr><tr><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=2>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=2>Stomach,endoscopic biopsy;</td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=1>Malignant lymphoma</td></tr><tr><td rowspan=1 colspan=1>REG_Path (1.000)</td><td rowspan=1 colspan=2>TeamTiger@REG2025 (1.000)</td><td rowspan=1 colspan=2>PathoMozhi (1.000)</td><td rowspan=1 colspan=1>NW-TIA (1.000)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=2>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=1>Malignant lymphoma</td></tr></table>

Table S8  
Metaplastic carcinoma mimicking lung squamous cell carcinoma in generated reports. The red circles highlight squamous diferentiation, supporting the ground truth diagnosis of metaplastic carcinoma. ICGI correctly identified metaplastic carcinoma. Some teams produced “invasive carcinoma of no special type”, which was categorized as indeterminate (organ correct but subtype not specified). Several other teams generated lung squamous cell carcinoma by the squamous morphology—morphologically plausible, yet contextually incorrect. ( =correct, =incorrect, =indeterminate)
<table><tr><td rowspan=1 colspan=4>Whole Slide Image</td></tr><tr><td rowspan=1 colspan=4><img src="images/6f19c346c5922bab9ffbabee56769411e63adebe3b87067a87b2fde1a7117bf1.jpg"/></td></tr><tr><td rowspan=1 colspan=2>Ground Truth</td><td rowspan=1 colspan=1>ADCT (0.3373)</td><td rowspan=1 colspan=1>ICGI (1.000)</td></tr><tr><td rowspan=1 colspan=2>Breast, core-needle biopsy;Metaplastic carcinoma</td><td rowspan=1 colspan=1>Breast, sono-guidedcore biospy;Invasive carcinoma of nospecial type, grade II(Tubule formation: 3,Nuclear grade: 2,Mitoses: 1)</td><td rowspan=1 colspan=1>Breast, core-needle biopsy;Metaplastic carcinoma</td></tr><tr><td rowspan=1 colspan=1>ICL_PathReport (0.2980)</td><td rowspan=1 colspan=1>IMAGINE_Lab (0.3071)</td><td rowspan=1 colspan=1>IUCompPath (0.2845)</td><td rowspan=1 colspan=1>MTS_REG_2025 (0.3207)</td></tr><tr><td rowspan=1 colspan=1>Lung, biopsy;Non-small cell carcinoma,not otherwise specified</td><td rowspan=1 colspan=1>lung, biopsy;squamous cell carcinoma</td><td rowspan=1 colspan=1>Lung, biopsy;Non-small cell carcinoma,favor adenocarcinoma</td><td rowspan=1 colspan=1>Lung, biopsy;Squamous cell carcinoma</td></tr><tr><td rowspan=1 colspan=1>MedInsight-ViseurAI (0.4090)</td><td rowspan=1 colspan=1>REG_Path (0.3207)</td><td rowspan=1 colspan=1>TeamTiger@REG2025 (0.2982)</td><td rowspan=1 colspan=1>PathoMozhi (0.3000)</td></tr><tr><td rowspan=1 colspan=1>Breast, core-needle biopsy;Invasive carcinoma of nospecial type, grade II(Tubule formation: 3,Nuclear grade: 2,Mitoses: 1)</td><td rowspan=1 colspan=1>Lung, biopsy;Squamous cell carcinoma</td><td rowspan=1 colspan=1>Lung, biopsy;Non-small cell carcinoma,favor squamous cell carcinoma</td><td rowspan=1 colspan=1>Structured Summary:Organ -Breast,Diagnosis -Invasive carcinomaof no special type, grade III(Tubule formation: 3,Nuclear grade: 3 and Mitoses: 2)</td></tr></table>

Table S9

Rare benign entity-limited recognition across teams. The ground truth diagnosis is fundic gland polyp, a minor category of benign polypoid lesion (only three cases were included in the training set). MTS\_REG\_2025 correctly generated “fundic gland polyp”. In contrast, most other teams generated “chronic gastritis,” reflecting substitution of a common, nonspecific diagnosis rather than recognizing the specific polyp entity. ( =correct, =incorrect)

<table><tr><td rowspan=1 colspan=10>Whole Slide Image</td></tr><tr><td rowspan=1 colspan=2>Ground Truth</td><td rowspan=1 colspan=2>ADCT (0.4858)</td><td rowspan=1 colspan=4>ICGI (0.5602)</td><td></td><td rowspan=1 colspan=1>ICL_PathReport (0.5602)</td></tr><tr><td rowspan=2 colspan=2>Stomach, endoscopic biopsy;Fundic gland polyp</td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=3></td><td rowspan=2 colspan=2>Chronic gastritis</td></tr><tr><td rowspan=1 colspan=2>Adenocarcinoma, poorlydifferentiated with poorlycohesive carcinoma component</td><td rowspan=1 colspan=4>Chronic gastritis</td></tr><tr><td rowspan=1 colspan=2>IMAGINE_Lab (0.5602)</td><td rowspan=1 colspan=2>IUCompPath (0.5602)</td><td rowspan=1 colspan=4>MTS_REG_2025 (1.000)</td><td></td><td rowspan=1 colspan=1>MedInsight-ViseurAI (0.5602)</td></tr><tr><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=3></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Chronic gastritis</td><td rowspan=1 colspan=2>Chronic gastritis</td><td rowspan=1 colspan=4>Fundic gland polyp</td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>REG_Path (0.5602)</td><td rowspan=1 colspan=2>TeamTiger@REG2025 (0.5602)</td><td rowspan=1 colspan=4>PathoMozhi (0.5602)</td><td></td><td rowspan=1 colspan=1>NW-TIA (0.5602)</td></tr><tr><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=4>Stomach, endoscopic biopsy;</td><td rowspan=4 colspan=2>Chronic gastritis</td></tr><tr><td rowspan=3 colspan=2>Chronic gastritis</td><td rowspan=3 colspan=2>Chronic gastritis</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=5>Chronic gastritis with lymphoid</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=4>aggregates</td></tr></table>

Table S10  
Stomach endoscopic biopsy: malignant lymphoma. Nine teams correctly generated malignant lymphoma. Two teams instead generated “small cell carcinoma”. Given the phenotypic overlap on H&E, the model’s output is consistent with High-Fidelity Histomorphology-Driven Phenotype Recognition. ( =correct, =incorrect)
<table><tr><td rowspan=1 colspan=2>Ground Truth</td><td rowspan=1 colspan=2>ADCT (0.5323)</td><td rowspan=1 colspan=2>ICGI (1.000)</td><td rowspan=1 colspan=2>ICL_PathReport (1.000)</td></tr><tr><td rowspan=2 colspan=2>Stomach, endoscopic biopsy;Malignant lymphoma</td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td></tr><tr><td rowspan=1 colspan=2>No significant pathologicfindings</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td></tr><tr><td rowspan=1 colspan=2>IMAGINE_Lab (1.000)</td><td rowspan=1 colspan=2>IUCompPath (1.000)</td><td rowspan=1 colspan=2>MTS_REG_2025 (0.5639)</td><td rowspan=1 colspan=2>MedInsight-ViseurAI (1.000)</td></tr><tr><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td></tr><tr><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Small cell carcinoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td></tr><tr><td rowspan=1 colspan=2>REG_Path (1.000)</td><td rowspan=1 colspan=2>TeamTiger@REG2025 (1.000)</td><td rowspan=1 colspan=2>PathoMozhi (0.5639)</td><td rowspan=1 colspan=2>NW-TIA (1.000)</td></tr><tr><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Stomach, endoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Stomach, endoscopic biopsy;</td></tr><tr><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Malignant lymphoma</td><td rowspan=1 colspan=2>Small cell carcinoma</td><td></td><td></td></tr></table>

Table S11  
Preservation of colorectal-type gland-forming morphology despite incorrect site prediction. Notably, despite the site mismatch, these outputs consistently captured a colorectal-type, gland-forming phenotype concordant with the suspected primary, illustrating the models’ ability to encode and reproduce diagnostically relevant histomorphologic features. (Red annotation: colorectal-type tumor portion, =correct, =incorrect, =indeterminate)
<table><tr><td rowspan=1 colspan=6>Whole Slide Image</td></tr><tr><td rowspan=1 colspan=1>Ground Truth</td><td rowspan=1 colspan=1>ADCT (0.5330)</td><td rowspan=1 colspan=2>ICGI (0.3817)</td><td rowspan=1 colspan=2>ICL_PathReport (0.3817)</td></tr><tr><td rowspan=2 colspan=1>Uterine cervix, colposcopic biopsy;Adenocarcinoma,favor colorectal primary</td><td rowspan=2 colspan=1>Uterine cervix, colposcopic biopsy;Endocervical adenocarcinoma,HPV-associated, usual type</td><td rowspan=1 colspan=2>Rectum, colonoscopic biopsy;</td><td rowspan=2 colspan=2>Rectum, colonoscopic biopsy;Adenocarcinoma,moderately differentiated</td></tr><tr><td rowspan=1 colspan=2>Adenocarcinoma,moderately differentiated</td></tr><tr><td rowspan=1 colspan=1>IMAGINE_Lab (0.3796)</td><td rowspan=1 colspan=1>IUCompPath (0.3817)</td><td rowspan=1 colspan=2>MTS_REG _2025 (0.2741)</td><td rowspan=1 colspan=2>MedInsight-ViseurAI (0.3796)</td></tr><tr><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;Adenocarcinoma,moderately differentiated</td><td rowspan=1 colspan=1>Rectum, colonoscopic biopsy;Adenocarcinoma,moderately differentiated</td><td rowspan=1 colspan=2>Lung, biopsy;Squamous cell carcinoma</td><td rowspan=1 colspan=2>Colon, colonoscopic biopsy;Adenocarcinoma,moderately differentiated</td></tr><tr><td rowspan=1 colspan=1>REG_Path (0.3796)</td><td rowspan=1 colspan=1>TeamTiger@REG2025 (0.3470)</td><td rowspan=1 colspan=2>PathoMozhi (0.3796)</td><td rowspan=1 colspan=2>NW-TIA (0.3796)</td></tr><tr><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=2 colspan=1>Lung, biopsy;Adenocarcinoma</td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Colon, colonoscopic biopsy;</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Adenocarcinoma,moderately differentiated</td><td rowspan=1 colspan=1>Adenocarcinoma,moderately differentiated</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Adenocarcinoma,moderately differentiated</td></tr></table>

Table S12  
Fine-grained histologic grading (Nottingham) reproduced in generated pathology reports. The Nottingham histologic grade is determined by summing three component scores—tubule (gland) formation, nuclear pleomorphism (variation), and mitotic count—each scored from 1 to 3 (total score 3–5 = Grade I, 6–7 = Grade II, 8–9 = Grade III). Tubule formation is graded based on the proportion of glandular structures (<10% = score 3, 10–75% = score 2, >75% = score 1). In this case, the component score were tubule formation=3 (glandular structure <10%), nuclear grade=3 (marked nuclear variation), and mitotic count = 2 (representative mitotic figures: red circle), yielding a total score of 8 (Grade III). ICGI and ICL\_PathReport correctly reproduced the diagnosis, the overall Nottingham grade, and the individual component scores. ( =correct, =incorrect)
<table><tr><td rowspan=1 colspan=6>Whole Slide Image<img src="images/505288cdf90973123fd78b344809f50e555f15a69f7b8eac26d2a3759e8b5b1d.jpg"/></td></tr><tr><td rowspan=1 colspan=3>Ground Truth</td><td rowspan=1 colspan=2>ADCT (0.8141)</td><td rowspan=1 colspan=1>ICGI (1.000)</td></tr><tr><td rowspan=2 colspan=3>Breast, sono-guided core biopsy;Invasive carcinoma of no special type, grade III(Tubule formation: 3, Nuclear grade: 3, Mitoses: 2)</td><td rowspan=1 colspan=2>Breast, sono-guided core biopsy;</td><td rowspan=2 colspan=1>Breast, sono-guided core biopsy;Invasive carcinoma of no           Invasive carcinoma of nospecial type, grade                special type, grade III(Tubule formation: 3,              (Tubule formation: 3,Nuclear grade:1,               Nuclear grade: 3,Mitoses:1)                    Mitoses: 2)</td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=2>ICL_PathReport (1.000)</td><td rowspan=1 colspan=1>IMAGINE_Lab (0.2525)</td><td rowspan=1 colspan=2>IUCompPath (0.2627)</td><td rowspan=1 colspan=1>MTS_REG_2025 (0.2667)</td></tr><tr><td rowspan=3 colspan=6>lung, biopsy;                     Lung, biopsy;                     Lung, biopsy;adenocarcinoma                  Non-small cell carcinoma,          Squamous cell carcinomafavor adenocarcinomaPathoMozhi (0.7060)Breast, sono-guided core biopsy;    Lung, biopsy;                    Lung, biopsy;                     Structured Summary:Invasive carcinoma of no           Non-small cell carcinoma,          Non-small cell carcinoma,       Organ -Breast,special type, gradeⅡI            favor adenocarcinoma              favor adenocarcinoma           Diagnosis -Invasive carcinoma(Tubule formation: 3,                                                                               of no special type, grade IIINuclear grade:2,                                                                                 (Tubule formation: 3,Mitoses:1)                                                                                      Nuclear grade: 3 andMitoses: 2)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>MedInsight-ViseurAI (0.8148)</td><td rowspan=1 colspan=1>REG_Path (0.2627)</td><td rowspan=1 colspan=2>TeamTiger@REG2025 (0.2627)</td></tr></table>