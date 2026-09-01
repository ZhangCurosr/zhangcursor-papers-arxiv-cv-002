# OCR-Based Field Extraction for Archaeological Pottery Metadata: The CENTURIA Dataset

Gissu Valentina Naghavi<sup>1</sup> , Dominik Hagmann<sup>2</sup> , Martin Kampel<sup>1</sup> , and Irene Ballester<sup>1</sup>

<sup>1</sup> Computer Vision Lab, TU Wien, Vienna, Austria

{gissu.naghavi,martin.kampel,irene.ballester}@tuwien.ac.at

Austrian Archaeological Institute, Austrian Academy of Sciences, Vienna, Austria {dominik.hagmann}@oeaw.ac.at

![](images/1c7ab4a61b42c61a91b4de84b404e5fa453f89f20a92b09ff20aa929d4db2de0.jpg)  
Fig. 1: From analogue archaeological documentation to structured metadata. Ceramic sherds (A) are recorded by hand (B) and retro-digitised (C), yet handwritten annotations remain inaccessible to computational analysis. We propose a pipeline for their automated conversion via HTR (D) into machine-readable metadata (E), and introduce the CENTURIA dataset to enable systematic benchmarking.

Abstract. Pottery is a primary source for reconstructing the chronological and economic dimensions of past societies. Archaeologists often document ceramic finds through technical drawings and handwritten metadata. This metadata is critical for dating, provenance attribution, and cross-site comparison, but remains inaccessible to computational analysis, requiring manual transcription of every record. We investigate whether state-of-the-art document analysis models can address this task, and introduce CENTURIA, a dataset of 507 pottery records from the Roman site of Carnuntum, providing transcriptions, bounding boxes, and structured field-level labels across seven metadata categories. Benchmarking five OCR models reveals a substantial domain gap: zeroshot transcription error reaches 15–32 % SpACER-M, far exceeding rates on printed archival documents, with domain-specific fields recovered in fewer than 3 % of cases. LoRA fine-tuning on just 57 samples, reflecting a realistic archival annotation budget, closes this gap, reducing transcription error to below 1.5 % and recovering overall field-level accuracy above 87 %. Our results show that a small expert-validated fine-tuning set sufices to convert handwritten pottery documentation into structured, searchable metadata ready for archaeological databases.

Keywords: archaeological archives · pottery documentation · handwritten text recognition · information extraction

## 1 Introduction

Archaeological archival collections comprise tens of thousands of technical 2D drawings of pottery sherds. Due to its durability and ubiquity, pottery is a primary source for reconstructing past societies [58]. Although digital recording methods are gaining importance, pottery documentation still largely follows traditional analogue procedures [22, 59, 77]: selected diagnostic fragments are drawn by hand as technical cross-sections, following comparatively standardised conventions that support comparison across sites and periods [12].

Even when these materials are retro-digitised, the annotations accompanying them remain inaccessible to computational analysis (Fig. 1 (A–C)): digitisation preserves only their visual appearance. These handwritten annotations are the primary carriers of excavation context, typological classification, and inventory numbers, indispensable for cross-site comparison, chronological reconstruction, and provenance attribution, yet their recovery requires intensive manual transcription, making archive-scale queries infeasible in practice [37].

Handwritten Text Recognition (HTR) ofers a practical path to making such archives machine-readable: approaches span from two-stage detection-recognition pipelines [50] to full-page Vision-Language Models (VLMs) [60] and lightweight specialised models [69], achieving character error rates below 5 % on standard handwritten benchmarks [21, 27]. However, it remains unclear how these advances transfer to archaeological material, as widely used benchmarks such as IAM [52] and READ [66] cover only sequential text and do not reflect archival conditions, including spatially scattered annotations without a canonical reading order, faded ink, and dense domain-specific abbreviations.

In addition, raw transcriptions are insuficient for archive-scale analysis: retrieving vessels by form, filtering by ware type, or aggregating measurements requires structured metadata. As Fig. 1 (D–E) shows, Key Information Extraction (KIE) [41,63] bridges this gap by mapping transcribed text onto predefined semantic fields, where character-level errors compound into degraded extraction quality, making automated error detection crucial for practical deployment.

In this paper, we investigate whether state-of-the-art models can automate handwritten archaeological pottery documentation, with the practical aim of enabling heritage practitioners to make informed decisions about deploying these tools on their own archives. We show that zero-shot HTR models fall short, with transcription errors between 15 and 32 %, but LoRA fine-tuning on 57 samples, representative of a constrained annotation setting, reduces this to below 1.5 %, recovering overall field-level accuracy above 87 %. Our contributions are:

– We introduce CENTURIA: the first benchmark for HTR and field extraction of handwritten archaeological records with scattered annotations and technical terminology, comprising 507 expert-validated samples.

– We present a systematic evaluation across five OCR models, three adaptation strategies, and two field extraction methods, assessing transcription quality, per-field performance, and error modes, showing that transcription quality is the primary determinant of field-level accuracy.

– We provide an end-to-end metadata extraction pipeline that closes the zeroshot domain gap with 57 fine-tuning samples, representative of realistic archival annotation budgets. By cross-combining models and extractors, we establish a heuristic agreement criterion that auto-accepts 56.4 % of scans at 95.5 % field precision, rising to 98.7 % on fields confirmed by all four pipelines, concentrating manual review on unresolved conflicts.

The complete CENTURIA dataset, fine-tuned checkpoints, and full code are released under CC-BY-4.0 at https://github.com/gissuvalentina/CENTURIA.

## 2 Related Work

Document Analysis in Archaeological Research. Archaeology increasingly draws on automated methods for visual recognition, classification, and 3D reconstruction of artefacts and sites [8, 23, 28, 57]. For pottery specifically, prior work addresses the visual layer: ArchAIDE [4] identifies ceramic types from photographs using shape- and appearance-based neural networks, while AutArch [45] detects objects and extracts geometric data from printed archaeological catalogues. For the archival text layer, OCR-based workflows produce structured metadata from printed heritage documents [9], and digital repositories such as tDAR [53] support access to and long-term preservation of digitised archaeological records. However, VLM-based generation of catalogue descriptions for archival photographic collections shows that hallucinations and domain-specific terminology make fully automated processing unreliable even at this level [1]. The challenge is more acute for handwritten records: recent attempts to digitise handwritten archaeological excavation notes confirm that training OCR on such material is time-intensive with low out-of-the-box accuracy [24]. None of these approaches provides a solution for the handwritten annotation layer of archaeological records. The closest work to ours is the PyPottery toolkit [13–15], an open-source suite for the digitisation of archaeological pottery documentation. As part of its transcription component, PyPotteryScan [16] applies OCR models to text annotations, but requires manual bounding box annotation as input and, to the best of our knowledge, reports no published evaluation of transcription quality or field extraction accuracy. We employ PyPotteryScan to generate preliminary transcriptions for the creation of ground truth; however, all documents require manual correction. Our work goes beyond PyPotteryScan by providing a fully automatic pipeline and the first systematic evaluation for handwritten pottery documentation.

Text Recognition Models and Field Extraction. HTR models evolved from linelevel recognition to end-to-end full-page processing [27]: architectures range from specialised HTR systems [19,62] and two-stage detection-recognition pipelines [50] to full-page VLMs [60, 61], generalist vision models [75], and compact reinforcement learning-optimised systems [69]. Large pre-trained models can match specialised HTR engines on historical scripts [21, 64], and fine-tuning on small domain-specific samples substantially reduces error rates [42,46]. Raw transcriptions alone, however, are insuficient for archive-scale analysis: KIE [41, 63] addresses this by mapping text onto structured schemas using layout-aware [40] and OCR-free architectures [44], but existing datasets [67] assume printed business documents with standard layouts, neither of which holds for handwritten archaeological records. Transferability of these models and extraction methods to handwritten archaeological archives with scattered annotations and no canonical reading order remains unaddressed; we provide such an assessment to inform deployment decisions for heritage practitioners.

![](images/02300e5ad6a162657400f23f74b97853d4062f1df7338c9bb86b86a15d9a602b.jpg)  
Fig. 2: CENTURIA document crops. From left to right: high-quality scan, lowcontrast scan, crossed-out text, and handwriting overlapping with a technical drawing.

Benchmarks for HTR in Historical and Cultural Heritage Documents. No existing benchmark covers archaeological documentation: systematic evaluation of HTR and KIE on historical documents centres on administrative and ecclesiastical records with well-defined structures. ESPOSALLES/IEHHR [26, 65] provides semantic labels for handwritten marriage records, SIMARA [71] targets key-value extraction from archival index cards, and POPP [20] covers population census tables. Medieval corpora such as CATMuS Medieval [17] address abbreviation-rich writing but focus on transcription rather than structured extraction. Record types and vocabulary across these resources are drawn from civil administration, genealogy, and ecclesiastical archives; archaeological field documentation, with its domain-specific abbreviations, heterogeneous record structures, and scattered annotations without a canonical reading order, is not represented. CENTURIA is the first benchmark assessing transcription and field-level KIE for this record type, covering model comparison, adaptation strategies, and extraction methods.

## 3 CENTURIA Dataset

CENTURIA comprises 507 annotated scans of analogue pottery records from the archaeological site of Carnuntum<sup>1</sup>, the largest Roman settlement in presentday Austria and part of the UNESCO World Heritage Site “Danube Limes” [36]. Each scan is annotated with bounding boxes for text regions and technical drawings, manually transcribed text, and structured field-level values. A predefined train/test split (train: $n = 5 7 ,$ , test: n = 450), drawn by proportional stratified random sampling across six campaigns, supports reproducible evaluation and fine-tuning.

Table 1: CENTURIA dataset overview. Dataset statistics, annotation efort, and stratification criteria; the latter reported as document counts and percentage share.
<table><tr><td colspan="2">Dataset Statistics</td><td colspan="3">Stratification Criteria</td></tr><tr><td>Total documents</td><td>507</td><td>Annotation Density</td><td></td><td></td></tr><tr><td>Total bounding boxes</td><td>3345</td><td>Low (1–5 bbox)</td><td>155</td><td>30.57%</td></tr><tr><td>Total characters</td><td>33767</td><td>Medium (6–8 bbox)</td><td>291</td><td>57.40%</td></tr><tr><td>Total words</td><td>8261</td><td>High (9+ bbox)</td><td>61</td><td>12.03%</td></tr><tr><td>Annotation Statistics</td><td></td><td>Special Cases</td><td></td><td></td></tr><tr><td>Correction (manual)</td><td>507 (100%)</td><td>Crossed-out text</td><td>15</td><td>2.96%</td></tr><tr><td>Expert consultation</td><td>48 (9.47%)</td><td>Overlapping annot.</td><td>60</td><td>11.83%</td></tr></table>

## 3.1 Data Collection

CENTURIA is compiled from the Carnuntum documentation archive at the Austrian Archaeological Institute (OeAI) [35, for selected data]. The archive comprises approximately 70,000 analogue pottery records from which CENTURIA samples 507 documents across six campaigns from 1976–2017. Each record documents a common-ware pottery sherd predominantly in abbreviated German, and exhibits variation in handwriting style, scribal conventions, and document condition across campaigns and authors. Sampling is stratified to capture text diversity (handwriting style, character types, and annotation density across campaigns) and special cases (crossed-out text and overlapping annotations; illustrated in Fig. 2, with statistics in Table 1). Appendix A lists the campaigns with their record counts and describes the sampling strategy in full.

## 3.2 Annotation Methodology

Transcription Annotation. Bounding boxes for text regions and technical drawings are manually drawn following the annotation protocol detailed in Appendix A.2. To generate preliminary transcriptions, we use PyPotteryScan [16], an open-source tool built on olmOCR2 [61] for semi-automatic transcription of archaeological pottery documentation. All 507 transcribed documents require manual correction, taking an average of 70 seconds per record. In 48 cases (9.47 % of documents), the correct reading cannot be determined from the scan alone and is resolved by consulting an expert in the archaeological domain. These cases cover illegible characters, ambiguous abbreviations, and site-specific terminology.

Field Annotation. Each transcription is mapped onto a structured schema of seven semantic categories (illustrated in Table 2), for example, pottery\_form:

Table 2: Metadata fields in CENTURIA with representative values. The schema covers seven semantic categories; all fields are optional.
<table><tr><td>Field</td><td>Subfield</td><td>Example Values</td></tr><tr><td rowspan="7"></td><td>provenance</td><td>Carn., Car., CAR</td></tr><tr><td>project</td><td>GS 2012, Survey GB 2017, Limesgasse 2016</td></tr><tr><td>fn</td><td>206, A 7/1, 67/95</td></tr><tr><td>se</td><td>109, 142</td></tr><tr><td>fl</td><td>1, 2</td></tr><tr><td>quadrant</td><td>Q 01-10, Q 44-02</td></tr><tr><td>kiste</td><td>Ki 1/76, Ki 131/76</td></tr><tr><td>pottery_form</td><td></td><td>Topf, Schüssel, Krug, Deckel, Amphore</td></tr><tr><td>artefact_category</td><td></td><td>f/ox GK, g/red GK, f/ox PGW, GK Mischbrand</td></tr><tr><td>publication_type</td><td></td><td>Gassner 3/4, Petznek 17.6, Grünewald 1979 Taf. 54/14</td></tr><tr><td>surface_treatment-</td><td></td><td>Üz, Üz a + i, Glasur, Grießbewurf</td></tr><tr><td rowspan="4">measurement</td><td>rd, bd</td><td>16 cm, 5 cm</td></tr><tr><td>h</td><td>2.1 cm</td></tr><tr><td></td><td>r_a, r_i, r_m 8 cm, 8 cm (5 %)</td></tr><tr><td></td><td>SPA, M 1:1, 14.12.2012, Gez. MBC</td></tr></table>

Krug (jug), artefact\_category: f/ox GK (oxidation-fired coarse ware). Full field descriptions are in Appendix A.2. To reduce manual annotation, a rulebased script applies regular expressions (REGEX) and controlled vocabularies to the corrected transcriptions, matching and assigning text spans to their corresponding fields. Unrecognised text is assigned to an others category. The resulting field annotations are verified by a Carnuntum site specialist, with corrections applied in 3.75 % of the documents, all involving edge cases such as crossed-out entries, field boundary errors, and ambiguous plate references (Taf. X).

## 4 Method

Given a scanned pottery documentation record, the method produces structured metadata fields through two stages: text recognition and field extraction (Fig. 3).

## 4.1 Handwritten Text Recognition

Each OCR model f<sub>θ</sub> takes a single-page scan x as input and produces a character sequence ${ \hat { y } } = f _ { \theta } ( x )$ as a transcription of the handwritten content. Raw model output undergoes post-processing $\tilde { y } = g ( \hat { y } )$ to normalise formatting artefacts before field extraction (post-processing steps are described in Appendix B).

Baseline Models. The models span the following paradigms: a traditional twostage detection-recognition pipeline (TrOCR), first- and second-generation fullpage VLMs (olmOCR, olmOCR2 ), a generalist multi-task model (Florence2 ), and a lightweight reinforcement-learning-optimised model (LightOnOCR). All models are open-source, prioritising reproducible and cost-free deployment for heritage practitioners; checkpoints are detailed in Appendix B.2.

![](images/6b977717ce623f290dd0994f2cf75dc2e80b537b663f431c9097e9a7c6199845.jpg)  
Fig. 3: Two-stage pipeline: HTR followed by structured field extraction. A scanned pottery record is transcribed by an HTR model (optionally adapted via instruction prompting, few-shot prompting, or LoRA fine-tuning), then parsed into structured metadata fields via REGEX or LLM-based (Qwen3-8B) extraction.

Two-stage detection-recognition. TrOCR [50] pairs a Vision Transformer (ViT) encoder with a language model decoder pre-trained on large-scale text data. The encoder splits the detected text into image patch sequences and the decoder generates the transcriptions at wordpiece level conditioned on these embeddings. As TrOCR requires pre-segmented text regions, we use CRAFT [5] via the Easy-OCR library [43] for line segmentation. Unlike full-page models, each region is processed independently without access to the document context, and transcription quality therefore depends directly on detection. The output consists of recognised text sequences with corresponding bounding box coordinates.

Full-page vision-language models. Unlike TrOCR, these models take the entire document image as input, bypassing explicit text detection. olmOCR [60] represents the first generation of full-page VLMs for OCR, built on Qwen2- VL-7B [72] and trained for structured document understanding across complex layouts. olmOCR2 [61] builds on Qwen2.5-VL-7B-Instruct [6], the successor of Qwen2-VL-7B, which extends its predecessor with dynamic resolution processing and finer-grained document parsing. It further specialises for OCR via reinforcement learning with verifiable rewards (RLVR) on synthetic unit tests, improving over olmOCR on structured document layouts. Florence2 [75] adopts a unified sequence-to-sequence architecture across vision tasks and is included to examine whether broad multi-task pre-training can compete with specialised approaches on handwritten domain-specific documents.

Lightweight specialised model. LightOnOCR [51, 69] is a 1B-parameter end-toend VLM that takes a full image as input and outputs a text transcription. It consists of a native resolution ViT encoder initialised from Mistral-Small-3.1 [54], a two-layer MLP projector that reduces visual token count via spatial patch merging, and a Qwen3 [76] decoder that generates the transcription conditioned on the projected visual tokens without task prompts at inference. The model is pretrained end-to-end by distillation from a larger VLM teacher and posttrained with RLVR. At 1B parameters, LightOnOCR is roughly an order of magnitude smaller than the strongest full-page VLMs on olmOCR-Bench [69] and represents the lightweight end of the baseline model spectrum.

Domain Adaptation Strategies. Pre-training covers general document OCR, with the exception of TrOCR, which uses a checkpoint specifically fine-tuned on the IAM handwriting dataset [49,52]. Nevertheless, a domain gap in archaeological vocabulary and annotation conventions limits zero-shot transcription quality across all models. We therefore explore three adaptation strategies: zero-shot instruction prompting [21], few-shot prompting [74], and LoRA fine-tuning [39].

Zero-shot instruction prompting augments the model input with a domainspecific instruction prompt $c ,$ conditioning the model as $\hat { y } = f _ { \theta } ( x , c )$ . Since the models are not exposed to archaeological terminology during training, explicit instructions aim to improve recognition of domain-specific abbreviations and notation without requiring labelled data. The full prompt is used for all models; design details are in Appendix B.5 and the complete prompt is in Appendix E.

Few-shot prompting extends this by providing k annotated examples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { k }$ alongside the input, conditioning the model as ${ \hat { y } } = f _ { \theta } ( x \mid ( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { k } , y _ { k } ) )$ Demonstrations of the expected input-output format allow the model to infer transcription conventions and domain vocabulary without weight updates. We use $k \in \{ 1 , 5 , 8 \}$ examples. For reproducibility, we provide full prompt templates and example selection details in Appendices B.5 and E.

LoRA fine-tuning [39] adapts model weights on the CENTURIA training set $( n = 5 7 )$ using parameter-eficient fine-tuning via low-rank weight decomposition $\varDelta W = B A$ , where $B \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ and $A \in \mathbb { R } ^ { r \times d _ { \mathrm { i } _ { \mathrm { i } } } }$ <sup>n</sup> with rank $r \ll$ min $( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ , applied to all linear projection layers. Hyperparameters are given in Appendix B.4.

## 4.2 Field Extraction

Field extraction (KIE [41, 63]) maps a transcription $\tilde { y }$ onto a predefined schema of $N = 7$ semantic categories $\mathcal { K } = \{ k _ { 1 } , \ldots , k _ { N } \}$ , producing field-value pairs $\mathcal { F } = \{ ( k _ { i } , v _ { i } ) \} _ { i = 1 } ^ { N }$ , where $v _ { i } \in \mathcal { V } _ { k _ { i } } \cup \{ \mathtt { n u l l } \}$ and $\nu _ { k _ { i } }$ is the value domain for field $k _ { i }$ . Since field boundaries are not marked in the original records, the mapping $\tilde { y }  \mathcal { F }$ relies entirely on textual patterns and domain vocabulary. We compare two extraction methods that answer diferent questions: a rule-based approach relying on exact pattern matching, which establishes the accuracy attainable under transcription noise, and an LLM-based approach operating on the full transcription sequence, which tests whether extraction transfers without archivespecific engineering.

REGEX-based extraction applies the same rule-based script used to produce ground truth metadata field annotations (Sec. 3.2), using regular expressions and controlled vocabularies $\mathcal { V } = \{ V _ { k } \} _ { k = 1 } ^ { N }$ derived from the field schema, such that $v _ { i } = \mathrm { m a t c h } ( \tilde { y } , V _ { k _ { i } } )$ for categorical fields and $v _ { i } = \mathrm { r e g e x } ( \tilde { y } , p _ { k _ { i } } )$ for open-domain fields such as measurements, where $p _ { k _ { i } }$ is the pattern for field $k _ { i } .$ . To prevent duplicate assignments, matched substrings are immediately removed from the prediction; remaining unmatched text is aggregated into others. Sharing the ground truth rule set, REGEX is correct by construction on verified transcriptions and fails only on OCR noise. It therefore isolates noise propagation and shows what deterministic rules achieve once transcription is accurate, without inference hardware but with archive-specific patterns.

LLM-based extraction uses an unmodified Qwen3-8B [76] to approximate $\tilde { y }  \mathcal { F }$ by prompting the model with cleaned transcription $\tilde { y }$ and the field schema $\kappa ,$ returning a structured JSON object with field-value pairs. The model is prompted in an 8-shot setting, with examples pairing corrected transcriptions with their expected field annotations; corrected examples perform slightly better than corrupted ones, particularly on publication type (Appendix C.3). Fields absent or unrecognisable in the transcription are assigned null; tokens that do not match any schema field are collected in the others list. Unlike REGEX, the LLM uses a natural language prompt rather than authored patterns, which simplifies adaptation and lets it infer fields from corrupted tokens. For reproducibility, further details and the complete prompt are provided in Appendices B.5 and E.

## 5 Evaluation

We evaluate all five models on transcription quality, and the two strongest baselines (olmOCR2 and LightOnOCR) on domain adaptation, field extraction, and cross-pipeline confidence estimation. Processing time is recorded to assess scalability. Full results, qualitative examples, details of error analysis, and confidence estimation are provided in Appendices C.1 to C.5 and D.

## 5.1 Evaluation Metrics

Transcription. Following [11], we evaluate transcription quality using SpACER, as CER assumes a fixed linear reading order that does not hold for scattered annotations in archaeological ceramic records. For models providing bounding boxes (TrOCR, Florence2), Micro SpACER (SpACER-m) anchors predictions to ground truth via IoU matching; for VLMs without spatial outputs, Macro SpACER (SpACER-M) pools characters page-wide:

$$
\mathrm { S p A C E R - m } = \frac { \sum _ { j } ^ { n } D _ { j } + \hat { E } } { 2 C } , \qquad \mathrm { S p A C E R - M } = \frac { D + \hat { E } } { 2 C }
$$

where $D _ { j }$ are deletions in matched region $j , D = \operatorname* { m a x } ( 0 , | g | - | p | )$ is the pagelevel deletion count, g and p are the ground truth and predicted character count vectors, ${ \hat { E } } = \| g - p \| _ { 1 }$ is the global character count diference, and $C = | g |$ is the total ground-truth character count. Following [68], we additionally report BoW-F1, computed as $\begin{array} { r } { F 1 = 2 \cdot \frac { | P \cap G | } { | P | + | G | } } \end{array}$ over the multi-set intersection of predicted and ground truth word sets, evaluating keyword presence regardless of position.

Table 3: Character and word-level OCR performance with field-level accuracy and computational eficiency. SpACER-m only reported for TrOCR and Florence2. EMR and ANLS are per-document averages across all GT fields (test set, $n = 4 5 0 )$ . Timing measured on NVIDIA GeForce RTX 3090 (2 × 24 GB VRAM). (+) denotes models fine-tuned with LoRA (n = 57). Best in bold, second best underlined.
<table><tr><td>Model</td><td>SpACER-m ↓(%)</td><td>SpACER-M ↓(%)</td><td>BoW-F1 ↑(%)</td><td>EMR ↑(%)</td><td>ANLS ↑(%)</td><td>t/Doc ↓ (s)</td></tr><tr><td>TrOCR</td><td>37.50</td><td>32.35</td><td>41.81</td><td>7.13</td><td>32.81</td><td>0.81</td></tr><tr><td>olmOCR</td><td></td><td>32.22</td><td>43.47</td><td>13.71</td><td>30.07</td><td>12.20</td></tr><tr><td>olmOCR2</td><td></td><td>15.58</td><td>56.74</td><td>30.55</td><td>57.43</td><td>2.19</td></tr><tr><td>Florence2</td><td>31.82</td><td>26.67</td><td>37.69</td><td>17.39</td><td>51.13</td><td>0.83</td></tr><tr><td>LightOnOCR</td><td></td><td>18.25</td><td>50.98</td><td>21.59</td><td>47.62</td><td>2.95</td></tr><tr><td>olmOCR2 (+)</td><td></td><td>1.11</td><td>96.24</td><td>92.08</td><td>95.29</td><td>2.26</td></tr><tr><td>LightOnOCR (+)</td><td></td><td>1.21</td><td>94.80</td><td>89.40</td><td>94.75</td><td>1.42</td></tr></table>

Field-Level Transcription. To isolate transcription errors from extraction errors and identify which field types are most afected by OCR noise, we use Exact Match Rate (EMR) and Average Normalised Levenshtein Similarity (ANLS), following [10]. EMR checks whether $v _ { i } ^ { * }$ appears exactly in ${ \tilde { y } } ,$ computed perdocument and per-field type. Since a single misread character yields ${ \mathrm { E M R } } = 0 .$ ANLS complements it with a normalised edit-distance similarity between $\hat { v } _ { i }$ and $v _ { i } ^ { * } ~ ( \tau = 0 . 5 )$ , tolerating minor OCR errors while assigning zero similarity to predictions that difer by more than half their characters.

Field Extraction. Predicted field values $\hat { \mathcal { F } }$ are matched against ground truth annotations ${ \mathcal { F } } ^ { * }$ after string normalisation, including lowercasing, whitespace collapsing, and delimiter standardisation. Following [7], we report Per-Field Accuracy and macro-averaged Overall Accuracy, ignoring unannotated fields.

## 5.2 Transcription Results

Baseline Performance. Table 3 shows that zero-shot transcription is challenging across all models (SpACER-M 15.58–32.35 %), exceeding the 10.6 % reported for olmOCR on printed archival documents [11] and confirming the difficulty of handwritten pottery documentation. Field-level accuracy is similarly low: even the strongest baseline, olmOCR2, achieves only 30.55 % EMR and 57.43 % ANLS, indicating that field values are rarely recovered verbatim from zero-shot output. Figure 4 illustrates the accuracy-eficiency trade-of: TrOCR and Florence2 are the fastest models (under 1 s/doc) but among the least accurate, while olmOCR is the least eficient (12.2 s/doc) without a corresponding accuracy advantage. We therefore restrict subsequent domain adaptation and field extraction to the two most accurate baselines, olmOCR2 and LightOnOCR.

Domain Adaptation. Table 4 evaluates prompt-based adaptation strategies on olmOCR2 and LightOnOCR; full results across all strategies and shot counts are in Appendix C.2. Zero-shot instruction prompting substantially improves olmOCR2 (SpACER-M: 15.58 % → 6.77 %, BoW-F1: 56.74 % → 83.08 %), while its efect on LightOnOCR is smaller. Few-shot prompting reduces olmOCR2 error (SpACER-M 5.38 % at 5 shots), but collapses LightOnOCR into repetitive output loops (SpACER-M 48.71 % at 1 shot, 100 % at 5 shots), consistent with known repetition failure modes [38] and likely reflecting its RLVR training towards short, prompt-free outputs (Appendix C.2). LoRA fine-tuning nonetheless dominates all prompting strategies. With just 57 samples, SpACER-M drops from over 15 % to below 2 % for both models, showing that even minimal finetuning data outperforms prompt engineering alone. LightOnOCR (+) additionally reduces inference time through shorter outputs, while olmOCR2 (+) shows no such gain (details in Appendix C.5). This confirms that efective domain adaptation requires only a small annotation efort, making the approach viable for archives without large labelled collections.

![](images/7b161fcf5a93a1f6eb70f3384b3da0780353dfd9fd4b21ed0fa002fa347241d0.jpg)  
Fig. 4: Transcription error vs. processing time. LoRA fine-tuned models (⋆) achieve the best accuracy-eficiency trade-of, with SpACER-M around 1 %.

![](images/8857db35d69ad5d0e880981646a0dcffdae7578cf6888339517fa82013839dd3.jpg)  
Fig. 5: Field-level EMR by category. Baseline models fail on domain-specific fields; LoRA fine-tuning achieves ∼ 90 % overall EMR.

Table 4: Domain adaptation strategies. Applied to olmOCR2 and LightOnOCR. Best in bold, second best underlined.
<table><tr><td>Model</td><td>Strategy</td><td>SpACER-M ↓(%)</td><td>BoW-F1 ↑(%)</td><td>EMR ↑(%)</td><td>ANLS ↑(%)</td></tr><tr><td rowspan="4">olmOCR2</td><td>Baseline (zero-shot)</td><td>15.58</td><td>56.74</td><td>30.55</td><td>57.43</td></tr><tr><td>Zero-shot instr. prompting</td><td>6.77</td><td>83.08</td><td>74.64</td><td>84.99</td></tr><tr><td>Few-shot (5 shots)</td><td>5.38</td><td>84.44</td><td>68.97</td><td>87.51</td></tr><tr><td>LoRA fine-tuning (2 epochs)</td><td>1.11</td><td>96.24</td><td>92.08</td><td>95.29</td></tr><tr><td rowspan="4">LightOnOCR</td><td>Baseline (zero-shot)</td><td>18.25</td><td>50.98</td><td>21.59</td><td>47.62</td></tr><tr><td>Zero-shot instr. prompting</td><td>14.13</td><td>53.71</td><td>35.27</td><td>65.14</td></tr><tr><td>Few-shot (1 shot)</td><td>48.71</td><td>43.57</td><td>28.02</td><td>67.24</td></tr><tr><td>LoRA fine-tuning (2 epochs)</td><td>1.21</td><td>94.80</td><td>89.40</td><td>94.75</td></tr></table>

Table 5: Field extraction accuracy (%) under REGEX and LLM extraction (Qwen3-8B). Overall is the per-document average across all fields. Excavation aggregates provenance, crate, project, FN, SE, FL, and quadrant (weighted by n). (+) denotes LoRA fine-tuned models. Bold: best per column.
<table><tr><td></td><td colspan="6">REGEX Extraction</td><td colspan="6">LLM Extraction (Qwen3-8B)</td></tr><tr><td>Model</td><td>Ovr.</td><td>Excav. Form</td><td></td><td>Ware</td><td>Pub.</td><td>Surf.</td><td>Ovr.</td><td>Excav. Form</td><td></td><td>Ware</td><td>Pub.</td><td>Surf.</td></tr><tr><td>olmOCR2</td><td>29.96</td><td>31.33</td><td>35.50</td><td>3.79</td><td>0.54</td><td>0.00</td><td>29.12</td><td>28.93</td><td>34.34</td><td>2.23</td><td>0.27</td><td>0.00</td></tr><tr><td>LightOnOCR</td><td>26.75</td><td>27.55</td><td>30.86</td><td>3.57</td><td>0.27</td><td>0.00</td><td>26.61</td><td>25.92</td><td>28.31</td><td>1.79</td><td>0.00</td><td>0.00</td></tr><tr><td>olmOCR2 (+)</td><td>91.73</td><td>96.05</td><td>90.02</td><td>90.18</td><td>85.91</td><td>61.11</td><td>90.76</td><td>95.96</td><td>88.86</td><td>88.84</td><td>82.11</td><td>57.41</td></tr><tr><td>LightOnOCR (+)</td><td>88.30</td><td>94.08</td><td>89.56</td><td>87.28</td><td>71.00</td><td>40.74</td><td>87.47</td><td>94.16</td><td>88.86</td><td>85.04</td><td>68.83</td><td>35.19</td></tr></table>

Field-level Performance. Fig. 5 shows field-level EMR for olmOCR2 and LightOnOCR; full results are in Table 6 (Appendix C.1). Baseline models handle measurements well (EMR 65–80 %) as numeric values are visually robust. Pottery form shows intermediate performance, with baseline models recovering it in 26–37 % of cases. Domain-specific fields fail entirely: Artefact Category and Publication Type reach near-zero EMR as OCR noise corrupts abbreviations and domain tokens beyond recognition (e.g. f/oxGK → florak). LoRA fine-tuning resolves this, with overall EMR exceeding 89 % for both models.

## 5.3 Field-extraction Results

Extraction Accuracy. Table 5 reports field extraction accuracy under REGEX and LLM (Qwen3-8B). At baseline, performance is low and nearly identical regardless of extraction method (29–30 % for olmOCR2, 26–27 % for LightOnOCR), pointing to the shared OCR input as the limiting factor rather than the extraction method itself. LoRA fine-tuning confirms this: once transcription quality improves, extraction accuracy follows, with olmOCR2 (+) outperforming LightOnOCR (+) (91.73 % vs. 88.30 % under REGEX) notably on Publication Type and Surface Treatment, with gaps of 14.91 and 20.37 percentage points.

Since REGEX strictly enforces the annotation convention, its primary failure mode is a character-level OCR error that breaks a match, whereas the LLM receives the same field convention as a prompt and can additionally deviate from it. Evaluating both extractors on ground truth transcriptions instead of OCR output separates the two efects: REGEX reaches 99.53 % overall accuracy, failing on the 16 expert-corrected records outside its rule coverage (Sec. 3.2), while the LLM achieves 96.76 %. This demonstrates that the baseline gap of 2.77 percentage points arises independently of OCR noise and reflects the LLM’s divergence from the annotation convention. On OCR output the two extractors difer by less than one point overall for both baseline and fine-tuned models, as OCR noise costs REGEX more than the LLM and partially ofsets this deviation. The REGEX–LLM gap is most pronounced on Publication Type for olmOCR2 (+) (85.91 % vs. 82.11 %), where the LLM frequently extracts only one reference from multi-reference entries, even though the prompt explicitly allows list output. The corresponding error breakdown and model overlap analysis are provided in Appendix C.3.

![](images/6992013bd92e21c87b57f93910253be2a77d46c8de333ac7211048c5fce18e7e.jpg)  
Fig. 6: Extraction performance across all fields. Error breakdown (%) for olmOCR2 (+) and LightOnOCR (+) across the seven metadata categories, showing correct extractions, omissions, and incorrect predictions, under REGEX and LLM extraction.

Error Modes. Fig. 6 breaks down extraction outcomes per field into three categories: correct, omission (field absent in the output), and incorrect (field present but wrong value). Publication Type shows the highest incorrect rate under both pipelines, driven by digit-level OCR errors producing plausible but wrong values. Fields with short or domain-specific tokens (Pottery Form, Artefact Category, Surface Treatment) show higher omission rates under REGEX, as corrupted tokens fall outside pattern coverage; under LLM extraction, these cases shift towards incorrect predictions, as the LLM selects the closest vocabulary match rather than returning null. This shift is most pronounced for Surface Treatment, where LightOnOCR (+) produces ∼41 % omissions under REGEX but ∼37 % incorrect predictions under LLM. Measurement shows notably higher omission under LLM than REGEX, suggesting the LLM more frequently fails to extract numeric values entirely. That the failures originate in OCR noise rather than extraction method choice is confirmed by model overlap (Table 8, Appendix C.3): 60 % of REGEX errors and 71 % of LLM errors are shared by both models.

Confidence Estimation. Cross-checking field predictions across two models and two extraction methods serves as a practical heuristic for archive-scale deployment. While overall field extraction accuracy already exceeds 87 %, fields on which all four pipeline combinations (olmOCR2 (+) and LightOnOCR (+), each paired with REGEX and LLM extraction) fully agree are correct in 98.7 % of cases. Extending acceptance to partial agreement, resolved by majority vote and field-specific rules, balances conditional precision against broader coverage by auto-accepting 56.4 % of documents at an overall precision of 95.5 %. Therefore, manual review concentrates on the remaining scans, where disagreement persists. A preliminary confidence scheme for partial disagreements is detailed in Appendix D.

## 6 Limitations and Future Work

While CENTURIA and the proposed pipeline demonstrate promising results for automated processing of handwritten pottery records, several limitations remain.

Remaining failure modes. Despite strong overall performance, systematic failure modes persist after fine-tuning. Short domain-specific tokens (e.g. surface treatment abbreviations) are prone to omissions under REGEX extraction, while digit-level errors in structured fields (excavation numbers, publication type) produce plausible but incorrect predictions across both pipelines. Percentage radius measurements (r\_a, r\_i) remain unresolved, as no pipeline reliably transcribes the % sign. Additionally, struck-through text in the original records is not filtered during transcription. OCR models read both the correction and the crossed-out text, which can introduce noise into downstream field extraction.

Quality validation and human evaluation. Cross-pipeline confidence estimation ofers practical triage: fields confirmed by all four pipelines are correct in 98.7 % of cases across the test set, and the 56.4 % of documents that are auto-accepted reach 95.5 % precision. Both figures are conditional on agreement-selected subsets rather than the full test set, and the pipelines are not fully independent (shared LoRA training data; REGEX shares the ground truth rule set), so agreement may overestimate reliability. While we explore a preliminary active learning scheme (Appendix D), we plan to formalise this into a full human-in-the-loop workflow. Future work will additionally evaluate time savings and error correction efort with domain experts in real archival settings.

Transferability and scope. As the first benchmark for HTR and field extraction on handwritten archaeological pottery documentation, CENTURIA opens the question of transferability across sites and documentation practices. The core challenges CENTURIA captures, namely handwritten abbreviations, spatially scattered annotations, and domain-specific vocabulary, are common to archaeological pottery documentation worldwide [18, 29, 48, 55, 56, 73]. CENTURIA itself spans six campaigns with variation in handwriting style, scribal conventions, and document condition, providing evidence that the approach remains efective under within-archive diversity, though cross-archive generalisation additionally depends on the target archive’s recording conventions, script, and language. The training split size of 57 is designed to reflect a realistic annotation budget in archival settings; how performance scales with annotation efort, and what minimum sufices for archives with diferent handwriting variation or recording conventions, remains an open question for future work. Cross-archive evaluation is non-trivial given restricted access to archaeological collections; we aim to extend CENTURIA to other sites where access permits. In the longer term, aggregating annotated records across multiple archives could support the development of a foundational OCR model for archaeological documentation, reducing the annotation burden for individual archives through shared pre-training.

## 7 Conclusion

We introduced CENTURIA, 507 annotated handwritten pottery records from the Carnuntum excavations, and used it to benchmark five OCR models on archaeological HTR and structured field extraction. On CENTURIA, transcription quality rather than extraction method governs end-to-end performance: zero-shot models fail (15–32 % SpACER-M), but LoRA fine-tuning on only 57 samples reduces transcription error below 1.5 % and lifts overall field accuracy above 87 %, after which structured metadata recovery follows from either extraction method (REGEX or LLM-based). Two further results make the approach practical for archival deployment. LightOnOCR (+) achieves comparable overall accuracy to olmOCR2 (+) at substantially lower computational cost, and crosspipeline agreement auto-accepts 56.4 % of documents at 95.5 % field precision, rising to 98.7 % on fields where all four pipelines agree, concentrating manual efort on genuine disagreements. Together, these show that a small, expertvalidated annotation efort is enough to begin converting decades of handwritten pottery documentation from Carnuntum into searchable, structured metadata, making archive-scale digitisation feasible for a collection that has so far remained inaccessible to computational analysis.

## Acknowledgements

This work was carried out within the project LEGION (machine LEarninGenabled Identification ofarchaeological Objects in the middle daNube river basin). The project is funded by the Austrian Academy of Sciences (OeAW) through the Heritage Science Austria 2.0 programme (grant number Heritage\_2024- $1 \mathcal { L } _ { - } L E G I O N )$

## Appendix

The appendix provides supplementary material supporting reproducibility and detailed analysis. Section A describes the CENTURIA sample selection strategy, annotation protocol, and data format. Section B details the inference setup, model checkpoints, post-processing steps, and LoRA fine-tuning hyperparameters used in all experiments. Section C reports full field-level OCR results, the complete adaptation strategy comparison, qualitative OCR examples, and finetuned model inference speed. Section D describes the Model Confidence and Smart Merge procedure with validation results. All prompts are collected in Section E.

## A Dataset Details

## A.1 Sample Selection Strategy

The sample is drawn from six excavation and field walking campaigns carried out at Carnuntum between 1976 and 2017 [36]. Text diversity is captured through handwriting variation across campaigns and documentation periods, character types (alphanumeric, special characters, German terminology, abbreviations such as Carn. and g/red), and annotation complexity. The dataset comprises eight sampling strata drawn from the following six campaigns:

## Excavation campaigns

– Gstettenbreite 1976 [33, 34], Sector 1–2 – 43 entries

– Gstettenbreite 1976 [33, 34], Sector 3–4 – 118 entries

Gstettenbreite 1976 [33, 34], Sector 5–6 – 121 entries

– Gstettenbreite 2017 [30] – 5 entries

Palastruine 1995 [34] – 5 entries

– Petroneller Burg (Limesgasse) 2016 [47] – 66 entries

Field walking campaigns

– Gstettenbreite 2017 [32] – 110 entries

– Käsemacherweide (Gladiatorenschule) 2012 [31] – 39 entries

Despite sharing a common annotation schema, campaigns difer in recording conventions, particularly regarding find identifiers (fn, quadrant, kiste) and measurement notation (RD/BD vs. radius and height), reflecting varied fieldwork practices across projects and time periods.

## A.2 Annotation Protocol

Bounding Box Conventions. Bounding boxes follow a strict left-to-right, topto-bottom reading order. Where handwriting is heavily slanted or irregularly spaced, boxes are split into individual words or short phrases. Non-overlapping boxes are preferred over pixel-perfect character enclosure, as region overlap causes character duplication in spatial metrics such as SpACER [11]. In addition to text regions, each scan carries one bounding box for the technical drawing. Where annotations are written across the drawing, text and drawing boxes overlap. This is the only allowed case and is captured as a stratification criterion shown in Table 1.

Transcription Conventions. Transcriptions preserve capitalisation, spacing, special characters, and abbreviations exactly as written. Where spacing is ambiguous, particularly around slashes and parentheses, no space is inserted. Measurements are transcribed with a single space between value and unit, for example, 12 cm.

Field Schema. Field annotation assigns labels by semantic content rather than spatial position, at the level of semantic units which may span multiple tokens. The seven categories are:

– Excavation: contextual find information including provenance (Carn. [Carnuntum]), project name, find number (FN [Fundnummer ]), stratigraphic unit (SE [Stratigrafische Einheit]), excavation area (FL [Fläche]), quadrant, and crate identifier (Kiste).

– Pottery Form: vessel typology using standard ceramic vocabulary, e.g., jug (Krug), bowl (Schüssel), lid (Deckel), grinding bowl (mortarium).

– Measurement: physical dimensions including rim diameter (RD), base di ameter (BD), and radius values $( R , R _ { A } , R _ { i } , R _ { M } )$ , optionally with a percentage indicating preserved rim segment.

– Artefact Category: firing technique recorded through standardised abbreviations such as f/ox GK, combining firing atmosphere and fabric grade.

– Publication Type: typological reference consisting of author name and plate or figure number, e.g., Gassner 1/10, Grünewald 1979 Taf. 33/4.

– Surface Treatment: surface finishing technique, e.g., Üz, Glasur, Grießbewurf.

Others: segments not matching any defined field, including scale annotations ([Maßstab] M 1:1 ), recording dates, and draughtsperson initials ([gezeichnet] gez. MBC ).

## A.3 Data Format and Structure

The dataset is provided as two JSON files. The transcription file contains, for each scan, the concatenated ground truth text and one entry per bounding box, storing position (x, y), size (w, h) in pixels, a reading-order index, and a drawing index. The field annotation file stores the same text alongside a structured field dictionary (id, text, fields) following the schema in Table 2.

## B Implementation Details

## B.1 Inference Setup

All experiments are conducted on NVIDIA GeForce RTX 3090 GPUs (24 GB VRAM each), as reported in Table 3. Models are loaded and executed using the Hugging Face Transformers library (version 5.7.0). All VLMs (olmOCR, olmOCR2, LightOnOCR) use bfloat16 precision, while Florence2 uses float16 and TrOCR runs in float32. Inference is performed with a batch size of 1 per image for all VLMs. TrOCR processes all text regions detected by EasyOCR (CRAFT) within a single image as a variable-size batch. olmOCR and olmOCR2 additionally use Flash Attention 2 for memory-eficient attention computation.

## B.2 Model Checkpoints

The following checkpoints are used in all experiments:

– TrOCR [49]: microsoft/trocr-large-handwritten

olmOCR [3]: allenai/olmOCR-7B-0225-preview

olmOCR2 [2]: allenai/olmOCR-2-7B-1025

– Florence2 [25]: florence-community/Florence-2-large

– LightOnOCR [51]: lightonai/LightOnOCR-2-1B

## B.3 HTR Post-processing

Post-processing applies the following cleaning steps to raw model output, in order:

– L<sup>A</sup>T<sub>E</sub>X markup (e.g. \frac, \text) is converted to plain text equivalents.

– Markdown image references (e.g. ![...](...)) are removed.

– Hyperlinks and URLs are removed.

– Model-generated commentary appended after the transcription is removed.

For olmOCR specifically, the transcription is extracted from the natural\_text field of the returned JSON object before cleaning. For TrOCR and Florence2, recognised text segments are concatenated in reading order prior to cleaning.

## B.4 LoRA Fine-tuning

Both olmOCR2 and LightOnOCR are fine-tuned for 2 epochs on the CENTURIA training set $( n = 5 7 )$ , with results also reported at 1 epoch. LoRA is applied with rank r = 16 (olmOCR2) and $r = 8 \ \mathrm { ( L i g h t O n O C R ) } , \alpha = 3 2$ , and dropout 0.05. Optimisation targets all linear projection layers using AdamW with learning rate $2 \times 1 0 ^ { - 4 }$ weight decay 0.01, gradient clipping at 1.0, a cosine decay schedule with 10 % warmup steps, and batch size 1. Images are resized to a maximum dimension of 1288 px (olmOCR2) and 768 px (LightOnOCR).

## B.5 Prompting Strategies

Zero-shot Instruction Prompt. The model input is augmented with a domain context prompt, which is applied equally to olmOCR2 and LightOnOCR. This replaces the default "Transcribe the document." instruction, providing structured vocabulary lists to guide the disambiguation of handwritten tokens, such as site abbreviations, excavation identifiers (e.g., crate, SE, FL and quadrant numbers), vessel forms, ware types, measurement conventions, surface treatments and publication author abbreviations, as well as other recurring tokens. The full prompt is provided in Fig. 10.

Few-shot Prompt. Few-shot examples are image-transcription pairs drawn from the CENTURIA training split. Each example is presented as a multiturn conversation: The model receives an image alongside the default instruction ("Transcribe the document.") and the expected transcription is provided as the assistant response. For $k \in \{ 1 , 5 , 8 \}$ , the first k examples from an ordered set of 8 are used. Examples are selected via stratified sampling from a pool covering all six campaigns, ensuring representative coverage across diverse excavation contexts, varying field combinations, and diferent label layouts. The 1-shot example is shown in Fig. 11.

Instruction + Few-shot Prompt. The combined strategy uses the domain context prompt (Fig. 10) as the task instruction in every conversation turn, including the 8 few-shot demonstration turns described in Fig. 11.

LLM Field Extraction Prompt. Field extraction uses Qwen3-8B in no-think mode (enable\_thinking=False), decoding greedily with max\_new\_tokens=400. The system prompt encodes the field schema as: (1) twelve hard rules constraining field assignment and output format, (2) per-field definitions with examples, and (3) a fixed JSON output template. Eight few-shot examples are provided, each providing a corrected transcription along with its expected field annotation. The efect of using corrupted versus corrected few-shot examples is evaluated in Sec. C.3. The full prompt is provided in Fig. 12.

## C Evaluation Details

## C.1 Per-field transcription analysis

Table 6 reports per-model field-level performance across all five baseline and two fine-tuned models. A notable outlier is olmOCR, which achieves the lowest pottery form EMR (8.6 %), omitting the field in 384 of 450 images, while reaching 64.4 % measurement ANLS. After LoRA fine-tuning, both models reach near-identical measurement EMR (91.67 % each), with olmOCR2 (+) leading on Artefact Category (89.51 % vs. 85.94 %) and Publication Type (86.18 % vs. 75.34 %).

Table 6: Field-level OCR performance. Overall is the per-document average of GT fields found in raw OCR output (test set, $n = 4 5 0 )$ . Measurement reports RD (rim diameter, $n = 3 7 2 )$ , evaluated with context-based matching to prevent false positives from crate and publication numbers. EMR uses exact substring matching; ANLS uses fuzzy matching $\left( \tau \ge 0 . 5 \right)$ . (+) indicates LoRA fine-tuned models. Best result bold, second best underlined.
<table><tr><td rowspan="2">Model</td><td colspan="2">Overall</td><td colspan="2">Pottery</td><td colspan="2">Measur.</td><td colspan="2">Art. Cat.</td><td colspan="2">Publ. Type</td></tr><tr><td>EMR ANLS ↑%</td><td>↑%</td><td>EMR ANLS ↑%</td><td>↑%</td><td>↑%</td><td>EMR ANLS ↑%</td><td>EMR ↑%</td><td>ANLS ↑%</td><td>EMR ↑%</td><td>ANLS ↑%</td></tr><tr><td>TrOCR</td><td>7.13</td><td>32.81</td><td>15.78</td><td>46.86</td><td>2.42</td><td>52.00</td><td>0.22</td><td>10.37</td><td>0.00</td><td>40.61</td></tr><tr><td>olmOCR</td><td>13.71</td><td>30.07</td><td>8.58</td><td>17.00</td><td>21.51</td><td>64.41</td><td>1.79</td><td>9.51</td><td>0.27</td><td>36.26</td></tr><tr><td>olmOCR2</td><td>30.55</td><td>57.43</td><td>37.35</td><td>68.21</td><td>79.30</td><td>92.98</td><td>2.46</td><td>19.31</td><td>1.90</td><td>64.07</td></tr><tr><td>Florence2</td><td>17.39</td><td>51.13</td><td>19.49</td><td>55.07</td><td>36.83</td><td>79.12</td><td>2.01</td><td>15.60</td><td>0.00</td><td>48.22</td></tr><tr><td>LightOnOCR</td><td>21.59</td><td>47.62</td><td>25.99</td><td>61.75</td><td>65.05</td><td>77.52</td><td>1.56</td><td>15.80</td><td>0.00</td><td>46.27</td></tr><tr><td>olmOCR2 (+)</td><td>92.08</td><td>95.29</td><td>91.18</td><td>93.98</td><td>91.67</td><td>96.78</td><td>89.51</td><td>95.64</td><td>86.18</td><td>97.08</td></tr><tr><td>LightOnOCR (+)</td><td>89.40</td><td>94.75</td><td>90.95</td><td>93.91</td><td>91.67</td><td>96.92</td><td>85.94</td><td>96.23</td><td>75.34</td><td>95.82</td></tr></table>

## C.2 Impact of improvement strategies on top-performing models

Table 7 reports the results for all adaptation strategies. Few-shot prompting reduces olmOCR2 error (SpACER-M 5.38 % at 5 shots) but collapses LightOnOCR into repetitive output loops (SpACER-M 48.71 % at 1 shot, 100 % at 5 shots). This behaviour is consistent with known repetition failure modes of autoregressive models [38]: the olmOCR authors report repetition as their most common failure and attribute it to inputs that deviate from the format the model was fine-tuned on [60], and narrow-domain few-shot examples can degrade generation more generally [70]. LightOnOCR’s RLVR-based training optimises for short, well-formed outputs; multi-image few-shot context is exactly such a deviation, causing degenerate repetition rather than guided transcription. LoRA fine-tuning avoids this instability entirely and dominates all prompting strategies for both models.

## C.3 Field extraction error analysis

The error breakdown per field is shown in Fig. 6 of the main paper. Table 8 details the underlying error patterns and root causes for both pipelines.

Shared errors. The majority of errors are shared by both models (356 of 592, 60 % under REGEX; 494 of 696, 71 % under LLM), confirming that failures originate in OCR noise rather than model or extraction method choice. Digit substitutions dominate across fields: crate numbers $( \mathrm { K i 2 } / 7 6 \to \mathrm { K i 5 0 } / 7 6 )$ , publication years (Grünewald 1979 → 1972), and measurement values (rd: 14 → rd: 16) all fail through visually similar handwritten digits. Rare domain tokens (Eingegl. Ker., Üz ) are consistently omitted by both models. Percentage radius measurements $( \mathbf { r } _ { - } \mathbf { a } , \mathbf { r } _ { - } \mathbf { i } )$ fail across all four pipelines, as neither model reliably transcribes the % sign.

Table 7: Impact of adaptation strategies on top-performing models. Baseline shows zero-shot performance. Best result bold, second best underlined.
<table><tr><td>Model / Strategy</td><td>SpACER-M BoW F1 EMR ANLS ↓(%)</td><td>↑(%)</td><td>↑(%)</td><td>↑(%)</td></tr><tr><td>olmOCR2</td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline (zero-shot)</td><td>15.58</td><td>56.74</td><td>30.55</td><td>57.43</td></tr><tr><td>Zero-shot instruction prompting</td><td>6.77</td><td>83.08</td><td>74.64</td><td>84.99</td></tr><tr><td>Few-shot (1 shot)</td><td>8.48</td><td>72.20</td><td>51.12</td><td>79.35</td></tr><tr><td>Few-shot (5 shots)</td><td>5.38</td><td>84.44</td><td>68.97</td><td>87.51</td></tr><tr><td>Few-shot (8 shots)</td><td>5.84</td><td>85.65</td><td>75.27</td><td>87.72</td></tr><tr><td>Instruction + few-shot (1 shot)</td><td>7.29</td><td>85.52</td><td>71.67</td><td>86.39</td></tr><tr><td>Instruction + few-shot (5 shots)</td><td>7.64</td><td>84.39</td><td>73.06</td><td>86.10</td></tr><tr><td>LoRA fine-tuning (1 epoch)</td><td>2.33</td><td>94.57</td><td>88.72</td><td>93.83</td></tr><tr><td>LoRA fine-tuning (2 epochs)</td><td>1.11</td><td>96.24</td><td>92.08</td><td>95.29</td></tr><tr><td>LightOnOCR</td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline (zero-shot)</td><td>18.25</td><td>50.98</td><td>21.59</td><td>47.62</td></tr><tr><td>Zero-shot instruction prompting</td><td>14.13</td><td>53.71</td><td>35.27</td><td>65.14</td></tr><tr><td>Few-shot (1 shot)</td><td>48.71</td><td>43.57</td><td>28.02</td><td>67.24</td></tr><tr><td>Few-shot (5 shots)</td><td>100.00</td><td>26.78</td><td>19.73</td><td>63.02</td></tr><tr><td>Few-shot (8 shots)</td><td>98.88</td><td>26.48</td><td>17.43</td><td>48.05</td></tr><tr><td>LoRA fine-tuning (1 epoch)</td><td>1.59</td><td>91.63</td><td>84.08</td><td>92.25</td></tr><tr><td>LoRA fine-tuning (2 epochs)</td><td>1.21</td><td>94.80</td><td>89.40</td><td>94.75</td></tr></table>

Model-specific errors. Among errors unique to one model, LightOnOCR is responsible for the majority: 77 % under REGEX (182 of 236) and 73 % under LLM (147 of 202). Its dominant weakness is Publication Type (67 REGEX, 52 LLM unique errors), where KE inventory references are truncated (Adler-Wölfl Taf. 96 KE 2790 → Taf. 96) and multi-reference entries are cut of. Under REGEX, it additionally omits Artefact Category and Surface Treatment tokens that fall outside pattern coverage. olmOCR2 produces fewer unique errors but shows a systematic weakness in Pottery Form, truncating compound forms (Deckel/Schüssel → Schüssel) or omitting the field entirely (18 REGEX, 17 LLM unique errors).

Efect of few-shot example quality. At inference, the LLM operates on noisy OCR predictions rather than clean transcriptions, so few-shot examples built from corrupted OCR output paired with the correct field annotations match the real inference condition more closely. We therefore compare both variants. Table 9 compares LLM extraction accuracy using corrupted versus corrected few-shot examples. The largest gain appears on Publication Type, where olmOCR2 (+) improves from 74.53 % to 82.11 % and LightOnOCR (+) from 62.87 % to 68.83 %, suggesting that accurate demonstrations of multi-reference entries help the LLM learn the expected list output format. For all other fields, corrupted and corrected examples show negligible diferences. For the fine-tuned models, corrected examples are equal or better on every field; on the zero-shot baselines, the two difer by under one point in either direction. All pipeline results reported in the main paper use corrected few-shot examples.

Table 8: Characteristic extraction errors under REGEX and LLM extraction. REGEX: 592 errors, 60 % shared; LLM: 696 errors, 71 % shared. Both: error shared by olmOCR2 (+) and LightOnOCR (+).
<table><tr><td>Field</td><td>Error type</td><td>Pipeline</td><td>GT → Prediction</td><td>Root cause</td><td>Scope</td></tr><tr><td rowspan="2">Excavation</td><td>Crate digit shift</td><td>Both</td><td>Ki 2/76 → Ki 50/76</td><td>Digit misread</td><td>Both</td></tr><tr><td>Slash drop</td><td>LLM</td><td>Ki 1/76 → Ki 1776</td><td>Delimiter OCR error</td><td>Both</td></tr><tr><td rowspan="2">Pottery Form</td><td>Misclassif.</td><td>Both</td><td>Topf → Schüssel</td><td>Visual form ambi- Both guity</td><td></td></tr><tr><td>Compound truncation</td><td>Both</td><td>Deckel/Schüssel → Schüs- sel</td><td>Prefix dropped</td><td>olmOCR</td></tr><tr><td rowspan="2">Artefact Cat.</td><td>Oxidation confusion</td><td>Both</td><td>g/red GK → f/ox GK</td><td>Firing code OCR: error</td><td>Both</td></tr><tr><td>Partial ex- traction</td><td>LLM</td><td>f/ox Glasierte GK → f/ox</td><td>Token truncation</td><td>olmOCR</td></tr><tr><td rowspan="4">Publication</td><td>Year digit er-1 ror</td><td>Both</td><td>Grünewald 1979 → 1972</td><td>Single digit OCR</td><td>Both</td></tr><tr><td>Multi-ref confusion</td><td>REGEX</td><td>Adler-Wölf 2010 . . . → Gassner 3/4</td><td>Ref. misassign- ment</td><td>Both</td></tr><tr><td>KE no. trun- LLM cation</td><td></td><td>Taf. 98 KE 2847 Taf.98/</td><td>Ref. cut off</td><td>Both</td></tr><tr><td>KE ref. trun- cation</td><td>REGEX</td><td>Adler-Wölfl Taf. 96 KE 2790 → Taf. 96</td><td>Ref. cut off</td><td>LightOn</td></tr><tr><td></td><td>Measurement Value shift Abbrev.</td><td>Both</td><td>rd: 14 → rd: 16 Üz → null</td><td>Digit OCR error un-</td><td>Both</td></tr><tr><td rowspan="2">Surface Treat.</td><td>&#x27;omission Partial</td><td>Both</td><td>Üz → Üz-</td><td>Short token matched</td><td>Both</td></tr><tr><td>cor- ruption</td><td>LLM</td><td></td><td>Trailing OCR</td><td>char LightOn</td></tr><tr><td>Others</td><td>Noise / omis- Both sion</td><td></td><td>SPA → (Anm |SPA) / null</td><td>Layout debris</td><td>Both</td></tr></table>

## C.4 Qualitative transcription examples

Fig. 7 contrasts a typical high-quality scan with one of the two degraded images identified in the dataset (0.4 %). On the high-quality example, both fine-tuned models transcribe all fields correctly, including the publication\_type (Petznek Taf. 20/1862) and artefact\_category (f/red GG). On the degraded scan, faded ink and low contrast introduce errors that afect both models and complicate human annotation: M1:1 is misread as H11 cm (olmOCR2) and omitted entirely by LightOnOCR, R = 4,3 7% loses the percentage in both models and is additionally read as R = 43 cm by olmOCR2, and H = 1,6 cm is read as H = 16 cm by both models. The rarity of such cases in CENTURIA (2 of 507 images) limits their impact on aggregate metrics, but illustrates the dificulty of low-contrast handwritten domain text.

Table 10 shows field extraction results for the two qualitative examples in Fig. 7 across all four model-annotation combinations. On the high-quality scan, all fields are extracted correctly by both models under both methods. On the degraded scan, all four combinations still assign every field to the correct category. However, three fields carry wrong values because the extractors reproduce what the OCR read: provenance is read as Carn. although the record shows Car., r loses its percentage in both models and is additionally corrupted to 43 by olmOCR2, and h loses its decimal point in both models (1,6 → 16). Only fn fails at the extraction stage for LightOnOCR under both methods, as spacing artefacts in the OCR output (A 14 / 2 vs. A 14/2 ) break the REGEX pattern and cause LLM misassignment. This confirms that OCR-level errors are the primary failure mode, not the annotation method.

Table 9: Field extraction accuracy (%) – LLM extraction with corrupted vs. corrected few-shot examples (Qwen3-8B). Overall is the per-document average across all fields. Excavation aggregates provenance, crate, project, FN, SE, FL, and quadrant (weighted by n). (+) denotes LoRA fine-tuned models. Bold: best per column.
<table><tr><td></td><td colspan="6">LLM Corrupted Few-Shots</td><td colspan="6">LLM 一 Corrected Few-Shots</td></tr><tr><td>Model</td><td>Ovr. Excav. Form</td><td></td><td></td><td>Ware</td><td>e Pub. Surf.</td><td></td><td>Ovr.</td><td>Excav. Form</td><td></td><td>Ware</td><td>Pub.</td><td>Surf.</td></tr><tr><td>olmOCR2</td><td>29.25</td><td>29.01</td><td>34.80</td><td>1.79</td><td>0.00</td><td>0.00</td><td>29.12</td><td>28.93</td><td>34.34</td><td>2.23</td><td>0.27</td><td>0.00</td></tr><tr><td>LightOnOCR</td><td>26.79</td><td>25.49</td><td>29.00</td><td>2.01</td><td>0.27</td><td>0.00</td><td>26.61</td><td>25.92</td><td>28.31</td><td>1.79</td><td>0.00</td><td>0.00</td></tr><tr><td>olmOCR2 (+)</td><td>89.71</td><td>95.79</td><td>88.63</td><td>88.39</td><td>74.53</td><td>57.41</td><td>90.76</td><td>95.96</td><td>88.86</td><td>88.84</td><td>82.11 57.41</td><td></td></tr><tr><td>LightOnOCR (+)</td><td>86.68</td><td>93.99</td><td>88.86</td><td>85.04</td><td>62.87 35.19</td><td></td><td>87.47</td><td>94.16</td><td>88.86</td><td>85.04</td><td>68.83 35.19</td><td></td></tr></table>

Table 10: Field extraction results for the qualitative examples in Fig. 7. R = REGEX, L = LLM (Qwen3-8B) extraction; subscripts denote olmOCR2 (+) and LightOnOCR (+). ✓ correct; ✗ extracted value shown in italics.
<table><tr><td>Field</td><td>GT</td><td> $\mathbf { R _ { o l m } }$ </td><td> $\mathbf { L _ { o l m } }$ </td><td> $\mathbf { R } _ { \mathrm { l i t } }$ </td><td> ${ \bf L } _ { \mathrm { l i t } }$ </td></tr><tr><td colspan="6">High-quality scan — all fields correctly extracted</td></tr><tr><td>provenance</td><td>Carn.</td><td></td><td></td><td></td><td></td></tr><tr><td>kiste</td><td>Ki179/76</td><td></td><td></td><td></td><td></td></tr><tr><td>pottery_form</td><td>Deckel</td><td></td><td></td><td></td><td></td></tr><tr><td>artefact_category</td><td>f/red GG</td><td></td><td></td><td></td><td></td></tr><tr><td>publication_type</td><td>Petznek</td><td></td><td></td><td></td><td></td></tr><tr><td>rd</td><td>Taf.20/. . . 10</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">Degraded scan — OCR errors propagate to extracted values</td></tr><tr><td>provenance</td><td>Car.</td><td></td><td>X Carn. X Carn. X Carn.</td><td></td><td>X Carn.</td></tr><tr><td>project</td><td>GS 2012</td><td></td><td></td><td></td><td></td></tr><tr><td>fn</td><td>A14/2</td><td></td><td></td><td>Xnull</td><td>XA14/2</td></tr><tr><td>artefact_category</td><td>GK red.</td><td></td><td></td><td></td><td></td></tr><tr><td>publication_type</td><td>Petznek 12.3</td><td></td><td></td><td></td><td></td></tr><tr><td>r</td><td>4,3 (7%)</td><td>×43</td><td>X43</td><td>×4,3</td><td>×4,3</td></tr><tr><td>h</td><td>1,6</td><td>×16</td><td>×16</td><td>×16</td><td>×16</td></tr></table>

![](images/b2fee6445e13f034ce14acc3ad67ab14e3fb5c318f5f5ce556c51cb01837cee2.jpg)  
Fig. 7: Qualitative OCR examples. Ground truth (black), olmOCR2 (+) (orange), and LightOnOCR (+) (blue). On the high-quality scan, both models transcribe all fields correctly. On the degraded scan, low contrast and faded ink cause systematic errors and challenge human annotation as well.

## C.5 Inference speed and output length

LightOnOCR (+) is substantially faster than its baseline (1.42 s vs. 2.95 s per document). This is consistent with an 86 % reduction in average output length (72 vs. 509 characters, n = 450). Domain adaptation suppresses verbose artefacts present in baseline outputs (LAT<sub>E</sub>X notation, Markdown image links, HTML tables, and explanatory commentary). This results in concise, label-like transcriptions. olmOCR2 (+) shows no speed improvement over its baseline (2.26 s vs. 2.19 s). Its baseline outputs are already concise (77 characters on average), leaving no room for length-based gains after fine-tuning (Table 11). Figs. 8 and 9 illustrate the output reduction for a single document.

Table 11: Average output length per document on the CENTURIA test set (n = 450). Values are mean ± standard deviation across documents.
<table><tr><td>Model</td><td>Avg. chars</td></tr><tr><td>olmOCR2</td><td> $7 7 . 1 \pm 3 6 . 0$ </td></tr><tr><td>olmOCR2 (+)</td><td> $7 2 . 0 \pm 2 1 . 4$ </td></tr><tr><td>LightOnOCR</td><td> $5 0 8 . 7 \pm 2 7 7 . 9$ </td></tr><tr><td>LightOnOCR (+)</td><td> $7 2 . 1 \pm 2 1 . 8$ </td></tr></table>

\$\mathcal{C}\_{om}\$ / \$ki\ 1|76\$ / \$Schiussel\ Possner\ 3/4\$ /   
\$PD = 16cm\$ / Flozell / ![image](image\_1.png) /   
Note: The image contains a simple sketch of a rectangular   
object with a vertical slot at one end ..  
Fig. 8: LightOnOCR baseline output (2,125 chars). Verbose output containing L<sup>A</sup>T<sub>E</sub>X notation, a Markdown image link, and model commentary.  
Fig. 9: LightOnOCR (+) fine-tuned output (53 chars) for the same document as Fig. 8: concise, label-like transcription.

## D Model Confidence and Smart Merge

We estimate annotation confidence by cross-checking field predictions across four pipeline combinations (olmOCR2 (+) and LightOnOCR (+), each with REGEX and LLM extraction). Each field is assigned a confidence level based on agreement after string normalisation (Table 12). Fields where not all four pipelines agree are resolved by Smart Merge: majority vote when 3 of 4 agree (HIGH), field-specific rules for two-against-two splits (INFERRED), or manual review when no rule applies (REVIEW). For two-against-two splits, a field-specific rule selects the more reliable source (Table 13). These splits can arise either between the two OCR models or between the two extraction methods. For example, if olmOCR2 (+) reads rd = 11 while LightOnOCR (+) reads rd = 16 (digit-shift), Smart Merge selects the olmOCR2 (+) value, matching the ground truth. Publication type year errors and percentage radius measurements $\left( \mathbf { r _ { - } } \mathbf { a } / \mathbf { r _ { - } } \mathbf { i } \right)$ are always sent to review, as no pipeline resolves them reliably.

A document is auto-accepted once every field is CONFIRMED, HIGH, or IN-FERRED. This holds for 254 of 450 images (56.4 %). Across the full test set, most fields reach full agreement (CONFIRMED: 2,524 fields, 98.7 % precision), while HIGH and INFERRED together cover only 2.7 % of fields (206 of 7,650) at lower precision (55.9 % and 43.0 %). Within the 254 auto-accepted documents, the auto-resolved fields reach 95.5 % precision, with the auto-resolved HIGH and INFERRED fields accounting for most of the residual error. The remaining 43.6 % of images contain an unresolved conflict and require manual review.

Table 12: Smart Merge confidence levels. Precision computed over all 450 test documents. Auto-accepted: 254 images (56.4 %); precision 95.5 % computed over all fields in auto-accepted documents. NULL: field not present in the record and all four pipelines correctly predict no value (4,649 fields).
<table><tr><td>Level</td><td>Prec.</td><td>Fields</td><td>Definition</td></tr><tr><td>CONFIRMED</td><td>98.7%</td><td>2,524</td><td>All 4 pipelines agree</td></tr><tr><td>HIGH</td><td>55.9%</td><td>92</td><td>3 of 4 pipelines agree</td></tr><tr><td>INFERRED</td><td>43.0%</td><td>114</td><td>2-vs-2 split, rule resolved</td></tr><tr><td>REVIEW</td><td></td><td>271</td><td>Unresolved conflict</td></tr></table>

Table 13: Characteristic pipeline errors and Smart Merge rules.
<table><tr><td>Pipeline</td><td>Field</td><td>Error type</td><td>Resolution</td></tr><tr><td>REGEX (both)</td><td>Pottery, Art. Cat.</td><td>Token omission</td><td>Prefer LLM</td></tr><tr><td>LLM (both)</td><td>Surface Treatment</td><td>Colour hallucination</td><td>Prefer REGEX</td></tr><tr><td>LLM (both)</td><td>Publication Type</td><td>Multi-ref. missed</td><td>Prefer REGEX</td></tr><tr><td>LightÓnOCR</td><td>Excavation, RD</td><td>Higher digit-shift rate</td><td>Prefer olmOCR2</td></tr><tr><td>olmOCR2</td><td>Pub. Type (year)</td><td>Single digit misread</td><td>Manual review</td></tr><tr><td>All pipelines</td><td>r_a/r_i</td><td>% sign unrecognised</td><td>Manual review</td></tr></table>

## E Prompts

![](images/448041f36ea0f011458a1997acac2c814e10c060bc0c383b2e76c85e2c5c50f9.jpg)

Fig. 10: Zero-shot Instruction prompt for OCR. Domain-specific vocabulary lists guide the model to resolve ambiguous handwriting, covering provenance tokens, excavation identifiers, vessel forms, ware types, measurements, surface treatments, and publication abbreviations.  
![](images/262ccd5e23ff678762d5547cdf45bff37591daa7bbff86e0eddb043c07e100b2.jpg)  
Fig. 11: 1-Shot few-shot example for OCR prompting. An image-transcription pair from the CENTURIA training set is added to the model input to demonstrate the expected output format and domain conventions.

![](images/4b2309d5dd4d8d5f54ab95d1b7bc373b5fc33037e62d56bd4becc89875f15c28.jpg)  
Fig. 12: System prompt for LLM-based field extraction (Qwen3-8B). The prompt defines the extraction schema, domain constraints, and output format.

## References

1. Abele, L., Anders, G., Aydın, T., Buder, J., Fischer, H., Kimmel, D., Huf, M.: ArchiveGPT: A human-centered evaluation of using a vision language model for image cataloguing. Humanities and Social Sciences Communications 13(1), 1241 (2026). https://doi.org/10.1057/s41599-026-08367-6

2. Allen Institute for AI: olmOCR-2-7B-1025 – Model Checkpoint. https : / / huggingface.co/allenai/olmOCR-2-7B-1025 (2025)

3. Allen Institute for AI: olmOCR-7B-0225-preview – Model Checkpoint. https:// huggingface.co/allenai/olmOCR-7B-0225-preview (2025)

4. Anichini, F., Banterle, F., Buxeda i Garrigós, J., Callieri, M., Dershowitz, N., Dubbini, N., Lucendo Diaz, D., Evans, T., Gattiglia, G., Green, K., Gualandi, M.L., Hervas, M.A., Itkin, B., Madrid i Fernandez, M., Miguel Gascón, E., Remmy, M., Richards, J., Scopigno, R., Vila, L., Wolf, L., Wright, H., Zallocco, M.: Developing the ArchAIDE Application: A Digital Workflow for Identifying, Organising and Sharing Archaeological Pottery Using Automated Image Recognition. Internet Archaeology 52 (2020). https://doi.org/10.11141/ia.52.7

5. Baek, Y., Lee, B., Han, D., Yun, S., Lee, H.: Character Region Awareness for Text Detection. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 9357–9366. IEEE (2019). https://doi.org/10.1109/ CVPR.2019.00959

6. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923 (2025). https://doi.org/10.48550/arXiv.2502.13923

7. Barzelay, U., Azulai, O., Shapira, I., Friedman, I., Abo Dahood, F., Lee, M., Daniels, A.: VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 7368–7376 (2026)

8. Bellat, M., Orellana Figueroa, J.D., Reeves, J.S., Taghizadeh-Mehrjardi, R., Tennie, C., Scholten, T.: Machine Learning Applications in Archaeological Practices: A Review. Journal of Computer Applications in Archaeology 8(1), 282–321 (2025). https://doi.org/10.5334/jcaa.201

9. Bergamaschi, S., De Nardis, S., Martoglia, R., Ruozzi, F., Sala, L., Vanzini, M., Vigliermo, R.A.: Novel Perspectives for the Management of Multilingual and Multialphabetic Heritages through Automatic Knowledge Extraction: The Digital-Maktaba Approach. Sensors 22(11), 3995 (2022). https://doi.org/10.3390/ s22113995

10. Biten, A.F., Tito, R., Mafla, A., Gomez, L., Rusiñol, M., Jawahar, C., Valveny, E., Karatzas, D.: Scene Text Visual Question Answering. In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4290–4300 (2019). https://doi.org/10.1109/ICCV.2019.00439

11. Bourne, J., Simbeye, M., Nockels, J.: The Character Error Vector: Decomposable errors for page-level OCR evaluation. arXiv preprint arXiv:2604.06160 (2026). https://doi.org/10.48550/arXiv.2604.06160

12. Buechner-Matthews, S., David, R.: Drawing and Photography. In: David, R. (ed.) Concise Manual for Ceramic Studies, pp. 71–77. Africae, Soleb (2022). https: //doi.org/10.4000/books.africae.5780

13. Cardarelli, L.: PyPottery: A collection of python tools for archaeological ceramics (2025), https://github.com/lrncrd/PyPottery, accessed: 2026-06-26

14. Cardarelli, L.: PyPotteryInk: One-step difusion model for sketch to publicationready archaeological drawings. Journal of Cultural Heritage 74, 300–310 (2025). https://doi.org/10.1016/j.culher.2025.06.016

15. Cardarelli, L.: PyPotteryLens: An Open-Source Deep Learning Framework for Automated Digitisation of Archaeological Pottery Documentation. Digital Applications in Archaeology and Cultural Heritage 38, e00452 (2025). https://doi.org/ 10.1016/j.daach.2025.e00452

16. Cardarelli, L.: PyPotteryScan. https://github.com/lrncrd/PyPotteryScan/ (2026), accessed: 2026-08-13

17. Clérice, T., Pinche, A., Vlachou-Efstathiou, M., Chagué, A., Camps, J.B., Levenson, M.G., Brisville-Fertin, O., Boschetti, F., Fischer, F., Gervers, M., et al.: CAT-MuS Medieval: A Multilingual Large-Scale Cross-Century Dataset in Latin Script for Handwritten Text Recognition and Beyond. In: International Conference on Document Analysis and Recognition. pp. 174–194. Springer Nature Switzerland, Cham (2024). https://doi.org/10.1007/978-3-031-70543-4\_11

18. Collett, L.: Introduction to Drawing Archaeological Pottery, CIfA Professional Practice Paper, vol. 10. Chartered Institute for Archaeologists (2017)

19. Colutto, S., Kahle, P., Hackl, G., Mühlberger, G.: Transkribus. A Platform for Automated Text Recognition and Searching of Historical Documents. 2019 15th International Conference on eScience (eScience) pp. 463–466 (2019). https://doi. org/10.1109/eScience.2019.00060

20. Constum, T., Kempf, N., Paquet, T., Tranouez, P., Chatelain, C., Brée, S., Merveille, F.: Recognition and Information Extraction in Historical Handwritten Tables: Toward Understanding Early 20th Century Paris Census. In: International Workshop on Document Analysis Systems. pp. 143–157. Springer International Publishing, Cham (2022). https://doi.org/10.1007/978-3-031-06555-2\_10

21. Crosilla, G., Klic, L., Colavizza, G.: Benchmarking large language models for handwritten text recognition. Journal of Documentation 81(7), 334–354 (2025). https://doi.org/10.1108/JD-03-2025-0082

22. Demján, P., Pavúk, P., Roosevelt, C.H.: Laser-Aided Profile Measurement and Cluster Analysis of Ceramic Shapes. Journal of Field Archaeology 48(1), 1–18 (2023). https://doi.org/10.1080/00934690.2022.2128549

23. Di Angelo, L., Di Stefano, P., Guardiani, E.: A Review of Computer-based Methods for Classification and Reconstruction of 3D high-density Scanned Archaeological Pottery. Journal of Cultural Heritage 56, 10–24 (2022). https://doi.org/10. 1016/j.culher.2022.05.001

24. Fletcher, E.C.: Creating a Software Methodology to Analyze and Preserve Archaeological Legacy Data. Advances in Archaeological Practice 11(2), 139–151 (2023). https://doi.org/10.1017/aap.2022.44

25. Florence-community: Florence-2-Large – Model Checkpoint. https : //huggingface.co/florence-community/Florence-2-large (2024)

26. Fornés, A., Romero, V., Baró, A., Toledo, J.I., Sánchez, J.A., Vidal, E., Lladós, J.: ICDAR2017 Competition on Information Extraction in Historical Handwritten Records. In: 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR). vol. 1, pp. 1389–1394. IEEE (2017). https://doi.org/10. 1109/ICDAR.2017.227

27. Garrido-Munoz, C., Rios-Vila, A., Calvo-Zaragoza, J.: Handwritten Text Recognition: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 48(4), 4367–4387 (2026). https://doi.org/10.1109/TPAMI.2025.3646002

28. Gattiglia, G.: Managing Artificial Intelligence in Archeology. An overview. Journal of Cultural Heritage 71, 225–233 (2025). https://doi.org/10.1016/j.culher. 2024.11.020

29. Girotto, E.: Ceramic Illustration. In: MacGinnis, J., Rey, S. (eds.) Laying the Foundations: Manual of the British Museum Iraq Scheme Archaeological Training Programme, pp. 213–230. Archaeopress (Jan 2022). https://doi.org/10.2307/ jj.15136034.26

30. Gugl, C., Humer, F., Radbauer, S., Schindel, N., Wallner, M., Zabehlicky, H.: Archäologische Prospektion und Ausgrabungen in der Flur Gstettenbreite: Gräber und Straßenverläufe im westlichen Vorfeld der Carnuntiner Zivilstadt. Carnuntum Jahrbuch 2019, 11–53 (2020). https://doi.org/10.1553/cjb\_2019s11

31. Gugl, C., Radbauer, S.: Der Oberflächensurvey im Bereich der sog. Gladiatorenschule in Carnuntum. Ein Beitrag zur Siedlungsentwicklung der Südperipherie der Zivilstadt. Carnuntum Jahrbuch 2016, 117–148 (2017). https://doi.org/10. 1553/cjb\_2016s117

32. Gugl, C., Radbauer, S., Wallner, M.: Archäologische Prospektion 2012–2017 in der Flur Gstettenbreite – ein Beitrag zur Entwicklung vorstädtischer Siedlungszonen in Carnuntum. Carnuntum Jahrbuch 2018, 47–85 (2019). https://doi.org/10. 1553/cjb\_2018s47

33. Gugl, C., Radbauer, S., Wallner, M., Humer, F., Pollhammer, E., Neubauer, W.: Vor den Toren der Stadt – Struktur und Entwicklung des westlichen Suburbiums der Carnuntiner Zivilstadt. Neubewertung der Notgrabung 1976 aufgrund der geophysikalischen Messungen 2012–2015. Carnuntum Jahrbuch 2020, 37–84 (2021). https://doi.org/10.1553/cjb\_2020s37

34. Gugl, C., Radbauer, S., Wallner, M., Pollhammer, E.: Ein Querschnitt durch die Stadt – Teil 1: Chronologie und Struktur der Carnuntiner Zivilstadt auf Basis von geophysikalischen Messungen und der Notgrabung 1976. Carnuntum Jahrbuch 2022, 55–100 (2023). https://doi.org/10.1553/cjb\_2022s55

35. Gugl, C., Vadeanu, C.: Archaeological research in Carnuntum: Selected data, https://hdl.handle.net/21.11115/0000-000F-9C36-5, accessed: 2026-08-01

36. Gugl, C., Wallner, M., Pollhammer, E.: Carnuntum – Eine antike Siedlungsagglomeration an der mittleren Donau. In: Horvat, J., Groh, S., Strobel, K., Belak, M. (eds.) Roman Urban Landscape: Towns and Minor Settlements from Aquileia to the Danube, Opera Instituti Archaeologici Sloveniae, vol. 47, pp. 377–401. Založba ZRC (2024). https://doi.org/10.3986/9789610508281\_19

37. Hand, D.J.: Dark Data: Why What You Don’t Know Matters. Princeton University Press (2020). https://doi.org/10.1515/9780691198859

38. Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y.: The Curious Case of Neural Text Degeneration. arXiv preprint arXiv:1904.09751 (2019). https://doi.org/ 10.48550/arXiv.1904.09751

39. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-Rank Adaptation of Large Language Models. In: International Conference on Learning Representations (2022), https://openreview.net/forum? id=nZeVKeeFYf9

40. Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F.: LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In: Proceedings of the 30th ACM International Conference on Multimedia. p. 4083–4091 (2022). https://doi.org/ 10.1145/3503161.3548112

41. Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., Jawahar, C.: ICDAR2019 competition on scanned receipt OCR and information extraction. In: 2019 Interna-

tional Conference on Document Analysis and Recognition (ICDAR). pp. 1516–1520 (2019). https://doi.org/10.1109/ICDAR.2019.00244

42. Hüttner, L., Mayr, M., Gorges, T., Wu, F., Seuret, M., Maier, A., Christlein, V.: Low-Rank Adaptation vs. Fine-Tuning for Handwritten Text Recognition. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW). pp. 1233–1242 (2025). https://doi.org/10.1109/WACVW65960.2025. 00146

43. JaidedAI: EasyOCR. https://github.com/JaidedAI/EasyOCR (2020), accessed: 2026-08-13

44. Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., Hwang, W., Yun, S., Han, D., Park, S.: OCR-Free Document Understanding Transformer. In: Computer Vision – ECCV 2022. pp. 498–517. Springer Nature Switzerland (2022). https: //doi.org/10.1007/978-3-031-19815-1\_29

45. Klein, K., Muller, A., Wohde, A., Gorelik, A.V., Heyd, V., Lämmel, R., Diekmann, Y., Brami, M.: An AI-Assisted Workflow for Object Detection and Data Collection from Archaeological Catalogues. Journal of Archaeological Science 179, 106244 (2025). https://doi.org/10.1016/j.jas.2025.106244

46. Kohút, J., Hradiš, M.: Fine-tuning Is a Surprisingly Efective Domain Adaptation Baseline in Handwriting Recognition. In: Document Analysis and Recognition - ICDAR 2023. pp. 269–286. Springer Nature Switzerland (2023). https://doi. org/10.1007/978-3-031-41685-9\_17

47. Konecny, A., Humer, F., Radbauer, S., Gugl, C., Igl, R., Fuchshuber, N.: Zwei Infrastruktureinrichtungen des römischen Carnuntum: der Aquädukt in der Flur Gstettenbreite und die Limesstraße. Carnuntum Jahrbuch 2020, 11–36 (2021). https://doi.org/10.1553/cjb\_2020s11

48. Kosak, H., Borlin, A., Hamel, H., Möller, H.: Instruction to the Archaeological Drawing of Ceramic Sherds. Summer School in Umm Qais 2023 (2023), https: //tutorials.idai.world/, accessed: 2026-08-13

49. Li, M., Lv, T., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., Wei, F.: TrOCR-Large-Handwritten – Model Checkpoint. https://huggingface.co/microsoft/ trocr-large-handwritten (2021), fine-tuned on the IAM handwriting dataset, accessed: 2026-08-13

50. Li, M., Lv, T., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., Wei, F.: TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models. Proceedings of the AAAI Conference on Artificial Intelligence 37(11), 13094–13102 (2023). https://doi.org/10.1609/aaai.v37i11.26538

51. LightOn AI: LightOnOCR-2-1B – Model Checkpoint. https://huggingface.co/ lightonai/LightOnOCR-2-1B (2026), accessed: 2026-08-13

52. Marti, U.V., Bunke, H.: The IAM-database: an English sentence database for offline handwriting recognition. International Journal on Document Analysis and Recognition 5(1), 39–46 (2002). https://doi.org/10.1007/s100320200071

53. McManamon, F.P., Kintigh, K.W., Ellison, L.A., Brin, A.: tDAR: A Cultural Heritage Archive for Twenty-First-Century Public Outreach, Research, and Resource Management. Advances in Archaeological Practice 5(3), 238–249 (2017). https://doi.org/10.1017/aap.2017.18

54. Mistral AI: Mistral Small 3.1. https://mistral.ai/news/mistral-small-3-1/ (2025), accessed: 2026-08-13

55. Moreno Martín, A., Quixal Santos, D.: Bordes, bases e informes: el dibujo arqueológico de material cerámico y la fotografía digital. Arqueoweb: Revista sobre Arqueologia en Internet 14(1), 178–214 (2013), https://webs.ucm.es/info/ arqueoweb/pdf/14/Moreno178-214.pdf, accessed: 2026-06-22

56. Okayama City Buried Cultural Properties Center: Shutsudobutsu jissoku manyuaru (ent¯o haniwa-hen) (2018), https://www.city.okayama.jp/kurashi/ cmsfiles/contents/0000005/5424/000334012.pdf, accessed: 2026-06-22

57. Orengo, H.A., Berganzo-Besga, I., Lumbreras, F.: Theory and practice of artificial intelligence in archaeology. Journal of Archaeological Science 190, 106571 (2026). https://doi.org/10.1016/j.jas.2026.106571

58. Orton, C., Hughes, M.: Pottery in Archaeology. Cambridge Manuals in Archaeology, Cambridge University Press, Cambridge, 2nd edn. (2013). https://doi.org/ 10.1017/CBO9780511920066

59. Pendić, J., Molloy, B.: The Use of 3D Documentation for Investigating Archaeological Artefacts. In: Hostettler, M., Buhlke, A., Drummer, C., Emmenegger, L., Reich, J., Stäheli, C. (eds.) The 3 Dimensions of Digitalised Archaeology: State-of-the-Art, Data Management and Current Challenges in Archaeological 3D-Documentation, pp. 9–26. Springer International Publishing, Cham (2024). https://doi.org/10.1007/978-3-031-53032-6\_2

60. Poznanski, J., Borchardt, J., Dunkelberger, J., Huf, R., Lin, D., Rangapur, A., Wilhelm, C., Lo, K., Soldaini, L.: olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models (2025). https://doi.org/10.48550/arXiv.2502. 18443

61. Poznanski, J., Soldaini, L., Lo, K.: olmOCR 2: Unit Test Rewards for Document OCR (2025). https://doi.org/10.48550/arXiv.2510.19817

62. Puigcerver, J., Mocholí, C.: PyLaia. https://github.com/jpuigcerver/PyLaia (2018), accessed: 2026-08-13

63. Rombach, A.M., Fettke, P.: Deep Learning based Key Information Extraction from Business Documents: Systematic Literature Review. ACM Computing Surveys 58(2), 1–37 (2025). https://doi.org/10.1145/3749369

64. Romein, C.A., Rabus, A., Leifert, G., Ströbel, P.: Assessing advanced handwritten text recognition engines for digitizing historical documents. International Journal of Digital Humanities 7(1), 115–134 (2025). https://doi.org/10.1007/s42803- 025-00100-0

65. Romero, V., Fornés, A., Serrano, N., Sánchez, J.A., Toselli, A.H., Frinken, V., Vidal, E., Lladós, J.: The ESPOSALLES database: An ancient marriage license corpus for of-line handwriting recognition. Pattern Recognition 46(6), 1658–1669 (2013). https://doi.org/10.1016/j.patcog.2012.11.024

66. Sánchez, J.A., Romero, V., Toselli, A.H., Vidal, E.: ICFHR2016 competition on handwritten text recognition on the READ dataset. In: 2016 15th International Conference on Frontiers in Handwriting Recognition (ICFHR). pp. 630–635 (2016). https://doi.org/10.1109/ICFHR.2016.0120

67. Šimsa, Š., Šulc, M., Uřičář, M., Patel, Y., Hamdi, A., Kocián, M., Skalick\`y, M., Matas, J., Doucet, A., Coustaty, M., Karatzas, D.: DocILE Benchmark for Document Information Localization and Extraction. In: Document Analysis and Recognition – ICDAR 2023. pp. 147–166. Springer Nature Switzerland (2023). https://doi.org/10.1007/978-3-031-41679-8\_9

68. Ströbel, P.B., Clematide, S., Volk, M.: How Much Data Do You Need? About the Creation of a Ground Truth for Black Letter and the Efectiveness of Neural OCR. In: Proceedings of the Twelfth Language Resources and Evaluation Conference. pp. 3551–3559 (2020). https://doi.org/10.5167/uzh-197209

69. Taghadouini, S., Cavaillès, A., Aubertin, B.: LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR (2026). https: //doi.org/10.48550/arXiv.2601.14251

70. Tang, Y., Tuncel, D., Koerner, C., Runkler, T.: The Few-shot Dilemma: Overprompting Large Language Models. arXiv preprint arXiv:2509.13196 (2025). https://doi.org/10.48550/arXiv.2509.13196

71. Tarride, S., Boillet, M., Mouflet, J.F., Kermorvant, C.: SIMARA: A Database for Key-Value Information Extraction from Full-Page Handwritten Documents. In: International Conference on Document Analysis and Recognition. pp. 421–437. Springer Nature Switzerland, Cham (2023). https://doi.org/10.1007/978-3- 031-41682-8\_26

72. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., Lin, J.: Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191 (2024). https: //doi.org/10.48550/arXiv.2409.12191

73. Wiegmann, M.: Zeichenrichtlinien des Landesamtes für Denkmalpflege und Archäologie Sachsen-Anhalt (LDA) (2018), https://www.lda-lsa.de/fileadmin/ landesmuseum/alle/pdf /pdf\_redaktion/lda\_zeichenrichtlinien.pdf, accessed: 2026-06-22

74. Wolf, F., Tüselmann, O., Matei, A., Hennies, L., Rass, C., Fink, G.A.: CM1 – A Dataset for Evaluating Few-Shot Information Extraction with Large Vision Language Models. In: Document Analysis and Recognition – ICDAR 2025. pp. 23–39. Springer Nature Switzerland, Cham (2026). https://doi.org/10.1007/978-3- 032-04617-8\_2

75. Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C., Yuan, L.: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4818–4829 (2024). https://doi.org/10.1109/CVPR52733. 2024.00461

76. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025). https://doi.org/10. 48550/arXiv.2505.09388

77. Zvietcovich, F., Castaneda, B., Castillo, L.J., Saldana, J.: A 3D Assessment Tool for Precise Recording of Ceramic Fragments Using Image Processing and Computational Geometry Tools. In: Traviglia, A. (ed.) Across Space and Time: Papers from the 41st Conference on Computer Applications and Quantitative Methods in Archaeology. Routledge (2015). https://doi.org/10.5117/9789089647153-46